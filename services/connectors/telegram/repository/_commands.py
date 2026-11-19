"""Transactional Telegram command acceptance and queue registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from google.cloud import firestore

from core.operations.registry import OPERATIONS, validate_arguments
from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record_in_transaction,
    prepare_outbox_record_in_transaction,
    write_prepared_outbox_record,
)
from core.integrations.firestore.operation_queue import (
    register_operation_in_transaction,
    validate_context_queue_lease,
)
from services.connectors.telegram.models import TelegramDraftState, build_context_id
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    SESSIONS,
    TelegramIntakeResult,
    _document_ref,
    _identifier,
    _integer,
    _receipt_result,
    _aware,
    _context_defaults,
    _read_routing_checkpoint_value,
    _routing_json_bytes,
    _routing_reference_data,
    _routing_checkpoint_path,
)

@dataclass(slots=True)
class _CommandIdentity:
    context_id: str
    receipt_ref: Any
    context_ref: Any
    command_ref: Any
    operation_id: str
    outbox_id: str
    input_ref: str
    identity_digest: str


@dataclass(slots=True)
class _CommandSnapshot:
    draft_id: str | None
    draft_data: dict[str, Any]
    message_ids: tuple[int, ...]
    cutoff_sequence: int
    predecessor_operation_id: str | None
    pending_save_candidate_operation_id: str | None


@dataclass(slots=True)
class _CommandAcceptanceState:
    context: dict[str, Any]
    snapshot: _CommandSnapshot
    command_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _QuestionPromptIdentity:
    notification_id: str
    revision: int
    digest: str


@dataclass(frozen=True, slots=True)
class _QuestionObservationScope:
    context_id: str
    user_id: int
    clarification_id: str
    generation: int
    chat_id: int
    thread_id: int | None

    @property
    def question_ref(self) -> str:
        return (
            f"{SESSIONS}/{self.context_id}/clarifications/"
            f"{self.clarification_id}/questions/{self.generation}"
        )


@dataclass(frozen=True, slots=True)
class _QuestionDeliveryTiming:
    delivered_at: datetime
    observed_at: datetime
    binding_kind: str


_UNSUPPORTED_ROUTE_METADATA_FIELDS = frozenset({
    "schema_version", "actor_id", "context_conversation_id", "context_ref",
    "context_digest", "decision_ref", "decision_digest", "intent_ref",
    "intent_digest", "result_category",
})


def _reject_unsupported_route_metadata(route: Mapping[str, Any]) -> None:
    """Reject the withdrawn metadata-only route wrapper without changing legacy pins."""
    if route.get("routing_protocol") not in (None,):
        raise ValueError("route protocol is unsupported")
    if any(route.get(field) is not None for field in _UNSUPPORTED_ROUTE_METADATA_FIELDS):
        raise ValueError("metadata-only route fields are unsupported")


def _prepare_command_identity(
    client: Any,
    *,
    context_id: str,
    update_id: int,
    command: Any,
) -> _CommandIdentity:
    """Build stable command, operation, and outbox identities before the transaction."""
    update_path = f"{SESSIONS}/{context_id}/updates/{update_id}"
    command_path = f"{SESSIONS}/{context_id}/commands/{update_id}"
    identity = hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
    operation_id = "tgop-" + identity
    return _CommandIdentity(
        context_id=context_id,
        receipt_ref=_document_ref(client, update_path),
        context_ref=client.collection(SESSIONS).document(context_id),
        command_ref=_document_ref(client, command_path),
        operation_id=operation_id,
        outbox_id="tgcommand-" + identity,
        input_ref=command_path,
        identity_digest=hashlib.sha256(
            json.dumps(
                {"kind": command.kind, "text": command.text,
                 "action_metadata": command.action_metadata},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _receipt_save_control_matches(
    *, context_exists: bool, context: Mapping[str, Any], control: Mapping[str, Any],
    row: Mapping[str, Any], draft: Mapping[str, Any], context_id: str,
    control_token: str, notification_id: Any, user_id: int, message_id: int,
) -> bool:
    """Check persisted actor, card, revision, draft, and queue bindings."""
    return bool(
        context_exists and control.get("state") == "active"
        and control.get("token") == control_token
        and control.get("context_id") == context_id
        and control.get("owner_user_id") == user_id
        and control.get("notification_id") == notification_id
        and row.get("receipt_card") is True
        and row.get("receipt_control_token") == control_token
        and row.get("receipt_control_owner_user_id") == user_id
        and row.get("revision") == control.get("issued_revision")
        and row.get("message_id") == message_id
        and row.get("chat_id") == context.get("chat_id")
        and row.get("thread_id") == context.get("thread_id")
        and row.get("context_id") == context_id
        and row.get("draft_id") == draft.get("draft_id")
        and context.get("latest_receipt_notification_id") == notification_id
        and context.get("active_draft_id") == draft.get("draft_id")
        and context.get("draft_status") == TelegramDraftState.COLLECTING.value
        and not context.get("reserved_operation_id")
        and not context.get("active_operation_id")
        and not context.get("pending_save_operation_id")
        and context.get("pending_receipt_groups", 0) == 0
        and draft.get("state") == TelegramDraftState.COLLECTING.value
        and draft.get("dialog_number") == row.get("dialog_number")
        and bool(draft.get("ordered_message_ids"))
    )


def _load_command_snapshot(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    context: Mapping[str, Any],
) -> _CommandSnapshot:
    """Read the collecting draft and immutable save cutoff before transaction writes."""
    draft_id = context.get("active_draft_id")
    draft_ref = (
        _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        if draft_id else None
    )
    draft_snapshot = draft_ref.get(transaction=transaction) if draft_ref is not None else None
    draft_data = (
        (draft_snapshot.to_dict() or {})
        if draft_snapshot is not None and draft_snapshot.exists
        else {}
    )
    if draft_data.get("state") != TelegramDraftState.COLLECTING.value:
        draft_id, draft_ref, draft_data = None, None, {}

    message_ids = tuple(int(value) for value in draft_data.get("ordered_message_ids", ()))
    cutoff_sequence = int(context.get("next_intake_sequence", 1)) - 1
    predecessor_id = context.get("queue_tail_operation_id")
    predecessor_operation_id = predecessor_id if isinstance(predecessor_id, str) and predecessor_id else None
    pending_save_candidate_operation_id = None
    pending_save_id = context.get("pending_save_operation_id")
    if isinstance(pending_save_id, str) and pending_save_id:
        pending_save_ref = _document_ref(
            client, f"{SESSIONS}/{context_id}/operations/{pending_save_id}"
        )
        pending_save_snapshot = pending_save_ref.get(transaction=transaction)
        pending_save_data = pending_save_snapshot.to_dict() or {} if pending_save_snapshot.exists else {}
        if pending_save_data.get("state") in {"pending", "running"}:
            pending_save_candidate_operation_id = pending_save_id
    return _CommandSnapshot(
        draft_id=draft_id,
        draft_data=draft_data,
        message_ids=message_ids,
        cutoff_sequence=cutoff_sequence,
        predecessor_operation_id=predecessor_operation_id,
        pending_save_candidate_operation_id=pending_save_candidate_operation_id,
    )


def _command_link_metadata(snapshot: _CommandSnapshot) -> dict[str, str]:
    """Return immutable predecessor and pending-save candidate metadata."""
    metadata = {}
    if snapshot.predecessor_operation_id is not None:
        metadata["predecessor_operation_id"] = snapshot.predecessor_operation_id
    if snapshot.pending_save_candidate_operation_id is not None:
        metadata["pending_save_candidate_operation_id"] = (
            snapshot.pending_save_candidate_operation_id
        )
    return metadata


def _command_data(
    *,
    update_id: int,
    context_id: str,
    user_id: int,
    operation_id: str,
    command: Any,
    snapshot: _CommandSnapshot,
    now: datetime,
    message_id: int | None,
    message_date: datetime | None,
    reply_to_message_id: int | None,
    observed_clarification_id: str | None = None,
    observed_clarification_generation: int | None = None,
    observed_presentation_id: str | None = None,
    observed_question_delivery_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the command candidate record without copying draft content."""
    return {
        "update_id": update_id,
        "context_id": context_id,
        "user_id": user_id,
        "kind": command.kind,
        "candidate": command.text,
        "message_id": message_id,
        "message_date": message_date,
        "reply_to_message_id": reply_to_message_id,
        "observed_clarification_id": observed_clarification_id,
        "observed_clarification_generation": observed_clarification_generation,
        "observed_presentation_id": observed_presentation_id,
        "observed_question_delivery_ref": (
            dict(observed_question_delivery_ref)
            if observed_question_delivery_ref is not None else None
        ),
        **({"action_metadata": dict(command.action_metadata)}
           if isinstance(command.action_metadata, Mapping) else {}),
        "draft_id": snapshot.draft_id,
        "cutoff_message_ids": list(snapshot.message_ids),
        "cutoff_intake_sequence": snapshot.cutoff_sequence,
        "operation_id": operation_id,
        "created_at": now,
        **_command_link_metadata(snapshot),
    }


def _load_command_acceptance_state(
    client: Any,
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    chat_id: int,
    user_id: int,
    thread_id: int | None,
    topic_name: str | None,
    command: Any,
    now: datetime,
    message_id: int | None,
    message_date: datetime | None,
    reply_to_message_id: int | None,
) -> _CommandAcceptanceState:
    """Load the context and prepare command data before reserving or writing."""
    context_snapshot = identity.context_ref.get(transaction=transaction)
    context = context_snapshot.to_dict() or {}
    for field_name, value in _context_defaults(chat_id, thread_id, topic_name, now).items():
        context.setdefault(field_name, value)
    snapshot = _load_command_snapshot(
        client,
        transaction,
        context_id=identity.context_id,
        context=context,
    )
    return _CommandAcceptanceState(
        context=context,
        snapshot=snapshot,
        command_data=_command_data(
            update_id=update_id,
            context_id=identity.context_id,
            user_id=user_id,
            operation_id=identity.operation_id,
            command=command,
            snapshot=snapshot,
            now=now,
            message_id=message_id,
            message_date=message_date,
            reply_to_message_id=reply_to_message_id,
        ),
    )


def _recorded_question_observation(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    context: Mapping[str, Any],
    user_id: int,
    chat_id: int,
    thread_id: int | None,
    message_date: datetime | None,
    reply_to_message_id: int | None,
    now: datetime,
) -> tuple[str | None, int | None, dict[str, Any] | None]:
    """Capture only a fully scoped current question and its settled delivery proof."""
    clarification_id = context.get("pending_clarification_id")
    generation = context.get("clarification_generation")
    if not _valid_pending_pointer(clarification_id, generation):
        return None, None, None
    scope = _QuestionObservationScope(
        context_id=context_id,
        user_id=user_id,
        clarification_id=clarification_id,
        generation=generation,
        chat_id=chat_id,
        thread_id=thread_id,
    )

    clarification_ref = (
        client.collection(SESSIONS).document(context_id).collection("clarifications")
        .document(clarification_id)
    )
    clarification_snapshot = clarification_ref.get(transaction=transaction)
    clarification = (clarification_snapshot.to_dict() or {}
                     if clarification_snapshot.exists else {})
    parent_created_utc = _validated_clarification_parent_time(
        clarification, scope, now,
    )
    if parent_created_utc is None:
        return None, None, None

    # An explicit Telegram reply cannot silently degrade into a plain answer.
    hint_id, hint_generation = clarification_id, generation
    if reply_to_message_id is not None:
        hint_id = hint_generation = None

    question_document = clarification_ref.collection("questions").document(str(generation))
    question_snapshot = question_document.get(transaction=transaction)
    question = question_snapshot.to_dict() or {} if question_snapshot.exists else {}
    if not question:
        return hint_id, hint_generation, None
    prompt = _question_prompt_identity(question, scope)
    if prompt is None:
        return hint_id, hint_generation, None

    notification_ref = client.collection(NOTIFICATIONS).document(prompt.notification_id)
    notification_snapshot = notification_ref.get(transaction=transaction)
    notification = notification_snapshot.to_dict() or {} if notification_snapshot.exists else {}
    if not _notification_matches_question(
        notification, prompt, question, scope,
    ):
        return hint_id, hint_generation, None

    proof = _question_delivery_proof(
        parent_created_utc,
        question,
        notification,
        prompt,
        scope,
        now=now,
        message_date=message_date,
        reply_to_message_id=reply_to_message_id,
    )
    if proof is not None and reply_to_message_id is not None:
        hint_id, hint_generation = clarification_id, generation
    return hint_id, hint_generation, proof


def _valid_pending_pointer(clarification_id: Any, generation: Any) -> bool:
    if (not isinstance(clarification_id, str) or type(generation) is not int
            or generation < 1):
        return False
    try:
        _identifier(clarification_id, "pending_clarification_id")
    except ValueError:
        return False
    return True


def _validated_clarification_parent_time(
    clarification: Mapping[str, Any],
    scope: _QuestionObservationScope,
    now: datetime,
) -> datetime | None:
    if not (
        clarification.get("clarification_id") == scope.clarification_id
        and clarification.get("context_id") == scope.context_id
        and clarification.get("protocol") == "clarification-v1"
        and type(clarification.get("actor_id")) is int
        and clarification.get("actor_id") == scope.user_id
        and clarification.get("state") == "waiting"
        and type(clarification.get("generation")) is int
        and clarification.get("generation") == scope.generation
    ):
        return None
    created_at = clarification.get("created_at")
    if not _is_aware_datetime(created_at):
        return None
    try:
        created_utc = created_at.astimezone(timezone.utc)
        intake_now_utc = now.astimezone(timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return created_utc if created_utc <= intake_now_utc else None


def _question_prompt_identity(
    question: Mapping[str, Any],
    scope: _QuestionObservationScope,
) -> _QuestionPromptIdentity | None:
    notification_id = question.get("prompt_notification_id")
    revision = question.get("prompt_notification_revision")
    digest = question.get("prompt_digest")
    if (not isinstance(notification_id, str) or not notification_id
            or type(revision) is not int or revision < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or question.get("clarification_id") != scope.clarification_id
            or question.get("context_id") != scope.context_id
            or type(question.get("actor_id")) is not int
            or question.get("actor_id") != scope.user_id
            or type(question.get("generation")) is not int
            or question.get("generation") != scope.generation):
        return None
    try:
        _identifier(notification_id, "prompt_notification_id")
    except ValueError:
        return None
    return _QuestionPromptIdentity(notification_id, revision, digest)


def _notification_matches_question(
    notification: Mapping[str, Any],
    prompt: _QuestionPromptIdentity,
    question: Mapping[str, Any],
    scope: _QuestionObservationScope,
) -> bool:
    body = notification.get("body")
    if not isinstance(body, str):
        return False
    try:
        body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return False
    return bool(
        notification.get("notification_id") == prompt.notification_id
        and notification.get("kind") == "clarification"
        and notification.get("context_id") == scope.context_id
        and type(notification.get("actor_id")) is int
        and notification.get("actor_id") == scope.user_id
        and notification.get("clarification_id") == scope.clarification_id
        and type(notification.get("generation")) is int
        and notification.get("generation") == scope.generation
        and type(notification.get("chat_id")) is int
        and notification.get("chat_id") == scope.chat_id
        and ((scope.thread_id is None and notification.get("thread_id") is None)
             or (type(scope.thread_id) is int and type(notification.get("thread_id")) is int
                 and notification.get("thread_id") == scope.thread_id))
        and type(notification.get("revision")) is int
        and notification.get("revision") == prompt.revision
        and type(notification.get("message_id")) is int
        and notification.get("message_id") > 0
        and notification.get("state") == "delivered"
        and prompt.digest == body_digest
        and _is_aware_datetime(question.get("created_at"))
        and _is_aware_datetime(notification.get("delivered_at"))
    )


def _question_delivery_timing(
    parent_created_utc: datetime,
    question_created_at: Any,
    delivered_at: Any,
    *,
    now: datetime,
    message_date: datetime | None,
    reply_to_message_id: int | None,
    prompt_message_id: Any,
) -> _QuestionDeliveryTiming | None:
    try:
        created_utc, delivered_utc, now_utc = (
            question_created_at.astimezone(timezone.utc),
            delivered_at.astimezone(timezone.utc),
            now.astimezone(timezone.utc),
        )
    except (OverflowError, OSError, ValueError):
        return None
    if not (parent_created_utc <= created_utc <= delivered_utc <= now_utc):
        return None
    try:
        message_date_utc = (message_date.astimezone(timezone.utc)
                            if _is_aware_datetime(message_date) else None)
    except (OverflowError, OSError, ValueError):
        message_date_utc = None
    if message_date_utc is not None and message_date_utc > now_utc:
        return None
    if reply_to_message_id is not None:
        if (prompt_message_id != reply_to_message_id
                or (message_date_utc is not None and created_utc > message_date_utc)):
            return None
        binding_kind = "explicit_reply"
    else:
        if (message_date_utc is None or delivered_utc >= message_date_utc
                or created_utc > message_date_utc):
            return None
        binding_kind = "plain"
    return _QuestionDeliveryTiming(
        delivered_at=delivered_utc,
        observed_at=now_utc,
        binding_kind=binding_kind,
    )


def _question_delivery_proof(
    parent_created_utc: datetime,
    question: Mapping[str, Any],
    notification: Mapping[str, Any],
    prompt: _QuestionPromptIdentity,
    scope: _QuestionObservationScope,
    *,
    now: datetime,
    message_date: datetime | None,
    reply_to_message_id: int | None,
) -> dict[str, Any] | None:
    timing = _question_delivery_timing(
        parent_created_utc,
        question.get("created_at"),
        notification.get("delivered_at"),
        now=now,
        message_date=message_date,
        reply_to_message_id=reply_to_message_id,
        prompt_message_id=notification.get("message_id"),
    )
    if timing is None:
        return None
    return {
        "context_id": scope.context_id,
        "actor_id": scope.user_id,
        "clarification_id": scope.clarification_id,
        "generation": scope.generation,
        "question_ref": scope.question_ref,
        "notification_id": prompt.notification_id,
        "notification_revision": prompt.revision,
        "message_id": notification["message_id"],
        "prompt_digest": prompt.digest,
        "delivered_at": timing.delivered_at,
        "observed_at": timing.observed_at,
        "binding_kind": timing.binding_kind,
    }


def _is_aware_datetime(value: Any) -> bool:
    return (isinstance(value, datetime) and value.tzinfo is not None
            and value.utcoffset() is not None)


def _trusted_transport_date(timestamp: int | None) -> datetime | None:
    if type(timestamp) is not int or timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _register_command_operation(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    identity: _CommandIdentity,
    now: datetime,
    snapshot: _CommandSnapshot,
    action_metadata: Mapping[str, Any] | None = None,
    ingress_binding_id: str | None = None,
) -> Any:
    """Reserve the FIFO operation and create its outbox before queue writes begin."""
    def create_command_outbox() -> None:
        create_outbox_record_in_transaction(
            client,
            transaction,
            identity.outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1,
            schedule_at=now,
            now=now,
            operation_id=identity.operation_id,
            context_id=identity.context_id,
        )

    operation_fields = _command_link_metadata(snapshot)
    if ingress_binding_id is not None:
        operation_fields["ingress_binding_id"] = ingress_binding_id
    if action_metadata is not None:
        operation_fields["typed_action"] = dict(action_metadata)
    return register_operation_in_transaction(
        client,
        transaction,
        context_ref,
        identity.operation_id,
        identity_digest=identity.identity_digest,
        input_ref=identity.input_ref,
        now=now,
        before_write=create_command_outbox,
        operation_fields=operation_fields,
    )


def _update_command_context(
    context: dict[str, Any],
    *,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    operation_id: str,
    sequence: int,
    now: datetime,
) -> None:
    """Record the new FIFO tail and keep its current head reserved."""
    context.update({
        "chat_id": chat_id,
        "thread_id": thread_id,
        "next_operation_sequence": sequence + 1,
        "queue_tail_operation_id": operation_id,
        "updated_at": now,
    })
    if not context.get("reserved_operation_id"):
        context.update({"reserved_operation_id": operation_id, "head_sequence": sequence})
    if topic_name is not None:
        context["topic_name"] = topic_name


def _persist_command_acceptance(
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    state: _CommandAcceptanceState,
    operation: Any,
    now: datetime,
    ingress_binding: Mapping[str, Any] | None = None,
) -> "TelegramIntakeResult":
    """Persist command, context, and replay receipt after all transaction reads."""
    snapshot = state.snapshot
    receipt = {
        "update_id": update_id,
        "context_id": identity.context_id,
        "accepted": True,
        "draft_id": snapshot.draft_id,
        "dialog_number": snapshot.draft_data.get("dialog_number"),
        "message_id": None,
        "intake_sequence": None,
        "notification_id": None,
        "outbox_id": identity.outbox_id,
        "operation_id": identity.operation_id,
        "queue_position": operation.sequence,
        "snapshot_message_ids": list(snapshot.message_ids),
        "created_at": now,
    }
    transaction.set(identity.command_ref, state.command_data)
    transaction.set(identity.context_ref, state.context)
    transaction.set(identity.receipt_ref, receipt)
    if ingress_binding is not None:
        binding_ref = identity.context_ref.collection("ingress_bindings").document(
            identity.operation_id
        )
        transaction.set(binding_ref, dict(ingress_binding))
    return TelegramIntakeResult(
        True,
        identity.context_id,
        snapshot.draft_id,
        snapshot.draft_data.get("dialog_number"),
        outbox_id=identity.outbox_id,
        operation_id=identity.operation_id,
        queue_position=operation.sequence,
        snapshot_message_ids=snapshot.message_ids,
    )


def _bind_text_to_recent_receipt_group(
    client: Any, transaction: Any, *, context_id: str, context: Mapping[str, Any],
    draft_id: str | None, user_id: int, message_id: int | None,
    transport_timestamp: int | None,
) -> dict[str, Any] | None:
    """Return an exact forward-first burst binding when transport evidence agrees."""
    group_id = context.get("latest_receipt_group_id")
    if (not isinstance(group_id, str) or not isinstance(draft_id, str)
            or message_id is None or type(transport_timestamp) is not int):
        return None
    group_ref = client.collection(SESSIONS).document(context_id).collection(
        "receipt_groups"
    ).document(group_id)
    snapshot = group_ref.get(transaction=transaction)
    group = snapshot.to_dict() or {} if snapshot.exists else {}
    start, end = group.get("transport_start_timestamp"), group.get("transport_end_timestamp")
    max_message_id = group.get("max_message_id")
    source_ids = group.get("source_message_ids")
    if not isinstance(source_ids, list):
        source_ids = []
    source_ids = [value for value in source_ids if type(value) is int]
    if type(max_message_id) is not int:
        max_message_id = max(source_ids, default=0)
    distance = (start - transport_timestamp if type(start) is int and transport_timestamp < start
                else transport_timestamp - end if type(end) is int and transport_timestamp > end
                else 0)
    if not (
        group.get("context_id") in (None, context_id)
        and group.get("draft_id") == draft_id
        and group.get("owner_user_id") in (None, 0, user_id)
        and group.get("state") in {"open", "closed"}
        and type(start) is int and type(end) is int and distance <= 1
        and max_message_id < message_id
    ):
        return None
    return {
        "draft_id": draft_id,
        "receipt_group_id": group_id,
        "matched_message_ids": source_ids,
        "state": "collecting",
    }


class _TelegramCommandMixin:
    """Persist idempotent command acceptance and operation registration."""

    @staticmethod
    def _validate_joint_contextual_shape(
        application_route: Any, *, context_id: str, operation_id: str,
        checkpoint_conversation_id: str,
    ) -> tuple[Mapping[str, Any], bool]:
        """Share the contextual DTO and tool-shape checks across prepare/read paths."""
        if not isinstance(application_route, Mapping) or "contextual" not in application_route:
            raise ValueError("contextual application route is absent")
        _routing_json_bytes(application_route)
        executable_fields = {"name", "arguments", "ordinal", "target_doc_id"}
        present = executable_fields.intersection(application_route)
        executable = present == executable_fields
        expected_top = {"contextual"}
        if executable:
            expected_top |= executable_fields
            if "call_id" in application_route:
                expected_top.add("call_id")
        if set(application_route) != expected_top:
            raise ValueError("contextual application route shape is invalid")
        contextual = application_route.get("contextual")
        required = {
            "schema_version", "context_ref", "context_digest", "context_id",
            "operation_id", "actor_id", "accepted_command", "accepted_command_ref",
            "full_history_ref", "source_conversation_id", "checkpoint_conversation_id",
            "decision", "resolved", "summary", "fixed_plan",
        }
        if (not isinstance(contextual, Mapping)
                or set(contextual) not in (required, required | {"intent_ref", "intent_digest"})
                or ("intent_ref" in contextual) != ("intent_digest" in contextual)
                or type(contextual.get("schema_version")) is not int
                or contextual.get("schema_version") != 1
                or contextual.get("context_id") != context_id
                or contextual.get("operation_id") != operation_id
                or type(contextual.get("actor_id")) is not int
                or contextual.get("actor_id") <= 0
                or contextual.get("checkpoint_conversation_id") != checkpoint_conversation_id
                or not isinstance(contextual.get("source_conversation_id"), str)
                or not contextual.get("source_conversation_id")
                or not isinstance(contextual.get("accepted_command"), str)):
            raise ValueError("contextual application route binding is invalid")
        return contextual, executable

    @staticmethod
    def _validate_joint_intent_binding(
        client: Any, contextual: Mapping[str, Any], snapshot: Any,
        context_reference: Mapping[str, Any], *, transaction: Any = None,
    ) -> None:
        """Read and verify a complete T011 clarification request for both joint paths."""
        from services.connectors.telegram.runtime.routing_context import RoutingDecision

        decision = RoutingDecision.from_dict(contextual.get("decision"))
        has_intent = "intent_ref" in contextual
        if (decision.disposition.value == "clarify") != has_intent:
            raise ValueError("clarification route requires its complete intent binding")
        if not has_intent:
            return
        context_id = snapshot.context_id
        operation_id = snapshot.operation_id
        intent_path = _routing_checkpoint_path(context_id, operation_id, "intent")
        if contextual.get("intent_ref") != intent_path:
            raise ValueError("contextual intent path is invalid")
        intent_header = _document_ref(client, intent_path).get(transaction=transaction)
        header = intent_header.to_dict() or {} if intent_header.exists else {}
        intent_reference = _routing_reference_data({key: header.get(key) for key in (
            "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
            "actor_id", "conversation_id", "ref", "digest", "byte_length",
        )})
        if (intent_reference["kind"] != "intent"
                or intent_reference["ref"] != intent_path
                or intent_reference["context_id"] != context_id
                or intent_reference["operation_id"] != operation_id
                or type(intent_reference["actor_id"]) is not int
                or intent_reference["actor_id"] != snapshot.actor_id
                or intent_reference["conversation_id"] != snapshot.selected_conversation_id
                or intent_reference["digest"] != contextual.get("intent_digest")):
            raise ValueError("contextual intent header scope changed")
        request = _read_routing_checkpoint_value(
            client, intent_reference, transaction=transaction,
        )
        if (not isinstance(request, Mapping)
                or set(request) != {
                    "protocol", "clarification", "context_reference", "question", "prompt",
                }
                or _routing_json_bytes(request.get("context_reference"))
                != _routing_json_bytes(context_reference)):
            raise ValueError("contextual clarification request is malformed")
        from services.connectors.telegram.repository._clarifications import (
            _decode_clarification_checkpoint,
        )
        _decoded_context_ref, pending, _decoded_snapshot, _prompt = (
            _decode_clarification_checkpoint(
                intent_reference, request, snapshot.to_dict(),
            )
        )
        resolved = contextual.get("resolved")
        if (pending.action != decision.action.value
                or tuple(pending.missing_slots) != tuple(decision.missing_slots)
                or not isinstance(resolved, Mapping)
                or _routing_json_bytes(
                    pending.to_dict()["resolved_parameters"]
                )
                != _routing_json_bytes(resolved)):
            raise ValueError("contextual intent does not match its decision")

    @staticmethod
    def _validate_joint_routing_snapshot(
        snapshot: Any, *, client: Any, context_id: str, operation_id: str,
        actor_id: int, input_ref: str, operation: Mapping[str, Any],
        command: Mapping[str, Any], conversation_id: str, transaction: Any = None,
    ) -> None:
        """Validate legacy minimal refs or the exact frozen T017 capture projection."""
        accepted = snapshot.accepted_command_ref
        history = snapshot.full_history_ref
        if (accepted.ref != input_ref
                or accepted.context_id != context_id or type(accepted.actor_id) is not int
                or accepted.actor_id != actor_id
                or accepted.conversation_id != conversation_id
                or accepted.operation_id != operation_id
                or history.ref != f"{SESSIONS}/{context_id}/conversations/{conversation_id}/turns"
                or history.context_id != context_id or type(history.actor_id) is not int
                or history.actor_id != actor_id or history.conversation_id != conversation_id
                or history.operation_id != operation_id):
            raise ValueError("contextual bound-reference scope is invalid")
        minimal_legacy = (
            accepted.payload == {"text": snapshot.accepted_command}
            and isinstance(history.payload, Mapping)
            and set(history.payload) == {"turns"}
        )
        if minimal_legacy:
            if (accepted.digest != hashlib.sha256(
                    snapshot.accepted_command.encode("utf-8")).hexdigest()
                    or not isinstance(history.payload["turns"], (list, tuple))
                    or history.digest != hashlib.sha256(_routing_json_bytes(
                        history.to_dict()["payload"]["turns"])).hexdigest()):
                raise ValueError("minimal contextual bound-reference digest is invalid")
            return
        from services.connectors.telegram.repository import TelegramRepository
        from services.connectors.telegram.runtime.store import TelegramRuntimeStore

        validator = TelegramRuntimeStore(
            TelegramRepository(client=client), context_id=context_id,
            operation_id=operation_id,
        )
        validator._validate_captured_snapshot(
            snapshot, actor_id=actor_id, input_ref=input_ref, operation=operation,
            command=command, conversation_id=conversation_id,
            allow_finalized_ingress=transaction is not None,
            transaction=transaction, frozen_identity_only=transaction is None,
        )

    def _load_joint_payload(
        self, stored: Any, kind: str, context_id: str, operation_id: str, transaction: Any,
    ) -> Any:
        from core.integrations.firestore.input_parts import MAX_PART_BYTES, read_input_parts
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

        if not isinstance(stored, Mapping):
            raise ValueError("prepared application payload is malformed")
        digest, byte_length = stored.get("sha256"), stored.get("byte_length")
        if (not isinstance(digest, str) or len(digest) != 64
                or type(byte_length) is not int or byte_length < 0):
            raise ValueError("prepared application payload metadata is invalid")
        if set(stored) == {"_inline", "sha256", "byte_length"}:
            value = stored["_inline"]
            payload = FirestoreAgentLedger._json_bytes(value)
            if len(payload) > 4096:
                raise ValueError("prepared application inline payload is oversized")
        elif set(stored) == {"_input_ref", "sha256", "byte_length", "max_part_bytes"}:
            data_operation = "telegram-ledger-" + hashlib.sha256(
                f"{context_id}\0{operation_id}\0{kind}\0{digest}".encode()
            ).hexdigest()
            expected_ref = f"dialog_operations/{data_operation}/input_parts/manifest"
            if (stored.get("_input_ref") != expected_ref
                    or type(stored.get("max_part_bytes")) is not int
                    or stored.get("max_part_bytes") != MAX_PART_BYTES
                    or byte_length <= 4096):
                raise ValueError("prepared application parts are not operation-owned")
            payload = read_input_parts(self.client, expected_ref, transaction=transaction)
            value = FirestoreAgentLedger._json_value(payload)
        else:
            raise ValueError("prepared application payload mode is invalid")
        if len(payload) != byte_length or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("prepared application payload integrity failed")
        return value

    @staticmethod
    def _validate_joint_folder_binding(
        selector: Any, *, target_id: Any, root_id: Any, label: str,
    ) -> None:
        from services.connectors.telegram.runtime.routing_context import FolderSelectorKind

        if selector is None or selector.kind is FolderSelectorKind.ROOT_DEFAULT:
            if target_id != root_id:
                raise ValueError(f"default {label} is not the captured root")
        elif (selector.kind is FolderSelectorKind.EXPLICIT_ID
              and selector.value != target_id):
            raise ValueError(f"explicit {label} conflicts with resolved identity")

    @staticmethod
    def _validate_joint_application_projection(
        application_route: Mapping[str, Any], summary: Mapping[str, Any],
    ) -> None:
        from services.connectors.telegram.sessions import _validate_stored_joint_projection
        from services.connectors.telegram.runtime.routing_context import (
            Action, FolderSelectorKind, RoutingDecision,
        )

        _validate_stored_joint_projection(application_route, summary)
        _TelegramCommandMixin._validate_joint_application_projection_fields(
            application_route, summary,
        )

    @staticmethod
    def _validate_joint_application_projection_fields(
        application_route: Mapping[str, Any], summary: Mapping[str, Any],
    ) -> None:
        from services.connectors.telegram.runtime.routing_context import (
            Action, RoutingDecision,
        )

        contextual = application_route.get("contextual")
        if not isinstance(contextual, Mapping):
            raise ValueError("contextual application decision is absent")
        decision = RoutingDecision.from_dict(contextual.get("decision"))
        choice = summary.get("tool_choice")
        executable_fields = {"name", "arguments", "ordinal", "target_doc_id"}
        present = executable_fields.intersection(application_route)
        if present and present != executable_fields:
            raise ValueError("contextual executable application route is partial")
        fixed_plan = contextual.get("fixed_plan")
        resolved = contextual.get("resolved")
        if not isinstance(resolved, Mapping):
            raise ValueError("resolved contextual parameters are malformed")
        if not present:
            if fixed_plan is not None or choice is not None:
                raise ValueError("metadata-only route contains an executable plan")
            return
        if (not isinstance(choice, Mapping)
                and decision.action is not Action.CREATE_THEN_MOVE):
            raise ValueError("application tool choice conflicts with summary")
        if (isinstance(choice, Mapping)
                and (application_route.get("name") != choice.get("name")
                     or application_route.get("arguments") != choice.get("arguments"))):
            raise ValueError("application tool choice conflicts with summary")
        if (type(application_route.get("ordinal")) is not int
                or application_route["ordinal"] < 0):
            raise ValueError("application ordinal is invalid")
        if application_route.get("target_doc_id") != summary.get("target_doc_id"):
            raise ValueError("application target conflicts with summary")
        if application_route.get("name") == "move_transcript":
            resolved_doc = resolved.get("doc_id")
            if (not isinstance(resolved_doc, str) or not resolved_doc
                    or application_route.get("arguments", {}).get("doc_id") != resolved_doc
                    or application_route.get("target_doc_id") != resolved_doc
                    or summary.get("target_doc_id") != resolved_doc):
                raise ValueError("move document target conflicts with resolved identity")
        if application_route.get("name") == "create_folder" and (
            application_route.get("target_doc_id") is not None
            or summary.get("target_doc_id") is not None
        ):
            raise ValueError("folder creation cannot carry a document target")
        action_to_tool = {
            "move_transcript": {"move_transcript"},
            "create_folder": {"create_folder"},
            "create_then_move": {"create_folder", "move_transcript"},
        }
        expected = action_to_tool.get(decision.action.value)
        if expected is not None and application_route["name"] not in expected:
            raise ValueError("application tool conflicts with semantic action")
        is_write = OPERATIONS[application_route["name"]].write
        if fixed_plan is None:
            if is_write:
                raise ValueError("mutating contextual route has no fixed plan")
            return
        _TelegramCommandMixin._validate_joint_fixed_application_plan(
            application_route, contextual, decision, fixed_plan, resolved,
        )

    @staticmethod
    def _validate_joint_fixed_application_plan(
        application_route: Mapping[str, Any], contextual: Mapping[str, Any],
        decision: Any, fixed_plan: Any, resolved: Mapping[str, Any],
    ) -> None:
        from services.connectors.telegram.runtime.routing_context import Action

        if not isinstance(fixed_plan, Mapping) or set(fixed_plan) != {"steps"}:
            raise ValueError("fixed application plan shape is invalid")
        steps = fixed_plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("fixed application plan is incomplete")
        if decision.action is Action.CREATE_THEN_MOVE or len(steps) != 1:
            raise ValueError("composite application plans are not supported")
        expected_plan_names = {
            "create_folder": ["create_folder"],
            "move_transcript": ["move_transcript"],
            "create_then_move": ["create_folder", "move_transcript"],
        }.get(decision.action.value)
        if expected_plan_names is not None and [
            step.get("name") if isinstance(step, Mapping) else None for step in steps
        ] != expected_plan_names:
            raise ValueError("fixed mutation plan shape does not match semantic action")
        if expected_plan_names is None and (
            len(steps) != 1 or not isinstance(steps[0], Mapping)
            or steps[0].get("name") != application_route["name"]
        ):
            raise ValueError("fixed operation plan does not match its application tool")
        parent_operation = contextual.get("operation_id")
        for candidate in steps:
            _TelegramCommandMixin._validate_joint_application_step(
                candidate, parent_operation=parent_operation,
                expected_ordinal=application_route["ordinal"],
                decision=decision, resolved=resolved,
            )
        _TelegramCommandMixin._validate_joint_application_step_projection(
            application_route, steps,
        )

    @staticmethod
    def _validate_joint_application_step(
        candidate: Any, *, parent_operation: Any, expected_ordinal: int,
        decision: Any, resolved: Mapping[str, Any],
    ) -> None:
        if (not isinstance(candidate, Mapping)
                or type(candidate.get("ordinal")) is not int
                or set(candidate) != {
                    "ordinal", "name", "arguments", "target_doc_id",
                    "operation_id", "effect_id", "resolved_targets",
                }
                or candidate.get("ordinal") != expected_ordinal
                or not isinstance(candidate.get("name"), str)
                or not isinstance(candidate.get("arguments"), Mapping)
                or not isinstance(candidate.get("operation_id"), str)
                or not isinstance(candidate.get("effect_id"), str)
                or not isinstance(candidate.get("resolved_targets"), Mapping)):
            raise ValueError("fixed application plan contains an incomplete step")
        try:
            validate_arguments(candidate["name"], candidate["arguments"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fixed application step arguments are invalid") from exc
        child_digest = hashlib.sha256(
            f"{parent_operation}\0{expected_ordinal}".encode("utf-8")
        ).hexdigest()
        expected_operation_id = "telegram-tool-" + child_digest
        if candidate["name"] == "add_dialog":
            from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id
            expected_operation_id = direct_add_dialog_operation_id(
                parent_operation, expected_ordinal, candidate["arguments"],
            )
        effect_digest = hashlib.sha256(
            f"{parent_operation}\0{expected_ordinal}".encode("utf-8")
        ).hexdigest()
        if (candidate["operation_id"] != expected_operation_id
                or candidate["effect_id"] != "telegram-effect-" + effect_digest):
            raise ValueError("fixed application step identity is not deterministic")
        if candidate["name"] not in OPERATIONS:
            raise ValueError("fixed application step tool is unsupported")
        if OPERATIONS[candidate["name"]].write:
            _TelegramCommandMixin._validate_joint_resolved_step_targets(
                candidate, decision=decision, resolved=resolved,
            )

    @staticmethod
    def _validate_joint_resolved_step_targets(
        candidate: Mapping[str, Any], *, decision: Any,
        resolved: Mapping[str, Any],
    ) -> None:
        step_targets = candidate["resolved_targets"]
        if candidate["name"] == "create_folder":
            target_keys = {"parent_id", "root_id", "intent_digest"}
            if (not target_keys.issubset(step_targets)
                    or any(not isinstance(step_targets.get(key), str)
                           or not step_targets[key] for key in target_keys)
                    or step_targets.get("root_id") != resolved.get("root_id")
                    or step_targets.get("parent_id") != resolved.get("parent_id")
                    or step_targets.get("intent_digest") != resolved.get("intent_digest")
                    or len(step_targets.get("intent_digest", "")) != 64
                    or any(ch not in "0123456789abcdef"
                           for ch in step_targets.get("intent_digest", ""))
                    or candidate["arguments"].get("name") != resolved.get("folder_name", candidate["arguments"].get("name"))):
                raise ValueError("folder plan targets conflict with resolved parameters")
            _TelegramCommandMixin._validate_joint_folder_binding(
                decision.folder_selector, target_id=step_targets.get("parent_id"),
                root_id=step_targets.get("root_id"), label="folder parent",
            )
        elif candidate["name"] == "move_transcript":
            target_keys = {"doc_id", "destination_id", "root_id", "intent_digest"}
            if (not target_keys.issubset(step_targets)
                    or any(not isinstance(step_targets.get(key), str)
                           or not step_targets[key] for key in target_keys)
                    or candidate["arguments"].get("doc_id") != step_targets.get("doc_id")
                    or step_targets.get("doc_id") != resolved.get("doc_id")
                    or step_targets.get("destination_id") != resolved.get("destination_id")
                    or step_targets.get("root_id") != resolved.get("root_id")
                    or step_targets.get("intent_digest") != resolved.get("intent_digest")
                    or len(step_targets.get("intent_digest", "")) != 64
                    or any(ch not in "0123456789abcdef"
                           for ch in step_targets.get("intent_digest", ""))):
                raise ValueError("move plan targets conflict with resolved parameters")
            _TelegramCommandMixin._validate_joint_folder_binding(
                decision.folder_selector,
                target_id=step_targets.get("destination_id"),
                root_id=step_targets.get("root_id"), label="move destination",
            )

    @staticmethod
    def _validate_joint_application_step_projection(
        application_route: Mapping[str, Any], steps: list[Any],
    ) -> None:
        matching = [step for step in steps if isinstance(step, Mapping)
                    and step.get("ordinal") == application_route["ordinal"]]
        if len(matching) != 1:
            raise ValueError("fixed plan does not bind application ordinal")
        step = matching[0]
        if (step.get("name") != application_route["name"]
                or step.get("arguments") != application_route["arguments"]
                or step.get("target_doc_id") != application_route.get("target_doc_id")
                or not isinstance(step.get("operation_id"), str)
                or not isinstance(step.get("effect_id"), str)
                or not isinstance(step.get("resolved_targets"), Mapping)):
            raise ValueError("fixed application step conflicts with executable route")

    def _read_joint_source_context(
        self, transaction: Any, checkpoint: Any, *, context_id: str,
        operation_id: str, actor_id: int, command_ref: Any,
        command: Mapping[str, Any], operation: Mapping[str, Any],
        safe_route: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Validate source identity, frozen routing input, intent and summary in-transaction."""
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
        from services.connectors.telegram.runtime.routing_context import (
            RoutingContextSnapshot, RoutingDecision,
        )

        if hashlib.sha256(checkpoint.route_payload).hexdigest() != checkpoint.route_digest:
            raise ValueError("prepared application route digest changed")
        application_route = FirestoreAgentLedger._json_value(checkpoint.route_payload)
        contextual, _executable = self._validate_joint_contextual_shape(
            application_route, context_id=context_id, operation_id=operation_id,
            checkpoint_conversation_id=checkpoint.conversation_id,
        )
        accepted_text = contextual.get("accepted_command")
        if ("candidate" in command and command.get("candidate") != accepted_text):
            raise ValueError("prepared accepted command text changed")
        command_digest = hashlib.sha256(json.dumps({
            "kind": command.get("kind"), "text": accepted_text,
            "action_metadata": command.get("action_metadata"),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (operation.get("identity_digest") != command_digest
                or operation.get("input_ref") != command_ref.path):
            raise ValueError("accepted command identity changed")
        context_path = _routing_checkpoint_path(context_id, operation_id, "context")
        if contextual.get("context_ref") != context_path:
            raise ValueError("prepared source context path is invalid")
        if contextual.get("actor_id") != actor_id:
            raise PermissionError("contextual actor does not match accepted command")
        context_header = _document_ref(self.client, context_path).get(transaction=transaction)
        header = context_header.to_dict() or {} if context_header.exists else {}
        context_reference = _routing_reference_data({key: header.get(key) for key in (
            "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
            "actor_id", "conversation_id", "ref", "digest", "byte_length",
        )})
        if (context_reference.get("kind") != "context"
                or context_reference.get("context_id") != context_id
                or context_reference.get("operation_id") != operation_id
                or context_reference.get("actor_id") != actor_id
                or context_reference.get("conversation_id") != contextual.get("source_conversation_id")
                or context_reference.get("digest") != contextual.get("context_digest")):
            raise ValueError("prepared source context reference is invalid")
        snapshot = RoutingContextSnapshot.from_dict(_read_routing_checkpoint_value(
            self.client, context_reference, transaction=transaction,
        ))
        if (snapshot.context_id != context_id or type(snapshot.actor_id) is not int
                or snapshot.actor_id != actor_id or snapshot.operation_id != operation_id
                or snapshot.accepted_command != accepted_text
                or snapshot.selected_conversation_id != contextual.get("source_conversation_id")
                or _routing_json_bytes(snapshot.accepted_command_ref.to_dict())
                != _routing_json_bytes(contextual.get("accepted_command_ref"))
                or _routing_json_bytes(snapshot.full_history_ref.to_dict())
                != _routing_json_bytes(contextual.get("full_history_ref"))):
            raise ValueError("prepared contextual snapshot binding is invalid")
        self._validate_joint_routing_snapshot(
            snapshot, client=self.client, context_id=context_id,
            operation_id=operation_id, actor_id=actor_id,
            input_ref=command_ref.path, operation=operation, command=command,
            conversation_id=contextual["source_conversation_id"], transaction=transaction,
        )
        decision = RoutingDecision.from_dict(contextual.get("decision"))
        decision.validate_command_spans(accepted_text)
        self._validate_joint_intent_binding(
            self.client, contextual, snapshot, context_reference,
            transaction=transaction,
        )
        summary = contextual.get("summary")
        if (not isinstance(summary, Mapping)
                or self._route_control_data(summary) != dict(safe_route)):
            raise ValueError("prepared contextual summary projection conflicts")
        self._validate_joint_application_projection(application_route, summary)
        return application_route, contextual

    @staticmethod
    def _prepared_lease_is_current(
        context: Mapping[str, Any], operation_id: str, owner: str, fence: int,
        checkpoint: Any,
    ) -> bool:
        late_now = checkpoint._clock()
        return validate_context_queue_lease(
            context, operation_id, owner=owner, fence=fence, now=late_now,
            expiry=context.get("lock_expires_at"),
        )

    def _read_joint_route_scope(
        self, transaction: Any, *, context_ref: Any, operation_ref: Any,
        context_id: str, operation_id: str, owner: str, fence: int, now: datetime,
    ) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]] | None:
        """Read the lease and accepted command scope before planning joint writes."""
        context_snapshot = context_ref.get(transaction=transaction)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not context_snapshot.exists or not operation_snapshot.exists:
            return None
        context = context_snapshot.to_dict() or {}
        operation = operation_snapshot.to_dict() or {}
        existing_route = operation.get("route_control")
        if isinstance(existing_route, Mapping):
            _reject_unsupported_route_metadata(existing_route)
        if (operation.get("operation_id") != operation_id
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, operation_id, owner=owner, fence=fence, now=now,
                    expiry=context.get("lock_expires_at"),
                )):
            return None
        command_ref, command = self._command_for_operation(
            transaction, context_id=context_id,
            operation_id=operation_id, operation=operation,
        )
        return context, operation, command_ref, command

    def _validate_joint_native_state(
        self, turn: Mapping[str, Any], *, prepared_route: Mapping[str, Any],
        prepared_calls: list[Any], checkpoint: Any, context_id: str,
        operation_id: str, transaction: Any,
    ) -> None:
        """Verify immutable route/call prefixes without changing native rounds or results."""
        existing_stored_route = turn.get("application_route")
        existing_app = None
        if existing_stored_route is not None:
            existing_app = self._load_joint_payload(
                existing_stored_route, "application-route", context_id,
                operation_id, transaction,
            )
            if existing_app != prepared_route:
                raise ValueError("persisted application route is immutable")
        existing_calls = []
        if turn.get("tool_calls") is not None:
            existing_calls = self._load_joint_payload(
                turn.get("tool_calls"), "calls", context_id, operation_id, transaction,
            )
            if (existing_app is None
                    and existing_calls != prepared_calls[:len(existing_calls)]) or (
                    existing_app is not None
                    and prepared_calls != existing_calls[:len(prepared_calls)]):
                raise ValueError("persisted agent calls are immutable")
        if ("application_route_round" in turn
                and turn.get("application_route_round") != checkpoint.route_round):
            raise ValueError("persisted application route is immutable")
        from services.connectors.telegram.agent import _contextual_call_provenance_valid
        model_response = (None if turn.get("model_response") is None else
                          self._load_joint_payload(
                              turn.get("model_response"), "model", context_id,
                              operation_id, transaction,
                          ))
        calls_to_validate = (
            existing_calls if existing_app is not None
            and isinstance(existing_app.get("contextual"), Mapping)
            else prepared_calls
        )
        if not _contextual_call_provenance_valid({
            "application_route": prepared_route,
            "application_route_round": checkpoint.route_round,
            "model_response": model_response, "tool_calls": calls_to_validate,
            "delivery_id": operation_id,
        }):
            raise ValueError("prepared calls lack native or declared application provenance")

    @staticmethod
    def _write_joint_route_plan(
        transaction: Any, *, turn_ref: Any, operation_ref: Any,
        turn: Mapping[str, Any], operation_id: str, safe_route: Mapping[str, Any],
        now: datetime, checkpoint: Any = None,
    ) -> None:
        """Apply the already validated native and summary write plan atomically."""
        if checkpoint is not None:
            transaction.set(turn_ref, {
                **turn,
                "conversation_id": checkpoint.conversation_id,
                "delivery_id": operation_id,
                "application_route": checkpoint.stored_route,
                "application_route_round": checkpoint.route_round,
                "tool_calls": checkpoint.stored_calls,
            }, merge=False)
        transaction.update(operation_ref, {
            "route_control": dict(safe_route),
            "route_control_recorded_at": now,
        })

    @staticmethod
    def _route_control_data(route: Mapping[str, Any]) -> dict[str, Any]:
        """Project route state to metadata safe for the operation ledger."""
        _reject_unsupported_route_metadata(route)
        allowed_status = {
            "ok", "success", "invalid_route", "uncertain", "error",
            "not_found", "unconfirmed",
        }
        allowed_kinds = {"pure_switch", "search", "substantive", "save", "retry", "annotation"}
        status = route.get("status")
        kind = route.get("kind")
        if not isinstance(status, str) or status not in allowed_status:
            raise ValueError("route status is invalid")
        if kind is not None and (not isinstance(kind, str) or kind not in allowed_kinds):
            raise ValueError("route kind is invalid")
        safe: dict[str, Any] = {"status": status}
        if kind is not None:
            safe["kind"] = kind
        selection_policy = route.get("selection_policy")
        if selection_policy is not None:
            if selection_policy != "silent_v1" or kind not in {"pure_switch", "substantive"}:
                raise ValueError("selection_policy is invalid")
            safe["selection_policy"] = selection_policy
        for field in (
            "target_doc_id", "conversation_id", "target_conversation_id", "target_draft_id",
            "target_operation_id", "dependency_operation_id",
        ):
            value = route.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field} is invalid")
                safe[field] = value
        retry_request_operation_id = route.get("retry_request_operation_id")
        if retry_request_operation_id is not None:
            if kind != "retry":
                raise ValueError("retry_request_operation_id is only valid for retry routes")
            safe["retry_request_operation_id"] = _identifier(
                retry_request_operation_id, "retry_request_operation_id",
            )
        number = route.get("dialog_number")
        if number is not None:
            if type(number) is not int or number < 1:
                raise ValueError("dialog_number is invalid")
            safe["dialog_number"] = number
        if retry_request_operation_id is not None and number is None:
            raise ValueError("retry work routes require a dialog_number")
        choice = route.get("tool_choice")
        search_mode = route.get("search_protocol") == "agent-v1"
        if choice is not None:
            if kind not in {"substantive", "search"} or not isinstance(choice, Mapping) or set(choice) - {
                "name", "arguments", "link_requested",
            }:
                raise ValueError("tool_choice is invalid")
            name = choice.get("name")
            link_requested = choice.get("link_requested", False)
            try:
                arguments = validate_arguments(name, choice.get("arguments"))
            except (TypeError, ValueError) as exc:
                raise ValueError("tool_choice is invalid") from exc
            if (not isinstance(name, str) or name not in OPERATIONS
                    or not isinstance(link_requested, bool)
                    or (link_requested and name != "get_transcript")
                    or (kind == "search" and (name != "query" or link_requested))
                    or (not search_mode and arguments.get("doc_id") is not None
                        and arguments["doc_id"] != safe.get("target_doc_id"))):
                raise ValueError("tool_choice is invalid")
            safe["tool_choice"] = {
                "name": name, "arguments": arguments,
                "link_requested": link_requested,
            }
        if kind == "search":
            if route.get("search_protocol") != "agent-v1" or choice is None:
                raise ValueError("search route protocol is invalid")
            safe["search_protocol"] = "agent-v1"
        refs = route.get("result_refs")
        if refs is not None:
            if (kind != "search" or not isinstance(refs, list)
                    or any(not isinstance(ref, str) or not ref for ref in refs)):
                raise ValueError("result_refs is invalid")
            safe["result_refs"] = list(refs)
        session_id = route.get("search_session_id")
        if session_id is not None:
            if kind != "search" or not isinstance(session_id, str) or not session_id:
                raise ValueError("search_session_id is invalid")
            safe["search_session_id"] = session_id
        if kind == "annotation":
            payload = route.get("annotation_payload")
            text_digest = route.get("annotation_text_sha256")
            legacy_text = route.get("annotation_text")
            if (payload is None and isinstance(legacy_text, str) and legacy_text.strip()):
                safe["annotation_text"] = legacy_text
            elif (not isinstance(payload, Mapping)
                    or not isinstance(text_digest, str)
                    or len(text_digest) != 64
                    or any(char not in "0123456789abcdef" for char in text_digest)):
                raise ValueError("annotation payload is invalid")
            if isinstance(payload, Mapping):
                inline = payload.get("_inline")
                input_ref = payload.get("_input_ref")
                if (isinstance(inline, str) and set(payload) <= {
                        "_inline", "sha256", "byte_length"
                    }):
                    if (not inline.strip() or type(payload.get("byte_length")) is not int
                        or payload["byte_length"] < 0
                        or not isinstance(payload.get("sha256"), str)):
                        raise ValueError("inline annotation payload is invalid")
                elif (isinstance(input_ref, str)
                    and input_ref.startswith("dialog_operations/telegram-ledger-")
                    and input_ref.endswith("/input_parts/manifest")
                    and type(payload.get("byte_length")) is int
                    and payload["byte_length"] > 0
                    and isinstance(payload.get("sha256"), str)
                    and type(payload.get("max_part_bytes")) is int):
                    safe["annotation_payload"] = dict(payload)
                else:
                    raise ValueError("annotation payload reference is invalid")
            if isinstance(payload, Mapping) and "_inline" in payload:
                safe["annotation_payload"] = dict(payload)
            if isinstance(text_digest, str):
                safe["annotation_text_sha256"] = text_digest
        if route.get("search_protocol") == "agent-v1" and kind != "search":
            if (kind != "substantive" or not isinstance(safe.get("conversation_id"), str)
                    or not safe.get("conversation_id")):
                raise ValueError("search continuation binding is invalid")
            safe["search_protocol"] = "agent-v1"
        return safe

    def _command_for_operation(
        self, transaction: Any, *, context_id: str, operation_id: str,
        operation: Mapping[str, Any],
    ):
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if (len(parts) != 4 or parts[:3] != [SESSIONS, context_id, "commands"]):
            return None, None
        try:
            _identifier(parts[3], "command_id")
        except ValueError:
            return None, None
        command_ref = _document_ref(self.client, input_ref)
        snapshot = command_ref.get(transaction=transaction)
        command = snapshot.to_dict() or {} if snapshot.exists else {}
        if (not snapshot.exists or command.get("context_id") != context_id
                or command.get("operation_id") != operation_id):
            return None, None
        return command_ref, command

    def _retry_work_route_is_pinned(
        self, transaction: Any, *, context_id: str, operation_id: str,
        operation: Mapping[str, Any], route: Mapping[str, Any],
    ) -> bool:
        """Validate a retry work replay against its immutable source command."""
        request_operation_id = operation.get("retry_request_operation_id")
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        pinned = operation.get("route_control")
        number = route.get("dialog_number")
        if (
            operation.get("operation_id") != operation_id
            or not isinstance(request_operation_id, str) or not request_operation_id
            or len(parts) != 4 or parts[:3] != [SESSIONS, context_id, "commands"]
            or not parts[3].isdigit()
            or route.get("kind") != "retry"
            or route.get("status") not in {"ok", "success"}
            or route.get("retry_request_operation_id") != request_operation_id
            or type(number) is not int or number < 1
            or not isinstance(pinned, Mapping) or dict(pinned) != dict(route)
        ):
            return False

        source_ref = _document_ref(self.client, input_ref)
        source_snapshot = source_ref.get(transaction=transaction)
        source_command = source_snapshot.to_dict() or {} if source_snapshot.exists else {}
        source_operation_ref = self.client.collection(SESSIONS).document(context_id).collection(
            "operations"
        ).document(request_operation_id)
        source_operation_snapshot = source_operation_ref.get(transaction=transaction)
        source_operation = (
            source_operation_snapshot.to_dict() or {}
            if source_operation_snapshot.exists else {}
        )
        receipt_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/updates/{parts[3]}",
        )
        receipt_snapshot = receipt_ref.get(transaction=transaction)
        receipt = receipt_snapshot.to_dict() or {} if receipt_snapshot.exists else {}
        source_route = source_operation.get("route_control")
        return bool(
            source_command.get("context_id") == context_id
            and source_command.get("operation_id") == request_operation_id
            and source_command.get("kind") == "freeform"
            and source_command.get("update_id") == int(parts[3])
            and source_operation.get("operation_id") == request_operation_id
            and source_operation.get("input_ref") == input_ref
            and isinstance(source_route, Mapping)
            and source_route.get("status") in {"ok", "success"}
            and source_route.get("kind") == "retry"
            and type(source_route.get("dialog_number")) is int
            and source_route.get("dialog_number") == number
            and receipt.get("accepted") is True
            and receipt.get("context_id") == context_id
            and receipt.get("operation_id") == request_operation_id
        )

    def persist_route(
        self, operation_id: str, route: Mapping[str, Any], *, context_id: str,
        owner: str, fence: int, now: datetime, application_checkpoint: Any = None,
    ) -> bool:
        """Persist sanitized route state only under the current operation lease."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(route, Mapping):
            raise TypeError("route must be a mapping")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        safe_route = self._route_control_data(route)
        prepared_type = None
        if application_checkpoint is not None:
            from services.connectors.telegram.runtime.ledger import (
                PreparedApplicationRoute, _PREPARED_APPLICATION_ROUTE_TOKEN,
            )
            prepared_type = PreparedApplicationRoute
            if (type(application_checkpoint) is not prepared_type
                    or getattr(application_checkpoint, "_token", None)
                    is not _PREPARED_APPLICATION_ROUTE_TOKEN):
                raise TypeError("application checkpoint must be server-prepared")
            if (application_checkpoint.context_id != context_id
                    or application_checkpoint.operation_id != operation_id
                    or application_checkpoint.delivery_id != operation_id
                    or application_checkpoint.owner != owner
                    or application_checkpoint.fence != fence
                    or application_checkpoint._client is not self.client):
                raise ValueError("prepared application checkpoint scope changed")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            scope = self._read_joint_route_scope(
                transaction, context_ref=context_ref, operation_ref=operation_ref,
                context_id=context_id, operation_id=operation_id,
                owner=owner, fence=fence, now=now,
            )
            if scope is None:
                return False
            context, operation, command_ref, command = scope
            turn_ref = request_ref = None
            turn_snapshot = request_snapshot = None
            turn = request = {}
            prepared_route = None
            prepared_calls = None
            if application_checkpoint is not None:
                checkpoint = application_checkpoint
                if (checkpoint._token is None
                        or checkpoint.delivery_id != operation_id
                        or checkpoint.owner != owner or checkpoint.fence != fence
                        or checkpoint._client is not self.client
                        or type(checkpoint.route_round) is not (int if checkpoint.route_round is not None else type(None))):
                    raise ValueError("prepared application checkpoint is invalid")
                if command is None:
                    return False
                actor_id = command.get("user_id")
                if type(actor_id) is not int or actor_id <= 0:
                    raise PermissionError("accepted command actor is invalid")
                contextual, context_data = self._read_joint_source_context(
                    transaction, checkpoint, context_id=context_id,
                    operation_id=operation_id, actor_id=actor_id,
                    command_ref=command_ref, command=command, operation=operation,
                    safe_route=safe_route,
                )
                turn_ref = (self.client.collection(SESSIONS).document(context_id)
                            .collection("operations").document(operation_id)
                            .collection("agent_checkpoints").document("turn"))
                request_ref = (self.client.collection(SESSIONS).document(context_id)
                               .collection("operations").document(operation_id)
                               .collection("agent_requests").document("request"))
                turn_snapshot = turn_ref.get(transaction=transaction)
                request_snapshot = request_ref.get(transaction=transaction)
                turn = turn_snapshot.to_dict() or {} if turn_snapshot.exists else {}
                request = request_snapshot.to_dict() or {} if request_snapshot.exists else {}
                if (turn.get("conversation_id") not in (None, checkpoint.conversation_id)
                        or request.get("conversation_id") not in (None, checkpoint.conversation_id)):
                    raise ValueError("agent checkpoint conversation binding changed")
                if turn.get("delivery_id") not in (None, operation_id) or request.get("delivery_id") not in (None, operation_id):
                    raise ValueError("agent checkpoint operation binding changed")
                if request.get("request") is not None:
                    self._load_joint_payload(request.get("request"), "request", context_id,
                                             operation_id, transaction)
                stored_route = checkpoint.stored_route
                route_value = self._load_joint_payload(stored_route, "application-route", context_id,
                                                       operation_id, transaction)
                if route_value != contextual:
                    raise ValueError("prepared application route payload changed")
                prepared_route = route_value
                prepared_calls = self._load_joint_payload(
                    checkpoint.stored_calls, "calls", context_id, operation_id, transaction,
                )
                if not isinstance(prepared_calls, list):
                    raise ValueError("prepared application calls are invalid")
                for expected_ordinal, call in enumerate(prepared_calls):
                    if (not isinstance(call, Mapping)
                            or type(call.get("ordinal")) is not int
                            or call.get("ordinal") != expected_ordinal):
                        raise ValueError("prepared application call ordinals are invalid")
                if {"name", "arguments", "ordinal", "target_doc_id"}.issubset(prepared_route):
                    matches = [call for call in prepared_calls
                               if call.get("ordinal") == prepared_route["ordinal"]]
                    expected_call_id = prepared_route.get("call_id")
                    if (len(matches) != 1
                            or matches[0].get("name") != prepared_route.get("name")
                            or matches[0].get("arguments") != prepared_route.get("arguments")
                            or (expected_call_id is not None
                                and matches[0].get("id") != expected_call_id)):
                        raise ValueError("prepared call does not match application route")
                self._validate_joint_native_state(
                    turn, prepared_route=prepared_route, prepared_calls=prepared_calls,
                    checkpoint=checkpoint, context_id=context_id,
                    operation_id=operation_id, transaction=transaction,
                )
                if (turn.get("application_route") is not None
                        and isinstance(prepared_route.get("contextual"), Mapping)
                        and (not isinstance(operation.get("route_control"), Mapping)
                             or dict(operation["route_control"]) != safe_route)):
                    # An existing contextual half is immutable authority only
                    # as a complete pair; never finish or upgrade an orphan.
                    return False
            if (application_checkpoint is not None
                    and isinstance(operation.get("route_control"), Mapping)
                    and dict(operation["route_control"]) == safe_route
                    and turn.get("application_route") is not None):
                return self._prepared_lease_is_current(
                    context, operation_id, owner, fence, application_checkpoint,
                )
            if command is None:
                existing_route = operation.get("route_control")
                if (not isinstance(existing_route, Mapping)
                        or not self._retry_work_route_is_pinned(
                            transaction, context_id=context_id,
                            operation_id=operation_id, operation=operation,
                            route=safe_route,
                        )):
                    return False
                return dict(existing_route) == safe_route
            existing_route = operation.get("route_control")
            if isinstance(existing_route, Mapping):
                if existing_route.get("status") in {"ok", "success"}:
                    same_summary = dict(existing_route) == safe_route
                    if application_checkpoint is None:
                        return same_summary
                    if not same_summary or not turn.get("application_route"):
                        return False
                    if (application_checkpoint is not None
                            and not self._prepared_lease_is_current(
                                context, operation_id, owner, fence, application_checkpoint,
                            )):
                        return False
                    return True
            if application_checkpoint is not None:
                if ("contextual" not in prepared_route
                        or turn.get("application_route") is not None
                        and turn.get("application_route") != checkpoint.stored_route):
                    raise ValueError("joint application route checkpoint conflicts")
                if not self._prepared_lease_is_current(
                    context, operation_id, owner, fence, checkpoint,
                ):
                    return False
            self._write_joint_route_plan(
                transaction, turn_ref=turn_ref, operation_ref=operation_ref,
                turn=turn, operation_id=operation_id, safe_route=safe_route,
                now=now, checkpoint=application_checkpoint,
            )
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def get_route_control(self, context_id: str, operation_id: str) -> dict[str, Any] | None:
        """Read the allowlisted pinned route for safe operation replay."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        ref = self.client.collection(SESSIONS).document(context_id).collection("operations").document(operation_id)
        snapshot = ref.get()
        if not snapshot.exists:
            return None
        value = (snapshot.to_dict() or {}).get("route_control")
        if not isinstance(value, Mapping):
            return None
        result = dict(value)
        _reject_unsupported_route_metadata(result)
        return result

    def get_ingress_binding(self, context_id: str, operation_id: str) -> dict[str, Any] | None:
        """Read this operation's server-owned provisional forwarding association."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        snapshot = (self.client.collection(SESSIONS).document(context_id)
                    .collection("ingress_bindings").document(operation_id).get())
        row = snapshot.to_dict() or {} if snapshot.exists else None
        if (not isinstance(row, Mapping) or row.get("context_id") != context_id
                or row.get("operation_id") != operation_id):
            return None
        return dict(row)

    def get_switch_notice_state(self, context_id: str, operation_id: str) -> str | None:
        """Read the durable switch-notice disposition for transport replay."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        ref = self.client.collection(SESSIONS).document(context_id).collection("operations").document(operation_id)
        snapshot = ref.get()
        state = (snapshot.to_dict() or {}).get("switch_notice_state") if snapshot.exists else None
        return state if state in {"pending", "delivered", "failed", "delivery_unknown"} else None

    def set_switch_notice_state(
        self, context_id: str, operation_id: str, state: str, *,
        owner: str, fence: int, now: datetime,
    ) -> bool:
        """Persist terminal notice disposition without changing the pinned target."""
        if state not in {"pending", "delivered", "failed", "delivery_unknown"}:
            raise ValueError("switch notice state is invalid")
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        fence, now = _integer(fence, "fence", minimum=1), _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not context_snapshot.exists or not operation_snapshot.exists:
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if (operation.get("operation_id") != operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"),
                    )):
                return False
            if not isinstance(operation.get("route_control"), Mapping):
                return False
            previous = operation.get("switch_notice_state")
            if previous in {"failed", "delivery_unknown"}:
                return state == previous
            if previous == "delivered" and state != "delivered":
                return False
            if previous == state:
                return True
            transaction.update(operation_ref, {"switch_notice_state": state, "updated_at": now})
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def clear_command_candidate(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        now: datetime,
    ) -> bool:
        """Delete only a successfully routed candidate under its current fence.

        The route record and command link are rechecked in this transaction.
        Replays after deletion are idempotent and preserve every other field.
        """
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not context_snapshot.exists or not operation_snapshot.exists:
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if (operation.get("operation_id") != operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"),
                    )):
                return False
            route = operation.get("route_control")
            if (not isinstance(route, Mapping)
                    or route.get("status") not in {"ok", "success"}):
                return False
            command_ref, command = self._command_for_operation(
                transaction, context_id=context_id,
                operation_id=operation_id, operation=operation,
            )
            if command_ref is None or command is None:
                return False
            if "candidate" not in command:
                return True
            if not isinstance(command.get("candidate"), str):
                return False
            transaction.update(command_ref, {"candidate": firestore.DELETE_FIELD})
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def _accept_command_in_transaction(
        self,
        transaction: Any,
        *,
        identity: _CommandIdentity,
        update_id: int,
        chat_id: int,
        user_id: int,
        thread_id: int | None,
        topic_name: str | None,
        command: Any,
        message_id: int | None,
        transport_timestamp: int | None,
        reply_to_message_id: int | None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Coordinate the all-reads-first freeform command reservation."""
        receipt_snapshot = identity.receipt_ref.get(transaction=transaction)
        if receipt_snapshot.exists:
            return _receipt_result(receipt_snapshot.to_dict() or {}, identity.context_id)

        state = _load_command_acceptance_state(
            self.client,
            transaction,
            identity=identity,
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
            thread_id=thread_id,
            topic_name=topic_name,
            command=command,
            now=now,
            message_id=message_id,
            message_date=_trusted_transport_date(transport_timestamp),
            reply_to_message_id=reply_to_message_id,
        )
        # Capture a single trusted addressed presentation hint only after the
        # duplicate receipt fast path above.  It is never a selector or write
        # authority; later routing validates its causal meaning.
        state.command_data["observed_presentation_id"] = (
            self.observe_published_presentation_in_transaction(
                transaction, context_id=identity.context_id,
                context=state.context, actor_id=user_id, now=now,
            )
        )
        message_date = _trusted_transport_date(transport_timestamp)
        observed_id, observed_generation, delivery_ref = _recorded_question_observation(
            self.client, transaction,
            context_id=identity.context_id,
            context=state.context,
            user_id=user_id,
            chat_id=chat_id,
            thread_id=thread_id,
            message_date=message_date,
            reply_to_message_id=reply_to_message_id,
            now=now,
        )
        state.command_data.update({
            "observed_clarification_id": observed_id,
            "observed_clarification_generation": observed_generation,
            "observed_question_delivery_ref": delivery_ref,
        })
        ingress_binding = None
        if command.kind == "freeform" and message_id is not None:
            recent_source = _bind_text_to_recent_receipt_group(
                self.client, transaction, context_id=identity.context_id,
                context=state.context, draft_id=state.snapshot.draft_id,
                user_id=user_id, message_id=message_id,
                transport_timestamp=transport_timestamp,
            )
            ingress_binding = {
                "context_id": identity.context_id,
                "operation_id": identity.operation_id,
                "owner_user_id": user_id,
                "candidate_message_id": message_id,
                "transport_timestamp": transport_timestamp,
                "state": "collecting" if recent_source else "pending",
                "draft_id": recent_source.get("draft_id") if recent_source else None,
                "receipt_group_id": recent_source.get("receipt_group_id") if recent_source else None,
                "matched_message_ids": recent_source.get("matched_message_ids", [])
                if recent_source else [],
                "source_binding_finalized": bool(recent_source),
                "created_at": now,
                "deadline_at": now + timedelta(seconds=1 if recent_source else 5),
                "max_deadline_at": now + timedelta(seconds=30),
            }
        operation = _register_command_operation(
            self.client,
            transaction,
            context_ref=identity.context_ref,
            identity=identity,
            now=now,
            snapshot=state.snapshot,
            action_metadata=(command.action_metadata
                             if isinstance(command.action_metadata, Mapping) else None),
            ingress_binding_id=(identity.operation_id if ingress_binding is not None else None),
        )
        _update_command_context(
            state.context,
            chat_id=chat_id,
            thread_id=thread_id,
            topic_name=topic_name,
            operation_id=identity.operation_id,
            sequence=operation.sequence,
            now=now,
        )
        if (ingress_binding is not None
                and state.context.get("reserved_operation_id") == identity.operation_id):
            state.context["pending_ingress_operation_id"] = identity.operation_id
        return _persist_command_acceptance(
            transaction,
            identity=identity,
            update_id=update_id,
            state=state,
            operation=operation,
            now=now,
            ingress_binding=ingress_binding,
        )


    def accept_command(
        self,
        update_id: int,
        *,
        chat_id: int,
        user_id: int,
        thread_id: int | None = None,
        topic_name: str | None = None,
        command: Any,
        message_id: int | None = None,
        transport_timestamp: int | None = None,
        reply_to_message_id: int | None = None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Accept command text and reserve its FIFO position with a save cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        now = _aware(now, "now")
        if thread_id == 0:
            thread_id = None
        elif thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(command, TelegramCommand):
            raise TypeError("command must be a TelegramCommand")
        if message_id is not None:
            message_id = _integer(message_id, "message_id", minimum=1)
        if transport_timestamp is not None:
            transport_timestamp = _integer(transport_timestamp, "transport_timestamp", minimum=1)
        if reply_to_message_id is not None:
            reply_to_message_id = _integer(reply_to_message_id, "reply_to_message_id", minimum=1)
        context_id = build_context_id(chat_id, thread_id)
        identity = _prepare_command_identity(
            self.client,
            context_id=context_id,
            update_id=update_id,
            command=command,
        )

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            return self._accept_command_in_transaction(
                transaction,
                identity=identity,
                update_id=update_id,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
                topic_name=topic_name,
                command=command,
                message_id=message_id,
                transport_timestamp=transport_timestamp,
                reply_to_message_id=reply_to_message_id,
                now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))


    def accept_receipt_save_callback(
        self,
        update_id: int,
        *,
        chat_id: int,
        user_id: int,
        thread_id: int | None,
        message_id: int,
        control_token: str,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Consume a current persisted Save control and reserve its exact draft cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        message_id = _integer(message_id, "message_id", minimum=1)
        thread_id = (None if thread_id == 0 else
                     _integer(thread_id, "thread_id", minimum=1)
                     if thread_id is not None else None)
        control_token = _identifier(control_token, "control_token")
        now = _aware(now, "now")
        context_id = build_context_id(chat_id, thread_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        control_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/receipt_controls/{control_token}",
        )
        control_snapshot = control_ref.get()
        control_data = control_snapshot.to_dict() or {} if control_snapshot.exists else {}
        notification_id = control_data.get("notification_id")
        notification_ref = (self.client.collection("notification_outbox").document(notification_id)
                            if isinstance(notification_id, str) else None)
        notification_snapshot = notification_ref.get() if notification_ref is not None else None
        notification = (notification_snapshot.to_dict() or {}
                        if notification_snapshot is not None and notification_snapshot.exists else {})
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: Any) -> TelegramIntakeResult:
            replay_snapshot = receipt_ref.get(transaction=transaction)
            if replay_snapshot.exists:
                return _receipt_result(replay_snapshot.to_dict() or {}, context_id)
            control_now = control_ref.get(transaction=transaction)
            control = control_now.to_dict() or {} if control_now.exists else {}
            context_snapshot = context_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            live_notification_snapshot = (
                notification_ref.get(transaction=transaction)
                if notification_ref is not None else None
            )
            row = (live_notification_snapshot.to_dict() or {}
                   if live_notification_snapshot is not None and live_notification_snapshot.exists else {})
            draft_id = row.get("draft_id")
            draft_ref = (_document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
                         if isinstance(draft_id, str) else None)
            draft_snapshot = draft_ref.get(transaction=transaction) if draft_ref is not None else None
            draft = (draft_snapshot.to_dict() or {}
                     if draft_snapshot is not None and draft_snapshot.exists else {})

            eligible = _receipt_save_control_matches(
                context_exists=context_snapshot.exists, context=context,
                control=control, row=row, draft=draft, context_id=context_id,
                control_token=control_token, notification_id=notification_id,
                user_id=user_id, message_id=message_id,
            )
            if not eligible:
                transaction.set(receipt_ref, {
                    "update_id": update_id, "context_id": context_id,
                    "accepted": False, "outcome": "receipt_save_rejected",
                    "notification_id": notification_id, "message_id": message_id,
                    "created_at": now,
                })
                return TelegramIntakeResult(False, context_id)

            action_metadata = {
                "action": "receipt_save", "notification_id": notification_id,
                "control_token": control_token, "draft_id": draft_id,
                "dialog_number": row.get("dialog_number"), "card_message_id": message_id,
                "card_revision": row.get("revision"), "user_id": user_id,
            }
            command = TelegramCommand("save", "", action_metadata)
            identity = _prepare_command_identity(
                self.client, context_id=context_id, update_id=update_id, command=command,
            )
            saving_outbox_id = "tgcard-saving-" + hashlib.sha256(
                f"{notification_id}:{identity.operation_id}".encode()
            ).hexdigest()
            saving_outbox = prepare_outbox_record_in_transaction(
                self.client, transaction, saving_outbox_id,
                target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
                schedule_at=now, now=now, context_id=context_id,
                notification_id=notification_id, operation_id=identity.operation_id,
            )
            result = self._accept_command_in_transaction(
                transaction, identity=identity, update_id=update_id,
                chat_id=chat_id, user_id=user_id, thread_id=thread_id,
                topic_name=context.get("topic_name"), command=command, now=now,
                message_id=None, transport_timestamp=None,
                reply_to_message_id=None,
            )
            write_prepared_outbox_record(transaction, saving_outbox)
            next_revision = _integer(row.get("revision", 1), "revision", minimum=1) + 1
            row.update({
                "body": f"Dialog #{row.get('dialog_number')}: Saving…",
                "reply_markup": {"inline_keyboard": []},
                "receipt_control_token": None,
                "receipt_control_owner_user_id": None,
                "save_operation_id": result.operation_id,
                "operation_id": result.operation_id,
                "revision": next_revision,
                "desired_status_revision": next_revision,
                "state": row.get("state") if row.get("in_flight_attempt_id") else "pending",
                "delivered_at": None, "updated_at": now,
            })
            transaction.set(notification_ref, row)
            transaction.update(control_ref, {"state": "consumed", "operation_id": result.operation_id,
                                             "consumed_at": now})
            transaction.update(receipt_ref, {
                "notification_id": notification_id, "receipt_group_id": row.get("receipt_group_id"),
                "card_message_id": message_id,
                "additional_outbox_ids": [saving_outbox_id],
            })
            return TelegramIntakeResult(
                result.accepted, result.context_id, result.draft_id, result.dialog_number,
                result.message_id, result.intake_sequence, result.notification_id,
                result.outbox_id, result.operation_id, result.queue_position,
                result.snapshot_message_ids, (saving_outbox_id,),
            )

        return _txn(self.client.transaction(max_attempts=5))
