"""Addressed clarification reads and transaction-scoped guard snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION, OutboxTargetKind, prepare_outbox_record_in_transaction,
    write_prepared_outbox_record,
)

from core.integrations.firestore.operation_queue import validate_context_queue_lease
from services.connectors.telegram.models import (
    ClarificationState,
    PendingClarification,
)
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS, SESSIONS, _document_ref, _read_routing_checkpoint_value,
    _routing_reference_data, _routing_json_bytes,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_PREPARED_TOKEN = object()


class _ClarificationReplayCommitted(Exception):
    """Abort a concurrent identical transaction and resolve it from durable state."""


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_clarification_request(
    pending: PendingClarification, context_reference: Mapping[str, Any], *,
    question: Mapping[str, Any], prompt_body: str, presentation: str | None = None,
) -> dict[str, Any]:
    """Build the full immutable v1 intent payload without clipping user content."""
    if not isinstance(pending, PendingClarification):
        raise TypeError("pending must be a PendingClarification")
    context_ref = _routing_reference_data(context_reference)
    if (context_ref["kind"] != "context" or context_ref["context_id"] != pending.context_id
            or context_ref["operation_id"] != pending.initiating_operation_id
            or context_ref["actor_id"] != pending.actor_id
            or context_ref["conversation_id"] != pending.origin_conversation_id):
        raise ValueError("clarification context checkpoint binding is invalid")
    if not isinstance(question, Mapping):
        raise TypeError("question must be a mapping")
    if not isinstance(prompt_body, str) or not prompt_body.strip() or len(prompt_body) > 4096:
        raise ValueError("clarification prompt body must be nonblank and at most 4096 characters")
    if presentation not in (None, "rich_markdown"):
        raise ValueError("unsupported clarification presentation")
    value = {
        "protocol": "clarification-request-v1",
        "clarification": {
            **pending.to_dict(),
            "created_at": _utc_iso(pending.created_at),
            "updated_at": _utc_iso(pending.updated_at),
        },
        "context_reference": dict(context_ref),
        "question": dict(question),
        "prompt": {"body": prompt_body, "presentation": presentation},
    }
    # Reuse the existing strict canonical JSON validator for arbitrary nested data.
    return json.loads(_routing_json_bytes(value))


@dataclass(frozen=True, slots=True, init=False)
class PreparedClarificationCompletion:
    """Factory-issued, immutable proof that both private checkpoints were verified."""

    intent_reference_json: bytes
    context_reference_json: bytes
    request_json: bytes
    context_value_json: bytes
    pending: PendingClarification
    command_identity: str
    command_kind: str
    command_action_metadata_json: bytes
    _token: object

    def __init__(self, token: object, *, intent_reference: Mapping[str, Any],
                 context_reference: Mapping[str, Any], request: Mapping[str, Any],
                 context_value: Mapping[str, Any], pending: PendingClarification,
                 command_identity: str, command_kind: str,
                 command_action_metadata: Any) -> None:
        if token is not _PREPARED_TOKEN:
            raise TypeError("prepared clarification completion must come from the repository")
        object.__setattr__(self, "_token", token)
        for name, value in (("intent_reference_json", intent_reference),
                            ("context_reference_json", context_reference),
                            ("request_json", request), ("context_value_json", context_value)):
            object.__setattr__(self, name, _routing_json_bytes(dict(value)))
        object.__setattr__(self, "pending", pending)
        object.__setattr__(self, "command_identity", command_identity)
        object.__setattr__(self, "command_kind", command_kind)
        object.__setattr__(self, "command_action_metadata_json", _routing_json_bytes(command_action_metadata))

    @property
    def intent_reference(self) -> dict[str, Any]:
        return json.loads(self.intent_reference_json)

    @property
    def context_reference(self) -> dict[str, Any]:
        return json.loads(self.context_reference_json)

    @property
    def request(self) -> dict[str, Any]:
        return json.loads(self.request_json)

    @property
    def context_value(self) -> dict[str, Any]:
        return json.loads(self.context_value_json)


def is_prepared_clarification_completion(value: Any) -> bool:
    return (type(value) is PreparedClarificationCompletion
            and getattr(value, "_token", None) is _PREPARED_TOKEN)


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return _routing_json_bytes(left) == _routing_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _operation_has_clarification_completion(
    operation: Mapping[str, Any], *, operation_id: str,
    identity_digest: str, input_ref: str, binding: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    return bool(
        type(operation.get("operation_id")) is str
        and operation.get("operation_id") == operation_id
        and type(operation.get("state")) is str
        and operation.get("state") == "completed"
        and type(operation.get("identity_digest")) is str
        and operation.get("identity_digest") == identity_digest
        and type(operation.get("input_ref")) is str
        and operation.get("input_ref") == input_ref
        and type(operation.get("result_type")) is str
        and operation.get("result_type") == "clarification"
        and _canonical_equal(operation.get("clarification_completion_binding"), binding)
        and _canonical_equal(operation.get("result"), result)
    )


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid clarification {field}")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"invalid clarification {field}")
    return value


def _validated_thread_id(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is int and value == 0:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("stored thread routing is invalid")
    return value


def _aware(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("invalid clarification now")
    return value


def _strict_scoped_record(
    data: Mapping[str, Any], *, context_id: str, clarification_id: str, actor_id: int,
) -> PendingClarification | None:
    if (data.get("context_id") != context_id
            or data.get("clarification_id") != clarification_id
            or type(data.get("actor_id")) is not int
            or data.get("actor_id") != actor_id
            or data.get("protocol") is None):
        return None
    if data.get("protocol") != "clarification-v1":
        raise ValueError("unsupported clarification protocol")
    return PendingClarification.from_dict(data)


def _read_waiting_target(
    repository: Any, transaction: Any, *, context_id: str, clarification_id: str,
    actor_id: int, generation: int, revision: int, pointer_id: Any,
    pointer_generation: Any,
) -> PendingClarification | None:
    if (pointer_id != clarification_id or type(pointer_generation) is not int
            or pointer_generation != generation):
        return None
    ref = _document_ref(
        repository.client,
        f"{SESSIONS}/{context_id}/clarifications/{clarification_id}",
    )
    snapshot = ref.get(transaction=transaction)
    if not snapshot.exists:
        return None
    clarification = _strict_scoped_record(
        snapshot.to_dict() or {}, context_id=context_id,
        clarification_id=clarification_id, actor_id=actor_id,
    )
    if (clarification is None or clarification.state is not ClarificationState.WAITING
            or clarification.generation != generation
            or clarification.revision != revision):
        return None
    return clarification


def _accepted_scope(
    repository: Any, transaction: Any, *, context_id: str, operation_id: str,
    actor_id: int, owner: str, fence: int, now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    context_ref = _document_ref(repository.client, f"{SESSIONS}/{context_id}")
    operation_ref = context_ref.collection("operations").document(operation_id)
    context_snapshot = context_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    if not context_snapshot.exists or not operation_snapshot.exists:
        return None
    context = context_snapshot.to_dict() or {}
    operation = operation_snapshot.to_dict() or {}
    if (operation.get("operation_id") != operation_id
            or operation.get("state") != "running"
            or not validate_context_queue_lease(
                context, operation_id, owner=owner, fence=fence, now=now,
                expiry=context.get("lock_expires_at"),
            )):
        return None
    _command_ref, command = repository._command_for_operation(
        transaction, context_id=context_id, operation_id=operation_id,
        operation=operation,
    )
    if (command is None or command.get("context_id") != context_id
            or command.get("operation_id") != operation_id
            or type(command.get("user_id")) is not int
            or command.get("user_id") != actor_id):
        return None
    return context, operation, command


@dataclass(frozen=True, slots=True)
class _ClarificationGuard:
    """Transaction-read snapshots; never a capability to write or execute."""

    context: Mapping[str, Any]
    operation: Mapping[str, Any]
    accepted_command: Mapping[str, Any]
    clarification: PendingClarification | None


def _read_clarification_publication_targets(
    transaction: Any, *, parent_ref: Any, question_ref: Any,
    notification_ref: Any, task_ref: Any,
) -> None:
    """Read all deterministic publication targets before any write is staged."""
    snapshots = (
        parent_ref.get(transaction=transaction),
        question_ref.get(transaction=transaction),
        notification_ref.get(transaction=transaction),
        task_ref.get(transaction=transaction),
    )
    if any(snapshot.exists for snapshot in snapshots):
        raise ValueError("clarification publication target already exists")


def _prepare_clarification_outboxes(
    client: Any, transaction: Any, *, context_id: str, next_id: str | None,
    next_data: dict[str, Any] | None, updated_context: dict[str, Any],
    prepare_wakeup: Any, task_id: str, notification_id: str, now: datetime,
) -> tuple[Any, list[Any]]:
    """Prepare prompt and next-head outbox writes using existing outbox primitives."""
    if next_id and next_data is not None and prepare_wakeup is None:
        raise ValueError("next queue head requires a wakeup preparer")
    wake_writes: list[Any] = []
    if next_id and next_data is not None:
        wake_writes = list(prepare_wakeup(transaction, next_id, next_data, updated_context))
    prompt_task = prepare_outbox_record_in_transaction(
        client, transaction, task_id, target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now,
        context_id=context_id, notification_id=notification_id,
    )
    return prompt_task, wake_writes


def _write_clarification_publication(
    transaction: Any, *, parent_ref: Any, parent_data: Mapping[str, Any],
    question_ref: Any, question_data: Mapping[str, Any], notification_ref: Any,
    notification_data: Mapping[str, Any], prompt_task: Any, wake_writes: list[Any],
) -> None:
    """Stage the already prepared clarification documents and outbox rows."""
    transaction.set(parent_ref, dict(parent_data))
    transaction.set(question_ref, dict(question_data))
    transaction.set(notification_ref, dict(notification_data))
    write_prepared_outbox_record(transaction, prompt_task)
    for prepared in wake_writes:
        write_prepared_outbox_record(transaction, prepared)


@dataclass(frozen=True, slots=True)
class _ClarificationPublicationPlan:
    binding: Mapping[str, Any]
    result: Mapping[str, Any]
    parent_ref: Any
    question_ref: Any
    notification_ref: Any
    task_ref: Any
    parent_data: Mapping[str, Any]
    question_data: Mapping[str, Any]
    notification_data: Mapping[str, Any]
    prompt_body: str
    accepted_input_ref: str


def _build_clarification_publication_plan(
    client: Any, *, context_document: Any, context_id: str, operation_id: str,
    completion: PreparedClarificationCompletion, now: datetime,
) -> _ClarificationPublicationPlan:
    """Freeze deterministic IDs and all immutable prompt/publication metadata."""
    pending = completion.pending
    intent_ref, context_ref = completion.intent_reference, completion.context_reference
    request, context_value = completion.request, completion.context_value
    clarification_id, generation = pending.clarification_id, pending.generation
    seed = hashlib.sha256(f"{clarification_id}:{generation}".encode()).hexdigest()
    notification_id, task_id = "tgclarif-" + seed, "tgclarifdispatch-" + seed
    parent_ref = context_document.collection("clarifications").document(clarification_id)
    question_ref = parent_ref.collection("questions").document(str(generation))
    notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
    task_ref = client.collection(OUTBOX_COLLECTION).document(task_id)
    prompt_body = request["prompt"]["body"]
    binding = {
        "context_id": context_id, "operation_id": operation_id,
        "actor_id": pending.actor_id,
        "intent_ref": intent_ref["ref"], "intent_digest": intent_ref["digest"],
        "context_ref": context_ref["ref"], "context_digest": context_ref["digest"],
        "identity_digest": completion.command_identity,
        "input_ref": context_value["accepted_command_ref"]["ref"],
        "clarification_id": clarification_id, "generation": generation,
    }
    result = {
        "clarification_id": clarification_id, "generation": generation,
        "question_ref": question_ref.path, "prompt_notification_id": notification_id,
        "prompt_notification_revision": 1, "prompt_outbox_id": task_id,
        "prompt_outbox_ref": f"{OUTBOX_COLLECTION}/{task_id}",
        "request_digest": intent_ref["digest"],
    }
    question_data = {
        "protocol": "clarification-question-v1", "context_id": context_id,
        "actor_id": pending.actor_id, "clarification_id": clarification_id,
        "generation": generation, "created_at": now,
        "prompt_notification_id": notification_id,
        "prompt_notification_revision": 1,
        "prompt_digest": hashlib.sha256(prompt_body.encode("utf-8")).hexdigest(),
        "missing_slots": list(pending.missing_slots), "question_ref": question_ref.path,
        "question_payload_ref": intent_ref["ref"],
        "question_payload_digest": intent_ref["digest"],
    }
    notification_data = {
        "notification_id": notification_id, "kind": "clarification",
        "context_id": context_id, "actor_id": pending.actor_id,
        "clarification_id": clarification_id, "generation": generation,
        "revision": 1, "desired_status_revision": 1, "body": prompt_body,
        "dialog_number": None, "presentation": request["prompt"].get("presentation"),
        "message_id": None, "operation_id": None,
        "producer_operation_id": operation_id, "state": "pending",
        "send_attempts_started": 0, "edit_attempts_started": 0,
        "unknown_send_outcome": False, "in_flight_attempt_id": None,
        "attempt_expires_at": None, "last_outcome": None, "retry_not_before": None,
        "terminal_failure": None, "last_error": None, "delivered_at": None,
        "failed_at": None, "created_at": now, "updated_at": now,
    }
    return _ClarificationPublicationPlan(
        binding=binding, result=result, parent_ref=parent_ref, question_ref=question_ref,
        notification_ref=notification_ref, task_ref=task_ref,
        parent_data=pending.to_dict(), question_data=question_data,
        notification_data=notification_data, prompt_body=prompt_body,
        accepted_input_ref=context_value["accepted_command_ref"]["ref"],
    )


def _publish_clarification_in_transaction(
    repository: Any, transaction: Any, *, context_id: str, operation_id: str,
    owner: str, fence: int, now: datetime, plan: _ClarificationPublicationPlan,
    completion: PreparedClarificationCompletion, intent_ref: Mapping[str, Any],
    context_ref: Mapping[str, Any], request: Mapping[str, Any],
    context_value: Mapping[str, Any], prepare_wakeup: Any,
    next_id: str | None, next_data: dict[str, Any] | None,
    updated_context: dict[str, Any], operation_data: dict[str, Any],
) -> None:
    """Read, validate, prepare, then stage the complete publication in one transaction."""
    guard = repository._read_clarification_guard(
        transaction, context_id=context_id, operation_id=operation_id,
        actor_id=completion.pending.actor_id, owner=owner, fence=fence, now=now,
    )
    operation_ref = repository.client.collection(SESSIONS).document(context_id).collection(
        "operations").document(operation_id)
    if guard is None:
        current_operation = operation_ref.get(transaction=transaction)
        if (current_operation.exists and _operation_has_clarification_completion(
                current_operation.to_dict() or {}, operation_id=operation_id,
                identity_digest=completion.command_identity, input_ref=plan.accepted_input_ref,
                binding=plan.binding, result=plan.result)):
            raise _ClarificationReplayCommitted()
        raise ValueError("clarification queue lease or free slot changed")
    current_operation = operation_ref.get(transaction=transaction)
    if (current_operation.exists and _operation_has_clarification_completion(
            current_operation.to_dict() or {}, operation_id=operation_id,
            identity_digest=completion.command_identity, input_ref=plan.accepted_input_ref,
            binding=plan.binding, result=plan.result)):
        raise _ClarificationReplayCommitted()
    command = guard.accepted_command
    if (command.get("kind") != completion.command_kind
            or _routing_json_bytes(command.get("action_metadata"))
            != completion.command_action_metadata_json
            or command.get("user_id") != completion.pending.actor_id
            or guard.operation.get("identity_digest") != completion.command_identity
            or guard.operation.get("input_ref") != plan.accepted_input_ref):
        raise ValueError("accepted command changed before clarification completion")
    verified_request = _read_routing_checkpoint_value(
        repository.client, intent_ref, transaction=transaction,
    )
    verified_context = _read_routing_checkpoint_value(
        repository.client, context_ref, transaction=transaction,
    )
    if verified_request != request or verified_context != context_value:
        raise ValueError("clarification checkpoints changed before completion")

    _read_clarification_publication_targets(
        transaction, parent_ref=plan.parent_ref, question_ref=plan.question_ref,
        notification_ref=plan.notification_ref, task_ref=plan.task_ref,
    )
    prompt_task, wake_writes = _prepare_clarification_outboxes(
        repository.client, transaction, context_id=context_id, next_id=next_id,
        next_data=next_data, updated_context=updated_context,
        prepare_wakeup=prepare_wakeup, task_id=plan.result["prompt_outbox_id"],
        notification_id=plan.result["prompt_notification_id"], now=now,
    )
    stored_context = dict(guard.context)
    chat_id, thread_id = stored_context.get("chat_id"), stored_context.get("thread_id")
    if type(chat_id) is not int:
        raise ValueError("stored chat routing is invalid")
    notification_data = {
        **plan.notification_data, "chat_id": chat_id,
        "thread_id": _validated_thread_id(thread_id),
    }
    pending = completion.pending
    updated_context["pending_clarification_id"] = pending.clarification_id
    updated_context["clarification_generation"] = pending.generation
    if updated_context.get("pending_save_operation_id") == operation_id:
        updated_context.pop("pending_save_operation_id", None)
    operation_data["result_type"] = "clarification"
    operation_data["result"] = dict(plan.result)
    operation_data["clarification_completion_binding"] = dict(plan.binding)

    _write_clarification_publication(
        transaction, parent_ref=plan.parent_ref, parent_data=plan.parent_data,
        question_ref=plan.question_ref, question_data=plan.question_data,
        notification_ref=plan.notification_ref, notification_data=notification_data,
        prompt_task=prompt_task, wake_writes=wake_writes,
    )


def _decode_clarification_checkpoint(
    intent_ref: Mapping[str, Any], request: Any, context_value: Any,
) -> tuple[dict[str, Any], PendingClarification, Any, Mapping[str, Any]]:
    """Purely decode and validate the already verified immutable checkpoints."""
    if not isinstance(request, Mapping) or request.get("protocol") != "clarification-request-v1":
        raise ValueError("clarification request protocol is invalid")
    context_ref = _routing_reference_data(request.get("context_reference"))
    if (context_ref["kind"] != "context"
            or context_ref["context_id"] != intent_ref["context_id"]
            or context_ref["operation_id"] != intent_ref["operation_id"]
            or context_ref["actor_id"] != intent_ref["actor_id"]
            or context_ref["conversation_id"] != intent_ref["conversation_id"]):
        raise ValueError("clarification context checkpoint scope is invalid")
    if not isinstance(context_value, Mapping):
        raise ValueError("clarification context snapshot is invalid")
    from services.connectors.telegram.runtime.routing_context import RoutingContextSnapshot
    try:
        snapshot = RoutingContextSnapshot.from_dict(context_value)
    except (TypeError, ValueError, KeyError):
        raise ValueError("clarification context snapshot is invalid") from None
    if (snapshot.context_id != intent_ref["context_id"]
            or snapshot.operation_id != intent_ref["operation_id"]
            or snapshot.actor_id != intent_ref["actor_id"]
            or snapshot.selected_conversation_id != intent_ref["conversation_id"]):
        raise ValueError("clarification context snapshot scope is invalid")
    raw_pending = request.get("clarification")
    if not isinstance(raw_pending, Mapping):
        raise ValueError("clarification request payload is invalid")
    pending_fields = dict(raw_pending)
    for field in ("created_at", "updated_at"):
        value = pending_fields.get(field)
        if not isinstance(value, str):
            raise ValueError("clarification request timestamp is invalid")
        try:
            pending_fields[field] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("clarification request timestamp is invalid") from None
    try:
        pending = PendingClarification.from_dict(pending_fields)
    except (TypeError, ValueError, KeyError):
        raise ValueError("clarification waiting intent is invalid") from None
    if (pending.state is not ClarificationState.WAITING or pending.generation != 1
            or pending.revision != 1 or pending.execution_operation_id is not None
            or pending.result_ref is not None or pending.safe_error is not None
            or pending.initiating_operation_id != intent_ref["operation_id"]
            or pending.context_id != intent_ref["context_id"]
            or pending.actor_id != intent_ref["actor_id"]
            or pending.origin_conversation_id != intent_ref["conversation_id"]
            or pending.original_command_ref != context_ref["ref"]
            or pending.original_command_digest != context_ref["digest"]
            or pending.evidence_snapshot_ref != context_ref["ref"]
            or pending.evidence_snapshot_digest != context_ref["digest"]):
        raise ValueError("clarification waiting intent binding is invalid")
    if not isinstance(request.get("question"), Mapping):
        raise ValueError("clarification question is invalid")
    prompt = request.get("prompt")
    if (not isinstance(prompt, Mapping) or not isinstance(prompt.get("body"), str)
            or not prompt["body"].strip() or len(prompt["body"]) > 4096
            or prompt.get("presentation") not in (None, "rich_markdown")):
        raise ValueError("clarification prompt is invalid")
    return context_ref, pending, snapshot, prompt


def _accepted_command_binding(
    *, operation: Mapping[str, Any], command: Mapping[str, Any], snapshot: Any,
    context_id: str, operation_id: str, actor_id: int,
) -> tuple[str, str, bytes]:
    """Bind the immutable context command to the accepted operation snapshot."""
    if (command.get("kind") is None or type(command.get("user_id")) is not int
            or command.get("user_id") != actor_id):
        raise ValueError("clarification accepted command scope is invalid")
    try:
        identity = hashlib.sha256(json.dumps(
            {"kind": command.get("kind"), "text": snapshot.accepted_command,
             "action_metadata": command.get("action_metadata")}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        metadata = _routing_json_bytes(command.get("action_metadata"))
    except (TypeError, ValueError):
        raise ValueError("clarification accepted command binding is invalid") from None
    command_ref = snapshot.accepted_command_ref
    if (operation.get("identity_digest") != identity
            or operation.get("input_ref") != command_ref.ref
            or command_ref.operation_id != operation_id
            or command_ref.context_id != context_id
            or command_ref.actor_id != actor_id):
        raise ValueError("clarification command checkpoint does not bind the accepted command")
    return identity, command["kind"], metadata


class _TelegramClarificationMixin:
    def prepare_clarification_completion(
        self, intent_reference: Mapping[str, Any],
    ) -> PreparedClarificationCompletion:
        """Verify a published clarification request before entering queue settlement."""
        intent_ref = _routing_reference_data(intent_reference)
        if intent_ref["kind"] != "intent":
            raise ValueError("clarification intent checkpoint is required")
        request = _read_routing_checkpoint_value(self.client, intent_ref)
        if not isinstance(request, Mapping):
            raise ValueError("clarification request payload is invalid")
        context_ref = _routing_reference_data(request.get("context_reference"))
        context_value = _read_routing_checkpoint_value(self.client, context_ref)
        context_ref, pending, snapshot, _prompt = _decode_clarification_checkpoint(
            intent_ref, request, context_value,
        )

        operation_ref = self.client.collection(SESSIONS).document(intent_ref["context_id"]).collection(
            "operations").document(intent_ref["operation_id"])
        operation_snapshot = operation_ref.get()
        if not operation_snapshot.exists:
            raise ValueError("clarification queue operation is absent")
        operation = operation_snapshot.to_dict() or {}
        if (operation.get("operation_id") != intent_ref["operation_id"]
                or operation.get("state") not in {"running", "completed"}):
            raise ValueError("clarification queue operation is not active")
        command_ref, command = self._command_for_operation(
            None, context_id=intent_ref["context_id"], operation_id=intent_ref["operation_id"],
            operation=operation,
        )
        if command_ref is None or not isinstance(command, Mapping):
            raise ValueError("clarification accepted command is absent")
        identity, command_kind, action_metadata = _accepted_command_binding(
            operation=operation, command=command, snapshot=snapshot,
            context_id=intent_ref["context_id"], operation_id=intent_ref["operation_id"],
            actor_id=intent_ref["actor_id"],
        )
        if operation.get("state") == "completed":
            expected_binding = {
                "context_id": intent_ref["context_id"],
                "operation_id": intent_ref["operation_id"],
                "actor_id": intent_ref["actor_id"],
                "intent_ref": intent_ref["ref"], "intent_digest": intent_ref["digest"],
                "context_ref": context_ref["ref"], "context_digest": context_ref["digest"],
                "identity_digest": identity,
                "input_ref": snapshot.accepted_command_ref.ref,
                "clarification_id": pending.clarification_id,
                "generation": pending.generation,
            }
            if (operation.get("result_type") != "clarification"
                    or not _canonical_equal(
                        operation.get("clarification_completion_binding"), expected_binding,
                    )):
                raise ValueError("completed clarification request binding does not match")
        return PreparedClarificationCompletion(
            _PREPARED_TOKEN, intent_reference=intent_ref, context_reference=context_ref,
            request=request, context_value=context_value, pending=pending,
            command_identity=identity, command_kind=command_kind,
            command_action_metadata=json.loads(action_metadata),
        )

    def complete_with_clarification(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        completion: PreparedClarificationCompletion, now: datetime,
        clock: Any = None, prepare_wakeup: Any = None,
    ) -> bool:
        """Atomically publish waiting clarification and settle its queue head."""
        if not is_prepared_clarification_completion(completion):
            raise TypeError("completion must be repository-prepared")
        pending = completion.pending
        intent_ref = completion.intent_reference
        context_ref_data = completion.context_reference
        request = completion.request
        context_value = completion.context_value
        if (context_id != pending.context_id or operation_id != pending.initiating_operation_id
                or intent_ref["context_id"] != context_id
                or intent_ref["operation_id"] != operation_id):
            raise ValueError("clarification completion scope does not match")
        now = _aware(now)
        context_document = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_document.collection("operations").document(operation_id)
        plan = _build_clarification_publication_plan(
            self.client, context_document=context_document, context_id=context_id,
            operation_id=operation_id, completion=completion, now=now,
        )

        def replay_matches() -> bool:
            snapshot = operation_ref.get()
            if not snapshot.exists:
                return False
            stored = snapshot.to_dict() or {}
            return _operation_has_clarification_completion(
                stored, operation_id=operation_id,
                identity_digest=completion.command_identity,
                input_ref=plan.accepted_input_ref,
                binding=plan.binding, result=plan.result,
            )

        if replay_matches():
            return True
        from core.integrations.firestore.operation_queue import complete_head

        def publish(transaction: Any, next_id: str | None,
                    next_data: dict[str, Any] | None, updated_context: dict[str, Any],
                    operation_data: dict[str, Any]) -> None:
            _publish_clarification_in_transaction(
                self, transaction, context_id=context_id, operation_id=operation_id,
                owner=owner, fence=fence, now=now, plan=plan, completion=completion,
                intent_ref=intent_ref, context_ref=context_ref_data, request=request,
                context_value=context_value, prepare_wakeup=prepare_wakeup,
                next_id=next_id, next_data=next_data,
                updated_context=updated_context, operation_data=operation_data,
            )

        try:
            settled = complete_head(
                self.client, context_document, operation_id=operation_id, owner=owner,
                fence=fence, now=now, after_advance_with_operation=publish, clock=clock,
            )
        except _ClarificationReplayCommitted:
            return replay_matches()
        return settled or replay_matches()

    def get_clarification(
        self, context_id: str, clarification_id: str, *, actor_id: int,
    ) -> PendingClarification | None:
        context_id = _identifier(context_id, "context_id")
        clarification_id = _identifier(clarification_id, "clarification_id")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/clarifications/{clarification_id}",
        )
        snapshot = ref.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return _strict_scoped_record(
            data, context_id=context_id, clarification_id=clarification_id,
            actor_id=actor_id,
        )

    def get_pending_clarification(
        self, context_id: str, *, actor_id: int,
    ) -> PendingClarification | None:
        context_id = _identifier(context_id, "context_id")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
        snapshot = context_ref.get()
        if not snapshot.exists:
            return None
        context = snapshot.to_dict() or {}
        clarification_id = context.get("pending_clarification_id")
        generation = context.get("clarification_generation", 0)
        if clarification_id is None:
            return None
        if (not isinstance(clarification_id, str) or not _IDENTIFIER.fullmatch(clarification_id)
                or type(generation) is not int or generation < 1):
            return None
        clarification = self.get_clarification(
            context_id, clarification_id, actor_id=actor_id,
        )
        if (clarification is None or clarification.state is not ClarificationState.WAITING
                or clarification.generation != generation):
            return None
        return clarification

    def _read_clarification_guard(
        self,
        transaction: Any,
        *,
        context_id: str,
        operation_id: str,
        actor_id: int,
        owner: str,
        fence: int,
        now: datetime,
        clarification_id: str | None = None,
        expected_generation: int | None = None,
        expected_revision: int | None = None,
    ) -> _ClarificationGuard | None:
        """Read accepted operation scope and optional waiting record in a transaction.

        This method performs no writes. A successful result only describes records
        observed under the supplied transaction and does not grant effect authority.
        """
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if transaction is None:
            raise ValueError("clarification guard requires a transaction")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid clarification owner")
        _positive_integer(fence, "fence")
        now = _aware(now)
        if clarification_id is None:
            if expected_generation is not None or expected_revision is not None:
                raise ValueError("invalid clarification expectations")
        else:
            clarification_id = _identifier(clarification_id, "clarification_id")
            _positive_integer(expected_generation, "expected_generation")
            _positive_integer(expected_revision, "expected_revision")

        scope = _accepted_scope(
            self, transaction, context_id=context_id, operation_id=operation_id,
            actor_id=actor_id, owner=owner, fence=fence, now=now,
        )
        if scope is None:
            return None
        context, operation, command = scope

        pointer_id = context.get("pending_clarification_id")
        pointer_generation = context.get("clarification_generation", 0)
        if clarification_id is None:
            if pointer_id is not None:
                return None
            if type(pointer_generation) is not int or pointer_generation < 0:
                return None
            clarification = None
        else:
            clarification = _read_waiting_target(
                self, transaction, context_id=context_id,
                clarification_id=clarification_id, actor_id=actor_id,
                generation=expected_generation, revision=expected_revision,
                pointer_id=pointer_id, pointer_generation=pointer_generation,
            )
            if clarification is None:
                return None
        return _ClarificationGuard(
            context=dict(context), operation=dict(operation),
            accepted_command=dict(command), clarification=clarification,
        )
