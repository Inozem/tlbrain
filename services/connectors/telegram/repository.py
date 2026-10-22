"""Transactional persistence for Telegram intake and dialog records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record_in_transaction,
)
from core.integrations.firestore.operation_queue import register_operation_in_transaction
from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from services.connectors.telegram.models import (
    PreparedUtterance,
    TelegramContext,
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
    TelegramTranscriptionState,
    build_context_id,
)


SESSIONS = "telegram_sessions"
NOTIFICATIONS = "notification_outbox"
NOTIFICATION_KINDS = frozenset({"draft_ack", "queued", "switch_notice", "progress", "final", "error"})
NOTIFICATION_MAX_SENDS = 3
NOTIFICATION_ATTEMPT_LEASE = timedelta(seconds=60)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_SOURCE_FIELDS = frozenset({
    "message_id", "original_date", "source_chat_id", "source_message_id", "source_user_id",
    "reliable_author_key", "display_speaker", "text", "caption", "audio_file_id",
    "audio_file_unique_id", "audio_mime_type", "audio_file_size", "transcription_state",
    "transcription_text", "transcription_ref", "transcription_error",
})


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a valid Firestore-safe identifier")
    return value


def _document_ref(client: Any, path: str) -> Any:
    parts = path.split("/")
    ref = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        ref = ref.collection(parts[index]).document(parts[index + 1])
    return ref


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _completion_records_match(
    *, context_id: str, draft_id: str, intent_id: str, operation_id: str,
    manifest_digest: str, telegram_certificate: Mapping[str, Any],
    draft: Mapping[str, Any], manifest: Mapping[str, Any],
    intent: Mapping[str, Any], operation: Mapping[str, Any],
) -> bool:
    """Check shared completion authority against the sealed Telegram draft."""
    binding = {"context_id": context_id, "draft_id": draft_id,
               "manifest_digest": manifest_digest}
    shared = intent.get("completion_certificate") or {}
    authorization = intent.get("cleanup_authorization") or {}
    return (
        draft.get("state") in {TelegramDraftState.PROCESSING.value,
                                TelegramDraftState.COMPLETED.value}
        and draft.get("manifest_digest") == manifest_digest
        and draft.get("manifest_ref") == f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}"
        and draft.get("active_save_operation_id") == operation_id
        and draft.get("immutable_save_intent_id") == intent_id
        and (draft.get("completion_certificate") is None
             or _telegram_certificate_matches(
                 draft.get("completion_certificate"), telegram_certificate,
             ))
        and manifest.get("draft_id") == draft_id
        and manifest.get("save_operation_id") == operation_id
        and manifest.get("manifest_digest") == manifest_digest
        and manifest.get("ordered_message_ids") == draft.get("ordered_message_ids")
        and intent.get("save_intent_id") == intent_id
        and intent.get("operation_id") == operation_id
        and shared.get("save_intent_id") == intent_id
        and shared.get("operation_id") == operation_id
        and shared.get("origin_binding") == binding
        and operation.get("operation_id") == operation_id
        and operation.get("save_intent_id") == intent_id
        and operation.get("state") == "completed"
        and operation.get("doc_id") == shared.get("doc_id")
        and bool(shared.get("certificate_id"))
        and bool(shared.get("doc_id"))
        and authorization == {
            "certificate_id": shared.get("certificate_id"),
            "save_intent_id": intent_id,
            "operation_id": operation_id,
            "origin_binding": binding,
        }
        and operation.get("cleanup_authorization") == authorization
    )


def _cleanup_authority_matches(
    *, context_id: str, draft_id: str, intent_id: str, operation_id: str,
    manifest_digest: str, telegram_certificate: Mapping[str, Any],
    context: Mapping[str, Any], draft: Mapping[str, Any],
    manifest: Mapping[str, Any], intent: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> bool:
    """Check that every authoritative record still identifies this draft save."""
    if not _completion_records_match(
        context_id=context_id, draft_id=draft_id, intent_id=intent_id,
        operation_id=operation_id, manifest_digest=manifest_digest,
        telegram_certificate=telegram_certificate, draft=draft, manifest=manifest,
        intent=intent, operation=operation,
    ):
        return False
    shared = intent.get("completion_certificate") or {}
    return (
        draft.get("state") == TelegramDraftState.COMPLETED.value
        and draft.get("doc_id") == shared.get("doc_id")
        and draft.get("completion_certificate_ref") == f"{SAVE_INTENTS_COLLECTION}/{intent_id}"
        and context.get("last_completion_draft_id") == draft_id
        and _telegram_certificate_matches(
            draft.get("completion_certificate"), telegram_certificate,
        )
    )


def _telegram_certificate_matches(left: Any, right: Mapping[str, Any]) -> bool:
    """Compare local certificates across Firestore's tuple-to-list conversion."""
    if not isinstance(left, Mapping):
        return False
    left_data, right_data = dict(left), dict(right)
    for record in (left_data, right_data):
        binding = record.get("origin_binding")
        if isinstance(binding, (list, tuple)):
            record["origin_binding"] = tuple(binding)
    return left_data == right_data


def _persisted_cleanup_binding_matches(
    *, context_id: str, draft_id: str, intent_id: str, operation_id: str,
    manifest_digest: str, telegram_certificate: Mapping[str, Any],
    authorization: Mapping[str, Any], draft: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    """Validate durable cleanup authority against the saved local certificate and manifest."""
    binding = {"context_id": context_id, "draft_id": draft_id,
               "manifest_digest": manifest_digest}
    stored_certificate = authorization.get("telegram_certificate")
    certificate_id = authorization.get("certificate_id")
    expected_authorization = {
        "certificate_id": certificate_id,
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "origin_binding": binding,
        "telegram_certificate": stored_certificate,
    }
    return (
        isinstance(certificate_id, str) and bool(certificate_id)
        and dict(authorization) == expected_authorization
        and _telegram_certificate_matches(stored_certificate, telegram_certificate)
        and telegram_certificate.get("context_id") == context_id
        and telegram_certificate.get("draft_id") == draft_id
        and telegram_certificate.get("save_intent_id") == intent_id
        and telegram_certificate.get("operation_id") == operation_id
        and telegram_certificate.get("manifest_digest") == manifest_digest
        and tuple(telegram_certificate.get("origin_binding", ()))
        == (context_id, draft_id, manifest_digest)
        and draft.get("draft_id") == draft_id
        and draft.get("state") in {TelegramDraftState.COMPLETED.value,
                                    TelegramDraftState.CLEANED.value}
        and _telegram_certificate_matches(
            draft.get("completion_certificate"), telegram_certificate,
        )
        and draft.get("completion_certificate_ref") == f"{SAVE_INTENTS_COLLECTION}/{intent_id}"
        and draft.get("immutable_save_intent_id") == intent_id
        and draft.get("active_save_operation_id") == operation_id
        and draft.get("manifest_digest") == manifest_digest
        and draft.get("manifest_ref") == f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}"
        and manifest.get("draft_id") == draft_id
        and manifest.get("save_operation_id") == operation_id
        and manifest.get("manifest_digest") == manifest_digest
        and manifest.get("ordered_message_ids") == draft.get("ordered_message_ids")
    )


def _draft_data(draft: TelegramDraft) -> dict[str, Any]:
    data = asdict(draft)
    data["state"] = draft.state.value
    data["ordered_message_ids"] = list(draft.ordered_message_ids)
    data["completed_audio_ids"] = list(draft.completed_audio_ids)
    return data


def _draft_from(data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    result.setdefault("state", TelegramDraftState.COLLECTING.value)
    result.setdefault("ordered_message_ids", [])
    result.setdefault("completed_audio_ids", [])
    result.setdefault("required_audio_count", 0)
    result.setdefault("prepared_count", 0)
    result.setdefault("context_capacity", None)
    result.setdefault("preparation_digest", None)
    return result


def _source_data(message: TelegramSourceMessage) -> dict[str, Any]:
    data = asdict(message)
    data["transcription_state"] = message.transcription_state.value
    return data


def _context_defaults(
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> dict[str, Any]:
    """Return persisted defaults for a newly-created Telegram context."""
    data = asdict(TelegramContext(
        chat_id=chat_id,
        thread_id=thread_id,
        topic_name=topic_name,
        created_at=now,
        updated_at=now,
    ))
    data["next_intake_sequence"] = 1
    return data


@dataclass(slots=True)
class _ForwardContext:
    data: dict[str, Any]
    active_ref: Any | None
    active_draft_id: str | None
    draft_data: dict[str, Any]
    busy: bool


@dataclass(slots=True)
class _ForwardDraft:
    is_new: bool
    ref: Any
    draft_id: str
    dialog_number: int
    conversation_id: str
    data: dict[str, Any]
    dialog_ref: Any | None = None
    dialog_data: dict[str, Any] | None = None


@dataclass(slots=True)
class _ForwardPreparation:
    source: TelegramSourceMessage
    message_ref: Any
    notification_ref: Any | None
    notification: dict[str, Any] | None
    task: Any | None
    notification_id: str | None
    outbox_id: str | None


@dataclass(frozen=True, slots=True)
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
    draft_ref: Any | None
    draft_data: dict[str, Any]
    message_ids: tuple[int, ...]
    cutoff_sequence: int
    manifest_ref: str | None
    manifest_doc_ref: Any | None
    manifest_digest: str | None


@dataclass(slots=True)
class _CommandAcceptanceState:
    context: dict[str, Any]
    snapshot: _CommandSnapshot
    command_data: dict[str, Any]


@dataclass(slots=True)
class _AckFailureState:
    context_id: str
    draft_id: str
    context_ref: Any
    draft_ref: Any
    context: dict[str, Any]
    draft: dict[str, Any]


@dataclass(slots=True)
class _AckFailureLinkage:
    parent_id: Any
    save_operation_id: Any
    parent_ref: Any | None = None
    parent: dict[str, Any] | None = None
    next_data: dict[str, Any] | None = None
    save_ref: Any | None = None
    save_operation: dict[str, Any] | None = None
    verified: bool = False


def _receipt_result(saved: Mapping[str, Any], context_id: str) -> "TelegramIntakeResult":
    """Restore the result recorded for a previously accepted update."""
    return TelegramIntakeResult(
        bool(saved["accepted"]), context_id, saved.get("draft_id"),
        saved.get("dialog_number"), saved.get("message_id"),
        saved.get("intake_sequence"), saved.get("notification_id"), saved.get("outbox_id"),
        saved.get("operation_id"), saved.get("queue_position"),
        tuple(saved.get("snapshot_message_ids", ())),
    )


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
                {"kind": command.kind, "text": command.text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _load_command_snapshot(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    context: Mapping[str, Any],
    operation_id: str,
    command: Any,
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
    manifest_ref = (
        f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}"
        if command.is_save and draft_id and message_ids else None
    )
    manifest_digest = None
    manifest_doc_ref = None
    if manifest_ref is not None:
        manifest_digest = hashlib.sha256(json.dumps(
            {"draft_id": draft_id, "message_ids": list(message_ids)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        manifest_doc_ref = _document_ref(client, manifest_ref)
        if manifest_doc_ref.get(transaction=transaction).exists:
            raise RuntimeError("save manifest already exists without its update receipt")
    return _CommandSnapshot(
        draft_id=draft_id,
        draft_ref=draft_ref,
        draft_data=draft_data,
        message_ids=message_ids,
        cutoff_sequence=cutoff_sequence,
        manifest_ref=manifest_ref,
        manifest_doc_ref=manifest_doc_ref,
        manifest_digest=manifest_digest,
    )


def _command_data(
    *,
    update_id: int,
    context_id: str,
    user_id: int,
    operation_id: str,
    command: Any,
    snapshot: _CommandSnapshot,
    now: datetime,
) -> dict[str, Any]:
    """Create the command candidate record without copying draft content."""
    return {
        "update_id": update_id,
        "context_id": context_id,
        "user_id": user_id,
        "kind": command.kind,
        "candidate": command.text,
        "draft_id": snapshot.draft_id,
        "cutoff_message_ids": list(snapshot.message_ids),
        "cutoff_intake_sequence": snapshot.cutoff_sequence,
        "operation_id": operation_id,
        "created_at": now,
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
        operation_id=identity.operation_id,
        command=command,
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
        ),
    )


def _register_command_operation(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    identity: _CommandIdentity,
    now: datetime,
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

    return register_operation_in_transaction(
        client,
        transaction,
        context_ref,
        identity.operation_id,
        identity_digest=identity.identity_digest,
        input_ref=identity.input_ref,
        now=now,
        before_write=create_command_outbox,
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


def _seal_command_draft(
    transaction: Any,
    *,
    snapshot: _CommandSnapshot,
    context: dict[str, Any],
    operation_id: str,
    now: datetime,
) -> None:
    """Seal a recognized save's exact source set and detach its collecting pointer."""
    if snapshot.draft_ref is None:
        return
    snapshot.draft_data.update({
        "state": TelegramDraftState.PROCESSING.value,
        "sealed_at": now,
        "ordered_message_ids": list(snapshot.message_ids),
        "manifest_ref": snapshot.manifest_ref,
        "manifest_digest": snapshot.manifest_digest,
        "active_save_operation_id": operation_id,
        "immutable_save_intent_id": operation_id,
    })
    context.update({"active_draft_id": None, "draft_status": None})
    if snapshot.manifest_doc_ref is not None:
        transaction.set(snapshot.manifest_doc_ref, {
            "draft_id": snapshot.draft_id,
            "ordered_message_ids": list(snapshot.message_ids),
            "manifest_digest": snapshot.manifest_digest,
            "save_operation_id": operation_id,
            "sealed_at": now,
        })
    transaction.set(snapshot.draft_ref, snapshot.draft_data)


def _persist_command_acceptance(
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    command: Any,
    state: _CommandAcceptanceState,
    operation: Any,
    now: datetime,
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
        "snapshot_message_ids": list(snapshot.message_ids) if command.is_save else [],
        "created_at": now,
    }
    transaction.set(identity.command_ref, state.command_data)
    transaction.set(identity.context_ref, state.context)
    transaction.set(identity.receipt_ref, receipt)
    return TelegramIntakeResult(
        True,
        identity.context_id,
        snapshot.draft_id,
        snapshot.draft_data.get("dialog_number"),
        outbox_id=identity.outbox_id,
        operation_id=identity.operation_id,
        queue_position=operation.sequence,
        snapshot_message_ids=snapshot.message_ids if command.is_save else (),
    )


def _load_forward_context(
    client: Any,
    transaction: Any,
    context_ref: Any,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> _ForwardContext:
    """Load context and active draft state before the transaction writes."""
    snapshot = context_ref.get(transaction=transaction)
    context = snapshot.to_dict() or {}
    for field_name, value in _context_defaults(chat_id, thread_id, topic_name, now).items():
        context.setdefault(field_name, value)
    active_draft_id = context.get("active_draft_id")
    active_ref = (
        _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{active_draft_id}")
        if active_draft_id else None
    )
    active_snap = active_ref.get(transaction=transaction) if active_ref is not None else None
    draft_data = (active_snap.to_dict() or {}) if active_snap is not None and active_snap.exists else {}
    if active_draft_id and draft_data.get("state") != TelegramDraftState.COLLECTING.value:
        active_draft_id, active_ref, draft_data = None, None, {}
    return _ForwardContext(
        context, active_ref, active_draft_id, draft_data,
        bool(context.get("reserved_operation_id") or context.get("active_operation_id")),
    )


def _persist_busy_receipt(
    transaction: Any,
    receipt_ref: Any,
    context_ref: Any,
    context: dict[str, Any],
    *,
    update_id: int,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> "TelegramIntakeResult":
    """Record a rejected update while a context operation holds the lock."""
    receipt = {"update_id": update_id, "context_id": context_id, "accepted": False,
               "draft_id": None, "dialog_number": None, "message_id": None,
               "intake_sequence": None, "notification_id": None, "outbox_id": None,
               "created_at": now}
    context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    transaction.set(receipt_ref, receipt)
    transaction.set(context_ref, context)
    return TelegramIntakeResult(False, context_id)


def _select_forward_draft(
    client: Any,
    transaction: Any,
    context: dict[str, Any],
    loaded: _ForwardContext,
    *,
    context_id: str,
    update_id: int,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> _ForwardDraft:
    """Select a collecting draft or allocate a new dialog and draft."""
    if loaded.active_ref is not None:
        draft_data = loaded.draft_data
        draft_id = loaded.active_draft_id
        dialog_number = _integer(draft_data.get("dialog_number"), "dialog_number", minimum=1)
        conversation_id = str(draft_data.get("conversation_id", f"{context_id}-{dialog_number}"))
        return _ForwardDraft(False, loaded.active_ref, draft_id, dialog_number,
                             conversation_id, draft_data)

    dialog_number = _integer(context.get("next_dialog_number", 1), "next_dialog_number", minimum=1)
    draft_id = "draft-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
    conversation_id = f"{context_id}-{dialog_number}"
    draft_ref = _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    dialog_ref = _document_ref(client, f"{SESSIONS}/{context_id}/dialogs/{dialog_number}")
    if dialog_ref.get(transaction=transaction).exists:
        raise RuntimeError("dialog allocator points at an existing dialog")
    draft_data = _draft_data(TelegramDraft(
        draft_id, dialog_number, TelegramDraftState.COLLECTING, conversation_id, now,
    ))
    context.update({"chat_id": chat_id, "thread_id": thread_id,
                    "active_draft_id": draft_id, "draft_status": "collecting",
                    "next_dialog_number": dialog_number + 1, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    dialog_data = {"dialog_number": dialog_number, "conversation_id": conversation_id,
                   "draft_id": draft_id, "doc_id": None,
                   "created_at": now, "updated_at": now}
    return _ForwardDraft(True, draft_ref, draft_id, dialog_number,
                         conversation_id, draft_data, dialog_ref, dialog_data)


def _build_forward_source(
    message: Mapping[str, object],
    *,
    message_id: int,
    intake_sequence: int,
    transcription_state: TelegramTranscriptionState,
    now: datetime,
    original_date: datetime | None,
) -> TelegramSourceMessage:
    """Build the durable normalized source-message record."""
    return TelegramSourceMessage(
        message_id=message_id, intake_sequence=intake_sequence, received_at=now,
        original_date=original_date,
        source_chat_id=message.get("source_chat_id"),
        source_message_id=message.get("source_message_id"),
        source_user_id=message.get("source_user_id"),
        reliable_author_key=message.get("reliable_author_key"),
        display_speaker=message.get("display_speaker"), text=message.get("text"),
        caption=message.get("caption"), audio_file_id=message.get("audio_file_id"),
        audio_file_unique_id=message.get("audio_file_unique_id"),
        audio_mime_type=message.get("audio_mime_type"),
        audio_file_size=message.get("audio_file_size"),
        transcription_state=transcription_state,
        transcription_text=message.get("transcription_text"),
        transcription_ref=message.get("transcription_ref"),
        transcription_error=message.get("transcription_error"),
    )


def _prepare_forward_persistence(
    client: Any,
    transaction: Any,
    draft: _ForwardDraft,
    message: Mapping[str, object],
    *,
    context_id: str,
    delivery_chat_id: int,
    delivery_thread_id: int | None,
    message_id: int,
    intake_sequence: int,
    transcription_state: TelegramTranscriptionState,
    original_date: datetime | None,
    now: datetime,
    notification_id: str,
    outbox_id: str,
) -> _ForwardPreparation:
    """Read message and delivery identities before creating any transaction writes."""
    source = _build_forward_source(
        message, message_id=message_id, intake_sequence=intake_sequence,
        transcription_state=transcription_state, now=now,
        original_date=original_date,
    )
    message_ref = _document_ref(
        client, f"{SESSIONS}/{context_id}/drafts/{draft.draft_id}/messages/{message_id}",
    )
    if message_ref.get(transaction=transaction).exists:
        raise ValueError("message ID already exists in the active draft")

    if not draft.is_new:
        return _ForwardPreparation(source, message_ref, None, None, None, None, None)

    notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
    if notification_ref.get(transaction=transaction).exists:
        raise RuntimeError("notification ID already exists without its update receipt")
    task = create_outbox_record_in_transaction(
        client, transaction, outbox_id,
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
        schedule_at=now, now=now, context_id=context_id,
        notification_id=notification_id,
    )
    notification = {
        "notification_id": notification_id, "context_id": context_id,
        "kind": "draft_ack",
        "body": f"\u0414\u0438\u0430\u043b\u043e\u0433 \u2116{draft.dialog_number}: \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043f\u0440\u0438\u043d\u044f\u0442\u043e",
        "chat_id": delivery_chat_id, "thread_id": delivery_thread_id or None,
        "draft_id": draft.draft_id, "dialog_number": draft.dialog_number,
        "operation_id": None, "desired_status_revision": 1,
        "message_id": None, "state": "pending", "revision": 1,
        "send_attempts_started": 0, "edit_attempts_started": 0, "unknown_send_outcome": False,
        "in_flight_attempt_id": None,
        "attempt_expires_at": None, "last_outcome": None, "retry_not_before": None,
        "terminal_failure": None, "last_error": None, "delivered_at": None, "failed_at": None,
        "created_at": now, "updated_at": now,
    }
    return _ForwardPreparation(
        source, message_ref, notification_ref, notification, task,
        notification_id, outbox_id,
    )


def _persist_accepted_forward(
    transaction: Any,
    *,
    receipt_ref: Any,
    context_ref: Any,
    message_ref: Any,
    notification_ref: Any | None,
    draft: _ForwardDraft,
    context: dict[str, Any],
    source: TelegramSourceMessage,
    update_id: int,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    intake_sequence: int,
    message_id: int,
    now: datetime,
    notification_id: str | None,
    outbox_id: str | None,
    notification: dict[str, Any] | None,
    task: Any | None,
) -> "TelegramIntakeResult":
    """Persist an accepted forward and its receipt atomically."""
    if draft.is_new:
        transaction.set(draft.dialog_ref, draft.dialog_data)
        transaction.set(draft.ref, draft.data)
        transaction.set(notification_ref, notification)
    transaction.set(message_ref, _source_data(source))
    draft.data["ordered_message_ids"] = list(draft.data.get("ordered_message_ids", [])) + [message_id]
    transaction.set(draft.ref, draft.data)
    context.update({"chat_id": chat_id, "thread_id": thread_id,
                    "active_draft_id": draft.draft_id, "draft_status": "collecting",
                    "next_intake_sequence": intake_sequence + 1, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    transaction.set(context_ref, context)
    receipt = {"update_id": update_id, "context_id": context_id, "accepted": True,
               "draft_id": draft.draft_id, "dialog_number": draft.dialog_number,
               "message_id": message_id, "intake_sequence": intake_sequence,
               "notification_id": notification_id,
               "outbox_id": task.outbox_id if task is not None else None,
               "created_at": now}
    transaction.set(receipt_ref, receipt)
    return TelegramIntakeResult(
        True, context_id, draft.draft_id, draft.dialog_number, message_id,
        intake_sequence, notification_id, task.outbox_id if task is not None else None,
    )


@dataclass(frozen=True, slots=True)
class TelegramIntakeResult:
    accepted: bool
    context_id: str
    draft_id: str | None = None
    dialog_number: int | None = None
    message_id: int | None = None
    intake_sequence: int | None = None
    notification_id: str | None = None
    outbox_id: str | None = None
    operation_id: str | None = None
    queue_position: int | None = None
    snapshot_message_ids: tuple[int, ...] = ()


class TelegramRepository:
    """Own Firestore transactions for durable Telegram intake state."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client if client is not None else get_client()

    def get_notification(self, notification_id: str) -> dict[str, Any] | None:
        """Load a server-authored notification ledger row by stable ID."""
        notification_id = _identifier(notification_id, "notification_id")
        snapshot = self.client.collection(NOTIFICATIONS).document(notification_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def revise_notification(
        self, notification_id: str, *, body: str, now: datetime,
    ) -> dict[str, Any]:
        """Persist a new server-authored body revision and fence older attempts."""
        notification_id = _identifier(notification_id, "notification_id")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                raise ValueError("notification does not exist")
            data = snap.to_dict() or {}
            if data.get("state") in {"failed", "delivery_unknown"}:
                raise ValueError("terminal notification cannot be revised")
            if data.get("kind") == "draft_ack" and data.get("message_id") is not None:
                raise ValueError("delivered draft acknowledgement cannot be revised")
            new_revision = _integer(data.get("revision", 1), "revision", minimum=1) + 1
            updates = {"body": body, "revision": new_revision,
                       "desired_status_revision": new_revision,
                "state": "pending", "in_flight_attempt_id": None,
                       "attempt_expires_at": None, "retry_not_before": None,
                       "delivered_at": None, "last_error": None, "updated_at": now}
            transaction.update(ref, updates)
            return {**data, **updates}

        return _txn(self.client.transaction(max_attempts=5))

    def create_notification(
        self, notification_id: str, *, context_id: str, kind: str, body: str, now: datetime,
        dialog_number: int | None = None, operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an explicit notification using only the persisted delivery route."""
        notification_id, context_id = _identifier(notification_id, "notification_id"), _identifier(context_id, "context_id")
        if kind not in NOTIFICATION_KINDS:
            raise ValueError("kind is not an allowed notification kind")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        if dialog_number is not None:
            dialog_number = _integer(dialog_number, "dialog_number", minimum=1)
        if operation_id is not None:
            operation_id = _identifier(operation_id, "operation_id")
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        notification_ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            context_snap = context_ref.get(transaction=transaction)
            if not context_snap.exists:
                raise ValueError("notification context does not exist")
            context = context_snap.to_dict() or {}
            chat_id = context.get("chat_id")
            if type(chat_id) is not int:
                raise ValueError("stored chat_id is invalid")
            thread_id = context.get("thread_id")
            if thread_id is not None:
                thread_id = None if thread_id == 0 else _integer(thread_id, "thread_id", minimum=1)
            snap = notification_ref.get(transaction=transaction)
            if snap.exists:
                stored = snap.to_dict() or {}
                identity = (stored.get("context_id"), stored.get("kind"), stored.get("body"), stored.get("chat_id"), stored.get("thread_id"), stored.get("dialog_number"), stored.get("operation_id"))
                if identity != (context_id, kind, body, chat_id, thread_id, dialog_number, operation_id):
                    raise ValueError("notification ID already belongs to a different identity")
                return stored
            row = {"notification_id": notification_id, "context_id": context_id, "kind": kind,
                   "body": body, "chat_id": chat_id, "thread_id": thread_id, "message_id": None,
                   "dialog_number": dialog_number, "operation_id": operation_id,
                   "state": "pending", "revision": 1, "desired_status_revision": 1, "send_attempts_started": 0,
                   "edit_attempts_started": 0, "unknown_send_outcome": False,
                   "in_flight_attempt_id": None, "attempt_expires_at": None, "last_outcome": None,
                   "retry_not_before": None, "terminal_failure": None, "last_error": None,
                   "delivered_at": None, "failed_at": None, "created_at": now, "updated_at": now}
            transaction.set(notification_ref, row)
            return row

        return _txn(self.client.transaction(max_attempts=5))

    def reserve_notification_attempt(
        self, notification_id: str, *, attempt_id: str, now: datetime,
    ) -> dict[str, Any] | None:
        """Reserve one bounded send or edit attempt before contacting Telegram."""
        notification_id = _identifier(notification_id, "notification_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if data.get("state") in {"delivered", "failed", "delivery_unknown"}:
                return {**data, "reserved": False}
            if data.get("retry_not_before") and data["retry_not_before"] > now:
                return {**data, "reserved": False}
            inflight = data.get("in_flight_attempt_id")
            expiry = data.get("attempt_expires_at")
            if inflight and isinstance(expiry, datetime) and expiry > now:
                return {**data, "reserved": False}
            abandoned_send = bool(inflight and data.get("message_id") is None)
            if abandoned_send:
                data["unknown_send_outcome"] = True
                data["last_outcome"] = "unknown"
            is_send = data.get("message_id") is None
            attempts = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
            if is_send and attempts >= NOTIFICATION_MAX_SENDS:
                terminal = "delivery_unknown" if data.get("unknown_send_outcome") else "failed"
                updates = {"state": terminal, "terminal_failure": terminal, "updated_at": now,
                           "last_error": "attempts_exhausted", "failed_at": now,
                           "in_flight_attempt_id": None, "attempt_expires_at": None}
                transaction.update(ref, updates)
                return {**data, **updates, "reserved": False}
            revision = _integer(data.get("revision", 1), "revision", minimum=1)
            updates = {"state": "sending" if is_send else "editing", "in_flight_attempt_id": attempt_id,
                       "attempt_revision": revision, "attempt_expires_at": now + NOTIFICATION_ATTEMPT_LEASE,
                       "updated_at": now}
            if abandoned_send:
                updates["last_outcome"] = "unknown"
                updates["unknown_send_outcome"] = True
            if is_send:
                updates["send_attempts_started"] = attempts + 1
            else:
                edit_attempts = _integer(data.get("edit_attempts_started", 0), "edit_attempts_started")
                if edit_attempts >= NOTIFICATION_MAX_SENDS:
                    updates.update({"state": "failed", "terminal_failure": "failed",
                                    "last_error": "attempts_exhausted", "failed_at": now,
                                    "in_flight_attempt_id": None, "attempt_expires_at": None})
                    transaction.update(ref, updates)
                    return {**data, **updates, "reserved": False}
                updates["edit_attempts_started"] = edit_attempts + 1
            transaction.update(ref, updates)
            return {**data, **updates, "reserved": True, "attempt_id": attempt_id,
                    "attempt_revision": revision, "is_send": is_send}

        return _txn(self.client.transaction(max_attempts=5))

    def settle_notification_attempt(
        self, notification_id: str, *, attempt_id: str, revision: int, outcome: str,
        now: datetime, message_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Settle only the current attempt and current notification revision."""
        notification_id = _identifier(notification_id, "notification_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        revision = _integer(revision, "revision", minimum=1)
        if outcome not in {"sent", "edited", "unknown", "rejected"}:
            raise ValueError("outcome is invalid")
        if outcome == "sent":
            message_id = _integer(message_id, "message_id", minimum=1)
        elif message_id is not None:
            raise ValueError("message_id is only valid for a sent outcome")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if (data.get("in_flight_attempt_id") != attempt_id
                    or data.get("attempt_revision") != revision
                    or data.get("revision") != revision
                    or data.get("state") in {"delivered", "failed", "delivery_unknown"}):
                return {**data, "settled": False}
            updates: dict[str, Any] = {"in_flight_attempt_id": None, "attempt_expires_at": None,
                                       "last_outcome": outcome, "updated_at": now}
            if outcome in {"sent", "edited"}:
                if outcome == "sent":
                    updates["message_id"] = message_id
                updates.update({"state": "delivered", "terminal_failure": None, "retry_not_before": None,
                                "last_error": None, "delivered_at": now, "failed_at": None})
            else:
                sends = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
                exhausted = (data.get("message_id") is None and sends >= NOTIFICATION_MAX_SENDS) or (
                    data.get("message_id") is not None
                    and _integer(data.get("edit_attempts_started", 0), "edit_attempts_started") >= NOTIFICATION_MAX_SENDS
                )
                if exhausted:
                    terminal = ("delivery_unknown" if data.get("unknown_send_outcome") or outcome == "unknown"
                                else "failed") if data.get("message_id") is None else "failed"
                    updates.update({"state": terminal, "terminal_failure": terminal, "retry_not_before": None,
                                    "last_error": outcome, "failed_at": now})
                else:
                    delay = min(2 ** max(0, sends - 1), 60)
                    updates.update({"state": "retry", "retry_not_before": now + timedelta(seconds=delay),
                                    "last_error": outcome})
                    if outcome == "unknown" and data.get("message_id") is None:
                        updates["unknown_send_outcome"] = True
            result = {**data, **updates, "settled": True}
            if (result.get("state") in {"failed", "delivery_unknown"}
                    and data.get("kind") == "draft_ack" and data.get("message_id") is None):
                action = self._settle_terminal_ack_failure(transaction, result, now)
                if action is not None:
                    result["ack_failure_action"] = action
                    updates["ack_failure_action"] = action
            transaction.update(ref, updates)
            return result

        return _txn(self.client.transaction(max_attempts=5))

    def _settle_terminal_ack_failure(
        self, transaction: Any, notification: dict[str, Any], now: datetime,
    ) -> dict[str, Any] | None:
        """Fail only a sealed draft waiting on its first acknowledgement barrier."""
        state = self._load_ack_failure_state(transaction, notification)
        if state is None:
            return None
        linkage = self._read_ack_failure_linkage(transaction, state)
        return self._write_ack_failure(transaction, state, linkage, now)

    def _load_ack_failure_state(
        self, transaction: Any, notification: dict[str, Any],
    ) -> _AckFailureState | None:
        """Load an eligible sealed draft and its context before any writes."""
        context_id, draft_id = notification.get("context_id"), notification.get("draft_id")
        if not isinstance(context_id, str) or not isinstance(draft_id, str):
            return None
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        context_snap, draft_snap = context_ref.get(transaction=transaction), draft_ref.get(transaction=transaction)
        if not draft_snap.exists:
            return None
        draft = draft_snap.to_dict() or {}
        if not draft.get("sealed_at") or not draft.get("manifest_ref") or not draft.get("manifest_digest"):
            return None
        if draft.get("state") in {TelegramDraftState.COMPLETED.value, TelegramDraftState.CLEANED.value}:
            return None
        context = context_snap.to_dict() or {}
        return _AckFailureState(context_id, draft_id, context_ref, draft_ref, context, draft)

    def _read_ack_failure_linkage(
        self, transaction: Any, state: _AckFailureState,
    ) -> _AckFailureLinkage:
        """Verify queue and save-operation links before the transaction writes."""
        context_id, draft_id = state.context_id, state.draft_id
        context, draft = state.context, state.draft
        save_operation_id = draft.get("active_save_operation_id")
        parent_id = context.get("reserved_operation_id")
        linkage = _AckFailureLinkage(parent_id, save_operation_id)
        if isinstance(parent_id, str) and parent_id:
            linkage.parent_ref = _document_ref(
                self.client, f"{SESSIONS}/{context_id}/operations/{parent_id}",
            )
            parent_snap = linkage.parent_ref.get(transaction=transaction)
            if parent_snap.exists:
                linkage.parent = parent_snap.to_dict() or {}
                linkage.verified = (
                    linkage.parent.get("operation_id") == parent_id
                    and linkage.parent.get("target_draft_id") == draft_id
                    and linkage.parent.get("save_operation_id") == save_operation_id
                    and bool(save_operation_id)
                    and context.get("head_sequence") == linkage.parent.get("sequence")
                    and linkage.parent.get("state") not in {"completed", "failed"}
                )
                next_id = linkage.parent.get("next_operation_id")
                if linkage.verified and next_id:
                    next_ref = _document_ref(
                        self.client, f"{SESSIONS}/{context_id}/operations/{next_id}",
                    )
                    next_snap = next_ref.get(transaction=transaction)
                    if next_snap.exists:
                        linkage.next_data = next_snap.to_dict() or {}
                        if (linkage.next_data.get("operation_id") != next_id
                                or linkage.next_data.get("state") in {"completed", "failed"}):
                            linkage.verified = False
                    else:
                        linkage.verified = False
                if linkage.verified:
                    linkage.save_ref = self.client.collection("dialog_operations").document(save_operation_id)
                    save_snap = linkage.save_ref.get(transaction=transaction)
                    if save_snap.exists:
                        linkage.save_operation = save_snap.to_dict() or {}
                        if linkage.save_operation.get("operation_id") != save_operation_id:
                            linkage.verified = False
        return linkage

    def _write_ack_failure(
        self,
        transaction: Any,
        state: _AckFailureState,
        linkage: _AckFailureLinkage,
        now: datetime,
    ) -> dict[str, Any]:
        """Fail the draft and release its queue only when every link verified."""
        context, draft = state.context, state.draft
        draft.update({"state": TelegramDraftState.FAILED.value, "ack_failure": "notification_delivery_failed",
                      "updated_at": now})
        if context.get("active_draft_id") == state.draft_id:
            context.update({"active_draft_id": None, "draft_status": None})
        transaction.set(state.draft_ref, draft)
        action = {"context_id": state.context_id, "draft_id": state.draft_id,
                  "dialog_number": draft.get("dialog_number"),
                  "operation_id": linkage.save_operation_id, "parent_operation_id": linkage.parent_id,
                  "queue_release_required": not linkage.verified}
        if linkage.verified and linkage.parent_ref is not None and linkage.parent is not None:
            linkage.parent.update({"state": "failed", "failure_code": "ack_delivery_failed", "finished_at": now})
            context.update({"active_operation_id": None, "lock_owner": None, "lock_expires_at": None})
            if linkage.next_data is not None:
                context.update({"reserved_operation_id": linkage.next_data.get("operation_id"),
                                "head_sequence": linkage.next_data.get("sequence")})
            else:
                context.update({"reserved_operation_id": None, "head_sequence": None,
                                "queue_tail_operation_id": None})
            transaction.set(linkage.parent_ref, linkage.parent)
            if (linkage.save_ref is not None and linkage.save_operation is not None
                    and linkage.save_operation.get("state") not in {"completed", "failed"}):
                linkage.save_operation.update({
                    "state": "failed", "stage": "notification", "finished_at": now,
                    "updated_at": now, "status_updated_at": now,
                    "lease_owner": None, "lease_expires_at": None,
                    "error": {"code": "ack_delivery_failed", "category": "notification",
                              "sanitized_message": "The initial acknowledgement could not be delivered.",
                              "retryable": False, "service": "telegram", "stage": "notification",
                              "occurred_at": now, "failures": 1,
                              "operation_id": linkage.save_operation_id},
                })
                transaction.set(linkage.save_ref, linkage.save_operation)
        transaction.set(state.context_ref, context)
        return action

    def accept_forward(
        self,
        update_id: int,
        *,
        chat_id: int,
        thread_id: int | None = None,
        topic_name: str | None = None,
        message: Mapping[str, object],
        now: datetime,
    ) -> TelegramIntakeResult:
        update_id = _integer(update_id, "update_id")
        context_id = build_context_id(chat_id, thread_id)
        now = _aware(now, "now")
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(message, Mapping) or set(message) - _SOURCE_FIELDS:
            raise ValueError("message must contain only normalized source-message fields")
        message_id = _integer(message.get("message_id"), "message_id", minimum=1)
        original_date = message.get("original_date")
        if original_date is not None:
            original_date = _aware(original_date, "original_date")  # type: ignore[arg-type]
        source_chat_id = message.get("source_chat_id")
        if source_chat_id is not None and (type(source_chat_id) is not int or source_chat_id == 0):
            raise ValueError("source_chat_id must be a non-zero integer")
        for field_name in ("source_message_id", "source_user_id", "audio_file_size"):
            value = message.get(field_name)
            if value is not None:
                _integer(value, field_name, minimum=0 if field_name == "audio_file_size" else 1)
        for field_name in (
            "reliable_author_key", "display_speaker", "text", "caption", "audio_file_id",
            "audio_file_unique_id", "audio_mime_type", "transcription_text",
            "transcription_ref", "transcription_error",
        ):
            value = message.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or None")
        try:
            transcription_state = TelegramTranscriptionState(
                message.get("transcription_state", TelegramTranscriptionState.NOT_REQUIRED)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("transcription_state is invalid") from exc
        notification_id = "tgack-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
        outbox_id = "tgdispatch-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            receipt_snap = receipt_ref.get(transaction=transaction)
            if receipt_snap.exists:
                return _receipt_result(receipt_snap.to_dict() or {}, context_id)
            loaded = _load_forward_context(
                self.client, transaction, context_ref, context_id,
                chat_id, thread_id, topic_name, now,
            )
            if loaded.busy:
                return _persist_busy_receipt(
                    transaction, receipt_ref, context_ref, loaded.data,
                    update_id=update_id, context_id=context_id,
                    chat_id=chat_id, thread_id=thread_id,
                    topic_name=topic_name, now=now,
                )
            draft = _select_forward_draft(
                self.client, transaction, loaded.data, loaded,
                context_id=context_id, update_id=update_id,
                chat_id=chat_id, thread_id=thread_id,
                topic_name=topic_name, now=now,
            )
            intake_sequence = _integer(loaded.data.get("next_intake_sequence", 1), "next_intake_sequence", minimum=1)
            prepared = _prepare_forward_persistence(
                self.client, transaction, draft, message,
                context_id=context_id, message_id=message_id,
                delivery_chat_id=loaded.data["chat_id"],
                delivery_thread_id=loaded.data.get("thread_id"),
                intake_sequence=intake_sequence,
                transcription_state=transcription_state,
                original_date=original_date, now=now,
                notification_id=notification_id, outbox_id=outbox_id,
            )
            return _persist_accepted_forward(
                transaction, receipt_ref=receipt_ref, context_ref=context_ref,
                message_ref=prepared.message_ref,
                notification_ref=prepared.notification_ref,
                draft=draft, context=loaded.data, source=prepared.source,
                update_id=update_id, context_id=context_id,
                chat_id=chat_id, thread_id=thread_id, topic_name=topic_name,
                intake_sequence=intake_sequence, message_id=message_id, now=now,
                notification_id=prepared.notification_id, outbox_id=prepared.outbox_id,
                notification=prepared.notification, task=prepared.task,
            )

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
        now: datetime,
    ) -> TelegramIntakeResult:
        """Coordinate the all-reads-first command reservation and optional save seal."""
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
        )
        operation = _register_command_operation(
            self.client,
            transaction,
            context_ref=identity.context_ref,
            identity=identity,
            now=now,
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
        if command.is_save:
            _seal_command_draft(
                transaction,
                snapshot=state.snapshot,
                context=state.context,
                operation_id=identity.operation_id,
                now=now,
            )
        return _persist_command_acceptance(
            transaction,
            identity=identity,
            update_id=update_id,
            command=command,
            state=state,
            operation=operation,
            now=now,
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
        now: datetime,
    ) -> TelegramIntakeResult:
        """Accept command text and reserve its FIFO position with a save cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        now = _aware(now, "now")
        if thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(command, TelegramCommand):
            raise TypeError("command must be a TelegramCommand")
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
                now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))

    def get_update_receipt(self, context_id: str, update_id: int) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        update_id = _integer(update_id, "update_id")
        snapshot = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}").get()
        return snapshot.to_dict() if snapshot.exists else None

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        snapshot = self.client.collection(SESSIONS).document(context_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def commit_completion(self, certificate: Any) -> bool:
        """Link an immutable save certificate to its Telegram draft transactionally."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(data.get("context_id"), "context_id")
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        manifest_digest = data.get("manifest_digest")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            raise ValueError("manifest_digest is required")
        if tuple(data.get("origin_binding", ())) != (context_id, draft_id, manifest_digest):
            raise ValueError("completion certificate origin binding does not match Telegram draft")
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )
        intent_ref = self.client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
        operation_ref = self.client.collection(OPERATIONS_COLLECTION).document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            draft_snapshot = draft_ref.get(transaction=transaction)
            context_snapshot = context_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            intent_snapshot = intent_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not all((draft_snapshot.exists, context_snapshot.exists, manifest_snapshot.exists,
                        intent_snapshot.exists, operation_snapshot.exists)):
                return False
            draft = draft_snapshot.to_dict() or {}
            context = context_snapshot.to_dict() or {}
            manifest = manifest_snapshot.to_dict() or {}
            intent = intent_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if not _completion_records_match(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, draft=draft, manifest=manifest,
                intent=intent, operation=operation,
            ):
                return False
            shared_certificate = intent["completion_certificate"]
            draft.update({
                "state": TelegramDraftState.COMPLETED.value,
                "completion_certificate": data,
                "completion_certificate_ref": f"{SAVE_INTENTS_COLLECTION}/{intent_id}",
                "doc_id": shared_certificate.get("doc_id"),
            })
            context["last_completion_draft_id"] = draft_id
            context["last_doc_id"] = shared_certificate.get("doc_id")
            transaction.set(draft_ref, draft)
            transaction.set(context_ref, context)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def load_completion_certificate(self, context_id: str) -> dict[str, Any] | None:
        """Load the locally linked certificate for the context's latest completed draft."""
        context_id = _identifier(context_id, "context_id")
        context = self.get_context(context_id) or {}
        draft_id = context.get("last_completion_draft_id")
        if not draft_id:
            return None
        draft = self.get_draft(context_id, _identifier(draft_id, "draft_id")) or {}
        certificate = draft.get("completion_certificate")
        return dict(certificate) if isinstance(certificate, Mapping) else None

    def authorize_cleanup(self, context_id: str, certificate: Any) -> bool:
        """Validate shared completion evidence and durably authorize child deletion."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(context_id, "context_id")
        if data.get("context_id") != context_id:
            return False
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        manifest_digest = data.get("manifest_digest")
        if (not isinstance(manifest_digest, str) or not manifest_digest
                or tuple(data.get("origin_binding", ())) != (context_id, draft_id, manifest_digest)):
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )
        intent_ref = self.client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
        operation_ref = self.client.collection(OPERATIONS_COLLECTION).document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            if not (context_snapshot.exists and draft_snapshot.exists and manifest_snapshot.exists):
                return False
            context = context_snapshot.to_dict() or {}
            draft = draft_snapshot.to_dict() or {}
            manifest = manifest_snapshot.to_dict() or {}
            if draft.get("cleanup_state") in {"authorized", "cleaned"}:
                return _persisted_cleanup_binding_matches(
                    context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                    operation_id=operation_id, manifest_digest=manifest_digest,
                    telegram_certificate=data,
                    authorization=draft.get("cleanup_authorization") or {},
                    draft=draft, manifest=manifest,
                )
            intent_snapshot = intent_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not (intent_snapshot.exists and operation_snapshot.exists):
                return False
            intent = intent_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if not _cleanup_authority_matches(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, context=context, draft=draft,
                manifest=manifest, intent=intent, operation=operation,
            ):
                return False
            shared_certificate = intent["completion_certificate"]
            cert_id = shared_certificate["certificate_id"]
            expected_binding = {"context_id": context_id, "draft_id": draft_id,
                                "manifest_digest": manifest_digest}
            persisted_authorization = {
                "certificate_id": cert_id,
                "save_intent_id": intent_id,
                "operation_id": operation_id,
                "origin_binding": expected_binding,
                "telegram_certificate": data,
            }
            draft["cleanup_state"] = "authorized"
            draft["cleanup_authorization"] = persisted_authorization
            transaction.set(draft_ref, draft)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def list_manifest_pages(
        self, draft_id: str, *, cursor: Any = None, limit: int = 200,
        context_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Read one bounded source-message page for authorized cleanup."""
        if context_id is None:
            raise ValueError("context_id is required for source page listing")
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        limit = min(_integer(limit, "limit", minimum=1), 400)
        collection = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}"
        ).collection("messages")
        query = collection.limit(limit)
        page = list(query.stream())
        rows = [snapshot.to_dict() or {} for snapshot in page]
        next_cursor = page[-1] if len(page) == limit else None
        return rows, next_cursor

    def delete_draft_page(self, draft_id: str, messages: Sequence[Mapping[str, Any]], *, context_id: str) -> None:
        """Delete at most 400 message checkpoints from the authorized draft."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        if len(messages) > 400:
            raise ValueError("cleanup page exceeds the safe write limit")
        draft = self.get_draft(context_id, draft_id)
        if draft is None or draft.get("cleanup_state") != "authorized":
            raise ValueError("draft cleanup has not been authorized")
        batch = self.client.batch()
        for row in messages:
            message_id = _integer(row.get("message_id"), "message_id", minimum=1)
            batch.delete(_document_ref(
                self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}"
            ))
        batch.commit()

    def list_prepared_pages(self, context_id: str, draft_id: str, *, limit: int = 200) -> list[Any]:
        """Read the first bounded page of prepared utterance documents."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        limit = min(_integer(limit, "limit", minimum=1), 400)
        return list(_document_ref(
            self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}"
        ).collection("prepared").limit(limit).stream())

    def delete_prepared_page(self, context_id: str, draft_id: str, page: Sequence[Any]) -> None:
        """Delete one bounded prepared utterance page idempotently."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        if len(page) > 400:
            raise ValueError("cleanup page exceeds the safe write limit")
        draft = self.get_draft(context_id, draft_id)
        if draft is None or draft.get("cleanup_state") != "authorized":
            raise ValueError("draft cleanup has not been authorized")
        expected_prefix = f"{SESSIONS}/{context_id}/drafts/{draft_id}/prepared/"
        batch = self.client.batch()
        for snapshot in page:
            path = getattr(getattr(snapshot, "reference", None), "path", "")
            if not isinstance(path, str) or not path.startswith(expected_prefix):
                raise ValueError("prepared cleanup page contains a foreign document")
            batch.delete(snapshot.reference)
        batch.commit()

    def finalize_cleanup(self, context_id: str, certificate: Any) -> bool:
        """Mark the authorized draft cleaned while preserving newer context state."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(context_id, "context_id")
        if data.get("context_id") != context_id:
            return False
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        manifest_digest = data.get("manifest_digest")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            if not (context_snapshot.exists and draft_snapshot.exists and manifest_snapshot.exists):
                return False
            context = context_snapshot.to_dict() or {}
            draft = draft_snapshot.to_dict() or {}
            authorization = draft.get("cleanup_authorization") or {}
            manifest = manifest_snapshot.to_dict() or {}
            if not _persisted_cleanup_binding_matches(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, authorization=authorization,
                draft=draft, manifest=manifest,
            ):
                return False
            if draft.get("cleanup_state") == "cleaned":
                return True
            if draft.get("cleanup_state") != "authorized":
                return False
            draft["cleanup_state"] = "cleaned"
            draft["state"] = TelegramDraftState.CLEANED.value
            if context.get("last_completion_draft_id") == draft_id:
                context["last_doc_id"] = draft.get("doc_id")
            if context.get("active_draft_id") == draft_id:
                context["active_draft_id"] = None
                context["draft_status"] = None
            transaction.set(draft_ref, draft)
            transaction.set(context_ref, context)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def get_draft(self, context_id: str, draft_id: str) -> dict[str, Any] | None:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        snapshot = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}").get()
        return _draft_from(snapshot.to_dict() or {}) if snapshot.exists else None

    def list_messages(self, context_id: str, draft_id: str) -> list[dict[str, Any]]:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        snapshots = list(ref.collection("messages").stream())
        result = [snapshot.to_dict() or {} for snapshot in snapshots]
        result.sort(key=lambda row: (row["intake_sequence"], row["message_id"]))
        return result

    def checkpoint_transcription(
        self,
        context_id: str,
        draft_id: str,
        message_id: int,
        text: str,
    ) -> None:
        """Persist one completed transcription without changing its source record."""
        context_id = _identifier(context_id, "context_id")
        draft_id = _identifier(draft_id, "draft_id")
        message_id = _integer(message_id, "message_id", minimum=1)
        if not isinstance(text, str) or not text:
            raise ValueError("transcription text must be non-empty text")
        message_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if _integer(source.get("message_id"), "stored message_id", minimum=1) != message_id:
                raise ValueError("source message association does not match")
            if source.get("transcription_state") == TelegramTranscriptionState.COMPLETED.value:
                if source.get("transcription_text") != text:
                    raise ValueError("completed transcription cannot be replaced")
                return
            transaction.update(message_ref, {
                "transcription_state": TelegramTranscriptionState.COMPLETED.value,
                "transcription_text": text,
                "transcription_error": None,
            })

        _txn(self.client.transaction(max_attempts=5))

    def checkpoint_transcription_failure(
        self,
        context_id: str,
        draft_id: str,
        message_id: int,
        error_code: str,
    ) -> None:
        """Persist a bounded safe transcription failure category for one source."""
        context_id = _identifier(context_id, "context_id")
        draft_id = _identifier(draft_id, "draft_id")
        message_id = _integer(message_id, "message_id", minimum=1)
        allowed_errors = {"unsupported_audio", "download_failed", "transcription_failed"}
        if error_code not in allowed_errors or len(error_code) > 48:
            raise ValueError("transcription error code is invalid")
        message_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if _integer(source.get("message_id"), "stored message_id", minimum=1) != message_id:
                raise ValueError("source message association does not match")
            if source.get("transcription_state") == TelegramTranscriptionState.COMPLETED.value:
                return
            transaction.update(message_ref, {
                "transcription_state": TelegramTranscriptionState.FAILED.value,
                "transcription_text": None,
                "transcription_error": error_code[:48],
            })

        _txn(self.client.transaction(max_attempts=5))

    def list_failed_drafts(self, context_id: str) -> list[dict[str, Any]]:
        context_id = _identifier(context_id, "context_id")
        collection = self.client.collection(SESSIONS).document(context_id).collection("drafts")
        snapshots = list(collection.stream())
        return [_draft_from(s.to_dict() or {}) for s in snapshots
                if (s.to_dict() or {}).get("state") == TelegramDraftState.FAILED.value]

    def save_prepared_utterances(
        self, context_id: str, draft_id: str, utterances: Sequence[PreparedUtterance], *,
        required_audio_count: int, completed_audio_ids: Sequence[int], context_capacity: int,
        preparation_digest: str, now: datetime,
    ) -> None:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        now = _aware(now, "now")
        required_audio_count = _integer(required_audio_count, "required_audio_count")
        context_capacity = _integer(context_capacity, "context_capacity", minimum=1)
        if not isinstance(preparation_digest, str) or not preparation_digest:
            raise ValueError("preparation_digest is required")
        ids = tuple(_integer(item, "completed_audio_id", minimum=1) for item in completed_audio_ids)
        rows = tuple(utterances)
        if any(not isinstance(row, PreparedUtterance) for row in rows):
            raise TypeError("utterances must contain PreparedUtterance records")
        if tuple(row.order_index for row in rows) != tuple(range(len(rows))):
            raise ValueError("prepared utterance order_index values must be contiguous from zero")
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        prepared_ref = draft_ref.collection("prepared")

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> None:
            snapshot = draft_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("draft does not exist")
            draft_data = snapshot.to_dict() or {}
            for row in rows:
                ref = prepared_ref.document(str(row.order_index))
                transaction.set(ref, {**asdict(row), "source_message_ids": list(row.source_message_ids)})
            draft_data.update({"required_audio_count": required_audio_count,
                               "completed_audio_ids": list(ids), "prepared_count": len(rows),
                               "context_capacity": context_capacity,
                               "preparation_digest": preparation_digest, "updated_at": now})
            transaction.set(draft_ref, draft_data)

        _txn(self.client.transaction(max_attempts=5))

    def bind_document(
        self, context_id: str, doc_id: str, conversation_id: str, *,
        metadata: Mapping[str, Any], now: datetime,
    ) -> int:
        context_id = _identifier(context_id, "context_id")
        if not isinstance(doc_id, str) or not doc_id or "/" in doc_id:
            raise ValueError("doc_id must be a non-empty document identifier")
        conversation_id = _identifier(conversation_id, "conversation_id")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        now = _aware(now, "now")
        key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        binding_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/document_bindings/{key}")
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> int:
            binding_snap = binding_ref.get(transaction=transaction)
            if binding_snap.exists:
                binding = binding_snap.to_dict() or {}
                if binding.get("doc_id") != doc_id:
                    raise RuntimeError("document binding hash collision")
                return _integer(binding["dialog_number"], "dialog_number", minimum=1)
            context_snap = context_ref.get(transaction=transaction)
            context = context_snap.to_dict() or {}
            number = _integer(context.get("next_dialog_number", 1), "next_dialog_number", minimum=1)
            dialog_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/dialogs/{number}")
            existing = dialog_ref.get(transaction=transaction)
            if existing.exists:
                raise RuntimeError("dialog allocator points at an existing dialog")
            dialog = {"dialog_number": number, "conversation_id": conversation_id,
                      "doc_id": doc_id, "metadata": dict(metadata), "created_at": now,
                      "updated_at": now}
            transaction.set(dialog_ref, dialog)
            transaction.set(binding_ref, {"doc_id": doc_id, "dialog_number": number,
                                           "conversation_id": conversation_id,
                                           "created_at": now, "updated_at": now})
            context.update({"next_dialog_number": number + 1, "last_doc_id": doc_id,
                            "updated_at": now})
            transaction.set(context_ref, context)
            return number

        return _txn(self.client.transaction(max_attempts=5))


__all__ = ["TelegramIntakeResult", "TelegramRepository"]
