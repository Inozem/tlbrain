"""Shared identifiers, validation, serialization and certificate checks for Telegram persistence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from services.connectors.telegram.models import (
    TelegramContext,
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
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



def _receipt_result(saved: Mapping[str, Any], context_id: str) -> "TelegramIntakeResult":
    """Restore the result recorded for a previously accepted update."""
    return TelegramIntakeResult(
        bool(saved["accepted"]), context_id, saved.get("draft_id"),
        saved.get("dialog_number"), saved.get("message_id"),
        saved.get("intake_sequence"), saved.get("notification_id"), saved.get("outbox_id"),
        saved.get("operation_id"), saved.get("queue_position"),
        tuple(saved.get("snapshot_message_ids", ())),
    )

