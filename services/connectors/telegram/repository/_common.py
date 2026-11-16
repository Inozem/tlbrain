"""Shared identifiers, validation, serialization and certificate checks for Telegram persistence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from core.integrations.firestore.input_parts import MAX_PART_BYTES
from services.connectors.telegram.models import (
    TelegramContext,
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
)

SESSIONS = "telegram_sessions"
ROUTING_PROTOCOL = "context-v2"
ROUTING_SCHEMA_VERSION = 1
ROUTING_KINDS = frozenset({"context", "decision", "intent"})
ROUTING_INLINE_CHECKPOINT_BYTES = 4 * 1024
NOTIFICATIONS = "notification_outbox"
NOTIFICATION_KINDS = frozenset({"draft_ack", "receipt_card", "unsupported_forward", "queued", "switch_notice", "progress", "final", "error"})
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


def _routing_checkpoint_path(context_id: str, operation_id: str, kind: str) -> str:
    if not isinstance(kind, str) or kind not in ROUTING_KINDS:
        raise ValueError("routing checkpoint kind is invalid")
    _identifier(context_id, "context_id")
    _identifier(operation_id, "operation_id")
    return f"{SESSIONS}/{context_id}/operations/{operation_id}/routing_checkpoints/{kind}"


def _routing_json_bytes(value: Any) -> bytes:
    """Canonical strict JSON for new routing artifacts (legacy codec is unchanged)."""
    def validate(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("routing checkpoint value must be JSON-safe")
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict:
            if any(not isinstance(key, str) for key in item):
                raise ValueError("routing checkpoint value must be JSON-safe")
            for child in item.values():
                validate(child)
            return
        raise ValueError("routing checkpoint value must be JSON-safe")

    try:
        validate(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("routing checkpoint value must be JSON-safe") from exc


def _routing_reference_data(reference: Any) -> dict[str, Any]:
    """Validate the exact connector-private v2 checkpoint reference shape."""
    expected = {"routing_protocol", "schema_version", "kind", "context_id",
                "operation_id", "actor_id", "conversation_id", "ref", "digest",
                "byte_length"}
    if not isinstance(reference, Mapping) or set(reference) != expected:
        raise ValueError("routing checkpoint reference shape is invalid")
    data = dict(reference)
    if (data.get("routing_protocol") != ROUTING_PROTOCOL
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != ROUTING_SCHEMA_VERSION
            or not isinstance(data.get("kind"), str)
            or data["kind"] not in ROUTING_KINDS
            or not isinstance(data.get("context_id"), str)
            or not isinstance(data.get("operation_id"), str)
            or type(data.get("actor_id")) is not int or data["actor_id"] <= 0
            or not isinstance(data.get("conversation_id"), str)
            or not data["conversation_id"]
            or data.get("ref") != _routing_checkpoint_path(
                data["context_id"], data["operation_id"], data["kind"],
            )
            or not isinstance(data.get("digest"), str)
            or len(data["digest"]) != 64
            or any(ch not in "0123456789abcdef" for ch in data["digest"])
            or type(data.get("byte_length")) is not int or data["byte_length"] <= 0):
        raise ValueError("routing checkpoint reference binding is invalid")
    _identifier(data["context_id"], "context_id")
    _identifier(data["operation_id"], "operation_id")
    return data


def _read_routing_checkpoint_value(client: Any, reference: Any) -> Any:
    """Read and verify the published record and inline or multipart payload envelope."""
    from core.integrations.firestore.input_parts import read_input_parts

    ref = _routing_reference_data(reference)
    row = _document_ref(client, ref["ref"]).get()
    if not row.exists:
        raise ValueError("routing checkpoint record is absent")
    record = row.to_dict() or {}
    if set(record) != set(ref) | {"payload"}:
        raise ValueError("routing checkpoint record shape is invalid")
    stored_ref = _routing_reference_data({key: record.get(key) for key in ref})
    if stored_ref != ref:
        raise ValueError("routing checkpoint record binding is invalid")
    stored = record.get("payload")
    if not isinstance(stored, Mapping) or ("_inline" in stored) == ("_input_ref" in stored):
        raise ValueError("routing checkpoint payload mode is invalid")
    if "_inline" in stored and set(stored) != {"_inline", "sha256", "byte_length"}:
        raise ValueError("routing checkpoint inline payload shape is invalid")
    if "_input_ref" in stored and set(stored) != {
        "_input_ref", "sha256", "byte_length", "max_part_bytes",
    }:
        raise ValueError("routing checkpoint multipart payload shape is invalid")
    if (stored.get("sha256") != ref["digest"]
            or type(stored.get("byte_length")) is not int
            or stored["byte_length"] != ref["byte_length"]):
        raise ValueError("routing checkpoint payload metadata is invalid")
    kind = "routing-" + ref["kind"]
    if "_inline" in stored:
        payload = _routing_json_bytes(stored["_inline"])
        if len(payload) > ROUTING_INLINE_CHECKPOINT_BYTES:
            raise ValueError("routing checkpoint inline payload exceeds threshold")
    else:
        digest = ref["digest"]
        data_operation = "telegram-ledger-" + hashlib.sha256(
            f"{ref['context_id']}\0{ref['operation_id']}\0{kind}\0{digest}".encode()
        ).hexdigest()
        expected_input_ref = f"dialog_operations/{data_operation}/input_parts/manifest"
        if (stored.get("_input_ref") != expected_input_ref
                or type(stored.get("max_part_bytes")) is not int
                or stored.get("max_part_bytes") != MAX_PART_BYTES):
            raise ValueError("routing checkpoint input ownership is invalid")
        payload = read_input_parts(client, expected_input_ref)
        if len(payload) <= ROUTING_INLINE_CHECKPOINT_BYTES:
            raise ValueError("routing checkpoint multipart payload is below threshold")
    if (len(payload) != ref["byte_length"]
            or hashlib.sha256(payload).hexdigest() != ref["digest"]):
        raise ValueError("routing checkpoint payload does not verify")
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("routing checkpoint envelope is corrupt") from exc
    expected_envelope = {key: ref[key] for key in (
        "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
        "actor_id", "conversation_id",
    )}
    if not isinstance(envelope, Mapping) or set(envelope) != set(expected_envelope) | {"value"}:
        raise ValueError("routing checkpoint envelope binding is invalid")
    if (type(envelope.get("routing_protocol")) is not str
            or type(envelope.get("schema_version")) is not int
            or type(envelope.get("kind")) is not str
            or type(envelope.get("context_id")) is not str
            or type(envelope.get("operation_id")) is not str
            or type(envelope.get("actor_id")) is not int
            or type(envelope.get("conversation_id")) is not str
            or any(envelope[key] != expected for key, expected in expected_envelope.items())):
        raise ValueError("routing checkpoint envelope binding is invalid")
    if _routing_json_bytes(envelope) != payload:
        raise ValueError("routing checkpoint envelope is not canonical")
    return envelope["value"]


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
    retry = operation.get("retry_authorization") or {}
    intent_operation_matches = (
        intent.get("operation_id", intent.get("dispatch_operation_id")) == operation_id
        or (
            retry.get("operation_id") == operation_id
            and retry.get("create_gate_operation_id", retry.get("previous_operation_id"))
                == intent.get("dispatch_operation_id")
            and retry.get("context_id") == context_id
            and retry.get("draft_id") == draft_id
            and retry.get("manifest_digest") == manifest_digest
            and retry.get("caller_binding") == operation.get("caller_binding")
            and retry.get("immutable_input_digest") == operation.get("immutable_input_digest")
            and operation.get("origin_binding") == binding
            and operation.get("retry_of_operation_id") == retry.get("previous_operation_id")
            and operation.get("retry_draft_id") == draft_id
        )
    )
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
        and intent_operation_matches
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


def _pending_clarification_clear_patch(
    context: Mapping[str, Any], *, clarification_id: str, generation: int,
) -> dict[str, Any] | None:
    """Return a pointer-only CAS patch when the expected clarification is current."""
    if (not isinstance(clarification_id, str) or not clarification_id
            or type(generation) is not int or generation < 0
            or context.get("pending_clarification_id") != clarification_id
            or type(context.get("clarification_generation")) is not int
            or context.get("clarification_generation") != generation):
        return None
    return {"pending_clarification_id": None}


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
    additional_outbox_ids: tuple[str, ...] = ()
    publish_rejection: bool = False



def _receipt_result(saved: Mapping[str, Any], context_id: str) -> "TelegramIntakeResult":
    """Restore the result recorded for a previously accepted update."""
    return TelegramIntakeResult(
        bool(saved["accepted"]), context_id, saved.get("draft_id"),
        saved.get("dialog_number"), saved.get("message_id"),
        saved.get("intake_sequence"), saved.get("notification_id"), saved.get("outbox_id"),
        saved.get("operation_id"), saved.get("queue_position"),
        tuple(saved.get("snapshot_message_ids", ())),
        tuple(saved.get("additional_outbox_ids", ())),
        bool(saved.get("publish_rejection", False)),
    )
