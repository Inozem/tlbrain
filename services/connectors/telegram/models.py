"""Immutable domain contracts for Telegram connector workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import copy
import hashlib
import re
from types import MappingProxyType
from typing import Any, Mapping


_CLARIFICATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLARIFICATION_PROTOCOL = "clarification-v1"
_CLARIFICATION_ACTIONS = frozenset({
    "move_transcript", "create_folder", "create_then_move", "pure_switch",
    "search", "substantive", "save", "retry", "annotation",
})
_CLARIFICATION_SLOTS = frozenset({
    "document", "destination", "parent", "folder_name", "explicit_conflict",
})
_PARAMETER_IDS = frozenset({"doc_id", "destination_id", "parent_id"})
_PARAMETER_KEYS = _PARAMETER_IDS | frozenset({"name", "folder_path", "parent_path"})


def _clarification_error(field: str) -> ValueError:
    return ValueError(f"invalid clarification {field}")


def _clarification_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _CLARIFICATION_ID.fullmatch(value):
        raise _clarification_error(field)
    return value


def _clarification_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _clarification_error(field)
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _normalize_clarification_parameters(
    action: str, state: ClarificationState, parameters: Mapping[str, Any],
    missing_slots: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(parameters, Mapping) or any(key not in _PARAMETER_KEYS for key in parameters):
        raise _clarification_error("resolved_parameters")
    safe: dict[str, Any] = {}
    slots = missing_slots
    missing_folder_name = False
    for key, value in parameters.items():
        if key in _PARAMETER_IDS:
            if not isinstance(value, str) or not value.strip():
                raise _clarification_error("resolved_parameters")
        elif key == "name":
            if value is not None and not isinstance(value, str):
                raise _clarification_error("resolved_parameters")
            if value is None or not value.strip():
                if (state is not ClarificationState.WAITING
                        or action not in {"create_folder", "create_then_move"}):
                    raise _clarification_error("resolved_parameters")
                missing_folder_name = True
                continue
        elif key in {"folder_path", "parent_path"}:
            if (not isinstance(value, (tuple, list))
                    or any(not isinstance(part, str) or not part.strip() for part in value)):
                raise _clarification_error("resolved_parameters")
            value = tuple(value)
        safe[key] = value
    if action == "create_folder" and ("doc_id" in safe or "document" in slots):
        raise _clarification_error("resolved_parameters")
    if state is ClarificationState.WAITING:
        if (action in {"create_folder", "create_then_move"}
                and (missing_folder_name or "name" not in safe)):
            if "folder_name" not in slots:
                slots += ("folder_name",)
        if not slots:
            raise _clarification_error("missing_slots")
        slot_to_key = {
            "document": "doc_id", "destination": "destination_id",
            "parent": "parent_id", "folder_name": "name",
        }
        for slot, key in slot_to_key.items():
            if slot in slots:
                if key in safe:
                    raise _clarification_error("resolved_parameters")
    elif slots:
        raise _clarification_error("missing_slots")
    return safe, slots


def _validate_execution_binding(
    *, state: ClarificationState, action: str, initiating_operation_id: str,
    execution_operation_id: str | None, plan_ref: str | None, plan_digest: str | None,
    parameters: Mapping[str, Any], context_id: str,
) -> None:
    if state is ClarificationState.WAITING and any(
            value is not None for value in (execution_operation_id, plan_ref, plan_digest)):
        raise _clarification_error("execution_binding")
    if state in {ClarificationState.CLAIMED, ClarificationState.UNKNOWN,
                 ClarificationState.COMPLETED} and any(
            value is None for value in (execution_operation_id, plan_ref, plan_digest)):
        raise _clarification_error("execution_binding")
    if state in {ClarificationState.CANCELLED, ClarificationState.REPLACED} and any(
            value is not None for value in (execution_operation_id, plan_ref, plan_digest)):
        raise _clarification_error("execution_binding")
    if ((execution_operation_id is None) != (plan_ref is None)
            or (plan_ref is None) != (plan_digest is None)):
        raise _clarification_error("execution_binding")
    if execution_operation_id is None:
        return
    _clarification_identifier(execution_operation_id, "execution_operation_id")
    if execution_operation_id == initiating_operation_id:
        raise _clarification_error("execution_operation_id")
    if not isinstance(plan_ref, str) or not plan_ref.strip():
        raise _clarification_error("execution_plan_ref")
    _validate_context_ref(plan_ref, context_id, "execution_plan_ref")
    _clarification_digest(plan_digest, "execution_plan_digest")
    required_parameters = {
        "move_transcript": {"doc_id", "destination_id"},
        "create_folder": {"parent_id", "name"},
        "create_then_move": {"doc_id", "parent_id", "name"},
    }.get(action, set())
    if any(not parameters.get(key) for key in required_parameters):
        raise _clarification_error("resolved_parameters")


def _validate_clarification_identity_refs(record: Any) -> None:
    for field in ("clarification_id", "context_id", "initiating_operation_id",
                  "origin_conversation_id"):
        _clarification_identifier(getattr(record, field), field)
    if type(record.actor_id) is not int or record.actor_id <= 0:
        raise _clarification_error("actor_id")
    expected_id = build_clarification_id(record.context_id, record.initiating_operation_id)
    if record.clarification_id != expected_id:
        raise _clarification_error("clarification_id")
    prefix = f"telegram_sessions/{record.context_id}/clarifications/{record.clarification_id}/"
    if record.questions_ref is None:
        object.__setattr__(record, "questions_ref", prefix + "questions")
    if record.answers_ref is None:
        object.__setattr__(record, "answers_ref", prefix + "answers")
    for field in ("original_command_ref", "evidence_snapshot_ref", "questions_ref", "answers_ref"):
        value = getattr(record, field)
        if not isinstance(value, str) or not value.strip():
            raise _clarification_error(field)
        _validate_context_ref(value, record.context_id, field)
    _clarification_digest(record.original_command_digest, "original_command_digest")
    _clarification_digest(record.evidence_snapshot_digest, "evidence_snapshot_digest")
    if record.protocol != _CLARIFICATION_PROTOCOL:
        raise _clarification_error("protocol")
    if not isinstance(record.action, str) or record.action not in _CLARIFICATION_ACTIONS:
        raise _clarification_error("action")
    if (record.questions_ref != prefix + "questions"
            or record.answers_ref != prefix + "answers"):
        raise _clarification_error("collection_refs")


def _validate_clarification_state(record: Any) -> tuple[ClarificationState, tuple[str, ...]]:
    state = record.state
    if not isinstance(state, ClarificationState):
        try:
            state = ClarificationState(state)
        except (ValueError, TypeError):
            raise _clarification_error("state") from None
    object.__setattr__(record, "state", state)
    if type(record.generation) is not int or record.generation < 1:
        raise _clarification_error("generation")
    if type(record.revision) is not int or record.revision < 1:
        raise _clarification_error("revision")
    if not isinstance(record.missing_slots, (tuple, list)):
        raise _clarification_error("missing_slots")
    slots = tuple(record.missing_slots)
    if (any(not isinstance(slot, str) or slot not in _CLARIFICATION_SLOTS for slot in slots)
            or len(set(slots)) != len(slots)):
        raise _clarification_error("missing_slots")
    object.__setattr__(record, "missing_slots", slots)
    return state, slots


def _validate_clarification_outcome_and_times(record: Any) -> tuple[datetime, datetime]:
    if record.result_ref is not None and (
            not isinstance(record.result_ref, str) or not record.result_ref.strip()):
        raise _clarification_error("result_ref")
    if record.result_ref is not None:
        _validate_context_ref(record.result_ref, record.context_id, "result_ref")
    if record.safe_error is not None and (
            not isinstance(record.safe_error, str) or not record.safe_error.strip()
            or len(record.safe_error) > 500
            or any(ord(character) < 32 or ord(character) == 127
                   for character in record.safe_error)):
        raise _clarification_error("safe_error")
    if (not isinstance(record.created_at, datetime) or record.created_at.tzinfo is None
            or record.created_at.utcoffset() is None or not isinstance(record.updated_at, datetime)
            or record.updated_at.tzinfo is None or record.updated_at.utcoffset() is None):
        raise _clarification_error("timestamps")
    created = record.created_at.astimezone(timezone.utc)
    updated = record.updated_at.astimezone(timezone.utc)
    if updated < created:
        raise _clarification_error("timestamps")
    return created, updated


def build_clarification_id(context_id: str, initiating_operation_id: str) -> str:
    """Derive a stable clarification identity from its context and source operation."""
    context_id = _clarification_identifier(context_id, "context_id")
    initiating_operation_id = _clarification_identifier(
        initiating_operation_id, "initiating_operation_id",
    )
    digest = hashlib.sha256(
        f"{context_id}\0{initiating_operation_id}".encode("utf-8")
    ).hexdigest()
    return f"tgclarification-{digest}"


class ClarificationState(StrEnum):
    WAITING = "waiting"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REPLACED = "replaced"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PendingClarification:
    """Validated immutable intent record; references are metadata, not authority."""

    clarification_id: str
    context_id: str
    actor_id: int
    initiating_operation_id: str
    origin_conversation_id: str
    original_command_ref: str
    original_command_digest: str
    action: str
    resolved_parameters: Mapping[str, Any]
    missing_slots: tuple[str, ...]
    evidence_snapshot_ref: str
    evidence_snapshot_digest: str
    state: ClarificationState
    created_at: datetime
    updated_at: datetime
    generation: int = 1
    revision: int = 1
    execution_operation_id: str | None = None
    execution_plan_ref: str | None = None
    execution_plan_digest: str | None = None
    result_ref: str | None = None
    safe_error: str | None = None
    questions_ref: str | None = None
    answers_ref: str | None = None
    protocol: str = _CLARIFICATION_PROTOCOL

    def __post_init__(self) -> None:
        _validate_clarification_identity_refs(self)
        state, slots = _validate_clarification_state(self)
        safe_parameters, slots = _normalize_clarification_parameters(
            self.action, state, self.resolved_parameters, slots,
        )
        object.__setattr__(self, "missing_slots", slots)
        object.__setattr__(self, "resolved_parameters", _freeze_json(safe_parameters))
        _validate_execution_binding(
            state=state, action=self.action,
            initiating_operation_id=self.initiating_operation_id,
            execution_operation_id=self.execution_operation_id,
            plan_ref=self.execution_plan_ref, plan_digest=self.execution_plan_digest,
            parameters=safe_parameters, context_id=self.context_id,
        )
        created, updated = _validate_clarification_outcome_and_times(self)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarification_id": self.clarification_id,
            "context_id": self.context_id,
            "actor_id": self.actor_id,
            "initiating_operation_id": self.initiating_operation_id,
            "origin_conversation_id": self.origin_conversation_id,
            "original_command_ref": self.original_command_ref,
            "original_command_digest": self.original_command_digest,
            "action": self.action,
            "resolved_parameters": _thaw_json(self.resolved_parameters),
            "missing_slots": list(self.missing_slots),
            "evidence_snapshot_ref": self.evidence_snapshot_ref,
            "evidence_snapshot_digest": self.evidence_snapshot_digest,
            "generation": self.generation,
            "revision": self.revision,
            "state": self.state.value,
            "execution_operation_id": self.execution_operation_id,
            "execution_plan_ref": self.execution_plan_ref,
            "execution_plan_digest": self.execution_plan_digest,
            "result_ref": self.result_ref,
            "safe_error": self.safe_error,
            "questions_ref": self.questions_ref,
            "answers_ref": self.answers_ref,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingClarification:
        if not isinstance(data, Mapping) or data.get("protocol") != _CLARIFICATION_PROTOCOL:
            raise _clarification_error("protocol")
        allowed = {
            "clarification_id", "context_id", "actor_id", "initiating_operation_id",
            "origin_conversation_id", "original_command_ref", "original_command_digest",
            "action", "resolved_parameters", "missing_slots", "evidence_snapshot_ref",
            "evidence_snapshot_digest", "generation", "revision", "state",
            "execution_operation_id", "execution_plan_ref", "execution_plan_digest",
            "result_ref", "safe_error", "questions_ref", "answers_ref", "created_at",
            "updated_at", "protocol",
        }
        if set(data) - allowed:
            raise _clarification_error("fields")
        fields = dict(data)
        required = {
            "clarification_id", "context_id", "actor_id", "initiating_operation_id",
            "origin_conversation_id", "original_command_ref", "original_command_digest",
            "action", "evidence_snapshot_ref", "evidence_snapshot_digest", "generation",
            "revision", "state", "created_at", "updated_at",
        }
        if not required.issubset(fields):
            raise _clarification_error("fields")
        for field in ("execution_operation_id", "execution_plan_ref", "execution_plan_digest",
                      "result_ref", "safe_error"):
            fields.setdefault(field, None)
        fields.setdefault("missing_slots", [])
        fields.setdefault("resolved_parameters", {})
        clarification_id, context_id = fields["clarification_id"], fields["context_id"]
        prefix = f"telegram_sessions/{context_id}/clarifications/{clarification_id}/"
        fields.setdefault("questions_ref", prefix + "questions")
        fields.setdefault("answers_ref", prefix + "answers")
        return cls(**fields)


def _validate_context_ref(value: str, context_id: str, field: str) -> None:
    if value.startswith("telegram_sessions/"):
        parts = value.split("/")
        if len(parts) < 3 or parts[1] != context_id:
            raise _clarification_error(field)


def build_context_id(chat_id: int, thread_id: int | None = None) -> str:
    """Build the documented context key without narrowing Telegram IDs."""
    if type(chat_id) is not int:
        raise TypeError("chat_id must be an integer")
    if thread_id is not None and type(thread_id) is not int:
        raise TypeError("thread_id must be an integer or None")
    return f"{chat_id}_{thread_id or 'default'}"


class TelegramDraftState(StrEnum):
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANED = "cleaned"

    def can_transition_to(
        self, target: TelegramDraftState, *, manual_retry: bool = False
    ) -> bool:
        if self is TelegramDraftState.COLLECTING:
            return target is TelegramDraftState.PROCESSING
        if self is TelegramDraftState.PROCESSING:
            return target in (TelegramDraftState.COMPLETED, TelegramDraftState.FAILED)
        if self is TelegramDraftState.COMPLETED:
            return target is TelegramDraftState.CLEANED
        if self is TelegramDraftState.FAILED:
            return target is TelegramDraftState.PROCESSING and manual_retry
        return False


class TelegramTranscriptionState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True, slots=True)
class TelegramContext:
    chat_id: int
    thread_id: int | None = None
    topic_name: str | None = None
    active_draft_id: str | None = None
    draft_status: str | None = None
    active_conversation_id: str | None = None
    selected_dialog_number: int | None = None
    selection_sequence: int = 0
    last_doc_id: str | None = None
    next_dialog_number: int = 1
    next_operation_sequence: int = 1
    head_sequence: int = 0
    active_operation_id: str | None = None
    lock_owner: str | None = None
    lock_expires_at: datetime | None = None
    fence: int = 0
    reserved_operation_id: str | None = None
    status_message_id: int | None = None
    status_revision: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Appended for positional-constructor compatibility with older callers.
    pending_clarification_id: str | None = None
    clarification_generation: int = 0

    def __post_init__(self) -> None:
        if self.draft_status not in (None, TelegramDraftState.COLLECTING.value):
            raise ValueError("draft_status must be 'collecting' or None")
        if (self.pending_clarification_id is not None
                and (not isinstance(self.pending_clarification_id, str)
                     or not _CLARIFICATION_ID.fullmatch(self.pending_clarification_id))):
            raise ValueError("pending_clarification_id must be a valid ID or None")
        if type(self.clarification_generation) is not int or self.clarification_generation < 0:
            raise ValueError("clarification_generation must be a nonnegative integer")

    @property
    def context_id(self) -> str:
        return build_context_id(self.chat_id, self.thread_id)


@dataclass(frozen=True, slots=True)
class TelegramDraft:
    draft_id: str
    dialog_number: int
    state: TelegramDraftState
    conversation_id: str
    created_at: datetime
    sealed_at: datetime | None = None
    ordered_message_ids: tuple[int, ...] = ()
    manifest_ref: str | None = None
    manifest_digest: str | None = None
    input_revision: int = 0
    active_save_operation_id: str | None = None
    immutable_save_intent_id: str | None = None
    doc_id: str | None = None
    completion_certificate_ref: str | None = None
    cleanup_state: str | None = None
    required_audio_count: int = 0
    completed_audio_ids: tuple[int, ...] = ()
    prepared_count: int = 0
    context_capacity: int | None = None
    preparation_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_message_ids", tuple(self.ordered_message_ids))
        object.__setattr__(self, "completed_audio_ids", tuple(self.completed_audio_ids))


@dataclass(frozen=True, slots=True)
class TelegramSourceMessage:
    message_id: int
    intake_sequence: int
    received_at: datetime
    original_date: datetime | None
    source_chat_id: int | None = None
    source_message_id: int | None = None
    source_user_id: int | None = None
    reliable_author_key: str | None = None
    display_speaker: str | None = None
    text: str | None = None
    caption: str | None = None
    audio_file_id: str | None = None
    audio_file_unique_id: str | None = None
    audio_mime_type: str | None = None
    audio_file_size: int | None = None
    transcription_state: TelegramTranscriptionState = TelegramTranscriptionState.NOT_REQUIRED
    transcription_text: str | None = None
    transcription_ref: str | None = None
    transcription_error: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedUtterance:
    order_index: int
    speaker: str
    reliable_author_key: str | None
    text: str
    word_count: int
    source_message_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_message_ids", tuple(self.source_message_ids))
