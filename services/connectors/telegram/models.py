"""Immutable domain contracts for Telegram connector workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


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

    def __post_init__(self) -> None:
        if self.draft_status not in (None, TelegramDraftState.COLLECTING.value):
            raise ValueError("draft_status must be 'collecting' or None")

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_message_ids", tuple(self.ordered_message_ids))


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
