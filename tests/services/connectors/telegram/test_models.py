from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone

import pytest

from services.connectors.telegram.models import (
    PreparedUtterance,
    TelegramContext,
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
    build_context_id,
)


def _field_names(model):
    return tuple(field.name for field in fields(model))


def test_build_context_id_preserves_full_integer_ids_and_documented_zero_behavior():
    chat_id = 9_223_372_036_854_775_123
    assert build_context_id(chat_id) == f"{chat_id}_default"
    assert build_context_id(chat_id, 4_294_967_301) == f"{chat_id}_4294967301"
    assert build_context_id(chat_id, 0) == f"{chat_id}_default"


@pytest.mark.parametrize("bad_id", [True, False, "42", 42.0])
def test_build_context_id_rejects_non_integer_chat_ids(bad_id):
    with pytest.raises(TypeError):
        build_context_id(bad_id)


@pytest.mark.parametrize("bad_id", [True, "42", 42.0])
def test_build_context_id_rejects_non_integer_thread_ids(bad_id):
    with pytest.raises(TypeError):
        build_context_id(42, bad_id)


def test_context_schema_key_and_immutability():
    expected = (
        "chat_id", "thread_id", "topic_name", "active_draft_id", "draft_status",
        "active_conversation_id", "selected_dialog_number", "selection_sequence",
        "last_doc_id", "next_dialog_number", "next_operation_sequence", "head_sequence",
        "active_operation_id", "lock_owner", "lock_expires_at", "fence",
        "reserved_operation_id", "status_message_id", "status_revision", "created_at",
        "updated_at",
    )
    assert _field_names(TelegramContext) == expected
    context = TelegramContext(chat_id=-8_000_000_001, thread_id=5_000_000_007)
    assert context.context_id == "-8000000001_5000000007"
    assert context.chat_id == -8_000_000_001
    assert context.thread_id == 5_000_000_007
    assert context.draft_status is None
    assert not hasattr(context, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.chat_id = 7
    with pytest.raises(ValueError):
        TelegramContext(chat_id=1, draft_status="processing")


@pytest.mark.parametrize(
    ("source", "target", "manual_retry", "allowed"),
    [
        (TelegramDraftState.COLLECTING, TelegramDraftState.PROCESSING, False, True),
        (TelegramDraftState.COLLECTING, TelegramDraftState.COMPLETED, False, False),
        (TelegramDraftState.PROCESSING, TelegramDraftState.COMPLETED, False, True),
        (TelegramDraftState.PROCESSING, TelegramDraftState.FAILED, False, True),
        (TelegramDraftState.COMPLETED, TelegramDraftState.CLEANED, False, True),
        (TelegramDraftState.FAILED, TelegramDraftState.PROCESSING, False, False),
        (TelegramDraftState.FAILED, TelegramDraftState.PROCESSING, True, True),
        (TelegramDraftState.CLEANED, TelegramDraftState.PROCESSING, True, False),
        (TelegramDraftState.CLEANED, TelegramDraftState.COLLECTING, False, False),
    ],
)
def test_draft_state_transition_contract(source, target, manual_retry, allowed):
    assert source.can_transition_to(target, manual_retry=manual_retry) is allowed


def test_draft_schema_nullable_doc_and_immutable_ordered_ids():
    expected = (
        "draft_id", "dialog_number", "state", "conversation_id", "created_at", "sealed_at",
        "ordered_message_ids", "manifest_ref", "manifest_digest", "input_revision",
        "active_save_operation_id", "immutable_save_intent_id", "doc_id",
        "completion_certificate_ref", "cleanup_state", "required_audio_count",
        "completed_audio_ids", "prepared_count", "context_capacity", "preparation_digest",
    )
    assert _field_names(TelegramDraft) == expected
    ids = [12, 9]
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    audio_ids = [101]
    draft = TelegramDraft(
        draft_id="draft-1", dialog_number=3, state=TelegramDraftState.COLLECTING,
        conversation_id="conversation-1", created_at=timestamp,
        ordered_message_ids=ids, immutable_save_intent_id=None, doc_id=None,
        completed_audio_ids=audio_ids,
    )
    ids.append(15)
    audio_ids.append(102)
    assert draft.ordered_message_ids == (12, 9)
    assert draft.completed_audio_ids == (101,)
    assert draft.required_audio_count == 0
    assert draft.prepared_count == 0
    assert draft.context_capacity is None
    assert draft.preparation_digest is None
    assert isinstance(draft.ordered_message_ids, tuple)
    assert draft.immutable_save_intent_id is None
    assert draft.doc_id is None
    assert not hasattr(draft, "__dict__")
    with pytest.raises(FrozenInstanceError):
        draft.dialog_number = 4


def test_source_message_schema_nullable_origin_and_audio_references_only():
    expected = (
        "message_id", "intake_sequence", "received_at", "original_date", "source_chat_id",
        "source_message_id", "source_user_id", "reliable_author_key", "display_speaker",
        "text", "caption", "audio_file_id", "audio_file_unique_id", "audio_mime_type",
        "audio_file_size", "transcription_state", "transcription_text", "transcription_ref",
        "transcription_error",
    )
    assert _field_names(TelegramSourceMessage) == expected
    message = TelegramSourceMessage(
        message_id=6,
        intake_sequence=1,
        received_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        original_date=None,
        source_chat_id=None,
        source_message_id=None,
        reliable_author_key=None,
        audio_file_id="file-id",
        audio_file_unique_id="unique-id",
        audio_mime_type="audio/ogg",
        audio_file_size=1234,
    )
    assert message.intake_sequence == 1
    assert message.source_chat_id is None
    assert message.source_message_id is None
    assert message.reliable_author_key is None
    assert message.audio_file_id == "file-id"
    assert message.audio_file_unique_id == "unique-id"
    assert not hasattr(message, "audio_bytes")
    assert not hasattr(message, "audio_binary")
    assert not hasattr(message, "token_url")
    assert not hasattr(message, "audio_token_url")
    assert not hasattr(message, "__dict__")


def test_prepared_utterance_schema_and_immutable_source_ids():
    assert _field_names(PreparedUtterance) == (
        "order_index", "speaker", "reliable_author_key", "text", "word_count",
        "source_message_ids",
    )
    ids = [12, 9]
    utterance = PreparedUtterance(0, "Speaker", None, "hello world", 2, ids)
    ids.append(15)
    assert utterance.reliable_author_key is None
    assert utterance.source_message_ids == (12, 9)
    assert isinstance(utterance.source_message_ids, tuple)
    assert not hasattr(utterance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        utterance.text = "changed"
