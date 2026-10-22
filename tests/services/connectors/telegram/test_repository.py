from datetime import datetime, timezone

import pytest

from services.connectors.telegram.models import PreparedUtterance
from services.connectors.telegram.repository import TelegramRepository


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _message(message_id=101, **fields):
    return {"message_id": message_id, "source_chat_id": 88,
            "source_message_id": 9, "source_user_id": None,
            "reliable_author_key": None, "display_speaker": None,
            "text": "forwarded text", "caption": None, **fields}


def _accept(repository, update_id=1, message_id=101, **kwargs):
    return repository.accept_forward(
        update_id, chat_id=42, thread_id=5, message=_message(message_id), now=NOW, **kwargs
    )


def _read(client, path):
    parts = path.split("/")
    ref = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        ref = ref.collection(parts[index]).document(parts[index + 1])
    snapshot = ref.get()
    return snapshot.to_dict() if snapshot.exists else None


def test_forward_transaction_persists_receipt_draft_dialog_message_and_ack(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    result = _accept(repo)
    context_id = result.context_id

    assert result.accepted is True
    assert result.dialog_number == 1
    assert result.intake_sequence == 1
    context = repo.get_context(context_id)
    assert context == {
        "chat_id": 42,
        "thread_id": 5,
        "topic_name": None,
        "active_draft_id": result.draft_id,
        "draft_status": "collecting",
        "active_conversation_id": None,
        "selected_dialog_number": None,
        "selection_sequence": 0,
        "last_doc_id": None,
        "next_dialog_number": 2,
        "next_operation_sequence": 1,
        "head_sequence": 0,
        "active_operation_id": None,
        "lock_owner": None,
        "lock_expires_at": None,
        "fence": 0,
        "reserved_operation_id": None,
        "status_message_id": None,
        "status_revision": 0,
        "created_at": NOW,
        "updated_at": NOW,
        "next_intake_sequence": 2,
    }
    assert _read(client, f"telegram_sessions/{context_id}/updates/1")["draft_id"] == result.draft_id
    draft = repo.get_draft(context_id, result.draft_id)
    assert draft["state"] == "collecting"
    assert draft["ordered_message_ids"] == [101]
    assert repo.list_messages(context_id, result.draft_id)[0]["text"] == "forwarded text"
    assert _read(client, f"telegram_sessions/{context_id}/dialogs/1")["doc_id"] is None
    notification = _read(client, f"notification_outbox/{result.notification_id}")
    task = _read(client, f"task_outbox/{result.outbox_id}")
    assert notification["kind"] == "draft_ack"
    assert notification["body"] == "Диалог №1: сообщение принято"
    assert task["notification_id"] == result.notification_id


def test_second_forward_appends_without_creating_another_ack_or_outbox(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    first = _accept(repo, update_id=1, message_id=101)
    second = _accept(repo, update_id=2, message_id=102)

    assert second.accepted is True
    assert second.draft_id == first.draft_id
    assert second.dialog_number == first.dialog_number
    assert second.intake_sequence == 2
    assert second.notification_id is None
    assert second.outbox_id is None
    assert repo.get_draft(first.context_id, first.draft_id)["ordered_message_ids"] == [101, 102]
    receipt = repo.get_update_receipt(first.context_id, 2)
    assert receipt["notification_id"] is None
    assert receipt["outbox_id"] is None
    assert len([path for path in firestore_fake.documents if path.startswith("notification_outbox/")]) == 1
    assert len([path for path in firestore_fake.documents if path.startswith("task_outbox/")]) == 1


def test_forward_retains_topic_name_when_later_forward_omits_it(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    first = _accept(repo, update_id=1, message_id=101, topic_name="Project updates")

    second = _accept(repo, update_id=2, message_id=102)
    assert repo.get_context(first.context_id)["topic_name"] == "Project updates"

    _accept(repo, update_id=3, message_id=103, topic_name="Renamed topic")
    assert repo.get_context(first.context_id)["topic_name"] == "Renamed topic"


def test_forward_fills_missing_context_defaults_without_resetting_existing_values(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id = "42_5"
    client.collection("telegram_sessions").document(context_id).set({
        "chat_id": 42,
        "thread_id": 5,
        "topic_name": "Existing topic",
        "next_dialog_number": 7,
        "next_intake_sequence": 11,
        "selection_sequence": 4,
        "next_operation_sequence": 8,
        "head_sequence": 6,
        "lock_owner": "worker-1",
        "fence": 3,
    })

    result = _accept(repo, topic_name="New topic")

    context = repo.get_context(context_id)
    assert result.dialog_number == 7
    assert result.intake_sequence == 11
    assert context["topic_name"] == "New topic"
    assert context["next_dialog_number"] == 8
    assert context["next_intake_sequence"] == 12
    assert context["selection_sequence"] == 4
    assert context["next_operation_sequence"] == 8
    assert context["head_sequence"] == 6
    assert context["lock_owner"] == "worker-1"
    assert context["fence"] == 3
    assert context["active_conversation_id"] is None
    assert context["status_revision"] == 0
    assert context["created_at"] == NOW


def test_duplicate_replays_exact_identity_and_failed_draft_does_not_block_next(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    first = _accept(repo)
    replay = _accept(repo)
    assert replay == first

    draft_ref = client.collection("telegram_sessions").document(first.context_id)
    draft_ref = draft_ref.collection("drafts").document(first.draft_id)
    draft_ref.update({"state": "failed"})
    second = _accept(repo, 2, 102)
    assert second.draft_id != first.draft_id
    assert second.dialog_number == first.dialog_number + 1
    assert [item["draft_id"] for item in repo.list_failed_drafts(first.context_id)] == [first.draft_id]


def test_busy_context_receipt_does_not_store_message_content(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id = "42_5"
    client.collection("telegram_sessions").document(context_id).set({
        "chat_id": 42, "thread_id": 5, "topic_name": "Busy topic",
        "reserved_operation_id": "op-1",
    })
    result = _accept(repo)
    assert result.accepted is False
    assert repo.get_context(context_id)["topic_name"] == "Busy topic"
    assert not any("/drafts/" in path for path in firestore_fake.documents)
    receipt = repo.get_update_receipt(context_id, 1)
    assert receipt["accepted"] is False
    assert "forwarded text" not in str(firestore_fake.documents)


def test_prepared_records_and_metadata_are_persisted_as_ordered_firestore_lists(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    result = _accept(repo)
    rows = [PreparedUtterance(0, "Ada", None, "hello", 1, [101])]
    repo.save_prepared_utterances(
        result.context_id, result.draft_id, rows, required_audio_count=1,
        completed_audio_ids=[101], context_capacity=25,
        preparation_digest="sha256:stable", now=NOW,
    )
    stored = _read(client, f"telegram_sessions/{result.context_id}/drafts/{result.draft_id}/prepared/0")
    draft = repo.get_draft(result.context_id, result.draft_id)
    assert stored["source_message_ids"] == [101]
    assert draft["prepared_count"] == 1
    assert draft["completed_audio_ids"] == [101]
    assert draft["context_capacity"] == 25
    assert draft["preparation_digest"] == "sha256:stable"


def test_document_binding_reuses_number_and_allocator_never_decrements(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    first = _accept(repo)
    number = repo.bind_document(
        first.context_id, "google-doc-1", "conversation-doc-1",
        metadata={"title": "Notes"}, now=NOW,
    )
    assert number == 2
    assert repo.bind_document(
        first.context_id, "google-doc-1", "conversation-doc-1",
        metadata={"title": "Ignored on exact replay"}, now=NOW,
    ) == number
    context = repo.get_context(first.context_id)
    assert context["next_dialog_number"] == 3
    assert _read(client, f"telegram_sessions/{first.context_id}/dialogs/2")["doc_id"] == "google-doc-1"


@pytest.mark.parametrize("bad_now", [datetime(2026, 10, 22), "now"])
def test_forward_rejects_naive_or_malformed_timestamp_before_writing(firestore_fake, bad_now):
    repo = TelegramRepository(firestore_fake.client_class())
    with pytest.raises(ValueError):
        repo.accept_forward(1, chat_id=42, message=_message(), now=bad_now)
    assert firestore_fake.documents == {}


def test_forward_rejects_raw_or_unknown_message_fields(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    with pytest.raises(ValueError, match="normalized"):
        repo.accept_forward(
            1, chat_id=42, message={**_message(), "raw_update": {"text": "private"}}, now=NOW
        )
    assert firestore_fake.documents == {}
