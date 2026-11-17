import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from services.connectors.telegram.models import (
    ClarificationState,
    PendingClarification,
    PreparedUtterance,
    build_clarification_id,
)
from services.connectors.telegram.repository import TelegramRepository
from services.connectors.telegram.save import TelegramCompletionCertificate
from services.connectors.telegram.commands import TelegramCommand
from core.integrations.firestore.operation_queue import claim_head, complete_head
from google.cloud import firestore


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
    assert {key: value for key, value in context.items()
            if key not in {"latest_receipt_notification_id", "open_text_receipt_group_id",
                           "latest_receipt_group_id", "pending_receipt_groups"}} == {
        "chat_id": 42,
        "thread_id": 5,
        "topic_name": None,
        "active_draft_id": result.draft_id,
        "draft_status": "collecting",
        "active_conversation_id": f"{context_id}-1",
        "selected_dialog_number": 1,
        "selection_sequence": 1,
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
        "pending_clarification_id": None,
        "clarification_generation": 0,
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
    assert notification["body"].startswith("Dialog #1: Thinking")
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
    assert second.notification_id == first.notification_id
    assert second.outbox_id is not None
    assert repo.get_draft(first.context_id, first.draft_id)["ordered_message_ids"] == [101, 102]
    receipt = repo.get_update_receipt(first.context_id, 2)
    assert receipt["notification_id"] == first.notification_id
    assert receipt["outbox_id"] == second.outbox_id
    assert len([path for path in firestore_fake.documents if path.startswith("notification_outbox/")]) == 1
    assert len([path for path in firestore_fake.documents if path.startswith("task_outbox/")]) == 3


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
    assert context["selection_sequence"] == 5
    assert context["next_operation_sequence"] == 8
    assert context["head_sequence"] == 6
    assert context["lock_owner"] == "worker-1"
    assert context["fence"] == 3
    assert context["active_conversation_id"] == f"{context_id}-7"
    assert context["selected_dialog_number"] == 7
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
    assert receipt["outcome"] == "busy_forward_refused"
    assert receipt["publish_rejection"] is True
    assert result.outbox_id and result.notification_id
    refusal = repo.get_notification(result.notification_id)
    assert "processing another request" in refusal["body"]
    assert result.notification_id
    assert repo.get_notification(result.notification_id)["operation_id"] is None
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


def test_prepared_checkpoint_pages_bound_transaction_writes_and_preserve_full_input(
    firestore_fake, monkeypatch,
):
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    result = _accept(repo)
    largest_transaction = {"writes": 0}
    original_set = FakeTransaction.set

    def counted_set(transaction, reference, data, merge=False):
        original_set(transaction, reference, data, merge)
        largest_transaction["writes"] = max(
            largest_transaction["writes"], len(transaction._writes),
        )

    monkeypatch.setattr(FakeTransaction, "set", counted_set)
    rows = [PreparedUtterance(i, "Speaker", None, f"row {i} " + ("x" * 2500),
                              3, (101,)) for i in range(501)]
    repo.save_prepared_utterances(
        result.context_id, result.draft_id, rows, required_audio_count=0,
        completed_audio_ids=(), context_capacity=521,
        preparation_digest="sha256:large", now=NOW,
    )

    stored = [path for path in firestore_fake.documents
              if path.startswith(f"telegram_sessions/{result.context_id}/drafts/{result.draft_id}/prepared/")]
    assert len(stored) == 501
    assert sum(len(firestore_fake.documents[path]["text"]) for path in stored) > 1_000_000
    assert repo.get_draft(result.context_id, result.draft_id)["prepared_count"] == 501
    assert largest_transaction["writes"] <= 400


def test_transcription_checkpoint_rejects_stale_queue_fence(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, operation_id = "42_5", "draft-1", "save-op"
    context = client.collection("telegram_sessions").document(context_id)
    context.set({"reserved_operation_id": operation_id, "active_operation_id": operation_id,
                 "lock_owner": "worker", "fence": 4, "lock_expires_at": NOW.replace(year=2027)})
    operation = context.collection("operations").document(operation_id)
    operation.set({"operation_id": operation_id, "state": "running",
                   "target_draft_id": draft_id, "save_operation_id": operation_id})
    draft = context.collection("drafts").document(draft_id)
    draft.set({"draft_id": draft_id, "active_save_operation_id": operation_id,
               "immutable_save_intent_id": operation_id})
    draft.collection("messages").document("101").set({"message_id": 101})

    with pytest.raises(PermissionError):
        repo.checkpoint_transcription(
            context_id, draft_id, 101, "must not persist",
            queue_operation_id=operation_id, queue_owner="worker", queue_fence=3, now=NOW,
        )
    assert draft.collection("messages").document("101").get().to_dict() == {"message_id": 101}


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


def test_transcription_checkpoint_is_message_scoped_idempotent_and_preserves_source_and_manifest(
    firestore_fake,
):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    first = _accept(repo, update_id=1, message_id=101)
    _accept(repo, update_id=2, message_id=102)
    manifest_ref = client.collection("telegram_sessions").document(first.context_id)
    manifest_ref = manifest_ref.collection("drafts").document(first.draft_id)
    manifest_ref.collection("save_manifests").document("manifest-1").set({
        "ordered_message_ids": [101, 102], "manifest_digest": "sealed-digest",
    })
    original = repo.list_messages(first.context_id, first.draft_id)

    repo.checkpoint_transcription(first.context_id, first.draft_id, 101, "verbatim transcript")
    after_first = repo.list_messages(first.context_id, first.draft_id)
    repo.checkpoint_transcription(first.context_id, first.draft_id, 101, "verbatim transcript")

    assert repo.list_messages(first.context_id, first.draft_id) == after_first
    assert after_first[0]["message_id"] == 101
    assert after_first[0]["transcription_state"] == "completed"
    assert after_first[0]["transcription_text"] == "verbatim transcript"
    assert after_first[0]["text"] == original[0]["text"]
    assert after_first[0]["audio_file_id"] == original[0]["audio_file_id"]
    assert after_first[1] == original[1]
    assert _read(
        client,
        f"telegram_sessions/{first.context_id}/drafts/{first.draft_id}/save_manifests/manifest-1",
    ) == {"ordered_message_ids": [101, 102], "manifest_digest": "sealed-digest"}


def test_transcription_failure_checkpoint_stores_only_safe_bounded_category(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accept(repo)
    before = repo.list_messages(accepted.context_id, accepted.draft_id)[0]

    repo.checkpoint_transcription_failure(
        accepted.context_id, accepted.draft_id, 101, "transcription_failed",
    )

    after = repo.list_messages(accepted.context_id, accepted.draft_id)[0]
    assert after["transcription_state"] == "failed"
    assert after["transcription_error"] == "transcription_failed"
    assert after["transcription_text"] is None
    assert after["text"] == before["text"]
    assert after["audio_file_id"] == before["audio_file_id"]
    with pytest.raises(ValueError, match="error code"):
        repo.checkpoint_transcription_failure(
            accepted.context_id, accepted.draft_id, 101,
            "provider exception with sensitive details",
        )


def _seed_completion_records(client, *, mismatch=None):
    context_id, draft_id = "42_5", "draft-save"
    intent_id, operation_id, digest, doc_id = "intent-save", "operation-save", "manifest-digest", "doc-save"
    binding = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest}
    certificate_id = "certificate-save"
    authorization = {
        "certificate_id": certificate_id,
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "origin_binding": binding,
    }
    shared_certificate = {
        "certificate_id": certificate_id,
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "doc_id": doc_id,
        "origin_binding": binding,
        "receipt_ref": "index_completions/doc-save",
        "receipt_version": 4,
    }
    operation_authorization = dict(authorization)
    intent_data = {
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "completion_certificate": shared_certificate,
        "cleanup_authorization": authorization,
    }
    operation_data = {
        "operation_id": operation_id,
        "save_intent_id": intent_id,
        "state": "completed",
        "doc_id": doc_id,
        "cleanup_authorization": operation_authorization,
    }
    if mismatch == "foreign_origin":
        intent_data["completion_certificate"]["origin_binding"] = {
            **binding, "draft_id": "draft-foreign",
        }
    elif mismatch == "foreign_operation_intent":
        operation_data["save_intent_id"] = "intent-foreign"
    elif mismatch == "foreign_document":
        operation_data["doc_id"] = "doc-foreign"
    client.collection("dialog_save_intents").document(intent_id).set(intent_data)
    client.collection("dialog_operations").document(operation_id).set(operation_data)
    client.collection("telegram_sessions").document(context_id).set({
        "last_completion_draft_id": draft_id,
        "active_draft_id": "draft-newer",
        "draft_status": "collecting",
        "active_conversation_id": "conversation-current",
        "selected_dialog_number": 3,
        "last_doc_id": doc_id,
    })
    draft_ref = client.collection("telegram_sessions").document(context_id).collection("drafts").document(draft_id)
    draft_ref.set({
        "draft_id": draft_id,
        "dialog_number": 2,
        "conversation_id": f"{context_id}-2",
        "state": "processing",
        "active_save_operation_id": operation_id,
        "immutable_save_intent_id": intent_id,
        "manifest_ref": f"telegram_sessions/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        "manifest_digest": digest,
        "ordered_message_ids": [101, 102],
        "doc_id": doc_id,
    })
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.collection("dialogs").document("2").set({
        "dialog_number": 2, "conversation_id": f"{context_id}-2",
        "draft_id": draft_id, "doc_id": None,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    })
    context_ref.collection("conversations").document(f"{context_id}-2").set({
        "conversation_id": f"{context_id}-2", "kind": "draft",
        "draft_id": draft_id, "doc_id": None, "dialog_number": 2,
        "search_filters": None,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "last_substantive_at": None, "last_turn_sequence": 1,
    })
    context_ref.collection("conversations").document(f"{context_id}-2").collection(
        "turns",
    ).document("turn-1").set({
        "turn_id": "turn-1", "conversation_id": f"{context_id}-2",
        "operation_id": "operation-prior", "role": "user",
        "parts": [{"text": "prior substantive turn"}],
        "turn_sequence": 1, "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
    })
    draft_ref.collection("save_manifests").document(operation_id).set({
        "draft_id": draft_id,
        "save_operation_id": operation_id,
        "manifest_digest": digest,
        "ordered_message_ids": [101, 102],
    })
    client.collection("telegram_sessions").document(context_id).collection("drafts").document("draft-newer").set({
        "draft_id": "draft-newer", "state": "collecting", "private_source": "keep untouched",
    })
    telegram_certificate = {
        "context_id": context_id,
        "draft_id": draft_id,
        "manifest_digest": digest,
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "receipt_ref": "index_completions/doc-save",
        "receipt_version": 4,
        "receipt_digest": "local-receipt-digest",
        "origin_binding": (context_id, draft_id, digest),
    }
    return context_id, draft_id, operation_id, telegram_certificate


def _install_completion_queue_lease(client, context_id, draft_id, *, owner="worker-save",
                                    fence=7, expires_at=None):
    expires_at = expires_at or NOW + timedelta(minutes=5)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.update({
        "reserved_operation_id": "operation-save",
        "active_operation_id": "operation-save",
        "lock_owner": "worker-save", "fence": 7,
        "lock_expires_at": expires_at,
    })
    context_ref.collection("operations").document("operation-save").set({
        "operation_id": "operation-save", "state": "running",
        "target_draft_id": draft_id, "save_intent_id": "intent-save",
    })
    return {"queue_operation_id": "operation-save", "queue_owner": owner,
            "queue_fence": fence, "queue_now": NOW}


def _completion_binding_state(client, context_id, draft_id):
    context = client.collection("telegram_sessions").document(context_id)
    return {
        "context": context.get().to_dict(),
        "draft": context.collection("drafts").document(draft_id).get().to_dict(),
        "dialog": context.collection("dialogs").document("2").get().to_dict(),
        "conversation": context.collection("conversations").document(
            f"{context_id}-2",
        ).get().to_dict(),
        "binding": context.collection("document_bindings").document(
            hashlib.sha256(b"doc-save").hexdigest(),
        ).get().to_dict(),
    }


def test_completion_cleanup_authorization_requires_exact_shared_bindings(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, operation_id, certificate = _seed_completion_records(client)

    assert repo.commit_completion(certificate) is True
    pending = repo.get_draft(context_id, draft_id)
    assert pending["cleanup_state"] == "pending"
    assert isinstance(pending["updated_at"], datetime)
    dialog = _read(client, f"telegram_sessions/{context_id}/dialogs/2")
    conversation = _read(client, f"telegram_sessions/{context_id}/conversations/{context_id}-2")
    assert dialog["doc_id"] == "doc-save"
    assert dialog["conversation_id"] == f"{context_id}-2"
    assert "draft_id" not in dialog
    assert conversation["kind"] == "document"
    assert conversation["draft_id"] is None
    assert conversation["doc_id"] == "doc-save"
    assert conversation["created_at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert conversation["last_turn_sequence"] == 1
    preserved_turn = _read(
        client, f"telegram_sessions/{context_id}/conversations/{context_id}-2/turns/turn-1",
    )
    assert preserved_turn == {
        "turn_id": "turn-1", "conversation_id": f"{context_id}-2",
        "operation_id": "operation-prior", "role": "user",
        "parts": [{"text": "prior substantive turn"}],
        "turn_sequence": 1, "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }
    assert repo.authorize_cleanup(context_id, certificate) is True
    draft = repo.get_draft(context_id, draft_id)
    assert draft["cleanup_state"] == "authorized"
    assert draft["cleanup_authorization"]["telegram_certificate"] == certificate
    assert draft["completion_certificate_ref"] == "dialog_save_intents/intent-save"
    assert draft["doc_id"] == "doc-save"
    client.collection("dialog_save_intents").document("intent-save").set({"changed_after_authorization": True})
    assert repo.authorize_cleanup(context_id, certificate) is True
    client.collection("telegram_sessions").document(context_id).update({
        "last_completion_draft_id": "draft-latest",
        "last_doc_id": "doc-newer",
    })
    assert repo.finalize_cleanup(context_id, certificate) is True
    context = repo.get_context(context_id)
    assert context["active_draft_id"] == "draft-newer"
    assert context["draft_status"] == "collecting"
    assert context["active_conversation_id"] == "conversation-current"
    assert context["selected_dialog_number"] == 3
    assert context["last_completion_draft_id"] == "draft-latest"
    assert context["last_doc_id"] == "doc-newer"
    assert repo.get_draft(context_id, "draft-newer") == {
        "draft_id": "draft-newer", "state": "collecting", "private_source": "keep untouched",
        "ordered_message_ids": [], "completed_audio_ids": [], "required_audio_count": 0,
        "prepared_count": 0, "context_capacity": None, "preparation_digest": None,
    }
    foreign_certificate = {**certificate, "receipt_digest": "foreign-receipt"}
    assert repo.finalize_cleanup(context_id, foreign_certificate) is False
    assert repo.finalize_cleanup(context_id, certificate) is True


def test_commit_completion_replay_accepts_firestore_list_origin_binding(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate_data = _seed_completion_records(client)
    certificate = TelegramCompletionCertificate(**certificate_data)

    assert repo.commit_completion(certificate) is True
    draft_ref = client.collection("telegram_sessions").document(context_id).collection("drafts").document(draft_id)
    draft = draft_ref.get().to_dict()
    draft["completion_certificate"]["origin_binding"] = list(
        draft["completion_certificate"]["origin_binding"]
    )
    draft_ref.set(draft)

    assert repo.commit_completion(certificate) is True


def test_commit_completion_accepts_current_queue_fence(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate = _seed_completion_records(client)
    fence = _install_completion_queue_lease(client, context_id, draft_id)

    assert repo.commit_completion(certificate, **fence) is True
    assert _read(client, f"telegram_sessions/{context_id}/dialogs/2")["doc_id"] == "doc-save"
    assert _read(client, f"telegram_sessions/{context_id}/document_bindings/"
                 f"{hashlib.sha256(b'doc-save').hexdigest()}")["dialog_number"] == 2


@pytest.mark.parametrize("failure", ["stale_fence", "wrong_owner", "expired_lease"])
def test_commit_completion_rejects_invalid_queue_fence_without_mutation(firestore_fake, failure):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate = _seed_completion_records(client)
    expires_at = NOW - timedelta(seconds=1) if failure == "expired_lease" else None
    fence = _install_completion_queue_lease(
        client, context_id, draft_id, expires_at=expires_at,
    )
    if failure == "stale_fence":
        fence["queue_fence"] -= 1
    elif failure == "wrong_owner":
        fence["queue_owner"] = "other-worker"
    before = _completion_binding_state(client, context_id, draft_id)

    with pytest.raises(PermissionError):
        repo.commit_completion(certificate, **fence)

    assert _completion_binding_state(client, context_id, draft_id) == before


@pytest.mark.parametrize(
    "conflict",
    ["dialog_document", "binding_number", "binding_conversation", "conversation_target"],
)
def test_commit_completion_rejects_binding_conflicts_without_overwrite(firestore_fake, conflict):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate = _seed_completion_records(client)
    context = client.collection("telegram_sessions").document(context_id)
    binding_ref = context.collection("document_bindings").document(
        hashlib.sha256(b"doc-save").hexdigest(),
    )
    if conflict == "dialog_document":
        context.collection("dialogs").document("2").update({"doc_id": "doc-other"})
    elif conflict == "binding_number":
        binding_ref.set({"doc_id": "doc-save", "dialog_number": 9,
                         "conversation_id": f"{context_id}-2"})
    elif conflict == "binding_conversation":
        binding_ref.set({"doc_id": "doc-save", "dialog_number": 2,
                         "conversation_id": "conversation-other"})
    else:
        context.collection("conversations").document(f"{context_id}-2").update({
            "doc_id": "doc-other",
        })
    before = _completion_binding_state(client, context_id, draft_id)

    assert repo.commit_completion(certificate) is False

    assert _completion_binding_state(client, context_id, draft_id) == before


@pytest.mark.parametrize("tamper", ["authorization", "manifest"])
def test_durable_cleanup_resume_rejects_local_binding_tampering(firestore_fake, tamper):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, operation_id, certificate = _seed_completion_records(client)
    assert repo.commit_completion(certificate) is True
    assert repo.authorize_cleanup(context_id, certificate) is True

    draft_ref = client.collection("telegram_sessions").document(context_id).collection("drafts").document(draft_id)
    manifest_ref = draft_ref.collection("save_manifests").document(operation_id)
    if tamper == "authorization":
        draft = draft_ref.get().to_dict()
        draft["cleanup_authorization"]["operation_id"] = "operation-foreign"
        draft_ref.set(draft)
    else:
        manifest = manifest_ref.get().to_dict()
        manifest["ordered_message_ids"] = [101, 999]
        manifest_ref.set(manifest)
    client.collection("dialog_save_intents").document("intent-save").set({"changed_after_authorization": True})

    assert repo.authorize_cleanup(context_id, certificate) is False
    assert repo.finalize_cleanup(context_id, certificate) is False


@pytest.mark.parametrize("mismatch", ["foreign_origin", "foreign_operation_intent", "foreign_document"])
def test_completion_cleanup_authorization_rejects_foreign_shared_evidence(firestore_fake, mismatch):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate = _seed_completion_records(
        client, mismatch=mismatch,
    )

    assert repo.commit_completion(certificate) is False
    assert repo.authorize_cleanup(context_id, certificate) is False
    draft = repo.get_draft(context_id, draft_id)
    assert draft["state"] == "processing"
    assert draft.get("cleanup_authorization") is None


def test_cleanup_child_page_methods_cap_reads_and_writes_at_four_hundred(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, _operation_id, certificate = _seed_completion_records(client)
    assert repo.commit_completion(certificate) is True
    assert repo.authorize_cleanup(context_id, certificate) is True
    draft_ref = client.collection("telegram_sessions").document(context_id).collection("drafts").document(draft_id)
    for index in range(405):
        draft_ref.collection("messages").document(str(index + 1)).set({
            "message_id": index + 1, "intake_sequence": index + 1,
        })
    for index in range(405):
        draft_ref.collection("prepared").document(str(index)).set({"order_index": index})

    source_page, _ = repo.list_manifest_pages(draft_id, context_id=context_id, limit=999)
    prepared_page = repo.list_prepared_pages(context_id, draft_id, limit=999)
    assert len(source_page) == len(prepared_page) == 400
    repo.delete_draft_page(draft_id, source_page, context_id=context_id)
    repo.delete_prepared_page(context_id, draft_id, prepared_page)
    remaining_source, _ = repo.list_manifest_pages(draft_id, context_id=context_id, limit=999)
    remaining_prepared = repo.list_prepared_pages(context_id, draft_id, limit=999)
    assert len(remaining_source) == len(remaining_prepared) == 5


def _pending_clarification(context_id="42_5", operation_id="op-origin", **overrides):
    clarification_id = build_clarification_id(context_id, operation_id)
    prefix = f"telegram_sessions/{context_id}/clarifications/{clarification_id}/"
    data = {
        "clarification_id": clarification_id,
        "context_id": context_id,
        "actor_id": 77,
        "initiating_operation_id": operation_id,
        "origin_conversation_id": f"{context_id}-1",
        "original_command_ref": f"telegram_sessions/{context_id}/commands/command-1",
        "original_command_digest": hashlib.sha256(b"command").hexdigest(),
        "action": "move_transcript",
        "resolved_parameters": {"doc_id": "doc-1"},
        "missing_slots": ("destination",),
        "evidence_snapshot_ref": f"telegram_sessions/{context_id}/operations/{operation_id}/evidence",
        "evidence_snapshot_digest": hashlib.sha256(b"evidence").hexdigest(),
        "generation": 1,
        "revision": 1,
        "state": ClarificationState.WAITING,
        "execution_operation_id": None,
        "execution_plan_ref": None,
        "execution_plan_digest": None,
        "result_ref": None,
        "safe_error": None,
        "questions_ref": prefix + "questions",
        "answers_ref": prefix + "answers",
        "created_at": NOW,
        "updated_at": NOW,
        "protocol": "clarification-v1",
    }
    data.update(overrides)
    return PendingClarification(**data)


def test_pending_clarification_validates_roundtrips_and_freezes_parameters():
    mutable_path = ["Archive", "2026"]
    clarification = _pending_clarification(
        action="create_then_move",
        resolved_parameters={"doc_id": "doc-1", "parent_path": mutable_path, "name": "  "},
        missing_slots=("folder_name",),
    )
    mutable_path.append("mutated")
    assert clarification.resolved_parameters["parent_path"] == ("Archive", "2026")
    with pytest.raises(TypeError):
        clarification.resolved_parameters["doc_id"] = "doc-2"
    encoded = clarification.to_dict()
    encoded["resolved_parameters"]["parent_path"].append("outside")
    assert clarification.resolved_parameters["parent_path"] == ("Archive", "2026")
    assert PendingClarification.from_dict(clarification.to_dict()) == clarification
    assert clarification.to_dict()["resolved_parameters"] == {
        "doc_id": "doc-1", "parent_path": ["Archive", "2026"],
    }
    assert build_clarification_id("42_5", "op-origin") == clarification.clarification_id
    with pytest.raises(ValueError, match="resolved_parameters"):
        _pending_clarification(
            resolved_parameters={"doc_id": "doc-1"},
            missing_slots=("document", "destination"),
        )
    missing_name = _pending_clarification(
        action="create_folder", resolved_parameters={"parent_id": "folder-parent", "name": None},
        missing_slots=(),
    )
    assert missing_name.missing_slots == ("folder_name",)
    assert "name" not in missing_name.resolved_parameters


@pytest.mark.parametrize("change", [
    {"actor_id": True},
    {"context_id": "foreign/context"},
    {"original_command_digest": "A" * 64},
    {"missing_slots": ("unknown",)},
    {"state": "waiting", "missing_slots": ()},
    {"updated_at": NOW - timedelta(seconds=1)},
    {"original_command_ref": "telegram_sessions/other/commands/command-1"},
    {"action": "create_folder", "resolved_parameters": {"doc_id": "doc-1"}},
    {"execution_operation_id": "op-origin", "execution_plan_ref": "plan/1",
     "execution_plan_digest": hashlib.sha256(b"plan").hexdigest()},
])
def test_pending_clarification_rejects_invalid_fields(change):
    with pytest.raises(ValueError, match="invalid clarification"):
        _pending_clarification(**change)


def test_pending_clarification_requires_tagged_v1_and_explicit_persisted_counters():
    data = _pending_clarification().to_dict()
    with pytest.raises(ValueError, match="protocol"):
        PendingClarification.from_dict({key: value for key, value in data.items() if key != "protocol"})
    with pytest.raises(ValueError, match="protocol"):
        PendingClarification.from_dict({**data, "protocol": "clarification-v2"})
    for field in ("generation", "revision"):
        with pytest.raises(ValueError, match="fields"):
            PendingClarification.from_dict({key: value for key, value in data.items() if key != field})


def test_pending_clarification_execution_binding_requires_fixed_write_parameters():
    plan_digest = hashlib.sha256(b"plan").hexdigest()
    with pytest.raises(ValueError, match="resolved_parameters"):
        _pending_clarification(
            state=ClarificationState.CLAIMED,
            missing_slots=(), resolved_parameters={"doc_id": "doc-1"},
            execution_operation_id="answer-op", execution_plan_ref="plan/answer-op",
            execution_plan_digest=plan_digest,
        )
    resolved = _pending_clarification(
        action="create_folder", state=ClarificationState.CLAIMED, missing_slots=(),
        resolved_parameters={"parent_id": "folder-parent", "name": "Archive"},
        execution_operation_id="answer-op", execution_plan_ref="plan/answer-op",
        execution_plan_digest=plan_digest,
    )
    assert resolved.execution_operation_id == "answer-op"


@pytest.mark.parametrize("state", list(ClarificationState))
def test_pending_clarification_state_binding_rules(state):
    digest = hashlib.sha256(b"plan").hexdigest()
    binding = {
        "execution_operation_id": "answer-op",
        "execution_plan_ref": "plan/answer-op",
        "execution_plan_digest": digest,
    }
    params = {"doc_id": "doc-1", "destination_id": "folder-1"}
    if state is ClarificationState.WAITING:
        record = _pending_clarification()
        assert record.execution_operation_id is None
    elif state in {ClarificationState.CANCELLED, ClarificationState.REPLACED}:
        record = _pending_clarification(state=state, missing_slots=())
        with pytest.raises(ValueError, match="execution_binding"):
            _pending_clarification(state=state, missing_slots=(), **binding)
    elif state is ClarificationState.FAILED:
        before_claim = _pending_clarification(state=state, missing_slots=())
        after_claim = _pending_clarification(
            state=state, missing_slots=(), resolved_parameters=params, **binding,
        )
        assert before_claim.execution_operation_id is None
        assert after_claim.execution_operation_id == "answer-op"
        record = after_claim
    else:
        record = _pending_clarification(
            state=state, missing_slots=(), resolved_parameters=params, **binding,
        )
    assert record.state is state
    if state in {ClarificationState.CLAIMED, ClarificationState.COMPLETED,
                 ClarificationState.UNKNOWN}:
        with pytest.raises(ValueError, match="execution_binding"):
            _pending_clarification(state=state, missing_slots=())
        with pytest.raises(ValueError, match="execution_operation_id"):
            _pending_clarification(
                state=state, missing_slots=(), resolved_parameters=params,
                **{**binding, "execution_operation_id": "op-origin"},
            )


def test_pending_clarification_constructor_and_optional_persisted_defaults():
    raw = _pending_clarification(state=ClarificationState.CANCELLED, missing_slots=()).to_dict()
    for field in ("generation", "revision", "execution_operation_id", "execution_plan_ref",
                  "execution_plan_digest", "result_ref", "safe_error", "questions_ref", "answers_ref"):
        raw.pop(field)
    raw.pop("protocol")
    constructed = PendingClarification(**raw)
    assert constructed.generation == constructed.revision == 1
    assert constructed.execution_operation_id is None
    assert constructed.questions_ref.endswith("/questions")
    assert constructed.answers_ref.endswith("/answers")

    persisted = _pending_clarification(state=ClarificationState.CANCELLED, missing_slots=()).to_dict()
    for field in ("execution_operation_id", "execution_plan_ref", "execution_plan_digest",
                  "result_ref", "safe_error", "questions_ref", "answers_ref"):
        persisted.pop(field)
    restored = PendingClarification.from_dict(persisted)
    assert restored.generation == restored.revision == 1
    assert restored.execution_operation_id is None


def test_pending_clarification_safe_error_and_utc_timestamp_validation():
    from datetime import timedelta as _timedelta

    offset = timezone(_timedelta(hours=3))
    record = _pending_clarification(
        safe_error="Папка недоступна для изменения",
        created_at=NOW.astimezone(offset), updated_at=NOW.astimezone(offset),
    )
    assert record.safe_error == "Папка недоступна для изменения"
    assert record.created_at.utcoffset() == timedelta(0)
    for value in ("", " \t ", "x" * 501):
        with pytest.raises(ValueError, match="safe_error"):
            _pending_clarification(safe_error=value)
    for field in ("generation", "revision"):
        with pytest.raises(ValueError, match=field):
            _pending_clarification(**{field: True})
    with pytest.raises(ValueError, match="digest"):
        _pending_clarification(evidence_snapshot_digest="a" * 63)
    with pytest.raises(ValueError, match="timestamps"):
        _pending_clarification(created_at=NOW.replace(tzinfo=None))


def test_clarification_addressed_read_uses_pointer_and_legacy_defaults(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    assert repo.get_context("42_5") is None
    assert repo.get_pending_clarification("42_5", actor_id=77) is None
    context_ref = client.collection("telegram_sessions").document("42_5")
    context_ref.set({"chat_id": 42, "thread_id": 5})
    assert repo.get_context("42_5")["pending_clarification_id"] is None
    assert repo.get_context("42_5")["clarification_generation"] == 0
    assert repo.get_pending_clarification("42_5", actor_id=77) is None

    clarification = _pending_clarification()
    context_ref.update({"pending_clarification_id": clarification.clarification_id,
                        "clarification_generation": clarification.generation})
    context_ref.collection("clarifications").document(clarification.clarification_id).set(
        clarification.to_dict(),
    )
    assert repo.get_pending_clarification("42_5", actor_id=77) == clarification
    assert repo.get_clarification("42_5", clarification.clarification_id, actor_id=77) == clarification
    assert repo.get_clarification("42_5", clarification.clarification_id, actor_id=88) is None
    assert repo.get_clarification("foreign", clarification.clarification_id, actor_id=77) is None
    assert repo.get_clarification("42_5", "untagged", actor_id=77) is None
    context_ref.collection("clarifications").document("untagged").set({"context_id": "42_5"})
    context_ref.update({"clarification_generation": 2})
    assert repo.get_pending_clarification("42_5", actor_id=77) is None


@pytest.mark.parametrize("state", [
    ClarificationState.CLAIMED, ClarificationState.COMPLETED,
    ClarificationState.CANCELLED, ClarificationState.REPLACED,
    ClarificationState.FAILED, ClarificationState.UNKNOWN,
])
def test_addressed_clarification_reader_returns_terminal_records(firestore_fake, state):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id = f"42_{state.value}"
    operation_id = f"origin-{state.value}"
    binding = {}
    params = {"doc_id": "doc-1", "destination_id": "folder-1"}
    if state in {ClarificationState.CLAIMED, ClarificationState.COMPLETED,
                 ClarificationState.UNKNOWN}:
        binding = {
            "execution_operation_id": f"answer-{state.value}",
            "execution_plan_ref": f"plan/answer-{state.value}",
            "execution_plan_digest": hashlib.sha256(b"plan").hexdigest(),
        }
    record = _pending_clarification(
        context_id=context_id, operation_id=operation_id, state=state,
        missing_slots=(), resolved_parameters=params if binding else {}, **binding,
    )
    ref = client.collection("telegram_sessions").document(context_id)
    ref.set({"pending_clarification_id": record.clarification_id,
             "clarification_generation": record.generation})
    ref.collection("clarifications").document(record.clarification_id).set(record.to_dict())
    assert repo.get_clarification(context_id, record.clarification_id, actor_id=77) == record
    assert repo.get_pending_clarification(context_id, actor_id=77) is None


def test_addressed_clarification_reader_fails_closed_for_tagged_rows(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    valid = _pending_clarification()
    row = client.collection("telegram_sessions").document(valid.context_id).collection(
        "clarifications",
    ).document(valid.clarification_id)
    malformed = valid.to_dict()
    malformed.pop("revision")
    row.set(malformed)
    with pytest.raises(ValueError, match="fields"):
        repo.get_clarification(valid.context_id, valid.clarification_id, actor_id=77)
    row.set({**valid.to_dict(), "protocol": "clarification-v9"})
    with pytest.raises(ValueError, match="protocol"):
        repo.get_clarification(valid.context_id, valid.clarification_id, actor_id=77)
    row.set({"context_id": valid.context_id, "clarification_id": valid.clarification_id,
             "actor_id": 77})
    assert repo.get_clarification(valid.context_id, valid.clarification_id, actor_id=77) is None


def test_pending_clarifications_are_independent_across_contexts(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    records = [
        _pending_clarification(context_id="42_5", operation_id="op-one"),
        _pending_clarification(context_id="42_6", operation_id="op-two"),
    ]
    for record in records:
        context = client.collection("telegram_sessions").document(record.context_id)
        context.set({"pending_clarification_id": record.clarification_id,
                     "clarification_generation": record.generation})
        context.collection("clarifications").document(record.clarification_id).set(
            record.to_dict(),
        )
    assert repo.get_pending_clarification("42_5", actor_id=77) == records[0]
    assert repo.get_pending_clarification("42_6", actor_id=77) == records[1]
    client.collection("telegram_sessions").document("42_6").update(
        {"clarification_generation": 2},
    )
    assert repo.get_pending_clarification("42_6", actor_id=77) is None
    assert repo.get_pending_clarification("42_5", actor_id=77) == records[0]


def _running_accepted_command(repo, client, *, actor_id=77):
    accepted = repo.accept_command(
        1, chat_id=42, user_id=actor_id, thread_id=5,
        command=TelegramCommand("freeform", "move the discussed transcript"), now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="clarification-worker", now=NOW)
    assert lease is not None and lease.operation_id == accepted.operation_id
    return accepted, context_ref, lease


def _run_clarification_guard(repo, client, *, context_id, operation_id, actor_id=77,
                             owner="clarification-worker", fence=1, now=NOW,
                             clarification_id=None, generation=None, revision=None):
    @firestore.transactional
    def _read(transaction):
        return repo._read_clarification_guard(
            transaction, context_id=context_id, operation_id=operation_id,
            actor_id=actor_id, owner=owner, fence=fence, now=now,
            clarification_id=clarification_id, expected_generation=generation,
            expected_revision=revision,
        )
    return _read(client.transaction())


def test_clarification_guard_reads_real_running_command_without_writes(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted, context_ref, lease = _running_accepted_command(repo, client)
    context_ref.update({"clarification_generation": 5})
    before = dict(firestore_fake.documents)
    creation_guard = _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence,
    )
    assert creation_guard is not None
    assert creation_guard.clarification is None
    assert creation_guard.accepted_command["user_id"] == 77
    assert creation_guard.context["clarification_generation"] == 5
    assert firestore_fake.documents == before

    clarification = _pending_clarification(operation_id=accepted.operation_id)
    context_ref.update({"pending_clarification_id": clarification.clarification_id,
                        "clarification_generation": clarification.generation})
    context_ref.collection("clarifications").document(clarification.clarification_id).set(
        clarification.to_dict(),
    )
    guarded = _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence, clarification_id=clarification.clarification_id,
        generation=1, revision=1,
    )
    assert guarded is not None and guarded.clarification == clarification
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        actor_id=88, fence=lease.fence,
    ) is None
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence + 1,
    ) is None
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence, clarification_id=clarification.clarification_id,
        generation=2, revision=1,
    ) is None
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence, clarification_id=clarification.clarification_id,
        generation=1, revision=2,
    ) is None

    for kwargs in (
        {"owner": "other-worker"},
        {"operation_id": "operation-not-current"},
    ):
        before_failure = dict(firestore_fake.documents)
        assert _run_clarification_guard(
            repo, client, context_id=accepted.context_id,
            operation_id=kwargs.get("operation_id", accepted.operation_id),
            owner=kwargs.get("owner", "clarification-worker"), fence=lease.fence,
            clarification_id=clarification.clarification_id, generation=1, revision=1,
        ) is None
        assert firestore_fake.documents == before_failure

    operation = context_ref.collection("operations").document(accepted.operation_id).get().to_dict()
    command_ref = client.collection("telegram_sessions").document(accepted.context_id)
    command_ref = command_ref.collection("commands").document(operation["input_ref"].split("/")[-1])
    command_ref.update({"user_id": 88})
    before_actor_failure = dict(firestore_fake.documents)
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence,
    ) is None
    assert firestore_fake.documents == before_actor_failure


def test_clarification_guard_requires_transaction(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    with pytest.raises(ValueError, match="requires a transaction"):
        repo._read_clarification_guard(
            None, context_id="42_5", operation_id="op-1", actor_id=77,
            owner="worker", fence=1, now=NOW,
        )


def test_clarification_guard_preserves_non_identifier_queue_owner(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_command(
        1, chat_id=42, user_id=77, thread_id=5,
        command=TelegramCommand("freeform", "move the discussed transcript"), now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="worker:one", now=NOW)
    assert lease is not None
    guard = _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner="worker:one", fence=lease.fence,
    )
    assert guard is not None
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner="worker:other", fence=lease.fence,
    ) is None


def test_answer_guard_runs_after_real_question_operation_completion(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    original = repo.accept_command(
        1, chat_id=42, user_id=77, thread_id=5,
        command=TelegramCommand("freeform", "move the discussed transcript"), now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(original.context_id)
    original_lease = claim_head(client, context_ref, owner="clarification-worker", now=NOW)
    assert original_lease is not None
    assert complete_head(
        client, context_ref, operation_id=original.operation_id,
        owner=original_lease.owner, fence=original_lease.fence, now=NOW,
    )
    clarification = _pending_clarification(operation_id=original.operation_id)
    context_ref.update({"pending_clarification_id": clarification.clarification_id,
                        "clarification_generation": clarification.generation})
    context_ref.collection("clarifications").document(clarification.clarification_id).set(
        clarification.to_dict(),
    )
    answer = repo.accept_command(
        2, chat_id=42, user_id=77, thread_id=5,
        command=TelegramCommand("freeform", "the second option"), now=NOW,
    )
    answer_lease = claim_head(
        client, context_ref, owner="clarification-worker", now=NOW,
    )
    assert answer_lease is not None and answer_lease.operation_id == answer.operation_id
    before = dict(firestore_fake.documents)
    guarded = _run_clarification_guard(
        repo, client, context_id=answer.context_id, operation_id=answer.operation_id,
        fence=answer_lease.fence, clarification_id=clarification.clarification_id,
        generation=clarification.generation, revision=clarification.revision,
    )
    assert guarded is not None and guarded.clarification == clarification
    assert firestore_fake.documents == before


def test_clarification_guard_rejects_expired_lease_and_existing_waiting_slot(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted, context_ref, lease = _running_accepted_command(repo, client)
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence, now=lease.expires_at,
    ) is None
    clarification = _pending_clarification(operation_id=accepted.operation_id)
    context_ref.update({"pending_clarification_id": clarification.clarification_id,
                        "clarification_generation": clarification.generation})
    assert _run_clarification_guard(
        repo, client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        fence=lease.fence,
    ) is None
