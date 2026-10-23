from datetime import datetime, timezone

import pytest

from services.connectors.telegram.models import PreparedUtterance
from services.connectors.telegram.repository import TelegramRepository
from services.connectors.telegram.save import TelegramCompletionCertificate


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
        "state": "processing",
        "active_save_operation_id": operation_id,
        "immutable_save_intent_id": intent_id,
        "manifest_ref": f"telegram_sessions/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        "manifest_digest": digest,
        "ordered_message_ids": [101, 102],
        "doc_id": doc_id,
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


def test_completion_cleanup_authorization_requires_exact_shared_bindings(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    context_id, draft_id, operation_id, certificate = _seed_completion_records(client)

    assert repo.commit_completion(certificate) is True
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
