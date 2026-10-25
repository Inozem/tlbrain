"""Manual retry and preservation contracts for failed Telegram drafts."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.integrations.firestore.operations import (
    OPERATIONS_COLLECTION,
    get_operation as get_shared_operation,
    get_stage,
)
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from core.integrations.firestore.operation_queue import (
    claim_head,
    complete_head,
    get_operation as get_queued_operation,
)
from core.operations.models import OperationOrigin, OperationState
from core.operations.retries import RetryBudgetExceeded, RetryGovernor
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.models import PreparedUtterance
from services.connectors.telegram.repository import SESSIONS, TelegramRepository
from services.connectors.telegram.save import TelegramCompletionCertificate


pytestmark = pytest.mark.offline

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _value(record, field):
    return record[field] if isinstance(record, dict) else getattr(record, field)


def _document(client, path):
    parts = path.split("/")
    reference = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        reference = reference.collection(parts[index]).document(parts[index + 1])
    return reference


def _seed_failed_draft(firestore_fake, *, update_id=12):
    """Persist a sealed save, then model its terminal failure without providers."""
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    forward = repository.accept_forward(
        update_id,
        chat_id=42,
        thread_id=11,
        message={
            "message_id": update_id + 100,
            "text": f"source for dialog {update_id}",
            "transcription_state": "pending",
        },
        now=NOW,
    )
    context_id, draft_id = forward.context_id, forward.draft_id
    message_id = update_id + 100
    repository.checkpoint_transcription(
        context_id, draft_id, message_id, f"transcript for {update_id}"
    )
    repository.save_prepared_utterances(
        context_id,
        draft_id,
        [PreparedUtterance(0, "speaker", None, f"prepared {update_id}", 2, (message_id,))],
        required_audio_count=1,
        completed_audio_ids=(message_id,),
        context_capacity=30,
        preparation_digest=f"preparation-{update_id}",
        now=NOW,
    )
    accepted = repository.accept_command(
        update_id + 1000,
        chat_id=42,
        thread_id=11,
        user_id=7,
        command=parse_command("save"),
        now=NOW,
    )
    assert accepted.draft_id == draft_id
    old_operation_id = accepted.operation_id
    context_ref = client.collection(SESSIONS).document(context_id)
    lease = claim_head(client, context_ref, owner="failed-draft-seal", now=NOW)
    assert lease is not None
    assert repository.confirm_save_command(
        context_id, old_operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    draft_ref = (
        client.collection(SESSIONS).document(context_id)
        .collection("drafts").document(draft_id)
    )
    draft = draft_ref.get().to_dict()
    intent_id = draft["immutable_save_intent_id"]
    # Keep a pre-existing document identity to prove retry does not allocate a
    # second document or replace the immutable save intent.
    draft_ref.update({"state": "failed", "doc_id": f"doc-{update_id}"})
    client.collection(SESSIONS).document(context_id).collection("operations").document(
        old_operation_id
    ).update({"state": "failed", "finished_at": NOW})
    client.collection(OPERATIONS_COLLECTION).document(old_operation_id).set({
        "operation_id": old_operation_id,
        "origin": OperationOrigin.TELEGRAM.value,
        "origin_operation_ref": f"{SESSIONS}/{context_id}/operations/{old_operation_id}",
        "caller_binding": f"telegram:{context_id}",
        "immutable_input_digest": f"digest-{old_operation_id}",
        "immutable_input_ref": f"telegram-draft:{context_id}:{draft_id}",
        "save_intent_id": intent_id,
        "doc_id": f"doc-{update_id}",
        "state": OperationState.FAILED.value,
        "created_at": NOW,
        "updated_at": NOW,
        "status_updated_at": NOW,
    })
    client.collection(SAVE_INTENTS_COLLECTION).document(intent_id).set({
        "save_intent_id": intent_id,
        "immutable_input_digest": f"digest-{old_operation_id}",
        "desired_content_hash": f"content-{old_operation_id}",
        "desired_parent_id_hash": None,
        "desired_parent_id": None,
        "immutable_title_digest": None,
        "immutable_provider_digest": None,
        "state": "confirmed",
        "create_outcome": "confirmed",
        "dispatch_token": f"dispatch-{old_operation_id}",
        "dispatch_operation_id": old_operation_id,
        "dispatch_owner": "original-save-worker",
        "dispatch_fence": 1,
        "doc_id": f"doc-{update_id}",
        "confirmed_result": {
            "document_id": f"doc-{update_id}",
            "title": f"doc-{update_id}",
            "resource_url": None,
        },
        "fill_checkpoint_ref": f"save_checkpoints/{old_operation_id}/fill",
        "index_checkpoint_ref": None,
        "created_at": NOW,
        "updated_at": NOW,
    })
    context_ref = client.collection(SESSIONS).document(context_id)
    context = context_ref.get().to_dict()
    context.update({
        "active_draft_id": None,
        "draft_status": None,
        "active_operation_id": None,
        "reserved_operation_id": None,
        "head_sequence": None,
        "queue_tail_operation_id": None,
        "lock_owner": None,
        "lock_expires_at": None,
    })
    context_ref.set(context)
    return repository, client, context_id, draft_id, old_operation_id, intent_id


def _retry(repository, context_id, dialog_number, *, update_id=900):
    service_type = importlib.import_module(
        "services.connectors.telegram.retry"
    ).TelegramRetryService
    return service_type(repository).retry_failed_draft(
        context_id, dialog_number, update_id=update_id, now=NOW,
    )


def test_forward_after_failed_draft_allocates_next_clean_dialog_and_preserves_old_progress(
    firestore_fake,
):
    repository, _client, context_id, old_draft_id, _old_operation_id, _intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    old_draft_before = repository.get_draft(context_id, old_draft_id)
    old_messages_before = repository.list_messages(context_id, old_draft_id)

    following = repository.accept_forward(
        30,
        chat_id=42,
        thread_id=11,
        message={"message_id": 130, "text": "next dialog source"},
        now=NOW,
    )

    assert following.accepted is True
    assert following.dialog_number == 2
    assert following.draft_id != old_draft_id
    assert repository.get_draft(context_id, old_draft_id) == old_draft_before
    assert repository.list_messages(context_id, old_draft_id) == old_messages_before
    assert repository.get_draft(context_id, old_draft_id)["state"] == "failed"
    assert repository.get_draft(context_id, old_draft_id)["prepared_count"] == 1
    assert repository.list_messages(context_id, old_draft_id)[0]["transcription_text"] == (
        "transcript for 12"
    )
    assert repository.get_context(context_id)["active_draft_id"] == following.draft_id
    assert repository.get_draft(context_id, following.draft_id)["prepared_count"] == 0


def test_manual_retry_reuses_failed_draft_intent_manifest_and_completed_progress(firestore_fake):
    repository, client, context_id, draft_id, old_operation_id, intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    before = repository.get_draft(context_id, draft_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    intent_before = intent_ref.get().to_dict()
    manifest = _document(client, before["manifest_ref"]).get().to_dict()
    messages = repository.list_messages(context_id, draft_id)

    retried = _retry(repository, context_id, before["dialog_number"])

    assert _value(retried, "accepted") is True
    assert _value(retried, "draft_id") == draft_id
    assert _value(retried, "dialog_number") == before["dialog_number"]
    assert _value(retried, "operation_id") != old_operation_id
    after = repository.get_draft(context_id, draft_id)
    assert intent_ref.get().to_dict() == intent_before
    assert after["state"] == "processing"
    assert after["active_save_operation_id"] == _value(retried, "operation_id")
    assert after["immutable_save_intent_id"] == intent_id
    assert after["doc_id"] == before["doc_id"]
    assert after["manifest_ref"].endswith(
        f"/save_manifests/{_value(retried, 'operation_id')}"
    )
    assert after["manifest_digest"] == before["manifest_digest"]
    assert after["ordered_message_ids"] == before["ordered_message_ids"] == [112]
    assert after["completed_audio_ids"] == before["completed_audio_ids"] == [112]
    assert after["prepared_count"] == before["prepared_count"] == 1
    assert _document(client, before["manifest_ref"]).get().to_dict() == manifest
    retry_manifest = _document(client, after["manifest_ref"]).get().to_dict()
    assert retry_manifest["draft_id"] == draft_id
    assert retry_manifest["save_operation_id"] == _value(retried, "operation_id")
    assert retry_manifest["manifest_digest"] == manifest["manifest_digest"]
    assert retry_manifest["ordered_message_ids"] == manifest["ordered_message_ids"] == [112]
    assert repository.list_messages(context_id, draft_id) == messages
    new_queued = get_queued_operation(
        client.collection(SESSIONS).document(context_id), _value(retried, "operation_id")
    )
    new_shared = get_shared_operation(client, _value(retried, "operation_id"))
    assert new_queued is not None and new_queued.state.value == "pending"
    assert new_shared is not None and new_shared.save_intent_id == intent_id
    assert new_shared.doc_id == before["doc_id"]
    assert _value(retried, "outbox_id")
    assert client.collection("task_outbox").document(_value(retried, "outbox_id")).get().exists


def test_manual_retry_replay_for_same_update_adds_no_records(firestore_fake):
    repository, _client, context_id, _draft_id, _old_operation_id, _intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    first = _retry(repository, context_id, 1, update_id=901)
    paths_after_first = set(firestore_fake.documents)
    documents_after_first = dict(firestore_fake.documents)

    replay = _retry(repository, context_id, 1, update_id=901)

    assert replay == first
    assert set(firestore_fake.documents) == paths_after_first
    assert firestore_fake.documents == documents_after_first


def test_unknown_dialog_retry_allocates_no_operation_or_outbox_and_preserves_selection(
    firestore_fake,
):
    repository, client, context_id, _draft_id, _old_operation_id, _intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    current = repository.accept_forward(
        31,
        chat_id=42,
        thread_id=11,
        message={"message_id": 131, "text": "current collecting source"},
        now=NOW,
    )
    before_context = repository.get_context(context_id)
    before_paths = set(firestore_fake.documents)

    rejected = _retry(repository, context_id, 99)

    assert _value(rejected, "accepted") is False
    assert _value(rejected, "operation_id") is None
    assert _value(rejected, "outbox_id") is None
    assert repository.get_context(context_id) == before_context
    assert repository.get_context(context_id)["active_draft_id"] == current.draft_id
    assert set(firestore_fake.documents) == before_paths


def test_retry_without_durable_save_intent_is_write_free(firestore_fake):
    repository, client, context_id, draft_id, _old_operation_id, intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    client.collection(SAVE_INTENTS_COLLECTION).document(intent_id).delete()
    draft_before = repository.get_draft(context_id, draft_id)
    context_before = repository.get_context(context_id)
    documents_before = dict(firestore_fake.documents)

    rejected = _retry(repository, context_id, 1, update_id=902)

    assert _value(rejected, "accepted") is False
    assert _value(rejected, "operation_id") is None
    assert _value(rejected, "outbox_id") is None
    assert firestore_fake.documents == documents_before
    assert repository.get_draft(context_id, draft_id) == draft_before
    assert repository.get_context(context_id) == context_before
    assert client.collection(SESSIONS).document(context_id).collection(
        "commands"
    ).document("902").get().exists is False
    assert client.collection(SESSIONS).document(context_id).collection(
        "updates"
    ).document("902").get().exists is False


def test_manual_retry_gets_a_fresh_three_failure_stage_budget(firestore_fake):
    repository, client, context_id, draft_id, old_operation_id, _intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    retried = _retry(repository, context_id, 1)
    operation_id = _value(retried, "operation_id")
    assert operation_id != old_operation_id
    governor = RetryGovernor(client, clock=SimpleNamespace(now=lambda: NOW))
    failed_calls = []

    for attempt in range(1, 4):
        with pytest.raises(ConnectionError, match="synthetic retry-stage failure"):
            governor.run_stage(
                operation_id,
                "save/fill",
                lambda: failed_calls.append(operation_id)
                or (_ for _ in ()).throw(ConnectionError("synthetic retry-stage failure")),
                now=NOW,
            )
        assert get_stage(client, operation_id, "save/fill").failures == attempt

    with pytest.raises(RetryBudgetExceeded):
        governor.run_stage(
            operation_id,
            "save/fill",
            lambda: failed_calls.append(operation_id),
            now=NOW,
        )

    assert failed_calls == [operation_id] * 3
    assert get_stage(client, operation_id, "save/fill").attempts_started == 3
    assert get_shared_operation(client, old_operation_id).state is OperationState.FAILED
    assert repository.get_draft(context_id, draft_id)["prepared_count"] == 1


def test_old_retry_completion_cannot_replace_a_newer_collecting_draft(firestore_fake):
    repository, client, context_id, old_draft_id, _old_operation_id, _intent_id = (
        _seed_failed_draft(firestore_fake)
    )
    newer = repository.accept_forward(
        32,
        chat_id=42,
        thread_id=11,
        message={"message_id": 132, "text": "dialog after retry"},
        now=NOW,
    )
    assert newer.dialog_number == 2
    newer_messages = repository.list_messages(context_id, newer.draft_id)
    retried = _retry(repository, context_id, 1)
    retry_operation_id = _value(retried, "operation_id")
    retry_draft = repository.get_draft(context_id, old_draft_id)
    intent_id = retry_draft["immutable_save_intent_id"]
    manifest_digest = retry_draft["manifest_digest"]
    binding = {
        "context_id": context_id,
        "draft_id": old_draft_id,
        "manifest_digest": manifest_digest,
    }
    # Model the persisted success evidence produced by the retry's save path.
    # The retry uses an operation-bound manifest with the same sealed source IDs.
    manifest_ref = (
        f"{SESSIONS}/{context_id}/drafts/{old_draft_id}/save_manifests/"
        f"{retry_operation_id}"
    )
    previous_manifest = _document(client, retry_draft["manifest_ref"]).get().to_dict()
    retry_manifest = {
        **previous_manifest,
        "draft_id": old_draft_id,
        "save_operation_id": retry_operation_id,
        "manifest_digest": manifest_digest,
        "ordered_message_ids": retry_draft["ordered_message_ids"],
    }
    _document(client, manifest_ref).set(retry_manifest)
    _document(
        client, f"{SESSIONS}/{context_id}/drafts/{old_draft_id}"
    ).update({"manifest_ref": manifest_ref})
    doc_id = retry_draft["doc_id"]
    certificate_id = f"certificate-{retry_operation_id}"
    shared_certificate = {
        "certificate_id": certificate_id,
        "save_intent_id": intent_id,
        "operation_id": retry_operation_id,
        "doc_id": doc_id,
        "origin_binding": binding,
        "receipt_ref": f"index_completions/{doc_id}",
        "receipt_version": 1,
    }
    cleanup_authorization = {
        "certificate_id": certificate_id,
        "save_intent_id": intent_id,
        "operation_id": retry_operation_id,
        "origin_binding": binding,
    }
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    intent_data = intent_ref.get().to_dict() or {"save_intent_id": intent_id}
    intent_data.update({
        "operation_id": retry_operation_id,
        "completion_certificate": shared_certificate,
        "cleanup_authorization": cleanup_authorization,
    })
    intent_ref.set(intent_data)
    operation_ref = client.collection(OPERATIONS_COLLECTION).document(retry_operation_id)
    operation_data = operation_ref.get().to_dict() or {"operation_id": retry_operation_id}
    operation_data.update({
        "save_intent_id": intent_id,
        "state": "completed",
        "doc_id": doc_id,
        "cleanup_authorization": cleanup_authorization,
    })
    operation_ref.set(operation_data)
    certificate = TelegramCompletionCertificate(
        context_id=context_id,
        draft_id=old_draft_id,
        manifest_digest=manifest_digest,
        save_intent_id=intent_id,
        operation_id=retry_operation_id,
        receipt_ref=shared_certificate["receipt_ref"],
        receipt_version=1,
        receipt_digest=f"receipt-{retry_operation_id}",
        origin_binding=(context_id, old_draft_id, manifest_digest),
    )
    assert repository.commit_completion(certificate) is True
    assert repository.authorize_cleanup(context_id, certificate) is True
    assert repository.finalize_cleanup(context_id, certificate) is True

    context_ref = client.collection(SESSIONS).document(context_id)
    lease = claim_head(client, context_ref, owner="failed-draft-test", now=NOW)
    assert lease is not None and lease.operation_id == retry_operation_id

    assert complete_head(
        client,
        context_ref,
        operation_id=retry_operation_id,
        owner=lease.owner,
        fence=lease.fence,
        now=NOW,
    ) is True

    context = repository.get_context(context_id)
    assert context["active_draft_id"] == newer.draft_id
    assert context["active_conversation_id"] == f"{context_id}-2"
    assert context["selected_dialog_number"] == 2
    assert repository.get_draft(context_id, old_draft_id)["state"] == "cleaned"
    assert repository.get_draft(context_id, old_draft_id)["dialog_number"] == 1
    assert repository.get_draft(context_id, newer.draft_id)["state"] == "collecting"
    assert repository.list_messages(context_id, newer.draft_id) == newer_messages

