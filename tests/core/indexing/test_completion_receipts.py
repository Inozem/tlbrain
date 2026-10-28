"""Offline contracts for generation receipts and guarded document completion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest


DOC_ID = "completion-doc"
PARENT_ID = "completion-parent"
ROOT_ID = "completion-root"
OPERATION_ID = "completion-operation"
OWNER = "completion-worker"
NOW = datetime(2026, 3, 4, tzinfo=timezone.utc)
RESOURCE = f"document:{DOC_ID}"


def _lease(client, *, operation_id=OPERATION_ID, now=NOW, lease_seconds=30, owner=OWNER):
    from core.integrations.firestore.operations import claim_operation, create_operation
    from core.operations.models import OperationOrigin

    create_operation(
        client,
        operation_id=operation_id,
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:synthetic-input",
        immutable_input_ref="synthetic-input-ref",
        now=now,
    )
    lease = claim_operation(
        client, operation_id, owner=owner, now=now, lease_seconds=lease_seconds,
    )
    assert lease is not None
    return lease


def _completion_module():
    # Function-local so this test module still collects before T042 adds the seam.
    from core.dialogs import completion

    return completion


def _stored_generation(firestore_fake, doc_id: str, content_hash: str):
    path = f"index_completions/{doc_id}/generations/{content_hash}"
    return firestore_fake.documents.get(path)


def _set_generation(client, doc_id: str, content_hash: str, receipt: dict):
    (client.collection("index_completions").document(doc_id)
     .collection("generations").document(content_hash).set(receipt))


def _assert_user_fact_excluded(delete_events):
    from qdrant_client.models import Filter, MatchAny

    assert delete_events
    for event in delete_events:
        selector = event["points_selector"]
        assert isinstance(selector, Filter)
        type_condition = next(condition for condition in selector.must if condition.key == "type")
        match = type_condition.match
        selected_types = set(match.any) if isinstance(match, MatchAny) else {match.value}
        assert selected_types <= {"utterance", "summary", "fact"}
        assert "user_fact" not in selected_types


def _save_fixture(
    client, sdk_fakes, *, operation_id=OPERATION_ID, owner=OWNER, now=NOW,
    lease_seconds=30, skipped_utterances=(),
):
    from core.integrations.firestore.operations import claim_operation, create_operation
    from core.integrations.firestore.save_intents import (
        create_save_intent,
        dispatch_save_intent,
        update_save_intent_checkpoints,
    )
    from core.operations.models import OperationOrigin, OperationResult

    save_intent_id = f"{operation_id}-save"
    content = "DATE: 2026-03-04\nPROVIDER: tldv\n---\nSpeaker A :: Completed synthetic transcript."
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    parent_hash = hashlib.sha256(PARENT_ID.encode("utf-8")).hexdigest()
    create_operation(
        client, operation_id=operation_id, origin=OperationOrigin.TELEGRAM,
        origin_operation_ref="synthetic-context", caller_binding="synthetic-caller",
        immutable_input_digest="sha256:synthetic-manifest",
        immutable_input_ref="synthetic-dialog-input-ref", provider="tldv",
        title="Synthetic transcript", explicit_folder_id=PARENT_ID,
        save_intent_id=save_intent_id, now=now,
    )
    create_save_intent(
        client, save_intent_id, operation_id=operation_id,
        desired_content_hash=content_hash, desired_parent_id_hash=parent_hash, now=now,
    )
    lease = claim_operation(client, operation_id, owner=owner, now=now, lease_seconds=lease_seconds)
    assert lease is not None
    result = OperationResult(document_id=DOC_ID, title="Synthetic transcript")
    assert dispatch_save_intent(
        client, save_intent_id, operation_id=operation_id, owner=owner, fence=lease.fence,
        now=now, create_callback=lambda: result,
    ) == result
    assert update_save_intent_checkpoints(
        client, save_intent_id, operation_id=operation_id, owner=owner, fence=lease.fence,
        now=now, fill_checkpoint_ref="synthetic-fill-checkpoint",
        index_checkpoint_ref="synthetic-index-checkpoint",
    ) is not None

    sdk_fakes.docs.queue("documents.get", response={
        "revisionId": "synthetic-revision",
        "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": line + "\n"}}]}}
            for line in content.splitlines()
        ]},
    })
    sdk_fakes.drive.queue("files.get", response={
        "id": DOC_ID, "name": "Synthetic transcript",
        "mimeType": "application/vnd.google-apps.document", "parents": [PARENT_ID], "trashed": False,
    })
    receipt = {
        "doc_id": DOC_ID, "content_hash": content_hash, "version": content_hash,
        "provider": "tldv", "parent_id": PARENT_ID, "parent_id_hash": parent_hash,
        "root_folder_id": ROOT_ID, "status": "completed",
        "skipped_utterances": list(skipped_utterances),
        "completed_at": now, "error": None, "owner": owner, "fence": lease.fence,
        "index_fence": 1, "resource_epoch": 0,
        "work_manifest_ref": "synthetic-work-manifest",
        "checkpoint_refs": ["synthetic-fill-checkpoint", "synthetic-index-checkpoint"],
    }
    _set_generation(client, DOC_ID, content_hash, receipt)
    client.collection("transcript_index").document(DOC_ID).set({
        **receipt, "status": "synced", "synced_at": now,
    })
    return lease, save_intent_id, receipt, content


def _finalize(client, *, operation_id=OPERATION_ID, save_intent_id=None, owner=OWNER,
              fence=1, now=NOW, clock=None):
    clock = clock or (lambda: now)
    return _completion_module().finalize_save(
        client, operation_id=operation_id,
        save_intent_id=save_intent_id or f"{operation_id}-save",
        owner=owner, fence=fence,
        origin_binding={
            "context_id": "synthetic-context", "draft_id": "synthetic-draft",
            "manifest_digest": "sha256:synthetic-manifest",
        },
        now=now,
        clock=clock,
    )


def _synthetic_google_oauth(monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "pytest-synthetic-refresh")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "pytest-synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "pytest-synthetic-secret")


def test_process_one_hashes_the_google_docs_reader_projection_and_publishes_after_all_writes(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    from core.indexing import processor

    from google.genai import types
    from core.integrations.qdrant import writer

    reader_projection = (
        "DATE: 2026-03-04\nTIME: 10:00\nPROVIDER: tldv\n---\n"
        "Speaker A :: Invented reader projection content."
    )
    monkeypatch.setattr(processor, "read_google_doc", lambda _doc_id: reader_projection)
    writer.upsert_user_facts([{
        "type": "user_fact", "doc_id": DOC_ID,
        "text": "Invented user fact remains available.", "root_folder_id": ROOT_ID,
    }], [[0.25]])
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part(text=json.dumps({"summary": "Synthetic summary.", "facts": ["Invented fact."]}))
        ]))
    ]))
    for _ in range(3):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)]
        ))
    effects = []
    for method_name in ("upsert_utterances", "upsert_summaries", "upsert_facts", "delete_old_versions",
                        "update_skipped_utterances", "update_index"):
        original = getattr(processor, method_name)
        def record(*args, _name=method_name, _original=original, **kwargs):
            effects.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(processor, method_name, record)
    original_upsert = processor.upsert_facts
    writes_before_receipt = []
    monkeypatch.setattr(
        processor,
        "upsert_facts",
        lambda *args, **kwargs: (writes_before_receipt.append("facts"), original_upsert(*args, **kwargs))[1],
    )

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID) == "processed"

    content_hash = hashlib.sha256(reader_projection.encode("utf-8")).hexdigest()
    receipt = _stored_generation(firestore_fake, DOC_ID, content_hash)
    assert receipt is not None
    assert receipt["content_hash"] == content_hash
    assert receipt["version"] == content_hash
    assert receipt["provider"] == "tldv"
    assert receipt["parent_id"] == PARENT_ID
    assert receipt["status"] == "completed"
    assert writes_before_receipt == ["facts"]
    qdrant_mutations = {"upsert_utterances", "upsert_summaries", "upsert_facts", "delete_old_versions"}
    qdrant_positions = [i for i, effect in enumerate(effects) if effect in qdrant_mutations]
    final_checkpoint = max(i for i, effect in enumerate(effects) if effect == "update_index")
    assert qdrant_mutations <= set(effects)
    assert max(qdrant_positions) < final_checkpoint
    assert "mark_synced" not in effects
    receipt_path = f"index_completions/{DOC_ID}/generations/{content_hash}"
    firestore_trace = firestore_fake.trace()
    receipt_commit = max(i for i, event in enumerate(firestore_trace)
                         if event["path"] == receipt_path and event["outcome"] == "committed")
    index_commits = [i for i, event in enumerate(firestore_trace)
                     if event["path"] == f"transcript_index/{DOC_ID}" and event["outcome"] == "committed"]
    # Receipt plus transcript synced status are one fenced Firestore commit;
    # the fake reports both document writes separately in that commit.
    assert index_commits and receipt_commit >= max(index_commits) - 1
    deletes = [event["arguments"] for event in sdk_fakes.trace
               if event["boundary"] == "qdrant" and event["operation"] == "delete"]
    _assert_user_fact_excluded(deletes)
    assert any(
        event["boundary"] == "qdrant" and event["operation"] == "upsert"
        for event in sdk_fakes.trace
    )


def test_unknown_index_batch_keeps_common_guard_blocking_after_lease_expiry(
    firestore_fake, fake_clock,
):
    from core.integrations.firestore.mutations import (
        acquire_index_guard, begin_index_effect, get_resource_guard,
        release_index_guard, settle_index_effect,
    )

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    client.collection("transcript_index").document(DOC_ID).set({"doc_id": DOC_ID, "status": "syncing"})
    owner = "synthetic-index-worker"
    acquired = acquire_index_guard(
        client, doc_id=DOC_ID, owner=owner, now=fake_clock.now(), lease_seconds=5,
        root_folder_id=ROOT_ID,
    )
    assert acquired is not None
    fence, _epoch = acquired
    assert begin_index_effect(
        client, doc_id=DOC_ID, owner=owner, fence=fence, effect_id="batch-1",
        kind="upsert_utterances", payload_digest="sha256:batch", now=fake_clock.now(),
    )
    assert settle_index_effect(
        client, doc_id=DOC_ID, owner=owner, fence=fence, effect_id="batch-1",
        now=fake_clock.now(), confirmed=False,
    )

    fake_clock.advance(seconds=6)

    assert acquire_index_guard(
        client, doc_id=DOC_ID, owner="takeover", now=fake_clock.now(), lease_seconds=30,
        root_folder_id=ROOT_ID,
    ) is None
    assert not settle_index_effect(
        client, doc_id=DOC_ID, owner=owner, fence=fence, effect_id="batch-1",
        now=fake_clock.now(), confirmed=True,
    )
    assert not release_index_guard(
        client, doc_id=DOC_ID, owner=owner, fence=fence, now=fake_clock.now(),
    )
    assert get_resource_guard(client, f"document:{DOC_ID}").unsettled_effect_ids == ("batch-1",)


def test_managed_index_reference_binds_route_operation_intent_and_outbox(
    firestore_fake, sdk_fakes,
):
    from core.indexing.processor import _validate_managed_reference
    from core.integrations.firestore.outbox import OutboxTargetKind, create_outbox_record
    from core.integrations.firestore.save_intents import update_save_intent_checkpoints

    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, _content = _save_fixture(client, sdk_fakes)
    outbox_id = "index-dispatch-managed-reference"
    create_outbox_record(
        client, outbox_id, target_kind=OutboxTargetKind.INDEX_DISPATCH, generation=1,
        schedule_at=NOW, now=NOW, operation_id=OPERATION_ID,
        document_id=DOC_ID, save_intent_id=save_intent_id,
    )
    assert update_save_intent_checkpoints(
        client, save_intent_id, operation_id=OPERATION_ID, owner=OWNER,
        fence=lease.fence, now=NOW, index_checkpoint_ref=outbox_id,
    ) is not None
    reference = {
        "schema_version": 1, "outbox_id": outbox_id, "target_kind": "index_dispatch",
        "generation": 1, "operation_id": OPERATION_ID, "document_id": DOC_ID,
        "save_intent_id": save_intent_id,
    }

    assert _validate_managed_reference(reference, DOC_ID, PARENT_ID) is not None
    with pytest.raises(ValueError, match="invalid managed index reference"):
        _validate_managed_reference(reference, "another-doc", PARENT_ID)


def test_managed_index_stage_uses_durable_budget_and_completed_replay(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.indexing import processor
    from core.integrations.firestore.operations import get_operation, release_operation
    from core.integrations.firestore.save_intents import get_save_intent

    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, content = _save_fixture(client, sdk_fakes)
    assert release_operation(client, OPERATION_ID, owner=OWNER, fence=lease.fence, now=NOW)
    monkeypatch.setattr(
        processor, "_validate_managed_reference",
        lambda _reference, _doc_id, _parent_id: (
            get_operation(client, OPERATION_ID), get_save_intent(client, save_intent_id),
        ),
    )
    calls = []

    def attempt(*_args, **_kwargs):
        calls.append("called")
        if len(calls) == 1:
            raise ValueError("synthetic retryable attempt")
        return "processed"

    monkeypatch.setattr(processor, "process_one", attempt)
    reference = {"unused": "validation is patched in this unit"}
    with pytest.raises(ValueError, match="synthetic retryable"):
        processor.process_managed_one(DOC_ID, PARENT_ID, ROOT_ID, content, reference)
    assert processor.process_managed_one(DOC_ID, PARENT_ID, ROOT_ID, content, reference) == "processed"
    assert processor.process_managed_one(DOC_ID, PARENT_ID, ROOT_ID, content, reference) == "processed"
    from core.operations.retries import RetryGovernor
    assert RetryGovernor(client).get_stage(OPERATION_ID, "managed_index").state.value == "completed"
    assert len(calls) == 2


def test_three_managed_index_failures_are_terminal_and_duplicate_does_not_dispatch(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.indexing import processor
    from core.integrations.firestore.operations import get_operation, release_operation
    from core.integrations.firestore.save_intents import get_save_intent
    from core.operations.retries import RetryGovernor

    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, content = _save_fixture(client, sdk_fakes)
    assert release_operation(client, OPERATION_ID, owner=OWNER, fence=lease.fence, now=NOW)
    monkeypatch.setattr(
        processor, "_validate_managed_reference",
        lambda _reference, _doc_id, _parent_id: (
            get_operation(client, OPERATION_ID), get_save_intent(client, save_intent_id),
        ),
    )
    calls = []

    def always_fail(*_args, **_kwargs):
        calls.append("called")
        raise ValueError("synthetic managed index failure")

    monkeypatch.setattr(processor, "process_one", always_fail)
    reference = {"unused": "validation is patched in this unit"}
    for _ in range(3):
        with pytest.raises(ValueError, match="synthetic managed index failure"):
            processor.process_managed_one(DOC_ID, PARENT_ID, ROOT_ID, content, reference)

    operation = get_operation(client, OPERATION_ID)
    stage = RetryGovernor(client).get_stage(OPERATION_ID, "managed_index")
    assert operation.state.value == "failed"
    assert stage.state.value == "failed"
    assert stage.failures == 3
    assert len(calls) == 3
    assert processor.process_managed_one(DOC_ID, PARENT_ID, ROOT_ID, content, reference) == "managed_terminal"
    assert len(calls) == 3


@pytest.mark.parametrize("payload_matches", [True, False])
def test_qdrant_upsert_readback_compares_stored_payload_digest(
    monkeypatch, payload_matches,
):
    from types import SimpleNamespace
    from core.indexing import processor
    from core.integrations.qdrant import client as qdrant_client
    from core.integrations.qdrant.schema import get_collection_name

    expected_payload = {"doc_id": DOC_ID, "type": "utterance", "text": "synthetic transcript text"}
    stored_payload = dict(expected_payload)
    if not payload_matches:
        stored_payload["text"] = "different synthetic text"
    digest = hashlib.sha256(json.dumps(
        expected_payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()

    class RetrieveOnlyQdrant:
        def __init__(self):
            self.calls = []

        def retrieve(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(id="expected-point-id", payload=stored_payload)]

    fake = RetrieveOnlyQdrant()
    monkeypatch.setattr(qdrant_client, "get_client", lambda: fake)
    intent = {"readback": {"action": "upsert", "points": [
        {"id": "expected-point-id", "payload_digest": digest},
    ]}}

    assert processor._readback_confirms(intent) is payload_matches
    assert fake.calls == [{
        "collection_name": get_collection_name(),
        "ids": ["expected-point-id"], "with_payload": True, "with_vectors": False,
    }]


@pytest.mark.parametrize("stale_field", ["synced_status", "content_hash", "version", "provider", "parent_id"])
def test_stale_metadata_or_receipt_identity_cannot_verify_current_generation(
    firestore_fake, stale_field,
):
    from core.indexing import processor

    client = firestore_fake.client_class()
    current_text = "DATE: 2026-03-04\nPROVIDER: tldv\n---\nSpeaker A :: Current content.\n"
    current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
    current = {
        "doc_id": DOC_ID,
        "root_folder_id": ROOT_ID,
        "content_hash": current_hash,
        "version": current_hash,
        "parent_id_hash": hashlib.sha256(PARENT_ID.encode("utf-8")).hexdigest(),
        "provider": "tldv",
        "parent_id": PARENT_ID,
        "status": "completed",
        "skipped_utterances": [],
        "completed_at": NOW,
        "error": None,
        "owner": "index-worker",
        "fence": 1,
        "index_fence": 1,
        "resource_epoch": 0,
        "work_manifest_ref": "synthetic-work-manifest",
        "checkpoint_refs": ["synthetic-index-checkpoint"],
    }
    client.collection("transcript_index").document(DOC_ID).set({
        **current,
        "status": "error" if stale_field == "synced_status" else "synced",
        "synced_at": NOW,
    })
    receipt = {**current, "status": "completed"}
    if stale_field != "synced_status":
        stale = dict(receipt)
        stale[stale_field] = "sha256:older-value"
        generation_key = stale["content_hash"] if stale_field == "content_hash" else current_hash
        _set_generation(client, DOC_ID, generation_key, stale)
    else:
        _set_generation(client, DOC_ID, current_hash, receipt)
    assert processor.verify_current_generation(DOC_ID, PARENT_ID, ROOT_ID, current_text) is False


def test_dirty_incremental_window_is_retried_after_hash_checkpoint_and_user_fact_is_preserved(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    from core.indexing import processor

    from tests.core.indexing.test_incremental_recovery import (
        DOC_ID as INCREMENTAL_DOC,
        PARENT_ID as INCREMENTAL_PARENT,
        ROOT_ID as INCREMENTAL_ROOT,
        _inputs,
        _queue_generation,
        _qdrant_events,
    )

    old, new, new_hashes = _inputs(
        app_root, sdk_fakes, firestore_fake, insertion_and_deletion=False,
        queue_responses=False,
    )
    from core.integrations.qdrant import writer

    writer.upsert_user_facts([{
        "type": "user_fact", "doc_id": INCREMENTAL_DOC,
        "text": "Invented user authored fact.", "root_folder_id": INCREMENTAL_ROOT,
    }], [[0.25]])
    _queue_generation(sdk_fakes)
    original = processor.upsert_summaries
    failed = False

    def fail_after_hash_checkpoint(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            error = RuntimeError("synthetic summary write interruption")
            error.outcome = "rejected"
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, "upsert_summaries", fail_after_hash_checkpoint)
    with pytest.raises(RuntimeError, match="summary write interruption"):
        processor.process_one(INCREMENTAL_DOC, INCREMENTAL_PARENT, INCREMENTAL_ROOT, new)
    interrupted = firestore_fake.documents[f"transcript_index/{INCREMENTAL_DOC}"]
    assert interrupted["utterance_hashes"]["7"] == new_hashes["7"]

    monkeypatch.setattr(processor, "upsert_summaries", original)
    _queue_generation(sdk_fakes)
    start = len(sdk_fakes.trace)
    assert processor.process_one(INCREMENTAL_DOC, INCREMENTAL_PARENT, INCREMENTAL_ROOT, new) == "processed"
    retry_trace = sdk_fakes.trace[start:]
    assert any(e["boundary"] == "qdrant" and e["operation"] == "upsert" for e in retry_trace)
    deletes = _qdrant_events(sdk_fakes, "delete")
    _assert_user_fact_excluded(deletes)
    assert old != new


def test_mutation_between_final_read_and_completion_commit_is_serialized_or_rejected(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.api_core.exceptions import Aborted
    from core.integrations.firestore.mutations import dispatch_mutation
    from core.indexing import processor

    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, _content = _save_fixture(client, sdk_fakes)
    _synthetic_google_oauth(monkeypatch)
    original_verify = processor.verify_current_generation
    mutation_operation_id = "completion-racing-mutation"
    mutation_lease = _lease(client, operation_id=mutation_operation_id, owner="mutation-worker")
    mutation_effects = []

    def mutate_after_final_read(*args, **kwargs):
        verified = original_verify(*args, **kwargs)
        result = dispatch_mutation(
            client, operation_id=mutation_operation_id, effect_id="edit-after-final-read",
            owner="mutation-worker", fence=mutation_lease.fence, kind="content_update",
            resources=(RESOURCE,), payload_digest="sha256:changed-doc",
            desired_state_digest="sha256:changed-doc", now=NOW,
            mutation_callback=lambda: mutation_effects.append("changed") or "updated",
        )
        if result is not None:
            mutation_effects.append(result)
        return verified

    monkeypatch.setattr(processor, "verify_current_generation", mutate_after_final_read)
    try:
        _finalize(client, save_intent_id=save_intent_id, fence=lease.fence)
    except Aborted:
        assert mutation_effects
    operation = firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]
    intent = firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]
    if mutation_effects:
        assert operation.get("result") is None
        assert operation.get("state") != "completed"
        assert operation.get("document_finalized") is None
        assert operation.get("cleanup_authorization") is None
        assert intent.get("completion_certificate") is None
        assert intent.get("cleanup_authorization") is None
    else:
        assert operation.get("state") == "completed"
        assert intent.get("completion_certificate") is not None


def test_stale_index_publisher_cannot_publish_after_new_fence(firestore_fake, fake_clock):
    from core.integrations.firestore import index_completions

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    parent_hash = hashlib.sha256(PARENT_ID.encode("utf-8")).hexdigest()
    transcript = {
        "doc_id": DOC_ID, "content_hash": "sha256:current-generation",
        "root_folder_id": ROOT_ID, "version": "sha256:current-generation",
        "parent_id_hash": parent_hash, "provider": "tldv", "parent_id": PARENT_ID,
        "status": "synced", "skipped_utterances": [], "completed_at": fake_clock.now(),
        "error": None, "owner": "index-worker", "fence": 2, "index_fence": 2,
        "resource_epoch": 0, "work_manifest_ref": "synthetic-current-manifest",
        "checkpoint_refs": ["synthetic-current-checkpoint"],
        "completion_ref": "sha256:previous-generation",
    }
    client.collection("transcript_index").document(DOC_ID).set(transcript)
    previous = {
        **transcript, "content_hash": "sha256:previous-generation",
        "version": "sha256:previous-generation", "status": "completed", "index_fence": 1,
        "fence": 1,
    }
    _set_generation(client, DOC_ID, previous["content_hash"], previous)
    before = dict(firestore_fake.documents)
    stale = {
        **transcript, "content_hash": "sha256:stale-index-generation",
        "version": "sha256:stale-index-generation", "status": "completed", "index_fence": 1,
        "fence": 1,
    }
    assert index_completions.publish_completion(client, stale, now=fake_clock.now()) is False
    assert firestore_fake.documents == before
    assert _stored_generation(firestore_fake, DOC_ID, stale["content_hash"]) is None


def test_expired_finalization_fence_cannot_commit_result_or_cleanup_authorization(
    firestore_fake, fake_clock, sdk_fakes,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    old, save_intent_id, _receipt, _content = _save_fixture(
        client, sdk_fakes, now=fake_clock.now(), lease_seconds=5,
    )
    fake_clock.advance(seconds=6)
    from core.integrations.firestore.operations import claim_operation
    current = claim_operation(
        client, OPERATION_ID, owner="new-worker", now=fake_clock.now(), lease_seconds=30,
    )
    assert current is not None and current.fence > old.fence
    _finalize(client, save_intent_id=save_intent_id, fence=old.fence, now=fake_clock.now())
    operation = firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]
    intent = firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]
    assert operation.get("result") is None
    assert operation.get("state") != "completed"
    assert operation.get("document_finalized") is None
    assert operation.get("cleanup_authorization") is None
    assert intent.get("completion_certificate") is None
    assert intent.get("cleanup_authorization") is None


def test_abandoned_finalization_guard_is_reclaimed_after_lease_expiry(
    firestore_fake, fake_clock, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.mutations import acquire_completion_guard, get_resource_guard
    from core.integrations.firestore.operations import claim_operation

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    old, save_intent_id, _receipt, _content = _save_fixture(
        client, sdk_fakes, now=fake_clock.now(), lease_seconds=5,
    )
    reservation = acquire_completion_guard(
        client, operation_id=OPERATION_ID, owner=OWNER, fence=old.fence,
        resource=RESOURCE, now=fake_clock.now(), clock=fake_clock.now,
    )
    assert reservation is not None
    # Model process death: deliberately leave the completion reservation in place.
    fake_clock.advance(seconds=6)
    current = claim_operation(
        client, OPERATION_ID, owner="restarted-worker", now=fake_clock.now(), lease_seconds=30,
    )
    assert current is not None and current.fence > old.fence
    assert not get_resource_guard(client, RESOURCE).unsettled_effect_ids

    _synthetic_google_oauth(monkeypatch)
    result = _finalize(
        client, save_intent_id=save_intent_id, owner="restarted-worker",
        fence=current.fence, now=fake_clock.now(), clock=fake_clock.now,
    )
    assert result is not None
    assert firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]["state"] == "completed"
    assert firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]["completion_certificate"]


def test_finalization_checks_lease_again_after_slow_external_read(
    firestore_fake, fake_clock, sdk_fakes, monkeypatch,
):
    from core.indexing import processor

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    lease, save_intent_id, _receipt, _content = _save_fixture(
        client, sdk_fakes, now=fake_clock.now(), lease_seconds=5,
    )
    _synthetic_google_oauth(monkeypatch)
    original_verify = processor.verify_current_generation

    def expire_lease_during_read(*args, **kwargs):
        verified = original_verify(*args, **kwargs)
        fake_clock.advance(seconds=6)
        return verified

    monkeypatch.setattr(processor, "verify_current_generation", expire_lease_during_read)
    assert _finalize(
        client, save_intent_id=save_intent_id, fence=lease.fence,
        now=fake_clock.now(), clock=fake_clock.now,
    ) is None
    operation = firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]
    intent = firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]
    assert operation.get("state") != "completed"
    assert operation.get("result") is None
    assert intent.get("completion_certificate") is None
    assert intent.get("cleanup_authorization") is None


def test_provider_mismatch_cannot_finalize(firestore_fake, sdk_fakes, monkeypatch):
    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, _content = _save_fixture(client, sdk_fakes)
    _synthetic_google_oauth(monkeypatch)
    operation_ref = client.collection("dialog_operations").document(OPERATION_ID)
    operation_ref.update({"provider": "fireflies"})

    assert _finalize(client, save_intent_id=save_intent_id, fence=lease.fence) is None
    operation = firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]
    intent = firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]
    assert operation.get("state") != "completed"
    assert intent.get("completion_certificate") is None
    assert intent.get("cleanup_authorization") is None


@pytest.mark.parametrize("certificate_field", ["operation_id", "save_intent_id", "doc_id"])
def test_misbound_completion_certificate_does_not_replay(
    firestore_fake, sdk_fakes, monkeypatch, certificate_field,
):
    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, _content = _save_fixture(client, sdk_fakes)
    _synthetic_google_oauth(monkeypatch)
    assert _finalize(client, save_intent_id=save_intent_id, fence=lease.fence) is not None
    intent_ref = client.collection("dialog_save_intents").document(save_intent_id)
    intent = intent_ref.get().to_dict()
    intent["completion_certificate"][certificate_field] = f"misbound-{certificate_field}"
    intent_ref.set(intent)
    trace_length = len(sdk_fakes.trace)

    assert _finalize(client, save_intent_id=save_intent_id, fence=lease.fence) is None
    assert len(sdk_fakes.trace) == trace_length


def test_new_fence_cannot_finalize_or_redispatch_while_old_qdrant_batch_is_unknown(
    firestore_fake, fake_clock, sdk_fakes,
):
    from core.integrations.firestore.mutations import dispatch_mutation, get_resource_guard
    from core.integrations.firestore.operations import claim_operation

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    old, save_intent_id, _receipt, _content = _save_fixture(
        client, sdk_fakes, now=fake_clock.now(), lease_seconds=5,
    )
    calls = []

    def unknown_batch():
        calls.append("old-qdrant-batch")
        raise TimeoutError("synthetic Qdrant outcome unknown")

    with pytest.raises(TimeoutError, match="outcome unknown"):
        dispatch_mutation(
            client, operation_id=OPERATION_ID, effect_id="old-index-batch", owner=OWNER,
            fence=old.fence, kind="provider_call", resources=(RESOURCE,),
            payload_digest="sha256:batch", now=fake_clock.now(), mutation_callback=unknown_batch,
        )
    fake_clock.advance(seconds=6)
    current = claim_operation(
        client, OPERATION_ID, owner="new-worker", now=fake_clock.now(), lease_seconds=30,
    )
    assert current is not None and current.fence > old.fence
    assert get_resource_guard(client, RESOURCE).unsettled_effect_ids == ("old-index-batch",)
    _finalize(client, save_intent_id=save_intent_id, owner="new-worker",
              fence=current.fence, now=fake_clock.now())
    operation = firestore_fake.documents[f"dialog_operations/{OPERATION_ID}"]
    intent = firestore_fake.documents[f"dialog_save_intents/{save_intent_id}"]
    assert operation.get("result") is None
    assert operation.get("state") != "completed"
    assert operation.get("document_finalized") is None
    assert operation.get("cleanup_authorization") is None
    assert intent.get("completion_certificate") is None
    assert intent.get("cleanup_authorization") is None
    assert dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="new-index-batch", owner="new-worker",
        fence=current.fence, kind="provider_call", resources=(RESOURCE,),
        payload_digest="sha256:batch", now=fake_clock.now(),
        mutation_callback=lambda: calls.append("conflicting-redispatch"),
    ) is None
    assert calls == ["old-qdrant-batch"]


@pytest.mark.parametrize("readback_confirms", [True, False])
def test_unknown_qdrant_effect_reconciliation_requires_readback(
    firestore_fake, monkeypatch, readback_confirms,
):
    from core.indexing import processor
    from core.indexing.processor import _IndexClaim
    from core.integrations.firestore.mutations import (
        acquire_index_guard, begin_index_effect, get_unsettled_index_effects,
        settle_index_effect,
    )

    client = firestore_fake.client_class()
    test_now = datetime.now(timezone.utc)
    old_owner = "old-index-worker"
    old_fence, _epoch = acquire_index_guard(
        client, doc_id=DOC_ID, owner=old_owner, now=test_now, lease_seconds=5,
        root_folder_id=ROOT_ID,
    )
    assert begin_index_effect(
        client, doc_id=DOC_ID, owner=old_owner, fence=old_fence, effect_id="uncertain-upsert",
        kind="upsert_utterances", payload_digest="sha256:batch", now=test_now,
        readback={"action": "upsert", "points": [{"id": "point-1", "payload_digest": "sha256:payload"}]},
    )
    assert settle_index_effect(
        client, doc_id=DOC_ID, owner=old_owner, fence=old_fence,
        effect_id="uncertain-upsert", now=test_now, confirmed=False,
    )
    takeover_at = test_now + timedelta(seconds=6)
    new_owner = "new-index-worker"
    new_fence, epoch = acquire_index_guard(
        client, doc_id=DOC_ID, owner=new_owner, now=takeover_at, lease_seconds=30,
        root_folder_id=ROOT_ID, allow_unsettled_index_effects=True,
    )
    assert get_unsettled_index_effects(client, DOC_ID)["uncertain-upsert"]["state"] == "unknown"
    assert get_unsettled_index_effects(client, DOC_ID)["uncertain-upsert"]["fence"] < new_fence
    monkeypatch.setattr(processor, "_readback_confirms", lambda _intent: readback_confirms)
    claim = _IndexClaim(DOC_ID, new_owner, new_fence, epoch, client)

    confirmed = processor._reconcile_unknown_effects(claim)
    assert confirmed is readback_confirms
    unsettled = get_unsettled_index_effects(client, DOC_ID)
    assert bool(unsettled) is (not readback_confirms)


def test_index_claim_waits_for_active_regular_resource_owner(firestore_fake):
    from core.integrations.firestore.mutations import _resource_ref, acquire_index_guard

    client = firestore_fake.client_class()
    guard = _resource_ref(client, RESOURCE)
    guard.set({"resource": RESOURCE, "owner_operation_id": "active-agent-op",
               "expires_at": NOW + timedelta(minutes=1), "resource_epoch": 3,
               "unsettled_effect_ids": []})
    assert acquire_index_guard(
        client, doc_id=DOC_ID, owner="index-worker", now=NOW, lease_seconds=30,
        root_folder_id=ROOT_ID,
    ) is None
    guard.set({"resource": RESOURCE, "owner_operation_id": "expired-agent-op",
               "expires_at": NOW - timedelta(seconds=1), "resource_epoch": 3,
               "unsettled_effect_ids": []})
    assert acquire_index_guard(
        client, doc_id=DOC_ID, owner="index-worker", now=NOW, lease_seconds=30,
        root_folder_id=ROOT_ID,
    ) is not None


def test_operation_result_and_immutable_completion_certificate_commit_atomically_and_replay(
    firestore_fake, sdk_fakes, monkeypatch,
):
    client = firestore_fake.client_class()
    lease, save_intent_id, receipt, _content = _save_fixture(
        client, sdk_fakes, skipped_utterances=(2, 7),
    )
    _synthetic_google_oauth(monkeypatch)
    operation_path = f"dialog_operations/{OPERATION_ID}"
    intent_path = f"dialog_save_intents/{save_intent_id}"
    firestore_fake.fail_next(
        "commit", path=intent_path, phase="before",
        exception=RuntimeError("synthetic pre-commit crash"),
    )
    with pytest.raises(RuntimeError, match="pre-commit crash"):
        _finalize(client, save_intent_id=save_intent_id, fence=lease.fence)
    assert firestore_fake.documents[operation_path].get("result") is None
    assert firestore_fake.documents[operation_path].get("state") != "completed"
    assert firestore_fake.documents[operation_path].get("input_cleanup_state") is None
    assert firestore_fake.documents[intent_path].get("completion_certificate") is None

    content = "DATE: 2026-03-04\nPROVIDER: tldv\n---\nSpeaker A :: Completed synthetic transcript."
    sdk_fakes.docs.queue("documents.get", response={
        "revisionId": "synthetic-revision",
        "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": line + "\n"}}]}}
            for line in content.splitlines()
        ]},
    })
    sdk_fakes.drive.queue("files.get", response={
        "id": DOC_ID, "name": "Synthetic transcript",
        "mimeType": "application/vnd.google-apps.document", "parents": [PARENT_ID], "trashed": False,
    })

    firestore_fake.fail_next(
        "commit", path=intent_path, phase="after",
        exception=RuntimeError("synthetic response lost after commit"),
    )
    with pytest.raises(RuntimeError, match="response lost after commit"):
        _finalize(client, save_intent_id=save_intent_id, fence=lease.fence)
    first_operation = firestore_fake.documents[operation_path]
    first_intent = firestore_fake.documents[intent_path]
    certificate = first_intent["completion_certificate"]
    assert first_operation["state"] == "completed"
    assert first_operation["input_cleanup_state"] == "pending"
    assert first_operation["cleanup_authorization"] == first_intent["cleanup_authorization"]
    assert first_operation["result"]["document_id"] == DOC_ID
    assert first_operation["result"]["skipped_utterances"] == [2, 7]
    assert certificate["save_intent_id"] == save_intent_id
    assert certificate["operation_id"] == OPERATION_ID
    assert certificate["doc_id"] == DOC_ID
    assert certificate["content_hash"] == receipt["content_hash"]
    assert certificate["origin_binding"] == {
        "context_id": "synthetic-context", "draft_id": "synthetic-draft",
        "manifest_digest": "sha256:synthetic-manifest",
    }
    assert first_operation["result"]["document_id"] == certificate["doc_id"]
    trace_after_commit = len(sdk_fakes.trace)

    replayed_result = _finalize(client, save_intent_id=save_intent_id, fence=lease.fence)

    assert firestore_fake.documents[operation_path] == first_operation
    assert firestore_fake.documents[intent_path] == first_intent
    assert len(sdk_fakes.trace) == trace_after_commit
    assert replayed_result.skipped_utterances == (2, 7)


def test_legacy_operation_results_without_skips_keep_their_serialized_shape():
    from core.integrations.firestore.operations import _result_data, _result_from
    from core.operations.models import OperationResult

    legacy_data = {"document_id": DOC_ID, "title": "Legacy", "resource_url": None}
    restored = _result_from(legacy_data)
    assert restored == OperationResult(document_id=DOC_ID, title="Legacy")
    assert restored.skipped_utterances is None
    assert _result_data(restored) == legacy_data


def test_independent_edit_after_commit_does_not_revoke_certificate(firestore_fake, sdk_fakes, monkeypatch):
    from core.integrations.firestore.mutations import dispatch_mutation

    client = firestore_fake.client_class()
    lease, save_intent_id, _receipt, _content = _save_fixture(client, sdk_fakes)
    _synthetic_google_oauth(monkeypatch)
    _finalize(client, save_intent_id=save_intent_id, fence=lease.fence)
    intent_path = f"dialog_save_intents/{save_intent_id}"
    certificate = firestore_fake.documents[intent_path]["completion_certificate"]

    mutation_operation_id = "completion-later-edit"
    mutation_lease = _lease(
        client, operation_id=mutation_operation_id, owner="later-editor",
        now=NOW + timedelta(seconds=1),
    )
    result = dispatch_mutation(
        client, operation_id=mutation_operation_id, effect_id="later-independent-edit", owner="later-editor",
        fence=mutation_lease.fence, kind="content_update", resources=(RESOURCE,),
        payload_digest="sha256:next-version", now=NOW + timedelta(seconds=1),
        mutation_callback=lambda: "updated",
    )
    assert result == "updated"

    assert firestore_fake.documents[intent_path]["completion_certificate"] == certificate
