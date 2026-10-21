from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from core.dialogs.saving import accept_save, advance_save
from core.integrations.firestore.input_parts import read_input_parts
from core.integrations.firestore.operations import claim_operation, get_operation, get_stage
from core.integrations.firestore.outbox import get_outbox_record
from core.integrations.firestore.transcripts import get_transcript_record
from core.operations.models import OperationOrigin
from core.operations.retries import RetryGovernor, UnknownOutcome
from tests.fixtures.sdk_fakes import fake_discovery_build


NOW = datetime.now(timezone.utc)
ARGUMENTS = {
    "provider": "manual", "title": "Planning", "folder_path": ["Client"],
    "utterances": [{"speaker": "Alex", "text": "first\nline"},
                   {"speaker": "Sam", "text": "second exact line"}],
}
FOLDERS = {"client": {"name": "Client", "parent_id": "root"}}


def _accept(client, arguments=ARGUMENTS, operation_id="operation-save", save_intent_id=None):
    return accept_save(
        client, arguments, operation_id=operation_id, caller_binding="caller-1",
        origin=OperationOrigin.MCP, now=NOW, save_intent_id=save_intent_id,
    )


def test_acceptance_is_side_effect_free_and_replay_is_stable(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    first = _accept(client)
    second = _accept(client)

    assert first.operation == second.operation
    assert first.save_intent_id == second.save_intent_id
    assert first.immutable_input_digest == second.immutable_input_digest
    assert sdk_fakes.trace == []
    assert not any(path.startswith("task_outbox/") for path in firestore_fake.documents)
    stored = read_input_parts(client, first.immutable_input_ref)
    assert b"first\\nline" in stored
    assert get_operation(client, first.operation.operation_id).save_intent_id == first.save_intent_id


def test_acceptance_stores_large_dialog_without_trimming_or_token_chunking(firestore_fake):
    client = firestore_fake.client_class()
    exact_text = "dialog line 🚀\n" * 100_000
    accepted = _accept(client, {
        "provider": "manual", "title": "Large", "utterances": [{"speaker": "Alex", "text": exact_text}],
    }, operation_id="large-save")
    raw = read_input_parts(client, accepted.immutable_input_ref)
    decoded = json.loads(raw)
    assert decoded["utterances"][0]["text"] == exact_text
    assert "\\n" in decoded["desired_content"]
    manifest = client.collection("dialog_operations").document("large-save").collection(
        "input_parts"
    ).document("manifest").get().to_dict()
    assert manifest["part_count"] > 1


def test_advance_resolves_folder_fills_once_updates_index_additively_and_dispatches_durably(
    firestore_fake, sdk_fakes,
):
    client = firestore_fake.client_class()
    accepted = _accept(client)
    document_id = "doc-saved"
    client.collection("transcript_index").document(document_id).set({
        "doc_id": document_id, "status": "syncing", "sync_stage": "vector_sync",
        "user_facts": {"keep": True}, "parent_id": "old-parent",
    })
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None

    sdk_fakes.drive.queue("files.create", response={"id": document_id})
    sdk_fakes.docs.queue(
        "documents.get", response={"revisionId": "rev-before", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )
    sdk_fakes.docs.queue(
        "documents.get", response={"revisionId": "rev-before", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-after"})
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "fill-time",
    })
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "post-fill-time",
    })

    progress = advance_save(
        client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
        folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
    )

    assert progress.document_id == document_id
    saved_operation = get_operation(client, accepted.operation.operation_id)
    assert saved_operation.doc_id == document_id
    assert saved_operation.create_outcome == "confirmed"
    assert progress.stages == ("validate", "classify", "create", "fill", "metadata", "index_dispatch")
    create = next(row for row in sdk_fakes.trace if row["operation"] == "files.create")
    assert create["arguments"]["body"]["parents"] == ["client"]
    fill = next(row for row in sdk_fakes.trace if row["operation"] == "documents.batchUpdate")
    inserted = fill["arguments"]["body"]["requests"][0]["insertText"]["text"]
    assert 'first\\nline' in inserted
    assert 'second exact line' in inserted
    record = get_transcript_record(document_id, client)
    assert record["modified_time"] == "post-fill-time"
    assert record["user_facts"] == {"keep": True}
    assert record["status"] == "syncing" and record["sync_stage"] == "vector_sync"
    assert record["parent_id"] == "client"
    outbox_ids = [key.split("/", 1)[1] for key in firestore_fake.documents if key.startswith("task_outbox/")]
    assert len(outbox_ids) == 1
    outbox = get_outbox_record(client, outbox_ids[0])
    assert outbox.target_kind.value == "index_dispatch"
    assert outbox.document_id == document_id
    assert outbox.operation_id == accepted.operation.operation_id
    assert get_stage(client, accepted.operation.operation_id, "index_dispatch").state.value == "completed"

    retry = _accept(client, ARGUMENTS, operation_id="operation-save-retry",
                    save_intent_id=accepted.save_intent_id)
    retry_lease = claim_operation(client, retry.operation.operation_id, owner="retry-worker", now=NOW, lease_seconds=600)
    assert retry_lease is not None
    retry_governor = RetryGovernor(client)
    for stage_key in ("classify", "fill", "metadata", "index_dispatch"):
        with pytest.raises(UnknownOutcome):
            retry_governor.run_stage(
                retry.operation.operation_id, stage_key,
                lambda: (_ for _ in ()).throw(UnknownOutcome("synthetic in-flight attempt")),
                now=NOW, owner="retry-worker", fence=retry_lease.fence,
            )
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "retry-metadata",
    })
    retry_progress = advance_save(
        client, retry.operation.operation_id, owner="retry-worker", fence=retry_lease.fence, now=NOW,
        folders=FOLDERS, root_folder_id="root",
        speaker_index_lookup=lambda _speaker: pytest.fail("manual retry reran folder classification"),
        suggestion=lambda *_args: pytest.fail("manual retry reran folder suggestion"),
        drive=fake_discovery_build("drive", "v3"), docs=fake_discovery_build("docs", "v1"),
    )
    assert retry_progress.document_id == document_id
    assert retry_progress.save_intent_id == accepted.save_intent_id
    assert get_stage(client, retry.operation.operation_id, "fill").state.value == "completed"
    assert get_operation(client, retry.operation.operation_id).selected_folder_id == "client"
    assert get_stage(client, retry.operation.operation_id, "metadata").state.value == "completed"
    assert get_stage(client, retry.operation.operation_id, "index_dispatch").state.value == "completed"
    outbox_after_retry = get_outbox_record(client, outbox_ids[0])
    assert outbox_after_retry.operation_id == accepted.operation.operation_id
    assert len([row for row in firestore_fake.documents if row.startswith("task_outbox/")]) == 1
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    event_count = len([row for row in sdk_fakes.trace if row["operation"] != "build"])
    replay = advance_save(
        client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
        folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
    )
    assert replay.document_id == document_id
    assert len([row for row in sdk_fakes.trace if row["operation"] != "build"]) == event_count
    replay_acceptance = _accept(client)
    assert replay_acceptance.save_intent_id == accepted.save_intent_id


def test_automatic_unassigned_fallback_creates_and_persists_root_folder(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    accepted = _accept(client, {
        "provider": "manual", "title": "Planning",
        "utterances": [{"speaker": "Alex", "text": "unmatched speaker"}],
    }, operation_id="unassigned-save")
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None
    ensured = []
    document_id = "doc-unassigned"
    sdk_fakes.drive.queue("files.create", response={"id": document_id})
    empty_doc = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
    ]}}
    sdk_fakes.docs.queue("documents.get", response=empty_doc)
    sdk_fakes.docs.queue("documents.get", response=empty_doc)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for timestamp in ("fill-time", "post-fill-time"):
        sdk_fakes.drive.queue("files.get", response={
            "id": document_id, "mimeType": "application/vnd.google-apps.document", "modifiedTime": timestamp,
        })

    progress = advance_save(
        client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
        folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
        ensure_unassigned=lambda root: (ensured.append(root) or "unassigned-root"),
    )

    assert progress.document_id == document_id
    assert ensured == ["root"]
    stored = get_operation(client, accepted.operation.operation_id)
    assert stored.selected_folder_id == "unassigned-root"
    assert stored.selected_folder_path == ("_unassigned",)
    create = next(row for row in sdk_fakes.trace if row["operation"] == "files.create")
    assert create["arguments"]["body"]["parents"] == ["unassigned-root"]


def test_explicit_folder_failure_does_not_invoke_unassigned_creation(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    accepted = _accept(client, {
        **ARGUMENTS, "folder_path": ["Missing"],
    }, operation_id="explicit-folder-save")
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None
    calls = []

    with pytest.raises(ValueError, match="not found under the configured root"):
        advance_save(
            client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
            folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
            ensure_unassigned=lambda root: calls.append(root) or "unassigned-root",
        )
    assert calls == []
    assert not [event for event in sdk_fakes.trace if event["operation"] not in {"build"}]


def test_advance_rejects_corrupt_input_before_provider_work(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    accepted = _accept(client)
    client.collection("dialog_operations").document(accepted.operation.operation_id).collection(
        "input_parts"
    ).document("manifest").update({"sha256": "0" * 64})
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None

    with pytest.raises(ValueError, match="does not verify"):
        advance_save(
            client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
            folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
        )
    validation = get_stage(client, accepted.operation.operation_id, "validate")
    assert validation.state.value == "failed" and validation.failures == 1
    assert not [event for event in sdk_fakes.trace if event["operation"] != "build"]


def test_in_flight_validate_reconciles_from_the_complete_input_manifest(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    accepted = _accept(client)
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None
    governor = RetryGovernor(client)
    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            accepted.operation.operation_id, "validate",
            lambda: (_ for _ in ()).throw(UnknownOutcome("validation response lost")),
            now=NOW, owner="worker", fence=lease.fence,
        )

    # Classification fails intentionally after validation so the test makes no provider calls.
    with pytest.raises(ValueError, match="not found under the configured root"):
        advance_save(
            client, accepted.operation.operation_id, owner="worker", fence=lease.fence, now=NOW,
            folders={}, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None, drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"), governor=governor,
        )
    assert get_stage(client, accepted.operation.operation_id, "validate").state.value == "completed"
    assert not [event for event in sdk_fakes.trace if event["operation"] not in {"build"}]


def test_advance_with_stale_owner_does_not_call_provider(firestore_fake, sdk_fakes):
    client = firestore_fake.client_class()
    accepted = _accept(client)
    lease = claim_operation(client, accepted.operation.operation_id, owner="worker", now=NOW, lease_seconds=600)
    assert lease is not None

    with pytest.raises(RuntimeError, match="lease is unavailable"):
        advance_save(
            client, accepted.operation.operation_id, owner="stale", fence=lease.fence, now=NOW,
            folders=FOLDERS, root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None, drive=sdk_fakes.drive, docs=sdk_fakes.docs,
        )
    assert not [event for event in sdk_fakes.trace if event["operation"] != "build"]
