from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.index_completions import (
    COMPLETIONS_COLLECTION,
    get_completion,
    publish_completion,
    write_completion,
)


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _receipt(**overrides):
    receipt = {
        "doc_id": "doc-1",
        "root_folder_id": "root-1",
        "content_hash": "sha256:content-1",
        "version": "sha256:content-1",
        "parent_id": "parent-1",
        "parent_id_hash": "sha256:parent-1",
        "provider": "tldv",
        "work_manifest_ref": "manifest-1",
        "checkpoint_refs": ["checkpoint-1"],
        "status": "completed",
        "skipped_utterances": [],
        "completed_at": NOW,
        "error": None,
        "owner": "worker-1",
        "fence": 3,
        "index_fence": 2,
        "resource_epoch": 0,
    }
    receipt.update(overrides)
    return receipt


def _path(receipt):
    return (f"{COMPLETIONS_COLLECTION}/{receipt['doc_id']}/generations/"
            f"{receipt['content_hash']}")


def _guard(client, receipt, *, expiry=None):
    from core.integrations.firestore.mutations import _resource_ref
    _resource_ref(client, f"document:{receipt['doc_id']}").set({
        "resource": f"document:{receipt['doc_id']}",
        "index_owner": receipt["owner"], "index_fence": receipt["index_fence"],
        "index_expires_at": expiry or NOW + timedelta(minutes=5),
        "resource_epoch": receipt["resource_epoch"], "unsettled_effect_ids": [],
    })


def test_write_and_read_use_generation_document_path_and_round_trip_fields(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt(status="building", completed_at=None)

    assert write_completion(client, receipt) == receipt
    assert firestore_fake.documents[_path(receipt)] == receipt
    assert get_completion(client, "doc-1", "sha256:content-1") == receipt
    assert get_completion(client, "doc-1", "missing") is None


@pytest.mark.parametrize("changes", [
    {"status": "synced"},
    {"status": "completed", "completed_at": None},
    {"doc_id": ""},
    {"content_hash": "bad/hash"},
    {"provider": ""},
    {"checkpoint_refs": "checkpoint-1"},
    {"parent_id": 42},
    {"skipped_utterances": ["8"]},
    {"skipped_utterances": [True]},
    {"fence": True},
    {"resource_epoch": -1},
])
def test_invalid_receipts_are_rejected(changes):
    with pytest.raises(ValueError):
        write_completion(None, _receipt(**changes))


def test_read_rejects_malformed_stored_generation(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt()
    client.collection(COMPLETIONS_COLLECTION).document("doc-1").collection("generations").document(
        "sha256:content-1"
    ).set({**receipt, "status": "synced"})

    with pytest.raises(ValueError, match="status"):
        get_completion(client, "doc-1", "sha256:content-1")


def test_publish_rejects_stale_generation_or_index_fence_without_writing(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt()
    client.collection("transcript_index").document("doc-1").set({
        **receipt,
        "content_hash": "sha256:newer-content",
        "version": "sha256:newer-content",
        "index_fence": 3,
        "status": "synced",
    })
    before = dict(firestore_fake.documents)

    assert publish_completion(client, receipt, now=NOW) is False
    assert firestore_fake.documents == before
    assert _path(receipt) not in firestore_fake.documents


def test_current_publication_is_idempotent(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt()
    client.collection("transcript_index").document("doc-1").set({**receipt, "status": "synced"})
    _guard(client, receipt)

    assert publish_completion(client, receipt, now=NOW) is True
    assert publish_completion(client, receipt, now=NOW) is True
    assert get_completion(client, "doc-1", "sha256:content-1") == receipt
    events = [event for event in firestore_fake.trace()
              if event["path"] == _path(receipt) and event["outcome"] == "committed"]
    assert len(events) == 1


def test_takeover_refreshes_same_generation_receipt_to_current_fence(firestore_fake):
    client = firestore_fake.client_class()
    old = _receipt()
    client.collection("transcript_index").document("doc-1").set({**old, "status": "synced"})
    _guard(client, old)
    assert publish_completion(client, old, now=NOW) is True

    current = {**old, "owner": "worker-2", "index_fence": old["index_fence"] + 1,
               "completed_at": NOW + timedelta(seconds=1)}
    client.collection("transcript_index").document("doc-1").set({**current, "status": "syncing"})
    _guard(client, current)
    assert publish_completion(client, current, now=NOW + timedelta(seconds=1)) is True
    assert get_completion(client, "doc-1", old["content_hash"]) == current


def test_publish_requires_live_matching_index_guard(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt()
    client.collection("transcript_index").document("doc-1").set({**receipt, "status": "syncing"})
    _guard(client, receipt, expiry=NOW)
    assert publish_completion(client, receipt, now=NOW) is False
    assert _path(receipt) not in firestore_fake.documents


def test_receipt_preserves_parent_id_and_integer_skipped_utterances(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt(status="building", completed_at=None, skipped_utterances=[0, 8])

    assert write_completion(client, receipt) == receipt
    stored = get_completion(client, "doc-1", "sha256:content-1")
    assert stored["parent_id"] == "parent-1"
    assert stored["skipped_utterances"] == [0, 8]


def test_parent_id_must_match_current_transcript_generation(firestore_fake):
    client = firestore_fake.client_class()
    receipt = _receipt()
    client.collection("transcript_index").document("doc-1").set({
        **receipt, "parent_id": "different-parent", "status": "synced",
    })

    assert publish_completion(client, receipt, now=NOW) is False
    assert _path(receipt) not in firestore_fake.documents


@pytest.mark.parametrize("incoming_status", ["building", "failed"])
def test_write_cannot_downgrade_completed_receipt(firestore_fake, incoming_status):
    client = firestore_fake.client_class()
    completed = _receipt()
    client.collection(COMPLETIONS_COLLECTION).document("doc-1").collection("generations").document(
        "sha256:content-1"
    ).set(completed)
    before = dict(firestore_fake.documents)
    trace_before = len(firestore_fake.trace())

    with pytest.raises(ValueError, match="terminal"):
        write_completion(client, _receipt(status=incoming_status, completed_at=None))

    assert firestore_fake.documents == before
    assert len(firestore_fake.trace()) == trace_before
