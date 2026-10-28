from __future__ import annotations

import pytest

from core.integrations.firestore.input_parts import (
    MAX_PART_BYTES,
    cleanup_completed_input_parts,
    read_input_parts,
    write_input_parts,
)


def _ref(operation_id: str) -> str:
    return f"dialog_operations/{operation_id}/input_parts/manifest"


def test_input_round_trip_across_multiple_megabytes_and_respects_part_bound(firestore_fake):
    client = firestore_fake.client_class()
    payload = (b"exact-bytes-" * 180_000)
    manifest = write_input_parts(client, _ref("large-input"), payload)

    assert manifest["byte_length"] == len(payload)
    assert manifest["part_count"] > 3
    assert read_input_parts(client, _ref("large-input")) == payload
    operation_ref = client.collection("dialog_operations").document("large-input")
    part_docs = [operation_ref.collection("input_parts").document(f"part-{i:06d}").get().to_dict()
                 for i in range(manifest["part_count"])]
    assert part_docs and all(len(row["payload"]) <= MAX_PART_BYTES for row in part_docs)


@pytest.mark.parametrize("corruption", ["missing", "reordered", "payload"])
def test_input_reader_rejects_incomplete_or_corrupt_manifest(firestore_fake, corruption):
    client = firestore_fake.client_class()
    write_input_parts(client, _ref("broken-input"), b"a" * (MAX_PART_BYTES + 3))
    parts = client.collection("dialog_operations").document("broken-input").collection("input_parts")
    if corruption == "missing":
        parts.document("part-000000").delete()
    elif corruption == "reordered":
        parts.document("part-000000").update({"index": 8})
    else:
        parts.document("part-000000").update({"payload": b"changed"})

    with pytest.raises(ValueError):
        read_input_parts(client, _ref("broken-input"))


def test_input_reader_rejects_manifest_without_ready_publication(firestore_fake):
    client = firestore_fake.client_class()
    write_input_parts(client, _ref("not-ready"), b"abc")
    client.collection("dialog_operations").document("not-ready").collection("input_parts").document("manifest").update({"ready": False})

    with pytest.raises(ValueError, match="not ready"):
        read_input_parts(client, _ref("not-ready"))


def _authorize_cleanup(client, operation_id: str, *, certificate_id: str = "certificate-1"):
    binding = {"context_id": "context-1", "draft_id": "draft-1", "manifest_digest": "digest-1"}
    authorization = {
        "certificate_id": certificate_id,
        "save_intent_id": "intent-1",
        "operation_id": operation_id,
        "origin_binding": binding,
    }
    certificate = {
        "certificate_id": certificate_id,
        "save_intent_id": "intent-1",
        "operation_id": operation_id,
        "doc_id": "doc-1",
        "origin_binding": binding,
    }
    client.collection("dialog_operations").document(operation_id).set({
        "state": "completed", "save_intent_id": "intent-1", "doc_id": "doc-1",
        "cleanup_authorization": authorization, "input_cleanup_state": "pending",
    })
    client.collection("dialog_save_intents").document("intent-1").set({
        "save_intent_id": "intent-1", "dispatch_operation_id": operation_id,
        "doc_id": "doc-1", "cleanup_authorization": authorization,
        "completion_certificate": certificate,
    })


def test_cleanup_is_certificate_gated_bounded_and_restartable(firestore_fake):
    client = firestore_fake.client_class()
    operation_id = "cleanup-resume"
    write_input_parts(client, _ref(operation_id), b"x" * (MAX_PART_BYTES * 4 + 1))
    _authorize_cleanup(client, operation_id)
    operation_ref = client.collection("dialog_operations").document(operation_id)
    parts = operation_ref.collection("input_parts")

    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=2) == 2
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "pending"
    assert len(list(parts.stream())) == 4

    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=2) == 2
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "pending"
    assert len(list(parts.stream())) == 2
    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=2) == 1
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "complete"
    assert len(list(parts.stream())) == 1  # The compact manifest remains as a tombstone.


@pytest.mark.parametrize("invalid_reason", ["unconfirmed", "foreign_certificate"])
def test_cleanup_retains_parts_for_unconfirmed_or_mismatched_certificate(firestore_fake, invalid_reason):
    client = firestore_fake.client_class()
    operation_id = "cleanup-unauthorized"
    write_input_parts(client, _ref(operation_id), b"retain me")
    _authorize_cleanup(client, operation_id)
    operation_ref = client.collection("dialog_operations").document(operation_id)
    if invalid_reason == "unconfirmed":
        operation_ref.update({"state": "running"})
    else:
        client.collection("dialog_save_intents").document("intent-1").update({
            "completion_certificate.certificate_id": "foreign-certificate",
        })
    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=2) == 0
    assert operation_ref.collection("input_parts").document("part-000000").get().exists
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "pending"
