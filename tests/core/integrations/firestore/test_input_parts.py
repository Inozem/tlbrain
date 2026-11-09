from __future__ import annotations

import hashlib

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


def test_document_writer_receives_each_part_and_ready_manifest(firestore_fake):
    client = firestore_fake.client_class()
    writes = []

    def write_document(reference, data):
        writes.append((reference.id, data.copy()))
        reference.set(data)

    payload = b"x" * (MAX_PART_BYTES + 1)
    write_input_parts(client, _ref("callback-input"), payload, write_document=write_document)

    assert [document_id for document_id, _ in writes] == [
        "part-000000", "part-000001", "manifest",
    ]
    assert writes[-1][1]["ready"] is True


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


def _authorize_retry_cleanup(client, *, digest):
    gate_id, retry_id = "retry-create-gate", "retry-operation"
    input_ref = _ref(gate_id)
    binding = {"context_id": "context-1", "draft_id": "draft-1", "manifest_digest": digest}
    retry_authorization = {
        "operation_id": retry_id,
        "create_gate_operation_id": gate_id,
        "previous_operation_id": gate_id,
        "context_id": binding["context_id"],
        "draft_id": binding["draft_id"],
        "manifest_digest": digest,
        "caller_binding": "principal:alice",
        "immutable_input_digest": digest,
    }
    client.collection("dialog_operations").document(gate_id).set({
        "origin_binding": binding,
        "immutable_input_ref": input_ref,
        "immutable_input_digest": digest,
        "origin_operation_ref": binding["context_id"],
    })
    intent_id = "retry-intent"
    client.collection("dialog_save_intents").document(intent_id).set({
        "save_intent_id": intent_id,
        "dispatch_operation_id": gate_id,
        "doc_id": "doc-retry",
        "cleanup_authorization": {},
    })
    cleanup_authorization = {
        "certificate_id": "retry-certificate",
        "save_intent_id": intent_id,
        "operation_id": retry_id,
        "origin_binding": binding,
    }
    certificate = {
        **cleanup_authorization,
        "doc_id": "doc-retry",
    }
    retry_operation = {
        "state": "completed",
        "save_intent_id": intent_id,
        "doc_id": "doc-retry",
        "cleanup_authorization": cleanup_authorization,
        "input_cleanup_state": "pending",
        "origin_binding": binding,
        "retry_authorization": retry_authorization,
        "retry_of_operation_id": gate_id,
        "retry_draft_id": binding["draft_id"],
        "caller_binding": "principal:alice",
        "immutable_input_ref": input_ref,
        "immutable_input_digest": digest,
    }
    client.collection("dialog_operations").document(retry_id).set(retry_operation)
    client.collection("dialog_save_intents").document(intent_id).update({
        "cleanup_authorization": cleanup_authorization,
        "completion_certificate": certificate,
    })
    return gate_id, retry_id, input_ref, digest


def test_retry_cleanup_removes_only_original_accepted_input_in_bounded_pages(firestore_fake):
    client = firestore_fake.client_class()
    payload = b"o" * (MAX_PART_BYTES * 2 + 1)
    digest = hashlib.sha256(payload).hexdigest()
    gate_id, retry_id, input_ref, digest = _authorize_retry_cleanup(client, digest=digest)
    manifest = write_input_parts(client, input_ref, payload)
    assert manifest["sha256"] == digest
    write_input_parts(client, _ref(retry_id), b"retry-local-bytes")
    parts = client.collection("dialog_operations").document(gate_id).collection("input_parts")
    retry_parts = client.collection("dialog_operations").document(retry_id).collection("input_parts")
    operation_ref = client.collection("dialog_operations").document(retry_id)

    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=1) == 1
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "pending"
    assert len(list(parts.stream())) == 3
    assert len(list(retry_parts.stream())) == 2
    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=1) == 1
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "pending"
    assert len(list(parts.stream())) == 2
    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=1) == 1
    assert operation_ref.get().to_dict()["input_cleanup_state"] == "complete"
    assert len(list(parts.stream())) == 1
    assert len(list(retry_parts.stream())) == 2
    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=1) == 0


@pytest.mark.parametrize("invalid_reason", [
    "input", "digest", "create_gate", "authorization", "foreign_certificate",
    "malformed_input_ref", "not_ready_manifest", "mismatched_manifest_digest",
])
def test_retry_cleanup_rejects_crossed_or_foreign_bindings(firestore_fake, invalid_reason):
    client = firestore_fake.client_class()
    original_payload = b"original input"
    digest = hashlib.sha256(original_payload).hexdigest()
    gate_id, retry_id, input_ref, _ = _authorize_retry_cleanup(client, digest=digest)
    write_input_parts(client, _ref(gate_id), original_payload)
    write_input_parts(client, _ref(retry_id), b"retry-local input")
    operation_ref = client.collection("dialog_operations").document(retry_id)
    if invalid_reason == "input":
        foreign_ref = _ref("foreign-input")
        write_input_parts(client, foreign_ref, b"foreign input")
        operation_ref.update({"immutable_input_ref": foreign_ref})
    elif invalid_reason == "digest":
        crossed_digest = "b" * 64
        operation_ref.update({
            "immutable_input_digest": crossed_digest,
            "retry_authorization.immutable_input_digest": crossed_digest,
        })
    elif invalid_reason == "create_gate":
        client.collection("dialog_save_intents").document("retry-intent").update({
            "dispatch_operation_id": "wrong-gate",
        })
    elif invalid_reason == "authorization":
        operation_ref.update({"retry_authorization.create_gate_operation_id": "wrong-gate"})
    elif invalid_reason == "foreign_certificate":
        client.collection("dialog_save_intents").document("retry-intent").update({
            "completion_certificate.operation_id": "foreign-operation",
        })
    elif invalid_reason == "malformed_input_ref":
        operation_ref.update({"immutable_input_ref": "dialog_operations//input_parts/manifest"})
    elif invalid_reason == "not_ready_manifest":
        client.collection("dialog_operations").document(gate_id).collection("input_parts").document("manifest").update({
            "ready": False,
        })
    elif invalid_reason == "mismatched_manifest_digest":
        client.collection("dialog_operations").document(gate_id).collection("input_parts").document("manifest").update({
            "sha256": "b" * 64,
        })

    assert cleanup_completed_input_parts(client, operation_limit=1, part_limit=10) == 0
    assert client.collection("dialog_operations").document(gate_id).collection("input_parts").document("part-000000").get().exists
    assert client.collection("dialog_operations").document(retry_id).collection("input_parts").document("part-000000").get().exists
    if invalid_reason == "input":
        assert client.collection("dialog_operations").document("foreign-input").collection("input_parts").document("part-000000").get().exists
