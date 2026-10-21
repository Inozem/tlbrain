from __future__ import annotations

import pytest

from core.integrations.firestore.input_parts import MAX_PART_BYTES, read_input_parts, write_input_parts


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
