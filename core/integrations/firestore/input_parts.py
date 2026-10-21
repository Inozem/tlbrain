"""Byte-partitioned immutable input storage for dialog save operations."""

from __future__ import annotations

import hashlib
from typing import Any

from core.integrations.firestore.client import get_client


OPERATIONS_COLLECTION = "dialog_operations"
INPUT_PARTS_COLLECTION = "input_parts"
MAX_PART_BYTES = 512 * 1024


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def input_manifest_ref(client: Any, input_ref: str) -> Any:
    parts = input_ref.split("/")
    if (len(parts) != 4 or parts[0] != OPERATIONS_COLLECTION
            or parts[2] != INPUT_PARTS_COLLECTION or parts[3] != "manifest"):
        raise ValueError("input_ref must identify an operation input manifest")
    return client.collection(OPERATIONS_COLLECTION).document(parts[1]).collection(INPUT_PARTS_COLLECTION).document("manifest")


def _parts_collection(client: Any, input_ref: str) -> Any:
    parts = input_ref.split("/")
    if (len(parts) != 4 or parts[0] != OPERATIONS_COLLECTION
            or parts[2] != INPUT_PARTS_COLLECTION or parts[3] != "manifest"):
        raise ValueError("input_ref must identify an operation input manifest")
    return client.collection(OPERATIONS_COLLECTION).document(parts[1]).collection(INPUT_PARTS_COLLECTION)


def write_input_parts(client: Any | None, input_ref: str, payload: bytes) -> dict[str, Any]:
    """Write immutable bytes as numbered parts, publishing a ready manifest last."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    db = _db(client)
    manifest_ref = input_manifest_ref(db, input_ref)
    parts_collection = _parts_collection(db, input_ref)
    digest = hashlib.sha256(payload).hexdigest()
    parts = [payload[i:i + MAX_PART_BYTES] for i in range(0, len(payload), MAX_PART_BYTES)] or [b""]
    expected = {"input_ref": input_ref, "ready": True, "part_count": len(parts),
                "byte_length": len(payload), "sha256": digest}
    current = manifest_ref.get()
    if current.exists:
        data = current.to_dict() or {}
        if any(data.get(key) != value for key, value in expected.items()):
            raise ValueError("input reference already belongs to different immutable bytes")
        read_input_parts(db, input_ref)
        return data
    for index, part in enumerate(parts):
        if len(part) > MAX_PART_BYTES:
            raise ValueError("input part exceeds the safe Firestore payload bound")
        parts_collection.document(f"part-{index:06d}").set(
            {"input_ref": input_ref, "index": index, "payload": part}
        )
    # The ready marker is the publication point: readers never consume partial writes.
    manifest_ref.set(expected)
    return expected


def read_input_parts(client: Any | None, input_ref: str) -> bytes:
    """Verify a complete manifest and reconstruct the exact original byte string."""
    db = _db(client)
    manifest_snapshot = input_manifest_ref(db, input_ref).get()
    if not manifest_snapshot.exists:
        raise ValueError("immutable input manifest is absent")
    manifest = manifest_snapshot.to_dict() or {}
    count, length, expected_digest = (manifest.get("part_count"), manifest.get("byte_length"), manifest.get("sha256"))
    if manifest.get("input_ref") != input_ref or manifest.get("ready") is not True:
        raise ValueError("immutable input manifest is not ready")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("immutable input manifest has an invalid part count")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("immutable input manifest has an invalid byte length")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("immutable input manifest has an invalid digest")
    payload_parts: list[bytes] = []
    parts_collection = _parts_collection(db, input_ref)
    for index in range(count):
        snapshot = parts_collection.document(f"part-{index:06d}").get()
        if not snapshot.exists:
            raise ValueError("immutable input is missing a part")
        part = snapshot.to_dict() or {}
        chunk = part.get("payload")
        if part.get("input_ref") != input_ref or part.get("index") != index or not isinstance(chunk, bytes):
            raise ValueError("immutable input parts are corrupt or reordered")
        if len(chunk) > MAX_PART_BYTES:
            raise ValueError("immutable input part exceeds the safe Firestore payload bound")
        payload_parts.append(chunk)
    payload = b"".join(payload_parts)
    if len(payload) != length or hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError("immutable input manifest does not verify")
    return payload
