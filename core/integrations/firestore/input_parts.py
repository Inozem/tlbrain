"""Byte-partitioned immutable input storage for dialog save operations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from core.integrations.firestore.client import get_client
from core.integrations.firestore.save_intents import _completion_operation_binding_matches


OPERATIONS_COLLECTION = "dialog_operations"
INPUT_PARTS_COLLECTION = "input_parts"
MAX_PART_BYTES = 512 * 1024
MAX_INPUT_CLEANUP_OPERATIONS = 50
MAX_INPUT_CLEANUP_PARTS = 400


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


def write_input_parts(
    client: Any | None, input_ref: str, payload: bytes, *,
    write_document: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
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
    write = write_document if write_document is not None else (
        lambda reference, data: reference.set(data)
    )
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
        write(
            parts_collection.document(f"part-{index:06d}"),
            {"input_ref": input_ref, "index": index, "payload": part},
        )
    # The ready marker is the publication point: readers never consume partial writes.
    write(manifest_ref, expected)
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


def _has_matching_completion_certificate(
    client: Any, operation_id: str, operation: dict[str, Any],
) -> bool:
    """Require the save intent's immutable certificate before deleting input bytes."""
    save_intent_id = operation.get("save_intent_id")
    doc_id = operation.get("doc_id")
    authorization = operation.get("cleanup_authorization")
    if (operation.get("state") != "completed" or not save_intent_id or not doc_id
            or not isinstance(authorization, dict)):
        return False
    intent_snapshot = client.collection("dialog_save_intents").document(save_intent_id).get()
    if not intent_snapshot.exists:
        return False
    intent = intent_snapshot.to_dict() or {}
    certificate = intent.get("completion_certificate")
    intent_authorization = intent.get("cleanup_authorization")
    if not isinstance(certificate, dict) or not isinstance(intent_authorization, dict):
        return False
    if not (
        intent.get("save_intent_id") == save_intent_id
        and intent.get("doc_id") == doc_id
        and intent_authorization == authorization
        and certificate.get("certificate_id") == authorization.get("certificate_id")
        and certificate.get("operation_id") == operation_id
        and certificate.get("save_intent_id") == save_intent_id
        and certificate.get("doc_id") == doc_id
        and certificate.get("origin_binding") == authorization.get("origin_binding")
        and bool(certificate.get("certificate_id"))
    ):
        return False
    dispatch_operation_id = intent.get("dispatch_operation_id")
    if dispatch_operation_id == operation_id:
        return True
    origin_binding = certificate.get("origin_binding")
    return (
        isinstance(origin_binding, dict)
        and _completion_operation_binding_matches(
            operation, intent, operation_id=operation_id, origin_binding=origin_binding,
        )
    )


def _input_operation_id(input_ref: Any) -> str | None:
    if not isinstance(input_ref, str):
        return None
    match = re.fullmatch(r"dialog_operations/([^/]+)/input_parts/manifest", input_ref)
    return match.group(1) if match else None


def _cleanup_source_operation_id(
    client: Any, operation_id: str, operation: dict[str, Any],
) -> str | None:
    """Resolve only the operation's own accepted input or its authorized create-gate input."""
    input_ref = operation.get("immutable_input_ref")
    if input_ref is None:
        return operation_id
    source_operation_id = _input_operation_id(input_ref)
    digest = operation.get("immutable_input_digest")
    if (source_operation_id is None or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        return None
    if source_operation_id != operation_id:
        save_intent_id = operation.get("save_intent_id")
        if not save_intent_id:
            return None
        intent_snapshot = client.collection("dialog_save_intents").document(save_intent_id).get()
        if not intent_snapshot.exists:
            return None
        intent = intent_snapshot.to_dict() or {}
        if intent.get("dispatch_operation_id") != source_operation_id:
            return None
        source_snapshot = client.collection(OPERATIONS_COLLECTION).document(source_operation_id).get()
        if not source_snapshot.exists:
            return None
        source_operation = source_snapshot.to_dict() or {}
        origin_binding = operation.get("origin_binding")
        if (not isinstance(origin_binding, dict)
                or source_operation.get("immutable_input_ref") != input_ref
                or source_operation.get("immutable_input_digest") != digest
                or not _completion_operation_binding_matches(
                    source_operation, intent, operation_id=source_operation_id,
                    origin_binding=origin_binding,
                )):
            return None
    elif input_ref != f"{OPERATIONS_COLLECTION}/{operation_id}/{INPUT_PARTS_COLLECTION}/manifest":
        return None

    manifest = input_manifest_ref(client, input_ref).get()
    if not manifest.exists:
        return None
    data = manifest.to_dict() or {}
    if (data.get("input_ref") != input_ref or data.get("ready") is not True
            or data.get("sha256") != digest):
        return None
    return source_operation_id


def cleanup_completed_input_parts(
    client: Any | None = None,
    *,
    operation_limit: int = MAX_INPUT_CLEANUP_OPERATIONS,
    part_limit: int = MAX_INPUT_CLEANUP_PARTS,
) -> int:
    """Delete one bounded page of parts for each certificate-authorized save.

    Pending operations are selected only by the marker written atomically with
    their completion certificate. Deleting a page leaves the first remaining
    page at the start of the subcollection query, so a later Checker run resumes
    after process failure without a cursor or historical migration.
    """
    if (isinstance(operation_limit, bool) or not isinstance(operation_limit, int)
            or not 1 <= operation_limit <= MAX_INPUT_CLEANUP_OPERATIONS):
        raise ValueError(f"operation_limit must be from 1 to {MAX_INPUT_CLEANUP_OPERATIONS}")
    if (isinstance(part_limit, bool) or not isinstance(part_limit, int)
            or not 1 <= part_limit <= MAX_INPUT_CLEANUP_PARTS):
        raise ValueError(f"part_limit must be from 1 to {MAX_INPUT_CLEANUP_PARTS}")

    db = _db(client)
    operations = db.collection(OPERATIONS_COLLECTION)
    candidates = operations.where("input_cleanup_state", "==", "pending").limit(operation_limit).stream()
    deleted_count = 0
    for snapshot in candidates:
        operation = snapshot.to_dict() or {}
        if not _has_matching_completion_certificate(db, snapshot.id, operation):
            continue
        source_operation_id = _cleanup_source_operation_id(db, snapshot.id, operation)
        if source_operation_id is None:
            continue
        part_collection = operations.document(source_operation_id).collection(INPUT_PARTS_COLLECTION)
        # The manifest has no index field and is retained as the input tombstone.
        parts = part_collection.where("index", ">=", 0).limit(part_limit).stream()
        batch = db.batch()
        page_count = 0
        for part in parts:
            batch.delete(part.reference)
            page_count += 1
        if page_count:
            batch.commit()
            deleted_count += page_count

        remaining = part_collection.where("index", ">=", 0).limit(1).stream()
        if not next(iter(remaining), None):
            operations.document(snapshot.id).update({"input_cleanup_state": "complete"})
    return deleted_count
