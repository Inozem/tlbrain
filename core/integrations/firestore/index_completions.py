"""Firestore persistence for per-generation index completion receipts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.client import get_client


COMPLETIONS_COLLECTION = "index_completions"
TRANSCRIPT_INDEX_COLLECTION = "transcript_index"
RECEIPT_STATUSES = frozenset({"building", "completed", "failed"})
_REQUIRED_FIELDS = frozenset({
    "doc_id", "root_folder_id", "content_hash", "version", "parent_id", "parent_id_hash", "provider",
    "work_manifest_ref", "checkpoint_refs", "status", "skipped_utterances", "completed_at",
    "error", "owner", "fence", "index_fence", "resource_epoch",
})
_CURRENT_GENERATION_FIELDS = (
    "doc_id", "root_folder_id", "content_hash", "version", "parent_id", "parent_id_hash", "provider",
    "fence", "index_fence", "resource_epoch",
)


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _document_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(f"{field} must be a non-empty Firestore document ID")
    return value


def _validate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("completion receipt must be a mapping")
    data = dict(receipt)
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"completion receipt is missing required fields: {', '.join(sorted(missing))}")
    for field in ("doc_id", "root_folder_id", "content_hash", "version", "parent_id_hash", "provider",
                  "work_manifest_ref", "owner"):
        value = data[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    if data["parent_id"] is not None and not isinstance(data["parent_id"], str):
        raise ValueError("parent_id must be a string or None")
    _document_id(data["doc_id"], "doc_id")
    _document_id(data["content_hash"], "content_hash")
    if data["status"] not in RECEIPT_STATUSES:
        raise ValueError("status must be building, completed, or failed")
    checkpoint_refs = data["checkpoint_refs"]
    if not isinstance(checkpoint_refs, (list, tuple)) or any(
        not isinstance(item, str) for item in checkpoint_refs
    ):
        raise ValueError("checkpoint_refs must be a list of strings")
    data["checkpoint_refs"] = list(checkpoint_refs)
    skipped_utterances = data["skipped_utterances"]
    if not isinstance(skipped_utterances, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in skipped_utterances
    ):
        raise ValueError("skipped_utterances must be a list of integers")
    data["skipped_utterances"] = list(skipped_utterances)
    for field in ("fence", "index_fence", "resource_epoch"):
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    completed_at = data["completed_at"]
    if completed_at is not None and (
        not isinstance(completed_at, datetime) or completed_at.tzinfo is None
        or completed_at.utcoffset() is None
    ):
        raise ValueError("completed_at must be a timezone-aware datetime or None")
    if data["status"] == "completed" and completed_at is None:
        raise ValueError("completed receipts require completed_at")
    if data["error"] is not None and not isinstance(data["error"], (str, Mapping)):
        raise ValueError("error must be a string, mapping, or None")
    return data


def _receipt_ref(db: Any, doc_id: str, content_hash: str) -> Any:
    return (db.collection(COMPLETIONS_COLLECTION).document(doc_id)
            .collection("generations").document(content_hash))


def _validated_stored(data: Mapping[str, Any], *, doc_id: str, content_hash: str) -> dict[str, Any]:
    receipt = _validate(data)
    if receipt["doc_id"] != doc_id or receipt["content_hash"] != content_hash:
        raise ValueError("stored completion receipt identity does not match its Firestore path")
    return receipt


def get_completion(client: Any | None, doc_id: str, content_hash: str) -> dict[str, Any] | None:
    """Read and validate one generation receipt, returning None when absent."""
    _document_id(doc_id, "doc_id")
    _document_id(content_hash, "content_hash")
    snapshot = _receipt_ref(_db(client), doc_id, content_hash).get()
    if not snapshot.exists:
        return None
    return _validated_stored(snapshot.to_dict() or {}, doc_id=doc_id, content_hash=content_hash)


def write_completion(client: Any | None, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a validated nonterminal receipt unless completion is already terminal."""
    data = _validate(receipt)
    if data["status"] == "completed":
        raise ValueError("completed receipts must be published with publish_completion")
    db = _db(client)
    receipt_ref = _receipt_ref(db, data["doc_id"], data["content_hash"])

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> dict[str, Any]:
        snapshot = receipt_ref.get(transaction=transaction)
        if snapshot.exists:
            existing = _validated_stored(
                snapshot.to_dict() or {}, doc_id=data["doc_id"], content_hash=data["content_hash"],
            )
            if existing["status"] == "completed":
                raise ValueError("completed receipts are terminal and cannot be overwritten")
        transaction.set(receipt_ref, data)
        return data

    return _txn(db.transaction())


def _is_current(transcript: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    return all(transcript.get(field) == receipt.get(field) for field in _CURRENT_GENERATION_FIELDS)


def publish_completion(
    client: Any | None,
    receipt: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """Publish a completed receipt only while its transcript generation and fence are current."""
    data = _validate(receipt)
    if data["status"] != "completed":
        raise ValueError("publish_completion requires a completed receipt")
    if now is not None and (
        not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None
    ):
        raise ValueError("now must be timezone-aware")
    db = _db(client)
    transcript_ref = db.collection(TRANSCRIPT_INDEX_COLLECTION).document(data["doc_id"])
    receipt_ref = _receipt_ref(db, data["doc_id"], data["content_hash"])

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        transcript_snapshot = transcript_ref.get(transaction=transaction)
        receipt_snapshot = receipt_ref.get(transaction=transaction)
        if not transcript_snapshot.exists:
            return False
        transcript = transcript_snapshot.to_dict() or {}
        if not _is_current(transcript, data):
            return False
        if receipt_snapshot.exists:
            existing = _validated_stored(
                receipt_snapshot.to_dict() or {}, doc_id=data["doc_id"], content_hash=data["content_hash"],
            )
            if existing != data:
                raise ValueError("completion generation already has a different receipt")
            return True
        transaction.set(receipt_ref, data)
        return True

    return _txn(db.transaction())
