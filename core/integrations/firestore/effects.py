"""Durable resource-ID allocations attached to operation effects."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Callable
from uuid import UUID

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.operations import OPERATIONS_COLLECTION, _valid_lease

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


def _is_drive_id(value: Any) -> bool:
    return isinstance(value, str) and _DRIVE_ID.fullmatch(value) is not None


class AllocationConflict(ValueError):
    """An effect resource slot is already bound to another immutable intent."""


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _valid_id(kind: str, value: Any) -> bool:
    if not isinstance(value, str) or not value or "/" in value:
        return False
    if kind == "user_fact":
        try:
            return str(UUID(value)) == value
        except (ValueError, AttributeError):
            return False
    return _is_drive_id(value)


def get_effect_allocation(client: Any | None, operation_id: str, effect_id: str) -> dict[str, Any] | None:
    """Read an effect's one persisted resource allocation, if present."""
    snapshot = (_db(client).collection(OPERATIONS_COLLECTION).document(operation_id)
                .collection("effects").document(effect_id).collection("allocations").document("resource").get())
    return snapshot.to_dict() if snapshot.exists else None


def validate_effect_allocation_lease(
    client: Any | None, operation_id: str, *, owner: str, fence: int, now: datetime,
) -> bool:
    """Check lease before candidate generation; the allocation transaction checks it again."""
    snapshot = _db(client).collection(OPERATIONS_COLLECTION).document(operation_id).get()
    return snapshot.exists and _valid_lease(snapshot.to_dict() or {}, owner, fence, now)


def allocate_effect_resource(
    client: Any | None, *, operation_id: str, effect_id: str, kind: str,
    intent_digest: str, owner: str, fence: int, now: datetime, candidate: str,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Persist one resource ID before provider mutation; transactions choose one concurrent winner."""
    if (not isinstance(operation_id, str) or not operation_id or "/" in operation_id
            or not isinstance(effect_id, str) or not effect_id or "/" in effect_id
            or kind not in {"user_fact", "drive_folder"}
            or not isinstance(intent_digest, str) or not _DIGEST.fullmatch(intent_digest)
            or not isinstance(owner, str) or not owner
            or isinstance(fence, bool) or not isinstance(fence, int) or fence < 1
            or not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None
            or not _valid_id(kind, candidate)):
        raise ValueError("invalid effect resource allocation")
    now = now.astimezone(timezone.utc)
    db = _db(client)
    op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    alloc_ref = op_ref.collection("effects").document(effect_id).collection("allocations").document("resource")
    proposed = {"operation_id": operation_id, "effect_id": effect_id, "kind": kind,
                "intent_digest": intent_digest, "resource_id": candidate,
                "intent_marker": _intent_marker(operation_id, effect_id, kind, intent_digest)}

    @firestore.transactional
    def _txn(transaction):
        commit_now = clock() if clock is not None else now
        if not isinstance(commit_now, datetime) or commit_now.tzinfo is None or commit_now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        op_snapshot = op_ref.get(transaction=transaction)
        alloc_snapshot = alloc_ref.get(transaction=transaction)
        if alloc_snapshot.exists:
            existing = alloc_snapshot.to_dict() or {}
            if any(existing.get(key) != proposed[key] for key in
                   ("operation_id", "effect_id", "kind", "intent_digest", "intent_marker")):
                raise AllocationConflict("effect resource identity conflicts")
            if not _valid_id(kind, existing.get("resource_id")):
                raise AllocationConflict("effect resource allocation is invalid")
            return existing
        if not op_snapshot.exists or not _valid_lease(op_snapshot.to_dict() or {}, owner, fence, commit_now):
            raise AllocationConflict("current operation lease is required")
        transaction.set(alloc_ref, proposed)
        return proposed

    return _txn(db.transaction())


def _intent_marker(operation_id: str, effect_id: str, kind: str, intent_digest: str) -> str:
    return "v1:" + hashlib.sha256("\0".join((operation_id, effect_id, kind, intent_digest)).encode()).hexdigest()
