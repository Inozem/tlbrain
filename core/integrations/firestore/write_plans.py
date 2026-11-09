"""Immutable, metadata-only plans for shared guarded write effects."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from collections.abc import Callable

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.operations import OPERATIONS_COLLECTION, _valid_lease


class WritePlanConflict(ValueError):
    """An effect was already planned with different arguments or targets."""


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def get_write_plan(client: Any | None, operation_id: str, effect_id: str) -> dict[str, Any] | None:
    db = client if client is not None else get_client()
    snap = (db.collection(OPERATIONS_COLLECTION).document(operation_id)
            .collection("write_plans").document(effect_id).get())
    return snap.to_dict() if snap.exists else None


def persist_write_plan(
    client: Any | None, *, operation_id: str, effect_id: str, owner: str, fence: int,
    now: datetime, arguments_digest: str, plan: dict[str, Any],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Persist a resolved plan before provider mutation and reject changed intent."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not isinstance(plan, dict) or any(not isinstance(k, str) for k in plan):
        raise ValueError("write plan must be a metadata dictionary")
    if not isinstance(arguments_digest, str) or not arguments_digest.startswith("sha256:"):
        raise ValueError("arguments digest is required")
    db = client if client is not None else get_client()
    op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    plan_ref = op_ref.collection("write_plans").document(effect_id)
    clean_plan = dict(plan)
    proposed = {
        "operation_id": operation_id, "effect_id": effect_id,
        "arguments_digest": arguments_digest, "plan_digest": digest(clean_plan),
        "plan": clean_plan, "updated_at": now.astimezone(timezone.utc),
    }

    @firestore.transactional
    def _txn(transaction):
        commit_now = clock() if clock is not None else now
        if not isinstance(commit_now, datetime) or commit_now.tzinfo is None or commit_now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        op_snap = op_ref.get(transaction=transaction)
        plan_snap = plan_ref.get(transaction=transaction)
        if not op_snap.exists or not _valid_lease(op_snap.to_dict() or {}, owner, fence, commit_now):
            raise WritePlanConflict("current operation lease is required")
        if plan_snap.exists:
            existing = plan_snap.to_dict() or {}
            if (existing.get("operation_id") != operation_id
                    or existing.get("effect_id") != effect_id
                    or existing.get("arguments_digest") != arguments_digest):
                raise WritePlanConflict("effect plan belongs to different arguments")
            return existing
        proposed["updated_at"] = commit_now.astimezone(timezone.utc)
        transaction.set(plan_ref, proposed)
        return proposed

    return _txn(db.transaction())
