"""Transactional Firestore adapter for generic per-context FIFO queues."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore

from core.operations.queue import (
    HEARTBEAT_INTERVAL_SECONDS,
    LEASE_SECONDS,
    OperationLease,
    QueueOperationState,
    QueuedOperation,
)


OPERATIONS_SUBCOLLECTION = "operations"


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _operation_ref(context_ref: Any, operation_id: str) -> Any:
    if not isinstance(operation_id, str) or not operation_id or "/" in operation_id:
        raise ValueError("operation_id must be a non-empty Firestore document ID")
    return context_ref.collection(OPERATIONS_SUBCOLLECTION).document(operation_id)


def _operation_from(data: dict[str, Any]) -> QueuedOperation:
    return QueuedOperation(
        operation_id=data["operation_id"], sequence=int(data["sequence"]),
        state=QueueOperationState(data["state"]), identity_digest=data["identity_digest"],
        input_ref=data["input_ref"], created_at=data["created_at"],
        finished_at=data.get("finished_at"),
    )


def register_operation(
    client: Any,
    context_ref: Any,
    operation_id: str,
    *,
    identity_digest: str,
    input_ref: str,
    now: datetime,
) -> QueuedOperation:
    """Append an operation atomically, with exact replay idempotency."""
    now = _aware(now)
    if not identity_digest or not input_ref:
        raise ValueError("identity_digest and input_ref are required")
    operation_ref = _operation_ref(context_ref, operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> QueuedOperation:
        context_snapshot = context_ref.get(transaction=transaction)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if operation_snapshot.exists:
            data = operation_snapshot.to_dict() or {}
            if data.get("identity_digest") != identity_digest or data.get("input_ref") != input_ref:
                raise ValueError("operation ID already belongs to a different immutable identity")
            return _operation_from(data)

        context = context_snapshot.to_dict() or {}
        sequence = int(context.get("next_operation_sequence", 1))
        tail_id = context.get("queue_tail_operation_id")
        tail_ref = _operation_ref(context_ref, tail_id) if tail_id else None
        tail_data: dict[str, Any] | None = None
        if tail_ref is not None:
            tail_snapshot = tail_ref.get(transaction=transaction)
            if not tail_snapshot.exists:
                raise RuntimeError("queue tail record is missing")
            tail_data = tail_snapshot.to_dict() or {}
            if tail_data.get("next_operation_id") is not None:
                raise RuntimeError("queue tail is already linked")

        data = {
            "operation_id": operation_id,
            "sequence": sequence,
            "state": QueueOperationState.PENDING.value,
            "identity_digest": identity_digest,
            "input_ref": input_ref,
            "created_at": now,
            "finished_at": None,
            "next_operation_id": None,
        }
        if tail_ref is not None and tail_data is not None:
            tail_data["next_operation_id"] = operation_id
            transaction.set(tail_ref, tail_data)
        transaction.set(operation_ref, data)
        context_update: dict[str, Any] = {
            "next_operation_sequence": sequence + 1,
            "queue_tail_operation_id": operation_id,
        }
        if not context.get("reserved_operation_id"):
            context_update.update({
                "head_sequence": sequence,
                "reserved_operation_id": operation_id,
            })
        transaction.set(context_ref, {**context, **context_update})
        return _operation_from(data)

    return _txn(client.transaction())


def get_operation(context_ref: Any, operation_id: str) -> QueuedOperation | None:
    snapshot = _operation_ref(context_ref, operation_id).get()
    return _operation_from(snapshot.to_dict() or {}) if snapshot.exists else None


def claim_head(
    client: Any,
    context_ref: Any,
    *,
    owner: str,
    now: datetime,
    lease_seconds: int | float = LEASE_SECONDS,
) -> OperationLease | None:
    """Claim only the reserved head, fencing every new ownership."""
    now = _aware(now)
    if not owner or lease_seconds <= 0:
        raise ValueError("owner and a positive lease_seconds are required")

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationLease | None:
        context_snapshot = context_ref.get(transaction=transaction)
        if not context_snapshot.exists:
            return None
        context = context_snapshot.to_dict() or {}
        operation_id = context.get("reserved_operation_id")
        if not operation_id:
            return None
        operation_ref = _operation_ref(context_ref, operation_id)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not operation_snapshot.exists:
            raise RuntimeError("reserved queue head record is missing")
        operation = operation_snapshot.to_dict() or {}
        if operation.get("state") in {QueueOperationState.COMPLETED.value, QueueOperationState.FAILED.value}:
            raise RuntimeError("reserved queue head is terminal")

        expiry = context.get("lock_expires_at")
        active_id = context.get("active_operation_id")
        if active_id and isinstance(expiry, datetime) and expiry > now:
            return None

        fence = int(context.get("fence", 0)) + 1
        expires_at = now + timedelta(seconds=lease_seconds)
        context.update({
            "active_operation_id": operation_id,
            "lock_owner": owner,
            "lock_expires_at": expires_at,
            "fence": fence,
        })
        operation["state"] = QueueOperationState.RUNNING.value
        transaction.set(context_ref, context)
        transaction.set(operation_ref, operation)
        return OperationLease(operation_id, int(operation["sequence"]), owner, fence, expires_at)

    return _txn(client.transaction())


def heartbeat(
    client: Any,
    context_ref: Any,
    *,
    operation_id: str,
    owner: str,
    fence: int,
    now: datetime,
    lease_seconds: int | float = LEASE_SECONDS,
) -> OperationLease | None:
    """Extend only the matching, unexpired head lease."""
    now = _aware(now)
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationLease | None:
        snapshot = context_ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        context = snapshot.to_dict() or {}
        expiry = context.get("lock_expires_at")
        if not _valid_lease(context, operation_id, owner, fence, now, expiry):
            return None
        expires_at = now + timedelta(seconds=lease_seconds)
        context["lock_expires_at"] = expires_at
        transaction.set(context_ref, context)
        return OperationLease(operation_id, int(context["head_sequence"]), owner, fence, expires_at)

    return _txn(client.transaction())


def release_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
) -> bool:
    """Yield the current lease while preserving the same queue head."""
    now = _aware(now)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        context_snapshot = context_ref.get(transaction=transaction)
        if not context_snapshot.exists:
            return False
        context = context_snapshot.to_dict() or {}
        expiry = context.get("lock_expires_at")
        if not _valid_lease(context, operation_id, owner, fence, now, expiry):
            return False
        operation_ref = _operation_ref(context_ref, operation_id)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not operation_snapshot.exists:
            return False
        operation = operation_snapshot.to_dict() or {}
        context.update({"active_operation_id": None, "lock_owner": None, "lock_expires_at": None})
        transaction.set(context_ref, context)
        return True

    return _txn(client.transaction())


def complete_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
) -> bool:
    return _finish_head(client, context_ref, operation_id=operation_id, owner=owner, fence=fence, now=now,
                        state=QueueOperationState.COMPLETED)


def fail_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
) -> bool:
    return _finish_head(client, context_ref, operation_id=operation_id, owner=owner, fence=fence, now=now,
                        state=QueueOperationState.FAILED)


def _finish_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
    state: QueueOperationState,
) -> bool:
    now = _aware(now)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        context_snapshot = context_ref.get(transaction=transaction)
        if not context_snapshot.exists:
            return False
        context = context_snapshot.to_dict() or {}
        expiry = context.get("lock_expires_at")
        if not _valid_lease(context, operation_id, owner, fence, now, expiry):
            return False
        operation_ref = _operation_ref(context_ref, operation_id)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not operation_snapshot.exists:
            return False
        operation = operation_snapshot.to_dict() or {}
        next_id = operation.get("next_operation_id")
        next_ref = _operation_ref(context_ref, next_id) if next_id else None
        next_data: dict[str, Any] | None = None
        if next_ref is not None:
            next_snapshot = next_ref.get(transaction=transaction)
            if not next_snapshot.exists:
                raise RuntimeError("next queue operation record is missing")
            next_data = next_snapshot.to_dict() or {}
            if next_data.get("state") in {QueueOperationState.COMPLETED.value, QueueOperationState.FAILED.value}:
                raise RuntimeError("next queue operation is unexpectedly terminal")

        operation.update({"state": state.value, "finished_at": now})
        context.update({"active_operation_id": None, "lock_owner": None, "lock_expires_at": None})
        if next_id and next_data is not None:
            context.update({"reserved_operation_id": next_id, "head_sequence": int(next_data["sequence"])})
        else:
            context.update({"reserved_operation_id": None, "head_sequence": None,
                            "queue_tail_operation_id": None})
        transaction.set(operation_ref, operation)
        transaction.set(context_ref, context)
        return True

    return _txn(client.transaction())


def _valid_lease(
    context: dict[str, Any], operation_id: str, owner: str, fence: int, now: datetime,
    expiry: datetime | None,
) -> bool:
    return (
        context.get("reserved_operation_id") == operation_id
        and context.get("active_operation_id") == operation_id
        and context.get("lock_owner") == owner
        and context.get("fence") == fence
        and isinstance(expiry, datetime)
        and expiry > now
    )


__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS", "LEASE_SECONDS", "claim_head", "complete_head",
    "fail_head", "get_operation", "heartbeat", "register_operation", "release_head",
]
