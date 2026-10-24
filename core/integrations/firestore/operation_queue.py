"""Transactional Firestore adapter for generic per-context FIFO queues."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from google.cloud import firestore

from core.operations.queue import (
    HEARTBEAT_INTERVAL_SECONDS,
    LEASE_SECONDS,
    OperationLease,
    QueueOperationState,
    QueuedOperation,
)
from core.operations.models import OperationError, OperationStageRecord, StageState


OPERATIONS_SUBCOLLECTION = "operations"
STAGES_SUBCOLLECTION = "stages"


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _operation_ref(context_ref: Any, operation_id: str) -> Any:
    if not isinstance(operation_id, str) or not operation_id or "/" in operation_id:
        raise ValueError("operation_id must be a non-empty Firestore document ID")
    return context_ref.collection(OPERATIONS_SUBCOLLECTION).document(operation_id)


def _stage_ref(context_ref: Any, operation_id: str, stage_key: str) -> Any:
    if not isinstance(stage_key, str) or not stage_key or "/" in stage_key:
        raise ValueError("stage_key must be a non-empty Firestore document ID")
    return _operation_ref(context_ref, operation_id).collection(STAGES_SUBCOLLECTION).document(stage_key)


def _stage_from(data: Mapping[str, Any]) -> OperationStageRecord:
    error = data.get("last_error")
    return OperationStageRecord(
        stage_key=data["stage_key"], state=StageState(data.get("state", "pending")),
        failures=int(data.get("failures", 0)), attempts_started=int(data.get("attempts_started", 0)),
        in_flight_attempt_id=data.get("in_flight_attempt_id"),
        retry_not_before=data.get("retry_not_before"),
        last_error=OperationError(**error) if isinstance(error, Mapping) else None,
        checkpoint_refs=tuple(data.get("checkpoint_refs", ())), updated_at=data.get("updated_at"),
        result=data.get("result"),
    )


class QueueOperationRetryStore:
    """RetryGovernor store backed by a context queue operation and its lease."""

    def __init__(self, client: Any, context_ref: Any) -> None:
        self.client = client
        self.context_ref = context_ref

    def _valid(self, transaction: Any, operation_id: str, owner: str, fence: int, now: datetime) -> bool:
        snapshot = self.context_ref.get(transaction=transaction)
        context = snapshot.to_dict() or {} if snapshot.exists else {}
        return _valid_lease(context, operation_id, owner, fence, now, context.get("lock_expires_at"))

    def validate_lease(self, *, operation_id: str, owner: str, fence: int, now: datetime) -> bool:
        context = self.context_ref.get().to_dict() or {}
        return _valid_lease(context, operation_id, owner, fence, _aware(now), context.get("lock_expires_at"))

    def get_stage(self, *, operation_id: str, stage_key: str) -> OperationStageRecord | None:
        snapshot = _stage_ref(self.context_ref, operation_id, stage_key).get()
        return _stage_from(snapshot.to_dict() or {}) if snapshot.exists else None

    def reserve_attempt(self, *, operation_id: str, stage_key: str, owner: str, fence: int,
                        attempt_id: str, now: datetime) -> OperationStageRecord | None:
        now = _aware(now)
        stage_ref = _stage_ref(self.context_ref, operation_id, stage_key)
        operation_ref = _operation_ref(self.context_ref, operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> OperationStageRecord | None:
            operation_snapshot = operation_ref.get(transaction=transaction)
            stage_snapshot = stage_ref.get(transaction=transaction)
            if (not operation_snapshot.exists or not self._valid(transaction, operation_id, owner, fence, now)):
                return None
            stage = _stage_from(stage_snapshot.to_dict() or {}) if stage_snapshot.exists else OperationStageRecord(stage_key, StageState.PENDING)
            if stage.state == StageState.COMPLETED or stage.in_flight_attempt_id is not None:
                return None
            reserved = OperationStageRecord(stage_key, StageState.RUNNING, stage.failures,
                stage.attempts_started + 1, attempt_id, stage.retry_not_before, stage.last_error,
                stage.checkpoint_refs, now, stage.result)
            transaction.set(stage_ref, _stage_data(reserved))
            return reserved
        return _txn(self.client.transaction())

    def complete_attempt(self, *, operation_id: str, stage_key: str, owner: str, fence: int,
                         attempt_id: str, result: Any, now: datetime,
                         checkpoint_refs: tuple[str, ...] = ()) -> bool:
        return self._finish_attempt(operation_id, stage_key, owner, fence, attempt_id, now,
                                    result=result, checkpoint_refs=checkpoint_refs)

    def fail_attempt(self, *, operation_id: str, stage_key: str, owner: str, fence: int,
                     attempt_id: str, error: OperationError, now: datetime,
                     retry_not_before: datetime | None = None) -> bool:
        return self._finish_attempt(operation_id, stage_key, owner, fence, attempt_id, now,
                                    error=error, retry_not_before=retry_not_before)

    def _finish_attempt(self, operation_id: str, stage_key: str, owner: str, fence: int,
                        attempt_id: str, now: datetime, *, result: Any = None,
                        error: OperationError | None = None, retry_not_before: datetime | None = None,
                        checkpoint_refs: tuple[str, ...] = ()) -> bool:
        now = _aware(now)
        stage_ref = _stage_ref(self.context_ref, operation_id, stage_key)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            stage_snapshot = stage_ref.get(transaction=transaction)
            if (not stage_snapshot.exists or not self._valid(transaction, operation_id, owner, fence, now)):
                return False
            stage = _stage_from(stage_snapshot.to_dict() or {})
            if stage.in_flight_attempt_id != attempt_id:
                return False
            if error is None:
                updated = OperationStageRecord(stage_key, StageState.COMPLETED, stage.failures,
                    stage.attempts_started, None, None, None,
                    tuple(checkpoint_refs) or stage.checkpoint_refs, now, result)
            else:
                failures = stage.failures + 1
                safe_error = OperationError(**{**error.__dict__, "failures": failures}) if hasattr(error, "__dict__") else OperationError(
                    code=error.code, category=error.category, message=error.message,
                    retryable=error.retryable, service=error.service, stage=error.stage,
                    sanitized_message=error.sanitized_message, occurred_at=now, failures=failures,
                    operation_id=operation_id)
                updated = OperationStageRecord(stage_key, StageState.FAILED, failures,
                    stage.attempts_started, None, retry_not_before, safe_error,
                    stage.checkpoint_refs, now, None)
            transaction.set(stage_ref, _stage_data(updated))
            return True
        return _txn(self.client.transaction())

    def fail_operation(self, *, operation_id: str, owner: str, fence: int,
                       error: OperationError, now: datetime) -> bool:
        """Queue settlement belongs to TelegramQueueWorker; stage state is terminal."""
        return self.validate_lease(operation_id=operation_id, owner=owner, fence=fence, now=now)


def _stage_data(stage: OperationStageRecord) -> dict[str, Any]:
    error = stage.last_error
    return {
        "stage_key": stage.stage_key, "state": stage.state.value, "failures": stage.failures,
        "attempts_started": stage.attempts_started,
        "in_flight_attempt_id": stage.in_flight_attempt_id,
        "retry_not_before": stage.retry_not_before,
        "last_error": ({name: getattr(error, name) for name in error.__dataclass_fields__}
                       if error is not None else None),
        "checkpoint_refs": list(stage.checkpoint_refs), "updated_at": stage.updated_at,
        "result": stage.result,
    }


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

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> QueuedOperation:
        return register_operation_in_transaction(
            client, transaction, context_ref, operation_id,
            identity_digest=identity_digest, input_ref=input_ref, now=now,
        )

    return _txn(client.transaction())


def register_operation_in_transaction(
    client: Any,
    transaction: Any,
    context_ref: Any,
    operation_id: str,
    *,
    identity_digest: str,
    input_ref: str,
    now: datetime,
    before_write: Any = None,
    operation_fields: Mapping[str, Any] | None = None,
) -> QueuedOperation:
    """Append an operation using a transaction owned by a larger workflow."""
    now = _aware(now)
    if not identity_digest or not input_ref:
        raise ValueError("identity_digest and input_ref are required")
    operation_ref = _operation_ref(context_ref, operation_id)
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

    if before_write is not None:
        before_write()

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
    if operation_fields:
        reserved = set(data).intersection(operation_fields)
        if reserved:
            raise ValueError(f"operation_fields cannot replace queue fields: {sorted(reserved)}")
        data.update(operation_fields)
    if tail_ref is not None and tail_data is not None:
        tail_data["next_operation_id"] = operation_id
        transaction.set(tail_ref, tail_data)
    transaction.set(operation_ref, data)
    context_update: dict[str, Any] = {
        "next_operation_sequence": sequence + 1,
        "queue_tail_operation_id": operation_id,
    }
    if not context.get("reserved_operation_id"):
        context_update.update({"head_sequence": sequence, "reserved_operation_id": operation_id})
    transaction.set(context_ref, {**context, **context_update})
    return _operation_from(data)


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
    after_release: Any = None,
) -> bool:
    """Yield the current lease while preserving the same queue head.

    ``after_release`` can add provider-neutral transaction writes after the
    lease has been validated and before the context is persisted. This lets a
    caller atomically schedule a continuation alongside the lease release.
    """
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
        if after_release is not None:
            after_release(transaction, context)
        transaction.set(context_ref, context)
        return True

    return _txn(client.transaction())


def complete_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
    after_advance: Any = None,
) -> bool:
    return _finish_head(client, context_ref, operation_id=operation_id, owner=owner, fence=fence, now=now,
                        state=QueueOperationState.COMPLETED, after_advance=after_advance)


def fail_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
    after_advance: Any = None,
) -> bool:
    return _finish_head(client, context_ref, operation_id=operation_id, owner=owner, fence=fence, now=now,
                        state=QueueOperationState.FAILED, after_advance=after_advance)


def _finish_head(
    client: Any,
    context_ref: Any, *, operation_id: str, owner: str, fence: int, now: datetime,
    state: QueueOperationState, after_advance: Any = None,
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
        if after_advance is not None:
            after_advance(transaction, next_id, next_data, context)
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
    "fail_head", "get_operation", "heartbeat", "register_operation",
    "register_operation_in_transaction", "release_head", "QueueOperationRetryStore",
]
