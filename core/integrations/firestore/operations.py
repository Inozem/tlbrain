"""Durable Firestore repository for shared dialog operations and stages."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.operations.models import (
    OperationError,
    OperationOrigin,
    OperationRecord,
    OperationResult,
    OperationStageRecord,
    OperationState,
    StageState,
)


OPERATIONS_COLLECTION = "dialog_operations"
MAX_STALE_OPERATION_LIMIT = 500


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _operation_data(record: OperationRecord) -> dict[str, Any]:
    return {
        "operation_id": record.operation_id,
        "origin": record.origin.value,
        "origin_operation_ref": record.origin_operation_ref,
        "caller_binding": record.caller_binding,
        "immutable_input_digest": record.immutable_input_digest,
        "immutable_input_ref": record.immutable_input_ref,
        "provider": record.provider,
        "title": record.title,
        "date": record.date,
        "explicit_folder_id": record.explicit_folder_id,
        "selected_folder_id": record.selected_folder_id,
        "selected_folder_path": list(record.selected_folder_path) if record.selected_folder_path is not None else None,
        "state": record.state.value,
        "stage": record.stage,
        "stage_failure_counts": [[key, count] for key, count in record.stage_failure_counts],
        "save_intent_id": record.save_intent_id,
        "doc_id": record.doc_id,
        "create_outcome": record.create_outcome,
        "filled_revision": record.filled_revision,
        "desired_content_hash": record.desired_content_hash,
        "desired_parent_id_hash": record.desired_parent_id_hash,
        "expected_provider": record.expected_provider,
        "index_completion_ref": record.index_completion_ref,
        "result": _result_data(record.result),
        "error": _error_data(record.error),
        "lease_owner": record.lease_owner,
        "lease_expires_at": record.lease_expires_at,
        "fence": record.fence,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "status_updated_at": record.status_updated_at,
        "updated_at": record.updated_at,
        "attempts_total": record.attempts_total,
    }


def _result_data(result: OperationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    data = {"document_id": result.document_id, "title": result.title, "resource_url": result.resource_url}
    if result.skipped_utterances is not None:
        data["skipped_utterances"] = list(result.skipped_utterances)
    return data


def _result_from(data: dict[str, Any] | None) -> OperationResult | None:
    if data is None:
        return None
    return OperationResult(
        document_id=data.get("document_id"), title=data.get("title"), resource_url=data.get("resource_url"),
        skipped_utterances=(tuple(data["skipped_utterances"])
                            if data.get("skipped_utterances") is not None else None),
    )


def _error_data(error: OperationError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "service": error.service, "stage": error.stage, "code": error.code, "category": error.category,
        "sanitized_message": error.sanitized_message if error.sanitized_message is not None else error.message,
        "occurred_at": error.occurred_at, "retryable": error.retryable, "failures": error.failures,
        "operation_id": error.operation_id,
    }


def _error_from(data: dict[str, Any] | None) -> OperationError | None:
    if data is None:
        return None
    return OperationError(
        code=data["code"], category=data["category"],
        message=data.get("sanitized_message", data.get("message", "")), retryable=bool(data["retryable"]),
        service=data.get("service"), stage=data.get("stage"),
        sanitized_message=data.get("sanitized_message"), occurred_at=data.get("occurred_at"),
        failures=data.get("failures"), operation_id=data.get("operation_id"),
    )


def _enrich_error(
    error: OperationError,
    *,
    operation_id: str,
    now: datetime,
    stage: str | None,
    failures: int,
) -> OperationError:
    occurred_at = _aware(error.occurred_at) if error.occurred_at is not None else now
    return OperationError(
        code=error.code,
        category=error.category,
        message=error.message,
        retryable=error.retryable,
        service=error.service,
        stage=error.stage if error.stage is not None else stage,
        sanitized_message=(error.sanitized_message if error.sanitized_message is not None else error.message),
        occurred_at=occurred_at,
        failures=error.failures if error.failures is not None else failures,
        operation_id=error.operation_id if error.operation_id is not None else operation_id,
    )


def _record_from(data: dict[str, Any]) -> OperationRecord:
    return OperationRecord(
        operation_id=data["operation_id"], origin=OperationOrigin(data["origin"]),
        origin_operation_ref=data.get("origin_operation_ref"), caller_binding=data["caller_binding"],
        immutable_input_digest=data["immutable_input_digest"], immutable_input_ref=data["immutable_input_ref"],
        provider=data.get("provider"), title=data.get("title"), date=data.get("date"),
        explicit_folder_id=data.get("explicit_folder_id"), selected_folder_id=data.get("selected_folder_id"),
        selected_folder_path=tuple(data["selected_folder_path"]) if data.get("selected_folder_path") is not None else None,
        state=OperationState(data.get("state", OperationState.PENDING.value)), stage=data.get("stage"),
        stage_failure_counts=tuple((str(key), int(count)) for key, count in data.get("stage_failure_counts", ())),
        save_intent_id=data.get("save_intent_id"), doc_id=data.get("doc_id"), create_outcome=data.get("create_outcome"),
        filled_revision=data.get("filled_revision"), desired_content_hash=data.get("desired_content_hash"),
        desired_parent_id_hash=data.get("desired_parent_id_hash"), expected_provider=data.get("expected_provider"),
        index_completion_ref=data.get("index_completion_ref"), result=_result_from(data.get("result")),
        error=_error_from(data.get("error")), lease_owner=data.get("lease_owner"),
        lease_expires_at=data.get("lease_expires_at"), fence=int(data.get("fence", 0)),
        created_at=data.get("created_at"), updated_at=data.get("updated_at"), started_at=data.get("started_at"),
        finished_at=data.get("finished_at"), status_updated_at=data.get("status_updated_at"),
        attempts_total=int(data.get("attempts_total", 0)),
    )


def _stage_id(operation_id: str, stage_key: str) -> str:
    return hashlib.sha256(f"{operation_id}\0{stage_key}".encode("utf-8")).hexdigest()


def _stage_ref(db: Any, operation_id: str, stage_key: str) -> Any:
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    return operation_ref.collection("stages").document(_stage_id(operation_id, stage_key))


def _stage_data(record: OperationStageRecord, operation_id: str) -> dict[str, Any]:
    return {
        "operation_id": operation_id, "stage_key": record.stage_key, "state": record.state.value,
        "failures": record.failures, "attempts_started": record.attempts_started,
        "in_flight_attempt_id": record.in_flight_attempt_id, "retry_not_before": record.retry_not_before,
        "last_error": _error_data(record.last_error), "checkpoint_refs": list(record.checkpoint_refs),
        "result": record.result, "updated_at": record.updated_at,
    }


def _stage_from(data: dict[str, Any]) -> OperationStageRecord:
    return OperationStageRecord(
        stage_key=data["stage_key"], state=StageState(data.get("state", StageState.PENDING.value)),
        failures=int(data.get("failures", 0)), attempts_started=int(data.get("attempts_started", 0)),
        in_flight_attempt_id=data.get("in_flight_attempt_id"), retry_not_before=data.get("retry_not_before"),
        last_error=_error_from(data.get("last_error")), checkpoint_refs=tuple(data.get("checkpoint_refs", ())),
        updated_at=data.get("updated_at"), result=data.get("result"),
    )


def _valid_lease(data: dict[str, Any], owner: str, fence: int, now: datetime) -> bool:
    expires = data.get("lease_expires_at")
    return (
        data.get("state") == OperationState.RUNNING.value
        and data.get("lease_owner") == owner
        and data.get("fence") == fence
        and isinstance(expires, datetime)
        and expires > now
    )


def create_operation(
    client: Any | None = None,
    *,
    operation_id: str,
    origin: OperationOrigin,
    caller_binding: str,
    immutable_input_digest: str,
    immutable_input_ref: str,
    now: datetime,
    origin_operation_ref: str | None = None,
    provider: str | None = None,
    title: str | None = None,
    date: str | None = None,
    explicit_folder_id: str | None = None,
    save_intent_id: str | None = None,
) -> OperationRecord:
    """Create an operation, allowing only an exact immutable-identity replay."""
    now = _aware(now)
    if not operation_id or "/" in operation_id:
        raise ValueError("operation_id must be a non-empty Firestore document ID")
    if not immutable_input_digest or not immutable_input_ref:
        raise ValueError("immutable input digest and reference are required")
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    proposed = OperationRecord(
        operation_id=operation_id, origin=OperationOrigin(origin), origin_operation_ref=origin_operation_ref,
        caller_binding=caller_binding, immutable_input_digest=immutable_input_digest,
        immutable_input_ref=immutable_input_ref, provider=provider, title=title, date=date,
        explicit_folder_id=explicit_folder_id, save_intent_id=save_intent_id,
        created_at=now, updated_at=now, status_updated_at=now,
    )

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationRecord:
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            current = _record_from(snapshot.to_dict() or {})
            identity = (
                current.origin, current.origin_operation_ref, current.caller_binding,
                current.immutable_input_digest, current.immutable_input_ref, current.provider,
                current.title, current.date, current.explicit_folder_id, current.save_intent_id,
            )
            requested = (
                proposed.origin, proposed.origin_operation_ref, proposed.caller_binding,
                proposed.immutable_input_digest, proposed.immutable_input_ref, proposed.provider,
                proposed.title, proposed.date, proposed.explicit_folder_id, proposed.save_intent_id,
            )
            if identity != requested:
                raise ValueError("operation ID already belongs to a different immutable identity")
            return current
        transaction.set(ref, _operation_data(proposed))
        return proposed

    return _txn(db.transaction())


def get_operation(client: Any | None, operation_id: str) -> OperationRecord | None:
    snapshot = _db(client).collection(OPERATIONS_COLLECTION).document(operation_id).get()
    return _record_from(snapshot.to_dict() or {}) if snapshot.exists else None


def _list_stale_operations_for_state(
    collection: Any,
    *,
    origin: OperationOrigin,
    state: OperationState,
    cutoff: datetime,
    limit: int,
) -> list[OperationRecord]:
    """Fetch one bounded state slice so pending and running backlogs get separate quotas."""
    query = collection.where("origin", "==", origin.value)
    query = query.where("state", "==", state.value)
    query = query.where("updated_at", "<=", cutoff).limit(limit)
    return [_record_from(snapshot.to_dict() or {}) for snapshot in query.stream()]


def list_stale_operations(
    client: Any | None,
    *,
    cutoff: datetime,
    limit: int,
    origin: OperationOrigin = OperationOrigin.MCP,
) -> list[OperationRecord]:
    """Return a bounded set of pending/running operations older than ``cutoff``."""
    cutoff = _aware(cutoff)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_STALE_OPERATION_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_STALE_OPERATION_LIMIT}")
    origin = OperationOrigin(origin)
    collection = _db(client).collection(OPERATIONS_COLLECTION)
    records: list[OperationRecord] = []
    pending_limit = (limit + 1) // 2
    state_limits = (
        (OperationState.PENDING, pending_limit),
        (OperationState.RUNNING, limit - pending_limit),
    )
    for state, state_limit in state_limits:
        if state_limit == 0:
            continue
        records.extend(_list_stale_operations_for_state(
            collection, origin=origin, state=state, cutoff=cutoff, limit=state_limit,
        ))
    records.sort(key=lambda record: (record.updated_at or cutoff, record.operation_id))
    return records[:limit]


def validate_operation_lease(
    client: Any | None, operation_id: str, *, owner: str, fence: int, now: datetime,
) -> bool:
    """Check the current lease immediately before a guarded provider callback."""
    now = _aware(now)
    data = _db(client).collection(OPERATIONS_COLLECTION).document(operation_id).get().to_dict() or {}
    return _valid_lease(data, owner, fence, now)


def claim_operation(
    client: Any | None,
    operation_id: str,
    *,
    owner: str,
    now: datetime,
    lease_seconds: int | float,
) -> OperationRecord | None:
    """Acquire an available operation or reclaim an expired lease with a new fence."""
    now = _aware(now)
    if not owner or lease_seconds <= 0:
        raise ValueError("owner and a positive lease_seconds are required")
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationRecord | None:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if data.get("state") in {OperationState.COMPLETED.value, OperationState.FAILED.value}:
            return None
        expiry = data.get("lease_expires_at")
        if data.get("state") == OperationState.RUNNING.value and isinstance(expiry, datetime) and expiry > now:
            return None
        updated = dict(data)
        updated.update({
            "state": OperationState.RUNNING.value, "lease_owner": owner,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
            "fence": int(data.get("fence", 0)) + 1,
            "started_at": data.get("started_at") or now,
            "status_updated_at": now, "updated_at": now,
        })
        transaction.set(ref, updated)
        return _record_from(updated)

    return _txn(db.transaction())


def renew_operation_lease(
    client: Any | None,
    operation_id: str,
    *,
    owner: str,
    fence: int,
    now: datetime,
    lease_seconds: int | float,
) -> OperationRecord | None:
    """Extend a matching unexpired lease without changing attempt counters."""
    now = _aware(now)
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationRecord | None:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if not _valid_lease(data, owner, fence, now):
            return None
        updated = dict(data)
        updated.update({"lease_expires_at": now + timedelta(seconds=lease_seconds), "updated_at": now})
        transaction.set(ref, updated)
        return _record_from(updated)

    return _txn(db.transaction())


def release_operation(client: Any | None, operation_id: str, *, owner: str, fence: int, now: datetime) -> bool:
    now = _aware(now)
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        data = snapshot.to_dict() or {}
        if not _valid_lease(data, owner, fence, now):
            return False
        transaction.update(ref, {
            "lease_owner": None, "lease_expires_at": None, "updated_at": now,
        })
        return True

    return _txn(db.transaction())


def complete_operation(
    client: Any | None,
    operation_id: str,
    *,
    owner: str,
    fence: int,
    result: OperationResult,
    now: datetime,
) -> bool:
    now = _aware(now)
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        data = snapshot.to_dict() or {}
        if not _valid_lease(data, owner, fence, now):
            return False
        result_data = _result_data(result) or {}
        transaction.update(ref, {
            "state": OperationState.COMPLETED.value, "result": result_data,
            "doc_id": result.document_id, "error": None, "lease_owner": None, "lease_expires_at": None,
            "finished_at": now, "status_updated_at": now, "updated_at": now,
        })
        return True

    return _txn(db.transaction())


def fail_operation(
    client: Any | None,
    operation_id: str,
    *,
    owner: str,
    fence: int,
    error: OperationError,
    now: datetime,
) -> bool:
    """Atomically record a safe terminal error for the current lease holder."""
    now = _aware(now)
    db = _db(client)
    ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        data = snapshot.to_dict() or {}
        if not _valid_lease(data, owner, fence, now):
            return False
        counts = {key: int(count) for key, count in data.get("stage_failure_counts", ())}
        stored_error = _enrich_error(
            error, operation_id=operation_id, now=now, stage=data.get("stage"),
            failures=counts.get(data.get("stage"), 0),
        )
        transaction.update(ref, {
            "state": OperationState.FAILED.value, "error": _error_data(stored_error),
            "lease_owner": None, "lease_expires_at": None, "finished_at": now,
            "status_updated_at": now, "updated_at": now,
        })
        return True

    return _txn(db.transaction())


def get_stage(client: Any | None, operation_id: str, stage_key: str) -> OperationStageRecord | None:
    snapshot = _stage_ref(_db(client), operation_id, stage_key).get()
    return _stage_from(snapshot.to_dict() or {}) if snapshot.exists else None


def reserve_stage_attempt(
    client: Any | None,
    operation_id: str,
    stage_key: str,
    *,
    owner: str,
    fence: int,
    attempt_id: str,
    now: datetime,
) -> OperationStageRecord | None:
    """Durably reserve one physical attempt before its provider callback begins."""
    now = _aware(now)
    if not stage_key or not attempt_id:
        raise ValueError("stage_key and attempt_id are required")
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    stage_ref = _stage_ref(db, operation_id, stage_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationStageRecord | None:
        operation_snapshot = operation_ref.get(transaction=transaction)
        stage_snapshot = stage_ref.get(transaction=transaction)
        if not operation_snapshot.exists:
            return None
        operation_data = operation_snapshot.to_dict() or {}
        if not _valid_lease(operation_data, owner, fence, now):
            return None
        stage = _stage_from(stage_snapshot.to_dict() or {"stage_key": stage_key}) if stage_snapshot.exists else OperationStageRecord(stage_key, StageState.PENDING)
        if stage.state == StageState.COMPLETED or stage.in_flight_attempt_id is not None:
            return None
        if stage.retry_not_before is not None and stage.retry_not_before > now:
            return None
        reserved = OperationStageRecord(
            stage_key=stage_key, state=StageState.RUNNING, failures=stage.failures,
            attempts_started=stage.attempts_started + 1, in_flight_attempt_id=attempt_id,
            retry_not_before=stage.retry_not_before, last_error=stage.last_error,
            checkpoint_refs=stage.checkpoint_refs, updated_at=now, result=stage.result,
        )
        next_op = dict(operation_data)
        next_op.update({"attempts_total": int(operation_data.get("attempts_total", 0)) + 1,
                        "stage": stage_key, "updated_at": now})
        transaction.set(stage_ref, _stage_data(reserved, operation_id))
        transaction.set(operation_ref, next_op)
        return reserved

    return _txn(db.transaction())


def complete_stage_attempt(
    client: Any | None,
    operation_id: str,
    stage_key: str,
    *,
    owner: str,
    fence: int,
    attempt_id: str,
    result: Any,
    now: datetime,
    checkpoint_refs: tuple[str, ...] = (),
) -> bool:
    now = _aware(now)
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    stage_ref = _stage_ref(db, operation_id, stage_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        operation_snapshot = operation_ref.get(transaction=transaction)
        stage_snapshot = stage_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not stage_snapshot.exists:
            return False
        operation_data = operation_snapshot.to_dict() or {}
        stage = _stage_from(stage_snapshot.to_dict() or {})
        if not _valid_lease(operation_data, owner, fence, now) or stage.in_flight_attempt_id != attempt_id:
            return False
        completed = OperationStageRecord(
            stage_key=stage_key, state=StageState.COMPLETED, failures=stage.failures,
            attempts_started=stage.attempts_started, checkpoint_refs=tuple(checkpoint_refs) or stage.checkpoint_refs,
            updated_at=now, result=result,
        )
        operation_data["updated_at"] = now
        transaction.set(stage_ref, _stage_data(completed, operation_id))
        transaction.set(operation_ref, operation_data)
        return True

    return _txn(db.transaction())


def release_stage_attempt(
    client: Any | None, operation_id: str, stage_key: str, *, owner: str, fence: int,
    attempt_id: str, now: datetime,
) -> bool:
    """Release a successful volatile observation without completing its stage."""
    now = _aware(now)
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    stage_ref = _stage_ref(db, operation_id, stage_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        operation_snapshot = operation_ref.get(transaction=transaction)
        stage_snapshot = stage_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not stage_snapshot.exists:
            return False
        operation_data = operation_snapshot.to_dict() or {}
        stage = _stage_from(stage_snapshot.to_dict() or {})
        if not _valid_lease(operation_data, owner, fence, now) or stage.in_flight_attempt_id != attempt_id:
            return False
        released = OperationStageRecord(
            stage_key=stage_key, state=StageState.PENDING, failures=stage.failures,
            attempts_started=stage.attempts_started, checkpoint_refs=stage.checkpoint_refs,
            updated_at=now, last_error=stage.last_error,
        )
        operation_data["updated_at"] = now
        transaction.set(stage_ref, _stage_data(released, operation_id))
        transaction.set(operation_ref, operation_data)
        return True
    return _txn(db.transaction())


def fail_stage_attempt(
    client: Any | None,
    operation_id: str,
    stage_key: str,
    *,
    owner: str,
    fence: int,
    attempt_id: str,
    error: OperationError,
    now: datetime,
    retry_not_before: datetime | None = None,
) -> bool:
    now = _aware(now)
    if retry_not_before is not None:
        retry_not_before = _aware(retry_not_before)
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    stage_ref = _stage_ref(db, operation_id, stage_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        operation_snapshot = operation_ref.get(transaction=transaction)
        stage_snapshot = stage_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not stage_snapshot.exists:
            return False
        operation_data = operation_snapshot.to_dict() or {}
        stage = _stage_from(stage_snapshot.to_dict() or {})
        if not _valid_lease(operation_data, owner, fence, now) or stage.in_flight_attempt_id != attempt_id:
            return False
        failures = stage.failures + 1
        stored_error = _enrich_error(
            error, operation_id=operation_id, now=now, stage=stage_key, failures=failures,
        )
        failed = OperationStageRecord(
            stage_key=stage_key, state=StageState.FAILED, failures=failures,
            attempts_started=stage.attempts_started, retry_not_before=retry_not_before,
            last_error=stored_error, checkpoint_refs=stage.checkpoint_refs, updated_at=now,
        )
        counts = {key: int(count) for key, count in operation_data.get("stage_failure_counts", ())}
        counts[stage_key] = failures
        operation_data.update({"stage_failure_counts": [[key, count] for key, count in sorted(counts.items())],
                               "updated_at": now})
        transaction.set(stage_ref, _stage_data(failed, operation_id))
        transaction.set(operation_ref, operation_data)
        return True

    return _txn(db.transaction())
