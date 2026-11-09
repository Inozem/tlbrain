"""Durable, provider-neutral task outbox records stored in Firestore."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable

from google.cloud import firestore


OUTBOX_COLLECTION = "task_outbox"
PAYLOAD_SCHEMA_VERSION = 1
MAX_DUE_LIMIT = 500
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,480}$")


class OutboxTargetKind(StrEnum):
    TELEGRAM_DISPATCH = "telegram_dispatch"
    TELEGRAM_STATUS_CHECK = "telegram_status_check"
    MCP_SAVE_RECOVERY = "mcp_save_recovery"
    INDEX_DISPATCH = "index_dispatch"


class OutboxState(StrEnum):
    PENDING = "pending"
    RETRY = "retry"
    DELIVERED = "delivered"
    FAILED = "failed"


UNDELIVERED_STATES = (OutboxState.PENDING, OutboxState.RETRY)
_REFERENCE_FIELDS = ("operation_id", "context_id", "document_id", "save_intent_id", "notification_id")


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    outbox_id: str
    target_kind: OutboxTargetKind
    generation: int
    schedule_at: datetime
    task_name: str
    operation_id: str | None = None
    context_id: str | None = None
    document_id: str | None = None
    save_intent_id: str | None = None
    notification_id: str | None = None
    state: OutboxState = OutboxState.PENDING
    delivery_attempts: int = 0
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_attempt_at: datetime | None = None


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty identifier containing only letters, digits, '_' or '-'")
    return value


def _from(data: dict[str, Any]) -> OutboxRecord:
    return OutboxRecord(
        outbox_id=data["outbox_id"], target_kind=OutboxTargetKind(data["target_kind"]),
        generation=int(data["generation"]), schedule_at=data["schedule_at"], task_name=data["task_name"],
        operation_id=data.get("operation_id"), context_id=data.get("context_id"),
        document_id=data.get("document_id"), save_intent_id=data.get("save_intent_id"),
        notification_id=data.get("notification_id"),
        state=OutboxState(data["state"]), delivery_attempts=int(data.get("delivery_attempts", 0)),
        last_error=data.get("last_error"), created_at=data.get("created_at"),
        updated_at=data.get("updated_at"), last_attempt_at=data.get("last_attempt_at"),
    )


def _identity(data: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(data.get(key) for key in (
        "target_kind", "generation", "schedule_at", *_REFERENCE_FIELDS,
    ))


def _task_name(outbox_id: str, generation: int) -> str:
    """Build a deterministic task name without incorporating caller content."""
    return f"task-{outbox_id}-g{generation}"


def create_outbox_record(
    client: Any,
    outbox_id: str,
    *,
    target_kind: OutboxTargetKind | str,
    generation: int,
    schedule_at: datetime,
    now: datetime,
    operation_id: str | None = None,
    context_id: str | None = None,
    document_id: str | None = None,
    save_intent_id: str | None = None,
    notification_id: str | None = None,
) -> OutboxRecord:
    """Create a durable task request or return an exact identity replay."""
    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OutboxRecord:
        return create_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=target_kind, generation=generation, schedule_at=schedule_at, now=now,
            operation_id=operation_id, context_id=context_id, document_id=document_id,
            save_intent_id=save_intent_id, notification_id=notification_id,
        )

    return _txn(client.transaction(max_attempts=5))


def create_outbox_record_in_transaction(
    client: Any,
    transaction: Any,
    outbox_id: str,
    *,
    target_kind: OutboxTargetKind | str,
    generation: int,
    schedule_at: datetime,
    now: datetime,
    operation_id: str | None = None,
    context_id: str | None = None,
    document_id: str | None = None,
    save_intent_id: str | None = None,
    notification_id: str | None = None,
) -> OutboxRecord:
    """Create/replay an outbox record inside a caller-owned transaction."""
    prepared = prepare_outbox_record_in_transaction(
        client, transaction, outbox_id,
        target_kind=target_kind, generation=generation, schedule_at=schedule_at, now=now,
        operation_id=operation_id, context_id=context_id, document_id=document_id,
        save_intent_id=save_intent_id, notification_id=notification_id,
    )
    write_prepared_outbox_record(transaction, prepared)
    return prepared[2]


def prepare_outbox_record_in_transaction(
    client: Any,
    transaction: Any,
    outbox_id: str,
    *,
    target_kind: OutboxTargetKind | str,
    generation: int,
    schedule_at: datetime,
    now: datetime,
    operation_id: str | None = None,
    context_id: str | None = None,
    document_id: str | None = None,
    save_intent_id: str | None = None,
    notification_id: str | None = None,
) -> tuple[Any, dict[str, Any], OutboxRecord, bool]:
    """Read and validate a row before a transaction performs any writes."""
    outbox_id = _id(outbox_id, "outbox_id")
    try:
        target = OutboxTargetKind(target_kind)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_kind is not an allowed outbox target") from exc
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("generation must be a positive integer")
    schedule_at = _aware(schedule_at, "schedule_at")
    now = _aware(now, "now")
    refs = {
        "operation_id": operation_id, "context_id": context_id,
        "document_id": document_id, "save_intent_id": save_intent_id,
        "notification_id": notification_id,
    }
    for name, value in tuple(refs.items()):
        if value is not None:
            refs[name] = _id(value, name)
    if not any(refs.values()):
        raise ValueError("at least one operation, context, document, or save intent reference is required")

    task_name = _task_name(outbox_id, generation)
    ref = client.collection(OUTBOX_COLLECTION).document(outbox_id)
    identity_data = {
        "outbox_id": outbox_id, "target_kind": target.value, "generation": generation,
        "schedule_at": schedule_at, **refs,
    }

    snapshot = ref.get(transaction=transaction)
    if snapshot.exists:
        stored = snapshot.to_dict() or {}
        if _identity(stored) != _identity(identity_data):
            raise ValueError("outbox ID already belongs to a different target identity")
        return ref, stored, _from(stored), False
    data = {
        **identity_data, "task_name": task_name,
        "state": OutboxState.PENDING.value, "delivery_attempts": 0,
        "last_error": None, "created_at": now, "updated_at": now,
        "last_attempt_at": None,
    }
    return ref, data, _from(data), True


def write_prepared_outbox_record(transaction: Any, prepared: tuple[Any, dict[str, Any], OutboxRecord, bool]) -> None:
    """Write a previously prepared outbox row without issuing transaction reads."""
    ref, data, _, should_write = prepared
    if should_write:
        transaction.set(ref, data)


def get_outbox_record(client: Any, outbox_id: str) -> OutboxRecord | None:
    """Load a record by its stable outbox document ID."""
    outbox_id = _id(outbox_id, "outbox_id")
    snapshot = client.collection(OUTBOX_COLLECTION).document(outbox_id).get()
    return _from(snapshot.to_dict() or {}) if snapshot.exists else None


def _list_due_outbox_records_for_all_targets(
    collection: Any,
    *,
    now: datetime,
    limit: int,
) -> list[OutboxRecord]:
    """Query all targets, using RETRY records to fill space left by PENDING ones."""
    pending = collection.where("state", "==", OutboxState.PENDING.value)
    pending = pending.where("schedule_at", "<=", now).limit(limit)
    snapshots = list(pending.stream())
    if len(snapshots) < limit:
        retries = collection.where("state", "==", OutboxState.RETRY.value)
        retries = retries.where("schedule_at", "<=", now).limit(limit - len(snapshots))
        snapshots.extend(retries.stream())
    return [_from(snapshot.to_dict() or {}) for snapshot in snapshots]


def _list_due_outbox_records_for_targets(
    collection: Any,
    *,
    now: datetime,
    limit: int,
    target_kinds: tuple[str, ...],
) -> list[OutboxRecord]:
    """Query selected targets independently with a fair share of the total limit."""
    base_limit, extra = divmod(limit, len(target_kinds))
    records: list[OutboxRecord] = []
    for index, target_kind in enumerate(target_kinds):
        per_kind_limit = base_limit + int(index < extra)
        if per_kind_limit == 0:
            break
        pending = collection.where("target_kind", "==", target_kind)
        pending = pending.where("state", "==", OutboxState.PENDING.value)
        pending = pending.where("schedule_at", "<=", now).limit(per_kind_limit)
        selected = list(pending.stream())
        if len(selected) < per_kind_limit:
            retries = collection.where("target_kind", "==", target_kind)
            retries = retries.where("state", "==", OutboxState.RETRY.value)
            retries = retries.where("schedule_at", "<=", now).limit(per_kind_limit - len(selected))
            selected.extend(retries.stream())
        records.extend(_from(snapshot.to_dict() or {}) for snapshot in selected)
    return records


def list_due_outbox_records(
    client: Any,
    *,
    now: datetime,
    limit: int,
    target_kinds: Iterable[OutboxTargetKind | str] | None = None,
) -> list[OutboxRecord]:
    """Return at most ``limit`` due records in undelivered states and targets."""
    now = _aware(now, "now")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_DUE_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_DUE_LIMIT}")
    kinds = None
    if target_kinds is not None:
        try:
            kinds = tuple(dict.fromkeys(OutboxTargetKind(kind).value for kind in target_kinds))
        except (TypeError, ValueError) as exc:
            raise ValueError("target_kinds contains an unsupported outbox target") from exc
        if not kinds:
            return []
    collection = client.collection(OUTBOX_COLLECTION)
    if kinds is None:
        records = _list_due_outbox_records_for_all_targets(collection, now=now, limit=limit)
    else:
        records = _list_due_outbox_records_for_targets(
            collection, now=now, limit=limit, target_kinds=kinds
        )
    records.sort(key=lambda record: (record.schedule_at, record.created_at or record.schedule_at, record.outbox_id))
    return records[:limit]


def record_outbox_delivery_result(
    client: Any,
    outbox_id: str,
    *,
    delivered: bool,
    now: datetime,
    safe_error: str | None = None,
) -> OutboxRecord | None:
    """Record one enqueue attempt without allowing a late failure to undo delivery.

    ``safe_error`` is a short, caller-controlled diagnostic code. Provider exception
    text must never be passed here because it can contain request data or secrets.
    """
    outbox_id = _id(outbox_id, "outbox_id")
    now = _aware(now, "now")
    if not isinstance(delivered, bool):
        raise ValueError("delivered must be a boolean")
    if delivered and safe_error is not None:
        raise ValueError("successful delivery cannot include an error")
    if safe_error is not None and (
        not isinstance(safe_error, str)
        or not re.fullmatch(r"[a-z_]{1,80}", safe_error)
    ):
        raise ValueError("safe_error must be a lowercase diagnostic code")
    ref = client.collection(OUTBOX_COLLECTION).document(outbox_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OutboxRecord | None:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        current = OutboxState(data["state"])
        updates: dict[str, Any] = {
            "delivery_attempts": int(data.get("delivery_attempts", 0)) + 1,
            "last_attempt_at": now,
            "updated_at": now,
        }
        if current is not OutboxState.DELIVERED:
            updates["state"] = (
                OutboxState.DELIVERED.value if delivered else OutboxState.RETRY.value
            )
            updates["last_error"] = None if delivered else safe_error
        transaction.update(ref, updates)
        return _from({**data, **updates})

    return _txn(client.transaction(max_attempts=5))


def task_payload(record: OutboxRecord) -> dict[str, Any]:
    """Build the only permitted Cloud Tasks body: schema version and identifiers."""
    if not isinstance(record, OutboxRecord):
        raise TypeError("record must be an OutboxRecord")
    _id(record.outbox_id, "outbox_id")
    if not isinstance(record.target_kind, OutboxTargetKind):
        raise ValueError("record target_kind is not an allowed outbox target")
    if isinstance(record.generation, bool) or not isinstance(record.generation, int) or record.generation < 1:
        raise ValueError("record generation must be a positive integer")
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "outbox_id": record.outbox_id,
        "target_kind": record.target_kind.value,
        "generation": record.generation,
    }
    for name in _REFERENCE_FIELDS:
        value = getattr(record, name)
        if value is not None:
            payload[name] = _id(value, name)
    return payload


__all__ = [
    "MAX_DUE_LIMIT", "OUTBOX_COLLECTION", "OutboxRecord", "OutboxState",
    "OutboxTargetKind", "PAYLOAD_SCHEMA_VERSION", "UNDELIVERED_STATES",
    "create_outbox_record", "get_outbox_record", "list_due_outbox_records", "task_payload",
    "create_outbox_record_in_transaction", "prepare_outbox_record_in_transaction",
    "write_prepared_outbox_record",
    "record_outbox_delivery_result",
]
