"""Durable, provider-neutral task outbox records stored in Firestore."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

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
_REFERENCE_FIELDS = ("operation_id", "context_id", "document_id", "save_intent_id")


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
        state=OutboxState(data["state"]), delivery_attempts=int(data.get("delivery_attempts", 0)),
        last_error=data.get("last_error"), created_at=data.get("created_at"),
        updated_at=data.get("updated_at"), last_attempt_at=data.get("last_attempt_at"),
    )


def _identity(data: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(data.get(key) for key in (
        "target_kind", "generation", "schedule_at", *_REFERENCE_FIELDS,
    ))


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
) -> OutboxRecord:
    """Create a durable task request or return an exact identity replay."""
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
    }
    for name, value in tuple(refs.items()):
        if value is not None:
            refs[name] = _id(value, name)
    if not any(refs.values()):
        raise ValueError("at least one operation, context, document, or save intent reference is required")

    # Cloud Tasks names are deterministic across retries and do not contain caller content.
    task_name = f"task-{outbox_id}-g{generation}"
    ref = client.collection(OUTBOX_COLLECTION).document(outbox_id)
    identity_data = {
        "outbox_id": outbox_id, "target_kind": target.value, "generation": generation,
        "schedule_at": schedule_at, **refs,
    }

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OutboxRecord:
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            stored = snapshot.to_dict() or {}
            if _identity(stored) != _identity(identity_data):
                raise ValueError("outbox ID already belongs to a different target identity")
            return _from(stored)
        data = {
            **identity_data, "task_name": task_name,
            "state": OutboxState.PENDING.value, "delivery_attempts": 0,
            "last_error": None, "created_at": now, "updated_at": now,
            "last_attempt_at": None,
        }
        transaction.set(ref, data)
        return _from(data)

    return _txn(client.transaction(max_attempts=5))


def get_outbox_record(client: Any, outbox_id: str) -> OutboxRecord | None:
    """Load a record by its stable outbox document ID."""
    outbox_id = _id(outbox_id, "outbox_id")
    snapshot = client.collection(OUTBOX_COLLECTION).document(outbox_id).get()
    return _from(snapshot.to_dict() or {}) if snapshot.exists else None


def list_due_outbox_records(
    client: Any,
    *,
    now: datetime,
    limit: int,
) -> list[OutboxRecord]:
    """Return at most ``limit`` due records in undelivered states."""
    now = _aware(now, "now")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_DUE_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_DUE_LIMIT}")
    collection = client.collection(OUTBOX_COLLECTION)
    query = collection.where("state", "==", OutboxState.PENDING.value)
    query = query.where("schedule_at", "<=", now).limit(limit)
    due = list(query.stream())
    if len(due) < limit:
        query = collection.where("state", "==", OutboxState.RETRY.value)
        query = query.where("schedule_at", "<=", now).limit(limit - len(due))
        due.extend(query.stream())
    return [_from(snapshot.to_dict() or {}) for snapshot in due]


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
]
