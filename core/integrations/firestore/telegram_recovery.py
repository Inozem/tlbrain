"""Firestore recovery for reserved Telegram queue-head wakeups."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxState,
    OutboxTargetKind,
    create_outbox_record_in_transaction,
)


TELEGRAM_SESSIONS = "telegram_sessions"
OPERATIONS_SUBCOLLECTION = "operations"


def _wake_outbox_id(context_id: str, operation_id: str, generation: int) -> str:
    digest = hashlib.sha256(f"{context_id}:{operation_id}:{generation}".encode()).hexdigest()
    return f"tgqueue-{digest}"


def _ensure_wakeup(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    operation_id: str,
    generation: int,
    now: datetime,
) -> int:
    while True:
        outbox_id = _wake_outbox_id(context_id, operation_id, generation)
        ref = client.collection(OUTBOX_COLLECTION).document(outbox_id)
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            matches = (
                data.get("target_kind") == OutboxTargetKind.TELEGRAM_DISPATCH.value
                and data.get("context_id") == context_id
                and data.get("operation_id") == operation_id
                and int(data.get("generation", 0)) == generation
            )
            if matches and data.get("state") in {
                OutboxState.PENDING.value, OutboxState.RETRY.value,
            }:
                break
            generation += 1
            continue
        create_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=generation,
            schedule_at=now,
            now=now,
            context_id=context_id,
            operation_id=operation_id,
        )
        break
    return generation


class FirestoreTelegramQueueRecoveryStore:
    """Bounded, transactional implementation of the core recovery protocol."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def recover_expired_heads(self, *, now: datetime, limit: int) -> int:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(timezone.utc)
        collection = self.client.collection(TELEGRAM_SESSIONS)
        # Inequality semantics exclude null/missing values, unlike `!= ""`,
        # which can admit null IDs and let empty contexts consume the bound.
        reserved = collection.where("reserved_operation_id", ">", "")
        expired = list(reserved.where("lock_expires_at", "<=", now).limit(limit).stream())
        remaining = limit - len(expired)
        lease_less = (
            list(reserved.where("lock_expires_at", "==", None).limit(remaining).stream())
            if remaining else []
        )
        snapshots = expired + [
            snapshot for snapshot in lease_less
            if all(snapshot.id != prior.id for prior in expired)
        ]
        recovered = 0
        for snapshot in snapshots:
            context_ref = snapshot.reference

            @firestore.transactional
            def _txn(transaction: Any) -> bool:
                current = context_ref.get(transaction=transaction)
                if not current.exists:
                    return False
                context = current.to_dict() or {}
                operation_id = context.get("reserved_operation_id")
                expiry = context.get("lock_expires_at")
                if not operation_id or (
                    expiry is not None
                    and (not isinstance(expiry, datetime) or expiry > now)
                ):
                    return False
                active_id = context.get("active_operation_id")
                if active_id not in (None, operation_id):
                    return False
                operation_ref = context_ref.collection(OPERATIONS_SUBCOLLECTION).document(operation_id)
                operation_snapshot = operation_ref.get(transaction=transaction)
                if not operation_snapshot.exists:
                    return False
                operation = operation_snapshot.to_dict() or {}
                if operation.get("state") in {"completed", "failed"}:
                    return False
                base = max(1, int(context.get("fence", 0)) + 1)
                generation = max(
                    base,
                    int(context.get("telegram_dispatch_wake_generation", base)),
                )
                generation = _ensure_wakeup(
                    self.client, transaction,
                    context_id=snapshot.id,
                    operation_id=operation_id,
                    generation=generation,
                    now=now,
                )
                context["telegram_dispatch_wake_generation"] = generation
                transaction.set(context_ref, context)
                return True

            recovered += int(_txn(self.client.transaction(max_attempts=5)))
        return recovered


__all__ = ["FirestoreTelegramQueueRecoveryStore"]
