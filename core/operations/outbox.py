"""Publish durable Firestore task-outbox records to Cloud Tasks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Mapping, Protocol
from urllib.parse import quote, urlsplit, urlunsplit

from core.integrations.cloud_tasks.client import enqueue_task
from core.integrations.firestore.outbox import (
    OutboxRecord,
    OutboxTargetKind,
    create_outbox_record,
    list_due_outbox_records,
    record_outbox_delivery_result,
    task_payload,
)
from core.integrations.firestore.operations import list_stale_operations
from core.operations.models import OperationOrigin, OperationRecord, OperationState


MCP_RECOVERY_STALE_AGE = timedelta(minutes=5)
MCP_RECOVERY_SWEEP_LIMIT = 50
TELEGRAM_RECOVERY_SWEEP_LIMIT = 50
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,480}$")


@dataclass(frozen=True, slots=True)
class TaskTargetConfig:
    """Deploy-confirmed Cloud Tasks destination and optional OIDC settings."""

    queue_name: str
    url: str
    service_account_email: str | None = None
    audience: str | None = None
    dispatch_deadline: timedelta | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.queue_name, str) or not self.queue_name.strip():
            raise ValueError("queue_name must be non-empty")
        if not isinstance(self.url, str):
            raise ValueError("url must be an HTTPS URL")
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("url must be an HTTPS URL without embedded credentials")
        for name in ("service_account_email", "audience"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when provided")
        if self.dispatch_deadline is not None and (
            not isinstance(self.dispatch_deadline, timedelta)
            or self.dispatch_deadline <= timedelta(0)
        ):
            raise ValueError("dispatch_deadline must be a positive timedelta")


@dataclass(frozen=True, slots=True)
class OutboxSweepResult:
    selected: int = 0
    published: int = 0
    already_exists: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass(frozen=True, slots=True)
class OutboxDispatchResult:
    """Outcome for publishing one durable outbox record."""

    outcome: str

    @property
    def retryable(self) -> bool:
        return self.outcome in {_PublicationOutcome.SKIPPED.value, _PublicationOutcome.FAILED.value}


class _PublicationOutcome(Enum):
    PUBLISHED = "published"
    ALREADY_EXISTS = "already_exists"
    SKIPPED = "skipped"
    FAILED = "failed"


class TelegramQueueRecoveryStore(Protocol):
    """Storage boundary for repairing reserved Telegram queue-head wakeups."""

    def recover_expired_heads(self, *, now: datetime, limit: int) -> int: ...


def _targets_by_kind(
    targets: Mapping[OutboxTargetKind | str, TaskTargetConfig | None],
) -> dict[OutboxTargetKind, TaskTargetConfig | None]:
    configured: dict[OutboxTargetKind, TaskTargetConfig | None] = {}
    for raw_kind, target in targets.items():
        try:
            kind = OutboxTargetKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("targets contains an unsupported outbox kind") from exc
        if target is not None and not isinstance(target, TaskTargetConfig):
            raise TypeError("each configured target must be a TaskTargetConfig or None")
        configured[kind] = target
    return configured


def _record_enqueue_failure(
    client: object, record: OutboxRecord, *, now: datetime
) -> None:
    """Store a safe retry marker without letting a failed write stop the sweep.

    Provider exceptions can include credentials, URLs, or payload data, so the
    outbox stores only a fixed error code. If this write also fails, the record
    remains eligible for a later sweep and the other selected records can proceed.
    """
    try:
        record_outbox_delivery_result(
            client,
            record.outbox_id,
            delivered=False,
            now=now,
            safe_error="cloud_tasks_enqueue_failed",
        )
    except Exception:
        pass


def _publish_record(
    client: object,
    record: OutboxRecord,
    target: TaskTargetConfig | None,
    *,
    now: datetime,
) -> _PublicationOutcome:
    """Publish one selected record and persist only its delivery outcome.

    A missing target leaves the record untouched. An enqueue error gets a safe
    retry marker; accepted tasks whose state write fails are retried by a later
    sweep with the same deterministic name, allowing Cloud Tasks to deduplicate.
    """
    if target is None:
        return _PublicationOutcome.SKIPPED

    target_url = target.url
    if record.target_kind is OutboxTargetKind.INDEX_DISPATCH:
        if not isinstance(record.document_id, str) or not record.document_id:
            _record_enqueue_failure(client, record, now=now)
            return _PublicationOutcome.FAILED
        parsed = urlsplit(target.url)
        target_url = urlunsplit((
            parsed.scheme, parsed.netloc,
            parsed.path.rstrip("/") + "/sync/doc/" + quote(record.document_id, safe=""),
            parsed.query, parsed.fragment,
        ))

    try:
        created = enqueue_task(
            queue_name=target.queue_name,
            url=target_url,
            task_id=record.task_name,
            body=task_payload(record),
            schedule_time=record.schedule_at,
            service_account_email=target.service_account_email,
            audience=target.audience,
            dispatch_deadline=target.dispatch_deadline,
        )
    except Exception:
        _record_enqueue_failure(client, record, now=now)
        return _PublicationOutcome.FAILED

    try:
        record_outbox_delivery_result(
            client,
            record.outbox_id,
            delivered=True,
            now=now,
        )
    except Exception:
        return _PublicationOutcome.FAILED
    return (
        _PublicationOutcome.PUBLISHED
        if created
        else _PublicationOutcome.ALREADY_EXISTS
    )


def publish_outbox_record(
    client: object,
    record: OutboxRecord,
    target: TaskTargetConfig | None,
    *,
    now: datetime,
) -> OutboxDispatchResult:
    """Publish one durable record without scanning other outbox work."""
    return OutboxDispatchResult(_publish_record(client, record, target, now=now).value)


def publish_due_outbox_records(
    client: object,
    targets: Mapping[OutboxTargetKind | str, TaskTargetConfig | None],
    *,
    now: datetime,
    limit: int,
) -> OutboxSweepResult:
    """Publish at most ``limit`` due records, preserving progress after failures.

    A false enqueue result means the stable named Cloud Task already exists. Both
    outcomes confirm dispatch only; this function never changes domain operations.
    """
    configured = _targets_by_kind(targets)
    enabled_kinds = [kind for kind, target in configured.items() if target is not None]
    records = list_due_outbox_records(
        client, now=now, limit=limit, target_kinds=enabled_kinds
    )
    totals = {outcome: 0 for outcome in _PublicationOutcome}
    for record in records:
        outcome = _publish_record(
            client, record, configured.get(record.target_kind), now=now
        )
        totals[outcome] += 1

    return OutboxSweepResult(
        selected=len(records),
        published=totals[_PublicationOutcome.PUBLISHED],
        already_exists=totals[_PublicationOutcome.ALREADY_EXISTS],
        skipped=totals[_PublicationOutcome.SKIPPED],
        failed=totals[_PublicationOutcome.FAILED],
    )


def enqueue_stale_mcp_save_recoveries(
    client: object,
    *,
    now: datetime,
    limit: int = MCP_RECOVERY_SWEEP_LIMIT,
) -> int:
    """Create stable recovery outbox records for eligible stale MCP saves."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    stale = list_stale_operations(
        client, cutoff=now - MCP_RECOVERY_STALE_AGE, limit=limit,
        origin=OperationOrigin.MCP,
    )
    enqueued = 0
    for operation in stale:
        lease_expiry = operation.lease_expires_at
        lease_expired = lease_expiry is None or (
            isinstance(lease_expiry, datetime)
            and lease_expiry.tzinfo is not None
            and lease_expiry.astimezone(now.tzinfo) <= now
        )
        if (
            operation.origin is not OperationOrigin.MCP
            or operation.state not in (OperationState.PENDING, OperationState.RUNNING)
            or not operation.save_intent_id
            or not operation.origin_operation_ref
            or (operation.state is OperationState.RUNNING and not lease_expired)
            or not _SAFE_TASK_ID.fullmatch(operation.operation_id)
            or not _SAFE_TASK_ID.fullmatch(operation.save_intent_id)
        ):
            continue

        try:
            create_mcp_save_recovery_record(client, operation, now=now)
        except (TypeError, ValueError):
            # Malformed legacy identifiers are not dispatchable task references.
            continue
        enqueued += 1
    return enqueued


def create_mcp_save_recovery_record(
    client: object, operation: OperationRecord, *, now: datetime,
    completed_index_dispatch: OutboxRecord | None = None,
) -> OutboxRecord:
    """Create a stable MCP recovery record from a fence or completed index dispatch."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    operation_id = operation.operation_id
    save_intent_id = operation.save_intent_id
    fence = operation.fence
    if (not isinstance(operation_id, str) or not _SAFE_TASK_ID.fullmatch(operation_id)
            or not isinstance(save_intent_id, str) or not _SAFE_TASK_ID.fullmatch(save_intent_id)
            or isinstance(fence, bool) or not isinstance(fence, int) or fence < 0):
        raise ValueError("operation is not a valid MCP recovery identity")
    now = now.astimezone(timezone.utc)
    source = completed_index_dispatch
    if source is not None and (
        source.target_kind is not OutboxTargetKind.INDEX_DISPATCH
        or source.operation_id != operation_id or source.save_intent_id != save_intent_id
        or not isinstance(source.generation, int) or source.generation < 1
    ):
        raise ValueError("completed index dispatch does not match the operation")
    identity_suffix = (
        f"index:{source.outbox_id}:{source.generation}" if source is not None else str(fence)
    )
    identity = f"{operation_id}\0{save_intent_id}\0{identity_suffix}"
    outbox_id = "mcp-save-recovery-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return create_outbox_record(
        client,
        outbox_id,
        target_kind=OutboxTargetKind.MCP_SAVE_RECOVERY,
        generation=source.generation if source is not None else max(1, fence + 1),
        schedule_at=(source.schedule_at if source is not None else
                     getattr(operation, "created_at", None) or getattr(operation, "updated_at", None) or now),
        now=now,
        operation_id=operation_id,
        save_intent_id=save_intent_id,
    )


def enqueue_expired_telegram_queue_recoveries(
    store: TelegramQueueRecoveryStore,
    *,
    now: datetime,
    limit: int = TELEGRAM_RECOVERY_SWEEP_LIMIT,
) -> int:
    """Ask the configured storage adapter to wake expired reserved queue heads."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer from 1 to 500")
    return store.recover_expired_heads(now=now.astimezone(timezone.utc), limit=limit)


__all__ = [
    "OutboxDispatchResult",
    "OutboxSweepResult",
    "TaskTargetConfig",
    "TelegramQueueRecoveryStore",
    "publish_outbox_record",
    "enqueue_expired_telegram_queue_recoveries",
    "enqueue_stale_mcp_save_recoveries",
    "create_mcp_save_recovery_record",
    "publish_due_outbox_records",
]
