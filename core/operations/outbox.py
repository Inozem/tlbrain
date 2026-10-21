"""Publish durable Firestore task-outbox records to Cloud Tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping
from urllib.parse import urlsplit

from core.integrations.cloud_tasks.client import enqueue_task
from core.integrations.firestore.outbox import (
    OutboxRecord,
    OutboxTargetKind,
    list_due_outbox_records,
    record_outbox_delivery_result,
    task_payload,
)


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


class _PublicationOutcome(Enum):
    PUBLISHED = "published"
    ALREADY_EXISTS = "already_exists"
    SKIPPED = "skipped"
    FAILED = "failed"


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

    try:
        created = enqueue_task(
            queue_name=target.queue_name,
            url=target.url,
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
    records = list_due_outbox_records(client, now=now, limit=limit)
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


__all__ = [
    "OutboxSweepResult",
    "TaskTargetConfig",
    "publish_due_outbox_records",
]
