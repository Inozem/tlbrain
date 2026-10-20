"""Protocol-neutral contracts for shared dialog operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class OperationOrigin(StrEnum):
    MCP = "mcp"
    TELEGRAM = "telegram"


class OperationState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {OperationState.COMPLETED, OperationState.FAILED}


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EffectState(StrEnum):
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OperationError:
    """Safe, transport-neutral error details suitable for recording."""

    code: str
    category: str
    message: str
    retryable: bool
    service: str | None = None
    stage: str | None = None
    sanitized_message: str | None = None
    occurred_at: datetime | None = None
    failures: int | None = None
    operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Small domain result returned after an operation completes."""

    document_id: str | None = None
    title: str | None = None
    resource_url: str | None = None


@dataclass(frozen=True, slots=True)
class OperationStageRecord:
    stage_key: str
    state: StageState
    failures: int = 0
    attempts_started: int = 0
    in_flight_attempt_id: str | None = None
    retry_not_before: datetime | None = None
    last_error: OperationError | None = None
    checkpoint_refs: tuple[str, ...] = ()
    updated_at: datetime | None = None
    result: object | None = None


@dataclass(frozen=True, slots=True)
class OperationEffect:
    effect_id: str
    operation_id: str
    kind: str
    target_resource: str
    desired_payload_digest: str
    expected_pre_state: str | None
    expected_revision: str | None
    state: EffectState
    dispatched_at: datetime | None = None
    confirmed_result: OperationResult | None = None
    fence: int = 0


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """Typed fields for a common ``dialog_operations/{id}`` record."""

    operation_id: str
    origin: OperationOrigin
    origin_operation_ref: str | None
    caller_binding: str
    immutable_input_digest: str
    immutable_input_ref: str
    provider: str | None = None
    title: str | None = None
    date: str | None = None
    explicit_folder_id: str | None = None
    selected_folder_id: str | None = None
    selected_folder_path: tuple[str, ...] | None = None
    state: OperationState = OperationState.PENDING
    stage: str | None = None
    stage_failure_counts: tuple[tuple[str, int], ...] = ()
    save_intent_id: str | None = None
    doc_id: str | None = None
    create_outcome: str | None = None
    filled_revision: str | None = None
    desired_content_hash: str | None = None
    desired_parent_id_hash: str | None = None
    expected_provider: str | None = None
    index_completion_ref: str | None = None
    result: OperationResult | None = None
    error: OperationError | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    fence: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status_updated_at: datetime | None = None
    attempts_total: int = 0


class Clock(Protocol):
    """Source of timezone-aware timestamps for operation services."""

    def now(self) -> datetime:
        """Return the current timezone-aware instant."""
