"""Provider-neutral contracts and timing for a per-context operation queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


LEASE_SECONDS = 120
HEARTBEAT_INTERVAL_SECONDS = 30


class QueueOperationState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED}


@dataclass(frozen=True, slots=True)
class QueuedOperation:
    """Minimal queue metadata; input content remains in its separate store."""

    operation_id: str
    sequence: int
    state: QueueOperationState
    identity_digest: str
    input_ref: str
    created_at: datetime
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OperationLease:
    """Fenced execution right for the current queue head."""

    operation_id: str
    sequence: int
    owner: str
    fence: int
    expires_at: datetime
