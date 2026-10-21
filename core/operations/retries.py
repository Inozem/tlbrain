"""Durable, one-callback-per-attempt stage execution governor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

from core.integrations.firestore.operations import (
    claim_operation,
    complete_stage_attempt,
    fail_operation,
    fail_stage_attempt,
    get_stage as repository_get_stage,
    release_operation,
    renew_operation_lease,
    reserve_stage_attempt,
)
from core.operations.models import OperationError, OperationStageRecord, StageState


DEFAULT_LEASE_SECONDS = 300


class RetryBudgetExceeded(RuntimeError):
    """Raised when a stage has exhausted its durable failure allowance."""


class UnknownOutcome(RuntimeError):
    """Raised when a provider attempt may have committed but is not confirmed."""


@dataclass(frozen=True, slots=True)
class ConfirmedStageSuccess:
    """Explicitly confirm a reconciled stage result, including ``None``."""

    result: Any


@dataclass(frozen=True, slots=True)
class ConfirmedStageFailure:
    """Explicitly confirm failure of a reconciled attempt with safe details."""

    error: OperationError


class ReconciledStageFailure(RuntimeError):
    """Raised after a reconciled failure has been durably recorded."""

    def __init__(self, error: OperationError) -> None:
        super().__init__(error.message)
        self.error = error


class RetryGovernor:
    """Persist stage attempts before invoking callbacks and retain their outcomes."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        clock: Any | None = None,
        max_failures: int = 3,
        lease_seconds: int | float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        if max_failures < 1 or max_failures > 3:
            raise ValueError("max_failures must be between 1 and 3")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.client = client
        self.clock = clock
        self.max_failures = max_failures
        self.lease_seconds = lease_seconds

    def _now(self, supplied: datetime | None = None) -> datetime:
        value = supplied
        if value is None:
            value = self.clock.now() if self.clock is not None else datetime.now(timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return value.astimezone(timezone.utc)

    def get_stage(self, operation_id: str, stage_key: str) -> OperationStageRecord:
        return repository_get_stage(self.client, operation_id, stage_key) or OperationStageRecord(
            stage_key=stage_key, state=StageState.PENDING,
        )

    def _fail_operation_if_terminal(
        self,
        operation_id: str,
        stage_key: str,
        *,
        owner: str,
        fence: int,
        error: OperationError,
        now: datetime,
    ) -> None:
        stage = self.get_stage(operation_id, stage_key)
        if error.retryable and stage.failures < self.max_failures:
            return
        if not fail_operation(
            self.client, operation_id, owner=owner, fence=fence, error=error, now=now,
        ):
            raise RuntimeError("could not persist terminal operation failure")

    def run_stage(
        self,
        operation_id: str,
        stage_key: str,
        callback: Callable[[], Any],
        *,
        now: datetime | None = None,
        retryable: Callable[[BaseException], bool] | None = None,
        reconcile: Callable[[], Any] | None = None,
        save_intent_id: str | None = None,
        checkpoint_refs: tuple[str, ...] = (),
    ) -> Any:
        """Run one reserved callback or reconcile its durable attempt.

        ``save_intent_id`` remains accepted for compatibility; immutable save intent
        belongs to the operation record.
        """
        del save_intent_id
        now = self._now(now)
        stage = self.get_stage(operation_id, stage_key)
        if stage.state == StageState.COMPLETED:
            return stage.result
        self._check_retry_eligibility(stage, stage_key, now)

        owner = f"retry-governor-{uuid4().hex}"
        lease = claim_operation(
            self.client, operation_id, owner=owner, now=now, lease_seconds=self.lease_seconds,
        )
        if lease is None:
            raise RuntimeError("operation lease is unavailable")
        fence = lease.fence
        attempt_id = stage.in_flight_attempt_id
        try:
            if attempt_id is not None:
                return self._reconcile_attempt(
                    operation_id, stage_key, attempt_id, reconcile,
                    owner=owner, fence=fence, checkpoint_refs=checkpoint_refs,
                )
            reserved, attempt_id_or_result = self._reserve_attempt(
                operation_id, stage_key, owner, fence, now,
            )
            if not reserved:
                return attempt_id_or_result
            return self._execute_attempt(
                operation_id, stage_key, attempt_id_or_result, callback, retryable,
                owner=owner, fence=fence, checkpoint_refs=checkpoint_refs,
            )
        finally:
            release_operation(self.client, operation_id, owner=owner, fence=fence, now=self._now())

    def _check_retry_eligibility(
        self, stage: OperationStageRecord, stage_key: str, now: datetime,
    ) -> None:
        if stage.in_flight_attempt_id is not None:
            return
        if stage.failures >= self.max_failures:
            raise RetryBudgetExceeded(f"stage {stage_key!r} exhausted its retry budget")
        if stage.last_error is not None and not stage.last_error.retryable:
            raise RetryBudgetExceeded(f"stage {stage_key!r} has a permanent failure")
        if stage.retry_not_before is not None and stage.retry_not_before > now:
            raise RetryBudgetExceeded(f"stage {stage_key!r} is waiting for its retry time")

    def _reserve_attempt(
        self, operation_id: str, stage_key: str, owner: str, fence: int, now: datetime,
    ) -> tuple[bool, Any]:
        """Return reservation status and its attempt ID or completed result."""
        attempt_id = uuid4().hex
        reserved = reserve_stage_attempt(
            self.client, operation_id, stage_key, owner=owner, fence=fence,
            attempt_id=attempt_id, now=now,
        )
        if reserved is not None:
            return True, attempt_id
        current = self.get_stage(operation_id, stage_key)
        if current.in_flight_attempt_id is not None:
            raise UnknownOutcome("stage has an unresolved provider outcome")
        if current.state == StageState.COMPLETED:
            return False, current.result
        if current.failures >= self.max_failures:
            raise RetryBudgetExceeded(f"stage {stage_key!r} exhausted its retry budget")
        raise RuntimeError("could not reserve stage attempt")

    def _reconcile_attempt(
        self, operation_id: str, stage_key: str, attempt_id: str,
        reconcile: Callable[[], Any] | None, *, owner: str, fence: int,
        checkpoint_refs: tuple[str, ...],
    ) -> Any:
        if reconcile is None:
            raise UnknownOutcome("stage has an unresolved provider outcome")
        try:
            reconciled = reconcile()
        except BaseException as error:
            raise UnknownOutcome("stage outcome remains unresolved") from error
        if reconciled is None:
            raise UnknownOutcome("stage outcome remains unresolved")
        if isinstance(reconciled, ConfirmedStageFailure):
            self._record_reconciled_failure(
                operation_id, stage_key, attempt_id, reconciled.error, owner=owner, fence=fence,
            )
            raise ReconciledStageFailure(reconciled.error) from None
        result = reconciled.result if isinstance(reconciled, ConfirmedStageSuccess) else reconciled
        self._complete_attempt(
            operation_id, stage_key, attempt_id, result, owner=owner, fence=fence,
            checkpoint_refs=checkpoint_refs, reconciled=True,
        )
        return result

    def _record_reconciled_failure(
        self, operation_id: str, stage_key: str, attempt_id: str, error: OperationError,
        *, owner: str, fence: int,
    ) -> None:
        if not fail_stage_attempt(
            self.client, operation_id, stage_key, owner=owner, fence=fence,
            attempt_id=attempt_id, error=error, now=self._now(),
        ):
            raise RuntimeError("could not persist reconciled stage failure")
        self._fail_operation_if_terminal(
            operation_id, stage_key, owner=owner, fence=fence, error=error, now=self._now(),
        )

    def _execute_attempt(
        self, operation_id: str, stage_key: str, attempt_id: str,
        callback: Callable[[], Any], retryable: Callable[[BaseException], bool] | None,
        *, owner: str, fence: int, checkpoint_refs: tuple[str, ...],
    ) -> Any:
        try:
            result = callback()
        except UnknownOutcome:
            raise
        except Exception as error:
            should_retry = retryable(error) if retryable is not None else True
            safe_error = OperationError(
                code=type(error).__name__.lower(), category="execution",
                message=f"Stage execution failed ({type(error).__name__})",
                retryable=bool(should_retry), stage=stage_key,
            )
            self._record_execution_failure(
                operation_id, stage_key, attempt_id, safe_error, owner=owner, fence=fence,
                original_error=error,
            )
            raise
        self._complete_attempt(
            operation_id, stage_key, attempt_id, result, owner=owner, fence=fence,
            checkpoint_refs=checkpoint_refs, reconciled=False,
        )
        return result

    def _record_execution_failure(
        self, operation_id: str, stage_key: str, attempt_id: str, error: OperationError,
        *, owner: str, fence: int, original_error: Exception,
    ) -> None:
        if not fail_stage_attempt(
            self.client, operation_id, stage_key, owner=owner, fence=fence,
            attempt_id=attempt_id, error=error, now=self._now(),
        ):
            raise RuntimeError("could not persist stage failure") from original_error
        self._fail_operation_if_terminal(
            operation_id, stage_key, owner=owner, fence=fence, error=error, now=self._now(),
        )

    def _complete_attempt(
        self, operation_id: str, stage_key: str, attempt_id: str, result: Any,
        *, owner: str, fence: int, checkpoint_refs: tuple[str, ...], reconciled: bool,
    ) -> None:
        if not complete_stage_attempt(
            self.client, operation_id, stage_key, owner=owner, fence=fence,
            attempt_id=attempt_id, result=result, now=self._now(), checkpoint_refs=checkpoint_refs,
        ):
            label = "reconciled stage result" if reconciled else "stage result"
            raise RuntimeError(f"could not persist {label}")

    def record_wait(self, operation_id: str, stage_key: str, *, until: datetime) -> None:
        """Record a scheduler wait in memory only; it is not a failed execution."""
        del operation_id, stage_key, until

    def record_yield(self, operation_id: str, stage_key: str, *, reason: str) -> None:
        """Yielding execution does not reserve or fail a provider attempt."""
        del operation_id, stage_key, reason

    def record_status(self, operation_id: str, stage_key: str, *, status: str) -> OperationStageRecord:
        """Read the durable stage without changing its execution budget."""
        del status
        return self.get_stage(operation_id, stage_key)

    def record_lease_renewal(
        self,
        operation_id: str,
        *,
        owner: str,
        fence: int,
        now: datetime | None = None,
    ) -> bool:
        renewed = renew_operation_lease(
            self.client, operation_id, owner=owner, fence=fence, now=self._now(now),
            lease_seconds=self.lease_seconds,
        )
        return renewed is not None
