"""Single-head worker for Telegram context operations."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.operation_queue import (
    HEARTBEAT_INTERVAL_SECONDS,
    claim_head,
    complete_head,
    fail_head,
    get_operation,
    heartbeat,
    release_head,
)
from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxState,
    OutboxTargetKind,
    create_outbox_record_in_transaction,
    get_outbox_record,
)
from core.operations.queue import OperationLease, QueueOperationState
from core.operations.retries import RetryGovernor, RetryBudgetExceeded, UnknownOutcome
from services.connectors.telegram.repository import SESSIONS


Processor = Callable[[str, Any, OperationLease], Awaitable[QueueOperationState]]
Clock = Callable[[], datetime]
_SYNC_POOL = ThreadPoolExecutor(thread_name_prefix="telegram-firestore")
MAX_EXECUTION_SLICE_SECONDS = 480.0


class _WorkerClock:
    def __init__(self, worker: "TelegramQueueWorker") -> None:
        self.worker = worker

    def now(self) -> datetime:
        return self.worker.clock()


async def _offload_sync(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run blocking SDK work off-loop while polling so Windows selector loops wake reliably."""
    future = _SYNC_POOL.submit(function, *args, **kwargs)
    while not future.done():
        await asyncio.sleep(0.01)
    return future.result()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _wake_outbox_id(context_id: str, operation_id: str, generation: int) -> str:
    digest = hashlib.sha256(f"{context_id}:{operation_id}:{generation}".encode()).hexdigest()
    return f"tgqueue-{digest}"


def _ensure_wakeup_outbox_record(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    operation_id: str,
    generation: int,
    schedule_at: datetime,
    now: datetime,
) -> int:
    """Reuse or create a wakeup; a consumed wakeup cannot wake the head again.

    Advancing its generation keeps this repair retryable and idempotent.
    """
    while True:
        outbox_id = _wake_outbox_id(context_id, operation_id, generation)
        outbox_ref = client.collection(OUTBOX_COLLECTION).document(outbox_id)
        snapshot = outbox_ref.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            matching = (
                data.get("target_kind") == OutboxTargetKind.TELEGRAM_DISPATCH.value
                and data.get("context_id") == context_id
                and data.get("operation_id") == operation_id
                and int(data.get("generation", 0)) == generation
            )
            if not matching:
                generation += 1
                continue
            if data.get("state") in {OutboxState.PENDING.value, OutboxState.RETRY.value}:
                break
            generation += 1
            continue
        create_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=generation,
            schedule_at=schedule_at,
            now=now,
            context_id=context_id,
            operation_id=operation_id,
        )
        break
    return generation


def _ensure_wakeup_in_transaction(
    client: Any,
    transaction: Any,
    context: dict[str, Any],
    *,
    context_id: str,
    operation_id: str,
    schedule_at: datetime,
    now: datetime,
) -> None:
    """Write/reuse a deterministic wakeup while the caller holds the context txn."""
    initial_generation = int(context.get("fence", 0)) + 1
    generation = max(
        initial_generation,
        int(context.get("telegram_dispatch_wake_generation", initial_generation)),
    )
    generation = _ensure_wakeup_outbox_record(
        client, transaction,
        context_id=context_id,
        operation_id=operation_id,
        generation=generation,
        schedule_at=schedule_at,
        now=now,
    )
    context["telegram_dispatch_wake_generation"] = generation


def _ensure_head_wakeup(client: Any, context_id: str, *, now: datetime) -> str | None:
    context_ref = client.collection(SESSIONS).document(context_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> str | None:
        snapshot = context_ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        context = snapshot.to_dict() or {}
        operation_id = context.get("reserved_operation_id")
        if not operation_id:
            return None
        expiry = context.get("lock_expires_at")
        active_id = context.get("active_operation_id")
        schedule_at = expiry if active_id == operation_id and isinstance(expiry, datetime) and expiry > now else now
        _ensure_wakeup_in_transaction(
            client, transaction, context,
            context_id=context_id, operation_id=operation_id,
            schedule_at=schedule_at, now=now,
        )
        # A write to the context makes repair serialize with a concurrent finish.
        transaction.set(context_ref, context)
        return operation_id

    return _txn(client.transaction(max_attempts=5))


class TelegramQueueWorker:
    """Validate an outbox delivery, claim its actual context head, and process it."""

    def __init__(
        self,
        client: Any,
        processor: Processor,
        *,
        clock: Clock = _utc_now,
        owner_factory: Callable[[], str] | None = None,
        heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
        max_slice_seconds: float = MAX_EXECUTION_SLICE_SECONDS,
    ) -> None:
        if not callable(processor) or not callable(clock):
            raise TypeError("processor and clock must be callable")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if max_slice_seconds <= 0 or max_slice_seconds > MAX_EXECUTION_SLICE_SECONDS:
            raise ValueError("max_slice_seconds must be positive and at most 480 seconds")
        self.client = client
        self.processor = processor
        self.clock = clock
        self.owner_factory = owner_factory or (lambda: uuid.uuid4().hex)
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.max_slice_seconds = max_slice_seconds

    def _retry_governor(self, context_ref: Any) -> RetryGovernor:
        from core.integrations.firestore.operation_queue import QueueOperationRetryStore

        return RetryGovernor(
            self.client, clock=_WorkerClock(self),
            retry_store=QueueOperationRetryStore(self.client, context_ref),
        )

    async def _process(self, context_id: str, operation: Any, lease: OperationLease,
                       retry_governor: RetryGovernor) -> QueueOperationState:
        """Pass durable attempt context to processors that opt into the seam."""
        try:
            parameters = inspect.signature(self.processor).parameters.values()
            supports_governor = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == "retry_governor"
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_governor = False
        if supports_governor:
            return await self.processor(context_id, operation, lease, retry_governor=retry_governor)
        return await self.processor(context_id, operation, lease)

    async def handle_task(self, payload: Mapping[str, Any]) -> bool:
        """Handle a schema-versioned task body; return whether processing settled."""
        delivery = await self._load_valid_delivery(payload)
        if delivery is None:
            return False
        record, delivery_time = delivery

        claim = await self._claim_delivery_head(record, delivery_time)
        if claim is None:
            return False
        context_id, context_ref, operation, lease = claim

        return await self._process_claimed_head(context_id, context_ref, operation, lease)

    async def _load_valid_delivery(self, payload: Mapping[str, Any]) -> tuple[Any, datetime] | None:
        """Validate the task envelope and stored outbox generation, repairing stale wakeups."""
        if not isinstance(payload, Mapping):
            raise ValueError("task payload must be a mapping")
        outbox_id = payload.get("outbox_id")
        generation = payload.get("generation")
        target_kind = payload.get("target_kind")
        if not isinstance(outbox_id, str) or not outbox_id:
            raise ValueError("outbox_id is required")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")

        record = await _offload_sync(get_outbox_record, self.client, outbox_id)
        if record is None:
            return None
        context_id = record.context_id
        now = self.clock()
        valid_delivery = (
            payload.get("schema_version") == 1
            and target_kind == OutboxTargetKind.TELEGRAM_DISPATCH.value
            and record.target_kind is OutboxTargetKind.TELEGRAM_DISPATCH
            and record.generation == generation
            and bool(context_id)
            and bool(record.operation_id)
        )
        if not valid_delivery:
            if context_id:
                await _offload_sync(_ensure_head_wakeup, self.client, context_id, now=now)
            return None
        return record, now

    async def _claim_delivery_head(
        self, record: Any, delivery_time: datetime,
    ) -> tuple[str, Any, Any, OperationLease] | None:
        """Confirm the delivered operation is the context head and claim its fenced lease."""
        context_id = record.context_id
        context_ref = self.client.collection(SESSIONS).document(context_id)
        context_snapshot = await _offload_sync(context_ref.get)
        context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        if context.get("reserved_operation_id") != record.operation_id:
            await _offload_sync(_ensure_head_wakeup, self.client, context_id, now=delivery_time)
            return None
        operation = await _offload_sync(get_operation, context_ref, record.operation_id)
        if operation is None or operation.state.is_terminal:
            await _offload_sync(_ensure_head_wakeup, self.client, context_id, now=delivery_time)
            return None

        lease = await _offload_sync(
            claim_head, self.client, context_ref, owner=self.owner_factory(), now=delivery_time,
        )
        if lease is None:
            await _offload_sync(_ensure_head_wakeup, self.client, context_id, now=self.clock())
            return None
        if lease.operation_id != record.operation_id:
            await _offload_sync(release_head,
                self.client, context_ref, operation_id=lease.operation_id,
                owner=lease.owner, fence=lease.fence, now=self.clock(),
            )
            await _offload_sync(_ensure_head_wakeup, self.client, context_id, now=self.clock())
            return None
        return context_id, context_ref, operation, lease

    async def _process_claimed_head(
        self, context_id: str, context_ref: Any, operation: Any, lease: OperationLease,
    ) -> bool:
        """Run the claimed operation with heartbeat and route its outcome to settlement."""
        try:
            retry_governor = self._retry_governor(context_ref)
            result = await self._run_with_heartbeat(context_id, operation, lease, context_ref,
                retry_governor)
        except Exception as error:
            retryable_stage = self._retryable_error_stage(error)
            if retryable_stage is None:
                raise
            return await self._settle_processor_error(
                context_id, context_ref, lease, retryable_stage,
            )

        return await self._settle_processor_result(context_id, context_ref, lease, result)

    @staticmethod
    def _retryable_error_stage(error: Exception) -> bool | None:
        """Resolve explicit retry classification while honoring governor outcome types."""
        retryable_stage = getattr(error, "retryable_stage", None)
        if isinstance(error, (RetryBudgetExceeded,)):
            retryable_stage = False
        if isinstance(error, UnknownOutcome):
            retryable_stage = True
        return retryable_stage if isinstance(retryable_stage, bool) else None

    async def _settle_processor_error(
        self, context_id: str, context_ref: Any, lease: OperationLease,
        retryable_stage: bool,
    ) -> bool:
        """Release retryable failures for another attempt or terminally fail and advance."""
        settled_at = self.clock()
        if retryable_stage:
            def wake_retry(transaction: Any, updated_context: dict[str, Any]) -> None:
                _ensure_wakeup_in_transaction(
                    self.client, transaction, updated_context,
                    context_id=context_id, operation_id=lease.operation_id,
                    schedule_at=settled_at, now=settled_at,
                )
            return await _offload_sync(release_head, self.client, context_ref,
                operation_id=lease.operation_id, owner=lease.owner, fence=lease.fence,
                now=settled_at, after_release=wake_retry) and False

        def wake_next_after_failure(transaction: Any, next_id: str | None,
                                    next_data: dict[str, Any] | None,
                                    updated_context: dict[str, Any]) -> None:
            if next_id and next_data is not None:
                _ensure_wakeup_in_transaction(
                    self.client, transaction, updated_context,
                    context_id=context_id, operation_id=next_id,
                    schedule_at=settled_at, now=settled_at,
                )
        return await _offload_sync(fail_head, self.client, context_ref,
            operation_id=lease.operation_id, owner=lease.owner, fence=lease.fence,
            now=settled_at, after_advance=wake_next_after_failure)

    async def _settle_processor_result(
        self, context_id: str, context_ref: Any, lease: OperationLease,
        result: QueueOperationState | None,
    ) -> bool:
        """Release a yielded head or settle a terminal result and wake the next head."""
        if result is None:
            return False
        if result is QueueOperationState.RUNNING:
            yielded_at = self.clock()

            def wake_same_head(transaction: Any, updated_context: dict[str, Any]) -> None:
                _ensure_wakeup_in_transaction(
                    self.client, transaction, updated_context,
                    context_id=context_id, operation_id=lease.operation_id,
                    schedule_at=yielded_at, now=yielded_at,
                )

            await _offload_sync(
                release_head, self.client, context_ref,
                operation_id=lease.operation_id, owner=lease.owner,
                fence=lease.fence, now=yielded_at,
                after_release=wake_same_head,
            )
            return False

        settle_now = self.clock()

        def wake_next(transaction: Any, next_id: str | None, next_data: dict[str, Any] | None,
                      updated_context: dict[str, Any]) -> None:
            if updated_context.get("pending_save_operation_id") == lease.operation_id:
                updated_context.pop("pending_save_operation_id", None)
            if next_id and next_data is not None:
                _ensure_wakeup_in_transaction(
                    self.client, transaction, updated_context,
                    context_id=context_id, operation_id=next_id,
                    schedule_at=settle_now, now=settle_now,
                )

        settle = complete_head if result is QueueOperationState.COMPLETED else fail_head
        settled = await _offload_sync(settle,
            self.client, context_ref, operation_id=lease.operation_id,
            owner=lease.owner, fence=lease.fence, now=settle_now,
            after_advance=wake_next,
        )
        if not settled:
            return False
        return True

    async def _run_with_heartbeat(
        self, context_id: str, operation: Any, lease: OperationLease, context_ref: Any,
        retry_governor: RetryGovernor,
    ) -> QueueOperationState | None:
        processor_task = asyncio.create_task(self._process(context_id, operation, lease, retry_governor))

        async def beat() -> bool:
            while True:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                renewed = await _offload_sync(heartbeat,
                    self.client, context_ref, operation_id=lease.operation_id,
                    owner=lease.owner, fence=lease.fence, now=self.clock(),
                )
                if renewed is None:
                    return False

        heartbeat_task = asyncio.create_task(beat())
        release_lease = False
        try:
            done, _ = await asyncio.wait(
                {processor_task, heartbeat_task}, timeout=self.max_slice_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                processor_task.cancel()
                await asyncio.gather(processor_task, return_exceptions=True)
                return QueueOperationState.RUNNING
            if heartbeat_task in done and not heartbeat_task.result():
                return None
            result = await processor_task
            if result is QueueOperationState.RUNNING:
                return QueueOperationState.RUNNING
            if not isinstance(result, QueueOperationState) or not result.is_terminal:
                raise ValueError("processor must return a terminal or RUNNING QueueOperationState")
            return result
        except BaseException as error:
            release_lease = not (
                isinstance(error, (RetryBudgetExceeded, UnknownOutcome))
                or isinstance(getattr(error, "retryable_stage", None), bool)
            )
            raise
        finally:
            for task in (processor_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(processor_task, heartbeat_task, return_exceptions=True)
            if release_lease:
                # Release only while this exact lease is still current. If it was
                # fenced out, release_head safely returns False.
                await _offload_sync(release_head,
                    self.client, context_ref, operation_id=lease.operation_id,
                    owner=lease.owner, fence=lease.fence, now=self.clock(),
                )


__all__ = ["TelegramQueueWorker"]
