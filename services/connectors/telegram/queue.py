"""Single-head worker for Telegram context operations."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import re
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
    get_outbox_record,
    prepare_outbox_record_in_transaction,
    write_prepared_outbox_record,
)
from core.operations.queue import OperationLease, QueueOperationState
from core.operations.retries import RetryGovernor, RetryBudgetExceeded, UnknownOutcome
from services.connectors.telegram.repository import NOTIFICATIONS, SESSIONS, TelegramRepository
from services.connectors.telegram.repository._common import _document_ref


Processor = Callable[[str, Any, OperationLease], Awaitable[QueueOperationState]]
Clock = Callable[[], datetime]
_SYNC_POOL = ThreadPoolExecutor(thread_name_prefix="telegram-firestore")
MAX_EXECUTION_SLICE_SECONDS = 480.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,480}$")


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


def _fixed_target(operation: Mapping[str, Any]) -> tuple[str | None, bool]:
    """Read one unambiguous target already captured in durable route metadata."""
    route = operation.get("route_result")
    candidates = [operation.get("target_doc_id")]
    if isinstance(route, Mapping):
        candidates.append(route.get("target_doc_id"))
    if any(value is not None and (not isinstance(value, str) or not value) for value in candidates):
        return None, False
    targets = {target for target in candidates if isinstance(target, str)}
    if len(targets) > 1:
        return None, False
    return (next(iter(targets)) if targets else None), True


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
    prepared_writes: list[Any] | None = None,
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
        prepared = prepare_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=generation,
            schedule_at=schedule_at,
            now=now,
            context_id=context_id,
            operation_id=operation_id,
        )
        if prepared_writes is None:
            write_prepared_outbox_record(transaction, prepared)
        else:
            prepared_writes.append(prepared)
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
    prepared_writes: list[Any] | None = None,
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
        prepared_writes=prepared_writes,
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
        dependency = await _offload_sync(
            self._dependency_state, context_ref, operation.operation_id, operation.sequence,
        )
        if dependency[0] == "waiting":
            return await self._settle_processor_result(
                context_id, context_ref, lease, QueueOperationState.RUNNING,
            )
        if dependency[0] == "failed":
            return await self._settle_dependency_failure(
                context_id, context_ref, lease, dependency[1], dependency[2],
            )
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

    def _dependency_state(
        self, context_ref: Any, operation_id: str, sequence: int,
    ) -> tuple[str, str | None, str | None]:
        """Validate the current operation and classify its persisted dependency state."""
        operation_ref = context_ref.collection("operations").document(operation_id)
        snapshot = operation_ref.get()
        if not snapshot.exists:
            return "failed", None, "The queued command record is unavailable."
        current = snapshot.to_dict() or {}
        current_sequence = current.get("sequence")
        if (current.get("operation_id") != operation_id
                or isinstance(current_sequence, bool)
                or not isinstance(current_sequence, int) or current_sequence < 1
                or current_sequence != sequence):
            return "failed", None, "The queued command record is invalid."
        dependency_id = current.get("dependency_operation_id")
        if dependency_id is None:
            return "ready", None, None
        if not isinstance(dependency_id, str) or not _SAFE_ID.fullmatch(dependency_id):
            return "failed", None, "The command's confirmed prerequisite is invalid."
        if dependency_id == operation_id:
            return "failed", dependency_id, "The command's confirmed prerequisite is invalid."

        dependency_ref = context_ref.collection("operations").document(dependency_id)
        dependency_snapshot = dependency_ref.get()
        if not dependency_snapshot.exists:
            return "failed", dependency_id, "The command's confirmed prerequisite is unavailable."
        parent = dependency_snapshot.to_dict() or {}
        parent_sequence = parent.get("sequence")
        if (parent.get("operation_id") != dependency_id
                or isinstance(parent_sequence, bool)
                or not isinstance(parent_sequence, int)
                or parent_sequence < 1
                or parent_sequence >= sequence):
            return "failed", dependency_id, "The command's confirmed prerequisite is invalid."
        state = parent.get("state")
        if state in {QueueOperationState.PENDING.value, QueueOperationState.RUNNING.value}:
            return "waiting", dependency_id, None
        if state == QueueOperationState.FAILED.value:
            return "failed", dependency_id, "The confirmed prerequisite did not complete successfully."
        if state != QueueOperationState.COMPLETED.value:
            return "failed", dependency_id, "The command's confirmed prerequisite is unavailable."

        return self._completed_dependency_result(current, parent, dependency_id)

    def _completed_dependency_result(
        self, current: Mapping[str, Any], parent: Mapping[str, Any], dependency_id: str,
    ) -> tuple[str, str | None, str | None]:
        """Require one exact child target matched to the parent's completed save result."""
        child_target, child_target_valid = _fixed_target(current)
        parent_route_target, parent_route_valid = _fixed_target(parent)
        if not child_target_valid or not parent_route_valid:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        if child_target is None:
            return "failed", dependency_id, "The completed prerequisite has no saved result for this command."

        shared_operation_id, _ = self._dependency_save_identity(parent, dependency_id)
        if shared_operation_id is None:
            return "failed", dependency_id, "The completed prerequisite's saved result is unavailable."
        shared_ref = self.client.collection("dialog_operations").document(shared_operation_id)
        shared_snapshot = shared_ref.get()
        if not shared_snapshot.exists:
            return "failed", dependency_id, "The completed prerequisite's saved result is unavailable."
        shared = shared_snapshot.to_dict() or {}
        result = shared.get("result")
        result_target = result.get("document_id") if isinstance(result, Mapping) else None
        shared_doc_id = shared.get("doc_id")
        if (shared.get("operation_id") != shared_operation_id
                or shared.get("state") != "completed"
                or not isinstance(result_target, str) or not result_target):
            return "failed", dependency_id, "The completed prerequisite's saved result is unavailable."
        if shared_doc_id is not None and shared_doc_id != result_target:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        if parent.get("save_intent_id") is not None and shared.get("save_intent_id") != parent["save_intent_id"]:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        if parent.get("doc_id") is not None and parent.get("doc_id") != result_target:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        if parent_route_target is not None and parent_route_target != result_target:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        if child_target != result_target:
            return "failed", dependency_id, "The completed prerequisite's saved result is inconsistent."
        return "ready", dependency_id, None

    @staticmethod
    def _dependency_save_identity(
        parent: Mapping[str, Any], dependency_id: str,
    ) -> tuple[str | None, bool]:
        """Resolve an explicit save link or a numbered retry's same-ID save binding."""
        linked_id = parent.get("save_operation_id")
        save_intent_id = parent.get("save_intent_id")
        retry_save = linked_id is None and isinstance(save_intent_id, str) and bool(save_intent_id)
        if retry_save and (
            not isinstance(parent.get("draft_id"), str) or not parent.get("draft_id")
            or type(parent.get("dialog_number")) is not int or parent["dialog_number"] < 1
            or not isinstance(parent.get("doc_id"), str) or not parent.get("doc_id")
        ):
            return None, retry_save
        shared_operation_id = linked_id or (dependency_id if retry_save else None)
        if not isinstance(shared_operation_id, str) or not _SAFE_ID.fullmatch(shared_operation_id):
            return None, retry_save
        if save_intent_id is not None and (
            not isinstance(save_intent_id, str) or not _SAFE_ID.fullmatch(save_intent_id)
        ):
            return None, retry_save
        return shared_operation_id, retry_save

    @staticmethod
    def _dependency_notification_identity(operation_id: str) -> tuple[str, str]:
        """Derive stable ledger and dispatch IDs without embedding command content."""
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return f"tgdep-{digest}", f"tgdepdispatch-{digest}"

    @staticmethod
    def _route_dialog_number(operation: Mapping[str, Any]) -> tuple[int | None, bool]:
        """Resolve a consistent positive dialog number from confirmed route metadata."""
        route = operation.get("route_result")
        candidates = [operation.get("target_dialog_number")]
        if isinstance(route, Mapping):
            candidates.append(route.get("target_dialog_number"))
            target = route.get("target")
            if isinstance(target, Mapping):
                candidates.append(target.get("dialog_number"))
        if any(value is not None and (type(value) is not int or value < 1) for value in candidates):
            return None, False
        numbers = {value for value in candidates if type(value) is int}
        if len(numbers) > 1:
            return None, False
        return (next(iter(numbers)) if numbers else None), True

    def _dependency_dialog_number(
        self, transaction: Any, context_ref: Any, dependency_id: str,
    ) -> int | None:
        """Use only a terminal context-local dependency and its verified save binding."""
        parent_ref = context_ref.collection("operations").document(dependency_id)
        parent_snapshot = parent_ref.get(transaction=transaction)
        parent = parent_snapshot.to_dict() or {} if parent_snapshot.exists else {}
        if (parent.get("operation_id") != dependency_id
                or parent.get("state") not in {QueueOperationState.COMPLETED.value,
                                                QueueOperationState.FAILED.value}):
            return None
        dialog_number, valid = self._route_dialog_number(parent)
        if not valid or dialog_number is not None:
            return dialog_number
        shared_operation_id, _ = self._dependency_save_identity(parent, dependency_id)
        if shared_operation_id is None:
            return None
        draft_id = parent.get("draft_id") or parent.get("target_draft_id")
        if isinstance(draft_id, str) and _SAFE_ID.fullmatch(draft_id):
            draft_ref = context_ref.collection("drafts").document(draft_id)
            draft_snapshot = draft_ref.get(transaction=transaction)
            draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
            draft_number = draft.get("dialog_number")
            parent_number = parent.get("dialog_number")
            if (draft.get("draft_id") == draft_id and type(draft_number) is int
                    and draft_number >= 1
                    and (parent_number is None or (type(parent_number) is int
                                                   and parent_number == draft_number))):
                return draft_number
        shared_ref = self.client.collection("dialog_operations").document(shared_operation_id)
        shared_snapshot = shared_ref.get(transaction=transaction)
        shared = shared_snapshot.to_dict() or {} if shared_snapshot.exists else {}
        if (shared.get("operation_id") != shared_operation_id
                or shared.get("state") not in {"completed", "failed"}
                or (parent.get("save_intent_id") is not None
                    and shared.get("save_intent_id") != parent.get("save_intent_id"))):
            return None
        dialog_number, valid = self._route_dialog_number(shared)
        return dialog_number if valid else None

    def _prepare_dependency_notification(
        self, transaction: Any, *, context_id: str, context_ref: Any,
        operation_data: Mapping[str, Any], dependency_id: str | None, message: str | None,
        now: datetime, notification_id: str, notification_outbox_id: str,
        updated_context: dict[str, Any], next_id: str | None,
        next_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Read and prepare notification and wakeup rows before settlement writes."""
        from services.connectors.telegram.errors import format_terminal_error

        notification_ref = self.client.collection(NOTIFICATIONS).document(notification_id)
        notification_snapshot = notification_ref.get(transaction=transaction)
        delivery = prepare_outbox_record_in_transaction(
            self.client, transaction, notification_outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1, schedule_at=now, now=now,
            context_id=context_id, notification_id=notification_id,
        )
        command_data = self._load_context_command(
            transaction, context_id, operation_data, operation_data.get("operation_id"),
        )
        dialog_number, route_number_valid = self._route_dialog_number(operation_data)
        if not command_data or not route_number_valid:
            dialog_number = None
        if command_data and route_number_valid and dialog_number is None and dependency_id:
            dialog_number = self._dependency_dialog_number(
                transaction, context_ref, dependency_id,
            )
        error_message = message or "The command's confirmed prerequisite is unavailable."
        body = format_terminal_error(
            dialog_number=dialog_number, operation_id=operation_data.get("operation_id"),
            service="telegram", stage="dependency", code="dependency_unavailable",
            category="dependency", sanitized_message=error_message,
            occurred_at=now, failures=0,
        )
        if dependency_id is not None and _SAFE_ID.fullmatch(dependency_id):
            body = f"{body} Prerequisite operation {dependency_id}."[:4096]
        notice = TelegramRepository(self.client).prepare_notification_in_transaction(
            notification_id, context_id=context_id, kind="error", body=body, now=now,
            context_data=updated_context,
            existing_data=(notification_snapshot.to_dict() or {}) if notification_snapshot.exists else None,
            dialog_number=dialog_number, operation_id=operation_data.get("operation_id"),
        )
        wakeups: list[Any] = []
        if next_id and next_data is not None:
            _ensure_wakeup_in_transaction(
                self.client, transaction, updated_context,
                context_id=context_id, operation_id=next_id,
                schedule_at=now, now=now, prepared_writes=wakeups,
            )
        return {
            "notification_ref": notification_ref,
            "notification_exists": notification_snapshot.exists,
            "notice": notice,
            "delivery": delivery,
            "wakeups": wakeups,
            "dialog_number": dialog_number,
            "error_message": error_message,
        }

    def _load_context_command(
        self, transaction: Any, context_id: str, operation_data: Mapping[str, Any],
        operation_id: Any,
    ) -> dict[str, Any]:
        """Load only this operation's command document and reject foreign input refs."""
        path = operation_data.get("input_ref")
        parts = path.split("/") if isinstance(path, str) else []
        if (len(parts) != 4 or parts[0] != SESSIONS or parts[1] != context_id
                or parts[2] != "commands" or not parts[3]):
            return {}
        snapshot = _document_ref(self.client, path).get(transaction=transaction)
        command = snapshot.to_dict() or {} if snapshot.exists else {}
        if command.get("context_id") != context_id or command.get("operation_id") != operation_id:
            return {}
        return command

    async def _settle_dependency_failure(
        self, context_id: str, context_ref: Any, lease: OperationLease,
        dependency_id: str | None, message: str | None,
    ) -> bool:
        """Atomically fail a blocked command, persist its notice, and wake the next head."""
        settled_at = self.clock()
        notification_id, notification_outbox_id = self._dependency_notification_identity(
            lease.operation_id,
        )

        def settle_dependency(
            transaction: Any, next_id: str | None, next_data: dict[str, Any] | None,
            updated_context: dict[str, Any], operation_data: dict[str, Any],
        ) -> None:
            prepared = self._prepare_dependency_notification(
                transaction, context_id=context_id, context_ref=context_ref,
                operation_data=operation_data, dependency_id=dependency_id, message=message,
                now=settled_at, notification_id=notification_id,
                notification_outbox_id=notification_outbox_id,
                updated_context=updated_context, next_id=next_id, next_data=next_data,
            )
            if updated_context.get("pending_save_operation_id") == lease.operation_id:
                updated_context.pop("pending_save_operation_id", None)
            error_message = prepared["error_message"]
            if operation_data.get("operation_id") == lease.operation_id:
                operation_data.update({
                    "stage": "dependency",
                    "status_updated_at": settled_at,
                    "error": {
                        "service": "telegram", "stage": "dependency",
                        "code": "dependency_unavailable", "category": "dependency",
                        "message": error_message,
                        "sanitized_message": error_message, "retryable": False,
                        "occurred_at": settled_at, "failures": 0,
                        "operation_id": lease.operation_id,
                    },
                    "notification_id": notification_id,
                    "finished_at": settled_at,
                })
            write_prepared_outbox_record(transaction, prepared["delivery"])
            if not prepared["notification_exists"]:
                transaction.set(prepared["notification_ref"], prepared["notice"])
            for wakeup in prepared["wakeups"]:
                write_prepared_outbox_record(transaction, wakeup)

        return await _offload_sync(
            fail_head, self.client, context_ref, operation_id=lease.operation_id,
            owner=lease.owner, fence=lease.fence, now=settled_at,
            after_advance_with_operation=settle_dependency,
        )

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
            if updated_context.get("pending_save_operation_id") == lease.operation_id:
                updated_context.pop("pending_save_operation_id", None)
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
