"""Lazy composition of concrete Telegram command runtime adapters."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import firestore
from typing import Any, Mapping

from core.integrations.gemini.agent import GeminiAgent
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.operations.queue import QueueOperationState
from core.operations.models import OperationError
from core.operations.retries import ConfirmedStageFailure, ConfirmedStageSuccess
from services.connectors.telegram.agent import TelegramAgent, TelegramCommandProcessor
from services.connectors.telegram.client import TelegramClient
from services.connectors.telegram.models import TelegramContext
from services.connectors.telegram.notifications import TelegramNotificationPublisher
from services.connectors.telegram.repository import TelegramRepository
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from services.connectors.telegram.runtime.routing import GeminiCommandRouter
from services.connectors.telegram.runtime.store import TelegramRuntimeStore
from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

_SYNC_POOL = ThreadPoolExecutor(thread_name_prefix="telegram-command-runtime")


async def _offload(function: Any, *args: Any, **kwargs: Any) -> Any:
    future = _SYNC_POOL.submit(function, *args, **kwargs)
    while not future.done():
        await asyncio.sleep(0.01)
    return future.result()


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


class _GovernedModel:
    """Apply the queue's durable retry budget to every physical model call."""
    def __init__(self, model: Any, governor: Any, operation_id: str, lease: Any,
                 clock: Any, lease_check: Any, payload_codec: Any = None):
        self.model, self.governor = model, governor
        self.operation_id, self.lease, self.clock = operation_id, lease, clock
        self.lease_check = lease_check
        self.payload_codec = payload_codec
        self.failure: tuple[BaseException, bool] | None = None
        self.failure_stage: str | None = None

    def generate(self, **kwargs: Any) -> Any:
        if not self.lease_check():
            raise PermissionError("Telegram operation lease is stale")
        if self.governor is None:
            return _safe_response(self.model.generate(**kwargs))
        request_digest = hashlib.sha256(json.dumps(
            _fingerprint(kwargs), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()[:40]
        mode = kwargs.get("mode", "unknown")
        stage_key = f"telegram-model-{mode}-{request_digest}"
        try:
            result = self.governor.run_stage(
                self.operation_id, stage_key,
                lambda: self._pack(stage_key, _safe_response(self.model.generate(**kwargs))), now=self.clock(),
                reconcile=lambda: ConfirmedStageFailure(OperationError(
                    code="unknown_outcome", category="execution",
                    message="Stage execution failed", retryable=True,
                    service="gemini", stage=stage_key,
                )),
                owner=_value(self.lease, "owner"), fence=_value(self.lease, "fence"),
            )
            return self._unpack(result)
        except Exception as exc:
            try:
                stage = self.governor.get_stage(self.operation_id, stage_key)
                terminal = (stage.failures >= self.governor.max_failures
                            or bool(stage.last_error and not stage.last_error.retryable))
            except Exception:
                terminal = False
            self.failure = (exc, terminal)
            self.failure_stage = stage_key
            raise

    def _pack(self, key: str, value: Any) -> Any:
        if self.payload_codec is None:
            return value
        return {"_telegram_retry_payload": self.payload_codec._store_payload(
            f"retry-model-{key}", value)}

    def _unpack(self, value: Any) -> Any:
        if (self.payload_codec is not None and isinstance(value, Mapping)
                and set(value) == {"_telegram_retry_payload"}):
            return self.payload_codec._load_payload(value["_telegram_retry_payload"])
        return value


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _fingerprint(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint(item) for item in value]
    from services.connectors.telegram.agent import _serialize_response
    return _fingerprint(_serialize_response(value))


def _safe_response(response: Any) -> dict[str, Any]:
    from services.connectors.telegram.agent import _serialize_response
    serialized = _serialize_response(response)
    if isinstance(serialized, Mapping):
        return dict(serialized)
    raise TypeError("Gemini response could not be serialized")


class _GovernedExecutor:
    def __init__(self, executor: TelegramRegistryExecutor, governor: Any,
                 operation_id: str, lease: Any, clock: Any, lease_check: Any,
                 payload_codec: Any = None):
        self.executor, self.governor = executor, governor
        self.operation_id, self.lease, self.clock = operation_id, lease, clock
        self.lease_check = lease_check
        self.payload_codec = payload_codec
        self.failure: tuple[BaseException, bool] | None = None
        self.failure_stage: str | None = None

    def execute(self, *, name: str, arguments: Any, call_ordinal: int) -> Any:
        if not self.lease_check():
            raise PermissionError("Telegram operation lease is stale")
        if self.governor is None:
            return self.executor.execute(name=name, arguments=arguments,
                                         call_ordinal=call_ordinal)
        stage_key = f"telegram-tool-{call_ordinal:02d}"
        try:
            result = self.governor.run_stage(
                self.operation_id, stage_key,
                lambda: self._pack(stage_key, self.executor.execute(
                    name=name, arguments=arguments, call_ordinal=call_ordinal)),
                reconcile=lambda: self._reconcile(stage_key, name, arguments, call_ordinal),
                now=self.clock(), owner=_value(self.lease, "owner"),
                fence=_value(self.lease, "fence"),
            )
            return self._unpack(result)
        except Exception as exc:
            try:
                stage = self.governor.get_stage(self.operation_id, stage_key)
                terminal = (stage.failures >= self.governor.max_failures
                            or bool(stage.last_error and not stage.last_error.retryable))
            except Exception:
                terminal = False
            self.failure = (exc, terminal)
            self.failure_stage = stage_key
            raise

    def _reconcile(self, stage_key: str, name: str, arguments: Any,
                   call_ordinal: int) -> Any:
        from core.operations import registry
        try:
            reconcile = getattr(self.executor, "reconcile", None)
            if callable(reconcile):
                result = reconcile(name=name, arguments=arguments,
                                   call_ordinal=call_ordinal)
            elif not registry.OPERATIONS[name].write:
                result = self.executor.execute(name=name, arguments=arguments,
                                               call_ordinal=call_ordinal)
            else:
                result = None
        except Exception:
            result = None
        if result is not None:
            return ConfirmedStageSuccess(self._pack(stage_key, result))
        return ConfirmedStageFailure(OperationError(
            code="unknown_outcome", category="execution",
            message="Stage execution failed",
            retryable=not registry.OPERATIONS[name].write,
            service="connected_service", stage=stage_key,
        ))

    def _pack(self, key: str, value: Any) -> Any:
        if self.payload_codec is None:
            return value
        return {"_telegram_retry_payload": self.payload_codec._store_payload(
            f"retry-tool-{key}", value)}

    def _unpack(self, value: Any) -> Any:
        if (self.payload_codec is not None and isinstance(value, Mapping)
                and set(value) == {"_telegram_retry_payload"}):
            return self.payload_codec._load_payload(value["_telegram_retry_payload"])
        return value


class _PrecomputedRouteModel:
    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)

    async def route(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.decision)


class _CapturingAgent:
    def __init__(self, agent: TelegramAgent):
        self.agent = agent
        self.answer: str | None = None

    def run(self, **kwargs: Any) -> dict[str, Any]:
        result = self.agent.run(**kwargs)
        answer = result.get("answer") if result.get("status") == "ok" else None
        self.answer = answer if isinstance(answer, str) else None
        if self.answer is None:
            raise RuntimeError("Telegram agent returned no answer")
        return result

    def dispatch_tool_choice(self, **kwargs: Any) -> Any:
        return self.agent.dispatch_tool_choice(**kwargs)


class _SearchPort:
    def __init__(self, store: TelegramRuntimeStore, clock: Any, *, governor: Any,
                 operation_id: str, lease: Any, lease_check: Any,
                 ledger: FirestoreAgentLedger):
        self.store, self.clock = store, clock
        self.governor, self.operation_id, self.lease = governor, operation_id, lease
        self.lease_check, self.ledger = lease_check, ledger
        self.failure: tuple[str, bool] | None = None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(key): _SearchPort._json_safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [_SearchPort._json_safe(item) for item in value]
        for method_name in ("model_dump", "to_json_dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    return _SearchPort._json_safe(method(mode="json", exclude_none=False))
                except TypeError:
                    return _SearchPort._json_safe(method())
        return str(value)

    async def search(self, query: str) -> list[str]:
        from core.operations import read
        if not await _offload(self.lease_check):
            raise PermissionError("Telegram operation lease is stale")
        if self.governor is None:
            try:
                segments, metadata = await _offload(read.query, query=query)
            except Exception:
                self.failure = ("search", False)
                raise
        else:
            stage_key = "telegram-search-" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]
            def retrieve() -> Any:
                segments, metadata = read.query(query=query)
                return {"segments": self._json_safe(segments),
                        "metadata": self._json_safe(metadata)}
            try:
                stored = await _offload(
                    self.governor.run_stage, self.operation_id, stage_key,
                    lambda: {"_telegram_retry_payload": self.ledger._store_payload(
                        f"retry-search-{stage_key}", retrieve())},
                    now=self.clock(), owner=_value(self.lease, "owner"),
                    fence=_value(self.lease, "fence"),
                    reconcile=lambda: ConfirmedStageFailure(OperationError(
                        code="unknown_outcome", category="execution",
                        message="Stage execution failed", retryable=True,
                        service="search", stage=stage_key,
                    )),
                )
                value = self.ledger._load_payload(stored["_telegram_retry_payload"])
                segments, metadata = value["segments"], value["metadata"]
            except Exception:
                try:
                    stage = self.governor.get_stage(self.operation_id, stage_key)
                    terminal = (stage.failures >= self.governor.max_failures
                                or bool(stage.last_error and not stage.last_error.retryable))
                except Exception:
                    terminal = False
                self.failure = (stage_key, terminal)
                raise
        self.store.pending_search = (query, segments, metadata)
        return [row["doc_id"] for row in segments if isinstance(row, Mapping)
                and isinstance(row.get("doc_id"), str)]


class _DeliveryStatus(str):
    """A durable reply disposition for every chunk in a command answer."""


class _DurableCommandReplies:
    """Persist and publish complete command answers in deterministic chunks."""

    def __init__(self, *, repository: Any, publisher: Any, context_id: str,
                 operation_id: str, owner: str, fence: int, clock: Any):
        self.repository, self.publisher = repository, publisher
        self.context_id, self.operation_id = context_id, operation_id
        self.owner, self.fence, self.clock = owner, fence, clock

    def notification_id(self, kind: str, index: int = 0) -> str:
        seed = (f"{self.operation_id}:{kind}" if index == 0
                else f"{self.operation_id}:{kind}:{index}")
        return "telegram-command-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()

    async def deliver(self, body: str, *, kind: str = "final") -> _DeliveryStatus:
        chunks = [body[index:index + 4000] for index in range(0, len(body), 4000)] or [""]
        context_ref = self.repository.client.collection("telegram_sessions").document(self.context_id)
        operation_ref = context_ref.collection("operations").document(self.operation_id)
        chunk_kinds = [kind if index == len(chunks) - 1 else "progress"
                       for index in range(len(chunks))]
        for index, chunk in enumerate(chunks):
            notification_id = self.notification_id(chunk_kinds[index], index)
            notification_ref = self.repository.client.collection("notification_outbox").document(
                notification_id)
            transaction = self.repository.client.transaction()
            @firestore.transactional
            def persist(tx: Any) -> bool:
                context_snapshot = context_ref.get(transaction=tx)
                operation_snapshot = operation_ref.get(transaction=tx)
                context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
                operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
                now = self.clock()
                if (not context_snapshot.exists or not operation_snapshot.exists
                        or operation.get("state") != "running"
                        or not validate_context_queue_lease(
                            context, self.operation_id, owner=self.owner, fence=self.fence,
                            now=now, expiry=context.get("lock_expires_at"))):
                    return _DeliveryStatus("stale")
                snapshot = notification_ref.get(transaction=tx)
                existing = snapshot.to_dict() or {} if snapshot.exists else None
                row = self.repository.prepare_notification_in_transaction(
                    notification_ref.id, context_id=self.context_id,
                    kind=chunk_kinds[index], body=chunk, now=now,
                    context_data=context, existing_data=existing,
                    operation_id=self.operation_id,
                )
                if not snapshot.exists:
                    tx.set(notification_ref, row)
                if index == len(chunks) - 1:
                    tx.update(operation_ref, {"notification_id": notification_ref.id})
                return True
            if persist(transaction) is not True:
                return _DeliveryStatus("stale")
            result = await self.publisher.publish(notification_id)
            if result != "delivered":
                disposition = "terminal" if result in {
                    "failed", "delivery_unknown", "superseded", "not_found", "stale",
                } else "pending"
                return _DeliveryStatus(disposition)
        return _DeliveryStatus("delivered")


def _terminal_failure_message(governor: Any, operation_id: str, stage_key: str,
                              *, service: str, stage: str,
                              dialog_number: int | None = None) -> str:
    """Format terminal diagnostics from the persisted safe stage record."""
    from services.connectors.telegram.errors import format_terminal_error
    try:
        record = governor.get_stage(operation_id, stage_key)
        error = record.last_error
    except Exception:
        record, error = None, None
    return format_terminal_error(
        dialog_number=dialog_number, operation_id=operation_id, service=service,
        stage=stage,
        code=getattr(error, "code", None), category=getattr(error, "category", None),
        sanitized_message=(getattr(error, "sanitized_message", None)
                           or getattr(error, "message", None)),
        occurred_at=getattr(error, "occurred_at", None),
        failures=getattr(record, "failures", None),
    )

class _NoopNotice:
    def send_switch_notice(self, **kwargs: Any) -> None:
        return None


class _NoopAction:
    """Deferred-hook adapter; validated tool execution runs after the notice barrier."""

    async def execute(self, **kwargs: Any) -> dict[str, bool]:
        return {"deferred": True}


class TelegramCommandRuntime:
    """Production queue processor with durable Firestore and provider ports.

    ``save_handler`` is required and implements async ``save_draft`` plus
    synchronous ``save_dialog``. The runtime never reports a save completed
    merely because a model routed a request to save.
    """

    def __init__(self, *, save_handler: Any, repository: Any | None = None,
                 client: Any | None = None, model: Any | None = None,
                 clock: Any = None, root_folder_id: str | None = None):
        if save_handler is None:
            raise TypeError("save_handler is required")
        self.save_handler = save_handler
        self.repository = repository or TelegramRepository()
        self.client = client
        self.model = model
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if root_folder_id is None:
            try:
                from core.config import get_root_folder_id
                root_folder_id = get_root_folder_id()
            except (KeyError, ValueError):
                root_folder_id = None
        self.root_folder_id = root_folder_id

    def _client(self) -> Any:
        if self.client is None:
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            if not token:
                raise ValueError("Telegram client is not configured")
            self.client = TelegramClient(token)
        return self.client

    def _model(self) -> Any:
        if self.model is None:
            self.model = GeminiAgent()
        return self.model

    def _load_context(self, context_id: str, operation: Any) -> TelegramContext | None:
        row = self.repository.get_context(context_id)
        if not isinstance(row, Mapping):
            return None
        values = {field.name: row[field.name] for field in fields(TelegramContext)
                  if field.name in row}
        try:
            context = TelegramContext(**values)
        except (TypeError, ValueError):
            return None
        if context.context_id != context_id:
            return None
        op_context = _value(operation, "context_id", context_id)
        return context if op_context == context_id else None

    def _queue_lease_live(self, context_id: str, operation_id: str,
                          owner: str, fence: int) -> bool:
        context_ref = self.repository.client.collection("telegram_sessions").document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        context_snapshot, operation_snapshot = context_ref.get(), operation_ref.get()
        context_data = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        operation_data = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        return bool(
            operation_snapshot.exists and operation_data.get("state") == "running"
            and validate_context_queue_lease(
                context_data, operation_id, owner=owner, fence=fence, now=self.clock(),
                expiry=context_data.get("lock_expires_at"),
            )
        )

    def _captured_draft_id(self, context_id: str, operation: Any) -> str | None:
        input_ref = _value(operation, "input_ref")
        prefix = f"telegram_sessions/{context_id}/commands/"
        if not isinstance(input_ref, str) or not input_ref.startswith(prefix):
            return None
        command_id = input_ref[len(prefix):]
        if not command_id or "/" in command_id:
            return None
        snapshot = (self.repository.client.collection("telegram_sessions")
                    .document(context_id).collection("commands").document(command_id).get())
        row = snapshot.to_dict() or {} if snapshot.exists else {}
        draft_id = row.get("draft_id")
        return draft_id if row.get("context_id") == context_id and row.get(
            "operation_id") == _value(operation, "operation_id") and isinstance(draft_id, str) and draft_id else None

    def _predecessor_save_candidates(self, context_id: str,
                                     operation: Any) -> tuple[dict[str, Any], ...]:
        """Return only fixed save bindings found in this operation's stored predecessor chain."""
        from services.connectors.telegram.sessions import _fixed_save_target

        operations = self.repository.client.collection("telegram_sessions").document(
            context_id).collection("operations")
        operation_id = _value(operation, "operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            return ()
        current_snapshot = operations.document(operation_id).get()
        current = current_snapshot.to_dict() or {} if current_snapshot.exists else {}
        if current.get("operation_id") != operation_id:
            return ()
        predecessor_id = current.get("predecessor_operation_id")
        seen = {operation_id}
        chain: list[dict[str, Any]] = []
        for _ in range(256):
            if not isinstance(predecessor_id, str) or not predecessor_id or predecessor_id in seen:
                break
            seen.add(predecessor_id)
            snapshot = operations.document(predecessor_id).get()
            row = snapshot.to_dict() or {} if snapshot.exists else {}
            if row.get("operation_id") != predecessor_id:
                break
            if row.get("context_id", context_id) != context_id:
                break
            chain.append(row)
            predecessor_id = row.get("predecessor_operation_id")
        candidates = []
        for row in chain:
            fixed = _fixed_save_target(row)
            if fixed is not None:
                candidates.append(fixed)
        return tuple(candidates)

    def _predecessor_search_results(self, context_id: str, operation: Any,
                                    cards: Any) -> tuple[dict[str, Any], ...]:
        """Return safe metadata in the nearest successful predecessor search order."""
        operations = self.repository.client.collection("telegram_sessions").document(
            context_id).collection("operations")
        operation_id = _value(operation, "operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            return ()
        snapshot = operations.document(operation_id).get()
        current = snapshot.to_dict() or {} if snapshot.exists else {}
        if (current.get("operation_id") != operation_id
                or current.get("context_id", context_id) != context_id):
            return ()
        predecessor_id = current.get("predecessor_operation_id")
        seen = {operation_id}
        rows_by_id = {row.get("doc_id"): row for row in cards
                      if isinstance(row, Mapping) and isinstance(row.get("doc_id"), str)}
        for _ in range(256):
            if not isinstance(predecessor_id, str) or not predecessor_id or predecessor_id in seen:
                break
            seen.add(predecessor_id)
            snapshot = operations.document(predecessor_id).get()
            row = snapshot.to_dict() or {} if snapshot.exists else {}
            if (row.get("operation_id") != predecessor_id
                    or row.get("context_id", context_id) != context_id):
                break
            route = row.get("route_control")
            if (row.get("state") == "completed" and isinstance(route, Mapping)
                    and route.get("status") in {"ok", "success"}
                    and route.get("kind") == "search"):
                results = []
                for doc_id in route.get("result_refs", []):
                    card = rows_by_id.get(doc_id)
                    if card is not None:
                        results.append({key: card[key] for key in (
                            "doc_id", "title", "date", "dialog_date", "created_at",
                            "folder", "folder_id", "folder_path", "dialog_number",
                            "conversation_id",
                        ) if key in card})
                return tuple(results)
            predecessor_id = row.get("predecessor_operation_id")
        return ()

    async def __call__(self, context_id: str, operation: Any, lease: Any, *,
                       command: Any, retry_governor: Any = None) -> QueueOperationState:
        """Create isolated invocation state and run one durable command."""
        return await _CommandExecution(
            self, context_id, operation, lease, command, retry_governor,
        ).run()


class _CommandExecution:
    """Own the context, ports, route, and reply state for one queue invocation."""

    def __init__(self, runtime: TelegramCommandRuntime, context_id: str,
                 operation: Any, lease: Any, command: Any, retry_governor: Any):
        self.runtime = runtime
        self.context_id, self.operation = context_id, operation
        self.lease, self.command, self.retry_governor = lease, command, retry_governor
        self.operation_id = _value(operation, "operation_id")
        self.owner, self.fence = _value(lease, "owner"), _value(lease, "fence")

    async def run(self) -> QueueOperationState:
        """Validate the lease, compose ports, route, then settle the selected flow."""
        if not self._valid_invocation():
            return QueueOperationState.FAILED
        if await self._load_context_and_replay_save():
            return self.replay_save_result
        self._compose_ports()
        await self._select_route()
        await self._prepare_search()
        self._compose_replies()
        early = await self._early_governed_failure()
        if early is not None:
            return early
        if self._is_global_flow():
            result = await self._run_global()
            settled = await self._settle_governed_failure()
            return settled if settled is not None else result
        return await self._run_document()

    def _valid_invocation(self) -> bool:
        """Reject malformed operation identity and lease metadata before IO."""
        return bool(
            isinstance(self.context_id, str) and self.context_id
            and _value(self.lease, "operation_id") == self.operation_id
            and isinstance(self.operation_id, str)
            and isinstance(self.owner, str) and type(self.fence) is int
        )

    async def _load_context_and_replay_save(self) -> bool:
        """Load the fenced context and replay an already-pinned save operation."""
        runtime = self.runtime
        self.context = await _offload(runtime._load_context, self.context_id, self.operation)
        if self.context is None:
            self.replay_save_result = QueueOperationState.FAILED
            return True
        self.lease_check = lambda: runtime._queue_lease_live(
            self.context_id, self.operation_id, self.owner, self.fence,
        )
        if not await _offload(self.lease_check):
            self.replay_save_result = QueueOperationState.FAILED
            return True
        if self.command is None:
            self.pinned = await _offload(runtime.repository.get_route_control,
                                         self.context_id, self.operation_id)
            retry_work = (
                isinstance(self.pinned, Mapping)
                and self.pinned.get("status") in {"ok", "success"}
                and self.pinned.get("kind") == "retry"
                and type(self.pinned.get("dialog_number")) is int
                and isinstance(_value(self.operation, "retry_request_operation_id"), str)
            )
            if retry_work or (isinstance(self.pinned, Mapping)
                              and self.pinned.get("kind") == "save"):
                self.replay_save_result = await runtime.save_handler.save_draft(
                    self.context_id, self.operation, self.lease,
                    retry_governor=self.retry_governor,
                )
                return True
        return False

    def _compose_ports(self) -> None:
        """Build durable store, ledger, governed model, executor, publisher, and search ports."""
        runtime = self.runtime
        self.store = TelegramRuntimeStore(
            runtime.repository, root_folder_id=runtime.root_folder_id,
            context_id=self.context_id, operation_id=self.operation_id,
        )
        self.ledger = FirestoreAgentLedger(
            runtime.repository.client, context_id=self.context_id,
            operation_id=self.operation_id, owner=self.owner, fence=self.fence,
            clock=runtime.clock,
        )
        self.governed_model = _GovernedModel(
            runtime._model(), self.retry_governor, self.operation_id, self.lease,
            runtime.clock, self.lease_check, payload_codec=self.ledger,
        )
        self.route_model = GeminiCommandRouter(self.governed_model)
        self.executor_context = {
            "context_id": self.context_id, "operation_id": self.operation_id,
            "owner": self.owner, "fence": self.fence, "lease": self.lease,
            "retry_governor": self.retry_governor,
            "last_doc_id": self.context.last_doc_id,
        }
        self.governed_executors: list[_GovernedExecutor] = []
        self.agent = TelegramAgent(
            model=self.governed_model, ledger=self.ledger,
            tool_executor=self.executor_factory(None, self.operation_id, self.executor_context),
        )
        self.publisher = TelegramNotificationPublisher(
            runtime.repository, runtime._client(), clock=runtime.clock,
        )
        self.search = _SearchPort(
            self.store, runtime.clock, governor=self.retry_governor,
            operation_id=self.operation_id, lease=self.lease,
            lease_check=self.lease_check, ledger=self.ledger,
        )

    def executor_factory(self, target_doc_id: str, delivery_id: str, extra: Any) -> Any:
        """Create a scoped registry executor wrapped in durable retry governance."""
        scoped = {**self.executor_context, "target_doc_id": target_doc_id,
                  "clock": self.runtime.clock}
        if isinstance(extra, Mapping):
            scoped.update(extra)
        executor = TelegramRegistryExecutor(
            client=self.runtime.repository.client, context=scoped,
            save_handler=self.runtime.save_handler, execution_context=scoped,
        )
        governed = _GovernedExecutor(
            executor, self.retry_governor, self.operation_id, self.lease,
            self.runtime.clock, self.lease_check, payload_codec=self.ledger,
        )
        self.governed_executors.append(governed)
        return governed

    async def _select_route(self) -> None:
        """Load safe candidates and replay or persist the model-only routing decision."""
        from services.connectors.telegram.sessions import _candidate_cards, _valid_response
        runtime = self.runtime
        self.cards = await _offload(_candidate_cards, self.store)
        self.save_candidates = await _offload(
            runtime._predecessor_save_candidates, self.context_id, self.operation,
        )
        self.recent_search_results = await _offload(
            runtime._predecessor_search_results, self.context_id, self.operation, self.cards,
        )
        self.pinned = await _offload(runtime.repository.get_route_control,
                                     self.context_id, self.operation_id)
        self.global_conversation_id = "telegram-global-" + hashlib.sha256(
            f"{self.context_id}\0{self.operation_id}".encode()).hexdigest()
        self.replay_request = (
            await _offload(self.ledger.load_request, self.global_conversation_id,
                           self.operation_id) if self.command is None else None
        )
        self.raw_decision = self._pinned_decision() if self._has_pinned_decision() else None
        if self.raw_decision is None:
            try:
                self.raw_decision = await _offload(
                    self.route_model.route,
                    text=self.command.text if self.command is not None else "",
                    candidates=self.cards, operation_candidates=self.save_candidates,
                    recent_search_results=self.recent_search_results,
                    context_metadata={
                        "active_conversation_id": self.context.active_conversation_id,
                        "selected_dialog_number": self.context.selected_dialog_number,
                        "last_doc_id": self.context.last_doc_id,
                    },
                ) if self.command is not None else {}
            except Exception:
                self.raw_decision = {}
        self.decision, self.decision_status = _valid_response(self.raw_decision)

    def _has_pinned_decision(self) -> bool:
        return isinstance(self.pinned, Mapping) and self.pinned.get("status") in {
            "ok", "success", "uncertain", "invalid_route", "error",
        }

    def _pinned_decision(self) -> dict[str, Any]:
        return {
            "kind": self.pinned.get("kind"), "target_ref": self.pinned.get("target_doc_id"),
            "target_operation_id": self.pinned.get("target_operation_id"),
            "dialog_number": self.pinned.get("dialog_number"),
            "result_refs": self.pinned.get("result_refs", []),
            "substantive_text": (
                self.command.text if self.command is not None else self.replay_request
            ) if self.pinned.get("kind") == "substantive" else None,
            "tool_choice": self.pinned.get("tool_choice"),
            "certainty": "certain" if self.pinned.get("status") in {"ok", "success"}
            else "uncertain",
        }

    async def _prepare_search(self) -> None:
        """Reuse pinned search results or perform retrieval for a new search route."""
        self.search_refs: list[str] | None = None
        self.search_session_id: str | None = None
        if self.decision is None or self.decision["kind"] != "search":
            return
        if isinstance(self.pinned, Mapping) and isinstance(
                self.pinned.get("search_session_id"), str):
            self.search_refs = list(self.pinned.get("result_refs", []))
            self.search_session_id = self.pinned["search_session_id"]
        elif self.command is not None:
            try:
                self.search_refs = await self.search.search(self.command.text)
            except Exception:
                self.search_refs = []
            self.search_session_id = "search-" + hashlib.sha256(
                f"{self.context_id}\0{self.operation_id}".encode()).hexdigest()

    def _compose_replies(self) -> None:
        """Create the durable chunk publisher and queue-state settlement callback."""
        self.replies = _DurableCommandReplies(
            repository=self.runtime.repository, publisher=self.publisher,
            context_id=self.context_id, operation_id=self.operation_id,
            owner=self.owner, fence=self.fence, clock=self.runtime.clock,
        )

    async def finish_reply(self, body: str, *, kind: str = "final",
                           failure_final: bool = False) -> QueueOperationState:
        """Settle queue state only after every deterministic reply chunk is handled."""
        status = await self.replies.deliver(body, kind=kind)
        if status == "delivered":
            return QueueOperationState.FAILED if failure_final else QueueOperationState.COMPLETED
        if status == "pending":
            return QueueOperationState.RUNNING
        return QueueOperationState.FAILED

    async def _early_governed_failure(self) -> QueueOperationState | None:
        """Stop before routing execution when search or routing has a governed failure."""
        if self.search.failure is not None and self.retry_governor is not None:
            stage_key, terminal = self.search.failure
            if not terminal:
                return QueueOperationState.RUNNING
            message = _terminal_failure_message(
                self.retry_governor, self.operation_id, stage_key, service="search",
                stage="retrieval", dialog_number=self.context.selected_dialog_number,
            )
            return await self.finish_reply(message, kind="error", failure_final=True)
        if self.governed_model.failure is not None and self.retry_governor is not None:
            _failure, terminal = self.governed_model.failure
            if not terminal:
                return QueueOperationState.RUNNING
            message = _terminal_failure_message(
                self.retry_governor, self.operation_id,
                self.governed_model.failure_stage or "routing", service="gemini",
                stage="routing", dialog_number=self.context.selected_dialog_number,
            )
            return await self.finish_reply(message, kind="error", failure_final=True)
        return None

    def _is_global_flow(self) -> bool:
        """Choose global control/chat work while leaving selected-document flow intact."""
        decision, command, context = self.decision, self.command, self.context
        current_doc = bool(
            isinstance(context.active_conversation_id, str)
            and any(item.get("conversation_id") == context.active_conversation_id
                    for item in self.cards if isinstance(item, Mapping))
        )
        global_tool_choice = bool(
            decision and decision.get("kind") == "substantive"
            and decision.get("target_ref") is None
            and decision.get("target_operation_id") is None
            and decision.get("dialog_number") is None
            and decision.get("tool_choice") is not None
            and decision.get("tool_choice", {}).get("arguments", {}).get("doc_id") is None
            and decision.get("tool_choice", {}).get("name") in {
                "query", "get_transcript", "list_recent_transcripts", "list_folders",
                "sync_status", "sync_changes",
            }
        )
        pinned_global = bool(
            command is None and isinstance(self.pinned, Mapping)
            and self.pinned.get("status") in {"ok", "success"}
            and self.pinned.get("kind") in {
                "substantive", "search", "retry", "pure_switch", "save",
            } and not self.pinned.get("target_doc_id")
        )
        pure_switch = bool(
            decision and decision.get("kind") == "pure_switch"
            and decision.get("target_ref") is None
            and decision.get("target_operation_id") is None
            and decision.get("dialog_number") is None
        )
        no_document = bool(
            decision and not current_doc and decision.get("kind") == "substantive"
            and decision.get("target_ref") is None
            and decision.get("target_operation_id") is None
            and decision.get("dialog_number") is None and decision.get("tool_choice") is None
        )
        return bool(pinned_global or (command is not None and (
            decision is None or decision["kind"] in {"save", "retry", "search"}
            or pure_switch or no_document or global_tool_choice
        )))

    async def _settle_governed_failure(self) -> QueueOperationState | None:
        """Publish a safe terminal stage diagnostic or leave retryable work running."""
        failure, stage_key, service = self.governed_model.failure, self.governed_model.failure_stage, "gemini"
        if failure is None:
            failed = next((item for item in self.governed_executors if item.failure is not None), None)
            if failed is not None:
                failure, stage_key, service = failed.failure, failed.failure_stage, "connected_service"
        if failure is None:
            return None
        if not failure[1]:
            return QueueOperationState.RUNNING
        message = _terminal_failure_message(
            self.retry_governor, self.operation_id, stage_key or "tool_execution",
            service=service, stage="tool_execution",
            dialog_number=self.context.selected_dialog_number,
        )
        return await self.finish_reply(message, kind="error", failure_final=True)

    async def _run_global(self) -> QueueOperationState:
        """Persist global route control and dispatch its specific command handler."""
        if not await self._persist_global_route():
            return QueueOperationState.FAILED
        if self.decision is None or self.decision_status != "ok":
            return await self.finish_reply(
                "I could not determine what you want me to do. Please clarify.", kind="error",
            )
        cleared = await _offload(
            self.runtime.repository.clear_command_candidate, self.context_id,
            self.operation_id, owner=self.owner, fence=self.fence,
            now=self.runtime.clock(),
        )
        if cleared is False:
            return QueueOperationState.RUNNING
        handlers = {
            "save": self._handle_global_save,
            "retry": self._handle_global_retry,
            "pure_switch": self._handle_global_pure_switch,
            "search": self._handle_global_search,
        }
        handler = handlers.get(self.decision["kind"], self._handle_global_chat)
        return await handler()

    async def _persist_global_route(self) -> bool:
        """Durably pin the selected route and its replay checkpoint before side effects."""
        decision = self.decision
        record = {"status": "ok" if decision is not None else "uncertain",
                  "kind": decision.get("kind") if decision else None}
        if decision is not None:
            for key in ("target_ref", "target_operation_id", "dialog_number", "result_refs", "tool_choice"):
                if decision.get(key) is not None and key not in {"target_ref", "result_refs"}:
                    record[key] = decision[key]
            if decision.get("kind") == "search":
                record["result_refs"] = list(self.search_refs or [])
                record["search_session_id"] = self.search_session_id
            prompt = self.command.text if self.command is not None else self.replay_request
            if decision.get("kind") in {"substantive", "search"} and isinstance(prompt, str):
                await _offload(self.ledger.save_request, self.global_conversation_id,
                               self.operation_id, prompt)
        try:
            return bool(await _offload(
                self.runtime.repository.persist_route, self.operation_id, record,
                context_id=self.context_id, owner=self.owner, fence=self.fence,
                now=self.runtime.clock(),
            ))
        except Exception:
            return False

    async def _handle_global_save(self) -> QueueOperationState:
        """Confirm the captured draft binding before invoking the required save service."""
        draft_id = await _offload(self.runtime._captured_draft_id,
                                  self.context_id, self.operation)
        if draft_id is None:
            return await self.finish_reply("Please forward messages first, then ask me to save them.")
        from services.connectors.telegram.sessions import confirm_save_route
        confirmed = await _offload(
            confirm_save_route, self.runtime.repository, context_id=self.context_id,
            operation_id=self.operation_id, owner=self.owner, fence=self.fence,
            now=self.runtime.clock(), route_result={"kind": "save"},
        )
        if not confirmed:
            return await self.finish_reply(
                "I could not confirm the save request. Please try again.", kind="error",
            )
        state = await self.runtime.save_handler.save_draft(
            self.context_id, self.operation, self.lease,
            retry_governor=self.retry_governor,
        )
        if state != QueueOperationState.COMPLETED:
            return state
        return await self.finish_reply("The save completed.")

    async def _handle_global_retry(self) -> QueueOperationState:
        """Queue the retry for the route's validated dialog number."""
        from services.connectors.telegram.retry import TelegramRetryService
        number = self.decision["dialog_number"]
        outcome = await _offload(
            TelegramRetryService(self.runtime.repository).retry_failed_draft_for_command,
            self.context_id, number, request_operation_id=self.operation_id,
            now=self.runtime.clock(),
        )
        if not getattr(outcome, "accepted", False):
            return QueueOperationState.FAILED
        return await self.finish_reply("Retry queued for dialog #{}.".format(number))

    async def _handle_global_pure_switch(self) -> QueueOperationState:
        """Explain that a global dialog switch needs a valid existing number."""
        return await self.finish_reply(
            "I could not find that dialog. Please choose an existing dialog number.",
            kind="error",
        )

    async def _handle_global_search(self) -> QueueOperationState:
        """Create the durable search session and render its visible result summaries."""
        query = self.command.text if self.command is not None else self.replay_request or ""
        session_id = await _offload(
            self.store.create_search_session, context_id=self.context_id, query=query,
            result_refs=self.search_refs or [],
            error="search unavailable" if self.search.failure else None,
        )
        turns = await _offload(self.runtime.repository.list_turns, self.context_id, session_id)
        rows = [row for turn in turns if turn.get("role") == "tool"
                for part in turn.get("parts", [])
                if isinstance(part, Mapping) and part.get("type") == "search_result"
                for row in part.get("results", [])]
        visible = [self._format_search_row(row) for row in rows if isinstance(row, Mapping)]
        visible = [item for item in visible if item]
        answer = (
            "Search is unavailable right now. Please try again." if self.search.failure else
            "No matching documents were found." if not visible else
            "Search results: " + "; ".join(visible)
        )
        return await self.finish_reply(
            answer, kind="error" if self.search.failure else "final",
            failure_final=bool(self.search.failure),
        )

    @staticmethod
    def _format_search_row(row: Mapping[str, Any]) -> str:
        """Format one result card using only its safe title, date, and folder metadata."""
        title = row.get("title", row.get("doc_id", ""))
        details = [str(title)] if title else []
        date_value = row.get("date", row.get("dialog_date", row.get("created_at")))
        folder = row.get("folder_path", row.get("folder"))
        if isinstance(date_value, str) and date_value:
            details.append(date_value)
        if isinstance(folder, list):
            folder = " / ".join(str(part) for part in folder)
        if isinstance(folder, str) and folder:
            details.append(folder)
        return " | ".join(details)

    async def _handle_global_chat(self) -> QueueOperationState:
        """Run global agent chat or a checkpointed route-selected tool call."""
        prompt = self.command.text if self.command is not None else self.replay_request
        if not isinstance(prompt, str) or not prompt:
            return QueueOperationState.FAILED
        executor = self.executor_factory(None, self.operation_id, self.executor_context)
        choice = self.decision.get("tool_choice")
        if choice is not None:
            return await self._handle_global_tool(executor, choice)
        try:
            record = await _offload(
                self.agent.run, conversation_id=self.global_conversation_id,
                delivery_id=self.operation_id,
                history=[{"role": "user", "parts": [{"text": prompt}]}],
                executor_factory=lambda _doc, _delivery, _ctx: executor,
                execution_context=self.executor_context,
            )
        except Exception:
            record = {"status": "error"}
        if self.governed_model.failure is not None or any(
                item.failure is not None for item in self.governed_executors):
            return QueueOperationState.RUNNING
        answer = record.get("answer") if record.get("status") == "ok" else None
        answer = answer if isinstance(answer, str) and answer.strip() else (
            "I could not complete that request. Please clarify."
        )
        return await self.finish_reply(answer, kind="final" if record.get("status") == "ok" else "error")

    async def _handle_global_tool(self, executor: Any, choice: Mapping[str, Any]) -> QueueOperationState:
        """Execute a preselected global tool exactly once using the existing turn checkpoint."""
        name, arguments = choice["name"], choice["arguments"]
        checkpoint = await _offload(self.ledger.load_turn,
                                    self.global_conversation_id, self.operation_id)
        saved = checkpoint.get("tool_results", {}) if checkpoint else {}
        if isinstance(saved, Mapping) and "0" in saved:
            result = saved["0"]
        else:
            await _offload(
                self.ledger.save_turn, self.global_conversation_id, self.operation_id,
                model_response={"text": "", "parts": []},
                tool_calls=[{"id": f"route-{self.operation_id}", "name": name,
                             "arguments": arguments, "ordinal": 0, "round": 0}],
            )
            try:
                result = await _offload(
                    executor.execute, name=name, arguments=arguments, call_ordinal=0,
                )
            except Exception:
                settled = await self._settle_governed_failure()
                if settled is not None:
                    return settled
                raise
            await _offload(self.ledger.save_tool_result, self.global_conversation_id,
                           self.operation_id, 0, result)
        return await self.finish_reply(json.dumps(result, ensure_ascii=False, default=str))

    async def _run_document(self) -> QueueOperationState:
        """Run the established Telegram document processor and settle its answer once."""
        processor = TelegramCommandProcessor(
            context_loader=lambda _context_id, _operation: self.context,
            router=self._route_callback, repository=self.runtime.repository,
            agent=_CapturingAgent(self.agent), notifications=self.publisher,
            executor_factory=self.executor_factory,
            selection_publisher=self._selection_callback, clock=self.runtime.clock,
        )
        self.capturing_agent = processor.agent
        result = await processor(
            self.context_id, self.operation, self.lease, command=self.command,
            retry_governor=self.retry_governor,
        )
        failure_result = await self._settle_governed_failure()
        if failure_result is not None:
            return failure_result
        return await self._settle_document_reply(result)

    async def _route_callback(self, **kwargs: Any) -> Any:
        """Pass the pinned model decision through the established session resolver."""
        from services.connectors.telegram.sessions import route_message
        return await route_message(
            text=kwargs["text"], context=kwargs["context"],
            model=_PrecomputedRouteModel(self.raw_decision), store=self.store,
            search=self.search, notices=_NoopNotice(), actions=_NoopAction(),
            operation_id=self.operation_id, owner=self.owner, fence=self.fence,
            now=self.runtime.clock(), save_candidates=self.save_candidates,
        )

    async def _selection_callback(self, pinned_operation_id: str, *,
                                  conversation_id: str, dialog_number: int) -> bool:
        """Publish a document selection under this command's queue fence."""
        return await _offload(
            self.store.publish_selection, pinned_operation_id,
            conversation_id=conversation_id, dialog_number=dialog_number,
            owner=self.owner, fence=self.fence, now=self.runtime.clock(),
        )

    async def _settle_document_reply(self, result: QueueOperationState) -> QueueOperationState:
        """Publish captured model text or recover the checkpointed final answer."""
        if self.capturing_agent.answer is not None and result == QueueOperationState.COMPLETED:
            delivery = await self.finish_reply(self.capturing_agent.answer)
            if delivery != QueueOperationState.COMPLETED:
                return delivery
        elif result == QueueOperationState.COMPLETED:
            answer = await self._load_document_answer()
            if isinstance(answer, str) and answer.strip():
                return await self.finish_reply(answer)
        elif result == QueueOperationState.FAILED and self.decision is not None:
            message = (
                "I could not find that dialog. Please choose an existing dialog number."
                if self.decision.get("kind") == "pure_switch" else
                "I could not complete that request. Please clarify."
            )
            return await self.finish_reply(message, kind="error", failure_final=True)
        return result

    async def _load_document_answer(self) -> Any:
        """Read only this operation's saved answer using its immutable route target."""
        route = await _offload(self.runtime.repository.get_route_control,
                               self.context_id, self.operation_id)
        if (not isinstance(route, Mapping) or route.get("status") not in {"ok", "success"}
                or route.get("kind") != "substantive"):
            return None
        conversation_id = route.get("conversation_id")
        target = route.get("target_doc_id")
        operation_ref = (self.runtime.repository.client.collection("telegram_sessions")
                         .document(self.context_id).collection("operations")
                         .document(self.operation_id))
        operation_snapshot = await _offload(operation_ref.get)
        queue_item = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        if (queue_item.get("operation_id") != self.operation_id
                or queue_item.get("context_id", self.context_id) != self.context_id
                or queue_item.get("state") != "running"
                or not isinstance(conversation_id, str) or not conversation_id
                or not isinstance(target, str) or not target):
            return None
        saved = await _offload(self.ledger.load_turn, conversation_id, self.operation_id)
        response = saved.get("model_response") if isinstance(saved, Mapping) else None
        rounds = response.get("rounds", []) if isinstance(response, Mapping) and response.get(
            "_telegram_tool_loop_rounds") == 1 else [response]
        final = (rounds[-1].get("response") if rounds and isinstance(rounds[-1], Mapping)
                 and "response" in rounds[-1] else (rounds[-1] if rounds else None))
        return _value(final, "text")
