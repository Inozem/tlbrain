"""Validated registry execution boundary for Telegram agent tools."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.operations import (
    claim_operation, complete_operation, create_operation, get_operation,
    release_operation, validate_operation_lease,
)
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.operations.models import OperationOrigin, OperationResult, OperationState
from core.operations import registry
from core.operations.write import WriteContext
from services.connectors.telegram.tools import invoke_tool


def direct_add_dialog_operation_id(parent_operation_id: str, ordinal: int,
                                  arguments: Mapping[str, Any]) -> str:
    """Derive one replay-stable direct-save child from its validated call identity."""
    validated = registry.validate_arguments("add_dialog", arguments)
    serialized = json.dumps(validated, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(
        f"{parent_operation_id}\0{int(ordinal)}\0{digest}".encode("utf-8")
    ).hexdigest()
    return f"telegram-tool-{identity}"


class TelegramRegistryExecutor:
    """Validate against the shared 13-operation registry before dispatch."""

    def __init__(self, *, client: Any, context: Mapping[str, Any],
                 save_handler: Any, execution_context: Mapping[str, Any]):
        self.client, self.context, self.save_handler = client, context, save_handler
        self.execution_context = dict(execution_context)

    def execute(self, *, name: str, arguments: Any, call_ordinal: int) -> Any:
        validated = registry.validate_arguments(name, arguments)
        pinned = self.execution_context.get("target_doc_id")
        arg_doc = validated.get("doc_id")
        if arg_doc is not None and pinned is not None and arg_doc != pinned:
            raise ValueError("tool target differs from pinned document")
        if name == "get_transcript":
            if arg_doc is None and isinstance(pinned, str) and pinned:
                validated["doc_id"] = pinned
            elif arg_doc is None:
                raise ValueError("transcript reads require a pinned document")
        self._assert_queue_lease_live()
        if name == "add_dialog":
            handler = getattr(self.save_handler, "save_dialog", None)
            if not callable(handler):
                raise TypeError("save_handler must implement save_dialog")
            return handler(validated, context_id=self.execution_context["context_id"],
                           operation_id=self.execution_context["operation_id"],
                           call_ordinal=call_ordinal, lease=self.execution_context["lease"],
                           retry_governor=self.execution_context.get("retry_governor"))
        op = registry.OPERATIONS[name]
        if not op.write:
            return invoke_tool(name, validated, context=self.context)
        return self._execute_write(name, validated, call_ordinal)

    def reconcile(self, *, name: str, arguments: Any, call_ordinal: int) -> Any:
        """Read durable evidence for an interrupted child without invoking a tool."""
        validated = registry.validate_arguments(name, arguments)
        if not registry.OPERATIONS[name].write:
            return None
        parent_id = self.execution_context["operation_id"]
        context_id = self.execution_context["context_id"]
        operation_id = (
            direct_add_dialog_operation_id(parent_id, call_ordinal, validated)
            if name == "add_dialog" else "telegram-tool-" + hashlib.sha256(
                f"{parent_id}\0{int(call_ordinal)}".encode("utf-8")
            ).hexdigest()
        )
        checkpoint_ref = (self.client.collection("telegram_sessions").document(context_id)
                          .collection("operations").document(parent_id)
                          .collection("agent_checkpoints").document(f"tool-{int(call_ordinal)}"))
        from core.integrations.firestore.operations import get_operation
        prior = get_operation(self.client, operation_id)
        checkpoint = checkpoint_ref.get().to_dict() or {}
        if (prior is not None and prior.state == OperationState.COMPLETED
                and "result" in checkpoint):
            return deepcopy(checkpoint["result"])
        return None

    def _assert_queue_lease_live(self) -> None:
        context_id = self.execution_context.get("context_id")
        operation_id = self.execution_context.get("operation_id")
        lease = self.execution_context.get("lease")
        owner = getattr(lease, "owner", None)
        fence = getattr(lease, "fence", None)
        if not isinstance(context_id, str) or not isinstance(operation_id, str):
            raise PermissionError("Telegram tool has no queue binding")
        if isinstance(lease, Mapping):
            owner, fence = lease.get("owner"), lease.get("fence")
        if not isinstance(owner, str) or type(fence) is not int:
            raise PermissionError("Telegram tool has no live queue lease")
        context_ref = self.client.collection("telegram_sessions").document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        context_snapshot, operation_snapshot = context_ref.get(), operation_ref.get()
        context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        now = (self.execution_context.get("clock") or
               (lambda: datetime.now(timezone.utc)))()
        if (not operation_snapshot.exists or operation.get("state") != "running"
                or operation.get("operation_id") != operation_id
                or not validate_context_queue_lease(
                    context, operation_id, owner=owner, fence=fence, now=now,
                    expiry=context.get("lock_expires_at"))):
            raise PermissionError("Telegram tool queue lease is stale")

    def _execute_write(self, name: str, arguments: dict[str, Any], call_ordinal: int) -> Any:
        """Claim a stable global operation and use canonical WriteContext/effects."""
        parent_id = self.execution_context["operation_id"]
        context_id = self.execution_context["context_id"]
        ordinal = int(call_ordinal)
        lease = self.execution_context["lease"]
        now_fn = self.execution_context.get("clock") or (lambda: datetime.now(timezone.utc))
        now = now_fn()
        identity = json.dumps({"name": name, "arguments": arguments}, sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        operation_id = "telegram-tool-" + hashlib.sha256(
            f"{parent_id}\0{ordinal}".encode("utf-8")
        ).hexdigest()
        input_ref = (f"telegram_sessions/{context_id}/operations/{parent_id}"
                     f"/agent_checkpoints/tool-{ordinal}")
        create_operation(
            self.client, operation_id=operation_id, origin=OperationOrigin.TELEGRAM,
            caller_binding=f"telegram:{context_id}:{parent_id}",
            immutable_input_digest=digest, immutable_input_ref=input_ref,
            origin_operation_ref=f"telegram_sessions/{context_id}/operations/{parent_id}",
            now=now,
        )
        prior = get_operation(self.client, operation_id)
        checkpoint_ref = (self.client.collection("telegram_sessions").document(context_id)
                          .collection("operations").document(parent_id)
                          .collection("agent_checkpoints").document(f"tool-{ordinal}"))
        checkpoint = checkpoint_ref.get().to_dict() or {}
        if prior is not None and prior.state == OperationState.COMPLETED and "result" in checkpoint:
            return deepcopy(checkpoint["result"])
        owner = f"telegram-agent-{parent_id[:32]}-{ordinal}"
        claimed = claim_operation(self.client, operation_id, owner=owner, now=now,
                                  lease_seconds=120)
        if claimed is None:
            prior = get_operation(self.client, operation_id)
            checkpoint = checkpoint_ref.get().to_dict() or {}
            if prior is not None and prior.state == OperationState.COMPLETED and "result" in checkpoint:
                return deepcopy(checkpoint["result"])
            raise RuntimeError("global tool operation lease unavailable")
        self._assert_queue_lease_live()
        self._save_input_checkpoint(
            checkpoint_ref, operation_id, owner=owner, fence=claimed.fence,
            name=name, arguments=arguments, digest=digest, now_fn=now_fn,
        )
        if "result" in checkpoint:
            result_value = deepcopy(checkpoint["result"])
            metadata = result_value if isinstance(result_value, Mapping) else {}
            operation_result = OperationResult(
                document_id=metadata.get("doc_id") if isinstance(metadata.get("doc_id"), str) else None,
                title=metadata.get("title") if isinstance(metadata.get("title"), str) else None,
                resource_url=metadata.get("url") if isinstance(metadata.get("url"), str) else None,
            )
            if not complete_operation(self.client, operation_id, owner=owner,
                                      fence=claimed.fence, result=operation_result, now=now_fn()):
                raise PermissionError("global tool operation completion was fenced")
            return result_value
        write_context = WriteContext(
            operation_id=operation_id, effect_id=stable_effect_id(parent_id, ordinal),
            owner=owner, fence=claimed.fence, client=self.client, clock=now_fn,
        )
        context = {**self.context, "last_doc_id": self.execution_context.get("last_doc_id")}
        self._assert_queue_lease_live()
        try:
            result = invoke_tool(name, arguments, context=context,
                                 execution_context=write_context)
        except Exception:
            release_operation(self.client, operation_id, owner=owner,
                              fence=claimed.fence, now=now_fn())
            raise
        safe_result = _json_safe(result)
        ref = self.client.collection("dialog_operations").document(operation_id)
        transaction = self.client.transaction()
        @firestore.transactional
        def save_result(tx: Any) -> None:
            current = ref.get(transaction=tx)
            data = current.to_dict() or {} if current.exists else {}
            lease_expiry = data.get("lease_expires_at")
            if (data.get("state") != OperationState.RUNNING.value
                    or data.get("lease_owner") != owner
                    or data.get("fence") != claimed.fence
                    or not isinstance(lease_expiry, datetime) or lease_expiry <= now_fn()):
                raise PermissionError("global tool operation lease is stale")
            old = data.get("telegram_tool_result")
            if old is not None and old != safe_result:
                raise ValueError("global tool result changed on replay")
            tx.set(checkpoint_ref, {"operation_id": operation_id, "result": safe_result}, merge=True)
            tx.set(ref, {"telegram_tool_result": safe_result}, merge=True)
        save_result(transaction)
        metadata = safe_result if isinstance(safe_result, Mapping) else {}
        operation_result = OperationResult(
            document_id=metadata.get("doc_id") if isinstance(metadata.get("doc_id"), str) else None,
            title=metadata.get("title") if isinstance(metadata.get("title"), str) else None,
            resource_url=metadata.get("url") if isinstance(metadata.get("url"), str) else None,
        )
        if not complete_operation(self.client, operation_id, owner=owner,
                                  fence=claimed.fence, result=operation_result, now=now_fn()):
            raise PermissionError("global tool operation completion was fenced")
        return safe_result

    def _save_input_checkpoint(self, checkpoint_ref: Any, operation_id: str, *,
                               owner: str, fence: int, name: str,
                               arguments: Mapping[str, Any], digest: str,
                               now_fn: Any) -> None:
        """Bind the exact validated arguments before the provider can mutate data."""
        context_id = self.execution_context["context_id"]
        parent_id = self.execution_context["operation_id"]
        lease = self.execution_context["lease"]
        queue_owner = lease.get("owner") if isinstance(lease, Mapping) else getattr(lease, "owner", None)
        queue_fence = lease.get("fence") if isinstance(lease, Mapping) else getattr(lease, "fence", None)
        context_ref = self.client.collection("telegram_sessions").document(context_id)
        parent_ref = context_ref.collection("operations").document(parent_id)
        global_ref = self.client.collection("dialog_operations").document(operation_id)
        tx = self.client.transaction()
        @firestore.transactional
        def commit(transaction: Any) -> None:
            context_snapshot = context_ref.get(transaction=transaction)
            parent_snapshot = parent_ref.get(transaction=transaction)
            global_snapshot = global_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            parent = parent_snapshot.to_dict() or {} if parent_snapshot.exists else {}
            global_data = global_snapshot.to_dict() or {} if global_snapshot.exists else {}
            now = now_fn()
            if (not context_snapshot.exists or not parent_snapshot.exists
                    or parent.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, parent_id, owner=queue_owner, fence=queue_fence,
                        now=now, expiry=context.get("lock_expires_at"))
                    or global_data.get("state") != OperationState.RUNNING.value
                    or global_data.get("lease_owner") != owner
                    or global_data.get("fence") != fence
                    or not isinstance(global_data.get("lease_expires_at"), datetime)
                    or global_data["lease_expires_at"] <= now):
                raise PermissionError("Telegram or global tool lease is stale")
            checkpoint = checkpoint_ref.get(transaction=transaction)
            row = checkpoint.to_dict() or {} if checkpoint.exists else {}
            input_record = {"name": name, "arguments": deepcopy(dict(arguments)),
                            "digest": digest}
            if row.get("validated_input") not in (None, input_record):
                raise ValueError("global tool input changed on replay")
            transaction.set(checkpoint_ref, {
                "operation_id": operation_id, "validated_input": input_record,
            }, merge=True)
        commit(tx)


def stable_effect_id(operation_id: str, call_ordinal: int) -> str:
    digest = hashlib.sha256(f"{operation_id}\0{call_ordinal}".encode()).hexdigest()
    return f"telegram-effect-{digest}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    method = getattr(value, "to_json_dict", None)
    if callable(method):
        return _json_safe(method())
    method = getattr(value, "model_dump", None)
    if callable(method):
        try:
            return _json_safe(method(mode="json", exclude_none=False))
        except TypeError:
            return _json_safe(method())
    return str(value)
