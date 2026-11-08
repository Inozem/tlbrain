"""Acceptance-relative status checks for Telegram save operations."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
    task_payload,
)
from services.connectors.telegram.annotation_notifications import (
    is_annotation_apply_operation,
    resolve_annotation_terminal_notification,
)


STATUS_CHECK_OFFSETS = (60, 120, 300)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _read_operation(client: Any, operation_id: str) -> dict[str, Any] | None:
    snapshot = client.collection("dialog_operations").document(operation_id).get()
    return snapshot.to_dict() if snapshot.exists else None


def _read_queue_operation(client: Any, context_id: str, operation_id: str) -> dict[str, Any] | None:
    snapshot = (client.collection("telegram_sessions").document(context_id)
                .collection("operations").document(operation_id).get())
    return snapshot.to_dict() if snapshot.exists else None


def _read_operation_dialog_number(client: Any, context_id: str, operation_id: str) -> int | None:
    """Resolve the dialog captured by this operation's accepted command and draft."""
    context_ref = client.collection("telegram_sessions").document(context_id)
    queue_operation = context_ref.collection("operations").document(operation_id).get()
    input_ref = (queue_operation.to_dict() or {}).get("input_ref") if queue_operation.exists else None
    prefix = f"telegram_sessions/{context_id}/commands/"
    if not isinstance(input_ref, str) or not input_ref.startswith(prefix):
        return None
    command_id = input_ref[len(prefix):]
    if not command_id or "/" in command_id:
        return None
    command = context_ref.collection("commands").document(command_id).get()
    draft_id = (command.to_dict() or {}).get("draft_id") if command.exists else None
    if not isinstance(draft_id, str) or not draft_id:
        return None
    draft = context_ref.collection("drafts").document(draft_id).get()
    dialog_number = (draft.to_dict() or {}).get("dialog_number") if draft.exists else None
    return dialog_number if type(dialog_number) is int and dialog_number > 0 else None


class TelegramStatusChecks:
    """Schedule and handle status reads without entering the execution queue."""

    def __init__(
        self,
        client: Any,
        *,
        status_reader: Callable[[str], dict[str, Any] | None] | None = None,
        notification_publisher: Any,
        clock: Callable[[], datetime] | None = None,
        terminal_notification_updater: Callable[[Any], str | None] | None = None,
        managed_runtime: bool = False,
    ) -> None:
        self.client = client
        self.status_reader = status_reader or (lambda operation_id: _read_operation(client, operation_id))
        self.notification_publisher = notification_publisher
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.managed_runtime = managed_runtime is True
        if terminal_notification_updater is None:
            from services.connectors.telegram.app.terminal_notifications import update_terminal_notification
            terminal_notification_updater = update_terminal_notification
        self.terminal_notification_updater = terminal_notification_updater

    def schedule_checks(
        self, context_id: str, operation_id: str, accepted_at: datetime,
    ) -> list[Any]:
        accepted_at = _aware(accepted_at, "accepted_at")
        now = _aware(self.clock(), "clock")
        identity = hashlib.sha256(f"{context_id}:{operation_id}".encode()).hexdigest()
        records = []
        for offset in STATUS_CHECK_OFFSETS:
            records.append(create_outbox_record(
                self.client,
                f"tgstatus-{identity}-{offset}",
                target_kind=OutboxTargetKind.TELEGRAM_STATUS_CHECK,
                generation=1,
                schedule_at=accepted_at + timedelta(seconds=offset),
                now=now,
                context_id=context_id,
                operation_id=operation_id,
            ))
        return records

    async def handle_task(self, payload: dict[str, Any]) -> str:
        record = await self._load_status_check_record(payload)
        if record is None:
            return "no_op"

        operation_id = record.operation_id
        if not operation_id:
            raise ValueError("status task has no operation")
        shared_operation, persisted, queue_operation = await self._read_operation_status(
            record, operation_id,
        )
        if persisted.get("final_delivery_state") == "delivered":
            return "no_op"

        status = await self._read_current_status(operation_id, queue_operation)
        state = self._status_state(persisted, status)
        if state in {"completed", "failed"}:
            if self.managed_runtime and is_annotation_apply_operation(queue_operation):
                annotation_notice = await asyncio.to_thread(
                    resolve_annotation_terminal_notification,
                    self.client, record, queue_operation, state,
                    now=_aware(self.clock(), "clock"),
                )
                delivery_state = await self.notification_publisher.publish(annotation_notice)
                if delivery_state not in {"delivered", "failed", "delivery_unknown"}:
                    raise RuntimeError("Telegram annotation notification delivery is pending")
                return str(state)
            if self.managed_runtime:
                linked_id = await asyncio.to_thread(
                    self._linked_terminal_notification_id, record, persisted, status, queue_operation, state,
                )
                if linked_id:
                    delivery_state = await self.notification_publisher.publish(linked_id)
                    if delivery_state not in {"delivered", "failed", "delivery_unknown"}:
                        raise RuntimeError("Telegram status notification delivery is pending")
                    return str(state)
                if await asyncio.to_thread(self._managed_save_is_processing, record, queue_operation):
                    progress_id = await asyncio.to_thread(
                        self._ensure_progress_notification, record, {"state": "running"},
                    )
                    delivery_state = await self.notification_publisher.publish(progress_id)
                    if delivery_state not in {"delivered", "failed", "delivery_unknown"}:
                        raise RuntimeError("Telegram status notification delivery is pending")
                    return str(state)
            return await self._handle_terminal_status(
                record, shared_operation, persisted, status, state,
            )
        return await self._handle_progress_status(record, persisted, status, state)

    async def _load_status_check_record(self, payload: dict[str, Any]) -> Any | None:
        if not isinstance(payload, dict) or payload.get("target_kind") != OutboxTargetKind.TELEGRAM_STATUS_CHECK.value:
            raise ValueError("invalid Telegram status task")
        outbox_id = payload.get("outbox_id")
        if not isinstance(outbox_id, str):
            raise ValueError("invalid Telegram status task")
        record = await asyncio.to_thread(get_outbox_record, self.client, outbox_id)
        if record is None or record.target_kind is not OutboxTargetKind.TELEGRAM_STATUS_CHECK:
            return None
        if payload != task_payload(record):
            raise ValueError("invalid Telegram status task")
        return record

    async def _read_operation_status(
        self, record: Any, operation_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
        operation_read = asyncio.to_thread(_read_operation, self.client, operation_id)
        queue_read = (
            asyncio.to_thread(_read_queue_operation, self.client, record.context_id, operation_id)
            if record.context_id else asyncio.sleep(0, result=None)
        )
        persisted, queue_operation = await asyncio.gather(operation_read, queue_read)
        if persisted is None and queue_operation is None:
            raise RuntimeError("Telegram operation status is unavailable")
        shared_operation = persisted
        persisted = persisted or queue_operation or {}
        return shared_operation, persisted, queue_operation

    async def _read_current_status(
        self, operation_id: str, queue_operation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Read shared status, falling back to the FIFO reservation before shared state exists."""
        if inspect.iscoroutinefunction(self.status_reader):
            status = await self.status_reader(operation_id)
        else:
            status = await asyncio.to_thread(self.status_reader, operation_id)
            if inspect.isawaitable(status):
                status = await status
        if status is None:
            if queue_operation is None:
                raise RuntimeError("Telegram operation status is unavailable")
            status = queue_operation
        return status

    @staticmethod
    def _status_state(persisted: dict[str, Any], status: dict[str, Any]) -> str:
        persisted_state = persisted.get("state")
        return (
            persisted_state if persisted_state in {"completed", "failed"}
            else status.get("state", persisted_state or "unknown")
        )

    async def _handle_terminal_status(
        self,
        record: Any,
        shared_operation: dict[str, Any] | None,
        persisted: dict[str, Any],
        status: dict[str, Any],
        state: str,
    ) -> str:
        if shared_operation is None and await asyncio.to_thread(
            self._terminal_notification_is_delivered, record, persisted, status,
        ):
            return "no_op"
        final_notification_id = await asyncio.to_thread(self.terminal_notification_updater, record)
        if not final_notification_id:
            stable_id = persisted.get("notification_id") or status.get("notification_id")
            if stable_id:
                from services.connectors.telegram.repository import TelegramRepository
                notification = await asyncio.to_thread(
                    TelegramRepository(self.client).get_notification, stable_id,
                )
                if notification and notification.get("kind") in {"final", "error"}:
                    final_notification_id = stable_id
            if not final_notification_id:
                final_notification_id = await asyncio.to_thread(
                    self._ensure_terminal_notification, record, state,
                )
        if final_notification_id:
            delivery_state = await self.notification_publisher.publish(final_notification_id)
            if delivery_state not in {"delivered", "failed", "delivery_unknown"}:
                raise RuntimeError("Telegram status notification delivery is pending")
        return str(state)

    async def _handle_progress_status(
        self,
        record: Any,
        persisted: dict[str, Any],
        status: dict[str, Any],
        state: str,
    ) -> str:
        notification_id = persisted.get("notification_id") or status.get("notification_id")
        if not notification_id:
            notification_id = await asyncio.to_thread(self._ensure_progress_notification, record, status)
        delivery_state = await self.notification_publisher.publish(notification_id)
        if delivery_state not in {"delivered", "failed", "delivery_unknown"}:
            raise RuntimeError("Telegram status notification delivery is pending")
        return str(state)

    def _terminal_notification_is_delivered(
        self, record: Any, persisted: dict[str, Any], status: dict[str, Any],
    ) -> bool:
        """Recognize a delivered terminal ledger row when shared operation state is absent."""
        from services.connectors.telegram.repository import TelegramRepository

        if not record.operation_id:
            return False
        candidate_ids = [
            persisted.get("notification_id"),
            status.get("notification_id"),
            self._fallback_terminal_notification_id(record.operation_id),
        ]
        repository = TelegramRepository(self.client)
        seen: set[str] = set()
        for notification_id in candidate_ids:
            if not isinstance(notification_id, str) or notification_id in seen:
                continue
            seen.add(notification_id)
            notification = repository.get_notification(notification_id)
            if (
                notification
                and (notification.get("kind") in {"final", "error"}
                     or (notification.get("receipt_card") is True
                         and notification.get("receipt_terminal") is True
                         and notification.get("save_operation_id") == record.operation_id))
                and notification.get("state") == "delivered"
            ):
                return True
        return False

    def _linked_terminal_notification_id(
        self, record: Any, persisted: dict[str, Any], status: dict[str, Any],
        queue_operation: dict[str, Any] | None, state: str,
    ) -> str | None:
        """Find only a canonical terminal row bound to this operation and context."""
        from services.connectors.telegram.repository import TelegramRepository

        if not record.context_id or not record.operation_id:
            return None
        expected_kind = "final" if state == "completed" else "error"
        digest = hashlib.sha256(record.operation_id.encode()).hexdigest()
        candidates = [
            (queue_operation or {}).get("notification_id"),
            persisted.get("notification_id"), status.get("notification_id"),
            f"tg{expected_kind}-{digest}",
        ]
        repository = TelegramRepository(self.client)
        seen: set[str] = set()
        for notification_id in candidates:
            if not isinstance(notification_id, str) or notification_id in seen:
                continue
            seen.add(notification_id)
            row = repository.get_notification(notification_id)
            if (row and row.get("notification_id") == notification_id
                    and row.get("context_id") == record.context_id
                    and row.get("operation_id") == record.operation_id
                    and (row.get("kind") == expected_kind
                         or (row.get("receipt_card") is True
                             and row.get("receipt_terminal") is True
                             and row.get("save_operation_id") == record.operation_id
                             and row.get("receipt_terminal_kind") == expected_kind))):
                return notification_id
        return None

    def _managed_save_dialog_number(
        self, record: Any, queue_operation: dict[str, Any] | None,
    ) -> int | None:
        """Return a number only for this operation's confirmed sealed source work."""
        if (not isinstance(record.context_id, str) or not _SAFE_ID.fullmatch(record.context_id)
                or not isinstance(record.operation_id, str) or not _SAFE_ID.fullmatch(record.operation_id)
                or not isinstance(queue_operation, dict)):
            return None
        route = queue_operation.get("route_control")
        if (
            queue_operation.get("operation_id") != record.operation_id
            or queue_operation.get("state") != "running"
            or queue_operation.get("save_operation_id") != record.operation_id
            or not isinstance(route, dict)
            or route.get("status") not in {"ok", "success"}
            or route.get("kind") not in {"save", "retry"}
        ):
            return None
        context_ref = self.client.collection("telegram_sessions").document(record.context_id)
        if route.get("kind") == "retry" and not self._retry_source_binding_is_valid(
            record, queue_operation, route, context_ref,
        ):
            return None
        draft_id = queue_operation.get("target_draft_id") or route.get("target_draft_id")
        if not isinstance(draft_id, str) or not _SAFE_ID.fullmatch(draft_id):
            return None
        return self._sealed_operation_dialog_number(
            record, queue_operation, route, context_ref, draft_id,
        )

    def _retry_source_binding_is_valid(
        self, record: Any, queue_operation: dict[str, Any],
        route: dict[str, Any], context_ref: Any,
    ) -> bool:
        """Validate a retry's accepted request link or its request-less source command."""
        request_id = queue_operation.get("retry_request_operation_id")
        if request_id is None:
            return self._requestless_retry_command_is_valid(
                record, queue_operation, route, context_ref,
            )
        if (not isinstance(request_id, str) or not _SAFE_ID.fullmatch(request_id)
                or route.get("retry_request_operation_id") != request_id):
            return False
        request_snapshot = context_ref.collection("operations").document(request_id).get()
        request_operation = request_snapshot.to_dict() or {} if request_snapshot.exists else {}
        request_route = request_operation.get("route_control")
        if (
            request_operation.get("operation_id") != request_id
            or request_operation.get("context_id", record.context_id) != record.context_id
            or not isinstance(request_route, dict)
            or request_route.get("status") not in {"ok", "success"}
            or request_route.get("kind") != "retry"
            or request_route.get("dialog_number") != route.get("dialog_number")
        ):
            return False
        return self._retry_command_receipt_is_valid(
            context_ref, request_operation, request_id, kind="freeform",
        )

    def _requestless_retry_command_is_valid(
        self, record: Any, queue_operation: dict[str, Any],
        route: dict[str, Any], context_ref: Any,
    ) -> bool:
        """Bind a legacy direct retry to its own accepted retry command and receipt."""
        draft_id = queue_operation.get("target_draft_id")
        if not isinstance(draft_id, str) or not _SAFE_ID.fullmatch(draft_id):
            return False
        if not self._retry_command_receipt_is_valid(
            context_ref, queue_operation, record.operation_id,
            kind="retry", draft_id=draft_id,
            dialog_number=route.get("dialog_number"),
        ):
            return False
        command_id = queue_operation["input_ref"].rsplit("/", 1)[1]
        command_snapshot = context_ref.collection("commands").document(command_id).get()
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        draft_snapshot = context_ref.collection("drafts").document(draft_id).get()
        draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
        return (
            command.get("save_intent_id") == draft.get("immutable_save_intent_id")
            and command.get("manifest_digest") == draft.get("manifest_digest")
        )

    def _retry_command_receipt_is_valid(
        self, context_ref: Any, operation: dict[str, Any], operation_id: str,
        *, kind: str, draft_id: str | None = None, dialog_number: int | None = None,
    ) -> bool:
        """Verify the command and accepted update receipt behind a retry operation."""
        input_ref = operation.get("input_ref")
        expected_prefix = f"telegram_sessions/{context_ref.id}/commands/"
        command_id = (
            input_ref[len(expected_prefix):]
            if isinstance(input_ref, str) and input_ref.startswith(expected_prefix)
            else ""
        )
        if not command_id.isdigit():
            return False
        command_snapshot = context_ref.collection("commands").document(command_id).get()
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        update_id = command.get("update_id")
        receipt_snapshot = (
            context_ref.collection("updates").document(str(update_id)).get()
            if type(update_id) is int and update_id > 0 else None
        )
        receipt = receipt_snapshot.to_dict() or {} if receipt_snapshot and receipt_snapshot.exists else {}
        if (command.get("context_id") != context_ref.id
                or command.get("operation_id") != operation_id
                or command.get("kind") != kind
                or type(update_id) is not int
                or receipt.get("accepted") is not True
                or receipt.get("context_id") != context_ref.id
                or receipt.get("operation_id") != operation_id):
            return False
        if draft_id is not None and command.get("draft_id") != draft_id:
            return False
        return dialog_number is None or command.get("dialog_number") == dialog_number

    def _sealed_operation_dialog_number(
        self, record: Any, queue_operation: dict[str, Any],
        route: dict[str, Any], context_ref: Any, draft_id: str,
    ) -> int | None:
        """Return a route-matched number only after validating its current manifest."""
        draft_snapshot = context_ref.collection("drafts").document(draft_id).get()
        draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
        intent_id = draft.get("immutable_save_intent_id")
        manifest_path = draft.get("manifest_ref")
        expected_manifest = (
            f"telegram_sessions/{record.context_id}/drafts/{draft_id}/save_manifests/"
            f"{record.operation_id}"
        )
        if (draft.get("draft_id") != draft_id or draft.get("sealed_at") is None
                or manifest_path != expected_manifest
                or not isinstance(draft.get("manifest_digest"), str)
                or not draft.get("manifest_digest")
                or draft.get("active_save_operation_id") != record.operation_id
                or not isinstance(intent_id, str) or not _SAFE_ID.fullmatch(intent_id)):
            return None
        manifest_snapshot = context_ref.collection("drafts").document(draft_id).collection(
            "save_manifests").document(record.operation_id).get()
        manifest = manifest_snapshot.to_dict() or {} if manifest_snapshot.exists else {}
        number = draft.get("dialog_number")
        if (manifest.get("draft_id") != draft_id
                or manifest.get("save_operation_id") != record.operation_id
                or manifest.get("manifest_digest") != draft.get("manifest_digest")
                or type(number) is not int or number < 1
                or route.get("dialog_number") != number):
            return None
        return number

    def _managed_save_is_processing(
        self, record: Any, queue_operation: dict[str, Any] | None,
    ) -> bool:
        """Defer terminal fallback for a queue-bound sealed save/retry during cleanup."""
        if (not isinstance(record.context_id, str) or not _SAFE_ID.fullmatch(record.context_id)
                or not isinstance(record.operation_id, str) or not _SAFE_ID.fullmatch(record.operation_id)
                or not isinstance(queue_operation, dict)):
            return False
        route = queue_operation.get("route_control")
        if (
            queue_operation.get("operation_id") != record.operation_id
            or queue_operation.get("state") != "running"
            or queue_operation.get("save_operation_id") != record.operation_id
            or not isinstance(route, dict)
            or route.get("status") not in {"ok", "success"}
            or route.get("kind") not in {"save", "retry"}
        ):
            return False
        draft_id = queue_operation.get("target_draft_id") or route.get("target_draft_id")
        if not isinstance(draft_id, str) or not _SAFE_ID.fullmatch(draft_id):
            return False
        context_ref = self.client.collection("telegram_sessions").document(record.context_id)
        snapshot = context_ref.collection("drafts").document(draft_id).get()
        draft = snapshot.to_dict() or {} if snapshot.exists else {}
        intent_id = draft.get("immutable_save_intent_id")
        return (
            draft.get("draft_id") == draft_id
            and draft.get("sealed_at") is not None
            and isinstance(draft.get("manifest_ref"), str) and bool(draft.get("manifest_ref"))
            and isinstance(draft.get("manifest_digest"), str) and bool(draft.get("manifest_digest"))
            and draft.get("active_save_operation_id") == record.operation_id
            and isinstance(intent_id, str) and bool(_SAFE_ID.fullmatch(intent_id))
        )

    @staticmethod
    def _fallback_terminal_notification_id(operation_id: str) -> str:
        return "tgterminal-" + hashlib.sha256(operation_id.encode()).hexdigest()

    def _ensure_terminal_notification(self, record: Any, state: str) -> str:
        """Create one deterministic terminal ledger row when no lifecycle row exists."""
        from services.connectors.telegram.repository import TelegramRepository

        if state not in {"completed", "failed"}:
            raise ValueError("terminal notification state is invalid")
        if not record.operation_id or not record.context_id:
            raise RuntimeError("Telegram operation notification is unavailable")

        notification_id = self._fallback_terminal_notification_id(record.operation_id)
        repository = TelegramRepository(self.client)
        existing = repository.get_notification(notification_id)
        expected_kind = "final" if state == "completed" else "error"
        if existing is not None:
            if existing.get("kind") in {"final", "error"}:
                return notification_id
            raise RuntimeError("terminal notification ID is already in use")

        dialog_number = _read_operation_dialog_number(
            self.client, record.context_id, record.operation_id,
        )
        if state == "completed":
            body = "Operation completed."
        else:
            from services.connectors.telegram.errors import format_terminal_error
            from core.integrations.firestore.operations import get_operation
            from services.connectors.telegram.app.terminal_notifications import _terminal_error_details

            operation = get_operation(self.client, record.operation_id)
            details = _terminal_error_details(self.client, record, operation)
            if details:
                body = format_terminal_error(
                    dialog_number=dialog_number,
                    operation_id=record.operation_id,
                    service=details.get("service") or "telegram",
                    stage=details.get("stage") or getattr(operation, "stage", None) or "processing",
                    code=details.get("code"), category=details.get("category"),
                    sanitized_message=details.get("sanitized_message"),
                    occurred_at=details.get("occurred_at"), failures=details.get("failures"),
                )
            else:
                body = format_terminal_error(
                    dialog_number=dialog_number, operation_id=record.operation_id,
                    service="telegram", stage="processing", code="operation_failed",
                    category="unknown", sanitized_message=None, occurred_at=None, failures=None,
                )
        repository.create_notification(
            notification_id, context_id=record.context_id, kind=expected_kind,
            body=body, now=_aware(self.clock(), "clock"), dialog_number=dialog_number,
            operation_id=record.operation_id,
        )
        return notification_id

    def _ensure_progress_notification(
        self, record: Any, status: dict[str, Any], notification_id: str | None = None,
    ) -> str:
        """Create or revise one deterministic progress ledger row for this operation."""
        from services.connectors.telegram.repository import SESSIONS
        from services.connectors.telegram.repository import TelegramRepository

        operation_id = record.operation_id
        context_id = record.context_id
        if not operation_id or not context_id:
            raise RuntimeError("Telegram operation notification is unavailable")
        notification_id = notification_id or "tgprogress-" + hashlib.sha256(operation_id.encode()).hexdigest()
        state = status.get("state") or "unknown"
        stage = status.get("stage")
        description = "Processing is still in progress"
        if state == "completed":
            description = "Operation completed"
        elif state == "failed":
            description = "Processing failed"
        elif state in {"pending", "queued"}:
            description = "Command received; processing is queued"
        elif stage in {"transcription", "preparation", "classification", "indexing"}:
            description = f"Processing is still in progress: {stage}"

        try:
            offset = int(record.outbox_id.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            raise ValueError("status check offset is invalid") from None
        elapsed = {
            60: "after 1 minute",
            120: "after 2 minutes",
            300: "after 5 minutes",
        }.get(offset)
        if elapsed is None:
            raise ValueError("status check offset is invalid")
        description = f"{description}; {elapsed}"

        context_ref = self.client.collection(SESSIONS).document(context_id)
        queue_operation = context_ref.collection("operations").document(operation_id).get()
        queue_data = queue_operation.to_dict() or {} if queue_operation.exists else None
        dialog_number = self._managed_save_dialog_number(record, queue_data)
        body_prefix = f"Dialog #{dialog_number}: " if dialog_number is not None else ""
        body = f"{body_prefix}{description}."
        repository = TelegramRepository(self.client)
        existing = repository.get_notification(notification_id)
        if existing is None:
            repository.create_notification(
                notification_id, context_id=context_id, kind="progress", body=body,
                now=_aware(self.clock(), "clock"), dialog_number=dialog_number,
                operation_id=operation_id,
            )
        elif existing.get("kind") == "progress" and existing.get("body") != body:
            repository.revise_notification(
                notification_id, body=body, now=_aware(self.clock(), "clock"),
            )
        return notification_id


__all__ = ["STATUS_CHECK_OFFSETS", "TelegramStatusChecks"]
