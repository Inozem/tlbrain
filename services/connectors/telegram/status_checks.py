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

    def _managed_save_is_processing(
        self, record: Any, queue_operation: dict[str, Any] | None,
    ) -> bool:
        """Defer fallback only for a queue-bound, sealed save/retry still cleaning up."""
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
        snapshot = (self.client.collection("telegram_sessions").document(record.context_id)
                    .collection("drafts").document(draft_id).get())
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
        input_ref = (queue_operation.to_dict() or {}).get("input_ref") if queue_operation.exists else None
        expected_prefix = f"{SESSIONS}/{context_id}/commands/"
        command_id = (
            input_ref[len(expected_prefix):]
            if isinstance(input_ref, str) and input_ref.startswith(expected_prefix)
            else ""
        )
        command = (
            context_ref.collection("commands").document(command_id).get()
            if command_id and "/" not in command_id else None
        )
        draft_id = (command.to_dict() or {}).get("draft_id") if command and command.exists else None
        draft_snapshot = context_ref.collection("drafts").document(draft_id).get() if draft_id else None
        draft = draft_snapshot.to_dict() or {} if draft_snapshot and draft_snapshot.exists else {}
        dialog_number = draft.get("dialog_number")
        if type(dialog_number) is not int or dialog_number < 1:
            context = context_ref.get().to_dict() or {}
            selected_number = context.get("selected_dialog_number")
            dialog_number = selected_number if type(selected_number) is int and selected_number > 0 else None
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
