"""Durable Telegram notification delivery over the notification ledger."""

from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from services.connectors.telegram.client import TelegramAPIError
from services.connectors.telegram.repository import TelegramRepository


_SYNC_POOL = ThreadPoolExecutor(thread_name_prefix="telegram-notification")
_SYNC_WAIT_SECONDS = 15.0
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


async def _offload_repository_call(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    future = _SYNC_POOL.submit(function, *args, **kwargs)
    deadline = time.monotonic() + _SYNC_WAIT_SECONDS
    while not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError("Telegram notification storage timed out")
        await asyncio.sleep(0.01)
    return future.result()


@dataclass(frozen=True, slots=True)
class _VerifiedTelegramLifecycle:
    """Server-derived terminal status; callers cannot supply a proof-shaped mapping."""

    dialog_number: int
    operation_id: str
    service: str
    stage: str
    reason: str
    folder_path: str | None = None
    completion_confirmed: bool = False
    classification_outcome: str | None = None
    code: str | None = None
    category: str | None = None
    occurred_at: datetime | None = None
    failures: int | None = None
    sanitized_message: str | None = None


def build_lifecycle_body(*, lifecycle: _VerifiedTelegramLifecycle) -> str:
    """Format a previously verified server result without accepting task content."""
    if not isinstance(lifecycle, _VerifiedTelegramLifecycle):
        raise TypeError("lifecycle must be a verified server result")
    if type(lifecycle.dialog_number) is not int or lifecycle.dialog_number < 1:
        raise ValueError("dialog_number must be positive")
    if not isinstance(lifecycle.operation_id, str) or not _SAFE_ID.fullmatch(lifecycle.operation_id):
        raise ValueError("operation_id is invalid")
    for name in ("service", "stage", "reason"):
        if not isinstance(getattr(lifecycle, name), str) or not _SAFE_CODE.fullmatch(getattr(lifecycle, name)):
            raise ValueError(f"{name} is invalid")
    prefix = f"Dialog #{lifecycle.dialog_number}"
    if lifecycle.completion_confirmed:
        path = " ".join(lifecycle.folder_path.split()) if lifecycle.folder_path else ""
        outcome = lifecycle.classification_outcome
        if outcome not in {None, "ambiguous", "unassigned"}:
            outcome = None
        note = {
            "ambiguous": " The closest match was selected.",
            "unassigned": " The folder could not be determined confidently; saved to _unassigned.",
        }.get(outcome, "")
        if path and len(path) <= 512 and not any(ord(char) < 32 for char in path):
            body = f"{prefix}: Saved and indexed. Folder: {path}."
        else:
            body = f"{prefix}: Saved and indexed."
        return f"{body}{note}"[:4096]
    if lifecycle.reason in {"in_progress", "index_pending"}:
        return f"{prefix}: Processing or indexing is still in progress."
    from services.connectors.telegram.errors import format_terminal_error

    return format_terminal_error(
        dialog_number=lifecycle.dialog_number,
        operation_id=lifecycle.operation_id,
        service=lifecycle.service,
        stage=lifecycle.stage,
        code=lifecycle.code or lifecycle.reason,
        category=lifecycle.category,
        sanitized_message=lifecycle.sanitized_message,
        occurred_at=lifecycle.occurred_at,
        failures=lifecycle.failures,
    )


class TelegramNotificationPublisher:
    """Publish a ledger notification without accepting delivery content from a task."""

    def __init__(
        self, repository: TelegramRepository, client: Any, *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.client = client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def publish(self, notification_id: str) -> str:
        """Deliver one notification attempt and return its durable disposition."""
        now = self.clock()
        attempt_id = "tgnotify-" + uuid4().hex
        reservation = await _offload_repository_call(
            self.repository.reserve_notification_attempt,
            notification_id, attempt_id=attempt_id, now=now,
        )
        if reservation is None:
            return "not_found"
        if not reservation.get("reserved"):
            state = reservation.get("state")
            return state if state in {"delivered", "failed", "delivery_unknown", "superseded"} else "busy"

        chat_id = reservation["chat_id"]
        thread_id = reservation.get("thread_id")
        body = reservation["body"]
        revision = reservation["attempt_revision"]
        try:
            if reservation["is_send"]:
                response = await self.client.send_message(
                    chat_id, body, message_thread_id=thread_id,
                )
                message_id = response.get("message_id") if isinstance(response, dict) else None
                if type(message_id) is not int or message_id < 1:
                    outcome, message_id = "unknown", None
                else:
                    outcome = "sent"
            else:
                await self.client.edit_message_text(chat_id, reservation["message_id"], body)
                outcome, message_id = "edited", None
        except TelegramAPIError as exc:
            outcome = "unknown" if exc.category in {
                "timeout", "transport", "unavailable", "invalid_response",
            } else "rejected"
            message_id = None
        except Exception:
            # A timeout or connection drop does not prove whether Telegram accepted the send.
            outcome, message_id = "unknown", None

        settled = await _offload_repository_call(
            self.repository.settle_notification_attempt,
            notification_id, attempt_id=attempt_id, revision=revision,
            outcome=outcome, message_id=message_id, now=self.clock(),
        )
        if settled is None:
            return "not_found"
        if not settled.get("settled"):
            return "stale"
        return str(settled["state"])

    async def send_switch_notice(
        self, *, context_id: str, operation_id: str, notification_id: str,
        target_doc_id: str, dialog_number: int,
    ) -> str:
        """Create and deliver the server-authored notice for a pinned target."""
        if not isinstance(target_doc_id, str) or not target_doc_id:
            raise ValueError("target_doc_id is invalid")
        if type(dialog_number) is not int or dialog_number < 1:
            raise ValueError("dialog_number must be positive")
        await _offload_repository_call(
            self.repository.create_notification,
            notification_id,
            context_id=context_id,
            kind="switch_notice",
            body=f"Starting to process dialog #{dialog_number}.",
            dialog_number=dialog_number,
            operation_id=operation_id,
            now=self.clock(),
        )
        return await self._deliver_switch_notice(notification_id)

    async def _deliver_switch_notice(self, notification_id: str) -> str:
        """Retry one durable notice within the ledger's bounded send budget.

        Reuse the notification ID so the ledger enforces its maximum of three
        physical sends. After a retry or a busy reservation, re-read the row
        and wait until its persisted retry time or active attempt lease expires.
        Poll at most five times, returning pending if the row remains active.
        """
        for _ in range(5):
            disposition = await self.publish(notification_id)
            if disposition != "retry":
                if disposition != "busy":
                    return "confirmed" if disposition == "delivered" else disposition
            row = await _offload_repository_call(self.repository.get_notification, notification_id)
            if row is None:
                return "not_found"
            if row.get("state") in {"delivered", "failed", "delivery_unknown"}:
                return "confirmed" if row["state"] == "delivered" else str(row["state"])
            retry_at = row.get("retry_not_before")
            expiry = row.get("attempt_expires_at")
            wait_until = retry_at if isinstance(retry_at, datetime) else expiry
            if not isinstance(wait_until, datetime):
                return "pending"
            delay = max(0.0, (wait_until - self.clock()).total_seconds())
            if delay:
                await asyncio.sleep(delay)
        return "pending"

__all__ = ["TelegramNotificationPublisher", "build_lifecycle_body"]
