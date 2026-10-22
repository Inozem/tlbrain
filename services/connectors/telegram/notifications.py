"""Durable Telegram notification delivery over the notification ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from services.connectors.telegram.client import TelegramAPIError
from services.connectors.telegram.repository import TelegramRepository


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
        reservation = self.repository.reserve_notification_attempt(
            notification_id, attempt_id=attempt_id, now=now,
        )
        if reservation is None:
            return "not_found"
        if not reservation.get("reserved"):
            state = reservation.get("state")
            return state if state in {"delivered", "failed", "delivery_unknown"} else "busy"

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

        settled = self.repository.settle_notification_attempt(
            notification_id, attempt_id=attempt_id, revision=revision,
            outcome=outcome, message_id=message_id, now=self.clock(),
        )
        if settled is None:
            return "not_found"
        if not settled.get("settled"):
            return "stale"
        return str(settled["state"])


__all__ = ["TelegramNotificationPublisher"]
