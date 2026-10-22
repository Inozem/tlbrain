"""Public Firestore repository facade for Telegram connector persistence."""

from __future__ import annotations

from typing import Any

from core.integrations.firestore.client import get_client
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    NOTIFICATION_ATTEMPT_LEASE,
    NOTIFICATION_KINDS,
    NOTIFICATION_MAX_SENDS,
    SESSIONS,
    TelegramIntakeResult,
)
from services.connectors.telegram.repository._commands import _TelegramCommandMixin
from services.connectors.telegram.repository._drafts import _TelegramDraftMixin
from services.connectors.telegram.repository._intake import _TelegramIntakeMixin
from services.connectors.telegram.repository._notifications import _TelegramNotificationMixin


class TelegramRepository(
    _TelegramIntakeMixin,
    _TelegramCommandMixin,
    _TelegramNotificationMixin,
    _TelegramDraftMixin,
):
    """Own Firestore transactions for durable Telegram intake state."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client if client is not None else get_client()


__all__ = [
    "NOTIFICATIONS",
    "NOTIFICATION_ATTEMPT_LEASE",
    "NOTIFICATION_KINDS",
    "NOTIFICATION_MAX_SENDS",
    "SESSIONS",
    "TelegramIntakeResult",
    "TelegramRepository",
]
