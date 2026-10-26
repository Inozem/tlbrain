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
from services.connectors.telegram.repository._conversations import _TelegramConversationMixin
from services.connectors.telegram.repository._drafts import _TelegramDraftMixin
from services.connectors.telegram.repository._intake import _TelegramIntakeMixin
from services.connectors.telegram.repository._notifications import _TelegramNotificationMixin
from services.connectors.telegram.repository._retry import _TelegramRetryMixin
from services.connectors.telegram.repository._save_commands import _TelegramSaveCommandMixin


class TelegramRepository(
    _TelegramIntakeMixin,
    _TelegramCommandMixin,
    _TelegramNotificationMixin,
    _TelegramDraftMixin,
    _TelegramConversationMixin,
    _TelegramSaveCommandMixin,
    _TelegramRetryMixin,
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
