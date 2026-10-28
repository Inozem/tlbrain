"""Server-side retry entry point for explicitly numbered Telegram dialogs."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class TelegramRetryService:
    """Retry a failed draft selected by an already-resolved dialog number."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def retry_failed_draft(
        self,
        context_id: str,
        dialog_number: int,
        *,
        update_id: int,
        now: datetime,
    ) -> Any:
        """Create a fresh queued operation for the exact failed dialog."""
        return self.repository.retry_failed_draft(
            context_id, dialog_number, update_id=update_id, now=now,
        )

    def retry_failed_draft_for_command(
        self,
        context_id: str,
        dialog_number: int,
        *,
        request_operation_id: str,
        now: datetime,
    ) -> Any:
        """Queue one idempotent retry for an already accepted Telegram command."""
        return self.repository.retry_failed_draft_for_command(
            context_id, dialog_number,
            request_operation_id=request_operation_id, now=now,
        )


__all__ = ["TelegramRetryService"]
