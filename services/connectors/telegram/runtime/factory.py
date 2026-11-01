"""Lazy construction of the real Telegram command and save runtime."""

from __future__ import annotations

from typing import Any


def build_default_queue_worker(client: Any) -> Any:
    """Compose the production queue processor without performing provider calls."""
    from services.connectors.telegram.queue import TelegramQueueWorker
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    repository = TelegramRepository(client=client)
    save_handler = TelegramSaveRuntime(repository=repository)
    processor = TelegramCommandRuntime(
        save_handler=save_handler, repository=repository,
    )
    return TelegramQueueWorker(client, processor, same_head_delay_seconds=5.0)
