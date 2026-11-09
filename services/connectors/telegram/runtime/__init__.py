"""Concrete production adapters for Telegram command processing.

The save handler is deliberately a required constructor dependency. It must
implement ``async save_draft(context_id, operation, lease, *,
retry_governor=None) -> QueueOperationState`` and synchronous
``save_dialog(arguments, *, context_id, operation_id, call_ordinal, lease,
retry_governor=None) -> JSON-safe result``.
"""

from .runtime import TelegramCommandRuntime

__all__ = ["TelegramCommandRuntime"]
