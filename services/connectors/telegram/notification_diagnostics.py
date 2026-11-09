"""Bounded, content-free diagnostics for Telegram notification delivery."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_SAFE_EXCEPTION_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TELEGRAM_CATEGORIES = frozenset({
    "timeout", "transport", "unavailable", "invalid_response", "not_modified",
    "rate_limited", "provider_error", "authorization", "permission_denied",
    "not_found", "rejected", "unknown",
})
_logger = logging.getLogger(__name__)


def _exception_type(error: BaseException | None) -> str | None:
    if error is None:
        return None
    name = type(error).__name__
    return name if _SAFE_EXCEPTION_TYPE.fullmatch(name) else "unknown"


def _emit(
    *, notification_id: Any, phase: str, category: str, action: str = "unknown",
    attempt_id: Any = None, operation_id: Any = None, revision: Any = None,
    known_message: bool = False, message_id: Any = None,
    error: BaseException | None = None,
) -> None:
    fields: dict[str, Any] = {
        "event": "telegram_notification_delivery", "phase": phase,
        "category": category, "action": action if action in {"send", "edit"} else "unknown",
        "known_message": known_message is True,
    }
    for key, value in (
        ("notification_id", notification_id), ("attempt_id", attempt_id),
        ("operation_id", operation_id),
    ):
        if isinstance(value, str) and _SAFE_ID.fullmatch(value):
            fields[key] = value
    if type(revision) is int and revision > 0:
        fields["reserved_revision"] = revision
    if type(message_id) is int and message_id > 0:
        fields["message_id"] = message_id
    exception_name = _exception_type(error)
    if exception_name is not None:
        fields["exception_type"] = exception_name
    _logger.warning("Telegram notification delivery diagnostic", extra=fields)


async def storage_call(
    offload: Callable[..., Any], function: Callable[..., Any], /, *args: Any,
    diagnostic_notification_id: Any, phase: str,
    diagnostic_attempt_id: Any = None, diagnostic_operation_id: Any = None,
    diagnostic_action: str = "unknown", diagnostic_revision: Any = None,
    diagnostic_known_message: bool = False, diagnostic_message_id: Any = None,
    **kwargs: Any,
) -> Any:
    """Run one repository call and report only its fixed storage phase on failure."""
    try:
        return await offload(function, *args, **kwargs)
    except Exception as error:
        _emit(
            notification_id=diagnostic_notification_id, attempt_id=diagnostic_attempt_id,
            operation_id=diagnostic_operation_id, action=diagnostic_action,
            revision=diagnostic_revision, known_message=diagnostic_known_message,
            message_id=diagnostic_message_id, phase=phase,
            category="storage_error", error=error,
        )
        raise


def log_provider_failure(
    *, notification_id: Any, attempt_id: Any, operation_id: Any,
    action: str, revision: Any, known_message: bool,
    category: Any, error: BaseException,
) -> None:
    safe_category = category if isinstance(category, str) and category in _TELEGRAM_CATEGORIES else "unknown"
    if safe_category == "authorization":
        safe_category = "permission_denied"
    _emit(
        notification_id=notification_id, attempt_id=attempt_id,
        operation_id=operation_id, action=action, revision=revision,
        known_message=known_message, phase=action, category=safe_category,
        error=error,
    )


def log_unexpected_failure(
    *, notification_id: Any, attempt_id: Any, operation_id: Any,
    action: str, revision: Any, known_message: bool,
    error: BaseException,
) -> None:
    _emit(
        notification_id=notification_id, attempt_id=attempt_id,
        operation_id=operation_id, action=action, revision=revision,
        known_message=known_message, phase=action, category="unexpected_error",
        error=error,
    )


def log_missing_message_id(
    *, notification_id: Any, attempt_id: Any, operation_id: Any,
    action: str, revision: Any,
) -> None:
    _emit(
        notification_id=notification_id, attempt_id=attempt_id,
        operation_id=operation_id, action=action, revision=revision,
        known_message=False, phase="response_validation", category="missing_message_id",
    )
