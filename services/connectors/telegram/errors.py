"""Safe formatting for persisted operation errors shown in Telegram."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SAFE_OPERATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_SAFE_STAGE_FAILURE = re.compile(r"^Stage execution failed \([A-Za-z][A-Za-z0-9_]{0,63}\)$")

# These are trusted, explicit provider classifications. Codes and exception text
# never participate in this decision.
_CONFIRMED_PROVIDER_CATEGORIES = {"billing", "quota"}


def _identifier(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else fallback


def _reason(category: str, sanitized_message: Any) -> str:
    if category in _CONFIRMED_PROVIDER_CATEGORIES:
        return (
            "The provider confirmed a billing issue."
            if category == "billing"
            else "The provider confirmed a quota limit."
        )
    if category == "rate_limited":
        return "The provider is temporarily limiting requests."
    dependency_reasons = {
        "The confirmed prerequisite did not complete successfully.": (
            "The prerequisite did not complete successfully. Retry the save, then submit the command again."
        ),
        "The command's confirmed prerequisite is invalid.": (
            "The prerequisite could not be verified. Retry the save, then submit the command again."
        ),
        "The command's confirmed prerequisite is unavailable.": (
            "The prerequisite is unavailable. Retry the save, then submit the command again."
        ),
        "The completed prerequisite has no saved result for this command.": (
            "The saved result is unavailable. Retry the save, then submit the command again."
        ),
        "The completed prerequisite's saved result is unavailable.": (
            "The saved result is unavailable. Retry the save, then submit the command again."
        ),
        "The completed prerequisite's saved result is inconsistent.": (
            "The saved result could not be verified. Retry the save, then submit the command again."
        ),
        "The completed prerequisite has an invalid saved result.": (
            "The saved result could not be verified. Retry the save, then submit the command again."
        ),
    }
    if category == "dependency" and isinstance(sanitized_message, str):
        return dependency_reasons.get(sanitized_message, "The prerequisite could not be verified.")
    known_reasons = {
        "Transcription outcome was not checkpointed": "Transcription could not be checkpointed.",
        "Telegram stage outcome was not durably recorded": "The stage result could not be recorded.",
        "Stage execution failed": "The stage could not be completed.",
    }
    if isinstance(sanitized_message, str):
        if sanitized_message in known_reasons:
            return known_reasons[sanitized_message]
        if _SAFE_STAGE_FAILURE.fullmatch(sanitized_message):
            return "The stage could not be completed."
    return "The operation could not be completed."


def format_terminal_error(
    *, dialog_number: int | None, operation_id: Any, service: Any, stage: Any,
    code: Any, category: Any, sanitized_message: Any, occurred_at: Any,
    failures: Any,
) -> str:
    """Build a bounded English terminal error without exposing provider text."""
    prefix = f"Dialog #{dialog_number}: " if type(dialog_number) is int and dialog_number > 0 else ""
    safe_operation_id = (
        operation_id if isinstance(operation_id, str) and _SAFE_OPERATION_ID.fullmatch(operation_id)
        else "unavailable"
    )
    safe_service = _identifier(service, "unknown")
    safe_stage = _identifier(stage, "unknown")
    safe_code = _identifier(code, "operation_failed")
    safe_category = _identifier(category, "unknown")
    timestamp = "unavailable"
    if isinstance(occurred_at, datetime) and occurred_at.tzinfo is not None and occurred_at.utcoffset() is not None:
        timestamp = occurred_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    details = (
        f"Error in {safe_service}/{safe_stage} (code={safe_code}, category={safe_category}); "
        f"reason: {_reason(safe_category, sanitized_message)} "
        f"Occurred: {timestamp}; operation_id={safe_operation_id}."
    )
    exhausted = type(failures) is int and failures >= 3
    if safe_category == "billing":
        details += " Check provider billing."
    if exhausted:
        details += " Automatic retries have stopped; contact an administrator to check the logs."
    return (prefix + details)[:4096]


__all__ = ["format_terminal_error"]
