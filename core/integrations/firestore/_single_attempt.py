"""Trusted Firestore read markers for Telegram's durable single-attempt path."""

from __future__ import annotations

from typing import Any, Callable

from core.utils.retry import ProviderBoundaryError


def single_attempt_read(callback: Callable[[], Any]) -> Any:
    """Mark a Firestore read error without exposing SDK response contents."""
    try:
        return callback()
    except Exception as error:
        raise ProviderBoundaryError("firestore", error) from error
