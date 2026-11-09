"""Pure normalization helpers for Telegram chat and topic context metadata."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_thread_id(value: Any) -> int | None:
    """Return a usable Telegram message thread ID, or the default-context marker."""
    return value if type(value) is int and value > 0 else None


def normalize_topic_name(value: Any) -> str | None:
    """Return a non-empty topic label suitable for an advisory context hint."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def normalize_has_topics_enabled(value: Any) -> bool | None:
    """Normalize optional getMe.has_topics_enabled without inferring capability."""
    return value if type(value) is bool else None


def topic_service_event(message: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """Return the topic event kind and optional normalized name from a message."""
    for event_kind in ("forum_topic_created", "forum_topic_edited"):
        payload = message.get(event_kind)
        if isinstance(payload, Mapping):
            return event_kind, normalize_topic_name(payload.get("name"))
    return None


def message_topic_name(message: Mapping[str, Any]) -> str | None:
    """Normalize a topic name carried alongside an ordinary Telegram message."""
    payload = message.get("forum_topic_created")
    return normalize_topic_name(payload.get("name")) if isinstance(payload, Mapping) else None


__all__ = [
    "normalize_has_topics_enabled",
    "message_topic_name",
    "normalize_thread_id",
    "normalize_topic_name",
    "topic_service_event",
]
