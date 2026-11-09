"""Firestore-safe envelopes for durable Telegram save stage checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STAGE_PAYLOAD_REF_FIELD = "telegram_save_stage_payload_ref"
_LEGACY_STAGE_PAYLOAD_REF_FIELD = "__telegram_save_stage_payload__"


def stage_payload_ref(value: Any) -> Any | None:
    """Extract a stage payload reference from current or legacy checkpoint envelopes."""
    if not isinstance(value, Mapping):
        return None
    for field in (STAGE_PAYLOAD_REF_FIELD, _LEGACY_STAGE_PAYLOAD_REF_FIELD):
        if set(value) == {field}:
            return value[field]
    return None
