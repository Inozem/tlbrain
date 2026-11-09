"""Small, side-effect-free parsing boundary for Telegram control text."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True, slots=True)
class TelegramCommand:
    kind: str
    text: str
    action_metadata: dict[str, object] | None = None


def parse_command(value: str) -> TelegramCommand:
    """Return the complete normalized command candidate without inspecting draft content."""
    text = unicodedata.normalize("NFKC", value).strip()
    return TelegramCommand("freeform", text)


__all__ = ["TelegramCommand", "parse_command"]
