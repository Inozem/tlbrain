"""Small, side-effect-free parsing boundary for Telegram control text."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata


MAX_COMMAND_CHARS = 1000


@dataclass(frozen=True, slots=True)
class TelegramCommand:
    kind: str
    text: str


def parse_command(value: str) -> TelegramCommand:
    """Return a bounded command candidate without inspecting draft content."""
    text = unicodedata.normalize("NFKC", value).strip()
    return TelegramCommand("freeform", text[:MAX_COMMAND_CHARS])


__all__ = ["MAX_COMMAND_CHARS", "TelegramCommand", "parse_command"]
