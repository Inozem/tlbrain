"""Small, side-effect-free parsing boundary for Telegram control text."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


MAX_COMMAND_CHARS = 1000
SAVE_ALIASES = frozenset({"сохрани", "сохранить", "save"})


@dataclass(frozen=True, slots=True)
class TelegramCommand:
    kind: str
    text: str
    is_save: bool


def parse_command(value: str) -> TelegramCommand:
    """Return a bounded command candidate without inspecting draft content."""
    text = unicodedata.normalize("NFKC", value).strip()
    bounded = text[:MAX_COMMAND_CHARS]
    token = re.sub(r"[.!?,;:]+$", "", text.casefold()).strip()
    is_save = (
        token in SAVE_ALIASES
        or bool(re.match(r"^(?:сохрани|сохранить)(?:\b|$)", token))
    )
    return TelegramCommand("save" if is_save else "freeform", bounded, is_save)


__all__ = ["MAX_COMMAND_CHARS", "TelegramCommand", "parse_command"]
