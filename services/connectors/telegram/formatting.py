"""Presentation-only helpers for durable Telegram rich Markdown replies."""

from __future__ import annotations

from collections.abc import Iterable
import re


_CROSS_CHUNK_MARKDOWN = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|\*\*[^*\n]*\*\*|__[^_\n]*__|"
    r"~~[^~\n]*~~|\|\|[^|\n]*\|\||==[^=\n]*==|"
    r"\[[^\]\n]+\]\([^)\n]+\)|(?<!\w)\*[^*\n]+\*(?!\*)|"
    r"(?<![\w])_[^_\n]+_(?!\w)"
)
_RAW_HTML_OR_MEDIA = re.compile(r"</?[A-Za-z][^>]*>|!\[[^\]]*\]\(")


def delivery_presentation(
    presentation: str | None, *, is_send: bool,
    reply_markup: object | None,
) -> str | None:
    """Choose one supported pre-I/O representation for a ledger revision.

    Rich edits support inline markup. Rich sends do not, so a send carrying a
    keyboard keeps the exact body and keyboard through ordinary sendMessage.
    """
    if presentation not in (None, "rich_markdown"):
        raise ValueError("notification presentation is invalid")
    if is_send and presentation == "rich_markdown" and reply_markup is not None:
        return None
    return presentation


def rich_markdown_safe_chunk_indices(chunks: Iterable[tuple[int, str]]) -> set[int]:
    """Identify only chunks crossed by Markdown constructs or raw HTML/media.

    Durable chunk membership is fixed at 4000 characters. This checks complete
    constructs across the concatenated original text, so list syntax, quote
    markers, parentheses, and literal underscores stay eligible for Rich
    Markdown. Constructs crossing a chunk boundary, plus HTML/media syntax,
    get deterministic plain-text delivery for the affected chunks.
    """
    ordered = sorted(chunks)
    if not ordered:
        return set()
    text = "".join(chunk for _index, chunk in ordered)
    boundaries: list[tuple[int, int, int]] = []
    offset = 0
    for index, chunk in ordered:
        boundaries.append((offset, offset + len(chunk), index))
        offset += len(chunk)

    unsafe: set[int] = set()
    for match in _RAW_HTML_OR_MEDIA.finditer(text):
        unsafe.update(_overlapping_chunk_indices(boundaries, match.start(), match.end()))
    for match in _CROSS_CHUNK_MARKDOWN.finditer(text):
        touched = _overlapping_chunk_indices(boundaries, match.start(), match.end())
        if len(touched) > 1:
            unsafe.update(range(min(touched), max(touched) + 1))
    return {index for index, _chunk in ordered if index not in unsafe}


def _overlapping_chunk_indices(
    boundaries: list[tuple[int, int, int]], start: int, end: int,
) -> set[int]:
    """Return source chunk indexes overlapped by one complete construct."""
    return {index for left, right, index in boundaries if left < end and start < right}
