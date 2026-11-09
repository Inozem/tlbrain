"""Pure preparation of Telegram source messages for downstream processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from services.connectors.telegram.models import PreparedUtterance


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _source_fragment(value: Any) -> str | None:
    """Preserve every nonempty source string exactly, including whitespace."""
    return value if isinstance(value, str) and value != "" else None


def _state_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _message_transcription(message: Any) -> str | None:
    """Choose one usable transcript, preferring pipeline output over saved text."""
    transcription = _nonempty_text(_get(message, "transcription"))
    persisted = _nonempty_text(_get(message, "transcription_text"))
    state = _state_value(_get(message, "transcription_state"))
    audio_file_id = _get(message, "audio_file_id")

    if audio_file_id and state == "failed":
        raise ValueError(
            f"audio message {_get(message, 'message_id')!r} has no completed transcription"
        )

    if audio_file_id and transcription is None and not (
        state == "completed" and persisted
    ):
        raise ValueError(
            f"audio message {_get(message, 'message_id')!r} has no completed transcription"
        )

    if transcription is not None:
        return transcription
    if persisted is not None and (state == "completed" or not audio_file_id):
        return persisted
    return None


def _message_text(message: Any) -> str:
    fragments = [
        fragment
        for fragment in (
            _source_fragment(_get(message, "text")),
            _source_fragment(_get(message, "caption")),
        )
        if fragment is not None
    ]

    transcription = _message_transcription(message)
    if transcription is not None:
        fragments.append(transcription)

    return "\n\n".join(fragments)


def _is_merge_eligible(author: str | None, word_count: int) -> bool:
    """Only reliably attributed messages under 50 words may join a merge block."""
    return author is not None and word_count < 50


def preprocess_messages(messages: Sequence[Any]) -> list[PreparedUtterance]:
    """Restore source order and combine eligible contiguous author messages."""
    ordered = sorted(
        messages,
        key=lambda message: (
            int(_get(message, "intake_sequence")),
            int(_get(message, "message_id")),
        ),
    )

    prepared: list[PreparedUtterance] = []
    block_messages: list[Any] = []
    block_texts: list[str] = []
    block_word_count = 0
    block_author: str | None = None

    def flush() -> None:
        nonlocal block_messages, block_texts, block_word_count, block_author
        if not block_messages:
            return
        first = block_messages[0]
        speaker = _get(first, "display_speaker") or _get(first, "display_name")
        if not isinstance(speaker, str) or not speaker.strip():
            speaker = "Unknown"
        prepared.append(PreparedUtterance(
            order_index=len(prepared),
            speaker=speaker,
            reliable_author_key=block_author,
            text="\n\n".join(block_texts),
            word_count=block_word_count,
            source_message_ids=tuple(int(_get(message, "message_id")) for message in block_messages),
        ))
        block_messages = []
        block_texts = []
        block_word_count = 0
        block_author = None

    for message in ordered:
        text = _message_text(message)
        if not text.strip():
            flush()
            continue
        count = len(text.split())
        author = _get(message, "reliable_author_key")
        if not isinstance(author, str) or not author.strip():
            author = None

        merge_eligible = _is_merge_eligible(author, count)
        can_merge = (
            merge_eligible
            and block_messages
            and block_author == author
            and block_word_count + count <= 100
        )
        if not can_merge:
            flush()

        if merge_eligible:
            block_messages.append(message)
            block_texts.append(text)
            block_word_count += count
            block_author = author
        else:
            block_messages.append(message)
            block_texts.append(text)
            block_word_count = count
            block_author = author
            flush()

    flush()
    return prepared


def context_word_capacity(utterance_count: int) -> int:
    """Return the deterministic context allowance for a prepared batch."""
    return max(30, int(utterance_count) + 20)
