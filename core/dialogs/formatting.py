"""Canonical text formatting for normalized dialogs."""

from __future__ import annotations

import json

from core.dialogs.models import NormalizedDialogInput


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def format_dialog(
    dialog: NormalizedDialogInput, *, dialog_time: str = ""
) -> tuple[str, list[tuple[int, int]]]:
    """Format a normalized dialog and return zero-based UTF-16 speaker ranges."""

    header = (
        f"DATE: {dialog.dialog_date}\n"
        f"TIME: {dialog_time}\n"
        f"PROVIDER: {dialog.provider}\n"
        "TEXT_ENCODING: json-string-v1\n"
        "---\n"
    )

    rows: list[str] = []
    speaker_ranges: list[tuple[int, int]] = []
    position = _utf16_length(header)
    for utterance in dialog.utterances:
        row = f"{utterance.speaker} :: {json.dumps(utterance.text, ensure_ascii=False)}"
        speaker_length = _utf16_length(utterance.speaker)
        speaker_ranges.append((position, position + speaker_length))
        rows.append(row)
        position += _utf16_length(row) + 2  # rows are separated by a blank line

    return header + "\n\n".join(rows), speaker_ranges
