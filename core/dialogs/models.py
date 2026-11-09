"""Protocol-neutral models and normalization for accepted dialog input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import re

from core.config import get_default_dialog_date


@dataclass(frozen=True, slots=True)
class DialogUtterance:
    """One ordered speaker/text pair in a dialog."""

    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class NormalizedDialogInput:
    """Validated dialog fields captured at request acceptance time."""

    provider: str
    title: str
    dialog_date: str
    folder_path: list[str] | None
    utterances: tuple[DialogUtterance, ...]
    folder_id: str | None = None


def _normalize_provider(arguments: Mapping[str, object]) -> str:
    provider = arguments.get("provider")
    if not isinstance(provider, str):
        raise TypeError("provider must be a string")
    provider = provider.strip()
    if not provider or "\r" in provider or "\n" in provider:
        raise ValueError("provider must be a nonempty single-line label")
    return provider


def _normalize_utterances(arguments: Mapping[str, object]) -> tuple[DialogUtterance, ...]:
    raw_utterances = arguments.get("utterances")
    if (
        isinstance(raw_utterances, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_utterances, Sequence)
    ):
        raise TypeError("utterances must be an ordered sequence")
    if not raw_utterances:
        raise ValueError("utterances must not be empty")

    utterances: list[DialogUtterance] = []
    for index, item in enumerate(raw_utterances):
        if not isinstance(item, Mapping):
            raise TypeError(f"utterance {index} must be a mapping")
        if set(item) != {"speaker", "text"}:
            raise ValueError(f"utterance {index} must contain only speaker and text")

        speaker = item["speaker"]
        text = item["text"]
        if not isinstance(speaker, str):
            raise TypeError(f"utterance {index} speaker must be a string")
        if "\r" in speaker or "\n" in speaker:
            raise ValueError(f"utterance {index} speaker must be single-line")
        speaker = speaker.replace("::", "").strip()
        if not speaker:
            raise ValueError(f"utterance {index} speaker must not be empty")
        if not isinstance(text, str):
            raise TypeError(f"utterance {index} text must be a string")
        if not text.strip():
            raise ValueError(f"utterance {index} text must not be blank")
        utterances.append(DialogUtterance(speaker=speaker, text=text))

    return tuple(utterances)


def _normalize_date_and_title(
    arguments: Mapping[str, object], *, accepted_at: datetime
) -> tuple[str, str]:
    if "dialog_date" in arguments:
        raw_date = arguments["dialog_date"]
        if not isinstance(raw_date, str):
            raise TypeError("dialog_date must be an ISO date string")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            raise ValueError("dialog_date must use YYYY-MM-DD format")
        try:
            dialog_date = date.fromisoformat(raw_date).isoformat()
        except ValueError as exc:
            raise ValueError("dialog_date must be a valid calendar date") from exc
    else:
        dialog_date = get_default_dialog_date(accepted_at).isoformat()

    if "title" in arguments:
        raw_title = arguments["title"]
        if not isinstance(raw_title, str):
            raise TypeError("title must be a string")
        title = raw_title.strip()
        if not title:
            raise ValueError("title must not be blank")
    else:
        title = f"TLBrain dialog {dialog_date}"

    return dialog_date, title


def _normalize_explicit_folder_path(
    arguments: Mapping[str, object],
) -> list[str] | None:
    if "folder_path" not in arguments:
        return None

    raw_path = arguments["folder_path"]
    if (
        isinstance(raw_path, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_path, Sequence)
    ):
        raise TypeError("folder_path must be a sequence of strings")
    folder_path: list[str] = []
    for index, segment in enumerate(raw_path):
        if not isinstance(segment, str):
            raise TypeError(f"folder_path segment {index} must be a string")
        if not segment.strip():
            raise ValueError(f"folder_path segment {index} must not be blank")
        folder_path.append(segment)
    return folder_path


def _normalize_folder_id(arguments: Mapping[str, object]) -> str | None:
    if "folder_id" not in arguments:
        return None
    value = arguments["folder_id"]
    if not isinstance(value, str):
        raise TypeError("folder_id must be a string")
    if not value.strip() or "/" in value:
        raise ValueError("folder_id must be a nonempty document ID")
    return value


def normalize_dialog_input(
    arguments: Mapping[str, object], *, accepted_at: datetime
) -> NormalizedDialogInput:
    """Validate and normalize a dialog request without performing I/O."""

    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    if not isinstance(accepted_at, datetime):
        raise TypeError("accepted_at must be a datetime")
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        raise ValueError("accepted_at must be timezone-aware")

    allowed_keys = {"provider", "title", "dialog_date", "folder_path", "folder_id", "utterances"}
    unknown_keys = set(arguments) - allowed_keys
    if unknown_keys:
        raise ValueError(f"unknown dialog input keys: {sorted(unknown_keys)!r}")

    provider = _normalize_provider(arguments)
    utterances = _normalize_utterances(arguments)
    dialog_date, title = _normalize_date_and_title(arguments, accepted_at=accepted_at)
    folder_path = _normalize_explicit_folder_path(arguments)
    folder_id = _normalize_folder_id(arguments)
    if folder_path is not None and folder_id is not None:
        raise ValueError("specify either folder_path or folder_id, not both")

    return NormalizedDialogInput(
        provider=provider,
        title=title,
        dialog_date=dialog_date,
        folder_path=folder_path,
        utterances=utterances,
        folder_id=folder_id,
    )
