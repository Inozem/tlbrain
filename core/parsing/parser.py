import json
import re
from typing import Any

_REQUIRED_HEADER_FIELDS = {"date", "provider"}
_JSON_STRING_ENCODING = "json-string-v1"


def is_valid_format(text: str) -> bool:
    """Return True if text matches the canonical TLBrain transcript format."""
    try:
        header_lines, body_lines = _split_header_body(text)
        if not header_lines:
            return False
        metadata = _parse_header(header_lines)
        if not _REQUIRED_HEADER_FIELDS.issubset(metadata.keys()):
            return False
        encoding = _parse_text_encoding(header_lines)
        return len(_parse_utterances(body_lines, encoding=encoding)) > 0
    except ValueError:
        return False


def parse_document(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    header_lines, body_lines = _split_header_body(text)
    metadata = _parse_header(header_lines)
    encoding = _parse_text_encoding(header_lines)
    utterances = _parse_utterances(body_lines, encoding=encoding)
    return metadata, utterances


def _parse_text_encoding(lines: list[str]) -> str | None:
    encoding: str | None = None
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip() == "TEXT_ENCODING":
            encoding = value.strip()
    if encoding is not None and encoding != _JSON_STRING_ENCODING:
        raise ValueError(f"Unsupported transcript text encoding: {encoding}")
    return encoding


def _split_header_body(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    try:
        sep = lines.index("---")
    except ValueError:
        return [], lines

    header_lines = lines[:sep]
    if _parse_text_encoding(header_lines) != _JSON_STRING_ENCODING:
        return header_lines, lines[sep + 1:]

    # splitlines() also treats U+2028, U+2029, and NEL as boundaries.
    # Those characters can occur as data in an ensure_ascii=False JSON
    # string, so recover the body boundary from the original text and only
    # split encoded records on physical CR/LF line endings.
    rows = text.splitlines(keepends=True)
    body_start = sum(len(row) for row in rows[: sep + 1])
    body = text[body_start:]
    return header_lines, re.split(r"\r\n|[\r\n]", body)


def _parse_header(lines: list[str]) -> dict[str, str]:
    mapping = {"DATE": "date", "TIME": "time", "PROVIDER": "provider"}
    result: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in mapping:
            result[mapping[key]] = value.strip()
    return result


def _parse_utterances(
    lines: list[str], *, encoding: str | None = None
) -> list[dict[str, Any]]:
    utterances = []
    for line in lines:
        if encoding == _JSON_STRING_ENCODING:
            if not line:
                continue
            if " :: " not in line:
                raise ValueError("Malformed encoded transcript utterance")
            speaker, _, text = line.partition(" :: ")
            speaker = speaker.strip()
            if not speaker:
                raise ValueError("Encoded transcript utterance speaker must not be empty")
            try:
                text = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed JSON string in encoded transcript") from exc
            if not isinstance(text, str):
                raise ValueError("Encoded transcript utterance text must be a JSON string")
            if not text.strip():
                raise ValueError("Encoded transcript utterance text must not be empty")
            utterances.append({
                "speaker": speaker,
                "text": text,
                "order_index": len(utterances),
            })
            continue

        if " :: " not in line:
            continue
        speaker, _, text = line.partition(" :: ")
        speaker = speaker.strip()
        text = text.strip()
        if speaker and text:
            utterances.append({
                "speaker": speaker,
                "text": text,
                "order_index": len(utterances),
            })
    return utterances
