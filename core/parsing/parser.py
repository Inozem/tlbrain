from typing import Any

_REQUIRED_HEADER_FIELDS = {"date", "provider"}


def is_valid_format(text: str) -> bool:
    """Return True if text matches the canonical TLBrain transcript format."""
    header_lines, body_lines = _split_header_body(text)
    if not header_lines:
        return False
    metadata = _parse_header(header_lines)
    if not _REQUIRED_HEADER_FIELDS.issubset(metadata.keys()):
        return False
    return len(_parse_utterances(body_lines)) > 0


def parse_document(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    header_lines, body_lines = _split_header_body(text)
    metadata = _parse_header(header_lines)
    utterances = _parse_utterances(body_lines)
    return metadata, utterances


def _split_header_body(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    try:
        sep = lines.index("---")
        return lines[:sep], lines[sep + 1:]
    except ValueError:
        return [], lines


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


def _parse_utterances(lines: list[str]) -> list[dict[str, Any]]:
    utterances = []
    for line in lines:
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
