"""TL;DV-specific transcript field interpretation and formatting."""

from datetime import datetime


def _parse_happened_at(happened_at: str) -> tuple[str, str]:
    if not happened_at:
        return "", ""
    try:
        dt = datetime.fromisoformat(happened_at.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except ValueError:
        return "", ""


def _format_transcript(meeting: dict, utterances: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    dialog_date, dialog_time = _parse_happened_at(meeting.get("happenedAt", ""))

    header = "\n".join([
        f"DATE: {dialog_date}",
        f"TIME: {dialog_time}",
        "PROVIDER: tldv",
        "---",
    ]) + "\n"

    speaker_ranges: list[tuple[int, int]] = []
    body_parts: list[str] = []
    pos = len(header)

    for utterance in utterances:
        speaker = (utterance.get("speaker") or utterance.get("speakerName", "")).replace("::", "")
        text = utterance.get("text") or utterance.get("transcript", "")
        if not text:
            continue
        if speaker:
            speaker_ranges.append((pos, pos + len(speaker)))
            line = f"{speaker} :: {text}"
        else:
            line = text
        body_parts.append(line)
        pos += len(line) + 2  # +2 for the blank-line separator

    return header + "\n\n".join(body_parts), speaker_ranges


def _get_speakers(utterances: list[dict]) -> list[str]:
    """Return normalized, unique speaker labels for Firestore storage."""
    speakers = {
        (utterance.get("speaker") or utterance.get("speakerName", "")).replace("::", "").strip()
        for utterance in utterances
        if utterance.get("text")
    }
    speakers.discard("")
    return sorted(speakers)
