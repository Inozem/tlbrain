"""TL;DV folder detection from speaker indexes and Gemini suggestions."""

import logging
import re

from core.integrations.firestore.folders import get_all_folders, get_folders_by_speaker, speaker_key
from core.integrations.gemini.llm import call_gemini_json
from core.integrations.google_drive.folder_service import get_unassigned_folder_id
from services.connectors.tldv.formatting import _get_speakers

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.8
_SPEAKER_MIN_CLIENTS = 1
_SPEAKER_MAX_CLIENTS = 4

_DETECT_PROMPT_STAGE1 = """\
You are deciding where to save a meeting transcript in Google Drive.

Meeting name: {meeting_name}
Available folders:
{folders}

In which folder should this transcript be saved? Reply with valid JSON only:
{{"folder_name": "...", "confidence": 0.0}}

Rules:
- folder_name must be copied VERBATIM from the available folders list above, or null if unsure
- confidence 0.0-1.0; only use >=0.8 when highly certain
- If no folder matches or you are unsure, use null and low confidence
"""

_DETECT_PROMPT_STAGE2 = """\
You are deciding where to save a meeting transcript in Google Drive.

Meeting name: {meeting_name}
Transcript excerpt:
{transcript}

Available folders:
{folders}

In which folder should this transcript be saved? Reply with valid JSON only:
{{"folder_name": "...", "confidence": 0.0}}

Rules:
- folder_name must be copied VERBATIM from the available folders list above, or null if unsure
- confidence 0.0-1.0; only use >=0.8 when highly certain
- If no folder matches or you are unsure, use null and low confidence
"""

_PLACEHOLDER_SPEAKER_RE = re.compile(
    r"^(speaker|спикер|participant|участник)\s*\d+$",
    re.IGNORECASE,
)


def _is_placeholder_speaker(name: str) -> bool:
    return bool(_PLACEHOLDER_SPEAKER_RE.match(name))


def _get_folders_by_speakers(db, speakers: list[str]) -> tuple[list[str], list[str]]:
    """Return speaker candidates and all unique indexed folder IDs."""
    all_folder_ids: set[str] = set()
    candidates: set[str] = set()

    for speaker in speakers:
        folder_ids: set[str] = set()
        docs = get_folders_by_speaker(speaker, db)
        for doc in docs:
            data = doc.to_dict() or {}
            if data.get("name") != "_unassigned":
                folder_ids.add(doc.id)
            if len(folder_ids) >= 5:
                break
        all_folder_ids.update(folder_ids)
        if _SPEAKER_MIN_CLIENTS <= len(folder_ids) <= _SPEAKER_MAX_CLIENTS:
            candidates.update(folder_ids)

    return sorted(candidates), sorted(all_folder_ids)


def _detect_target_folder_id(
    db,
    meeting: dict,
    utterances: list[dict],
    drive,
    root_folder_id: str,
    *,
    gemini_json=None,
) -> str:
    """Select the folder ID for a TL;DV transcript using the supplied Firestore client."""
    gemini_json = gemini_json or call_gemini_json
    meeting_name = meeting.get("name", "")
    all_folders = get_all_folders(db)

    speakers = _get_speakers(utterances)

    if all(_is_placeholder_speaker(speaker) for speaker in speakers):
        logger.info("Stage 1: all speakers are placeholders (%s), skipping", speakers)
        candidates, all_candidates = [], []
    else:
        logger.info("Stage 1: speakers from TL;DV: %s", {s: speaker_key(s) for s in speakers})
        candidates, all_candidates = _get_folders_by_speakers(db, speakers)

    if len(candidates) == 1:
        logger.info("Folder detected via speakers: %s", candidates[0])
        return candidates[0]
    elif len(candidates) >= 2:
        logger.info("Stage 1: %d speaker candidates: %s", len(candidates), candidates)
    else:
        logger.info("Stage 1: no speaker candidates found")

    if not all_candidates:
        all_candidates = [
            folder_id for folder_id, data in all_folders.items()
            if data.get("name") != "_unassigned"
        ]

    if not all_candidates:
        logger.info("No known folders, falling back to _unassigned")
        return get_unassigned_folder_id(drive, root_folder_id)

    ids_to_check = candidates if candidates else all_candidates
    if not candidates:
        logger.info("Stage 1: falling back to all %d folders", len(ids_to_check))

    folder_names = [all_folders.get(folder_id, {}).get("name", folder_id) for folder_id in ids_to_check]
    folders_str = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(folder_names))
    name_to_ids: dict[str, list[str]] = {}
    for folder_id, name in zip(ids_to_check, folder_names):
        name_to_ids.setdefault(name, []).append(folder_id)

    try:
        result = gemini_json(_DETECT_PROMPT_STAGE1.format(
            meeting_name=meeting_name,
            folders=folders_str,
        ))
        folder_name = result.get("folder_name")
        confidence = result.get("confidence", 0)
        logger.info("Stage 2: Gemini returned folder=%r confidence=%.2f (threshold=%.1f)", folder_name, confidence, _CONFIDENCE_THRESHOLD)
        if folder_name and folder_name in name_to_ids and confidence >= _CONFIDENCE_THRESHOLD:
            folder_id = name_to_ids[folder_name][0]
            logger.info("Folder detected via meeting name: %s (%s)", folder_name, folder_id)
            return folder_id
    except Exception as exc:
        logger.warning("Stage 2 folder detection failed: %s", exc)

    excerpt = "\n".join(
        f"{utterance.get('speaker', '')}: {utterance.get('text', '')}"
        for utterance in utterances[:15]
        if utterance.get("text")
    )
    try:
        result = gemini_json(_DETECT_PROMPT_STAGE2.format(
            meeting_name=meeting_name,
            transcript=excerpt,
            folders=folders_str,
        ))
        folder_name = result.get("folder_name")
        confidence = result.get("confidence", 0)
        logger.info("Stage 3: Gemini returned folder=%r confidence=%.2f (threshold=%.1f)", folder_name, confidence, _CONFIDENCE_THRESHOLD)
        if folder_name and folder_name in name_to_ids and confidence >= _CONFIDENCE_THRESHOLD:
            folder_id = name_to_ids[folder_name][0]
            logger.info("Folder detected via transcript: %s (%s)", folder_name, folder_id)
            return folder_id
    except Exception as exc:
        logger.warning("Stage 3 folder detection failed: %s", exc)

    logger.info("Folder not detected, falling back to _unassigned")
    return get_unassigned_folder_id(drive, root_folder_id)
