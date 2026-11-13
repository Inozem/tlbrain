"""TL;DV folder detection from speaker indexes and Gemini suggestions."""

import logging
import re

from core.dialogs.classification import FolderSelectionRequest, classify_dialog_folder
from core.dialogs.models import DialogUtterance
from core.integrations.firestore.folders import get_all_folders, get_folders_by_speaker
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

    def speaker_lookup(speaker: str) -> list[str]:
        docs = get_folders_by_speaker(speaker, db)
        folder_ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            data = doc.to_dict() or {}
            if data.get("name") != "_unassigned" and doc.id not in seen:
                seen.add(doc.id)
                folder_ids.append(doc.id)
            if len(folder_ids) >= 5:
                break
        return folder_ids

    def suggest(stage, candidate_ids, _title, _excerpt, _topic_hint):
        folder_names = [all_folders.get(folder_id, {}).get("name", folder_id) for folder_id in candidate_ids]
        folders_str = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(folder_names))
        if stage == "title":
            prompt = _DETECT_PROMPT_STAGE1.format(meeting_name=meeting_name, folders=folders_str)
        else:
            excerpt = "\n".join(
                f"{utterance.get('speaker', '')}: {utterance.get('text', '')}"
                for utterance in utterances[:15]
                if utterance.get("text")
            )
            prompt = _DETECT_PROMPT_STAGE2.format(
                meeting_name=meeting_name,
                transcript=excerpt,
                folders=folders_str,
            )
        try:
            result = gemini_json(prompt)
            folder_name = result.get("folder_name")
            confidence = result.get("confidence", 0)
            logger.info(
                "Stage %s: Gemini returned folder=%r confidence=%.2f (threshold=%.1f)",
                "2" if stage == "title" else "3",
                folder_name,
                confidence,
                _CONFIDENCE_THRESHOLD,
            )
            return folder_name, confidence
        except Exception as exc:
            logger.warning(
                "Stage %s folder detection failed (%s)",
                "2" if stage == "title" else "3",
                type(exc).__name__,
            )
            raise

    normalized_utterances = tuple(
        DialogUtterance(
            speaker=str(utterance.get("speaker", "")),
            text=str(utterance.get("text", "")),
        )
        for utterance in utterances
    )
    result = classify_dialog_folder(
        FolderSelectionRequest(
            title=meeting_name,
            utterances=normalized_utterances,
            reliable_speakers=tuple(speakers),
        ),
        all_folders,
        root_folder_id,
        speaker_index_lookup=speaker_lookup,
        suggestion=suggest,
    )
    if result.folder_id is not None:
        return result.folder_id
    logger.info("Folder not detected, falling back to _unassigned")
    return get_unassigned_folder_id(drive, root_folder_id)
