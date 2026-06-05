import logging
import os
import re
from datetime import datetime, timezone

import google.auth
import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.cloud import firestore
from googleapiclient.discovery import build

from core.config import get_root_folder_id
from core.gemini.llm import call_gemini_json
from core.google_drive.firestore import CLIENTS_COLLECTION, COLLECTION_NAME, _speaker_key, delete_queued_placeholder, get_all_client_names, mark_download_error, mark_downloading, update_client_speakers
from core.utils.logging import configure_logging
from core.utils.tasks import enqueue_task
from tldv_client import tldv_get

configure_logging()
logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

app = FastAPI()


def _build_credentials():
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if refresh_token:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
        )
        creds.refresh(Request())
        return creds
    creds, _ = google.auth.default(scopes=_SCOPES)
    return creds


def _build_drive():
    return build("drive", "v3", credentials=_build_credentials())


def _build_docs():
    return build("docs", "v1", credentials=_build_credentials())


def _get_or_create_folder(drive, parent_id: str, name: str) -> str:
    results = drive.files().list(
        q=f"name={name!r} and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false",
        fields="files(id)",
        pageSize=1,
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = drive.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return folder["id"]


def _parse_happened_at(happened_at: str) -> tuple[str, str]:
    if not happened_at:
        return "", ""
    try:
        dt = datetime.fromisoformat(happened_at.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except ValueError:
        return "", ""


def _format_transcript(meeting: dict, utterances: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    source_file = meeting.get("name") or f"TL;DV {meeting.get('id', '')}"
    dialog_date, dialog_time = _parse_happened_at(meeting.get("happenedAt", ""))

    header = "\n".join([
        f"DATE: {dialog_date}",
        f"TIME: {dialog_time}",
        "PROVIDER: tldv",
        f"SOURCE_FILE: {source_file}",
        "---",
    ]) + "\n"

    speaker_ranges: list[tuple[int, int]] = []
    body_parts: list[str] = []
    pos = len(header)

    for u in utterances:
        speaker = (u.get("speaker") or u.get("speakerName", "")).replace("::", "")
        text = u.get("text") or u.get("transcript", "")
        if not text:
            continue
        if speaker:
            speaker_ranges.append((pos, pos + len(speaker)))
            line = f"{speaker} :: {text}"
        else:
            line = text
        body_parts.append(line)
        pos += len(line) + 2  # +2 for \n\n separator

    return header + "\n\n".join(body_parts), speaker_ranges


def _create_google_doc(drive, folder_id: str, title: str, text: str, speaker_ranges: list[tuple[int, int]]) -> tuple[str, str]:
    docs = _build_docs()

    file_meta = drive.files().create(
        body={
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [folder_id],
        },
        fields="id,modifiedTime",
    ).execute()
    doc_id = file_meta["id"]
    modified_time = file_meta.get("modifiedTime", "")

    requests_body = [{"insertText": {"location": {"index": 1}, "text": text}}]
    for start, end in speaker_ranges:
        requests_body.append({
            "updateTextStyle": {
                "range": {"startIndex": start + 1, "endIndex": end + 1},
                "textStyle": {"bold": True},
                "fields": "bold",
            }
        })

    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests_body}).execute()

    return doc_id, modified_time


_CONFIDENCE_THRESHOLD = 0.8

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


_SPEAKER_MIN_CLIENTS = 1
_SPEAKER_MAX_CLIENTS = 4

_PLACEHOLDER_SPEAKER_RE = re.compile(
    r"^(speaker|спикер|participant|участник)\s*\d+$",
    re.IGNORECASE,
)


def _is_placeholder_speaker(name: str) -> bool:
    return bool(_PLACEHOLDER_SPEAKER_RE.match(name))


def _get_speakers(utterances: list[dict]) -> list[str]:
    speakers = {
        (u.get("speaker") or u.get("speakerName", "")).replace("::", "").strip()
        for u in utterances
        if u.get("text")
    }
    speakers.discard("")
    return sorted(speakers)


def _get_clients_by_speakers(db: firestore.Client, speakers: list[str]) -> tuple[list[str], list[str]]:
    """Query clients collection by speaker index.

    Returns (candidates, all_clients):
    - candidates: clients where at least one speaker appears in 1-4 unique clients
    - all_clients: all unique clients seen across all speaker queries (fallback for Gemini)
    """
    all_clients: set[str] = set()
    candidates: set[str] = set()

    for speaker in speakers:
        client_names: set[str] = set()
        docs = db.collection(CLIENTS_COLLECTION) \
            .where(filter=firestore.FieldFilter(f"speakers.{_speaker_key(speaker)}", ">", 0)) \
            .stream()
        for doc in docs:
            if doc.id != "_unassigned":
                client_names.add(doc.id)
            if len(client_names) >= 5:
                break
        all_clients.update(client_names)
        if _SPEAKER_MIN_CLIENTS <= len(client_names) <= _SPEAKER_MAX_CLIENTS:
            candidates.update(client_names)

    return sorted(candidates), sorted(all_clients)


def _detect_client_name(db: firestore.Client, meeting: dict, utterances: list[dict]) -> str:
    meeting_name = meeting.get("name", "")

    # Stage 1: speaker frequency analysis
    speakers = _get_speakers(utterances)
    if all(_is_placeholder_speaker(s) for s in speakers):
        logger.info("Stage 1: all speakers are placeholders (%s), skipping", speakers)
        candidates, all_clients = [], []
    else:
        logger.info("Stage 1: speakers from TL;DV: %s", {s: _speaker_key(s) for s in speakers})
        candidates, all_clients = _get_clients_by_speakers(db, speakers)

    if len(candidates) == 1:
        logger.info("Client detected via speakers: %s", candidates[0])
        return candidates[0]
    elif len(candidates) >= 2:
        logger.info("Stage 1: %d speaker candidates: %s", len(candidates), candidates)
    else:
        logger.info("Stage 1: no speaker candidates found")

    if not all_clients:
        all_clients = get_all_client_names()

    if not all_clients:
        logger.info("No known clients, falling back to _unassigned")
        return "_unassigned"

    clients_to_check = candidates if candidates else all_clients
    if not candidates:
        logger.info("Stage 1: falling back to all %d clients", len(clients_to_check))
    folders_str = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(clients_to_check))

    # Stage 2: Gemini by meeting name (restricted to candidates or all)
    try:
        result = call_gemini_json(_DETECT_PROMPT_STAGE1.format(
            meeting_name=meeting_name,
            folders=folders_str,
        ))
        client_name = result.get("folder_name")
        confidence = result.get("confidence", 0)
        logger.info("Stage 2: Gemini returned folder=%r confidence=%.2f (threshold=%.1f)", client_name, confidence, _CONFIDENCE_THRESHOLD)
        if client_name and client_name in clients_to_check and confidence >= _CONFIDENCE_THRESHOLD:
            logger.info("Client detected via meeting name: %s", client_name)
            return client_name
    except Exception as exc:
        logger.warning("Stage 2 client detection failed: %s", exc)

    # Stage 3: Gemini by first 15 utterances (restricted to candidates or all)
    excerpt = "\n".join(
        f"{u.get('speaker', '')}: {u.get('text', '')}"
        for u in utterances[:15]
        if u.get("text")
    )
    try:
        result = call_gemini_json(_DETECT_PROMPT_STAGE2.format(
            meeting_name=meeting_name,
            transcript=excerpt,
            folders=folders_str,
        ))
        client_name = result.get("folder_name")
        confidence = result.get("confidence", 0)
        logger.info("Stage 3: Gemini returned folder=%r confidence=%.2f (threshold=%.1f)", client_name, confidence, _CONFIDENCE_THRESHOLD)
        if client_name and client_name in clients_to_check and confidence >= _CONFIDENCE_THRESHOLD:
            logger.info("Client detected via transcript: %s", client_name)
            return client_name
    except Exception as exc:
        logger.warning("Stage 3 client detection failed: %s", exc)

    logger.info("Client not detected, falling back to _unassigned")
    return "_unassigned"


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/import")
async def import_meeting(request: Request):
    body = await request.json()
    meeting_id = body.get("meeting_id")
    if not meeting_id:
        return JSONResponse({"error": "missing meeting_id"}, status_code=400)

    logger.info("Importing meeting_id=%s", meeting_id)

    db = firestore.Client()
    existing = list(
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("meeting_id", "==", meeting_id))
        .stream()
    )
    if any(d.to_dict().get("status") != "queued" for d in existing):
        logger.info("Already imported: %s", meeting_id)
        return {"ok": True, "skipped": True}

    mark_downloading(meeting_id)

    try:
        meeting = tldv_get(f"/meetings/{meeting_id}")
    except http_requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("Meeting not found in TL;DV API, deleting placeholder: %s", meeting_id)
            delete_queued_placeholder(meeting_id)
            return {"ok": True, "skipped": True, "reason": "meeting_not_found"}
        if exc.response is not None and exc.response.status_code == 403:
            logger.warning("Meeting forbidden in TL;DV API (403), marking error: %s", meeting_id)
            mark_download_error(meeting_id, "HTTP 403 fetching meeting")
            return {"ok": True, "skipped": True, "reason": "meeting_http_403"}
        raise
    logger.info("Meeting: %r", meeting.get("name"))

    try:
        transcript_resp = tldv_get(f"/meetings/{meeting_id}/transcript")
    except http_requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (403, 404):
            status_code = exc.response.status_code
            logger.warning("Transcript HTTP %s for meeting_id=%s, marking error", status_code, meeting_id)
            mark_download_error(meeting_id, f"HTTP {status_code} fetching transcript")
            return {"ok": True, "skipped": True, "reason": f"transcript_http_{status_code}"}
        raise
    utterances = transcript_resp.get("results", transcript_resp.get("data", []))
    logger.info("Utterances: %d", len(utterances))

    if not utterances:
        logger.warning("Empty transcript for meeting_id=%s, marking error", meeting_id)
        mark_download_error(meeting_id, "empty transcript")
        return {"ok": True, "skipped": True, "reason": "empty_transcript"}

    client_name = _detect_client_name(db, meeting, utterances)
    title = meeting.get("name") or f"TL;DV {meeting_id}"
    text, speaker_ranges = _format_transcript(meeting, utterances)
    dialog_date, _ = _parse_happened_at(meeting.get("happenedAt", ""))

    root_folder_id = get_root_folder_id()
    drive = _build_drive()
    client_folder_id = _get_or_create_folder(drive, root_folder_id, client_name)

    doc_id, modified_time = _create_google_doc(drive, client_folder_id, title, text, speaker_ranges)
    logger.info("Created Google Doc: doc_id=%s title=%r", doc_id, title)

    speakers = _get_speakers(utterances)
    db.collection(COLLECTION_NAME).document(doc_id).set({
        "doc_id": doc_id,
        "root_folder_id": root_folder_id,
        "client_name": client_name,
        "drive_folder": client_folder_id,
        "dialog_date": dialog_date,
        "provider": "tldv",
        "source_file": title,
        "meeting_id": meeting_id,
        "speakers": speakers,
        "speakers_indexed": True,
        "modifiedTime": modified_time,
        "status": "imported",
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Saved to Firestore: doc_id=%s", doc_id)
    update_client_speakers(client_name, speakers)
    delete_queued_placeholder(meeting_id)

    vector_sync_url = os.environ.get("VECTOR_SYNC_URL", "")
    vector_sync_queue = os.environ.get("VECTOR_SYNC_QUEUE", "")

    if vector_sync_url and vector_sync_queue:
        enqueue_task(
            queue_name=vector_sync_queue,
            task_id=doc_id,
            url=f"{vector_sync_url}/sync/doc/{doc_id}",
        )
        logger.info("Queued vector sync: doc_id=%s", doc_id)
    else:
        logger.warning("VECTOR_SYNC_URL or VECTOR_SYNC_QUEUE not set, skipping sync queue")

    return {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
