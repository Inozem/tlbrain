import logging
import os

import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.cloud import firestore
from googleapiclient.discovery import build

from core.config import get_root_folder_id
from core.integrations.firestore.folders import increment_folder_speakers, speaker_key
from core.integrations.firestore.transcripts import get_index_records_by_meeting_id, save_index
from core.integrations.gemini.llm import call_gemini_json
from core.integrations.google_auth.credentials import credentials
from core.integrations.google_docs.client import build_docs_service
from core.integrations.google_docs.writer import create_google_doc
from core.google_drive.firestore import delete_queued_placeholder, mark_download_error, mark_downloading
from core.utils.logging import configure_logging
from core.integrations.cloud_tasks.client import enqueue_task
from tldv_client import tldv_get
from services.connectors.tldv.formatting import _format_transcript, _get_speakers, _parse_happened_at
from services.connectors.tldv.folder_selection import _detect_target_folder_id

configure_logging()
logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

app = FastAPI()


def _build_credentials():
    return credentials(_SCOPES, allow_adc_fallback=True)


def _build_drive():
    return build("drive", "v3", credentials=_build_credentials())


def _build_docs():
    return build_docs_service(_SCOPES, allow_adc_fallback=True)


def _create_google_doc(drive, folder_id: str, title: str, text: str, speaker_ranges: list[tuple[int, int]]) -> tuple[str, str]:
    docs = _build_docs()
    return create_google_doc(drive, docs, folder_id, title, text, speaker_ranges)


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
    existing = get_index_records_by_meeting_id(meeting_id, db)
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

    title = meeting.get("name") or f"TL;DV {meeting_id}"
    text, speaker_ranges = _format_transcript(meeting, utterances)
    dialog_date, _ = _parse_happened_at(meeting.get("happenedAt", ""))

    root_folder_id = get_root_folder_id()
    drive = _build_drive()
    target_folder_id = _detect_target_folder_id(
        db, meeting, utterances, drive, root_folder_id, gemini_json=call_gemini_json
    )

    doc_id, modified_time = _create_google_doc(drive, target_folder_id, title, text, speaker_ranges)
    logger.info("Created Google Doc: doc_id=%s title=%r", doc_id, title)

    speakers = _get_speakers(utterances)
    save_index(doc_id, {
        "doc_id": doc_id,
        "parent_id": target_folder_id,
        "dialog_date": dialog_date,
        "provider": "tldv",
        "source_file": title,
        "meeting_id": meeting_id,
        "speakers": speakers,
        "modifiedTime": modified_time,
        "status": "imported",
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    }, db=db)
    logger.info("Saved to Firestore: doc_id=%s", doc_id)

    if speakers:
        increment_folder_speakers(target_folder_id, speakers, db=db)

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
