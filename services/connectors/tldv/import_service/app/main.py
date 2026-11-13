import logging
import os

import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.auth.exceptions import RefreshError
from google.cloud import firestore
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.integrations.firestore.folders import increment_folder_speakers, speaker_key
from core.integrations.firestore.transcripts import (
    TLDV_UNKNOWN_WRITE_ERROR_PREFIX,
    claim_tldv_import,
    delete_queued_placeholder,
    get_index_records_by_meeting_id,
    get_transcript_record,
    is_tldv_transcript_record,
    is_tldv_placeholder_record,
    is_tldv_unknown_write_outcome,
    mark_download_error,
    save_tldv_index,
    settle_tldv_dispatch,
)
from core.integrations.gemini.llm import call_gemini_json
from core.integrations.google_auth.credentials import credentials
from core.integrations.google_docs.client import build_docs_service
from core.integrations.google_docs.writer import create_google_doc
from core.utils.logging import configure_logging
from core.integrations.cloud_tasks.client import enqueue_task
from tldv_client import tldv_get
from services.connectors.tldv.formatting import _format_transcript, _get_speakers, _parse_happened_at
from services.connectors.tldv.folder_selection import _detect_target_folder_id

configure_logging()
logger = logging.getLogger(__name__)
_UNKNOWN_DOC_WRITE_PREFIX = TLDV_UNKNOWN_WRITE_ERROR_PREFIX

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


def _create_google_doc(
    drive, folder_id: str, title: str, text: str, speaker_ranges: list[tuple[int, int]], docs=None,
) -> tuple[str, str]:
    docs = docs if docs is not None else _build_docs()
    return create_google_doc(drive, docs, folder_id, title, text, speaker_ranges)


def _credential_error(exc: Exception) -> bool:
    return isinstance(exc, RefreshError) and any(
        isinstance(value, dict) and value.get("error") == "invalid_grant"
        for value in exc.args
    )


def _google_http_detail(stage: str, exc: HttpError) -> str | None:
    """Format only a validated Google API status and controlled request stage."""
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        return None

    if stage in ("authorizing Google Drive", "selecting destination folder"):
        component = "Google Drive"
    elif stage == "authorizing Google Docs":
        component = "Google Docs"
    elif stage == "creating Google Doc":
        component = "Google Drive/Docs"
    else:
        component = "Google API"
    return f"{component} HTTP {status} during {stage}"


def _safe_failure(stage: str, exc: Exception) -> str:
    if _credential_error(exc):
        return "Google Drive authorization expired or was revoked (invalid_grant); refresh the configured credentials."
    if isinstance(exc, RefreshError):
        return "Google credentials refresh failed (RefreshError)."
    if isinstance(exc, HttpError):
        detail = _google_http_detail(stage, exc)
        if detail is not None:
            return f"Google API request failed ({detail})."
        return f"Google API request failed during {stage} (HttpError)."
    return f"TL;DV import failed during {stage} ({type(exc).__name__})."


def _failure_response(message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=500)


class _CanonicalIndexConflict(Exception):
    """Refuse publication when a Firestore ID is already owned by another Doc."""


def _saved_record(record) -> bool:
    data = record.to_dict() or {}
    if not is_tldv_transcript_record(data):
        return False
    if data.get("doc_id") is not None and data.get("doc_id") != record.id:
        raise _CanonicalIndexConflict()
    return not is_tldv_placeholder_record(record.id, data)


class _ImportProgress:
    """Track the last safe stage for an import failure response."""

    def __init__(self, stage: str):
        self.stage = stage

    def set(self, stage: str) -> None:
        self.stage = stage


def _find_saved_record(meeting_id: str, db):
    for record in get_index_records_by_meeting_id(meeting_id, db):
        if _saved_record(record):
            return record
    return None


def _saved_doc_has_advanced(doc_id: str, meeting_id: str, db) -> bool:
    data = get_transcript_record(doc_id, db=db) or {}
    if (
        not is_tldv_transcript_record(data)
        or data.get("meeting_id") != meeting_id
        or data.get("doc_id") not in (None, doc_id)
        or is_tldv_placeholder_record(doc_id, data)
    ):
        return False
    return data.get("status") in ("syncing", "synced") or (
        data.get("status") == "error" and data.get("error_stage") == "vector_sync"
    )


def _dispatch_saved_record(record, meeting_id: str, db):
    """Resume only a canonical Doc that has not entered vector indexing."""
    data = record.to_dict() or {}
    if data.get("status") in ("syncing", "synced") or (
        data.get("status") == "error" and data.get("error_stage") == "vector_sync"
    ):
        return {"ok": True, "skipped": True}
    dispatch_error = data.get("status") == "error" and data.get("error_stage") == "import"
    if data.get("status") not in ("imported", "queued") and not dispatch_error:
        return {"ok": True, "skipped": True}

    try:
        enqueued = _enqueue_vector_sync(record.id)
        if not enqueued:
            if data.get("status") == "imported" and not dispatch_error:
                return {"ok": True, "skipped": True}
            error = "Vector sync dispatch is not configured; set VECTOR_SYNC_URL and VECTOR_SYNC_QUEUE."
            if not settle_tldv_dispatch(record.id, meeting_id, error=error, db=db):
                if _saved_doc_has_advanced(record.id, meeting_id, db):
                    return {"ok": True, "skipped": True}
                return _failure_response("TL;DV vector sync dispatch state could not be settled safely.")
            return _failure_response(error)
    except Exception as exc:
        message = _safe_failure("dispatching vector sync", exc)
        if not settle_tldv_dispatch(record.id, meeting_id, error=message, db=db):
            if _saved_doc_has_advanced(record.id, meeting_id, db):
                return {"ok": True, "skipped": True}
            return _failure_response("TL;DV vector sync dispatch state could not be settled safely.")
        logger.error("TL;DV vector sync dispatch failed (%s)", type(exc).__name__)
        return _failure_response(message)

    if not settle_tldv_dispatch(record.id, meeting_id, db=db):
        if _saved_doc_has_advanced(record.id, meeting_id, db):
            return {"ok": True, "skipped": True}
        return _failure_response("TL;DV vector sync dispatch state could not be settled safely.")
    delete_queued_placeholder(meeting_id, db=db)
    return {"ok": True, "skipped": True}


def _run_new_import(meeting_id: str, db, progress: _ImportProgress):
    meeting, response = _fetch_meeting(meeting_id, db)
    if response is not None:
        return response
    logger.info("Meeting metadata received for meeting_id=%s", meeting_id)

    progress.set("fetching transcript")
    utterances, response = _fetch_utterances(meeting_id, db)
    if response is not None:
        return response

    progress.set("preparing transcript")
    doc_id = _persist_imported_document(meeting_id, meeting, utterances, db, set_stage=progress.set)
    progress.set("dispatching vector sync")
    try:
        _enqueue_vector_sync(doc_id)
    except Exception as exc:
        message = _safe_failure(progress.stage, exc)
        if not settle_tldv_dispatch(doc_id, meeting_id, error=message, db=db):
            record = get_transcript_record(doc_id, db=db)
            if record and _saved_doc_has_advanced(doc_id, meeting_id, db):
                return {"ok": True, "skipped": True}
            return _failure_response("TL;DV vector sync dispatch state could not be settled safely.")
        logger.error("TL;DV vector sync dispatch failed (%s)", type(exc).__name__)
        return _failure_response(message)
    return {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}


def _settle_import_failure(meeting_id: str, progress: _ImportProgress, exc: Exception, db):
    """Persist a safe failure while avoiding retries after an uncertain Doc write."""
    message = _safe_failure(progress.stage, exc)
    try:
        saved_record = _find_saved_record(meeting_id, db)
    except Exception as read_exc:
        saved_record = None
        logger.error("TL;DV failure diagnosis read failed (%s)", type(read_exc).__name__)

    write_may_have_started = progress.stage in ("creating Google Doc", "saving transcript status")
    if saved_record is not None:
        try:
            settled = settle_tldv_dispatch(saved_record.id, meeting_id, error=message, db=db)
        except Exception as settle_exc:
            settled = False
            logger.error("TL;DV failure settlement failed (%s)", type(settle_exc).__name__)
        if settled:
            logger.error("TL;DV import failed during %s (%s)", progress.stage, type(exc).__name__)
            return _failure_response(message)
        try:
            if _saved_doc_has_advanced(saved_record.id, meeting_id, db):
                return _failure_response(message)
        except Exception as read_exc:
            logger.error("TL;DV failure state read failed (%s)", type(read_exc).__name__)
        return _failure_response("TL;DV import failed and its saved Doc status could not be settled safely.")

    if write_may_have_started:
        auth_detail = "; authorization may have expired" if _credential_error(exc) else ""
        http_detail = _google_http_detail(progress.stage, exc) if isinstance(exc, HttpError) else None
        safe_cause = f"; {http_detail}" if http_detail else ""
        message = f"{_UNKNOWN_DOC_WRITE_PREFIX} ({type(exc).__name__}{auth_detail}{safe_cause}); reconcile the saved document before retrying import."
    try:
        persisted = mark_download_error(meeting_id, message, db=db)
    except Exception as persist_exc:
        logger.error("Failed to persist TL;DV import failure (%s)", type(persist_exc).__name__)
        return _failure_response("TL;DV import failed and its error status could not be recorded.")
    if not persisted:
        logger.error("TL;DV failure status was not written (%s)", type(exc).__name__)
        if message.startswith(_UNKNOWN_DOC_WRITE_PREFIX):
            return _failure_response(
                f"{message} The error status could not be written without altering the existing provider record."
            )
        return _failure_response("TL;DV import failed and its error status could not be recorded safely.")
    logger.error("TL;DV import failed during %s (%s)", progress.stage, type(exc).__name__)
    return _failure_response(message)


def _fetch_meeting(meeting_id: str, db):
    try:
        return tldv_get(f"/meetings/{meeting_id}"), None
    except http_requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("Meeting not found in TL;DV API, deleting placeholder: %s", meeting_id)
            delete_queued_placeholder(meeting_id, db=db)
            return None, {"ok": True, "skipped": True, "reason": "meeting_not_found"}
        if exc.response is not None and exc.response.status_code == 403:
            logger.warning("Meeting forbidden in TL;DV API (403), marking error: %s", meeting_id)
            mark_download_error(meeting_id, "HTTP 403 fetching meeting", db=db)
            return None, {"ok": True, "skipped": True, "reason": "meeting_http_403"}
        raise


def _fetch_utterances(meeting_id: str, db):
    try:
        transcript_resp = tldv_get(f"/meetings/{meeting_id}/transcript")
    except http_requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (403, 404):
            status_code = exc.response.status_code
            logger.warning("Transcript HTTP %s for meeting_id=%s, marking error", status_code, meeting_id)
            mark_download_error(meeting_id, f"HTTP {status_code} fetching transcript", db=db)
            return None, {"ok": True, "skipped": True, "reason": f"transcript_http_{status_code}"}
        raise

    utterances = transcript_resp.get("results", transcript_resp.get("data", []))
    logger.info("Utterances: %d", len(utterances))
    if not utterances:
        logger.warning("Empty transcript for meeting_id=%s, marking error", meeting_id)
        mark_download_error(meeting_id, "empty transcript", db=db)
        return None, {"ok": True, "skipped": True, "reason": "empty_transcript"}
    return utterances, None


def _persist_imported_document(meeting_id: str, meeting, utterances, db, set_stage=None) -> str:
    if set_stage:
        set_stage("preparing transcript")
    title = meeting.get("name") or f"TL;DV {meeting_id}"
    text, speaker_ranges = _format_transcript(meeting, utterances)
    dialog_date, _ = _parse_happened_at(meeting.get("happenedAt", ""))

    root_folder_id = get_root_folder_id()
    if set_stage:
        set_stage("authorizing Google Drive")
    drive = _build_drive()
    if set_stage:
        set_stage("selecting destination folder")
    target_folder_id = _detect_target_folder_id(
        db, meeting, utterances, drive, root_folder_id, gemini_json=call_gemini_json
    )

    if set_stage:
        set_stage("authorizing Google Docs")
    docs = _build_docs()
    if set_stage:
        set_stage("creating Google Doc")
    doc_id, modified_time = _create_google_doc(drive, target_folder_id, title, text, speaker_ranges, docs=docs)
    logger.info("Created Google Doc: doc_id=%s", doc_id)

    speakers = _get_speakers(utterances)
    if set_stage:
        set_stage("saving transcript status")
    saved = save_tldv_index(doc_id, meeting_id, {
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
    if not saved:
        raise _CanonicalIndexConflict()
    logger.info("Saved to Firestore: doc_id=%s", doc_id)

    if speakers:
        if set_stage:
            set_stage("saving speaker metadata")
        increment_folder_speakers(target_folder_id, speakers, db=db)

    if doc_id == f"tldv-{meeting_id}":
        return doc_id
    delete_queued_placeholder(meeting_id, db=db)
    return doc_id


def _enqueue_vector_sync(doc_id: str) -> bool:
    vector_sync_url = os.environ.get("VECTOR_SYNC_URL", "")
    vector_sync_queue = os.environ.get("VECTOR_SYNC_QUEUE", "")

    if vector_sync_url and vector_sync_queue:
        enqueue_task(
            queue_name=vector_sync_queue,
            task_id=doc_id,
            url=f"{vector_sync_url}/sync/doc/{doc_id}",
        )
        logger.info("Queued vector sync: doc_id=%s", doc_id)
        return True
    else:
        logger.warning("VECTOR_SYNC_URL or VECTOR_SYNC_QUEUE not set, skipping sync queue")
        return False


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
    progress = _ImportProgress("opening Firestore")
    try:
        db = firestore.Client()
        progress.set("reading import state")
        saved_record = _find_saved_record(meeting_id, db)
        if saved_record is not None:
            return _dispatch_saved_record(saved_record, meeting_id, db)

        progress.set("claiming import")
        if not claim_tldv_import(meeting_id, db=db):
            record = get_transcript_record(f"tldv-{meeting_id}", db=db)
            if is_tldv_unknown_write_outcome(record):
                return _failure_response("Google Docs write outcome is unknown; reconcile the saved document before retrying.")
            saved_record = _find_saved_record(meeting_id, db)
            if saved_record is not None:
                return _dispatch_saved_record(saved_record, meeting_id, db)
            if record is None:
                return _failure_response("TL;DV import state could not be claimed.")
            return JSONResponse({"ok": False, "error": "TL;DV import is already in progress."}, status_code=503)

        progress.set("fetching meeting")
        return _run_new_import(meeting_id, db, progress)
    except Exception as exc:
        if "db" not in locals():
            logger.error("TL;DV import failed during %s (%s)", progress.stage, type(exc).__name__)
            return _failure_response(_safe_failure(progress.stage, exc))
        return _settle_import_failure(meeting_id, progress, exc, db)
