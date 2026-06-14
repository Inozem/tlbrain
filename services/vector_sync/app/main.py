import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.google_drive.docs_reader import read_google_doc
from core.google_drive.drive_client import get_file_parent_folder_id, get_folder_info
from core.google_drive.firestore import ensure_imported, mark_error, reconcile_client_record, update_client_speakers
from core.parsing.parser import is_valid_format
from core.qdrant.setup import ensure_collection
from core.qdrant.writer import delete_by_doc_id
from core.utils.logging import configure_logging
from services.vector_sync.app.index_store import delete_index, load_index
from services.vector_sync.app.processor import process_one

configure_logging()
logger = logging.getLogger(__name__)
ensure_collection()

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/client/rename")
async def rename_client_endpoint(payload: dict):
    """Folder task: reconcile the clients collection for an in-place folder rename.

    Copies clients/{old} → clients/{new} (carrying speakers), deletes the old record.
    Idempotent. Enqueues nothing — per-doc file tasks fix Qdrant and transcript_index.
    """
    old = (payload.get("old_client_name") or "").strip()
    new = (payload.get("new_client_name") or "").strip()
    folder_id = (payload.get("folder_id") or "").strip()
    if not old or not new or not folder_id:
        return JSONResponse(
            content={"status": "error", "details": "old_client_name, new_client_name and folder_id are required"},
            status_code=400,
        )
    reconcile_client_record(old, new, folder_id)
    return JSONResponse(content={"status": "ok", "old": old, "new": new})


@app.post("/sync/doc/{doc_id}")
async def sync_doc_endpoint(doc_id: str):
    root_folder_id = get_root_folder_id()

    existing = load_index(doc_id)
    should_delete = False
    folder_id = None

    file_name = None
    try:
        folder_id, is_trashed, file_name = get_file_parent_folder_id(doc_id)
        if is_trashed:
            logger.warning("File is trashed in Drive, deleting: %s", doc_id)
            should_delete = True
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning("File not found in Drive (404), deleting: %s", doc_id)
            should_delete = True
        else:
            raise

    client_name = None
    if not should_delete:
        if folder_id:
            # Live Drive folder is the single source of truth for client_name. Validity =
            # the folder is a direct child of ROOT; never consult the (rename-stale) clients map.
            folder_name, folder_parent = get_folder_info(folder_id)
            if folder_parent == root_folder_id:
                client_name = folder_name

        if not client_name:
            if existing:
                # File was in our system but folder is no longer a TLBrain client → moved out → delete
                logger.warning("File moved outside TLBrain, deleting: %s", doc_id)
                should_delete = True
            else:
                # No record, unknown folder → skip
                logger.warning("Could not determine client for doc, skipping: %s", doc_id)
                return JSONResponse(content={"status": "ok", "result": "skipped_no_client"})

    if should_delete:
        if existing:
            speakers = existing.get("speakers", [])
            client_name_existing = existing.get("client_name", "")
            delete_by_doc_id(doc_id, root_folder_id)
            if speakers and client_name_existing:
                update_client_speakers(client_name_existing, speakers, delta=-1)
            delete_index(doc_id)
        return JSONResponse(content={"status": "ok", "result": "deleted"})

    raw_text = read_google_doc(doc_id)
    if not is_valid_format(raw_text):
        logger.warning("File does not match TLBrain format, skipping: %s", doc_id)
        if existing:
            mark_error(doc_id, "invalid TLBrain document format", error_stage="invalid_format")
        return JSONResponse(content={"status": "ok", "result": "skipped_invalid_format"})

    ensure_imported(doc_id, client_name, folder_id or "", source_file=file_name or "")

    try:
        result = process_one(doc_id, client_name, root_folder_id, folder_id=folder_id, raw_text=raw_text)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )
