import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.google_drive.docs_reader import read_google_doc
from core.google_drive.drive_client import get_file_parent_folder_id
from core.google_drive.firestore import ensure_imported, get_folder_by_id, mark_error
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


@app.post("/sync/doc/{doc_id}")
async def sync_doc_endpoint(doc_id: str):
    root_folder_id = get_root_folder_id()

    existing = load_index(doc_id)
    should_delete = False
    parent_id = None

    file_name = None
    try:
        parent_id, is_trashed, file_name = get_file_parent_folder_id(doc_id)
        if is_trashed:
            logger.warning("File is trashed in Drive, deleting: %s", doc_id)
            should_delete = True
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning("File not found in Drive (404), deleting: %s", doc_id)
            should_delete = True
        else:
            raise

    if not should_delete:
        if not parent_id or not get_folder_by_id(parent_id):
            if existing:
                logger.warning("File moved outside TLBrain, deleting: %s", doc_id)
                should_delete = True
            else:
                logger.warning("Could not determine folder for doc, skipping: %s", doc_id)
                return JSONResponse(content={"status": "ok", "result": "skipped_no_folder"})

    if should_delete:
        if existing:
            delete_by_doc_id(doc_id, root_folder_id)
            delete_index(doc_id)
        return JSONResponse(content={"status": "ok", "result": "deleted"})

    raw_text = read_google_doc(doc_id)
    if not is_valid_format(raw_text):
        logger.warning("File does not match TLBrain format, skipping: %s", doc_id)
        if existing:
            mark_error(doc_id, "invalid TLBrain document format", error_stage="invalid_format")
        return JSONResponse(content={"status": "ok", "result": "skipped_invalid_format"})

    ensure_imported(doc_id, parent_id, source_file=file_name or "")

    try:
        result = process_one(doc_id, parent_id, root_folder_id, raw_text=raw_text)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )
