import logging
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.integrations.google_docs.reader import read_google_doc
from core.integrations.google_drive.files import get_file_parent_folder_id
from core.integrations.firestore.folders import get_folder_by_id
from core.integrations.firestore.transcripts import ensure_imported, load_index
from core.documents import operations as document_operations
from core.integrations.qdrant.setup import ensure_collection
from core.utils.logging import configure_logging
from core.indexing.processor import process_one, process_managed_one

configure_logging()
logger = logging.getLogger(__name__)
ensure_collection()

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/sync/doc/{doc_id}")
async def sync_doc_endpoint(doc_id: str, request: Request):
    raw_body = await request.body()
    managed_reference = None
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return JSONResponse(content={"status": "error", "details": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse(content={"status": "error", "details": "request body must be an object"}, status_code=400)
        if payload.get("target_kind") == "index_dispatch":
            managed_reference = payload
        elif payload.get("target_kind") is None:
            if payload.get("doc_id", doc_id) != doc_id:
                return JSONResponse(content={"status": "error", "details": "route document ID mismatch"}, status_code=400)
        else:
            return JSONResponse(content={"status": "error", "details": "unsupported managed target kind"}, status_code=400)
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
        document_operations.delete_indexed_transcript(doc_id, root_folder_id, bool(existing))
        return JSONResponse(content={"status": "ok", "result": "deleted"})

    raw_text = read_google_doc(doc_id)
    if not document_operations.validate_document_format(doc_id, raw_text, bool(existing)):
        logger.warning("File does not match TLBrain format, skipping: %s", doc_id)
        return JSONResponse(content={"status": "ok", "result": "skipped_invalid_format"})

    ensure_imported(doc_id, parent_id, source_file=file_name or "")

    try:
        if managed_reference is not None:
            result = process_managed_one(
                doc_id, parent_id, root_folder_id, raw_text, managed_reference,
            )
        else:
            result = process_one(doc_id, parent_id, root_folder_id, raw_text=raw_text)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )
