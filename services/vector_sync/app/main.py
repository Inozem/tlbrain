from fastapi import FastAPI
from fastapi.responses import JSONResponse
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.google_drive.drive_client import get_file_parent_folder_id
from core.google_drive.firestore import get_client_name_by_folder_id
from core.qdrant.setup import ensure_collection
from core.qdrant.writer import delete_by_doc_id
from core.utils.logging import configure_logging
from services.vector_sync.app.index_store import delete_index, load_index
from services.vector_sync.app.processor import process_one

configure_logging()
ensure_collection()

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}



@app.post("/sync/doc/{doc_id}")
async def sync_doc_endpoint(doc_id: str):
    root_folder_id = get_root_folder_id()

    try:
        folder_id = get_file_parent_folder_id(doc_id)
    except HttpError as e:
        if e.resp.status == 404:
            existing = load_index(doc_id)
            if existing:
                delete_by_doc_id(doc_id, root_folder_id)
                delete_index(doc_id)
            return JSONResponse(content={"status": "ok", "result": "deleted"})
        raise

    client_name = get_client_name_by_folder_id(folder_id) if folder_id else None

    if not client_name:
        existing = load_index(doc_id)
        client_name = (existing or {}).get("client_name", "")

    if not client_name:
        return JSONResponse(
            content={"status": "error", "details": "could not determine client for doc"},
            status_code=404,
        )

    try:
        result = process_one(doc_id, client_name, root_folder_id)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )
