from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.config import get_root_folder_id
from core.qdrant.setup import ensure_collection
from core.utils.logging import configure_logging
from services.vector_sync.app.index_store import load_index
from services.vector_sync.app.processor import process_one
from services.vector_sync.app.runner import run_sync

configure_logging()
ensure_collection()

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/sync")
async def sync_endpoint():
    try:
        result = run_sync()
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )


@app.post("/sync/doc/{doc_id}")
async def sync_doc_endpoint(doc_id: str):
    existing = load_index(doc_id)
    if not existing:
        return JSONResponse(
            content={"status": "error", "details": "doc_id not found in index"},
            status_code=404,
        )

    client_name = existing.get("client_name", "")
    root_folder_id = get_root_folder_id()

    try:
        result = process_one(doc_id, client_name, root_folder_id)
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "details": str(e)},
            status_code=500,
        )
