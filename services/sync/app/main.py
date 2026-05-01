from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.utils.logging import configure_logging
from services.sync.app.runner import run_sync

configure_logging()

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/sync")
async def sync_endpoint():
    try:
        result = run_sync()

        return JSONResponse(
            content={
                "status": "ok",
                "result": result,
            }
        )

    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "details": str(e),
            },
            status_code=500,
        )
