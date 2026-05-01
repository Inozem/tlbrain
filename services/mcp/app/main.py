import json
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from core.utils.logging import configure_logging
from services.mcp.app.mcp.handler import handle_mcp_request

configure_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Invalid JSON body"},
            status_code=400
        )

    if "id" not in body:
        return JSONResponse(content={}, status_code=202)

    try:
        response = await handle_mcp_request(body)
    except Exception as e:
        return JSONResponse(
            content={"error": "Internal MCP error", "details": str(e)},
            status_code=500
        )

    accept = request.headers.get("accept", "")

    if "text/event-stream" in accept:
        async def event_stream():
            yield f"data: {json.dumps(response)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Mcp-Session-Id": str(uuid.uuid4()),
            }
        )

    return JSONResponse(content=response)
