import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from services.mcp.app.mcp.handler import handle_mcp_request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

active_sessions = {}


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/sse")
async def sse_endpoint():
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()

    active_sessions[session_id] = {
        "queue": queue,
        "created": time.time(),
    }

    async def event_generator():
        yield f"event: endpoint\ndata: /messages/{session_id}\n\n"

        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=15)
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/messages/{session_id}")
async def messages_endpoint(session_id: str, request: Request):
    session = active_sessions.get(session_id)

    if not session:
        return JSONResponse(
            content={"error": "Invalid session_id"},
            status_code=404
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Invalid JSON body"},
            status_code=400
        )

    try:
        response = await handle_mcp_request(body)
    except Exception as e:
        return JSONResponse(
            content={"error": "Internal MCP error", "details": str(e)},
            status_code=500
        )

    if response is not None:
        await session["queue"].put(response)

    return JSONResponse(content={}, status_code=202)
