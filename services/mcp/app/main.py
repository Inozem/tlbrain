import asyncio
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from services.mcp.app.mcp.handler import handle_mcp_request

app = FastAPI()

active_sessions = {}


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

    response = await handle_mcp_request(body)
    return JSONResponse(content=response)


@app.get("/sse")
async def sse_endpoint():
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = True

    async def event_generator():
        hello_event = {
            "session_id": session_id,
            "messages_url": f"/messages/{session_id}"
        }

        yield f"data: {json.dumps(hello_event)}\n\n"

        while True:
            await asyncio.sleep(15)

            if session_id not in active_sessions:
                break

            yield ":\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/messages/{session_id}")
async def messages_endpoint(session_id: str, request: Request):
    if session_id not in active_sessions:
        return JSONResponse(
            content={"error": "Invalid session"},
            status_code=404
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Invalid JSON body"},
            status_code=400
        )

    response = await handle_mcp_request(body)

    return JSONResponse(content=response)
