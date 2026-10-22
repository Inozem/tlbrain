import json
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from core.utils.logging import configure_logging
from services.mcp.app import oauth
from services.mcp.app.mcp.handler import handle_mcp_request
from services.mcp.app.mcp.request_context import (
    MCPRequestContext,
    reset_request_context,
    set_request_context,
)
from core.integrations.firestore.mcp_sessions import (
    SessionUnavailableError,
    create_session,
    get_session,
)

configure_logging()

app = FastAPI()
app.include_router(oauth.router)
app.add_exception_handler(oauth.OAuthError, oauth.error_response)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)

@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    identity = oauth.request_identity(request)
    if identity is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401, headers={
            "WWW-Authenticate": f'Bearer resource_metadata="{oauth.base_url(request)}/.well-known/oauth-protected-resource/mcp"',
        })

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Invalid JSON body"},
            status_code=400,
        )

    if "id" not in body:
        return JSONResponse(content={}, status_code=202)

    session_id = request.headers.get("Mcp-Session-Id")
    session = None
    session_state = "absent"
    if session_id is not None:
        try:
            session = get_session(
                None,
                session_id,
                caller_binding=identity.caller_binding,
                now=datetime.now(timezone.utc),
            )
            session_state = "active"
        except (SessionUnavailableError, ValueError):
            session_state = "invalid"

    context_token = set_request_context(MCPRequestContext(
        identity=identity,
        session_state=session_state,
        session_id=session_id if session_state == "active" else None,
        session=session,
    ))
    try:
        response = await handle_mcp_request(body)
    except Exception as e:
        return JSONResponse(
            content={"error": "Internal MCP error", "details": str(e)},
            status_code=500,
        )
    finally:
        reset_request_context(context_token)

    params = body.get("params") if isinstance(body, dict) else None
    is_add_dialog = (
        body.get("method") == "tools/call" and isinstance(params, dict)
        and params.get("name") == "add_dialog"
    )
    if is_add_dialog and session_state == "absent":
        return JSONResponse(content={"error": "An initialized MCP session is required"}, status_code=400)
    if is_add_dialog and session_state == "invalid":
        return JSONResponse(content={"error": "The MCP session is unavailable"}, status_code=404)

    response_session_id = session_id if session_state == "active" else None
    if body.get("method") == "initialize" and "error" not in response:
        protocol_version = response.get("result", {}).get("protocolVersion")
        if isinstance(protocol_version, str) and protocol_version:
            initialized = create_session(
                None,
                caller_binding=identity.caller_binding,
                protocol_version=protocol_version,
                now=datetime.now(timezone.utc),
            )
            response_session_id = initialized.session_id

    accept = request.headers.get("accept", "")

    if "text/event-stream" in accept:
        async def event_stream():
            yield f"data: {json.dumps(response)}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if response_session_id is not None:
            headers["Mcp-Session-Id"] = response_session_id
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    headers = {"Mcp-Session-Id": response_session_id} if response_session_id is not None else None
    return JSONResponse(content=response, headers=headers)
