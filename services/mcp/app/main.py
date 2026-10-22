import json
import os
from datetime import datetime, timezone
from urllib.parse import urlsplit

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
from core.integrations.firestore.client import get_client
from core.integrations.google_auth.oidc import (
    OIDCConfigurationError,
    verify_google_id_token,
)
from services.mcp.app.dialog_worker import (
    InvalidTask,
    RecoveryBindingUnavailable,
    process_dialog_operation,
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


def _dialog_recovery_ready() -> bool:
    """Require a deploy-configured HTTPS callback and matching dispatch identity."""
    callback_url = os.environ.get("INTERNAL_DIALOG_OPERATIONS_URL")
    queue = os.environ.get("DIALOG_OPERATIONS_QUEUE")
    audience = (os.environ.get("CLOUD_TASKS_OIDC_AUDIENCE")
                or os.environ.get("TASK_OIDC_AUDIENCE"))
    service_account = (os.environ.get("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL")
                       or os.environ.get("TASK_OIDC_SERVICE_ACCOUNT_EMAIL")
                       or os.environ.get("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL")
                       or os.environ.get("TASK_SERVICE_ACCOUNT_EMAIL"))
    expected_audience = os.environ.get("OIDC_AUDIENCE")
    expected_account = os.environ.get("OIDC_SERVICE_ACCOUNT_EMAIL")
    try:
        parsed = urlsplit(callback_url or "")
        valid_url = (
            parsed.scheme == "https" and bool(parsed.hostname)
            and parsed.username is None and parsed.password is None
            and parsed.path == "/internal/dialog-operations"
            and not parsed.query and not parsed.fragment
        )
    except ValueError:
        valid_url = False
    return bool(
        valid_url and queue and queue.strip() and audience and audience.strip()
        and service_account and service_account.strip()
        and expected_audience and audience == expected_audience
        and expected_account and service_account == expected_account
    )


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


@app.post("/internal/dialog-operations")
async def dialog_operations_endpoint(request: Request):
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        identity = verify_google_id_token(token.strip())
    except OIDCConfigurationError:
        return JSONResponse({"error": "Recovery authentication unavailable"}, status_code=503)
    if identity is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not identity.authorized:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    try:
        payload = json.loads(
            (await request.body()).decode("utf-8"), object_pairs_hook=_unique_json_object,
        )
    except Exception:
        return JSONResponse({"error": "Invalid task"}, status_code=400)
    try:
        result = process_dialog_operation(get_client(), payload)
    except InvalidTask:
        return JSONResponse({"error": "Invalid task"}, status_code=400)
    except RecoveryBindingUnavailable:
        return JSONResponse({"error": "Recovery unavailable"}, status_code=503)
    except Exception:
        return JSONResponse({"error": "Recovery unavailable"}, status_code=503)
    return JSONResponse({"status": result.status}, status_code=200)


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

    params = body.get("params") if isinstance(body, dict) else None
    is_add_dialog = (
        body.get("method") == "tools/call" and isinstance(params, dict)
        and params.get("name") == "add_dialog"
    )
    if is_add_dialog and not _dialog_recovery_ready():
        return JSONResponse(
            content={"error": "Dialog recovery is not configured"}, status_code=503,
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
