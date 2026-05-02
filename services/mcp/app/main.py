import json
import os
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from core.utils.logging import configure_logging
from services.mcp.app.mcp.handler import handle_mcp_request

configure_logging()

import logging
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_TOKENINFO_URL = "https://www.googleapis.com/oauth2/v2/tokeninfo"


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _oauth_enabled() -> bool:
    return bool(os.environ.get("ALLOWED_EMAIL"))


async def _check_token(request: Request) -> bool:
    if not _oauth_enabled():
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[7:]
    async with httpx.AsyncClient() as client:
        resp = await client.get(_GOOGLE_TOKENINFO_URL, params={"access_token": token})
        if resp.status_code != 200:
            return False
        email = resp.json().get("email")
    return email == os.environ.get("ALLOWED_EMAIL")


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/.well-known/oauth-protected-resource")
@app.get("/.well-known/oauth-protected-resource/mcp")
async def oauth_protected_resource(request: Request):
    base = _base_url(request)
    return JSONResponse({
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
    })


@app.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server(request: Request):
    base = _base_url(request)
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "scopes_supported": ["openid", "email", "profile"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
    })


@app.post("/token")
async def token(request: Request):
    body = await request.body()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _GOOGLE_TOKEN_URL,
            content=body,
            headers={"Content-Type": request.headers.get("Content-Type", "application/x-www-form-urlencoded")},
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/authorize")
async def authorize(request: Request):
    params = dict(request.query_params)
    logger.info("authorize request", extra={"params": params})
    if "scope" not in params:
        params["scope"] = "openid email profile"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{query}")


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    if not await _check_token(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Invalid JSON body"},
            status_code=400,
        )

    if "id" not in body:
        return JSONResponse(content={}, status_code=202)

    try:
        response = await handle_mcp_request(body)
    except Exception as e:
        return JSONResponse(
            content={"error": "Internal MCP error", "details": str(e)},
            status_code=500,
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
            },
        )

    return JSONResponse(content=response)
