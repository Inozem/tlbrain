"""MCP OAuth authorization server with Google login and public PKCE clients."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import logging
import os
import re
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from services.mcp.app import oauth_store as store


router = APIRouter()
logger = logging.getLogger(__name__)
SCOPES = ["openid", "email", "profile"]
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
COOKIE = "tlbrain_oauth"
CONSENT_COOKIE = "tlbrain_consent"
LOGIN_TTL = 600
CODE_TTL = 120
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 24 * 3600


class OAuthError(Exception):
    def __init__(self, error: str, description: str, status: int = 400):
        self.error = error
        self.description = description
        self.status = status


async def error_response(request: Request, exc: OAuthError):
    return JSONResponse(
        {"error": exc.error, "error_description": exc.description},
        status_code=exc.status,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


def base_url(request: Request) -> str:
    url = request.base_url
    if os.environ.get("K_SERVICE"):
        url = url.replace(scheme="https")
    return str(url).rstrip("/")


def enabled() -> bool:
    return bool(os.environ.get("ALLOWED_EMAIL", "").strip())


def _configured() -> None:
    if not enabled() or not all(os.environ.get(key) for key in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET")):
        raise OAuthError("temporarily_unavailable", "Google OAuth is not configured.", 503)


def _allowed(email: str | None) -> bool:
    allowed_emails = {
        value.strip().casefold()
        for value in os.environ.get("ALLOWED_EMAIL", "").split(",")
        if value.strip()
    }
    return isinstance(email, str) and email.casefold() in allowed_emails


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def _binding(cookie: str) -> str:
    return hashlib.sha256(cookie.encode()).hexdigest()


def _client(client_id: str) -> dict:
    value = store.load("client", client_id)
    if value is None:
        raise OAuthError("invalid_client", "Register the OAuth client again.")
    return value


def _scope(value: str) -> str:
    if not isinstance(value, str):
        raise OAuthError("invalid_scope", "OAuth scope must be a string.")
    scopes = value.split()
    if not scopes or not set(scopes).issubset(SCOPES):
        raise OAuthError("invalid_scope", "Supported scopes are openid, email and profile.")
    return " ".join(dict.fromkeys(scopes))


def _redirect_uri(value: str) -> bool:
    try:
        uri = urlsplit(value)
        port = uri.port
    except ValueError:
        return False
    loopback = uri.hostname in ("localhost", "127.0.0.1", "::1")
    return bool(
        len(value) <= 2048 and not any(ord(char) <= 32 or ord(char) == 127 for char in value)
        and uri.hostname and not uri.username and not uri.password
        and not uri.fragment and "*" not in value
        and (uri.scheme == "https" or (uri.scheme == "http" and loopback and port is not None))
    )


def _resource(request: Request, value: str | None) -> str:
    expected = f"{base_url(request)}/mcp"
    if value is not None and value != expected:
        raise OAuthError("invalid_target", "The requested resource is not this MCP server.")
    return expected


async def _form(request: Request) -> dict[str, str]:
    body = await request.body()
    if len(body) > 16384 or not request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        raise OAuthError("invalid_request", "Expected a URL-encoded OAuth request.")
    try:
        parsed = parse_qs(body.decode(), keep_blank_values=True, max_num_fields=32)
    except (UnicodeError, ValueError):
        raise OAuthError("invalid_request", "Invalid OAuth request.") from None
    if any(len(values) != 1 for values in parsed.values()):
        raise OAuthError("invalid_request", "Duplicate OAuth parameters are not allowed.")
    return {key: values[0] for key, values in parsed.items()}


def _cookie_name(base: str, name: str) -> str:
    return f"__Host-{name}" if base.startswith("https://") else name


def _browser_matches(request: Request, value: dict, name: str = COOKIE) -> bool:
    cookie = request.cookies.get(_cookie_name(base_url(request), name), "")
    return bool(cookie) and hmac.compare_digest(value["browser"], _binding(cookie))


def _client_redirect(value: dict, **params: str) -> RedirectResponse:
    uri = urlsplit(value["redirect_uri"])
    query = parse_qs(uri.query, keep_blank_values=True)
    for key in ("code", "state", "iss", "error", "error_description"):
        query.pop(key, None)
    query.update({key: [item] for key, item in params.items()})
    query["state"] = [value["state"]]
    query["iss"] = [value["issuer"]]
    response = RedirectResponse(urlunsplit(uri._replace(query=urlencode(query, doseq=True))), status_code=303)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    for name in (COOKIE, CONSENT_COOKIE):
        response.delete_cookie(_cookie_name(value["issuer"], name), path="/")
    return response


def _query(request: Request):
    params = request.query_params
    if any(len(params.getlist(key)) != 1 for key in params):
        raise OAuthError("invalid_request", "Duplicate OAuth parameters are not allowed.")
    return params


@router.get("/.well-known/oauth-protected-resource")
@router.get("/.well-known/oauth-protected-resource/mcp")
async def protected_resource(request: Request):
    base = base_url(request)
    return {"resource": f"{base}/mcp", "authorization_servers": [base]}


@router.get("/.well-known/oauth-authorization-server")
async def authorization_server(request: Request):
    base = base_url(request)
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "token_endpoint_auth_methods_supported": ["none"],
        "authorization_response_iss_parameter_supported": True,
        "scopes_supported": SCOPES,
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
    }


@router.post("/register")
async def register(request: Request):
    _configured()
    body = await request.body()
    if len(body) > 16384:
        raise OAuthError("invalid_client_metadata", "Client metadata is too large.")
    try:
        data = await request.json()
    except ValueError:
        raise OAuthError("invalid_client_metadata", "Expected JSON client metadata.") from None
    if not isinstance(data, dict):
        raise OAuthError("invalid_client_metadata", "Expected JSON client metadata.")
    redirects = data.get("redirect_uris")
    if (
        not isinstance(redirects, list) or not 1 <= len(redirects) <= 10
        or any(not isinstance(uri, str) or not _redirect_uri(uri) for uri in redirects)
    ):
        raise OAuthError("invalid_redirect_uri", "Use exact HTTPS or loopback redirect URIs.")
    if data.get("token_endpoint_auth_method", "none") != "none":
        raise OAuthError("invalid_client_metadata", "Public OAuth clients must use PKCE without a client secret.")
    grants = data.get("grant_types", ["authorization_code"])
    if (
        not isinstance(grants, list) or "authorization_code" not in grants
        or any(not isinstance(grant, str) for grant in grants)
        or not set(grants).issubset({"authorization_code", "refresh_token"})
        or data.get("response_types", ["code"]) != ["code"]
    ):
        raise OAuthError("invalid_client_metadata", "Only the authorization-code and refresh-token flows are supported.")
    name = data.get("client_name", "MCP client")
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise OAuthError("invalid_client_metadata", "Provide a short client name.")
    client_id = secrets.token_urlsafe(32)
    metadata = {
        "client_name": name,
        "redirect_uris": list(dict.fromkeys(redirects)),
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "scope": _scope(data.get("scope", " ".join(SCOPES))),
        "client_id_issued_at": int(time.time()),
    }
    store.save("client", client_id, metadata)
    return JSONResponse({"client_id": client_id, **metadata}, status_code=201, headers={"Cache-Control": "no-store"})


def _authorization(request: Request) -> dict:
    params = _query(request)
    client_id = params.get("client_id", "")
    client = _client(client_id)
    redirect = params.get("redirect_uri", "")
    if redirect not in client["redirect_uris"]:
        raise OAuthError("invalid_request", "The redirect URI is not registered for this client.")
    state = params.get("state", "")
    if len(state) > 2048:
        raise OAuthError("invalid_request", "OAuth state is too large.")
    return {
        "client_id": client_id, "client_name": client["client_name"], "redirect_uri": redirect,
        "state": state, "issuer": base_url(request), "registered_scope": client["scope"],
    }


def _authorization_options(request: Request, value: dict) -> None:
    params = request.query_params
    if params.get("response_type") != "code":
        raise OAuthError("unsupported_response_type", "Only the authorization-code flow is supported.")
    if params.get("code_challenge_method") != "S256" or not re.fullmatch(r"[A-Za-z0-9_-]{43}", params.get("code_challenge", "")):
        raise OAuthError("invalid_request", "S256 PKCE is required.")
    scope = _scope(params.get("scope", " ".join(SCOPES)))
    if not set(scope.split()).issubset(value.pop("registered_scope").split()):
        raise OAuthError("invalid_scope", "The requested scopes were not registered.")
    value.update(scope=scope, resource=_resource(request, params.get("resource")), code_challenge=params["code_challenge"])


@router.get("/authorize")
async def authorize(request: Request):
    _configured()
    value = _authorization(request)
    try:
        _authorization_options(request, value)
    except OAuthError as error:
        return _client_redirect(value, error=error.error, error_description=error.description)
    consent = secrets.token_urlsafe(32)
    cookie = secrets.token_urlsafe(32)
    value["browser"] = _binding(cookie)
    store.save("consent", consent, store.record(value, LOGIN_TTL))
    logger.info("authorize request")
    response = _consent_page(consent, value)
    response.set_cookie(
        _cookie_name(value["issuer"], CONSENT_COOKIE), cookie, max_age=LOGIN_TTL,
        secure=value["issuer"].startswith("https://"), httponly=True, samesite="lax",
    )
    return response


def _google_redirect(value: dict) -> RedirectResponse:
    cookie = secrets.token_urlsafe(32)
    google_state = secrets.token_urlsafe(32)
    google_verifier = secrets.token_urlsafe(48)
    base = value["issuer"]
    value.update(google_verifier=google_verifier, browser=_binding(cookie), consent_approved=True)
    store.save("state", google_state, store.record(value, LOGIN_TTL))
    query = urlencode({
        "client_id": os.environ["GOOGLE_CLIENT_ID"], "redirect_uri": f"{base}/oauth/callback",
        "response_type": "code", "scope": " ".join(SCOPES), "state": google_state,
        "code_challenge": _challenge(google_verifier), "code_challenge_method": "S256",
        "prompt": "select_account",
    })
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{query}", status_code=303)
    response.delete_cookie(_cookie_name(base, CONSENT_COOKIE), path="/")
    response.set_cookie(_cookie_name(base, COOKIE), cookie, max_age=LOGIN_TTL, secure=base.startswith("https://"), httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def _google_email(code: str, value: dict) -> str | None:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(GOOGLE_TOKEN_URL, data={
            "grant_type": "authorization_code", "code": code,
            "client_id": os.environ["GOOGLE_CLIENT_ID"], "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "redirect_uri": f"{value['issuer']}/oauth/callback", "code_verifier": value["google_verifier"],
        })
        if response.status_code != 200:
            return None
        token_data = response.json()
        token = token_data.get("access_token") if isinstance(token_data, dict) else None
        if not isinstance(token, str) or not token:
            return None
        identity = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {token}"})
        if identity.status_code != 200:
            return None
        info = identity.json()
        return info.get("email") if isinstance(info, dict) and info.get("email_verified") is True else None


def _consent_page(consent: str, value: dict) -> HTMLResponse:
    name = html.escape(value["client_name"])
    redirect = html.escape(value["redirect_uri"])
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Connect to TLBrain</title>
<h1>Connect {name} to TLBrain?</h1><p>This connection will have access to TLBrain document search and operations.</p>
<p>To sign in, Google will ask for access to your email address and basic profile (openid, email, profile).</p>
<p>Return address: <code>{redirect}</code></p><form method="post" action="{html.escape(value['issuer'])}/oauth/consent">
<input type="hidden" name="consent" value="{consent}"><button name="decision" value="approve">Allow</button>
<button name="decision" value="deny">Cancel</button></form></html>"""
    return HTMLResponse(body, headers={
        "Cache-Control": "no-store", "Referrer-Policy": "same-origin", "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
    })


def _consent_navigation(redirect: RedirectResponse) -> HTMLResponse:
    target = html.escape(redirect.headers["location"])
    page = HTMLResponse(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Continue</title>'
        f'<meta http-equiv="refresh" content="0; url={target}">'
        f'<p><a href="{target}">Continue</a></p></html>',
        headers={
            "Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'",
        },
    )
    page.raw_headers.extend((key, value) for key, value in redirect.raw_headers if key == b"set-cookie")
    return page


@router.get("/oauth/callback")
async def callback(request: Request):
    _configured()
    params = _query(request)
    state = params.get("state", "")
    value = store.exchange("state", state, lambda data: _browser_matches(request, data) and data["issuer"] == base_url(request) and data.get("consent_approved") is True)
    if value is None:
        raise OAuthError("invalid_request", "The login session expired or does not match this browser.")
    if params.get("error") or not params.get("code"):
        return _client_redirect(value, error="access_denied", error_description="Google login was not completed.")
    try:
        email = await _google_email(params["code"], value)
    except (httpx.HTTPError, ValueError):
        return _client_redirect(value, error="temporarily_unavailable", error_description="Google login is temporarily unavailable.")
    if not _allowed(email):
        return _client_redirect(value, error="access_denied", error_description="This Google account is not allowed to access TLBrain.")
    code = secrets.token_urlsafe(32)
    value.pop("google_verifier")
    value["email"] = email
    store.save("code", code, store.record(value, CODE_TTL))
    return _client_redirect(value, code=code)


@router.post("/oauth/consent")
async def consent(request: Request):
    _configured()
    params = await _form(request)
    if params.get("decision") not in ("approve", "deny"):
        raise OAuthError("invalid_request", "Choose whether to allow this connection.")
    origin = request.headers.get("origin")
    uri = urlsplit(base_url(request))
    if origin and origin != urlunsplit((uri.scheme, uri.netloc, "", "", "")):
        raise OAuthError("invalid_request", "The consent request origin does not match.")
    value = store.load("consent", params.get("consent", ""))
    if value is None or not _browser_matches(request, value, CONSENT_COOKIE) or value["issuer"] != base_url(request):
        raise OAuthError("invalid_request", "The consent session expired or does not match this browser.")
    value = store.exchange("consent", params["consent"], lambda data: _browser_matches(request, data, CONSENT_COOKIE) and data["issuer"] == base_url(request))
    if value is None:
        raise OAuthError("invalid_request", "This consent session has already been used.")
    if params["decision"] != "approve":
        return _consent_navigation(_client_redirect(value, error="access_denied", error_description="Access was not granted."))
    return _consent_navigation(_google_redirect(value))


def _token_records(value: dict) -> tuple[dict, list[tuple[str, str, dict]]]:
    access = secrets.token_urlsafe(32)
    refresh = secrets.token_urlsafe(32)
    data = {key: value[key] for key in ("email", "client_id", "issuer", "resource", "scope")}
    return {
        "access_token": access, "token_type": "Bearer", "expires_in": ACCESS_TTL,
        "refresh_token": refresh, "scope": value["scope"],
    }, [
        ("access", access, store.record(data, ACCESS_TTL)),
        ("refresh", refresh, store.record(data, REFRESH_TTL)),
    ]


def _matches_code(data: dict, params: dict) -> bool:
    verifier = params.get("code_verifier", "")
    return bool(
        params.get("redirect_uri") == data["redirect_uri"]
        and re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", verifier)
        and hmac.compare_digest(data["code_challenge"], _challenge(verifier))
    )


def _redeem(request: Request, params: dict, kind: str, secret: str) -> dict:
    client_id = params.get("client_id", "")
    _client(client_id)
    value = store.load(kind, secret)
    if value is None:
        raise OAuthError("invalid_grant", "The OAuth grant expired or has already been used.")
    resource = _resource(request, params.get("resource"))
    scope = _scope(params["scope"]) if "scope" in params and kind == "refresh" else value["scope"]
    def accepts(data):
        return (
            data["client_id"] == client_id and data["resource"] == resource
            and data["issuer"] == base_url(request) and _allowed(data["email"])
            and set(scope.split()).issubset(data["scope"].split())
            and (kind != "code" or _matches_code(data, params))
        )
    response, replacements = _token_records({**value, "scope": scope})
    if store.exchange(kind, secret, accepts, replacements) is None:
        raise OAuthError("invalid_grant", "The OAuth grant does not match this client, resource or PKCE verifier.")
    return response


@router.post("/token")
async def token(request: Request):
    _configured()
    params = await _form(request)
    if params.get("client_secret") or request.headers.get("authorization"):
        raise OAuthError("invalid_client", "This public client uses PKCE without a client secret.")
    grant = params.get("grant_type")
    if grant == "authorization_code":
        result = _redeem(request, params, "code", params.get("code", ""))
    elif grant == "refresh_token":
        result = _redeem(request, params, "refresh", params.get("refresh_token", ""))
    else:
        raise OAuthError("unsupported_grant_type", "Use authorization_code or refresh_token.")
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


def authorized(request: Request) -> bool:
    if not enabled():
        return True
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    value = store.load("access", token)
    return bool(value and _allowed(value["email"]) and value["issuer"] == base_url(request)
                and value["resource"] == f"{base_url(request)}/mcp" and value["scope"].split()
                and set(value["scope"].split()).issubset(SCOPES))
