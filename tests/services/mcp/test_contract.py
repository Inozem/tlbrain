from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from starlette.requests import Request

from services.mcp.app import main
from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


BASELINE = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_baseline.json"


@pytest.fixture(autouse=True)
def disable_auth_by_default(monkeypatch):
    monkeypatch.delenv("ALLOWED_EMAIL", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)


def _run(coroutine):
    import sniffio

    context_token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
    finally:
        sniffio.current_async_library_cvar.reset(context_token)
        coroutine.close()
    raise AssertionError("in-process ASGI coroutine unexpectedly suspended")


def _request(method: str, path: str, **kwargs):
    async def request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as client:
            return await getattr(client, method)(path, **kwargs)

    return _run(request())


def _request_with_client(client, method: str, path: str, **kwargs):
    return _run(getattr(client, method)(path, **kwargs))


def _rpc(method: str, request_id: int | str | None = 1, params: dict | None = None) -> dict:
    request = {"jsonrpc": "2.0", "method": method}
    if request_id is not ...:
        request["id"] = request_id
    if params is not None:
        request["params"] = params
    return request


def test_tools_list_matches_static_baseline():
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    response = _request("post", "/mcp", json=_rpc("tools/list", "baseline-tools"))
    assert response.status_code == 200
    assert response.json() == expected
    assert len(expected["result"]["tools"]) == 13


def test_initialize_and_jsonrpc_id_types(monkeypatch):
    monkeypatch.setenv("VERSION", "synthetic-version")
    expected_result = {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "tlbrain", "version": "synthetic-version"},
    }
    for request_id in (7, "correlation-id"):
        response = _request("post", "/mcp", json=_rpc("initialize", request_id))
        assert response.json() == {"jsonrpc": "2.0", "id": request_id, "result": expected_result}


def test_notifications_and_invalid_json_route_semantics():
    no_id = _request("post", "/mcp", json=_rpc("initialize", ...))
    assert no_id.status_code == 202
    assert no_id.json() == {}

    initialized = _request("post", "/mcp", json=_rpc("notifications/initialized", "notification-id"))
    assert initialized.status_code == 200
    assert initialized.json() == {}

    null_id = _request("post", "/mcp", json=_rpc("initialize", None))
    assert null_id.status_code == 200
    assert null_id.json() == {
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tlbrain", "version": "latest"},
        },
    }

    invalid = _request("post", "/mcp", content="{", headers={"content-type": "application/json"})
    assert invalid.status_code == 400
    assert invalid.json() == {"error": "Invalid JSON body"}


def test_json_and_sse_envelopes():
    body = _rpc("initialize", "stable-id")
    json_response = _request("post", "/mcp", json=body)
    assert json_response.headers["content-type"].startswith("application/json")
    expected = json_response.json()

    request_body = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    request = Request({
        "type": "http", "method": "POST", "path": "/mcp", "raw_path": b"/mcp",
        "query_string": b"", "headers": [(b"accept", b"text/event-stream")],
        "scheme": "http", "server": ("mcp.test", 80), "client": ("127.0.0.1", 123),
    }, receive)
    response = _run(main.mcp_endpoint(request))
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", response.headers["mcp-session-id"])
    event = _run(response.body_iterator.__anext__())
    assert event == f"data: {json.dumps(expected)}\n\n"


def test_protocol_and_operation_errors(monkeypatch):
    invalid_request = _request("post", "/mcp", json={"method": "initialize", "id": "bad"})
    assert invalid_request.json()["error"]["code"] == -32600
    assert invalid_request.json()["error"]["message"] == "Invalid Request"

    unknown_method = _request("post", "/mcp", json=_rpc("unknown/method", "method-id"))
    assert unknown_method.json() == {
        "jsonrpc": "2.0", "id": "method-id", "error": {"code": -32601, "message": "Method not found"}
    }

    invalid_tool = _request("post", "/mcp", json=_rpc("tools/call", 9, {"name": "unknown"}))
    assert invalid_tool.json() == {
        "jsonrpc": "2.0", "id": 9, "error": {"code": -32602, "message": "Invalid tool"}
    }

    operation_error = _request("post", "/mcp", json=_rpc("tools/call", 10, {"name": "move_folder", "arguments": {}}))
    assert operation_error.json() == {
        "jsonrpc": "2.0", "id": 10, "error": {"code": -32602, "message": "folder_path is required"}
    }

    missing_config = _request("post", "/mcp", json=_rpc("tools/call", 11, {"name": "import_all_transcripts"}))
    assert missing_config.json() == {
        "jsonrpc": "2.0", "id": 11,
        "error": {"code": -32603, "message": "TLDV_RECONCILIATION_URL is not configured"},
    }

    def fail_retrieval(**kwargs):
        raise RuntimeError("synthetic retrieval failure")

    monkeypatch.setattr(handler, "get_transcripts", fail_retrieval)
    failed_operation = _request("post", "/mcp", json=_rpc("tools/call", 12, {
        "name": "get_transcript", "arguments": {"doc_id": "synthetic-doc"}
    }))
    assert failed_operation.json() == {
        "jsonrpc": "2.0", "id": 12,
        "error": {
            "code": -32603,
            "message": "Transcript retrieval failed",
            "details": "synthetic retrieval failure",
        },
    }


def test_unicode_tool_content_round_trip(monkeypatch):
    payload = "שלום, réunion 🌍"
    monkeypatch.setattr(handler, "get_transcripts", lambda **kwargs: ([{"doc_id": "d1", "text": payload}], {
        "truncated": False, "total_matches": 1, "returned_segments": 1
    }))
    response = _request("post", "/mcp", json=_rpc("tools/call", "unicode", {
        "name": "get_transcript", "arguments": {"doc_id": "d1"}
    }))
    text = response.json()["result"]["content"][0]["text"]
    assert payload in text
    assert json.loads(text)["segments"][0]["text"] == payload


def test_existing_tool_argument_defaults_and_conversions(monkeypatch):
    transcript_calls = []

    def get_transcripts(**kwargs):
        transcript_calls.append(kwargs)
        return [], {"truncated": False, "total_matches": 0, "returned_segments": 0}

    monkeypatch.setattr(handler, "get_transcripts", get_transcripts)
    request = JSONRPCRequest(**_rpc("tools/call", "get", {"name": "get_transcript", "arguments": {"limit": 0}}))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert transcript_calls[-1]["limit"] == 1
    request = JSONRPCRequest(**_rpc("tools/call", "get-string", {"name": "get_transcript", "arguments": {"limit": "0"}}))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert transcript_calls[-1]["limit"] == "0"

    recent_calls = []
    monkeypatch.setattr(handler, "list_transcripts", lambda **kwargs: recent_calls.append(kwargs) or {
        "transcripts": [], "total": 0, "returned": 0
    })
    request = JSONRPCRequest(**_rpc("tools/call", "recent", {
        "name": "list_recent_transcripts", "arguments": {"limit": 0, "offset": "3"}
    }))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert recent_calls[-1]["limit"] == 20
    assert recent_calls[-1]["offset"] == 3
    request = JSONRPCRequest(**_rpc("tools/call", "recent-string", {
        "name": "list_recent_transcripts", "arguments": {"limit": "7", "offset": "0"}
    }))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert recent_calls[-1]["limit"] == 7
    assert recent_calls[-1]["offset"] == 0

    monkeypatch.setenv("TLDV_RECONCILIATION_URL", "http://reconciliation.test/run")
    posted = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"queued": 0, "remaining": 0}

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: posted.append((args, kwargs)) or Response())
    request = JSONRPCRequest(**_rpc("tools/call", "import", {
        "name": "import_all_transcripts", "arguments": {"limit": 0, "since": ""}
    }))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert posted[-1][1]["json"] == {"limit": 10, "full_scan": True}
    request = JSONRPCRequest(**_rpc("tools/call", "import-string", {
        "name": "import_all_transcripts", "arguments": {"limit": "4", "since": "2025-01-02"}
    }))
    _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))
    assert posted[-1][1]["json"] == {"limit": "4", "full_scan": True, "since": "2025-01-02"}


def test_oauth_discovery_and_authorize_params():
    protected = _request("get", "/.well-known/oauth-protected-resource")
    assert protected.json() == {
        "resource": "http://mcp.test/mcp", "authorization_servers": ["http://mcp.test"]
    }
    protected_mcp = _request("get", "/.well-known/oauth-protected-resource/mcp")
    assert protected_mcp.json() == protected.json()
    discovery = _request("get", "/.well-known/oauth-authorization-server")
    assert discovery.json() == {
        "issuer": "http://mcp.test",
        "authorization_endpoint": "http://mcp.test/authorize",
        "token_endpoint": "http://mcp.test/token",
        "scopes_supported": ["openid", "email", "profile"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
    }

    default_scope = _request("get", "/authorize?client_id=synthetic&state=flow-state")
    assert default_scope.status_code == 307
    default_params = parse_qs(urlparse(default_scope.headers["location"]).query)
    assert default_params == {"client_id": ["synthetic"], "state": ["flow-state"], "scope": ["openid email profile"]}

    supplied_scope = _request("get", "/authorize?client_id=synthetic&scope=openid&state=flow-state")
    supplied_params = parse_qs(urlparse(supplied_scope.headers["location"]).query)
    assert supplied_params == {"client_id": ["synthetic"], "scope": ["openid"], "state": ["flow-state"]}

    empty_scope = _request("get", "/authorize?client_id=synthetic&scope=&state=flow-state")
    empty_params = parse_qs(urlparse(empty_scope.headers["location"]).query, keep_blank_values=True)
    assert empty_params == {"client_id": ["synthetic"], "scope": [""], "state": ["flow-state"]}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/.well-known/oauth-protected-resource", {
            "resource": "https://mcp.test/mcp", "authorization_servers": ["https://mcp.test"]
        }),
        ("/.well-known/oauth-protected-resource/mcp", {
            "resource": "https://mcp.test/mcp", "authorization_servers": ["https://mcp.test"]
        }),
        ("/.well-known/oauth-authorization-server", {
            "issuer": "https://mcp.test",
            "authorization_endpoint": "https://mcp.test/authorize",
            "token_endpoint": "https://mcp.test/token",
            "scopes_supported": ["openid", "email", "profile"],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
        }),
    ],
)
def test_oauth_discovery_uses_https_on_cloud_run(monkeypatch, path, expected):
    monkeypatch.setenv("K_SERVICE", "synthetic-service")
    response = _request("get", path)
    assert response.status_code == 200
    assert response.json() == expected


def test_oauth_discovery_preserves_local_scheme_and_origin_with_forwarded_headers(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    request = Request({
        "type": "http", "method": "GET", "path": "/", "raw_path": b"/",
        "query_string": b"", "headers": [
            (b"host", b"mcp.test:8443"),
            (b"forwarded", b"proto=https;host=spoof.test"),
            (b"x-forwarded-proto", b"https"),
            (b"x-forwarded-host", b"spoof.test"),
        ],
        "scheme": "http", "server": ("mcp.test", 8443), "client": ("127.0.0.1", 123),
        "root_path": "/prefix",
    })
    assert main._base_url(request) == "http://mcp.test:8443/prefix"
    monkeypatch.setenv("K_SERVICE", "synthetic-service")
    assert main._base_url(request) == "https://mcp.test:8443/prefix"
    monkeypatch.delenv("K_SERVICE", raising=False)

    secure_request = Request({
        "type": "http", "method": "GET", "path": "/", "raw_path": b"/",
        "query_string": b"", "headers": [(b"host", b"mcp.test")],
        "scheme": "https", "server": ("mcp.test", 443), "client": ("127.0.0.1", 123),
    })
    assert main._base_url(secure_request) == "https://mcp.test"


def test_oauth_discovery_cloud_run_ignores_forged_forwarded_origin(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "synthetic-service")
    response = _request("get", "/.well-known/oauth-authorization-server", headers={
        "forwarded": "proto=http;host=spoof.test",
        "x-forwarded-proto": "http",
        "x-forwarded-host": "spoof.test",
    })
    assert response.json()["issuer"] == "https://mcp.test"


def test_token_proxy_forwards_body_content_type_status_and_json(monkeypatch):
    captured = []

    async def mock_request(request):
        captured.append(request)
        return httpx.Response(201, json={"access_token": "synthetic-access", "token_type": "Bearer"})

    transport = httpx.MockTransport(mock_request)
    original = main.httpx.AsyncClient

    def client_with_mock_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    outer = original(transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test")
    monkeypatch.setattr(main.httpx, "AsyncClient", client_with_mock_transport)
    body = b"grant_type=authorization_code&code=synthetic-code"
    response = _request_with_client(outer, "post", "/token", content=body, headers={"content-type": "application/x-www-form-urlencoded; charset=utf-8"})
    _run(outer.aclose())
    assert response.status_code == 201
    assert response.json() == {"access_token": "synthetic-access", "token_type": "Bearer"}
    assert len(captured) == 1
    assert captured[0].url == "https://oauth2.googleapis.com/token"
    assert captured[0].content == body
    assert captured[0].headers["content-type"] == "application/x-www-form-urlencoded; charset=utf-8"


@pytest.mark.parametrize(
    ("email", "status", "authorization", "accepted"),
    [
        ("allowed@example.test", 200, "Bearer synthetic-good", True),
        ("allowed@example.test", 400, "Bearer synthetic-bad", False),
        (None, 200, "Bearer synthetic-missing-email", False),
        ("other@example.test", 200, "Bearer synthetic-wrong-email", False),
        ("allowed@example.test", 200, None, False),
    ],
)
def test_mcp_oauth_enabled_branches(monkeypatch, email, status, authorization, accepted):
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    calls = []

    async def mock_request(request):
        calls.append(request)
        return httpx.Response(status, json={"email": email} if email is not None else {"expires_in": 3600})

    transport = httpx.MockTransport(mock_request)
    original = main.httpx.AsyncClient

    def client_with_mock_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    outer = original(transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test")
    monkeypatch.setattr(main.httpx, "AsyncClient", client_with_mock_transport)
    headers = {"authorization": authorization} if authorization else {}
    response = _request_with_client(outer, "post", "/mcp", json=_rpc("initialize", "auth-id"), headers=headers)
    _run(outer.aclose())
    if accepted:
        assert response.status_code == 200
        assert response.json()["id"] == "auth-id"
    else:
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}
    if authorization:
        assert len(calls) == 1
        assert calls[0].url == "https://www.googleapis.com/oauth2/v2/tokeninfo?access_token=" + authorization[7:].replace(" ", "%20")
    else:
        assert calls == []


def test_mcp_tokeninfo_is_checked_on_every_request(monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    calls = []

    async def mock_request(request):
        calls.append(request)
        return httpx.Response(200, json={"email": "allowed@example.test"})

    transport = httpx.MockTransport(mock_request)
    original = main.httpx.AsyncClient

    def client_with_mock_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    outer = original(transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test")
    monkeypatch.setattr(main.httpx, "AsyncClient", client_with_mock_transport)
    headers = {"authorization": "Bearer synthetic-no-cache"}
    first = _request_with_client(outer, "post", "/mcp", json=_rpc("initialize", "first"), headers=headers)
    second = _request_with_client(outer, "post", "/mcp", json=_rpc("initialize", "second"), headers=headers)
    _run(outer.aclose())

    assert first.status_code == second.status_code == 200
    assert [first.json()["id"], second.json()["id"]] == ["first", "second"]
    assert len(calls) == 2
    assert calls[0].url == calls[1].url
    assert str(calls[0].url).startswith("https://www.googleapis.com/oauth2/v2/tokeninfo?")


def test_mcp_auth_is_disabled_when_allowed_email_is_unset(monkeypatch):
    monkeypatch.delenv("ALLOWED_EMAIL", raising=False)
    response = _request("post", "/mcp", json=_rpc("initialize", "disabled-auth"))
    assert response.status_code == 200
    assert response.json()["id"] == "disabled-auth"
