"""Offline OAuth browser flow using the real Firestore transaction boundary."""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from core.utils.logging import configure_logging
from services.mcp.app import main, oauth, oauth_store as store
from tests.services.mcp.test_contract import _run


REDIRECT = "http://127.0.0.1:8123/callback?existing=1"
RESOURCE = "http://mcp.test/mcp"
VERIFIER = "a" * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()


@pytest.fixture
def connection(monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-google-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-google-secret")
    monkeypatch.delenv("K_SERVICE", raising=False)
    calls = []
    identity = {"email": "allowed@example.test", "email_verified": True}

    async def upstream(request):
        calls.append(request)
        if identity.get("_network_error"):
            raise httpx.ConnectError("synthetic upstream failure", request=request)
        if str(request.url) == oauth.GOOGLE_TOKEN_URL:
            return httpx.Response(identity.get("_token_status", 200), json={"access_token": "synthetic-provider-access", "refresh_token": "synthetic-provider-refresh"})
        assert str(request.url) == oauth.GOOGLE_USERINFO_URL
        return httpx.Response(identity.get("_identity_status", 200), json=identity)

    original = httpx.AsyncClient
    client = original(transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test")
    monkeypatch.setattr(oauth.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(upstream), **kwargs))
    yield client, calls, identity
    _run(client.aclose())


def request(connection, method, path, **kwargs):
    return _run(getattr(connection[0], method)(path, **kwargs))


def register(connection, **metadata):
    response = request(connection, "post", "/register", json={"client_name": "Codex test", "redirect_uris": [REDIRECT], **metadata})
    assert response.status_code == 201, response.text
    return response.json()["client_id"]


def navigation_target(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert "location" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "form-action 'none'" in response.headers["content-security-policy"]
    target = html.unescape(re.search(r'<a href="([^"]+)">Continue</a>', response.text)[1])
    refresh = html.unescape(re.search(r'<meta http-equiv="refresh" content="([^"]+)">', response.text)[1])
    assert refresh == f"0; url={target}"
    return target


def begin(connection, **params):
    client_id = params.pop("client_id", None) or register(connection)
    arguments = {
        "client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256",
        "state": "synthetic-client-state", "resource": RESOURCE, **params,
    }
    response = request(connection, "get", "/authorize", params=arguments)
    assert response.status_code == 200, response.text
    consent = re.search(r'name="consent" value="([^"]+)"', response.text)[1]
    assert not connection[0].cookies.get(oauth.COOKIE)
    approved = request(connection, "post", "/oauth/consent", data={"consent": consent, "decision": "approve"})
    query = parse_qs(urlsplit(navigation_target(approved)).query)
    return client_id, query["state"][0], query


def code_flow(connection):
    client_id, state, google = begin(connection)
    response = request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    assert response.status_code == 303, response.text
    result = parse_qs(urlsplit(response.headers["location"]).query)
    assert result["state"] == ["synthetic-client-state"]
    assert result["iss"] == ["http://mcp.test"]
    assert result["existing"] == ["1"]
    return client_id, result["code"][0]


def redeem(connection, client_id, code, **params):
    return request(connection, "post", "/token", data={
        "grant_type": "authorization_code", "client_id": client_id,
        "code": code, "code_verifier": VERIFIER, "redirect_uri": REDIRECT, "resource": RESOURCE, **params,
    })


def mcp(connection, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return request(connection, "post", "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}, headers=headers)


def test_codex_registration_browser_login_pkce_and_refresh(connection, firestore_fake):
    client_id, code = code_flow(connection)
    response = redeem(connection, client_id, code)
    assert response.status_code == 200
    tokens = response.json()
    assert tokens["token_type"] == "Bearer"
    assert tokens["expires_in"] == 3600
    assert tokens["scope"] == "openid email profile"
    assert response.headers["cache-control"] == "no-store"
    assert mcp(connection, tokens["access_token"]).status_code == 200
    assert mcp(connection, "synthetic-provider-access").status_code == 401
    assert redeem(connection, client_id, code).json()["error"] == "invalid_grant"
    refreshed = request(connection, "post", "/token", data={
        "grant_type": "refresh_token", "client_id": client_id,
        "refresh_token": tokens["refresh_token"], "resource": RESOURCE,
    })
    assert refreshed.status_code == 200
    assert refreshed.json()["refresh_token"] != tokens["refresh_token"]
    assert mcp(connection, refreshed.json()["access_token"]).status_code == 200
    assert request(connection, "post", "/token", data={
        "grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"],
    }).json()["error"] == "invalid_grant"
    docs = firestore_fake.documents
    assert all(path.startswith("mcp_oauth/") for path in docs)
    serialized = repr(docs)
    assert tokens["access_token"] not in serialized
    assert tokens["refresh_token"] not in serialized
    assert "synthetic-provider-access" not in serialized
    assert "synthetic-provider-refresh" not in serialized
    assert "synthetic-google-secret" not in serialized


@pytest.mark.parametrize("metadata", [
    {"redirect_uris": ["https://client.test/callback#fragment"]},
    {"redirect_uris": ["http://evil.test/callback"]},
    {"redirect_uris": ["https://user:password@client.test/callback"]},
    {"redirect_uris": ["https://*.test/callback"]},
    {"redirect_uris": []}, {"redirect_uris": [None]}, {"redirect_uris": "https://client.test/callback"},
    {"token_endpoint_auth_method": "client_secret_post"},
    {"grant_types": ["client_credentials"]}, {"grant_types": None}, {"grant_types": [{}]},
    {"response_types": ["token"]}, {"scope": None}, {"client_name": []},
])
def test_invalid_registration_is_rejected(connection, metadata, firestore_fake):
    response = request(connection, "post", "/register", json={"redirect_uris": [REDIRECT], **metadata})
    assert response.status_code == 400
    assert firestore_fake.documents == {}


@pytest.mark.parametrize("uri", ["https://client.test/callback", "http://localhost:8000/callback", "http://[::1]:8000/callback"])
def test_exact_https_and_loopback_registration(connection, uri):
    assert register(connection, redirect_uris=[uri])


@pytest.mark.parametrize("changes", [
    {"redirect_uri": "http://127.0.0.1:8123/other"}, {"client_id": "unregistered"},
    {"response_type": "token"}, {"code_challenge_method": "plain"}, {"code_challenge": "short"},
    {"resource": "http://other.test/mcp"}, {"scope": "drive.read"}, {"scope": ""},
])
def test_authorize_validates_client_redirect_pkce_scope_and_resource(connection, changes):
    client_id = register(connection)
    response = request(connection, "get", "/authorize", params={
        "client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": RESOURCE, **changes,
    })
    if "client_id" in changes or "redirect_uri" in changes:
        assert response.status_code == 400
    else:
        assert response.status_code == 303
        assert parse_qs(urlsplit(response.headers["location"]).query)["iss"] == ["http://mcp.test"]


def test_google_receives_server_client_and_callback_with_independent_pkce(connection):
    client_id, state, query = begin(connection)
    assert query["client_id"] == ["synthetic-google-client"]
    assert query["redirect_uri"] == ["http://mcp.test/oauth/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] != [CHALLENGE]
    assert query["state"] != ["synthetic-client-state"]
    request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    [exchange, userinfo] = connection[1]
    form = parse_qs(exchange.content.decode())
    assert form["client_id"] == ["synthetic-google-client"]
    assert form["client_secret"] == ["synthetic-google-secret"]
    assert form["redirect_uri"] == ["http://mcp.test/oauth/callback"]
    assert base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest()).rstrip(b"=").decode() == query["code_challenge"][0]
    assert client_id not in exchange.content.decode()
    assert userinfo.headers["authorization"] == "Bearer synthetic-provider-access"


@pytest.mark.parametrize("identity", [
    {"email": "other@example.test", "email_verified": True},
    {"email": "allowed@example.test", "email_verified": False},
    {"email_verified": True},
])
def test_wrong_or_unverified_google_account_cannot_get_code(connection, identity):
    connection[2].clear()
    connection[2].update(identity)
    _, state, _ = begin(connection)
    response = request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert response.status_code == 303
    assert query["error"] == ["access_denied"]
    assert query["iss"] == ["http://mcp.test"]
    assert "code" not in query


def test_browser_binding_and_state_replay(connection):
    _, state, _ = begin(connection)
    cookie = connection[0].cookies.get(oauth.COOKIE)
    connection[0].cookies.clear()
    assert request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"}).status_code == 400
    assert connection[1] == []
    connection[0].cookies.set(oauth.COOKIE, cookie)
    assert request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"}).status_code == 303
    assert request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"}).status_code == 400


def test_google_cancellation_preserves_state_and_issuer(connection):
    _, state, _ = begin(connection)
    response = request(connection, "get", "/oauth/callback", params={"state": state, "error": "access_denied"})
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["error"] == ["access_denied"]
    assert query["state"] == ["synthetic-client-state"]
    assert query["iss"] == ["http://mcp.test"]
    assert connection[1] == []


def test_consent_is_required_escaped_and_bound_to_browser(connection, firestore_fake):
    client_id = register(connection, client_name="<script>alert(1)</script>")
    page = request(connection, "get", "/authorize", params={
        "client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": RESOURCE,
    })
    assert "<script>" not in page.text
    assert "&lt;script&gt;" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert not connection[0].cookies.get(oauth.COOKIE)
    assert not any(path.split("/")[1].startswith(("state-", "code-", "access-", "refresh-")) for path in firestore_fake.documents)
    consent = re.search(r'name="consent" value="([^"]+)"', page.text)[1]
    assert request(connection, "post", "/oauth/consent", data={"consent": consent, "decision": "approve"}, headers={"origin": "https://evil.test"}).status_code == 400
    denied = request(connection, "post", "/oauth/consent", data={"consent": consent, "decision": "deny"})
    target = navigation_target(denied)
    assert "error=access_denied" in target
    assert "code=" not in target
    assert request(connection, "post", "/oauth/consent", data={"consent": consent, "decision": "approve"}).status_code == 400


def test_cloud_run_sets_host_cookie_only_after_explicit_consent(connection, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "synthetic-cloud-run-service")
    connection[0].base_url = "https://mcp.test"
    client_id = register(connection)
    page = request(connection, "get", "/authorize", params={
        "client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": "https://mcp.test/mcp",
    })
    assert page.status_code == 200
    assert connection[0].cookies.get("__Host-tlbrain_consent")
    assert not connection[0].cookies.get("__Host-tlbrain_oauth")
    consent = re.search(r'name="consent" value="([^"]+)"', page.text)[1]
    approved = request(connection, "post", "/oauth/consent", data={"consent": consent, "decision": "approve"})
    headers = approved.headers.get_list("set-cookie")
    [cookie] = [header for header in headers if header.startswith("__Host-tlbrain_oauth=")]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/" in cookie
    assert "Domain=" not in cookie
    assert parse_qs(urlsplit(navigation_target(approved)).query)["redirect_uri"] == ["https://mcp.test/oauth/callback"]


def test_consent_form_preserves_origin_on_cloud_run(connection, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "synthetic-cloud-run-service")
    connection[0].base_url = "https://mcp.test"
    client_id = register(connection)
    page = request(connection, "get", "/authorize", params={
        "client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": "https://mcp.test/mcp",
    })
    assert page.status_code == 200
    assert page.headers["referrer-policy"] == "same-origin"
    consent = re.search(r'name="consent" value="([^"]+)"', page.text)[1]
    for origin in ("null", "https://evil.test", "http://mcp.test", "https://mcp.test.evil.test"):
        rejected = request(connection, "post", "/oauth/consent", data={
            "consent": consent, "decision": "approve",
        }, headers={"origin": origin})
        assert rejected.status_code == 400
        assert rejected.json()["error_description"] == "The consent request origin does not match."
        assert store.load("consent", consent) is not None
    approved = request(connection, "post", "/oauth/consent", data={
        "consent": consent, "decision": "approve",
    }, headers={"origin": "https://mcp.test"})
    assert approved.headers["referrer-policy"] == "no-referrer"
    google = urlsplit(navigation_target(approved))
    assert f"{google.scheme}://{google.netloc}{google.path}" == oauth.GOOGLE_AUTH_URL
    assert parse_qs(google.query)["redirect_uri"] == ["https://mcp.test/oauth/callback"]


@pytest.mark.parametrize("redirect", [
    "https://claude.ai/api/mcp/auth_callback", REDIRECT,
    "http://localhost:8123/callback", "http://[::1]:8123/callback",
    "https://client.test/callback;segment,part?existing=1&other=value",
])
def test_consent_navigation_preserves_registered_callback(connection, redirect):
    client_id = register(connection, redirect_uris=[redirect])
    page = request(connection, "get", "/authorize", params={
        "client_id": client_id, "redirect_uri": redirect, "response_type": "code",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": RESOURCE,
    })
    assert page.status_code == 200
    directives = dict(
        directive.strip().split(None, 1)
        for directive in page.headers["content-security-policy"].split(";")
    )
    assert directives == {
        "default-src": "'none'",
        "form-action": "'self'",
        "frame-ancestors": "'none'",
        "base-uri": "'none'",
    }
    consent = re.search(r'name="consent" value="([^"]+)"', page.text)[1]
    denied = request(connection, "post", "/oauth/consent", data={
        "consent": consent, "decision": "deny",
    }, headers={"origin": "http://mcp.test"})
    target = urlsplit(navigation_target(denied))
    assert target.netloc == urlsplit(redirect).netloc
    assert target.path == urlsplit(redirect).path
    query = parse_qs(target.query)
    assert query["error"] == ["access_denied"]
    for key, value in parse_qs(urlsplit(redirect).query).items():
        assert query[key] == value


@pytest.mark.parametrize("changes", [
    {"code_verifier": "b" * 64}, {"redirect_uri": "http://127.0.0.1:8123/other"},
    {"resource": "http://other.test/mcp"},
])
def test_invalid_exchange_does_not_consume_legitimate_code(connection, changes):
    client_id, code = code_flow(connection)
    assert redeem(connection, client_id, code, **changes).status_code == 400
    assert redeem(connection, client_id, code).status_code == 200


def test_code_is_bound_to_registered_client(connection):
    client_id, code = code_flow(connection)
    assert redeem(connection, register(connection), code).status_code == 400
    assert redeem(connection, client_id, code).status_code == 200


@pytest.mark.parametrize("kind", ["code", "access", "refresh"])
def test_expired_oauth_records_are_rejected(connection, kind):
    client_id, code = code_flow(connection)
    if kind == "code":
        secret = code
    else:
        tokens = redeem(connection, client_id, code).json()
        secret = tokens[f"{kind}_token"]
    value = store.load(kind, secret)
    store.save(kind, secret, store.record(value, -1))
    if kind == "code":
        assert redeem(connection, client_id, code).status_code == 400
    elif kind == "access":
        assert mcp(connection, secret).status_code == 401
    else:
        assert request(connection, "post", "/token", data={"grant_type": "refresh_token", "client_id": client_id, "refresh_token": secret}).status_code == 400


def test_allowlist_change_revokes_access_and_refresh(connection, monkeypatch):
    client_id, code = code_flow(connection)
    tokens = redeem(connection, client_id, code).json()
    monkeypatch.setenv("ALLOWED_EMAIL", "other@example.test")
    assert mcp(connection, tokens["access_token"]).status_code == 401
    assert request(connection, "post", "/token", data={"grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"]}).status_code == 400


@pytest.mark.parametrize("configured,email,allowed", [
    ("first@example.test", "first@example.test", True),
    ("first@example.test,second@example.test", "first@example.test", True),
    ("first@example.test,second@example.test", "second@example.test", True),
    (" First@Example.Test, ,SECOND@example.test,first@example.test, ", "FIRST@example.test", True),
    (" First@Example.Test, ,SECOND@example.test,first@example.test, ", "second@example.test", True),
    ("first@example.test,second@example.test", "other@example.test", False),
    ("first@example.test,second@example.test", "first@example.test.evil", False),
    ("first@example.test,second@example.test", "irst@example.test", False),
    ("*@example.test", "first@example.test", False),
    ("first@example.test", "", False),
    ("first@example.test", None, False),
])
def test_allowed_email_accepts_exact_comma_separated_members(monkeypatch, configured, email, allowed):
    monkeypatch.setenv("ALLOWED_EMAIL", configured)
    assert oauth._allowed(email) is allowed


def test_multiple_accounts_can_login_and_removal_revokes_only_removed_account(connection, monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL", " allowed@example.test,SECOND@example.test, ")
    sessions = {}
    for email in ("allowed@example.test", "second@example.test"):
        connection[2]["email"] = email
        client_id, code = code_flow(connection)
        tokens = redeem(connection, client_id, code).json()
        assert mcp(connection, tokens["access_token"]).status_code == 200
        refreshed = request(connection, "post", "/token", data={
            "grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"],
        })
        assert refreshed.status_code == 200
        sessions[email] = client_id, refreshed.json()

    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    for email, (client_id, tokens) in sessions.items():
        retained = email == "allowed@example.test"
        assert mcp(connection, tokens["access_token"]).status_code == (200 if retained else 401)
        refreshed = request(connection, "post", "/token", data={
            "grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"],
        })
        assert refreshed.status_code == (200 if retained else 400)


@pytest.mark.parametrize("configured", [",", ", , ", " ,\t, "])
def test_empty_list_entries_do_not_disable_auth(connection, monkeypatch, configured):
    monkeypatch.setenv("ALLOWED_EMAIL", configured)
    assert oauth.enabled()
    assert mcp(connection).status_code == 401
    _, state, _ = begin(connection)
    response = request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["error"] == ["access_denied"]
    assert "code" not in query


@pytest.mark.parametrize("scope", ["openid email profile", "email"])
def test_refresh_accepts_same_or_narrower_scope(connection, scope):
    client_id, code = code_flow(connection)
    tokens = redeem(connection, client_id, code).json()
    response = request(connection, "post", "/token", data={
        "grant_type": "refresh_token", "client_id": client_id,
        "refresh_token": tokens["refresh_token"], "scope": scope,
    })
    assert response.status_code == 200
    assert response.json()["scope"] == scope
    assert mcp(connection, response.json()["access_token"]).status_code == 200


@pytest.mark.parametrize("changes", [{"resource": "https://other.test/mcp"}, {"issuer": "https://other.test"}, {"scope": "drive.read"}, {"scope": ""}])
def test_mcp_checks_token_audience_issuer_and_scope(connection, changes):
    client_id, code = code_flow(connection)
    token = redeem(connection, client_id, code).json()["access_token"]
    value = store.load("access", token)
    store.save("access", token, {**value, **changes})
    assert mcp(connection, token).status_code == 401


@pytest.mark.parametrize("failure", [{"_token_status": 400}, {"_identity_status": 500}, {"_network_error": True}])
def test_google_failures_never_issue_credentials(connection, failure, firestore_fake):
    connection[2].update(failure)
    _, state, _ = begin(connection)
    response = request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert response.status_code == 303
    assert "error" in query and "code" not in query
    assert not any(path.split("/")[1].startswith(("consent-", "code-", "access-", "refresh-")) for path in firestore_fake.documents)


def test_duplicate_callback_parameters_do_not_consume_state(connection):
    _, state, _ = begin(connection)
    response = request(connection, "get", "/oauth/callback", params=[("state", state), ("state", "other-state"), ("code", "synthetic-google-code")])
    assert response.status_code == 400
    assert connection[1] == []
    assert store.load("state", state) is not None


def test_oauth_storage_preserves_drive_cursor_and_transcript_records(connection, firestore_fake):
    db = store.get_client()
    db.collection("tokens").document("drive_sync").set({"page_token": "synthetic-drive-cursor"})
    db.collection("transcript_index").document("document-1").set({"status": "synced", "content_hash": "synthetic-hash"})
    previous = firestore_fake.documents
    client_id, code = code_flow(connection)
    token = redeem(connection, client_id, code).json()["access_token"]
    second_client = type(connection[0])(transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test")
    response = _run(second_client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "initialize"}, headers={"Authorization": f"Bearer {token}"}))
    _run(second_client.aclose())
    assert response.status_code == 200
    assert {path: firestore_fake.documents[path] for path in previous} == previous


def test_unauthorized_challenge_supports_oauth_discovery(connection):
    response = mcp(connection)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer resource_metadata="http://mcp.test/.well-known/oauth-protected-resource/mcp"'


def test_oauth_requires_google_configuration(connection, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET")
    assert request(connection, "post", "/register", json={"redirect_uris": [REDIRECT]}).status_code == 503


def test_oauth_secrets_are_not_logged_or_forwarded_from_client(connection, capsys):
    configure_logging()
    client_id, state, query = begin(connection, access_token="synthetic-injected-secret")
    assert "access_token" not in query
    request(connection, "get", "/oauth/callback", params={"state": state, "code": "synthetic-google-code"})
    logs = capsys.readouterr().err
    for secret in (state, "synthetic-client-state", "synthetic-injected-secret", "synthetic-google-code", "synthetic-google-secret", "synthetic-provider-access"):
        assert secret not in logs


def test_store_consumes_only_once_under_concurrent_exchanges():
    store.save("code", "one-time-code", store.record({"valid": True}, 120))
    def exchange(_):
        return store.exchange("code", "one-time-code", lambda value: value["valid"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(exchange, range(2)))
    assert sum(value is not None for value in results) == 1
