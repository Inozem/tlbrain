"""MCP session identity and request replay at the public ASGI boundary."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import httpx
import pytest

from services.mcp.app import main, oauth_store
from services.mcp.app.mcp.request_context import get_request_context
from tests.services.mcp.test_add_dialog import (
    _initialized_session,
    _install_document_boundary,
    _payload,
    _post,
    _rpc,
    _selector_loop,
)


@pytest.fixture(autouse=True)
def open_mcp(monkeypatch):
    monkeypatch.delenv("ALLOWED_EMAIL", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "pytest-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "pytest-client-secret")


def _dialog(text: str = "Synthetic session replay.") -> dict:
    return {
        "provider": "manual",
        "utterances": [{"speaker": "Ada", "text": text}],
        "title": "Synthetic session test",
        "dialog_date": "2026-01-01",
        "folder_path": [],
    }


def _call(request_id, arguments: dict, session_id: str | None, *, token: str | None = None):
    headers = {}
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return _post(_rpc("tools/call", request_id, {
        "name": "add_dialog", "arguments": arguments,
    }), headers=headers)


def _seed_token(token: str, email: str, *, client_id: str = "synthetic-client"):
    oauth_store.save("access", token, oauth_store.record({
        "email": email,
        "client_id": client_id,
        "issuer": "http://mcp.test",
        "resource": "http://mcp.test/mcp",
        "scope": "openid email profile",
    }, 3600))


def _committed_firestore_writes(firestore_fake):
    return [event for event in firestore_fake.trace() if event.get("outcome") == "committed"]


def test_json_initialize_issues_a_session_header_reused_on_session_aware_response(firestore_fake):
    body = _rpc("initialize", "session-init")
    initialized = _post(body)
    assert initialized.status_code == 200
    session_id = initialized.headers["mcp-session-id"]
    key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    stored = firestore_fake.client_class().collection("mcp_sessions").document(key).get().to_dict()
    assert stored["caller_binding"] == "anonymous:installation"
    assert stored["protocol_version"] == "2024-11-05"
    assert stored["created_at"].tzinfo is not None
    assert session_id not in json.dumps(stored, default=str)

    subsequent = _post(_rpc("tools/list", "session-tools"), headers={"Mcp-Session-Id": session_id})
    assert subsequent.status_code == 200
    assert subsequent.headers["mcp-session-id"] == session_id


def test_sse_initialize_issues_a_session_header_reused_on_session_aware_response():
    body = _rpc("initialize", "session-init-sse")

    async def send_sse():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test"
        ) as client:
            initialized_sse = await client.post(
                "/mcp", json=body, headers={"accept": "text/event-stream"}
            )
            session = initialized_sse.headers["mcp-session-id"]
            subsequent_sse = await client.post(
                "/mcp", json=_rpc("tools/list", "session-tools-sse"), headers={
                    "accept": "text/event-stream", "Mcp-Session-Id": session,
                }
            )
            return initialized_sse, subsequent_sse

    sse, subsequent_sse = asyncio.run(send_sse(), loop_factory=_selector_loop)
    assert sse.status_code == 200
    assert sse.headers["content-type"].startswith("text/event-stream")
    assert sse.headers["mcp-session-id"]
    assert subsequent_sse.status_code == 200
    assert subsequent_sse.headers["mcp-session-id"] == sse.headers["mcp-session-id"]


def test_sessionless_legacy_tool_and_protected_auth_contract_remain_compatible(monkeypatch):
    legacy = _post(_rpc("tools/list", "legacy-sessionless"))
    assert legacy.status_code == 200
    legacy_call = _post(_rpc("tools/call", "legacy-sync-status", {
        "name": "sync_status", "arguments": {},
    }))
    assert legacy_call.status_code == 200
    assert _payload(legacy_call)["total"] == 0

    arbitrary_session = _post(
        _rpc("tools/list", "legacy-arbitrary-session"),
        headers={"Mcp-Session-Id": "untrusted-session"},
    )
    assert arbitrary_session.status_code == 200
    assert "mcp-session-id" not in arbitrary_session.headers

    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    unauthorized_legacy_call = _post(_rpc("tools/call", "protected-sync-status", {
        "name": "sync_status", "arguments": {},
    }))
    assert unauthorized_legacy_call.status_code == 401
    assert unauthorized_legacy_call.headers["www-authenticate"].startswith("Bearer ")
    unauthorized = _post(
        _rpc("initialize", "auth-required"),
        headers={"Mcp-Session-Id": "untrusted-session"},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"].startswith("Bearer ")


def test_add_dialog_requires_initialized_session_before_side_effects(
    monkeypatch, sdk_fakes, firestore_fake,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    before = copy.deepcopy(firestore_fake.documents)
    writes_before = _committed_firestore_writes(firestore_fake)
    response = _call("missing-session", _dialog(), None)
    assert response.status_code == 400
    assert firestore_fake.documents == before
    assert _committed_firestore_writes(firestore_fake) == writes_before
    assert not any(event["boundary"] in {"drive", "docs"} for event in sdk_fakes.trace)


def test_expired_session_is_rejected_before_side_effects(
    monkeypatch, sdk_fakes, firestore_fake, fake_clock,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    firestore_fake.set_clock(fake_clock)
    session_id = _initialized_session()
    before = copy.deepcopy(firestore_fake.documents)
    writes_before = _committed_firestore_writes(firestore_fake)
    fake_clock.advance(seconds=24 * 60 * 60 + 1)

    response = _call("expired-session", _dialog(), session_id)
    assert response.status_code == 404
    assert firestore_fake.documents == before
    assert _committed_firestore_writes(firestore_fake) == writes_before
    assert not any(event["boundary"] in {"drive", "docs"} for event in sdk_fakes.trace)


def test_different_verified_caller_cannot_replay_session_operation(
    monkeypatch, sdk_fakes, firestore_fake,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test,other@example.test")
    _seed_token("synthetic-owner-token", "allowed@example.test")
    _seed_token("synthetic-other-token", "other@example.test")
    initialized = _post(_rpc("initialize", "owner-init"), headers={
        "Authorization": "Bearer synthetic-owner-token",
    })
    session_id = initialized.headers["mcp-session-id"]
    created = _call("owner-operation", _dialog(), session_id, token="synthetic-owner-token")
    assert created.status_code == 200
    before = copy.deepcopy(firestore_fake.documents)
    writes_before = _committed_firestore_writes(firestore_fake)
    trace_before = list(sdk_fakes.trace)

    denied = _call("owner-operation", _dialog(), session_id, token="synthetic-other-token")
    assert "result" not in denied.json()
    assert "operation_id" not in json.dumps(denied.json())
    assert firestore_fake.documents == before
    assert _committed_firestore_writes(firestore_fake) == writes_before
    assert sdk_fakes.trace == trace_before


def test_rotated_access_token_for_same_verified_principal_keeps_session(monkeypatch):
    # Keep OAuth enabled so the two bearer values are independently checked.
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test")
    _seed_token("synthetic-access-before-refresh", "allowed@example.test")
    _seed_token("synthetic-access-after-refresh", "allowed@example.test")
    initialized = _post(_rpc("initialize", "refresh-init"), headers={
        "Authorization": "Bearer synthetic-access-before-refresh",
    })
    assert initialized.status_code == 200
    session_id = initialized.headers["mcp-session-id"]

    continued = _post(_rpc("tools/list", "refresh-continue"), headers={
        "Authorization": "Bearer synthetic-access-after-refresh",
        "Mcp-Session-Id": session_id,
    })
    assert continued.status_code == 200
    assert continued.headers["mcp-session-id"] == session_id


def test_protected_session_binding_uses_verified_email_client_and_resource(firestore_fake, monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL", "allowed@example.test,other@example.test")
    _seed_token("synthetic-owner-token", "Allowed@Example.Test", client_id="client-one")
    _seed_token("synthetic-same-token", "allowed@example.test", client_id="client-one")
    _seed_token("synthetic-other-client", "allowed@example.test", client_id="client-two")
    _seed_token("synthetic-other-user", "other@example.test", client_id="client-one")

    initialized = _post(_rpc("initialize", "identity-init"), headers={
        "Authorization": "Bearer synthetic-owner-token",
    })
    assert initialized.status_code == 200
    session_id = initialized.headers["mcp-session-id"]
    key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    stored = firestore_fake.client_class().collection("mcp_sessions").document(key).get().to_dict()
    expected = "oauth:" + hashlib.sha256(
        "allowed@example.test\0client-one\0http://mcp.test/mcp".encode("utf-8")
    ).hexdigest()
    assert stored["caller_binding"] == expected
    assert "synthetic-owner-token" not in json.dumps(stored, default=str)

    same_caller = _post(_rpc("tools/list", "same-identity"), headers={
        "Authorization": "Bearer synthetic-same-token", "Mcp-Session-Id": session_id,
    })
    other_client = _post(_rpc("tools/list", "other-client"), headers={
        "Authorization": "Bearer synthetic-other-client", "Mcp-Session-Id": session_id,
    })
    other_user = _post(_rpc("tools/list", "other-user"), headers={
        "Authorization": "Bearer synthetic-other-user", "Mcp-Session-Id": session_id,
    })
    assert same_caller.headers["mcp-session-id"] == session_id
    assert "mcp-session-id" not in other_client.headers
    assert "mcp-session-id" not in other_user.headers


def test_cors_exposes_mcp_session_header():
    initialized = _post(_rpc("initialize", "cors-init"), headers={"Origin": "https://client.test"})
    assert initialized.status_code == 200
    assert initialized.headers["access-control-expose-headers"] == "Mcp-Session-Id"


def test_request_context_exposes_identity_and_invalid_session_then_resets(monkeypatch):
    observed = []

    async def inspect_context(body):
        observed.append(get_request_context())
        return {"jsonrpc": "2.0", "id": body["id"], "result": {}}

    monkeypatch.setattr(main, "handle_mcp_request", inspect_context)
    response = _post(_rpc("tools/list", "context-invalid"), headers={
        "Mcp-Session-Id": "untrusted-session",
    })
    assert response.status_code == 200
    assert observed[0].identity.caller_binding == "anonymous:installation"
    assert observed[0].session_state == "invalid"
    assert get_request_context() is None


def test_same_key_and_arguments_replay_one_operation_and_result(monkeypatch, sdk_fakes):
    _install_document_boundary(monkeypatch, sdk_fakes)
    session_id = _initialized_session()
    args = _dialog()
    first = _call("same-key", args, session_id)
    second = _call("same-key", args, session_id)
    first_payload = _payload(first)
    second_payload = _payload(second)
    assert first_payload["operation_id"] == second_payload["operation_id"]
    assert first_payload["doc_id"] == second_payload["doc_id"]
    assert len([event for event in sdk_fakes.trace if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1


def test_same_key_with_changed_arguments_conflicts_without_another_write(
    monkeypatch, sdk_fakes, firestore_fake,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    session_id = _initialized_session()
    first = _call("conflicting-key", _dialog("Original text."), session_id)
    before = list(sdk_fakes.trace)
    documents_before = copy.deepcopy(firestore_fake.documents)
    writes_before = _committed_firestore_writes(firestore_fake)

    conflict = _call("conflicting-key", _dialog("Changed text."), session_id)
    assert conflict.json()["error"]["code"] == -32602
    assert sdk_fakes.trace == before
    assert firestore_fake.documents == documents_before
    assert _committed_firestore_writes(firestore_fake) == writes_before
    assert _payload(first)["operation_id"]


def test_integer_and_string_request_ids_are_distinct_replay_keys(monkeypatch, sdk_fakes):
    _install_document_boundary(monkeypatch, sdk_fakes)
    session_id = _initialized_session()
    integer_result = _payload(_call(7, _dialog(), session_id))
    string_result = _payload(_call("7", _dialog(), session_id))
    assert integer_result["operation_id"] != string_result["operation_id"]
    assert integer_result["doc_id"] != string_result["doc_id"]


def test_concurrent_same_key_replay_performs_one_save(monkeypatch, sdk_fakes):
    _install_document_boundary(monkeypatch, sdk_fakes)
    session_id = _initialized_session()
    body = _rpc("tools/call", "concurrent-key", {
        "name": "add_dialog", "arguments": _dialog(),
    })

    async def send_pair():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test"
        ) as client:
            headers = {"Mcp-Session-Id": session_id}
            return await asyncio.gather(
                client.post("/mcp", json=body, headers=headers),
                client.post("/mcp", json=body, headers=headers),
            )

    first, second = asyncio.run(send_pair(), loop_factory=_selector_loop)
    assert first.status_code == second.status_code == 200
    assert _payload(first)["operation_id"] == _payload(second)["operation_id"]
    assert len([event for event in sdk_fakes.trace if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1


def test_distinct_sessions_with_same_request_id_create_independent_operations(
    monkeypatch, sdk_fakes,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    first_session = _initialized_session()
    second_session = _initialized_session()
    assert first_session != second_session

    first = _payload(_call("session-independent", _dialog(), first_session))
    second = _payload(_call("session-independent", _dialog(), second_session))
    assert first["operation_id"] != second["operation_id"]
    assert first["doc_id"] != second["doc_id"]

