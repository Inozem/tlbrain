"""Offline tests for Telegram webhook bootstrap and disable lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.requests import Request


pytestmark = pytest.mark.offline


def _route(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def _request(path, token="valid"):
    headers = [(b"content-type", b"application/json")]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    return Request(
        {"type": "http", "method": "POST", "path": path,
         "headers": headers, "query_string": b""},
        receive,
    )


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _selector_loop():
    """Use loopback datagrams for asyncio's Windows self-pipe under socket guard."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    old_socketpair = socket.socketpair
    local_pairs = []

    def local_datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            local_pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_datagram_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = old_socketpair


def _response_text(response):
    body = getattr(response, "body", None)
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(response)


def test_bootstrap_secret_concurrent_first_use_converges_and_later_reuses(firestore_fake):
    from core.integrations.firestore.telegram_secret import (
        get_or_create_telegram_webhook_secret,
    )

    client = firestore_fake.client_class()
    sequence = iter(f"synthetic-secret-{number}" for number in range(32))

    def secret_factory():
        return next(sequence)

    with ThreadPoolExecutor(max_workers=12) as pool:
        secrets = list(pool.map(
            lambda _: get_or_create_telegram_webhook_secret(
                client, secret_factory=secret_factory,
            ),
            range(24),
        ))

    assert len(set(secrets)) == 1
    stored_secret = secrets[0]
    assert "tokens/telegram_webhook" in firestore_fake.documents
    assert stored_secret in repr(firestore_fake.documents["tokens/telegram_webhook"])
    assert get_or_create_telegram_webhook_secret(
        client,
        secret_factory=lambda: pytest.fail("existing secret must be reused"),
    ) == stored_secret


def test_webhook_registration_subscribes_to_messages_without_dropping_pending(
    firestore_fake, caplog,
):
    from infra.deploy.connectors.telegram.webhook_admin import register_telegram_webhook

    bot_token = "synthetic-bot-token-do-not-log"
    webhook_url = "https://telegram.synthetic.invalid/hook-private"
    client = firestore_fake.client_class()

    class TelegramClient:
        def __init__(self):
            self.calls = []

        def set_webhook(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True, "description": "webhook registered"}

    telegram_client = TelegramClient()
    with caplog.at_level(logging.DEBUG):
        result = register_telegram_webhook(
            client=client,
            bot_token=bot_token,
            webhook_url=webhook_url,
            telegram_client=telegram_client,
        )

    assert len(telegram_client.calls) == 1
    call = telegram_client.calls[0]
    stored_secret_values = {
        value
        for value in firestore_fake.documents["tokens/telegram_webhook"].values()
        if isinstance(value, str)
    }
    assert call["url"] == webhook_url
    assert call["allowed_updates"] == ["message"]
    assert call.get("drop_pending_updates") is not True
    assert call["secret_token"] in stored_secret_values
    diagnostics = f"{result!r}\n{caplog.text}"
    assert bot_token not in diagnostics
    assert webhook_url not in diagnostics


def test_webhook_registration_default_path_posts_safely_with_bot_token(
    firestore_fake, monkeypatch,
):
    from infra.deploy.connectors.telegram.webhook_admin import register_telegram_webhook

    bot_token = "synthetic-default-path-token"
    webhook_url = "https://telegram.synthetic.invalid/default-path"
    client = firestore_fake.client_class()
    observed = {}

    class Response:
        status = 200

        def read(self, limit):
            observed["read_limit"] = limit
            return b'{"ok": true, "result": true}'

    class Connection:
        def __init__(self, host, *, timeout):
            observed["host"] = host
            observed["timeout"] = timeout

        def request(self, method, path, *, body, headers):
            observed.update(method=method, path=path, body=body, headers=headers)

        def getresponse(self):
            return Response()

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(
        "infra.deploy.connectors.telegram.webhook_admin.http.client.HTTPSConnection",
        Connection,
    )
    result = register_telegram_webhook(
        client=client,
        bot_token=bot_token,
        webhook_url=webhook_url,
    )

    assert result is True
    assert observed["host"] == "api.telegram.org"
    assert observed["timeout"] == 10
    assert observed["method"] == "POST"
    assert observed["path"] == f"/bot{bot_token}/setWebhook"
    payload = json.loads(observed["body"])
    stored_secret_values = {
        value
        for value in firestore_fake.documents["tokens/telegram_webhook"].values()
        if isinstance(value, str)
    }
    assert payload["url"] == webhook_url
    assert payload["allowed_updates"] == ["message"]
    assert payload["secret_token"] in stored_secret_values
    assert "drop_pending_updates" not in payload
    assert type(result) is bool
    assert observed["closed"] is True


def test_webhook_registration_default_path_does_not_claim_provider_failure(
    firestore_fake, monkeypatch,
):
    from infra.deploy.connectors.telegram.webhook_admin import register_telegram_webhook

    client = firestore_fake.client_class()

    class Response:
        status = 200

        def read(self, _limit):
            return b'{"ok": false, "description": "synthetic failure"}'

    class Connection:
        def __init__(self, _host, *, timeout):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(
        "infra.deploy.connectors.telegram.webhook_admin.http.client.HTTPSConnection",
        Connection,
    )

    result = register_telegram_webhook(
        client=client,
        bot_token="synthetic-failure-token",
        webhook_url="https://telegram.synthetic.invalid/failure",
    )

    assert result is False


def test_disable_uses_runtime_telegram_identity_without_deploy_token_and_preserves_records(
    firestore_fake, monkeypatch,
):
    from services.connectors.telegram.app.main import create_app

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL", "telegram-deployer@example.test")
    monkeypatch.setattr(asyncio, "to_thread", lambda func, /, *args, **kwargs: func(*args, **kwargs))
    client = firestore_fake.client_class()
    client.collection("documents").document("saved-doc").set(
        {"document_id": "doc-7", "content": "synthetic saved document", "state": "saved"}
    )
    client.collection("telegram_sessions").document("ctx-7").set(
        {"context_id": "ctx-7", "draft_id": "draft-7", "state": "pending"}
    )
    client.collection("telegram_sessions").document("ctx-7").collection("drafts").document(
        "draft-7"
    ).set({"text": "synthetic draft", "state": "pending"})
    client.collection("operation_records").document("failed-7").set(
        {"operation_id": "failed-7", "state": "failed", "error": "synthetic failure"}
    )
    before = json.dumps(firestore_fake.documents, sort_keys=True, separators=(",", ":")).encode()

    class RuntimeTelegramIdentity:
        def __init__(self):
            self.delete_calls = 0

        def delete_webhook(self):
            self.delete_calls += 1
            return {"ok": True, "result": True}

    telegram_client = RuntimeTelegramIdentity()
    app = create_app(
        oidc_verifier=lambda token: type("Identity", (), {"email": "telegram-deployer@example.test", "authorized": False})(),
        telegram_client=telegram_client,
    )
    endpoint = _route(app, "/internal/disable-webhook")
    response = _run(endpoint(_request("/internal/disable-webhook")))

    assert getattr(response, "status_code", 200) < 400
    assert telegram_client.delete_calls == 1
    assert response == {"status": "disabled"}
    after = json.dumps(firestore_fake.documents, sort_keys=True, separators=(",", ":")).encode()
    assert after == before


def test_disable_reports_telegram_api_failure_without_claiming_success(
    firestore_fake, monkeypatch,
):
    from services.connectors.telegram.app.main import create_app

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL", "telegram-deployer@example.test")
    monkeypatch.setattr(asyncio, "to_thread", lambda func, /, *args, **kwargs: func(*args, **kwargs))

    class UnavailableTelegramClient:
        def delete_webhook(self):
            raise RuntimeError("synthetic-bot-token-do-not-log synthetic Telegram API outage")

    app = create_app(
        oidc_verifier=lambda token: type("Identity", (), {"email": "telegram-deployer@example.test", "authorized": False})(),
        telegram_client=UnavailableTelegramClient(),
    )
    response = _run(_route(app, "/internal/disable-webhook")(
        _request("/internal/disable-webhook")
    ))

    assert getattr(response, "status_code", 200) >= 400
    assert "success" not in _response_text(response).lower()
    assert "synthetic-bot-token-do-not-log" not in _response_text(response)


def test_disable_rejects_unauthorized_oidc_before_calling_telegram(monkeypatch):
    from services.connectors.telegram.app.main import create_app

    monkeypatch.setattr(asyncio, "to_thread", lambda func, /, *args, **kwargs: func(*args, **kwargs))
    monkeypatch.setenv("TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL", "telegram-deployer@example.test")

    class TelegramClient:
        def __init__(self):
            self.delete_calls = 0

        def delete_webhook(self, **kwargs):
            self.delete_calls += 1
            return {"ok": True, "result": True}

    telegram_client = TelegramClient()
    app = create_app(
        oidc_verifier=lambda token: type("Identity", (), {"email": "other@example.test", "authorized": True})(),
        telegram_client=telegram_client,
    )
    response = _run(_route(app, "/internal/disable-webhook")(
        _request("/internal/disable-webhook")
    ))

    assert getattr(response, "status_code", 200) == 403
    assert telegram_client.delete_calls == 0


def test_disable_rejects_task_invoker_even_when_task_authorized(monkeypatch):
    from services.connectors.telegram.app.main import create_app

    monkeypatch.setenv("TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL", "telegram-deployer@example.test")
    telegram_client = type("TelegramClient", (), {"delete_webhook": lambda self: pytest.fail("called")})()
    app = create_app(
        oidc_verifier=lambda _token: type("Identity", (), {
            "email": "telegram-task@example.test", "authorized": True,
        })(),
        telegram_client=telegram_client,
    )
    response = _run(_route(app, "/internal/disable-webhook")(
        _request("/internal/disable-webhook")
    ))

    assert getattr(response, "status_code", 200) == 403


def test_disabled_webhook_does_not_read_or_store_body(monkeypatch):
    from services.connectors.telegram.app.main import create_app

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    app = create_app(intake=type("Intake", (), {"handle_update": lambda *_: pytest.fail("intake called")})(),
                     webhook_secret="synthetic-secret")
    body_reads = []

    async def receive():
        body_reads.append(True)
        return {"type": "http.request", "body": b'{"message":"private"}', "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/telegram/webhook",
                       "headers": [(b"X-Telegram-Bot-Api-Secret-Token", b"synthetic-secret")],
                       "query_string": b""}, receive)
    response = _run(_route(app, "/telegram/webhook")(request))

    assert getattr(response, "status_code", 200) == 503
    assert body_reads == []
