"""Safe operational diagnostics for the public Telegram webhook."""

from __future__ import annotations

import asyncio
import os
import socket

import httpx
import pytest

from services.connectors.telegram.app.main import create_app


pytestmark = pytest.mark.offline

SECRET = "webhook-secret-marker"
BODY_MARKER = "private-message-marker"
EXCEPTION_MARKER = "provider-url-marker https://api.example.test/bot-token?secret=value"


def _selector_loop():
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    original_socketpair = socket.socketpair

    def loopback_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reader.bind(("127.0.0.1", 0))
        writer.bind(("127.0.0.1", 0))
        writer.connect(reader.getsockname())
        reader.connect(writer.getsockname())
        return reader, writer

    socket.socketpair = loopback_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = original_socketpair


def _post(app, body, headers=None):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook", content=body,
                headers=headers or {"X-Telegram-Bot-Api-Secret-Token": SECRET},
            )

    return asyncio.run(send(), loop_factory=_selector_loop)


class _FailingIntake:
    def handle_update(self, _update):
        raise RuntimeError(EXCEPTION_MARKER)


@pytest.mark.parametrize("stage", ["secret_load", "intake_init", "handle_update"])
def test_webhook_failures_log_safe_stage_and_exception_type_chain(
    stage, monkeypatch, caplog,
):
    import services.connectors.telegram.app.main as app_module

    update_id = 721
    body = ('{"update_id":721,"message":{"text":"' + BODY_MARKER + '"}}').encode()
    if stage == "secret_load":
        def fail_secret():
            try:
                raise PermissionError(EXCEPTION_MARKER)
            except PermissionError as cause:
                raise ValueError(EXCEPTION_MARKER) from cause

        monkeypatch.setattr(app_module, "_firestore_secret", fail_secret)
        app = create_app(telegram_enabled=True)
        expected = {"detail": "webhook unavailable"}
        headers = {"X-Telegram-Bot-Api-Secret-Token": SECRET}
    elif stage == "intake_init":
        def fail_init():
            try:
                raise PermissionError(EXCEPTION_MARKER)
            except PermissionError as cause:
                raise ValueError(EXCEPTION_MARKER) from cause

        monkeypatch.setattr(app_module, "_default_intake", fail_init)
        app = create_app(webhook_secret=SECRET, telegram_enabled=True)
        expected = {"detail": "webhook unavailable"}
        headers = None
    else:
        app = create_app(
            intake=_FailingIntake(), webhook_secret=SECRET, telegram_enabled=True,
        )
        expected = {"detail": "intake unavailable"}
        headers = None

    caplog.clear()
    response = _post(app, body, headers)

    assert response.status_code == 503
    assert response.json() == expected
    record = next(record for record in caplog.records
                  if getattr(record, "event", None) == "webhook_failure")
    assert record.stage == stage
    assert record.error_type == ("RuntimeError" if stage == "handle_update" else "ValueError")
    assert record.error_chain == (["RuntimeError"] if stage == "handle_update"
                                  else ["ValueError", "PermissionError"])
    if stage != "secret_load":
        assert record.update_id == update_id
    assert BODY_MARKER not in caplog.text
    assert SECRET not in caplog.text
    assert EXCEPTION_MARKER not in caplog.text
    assert BODY_MARKER not in response.text
    assert EXCEPTION_MARKER not in response.text


def test_webhook_success_logs_completion_with_only_valid_update_id(caplog):
    class Intake:
        def handle_update(self, _update):
            return None

    app = create_app(intake=Intake(), webhook_secret=SECRET, telegram_enabled=True)
    caplog.clear()

    response = _post(app, b'{"update_id":812,"message":{"text":"' + BODY_MARKER.encode() + b'"}}')

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    completed = next(record for record in caplog.records
                     if getattr(record, "event", None) == "webhook_completion")
    assert completed.update_id == 812
    assert not hasattr(completed, "chat_id")
    assert not hasattr(completed, "user_id")
    assert BODY_MARKER not in caplog.text
    assert SECRET not in caplog.text


def test_forged_secret_logs_denial_without_receipt_or_body(caplog):
    app = create_app(intake=_FailingIntake(), webhook_secret=SECRET, telegram_enabled=True)
    forged_secret = "forged-secret-marker"
    caplog.clear()

    response = _post(
        app, BODY_MARKER.encode(),
        {"X-Telegram-Bot-Api-Secret-Token": forged_secret},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "forbidden"}
    assert any(getattr(record, "event", None) == "webhook_denied" for record in caplog.records)
    assert not any(getattr(record, "event", None) == "webhook_receipt" for record in caplog.records)
    assert forged_secret not in caplog.text + response.text
    assert SECRET not in caplog.text + response.text
    assert BODY_MARKER not in caplog.text + response.text


def test_malformed_authorized_body_logs_only_fixed_receipt_and_stage(caplog):
    app = create_app(intake=_FailingIntake(), webhook_secret=SECRET, telegram_enabled=True)
    caplog.clear()

    response = _post(app, BODY_MARKER.encode() + b"{", {
        "X-Telegram-Bot-Api-Secret-Token": SECRET,
        "X-Debug-Marker": "header-marker",
    })

    assert response.status_code == 400
    assert response.json() == {"detail": "malformed JSON"}
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "webhook_receipt" in events
    malformed = next(record for record in caplog.records
                     if getattr(record, "event", None) == "webhook_malformed")
    assert malformed.stage == "json"
    assert BODY_MARKER not in caplog.text + response.text
    assert "header-marker" not in caplog.text + response.text
    assert SECRET not in caplog.text + response.text
