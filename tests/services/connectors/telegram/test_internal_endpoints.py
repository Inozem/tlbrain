"""Internal Telegram task route authentication and payload boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from datetime import datetime, timezone

import pytest
from starlette.requests import Request

from core.integrations.firestore.outbox import OutboxTargetKind, create_outbox_record, task_payload
from services.connectors.telegram.app.main import create_app


pytestmark = pytest.mark.offline


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _selector_loop():
    """Use loopback datagrams for asyncio's Windows self-pipe under the socket guard."""
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
    except BaseException:
        for reader, writer in local_pairs:
            reader.close()
            writer.close()
        raise
    finally:
        socket.socketpair = old_socketpair


def _request(path, payload, token=None):
    headers = [(b"content-type", b"application/json")]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": path,
                    "headers": headers, "query_string": b""}, receive)


def _route(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


async def _direct_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def test_dispatch_requires_oidc_and_exact_server_outbox_payload(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client = firestore_fake.client_class()
    record = create_outbox_record(
        client, "task-a", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=2, schedule_at=datetime.now(timezone.utc), now=datetime.now(timezone.utc),
        context_id="ctx", operation_id="op",
    )

    class Worker:
        async def handle_task(self, payload):
            return True

    app = create_app(queue_worker=Worker(), oidc_verifier=lambda token: type(
        "Identity", (), {"authorized": token == "valid"}
    )() if token in {"valid", "wrong"} else None)
    # Route Firestore through the same synthetic client used to create the record.
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    monkeypatch.setattr("services.connectors.telegram.app.main._first_ack_state", lambda record: "delivered")

    endpoint = _route(app, "/internal/dispatch")

    async def invoke():
        denied = await endpoint(_request("/internal/dispatch", task_payload(record)))
        wrong = await endpoint(_request("/internal/dispatch", task_payload(record), "wrong"))
        payload = task_payload(record)
        payload["text"] = "must not be accepted"
        extra = await endpoint(_request("/internal/dispatch", payload, "valid"))
        accepted = await endpoint(_request("/internal/dispatch", task_payload(record), "valid"))
        return denied, wrong, extra, accepted

    denied, wrong, extra, accepted = _run(invoke())
    assert denied.status_code == 401
    assert wrong.status_code == 403
    assert extra.status_code == 400
    assert accepted == {"status": "processed"}


def test_delivery_forwards_only_server_notification_id(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client = firestore_fake.client_class()
    now = datetime.now(timezone.utc)
    record = create_outbox_record(
        client, "notify-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now, context_id="ctx", notification_id="notify-1",
    )
    calls = []

    class Publisher:
        async def publish(self, notification_id):
            calls.append(notification_id)
            return "delivered"

    app = create_app(notification_publisher=Publisher(), oidc_verifier=lambda _: type(
        "Identity", (), {"authorized": True}
    )())
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)

    endpoint = _route(app, "/internal/deliver")
    payload = task_payload(record)
    forbidden = dict(payload, body="caller content")

    async def invoke():
        rejected = await endpoint(_request("/internal/deliver", forbidden, "valid"))
        delivered = await endpoint(_request("/internal/deliver", payload, "valid"))
        return rejected, delivered

    rejected, delivered = _run(invoke())
    assert rejected.status_code == 400
    assert delivered == {"status": "delivered"}
    assert calls == ["notify-1"]


def test_delivery_retries_nonterminal_publisher_results(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client = firestore_fake.client_class()
    now = datetime.now(timezone.utc)
    record = create_outbox_record(
        client, "notify-retry-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now, context_id="ctx", notification_id="notify-retry",
    )
    calls = []
    result = {"value": "retry"}

    class Publisher:
        async def publish(self, notification_id):
            calls.append(notification_id)
            return result["value"]

    app = create_app(notification_publisher=Publisher(), oidc_verifier=lambda _: type(
        "Identity", (), {"authorized": True}
    )())
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    endpoint = _route(app, "/internal/deliver")
    payload = task_payload(record)

    async def invoke():
        responses = []
        for state in ("retry", "busy", "stale", "delivered", "failed", "delivery_unknown", "not_found"):
            result["value"] = state
            responses.append((state, await endpoint(_request("/internal/deliver", payload, "valid"))))
        return responses

    responses = _run(invoke())
    for state, response in responses:
        status_code = response.status_code if hasattr(response, "status_code") else 200
        assert status_code == (503 if state in {"retry", "busy", "stale"} else 200)
    assert calls == ["notify-retry"] * 7


def test_save_processor_waits_for_confirmed_first_ack(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client = firestore_fake.client_class()
    now = datetime.now(timezone.utc)
    record = create_outbox_record(
        client, "save-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now, context_id="ctx", operation_id="save-op",
    )
    ack_state = {"value": "pending"}
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    monkeypatch.setattr("services.connectors.telegram.app.main._first_ack_state",
                        lambda _: ack_state["value"])
    processed = []
    events = []

    class Worker:
        async def handle_task(self, payload):
            processed.append(payload["operation_id"])
            events.append("process")
            return True

    class Publisher:
        async def publish(self, notification_id):
            events.append(("publish", notification_id))
            return "delivered"

    app = create_app(queue_worker=Worker(), notification_publisher=Publisher(), oidc_verifier=lambda _: type(
        "Identity", (), {"authorized": True}
    )())
    def lifecycle(_):
        events.append("lifecycle")
        return "tgfinal-save-op"

    monkeypatch.setattr("services.connectors.telegram.app.main._update_terminal_notification", lifecycle)
    endpoint = _route(app, "/internal/dispatch")

    async def invoke():
        before = await endpoint(_request("/internal/dispatch", task_payload(record), "valid"))
        ack_state["value"] = "failed"
        terminal = await endpoint(_request("/internal/dispatch", task_payload(record), "valid"))
        ack_state["value"] = "delivered"
        after = await endpoint(_request("/internal/dispatch", task_payload(record), "valid"))
        return before, terminal, after

    before, terminal, after = _run(invoke())
    assert before.status_code == 503
    assert terminal == {"status": "failed"}
    assert processed == ["save-op"]
    assert after == {"status": "processed"}
    assert events == ["process", "lifecycle", ("publish", "tgfinal-save-op")]


def test_completed_operation_without_verified_proof_maps_to_terminal_verification_error():
    from services.connectors.telegram.app.main import _terminal_lifecycle_status

    assert _terminal_lifecycle_status("completed", False) == (
        "save", "verification", "completion_unverified", "error",
    )
    assert _terminal_lifecycle_status("completed", True) == (
        "save", "complete", "completed", "final",
    )
