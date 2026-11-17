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


def _t011_notification_setup_helper():
    """Load the shared real T011 fixture builder without package assumptions."""
    import importlib.util
    import sys
    from pathlib import Path

    name = "_telegram_test_notifications_helper"
    module = sys.modules.get(name)
    if module is None:
        path = Path(__file__).with_name("test_notifications.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module._publish_t011_clarification


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


def test_superseded_clarification_tasks_acknowledge_without_worker_or_retry(firestore_fake, monkeypatch):
    from core.integrations.firestore.outbox import get_outbox_record
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.repository._common import SESSIONS
    _publish_t011_clarification = _t011_notification_setup_helper()

    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client, repository, accepted, pending = _publish_t011_clarification(firestore_fake)
    now = datetime.now(timezone.utc)
    operation = client.collection(SESSIONS).document(accepted.context_id).collection(
        "operations").document(accepted.operation_id).get().to_dict()
    notification_id = operation["result"]["prompt_notification_id"]
    record = get_outbox_record(client, operation["result"]["prompt_outbox_id"])
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    context = context_ref.get().to_dict()
    context["pending_clarification_id"] = None
    context_ref.set(context)

    operation_record = create_outbox_record(
        client, "tgclarif-operation-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now,
        context_id=accepted.context_id, notification_id=notification_id,
        operation_id="forbidden-queue-operation",
    )
    wrong_context_record = create_outbox_record(
        client, "tgclarif-wrong-context-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now,
        context_id="foreign-context", notification_id=notification_id,
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)

    class Telegram:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, **kwargs):
            self.sent.append((chat_id, text))
            return {"message_id": 9}

        async def edit_message_text(self, *args, **kwargs):
            raise AssertionError("stale clarification must not edit")

    class Worker:
        def __init__(self):
            self.calls = []

        async def handle_task(self, payload):
            self.calls.append(payload)
            return True

    telegram = Telegram()
    worker = Worker()
    publisher = TelegramNotificationPublisher(repository, telegram, clock=lambda: now)
    app = create_app(
        queue_worker=worker, notification_publisher=publisher,
        oidc_verifier=lambda _: type("Identity", (), {"authorized": True})(),
    )
    payload = task_payload(record)
    operation_payload = task_payload(operation_record)
    wrong_context_payload = task_payload(wrong_context_record)

    async def invoke():
        operation_dispatch = await _route(app, "/internal/dispatch")(
            _request("/internal/dispatch", operation_payload, "valid"),
        )
        wrong_context_delivery = await _route(app, "/internal/deliver")(
            _request("/internal/deliver", wrong_context_payload, "valid"),
        )
        dispatch = await _route(app, "/internal/dispatch")(
            _request("/internal/dispatch", payload, "valid"),
        )
        deliver = await _route(app, "/internal/deliver")(
            _request("/internal/deliver", payload, "valid"),
        )
        return operation_dispatch, wrong_context_delivery, dispatch, deliver

    operation_dispatch, wrong_context_delivery, dispatch, deliver = _run(invoke())
    assert operation_dispatch.status_code == 400
    assert wrong_context_delivery.status_code == 400
    assert dispatch == {"status": "superseded"}
    assert deliver == {"status": "superseded"}
    assert telegram.sent == []
    assert worker.calls == []
    assert repository.get_notification(notification_id)["send_attempts_started"] == 0


def test_clarification_tasks_deliver_through_both_endpoints_without_queue_worker(
    firestore_fake, monkeypatch,
):
    from core.integrations.firestore.outbox import get_outbox_record
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository._common import SESSIONS
    _publish_t011_clarification = _t011_notification_setup_helper()

    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client, repository, accepted_dispatch, pending_dispatch = _publish_t011_clarification(
        firestore_fake, update_id=99101, chat_id=921,
    )
    client2, repository2, accepted_deliver, pending_deliver = _publish_t011_clarification(
        firestore_fake, update_id=99102, chat_id=922,
    )
    operation = client.collection(SESSIONS).document(accepted_dispatch.context_id).collection(
        "operations").document(accepted_dispatch.operation_id).get().to_dict()
    operation2 = client2.collection(SESSIONS).document(accepted_deliver.context_id).collection(
        "operations").document(accepted_deliver.operation_id).get().to_dict()
    record = get_outbox_record(client, operation["result"]["prompt_outbox_id"])
    record2 = get_outbox_record(client2, operation2["result"]["prompt_outbox_id"])

    class Telegram:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, **kwargs):
            self.sent.append((chat_id, text, kwargs.get("message_thread_id")))
            return {"message_id": 900 + len(self.sent)}

        async def edit_message_text(self, *args, **kwargs):
            raise AssertionError("new clarification prompt must send")

    telegram = Telegram()
    publisher = TelegramNotificationPublisher(repository, telegram, clock=lambda: datetime.now(timezone.utc))
    publisher2 = TelegramNotificationPublisher(repository2, telegram, clock=lambda: datetime.now(timezone.utc))

    class Worker:
        def __init__(self):
            self.calls = []

        async def handle_task(self, payload):
            self.calls.append(payload)
            return True

    worker = Worker()
    app = create_app(
        queue_worker=worker, notification_publisher=publisher,
        oidc_verifier=lambda _: type("Identity", (), {"authorized": True})(),
    )
    app2 = create_app(
        queue_worker=worker, notification_publisher=publisher2,
        oidc_verifier=lambda _: type("Identity", (), {"authorized": True})(),
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    payload, payload2 = task_payload(record), task_payload(record2)

    async def invoke():
        dispatched = await _route(app, "/internal/dispatch")(
            _request("/internal/dispatch", payload, "valid"),
        )
        delivered = await _route(app2, "/internal/deliver")(
            _request("/internal/deliver", payload2, "valid"),
        )
        return dispatched, delivered

    dispatched, delivered = _run(invoke())
    assert dispatched == {"status": "delivered"}
    assert delivered == {"status": "delivered"}
    assert len(telegram.sent) == 2
    assert worker.calls == []
    assert repository.get_notification(operation["result"]["prompt_notification_id"])["message_id"] == 901
    assert repository2.get_notification(operation2["result"]["prompt_notification_id"])["message_id"] == 902


def test_noncanonical_dispatch_keeps_noop_order_before_publisher_initialization(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    now = datetime.now(timezone.utc)
    record = create_outbox_record(
        client, "noncanonical-notification-task",
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now,
        context_id="ctx-noncanonical", notification_id="notification-noncanonical",
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)

    def unavailable_runtime():
        raise AssertionError("publisher must not initialize for a noncanonical task")

    monkeypatch.setattr("services.connectors.telegram.app.main._notification_runtime", unavailable_runtime)
    app = create_app(oidc_verifier=lambda _: type("Identity", (), {"authorized": True})())
    payload = task_payload(record)
    payload.pop("context_id")

    async def invoke():
        dispatch = await _route(app, "/internal/dispatch")(
            _request("/internal/dispatch", payload, "valid"),
        )
        deliver = await _route(app, "/internal/deliver")(
            _request("/internal/deliver", payload, "valid"),
        )
        missing_notification = dict(payload)
        missing_notification.pop("notification_id")
        deliver_precheck = await _route(app, "/internal/deliver")(
            _request("/internal/deliver", missing_notification, "valid"),
        )
        return dispatch, deliver, deliver_precheck

    dispatch, deliver, deliver_precheck = _run(invoke())
    assert dispatch == {"status": "noop"}
    assert deliver == {"status": "noop"}
    assert deliver_precheck.status_code == 400


def test_legacy_progress_superseded_delivery_keeps_retry_responses(firestore_fake, monkeypatch):
    from services.connectors.telegram.repository import TelegramRepository

    client = firestore_fake.client_class()
    now = datetime.now(timezone.utc)
    context_id, operation_id, notification_id = "ctx-progress", "op-progress", "n-progress"
    client.collection("telegram_sessions").document(context_id).set({"chat_id": 910, "thread_id": None})
    client.collection("notification_outbox").document(notification_id).set({
        "notification_id": notification_id, "kind": "progress", "context_id": context_id,
        "operation_id": operation_id, "body": "Processing", "chat_id": 910,
        "thread_id": None, "state": "pending", "revision": 1,
    })
    record = create_outbox_record(
        client, "progress-task", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=now, now=now, context_id=context_id,
        notification_id=notification_id, operation_id=operation_id,
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)

    class Publisher:
        def __init__(self):
            self.repository = TelegramRepository(client)

        async def publish(self, _notification_id):
            return "superseded"

    app = create_app(
        notification_publisher=Publisher(),
        oidc_verifier=lambda _: type("Identity", (), {"authorized": True})(),
    )
    payload = task_payload(record)

    async def invoke():
        dispatch = await _route(app, "/internal/dispatch")(
            _request("/internal/dispatch", payload, "valid"),
        )
        deliver = await _route(app, "/internal/deliver")(
            _request("/internal/deliver", payload, "valid"),
        )
        return dispatch, deliver

    dispatch, deliver = _run(invoke())
    assert dispatch.status_code == 503
    assert deliver.status_code == 503
