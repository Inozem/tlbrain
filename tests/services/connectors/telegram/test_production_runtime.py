"""Offline composition checks for Telegram's real default runtime."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from datetime import datetime, timezone

import httpx
import pytest


pytestmark = pytest.mark.offline


def test_create_app_selects_lazy_managed_default_without_provider_construction():
    from services.connectors.telegram.app.main import create_app

    app = create_app()

    assert app.state.telegram_managed_runtime is True
    assert app.state.telegram_queue_worker is None


def test_explicit_none_queue_worker_keeps_injected_caller_contract():
    from services.connectors.telegram.app.main import create_app

    app = create_app(queue_worker=None)

    assert app.state.telegram_managed_runtime is False
    assert app.state.telegram_queue_worker is None


def test_default_factory_builds_real_worker_command_runtime_and_save_workflow(firestore_fake):
    from services.connectors.telegram.queue import TelegramQueueWorker
    from services.connectors.telegram.runtime.factory import build_default_queue_worker
    from services.connectors.telegram.runtime.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    client = firestore_fake.client_class()
    worker = build_default_queue_worker(client)

    assert isinstance(worker, TelegramQueueWorker)
    assert isinstance(worker.processor, TelegramCommandRuntime)
    assert isinstance(worker.processor.save_handler, TelegramSaveRuntime)
    assert worker.client is client
    # Factory construction remains lazy; no Gemini/Telegram client is created until work runs.
    assert worker.processor.model is None
    assert worker.processor.save_handler.model is None
    assert worker.processor.save_handler.telegram_client is None


def _selector_loop():
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    original_socketpair = socket.socketpair

    def local_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reader.bind(("127.0.0.1", 0))
        writer.bind(("127.0.0.1", 0))
        writer.connect(reader.getsockname())
        reader.connect(writer.getsockname())
        return reader, writer

    socket.socketpair = local_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = original_socketpair


def test_default_app_webhook_dispatch_routes_real_greeting_and_delivers_durable_reply(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Exercise default intake, real queue processor/router/agent and Telegram transport."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    # The shared index publisher binds this SDK seam at module import time.
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        transport = httpx.MockTransport(bot_response)
        return TelegramClient.__original_init__(
            self, token, http_client=httpx.AsyncClient(transport=transport), timeout=timeout,
        )

    original_init = TelegramClient.__init__
    monkeypatch.setattr(TelegramClient, "__original_init__", original_init, raising=False)
    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    route = {
        "kind": "substantive", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "Hello!",
        "tool_choice": None, "certainty": "certain",
    }
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="route_command", args=route),
        )]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="Hello there!")]))],
    ))
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    assert app.state.telegram_queue_worker is None
    update = {"update_id": 14001, "message": {
        "message_id": 61, "date": 1793320000,
        "chat": {"id": 77, "type": "private"},
        "from": {"id": 7, "first_name": "Tester"},
        "text": "Hello",
    }}

    async def request_webhook():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=update,
            )

    response = asyncio.run(request_webhook(), loop_factory=_selector_loop)
    assert response.status_code == 200
    assert app.state.telegram_queue_worker is None
    dispatch = next(item for item in task_requests
                    if item["body"].get("target_kind") == "telegram_dispatch")
    body = dispatch["body"]
    persisted_record = get_outbox_record(db, body["outbox_id"])
    assert persisted_record is not None, (body, task_requests)
    assert task_payload(persisted_record) == body, (task_payload(persisted_record), body)

    async def dispatch_task():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/internal/dispatch", headers={"Authorization": "Bearer fake-task"}, json=body,
            )

    result = asyncio.run(dispatch_task(), loop_factory=_selector_loop)
    assert result.status_code == 200, result.text
    assert result.json() == {"status": "processed"}
    assert app.state.telegram_queue_worker is not None
    assert len([row for row in sdk_fakes.trace
                if row["boundary"] == "gemini" and row["operation"] == "generate_content"]) == 2
    assert [payload["text"] for method, payload in sent if method == "sendMessage"] == ["Hello there!"]
    context_id, operation_id = body["context_id"], body["operation_id"]
    operation = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(operation_id).get().to_dict())
    assert operation["state"] == "completed"
    assert operation["notification_id"]
    notification = db.collection("notification_outbox").document(
        operation["notification_id"]
    ).get().to_dict()
    assert notification["state"] == "delivered"


def test_default_app_forward_question_followup_use_current_draft_full_history(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Exercise default intake, route/model SDK calls, session persistence and replay."""
    from google.genai import types
    from services.connectors.telegram.app.main import create_app

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")

    tasks, sent = [], []
    enqueue_fake = lambda **kwargs: tasks.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 700 + len(sent)}})

    from services.connectors.telegram.client import TelegramClient
    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response)), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    def queue_route(text):
        route = {
            "kind": "substantive", "target_ref": None, "target_operation_id": None,
            "dialog_number": None, "result_refs": [], "substantive_text": text,
            "tool_choice": None, "certainty": "certain",
        }
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                function_call=types.FunctionCall(name="route_command", args=route),
            )]))],
        ))

    queue_route("When is the kickoff?")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text="The kickoff is April 17.",
        )]))],
    ))
    queue_route("And what day?")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text="It is April 17.",
        )]))],
    ))
    long_question = "Which detail should I keep? " + ("full-context tail " * 89) + "full-context tail"
    queue_route(long_question)
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text="Keep the April 17 kickoff date.",
        )]))],
    ))
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    async def submit(update):
        start = len(tasks)
        response = await post("/telegram/webhook", update, secret=True)
        assert response.status_code == 200, response.text
        fresh = [item["body"] for item in tasks[start:]
                 if item["body"].get("target_kind") == "telegram_dispatch"]
        for body in fresh:
            if body.get("notification_id"):
                if body.get("outbox_id", "").startswith("tgcard-close-"):
                    continue  # The controlled early-close integration is tested separately.
                delivered = await post("/internal/deliver", body)
                assert delivered.status_code == 200, delivered.text
        operation = next((body for body in fresh if not body.get("notification_id")), None)
        if operation is None:
            return fresh[0]
        result = await post("/internal/dispatch", operation)
        assert result.status_code == 200, result.text
        return operation

    source_text = "Kickoff is April 17."
    forward = asyncio.run(submit({"update_id": 14101, "message": {
        "message_id": 81, "date": 1793320000,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Nora"},
                           "date": 1793310000},
        "text": source_text,
    }}), loop_factory=_selector_loop)
    first = asyncio.run(submit({"update_id": 14102, "message": {
        "message_id": 82, "date": 1793320010,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "text": "When is the kickoff?",
    }}), loop_factory=_selector_loop)
    second = asyncio.run(submit({"update_id": 14103, "message": {
        "message_id": 83, "date": 1793320020,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "text": "And what day?",
    }}), loop_factory=_selector_loop)
    third = asyncio.run(submit({"update_id": 14104, "message": {
        "message_id": 84, "date": 1793320030,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "text": long_question,
    }}), loop_factory=_selector_loop)

    assert [payload["text"] for method, payload in sent if method == "sendMessage"][-3:] == [
        "The kickoff is April 17.", "It is April 17.", "Keep the April 17 kickoff date.",
    ]
    gemini_calls = [event["arguments"] for event in sdk_fakes.trace
                    if event["boundary"] == "gemini" and event["operation"] == "generate_content"]
    assert len(gemini_calls) == 6
    is_routing = lambda call: getattr(call["config"], "tool_config", None) is not None
    route_inputs = [json.loads(call["contents"][0]["parts"][0]["text"])
                    for call in gemini_calls if is_routing(call)]
    assert [row["command"] for row in route_inputs] == [
        "When is the kickoff?", "And what day?", long_question,
    ]
    assert len(route_inputs[2]["command"]) == len(long_question)
    assert [turn["parts"][0]["text"] for turn in route_inputs[1]["conversation_history"]] == [
        "When is the kickoff?", "The kickoff is April 17.",
    ]
    assert [turn["parts"][0]["text"] for turn in route_inputs[2]["conversation_history"]] == [
        "When is the kickoff?", "The kickoff is April 17.",
        "And what day?", "It is April 17.",
    ]
    main_inputs = [json.dumps(call["contents"], ensure_ascii=False, default=str)
                   for call in gemini_calls if not is_routing(call)]
    assert all(source_text in request for request in main_inputs)
    assert all("untrusted_forwarded_source" in request for request in main_inputs)
    assert any(str(call["config"].tool_config.function_calling_config.mode).endswith("ANY")
               for call in gemini_calls if is_routing(call))

    context_id = first["context_id"]
    context = db.collection("telegram_sessions").document(context_id).get().to_dict()
    conversation_id = context["active_conversation_id"]
    turns = (db.collection("telegram_sessions").document(context_id)
             .collection("conversations").document(conversation_id).collection("turns").stream())
    rows = sorted((row.to_dict() for row in turns), key=lambda row: row["turn_sequence"])
    assert [row["role"] for row in rows] == [
        "user", "model", "user", "model", "user", "model",
    ]
    assert rows[4]["parts"][0]["text"] == long_question
    assert source_text not in repr(rows)
    draft_id = context["active_draft_id"]
    assert db.collection("telegram_sessions").document(context_id).collection(
        "drafts").document(draft_id).collection("messages").document("81").get().exists
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]
    assert forward["context_id"] == first["context_id"] == second["context_id"] == third["context_id"]


def test_default_app_malformed_accepted_draft_cutoff_settles_safely_without_main_model(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A broken accepted source manifest settles once without invoking the main model."""
    from google.genai import types
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")

    tasks, sent = [], []
    enqueue_fake = lambda **kwargs: tasks.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 800 + len(sent)}})

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response)), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    route = {
        "kind": "substantive", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "What was forwarded?",
        "tool_choice": None, "certainty": "certain",
    }
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="route_command", args=route),
        )]))],
    ))
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    async def accept(update):
        start = len(tasks)
        response = await post("/telegram/webhook", update, secret=True)
        assert response.status_code == 200, response.text
        fresh = [item["body"] for item in tasks[start:]
                 if item["body"].get("target_kind") == "telegram_dispatch"]
        for body in fresh:
            if body.get("notification_id"):
                if body.get("outbox_id", "").startswith("tgcard-close-"):
                    continue  # The controlled early-close integration is tested separately.
                delivered = await post("/internal/deliver", body)
                assert delivered.status_code == 200, delivered.text
        return fresh

    source_text = "Accepted source must remain available after cutoff validation fails."
    forward = asyncio.run(accept({"update_id": 14201, "message": {
        "message_id": 91, "date": 1793321000,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Nora"},
                           "date": 1793320000},
        "text": source_text,
    }}), loop_factory=_selector_loop)
    question_tasks = asyncio.run(accept({"update_id": 14202, "message": {
        "message_id": 92, "date": 1793321010,
        "chat": {"id": 79, "type": "private"}, "from": {"id": 7},
        "text": "What was forwarded?",
    }}), loop_factory=_selector_loop)
    operation_dispatch = next(body for body in question_tasks if not body.get("notification_id"))
    operation_ref = (db.collection("telegram_sessions").document(operation_dispatch["context_id"])
                     .collection("operations").document(operation_dispatch["operation_id"]))
    operation = operation_ref.get().to_dict()
    command_path = operation["input_ref"].split("/")
    command_ref = (db.collection(command_path[0]).document(command_path[1])
                   .collection(command_path[2]).document(command_path[3]))
    command_ref.update({"cutoff_message_ids": [99999]})

    result = asyncio.run(post("/internal/dispatch", operation_dispatch),
                         loop_factory=_selector_loop)
    assert result.status_code == 200, result.text
    assert result.json() == {"status": "processed"}
    settled = operation_ref.get().to_dict()
    assert settled["state"] == "failed"
    assert settled.get("notification_id")
    terminal = db.collection("notification_outbox").document(settled["notification_id"]).get().to_dict()
    assert terminal["state"] == "delivered"
    assert "dialog #1" in terminal["body"]
    assert operation_dispatch["operation_id"] in terminal["body"]
    assert source_text not in terminal["body"]
    draft_id = db.collection("telegram_sessions").document(
        operation_dispatch["context_id"]).get().to_dict()["active_draft_id"]
    source_ref = (db.collection("telegram_sessions").document(operation_dispatch["context_id"])
                  .collection("drafts").document(draft_id).collection("messages").document("91"))
    assert source_ref.get().exists

    sdk_count = len([event for event in sdk_fakes.trace
                     if event["boundary"] == "gemini" and event["operation"] == "generate_content"])
    assert sdk_count == 1
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]
    status_dispatch = next(body for item in tasks for body in [item["body"]]
                           if body.get("target_kind") == "telegram_status_check"
                           and body.get("operation_id") == operation_dispatch["operation_id"])
    status_result = asyncio.run(post("/internal/status-check", status_dispatch),
                                loop_factory=_selector_loop)
    assert status_result.status_code == 200, status_result.text
    duplicate = asyncio.run(post("/internal/dispatch", operation_dispatch),
                             loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == sdk_count
    assert len([payload for method, payload in sent if method == "sendMessage"]) == 2


def test_default_app_gemini_402_fails_save_once_and_preserves_numbered_draft(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """Exercise default intake, routing, terminal settlement, replay, and safe delivery."""
    from google.genai.errors import ClientError
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("TELEGRAM_DELIVERY_URL", "https://telegram.pytest.invalid/internal/deliver")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")

    task_requests, sent = [], []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response)), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    private_marker = "synthetic-provider-private-marker"
    sdk_fakes.gemini.queue("generate_content", error=ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": private_marker,
    }}))
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = {"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"} if secret else {
                "Authorization": "Bearer fake-task",
            }
            return await client.post(path, headers=headers, json=payload)

    source_text = "Synthetic forwarded source that must remain in its draft."
    forward = {"update_id": 14011, "message": {
        "message_id": 71, "date": 1793320000,
        "chat": {"id": 78, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000},
        "text": source_text,
    }}
    accepted = asyncio.run(post("/telegram/webhook", forward, secret=True), loop_factory=_selector_loop)
    assert accepted.status_code == 200, accepted.text
    forward_dispatch = next(item["body"] for item in task_requests
                            if item["body"].get("target_kind") == "telegram_dispatch")
    assert get_outbox_record(db, forward_dispatch["outbox_id"]) is not None
    delivered_ack = asyncio.run(post("/internal/deliver", forward_dispatch), loop_factory=_selector_loop)
    assert delivered_ack.status_code == 200, delivered_ack.text

    command = {"update_id": 14012, "message": {
        "message_id": 72, "date": 1793320010,
        "chat": {"id": 78, "type": "private"}, "from": {"id": 7}, "text": "save this dialog",
    }}
    command_result = asyncio.run(post("/telegram/webhook", command, secret=True), loop_factory=_selector_loop)
    assert command_result.status_code == 200, command_result.text
    command_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    operation_dispatch = next(body for body in reversed(command_tasks) if not body.get("notification_id"))
    from core.integrations.firestore.operation_queue import get_operation
    queued_operation = get_operation(
        db.collection("telegram_sessions").document(operation_dispatch["context_id"]),
        operation_dispatch["operation_id"],
    )
    assert queued_operation is not None
    assert not hasattr(queued_operation, "typed_action")
    assert not hasattr(queued_operation, "receipt_card_notification_id")
    for body in reversed(command_tasks):
        if body.get("notification_id"):
            if body.get("outbox_id", "").startswith("tgcard-close-"):
                continue
            ack = asyncio.run(post("/internal/deliver", body), loop_factory=_selector_loop)
            assert ack.status_code == 200, ack.text
            break

    result = asyncio.run(post("/internal/dispatch", operation_dispatch),
                         loop_factory=_selector_loop)
    assert result.status_code == 200, result.text
    assert result.json() == {"status": "processed"}
    status_dispatch = next(item["body"] for item in task_requests
                           if item["body"].get("target_kind") == "telegram_status_check"
                           and item["body"].get("operation_id") == operation_dispatch["operation_id"])
    status_result = asyncio.run(post("/internal/status-check", status_dispatch),
                                loop_factory=_selector_loop)
    assert status_result.status_code == 200, status_result.text
    duplicate = asyncio.run(post("/internal/dispatch", operation_dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == 1
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]

    operation_ref = (db.collection("telegram_sessions").document(operation_dispatch["context_id"])
                     .collection("operations").document(operation_dispatch["operation_id"]))
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "failed"
    context = db.collection("telegram_sessions").document(operation_dispatch["context_id"]).get().to_dict()
    draft_id = (operation.get("target_draft_id") or operation.get("draft_id")
                or context.get("active_draft_id"))
    assert draft_id
    context_ref = db.collection("telegram_sessions").document(operation_dispatch["context_id"])
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft = draft_ref.get().to_dict()
    assert draft["dialog_number"] > 0
    assert draft_ref.collection("messages").document("71").get().to_dict()["text"] == source_text
    stage_rows = [row for path, row in firestore_fake.documents.items()
                  if path.startswith(
                      f"telegram_sessions/{operation_dispatch['context_id']}/operations/"
                      f"{operation_dispatch['operation_id']}/stages/"
                  )]
    assert len(stage_rows) == 1
    stage = stage_rows[0]
    assert stage["failures"] == 1
    assert stage["last_error"]["category"] == "billing"
    assert stage["last_error"]["service"] == "gemini"
    assert stage["last_error"]["retryable"] is False
    assert private_marker not in repr(stage)

    error_messages = [payload["text"] for method, payload in sent if method == "sendMessage"]
    terminal = next(text for text in error_messages if "Prepay credits" in text)
    assert f"Dialog #{draft['dialog_number']}" in terminal
    assert "Error in gemini/routing" in terminal
    assert "category=billing" in terminal
    assert operation_dispatch["operation_id"] in terminal
    assert "failures=1" in terminal
    assert "https://aistudio.google.com/billing" in terminal
    assert private_marker not in repr(error_messages)
    assert "GEMINI_API_KEY" not in terminal
    assert source_text not in terminal
    assert private_marker not in caplog.text
    assert source_text not in caplog.text
    stored_errors = [row for path, row in firestore_fake.documents.items()
                     if path.startswith("notification_outbox/")
                     and row.get("operation_id") == operation_dispatch["operation_id"]]
    assert private_marker not in repr(stored_errors)
    assert source_text not in repr(stored_errors)


def test_default_save_repository_roundtrips_large_stage_payload_as_bounded_parts(firestore_fake):
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.queue import claim_head
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    forward = repository.accept_forward(5101, chat_id=77, thread_id=None,
        message={"message_id": 41, "text": "source", "transcription_state": "not_required"},
        now=datetime.now(timezone.utc))
    accepted = repository.accept_command(5102, chat_id=77, thread_id=None, user_id=7,
        command=parse_command("save"), now=datetime.now(timezone.utc))
    context_ref = client.collection("telegram_sessions").document(forward.context_id)
    lease = claim_head(client, context_ref, owner="large-stage-worker", now=datetime.now(timezone.utc))
    assert lease is not None and lease.operation_id == accepted.operation_id
    assert repository.confirm_save_command(
        forward.context_id, accepted.operation_id, owner=lease.owner, fence=lease.fence,
        now=datetime.now(timezone.utc),
    ) is True
    ledger = FirestoreAgentLedger(
        client, context_id=forward.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: datetime.now(timezone.utc),
    )
    save_repository = FirestoreTelegramSaveRepository(
        repository, context_id=forward.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: datetime.now(timezone.utc),
        ledger=ledger,
    )
    value = {"utterances": [{"speaker": "Speaker", "text": f"{index}:" + "x" * 2600}
                            for index in range(430)]}
    assert len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 1_000_000

    async def roundtrip():
        ref = await save_repository.store_retry_stage_payload(accepted.operation_id, "analysis", value)
        return ref, await save_repository.load_retry_stage_payload(ref)

    reference, restored = asyncio.run(roundtrip(), loop_factory=_selector_loop)
    assert restored == value
    assert isinstance(reference.get("_input_ref"), str)
    assert reference["_input_ref"].endswith("/input_parts/manifest")
    stored_sizes = [len(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8"))
                    for path, row in firestore_fake.documents.items()
                    if path.startswith(f"telegram_sessions/{forward.context_id}/operations/")]
    assert stored_sizes and max(stored_sizes) < 1_000_000


def test_default_app_forward_then_save_runs_canonical_pipeline_and_waits_for_real_index_receipt(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Run the default app through a real managed save and Vector Sync endpoint."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.folders import upsert_folder
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.repository import TelegramRepository
    from datetime import timedelta

    clock_now = [datetime.now(timezone.utc)]
    real_datetime = datetime

    class ControlledDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = clock_now[0]
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr("services.connectors.telegram.intake.datetime", ControlledDateTime)
    from services.connectors.telegram.app import main as telegram_main
    original_notification_runtime = telegram_main._notification_runtime

    def controlled_notification_runtime():
        publisher = original_notification_runtime()
        publisher.clock = lambda: clock_now[0]
        return publisher

    monkeypatch.setattr(telegram_main, "_notification_runtime", controlled_notification_runtime)

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("TELEGRAM_DELIVERY_URL", "https://telegram.pytest.invalid/internal/deliver")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("DIALOG_OPERATIONS_QUEUE", "pytest-dialog-operations-queue")
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://vector-sync.pytest.invalid")
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.google.com/drive/folders/pytest-synthetic-root")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "pytest-synthetic-refresh")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "pytest-synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "pytest-synthetic-secret")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []
    lost_final_edit = [False]
    def bot_response(request):
        payload = json.loads(request.content)
        method = request.url.path.rsplit("/", 1)[-1]
        sent.append((method, payload))
        if method == "answerCallbackQuery":
            return httpx.Response(200, json={"ok": True, "result": True})
        if (method == "editMessageText"
                and "Saved and indexed." in payload.get("text", "")
                and not lost_final_edit[0]):
            lost_final_edit[0] = True
            raise httpx.ReadTimeout("synthetic lost final edit acknowledgement")
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})
    from services.connectors.telegram.client import TelegramClient
    original_init = TelegramClient.__init__
    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response)), timeout=timeout)
    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    upsert_folder("pytest-synthetic-root", "Root", None, db)
    upsert_folder("folder-client", "Client", "pytest-synthetic-root", db)
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "A complete saved dialog", "folder_path": ["Client"]}),
        )]))],
    ))
    # The summary stage may safely classify a short one-turn dialog as skipped.
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    sdk_fakes.drive.queue("files.create", response={"id": "doc-default-save"})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": "doc-default-save", "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })
    app = create_app(webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
                     oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})())

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = {"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"} if secret else {
                "Authorization": "Bearer fake-task",
            }
            return await client.post(path, headers=headers, json=payload)

    forward = {"update_id": 15001, "message": {
        "message_id": 61, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000},
        "text": "The exact forwarded source line.",
    }}
    forward_response = asyncio.run(post("/telegram/webhook", forward, secret=True),
                                   loop_factory=_selector_loop)
    assert forward_response.status_code == 200, forward_response.text
    forward_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    forward_ack = next(body for body in forward_tasks if body.get("notification_id"))
    ack_done = asyncio.run(post("/internal/deliver", forward_ack),
                           loop_factory=_selector_loop)
    assert ack_done.status_code == 200, ack_done.text
    assert any(method == "sendMessage" for method, _ in sent)  # durable forward acknowledgement
    forward_close = next(body for body in forward_tasks
                         if body.get("notification_id") == forward_ack["notification_id"]
                         and body.get("outbox_id") != forward_ack["outbox_id"])
    early_close = asyncio.run(post("/internal/deliver", forward_close),
                              loop_factory=_selector_loop)
    assert early_close.status_code == 503, early_close.text
    clock_now[0] += timedelta(seconds=1)
    close_done = asyncio.run(post("/internal/deliver", forward_close),
                             loop_factory=_selector_loop)
    assert close_done.status_code == 200, close_done.text
    ready_card = TelegramRepository(db).get_notification(forward_ack["notification_id"])
    callback_data = ready_card["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    callback = {"update_id": 15002, "callback_query": {
        "id": "synthetic-save-click", "from": {"id": 7}, "data": callback_data,
        "message": {"message_id": ready_card["message_id"], "date": 1793320100,
                    "chat": {"id": 77, "type": "private"}},
    }}
    callback_response = asyncio.run(post("/telegram/webhook", callback, secret=True),
                                    loop_factory=_selector_loop)
    assert callback_response.status_code == 200, callback_response.text
    assert any(method == "answerCallbackQuery" for method, _ in sent)
    saving_card = TelegramRepository(db).get_notification(forward_ack["notification_id"])
    assert saving_card["body"] == "Dialog #1: Saving…"
    assert saving_card["reply_markup"] == {"inline_keyboard": []}
    command_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    save_dispatch = next(body for body in reversed(command_tasks)
                         if not body.get("notification_id")
                         and body.get("operation_id"))
    first_slice = asyncio.run(post("/internal/dispatch", save_dispatch),
                              loop_factory=_selector_loop)
    assert first_slice.status_code == 200, first_slice.text
    assert first_slice.json()["status"] in {"noop", "processed"}
    saving_edits = [(method, payload) for method, payload in sent
                    if method == "editMessageText"
                    and payload.get("message_id") == ready_card["message_id"]
                    and "Saving" in payload.get("text", "")]
    assert len(saving_edits) == 1
    assert saving_edits[0][1]["reply_markup"] == {"inline_keyboard": []}
    operation = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                 .collection("operations").document(save_dispatch["operation_id"]).get().to_dict())
    accepted_command = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                        .collection("commands").document("15002").get().to_dict())
    assert operation.get("typed_action") == accepted_command.get("action_metadata")
    assert accepted_command.get("cutoff_message_ids") == [61]
    assert operation["state"] == "running"
    assert operation["route_control"]["kind"] == "save"
    core_operation = db.collection("dialog_operations").document(save_dispatch["operation_id"]).get().to_dict()
    assert core_operation["doc_id"] == "doc-default-save"
    index_task = next(item for item in task_requests
                      if item["url"].startswith("https://vector-sync.pytest.invalid/sync/doc/"))
    assert index_task["url"] == "https://vector-sync.pytest.invalid/sync/doc/doc-default-save"
    index_record = get_outbox_record(db, index_task["body"]["outbox_id"])
    assert index_record is not None and task_payload(index_record) == index_task["body"]

    # Feed the exact published payload to the actual Vector Sync endpoint. Only
    # its Google Docs/Drive boundary is faked; the managed index and receipt path
    # remain real, including Qdrant and canonical completion publication.
    from services.vector_sync.app.main import app as vector_app

    fill_events = [event for event in sdk_fakes.trace
                   if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]
    assert len(fill_events) == 1
    reader_text = fill_events[0]["arguments"]["body"]["requests"][0]["insertText"]["text"]
    def read_saved_document(_doc_id, **_kwargs):
        return reader_text
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: ("folder-client", False, "A complete saved dialog"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", read_saved_document)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc", read_saved_document)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", read_saved_document)

    async def sync_document():
        transport = httpx.ASGITransport(app=vector_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector.test") as client:
            return await client.post(
                "/sync/doc/doc-default-save", json=index_task["body"],
            )
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["status"] == "ok"
    receipt = db.collection("index_completions").document("doc-default-save").collection(
        "generations").document(core_operation["desired_content_hash"]).get().to_dict()
    assert receipt["status"] == "completed"

    # The real queue worker resumes after the verified index receipt exists.
    wake_task_kwargs = next(item for item in task_requests
                            if item["body"].get("target_kind") == "telegram_dispatch"
                            and item["body"].get("operation_id") == save_dispatch["operation_id"]
                            and not item["body"].get("notification_id")
                            and item["body"].get("outbox_id") != save_dispatch["outbox_id"])
    wake = get_outbox_record(db, wake_task_kwargs["body"]["outbox_id"])
    assert wake is not None and wake.operation_id == save_dispatch["operation_id"]
    assert wake.schedule_at > wake.created_at
    wake_task = task_payload(wake)
    scheduled_clock = lambda: wake.schedule_at
    worker = app.state.telegram_queue_worker
    worker.clock = scheduled_clock
    worker.processor.clock = scheduled_clock
    if getattr(worker.processor, "save_handler", None) is not None:
        worker.processor.save_handler.clock = scheduled_clock
    completed = asyncio.run(post("/internal/dispatch", wake_task),
                            loop_factory=_selector_loop)
    assert completed.status_code == 200, completed.text
    queue_operation = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                       .collection("operations").document(save_dispatch["operation_id"]).get().to_dict())
    assert lost_final_edit[0] is True
    assert queue_operation["state"] == "running"
    durable_card = TelegramRepository(db).get_notification(forward_ack["notification_id"])
    assert durable_card["receipt_terminal"] is True
    assert durable_card["state"] in {"pending", "retry", "delivery_unknown"}
    draft_ref = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                 .collection("drafts").document(queue_operation["target_draft_id"]))
    manifest_before_recovery = draft_ref.collection("save_manifests").document(
        save_dispatch["operation_id"],
    ).get().to_dict()
    draft_before_recovery = draft_ref.get().to_dict()
    certificate_before_recovery = draft_before_recovery["completion_certificate"]
    model_work_before_recovery = len([event for event in sdk_fakes.trace
                                      if event["boundary"] == "gemini"
                                      and event["operation"] == "generate_content"])
    assert model_work_before_recovery > 0
    app = create_app(webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
                     oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})())
    retry_wakes = [item["body"] for item in task_requests
                   if item["body"].get("target_kind") == "telegram_dispatch"
                   and item["body"].get("operation_id") == save_dispatch["operation_id"]
                   and not item["body"].get("notification_id")
                   and item["body"].get("outbox_id") != wake_task["outbox_id"]]
    retry_wake = retry_wakes[-1] if retry_wakes else wake_task
    retry_record = get_outbox_record(db, retry_wake["outbox_id"])
    clock_now[0] = max(clock_now[0], retry_record.schedule_at)
    from services.connectors.telegram.app.main import _default_queue_worker
    restarted_worker = _default_queue_worker()
    restarted_clock = lambda: retry_record.schedule_at
    restarted_worker.clock = restarted_clock
    restarted_worker.processor.clock = restarted_clock
    if getattr(restarted_worker.processor, "save_handler", None) is not None:
        restarted_worker.processor.save_handler.clock = restarted_clock
    app.state.telegram_queue_worker = restarted_worker
    replay_after_restart = asyncio.run(post("/internal/dispatch", retry_wake),
                                       loop_factory=_selector_loop)
    assert replay_after_restart.status_code == 200, replay_after_restart.text
    queue_operation = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                       .collection("operations").document(save_dispatch["operation_id"])
                       .get().to_dict())
    shared_after_wake = db.collection("dialog_operations").document(save_dispatch["operation_id"]).get().to_dict()
    draft_after_wake = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                        .collection("drafts").document(queue_operation["target_draft_id"]).get().to_dict())
    assert completed.json()["status"] in {"processed", "noop"}, (
        completed.json(), queue_operation.get("state"), shared_after_wake.get("state"),
        shared_after_wake.get("result"), shared_after_wake.get("stage"),
        draft_after_wake.get("state"), draft_after_wake.get("completion_certificate"),
    )
    assert queue_operation["state"] == "completed"
    assert queue_operation["save_operation_id"] == save_dispatch["operation_id"]
    assert queue_operation["route_result"]["kind"] == "save"
    assert queue_operation.get("doc_id") == "doc-default-save", {
        key: queue_operation.get(key) for key in (
            "state", "save_operation_id", "route_result", "doc_id", "target_doc_id",
            "dialog_number", "target_dialog_number", "conversation_id", "target_conversation_id",
        )
    }
    draft_ref = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                 .collection("drafts").document(queue_operation["target_draft_id"]))
    assert draft_ref.collection("messages").stream() == []
    assert draft_ref.collection("prepared").stream() == []
    manifest = draft_ref.collection("save_manifests").document(
        save_dispatch["operation_id"],
    ).get().to_dict()
    assert manifest["ordered_message_ids"] == [61]
    assert manifest["manifest_digest"] == draft_after_wake["manifest_digest"]
    assert draft_after_wake["completion_certificate"]
    completion_certificate = draft_after_wake["completion_certificate"]
    assert manifest_before_recovery == draft_ref.collection("save_manifests").document(
        save_dispatch["operation_id"],
    ).get().to_dict()
    assert certificate_before_recovery == completion_certificate
    assert model_work_before_recovery == len([event for event in sdk_fakes.trace
                                               if event["boundary"] == "gemini"
                                               and event["operation"] == "generate_content"])
    stages = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
              .collection("operations").document(save_dispatch["operation_id"])
              .collection("stages"))
    for stage_key in ("telegram.preparation", "telegram.analysis", "telegram.classification"):
        stage = stages.document(stage_key).get().to_dict()
        assert stage is not None and stage["state"] == "completed"
        assert stage["result"]["purged"] is True
    request = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
               .collection("operations").document(save_dispatch["operation_id"])
               .collection("agent_requests").document("request").get().to_dict())
    assert request["request"]["purged"] is True
    assert "save this dialog" not in str(request["request"])
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    runtime_store = TelegramRuntimeStore(
        TelegramRepository(db), context_id=save_dispatch["context_id"],
    )
    assert runtime_store.resolve_dialog_number(save_dispatch["context_id"], 1) == "doc-default-save"
    assert runtime_store.get_or_assign_dialog_number(
        save_dispatch["context_id"], "doc-default-save",
    ) == 1
    saved_conversation = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                          .collection("conversations").document(draft_after_wake["conversation_id"])
                          .get().to_dict())
    assert saved_conversation["kind"] == "document"
    assert saved_conversation["doc_id"] == "doc-default-save"
    assert saved_conversation["dialog_number"] == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    save_finals = [payload for method, payload in sent
                   if method == "editMessageText"
                   and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]
    assert len(save_finals) == (2 if lost_final_edit[0] else 1), sent
    assert all(payload["chat_id"] == 77 and payload["message_id"] == ready_card["message_id"]
               for payload in save_finals)
    generation_count = len([event for event in sdk_fakes.trace
                            if event["boundary"] == "gemini"
                            and event["operation"] == "generate_content"])
    final_notices = [row for path, row in firestore_fake.documents.items()
                     if path.startswith("notification_outbox/")
                     and row.get("operation_id") == save_dispatch["operation_id"]
                     and row.get("receipt_terminal") is True]
    assert len(final_notices) == 1, final_notices
    assert final_notices[0]["notification_id"] == forward_ack["notification_id"]

    # Replaying the canonical wake dispatch must reuse the delivered ledger row.
    duplicate = asyncio.run(post("/internal/dispatch", wake_task),
                            loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini"
                and event["operation"] == "generate_content"]) == generation_count
    assert draft_ref.collection("save_manifests").document(
        save_dispatch["operation_id"],
    ).get().to_dict() == manifest
    assert draft_ref.get().to_dict()["completion_certificate"] == completion_certificate
    assert len([payload for method, payload in sent
                if method == "editMessageText"
                and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]) == (
                    2 if lost_final_edit[0] else 1
                )


def test_default_asgi_selected_document_routes_tool_then_answers_over_real_gemini_http(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Exercise saved-document routing, scoped registry read, and durable answer end to end."""
    import httpx
    from google import genai
    from google.genai import types
    from tests.fixtures import sdk_fakes as sdk_fake_module
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.google.com/drive/folders/pytest-synthetic-root")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return TelegramClient.__original_init__(
            self, token, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(bot_response),
            ), timeout=timeout,
        )

    original_telegram_init = TelegramClient.__init__
    monkeypatch.setattr(TelegramClient, "__original_init__", original_telegram_init, raising=False)
    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    repository = TelegramRepository(client=db)
    context_id = "77_default"
    now = datetime.now(timezone.utc)
    db.collection("folders").document("pytest-synthetic-root").set({
        "folder_id": "pytest-synthetic-root", "name": "Root", "parent_id": None,
    })
    db.collection("folders").document("folder-client").set({
        "folder_id": "folder-client", "name": "Client",
        "parent_id": "pytest-synthetic-root",
    })
    db.collection("transcript_index").document("doc-selected").set({
        "source_file": "Selected meeting notes", "parent_id": "folder-client",
        "dialog_date": "2026-10-22", "provider": "telegram", "status": "synced",
    })
    number = repository.bind_document(
        context_id, "doc-selected", "conversation-selected",
        metadata={"title": "Selected meeting notes", "folder_path": []},
        now=now, update_last_doc=False,
    )
    repository.ensure_conversation(
        context_id, "conversation-selected", kind="document", doc_id="doc-selected",
        dialog_number=number, now=now,
    )
    db.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-selected",
        "selected_dialog_number": number,
    })
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-question",
        operation_id="prior-operation", role="user",
        parts=[{"text": "What did the team decide about the schedule?"}], now=now,
    )
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-answer",
        operation_id="prior-operation", role="model",
        parts=[{"text": "They moved the launch to May."}], now=now,
    )

    route = {
        "kind": "substantive", "target_ref": "doc-selected",
        "target_operation_id": None, "dialog_number": number, "result_refs": [],
        "substantive_text": "What is the milestone date in the saved notes?",
        "tool_choice": {
            "name": "list_recent_transcripts", "arguments": {"folder_path": ["Client"]},
        },
        "certainty": "certain",
    }
    wire_requests = []
    application_result = {
        "total": 1, "returned": 1, "offset": 0, "limit": 20,
        "has_more": False, "transcripts": [{
            "doc_id": "doc-selected", "parent_id": "folder-client",
            "title": "Selected meeting notes", "dialog_date": "2026-10-22",
            "status": "synced",
        }],
    }

    def gemini_response(request):
        wire_requests.append(request)
        payload = json.loads(request.content)
        serialized_contents = json.dumps(payload["contents"], ensure_ascii=False)
        if len(wire_requests) == 1:
            assert "What did the team decide about the schedule?" in serialized_contents
            assert "What is the milestone date in the saved notes?" in serialized_contents
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "route_command", "args": route},
                    "thoughtSignature": "c3ludGhldGljLXJvdXRlLXNpZw==",
                }],
            }}]})

        contents = payload["contents"]
        assert "What did the team decide about the schedule?" in serialized_contents
        assert "They moved the launch to May." in serialized_contents
        assert contents[-1]["parts"][0]["text"] == route["substantive_text"]
        markers = [part["text"] for content in contents for part in content.get("parts", [])
                   if isinstance(part, dict) and isinstance(part.get("text"), str)
                   and "<application-tool-result-data>" in part["text"]]
        assert len(markers) == 1
        routed_data = json.loads(markers[0].split(
            "<application-tool-result-data>\n", 1
        )[1].split("\n</application-tool-result-data>", 1)[0])
        assert routed_data == {
            "arguments": {"folder_path": ["Client"]},
            "name": "list_recent_transcripts",
            "result": application_result,
        }
        parts = [part for content in contents for part in content.get("parts", [])]
        assert not any("functionCall" in part or "functionResponse" in part for part in parts)
        assert payload["generationConfig"]["thinkingConfig"]["thinking_level"] == "MEDIUM"
        assert {item["name"] for item in payload["tools"][0]["functionDeclarations"]} == {
            "query", "get_transcript", "list_recent_transcripts", "list_folders",
            "create_folder", "rename_folder", "move_folder", "move_transcript",
            "rename_transcript", "add_fact", "sync_status", "sync_changes", "add_dialog",
        }
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": "The milestone date is June 12."}],
        }}]})

    monkeypatch.setattr(genai, "Client", sdk_fake_module._REAL_GENAI_CLIENT)
    gemini_http = httpx.Client(transport=httpx.MockTransport(gemini_response))
    sdk_client = genai.Client(
        api_key="synthetic-gemini-key",
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1/v1beta", httpx_client=gemini_http,
        ),
    )
    monkeypatch.setattr(
        "core.integrations.gemini.agent.gemini_client.make_client",
        lambda **_kwargs: sdk_client,
    )
    tool_calls = []
    from core.operations import registry
    original_invoke_operation = registry.invoke_operation

    def observe_registry_call(name, arguments, *, context=None, retry_policy=None):
        tool_calls.append((name, arguments, context, retry_policy))
        return original_invoke_operation(
            name, arguments, context=context, retry_policy=retry_policy,
        )

    monkeypatch.setattr(registry, "invoke_operation", observe_registry_call)
    monkeypatch.setattr(
        "core.integrations.firestore.transcripts.get_client", lambda: db,
    )
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    update = {"update_id": 24001, "message": {
        "message_id": 161, "date": 1793320000,
        "chat": {"id": 77, "type": "private"},
        "from": {"id": 7, "first_name": "Tester"},
        "text": route["substantive_text"],
    }}

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    intake = asyncio.run(post("/telegram/webhook", update, secret=True),
                         loop_factory=_selector_loop)
    assert intake.status_code == 200, intake.text
    dispatch = next(item["body"] for item in task_requests
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and not item["body"].get("notification_id"))
    processed = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert processed.status_code == 200, processed.text
    assert processed.json() in ({"status": "processed"}, {"status": "noop"})
    operation = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(dispatch["operation_id"]).get().to_dict())
    assert operation["state"] == "completed"

    assert len(wire_requests) == 2
    assert len(tool_calls) == 1
    assert tool_calls[0][0:2] == (
        "list_recent_transcripts", {"folder_path": ["Client"]},
    )
    assert tool_calls[0][2] is None
    assert tool_calls[0][3] is not None
    assert [payload["text"] for method, payload in sent if method == "sendMessage"] == [
        "The milestone date is June 12.",
    ], sent
    assert operation["route_control"]["target_doc_id"] == "doc-selected"
    assert operation["route_control"]["tool_choice"]["name"] == "list_recent_transcripts"
    turns = repository.list_turns(context_id, "conversation-selected")
    assert [turn["role"] for turn in turns] == ["user", "model", "user", "model"]
    assert turns[-2]["operation_id"] == dispatch["operation_id"]
    assert turns[-2]["parts"][0]["text"] == route["substantive_text"]
    assert turns[-1]["parts"][0]["text"] == "The milestone date is June 12."
    assert all("application-tool-result-data" not in str(turn) for turn in turns)

    duplicate = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(wire_requests) == 2
    assert len(tool_calls) == 1
    assert len([payload for method, payload in sent
                if method == "sendMessage"
                and payload.get("text") == "The milestone date is June 12."]) == 1
    gemini_http.close()


def test_default_asgi_malformed_route_error_survives_delivery_replay_and_uncertainty_comparison(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Exercise malformed save routing, durable error replay, and genuine uncertainty."""
    import httpx
    from google import genai
    from google.genai import types
    from tests.fixtures import sdk_fakes as sdk_fake_module
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    for name, value in {
        "TELEGRAM_BOT_TOKEN": "123456:synthetic-token",
        "TELEGRAM_CONTROL_QUEUE": "pytest-telegram-control-queue",
        "TELEGRAM_CONTROL_URL": "https://telegram.pytest.invalid/internal/dispatch",
        "CLOUD_TASKS_OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
        "OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
    }.items():
        monkeypatch.setenv(name, value)

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    processing_error = "An error occurred while processing your request. Please try again."
    send_attempts = []
    delivered = []
    unknown_error_attempted = False

    def telegram_response(request):
        nonlocal unknown_error_attempted
        payload = json.loads(request.content)
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "sendMessage":
            send_attempts.append(payload)
            if payload.get("text") == processing_error and not unknown_error_attempted:
                # Simulate an accepted send whose acknowledgement was lost.
                unknown_error_attempted = True
                return httpx.Response(200, json={"ok": True, "result": {}})
            delivered.append(payload)
            return httpx.Response(
                200, json={"ok": True, "result": {"message_id": 900 + len(delivered)}},
            )
        return httpx.Response(200, json={"ok": True, "result": True})

    original_telegram_init = TelegramClient.__init__
    monkeypatch.setattr(TelegramClient, "__original_init__", original_telegram_init, raising=False)
    monkeypatch.setattr(
        TelegramClient, "__init__",
        lambda self, token, *, http_client=None, timeout=30.0: original_telegram_init(
            self, token, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(telegram_response),
            ), timeout=timeout,
        ),
    )

    malformed_save = {
        "kind": "save", "target_ref": None, "target_operation_id": "synthetic-forbidden-target",
        "dialog_number": None, "result_refs": [], "substantive_text": None,
        "tool_choice": None, "certainty": "certain",
    }
    uncertain_route = {
        "kind": "substantive", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "synthetic question",
        "tool_choice": None, "certainty": "uncertain",
    }
    route_responses = [malformed_save, uncertain_route]
    wire_requests = []

    def gemini_response(request):
        wire_requests.append(request)
        assert route_responses, "unexpected extra Gemini generation"
        route = route_responses.pop(0)
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"functionCall": {
                "name": "route_command", "args": route,
            }}],
        }}]})

    monkeypatch.setattr(genai, "Client", sdk_fake_module._REAL_GENAI_CLIENT)
    gemini_http = httpx.Client(transport=httpx.MockTransport(gemini_response))
    sdk_client = genai.Client(
        api_key="synthetic-gemini-key",
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1/v1beta", httpx_client=gemini_http,
        ),
    )
    monkeypatch.setattr(
        "core.integrations.gemini.agent.gemini_client.make_client",
        lambda **_kwargs: sdk_client,
    )
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://telegram.test",
        ) as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    repository = TelegramRepository(client=db)
    forward = {"update_id": 17701, "message": {
        "message_id": 71, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Synthetic"},
                           "date": 1793310000},
        "text": "A synthetic forwarded line remains source data.",
    }}
    accepted_forward = asyncio.run(post("/telegram/webhook", forward, secret=True),
                                   loop_factory=_selector_loop)
    assert accepted_forward.status_code == 200, accepted_forward.text
    forward_ack = next(item["body"] for item in task_requests
                       if item["body"].get("target_kind") == "telegram_dispatch"
                       and item["body"].get("notification_id"))
    assert asyncio.run(post("/internal/deliver", forward_ack),
                       loop_factory=_selector_loop).status_code == 200

    async def accept_and_get_command_dispatch(update_id, message_id, text):
        response = await post("/telegram/webhook", {"update_id": update_id, "message": {
            "message_id": message_id, "date": 1793320010,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7}, "text": text,
        }}, secret=True)
        assert response.status_code == 200, response.text
        rows = [item["body"] for item in task_requests
                if item["body"].get("target_kind") == "telegram_dispatch"]
        return next(row for row in reversed(rows) if not row.get("notification_id"))

    save_dispatch = asyncio.run(
        accept_and_get_command_dispatch(17702, 72, "save this synthetic dialog"),
        loop_factory=_selector_loop,
    )
    first = asyncio.run(post("/internal/dispatch", save_dispatch),
                        loop_factory=_selector_loop)
    assert first.status_code == 200, first.text
    assert first.json() == {"status": "noop"}
    context_id, operation_id = save_dispatch["context_id"], save_dispatch["operation_id"]
    operation_ref = (db.collection("telegram_sessions").document(context_id)
                     .collection("operations").document(operation_id))
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "running"
    assert operation["route_control"]["status"] == "invalid_route"
    assert len(wire_requests) == 1
    assert not any(event["boundary"] == "drive" and event["operation"] == "files.create"
                   for event in sdk_fakes.trace)
    assert not db.collection("dialog_operations").document(operation_id).get().exists
    assert not any(item["url"].startswith("https://vector-sync.pytest.invalid")
                   for item in task_requests)
    draft_id = db.collection("telegram_sessions").document(context_id).get().to_dict()[
        "active_draft_id"
    ]
    assert repository.list_messages(context_id, draft_id)[0]["text"] == (
        "A synthetic forwarded line remains source data."
    )

    wake = next(item["body"] for item in task_requests
                if item["body"].get("target_kind") == "telegram_dispatch"
                and item["body"].get("operation_id") == operation_id
                and item["body"].get("outbox_id") != save_dispatch["outbox_id"])
    from core.integrations.firestore.outbox import get_outbox_record
    wake_record = get_outbox_record(db, wake["outbox_id"])
    assert wake_record is not None
    worker = app.state.telegram_queue_worker
    replay_clock = lambda: wake_record.schedule_at
    worker.clock = replay_clock
    worker.processor.clock = replay_clock
    replayed = asyncio.run(post("/internal/dispatch", wake), loop_factory=_selector_loop)
    assert replayed.status_code == 200, replayed.text
    assert replayed.json() == {"status": "processed"}
    operation = operation_ref.get().to_dict()
    context = db.collection("telegram_sessions").document(context_id).get().to_dict()
    assert operation["state"] == "failed"
    assert operation["route_control"]["status"] == "invalid_route"
    assert context.get("active_operation_id") is None
    assert context.get("lock_owner") is None
    assert len(wire_requests) == 1
    assert [item.get("text") for item in delivered].count(processing_error) == 1
    assert len([item for item in send_attempts if item.get("text") == processing_error]) == 2
    assert repository.list_messages(context_id, draft_id)

    duplicate = asyncio.run(post("/internal/dispatch", wake), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == {"status": "noop"}
    assert len(wire_requests) == 1
    assert len([item for item in send_attempts if item.get("text") == processing_error]) == 2

    uncertain_dispatch = asyncio.run(
        accept_and_get_command_dispatch(17703, 73, "is this synthetic request clear?"),
        loop_factory=_selector_loop,
    )
    uncertain_result = asyncio.run(post("/internal/dispatch", uncertain_dispatch),
                                   loop_factory=_selector_loop)
    assert uncertain_result.status_code == 200, uncertain_result.text
    assert uncertain_result.json() == {"status": "processed"}
    uncertain_operation = (db.collection("telegram_sessions").document(context_id)
                           .collection("operations").document(
                               uncertain_dispatch["operation_id"]).get().to_dict())
    assert uncertain_operation["state"] == "completed"
    assert uncertain_operation["route_control"]["status"] == "uncertain"
    assert delivered[-1]["text"] == "I could not determine what you want me to do. Please clarify."
    assert len(wire_requests) == 2
    assert repository.list_messages(context_id, draft_id)
    gemini_http.close()


def test_default_asgi_selected_document_routes_query_then_answers_over_real_gemini_http(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Route a selected-document query through shared retrieval and untrusted result context."""
    import httpx
    from google import genai
    from google.genai import types
    from types import SimpleNamespace
    from tests.fixtures import sdk_fakes as sdk_fake_module
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.google.com/drive/folders/pytest-synthetic-root")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return TelegramClient.__original_init__(
            self, token, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(bot_response),
            ), timeout=timeout,
        )

    original_telegram_init = TelegramClient.__init__
    monkeypatch.setattr(TelegramClient, "__original_init__", original_telegram_init, raising=False)
    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    repository = TelegramRepository(client=db)
    context_id = "77_default"
    now = datetime.now(timezone.utc)
    db.collection("folders").document("pytest-synthetic-root").set({
        "folder_id": "pytest-synthetic-root", "name": "Root", "parent_id": None,
    })
    db.collection("folders").document("folder-client").set({
        "folder_id": "folder-client", "name": "Client",
        "parent_id": "pytest-synthetic-root",
    })
    db.collection("transcript_index").document("doc-selected").set({
        "source_file": "Selected meeting notes", "parent_id": "folder-client",
        "dialog_date": "2026-10-22", "provider": "telegram", "status": "synced",
    })
    number = repository.bind_document(
        context_id, "doc-selected", "conversation-selected",
        metadata={"title": "Selected meeting notes", "folder_path": []},
        now=now, update_last_doc=False,
    )
    repository.ensure_conversation(
        context_id, "conversation-selected", kind="document", doc_id="doc-selected",
        dialog_number=number, now=now,
    )
    db.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-selected",
        "selected_dialog_number": number,
    })
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-question",
        operation_id="prior-operation", role="user",
        parts=[{"text": "What did the team decide about the schedule?"}], now=now,
    )
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-answer",
        operation_id="prior-operation", role="model",
        parts=[{"text": "They moved the launch to May."}], now=now,
    )

    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    from core.integrations.qdrant import client as qdrant_client
    monkeypatch.setattr(qdrant_client, "_single_attempt_client", None)
    monkeypatch.setattr(qdrant_client, "_client", None)
    summary_hit = SimpleNamespace(
        id="synthetic-summary", version=1, score=0.95,
        payload={
            "doc_id": "doc-selected", "parent_id": "folder-client",
            "type": "summary", "center_index": 0, "covered_range": [0, 0],
        },
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[summary_hit]),
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[summary_hit]),
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[]),
    )
    sdk_fakes.qdrant.queue("scroll", response=([SimpleNamespace(
        id="synthetic-utterance",
        payload={
            "doc_id": "doc-selected", "type": "utterance", "order_index": 0,
            "dialog_date": "2026-10-22", "speaker": "Team",
            "text": "The milestone date is June 12.",
        },
    )], None))

    route = {
        "kind": "substantive", "target_ref": "doc-selected",
        "target_operation_id": None, "dialog_number": number, "result_refs": [],
        "substantive_text": "What is the milestone date in the saved notes?",
        "tool_choice": {
            "name": "query", "arguments": {"query": "milestone date in saved notes"},
        },
        "certainty": "certain",
    }
    wire_requests = []
    generation_requests = 0

    def gemini_response(request):
        nonlocal generation_requests
        wire_requests.append(request)
        if request.url.path.endswith(":batchEmbedContents"):
            return httpx.Response(200, json={"embeddings": [{"values": [0.25] * 768}]})
        generation_requests += 1
        payload = json.loads(request.content)
        serialized_contents = json.dumps(payload["contents"], ensure_ascii=False)
        assert "What did the team decide about the schedule?" in serialized_contents
        assert "They moved the launch to May." in serialized_contents
        if generation_requests == 1:
            assert payload["tools"][0]["functionDeclarations"][0]["name"] == "route_command"
            assert payload["generationConfig"]["thinkingConfig"]["thinking_level"] == "MINIMAL"
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "route_command", "args": route},
                    "thoughtSignature": "c3ludGhldGljLXJvdXRlLXNpZw==",
                }],
            }}]})

        assert generation_requests == 2
        assert payload["generationConfig"]["thinkingConfig"]["thinking_level"] == "MEDIUM"
        assert payload["contents"][-1]["parts"][0]["text"] == route["substantive_text"]
        markers = [part["text"] for content in payload["contents"]
                   for part in content.get("parts", [])
                   if isinstance(part, dict) and isinstance(part.get("text"), str)
                   and "<application-tool-result-data>" in part["text"]]
        assert len(markers) == 1
        routed_data = json.loads(markers[0].split(
            "<application-tool-result-data>\n", 1
        )[1].split("\n</application-tool-result-data>", 1)[0])
        assert routed_data["name"] == "query"
        assert routed_data["arguments"] == route["tool_choice"]["arguments"]
        result_rows, result_metadata = routed_data["result"]
        assert result_rows[0]["doc_id"] == "doc-selected"
        assert result_rows[0]["segments"][0]["dialog"][0]["text"] == (
            "The milestone date is June 12."
        )
        assert result_metadata["total_matches"] == 1
        parts = [part for content in payload["contents"]
                 for part in content.get("parts", [])]
        assert not any("functionCall" in part or "functionResponse" in part for part in parts)
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": "The milestone date is June 12."}],
        }}]})

    monkeypatch.setattr(genai, "Client", sdk_fake_module._REAL_GENAI_CLIENT)
    gemini_http = httpx.Client(transport=httpx.MockTransport(gemini_response))
    sdk_client = genai.Client(
        api_key="synthetic-gemini-key",
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1/v1beta", httpx_client=gemini_http,
        ),
    )
    monkeypatch.setattr(
        "core.integrations.gemini.agent.gemini_client.make_client",
        lambda **_kwargs: sdk_client,
    )
    monkeypatch.setattr(
        "core.integrations.gemini.embeddings.make_client",
        lambda **_kwargs: sdk_client,
    )
    tool_calls = []
    from core.operations import registry
    original_invoke_operation = registry.invoke_operation

    def observe_registry_call(name, arguments, *, context=None, retry_policy=None):
        tool_calls.append((name, arguments, context, retry_policy))
        return original_invoke_operation(
            name, arguments, context=context, retry_policy=retry_policy,
        )

    monkeypatch.setattr(registry, "invoke_operation", observe_registry_call)
    monkeypatch.setattr(
        "core.integrations.firestore.transcripts.get_client", lambda: db,
    )
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    update = {"update_id": 24001, "message": {
        "message_id": 161, "date": 1793320000,
        "chat": {"id": 77, "type": "private"},
        "from": {"id": 7, "first_name": "Tester"},
        "text": route["substantive_text"],
    }}

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    intake = asyncio.run(post("/telegram/webhook", update, secret=True),
                         loop_factory=_selector_loop)
    assert intake.status_code == 200, intake.text
    dispatch = next(item["body"] for item in task_requests
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and not item["body"].get("notification_id"))
    processed = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert processed.status_code == 200, processed.text
    assert processed.json() in ({"status": "processed"}, {"status": "noop"})
    operation = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(dispatch["operation_id"]).get().to_dict())
    assert operation["state"] == "completed"

    assert len(wire_requests) == 4
    assert len(tool_calls) == 1
    assert tool_calls[0][0:2] == (
        "query", {"query": "milestone date in saved notes"},
    )
    assert tool_calls[0][2] is None
    assert tool_calls[0][3] is not None
    delivered = [payload["text"] for method, payload in sent if method == "sendMessage"]
    assert len(delivered) == 1
    assert delivered == ["The milestone date is June 12."]
    assert operation["route_control"]["target_doc_id"] == "doc-selected"
    assert operation["route_control"]["tool_choice"]["name"] == "query"
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant"
                and event["operation"] == "query_points"]) == 3
    assert sum(request.url.path.endswith(":generateContent")
               for request in wire_requests) == 2
    assert sum(request.url.path.endswith(":batchEmbedContents")
               for request in wire_requests) == 2
    turns = repository.list_turns(context_id, "conversation-selected")
    assert [turn["role"] for turn in turns] == ["user", "model", "user", "model"]
    assert turns[0]["parts"][0]["text"] == "What did the team decide about the schedule?"
    assert turns[1]["parts"][0]["text"] == "They moved the launch to May."
    assert turns[-2]["operation_id"] == dispatch["operation_id"]
    assert turns[-2]["parts"][0]["text"] == route["substantive_text"]
    assert turns[-1]["parts"][0]["text"] == delivered[0]
    assert all("application-tool-result-data" not in str(turn) for turn in turns)

    duplicate = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(wire_requests) == 4
    assert len(tool_calls) == 1
    assert len([payload for method, payload in sent
                if method == "sendMessage"
                and payload.get("text") == delivered[0]]) == 1
    gemini_http.close()


@pytest.mark.parametrize("include_result", [True, False])
def test_default_asgi_selected_dialog_search_creates_isolated_session_over_real_gemini_http(
    firestore_fake, sdk_fakes, monkeypatch, include_result,
):
    """Route search from a selected document into a separate durable session."""
    import httpx
    from google import genai
    from google.genai import types
    from types import SimpleNamespace
    from tests.fixtures import sdk_fakes as sdk_fake_module
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.google.com/drive/folders/pytest-synthetic-root")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)

    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return TelegramClient.__original_init__(
            self, token, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(bot_response),
            ), timeout=timeout,
        )

    original_telegram_init = TelegramClient.__init__
    monkeypatch.setattr(TelegramClient, "__original_init__", original_telegram_init, raising=False)
    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    repository = TelegramRepository(client=db)
    context_id = "77_default"
    now = datetime.now(timezone.utc)
    db.collection("folders").document("pytest-synthetic-root").set({
        "folder_id": "pytest-synthetic-root", "name": "Root", "parent_id": None,
    })
    db.collection("folders").document("folder-client").set({
        "folder_id": "folder-client", "name": "Client",
        "parent_id": "pytest-synthetic-root",
    })
    db.collection("transcript_index").document("doc-selected").set({
        "source_file": "Selected meeting notes", "parent_id": "folder-client",
        "dialog_date": "2026-10-22", "provider": "telegram", "status": "synced",
    })
    number = repository.bind_document(
        context_id, "doc-selected", "conversation-selected",
        metadata={"title": "Selected meeting notes", "folder_path": []},
        now=now, update_last_doc=False,
    )
    repository.ensure_conversation(
        context_id, "conversation-selected", kind="document", doc_id="doc-selected",
        dialog_number=number, now=now,
    )
    db.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-selected",
        "selected_dialog_number": number,
        "last_doc_id": "doc-selected",
    })
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-question",
        operation_id="prior-operation", role="user",
        parts=[{"text": "What did the team decide about the schedule?"}], now=now,
    )
    repository.append_turn(
        context_id, "conversation-selected", turn_id="prior-answer",
        operation_id="prior-operation", role="model",
        parts=[{"text": "They moved the launch to May."}], now=now,
    )

    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    from core.integrations.qdrant import client as qdrant_client
    monkeypatch.setattr(qdrant_client, "_single_attempt_client", None)
    monkeypatch.setattr(qdrant_client, "_client", None)
    summary_hit = SimpleNamespace(
        id="synthetic-summary", version=1, score=0.95,
        payload={
            "doc_id": "doc-selected", "parent_id": "folder-client",
            "type": "summary", "center_index": 0, "covered_range": [0, 0],
        },
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[summary_hit] if include_result else []),
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[summary_hit] if include_result else []),
    )
    sdk_fakes.qdrant.queue(
        "query_points", response=SimpleNamespace(points=[]),
    )
    sdk_fakes.qdrant.queue("scroll", response=([SimpleNamespace(
        id="synthetic-utterance",
        payload={
            "doc_id": "doc-selected", "type": "utterance", "order_index": 0,
            "dialog_date": "2026-10-22", "speaker": "Team",
            "text": "The milestone date is June 12.",
        },
    )] if include_result else [], None))

    route = {
        "kind": "search", "target_ref": None,
        "target_operation_id": None, "dialog_number": None, "result_refs": [],
        "substantive_text": None, "tool_choice": None,
        "certainty": "certain",
    }
    search_text = "Find the synthetic schedule notes"
    wire_requests = []
    generation_requests = 0

    def gemini_response(request):
        nonlocal generation_requests
        wire_requests.append(request)
        if request.url.path.endswith(":batchEmbedContents"):
            return httpx.Response(200, json={"embeddings": [{"values": [0.25] * 768}]})
        generation_requests += 1
        payload = json.loads(request.content)
        serialized_contents = json.dumps(payload["contents"], ensure_ascii=False)
        assert "What did the team decide about the schedule?" in serialized_contents
        assert "They moved the launch to May." in serialized_contents
        assert search_text in serialized_contents
        if generation_requests == 1:
            assert payload["tools"][0]["functionDeclarations"][0]["name"] == "route_command"
            assert payload["generationConfig"]["thinkingConfig"]["thinking_level"] == "MINIMAL"
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "route_command", "args": route},
                    "thoughtSignature": "c3ludGhldGljLXJvdXRlLXNpZw==",
                }],
            }}]})

        raise AssertionError("search route must not start a second generation")

    monkeypatch.setattr(genai, "Client", sdk_fake_module._REAL_GENAI_CLIENT)
    gemini_http = httpx.Client(transport=httpx.MockTransport(gemini_response))
    sdk_client = genai.Client(
        api_key="synthetic-gemini-key",
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1/v1beta", httpx_client=gemini_http,
        ),
    )
    monkeypatch.setattr(
        "core.integrations.gemini.agent.gemini_client.make_client",
        lambda **_kwargs: sdk_client,
    )
    monkeypatch.setattr(
        "core.integrations.gemini.embeddings.make_client",
        lambda **_kwargs: sdk_client,
    )
    monkeypatch.setattr(
        "core.integrations.firestore.transcripts.get_client", lambda: db,
    )
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    update = {"update_id": 24001, "message": {
        "message_id": 161, "date": 1793320000,
        "chat": {"id": 77, "type": "private"},
        "from": {"id": 7, "first_name": "Tester"},
        "text": search_text,
    }}

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    intake = asyncio.run(post("/telegram/webhook", update, secret=True),
                         loop_factory=_selector_loop)
    assert intake.status_code == 200, intake.text
    dispatch = next(item["body"] for item in task_requests
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and not item["body"].get("notification_id"))
    processed = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert processed.status_code == 200, processed.text
    assert processed.json() in ({"status": "processed"}, {"status": "noop"})
    operation = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(dispatch["operation_id"]).get().to_dict())
    assert operation["state"] == "completed"

    delivered = [payload["text"] for method, payload in sent if method == "sendMessage"]
    assert len(delivered) == 1
    if include_result:
        assert delivered[0].startswith("Search results:")
        assert "Selected meeting notes" in delivered[0]
    else:
        assert delivered[0] == "No matching documents were found."
    assert operation["route_control"]["kind"] == "search"
    assert operation["route_control"].get("tool_choice") is None
    search_session_id = operation["route_control"]["search_session_id"]
    search_conversation = repository.get_conversation(context_id, search_session_id)
    assert search_conversation.kind.value == "search"
    session_turns = repository.list_turns(context_id, search_session_id)
    assert any(turn["role"] == "tool" for turn in session_turns)
    context_record = db.collection("telegram_sessions").document(context_id).get().to_dict()
    assert context_record["active_conversation_id"] == search_session_id
    assert context_record["selected_dialog_number"] is None
    assert context_record["last_doc_id"] == "doc-selected"
    assert len([payload for method, payload in sent
                if method == "sendMessage" and "Starting to process dialog" in payload["text"]]) == 0
    assert len(db.collection("transcript_index").stream()) == 1
    drafts = db.collection("telegram_sessions").document(context_id).collection("drafts").stream()
    assert drafts == []
    turns = repository.list_turns(context_id, "conversation-selected")
    assert [turn["role"] for turn in turns] == ["user", "model"]
    assert turns[0]["parts"][0]["text"] == "What did the team decide about the schedule?"
    assert turns[1]["parts"][0]["text"] == "They moved the launch to May."
    retrieval_query_count = len([event for event in sdk_fakes.trace
                                 if event["boundary"] == "qdrant"
                                 and event["operation"] == "query_points"])
    assert retrieval_query_count == 3
    assert sum(request.url.path.endswith(":generateContent")
               for request in wire_requests) == 1
    assert sum(request.url.path.endswith(":batchEmbedContents")
               for request in wire_requests) == 2

    duplicate = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(wire_requests) == 3
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant"
                and event["operation"] == "query_points"]) == retrieval_query_count
    assert len([payload for method, payload in sent
                if method == "sendMessage"
                and payload.get("text") == delivered[0]]) == 1
    gemini_http.close()
