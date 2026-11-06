"""Offline composition checks for Telegram's real default runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timedelta, timezone

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
    agent_answer = "* **Root & Co**\n  * **Nested_Folder**\n* **Another**"
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=agent_answer)]))],
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
    context_id, operation_id = body["context_id"], body["operation_id"]
    rich_replies = [payload["rich_message"]["markdown"] for method, payload in sent
                    if method == "sendRichMessage"]
    assert rich_replies == [agent_answer]
    assert len([payload for method, payload in sent if method == "sendRichMessage"]) == 1
    physical_sends = [(method, payload) for method, payload in sent
                      if method in {"sendMessage", "sendRichMessage"}]
    assert len(physical_sends) == 1
    assert physical_sends[0][0] == "sendRichMessage"
    assert physical_sends[0][1]["rich_message"]["markdown"] == agent_answer
    assert all("text" not in payload and "parse_mode" not in payload
               for method, payload in sent if method == "sendRichMessage")
    operation = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(operation_id).get().to_dict())
    assert operation["state"] == "completed"
    assert operation["notification_id"]
    notification = db.collection("notification_outbox").document(
        operation["notification_id"]
    ).get().to_dict()
    assert notification["state"] == "delivered"
    assert notification["body"] == agent_answer
    assert notification["presentation"] == "rich_markdown"
    context = db.collection("telegram_sessions").document(context_id).get().to_dict()
    turns = TelegramRepository(db).list_turns(context_id, context["active_conversation_id"])
    assert turns[-1]["parts"][0]["text"] == agent_answer

    sends_before_replay = len(sent)
    model_calls_before_replay = len([
        row for row in sdk_fakes.trace
        if row["boundary"] == "gemini" and row["operation"] == "generate_content"
    ])
    replay = asyncio.run(dispatch_task(), loop_factory=_selector_loop)
    assert replay.status_code == 200
    assert replay.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(sent) == sends_before_replay
    assert len([row for row in sdk_fakes.trace
                if row["boundary"] == "gemini"
                and row["operation"] == "generate_content"]) == model_calls_before_replay
    assert len([(method, payload) for method, payload in sent
                if method in {"sendMessage", "sendRichMessage"}]) == 1
    assert [payload["rich_message"]["markdown"] for method, payload in sent
            if method == "sendRichMessage"] == [agent_answer]


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

    assert [payload["rich_message"]["markdown"] for method, payload in sent
            if method == "sendRichMessage"][-3:] == [
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


def test_default_app_reuses_one_card_for_transport_burst_after_sequential_webhook_gaps(
    firestore_fake, monkeypatch,
):
    """Drive delayed burst members through authenticated ASGI and canonical tasks."""
    from datetime import timedelta
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository

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

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []

    def bot_response(request):
        method = request.url.path.rsplit("/", 1)[-1]
        payload = json.loads(request.content)
        sent.append((method, payload))
        if method == "answerCallbackQuery":
            return httpx.Response(200, json={"ok": True, "result": True})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response)), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    seen_outboxes = set()

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    def deliver_due_notifications():
        due = [item for item in task_requests
               if item.get("body", {}).get("target_kind") == "telegram_dispatch"
               and item["body"].get("notification_id")
               and item["body"].get("outbox_id") not in seen_outboxes
               and (item.get("schedule_time") is None
                    or item["schedule_time"] <= clock_now[0])]
        for item in due:
            body = item["body"]
            record = get_outbox_record(db, body["outbox_id"])
            assert record is not None and task_payload(record) == body
            response = asyncio.run(post("/internal/deliver", body), loop_factory=_selector_loop)
            assert response.status_code == 200, response.text
            seen_outboxes.add(body["outbox_id"])

    transport_base = 1_793_320_000
    previous_webhook_time = clock_now[0]
    old_save_data = None
    old_card = None
    for index, (gap, transport_offset) in enumerate(((0, 0), (9.7, 1), (1.2, 1), (1.2, 2))):
        if gap:
            elapsed = (clock_now[0] - previous_webhook_time).total_seconds()
            clock_now[0] += timedelta(seconds=max(0, gap - elapsed))
        update = {"update_id": 19001 + index, "message": {
            "message_id": 101 + index, "date": transport_base + transport_offset,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
            "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                               "date": transport_base - 5000 - index},
            "text": f"synthetic forwarded source {index + 1}",
        }}
        accepted = asyncio.run(post("/telegram/webhook", update, secret=True),
                               loop_factory=_selector_loop)
        assert accepted.status_code == 200, accepted.text
        previous_webhook_time = clock_now[0]
        deliver_due_notifications()
        if index == 1:
            reopened_edits = [(method, payload) for method, payload in sent
                              if method == "editMessageText"
                              and payload.get("message_id") == old_card["message_id"]]
            assert reopened_edits
            thinking_edit = reopened_edits[-1][1]
            assert "Thinking" in thinking_edit["text"]
            assert thinking_edit["reply_markup"] == {"inline_keyboard": []}
            stale = {"update_id": 19100, "callback_query": {
                "id": "stale-burst-save", "from": {"id": 7}, "data": old_save_data,
                "message": {"message_id": old_card["message_id"],
                            "date": transport_base + 10,
                            "chat": {"id": 77, "type": "private"}},
            }}
            stale_response = asyncio.run(post("/telegram/webhook", stale, secret=True),
                                         loop_factory=_selector_loop)
            assert stale_response.status_code == 200, stale_response.text
            assert not any(row.to_dict().get("update_id") == 19100 for row in
                           db.collection("telegram_sessions").document(old_card["context_id"])
                           .collection("commands").stream())
        if index < 3:
            clock_now[0] += timedelta(seconds=1.1)
            deliver_due_notifications()
            if index == 0:
                initial_card = next(item["body"]["notification_id"] for item in task_requests
                                    if item.get("body", {}).get("target_kind") == "telegram_dispatch"
                                    and item["body"].get("notification_id"))
                old_card = TelegramRepository(db).get_notification(initial_card)
                old_save_data = old_card["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    clock_now[0] += timedelta(seconds=1.1)
    deliver_due_notifications()

    sends = [(method, payload) for method, payload in sent if method == "sendMessage"]
    edits = [(method, payload) for method, payload in sent if method == "editMessageText"]
    assert len(sends) == 1
    notification_id = next(item["body"]["notification_id"] for item in task_requests
                           if item.get("body", {}).get("target_kind") == "telegram_dispatch"
                           and item["body"].get("notification_id"))
    card = TelegramRepository(db).get_notification(notification_id)
    assert card["receipt_accepted_count"] == 4
    assert card["state"] == "delivered"
    assert card["reply_markup"]["inline_keyboard"][0][0]["text"] == "Save"
    assert {payload["message_id"] for _, payload in edits} == {card["message_id"]}
    assert any("Ready" in payload.get("text", "") for _, payload in edits)
    assert any("Dialog #1" in payload.get("text", "") for _, payload in edits)
    assert len({item["body"]["notification_id"] for item in task_requests
                if item.get("body", {}).get("target_kind") == "telegram_dispatch"
                and item["body"].get("notification_id")}) == 1

    callback_data = card["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    current = {"update_id": 19101, "callback_query": {
        "id": "current-burst-save", "from": {"id": 7}, "data": callback_data,
        "message": {"message_id": card["message_id"], "date": transport_base + 10,
                    "chat": {"id": 77, "type": "private"}},
    }}
    current_response = asyncio.run(post("/telegram/webhook", current, secret=True),
                                   loop_factory=_selector_loop)
    assert current_response.status_code == 200, current_response.text
    commands = [row.to_dict() for row in db.collection("telegram_sessions").document(
        card["context_id"]).collection("commands").stream()]
    accepted_save = next(command for command in commands if command.get("update_id") == 19101)
    assert accepted_save["cutoff_message_ids"] == [101, 102, 103, 104]
    assert accepted_save["action_metadata"]["action"] == "receipt_save"


def test_default_app_composer_annotation_then_save_runs_canonical_pipeline_and_waits_for_real_index_receipt(
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
    annotation_text = (
        "Мой диалог с мамой; дополнительный контекст 🧭 " * 699
        + "Мой диалог с мамой; дополнительный контекст 🧭"
    )
    annotation_route = {
        "kind": "annotation", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": annotation_text,
        "tool_choice": None, "certainty": "certain",
    }
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="route_command", args=annotation_route),
        )]))],
    ))
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

    candidate = {"update_id": 15000, "message": {
        "message_id": 60, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": annotation_text,
    }}
    candidate_response = asyncio.run(post("/telegram/webhook", candidate, secret=True),
                                     loop_factory=_selector_loop)
    assert candidate_response.status_code == 200, candidate_response.text
    candidate_dispatch = next(item["body"] for item in reversed(task_requests)
                              if item["body"].get("target_kind") == "telegram_dispatch"
                              and not item["body"].get("notification_id"))
    first_routing = asyncio.run(post("/internal/dispatch", candidate_dispatch),
                                loop_factory=_selector_loop)
    assert first_routing.status_code == 200, first_routing.text
    candidate_operation = (db.collection("telegram_sessions").document(
        candidate_dispatch["context_id"]).collection("operations").document(
            candidate_dispatch["operation_id"]).get().to_dict())
    assert candidate_operation["state"] in {"running", "pending"}
    assert candidate_operation["route_control"]["kind"] == "annotation"
    assert candidate_operation["route_control"].get("target_doc_id") is None

    for index in range(6):
        forward = {"update_id": 15001 + index, "message": {
        "message_id": 61 + index, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000 + index},
        "text": f"The exact forwarded source line {index + 1}.",
    }}
        forward_response = asyncio.run(post("/telegram/webhook", forward, secret=True),
                                       loop_factory=_selector_loop)
        assert forward_response.status_code == 200, forward_response.text
    forward_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    forward_ack = next(
        body for body in forward_tasks if body.get("notification_id")
        and TelegramRepository(db).get_notification(body["notification_id"]).get("receipt_card") is True
    )
    ack_done = asyncio.run(post("/internal/deliver", forward_ack),
                           loop_factory=_selector_loop)
    assert ack_done.status_code == 200, ack_done.text
    assert any(method == "sendMessage" for method, _ in sent)  # durable forward acknowledgement
    annotation_wake = next(item["body"] for item in reversed(task_requests)
                           if item["body"].get("target_kind") == "telegram_dispatch"
                           and item["body"].get("operation_id") == candidate_dispatch["operation_id"]
                           and item["body"].get("outbox_id") != candidate_dispatch["outbox_id"]
                           and not item["body"].get("notification_id"))
    annotation_wake_record = get_outbox_record(db, annotation_wake["outbox_id"])
    annotation_clock = lambda: annotation_wake_record.schedule_at
    app.state.telegram_queue_worker.clock = annotation_clock
    app.state.telegram_queue_worker.processor.clock = annotation_clock
    annotation_resumed = asyncio.run(post("/internal/dispatch", annotation_wake),
                                     loop_factory=_selector_loop)
    assert annotation_resumed.status_code == 200, annotation_resumed.text
    annotation_operation = (db.collection("telegram_sessions").document(
        candidate_dispatch["context_id"]).collection("operations").document(
            candidate_dispatch["operation_id"]).get().to_dict())
    assert annotation_operation["state"] == "completed"
    draft_id = annotation_operation["route_control"]["target_draft_id"]
    annotation = TelegramRepository(db).list_draft_annotations(
        candidate_dispatch["context_id"], draft_id,
    )
    assert len(annotation) == 1 and annotation[0]["text"] == annotation_text
    assert annotation[0]["state"] == "pending_save"
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
    callback = {"update_id": 15007, "callback_query": {
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
                        .collection("commands").document("15007").get().to_dict())
    assert operation.get("typed_action") == accepted_command.get("action_metadata")
    assert accepted_command.get("cutoff_message_ids") == list(range(61, 67))
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
    queued_fact_operations = [snapshot.to_dict() or {} for snapshot in
                              db.collection("telegram_sessions").document(save_dispatch["context_id"])
                              .collection("operations").stream()
                              if (snapshot.to_dict() or {}).get("telegram_internal_kind")
                              == "annotation_apply"]
    assert len(queued_fact_operations) == 1
    queued_fact_id = queued_fact_operations[0]["operation_id"]
    from core.integrations.firestore.outbox import get_outbox_record
    initial_fact_outbox_id = "tgfacttask-" + hashlib.sha256(
        annotation[0]["comment_id"].encode("utf-8"),
    ).hexdigest()
    initial_fact_record = get_outbox_record(db, initial_fact_outbox_id)
    assert initial_fact_record is not None and initial_fact_record.operation_id == queued_fact_id
    initial_fact_dispatch = task_payload(initial_fact_record)
    early_fact_result = asyncio.run(post("/internal/dispatch", initial_fact_dispatch),
                                    loop_factory=_selector_loop)
    assert early_fact_result.status_code == 200, early_fact_result.text
    early_fact_row = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                      .collection("operations").document(queued_fact_id).get().to_dict())
    early_context = db.collection("telegram_sessions").document(
        save_dispatch["context_id"],
    ).get().to_dict()
    assert early_fact_row["state"] == "pending"
    assert early_context["reserved_operation_id"] == save_dispatch["operation_id"]
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
    assert manifest["ordered_message_ids"] == list(range(61, 67))
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

    annotation_ops = [snapshot.to_dict() or {} for snapshot in
                      db.collection("telegram_sessions").document(save_dispatch["context_id"])
                      .collection("operations").stream()
                      if (snapshot.to_dict() or {}).get("telegram_internal_kind") == "annotation_apply"]
    live_annotation = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                       .collection("drafts").document(draft_id).collection("annotations")
                       .document(annotation[0]["comment_id"]).get().to_dict())
    completion_receipt = None
    receipt_path = completion_certificate.get("receipt_ref")
    if isinstance(receipt_path, str):
        from services.connectors.telegram.repository._common import _document_ref
        completion_receipt = _document_ref(db, receipt_path).get().to_dict()
    assert completion_receipt is not None
    assert completion_receipt["status"] == "completed"
    assert completion_receipt["version"] == completion_certificate["receipt_version"]
    assert len(annotation_ops) == 1
    fact_operation = annotation_ops[0]
    fact_operation_id = fact_operation["operation_id"]
    from services.connectors.telegram.queue import _wake_outbox_id
    from core.integrations.firestore.outbox import get_outbox_record
    current_fact_context = db.collection("telegram_sessions").document(
        save_dispatch["context_id"],
    ).get().to_dict()
    current_fact_generation = current_fact_context.get("telegram_dispatch_wake_generation")
    fact_wakeup = get_outbox_record(db, _wake_outbox_id(
        save_dispatch["context_id"], fact_operation_id, current_fact_generation,
    )) if type(current_fact_generation) is int else None
    assert fact_wakeup is not None and fact_wakeup.operation_id == fact_operation_id
    fact_dispatch = task_payload(fact_wakeup)
    from google.genai import types
    from core.indexing.user_facts import prepare_user_fact
    _, _, fact_payload = prepare_user_fact(
        "doc-default-save", annotation_text, db=db,
    )
    for existing in ([], [],
                     [type("Point", (), {"payload": fact_payload})()],
                     [type("Point", (), {"payload": fact_payload})()],
                     [type("Point", (), {"payload": fact_payload})()]):
        sdk_fakes.qdrant.queue("retrieve", response=existing)
    sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.375] * 768)],
    ))
    from uuid import UUID
    stable_point_id = "bd62d3bd-5237-4c63-ae76-b38a1f5dfb7f"
    monkeypatch.setattr("core.operations.effects.uuid4", lambda: UUID(stable_point_id))
    upserts_before_fact = len([event for event in sdk_fakes.trace
                               if event["boundary"] == "qdrant"
                               and event["operation"] == "upsert"])
    fact_time = [fact_wakeup.schedule_at]
    fact_clock = lambda: fact_time[0]
    app.state.telegram_queue_worker.clock = fact_clock
    app.state.telegram_queue_worker.processor.clock = fact_clock
    from services.connectors.telegram import queue as telegram_queue
    complete_head = telegram_queue.complete_head
    monkeypatch.setattr(telegram_queue, "complete_head", lambda *_args, **_kwargs: False)
    interrupted_fact = asyncio.run(post("/internal/dispatch", fact_dispatch),
                                   loop_factory=_selector_loop)
    monkeypatch.setattr(telegram_queue, "complete_head", complete_head)
    assert interrupted_fact.status_code == 200, interrupted_fact.text
    assert interrupted_fact.json()["status"] == "noop"
    interrupted_row = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                       .collection("drafts").document(draft_id).collection("annotations")
                       .document(annotation[0]["comment_id"]).get().to_dict())
    assert interrupted_row["state"] == "applied"
    interrupted_operation = (db.collection("telegram_sessions").document(
        save_dispatch["context_id"],
    ).collection("operations").document(fact_operation_id).get().to_dict())
    assert interrupted_operation["state"] == "running"
    interrupted_context = db.collection("telegram_sessions").document(
        save_dispatch["context_id"],
    ).get().to_dict()
    assert interrupted_context["reserved_operation_id"] == fact_operation_id
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
               ) == upserts_before_fact + 1
    fact_time[0] = interrupted_context["lock_expires_at"] + timedelta(seconds=1)
    fact_result = asyncio.run(post("/internal/dispatch", fact_dispatch),
                              loop_factory=_selector_loop)
    assert fact_result.status_code == 200, fact_result.text
    assert fact_result.json()["status"] == "processed"
    replayed_context = db.collection("telegram_sessions").document(
        save_dispatch["context_id"],
    ).get().to_dict()
    assert replayed_context["fence"] > interrupted_context["fence"]
    fact_row = (db.collection("telegram_sessions").document(save_dispatch["context_id"])
                .collection("drafts").document(draft_id).collection("annotations")
                .document(annotation[0]["comment_id"]).get().to_dict())
    assert fact_row["state"] == "applied", fact_result.json()
    assert fact_row["applied_doc_id"] == "doc-default-save"
    saved_conversation_id = draft_ref.get().to_dict()["conversation_id"]
    visible_annotations = TelegramRepository(db).list_conversation_annotations(
        save_dispatch["context_id"], saved_conversation_id,
    )
    assert len(visible_annotations) == 1
    assert visible_annotations[0]["text"] == annotation_text
    assert visible_annotations[0]["state"] == "applied"
    stable_fact_operation_id = "telegram-tool-" + hashlib.sha256(
        f"{fact_row['application_operation_id']}\0{0}".encode("utf-8"),
    ).hexdigest()
    assert db.collection("dialog_operations").document(
        stable_fact_operation_id,
    ).get().to_dict()["state"] == "completed"
    fact_upserts = [event for event in sdk_fakes.trace
                    if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
    assert len(fact_upserts) == upserts_before_fact + 1
    point = fact_upserts[-1]["arguments"]["points"][0]
    assert point.id == stable_point_id
    assert point.payload == fact_payload
    assert point.payload["text"] == annotation_text
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "embed_content"]) == 1
    fact_replay = asyncio.run(post("/internal/dispatch", fact_dispatch),
                              loop_factory=_selector_loop)
    assert fact_replay.status_code == 200, fact_replay.text
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "upsert"]) == (
                    upserts_before_fact + 1
                )
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]) == 1

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


@pytest.mark.parametrize("selected_kind", ["search", "document"])
@pytest.mark.parametrize("selector", ["correct_number", "none", "unknown_number"])
def test_default_app_later_annotation_targets_accepted_draft_independent_of_focus(
    firestore_fake, sdk_fakes, monkeypatch, selected_kind, selector,
):
    """A correction targets its accepted collecting draft, never old selected history."""
    from services.connectors.telegram.repository import TelegramRepository
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    repository = TelegramRepository(runtime.db)
    context_id = "77_default"
    doc_id, old_conversation = "synthetic-old-doc", "synthetic-old-conversation"
    runtime.db.collection("transcript_index").document(doc_id).set({
        "doc_id": doc_id, "source_file": "Unrelated saved meeting", "status": "synced",
        "provider": "telegram", "parent_id": "folder-client",
    })
    old_number = repository.bind_document(
        context_id, doc_id, old_conversation,
        metadata={"title": "Unrelated saved meeting"}, now=runtime.clock(),
    )
    repository.ensure_conversation(
        context_id, old_conversation, kind="document", doc_id=doc_id,
        dialog_number=old_number, now=runtime.clock(),
    )
    common = {"chat": {"id": 77, "type": "private"}, "from": {"id": 7}}
    for i in range(8):
        response = asyncio.run(runtime.webhook({"update_id": 11000 + i, "message": {
            **common, "message_id": 12000 + i, "date": 1793320000,
            "text": f"Fictional forwarded utterance {i + 1}.",
            "forward_origin": {"type": "user", "date": 1793310000 + i,
                               "sender_user": {"id": 44, "first_name": "Ada"}},
        }}))
        assert response.status_code == 200, response.text
    draft_id = repository.get_context(context_id)["active_draft_id"]
    draft_before = repository.get_draft(context_id, draft_id)
    number = draft_before["dialog_number"]
    assert number != old_number
    ack = next(item["body"] for item in runtime.task_requests
               if item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack)).status_code == 200

    selected_id = old_conversation
    if selected_kind == "search":
        selected_id = "synthetic-selected-search"
        repository.ensure_conversation(
            context_id, selected_id, kind="search", search_filters={"query": "unrelated"},
            now=runtime.clock(),
        )
    context_ref = runtime.db.collection("telegram_sessions").document(context_id)
    context_ref.update({"active_conversation_id": selected_id,
                        "selected_dialog_number": old_number if selected_kind == "document" else None,
                        "last_doc_id": doc_id})
    turns_before = repository.list_turns(context_id, selected_id)
    note_text = "This newly forwarded conversation concerns a fictional research integration."
    requested_number = (number if selector == "correct_number" else
                        number + 30 if selector == "unknown_number" else None)
    runtime.route_response("annotation", dialog_number=requested_number, substantive_text=note_text)
    response = asyncio.run(runtime.webhook({"update_id": 11020, "message": {
        **common, "message_id": 12020, "date": 1793320600,
        "text": "I mean the forwarded conversation, not those search results.",
    }}))
    assert response.status_code == 200, response.text
    dispatch = next(item["body"] for item in reversed(runtime.task_requests)
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and item["body"].get("operation_id") and not item["body"].get("notification_id"))
    response = asyncio.run(runtime.dispatch(dispatch))
    assert response.status_code == 200, response.text
    routed = context_ref.collection("operations").document(dispatch["operation_id"]).get().to_dict()
    assert routed.get("route_control", {}).get("kind") == "annotation", routed
    notes = repository.list_draft_annotations(context_id, draft_id)
    if selector == "unknown_number":
        assert notes == []
        assert routed["state"] == "failed"
        assert repository.list_conversation_annotations(context_id, old_conversation) == []
        assert not [event for event in sdk_fakes.trace
                    if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
        return
    assert len(notes) == 1, {"state": routed.get("state"), "route": routed.get("route_control")}
    assert notes[0]["text"] == note_text and notes[0]["state"] == "pending_save"
    assert repository.list_conversation_annotations(context_id, old_conversation) == []
    assert repository.list_turns(context_id, selected_id) == turns_before
    assert repository.get_draft(context_id, draft_id)["ordered_message_ids"] == draft_before["ordered_message_ids"]
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "upsert"]

    model_calls = len([event for event in sdk_fakes.trace
                       if event["boundary"] == "gemini" and event["operation"] == "generate_content"])
    assert asyncio.run(runtime.dispatch(dispatch)).status_code == 200
    assert len(repository.list_draft_annotations(context_id, draft_id)) == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == model_calls


def test_default_app_forwarding_topic_comment_refreshes_after_sources_arrive_during_routing(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Reclassify once against newly verified source metadata before any search effect."""
    import json

    from services.connectors.telegram.repository import TelegramRepository
    from tests.fixtures import sdk_fakes as sdk_boundary
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    repository = TelegramRepository(runtime.db)
    common = {"chat": {"id": 77, "type": "private"}, "from": {"id": 7}}
    topic_comment = "Atlas Notes and Harbor rollout"
    runtime.route_response(
        "search", substantive_text="Find Atlas Notes conversations.",
        tool_choice={"name": "query", "arguments": {"query": "Atlas Notes"}},
    )
    runtime.route_response("annotation", substantive_text=topic_comment)
    prompts = []
    original = sdk_boundary._GeminiModels.generate_content

    def add_sources_during_route(sdk_models, **kwargs):
        contents = kwargs.get("contents")
        content = contents[0]
        parts = content.get("parts") if isinstance(content, dict) else content.parts
        first_part = parts[0]
        text = first_part.get("text") if isinstance(first_part, dict) else first_part.text
        try:
            prompt = json.loads(text)
        except (TypeError, ValueError):
            return original(sdk_models, **kwargs)
        if "response_schema" in prompt and "command" in prompt:
            prompts.append(prompt)
            if len(prompts) == 1:
                for index in range(8):
                    response = asyncio.run(runtime.webhook({"update_id": 13001 + index, "message": {
                        **common, "message_id": 14001 + index, "date": 1793320000,
                        "text": f"Synthetic source {index + 1}.",
                        "forward_origin": {"type": "user", "date": 1793310000 + index,
                                           "sender_user": {"id": 44, "first_name": "Ada"}},
                    }}))
                    assert response.status_code == 200, response.text
        return original(sdk_models, **kwargs)

    monkeypatch.setattr(sdk_boundary._GeminiModels, "generate_content", add_sources_during_route)
    response = asyncio.run(runtime.webhook({"update_id": 13000, "message": {
        **common, "message_id": 14000, "date": 1793320000, "text": topic_comment,
    }}))
    assert response.status_code == 200, response.text
    dispatch = next(item["body"] for item in reversed(runtime.task_requests)
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and item["body"].get("operation_id") and not item["body"].get("notification_id"))
    response = asyncio.run(runtime.dispatch(dispatch))
    assert response.status_code == 200, response.text
    assert len(prompts) == 2
    assert prompts[0]["context"]["forwarding_source_bound"] is False
    assert prompts[1]["context"]["forwarding_source_bound"] is True
    context = repository.get_context("77_default")
    draft_id = context["active_draft_id"]
    notes = repository.list_draft_annotations("77_default", draft_id)
    assert len(notes) == 1 and notes[0]["text"] == topic_comment
    assert notes[0]["state"] == "pending_save"
    assert repository.get_draft("77_default", draft_id)["ordered_message_ids"] == list(range(14001, 14009))
    assert not [event for event in sdk_fakes.trace if event["boundary"] == "qdrant"
                and event["operation"] in {"query_points", "query_batch_points", "upsert", "scroll"}]
    assert not [event for event in sdk_fakes.trace if event["boundary"] == "gemini"
                and event["operation"] == "embed_content"]

    receipt_ack = next(item["body"] for item in runtime.task_requests
                       if item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(receipt_ack)).status_code == 200
    model_calls = len([event for event in sdk_fakes.trace
                       if event["boundary"] == "gemini" and event["operation"] == "generate_content"])
    assert asyncio.run(runtime.dispatch(dispatch)).status_code == 200
    assert len(prompts) == 2
    assert len(repository.list_draft_annotations("77_default", draft_id)) == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == model_calls


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
    other_number = repository.bind_document(
        context_id, "doc-other", "conversation-other",
        metadata={"title": "Other notes", "folder_path": []},
        now=now, update_last_doc=False,
    )
    repository.ensure_conversation(
        context_id, "conversation-other", kind="document", doc_id="doc-other",
        dialog_number=other_number, now=now,
    )
    repository.append_turn(
        context_id, "conversation-other", turn_id="other-question",
        operation_id="other-operation", role="user",
        parts=[{"text": "An unrelated prior question."}], now=now,
    )
    repository.append_turn(
        context_id, "conversation-other", turn_id="other-answer",
        operation_id="other-operation", role="model",
        parts=[{"text": "An unrelated prior answer."}], now=now,
    )
    db.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-other",
        "selected_dialog_number": other_number,
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
            assert "An unrelated prior question." in serialized_contents
            assert "An unrelated prior answer." in serialized_contents
            assert "They moved the launch to May." not in serialized_contents
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
        assert "unrelated prior" not in serialized_contents
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
    assert [payload["rich_message"]["markdown"] for method, payload in sent
            if method == "sendRichMessage"] == [
        "The milestone date is June 12.",
    ], sent
    assert operation["route_control"]["target_doc_id"] == "doc-selected"
    assert operation["route_control"]["selection_policy"] == "silent_v1"
    assert len(sent) == 1
    assert not any("Starting to process dialog" in str(payload.get("text", ""))
                   for method, payload in sent)
    assert operation["route_control"]["tool_choice"]["name"] == "list_recent_transcripts"
    turns = repository.list_turns(context_id, "conversation-selected")
    assert [turn["role"] for turn in turns] == ["user", "model", "user", "model"]
    assert turns[-2]["operation_id"] == dispatch["operation_id"]
    assert turns[-2]["parts"][0]["text"] == route["substantive_text"]
    assert turns[-1]["parts"][0]["text"] == "The milestone date is June 12."
    assert all("application-tool-result-data" not in str(turn) for turn in turns)
    selected_context = repository.get_context(context_id)
    assert selected_context["active_conversation_id"] == "conversation-selected"
    assert selected_context["selected_dialog_number"] == number
    assert repository.list_turns(context_id, "conversation-other")[-2:]

    duplicate = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(wire_requests) == 2
    assert len(tool_calls) == 1
    assert len([payload for method, payload in sent
                if method == "sendRichMessage"
                and payload.get("rich_message", {}).get("markdown") == "The milestone date is June 12."]) == 1
    gemini_http.close()


def test_default_asgi_contextual_search_uses_model_args_and_hydrates_full_tool_history(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Exercise two contextual searches through default authenticated ASGI and real SDK wire."""
    import httpx
    from google import genai
    from google.genai import types
    from tests.fixtures import sdk_fakes as sdk_fake_module
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository
    from core.operations import registry

    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("core.integrations.firestore.transcripts.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-search-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.google.com/drive/folders/pytest-search-root")

    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 800 + len(sent)}})

    original_telegram_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_telegram_init(
            self, token, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(bot_response),
            ), timeout=timeout,
        )

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    repository = TelegramRepository(client=db)
    now = datetime.now(timezone.utc)
    from dataclasses import asdict
    from services.connectors.telegram.models import TelegramContext
    db.collection("telegram_sessions").document("77_default").set(asdict(
        TelegramContext(chat_id=77, active_conversation_id="telegram-chat-history",
                        created_at=now, updated_at=now),
    ))
    root_id = "pytest-search-root"
    db.collection("folders").document(root_id).set({
        "folder_id": root_id, "name": "Root", "parent_id": None,
    })
    db.collection("folders").document("search-client-folder").set({
        "folder_id": "search-client-folder", "name": "Client", "parent_id": root_id,
    })
    db.collection("transcript_index").document("doc-launch").set({
        "source_file": "May launch planning", "parent_id": "search-client-folder",
        "dialog_date": "2026-05-15", "provider": "telegram", "status": "synced",
    })
    # Seed substantive selected chat history so the first router has contextual input.
    repository.ensure_conversation("77_default", "telegram-chat-history", kind="chat", now=now)
    repository.append_turn(
        "77_default", "telegram-chat-history", turn_id="search-prior-user",
        operation_id="prior-chat", role="user", parts=[{"text": "We discussed a May launch."}], now=now,
    )
    repository.append_turn(
        "77_default", "telegram-chat-history", turn_id="search-prior-model",
        operation_id="prior-chat", role="model", parts=[{"text": "Pavel K joined that discussion."}], now=now,
    )

    long_source = "Pavel K :: The launch date remains May 15. \U0001f33f " * 24000
    assert len(long_source.encode("utf-8")) > 1024 * 1024
    query_results = [
        ([{"doc_id": "doc-launch", "folder_path": ["Client"],
           "dialog_date": "2026-05-15", "segments": [{"range": [4, 5], "dialog": [
               {"speaker": "Pavel K", "text": long_source, "order_index": 4},
               {"speaker": "Maya", "text": "Pavel K was also mentioned in a later note.", "order_index": 5},
           ]}]}], {"total_matches": 2, "returned_segments": 2, "truncated": False}),
        ([{"doc_id": "doc-launch", "folder_path": ["Client"],
           "dialog_date": "2026-05-15", "segments": [{"range": [4, 5], "dialog": [
               {"speaker": "Pavel K", "text": long_source, "order_index": 4},
               {"speaker": "Maya", "text": "Pavel K was also mentioned in a later note.", "order_index": 5},
           ]}]}], {"total_matches": 2, "returned_segments": 2, "truncated": False}),
    ]
    tool_calls = []
    original_invoke = registry.invoke_operation

    def observe_query(name, arguments, *, context=None, retry_policy=None):
        tool_calls.append((name, dict(arguments), retry_policy))
        if name == "query":
            return query_results[len([item for item in tool_calls if item[0] == "query"]) - 1]
        if name == "get_transcript":
            return query_results[0]
        return original_invoke(name, arguments, context=context, retry_policy=retry_policy)

    monkeypatch.setattr(registry, "invoke_operation", observe_query)

    first_route = {
        "kind": "search", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [],
        "substantive_text": "Find the May launch discussion and identify Pavel K's contribution.",
        "tool_choice": {"name": "query", "arguments": {
            "query": "May launch discussion participants and decisions",
            "folder_path": ["Client"], "date_from": "2026-01-01", "date_to": "2026-12-31",
        }},
        "certainty": "certain",
    }
    second_route = {
        "kind": "substantive", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [],
        "substantive_text": "Find Pavel K's contribution in the same May launch discussions.",
        "tool_choice": {"name": "query", "arguments": {
            "query": "Pavel K contribution to May launch",
            "folder_path": ["Client"], "date_from": "2026-01-01", "date_to": "2026-12-31",
        }},
        "certainty": "certain",
    }
    routes = [first_route, second_route]
    wire_requests = []
    route_count = 0
    tool_loop_count = 0

    def gemini_response(request):
        nonlocal route_count, tool_loop_count
        wire_requests.append(request)
        payload = json.loads(request.content)
        declarations = payload.get("tools", [{}])[0].get("functionDeclarations", [])
        if any(item.get("name") == "route_command" for item in declarations):
            route_index = route_count
            route_count += 1
            if route_index == 0:
                route_prompt = json.loads(payload["contents"][0]["parts"][0]["text"])
                assert "We discussed a May launch." in json.dumps(route_prompt["conversation_history"])
            else:
                route_prompt = json.loads(payload["contents"][0]["parts"][0]["text"])
                history_text = json.dumps(route_prompt["conversation_history"], ensure_ascii=False)
                assert long_source in history_text
                assert "search_exchange_ref" not in history_text
            route = routes[route_index]
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "route_command", "args": route},
                    "thoughtSignature": "c3ludGhldGljLXJvdXRlLXNpZw==",
                }],
            }}]})
        tool_loop_count += 1
        serialized_contents = json.dumps(payload["contents"], ensure_ascii=False)
        if tool_loop_count == 1:
            assert long_source in serialized_contents
            assert {item["name"] for item in declarations} == {
                "query", "get_transcript", "list_recent_transcripts", "list_folders",
            }
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"id": "read-call-1", "name": "get_transcript",
                                     "args": {"doc_id": "doc-launch",
                                              "folder_path": ["Client"],
                                              "date_from": "2026-01-01",
                                              "date_to": "2026-12-31"}},
                    "thoughtSignature": "c3ludGhldGljLXJlYWQtY2FsbA==",
                }],
            }}]})
        if tool_loop_count == 2:
            assert long_source in serialized_contents
            assert "functionResponse" in serialized_contents
            answer = "Pavel K was a participant and contributed the May 15 date."
        elif tool_loop_count == 3:
            answer = "Pavel K again appears as a participant in the same source evidence."
        else:
            raise AssertionError("unexpected model generation")
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": answer}],
        }}]})

    monkeypatch.setattr(genai, "Client", sdk_fake_module._REAL_GENAI_CLIENT)
    gemini_http = httpx.Client(transport=httpx.MockTransport(gemini_response))
    sdk_client = genai.Client(
        api_key="synthetic-gemini-key",
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1/v1beta", httpx_client=gemini_http,
        ),
    )
    monkeypatch.setattr("core.integrations.gemini.agent.gemini_client.make_client",
                        lambda **_kwargs: sdk_client)
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

    def deliver_command(update_id, message_id, text):
        prior_task_count = len(task_requests)
        intake = asyncio.run(post("/telegram/webhook", {"update_id": update_id, "message": {
            "message_id": message_id, "date": 1793320000,
            "chat": {"id": 77, "type": "private"},
            "from": {"id": 7, "first_name": "Tester"}, "text": text,
        }}, secret=True), loop_factory=_selector_loop)
        assert intake.status_code == 200, intake.text
        matching = [item["body"] for item in task_requests[prior_task_count:]
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and not item["body"].get("notification_id")]
        dispatch = matching[-1]
        response = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
        assert response.status_code == 200, response.text
        return dispatch

    first_dispatch = deliver_command(25001, 181, "Who participated in the May launch discussion?")
    context_id = "77_default"
    context_row = repository.get_context(context_id)
    search_session_id = context_row["active_conversation_id"]
    assert repository.get_conversation(context_id, search_session_id).kind.value == "search"
    first_operation = (db.collection("telegram_sessions").document(context_id)
                       .collection("operations").document(first_dispatch["operation_id"])
                       .get().to_dict())
    first_control = first_operation["route_control"]
    assert first_control["tool_choice"]["arguments"] == first_route["tool_choice"]["arguments"]
    assert first_control["search_protocol"] == "agent-v1"
    assert first_control["search_session_id"] == search_session_id

    second_dispatch = deliver_command(25002, 182, "And Pavel K?")
    assert second_dispatch["operation_id"] != first_dispatch["operation_id"]
    assert len(wire_requests) == 5, (tool_calls, sent)
    assert [call[0:2] for call in tool_calls] == [
        ("query", first_route["tool_choice"]["arguments"]),
        ("get_transcript", {"doc_id": "doc-launch", "folder_path": ["Client"],
                            "date_from": "2026-01-01", "date_to": "2026-12-31"}),
        ("query", second_route["tool_choice"]["arguments"]),
    ]
    assert all(call[2] is not None for call in tool_calls)
    assert all(len(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8"))
               < 1024 * 1024 for row in firestore_fake.documents.values())
    assert [payload["rich_message"]["markdown"] for method, payload in sent
            if method == "sendRichMessage"] == [
        "Pavel K was a participant and contributed the May 15 date.",
        "Pavel K again appears as a participant in the same source evidence.",
    ]
    turns = repository.list_turns(context_id, search_session_id)
    assert [turn["role"] for turn in turns if turn["operation_id"] == first_dispatch["operation_id"]] == [
        "user", "tool", "tool", "tool", "model",
    ]
    first_tool = next(turn for turn in turns if turn["operation_id"] == first_dispatch["operation_id"]
                      and turn.get("role") == "tool"
                      and isinstance(turn.get("tool_correlation"), dict)
                      and turn["tool_correlation"].get("name") == "query")
    reference = first_tool["parts"][0]["reference"]
    assert reference["payload"].get("_input_ref")
    assert len(json.dumps(first_tool, ensure_ascii=False, default=str)) < 10000

    before_duplicate = (len(wire_requests), len(tool_calls), len(sent))
    duplicate = asyncio.run(post("/internal/dispatch", first_dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert (len(wire_requests), len(tool_calls), len(sent)) == before_duplicate
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
    delivered = [payload["rich_message"]["markdown"] for method, payload in sent
                 if method == "sendRichMessage"]
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
                if method == "sendRichMessage"
                and payload.get("rich_message", {}).get("markdown") == delivered[0]]) == 1
    gemini_http.close()


@pytest.mark.parametrize(("include_result", "folder_read"), [
    (True, False), (False, False), (True, True),
])
def test_default_asgi_selected_dialog_search_creates_isolated_session_over_real_gemini_http(
    firestore_fake, sdk_fakes, monkeypatch, include_result, folder_read,
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

    tool_execution = []
    from services.connectors.telegram.runtime import tools as runtime_tools
    original_invoke_tool = runtime_tools.invoke_tool

    def observe_tool(name, arguments, **kwargs):
        tool_execution.append((name, dict(arguments), dict(kwargs.get("context", {}))))
        return original_invoke_tool(name, arguments, **kwargs)

    monkeypatch.setattr(runtime_tools, "invoke_tool", observe_tool)

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
    delivery_attempts = []
    block_followup_delivery = [True]

    def bot_response(request):
        payload = json.loads(request.content)
        method = request.url.path.rsplit("/", 1)[-1]
        delivery_attempts.append((method, payload))
        if (method == "sendRichMessage" and block_followup_delivery[0]
                and payload.get("rich_message", {}).get("markdown", "").startswith(
                    "Pavel K's confirmed transcript")):
            return httpx.Response(503, json={
                "ok": False, "error_code": 503, "description": "synthetic lost ack",
            })
        sent.append((method, payload))
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
    sdk_fakes.qdrant.queue("scroll", response=([SimpleNamespace(
        id="synthetic-transcript-read",
        payload={
            "doc_id": "doc-selected", "type": "utterance", "order_index": 0,
            "dialog_date": "2026-10-22", "speaker": "Team",
            "text": "The milestone date is June 12.",
        },
    )] if include_result else [], None))
    sdk_fakes.qdrant.queue("scroll", response=([SimpleNamespace(
        id="synthetic-follow-up-read",
        payload={
            "doc_id": "doc-selected", "type": "utterance", "order_index": 0,
            "dialog_date": "2026-10-22", "speaker": "Team",
            "text": "The milestone date is June 12.",
        },
    )] if include_result else [], None))

    search_route = {
        "kind": "search", "target_ref": None,
        "target_operation_id": None, "dialog_number": None, "result_refs": [],
        "substantive_text": "Find the synthetic schedule notes",
        "tool_choice": {"name": "query", "arguments": {
            "query": "Find the synthetic schedule notes",
            "folder_path": ["Client"], "date_from": "2026-01-01",
            "date_to": "2026-12-31",
        }},
        "certainty": "certain",
    }
    search_text = "Find the synthetic schedule notes"
    continuation_route = {
        "kind": "substantive", "target_ref": None,
        "target_operation_id": None, "dialog_number": None, "result_refs": [],
        "substantive_text": "Read the confirmed synthetic schedule transcript.",
        "tool_choice": {"name": "get_transcript", "arguments": {
            "doc_id": "doc-selected",
        }},
        "certainty": "certain",
    }
    routes = [search_route, continuation_route]
    wire_requests = []
    generation_requests = 0
    route_count = 0
    native_read_count = 0

    def gemini_response(request):
        nonlocal generation_requests, route_count, native_read_count
        wire_requests.append(request)
        if request.url.path.endswith(":batchEmbedContents"):
            return httpx.Response(200, json={"embeddings": [{"values": [0.25] * 768}]})
        generation_requests += 1
        payload = json.loads(request.content)
        assert all(content.get("role") in {"user", "model"}
                   for content in payload["contents"])
        serialized_contents = json.dumps(payload["contents"], ensure_ascii=False)
        assert search_text in serialized_contents
        route_call = (payload.get("tools", [{}])[0].get("functionDeclarations", [{}])[0]
                      .get("name") == "route_command")
        if route_call:
            route_count += 1
            if route_count == 1:
                assert "What did the team decide about the schedule?" in serialized_contents
                assert "They moved the launch to May." in serialized_contents
            else:
                assert "The milestone date is June 12." in serialized_contents
            assert payload["generationConfig"]["thinkingConfig"]["thinking_level"] == "MINIMAL"
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "route_command", "args": routes[route_count - 1]},
                    "thoughtSignature": "c3ludGhldGljLXJvdXRlLXNpZw==",
                }],
            }}]})
        if route_count == 1 and include_result and folder_read and native_read_count == 0:
            native_read_count += 1
            assert "application-tool-result-data" in serialized_contents
            assert {item["name"] for item in payload["tools"][0]["functionDeclarations"]} == {
                "query", "get_transcript", "list_recent_transcripts", "list_folders",
            }
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"id": "folder-list-call", "name": "list_folders",
                                     "args": {}},
                    "thoughtSignature": "c3ludGhldGljLWZvbGRlci1saXN0",
                }],
            }}]})
        if route_count == 1 and include_result and folder_read and native_read_count == 1:
            native_read_count += 1
            assert "functionResponse" in serialized_contents
            assert "Client" in serialized_contents
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"id": "folder-transcript-call", "name": "get_transcript",
                                     "args": {"folder_path": ["Client"],
                                              "date_from": "2026-01-01",
                                              "date_to": "2026-12-31", "limit": 5}},
                    "thoughtSignature": "c3ludGhldGljLWZvbGRlci1yZWFk",
                }],
            }}]})
        if route_count == 1 and include_result and native_read_count == 0:
            native_read_count += 1
            assert "application-tool-result-data" in serialized_contents
            assert {item["name"] for item in payload["tools"][0]["functionDeclarations"]} == {
                "query", "get_transcript", "list_recent_transcripts", "list_folders",
            }
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"id": "verified-read-1", "name": "get_transcript",
                                     "args": {"doc_id": "doc-selected"}},
                    "thoughtSignature": "c3ludGhldGljLXJlYWQtc2ln",
                }],
            }}]})
        if route_count == 1 and include_result:
            assert "functionResponse" in serialized_contents
            assert "The milestone date is June 12." in serialized_contents
        elif route_count == 2:
            assert "application-tool-result-data" in serialized_contents
            assert "The milestone date is June 12." in serialized_contents
        answer = ("The schedule note says the milestone date is June 12."
                  if include_result and route_count == 1 else
                  "Pavel K's confirmed transcript evidence is the June 12 milestone."
                  if route_count == 2 else
                  "I found no matching schedule notes in the selected evidence.")
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": answer}],
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
    if folder_read:
        folder_read_call = next(call for call in tool_execution
                                if call[0] == "get_transcript")
        assert folder_read_call[1] == {
            "folder_path": ["Client"], "date_from": "2026-01-01",
            "date_to": "2026-12-31", "limit": 5,
        }
        assert folder_read_call[2]["target_doc_id"] is None
        assert folder_read_call[2]["last_doc_id"] is None
        assert folder_read_call[2]["allow_scoped_transcript"] is True

    second_dispatch = None
    if include_result:
        prior_task_count = len(task_requests)
        second_intake = asyncio.run(post("/telegram/webhook", {
            "update_id": 24002, "message": {
                "message_id": 162, "date": 1793320001,
                "chat": {"id": 77, "type": "private"},
                "from": {"id": 7, "first_name": "Tester"}, "text": "And Pavel K?",
            },
        }, secret=True), loop_factory=_selector_loop)
        assert second_intake.status_code == 200, second_intake.text
        second_dispatch = next(item["body"] for item in task_requests[prior_task_count:]
                               if item["body"].get("target_kind") == "telegram_dispatch"
                               and not item["body"].get("notification_id"))
        second_processed = asyncio.run(post("/internal/dispatch", second_dispatch),
                                        loop_factory=_selector_loop)
        assert second_processed.status_code == 200, second_processed.text
        assert second_processed.json() in ({"status": "processed"}, {"status": "noop"})
        second_operation = (db.collection("telegram_sessions").document(context_id)
                            .collection("operations").document(
                                second_dispatch["operation_id"]).get().to_dict())
        assert second_operation["state"] == "running", repr(delivery_attempts)
        retry_wire_count = len(wire_requests)
        retry_retrieval_count = len([event for event in sdk_fakes.trace
                                     if event["boundary"] == "qdrant"])
        retry_dispatch = next((item["body"] for item in reversed(task_requests)
                               if item["body"].get("target_kind") == "telegram_dispatch"
                               and item["body"].get("operation_id") == second_dispatch["operation_id"]
                               and item["body"].get("outbox_id") != second_dispatch.get("outbox_id")
                               and not item["body"].get("notification_id")), None)
        assert retry_dispatch is not None, task_requests
        block_followup_delivery[0] = False
        # A fresh ASGI app reconstructs the runtime around the persisted operation,
        # checkpointed answer, source refs, and pending delivery chunk.
        app = create_app(
            webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
            oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
        )
        time.sleep(6.2)
        recovered = asyncio.run(post("/internal/dispatch", retry_dispatch),
                                loop_factory=_selector_loop)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json() in ({"status": "processed"}, {"status": "noop"})
        second_operation = (db.collection("telegram_sessions").document(context_id)
                            .collection("operations").document(
                                second_dispatch["operation_id"]).get().to_dict())
        assert second_operation["state"] == "completed", f"response={recovered.json()} op={second_operation} tasks={task_requests} attempts={delivery_attempts}"
        assert len(wire_requests) == retry_wire_count
        assert len([event for event in sdk_fakes.trace
                    if event["boundary"] == "qdrant"]) == retry_retrieval_count
        assert second_operation["route_control"]["search_protocol"] == "agent-v1"
        assert second_operation["route_control"]["tool_choice"] == {
            "name": "get_transcript", "arguments": {"doc_id": "doc-selected"},
            "link_requested": False,
        }

    delivered = [
        payload.get("text", payload.get("rich_message", {}).get("markdown"))
        for method, payload in sent
        if method in {"sendMessage", "sendRichMessage"}
    ]
    assert len(delivered) == (2 if include_result else 1)
    if include_result:
        assert "milestone date is June 12" in delivered[0]
    else:
        assert "no matching schedule notes" in delivered[0]
    assert operation["route_control"]["kind"] == "search"
    assert operation["route_control"]["tool_choice"] == {
        "name": "query", "arguments": {
            "query": search_text, "folder_path": ["Client"],
            "date_from": "2026-01-01", "date_to": "2026-12-31",
        }, "link_requested": False,
    }
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
               for request in wire_requests) == (
                   6 if folder_read else 5 if include_result else 2
               )
    assert sum(request.url.path.endswith(":batchEmbedContents")
               for request in wire_requests) == 2

    duplicate = asyncio.run(post("/internal/dispatch", dispatch), loop_factory=_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() in ({"status": "noop"}, {"status": "processed"})
    assert len(wire_requests) == (8 if folder_read else 7 if include_result else 4)
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant"
                and event["operation"] == "query_points"]) == retrieval_query_count
    assert len([payload for method, payload in sent
                if method in {"sendMessage", "sendRichMessage"}
                and payload.get("text", payload.get("rich_message", {}).get("markdown"))
                == delivered[0]]) == 1
    gemini_http.close()
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
