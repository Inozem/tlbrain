"""Offline checks for the concrete Telegram runtime adapters."""

from __future__ import annotations

import asyncio
from functools import partial
import inspect
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.offline


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


async def _polling_offload(function, *args, **kwargs):
    from services.connectors.telegram.runtime.runtime import _SYNC_POOL
    future = _SYNC_POOL.submit(partial(function, *args, **kwargs))
    while not future.done():
        await asyncio.sleep(0.01)
    result = future.result()
    return await result if inspect.isawaitable(result) else result


def test_runtime_requires_the_concrete_save_handler_and_is_lazy(firestore_fake, sdk_fakes):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    repository = TelegramRepository(client=firestore_fake.client_class())
    with pytest.raises(TypeError):
        TelegramCommandRuntime(repository=repository)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object())
    assert runtime.model is None
    assert runtime.client is None
    assert sdk_fakes.trace == []


def test_router_uses_structured_gemini_call_and_all_route_fields():
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter

    route = {
        "kind": "substantive", "target_ref": None,
        "target_operation_id": None, "dialog_number": None,
        "result_refs": [], "substantive_text": "Summarize the decision.",
        "tool_choice": None, "certainty": "certain",
    }

    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {"parts": [{"function_call": {
                "name": "route_command", "args": route,
            }}]}

    model = Model()
    router = GeminiCommandRouter(model)
    assert router.route(text="Summarize this", candidates=[]) == route
    assert model.calls[0]["mode"] == "routing"
    declaration = model.calls[0]["tools"][0]
    assert declaration["name"] == "route_command"
    assert set(declaration["parameters_json_schema"]["required"]) == {
        "kind", "target_ref", "target_operation_id", "dialog_number",
        "result_refs", "substantive_text", "tool_choice", "certainty",
    }


def test_router_prompt_carries_only_safe_document_and_focus_metadata():
    import json
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter

    class Model:
        def generate(self, **kwargs):
            self.request = kwargs
            return {"parts": [{"function_call": {"name": "route_command", "args": _route()}}]}

    model = Model()
    GeminiCommandRouter(model).route(
        text="continue this topic", candidates=[{
            "doc_id": "doc-1", "title": "Planning", "created_at": datetime.now(timezone.utc),
            "dialog_number": 4, "conversation_id": "conversation-4",
            "private_transcript": "must not be included",
        }], context_metadata={"selected_dialog_number": 4, "last_doc_id": "doc-1",
                               "selected_conversation_kind": "draft",
                               "selected_draft_id": "draft-4"},
        history=[{"role": "user", "parts": [{"text": "Earlier question"}]}],
    )
    prompt = json.loads(model.request["contents"][0]["parts"][0]["text"])
    assert prompt["documents"] == [{
        "doc_id": "doc-1", "title": "Planning", "created_at": prompt["documents"][0]["created_at"],
        "dialog_number": 4, "conversation_id": "conversation-4",
    }]
    assert prompt["context"] == {
        "selected_dialog_number": 4, "last_doc_id": "doc-1",
        "selected_conversation_kind": "draft", "selected_draft_id": "draft-4",
    }
    assert prompt["conversation_history"] == [
        {"role": "user", "parts": [{"text": "Earlier question"}]},
    ]
    assert "must not be included" not in repr(prompt)


def test_router_reads_real_gemini_candidate_envelope_and_governor_replays_it(firestore_fake):
    from google.genai import types
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter
    from services.connectors.telegram.runtime.runtime import _GovernedModel

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    calls = []
    route = _route()

    class SDKModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name="route_command", args=route))]),
            )])

    context_ref = client.collection("telegram_sessions").document(context_id)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(client, context_ref),
    )
    model = _GovernedModel(SDKModel(), governor, operation["operation_id"], lease,
                           lambda: now, lambda: True)
    router = GeminiCommandRouter(model)
    first = router.route(text="Summarize this", candidates=[])
    second = router.route(text="Summarize this", candidates=[])

    assert first == second == route
    assert len(calls) == 1
    stages = [value for path, value in firestore_fake.documents.items()
              if "/stages/" in path]
    assert len(stages) == 1 and stages[0]["state"] == "completed"


def test_governed_model_terminalizes_genai_402_without_replay_call(firestore_fake):
    from google.genai.errors import ClientError
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryBudgetExceeded, RetryGovernor
    from services.connectors.telegram.runtime.runtime import _GovernedModel
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    error = ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})

    class Model:
        calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            raise error

    source = Model()
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    model = _GovernedModel(source, governor, operation["operation_id"], lease,
                           lambda: now, lambda: True)
    with pytest.raises(ClientError):
        model.generate(contents=[{"role": "user", "parts": [{"text": "save"}]}],
                       mode="tool_loop")

    stage_key = model.failure_stage
    stage = governor.get_stage(operation["operation_id"], stage_key)
    assert source.calls == 1
    assert stage.failures == 1
    assert stage.last_error.category == "billing"
    assert stage.last_error.service == "gemini"
    assert stage.last_error.retryable is False
    with pytest.raises(RetryBudgetExceeded):
        model.generate(contents=[{"role": "user", "parts": [{"text": "save"}]}],
                       mode="tool_loop")
    assert source.calls == 1
    stored = client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).collection(
        "stages").document(stage_key).get().to_dict()
    assert "private-marker" not in repr(stored)


def test_governed_model_round_identity_advances_after_interrupted_round(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import _GovernedModel

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    class Model:
        def __init__(self):
            self.calls = []
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {"text": f"round-{len(self.calls)}", "parts": [{"text": f"round-{len(self.calls)}"}]}
    model = Model()
    governed = _GovernedModel(model, governor, operation["operation_id"], lease,
                              lambda: now, lambda: True)
    initial = {"role": "user", "parts": [{"text": "request"}]}
    first = governed.generate(contents=[initial], mode="tool_loop")
    replay = governed.generate(contents=[initial], mode="tool_loop")
    next_round = governed.generate(contents=[initial, {"role": "model", "parts": first["parts"]},
                                             {"role": "user", "parts": [{"text": "tool result"}]}],
                                   mode="tool_loop")

    assert first == replay
    assert next_round["text"] == "round-2"
    assert len(model.calls) == 2
    assert len([row for path, row in firestore_fake.documents.items() if "/stages/" in path]) == 2


def test_governed_model_fingerprint_includes_opaque_part_signature(firestore_fake):
    from google.genai import types
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.runtime.runtime import _GovernedModel

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    class Model:
        def __init__(self):
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return {"text": "same response", "parts": [{"text": "same response"}]}
    model = Model()
    governed = _GovernedModel(model, governor, operation["operation_id"], lease,
                              lambda: now, lambda: True)
    def request(signature):
        part = types.Part(function_call=types.FunctionCall(name="list_folders", args={}),
                          thought_signature=signature)
        return {"contents": [{"role": "model", "parts": [part]}], "mode": "tool_loop"}

    first_request, second_request = request(b"sig-a"), request(b"sig-b")
    first = governed.generate(**first_request)
    first_replay = governed.generate(**first_request)
    second = governed.generate(**second_request)

    assert first == first_replay == second
    assert model.calls == 2


def test_governor_spills_large_tool_results_and_replays_exact_value(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    ledger = FirestoreAgentLedger(client, context_id=context_id,
                                  operation_id=operation["operation_id"],
                                  owner=lease.owner, fence=lease.fence,
                                  clock=lambda: now)
    class LargeExecutor:
        def __init__(self):
            self.calls = 0
        def execute(self, **kwargs):
            self.calls += 1
            return {"transcript": "z" * (1024 * 1024 + 32)}
    inner = LargeExecutor()
    governed = _GovernedExecutor(
        inner, governor, operation["operation_id"], lease, lambda: now,
        lambda: True, payload_codec=ledger,
    )

    first = governed.execute(name="get_transcript", arguments={"doc_id": "doc-1"},
                             call_ordinal=2)
    second = governed.execute(name="get_transcript", arguments={"doc_id": "doc-1"},
                              call_ordinal=2)

    assert first == second == {"transcript": "z" * (1024 * 1024 + 32)}
    assert inner.calls == 1
    assert all(len(json.dumps(value, default=str).encode("utf-8")) < 1024 * 1024
               for path, value in firestore_fake.documents.items()
               if "/stages/" in path or path.endswith("/turn"))


def test_governor_reconciles_an_interrupted_tool_stage_without_duplicate_effect(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor, UnknownOutcome
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    stage_key = "telegram-tool-03"
    with pytest.raises(UnknownOutcome):
        governor.run_stage(operation["operation_id"], stage_key,
                           lambda: (_ for _ in ()).throw(UnknownOutcome("process stopped")),
                           now=now, owner=lease.owner, fence=lease.fence)
    class Executor:
        def __init__(self):
            self.calls = 0
        def execute(self, **kwargs):
            self.calls += 1
            return {"confirmed": True}
    inner = Executor()
    governed = _GovernedExecutor(inner, governor, operation["operation_id"],
                                 lease, lambda: now, lambda: True)

    result = governed.execute(name="list_folders", arguments={}, call_ordinal=3)

    assert result == {"confirmed": True}
    assert inner.calls == 1


def test_governed_tool_terminalizes_genai_402_without_second_sdk_call(firestore_fake):
    from google.genai.errors import ClientError
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryBudgetExceeded, RetryGovernor
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    error = ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})

    class Executor:
        calls = 0

        def execute(self, **_kwargs):
            self.calls += 1
            raise error

    inner = Executor()
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    executor = _GovernedExecutor(
        inner, governor, operation["operation_id"], lease, lambda: now, lambda: True,
    )
    with pytest.raises(ClientError):
        executor.execute(name="get_transcript", arguments={"doc_id": "doc"}, call_ordinal=0)

    stage = governor.get_stage(operation["operation_id"], "telegram-tool-00")
    assert inner.calls == 1
    assert stage.failures == 1
    assert stage.last_error.category == "billing"
    assert stage.last_error.service == "gemini"
    assert stage.last_error.retryable is False
    with pytest.raises(RetryBudgetExceeded):
        executor.execute(name="get_transcript", arguments={"doc_id": "doc"}, call_ordinal=0)
    assert inner.calls == 1


def test_governed_model_keeps_transient_genai_503_within_three_attempts(firestore_fake):
    from google.genai.errors import ClientError
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryBudgetExceeded, RetryGovernor
    from services.connectors.telegram.runtime.runtime import _GovernedModel
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    error = ClientError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})

    class Model:
        calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            raise error

    source = Model()
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    model = _GovernedModel(source, governor, operation["operation_id"], lease,
                           lambda: now, lambda: True)
    request = {"contents": [{"role": "user", "parts": [{"text": "save"}]}], "mode": "tool_loop"}
    for _ in range(3):
        with pytest.raises(ClientError):
            model.generate(**request)
    with pytest.raises(RetryBudgetExceeded):
        model.generate(**request)

    assert source.calls == 3
    stage = governor.get_stage(operation["operation_id"], model.failure_stage)
    assert stage.failures == 3
    assert stage.last_error.retryable is True


def test_interrupted_registry_write_reconciliation_never_reinvokes_mutation(
    firestore_fake, monkeypatch,
):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor, UnknownOutcome
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    mutations = []
    monkeypatch.setattr("services.connectors.telegram.runtime.tools.invoke_tool",
                        lambda *args, **kwargs: mutations.append((args, kwargs)) or {"ok": True})
    stage_key = "telegram-tool-00"
    with pytest.raises(UnknownOutcome):
        governor.run_stage(operation["operation_id"], stage_key,
                           lambda: (_ for _ in ()).throw(UnknownOutcome("stopped")),
                           now=now, owner=lease.owner, fence=lease.fence)
    scoped = {"context_id": context_id, "operation_id": operation["operation_id"],
              "owner": lease.owner, "fence": lease.fence, "lease": lease,
              "retry_governor": governor, "target_doc_id": None, "clock": lambda: now}
    executor = TelegramRegistryExecutor(client=client, context=scoped,
                                        save_handler=object(), execution_context=scoped)
    governed = _GovernedExecutor(executor, governor, operation["operation_id"],
                                 lease, lambda: now, lambda: True)

    with pytest.raises(Exception):
        governed.execute(name="create_folder", arguments={"name": "Research"},
                         call_ordinal=0)

    assert mutations == []
    assert governor.get_stage(operation["operation_id"], stage_key).failures == 1


def test_registry_executor_rejects_unregistered_capability_before_dispatch():
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

    executor = TelegramRegistryExecutor(
        client=object(), context={}, save_handler=object(),
        execution_context={"context_id": "42_default", "operation_id": "op",
                           "lease": object()},
    )
    with pytest.raises(ValueError):
        executor.execute(name="delete_all", arguments={}, call_ordinal=0)


def test_save_dialog_requires_and_calls_explicit_save_handler(firestore_fake):
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor
    from services.connectors.telegram.repository import TelegramRepository

    class SaveHandler:
        def __init__(self):
            self.calls = []

        def save_dialog(self, arguments, **kwargs):
            self.calls.append((arguments, kwargs))
            return {"accepted": True}

    handler = SaveHandler()
    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    executor = TelegramRegistryExecutor(
        client=client, context={}, save_handler=handler,
        execution_context={"context_id": context_id, "operation_id": operation["operation_id"],
                           "owner": lease.owner, "fence": lease.fence, "lease": lease},
    )
    result = executor.execute(
        name="add_dialog",
        arguments={"provider": "test", "utterances": [{"speaker": "Ada", "text": "Hello"}]},
        call_ordinal=3,
    )
    assert result == {"accepted": True}
    assert handler.calls[0][1]["call_ordinal"] == 3
    assert handler.calls[0][1]["lease"] is lease


def test_failed_write_releases_child_lease_for_immediate_retry(monkeypatch, firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    calls = []
    def invoke(name, arguments, **kwargs):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("offline provider")
        return {"folder_id": "folder-new", "name": arguments["name"]}
    monkeypatch.setattr("services.connectors.telegram.runtime.tools.invoke_tool", invoke)
    now = datetime.now(timezone.utc)
    executor = TelegramRegistryExecutor(
        client=client, context={}, save_handler=object(),
        execution_context={"context_id": context_id,
                           "operation_id": operation["operation_id"],
                           "owner": lease.owner, "fence": lease.fence, "lease": lease,
                           "clock": lambda: now},
    )

    with pytest.raises(RuntimeError):
        executor.execute(name="create_folder", arguments={"name": "Research"},
                         call_ordinal=1)
    result = executor.execute(name="create_folder", arguments={"name": "Research"},
                              call_ordinal=1)

    assert result == {"folder_id": "folder-new", "name": "Research"}
    assert calls == ["create_folder", "create_folder"]


def _runtime_queue_item(client, *, operation_id="op-runtime"):
    from services.connectors.telegram.commands import TelegramCommand

    now = datetime.now(timezone.utc)
    context_id = "42_default"
    client.collection("telegram_sessions").document(context_id).set({
        "chat_id": 42, "thread_id": None, "active_draft_id": None,
        "active_conversation_id": None, "selected_dialog_number": None,
        "selection_sequence": 0, "last_doc_id": None,
        "next_dialog_number": 1, "next_operation_sequence": 2,
        "head_sequence": 1, "active_operation_id": operation_id,
        "reserved_operation_id": operation_id, "lock_owner": "worker",
        "lock_expires_at": now + timedelta(minutes=2), "fence": 1,
        "created_at": now, "updated_at": now,
    })
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation_id).set({
            "operation_id": operation_id, "context_id": context_id,
            "state": "running", "sequence": 1,
            "input_ref": f"telegram_sessions/{context_id}/commands/9",
        })
    client.collection("telegram_sessions").document(context_id).collection(
        "commands").document("9").set({
            "context_id": context_id, "operation_id": operation_id,
            "kind": "freeform", "text": "private request",
        })
    operation = {"operation_id": operation_id, "context_id": context_id,
                 "sequence": 1, "dependency_operation_id": None}
    lease = SimpleNamespace(operation_id=operation_id, owner="worker", fence=1)
    return context_id, operation, lease, TelegramCommand("freeform", "private request")


class _Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, *, message_thread_id=None):
        self.sent.append((chat_id, text, message_thread_id))
        return {"message_id": len(self.sent)}

    async def edit_message_text(self, chat_id, message_id, text):
        self.sent.append((chat_id, text, None))
        return {"message_id": message_id}


class _RuntimeModel:
    def __init__(self, route, answer="A safe answer."):
        self.route_value, self.answer, self.calls = route, answer, []

    def generate(self, *, contents, mode, tools=None):
        self.calls.append((mode, contents, tools))
        if mode == "routing":
            return {"parts": [{"function_call": {
                "name": "route_command", "args": self.route_value,
            }}]}
        return {"text": self.answer, "parts": [{"text": self.answer}]}


def _route(*, tool_choice=None):
    return {"kind": "substantive", "target_ref": None,
            "target_operation_id": None, "dialog_number": None,
            "result_refs": [], "substantive_text": "answer this request",
            "tool_choice": tool_choice, "certainty": "certain"}


def test_no_document_command_runs_agent_and_delivers_durable_answer(firestore_fake):
    from services.connectors.telegram.commands import TelegramCommand
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    command = TelegramCommand("freeform", "switch to dialog #2 and what was decided?")
    route = _route()
    route["substantive_text"] = "what was decided?"
    bot, model = _Bot(), _RuntimeModel(route, "A useful answer.")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    assert TelegramRuntimeStore(repository, root_folder_id="pytest-synthetic-root",
                                context_id=context_id).existing_dialog_candidates() == ()
    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert [row[0] for row in model.calls] == ["routing", "tool_loop"], (bot.sent, firestore_fake.documents)
    assert len(bot.sent) == 1
    assert bot.sent[0][1] == "A useful answer."
    chat_id = "telegram-chat-" + __import__("hashlib").sha256(
        context_id.encode("utf-8")).hexdigest()
    assert repository.get_conversation(context_id, chat_id).kind.value == "chat"
    assert [row["parts"][0]["text"] for row in repository.list_turns(
        context_id, chat_id
    )] == ["what was decided?", "A useful answer."]
    assert [turn["role"] for turn in repository.list_turns(context_id, chat_id)] == [
        "user", "model",
    ]
    notifications = [row for path, row in firestore_fake.documents.items()
                     if path.startswith("notification_outbox/")]
    assert notifications and notifications[0]["body"] == "A useful answer."
    sent_count, call_count = len(bot.sent), len(model.calls)
    replay = asyncio.run(runtime(context_id, operation, lease, command=None),
                         loop_factory=_selector_loop)
    assert replay.value == "completed"
    assert len(bot.sent) == sent_count
    assert len(model.calls) == call_count


def test_pending_first_answer_stays_running_and_replays_without_model_call(
    firestore_fake, monkeypatch,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.notifications import TelegramNotificationPublisher

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    answer = "A useful answer." * 600
    bot, model = _Bot(), _RuntimeModel(_route(), answer)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)
    attempts = {}
    delivered = set()
    sends = {}
    async def pending_then_delivered(self, notification_id):
        attempts[notification_id] = attempts.get(notification_id, 0) + 1
        if len(attempts) == 2 and attempts[notification_id] == 1:
            return "retry"
        if notification_id not in delivered:
            delivered.add(notification_id)
            sends[notification_id] = sends.get(notification_id, 0) + 1
        return "delivered"
    monkeypatch.setattr(TelegramNotificationPublisher, "publish", pending_then_delivered)

    first = asyncio.run(runtime(context_id, operation, lease, command=command),
                        loop_factory=_selector_loop)
    call_count = len(model.calls)
    second = asyncio.run(runtime(context_id, operation, lease, command=None),
                         loop_factory=_selector_loop)

    assert first.value == "running"
    assert second.value == "completed"
    assert [mode for mode, _contents, _tools in model.calls] == ["routing", "tool_loop"]
    assert len(model.calls) == call_count
    assert len(delivered) == 3
    assert set(sends.values()) == {1}


def test_long_global_answer_is_delivered_as_complete_durable_chunks(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    answer = "x" * 8500
    bot, model = _Bot(), _RuntimeModel(_route(), answer)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert len(bot.sent) == 3
    assert all(len(row[1]) <= 4000 for row in bot.sent)
    assert "".join(row[1] for row in bot.sent) == answer


def test_provider_pending_chunk_blocks_later_chunks_and_status_check_reuses_final(
    firestore_fake, monkeypatch,
):
    import httpx
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.status_checks import TelegramStatusChecks
    from core.integrations.firestore.outbox import task_payload
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = [datetime.now(timezone.utc)]
    sends = []
    responses = [
        httpx.Response(200, json={"ok": True, "result": {"message_id": 101}}),
        httpx.Response(503, json={"ok": False}),
        httpx.Response(200, json={"ok": True, "result": {"message_id": 102}}),
        httpx.Response(200, json={"ok": True, "result": {"message_id": 103}}),
    ]
    def handler(request):
        import json as json_module
        payload = json_module.loads(request.content)
        sends.append(payload["text"])
        return responses.pop(0)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    telegram = TelegramClient("123456:synthetic-token", http_client=http)
    model = _RuntimeModel(_route(), "A" * 4000 + "B" * 4000 + "C" * 500)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=telegram, model=model,
                                     clock=lambda: now[0])

    first = asyncio.run(runtime(context_id, operation, lease, command=command),
                        loop_factory=_selector_loop)
    assert first.value == "running"
    assert len(sends) == 2
    assert len(sends[0]) == 4000 and len(sends[1]) == 4000
    now[0] += timedelta(seconds=3)
    second = asyncio.run(runtime(context_id, operation, lease, command=None),
                         loop_factory=_selector_loop)
    assert second.value == "completed"
    assert sends == ["A" * 4000, "B" * 4000, "B" * 4000, "C" * 500]
    assert len(model.calls) == 2

    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({"state": "completed"})
    notification_publisher = TelegramNotificationPublisher(
        repository, telegram, clock=lambda: now[0])
    checks = TelegramStatusChecks(
        client, notification_publisher=notification_publisher,
        clock=lambda: now[0], terminal_notification_updater=lambda _record: None,
    )
    status_record = checks.schedule_checks(context_id, operation["operation_id"], now[0])[0]
    extra_sends = len(sends)
    async def direct_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)
    original_to_thread = asyncio.to_thread
    asyncio.to_thread = direct_to_thread
    try:
        result = asyncio.run(checks.handle_task(task_payload(status_record)),
                             loop_factory=_selector_loop)
    finally:
        asyncio.to_thread = original_to_thread
    assert result == "no_op"
    assert len(sends) == extra_sends
    assert repository.get_notification(
        client.collection("telegram_sessions").document(context_id)
        .collection("operations").document(operation["operation_id"])
        .get().to_dict()["notification_id"]
    )["kind"] == "final"
    asyncio.run(telegram.aclose(), loop_factory=_selector_loop)


def test_no_document_list_folders_uses_validated_registry_executor(firestore_fake):
    import json
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "search-existing",
    })
    search = repository.ensure_conversation(
        context_id, "search-existing", kind="search",
        search_filters={"source": "current-topic"}, now=datetime.now(timezone.utc),
    )
    for index, role in enumerate(("user", "model", "user", "model"), 1):
        repository.append_turn(
            context_id, search.conversation_id, turn_id=f"search-history-{index}",
            operation_id=f"old-search-{index}", role=role,
            parts=[{"text": f"search-turn-{index}"}], now=datetime.now(timezone.utc),
        )
    repository.accept_forward(
        991, chat_id=42, thread_id=None,
        message={"message_id": 991, "text": "unrelated active draft secret"},
        now=datetime.now(timezone.utc),
    )
    model = _RuntimeModel(_route(tool_choice={"name": "list_folders", "arguments": {}}))
    bot = _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, root_folder_id="pytest-synthetic-root")
    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert [row[0] for row in model.calls] == ["routing"]
    assert len(bot.sent) == 1
    assert repository.get_conversation(context_id, "search-existing").kind.value == "search"
    route_prompt = json.loads(model.calls[0][1][0]["parts"][0]["text"])
    assert [turn["parts"][0]["text"] for turn in route_prompt["conversation_history"]] == [
        "search-turn-1", "search-turn-2", "search-turn-3", "search-turn-4",
    ]
    assert route_prompt["context"]["selected_search_filters"] == {"source": "current-topic"}
    assert "unrelated active draft secret" not in repr(route_prompt)
    assert not any("telegram-chat-" in path for path in firestore_fake.documents)


def test_selected_document_global_tool_keeps_document_identity_without_draft_source(
    firestore_fake,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    number = repository.bind_document(
        context_id, "doc-global-tool", "conversation-global-tool",
        metadata={"title": "Selected transcript"}, now=datetime.now(timezone.utc),
    )
    repository.ensure_conversation(
        context_id, "conversation-global-tool", kind="document",
        doc_id="doc-global-tool", dialog_number=number,
        now=datetime.now(timezone.utc),
    )
    client.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-global-tool",
        "selected_dialog_number": number,
    })
    repository.accept_forward(
        992, chat_id=42, thread_id=None,
        message={"message_id": 992, "text": "other collecting draft secret"},
        now=datetime.now(timezone.utc),
    )
    model = _RuntimeModel(_route(tool_choice={"name": "list_folders", "arguments": {}}))
    bot = _Bot()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot, model=model,
        root_folder_id="pytest-synthetic-root",
    )

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    conversation = repository.get_conversation(context_id, "conversation-global-tool")
    assert conversation.kind.value == "document"
    assert conversation.doc_id == "doc-global-tool"
    turns = repository.list_turns(context_id, "conversation-global-tool")
    assert [turn["role"] for turn in turns] == ["user", "model"]
    assert turns[0]["parts"][0]["text"] == "answer this request"
    assert "other collecting draft secret" not in repr(model.calls)


def test_unselected_stable_chat_loads_all_history_then_selects_after_substantive_route(firestore_fake):
    import hashlib
    import json
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    conversation_id = "telegram-chat-" + hashlib.sha256(context_id.encode("utf-8")).hexdigest()
    repository.ensure_conversation(
        context_id, conversation_id, kind="chat", now=datetime.now(timezone.utc),
    )
    for index, role in enumerate(("user", "model", "user", "model", "user", "model"), 1):
        repository.append_turn(
            context_id, conversation_id, turn_id=f"chat-history-{index}",
            operation_id=f"prior-chat-{index}", role=role,
            parts=[{"text": f"chat-turn-{index}"}], now=datetime.now(timezone.utc),
        )
    model, bot = _RuntimeModel(_route(), "Current reply."), _Bot()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot, model=model,
    )

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)
    assert result.value == "completed"
    prompt = json.loads(model.calls[0][1][0]["parts"][0]["text"])
    assert [turn["parts"][0]["text"] for turn in prompt["conversation_history"]] == [
        f"chat-turn-{index}" for index in range(1, 7)
    ]
    assert repository.get_context(context_id)["active_conversation_id"] == conversation_id
    assert [row["parts"][0]["text"] for row in repository.list_turns(
        context_id, conversation_id
    )][-2:] == ["answer this request", "Current reply."]
    assert any(path.startswith("notification_outbox/") for path in firestore_fake.documents)


def test_unselected_transcript_does_not_force_a_document_conversation(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("transcript_index").document("doc-unselected").set({
        "source_file": "Planning", "parent_id": "root-unselected",
    })
    client.collection("folders").document("root-unselected").set({
        "folder_id": "root-unselected", "name": "Root", "parent_id": "pytest-root",
    })
    bot, model = _Bot(), _RuntimeModel(_route(), "Hello. What would you like to do?")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, root_folder_id="pytest-root")

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert [mode for mode, _contents, _tools in model.calls] == ["routing", "tool_loop"]
    assert bot.sent[0][1] == "Hello. What would you like to do?"
    chat_id = "telegram-chat-" + __import__("hashlib").sha256(
        context_id.encode("utf-8")).hexdigest()
    assert repository.get_conversation(context_id, chat_id).kind.value == "chat"


def test_current_draft_question_and_followup_use_full_history_and_ephemeral_sources(firestore_fake):
    from services.connectors.telegram.commands import TelegramCommand
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.conversations import ConversationKind

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _ = _runtime_queue_item(client)
    operation["input_ref"] = f"telegram_sessions/{context_id}/commands/9"
    session_ref = client.collection("telegram_sessions").document(context_id)
    session_ref.update({"active_operation_id": None, "reserved_operation_id": None,
                        "lock_owner": None, "head_sequence": 0})
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({"state": "completed"})
    accepted = repository.accept_forward(
        6101, chat_id=42, thread_id=None,
        message={"message_id": 91, "text": "Project kickoff is on April 17.",
                 "display_speaker": "Nora", "source_chat_id": 815,
                 "source_message_id": 4501},
        now=datetime.now(timezone.utc),
    )
    accepted_audio = repository.accept_forward(
        6102, chat_id=42, thread_id=None,
        message={"message_id": 92, "audio_file_id": "pending-audio",
                 "transcription_state": "pending",
                 "transcription_text": "unconfirmed partial transcript",
                 "source_user_id": 47, "reliable_author_key": "telegram-user:47"},
        now=datetime.now(timezone.utc),
    )
    session_ref.update({"active_operation_id": operation["operation_id"],
                        "reserved_operation_id": operation["operation_id"],
                        "lock_owner": "worker", "head_sequence": 1,
                        "lock_expires_at": datetime.now(timezone.utc) + timedelta(minutes=2)})
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({"state": "running"})
    operation.update({"draft_id": accepted.draft_id, "dialog_number": accepted.dialog_number,
                      "cutoff_message_ids": [accepted.message_id, accepted_audio.message_id]})
    client.collection("telegram_sessions").document(context_id).collection(
        "commands").document("9").update({
            "draft_id": accepted.draft_id, "dialog_number": accepted.dialog_number,
            "cutoff_message_ids": [accepted.message_id, accepted_audio.message_id],
        })
    route_rows = []
    tool_inputs = []

    class ConversationModel:
        def generate(self, *, contents, mode, tools=None):
            if mode == "routing":
                prompt = json.loads(contents[0]["parts"][0]["text"])
                route_rows.append(prompt)
                return {"parts": [{"function_call": {
                    "name": "route_command", "args": {
                        "kind": "substantive", "target_ref": None,
                        "target_operation_id": None,
                        "dialog_number": accepted.dialog_number,
                        "result_refs": [], "substantive_text": prompt["command"],
                        "tool_choice": None, "certainty": "certain",
                    },
                }}]}
            tool_inputs.append(contents)
            response = "The kickoff is April 17." if len(tool_inputs) == 1 else "That date is April 17."
            return {"text": response, "parts": [{"text": response}]}

    bot = _Bot()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot, model=ConversationModel(),
    )
    first = asyncio.run(runtime(
        context_id, operation, lease, command=TelegramCommand("freeform", "When is the kickoff?"),
    ), loop_factory=_selector_loop)
    assert first.value == "completed", (first, accepted, firestore_fake.documents)
    assert route_rows
    conversation_id = f"{context_id}-{accepted.dialog_number}"
    conversation = repository.get_conversation(context_id, conversation_id)
    assert conversation is not None, (accepted, route_rows[0]["context"],
                                      [path for path in firestore_fake.documents if "/conversations/" in path])
    assert conversation.kind is ConversationKind.DRAFT
    rows = repository.list_turns(context_id, conversation_id)
    assert [(row["role"], row["parts"][0]["text"]) for row in rows] == [
        ("user", "When is the kickoff?"),
        ("model", "The kickoff is April 17."),
    ]
    assert route_rows[0]["context"]["selected_draft_id"] == accepted.draft_id
    assert "April 17" not in json.dumps(route_rows[0]["conversation_history"])
    assert "Project kickoff is on April 17." in repr(tool_inputs[0])
    assert "Nora" in repr(tool_inputs[0]) and "4501" in repr(tool_inputs[0])
    assert "untrusted_forwarded_source" in repr(tool_inputs[0])
    assert repr(tool_inputs[0]).index("untrusted_forwarded_source") < repr(
        tool_inputs[0]
    ).rindex("When is the kickoff?")
    assert '"dialog_number": 1' in repr(tool_inputs[0])
    assert "unconfirmed partial transcript" not in repr(tool_inputs[0])
    assert '"audio_pending": true' in repr(tool_inputs[0])
    assert "telegram-user:47" in repr(tool_inputs[0])
    assert not any(path.startswith("transcript_index/") for path in firestore_fake.documents)

    calls_before_replay = len(route_rows) + len(tool_inputs)
    replay = asyncio.run(runtime(context_id, operation, lease, command=None),
                         loop_factory=_selector_loop)
    assert replay.value == "completed"
    assert len(route_rows) + len(tool_inputs) == calls_before_replay
    assert len(repository.list_turns(context_id, conversation_id)) == 2

    second_id = "op-followup"
    now = datetime.now(timezone.utc)
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({"state": "completed"})
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(second_id).set({
            "operation_id": second_id, "context_id": context_id,
            "state": "running", "sequence": 2,
            "input_ref": f"telegram_sessions/{context_id}/commands/10",
        })
    client.collection("telegram_sessions").document(context_id).collection(
        "commands").document("10").set({
            "context_id": context_id, "operation_id": second_id,
            "kind": "freeform", "text": "And what day?",
            "draft_id": accepted.draft_id, "dialog_number": accepted.dialog_number,
            "cutoff_message_ids": [accepted.message_id],
        })
    client.collection("telegram_sessions").document(context_id).update({
        "active_operation_id": second_id, "reserved_operation_id": second_id,
        "head_sequence": 2, "next_operation_sequence": 3,
        "lock_owner": "worker", "lock_expires_at": now + timedelta(minutes=2),
        "fence": 2,
    })
    assert session_ref.get().to_dict().get("active_conversation_id") == conversation_id
    second_operation = {
        "operation_id": second_id, "context_id": context_id, "sequence": 2,
        "dependency_operation_id": None,
        "input_ref": f"telegram_sessions/{context_id}/commands/10",
    }
    second_lease = SimpleNamespace(operation_id=second_id, owner="worker", fence=2)
    second = asyncio.run(runtime(
        context_id, second_operation, second_lease,
        command=TelegramCommand("freeform", "And what day?"),
    ), loop_factory=_selector_loop)
    assert second.value == "completed"
    assert len(route_rows) == 2, (bot.sent, getattr(runtime, "debug_route_context_error", None))
    assert [turn["parts"][0]["text"] for turn in route_rows[1]["conversation_history"]] == [
        "When is the kickoff?", "The kickoff is April 17.",
    ]
    assert "Project kickoff is on April 17." in repr(tool_inputs[1])
    final_rows = repository.list_turns(context_id, conversation_id)
    assert [row["parts"][0]["text"] for row in final_rows] == [
        "When is the kickoff?", "The kickoff is April 17.",
        "And what day?", "That date is April 17.",
    ]
    assert len([path for path in firestore_fake.documents if path.startswith("transcript_index/")]) == 0


def test_stale_model_turn_append_cannot_select_or_write_and_valid_replay_is_idempotent(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, stale_lease, _ = _runtime_queue_item(client)
    conversation_id = "telegram-chat-stale-fence"
    now = datetime.now(timezone.utc)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.update({"lock_owner": "replacement-worker", "fence": 2,
                        "lock_expires_at": now + timedelta(minutes=2)})

    with pytest.raises(PermissionError, match="lease is stale"):
        repository.ensure_conversation(
            context_id, conversation_id, kind="chat", now=now,
            queue_operation_id=operation["operation_id"], queue_owner=stale_lease.owner,
            queue_fence=stale_lease.fence, select_if_unselected=True,
        )
    assert context_ref.get().to_dict().get("active_conversation_id") is None
    assert repository.get_conversation(context_id, conversation_id) is None

    live_lease = SimpleNamespace(
        operation_id=operation["operation_id"], owner="replacement-worker", fence=2,
    )
    conversation = repository.ensure_conversation(
        context_id, conversation_id, kind="chat", now=now,
        queue_operation_id=operation["operation_id"], queue_owner=live_lease.owner,
        queue_fence=live_lease.fence, select_if_unselected=True,
    )
    first = repository.append_turn(
        context_id, conversation_id, turn_id="model-after-takeover",
        operation_id=operation["operation_id"], role="model",
        parts=[{"text": "complete answer"}], now=now,
        queue_operation_id=operation["operation_id"], queue_owner=live_lease.owner,
        queue_fence=live_lease.fence, select_if_unselected=True,
    )
    replay = repository.append_turn(
        context_id, conversation_id, turn_id="model-after-takeover",
        operation_id=operation["operation_id"], role="model",
        parts=[{"text": "complete answer"}], now=now,
        queue_operation_id=operation["operation_id"], queue_owner=live_lease.owner,
        queue_fence=live_lease.fence, select_if_unselected=True,
    )
    assert conversation.kind.value == "chat"
    assert first == replay
    assert len(repository.list_turns(context_id, conversation_id)) == 1
    assert context_ref.get().to_dict()["active_conversation_id"] == conversation_id


def test_save_without_captured_draft_delivers_forward_messages_guidance(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    class SaveHandler:
        def __init__(self):
            self.calls = []
        async def save_draft(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("must not delegate an uncaptured save")
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    route = {"kind": "save", "target_ref": None, "target_operation_id": None,
             "dialog_number": None, "result_refs": [], "substantive_text": None,
             "tool_choice": None, "certainty": "certain"}
    bot, handler = _Bot(), SaveHandler()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=handler,
                                     client=bot, model=_RuntimeModel(route))

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert handler.calls == []
    assert bot.sent[0][1] == "Please forward messages first, then ask me to save them."


def test_stale_queue_fence_stops_before_routing_or_delivery(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("telegram_sessions").document(context_id).update({"lock_owner": "new-owner"})
    model, bot = _RuntimeModel(_route()), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)
    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert model.calls == []
    assert bot.sent == []


def test_model_routing_failure_delivers_safe_clarification(firestore_fake, caplog):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    class FailingModel:
        def generate(self, **kwargs):
            raise RuntimeError("provider secret detail")

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    bot = _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=FailingModel())
    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert bot.sent[0][1] == "I could not determine what you want me to do. Please clarify."
    assert "provider secret detail" not in caplog.text


def test_governed_provider_failure_retries_three_times_then_publishes_safe_error(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    class FailingModel:
        def __init__(self):
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            raise RuntimeError("secret provider details")
    model, bot = FailingModel(), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(context_id, operation, lease, command=command,
                                  retry_governor=governor), loop_factory=_selector_loop)
              for _ in range(3)]

    assert [state.value for state in states] == ["running", "running", "failed"]
    assert model.calls == 3
    body = bot.sent[-1][1]
    assert "Error in gemini/routing" in body and "category=execution" in body
    assert f"operation_id={operation['operation_id']}" in body
    assert "administrator" in body and "secret provider details" not in body


def test_global_tool_loop_provider_failure_uses_persisted_three_attempt_budget(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    class RoutedThenFailingModel:
        def __init__(self):
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            if kwargs.get("mode") == "routing":
                return {"parts": [{"function_call": {
                    "name": "route_command", "args": _route(),
                }}]}
            raise RuntimeError("private global provider failure")

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    model, bot = RoutedThenFailingModel(), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(context_id, operation, lease, command=command,
                                  retry_governor=governor), loop_factory=_selector_loop)
              for _ in range(3)]

    assert [state.value for state in states] == ["running", "running", "failed"]
    assert model.calls == 4
    assert "Error in gemini/tool_execution" in bot.sent[-1][1]
    assert "private global provider failure" not in bot.sent[-1][1]


def test_malformed_governed_route_clarifies_without_retrying_provider(firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    class MalformedModel:
        def __init__(self):
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return {"text": "not a structured route", "parts": [{"text": "not a structured route"}]}
    model, bot = MalformedModel(), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    result = asyncio.run(runtime(context_id, operation, lease, command=command,
                                 retry_governor=governor), loop_factory=_selector_loop)

    assert result.value == "completed"
    assert model.calls == 1
    assert bot.sent[0][1] == "I could not determine what you want me to do. Please clarify."


def _search_route():
    return {"kind": "search", "target_ref": None,
            "target_operation_id": None, "dialog_number": None,
            "result_refs": [], "substantive_text": None,
            "tool_choice": None, "certainty": "certain"}


def test_search_route_is_governed_replayable_and_isolated(monkeypatch, firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    calls = []
    monkeypatch.setattr("core.operations.read.query",
                        lambda **kwargs: calls.append(kwargs) or (
                            [{"doc_id": "doc-search", "title": "Planning"}],
                            {"total": 1, "query": kwargs["query"]},
                        ))
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    bot, model = _Bot(), _RuntimeModel(_search_route())
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    first = asyncio.run(runtime(context_id, operation, lease, command=command,
                                retry_governor=governor), loop_factory=_selector_loop)
    sent = len(bot.sent)
    replay = asyncio.run(runtime(context_id, operation, lease, command=None,
                                 retry_governor=governor), loop_factory=_selector_loop)

    assert first.value == replay.value == "completed"
    assert len(calls) == 1
    assert len(bot.sent) == sent
    assert bot.sent[0][1] == "Search results: Planning"
    assert not any(value.get("kind") == "document"
                   for path, value in firestore_fake.documents.items()
                   if path.startswith(f"telegram_sessions/{context_id}/conversations/"))
    assert any(value.get("kind") == "search"
               for path, value in firestore_fake.documents.items()
               if path.startswith(f"telegram_sessions/{context_id}/conversations/"))


def test_search_provider_failures_use_three_stage_budget_and_safe_error(monkeypatch, firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    calls = []
    def fail_query(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private search provider details")
    monkeypatch.setattr("core.operations.read.query", fail_query)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    bot, model = _Bot(), _RuntimeModel(_search_route())
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(context_id, operation, lease, command=command,
                                  retry_governor=governor), loop_factory=_selector_loop)
              for _ in range(3)]

    assert [state.value for state in states] == ["running", "running", "failed"]
    assert len(calls) == 3
    assert "operation_id=op-runtime" in bot.sent[-1][1]
    assert "private search provider details" not in bot.sent[-1][1]


def test_search_route_uses_one_real_embedding_attempt_per_durable_retry(
    monkeypatch, firestore_fake, sdk_fakes,
):
    from types import SimpleNamespace
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    for _ in range(3):
        sdk_fakes.gemini.queue(
            "embed_content", error=ConnectionError("synthetic embedding outage"),
        )
        sdk_fakes.qdrant.queue(
            "query_points", response=SimpleNamespace(points=[]),
        )
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    bot, model = _Bot(), _RuntimeModel(_search_route())
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(
        context_id, operation, lease, command=command,
        retry_governor=governor,
    ), loop_factory=_selector_loop) for _ in range(3)]

    embedding_calls = [event for event in sdk_fakes.trace
                       if event["boundary"] == "gemini"
                       and event["operation"] == "embed_content"]
    assert [state.value for state in states] == ["running", "running", "failed"]
    assert len(embedding_calls) == 3
    assert not any("synthetic embedding outage" in str(event)
                   for event in sdk_fakes.trace)
    assert "synthetic embedding outage" not in str(bot.sent)
    search_stage_key = next(
        path.rsplit("/", 1)[-1] for path in firestore_fake.documents
        if path.startswith(
            f"telegram_sessions/{context_id}/operations/{operation['operation_id']}/stages/"
        ) and "telegram-search-" in path
    )
    assert governor.get_stage(
        operation["operation_id"], search_stage_key,
    ).last_error.service == "gemini"


def test_search_route_governs_qdrant_failures_from_both_parallel_branches(
    monkeypatch, firestore_fake, sdk_fakes,
):
    from types import SimpleNamespace
    from google.genai import types
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    embedding = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    )
    for _ in range(3):
        sdk_fakes.gemini.queue("embed_content", response=embedding)
        for _branch in range(2):
            sdk_fakes.qdrant.queue(
                "query_points", error=ConnectionError("synthetic qdrant outage"),
            )
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    bot, model = _Bot(), _RuntimeModel(_search_route())
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(
        context_id, operation, lease, command=command, retry_governor=governor,
    ), loop_factory=_selector_loop) for _ in range(3)]

    calls = [event for event in sdk_fakes.trace
             if event["boundary"] == "qdrant" and event["operation"] == "query_points"]
    assert [state.value for state in states] == ["running", "running", "failed"]
    assert len(calls) == 6
    search_stage_key = next(
        path.rsplit("/", 1)[-1] for path in firestore_fake.documents
        if path.startswith(
            f"telegram_sessions/{context_id}/operations/{operation['operation_id']}/stages/"
        ) and "telegram-search-" in path
    )
    stage = governor.get_stage(operation["operation_id"], search_stage_key)
    assert stage.failures == 3
    assert stage.last_error.service == "qdrant"
    assert "synthetic qdrant outage" not in str(bot.sent)


def test_default_telegram_search_constructs_qdrant_without_background_probe(
    monkeypatch, firestore_fake, sdk_fakes,
):
    from types import SimpleNamespace
    from google.genai import types
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    monkeypatch.setenv("QDRANT_URL", "https://synthetic.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "synthetic-qdrant-secret")
    from core.integrations.qdrant import client as qdrant_client
    monkeypatch.setattr(qdrant_client, "_single_attempt_client", None)
    monkeypatch.setattr(qdrant_client, "_client", None)
    empty_embedding = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    )
    sdk_fakes.gemini.queue("embed_content", response=empty_embedding)
    sdk_fakes.gemini.queue("embed_content", response=empty_embedding)
    for _ in range(3):
        sdk_fakes.qdrant.queue("query_points", response=SimpleNamespace(points=[]))

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    bot, model = _Bot(), _RuntimeModel(_search_route())
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    result = asyncio.run(runtime(
        context_id, operation, lease, command=command, retry_governor=governor,
    ), loop_factory=_selector_loop)

    assert result.value == "completed"
    client_events = [event for event in sdk_fakes.trace
                     if event["boundary"] == "qdrant" and event["operation"] == "client"]
    assert len(client_events) == 1
    assert client_events[0]["arguments"].get("check_compatibility") is False
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant"
                and event["operation"] == "query_points"]) == 3


def test_agent_ledger_replays_saved_tool_result_and_rejects_stale_fence(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(client, context_id=context_id,
                                  operation_id=operation["operation_id"],
                                  owner=lease.owner, fence=lease.fence,
                                  clock=lambda: now)
    calls = [{"id": "call-0", "name": "list_folders", "arguments": {},
              "ordinal": 0, "round": 0}]
    ledger.save_turn("global-conversation", operation["operation_id"],
                     model_response={"_telegram_tool_loop_rounds": 1, "rounds": [
                         {"response": {"text": "round-0"}, "call_start": 0, "call_count": 1},
                     ]}, tool_calls=calls)
    ledger.save_tool_result("global-conversation", operation["operation_id"], 0,
                            {"folders": ["Research"]})
    replay = ledger.load_turn("global-conversation", operation["operation_id"])
    assert replay["tool_calls"] == calls
    assert replay["tool_results"] == {"0": {"folders": ["Research"]}}
    second_call = {**calls[0], "id": "call-1", "ordinal": 1, "round": 1}
    ledger.save_turn("global-conversation", operation["operation_id"],
                     model_response={"_telegram_tool_loop_rounds": 1, "rounds": [
                         {"response": {"text": "round-0"}, "call_start": 0, "call_count": 1},
                         {"response": {"text": "round-1"}, "call_start": 1, "call_count": 1},
                     ]}, tool_calls=[*calls, second_call])
    with pytest.raises(ValueError):
        ledger.save_turn("global-conversation", operation["operation_id"],
                         model_response={"_telegram_tool_loop_rounds": 1, "rounds": [
                             {"response": {"text": "tampered"}, "call_start": 0, "call_count": 1},
                         ]}, tool_calls=calls)

    stale = FirestoreAgentLedger(client, context_id=context_id,
                                 operation_id=operation["operation_id"],
                                 owner="stale-owner", fence=lease.fence,
                                 clock=lambda: now)
    with pytest.raises(PermissionError):
        stale.save_tool_result("global-conversation", operation["operation_id"], 1,
                               {"folders": []})


def test_agent_ledger_checkpoints_application_route_outside_model_round(firestore_fake):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(
        client, operation_id="op-application-route",
    )
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    route = {"name": "list_recent_transcripts", "arguments": {}, "ordinal": 0,
             "target_doc_id": "doc-selected"}
    application_call = {
        "id": None, "name": route["name"], "arguments": {},
        "ordinal": 0, "round": -1,
    }
    ledger.save_application_route(
        "document-conversation", operation["operation_id"], route=route,
        tool_calls=[application_call], application_route_round=None,
    )
    result = [{"doc_id": "doc-selected", "title": "Synthetic dialog"}]
    ledger.save_tool_result(
        "document-conversation", operation["operation_id"], 0, result,
    )

    checkpoint = ledger.load_turn("document-conversation", operation["operation_id"])
    assert checkpoint["model_response"] is None
    assert checkpoint["application_route"] == route
    assert checkpoint["application_route_round"] is None
    assert checkpoint["tool_calls"] == [application_call]
    assert checkpoint["tool_results"] == {"0": result}

    # Completing the answer appends a real model round without introducing an
    # app call/response exchange or mutating the already checkpointed route.
    ledger.save_turn(
        "document-conversation", operation["operation_id"],
        model_response={"_telegram_tool_loop_rounds": 1, "rounds": [
            {"response": {"text": "The dialog was synthetic."},
             "call_start": 1, "call_count": 0},
        ]},
        tool_calls=[application_call], application_route=route,
    )
    completed = ledger.load_turn("document-conversation", operation["operation_id"])
    assert completed["application_route"] == route
    assert completed["model_response"]["rounds"][0]["response"]["text"] == (
        "The dialog was synthetic."
    )


def test_agent_ledger_parts_large_application_route_and_preserves_it(firestore_fake):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from core.integrations.firestore.input_parts import MAX_PART_BYTES

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(
        client, operation_id="op-large-application-route",
    )
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    arguments = {"utterances": [{"speaker": "Speaker", "text": "x" * (1024 * 1024 + 8192)}]}
    route = {"name": "add_dialog", "arguments": arguments, "ordinal": 0,
             "target_doc_id": "doc-large"}
    application_call = {
        "id": None, "name": "add_dialog", "arguments": arguments,
        "ordinal": 0, "round": -1,
    }
    ledger.save_application_route(
        "document-conversation", operation["operation_id"], route=route,
        tool_calls=[application_call], application_route_round=None,
    )
    ledger.save_tool_result(
        "document-conversation", operation["operation_id"], 0, {"accepted": True},
    )

    checkpoint_ref = ledger._ref(operation["operation_id"])
    parent = checkpoint_ref.get().to_dict()
    assert len(ledger._json_bytes(parent)) < 4 * 1024
    assert parent["application_route"].get("_input_ref")
    assert parent["tool_calls"].get("_input_ref")
    assert not any("__" in key for key in parent)
    part_payloads = [row["payload"] for path, row in firestore_fake.documents.items()
                     if "/input_parts/part-" in path]
    assert part_payloads and max(map(len, part_payloads)) <= MAX_PART_BYTES

    # A following native round may omit the route argument; the ledger keeps
    # the already checkpointed route envelope and restores its exact value.
    native_call = {
        "id": "native-call-1", "name": "query", "arguments": {"query": "synthetic"},
        "ordinal": 1, "round": 1,
    }
    ledger.save_turn(
        "document-conversation", operation["operation_id"],
        model_response={"_telegram_tool_loop_rounds": 1, "rounds": [
            {"response": {"text": "Synthetic native round."},
             "call_start": 1, "call_count": 1},
        ]},
        tool_calls=[application_call, native_call],
    )
    restored = ledger.load_turn("document-conversation", operation["operation_id"])
    assert restored["application_route"] == route
    assert restored["application_route_round"] is None
    assert restored["tool_calls"] == [application_call, native_call]
    assert restored["tool_results"] == {"0": {"accepted": True}}
    stored_after_round = checkpoint_ref.get().to_dict()
    assert stored_after_round["application_route"] == parent["application_route"]
    assert len(ledger._json_bytes(stored_after_round)) < 4 * 1024

    with pytest.raises(ValueError, match="application route is immutable"):
        ledger.save_application_route(
            "document-conversation", operation["operation_id"],
            route={**route, "target_doc_id": "other-doc"},
            tool_calls=[application_call, native_call], application_route_round=None,
        )
    with pytest.raises(ValueError, match="application route is immutable"):
        ledger.save_application_route(
            "document-conversation", operation["operation_id"], route=route,
            tool_calls=[application_call, native_call], application_route_round=0,
        )

    stale = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner="stale-owner", fence=lease.fence, clock=lambda: now,
    )
    with pytest.raises(PermissionError, match="lease is stale"):
        stale.save_application_route(
            "document-conversation", operation["operation_id"], route=route,
            tool_calls=[application_call, native_call], application_route_round=None,
        )


def test_agent_ledger_loads_legacy_raw_application_route_map(firestore_fake):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(
        client, operation_id="op-legacy-application-route",
    )
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    route = {"name": "get_transcript", "arguments": {"doc_id": "doc-legacy"},
             "ordinal": 0, "target_doc_id": "doc-legacy"}
    ledger._ref(operation["operation_id"]).set({
        "conversation_id": "document-conversation",
        "delivery_id": operation["operation_id"],
        "model_response": None, "tool_calls": [],
        "application_route": route, "application_route_round": None,
    })
    assert ledger.load_turn(
        "document-conversation", operation["operation_id"],
    )["application_route"] == route


def test_agent_ledger_parts_large_results_without_truncating(firestore_fake):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from core.integrations.firestore.input_parts import MAX_PART_BYTES

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(client, context_id=context_id,
                                  operation_id=operation["operation_id"],
                                  owner=lease.owner, fence=lease.fence,
                                  clock=lambda: now)
    large = {"text": "x" * (1024 * 1024 + 8192)}
    ledger.save_turn("global-conversation", operation["operation_id"],
                     model_response={"text": "done", "parts": []}, tool_calls=[])
    ledger.save_tool_result("global-conversation", operation["operation_id"], 0, large)
    restored = ledger.load_turn("global-conversation", operation["operation_id"])

    assert restored["tool_results"]["0"] == large
    payloads = [value["payload"] for path, value in firestore_fake.documents.items()
                if "/input_parts/part-" in path]
    assert len(payloads) >= 3
    assert max(map(len, payloads)) <= MAX_PART_BYTES

    inline = FirestoreAgentLedger(client, context_id=context_id,
                                  operation_id=operation["operation_id"],
                                  owner=lease.owner, fence=lease.fence,
                                  clock=lambda: now)
    for ordinal in range(1, 65):
        inline.save_tool_result("global-conversation", operation["operation_id"],
                                ordinal, {"text": "y" * 20_000})
    for ordinal in range(100, 400):
        inline.save_tool_result("global-conversation", operation["operation_id"],
                                ordinal, {"text": "z" * 3900})
    restored_many = inline.load_turn("global-conversation", operation["operation_id"])
    assert len(restored_many["tool_results"]) == 365
    assert restored_many["tool_results"]["399"] == {"text": "z" * 3900}
    for path, value in firestore_fake.documents.items():
        if (path.endswith("/agent_checkpoints/turn")
                or "/agent_checkpoints/turn/tool_results/" in path
                or "/input_parts/part-" in path):
            assert len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")) < 1024 * 1024
    checkpoint = next(value for path, value in firestore_fake.documents.items()
                      if path.endswith("/agent_checkpoints/turn"))
    assert len(json.dumps(checkpoint, ensure_ascii=False).encode("utf-8")) < 1024 * 1024


def test_runtime_store_returns_bound_and_unbound_scoped_documents(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    client.collection("telegram_sessions").document(context_id).update({
        "last_doc_id": "saved-doc-b",
    })
    client.collection("folders").document("root-1").set({
        "folder_id": "root-1", "name": "Root", "parent_id": "pytest-synthetic-root",
    })
    client.collection("folders").document("child-1").set({
        "folder_id": "child-1", "name": "Child", "parent_id": "root-1",
    })
    for doc_id in ("bound-doc", "unbound-doc", "saved-doc-b"):
        client.collection("transcript_index").document(doc_id).set({
            "source_file": doc_id, "parent_id": "child-1",
            "created_at": datetime.now(timezone.utc),
        })
    number = repository.bind_document(context_id, "bound-doc", "conversation-bound",
                                      metadata={}, now=datetime.now(timezone.utc),
                                      update_last_doc=False)
    store = TelegramRuntimeStore(repository, root_folder_id="pytest-synthetic-root",
                                 context_id=context_id)

    candidates = {row["doc_id"]: row for row in store.existing_dialog_candidates()}

    assert candidates["bound-doc"]["dialog_number"] == number
    assert candidates["bound-doc"]["conversation_id"] == "conversation-bound"
    assert candidates["bound-doc"]["title"] == "bound-doc"
    assert candidates["bound-doc"]["folder_path"] == ["Root", "Child"]
    assert candidates["unbound-doc"]["conversation_id"].startswith("document-")
    assert "dialog_number" not in candidates["unbound-doc"]
    assert store.publish_selection(
        operation["operation_id"], conversation_id="conversation-bound",
        dialog_number=number, owner=lease.owner, fence=lease.fence,
        now=datetime.now(timezone.utc),
    )
    from services.connectors.telegram.models import TelegramContext
    context_row = repository.get_context(context_id)
    context = TelegramContext(**{field.name: context_row[field.name]
                                 for field in __import__("dataclasses").fields(TelegramContext)
                                 if field.name in context_row})
    from services.connectors.telegram.sessions import _resolve_target
    assert _resolve_target({"dialog_number": None, "target_ref": "last"},
                           context=context, store=store,
                           candidates=tuple(candidates)) == ("saved-doc-b", 2, None)
    assert repository.get_context(context_id)["last_doc_id"] == "saved-doc-b"


def test_runtime_offers_only_confirmed_saves_in_the_immutable_predecessor_chain(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, _lease, _command = _runtime_queue_item(client)
    operations = client.collection("telegram_sessions").document(context_id).collection("operations")
    operations.document(operation["operation_id"]).update({
        "predecessor_operation_id": "op-save",
    })
    operations.document("op-save").set({
        "operation_id": "op-save", "context_id": context_id,
        "state": "completed", "predecessor_operation_id": "op-old",
        "save_operation_id": "op-save", "target_doc_id": "doc-saved",
        "target_dialog_number": 5, "target_conversation_id": "conversation:doc-saved",
        "title": "Saved topic",
        "route_result": {"kind": "save", "target_doc_id": "doc-saved",
                         "target_dialog_number": 5,
                         "target_conversation_id": "conversation:doc-saved"},
    })
    operations.document("op-old").set({"operation_id": "op-old", "context_id": context_id,
                                        "state": "failed"})
    operations.document("foreign-save").set({
        "operation_id": "foreign-save", "context_id": context_id,
        "state": "completed", "save_operation_id": "foreign-save",
        "target_doc_id": "doc-foreign", "target_dialog_number": 9,
        "target_conversation_id": "conversation:foreign",
        "route_result": {"kind": "save", "target_doc_id": "doc-foreign",
                         "target_dialog_number": 9,
                         "target_conversation_id": "conversation:foreign"},
    })
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object())

    candidates = runtime._predecessor_save_candidates(context_id, operation)

    assert candidates == ({
        "operation_id": "op-save", "doc_id": "doc-saved", "dialog_number": 5,
        "conversation_id": "conversation:doc-saved", "title": "Saved topic",
        "state": "completed", "save_operation_id": "op-save",
        "target_doc_id": "doc-saved", "target_dialog_number": 5,
        "target_conversation_id": "conversation:doc-saved",
        "route_result": {"kind": "save"},
    },)


def test_document_route_publishes_notice_then_answer_and_replays_saved_history(
    firestore_fake, monkeypatch,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("transcript_index").document("doc-1").set({
        "meeting_id": "meeting-1", "source_file": "Planning", "parent_id": "root-1",
        "dialog_date": "2026-01-02", "status": "synced",
    })
    client.collection("folders").document("root-1").set({
        "folder_id": "root-1", "name": "Root", "parent_id": "pytest-synthetic-root",
    })
    route = _route()
    route.update({"target_ref": "doc-1", "substantive_text": "Summarize the decision."})
    bot, model = _Bot(), _RuntimeModel(route, "The decision was to proceed.")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-synthetic-root")
    from services.connectors.telegram.runtime import runtime as runtime_module
    deliver = runtime_module._DurableCommandReplies.deliver
    crashes = {"count": 0}
    async def interrupted(self, body, *, kind="final"):
        if kind == "final" and crashes["count"] == 0:
            crashes["count"] += 1
            raise RuntimeError("simulated interruption after model checkpoint")
        return await deliver(self, body, kind=kind)
    monkeypatch.setattr(runtime_module._DurableCommandReplies, "deliver", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        asyncio.run(runtime(context_id, operation, lease, command=command),
                    loop_factory=_selector_loop)
    client.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-foreign",
    })
    first = asyncio.run(runtime(context_id, operation, lease, command=command),
                        loop_factory=_selector_loop)
    assert first.value == "completed", (
        repository.get_route_control(context_id, operation["operation_id"]),
        [row[0] for row in model.calls], bot.sent,
        [(path, value.get("state"), value.get("kind"), value.get("body"))
         for path, value in firestore_fake.documents.items()
         if path.startswith("notification_outbox/")],
    )
    assert [row[1] for row in bot.sent] == [
        "Starting to process dialog #1.", "The decision was to proceed.",
    ]
    turns = repository.list_turns(context_id, "document-" + __import__(
        "hashlib").sha256(f"{context_id}\0doc-1".encode()).hexdigest())
    assert [turn["role"] for turn in turns] == ["user", "model"]
    assert turns[0]["parts"][0]["text"] == "Summarize the decision."
    assert turns[1]["parts"][0]["text"] == bot.sent[-1][1]

    sent_count = len(bot.sent)
    second = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)
    assert second.value == "completed"
    assert len(bot.sent) == sent_count
    assert [mode for mode, _contents, _tools in model.calls].count("tool_loop") == 1


def test_router_receives_ordered_safe_results_from_predecessor_search(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, _lease, _command = _runtime_queue_item(client)
    operations = client.collection("telegram_sessions").document(context_id).collection("operations")
    operations.document(operation["operation_id"]).update({"predecessor_operation_id": "op-search"})
    operations.document("op-search").set({
        "operation_id": "op-search", "context_id": context_id, "state": "completed",
        "route_control": {"status": "ok", "kind": "search",
                          "result_refs": ["first", "second"]},
    })
    cards = [
        {"doc_id": "second", "title": "Second", "private_transcript": "secret"},
        {"doc_id": "first", "title": "First"},
    ]
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object())
    recent = runtime._predecessor_search_results(context_id, operation, cards)

    class Model:
        def generate(self, **kwargs):
            self.prompt = json.loads(kwargs["contents"][0]["parts"][0]["text"])
            route = _route()
            route["target_ref"] = self.prompt["recent_search_results"][0]["doc_id"]
            return {"parts": [{"function_call": {"name": "route_command", "args": route}}]}

    model = Model()
    selected = GeminiCommandRouter(model).route(text="first of found", candidates=cards,
                                                 recent_search_results=recent)
    assert [item["doc_id"] for item in model.prompt["recent_search_results"]] == ["first", "second"]
    assert selected["target_ref"] == "first"
    assert "private_transcript" not in repr(model.prompt)


def test_global_registry_tool_failure_uses_governed_settlement(monkeypatch, firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    calls = []
    def fail_tool(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("private tool provider details")
    monkeypatch.setattr("services.connectors.telegram.runtime.tools.invoke_tool", fail_tool)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
    )
    bot, model = _Bot(), _RuntimeModel(
        _route(tool_choice={"name": "list_folders", "arguments": {}}))
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model, clock=lambda: now)

    states = [asyncio.run(runtime(context_id, operation, lease, command=command,
                                  retry_governor=governor), loop_factory=_selector_loop)
              for _ in range(3)]

    assert [state.value for state in states] == ["running", "running", "failed"]
    assert len(calls) == 3
    assert "operation_id=op-runtime" in bot.sent[-1][1]
    assert "private tool provider details" not in bot.sent[-1][1]


def test_explicit_document_tool_target_uses_pinned_session_executor(monkeypatch, firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    calls = []
    monkeypatch.setattr("services.connectors.telegram.runtime.tools.invoke_tool",
                        lambda name, arguments, **kwargs: calls.append((name, arguments, kwargs))
                        or {"doc_id": "doc-1", "transcript": "approved context"})

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("transcript_index").document("doc-1").set({
        "source_file": "Planning", "parent_id": "root-1",
    })
    client.collection("folders").document("root-1").set({
        "folder_id": "root-1", "name": "Root", "parent_id": "pytest-synthetic-root",
    })
    client.collection("telegram_sessions").document(context_id).update({
        "last_doc_id": "doc-b",
    })
    route = _route(tool_choice={"name": "get_transcript", "arguments": {}})
    route.update({"target_ref": "doc-1", "substantive_text": "Summarize this document."})
    bot, model = _Bot(), _RuntimeModel(route, "The document summary.")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-synthetic-root")

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert bot.sent[0][1] == "Starting to process dialog #1."
    assert calls[0][0] == "get_transcript"
    assert calls[0][1]["doc_id"] == "doc-1"
    assert calls[0][2]["context"]["target_doc_id"] == "doc-1"
    assert len(bot.sent) == 2
    assert repository.get_context(context_id)["last_doc_id"] == "doc-b"


def test_runtime_pure_switch_sends_notice_without_appending_turns(firestore_fake, monkeypatch):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    client.collection("transcript_index").document("doc-switch").set({
        "source_file": "Planning", "parent_id": "root-switch",
    })
    client.collection("folders").document("root-switch").set({
        "folder_id": "root-switch", "name": "Root", "parent_id": "pytest-switch-root",
    })
    route = {"kind": "pure_switch", "target_ref": "doc-switch",
             "target_operation_id": None, "dialog_number": None,
             "result_refs": [], "substantive_text": None,
             "tool_choice": None, "certainty": "certain"}
    bot, model = _Bot(), _RuntimeModel(route)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-switch-root")

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert [row[1] for row in bot.sent] == ["Starting to process dialog #1."]
    assert [mode for mode, _contents, _tools in model.calls] == ["routing"]
    context = repository.get_context(context_id)
    assert context["selected_dialog_number"] == 1
    assert repository.list_turns(context_id, context["active_conversation_id"]) == []


def test_unknown_explicit_document_tool_target_fails_closed(firestore_fake, monkeypatch):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    route = _route(tool_choice={"name": "get_transcript", "arguments": {"doc_id": "foreign-doc"}})
    route.update({"target_ref": "foreign-doc", "substantive_text": "Summarize this document."})
    bot, model = _Bot(), _RuntimeModel(route, "must not run")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-synthetic-root")

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert [mode for mode, _contents, _tools in model.calls] == ["routing"]
    assert bot.sent[0][1] == "I could not complete that request. Please clarify."
    assert not any(row[1] == "Starting to process dialog #1." for row in bot.sent)
    context = repository.get_context(context_id)
    assert context["last_doc_id"] is None and context["active_conversation_id"] is None
