"""Offline checks for the concrete Telegram runtime adapters."""

from __future__ import annotations

import asyncio
from functools import partial
import inspect
import json
import os
import socket
from collections.abc import Mapping
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


def test_router_prompt_treats_bare_topic_description_as_annotation_and_exposes_verified_draft():
    import json
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter

    class Model:
        def generate(self, **kwargs):
            self.request = kwargs
            return {"parts": [{"function_call": {"name": "route_command", "args": _route()}}]}

    model = Model()
    GeminiCommandRouter(model).route(
        text="Project Atlas planning", candidates=[],
        context_metadata={
            "selected_conversation_kind": "search",
            "selected_draft_id": "untrusted-selected-draft",
            "captured_collecting_draft": {
                "draft_id": "accepted-draft-8", "dialog_number": 8, "state": "collecting",
            },
            "unapproved_extra": "must not be included",
        },
    )
    prompt = json.loads(model.request["contents"][0]["parts"][0]["text"])
    assert prompt["context"].get("captured_collecting_draft") == {
        "draft_id": "accepted-draft-8", "dialog_number": 8, "state": "collecting",
    }
    assert "declarative topic or title" in prompt["instruction"]
    assert "does not authorize corpus search" in prompt["instruction"]
    assert "unapproved_extra" not in repr(prompt)


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

    async def send_message(self, chat_id, text, *, message_thread_id=None, presentation=None):
        self.sent.append((chat_id, text, message_thread_id))
        return {"message_id": len(self.sent)}

    async def edit_message_text(self, chat_id, message_id, text, *, presentation=None):
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


def test_long_receipt_save_result_keeps_card_and_durable_overflow_chunks(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    operation_ref = (client.collection("telegram_sessions").document(context_id)
                     .collection("operations").document(operation["operation_id"]))
    operation_ref.update({"receipt_card_notification_id": "receipt-card",
                          "target_draft_id": "draft-1"})
    client.collection("notification_outbox").document("receipt-card").set({
        "notification_id": "receipt-card", "context_id": context_id,
        "draft_id": "draft-1", "receipt_card": True,
        "save_operation_id": operation["operation_id"],
        "receipt_terminal": False, "revision": 1, "state": "delivered",
        "body": "Dialog #1: Saving...", "reply_markup": {"inline_keyboard": []},
        "chat_id": 42, "message_id": 901,
    })

    class Publisher:
        def __init__(self):
            self.calls = []

        async def publish(self, notification_id):
            self.calls.append(notification_id)
            return "delivered"

    publisher = Publisher()
    body = "Dialog #1: Saved and indexed. Folder: " + "Nested/" * 700
    replies = _DurableCommandReplies(
        repository=repository, publisher=publisher, context_id=context_id,
        operation_id=operation["operation_id"], owner=lease.owner,
        fence=lease.fence, clock=lambda: datetime.now(timezone.utc),
    )
    result = asyncio.run(replies.deliver(body), loop_factory=_selector_loop)

    card = repository.get_notification("receipt-card")
    overflow = [repository.get_notification(item) for item in publisher.calls
                if item != "receipt-card"]
    delivered_text = card["body"] + "".join(row["body"] for row in overflow)
    assert result == "delivered"
    assert card["receipt_terminal"] is True
    assert card["body"].startswith("Dialog #1: Saved and indexed. Folder: ")
    assert delivered_text == body
    assert len(publisher.calls) == 2


def test_receipt_terminal_rich_edit_replays_same_card_after_lost_ack(firestore_fake):
    import httpx
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    operation_ref = (client.collection("telegram_sessions").document(context_id)
                     .collection("operations").document(operation["operation_id"]))
    operation_ref.update({"receipt_card_notification_id": "receipt-rich-card",
                          "target_draft_id": "draft-1"})
    card_ref = client.collection("notification_outbox").document("receipt-rich-card")
    card_ref.set({
        "notification_id": "receipt-rich-card", "context_id": context_id,
        "draft_id": "draft-1", "receipt_card": True, "kind": "draft_ack",
        "save_operation_id": operation["operation_id"],
        "receipt_terminal": False, "revision": 1, "state": "delivered",
        "body": "Dialog #1: Saving...", "reply_markup": {"inline_keyboard": []},
        "chat_id": 42, "message_id": 901, "send_attempts_started": 1,
        "edit_attempts_started": 0,
    })
    now = [datetime.now(timezone.utc)]
    calls = []
    lose_first_edit_ack = {"value": True}

    def response(request):
        payload = json.loads(request.content)
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append((method, payload))
        if method == "editMessageText" and lose_first_edit_ack["value"]:
            lose_first_edit_ack["value"] = False
            raise httpx.ReadTimeout("synthetic lost edit acknowledgement")
        if method == "editMessageText":
            return httpx.Response(400, json={
                "ok": False, "error_code": 400,
                "description": "Bad Request: message is not modified",
            })
        return httpx.Response(200, json={
            "ok": True, "result": {"message_id": payload.get("message_id", 901)},
        })

    http = httpx.AsyncClient(transport=httpx.MockTransport(response))
    first_publisher = TelegramNotificationPublisher(
        repository, TelegramClient("123456:synthetic-token", http_client=http),
        clock=lambda: now[0],
    )
    body = "Answer: **Saved** in _unassigned."

    def replies(publisher):
        return _DurableCommandReplies(
            repository=repository, publisher=publisher, context_id=context_id,
            operation_id=operation["operation_id"], owner=lease.owner,
            fence=lease.fence, clock=lambda: now[0],
        )

    async def exercise():
        try:
            first = await replies(first_publisher).deliver(
                body, presentation="rich_markdown",
            )
            row = repository.get_notification("receipt-rich-card")
            assert first == "pending"
            assert row["message_id"] == 901 and row["presentation"] == "rich_markdown"
            assert row["reply_markup"] == {"inline_keyboard": []}
            assert row["edit_attempts_by_revision"]["2"] == 1

            now[0] += timedelta(seconds=1)
            restarted_publisher = TelegramNotificationPublisher(
                repository, TelegramClient("123456:synthetic-token", http_client=http),
                clock=lambda: now[0],
            )
            assert await replies(restarted_publisher).deliver(
                body, presentation="rich_markdown",
            ) == "delivered"
            assert await replies(restarted_publisher).deliver(
                body, presentation="rich_markdown",
            ) == "delivered"
        finally:
            await http.aclose()

    asyncio.run(exercise(), loop_factory=_selector_loop)
    assert [method for method, _payload in calls] == ["editMessageText", "editMessageText"]
    assert calls[0][1]["rich_message"] == {"markdown": body}
    assert calls[1][1]["rich_message"] == {"markdown": body}
    assert all(payload["reply_markup"] == {"inline_keyboard": []}
               and "text" not in payload and "parse_mode" not in payload
               for _method, payload in calls)
    card = repository.get_notification("receipt-rich-card")
    assert card["message_id"] == 901 and card["presentation"] == "rich_markdown"
    assert card["state"] == "delivered" and card["edit_attempts_by_revision"]["2"] == 2
    assert card["send_attempts_started"] == 1


def test_stale_reply_fence_never_publishes_a_status_as_notification_id(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    expired_now = datetime.now(timezone.utc) + timedelta(minutes=5)
    client.collection("telegram_sessions").document(context_id).update({
        "lock_expires_at": expired_now - timedelta(seconds=1),
    })

    class Publisher:
        calls = []

        async def publish(self, notification_id):
            self.calls.append(notification_id)
            return "delivered"

    publisher = Publisher()
    replies = _DurableCommandReplies(
        repository=repository, publisher=publisher, context_id=context_id,
        operation_id=operation["operation_id"], owner=lease.owner,
        fence=lease.fence, clock=lambda: expired_now,
    )
    result = asyncio.run(replies.deliver("A safe final answer."),
                         loop_factory=_selector_loop)

    assert result == "stale"
    assert publisher.calls == []


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
        sends.append(payload.get("text") or payload["rich_message"]["markdown"])
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
    assert [row[0] for row in model.calls] == ["routing", "tool_loop"]
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


def test_ungoverned_model_routing_failure_delivers_safe_processing_error(firestore_fake, caplog):
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

    assert result.value == "failed"
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    assert repository.get_route_control(context_id, operation["operation_id"])["status"] == "invalid_route"
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


def test_malformed_governed_route_fails_truthfully_without_retrying_provider(firestore_fake):
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

    assert result.value == "failed"
    assert model.calls == 1
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    assert repository.get_route_control(context_id, operation["operation_id"])["status"] == "invalid_route"


def test_explicit_model_uncertainty_is_a_completed_clarification(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    route = {**_route(), "certainty": "uncertain"}
    model, bot = _RuntimeModel(route), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "completed"
    assert len(model.calls) == 1
    assert bot.sent[0][1] == "I could not determine what you want me to do. Please clarify."
    assert repository.get_route_control(context_id, operation["operation_id"])["status"] == "uncertain"


def test_malformed_certain_save_route_is_failed_without_save_effect(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    class SaveHandler:
        def __init__(self):
            self.calls = []
        def save_draft(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    route = {
        "kind": "save", "target_ref": None, "target_operation_id": "synthetic-save",
        "dialog_number": None, "result_refs": [], "substantive_text": None,
        "tool_choice": None, "certainty": "certain",
    }
    handler, bot = SaveHandler(), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=handler,
                                     client=bot, model=_RuntimeModel(route))

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert handler.calls == []
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    assert repository.get_route_control(context_id, operation["operation_id"])["status"] == "invalid_route"


@pytest.mark.parametrize("answer", ["", " \n "])
def test_blank_global_model_answer_is_a_failed_processing_error(firestore_fake, answer):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    bot = _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=_RuntimeModel(_route(), answer=answer))

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    conversation_id = "telegram-chat-" + __import__("hashlib").sha256(
        context_id.encode("utf-8")).hexdigest()
    assert [turn["role"] for turn in repository.list_turns(
        context_id, conversation_id,
    )] == ["user"]


def test_ungoverned_tool_loop_exception_delivers_safe_failure(firestore_fake, caplog):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    class FailingAnswerModel:
        def __init__(self):
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            if kwargs.get("mode") == "routing":
                return {"parts": [{"function_call": {
                    "name": "route_command", "args": _route(),
                }}]}
            raise RuntimeError("private answer provider text")

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    model, bot = FailingAnswerModel(), _Bot()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model)

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert model.calls == 2
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    assert "private answer provider text" not in caplog.text


@pytest.mark.parametrize(("status", "expected_state", "expected_reply"), [
    ("invalid_route", "failed",
     "An error occurred while processing your request. Please try again."),
    ("uncertain", "completed",
     "I could not determine what you want me to do. Please clarify."),
    ("error", "failed",
     "An error occurred while processing your request. Please try again."),
])
@pytest.mark.parametrize("command_present", [False, True])
@pytest.mark.parametrize("kind", ["save", "substantive", None])
def test_pinned_route_failure_replay_does_not_call_model_or_save(
    firestore_fake, status, expected_state, expected_reply, command_present, kind,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    assert repository.persist_route(
        operation["operation_id"], {"status": status, "kind": kind},
        context_id=context_id, owner=lease.owner, fence=lease.fence, now=now,
    )
    model, bot = _RuntimeModel(_route()), _Bot()
    class SaveHandler:
        def __init__(self):
            self.calls = []
        def save_draft(self, *args, **kwargs):
            self.calls.append((args, kwargs))
    handler = SaveHandler()
    runtime = TelegramCommandRuntime(repository=repository, save_handler=handler,
                                     client=bot, model=model, clock=lambda: now)

    pinned_before = repository.get_route_control(context_id, operation["operation_id"])
    result = asyncio.run(runtime(
        context_id, operation, lease,
        command=_command if command_present else None,
    ),
                         loop_factory=_selector_loop)

    assert result.value == expected_state
    assert model.calls == []
    assert handler.calls == []
    assert bot.sent[0][1] == expected_reply
    assert repository.get_route_control(context_id, operation["operation_id"]) == pinned_before


def _search_route():
    return {"kind": "search", "target_ref": None,
            "target_operation_id": None, "dialog_number": None,
            "result_refs": [], "substantive_text": "find planning notes",
            "tool_choice": {"name": "query", "arguments": {"query": "find planning notes"}},
            "certainty": "certain"}


def _patch_registry_query(monkeypatch, function):
    from dataclasses import replace
    from core.operations import registry
    operations = dict(registry.OPERATIONS)
    operations["query"] = replace(operations["query"], function=function)
    monkeypatch.setattr(registry, "OPERATIONS", operations)


def test_search_route_is_governed_replayable_and_isolated(monkeypatch, firestore_fake):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    calls = []
    long_source = "Павел обсудил план 🌿 " * 3000
    _patch_registry_query(monkeypatch,
                        lambda **kwargs: calls.append(kwargs) or (
                            [{"doc_id": "doc-search", "title": "Planning",
                              "dialog_date": "2026-10-01", "folder_path": ["Research"],
                              "segments": [{"range": [4, 4], "dialog": [{
                                  "speaker": "Pavel", "text": long_source,
                                  "order_index": 4,
                              }]}]}],
                            {"total_matches": 1, "returned_segments": 1,
                             "truncated": False, "query": kwargs["query"]},
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
    model_contents = next(contents for mode, contents, _tools in model.calls
                          if mode == "tool_loop")
    application_data = next(part["text"] for turn in model_contents
                            if turn.get("role") == "user"
                            for part in turn.get("parts", [])
                            if isinstance(part, Mapping)
                            and "Untrusted application tool result data follows" in part.get("text", ""))
    assert '"query":"find planning notes"' in application_data
    assert long_source in application_data
    turns = repository.list_turns(
        context_id, repository.get_context(context_id)["active_conversation_id"],
    )
    tool_turn = next(turn for turn in turns if turn.get("role") == "tool"
                     and isinstance(turn.get("tool_correlation"), Mapping)
                     and turn["tool_correlation"].get("name") == "query")
    reference = tool_turn["parts"][0]["reference"]
    assert reference["payload"].get("_input_ref")
    assert tool_turn["tool_correlation"]["arguments"] == {"query": "find planning notes"}
    assert len(bot.sent) == sent
    assert bot.sent[0][1] == "A safe answer."
    assert calls[0]["query"] == "find planning notes"
    assert not any(value.get("kind") == "document"
                   for path, value in firestore_fake.documents.items()
                   if path.startswith(f"telegram_sessions/{context_id}/conversations/"))
    assert any(value.get("kind") == "search"
               for path, value in firestore_fake.documents.items()
               if path.startswith(f"telegram_sessions/{context_id}/conversations/"))


def test_search_provider_failures_use_three_stage_budget_and_safe_error(
    monkeypatch, firestore_fake, caplog,
):
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.runtime import _CommandExecution
    caplog.set_level("WARNING")
    settlements = []
    settle = _CommandExecution._settle_governed_failure

    async def count_settlement(self):
        settlements.append(self.operation_id)
        return await settle(self)

    monkeypatch.setattr(_CommandExecution, "_settle_governed_failure", count_settlement)
    calls = []
    def fail_query(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private search provider details")
    _patch_registry_query(monkeypatch, fail_query)
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
    assert settlements == [operation["operation_id"]] * 3
    assert "operation_id=op-runtime" in bot.sent[-1][1]
    assert "private search provider details" not in bot.sent[-1][1]
    assert "private search provider details" not in caplog.text


def test_route_failure_diagnostic_contains_only_bounded_safe_fields(
    firestore_fake, caplog,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    class FailingModel:
        def generate(self, **_kwargs):
            raise RuntimeError("synthetic provider body with raw request and source text")

    caplog.set_level("WARNING")
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    bot = _Bot()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=FailingModel(),
    )

    state = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert state.value == "failed"
    record = next(record for record in caplog.records
                  if record.message == "telegram route selection failed")
    assert record.operation_id == operation["operation_id"]
    assert record.phase == "routing"
    assert record.code == "route_selection_failed"
    assert record.exception_class == "RuntimeError"
    assert "synthetic provider body" not in caplog.text
    assert "raw request" not in caplog.text
    assert "private request" not in caplog.text


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
        ) and "telegram-tool-00" in path
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
        ) and "telegram-tool-00" in path
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


def test_runtime_store_publishes_search_selection_idempotently_and_preserves_context(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    number = repository.bind_document(
        context_id, "doc-existing", "conversation-existing", metadata={},
        now=now, update_last_doc=False,
    )
    repository.ensure_conversation(
        context_id, "conversation-existing", kind="document", doc_id="doc-existing",
        dialog_number=number, now=now,
    )
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.update({
        "active_conversation_id": "conversation-existing",
        "selected_dialog_number": number,
        "last_doc_id": "doc-existing",
        "active_draft_id": "draft-collecting",
        "selection_sequence": 4,
    })
    operations = context_ref.collection("operations")
    operations.document(operation["operation_id"]).update({
        "route_control": {
            "status": "ok", "kind": "search", "search_session_id": "search-current",
        },
    })
    repository.ensure_conversation(
        context_id, "search-current", kind="search", now=now,
    )
    store = TelegramRuntimeStore(repository, context_id=context_id)
    kwargs = {
        "conversation_id": "search-current", "owner": lease.owner,
        "fence": lease.fence, "now": now,
    }

    assert store.publish_search_selection(operation["operation_id"], **kwargs)
    first_context = repository.get_context(context_id)
    assert first_context["active_conversation_id"] == "search-current"
    assert first_context["selected_dialog_number"] is None
    assert first_context["selection_sequence"] == 5
    assert first_context["last_doc_id"] == "doc-existing"
    assert first_context["active_draft_id"] == "draft-collecting"

    assert store.publish_search_selection(operation["operation_id"], **kwargs)
    second_context = repository.get_context(context_id)
    assert second_context["selection_sequence"] == 5
    binding = context_ref.collection("document_bindings").stream()
    assert len(binding) == 1
    assert binding[0].to_dict()["doc_id"] == "doc-existing"


@pytest.mark.parametrize(("route_kind", "route_session_id", "conversation_kind", "fence", "binding_error"), [
    ("search", "search-other", "search", 1, True),
    ("substantive", "search-current", "search", 1, True),
    ("search", "search-current", "document", 1, True),
    ("search", "search-current", "search", 2, False),
])
def test_runtime_store_refuses_stale_or_mismatched_search_selection(
    firestore_fake, route_kind, route_session_id, conversation_kind, fence, binding_error,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.collection("operations").document(operation["operation_id"]).update({
        "route_control": {
            "status": "ok", "kind": route_kind, "search_session_id": route_session_id,
        },
    })
    context_ref.collection("conversations").document("search-current").set({
        "conversation_id": "search-current", "kind": conversation_kind,
    })
    store = TelegramRuntimeStore(repository, context_id=context_id)

    if binding_error:
        with pytest.raises(ValueError, match="search selection binding is invalid"):
            store.publish_search_selection(
                operation["operation_id"], conversation_id="search-current",
                owner=lease.owner, fence=fence, now=now,
            )
    else:
        assert not store.publish_search_selection(
            operation["operation_id"], conversation_id="search-current",
            owner=lease.owner, fence=fence, now=now,
        )
    unchanged = repository.get_context(context_id)
    assert unchanged["active_conversation_id"] is None
    assert unchanged["selected_dialog_number"] is None
    assert unchanged["selection_sequence"] == 0


@pytest.mark.parametrize(("binding_error", "expected"), [
    (False, "running"), (True, "failed"),
])
def test_search_selection_guard_retries_stale_lease_but_settles_binding_error(binding_error, expected):
    from services.connectors.telegram.runtime.runtime import _CommandExecution
    from services.connectors.telegram.runtime.runtime import QueueOperationState

    replies = []

    class Store:
        def create_search_session(self, **_kwargs):
            return "search-current"

        def publish_search_selection(self, *_args, **_kwargs):
            if binding_error:
                raise ValueError("search selection binding is invalid")
            return False

    async def finish_reply(message, **kwargs):
        replies.append((message, kwargs))
        return QueueOperationState.FAILED

    execution = object.__new__(_CommandExecution)
    execution.context_id = "42_default"
    execution.operation_id = "op-search"
    execution.owner = "worker"
    execution.fence = 1
    execution.command = SimpleNamespace(text="synthetic search")
    execution.replay_request = None
    execution.pinned = {"status": "ok", "kind": "search",
                        "search_session_id": "search-current", "result_refs": []}
    execution.search_session_id = "search-current"
    execution.search_refs = []
    execution.store = Store()
    execution.search = SimpleNamespace(failure=None)
    execution.runtime = SimpleNamespace(
        clock=lambda: datetime.now(timezone.utc),
        repository=SimpleNamespace(list_turns=lambda *_args: pytest.fail("selection failure continued")),
    )
    execution.finish_reply = finish_reply

    result = asyncio.run(execution._handle_global_search())

    assert result.value == expected
    if binding_error:
        assert len(replies) == 1
        assert replies[0][1] == {"kind": "error", "failure_final": True}
        assert "search session" in replies[0][0]
    else:
        assert replies == []


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
    async def interrupted(self, body, *, kind="final", presentation="plain"):
        if kind == "final" and crashes["count"] == 0:
            crashes["count"] += 1
            raise RuntimeError("simulated interruption after model checkpoint")
        return await deliver(self, body, kind=kind, presentation=presentation)
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
    assert [row[1] for row in bot.sent] == ["The decision was to proceed."]
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


@pytest.mark.parametrize("answer_failure", ["blank", "exception"])
def test_selected_document_blank_or_failed_answer_is_a_safe_failed_reply(
    firestore_fake, monkeypatch, answer_failure,
):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)

    class FailingDocumentModel(_RuntimeModel):
        def generate(self, **kwargs):
            if kwargs.get("mode") == "tool_loop" and answer_failure == "exception":
                self.calls.append(("tool_loop", kwargs.get("contents"), kwargs.get("tools")))
                raise RuntimeError("synthetic private tool-loop failure")
            return super().generate(**kwargs)

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
    answer = "   " if answer_failure == "blank" else "unused"
    bot, model = _Bot(), FailingDocumentModel(route, answer)
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-synthetic-root")

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "failed"
    assert [mode for mode, _contents, _tools in model.calls] == ["routing", "tool_loop"]
    assert [row[1] for row in bot.sent] == [
        "An error occurred while processing your request. Please try again.",
    ]
    conversation_id = repository.get_route_control(
        context_id, operation["operation_id"],
    )["conversation_id"]
    turns = repository.list_turns(context_id, conversation_id)
    assert [turn["role"] for turn in turns] == ["user"]


def test_document_answer_persistence_interruption_replays_checkpointed_answer(
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
    bot, model = _Bot(), _RuntimeModel(route, "The saved answer is available.")
    runtime = TelegramCommandRuntime(repository=repository, save_handler=object(),
                                     client=bot, model=model,
                                     root_folder_id="pytest-synthetic-root")
    append_turn = repository.append_turn
    injected = {"fail": True}

    def fail_first_model_turn(context_id, conversation_id, *, role, **kwargs):
        if role == "model" and injected["fail"]:
            injected["fail"] = False
            raise RuntimeError("synthetic answer persistence interruption")
        return append_turn(context_id, conversation_id, role=role, **kwargs)

    repository.append_turn = fail_first_model_turn
    first = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)
    assert first.value == "running"
    assert [row[1] for row in bot.sent] == []

    second = asyncio.run(runtime(context_id, operation, lease, command=command),
                          loop_factory=_selector_loop)

    assert second.value == "completed"
    assert [row[1] for row in bot.sent] == ["The saved answer is available."]
    assert [mode for mode, _contents, _tools in model.calls].count("tool_loop") == 1
    conversation_id = repository.get_route_control(
        context_id, operation["operation_id"],
    )["conversation_id"]
    assert [turn["role"] for turn in repository.list_turns(
        context_id, conversation_id,
    )] == ["user", "model"]


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
    assert bot.sent[0][1] == "The document summary."
    assert calls[0][0] == "get_transcript"
    assert calls[0][1]["doc_id"] == "doc-1"
    assert calls[0][2]["context"]["target_doc_id"] == "doc-1"
    assert len(bot.sent) == 1
    assert repository.get_context(context_id)["last_doc_id"] == "doc-b"


def test_runtime_pure_switch_acknowledges_without_appending_turns(firestore_fake, monkeypatch):
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
    assert [row[1] for row in bot.sent] == ["Selected dialog #1."]
    assert [mode for mode, _contents, _tools in model.calls] == ["routing"]
    context = repository.get_context(context_id)
    assert context["selected_dialog_number"] == 1
    assert repository.list_turns(context_id, context["active_conversation_id"]) == []


def _same_target_switch_runtime(client, *, operation_id="op-same-switch"):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(
        client, operation_id=operation_id,
    )
    client.collection("transcript_index").document("doc-same").set({
        "source_file": "Same target", "parent_id": "root-same",
    })
    client.collection("folders").document("root-same").set({
        "folder_id": "root-same", "name": "Root", "parent_id": "pytest-switch-root",
    })
    now = datetime.now(timezone.utc)
    number = repository.bind_document(
        context_id, "doc-same", "conversation-same",
        metadata={"title": "Same target", "folder_path": []}, now=now,
    )
    repository.ensure_conversation(
        context_id, "conversation-same", kind="document", doc_id="doc-same",
        dialog_number=number, now=now,
    )
    repository.client.collection("telegram_sessions").document(context_id).update({
        "active_conversation_id": "conversation-same",
        "selected_dialog_number": number,
    })
    route = {
        "kind": "pure_switch", "target_ref": "doc-same",
        "target_operation_id": None, "dialog_number": number,
        "result_refs": [], "substantive_text": None,
        "tool_choice": None, "certainty": "certain",
    }
    bot, model = _Bot(), _RuntimeModel(route)
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=model, root_folder_id="pytest-switch-root",
    )
    return repository, context_id, operation, lease, command, runtime, bot, model, number


def test_same_target_pure_switch_publishes_under_fence_and_replay_sends_no_extra_ack(
    firestore_fake, monkeypatch,
):
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    values = _same_target_switch_runtime(firestore_fake.client_class())
    repository, context_id, operation, lease, command, runtime, bot, model, number = values
    sequence_before = repository.get_context(context_id)["selection_sequence"]

    first = asyncio.run(runtime(context_id, operation, lease, command=command),
                        loop_factory=_selector_loop)
    second = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert first.value == second.value == "completed", (
        first, second, repository.get_route_control(context_id, operation["operation_id"]),
        bot.sent, model.calls,
    )
    assert [row[1] for row in bot.sent] == [f"Selected dialog #{number}."]
    assert [mode for mode, _contents, _tools in model.calls] == ["routing"]
    assert repository.get_context(context_id)["active_conversation_id"] == "conversation-same"
    assert repository.get_context(context_id)["selection_sequence"] == sequence_before


@pytest.mark.parametrize("published", [False, None])
def test_same_target_pure_switch_does_not_ack_unconfirmed_fenced_publication(
    firestore_fake, monkeypatch, published,
):
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    monkeypatch.setattr("services.connectors.telegram.agent._call_sync_or_async",
                        _polling_offload)
    monkeypatch.setattr(TelegramRuntimeStore, "publish_selection",
                        lambda *_args, **_kwargs: published)
    values = _same_target_switch_runtime(firestore_fake.client_class())
    repository, context_id, operation, lease, command, runtime, bot, model, _number = values

    result = asyncio.run(runtime(context_id, operation, lease, command=command),
                         loop_factory=_selector_loop)

    assert result.value == "running", (
        result, repository.get_route_control(context_id, operation["operation_id"]),
        bot.sent, model.calls,
    )
    assert bot.sent == []
    assert [mode for mode, _contents, _tools in model.calls] == ["routing"]
    assert repository.get_context(context_id)["active_conversation_id"] == "conversation-same"


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
    assert bot.sent[0][1] == "An error occurred while processing your request. Please try again."
    assert not any(row[1] == "Starting to process dialog #1." for row in bot.sent)
    context = repository.get_context(context_id)
    assert context["last_doc_id"] is None and context["active_conversation_id"] is None


def test_search_executor_rejects_filter_widening_before_provider_io():
    from services.connectors.telegram.runtime.runtime import _SearchOnlyExecutor

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return ([], {"total_matches": 0})

    executor = Executor()
    guarded = _SearchOnlyExecutor(
        executor,
        initial_filters={"query": "launch", "folder_path": ["Client"],
                         "date_from": "2026-01-01", "date_to": "2026-12-31"},
    )
    with pytest.raises(ValueError, match="removed a selected filter"):
        guarded.execute(
            name="query", arguments={"query": "launch", "folder_path": ["Client"],
                                      "date_to": "2026-12-31"},
            call_ordinal=1,
        )
    assert executor.calls == []
    result = guarded.execute(
        name="query", arguments={"query": "refined launch", "keywords": "new terms",
                                  "folder_path": ["Client", "Research"],
                                  "date_from": "2026-02-01", "date_to": "2026-11-30"},
        call_ordinal=2,
    )
    assert result["segments"] == []
    assert executor.calls == [{
        "name": "query", "arguments": {
            "query": "refined launch", "keywords": "new terms",
            "folder_path": ["Client", "Research"],
            "date_from": "2026-02-01", "date_to": "2026-11-30",
        }, "call_ordinal": 2,
    }]


def test_search_executor_transcript_read_uses_confirmed_doc_without_query_filters():
    from services.connectors.telegram.runtime.runtime import _SearchOnlyExecutor

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return ([], {"truncated": False})

    executor = Executor()
    guarded = _SearchOnlyExecutor(
        executor, allowed_doc_ids=lambda: {"known-doc"},
        initial_filters={"query": "Pavel", "keywords": "speakers",
                         "folder_path": ["Client"], "date_from": "2026-01-01"},
    )
    assert guarded.execute(
        name="get_transcript", arguments={"doc_id": "known-doc"}, call_ordinal=1,
    )["segments"] == []
    assert executor.calls[0]["name"] == "get_transcript"
    with pytest.raises(PermissionError, match="read-only"):
        guarded.execute(name="create_folder", arguments={"name": "Forbidden"}, call_ordinal=2)
    assert len(executor.calls) == 1


def test_search_executor_allows_only_server_scoped_folder_transcript_read():
    from services.connectors.telegram.runtime.runtime import _SearchOnlyExecutor

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return ([], {"truncated": False})

    args = {"folder_path": ["Synthetic Client"], "date_from": "2026-01-01",
            "date_to": "2026-12-31", "limit": 5}
    executor = Executor()
    allowed = _SearchOnlyExecutor(
        executor, initial_filters={"query": "synthetic question",
                                   "folder_path": ["Synthetic Client"],
                                   "date_from": "2026-01-01", "date_to": "2026-12-31"},
        allow_scoped_transcript=True,
    )
    assert allowed.execute(name="get_transcript", arguments=args, call_ordinal=1)["segments"] == []
    assert executor.calls[0]["arguments"] == args

    ordinary = _SearchOnlyExecutor(executor)
    with pytest.raises(PermissionError, match="confirmed result"):
        ordinary.execute(name="get_transcript", arguments=args, call_ordinal=2)
    assert len(executor.calls) == 1


def test_registry_executor_requires_server_scoped_flag_for_folder_transcript(monkeypatch):
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

    calls = []
    monkeypatch.setattr(
        "services.connectors.telegram.runtime.tools.invoke_tool",
        lambda name, arguments, **kwargs: calls.append((name, arguments, kwargs))
        or {"segments": [], "metadata": {"truncated": False}},
    )
    arguments = {"folder_path": ["Synthetic Client"], "limit": 3}
    ordinary = TelegramRegistryExecutor(
        client=None, context={"last_doc_id": None}, save_handler=object(),
        execution_context={"target_doc_id": None},
    )
    ordinary._assert_queue_lease_live = lambda: None
    with pytest.raises(ValueError, match="pinned document"):
        ordinary.execute(name="get_transcript", arguments=arguments, call_ordinal=0)
    assert calls == []

    scoped = TelegramRegistryExecutor(
        client=None, context={"last_doc_id": None}, save_handler=object(),
        execution_context={"target_doc_id": None, "allow_scoped_transcript": True},
    )
    scoped._assert_queue_lease_live = lambda: None
    scoped.execute(name="get_transcript", arguments=arguments, call_ordinal=1)
    assert calls[0][0:2] == ("get_transcript", arguments)
    assert calls[0][2]["context"]["last_doc_id"] is None


@pytest.mark.parametrize(("arguments", "confirmed", "accepted"), [
    ({"folder_path": ["Synthetic Client"], "date_from": "2026-01-01",
      "limit": 3}, set(), True),
    ({"doc_id": "synthetic-confirmed-doc"}, {"synthetic-confirmed-doc"}, True),
    ({"doc_id": "synthetic-unconfirmed-doc"}, {"synthetic-confirmed-doc"}, False),
    ({"limit": 3}, set(), True),
])
def test_search_continuation_accepts_scoped_folder_read_and_keeps_doc_confirmation(
    arguments, confirmed, accepted,
):
    from services.connectors.telegram.runtime.runtime import _CommandExecution

    execution = object.__new__(_CommandExecution)
    execution.decision = {
        "kind": "substantive", "target_ref": None,
        "target_operation_id": None, "dialog_number": None,
        "tool_choice": {"name": "get_transcript", "arguments": arguments},
    }
    execution.route_conversation = SimpleNamespace(
        kind=SimpleNamespace(value="search"), search_filters=(
            {} if arguments == {"limit": 3}
            else {"folder_path": ["Synthetic Client"]}
        ),
    )
    execution.chat_conversation_id = "synthetic-search-session"
    execution.pinned = None
    execution._search_allowed_doc_ids = lambda: confirmed

    asyncio.run(execution._prepare_search())

    assert execution.search_continuation is accepted
    if accepted:
        assert execution.search_session_id == "synthetic-search-session"
        assert execution.decision is not None
    else:
        assert execution.search_session_id == "synthetic-search-session"
        assert execution.decision is None


def test_search_confirmations_follow_current_query_and_verified_exchange_only():
    from services.connectors.telegram.runtime.search_agent import SearchAgentMixin

    old_args = {"query": "synthetic old", "folder_path": ["Old Client"]}
    new_args = {"query": "synthetic new", "folder_path": ["New Client"]}
    prior_turn = {
        "role": "tool", "operation_id": "synthetic-prior-op",
        "tool_correlation": {"name": "query", "arguments": old_args},
        "source_refs": [{"doc_id": "synthetic-unverified-doc"}],
        "parts": [{"type": "search_exchange_ref", "reference": {"version": 1},
                   "call_id": "synthetic-prior-call", "ordinal": 2,
                   "checkpoint_conversation_id": "synthetic-prior-checkpoint"}],
    }
    value = object.__new__(SearchAgentMixin)
    value.context_id = "synthetic-context"
    value.operation_id = "synthetic-current-op"
    value.global_conversation_id = "synthetic-current-checkpoint"
    value.search_session_id = "synthetic-search-session"
    value.route_conversation = SimpleNamespace(
        kind=SimpleNamespace(value="search"), search_filters=old_args,
    )
    checkpoint = {
        "application_route": {"name": "query", "arguments": new_args},
        "tool_calls": [{"ordinal": 0, "name": "query", "arguments": new_args}],
        "tool_results": {"0": {"segments": [{"doc_id": "synthetic-new-doc"}]}},
    }
    value.ledger = SimpleNamespace(load_turn=lambda *_: checkpoint)
    value.runtime = SimpleNamespace(
        repository=SimpleNamespace(list_turns=lambda *_: [prior_turn]),
    )
    value._load_search_exchange = lambda *_a, **_k: {
        "call": {"name": "query", "arguments": old_args},
        "result": {"segments": [{"doc_id": "synthetic-old-doc"}]},
    }

    assert value._search_allowed_doc_ids() == {"synthetic-new-doc"}

    checkpoint.clear()
    checkpoint.update({
        "tool_calls": [{"ordinal": 0, "name": "list_recent_transcripts",
                        "arguments": {"folder_path": ["Old Client"]}}],
        "tool_results": {"0": {"transcripts": [
            {"doc_id": "synthetic-listed-doc", "parent_id": "synthetic-folder"},
        ], "total": 1}},
    })
    value.route_conversation.search_filters = {"folder_path": ["Old Client"]}
    value._load_search_exchange = lambda *_a, **_k: {
        "call": {"name": "query", "arguments": old_args},
        "result": {"segments": [{"doc_id": "synthetic-confirmed-doc"}]},
    }
    assert value._search_allowed_doc_ids() == {
        "synthetic-listed-doc", "synthetic-confirmed-doc",
    }


@pytest.mark.parametrize("protocol", [None, "agent-v1"])
def test_legacy_search_card_confirmation_requires_bound_operation(firestore_fake, protocol):
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.search_agent import SearchAgentMixin

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, source_operation = "synthetic-context", "synthetic-legacy-operation"
    session_id = "search-" + hashlib.sha256(
        f"{context_id}\0{source_operation}".encode("utf-8")
    ).hexdigest()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    filters = {"query": "synthetic discovery", "folder_path": None,
               "date_from": None, "date_to": None}
    repository.ensure_conversation(context_id, session_id, kind="search",
                                   search_filters=filters, now=now)
    route_control = {"kind": "search", "status": "ok",
                     "result_refs": ["synthetic-legacy-confirmed-doc"],
                     "search_session_id": session_id}
    if protocol is not None:
        route_control["search_protocol"] = protocol
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(source_operation).set({
            "operation_id": source_operation, "context_id": context_id,
            "state": "completed",
            "route_control": route_control,
        })
    repository.append_turn(
        context_id, session_id, turn_id="synthetic-legacy-card",
        operation_id=source_operation, role="tool", now=now,
        parts=[{"type": "search_result", "status": "ok", "results": [
            {"doc_id": "synthetic-legacy-confirmed-doc"},
        ], "metadata": {"truncated": False}, "error": None}],
    )
    value = object.__new__(SearchAgentMixin)
    value.context_id = context_id
    value.operation_id = "synthetic-current-operation"
    value.global_conversation_id = "synthetic-private-checkpoint"
    value.search_session_id = session_id
    value.route_conversation = repository.get_conversation(context_id, session_id)
    value.runtime = SimpleNamespace(repository=repository)
    value.ledger = SimpleNamespace(load_turn=lambda *_: {})

    expected = set() if protocol == "agent-v1" else {"synthetic-legacy-confirmed-doc"}
    assert value._search_allowed_doc_ids() == expected

    if protocol is None:
        client.collection("telegram_sessions").document(context_id).collection(
            "operations").document(source_operation).update({"context_id": "other-context"})
        assert value._search_allowed_doc_ids() == set()


@pytest.mark.parametrize("name,args", [
    ("get_transcript", {"folder_path": ["Synthetic Client"], "limit": 2}),
    ("list_recent_transcripts", {"date_from": "2026-02-01", "limit": 2}),
])
def test_search_content_and_listing_reads_cannot_remove_accepted_bounds(name, args):
    from services.connectors.telegram.runtime.runtime import _SearchOnlyExecutor

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return ([], {}) if kwargs["name"] == "get_transcript" else {"transcripts": []}

    executor = Executor()
    guarded = _SearchOnlyExecutor(
        executor, allowed_doc_ids=lambda: {"synthetic-doc"},
        initial_filters={"query": "synthetic question",
                         "folder_path": ["Synthetic Client"],
                         "date_from": "2026-01-01", "date_to": "2026-12-31"},
        allow_scoped_transcript=True,
    )
    with pytest.raises(ValueError, match="removed a selected filter"):
        guarded.execute(name=name, arguments=args, call_ordinal=1)
    assert executor.calls == []


def test_search_transcript_confirmation_recognizes_listing_result_shape():
    from services.connectors.telegram.runtime.search_agent import (
        _arguments_preserve_search_bounds, _confirmed_result_documents,
    )

    result = {"transcripts": [{"doc_id": "synthetic-listed-doc",
                               "parent_id": "synthetic-folder",
                               "dialog_date": "2026-04-03"}],
              "total": 1, "returned": 1, "offset": 0, "limit": 20,
              "has_more": False}
    assert _confirmed_result_documents(result)[0]["doc_id"] == "synthetic-listed-doc"
    bounds = {"folder_path": ["Synthetic Client"], "date_from": "2026-01-01"}
    assert _arguments_preserve_search_bounds(
        bounds, {"folder_path": ["Synthetic Client", "Subfolder"],
                 "date_from": "2026-03-01"},
    )
    assert not _arguments_preserve_search_bounds(
        bounds, {"date_from": "2026-03-01"},
    )


def test_search_history_persists_confirmed_prefix_and_skips_pending_call():
    from services.connectors.telegram.runtime.search_agent import SearchAgentMixin

    class Ledger:
        def load_turn(self, _conversation_id, _operation_id):
            return {"tool_calls": [
                {"id": "synthetic-call-0", "name": "list_recent_transcripts",
                 "arguments": {"folder_path": ["Synthetic Client"]}, "ordinal": 0},
                {"id": "synthetic-call-1", "name": "get_transcript",
                 "arguments": {"doc_id": "synthetic-listed-doc"}, "ordinal": 1},
            ], "tool_results": {"0": {"transcripts": [{
                "doc_id": "synthetic-listed-doc", "parent_id": "synthetic-folder",
            }], "total": 1}}}

        def save_search_exchange(self, _conversation_id, **kwargs):
            return f"synthetic-exchange-{kwargs['call']['ordinal']}"

    class Repository:
        def __init__(self):
            self.rows = []

        def append_turn(self, *args, **kwargs):
            self.rows.append(kwargs)

    harness = object.__new__(SearchAgentMixin)
    harness.context_id = "synthetic-context"
    harness.operation_id = "synthetic-operation"
    harness.global_conversation_id = "synthetic-checkpoint"
    harness.ledger = Ledger()
    harness.runtime = SimpleNamespace(
        repository=Repository(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    harness.owner, harness.fence = "synthetic-owner", 1

    harness._persist_search_tool_history("synthetic-search-session")

    assert len(harness.runtime.repository.rows) == 1
    row = harness.runtime.repository.rows[0]
    assert row["tool_correlation"]["name"] == "list_recent_transcripts"
    assert row["source_refs"] == [{"doc_id": "synthetic-listed-doc"}]


def test_listing_confirmation_hydrates_from_real_ledger_after_execution_restart(
    firestore_fake,
):
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.runtime import _CommandExecution

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    session_id = "search-" + hashlib.sha256(
        f"{context_id}\0{operation['operation_id']}".encode("utf-8")
    ).hexdigest()
    private_id = "telegram-global-" + hashlib.sha256(
        f"{context_id}\0{operation['operation_id']}".encode("utf-8")
    ).hexdigest()
    now = datetime.now(timezone.utc)
    repository.ensure_conversation(
        context_id, session_id, kind="search",
        search_filters={"query": "synthetic listing"}, now=now,
    )
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({
            "route_control": {"kind": "search", "status": "ok",
                              "search_protocol": "agent-v1",
                              "search_session_id": session_id},
        })
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    call = {"id": "synthetic-list-call", "name": "list_recent_transcripts",
            "arguments": {"folder_path": ["Synthetic Client"]},
            "ordinal": 0, "round": 0}
    result = {"transcripts": [{
        "doc_id": "synthetic-listed-doc", "parent_id": "synthetic-client-folder",
        "title": "Synthetic meeting", "dialog_date": "2026-04-03",
        "status": "synced",
    }], "total": 1, "returned": 1, "offset": 0, "limit": 20,
        "has_more": False}
    ledger.save_turn(private_id, operation["operation_id"],
                     model_response={"parts": []}, tool_calls=[call])
    ledger.save_tool_result(private_id, operation["operation_id"], 0, result)
    reference = ledger.save_search_exchange(
        session_id, checkpoint_conversation_id=private_id,
        call=call, canonical_call_id=call["id"], result=result,
    )
    repository.append_turn(
        context_id, session_id, turn_id="synthetic-list-exchange",
        operation_id=operation["operation_id"], role="tool", now=now,
        parts=[{"type": "search_exchange_ref", "reference": reference,
                "call_id": call["id"], "ordinal": 0,
                "checkpoint_conversation_id": private_id}],
        tool_correlation={"name": call["name"], "arguments": call["arguments"]},
        source_refs=[{"doc_id": "synthetic-listed-doc"}],
    )

    restarted = object.__new__(_CommandExecution)
    restarted.runtime = SimpleNamespace(repository=repository, clock=lambda: now)
    restarted.context_id = context_id
    restarted.operation_id = operation["operation_id"]
    restarted.global_conversation_id = private_id
    restarted.search_session_id = session_id
    restarted.route_conversation = repository.get_conversation(context_id, session_id)
    restarted.owner, restarted.fence = lease.owner, lease.fence
    restarted.ledger = ledger

    assert restarted._search_allowed_doc_ids() == {"synthetic-listed-doc"}
    assert ledger.load_search_exchange(
        reference, conversation_id=session_id, operation_id=operation["operation_id"],
        checkpoint_conversation_id=private_id, call_id=call["id"], ordinal=0,
    )["result"]["transcripts"][0]["doc_id"] == "synthetic-listed-doc"


def test_search_query_stage_keeps_provider_billing_error_nonretryable(firestore_fake):
    from google.genai.errors import ClientError
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryBudgetExceeded, RetryGovernor
    from core.utils.retry import ProviderBoundaryError
    from services.connectors.telegram.runtime.runtime import (
        _GovernedExecutor, _SearchOnlyExecutor,
    )
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    error = ProviderBoundaryError("gemini", ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED",
    }}))

    class Executor:
        calls = 0

        def execute(self, **_kwargs):
            self.calls += 1
            raise error

    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(
            client, client.collection("telegram_sessions").document(context_id)),
        error_classifier=telegram_error_classifier,
    )
    inner = Executor()
    guarded = _SearchOnlyExecutor(_GovernedExecutor(
        inner, governor, operation["operation_id"], lease,
        lambda: now, lambda: True,
    ))
    with pytest.raises(ProviderBoundaryError):
        guarded.execute(name="query", arguments={"query": "synthetic query"}, call_ordinal=0)
    stage = governor.get_stage(operation["operation_id"], "telegram-tool-00")
    assert stage.failures == 1
    assert stage.last_error.retryable is False
    with pytest.raises(RetryBudgetExceeded):
        guarded.execute(name="query", arguments={"query": "synthetic query"}, call_ordinal=0)
    assert inner.calls == 1


def test_search_exchange_payload_is_bounded_verified_and_fenced(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    _repository = TelegramRepository(client=client)
    context_id, operation, lease, _command = _runtime_queue_item(client)
    now = datetime.now(timezone.utc)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    source = "speaker :: synthetic Unicode evidence 🌿 " * 25000
    call = {"id": "call-1", "name": "query", "arguments": {"query": "synthetic"},
            "ordinal": 0, "round": -1, "thought_signature": b"opaque-signature"}
    reference = ledger.save_search_exchange(
        "search-session", checkpoint_conversation_id="private-checkpoint",
        call=call, canonical_call_id="call-1",
        result={"segments": [{"text": source}], "metadata": {"truncated": False}},
    )
    exchange = ledger.load_search_exchange(
        reference, conversation_id="search-session", operation_id=operation["operation_id"],
        checkpoint_conversation_id="private-checkpoint", call_id="call-1", ordinal=0,
    )
    assert exchange["call"]["thought_signature"] == b"opaque-signature"
    assert exchange["result"]["segments"][0]["text"] == source
    assert all(len(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8"))
               < 1024 * 1024 for row in firestore_fake.documents.values())

    with pytest.raises(ValueError, match="binding"):
        ledger.load_search_exchange(
            {**reference, "call_id": "tampered"}, conversation_id="search-session",
            operation_id=operation["operation_id"],
            checkpoint_conversation_id="private-checkpoint", call_id="call-1", ordinal=0,
        )
    other_context = FirestoreAgentLedger(
        client, context_id="other-context", operation_id=operation["operation_id"],
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    with pytest.raises(ValueError, match="binding"):
        other_context.load_search_exchange(
            reference, conversation_id="search-session", operation_id=operation["operation_id"],
            checkpoint_conversation_id="private-checkpoint", call_id="call-1", ordinal=0,
        )
    with pytest.raises(ValueError, match="binding"):
        ledger.load_search_exchange(
            None, conversation_id="search-session", operation_id=operation["operation_id"],
            checkpoint_conversation_id="private-checkpoint", call_id="call-1", ordinal=0,
        )

    client.collection("telegram_sessions").document(context_id).update({"fence": 2})
    with pytest.raises(PermissionError, match="stale"):
        ledger.save_search_exchange(
            "search-session", checkpoint_conversation_id="private-checkpoint",
            call={**call, "ordinal": 1}, canonical_call_id="call-2",
            result={"segments": [{"text": source}]},
        )


@pytest.mark.parametrize("command_present", [True, False])
def test_trusted_legacy_search_pin_replays_canonical_cards_without_model(
    firestore_fake, command_present,
):
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime import TelegramCommandRuntime

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id, operation, lease, command = _runtime_queue_item(client)
    session_id = "search-" + hashlib.sha256(
        f"{context_id}\0{operation['operation_id']}".encode("utf-8")
    ).hexdigest()
    now = datetime.now(timezone.utc)
    repository.ensure_conversation(
        context_id, session_id, kind="search",
        search_filters={"query": command.text, "keywords": None,
                        "folder_path": None, "date_from": None, "date_to": None}, now=now,
    )
    repository.append_turn(
        context_id, session_id, turn_id="legacy-search-card",
        operation_id=operation["operation_id"], role="tool",
        parts=[{"type": "search_result", "status": "ok",
                "results": [{"doc_id": "doc-one", "title": "Known result"}],
                "metadata": {"truncated": False}, "error": None}], now=now,
    )
    client.collection("telegram_sessions").document(context_id).collection(
        "operations").document(operation["operation_id"]).update({
            "route_control": {"status": "ok", "kind": "search",
                              "result_refs": ["doc-one"],
                              "search_session_id": session_id},
        })
    model, bot = _RuntimeModel(_route()), _Bot()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=model, clock=lambda: now,
    )

    state = asyncio.run(
        runtime(context_id, operation, lease,
                command=command if command_present else None),
        loop_factory=_selector_loop,
    )

    assert state.value == "completed"
    assert model.calls == []
    assert bot.sent and "Known result" in bot.sent[0][1]
