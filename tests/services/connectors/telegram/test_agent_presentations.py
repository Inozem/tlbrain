"""Offline contracts for the private terminal search-presentation tool."""

from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.offline


SCOPE = {
    "context_id": "42017_default",
    "conversation_id": "search-conversation",
    "producing_operation_id": "producer-operation",
    "presentation_id": "presentation-1",
    "answer_turn_id": "answer-turn-1",
}


class MemoryLedger:
    """Small ledger fake that retains the agent's existing turn envelope."""

    def __init__(self):
        self.turns = {}

    def load_turn(self, conversation_id, delivery_id):
        return deepcopy(self.turns.get((conversation_id, delivery_id)))

    def save_turn(self, conversation_id, delivery_id, *, model_response, tool_calls,
                  application_route=None):
        previous = self.turns.get((conversation_id, delivery_id), {})
        self.turns[(conversation_id, delivery_id)] = {
            "model_response": deepcopy(model_response),
            "tool_calls": deepcopy(tool_calls),
            "tool_results": deepcopy(previous.get("tool_results", {})),
        }

    def save_tool_result(self, conversation_id, delivery_id, ordinal, result):
        self.turns[(conversation_id, delivery_id)]["tool_results"][ordinal] = deepcopy(result)


class SequenceModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, *, contents, mode, tools=None):
        self.requests.append({"contents": deepcopy(contents), "mode": mode,
                              "tools": deepcopy(tools)})
        if not self.responses:
            raise AssertionError("unexpected model generation")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Executor:
    def __init__(self):
        self.calls = []

    def execute(self, *, name, arguments, call_ordinal):
        self.calls.append((name, deepcopy(arguments), call_ordinal))
        return {"ok": True}


def _call(name, args, *, call_id="native-1", signature="opaque-signature"):
    return SimpleNamespace(id=call_id, name=name, args=deepcopy(args),
                           thought_signature=signature)


def _response(*parts, text="provider text"):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
        usage_metadata={"prompt_token_count": 4, "candidates_token_count": 2},
        text=text,
    )


def _presented_run(agent, *, delivery_id="producer-operation", loader=None, **kwargs):
    options = {
        "conversation_id": "search-conversation", "delivery_id": delivery_id,
        "history": [{"role": "user", "parts": [{"text": "Summarize the result."}]}],
        "presentation_scope": dict(SCOPE),
        "presentation_sources": loader or (lambda: []), **kwargs,
    }
    # Before T020 lands, use the old public signature to expose the actual
    # missing terminal handling instead of stopping at an unknown-keyword error.
    parameters = inspect.signature(agent.run).parameters
    return agent.run(**{key: value for key, value in options.items()
                        if key in parameters or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters.values()
                        )})


def _saved_terminal_checkpoint():
    from services.connectors.telegram.agent import TelegramAgent

    ledger = MemoryLedger()
    result = _presented_run(TelegramAgent(
        model=SequenceModel(_response(_call("present_search_answer", {"blocks": [{
            "type": "paragraph", "text": "Saved exact body.", "source_refs": [],
        }]}))), ledger=ledger, tool_executor=Executor(),
    ))
    assert result["status"] == "ok"
    return ledger, result


def test_gemini_presented_mode_uses_medium_thinking_and_native_any_calling():
    from core.integrations.gemini.agent import GeminiAgent

    class Models:
        def __init__(self):
            self.request = None

        def generate_content(self, **kwargs):
            self.request = kwargs
            return object()

    models = Models()
    agent = GeminiAgent(client=SimpleNamespace(models=models))
    declaration = {"name": "present_search_answer", "description": "Finish the answer",
                   "parameters_json_schema": {
                       "type": "object", "properties": {"blocks": {"type": "array"}},
                       "required": ["blocks"], "additionalProperties": False,
                   }}
    response = agent.generate(contents=[], mode="tool_loop_presented", tools=[declaration])

    assert response is not None
    config = models.request["config"]
    assert config.thinking_config.thinking_level == "MEDIUM"
    assert config.tool_config.function_calling_config.mode == "ANY"
    assert config.tools[0].function_declarations[0].name == "present_search_answer"


def test_native_terminal_declaration_and_opaque_signature_reach_real_sdk_wire(monkeypatch):
    import json
    from tests.services.connectors.telegram.test_agent import _real_sdk_client
    from services.connectors.telegram.agent import TelegramAgent
    from core.integrations.gemini.agent import GeminiAgent

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return __import__("httpx").Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{
                "functionCall": {"id": "native/雪-terminal", "name": "present_search_answer",
                                 "args": {"blocks": [{
                                     "type": "paragraph", "text": "Wire answer.",
                                     "source_refs": [],
                                 }]}},
                "thoughtSignature": "Y2hhbmdlLWZha2Utc2lnbmF0dXJl",
            }],
        }}]})

    client, http_client = _real_sdk_client(monkeypatch, respond)
    try:
        ledger, executor = MemoryLedger(), Executor()
        result = _presented_run(TelegramAgent(
            model=GeminiAgent(client=client), ledger=ledger, tool_executor=executor,
        ))
        assert result["status"] == "ok" and result["answer"] == "Wire answer."
        config = requests[0]["generationConfig"]
        assert config["thinkingConfig"]["thinking_level"] == "MEDIUM"
        assert requests[0]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
        declarations = [declaration for tool in requests[0]["tools"]
                        for declaration in tool["functionDeclarations"]]
        terminal = next(item for item in declarations
                        if item["name"] == "present_search_answer")
        schema = terminal.get("parametersJsonSchema", terminal.get("parameters_json_schema"))
        assert schema["required"] == ["blocks"]
        block_schemas = schema["properties"]["blocks"]["items"]["oneOf"]
        selector = block_schemas[0]["properties"]["source_refs"]["items"]
        assert selector["required"] == ["doc_id"]
        assert selector["properties"]["doc_id"]["type"] == "string"
        assert selector["properties"]["ordinal"]["minimum"] == 0
        assert selector["properties"]["order_index"]["minimum"] == 0
        assert selector["properties"]["reference"]["type"] == "object"
        assert "enum" not in json.dumps(schema)
        checkpoint = ledger.load_turn("search-conversation", "producer-operation")
        assert "Y2hhbmdlLWZha2Utc2lnbmF0dXJl" in repr(
            checkpoint["model_response"]["rounds"][-1]["response"]
        )
        assert executor.calls == []
    finally:
        http_client.close()


def test_terminal_call_accepts_missing_native_id():
    from services.connectors.telegram.agent import TelegramAgent

    args = {"blocks": [{"type": "paragraph", "text": "ID-free result.", "source_refs": []}]}
    result = _presented_run(TelegramAgent(
        model=SequenceModel(_response(_call("present_search_answer", args, call_id=None))),
        ledger=MemoryLedger(), tool_executor=Executor(),
    ))
    assert result["status"] == "ok" and result["answer"] == "ID\\-free result."


def test_saved_terminal_replays_with_scope_but_without_source_loader():
    from services.connectors.telegram.agent import TelegramAgent

    ledger, first = _saved_terminal_checkpoint()
    replay = TelegramAgent(
        model=SequenceModel(AssertionError("saved replay called model")),
        ledger=ledger, tool_executor=Executor(),
    ).run(
        conversation_id="search-conversation", delivery_id="producer-operation", history=[],
        presentation_scope=dict(SCOPE),
        executor_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("saved replay constructed an executor")),
    )
    assert replay == first


def test_source_loader_operational_failure_propagates_without_terminal_checkpoint():
    from services.connectors.telegram.agent import TelegramAgent

    ledger = MemoryLedger()
    agent = TelegramAgent(
        model=SequenceModel(_response(_call("present_search_answer", {"blocks": [{
            "type": "paragraph", "text": "Cannot confirm.", "source_refs": [],
        }]}))), ledger=ledger, tool_executor=Executor(),
    )

    def broken_source_load():
        raise RuntimeError("synthetic evidence store outage")

    with pytest.raises(RuntimeError, match="synthetic evidence store outage"):
        _presented_run(agent, loader=broken_source_load)
    assert ledger.load_turn("search-conversation", "producer-operation") is None


def test_non_json_terminal_args_persist_fixed_failure_and_replay():
    from services.connectors.telegram.agent import TelegramAgent

    ledger, executor = MemoryLedger(), Executor()
    arguments = {"blocks": [], "unexpected": float("nan")}
    first = _presented_run(TelegramAgent(
        model=SequenceModel(_response(_call("present_search_answer", arguments))),
        ledger=ledger, tool_executor=executor,
    ), loader=lambda: [])
    assert first["status"] == "failed"
    replay = _presented_run(TelegramAgent(
        model=SequenceModel(AssertionError("non-json replay called model")),
        ledger=ledger, tool_executor=executor,
    ), loader=lambda: (_ for _ in ()).throw(AssertionError("non-json replay loaded sources")))
    assert replay == first and executor.calls == []


def test_fresh_presented_prose_is_processing_failure_and_not_replayable_as_success():
    from services.connectors.telegram.agent import TelegramAgent

    ledger, executor, loaded = MemoryLedger(), Executor(), []
    first = _presented_run(TelegramAgent(
        model=SequenceModel(_response(text="Unstructured final prose.")),
        ledger=ledger, tool_executor=executor,
    ), loader=lambda: loaded.append(True) or [])
    assert first["status"] == "failed"
    assert loaded == [] and executor.calls == []
    replay = _presented_run(TelegramAgent(
        model=SequenceModel(AssertionError("prose replay called model")),
        ledger=ledger, tool_executor=executor,
    ), loader=lambda: (_ for _ in ()).throw(AssertionError("prose replay loaded sources")))
    assert replay == first


def test_private_terminal_returns_checkpointed_rendered_body_and_replays_without_work():
    from services.connectors.telegram.agent import TelegramAgent

    blocks = [{"type": "paragraph", "text": "The search found no matching result.",
               "source_refs": []}]
    model = SequenceModel(_response(_call("present_search_answer", {"blocks": blocks})))
    ledger, executor = MemoryLedger(), Executor()
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    loaded = []

    first = _presented_run(agent, loader=lambda: loaded.append(True) or [])
    assert first["status"] == "ok"
    assert first["answer"] == "The search found no matching result."
    assert first["presentation_manifest"]["body"] == first["answer"]
    assert model.requests[0]["mode"] == "tool_loop_presented"
    catalog_names = {tool["name"] for tool in model.requests[0]["tools"]}
    assert "present_search_answer" in catalog_names
    from services.connectors.telegram.tools import get_tool_catalog
    assert "present_search_answer" not in {
        tool["name"] for tool in get_tool_catalog()["tools"]
    }
    assert executor.calls == [] and loaded == [True]

    def fail(*_args, **_kwargs):
        raise AssertionError("terminal replay must not call model, loader, or executor factory")

    replay_model = SequenceModel(AssertionError("model called on replay"))
    replay_agent = TelegramAgent(model=replay_model, ledger=ledger, tool_executor=executor)
    replay = _presented_run(replay_agent, loader=fail, executor_factory=fail)
    assert replay == first
    assert replay_model.requests == []
    assert executor.calls == []


def test_mixed_terminal_and_read_is_rejected_without_running_read_or_loader():
    from services.connectors.telegram.agent import TelegramAgent

    response = _response(
        _call("query", {"query": "project"}, call_id="read-1"),
        _call("present_search_answer", {"blocks": [{
            "type": "paragraph", "text": "Answer.", "source_refs": [],
        }]}, call_id="terminal-1"),
    )
    model, executor, ledger = SequenceModel(response), Executor(), MemoryLedger()
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    loaded = []

    result = _presented_run(agent, loader=lambda: loaded.append(True) or [])

    assert result["status"] == "failed"
    assert executor.calls == []
    assert loaded == []
    saved = ledger.load_turn("search-conversation", "producer-operation")
    assert saved is not None
    assert saved["model_response"]["_telegram_tool_loop_rounds"] == 1
    replay = _presented_run(
        TelegramAgent(model=SequenceModel(AssertionError("failure replay called model")),
                      ledger=ledger, tool_executor=executor),
        loader=lambda: (_ for _ in ()).throw(AssertionError("failure replay loaded evidence")),
    )
    assert replay["status"] == "failed"
    assert executor.calls == []


def test_duplicate_terminal_calls_are_rejected_before_source_loading():
    from services.connectors.telegram.agent import TelegramAgent

    args = {"blocks": [{"type": "paragraph", "text": "No findings.", "source_refs": []}]}
    response = _response(
        _call("present_search_answer", args, call_id="terminal-1"),
        _call("present_search_answer", args, call_id="terminal-2"),
    )
    ledger, executor, loaded = MemoryLedger(), Executor(), []
    result = _presented_run(
        TelegramAgent(model=SequenceModel(response), ledger=ledger, tool_executor=executor),
        loader=lambda: loaded.append(True) or [],
    )
    assert result["status"] == "failed"
    assert loaded == [] and executor.calls == []
    saved = ledger.load_turn("search-conversation", "producer-operation")
    assert saved["model_response"]["rounds"][-1]["terminal"]["status"] == "failed"


def test_presented_model_round_budget_includes_terminal_round():
    from services.connectors.telegram.agent import TelegramAgent

    model = SequenceModel(_response(_call("query", {"query": "synthetic lookup"})))
    executor, ledger, source_calls = Executor(), MemoryLedger(), []
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor, max_rounds=1)

    result = _presented_run(agent, loader=lambda: source_calls.append(True) or [])

    assert result["status"] == "failed"
    assert result["failure"] == "The tool loop exceeded the 1 model-round limit."
    assert len(model.requests) == 1
    assert len(executor.calls) == 1 and source_calls == []


def test_presentation_capability_is_invocation_local():
    from services.connectors.telegram.agent import TelegramAgent

    ledger = MemoryLedger()
    model = SequenceModel(
        _response(_call("present_search_answer", {"blocks": [{
            "type": "paragraph", "text": "Presented.", "source_refs": [],
        }]})),
        _response(text="Legacy answer exactly."),
    )
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=Executor())
    first = _presented_run(agent, loader=lambda: [])
    assert first["status"] == "ok"

    legacy = agent.run(
        conversation_id="another-conversation", delivery_id="legacy-operation",
        history=[{"role": "user", "parts": [{"text": "ordinary question"}]}],
    )

    assert legacy == {"status": "ok", "answer": "Legacy answer exactly.",
                      "usage": {"prompt_token_count": 4, "candidates_token_count": 2},
                      "tool_results": {}}
    assert model.requests[1]["mode"] == "tool_loop"
    assert "present_search_answer" not in {tool["name"] for tool in model.requests[1]["tools"]}

    # An old free-text final answer remains exactly the saved response even if
    # the same operation is later invoked with presentation capability.
    legacy_replay = agent.run(
        conversation_id="another-conversation", delivery_id="legacy-operation",
        history=[], presentation_scope={**SCOPE, "producing_operation_id": "legacy-operation"},
        presentation_sources=lambda: (_ for _ in ()).throw(
            AssertionError("legacy prose replay loaded new source evidence")),
    )
    assert legacy_replay == legacy


@pytest.mark.parametrize("terminal_args,load_sources", [
    ({"blocks": [{"type": "unsupported", "text": "x"}]}, True),
    ({"blocks": [{"type": "paragraph", "text": "A", "source_refs": [
        {"doc_id": "unverified-document"},
    ]}]}, True),
    ({"blocks": [{"type": "quote", "doc_id": "doc-a", "text": "not sourced",
                   "source_refs": [{"doc_id": "doc-a"}]}]}, True),
    ({"blocks": [{"type": "paragraph", "text": "A", "source_refs": []}],
      "extra": "not permitted"}, False),
])
def test_invalid_terminal_artifact_is_durable_and_replays_as_failure(terminal_args, load_sources):
    from services.connectors.telegram.agent import TelegramAgent

    model = SequenceModel(_response(_call("present_search_answer", terminal_args)))
    ledger, executor = MemoryLedger(), Executor()
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    source_calls = []

    def loader():
        source_calls.append(True)
        return []

    first = _presented_run(agent, loader=loader)
    assert first["status"] == "failed"
    assert source_calls == ([True] if load_sources else [])
    saved = ledger.load_turn("search-conversation", "producer-operation")
    round_row = saved["model_response"]["rounds"][-1]
    assert round_row["terminal"]["status"] == "failed"
    assert round_row["call_count"] == 0
    assert saved["tool_calls"] == []

    replay = _presented_run(
        TelegramAgent(model=SequenceModel(AssertionError("invalid replay called model")),
                      ledger=ledger, tool_executor=executor),
        loader=lambda: (_ for _ in ()).throw(AssertionError("invalid replay loaded evidence")),
    )
    assert replay == first
    assert executor.calls == []


@pytest.mark.parametrize("corruption", ["body_hash", "scope", "round_position"])
def test_terminal_replay_fails_closed_on_corrupt_marker(corruption):
    from services.connectors.telegram.agent import TelegramAgent

    ledger, executor = MemoryLedger(), Executor()
    scope = dict(SCOPE)
    first = _presented_run(TelegramAgent(
        model=SequenceModel(_response(_call("present_search_answer", {"blocks": [{
            "type": "paragraph", "text": "Saved exact body.", "source_refs": [],
        }]}))), ledger=ledger, tool_executor=executor,
    ))
    assert first["status"] == "ok"
    saved = ledger.turns[("search-conversation", "producer-operation")]
    terminal_round = saved["model_response"]["rounds"][-1]
    if corruption == "body_hash":
        terminal_round["terminal"]["manifest"]["body_sha256"] = "0" * 64
    elif corruption == "scope":
        terminal_round["terminal"]["manifest"]["presentation_id"] = "other-presentation"
    else:
        terminal_round["terminal"]["round_index"] = 9

    replay = _presented_run(
        TelegramAgent(model=SequenceModel(AssertionError("corrupt replay called model")),
                      ledger=ledger, tool_executor=executor),
        loader=lambda: (_ for _ in ()).throw(AssertionError("corrupt replay loaded evidence")),
    )
    assert replay["status"] == "failed"
    assert "presentation_manifest" not in replay


@pytest.mark.parametrize("mutation", [
    "delete_marker", "none_marker", "string_marker", "bool_round_index",
    "bool_call_count", "bool_call_start", "unknown_marker_field",
])
def test_terminal_marker_damage_never_becomes_legacy_prose(mutation):
    from services.connectors.telegram.agent import TelegramAgent

    ledger, _first = _saved_terminal_checkpoint()
    response = ledger.turns[("search-conversation", "producer-operation")]["model_response"]
    terminal_round = response["rounds"][-1]
    if mutation == "delete_marker":
        del terminal_round["terminal"]
    elif mutation == "none_marker":
        terminal_round["terminal"] = None
    elif mutation == "string_marker":
        terminal_round["terminal"] = "terminal"
    elif mutation == "bool_round_index":
        terminal_round["terminal"]["round_index"] = False
    elif mutation == "bool_call_count":
        terminal_round["call_count"] = False
    elif mutation == "bool_call_start":
        terminal_round["call_start"] = False
    else:
        terminal_round["terminal"]["unexpected"] = True

    replay = TelegramAgent(
        model=SequenceModel(AssertionError("damaged replay called model")),
        ledger=ledger, tool_executor=Executor(),
    ).run(conversation_id="search-conversation", delivery_id="producer-operation", history=[])
    assert replay["status"] == "failed"
    assert "presentation_manifest" not in replay


def test_coherently_rewritten_manifest_body_fails_raw_argument_pin():
    import hashlib
    from services.connectors.telegram.agent import TelegramAgent

    ledger, _first = _saved_terminal_checkpoint()
    terminal = ledger.turns[("search-conversation", "producer-operation")][
        "model_response"]["rounds"][-1]["terminal"]
    manifest = terminal["manifest"]
    manifest["blocks"][0]["text"] = "Tampered body."
    manifest["body"] = "Tampered body."
    manifest["body_sha256"] = hashlib.sha256(manifest["body"].encode("utf-8")).hexdigest()

    replay = TelegramAgent(
        model=SequenceModel(AssertionError("rewritten manifest replay called model")),
        ledger=ledger, tool_executor=Executor(),
    ).run(conversation_id="search-conversation", delivery_id="producer-operation", history=[])
    assert replay["status"] == "failed"


@pytest.mark.parametrize("mutation", [
    "unsupported_tag", "missing_rounds", "non_list_rounds", "unknown_wrapper_field",
])
def test_malformed_persisted_round_wrapper_fails_closed(mutation):
    from services.connectors.telegram.agent import TelegramAgent

    ledger, _first = _saved_terminal_checkpoint()
    response = ledger.turns[("search-conversation", "producer-operation")]["model_response"]
    if mutation == "unsupported_tag":
        response["_telegram_tool_loop_rounds"] = 2
    elif mutation == "missing_rounds":
        response.pop("rounds")
    elif mutation == "non_list_rounds":
        response["rounds"] = "malformed"
    else:
        response["unexpected"] = True

    replay = TelegramAgent(
        model=SequenceModel(AssertionError("malformed wrapper replay called model")),
        ledger=ledger, tool_executor=Executor(),
    ).run(conversation_id="search-conversation", delivery_id="producer-operation", history=[])
    assert replay["status"] == "failed"


def test_ordinary_invalid_native_call_retry_never_becomes_prose():
    from services.connectors.telegram.agent import TelegramAgent

    ledger = MemoryLedger()
    first = TelegramAgent(
        model=SequenceModel(_response(_call("query", {"unknown": "argument"}))),
        ledger=ledger, tool_executor=Executor(),
    ).run(conversation_id="ordinary-conversation", delivery_id="invalid-call", history=[])
    assert first["status"] == "failed"
    replay = TelegramAgent(
        model=SequenceModel(AssertionError("invalid native replay called model")),
        ledger=ledger, tool_executor=Executor(),
    ).run(conversation_id="ordinary-conversation", delivery_id="invalid-call", history=[])
    assert replay["status"] == "failed"


def test_presented_retry_rejects_pending_native_write_before_executor_effect():
    from services.connectors.telegram.agent import TelegramAgent

    ledger = MemoryLedger()

    class LostBeforeEffect:
        def execute(self, **_kwargs):
            raise RuntimeError("synthetic failure before write effect")

    with pytest.raises(RuntimeError, match="before write effect"):
        TelegramAgent(
            model=SequenceModel(_response(_call(
                "create_folder", {"name": "pending"}, call_id="pending-write",
            ))), ledger=ledger, tool_executor=LostBeforeEffect(),
        ).run(conversation_id="search-conversation", delivery_id="producer-operation", history=[])

    spy = Executor()
    replay = _presented_run(TelegramAgent(
        model=SequenceModel(AssertionError("pending write retry called model")),
        ledger=ledger, tool_executor=spy,
    ), loader=lambda: (_ for _ in ()).throw(AssertionError("pending write loaded evidence")))
    assert replay["status"] == "failed"
    assert spy.calls == []


def test_presented_replay_preserves_completed_legacy_write_result_and_prose():
    from services.connectors.telegram.agent import TelegramAgent

    ledger, executor = MemoryLedger(), Executor()
    first = TelegramAgent(
        model=SequenceModel(
            _response(_call("create_folder", {"name": "complete"}, call_id="completed-write")),
            _response(text="Folder created."),
        ), ledger=ledger, tool_executor=executor,
    ).run(conversation_id="search-conversation", delivery_id="producer-operation", history=[])
    assert first["status"] == "ok" and first["answer"] == "Folder created."
    initial_calls = list(executor.calls)
    replay_executor = Executor()
    replay = _presented_run(TelegramAgent(
        model=SequenceModel(AssertionError("completed prose replay called model")),
        ledger=ledger, tool_executor=replay_executor,
    ), loader=lambda: (_ for _ in ()).throw(AssertionError("completed prose loaded evidence")))
    assert replay == first
    assert replay_executor.calls == []
    assert executor.calls == initial_calls


@pytest.mark.parametrize("fault", ["stale_fence", "commit_then_raise"])
def test_terminal_persistence_fault_never_returns_uncommitted_body(
    firestore_fake, fault, monkeypatch,
):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    client_db = firestore_fake.client_class()
    fixture = _source_search_fixture(client_db)
    terminal_response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
            name="present_search_answer", args={"blocks": [{
                "type": "paragraph", "text": "Safe body.", "source_refs": [],
            }]},
        ))]),
    )])

    class Models:
        def generate_content(self, **_request):
            return terminal_response

    source_scope = {
        "context_id": fixture["accepted"].context_id,
        "conversation_id": fixture["conversation_id"],
        "producing_operation_id": fixture["accepted"].operation_id,
        "presentation_id": "fault-presentation",
        "answer_turn_id": "fault-answer",
    }
    if fault == "stale_fence":
        ledger = FirestoreAgentLedger(
            client_db, context_id=fixture["accepted"].context_id,
            operation_id=fixture["accepted"].operation_id, owner="stale-owner",
            fence=fixture["lease"].fence, clock=lambda: fixture["now"],
        )
    else:
        ledger = fixture["ledger"]
        save = ledger.save_turn

        def commit_then_raise(*args, **kwargs):
            save(*args, **kwargs)
            raise RuntimeError("synthetic post-commit response loss")

        monkeypatch.setattr(ledger, "save_turn", commit_then_raise)

    with pytest.raises((RuntimeError, ValueError, PermissionError)):
        TelegramAgent(model=GeminiAgent(client=SimpleNamespace(models=Models())),
                      ledger=ledger, tool_executor=Executor()).run(
            conversation_id=fixture["checkpoint_id"],
            delivery_id=fixture["accepted"].operation_id,
            history=[], presentation_scope=source_scope,
            presentation_sources=lambda: [],
        )

    if fault == "commit_then_raise":
        monkeypatch.setattr(ledger, "save_turn", save)
        retry_model = SimpleNamespace(generate=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("committed terminal retried generation")))
        retry_sources = lambda: (_ for _ in ()).throw(
            AssertionError("committed terminal retried source loading"))
        client_db.collection("telegram_sessions").document(
            fixture["accepted"].context_id,
        ).collection("operations").document(fixture["accepted"].operation_id).update({
            "state": "completed",
        })
        replay = TelegramAgent(model=retry_model, ledger=ledger,
                               tool_executor=Executor()).run(
            conversation_id=fixture["checkpoint_id"],
            delivery_id=fixture["accepted"].operation_id,
            history=[], presentation_scope=source_scope,
            presentation_sources=retry_sources,
            executor_factory=lambda *_args: (_ for _ in ()).throw(
                AssertionError("committed terminal reconstructed executor")),
        )
        assert replay["status"] == "ok" and replay["answer"] == "Safe body."
    else:
        retry = TelegramAgent(model=GeminiAgent(client=SimpleNamespace(models=Models())),
                              ledger=fixture["ledger"], tool_executor=Executor()).run(
            conversation_id=fixture["checkpoint_id"],
            delivery_id=fixture["accepted"].operation_id,
            history=[], presentation_scope=source_scope,
            presentation_sources=lambda: [],
        )
        assert retry["status"] == "ok" and retry["answer"] == "Safe body."


def test_real_checkpoint_read_evidence_builds_list_and_quote_terminal(firestore_fake):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    client_db = firestore_fake.client_class()
    fixture = _source_search_fixture(client_db)
    source_call = fixture["call"]
    source_refs = [{
        "doc_id": "doc-b", "order_index": 2,
        "operation_id": fixture["accepted"].operation_id,
        "checkpoint_conversation_id": fixture["checkpoint_id"],
        "call_id": source_call["id"], "ordinal": source_call["ordinal"],
        "reference": fixture["reference"],
    }]
    blocks = [
        {"type": "document_list", "list_id": "results-1", "entries": [
            {"doc_id": "doc-a", "label": "Alpha", "source_refs": [{
                **source_refs[0], "doc_id": "doc-a", "order_index": 1,
            }]},
            {"doc_id": "doc-b", "label": "Beta", "source_refs": source_refs},
        ]},
        {"type": "quote", "doc_id": "doc-b", "text": "The exact verified sentence.",
         "source_refs": source_refs},
        {"type": "paragraph", "text": "Verified context: " + ("x" * 5000),
         "source_refs": source_refs},
    ]
    terminal_response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(
                id="slash/雪-terminal", name="present_search_answer", args={"blocks": blocks},
            ), thought_signature="Y2hhbmdlLWZha2Utc2lnbmF0dXJl",
        )]),
    )], usage_metadata=types.GenerateContentResponseUsageMetadata(
        prompt_token_count=8, candidates_token_count=3, total_token_count=11,
    ))

    class Models:
        def __init__(self, response):
            self.response = response
            self.requests = []

        def generate_content(self, **request):
            self.requests.append(request)
            response, self.response = self.response, None
            if response is None:
                raise AssertionError("terminal replay called Gemini")
            return response

    models = Models(terminal_response)
    model = GeminiAgent(client=SimpleNamespace(models=models))
    executor = Executor()
    agent = TelegramAgent(model=model, ledger=fixture["ledger"], tool_executor=executor)
    scope = {
        "context_id": fixture["accepted"].context_id,
        "conversation_id": fixture["conversation_id"],
        "producing_operation_id": fixture["accepted"].operation_id,
        "presentation_id": "presentation-real-checkpoint",
        "answer_turn_id": "answer-turn-real-checkpoint",
    }

    def load_verified_source():
        exchange = fixture["ledger"].load_search_exchange(
            fixture["reference"], conversation_id=fixture["conversation_id"],
            operation_id=fixture["accepted"].operation_id,
            checkpoint_conversation_id=fixture["checkpoint_id"],
            call_id=source_call["id"], ordinal=source_call["ordinal"],
        )
        return [{"reference": fixture["reference"], "exchange": exchange,
                 "source_filters": {
                     "query": "synthetic project discussions", "keywords": None,
                     "folder_path": None, "date_from": None, "date_to": None,
                 }}]

    first = agent.run(
        conversation_id=fixture["checkpoint_id"],
        delivery_id=fixture["accepted"].operation_id,
        history=[], presentation_scope=scope, presentation_sources=load_verified_source,
    )
    assert first["status"] == "ok"
    assert first["answer"].startswith("1. Alpha\n2. Beta\n\n> The exact verified sentence.")
    assert len(first["answer"].encode("utf-8")) > 4096
    assert first["presentation_manifest"]["blocks"][0]["entries"][1]["doc_id"] == "doc-b"
    assert first["presentation_manifest"]["blocks"][1]["evidence_refs"][0]["order_index"] == 2
    assert executor.calls == []
    saved_record = fixture["ledger"].load_turn(
        fixture["checkpoint_id"], fixture["accepted"].operation_id,
    )
    saved_rounds = saved_record["model_response"]["rounds"]
    assert "Y2hhbmdlLWZha2Utc2lnbmF0dXJl" in repr(saved_rounds[-1]["response"])
    stored = fixture["ledger"]._ref(fixture["accepted"].operation_id).get().to_dict()
    assert stored["model_response"].get("_input_ref", "").endswith("/input_parts/manifest")
    request = models.requests[0]
    assert request["config"].tool_config.function_calling_config.mode == "ANY"
    assert {decl.name for tool in request["config"].tools for decl in tool.function_declarations} >= {
        "query", "present_search_answer",
    }

    operation_ref = (client_db.collection("telegram_sessions")
                     .document(fixture["accepted"].context_id)
                     .collection("operations").document(fixture["accepted"].operation_id))
    operation_ref.update({"state": "completed"})
    replay = TelegramAgent(
        model=SimpleNamespace(generate=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal replay generated a model response"))),
        ledger=fixture["ledger"], tool_executor=executor,
    ).run(
        conversation_id=fixture["checkpoint_id"],
        delivery_id=fixture["accepted"].operation_id,
        history=[],
        executor_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal replay constructed an executor")),
    )
    assert replay == first


def test_native_read_then_terminal_runs_in_one_real_ledger_invocation(firestore_fake):
    import hashlib
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from tests.services.connectors.telegram.test_search_presentation import _running_command

    client_db = firestore_fake.client_class()
    _repository, accepted, _operation, lease, now = _running_command(
        client_db, text="Find the synthetic project discussions",
    )
    digest = hashlib.sha256(
        f"{accepted.context_id}\0{accepted.operation_id}".encode("utf-8")
    ).hexdigest()
    search_conversation_id = "search-" + digest
    checkpoint_conversation_id = "telegram-global-" + digest
    ledger = FirestoreAgentLedger(
        client_db, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    query_id = "native/query-雪"
    result_rows = [
        {"doc_id": "doc-a", "order_index": 1, "quote": "Alpha verified."},
        {"doc_id": "doc-b", "order_index": 2, "quote": "Beta exact sentence."},
    ]
    search_result = {"segments": result_rows, "metadata": {
        "truncated": False, "total_matches": 2, "returned_segments": 2,
    }}

    class SearchExecutor:
        def __init__(self):
            self.calls = []
            self.reference = None

        def execute(self, *, name, arguments, call_ordinal):
            self.calls.append((name, deepcopy(arguments), call_ordinal))
            assert name == "query" and call_ordinal == 0
            call = {"id": query_id, "name": name, "arguments": dict(arguments),
                    "ordinal": call_ordinal, "round": 0}
            self.reference = ledger.save_search_exchange(
                search_conversation_id, checkpoint_conversation_id=checkpoint_conversation_id,
                call=call, canonical_call_id=query_id, result=search_result,
            )
            return search_result

    executor = SearchExecutor()

    def native_part(name, args, call_id, signature):
        return types.Part(function_call=types.FunctionCall(
            name=name, args=args, id=call_id,
        ), thought_signature=signature)

    class Models:
        def __init__(self):
            self.requests = []

        def generate_content(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return types.GenerateContentResponse(candidates=[types.Candidate(
                    content=types.Content(parts=[native_part(
                        "query", {"query": "synthetic project discussions"}, query_id,
                        "Y2hhbmdlLXF1ZXJ5LXNpZw==",
                    )]),
                )])
            assert len(self.requests) == 2 and executor.reference is not None
            refs = [{"doc_id": "doc-a", "order_index": 1,
                     "reference": executor.reference}]
            beta_ref = [{"doc_id": "doc-b", "order_index": 2,
                         "reference": executor.reference}]
            blocks = [
                {"type": "document_list", "list_id": "same-run-list", "entries": [
                    {"doc_id": "doc-a", "label": "Alpha", "source_refs": refs},
                    {"doc_id": "doc-b", "label": "Beta", "source_refs": beta_ref},
                ]},
                {"type": "quote", "doc_id": "doc-b", "text": "Beta exact sentence.",
                 "source_refs": beta_ref},
            ]
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[native_part(
                    "present_search_answer", {"blocks": blocks}, "terminal/雪",
                    "Y2hhbmdlLXRlcm1pbmFsLXNpZw==",
                )]),
            )])

    models = Models()
    model = GeminiAgent(client=SimpleNamespace(models=models))
    scope = {
        "context_id": accepted.context_id,
        "conversation_id": search_conversation_id,
        "producing_operation_id": accepted.operation_id,
        "presentation_id": "same-run-presentation",
        "answer_turn_id": "same-run-answer",
    }

    def load_evidence():
        loaded = ledger.load_search_exchange(
            executor.reference, conversation_id=search_conversation_id,
            operation_id=accepted.operation_id,
            checkpoint_conversation_id=checkpoint_conversation_id,
            call_id=query_id, ordinal=0,
        )
        return [{"reference": executor.reference, "exchange": loaded,
                 "source_filters": {
                     "query": "synthetic project discussions", "keywords": None,
                     "folder_path": None, "date_from": None, "date_to": None,
                 }}]

    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    first = agent.run(
        conversation_id=checkpoint_conversation_id, delivery_id=accepted.operation_id,
        history=[{"role": "user", "parts": [{"text": "Find and list these discussions."}]}],
        presentation_scope=scope, presentation_sources=load_evidence,
    )
    assert first["status"] == "ok"
    assert first["answer"] == "1. Alpha\n2. Beta\n\n> Beta exact sentence."
    assert executor.calls == [("query", {"query": "synthetic project discussions"}, 0)]
    assert len(models.requests) == 2
    response_parts = [part for content in models.requests[1]["contents"]
                      for part in (content.get("parts", []) if isinstance(content, dict)
                                   else getattr(content, "parts", []))]
    function_responses = [getattr(part, "function_response", None) for part in response_parts]
    query_response = next(item for item in function_responses if item is not None)
    assert query_response.name == "query" and query_response.id == query_id
    saved = ledger.load_turn(checkpoint_conversation_id, accepted.operation_id)
    assert [call["ordinal"] for call in saved["tool_calls"]] == [0]
    rounds = saved["model_response"]["rounds"]
    assert len(rounds) == 2 and rounds[-1]["call_count"] == 0
    assert "Y2hhbmdlLXF1ZXJ5LXNpZw==" in repr(rounds[0]["response"])
    assert rounds[-1]["terminal"]["manifest"]["body"] == first["answer"]

    client_db.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations").document(accepted.operation_id).update({"state": "completed"})
    replay = TelegramAgent(
        model=SimpleNamespace(generate=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed real terminal replay called model"))),
        ledger=ledger, tool_executor=executor,
    ).run(
        conversation_id=checkpoint_conversation_id, delivery_id=accepted.operation_id,
        history=[], executor_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("completed real terminal replay built executor")),
    )
    assert replay == first and executor.calls == [
        ("query", {"query": "synthetic project discussions"}, 0),
    ]
