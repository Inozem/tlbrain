"""Persisted Telegram tool orchestration over the shared operation registry."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from core.integrations.gemini.agent import GeminiAgent
from core.operations import registry
from services.connectors.telegram.tools import get_tool_catalog

_PERSISTED_ROUNDS = "_telegram_tool_loop_rounds"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return _json_safe(to_json_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json", exclude_none=False))
        except TypeError:
            return _json_safe(model_dump())
    if hasattr(value, "__dict__"):
        return {key: _json_safe(item) for key, item in vars(value).items()
                if not key.startswith("_")}
    return str(value)


def _serialize_response(response: Any) -> Any:
    serialized = _json_safe(response)
    text = _field(response, "text")
    if isinstance(serialized, dict) and isinstance(text, str) and "text" not in serialized:
        serialized["text"] = text
    return serialized


def _response_parts(response: Any) -> list[Any]:
    parts = _field(response, "parts")
    if parts is not None:
        return list(parts)
    candidates = _field(response, "candidates", ()) or ()
    if not candidates:
        return []
    content = _field(candidates[0], "content")
    return list(_field(content, "parts", ()) or ())


def _call_part(part: Any) -> Any | None:
    return _field(part, "function_call", part if _field(part, "name") else None)


def _calls_from(response: Any) -> list[dict[str, Any]]:
    calls = []
    for part in _response_parts(response):
        call = _call_part(part)
        if call is None:
            continue
        name = _field(call, "name")
        arguments = _field(call, "args", _field(call, "arguments"))
        calls.append({
            "id": _field(call, "id"),
            "name": name,
            "arguments": arguments,
            "thought_signature": _field(call, "thought_signature"),
        })
    return calls


def _tool_declarations() -> list[dict[str, Any]]:
    return [
        {"name": item["name"], "description": item["description"],
         "parameters": item["inputSchema"]}
        for item in get_tool_catalog()["tools"]
    ]


def _usage(response: Any) -> dict[str, int]:
    metadata = _field(response, "usage_metadata")
    if metadata is None:
        return {}
    keys = ("prompt_token_count", "candidates_token_count", "total_token_count",
            "thoughts_token_count", "cached_content_token_count")
    return {key: value for key in keys
            if isinstance((value := _field(metadata, key)), int) and not isinstance(value, bool)}


def _aggregate_usage(rounds: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for saved_round in rounds:
        for key, value in _usage(saved_round["response"]).items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _rounds_from_record(record: Any) -> list[dict[str, Any]]:
    response = _field(record, "model_response")
    if isinstance(response, Mapping) and response.get(_PERSISTED_ROUNDS) == 1:
        return list(response.get("rounds", ()))
    calls = list(_field(record, "tool_calls", ()) or ())
    return [{"response": response, "call_start": 0, "call_count": len(calls)}]


@dataclass
class _ToolLoopState:
    rounds: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[int, Any] = field(default_factory=dict)

    @property
    def usage(self) -> dict[str, int]:
        return _aggregate_usage(self.rounds)


class TelegramAgent:
    """Run persisted, validated tool steps and return the current answer."""

    def __init__(self, *, model: Any, ledger: Any, tool_executor: Any, max_rounds: int = 16):
        self.model = model
        self.ledger = ledger
        self.tool_executor = tool_executor
        self.max_rounds = min(max_rounds, 16)

    def run(self, *, conversation_id: str, delivery_id: str, history: list[Any]) -> dict[str, Any]:
        contents = list(history)
        state = self._load_state(conversation_id, delivery_id)
        replay_failure = self._replay_saved_rounds(
            contents, state, conversation_id=conversation_id, delivery_id=delivery_id,
        )
        if replay_failure:
            return replay_failure
        if self._has_final_answer(state):
            return self._final_answer(state)

        while len(state.rounds) < self.max_rounds:
            result = self._run_model_round(
                contents, state, conversation_id=conversation_id, delivery_id=delivery_id,
            )
            if result is not None:
                return result

        return self._failure(
            f"The tool loop exceeded the {self.max_rounds} model-round limit.", state.usage,
        )

    def _load_state(self, conversation_id: str, delivery_id: str) -> _ToolLoopState:
        record = self.ledger.load_turn(conversation_id, delivery_id)
        if record is None:
            return _ToolLoopState()

        state = _ToolLoopState(
            rounds=_rounds_from_record(record),
            calls=list(_field(record, "tool_calls", ()) or ()),
        )
        saved_results = _field(record, "tool_results", {}) or {}
        if isinstance(saved_results, Mapping):
            state.tool_results = {int(ordinal): result
                                  for ordinal, result in saved_results.items()}
        return state

    def _replay_saved_rounds(
        self, contents: list[Any], state: _ToolLoopState, *,
        conversation_id: str, delivery_id: str,
    ) -> dict[str, Any] | None:
        for round_index, saved_round in enumerate(state.rounds):
            response = saved_round["response"]
            round_calls = [call for call in state.calls
                           if int(call.get("round", 0)) == round_index]
            if not round_calls and _calls_from(response):
                return self._failure("The persisted tool request is incomplete.", state.usage)

            self._append_model_response(contents, response)
            failure = self._run_calls(
                contents, response, round_calls, state.tool_results,
                conversation_id=conversation_id, delivery_id=delivery_id,
            )
            if failure:
                return failure
        return None

    @staticmethod
    def _has_final_answer(state: _ToolLoopState) -> bool:
        return bool(state.rounds and int(state.rounds[-1].get("call_count", 0)) == 0)

    @staticmethod
    def _final_answer(state: _ToolLoopState) -> dict[str, Any]:
        answer = _field(state.rounds[-1]["response"], "text", "") or ""
        return {"status": "ok", "answer": answer, "usage": state.usage,
                "tool_results": dict(state.tool_results)}

    def _run_model_round(
        self, contents: list[Any], state: _ToolLoopState, *,
        conversation_id: str, delivery_id: str,
    ) -> dict[str, Any] | None:
        response = self.model.generate(
            contents=contents, mode="tool_loop", tools=_tool_declarations(),
        )
        try:
            calls = self._validate_calls(response, state)
        except (ValueError, TypeError):
            self._persist_model_round(response, [], state, conversation_id, delivery_id)
            return self._failure("The requested tool operation was invalid.", state.usage)

        serialized = self._persist_model_round(
            response, calls, state, conversation_id, delivery_id,
        )
        self._append_model_response(contents, serialized)
        failure = self._run_calls(
            contents, serialized, calls, state.tool_results,
            conversation_id=conversation_id, delivery_id=delivery_id,
        )
        if failure:
            return failure
        if not calls:
            return self._final_answer(state)
        return None

    @staticmethod
    def _validate_calls(response: Any, state: _ToolLoopState) -> list[dict[str, Any]]:
        next_ordinal = max((int(call["ordinal"]) for call in state.calls), default=-1) + 1
        round_index = len(state.rounds)
        validated = []
        for index, call in enumerate(_calls_from(response)):
            arguments = registry.validate_arguments(call["name"], call["arguments"])
            validated.append({**call, "arguments": arguments,
                              "ordinal": next_ordinal + index, "round": round_index})
        return validated

    def _persist_model_round(
        self, response: Any, calls: list[dict[str, Any]], state: _ToolLoopState,
        conversation_id: str, delivery_id: str,
    ) -> Any:
        serialized = _serialize_response(response)
        all_rounds = state.rounds + [{
            "response": serialized,
            "call_start": len(state.calls),
            "call_count": len(calls),
        }]
        self.ledger.save_turn(
            conversation_id, delivery_id,
            model_response=(serialized if not state.rounds else self._persisted_response(all_rounds)),
            tool_calls=state.calls + calls,
        )
        state.rounds.append(all_rounds[-1])
        state.calls.extend(calls)
        return serialized

    @staticmethod
    def _persisted_response(rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return {_PERSISTED_ROUNDS: 1, "rounds": deepcopy(rounds)}

    @staticmethod
    def _append_model_response(contents: list[Any], response: Any) -> None:
        contents.append({"role": "model", "parts": _response_parts(response)})

    def _run_calls(
        self, contents: list[Any], response: Any, calls: list[dict[str, Any]],
        tool_results: dict[int, Any], *, conversation_id: str, delivery_id: str,
    ) -> dict[str, Any] | None:
        for call in calls:
            ordinal = int(call["ordinal"])
            if ordinal in tool_results:
                result = tool_results[ordinal]
            else:
                result = self.tool_executor.execute(
                    name=call["name"], arguments=call["arguments"], call_ordinal=ordinal,
                )
                result = _json_safe(result)
                self.ledger.save_tool_result(conversation_id, delivery_id, ordinal, result)
                tool_results[ordinal] = result
            source_call = self._source_call(response, call)
            response_part = GeminiAgent.make_function_response(call=source_call, response=result)
            contents.append({"role": "user", "parts": [response_part]})
        return None

    @staticmethod
    def _source_call(response: Any, saved_call: Mapping[str, Any]) -> Any:
        for part in _response_parts(response):
            call = _call_part(part)
            if call is not None and _field(call, "id") == saved_call.get("id"):
                if isinstance(call, Mapping):
                    return SimpleNamespace(**call)
                return call
        return SimpleNamespace(id=saved_call.get("id"), name=saved_call["name"])

    @staticmethod
    def _failure(message: str, usage: dict[str, int]) -> dict[str, Any]:
        return {"status": "failed", "failure": message, "usage": usage}
