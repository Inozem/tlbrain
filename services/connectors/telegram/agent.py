"""Persisted Telegram tool orchestration over the shared operation registry."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
from types import SimpleNamespace
from typing import Any

from core.integrations.gemini.agent import GeminiAgent
from core.operations import registry
from services.connectors.telegram.tools import get_tool_catalog

_PERSISTED_ROUNDS = "_telegram_tool_loop_rounds"


async def _call_sync_or_async(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run synchronous collaborators off-loop and await async collaborators."""
    result = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


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


def _wire_model_part(part: Any) -> Any:
    """Normalize legacy routed-call fields only when replaying to Gemini."""
    if not isinstance(part, Mapping) or "function_call" in part:
        return part
    if not {"name", "args"}.issubset(part):
        return part
    call = {key: part[key] for key in ("id", "name", "args") if key in part}
    return {
        **{key: value for key, value in part.items() if key not in {"id", "name", "args"}},
        "function_call": call,
    }


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

    def run(
        self, *, conversation_id: str, delivery_id: str, history: list[Any],
        target_doc_id: str | None = None,
        executor_factory: Callable[[str, str, Any], Any] | None = None,
        execution_context: Any = None,
    ) -> dict[str, Any]:
        """Run one operation with an optional per-invocation pinned executor."""
        if target_doc_id is not None and (not isinstance(target_doc_id, str) or not target_doc_id):
            return self._failure("The operation target is invalid.", {})
        if target_doc_id is not None and executor_factory is None:
            return self._failure("The operation target is not bound to a tool executor.", {})
        try:
            executor = (
                executor_factory(target_doc_id, delivery_id, execution_context)
                if target_doc_id is not None and executor_factory is not None
                else self.tool_executor
            )
        except Exception:
            return self._failure("The operation target could not be bound to a tool executor.", {})
        if not callable(getattr(executor, "execute", None)):
            return self._failure("The operation tool executor is unavailable.", {})
        contents = list(history)
        state = self._load_state(conversation_id, delivery_id)
        replay_failure = self._replay_saved_rounds(
            contents, state, conversation_id=conversation_id, delivery_id=delivery_id,
            executor=executor, target_doc_id=target_doc_id,
        )
        if replay_failure:
            return replay_failure
        if self._has_final_answer(state):
            return self._final_answer(state)

        while len(state.rounds) < self.max_rounds:
            result = self._run_model_round(
                contents, state, conversation_id=conversation_id, delivery_id=delivery_id,
                executor=executor, target_doc_id=target_doc_id,
            )
            if result is not None:
                return result

        return self._failure(
            f"The tool loop exceeded the {self.max_rounds} model-round limit.", state.usage,
        )

    def dispatch_tool_choice(
        self, *, name: str, arguments: Any, target_doc_id: str,
        conversation_id: str, delivery_id: str,
        executor_factory: Callable[[str, str, Any], Any],
        execution_context: Any = None,
        link_requested: bool = False,
    ) -> Any:
        """Dispatch one structured route choice through the common operation schema."""
        validated = registry.validate_arguments(name, arguments)
        if not isinstance(target_doc_id, str) or not target_doc_id:
            raise ValueError("operation target is invalid")
        argument_doc_id = validated.get("doc_id")
        if argument_doc_id is not None and argument_doc_id != target_doc_id:
            raise ValueError("operation target does not match the routed target")
        if not isinstance(link_requested, bool) or (link_requested and name != "get_transcript"):
            raise ValueError("requested link is invalid")
        if link_requested:
            execution_context = {
                **(execution_context if isinstance(execution_context, Mapping) else {}),
                "include_link": True,
            }
        executor = executor_factory(target_doc_id, delivery_id, execution_context)
        execute = getattr(executor, "execute", None)
        if not callable(execute):
            raise TypeError("operation tool executor is unavailable")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("operation conversation is invalid")

        state = self._load_state(conversation_id, delivery_id)
        response: Any = {"parts": [{"function_call": {
            "id": f"route-{delivery_id}", "name": name, "args": validated,
        }}]}
        if state.calls:
            saved = state.calls[0]
            if (saved.get("name") != name or saved.get("arguments") != validated
                    or not state.rounds
                    or int(saved.get("round", 0)) != 0
                    or int(state.rounds[0].get("call_count", 0)) != 1
                    or sum(int(call.get("round", 0)) == 0 for call in state.calls) != 1):
                raise ValueError("persisted tool choice does not match the routed operation")
            response = state.rounds[0]["response"]
            calls = [saved]
        else:
            if state.rounds:
                raise ValueError("persisted tool choice does not match the routed operation")
            calls = self._validate_calls(response, state, target_doc_id=target_doc_id)
            self._persist_model_round(response, calls, state, conversation_id, delivery_id)
        ordinal = int(calls[0]["ordinal"])
        if ordinal not in state.tool_results:
            failure = self._run_calls(
                [], response, calls, state.tool_results,
                conversation_id=conversation_id, delivery_id=delivery_id,
                executor=executor, target_doc_id=target_doc_id,
            )
            if failure:
                raise RuntimeError("routed tool operation failed")
        return state.tool_results[ordinal]

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
        conversation_id: str, delivery_id: str, executor: Any,
        target_doc_id: str | None,
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
                executor=executor, target_doc_id=target_doc_id,
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
        conversation_id: str, delivery_id: str, executor: Any,
        target_doc_id: str | None,
    ) -> dict[str, Any] | None:
        response = self.model.generate(
            contents=contents, mode="tool_loop", tools=_tool_declarations(),
        )
        try:
            calls = self._validate_calls(response, state, target_doc_id=target_doc_id)
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
            executor=executor, target_doc_id=target_doc_id,
        )
        if failure:
            return failure
        if not calls:
            return self._final_answer(state)
        return None

    @staticmethod
    def _validate_calls(
        response: Any, state: _ToolLoopState, *, target_doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        next_ordinal = max((int(call["ordinal"]) for call in state.calls), default=-1) + 1
        round_index = len(state.rounds)
        validated = []
        for index, call in enumerate(_calls_from(response)):
            arguments = registry.validate_arguments(call["name"], call["arguments"])
            if (target_doc_id is not None and "doc_id" in arguments
                    and arguments["doc_id"] != target_doc_id):
                raise ValueError("tool target does not match the routed operation")
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
        contents.append({
            "role": "model",
            "parts": [_wire_model_part(part) for part in _response_parts(response)],
        })

    def _run_calls(
        self, contents: list[Any], response: Any, calls: list[dict[str, Any]],
        tool_results: dict[int, Any], *, conversation_id: str, delivery_id: str,
        executor: Any, target_doc_id: str | None,
    ) -> dict[str, Any] | None:
        for call in calls:
            ordinal = int(call["ordinal"])
            if ordinal in tool_results:
                result = tool_results[ordinal]
            else:
                result = executor.execute(
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
        saved_id = saved_call.get("id")
        saved_name = saved_call["name"]
        saved_arguments = saved_call.get("arguments")
        for part in _response_parts(response):
            call = _call_part(part)
            if call is None:
                continue
            call_id = _field(call, "id")
            if saved_id is not None:
                matches = call_id == saved_id
            else:
                # FunctionCall.id is optional. Correlate an id-less call by
                # its persisted name and validated arguments, never by None
                # alone (which could associate a different call in the same
                # model round).
                call_arguments = _field(call, "args", _field(call, "arguments"))
                matches = (call_id is None and _field(call, "name") == saved_name
                           and call_arguments == saved_arguments)
            if not matches:
                continue
            if isinstance(call, Mapping):
                source = dict(call)
                source["id"] = call_id if call_id is not None else saved_id
                source["name"] = saved_name
                return SimpleNamespace(**source)
            if call_id is not None and _field(call, "name") == saved_name:
                return call
            # SDK objects may omit the optional attribute entirely. Keep
            # available opaque fields (such as thought_signature) intact.
            source = dict(vars(call)) if hasattr(call, "__dict__") else {}
            source["id"] = call_id if call_id is not None else saved_id
            source["name"] = saved_name
            return SimpleNamespace(**source)
        return SimpleNamespace(id=saved_id, name=saved_name, args=saved_arguments)

    @staticmethod
    def _failure(message: str, usage: dict[str, int]) -> dict[str, Any]:
        return {"status": "failed", "failure": message, "usage": usage}


@dataclass(frozen=True)
class _CommandReplay:
    route: dict[str, Any] | None
    text: str
    dependency_operation_id: Any


class TelegramCommandProcessor:
    """Queue adapter for a persisted command candidate and structured router.

    Context loading, model routing, notifications, and executor construction
    stay injectable so the adapter can be exercised offline and bound to the
    accepted operation's lease and routed target.
    """

    def __init__(
        self, *, context_loader: Callable[..., Any], router: Callable[..., Any],
        repository: Any, agent: TelegramAgent, notifications: Any,
        executor_factory: Callable[[str, str, Any], Any],
        queue: Any = None,
        selection_publisher: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        for name, value in (("context_loader", context_loader), ("router", router),
                            ("executor_factory", executor_factory)):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if not callable(getattr(agent, "run", None)):
            raise TypeError("agent must provide run")
        self.context_loader = context_loader
        self.router = router
        self.repository = repository
        self.agent = agent
        self.notifications = notifications
        self.executor_factory = executor_factory
        self.queue = queue
        self.selection_publisher = selection_publisher
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def __call__(
        self, context_id: str, operation: Any, lease: Any, *,
        command: Any, retry_governor: Any = None,
    ) -> Any:
        from core.operations.queue import QueueOperationState

        if not self._valid_request(context_id, operation, lease, command):
            return QueueOperationState.FAILED
        operation_id = _field(operation, "operation_id")
        context = await self._load_valid_context(context_id, operation)
        if context is None:
            return QueueOperationState.FAILED
        replay = await self._load_replay(context_id, operation, command)
        if replay is None:
            return QueueOperationState.FAILED
        routed: dict[str, Any] = {}
        route = self._make_route_callback(operation, command, replay, routed)
        run_agent = self._make_agent_callback(
            context_id, operation_id, lease, replay, routed,
        )
        result = await self._process_operation(
            operation_id=operation_id, context=context,
            replay=replay, route=route, run_agent=run_agent, lease=lease,
        )
        return self._map_queue_state(result)

    @staticmethod
    def _valid_request(
        context_id: str, operation: Any, lease: Any, command: Any,
    ) -> bool:
        """Reject malformed queue inputs before loading context or replay state."""
        from services.connectors.telegram.commands import TelegramCommand

        operation_id = _field(operation, "operation_id")
        return not (
            not isinstance(context_id, str) or not context_id
            or not isinstance(operation_id, str) or not operation_id
            or _field(lease, "operation_id") != operation_id
            or (command is not None and (
                not isinstance(command, TelegramCommand)
                or command.kind != "freeform" or not command.text
            ))
        )

    async def _load_valid_context(self, context_id: str, operation: Any) -> Any | None:
        """Load context off-loop and confirm it still belongs to this queue item."""
        context = await _call_sync_or_async(self.context_loader, context_id, operation)
        if not context or getattr(context, "context_id", context_id) != context_id:
            return None
        return context

    async def _load_replay(
        self, context_id: str, operation: Any, command: Any,
    ) -> _CommandReplay | None:
        """Resolve persisted route state and recover its saved user request."""
        dependency_id = _field(operation, "dependency_operation_id")
        if command is not None:
            return _CommandReplay(None, command.text, dependency_id)

        operation_id = _field(operation, "operation_id")
        dependency_valid, dependency_id = await self._load_replay_dependency(
            context_id, operation_id, dependency_id,
        )
        if not dependency_valid:
            return None
        route = await self._load_pinned_route(
            context_id, operation_id, dependency_id,
        )
        if route is None:
            return None
        return _CommandReplay(
            route,
            route.get("substantive_text", "") if route.get("kind") == "substantive" else "",
            dependency_id,
        )

    async def _load_replay_dependency(
        self, context_id: str, operation_id: str, dependency_id: Any,
    ) -> tuple[bool, Any]:
        """Cross-check the queue dependency against its durable operation record."""
        client = getattr(self.repository, "client", None)
        if client is None:
            return True, dependency_id

        from services.connectors.telegram.repository import SESSIONS

        def read_operation():
            return client.collection(SESSIONS).document(context_id).collection(
                "operations"
            ).document(operation_id).get()

        operation_snapshot = await _call_sync_or_async(read_operation)
        operation_data = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        if operation_data.get("operation_id") != operation_id:
            return False, dependency_id
        durable_dependency = operation_data.get("dependency_operation_id")
        if dependency_id is not None and dependency_id != durable_dependency:
            return False, dependency_id
        return True, durable_dependency

    async def _load_pinned_route(
        self, context_id: str, operation_id: str, dependency_id: Any,
    ) -> dict[str, Any] | None:
        """Load a pinned route and verify its dependency and kind-specific data."""
        get_route = getattr(self.repository, "get_route_control", None)
        pinned = (
            await _call_sync_or_async(get_route, context_id, operation_id)
            if callable(get_route) else None
        )
        if (not isinstance(pinned, Mapping)
                or pinned.get("status") not in {"ok", "success"}
                or pinned.get("kind") not in {
                    "pure_switch", "substantive", "search", "save", "retry",
                }):
            return None
        if (pinned.get("dependency_operation_id") is not None
                and pinned.get("dependency_operation_id") != dependency_id):
            return None
        if (pinned.get("target_operation_id") is not None
                and pinned.get("target_operation_id") != dependency_id):
            return None

        replay_route = dict(pinned)
        kind = pinned.get("kind")
        if kind == "substantive":
            return await self._validate_substantive_replay(
                context_id, operation_id, pinned, replay_route,
            )
        if kind == "search" and not self._valid_search_replay(pinned):
            return None
        return replay_route

    async def _validate_substantive_replay(
        self, context_id: str, operation_id: str, pinned: Mapping[str, Any],
        replay_route: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate a pinned document route and recover its sole saved user turn."""
        conversation_id = pinned.get("conversation_id")
        target_doc_id = pinned.get("target_doc_id")
        number = pinned.get("dialog_number")
        if (not isinstance(conversation_id, str) or not conversation_id
                or not isinstance(target_doc_id, str) or not target_doc_id
                or type(number) is not int or number < 1):
            return None

        choice = pinned.get("tool_choice")
        if choice is not None:
            if not isinstance(choice, Mapping) or set(choice) - {
                "name", "arguments", "link_requested",
            }:
                return None
            name = choice.get("name")
            link_requested = choice.get("link_requested", False)
            try:
                arguments = registry.validate_arguments(name, choice.get("arguments"))
            except (TypeError, ValueError):
                return None
            if (not isinstance(name, str) or name not in registry.OPERATIONS
                    or not isinstance(link_requested, bool)
                    or (link_requested and name != "get_transcript")
                    or (arguments.get("doc_id") is not None
                        and arguments["doc_id"] != target_doc_id)):
                return None
            replay_route["tool_choice"] = {
                "name": name, "arguments": arguments,
                "link_requested": link_requested,
            }

        turns = await _call_sync_or_async(
            self.repository.list_turns, context_id, conversation_id,
        )
        user_turns = [
            turn for turn in turns
            if isinstance(turn, Mapping)
            and turn.get("operation_id") == operation_id
            and turn.get("role") == "user"
        ]
        if len(user_turns) != 1:
            return None
        parts = user_turns[0].get("parts")
        if (not isinstance(parts, list) or len(parts) != 1
                or not isinstance(parts[0], Mapping)
                or not isinstance(parts[0].get("text"), str)
                or not parts[0]["text"].strip()):
            return None
        replay_route["substantive_text"] = parts[0]["text"]
        return replay_route

    @staticmethod
    def _valid_search_replay(pinned: Mapping[str, Any]) -> bool:
        """Require the saved result references and search session for replay."""
        refs = pinned.get("result_refs")
        session_id = pinned.get("search_session_id")
        return (
            isinstance(refs, list)
            and all(isinstance(ref, str) and ref for ref in refs)
            and isinstance(session_id, str) and bool(session_id)
        )

    def _make_route_callback(
        self, operation: Any, command: Any, replay: _CommandReplay,
        routed: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build the route callback for fresh routing or pinned replay."""
        async def route(*, operation_id: str, context: Any, text: str) -> Any:
            if replay.route is not None:
                routed.clear()
                routed.update(replay.route)
                return dict(replay.route)
            if operation_id != _field(operation, "operation_id") or text != command.text:
                return {"status": "invalid_route"}
            result = await _call_sync_or_async(
                self.router,
                operation=operation, operation_id=operation_id,
                context=context, text=command.text,
            )
            if isinstance(result, Mapping):
                routed.clear()
                routed.update(result)
            return result

        return route

    def _make_agent_callback(
        self, context_id: str, operation_id: str, lease: Any,
        replay: _CommandReplay, routed: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build the agent callback with a fixed target and lease fence."""
        async def run_agent(*, prompt: Any, target_doc_id: str) -> str:
            if not isinstance(prompt, Mapping):
                raise ValueError("document prompt is invalid")
            route_result = routed
            conversation_id = route_result.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError("routed conversation is unavailable")
            turns = prompt.get("history")
            request = prompt.get("request")
            if not isinstance(turns, list) or not isinstance(request, str) or not request.strip():
                raise ValueError("document prompt is incomplete")
            history = [*turns, {"role": "user", "parts": [{"text": request}]}]
            execution_context = {
                "context_id": context_id,
                "operation_id": operation_id,
                "owner": _field(lease, "owner"),
                "fence": _field(lease, "fence"),
                "target_doc_id": target_doc_id,
                "dependency_operation_id": replay.dependency_operation_id,
            }
            choice = route_result.get("tool_choice")
            if choice is not None:
                if choice.get("link_requested", False):
                    execution_context = {**execution_context, "include_link": True}
                await _call_sync_or_async(
                    self.agent.dispatch_tool_choice,
                    name=choice["name"], arguments=choice["arguments"],
                    target_doc_id=target_doc_id, conversation_id=conversation_id,
                    delivery_id=operation_id,
                    executor_factory=self.executor_factory,
                    execution_context=execution_context,
                    link_requested=choice.get("link_requested", False),
                )

            result = await _call_sync_or_async(
                self.agent.run,
                conversation_id=conversation_id,
                delivery_id=operation_id,
                history=history,
                target_doc_id=target_doc_id,
                executor_factory=self.executor_factory,
                execution_context=execution_context,
            )
            if result.get("status") != "ok":
                raise RuntimeError("Telegram agent failed")
            answer = result.get("answer")
            if not isinstance(answer, str):
                raise RuntimeError("Telegram agent returned no answer")
            return answer

        return run_agent

    async def _process_operation(
        self, *, operation_id: str, context: Any, replay: _CommandReplay,
        route: Callable[..., Any], run_agent: Callable[..., Any], lease: Any,
    ) -> Any:
        """Invoke the shared session flow with callbacks and the current lease."""
        from services.connectors.telegram.sessions import process_operation

        lease_now = self.clock()
        return await process_operation(
            operation_id=operation_id, context=context,
            text=replay.text, repository=self.repository, router=route, agent=run_agent,
            notifications=self.notifications, queue=self.queue,
            selection_publisher=self.selection_publisher,
            owner=_field(lease, "owner"), fence=_field(lease, "fence"), now=lease_now,
        )

    @staticmethod
    def _map_queue_state(result: Any) -> Any:
        """Map session outcomes to the queue's persisted terminal or retry state."""
        from core.operations.queue import QueueOperationState

        status = result.get("status") if isinstance(result, Mapping) else None
        if status in {"completed", "switched", "ok", "processed"}:
            return QueueOperationState.COMPLETED
        if status in {"switch_notice_pending", "switch_notice_undelivered"}:
            return QueueOperationState.RUNNING
        return QueueOperationState.FAILED
