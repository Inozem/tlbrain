"""Structured routing for Telegram command messages.

The router deliberately treats the model as the sole source of intent. This
module validates its response and resolves references against current dialog
metadata before allowing connector effects.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from core.operations.registry import OPERATIONS, validate_arguments
from services.connectors.telegram.conversations import TurnRole, build_agent_prompt


_ROUTE_KINDS = frozenset({"pure_switch", "search", "substantive", "save", "retry"})


def _failure(status: str, *, kind: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if kind is not None:
        result["kind"] = kind
    return result


def _valid_response(response: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(response, Mapping):
        return None, "invalid_route"
    if set(response) != {
        "kind", "target_ref", "dialog_number", "result_refs",
        "substantive_text", "tool_choice", "certainty", "target_operation_id",
    }:
        return None, "invalid_route"
    kind = response.get("kind")
    if kind not in _ROUTE_KINDS:
        return None, "invalid_route"
    if response.get("certainty") != "certain":
        return None, "uncertain"
    target_ref = response.get("target_ref")
    if target_ref is not None and not isinstance(target_ref, str):
        return None, "invalid_route"
    target_operation_id = response.get("target_operation_id")
    if target_operation_id is not None and (
        not isinstance(target_operation_id, str) or not target_operation_id
    ):
        return None, "invalid_route"
    dialog_number = response.get("dialog_number")
    if dialog_number is not None and (
        isinstance(dialog_number, bool) or not isinstance(dialog_number, int) or dialog_number < 1
    ):
        return None, "invalid_route"
    text = response.get("substantive_text")
    if text is not None and not isinstance(text, str):
        return None, "invalid_route"
    result_refs = response.get("result_refs", [])
    if not isinstance(result_refs, list) or any(not isinstance(item, str) for item in result_refs):
        return None, "invalid_route"
    tool_choice = response.get("tool_choice")
    if tool_choice is not None:
        if not isinstance(tool_choice, Mapping) or set(tool_choice) - {
            "name", "arguments", "link_requested"
        }:
            return None, "invalid_route"
        name = tool_choice.get("name")
        if not isinstance(name, str) or name not in OPERATIONS:
            return None, "invalid_route"
        try:
            arguments = validate_arguments(name, tool_choice.get("arguments"))
        except (TypeError, ValueError):
            return None, "invalid_route"
        link_requested = tool_choice.get("link_requested", False)
        if not isinstance(link_requested, bool):
            return None, "invalid_route"
        if link_requested and name != "get_transcript":
            return None, "invalid_route"
        tool_choice = {"name": name, "arguments": arguments,
                       "link_requested": link_requested}
    if kind != "substantive" and tool_choice is not None:
        return None, "invalid_route"
    if target_operation_id is not None and kind not in {"pure_switch", "substantive"}:
        return None, "invalid_route"
    if kind == "substantive":
        if not isinstance(text, str) or not text.strip():
            return None, "invalid_route"
    return {
        "kind": kind,
        "target_ref": target_ref,
        "target_operation_id": target_operation_id,
        "dialog_number": dialog_number,
        "result_refs": list(result_refs),
        "substantive_text": text if kind == "substantive" else None,
        "tool_choice": tool_choice,
    }, "ok"


_CARD_FIELDS = (
    "title", "date", "dialog_date", "created_at", "folder", "folder_id",
    "folder_path",
)


def _candidate_cards(store: Any) -> tuple[dict[str, Any], ...]:
    """Build model-visible cards from an explicit metadata allowlist."""
    candidates = store.existing_dialog_candidates()
    if not isinstance(candidates, (tuple, list)):
        return ()
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            doc_id, metadata = candidate, {}
        elif isinstance(candidate, Mapping):
            doc_id = candidate.get("doc_id")
            metadata = candidate
        else:
            continue
        if not isinstance(doc_id, str) or doc_id in seen:
            continue
        seen.add(doc_id)
        card = {"doc_id": doc_id}
        for field in _CARD_FIELDS:
            value = metadata.get(field)
            if field == "created_at" and isinstance(value, datetime):
                value = value.isoformat()
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                card[field] = value
            elif field in {"folder_path"} and isinstance(value, (tuple, list)) and all(
                isinstance(item, str) for item in value
            ):
                card[field] = list(value)
        cards.append(card)
    return tuple(cards)


def _fixed_save_target(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract one fully fixed target from an already confirmed save route."""
    operation_id = operation.get("operation_id")
    route = operation.get("route_result")
    if (
        not isinstance(operation_id, str) or not operation_id
        or operation.get("state") != "completed"
        or operation.get("save_operation_id") != operation_id
        or not isinstance(route, Mapping) or route.get("kind") != "save"
    ):
        return None
    nested_target = route.get("target")
    nested_target = nested_target if isinstance(nested_target, Mapping) else {}
    doc_values = (
        operation.get("target_doc_id"), route.get("target_doc_id"),
        nested_target.get("doc_id"), operation.get("doc_id"),
    )
    doc_ids = {value for value in doc_values if isinstance(value, str) and value}
    if any(value is not None and (not isinstance(value, str) or not value) for value in doc_values):
        return None
    if len(doc_ids) != 1:
        return None
    number_values = (
        operation.get("target_dialog_number"), route.get("target_dialog_number"),
        nested_target.get("dialog_number"), operation.get("dialog_number"),
    )
    numbers = {value for value in number_values if type(value) is int and value > 0}
    if any(value is not None and (type(value) is not int or value < 1)
           for value in number_values) or len(numbers) != 1:
        return None
    conversation_values = (
        operation.get("target_conversation_id"), route.get("target_conversation_id"),
        nested_target.get("conversation_id"), operation.get("conversation_id"),
    )
    conversations = {value for value in conversation_values if isinstance(value, str) and value}
    if any(value is not None and (not isinstance(value, str) or not value)
           for value in conversation_values) or len(conversations) != 1:
        return None
    return {
        "operation_id": operation_id,
        "doc_id": next(iter(doc_ids)),
        "dialog_number": next(iter(numbers)),
        "conversation_id": next(iter(conversations)),
        **{
            field: operation[field]
            for field in _CARD_FIELDS
            if isinstance(operation.get(field), (str, int, float))
            and not isinstance(operation.get(field), bool)
        },
    }


def _safe_save_candidates(save_candidates: Any) -> dict[str, dict[str, Any]]:
    """Sanitize candidates supplied from a trusted snapshot and routed chain.

    The caller must derive this sequence from the accepted immutable operation
    snapshot and its context-local predecessor chain. It should contain only
    predecessor records; route text and transcript content are discarded here.
    """
    if not isinstance(save_candidates, (tuple, list)):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in save_candidates:
        if not isinstance(item, Mapping):
            return {}
        candidate = _fixed_save_target(item)
        if candidate is None or candidate["operation_id"] in result:
            return {}
        result[candidate["operation_id"]] = candidate
    return result


def _doc_for_number(store: Any, context_id: str, number: int) -> str | None:
    resolver = getattr(store, "resolve_dialog_number", None)
    if callable(resolver):
        value = resolver(context_id, number)
        return value if isinstance(value, str) else None
    dialogs = getattr(store, "dialogs", None)
    if isinstance(dialogs, Mapping):
        scoped = dialogs.get(context_id, {})
        if isinstance(scoped, Mapping):
            value = scoped.get(number)
            return value if isinstance(value, str) else None
    return None


def _metadata(store: Any, doc_id: str) -> Mapping[str, Any] | None:
    try:
        value = store.get_doc(doc_id)
    except Exception:
        return None
    return value if isinstance(value, Mapping) else None


def _current_doc(store: Any, candidates: tuple[str, ...], context: Any) -> str | None:
    conversation_id = context.active_conversation_id
    if not isinstance(conversation_id, str):
        return None
    for doc_id in candidates:
        info = _metadata(store, doc_id)
        if info is not None and info.get("conversation_id") == conversation_id:
            return doc_id
    return None


def _resolve_target(
    decision: Mapping[str, Any], *, context: Any, store: Any, candidates: tuple[str, ...]
) -> tuple[str | None, int | None, str | None]:
    """Return target ID, stable number, and safe failure status."""
    try:
        context_id = context.context_id
        number = decision["dialog_number"]
        # An explicit number is the strongest target signal, even if semantic
        # metadata points somewhere else. Unknown numbers never fall back.
        if number is not None:
            target = _doc_for_number(store, context_id, number)
            if target is None or target not in candidates:
                return None, None, "not_found"
        else:
            ref = decision["target_ref"]
            if ref == "last":
                target = context.last_doc_id
                if not isinstance(target, str) or target not in candidates:
                    return None, None, "not_found"
            elif ref is not None:
                target = ref
                if target not in candidates:
                    return None, None, "not_found"
            else:
                target = _current_doc(store, candidates, context)
                if target is None:
                    return None, None, "not_found"
        metadata = _metadata(store, target)
        if metadata is None:
            return None, None, "not_found"
        conversation_id = metadata.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            return None, None, "invalid_route"
        if number is None:
            number = store.get_or_assign_dialog_number(context_id, target)
        if type(number) is not int or number < 1:
            return None, None, "invalid_route"
        return target, number, None
    except Exception:
        return None, None, "error"


async def _request_route(
    *, text: str, model: Any, cards: tuple[dict[str, Any], ...],
    operation_candidates: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Ask the model for one structured route, with sanitized save targets."""
    try:
        if operation_candidates:
            raw = await model.route(
                text=text, candidates=cards,
                operation_candidates=tuple(operation_candidates.values()),
            )
        else:
            raw = await model.route(text=text, candidates=cards)
    except Exception:
        return None, "error"
    return _valid_response(raw)


async def _run_search_route(
    *, text: str, context: Any, store: Any, search: Any,
) -> dict[str, Any]:
    """Search in isolation and record the result session without switching focus."""
    error = None
    try:
        refs = list(await search.search(text))
    except Exception:
        refs, error = [], "search unavailable"
    session_id = store.create_search_session(
        context_id=context.context_id, query=text, result_refs=refs, error=error,
    )
    return {
        "status": "ok", "kind": "search", "result_refs": refs,
        "search_session_id": session_id,
    }


def _confirm_save_decision(
    *, context: Any, store: Any, operation_id: str | None, owner: str | None,
    fence: int | None, now: datetime | None,
) -> dict[str, Any]:
    """Confirm a model-selected save through the fenced repository boundary."""
    confirmed = confirm_save_route(
        store, context_id=context.context_id, operation_id=operation_id,
        owner=owner, fence=fence, now=now,
        route_result={"kind": "save"},
    )
    return {
        "status": "ok" if confirmed else "unconfirmed", "kind": "save",
        "confirmed": confirmed,
    }


def _resolved_target(
    target: str, number: int, conversation_id: str,
    dependency_operation_id: str | None = None,
) -> dict[str, Any]:
    """Package a validated target and its optional operation dependency."""
    result = {
        "target_doc_id": target,
        "dialog_number": number,
        "conversation_id": conversation_id,
    }
    if dependency_operation_id is not None:
        result["target_operation_id"] = dependency_operation_id
        result["dependency_operation_id"] = dependency_operation_id
    return result


def _resolve_route_target(
    decision: Mapping[str, Any], *, context: Any, store: Any,
    candidates: tuple[str, ...],
    operation_candidates: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve explicit numbers, semantic targets, and confirmed save targets."""
    target_operation_id = decision["target_operation_id"]
    if target_operation_id is not None and decision["dialog_number"] is not None:
        target, number, failure = _resolve_target(
            decision, context=context, store=store, candidates=candidates,
        )
        if failure:
            return None, failure
        fixed_target = operation_candidates.get(target_operation_id)
        if (fixed_target is not None and fixed_target["doc_id"] == target
                and fixed_target["dialog_number"] == number):
            conversation_id = fixed_target["conversation_id"]
            dependency_operation_id = target_operation_id
        else:
            metadata = _metadata(store, target)
            conversation_id = metadata.get("conversation_id") if metadata else None
            if not isinstance(conversation_id, str) or not conversation_id:
                return None, "invalid_route"
            dependency_operation_id = None
        return _resolved_target(
            target, number, conversation_id, dependency_operation_id,
        ), None

    if target_operation_id is not None:
        fixed_target = operation_candidates.get(target_operation_id)
        if fixed_target is None:
            return None, "invalid_route"
        concrete_ref = decision["target_ref"]
        if concrete_ref is not None and concrete_ref not in {
            "last", fixed_target["doc_id"],
        }:
            return None, "invalid_route"
        return _resolved_target(
            fixed_target["doc_id"], fixed_target["dialog_number"],
            fixed_target["conversation_id"], target_operation_id,
        ), None

    target, number, failure = _resolve_target(
        decision, context=context, store=store, candidates=candidates,
    )
    if failure:
        return None, failure
    metadata = _metadata(store, target)
    conversation_id = metadata.get("conversation_id") if metadata is not None else None
    if not isinstance(conversation_id, str) or not conversation_id:
        return None, "invalid_route"
    if type(number) is not int or number < 1:
        return None, "invalid_route"
    return _resolved_target(target, number, conversation_id), None


def _action_matches_target(decision: Mapping[str, Any], target: str) -> bool:
    """Check that an operation's explicit document agrees with the route target."""
    choice = decision["tool_choice"]
    if decision["kind"] != "substantive" or choice is None:
        return True
    argument_doc_id = choice["arguments"].get("doc_id")
    return argument_doc_id is None or argument_doc_id == target


async def _finish_targeted_route(
    decision: Mapping[str, Any], *, target: Mapping[str, Any],
    context: Any, notices: Any, actions: Any, notify_switch: bool,
) -> dict[str, Any]:
    """Build a targeted result, notify focus changes, then run its action."""
    kind = decision["kind"]
    target_doc_id = target["target_doc_id"]
    number = target["dialog_number"]
    conversation_id = target["conversation_id"]
    changed_focus = context.active_conversation_id != conversation_id
    result = {
        "status": "ok", "kind": kind,
        "target_doc_id": target_doc_id,
        "dialog_number": number,
        "conversation_id": conversation_id,
        "substantive_text": (
            None if kind == "pure_switch" else decision["substantive_text"]
        ),
        "tool_choice": decision["tool_choice"],
        "link_requested": bool(
            decision["tool_choice"] and decision["tool_choice"]["link_requested"]
        ),
    }
    dependency_operation_id = target.get("dependency_operation_id")
    if dependency_operation_id is not None:
        result["target_operation_id"] = dependency_operation_id
        result["dependency_operation_id"] = dependency_operation_id
    result["selection_changed"] = changed_focus
    result["notice_sent"] = bool(
        notify_switch and changed_focus and kind in {"pure_switch", "substantive"}
    )
    if not _action_matches_target(decision, target_doc_id):
        return _failure("invalid_route", kind=kind)
    if notify_switch and kind in {"pure_switch", "substantive"} and changed_focus:
        notices.send_switch_notice(doc_id=target_doc_id, dialog_number=number)
    if kind == "substantive" and decision["tool_choice"] is not None:
        choice = decision["tool_choice"]
        try:
            result["action_result"] = await actions.execute(
                name=choice["name"], arguments=choice["arguments"],
                target_doc_id=target_doc_id,
                link_requested=choice["link_requested"],
            )
        except Exception:
            return _failure("error", kind=kind)
    return result


async def route_message(
    *, text: str, context: Any, model: Any, store: Any, search: Any,
    notices: Any, actions: Any, operation_id: str | None = None,
    owner: str | None = None, fence: int | None = None,
    now: datetime | None = None, save_candidates: Any = None,
    notify_switch: bool = True,
) -> dict[str, Any]:
    """Route one command using only a validated structured model response.

    ``save_candidates`` is a server-built immutable sequence derived from the
    accepted command's snapshot and routed predecessor chain. Each item must
    be a confirmed save operation with a fixed document, number, and
    conversation target. This seam validates and sanitizes those records but
    does not load history or persistence state itself.
    """
    cards = _candidate_cards(store)
    candidates = tuple(card["doc_id"] for card in cards)
    operation_candidates = _safe_save_candidates(save_candidates)
    decision, status = await _request_route(
        text=text, model=model, cards=cards,
        operation_candidates=operation_candidates,
    )
    if decision is None:
        return _failure(status)
    kind = decision["kind"]
    if kind == "search":
        return await _run_search_route(
            text=text, context=context, store=store, search=search,
        )

    if kind == "save":
        return _confirm_save_decision(
            context=context, store=store, operation_id=operation_id,
            owner=owner, fence=fence, now=now,
        )

    target, failure = _resolve_route_target(
        decision, context=context, store=store, candidates=candidates,
        operation_candidates=operation_candidates,
    )
    if failure:
        return _failure(failure, kind=kind)
    return await _finish_targeted_route(
        decision, target=target, context=context, notices=notices, actions=actions,
        notify_switch=notify_switch,
    )


def _control_route_record(route: Any) -> dict[str, Any]:
    """Keep only target and status metadata in the operation control ledger."""
    if not isinstance(route, Mapping):
        return {"status": "invalid_route"}
    safe: dict[str, Any] = {}
    allowed_statuses = {
        "ok", "success", "invalid_route", "uncertain", "error",
        "not_found", "unconfirmed",
    }
    allowed_kinds = _ROUTE_KINDS
    for field in (
        "status", "kind", "target_doc_id", "conversation_id",
        "target_draft_id", "target_operation_id", "dependency_operation_id",
    ):
        value = route.get(field)
        if field == "status" and (not isinstance(value, str) or value not in allowed_statuses):
            continue
        if field == "kind" and (not isinstance(value, str) or value not in allowed_kinds):
            continue
        if isinstance(value, str) and value:
            safe[field] = value
    number = route.get("dialog_number")
    if type(number) is int and number > 0:
        safe["dialog_number"] = number
    safe.setdefault("status", "error")
    return safe


def _operation_route_status(route: Any) -> str:
    """Validate the operation route before recording a successful status."""
    if not isinstance(route, Mapping):
        return "invalid_route"
    status = route.get("status")
    if status is not None and (not isinstance(status, str) or status not in {"ok", "success"}):
        return status if isinstance(status, str) and status in {
            "invalid_route", "uncertain", "error", "not_found", "unconfirmed",
        } else "invalid_route"
    kind = route.get("kind")
    if not isinstance(kind, str) or kind not in {
        "pure_switch", "substantive", "search", "save", "retry",
    }:
        return "invalid_route"
    if kind == "search":
        return "ok"
    if kind == "save":
        if type(route.get("confirmed")) is not bool:
            return "invalid_route"
        return "ok" if route.get("confirmed") else "unconfirmed"
    if (
        not isinstance(route.get("target_doc_id"), str)
        or not route.get("target_doc_id")
        or not isinstance(route.get("conversation_id"), str)
        or not route.get("conversation_id")
        or type(route.get("dialog_number")) is not int
        or route.get("dialog_number") < 1
    ):
        return "invalid_route"
    if kind == "substantive":
        value = route.get("substantive_text")
        if not isinstance(value, str) or not value.strip():
            return "invalid_route"
    return "ok"


@dataclass(frozen=True)
class _PersistedOperationRoute:
    """Carry a sanitized route and the lease-bound candidate cleanup handle."""

    route: Any
    route_record: dict[str, Any]
    context_id: Any
    clear_candidate: Callable[..., Any] | None


async def _invoke_operation_router(
    *, router: Any, operation_id: str, context: Any, text: str,
) -> Any:
    """Resolve the structured route before this seam reads or changes history."""
    try:
        return await router(
            operation_id=operation_id, context=context, text=text,
        )
    except Exception:
        return {"status": "error"}


def _persist_operation_route(
    route: Any, *, operation_id: str, context: Any, repository: Any,
    owner: str | None, fence: int | None, now: datetime | None,
) -> _PersistedOperationRoute | None:
    """Validate and persist only allowlisted route metadata before any side effect."""
    validation_status = _operation_route_status(route)
    route_record = _control_route_record(route)
    route_record["status"] = validation_status
    clear_candidate = getattr(repository, "clear_command_candidate", None)
    context_id = None
    if callable(clear_candidate):
        context_id = getattr(context, "context_id", None)
        if (
            not isinstance(context_id, str) or not context_id
            or not isinstance(owner, str) or not owner
            or type(fence) is not int or fence < 1
            or not isinstance(now, datetime)
        ):
            return None
        persisted = repository.persist_route(
            operation_id, route_record, context_id=context_id,
            owner=owner, fence=fence, now=now,
        )
    else:
        persisted = repository.persist_route(operation_id, route_record)
    if persisted is False:
        return None
    return _PersistedOperationRoute(
        route=route, route_record=route_record, context_id=context_id,
        clear_candidate=clear_candidate if callable(clear_candidate) else None,
    )


def _route_failure_result(state: _PersistedOperationRoute) -> dict[str, Any] | None:
    """Return a persisted routing failure without entering a control flow."""
    route_record = state.route_record
    if route_record.get("status") in {"ok", "success"}:
        return None
    if route_record.get("kind") == "save":
        confirmed = state.route.get("confirmed") if isinstance(state.route, Mapping) else False
        return {
            "status": route_record["status"], "kind": "save",
            "confirmed": confirmed is True,
        }
    return route_record


def _clear_command_candidate(
    state: _PersistedOperationRoute, *, operation_id: str,
    owner: str | None, fence: int | None, now: datetime | None,
) -> bool:
    """Clear the raw envelope only through the previously validated live lease."""
    if state.clear_candidate is None:
        return True
    return bool(state.clear_candidate(
        state.context_id, operation_id, owner=owner, fence=fence, now=now,
    ))


def _handle_control_route(
    state: _PersistedOperationRoute, *, kind: Any, operation_id: str,
    owner: str | None, fence: int | None, now: datetime | None,
) -> dict[str, Any] | None:
    """Finish search, save, and retry routes without creating conversation turns."""
    if kind == "search":
        result: dict[str, Any] = {"status": "ok", "kind": "search"}
        route = state.route
        refs = route.get("result_refs") if isinstance(route, Mapping) else None
        session_id = route.get("search_session_id") if isinstance(route, Mapping) else None
        if isinstance(refs, list) and all(isinstance(item, str) for item in refs):
            result["result_refs"] = list(refs)
        if isinstance(session_id, str) and session_id:
            result["search_session_id"] = session_id
        if not _clear_command_candidate(
            state, operation_id=operation_id, owner=owner, fence=fence, now=now,
        ):
            return {"status": "error", "kind": "search"}
        return result
    if kind in {"save", "retry"}:
        if not _clear_command_candidate(
            state, operation_id=operation_id, owner=owner, fence=fence, now=now,
        ):
            return {"status": "error", "kind": kind}
        if kind == "save":
            return {"status": "ok", "kind": "save", "confirmed": True}
        return {"status": "ok", "kind": kind}
    return None


def _conversation_route_target(
    route_record: Mapping[str, Any],
) -> tuple[str, str, int] | None:
    """Extract the fixed target accepted by successful conversation routes."""
    kind = route_record.get("kind")
    target_doc_id = route_record.get("target_doc_id")
    conversation_id = route_record.get("conversation_id")
    number = route_record.get("dialog_number")
    if (
        kind not in {"pure_switch", "substantive"}
        or not isinstance(target_doc_id, str)
        or not isinstance(conversation_id, str)
        or type(number) is not int
    ):
        return None
    return target_doc_id, conversation_id, number


def _substantive_route_text(route: Any, kind: str) -> str | None:
    """Accept text only for substantive routes; switches never retain payloads."""
    substantive_text = (
        route.get("substantive_text")
        if kind == "substantive" and isinstance(route, Mapping)
        else None
    )
    if kind == "substantive" and (
        not isinstance(substantive_text, str) or not substantive_text.strip()
    ):
        return None
    return substantive_text


async def _confirm_selection_change(
    *, route: Any, operation_id: str, target_doc_id: str,
    conversation_id: str, number: int, context: Any, notifications: Any,
    selection_publisher: Callable[..., Any] | None, repository: Any = None,
    owner: str | None = None, fence: int | None = None,
    now: datetime | None = None,
) -> bool | str | None:
    """Confirm the notice before publishing a changed selection, unless already sent."""
    context_id = getattr(context, "context_id", None)
    set_notice_state = getattr(repository, "set_switch_notice_state", None)
    prior_notice_state = None
    getter = getattr(repository, "get_switch_notice_state", None)
    if callable(getter) and isinstance(context_id, str):
        try:
            prior_notice_state = getter(context_id, operation_id)
        except Exception:
            prior_notice_state = None
    records = getattr(repository, "route_records", None)
    if prior_notice_state is None and isinstance(records, Mapping):
        prior = records.get(operation_id)
        if isinstance(prior, Mapping):
            prior_notice_state = prior.get("switch_notice_state")
    if (context.active_conversation_id == conversation_id
            and prior_notice_state != "pending"):
        return False
    def remember_notice_state(value: str) -> None:
        records = getattr(repository, "route_records", None)
        if isinstance(records, dict) and isinstance(records.get(operation_id), dict):
            records[operation_id]["switch_notice_state"] = value
        states = getattr(repository, "switch_notice_states", None)
        if not isinstance(states, dict):
            try:
                states = {}
                setattr(repository, "switch_notice_states", states)
            except Exception:
                states = None
        if isinstance(states, dict):
            states[operation_id] = value
    if prior_notice_state == "delivered":
        notice_result = "confirmed"
    elif callable(set_notice_state) and isinstance(context_id, str):
        if isinstance(owner, str) and type(fence) is int and isinstance(now, datetime):
            persisted = set_notice_state(
                context_id, operation_id, "pending", owner=owner, fence=fence, now=now,
            )
            if persisted is False:
                return "pending"
        else:
            # Lightweight repositories used by offline callers may not expose
            # the fenced Firestore seam.
            remember_notice_state("pending")
    notification_id = f"switch-{operation_id}"
    if prior_notice_state != "delivered":
        try:
            notice_result = notifications.send_switch_notice(
                context_id=context_id,
                operation_id=operation_id,
                notification_id=notification_id,
                target_doc_id=target_doc_id,
                dialog_number=number,
            )
            if inspect.isawaitable(notice_result):
                notice_result = await notice_result
        except Exception:
            # An exception does not prove a terminal ledger outcome. Keep the
            # route pinned and let the same operation resume through its notice ID.
            remember_notice_state("pending")
            return "pending"
    if notice_result in {"busy", "retry", "pending", "sending"}:
        remember_notice_state("pending")
        return "pending"
    if notice_result in {"failed", "delivery_unknown"}:
        disposition = notice_result
        if callable(set_notice_state) and isinstance(context_id, str) and isinstance(owner, str) and type(fence) is int and isinstance(now, datetime):
            persisted = set_notice_state(context_id, operation_id, disposition, owner=owner, fence=fence, now=now)
            if persisted is False:
                return "pending"
        else:
            remember_notice_state(disposition)
        return None
    if notice_result != "confirmed":
        remember_notice_state("pending")
        return "pending"
    if callable(set_notice_state) and isinstance(context_id, str) and isinstance(owner, str) and type(fence) is int and isinstance(now, datetime):
        persisted = set_notice_state(
            context_id, operation_id, "delivered", owner=owner, fence=fence, now=now,
        )
        if persisted is False:
            return "pending"
    else:
        remember_notice_state("delivered")
    if context.active_conversation_id == conversation_id:
        return False
    if selection_publisher is not None:
        published = selection_publisher(
            operation_id, conversation_id=conversation_id,
            dialog_number=number,
        )
        if inspect.isawaitable(published):
            await published
    return True


async def _process_substantive_turn(
    *, operation_id: str, context_id: str, kind: str, substantive_text: str,
    target_doc_id: str, conversation_id: str, number: int, repository: Any,
    agent: Any, state: _PersistedOperationRoute, owner: str | None,
    fence: int | None, now: datetime | None,
) -> dict[str, Any]:
    """Append the substantive question once, skip completed replays, and run its pinned target."""
    history = repository.list_turns(context_id, conversation_id)
    prompt_history = [
        turn for turn in history
        if not isinstance(turn, Mapping) or turn.get("operation_id") != operation_id
    ]
    completed = any(
        isinstance(turn, Mapping)
        and turn.get("operation_id") == operation_id
        and turn.get("role") == TurnRole.MODEL.value
        for turn in history
    )
    if completed:
        if not _clear_command_candidate(
            state, operation_id=operation_id, owner=owner, fence=fence, now=now,
        ):
            return {"status": "error", "kind": kind}
        return {
            "status": "completed", "kind": kind,
            "target_doc_id": target_doc_id,
            "conversation_id": conversation_id,
            "dialog_number": number,
        }
    prompt = build_agent_prompt(prompt_history, substantive_text)
    turn_now = now or datetime.now(timezone.utc)
    turn_id = lambda role: "telegram-" + hashlib.sha256(
        f"{operation_id}:{role}".encode("utf-8")
    ).hexdigest()
    repository.append_turn(
        context_id, conversation_id,
        turn_id=turn_id(TurnRole.USER.value), operation_id=operation_id,
        role=TurnRole.USER.value, parts=[{"text": substantive_text}], now=turn_now,
    )
    if not _clear_command_candidate(
        state, operation_id=operation_id, owner=owner, fence=fence, now=now,
    ):
        return {"status": "error", "kind": kind}
    try:
        answer = await agent(prompt=prompt, target_doc_id=target_doc_id)
    except Exception:
        return {"status": "error", "kind": kind}
    repository.append_turn(
        context_id, conversation_id,
        turn_id=turn_id(TurnRole.MODEL.value), operation_id=operation_id,
        role=TurnRole.MODEL.value, parts=[{"text": answer}], now=turn_now,
    )
    return {
        "status": "completed", "kind": kind,
        "target_doc_id": target_doc_id,
        "conversation_id": conversation_id,
        "dialog_number": number,
    }


def _load_operation_replay_state(
    repository: Any, *, context_id: str, operation_id: str,
) -> tuple[Any, Any]:
    """Load the saved route and switch-notice disposition for an operation replay."""
    get_route = getattr(repository, "get_route_control", None)
    existing_route = None
    if callable(get_route):
        try:
            existing_route = get_route(context_id, operation_id)
        except Exception:
            existing_route = None
    else:
        records = getattr(repository, "route_records", None)
        if isinstance(records, Mapping):
            existing_route = records.get(operation_id)

    notice_states = getattr(repository, "switch_notice_states", None)
    notice_state = notice_states.get(operation_id) if isinstance(notice_states, Mapping) else None
    if notice_state is None and isinstance(existing_route, Mapping):
        notice_state = existing_route.get("switch_notice_state")
    if notice_state is None:
        getter = getattr(repository, "get_switch_notice_state", None)
        if callable(getter):
            try:
                notice_state = getter(context_id, operation_id)
            except Exception:
                notice_state = None
    return existing_route, notice_state


async def _route_operation_replay(
    *, operation_id: str, context: Any, text: str, router: Any,
    existing_route: Any, notice_state: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Reuse pinned routes where possible and reject replay target changes."""
    if notice_state in {"failed", "delivery_unknown"}:
        return None, {
            "status": "switch_notice_undelivered",
            "kind": (existing_route or {}).get("kind"),
        }
    if (notice_state == "pending" and isinstance(existing_route, Mapping)
            and existing_route.get("kind") == "pure_switch"):
        # A pure switch has all the route data needed to resume from the
        # pinned control record, so it can retry without consulting the router.
        return dict(existing_route), None

    route = await _invoke_operation_router(
        router=router, operation_id=operation_id, context=context, text=text,
    )
    if isinstance(existing_route, Mapping):
        old_target = _conversation_route_target(existing_route)
        new_target = _conversation_route_target(_control_route_record(route))
        if old_target is not None and new_target != old_target:
            return None, {"status": "invalid_route", "kind": existing_route.get("kind")}
        if notice_state == "pending" and old_target is None:
            return None, {"status": "switch_notice_pending", "kind": existing_route.get("kind")}
    return route, None


async def _release_notice_waiter(queue: Any, operation_id: str) -> None:
    """Release the queue lease when a switch notice needs a later replay."""
    if queue is None:
        return
    release = getattr(queue, "release_head", None)
    if not callable(release):
        return
    released = release(operation_id)
    if inspect.isawaitable(released):
        await released


async def _wait_for_switch_notice(
    *, route: Any, operation_id: str, target_doc_id: str,
    conversation_id: str, number: int, context: Any, notifications: Any,
    selection_publisher: Callable[..., Any] | None, repository: Any,
    queue: Any, owner: str | None, fence: int | None, now: datetime | None,
    kind: Any,
) -> dict[str, Any] | None:
    """Return a retry/error result until the notice barrier allows selection."""
    changed_focus = await _confirm_selection_change(
        route=route, operation_id=operation_id,
        target_doc_id=target_doc_id, conversation_id=conversation_id,
        number=number, context=context, notifications=notifications,
        selection_publisher=selection_publisher, repository=repository,
        owner=owner, fence=fence, now=now,
    )
    if changed_focus is None:
        await _release_notice_waiter(queue, operation_id)
        return {"status": "switch_notice_undelivered", "kind": kind}
    if changed_focus == "pending":
        await _release_notice_waiter(queue, operation_id)
        return {"status": "switch_notice_pending", "kind": kind}
    return None


async def _finish_conversation_route(
    *, state: _PersistedOperationRoute, kind: Any, operation_id: str,
    context: Any, repository: Any, agent: Any, notifications: Any,
    queue: Any, selection_publisher: Callable[..., Any] | None,
    owner: str | None, fence: int | None, now: datetime | None,
) -> dict[str, Any]:
    """Enforce the switch notice barrier before a pure switch or conversation turn."""
    target = _conversation_route_target(state.route_record)
    if target is None:
        return {"status": "invalid_route", **({"kind": kind} if kind else {})}
    target_doc_id, conversation_id, number = target
    substantive_text = _substantive_route_text(state.route, kind)
    if kind == "substantive" and substantive_text is None:
        return {"status": "invalid_route", "kind": kind}

    notice_result = await _wait_for_switch_notice(
        route=state.route, operation_id=operation_id,
        target_doc_id=target_doc_id, conversation_id=conversation_id,
        number=number, context=context, notifications=notifications,
        selection_publisher=selection_publisher, repository=repository,
        queue=queue, owner=owner, fence=fence, now=now, kind=kind,
    )
    if notice_result is not None:
        return notice_result

    if kind == "pure_switch":
        if not _clear_command_candidate(
            state, operation_id=operation_id, owner=owner, fence=fence, now=now,
        ):
            return {"status": "error", "kind": kind}
        return {
            "status": "switched", "kind": kind,
            "target_doc_id": target_doc_id,
            "conversation_id": conversation_id,
            "dialog_number": number,
        }

    return await _process_substantive_turn(
        operation_id=operation_id, context_id=context.context_id, kind=kind,
        substantive_text=substantive_text, target_doc_id=target_doc_id,
        conversation_id=conversation_id, number=number, repository=repository,
        agent=agent, state=state, owner=owner, fence=fence, now=now,
    )


async def process_operation(
    *, operation_id: str, context: Any, text: str, repository: Any,
    router: Any, agent: Any, notifications: Any, queue: Any = None,
    selection_publisher: Callable[..., Any] | None = None,
    owner: str | None = None, fence: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Route first, persist safe metadata, then finish controls or the pinned turn."""
    existing_route, notice_state = _load_operation_replay_state(
        repository, context_id=context.context_id, operation_id=operation_id,
    )
    route, replay_result = await _route_operation_replay(
        operation_id=operation_id, context=context, text=text, router=router,
        existing_route=existing_route, notice_state=notice_state,
    )
    if replay_result is not None:
        return replay_result
    state = _persist_operation_route(
        route, operation_id=operation_id, context=context, repository=repository,
        owner=owner, fence=fence, now=now,
    )
    if state is None:
        return {"status": "error"}
    failure = _route_failure_result(state)
    if failure is not None:
        return failure

    kind = state.route_record.get("kind")
    control_result = _handle_control_route(
        state, kind=kind, operation_id=operation_id,
        owner=owner, fence=fence, now=now,
    )
    if control_result is not None:
        return control_result
    return await _finish_conversation_route(
        state=state, kind=kind, operation_id=operation_id, context=context,
        repository=repository, agent=agent, notifications=notifications,
        queue=queue, selection_publisher=selection_publisher,
        owner=owner, fence=fence, now=now,
    )


def confirm_save_route(
    store: Any, *, context_id: str, operation_id: str | None, owner: str | None,
    fence: int | None, now: datetime | None, route_result: Mapping[str, Any],
) -> bool:
    """Confirm a validated save route through the repository's fenced boundary.

    This adapter intentionally requires the live queue identity. A resolver
    result alone cannot prove ownership of the accepted command snapshot.
    """
    if route_result.get("kind") != "save":
        return False
    if (
        not isinstance(operation_id, str) or not operation_id
        or not isinstance(owner, str) or not owner
        or isinstance(fence, bool) or not isinstance(fence, int) or fence < 1
        or not isinstance(now, datetime)
    ):
        return False
    confirm = getattr(store, "confirm_save_command", None)
    if not callable(confirm):
        return False
    return bool(confirm(context_id, operation_id, owner=owner, fence=fence, now=now))


__all__ = ["confirm_save_route", "process_operation", "route_message"]
