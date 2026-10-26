"""Structured routing for Telegram command messages.

The router deliberately treats the model as the sole source of intent. This
module validates its response and resolves references against current dialog
metadata before allowing connector effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from core.operations.registry import OPERATIONS, validate_arguments


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


async def route_message(
    *, text: str, context: Any, model: Any, store: Any, search: Any,
    notices: Any, actions: Any, operation_id: str | None = None,
    owner: str | None = None, fence: int | None = None,
    now: datetime | None = None, save_candidates: Any = None,
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
    try:
        if operation_candidates:
            raw = await model.route(
                text=text, candidates=cards,
                operation_candidates=tuple(operation_candidates.values()),
            )
        else:
            raw = await model.route(text=text, candidates=cards)
    except Exception:
        return _failure("error")
    decision, status = _valid_response(raw)
    if decision is None:
        return _failure(status)
    kind = decision["kind"]
    if kind == "search":
        error = None
        try:
            refs = list(await search.search(text))
        except Exception:
            refs, error = [], "search unavailable"
        session_id = store.create_search_session(
            context_id=context.context_id, query=text, result_refs=refs, error=error,
        )
        return {"status": "ok", "kind": kind, "result_refs": refs,
                "search_session_id": session_id}

    # Saving is bound to the accepted command's captured candidate, never to
    # the model's document target or a stale context.last_doc_id hint.
    if kind == "save":
        confirmed = confirm_save_route(
            store, context_id=context.context_id, operation_id=operation_id,
            owner=owner, fence=fence, now=now,
            route_result={"kind": "save"},
        )
        return {"status": "ok" if confirmed else "unconfirmed", "kind": kind,
                "confirmed": confirmed}

    target_operation_id = decision["target_operation_id"]
    dependency_operation_id = None
    if target_operation_id is not None and decision["dialog_number"] is not None:
        # A concrete dialog number has priority over an unavailable operation
        # candidate. It can still carry the dependency if both resolve to the
        # exact same fixed save target.
        target, number, failure = _resolve_target(
            decision, context=context, store=store, candidates=candidates,
        )
        if failure:
            return _failure(failure, kind=kind)
        fixed_target = operation_candidates.get(target_operation_id)
        if (fixed_target is not None and fixed_target["doc_id"] == target
                and fixed_target["dialog_number"] == number):
            conversation_id = fixed_target["conversation_id"]
            dependency_operation_id = target_operation_id
        else:
            metadata = _metadata(store, target)
            conversation_id = metadata.get("conversation_id") if metadata else None
            if not isinstance(conversation_id, str) or not conversation_id:
                return _failure("invalid_route", kind=kind)
    elif target_operation_id is not None:
        fixed_target = operation_candidates.get(target_operation_id)
        if fixed_target is None:
            return _failure("invalid_route", kind=kind)
        concrete_ref = decision["target_ref"]
        if concrete_ref is not None and concrete_ref not in {"last", fixed_target["doc_id"]}:
            return _failure("invalid_route", kind=kind)
        target = fixed_target["doc_id"]
        number = fixed_target["dialog_number"]
        conversation_id = fixed_target["conversation_id"]
        dependency_operation_id = target_operation_id
    else:
        target, number, failure = _resolve_target(
            decision, context=context, store=store, candidates=candidates,
        )
        if failure:
            return _failure(failure, kind=kind)
        metadata = _metadata(store, target)
        conversation_id = metadata.get("conversation_id") if metadata is not None else None
        if not isinstance(conversation_id, str) or not conversation_id:
            return _failure("invalid_route", kind=kind)
        if type(number) is not int or number < 1:
            return _failure("invalid_route", kind=kind)
    result = {
        "status": "ok", "kind": kind, "target_doc_id": target,
        "dialog_number": number, "conversation_id": conversation_id,
        "substantive_text": (None if kind == "pure_switch"
                              else decision["substantive_text"]),
        "tool_choice": decision["tool_choice"],
        "link_requested": bool(decision["tool_choice"] and
                                decision["tool_choice"]["link_requested"]),
    }
    if dependency_operation_id is not None:
        result["target_operation_id"] = dependency_operation_id
        result["dependency_operation_id"] = dependency_operation_id
    changed_focus = context.active_conversation_id != conversation_id
    result["selection_changed"] = changed_focus
    choice = decision["tool_choice"]
    if kind == "substantive" and choice is not None:
        argument_doc_id = choice["arguments"].get("doc_id")
        if argument_doc_id is not None and argument_doc_id != target:
            return _failure("invalid_route", kind=kind)
    if kind == "pure_switch" or kind == "substantive":
        if changed_focus:
            notices.send_switch_notice(doc_id=target, dialog_number=number)
    if kind == "substantive" and choice is not None:
        try:
            result["action_result"] = await actions.execute(
                name=choice["name"], arguments=choice["arguments"],
                target_doc_id=target, link_requested=choice["link_requested"],
            )
        except Exception:
            return _failure("error", kind=kind)
    return result


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


__all__ = ["confirm_save_route", "route_message"]
