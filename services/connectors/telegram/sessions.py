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
from services.connectors.telegram.context import build_document_context
from services.connectors.telegram.conversations import TurnRole
from services.connectors.telegram.save import TelegramSavePending
from services.connectors.telegram.errors import TelegramDirectSaveInterrupted
from services.connectors.telegram.runtime.routing_context import (
    Action, Certainty, CommandSpan, Disposition, DocumentSelector,
    DocumentSelectorKind, ExplicitReferenceKind, MissingSlot, RoutingDecision,
)


_ROUTE_KINDS = frozenset({"pure_switch", "search", "substantive", "save", "retry", "annotation"})
_FRESH_FIELDS = frozenset({
    "disposition", "action", "substantive_text", "document_selector",
    "folder_selector", "folder_name", "explicit_references", "missing_slots",
    "evidence_refs", "clarification_id", "clarification_generation",
    "certainty", "tool_choice", "target_operation_id",
})
_LEGACY_FIELDS = frozenset({
    "kind", "target_ref", "dialog_number", "result_refs", "substantive_text",
    "tool_choice", "certainty", "target_operation_id",
})


def _failure(status: str, *, kind: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if kind is not None:
        result["kind"] = kind
    return result


def _valid_persisted_route(response: Any) -> tuple[dict[str, Any] | None, str]:
    """Validate a server-owned legacy route projection; never use for fresh model data."""
    if not isinstance(response, Mapping):
        return None, "invalid_route"
    if set(response) != _LEGACY_FIELDS:
        return None, "invalid_route"
    kind = response.get("kind")
    if not isinstance(kind, str) or kind not in _ROUTE_KINDS:
        return None, "invalid_route"
    certainty = response.get("certainty")
    if certainty != "certain" and certainty != "uncertain":
        return None, "invalid_route"
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
    if kind == "search":
        if (not isinstance(text, str) or not text.strip() or tool_choice is None
                or tool_choice["name"] != "query" or target_ref is not None
                or target_operation_id is not None or dialog_number is not None
                or result_refs):
            return None, "invalid_route"
    elif kind != "substantive" and tool_choice is not None:
        return None, "invalid_route"
    if target_operation_id is not None and kind not in {"pure_switch", "substantive"}:
        return None, "invalid_route"
    if kind == "substantive":
        if not isinstance(text, str) or not text.strip():
            return None, "invalid_route"
    if kind == "annotation":
        if (not isinstance(text, str) or not text.strip() or tool_choice is not None
                or target_operation_id is not None or result_refs):
            return None, "invalid_route"
    if kind == "retry" and type(dialog_number) is not int:
        return None, "invalid_route"
    if certainty == "uncertain":
        return None, "uncertain"
    return {
        "kind": kind,
        "target_ref": target_ref,
        "target_operation_id": target_operation_id,
        "dialog_number": dialog_number,
        "result_refs": list(result_refs),
        "substantive_text": text if kind in {"substantive", "search", "annotation"} else None,
        "tool_choice": tool_choice,
    }, "ok"


def _response_span(span: CommandSpan, accepted_command: str) -> bool:
    if span.end > len(accepted_command) or accepted_command[span.start:span.end] != span.raw_text:
        return False
    if span.kind is ExplicitReferenceKind.LOCAL_NUMBER:
        from services.connectors.telegram.runtime.store import TelegramRuntimeStore
        try:
            return TelegramRuntimeStore._reference_value(span) == span.parsed_value
        except (TypeError, ValueError):
            return False
    if span.kind is ExplicitReferenceKind.ID:
        from services.connectors.telegram.runtime.store import TelegramRuntimeStore
        try:
            return TelegramRuntimeStore._reference_value(span) == span.parsed_value
        except (TypeError, ValueError):
            return False
    if span.kind is ExplicitReferenceKind.URL:
        from services.connectors.telegram.runtime.store import TelegramRuntimeStore
        try:
            return TelegramRuntimeStore._reference_value(span) is not None
        except (TypeError, ValueError):
            return False
    return span.kind is ExplicitReferenceKind.LAST_SAVED and span.parsed_value == span.raw_text


def _strict_fresh_shapes(response: Mapping[str, Any]) -> bool:
    """Match the native schema's closed selector/span variants before DTO decoding."""
    span_fields = {"kind", "start", "end", "raw_text", "parsed_value"}
    selector_fields = {
        "explicit_reference": {"kind", "raw_value", "span"},
        "dialog_number": {"kind", "dialog_number", "span"},
        "last_saved": {"kind", "span"},
        "active_document": {"kind"},
        "evidence": {"kind", "doc_id", "evidence_refs"},
        "shown_position": {"kind", "list_id", "position"},
    }
    selector = response.get("document_selector")
    if selector is not None:
        if not isinstance(selector, Mapping):
            return False
        kind = selector.get("kind")
        if kind not in selector_fields or set(selector) != selector_fields[kind]:
            return False
        expected_span_kind = {
            "explicit_reference": {ExplicitReferenceKind.ID.value, ExplicitReferenceKind.URL.value},
            "dialog_number": {ExplicitReferenceKind.LOCAL_NUMBER.value},
            "last_saved": {ExplicitReferenceKind.LAST_SAVED.value},
        }.get(kind)
        if expected_span_kind is not None:
            span = selector.get("span")
            if not isinstance(span, Mapping) or span.get("kind") not in expected_span_kind:
                return False
    if (not isinstance(response.get("explicit_references"), list)
            or not isinstance(response.get("missing_slots"), list)
            or not isinstance(response.get("evidence_refs"), list)):
        return False
    evidence_selector = selector if isinstance(selector, Mapping) and selector.get("kind") == "evidence" else None
    if evidence_selector is not None and not evidence_selector.get("evidence_refs"):
        return False
    spans = list(response["explicit_references"])
    if selector is not None and isinstance(selector, Mapping) and "span" in selector:
        spans.append(selector["span"])
    for span in spans:
        if not isinstance(span, Mapping) or set(span) != span_fields:
            return False
        kind = span.get("kind")
        parsed = span.get("parsed_value")
        if kind == ExplicitReferenceKind.LOCAL_NUMBER.value:
            if type(parsed) is not int or parsed < 1:
                return False
        elif kind in {item.value for item in ExplicitReferenceKind}:
            if not isinstance(parsed, str) or not parsed:
                return False
        else:
            return False
    def folder_shape(folder: Any, depth: int = 0) -> bool:
        if folder is None:
            return True
        if not isinstance(folder, Mapping) or set(folder) != {"kind", "value", "parent"}:
            return False
        if depth > 1 or folder.get("kind") == "create_step_output":
            return False
        kind, value, parent = folder.get("kind"), folder.get("value"), folder.get("parent")
        if kind == "root_default":
            return value is None and parent is None
        if kind not in {"explicit_path", "explicit_id", "name"}:
            return False
        if not isinstance(value, str) or not value:
            return False
        if parent is not None:
            if depth != 0 or not isinstance(parent, Mapping):
                return False
            if set(parent) != {"kind", "value", "parent"} or parent.get("parent") is not None:
                return False
            p_kind, p_value = parent.get("kind"), parent.get("value")
            if p_kind == "root_default":
                return p_value is None
            return p_kind in {"explicit_path", "explicit_id", "name"} and isinstance(p_value, str) and bool(p_value)
        return True
    return folder_shape(response.get("folder_selector"))


@dataclass(frozen=True)
class _SemanticConflictFacts:
    """Resolved literal and local-number signals used by conflict checks."""

    explicit_ids: set[str]
    unresolved_urls: bool
    local_numbers: set[int]
    number_bindings: dict[int, set[str]]
    bound_local_ids: set[str]
    unresolved_local: bool
    last_signals: bool


def _semantic_conflict_facts(
    decision: RoutingDecision, candidates: Any,
) -> _SemanticConflictFacts:
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    spans = decision.explicit_references
    explicit_ids: set[str] = set()
    unresolved_urls = False
    for span in decision.explicit_references:
        if span.kind in (ExplicitReferenceKind.ID, ExplicitReferenceKind.URL):
            try:
                value = TelegramRuntimeStore._reference_value(span)
            except (TypeError, ValueError):
                value = None
            if isinstance(value, str):
                explicit_ids.add(value)
            else:
                unresolved_urls = True
    local_numbers = {span.parsed_value for span in spans
                     if span.kind is ExplicitReferenceKind.LOCAL_NUMBER}
    number_bindings: dict[int, set[str]] = {}
    if isinstance(candidates, (tuple, list)):
        for card in candidates:
            if (isinstance(card, Mapping) and type(card.get("dialog_number")) is int
                    and card["dialog_number"] > 0 and isinstance(card.get("doc_id"), str)):
                number_bindings.setdefault(card["dialog_number"], set()).add(card["doc_id"])
    bound_local_ids = {
        next(iter(number_bindings[number])) for number in local_numbers
        if len(number_bindings.get(number, ())) == 1
    }
    unresolved_local = any(len(number_bindings.get(number, ())) != 1 for number in local_numbers)
    last_signals = any(
        span.kind is ExplicitReferenceKind.LAST_SAVED
        for span in decision.explicit_references
    )
    return _SemanticConflictFacts(
        explicit_ids=explicit_ids,
        unresolved_urls=unresolved_urls,
        local_numbers=local_numbers,
        number_bindings=number_bindings,
        bound_local_ids=bound_local_ids,
        unresolved_local=unresolved_local,
        last_signals=last_signals,
    )


def _conflicts_between_explicit_signals(
    facts: _SemanticConflictFacts, decision: RoutingDecision,
    tool_choice: Mapping[str, Any] | None, last_doc_id: str | None,
    target_doc_id: str | None,
) -> bool:
    selector = decision.document_selector
    explicit_ids = facts.explicit_ids
    local_numbers = facts.local_numbers
    bound_local_ids = facts.bound_local_ids
    last_signals = facts.last_signals
    if facts.unresolved_urls or len(explicit_ids) > 1 or len(local_numbers) > 1:
        return True
    if facts.unresolved_local and (explicit_ids or last_signals or selector is not None
                                   and selector.kind is not DocumentSelectorKind.DIALOG_NUMBER
                                   or (tool_choice is not None
                                       and isinstance(tool_choice.get("arguments", {}).get("doc_id"), str))):
        return True
    if explicit_ids and bound_local_ids and explicit_ids != bound_local_ids:
        return True
    if explicit_ids and local_numbers and not bound_local_ids:
        return True
    if last_signals and (not isinstance(last_doc_id, str) or not last_doc_id):
        return True
    all_literal_ids = set(explicit_ids) | set(bound_local_ids)
    if last_signals and all_literal_ids and all_literal_ids != {last_doc_id}:
        return True
    if last_signals and isinstance(target_doc_id, str) and target_doc_id != last_doc_id:
        return True
    if last_signals and selector is None:
        return True
    return False


def _selector_conflicts_with_signals(
    facts: _SemanticConflictFacts, decision: RoutingDecision,
    active_doc_id: str | None, last_doc_id: str | None, target_doc_id: str | None,
) -> bool:
    selector = decision.document_selector
    if selector is None:
        return bool(facts.explicit_ids or facts.local_numbers) and (
            not isinstance(target_doc_id, str) or not target_doc_id
            or (facts.explicit_ids and facts.explicit_ids != {target_doc_id})
            or (facts.bound_local_ids and facts.bound_local_ids != {target_doc_id})
        ) or facts.last_signals

    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    if selector.kind is DocumentSelectorKind.EXPLICIT_REFERENCE:
        selected = TelegramRuntimeStore._reference_value(selector.span)
        if facts.explicit_ids and facts.explicit_ids != {selected}:
            return True
        if facts.last_signals and selected != last_doc_id:
            return True
    elif selector.kind is DocumentSelectorKind.DIALOG_NUMBER:
        if facts.local_numbers and facts.local_numbers != {selector.dialog_number}:
            return True
        bound = facts.number_bindings.get(selector.dialog_number, set())
        if len(bound) > 1 or (facts.explicit_ids and bound and facts.explicit_ids != bound):
            return True
        if facts.last_signals and (len(bound) != 1 or bound != {last_doc_id}):
            return True
    elif selector.kind is DocumentSelectorKind.LAST_SAVED:
        if ((facts.explicit_ids and facts.explicit_ids != {last_doc_id})
                or (facts.bound_local_ids and facts.bound_local_ids != {last_doc_id})):
            return True
    elif selector.kind is DocumentSelectorKind.ACTIVE_DOCUMENT:
        if (facts.explicit_ids and (not isinstance(active_doc_id, str)
                                    or facts.explicit_ids != {active_doc_id})) or (
                facts.bound_local_ids and facts.bound_local_ids != {active_doc_id}):
            return True
        if facts.last_signals and active_doc_id != last_doc_id:
            return True
    elif selector.kind in {DocumentSelectorKind.EVIDENCE,
                           DocumentSelectorKind.SHOWN_POSITION}:
        if facts.explicit_ids or facts.local_numbers or facts.last_signals:
            return True
    return False


def _tool_choice_conflicts_with_selector(
    facts: _SemanticConflictFacts, decision: RoutingDecision,
    tool_choice: Mapping[str, Any], active_doc_id: str | None,
    last_doc_id: str | None, target_doc_id: str | None,
) -> bool:
    selector = decision.document_selector
    argument_doc_id = tool_choice.get("arguments", {}).get("doc_id")
    if isinstance(argument_doc_id, str) and facts.explicit_ids:
        if facts.explicit_ids != {argument_doc_id}:
            return True
    if isinstance(argument_doc_id, str) and selector is not None:
        if selector.kind is DocumentSelectorKind.EXPLICIT_REFERENCE:
            from services.connectors.telegram.runtime.store import TelegramRuntimeStore
            selected = TelegramRuntimeStore._reference_value(selector.span)
            if selected != argument_doc_id:
                return True
        elif selector.kind is DocumentSelectorKind.DIALOG_NUMBER:
            bound = facts.number_bindings.get(selector.dialog_number, set())
            if len(bound) != 1 or argument_doc_id not in bound:
                return True
        elif selector.kind in {DocumentSelectorKind.SHOWN_POSITION,
                               DocumentSelectorKind.EVIDENCE}:
            # Equality requires resolving an independently scoped binding.
            return True
        elif selector.kind is DocumentSelectorKind.LAST_SAVED:
            if not isinstance(last_doc_id, str) or argument_doc_id != last_doc_id:
                return True
        elif selector.kind is DocumentSelectorKind.ACTIVE_DOCUMENT:
            if not isinstance(active_doc_id, str) or argument_doc_id != active_doc_id:
                return True
    elif isinstance(argument_doc_id, str) and selector is None:
        if ((OPERATIONS[tool_choice["name"]].write or target_doc_id is not None)
                and argument_doc_id != target_doc_id):
            return True
    return False


def _semantic_conflict(
    decision: RoutingDecision, tool_choice: Mapping[str, Any] | None, *,
    active_doc_id: str | None = None, last_doc_id: str | None = None,
    candidates: Any = (), target_doc_id: str | None = None,
) -> bool:
    """Detect explicit contradictions that do not need resource lookup to prove."""
    facts = _semantic_conflict_facts(decision, candidates)
    if _conflicts_between_explicit_signals(
        facts, decision, tool_choice, last_doc_id, target_doc_id
    ):
        return True
    if _selector_conflicts_with_signals(
        facts, decision, active_doc_id, last_doc_id, target_doc_id
    ):
        return True
    if tool_choice is not None and _tool_choice_conflicts_with_selector(
        facts, decision, tool_choice, active_doc_id, last_doc_id, target_doc_id
    ):
        return True
    return False


def _unique_candidate_for_number(candidates: Any, number: int) -> str | None:
    matches = {
        card.get("doc_id") for card in candidates
        if isinstance(card, Mapping) and type(card.get("dialog_number")) is int
        and card.get("dialog_number") == number
        and isinstance(card.get("doc_id"), str)
    } if isinstance(candidates, (tuple, list)) else set()
    return next(iter(matches)) if len(matches) == 1 else None


def _normalize_fresh_decision(response: Mapping[str, Any], accepted_command: str) -> RoutingDecision:
    if not _strict_fresh_shapes(response):
        raise ValueError("fresh selector shape does not match the static schema")
    values = {key: response[key] for key in _FRESH_FIELDS
              if key not in {"tool_choice", "target_operation_id"}}
    if isinstance(values.get("folder_name"), str) and not values["folder_name"].strip():
        values["folder_name"] = None
        missing = list(values.get("missing_slots", []))
        if MissingSlot.FOLDER_NAME.value not in missing:
            missing.append(MissingSlot.FOLDER_NAME.value)
        values["missing_slots"] = missing
        if values.get("disposition") == Disposition.EXECUTE.value:
            values["disposition"] = Disposition.CLARIFY.value
            values["certainty"] = Certainty.UNCERTAIN.value
    decision = RoutingDecision.from_dict(values)
    decision.validate_command_spans(accepted_command)
    for span in decision.explicit_references:
        if not _response_span(span, accepted_command):
            raise ValueError("explicit reference does not match its accepted literal")
    selector = decision.document_selector
    selector_span = selector.span if selector is not None else None
    if selector_span is not None:
        if not _response_span(selector_span, accepted_command):
            raise ValueError("document selector span does not match accepted command")
        if (selector.kind in {DocumentSelectorKind.EXPLICIT_REFERENCE,
                              DocumentSelectorKind.DIALOG_NUMBER,
                              DocumentSelectorKind.LAST_SAVED}
                and selector_span not in decision.explicit_references):
            raise ValueError("explicit document selector span is missing from references")
    _validate_folder_selector_depth(decision.folder_selector, depth=0)
    return decision


def _validate_folder_selector_depth(selector: Any, *, depth: int) -> None:
    if selector is None:
        return
    if depth > 1:
        raise ValueError("fresh folder selectors are limited to one typed parent")
    _validate_folder_selector_depth(selector.parent, depth=depth + 1)


def _validate_fresh_action(
    decision: RoutingDecision, *, tool_choice: Mapping[str, Any] | None,
    target_operation_id: str | None,
) -> RoutingDecision:
    if decision.action in {Action.SUBSTANTIVE, Action.SEARCH, Action.ANNOTATION} and (
        not decision.substantive_text.strip()
    ):
        raise ValueError("action requires substantive text")
    if target_operation_id is not None and decision.action not in {
        Action.PURE_SWITCH, Action.SUBSTANTIVE, Action.MOVE_TRANSCRIPT,
        Action.CREATE_THEN_MOVE,
    }:
        raise ValueError("predecessor save is only valid for ordinary routes")
    if decision.action is Action.SEARCH:
        if (tool_choice is None or tool_choice["name"] != "query"
                or not decision.substantive_text.strip()
                or decision.document_selector is not None
                or decision.folder_selector is not None or target_operation_id is not None):
            raise ValueError("search requires a nonblank query route")
    elif decision.action is Action.MOVE_TRANSCRIPT:
        if tool_choice is not None and tool_choice["name"] != "move_transcript":
            raise ValueError("move action requires move_transcript tool")
    elif decision.action is Action.CREATE_FOLDER:
        if tool_choice is not None and tool_choice["name"] != "create_folder":
            raise ValueError("create action requires create_folder tool")
        if target_operation_id is not None:
            raise ValueError("create action cannot target a predecessor save")
    elif decision.action is Action.CREATE_THEN_MOVE:
        if tool_choice is not None:
            raise ValueError("composite action has no fixed tool or predecessor")
    elif decision.action is not Action.SUBSTANTIVE and tool_choice is not None:
        raise ValueError("tool choice is incompatible with this action")
    elif decision.action is Action.SUBSTANTIVE and tool_choice is not None:
        if tool_choice["name"] in {"move_transcript", "create_folder"}:
            raise ValueError("move/create tools require their semantic action")
    if decision.disposition is Disposition.DELEGATE and tool_choice is not None:
        if OPERATIONS[tool_choice["name"]].write:
            raise ValueError("delegation cannot authorize a write tool")
    if decision.disposition is Disposition.DELEGATE and decision.action in {
        Action.MOVE_TRANSCRIPT, Action.CREATE_FOLDER, Action.CREATE_THEN_MOVE, Action.SAVE,
    }:
        raise ValueError("delegation cannot describe a write action")
    if decision.action is Action.RETRY and (
        decision.document_selector is None
        or decision.document_selector.kind is not DocumentSelectorKind.DIALOG_NUMBER
    ):
        raise ValueError("retry requires an explicit local dialog number")

    missing = list(decision.missing_slots)
    folder_name = decision.folder_name
    if decision.action in {Action.CREATE_FOLDER, Action.CREATE_THEN_MOVE}:
        if (folder_name is None and MissingSlot.FOLDER_NAME not in missing
                and decision.disposition is Disposition.EXECUTE):
            missing.append(MissingSlot.FOLDER_NAME)
        if decision.folder_selector is None and decision.disposition is Disposition.EXECUTE:
            from services.connectors.telegram.runtime.routing_context import FolderSelector, FolderSelectorKind
            folder = FolderSelector(FolderSelectorKind.ROOT_DEFAULT)
        else:
            folder = decision.folder_selector
    else:
        folder = decision.folder_selector
    if decision.action is Action.CREATE_FOLDER and decision.document_selector is not None:
        raise ValueError("folder-only action cannot select a document")
    if (decision.action is Action.CREATE_THEN_MOVE and decision.document_selector is None
            and target_operation_id is None
            and decision.disposition is Disposition.EXECUTE):
        if MissingSlot.DOCUMENT not in missing:
            missing.append(MissingSlot.DOCUMENT)
    if decision.action is Action.MOVE_TRANSCRIPT and decision.disposition is Disposition.EXECUTE:
        if (decision.document_selector is None and target_operation_id is None
                and MissingSlot.DOCUMENT not in missing):
            missing.append(MissingSlot.DOCUMENT)
        if folder is None and MissingSlot.DESTINATION not in missing:
            missing.append(MissingSlot.DESTINATION)
    if missing and decision.disposition is Disposition.EXECUTE:
        decision_status = Disposition.CLARIFY
        certainty = Certainty.UNCERTAIN
    else:
        decision_status = decision.disposition
        certainty = decision.certainty
    if (tuple(missing) != decision.missing_slots or folder is not decision.folder_selector
            or decision_status is not decision.disposition or certainty is not decision.certainty):
        return RoutingDecision(
            disposition=decision_status,
            action=decision.action, substantive_text=decision.substantive_text,
            document_selector=decision.document_selector, folder_selector=folder,
            folder_name=folder_name, explicit_references=decision.explicit_references,
            missing_slots=tuple(missing), evidence_refs=decision.evidence_refs,
            clarification_id=decision.clarification_id,
            clarification_generation=decision.clarification_generation,
            certainty=certainty,
        )
    return decision


def _first_operation_candidate(operation_candidates: Any, operation_id: str | None) -> Any:
    if operation_id is None or not isinstance(operation_candidates, (tuple, list)):
        return None
    return next((item for item in operation_candidates
                 if isinstance(item, Mapping) and item.get("operation_id") == operation_id), None)


def _semantic_route_shell(
    decision: RoutingDecision, *, tool_choice: Mapping[str, Any] | None,
    target_operation_id: str | None, raw_response: Mapping[str, Any],
    operation_candidate: Any,
) -> dict[str, Any]:
    semantic = decision.to_dict()
    ordinary_kind = {
        Action.MOVE_TRANSCRIPT: "substantive", Action.CREATE_FOLDER: "substantive",
        Action.CREATE_THEN_MOVE: "substantive",
    }.get(decision.action, decision.action.value)
    route = {
        "kind": ordinary_kind, "target_ref": None,
        "target_operation_id": target_operation_id, "dialog_number": None,
        "result_refs": [], "substantive_text": decision.substantive_text,
        "tool_choice": tool_choice, "certainty": decision.certainty.value,
        "semantic_decision": semantic, "semantic_response": dict(raw_response),
    }
    operation_doc_id = operation_candidate.get("doc_id") if isinstance(operation_candidate, Mapping) else None
    if isinstance(operation_doc_id, str) and operation_doc_id:
        # The confirmed predecessor is an independent known document source,
        # even when another required slot needs clarification.
        route["target_ref"] = operation_doc_id
    return route


def _project_semantic_selector(
    route: dict[str, Any], decision: RoutingDecision, candidates: tuple[Any, ...],
    active_doc_id: str | None,
) -> str | None:
    selector = decision.document_selector
    if selector is None:
        return None
    if selector.kind is DocumentSelectorKind.EXPLICIT_REFERENCE:
        if selector.span.kind is ExplicitReferenceKind.ID:
            route["target_ref"] = selector.span.parsed_value
        else:
            from services.connectors.telegram.runtime.store import TelegramRuntimeStore
            route["target_ref"] = TelegramRuntimeStore._reference_value(selector.span)
    elif selector.kind is DocumentSelectorKind.DIALOG_NUMBER:
        route["dialog_number"] = selector.dialog_number
        bound_doc = _unique_candidate_for_number(candidates, selector.dialog_number)
        if bound_doc is not None:
            route["target_ref"] = bound_doc
    elif selector.kind is DocumentSelectorKind.LAST_SAVED:
        route["target_ref"] = "last"
    elif selector.kind is DocumentSelectorKind.ACTIVE_DOCUMENT:
        if isinstance(active_doc_id, str) and active_doc_id:
            route["target_ref"] = active_doc_id
    else:
        # Evidence and shown positions require scoped snapshot/manifest resolution.
        return "uncertain"
    return None


def _validate_semantic_predecessor(
    route: dict[str, Any], selector: Any, operation_candidate: Any,
    operation_candidates: Any,
    target_operation_id: str | None, *, active_doc_id: str | None,
    last_doc_id: str | None,
) -> tuple[dict[str, Any], str] | None:
    if target_operation_id is None:
        return None
    available = {item.get("operation_id") for item in operation_candidates
                 if isinstance(item, Mapping)} if isinstance(operation_candidates, (tuple, list)) else set()
    if target_operation_id not in available:
        return route, "invalid_route"
    if operation_candidate is None:
        return route, "invalid_route"
    if not isinstance(operation_candidate.get("doc_id"), str):
        return route, "invalid_route"
    fixed_doc_id = operation_candidate["doc_id"]
    if selector is None:
        route["target_ref"] = fixed_doc_id
    elif selector.kind in {DocumentSelectorKind.EVIDENCE,
                           DocumentSelectorKind.SHOWN_POSITION}:
        # Those selectors need their scoped snapshot/manifest binding.
        return route, "uncertain"
    elif route.get("target_ref") not in {fixed_doc_id, "last"}:
        return _mark_explicit_conflict(route)
    elif (selector.kind is DocumentSelectorKind.DIALOG_NUMBER
          and selector.dialog_number != operation_candidate.get("dialog_number")):
        return _mark_explicit_conflict(route)
    elif (selector.kind is DocumentSelectorKind.LAST_SAVED
          and last_doc_id != fixed_doc_id):
        return _mark_explicit_conflict(route)
    elif (selector.kind is DocumentSelectorKind.ACTIVE_DOCUMENT
          and active_doc_id != fixed_doc_id):
        return _mark_explicit_conflict(route)
    return None


def _project_semantic_tool_choice(
    route: dict[str, Any], decision: RoutingDecision, *,
    tool_choice: Mapping[str, Any] | None, last_doc_id: str | None,
) -> tuple[dict[str, Any], str] | None:
    if decision.action in {Action.MOVE_TRANSCRIPT, Action.CREATE_FOLDER}:
        generated = _semantic_write_tool(decision, route, last_doc_id=last_doc_id)
        if generated is None and decision.action is Action.MOVE_TRANSCRIPT and route.get("dialog_number") is not None:
            # The scoped dialog-number lookup occurs immediately before dispatch.
            route["tool_choice"] = tool_choice
            generated = None
        elif generated is None:
            return route, "uncertain"
        if tool_choice is not None and generated is not None:
            provided_choice = dict(tool_choice)
            provided_args = dict(provided_choice.get("arguments", {}))
            if (provided_choice.get("name") == "create_folder"
                    and provided_args.get("parent_path") == []):
                provided_args.pop("parent_path")
            provided_choice["arguments"] = provided_args
            if provided_choice != generated:
                return _mark_explicit_conflict(route)
        if generated is not None:
            route["tool_choice"] = generated
    elif decision.action is Action.CREATE_THEN_MOVE:
        route["tool_choice"] = None
    return None


def _project_semantic_decision(
    decision: RoutingDecision, *, tool_choice: Mapping[str, Any] | None,
    target_operation_id: str | None, operation_candidates: Any,
    raw_response: Mapping[str, Any], active_doc_id: str | None = None,
    last_doc_id: str | None = None, candidates: Any = (),
) -> tuple[dict[str, Any], str]:
    """Project established ordinary routes while retaining the rich decision in memory."""
    operation_candidate = _first_operation_candidate(operation_candidates, target_operation_id)
    operation_doc_id = operation_candidate.get("doc_id") if isinstance(operation_candidate, Mapping) else None
    route = _semantic_route_shell(
        decision, tool_choice=tool_choice, target_operation_id=target_operation_id,
        raw_response=raw_response, operation_candidate=operation_candidate,
    )
    scoped_candidates = tuple(candidates) if isinstance(candidates, (tuple, list)) else ()
    if isinstance(operation_candidates, (tuple, list)):
        scoped_candidates += tuple(operation_candidates)
    if _semantic_conflict(decision, tool_choice, active_doc_id=active_doc_id,
                          last_doc_id=last_doc_id, candidates=scoped_candidates,
                          target_doc_id=operation_doc_id):
        semantic = route["semantic_decision"]
        semantic["disposition"] = Disposition.CLARIFY.value
        semantic["certainty"] = Certainty.UNCERTAIN.value
        semantic["missing_slots"] = list(dict.fromkeys(
            [*semantic["missing_slots"], "explicit_conflict"]
        ))
        route["semantic_decision"] = semantic
        return route, "uncertain"
    if (decision.disposition not in {Disposition.EXECUTE, Disposition.DELEGATE}
            or decision.certainty is not Certainty.CERTAIN or decision.missing_slots):
        return route, "uncertain"
    selector = decision.document_selector
    selector_status = _project_semantic_selector(
        route, decision, scoped_candidates, active_doc_id
    )
    if selector_status is not None:
        return route, selector_status
    predecessor_result = _validate_semantic_predecessor(
        route, selector, operation_candidate, operation_candidates, target_operation_id,
        active_doc_id=active_doc_id, last_doc_id=last_doc_id,
    )
    if predecessor_result is not None:
        return predecessor_result
    if (decision.action in {Action.MOVE_TRANSCRIPT, Action.CREATE_FOLDER,
                            Action.CREATE_THEN_MOVE}
            and _folder_parent_conflict(decision.folder_selector)):
        return _mark_explicit_conflict(route)
    tool_result = _project_semantic_tool_choice(
        route, decision, tool_choice=tool_choice, last_doc_id=last_doc_id
    )
    if tool_result is not None:
        return tool_result
    if route["kind"] not in _ROUTE_KINDS:
        return route, "invalid_route"
    return route, "ok"


def _folder_path(selector: Any) -> list[str] | None:
    """Normalize literal paths: absolute paths start at root, relative ones use parent."""
    if selector is None:
        return None
    if selector.kind.value == "root_default":
        return []
    if selector.kind.value != "explicit_path":
        return None
    parent_path = _folder_path(selector.parent) if selector.parent is not None else []
    if parent_path is None:
        return None
    raw_path = selector.value
    absolute = raw_path.startswith("/")
    components = (raw_path[1:] if absolute else raw_path).split("/")
    if not components or any(not item or item in {".", ".."} for item in components):
        return None
    if absolute:
        if parent_path and components[:len(parent_path)] != parent_path:
            return None
        return components
    return [*parent_path, *components]


def _folder_parent_conflict(selector: Any) -> bool:
    if selector is None or selector.kind.value == "root_default":
        return False
    if selector.parent is not None and _folder_parent_conflict(selector.parent):
        return True
    if selector.kind.value != "explicit_path" or not selector.value.startswith("/"):
        return False
    parent_path = _folder_path(selector.parent) if selector.parent is not None else []
    if parent_path is None:
        return False
    components = selector.value[1:].split("/")
    return bool(parent_path and components[:len(parent_path)] != parent_path)


def _mark_explicit_conflict(route: dict[str, Any]) -> tuple[dict[str, Any], str]:
    semantic = route.get("semantic_decision")
    if isinstance(semantic, Mapping):
        semantic = dict(semantic)
        semantic["disposition"] = Disposition.CLARIFY.value
        semantic["certainty"] = Certainty.UNCERTAIN.value
        semantic["missing_slots"] = list(dict.fromkeys(
            [*semantic.get("missing_slots", []), MissingSlot.EXPLICIT_CONFLICT.value]
        ))
        route["semantic_decision"] = semantic
    route["certainty"] = Certainty.UNCERTAIN.value
    return route, "uncertain"


def _semantic_write_tool(
    decision: RoutingDecision, route: Mapping[str, Any], *, last_doc_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a registry write choice only when semantic inputs are concrete."""
    selector = decision.folder_selector
    if decision.action is Action.CREATE_FOLDER:
        if decision.folder_name is None:
            return None
        parent_path = _folder_path(selector)
        if parent_path is None:
            return None
        args: dict[str, Any] = {"name": decision.folder_name}
        if parent_path:
            args["parent_path"] = parent_path
        return {"name": "create_folder", "arguments": args, "link_requested": False}
    if decision.action is Action.MOVE_TRANSCRIPT:
        doc_id = route.get("target_ref")
        if (decision.document_selector is not None
                and decision.document_selector.kind is DocumentSelectorKind.LAST_SAVED
                and doc_id == "last"):
            doc_id = last_doc_id
        folder_path = _folder_path(selector)
        if not isinstance(doc_id, str) or not doc_id or not folder_path:
            return None
        return {"name": "move_transcript",
                "arguments": {"doc_id": doc_id, "folder_path": folder_path},
                "link_requested": False}
    return None


def _valid_response(
    response: Any, *, accepted_command: str,
    operation_candidates: Any = (), active_doc_id: str | None = None,
    last_doc_id: str | None = None, candidates: Any = (),
) -> tuple[dict[str, Any] | None, str]:
    """Strictly validate a fresh semantic response against its exact command."""
    if not isinstance(response, Mapping) or set(response) != _FRESH_FIELDS:
        return None, "invalid_route"
    tool_choice = response.get("tool_choice")
    if tool_choice is not None:
        if not isinstance(tool_choice, Mapping) or not {"name", "arguments"}.issubset(tool_choice):
            return None, "invalid_route"
        name = tool_choice.get("name")
        if not isinstance(name, str) or name not in OPERATIONS:
            return None, "invalid_route"
        allowed_tool_fields = {"name", "arguments"} | ({"link_requested"} if name == "get_transcript" else set())
        if not {"name", "arguments"}.issubset(tool_choice) or set(tool_choice) - allowed_tool_fields:
            return None, "invalid_route"
        try:
            arguments = validate_arguments(name, tool_choice.get("arguments"))
        except (TypeError, ValueError):
            return None, "invalid_route"
        link_requested = tool_choice.get("link_requested", False)
        if not isinstance(link_requested, bool) or (link_requested and name != "get_transcript"):
            return None, "invalid_route"
        tool_choice = {"name": name, "arguments": arguments,
                       "link_requested": link_requested}
    try:
        target_operation_id = response.get("target_operation_id")
        if target_operation_id is not None and (
            not isinstance(target_operation_id, str) or not target_operation_id
        ):
            return None, "invalid_route"
        decision = _normalize_fresh_decision(response, accepted_command)
        decision = _validate_fresh_action(
            decision, tool_choice=tool_choice, target_operation_id=target_operation_id,
        )
        if target_operation_id is not None:
            available = {item.get("operation_id") for item in operation_candidates
                         if isinstance(item, Mapping)} if isinstance(operation_candidates, (tuple, list)) else set()
            if target_operation_id not in available:
                return None, "invalid_route"
        projected, status = _project_semantic_decision(
            decision, tool_choice=tool_choice, target_operation_id=target_operation_id,
            operation_candidates=operation_candidates, raw_response=response,
            active_doc_id=active_doc_id, last_doc_id=last_doc_id,
            candidates=candidates,
        )
        if status == "ok" and decision.action is Action.RETRY:
            projected["tool_choice"] = None
        return projected, status
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, "invalid_route"


_CARD_FIELDS = (
    "title", "date", "dialog_date", "created_at", "folder", "folder_id",
    "folder_path", "dialog_number", "conversation_id",
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
        if number is not None:
            target = _doc_for_number(store, context_id, number)
            if target is None or target not in candidates:
                return None, None, "not_found"
            semantic = decision.get("semantic_decision")
            selector = (semantic.get("document_selector")
                        if isinstance(semantic, Mapping) else None)
            if (isinstance(selector, Mapping)
                    and selector.get("kind") == DocumentSelectorKind.ACTIVE_DOCUMENT.value):
                active = _current_doc(store, candidates, context)
                if active is not None and active != target:
                    return None, None, "uncertain"
        else:
            ref = decision["target_ref"]
            semantic = decision.get("semantic_decision")
            selector = (semantic.get("document_selector")
                        if isinstance(semantic, Mapping) else None)
            semantic_last = (isinstance(selector, Mapping)
                             and selector.get("kind") == DocumentSelectorKind.LAST_SAVED.value)
            legacy_last = semantic is None and ref == "last"
            if (semantic_last or legacy_last) and ref == "last":
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
    active_doc_id: str | None = None, last_doc_id: str | None = None,
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
    return _valid_response(
        raw, accepted_command=text,
        operation_candidates=tuple(operation_candidates.values()),
        active_doc_id=active_doc_id, last_doc_id=last_doc_id, candidates=cards,
    )


def _active_doc_id(cards: Any, context: Any) -> str | None:
    """Return the active document only from the cards already captured for this route."""
    conversation_id = getattr(context, "active_conversation_id", None)
    if not isinstance(conversation_id, str):
        return None
    matches = {card["doc_id"] for card in cards
               if isinstance(card, Mapping) and card.get("conversation_id") == conversation_id
               and isinstance(card.get("doc_id"), str)} if isinstance(cards, (tuple, list)) else set()
    return next(iter(matches)) if len(matches) == 1 else None


async def _consume_validated_route(
    decision: Mapping[str, Any] | None, status: str, *, text: str,
    context: Any, store: Any, search: Any, notices: Any, actions: Any,
    operation_id: str | None, owner: str | None, fence: int | None,
    now: datetime | None, cards: tuple[dict[str, Any], ...],
    operation_candidates: Mapping[str, dict[str, Any]], notify_switch: bool,
) -> dict[str, Any]:
    """Consume a server-validated ordinary route projection without revalidation."""
    if decision is None or status != "ok":
        return _failure(status)
    kind = decision["kind"]
    if kind == "search":
        return await _run_search_route(
            text=text, arguments=decision["tool_choice"]["arguments"],
            context=context, store=store, search=search,
        )
    if kind == "save":
        return _confirm_save_decision(
            context=context, store=store, operation_id=operation_id,
            owner=owner, fence=fence, now=now,
        )
    if kind == "retry":
        return {"status": "ok", "kind": "retry",
                "dialog_number": decision["dialog_number"]}
    semantic = decision.get("semantic_decision")
    if (kind == "substantive" and isinstance(semantic, Mapping)
            and semantic.get("action") == Action.CREATE_FOLDER.value
            and decision.get("tool_choice") is not None):
        choice = decision["tool_choice"]
        try:
            action_result = await actions.execute(
                name=choice["name"], arguments=choice["arguments"],
                target_doc_id=None, link_requested=False,
            )
        except Exception:
            return _failure("error", kind=kind)
        return {"status": "ok", "kind": kind, "target_doc_id": None,
                "substantive_text": decision["substantive_text"],
                "tool_choice": choice, "action_result": action_result}
    target, failure = _resolve_route_target(
        decision, context=context, store=store,
        candidates=tuple(card["doc_id"] for card in cards),
        operation_candidates=operation_candidates,
    )
    if failure:
        return _failure(failure, kind=kind)
    if (kind == "substantive" and isinstance(semantic, Mapping)
            and semantic.get("action") == Action.MOVE_TRANSCRIPT.value
            and decision.get("tool_choice") is None):
        try:
            typed = RoutingDecision.from_dict(semantic)
            generated = _semantic_write_tool(
                typed, {"target_ref": target["target_doc_id"]},
            )
        except (TypeError, ValueError):
            generated = None
        if generated is None:
            return _failure("uncertain", kind=kind)
        decision = {**decision, "tool_choice": generated}
    return await _finish_targeted_route(
        decision, target=target, context=context, notices=notices, actions=actions,
        notify_switch=notify_switch,
    )


async def _run_search_route(
    *, text: str, arguments: Mapping[str, Any], context: Any, store: Any, search: Any,
) -> dict[str, Any]:
    """Run the validated query arguments and record an isolated result session."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"status": "invalid_route"}
    error = None
    try:
        refs = list(await search.search(dict(arguments)))
    except Exception:
        refs, error = [], "search unavailable"
    session_id = store.create_search_session(
        context_id=context.context_id, query=query, result_refs=refs, error=error,
        search_filters=dict(arguments),
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
        semantic = decision.get("semantic_decision")
        selector = (semantic.get("document_selector")
                    if isinstance(semantic, Mapping) else None)
        if (isinstance(selector, Mapping)
                and selector.get("kind") == DocumentSelectorKind.ACTIVE_DOCUMENT.value):
            active = _current_doc(store, candidates, context)
            if active is None or active != fixed_target["doc_id"]:
                # Active focus and the predecessor's fixed target are independent
                # signals; without equality, leave the route unresolved.
                return None, "uncertain"
        concrete_ref = decision["target_ref"]
        semantic = decision.get("semantic_decision")
        selector = (semantic.get("document_selector")
                    if isinstance(semantic, Mapping) else None)
        semantic_last = (isinstance(selector, Mapping)
                         and selector.get("kind") == DocumentSelectorKind.LAST_SAVED.value)
        allowed_refs = {fixed_target["doc_id"]}
        if semantic_last:
            allowed_refs.add("last")
            if context.last_doc_id != fixed_target["doc_id"]:
                return None, "uncertain"
        if concrete_ref is not None and concrete_ref not in allowed_refs:
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
        active_doc_id=_active_doc_id(cards, context),
        last_doc_id=getattr(context, "last_doc_id", None),
    )
    return await _consume_validated_route(
        decision, status, text=text, context=context, store=store, search=search,
        notices=notices, actions=actions, operation_id=operation_id, owner=owner,
        fence=fence, now=now, cards=cards, operation_candidates=operation_candidates,
        notify_switch=notify_switch,
    )


def _control_route_record(route: Any) -> dict[str, Any]:
    """Keep allowlisted, validated replay metadata in the operation control ledger."""
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
        "target_draft_id", "target_conversation_id",
        "target_operation_id", "dependency_operation_id",
        "retry_request_operation_id", "annotation_text",
    ):
        value = route.get(field)
        if field == "status" and (not isinstance(value, str) or value not in allowed_statuses):
            continue
        if field == "kind" and (not isinstance(value, str) or value not in allowed_kinds):
            continue
        if isinstance(value, str) and value:
            safe[field] = value
    if (route.get("selection_policy") == "silent_v1"
            and safe.get("kind") in {"pure_switch", "substantive"}):
        safe["selection_policy"] = "silent_v1"
    number = route.get("dialog_number")
    if type(number) is int and number > 0:
        safe["dialog_number"] = number
    route_kind = route.get("kind")
    if isinstance(route_kind, str) and route_kind in {"substantive", "search"}:
        choice = route.get("tool_choice")
        if choice is not None:
            if not isinstance(choice, Mapping) or set(choice) - {
                "name", "arguments", "link_requested",
            }:
                safe["status"] = "invalid_route"
                return safe
            name = choice.get("name")
            link_requested = choice.get("link_requested", False)
            try:
                arguments = validate_arguments(name, choice.get("arguments"))
            except (TypeError, ValueError):
                safe["status"] = "invalid_route"
                return safe
            if (not isinstance(name, str) or name not in OPERATIONS
                    or not isinstance(link_requested, bool)
                    or (link_requested and name != "get_transcript")
                    or (route_kind == "search"
                        and (name != "query" or link_requested))
                    or (arguments.get("doc_id") is not None
                        and arguments["doc_id"] != safe.get("target_doc_id"))):
                safe["status"] = "invalid_route"
                return safe
            safe["tool_choice"] = {
                "name": name, "arguments": arguments,
                "link_requested": link_requested,
            }
    if route_kind == "search":
        if route.get("search_protocol") == "agent-v1" and "tool_choice" not in safe:
            safe["status"] = "invalid_route"
            return safe
        if route.get("search_protocol") == "agent-v1":
            safe["search_protocol"] = "agent-v1"
        refs = route.get("result_refs")
        session_id = route.get("search_session_id")
        if (not isinstance(refs, list)
                or any(not isinstance(ref, str) or not ref for ref in refs)
                or not isinstance(session_id, str) or not session_id):
            safe["status"] = "invalid_route"
            return safe
        safe["result_refs"] = list(refs)
        safe["search_session_id"] = session_id
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
        "pure_switch", "substantive", "search", "save", "retry", "annotation",
    }:
        return "invalid_route"
    if kind == "search":
        refs = route.get("result_refs")
        session_id = route.get("search_session_id")
        if (not isinstance(refs, list)
                or any(not isinstance(ref, str) or not ref for ref in refs)
                or not isinstance(session_id, str) or not session_id):
            return "invalid_route"
        return "ok"
    if kind == "save":
        if type(route.get("confirmed")) is not bool:
            return "invalid_route"
        return "ok" if route.get("confirmed") else "unconfirmed"
    if kind == "retry":
        number = route.get("dialog_number")
        return "ok" if type(number) is int and number > 0 else "invalid_route"
    if kind == "annotation":
        text = route.get("annotation_text")
        has_target = (isinstance(route.get("target_doc_id"), str)
                      or isinstance(route.get("target_draft_id"), str))
        return "ok" if (has_target and isinstance(text, str) and text.strip()) else "invalid_route"
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
        choice = route.get("tool_choice")
        if choice is not None:
            if not isinstance(choice, Mapping) or set(choice) - {
                "name", "arguments", "link_requested",
            }:
                return "invalid_route"
            name = choice.get("name")
            link_requested = choice.get("link_requested", False)
            try:
                arguments = validate_arguments(name, choice.get("arguments"))
            except (TypeError, ValueError):
                return "invalid_route"
            if (not isinstance(name, str) or name not in OPERATIONS
                    or not isinstance(link_requested, bool)
                    or (link_requested and name != "get_transcript")
                    or (arguments.get("doc_id") is not None
                        and arguments["doc_id"] != route.get("target_doc_id"))):
                return "invalid_route"
    return "ok"


@dataclass(frozen=True)
class _PersistedOperationRoute:
    """Carry a sanitized route and the lease-bound candidate cleanup handle."""

    route: Any
    route_record: dict[str, Any]
    context_id: Any
    clear_candidate: Callable[..., Any] | None
    repository: Any


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
    selection_policy: str | None = None,
) -> _PersistedOperationRoute | None:
    """Validate and persist only allowlisted route metadata before any side effect."""
    validation_status = _operation_route_status(route)
    route_record = _control_route_record(route)
    route_record["status"] = validation_status
    if (selection_policy == "silent_v1" and validation_status == "ok"
            and route_record.get("kind") in {"pure_switch", "substantive"}):
        route_record["selection_policy"] = selection_policy
    clear_candidate = getattr(repository, "clear_command_candidate", None)
    context_id = getattr(context, "context_id", None)
    if callable(clear_candidate):
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
        repository=repository,
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


def _is_retry_work_replay(route_record: Mapping[str, Any]) -> bool:
    """Identify a retry work operation by its persisted source-command marker."""
    request_operation_id = route_record.get("retry_request_operation_id")
    return isinstance(request_operation_id, str) and bool(request_operation_id)


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
        if kind == "save":
            if not _clear_command_candidate(
                state, operation_id=operation_id, owner=owner, fence=fence, now=now,
            ):
                return {"status": "error", "kind": kind}
            return {"status": "ok", "kind": "save", "confirmed": True}
        if _is_retry_work_replay(state.route_record):
            return {"status": "ok", "kind": kind,
                    "dialog_number": state.route_record.get("dialog_number")}
        number = state.route_record.get("dialog_number")
        if type(number) is not int or number < 1:
            return {"status": "invalid_route", "kind": kind}
        from services.connectors.telegram.retry import TelegramRetryService

        result = TelegramRetryService(state.repository).retry_failed_draft_for_command(
            state.context_id, number, request_operation_id=operation_id, now=now,
        )
        if not getattr(result, "accepted", False):
            if not _clear_command_candidate(
                state, operation_id=operation_id, owner=owner, fence=fence, now=now,
            ):
                return {"status": "error", "kind": kind}
            return {"status": "not_found", "kind": kind,
                    "dialog_number": number}
        if not _clear_command_candidate(
            state, operation_id=operation_id, owner=owner, fence=fence, now=now,
        ):
            return {"status": "error", "kind": kind}
        return {"status": "ok", "kind": kind, "dialog_number": number,
                "retry_operation_id": getattr(result, "operation_id", None)}
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
    """Publish silent selections under the live fence or satisfy the legacy notice barrier."""
    context_id = getattr(context, "context_id", None)
    policy = route.get("selection_policy") if isinstance(route, Mapping) else None
    if policy == "silent_v1":
        same_target = context.active_conversation_id == conversation_id
        requires_publish = (
            not same_target
            or (isinstance(route, Mapping) and route.get("kind") == "pure_switch")
        )
        if not requires_publish:
            return False
        if selection_publisher is None:
            return "pending"
        try:
            published = selection_publisher(
                operation_id, conversation_id=conversation_id,
                dialog_number=number,
            )
            if inspect.isawaitable(published):
                published = await published
        except Exception:
            return "pending"
        return True if published is True else "pending"
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
    prompt = build_document_context(
        history, substantive_text,
        conversation_id=conversation_id, operation_id=operation_id,
    )
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
    except (TelegramSavePending, TelegramDirectSaveInterrupted):
        raise
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


async def _confirm_route_selection(
    *, route: Any, operation_id: str, target_doc_id: str,
    conversation_id: str, number: int, context: Any, notifications: Any,
    selection_publisher: Callable[..., Any] | None, repository: Any,
    queue: Any, owner: str | None, fence: int | None, now: datetime | None,
    kind: Any,
) -> dict[str, Any] | None:
    """Return a retry/error result until this route's selection policy is satisfied."""
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
        status = "selection_pending" if (
            isinstance(route, Mapping) and route.get("selection_policy") == "silent_v1"
        ) else "switch_notice_pending"
        return {"status": status, "kind": kind}
    return None


async def _finish_conversation_route(
    *, state: _PersistedOperationRoute, kind: Any, operation_id: str,
    context: Any, repository: Any, agent: Any, notifications: Any,
    queue: Any, selection_publisher: Callable[..., Any] | None,
    owner: str | None, fence: int | None, now: datetime | None,
) -> dict[str, Any]:
    """Select a validated target before finishing a pure switch or conversation turn."""
    target = _conversation_route_target(state.route_record)
    if target is None:
        return {"status": "invalid_route", **({"kind": kind} if kind else {})}
    target_doc_id, conversation_id, number = target
    substantive_text = _substantive_route_text(state.route, kind)
    if kind == "substantive" and substantive_text is None:
        return {"status": "invalid_route", "kind": kind}

    selection_result = await _confirm_route_selection(
        route=state.route_record, operation_id=operation_id,
        target_doc_id=target_doc_id, conversation_id=conversation_id,
        number=number, context=context, notifications=notifications,
        selection_publisher=selection_publisher, repository=repository,
        queue=queue, owner=owner, fence=fence, now=now, kind=kind,
    )
    if selection_result is not None:
        return selection_result

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
    if (isinstance(existing_route, Mapping)
            and existing_route.get("selection_policy") == "silent_v1"):
        selection_policy = "silent_v1"
    else:
        selection_policy = (
            "silent_v1"
            if existing_route is None and _operation_route_status(route) == "ok"
            and isinstance(route, Mapping)
            and route.get("kind") in {"pure_switch", "substantive"}
            else None
        )
    state = _persist_operation_route(
        route, operation_id=operation_id, context=context, repository=repository,
        owner=owner, fence=fence, now=now, selection_policy=selection_policy,
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
