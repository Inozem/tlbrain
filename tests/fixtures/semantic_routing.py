"""Explicit migration helpers for old-shaped test model responses only."""

from __future__ import annotations

import re
from copy import deepcopy


def legacy_route_fixture_to_semantic(
    route: dict, command: str, *, active_doc_id: str | None = None,
    active_dialog_number: int | None = None,
    source_bound_draft: bool = False, source_bound_dialog_number: int | None = None,
    known_doc_ids=(),
) -> dict:
    """Convert a historical test fixture without manufacturing command evidence."""
    from services.connectors.telegram.runtime.routing_context import (
        Action, DocumentSelectorKind, ExplicitReferenceKind,
    )
    from services.connectors.telegram.sessions import _valid_persisted_route

    _normalized, status = _valid_persisted_route(route)
    if status not in {"ok", "uncertain"}:
        return deepcopy(route)

    old = deepcopy(route)
    try:
        action = Action(old.get("kind")).value
    except (TypeError, ValueError):
        return old
    tool_choice = old.get("tool_choice")
    arguments = (tool_choice.get("arguments", {})
                 if isinstance(tool_choice, dict) else {})
    if isinstance(tool_choice, dict):
        if tool_choice.get("name") != "get_transcript":
            tool_choice.pop("link_requested", None)
        if tool_choice.get("name") == "move_transcript":
            action = Action.MOVE_TRANSCRIPT.value
        elif tool_choice.get("name") == "create_folder":
            action = Action.CREATE_FOLDER.value
    spans = []
    selector = None
    missing = []
    literal_matches = []
    for candidate in known_doc_ids:
        if not isinstance(candidate, str) or not candidate:
            continue
        match = re.search(rf"(?<![A-Za-z0-9_-]){re.escape(candidate)}(?![A-Za-z0-9_-])", command)
        if match is not None:
            literal_matches.append((candidate, match))
    for candidate, match in literal_matches:
        span = {"kind": ExplicitReferenceKind.ID.value,
                "start": match.start(), "end": match.end(),
                "raw_text": match.group(), "parsed_value": candidate}
        spans.append(span)
    if len(literal_matches) == 1:
        candidate, match = literal_matches[0]
        span = spans[0]
        selector = {"kind": DocumentSelectorKind.EXPLICIT_REFERENCE.value,
                    "raw_value": candidate, "span": span}
    target = old.get("target_ref")
    number = old.get("dialog_number")
    if type(number) is int and number > 0:
        match = re.search(rf"(?<!\d){number}(?!\d)", command)
        if match:
            span = {"kind": ExplicitReferenceKind.LOCAL_NUMBER.value,
                    "start": match.start(), "end": match.end(),
                    "raw_text": match.group(), "parsed_value": number}
            spans.append(span)
            if selector is None:
                selector = {"kind": DocumentSelectorKind.DIALOG_NUMBER.value,
                            "dialog_number": number, "span": span}
        elif source_bound_draft:
            if source_bound_dialog_number != number:
                # A model-only number cannot be repaired into a captured-draft
                # selector. Keep the old payload malformed for the fresh reader.
                return old
        elif active_doc_id is not None and active_dialog_number == number:
            if selector is None:
                selector = {"kind": DocumentSelectorKind.ACTIVE_DOCUMENT.value}
        elif selector is None:
            # Unbound model-only numbers are not accepted-command evidence.
            return old
    if isinstance(target, str) and target == "last":
        # A literal resource ID named "last" is still an ID. Interpret the
        # historical sentinel only when the accepted command explicitly says
        # "last" and does not contain that literal ID.
        has_literal_last_id = any(candidate == "last" for candidate, _ in literal_matches)
        match = None if has_literal_last_id else re.search(
            r"\blast\b|\u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0439\s+\u0434\u0438\u0430\u043b\u043e\u0433", command, flags=re.IGNORECASE,
        )
        if match:
            span = {"kind": ExplicitReferenceKind.LAST_SAVED.value,
                    "start": match.start(), "end": match.end(),
                    "raw_text": match.group(), "parsed_value": match.group()}
            spans.append(span)
            selector = {"kind": DocumentSelectorKind.LAST_SAVED.value, "span": span}
        elif selector is None and old.get("target_operation_id") is None:
            if target == active_doc_id:
                selector = {"kind": DocumentSelectorKind.ACTIVE_DOCUMENT.value}
            else:
                return old
    elif isinstance(target, str) and target:
        match = re.search(re.escape(target), command)
        if match:
            span = {"kind": ExplicitReferenceKind.ID.value,
                    "start": match.start(), "end": match.end(),
                    "raw_text": match.group(), "parsed_value": target}
            if span not in spans:
                spans.append(span)
            if selector is None:
                selector = {"kind": DocumentSelectorKind.EXPLICIT_REFERENCE.value,
                            "raw_value": target, "span": span}
        elif selector is None and not spans and target == active_doc_id:
            # A server-seeded active binding is a valid legacy fixture source.
            # The model's absent literal ID does not become command evidence.
            selector = {"kind": DocumentSelectorKind.ACTIVE_DOCUMENT.value}
        elif selector is None and old.get("target_operation_id") is None:
            # The old model's ungrounded target guess is not an explicit signal.
            # Preserve only server-seeded bindings; otherwise leave this
            # unrepresentable model guess malformed for the fresh reader.
            if not source_bound_draft:
                return old
    certainty = old.get("certainty", "certain")
    if action in {Action.SEARCH.value, Action.CREATE_FOLDER.value}:
        # These routes do not select an existing document; old 8-field target
        # hints were unrelated metadata and cannot be promoted into a selector.
        selector = None
        if action == Action.CREATE_FOLDER.value:
            missing = []
    if (selector is None and not spans and not missing and active_doc_id is not None
            and not source_bound_draft
            and old.get("target_operation_id") is None
            and action in {Action.PURE_SWITCH.value, Action.SUBSTANTIVE.value,
                           Action.ANNOTATION.value}):
        selector = {"kind": DocumentSelectorKind.ACTIVE_DOCUMENT.value}
    folder_selector, folder_name = None, None
    if action == Action.MOVE_TRANSCRIPT.value:
        path = arguments.get("folder_path")
        if isinstance(path, list) and path and all(isinstance(item, str) and item for item in path):
            folder_selector = {"kind": "explicit_path", "value": "/".join(path),
                               "parent": None}
    elif action == Action.CREATE_FOLDER.value:
        parent_path = arguments.get("parent_path", [])
        folder_name = arguments.get("name")
        if isinstance(parent_path, list) and all(isinstance(item, str) and item for item in parent_path):
            folder_selector = ({"kind": "root_default", "value": None, "parent": None}
                               if not parent_path else
                               {"kind": "explicit_path", "value": "/".join(parent_path),
                                "parent": None})
    if missing and certainty == "certain":
        certainty = "uncertain"
    substantive_text = old.get("substantive_text")
    if substantive_text is None and action in {
        Action.SAVE.value, Action.PURE_SWITCH.value, Action.RETRY.value,
    }:
        substantive_text = ""
    return {
        "disposition": "execute" if certainty == "certain" else "clarify",
        "action": action,
        "substantive_text": substantive_text,
        "document_selector": selector,
        "folder_selector": folder_selector, "folder_name": folder_name,
        "explicit_references": spans, "missing_slots": missing,
        "evidence_refs": [], "clarification_id": None,
        "clarification_generation": None,
        "certainty": certainty, "tool_choice": tool_choice,
        "target_operation_id": old.get("target_operation_id"),
    }


def legacy_route_fixture_from_prompt(route: dict, contents: list) -> dict:
    """Convert one historical fake Gemini reply using its actual prompt text."""
    if not isinstance(route, dict) or "kind" not in route:
        return deepcopy(route)
    try:
        import json
        prompt = json.loads(contents[0]["parts"][0]["text"])
    except (IndexError, KeyError, TypeError, ValueError):
        return deepcopy(route)
    active_doc_id = None
    source_bound_draft = False
    source_bound_dialog_number = None
    active_dialog_number = None
    context = prompt.get("context")
    source_bound_draft = bool(
        isinstance(context, dict)
        and isinstance(context.get("captured_collecting_draft"), dict)
    )
    if source_bound_draft:
        source_bound_dialog_number = context["captured_collecting_draft"].get("dialog_number")
    active_conversation_id = (context.get("active_conversation_id")
                              if isinstance(context, dict) else None)
    for card in prompt.get("documents", []):
        if (isinstance(card, dict) and card.get("conversation_id") == active_conversation_id
                and isinstance(card.get("doc_id"), str)):
            active_doc_id = card["doc_id"]
            number = card.get("dialog_number")
            active_dialog_number = number if type(number) is int else None
            break
    return legacy_route_fixture_to_semantic(
        route, prompt.get("command", ""), active_doc_id=active_doc_id,
        active_dialog_number=active_dialog_number,
        source_bound_draft=source_bound_draft,
        source_bound_dialog_number=source_bound_dialog_number,
        known_doc_ids=[item.get("doc_id") for item in prompt.get("documents", [])
                       if isinstance(item, dict)],
    )
