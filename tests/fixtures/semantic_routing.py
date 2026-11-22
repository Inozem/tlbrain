"""Explicit migration helpers for old-shaped test model responses only."""

from __future__ import annotations

import re
from copy import deepcopy


def _shown_position_proposal(command: str, presentation_refs) -> tuple[dict | None, list[str]]:
    """Build a closed shown-position selector from actual prompt list refs."""
    match = re.search(
        r"\b(first|second|third|last)\s+(?:search\s+)?results?\b|"
        r"\bresults?\s+(?:number\s+)?(\d+|first|second|third|last)\b",
        command, flags=re.IGNORECASE,
    )
    if match is None:
        return None, []
    token = next((group for group in match.groups() if group), "")
    position = {"first": 1, "second": 2, "third": 3,
                "last": "last"}.get(token.lower(),
                                     int(token) if token.isdigit() else None)
    if position is None:
        return None, []
    lists = []
    if isinstance(presentation_refs, (list, tuple)):
        for presentation in presentation_refs:
            if not isinstance(presentation, dict) or not isinstance(presentation.get("ref"), str):
                continue
            for row in presentation.get("lists", ()):
                if (isinstance(row, dict) and isinstance(row.get("list_id"), str)
                        and row["list_id"]):
                    lists.append({"ref": presentation["ref"],
                                  "presentation_id": presentation.get("presentation_id"),
                                  "list_id": row["list_id"]})
    older_requested = re.search(r"\b(?:older|earlier|previous)\s+list\b",
                                command, flags=re.IGNORECASE) is not None
    if older_requested and lists:
        # The actual prompt projection intentionally exposes only a source ref
        # and its list rows. Resolve "older" by that captured order, not by a
        # current pointer or a fixture-only presentation ID.
        oldest_ref = lists[0]["ref"]
        lists = [row for row in lists if row["ref"] == oldest_ref]
    identities = {(row["ref"], row["list_id"]) for row in lists}
    if older_requested:
        if len(identities) != 1:
            return None, []
    elif len(identities) != 1:
        # The closed selector can name one list only; preserve ambiguity for
        # the real semantic validator instead of choosing a presentation.
        return None, []
    ref, list_id = next(iter(identities)) if identities else (None, None)
    if not isinstance(ref, str) or not isinstance(list_id, str):
        return None, []
    return {"kind": "shown_position", "list_id": list_id,
            "position": position}, [ref]


def _verified_evidence_proposal(contextual_evidence) -> tuple[dict | None, list[str]]:
    """Return a document selector only when captured exchange refs are unique."""
    if not isinstance(contextual_evidence, dict):
        return None, []
    by_doc = {}
    for doc_id, rows in contextual_evidence.items():
        if not isinstance(doc_id, str) or not isinstance(rows, (list, tuple)):
            continue
        refs = []
        for row in rows:
            if not isinstance(row, dict) or row.get("source_kind") != "verified_exchange":
                continue
            source_ref = row.get("source_ref")
            if isinstance(source_ref, str) and source_ref:
                refs.append(source_ref)
            fragments = row.get("fragment_refs", ())
            if isinstance(fragments, (list, tuple)):
                refs.extend(ref for ref in fragments if isinstance(ref, str) and ref)
        if refs:
            by_doc[doc_id] = list(dict.fromkeys(refs))
    if len(by_doc) != 1:
        return None, []
    doc_id, refs = next(iter(by_doc.items()))
    return {"kind": "evidence", "doc_id": doc_id,
            "evidence_refs": refs}, refs


def legacy_route_fixture_to_semantic(
    route: dict, command: str, *, active_doc_id: str | None = None,
    active_dialog_number: int | None = None,
    source_bound_draft: bool = False, source_bound_dialog_number: int | None = None,
    known_doc_ids=(), contextual_evidence=None, presentation_refs=None,
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
    url_pattern = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
    url_matches = list(url_pattern.finditer(command))
    for match in url_matches:
        raw_url = match.group()
        span = {"kind": ExplicitReferenceKind.URL.value,
                "start": match.start(), "end": match.end(),
                "raw_text": raw_url, "parsed_value": raw_url}
        spans.append(span)
        if selector is None:
            selector = {"kind": DocumentSelectorKind.EXPLICIT_REFERENCE.value,
                        "raw_value": raw_url, "span": span}
    for candidate in known_doc_ids:
        if not isinstance(candidate, str) or not candidate:
            continue
        match = re.search(rf"(?<![A-Za-z0-9_-]){re.escape(candidate)}(?![A-Za-z0-9_-])", command)
        if match is not None and not any(
            match.start() < url.end() and url.start() < match.end()
            for url in url_matches
        ):
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
    # Keep every accepted local-number signal even when the scripted legacy
    # route omitted its corresponding convenience field. In particular this
    # preserves a URL + local-number contradiction for server validation.
    local_match = re.search(
        r"\b(?:dialog|transcript)\s*(#\s*\d+|no\.?\s*\d+|\d+)",
        command, flags=re.IGNORECASE,
    )
    local_value = None
    if local_match is not None:
        raw_local = local_match.group(1)
        digits = re.search(r"\d+", raw_local)
        if digits is not None:
            local_value = int(digits.group())
            raw_start = local_match.start(1) + digits.start()
            raw_text = digits.group()
            span = {"kind": ExplicitReferenceKind.LOCAL_NUMBER.value,
                    "start": raw_start, "end": raw_start + len(raw_text),
                    "raw_text": raw_text, "parsed_value": local_value}
            if span not in spans:
                spans.append(span)
            if selector is None:
                selector = {"kind": DocumentSelectorKind.DIALOG_NUMBER.value,
                            "dialog_number": local_value, "span": span}
    if type(number) is int and number > 0:
        match = re.search(rf"(?<!\d){number}(?!\d)", command)
        if match:
            span = {"kind": ExplicitReferenceKind.LOCAL_NUMBER.value,
                    "start": match.start(), "end": match.end(),
                    "raw_text": match.group(), "parsed_value": number}
            if not any(item.get("kind") == ExplicitReferenceKind.LOCAL_NUMBER.value
                       and item.get("parsed_value") == number for item in spans):
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
            # Keep it only as a proposed tool argument. The fresh response has
            # no document selector and must be clarified/rejected by runtime.
            pass
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
            and target in (None, active_doc_id)
            and action in {Action.PURE_SWITCH.value, Action.SUBSTANTIVE.value}):
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
    evidence_by_doc = contextual_evidence if isinstance(contextual_evidence, dict) else {}
    selector_doc_id = None
    if isinstance(selector, dict):
        if selector.get("kind") == DocumentSelectorKind.EXPLICIT_REFERENCE.value:
            selector_doc_id = selector.get("raw_value")
        elif selector.get("kind") == DocumentSelectorKind.ACTIVE_DOCUMENT.value:
            selector_doc_id = active_doc_id
        elif selector.get("kind") == DocumentSelectorKind.DIALOG_NUMBER.value:
            matches = []
            for doc_id, rows in evidence_by_doc.items():
                if isinstance(rows, (list, tuple)) and any(
                    isinstance(row, dict) and row.get("source_kind") == "local_number"
                    for row in rows
                ):
                    matches.append(doc_id)
            if len(matches) == 1:
                selector_doc_id = matches[0]
    evidence_rows = evidence_by_doc.get(selector_doc_id, ())
    evidence_refs = []
    if isinstance(evidence_rows, (list, tuple)):
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            source_ref = row.get("source_ref")
            if isinstance(source_ref, str) and source_ref:
                evidence_refs.append(source_ref)
            fragments = row.get("fragment_refs", [])
            if isinstance(fragments, (list, tuple)):
                evidence_refs.extend(ref for ref in fragments if isinstance(ref, str) and ref)
    evidence_refs = list(dict.fromkeys(evidence_refs))
    # Keep the model's proposed doc ID intact while expressing the position
    # named by the accepted command and scoped visible lists. The source
    # citation belongs to the top-level decision evidence_refs field.
    shown_selector, shown_refs = _shown_position_proposal(command, presentation_refs)
    if shown_selector is not None:
        selector = shown_selector
        evidence_refs = shown_refs
    elif (selector is None and not spans and not missing
          and action != Action.SEARCH.value
          and not (
              action == Action.SUBSTANTIVE.value
              and target is None and number is None
              and (
                  tool_choice is None
                  or (isinstance(tool_choice, dict)
                      and tool_choice.get("name") in {"get_transcript", "query"})
              )
          )):
        evidence_selector, cited_refs = _verified_evidence_proposal(contextual_evidence)
        if evidence_selector is not None:
            selector = evidence_selector
            selector_doc_id = evidence_selector["doc_id"]
            evidence_refs = cited_refs
        elif isinstance(contextual_evidence, dict):
            verified_refs = []
            verified_docs = set()
            for doc_id, rows in contextual_evidence.items():
                if not isinstance(doc_id, str) or not isinstance(rows, (list, tuple)):
                    continue
                for row in rows:
                    if not isinstance(row, dict) or row.get("source_kind") != "verified_exchange":
                        continue
                    verified_docs.add(doc_id)
                    source_ref = row.get("source_ref")
                    if isinstance(source_ref, str) and source_ref:
                        verified_refs.append(source_ref)
                    fragments = row.get("fragment_refs", ())
                    if isinstance(fragments, (list, tuple)):
                        verified_refs.extend(ref for ref in fragments
                                             if isinstance(ref, str) and ref)
            if len(verified_docs) > 1:
                # The quoted-source request has multiple actual sources. Make
                # the ambiguity explicit in the model proposal and cite all
                # captured source refs so the clarification can name them.
                missing = ["document"]
                certainty = "uncertain"
                evidence_refs = list(dict.fromkeys(verified_refs))
    if (isinstance(selector, dict)
            and selector.get("kind") == DocumentSelectorKind.EVIDENCE.value):
        selector["evidence_refs"] = list(evidence_refs)
    if (isinstance(selector, dict)
            and selector.get("kind") == DocumentSelectorKind.EXPLICIT_REFERENCE.value
            and local_value is not None):
        explicit_value = selector.get("raw_value")
        if isinstance(explicit_value, str) and explicit_value.startswith(("http://", "https://")):
            url_id = re.search(
                r"/(?:document|file)/d/([A-Za-z0-9_-]+)(?:/|$)", explicit_value,
                flags=re.IGNORECASE,
            )
            explicit_value = url_id.group(1) if url_id else None
        local_docs = {
            doc_id for doc_id, rows in evidence_by_doc.items()
            if isinstance(rows, (list, tuple)) and any(
                isinstance(row, dict) and row.get("source_kind") == "local_number"
                for row in rows
            )
        }
        if len(local_docs) == 1 and explicit_value not in local_docs:
            # Keep both literal spans, but do not serialize the URL/ID as a
            # chosen selector when the captured numbered binding disagrees.
            selector = None
    return {
        "disposition": "execute" if certainty == "certain" else "clarify",
        "action": action,
        "substantive_text": substantive_text,
        "document_selector": selector,
        "folder_selector": folder_selector, "folder_name": folder_name,
        "explicit_references": spans, "missing_slots": missing,
        "evidence_refs": evidence_refs, "clarification_id": None,
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
    prompt_context = prompt.get("context")
    if not isinstance(prompt_context, dict):
        prompt_context = {}
    return legacy_route_fixture_to_semantic(
        route, prompt.get("command", ""), active_doc_id=active_doc_id,
        active_dialog_number=active_dialog_number,
        source_bound_draft=source_bound_draft,
        source_bound_dialog_number=source_bound_dialog_number,
        known_doc_ids=[item.get("doc_id") for item in prompt.get("documents", [])
                       if isinstance(item, dict)],
        contextual_evidence=prompt_context.get("contextual_evidence"),
        presentation_refs=prompt_context.get("presentation_refs"),
    )
