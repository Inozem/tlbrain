"""Focused contracts for the strict fresh semantic routing envelope."""

from __future__ import annotations

from copy import deepcopy

import pytest

pytestmark = pytest.mark.offline


def _fresh(action="substantive", **overrides):
    value = {
        "disposition": "execute", "action": action,
        "substantive_text": "continue the current work",
        "document_selector": {"kind": "active_document"},
        "folder_selector": None, "folder_name": None,
        "explicit_references": [], "missing_slots": [], "evidence_refs": [],
        "clarification_id": None, "clarification_generation": None,
        "target_operation_id": None, "tool_choice": None, "certainty": "certain",
    }
    value.update(overrides)
    return value


def test_fresh_validator_rejects_legacy_route_shape_and_protocol():
    from services.connectors.telegram.sessions import _valid_response

    legacy = {
        "kind": "substantive", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "hello",
        "tool_choice": None, "certainty": "certain",
    }
    assert _valid_response(legacy, accepted_command="hello") == (None, "invalid_route")
    with_protocol = _fresh()
    with_protocol["protocol"] = "context-v2"
    assert _valid_response(with_protocol, accepted_command="hello") == (None, "invalid_route")


def test_schema_is_static_and_contains_no_document_or_predecessor_id_enums():
    from services.connectors.telegram.runtime.routing import GeminiCommandRouter, _ROUTE_SCHEMA

    schemas = [GeminiCommandRouter._invocation_schema(
        [{"doc_id": f"doc-{i}"} for i in range(count)], [],
        [{"operation_id": f"op-{i}"} for i in range(count)], f"last-{count}",
    ) for count in (0, 1, 324, 10_000)]
    assert all(schema == schemas[0] == _ROUTE_SCHEMA for schema in schemas)
    serialized = repr(_ROUTE_SCHEMA)
    assert "doc-324" not in serialized and "op-324" not in serialized
    assert "last-10000" not in serialized
    assert "target_ref" not in _ROUTE_SCHEMA["properties"]
    assert "target_operation_id" in _ROUTE_SCHEMA["properties"]

    def check_node(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                required = node.get("required", [])
                assert len(required) == len(set(required))
                assert set(required) <= set(node.get("properties", {}))
            for child in node.values():
                check_node(child)
        elif isinstance(node, list):
            for child in node:
                check_node(child)
    check_node(_ROUTE_SCHEMA)


def test_effective_action_schema_keeps_root_required_fields_possible():
    from services.connectors.telegram.runtime.routing import _ROUTE_SCHEMA

    required = set(_ROUTE_SCHEMA["required"])
    assert required == set(_ROUTE_SCHEMA["properties"])
    for branch in _ROUTE_SCHEMA["anyOf"]:
        assert branch["properties"]["action"]["enum"]
        # Branches add action policy; the closed root alone owns global fields.
        assert branch.get("additionalProperties") is not False
        assert set(branch.get("required", ())) <= required


def test_scripted_sdk_fixture_adapter_changes_only_valid_route_call_args():
    import json
    from types import SimpleNamespace
    from google.genai import types
    from tests.fixtures.sdk_fakes import _adapt_scripted_route_response

    prompt = {
        "command": "read doc-a", "context": {}, "documents": [{"doc_id": "doc-a"}],
        "response_schema": {"type": "object"},
    }
    contents = [types.Content(parts=[types.Part(text=json.dumps(prompt))])]
    declaration = SimpleNamespace(
        name="route_command",
        parameters_json_schema={
            "properties": {"action": {}, "disposition": {},
                           "substantive_text": {}, "document_selector": {}},
            "required": ["action", "disposition", "substantive_text", "document_selector"],
        },
    )
    config = SimpleNamespace(tools=[SimpleNamespace(function_declarations=[declaration])])
    old_route = {
        "kind": "substantive", "target_ref": "doc-a", "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "read doc-a",
        "tool_choice": None, "certainty": "certain",
    }
    route_part = types.Part(
        function_call=types.FunctionCall(id="route-call-id", name="route_command", args=old_route),
        thought_signature="c3ludGhldGljLXJvdXRlLXNpZw==",
    )
    original_signature = route_part.thought_signature
    unrelated_part = types.Part(text="keep this part")
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[route_part, unrelated_part]),
    )])
    adapted = _adapt_scripted_route_response(
        response, {"contents": contents, "config": config},
    )
    call = adapted.candidates[0].content.parts[0].function_call
    assert call.id == "route-call-id"
    assert call.name == "route_command"
    assert call.args["action"] == "substantive"
    assert call.args["document_selector"]["kind"] == "explicit_reference"
    assert adapted.candidates[0].content.parts[0].thought_signature == original_signature
    assert adapted.candidates[0].content.parts[1].text == "keep this part"

    malformed = {**old_route, "unexpected": True}
    malformed_response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
            id="malformed-call-id", name="route_command", args=malformed,
        ))]),
    )])
    assert _adapt_scripted_route_response(
        malformed_response, {"contents": contents, "config": config},
    ) is malformed_response
    assert _adapt_scripted_route_response(
        response, {"contents": contents, "config": SimpleNamespace(tools=[])},
    ) is response


def test_fresh_selector_and_tool_shapes_match_static_discriminants():
    from services.connectors.telegram.runtime.routing import _ROUTE_SCHEMA
    from services.connectors.telegram.sessions import _valid_response

    bad_url_for_dialog = _fresh("retry", substantive_text="retry",
        document_selector={"kind": "dialog_number", "dialog_number": 2,
            "span": {"kind": "url", "start": 12, "end": 13,
                     "raw_text": "2", "parsed_value": "2"}},
        explicit_references=[])
    assert _valid_response(bad_url_for_dialog, accepted_command="retry #2")[1] == "invalid_route"

    evidence = _fresh("substantive", document_selector={
        "kind": "evidence", "doc_id": "doc-A", "evidence_refs": [],
    })
    assert _valid_response(evidence, accepted_command="answer")[1] == "invalid_route"

    invalid_link_flag = _fresh("substantive", tool_choice={
        "name": "list_folders", "arguments": {}, "link_requested": False,
    })
    assert _valid_response(invalid_link_flag, accepted_command="list folders")[1] == "invalid_route"

    variants = {item["properties"]["action"]["enum"][0]: item["properties"]
                for item in _ROUTE_SCHEMA["anyOf"]}
    assert variants["move_transcript"]["target_operation_id"]["anyOf"]
    assert variants["create_then_move"]["target_operation_id"]["anyOf"]


def test_exact_local_number_and_shown_position_are_distinct():
    from services.connectors.telegram.sessions import _valid_response

    span = {"kind": "local_number", "start": 13, "end": 14,
            "raw_text": "2", "parsed_value": 2}
    valid = _fresh("retry", substantive_text="retry", document_selector={
        "kind": "dialog_number", "dialog_number": 2, "span": span,
    }, explicit_references=[span])
    decision, status = _valid_response(valid, accepted_command="please retry 2")
    assert status == "ok" and decision["dialog_number"] == 2

    invalid = deepcopy(valid)
    invalid["document_selector"]["span"]["parsed_value"] = 3
    assert _valid_response(invalid, accepted_command="please retry 2")[1] == "invalid_route"

    shown = _fresh("substantive", document_selector={
        "kind": "shown_position", "list_id": "opaque-list", "position": 2,
    })
    projected, status = _valid_response(shown, accepted_command="second result")
    assert status == "uncertain"
    assert projected["dialog_number"] is None
    assert projected["semantic_decision"]["document_selector"]["position"] == 2


def test_last_saved_requires_an_explicit_span_and_never_aliases_shown_last():
    from services.connectors.telegram.sessions import _valid_response

    span = {"kind": "last_saved", "start": 13, "end": 17,
            "raw_text": "last", "parsed_value": "last"}
    explicit = _fresh("substantive", substantive_text="continue last",
                      document_selector={"kind": "last_saved", "span": span},
                      explicit_references=[span])
    decision, status = _valid_response(explicit, accepted_command="continue the last",
                                      last_doc_id="doc-last")
    assert status == "ok" and decision["target_ref"] == "last"

    missing = _fresh("substantive", document_selector=None)
    projected, status = _valid_response(missing, accepted_command="continue it")
    assert status == "ok" and projected["target_ref"] is None

    shown_last = _fresh("substantive", document_selector={
        "kind": "shown_position", "list_id": "list-opaque", "position": "last",
    })
    projected, status = _valid_response(shown_last, accepted_command="last result")
    assert status == "uncertain" and projected["target_ref"] is None


def test_last_saved_span_is_not_interpreted_as_a_url_or_resource_reference():
    from services.connectors.telegram.runtime.routing_context import CommandSpan, ExplicitReferenceKind
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore

    span = CommandSpan(ExplicitReferenceKind.LAST_SAVED, 0, 4, "last", "last")
    assert TelegramRuntimeStore._reference_value(span) is None


@pytest.mark.parametrize("literal", [
    "last", "https://docs.google.com/document/d/last/edit",
])
def test_literal_reference_named_last_never_aliases_the_last_saved_selector(literal):
    from services.connectors.telegram.sessions import _valid_response

    command = f"open {literal}"
    span = _reference_span("url" if literal.startswith("https:") else "id",
                           command, literal, literal)
    response = _fresh("substantive", document_selector={
        "kind": "explicit_reference", "raw_value": literal, "span": span,
    }, explicit_references=[span])
    projected, status = _valid_response(
        response, accepted_command=command, last_doc_id="doc-A",
    )
    assert status == "ok" and projected["target_ref"] == "last"
    assert projected["semantic_decision"]["document_selector"]["kind"] == "explicit_reference"


def test_conflicting_explicit_signals_clarify_and_keep_semantic_decision():
    from services.connectors.telegram.sessions import _valid_response

    id_span = {"kind": "id", "start": 5, "end": 10,
               "raw_text": "doc-A", "parsed_value": "doc-A"}
    number_span = {"kind": "local_number", "start": 22, "end": 23,
                   "raw_text": "2", "parsed_value": 2}
    response = _fresh("substantive", document_selector={
        "kind": "explicit_reference", "raw_value": "doc-A", "span": id_span,
    }, explicit_references=[id_span, number_span])
    projected, status = _valid_response(response, accepted_command="move doc-A and dialog 2")
    assert status == "uncertain"
    assert projected["semantic_decision"]["missing_slots"] == ["explicit_conflict"]
    assert len(projected["semantic_decision"]["explicit_references"]) == 2
    assert projected["target_ref"] is None and projected["dialog_number"] is None


def _reference_span(kind, command, literal, parsed):
    start = command.index(literal)
    return {"kind": kind, "start": start, "end": start + len(literal),
            "raw_text": literal, "parsed_value": parsed}


def test_active_selector_cannot_override_an_explicit_different_document():
    from services.connectors.telegram.sessions import _valid_response

    command = "open doc-B"
    span = _reference_span("id", command, "doc-B", "doc-B")
    response = _fresh("substantive", document_selector={"kind": "active_document"},
                      explicit_references=[span])
    projected, status = _valid_response(
        response, accepted_command=command, active_doc_id="doc-A",
    )
    assert status == "uncertain"
    assert projected["target_ref"] is None
    assert projected["semantic_decision"]["explicit_references"] == [span]


def test_unselected_read_tool_argument_does_not_claim_document_selection():
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("substantive", document_selector=None, tool_choice={
        "name": "get_transcript", "arguments": {"doc_id": "doc-A"},
    })
    projected, status = _valid_response(response, accepted_command="And Pavel K?")
    assert status == "ok"
    assert projected["target_ref"] is None
    assert projected["tool_choice"]["arguments"]["doc_id"] == "doc-A"


def test_unselected_move_tool_document_argument_is_never_authority():
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("move_transcript", document_selector=None,
        folder_selector={"kind": "explicit_path", "value": "Archive", "parent": None},
        tool_choice={"name": "move_transcript", "arguments": {
            "doc_id": "doc-A", "folder_path": ["Archive"],
        }})
    projected, status = _valid_response(response, accepted_command="move it to Archive")
    assert status == "uncertain"
    assert projected["target_ref"] is None
    assert set(projected["semantic_decision"]["missing_slots"]) == {
        "document", "explicit_conflict",
    }


def test_active_document_requires_a_unique_captured_card_binding():
    from services.connectors.telegram.models import TelegramContext
    from services.connectors.telegram.sessions import _active_doc_id

    context = TelegramContext(
        chat_id=42, thread_id=7, active_conversation_id="conversation-active",
        selected_dialog_number=1, last_doc_id="doc-last",
    )
    cards = [
        {"doc_id": "doc-A", "conversation_id": "conversation-active"},
        {"doc_id": "doc-B", "conversation_id": "conversation-active"},
    ]
    assert _active_doc_id(cards, context) is None


def test_last_span_conflicts_with_active_and_explicit_selectors():
    from services.connectors.telegram.sessions import _valid_response

    command = "continue last"
    last_span = _reference_span("last_saved", command, "last", "last")
    active = _fresh("substantive", document_selector={"kind": "active_document"},
                    explicit_references=[last_span])
    projected, status = _valid_response(
        active, accepted_command=command, active_doc_id="doc-A", last_doc_id="doc-B",
    )
    assert status == "uncertain" and projected["target_ref"] is None

    command = "continue doc-B last"
    doc_span = _reference_span("id", command, "doc-B", "doc-B")
    last_span = _reference_span("last_saved", command, "last", "last")
    explicit = _fresh("substantive", document_selector={
        "kind": "explicit_reference", "raw_value": "doc-B", "span": doc_span,
    }, explicit_references=[doc_span, last_span])
    agrees, agreed_status = _valid_response(
        explicit, accepted_command=command, last_doc_id="doc-B",
    )
    assert agreed_status == "ok" and agrees["target_ref"] == "doc-B"
    disagrees, disagreed_status = _valid_response(
        explicit, accepted_command=command, last_doc_id="doc-A",
    )
    assert disagreed_status == "uncertain" and disagrees["target_ref"] is None


def test_id_and_recognized_url_are_canonicalized_before_conflict_checks():
    from services.connectors.telegram.sessions import _valid_response

    command = "open doc-A and doc-B or https://docs.google.com/document/d/doc-A/edit"
    id_span = _reference_span("id", command, "doc-A", "doc-A")
    url = "https://docs.google.com/document/d/doc-A/edit"
    url_span = _reference_span("url", command, url, url)
    response = _fresh("substantive", document_selector={
        "kind": "explicit_reference", "raw_value": url, "span": url_span,
    }, explicit_references=[id_span, url_span])
    projected, status = _valid_response(response, accepted_command=command)
    assert status == "ok" and projected["target_ref"] == "doc-A"

    conflict = deepcopy(response)
    id_b = _reference_span("id", command, "doc-B", "doc-B")
    conflict["explicit_references"][0] = id_b
    projected, status = _valid_response(conflict, accepted_command=command)
    assert status == "uncertain" and projected["target_ref"] is None


def test_unsupported_url_is_rejected_instead_of_becoming_an_unselected_route():
    from services.connectors.telegram.sessions import _valid_response

    command = "open https://example.invalid/docs/doc-A"
    url = "https://example.invalid/docs/doc-A"
    span = _reference_span("url", command, url, url)
    response = _fresh("substantive", document_selector={
        "kind": "explicit_reference", "raw_value": url, "span": span,
    }, explicit_references=[span])
    projected, status = _valid_response(response, accepted_command=command)
    assert projected is None and status == "invalid_route"


@pytest.mark.parametrize(("action", "tool_name"), [
    ("substantive", "move_transcript"), ("substantive", "create_folder"),
])
def test_write_tools_cannot_bypass_their_semantic_action(action, tool_name):
    from services.connectors.telegram.sessions import _valid_response

    args = ({"doc_id": "doc-A", "folder_path": ["Archive"]}
            if tool_name == "move_transcript" else {"name": "Archive"})
    response = _fresh(action, tool_choice={"name": tool_name, "arguments": args})
    projected, status = _valid_response(response, accepted_command="write Archive")
    assert projected is None and status == "invalid_route"


def test_complete_move_create_and_composite_project_to_ordinary_routes():
    from services.connectors.telegram.sessions import _valid_response

    command = "move doc-A to Archive"
    span = _reference_span("id", command, "doc-A", "doc-A")
    move = _fresh("move_transcript", substantive_text="move the selected document",
                  document_selector={"kind": "explicit_reference", "raw_value": "doc-A",
                                     "span": span},
                  folder_selector={"kind": "explicit_path", "value": "Archive",
                                   "parent": None}, explicit_references=[span])
    projected, status = _valid_response(move, accepted_command=command)
    assert status == "ok" and projected["kind"] == "substantive"
    assert projected["tool_choice"]["name"] == "move_transcript"
    assert projected["tool_choice"]["arguments"] == {
        "doc_id": "doc-A", "folder_path": ["Archive"],
    }

    create = _fresh("create_folder", substantive_text="create Archive",
                    document_selector=None, folder_selector=None, folder_name="Archive")
    projected, status = _valid_response(create, accepted_command="create Archive")
    assert status == "ok" and projected["kind"] == "substantive"
    assert projected["tool_choice"]["name"] == "create_folder"
    assert projected["target_ref"] is None and projected["dialog_number"] is None

    composite = _fresh("create_then_move", substantive_text="create Archive for doc-A",
                       document_selector={"kind": "explicit_reference", "raw_value": "doc-A",
                                          "span": span},
                       folder_selector={"kind": "root_default", "value": None,
                                        "parent": None}, folder_name="Archive",
                       explicit_references=[span])
    # Use an accepted command containing the same exact document span.
    projected, status = _valid_response(composite, accepted_command=command)
    assert status == "ok" and projected["kind"] == "substantive"
    assert projected["tool_choice"] is None
    assert projected["semantic_decision"]["action"] == "create_then_move"
    assert "ordinal" not in projected and "step0" not in repr(projected)


def test_absolute_destination_respects_parent_scope_and_equivalent_root_arguments():
    from services.connectors.telegram.sessions import _valid_response

    command = "move doc-A to Clients"
    doc_span = _reference_span("id", command, "doc-A", "doc-A")
    move = _fresh("move_transcript", substantive_text="move selected document",
        document_selector={"kind": "explicit_reference", "raw_value": "doc-A",
                           "span": doc_span},
        explicit_references=[doc_span],
        folder_selector={"kind": "explicit_path", "value": "/Clients/Archive",
                         "parent": {"kind": "explicit_path", "value": "/Clients",
                                    "parent": None}},
        tool_choice=None)
    projected, status = _valid_response(move, accepted_command=command)
    assert status == "ok"
    assert projected["tool_choice"]["arguments"]["folder_path"] == ["Clients", "Archive"]

    contradictory = deepcopy(move)
    contradictory["folder_selector"]["parent"]["value"] = "/Other"
    projected, status = _valid_response(contradictory, accepted_command=command)
    assert status == "uncertain"
    semantic = projected["semantic_decision"]
    assert semantic["disposition"] == "clarify"
    assert semantic["certainty"] == "uncertain"
    assert "explicit_conflict" in semantic["missing_slots"]
    assert semantic["document_selector"]["raw_value"] == "doc-A"

    create = _fresh("create_folder", substantive_text="create Archive",
        document_selector=None, folder_selector=None, folder_name="Archive",
        tool_choice={"name": "create_folder", "arguments": {
            "name": "Archive", "parent_path": [],
        }})
    projected, status = _valid_response(create, accepted_command="create Archive")
    assert status == "ok"
    assert projected["tool_choice"]["arguments"] == {"name": "Archive"}


def test_composite_parent_path_conflict_is_retained_for_clarification():
    from services.connectors.telegram.sessions import _valid_response

    command = "create folder for doc-A"
    span = _reference_span("id", command, "doc-A", "doc-A")
    response = _fresh("create_then_move", substantive_text="create Archive for doc-A",
        document_selector={"kind": "explicit_reference", "raw_value": "doc-A",
                           "span": span},
        explicit_references=[span],
        folder_selector={"kind": "explicit_path", "value": "/Clients/Archive",
                         "parent": {"kind": "explicit_path", "value": "/Other",
                                    "parent": None}},
        folder_name="Archive")
    projected, status = _valid_response(response, accepted_command=command)
    assert status == "uncertain"
    semantic = projected["semantic_decision"]
    assert semantic["action"] == "create_then_move"
    assert semantic["disposition"] == "clarify"
    assert semantic["certainty"] == "uncertain"
    assert "explicit_conflict" in semantic["missing_slots"]
    assert semantic["document_selector"]["raw_value"] == "doc-A"
    assert semantic["folder_selector"]["value"] == "/Clients/Archive"
    assert semantic["folder_selector"]["parent"]["value"] == "/Other"
    assert semantic["folder_name"] == "Archive"


def test_numbered_move_uses_unique_scoped_number_binding_not_last_or_list_position():
    from services.connectors.telegram.sessions import _valid_response

    command = "move #2 to Archive"
    span = _reference_span("local_number", command, "#2", 2)
    move = _fresh("move_transcript", substantive_text="move selected transcript",
                  document_selector={"kind": "dialog_number", "dialog_number": 2,
                                     "span": span},
                  explicit_references=[span],
                  folder_selector={"kind": "explicit_path", "value": "Archive",
                                   "parent": None})
    cards = [
        {"doc_id": "doc-list-position-2", "dialog_number": 7},
        {"doc_id": "doc-number-2", "dialog_number": 2},
    ]
    projected, status = _valid_response(move, accepted_command=command, candidates=cards,
                                        last_doc_id="doc-last")
    assert status == "ok" and projected["target_ref"] == "doc-number-2"
    assert projected["tool_choice"]["arguments"]["doc_id"] == "doc-number-2"

    unresolved, status = _valid_response(move, accepted_command=command,
                                         candidates=[cards[0]], last_doc_id="doc-last")
    assert status == "ok" and unresolved["dialog_number"] == 2
    assert unresolved["target_ref"] is None and unresolved["tool_choice"] is None


def test_confirmed_predecessor_can_scope_a_fresh_move_without_model_document_id():
    from services.connectors.telegram.sessions import _valid_response

    candidate = {"operation_id": "op-save", "doc_id": "doc-A",
                 "dialog_number": 2, "conversation_id": "conversation-A"}
    move = _fresh("move_transcript", substantive_text="move saved transcript",
        document_selector=None, target_operation_id="op-save",
        folder_selector={"kind": "explicit_path", "value": "Archive", "parent": None})
    projected, status = _valid_response(
        move, accepted_command="move it to Archive", operation_candidates=[candidate],
    )
    assert status == "ok" and projected["target_ref"] == "doc-A"
    assert projected["tool_choice"]["arguments"]["doc_id"] == "doc-A"

    unknown = deepcopy(move)
    unknown["target_operation_id"] = "op-foreign"
    assert _valid_response(unknown, accepted_command="move it to Archive",
                           operation_candidates=[candidate])[1] == "invalid_route"


def test_confirmed_predecessor_does_not_supply_a_missing_move_destination():
    from services.connectors.telegram.sessions import _valid_response

    candidate = {"operation_id": "op-save", "doc_id": "doc-A",
                 "dialog_number": 2, "conversation_id": "conversation-A"}
    move = _fresh("move_transcript", substantive_text="move saved transcript",
        document_selector=None, target_operation_id="op-save", folder_selector=None)
    projected, status = _valid_response(
        move, accepted_command="move it somewhere", operation_candidates=[candidate],
    )
    assert status == "uncertain"
    semantic = projected["semantic_decision"]
    assert semantic["missing_slots"] == ["destination"]
    assert projected["target_operation_id"] == "op-save"
    assert projected["target_ref"] == "doc-A"


def test_delegate_accepts_read_routes_and_rejects_write_routes():
    from services.connectors.telegram.sessions import _valid_response

    read = _fresh("substantive", disposition="delegate", tool_choice={
        "name": "list_folders", "arguments": {},
    })
    projected, status = _valid_response(read, accepted_command="list folders")
    assert status == "ok" and projected["kind"] == "substantive"

    write = _fresh("substantive", disposition="delegate", tool_choice={
        "name": "rename_transcript", "arguments": {
            "doc_id": "doc-A", "new_title": "New",
        },
    })
    assert _valid_response(write, accepted_command="rename doc-A")[1] == "invalid_route"


@pytest.mark.parametrize("malform", [
    lambda value: value.update(document_selector={"kind": "active_document",
                                                   "raw_value": None}),
    lambda value: value.update(folder_selector={"kind": "create_step_output",
                                                 "value": "step0", "parent": None}),
    lambda value: value.update(document_selector={"kind": "dialog_number",
        "dialog_number": 1, "span": {"kind": "local_number", "start": 0,
        "end": 1, "raw_text": "1", "parsed_value": 1}, "raw_value": None}),
])
def test_fresh_nested_variants_fail_closed_on_schema_extras_or_server_outputs(malform):
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("substantive")
    malform(response)
    assert _valid_response(response, accepted_command="answer this")[1] == "invalid_route"


def test_fresh_action_policies_match_static_shape_requirements():
    from services.connectors.telegram.sessions import _valid_response

    retry_without_number = _fresh("retry", substantive_text="retry",
                                  document_selector=None)
    assert _valid_response(retry_without_number, accepted_command="retry")[1] == "invalid_route"

    search = _fresh("search", substantive_text="find recent notes", document_selector=None,
                    tool_choice={"name": "query", "arguments": {"query": "notes"}})
    search["folder_selector"] = None
    projected, status = _valid_response(search, accepted_command="find recent notes")
    assert status == "ok" and projected["kind"] == "search"
    search["document_selector"] = {"kind": "active_document"}
    assert _valid_response(search, accepted_command="find recent notes")[1] == "invalid_route"

    root = _fresh("substantive", substantive_text=None)
    assert _valid_response(root, accepted_command="hello")[1] == "invalid_route"


def test_evidence_id_without_a_scoped_binding_does_not_project_as_a_target():
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("substantive", document_selector={
        "kind": "evidence", "doc_id": "doc-from-title", "evidence_refs": ["opaque-ref"],
    })
    projected, status = _valid_response(response, accepted_command="that price discussion")
    assert status == "uncertain"
    assert projected["target_ref"] is None
    assert projected["semantic_decision"]["document_selector"]["doc_id"] == "doc-from-title"


@pytest.mark.parametrize("folder_name", [None, "", "  \t"])
def test_folder_creation_retains_parent_and_clarifies_missing_name(folder_name):
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("create_folder", substantive_text="create folder", document_selector=None,
                      folder_selector={"kind": "explicit_path", "value": "Archive", "parent": None},
                      folder_name=folder_name)
    projected, status = _valid_response(response, accepted_command="create folder")
    assert status == "uncertain"
    semantic = projected["semantic_decision"]
    assert semantic["document_selector"] is None
    assert semantic["folder_selector"]["value"] == "Archive"
    assert "folder_name" in semantic["missing_slots"]
    assert semantic["folder_name"] is None


def test_folder_only_creation_defaults_parent_to_root_without_document():
    from services.connectors.telegram.sessions import _valid_response

    response = _fresh("create_folder", substantive_text="create Archive", document_selector=None,
                      folder_selector=None, folder_name="Archive")
    projected, status = _valid_response(response, accepted_command="create Archive")
    assert status == "ok"
    assert projected["kind"] == "substantive"
    assert projected["tool_choice"] == {
        "name": "create_folder", "arguments": {"name": "Archive"},
        "link_requested": False,
    }
    semantic = projected["semantic_decision"]
    assert semantic["document_selector"] is None
    assert semantic["folder_selector"]["kind"] == "root_default"
    assert semantic["folder_name"] == "Archive"

