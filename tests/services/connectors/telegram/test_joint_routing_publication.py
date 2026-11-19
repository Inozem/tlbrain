"""Focused boundaries for contextual route publication capabilities."""

import copy
import hashlib

import pytest

from tests.services.connectors.telegram.test_protocol_compatibility import (
    _t016_save_real_native_round, _t016_seed_real_folder_application,
)
from tests.services.connectors.telegram.test_protocol_compatibility import BASE_TIME


def test_prepared_route_detaches_mutable_inputs_and_publishes_the_original(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    calls = [{
        "id": route["call_id"], "name": route["name"], "arguments": arguments,
        "ordinal": 0, "round": -1,
    }]
    expected_route = copy.deepcopy(route)
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route, tool_calls=calls,
        application_route_round=None,
    )
    route["arguments"]["name"] = "mutated after prepare"
    calls[0]["arguments"]["name"] = "mutated after prepare"
    with pytest.raises(AttributeError):
        prepared.route_digest = "0" * 64

    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    stored = ledger.load_turn(conversation_id, operation_id)
    assert stored["application_route"] == expected_route
    assert stored["tool_calls"] == [{
        "id": expected_route["call_id"], "name": expected_route["name"],
        "arguments": expected_route["arguments"], "ordinal": 0, "round": -1,
    }]
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    ), "an exact joint retry should preserve the immutable pair"
    assert ledger.load_turn(conversation_id, operation_id)["application_route"] == expected_route


def test_executable_contextual_route_without_optional_call_id_replays_stable_id(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from services.connectors.telegram.agent import TelegramAgent

    _agent, ledger, executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, folder_id, summary, _capability = seeded
    generated_id = route.pop("call_id")
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": generated_id, "name": route["name"], "arguments": arguments,
            "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    ledger.save_tool_result(conversation_id, operation_id, 0, folder_id)
    restored = ledger.load_turn(conversation_id, operation_id)
    assert "call_id" not in restored["application_route"]
    assert restored["tool_calls"][0]["id"] == generated_id
    agent = TelegramAgent(model=object(), ledger=ledger, tool_executor=executor)
    assert agent.dispatch_tool_choice(
        name=route["name"], arguments=arguments, target_doc_id=None,
        conversation_id=conversation_id, delivery_id=operation_id,
        executor_factory=lambda *_args: executor,
    ) == folder_id


def test_contextual_actor_must_match_accepted_command_before_either_write(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder_id, summary, _capability = seeded
    from tests.services.connectors.telegram.test_protocol_compatibility import (
        _t016_save_real_native_round,
    )
    _t016_save_real_native_round(
        sdk_fakes, ledger, conversation_id, operation_id,
    )
    native_checkpoint = copy.deepcopy(ledger._ref(operation_id).get().to_dict())
    route["contextual"]["actor_id"] = 18
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    with pytest.raises(PermissionError, match="contextual actor"):
        repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
    assert repository.get_route_control(ledger.context_id, operation_id) is None
    after = ledger._ref(operation_id).get().to_dict()
    assert after["model_response"] == native_checkpoint["model_response"]
    assert after["tool_calls"] == native_checkpoint["tool_calls"]
    assert after.get("application_route") is None


def test_joint_publication_does_not_complete_an_orphan_contextual_half(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder_id, summary, _capability = seeded
    ledger.save_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    turn_ref = ledger._ref(operation_id)
    before = copy.deepcopy(turn_ref.get().to_dict())
    operation_ref = (ledger.client.collection("telegram_sessions").document(ledger.context_id)
                     .collection("operations").document(operation_id))
    operation_ref.update({"route_control": None})
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    ) is False
    assert repository.get_route_control(ledger.context_id, operation_id) is None
    assert turn_ref.get().to_dict() == before


def test_stored_projection_requires_frozen_operation_and_literal_document_bindings():
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, CommandSpan, Disposition, DocumentSelector,
        DocumentSelectorKind, ExplicitReferenceKind, RoutingDecision,
    )
    from services.connectors.telegram.sessions import _validate_stored_joint_projection

    predecessor = "frozen-predecessor-operation"
    operation_decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text="Continue the saved work", certainty=Certainty.CERTAIN,
    )
    operation_route = {"contextual": {
        "decision": operation_decision.to_dict(),
        "resolved": {"target_operation_id": predecessor},
        "accepted_command_ref": {"payload": {
            "text": "Continue the saved work",
            "predecessor_operation_id": predecessor,
        }},
    }}
    operation_summary = {
        "status": "ok", "kind": "substantive",
        "target_operation_id": predecessor,
    }
    _validate_stored_joint_projection(operation_route, operation_summary)
    forged_operation_route = copy.deepcopy(operation_route)
    forged_operation_route["contextual"]["resolved"]["target_operation_id"] = "forged-operation"
    forged_operation_summary = {**operation_summary, "target_operation_id": "forged-operation"}
    with pytest.raises(ValueError, match="frozen accepted-command"):
        _validate_stored_joint_projection(forged_operation_route, forged_operation_summary)

    span = CommandSpan(
        ExplicitReferenceKind.ID, start=5, end=10, raw_text="doc-a", parsed_value="doc-a",
    )
    selector = DocumentSelector(
        kind=DocumentSelectorKind.EXPLICIT_REFERENCE, raw_value="doc-a", span=span,
    )
    selected_decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text="Open doc-a", document_selector=selector,
        explicit_references=(span,), certainty=Certainty.CERTAIN,
    )
    selected_route = {"contextual": {
        "decision": selected_decision.to_dict(), "resolved": {"doc_id": "doc-a"},
        "accepted_command_ref": {"payload": {"text": "Open doc-a"}},
    }}
    _validate_stored_joint_projection(
        selected_route,
        {"status": "ok", "kind": "substantive", "target_ref": "doc-a"},
    )
    mismatched_span = CommandSpan(
        ExplicitReferenceKind.ID, start=5, end=10, raw_text="doc-a", parsed_value="doc-b",
    )
    mismatched_selector = DocumentSelector(
        kind=DocumentSelectorKind.EXPLICIT_REFERENCE, raw_value="doc-b", span=mismatched_span,
    )
    mismatched_decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text="Open doc-a", document_selector=mismatched_selector,
        explicit_references=(mismatched_span,), certainty=Certainty.CERTAIN,
    )
    mismatched_route = {"contextual": {
        "decision": mismatched_decision.to_dict(), "resolved": {"doc_id": "doc-b"},
        "accepted_command_ref": {"payload": {"text": "Open doc-a"}},
    }}
    _validate_stored_joint_projection(
        selected_route,
        {"status": "ok", "kind": "substantive", "target_doc_id": "doc-a"},
    )
    with pytest.raises(ValueError, match="literal selector"):
        _validate_stored_joint_projection(
            selected_route,
            {"status": "ok", "kind": "substantive", "target_doc_id": "doc-b"},
        )
    with pytest.raises(ValueError):
        _validate_stored_joint_projection(
            mismatched_route,
            {"status": "ok", "kind": "substantive", "target_ref": "doc-b"},
        )


def test_clarification_allows_absent_target_but_rejects_conflicting_optional_binding():
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, CommandSpan, Disposition, DocumentSelector,
        DocumentSelectorKind, ExplicitReferenceKind, RoutingDecision,
    )
    from services.connectors.telegram.sessions import _validate_stored_joint_projection

    command = "Move document doc-b"
    start = command.index("doc-b")
    span = CommandSpan(
        ExplicitReferenceKind.ID, start=start, end=start + len("doc-b"),
        raw_text="doc-b", parsed_value="doc-b",
    )
    decision = RoutingDecision(
        disposition=Disposition.CLARIFY, action=Action.MOVE_TRANSCRIPT,
        substantive_text=command,
        document_selector=DocumentSelector(
            kind=DocumentSelectorKind.EXPLICIT_REFERENCE,
            raw_value="doc-b", span=span,
        ),
        explicit_references=(span,), missing_slots=("destination",),
        certainty=Certainty.UNCERTAIN,
    )
    route = {"contextual": {
        "decision": decision.to_dict(), "resolved": {"doc_id": "doc-b"},
        "fixed_plan": None,
    }}
    uncertain = {"status": "uncertain", "kind": "substantive"}
    _validate_stored_joint_projection(route, uncertain)
    with pytest.raises(ValueError, match="literal selector"):
        _validate_stored_joint_projection(
            route, {**uncertain, "target_doc_id": "doc-c"},
        )


def test_retry_clarification_allows_missing_dialog_selector():
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, MissingSlot, RoutingDecision,
    )
    from services.connectors.telegram.sessions import (
        _validate_fresh_action, _validate_stored_joint_projection,
    )

    for missing_slot in (MissingSlot.DOCUMENT, MissingSlot.EXPLICIT_CONFLICT):
        decision = RoutingDecision(
            disposition=Disposition.CLARIFY, action=Action.RETRY,
            substantive_text="retry the failed save",
            document_selector=None, missing_slots=(missing_slot,),
            certainty=Certainty.UNCERTAIN,
        )
        validated = _validate_fresh_action(
            decision, tool_choice=None, target_operation_id=None,
        )
        assert validated.document_selector is None

        application_route = {"contextual": {
            "decision": validated.to_dict(), "resolved": {}, "fixed_plan": None,
        }}
        _validate_stored_joint_projection(
            application_route, {"status": "uncertain", "kind": "retry"},
        )


def test_fixed_plan_rejects_default_or_explicit_parent_conflicts_and_bad_digest(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from dataclasses import replace
    from services.connectors.telegram.runtime.routing_context import (
        FolderSelector, FolderSelectorKind, RoutingDecision,
    )

    _agent, _ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    _operation_id, _conversation_id, route, _arguments, _folder, summary, _capability = seeded
    repository._validate_joint_application_projection(route, summary)

    default_parent = copy.deepcopy(route)
    contextual = default_parent["contextual"]
    decision = RoutingDecision.from_dict(contextual["decision"])
    contextual["decision"] = replace(decision, folder_selector=None).to_dict()
    contextual["resolved"]["parent_id"] = "other-parent"
    contextual["fixed_plan"]["steps"][0]["resolved_targets"]["parent_id"] = "other-parent"
    with pytest.raises(ValueError, match="default folder parent"):
        repository._validate_joint_application_projection(default_parent, summary)

    explicit_parent = copy.deepcopy(route)
    contextual = explicit_parent["contextual"]
    decision = RoutingDecision.from_dict(contextual["decision"])
    contextual["decision"] = replace(
        decision, folder_selector=FolderSelector(FolderSelectorKind.EXPLICIT_ID, "expected-parent"),
    ).to_dict()
    contextual["resolved"]["parent_id"] = "other-parent"
    contextual["fixed_plan"]["steps"][0]["resolved_targets"]["parent_id"] = "other-parent"
    with pytest.raises(ValueError, match="explicit folder parent"):
        repository._validate_joint_application_projection(explicit_parent, summary)

    malformed_digest = copy.deepcopy(route)
    malformed_digest["contextual"]["resolved"]["intent_digest"] = "g" * 64
    malformed_digest["contextual"]["fixed_plan"]["steps"][0]["resolved_targets"][
        "intent_digest"] = "g" * 64
    with pytest.raises(ValueError, match="folder plan targets"):
        repository._validate_joint_application_projection(malformed_digest, summary)

    repository._validate_joint_folder_binding(
        FolderSelector(FolderSelectorKind.ROOT_DEFAULT),
        target_id="captured-root", root_id="captured-root", label="move destination",
    )
    repository._validate_joint_folder_binding(
        FolderSelector(FolderSelectorKind.EXPLICIT_ID, "folder-a"),
        target_id="folder-a", root_id="captured-root", label="move destination",
    )
    with pytest.raises(ValueError, match="move destination.*captured root"):
        repository._validate_joint_folder_binding(
            FolderSelector(FolderSelectorKind.ROOT_DEFAULT),
            target_id="folder-a", root_id="captured-root", label="move destination",
        )
    with pytest.raises(ValueError, match="explicit move destination"):
        repository._validate_joint_folder_binding(
            FolderSelector(FolderSelectorKind.EXPLICIT_ID, "folder-a"),
            target_id="folder-b", root_id="captured-root", label="move destination",
        )


def test_contextual_route_publication_rejects_non_finite_json(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    route["contextual"]["resolved"]["numeric_hint"] = float("nan")
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route, tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }], application_route_round=None,
    )
    with pytest.raises((TypeError, ValueError)):
        repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
    assert repository.get_route_control(ledger.context_id, operation_id) is None


def test_exact_joint_retry_after_new_native_read_preserves_full_appended_state(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from services.connectors.telegram.agent import _calls_from, _serialize_response
    from services.connectors.telegram.agent import TelegramAgent

    _agent, ledger, executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, folder_id, summary, _capability = seeded
    app_call = {
        "id": route["call_id"], "name": route["name"], "arguments": arguments,
        "ordinal": 0, "round": -1,
    }
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[app_call], application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    ledger.save_tool_result(conversation_id, operation_id, 0, folder_id)
    agent = TelegramAgent(model=object(), ledger=ledger, tool_executor=executor)
    assert agent.dispatch_tool_choice(
        name=route["name"], arguments=arguments, target_doc_id=None,
        conversation_id=conversation_id, delivery_id=operation_id,
        executor_factory=lambda *_args: executor,
    ) == folder_id
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(
                id="t023-read-after-app", name="sync_status", args={},
            ), thought_signature=b"t023-post-application-signature",
        )]),
    )])
    native = _calls_from(response)[0]
    native.update({"ordinal": 1, "round": 0})
    read_result = executor.execute(name="sync_status", arguments={}, call_ordinal=1)
    ledger.save_turn(
        conversation_id, operation_id,
        model_response={"_telegram_tool_loop_rounds": 1, "rounds": [{
            "response": _serialize_response(response), "call_start": 1, "call_count": 1,
        }]},
        tool_calls=[app_call, native],
    )
    ledger.save_tool_result(conversation_id, operation_id, 1, read_result)
    before_raw = copy.deepcopy(ledger._ref(operation_id).get().to_dict())
    before_loaded = ledger.load_turn(conversation_id, operation_id)
    assert before_loaded["tool_calls"] == [app_call, native]
    assert before_loaded["tool_results"]["0"] == folder_id
    assert before_loaded["tool_results"]["1"] == read_result

    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    assert ledger._ref(operation_id).get().to_dict() == before_raw


@pytest.mark.parametrize("call_id_length", [4200, 530_000])
def test_joint_publication_round_trips_large_raw_application_route_payloads(
    firestore_fake, sdk_fakes, monkeypatch, call_id_length,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    route["call_id"] = "c" * call_id_length
    calls = [{
        "id": route["call_id"], "name": route["name"], "arguments": arguments,
        "ordinal": 0, "round": -1,
    }]
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route, tool_calls=calls,
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    restored = ledger.load_turn(conversation_id, operation_id)
    assert restored["application_route"] == route
    assert restored["tool_calls"] == calls


def test_legacy_business_mapping_with_digest_looking_keys_is_not_an_envelope(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, _repository, _seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    legacy = {
        "sha256": "business-field-value", "byte_length": 17,
        "max_part_bytes": "customer metadata", "payload": {"result": "preserve"},
    }
    assert ledger._load_payload(legacy, kind="tool-0") == legacy


def test_joint_publication_accepts_real_frozen_t017_capture_shape(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, RoutingDecision,
    )
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    from tests.services.connectors.telegram.test_protocol_compatibility import (
        _executor_for_folder_write,
    )

    now = BASE_TIME
    accepted, _operation, lease, repository, _executor = _executor_for_folder_write(
        firestore_fake.client_class(), now=now, chat_id=93629, monkeypatch=monkeypatch,
    )
    source_conversation = "t023-captured-source"
    checkpoint_conversation = "t023-captured-agent"
    repository.ensure_conversation(
        accepted.context_id, source_conversation, kind="chat",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=now,
    )
    repository.append_turn(
        accepted.context_id, source_conversation, turn_id="t023-accepted-input",
        operation_id=accepted.operation_id, role="user",
        parts=[{"text": "Create the folder Archive"}], now=now,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence,
    )
    store = TelegramRuntimeStore(
        repository, context_id=accepted.context_id, operation_id=accepted.operation_id,
    )
    snapshot = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence,
        search_exchange_loader=lambda *_args, **_kwargs: {}, clock=lambda: now,
    )
    assert "kind" in snapshot.accepted_command_ref.payload
    assert set(snapshot.full_history_ref.payload) == {"turns", "conversation"}
    context_checkpoint = FirestoreAgentLedger(
        repository.client, context_id=accepted.context_id,
        operation_id=accepted.operation_id, owner=lease.owner, fence=lease.fence,
        clock=lambda: now,
    )
    context_reference = context_checkpoint.load_routing_checkpoint(
        "context", actor_id=17,
        conversation_id=snapshot.selected_conversation_id,
    )["reference"]
    decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text=snapshot.accepted_command, certainty=Certainty.CERTAIN,
    )
    summary = {"status": "ok", "kind": "substantive"}
    application_route = {"contextual": {
        "schema_version": 1, "context_ref": context_reference["ref"],
        "context_digest": context_reference["digest"],
        "context_id": accepted.context_id, "operation_id": accepted.operation_id,
        "actor_id": 17, "accepted_command": snapshot.accepted_command,
        "accepted_command_ref": snapshot.accepted_command_ref.to_dict(),
        "full_history_ref": snapshot.full_history_ref.to_dict(),
        "source_conversation_id": snapshot.selected_conversation_id,
        "checkpoint_conversation_id": checkpoint_conversation,
        "decision": decision.to_dict(), "resolved": {}, "summary": summary,
        "fixed_plan": None,
    }}
    prepared = context_checkpoint.prepare_application_route(
        checkpoint_conversation, accepted.operation_id,
        route=application_route, tool_calls=[], application_route_round=None,
    )
    assert repository.persist_route(
        accepted.operation_id, summary, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
        application_checkpoint=prepared,
    )
    loaded = context_checkpoint.load_turn(checkpoint_conversation, accepted.operation_id)
    assert loaded["application_route"] == application_route
    assert loaded["model_response"] is None and loaded["tool_calls"] == []


@pytest.mark.parametrize("staged_mode", ["inline", "parts"])
def test_tampered_prepared_payload_cannot_publish_either_half(
    firestore_fake, sdk_fakes, monkeypatch, staged_mode,
):
    from core.integrations.firestore.input_parts import MAX_PART_BYTES

    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    _t016_save_real_native_round(sdk_fakes, ledger, conversation_id, operation_id)
    if staged_mode == "parts":
        route["call_id"] = "c" * (MAX_PART_BYTES + 20)
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    if staged_mode == "inline":
        object.__setattr__(prepared, "route_payload", prepared.route_payload + b" ")
    else:
        stored = prepared.stored_route
        data_operation = stored["_input_ref"].split("/")[1]
        part_ref = (ledger.client.collection("dialog_operations").document(data_operation)
                    .collection("input_parts").document("part-000000"))
        part = part_ref.get().to_dict()
        part["payload"] = bytes(part["payload"][:-1]) + b"x"
        part_ref.set(part)
    with pytest.raises((ValueError, PermissionError)):
        repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
    assert repository.get_route_control(ledger.context_id, operation_id) is None
    restored = ledger.load_turn(conversation_id, operation_id)
    assert restored["application_route"] is None
    assert restored["model_response"]["rounds"][0]["call_count"] == 0


def test_joint_publisher_rejects_duck_typed_prepared_capability(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, _conversation_id, _route, _arguments, _folder, summary, _capability = seeded
    with pytest.raises(TypeError, match="server-prepared"):
        repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint={"route": "not a capability"},
        )
    assert repository.get_route_control(ledger.context_id, operation_id) is None


def test_identical_legacy_summary_is_not_enriched_by_joint_publication(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
    )
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    ) is False
    assert repository.get_route_control(ledger.context_id, operation_id) == summary
    legacy_turn = ledger.load_turn(conversation_id, operation_id)
    assert legacy_turn is None or legacy_turn.get("application_route") is None


def test_agent_continues_after_server_application_and_keeps_prior_quiet_round(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    _seeded_agent, ledger, executor, _repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    _operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    native_round = _t016_save_real_native_round(
        sdk_fakes, ledger, conversation_id, _operation_id,
    )
    application_call = {
        "id": route["call_id"], "name": route["name"],
        "arguments": arguments, "ordinal": 0, "round": -1,
    }
    prepared = ledger.prepare_application_route(
        conversation_id, _operation_id, route=route,
        tool_calls=[application_call], application_route_round=None,
    )
    assert _repository.persist_route(
        _operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    ledger.save_tool_result(conversation_id, _operation_id, 0, "synthetic-folder-id")

    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text="after the route")],
        ))],
    ))
    model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t023-chronology-key"))
    result = TelegramAgent(model=model, ledger=ledger, tool_executor=executor).run(
        conversation_id=conversation_id, delivery_id=_operation_id,
        history=[{"role": "user", "parts": [{"text": "Create the folder Archive"}]}],
    )
    assert result["answer"] == "after the route"
    events = [event for event in sdk_fakes.trace
              if event["boundary"] == "gemini"
              and event["operation"] == "generate_content"]
    assert len(events) >= 2  # native prefix capture, then replayed application context
    contents = events[-1]["arguments"]["contents"]
    native_index = next(
        index for index, item in enumerate(contents)
        if isinstance(item, dict) and item.get("role") == "model"
        and any(part.get("thought_signature") for part in item.get("parts", [])
                if isinstance(part, dict))
    )
    marker_index = next(
        index for index, item in enumerate(contents)
        if isinstance(item, dict) and any(
            "Untrusted application tool result data follows" in str(part.get("text", ""))
            for part in item.get("parts", []) if isinstance(part, dict)
        )
    )
    assert native_index < marker_index
    assert contents[native_index]["parts"][0]["thought_signature"] == "b3BhcXVlLXQwMTYtc2lnbmF0dXJl"
    stored = ledger.load_turn(conversation_id, _operation_id)
    assert stored["model_response"]["rounds"][0] == native_round["rounds"][0]
    assert stored["model_response"]["rounds"][1]["call_start"] == 1
    assert stored["model_response"]["rounds"][1]["call_count"] == 0
    model_response_after_append = copy.deepcopy(stored["model_response"])
    assert _repository.persist_route(
        _operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    retried = ledger.load_turn(conversation_id, _operation_id)
    assert retried["model_response"] == model_response_after_append
    assert retried["tool_results"] == {"0": "synthetic-folder-id"}


def test_metadata_route_replays_a_real_native_read_call_without_repeating_it(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, RoutingDecision,
    )
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    _seeded_agent, ledger, executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, old_route, _arguments, _folder, _old_summary, _capability = seeded
    accepted_command = old_route["contextual"]["accepted_command"]
    summary = {"status": "ok", "kind": "substantive"}
    decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text=accepted_command, certainty=Certainty.CERTAIN,
    )
    contextual = {
        **old_route["contextual"], "decision": decision.to_dict(),
        "resolved": {}, "summary": summary, "fixed_plan": None,
    }
    route = {"contextual": contextual}
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route, tool_calls=[],
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(
                id="t023-native-sync-read", name="sync_status", args={},
            ), thought_signature=b"t023-native-read-signature"),
        ]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text="The sync is complete.")],
        ))],
    ))
    real_model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t023-read-key"))
    real_calls = []

    class CountingExecutor:
        def execute(self, **kwargs):
            real_calls.append(dict(kwargs))
            return executor.execute(**kwargs)

    agent = TelegramAgent(model=real_model, ledger=ledger, tool_executor=CountingExecutor())
    result = agent.run(
        conversation_id=conversation_id, delivery_id=operation_id,
        history=[{"role": "user", "parts": [{"text": accepted_command}]}],
    )
    assert result["answer"] == "The sync is complete."
    assert len(real_calls) == 1 and real_calls[0]["name"] == "sync_status"
    stored = ledger.load_turn(conversation_id, operation_id)
    assert stored["application_route"] == route
    assert stored["tool_calls"][0]["ordinal"] == 0
    assert stored["tool_calls"][0]["round"] == 0
    assert stored["tool_results"]["0"] == result["tool_results"][0]
    before_replay = copy.deepcopy(sdk_fakes.trace)

    class NoModel:
        def generate(self, **_kwargs):
            raise AssertionError("completed native read replay called the model")

    replay = TelegramAgent(model=NoModel(), ledger=ledger,
                           tool_executor=CountingExecutor()).run(
        conversation_id=conversation_id, delivery_id=operation_id,
        history=[{"role": "user", "parts": [{"text": accepted_command}]}],
    )
    assert replay["answer"] == result["answer"]
    assert replay["tool_results"] == result["tool_results"]
    assert len(real_calls) == 1
    assert sdk_fakes.trace == before_replay


def test_contextual_terminal_presentation_replays_without_shared_call_or_model(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, RoutingDecision,
    )
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore
    from tests.fixtures.sdk_fakes import FakeGeminiClient
    from tests.services.connectors.telegram.test_protocol_compatibility import (
        _executor_for_folder_write,
    )

    accepted, _operation, lease, repository, executor = _executor_for_folder_write(
        firestore_fake.client_class(), now=BASE_TIME, chat_id=93630,
        monkeypatch=monkeypatch,
    )
    source_conversation = "t023-terminal-source"
    checkpoint_conversation = "t023-terminal-agent"
    repository.ensure_conversation(
        accepted.context_id, source_conversation, kind="chat",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    repository.append_turn(
        accepted.context_id, source_conversation, turn_id="t023-terminal-command",
        operation_id=accepted.operation_id, role="user",
        parts=[{"text": "Give me a short update"}], now=BASE_TIME,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence,
    )
    store = TelegramRuntimeStore(
        repository, context_id=accepted.context_id, operation_id=accepted.operation_id,
    )
    snapshot = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence,
        search_exchange_loader=lambda *_args, **_kwargs: {}, clock=lambda: BASE_TIME,
    )
    ledger = FirestoreAgentLedger(
        repository.client, context_id=accepted.context_id,
        operation_id=accepted.operation_id, owner=lease.owner, fence=lease.fence,
        clock=lambda: BASE_TIME,
    )
    context_ref = ledger.load_routing_checkpoint(
        "context", actor_id=17, conversation_id=source_conversation,
    )["reference"]
    summary = {"status": "ok", "kind": "substantive"}
    decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text=snapshot.accepted_command, certainty=Certainty.CERTAIN,
    )
    route = {"contextual": {
        "schema_version": 1, "context_ref": context_ref["ref"],
        "context_digest": context_ref["digest"],
        "context_id": accepted.context_id, "operation_id": accepted.operation_id,
        "actor_id": 17, "accepted_command": snapshot.accepted_command,
        "accepted_command_ref": snapshot.accepted_command_ref.to_dict(),
        "full_history_ref": snapshot.full_history_ref.to_dict(),
        "source_conversation_id": source_conversation,
        "checkpoint_conversation_id": checkpoint_conversation,
        "decision": decision.to_dict(), "resolved": {}, "summary": summary,
        "fixed_plan": None,
    }}
    prepared = ledger.prepare_application_route(
        checkpoint_conversation, accepted.operation_id, route=route,
        tool_calls=[], application_route_round=None,
    )
    assert repository.persist_route(
        accepted.operation_id, summary, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(
                id="terminal-answer", name="present_search_answer", args={"blocks": [
                    {"type": "paragraph", "text": "A durable terminal answer.",
                     "source_refs": []},
                ]},
            )),
        ]))],
    ))
    model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t023-terminal-key"))
    source_loads = []
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    scope = {
        "context_id": accepted.context_id,
        "conversation_id": source_conversation,
        "producing_operation_id": accepted.operation_id,
        "presentation_id": "t023-terminal-presentation",
        "answer_turn_id": "t023-terminal-answer",
    }

    def sources():
        source_loads.append(True)
        return []

    result = agent.run(
        conversation_id=checkpoint_conversation, delivery_id=accepted.operation_id,
        history=[{"role": "user", "parts": [{"text": snapshot.accepted_command}]}],
        presentation_scope=scope, presentation_sources=sources,
    )
    assert result["answer"] == "A durable terminal answer."
    stored = ledger.load_turn(checkpoint_conversation, accepted.operation_id)
    assert stored["tool_calls"] == [] and stored["tool_results"] == {}
    assert len(stored["model_response"]["rounds"]) == 1
    terminal_round = stored["model_response"]["rounds"][0]
    assert terminal_round["call_count"] == 0
    assert len(terminal_round["response"]["candidates"][0]["content"]["parts"]) == 1
    assert terminal_round["response"]["candidates"][0]["content"]["parts"][0]["function_call"]["name"] == "present_search_answer"
    assert terminal_round["terminal"]["status"] == "ok"
    trace = copy.deepcopy(sdk_fakes.trace)
    assert len(source_loads) == 1

    replay = agent.run(
        conversation_id=checkpoint_conversation, delivery_id=accepted.operation_id,
        history=[{"role": "user", "parts": [{"text": snapshot.accepted_command}]}],
        presentation_scope=scope, presentation_sources=sources,
    )
    assert replay["answer"] == result["answer"]
    assert replay["presentation_manifest"] == result["presentation_manifest"]
    assert len(source_loads) == 1
    assert sdk_fakes.trace == trace

    raw_turn = ledger._ref(accepted.operation_id).get().to_dict()
    raw_response = raw_turn["model_response"]["_inline"]
    raw_response["rounds"][0]["terminal"]["manifest_sha256"] = "0" * 64
    raw_turn["model_response"] = ledger._store_payload("model", raw_response)
    ledger._ref(accepted.operation_id).set(raw_turn, merge=False)
    invalid = agent.run(
        conversation_id=checkpoint_conversation, delivery_id=accepted.operation_id,
        history=[{"role": "user", "parts": [{"text": snapshot.accepted_command}]}],
        presentation_scope=scope, presentation_sources=sources,
    )
    assert invalid["status"] == "failed"
    assert len(source_loads) == 1 and sdk_fakes.trace == trace


@pytest.mark.parametrize("tamper", [
    "corrupt_inline", "unknown_contextual", "missing_summary",
    "legacy_unverified_contextual",
])
def test_contextual_reader_and_dispatch_fail_closed_before_executor(
    firestore_fake, sdk_fakes, monkeypatch, tamper,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    )
    turn_ref = ledger._ref(operation_id)
    turn = turn_ref.get().to_dict()
    if tamper == "missing_summary":
        operation_ref = (ledger.client.collection("telegram_sessions").document(ledger.context_id)
                         .collection("operations").document(operation_id))
        operation_ref.update({"route_control": {"status": "ok", "kind": "substantive"}})
    else:
        stored = turn["application_route"]
        payload = stored["_inline"]
        if tamper == "corrupt_inline":
            payload["contextual"]["decision"]["action"] = "unknown-action"
            # Leave the original digest untouched to exercise payload integrity.
            turn["application_route"] = stored
            turn_ref.set(turn)
        elif tamper == "unknown_contextual":
            payload["contextual"]["unexpected"] = True
            encoded = ledger._json_bytes(payload)
            stored = {"_inline": payload,
                      "sha256": __import__("hashlib").sha256(encoded).hexdigest(),
                      "byte_length": len(encoded)}
            turn["application_route"] = stored
            turn_ref.set(turn)
        else:
            turn["application_route"] = {"_inline": payload}
            turn_ref.set(turn)
    with pytest.raises(ValueError):
        ledger.load_turn(conversation_id, operation_id)
    if tamper != "missing_summary":
        turn_ref.set(turn)
    constructed = []
    from services.connectors.telegram.agent import TelegramAgent
    agent = TelegramAgent(model=object(), ledger=ledger, tool_executor=object())
    with pytest.raises(ValueError, match="persisted application checkpoint is invalid"):
        agent.dispatch_tool_choice(
            name=route["name"], arguments=arguments, target_doc_id=None,
            conversation_id=conversation_id, delivery_id=operation_id,
            executor_factory=lambda *_args: constructed.append(True),
        )
    assert constructed == []


def test_prepared_capability_is_bound_to_issuing_client_and_live_fence(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.operation_queue import release_head
    from services.connectors.telegram.repository import TelegramRepository

    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    foreign = TelegramRepository(client=firestore_fake.client_class())
    with pytest.raises(ValueError, match="scope changed"):
        foreign.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
    context_ref = repository.client.collection("telegram_sessions").document(ledger.context_id)
    assert release_head(
        repository.client, context_ref, operation_id=operation_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
    )
    assert repository.persist_route(
        operation_id, summary, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
        application_checkpoint=prepared,
    ) is False
    assert repository.get_route_control(ledger.context_id, operation_id) is None


@pytest.mark.parametrize("failure", [
    "metadata_synthetic", "native_mismatch", "extra_native", "native_valid",
])
def test_joint_publisher_rejects_calls_without_exact_native_or_server_provenance(
    firestore_fake, sdk_fakes, monkeypatch, failure,
):
    from google.genai import types
    from services.connectors.telegram.agent import _calls_from, _serialize_response

    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    if failure == "metadata_synthetic":
        contextual = {
            **route["contextual"], "resolved": {},
            "summary": {"status": "ok", "kind": "substantive"}, "fixed_plan": None,
        }
        contextual["decision"] = {
            **contextual["decision"], "action": "substantive",
            "disposition": "delegate", "certainty": "certain",
        }
        route = {"contextual": contextual}
        summary = contextual["summary"]
        calls = [{"id": "invented", "name": "sync_status", "arguments": {},
                  "ordinal": 0, "round": -1}]
    else:
        response = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(
                    id="t023-native-read", name="sync_status", args={},
                ), thought_signature=b"t023-native-signature",
            )]),
        )])
        serialized = _serialize_response(response)
        native = _calls_from(response)[0]
        if failure == "native_mismatch":
            native["id"] = "forged-native-id"
        native.update({"ordinal": 0, "round": 0})
        ledger.save_turn(
            conversation_id, operation_id,
            model_response=(serialized if failure == "native_valid" else {
                "_telegram_tool_loop_rounds": 1, "rounds": [{
                    "response": serialized, "call_start": 0, "call_count": 1,
                }],
            }),
            tool_calls=[] if failure == "native_mismatch" else [native],
        )
        step = route["contextual"]["fixed_plan"]["steps"][0]
        ordinal = 1
        identity = hashlib.sha256(f"{operation_id}\0{ordinal}".encode()).hexdigest()
        step["ordinal"] = ordinal
        step["operation_id"] = "telegram-tool-" + identity
        step["effect_id"] = "telegram-effect-" + identity
        route["ordinal"] = ordinal
        route["call_id"] = "t023-server-application-1"
        calls = [native, {
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": ordinal, "round": -1,
        }]
        if failure == "extra_native":
            calls.append({"id": "invented-extra", "name": "sync_status",
                          "arguments": {}, "ordinal": 2, "round": -1})
    prepared = ledger.prepare_application_route(
        conversation_id, operation_id, route=route, tool_calls=calls,
        application_route_round=None,
    )
    if failure == "native_valid":
        assert repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
        restored = ledger.load_turn(conversation_id, operation_id)
        assert restored["application_route"]["ordinal"] == 1
        assert restored["tool_calls"] == calls
        assert restored["model_response"] == serialized
        return
    with pytest.raises((ValueError, PermissionError)):
        repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence, now=BASE_TIME,
            application_checkpoint=prepared,
        )
    assert repository.get_route_control(ledger.context_id, operation_id) is None
    restored = ledger.load_turn(conversation_id, operation_id)
    assert restored is None or restored["application_route"] is None

