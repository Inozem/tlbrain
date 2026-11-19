"""Structural validation tests for private context-v2 routing records."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tests.fixtures.telegram_contextual import BASE_TIME, seed_history
from services.connectors.telegram.runtime.routing_context import (
    Action,
    BoundRef,
    Certainty,
    CommandSpan,
    Disposition,
    DocumentEvidence,
    DocumentSelector,
    DocumentSelectorKind,
    ExplicitReferenceKind,
    FolderSelector,
    FolderSelectorKind,
    MissingSlot,
    RoutingContextSnapshot,
    RoutingDecision,
    SourceKind,
)


DIGEST = "a" * 64


def ref(*, ref_id="ref-1", context="ctx", actor=123, conversation="conv-1",
        operation="op-1", fragment=False, ordinal=2, payload=None):
    return BoundRef(
        ref=ref_id,
        digest=DIGEST,
        context_id=context,
        actor_id=actor,
        conversation_id=conversation,
        operation_id=operation,
        checkpoint_id="checkpoint-1" if fragment else None,
        call_id="call-1" if fragment else None,
        ordinal=ordinal if fragment else None,
        payload=payload,
    )


def snapshot(**overrides):
    values = {
        "context_id": "ctx",
        "actor_id": 123,
        "operation_id": "op-current",
        "selected_conversation_id": "conv-1",
        "selected_conversation_kind": "search",
        "accepted_command": "😀 move dialog №2",
        "accepted_command_ref": ref(ref_id="command", operation="op-current"),
        "full_history_ref": ref(ref_id="history", operation="op-earlier"),
        "explicit_references": (CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, 14, 16, "№2", 2),),
    }
    values.update(overrides)
    return RoutingContextSnapshot(**values)


def evidence(doc_id, kind, *, source=None, conversation="conv-1", operation="op-old",
             predecessor=None, fragments=(), context="ctx", actor=123, metadata=None,
             command_span=None):
    source = source or ref(ref_id=f"source-{doc_id}", context=context, actor=actor,
                           conversation=conversation, operation=operation)
    return DocumentEvidence(
        doc_id=doc_id,
        source_kind=kind,
        source=source,
        context_id=context,
        actor_id=actor,
        conversation_id=conversation,
        operation_id=operation,
        fragment_refs=fragments,
        command_span=command_span,
        predecessor_operation_id=predecessor,
        metadata=metadata if metadata is not None else {"title": "inert title", "participants": ["A"]},
    )


def test_folder_only_snapshot_and_decision_roundtrip_with_null_document():
    item = snapshot(selected_doc_id=None, explicit_references=(), local_number=None)
    assert RoutingContextSnapshot.from_dict(json.loads(json.dumps(item.to_dict()))) == item

    decision = RoutingDecision(
        disposition=Disposition.EXECUTE,
        action=Action.CREATE_FOLDER,
        substantive_text="Create an archive folder",
        folder_selector=FolderSelector(FolderSelectorKind.ROOT_DEFAULT),
        folder_name="Archive",
        document_selector=None,
        certainty=Certainty.CERTAIN,
    )
    restored = RoutingDecision.from_dict(json.loads(json.dumps(decision.to_dict())))
    assert restored == decision
    assert restored.document_selector is None
    assert restored.folder_name == "Archive"


def test_all_source_kinds_and_full_refs_roundtrip_including_cross_conversation_save():
    predecessor = ref(ref_id="save-result", conversation="older-conv", operation="op-save")
    explicit_span = CommandSpan(ExplicitReferenceKind.ID, 17, 22, "doc-B", "doc-B")
    accepted = snapshot()
    spans = (*accepted.explicit_references, explicit_span)
    presentation = ref(ref_id="list", operation="op-older")
    evidence_map = {
        "doc-active": (evidence("doc-active", SourceKind.ACTIVE_DOCUMENT),),
        "doc-explicit": (evidence("doc-explicit", SourceKind.EXPLICIT_REFERENCE,
                                   command_span=explicit_span),),
        "doc-number": (evidence("doc-number", SourceKind.LOCAL_NUMBER,
                                command_span=accepted.explicit_references[0]),),
        "doc-exchange": (evidence("doc-exchange", SourceKind.VERIFIED_EXCHANGE,
                                  fragments=(ref(ref_id="fragment", operation="op-older",
                                                 fragment=True, ordinal=0),)),),
        "doc-presentation": (evidence("doc-presentation", SourceKind.PRESENTATION_ENTRY,
                                      source=presentation, operation=presentation.operation_id),),
        "doc-save": (evidence("doc-save", SourceKind.CONFIRMED_SAVE, source=predecessor,
                              conversation="older-conv", operation="op-save", predecessor="op-save"),),
        "doc-multi": (evidence("doc-multi", SourceKind.VERIFIED_EXCHANGE,
                               fragments=(ref(ref_id="fragment-a", operation="op-a", fragment=True),)),
                      evidence("doc-multi", SourceKind.VERIFIED_EXCHANGE,
                               fragments=(ref(ref_id="fragment-b", operation="op-b", fragment=True),))),
    }
    item = snapshot(accepted_command="😀 move dialog №2 doc-B", selected_doc_id="doc-active",
                    explicit_references=spans, parsed_id="doc-B", local_number=2,
                    document_evidence=evidence_map,
                    presentation_refs=(presentation,),
                    confirmed_predecessor_saves=(predecessor,))
    restored = RoutingContextSnapshot.from_dict(json.loads(json.dumps(item.to_dict())))
    assert restored.to_dict() == item.to_dict()
    assert set(restored.document_evidence) == set(evidence_map)
    assert restored.document_evidence["doc-exchange"][0].fragment_refs[0].call_id == "call-1"
    assert restored.document_evidence["doc-save"][0].source.conversation_id == "older-conv"
    assert restored.document_evidence["doc-save"][0].metadata["title"] == "inert title"
    assert len(restored.document_evidence["doc-multi"]) == 2
    assert restored.document_evidence["doc-explicit"][0].command_span == explicit_span


def test_real_seeded_ledger_reference_keeps_canonical_payload_path_and_zero_ordinal(firestore_fake):
    client = firestore_fake.client_class()
    seeded = seed_history(
        client,
        context_id="42017_73",
        now=BASE_TIME,
        operation_id="op-ledger-roundtrip",
        conversation_id="ledger-search",
        exchanges=({
            "call": {"id": "call-zero", "name": "query", "arguments": {"query": "price"},
                     "ordinal": 0, "round": 0},
            "result": {"segments": [{"doc_id": "doc-real", "text": "€42🧾" * 1000}]},
            "source_refs": [{"doc_id": "doc-real", "order_index": 0}],
        },),
    )
    reference = seeded["references"][0]
    loaded = seeded["ledger"].load_search_exchange(
        reference,
        conversation_id=seeded["conversation_id"],
        operation_id=seeded["operation_id"],
        checkpoint_conversation_id="private-search-checkpoint",
        call_id="call-zero",
        ordinal=0,
    )
    storage_ref = reference["payload"]["_input_ref"]
    assert storage_ref.startswith("dialog_operations/")
    assert storage_ref.endswith("/input_parts/manifest")
    source = BoundRef(
        ref=storage_ref,
        digest=reference["payload"]["sha256"],
        context_id=seeded["context_id"],
        actor_id=123,
        conversation_id=seeded["conversation_id"],
        operation_id=seeded["operation_id"],
        checkpoint_id=reference["checkpoint_conversation_id"],
        call_id=reference["call_id"],
        ordinal=reference["ordinal"],
        payload=reference,
    )
    restored = BoundRef.from_dict(json.loads(json.dumps(source.to_dict())))
    assert restored.to_dict() == source.to_dict()
    assert restored.ordinal == 0
    assert restored.ref == storage_ref
    assert restored.payload == source.payload


def test_exact_unicode_span_uses_python_codepoint_indices_and_rejects_wrong_text_or_bounds():
    item = snapshot()
    item.validate_command_spans("😀 move dialog №2")
    with pytest.raises(ValueError, match="does not match"):
        item.validate_command_spans("😀 move dialog XX")
    with pytest.raises(ValueError, match="does not match"):
        item.validate_command_spans("😀 move dialog №")
    wrong_span = CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, 0, 2, "😀", 2)
    with pytest.raises(ValueError, match="does not match"):
        snapshot(explicit_references=(wrong_span,))


@pytest.mark.parametrize("change", [
    {"context_id": "other"},
    {"actor_id": 124},
    {"conversation_id": "other-conversation"},
])
def test_evidence_with_cross_scope_source_is_rejected(change):
    source = ref(ref_id="foreign", context=change.get("context_id", "ctx"),
                 actor=change.get("actor_id", 123),
                 conversation=change.get("conversation_id", "conv-1"))
    with pytest.raises(ValueError):
        snapshot(document_evidence={"doc": evidence("doc", SourceKind.VERIFIED_EXCHANGE, source=source)})


def test_unknown_provenance_key_and_document_map_key_mismatch_fail_closed():
    with pytest.raises(ValueError, match="unsupported source_kind"):
        evidence("doc", "made_up")
    item = evidence("doc", SourceKind.VERIFIED_EXCHANGE)
    with pytest.raises(ValueError, match="key must equal"):
        snapshot(document_evidence={"different-doc": item})
    payload = snapshot().to_dict()
    payload["untrusted_extra"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        RoutingContextSnapshot.from_dict(payload)


def test_fragment_refs_require_checkpoint_call_and_nonnegative_ordinal():
    with pytest.raises(ValueError, match="require checkpoint_id"):
        BoundRef("fragment", DIGEST, "ctx", 123, "conv-1", "op", checkpoint_id="cp")
    fragment = ref(fragment=True, ordinal=0)
    assert fragment.is_fragment
    assert fragment.ordinal == 0
    payload = fragment.to_dict()
    payload["ordinal"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        BoundRef.from_dict(payload)
    payload["ordinal"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        BoundRef.from_dict(payload)


def test_invalid_digest_and_full_opaque_reference_payload_are_preserved():
    with pytest.raises(ValueError, match="SHA-256"):
        BoundRef("ref", "not-a-digest", "ctx", 123, "conv-1", "op")
    original = {"kind": "ledger_search_exchange", "payload": {"exchange_id": "exchange-7", "parts": [1, 2]}}
    item = BoundRef("exchange-ref", DIGEST, "ctx", 123, "conv-1", "op", payload=original)
    original["payload"]["parts"].append(3)
    restored = BoundRef.from_dict(json.loads(json.dumps(item.to_dict())))
    assert restored.to_dict()["payload"] == {
        "kind": "ledger_search_exchange", "payload": {"exchange_id": "exchange-7", "parts": [1, 2]},
    }
    assert restored.to_dict() == item.to_dict()


def test_metadata_without_source_cannot_be_evidence():
    with pytest.raises(TypeError, match="BoundRef"):
        DocumentEvidence("doc", SourceKind.VERIFIED_EXCHANGE, None, "ctx", 123,
                         "conv-1", "op", metadata={"title": "looks authoritative"})


def test_active_selected_document_and_other_explicit_document_are_both_representable():
    active = DocumentSelector(DocumentSelectorKind.ACTIVE_DOCUMENT)
    explicit_span = CommandSpan(ExplicitReferenceKind.ID, 0, 5, "doc-B", "doc-B")
    explicit = DocumentSelector(DocumentSelectorKind.EXPLICIT_REFERENCE, raw_value="doc-B", span=explicit_span)
    assert active.kind is DocumentSelectorKind.ACTIVE_DOCUMENT
    assert explicit.raw_value == "doc-B"
    decision = RoutingDecision(Disposition.EXECUTE, Action.MOVE_TRANSCRIPT, "move B",
                               document_selector=explicit)
    decision.validate_command_spans("doc-B")
    with pytest.raises(ValueError, match="inconsistent with explicit references"):
        RoutingDecision(Disposition.EXECUTE, Action.MOVE_TRANSCRIPT, "move B",
                        document_selector=explicit,
                        explicit_references=(CommandSpan(ExplicitReferenceKind.ID, 0, 3, "doc", "doc"),)
                        ).validate_command_spans("doc-B")
    # Selector-only evidence still receives a contextual bounds/text check.
    selector_only = RoutingDecision(Disposition.EXECUTE, Action.MOVE_TRANSCRIPT, "move B",
                                    document_selector=explicit)
    with pytest.raises(ValueError, match="does not match"):
        selector_only.validate_command_spans("other")
    with pytest.raises(TypeError, match="must be text"):
        selector_only.validate_command_spans(None)


@pytest.mark.parametrize("raw_value", [2, True])
def test_explicit_selector_rejects_non_text_raw_value_in_constructor_and_codec(raw_value):
    span = CommandSpan(ExplicitReferenceKind.ID, 0, 1, "2", "2")
    with pytest.raises(ValueError, match="raw_value must be non-empty text"):
        DocumentSelector(DocumentSelectorKind.EXPLICIT_REFERENCE, raw_value=raw_value, span=span)
    payload = {"kind": "explicit_reference", "raw_value": raw_value,
               "span": span.to_dict()}
    with pytest.raises(ValueError, match="raw_value must be non-empty text"):
        DocumentSelector.from_dict(payload)


def test_explicit_selector_rejects_local_number_span_and_roundtrips_url_span():
    number_span = CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, 0, 2, "№2", 2)
    with pytest.raises(ValueError, match="ID or URL"):
        DocumentSelector(DocumentSelectorKind.EXPLICIT_REFERENCE, raw_value="2", span=number_span)
    payload = {"kind": "explicit_reference", "raw_value": "2",
               "span": number_span.to_dict()}
    with pytest.raises(ValueError, match="ID or URL"):
        DocumentSelector.from_dict(payload)

    url = "https://example.test/document/123"
    url_span = CommandSpan(ExplicitReferenceKind.URL, 0, len(url), url, url)
    selector = DocumentSelector(DocumentSelectorKind.EXPLICIT_REFERENCE,
                                raw_value=url, span=url_span)
    assert DocumentSelector.from_dict(selector.to_dict()) == selector


def test_local_dialog_number_and_shown_position_are_distinct_typed_selectors():
    span = CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, 0, 2, "№2", 2)
    local = DocumentSelector(DocumentSelectorKind.DIALOG_NUMBER, span=span, dialog_number=2)
    position = DocumentSelector(DocumentSelectorKind.SHOWN_POSITION, list_id="list-1", position=2)
    last = DocumentSelector(DocumentSelectorKind.SHOWN_POSITION, list_id="list-1", position="last")
    assert local.kind is not position.kind
    assert position.position == 2 and last.position == "last"
    with pytest.raises(ValueError, match="positive integer"):
        DocumentSelector(DocumentSelectorKind.SHOWN_POSITION, list_id="list-1", position=True)


def test_explicit_last_saved_is_distinct_and_round_trips_with_exact_span():
    raw = "последний сохранённый диалог"
    start = raw.index("последний")
    token = raw[start:start + len("последний")]
    span = CommandSpan(ExplicitReferenceKind.LAST_SAVED, start, start + len(token),
                       token, token)
    selector = DocumentSelector(DocumentSelectorKind.LAST_SAVED, span=span)
    assert DocumentSelector.from_dict(selector.to_dict()) == selector
    shown_last = DocumentSelector(DocumentSelectorKind.SHOWN_POSITION,
                                  list_id="opaque-list", position="last")
    assert shown_last != selector
    with pytest.raises(ValueError, match="last-saved selector"):
        CommandSpan(ExplicitReferenceKind.LAST_SAVED, 0, 4, "last", "other")


def test_folder_selector_and_missing_slots_are_independent_from_document():
    parent = FolderSelector(FolderSelectorKind.EXPLICIT_ID, "folder-parent")
    decision = RoutingDecision(
        disposition=Disposition.CLARIFY,
        action=Action.CREATE_FOLDER,
        substantive_text="Create a folder",
        document_selector=None,
        folder_selector=parent,
        folder_name=None,
        missing_slots=(MissingSlot.FOLDER_NAME,),
        certainty=Certainty.UNCERTAIN,
    )
    restored = RoutingDecision.from_dict(decision.to_dict())
    assert restored.document_selector is None
    assert restored.folder_selector == parent
    assert restored.missing_slots == (MissingSlot.FOLDER_NAME,)
    assert restored.folder_name is None


def test_routing_decision_wire_contract_has_no_operation_protocol():
    decision = RoutingDecision(
        Disposition.EXECUTE,
        Action.MOVE_TRANSCRIPT,
        "move the second item",
        document_selector=DocumentSelector(
            kind=DocumentSelectorKind.SHOWN_POSITION, list_id="prices", position=2,
        ),
    )

    encoded = decision.to_dict()
    assert "protocol" not in encoded
    assert RoutingDecision.from_dict(encoded) == decision

    # The protocol remains a private snapshot/checkpoint tag. It is not a model
    # decision field, so an obsolete marker cannot silently enter the DTO.
    encoded["protocol"] = "context-v2"
    with pytest.raises(ValueError, match="unknown fields"):
        RoutingDecision.from_dict(encoded)


@pytest.mark.parametrize("payload, message", [
    ({"protocol": "context-v1"}, "protocol"),
    ({"schema_version": True}, "schema_version"),
    ({"actor_id": True}, "actor_id"),
])
def test_snapshot_rejects_unknown_protocol_version_and_boolean_identity(payload, message):
    values = snapshot().to_dict()
    values.update(payload)
    with pytest.raises(ValueError, match=message):
        RoutingContextSnapshot.from_dict(values)


def test_decision_rejects_unknown_enums_bad_ids_and_boolean_generation():
    values = RoutingDecision(Disposition.CLARIFY, Action.MOVE_TRANSCRIPT, "move").to_dict()
    values["action"] = "unlisted_action"
    with pytest.raises(ValueError, match="unsupported action"):
        RoutingDecision.from_dict(values)
    values["action"] = Action.MOVE_TRANSCRIPT.value
    values["clarification_id"] = "clarification"
    values["clarification_generation"] = True
    with pytest.raises(ValueError, match="positive integer"):
        RoutingDecision.from_dict(values)


def test_snapshot_payload_is_copy_isolated_and_frozen():
    supplied_metadata = {"title": "before", "nested": {"labels": ["first"]}}
    presentation = ref(ref_id="presentation")
    item = snapshot(document_evidence={
        "doc": evidence("doc", SourceKind.PRESENTATION_ENTRY, metadata=supplied_metadata,
                         source=presentation, operation=presentation.operation_id),
    }, presentation_refs=(presentation,))
    supplied_metadata["title"] = "after"
    supplied_metadata["nested"]["labels"].append("second")
    metadata = item.document_evidence["doc"][0].metadata
    assert metadata["title"] == "before"
    assert metadata["nested"]["labels"] == ("first",)
    with pytest.raises(TypeError):
        metadata["title"] = "mutated"
    encoded = item.to_dict()
    encoded["document_evidence"]["doc"][0]["metadata"]["title"] = "payload mutation"
    assert item.document_evidence["doc"][0].metadata["title"] == "before"


def test_document_evidence_rejects_one_foreign_entry_in_multi_source_value():
    local = evidence("doc", SourceKind.VERIFIED_EXCHANGE,
                     fragments=(ref(ref_id="local-fragment", fragment=True),))
    foreign = evidence("doc", SourceKind.VERIFIED_EXCHANGE,
                       source=ref(ref_id="foreign-source", context="other"),
                       fragments=(ref(ref_id="foreign-fragment", context="other", fragment=True),),
                       context="other")
    with pytest.raises(ValueError, match="crosses context or actor"):
        snapshot(document_evidence={"doc": (local, foreign)})


def test_provenance_kinds_require_their_structural_anchors():
    with pytest.raises(ValueError, match="selected_doc_id"):
        snapshot(selected_doc_id="selected", document_evidence={
            "other": (evidence("other", SourceKind.ACTIVE_DOCUMENT),),
        })
    presentation = ref(ref_id="unshown-list")
    with pytest.raises(ValueError, match="not anchored"):
        snapshot(document_evidence={
            "doc": (evidence("doc", SourceKind.PRESENTATION_ENTRY, source=presentation,
                             operation=presentation.operation_id),),
        })
    with pytest.raises(ValueError, match="checkpoint/call/ordinal"):
        snapshot(document_evidence={
            "doc": (evidence("doc", SourceKind.VERIFIED_EXCHANGE),),
        })
    with pytest.raises(ValueError, match="exact command span"):
        evidence("doc", SourceKind.EXPLICIT_REFERENCE)
    span = CommandSpan(ExplicitReferenceKind.ID, 0, 5, "doc-B", "doc-B")
    with pytest.raises(ValueError, match="not a captured explicit reference"):
        snapshot(document_evidence={
            "doc": (evidence("doc", SourceKind.EXPLICIT_REFERENCE, command_span=span),),
        })


def test_typed_payload_metadata_and_collections_reject_malformed_shapes():
    with pytest.raises(TypeError, match="payload must be an object"):
        BoundRef("ref", DIGEST, "ctx", 123, "conv-1", "op", payload=["not", "mapping"])
    with pytest.raises(TypeError, match="metadata must be an object"):
        DocumentEvidence("doc", SourceKind.VERIFIED_EXCHANGE,
                         ref(ref_id="x", fragment=True), "ctx", 123, "conv-1", "op",
                         metadata=["not", "mapping"])
    with pytest.raises(TypeError, match="evidence_refs must be an array"):
        DocumentSelector(DocumentSelectorKind.EVIDENCE, doc_id="doc", evidence_refs="ref/path")
    with pytest.raises(TypeError, match="reference strings"):
        DocumentSelector(DocumentSelectorKind.EVIDENCE, doc_id="doc", evidence_refs=[1])
    with pytest.raises(TypeError, match="explicit_references must be an array"):
        snapshot(explicit_references="№2")
    with pytest.raises(TypeError, match="fragment_refs must be an array"):
        DocumentEvidence("doc", SourceKind.VERIFIED_EXCHANGE,
                         ref(ref_id="x", fragment=True), "ctx", 123, "conv-1", "op",
                         fragment_refs={"ref": ref(fragment=True)})


def test_confirmed_save_requires_exact_server_captured_predecessor_reference():
    original_payload = {"envelope": {"entries": ["confirmed", {"ordinal": 0}]}}
    supplied = ref(ref_id="accepted-save", conversation="older", operation="save-op",
                   payload=original_payload)
    proof = evidence("doc", SourceKind.CONFIRMED_SAVE, source=supplied,
                     conversation="older", operation="save-op", predecessor="save-op")
    item = snapshot(document_evidence={"doc": (proof,)}, confirmed_predecessor_saves=(supplied,))
    original_payload["envelope"]["entries"].append("caller mutation")
    restored = RoutingContextSnapshot.from_dict(json.loads(json.dumps(item.to_dict())))
    assert restored.document_evidence["doc"][0].source.to_dict()["payload"] == {
        "envelope": {"entries": ["confirmed", {"ordinal": 0}]},
    }

    for impostor in (
        ref(ref_id="different-ref", conversation="older", operation="save-op",
            payload={"envelope": {"entries": ["confirmed", {"ordinal": 0}]}}),
        ref(ref_id="accepted-save", conversation="older", operation="save-op",
            payload={"envelope": {"entries": ["confirmed", {"ordinal": 0}]}},
            ),
    ):
        if impostor.ref == supplied.ref:
            # Same opaque path with a different digest fails exact predecessor binding.
            impostor = BoundRef(impostor.ref, "b" * 64, impostor.context_id, impostor.actor_id,
                                impostor.conversation_id, impostor.operation_id, payload=impostor.payload)
        mismatched = evidence("doc", SourceKind.CONFIRMED_SAVE, source=impostor,
                              conversation="older", operation="save-op", predecessor="save-op")
        with pytest.raises(ValueError, match="not an accepted predecessor"):
            snapshot(document_evidence={"doc": (mismatched,)}, confirmed_predecessor_saves=(supplied,))
