"""Private presentation manifest validation and rendering contracts."""

from __future__ import annotations

import copy
import hashlib
from datetime import timedelta

import pytest

pytestmark = pytest.mark.offline


def _evidence(*, doc_id="doc-a", order_index=1, quote="Verified source text.",
              context_id="ctx-1", operation_id="op-1", conversation_id="conv-1", call_id="query-1",
              ordinal=0, reference=None):
    return {
        "context_id": context_id,
        "conversation_id": conversation_id,
        "operation_id": operation_id,
        "checkpoint_conversation_id": "checkpoint-1",
        "call_id": call_id,
        "ordinal": ordinal,
        "reference": reference or {"version": 1, "ref": "exchange-1"},
        "doc_id": doc_id,
        "order_index": order_index,
        "quote": quote,
    }


def _scope():
    return {
        "context_id": "ctx-1",
        "conversation_id": "conv-1",
        "producing_operation_id": "op-1",
        "presentation_id": "pres-1",
        "answer_turn_id": "turn-1",
    }


def _proof_for_row(row, *, result_metadata=None, source_filters=None, name="query", arguments=None,
                   native_call_id="from-row"):
    """Create a self-consistent inline exchange DTO for helper-level schema tests."""
    from services.connectors.telegram.runtime.presentation import _ledger_json_bytes

    operation_id = row["operation_id"]
    call_id = row["call_id"] if native_call_id == "from-row" else native_call_id
    call = {"id": call_id, "name": name,
            "arguments": arguments or row.get("search_arguments") or {"query": "unit proof"},
            "ordinal": row["ordinal"],
            "round": 0}
    item = {"doc_id": row["doc_id"]}
    if "order_index" in row:
        item["order_index"] = row["order_index"]
    if "quote" in row:
        item["quote"] = row["quote"]
    result = {"segments": [item], "metadata": result_metadata or {
        "truncated": False, "total_matches": 1, "returned_segments": 1,
    }}
    if row.get("error") is not None:
        result["error"] = row["error"]
    envelope = {"context_id": row["context_id"], "conversation_id": row["conversation_id"],
                "checkpoint_conversation_id": row["checkpoint_conversation_id"],
                "operation_id": operation_id, "call_id": row["call_id"],
                "ordinal": row["ordinal"], "provenance": "model", "call": call,
                "result": result}
    kind = f"search-history-{row['ordinal']:08d}"
    encoded = _ledger_json_bytes(envelope)
    digest = hashlib.sha256(encoded).hexdigest()
    reference = {"version": 1, "context_id": row["context_id"],
                 "conversation_id": row["conversation_id"], "operation_id": operation_id,
                 "checkpoint_conversation_id": row["checkpoint_conversation_id"],
                 "call_id": row["call_id"], "ordinal": row["ordinal"],
                 "payload_kind": kind,
                 "payload": {"_inline": envelope, "sha256": digest,
                             "byte_length": len(encoded)}}
    return {"reference": reference, "exchange": envelope,
            "source_filters": source_filters or {}}


def _proofs(rows, **kwargs):
    return [_proof_for_row(row, **kwargs) for row in rows]


def _reseal_inline(proof):
    from services.connectors.telegram.runtime.presentation import _ledger_json_bytes

    encoded = _ledger_json_bytes(proof["exchange"])
    payload = proof["reference"]["payload"]
    payload["_inline"] = proof["exchange"]
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    payload["byte_length"] = len(encoded)
    return proof


def _real_recent_list_exchange(db, fixture, monkeypatch, *, offset, ordinal):
    from core.integrations.firestore import transcripts as transcript_repository
    from core.integrations.firestore.client import COLLECTION_NAME
    from core.operations.read import list_recent_transcripts

    db.collection(COLLECTION_NAME).document("doc-recent").set({
        "source_file": "", "status": "synced", "dialog_date": "2026-04-02",
        "parent_id": None,
    })
    monkeypatch.setattr(transcript_repository, "get_client", lambda: db)
    result = list_recent_transcripts(limit=3, offset=offset)
    call_id = f"recent-{ordinal}"
    call = {"id": call_id, "name": "list_recent_transcripts", "ordinal": ordinal,
            "arguments": {"limit": 3, "offset": offset}}
    reference = fixture["ledger"].save_search_exchange(
        fixture["conversation_id"], checkpoint_conversation_id=fixture["checkpoint_id"],
        call=call, canonical_call_id=call_id, result=result,
    )
    exchange = fixture["ledger"].load_search_exchange(
        reference, conversation_id=fixture["conversation_id"],
        operation_id=fixture["accepted"].operation_id,
        checkpoint_conversation_id=fixture["checkpoint_id"], call_id=call_id, ordinal=ordinal,
    )
    proof = {"reference": reference, "exchange": exchange,
             "source_filters": {"folder_path": None, "date_from": None, "date_to": None}}
    return result, proof


def test_builder_renders_native_blocks_with_server_positions_and_hash():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    blocks = [
        {"type": "paragraph", "text": "Verified summary.", "source_refs": [
            {"doc_id": "doc-a", "order_index": 1},
        ]},
        {"type": "document_list", "list_id": "visible-1", "entries": [
            {"doc_id": "doc-b", "label": "Beta", "source_refs": [
                {"doc_id": "doc-b", "order_index": 2},
            ]},
            {"doc_id": "doc-a", "label": "Alpha", "source_refs": [
                {"doc_id": "doc-a", "order_index": 1},
            ]},
        ]},
        {"type": "quote", "doc_id": "doc-b", "text": "Verified source text.",
         "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
    ]
    manifest = build_presentation_manifest(
        scope=_scope(), blocks=blocks,
        verified_exchanges=_proofs([_evidence(), _evidence(doc_id="doc-b", order_index=2)]),
    )
    result = manifest.to_dict()

    assert result["schema_version"] == 1
    assert result["body"] == "Verified summary.\n\n1. Beta\n2. Alpha\n\n> Verified source text."
    assert result["body_sha256"] == hashlib.sha256(result["body"].encode()).hexdigest()
    assert [entry["position"] for entry in result["blocks"][1]["entries"]] == [1, 2]
    assert result["blocks"][2]["text"] == "Verified source text."


def test_lists_are_separate_and_reject_duplicate_ids_and_bad_positions():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    def build(blocks):
        return build_presentation_manifest(
            scope=_scope(), blocks=blocks,
            verified_exchanges=_proofs([_evidence(), _evidence(doc_id="doc-b", order_index=2)]),
        )

    first = {"type": "document_list", "list_id": "first", "entries": [
        {"doc_id": "doc-a", "label": "A", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
    ]}
    second = {"type": "document_list", "list_id": "second", "entries": [
        {"doc_id": "doc-a", "label": "A again", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
    ]}
    assert [block["list_id"] for block in build([first, second]).to_dict()["blocks"]] == [
        "first", "second",
    ]
    with pytest.raises((TypeError, ValueError)):
        build([first, {**copy.deepcopy(second), "list_id": "first"}])
    duplicate = copy.deepcopy(first)
    duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0]))
    with pytest.raises((TypeError, ValueError)):
        build([duplicate])
    gap = copy.deepcopy(first)
    gap["entries"][0]["position"] = 2
    with pytest.raises((TypeError, ValueError)):
        build([gap])
    empty = copy.deepcopy(first)
    empty["entries"] = []
    with pytest.raises((TypeError, ValueError)):
        build([empty])
    unknown = copy.deepcopy(first)
    unknown["entries"][0]["doc_id"] = "not-in-proof"
    with pytest.raises((TypeError, ValueError)):
        build([unknown])


def test_foreign_scope_ambiguous_source_and_inexact_quote_fail_closed():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    block = {"type": "paragraph", "text": "claim", "source_refs": [
        {"doc_id": "doc-a", "order_index": 1},
    ]}
    args = {"scope": _scope(), "blocks": [block], "verified_exchanges": _proofs([_evidence()])}
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(**{**args, "verified_exchanges": _proofs([
            _evidence(conversation_id="foreign-conv"),
        ])})
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(**{**args, "verified_exchanges": _proofs([
            _evidence(context_id="foreign-context"),
        ])})
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(**{**args, "verified_exchanges": _proofs([
            _evidence(), _evidence(call_id="query-2", reference={"version": 1, "ref": "other"}),
        ])})
    proof_rows = _proofs([
        _evidence(), _evidence(call_id="query-2", reference={"version": 1, "ref": "other"}),
    ])
    exact = build_presentation_manifest(
        scope=_scope(),
        blocks=[{"type": "quote", "doc_id": "doc-a", "text": "Verified source text.",
                 "source_refs": [{"doc_id": "doc-a", "order_index": 1,
                                  "operation_id": "op-1", "checkpoint_conversation_id": "checkpoint-1",
                                  "call_id": "query-1", "ordinal": 0,
                                  "reference": proof_rows[0]["reference"]}]}],
        verified_exchanges=proof_rows,
    )
    assert exact.to_dict()["blocks"][0]["evidence_refs"][0]["source_id"]
    quote = {"type": "quote", "doc_id": "doc-a", "text": "paraphrase",
             "source_refs": [{"doc_id": "doc-a", "order_index": 1}]}
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(**{**args, "blocks": [quote]})


def test_manifest_is_copy_isolated_revalidated_and_rich_safe():
    from services.connectors.telegram.runtime.presentation import (
        build_presentation_manifest, validate_presentation_manifest,
    )

    original_text = "<b>literal</b> *stars*\n1. injected\n![x](https://invalid.test)"
    manifest = build_presentation_manifest(
        scope=_scope(),
        blocks=[{"type": "paragraph", "text": original_text,
                 "source_refs": [{"doc_id": "doc-a", "order_index": 1}]}],
        verified_exchanges=_proofs([_evidence()]),
    )
    copied = manifest.to_dict()
    copied["blocks"][0]["text"] = "mutated"
    assert manifest.to_dict()["blocks"][0]["text"] == original_text
    validated = validate_presentation_manifest(
        manifest.to_dict(), expected_scope=_scope(), verified_exchanges=_proofs([_evidence()]),
    )
    assert validated.to_dict() == manifest.to_dict()
    assert "<b>literal</b>" not in manifest.to_dict()["body"]
    assert "1. injected" not in manifest.to_dict()["body"]
    assert "![x](" not in manifest.to_dict()["body"]
    corrupted = manifest.to_dict()
    corrupted["body"] += "tampered"
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(corrupted, expected_scope=_scope())


def test_bool_schema_version_and_missing_scope_are_rejected():
    from services.connectors.telegram.runtime.presentation import validate_presentation_manifest

    for row in (
        {"schema_version": True},
        {"schema_version": 1, "context_id": "ctx-1"},
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_presentation_manifest(row)


def test_real_ledger_inline_and_multipart_references_and_native_ranges(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    db = firestore_fake.client_class()
    fixture = _source_search_fixture(db)
    inline = fixture["verified_exchange"]
    scope = {
        "context_id": fixture["accepted"].context_id,
        "conversation_id": fixture["conversation_id"],
        "producing_operation_id": fixture["accepted"].operation_id,
        "presentation_id": "inline-pres", "answer_turn_id": "inline-turn",
    }
    inline_proof = {"reference": fixture["reference"], "exchange": inline,
                    "source_filters": {"folder_path": None, "date_from": None, "date_to": None}}
    inline_manifest = build_presentation_manifest(
        scope=scope, blocks=[{"type": "quote", "doc_id": "doc-a",
                             "text": "Alpha source.", "source_refs": [
                                 {"doc_id": "doc-a", "order_index": 1},
                             ]}], verified_exchanges=[inline_proof],
    ).to_dict()
    assert inline_manifest["sources"][0]["reference"]["call_id"] == fixture["call"]["id"]
    assert "_inline" not in inline_manifest["sources"][0]["reference"]["payload"]
    assert inline_manifest["sources"][0]["reference"]["payload"]["sha256"] == \
        fixture["reference"]["payload"]["sha256"]
    assert "quote" not in inline_manifest["sources"][0]

    long_quote = "Начало. " + ("длинный проверенный текст 🧪 " * 800)
    call = {"id": "query-2", "name": "query", "ordinal": 1, "round": 1,
            "arguments": {"query": "bounded", "folder_path": ["Sales"],
                           "date_from": "2026-01-01", "date_to": "2026-12-31"}}
    large_result = {"segments": [{"doc_id": "doc-range", "segments": [{
        "range": [4, 9], "dialog": [{"speaker": "A", "text": long_quote,
                                     "order_index": 7}],
    }]}], "metadata": {"truncated": True, "total_matches": 91,
                         "returned_segments": 1}}
    reference = fixture["ledger"].save_search_exchange(
        fixture["conversation_id"], checkpoint_conversation_id=fixture["checkpoint_id"],
        call=call, canonical_call_id=call["id"], result=large_result,
    )
    assert reference["payload"].get("_input_ref")
    corrupted_ref = copy.deepcopy(reference)
    corrupted_ref["payload"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        fixture["ledger"].load_search_exchange(
            corrupted_ref, conversation_id=fixture["conversation_id"],
            operation_id=scope["producing_operation_id"],
            checkpoint_conversation_id=fixture["checkpoint_id"], call_id="query-2", ordinal=1,
        )
    verified = fixture["ledger"].load_search_exchange(
        reference, conversation_id=fixture["conversation_id"],
        operation_id=scope["producing_operation_id"],
        checkpoint_conversation_id=fixture["checkpoint_id"], call_id="query-2", ordinal=1,
    )
    row = verified["result"]["segments"][0]
    evidence = {
        "context_id": scope["context_id"], "conversation_id": scope["conversation_id"],
        "operation_id": scope["producing_operation_id"],
        "checkpoint_conversation_id": fixture["checkpoint_id"], "call_id": "query-2",
        "ordinal": 1, "reference": reference, "doc_id": row["doc_id"],
        "segments": row["segments"], "search_arguments": call["arguments"],
    }
    manifest = build_presentation_manifest(
        scope={**scope, "presentation_id": "range-pres", "answer_turn_id": "range-turn"},
        blocks=[{"type": "quote", "doc_id": "doc-range", "text": long_quote,
                 "source_refs": [{"doc_id": "doc-range", "order_index": 7,
                                  "range_start": 4, "range_end": 9}]}],
        verified_exchanges=[{"reference": reference, "exchange": verified,
                             "source_filters": {"folder_path": ["Sales"],
                                                "date_from": "2026-01-01",
                                                "date_to": "2026-12-31"}}],
    ).to_dict()
    assert manifest["sources"][0]["completeness"] == verified["result"]["metadata"]
    assert "quote" not in manifest["sources"][0]
    assert manifest["blocks"][0]["evidence_refs"][0]["range_start"] == 4
    assert manifest["body"] == "> " + long_quote
    assert "длинный проверенный текст 🧪" in manifest["body"]
    from services.connectors.telegram.runtime.presentation import validate_presentation_manifest
    assert validate_presentation_manifest(
        manifest, verified_exchanges=[{"reference": reference, "exchange": verified,
                                      "source_filters": {"folder_path": ["Sales"],
                                                         "date_from": "2026-01-01",
                                                         "date_to": "2026-12-31"}}],
    ).to_dict() == manifest
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope={**scope, "presentation_id": "wrong-range", "answer_turn_id": "wrong-range"},
            blocks=[{"type": "quote", "doc_id": "doc-range", "text": long_quote,
                     "source_refs": [{"doc_id": "doc-range", "order_index": 7,
                                      "range_start": 5, "range_end": 9}]}],
            verified_exchanges=[{"reference": reference, "exchange": verified,
                                 "source_filters": {"folder_path": ["Sales"],
                                                    "date_from": "2026-01-01",
                                                    "date_to": "2026-12-31"}}],
        )


def test_widened_bounds_error_results_and_contradictory_source_fields_reject():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    evidence = _evidence()
    evidence["search_arguments"] = {"folder_path": ["Other"], "date_from": "2025-01-01"}
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "x",
                                    "source_refs": [{"doc_id": "doc-a", "order_index": 1}]}],
            verified_exchanges=[_proof_for_row(
                evidence, source_filters={"folder_path": ["Sales"], "date_from": "2026-01-01"},
            )],
        )
    error = _evidence()
    error["error"] = {"code": "read_failed"}
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "x",
                                    "source_refs": [{"doc_id": "doc-a", "order_index": 1}]}],
            verified_exchanges=_proofs([error]),
        )
    contradictory = {"doc_id": "doc-b", "order_index": 1}
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "quote", "doc_id": "doc-a", "text": "Verified source text.",
                                    "source_refs": [contradictory]}],
            verified_exchanges=_proofs([_evidence()]),
        )


@pytest.mark.parametrize("corruption", [
    "operation", "checkpoint", "call", "ordinal", "bool_ordinal", "bool_version",
    "digest", "partial", "extra",
])
def test_inconsistent_exchange_references_fail_closed(corruption):
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    proof = _proof_for_row(_evidence())
    reference = proof["reference"]
    if corruption == "operation":
        reference["operation_id"] = "foreign-op"
    elif corruption == "checkpoint":
        reference["checkpoint_conversation_id"] = "foreign-checkpoint"
    elif corruption == "call":
        reference["call_id"] = "foreign-call"
    elif corruption == "ordinal":
        reference["ordinal"] = 9
    elif corruption == "bool_ordinal":
        reference["ordinal"] = True
    elif corruption == "bool_version":
        reference["version"] = True
    elif corruption == "digest":
        reference["payload"]["sha256"] = "0" * 64
    elif corruption == "partial":
        reference.pop("checkpoint_conversation_id")
    elif corruption == "extra":
        reference["unrecognized"] = "value"
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "summary", "source_refs": []}],
            verified_exchanges=[proof],
        )


@pytest.mark.parametrize("selector_change", [
    "ordinal_bool", "ordinal_float", "reference_version_bool", "reference_version_float",
    "reference_ordinal_float", "reference_length_float", "reference_part_size_float",
])
def test_model_selector_identity_aliases_fail_closed(selector_change):
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    row = _evidence()
    proof = _proof_for_row(row)
    selector = {"doc_id": "doc-a", "order_index": 1}
    reference = copy.deepcopy(proof["reference"])
    if selector_change == "ordinal_bool":
        selector["ordinal"] = False
    elif selector_change == "ordinal_float":
        selector["ordinal"] = 0.0
    elif selector_change == "reference_version_bool":
        reference["version"] = True
        selector["reference"] = reference
    elif selector_change == "reference_version_float":
        reference["version"] = 1.0
        selector["reference"] = reference
    elif selector_change == "reference_ordinal_float":
        reference["ordinal"] = 0.0
        selector["reference"] = reference
    elif selector_change == "reference_length_float":
        reference["payload"]["byte_length"] = float(reference["payload"]["byte_length"])
        selector["reference"] = reference
    else:
        payload = reference["payload"]
        kind = reference["payload_kind"]
        data_operation = "telegram-ledger-" + hashlib.sha256(
            f"{reference['context_id']}\0{reference['operation_id']}\0{kind}\0"
            f"{payload['sha256']}".encode()
        ).hexdigest()
        proof["reference"]["payload"] = {
            "_input_ref": f"dialog_operations/{data_operation}/input_parts/manifest",
            "sha256": payload["sha256"], "byte_length": payload["byte_length"],
            "max_part_bytes": 64,
        }
        reference = copy.deepcopy(proof["reference"])
        reference["payload"]["max_part_bytes"] = 64.0
        selector["reference"] = reference
    with pytest.raises((TypeError, ValueError)):
        build_presentation_manifest(
            scope=_scope(),
            blocks=[{"type": "paragraph", "text": "summary", "source_refs": [selector]}],
            verified_exchanges=[proof],
        )


@pytest.mark.parametrize("native_id", [None, ""])
def test_missing_native_call_id_uses_existing_ledger_fallback(native_id):
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    row = _evidence(call_id="telegram-call-op-1-0")
    proof = _proof_for_row(row, native_call_id=native_id)
    manifest = build_presentation_manifest(
        scope=_scope(), blocks=[{"type": "paragraph", "text": "summary", "source_refs": [
            {"doc_id": "doc-a", "order_index": 1},
        ]}], verified_exchanges=[proof],
    ).to_dict()
    assert manifest["sources"][0]["call_id"] == row["call_id"]


def test_nonstring_native_call_id_rejects():
    from services.connectors.telegram.runtime.presentation import (
        build_presentation_manifest,
    )

    proof = _proof_for_row(_evidence(call_id="telegram-call-op-1-0"), native_call_id=17)
    _reseal_inline(proof)
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "summary", "source_refs": []}],
            verified_exchanges=[proof],
        )

def test_structural_validator_rejects_blank_paragraph_and_reversed_ranges():
    from services.connectors.telegram.runtime.presentation import (
        build_presentation_manifest, validate_presentation_manifest,
    )

    valid = build_presentation_manifest(
        scope=_scope(), blocks=[{"type": "paragraph", "text": "summary", "source_refs": []}],
        verified_exchanges=_proofs([_evidence()]),
    ).to_dict()
    blank = copy.deepcopy(valid)
    blank["blocks"][0]["text"] = "  "
    blank["body"] = "  "
    blank["body_sha256"] = hashlib.sha256(blank["body"].encode()).hexdigest()
    with pytest.raises(ValueError):
        validate_presentation_manifest(blank)
    reversed_range = copy.deepcopy(valid)
    reversed_range["blocks"][0]["evidence_refs"] = [{
        "source_id": valid["sources"][0]["source_id"], "doc_id": "doc-a",
        "order_index": 1, "range_start": 9, "range_end": 4,
    }]
    with pytest.raises(ValueError):
        validate_presentation_manifest(reversed_range)


def test_unsupported_tool_and_error_payload_cannot_authorize_sources():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    for proof in (_proof_for_row(_evidence(), name="list_folders"),
                  _proof_for_row(_evidence(), name="query")):
        if proof["exchange"]["call"]["name"] == "query":
            proof["exchange"]["result"]["is_error"] = True
            # Make the inline ref internally self-consistent; the error result is still rejected.
            _reseal_inline(proof)
        with pytest.raises(ValueError):
            build_presentation_manifest(
                scope=_scope(), blocks=[{"type": "document_list", "list_id": "x", "entries": [
                    {"doc_id": "doc-a", "label": "A", "source_refs": [{"doc_id": "doc-a"}]},
                ]}], verified_exchanges=[proof],
            )


def test_empty_unknown_and_bool_position_blocks_reject():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    common = {"scope": _scope(), "verified_exchanges": _proofs([_evidence()])}
    for blocks in ([], [{"type": "unknown", "text": "x"}], [
        {"type": "document_list", "list_id": "one", "entries": [
            {"position": True, "doc_id": "doc-a", "label": "A",
             "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
        ]},
    ]):
        with pytest.raises((TypeError, ValueError)):
            build_presentation_manifest(**common, blocks=blocks)


def _verified_exchange_item(fixture, *, result=None, ordinal=0, name="query", arguments=None):
    exchange = copy.deepcopy(fixture["verified_exchange"])
    reference = copy.deepcopy(fixture["reference"])
    if result is not None:
        exchange["result"] = result
    if name != "query" or arguments is not None:
        exchange["call"]["name"] = name
        exchange["call"]["arguments"] = arguments or {"query": "same session"}
    return {"reference": reference, "exchange": exchange,
            "source_filters": {"folder_path": None, "date_from": None, "date_to": None}}


def test_prior_operation_proof_can_be_reused_later_in_same_conversation(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.operation_queue import complete_head, claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    db = firestore_fake.client_class()
    fixture = _source_search_fixture(db)
    prior_operation = fixture["accepted"].operation_id
    conversation_id = fixture["conversation_id"]
    context_ref = db.collection("telegram_sessions").document(fixture["accepted"].context_id)
    assert complete_head(
        db, context_ref, operation_id=prior_operation, owner=fixture["lease"].owner,
        fence=fixture["lease"].fence, now=fixture["now"] + timedelta(seconds=1),
    )
    accepted = fixture["repository"].accept_command(
        88102, chat_id=42017, thread_id=73, user_id=17,
        command=parse_command("Use the previously shown search results"),
        now=fixture["now"] + timedelta(seconds=2),
    )
    lease = claim_head(db, context_ref, owner="presentation-later-worker",
                       now=fixture["now"] + timedelta(seconds=2))
    assert lease is not None
    assert lease.operation_id == accepted.operation_id and accepted.operation_id != prior_operation
    assert fixture["repository"].get_context(accepted.context_id)["active_conversation_id"] == conversation_id
    old = fixture["verified_exchange"]
    row = next(item for item in old["result"]["segments"] if item["doc_id"] == "doc-a")
    scope = {"context_id": accepted.context_id, "conversation_id": conversation_id,
             "producing_operation_id": accepted.operation_id, "presentation_id": "later-pres",
             "answer_turn_id": "later-turn"}
    item = _verified_exchange_item(fixture)
    manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "quote", "doc_id": "doc-a", "text": row["quote"],
                 "source_refs": [{"doc_id": "doc-a", "order_index": row["order_index"]}]}],
        verified_exchanges=[item],
    )
    proof = manifest.to_dict()["sources"][0]
    assert proof["operation_id"] == prior_operation
    assert proof["operation_id"] != scope["producing_operation_id"]


def test_verified_exchange_roundtrip_reauthenticates_quote_refs_and_metadata(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from services.connectors.telegram.runtime.presentation import (
        build_presentation_manifest, validate_presentation_manifest,
    )
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    db = firestore_fake.client_class()
    fixture = _source_search_fixture(db)
    row = next(item for item in fixture["verified_exchange"]["result"]["segments"]
               if item["doc_id"] == "doc-b")
    scope = {"context_id": fixture["accepted"].context_id,
             "conversation_id": fixture["conversation_id"],
             "producing_operation_id": fixture["accepted"].operation_id,
             "presentation_id": "proof-pres", "answer_turn_id": "proof-turn"}
    proof = _verified_exchange_item(fixture)
    blocks = [{"type": "quote", "doc_id": "doc-b", "text": row["quote"],
               "source_refs": [{"doc_id": "doc-b", "order_index": row["order_index"]}]}]
    manifest = build_presentation_manifest(scope=scope, blocks=blocks,
                                           verified_exchanges=[proof]).to_dict()
    assert validate_presentation_manifest(
        manifest, expected_scope=scope, verified_exchanges=[proof],
    ).to_dict() == manifest

    forged_quote = copy.deepcopy(manifest)
    forged_quote["blocks"][0]["text"] = "forged but hashed"
    forged_quote["body"] = "> forged but hashed"
    forged_quote["body_sha256"] = hashlib.sha256(forged_quote["body"].encode()).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(forged_quote, verified_exchanges=[proof])
    # Structural validation checks only shape and bytes; it is not source authority.
    assert validate_presentation_manifest(forged_quote).to_dict() == forged_quote
    forged_ref = copy.deepcopy(manifest)
    forged_ref["blocks"][0]["evidence_refs"][0]["ordinal"] = 3
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(forged_ref, verified_exchanges=[proof])
    forged_filters = copy.deepcopy(manifest)
    forged_filters["sources"][0]["source_filters"] = {"folder_path": ["widened"]}
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(forged_filters, verified_exchanges=[proof])
    forged_completeness = copy.deepcopy(manifest)
    forged_completeness["sources"][0]["completeness"]["total_matches"] = 99
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(forged_completeness, verified_exchanges=[proof])
    wrong_hash = copy.deepcopy(manifest)
    wrong_hash["body_sha256"] = "0" * 64
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(wrong_hash, verified_exchanges=[proof])


def test_no_result_paragraph_and_recent_transcript_list_need_no_utterance_index(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    db = firestore_fake.client_class()
    fixture = _source_search_fixture(db)
    ledger = fixture["ledger"]
    empty_call = {"id": "query-empty", "name": "query", "ordinal": 1,
                  "arguments": {"query": "no results"}}
    empty_ref = ledger.save_search_exchange(
        fixture["conversation_id"], checkpoint_conversation_id=fixture["checkpoint_id"],
        call=empty_call, canonical_call_id=empty_call["id"],
        result={"segments": [], "metadata": {"truncated": False, "total_matches": 0,
                                                   "returned_segments": 0}},
    )
    empty_exchange = ledger.load_search_exchange(
        empty_ref, conversation_id=fixture["conversation_id"],
        operation_id=fixture["accepted"].operation_id,
        checkpoint_conversation_id=fixture["checkpoint_id"], call_id="query-empty", ordinal=1,
    )
    scope = {"context_id": fixture["accepted"].context_id,
             "conversation_id": fixture["conversation_id"],
             "producing_operation_id": fixture["accepted"].operation_id,
             "presentation_id": "empty-pres", "answer_turn_id": "empty-turn"}
    empty_manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "paragraph", "text": "No matching discussions were found.",
                 "source_refs": []}],
        verified_exchanges=[{"reference": empty_ref, "exchange": empty_exchange,
                             "source_filters": {"folder_path": None, "date_from": None,
                                                "date_to": None}}],
    ).to_dict()
    assert empty_manifest["body"] == "No matching discussions were found."
    assert len(empty_manifest["sources"]) == 1
    assert empty_manifest["sources"][0]["completeness"] == empty_exchange["result"]["metadata"]

    list_result, list_proof = _real_recent_list_exchange(
        db, fixture, monkeypatch, offset=0, ordinal=2,
    )
    assert set(list_result) == {"total", "returned", "offset", "limit", "has_more", "transcripts"}
    assert list_result["transcripts"][0]["title"] == ""
    list_manifest = build_presentation_manifest(
        scope={**scope, "presentation_id": "recent-pres", "answer_turn_id": "recent-turn"},
        blocks=[{"type": "document_list", "list_id": "recent-list", "entries": [
            {"doc_id": "doc-recent", "label": "Recent title", "source_refs": [
                {"doc_id": "doc-recent"},
            ]},
        ]}],
        verified_exchanges=[list_proof],
    ).to_dict()
    assert list_manifest["blocks"][0]["entries"][0]["position"] == 1
    assert list_manifest["sources"][0]["completeness"] == {
        "total": 1, "returned": 1, "offset": 0, "limit": 3, "has_more": False,
    }

    from services.connectors.telegram.runtime.presentation import validate_presentation_manifest

    beyond_result, beyond_proof = _real_recent_list_exchange(
        db, fixture, monkeypatch, offset=99, ordinal=3,
    )
    assert beyond_result == {
        "total": 1, "returned": 0, "offset": 99, "limit": 3,
        "has_more": False, "transcripts": [],
    }
    beyond_manifest = build_presentation_manifest(
        scope={**scope, "presentation_id": "beyond-pres", "answer_turn_id": "beyond-turn"},
        blocks=[{"type": "paragraph", "text": "No transcripts on this page.", "source_refs": []}],
        verified_exchanges=[beyond_proof],
    ).to_dict()
    assert beyond_manifest["body"] == "No transcripts on this page."
    assert beyond_manifest["sources"][0]["completeness"] == {
        "total": 1, "returned": 0, "offset": 99, "limit": 3, "has_more": False,
    }
    assert validate_presentation_manifest(
        beyond_manifest, verified_exchanges=[beyond_proof],
    ).to_dict() == beyond_manifest


@pytest.mark.parametrize(("field", "alias"), [("offset", False), ("limit", 3.0)])
def test_full_proof_rejects_typed_argument_aliases_in_real_recent_list(
    firestore_fake, monkeypatch, field, alias,
):
    from services.connectors.telegram.runtime.presentation import (
        build_presentation_manifest, validate_presentation_manifest,
    )
    from tests.services.connectors.telegram.test_search_presentation import _source_search_fixture

    db = firestore_fake.client_class()
    fixture = _source_search_fixture(db)
    _result, proof = _real_recent_list_exchange(db, fixture, monkeypatch, offset=0, ordinal=2)
    scope = {"context_id": fixture["accepted"].context_id,
             "conversation_id": fixture["conversation_id"],
             "producing_operation_id": fixture["accepted"].operation_id,
             "presentation_id": f"strict-{field}", "answer_turn_id": f"strict-{field}-turn"}
    manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "paragraph", "text": "The page was checked.", "source_refs": []}],
        verified_exchanges=[proof],
    ).to_dict()
    assert validate_presentation_manifest(
        manifest, expected_scope=scope, verified_exchanges=[proof],
    ).to_dict() == manifest

    typed_alias = copy.deepcopy(manifest)
    typed_alias["sources"][0]["arguments"][field] = alias
    with pytest.raises((TypeError, ValueError)):
        validate_presentation_manifest(
            typed_alias, expected_scope=scope, verified_exchanges=[proof],
        )


def test_json_values_and_literal_rich_escaping_are_strict_and_deterministic():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    manifest = build_presentation_manifest(
        scope=_scope(),
        blocks=[
            {"type": "paragraph", "text": "Literal > and <, & \\ and <b>HTML</b>",
             "source_refs": []},
            {"type": "document_list", "list_id": "lines", "entries": [
                {"doc_id": "doc-a", "label": "Alpha\n1. literal\n> quote\n![media](x)",
                 "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
            ]},
        ], verified_exchanges=_proofs([_evidence()]),
    )
    first, second = manifest.to_dict(), manifest.to_dict()
    assert first == second
    assert first["body_sha256"] == hashlib.sha256(first["body"].encode("utf-8")).hexdigest()
    assert "&gt;" not in first["body"] and "&lt;" not in first["body"]
    assert "1\\. literal" in first["body"]
    assert "   > quote" not in first["body"]
    assert "Literal \\> and \\<, & \\\\ and \\<b\\>HTML\\</b\\>" in first["body"]

    with pytest.raises(TypeError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "x", "source_refs": []}],
            verified_exchanges=[_proof_for_row(_evidence(), source_filters={1: "coerced key"})],
        )
    malformed = _proof_for_row(_evidence())
    malformed["exchange"]["result"]["metadata"]["returned_segments"] = True
    _reseal_inline(malformed)
    with pytest.raises(ValueError):
        build_presentation_manifest(
            scope=_scope(), blocks=[{"type": "paragraph", "text": "text", "source_refs": []}],
            verified_exchanges=[malformed],
        )


def test_multiline_quote_keeps_literal_html_ampersand_and_greater_than_visible():
    from services.connectors.telegram.runtime.presentation import build_presentation_manifest

    quote = "Line one > literal\n<b>Line two & more</b>"
    proof = _proof_for_row(_evidence(quote=quote))
    manifest = build_presentation_manifest(
        scope=_scope(), blocks=[{"type": "quote", "doc_id": "doc-a", "text": quote,
                                "source_refs": [{"doc_id": "doc-a", "order_index": 1}]}],
        verified_exchanges=[proof],
    ).to_dict()
    assert manifest["blocks"][0]["text"] == quote
    assert manifest["body"] == "> Line one \\> literal\n> \\<b\\>Line two & more\\</b\\>"
