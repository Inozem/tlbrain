"""Telegram search answer validation and rendering.

Only values from successful, scoped ``FirestoreAgentLedger.load_search_exchange``
calls should be passed to the builder. A loaded envelope remains data until its
reference, call, supported read operation, result shape, accepted bounds and
same-context scope are checked here. Structural manifest validation alone
rechecks JSON/schema/render/hash consistency; it does not authenticate sources
or grant selection/delivery authority. Supplying ``verified_exchanges`` performs
that separate provenance check against the original loaded envelopes.

Quote blocks accept exact text and a source reference; explanatory prose belongs
in a separate paragraph block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


_IDENTITY = ("context_id", "conversation_id", "producing_operation_id",
             "presentation_id", "answer_turn_id")
_MANIFEST_FIELDS = frozenset((*_IDENTITY, "schema_version", "blocks", "body",
                              "body_sha256", "sources", "notification_refs",
                              "revision_refs"))
_REFERENCE_FIELDS = frozenset(("version", "context_id", "conversation_id", "operation_id",
                               "checkpoint_conversation_id", "call_id", "ordinal",
                               "payload_kind", "payload"))
_SOURCE_FIELDS = frozenset(("source_id", "operation_id", "checkpoint_conversation_id",
                            "call_id", "ordinal", "reference", "call_name", "arguments",
                            "source_filters", "completeness"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LIST_PREFIX = re.compile(r"^(\s*)\d+([.)])(\s+)")


def _json_copy(value: Any, name: str) -> Any:
    """Copy strict JSON values without coercing tuples, keys, or non-finite floats."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise TypeError
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError
                check(child)
            return
        raise TypeError

    try:
        check(value)
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")))
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{name} must contain strict JSON values") from None


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _scope(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY):
        raise ValueError("presentation scope fields are incomplete or unknown")
    return {key: _nonempty(value[key], key) for key in _IDENTITY}


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON structures without Python's bool/int/float equality aliases."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (left.keys() == right.keys()
                and all(_strict_json_equal(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _ledger_json_bytes(value: Any) -> bytes:
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    return FirestoreAgentLedger._json_bytes(value)


def _minimal_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(reference)
    payload = reference["payload"]
    result["payload"] = {key: payload[key] for key in (
        "_input_ref", "sha256", "byte_length", "max_part_bytes",
    ) if key in payload}
    return result


def _validate_reference(reference: Any, envelope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_FIELDS:
        raise ValueError("search exchange reference is incomplete or has unknown fields")
    expected = {
        "version": 1, "context_id": envelope["context_id"],
        "conversation_id": envelope["conversation_id"],
        "operation_id": envelope["operation_id"],
        "checkpoint_conversation_id": envelope["checkpoint_conversation_id"],
        "call_id": envelope["call_id"], "ordinal": envelope["ordinal"],
        "payload_kind": f"search-history-{envelope['ordinal']:08d}",
    }
    for key, value in expected.items():
        if type(reference.get(key)) is not type(value) or reference.get(key) != value:
            raise ValueError("search exchange reference identity is contradictory")
    payload = reference.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("search exchange payload reference is malformed")
    digest, length = payload.get("sha256"), payload.get("byte_length")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("search exchange digest is invalid")
    if type(length) is not int or length < 0:
        raise ValueError("search exchange byte length is invalid")
    encoded_envelope = _ledger_json_bytes(envelope)
    if len(encoded_envelope) != length or hashlib.sha256(encoded_envelope).hexdigest() != digest:
        raise ValueError("loaded exchange bytes do not match the original ref digest")
    if "_inline" in payload:
        if set(payload) != {"_inline", "sha256", "byte_length"}:
            raise ValueError("inline search payload reference is malformed")
        inline = payload["_inline"]
        if inline != {key: value for key, value in envelope.items()}:
            raise ValueError("inline reference does not contain the verified exchange")
    else:
        if set(payload) != {"_input_ref", "sha256", "byte_length", "max_part_bytes"}:
            raise ValueError("multipart search payload reference is malformed")
        part_bytes = payload.get("max_part_bytes")
        if type(part_bytes) is not int or part_bytes < 1:
            raise ValueError("multipart search part size is invalid")
        data_operation = "telegram-ledger-" + hashlib.sha256(
            f"{envelope['context_id']}\0{envelope['operation_id']}\0"
            f"{expected['payload_kind']}\0{digest}".encode("utf-8")
        ).hexdigest()
        expected_path = f"dialog_operations/{data_operation}/input_parts/manifest"
        if payload.get("_input_ref") != expected_path:
            raise ValueError("multipart search reference is not bound to its digest")
    return _minimal_reference(reference)


def _validated_exchange_input(
    raw: Any, scope: Mapping[str, str],
) -> tuple[Mapping[str, Any], dict[str, Any], str, Mapping[str, Any], dict[str, Any]]:
    if not isinstance(raw, Mapping) or set(raw) != {"reference", "exchange", "source_filters"}:
        raise ValueError("verified exchange input is incomplete or has unknown fields")
    envelope = raw["exchange"]
    required = {"context_id", "conversation_id", "checkpoint_conversation_id", "operation_id",
                "call_id", "ordinal", "provenance", "call", "result"}
    if not isinstance(envelope, Mapping) or set(envelope) != required:
        raise ValueError("loaded search exchange envelope is incomplete or malformed")
    for key in ("context_id", "conversation_id", "operation_id", "checkpoint_conversation_id",
                "call_id"):
        _nonempty(envelope.get(key), key)
    if envelope["context_id"] != scope["context_id"] or envelope["conversation_id"] != scope["conversation_id"]:
        raise ValueError("search evidence belongs to a foreign context or conversation")
    if type(envelope["ordinal"]) is not int or envelope["ordinal"] < 0:
        raise ValueError("search exchange ordinal is invalid")
    if (not isinstance(envelope["provenance"], str)
            or envelope["provenance"] not in ("application", "model")):
        raise ValueError("search exchange provenance is invalid")
    reference = _validate_reference(raw["reference"], envelope)
    call = envelope["call"]
    if not isinstance(call, Mapping) or not {"name", "arguments"}.issubset(call):
        raise ValueError("search call is incomplete")
    if set(call) - {"id", "name", "arguments", "ordinal", "round", "thought_signature"}:
        raise ValueError("search call has unknown fields")
    thought_signature = call.get("thought_signature")
    if ("thought_signature" in call and thought_signature is not None
            and not isinstance(thought_signature, str)):
        raise ValueError("search call thought signature must be a string or null")
    if type(call.get("ordinal")) is not int or call["ordinal"] != envelope["ordinal"]:
        raise ValueError("search call ordinal is contradictory")
    native_id = call.get("id")
    if "id" in call and native_id is not None and not isinstance(native_id, str):
        raise ValueError("native search call ID must be a string or null")
    canonical_id = native_id if isinstance(native_id, str) and native_id else (
        f"telegram-call-{envelope['operation_id']}-{envelope['ordinal']}"
    )
    if canonical_id != envelope["call_id"]:
        raise ValueError("search call ID is contradictory")
    name, arguments = call.get("name"), call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("search call name or arguments are invalid")
    from services.connectors.telegram.runtime.search_agent import (
        _arguments_preserve_search_bounds, _evidence_is_scoped_read,
    )
    if not _evidence_is_scoped_read(name, arguments):
        raise ValueError("unsupported reads cannot establish presentation evidence")
    filters = _json_copy(raw["source_filters"], "source filters")
    if not isinstance(filters, dict):
        raise ValueError("source filters must be an object")
    if not _arguments_preserve_search_bounds(filters, arguments):
        raise ValueError("search read did not preserve accepted bounds")
    return envelope, reference, name, arguments, filters


def _result_rows_and_completeness(
    name: str, result: Any,
) -> tuple[list[Any], dict[str, Any]]:
    if (not isinstance(result, Mapping) or result.get("error") is not None
            or ("is_error" in result and type(result["is_error"]) is not bool)
            or result.get("is_error") is True):
        raise ValueError("error results cannot establish presentation evidence")
    if name == "list_recent_transcripts":
        fields = {"total", "returned", "offset", "limit", "has_more", "transcripts"}
        if set(result) != fields or not isinstance(result["transcripts"], list):
            raise ValueError("recent transcript result pagination is incomplete")
        for key in ("total", "returned", "offset", "limit"):
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError("recent transcript pagination integer is invalid")
        if type(result["has_more"]) is not bool or result["returned"] != len(result["transcripts"]):
            raise ValueError("recent transcript pagination metadata is contradictory")
        if result["returned"] > result["limit"] or result["returned"] > max(
                result["total"] - result["offset"], 0):
            raise ValueError("recent transcript pagination exceeds its total")
        if result["has_more"] != (result["offset"] + result["returned"] < result["total"]):
            raise ValueError("recent transcript has_more flag is contradictory")
        completeness = {key: result[key] for key in ("total", "returned", "offset", "limit", "has_more")}
        return list(result["transcripts"]), completeness

    present_content = [key for key in ("segments", "results", "transcripts") if key in result]
    if len(present_content) != 1 or not isinstance(result[present_content[0]], list):
        raise ValueError("search result has an unsupported or partial content shape")
    from services.connectors.telegram.runtime.search_agent import _confirmed_result_documents

    rows = _confirmed_result_documents(result)
    metadata = result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("search completeness metadata is missing")
    completeness = _json_copy(dict(metadata), "search completeness metadata")
    if (not {"truncated", "total_matches", "returned_segments"}.issubset(completeness)
            or type(completeness["truncated"]) is not bool
            or type(completeness["total_matches"]) is not int or completeness["total_matches"] < 0
            or type(completeness["returned_segments"]) is not int
            or completeness["returned_segments"] < 0):
        raise ValueError("search completeness metadata is incomplete or malformed")
    return list(rows), completeness


def _source_record(
    envelope: Mapping[str, Any], reference: Mapping[str, Any], call_name: str,
    arguments: Mapping[str, Any], filters: Mapping[str, Any], completeness: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = hashlib.sha256(json.dumps({
        "context_id": envelope["context_id"], "conversation_id": envelope["conversation_id"],
        "operation_id": envelope["operation_id"],
        "checkpoint_conversation_id": envelope["checkpoint_conversation_id"],
        "call_id": envelope["call_id"], "ordinal": envelope["ordinal"],
        "digest": reference["payload"]["sha256"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "source_id": source_id, "operation_id": envelope["operation_id"],
        "checkpoint_conversation_id": envelope["checkpoint_conversation_id"],
        "call_id": envelope["call_id"], "ordinal": envelope["ordinal"],
        "reference": reference, "call_name": call_name,
        "arguments": _json_copy(dict(arguments), "search arguments"),
        "source_filters": _json_copy(dict(filters), "source filters"),
        "completeness": _json_copy(dict(completeness), "source completeness"),
    }


def _expand_result_rows(
    name: str, source: Mapping[str, Any], rows: Sequence[Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("search result row is malformed")
        doc_id = _nonempty(row.get("doc_id"), "result doc_id")
        if isinstance(row.get("segments"), list):
            nested_indexes = []
            for segment in row["segments"]:
                if not isinstance(segment, Mapping):
                    raise ValueError("result segment is malformed")
                source_range = segment.get("range")
                dialog = segment.get("dialog")
                if (not isinstance(source_range, list) or len(source_range) != 2
                        or any(type(v) is not int or v < 0 for v in source_range)
                        or source_range[1] < source_range[0] or not isinstance(dialog, list)):
                    raise ValueError("result segment range/dialog is malformed")
                for utterance in dialog:
                    if not isinstance(utterance, Mapping):
                        raise ValueError("result utterance is malformed")
                    index, text = utterance.get("order_index"), utterance.get("text")
                    if (type(index) is not int or index < source_range[0] or index > source_range[1]
                            or not isinstance(text, str)):
                        raise ValueError("result utterance does not match its exact range")
                    nested_indexes.append((index, text))
                    candidates.append({**source, "doc_id": doc_id, "order_index": index,
                                       "range_start": source_range[0], "range_end": source_range[1],
                                       "quote": text, "label": row.get("title") or doc_id})
            if "order_index" in row and (
                    type(row["order_index"]) is not int
                    or not any(index == row["order_index"] for index, _text in nested_indexes)):
                raise ValueError("row and nested utterance order identities conflict")
            if "quote" in row and (not isinstance(row["quote"], str)
                                   or not any(text == row["quote"] for _index, text in nested_indexes)):
                raise ValueError("row and nested quote content conflict")
        elif "order_index" in row:
            index = row["order_index"]
            if type(index) is not int or index < 0:
                raise ValueError("result order_index is invalid")
            candidate = {**source, "doc_id": doc_id, "order_index": index,
                         "label": row.get("title") or doc_id}
            if "quote" in row:
                if not isinstance(row["quote"], str):
                    raise ValueError("result quote must be text")
                candidate["quote"] = row["quote"]
            candidates.append(candidate)
        elif name == "list_recent_transcripts":
            title = row.get("title")
            # The real listing contract permits empty source_file titles. The
            # model must still provide a separate nonempty visible label.
            candidates.append({**source, "doc_id": doc_id,
                               "label": title if isinstance(title, str) and title.strip() else doc_id})
        else:
            raise ValueError("document result lacks verified utterance provenance")
    return candidates


def _exchange_record(raw: Any, scope: Mapping[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    envelope, reference, name, arguments, filters = _validated_exchange_input(raw, scope)
    result = envelope["result"]
    rows, completeness = _result_rows_and_completeness(name, result)
    source = _source_record(
        envelope, reference, name, arguments, filters, completeness,
    )
    source["original_reference"] = _json_copy(dict(raw["reference"]), "search exchange reference")
    return source, _expand_result_rows(name, source, rows)


def _verified_inputs(values: Any, scope: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("verified_exchanges must be an array")
    sources, candidates = [], []
    source_ids: set[str] = set()
    candidate_ids: set[tuple[Any, ...]] = set()
    for raw in values:
        source, rows = _exchange_record(raw, scope)
        if source["source_id"] in source_ids:
            raise ValueError("duplicate verified exchange source")
        source_ids.add(source["source_id"])
        sources.append(source)
        for candidate in rows:
            identity = _candidate_identity(candidate)
            if identity in candidate_ids:
                raise ValueError("verified exchanges contain a duplicate source row")
            candidate_ids.add(identity)
        candidates.extend(rows)
    return sources, candidates


def _escape_text(value: str) -> str:
    """Escape Rich Markdown syntax while keeping literal characters visible."""
    escaped_lines = []
    for line in value.split("\n"):
        line = line.replace("\\", "\\\\")
        for char in "_*[]()~`<>#+-=|{}!":
            line = line.replace(char, "\\" + char)
        match = _LIST_PREFIX.match(line)
        if match is not None:
            start, end = match.span(2)
            line = line[:start] + "\\" + line[start:end] + line[end:]
        escaped_lines.append(line)
    return "\n".join(escaped_lines)


def _render(blocks: Sequence[Mapping[str, Any]]) -> str:
    rendered = []
    for block in blocks:
        if block["type"] == "paragraph":
            rendered.append(_escape_text(block["text"]))
        elif block["type"] == "document_list":
            lines = []
            for position, entry in enumerate(block["entries"], 1):
                label = _escape_text(entry["label"])
                lines.append(f"{position}. " + label.replace("\n", "\n   "))
            rendered.append("\n".join(lines))
        else:
            rendered.append("\n".join("> " + line for line in _escape_text(block["text"]).split("\n")))
    return "\n\n".join(rendered)


def _candidate_identity(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (candidate["source_id"], candidate["doc_id"], candidate.get("order_index"),
            candidate.get("range_start"), candidate.get("range_end"))


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = {"source_id": candidate["source_id"], "doc_id": candidate["doc_id"]}
    for key in ("order_index", "range_start", "range_end"):
        if key in candidate:
            result[key] = candidate[key]
    return result


def _resolve_source(selector: Any, candidates: Sequence[Mapping[str, Any]], *,
                    doc_id: str | None = None) -> dict[str, Any]:
    if not isinstance(selector, Mapping):
        raise TypeError("source selector must be an object")
    allowed = {"source_id", "doc_id", "order_index", "range_start", "range_end",
               "operation_id", "checkpoint_conversation_id", "call_id", "ordinal", "reference"}
    if not {"doc_id"}.issubset(selector) or set(selector) - allowed:
        raise ValueError("source selector is incomplete or has unknown fields")
    selected_doc = _nonempty(selector["doc_id"], "source doc_id")
    if doc_id is not None and selected_doc != doc_id:
        raise ValueError("source selector contradicts block document")
    if "order_index" in selector and (type(selector["order_index"]) is not int
                                      or selector["order_index"] < 0):
        raise ValueError("source order_index is invalid")
    if ("range_start" in selector) != ("range_end" in selector):
        raise ValueError("source selector range is incomplete")
    for field in ("range_start", "range_end"):
        if field in selector and (type(selector[field]) is not int or selector[field] < 0):
            raise ValueError("source selector range is invalid")
    if ("range_start" in selector
            and selector["range_end"] < selector["range_start"]):
        raise ValueError("source selector range is reversed")
    for field in ("source_id", "operation_id", "checkpoint_conversation_id", "call_id"):
        if field in selector:
            _nonempty(selector[field], field)
    if "ordinal" in selector and (type(selector["ordinal"]) is not int or selector["ordinal"] < 0):
        raise ValueError("source selector ordinal is invalid")
    matches = []
    for candidate in candidates:
        if candidate["doc_id"] != selected_doc:
            continue
        if "order_index" in selector and candidate.get("order_index") != selector["order_index"]:
            continue
        if "range_start" in selector and (
                candidate.get("range_start") != selector["range_start"]
                or candidate.get("range_end") != selector["range_end"]):
            continue
        identity_values = {
            "source_id": candidate["source_id"], "operation_id": candidate["operation_id"],
            "checkpoint_conversation_id": candidate["checkpoint_conversation_id"],
            "call_id": candidate["call_id"], "ordinal": candidate["ordinal"],
        }
        if any(key in selector and not _strict_json_equal(selector[key], value)
               for key, value in identity_values.items()):
            continue
        if "reference" in selector:
            supplied_ref = _json_copy(selector["reference"], "source selector reference")
            if (not _strict_json_equal(supplied_ref, candidate["reference"])
                    and not _strict_json_equal(supplied_ref, candidate.get("original_reference"))):
                continue
        matches.append(candidate)
    if not matches:
        raise ValueError("source selector is not in verified search results")
    if len(matches) != 1:
        raise ValueError("source selector is ambiguous across verified results")
    return dict(matches[0])


def _paragraph_block(block: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if set(block) != {"type", "text", "source_refs"}:
        raise ValueError("paragraph fields are incomplete or unknown")
    if not isinstance(block["text"], str) or not block["text"].strip():
        raise ValueError("paragraph text must be nonempty")
    refs = block["source_refs"]
    if not isinstance(refs, list):
        raise TypeError("paragraph source_refs must be an array")
    resolved = [_compact_candidate(_resolve_source(ref, candidates)) for ref in refs]
    return {"type": "paragraph", "text": block["text"], "evidence_refs": resolved}


def _document_list_block(
    block: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], list_ids: set[str],
) -> dict[str, Any]:
    if set(block) != {"type", "list_id", "entries"}:
        raise ValueError("document list fields are incomplete or unknown")
    list_id = _nonempty(block["list_id"], "list_id")
    if list_id in list_ids:
        raise ValueError("list_id must be unique")
    list_ids.add(list_id)
    entries = block["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("document list entries must be nonempty")
    seen_docs: set[str] = set()
    output = []
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, Mapping):
            raise TypeError("list entry must be an object")
        if set(entry) - {"position", "doc_id", "label", "source_refs"}:
            raise ValueError("list entry has unknown fields")
        if not {"doc_id", "label", "source_refs"}.issubset(entry):
            raise ValueError("list entry is incomplete")
        if "position" in entry and (type(entry["position"]) is not int
                                       or entry["position"] != position):
            raise ValueError("list positions must be contiguous and one-based")
        doc_id = _nonempty(entry["doc_id"], "doc_id")
        label = entry["label"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("list label must be nonempty")
        if doc_id in seen_docs:
            raise ValueError("document IDs must be unique within one list")
        seen_docs.add(doc_id)
        refs = entry["source_refs"]
        if not isinstance(refs, list) or not refs:
            raise ValueError("list entry needs a verified source")
        resolved = [_compact_candidate(_resolve_source(ref, candidates, doc_id=doc_id))
                    for ref in refs]
        output.append({"position": position, "doc_id": doc_id, "label": label,
                       "evidence_refs": resolved})
    return {"type": "document_list", "list_id": list_id, "entries": output}


def _quote_block(block: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if set(block) != {"type", "text", "doc_id", "source_refs"}:
        raise ValueError("quote fields are incomplete or unknown")
    doc_id = _nonempty(block["doc_id"], "doc_id")
    text = block["text"]
    if not isinstance(text, str) or not text:
        raise ValueError("quote text must be nonempty")
    refs = block["source_refs"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("quote needs a verified source")
    resolved = [_resolve_source(ref, candidates, doc_id=doc_id) for ref in refs]
    if len(resolved) != 1 or resolved[0].get("quote") != text:
        raise ValueError("quote must exactly match one verified source")
    return {"type": "quote", "text": text, "doc_id": doc_id,
            "evidence_refs": [_compact_candidate(resolved[0])]}


def _normalize_model_blocks(blocks: Any, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("presentation blocks must be a nonempty array")
    normalized, list_ids = [], set()
    validators = {"paragraph": _paragraph_block, "quote": _quote_block}
    for block in blocks:
        if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
            raise TypeError("presentation block must be a typed object")
        kind = block["type"]
        if kind == "document_list":
            normalized.append(_document_list_block(block, candidates, list_ids))
        elif kind in validators:
            normalized.append(validators[kind](block, candidates))
        else:
            raise ValueError("unsupported presentation block type")
    return normalized


def _validate_manifest_blocks(value: Any, sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("presentation blocks must be a nonempty array")
    source_ids = {source["source_id"] for source in sources}
    blocks = []
    seen_lists: set[str] = set()
    for block in value:
        if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
            raise TypeError("manifest block is invalid")
        kind = block["type"]
        rows = []
        if kind == "paragraph":
            if (set(block) != {"type", "text", "evidence_refs"}
                    or not isinstance(block["text"], str) or not block["text"].strip()):
                raise ValueError("manifest paragraph is malformed")
            refs = block["evidence_refs"]
            if not isinstance(refs, list):
                raise ValueError("manifest paragraph evidence is malformed")
            rows = refs
        elif kind == "document_list":
            if set(block) != {"type", "list_id", "entries"}:
                raise ValueError("manifest document list is malformed")
            list_id = _nonempty(block["list_id"], "list_id")
            if list_id in seen_lists:
                raise ValueError("manifest list_id is duplicated")
            seen_lists.add(list_id)
            entries = block["entries"]
            if not isinstance(entries, list) or not entries:
                raise ValueError("manifest list entries are malformed")
            normalized_entries, seen_docs = [], set()
            for position, entry in enumerate(entries, 1):
                if not isinstance(entry, Mapping) or set(entry) != {
                        "position", "doc_id", "label", "evidence_refs"}:
                    raise ValueError("manifest list entry is malformed")
                if type(entry["position"]) is not int or entry["position"] != position:
                    raise ValueError("manifest positions are not contiguous")
                doc_id = _nonempty(entry["doc_id"], "doc_id")
                if doc_id in seen_docs:
                    raise ValueError("manifest list contains duplicate documents")
                seen_docs.add(doc_id)
                if not isinstance(entry["label"], str) or not entry["label"].strip():
                    raise ValueError("manifest list label is invalid")
                refs = entry["evidence_refs"]
                if not isinstance(refs, list) or not refs:
                    raise ValueError("manifest list entry has no evidence")
                normalized_entries.append({"position": position, "doc_id": doc_id,
                                           "label": entry["label"],
                                           "evidence_refs": [_validate_compact_ref(ref, source_ids, doc_id)
                                                             for ref in refs]})
            blocks.append({"type": kind, "list_id": list_id, "entries": normalized_entries})
            continue
        elif kind == "quote":
            if set(block) != {"type", "text", "doc_id", "evidence_refs"}:
                raise ValueError("manifest quote is malformed")
            if not isinstance(block["text"], str) or not block["text"]:
                raise ValueError("manifest quote text is invalid")
            doc_id = _nonempty(block["doc_id"], "doc_id")
            refs = block["evidence_refs"]
            if not isinstance(refs, list) or len(refs) != 1:
                raise ValueError("manifest quote must have one evidence reference")
            ref = _validate_compact_ref(refs[0], source_ids, doc_id)
            blocks.append({"type": kind, "text": block["text"], "doc_id": doc_id,
                           "evidence_refs": [ref]})
            continue
        else:
            raise ValueError("unknown manifest block type")
        blocks.append({"type": kind, "text": block["text"],
                       "evidence_refs": [_validate_compact_ref(ref, source_ids) for ref in rows]})
    return blocks


def _validate_compact_ref(value: Any, source_ids: set[str], doc_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - {
            "source_id", "doc_id", "order_index", "range_start", "range_end"}:
        raise ValueError("manifest evidence reference has unknown fields")
    if not {"source_id", "doc_id"}.issubset(value):
        raise ValueError("manifest evidence reference is incomplete")
    source_id = _nonempty(value["source_id"], "source_id")
    selected_doc = _nonempty(value["doc_id"], "doc_id")
    if source_id not in source_ids or (doc_id is not None and selected_doc != doc_id):
        raise ValueError("manifest evidence points to an unknown source/document")
    if "order_index" in value and (type(value["order_index"]) is not int or value["order_index"] < 0):
        raise ValueError("manifest evidence order index is invalid")
    if ("range_start" in value) != ("range_end" in value):
        raise ValueError("manifest evidence range is incomplete")
    for key in ("range_start", "range_end"):
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise ValueError("manifest evidence range is invalid")
    if "range_start" in value and value["range_end"] < value["range_start"]:
        raise ValueError("manifest evidence range is reversed")
    return dict(value)


def _proof_rows(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: source[key] for key in _SOURCE_FIELDS} for source in sources]


def _validate_source_manifest(source: Any, scope: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise ValueError("presentation source metadata is incomplete or malformed")
    for field in ("source_id", "operation_id", "checkpoint_conversation_id", "call_id"):
        _nonempty(source[field], field)
    if not _SHA256.fullmatch(source["source_id"]):
        raise ValueError("presentation source ID is invalid")
    if type(source["ordinal"]) is not int or source["ordinal"] < 0:
        raise ValueError("presentation source ordinal is invalid")
    reference = source["reference"]
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_FIELDS:
        raise ValueError("manifest source reference is incomplete")
    for key, expected in {
        "version": 1, "context_id": scope["context_id"],
        "conversation_id": scope["conversation_id"], "operation_id": source["operation_id"],
        "checkpoint_conversation_id": source["checkpoint_conversation_id"],
        "call_id": source["call_id"], "ordinal": source["ordinal"],
        "payload_kind": f"search-history-{source['ordinal']:08d}",
    }.items():
        if type(reference.get(key)) is not type(expected) or reference.get(key) != expected:
            raise ValueError("manifest source ref identity is contradictory")
    payload = reference["payload"]
    if not isinstance(payload, Mapping) or set(payload) not in (
            {"sha256", "byte_length"},
            {"_input_ref", "sha256", "byte_length", "max_part_bytes"}):
        raise ValueError("manifest source payload ref is malformed")
    if not isinstance(payload["sha256"], str) or not _SHA256.fullmatch(payload["sha256"]):
        raise ValueError("manifest source digest is invalid")
    if type(payload["byte_length"]) is not int or payload["byte_length"] < 0:
        raise ValueError("manifest source byte length is invalid")
    if "_input_ref" in payload:
        if (not isinstance(payload["_input_ref"], str)
                or type(payload["max_part_bytes"]) is not int or payload["max_part_bytes"] < 1):
            raise ValueError("manifest multipart source reference is malformed")
    if source["call_name"] not in {"query", "list_recent_transcripts", "get_transcript"}:
        raise ValueError("manifest source call is unsupported")
    if not isinstance(source["arguments"], Mapping) or not isinstance(source["source_filters"], Mapping):
        raise ValueError("manifest source arguments or filters are invalid")
    if not isinstance(source["completeness"], Mapping):
        raise ValueError("manifest source completeness metadata is invalid")
    completeness = source["completeness"]
    if source["call_name"] == "list_recent_transcripts":
        if (set(completeness) != {"total", "returned", "offset", "limit", "has_more"}
                or any(type(completeness[key]) is not int or completeness[key] < 0
                       for key in ("total", "returned", "offset", "limit"))
                or type(completeness["has_more"]) is not bool
                or completeness["returned"] > completeness["limit"]
                or completeness["returned"] > max(
                    completeness["total"] - completeness["offset"], 0)
                or completeness["has_more"] != (
                    completeness["offset"] + completeness["returned"] < completeness["total"]
                )):
            raise ValueError("manifest source pagination completeness is malformed")
    elif (not {"truncated", "total_matches", "returned_segments"}.issubset(completeness)
          or type(completeness["truncated"]) is not bool
          or type(completeness["total_matches"]) is not int or completeness["total_matches"] < 0
          or type(completeness["returned_segments"]) is not int
          or completeness["returned_segments"] < 0):
        raise ValueError("manifest source completeness metadata is incomplete")
    expected_source_id = hashlib.sha256(json.dumps({
        "context_id": scope["context_id"], "conversation_id": scope["conversation_id"],
        "operation_id": source["operation_id"],
        "checkpoint_conversation_id": source["checkpoint_conversation_id"],
        "call_id": source["call_id"], "ordinal": source["ordinal"],
        "digest": payload["sha256"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source["source_id"] != expected_source_id:
        raise ValueError("manifest source ID does not bind to its ref digest")
    return _json_copy(dict(source), "manifest source")


def _check_manifest_proof(manifest: Mapping[str, Any], sources: Sequence[Mapping[str, Any]],
                          candidates: Sequence[Mapping[str, Any]]) -> None:
    expected_sources = _proof_rows(sources)
    actual_sources = manifest.get("sources")
    if not _strict_json_equal(actual_sources, expected_sources):
        raise ValueError("manifest source metadata differs from verified exchange proofs")
    candidates_by_key = {_candidate_identity(row): row for row in candidates}
    for block in manifest["blocks"]:
        evidence_rows = (block["evidence_refs"] if block["type"] != "document_list"
                         else [ref for entry in block["entries"] for ref in entry["evidence_refs"]])
        full_rows = []
        for ref in evidence_rows:
            key = (ref["source_id"], ref["doc_id"], ref.get("order_index"),
                   ref.get("range_start"), ref.get("range_end"))
            candidate = candidates_by_key.get(key)
            if candidate is None:
                raise ValueError("manifest evidence reference differs from verified result")
            full_rows.append(candidate)
        if block["type"] == "quote" and (len(full_rows) != 1 or full_rows[0].get("quote") != block["text"]):
            raise ValueError("manifest quote differs from verified source text")


@dataclass(frozen=True, slots=True)
class PresentationManifest:
    """Frozen JSON envelope; callers receive a detached dictionary copy."""

    _encoded: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._encoded)


def build_presentation_manifest(
    *, scope: Mapping[str, Any], blocks: Any,
    verified_exchanges: Sequence[Mapping[str, Any]],
    notification_refs: Any = None, revision_refs: Any = None,
) -> PresentationManifest:
    identity = _scope(scope)
    sources, candidates = _verified_inputs(verified_exchanges, identity)
    normalized = _normalize_model_blocks(blocks, candidates)
    manifest: dict[str, Any] = {
        **identity, "schema_version": 1, "blocks": normalized,
        "body": _render(normalized), "sources": _proof_rows(sources),
    }
    manifest["body_sha256"] = hashlib.sha256(manifest["body"].encode("utf-8")).hexdigest()
    if notification_refs is not None:
        manifest["notification_refs"] = _prepared_refs(notification_refs, "notification_refs")
    if revision_refs is not None:
        manifest["revision_refs"] = _prepared_refs(revision_refs, "revision_refs")
    return validate_presentation_manifest(
        manifest, expected_scope=identity, verified_exchanges=verified_exchanges,
    )


def _prepared_refs(value: Any, name: str) -> list[dict[str, Any]]:
    copied = _json_copy(value, name)
    if not isinstance(copied, list):
        raise TypeError(f"{name} must be an array")
    for row in copied:
        if (not isinstance(row, dict) or set(row) != {"notification_id", "revision"}
                or not isinstance(row["notification_id"], str) or not row["notification_id"].strip()
                or type(row["revision"]) is not int or row["revision"] < 1):
            raise ValueError(f"{name} must contain prepared revision references only")
    return copied


def validate_presentation_manifest(
    value: Any, *, expected_scope: Mapping[str, Any] | None = None,
    verified_exchanges: Sequence[Mapping[str, Any]] | None = None,
) -> PresentationManifest:
    """Validate syntax/render/hash; with proofs, also revalidate all evidence.

    Without ``verified_exchanges`` this is explicitly a structural check and
    cannot establish source authenticity or selection authority.
    """
    if not isinstance(value, Mapping) or set(value) - _MANIFEST_FIELDS:
        raise ValueError("presentation manifest is not a supported object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("presentation schema version must be integer 1")
    identity = _scope({key: value.get(key) for key in _IDENTITY})
    if expected_scope is not None and identity != _scope(expected_scope):
        raise ValueError("presentation scope does not match")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise ValueError("presentation sources must be an array")
    sources = [_validate_source_manifest(source, identity) for source in sources]
    blocks = _validate_manifest_blocks(value.get("blocks"), sources)
    body, digest = value.get("body"), value.get("body_sha256")
    if not isinstance(body, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("presentation body or hash is invalid")
    if body != _render(blocks) or hashlib.sha256(body.encode("utf-8")).hexdigest() != digest:
        raise ValueError("presentation body/hash do not match blocks")
    normalized: dict[str, Any] = {**identity, "schema_version": 1, "blocks": blocks,
                                  "body": body, "body_sha256": digest,
                                  "sources": _json_copy(sources, "manifest sources")}
    for key in ("notification_refs", "revision_refs"):
        if key in value:
            normalized[key] = _prepared_refs(value[key], key)
    if verified_exchanges is not None:
        confirmed_sources, candidates = _verified_inputs(verified_exchanges, identity)
        _check_manifest_proof(normalized, confirmed_sources, candidates)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return PresentationManifest(encoded)
