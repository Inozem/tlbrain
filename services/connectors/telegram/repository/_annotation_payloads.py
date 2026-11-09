"""Verified bounded storage for private Telegram annotation text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.integrations.firestore.input_parts import MAX_PART_BYTES, read_input_parts
from services.connectors.telegram.repository._common import SESSIONS, _document_ref


def load_annotation_payload(
    client: Any, payload: Mapping[str, Any], *, context_id: str,
    operation_id: str, payload_kind: str = "annotation",
) -> str | None:
    """Load an annotation payload only from its original accepted operation."""
    if (not isinstance(context_id, str) or not context_id
            or not isinstance(operation_id, str) or not operation_id
            or payload_kind != "annotation"):
        return None
    digest = payload.get("sha256")
    byte_length = payload.get("byte_length")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
            or type(byte_length) is not int or byte_length <= 0):
        return None
    inline = payload.get("_inline")
    if isinstance(inline, str):
        if set(payload) != {"_inline", "sha256", "byte_length"}:
            return None
        encoded = json.dumps(inline, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() == digest and len(encoded) == byte_length:
            return inline
        return None
    input_ref = payload.get("_input_ref")
    if (not isinstance(input_ref, str) or set(payload) != {
            "_input_ref", "sha256", "byte_length", "max_part_bytes",
        } or type(payload.get("max_part_bytes")) is not int
            or payload["max_part_bytes"] != MAX_PART_BYTES):
        return None
    ledger_id = "telegram-ledger-" + hashlib.sha256(
        f"{context_id}\0{operation_id}\0{payload_kind}\0{digest}".encode("utf-8")
    ).hexdigest()
    if input_ref != f"dialog_operations/{ledger_id}/input_parts/manifest":
        return None
    try:
        encoded = read_input_parts(client, input_ref)
        if hashlib.sha256(encoded).hexdigest() != digest or len(encoded) != byte_length:
            return None
        text = json.loads(encoded.decode("utf-8"))
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        return None
    return text if isinstance(text, str) else None


def load_annotation_row_text(client: Any, row: Mapping[str, Any]) -> str | None:
    """Verify a note against its accepted operation and bound route before reading it."""
    context_id = row.get("context_id")
    operation_id = row.get("operation_id")
    if (not isinstance(context_id, str) or not isinstance(operation_id, str)
            or not context_id or not operation_id):
        return None
    operation_ref = (client.collection(SESSIONS).document(context_id)
                     .collection("operations").document(operation_id))
    snapshot = operation_ref.get()
    operation = snapshot.to_dict() or {} if snapshot.exists else {}
    route = operation.get("route_control")
    command_ref = operation.get("input_ref")
    command_parts = command_ref.split("/") if isinstance(command_ref, str) else []
    command_snapshot = (
        _document_ref(client, command_ref).get()
        if len(command_parts) == 4 and command_parts[:3] == [SESSIONS, context_id, "commands"]
        and command_parts[3].isdigit() else None
    )
    command = (command_snapshot.to_dict() or {}
               if command_snapshot is not None and command_snapshot.exists else {})
    if (not snapshot.exists or operation.get("operation_id") != operation_id
            or len(command_parts) != 4 or command_parts[:3] != [SESSIONS, context_id, "commands"]
            or not command_parts[3].isdigit()
            or not command_snapshot or not command_snapshot.exists
            or command.get("context_id") != context_id
            or command.get("operation_id") != operation_id
            or not isinstance(route, Mapping) or route.get("kind") != "annotation"
            or route.get("status") not in {"ok", "success"}):
        return None
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("kind") != "telegram_user_annotation":
        return None
    payload = row.get("text_payload")
    route_payload = route.get("annotation_payload")
    if isinstance(payload, Mapping):
        if (not isinstance(route_payload, Mapping)
                or dict(route_payload) != dict(payload)
                or route.get("annotation_text_sha256") != row.get("text_sha256")):
            return None
        text = load_annotation_payload(
            client, payload, context_id=context_id, operation_id=operation_id,
        )
    else:
        text = row.get("text")
        if not isinstance(text, str) or route.get("annotation_text") != text:
            return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None
    if (not isinstance(text, str) or not text.strip()
            or digest != row.get("text_sha256", digest)
            or (isinstance(route.get("annotation_text_sha256"), str)
                and digest != route.get("annotation_text_sha256"))):
        return None
    target_draft_id = row.get("draft_id")
    target_doc_id = row.get("doc_id")
    if (target_draft_id is not None and route.get("target_draft_id") != target_draft_id
            or target_doc_id is not None and route.get("target_doc_id") != target_doc_id):
        return None
    return text
