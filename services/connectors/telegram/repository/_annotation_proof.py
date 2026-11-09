"""Canonical proof for applying a private note to a saved Telegram dialog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from services.connectors.telegram.repository._common import SESSIONS, _document_ref


def _origin_matches(value: Any, context_id: str, draft_id: str, digest: str) -> bool:
    expected = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest}
    if isinstance(value, Mapping):
        return dict(value) == expected
    return isinstance(value, (tuple, list)) and tuple(value) == (
        context_id, draft_id, digest,
    )


def validate_annotation_completion_proof(
    *, context_id: str, draft_id: str, draft: Mapping[str, Any],
    intent_id: str, intent: Mapping[str, Any], core_operation: Mapping[str, Any],
    receipt: Mapping[str, Any], expected_doc_id: str | None = None,
) -> dict[str, Any] | None:
    """Validate the same committed save, cleanup lineage, and receipt for every caller."""
    local = draft.get("completion_certificate")
    shared = intent.get("completion_certificate")
    digest = draft.get("manifest_digest")
    if (not isinstance(local, Mapping) or not isinstance(shared, Mapping)
            or not isinstance(digest, str) or not digest
            or draft.get("immutable_save_intent_id") != intent_id
            or draft.get("completion_certificate_ref") != f"{SAVE_INTENTS_COLLECTION}/{intent_id}"):
        return None
    operation_id = local.get("operation_id")
    doc_id = shared.get("doc_id")
    content_hash = shared.get("content_hash")
    receipt_ref = shared.get("receipt_ref")
    binding = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest}
    cleanup_authorization = {
        "certificate_id": shared.get("certificate_id"),
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "origin_binding": binding,
    }
    expected_receipt_ref = (
        f"index_completions/{doc_id}/generations/{content_hash}"
        if isinstance(doc_id, str) and isinstance(content_hash, str) and content_hash else None
    )
    if (not isinstance(operation_id, str) or not operation_id
            or draft.get("active_save_operation_id") != operation_id
            or core_operation.get("operation_id") != operation_id
            or core_operation.get("state") != "completed"
            or core_operation.get("doc_id") != doc_id
            or core_operation.get("save_intent_id") != intent_id
            or intent.get("doc_id") != doc_id
            or shared.get("operation_id") != operation_id
            or shared.get("save_intent_id") != intent_id
            or not isinstance(shared.get("certificate_id"), str)
            or not shared.get("certificate_id")
            or not isinstance(shared.get("origin_binding"), Mapping)
            or dict(shared["origin_binding"]) != binding
            or not _origin_matches(local.get("origin_binding"), context_id, draft_id, digest)
            or not isinstance(core_operation.get("cleanup_authorization"), Mapping)
            or dict(core_operation["cleanup_authorization"]) != cleanup_authorization
            or intent.get("cleanup_authorization") != cleanup_authorization
            or (core_operation.get("origin_binding") is not None
                and core_operation.get("origin_binding") != binding)
            or (expected_doc_id is not None and expected_doc_id != doc_id)
            or local.get("context_id") != context_id
            or local.get("draft_id") != draft_id
            or local.get("manifest_digest") != digest
            or local.get("save_intent_id") != intent_id
            or local.get("receipt_ref") != receipt_ref
            or local.get("receipt_version") != shared.get("receipt_version")
            or local.get("doc_id") not in (None, doc_id)
            or receipt_ref != expected_receipt_ref
            or receipt.get("status") != "completed"
            or receipt.get("doc_id") != doc_id
            or receipt.get("content_hash") != content_hash
            or receipt.get("version") != shared.get("receipt_version")):
        return None
    return {
        "context_id": context_id, "draft_id": draft_id,
        "manifest_digest": digest, "save_intent_id": intent_id,
        "operation_id": operation_id, "doc_id": doc_id,
        "content_hash": content_hash, "receipt_ref": receipt_ref,
        "receipt_version": receipt.get("version"),
    }


def load_annotation_completion_proof(
    client: Any, context_id: str, draft_id: str, draft: Mapping[str, Any], *,
    transaction: Any = None, expected_doc_id: str | None = None,
) -> dict[str, Any] | None:
    """Load all shared save evidence, optionally in the caller's transaction."""
    local = draft.get("completion_certificate")
    intent_id = local.get("save_intent_id") if isinstance(local, Mapping) else None
    operation_id = local.get("operation_id") if isinstance(local, Mapping) else None
    if not isinstance(intent_id, str) or not intent_id or not isinstance(operation_id, str):
        return None
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    operation_ref = client.collection("dialog_operations").document(operation_id)
    intent_snapshot = intent_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    intent = intent_snapshot.to_dict() or {} if intent_snapshot.exists else {}
    core_operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
    shared = intent.get("completion_certificate")
    doc_id = shared.get("doc_id") if isinstance(shared, Mapping) else None
    content_hash = shared.get("content_hash") if isinstance(shared, Mapping) else None
    receipt_path = shared.get("receipt_ref") if isinstance(shared, Mapping) else None
    expected_receipt = (
        f"index_completions/{doc_id}/generations/{content_hash}"
        if isinstance(doc_id, str) and isinstance(content_hash, str) and content_hash else None
    )
    receipt = {}
    if isinstance(receipt_path, str) and receipt_path == expected_receipt:
        receipt_snapshot = _document_ref(client, receipt_path).get(transaction=transaction)
        receipt = receipt_snapshot.to_dict() or {} if receipt_snapshot.exists else {}
    return validate_annotation_completion_proof(
        context_id=context_id, draft_id=draft_id, draft=draft,
        intent_id=intent_id, intent=intent, core_operation=core_operation,
        receipt=receipt, expected_doc_id=expected_doc_id,
    )
