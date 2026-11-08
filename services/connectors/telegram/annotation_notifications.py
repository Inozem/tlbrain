"""Proof-bound terminal notices for internal Telegram context-note operations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


_DRAFT_PROOF_FIELDS = (
    "draft_id", "dialog_number", "active_save_operation_id",
    "immutable_save_intent_id", "completion_certificate_ref",
    "manifest_digest", "completion_certificate",
)
_COMMENT_NOTICE_FIELDS = (
    "context_id", "draft_id", "comment_id", "application_operation_id",
    "applied_doc_id", "completion_certificate_ref", "completion_receipt_ref",
    "state", "error_code",
)
_ERROR_BODY = (
    "The dialog was saved and indexed, but its context note could not be added. "
    "The note remains stored with the dialog."
)


def is_annotation_apply_operation(operation: Any) -> bool:
    """Return whether a persisted queue record carries the private apply marker."""
    return isinstance(operation, dict) and operation.get("telegram_internal_kind") == "annotation_apply"


def _operation_binding(
    record: Any, operation: Mapping[str, Any], state: str,
) -> tuple[str, str]:
    context_id = record.context_id
    operation_id = record.operation_id
    if (state not in {"completed", "failed"}
            or operation.get("state") != state
            or not isinstance(context_id, str) or not isinstance(operation_id, str)
            or operation.get("context_id") != context_id
            or operation.get("operation_id") != operation_id):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    input_ref = operation.get("input_ref")
    parts = input_ref.split("/") if isinstance(input_ref, str) else []
    if (len(parts) != 6 or parts[:2] != ["telegram_sessions", context_id]
            or parts[2] != "drafts" or parts[4] != "annotations"
            or not parts[3] or not parts[5]):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    draft_id, comment_id = parts[3], parts[5]
    expected = f"telegram_sessions/{context_id}/drafts/{draft_id}/annotations/{comment_id}"
    if (input_ref != expected or operation.get("draft_id") != draft_id
            or operation.get("annotation_id") != comment_id):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    return draft_id, comment_id


def _delivered_legacy_notice(repository: Any, record: Any) -> str | None:
    operation_id = record.operation_id
    legacy_id = "tgterminal-" + hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
    legacy = repository.get_notification(legacy_id)
    if (legacy and legacy.get("notification_id") == legacy_id
            and legacy.get("context_id") == record.context_id
            and legacy.get("operation_id") == operation_id
            and legacy.get("kind") in {"final", "error"}
            and legacy.get("state") == "delivered"):
        return legacy_id
    return None


def _load_bound_records(
    client: Any, record: Any, operation: Mapping[str, Any], draft_id: str, comment_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    context_ref = client.collection("telegram_sessions").document(record.context_id)
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft_snapshot = draft_ref.get(field_paths=_DRAFT_PROOF_FIELDS)
    comment_snapshot = draft_ref.collection("annotations").document(comment_id).get(
        field_paths=_COMMENT_NOTICE_FIELDS,
    )
    draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
    comment = comment_snapshot.to_dict() or {} if comment_snapshot.exists else {}
    target_doc_id = operation.get("target_doc_id")
    if (not draft_snapshot.exists or not comment_snapshot.exists
            or draft.get("draft_id") != draft_id
            or not isinstance(target_doc_id, str) or not target_doc_id
            or comment.get("context_id") != record.context_id
            or comment.get("draft_id") != draft_id
            or comment.get("comment_id") != comment_id
            or comment.get("application_operation_id") != record.operation_id
            or comment.get("applied_doc_id") != target_doc_id
            or comment.get("state") not in {"applied", "failed"}
            or (comment.get("state") == "applied" and operation.get("state") != "completed")
            or type(draft.get("dialog_number")) is not int
            or draft["dialog_number"] < 1):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    return draft, comment, target_doc_id


def _saved_document_proof_matches(
    client: Any, record: Any, operation: Mapping[str, Any],
    draft_id: str, draft: Mapping[str, Any], comment: Mapping[str, Any], target_doc_id: str,
) -> bool:
    from services.connectors.telegram.repository._annotation_proof import (
        load_annotation_completion_proof,
    )

    proof = load_annotation_completion_proof(
        client, record.context_id, draft_id, draft, expected_doc_id=target_doc_id,
    )
    return (
        isinstance(proof, Mapping)
        and operation.get("completion_receipt_ref") == proof.get("receipt_ref")
        and operation.get("completion_certificate_ref") == draft.get("completion_certificate_ref")
        and comment.get("completion_receipt_ref") == proof.get("receipt_ref")
        and comment.get("completion_certificate_ref") == draft.get("completion_certificate_ref")
    )


def _notice_identity_and_body(comment: Mapping[str, Any], dialog_number: int) -> tuple[str, str, str]:
    digest = hashlib.sha256(str(comment["comment_id"]).encode("utf-8")).hexdigest()
    if comment.get("state") == "failed":
        return "tgannotation-error-" + digest, "error", _ERROR_BODY
    return (
        "tgannotation-final-" + digest, "final",
        f"Dialog #{dialog_number}: Context note added to search.",
    )


def resolve_annotation_terminal_notification(
    client: Any, record: Any, operation: dict[str, Any] | None, state: str, *, now: Any,
) -> str:
    """Return a canonical annotation notice after validating its full lineage."""
    from services.connectors.telegram.repository import TelegramRepository

    if not is_annotation_apply_operation(operation):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    draft_id, comment_id = _operation_binding(record, operation, state)
    repository = TelegramRepository(client)
    legacy_id = _delivered_legacy_notice(repository, record)
    if legacy_id is not None:
        return legacy_id
    draft, comment, target_doc_id = _load_bound_records(
        client, record, operation, draft_id, comment_id,
    )
    if not _saved_document_proof_matches(
        client, record, operation, draft_id, draft, comment, target_doc_id,
    ):
        raise RuntimeError("Telegram annotation completion proof is unavailable")
    notification_id, kind, body = _notice_identity_and_body(comment, draft["dialog_number"])
    repository.create_notification(
        notification_id, context_id=record.context_id, kind=kind, body=body,
        dialog_number=draft["dialog_number"], operation_id=record.operation_id,
        now=now,
    )
    return notification_id
