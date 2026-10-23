"""Build Telegram lifecycle notifications from verified operation evidence."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any


def terminal_lifecycle_status(state: str, completion_confirmed: bool) -> tuple[str, str, str, str]:
    """Map a completed save to its verified final state or a verification error."""
    if state == "completed":
        if completion_confirmed:
            return "save", "complete", "completed", "final"
        return "save", "verification", "completion_unverified", "error"
    return "save", "processing", "in_progress", "progress"


def _load_terminal_notification_draft(
    client: Any, context_id: str, operation_id: str,
) -> tuple[dict[str, Any], int] | None:
    """Resolve the persisted command draft and its public dialog number."""
    from services.connectors.telegram.repository import SESSIONS

    context_ref = client.collection(SESSIONS).document(context_id)
    queue_operation = context_ref.collection("operations").document(operation_id).get()
    if not queue_operation.exists:
        return None
    input_ref = (queue_operation.to_dict() or {}).get("input_ref")
    if not isinstance(input_ref, str):
        return None
    command = client.document(input_ref).get()
    if not command.exists:
        return None
    draft_id = (command.to_dict() or {}).get("draft_id")
    if not isinstance(draft_id, str) or not draft_id:
        return None
    draft_snapshot = context_ref.collection("drafts").document(draft_id).get()
    if not draft_snapshot.exists:
        return None
    draft = draft_snapshot.to_dict() or {}
    dialog_number = draft.get("dialog_number")
    if type(dialog_number) is not int or dialog_number < 1:
        return None
    return draft, dialog_number


def _load_completion_certificate(
    client: Any, operation: Any, operation_id: str, draft: dict[str, Any],
) -> dict[str, Any] | None:
    """Load the shared completion certificate when it matches the local draft."""
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION

    local_certificate = draft.get("completion_certificate") or {}
    intent_id = local_certificate.get("save_intent_id")
    if not isinstance(intent_id, str):
        return None
    if draft.get("completion_certificate_ref") != f"{SAVE_INTENTS_COLLECTION}/{intent_id}":
        return None
    intent_snapshot = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id).get()
    if not intent_snapshot.exists:
        return None
    intent = intent_snapshot.to_dict() or {}
    certificate = intent.get("completion_certificate") or {}
    confirmed = intent.get("confirmed_result") or {}
    if not _completion_certificate_matches_operation(
        certificate, local_certificate, confirmed, operation, operation_id, intent_id,
    ):
        return None
    return certificate


def _completion_certificate_matches_operation(
    certificate: dict[str, Any], local_certificate: dict[str, Any], confirmed: dict[str, Any],
    operation: Any, operation_id: str, intent_id: str,
) -> bool:
    """Check that completion evidence belongs to this operation and save intent."""
    doc_id = certificate.get("doc_id")
    content_hash = certificate.get("content_hash")
    receipt_ref = certificate.get("receipt_ref")
    return (
        certificate.get("operation_id") == operation_id
        and certificate.get("save_intent_id") == intent_id
        and local_certificate.get("operation_id") == operation_id
        and local_certificate.get("save_intent_id") == intent_id
        and operation.save_intent_id == intent_id
        and confirmed.get("document_id") == doc_id == operation.doc_id
        and operation.result is not None
        and operation.result.document_id == doc_id
        and isinstance(doc_id, str) and isinstance(content_hash, str)
        and receipt_ref == f"index_completions/{doc_id}/generations/{content_hash}"
        and operation.index_completion_ref == receipt_ref
        and operation.desired_content_hash == content_hash
    )


def _completion_receipt_matches_index(client: Any, certificate: dict[str, Any]) -> bool:
    """Verify the completion receipt against the currently indexed transcript generation."""
    from core.integrations.firestore.index_completions import get_completion

    doc_id = certificate.get("doc_id")
    content_hash = certificate.get("content_hash")
    receipt = get_completion(client, doc_id, content_hash)
    transcript_snapshot = client.collection("transcript_index").document(doc_id).get()
    if receipt is None or not transcript_snapshot.exists:
        return False
    transcript = transcript_snapshot.to_dict() or {}
    generation_fields = (
        "doc_id", "root_folder_id", "content_hash", "version", "parent_id",
        "parent_id_hash", "provider", "fence", "index_fence", "resource_epoch",
    )
    return (
        receipt.get("status") == "completed"
        and receipt.get("version") == certificate.get("receipt_version")
        and receipt.get("index_fence") == certificate.get("index_fence")
        and receipt.get("resource_epoch") == certificate.get("resource_epoch")
        and transcript.get("status") == "synced"
        and all(transcript.get(field) == receipt.get(field) for field in generation_fields)
    )


def _verified_completion_details(
    client: Any, operation: Any, operation_id: str, draft: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
    """Return verified folder details and a bounded Telegram classification outcome."""
    if operation.state.value != "completed":
        return False, None, None
    certificate = _load_completion_certificate(client, operation, operation_id, draft)
    if certificate is None or not _completion_receipt_matches_index(client, certificate):
        return False, None, None
    path_parts = operation.selected_folder_path
    folder_path = (
        "/".join(path_parts)
        if path_parts and all(isinstance(part, str) for part in path_parts)
        else None
    )
    local_certificate = draft.get("completion_certificate") or {}
    classification_outcome = local_certificate.get("classification_outcome")
    if classification_outcome not in {"ambiguous", "unassigned"}:
        classification_outcome = None
    elif classification_outcome == "unassigned" and path_parts != ("_unassigned",):
        classification_outcome = None
    return True, folder_path, classification_outcome


def _safe_lifecycle_code(value: Any, fallback: str) -> str:
    """Keep externally rendered lifecycle codes within their public identifier format."""
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) else fallback


def _terminal_notification_status(operation: Any, completion_confirmed: bool) -> tuple[str, str, str, str]:
    """Map a terminal operation into the safe notification status fields."""
    if operation.state.value == "failed":
        error = operation.error
        service = getattr(error, "service", None) or "telegram"
        stage = getattr(error, "stage", None) or operation.stage or "processing"
        reason = getattr(error, "code", None) or "operation_failed"
        kind = "error"
    else:
        service, stage, reason, kind = terminal_lifecycle_status(
            operation.state.value, completion_confirmed,
        )
    return (
        _safe_lifecycle_code(service, "telegram"),
        _safe_lifecycle_code(stage, "processing"),
        _safe_lifecycle_code(reason, "operation_failed"),
        kind,
    )


def _persist_terminal_notification(
    client: Any, record: Any, dialog_number: int, operation: Any,
    completion_confirmed: bool, folder_path: str | None,
    classification_outcome: str | None = None,
) -> str:
    """Build the terminal message and create its idempotent notification ledger entry."""
    from services.connectors.telegram.notifications import (
        _VerifiedTelegramLifecycle, build_lifecycle_body,
    )
    from services.connectors.telegram.repository import TelegramRepository

    service, stage, reason, kind = _terminal_notification_status(operation, completion_confirmed)
    lifecycle = _VerifiedTelegramLifecycle(
        dialog_number=dialog_number, operation_id=record.operation_id,
        service=service, stage=stage, reason=reason, folder_path=folder_path,
        completion_confirmed=completion_confirmed,
        classification_outcome=classification_outcome,
    )
    body = build_lifecycle_body(lifecycle=lifecycle)
    notification_id = "tg" + kind + "-" + hashlib.sha256(record.operation_id.encode("utf-8")).hexdigest()
    repository = TelegramRepository(client)
    if repository.get_notification(notification_id) is None:
        repository.create_notification(
            notification_id, context_id=record.context_id, kind=kind, body=body,
            now=datetime.now(timezone.utc), dialog_number=dialog_number,
            operation_id=record.operation_id,
        )
    return notification_id


def update_terminal_notification(record: Any) -> str | None:
    """Create a stable lifecycle ledger entry only from persisted terminal operation evidence."""
    if not record.operation_id or not record.context_id:
        return None
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.operations import get_operation

    client = get_client()
    operation = get_operation(client, record.operation_id)
    if operation is None or not operation.state.is_terminal:
        return None
    draft_details = _load_terminal_notification_draft(client, record.context_id, record.operation_id)
    if draft_details is None:
        return None
    draft, dialog_number = draft_details
    completion_confirmed, folder_path, classification_outcome = _verified_completion_details(
        client, operation, record.operation_id, draft,
    )
    return _persist_terminal_notification(
        client, record, dialog_number, operation, completion_confirmed, folder_path,
        classification_outcome,
    )
