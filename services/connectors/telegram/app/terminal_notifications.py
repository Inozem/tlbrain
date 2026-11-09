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
    from services.connectors.telegram.repository._common import _document_ref

    context_ref = client.collection(SESSIONS).document(context_id)
    queue_operation = context_ref.collection("operations").document(operation_id).get()
    if not queue_operation.exists:
        return None
    input_ref = (queue_operation.to_dict() or {}).get("input_ref")
    if not isinstance(input_ref, str):
        return None
    command = _document_ref(client, input_ref).get()
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
    core_operation = (client.collection("dialog_operations").document(operation_id)
                      .get().to_dict() or {})
    if not _completion_certificate_matches_operation(
        certificate, local_certificate, confirmed, operation, operation_id, intent_id,
    ):
        return None
    origin = certificate.get("origin_binding")
    expected_origin = {
        "context_id": local_certificate.get("context_id"),
        "draft_id": local_certificate.get("draft_id"),
        "manifest_digest": local_certificate.get("manifest_digest"),
    }
    cleanup_authorization = {
        "certificate_id": certificate.get("certificate_id"),
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "origin_binding": origin,
    }
    if (not isinstance(certificate.get("certificate_id"), str)
            or not certificate.get("certificate_id")
            or origin != expected_origin
            or (core_operation.get("origin_binding") is not None
                and core_operation.get("origin_binding") != expected_origin)
            or intent.get("cleanup_authorization") != cleanup_authorization
            or core_operation.get("cleanup_authorization") != cleanup_authorization
            or local_certificate.get("receipt_ref") != certificate.get("receipt_ref")
            or local_certificate.get("receipt_version") != certificate.get("receipt_version")):
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
    """Verify the pinned completed receipt without revoking an immutable save later."""
    from core.integrations.firestore.index_completions import get_completion

    doc_id = certificate.get("doc_id")
    content_hash = certificate.get("content_hash")
    receipt = get_completion(client, doc_id, content_hash)
    if receipt is None:
        return False
    return (
        receipt.get("status") == "completed"
        and receipt.get("version") == certificate.get("receipt_version")
        and certificate.get("receipt_ref")
        == f"index_completions/{doc_id}/generations/{content_hash}"
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
    folder_path = _verified_folder_path(path_parts)
    local_certificate = draft.get("completion_certificate") or {}
    classification_outcome = local_certificate.get("classification_outcome")
    if classification_outcome not in {"ambiguous", "unassigned"}:
        classification_outcome = None
    elif classification_outcome == "unassigned" and path_parts != ("_unassigned",):
        classification_outcome = None
    return True, folder_path, classification_outcome


def _verified_folder_path(path_parts: Any) -> str | None:
    """Preserve root as `/`, while keeping unknown or malformed paths distinct."""
    if path_parts is None:
        return None
    if not isinstance(path_parts, (tuple, list)):
        return None
    if len(path_parts) == 0:
        return "/"
    if not all(
        isinstance(part, str) and part for part in path_parts
    ):
        return None
    return "/".join(path_parts)


def _safe_lifecycle_code(value: Any, fallback: str) -> str:
    """Keep externally rendered lifecycle codes within their public identifier format."""
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value) else fallback


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


def _has_failed_stage_error(stage_data: dict[str, Any]) -> bool:
    """Accept only failed stage documents that contain a structured last error."""
    return stage_data.get("state") == "failed" and isinstance(stage_data.get("last_error"), dict)


def _preferred_failed_stage(stages_ref: Any, preferred_stage: Any) -> dict[str, Any]:
    """Load the preferred stage only when it records a structured failure."""
    if not isinstance(preferred_stage, str) or not preferred_stage:
        return {}
    snapshot = stages_ref.document(preferred_stage).get()
    candidate = snapshot.to_dict() or {} if snapshot.exists else {}
    return candidate if _has_failed_stage_error(candidate) else {}


def _first_failed_stage(stages_ref: Any) -> dict[str, Any]:
    """Choose the failed stage with the smallest stage key for deterministic fallback."""
    failed_stages = []
    for snapshot in stages_ref.stream():
        candidate = snapshot.to_dict() or {}
        if _has_failed_stage_error(candidate):
            key = candidate.get("stage_key")
            if not isinstance(key, str) or not key:
                key = snapshot.id
            failed_stages.append((key, candidate))
    return min(failed_stages, key=lambda item: item[0])[1] if failed_stages else {}


def _queue_failed_stage_data(client: Any, record: Any, operation: Any) -> dict[str, Any]:
    """Resolve this queued operation's preferred failed stage, then its deterministic fallback."""
    queue_ref = (client.collection("telegram_sessions").document(record.context_id)
                 .collection("operations").document(record.operation_id))
    queue_snapshot = queue_ref.get()
    queue_operation = queue_snapshot.to_dict() or {} if queue_snapshot.exists else {}
    stages_ref = queue_ref.collection("stages")
    preferred_stage = getattr(operation, "stage", None) or queue_operation.get("stage")
    return _preferred_failed_stage(stages_ref, preferred_stage) or _first_failed_stage(stages_ref)


def _error_field(error: Any, name: str) -> Any:
    """Read an error field from either a mapping or an attribute-based error object."""
    return error.get(name) if isinstance(error, dict) else getattr(error, name, None)


def _terminal_error_details(client: Any, record: Any, operation: Any) -> dict[str, Any]:
    """Resolve safe shared or Telegram stage error fields for a failed operation."""
    error = getattr(operation, "error", None)
    stage_data: dict[str, Any] = {}
    if error is None and record.context_id and record.operation_id:
        stage_data = _queue_failed_stage_data(client, record, operation)
        error = stage_data.get("last_error") if stage_data else None
    if error is None:
        return {}
    return {
        "service": _error_field(error, "service"),
        "stage": _error_field(error, "stage") or stage_data.get("stage_key"),
        "code": _error_field(error, "code"),
        "category": _error_field(error, "category"),
        "sanitized_message": _error_field(error, "sanitized_message"),
        "occurred_at": _error_field(error, "occurred_at"),
        "failures": _error_field(error, "failures"),
    }


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
    error_details = (
        _terminal_error_details(client, record, operation)
        if operation.state.value == "failed" else {}
    )
    service = _safe_lifecycle_code(error_details.get("service"), service)
    stage = _safe_lifecycle_code(error_details.get("stage"), stage)
    lifecycle = _VerifiedTelegramLifecycle(
        dialog_number=dialog_number, operation_id=record.operation_id,
        service=service, stage=stage, reason=reason, folder_path=folder_path,
        completion_confirmed=completion_confirmed,
        classification_outcome=classification_outcome,
        code=error_details.get("code"), category=error_details.get("category"),
        occurred_at=error_details.get("occurred_at"),
        failures=error_details.get("failures"),
        sanitized_message=error_details.get("sanitized_message"),
        title=operation.title if completion_confirmed else None,
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


def _linked_final_notification_id(client: Any, record: Any) -> str | None:
    """Return this operation's linked final ledger ID only when its identity is valid."""
    from services.connectors.telegram.repository import SESSIONS, TelegramRepository

    if not record.context_id or not record.operation_id:
        return None
    operation = (client.collection(SESSIONS).document(record.context_id)
                 .collection("operations").document(record.operation_id).get())
    notification_id = (operation.to_dict() or {}).get("notification_id") if operation.exists else None
    if not isinstance(notification_id, str) or not notification_id:
        return None
    notification = TelegramRepository(client).get_notification(notification_id)
    receipt_terminal = bool(
        notification and notification.get("receipt_card") is True
        and notification.get("receipt_terminal") is True
        and notification.get("save_operation_id") == record.operation_id
        and notification.get("context_id") == record.context_id
    )
    if not notification or (
        notification.get("notification_id") != notification_id
        or notification.get("context_id") != record.context_id
        or notification.get("operation_id") != record.operation_id
        or (notification.get("kind") not in {"final", "error"} and not receipt_terminal)
        or notification.get("state") not in {
            "pending", "retry", "sending", "editing", "delivered", "failed",
            "delivery_unknown",
        }
    ):
        return None
    return notification_id


def update_terminal_notification(record: Any) -> str | None:
    """Create a stable lifecycle ledger entry only from persisted terminal operation evidence."""
    if not record.operation_id or not record.context_id:
        return None
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.operations import get_operation

    client = get_client()
    linked_notification_id = _linked_final_notification_id(client, record)
    if linked_notification_id:
        return linked_notification_id
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
