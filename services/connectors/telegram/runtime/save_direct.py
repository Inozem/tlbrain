"""Proof-checked direct Telegram save failure and completion handling."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from core.operations.models import OperationOrigin, OperationState
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from services.connectors.telegram.save import TelegramSavePending

logger = logging.getLogger("services.connectors.telegram.runtime.saving")


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _normalize_completed_direct_arguments(
    arguments: Mapping[str, Any], *, operation_date: str, accepted_at: datetime,
) -> Any:
    """Normalize replays against immutable accepted dates when date was omitted."""
    from core.dialogs.models import normalize_dialog_input

    comparison_arguments = dict(arguments)
    if "dialog_date" not in comparison_arguments:
        comparison_arguments["dialog_date"] = operation_date
    return normalize_dialog_input(comparison_arguments, accepted_at=accepted_at)


def _classification_failure_details(error: Any, failures: Any) -> dict[str, Any] | None:
    """Expose a classification error only after its durable failure budget is terminal."""
    terminal = (
        not _get(error, "retryable", True)
        or (type(failures) is int and failures >= 3)
    )
    if not terminal:
        return None
    return {
        "service": _get(error, "service") or "gemini",
        "stage": _get(error, "stage") or "classification",
        "code": _get(error, "code"),
        "category": _get(error, "category"),
        "failures": failures,
        "occurred_at": _get(error, "occurred_at"),
        "sanitized_message": _get(error, "sanitized_message") or _get(error, "message"),
    }


def _finalization_failure_details(
    governor: Any, parent_operation_id: str, child_id: str, child: Any,
) -> dict[str, Any] | None:
    """Read an exhausted Docs/Drive finalizer stage after the child is terminal."""
    for boundary, service in (("docs", "google_docs"), ("drive", "google_drive")):
        stage_key = f"telegram.finalize.{child_id}.{boundary}"
        final_stage = governor.get_stage(parent_operation_id, stage_key)
        error = _get(final_stage, "last_error")
        failures = _get(final_stage, "failures", 0)
        if error is not None and (
            not _get(error, "retryable", True)
            or failures >= getattr(governor, "max_failures", 3)
        ):
            return {
                "service": _get(error, "service") or service,
                "stage": "finalization",
                "code": _get(error, "code"),
                "category": _get(error, "category"),
                "failures": failures,
                "occurred_at": _get(error, "occurred_at"),
                "sanitized_message": _get(error, "sanitized_message")
                or _get(error, "message"),
            }
    return None


def _format_direct_failure(
    parent_operation_id: str, context_id: str, child_id: str | None,
    child: Any, details: Mapping[str, Any],
) -> str | None:
    """Log and render one verified, exhausted direct-save failure safely."""
    from services.connectors.telegram.errors import format_terminal_error

    failures = details.get("failures")
    if type(failures) is not int or failures < 1:
        return None
    logger.error("Telegram direct save failed", extra={
        "event": "telegram_direct_save_settlement",
        "operation_id": parent_operation_id,
        **({"child_operation_id": child_id} if child is not None else {}),
        "context_id": context_id, "doc_id": child.doc_id if child is not None else None,
        "service": details["service"], "stage": details["stage"],
        "error_code": details["code"], "error_category": details["category"],
        "failures": failures, "outcome": "failed",
    })
    return format_terminal_error(
        dialog_number=None, operation_id=parent_operation_id, service=details["service"],
        stage=details["stage"] or "save", code=details["code"],
        category=details["category"], sanitized_message=details["sanitized_message"],
        occurred_at=details["occurred_at"], failures=failures,
    )


def direct_failure_reply(runtime, context_id: str, parent_operation_id: str,
                         governor: Any, conversation_id: str | None = None
                         ) -> str | None | bool:
    """Return a terminal reply only for a validated persisted direct-save call.

    ``False`` means no direct-save call is evidenced, ``None`` means a proven
    direct save is retryable or its outcome is still unknown.
    """
    from core.operations import registry
    from core.integrations.firestore.operations import get_operation, get_stage
    from services.connectors.telegram.errors import TELEGRAM_SAVE_INPUT_REFUSAL
    from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id

    if not isinstance(conversation_id, str) or not conversation_id:
        return False
    ledger = FirestoreAgentLedger(
        runtime.repository.client, context_id=context_id, operation_id=parent_operation_id,
        owner="direct-save-read", fence=0, clock=runtime.clock,
    )
    turn = ledger.load_turn(conversation_id, parent_operation_id)
    if not isinstance(turn, Mapping):
        return False
    calls = turn.get("tool_calls")
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, Mapping) or call.get("name") != "add_dialog":
            continue
        ordinal, arguments = call.get("ordinal"), call.get("arguments")
        if type(ordinal) is not int or not isinstance(arguments, Mapping):
            continue
        try:
            validated = registry.validate_arguments("add_dialog", dict(arguments))
        except Exception:
            continue
        child_id = direct_add_dialog_operation_id(parent_operation_id, ordinal, validated)
        checkpoint = (runtime.repository.client.collection("telegram_sessions").document(context_id)
                      .collection("operations").document(parent_operation_id)
                      .collection("agent_checkpoints").document(f"tool-{ordinal}").get()
                      .to_dict() or {})
        persisted_args = ledger._load_payload(checkpoint.get("tool_arguments"))
        serialized = json.dumps(validated, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False)
        input_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if (checkpoint.get("tool_name") != "add_dialog"
                or checkpoint.get("call_ordinal") != ordinal
                or persisted_args != validated
                or checkpoint.get("tool_input_digest") != input_digest):
            return None
        # Pure domain validation is safe to repeat against this server-held,
        # registry-validated call. It catches refusal before a child operation
        # exists, while SDK/storage failures remain governed by durable state.
        from core.dialogs.models import normalize_dialog_input
        try:
            normalize_dialog_input(validated, accepted_at=runtime.clock())
        except ValueError:
            return TELEGRAM_SAVE_INPUT_REFUSAL
        direct_stage_key = f"telegram.direct-classification.{ordinal}"
        try:
            stage = governor.get_stage(parent_operation_id, direct_stage_key) if governor else None
        except Exception:
            return None
        stage_error = _get(stage, "last_error")
        stage_failures = _get(stage, "failures")
        stage_state = getattr(_get(stage, "state"), "value", _get(stage, "state"))
        child = None
        if stage_error is not None:
            details = _classification_failure_details(stage_error, stage_failures)
            if details is None:
                return None
        else:
            child = get_operation(runtime.repository.client, child_id)
            if child is None:
                continue
            input_ref = (f"telegram_sessions/{context_id}/operations/{parent_operation_id}"
                         f"/agent_checkpoints/tool-{ordinal}")
            child_data = (runtime.repository.client.collection("dialog_operations")
                          .document(child_id).get().to_dict() or {})
            if child_data.get("origin_binding") != {
                    "context_id": context_id, "draft_id": input_ref,
                    "manifest_digest": input_digest,
            }:
                return None
            if (child.origin is not OperationOrigin.TELEGRAM
                    or child.origin_operation_ref != input_ref
                    or child.caller_binding != f"telegram:{context_id}:{parent_operation_id}"):
                continue
            error = child.error
            try:
                core_stage = get_stage(runtime.repository.client, child_id, child.stage) \
                    if child.stage else None
            except Exception:
                return None
            core_error = _get(core_stage, "last_error")
            core_failures = _get(core_stage, "failures")
            core_terminal = (core_error is not None and (
                not _get(core_error, "retryable", True)
                or (type(core_failures) is int and core_failures >= 3)
            ))
            if child.state in {OperationState.PENDING, OperationState.RUNNING}:
                return None
            if child.state is not OperationState.FAILED:
                continue
            if governor is not None:
                try:
                    finalizer_error = _finalization_failure_details(
                        governor, parent_operation_id, child_id, child,
                    )
                except Exception:
                    return None
                if finalizer_error is not None:
                    return _format_direct_failure(
                        parent_operation_id, context_id, child_id, child, finalizer_error,
                    )
            if error is None and not core_terminal:
                return None
            # Stage ledger counts, rather than the operation error's retryable
            # flag, decide whether a transient child failure is exhausted.
            if not core_terminal and _get(error, "retryable", True) and not (
                    type(core_failures) is int and core_failures >= 3):
                return None
            source_error = core_error or error
            details = {
                "service": _get(source_error, "service") or _get(error, "service"),
                "stage": _get(source_error, "stage") or _get(error, "stage") or child.stage,
                "code": _get(source_error, "code") or _get(error, "code"),
                "category": _get(source_error, "category") or _get(error, "category"),
                "failures": core_failures if type(core_failures) is int
                else _get(error, "failures"),
                "occurred_at": _get(source_error, "occurred_at") or _get(error, "occurred_at"),
                "sanitized_message": _get(source_error, "sanitized_message")
                or _get(source_error, "message") or _get(error, "sanitized_message")
                or _get(error, "message"),
            }
            if not details["service"]:
                details["service"] = {
                    "fill": "google_docs", "create": "google_drive",
                }.get(details["stage"], "core")
        failures = details["failures"]
        if type(failures) is not int or failures < 1:
            return None
        return _format_direct_failure(
            parent_operation_id, context_id, child_id, child, details,
        )
    return False



def _completed_tool_provenance(
    runtime: Any, operation_id: str, context_id: str, parent_operation_id: str,
    call_ordinal: int, lease: Any, *, input_ref: str, input_digest: str,
    arguments: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Verify parent checkpoint, child identity, and normalized accepted metadata."""
    from core.integrations.firestore.operations import get_operation
    from core.operations import registry
    from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id

    operation = get_operation(runtime.repository.client, operation_id)
    if operation is None or operation.state is not OperationState.COMPLETED:
        raise TelegramSavePending(stage="indexing")
    if not operation.doc_id or not operation.title:
        raise RuntimeError("completed dialog result is unavailable")
    data = runtime.repository.client.collection("dialog_operations").document(operation_id).get().to_dict() or {}
    checkpoint_ref = (runtime.repository.client.collection("telegram_sessions")
                      .document(context_id).collection("operations")
                      .document(parent_operation_id).collection("agent_checkpoints")
                      .document(f"tool-{int(call_ordinal)}"))
    checkpoint = checkpoint_ref.get().to_dict() or {}
    ledger = runtime._ledger(context_id, parent_operation_id, lease)
    stored_arguments = ledger._load_payload(checkpoint.get("tool_arguments"))
    if (checkpoint.get("tool_name") != "add_dialog"
            or checkpoint.get("call_ordinal") != call_ordinal
            or checkpoint.get("tool_input_digest") != input_digest
            or stored_arguments != dict(arguments)
            or data.get("origin_binding") != {
                "context_id": context_id, "draft_id": input_ref,
                "manifest_digest": input_digest,
            }
            or operation.origin_operation_ref != input_ref
            or operation.caller_binding != f"telegram:{context_id}:{parent_operation_id}"):
        raise PermissionError("completed direct save does not match its parent tool input")
    validated = registry.validate_arguments("add_dialog", dict(arguments))
    serialized = json.dumps(validated, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False)
    if (hashlib.sha256(serialized.encode("utf-8")).hexdigest() != input_digest
            or direct_add_dialog_operation_id(
                parent_operation_id, call_ordinal, validated,
            ) != operation_id):
        raise PermissionError("completed direct save input does not match its child operation")
    normalized = _normalize_completed_direct_arguments(
        validated, operation_date=operation.date, accepted_at=operation.created_at,
    )
    if (operation.provider != normalized.provider
            or operation.title != normalized.title
            or operation.date != normalized.dialog_date
            or (normalized.folder_path is not None
                and (operation.selected_folder_path is None
                     or tuple(operation.selected_folder_path)
                     != tuple(normalized.folder_path)))):
        raise PermissionError("completed direct save metadata does not match its tool call")
    return operation, data, validated


def _verified_completion_certificate(
    runtime: Any, operation_id: str, operation: Any, data: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Require matching shared intent, cleanup grant, and actual index receipt."""
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION, get_save_intent
    from services.connectors.telegram.app.terminal_notifications import (
        _completion_receipt_matches_index,
    )

    client = runtime.repository.client
    intent = get_save_intent(client, operation.save_intent_id)
    shared = (client.collection(SAVE_INTENTS_COLLECTION).document(
        operation.save_intent_id,
    ).get().to_dict() or {})
    certificate = shared.get("completion_certificate") or {}
    origin_binding = data.get("origin_binding")
    cleanup_authorization = {
        "certificate_id": certificate.get("certificate_id"),
        "save_intent_id": operation.save_intent_id,
        "operation_id": operation_id,
        "origin_binding": origin_binding,
    }
    result = operation.result
    if (intent is None or certificate.get("operation_id") != operation_id
            or certificate.get("save_intent_id") != operation.save_intent_id
            or certificate.get("doc_id") != operation.doc_id
            or certificate.get("provider") != operation.provider
            or certificate.get("receipt_ref") != operation.index_completion_ref
            or certificate.get("content_hash") != operation.desired_content_hash
            or result is None or result.document_id != operation.doc_id
            or intent.doc_id != operation.doc_id
            or certificate.get("origin_binding") != origin_binding
            or not isinstance(certificate.get("certificate_id"), str)
            or not certificate.get("certificate_id")
            or shared.get("cleanup_authorization") != cleanup_authorization
            or data.get("cleanup_authorization") != cleanup_authorization
            or not _completion_receipt_matches_index(client, certificate)):
        return None
    return certificate


def _completed_dialog_result(
    runtime, operation_id: str, context_id: str, parent_operation_id: str,
    call_ordinal: int, lease: Any, *, input_ref: str, input_digest: str,
    arguments: Mapping[str, Any], classification_decision: Mapping[str, Any] | None = None,
    log_completion: bool = True,
) -> dict[str, Any]:
    operation, data, _validated = _completed_tool_provenance(
        runtime, operation_id, context_id, parent_operation_id, call_ordinal, lease,
        input_ref=input_ref, input_digest=input_digest, arguments=arguments,
    )
    if _verified_completion_certificate(runtime, operation_id, operation, data) is None:
        raise TelegramSavePending(stage="indexing")
    folder_id = data.get("selected_folder_id")
    folder = None
    if isinstance(folder_id, str):
        from core.integrations.firestore.folders import get_all_folders
        folder = get_all_folders(runtime.repository.client).get(folder_id)
    conversation_id = "telegram-dialog-" + hashlib.sha256(
        operation_id.encode("utf-8"),
    ).hexdigest()
    dialog_number = runtime.repository.bind_document(
        context_id, operation.doc_id, conversation_id,
        metadata={"title": operation.title, "provider": operation.provider,
                  "folder_id": folder_id},
        now=runtime.clock(), update_last_doc=True,
        operation_id=parent_operation_id, owner=_get(lease, "owner"),
        fence=_get(lease, "fence"),
        ensure_document_conversation=True,
    )
    path = operation.selected_folder_path
    folder_path = (
        list(path) if path is not None and all(isinstance(item, str) for item in path)
        else None
    )
    result = {"doc_id": operation.doc_id, "title": operation.title,
            "folder_id": folder_id, "folder": folder.get("name") if folder else None,
            "folder_path": folder_path, "dialog_number": dialog_number,
            "conversation_id": conversation_id, "provider": operation.provider,
            "status": "synced", "indexed": True}
    if classification_decision:
        if classification_decision.get("requires_unassigned"):
            result["notice"] = (
                "No confident folder match was found. The dialog was saved to _unassigned."
            )
        elif classification_decision.get("speaker_candidates_ambiguous"):
            result["notice"] = (
                "Speaker matches were ambiguous. The dialog was saved to the selected folder."
            )
    if log_completion:
        logger.info("Telegram direct save completed", extra={
            "event": "telegram_direct_save_settlement",
            "operation_id": parent_operation_id,
            "child_operation_id": operation_id,
            "context_id": context_id, "doc_id": operation.doc_id,
            "service": "telegram", "stage": "save", "failures": 0,
            "outcome": "completed",
        })
    return result

