"""Verified hydration helpers for private Telegram context annotations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from core.operations.queue import QueueOperationState
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from services.connectors.telegram.repository._annotation_payloads import load_annotation_row_text


def hydrate_annotation_text(
    client: Any, row: Mapping[str, Any], *, context_id: str,
    operation_id: str, owner: str, fence: int, clock: Any,
) -> str | None:
    """Load a source-bound note while the current caller still owns its queue fence."""
    if (row.get("context_id") != context_id or not isinstance(operation_id, str)
            or not operation_id or not isinstance(owner, str) or not owner
            or type(fence) is not int or fence < 1 or not callable(clock)):
        return None
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_snapshot = context_ref.get()
    context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
    if (not context_snapshot.exists or not validate_context_queue_lease(
            context, operation_id, owner=owner, fence=fence, now=clock(),
            expiry=context.get("lock_expires_at"))):
        return None
    text = load_annotation_row_text(client, row)
    expected = row.get("text_sha256")
    if (not isinstance(text, str) or not text.strip()
            or not isinstance(expected, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected):
        return None
    return text


async def _fail_application(
    processor: Any, *, context_id: str, operation_id: str, lease: Any,
    draft_id: str, comment_id: str, doc_id: str, dialog_number: Any,
    error_code: str,
) -> Any:
    """Retain a failed note and publish one truthful annotation failure notice."""
    from services.connectors.telegram.runtime.runtime import _offload, _value

    settled = await _offload(
        processor.repository.settle_draft_annotation,
        context_id, operation_id, owner=_value(lease, "owner"),
        fence=_value(lease, "fence"), draft_id=draft_id,
        comment_id=comment_id, doc_id=doc_id, state="failed",
        now=processor.clock(), error_code=error_code,
    )
    if not settled:
        return QueueOperationState.RUNNING
    return await _deliver_failure_notice(
        processor, context_id=context_id, operation_id=operation_id,
        comment_id=comment_id, dialog_number=dialog_number,
    )


async def _deliver_failure_notice(
    processor: Any, *, context_id: str, operation_id: str,
    comment_id: str, dialog_number: Any,
) -> Any:
    """Replay one stable error notice until the canonical publisher settles it."""
    from services.connectors.telegram.runtime.runtime import _offload
    from services.connectors.telegram.notifications import TelegramNotificationPublisher

    notification_id = "tgannotation-error-" + hashlib.sha256(
        comment_id.encode("utf-8")
    ).hexdigest()
    await _offload(
        processor.repository.create_notification, notification_id,
        context_id=context_id, kind="error",
        body="The dialog was saved and indexed, but its context note could not be added. The note remains stored with the dialog.",
        dialog_number=dialog_number, operation_id=operation_id,
        now=processor.clock(),
    )
    publisher = TelegramNotificationPublisher(
        processor.repository, processor._client(), clock=processor.clock,
    )
    disposition = await publisher.publish(notification_id)
    if disposition == "delivered":
        return QueueOperationState.COMPLETED
    if disposition in {"retry", "busy", "pending"}:
        return QueueOperationState.RUNNING
    return QueueOperationState.FAILED


async def _load_application_claim(
    processor: Any, context_id: str, operation: Any, lease: Any,
) -> dict[str, Any] | None:
    """Load and fence the stable queue operation and annotation row."""
    from services.connectors.telegram.runtime.runtime import _offload, _value
    operation_id = _value(operation, "operation_id")
    input_ref = _value(operation, "input_ref")
    parts = input_ref.split("/") if isinstance(input_ref, str) else []
    if (not isinstance(operation_id, str) or len(parts) != 6
            or parts[:2] != ["telegram_sessions", context_id]
            or parts[2] != "drafts" or parts[4] != "annotations"
            or not all(parts[index] for index in (3, 5))):
        return None
    draft_id, comment_id = parts[3], parts[5]
    client = processor.repository.client
    context_ref = client.collection("telegram_sessions").document(context_id)
    operation_ref = context_ref.collection("operations").document(operation_id)
    comment_ref = (context_ref.collection("drafts").document(draft_id)
                   .collection("annotations").document(comment_id))
    context_snapshot, operation_snapshot, comment_snapshot = (
        await _offload(context_ref.get), await _offload(operation_ref.get),
        await _offload(comment_ref.get),
    )
    context_data = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
    operation_data = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
    comment = comment_snapshot.to_dict() or {} if comment_snapshot.exists else {}
    if (not operation_snapshot.exists or not comment_snapshot.exists
            or operation_data.get("operation_id") != operation_id
            or operation_data.get("input_ref") != input_ref
            or operation_data.get("telegram_internal_kind") != "annotation_apply"
            or operation_data.get("annotation_id") != comment_id
            or operation_data.get("draft_id") != draft_id
            or operation_data.get("state") != "running"
            or comment.get("application_operation_id") != operation_id
            or comment.get("comment_id") != comment_id
            or comment.get("draft_id") != draft_id
            or comment.get("state") not in {"queued_fact", "failed", "applied"}
            or not validate_context_queue_lease(
                context_data, operation_id, owner=_value(lease, "owner"),
                fence=_value(lease, "fence"), now=processor.clock(),
                expiry=context_data.get("lock_expires_at"))):
        return None
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft_snapshot = await _offload(draft_ref.get)
    if not draft_snapshot.exists:
        return None
    return {
        "operation_id": operation_id, "draft_id": draft_id,
        "comment_id": comment_id, "operation_data": operation_data,
        "comment": comment, "draft_ref": draft_ref,
        "draft": draft_snapshot.to_dict() or {},
    }


async def _load_completion_proof(
    processor: Any, context_id: str, claim: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Load the exact saved document authorization and its completed index receipt."""
    from services.connectors.telegram.runtime.runtime import _offload
    from services.connectors.telegram.repository._annotation_proof import (
        load_annotation_completion_proof,
    )
    operation_data = claim["operation_data"]
    draft = claim["draft"]
    doc_id = operation_data.get("target_doc_id")
    proof = await _offload(
        load_annotation_completion_proof, processor.repository.client,
        context_id, claim["draft_id"], draft, expected_doc_id=doc_id,
    )
    if (not isinstance(proof, Mapping)
            or proof.get("operation_id") != draft.get("active_save_operation_id")
            or operation_data.get("completion_receipt_ref") != proof.get("receipt_ref")
            or claim["comment"].get("applied_doc_id") != doc_id):
        return None, None
    return doc_id, proof


async def _read_index_state(processor: Any, doc_id: str) -> tuple[Mapping[str, Any] | None, bool]:
    """Read current index readiness; read errors become a retained-note failure."""
    from services.connectors.telegram.runtime.runtime import _offload
    from core.integrations.firestore.transcripts import get_transcript_record
    try:
        record = await _offload(get_transcript_record, doc_id, processor.repository.client)
    except Exception:
        return None, True
    return (record if isinstance(record, Mapping) else None), False


async def _apply_canonical_fact(
    processor: Any, context_id: str, operation_id: str, lease: Any,
    doc_id: str, text: str, retry_governor: Any, *, require_existing: bool = False,
) -> Any:
    """Apply one exact note through the shared governed add_fact executor."""
    from services.connectors.telegram.runtime.runtime import _offload, _value
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor
    client = processor.repository.client
    execution_context = {
        "context_id": context_id, "operation_id": operation_id,
        "owner": _value(lease, "owner"), "fence": _value(lease, "fence"),
        "lease": lease, "clock": processor.clock,
        "target_doc_id": doc_id,
    }
    executor = TelegramRegistryExecutor(
        client=client, context=execution_context,
        save_handler=processor.save_handler, execution_context=execution_context,
    )
    arguments = {"doc_id": doc_id, "text": text}
    reconcile_fact = lambda: executor.reconcile(
        name="add_fact", arguments=arguments, call_ordinal=0,
    )
    if require_existing:
        existing = await _offload(reconcile_fact)
        if existing != doc_id:
            return None
    execute_fact = lambda: executor.execute(
        name="add_fact", arguments=arguments, call_ordinal=0,
    )
    if retry_governor is None:
        return existing if require_existing else await _offload(execute_fact)
    return await _offload(
        retry_governor.run_stage, operation_id, "telegram_annotation_fact",
        execute_fact, now=processor.clock(), owner=_value(lease, "owner"),
        fence=_value(lease, "fence"),
        reconcile=reconcile_fact,
    )


async def _settle_application(
    processor: Any, context_id: str, operation_id: str, lease: Any,
    draft_id: str, comment_id: str, doc_id: str, state: str,
) -> Any:
    """Persist the note's application disposition while the fact lease is live."""
    from services.connectors.telegram.runtime.runtime import _offload, _value
    settled = await _offload(
        processor.repository.settle_draft_annotation,
        context_id, operation_id, owner=_value(lease, "owner"),
        fence=_value(lease, "fence"), draft_id=draft_id,
        comment_id=comment_id, doc_id=doc_id, state=state, now=processor.clock(),
    )
    return QueueOperationState.COMPLETED if settled else QueueOperationState.RUNNING


async def apply_annotation_operation(
    processor: Any, context_id: str, operation: Any, lease: Any, *, retry_governor: Any,
) -> Any:
    """Wait for verified index readiness, apply the note, then settle its state."""
    from services.connectors.telegram.runtime.runtime import _offload, _value
    claim = await _load_application_claim(processor, context_id, operation, lease)
    if claim is None:
        return QueueOperationState.FAILED
    operation_id, draft_id, comment_id = (
        claim["operation_id"], claim["draft_id"], claim["comment_id"],
    )
    comment, draft = claim["comment"], claim["draft"]
    client = processor.repository.client
    doc_id, proof = await _load_completion_proof(processor, context_id, claim)
    if proof is None or not isinstance(doc_id, str):
        return QueueOperationState.FAILED
    if comment.get("state") == "failed":
        if not isinstance(comment.get("error_code"), str):
            return QueueOperationState.FAILED
        return await _deliver_failure_notice(
            processor, context_id=context_id, operation_id=operation_id,
            comment_id=comment_id, dialog_number=draft.get("dialog_number"),
        )
    if comment.get("state") == "applied":
        text = await _offload(
            hydrate_annotation_text, client, comment, context_id=context_id,
            operation_id=operation_id, owner=_value(lease, "owner"),
            fence=_value(lease, "fence"), clock=processor.clock,
        )
        if text is None:
            return QueueOperationState.FAILED
        try:
            result = await _apply_canonical_fact(
                processor, context_id, operation_id, lease, doc_id, text,
                retry_governor, require_existing=True,
            )
        except Exception as error:
            retryable = getattr(error, "retryable_stage", None)
            if retryable is True or type(error).__name__ == "UnknownOutcome":
                raise
            return QueueOperationState.RUNNING
        return (QueueOperationState.COMPLETED if result == doc_id
                else QueueOperationState.RUNNING)
    record, read_failed = await _read_index_state(processor, doc_id)
    if read_failed:
        return await _fail_application(
            processor, context_id=context_id, operation_id=operation_id, lease=lease,
            draft_id=draft_id, comment_id=comment_id, doc_id=doc_id,
            dialog_number=draft.get("dialog_number"), error_code="index_read_failed",
        )
    if isinstance(record, Mapping) and record.get("status") in {
            "queued", "downloading", "pending", "running", "syncing"}:
        return QueueOperationState.RUNNING
    if not isinstance(record, Mapping) or record.get("status") != "synced":
        return await _fail_application(
            processor, context_id=context_id, operation_id=operation_id, lease=lease,
            draft_id=draft_id, comment_id=comment_id, doc_id=doc_id,
            dialog_number=draft.get("dialog_number"), error_code="index_unavailable",
        )
    text = await _offload(
        hydrate_annotation_text, client, comment, context_id=context_id,
        operation_id=operation_id, owner=_value(lease, "owner"),
        fence=_value(lease, "fence"), clock=processor.clock,
    )
    if text is None:
        return await _fail_application(
            processor, context_id=context_id, operation_id=operation_id, lease=lease,
            draft_id=draft_id, comment_id=comment_id, doc_id=doc_id,
            dialog_number=draft.get("dialog_number"), error_code="annotation_payload_invalid",
        )
    try:
        result = await _apply_canonical_fact(
            processor, context_id, operation_id, lease, doc_id, text, retry_governor,
        )
    except Exception as error:
        retryable = getattr(error, "retryable_stage", None)
        if retryable is True or type(error).__name__ == "UnknownOutcome":
            raise
        return await _fail_application(
            processor, context_id=context_id, operation_id=operation_id, lease=lease,
            draft_id=draft_id, comment_id=comment_id, doc_id=doc_id,
            dialog_number=draft.get("dialog_number"),
            error_code="fact_application_failed",
        )
    if result != doc_id:
        return await _fail_application(
            processor, context_id=context_id, operation_id=operation_id, lease=lease,
            draft_id=draft_id, comment_id=comment_id, doc_id=doc_id,
            dialog_number=draft.get("dialog_number"), error_code="fact_result_invalid",
        )
    return await _settle_application(
        processor, context_id, operation_id, lease,
        draft_id, comment_id, doc_id, "applied",
    )
