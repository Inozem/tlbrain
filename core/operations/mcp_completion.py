"""Immediate MCP completion dispatch after a verified managed index run."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from core.integrations.firestore.outbox import OutboxRecord, OutboxTargetKind, get_outbox_record
from core.integrations.firestore.operations import get_operation, get_stage
from core.integrations.firestore.save_intents import (
    SaveIntentRecord,
    SaveIntentState,
    get_save_intent,
)
from core.indexing.hashing import sha256_text
from core.indexing.processor import verify_current_generation
from core.operations.models import OperationOrigin, OperationRecord, StageState
from core.operations.outbox import (
    OutboxDispatchResult,
    TaskTargetConfig,
    create_mcp_save_recovery_record,
    publish_outbox_record,
)


_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_REFERENCE_FIELDS = {
    "schema_version", "outbox_id", "target_kind", "generation",
    "operation_id", "document_id", "save_intent_id",
}


def dispatch_completed_managed_mcp_index(
    client: Any,
    reference: object,
    *,
    doc_id: str,
    parent_id: str,
    root_folder_id: str,
    raw_text: str,
    target: TaskTargetConfig | None,
    now: datetime,
) -> OutboxDispatchResult | None:
    """Validate a current managed-index receipt, persist recovery, and publish it."""
    binding = _managed_index_binding(
        client, reference, doc_id=doc_id, parent_id=parent_id, raw_text=raw_text,
    )
    if binding is None:
        return None
    operation, intent, index_dispatch = binding
    if not _completed_index_receipt(
        client, operation, intent, index_dispatch, doc_id=doc_id,
        parent_id=parent_id, root_folder_id=root_folder_id, raw_text=raw_text,
    ):
        return None
    recovery = create_mcp_save_recovery_record(
        client, operation, now=now, completed_index_dispatch=index_dispatch,
    )
    return publish_outbox_record(client, recovery, target, now=now)


def _managed_index_binding(
    client: Any,
    reference: object,
    *,
    doc_id: str,
    parent_id: str,
    raw_text: str,
) -> tuple[OperationRecord, SaveIntentRecord, OutboxRecord] | None:
    """Resolve canonical MCP operation, save-intent, and index-dispatch records."""
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_FIELDS:
        return None
    if (type(reference.get("schema_version")) is not int or reference["schema_version"] != 1
            or type(reference.get("generation")) is not int or reference["generation"] < 1
            or reference.get("target_kind") != OutboxTargetKind.INDEX_DISPATCH.value
            or reference.get("document_id") != doc_id):
        return None
    if any(not isinstance(reference.get(field), str) or not _TASK_ID.fullmatch(reference[field])
           for field in ("outbox_id", "operation_id", "save_intent_id")):
        return None

    operation = get_operation(client, reference["operation_id"])
    intent = get_save_intent(client, reference["save_intent_id"])
    index_dispatch = get_outbox_record(client, reference["outbox_id"])
    if operation is None or intent is None or index_dispatch is None:
        return None
    if not _managed_index_bindings_match(
        operation, intent, index_dispatch, reference,
        doc_id=doc_id, parent_id=parent_id, raw_text=raw_text,
    ):
        return None
    return operation, intent, index_dispatch


def _managed_index_bindings_match(
    operation: OperationRecord,
    intent: SaveIntentRecord,
    index_dispatch: OutboxRecord,
    reference: dict,
    *,
    doc_id: str,
    parent_id: str,
    raw_text: str,
) -> bool:
    """Check durable operation, intent, payload, and source-dispatch identity."""
    parent_hash = sha256_text(parent_id)
    content_hash = sha256_text(raw_text)
    return all((
        operation.origin is OperationOrigin.MCP,
        operation.state.value in {"pending", "running"},
        operation.save_intent_id == intent.save_intent_id,
        intent.dispatch_operation_id == operation.operation_id,
        operation.doc_id in (None, doc_id),
        intent.doc_id == doc_id,
        intent.state is SaveIntentState.CONFIRMED,
        intent.index_checkpoint_ref == reference["outbox_id"],
        bool(operation.immutable_input_ref),
        bool(intent.fill_checkpoint_ref),
        bool(intent.index_checkpoint_ref),
        operation.desired_content_hash == content_hash == intent.desired_content_hash,
        intent.desired_parent_id_hash == parent_hash,
        index_dispatch.target_kind is OutboxTargetKind.INDEX_DISPATCH,
        index_dispatch.generation == reference["generation"],
        index_dispatch.operation_id == operation.operation_id,
        index_dispatch.document_id == doc_id,
        index_dispatch.save_intent_id == intent.save_intent_id,
    ))


def _completed_index_receipt(
    client: Any,
    operation: OperationRecord,
    intent: SaveIntentRecord,
    index_dispatch: OutboxRecord,
    *,
    doc_id: str,
    parent_id: str,
    root_folder_id: str,
    raw_text: str,
) -> bool:
    """Require the completed managed stage and current-generation receipt proof."""
    from core.integrations.firestore.index_completions import get_completion

    stage = get_stage(client, operation.operation_id, "managed_index")
    if stage is None or stage.state is not StageState.COMPLETED:
        return False
    receipt = get_completion(client, doc_id, sha256_text(raw_text))
    if (receipt is None or receipt.get("status") != "completed"
            or receipt.get("work_manifest_ref") != operation.immutable_input_ref
            or receipt.get("checkpoint_refs", [])[:2]
            != [intent.fill_checkpoint_ref, intent.index_checkpoint_ref]
            or not verify_current_generation(doc_id, parent_id, root_folder_id, raw_text)):
        return False
    return index_dispatch.document_id == doc_id
