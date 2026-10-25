"""Transactional persistence for explicitly numbered failed-draft retries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record_in_transaction,
)
from core.integrations.firestore.operation_queue import register_operation_in_transaction
from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from core.operations.models import OperationOrigin
from services.connectors.telegram.models import TelegramDraftState
from services.connectors.telegram.repository._common import (
    SESSIONS,
    TelegramIntakeResult,
    _aware,
    _document_ref,
    _identifier,
    _integer,
    _receipt_result,
)


@dataclass(slots=True)
class _RetryState:
    context_ref: Any
    context: dict[str, Any]
    draft_ref: Any
    draft: dict[str, Any]
    draft_id: str
    dialog_number: int
    previous_operation_id: str
    intent_id: str
    doc_id: str
    digest: str
    ordered_ids: list[int]
    old_shared: dict[str, Any]
    old_manifest: dict[str, Any]


def _load_retry_state(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    context_id: str,
    dialog_ref: Any,
    dialog_number: int,
) -> _RetryState | None:
    """Read and validate the exact failed draft and its durable save intent."""
    context_snapshot = context_ref.get(transaction=transaction)
    dialog_snapshot = dialog_ref.get(transaction=transaction)
    if not context_snapshot.exists or not dialog_snapshot.exists:
        return None
    context = context_snapshot.to_dict() or {}
    dialog = dialog_snapshot.to_dict() or {}
    draft_id = dialog.get("draft_id")
    if dialog.get("dialog_number") != dialog_number or not isinstance(draft_id, str):
        return None
    try:
        _identifier(draft_id, "draft_id")
    except ValueError:
        return None

    draft_ref = _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    draft_snapshot = draft_ref.get(transaction=transaction)
    if not draft_snapshot.exists:
        return None
    draft = draft_snapshot.to_dict() or {}
    if (draft.get("draft_id") != draft_id
            or draft.get("dialog_number") != dialog_number
            or draft.get("state") != TelegramDraftState.FAILED.value):
        return None

    previous_operation_id = draft.get("active_save_operation_id")
    intent_id = draft.get("immutable_save_intent_id")
    digest = draft.get("manifest_digest")
    manifest_path = draft.get("manifest_ref")
    doc_id = draft.get("doc_id")
    raw_ordered_ids = draft.get("ordered_message_ids")
    if (not all(isinstance(value, str) and value for value in
                (previous_operation_id, intent_id, digest, manifest_path, doc_id))
            or not isinstance(raw_ordered_ids, list) or not raw_ordered_ids):
        return None
    try:
        _identifier(previous_operation_id, "operation_id")
        _identifier(intent_id, "save_intent_id")
        _identifier(doc_id, "doc_id")
        ordered_ids = [_integer(value, "message_id", minimum=1) for value in raw_ordered_ids]
        if len(set(ordered_ids)) != len(ordered_ids):
            return None
        expected_manifest_path = (
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/"
            f"{previous_operation_id}"
        )
        if manifest_path != expected_manifest_path:
            return None
        manifest_ref = _document_ref(client, manifest_path)
    except ValueError:
        return None

    old_operation_ref = context_ref.collection("operations").document(previous_operation_id)
    old_shared_ref = client.collection(OPERATIONS_COLLECTION).document(previous_operation_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    old_operation_snapshot = old_operation_ref.get(transaction=transaction)
    old_shared_snapshot = old_shared_ref.get(transaction=transaction)
    intent_snapshot = intent_ref.get(transaction=transaction)
    manifest_snapshot = manifest_ref.get(transaction=transaction)
    if not all((old_operation_snapshot.exists, old_shared_snapshot.exists,
                intent_snapshot.exists, manifest_snapshot.exists)):
        return None

    old_operation = old_operation_snapshot.to_dict() or {}
    old_shared = old_shared_snapshot.to_dict() or {}
    intent = intent_snapshot.to_dict() or {}
    old_manifest = manifest_snapshot.to_dict() or {}
    if (old_operation.get("state") != "failed"
            or old_shared.get("state") != "failed"
            or old_shared.get("save_intent_id") != intent_id
            or old_shared.get("doc_id") != doc_id
            or intent.get("save_intent_id") != intent_id
            or old_manifest.get("draft_id") != draft_id
            or old_manifest.get("manifest_digest") != digest
            or old_manifest.get("ordered_message_ids") != ordered_ids
            or old_manifest.get("save_operation_id") != previous_operation_id):
        return None
    return _RetryState(
        context_ref=context_ref,
        context=context,
        draft_ref=draft_ref,
        draft=draft,
        draft_id=draft_id,
        dialog_number=dialog_number,
        previous_operation_id=previous_operation_id,
        intent_id=intent_id,
        doc_id=doc_id,
        digest=digest,
        ordered_ids=ordered_ids,
        old_shared=old_shared,
        old_manifest=old_manifest,
    )


def _update_context_queue(
    context: dict[str, Any], *, sequence: int, operation_id: str, now: datetime,
) -> None:
    """Record the retry at the FIFO tail while retaining the current dialog selection."""
    context.update({
        "next_operation_sequence": sequence + 1,
        "queue_tail_operation_id": operation_id,
        "updated_at": now,
    })
    if not context.get("reserved_operation_id"):
        context.update({"head_sequence": sequence, "reserved_operation_id": operation_id})


def _persist_retry(
    client: Any,
    transaction: Any,
    *,
    state: _RetryState,
    context_id: str,
    update_id: int,
    operation_id: str,
    outbox_id: str,
    command_ref: Any,
    receipt_ref: Any,
    now: datetime,
) -> TelegramIntakeResult:
    """Append queue and outbox records, then persist the retry's immutable envelope."""
    retry_manifest_ref = state.draft_ref.collection("save_manifests").document(operation_id)
    retry_manifest_path = (
        f"{SESSIONS}/{context_id}/drafts/{state.draft_id}/save_manifests/{operation_id}"
    )
    command_path = f"{SESSIONS}/{context_id}/commands/{update_id}"
    command_data = {
        "update_id": update_id, "context_id": context_id, "kind": "retry",
        "dialog_number": state.dialog_number, "draft_id": state.draft_id,
        "operation_id": operation_id, "save_intent_id": state.intent_id,
        "doc_id": state.doc_id, "manifest_digest": state.digest,
        "cutoff_message_ids": list(state.ordered_ids), "created_at": now,
    }
    input_digest = hashlib.sha256(json.dumps(
        command_data, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()

    def create_retry_outbox() -> None:
        create_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1, schedule_at=now, now=now,
            operation_id=operation_id, context_id=context_id,
        )

    queue_operation = register_operation_in_transaction(
        client, transaction, state.context_ref, operation_id,
        identity_digest=input_digest, input_ref=command_path, now=now,
        before_write=create_retry_outbox,
        operation_fields={
            "draft_id": state.draft_id, "dialog_number": state.dialog_number,
            "save_intent_id": state.intent_id, "doc_id": state.doc_id,
        },
    )
    transaction.set(command_ref, command_data)
    shared_ref = client.collection(OPERATIONS_COLLECTION).document(operation_id)
    transaction.set(shared_ref, _retry_operation_data(
        operation_id=operation_id, context_id=context_id, draft_id=state.draft_id,
        intent_id=state.intent_id, doc_id=state.doc_id, digest=state.digest,
        manifest_path=retry_manifest_path, now=now, previous=state.old_shared,
    ))
    transaction.set(retry_manifest_ref, {
        **state.old_manifest, "save_operation_id": operation_id, "sealed_at": now,
    })
    state.draft.update({
        "state": TelegramDraftState.PROCESSING.value,
        "active_save_operation_id": operation_id,
        "manifest_ref": retry_manifest_path,
        "updated_at": now,
    })
    transaction.set(state.draft_ref, state.draft)
    _update_context_queue(
        state.context, sequence=queue_operation.sequence, operation_id=operation_id, now=now,
    )
    transaction.set(state.context_ref, state.context)
    receipt = {
        "update_id": update_id, "context_id": context_id, "accepted": True,
        "draft_id": state.draft_id, "dialog_number": state.dialog_number,
        "message_id": None, "intake_sequence": None,
        "notification_id": None, "outbox_id": outbox_id,
        "operation_id": operation_id, "queue_position": queue_operation.sequence,
        "snapshot_message_ids": list(state.ordered_ids), "created_at": now,
    }
    transaction.set(receipt_ref, receipt)
    return TelegramIntakeResult(
        True, context_id, state.draft_id, state.dialog_number, outbox_id=outbox_id,
        operation_id=operation_id, queue_position=queue_operation.sequence,
        snapshot_message_ids=tuple(state.ordered_ids),
    )


def _retry_operation_data(
    *, operation_id: str, context_id: str, draft_id: str, intent_id: str,
    doc_id: str, digest: str, manifest_path: str, now: datetime,
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Initialize a clean shared operation while carrying immutable save identity."""
    return {
        "operation_id": operation_id,
        "origin": OperationOrigin.TELEGRAM.value,
        "origin_operation_ref": f"{SESSIONS}/{context_id}/operations/{operation_id}",
        "caller_binding": f"telegram:{context_id}",
        "immutable_input_digest": digest,
        "immutable_input_ref": manifest_path,
        "provider": "telegram",
        "title": previous.get("title"),
        "date": previous.get("date"),
        "explicit_folder_id": previous.get("explicit_folder_id"),
        "selected_folder_id": previous.get("selected_folder_id"),
        "selected_folder_path": previous.get("selected_folder_path"),
        "state": "pending",
        "stage": None,
        "stage_failure_counts": [],
        "save_intent_id": intent_id,
        "doc_id": doc_id,
        "create_outcome": None,
        "filled_revision": None,
        "desired_content_hash": previous.get("desired_content_hash"),
        "desired_parent_id_hash": previous.get("desired_parent_id_hash"),
        "expected_provider": "telegram",
        "index_completion_ref": None,
        "result": None,
        "error": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "fence": 0,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "status_updated_at": now,
        "attempts_total": 0,
        "retry_of_operation_id": previous.get("operation_id"),
        "retry_draft_id": draft_id,
    }


class _TelegramRetryMixin:
    """Append a retry only after validating its failed-draft save evidence."""

    def retry_failed_draft(
        self,
        context_id: str,
        dialog_number: int,
        *,
        update_id: int,
        now: datetime,
    ) -> TelegramIntakeResult:
        context_id = _identifier(context_id, "context_id")
        dialog_number = _integer(dialog_number, "dialog_number", minimum=1)
        update_id = _integer(update_id, "update_id")
        now = _aware(now, "now")
        identity = hashlib.sha256(f"{context_id}:{update_id}:retry".encode()).hexdigest()
        operation_id = "tgretry-" + identity
        outbox_id = "tgretrytask-" + identity
        context_ref = self.client.collection(SESSIONS).document(context_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        dialog_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/dialogs/{dialog_number}")
        command_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/commands/{update_id}")

        @firestore.transactional
        def _txn(transaction: Any) -> TelegramIntakeResult:
            receipt_snapshot = receipt_ref.get(transaction=transaction)
            if receipt_snapshot.exists:
                return _receipt_result(receipt_snapshot.to_dict() or {}, context_id)
            state = _load_retry_state(
                self.client, transaction, context_ref=context_ref, context_id=context_id,
                dialog_ref=dialog_ref, dialog_number=dialog_number,
            )
            if state is None:
                return TelegramIntakeResult(False, context_id)
            return _persist_retry(
                self.client, transaction, state=state, context_id=context_id,
                update_id=update_id, operation_id=operation_id, outbox_id=outbox_id,
                command_ref=command_ref, receipt_ref=receipt_ref, now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))


__all__ = ["_TelegramRetryMixin"]
