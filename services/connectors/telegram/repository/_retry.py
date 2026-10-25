"""Transactional persistence for explicitly numbered failed-draft retries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
    create_gate_operation_id: str
    intent_id: str
    doc_id: str | None
    digest: str
    core_input_digest: str
    core_input_ref: str
    ordered_ids: list[int]
    old_shared: dict[str, Any]
    completed_stages: dict[str, dict[str, Any]]
    old_manifest: dict[str, Any]


@dataclass(slots=True)
class _RetryDraft:
    context: dict[str, Any]
    draft_ref: Any
    draft: dict[str, Any]
    draft_id: str
    dialog_number: int
    previous_operation_id: str
    intent_id: str
    doc_id: str | None
    digest: str
    manifest_ref: Any
    ordered_ids: list[int]


@dataclass(slots=True)
class _RetrySourceRecords:
    old_operation: dict[str, Any]
    old_shared: dict[str, Any]
    intent: dict[str, Any]
    old_manifest: dict[str, Any]


def _retry_intent_binding_is_valid(
    intent: Mapping[str, Any], *, dispatch_operation_ids: tuple[str, ...],
    draft_doc_id: str | None, shared_doc_id: str | None,
) -> bool:
    """Accept only internally consistent create-gate states for an explicit retry."""
    state = intent.get("state")
    outcome = intent.get("create_outcome")
    token = intent.get("dispatch_token")
    dispatch_operation_id = intent.get("dispatch_operation_id")
    dispatch_owner = intent.get("dispatch_owner")
    dispatch_fence = intent.get("dispatch_fence")
    doc_id = intent.get("doc_id")
    result = intent.get("confirmed_result")

    if state == "planned":
        return (
            outcome in {None, "rejected"}
            and token is None and dispatch_operation_id is None
            and dispatch_owner is None and dispatch_fence is None
            and doc_id is None and result is None
            and draft_doc_id is None and shared_doc_id is None
        )
    if state in {"dispatched", "unknown"}:
        expected_outcome = "dispatched" if state == "dispatched" else "unknown"
        return (
            outcome == expected_outcome
            and dispatch_operation_id in dispatch_operation_ids
            and isinstance(token, str) and bool(token)
            and isinstance(dispatch_owner, str) and bool(dispatch_owner)
            and isinstance(dispatch_fence, int) and dispatch_fence > 0
            and doc_id is None and result is None
            and draft_doc_id is None and shared_doc_id is None
        )
    if state == "confirmed":
        return (
            outcome == "confirmed"
            and dispatch_operation_id in dispatch_operation_ids
            and isinstance(token, str) and bool(token)
            and isinstance(dispatch_owner, str) and bool(dispatch_owner)
            and isinstance(dispatch_fence, int) and dispatch_fence > 0
            and isinstance(doc_id, str) and bool(doc_id)
            and (draft_doc_id is None or draft_doc_id == doc_id)
            and (shared_doc_id is None or shared_doc_id == doc_id)
            and isinstance(result, Mapping)
            and result.get("document_id") == doc_id
        )
    return False


def _load_retry_draft(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    context_id: str,
    dialog_ref: Any,
    dialog_number: int,
) -> _RetryDraft | None:
    """Read the context, dialog, and exact failed draft in the caller's transaction."""
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
                (previous_operation_id, intent_id, digest, manifest_path))
            or (doc_id is not None and (not isinstance(doc_id, str) or not doc_id))
            or not isinstance(raw_ordered_ids, list) or not raw_ordered_ids):
        return None

    try:
        _identifier(previous_operation_id, "operation_id")
        _identifier(intent_id, "save_intent_id")
        if doc_id is not None:
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

    return _RetryDraft(
        context=context,
        draft_ref=draft_ref,
        draft=draft,
        draft_id=draft_id,
        dialog_number=dialog_number,
        previous_operation_id=previous_operation_id,
        intent_id=intent_id,
        doc_id=doc_id,
        digest=digest,
        manifest_ref=manifest_ref,
        ordered_ids=ordered_ids,
    )


def _load_retry_source_records(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    previous_operation_id: str,
    intent_id: str,
    manifest_ref: Any,
) -> _RetrySourceRecords | None:
    """Read the failed operation, shared operation, intent, and sealed manifest."""
    old_operation_ref = context_ref.collection("operations").document(previous_operation_id)
    old_shared_ref = client.collection(OPERATIONS_COLLECTION).document(previous_operation_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    snapshots = (
        old_operation_ref.get(transaction=transaction),
        old_shared_ref.get(transaction=transaction),
        intent_ref.get(transaction=transaction),
        manifest_ref.get(transaction=transaction),
    )
    if not all(snapshot.exists for snapshot in snapshots):
        return None
    old_operation_snapshot, old_shared_snapshot, intent_snapshot, manifest_snapshot = snapshots
    return _RetrySourceRecords(
        old_operation=old_operation_snapshot.to_dict() or {},
        old_shared=old_shared_snapshot.to_dict() or {},
        intent=intent_snapshot.to_dict() or {},
        old_manifest=manifest_snapshot.to_dict() or {},
    )


def _retry_create_gate_identity(
    draft_id: str,
    context_id: str,
    previous_operation_id: str,
    digest: str,
    source: _RetrySourceRecords,
) -> tuple[list[str], str]:
    """Keep the original create-gate identity while retries advance their predecessor."""
    old_shared = source.old_shared
    intent = source.intent
    dispatch_operation_ids = [previous_operation_id]
    prior_create_gate_operation_id = previous_operation_id
    retry_authorization = old_shared.get("retry_authorization")
    binding = {
        "context_id": context_id, "draft_id": draft_id, "manifest_digest": digest,
    }
    if (
        isinstance(retry_authorization, Mapping)
        and retry_authorization.get("operation_id") == previous_operation_id
        and retry_authorization.get("previous_operation_id") == old_shared.get("retry_of_operation_id")
        and retry_authorization.get("context_id") == context_id
        and retry_authorization.get("draft_id") == draft_id
        and retry_authorization.get("manifest_digest") == digest
        and retry_authorization.get("caller_binding") == old_shared.get("caller_binding")
        and retry_authorization.get("immutable_input_digest") == old_shared.get("immutable_input_digest")
        and old_shared.get("retry_draft_id") == draft_id
        and old_shared.get("origin_binding") == binding
    ):
        prior_create_gate_operation_id = (
            retry_authorization.get("create_gate_operation_id")
            or retry_authorization["previous_operation_id"]
        )
        dispatch_operation_ids.append(prior_create_gate_operation_id)
    create_gate_operation_id = (
        intent.get("dispatch_operation_id") or prior_create_gate_operation_id
    )
    return dispatch_operation_ids, create_gate_operation_id


def _retry_source_is_valid(
    draft: _RetryDraft,
    source: _RetrySourceRecords,
    *,
    dispatch_operation_ids: list[str],
) -> bool:
    """Validate failed-save lineage, immutable inputs, intent state, and manifest content."""
    old_operation = source.old_operation
    old_shared = source.old_shared
    intent = source.intent
    old_manifest = source.old_manifest
    core_input_digest = old_shared.get("immutable_input_digest")
    core_input_ref = old_shared.get("immutable_input_ref")
    return not (
        old_operation.get("state") != "failed"
        or old_shared.get("state") != "failed"
        or old_shared.get("save_intent_id") != draft.intent_id
        or (draft.doc_id is not None and old_shared.get("doc_id") != draft.doc_id)
        or intent.get("save_intent_id") != draft.intent_id
        or not _retry_intent_binding_is_valid(
            intent,
            dispatch_operation_ids=tuple(dispatch_operation_ids),
            draft_doc_id=draft.doc_id,
            shared_doc_id=old_shared.get("doc_id"),
        )
        or not isinstance(core_input_digest, str)
        or not core_input_digest
        or not isinstance(core_input_ref, str)
        or not core_input_ref
        or intent.get("immutable_input_digest") != core_input_digest
        or old_manifest.get("draft_id") != draft.draft_id
        or old_manifest.get("manifest_digest") != draft.digest
        or old_manifest.get("ordered_message_ids") != draft.ordered_ids
        or old_manifest.get("save_operation_id") != draft.previous_operation_id
    )


def _load_completed_retry_stages(
    transaction: Any,
    *,
    context_ref: Any,
    previous_operation_id: str,
) -> dict[str, dict[str, Any]]:
    """Load completed model checkpoints that are safe to carry into a manual retry.

    Audio transcription is already persisted on each source message, so this
    reads only completed preparation, analysis, and classification stages to
    avoid rerunning those model calls.
    """
    stage_keys = ["telegram.preparation", "telegram.analysis", "telegram.classification"]
    completed_stages: dict[str, dict[str, Any]] = {}
    old_queue_operation_ref = context_ref.collection("operations").document(previous_operation_id)
    for stage_key in stage_keys:
        stage_snapshot = old_queue_operation_ref.collection("stages").document(stage_key).get(
            transaction=transaction,
        )
        if stage_snapshot.exists:
            stage = stage_snapshot.to_dict() or {}
            if stage.get("stage_key") == stage_key and stage.get("state") == "completed":
                completed_stages[stage_key] = stage
    return completed_stages


def _build_retry_state(
    context_ref: Any,
    draft: _RetryDraft,
    source: _RetrySourceRecords,
    *,
    create_gate_operation_id: str,
    completed_stages: dict[str, dict[str, Any]],
    core_input_digest: str,
    core_input_ref: str,
) -> _RetryState:
    """Assemble the validated retry envelope and its confirmed checkpoint."""
    intent = source.intent
    return _RetryState(
        context_ref=context_ref,
        context=draft.context,
        draft_ref=draft.draft_ref,
        draft=draft.draft,
        draft_id=draft.draft_id,
        dialog_number=draft.dialog_number,
        previous_operation_id=draft.previous_operation_id,
        create_gate_operation_id=create_gate_operation_id,
        intent_id=draft.intent_id,
        doc_id=intent.get("doc_id") if intent.get("state") == "confirmed" else draft.doc_id,
        digest=draft.digest,
        core_input_digest=core_input_digest,
        core_input_ref=core_input_ref,
        ordered_ids=draft.ordered_ids,
        old_shared=source.old_shared,
        completed_stages=completed_stages,
        old_manifest=source.old_manifest,
    )


def _load_retry_state(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    context_id: str,
    dialog_ref: Any,
    dialog_number: int,
) -> _RetryState | None:
    """Load and validate the exact failed draft and its durable save evidence."""
    draft = _load_retry_draft(
        client, transaction, context_ref=context_ref, context_id=context_id,
        dialog_ref=dialog_ref, dialog_number=dialog_number,
    )
    if draft is None:
        return None

    source = _load_retry_source_records(
        client, transaction, context_ref=context_ref,
        previous_operation_id=draft.previous_operation_id,
        intent_id=draft.intent_id, manifest_ref=draft.manifest_ref,
    )
    if source is None:
        return None

    dispatch_operation_ids, create_gate_operation_id = _retry_create_gate_identity(
        draft.draft_id, context_id, draft.previous_operation_id, draft.digest, source,
    )
    if not _retry_source_is_valid(
        draft, source, dispatch_operation_ids=dispatch_operation_ids,
    ):
        return None

    core_input_digest = source.old_shared["immutable_input_digest"]
    core_input_ref = source.old_shared["immutable_input_ref"]
    completed_stages = _load_completed_retry_stages(
        transaction, context_ref=context_ref,
        previous_operation_id=draft.previous_operation_id,
    )
    return _build_retry_state(
        context_ref, draft, source,
        create_gate_operation_id=create_gate_operation_id,
        completed_stages=completed_stages,
        core_input_digest=core_input_digest,
        core_input_ref=core_input_ref,
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
        core_input_digest=state.core_input_digest, core_input_ref=state.core_input_ref,
        create_gate_operation_id=state.create_gate_operation_id,
        manifest_path=retry_manifest_path, now=now, previous=state.old_shared,
    ))
    for stage_key, source_stage in state.completed_stages.items():
        queue_operation_ref = state.context_ref.collection("operations").document(operation_id)
        transaction.set(queue_operation_ref.collection("stages").document(stage_key), {
            **source_stage,
            "operation_id": operation_id,
            "stage_key": stage_key,
            "state": "completed",
            "failures": 0,
            "attempts_started": 0,
            "in_flight_attempt_id": None,
            "retry_not_before": None,
            "last_error": None,
            "updated_at": now,
        })
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
    doc_id: str | None, digest: str, core_input_digest: str, core_input_ref: str,
    create_gate_operation_id: str,
    manifest_path: str, now: datetime,
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Initialize a clean shared operation while carrying immutable save identity."""
    return {
        "operation_id": operation_id,
        "origin": OperationOrigin.TELEGRAM.value,
        "origin_operation_ref": context_id,
        "caller_binding": previous.get("caller_binding") or f"telegram:{context_id}",
        "immutable_input_digest": core_input_digest,
        "immutable_input_ref": core_input_ref,
        "origin_binding": {
            "context_id": context_id, "draft_id": draft_id, "manifest_digest": digest,
        },
        "retry_authorization": {
            "operation_id": operation_id,
            "previous_operation_id": previous.get("operation_id"),
            "create_gate_operation_id": create_gate_operation_id,
            "context_id": context_id,
            "draft_id": draft_id,
            "manifest_digest": digest,
            "caller_binding": previous.get("caller_binding"),
            "immutable_input_digest": core_input_digest,
        },
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
