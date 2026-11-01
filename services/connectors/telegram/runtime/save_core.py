"""Shared core save lifecycle and Telegram parent fencing."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from core.operations.models import OperationOrigin, OperationState
from services.connectors.telegram.runtime.save_fencing import (
    _GuardedGoogleProxy, _ParentBoundRetryGovernor, _RuntimeClock,
    _TelegramSaveRetryGovernor,
)
from services.connectors.telegram.save import TelegramSaveError, TelegramSavePending


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _accept_core_save(
    runtime: Any, *, context_id: str, draft_id: str, parent_operation_id: str,
    lease: Any, classified: Mapping[str, Any], operation_id: str,
    save_intent_id: str | None, origin_binding: Mapping[str, str] | None,
    origin_operation_ref: str | None,
) -> tuple[dict[str, str], Any, str, Any]:
    """Validate immutable Telegram provenance and accept or reload the shared save."""
    from core.dialogs.saving import accept_save
    from core.integrations.firestore.operations import get_operation

    client = runtime.repository.client
    if origin_binding is None:
        draft = runtime.repository.get_draft(context_id, draft_id) or {}
        origin_binding = {
            "context_id": context_id, "draft_id": draft_id,
            "manifest_digest": draft.get("manifest_digest", ""),
        }
    binding = dict(origin_binding)
    if set(binding) != {"context_id", "draft_id", "manifest_digest"} or any(
        not isinstance(value, str) or not value for value in binding.values()
    ):
        raise ValueError("save origin binding is invalid")
    now = runtime.clock()
    core_operation = get_operation(client, operation_id)
    if core_operation is None:
        accept_save(
            client, dict(classified), operation_id=operation_id,
            caller_binding=f"telegram:{context_id}:{parent_operation_id}",
            origin=OperationOrigin.TELEGRAM, now=now,
            origin_operation_ref=origin_operation_ref or
                f"telegram_sessions/{context_id}/drafts/{draft_id}",
            dialog_time="", save_intent_id=save_intent_id or operation_id,
            write_document=runtime._ledger(
                context_id, parent_operation_id, lease,
            )._write_input_document,
        )
        core_operation = get_operation(client, operation_id)
    if core_operation is None:
        raise RuntimeError("shared save acceptance is unavailable")
    if save_intent_id is not None and core_operation.save_intent_id != save_intent_id:
        raise ValueError("accepted save intent differs from the sealed draft")
    intent_id = core_operation.save_intent_id
    if binding.get("manifest_digest") == "pending":
        binding = {**binding, "manifest_digest": core_operation.immutable_input_digest}
    return binding, core_operation, intent_id, now


class _TelegramSharedSaveRuntime:
    """Coordinate shared save acceptance, receipt finalization, and queue fencing."""

    def _save_core(self, *, context_id: str, draft_id: str, parent_operation_id: str,
                   lease: Any, classified: Mapping[str, Any], core_operation_id: str | None = None,
                   save_intent_id: str | None = None,
                   origin_binding: Mapping[str, str] | None = None,
                   origin_operation_ref: str | None = None, **_kwargs: Any) -> Mapping[str, Any]:
        """Advance one provenance-bound shared save through real index confirmation."""
        from core.config import get_root_folder_id
        from core.dialogs.saving import advance_save
        from core.dialogs.completion import finalize_save
        from core.integrations.firestore.folders import get_all_folders, get_folders_by_speaker
        from core.integrations.firestore.operations import claim_operation, get_operation, release_operation
        from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION, get_save_intent
        from core.integrations.google_docs.client import build_docs_service
        from core.integrations.google_drive.client import build_drive_service_rw
        from core.operations.outbox import TaskTargetConfig, publish_due_outbox_records
        from core.integrations.firestore.outbox import OutboxTargetKind

        client = self.repository.client
        self._assert_parent_lease(context_id, parent_operation_id, lease)
        operation_id = core_operation_id or parent_operation_id
        binding, core_op, save_intent_id, now = _accept_core_save(
            self, context_id=context_id, draft_id=draft_id,
            parent_operation_id=parent_operation_id, lease=lease,
            classified=classified, operation_id=operation_id,
            save_intent_id=save_intent_id, origin_binding=origin_binding,
            origin_operation_ref=origin_operation_ref,
        )
        owner = f"tg-save-{_get(lease, 'owner')}-{_get(lease, 'fence')}"
        self._assert_parent_lease(context_id, parent_operation_id, lease)
        global_lease = claim_operation(client, operation_id, owner=owner, now=now, lease_seconds=600)
        if global_lease is None:
            current = get_operation(client, operation_id)
            if current is not None and current.state is OperationState.FAILED:
                raise TelegramSaveError(
                    "save", "shared_save_failed", retryable_stage=False,
                )
            raise TelegramSavePending(stage="save")
        try:
            self._bind_origin(client, operation_id, owner, global_lease.fence, binding)
            if core_op.state is not OperationState.COMPLETED:
                self._assert_parent_lease(context_id, parent_operation_id, lease)
                root = get_root_folder_id()
                folders = get_all_folders(client)
                drive = build_drive_service_rw()
                docs = build_docs_service(["https://www.googleapis.com/auth/documents"])
                advance_save(
                    client, operation_id, owner=owner, fence=global_lease.fence, now=self.clock(),
                    folders=folders, root_folder_id=root,
                    speaker_index_lookup=lambda speaker: [
                        row.id for row in get_folders_by_speaker(speaker, client)
                    ], suggestion=lambda *_args: None,
                 drive=_GuardedGoogleProxy(
                     drive, lambda: self._assert_parent_lease(context_id, parent_operation_id, lease),
                 ),
                 docs=_GuardedGoogleProxy(
                     docs, lambda: self._assert_parent_lease(context_id, parent_operation_id, lease),
                 ), governor=_ParentBoundRetryGovernor(
                     _TelegramSaveRetryGovernor(client, clock=_RuntimeClock(self.clock)),
                     clock=self.clock,
                     assert_current=lambda: self._assert_parent_lease(
                         context_id, parent_operation_id, lease,
                     ),
                 ),
                )
            target_url = os.environ.get("VECTOR_SYNC_URL", "").rstrip("/")
            task_queue = os.environ.get("DIALOG_OPERATIONS_QUEUE")
            service_account = os.environ.get("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL")
            if not target_url or not task_queue or not service_account:
                raise RuntimeError("index dispatch is not configured")
            self._assert_parent_lease(context_id, parent_operation_id, lease)
            publish_due_outbox_records(client, {
                OutboxTargetKind.INDEX_DISPATCH: TaskTargetConfig(
                    task_queue, target_url, service_account_email=service_account,
                    audience=target_url,
                )}, now=self.clock(), limit=20)
            self._assert_parent_lease(context_id, parent_operation_id, lease)
            result = finalize_save(
                client, operation_id=operation_id, save_intent_id=save_intent_id,
                owner=owner, fence=global_lease.fence,
                origin_binding=binding,
                now=self.clock(), clock=self.clock,
            )
            if result is None:
                raise TelegramSavePending(stage="indexing")
            record = get_operation(client, operation_id)
            intent = get_save_intent(client, save_intent_id)
            certificate = (client.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)
                          .get().to_dict() or {}).get("completion_certificate") or {}
            receipt_ref = certificate.get("receipt_ref")
            receipt_version = certificate.get("receipt_version")
            if not isinstance(receipt_ref, str) or not receipt_ref or receipt_version is None:
                raise RuntimeError("verified index receipt is unavailable")
            return {"operation_id": operation_id, "save_intent_id": save_intent_id,
                    "doc_id": result.document_id, "receipt_ref": receipt_ref,
                    "receipt_version": receipt_version,
                    "folder_id": record.selected_folder_id,
                    "folder_path": (
                        list(record.selected_folder_path)
                        if record.selected_folder_path is not None
                        and all(isinstance(item, str) for item in record.selected_folder_path)
                        else None
                    ),
                    "indexed": True,
                    "completion_certificate": certificate, "title": record.title}
        finally:
            current = get_operation(client, operation_id)
            if current is not None and current.lease_owner == owner:
                release_operation(client, operation_id, owner=owner,
                                  fence=global_lease.fence, now=self.clock())

    def _assert_parent_lease(self, context_id: str, operation_id: str, lease: Any) -> None:
        from core.integrations.firestore.operation_queue import validate_context_queue_lease
        context_ref = self.repository.client.collection("telegram_sessions").document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        context_snapshot, operation_snapshot = context_ref.get(), operation_ref.get()
        context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        if (not context_snapshot.exists or not operation_snapshot.exists
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, operation_id, owner=_get(lease, "owner"),
                    fence=_get(lease, "fence"), now=self.clock(),
                    expiry=context.get("lock_expires_at"),
                )):
            raise PermissionError("Telegram parent queue lease is stale")

    def _bind_origin(self, client: Any, operation_id: str, owner: str, fence: int,
                     binding: Mapping[str, str]) -> None:
        """Persist immutable protocol provenance under the current global save lease."""
        from google.cloud import firestore

        ref = client.collection("dialog_operations").document(operation_id)
        now = self.clock()
        @firestore.transactional
        def bind(tx: Any) -> None:
            snapshot = ref.get(transaction=tx)
            data = snapshot.to_dict() or {} if snapshot.exists else {}
            expiry = data.get("lease_expires_at")
            if (not snapshot.exists or data.get("state") != "running"
                    or data.get("lease_owner") != owner or data.get("fence") != fence
                    or not isinstance(expiry, datetime) or expiry <= now):
                raise PermissionError("shared save lease is stale")
            old = data.get("origin_binding")
            if old is not None and old != dict(binding):
                raise ValueError("shared save origin binding changed")
            if old is None:
                tx.update(ref, {"origin_binding": dict(binding), "updated_at": now})
        bind(client.transaction())
