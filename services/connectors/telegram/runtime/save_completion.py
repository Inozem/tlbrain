"""Committed Telegram save execution, proof recovery, and cleanup coordination."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from core.operations.models import OperationState
from core.operations.queue import QueueOperationState
from services.connectors.telegram.audio import TelegramAudioTranscriber
from services.connectors.telegram.cleanup import TelegramDraftCleanup
from services.connectors.telegram.preprocessing import preprocess_messages
from services.connectors.telegram.repository._common import _document_ref
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from services.connectors.telegram.runtime.save_fencing import _ParentBoundRetryGovernor
from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository
from services.connectors.telegram.save import (
    TelegramClassifiedDialog, TelegramCompletionCertificate, TelegramSaveError,
    TelegramSavePending, TelegramSaveWorkflow,
)
logger = logging.getLogger("services.connectors.telegram.runtime.saving")


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _committed_save_proof(
    runtime: Any, context_id: str, draft_id: str, operation_id: str, intent_id: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], str] | None:
    """Validate the shared intent, origin binding, and cleanup authorization lineage."""
    from core.integrations.firestore.operations import get_operation
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION, get_save_intent

    client = runtime.repository.client
    operation = get_operation(client, operation_id)
    if operation is None or operation.state is not OperationState.COMPLETED:
        return None
    draft = runtime.repository.get_draft(context_id, draft_id)
    if not isinstance(draft, Mapping):
        return None
    digest = draft.get("manifest_digest")
    if (draft.get("immutable_save_intent_id") != intent_id
            or not isinstance(digest, str) or not digest):
        return None
    intent = get_save_intent(client, intent_id)
    if intent is None:
        return None
    intent_data = (client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
                   .get().to_dict() or {})
    operation_data = client.collection("dialog_operations").document(operation_id).get().to_dict() or {}
    shared = intent_data.get("completion_certificate") or {}
    binding = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest}
    cleanup_authorization = {
        "certificate_id": shared.get("certificate_id"),
        "save_intent_id": intent_id,
        "operation_id": operation_id,
        "origin_binding": binding,
    }
    if (shared.get("operation_id") != operation_id
            or shared.get("save_intent_id") != intent_id
            or shared.get("doc_id") != operation.doc_id
            or operation.save_intent_id != intent_id
            or intent.doc_id != operation.doc_id
            or shared.get("origin_binding") != binding
            or not isinstance(shared.get("certificate_id"), str)
            or not shared.get("certificate_id")
            or not shared.get("receipt_ref")
            or (operation_data.get("origin_binding") is not None
                and operation_data.get("origin_binding") != binding)
            or intent_data.get("cleanup_authorization") != cleanup_authorization
            or operation_data.get("cleanup_authorization")
            != intent_data.get("cleanup_authorization")):
        return None
    return operation, draft, shared, digest


class _TelegramCommittedSaveRuntime:
    """Own sealed save execution and repair from committed completion evidence."""

    async def save_draft(self, context_id: str, operation: Any, lease: Any, *,
                         retry_governor: Any = None) -> QueueOperationState:
        from services.connectors.telegram.runtime.runtime import _offload

        operation_id = _get(operation, "operation_id")
        try:
            binding = await _offload(self._captured_binding, context_id, operation)
            if binding is None:
                return QueueOperationState.FAILED
            draft_id, intent_id = binding
            ledger = self._ledger(context_id, operation_id, lease)
            repository = FirestoreTelegramSaveRepository(
                self.repository, context_id=context_id, operation_id=operation_id,
                owner=_get(lease, "owner"), fence=_get(lease, "fence"), clock=self.clock,
                ledger=ledger,
            )
            bound_governor = (
                _ParentBoundRetryGovernor(
                    retry_governor, clock=self.clock,
                    assert_current=repository._assert_current,
                ) if retry_governor is not None else None
            )
            if await self._recover_completed_save(
                repository, context_id, draft_id, operation_id, intent_id, lease,
            ):
                recovered_certificate = await _offload(
                    repository._load_current_completion_certificate, context_id,
                )
                cleaned = await TelegramDraftCleanup(repository=repository).cleanup(
                    context_id=context_id, certificate=recovered_certificate,
                )
                if not cleaned:
                    return QueueOperationState.RUNNING
                return QueueOperationState.COMPLETED
            request = await _offload(
                self._load_saved_request, context_id, intent_id, lease, operation,
            )
            audio = TelegramAudioTranscriber(
                telegram_client=self._telegram_client(), repository=repository,
                context_id=context_id, assert_current=repository._assert_current,
            )
            async def save_pipeline(result: Any, *, intent_id: str, **_kwargs: Any) -> Any:
                return await _offload(
                    self._save_core, context_id=context_id, draft_id=draft_id,
                    parent_operation_id=operation_id, lease=lease, classified=result,
                    save_intent_id=intent_id, retry_governor=bound_governor,
                )

            async def analyze(prepared: Any, *, context: Mapping[str, Any]) -> Any:
                return await _offload(
                    self._analyze_for_operation, prepared, context, request, operation_id,
                )

            async def classify(result: Any, **kwargs: Any) -> Any:
                return await _offload(self._classify, result, **kwargs)

            workflow = TelegramSaveWorkflow(
                repository=repository, audio_transcriber=audio, context_id=context_id,
                preprocess=preprocess_messages,
                analyze=analyze,
                classify=classify,
                save=save_pipeline,
            )
            certificate = await workflow.save(
                draft_id=draft_id, intent_id=intent_id, operation_id=operation_id,
                retry_governor=bound_governor, operation_lease=lease,
            )
            cleaned = await TelegramDraftCleanup(repository=repository).cleanup(
                context_id=context_id, certificate=certificate,
            )
            if not cleaned:
                return QueueOperationState.RUNNING
            return QueueOperationState.COMPLETED
        except TelegramSavePending:
            return QueueOperationState.RUNNING
        except TelegramSaveError as error:
            if error.retryable_stage is False:
                try:
                    failed = await _offload(
                        self.repository.mark_sealed_save_failed,
                        context_id, draft_id, operation_id=operation_id,
                        owner=_get(lease, "owner"), fence=_get(lease, "fence"),
                        error_code=error.error_code, now=self.clock(),
                    )
                except Exception:
                    failed = False
                if failed:
                    return QueueOperationState.FAILED
            raise

    async def _recover_completed_save(self, repository: FirestoreTelegramSaveRepository,
                                     context_id: str, draft_id: str,
                                     operation_id: str, intent_id: str, lease: Any) -> bool:
        """Restore cleanup from an existing local certificate or committed core evidence."""
        from services.connectors.telegram.runtime.runtime import _offload

        recovered = await _offload(
            self._completed_save_certificate, context_id, draft_id, operation_id, intent_id,
            self._ledger(context_id, operation_id, lease),
        )
        if recovered is None:
            return False
        certificate, already_persisted = recovered
        await _offload(repository._assert_current)
        if already_persisted:
            return True
        return await repository.commit_completion(certificate)

    def _completed_save_certificate(self, context_id: str, draft_id: str,
                                    operation_id: str, intent_id: str, ledger: Any) -> Any:
        """Return an immutable local certificate or rebuild it from real core evidence."""
        from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
        from core.integrations.firestore.index_completions import get_completion

        proof = _committed_save_proof(self, context_id, draft_id, operation_id, intent_id)
        if proof is None:
            return None
        core_operation, draft, shared, manifest_digest = proof
        client = self.repository.client
        receipt = get_completion(client, core_operation.doc_id, shared.get("content_hash"))
        if (receipt is None or receipt.get("status") != "completed"
                or receipt.get("version") != shared.get("receipt_version")
                or shared.get("receipt_ref") != (
                    f"index_completions/{core_operation.doc_id}/generations/"
                    f"{shared.get('content_hash')}")):
            return None
        local = draft.get("completion_certificate")
        if isinstance(local, Mapping):
            if (draft.get("completion_certificate_ref") != f"{SAVE_INTENTS_COLLECTION}/{intent_id}"
                    or local.get("context_id") != context_id or local.get("draft_id") != draft_id
                    or local.get("manifest_digest") != manifest_digest
                    or local.get("save_intent_id") != intent_id
                    or local.get("operation_id") != operation_id
                    or local.get("receipt_ref") != shared.get("receipt_ref")
                    or local.get("receipt_version") != shared.get("receipt_version")
                    or tuple(local.get("origin_binding", ())) != (
                        context_id, draft_id, manifest_digest)):
                return None
            return dict(local), True
        receipt = self._completed_receipt_payload(core_operation, shared)
        if receipt is None:
            return None
        receipt_payload = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        classification_outcome = self._recovered_classification_outcome(ledger, operation_id)
        return TelegramCompletionCertificate(
            context_id=context_id, draft_id=draft_id, manifest_digest=manifest_digest,
            save_intent_id=intent_id, operation_id=operation_id,
            receipt_ref=shared["receipt_ref"], receipt_version=shared.get("receipt_version"),
            receipt_digest=hashlib.sha256(receipt_payload).hexdigest(),
            origin_binding=(context_id, draft_id, manifest_digest),
            dialog_date=core_operation.date or "",
            classification_outcome=classification_outcome,
        ), False

    @staticmethod
    def _recovered_classification_outcome(ledger: Any, operation_id: str) -> str | None:
        """Recover only an already checkpointed classification outcome."""
        from core.integrations.firestore.operation_queue import QueueOperationRetryStore

        context_ref = ledger.client.collection("telegram_sessions").document(ledger.context_id)
        stage = QueueOperationRetryStore(ledger.client, context_ref).get_stage(
            operation_id=operation_id, stage_key="telegram.classification",
        )
        if stage is None or getattr(stage.state, "value", stage.state) != "completed":
            return None
        result = stage.result
        from services.connectors.telegram.save_checkpoints import stage_payload_ref
        payload_ref = stage_payload_ref(result)
        if payload_ref is not None:
            try:
                result = ledger._load_payload(payload_ref)
            except Exception:
                return None
        from services.connectors.telegram.save import _restore_classified
        classified = _restore_classified(result)
        if not isinstance(classified, TelegramClassifiedDialog):
            return None
        selection = classified.selection
        if selection.requires_unassigned:
            return "unassigned"
        if selection.speaker_candidates_ambiguous:
            return "ambiguous"
        return None

    @staticmethod
    def _completed_receipt_payload(operation: Any, certificate: Mapping[str, Any]) -> dict[str, Any] | None:
        """Recreate the receipt returned by the save adapter from durable core fields."""
        if not operation.doc_id or not operation.title:
            return None
        path = operation.selected_folder_path
        folder_path = (
            list(path) if path is not None and all(isinstance(item, str) for item in path)
            else None
        )
        return {
            "operation_id": operation.operation_id,
            "save_intent_id": operation.save_intent_id,
            "doc_id": operation.doc_id,
            "receipt_ref": certificate.get("receipt_ref"),
            "receipt_version": certificate.get("receipt_version"),
            "folder_id": operation.selected_folder_id,
            "folder_path": folder_path,
            "indexed": True,
            "completion_certificate": dict(certificate),
            "title": operation.title,
        }

    def _queue_row(self, context_id: str, operation: Any) -> dict[str, Any] | None:
        operation_id = _get(operation, "operation_id")
        if not isinstance(operation_id, str):
            return None
        ref = (self.repository.client.collection("telegram_sessions").document(context_id)
               .collection("operations").document(operation_id))
        snapshot = ref.get()
        row = snapshot.to_dict() or {} if snapshot.exists else {}
        return row if row.get("operation_id") == operation_id else None

    def _audio_failure_candidates(
        self, context_id: str, operation_id: str, binding: Any, draft: Any,
    ) -> list[tuple[str, str, str]]:
        """Select only per-message audio stages authorized by the sealed manifest."""
        if binding is None or draft is None:
            return []
        manifest_path = (
            f"telegram_sessions/{context_id}/drafts/{binding[0]}"
            f"/save_manifests/{operation_id}"
        )
        if _get(draft, "manifest_ref") != manifest_path:
            return []
        try:
            manifest = _document_ref(self.repository.client, manifest_path).get().to_dict() or {}
        except Exception:
            return []
        draft_ids = _get(draft, "ordered_message_ids")
        manifest_ids = manifest.get("ordered_message_ids")
        if not (
            manifest.get("draft_id") == binding[0]
            and manifest.get("save_operation_id") == operation_id
            and manifest.get("manifest_digest") == _get(draft, "manifest_digest")
            and manifest_ids == draft_ids
            and isinstance(manifest_ids, (list, tuple))
            and all(type(message_id) is int for message_id in manifest_ids)
        ):
            return []
        return [
            (f"telegram.audio.{message_id}", "gemini", "transcription")
            for message_id in manifest_ids
        ]

    def failure_reply(self, context_id: str, operation: Any, governor: Any) -> str:
        """Format terminal save failure from durable queue or shared-operation evidence."""
        from services.connectors.telegram.errors import format_terminal_error
        from core.integrations.firestore.operations import get_operation

        operation_id = _get(operation, "operation_id")
        row = self._queue_row(context_id, operation) or {}
        binding = self._captured_binding(context_id, operation)
        draft = self.repository.get_draft(context_id, binding[0]) if binding else None
        number = _get(draft, "dialog_number", row.get("dialog_number"))
        candidates = self._audio_failure_candidates(context_id, operation_id, binding, draft)
        candidates.extend((
            ("telegram.preparation", "telegram", "preparation"),
            ("telegram.analysis", "gemini", "metadata"),
            ("telegram.classification", "gemini", "classification"),
            (f"telegram.finalize.{operation_id}.docs", "google_docs", "finalization"),
            (f"telegram.finalize.{operation_id}.drive", "google_drive", "finalization"),
        ))
        for stage_key, service, stage_name in candidates:
            try:
                stage = governor.get_stage(operation_id, stage_key)
            except Exception:
                continue
            error = _get(stage, "last_error")
            failures = _get(stage, "failures", 0)
            state = getattr(_get(stage, "state"), "value", _get(stage, "state"))
            if error is not None and (state in {"failed", "exhausted"}
                                      or failures >= getattr(governor, "max_failures", 3)):
                reported_service = _get(error, "service")
                if reported_service:
                    service = reported_service
                elif stage_key.startswith("telegram.audio.") and _get(error, "code") in {
                    "telegramapierror", "filenotfounderror", "telegramaudioinputerror",
                }:
                    service = "telegram"
                logger.error("Telegram save failed", extra={
                    "event": "telegram_save_settlement", "operation_id": operation_id,
                    "context_id": context_id,
                    **({"doc_id": _get(draft, "doc_id")} if isinstance(
                        _get(draft, "doc_id"), str) and _get(draft, "doc_id") else {}),
                    "service": service, "stage": stage_name,
                    "error_code": _get(error, "code"),
                    "error_category": _get(error, "category"),
                    "failures": failures, "outcome": "failed",
                })
                return format_terminal_error(
                    dialog_number=number, operation_id=operation_id, service=service,
                    stage=stage_name, code=_get(error, "code"),
                    category=_get(error, "category"),
                    sanitized_message=_get(error, "sanitized_message") or _get(error, "message"),
                    occurred_at=_get(error, "occurred_at"), failures=failures,
                )

        core_operation = get_operation(self.repository.client, operation_id)
        error = _get(core_operation, "error")
        if error is not None:
            failures = _get(error, "failures")
            logger.error("Telegram save failed", extra={
                "event": "telegram_save_settlement", "operation_id": operation_id,
                "context_id": context_id,
                **({"doc_id": _get(core_operation, "doc_id")} if isinstance(
                    _get(core_operation, "doc_id"), str) and _get(core_operation, "doc_id") else {}),
                "service": _get(error, "service") or "core",
                "stage": _get(error, "stage") or _get(core_operation, "stage") or "save",
                "error_code": _get(error, "code"),
                "error_category": _get(error, "category"),
                "failures": failures, "outcome": "failed",
            })
            return format_terminal_error(
                dialog_number=number, operation_id=operation_id,
                service=_get(error, "service") or "core",
                stage=_get(error, "stage") or _get(core_operation, "stage") or "save",
                code=_get(error, "code"),
                category=_get(error, "category"),
                sanitized_message=_get(error, "sanitized_message") or _get(error, "message"),
                occurred_at=_get(error, "occurred_at"), failures=_get(error, "failures"),
            )
        return format_terminal_error(
            dialog_number=number, operation_id=operation_id, service="telegram",
            stage="save", code="save_failed", category="execution",
            sanitized_message="Stage execution failed", occurred_at=None,
            failures=0,
        )
