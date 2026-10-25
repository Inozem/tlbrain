"""Persist Telegram draft state, save completion and bounded cleanup."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from google.cloud import firestore

from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from services.connectors.telegram.models import (
    PreparedUtterance,
    TelegramDraftState,
    TelegramTranscriptionState,
)
from services.connectors.telegram.repository._common import (
    SESSIONS,
    _aware,
    _cleanup_authority_matches,
    _completion_records_match,
    _document_ref,
    _draft_from,
    _identifier,
    _integer,
    _persisted_cleanup_binding_matches,
    _telegram_certificate_matches,
)

class _TelegramDraftMixin:
    """Persist draft, completion-certificate and cleanup records."""

    def commit_completion(self, certificate: Any) -> bool:
        """Link an immutable save certificate to its Telegram draft transactionally."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(data.get("context_id"), "context_id")
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        manifest_digest = data.get("manifest_digest")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            raise ValueError("manifest_digest is required")
        if tuple(data.get("origin_binding", ())) != (context_id, draft_id, manifest_digest):
            raise ValueError("completion certificate origin binding does not match Telegram draft")
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )
        intent_ref = self.client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
        operation_ref = self.client.collection(OPERATIONS_COLLECTION).document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            draft_snapshot = draft_ref.get(transaction=transaction)
            context_snapshot = context_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            intent_snapshot = intent_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not all((draft_snapshot.exists, context_snapshot.exists, manifest_snapshot.exists,
                        intent_snapshot.exists, operation_snapshot.exists)):
                return False
            draft = draft_snapshot.to_dict() or {}
            context = context_snapshot.to_dict() or {}
            manifest = manifest_snapshot.to_dict() or {}
            intent = intent_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if not _completion_records_match(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, draft=draft, manifest=manifest,
                intent=intent, operation=operation,
            ):
                return False
            shared_certificate = intent["completion_certificate"]
            draft.update({
                "state": TelegramDraftState.COMPLETED.value,
                "completion_certificate": data,
                "completion_certificate_ref": f"{SAVE_INTENTS_COLLECTION}/{intent_id}",
                "doc_id": shared_certificate.get("doc_id"),
            })
            if draft.get("cleanup_state") not in {"pending", "authorized", "cleaned"}:
                draft["cleanup_state"] = "pending"
                draft["updated_at"] = datetime.now(timezone.utc)
            context["last_completion_draft_id"] = draft_id
            context["last_doc_id"] = shared_certificate.get("doc_id")
            transaction.set(draft_ref, draft)
            transaction.set(context_ref, context)
            return True

        return _txn(self.client.transaction(max_attempts=5))


    def load_completion_certificate(self, context_id: str) -> dict[str, Any] | None:
        """Load the oldest unfinished completion so delayed cleanup can resume."""
        context_id = _identifier(context_id, "context_id")
        draft_collection = self.client.collection(SESSIONS).document(context_id).collection("drafts")
        pending = []
        for cleanup_state in ("pending", "authorized"):
            query = draft_collection.where("cleanup_state", "==", cleanup_state)
            order_by = getattr(query, "order_by", None)
            if callable(order_by):
                query = order_by("updated_at")
            snapshots = list(query.limit(1).stream())
            if snapshots:
                draft = snapshots[0].to_dict() or {}
                certificate = draft.get("completion_certificate")
                if isinstance(certificate, Mapping):
                    pending.append((str(draft.get("updated_at", "")), draft))
        if pending:
            pending.sort(key=lambda item: (item[0], item[1].get("draft_id", "")))
            return dict(pending[0][1]["completion_certificate"])
        context = self.get_context(context_id) or {}
        draft_id = context.get("last_completion_draft_id")
        if not draft_id:
            return None
        draft = self.get_draft(context_id, _identifier(draft_id, "draft_id")) or {}
        if draft.get("cleanup_state") == "cleaned":
            return None
        certificate = draft.get("completion_certificate")
        return dict(certificate) if isinstance(certificate, Mapping) else None


    def authorize_cleanup(self, context_id: str, certificate: Any) -> bool:
        """Validate shared completion evidence and durably authorize child deletion."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(context_id, "context_id")
        if data.get("context_id") != context_id:
            return False
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        manifest_digest = data.get("manifest_digest")
        if (not isinstance(manifest_digest, str) or not manifest_digest
                or tuple(data.get("origin_binding", ())) != (context_id, draft_id, manifest_digest)):
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )
        intent_ref = self.client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
        operation_ref = self.client.collection(OPERATIONS_COLLECTION).document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            if not (context_snapshot.exists and draft_snapshot.exists and manifest_snapshot.exists):
                return False
            context = context_snapshot.to_dict() or {}
            draft = draft_snapshot.to_dict() or {}
            manifest = manifest_snapshot.to_dict() or {}
            if draft.get("cleanup_state") in {"authorized", "cleaned"}:
                return _persisted_cleanup_binding_matches(
                    context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                    operation_id=operation_id, manifest_digest=manifest_digest,
                    telegram_certificate=data,
                    authorization=draft.get("cleanup_authorization") or {},
                    draft=draft, manifest=manifest,
                )
            intent_snapshot = intent_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not (intent_snapshot.exists and operation_snapshot.exists):
                return False
            intent = intent_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if not _cleanup_authority_matches(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, context=context, draft=draft,
                manifest=manifest, intent=intent, operation=operation,
            ):
                return False
            shared_certificate = intent["completion_certificate"]
            cert_id = shared_certificate["certificate_id"]
            expected_binding = {"context_id": context_id, "draft_id": draft_id,
                                "manifest_digest": manifest_digest}
            persisted_authorization = {
                "certificate_id": cert_id,
                "save_intent_id": intent_id,
                "operation_id": operation_id,
                "origin_binding": expected_binding,
                "telegram_certificate": data,
            }
            draft["cleanup_state"] = "authorized"
            draft["cleanup_authorization"] = persisted_authorization
            transaction.set(draft_ref, draft)
            return True

        return _txn(self.client.transaction(max_attempts=5))


    def list_manifest_pages(
        self, draft_id: str, *, cursor: Any = None, limit: int = 200,
        context_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Read one bounded source-message page for authorized cleanup."""
        if context_id is None:
            raise ValueError("context_id is required for source page listing")
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        limit = min(_integer(limit, "limit", minimum=1), 400)
        collection = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}"
        ).collection("messages")
        query = collection.limit(limit)
        page = list(query.stream())
        rows = [snapshot.to_dict() or {} for snapshot in page]
        next_cursor = page[-1] if len(page) == limit else None
        return rows, next_cursor


    def delete_draft_page(self, draft_id: str, messages: Sequence[Mapping[str, Any]], *, context_id: str) -> None:
        """Delete at most 400 message checkpoints from the authorized draft."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        if len(messages) > 400:
            raise ValueError("cleanup page exceeds the safe write limit")
        draft = self.get_draft(context_id, draft_id)
        if draft is None or draft.get("cleanup_state") != "authorized":
            raise ValueError("draft cleanup has not been authorized")
        batch = self.client.batch()
        for row in messages:
            message_id = _integer(row.get("message_id"), "message_id", minimum=1)
            batch.delete(_document_ref(
                self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}"
            ))
        batch.commit()


    def list_prepared_pages(self, context_id: str, draft_id: str, *, limit: int = 200) -> list[Any]:
        """Read the first bounded page of prepared utterance documents."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        limit = min(_integer(limit, "limit", minimum=1), 400)
        return list(_document_ref(
            self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}"
        ).collection("prepared").limit(limit).stream())


    def delete_prepared_page(self, context_id: str, draft_id: str, page: Sequence[Any]) -> None:
        """Delete one bounded prepared utterance page idempotently."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        if len(page) > 400:
            raise ValueError("cleanup page exceeds the safe write limit")
        draft = self.get_draft(context_id, draft_id)
        if draft is None or draft.get("cleanup_state") != "authorized":
            raise ValueError("draft cleanup has not been authorized")
        expected_prefix = f"{SESSIONS}/{context_id}/drafts/{draft_id}/prepared/"
        batch = self.client.batch()
        for snapshot in page:
            path = getattr(getattr(snapshot, "reference", None), "path", "")
            if not isinstance(path, str) or not path.startswith(expected_prefix):
                raise ValueError("prepared cleanup page contains a foreign document")
            batch.delete(snapshot.reference)
        batch.commit()


    def finalize_cleanup(self, context_id: str, certificate: Any) -> bool:
        """Mark the authorized draft cleaned while preserving newer context state."""
        data = asdict(certificate) if hasattr(certificate, "__dataclass_fields__") else dict(certificate)
        context_id = _identifier(context_id, "context_id")
        if data.get("context_id") != context_id:
            return False
        draft_id = _identifier(data.get("draft_id"), "draft_id")
        intent_id = _identifier(data.get("save_intent_id"), "save_intent_id")
        operation_id = _identifier(data.get("operation_id"), "operation_id")
        manifest_digest = data.get("manifest_digest")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        manifest_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            if not (context_snapshot.exists and draft_snapshot.exists and manifest_snapshot.exists):
                return False
            context = context_snapshot.to_dict() or {}
            draft = draft_snapshot.to_dict() or {}
            authorization = draft.get("cleanup_authorization") or {}
            manifest = manifest_snapshot.to_dict() or {}
            if not _persisted_cleanup_binding_matches(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, authorization=authorization,
                draft=draft, manifest=manifest,
            ):
                return False
            if draft.get("cleanup_state") == "cleaned":
                return True
            if draft.get("cleanup_state") != "authorized":
                return False
            draft["cleanup_state"] = "cleaned"
            draft["state"] = TelegramDraftState.CLEANED.value
            if context.get("last_completion_draft_id") == draft_id:
                context["last_doc_id"] = draft.get("doc_id")
            if context.get("active_draft_id") == draft_id:
                context["active_draft_id"] = None
                context["draft_status"] = None
            transaction.set(draft_ref, draft)
            transaction.set(context_ref, context)
            return True

        return _txn(self.client.transaction(max_attempts=5))


    def get_draft(self, context_id: str, draft_id: str) -> dict[str, Any] | None:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        snapshot = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}").get()
        return _draft_from(snapshot.to_dict() or {}) if snapshot.exists else None


    def list_messages(self, context_id: str, draft_id: str) -> list[dict[str, Any]]:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        snapshots = list(ref.collection("messages").stream())
        result = [snapshot.to_dict() or {} for snapshot in snapshots]
        result.sort(key=lambda row: (row["intake_sequence"], row["message_id"]))
        return result


    def checkpoint_transcription(
        self,
        context_id: str,
        draft_id: str,
        message_id: int,
        text: str,
    ) -> None:
        """Persist one completed transcription without changing its source record."""
        context_id = _identifier(context_id, "context_id")
        draft_id = _identifier(draft_id, "draft_id")
        message_id = _integer(message_id, "message_id", minimum=1)
        if not isinstance(text, str) or not text:
            raise ValueError("transcription text must be non-empty text")
        message_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if _integer(source.get("message_id"), "stored message_id", minimum=1) != message_id:
                raise ValueError("source message association does not match")
            if source.get("transcription_state") == TelegramTranscriptionState.COMPLETED.value:
                if source.get("transcription_text") != text:
                    raise ValueError("completed transcription cannot be replaced")
                return
            transaction.update(message_ref, {
                "transcription_state": TelegramTranscriptionState.COMPLETED.value,
                "transcription_text": text,
                "transcription_error": None,
            })

        _txn(self.client.transaction(max_attempts=5))


    def checkpoint_transcription_failure(
        self,
        context_id: str,
        draft_id: str,
        message_id: int,
        error_code: str,
    ) -> None:
        """Persist a bounded safe transcription failure category for one source."""
        context_id = _identifier(context_id, "context_id")
        draft_id = _identifier(draft_id, "draft_id")
        message_id = _integer(message_id, "message_id", minimum=1)
        allowed_errors = {"unsupported_audio", "download_failed", "transcription_failed"}
        if error_code not in allowed_errors or len(error_code) > 48:
            raise ValueError("transcription error code is invalid")
        message_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/drafts/{draft_id}/messages/{message_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if _integer(source.get("message_id"), "stored message_id", minimum=1) != message_id:
                raise ValueError("source message association does not match")
            if source.get("transcription_state") == TelegramTranscriptionState.COMPLETED.value:
                return
            transaction.update(message_ref, {
                "transcription_state": TelegramTranscriptionState.FAILED.value,
                "transcription_text": None,
                "transcription_error": error_code[:48],
            })

        _txn(self.client.transaction(max_attempts=5))


    def list_failed_drafts(self, context_id: str) -> list[dict[str, Any]]:
        context_id = _identifier(context_id, "context_id")
        collection = self.client.collection(SESSIONS).document(context_id).collection("drafts")
        snapshots = list(collection.stream())
        return [_draft_from(s.to_dict() or {}) for s in snapshots
                if (s.to_dict() or {}).get("state") == TelegramDraftState.FAILED.value]


    def save_prepared_utterances(
        self, context_id: str, draft_id: str, utterances: Sequence[PreparedUtterance], *,
        required_audio_count: int, completed_audio_ids: Sequence[int], context_capacity: int,
        preparation_digest: str, now: datetime,
    ) -> None:
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        now = _aware(now, "now")
        required_audio_count = _integer(required_audio_count, "required_audio_count")
        context_capacity = _integer(context_capacity, "context_capacity", minimum=1)
        if not isinstance(preparation_digest, str) or not preparation_digest:
            raise ValueError("preparation_digest is required")
        ids = tuple(_integer(item, "completed_audio_id", minimum=1) for item in completed_audio_ids)
        rows = tuple(utterances)
        if any(not isinstance(row, PreparedUtterance) for row in rows):
            raise TypeError("utterances must contain PreparedUtterance records")
        if tuple(row.order_index for row in rows) != tuple(range(len(rows))):
            raise ValueError("prepared utterance order_index values must be contiguous from zero")
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        prepared_ref = draft_ref.collection("prepared")

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> None:
            snapshot = draft_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("draft does not exist")
            draft_data = snapshot.to_dict() or {}
            for row in rows:
                ref = prepared_ref.document(str(row.order_index))
                transaction.set(ref, {**asdict(row), "source_message_ids": list(row.source_message_ids)})
            draft_data.update({"required_audio_count": required_audio_count,
                               "completed_audio_ids": list(ids), "prepared_count": len(rows),
                               "context_capacity": context_capacity,
                               "preparation_digest": preparation_digest, "updated_at": now})
            transaction.set(draft_ref, draft_data)

        _txn(self.client.transaction(max_attempts=5))


    def bind_document(
        self, context_id: str, doc_id: str, conversation_id: str, *,
        metadata: Mapping[str, Any], now: datetime,
    ) -> int:
        context_id = _identifier(context_id, "context_id")
        if not isinstance(doc_id, str) or not doc_id or "/" in doc_id:
            raise ValueError("doc_id must be a non-empty document identifier")
        conversation_id = _identifier(conversation_id, "conversation_id")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        now = _aware(now, "now")
        key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        binding_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/document_bindings/{key}")
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> int:
            binding_snap = binding_ref.get(transaction=transaction)
            if binding_snap.exists:
                binding = binding_snap.to_dict() or {}
                if binding.get("doc_id") != doc_id:
                    raise RuntimeError("document binding hash collision")
                return _integer(binding["dialog_number"], "dialog_number", minimum=1)
            context_snap = context_ref.get(transaction=transaction)
            context = context_snap.to_dict() or {}
            number = _integer(context.get("next_dialog_number", 1), "next_dialog_number", minimum=1)
            dialog_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/dialogs/{number}")
            existing = dialog_ref.get(transaction=transaction)
            if existing.exists:
                raise RuntimeError("dialog allocator points at an existing dialog")
            dialog = {"dialog_number": number, "conversation_id": conversation_id,
                      "doc_id": doc_id, "metadata": dict(metadata), "created_at": now,
                      "updated_at": now}
            transaction.set(dialog_ref, dialog)
            transaction.set(binding_ref, {"doc_id": doc_id, "dialog_number": number,
                                           "conversation_id": conversation_id,
                                           "created_at": now, "updated_at": now})
            context.update({"next_dialog_number": number + 1, "last_doc_id": doc_id,
                            "updated_at": now})
            transaction.set(context_ref, context)
            return number

        return _txn(self.client.transaction(max_attempts=5))

