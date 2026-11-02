"""Context-bound repository ports for the production Telegram save workflow."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from google.cloud import firestore
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.integrations.firestore.input_parts import MAX_PART_BYTES
from services.connectors.telegram.repository import SESSIONS
from services.connectors.telegram.repository._common import _document_ref


async def _offload(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    from services.connectors.telegram.runtime.runtime import _offload as run
    return await run(function, *args, **kwargs)


class FirestoreTelegramSaveRepository:
    """Expose only the sealed input and current queue lease bound to one invocation."""

    def __init__(self, repository: Any, *, context_id: str, operation_id: str,
                 owner: str, fence: int, clock: Any, ledger: Any = None):
        self.repository = repository
        self.client = repository.client
        self.context_id, self.operation_id = context_id, operation_id
        self.owner, self.fence, self.clock = owner, fence, clock
        self.ledger = ledger

    def _refs(self):
        context = self.client.collection(SESSIONS).document(self.context_id)
        operation = context.collection("operations").document(self.operation_id)
        return context, operation

    def _assert_current(self) -> tuple[dict[str, Any], dict[str, Any]]:
        context_ref, operation_ref = self._refs()
        context_snapshot, operation_snapshot = context_ref.get(), operation_ref.get()
        context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        now = self.clock()
        if (not context_snapshot.exists or not operation_snapshot.exists
                or operation.get("operation_id") != self.operation_id
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, self.operation_id, owner=self.owner, fence=self.fence,
                    now=now, expiry=context.get("lock_expires_at"),
                )):
            raise PermissionError("Telegram save queue lease is stale")
        return context, operation

    def _retry_command_is_bound(
        self, command: Mapping[str, Any], operation: Mapping[str, Any],
        route: Mapping[str, Any], draft_id: str, input_ref: str,
    ) -> bool:
        """Match canonical retry work to its direct command or accepted request."""
        intent_id = operation.get("save_intent_id")
        request_id = operation.get("retry_request_operation_id")
        dialog_number = route.get("dialog_number")
        if (route.get("status") not in {"ok", "success"}
                or type(dialog_number) is not int or dialog_number < 1
                or operation.get("draft_id") != draft_id
                or operation.get("target_draft_id") != draft_id
                or operation.get("save_operation_id") != self.operation_id
                or not isinstance(intent_id, str) or not intent_id
                or (request_id is not None and (
                    not isinstance(request_id, str) or not request_id or "/" in request_id
                    or route.get("retry_request_operation_id") != request_id
                ))
                or (request_id is None
                    and route.get("retry_request_operation_id") is not None)):
            return False
        if request_id is None:
            return (
                command.get("kind") == "retry"
                and command.get("operation_id") == self.operation_id
                and command.get("draft_id") == draft_id
                and command.get("save_intent_id") == intent_id
                and command.get("dialog_number") == dialog_number
            )

        request = (_document_ref(
            self.client, f"{SESSIONS}/{self.context_id}/retry_requests/{request_id}",
        ).get().to_dict() or {})
        source_operation = (self.client.collection(SESSIONS).document(self.context_id)
                            .collection("operations").document(request_id).get().to_dict() or {})
        source_route = source_operation.get("route_control")
        return bool(
            command.get("kind") == "freeform"
            and command.get("operation_id") == request_id
            and request.get("accepted") is True
            and request.get("request_operation_id") == request_id
            and request.get("context_id") == self.context_id
            and request.get("operation_id") == self.operation_id
            and request.get("draft_id") == draft_id
            and request.get("dialog_number") == dialog_number
            and source_operation.get("operation_id") == request_id
            and source_operation.get("input_ref") == input_ref
            and isinstance(source_route, Mapping)
            and source_route.get("status") in {"ok", "success"}
            and source_route.get("kind") == "retry"
            and source_route.get("dialog_number") == dialog_number
        )

    def _load_sealed_draft(self, draft_id: str) -> dict[str, Any] | None:
        """Load exactly the manifest captured by this save command."""
        _context, operation = self._assert_current()
        input_ref = operation.get("input_ref")
        prefix = f"{SESSIONS}/{self.context_id}/commands/"
        if not isinstance(input_ref, str) or not input_ref.startswith(prefix):
            return None
        command = _document_ref(self.client, input_ref).get().to_dict() or {}
        route = operation.get("route_control")
        is_retry = isinstance(route, Mapping) and route.get("kind") == "retry"
        if command.get("context_id") != self.context_id:
            return None
        if is_retry:
            if not self._retry_command_is_bound(command, operation, route, draft_id, input_ref):
                return None
        elif (command.get("operation_id") != self.operation_id
              or command.get("draft_id") != draft_id):
            return None
        draft = self.repository.get_draft(self.context_id, draft_id)
        if not isinstance(draft, Mapping):
            return None
        manifest_path = draft.get("manifest_ref")
        expected_path = (f"{SESSIONS}/{self.context_id}/drafts/{draft_id}/"
                         f"save_manifests/{self.operation_id}")
        if manifest_path != expected_path:
            return None
        manifest = _document_ref(self.client, expected_path).get().to_dict() or {}
        immutable_intent_id = (
            operation.get("save_intent_id") if is_retry else
            operation.get("save_intent_id") or operation.get("save_operation_id") or self.operation_id
        )
        ids = manifest.get("ordered_message_ids")
        if (manifest.get("draft_id") != draft_id
                or manifest.get("save_operation_id") != self.operation_id
                or ids != draft.get("ordered_message_ids")
                or manifest.get("manifest_digest") != draft.get("manifest_digest")
                or draft.get("immutable_save_intent_id") != immutable_intent_id):
            return None
        expected_digest = hashlib.sha256(json.dumps(
            {"draft_id": draft_id, "message_ids": ids},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if manifest.get("manifest_digest") != expected_digest:
            return None
        wanted = set(ids or ())
        messages = [row for row in self.repository.list_messages(self.context_id, draft_id)
                    if row.get("message_id") in wanted]
        messages.sort(key=lambda row: (row.get("intake_sequence", 0), row.get("message_id", 0)))
        if [row.get("message_id") for row in messages] != ids:
            return None
        return {
            **dict(draft), "draft_id": draft_id, "context_id": self.context_id,
            "status": "sealed", "messages": messages, "source_manifest": messages,
        }

    async def load_sealed_draft(self, draft_id: str) -> dict[str, Any] | None:
        return await _offload(self._load_sealed_draft, draft_id)

    async def checkpoint_transcription(self, context_id: str, draft_id: str,
                                       message_id: int, text: str) -> None:
        await _offload(self._assert_current)
        if context_id != self.context_id:
            raise PermissionError("transcription context changed")
        await _offload(self.repository.checkpoint_transcription,
            self.context_id, draft_id, message_id, text,
            queue_operation_id=self.operation_id, queue_owner=self.owner,
            queue_fence=self.fence, now=self.clock(),
        )

    async def checkpoint_transcription_failure(self, context_id: str, draft_id: str,
                                              message_id: int, error_code: str) -> None:
        await _offload(self._assert_current)
        if context_id != self.context_id:
            raise PermissionError("transcription context changed")
        await _offload(self.repository.checkpoint_transcription_failure,
            self.context_id, draft_id, message_id, error_code,
            queue_operation_id=self.operation_id, queue_owner=self.owner,
            queue_fence=self.fence, now=self.clock(),
        )

    async def save_prepared_utterances(self, draft_id: str, utterances: Any, *,
                                       required_audio_count: int,
                                       completed_audio_ids: Any,
                                       context_capacity: int,
                                       preparation_digest: str) -> None:
        await _offload(self._assert_current)
        await _offload(self.repository.save_prepared_utterances,
            self.context_id, draft_id, utterances,
            required_audio_count=required_audio_count,
            completed_audio_ids=completed_audio_ids, context_capacity=context_capacity,
            preparation_digest=preparation_digest, now=self.clock(),
            queue_operation_id=self.operation_id, queue_owner=self.owner,
            queue_fence=self.fence,
        )

    async def record_skipped_utterance_count(self, count: int) -> None:
        """Checkpoint the deterministic projection count under the current queue fence."""
        if type(count) is not int or count < 1:
            raise ValueError("skipped utterance count must be positive")
        await _offload(self._record_skipped_utterance_count, count)

    def _record_skipped_utterance_count(self, count: int) -> None:
        context_ref, operation_ref = self._refs()
        now = self.clock()

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            if (not context_snapshot.exists or not operation_snapshot.exists
                    or operation.get("operation_id") != self.operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, self.operation_id, owner=self.owner, fence=self.fence,
                        now=now, expiry=context.get("lock_expires_at"),
                    )):
                raise PermissionError("Telegram save queue lease is stale")
            prior = operation.get("skipped_utterance_count")
            if prior is not None and prior != count:
                raise ValueError("skipped utterance count changed during save replay")
            operation["skipped_utterance_count"] = count
            operation["updated_at"] = now
            transaction.set(operation_ref, operation)

        _txn(self.client.transaction(max_attempts=5))

    async def mark_sealed_save_failed(self, draft_id: str, error_code: str) -> bool:
        await _offload(self._assert_current)
        return await _offload(self.repository.mark_sealed_save_failed,
            self.context_id, draft_id, operation_id=self.operation_id,
            owner=self.owner, fence=self.fence, error_code=error_code, now=self.clock(),
        )

    async def store_retry_stage_payload(self, operation_id: str, stage: str, value: Any) -> Any:
        if self.ledger is None or operation_id != self.operation_id:
            raise PermissionError("save stage payload has no invocation binding")
        if stage not in {"preparation", "analysis", "classification"}:
            raise ValueError("save stage payload kind is not eligible for temporary allocation")
        await _offload(self._assert_current)
        kind = f"save-stage-{stage}"
        payload = self.ledger._json_bytes(value)
        if len(payload) > self._inline_checkpoint_limit():
            digest = hashlib.sha256(payload).hexdigest()
            allocation = await _offload(
                self._stage_payload_allocation, kind, digest, len(payload),
            )
            if not await _offload(self._persist_stage_payload_allocation, allocation):
                raise PermissionError("save stage payload allocation lost its queue lease")
        assertion = (
            self._assert_stage_payload_allocation(allocation)
            if len(payload) > self._inline_checkpoint_limit() else None
        )
        return await _offload(
            self.ledger._store_payload, kind, value, write_assertion=assertion,
        )

    def _assert_stage_payload_allocation(self, allocation: Mapping[str, Any]):
        allocation_ref = (self._refs()[1].collection("save_payloads").document(
            allocation["allocation_id"],
        ))
        expected = dict(allocation)

        def assert_allocated(transaction: Any) -> None:
            snapshot = allocation_ref.get(transaction=transaction)
            current = snapshot.to_dict() or {} if snapshot.exists else {}
            if current != expected or current.get("cleanup_state") != "allocated":
                raise PermissionError("save payload allocation is no longer writable")

        return assert_allocated

    @staticmethod
    def _inline_checkpoint_limit() -> int:
        from services.connectors.telegram.runtime.ledger import _INLINE_CHECKPOINT_VALUE_BYTES
        return _INLINE_CHECKPOINT_VALUE_BYTES

    def _stage_payload_allocation(self, kind: str, digest: str, byte_length: int) -> dict[str, Any]:
        from services.connectors.telegram.runtime.save_cleanup import _allocation_document_id, _data_operation

        _context_ref, operation_ref = self._refs()
        operation = operation_ref.get().to_dict() or {}
        draft_id = operation.get("target_draft_id") or operation.get("draft_id")
        intent_id = operation.get("save_intent_id") or operation.get("save_operation_id") or self.operation_id
        draft = self.repository.get_draft(self.context_id, draft_id) if isinstance(draft_id, str) else None
        manifest_digest = draft.get("manifest_digest") if isinstance(draft, Mapping) else None
        if (not isinstance(draft_id, str) or not isinstance(intent_id, str)
                or not isinstance(manifest_digest, str) or not manifest_digest
                or draft.get("immutable_save_intent_id") != intent_id):
            raise PermissionError("save stage payload has no sealed draft binding")
        return {
            "allocation_id": _allocation_document_id(kind, digest),
            "context_id": self.context_id, "operation_id": self.operation_id,
            "draft_id": draft_id, "manifest_digest": manifest_digest,
            "save_intent_id": intent_id, "kind": kind,
            "input_ref": f"dialog_operations/{_data_operation(self.context_id, self.operation_id, kind, digest)}/input_parts/manifest",
            "sha256": digest, "byte_length": byte_length,
            "max_part_bytes": MAX_PART_BYTES, "cleanup_state": "allocated",
        }

    def _persist_stage_payload_allocation(self, allocation: Mapping[str, Any]) -> bool:
        """Publish a fenced ownership tombstone before writing large payload parts."""
        context_ref, operation_ref = self._refs()
        draft_ref = context_ref.collection("drafts").document(allocation["draft_id"])
        manifest_ref = draft_ref.collection("save_manifests").document(self.operation_id)
        allocation_ref = operation_ref.collection("save_payloads").document(
            allocation["allocation_id"],
        )

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            manifest_snapshot = manifest_ref.get(transaction=transaction)
            if not all(snapshot.exists for snapshot in (
                    context_snapshot, operation_snapshot, draft_snapshot, manifest_snapshot)):
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            draft = draft_snapshot.to_dict() or {}
            manifest = manifest_snapshot.to_dict() or {}
            if (operation.get("operation_id") != self.operation_id
                    or operation.get("state") != "running"
                    or (operation.get("target_draft_id") or operation.get("draft_id"))
                    != allocation["draft_id"]
                    or (operation.get("save_intent_id") or operation.get("save_operation_id")
                        or self.operation_id) != allocation["save_intent_id"]
                    or draft.get("immutable_save_intent_id") != allocation["save_intent_id"]
                    or draft.get("manifest_digest") != allocation["manifest_digest"]
                    or manifest.get("draft_id") != allocation["draft_id"]
                    or manifest.get("save_operation_id") != self.operation_id
                    or manifest.get("manifest_digest") != allocation["manifest_digest"]
                    or not validate_context_queue_lease(
                        context, self.operation_id, owner=self.owner, fence=self.fence,
                        now=self.clock(), expiry=context.get("lock_expires_at"),
                    )):
                return False
            prior_snapshot = allocation_ref.get(transaction=transaction)
            prior = prior_snapshot.to_dict() or {} if prior_snapshot.exists else {}
            if prior and prior != dict(allocation):
                return False
            transaction.set(allocation_ref, dict(allocation), merge=False)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    async def load_retry_stage_payload(self, payload_ref: Any) -> Any:
        if self.ledger is None:
            raise ValueError("save stage payload is unavailable")
        await _offload(self._assert_current)
        return await _offload(self.ledger._load_payload, payload_ref)

    async def load_previous_dialogue_context(self, draft_id: str) -> list[Any]:
        await _offload(self._assert_current)
        draft = await _offload(self.repository.get_draft, self.context_id, draft_id) or {}
        conversation_id = draft.get("conversation_id")
        if not conversation_id:
            return []
        turns = await _offload(self.repository.list_turns, self.context_id, conversation_id)
        return turns[-20:]

    async def get_context(self, context_id: str) -> dict[str, Any] | None:
        await _offload(self._assert_current)
        if context_id != self.context_id:
            return None
        return await _offload(self.repository.get_context, context_id)

    async def load_current_guard(self, draft_id: str) -> dict[str, Any] | None:
        context, operation = await _offload(self._assert_current)
        draft = await _offload(self.repository.get_draft, self.context_id, draft_id)
        immutable_intent_id = (operation.get("save_intent_id")
                               or operation.get("save_operation_id")
                               or self.operation_id)
        if (not isinstance(draft, Mapping)
                or draft.get("immutable_save_intent_id") != immutable_intent_id
                or operation.get("target_draft_id", operation.get("draft_id")) != draft_id
                or context.get("reserved_operation_id") != self.operation_id):
            return None
        return {
            "context_id": self.context_id, "operation_id": self.operation_id,
            "owner": self.owner, "fence": self.fence,
            "draft_id": draft_id, "manifest_ref": draft.get("manifest_ref"),
            "manifest_digest": draft.get("manifest_digest"),
            "save_intent_id": draft.get("immutable_save_intent_id"),
        }

    async def verify_save_receipt(self, receipt: Mapping[str, Any], *, draft_id: str,
                                  manifest_digest: str, guard: Mapping[str, Any]) -> bool:
        """Verify the actual shared completion certificate and immutable origin binding."""
        if not isinstance(guard, Mapping) or not guard:
            return False
        if await self.load_current_guard(draft_id) != guard:
            return False
        return await _offload(
            self._verify_save_receipt, receipt, draft_id, manifest_digest,
        )

    def _verify_save_receipt(self, receipt: Mapping[str, Any], draft_id: str,
                             manifest_digest: str) -> bool:
        from core.integrations.firestore.operations import get_operation
        from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION, get_save_intent

        operation = get_operation(self.client, self.operation_id)
        core_operation = get_operation(self.client, self.operation_id)
        intent_id = core_operation.save_intent_id if core_operation is not None else None
        if not isinstance(intent_id, str) or not intent_id:
            return False
        intent = get_save_intent(self.client, intent_id)
        if operation is None or intent is None:
            return False
        certificate = (self.client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
                       .get().to_dict() or {}).get("completion_certificate") or {}
        origin = {"context_id": self.context_id, "draft_id": draft_id,
                  "manifest_digest": manifest_digest}
        return bool(
            receipt.get("operation_id") == self.operation_id
            and receipt.get("save_intent_id") == intent_id
            and receipt.get("doc_id") == operation.doc_id
            and operation.state.value == "completed"
            and intent.doc_id == operation.doc_id
            and certificate.get("operation_id") == self.operation_id
            and certificate.get("save_intent_id") == intent_id
            and certificate.get("doc_id") == operation.doc_id
            and certificate.get("receipt_ref") == receipt.get("receipt_ref")
            and certificate.get("receipt_version") == receipt.get("receipt_version")
            and certificate.get("origin_binding") == origin
        )

    async def commit_completion(self, certificate: Any) -> bool:
        await _offload(self._assert_current)
        return await _offload(
            self.repository.commit_completion, certificate,
            queue_operation_id=self.operation_id, queue_owner=self.owner,
            queue_fence=self.fence, queue_now=self.clock(),
        )

    async def load_completion_certificate(self, context_id: str) -> Any:
        return await _offload(self._load_current_completion_certificate, context_id)

    def _load_current_completion_certificate(self, context_id: str) -> Any:
        """Load only this invocation's certificate, never another pending draft."""
        if context_id != self.context_id:
            return None
        operation = self._refs()[1].get().to_dict() or {}
        draft_id = operation.get("target_draft_id") or operation.get("draft_id")
        if not isinstance(draft_id, str) or not draft_id:
            return None
        draft = self.repository.get_draft(context_id, draft_id)
        if not isinstance(draft, Mapping):
            return None
        certificate = draft.get("completion_certificate")
        if (not isinstance(certificate, Mapping)
                or certificate.get("context_id") != context_id
                or certificate.get("draft_id") != draft_id
                or certificate.get("operation_id") != self.operation_id
                or draft.get("cleanup_state") not in {"pending", "authorized", "cleaned"}):
            return None
        return dict(certificate)

    async def authorize_cleanup(self, context_id: str, certificate: Any) -> bool:
        return await _offload(self.repository.authorize_cleanup, context_id, certificate)

    async def cleanup_temporary_save_payloads(self, context_id: str, certificate: Any) -> bool:
        if context_id != self.context_id:
            return False
        from services.connectors.telegram.runtime.save_cleanup import TemporarySavePayloadCleanup
        data = (certificate if isinstance(certificate, Mapping) else
                {name: getattr(certificate, name) for name in certificate.__dataclass_fields__})
        return await _offload(
            TemporarySavePayloadCleanup(
                self.client, context_id=self.context_id,
                authorize=self.repository.authorize_cleanup,
            ).cleanup, data,
        )

    async def list_manifest_pages(self, draft_id: str, *, cursor: Any, limit: int):
        return await _offload(self.repository.list_manifest_pages,
            draft_id, cursor=cursor, limit=limit, context_id=self.context_id,
        )

    async def delete_draft_page(self, draft_id: str, messages: Any) -> None:
        await _offload(self.repository.delete_draft_page,
                       draft_id, messages, context_id=self.context_id)

    async def list_prepared_pages(self, context_id: str, draft_id: str, *, limit: int):
        return await _offload(self.repository.list_prepared_pages,
                              context_id, draft_id, limit=limit)

    async def delete_prepared_page(self, context_id: str, draft_id: str, page: Any) -> None:
        await _offload(self.repository.delete_prepared_page, context_id, draft_id, page)

    async def finalize_cleanup(self, context_id: str, certificate: Any) -> bool:
        return await _offload(self.repository.finalize_cleanup, context_id, certificate)
