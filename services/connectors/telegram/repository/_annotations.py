"""Private context annotation persistence and post-save application queue."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.outbox import (
    OutboxTargetKind, create_outbox_record_in_transaction,
)
from core.integrations.firestore.operation_queue import (
    register_operation_in_transaction, validate_context_queue_lease,
)
from services.connectors.telegram.models import TelegramDraftState
from services.connectors.telegram.conversations import build_conversation
from services.connectors.telegram.repository._common import (
    SESSIONS, _document_ref, _identifier, _integer, _aware,
)


def _annotation_source_message_ids(binding: Mapping[str, Any], command: Mapping[str, Any]) -> list[int] | None:
    """Return the immutable accepted source IDs for this annotation candidate."""
    source_ids = binding.get("matched_message_ids")
    if not isinstance(source_ids, list) or not source_ids:
        source_ids = command.get("cutoff_message_ids")
    if (not isinstance(source_ids, list) or not source_ids
            or any(type(value) is not int or value < 1 for value in source_ids)
            or len(set(source_ids)) != len(source_ids)):
        return None
    return list(source_ids)


def _draft_annotation_conversation(
    client: Any, transaction: Any, *, context_id: str, draft_id: str,
    draft: Mapping[str, Any], now: datetime,
) -> tuple[Any, dict[str, Any]] | None:
    """Resolve the exact draft conversation and dialog records before any write."""
    conversation_id = draft.get("conversation_id")
    dialog_number = draft.get("dialog_number")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    conversation_ref = _document_ref(
        client, f"{SESSIONS}/{context_id}/conversations/{conversation_id}",
    )
    conversation_snapshot = conversation_ref.get(transaction=transaction)
    conversation = conversation_snapshot.to_dict() or {} if conversation_snapshot.exists else {}
    dialog_ref = (_document_ref(client, f"{SESSIONS}/{context_id}/dialogs/{dialog_number}")
                  if type(dialog_number) is int and dialog_number > 0 else None)
    dialog_snapshot = dialog_ref.get(transaction=transaction) if dialog_ref is not None else None
    dialog = (dialog_snapshot.to_dict() or {}
              if dialog_snapshot is not None and dialog_snapshot.exists else {})
    if (not conversation_snapshot.exists
            and (dialog_snapshot is None or not dialog_snapshot.exists)):
        return None
    if (dialog.get("conversation_id") != conversation_id
            or dialog.get("draft_id") != draft_id or dialog.get("doc_id") is not None):
        return None
    if conversation_snapshot.exists:
        if (conversation.get("conversation_id") != conversation_id
                or conversation.get("dialog_number") != dialog_number
                or conversation.get("kind") != "draft"
                or conversation.get("draft_id") != draft_id):
            return None
    else:
        conversation = build_conversation(
            conversation_id, "draft", draft_id=draft_id,
            dialog_number=dialog_number, now=now,
        ).to_dict()
    return conversation_ref, conversation


def _annotation_route_matches(
    operation: Mapping[str, Any], *, text: str, text_digest: str, draft_id: str,
) -> bool:
    """Check that the pinned private route carries this note and target draft."""
    route = operation.get("route_control")
    return bool(
        isinstance(route, Mapping) and route.get("kind") == "annotation"
        and (route.get("annotation_text") == text
             or route.get("annotation_text_sha256") == text_digest)
        and route.get("target_draft_id") in (None, draft_id)
    )


def _register_draft_fact_application(
    client: Any, transaction: Any, *, context_ref: Any, comment_ref: Any,
    comment: dict[str, Any], context_id: str, draft_id: str, comment_id: str,
    doc_id: str, certificate_ref: str, receipt_ref: str, now: datetime,
) -> str:
    """Atomically append a stable shared-fact operation and its dispatch outbox."""
    fact_operation_id = "tgfactop-" + hashlib.sha256(comment_id.encode("utf-8")).hexdigest()
    input_ref = f"{SESSIONS}/{context_id}/drafts/{draft_id}/annotations/{comment_id}"
    identity = hashlib.sha256(json.dumps({
        "kind": "telegram_annotation_apply", "context_id": context_id,
        "draft_id": draft_id, "comment_id": comment_id, "doc_id": doc_id,
        "certificate_ref": certificate_ref,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    outbox_id = "tgfacttask-" + hashlib.sha256(comment_id.encode("utf-8")).hexdigest()

    def create_task() -> None:
        create_outbox_record_in_transaction(
            client, transaction, outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1, schedule_at=now, now=now,
            operation_id=fact_operation_id, context_id=context_id,
        )

    register_operation_in_transaction(
        client, transaction, context_ref, fact_operation_id,
        identity_digest=identity, input_ref=input_ref, now=now,
        before_write=create_task,
        operation_fields={
            "telegram_internal_kind": "annotation_apply",
            "context_id": context_id, "draft_id": draft_id,
            "annotation_id": comment_id, "target_doc_id": doc_id,
            "completion_certificate_ref": certificate_ref,
            "completion_receipt_ref": receipt_ref,
        },
    )
    comment.update({
        "state": "queued_fact", "application_operation_id": fact_operation_id,
        "completion_certificate_ref": certificate_ref,
        "completion_receipt_ref": receipt_ref,
        "applied_doc_id": doc_id, "updated_at": now,
    })
    transaction.set(comment_ref, comment)
    return fact_operation_id


class _TelegramAnnotationMixin:
    """Persist user-provided dialog context separately from transcript sources."""

    def stage_draft_annotation(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        draft_id: str, text: str, now: datetime,
    ) -> dict[str, Any] | None:
        """Persist a private user comment under its exact accepted collecting draft."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        draft_id = _identifier(draft_id, "draft_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("annotation text is invalid")
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        binding_ref = context_ref.collection("ingress_bindings").document(operation_id)
        comment_id = "tgcomment-" + hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        comment_ref = draft_ref.collection("annotations").document(comment_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            binding_snapshot = binding_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
            binding = binding_snapshot.to_dict() or {} if binding_snapshot.exists else {}
            input_ref = operation.get("input_ref")
            parts = input_ref.split("/") if isinstance(input_ref, str) else []
            if (len(parts) == 4 and parts[:3] == [SESSIONS, context_id, "commands"]
                    and parts[3].isdigit()):
                resolved_command_ref = _document_ref(self.client, input_ref)
                command_snapshot = resolved_command_ref.get(transaction=transaction)
                command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
            else:
                resolved_command_ref, command = None, {}
            source_ids = _annotation_source_message_ids(binding, command)
            if source_ids is None:
                return None
            source_refs = [draft_ref.collection("messages").document(str(value))
                           for value in source_ids]
            source_snapshots = [ref.get(transaction=transaction) for ref in source_refs]
            if not (context_snapshot.exists and operation_snapshot.exists and draft_snapshot.exists
                    and operation.get("state") == "running"
                    and operation.get("operation_id") == operation_id
                    and validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"))
                    and command.get("context_id") == context_id
                    and command.get("operation_id") == operation_id
                    and draft.get("draft_id") == draft_id
                    and draft.get("state") == TelegramDraftState.COLLECTING.value
                    and binding.get("context_id") == context_id
                    and binding.get("operation_id") == operation_id
                    and (binding.get("draft_id") == draft_id
                         or (binding.get("draft_id") is None
                             and command.get("draft_id") == draft_id
                             and bool(command.get("cutoff_message_ids"))))
                    and context.get("active_draft_id") == draft_id
                    and not context.get("pending_save_operation_id")
                    and all(snapshot.exists for snapshot in source_snapshots)
                    and _annotation_route_matches(
                        operation, text=text, text_digest=text_digest, draft_id=draft_id,
                    )):
                return None
            existing_snapshot = comment_ref.get(transaction=transaction)
            resolved_conversation = _draft_annotation_conversation(
                self.client, transaction, context_id=context_id, draft_id=draft_id,
                draft=draft, now=now,
            )
            if resolved_conversation is None:
                return None
            conversation_ref, conversation = resolved_conversation
            if existing_snapshot.exists:
                existing = existing_snapshot.to_dict() or {}
                if (existing.get("comment_id") != comment_id
                        or existing.get("operation_id") != operation_id
                        or existing.get("draft_id") != draft_id
                        or (existing.get("text") != text
                            and existing.get("text_sha256") != text_digest)):
                    return None
                return existing
            row = {
                "comment_id": comment_id, "context_id": context_id,
                "draft_id": draft_id, "operation_id": operation_id,
                **({"text": text} if isinstance(
                    operation["route_control"].get("annotation_text"), str
                ) else {
                    "text_payload": dict(operation["route_control"]["annotation_payload"]),
                    "text_sha256": text_digest,
                }),
                "state": "pending_save",
                "provenance": {
                    "kind": "telegram_user_annotation",
                    "command_update_id": command.get("update_id"),
                    "candidate_message_id": binding.get("candidate_message_id"),
                    "receipt_group_id": binding.get("receipt_group_id"),
                    "source_message_ids": list(source_ids),
                },
                "created_at": now, "updated_at": now,
            }
            route = dict(operation["route_control"])
            route["target_draft_id"] = draft_id
            annotation_drafts = conversation.get("annotation_draft_ids", [])
            if not isinstance(annotation_drafts, list):
                annotation_drafts = []
            conversation["annotation_draft_ids"] = list(dict.fromkeys(
                [value for value in annotation_drafts if isinstance(value, str)] + [draft_id]
            ))
            transaction.update(operation_ref, {"route_control": route})
            transaction.set(comment_ref, row)
            transaction.set(conversation_ref, conversation)
            return row

        return _txn(self.client.transaction(max_attempts=5))

    def list_draft_annotations(self, context_id: str, draft_id: str) -> list[dict[str, Any]]:
        """Read durable user comments without mixing them into source messages."""
        context_id = _identifier(context_id, "context_id")
        draft_id = _identifier(draft_id, "draft_id")
        ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        rows = ref.collection("annotations").stream()
        return [self._annotation_row(snapshot.to_dict() or {}) for snapshot in rows]

    def _annotation_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Expose full text only after the stored payload and text digest verify."""
        if isinstance(row.get("text"), str) or isinstance(row.get("text_payload"), Mapping):
            from services.connectors.telegram.repository._annotation_payloads import (
                load_annotation_row_text,
            )
            text = load_annotation_row_text(self.client, row)
        else:
            text = None
        if isinstance(text, str):
            return {**row, "text": text}
        if isinstance(row.get("text"), str):
            return {key: value for key, value in row.items() if key != "text"}
        return row

    def list_conversation_annotations(
        self, context_id: str, conversation_id: str,
    ) -> list[dict[str, Any]]:
        """Load supplemental annotations through the conversation's stable binding."""
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/conversations/{conversation_id}",
        )
        snapshot = ref.get()
        row = snapshot.to_dict() or {} if snapshot.exists else {}
        draft_ids = row.get("annotation_draft_ids", [])
        if not isinstance(draft_ids, list):
            draft_ids = []
        annotations = []
        for draft_id in draft_ids:
            if isinstance(draft_id, str) and draft_id:
                annotations.extend(self.list_draft_annotations(context_id, draft_id))
        annotations.extend(self._annotation_row(snapshot.to_dict() or {})
                           for snapshot in ref.collection("annotations").stream())
        return annotations

    def stage_conversation_annotation(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        conversation_id: str, doc_id: str, text: str, now: datetime,
    ) -> dict[str, Any] | None:
        """Keep a saved-document clarification on its exact canonical conversation."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        doc_id = _identifier(doc_id, "doc_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("annotation text is invalid")
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        conversation_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/conversations/{conversation_id}",
        )
        comment_id = "tgcomment-" + hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        comment_ref = conversation_ref.collection("annotations").document(comment_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            conversation_snapshot = conversation_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            conversation = conversation_snapshot.to_dict() or {} if conversation_snapshot.exists else {}
            route = operation.get("route_control")
            if (not context_snapshot.exists or not operation_snapshot.exists
                    or not conversation_snapshot.exists
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"))
                    or not isinstance(route, Mapping)
                    or route.get("kind") != "annotation"
                    or route.get("target_doc_id") != doc_id
                    or route.get("target_conversation_id") != conversation_id
                    or (route.get("annotation_text") != text
                        and route.get("annotation_text_sha256") != text_digest)
                    or conversation.get("conversation_id") != conversation_id
                    or conversation.get("kind") != "document"
                    or conversation.get("doc_id") != doc_id):
                return None
            existing_snapshot = comment_ref.get(transaction=transaction)
            if existing_snapshot.exists:
                existing = existing_snapshot.to_dict() or {}
                if (existing.get("comment_id") != comment_id
                        or existing.get("operation_id") != operation_id
                        or existing.get("doc_id") != doc_id
                        or (existing.get("text") != text
                            and existing.get("text_sha256") != text_digest)):
                    return None
                return existing
            row = {
                "comment_id": comment_id, "context_id": context_id,
                "conversation_id": conversation_id, "doc_id": doc_id,
                "operation_id": operation_id,
                **({"text": text} if isinstance(route.get("annotation_text"), str)
                   else {"text_payload": dict(route["annotation_payload"]),
                         "text_sha256": text_digest}),
                "state": "pending_fact",
                "provenance": {"kind": "telegram_user_annotation"},
                "created_at": now, "updated_at": now,
            }
            transaction.set(comment_ref, row)
            return row

        return _txn(self.client.transaction(max_attempts=5))

    def settle_conversation_annotation(
        self, context_id: str, conversation_id: str, comment_id: str, *,
        operation_id: str, owner: str, fence: int, state: str, now: datetime,
    ) -> bool:
        """Settle the visible saved-dialog clarification under its live queue fence."""
        if state not in {"applied", "failed"}:
            raise ValueError("annotation state is invalid")
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        operation_id = _identifier(operation_id, "operation_id")
        comment_id = _identifier(comment_id, "comment_id")
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        comment_ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/conversations/{conversation_id}/annotations/{comment_id}",
        )

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            comment_snapshot = comment_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            comment = comment_snapshot.to_dict() or {} if comment_snapshot.exists else {}
            if (not comment_snapshot.exists or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"))
                    or comment.get("operation_id") != operation_id
                    or comment.get("conversation_id") != conversation_id):
                return False
            comment.update({"state": state, "updated_at": now})
            transaction.set(comment_ref, comment)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def settle_draft_annotation(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        draft_id: str, comment_id: str, doc_id: str,
        state: str, now: datetime, error_code: str | None = None,
    ) -> bool:
        """Record fact application only against the live save certificate and queue fence."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        draft_id = _identifier(draft_id, "draft_id")
        comment_id = _identifier(comment_id, "comment_id")
        doc_id = _identifier(doc_id, "doc_id")
        if state not in {"applied", "failed", "pending_index"}:
            raise ValueError("annotation state is invalid")
        if error_code is not None and (not isinstance(error_code, str) or len(error_code) > 80):
            raise ValueError("error_code is invalid")
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        comment_ref = draft_ref.collection("annotations").document(comment_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            comment_snapshot = comment_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
            comment = comment_snapshot.to_dict() or {} if comment_snapshot.exists else {}
            proof = None
            if draft_snapshot.exists:
                from services.connectors.telegram.repository._annotation_proof import (
                    load_annotation_completion_proof,
                )
                proof = load_annotation_completion_proof(
                    self.client, context_id, draft_id, draft,
                    transaction=transaction, expected_doc_id=doc_id,
                )
            if not (
                context_snapshot.exists and operation_snapshot.exists and draft_snapshot.exists
                and comment_snapshot.exists and operation.get("state") == "running"
                and validate_context_queue_lease(
                    context, operation_id, owner=owner, fence=fence, now=now,
                    expiry=context.get("lock_expires_at"))
                and isinstance(proof, Mapping) and proof.get("doc_id") == doc_id
                and comment.get("comment_id") == comment_id
                and comment.get("draft_id") == draft_id
                and comment.get("context_id") == context_id
            ):
                return False
            if comment.get("state") == "applied":
                return comment.get("applied_doc_id") == doc_id
            comment.update({
                "state": state, "applied_doc_id": doc_id,
                "application_operation_id": operation_id, "updated_at": now,
            })
            if error_code is not None:
                comment["error_code"] = error_code
            else:
                comment.pop("error_code", None)
            transaction.set(comment_ref, comment)
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def enqueue_draft_annotation_application(
        self, context_id: str, save_operation_id: str, *, owner: str, fence: int,
        draft_id: str, comment_id: str, now: datetime,
    ) -> str | None:
        """Append one stable fact-application operation after a genuine save certificate."""
        context_id = _identifier(context_id, "context_id")
        save_operation_id = _identifier(save_operation_id, "save_operation_id")
        draft_id = _identifier(draft_id, "draft_id")
        comment_id = _identifier(comment_id, "comment_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        save_ref = context_ref.collection("operations").document(save_operation_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        comment_ref = draft_ref.collection("annotations").document(comment_id)

        @firestore.transactional
        def _txn(transaction: Any) -> str | None:
            context_snapshot = context_ref.get(transaction=transaction)
            save_snapshot = save_ref.get(transaction=transaction)
            draft_snapshot = draft_ref.get(transaction=transaction)
            comment_snapshot = comment_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            save = save_snapshot.to_dict() or {} if save_snapshot.exists else {}
            draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
            comment = comment_snapshot.to_dict() or {} if comment_snapshot.exists else {}
            certificate_ref = draft.get("completion_certificate_ref")
            expected_intent_id = (
                save.get("save_intent_id") or save.get("save_operation_id") or save_operation_id
            )
            from services.connectors.telegram.repository._annotation_proof import (
                load_annotation_completion_proof,
            )
            proof = (load_annotation_completion_proof(
                self.client, context_id, draft_id, draft, transaction=transaction,
            ) if draft_snapshot.exists else None)
            doc_id = proof.get("doc_id") if isinstance(proof, Mapping) else None
            receipt_path = proof.get("receipt_ref") if isinstance(proof, Mapping) else None
            if (not context_snapshot.exists or not save_snapshot.exists or not draft_snapshot.exists
                    or not comment_snapshot.exists or save.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, save_operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"))
                    or not isinstance(proof, Mapping)
                    or proof.get("operation_id") != save_operation_id
                    or proof.get("save_intent_id") != expected_intent_id
                    or not isinstance(doc_id, str) or not doc_id
                    or not isinstance(certificate_ref, str)
                    or comment.get("context_id") != context_id
                    or comment.get("draft_id") != draft_id
                    or comment.get("comment_id") != comment_id
                    or comment.get("state") not in {"pending_save", "pending_index", "queued_fact"}):
                return None
            return _register_draft_fact_application(
                self.client, transaction, context_ref=context_ref,
                comment_ref=comment_ref, comment=comment,
                context_id=context_id, draft_id=draft_id, comment_id=comment_id,
                doc_id=doc_id, certificate_ref=certificate_ref,
                receipt_ref=receipt_path, now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))
