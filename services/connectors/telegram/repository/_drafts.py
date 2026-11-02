"""Persist Telegram draft state, save completion and bounded cleanup."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from google.cloud import firestore

from core.integrations.firestore.operations import OPERATIONS_COLLECTION
from core.integrations.firestore.operation_queue import validate_context_queue_lease
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


def _validate_save_fence_args(operation_id: Any, owner: Any, fence: Any, now: Any) -> None:
    if (not isinstance(operation_id, str) or not operation_id
            or not isinstance(owner, str) or not owner
            or type(fence) is not int or fence < 1
            or not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None):
        raise ValueError("queue save fence binding is incomplete")


def _assert_save_fence(
    client: Any, transaction: Any, *, context_id: str, draft_id: str,
    operation_id: str, owner: str, fence: int, now: datetime,
) -> None:
    context_ref = client.collection(SESSIONS).document(context_id)
    operation_ref = context_ref.collection("operations").document(operation_id)
    draft_ref = context_ref.collection("drafts").document(draft_id)
    context_snapshot = context_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    draft_snapshot = draft_ref.get(transaction=transaction)
    context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
    operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
    draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
    save_intent_id = (operation.get("save_intent_id") or operation.get("save_operation_id")
                      or operation_id)
    target_draft_id = operation.get("target_draft_id") or operation.get("draft_id")
    if (not context_snapshot.exists or not operation_snapshot.exists or not draft_snapshot.exists
            or operation.get("operation_id") != operation_id
            or operation.get("state") != "running"
            or target_draft_id != draft_id
            or draft.get("active_save_operation_id") != operation_id
            or draft.get("immutable_save_intent_id") != save_intent_id
            or context.get("reserved_operation_id") != operation_id
            or not validate_context_queue_lease(
                context, operation_id, owner=owner, fence=fence, now=now,
                expiry=context.get("lock_expires_at"),
            )):
        raise PermissionError("Telegram save checkpoint queue lease is stale")


def _completion_dialog_matches(
    dialog: Mapping[str, Any], *, dialog_number: int, conversation_id: str,
    draft_id: str, doc_id: str,
) -> bool:
    """Allow only this numbered draft dialog or its exact completed form."""
    return (
        dialog.get("dialog_number") == dialog_number
        and dialog.get("conversation_id") == conversation_id
        and (
            (dialog.get("draft_id") == draft_id and dialog.get("doc_id") is None)
            or (dialog.get("draft_id") is None and dialog.get("doc_id") == doc_id)
        )
    )


def _completion_conversation_matches(
    conversation: Mapping[str, Any], *, conversation_id: str,
    dialog_number: int, draft_id: str, doc_id: str, allow_create: bool = False,
) -> bool:
    """Allow only this draft conversation or its exact completed form."""
    if not conversation and allow_create:
        return True
    return (
        conversation.get("conversation_id") == conversation_id
        and conversation.get("dialog_number") == dialog_number
        and (
            (conversation.get("kind") == "draft"
             and conversation.get("draft_id") == draft_id
             and conversation.get("doc_id") is None)
            or (conversation.get("kind") == "document"
                and conversation.get("draft_id") is None
                and conversation.get("doc_id") == doc_id)
        )
    )


def _load_completion_transaction_records(
    client: Any, transaction: Any, *, context_id: str, draft_id: str,
    operation_id: str, intent_id: str,
) -> dict[str, Any] | None:
    """Read all authoritative completion and numbered-conversation records."""
    context_ref = client.collection(SESSIONS).document(context_id)
    draft_ref = context_ref.collection("drafts").document(draft_id)
    manifest_ref = draft_ref.collection("save_manifests").document(operation_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    operation_ref = client.collection(OPERATIONS_COLLECTION).document(operation_id)
    draft_snapshot = draft_ref.get(transaction=transaction)
    context_snapshot = context_ref.get(transaction=transaction)
    manifest_snapshot = manifest_ref.get(transaction=transaction)
    intent_snapshot = intent_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    draft = draft_snapshot.to_dict() or {} if draft_snapshot.exists else {}
    dialog_number = draft.get("dialog_number")
    conversation_id = draft.get("conversation_id")
    dialog_ref = (context_ref.collection("dialogs").document(str(dialog_number))
                  if type(dialog_number) is int and dialog_number > 0 else None)
    conversation_ref = (context_ref.collection("conversations").document(conversation_id)
                        if isinstance(conversation_id, str) and conversation_id else None)
    dialog_snapshot = dialog_ref.get(transaction=transaction) if dialog_ref is not None else None
    conversation_snapshot = (
        conversation_ref.get(transaction=transaction)
        if conversation_ref is not None else None
    )
    if not all((draft_snapshot.exists, context_snapshot.exists, manifest_snapshot.exists,
                intent_snapshot.exists, operation_snapshot.exists,
                dialog_snapshot is not None, conversation_snapshot is not None)):
        return None
    return {
        "draft_ref": draft_ref, "context_ref": context_ref,
        "manifest_ref": manifest_ref, "intent_ref": intent_ref,
        "operation_ref": operation_ref, "dialog_ref": dialog_ref,
        "conversation_ref": conversation_ref,
        "draft": draft, "context": context_snapshot.to_dict() or {},
        "manifest": manifest_snapshot.to_dict() or {},
        "intent": intent_snapshot.to_dict() or {},
        "operation": operation_snapshot.to_dict() or {},
        "dialog": dialog_snapshot.to_dict() or {},
        "conversation": conversation_snapshot.to_dict() or {},
        "conversation_exists": conversation_snapshot.exists,
    }


def _prepare_completion_binding(
    transaction: Any, records: Mapping[str, Any], *, draft_id: str,
) -> dict[str, Any] | None:
    """Validate the numbered dialog target and prepare compatible mappings."""
    draft = records["draft"]
    intent = records["intent"]
    operation = records["operation"]
    dialog_number = draft.get("dialog_number")
    conversation_id = draft.get("conversation_id")
    doc_id = (intent.get("completion_certificate") or {}).get("doc_id")
    if (not isinstance(doc_id, str) or not doc_id
            or type(dialog_number) is not int or dialog_number < 1
            or not isinstance(conversation_id, str) or not conversation_id
            or draft.get("draft_id") != draft_id
            or draft.get("dialog_number") != dialog_number
            or draft.get("conversation_id") != conversation_id
            or not _completion_dialog_matches(
                records["dialog"], dialog_number=dialog_number,
                conversation_id=conversation_id, draft_id=draft_id, doc_id=doc_id,
            )
            or not _completion_conversation_matches(
                records["conversation"], conversation_id=conversation_id,
                dialog_number=dialog_number, draft_id=draft_id, doc_id=doc_id,
                allow_create=not records["conversation_exists"],
            )):
        return None
    binding_ref = records["context_ref"].collection("document_bindings").document(
        hashlib.sha256(doc_id.encode("utf-8")).hexdigest(),
    )
    binding_snapshot = binding_ref.get(transaction=transaction)
    binding = binding_snapshot.to_dict() or {} if binding_snapshot.exists else {}
    if binding_snapshot.exists and (
            binding.get("doc_id") != doc_id
            or binding.get("conversation_id") != conversation_id
            or binding.get("dialog_number") != dialog_number):
        return None
    now = datetime.now(timezone.utc)
    dialog = dict(records["dialog"])
    dialog.pop("draft_id", None)
    dialog.update({
        "dialog_number": dialog_number, "conversation_id": conversation_id,
        "doc_id": doc_id,
        "metadata": {
            "title": operation.get("title"), "provider": "telegram",
            "folder_id": operation.get("selected_folder_id"),
            "folder_path": list(operation.get("selected_folder_path") or ()),
        },
        "created_at": records["dialog"].get("created_at", now),
        "updated_at": now,
    })
    conversation = dict(records["conversation"])
    conversation.update({
        "conversation_id": conversation_id, "kind": "document",
        "draft_id": None, "doc_id": doc_id,
        "dialog_number": dialog_number, "search_filters": None,
        "created_at": conversation.get("created_at", draft.get("created_at", now)),
        "last_substantive_at": conversation.get("last_substantive_at"),
        "last_turn_sequence": conversation.get("last_turn_sequence", 0),
    })
    return {
        "doc_id": doc_id, "dialog_number": dialog_number,
        "conversation_id": conversation_id, "binding_ref": binding_ref,
        "binding_exists": binding_snapshot.exists, "dialog": dialog,
        "conversation": conversation, "now": now,
    }


def _write_completion_binding(
    transaction: Any, records: Mapping[str, Any], binding: Mapping[str, Any], *,
    certificate: Mapping[str, Any], intent_id: str, draft_id: str,
) -> None:
    """Persist the prepared document binding and completion state atomically."""
    transaction.set(records["dialog_ref"], binding["dialog"])
    transaction.set(records["conversation_ref"], binding["conversation"])
    if not binding["binding_exists"]:
        transaction.set(binding["binding_ref"], {
            "doc_id": binding["doc_id"],
            "dialog_number": binding["dialog_number"],
            "conversation_id": binding["conversation_id"],
            "created_at": binding["now"], "updated_at": binding["now"],
        })
    draft = records["draft"]
    context = records["context"]
    draft.update({
        "state": TelegramDraftState.COMPLETED.value,
        "completion_certificate": dict(certificate),
        "completion_certificate_ref": f"{SAVE_INTENTS_COLLECTION}/{intent_id}",
        "doc_id": binding["doc_id"],
    })
    if draft.get("cleanup_state") not in {"pending", "authorized", "cleaned"}:
        draft["cleanup_state"] = "pending"
        draft["updated_at"] = binding["now"]
    context["last_completion_draft_id"] = draft_id
    context["last_doc_id"] = binding["doc_id"]
    transaction.set(records["draft_ref"], draft)
    transaction.set(records["context_ref"], context)


def _read_document_binding_state(
    transaction: Any, *, context_ref: Any, binding_ref: Any,
    conversation_ref: Any, operation_id: str | None, owner: str | None,
    fence: int | None, now: datetime,
) -> dict[str, Any]:
    """Read the parent fence and every existing binding record before planning writes."""
    context_snapshot = context_ref.get(transaction=transaction)
    context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
    operation_snapshot = None
    if operation_id is not None:
        operation_ref = context_ref.collection("operations").document(operation_id)
        operation_snapshot = operation_ref.get(transaction=transaction)
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        if (not context_snapshot.exists or not operation_snapshot.exists
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, operation_id, owner=owner, fence=fence, now=now,
                    expiry=context.get("lock_expires_at"),
                )):
            raise PermissionError("document binding queue lease is stale")
    binding_snapshot = binding_ref.get(transaction=transaction)
    conversation_snapshot = (
        conversation_ref.get(transaction=transaction)
        if conversation_ref is not None else None
    )
    return {
        "context_snapshot": context_snapshot,
        "context": context,
        "binding_snapshot": binding_snapshot,
        "existing_conversation": conversation_snapshot,
    }


def _plan_document_binding(
    transaction: Any, state: Mapping[str, Any], *, client: Any,
    context_ref: Any, binding_ref: Any, conversation_ref: Any,
    context_id: str, doc_id: str, conversation_id: str,
    metadata: Mapping[str, Any], now: datetime, update_last_doc: bool,
) -> dict[str, Any]:
    """Validate existing targets and prepare all document writes without applying them."""
    context = dict(state["context"])
    binding_snapshot = state["binding_snapshot"]
    existing_conversation_snapshot = state["existing_conversation"]
    existing_conversation = (
        existing_conversation_snapshot.to_dict() or {}
        if existing_conversation_snapshot and existing_conversation_snapshot.exists else {}
    )
    if binding_snapshot.exists:
        binding = binding_snapshot.to_dict() or {}
        if binding.get("doc_id") != doc_id:
            raise RuntimeError("document binding hash collision")
        if binding.get("conversation_id") != conversation_id:
            raise ValueError("document is already bound to a different conversation")
        number = _integer(binding["dialog_number"], "dialog_number", minimum=1)
        dialog_ref = _document_ref(client, f"{SESSIONS}/{context_id}/dialogs/{number}")
        dialog_snapshot = dialog_ref.get(transaction=transaction)
        dialog = dialog_snapshot.to_dict() or {}
        if (not dialog_snapshot.exists or dialog.get("doc_id") != doc_id
                or dialog.get("conversation_id") != conversation_id
                or dialog.get("dialog_number") != number):
            raise RuntimeError("document binding and dialog record disagree")
        if conversation_ref is not None:
            if existing_conversation and any(
                existing_conversation.get(key) != value for key, value in {
                    "conversation_id": conversation_id, "kind": "document",
                    "draft_id": None, "doc_id": doc_id,
                    "dialog_number": number, "search_filters": None,
                }.items()
            ):
                raise ValueError("conversation ID is already bound to a different target")
            if update_last_doc:
                context.update({"last_doc_id": doc_id, "updated_at": now})
        return {
            "kind": "existing", "number": number,
            "conversation": (None if existing_conversation else {
                "conversation_id": conversation_id, "kind": "document",
                "draft_id": None, "doc_id": doc_id,
                "dialog_number": number, "search_filters": None,
                "created_at": now, "last_substantive_at": None,
                "last_turn_sequence": 0,
            }) if conversation_ref is not None else None,
            "context": context if conversation_ref is not None and update_last_doc else None,
        }

    number = _integer(context.get("next_dialog_number", 1), "next_dialog_number", minimum=1)
    dialog_ref = _document_ref(client, f"{SESSIONS}/{context_id}/dialogs/{number}")
    if dialog_ref.get(transaction=transaction).exists:
        raise RuntimeError("dialog allocator points at an existing dialog")
    if conversation_ref is not None and existing_conversation:
        raise ValueError("conversation ID is already bound to a different target")
    context.update({"next_dialog_number": number + 1, "updated_at": now})
    if update_last_doc:
        context["last_doc_id"] = doc_id
    return {
        "kind": "new", "number": number, "dialog_ref": dialog_ref,
        "dialog": {"dialog_number": number, "conversation_id": conversation_id,
                   "doc_id": doc_id, "metadata": dict(metadata),
                   "created_at": now, "updated_at": now},
        "conversation": ({
            "conversation_id": conversation_id, "kind": "document",
            "draft_id": None, "doc_id": doc_id,
            "dialog_number": number, "search_filters": None,
            "created_at": now, "last_substantive_at": None,
            "last_turn_sequence": 0,
        } if conversation_ref is not None else None),
        "context": context,
    }


def _write_document_binding_plan(
    transaction: Any, plan: Mapping[str, Any], *, context_ref: Any,
    binding_ref: Any, conversation_ref: Any, doc_id: str,
    conversation_id: str, now: datetime,
) -> int:
    """Apply a validated existing or newly allocated binding plan atomically."""
    if plan["kind"] == "existing":
        if conversation_ref is not None and plan["conversation"] is not None:
            transaction.set(conversation_ref, plan["conversation"])
        if plan["context"] is not None:
            transaction.set(context_ref, plan["context"])
        return plan["number"]
    number = plan["number"]
    transaction.set(plan["dialog_ref"], plan["dialog"])
    transaction.set(binding_ref, {
        "doc_id": doc_id, "dialog_number": number,
        "conversation_id": conversation_id,
        "created_at": now, "updated_at": now,
    })
    if conversation_ref is not None:
        transaction.set(conversation_ref, plan["conversation"])
    transaction.set(context_ref, plan["context"])
    return number


def _bind_document_transaction(
    transaction: Any, *, client: Any, context_ref: Any, binding_ref: Any,
    conversation_ref: Any, operation_id: str | None, owner: str | None,
    fence: int | None, now: datetime, context_id: str, doc_id: str,
    conversation_id: str, metadata: Mapping[str, Any], update_last_doc: bool,
) -> int:
    """Separate transactional reads, target validation, write planning, and commit."""
    state = _read_document_binding_state(
        transaction, context_ref=context_ref, binding_ref=binding_ref,
        conversation_ref=conversation_ref, operation_id=operation_id,
        owner=owner, fence=fence, now=now,
    )
    plan = _plan_document_binding(
        transaction, state, client=client, context_ref=context_ref,
        binding_ref=binding_ref, conversation_ref=conversation_ref,
        context_id=context_id, doc_id=doc_id, conversation_id=conversation_id,
        metadata=metadata, now=now, update_last_doc=update_last_doc,
    )
    return _write_document_binding_plan(
        transaction, plan, context_ref=context_ref, binding_ref=binding_ref,
        conversation_ref=conversation_ref, doc_id=doc_id,
        conversation_id=conversation_id, now=now,
    )


class _TelegramDraftMixin:
    """Persist draft, completion-certificate and cleanup records."""

    def commit_completion(
        self, certificate: Any, *, queue_operation_id: str | None = None,
        queue_owner: str | None = None, queue_fence: int | None = None,
        queue_now: datetime | None = None,
    ) -> bool:
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
        if any(value is not None for value in (
                queue_operation_id, queue_owner, queue_fence, queue_now)):
            _validate_save_fence_args(queue_operation_id, queue_owner, queue_fence, queue_now)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            records = _load_completion_transaction_records(
                self.client, transaction, context_id=context_id,
                draft_id=draft_id, operation_id=operation_id, intent_id=intent_id,
            )
            if records is None:
                return False
            if queue_operation_id is not None:
                _assert_save_fence(
                    self.client, transaction, context_id=context_id, draft_id=draft_id,
                    operation_id=queue_operation_id, owner=queue_owner,
                    fence=queue_fence, now=queue_now,
                )
            if not _completion_records_match(
                context_id=context_id, draft_id=draft_id, intent_id=intent_id,
                operation_id=operation_id, manifest_digest=manifest_digest,
                telegram_certificate=data, draft=records["draft"],
                manifest=records["manifest"], intent=records["intent"],
                operation=records["operation"],
            ):
                return False
            binding = _prepare_completion_binding(
                transaction, records, draft_id=draft_id,
            )
            if binding is None:
                return False
            _write_completion_binding(
                transaction, records, binding,
                certificate=data, intent_id=intent_id, draft_id=draft_id,
            )
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

    def mark_sealed_save_failed(
        self, context_id: str, draft_id: str, *, operation_id: str, owner: str,
        fence: int, error_code: str, now: datetime,
    ) -> bool:
        """Retain an unprepared sealed draft in failed state under its live queue fence."""
        context_id, draft_id = _identifier(context_id, "context_id"), _identifier(draft_id, "draft_id")
        if not isinstance(error_code, str) or not error_code or len(error_code) > 48:
            raise ValueError("error_code is invalid")
        now = _aware(now, "now")
        _validate_save_fence_args(operation_id, owner, fence, now)
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = context_ref.collection("drafts").document(draft_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            snapshot = draft_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            draft = snapshot.to_dict() or {}
            operation_snapshot = operation_ref.get(transaction=transaction)
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            _assert_save_fence(
                self.client, transaction, context_id=context_id, draft_id=draft_id,
                operation_id=operation_id, owner=owner, fence=fence, now=now,
            )
            if (draft.get("active_save_operation_id") != operation_id
                    or draft.get("immutable_save_intent_id") is None
                    or draft.get("state") not in {TelegramDraftState.PROCESSING.value,
                                                    TelegramDraftState.FAILED.value}):
                return False
            if (error_code == "no_supported_content"
                    and (not operation_snapshot.exists
                         or operation.get("operation_id") != operation_id)):
                return False
            draft.update({"state": TelegramDraftState.FAILED.value,
                          "save_error_code": error_code, "updated_at": now})
            transaction.set(draft_ref, draft)
            if error_code == "no_supported_content":
                operation["safe_save_refusal"] = {
                    "code": "no_supported_content",
                    "message": (
                        "No supported text or completed audio was available. "
                        "The source is retained."
                    ),
                }
                operation["updated_at"] = now
                transaction.set(operation_ref, operation)
            return True
        return _txn(self.client.transaction(max_attempts=5))


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
        *,
        queue_operation_id: str | None = None,
        queue_owner: str | None = None,
        queue_fence: int | None = None,
        now: datetime | None = None,
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
        if any(value is not None for value in (queue_operation_id, queue_owner, queue_fence, now)):
            _validate_save_fence_args(queue_operation_id, queue_owner, queue_fence, now)

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if queue_operation_id is not None:
                _assert_save_fence(
                    self.client, transaction, context_id=context_id, draft_id=draft_id,
                    operation_id=queue_operation_id, owner=queue_owner, fence=queue_fence,
                    now=now,
                )
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
        *,
        queue_operation_id: str | None = None,
        queue_owner: str | None = None,
        queue_fence: int | None = None,
        now: datetime | None = None,
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
        if any(value is not None for value in (queue_operation_id, queue_owner, queue_fence, now)):
            _validate_save_fence_args(queue_operation_id, queue_owner, queue_fence, now)

        @firestore.transactional
        def _txn(transaction: Any) -> None:
            snapshot = message_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("source message does not exist")
            source = snapshot.to_dict() or {}
            if queue_operation_id is not None:
                _assert_save_fence(
                    self.client, transaction, context_id=context_id, draft_id=draft_id,
                    operation_id=queue_operation_id, owner=queue_owner, fence=queue_fence,
                    now=now,
                )
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
        queue_operation_id: str | None = None, queue_owner: str | None = None,
        queue_fence: int | None = None,
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
        if any(value is not None for value in (queue_operation_id, queue_owner, queue_fence)):
            _validate_save_fence_args(queue_operation_id, queue_owner, queue_fence, now)
        # Keep each transaction below Firestore's 500-write ceiling and publish the
        # completion metadata only after every prepared row has been checkpointed.
        for offset in range(0, len(rows), 400):
            page = rows[offset:offset + 400]

            @firestore.transactional
            def _write_page(transaction: firestore.Transaction) -> None:
                snapshot = draft_ref.get(transaction=transaction)
                if not snapshot.exists:
                    raise ValueError("draft does not exist")
                if queue_operation_id is not None:
                    _assert_save_fence(
                        self.client, transaction, context_id=context_id, draft_id=draft_id,
                        operation_id=queue_operation_id, owner=queue_owner,
                        fence=queue_fence, now=now,
                    )
                for row in page:
                    transaction.set(prepared_ref.document(str(row.order_index)), {
                        **asdict(row), "source_message_ids": list(row.source_message_ids),
                    })
            _write_page(self.client.transaction(max_attempts=5))

        @firestore.transactional
        def _finish(transaction: firestore.Transaction) -> None:
            snapshot = draft_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise ValueError("draft does not exist")
            draft_data = snapshot.to_dict() or {}
            if queue_operation_id is not None:
                _assert_save_fence(
                    self.client, transaction, context_id=context_id, draft_id=draft_id,
                    operation_id=queue_operation_id, owner=queue_owner,
                    fence=queue_fence, now=now,
                )
            draft_data.update({"required_audio_count": required_audio_count,
                               "completed_audio_ids": list(ids), "prepared_count": len(rows),
                               "context_capacity": context_capacity,
                               "preparation_digest": preparation_digest, "updated_at": now})
            transaction.set(draft_ref, draft_data)
        _finish(self.client.transaction(max_attempts=5))


    def bind_document(
        self, context_id: str, doc_id: str, conversation_id: str, *,
        metadata: Mapping[str, Any], now: datetime, update_last_doc: bool = True,
        operation_id: str | None = None, owner: str | None = None,
        fence: int | None = None, ensure_document_conversation: bool = False,
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
        if any(value is not None for value in (operation_id, owner, fence)):
            _validate_save_fence_args(operation_id, owner, fence, now)
        if ensure_document_conversation and not operation_id:
            raise ValueError("document conversation creation requires a queue fence")
        conversation_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/conversations/{conversation_id}",
        ) if ensure_document_conversation else None

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> int:
            return _bind_document_transaction(
                transaction, client=self.client, context_ref=context_ref,
                binding_ref=binding_ref, conversation_ref=conversation_ref,
                operation_id=operation_id, owner=owner, fence=fence, now=now,
                context_id=context_id, doc_id=doc_id, conversation_id=conversation_id,
                metadata=metadata, update_last_doc=update_last_doc,
            )

        return _txn(self.client.transaction(max_attempts=5))
