"""Transactional persistence for Telegram conversations and ordered turns."""

from __future__ import annotations

import hashlib
from typing import Any

from google.cloud import firestore
from core.integrations.firestore.operation_queue import validate_context_queue_lease

from services.connectors.telegram.conversations import (
    ConversationRecord,
    build_conversation,
    build_turn_payload,
)
from services.connectors.telegram.repository._common import (
    SESSIONS,
    _document_ref,
    _identifier,
    _integer,
)


def _select_context_conversation(transaction: Any, context_ref: Any,
                                 context: dict[str, Any], conversation: ConversationRecord) -> None:
    """Select a session only when the current queue-fenced transaction still has no target."""
    if context.get("active_conversation_id") not in (None, conversation.conversation_id):
        return
    selection_sequence = _integer(context.get("selection_sequence", 0), "selection_sequence")
    selected_number = conversation.dialog_number
    if (context.get("active_conversation_id") != conversation.conversation_id
            or context.get("selected_dialog_number") != selected_number):
        context.update({
            "active_conversation_id": conversation.conversation_id,
            "selected_dialog_number": selected_number,
            "selection_sequence": selection_sequence + 1,
        })
        transaction.set(context_ref, context)


class _TelegramConversationMixin:
    def _conversation_ref(self, context_id: str, conversation_id: str):
        return _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/conversations/{conversation_id}",
        )

    def ensure_conversation(
        self,
        context_id: str,
        conversation_id: str,
        *,
        kind: str,
        draft_id: str | None = None,
        doc_id: str | None = None,
        dialog_number: int | None = None,
        search_filters: Any = None,
        queue_operation_id: str | None = None,
        queue_owner: str | None = None,
        queue_fence: int | None = None,
        select_if_unselected: bool = False,
        now,
    ) -> ConversationRecord:
        context_id = _identifier(context_id, "context_id")
        requested = build_conversation(
            conversation_id, kind, draft_id=draft_id, doc_id=doc_id,
            dialog_number=dialog_number, search_filters=search_filters, now=now,
        )
        ref = self._conversation_ref(context_id, requested.conversation_id)
        queue_values = (queue_operation_id, queue_owner, queue_fence)
        if any(value is not None for value in queue_values):
            if (not isinstance(queue_operation_id, str) or not queue_operation_id
                    or not isinstance(queue_owner, str) or not queue_owner
                    or type(queue_fence) is not int or queue_fence < 1):
                raise ValueError("conversation queue fence is incomplete")
        elif select_if_unselected:
            raise ValueError("queue fencing is required to select a conversation")

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> dict[str, Any]:
            context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
            operation_ref = (context_ref.collection("operations").document(queue_operation_id)
                             if queue_operation_id is not None else None)
            context_snapshot = context_ref.get(transaction=transaction) if queue_operation_id else None
            operation_snapshot = operation_ref.get(transaction=transaction) if operation_ref else None
            context = context_snapshot.to_dict() or {} if context_snapshot and context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot and operation_snapshot.exists else {}
            if queue_operation_id is not None and (
                not operation_snapshot.exists
                or operation.get("operation_id") != queue_operation_id
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, queue_operation_id, owner=queue_owner, fence=queue_fence,
                    now=now, expiry=context.get("lock_expires_at"),
                )
            ):
                raise PermissionError("Telegram operation lease is stale")
            if requested.kind.value in {"draft", "document"}:
                dialog_ref = _document_ref(
                    self.client,
                    f"{SESSIONS}/{context_id}/dialogs/{requested.dialog_number}",
                )
                dialog_snapshot = dialog_ref.get(transaction=transaction)
                dialog = dialog_snapshot.to_dict() or {}
                if (
                    not dialog_snapshot.exists
                    or dialog.get("dialog_number") != requested.dialog_number
                    or dialog.get("conversation_id") != requested.conversation_id
                    or dialog.get("draft_id") != requested.draft_id
                    or dialog.get("doc_id") != requested.doc_id
                ):
                    raise ValueError("conversation must match its numbered dialog")
            if requested.kind.value == "document":
                binding_key = hashlib.sha256(requested.doc_id.encode("utf-8")).hexdigest()
                binding_ref = _document_ref(
                    self.client,
                    f"{SESSIONS}/{context_id}/document_bindings/{binding_key}",
                )
                binding_snapshot = binding_ref.get(transaction=transaction)
                binding = binding_snapshot.to_dict() or {}
                if (
                    not binding_snapshot.exists
                    or binding.get("doc_id") != requested.doc_id
                    or binding.get("dialog_number") != requested.dialog_number
                    or binding.get("conversation_id") != requested.conversation_id
                ):
                    raise ValueError("document conversation must match its stable document binding")
            snapshot = ref.get(transaction=transaction)
            if snapshot.exists:
                current = snapshot.to_dict() or {}
                identity_fields = (
                    "conversation_id", "kind", "draft_id", "doc_id",
                    "dialog_number", "search_filters",
                )
                expected = requested.to_dict()
                if any(current.get(field) != expected[field] for field in identity_fields):
                    raise ValueError("conversation ID is already bound to a different target")
                if select_if_unselected and context.get("active_conversation_id") in (
                    None, requested.conversation_id,
                ):
                    _select_context_conversation(
                        transaction, context_ref, context, requested,
                    )
                return current
            created = requested.to_dict()
            transaction.set(ref, created)
            if select_if_unselected and context.get("active_conversation_id") in (
                None, requested.conversation_id,
            ):
                _select_context_conversation(
                    transaction, context_ref, context, requested,
                )
            return created

        return ConversationRecord.from_dict(_txn(self.client.transaction(max_attempts=5)))

    def get_conversation(
        self, context_id: str, conversation_id: str
    ) -> ConversationRecord | None:
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        ref = self._conversation_ref(context_id, conversation_id)
        if ref is None:
            return None
        snapshot = ref.get()
        return ConversationRecord.from_dict(snapshot.to_dict() or {}) if snapshot.exists else None

    def append_turn(
        self,
        context_id: str,
        conversation_id: str,
        *,
        turn_id: str,
        operation_id: str,
        role: str,
        parts: Any,
        tool_correlation: Any = None,
        source_refs: Any = None,
        queue_operation_id: str | None = None,
        queue_owner: str | None = None,
        queue_fence: int | None = None,
        select_if_unselected: bool = False,
        now,
    ) -> dict[str, Any]:
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        payload = build_turn_payload(
            conversation_id, turn_id=turn_id, operation_id=operation_id,
            role=role, parts=parts, tool_correlation=tool_correlation,
            source_refs=source_refs, now=now,
        )
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        if conversation_ref is None:
            raise ValueError("conversation does not exist")
        turn_ref = conversation_ref.collection("turns").document(payload["turn_id"])
        queue_values = (queue_operation_id, queue_owner, queue_fence)
        if any(value is not None for value in queue_values):
            if (not isinstance(queue_operation_id, str) or not queue_operation_id
                    or not isinstance(queue_owner, str) or not queue_owner
                    or type(queue_fence) is not int or queue_fence < 1):
                raise ValueError("turn queue fence is incomplete")
        elif select_if_unselected:
            raise ValueError("queue fencing is required to select a conversation")

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> dict[str, Any]:
            context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
            operation_ref = (context_ref.collection("operations").document(queue_operation_id)
                             if queue_operation_id is not None else None)
            context_snapshot = context_ref.get(transaction=transaction) if queue_operation_id else None
            operation_snapshot = operation_ref.get(transaction=transaction) if operation_ref else None
            context = context_snapshot.to_dict() or {} if context_snapshot and context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot and operation_snapshot.exists else {}
            if queue_operation_id is not None and (
                not operation_snapshot.exists
                or operation.get("operation_id") != queue_operation_id
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, queue_operation_id, owner=queue_owner, fence=queue_fence,
                    now=payload["created_at"], expiry=context.get("lock_expires_at"),
                )
            ):
                raise PermissionError("Telegram operation lease is stale")
            conversation_snapshot = conversation_ref.get(transaction=transaction)
            if not conversation_snapshot.exists:
                raise ValueError("conversation does not exist")
            turn_snapshot = turn_ref.get(transaction=transaction)
            if turn_snapshot.exists:
                current = turn_snapshot.to_dict() or {}
                immutable_fields = (
                    "turn_id", "conversation_id", "operation_id", "role", "parts",
                    "tool_correlation", "source_refs",
                )
                if any(current.get(field) != payload[field] for field in immutable_fields):
                    raise ValueError("turn ID is already bound to different content")
                if select_if_unselected and context.get("active_conversation_id") in (
                    None, conversation_id,
                ):
                    _select_context_conversation(transaction, context_ref, context,
                                                 ConversationRecord.from_dict(
                                                     conversation_snapshot.to_dict() or {}))
                return current

            conversation = conversation_snapshot.to_dict() or {}
            sequence = _integer(conversation.get("last_turn_sequence", 0), "last_turn_sequence") + 1
            stored = {**payload, "turn_sequence": sequence}
            conversation.update({
                "last_turn_sequence": sequence,
                "last_substantive_at": payload["created_at"],
            })
            transaction.set(turn_ref, stored)
            transaction.set(conversation_ref, conversation)
            if select_if_unselected and context.get("active_conversation_id") in (
                None, conversation_id,
            ):
                _select_context_conversation(transaction, context_ref, context,
                                             ConversationRecord.from_dict(conversation))
            return stored

        return _txn(self.client.transaction(max_attempts=5))

    def replace_model_turn_text(
        self,
        context_id: str,
        conversation_id: str,
        operation_id: str,
        text: str,
        *,
        queue_owner: str,
        queue_fence: int,
        now,
    ) -> bool:
        """Replace one completed model turn under the operation's live queue fence."""
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(text, str) or not text:
            raise ValueError("model turn text must be non-empty")
        turn_id = "telegram-" + hashlib.sha256(
            f"{operation_id}:model".encode("utf-8")
        ).hexdigest()
        context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
        operation_ref = context_ref.collection("operations").document(operation_id)
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        turn_ref = conversation_ref.collection("turns").document(turn_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            conversation_snapshot = conversation_ref.get(transaction=transaction)
            turn_snapshot = turn_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            turn = turn_snapshot.to_dict() or {} if turn_snapshot.exists else {}
            if (
                not operation_snapshot.exists
                or operation.get("operation_id") != operation_id
                or operation.get("state") != "running"
                or not validate_context_queue_lease(
                    context, operation_id, owner=queue_owner, fence=queue_fence,
                    now=now, expiry=context.get("lock_expires_at"),
                )
            ):
                raise PermissionError("Telegram operation lease is stale")
            if (
                not conversation_snapshot.exists or not turn_snapshot.exists
                or turn.get("conversation_id") != conversation_id
                or turn.get("operation_id") != operation_id
                or turn.get("role") != "model"
            ):
                return False
            updated = dict(turn)
            updated["parts"] = [{"text": text}]
            transaction.set(turn_ref, updated)
            return True

        return bool(_txn(self.client.transaction(max_attempts=5)))

    def list_turns(self, context_id: str, conversation_id: str) -> list[dict[str, Any]]:
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        if conversation_ref is None:
            return []
        turns_ref = conversation_ref.collection("turns")
        rows = [snapshot.to_dict() or {} for snapshot in turns_ref.stream()]
        return sorted(rows, key=lambda row: _integer(row.get("turn_sequence"), "turn_sequence", minimum=1))
