"""Transactional persistence for Telegram conversations and ordered turns."""

from __future__ import annotations

import hashlib
from typing import Any

from google.cloud import firestore

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
        now,
    ) -> ConversationRecord:
        context_id = _identifier(context_id, "context_id")
        requested = build_conversation(
            conversation_id, kind, draft_id=draft_id, doc_id=doc_id,
            dialog_number=dialog_number, search_filters=search_filters, now=now,
        )
        ref = self._conversation_ref(context_id, requested.conversation_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> dict[str, Any]:
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
                return current
            created = requested.to_dict()
            transaction.set(ref, created)
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

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> dict[str, Any]:
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
            return stored

        return _txn(self.client.transaction(max_attempts=5))

    def list_turns(self, context_id: str, conversation_id: str) -> list[dict[str, Any]]:
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        if conversation_ref is None:
            return []
        turns_ref = conversation_ref.collection("turns")
        rows = [snapshot.to_dict() or {} for snapshot in turns_ref.stream()]
        return sorted(rows, key=lambda row: _integer(row.get("turn_sequence"), "turn_sequence", minimum=1))
