"""Transactional persistence for Telegram conversations and ordered turns."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore
from core.integrations.firestore.operation_queue import validate_context_queue_lease

from services.connectors.telegram.conversations import (
    ConversationRecord,
    build_conversation,
    build_turn_payload,
)
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    SESSIONS,
    _document_ref,
    _identifier,
    _integer,
    _aware,
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
            if operation.get("presentation_answer_turn_id") is not None:
                raise ValueError("presentation-bound model turn is immutable")
            if (
                not conversation_snapshot.exists or not turn_snapshot.exists
                or turn.get("conversation_id") != conversation_id
                or turn.get("operation_id") != operation_id
                or turn.get("role") != "model"
            ):
                return False
            if turn.get("presentation_binding") is not None:
                raise ValueError("presentation-bound model turn is immutable")
            updated = dict(turn)
            updated["parts"] = [{"text": text}]
            transaction.set(turn_ref, updated)
            return True

        return bool(_txn(self.client.transaction(max_attempts=5)))

    @staticmethod
    def _presentation_scope(manifest: Mapping[str, Any]) -> dict[str, str]:
        fields = ("context_id", "conversation_id", "producing_operation_id",
                  "presentation_id", "answer_turn_id")
        scope = {}
        for field in fields:
            scope[field] = _identifier(manifest.get(field), field)
        return scope

    def _presentation_ledger(
        self, *, context_id: str, operation_id: str, owner: str, fence: int,
        clock: Any = None,
    ):
        """Build the fenced ledger codec with a fresh default UTC clock.

        Keep this runtime dependency local: runtime.__init__ imports this
        repository, so importing the ledger at module load time would create a
        package initialization cycle.
        """
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

        return FirestoreAgentLedger(
            self.client, context_id=context_id, operation_id=operation_id,
            owner=owner, fence=fence,
            clock=clock if callable(clock) else lambda: datetime.now(timezone.utc),
        )

    def _presentation_original_ledger_exchange(
        self, loader: Any, *, operation_id: str, call_id: str, ordinal: int,
        checkpoint_conversation_id: str, exchange: Mapping[str, Any],
        transaction: Any,
    ) -> tuple[dict[str, Any], list[Any]]:
        """Verify an anchored exchange against its original immutable checkpoint."""
        checkpoint_ref = loader._ref(operation_id)
        checkpoint_snapshot = checkpoint_ref.get(transaction=transaction)
        checkpoint = checkpoint_snapshot.to_dict() or {} if checkpoint_snapshot.exists else {}
        if (not checkpoint_snapshot.exists
                or checkpoint.get("conversation_id") != checkpoint_conversation_id):
            raise ValueError("presentation source checkpoint is missing")
        calls = loader._load_payload(checkpoint.get("tool_calls", []), transaction=transaction)
        if not isinstance(calls, list):
            raise ValueError("presentation source checkpoint is malformed")
        matches = [item for item in calls if isinstance(item, Mapping)
                   and type(item.get("ordinal")) is int and item.get("ordinal") == ordinal]
        saved_call = matches[0] if len(matches) == 1 else None
        result_ref = checkpoint_ref.collection("tool_results").document(
            f"result-{ordinal:08d}",
        )
        result_snapshot = result_ref.get(transaction=transaction)
        result_row = result_snapshot.to_dict() or {} if result_snapshot.exists else {}
        saved_result = loader._load_payload(result_row.get("result"), transaction=transaction)
        native_id = saved_call.get("id") if isinstance(saved_call, Mapping) else None
        expected_call_id = native_id if isinstance(native_id, str) and native_id else (
            f"telegram-call-{operation_id}-{ordinal}")
        if (not result_snapshot.exists or type(result_row.get("ordinal")) is not int
                or result_row.get("ordinal") != ordinal
                or not isinstance(saved_call, Mapping) or expected_call_id != call_id
                or not self._presentation_strict_equal(dict(saved_call), exchange.get("call"))
                or not self._presentation_strict_equal(saved_result, exchange.get("result"))):
            raise ValueError("presentation source differs from its original ledger proof")
        return checkpoint, calls

    def _presentation_original_source_filters(
        self, loader: Any, checkpoint: Mapping[str, Any], calls: list[Any], *, context_id: str,
        conversation_id: str, source: Mapping[str, Any], transaction: Any,
    ) -> dict[str, Any]:
        """Derive source bounds from the accepted route or original search conversation."""
        route = loader._load_payload(checkpoint.get("application_route"), transaction=transaction)
        route_args = (route.get("arguments") if isinstance(route, Mapping)
                      and route.get("name") == "query" else None)
        if not isinstance(route_args, Mapping):
            first_query = next((item for item in calls if isinstance(item, Mapping)
                                and item.get("name") == "query"), None)
            route_args = first_query.get("arguments") if isinstance(first_query, Mapping) else None
        source_filters = source.get("source_filters")
        if not isinstance(source_filters, Mapping):
            raise ValueError("presentation source filters are malformed")
        if isinstance(route_args, Mapping):
            expected_filters = dict(route_args)
        else:
            conversation_snapshot = self._conversation_ref(context_id, conversation_id).get(
                transaction=transaction,
            )
            if not conversation_snapshot.exists:
                raise ValueError("presentation parent conversation is missing")
            record = ConversationRecord.from_dict(conversation_snapshot.to_dict() or {})
            if record.kind.value != "search":
                raise ValueError("presentation parent is not a search conversation")
            expected_filters = dict(record.search_filters or {})
        if not self._presentation_strict_equal(dict(source_filters), expected_filters):
            raise ValueError("presentation source bounds differ from original accepted bounds")
        return dict(source_filters)

    @staticmethod
    def _presentation_strict_equal(left: Any, right: Any) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(left, Mapping):
            return (set(left) == set(right) and all(
                _TelegramConversationMixin._presentation_strict_equal(left[key], right[key])
                for key in left
            ))
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                _TelegramConversationMixin._presentation_strict_equal(a, b)
                for a, b in zip(left, right)
            )
        return left == right

    @staticmethod
    def _presentation_utc_timestamp(value: Any, name: str) -> datetime:
        if (not isinstance(value, datetime) or value.tzinfo is None
                or value.utcoffset() is None
                or value.utcoffset().total_seconds() != 0):
            raise ValueError(f"{name} must be an aware UTC timestamp")
        return value.astimezone(timezone.utc)

    def _presentation_source_command(
        self, *, context_id: str, operation_id: str, actor_id: int,
        transaction: Any,
    ) -> None:
        operation_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/operations/{operation_id}",
        )
        operation_snapshot = operation_ref.get(transaction=transaction)
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if (operation.get("operation_id") != operation_id
                or operation.get("context_id", context_id) != context_id
                or len(parts) != 4 or parts[:3] != [SESSIONS, context_id, "commands"]):
            raise ValueError("presentation source has no accepted command")
        command_snapshot = _document_ref(self.client, input_ref).get(transaction=transaction)
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        if (not command_snapshot.exists or command.get("context_id") != context_id
                or command.get("operation_id") != operation_id
                or type(command.get("user_id")) is not int
                or command.get("user_id") != actor_id):
            raise ValueError("presentation source actor is not trusted")

    def _presentation_source_turn_reference(
        self, *, context_id: str, conversation_id: str, operation_id: str,
        ordinal: int, call_id: str, turn_id: str, expected_reference: Any,
        transaction: Any,
    ) -> dict[str, Any]:
        turn_ref = self._conversation_ref(context_id, conversation_id).collection(
            "turns",
        ).document(turn_id)
        snapshot = turn_ref.get(transaction=transaction)
        turn = snapshot.to_dict() or {} if snapshot.exists else {}
        if (not snapshot.exists or turn.get("turn_id") != turn_id
                or turn.get("conversation_id") != conversation_id
                or turn.get("operation_id") != operation_id or turn.get("role") != "tool"):
            raise ValueError("presentation source is absent from canonical conversation history")
        matches = []
        for part in turn.get("parts", []) if isinstance(turn.get("parts"), list) else []:
            if (isinstance(part, Mapping) and part.get("type") == "search_exchange_ref"
                    and isinstance(part.get("reference"), Mapping)):
                ref = part["reference"]
                if (ref.get("operation_id") == operation_id
                        and ref.get("ordinal") == ordinal and ref.get("call_id") == call_id):
                    matches.append(dict(ref))
        if len(matches) != 1:
            raise ValueError("presentation source reference is missing or ambiguous")
        original = matches[0]
        raw_payload = original.get("payload")
        if not isinstance(raw_payload, Mapping):
            raise ValueError("canonical search reference payload is malformed")
        compact = {**original, "payload": {key: raw_payload[key] for key in (
            "_input_ref", "sha256", "byte_length", "max_part_bytes",
        ) if key in raw_payload}}
        if not self._presentation_strict_equal(compact, expected_reference):
            raise ValueError("presentation source reference differs from canonical history")
        return original

    def _presentation_source_ledger_exchange(
        self, *, context_id: str, conversation_id: str, operation_id: str,
        checkpoint_conversation_id: str, call_id: str, ordinal: int,
        original_reference: Mapping[str, Any], source: Mapping[str, Any],
        transaction: Any, owner: str, fence: int, clock: Any,
    ) -> dict[str, Any]:
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

        loader = FirestoreAgentLedger(
            self.client, context_id=context_id, operation_id=operation_id,
            owner=owner, fence=fence,
            clock=clock if callable(clock) else lambda: datetime.now(timezone.utc),
        )
        exchange = loader.load_search_exchange(
            original_reference, conversation_id=conversation_id,
            operation_id=operation_id,
            checkpoint_conversation_id=checkpoint_conversation_id,
            call_id=call_id, ordinal=ordinal, transaction=transaction,
        )
        checkpoint, calls = self._presentation_original_ledger_exchange(
            loader, operation_id=operation_id, call_id=call_id, ordinal=ordinal,
            checkpoint_conversation_id=checkpoint_conversation_id,
            exchange=exchange, transaction=transaction,
        )
        source_filters = self._presentation_original_source_filters(
            loader, checkpoint, calls, context_id=context_id, conversation_id=conversation_id,
            source=source, transaction=transaction,
        )
        return {"reference": dict(original_reference), "exchange": exchange,
                "source_filters": dict(source_filters)}

    def _verified_presentation_exchanges(
        self, *, context_id: str, conversation_id: str, actor_id: int,
        sources: Any, source_turn_refs: Any, transaction: Any = None,
        owner: str = "presentation-reader", fence: int = 1, clock: Any = None,
    ) -> list[dict[str, Any]]:
        """Verify explicit anchored source turns against their accepted ledgers."""
        if not isinstance(sources, list) or not isinstance(source_turn_refs, Mapping):
            raise ValueError("presentation sources are malformed")
        source_ids = [item.get("source_id") if isinstance(item, Mapping) else None
                      for item in sources]
        if (any(not isinstance(item, str) or not item for item in source_ids)
                or len(source_ids) != len(set(source_ids))
                or set(source_turn_refs) != set(source_ids)):
            raise ValueError("presentation source turn anchors are incomplete or ambiguous")
        verified = []
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("presentation source is malformed")
            operation_id = _identifier(source.get("operation_id"), "source operation_id")
            ordinal = _integer(source.get("ordinal"), "source ordinal")
            call_id = source.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("source call_id must be a nonempty opaque ID")
            checkpoint_id = _identifier(
                source.get("checkpoint_conversation_id"), "checkpoint conversation_id",
            )
            self._presentation_source_command(
                context_id=context_id, operation_id=operation_id,
                actor_id=actor_id, transaction=transaction,
            )
            turn_id = _identifier(source_turn_refs[source["source_id"]], "source turn_id")
            original = self._presentation_source_turn_reference(
                context_id=context_id, conversation_id=conversation_id,
                operation_id=operation_id, ordinal=ordinal, call_id=call_id,
                turn_id=turn_id, expected_reference=source.get("reference"),
                transaction=transaction,
            )
            verified.append(self._presentation_source_ledger_exchange(
                context_id=context_id, conversation_id=conversation_id,
                operation_id=operation_id, checkpoint_conversation_id=checkpoint_id,
                call_id=call_id, ordinal=ordinal, original_reference=original,
                source=source, transaction=transaction, owner=owner, fence=fence,
                clock=clock,
            ))
        return verified

    def _presentation_manifest_payload(
        self, *, context_id: str, operation_id: str, owner: str, fence: int,
        clock: Any, row: Mapping[str, Any], body: str, transaction: Any = None,
    ) -> dict[str, Any]:
        from services.connectors.telegram.runtime.presentation import (
            validate_presentation_manifest,
        )

        expected = {
            key: row[key] for key in (
                "context_id", "conversation_id", "producing_operation_id",
                "presentation_id", "answer_turn_id",
            )
        }
        stored = row.get("manifest_payload")
        if not isinstance(stored, Mapping):
            raise ValueError("presentation manifest payload is missing or malformed")
        if set(stored) == {"_inline", "sha256", "byte_length"}:
            pass
        elif set(stored) == {"_input_ref", "sha256", "byte_length", "max_part_bytes"}:
            if type(stored.get("max_part_bytes")) is not int or stored["max_part_bytes"] < 1:
                raise ValueError("presentation multipart reference is malformed")
        else:
            raise ValueError("presentation payload reference has unknown fields")
        if (not isinstance(stored.get("sha256"), str)
                or len(stored["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in stored["sha256"])
                or type(stored.get("byte_length")) is not int
                or stored["byte_length"] < 0):
            raise ValueError("presentation payload integrity metadata is malformed")
        if "_input_ref" in stored:
            from core.integrations.firestore.input_parts import MAX_PART_BYTES
            if stored["max_part_bytes"] != MAX_PART_BYTES:
                raise ValueError("presentation payload part size is invalid")
            expected_data_operation = "telegram-ledger-" + hashlib.sha256(
                f"{context_id}\0{operation_id}\0presentation-{row.get('presentation_id')}\0{stored['sha256']}".encode()
            ).hexdigest()
            if stored.get("_input_ref") != (
                    f"dialog_operations/{expected_data_operation}/input_parts/manifest"):
                raise ValueError("presentation payload path is not bound to its scope and digest")
        ledger = self._presentation_ledger(
            context_id=context_id, operation_id=operation_id,
            owner=owner, fence=fence, clock=clock,
        )
        payload = ledger._load_payload(stored, transaction=transaction)
        if not isinstance(payload, Mapping) or "body" in payload:
            raise ValueError("stored presentation payload has an invalid body representation")
        encoded = ledger._json_bytes(payload)
        if (hashlib.sha256(encoded).hexdigest() != stored.get("sha256")
                or len(encoded) != stored.get("byte_length")):
            raise ValueError("presentation payload bytes do not verify")
        manifest = {**dict(payload), "body": body}
        verified_exchanges = self._verified_presentation_exchanges(
            context_id=context_id, conversation_id=expected["conversation_id"],
            actor_id=row.get("producer_actor_id"), sources=manifest.get("sources"),
            source_turn_refs=row.get("source_turn_refs", {}),
            transaction=transaction, owner=owner, fence=fence, clock=clock,
        )
        verified = validate_presentation_manifest(
            manifest, expected_scope=expected, verified_exchanges=verified_exchanges,
        ).to_dict()
        if (verified.get("body_sha256") != row.get("body_sha256")
                or verified.get("answer_turn_id") != row.get("answer_turn_id")):
            raise ValueError("presentation header does not match its body reference")
        return verified

    def _presentation_header_address_in_transaction(
        self, transaction: Any, *, context_id: str, conversation_id: str,
        presentation_id: str, row: Mapping[str, Any],
    ) -> tuple[Any, dict[str, str], datetime]:
        """Validate exact header shape, address, parent, and publication time."""
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        scope = self._presentation_scope(row)
        if set(row) != set(scope) | {
                "schema_version", "producer_actor_id", "answer_turn_sequence",
                "body_ref", "body_sha256", "manifest_payload", "source_turn_refs",
                "created_at"}:
            raise ValueError("presentation header shape is malformed")
        if (scope["context_id"] != context_id
                or scope["conversation_id"] != conversation_id
                or scope["presentation_id"] != presentation_id):
            raise ValueError("presentation address does not match its immutable scope")
        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("presentation schema version is invalid")
        created_at = self._presentation_utc_timestamp(row.get("created_at"), "created_at")
        conversation_snapshot = conversation_ref.get(transaction=transaction)
        if not conversation_snapshot.exists:
            raise ValueError("presentation parent conversation is missing")
        conversation = conversation_snapshot.to_dict() or {}
        record = ConversationRecord.from_dict(conversation)
        if record.conversation_id != conversation_id:
            raise ValueError("presentation parent conversation is malformed")
        return conversation_ref, scope, created_at

    def _presentation_producer_binding_in_transaction(
        self, transaction: Any, *, context_id: str, scope: Mapping[str, str],
        row: Mapping[str, Any],
    ) -> None:
        """Verify the accepted producer actor and its immutable answer-turn pin."""
        producer_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/operations/{scope['producing_operation_id']}",
        )
        producer_snapshot = producer_ref.get(transaction=transaction)
        producer = producer_snapshot.to_dict() or {} if producer_snapshot.exists else {}
        input_ref = producer.get("input_ref")
        input_parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if (producer.get("operation_id") != scope["producing_operation_id"]
                or len(input_parts) != 4
                or input_parts[:3] != [SESSIONS, context_id, "commands"]):
            raise ValueError("presentation producer command binding is missing")
        command_ref = _document_ref(self.client, input_ref)
        command_snapshot = command_ref.get(transaction=transaction)
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        if (command.get("context_id") != context_id
                or command.get("operation_id") != scope["producing_operation_id"]
                or type(command.get("user_id")) is not int
                or command["user_id"] <= 0
                or type(row.get("producer_actor_id")) is not int
                or row.get("producer_actor_id") != command.get("user_id")):
            raise ValueError("presentation producer actor binding is invalid")
        if producer.get("presentation_answer_turn_id") != scope["answer_turn_id"]:
            raise ValueError("presentation producer answer-turn binding is invalid")

    def _presentation_bound_turn_body_in_transaction(
        self, transaction: Any, *, conversation_ref: Any, scope: Mapping[str, str],
        row: Mapping[str, Any], created_at: datetime,
    ) -> str:
        """Verify the canonical answer body, sequence, reference, and full marker."""
        turn_ref = conversation_ref.collection("turns").document(scope["answer_turn_id"])
        turn_snapshot = turn_ref.get(transaction=transaction)
        if not turn_snapshot.exists:
            raise ValueError("presentation answer turn is missing")
        turn = turn_snapshot.to_dict() or {}
        body = self._presentation_turn_body(turn, scope)
        binding = turn.get("presentation_binding")
        body_ref = row.get("body_ref")
        if (not isinstance(body_ref, Mapping)
                or set(body_ref) != {"turn_id", "turn_sequence", "body_sha256"}
                or body_ref.get("turn_id") != scope["answer_turn_id"]
                or body_ref.get("body_sha256") != row.get("body_sha256")):
            raise ValueError("presentation body reference is malformed")
        turn_sequence = _integer(turn.get("turn_sequence"), "turn_sequence", minimum=1)
        if (_integer(body_ref.get("turn_sequence"), "body turn_sequence", minimum=1)
                != turn_sequence
                or _integer(row.get("answer_turn_sequence"), "answer_turn_sequence", minimum=1)
                != turn_sequence):
            raise ValueError("presentation body turn sequence is contradictory")
        # Multipart reads are addressed and integrity checked by the existing
        # ledger codec.  They intentionally do not depend on the producer lease.
        binding_expected = {
            "scope": scope,
            "body_sha256": row.get("body_sha256"),
            "answer_turn_sequence": turn_sequence,
            "manifest_payload": row.get("manifest_payload"),
            "source_turn_refs": row.get("source_turn_refs"),
            "created_at": created_at,
        }
        if not self._presentation_strict_equal(binding, binding_expected):
            raise ValueError("presentation answer turn binding is malformed")
        return body

    def _presentation_header_body_in_transaction(
        self, transaction: Any, *, context_id: str, conversation_id: str,
        presentation_id: str, row: Mapping[str, Any],
    ) -> tuple[dict[str, str], str]:
        """Verify an addressed presentation's accepted producer and canonical body."""
        conversation_ref, scope, created_at = self._presentation_header_address_in_transaction(
            transaction, context_id=context_id, conversation_id=conversation_id,
            presentation_id=presentation_id, row=row,
        )
        self._presentation_producer_binding_in_transaction(
            transaction, context_id=context_id, scope=scope, row=row,
        )
        body = self._presentation_bound_turn_body_in_transaction(
            transaction, conversation_ref=conversation_ref, scope=scope,
            row=row, created_at=created_at,
        )
        return scope, body

    def _read_presentation_manifest_in_transaction(
        self, transaction: Any, *, context_id: str, conversation_id: str,
        presentation_id: str, owner: str = "presentation-reader", fence: int = 1,
        now: Any = None,
    ) -> dict[str, Any] | None:
        """Read one addressed immutable artifact; missing differs from corrupt."""
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        presentation_id = _identifier(presentation_id, "presentation_id")
        header_ref = self._conversation_ref(context_id, conversation_id).collection(
            "presentations",
        ).document(presentation_id)
        header_snapshot = header_ref.get(transaction=transaction)
        if not header_snapshot.exists:
            return None
        row = header_snapshot.to_dict() or {}
        scope, body = self._presentation_header_body_in_transaction(
            transaction, context_id=context_id, conversation_id=conversation_id,
            presentation_id=presentation_id, row=row,
        )
        manifest = self._presentation_manifest_payload(
            context_id=context_id, operation_id=scope["producing_operation_id"],
            owner=owner, fence=fence, clock=None, row=row, body=body,
            transaction=transaction,
        )
        return manifest

    @staticmethod
    def _presentation_turn_body(turn: Mapping[str, Any], scope: Mapping[str, str]) -> str:
        if (turn.get("turn_id") != scope["answer_turn_id"]
                or turn.get("conversation_id") != scope["conversation_id"]
                or turn.get("operation_id") != scope["producing_operation_id"]
                or turn.get("role") != "model"):
            raise ValueError("presentation must reference its producer's canonical model turn")
        parts = turn.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("presentation answer turn body is malformed")
        text_parts = []
        for part in parts:
            if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                raise ValueError("presentation answer turn body is malformed")
            text_parts.append(part["text"])
        return "".join(text_parts)

    def _read_presentation_commit_state(
        self, transaction: Any, *, context_ref: Any, operation_ref: Any,
        conversation_ref: Any, turn_ref: Any, header_ref: Any, context_id: str,
    ) -> dict[str, Any]:
        """Read publication records in one transaction before any writes."""
        context_snapshot = context_ref.get(transaction=transaction)
        operation_snapshot = operation_ref.get(transaction=transaction)
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        command_ref = None
        command = {}
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if len(parts) == 4 and parts[:3] == [SESSIONS, context_id, "commands"]:
            command_ref = _document_ref(self.client, input_ref)
            command_snapshot = command_ref.get(transaction=transaction)
            command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        conversation_snapshot = conversation_ref.get(transaction=transaction)
        turn_snapshot = turn_ref.get(transaction=transaction)
        header_snapshot = header_ref.get(transaction=transaction)
        return {
            "context_snapshot": context_snapshot,
            "operation_snapshot": operation_snapshot,
            "operation": operation,
            "command_ref": command_ref,
            "command": command,
            "conversation_snapshot": conversation_snapshot,
            "turn_snapshot": turn_snapshot,
            "header_snapshot": header_snapshot,
            "context": context_snapshot.to_dict() or {} if context_snapshot.exists else {},
        }

    def _validate_presentation_commit_state(
        self, *, state: Mapping[str, Any], queue_operation_id: str,
        context_id: str, conversation_id: str, scope: Mapping[str, str],
        validated: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any], int]:
        """Check accepted actor, current selection, and unbound answer turn."""
        context_snapshot = state["context_snapshot"]
        operation_snapshot = state["operation_snapshot"]
        operation = state["operation"]
        command_ref = state["command_ref"]
        command = state["command"]
        conversation_snapshot = state["conversation_snapshot"]
        turn_snapshot = state["turn_snapshot"]
        context = state["context"]
        pinned_turn = operation.get("presentation_answer_turn_id")
        if pinned_turn is not None and pinned_turn != scope["answer_turn_id"]:
            raise ValueError("producer operation is already bound to another answer turn")
        if (not context_snapshot.exists or not operation_snapshot.exists
                or operation.get("operation_id") != queue_operation_id
                or operation.get("state") != "running"):
            raise PermissionError("Telegram operation lease is stale")
        if (command_ref is None or command.get("context_id") != context_id
                or command.get("operation_id") != queue_operation_id
                or type(command.get("user_id")) is not int
                or command["user_id"] <= 0):
            raise PermissionError("presentation has no accepted actor-bound command")
        if not conversation_snapshot.exists or not turn_snapshot.exists:
            raise ValueError("presentation conversation or answer turn is missing")
        conversation = conversation_snapshot.to_dict() or {}
        record = ConversationRecord.from_dict(conversation)
        if record.conversation_id != conversation_id or record.kind.value != "search":
            raise ValueError("presentation parent conversation is invalid")
        if context.get("active_conversation_id") != conversation_id:
            raise ValueError("presentation conversation is not the operation's selected conversation")
        turn = turn_snapshot.to_dict() or {}
        if self._presentation_turn_body(turn, scope) != validated["body"]:
            raise ValueError("presentation body differs from its canonical answer turn")
        turn_sequence = _integer(turn.get("turn_sequence"), "turn_sequence", minimum=1)
        if turn.get("presentation_binding") is not None:
            raise ValueError("canonical answer turn is already bound to another presentation")
        return command, turn, turn_sequence

    def _prepare_presentation_commit_writes(
        self, transaction: Any, *, state: Mapping[str, Any], context_id: str,
        queue_operation_id: str, queue_owner: str, queue_fence: int,
        conversation_id: str, scope: Mapping[str, str],
        structurally_valid: Mapping[str, Any], validated: Mapping[str, Any],
        stored_payload: Mapping[str, Any], anchors: Mapping[str, str], clock: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Recheck canonical evidence and prepare immutable publication rows."""
        from services.connectors.telegram.runtime.presentation import (
            validate_presentation_manifest,
        )

        operation = state["operation"]
        context = state["context"]
        command, turn, turn_sequence = self._validate_presentation_commit_state(
            state=state, queue_operation_id=queue_operation_id,
            context_id=context_id, conversation_id=conversation_id,
            scope=scope, validated=validated,
        )
        transaction_exchanges = self._verified_presentation_exchanges(
            context_id=context_id, conversation_id=conversation_id,
            actor_id=command["user_id"], sources=validated["sources"],
            source_turn_refs=anchors, transaction=transaction,
            owner=queue_owner, fence=queue_fence, clock=clock,
        )
        transaction_validated = validate_presentation_manifest(
            structurally_valid, expected_scope=scope,
            verified_exchanges=transaction_exchanges,
        ).to_dict()
        if not self._presentation_strict_equal(transaction_validated, validated):
            raise ValueError("presentation source proof changed during preparation")
        fresh_now = _aware(
            clock() if callable(clock) else datetime.now(timezone.utc), "clock",
        )
        if not validate_context_queue_lease(
                context, queue_operation_id, owner=queue_owner, fence=queue_fence,
                now=fresh_now, expiry=context.get("lock_expires_at")):
            raise PermissionError("Telegram operation lease is stale")
        header = {
            **scope, "schema_version": 1,
            "producer_actor_id": command["user_id"],
            "answer_turn_sequence": turn_sequence,
            "body_ref": {"turn_id": scope["answer_turn_id"],
                         "turn_sequence": turn_sequence,
                         "body_sha256": validated["body_sha256"]},
            "body_sha256": validated["body_sha256"],
            "manifest_payload": dict(stored_payload),
            "source_turn_refs": dict(anchors), "created_at": fresh_now,
        }
        binding = {
            "scope": dict(scope), "body_sha256": validated["body_sha256"],
            "answer_turn_sequence": turn_sequence,
            "manifest_payload": dict(stored_payload),
            "source_turn_refs": dict(anchors), "created_at": fresh_now,
        }
        updated_turn = dict(turn)
        updated_turn["presentation_binding"] = binding
        updated_operation = dict(operation)
        updated_operation["presentation_answer_turn_id"] = scope["answer_turn_id"]
        return header, updated_turn, updated_operation

    def _commit_prepared_presentation(
        self, *, context_id: str, queue_operation_id: str, queue_owner: str,
        queue_fence: int, scope: Mapping[str, str], structurally_valid: Mapping[str, Any],
        validated: Mapping[str, Any], stored_payload: Mapping[str, Any],
        anchors: Mapping[str, str], now: datetime, clock: Any,
    ) -> dict[str, Any]:
        """Atomically publish the staged artifact, answer binding, and op pin."""
        conversation_ref = self._conversation_ref(context_id, scope["conversation_id"])
        context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
        operation_ref = context_ref.collection("operations").document(queue_operation_id)
        header_ref = conversation_ref.collection("presentations").document(scope["presentation_id"])
        turn_ref = conversation_ref.collection("turns").document(scope["answer_turn_id"])

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            state = self._read_presentation_commit_state(
                transaction, context_ref=context_ref, operation_ref=operation_ref,
                conversation_ref=conversation_ref, turn_ref=turn_ref,
                header_ref=header_ref, context_id=context_id,
            )
            header_snapshot = state["header_snapshot"]
            if header_snapshot.exists:
                concurrent = self._read_presentation_manifest_in_transaction(
                    transaction, context_id=context_id,
                    conversation_id=scope["conversation_id"],
                    presentation_id=scope["presentation_id"],
                    owner=queue_owner, fence=queue_fence, now=now,
                )
                if self._presentation_strict_equal(concurrent, validated):
                    return concurrent
                raise ValueError("presentation ID was concurrently published with different content")
            header, updated_turn, updated_operation = self._prepare_presentation_commit_writes(
                transaction, state=state, context_id=context_id,
                queue_operation_id=queue_operation_id, queue_owner=queue_owner,
                queue_fence=queue_fence, conversation_id=scope["conversation_id"],
                scope=scope, structurally_valid=structurally_valid,
                validated=validated, stored_payload=stored_payload,
                anchors=anchors, clock=clock,
            )
            transaction.set(header_ref, header)
            transaction.set(turn_ref, updated_turn)
            transaction.set(operation_ref, updated_operation)
            return dict(validated)

        return _txn(self.client.transaction(max_attempts=5))

    def save_presentation_manifest(
        self, context_id: str, manifest: Mapping[str, Any], *,
        queue_operation_id: str, queue_owner: str, queue_fence: int, now,
        verified_exchanges: Any = None, source_turn_refs: Any = None,
        clock: Any = None,
    ) -> dict[str, Any]:
        """Persist one immutable manifest linked atomically to its existing answer turn."""
        from services.connectors.telegram.runtime.presentation import (
            validate_presentation_manifest,
        )

        context_id = _identifier(context_id, "context_id")
        queue_operation_id = _identifier(queue_operation_id, "queue_operation_id")
        queue_owner = _identifier(queue_owner, "queue_owner")
        queue_fence = _integer(queue_fence, "queue_fence", minimum=1)
        now = _aware(now, "now")
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        scope = self._presentation_scope(manifest)
        if (scope["context_id"] != context_id
                or scope["producing_operation_id"] != queue_operation_id):
            raise ValueError("presentation producer scope does not match its queue operation")
        structurally_valid = validate_presentation_manifest(
            manifest, expected_scope=scope,
        ).to_dict()
        conversation_ref = self._conversation_ref(context_id, scope["conversation_id"])
        header_ref = conversation_ref.collection("presentations").document(scope["presentation_id"])
        # Exact addressed retries use the already-pinned anchors and source
        # proofs. They need no live producer selection, lease, or caller anchors.
        addressed = header_ref.get()
        if addressed.exists:
            existing = self._read_presentation_manifest_in_transaction(
                None, context_id=context_id, conversation_id=scope["conversation_id"],
                presentation_id=scope["presentation_id"],
            )
            if not self._presentation_strict_equal(existing, structurally_valid):
                raise ValueError("presentation ID is already bound to different content")
            return existing
        return self._prepare_presentation_manifest_commit(
            context_id=context_id, queue_operation_id=queue_operation_id,
            queue_owner=queue_owner, queue_fence=queue_fence, scope=scope,
            structurally_valid=structurally_valid,
            source_turn_refs=source_turn_refs, verified_exchanges=verified_exchanges,
            now=now, clock=clock,
        )

    def _prepare_presentation_manifest_commit(
        self, *, context_id: str, queue_operation_id: str, queue_owner: str,
        queue_fence: int, scope: Mapping[str, str],
        structurally_valid: Mapping[str, Any], source_turn_refs: Any,
        verified_exchanges: Any, now: datetime, clock: Any,
    ) -> dict[str, Any]:
        """Verify initial source evidence, stage its codec payload, then commit."""
        from services.connectors.telegram.runtime.presentation import (
            validate_presentation_manifest,
        )

        if source_turn_refs is not None and not isinstance(source_turn_refs, Mapping):
            raise TypeError("source_turn_refs must be a mapping")
        anchors = dict(source_turn_refs or {})
        source_ids = [item.get("source_id") for item in structurally_valid["sources"]]
        if (any(not isinstance(item, str) for item in source_ids)
                or len(source_ids) != len(set(source_ids))
                or set(anchors) != set(source_ids)):
            raise ValueError("presentation source turn anchors are incomplete")
        for turn_id in anchors.values():
            _identifier(turn_id, "source turn_id")
        operation_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/operations/{queue_operation_id}",
        )
        producer = operation_ref.get().to_dict() or {}
        pinned_answer_turn = producer.get("presentation_answer_turn_id")
        if pinned_answer_turn is not None and pinned_answer_turn != scope["answer_turn_id"]:
            raise ValueError("producer operation is already bound to another answer turn")
        producer_input = producer.get("input_ref")
        producer_parts = producer_input.split("/") if isinstance(producer_input, str) else []
        if (producer.get("operation_id") != queue_operation_id
                or len(producer_parts) != 4
                or producer_parts[:3] != [SESSIONS, context_id, "commands"]):
            raise PermissionError("presentation has no accepted actor-bound command")
        producer_command_ref = _document_ref(self.client, producer_input)
        producer_command = producer_command_ref.get().to_dict() or {}
        actor_id = producer_command.get("user_id")
        if (producer_command.get("context_id") != context_id
                or producer_command.get("operation_id") != queue_operation_id
                or type(actor_id) is not int or actor_id <= 0):
            raise PermissionError("presentation has no accepted actor-bound command")
        trusted_exchanges = self._verified_presentation_exchanges(
            context_id=context_id, conversation_id=scope["conversation_id"],
            actor_id=actor_id, sources=structurally_valid["sources"],
            source_turn_refs=anchors, owner=queue_owner, fence=queue_fence, clock=clock,
        )
        if (verified_exchanges is not None
                and not self._presentation_strict_equal(verified_exchanges, trusted_exchanges)):
            raise ValueError("presentation proofs differ from canonical ledger exchanges")
        validated = validate_presentation_manifest(
            structurally_valid, expected_scope=scope,
            verified_exchanges=trusted_exchanges,
        ).to_dict()
        body = validated["body"]
        payload = dict(validated)
        payload.pop("body")
        ledger = self._presentation_ledger(
            context_id=context_id, operation_id=queue_operation_id,
            owner=queue_owner, fence=queue_fence, clock=clock,
        )
        stored_payload = ledger._store_payload(
            "presentation-" + scope["presentation_id"], payload,
        )
        return self._commit_prepared_presentation(
            context_id=context_id, queue_operation_id=queue_operation_id,
            queue_owner=queue_owner, queue_fence=queue_fence, scope=scope,
            structurally_valid=structurally_valid, validated=validated,
            stored_payload=stored_payload, anchors=anchors, now=now, clock=clock,
        )

    def get_presentation_manifest(
        self, context_id: str, conversation_id: str, presentation_id: str,
    ) -> dict[str, Any] | None:
        """Load an immutable, addressed presentation independent of current selection/lease."""
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        presentation_id = _identifier(presentation_id, "presentation_id")
        return self._read_presentation_manifest_in_transaction(
            None, context_id=context_id, conversation_id=conversation_id,
            presentation_id=presentation_id,
        )

    def get_verified_presentation_list_ids(
        self, context_id: str, conversation_id: str, presentation_id: str,
        *, actor_id: int, now: datetime, eligible_at: datetime | None = None,
    ) -> tuple[str, ...]:
        """Return only list IDs whose own rendered spans have delivery proof."""
        if type(actor_id) is not int or actor_id <= 0:
            return ()
        manifest = self.get_presentation_manifest(
            context_id, conversation_id, presentation_id,
        )
        if not isinstance(manifest, Mapping):
            return ()
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        header_snapshot = conversation_ref.collection("presentations").document(
            presentation_id,
        ).get()
        header = header_snapshot.to_dict() or {} if header_snapshot.exists else {}
        if (header.get("producer_actor_id") != actor_id
                or type(header.get("producer_actor_id")) is not int):
            return ()
        try:
            chunks, list_spans = self._presentation_chunk_plan(manifest)
            prepared = manifest.get("notification_refs", [])
            _indexes, revisions = self._presentation_notification_indexes(manifest, chunks)
            proof_by_index = self._read_presentation_delivery_proofs(
                None, context_id=context_id, conversation_id=conversation_id,
                presentation_id=presentation_id, prepared=prepared,
                revisions=revisions, chunks=chunks,
                header_created_at=self._presentation_utc_timestamp(
                    header.get("created_at"), "created_at",
                ), now=self._presentation_utc_timestamp(now, "now"),
            )
            if eligible_at is not None:
                cutoff = self._presentation_utc_timestamp(eligible_at, "eligible_at")
                proof_by_index = {
                    index: proof for index, proof in proof_by_index.items()
                    if proof.get("delivered_at") <= cutoff
                }
            return tuple(self._complete_presentation_lists(
                list_spans, chunks, proof_by_index,
            ))
        except (KeyError, TypeError, ValueError):
            return ()

    @staticmethod
    def _presentation_chunk_plan(manifest: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, int, int]]]:
        from services.connectors.telegram.runtime.presentation import _render

        body = manifest.get("body")
        blocks = manifest.get("blocks")
        if not isinstance(body, str) or not isinstance(blocks, list):
            raise ValueError("presentation body or blocks are malformed")
        chunks = [body[index:index + 4000] for index in range(0, len(body), 4000)]
        rendered = [_render([block]) for block in blocks]
        if "\n\n".join(rendered) != body:
            raise ValueError("presentation block renderer does not match its body")
        spans = []
        offset = 0
        for index, (block, text) in enumerate(zip(blocks, rendered)):
            end = offset + len(text)
            if block.get("type") == "document_list":
                spans.append((block["list_id"], offset, end))
            offset = end + (2 if index + 1 < len(rendered) else 0)
        return chunks, spans

    @staticmethod
    def _presentation_delivery_proof_valid(
        proof: Any, *, context_id: str, conversation_id: str, presentation_id: str,
        notification_id: str, revision: int, chunk_index: int, chunk: str,
        header_created_at: datetime, now: datetime,
    ) -> bool:
        if not isinstance(proof, Mapping) or set(proof) != {
                "context_id", "conversation_id", "presentation_id", "notification_id",
                "revision", "chunk_index", "chunk_start", "chunk_end", "body_sha256",
                "message_id", "delivered_at"}:
            return False
        delivered_at = proof.get("delivered_at")
        return (
            proof.get("context_id") == context_id
            and proof.get("conversation_id") == conversation_id
            and proof.get("presentation_id") == presentation_id
            and proof.get("notification_id") == notification_id
            and type(proof.get("revision")) is int and proof.get("revision") == revision
            and type(proof.get("chunk_index")) is int and proof.get("chunk_index") == chunk_index
            and type(proof.get("chunk_start")) is int and proof.get("chunk_start") == chunk_index * 4000
            and type(proof.get("chunk_end")) is int
            and proof.get("chunk_end") == chunk_index * 4000 + len(chunk)
            and proof.get("body_sha256") == hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            and type(proof.get("message_id")) is int and proof.get("message_id") > 0
            and isinstance(delivered_at, datetime)
            and delivered_at.tzinfo is not None and delivered_at.utcoffset() is not None
            and delivered_at.utcoffset().total_seconds() == 0
            and delivered_at >= header_created_at
            and delivered_at <= now
        )

    @staticmethod
    def _presentation_notification_indexes(
        manifest: Mapping[str, Any], chunks: list[str],
    ) -> tuple[dict[str, int], dict[str, int]]:
        prepared = manifest.get("notification_refs", [])
        revisions = manifest.get("revision_refs", [])
        if (not isinstance(prepared, list) or len(prepared) != len(chunks)
                or not isinstance(revisions, list)):
            raise ValueError("prepared presentation notifications do not match body chunks")
        for ref in [*prepared, *revisions]:
            if not isinstance(ref, Mapping) or set(ref) != {"notification_id", "revision"}:
                raise ValueError("prepared presentation notification reference is malformed")
            _identifier(ref.get("notification_id"), "notification_id")
            _integer(ref.get("revision"), "notification revision", minimum=1)
        identities = [ref["notification_id"] for ref in prepared]
        if len(set(identities)) != len(identities):
            raise ValueError("prepared presentation notification IDs are ambiguous")
        index_by_id = {identity: index for index, identity in enumerate(identities)}
        if any(ref["notification_id"] not in index_by_id for ref in revisions):
            raise ValueError("presentation revision ref names an unprepared notification")
        revision_by_id = {ref["notification_id"]: ref["revision"] for ref in prepared}
        for identity in identities:
            variants = {ref["revision"] for ref in revisions
                        if ref["notification_id"] == identity}
            if len(variants) > 1:
                raise ValueError("presentation notification revisions are ambiguous")
            if variants:
                revision_by_id[identity] = next(iter(variants))
        return index_by_id, revision_by_id

    @staticmethod
    def _presentation_notification_is_delivered(
        notification: Mapping[str, Any], *, context: Mapping[str, Any], scope: Mapping[str, Any],
        notification_id: str, revision: int, body: str, message_id: int,
        delivered_at: datetime,
    ) -> bool:
        """Check the mutable notification row against the exact settled chunk."""
        created_at = notification.get("created_at")
        chat_id = notification.get("chat_id")
        thread_id = notification.get("thread_id")
        expected_chat = context.get("chat_id")
        expected_thread = context.get("thread_id")
        chat_matches = (type(chat_id) is int and type(expected_chat) is int
                        and chat_id == expected_chat)
        thread_matches = (
            thread_id is None and expected_thread is None
            or type(thread_id) is int and type(expected_thread) is int
            and thread_id == expected_thread
        )
        created_is_aware = (
            isinstance(created_at, datetime) and created_at.tzinfo is not None
            and created_at.utcoffset() is not None
        )
        return (
            notification.get("notification_id") == notification_id
            and notification.get("context_id") == scope["context_id"]
            and notification.get("operation_id") == scope["producing_operation_id"]
            and notification.get("state") == "delivered"
            and type(notification.get("revision")) is int
            and notification.get("revision") == revision
            and type(notification.get("desired_status_revision")) is int
            and notification.get("desired_status_revision") == revision
            and type(notification.get("attempt_revision")) is int
            and notification.get("attempt_revision") == revision
            and notification.get("kind") in {"progress", "final"}
            and notification.get("body") == body
            and type(notification.get("message_id")) is int
            and notification.get("message_id") == message_id
            and notification.get("delivered_at") == delivered_at
            and created_is_aware and created_at <= delivered_at
            and delivered_at.utcoffset() is not None
            and delivered_at.utcoffset().total_seconds() == 0
            and notification.get("last_outcome") in {"sent", "edited"}
            and chat_matches and thread_matches
        )

    def _read_presentation_delivery_proofs(
        self, transaction: Any, *, context_id: str, conversation_id: str,
        presentation_id: str, prepared: list[Mapping[str, Any]],
        revisions: Mapping[str, int], chunks: list[str], header_created_at: datetime,
        now: datetime,
    ) -> dict[int, Mapping[str, Any]]:
        """Load and validate prior immutable per-chunk delivery evidence."""
        rows: list[Any] = []
        for entry in prepared:
            notification_id = entry["notification_id"]
            revision = revisions[notification_id]
            proof_id = hashlib.sha256(
                f"{context_id}\0{conversation_id}\0{presentation_id}\0{revision}".encode(),
            ).hexdigest()
            proof_ref = self.client.collection(NOTIFICATIONS).document(notification_id).collection(
                "presentation_delivery_proofs",
            ).document(proof_id)
            rows.append(proof_ref.get(transaction=transaction))
        proofs: dict[int, Mapping[str, Any]] = {}
        for index, snapshot in enumerate(rows):
            if not snapshot.exists:
                continue
            notification_id = prepared[index]["notification_id"]
            proof = snapshot.to_dict() or {}
            if not self._presentation_delivery_proof_valid(
                    proof, context_id=context_id, conversation_id=conversation_id,
                    presentation_id=presentation_id, notification_id=notification_id,
                    revision=revisions[notification_id], chunk_index=index,
                    chunk=chunks[index], header_created_at=header_created_at, now=now):
                raise ValueError("historical presentation delivery proof is malformed")
            proofs[index] = proof
        return proofs

    @staticmethod
    def _complete_presentation_lists(
        list_spans: list[tuple[str, int, int]], chunks: list[str],
        proof_by_index: Mapping[int, Any],
    ) -> list[str]:
        complete: list[str] = []
        for list_id, start, end in list_spans:
            indexes = [index for index, chunk in enumerate(chunks)
                       if index * 4000 < end and index * 4000 + len(chunk) > start]
            if indexes and all(index in proof_by_index for index in indexes):
                complete.append(list_id)
        return complete

    def _prepare_presentation_delivery_evidence(
        self, transaction: Any, *, context_id: str, conversation_id: str,
        presentation_id: str, notification_id: str, revision: int, body: str,
        body_sha256: str, message_id: int, delivered_at: datetime, now: datetime,
        context: Mapping[str, Any], conversation: Mapping[str, Any],
        header: Mapping[str, Any], evidence_ref: Any,
    ) -> dict[str, Any] | None:
        """Resolve addressed chunk identity and handle immutable proof replay."""
        conversation_record = ConversationRecord.from_dict(conversation)
        if conversation_record.conversation_id != conversation_id:
            raise ValueError("presentation parent conversation is malformed")
        scope = self._presentation_scope(header)
        if (scope["context_id"] != context_id
                or scope["conversation_id"] != conversation_id
                or scope["presentation_id"] != presentation_id
                or header.get("schema_version") != 1
                or type(header.get("schema_version")) is not int):
            raise ValueError("presentation delivery scope is malformed")
        manifest = self._read_presentation_manifest_in_transaction(
            transaction, context_id=context_id, conversation_id=conversation_id,
            presentation_id=presentation_id,
        )
        if manifest is None:
            return None
        header_created_at = self._presentation_utc_timestamp(
            header.get("created_at"), "created_at",
        )
        if delivered_at < header_created_at:
            return None
        chunks, list_spans = self._presentation_chunk_plan(manifest)
        try:
            index_by_id, revision_by_id = self._presentation_notification_indexes(
                manifest, chunks,
            )
        except (KeyError, TypeError, ValueError):
            return None
        prepared = manifest.get("notification_refs", [])
        if (notification_id not in index_by_id
                or revision_by_id[notification_id] != revision):
            return None
        chunk_index = index_by_id[notification_id]
        expected_chunk = chunks[chunk_index]
        if body != expected_chunk:
            return None
        proof = {
            "context_id": context_id,
            "conversation_id": conversation_id,
            "presentation_id": presentation_id,
            "notification_id": notification_id,
            "revision": revision,
            "chunk_index": chunk_index,
            "chunk_start": chunk_index * 4000,
            "chunk_end": chunk_index * 4000 + len(expected_chunk),
            "body_sha256": body_sha256,
            "message_id": message_id,
            "delivered_at": delivered_at,
        }
        evidence_snapshot = evidence_ref.get(transaction=transaction)
        existing_proof = evidence_snapshot.to_dict() if evidence_snapshot.exists else None
        if existing_proof is not None:
            if not self._presentation_strict_equal(existing_proof, proof):
                raise ValueError("presentation delivery evidence is immutable")
            return {"replay": True}
        return {
            "replay": False, "scope": scope, "manifest": manifest,
            "header_created_at": header_created_at, "chunks": chunks,
            "list_spans": list_spans, "prepared": prepared,
            "revision_by_id": revision_by_id, "chunk_index": chunk_index,
            "expected_chunk": expected_chunk, "proof": proof,
        }

    def _presentation_delivery_row_matches(
        self, *, notification: Mapping[str, Any], context: Mapping[str, Any],
        scope: Mapping[str, Any], notification_id: str, revision: int,
        body: str, message_id: int, delivered_at: datetime, header: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> bool:
        """Apply the actual delivered notification row and body-reference CAS."""
        if not self._presentation_notification_is_delivered(
                notification, context=context, scope=scope,
                notification_id=notification_id, revision=revision,
                body=body, message_id=message_id, delivered_at=delivered_at):
            return False
        body_ref = header.get("body_ref")
        if (not isinstance(body_ref, Mapping)
                or body_ref.get("body_sha256") != manifest.get("body_sha256")):
            raise ValueError("presentation body reference is malformed")
        return True

    def _presentation_delivery_pointer_state(
        self, transaction: Any, *, conversation_ref: Any,
        conversation: Mapping[str, Any], presentation_id: str, header: Mapping[str, Any],
    ) -> tuple[int, int]:
        """Read the prior pointer sequence before deciding whether this can advance it."""
        old_presentation_id = conversation.get("last_presentation_id")
        previous_sequence = 0
        if isinstance(old_presentation_id, str) and old_presentation_id != presentation_id:
            previous_ref = conversation_ref.collection("presentations").document(
                old_presentation_id,
            )
            previous_snapshot = previous_ref.get(transaction=transaction)
            if not previous_snapshot.exists:
                raise ValueError("current presentation pointer is malformed")
            previous = previous_snapshot.to_dict() or {}
            previous_sequence = _integer(
                previous.get("answer_turn_sequence"), "answer_turn_sequence", minimum=1,
            )
        new_sequence = _integer(
            header.get("answer_turn_sequence"), "answer_turn_sequence", minimum=1,
        )
        return previous_sequence, new_sequence

    @staticmethod
    def _write_presentation_delivery_result(
        transaction: Any, *, evidence_ref: Any, proof: Mapping[str, Any],
        conversation_ref: Any, conversation: Mapping[str, Any],
        presentation_id: str, list_spans: list[tuple[str, int, int]],
        fully_shown: list[str], previous_sequence: int, new_sequence: int,
    ) -> bool:
        """Persist immutable proof and advance the shown pointer only when eligible."""
        transaction.set(evidence_ref, dict(proof))
        if new_sequence <= previous_sequence or not fully_shown:
            return True
        updated = dict(conversation)
        updated["last_presentation_id"] = presentation_id
        updated["last_presented_list_id"] = (
            fully_shown[0] if len(list_spans) == 1 and len(fully_shown) == 1 else None
        )
        transaction.set(conversation_ref, updated)
        return True

    @staticmethod
    def _validate_presentation_delivery_input(
        context_id: str, conversation_id: str, presentation_id: str, *,
        notification_id: str, revision: Any, body: Any, body_sha256: Any,
        message_id: Any, delivered_at: Any, now: Any,
    ) -> tuple[str, str, str, str, int, str, str, int, datetime, datetime]:
        """Validate public scalar values and the exact delivered body digest."""
        import hashlib

        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        presentation_id = _identifier(presentation_id, "presentation_id")
        notification_id = _identifier(notification_id, "notification_id")
        revision = _integer(revision, "revision", minimum=1)
        message_id = _integer(message_id, "message_id", minimum=1)
        delivered_at = _aware(delivered_at, "delivered_at")
        now = _aware(now, "now")
        if delivered_at > now:
            raise ValueError("notification delivery time is in the future")
        if not isinstance(body, str) or not isinstance(body_sha256, str):
            raise TypeError("notification body and SHA-256 must be supplied")
        actual_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual_digest != body_sha256:
            raise ValueError("notification body digest does not match")
        return (context_id, conversation_id, presentation_id, notification_id,
                revision, body, body_sha256, message_id, delivered_at, now)

    def publish_presentation_delivery(
        self, context_id: str, conversation_id: str, presentation_id: str, *,
        notification_id: str, revision: int, body: str, body_sha256: str,
        message_id: int, delivered_at, now,
    ) -> bool:
        """Record actual notification delivery and publish fully shown list identity.

        The notification ledger row, not caller-supplied delivery claims, is the
        authority. Delivery evidence is immutable per notification revision so
        a later edit/retry cannot erase what the user already saw.
        """
        (context_id, conversation_id, presentation_id, notification_id,
         revision, body, body_sha256, message_id, delivered_at, now) = (
            self._validate_presentation_delivery_input(
                context_id, conversation_id, presentation_id,
                notification_id=notification_id, revision=revision, body=body,
                body_sha256=body_sha256, message_id=message_id,
                delivered_at=delivered_at, now=now,
            )
        )
        import hashlib

        context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        presentation_ref = conversation_ref.collection("presentations").document(presentation_id)
        notification_ref = self.client.collection(NOTIFICATIONS).document(notification_id)
        proof_id = hashlib.sha256(
            f"{context_id}\0{conversation_id}\0{presentation_id}\0{revision}".encode(),
        ).hexdigest()
        evidence_ref = notification_ref.collection("presentation_delivery_proofs").document(
            proof_id,
        )

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            conversation_snapshot = conversation_ref.get(transaction=transaction)
            presentation_snapshot = presentation_ref.get(transaction=transaction)
            notification_snapshot = notification_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            conversation = conversation_snapshot.to_dict() or {} if conversation_snapshot.exists else {}
            header = presentation_snapshot.to_dict() or {} if presentation_snapshot.exists else {}
            notification = notification_snapshot.to_dict() or {} if notification_snapshot.exists else {}
            if (not context_snapshot.exists or not conversation_snapshot.exists
                    or not presentation_snapshot.exists or not notification_snapshot.exists):
                return False
            prepared = self._prepare_presentation_delivery_evidence(
                transaction, context_id=context_id, conversation_id=conversation_id,
                presentation_id=presentation_id, notification_id=notification_id,
                revision=revision, body=body, body_sha256=body_sha256,
                message_id=message_id, delivered_at=delivered_at, now=now,
                context=context, conversation=conversation, header=header,
                evidence_ref=evidence_ref,
            )
            if prepared is None:
                return False
            if prepared["replay"]:
                return True
            scope = prepared["scope"]
            manifest = prepared["manifest"]
            if not self._presentation_delivery_row_matches(
                    notification=notification, context=context, scope=scope,
                    notification_id=notification_id, revision=revision,
                    body=body, message_id=message_id, delivered_at=delivered_at,
                    header=header, manifest=manifest):
                return False

            proof_by_index = self._read_presentation_delivery_proofs(
                transaction, context_id=context_id, conversation_id=conversation_id,
                presentation_id=presentation_id, prepared=prepared["prepared"],
                revisions=prepared["revision_by_id"], chunks=prepared["chunks"],
                header_created_at=prepared["header_created_at"], now=now,
            )
            proof_by_index[prepared["chunk_index"]] = prepared["proof"]
            fully_shown = self._complete_presentation_lists(
                prepared["list_spans"], prepared["chunks"], proof_by_index,
            )
            previous_sequence, new_sequence = self._presentation_delivery_pointer_state(
                transaction, conversation_ref=conversation_ref,
                conversation=conversation, presentation_id=presentation_id, header=header,
            )
            return self._write_presentation_delivery_result(
                transaction, evidence_ref=evidence_ref, proof=prepared["proof"],
                conversation_ref=conversation_ref, conversation=conversation,
                presentation_id=presentation_id, list_spans=prepared["list_spans"],
                fully_shown=fully_shown, previous_sequence=previous_sequence,
                new_sequence=new_sequence,
            )

        return bool(_txn(self.client.transaction(max_attempts=5)))

    def observe_published_presentation_in_transaction(
        self, transaction: Any, *, context_id: str, context: Mapping[str, Any],
        actor_id: int, now: Any = None,
    ) -> str | None:
        """Return only the selected conversation's confirmed presentation identity."""
        if type(actor_id) is not int or actor_id <= 0:
            return None
        conversation_id = context.get("active_conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            return None
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        conversation_snapshot = conversation_ref.get(transaction=transaction)
        if not conversation_snapshot.exists:
            return None
        conversation = conversation_snapshot.to_dict() or {}
        try:
            record = ConversationRecord.from_dict(conversation)
        except (KeyError, TypeError, ValueError):
            return None
        if record.conversation_id != conversation_id:
            return None
        presentation_id = conversation.get("last_presentation_id")
        if not isinstance(presentation_id, str) or not presentation_id:
            return None
        try:
            manifest = self._read_presentation_manifest_in_transaction(
                transaction, context_id=context_id, conversation_id=conversation_id,
                presentation_id=presentation_id,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if manifest is None:
            return None
        header_ref = conversation_ref.collection("presentations").document(presentation_id)
        header_snapshot = header_ref.get(transaction=transaction)
        header = header_snapshot.to_dict() or {} if header_snapshot.exists else {}
        if (type(header.get("producer_actor_id")) is not int
                or header.get("producer_actor_id") != actor_id
                or type(header.get("producer_actor_id")) is not int
                or manifest.get("context_id") != context_id
                or manifest.get("conversation_id") != conversation_id
                or manifest.get("presentation_id") != presentation_id):
            return None
        chunks, list_spans = self._presentation_chunk_plan(manifest)
        if not list_spans:
            return None
        prepared = manifest.get("notification_refs", [])
        try:
            _indexes, effective_revisions = self._presentation_notification_indexes(
                manifest, chunks,
            )
        except (KeyError, TypeError, ValueError):
            return None
        now = _aware(now if now is not None else datetime.now(timezone.utc), "now")
        try:
            proof_by_index = self._read_presentation_delivery_proofs(
                transaction, context_id=context_id, conversation_id=conversation_id,
                presentation_id=presentation_id, prepared=prepared,
                revisions=effective_revisions, chunks=chunks,
                header_created_at=self._presentation_utc_timestamp(
                    header.get("created_at"), "created_at",
                ), now=now,
            )
        except (KeyError, TypeError, ValueError):
            return None
        complete_lists = self._complete_presentation_lists(
            list_spans, chunks, proof_by_index,
        )
        pointer_list = conversation.get("last_presented_list_id")
        if len(list_spans) > 1:
            if pointer_list is not None or not complete_lists:
                return None
        elif (pointer_list is None or pointer_list != list_spans[0][0]
              or pointer_list not in complete_lists):
            return None
        return presentation_id

    def list_turns(self, context_id: str, conversation_id: str) -> list[dict[str, Any]]:
        context_id = _identifier(context_id, "context_id")
        conversation_id = _identifier(conversation_id, "conversation_id")
        conversation_ref = self._conversation_ref(context_id, conversation_id)
        if conversation_ref is None:
            return []
        turns_ref = conversation_ref.collection("turns")
        rows = [snapshot.to_dict() or {} for snapshot in turns_ref.stream()]
        return sorted(rows, key=lambda row: _integer(row.get("turn_sequence"), "turn_sequence", minimum=1))
