"""Transactional Telegram command acceptance and queue registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.outbox import OutboxTargetKind, create_outbox_record_in_transaction
from core.integrations.firestore.operation_queue import register_operation_in_transaction
from services.connectors.telegram.models import TelegramDraftState, build_context_id
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    SESSIONS,
    TelegramIntakeResult,
    _document_ref,
    _draft_data,
    _identifier,
    _integer,
    _receipt_result,
    _aware,
    _context_defaults,
)

@dataclass(slots=True)
class _CommandIdentity:
    context_id: str
    receipt_ref: Any
    context_ref: Any
    command_ref: Any
    operation_id: str
    outbox_id: str
    input_ref: str
    identity_digest: str


@dataclass(slots=True)
class _CommandSnapshot:
    draft_id: str | None
    draft_ref: Any | None
    draft_data: dict[str, Any]
    message_ids: tuple[int, ...]
    cutoff_sequence: int
    manifest_ref: str | None
    manifest_doc_ref: Any | None
    manifest_digest: str | None
    predecessor_operation_id: str | None
    pending_save_candidate_operation_id: str | None


@dataclass(slots=True)
class _CommandAcceptanceState:
    context: dict[str, Any]
    snapshot: _CommandSnapshot
    command_data: dict[str, Any]


def _prepare_command_identity(
    client: Any,
    *,
    context_id: str,
    update_id: int,
    command: Any,
) -> _CommandIdentity:
    """Build stable command, operation, and outbox identities before the transaction."""
    update_path = f"{SESSIONS}/{context_id}/updates/{update_id}"
    command_path = f"{SESSIONS}/{context_id}/commands/{update_id}"
    identity = hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
    operation_id = "tgop-" + identity
    return _CommandIdentity(
        context_id=context_id,
        receipt_ref=_document_ref(client, update_path),
        context_ref=client.collection(SESSIONS).document(context_id),
        command_ref=_document_ref(client, command_path),
        operation_id=operation_id,
        outbox_id="tgcommand-" + identity,
        input_ref=command_path,
        identity_digest=hashlib.sha256(
            json.dumps(
                {"kind": command.kind, "text": command.text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _load_command_snapshot(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    context: Mapping[str, Any],
    operation_id: str,
    command: Any,
) -> _CommandSnapshot:
    """Read the collecting draft and immutable save cutoff before transaction writes."""
    draft_id = context.get("active_draft_id")
    draft_ref = (
        _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        if draft_id else None
    )
    draft_snapshot = draft_ref.get(transaction=transaction) if draft_ref is not None else None
    draft_data = (
        (draft_snapshot.to_dict() or {})
        if draft_snapshot is not None and draft_snapshot.exists
        else {}
    )
    if draft_data.get("state") != TelegramDraftState.COLLECTING.value:
        draft_id, draft_ref, draft_data = None, None, {}

    message_ids = tuple(int(value) for value in draft_data.get("ordered_message_ids", ()))
    cutoff_sequence = int(context.get("next_intake_sequence", 1)) - 1
    manifest_ref = (
        f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}"
        if command.is_save and draft_id and message_ids else None
    )
    manifest_digest = None
    manifest_doc_ref = None
    if manifest_ref is not None:
        manifest_digest = hashlib.sha256(json.dumps(
            {"draft_id": draft_id, "message_ids": list(message_ids)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        manifest_doc_ref = _document_ref(client, manifest_ref)
        if manifest_doc_ref.get(transaction=transaction).exists:
            raise RuntimeError("save manifest already exists without its update receipt")
    predecessor_id = context.get("queue_tail_operation_id")
    predecessor_operation_id = predecessor_id if isinstance(predecessor_id, str) and predecessor_id else None
    pending_save_candidate_operation_id = None
    pending_save_id = context.get("pending_save_operation_id")
    if isinstance(pending_save_id, str) and pending_save_id:
        pending_save_ref = _document_ref(
            client, f"{SESSIONS}/{context_id}/operations/{pending_save_id}"
        )
        pending_save_snapshot = pending_save_ref.get(transaction=transaction)
        pending_save_data = pending_save_snapshot.to_dict() or {} if pending_save_snapshot.exists else {}
        if pending_save_data.get("state") in {"pending", "running"}:
            pending_save_candidate_operation_id = pending_save_id
    return _CommandSnapshot(
        draft_id=draft_id,
        draft_ref=draft_ref,
        draft_data=draft_data,
        message_ids=message_ids,
        cutoff_sequence=cutoff_sequence,
        manifest_ref=manifest_ref,
        manifest_doc_ref=manifest_doc_ref,
        manifest_digest=manifest_digest,
        predecessor_operation_id=predecessor_operation_id,
        pending_save_candidate_operation_id=pending_save_candidate_operation_id,
    )


def _command_link_metadata(snapshot: _CommandSnapshot) -> dict[str, str]:
    """Return immutable predecessor and pending-save candidate metadata."""
    metadata = {}
    if snapshot.predecessor_operation_id is not None:
        metadata["predecessor_operation_id"] = snapshot.predecessor_operation_id
    if snapshot.pending_save_candidate_operation_id is not None:
        metadata["pending_save_candidate_operation_id"] = (
            snapshot.pending_save_candidate_operation_id
        )
    return metadata


def _command_data(
    *,
    update_id: int,
    context_id: str,
    user_id: int,
    operation_id: str,
    command: Any,
    snapshot: _CommandSnapshot,
    now: datetime,
) -> dict[str, Any]:
    """Create the command candidate record without copying draft content."""
    return {
        "update_id": update_id,
        "context_id": context_id,
        "user_id": user_id,
        "kind": command.kind,
        "candidate": command.text,
        "draft_id": snapshot.draft_id,
        "cutoff_message_ids": list(snapshot.message_ids),
        "cutoff_intake_sequence": snapshot.cutoff_sequence,
        "operation_id": operation_id,
        "created_at": now,
        **_command_link_metadata(snapshot),
    }


def _load_command_acceptance_state(
    client: Any,
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    chat_id: int,
    user_id: int,
    thread_id: int | None,
    topic_name: str | None,
    command: Any,
    now: datetime,
) -> _CommandAcceptanceState:
    """Load the context and prepare command data before reserving or writing."""
    context_snapshot = identity.context_ref.get(transaction=transaction)
    context = context_snapshot.to_dict() or {}
    for field_name, value in _context_defaults(chat_id, thread_id, topic_name, now).items():
        context.setdefault(field_name, value)
    snapshot = _load_command_snapshot(
        client,
        transaction,
        context_id=identity.context_id,
        context=context,
        operation_id=identity.operation_id,
        command=command,
    )
    return _CommandAcceptanceState(
        context=context,
        snapshot=snapshot,
        command_data=_command_data(
            update_id=update_id,
            context_id=identity.context_id,
            user_id=user_id,
            operation_id=identity.operation_id,
            command=command,
            snapshot=snapshot,
            now=now,
        ),
    )


def _register_command_operation(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    identity: _CommandIdentity,
    now: datetime,
    snapshot: _CommandSnapshot,
) -> Any:
    """Reserve the FIFO operation and create its outbox before queue writes begin."""
    def create_command_outbox() -> None:
        create_outbox_record_in_transaction(
            client,
            transaction,
            identity.outbox_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1,
            schedule_at=now,
            now=now,
            operation_id=identity.operation_id,
            context_id=identity.context_id,
        )

    return register_operation_in_transaction(
        client,
        transaction,
        context_ref,
        identity.operation_id,
        identity_digest=identity.identity_digest,
        input_ref=identity.input_ref,
        now=now,
        before_write=create_command_outbox,
        operation_fields=_command_link_metadata(snapshot),
    )


def _update_command_context(
    context: dict[str, Any],
    *,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    operation_id: str,
    sequence: int,
    now: datetime,
) -> None:
    """Record the new FIFO tail and keep its current head reserved."""
    context.update({
        "chat_id": chat_id,
        "thread_id": thread_id,
        "next_operation_sequence": sequence + 1,
        "queue_tail_operation_id": operation_id,
        "updated_at": now,
    })
    if not context.get("reserved_operation_id"):
        context.update({"reserved_operation_id": operation_id, "head_sequence": sequence})
    if topic_name is not None:
        context["topic_name"] = topic_name


def _seal_command_draft(
    transaction: Any,
    *,
    snapshot: _CommandSnapshot,
    context: dict[str, Any],
    operation_id: str,
    now: datetime,
) -> None:
    """Seal a recognized save's exact source set and detach its collecting pointer."""
    if snapshot.draft_ref is None:
        return
    snapshot.draft_data.update({
        "state": TelegramDraftState.PROCESSING.value,
        "sealed_at": now,
        "ordered_message_ids": list(snapshot.message_ids),
        "manifest_ref": snapshot.manifest_ref,
        "manifest_digest": snapshot.manifest_digest,
        "active_save_operation_id": operation_id,
        "immutable_save_intent_id": operation_id,
    })
    context.update({"active_draft_id": None, "draft_status": None})
    if snapshot.manifest_doc_ref is not None:
        transaction.set(snapshot.manifest_doc_ref, {
            "draft_id": snapshot.draft_id,
            "ordered_message_ids": list(snapshot.message_ids),
            "manifest_digest": snapshot.manifest_digest,
            "save_operation_id": operation_id,
            "sealed_at": now,
        })
    transaction.set(snapshot.draft_ref, snapshot.draft_data)


def _persist_command_acceptance(
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    command: Any,
    state: _CommandAcceptanceState,
    operation: Any,
    now: datetime,
) -> "TelegramIntakeResult":
    """Persist command, context, and replay receipt after all transaction reads."""
    snapshot = state.snapshot
    receipt = {
        "update_id": update_id,
        "context_id": identity.context_id,
        "accepted": True,
        "draft_id": snapshot.draft_id,
        "dialog_number": snapshot.draft_data.get("dialog_number"),
        "message_id": None,
        "intake_sequence": None,
        "notification_id": None,
        "outbox_id": identity.outbox_id,
        "operation_id": identity.operation_id,
        "queue_position": operation.sequence,
        "snapshot_message_ids": list(snapshot.message_ids) if command.is_save else [],
        "created_at": now,
    }
    transaction.set(identity.command_ref, state.command_data)
    transaction.set(identity.context_ref, state.context)
    transaction.set(identity.receipt_ref, receipt)
    return TelegramIntakeResult(
        True,
        identity.context_id,
        snapshot.draft_id,
        snapshot.draft_data.get("dialog_number"),
        outbox_id=identity.outbox_id,
        operation_id=identity.operation_id,
        queue_position=operation.sequence,
        snapshot_message_ids=snapshot.message_ids if command.is_save else (),
    )


class _TelegramCommandMixin:
    """Persist idempotent command acceptance and operation registration."""

    def _accept_command_in_transaction(
        self,
        transaction: Any,
        *,
        identity: _CommandIdentity,
        update_id: int,
        chat_id: int,
        user_id: int,
        thread_id: int | None,
        topic_name: str | None,
        command: Any,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Coordinate the all-reads-first command reservation and optional save seal."""
        receipt_snapshot = identity.receipt_ref.get(transaction=transaction)
        if receipt_snapshot.exists:
            return _receipt_result(receipt_snapshot.to_dict() or {}, identity.context_id)

        state = _load_command_acceptance_state(
            self.client,
            transaction,
            identity=identity,
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
            thread_id=thread_id,
            topic_name=topic_name,
            command=command,
            now=now,
        )
        operation = _register_command_operation(
            self.client,
            transaction,
            context_ref=identity.context_ref,
            identity=identity,
            now=now,
            snapshot=state.snapshot,
        )
        _update_command_context(
            state.context,
            chat_id=chat_id,
            thread_id=thread_id,
            topic_name=topic_name,
            operation_id=identity.operation_id,
            sequence=operation.sequence,
            now=now,
        )
        if command.is_save:
            state.context["pending_save_operation_id"] = identity.operation_id
            _seal_command_draft(
                transaction,
                snapshot=state.snapshot,
                context=state.context,
                operation_id=identity.operation_id,
                now=now,
            )
        return _persist_command_acceptance(
            transaction,
            identity=identity,
            update_id=update_id,
            command=command,
            state=state,
            operation=operation,
            now=now,
        )


    def accept_command(
        self,
        update_id: int,
        *,
        chat_id: int,
        user_id: int,
        thread_id: int | None = None,
        topic_name: str | None = None,
        command: Any,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Accept command text and reserve its FIFO position with a save cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        now = _aware(now, "now")
        if thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(command, TelegramCommand):
            raise TypeError("command must be a TelegramCommand")
        context_id = build_context_id(chat_id, thread_id)
        identity = _prepare_command_identity(
            self.client,
            context_id=context_id,
            update_id=update_id,
            command=command,
        )

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            return self._accept_command_in_transaction(
                transaction,
                identity=identity,
                update_id=update_id,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
                topic_name=topic_name,
                command=command,
                now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))

