"""Transactional Telegram forwarded-message intake persistence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.outbox import OutboxTargetKind, create_outbox_record_in_transaction
from services.connectors.telegram.models import (
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
    TelegramTranscriptionState,
    build_context_id,
)
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    SESSIONS,
    TelegramIntakeResult,
    _SOURCE_FIELDS,
    _context_defaults,
    _document_ref,
    _draft_data,
    _identifier,
    _integer,
    _receipt_result,
    _source_data,
    _aware,
)

@dataclass(slots=True)
class _ForwardContext:
    data: dict[str, Any]
    active_ref: Any | None
    active_draft_id: str | None
    draft_data: dict[str, Any]
    busy: bool


@dataclass(slots=True)
class _ForwardDraft:
    is_new: bool
    ref: Any
    draft_id: str
    dialog_number: int
    conversation_id: str
    data: dict[str, Any]
    dialog_ref: Any | None = None
    dialog_data: dict[str, Any] | None = None


@dataclass(slots=True)
class _ForwardPreparation:
    source: TelegramSourceMessage
    message_ref: Any
    notification_ref: Any | None
    notification: dict[str, Any] | None
    task: Any | None
    notification_id: str | None
    outbox_id: str | None


def _load_forward_context(
    client: Any,
    transaction: Any,
    context_ref: Any,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> _ForwardContext:
    """Load context and active draft state before the transaction writes."""
    snapshot = context_ref.get(transaction=transaction)
    context = snapshot.to_dict() or {}
    for field_name, value in _context_defaults(chat_id, thread_id, topic_name, now).items():
        context.setdefault(field_name, value)
    active_draft_id = context.get("active_draft_id")
    active_ref = (
        _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{active_draft_id}")
        if active_draft_id else None
    )
    active_snap = active_ref.get(transaction=transaction) if active_ref is not None else None
    draft_data = (active_snap.to_dict() or {}) if active_snap is not None and active_snap.exists else {}
    if active_draft_id and draft_data.get("state") != TelegramDraftState.COLLECTING.value:
        active_draft_id, active_ref, draft_data = None, None, {}
    return _ForwardContext(
        context, active_ref, active_draft_id, draft_data,
        bool(context.get("reserved_operation_id") or context.get("active_operation_id")),
    )


def _persist_busy_receipt(
    transaction: Any,
    receipt_ref: Any,
    context_ref: Any,
    context: dict[str, Any],
    *,
    update_id: int,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> "TelegramIntakeResult":
    """Record a rejected update while a context operation holds the lock."""
    receipt = {"update_id": update_id, "context_id": context_id, "accepted": False,
               "draft_id": None, "dialog_number": None, "message_id": None,
               "intake_sequence": None, "notification_id": None, "outbox_id": None,
               "created_at": now}
    context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    transaction.set(receipt_ref, receipt)
    transaction.set(context_ref, context)
    return TelegramIntakeResult(False, context_id)


def _select_forward_draft(
    client: Any,
    transaction: Any,
    context: dict[str, Any],
    loaded: _ForwardContext,
    *,
    context_id: str,
    update_id: int,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> _ForwardDraft:
    """Select a collecting draft or allocate a new dialog and draft."""
    if loaded.active_ref is not None:
        draft_data = loaded.draft_data
        draft_id = loaded.active_draft_id
        dialog_number = _integer(draft_data.get("dialog_number"), "dialog_number", minimum=1)
        conversation_id = str(draft_data.get("conversation_id", f"{context_id}-{dialog_number}"))
        return _ForwardDraft(False, loaded.active_ref, draft_id, dialog_number,
                             conversation_id, draft_data)

    dialog_number = _integer(context.get("next_dialog_number", 1), "next_dialog_number", minimum=1)
    draft_id = "draft-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
    conversation_id = f"{context_id}-{dialog_number}"
    draft_ref = _document_ref(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    dialog_ref = _document_ref(client, f"{SESSIONS}/{context_id}/dialogs/{dialog_number}")
    if dialog_ref.get(transaction=transaction).exists:
        raise RuntimeError("dialog allocator points at an existing dialog")
    draft_data = _draft_data(TelegramDraft(
        draft_id, dialog_number, TelegramDraftState.COLLECTING, conversation_id, now,
    ))
    context.update({"chat_id": chat_id, "thread_id": thread_id,
                    "active_draft_id": draft_id, "draft_status": "collecting",
                    "next_dialog_number": dialog_number + 1, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    dialog_data = {"dialog_number": dialog_number, "conversation_id": conversation_id,
                   "draft_id": draft_id, "doc_id": None,
                   "created_at": now, "updated_at": now}
    return _ForwardDraft(True, draft_ref, draft_id, dialog_number,
                         conversation_id, draft_data, dialog_ref, dialog_data)


def _build_forward_source(
    message: Mapping[str, object],
    *,
    message_id: int,
    intake_sequence: int,
    transcription_state: TelegramTranscriptionState,
    now: datetime,
    original_date: datetime | None,
) -> TelegramSourceMessage:
    """Build the durable normalized source-message record."""
    return TelegramSourceMessage(
        message_id=message_id, intake_sequence=intake_sequence, received_at=now,
        original_date=original_date,
        source_chat_id=message.get("source_chat_id"),
        source_message_id=message.get("source_message_id"),
        source_user_id=message.get("source_user_id"),
        reliable_author_key=message.get("reliable_author_key"),
        display_speaker=message.get("display_speaker"), text=message.get("text"),
        caption=message.get("caption"), audio_file_id=message.get("audio_file_id"),
        audio_file_unique_id=message.get("audio_file_unique_id"),
        audio_mime_type=message.get("audio_mime_type"),
        audio_file_size=message.get("audio_file_size"),
        transcription_state=transcription_state,
        transcription_text=message.get("transcription_text"),
        transcription_ref=message.get("transcription_ref"),
        transcription_error=message.get("transcription_error"),
    )


def _prepare_forward_persistence(
    client: Any,
    transaction: Any,
    draft: _ForwardDraft,
    message: Mapping[str, object],
    *,
    context_id: str,
    delivery_chat_id: int,
    delivery_thread_id: int | None,
    message_id: int,
    intake_sequence: int,
    transcription_state: TelegramTranscriptionState,
    original_date: datetime | None,
    now: datetime,
    notification_id: str,
    outbox_id: str,
) -> _ForwardPreparation:
    """Read message and delivery identities before creating any transaction writes."""
    source = _build_forward_source(
        message, message_id=message_id, intake_sequence=intake_sequence,
        transcription_state=transcription_state, now=now,
        original_date=original_date,
    )
    message_ref = _document_ref(
        client, f"{SESSIONS}/{context_id}/drafts/{draft.draft_id}/messages/{message_id}",
    )
    if message_ref.get(transaction=transaction).exists:
        raise ValueError("message ID already exists in the active draft")

    if not draft.is_new:
        return _ForwardPreparation(source, message_ref, None, None, None, None, None)

    notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
    if notification_ref.get(transaction=transaction).exists:
        raise RuntimeError("notification ID already exists without its update receipt")
    task = create_outbox_record_in_transaction(
        client, transaction, outbox_id,
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
        schedule_at=now, now=now, context_id=context_id,
        notification_id=notification_id,
    )
    notification = {
        "notification_id": notification_id, "context_id": context_id,
        "kind": "draft_ack",
        "body": f"Dialog #{draft.dialog_number}: Message received",
        "chat_id": delivery_chat_id, "thread_id": delivery_thread_id or None,
        "draft_id": draft.draft_id, "dialog_number": draft.dialog_number,
        "operation_id": None, "desired_status_revision": 1,
        "message_id": None, "state": "pending", "revision": 1,
        "send_attempts_started": 0, "edit_attempts_started": 0, "unknown_send_outcome": False,
        "in_flight_attempt_id": None,
        "attempt_expires_at": None, "last_outcome": None, "retry_not_before": None,
        "terminal_failure": None, "last_error": None, "delivered_at": None, "failed_at": None,
        "created_at": now, "updated_at": now,
    }
    return _ForwardPreparation(
        source, message_ref, notification_ref, notification, task,
        notification_id, outbox_id,
    )


def _persist_accepted_forward(
    transaction: Any,
    *,
    receipt_ref: Any,
    context_ref: Any,
    message_ref: Any,
    notification_ref: Any | None,
    draft: _ForwardDraft,
    context: dict[str, Any],
    source: TelegramSourceMessage,
    update_id: int,
    context_id: str,
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    intake_sequence: int,
    message_id: int,
    now: datetime,
    notification_id: str | None,
    outbox_id: str | None,
    notification: dict[str, Any] | None,
    task: Any | None,
) -> "TelegramIntakeResult":
    """Persist an accepted forward and its receipt atomically."""
    if draft.is_new:
        transaction.set(draft.dialog_ref, draft.dialog_data)
        transaction.set(draft.ref, draft.data)
        transaction.set(notification_ref, notification)
    transaction.set(message_ref, _source_data(source))
    draft.data["ordered_message_ids"] = list(draft.data.get("ordered_message_ids", [])) + [message_id]
    transaction.set(draft.ref, draft.data)
    context.update({"chat_id": chat_id, "thread_id": thread_id,
                    "active_draft_id": draft.draft_id, "draft_status": "collecting",
                    "next_intake_sequence": intake_sequence + 1, "updated_at": now})
    selected_target = (draft.conversation_id, draft.dialog_number)
    current_target = (context.get("active_conversation_id"), context.get("selected_dialog_number"))
    context["active_conversation_id"], context["selected_dialog_number"] = selected_target
    if current_target != selected_target:
        selection_sequence = _integer(context.get("selection_sequence", 0), "selection_sequence")
        context["selection_sequence"] = selection_sequence + 1
    if topic_name is not None:
        context["topic_name"] = topic_name
    transaction.set(context_ref, context)
    receipt = {"update_id": update_id, "context_id": context_id, "accepted": True,
               "draft_id": draft.draft_id, "dialog_number": draft.dialog_number,
               "message_id": message_id, "intake_sequence": intake_sequence,
               "notification_id": notification_id,
               "outbox_id": task.outbox_id if task is not None else None,
               "created_at": now}
    transaction.set(receipt_ref, receipt)
    return TelegramIntakeResult(
        True, context_id, draft.draft_id, draft.dialog_number, message_id,
        intake_sequence, notification_id, task.outbox_id if task is not None else None,
    )


class _TelegramIntakeMixin:
    """Persist forwarded source messages and read intake state."""

    def record_topic_event(
        self,
        update_id: int,
        *,
        chat_id: int,
        thread_id: int | None,
        event_kind: str,
        topic_name: str | None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Persist topic metadata and a minimal idempotency receipt only."""
        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        if thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if event_kind not in {"forum_topic_created", "forum_topic_edited"}:
            raise ValueError("event_kind must identify a Telegram topic event")
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        now = _aware(now, "now")
        context_id = build_context_id(chat_id, thread_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            receipt_snap = receipt_ref.get(transaction=transaction)
            if receipt_snap.exists:
                return _receipt_result(receipt_snap.to_dict() or {}, context_id)

            context_snap = context_ref.get(transaction=transaction)
            context = context_snap.to_dict() or {}
            defaults = _context_defaults(chat_id, thread_id, None, now)
            for field_name, value in defaults.items():
                context.setdefault(field_name, value)
            context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
            if topic_name is not None:
                context["topic_name"] = topic_name

            receipt = {
                "update_id": update_id,
                "context_id": context_id,
                "accepted": False,
                "draft_id": None,
                "dialog_number": None,
                "message_id": None,
                "intake_sequence": None,
                "notification_id": None,
                "outbox_id": None,
                "event_kind": event_kind,
                "outcome": "topic_metadata_updated" if topic_name is not None else "topic_metadata_preserved",
                "created_at": now,
            }
            transaction.set(receipt_ref, receipt)
            transaction.set(context_ref, context)
            return TelegramIntakeResult(False, context_id)

        return _txn(self.client.transaction(max_attempts=5))

    def accept_forward(
        self,
        update_id: int,
        *,
        chat_id: int,
        thread_id: int | None = None,
        topic_name: str | None = None,
        message: Mapping[str, object],
        now: datetime,
    ) -> TelegramIntakeResult:
        update_id = _integer(update_id, "update_id")
        context_id = build_context_id(chat_id, thread_id)
        now = _aware(now, "now")
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(message, Mapping) or set(message) - _SOURCE_FIELDS:
            raise ValueError("message must contain only normalized source-message fields")
        message_id = _integer(message.get("message_id"), "message_id", minimum=1)
        original_date = message.get("original_date")
        if original_date is not None:
            original_date = _aware(original_date, "original_date")  # type: ignore[arg-type]
        source_chat_id = message.get("source_chat_id")
        if source_chat_id is not None and (type(source_chat_id) is not int or source_chat_id == 0):
            raise ValueError("source_chat_id must be a non-zero integer")
        for field_name in ("source_message_id", "source_user_id", "audio_file_size"):
            value = message.get(field_name)
            if value is not None:
                _integer(value, field_name, minimum=0 if field_name == "audio_file_size" else 1)
        for field_name in (
            "reliable_author_key", "display_speaker", "text", "caption", "audio_file_id",
            "audio_file_unique_id", "audio_mime_type", "transcription_text",
            "transcription_ref", "transcription_error",
        ):
            value = message.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or None")
        try:
            transcription_state = TelegramTranscriptionState(
                message.get("transcription_state", TelegramTranscriptionState.NOT_REQUIRED)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("transcription_state is invalid") from exc
        notification_id = "tgack-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
        outbox_id = "tgdispatch-" + hashlib.sha256(f"{context_id}:{update_id}".encode()).hexdigest()
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            receipt_snap = receipt_ref.get(transaction=transaction)
            if receipt_snap.exists:
                return _receipt_result(receipt_snap.to_dict() or {}, context_id)
            loaded = _load_forward_context(
                self.client, transaction, context_ref, context_id,
                chat_id, thread_id, topic_name, now,
            )
            if loaded.busy:
                return _persist_busy_receipt(
                    transaction, receipt_ref, context_ref, loaded.data,
                    update_id=update_id, context_id=context_id,
                    chat_id=chat_id, thread_id=thread_id,
                    topic_name=topic_name, now=now,
                )
            draft = _select_forward_draft(
                self.client, transaction, loaded.data, loaded,
                context_id=context_id, update_id=update_id,
                chat_id=chat_id, thread_id=thread_id,
                topic_name=topic_name, now=now,
            )
            intake_sequence = _integer(loaded.data.get("next_intake_sequence", 1), "next_intake_sequence", minimum=1)
            prepared = _prepare_forward_persistence(
                self.client, transaction, draft, message,
                context_id=context_id, message_id=message_id,
                delivery_chat_id=loaded.data["chat_id"],
                delivery_thread_id=loaded.data.get("thread_id"),
                intake_sequence=intake_sequence,
                transcription_state=transcription_state,
                original_date=original_date, now=now,
                notification_id=notification_id, outbox_id=outbox_id,
            )
            return _persist_accepted_forward(
                transaction, receipt_ref=receipt_ref, context_ref=context_ref,
                message_ref=prepared.message_ref,
                notification_ref=prepared.notification_ref,
                draft=draft, context=loaded.data, source=prepared.source,
                update_id=update_id, context_id=context_id,
                chat_id=chat_id, thread_id=thread_id, topic_name=topic_name,
                intake_sequence=intake_sequence, message_id=message_id, now=now,
                notification_id=prepared.notification_id, outbox_id=prepared.outbox_id,
                notification=prepared.notification, task=prepared.task,
            )

        return _txn(self.client.transaction(max_attempts=5))


    def record_unsupported_forward(
        self,
        update_id: int,
        *,
        chat_id: int,
        thread_id: int | None = None,
        topic_name: str | None = None,
        message_id: int,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Durably acknowledge a contentless/unsupported forward without a draft."""
        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        message_id = _integer(message_id, "message_id", minimum=1)
        if thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        now = _aware(now, "now")
        context_id = build_context_id(chat_id, thread_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        notification_id = "tgunsupported-" + hashlib.sha256(
            f"{context_id}:{update_id}".encode()
        ).hexdigest()
        outbox_id = "tgunsupported-dispatch-" + hashlib.sha256(
            f"{context_id}:{update_id}".encode()
        ).hexdigest()
        notification_ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> TelegramIntakeResult:
            receipt_snap = receipt_ref.get(transaction=transaction)
            if receipt_snap.exists:
                return _receipt_result(receipt_snap.to_dict() or {}, context_id)
            context_snap = context_ref.get(transaction=transaction)
            context = context_snap.to_dict() or {}
            for key, value in _context_defaults(chat_id, thread_id, topic_name, now).items():
                context.setdefault(key, value)
            context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
            if topic_name is not None:
                context["topic_name"] = topic_name
            notification = {
                "notification_id": notification_id, "context_id": context_id,
                "kind": "unsupported_forward",
                "body": "This forwarded item has no supported text or audio, so it was skipped. Send text, a caption, or audio to add content.",
                "chat_id": chat_id, "thread_id": thread_id,
                "message_id": None, "dialog_number": None, "operation_id": None,
                "state": "pending", "revision": 1, "desired_status_revision": 1,
                "send_attempts_started": 0, "edit_attempts_started": 0,
                "unknown_send_outcome": False, "in_flight_attempt_id": None,
                "attempt_expires_at": None, "last_outcome": None, "retry_not_before": None,
                "terminal_failure": None, "last_error": None, "delivered_at": None,
                "failed_at": None, "created_at": now, "updated_at": now,
            }
            task = create_outbox_record_in_transaction(
                self.client, transaction, outbox_id,
                target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
                schedule_at=now, now=now, context_id=context_id,
                notification_id=notification_id,
            )
            transaction.set(notification_ref, notification)
            transaction.set(context_ref, context)
            transaction.set(receipt_ref, {
                "update_id": update_id, "context_id": context_id,
                "accepted": True, "outcome": "unsupported_forward_skipped",
                "draft_id": None, "dialog_number": None, "message_id": message_id,
                "intake_sequence": None, "notification_id": notification_id,
                "outbox_id": task.outbox_id, "created_at": now,
            })
            return TelegramIntakeResult(
                True, context_id, message_id=message_id,
                notification_id=notification_id, outbox_id=task.outbox_id,
            )

        return _txn(self.client.transaction(max_attempts=5))


    def get_update_receipt(self, context_id: str, update_id: int) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        update_id = _integer(update_id, "update_id")
        snapshot = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}").get()
        return snapshot.to_dict() if snapshot.exists else None


    def get_context(self, context_id: str) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        snapshot = self.client.collection(SESSIONS).document(context_id).get()
        return snapshot.to_dict() if snapshot.exists else None
