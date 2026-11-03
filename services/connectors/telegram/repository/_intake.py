"""Transactional Telegram forwarded-message intake persistence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    prepare_outbox_record_in_transaction,
    write_prepared_outbox_record,
)
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
    receipt_group_ref: Any | None = None
    receipt_group: dict[str, Any] | None = None
    previous_notification_ref: Any | None = None
    previous_notification: dict[str, Any] | None = None
    additional_outbox_ids: tuple[str, ...] = ()
    receipt_close_task: Any | None = None
    receipt_group_new: bool = False


@dataclass(slots=True)
class _ReceiptGroupPreparation:
    group_ref: Any
    group: dict[str, Any]
    notification_ref: Any
    notification: dict[str, Any]
    previous_notification_ref: Any | None
    previous_notification: dict[str, Any] | None
    initial_task: Any | None
    close_task: Any
    additional_outbox_ids: tuple[str, ...]
    is_new: bool


@dataclass(slots=True)
class _ReceiptGroupSelection:
    """Resolved durable open group before its card or revision is prepared."""
    group_id: str
    group_ref: Any
    existing_group: dict[str, Any]


RECEIPT_TEXT_QUIET_SECONDS = 1


def _receipt_group_is_open_for_draft(
    group: Mapping[str, Any], draft: _ForwardDraft, now: datetime,
) -> bool:
    deadline = group.get("deadline_at")
    return (
        group.get("state") == "open"
        and isinstance(deadline, datetime) and deadline > now
        and group.get("draft_id") in (None, draft.draft_id)
    )


def _receipt_control_is_active(notification: Mapping[str, Any]) -> bool:
    """Return whether a receipt row still exposes a live Save control."""
    return (
        notification.get("receipt_card") is True
        and notification.get("receipt_terminal") is not True
        and notification.get("save_operation_id") is None
        and notification.get("state") not in {"failed", "delivery_unknown"}
        and isinstance(notification.get("receipt_control_token"), str)
        and bool(notification.get("receipt_control_token"))
    )


def _receipt_id(value: str) -> str:
    return "receipt-" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_receipt_group(
    client: Any, transaction: Any, *, context_id: str, context: dict[str, Any],
    draft: _ForwardDraft, update_id: int, media_group_id: str | None, now: datetime,
) -> _ReceiptGroupSelection:
    """Join a matching open native/text group or create a fresh group epoch."""
    if media_group_id is not None:
        seed = f"{context_id}:media:{media_group_id}"
        lookup_key = hashlib.sha256(media_group_id.encode("utf-8")).hexdigest()
        media_groups = dict(context.get("open_media_receipt_groups", {}))
        group_id = media_groups.get(lookup_key)
        group_ref = (_document_ref(client, f"{SESSIONS}/{context_id}/receipt_groups/{group_id}")
                     if isinstance(group_id, str) else None)
        snapshot = group_ref.get(transaction=transaction) if group_ref is not None else None
        existing = (snapshot.to_dict() or {} if snapshot is not None and snapshot.exists else {})
        if not _receipt_group_is_open_for_draft(existing, draft, now):
            text_id = context.get("open_text_receipt_group_id")
            text_ref = (_document_ref(client, f"{SESSIONS}/{context_id}/receipt_groups/{text_id}")
                        if isinstance(text_id, str) and text_id != group_id else None)
            text_snapshot = text_ref.get(transaction=transaction) if text_ref is not None else None
            text_group = (text_snapshot.to_dict() or {}
                          if text_snapshot is not None and text_snapshot.exists else {})
            if (_receipt_group_is_open_for_draft(text_group, draft, now)
                    and text_group.get("media_group_id") in (None, media_group_id)):
                group_id, group_ref, existing = text_id, text_ref, text_group
            else:
                group_id = None
        if group_id is None:
            group_id = _receipt_id(f"{seed}:{update_id}")
            group_ref = _document_ref(client, f"{SESSIONS}/{context_id}/receipt_groups/{group_id}")
            group_ref.get(transaction=transaction)
            existing = {}
        media_groups[lookup_key] = group_id
        context["open_media_receipt_groups"] = media_groups
    else:
        group_id = context.get("open_text_receipt_group_id")
        group_ref = (_document_ref(client, f"{SESSIONS}/{context_id}/receipt_groups/{group_id}")
                     if isinstance(group_id, str) else None)
        snapshot = group_ref.get(transaction=transaction) if group_ref is not None else None
        existing = (snapshot.to_dict() or {} if snapshot is not None and snapshot.exists else {})
        if not _receipt_group_is_open_for_draft(existing, draft, now):
            group_id = _receipt_id(f"{context_id}:text:{draft.draft_id}:{update_id}")
            group_ref = _document_ref(client, f"{SESSIONS}/{context_id}/receipt_groups/{group_id}")
            group_ref.get(transaction=transaction)
            existing = {}
    assert isinstance(group_id, str) and group_ref is not None
    return _ReceiptGroupSelection(group_id, group_ref, existing)


def _prepare_receipt_group(
    client: Any, transaction: Any, *, context_id: str, context: dict[str, Any],
    draft: _ForwardDraft, update_id: int, media_group_id: str | None,
    accepted: bool, user_id: int, now: datetime,
) -> _ReceiptGroupPreparation:
    """Read and stage one durable card group, its initial send and quiet-close wakeup."""
    selection = _select_receipt_group(
        client, transaction, context_id=context_id, context=context, draft=draft,
        update_id=update_id, media_group_id=media_group_id, now=now,
    )
    group_id, group_ref = selection.group_id, selection.group_ref
    existing_group = selection.existing_group
    notification_id = "tgcard-" + hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
    notification_snapshot = notification_ref.get(transaction=transaction)
    notification = notification_snapshot.to_dict() or {} if notification_snapshot.exists else {}
    is_new = not bool(existing_group)
    if notification and (
            notification.get("receipt_terminal") is True
            or notification.get("save_operation_id") is not None
            or notification.get("state") in {"failed", "delivery_unknown"}):
        notification_id = "tgcard-" + hashlib.sha256(
            f"{group_id}:card:{update_id}".encode("utf-8")
        ).hexdigest()
        notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
        notification_snapshot = notification_ref.get(transaction=transaction)
        notification = notification_snapshot.to_dict() or {} if notification_snapshot.exists else {}
    card_new = not bool(notification)
    previous_ref = previous_data = None
    if is_new:
        previous_id = context.get("latest_receipt_notification_id")
        if isinstance(previous_id, str) and previous_id != notification_id:
            previous_ref = client.collection(NOTIFICATIONS).document(previous_id)
            previous_snapshot = previous_ref.get(transaction=transaction)
            previous_data = previous_snapshot.to_dict() or {} if previous_snapshot.exists else None
            if previous_data is not None and not _receipt_control_is_active(previous_data):
                previous_ref, previous_data = None, None

    deadline = now + timedelta(seconds=RECEIPT_TEXT_QUIET_SECONDS)
    accepted_ids = list(existing_group.get("accepted_message_ids", []))
    skipped_count = _integer(existing_group.get("skipped_count", 0), "skipped_count")
    if accepted:
        accepted_ids.append(update_id)
    else:
        skipped_count += 1
    group = {
        **existing_group,
        "group_id": group_id, "context_id": context_id,
        "draft_id": draft.draft_id if draft.draft_id else existing_group.get("draft_id"),
        "dialog_number": draft.dialog_number if draft.draft_id else existing_group.get("dialog_number"),
        "notification_id": notification_id, "state": "open",
        "deadline_at": deadline, "accepted_message_ids": accepted_ids,
        "skipped_count": skipped_count, "owner_user_id": user_id,
        "media_group_id": media_group_id or existing_group.get("media_group_id"),
        "created_at": existing_group.get("created_at", now), "updated_at": now,
    }
    if card_new:
        body = (f"Dialog #{draft.dialog_number}: Thinking…\n\n"
                "Forward the remaining messages from this conversation, then press Save.") if draft.draft_id else (
                "This forwarded item has no supported text or audio, so it was skipped. "
                "Send text, a caption, or audio to add content.")
        notification = {
            "notification_id": notification_id, "context_id": context_id,
            "kind": "draft_ack", "receipt_card": True, "receipt_group_id": group_id,
            "receipt_deadline": deadline, "receipt_control_token": None,
            "receipt_owner_user_id": user_id, "receipt_accepted_count": len(accepted_ids),
            "receipt_skipped_count": skipped_count, "body": body,
            "receipt_ack_revision": 1 if draft.draft_id else None,
            "receipt_numbered_revision": 1 if draft.draft_id else None,
            "reply_markup": {"inline_keyboard": []},
            "chat_id": context["chat_id"], "thread_id": context.get("thread_id"),
            "draft_id": draft.draft_id if draft.draft_id else None,
            "dialog_number": draft.dialog_number if draft.draft_id else None,
            "operation_id": None, "desired_status_revision": 1,
            "message_id": None, "state": "pending", "revision": 1,
            "send_attempts_started": 0, "edit_attempts_started": 0,
            "edit_attempts_by_revision": {}, "unknown_send_outcome": False,
            "in_flight_attempt_id": None, "attempt_expires_at": None,
            "last_outcome": None, "retry_not_before": None,
            "terminal_failure": None, "last_error": None, "delivered_at": None,
            "failed_at": None, "created_at": now, "updated_at": now,
        }
    else:
        body = notification.get("body", "")
        if draft.draft_id:
            body = (f"Dialog #{draft.dialog_number}: Thinking…\n\n"
                    "Forward the remaining messages from this conversation, then press Save.")
        elif skipped_count:
            body = ("This forwarded group includes an unsupported item, which was skipped. "
                    "Send text, a caption, or audio to add content.")
        next_revision = _integer(notification.get("revision", 1), "revision", minimum=1) + 1
        notification.update({
            "draft_id": draft.draft_id if draft.draft_id else notification.get("draft_id"),
            "dialog_number": draft.dialog_number if draft.draft_id else notification.get("dialog_number"),
            "body": body, "receipt_deadline": deadline,
            "receipt_accepted_count": len(accepted_ids),
            "receipt_skipped_count": skipped_count,
            "receipt_control_token": None,
            "reply_markup": {"inline_keyboard": []},
            "revision": next_revision,
            "desired_status_revision": next_revision,
            "state": notification.get("state") if notification.get("in_flight_attempt_id") else "pending",
            "updated_at": now,
        })
        if draft.draft_id:
            notification["receipt_ack_revision"] = next_revision
            notification["receipt_numbered_revision"] = next_revision

    initial_task = None
    initial_prepared = None
    retire_prepared = None
    extra_ids: list[str] = []
    if card_new:
        initial_id = "tgcard-send-" + hashlib.sha256(notification_id.encode()).hexdigest()
        initial_prepared = prepare_outbox_record_in_transaction(
            client, transaction, initial_id,
            target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
            schedule_at=now, now=now, context_id=context_id,
            notification_id=notification_id,
        )
        if previous_ref is not None and previous_data is not None and _receipt_control_is_active(previous_data):
            retire_id = "tgcard-retire-" + hashlib.sha256(
                f"{previous_data.get('notification_id')}:{update_id}".encode()
            ).hexdigest()
            retire_prepared = prepare_outbox_record_in_transaction(
                client, transaction, retire_id,
                target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
                schedule_at=now, now=now, context_id=context_id,
                notification_id=previous_data["notification_id"],
            )
            extra_ids.append(retire_prepared[2].outbox_id)
    close_id = "tgcard-close-" + hashlib.sha256(f"{group_id}:{update_id}".encode()).hexdigest()
    close_prepared = prepare_outbox_record_in_transaction(
        client, transaction, close_id,
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
        schedule_at=deadline, now=now, context_id=context_id,
        notification_id=notification_id,
    )
    if initial_prepared is not None:
        write_prepared_outbox_record(transaction, initial_prepared)
        initial_task = initial_prepared[2]
    if retire_prepared is not None:
        write_prepared_outbox_record(transaction, retire_prepared)
    write_prepared_outbox_record(transaction, close_prepared)
    close_task = close_prepared[2]
    if is_new:
        context["pending_receipt_groups"] = _integer(
            context.get("pending_receipt_groups", 0), "pending_receipt_groups"
        ) + 1
    previous_group_id = (context.get("latest_receipt_notification_id") and
                         context.get("latest_receipt_group_id"))
    becomes_latest = is_new or previous_group_id in (None, group_id)
    if becomes_latest:
        context["open_text_receipt_group_id"] = group_id
        context["latest_receipt_group_id"] = group_id
        context["latest_receipt_notification_id"] = notification_id
    group["notification_id"] = notification_id
    return _ReceiptGroupPreparation(
        group_ref, group, notification_ref, notification, previous_ref, previous_data,
        initial_task, close_task, tuple(extra_ids), is_new,
    )


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
    client: Any,
    transaction: Any,
    receipt_ref: Any,
    context_ref: Any,
    context: dict[str, Any],
    *,
    update_id: int,
    context_id: str,
    chat_id: int,
    message_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> "TelegramIntakeResult":
    """Record and queue a separate refusal while a context operation holds the lock."""
    identity = hashlib.sha256(f"{context_id}:{update_id}:busy".encode()).hexdigest()
    notification_id = "tgforward-busy-" + identity
    outbox_id = "tgforward-busy-task-" + identity
    body = ("This dialog is processing another request. Please wait until it finishes "
            "before forwarding more messages.")
    notification_ref = client.collection(NOTIFICATIONS).document(notification_id)
    notification = {
        "notification_id": notification_id, "context_id": context_id,
        "kind": "error", "body": body, "chat_id": chat_id,
        "thread_id": thread_id, "message_id": None, "dialog_number": None,
        "operation_id": None, "state": "pending", "revision": 1,
        "desired_status_revision": 1, "send_attempts_started": 0,
        "edit_attempts_started": 0, "unknown_send_outcome": False,
        "in_flight_attempt_id": None, "attempt_expires_at": None,
        "last_outcome": None, "retry_not_before": None,
        "terminal_failure": None, "last_error": None, "delivered_at": None,
        "failed_at": None, "created_at": now, "updated_at": now,
    }
    prepared_outbox = prepare_outbox_record_in_transaction(
        client, transaction, outbox_id,
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
        schedule_at=now, now=now, context_id=context_id,
        notification_id=notification_id,
    )
    receipt = {"update_id": update_id, "context_id": context_id, "accepted": False,
               "outcome": "busy_forward_refused", "publish_rejection": True,
               "draft_id": None, "dialog_number": None, "message_id": message_id,
               "intake_sequence": None, "notification_id": notification_id,
               "outbox_id": outbox_id, "created_at": now}
    context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
    if topic_name is not None:
        context["topic_name"] = topic_name
    transaction.set(notification_ref, notification)
    write_prepared_outbox_record(transaction, prepared_outbox)
    transaction.set(receipt_ref, receipt)
    transaction.set(context_ref, context)
    return TelegramIntakeResult(
        False, context_id, message_id=message_id, notification_id=notification_id,
        outbox_id=outbox_id, publish_rejection=True,
    )


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
    message_ref: Any,
    delivery_chat_id: int,
    delivery_thread_id: int | None,
    message_id: int,
    intake_sequence: int,
    transcription_state: TelegramTranscriptionState,
    original_date: datetime | None,
    now: datetime,
    notification_id: str,
    outbox_id: str,
    receipt_work: _ReceiptGroupPreparation,
) -> _ForwardPreparation:
    """Read message and delivery identities before creating any transaction writes."""
    source = _build_forward_source(
        message, message_id=message_id, intake_sequence=intake_sequence,
        transcription_state=transcription_state, now=now,
        original_date=original_date,
    )
    return _ForwardPreparation(
        source, message_ref, receipt_work.notification_ref, receipt_work.notification,
        receipt_work.initial_task,
        notification_id,
        (receipt_work.initial_task.outbox_id if receipt_work.initial_task is not None
         else receipt_work.close_task.outbox_id),
        receipt_work.group_ref, receipt_work.group,
        receipt_work.previous_notification_ref, receipt_work.previous_notification,
        ((*receipt_work.additional_outbox_ids, receipt_work.close_task.outbox_id)
         if receipt_work.initial_task is not None else receipt_work.additional_outbox_ids),
        receipt_work.close_task, receipt_work.is_new,
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
    receipt_group_ref: Any,
    receipt_group: dict[str, Any],
    previous_notification_ref: Any | None,
    previous_notification: dict[str, Any] | None,
    additional_outbox_ids: tuple[str, ...],
    receipt_group_new: bool,
) -> "TelegramIntakeResult":
    """Persist an accepted forward and its receipt atomically."""
    if draft.is_new:
        transaction.set(draft.dialog_ref, draft.dialog_data)
        transaction.set(draft.ref, draft.data)
    transaction.set(notification_ref, notification)
    transaction.set(receipt_group_ref, receipt_group)
    if previous_notification_ref is not None and previous_notification is not None:
        old_revision = _integer(previous_notification.get("revision", 1), "revision", minimum=1)
        previous_notification.update({
            "receipt_control_token": None,
            "reply_markup": {"inline_keyboard": []},
            "revision": old_revision + 1,
            "desired_status_revision": old_revision + 1,
            "state": previous_notification.get("state")
                      if previous_notification.get("in_flight_attempt_id") else "pending",
            "updated_at": now,
        })
        transaction.set(previous_notification_ref, previous_notification)
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
               "outbox_id": outbox_id,
               "receipt_group_id": receipt_group.get("group_id"),
               "additional_outbox_ids": list(additional_outbox_ids),
               "created_at": now}
    transaction.set(receipt_ref, receipt)
    return TelegramIntakeResult(
        True, context_id, draft.draft_id, draft.dialog_number, message_id,
        intake_sequence, notification_id,
        outbox_id,
        additional_outbox_ids=additional_outbox_ids,
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
        if thread_id == 0:
            thread_id = None
        elif thread_id is not None:
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
        user_id: int = 0,
        thread_id: int | None = None,
        topic_name: str | None = None,
        message: Mapping[str, object],
        media_group_id: str | None = None,
        now: datetime,
    ) -> TelegramIntakeResult:
        update_id = _integer(update_id, "update_id")
        user_id = _integer(user_id, "user_id", minimum=0)
        if thread_id == 0:
            thread_id = None
        elif thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        context_id = build_context_id(chat_id, thread_id)
        now = _aware(now, "now")
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(message, Mapping) or set(message) - _SOURCE_FIELDS:
            raise ValueError("message must contain only normalized source-message fields")
        if media_group_id is not None and (not isinstance(media_group_id, str) or not media_group_id):
            raise ValueError("media_group_id must be non-empty text or None")
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
                    self.client, transaction, receipt_ref, context_ref, loaded.data,
                    update_id=update_id, context_id=context_id,
                    chat_id=chat_id, message_id=message_id, thread_id=thread_id,
                    topic_name=topic_name, now=now,
                )
            draft = _select_forward_draft(
                self.client, transaction, loaded.data, loaded,
                context_id=context_id, update_id=update_id,
                chat_id=chat_id, thread_id=thread_id,
                topic_name=topic_name, now=now,
            )
            message_ref = _document_ref(
                self.client,
                f"{SESSIONS}/{context_id}/drafts/{draft.draft_id}/messages/{message_id}",
            )
            if message_ref.get(transaction=transaction).exists:
                raise ValueError("message ID already exists in the active draft")
            receipt_work = _prepare_receipt_group(
                self.client, transaction, context_id=context_id,
                context=loaded.data, draft=draft, update_id=update_id,
                media_group_id=media_group_id, accepted=True, user_id=user_id, now=now,
            )
            notification_id = receipt_work.notification_ref.id
            outbox_id = (receipt_work.initial_task.outbox_id
                         if receipt_work.initial_task is not None
                         else receipt_work.close_task.outbox_id)
            intake_sequence = _integer(loaded.data.get("next_intake_sequence", 1), "next_intake_sequence", minimum=1)
            prepared = _prepare_forward_persistence(
                self.client, transaction, draft, message,
                context_id=context_id, message_id=message_id,
                message_ref=message_ref,
                delivery_chat_id=loaded.data["chat_id"],
                delivery_thread_id=loaded.data.get("thread_id"),
                intake_sequence=intake_sequence,
                transcription_state=transcription_state,
                original_date=original_date, now=now,
                notification_id=notification_id, outbox_id=outbox_id,
                receipt_work=receipt_work,
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
                receipt_group_ref=prepared.receipt_group_ref,
                receipt_group=prepared.receipt_group,
                previous_notification_ref=prepared.previous_notification_ref,
                previous_notification=prepared.previous_notification,
                additional_outbox_ids=prepared.additional_outbox_ids,
                receipt_group_new=prepared.receipt_group_new,
            )

        return _txn(self.client.transaction(max_attempts=5))


    def record_unsupported_forward(
        self,
        update_id: int,
        *,
        chat_id: int,
        user_id: int = 0,
        thread_id: int | None = None,
        topic_name: str | None = None,
        message_id: int,
        media_group_id: str | None = None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Durably acknowledge a contentless/unsupported forward without a draft."""
        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=0)
        message_id = _integer(message_id, "message_id", minimum=1)
        if thread_id == 0:
            thread_id = None
        elif thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        now = _aware(now, "now")
        context_id = build_context_id(chat_id, thread_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        if media_group_id is not None and (not isinstance(media_group_id, str) or not media_group_id):
            raise ValueError("media_group_id must be non-empty text or None")

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
                    self.client, transaction, receipt_ref, context_ref, loaded.data,
                    update_id=update_id, context_id=context_id,
                    chat_id=chat_id, message_id=message_id, thread_id=thread_id,
                    topic_name=topic_name, now=now,
                )
            draft = (_ForwardDraft(
                False, loaded.active_ref, loaded.active_draft_id,
                _integer(loaded.draft_data.get("dialog_number"), "dialog_number", minimum=1),
                str(loaded.draft_data.get("conversation_id", "")), loaded.draft_data,
            ) if loaded.active_ref is not None and not loaded.busy else
                _ForwardDraft(False, None, "", 0, "", {}))
            work = _prepare_receipt_group(
                self.client, transaction, context_id=context_id, context=loaded.data,
                draft=draft, update_id=update_id, media_group_id=media_group_id,
                accepted=False, user_id=user_id, now=now,
            )
            group_ref, group = work.group_ref, work.group
            notification_ref, notification = work.notification_ref, work.notification
            previous_ref, previous_data = work.previous_notification_ref, work.previous_notification
            initial_task, close_task = work.initial_task, work.close_task
            extra_ids = work.additional_outbox_ids
            transaction.set(notification_ref, notification)
            transaction.set(group_ref, group)
            if previous_ref is not None and previous_data is not None:
                old_revision = _integer(previous_data.get("revision", 1), "revision", minimum=1)
                previous_data.update({
                    "receipt_control_token": None,
                    "reply_markup": {"inline_keyboard": []},
                    "revision": old_revision + 1,
                    "desired_status_revision": old_revision + 1,
                    "state": previous_data.get("state")
                              if previous_data.get("in_flight_attempt_id") else "pending",
                    "updated_at": now,
                })
                transaction.set(previous_ref, previous_data)
            context = loaded.data
            context.update({"chat_id": chat_id, "thread_id": thread_id, "updated_at": now})
            if topic_name is not None:
                context["topic_name"] = topic_name
            transaction.set(context_ref, context)
            primary_task = initial_task or close_task
            additional = ((*extra_ids, close_task.outbox_id) if initial_task is not None else extra_ids)
            transaction.set(receipt_ref, {
                "update_id": update_id, "context_id": context_id,
                "accepted": True, "outcome": "unsupported_forward_skipped",
                "draft_id": None, "dialog_number": None, "message_id": message_id,
                "intake_sequence": None, "notification_id": notification_ref.id,
                "receipt_group_id": group.get("group_id"),
                "outbox_id": primary_task.outbox_id,
                "additional_outbox_ids": list(additional), "created_at": now,
            })
            return TelegramIntakeResult(
                True, context_id, draft.draft_id or None,
                draft.dialog_number or None, message_id=message_id,
                notification_id=notification_ref.id, outbox_id=primary_task.outbox_id,
                additional_outbox_ids=additional,
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
