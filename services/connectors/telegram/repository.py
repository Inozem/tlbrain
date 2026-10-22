"""Transactional persistence for Telegram intake and dialog records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record_in_transaction,
)
from services.connectors.telegram.models import (
    PreparedUtterance,
    TelegramContext,
    TelegramDraft,
    TelegramDraftState,
    TelegramSourceMessage,
    TelegramTranscriptionState,
    build_context_id,
)


SESSIONS = "telegram_sessions"
NOTIFICATIONS = "notification_outbox"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_SOURCE_FIELDS = frozenset({
    "message_id", "original_date", "source_chat_id", "source_message_id", "source_user_id",
    "reliable_author_key", "display_speaker", "text", "caption", "audio_file_id",
    "audio_file_unique_id", "audio_mime_type", "audio_file_size", "transcription_state",
    "transcription_text", "transcription_ref", "transcription_error",
})


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a valid Firestore-safe identifier")
    return value


def _document_ref(client: Any, path: str) -> Any:
    parts = path.split("/")
    ref = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        ref = ref.collection(parts[index]).document(parts[index + 1])
    return ref


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _draft_data(draft: TelegramDraft) -> dict[str, Any]:
    data = asdict(draft)
    data["state"] = draft.state.value
    data["ordered_message_ids"] = list(draft.ordered_message_ids)
    data["completed_audio_ids"] = list(draft.completed_audio_ids)
    return data


def _draft_from(data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    result.setdefault("state", TelegramDraftState.COLLECTING.value)
    result.setdefault("ordered_message_ids", [])
    result.setdefault("completed_audio_ids", [])
    result.setdefault("required_audio_count", 0)
    result.setdefault("prepared_count", 0)
    result.setdefault("context_capacity", None)
    result.setdefault("preparation_digest", None)
    return result


def _source_data(message: TelegramSourceMessage) -> dict[str, Any]:
    data = asdict(message)
    data["transcription_state"] = message.transcription_state.value
    return data


def _context_defaults(
    chat_id: int,
    thread_id: int | None,
    topic_name: str | None,
    now: datetime,
) -> dict[str, Any]:
    """Return persisted defaults for a newly-created Telegram context."""
    data = asdict(TelegramContext(
        chat_id=chat_id,
        thread_id=thread_id,
        topic_name=topic_name,
        created_at=now,
        updated_at=now,
    ))
    data["next_intake_sequence"] = 1
    return data


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


def _receipt_result(saved: Mapping[str, Any], context_id: str) -> "TelegramIntakeResult":
    """Restore the result recorded for a previously accepted update."""
    return TelegramIntakeResult(
        bool(saved["accepted"]), context_id, saved.get("draft_id"),
        saved.get("dialog_number"), saved.get("message_id"),
        saved.get("intake_sequence"), saved.get("notification_id"), saved.get("outbox_id"),
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
        "body": f"\u0414\u0438\u0430\u043b\u043e\u0433 \u2116{draft.dialog_number}: \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043f\u0440\u0438\u043d\u044f\u0442\u043e",
        "created_at": now,
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


@dataclass(frozen=True, slots=True)
class TelegramIntakeResult:
    accepted: bool
    context_id: str
    draft_id: str | None = None
    dialog_number: int | None = None
    message_id: int | None = None
    intake_sequence: int | None = None
    notification_id: str | None = None
    outbox_id: str | None = None


class TelegramRepository:
    """Own Firestore transactions for durable Telegram intake state."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client if client is not None else get_client()

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
        for field_name in ("source_chat_id", "source_message_id", "source_user_id", "audio_file_size"):
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

    def get_update_receipt(self, context_id: str, update_id: int) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        update_id = _integer(update_id, "update_id")
        snapshot = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}").get()
        return snapshot.to_dict() if snapshot.exists else None

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        context_id = _identifier(context_id, "context_id")
        snapshot = self.client.collection(SESSIONS).document(context_id).get()
        return snapshot.to_dict() if snapshot.exists else None

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


__all__ = ["TelegramIntakeResult", "TelegramRepository"]
