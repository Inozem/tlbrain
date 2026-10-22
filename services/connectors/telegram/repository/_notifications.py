"""Durable Telegram notification ledger and acknowledgement failure state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from google.cloud import firestore

from services.connectors.telegram.models import TelegramDraftState
from services.connectors.telegram.repository._common import (
    NOTIFICATIONS,
    NOTIFICATION_ATTEMPT_LEASE,
    NOTIFICATION_KINDS,
    NOTIFICATION_MAX_SENDS,
    SESSIONS,
    _document_ref,
    _identifier,
    _integer,
    _aware,
)

@dataclass(slots=True)
class _AckFailureState:
    context_id: str
    draft_id: str
    context_ref: Any
    draft_ref: Any
    context: dict[str, Any]
    draft: dict[str, Any]


@dataclass(slots=True)
class _AckFailureLinkage:
    parent_id: Any
    save_operation_id: Any
    parent_ref: Any | None = None
    parent: dict[str, Any] | None = None
    next_data: dict[str, Any] | None = None
    save_ref: Any | None = None
    save_operation: dict[str, Any] | None = None
    verified: bool = False


class _TelegramNotificationMixin:
    """Store and settle bounded notification delivery attempts."""

    def get_notification(self, notification_id: str) -> dict[str, Any] | None:
        """Load a server-authored notification ledger row by stable ID."""
        notification_id = _identifier(notification_id, "notification_id")
        snapshot = self.client.collection(NOTIFICATIONS).document(notification_id).get()
        return snapshot.to_dict() if snapshot.exists else None


    def revise_notification(
        self, notification_id: str, *, body: str, now: datetime,
    ) -> dict[str, Any]:
        """Persist a new server-authored body revision and fence older attempts."""
        notification_id = _identifier(notification_id, "notification_id")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                raise ValueError("notification does not exist")
            data = snap.to_dict() or {}
            if data.get("state") in {"failed", "delivery_unknown"}:
                raise ValueError("terminal notification cannot be revised")
            if data.get("kind") == "draft_ack" and data.get("message_id") is not None:
                raise ValueError("delivered draft acknowledgement cannot be revised")
            new_revision = _integer(data.get("revision", 1), "revision", minimum=1) + 1
            updates = {"body": body, "revision": new_revision,
                       "desired_status_revision": new_revision,
                "state": "pending", "in_flight_attempt_id": None,
                       "attempt_expires_at": None, "retry_not_before": None,
                       "delivered_at": None, "last_error": None, "updated_at": now}
            transaction.update(ref, updates)
            return {**data, **updates}

        return _txn(self.client.transaction(max_attempts=5))


    def create_notification(
        self, notification_id: str, *, context_id: str, kind: str, body: str, now: datetime,
        dialog_number: int | None = None, operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an explicit notification using only the persisted delivery route."""
        notification_id, context_id = _identifier(notification_id, "notification_id"), _identifier(context_id, "context_id")
        if kind not in NOTIFICATION_KINDS:
            raise ValueError("kind is not an allowed notification kind")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        if dialog_number is not None:
            dialog_number = _integer(dialog_number, "dialog_number", minimum=1)
        if operation_id is not None:
            operation_id = _identifier(operation_id, "operation_id")
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        notification_ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            context_snap = context_ref.get(transaction=transaction)
            if not context_snap.exists:
                raise ValueError("notification context does not exist")
            context = context_snap.to_dict() or {}
            chat_id = context.get("chat_id")
            if type(chat_id) is not int:
                raise ValueError("stored chat_id is invalid")
            thread_id = context.get("thread_id")
            if thread_id is not None:
                thread_id = None if thread_id == 0 else _integer(thread_id, "thread_id", minimum=1)
            snap = notification_ref.get(transaction=transaction)
            if snap.exists:
                stored = snap.to_dict() or {}
                identity = (stored.get("context_id"), stored.get("kind"), stored.get("body"), stored.get("chat_id"), stored.get("thread_id"), stored.get("dialog_number"), stored.get("operation_id"))
                if identity != (context_id, kind, body, chat_id, thread_id, dialog_number, operation_id):
                    raise ValueError("notification ID already belongs to a different identity")
                return stored
            row = {"notification_id": notification_id, "context_id": context_id, "kind": kind,
                   "body": body, "chat_id": chat_id, "thread_id": thread_id, "message_id": None,
                   "dialog_number": dialog_number, "operation_id": operation_id,
                   "state": "pending", "revision": 1, "desired_status_revision": 1, "send_attempts_started": 0,
                   "edit_attempts_started": 0, "unknown_send_outcome": False,
                   "in_flight_attempt_id": None, "attempt_expires_at": None, "last_outcome": None,
                   "retry_not_before": None, "terminal_failure": None, "last_error": None,
                   "delivered_at": None, "failed_at": None, "created_at": now, "updated_at": now}
            transaction.set(notification_ref, row)
            return row

        return _txn(self.client.transaction(max_attempts=5))


    def reserve_notification_attempt(
        self, notification_id: str, *, attempt_id: str, now: datetime,
    ) -> dict[str, Any] | None:
        """Reserve one bounded send or edit attempt before contacting Telegram."""
        notification_id = _identifier(notification_id, "notification_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if data.get("state") in {"delivered", "failed", "delivery_unknown"}:
                return {**data, "reserved": False}
            if data.get("retry_not_before") and data["retry_not_before"] > now:
                return {**data, "reserved": False}
            inflight = data.get("in_flight_attempt_id")
            expiry = data.get("attempt_expires_at")
            if inflight and isinstance(expiry, datetime) and expiry > now:
                return {**data, "reserved": False}
            abandoned_send = bool(inflight and data.get("message_id") is None)
            if abandoned_send:
                data["unknown_send_outcome"] = True
                data["last_outcome"] = "unknown"
            is_send = data.get("message_id") is None
            attempts = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
            if is_send and attempts >= NOTIFICATION_MAX_SENDS:
                terminal = "delivery_unknown" if data.get("unknown_send_outcome") else "failed"
                updates = {"state": terminal, "terminal_failure": terminal, "updated_at": now,
                           "last_error": "attempts_exhausted", "failed_at": now,
                           "in_flight_attempt_id": None, "attempt_expires_at": None}
                transaction.update(ref, updates)
                return {**data, **updates, "reserved": False}
            revision = _integer(data.get("revision", 1), "revision", minimum=1)
            updates = {"state": "sending" if is_send else "editing", "in_flight_attempt_id": attempt_id,
                       "attempt_revision": revision, "attempt_expires_at": now + NOTIFICATION_ATTEMPT_LEASE,
                       "updated_at": now}
            if abandoned_send:
                updates["last_outcome"] = "unknown"
                updates["unknown_send_outcome"] = True
            if is_send:
                updates["send_attempts_started"] = attempts + 1
            else:
                edit_attempts = _integer(data.get("edit_attempts_started", 0), "edit_attempts_started")
                if edit_attempts >= NOTIFICATION_MAX_SENDS:
                    updates.update({"state": "failed", "terminal_failure": "failed",
                                    "last_error": "attempts_exhausted", "failed_at": now,
                                    "in_flight_attempt_id": None, "attempt_expires_at": None})
                    transaction.update(ref, updates)
                    return {**data, **updates, "reserved": False}
                updates["edit_attempts_started"] = edit_attempts + 1
            transaction.update(ref, updates)
            return {**data, **updates, "reserved": True, "attempt_id": attempt_id,
                    "attempt_revision": revision, "is_send": is_send}

        return _txn(self.client.transaction(max_attempts=5))


    def settle_notification_attempt(
        self, notification_id: str, *, attempt_id: str, revision: int, outcome: str,
        now: datetime, message_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Settle only the current attempt and current notification revision."""
        notification_id = _identifier(notification_id, "notification_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        revision = _integer(revision, "revision", minimum=1)
        if outcome not in {"sent", "edited", "unknown", "rejected"}:
            raise ValueError("outcome is invalid")
        if outcome == "sent":
            message_id = _integer(message_id, "message_id", minimum=1)
        elif message_id is not None:
            raise ValueError("message_id is only valid for a sent outcome")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any] | None:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if (data.get("in_flight_attempt_id") != attempt_id
                    or data.get("attempt_revision") != revision
                    or data.get("revision") != revision
                    or data.get("state") in {"delivered", "failed", "delivery_unknown"}):
                return {**data, "settled": False}
            updates: dict[str, Any] = {"in_flight_attempt_id": None, "attempt_expires_at": None,
                                       "last_outcome": outcome, "updated_at": now}
            if outcome in {"sent", "edited"}:
                if outcome == "sent":
                    updates["message_id"] = message_id
                updates.update({"state": "delivered", "terminal_failure": None, "retry_not_before": None,
                                "last_error": None, "delivered_at": now, "failed_at": None})
            else:
                sends = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
                exhausted = (data.get("message_id") is None and sends >= NOTIFICATION_MAX_SENDS) or (
                    data.get("message_id") is not None
                    and _integer(data.get("edit_attempts_started", 0), "edit_attempts_started") >= NOTIFICATION_MAX_SENDS
                )
                if exhausted:
                    terminal = ("delivery_unknown" if data.get("unknown_send_outcome") or outcome == "unknown"
                                else "failed") if data.get("message_id") is None else "failed"
                    updates.update({"state": terminal, "terminal_failure": terminal, "retry_not_before": None,
                                    "last_error": outcome, "failed_at": now})
                else:
                    delay = min(2 ** max(0, sends - 1), 60)
                    updates.update({"state": "retry", "retry_not_before": now + timedelta(seconds=delay),
                                    "last_error": outcome})
                    if outcome == "unknown" and data.get("message_id") is None:
                        updates["unknown_send_outcome"] = True
            result = {**data, **updates, "settled": True}
            if (result.get("state") in {"failed", "delivery_unknown"}
                    and data.get("kind") == "draft_ack" and data.get("message_id") is None):
                action = self._settle_terminal_ack_failure(transaction, result, now)
                if action is not None:
                    result["ack_failure_action"] = action
                    updates["ack_failure_action"] = action
            transaction.update(ref, updates)
            return result

        return _txn(self.client.transaction(max_attempts=5))


    def _settle_terminal_ack_failure(
        self, transaction: Any, notification: dict[str, Any], now: datetime,
    ) -> dict[str, Any] | None:
        """Fail only a sealed draft waiting on its first acknowledgement barrier."""
        state = self._load_ack_failure_state(transaction, notification)
        if state is None:
            return None
        linkage = self._read_ack_failure_linkage(transaction, state)
        return self._write_ack_failure(transaction, state, linkage, now)


    def _load_ack_failure_state(
        self, transaction: Any, notification: dict[str, Any],
    ) -> _AckFailureState | None:
        """Load an eligible sealed draft and its context before any writes."""
        context_id, draft_id = notification.get("context_id"), notification.get("draft_id")
        if not isinstance(context_id, str) or not isinstance(draft_id, str):
            return None
        context_ref = self.client.collection(SESSIONS).document(context_id)
        draft_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
        context_snap, draft_snap = context_ref.get(transaction=transaction), draft_ref.get(transaction=transaction)
        if not draft_snap.exists:
            return None
        draft = draft_snap.to_dict() or {}
        if not draft.get("sealed_at") or not draft.get("manifest_ref") or not draft.get("manifest_digest"):
            return None
        if draft.get("state") in {TelegramDraftState.COMPLETED.value, TelegramDraftState.CLEANED.value}:
            return None
        context = context_snap.to_dict() or {}
        return _AckFailureState(context_id, draft_id, context_ref, draft_ref, context, draft)


    def _read_ack_failure_linkage(
        self, transaction: Any, state: _AckFailureState,
    ) -> _AckFailureLinkage:
        """Verify queue and save-operation links before the transaction writes."""
        context_id, draft_id = state.context_id, state.draft_id
        context, draft = state.context, state.draft
        save_operation_id = draft.get("active_save_operation_id")
        parent_id = context.get("reserved_operation_id")
        linkage = _AckFailureLinkage(parent_id, save_operation_id)
        if isinstance(parent_id, str) and parent_id:
            linkage.parent_ref = _document_ref(
                self.client, f"{SESSIONS}/{context_id}/operations/{parent_id}",
            )
            parent_snap = linkage.parent_ref.get(transaction=transaction)
            if parent_snap.exists:
                linkage.parent = parent_snap.to_dict() or {}
                linkage.verified = (
                    linkage.parent.get("operation_id") == parent_id
                    and linkage.parent.get("target_draft_id") == draft_id
                    and linkage.parent.get("save_operation_id") == save_operation_id
                    and bool(save_operation_id)
                    and context.get("head_sequence") == linkage.parent.get("sequence")
                    and linkage.parent.get("state") not in {"completed", "failed"}
                )
                next_id = linkage.parent.get("next_operation_id")
                if linkage.verified and next_id:
                    next_ref = _document_ref(
                        self.client, f"{SESSIONS}/{context_id}/operations/{next_id}",
                    )
                    next_snap = next_ref.get(transaction=transaction)
                    if next_snap.exists:
                        linkage.next_data = next_snap.to_dict() or {}
                        if (linkage.next_data.get("operation_id") != next_id
                                or linkage.next_data.get("state") in {"completed", "failed"}):
                            linkage.verified = False
                    else:
                        linkage.verified = False
                if linkage.verified:
                    linkage.save_ref = self.client.collection("dialog_operations").document(save_operation_id)
                    save_snap = linkage.save_ref.get(transaction=transaction)
                    if save_snap.exists:
                        linkage.save_operation = save_snap.to_dict() or {}
                        if linkage.save_operation.get("operation_id") != save_operation_id:
                            linkage.verified = False
        return linkage


    def _write_ack_failure(
        self,
        transaction: Any,
        state: _AckFailureState,
        linkage: _AckFailureLinkage,
        now: datetime,
    ) -> dict[str, Any]:
        """Fail the draft and release its queue only when every link verified."""
        context, draft = state.context, state.draft
        draft.update({"state": TelegramDraftState.FAILED.value, "ack_failure": "notification_delivery_failed",
                      "updated_at": now})
        if context.get("active_draft_id") == state.draft_id:
            context.update({"active_draft_id": None, "draft_status": None})
        transaction.set(state.draft_ref, draft)
        action = {"context_id": state.context_id, "draft_id": state.draft_id,
                  "dialog_number": draft.get("dialog_number"),
                  "operation_id": linkage.save_operation_id, "parent_operation_id": linkage.parent_id,
                  "queue_release_required": not linkage.verified}
        if linkage.verified and linkage.parent_ref is not None and linkage.parent is not None:
            linkage.parent.update({"state": "failed", "failure_code": "ack_delivery_failed", "finished_at": now})
            context.update({"active_operation_id": None, "lock_owner": None, "lock_expires_at": None})
            if linkage.next_data is not None:
                context.update({"reserved_operation_id": linkage.next_data.get("operation_id"),
                                "head_sequence": linkage.next_data.get("sequence")})
            else:
                context.update({"reserved_operation_id": None, "head_sequence": None,
                                "queue_tail_operation_id": None})
            transaction.set(linkage.parent_ref, linkage.parent)
            if (linkage.save_ref is not None and linkage.save_operation is not None
                    and linkage.save_operation.get("state") not in {"completed", "failed"}):
                linkage.save_operation.update({
                    "state": "failed", "stage": "notification", "finished_at": now,
                    "updated_at": now, "status_updated_at": now,
                    "lease_owner": None, "lease_expires_at": None,
                    "error": {"code": "ack_delivery_failed", "category": "notification",
                              "sanitized_message": "The initial acknowledgement could not be delivered.",
                              "retryable": False, "service": "telegram", "stage": "notification",
                              "occurred_at": now, "failures": 1,
                              "operation_id": linkage.save_operation_id},
                })
                transaction.set(linkage.save_ref, linkage.save_operation)
        transaction.set(state.context_ref, context)
        return action

