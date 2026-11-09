"""Durable Telegram notification ledger and acknowledgement failure state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from typing import Any, Mapping

from google.cloud import firestore

from services.connectors.telegram.formatting import delivery_presentation
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


def _retry_deadline(now: datetime, delay_seconds: int) -> datetime:
    """Return a retry deadline without overflowing Python's datetime range."""
    try:
        return now + timedelta(seconds=delay_seconds)
    except OverflowError:
        return datetime.max.replace(tzinfo=timezone.utc)


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


def _stage_receipt_card_revision(
    transaction: Any, *, ref: Any, row: dict[str, Any], body: str,
    markup: dict[str, Any], token: str | None, owner_user_id: Any,
    total_count: int, latest_id: str, control_ref: Any, context_id: str,
    now: datetime,
) -> bool:
    """Stage one bounded card revision and its server-side Save control binding."""
    changed = (row.get("body") != body or row.get("reply_markup") != markup
               or row.get("receipt_control_token") != token)
    if not changed or row.get("state") in {"failed", "delivery_unknown"}:
        return False
    revision = _integer(row.get("revision", 1), "revision", minimum=1) + 1
    row.update({
        "body": body, "reply_markup": markup,
        "receipt_control_token": token,
        "receipt_control_owner_user_id": owner_user_id if token else None,
        "receipt_accepted_count": total_count,
        "revision": revision, "desired_status_revision": revision,
        "state": row.get("state") if row.get("in_flight_attempt_id") else "pending",
        "delivered_at": None, "last_error": None, "updated_at": now,
    })
    if body.startswith("Dialog #"):
        row["receipt_numbered_revision"] = revision
    transaction.set(ref, row)
    if token and ref.id == latest_id and row.get("draft_id") and owner_user_id:
        transaction.set(control_ref, {
            "token": token, "context_id": context_id,
            "notification_id": ref.id, "draft_id": row["draft_id"],
            "dialog_number": row.get("dialog_number"),
            "owner_user_id": owner_user_id,
            "issued_revision": revision, "issued_at": now,
            "state": "active",
        })
    return True


def _receipt_status_body(
    dialog_number: Any, accepted_count: int, *, ready: bool, skipped_count: int,
) -> str:
    """Format a fixed-English receipt status for a collecting draft."""
    prefix = f"Dialog #{dialog_number}: "
    if ready:
        body = (f"{prefix}Ready - {accepted_count} messages received.\n\n"
                "Forward the remaining messages from this conversation, then press Save.")
    else:
        body = f"{prefix}Thinking... {accepted_count} messages received."
    if skipped_count:
        body += f"\n\nSkipped {skipped_count} unsupported item(s)."
    return body


class _TelegramNotificationMixin:
    """Store and settle bounded notification delivery attempts."""

    def get_notification(self, notification_id: str) -> dict[str, Any] | None:
        """Load a server-authored notification ledger row by stable ID."""
        notification_id = _identifier(notification_id, "notification_id")
        snapshot = self.client.collection(NOTIFICATIONS).document(notification_id).get()
        return snapshot.to_dict() if snapshot.exists else None


    def refresh_receipt_card(self, notification_id: str, *, now: datetime) -> tuple[str, ...] | None:
        """Close an expired group and refresh its result plus the latest eligible card."""
        notification_id = _identifier(notification_id, "notification_id")
        now = _aware(now, "now")
        callback_token = secrets.token_urlsafe(18)
        initial_ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> tuple[str, ...] | None:
            initial_snapshot = initial_ref.get(transaction=transaction)
            if not initial_snapshot.exists:
                return None
            initial = initial_snapshot.to_dict() or {}
            group_id = initial.get("receipt_group_id")
            context_id = initial.get("context_id")
            if (not initial.get("receipt_card") or not isinstance(group_id, str)
                    or not isinstance(context_id, str)):
                return None
            group_ref = _document_ref(
                self.client, f"{SESSIONS}/{context_id}/receipt_groups/{group_id}",
            )
            context_ref = self.client.collection(SESSIONS).document(context_id)
            group_snapshot = group_ref.get(transaction=transaction)
            context_snapshot = context_ref.get(transaction=transaction)
            if not group_snapshot.exists or not context_snapshot.exists:
                return None
            group = group_snapshot.to_dict() or {}
            context = context_snapshot.to_dict() or {}
            draft_id = group.get("draft_id")
            draft_ref = (_document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
                         if isinstance(draft_id, str) and draft_id else None)
            draft_snapshot = draft_ref.get(transaction=transaction) if draft_ref is not None else None
            draft = (draft_snapshot.to_dict() or {}
                     if draft_snapshot is not None and draft_snapshot.exists else {})
            latest_id = context.get("latest_receipt_notification_id")
            latest_ref = (self.client.collection(NOTIFICATIONS).document(latest_id)
                          if isinstance(latest_id, str) else initial_ref)
            latest_snapshot = (latest_ref.get(transaction=transaction)
                               if latest_ref.id != initial_ref.id else initial_snapshot)
            latest = latest_snapshot.to_dict() or {} if latest_snapshot.exists else {}
            latest_draft_id = latest.get("draft_id")
            if latest_draft_id == draft_id:
                latest_draft = draft
            else:
                latest_draft_ref = (
                    _document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{latest_draft_id}")
                    if isinstance(latest_draft_id, str) and latest_draft_id else None
                )
                latest_draft_snapshot = (
                    latest_draft_ref.get(transaction=transaction)
                    if latest_draft_ref is not None else None
                )
                latest_draft = (
                    latest_draft_snapshot.to_dict() or {}
                    if latest_draft_snapshot is not None and latest_draft_snapshot.exists else {}
                )
            control_token = latest.get("receipt_control_token")
            if not isinstance(control_token, str):
                control_token = callback_token
            control_ref = _document_ref(
                self.client, f"{SESSIONS}/{context_id}/receipt_controls/{control_token}",
            )
            control_ref.get(transaction=transaction)

            deadline = group.get("deadline_at")
            if group.get("state") == "open":
                if not isinstance(deadline, datetime) or deadline > now:
                    return (initial_ref.id,)
                group.update({"state": "closed", "closed_at": now, "updated_at": now})
                context["pending_receipt_groups"] = max(
                    0, _integer(context.get("pending_receipt_groups", 0),
                                "pending_receipt_groups") - 1,
                )
                if context.get("open_text_receipt_group_id") == group_id:
                    context["open_text_receipt_group_id"] = None

            save_owned = (isinstance(initial.get("save_operation_id"), str)
                          or initial.get("receipt_terminal") is True)
            if save_owned:
                targets = [initial_ref.id]
                if (group.get("state") == "closed" and latest_ref.id != initial_ref.id
                        and latest.get("receipt_card") is True
                        and latest.get("save_operation_id") is None
                        and latest.get("receipt_terminal") is not True
                        and context.get("latest_receipt_notification_id") == latest_ref.id
                        and context.get("active_draft_id") == latest_draft_id
                        and latest_draft.get("state") == TelegramDraftState.COLLECTING.value):
                    accepted_latest_count = len(latest_draft.get("ordered_message_ids", []))
                    if accepted_latest_count:
                        ready = context.get("pending_receipt_groups", 0) == 0
                        latest_skipped = _integer(latest.get("receipt_skipped_count", 0),
                                                  "receipt_skipped_count")
                        body = _receipt_status_body(
                            latest_draft.get("dialog_number"), accepted_latest_count,
                            ready=ready, skipped_count=latest_skipped,
                        )
                        token = ((latest.get("receipt_control_token") or callback_token)
                                 if ready else None)
                        markup = ({"inline_keyboard": [[{"text": "Save",
                                                          "callback_data": f"s:{token}"}]]}
                                  if ready else {"inline_keyboard": []})
                        _stage_receipt_card_revision(
                            transaction, ref=latest_ref, row=latest, body=body,
                            markup=markup, token=token if isinstance(token, str) else None,
                            owner_user_id=latest.get("receipt_owner_user_id"),
                            total_count=accepted_latest_count, latest_id=latest_ref.id,
                            control_ref=control_ref, context_id=context_id, now=now,
                        )
                        targets.append(latest_ref.id)
                transaction.set(group_ref, group)
                transaction.set(context_ref, context)
                return tuple(targets)

            is_latest = (
                latest.get("receipt_card") is True
                and latest_ref.id == initial_ref.id
                and latest_ref.id == context.get("latest_receipt_notification_id")
            )
            targets: list[str] = []

            accepted_count = len(draft.get("ordered_message_ids", []))
            if group.get("state") == "closed" and initial_ref.id != latest_ref.id:
                group_count = len(group.get("accepted_message_ids", []))
                skipped_count = _integer(initial.get("receipt_skipped_count", 0),
                                         "receipt_skipped_count")
                if initial.get("draft_id") and accepted_count:
                    body = (f"Dialog #{initial.get('dialog_number')}: Received "
                            f"{group_count} forwarded message(s). {accepted_count} messages "
                            "are accepted in this draft. Forward the remaining messages "
                            "from this conversation, then press Save on the latest card.")
                    if skipped_count:
                        body += f"\n\nSkipped {skipped_count} unsupported item(s)."
                else:
                    body = ("This forwarded group has no supported messages. "
                            f"Skipped {skipped_count} unsupported item(s).")
                _stage_receipt_card_revision(
                    transaction, ref=initial_ref, row=initial, body=body,
                    markup={"inline_keyboard": []}, token=None, owner_user_id=None,
                    total_count=accepted_count, latest_id=latest_ref.id,
                    control_ref=control_ref, context_id=context_id, now=now,
                )
                targets.append(initial_ref.id)

            if not is_latest:
                if group.get("state") == "closed":
                    transaction.set(group_ref, group)
                    transaction.set(context_ref, context)
                latest_count = len(latest_draft.get("ordered_message_ids", []))
                if (targets and latest
                        and latest.get("save_operation_id") is None
                        and latest.get("receipt_terminal") is not True
                        and context.get("active_draft_id") == latest.get("draft_id")
                        and latest_draft.get("state") == TelegramDraftState.COLLECTING.value
                        and latest_count > 0):
                    ready = context.get("pending_receipt_groups", 0) == 0
                    skipped = _integer(latest.get("receipt_skipped_count", 0),
                                       "receipt_skipped_count")
                    body = _receipt_status_body(
                        latest_draft.get("dialog_number"), latest_count,
                        ready=ready, skipped_count=skipped,
                    )
                    token = ((latest.get("receipt_control_token") or callback_token)
                             if ready else None)
                    markup = ({"inline_keyboard": [[{"text": "Save",
                                                       "callback_data": f"s:{token}"}]]}
                              if ready else {"inline_keyboard": []})
                    _stage_receipt_card_revision(
                        transaction, ref=latest_ref, row=latest, body=body,
                        markup=markup, token=token if isinstance(token, str) else None,
                        owner_user_id=latest.get("receipt_owner_user_id"),
                        total_count=latest_count, latest_id=latest_ref.id,
                        control_ref=control_ref, context_id=context_id, now=now,
                    )
                    targets.append(latest_ref.id)
                return tuple(targets) or (initial_ref.id,)

            accepted_count = len(draft.get("ordered_message_ids", []))
            skipped = _integer(latest.get("receipt_skipped_count", 0), "receipt_skipped_count")
            current_draft = (
                context.get("active_draft_id") == latest.get("draft_id")
                and draft.get("state") == TelegramDraftState.COLLECTING.value
                and latest.get("save_operation_id") is None
                and latest.get("receipt_terminal") is not True
                and accepted_count > 0
            )
            ready = current_draft and context.get("pending_receipt_groups", 0) == 0
            if current_draft:
                body = _receipt_status_body(
                    draft.get("dialog_number"), accepted_count,
                    ready=ready, skipped_count=skipped,
                )
                token = (latest.get("receipt_control_token") or callback_token) if ready else None
                markup = ({"inline_keyboard": [[{"text": "Save",
                                                   "callback_data": f"s:{token}"}]]}
                          if ready else {"inline_keyboard": []})
                owner_user_id = latest.get("receipt_owner_user_id")
            else:
                body = latest.get("body", "This forwarded item was skipped.")
                if skipped:
                    body = ("This forwarded group has no supported messages. "
                            f"Skipped {skipped} unsupported item(s).")
                markup, token, owner_user_id = {"inline_keyboard": []}, None, None

            _stage_receipt_card_revision(
                transaction, ref=latest_ref, row=latest, body=body,
                markup=markup, token=token, owner_user_id=owner_user_id,
                total_count=accepted_count, latest_id=latest_ref.id,
                control_ref=control_ref, context_id=context_id, now=now,
            )
            transaction.set(group_ref, group)
            transaction.set(context_ref, context)
            return (latest_ref.id,)

        return _txn(self.client.transaction(max_attempts=5))


    def revise_notification(
        self, notification_id: str, *, body: str, now: datetime,
        reply_markup: Mapping[str, Any] | None = None,
        presentation: str | None = None,
    ) -> dict[str, Any]:
        """Persist a new body revision without releasing an active delivery lease."""
        notification_id = _identifier(notification_id, "notification_id")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        if presentation not in (None, "rich_markdown"):
            raise ValueError("presentation is invalid")
        now = _aware(now, "now")
        ref = self.client.collection(NOTIFICATIONS).document(notification_id)

        @firestore.transactional
        def _txn(transaction: Any) -> dict[str, Any]:
            snap = ref.get(transaction=transaction)
            if not snap.exists:
                raise ValueError("notification does not exist")
            data = snap.to_dict() or {}
            if data.get("receipt_terminal") is True:
                raise ValueError("terminal receipt card cannot be revised")
            if data.get("state") in {"failed", "delivery_unknown"}:
                raise ValueError("terminal notification cannot be revised")
            if data.get("kind") in {"final", "error"}:
                raise ValueError("terminal notification cannot be revised")
            if (data.get("kind") == "draft_ack" and not data.get("receipt_card")
                    and data.get("message_id") is not None):
                raise ValueError("delivered draft acknowledgement cannot be revised")
            if reply_markup is not None and not isinstance(reply_markup, Mapping):
                raise ValueError("reply_markup must be a mapping")
            new_revision = _integer(data.get("revision", 1), "revision", minimum=1) + 1
            retry_not_before = (
                data.get("retry_not_before")
                if data.get("last_outcome") == "rate_limited"
                and isinstance(data.get("retry_not_before"), datetime)
                else None
            )
            updates = {"body": body, "presentation": presentation,
                       "revision": new_revision,
                       "desired_status_revision": new_revision,
                       "state": (data.get("state") if data.get("in_flight_attempt_id")
                                 else "pending"),
                       "retry_not_before": retry_not_before, "delivered_at": None,
                       "last_error": None, "updated_at": now}
            if reply_markup is not None:
                updates["reply_markup"] = dict(reply_markup)
            transaction.update(ref, updates)
            return {**data, **updates}

        return _txn(self.client.transaction(max_attempts=5))


    def create_notification(
        self, notification_id: str, *, context_id: str, kind: str, body: str, now: datetime,
        dialog_number: int | None = None, operation_id: str | None = None,
        presentation: str | None = None,
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
            snap = notification_ref.get(transaction=transaction)
            row = self.prepare_notification_in_transaction(
                notification_id, context_id=context_id, kind=kind, body=body, now=now,
                context_data=context_snap.to_dict() or {}, existing_data=(
                    snap.to_dict() or {} if snap.exists else None
                ), dialog_number=dialog_number, operation_id=operation_id,
                presentation=presentation,
            )
            if not snap.exists:
                transaction.set(notification_ref, row)
            return row
        return _txn(self.client.transaction(max_attempts=5))


    def prepare_notification_in_transaction(
        self, notification_id: str, *, context_id: str, kind: str, body: str, now: datetime,
        context_data: Mapping[str, Any], existing_data: Mapping[str, Any] | None = None,
        dialog_number: int | None = None, operation_id: str | None = None,
        presentation: str | None = None,
    ) -> dict[str, Any]:
        """Validate/build a notification row after its context and ledger reads."""
        notification_id = _identifier(notification_id, "notification_id")
        context_id = _identifier(context_id, "context_id")
        if kind not in NOTIFICATION_KINDS:
            raise ValueError("kind is not an allowed notification kind")
        if not isinstance(body, str) or not body.strip() or len(body) > 4096:
            raise ValueError("body must be non-empty text of at most 4096 characters")
        if dialog_number is not None:
            dialog_number = _integer(dialog_number, "dialog_number", minimum=1)
        if operation_id is not None:
            operation_id = _identifier(operation_id, "operation_id")
        if presentation not in (None, "rich_markdown"):
            raise ValueError("presentation is invalid")
        now = _aware(now, "now")
        chat_id = context_data.get("chat_id")
        if type(chat_id) is not int:
            raise ValueError("stored chat_id is invalid")
        thread_id = context_data.get("thread_id")
        if thread_id is not None:
            thread_id = None if thread_id == 0 else _integer(thread_id, "thread_id", minimum=1)
        if existing_data is not None:
            identity = (
                existing_data.get("context_id"), existing_data.get("kind"),
                existing_data.get("body"), existing_data.get("chat_id"),
                existing_data.get("thread_id"), existing_data.get("dialog_number"),
                existing_data.get("operation_id"),
            )
            expected = (context_id, kind, body, chat_id, thread_id, dialog_number, operation_id)
            if identity != expected:
                raise ValueError("notification ID already belongs to a different identity")
            return dict(existing_data)
        row = {
            "notification_id": notification_id, "context_id": context_id, "kind": kind,
            "body": body, "chat_id": chat_id, "thread_id": thread_id, "message_id": None,
            "dialog_number": dialog_number, "operation_id": operation_id,
            "state": "pending", "revision": 1, "desired_status_revision": 1,
            "send_attempts_started": 0, "edit_attempts_started": 0,
            "unknown_send_outcome": False, "in_flight_attempt_id": None,
            "attempt_expires_at": None, "last_outcome": None, "retry_not_before": None,
            "terminal_failure": None, "last_error": None, "delivered_at": None,
            "failed_at": None, "created_at": now, "updated_at": now,
        }
        if presentation is not None:
            row["presentation"] = presentation
        return row


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
            if data.get("kind") == "progress" and data.get("operation_id"):
                operation_ref = self.client.collection("dialog_operations").document(data["operation_id"])
                operation_snap = operation_ref.get(transaction=transaction)
                operation = operation_snap.to_dict() or {} if operation_snap.exists else {}
                if operation.get("final_delivery_state") == "delivered":
                    return {**data, "reserved": False, "state": "superseded"}
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
            original_presentation = data.get("presentation")
            resolved_presentation = delivery_presentation(
                original_presentation, is_send=is_send,
                reply_markup=data.get("reply_markup"),
            )
            rich_markup_fallback = resolved_presentation != original_presentation
            if rich_markup_fallback:
                # Keep the row's revision metadata aligned with the supported
                # one-request plain send that must retain this keyboard.
                data["presentation"] = resolved_presentation
            attempts = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
            if is_send and attempts >= NOTIFICATION_MAX_SENDS:
                terminal = "delivery_unknown" if data.get("unknown_send_outcome") else "failed"
                updates = {"state": terminal, "terminal_failure": terminal, "updated_at": now,
                           "last_error": "attempts_exhausted", "failed_at": now,
                           "in_flight_attempt_id": None, "attempt_expires_at": None}
                self._persist_final_delivery_state(transaction, data, terminal, now)
                transaction.update(ref, updates)
                return {**data, **updates, "reserved": False}
            revision = _integer(data.get("revision", 1), "revision", minimum=1)
            updates = {"state": "sending" if is_send else "editing", "in_flight_attempt_id": attempt_id,
                       "attempt_revision": revision, "attempt_expires_at": now + NOTIFICATION_ATTEMPT_LEASE,
                       "updated_at": now}
            if rich_markup_fallback:
                updates["presentation"] = None
            numbered_revision = data.get("receipt_numbered_revision")
            numbered_body = (isinstance(data.get("body"), str)
                             and data["body"].startswith("Dialog #"))
            updates["attempt_receipt_ack_revision"] = (
                revision if (data.get("receipt_card") is True and numbered_body
                            and type(numbered_revision) is int and numbered_revision <= revision)
                else None
            )
            if abandoned_send:
                updates["last_outcome"] = "unknown"
                updates["unknown_send_outcome"] = True
            if is_send:
                updates["send_attempts_started"] = attempts + 1
            else:
                edit_attempts = _integer(data.get("edit_attempts_started", 0), "edit_attempts_started")
                per_revision = dict(data.get("edit_attempts_by_revision", {}))
                revision_key = str(revision)
                revision_attempts = (
                    _integer(per_revision.get(revision_key, 0), "revision edit attempts")
                    if data.get("receipt_card") is True else edit_attempts
                )
                if revision_attempts >= NOTIFICATION_MAX_SENDS:
                    updates.update({"state": "failed", "terminal_failure": "failed",
                                    "last_error": "attempts_exhausted", "failed_at": now,
                                    "in_flight_attempt_id": None, "attempt_expires_at": None})
                    self._persist_final_delivery_state(transaction, data, "failed", now)
                    transaction.update(ref, updates)
                    return {**data, **updates, "reserved": False}
                updates["edit_attempts_started"] = edit_attempts + 1
                if data.get("receipt_card") is True:
                    per_revision[revision_key] = revision_attempts + 1
                    updates["edit_attempts_by_revision"] = per_revision
            transaction.update(ref, updates)
            return {**data, **updates, "reserved": True, "attempt_id": attempt_id,
                    "attempt_revision": revision, "is_send": is_send}

        return _txn(self.client.transaction(max_attempts=5))


    def _persist_final_delivery_state(
        self, transaction: Any, notification: Mapping[str, Any], state: str, now: datetime,
    ) -> None:
        """Record terminal final/error delivery without changing save execution state."""
        snapshot = self._read_final_delivery_operation(transaction, notification)
        self._write_final_delivery_state(transaction, snapshot, state, now)


    def _read_final_delivery_operation(
        self, transaction: Any, notification: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]] | None:
        """Read the linked operation before a settlement transaction performs writes."""
        if (notification.get("kind") not in {"final", "error"}
                and not (notification.get("receipt_card") is True
                         and notification.get("receipt_terminal") is True)):
            return
        operation_id = notification.get("operation_id")
        if not isinstance(operation_id, str):
            return
        operation_ref = self.client.collection("dialog_operations").document(operation_id)
        operation_snap = operation_ref.get(transaction=transaction)
        if not operation_snap.exists:
            return
        return operation_ref, operation_snap.to_dict() or {}


    @staticmethod
    def _write_final_delivery_state(
        transaction: Any, snapshot: tuple[Any, dict[str, Any]] | None,
        state: str, now: datetime,
    ) -> None:
        """Apply a previously read terminal delivery update without another read."""
        if snapshot is None:
            return
        operation_ref, operation = snapshot
        previous = operation.get("final_delivery_state")
        if previous == "delivered" and state != "delivered":
            return
        if previous == state:
            return
        transaction.update(operation_ref, {"final_delivery_state": state, "updated_at": now})


    def settle_notification_attempt(
        self, notification_id: str, *, attempt_id: str, revision: int, outcome: str,
        now: datetime, message_id: int | None = None, retry_after: int | None = None,
    ) -> dict[str, Any] | None:
        """Settle the reserved attempt, retaining success from a superseded revision."""
        notification_id = _identifier(notification_id, "notification_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        revision = _integer(revision, "revision", minimum=1)
        if outcome not in {"sent", "edited", "unknown", "rejected", "rate_limited"}:
            raise ValueError("outcome is invalid")
        if retry_after is not None and (type(retry_after) is not int or retry_after < 0):
            retry_after = None
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
                    or data.get("state") in {"delivered", "failed", "delivery_unknown"}):
                return {**data, "settled": False}
            stale_revision = data.get("revision") != revision
            final_operation = self._read_final_delivery_operation(transaction, data)
            ack_draft_ref = None
            ack_draft_snapshot = None
            if (outcome in {"sent", "edited"}
                    and data.get("receipt_card") is True
                    and data.get("attempt_receipt_ack_revision") == revision
                    and isinstance(data.get("context_id"), str)
                    and isinstance(data.get("draft_id"), str)):
                ack_draft_ref = _document_ref(
                    self.client,
                    f"{SESSIONS}/{data['context_id']}/drafts/{data['draft_id']}",
                )
                ack_draft_snapshot = ack_draft_ref.get(transaction=transaction)
            updates: dict[str, Any] = {"in_flight_attempt_id": None, "attempt_expires_at": None,
                                       "last_outcome": outcome, "updated_at": now}
            if outcome in {"sent", "edited"}:
                if outcome == "sent":
                    updates["message_id"] = message_id
                if stale_revision:
                    updates.update({"state": "pending", "terminal_failure": None,
                                    "retry_not_before": None, "last_error": None,
                                    "delivered_at": None, "failed_at": None})
                else:
                    updates.update({"state": "delivered", "terminal_failure": None, "retry_not_before": None,
                                    "last_error": None, "delivered_at": now, "failed_at": None})
                if (data.get("receipt_card") is True
                        and data.get("attempt_receipt_ack_revision") == revision
                        and isinstance(data.get("draft_id"), str)
                        and type(data.get("dialog_number")) is int):
                    updates.update({"first_ack_confirmed": True,
                                    "first_ack_confirmed_at": now,
                                    "first_ack_confirmed_revision": revision})
            else:
                sends = _integer(data.get("send_attempts_started", 0), "send_attempts_started")
                edit_counts = data.get("edit_attempts_by_revision")
                revision_edits = (
                    _integer(edit_counts.get(str(revision), 0), "revision edit attempts")
                    if data.get("receipt_card") is True and isinstance(edit_counts, Mapping) else
                    _integer(data.get("edit_attempts_started", 0), "edit_attempts_started")
                )
                exhausted = (data.get("message_id") is None and sends >= NOTIFICATION_MAX_SENDS) or (
                    data.get("message_id") is not None and revision_edits >= NOTIFICATION_MAX_SENDS
                )
                definitive_send_rejection = (
                    data.get("message_id") is None and outcome == "rejected" and not stale_revision
                )
                if exhausted or definitive_send_rejection:
                    terminal = ("delivery_unknown" if data.get("unknown_send_outcome") or outcome == "unknown"
                                else "failed") if data.get("message_id") is None else "failed"
                    updates.update({"state": terminal, "terminal_failure": terminal, "retry_not_before": None,
                                    "last_error": outcome, "failed_at": now})
                else:
                    delay = min(2 ** max(0, sends - 1), 60)
                    if outcome == "rate_limited" and retry_after is not None:
                        delay = max(delay, retry_after)
                    updates.update({"state": "pending" if stale_revision else "retry",
                                    "retry_not_before": (
                                        None if stale_revision and outcome != "rate_limited"
                                        else _retry_deadline(now, delay)
                                    ),
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
            if (ack_draft_ref is not None and ack_draft_snapshot is not None
                    and ack_draft_snapshot.exists
                    and result.get("attempt_receipt_ack_revision") == revision):
                draft = ack_draft_snapshot.to_dict() or {}
                if draft.get("first_ack_confirmed_at") is None:
                    transaction.update(ack_draft_ref, {
                        "first_ack_confirmed_at": now,
                        "first_ack_notification_id": notification_id,
                        "first_ack_confirmed_revision": revision,
                    })
            transaction.update(ref, updates)
            if result.get("state") in {"delivered", "failed", "delivery_unknown"}:
                self._write_final_delivery_state(transaction, final_operation, result["state"], now)
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
        if draft.get("first_ack_confirmed_at") is not None:
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
