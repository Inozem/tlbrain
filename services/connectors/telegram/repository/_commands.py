"""Transactional Telegram command acceptance and queue registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from google.cloud import firestore

from core.operations.registry import OPERATIONS, validate_arguments
from core.integrations.firestore.outbox import (
    OutboxTargetKind,
    create_outbox_record_in_transaction,
    prepare_outbox_record_in_transaction,
    write_prepared_outbox_record,
)
from core.integrations.firestore.operation_queue import (
    register_operation_in_transaction,
    validate_context_queue_lease,
)
from services.connectors.telegram.models import TelegramDraftState, build_context_id
from services.connectors.telegram.repository._common import (
    SESSIONS,
    TelegramIntakeResult,
    _document_ref,
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
    draft_data: dict[str, Any]
    message_ids: tuple[int, ...]
    cutoff_sequence: int
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
                {"kind": command.kind, "text": command.text,
                 "action_metadata": command.action_metadata},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _receipt_save_control_matches(
    *, context_exists: bool, context: Mapping[str, Any], control: Mapping[str, Any],
    row: Mapping[str, Any], draft: Mapping[str, Any], context_id: str,
    control_token: str, notification_id: Any, user_id: int, message_id: int,
) -> bool:
    """Check persisted actor, card, revision, draft, and queue bindings."""
    return bool(
        context_exists and control.get("state") == "active"
        and control.get("token") == control_token
        and control.get("context_id") == context_id
        and control.get("owner_user_id") == user_id
        and control.get("notification_id") == notification_id
        and row.get("receipt_card") is True
        and row.get("receipt_control_token") == control_token
        and row.get("receipt_control_owner_user_id") == user_id
        and row.get("revision") == control.get("issued_revision")
        and row.get("message_id") == message_id
        and row.get("chat_id") == context.get("chat_id")
        and row.get("thread_id") == context.get("thread_id")
        and row.get("context_id") == context_id
        and row.get("draft_id") == draft.get("draft_id")
        and context.get("latest_receipt_notification_id") == notification_id
        and context.get("active_draft_id") == draft.get("draft_id")
        and context.get("draft_status") == TelegramDraftState.COLLECTING.value
        and not context.get("reserved_operation_id")
        and not context.get("active_operation_id")
        and not context.get("pending_save_operation_id")
        and context.get("pending_receipt_groups", 0) == 0
        and draft.get("state") == TelegramDraftState.COLLECTING.value
        and draft.get("dialog_number") == row.get("dialog_number")
        and bool(draft.get("ordered_message_ids"))
    )


def _load_command_snapshot(
    client: Any,
    transaction: Any,
    *,
    context_id: str,
    context: Mapping[str, Any],
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
        draft_data=draft_data,
        message_ids=message_ids,
        cutoff_sequence=cutoff_sequence,
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
        **({"action_metadata": dict(command.action_metadata)}
           if isinstance(command.action_metadata, Mapping) else {}),
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
    action_metadata: Mapping[str, Any] | None = None,
    ingress_binding_id: str | None = None,
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

    operation_fields = _command_link_metadata(snapshot)
    if ingress_binding_id is not None:
        operation_fields["ingress_binding_id"] = ingress_binding_id
    if action_metadata is not None:
        operation_fields["typed_action"] = dict(action_metadata)
    return register_operation_in_transaction(
        client,
        transaction,
        context_ref,
        identity.operation_id,
        identity_digest=identity.identity_digest,
        input_ref=identity.input_ref,
        now=now,
        before_write=create_command_outbox,
        operation_fields=operation_fields,
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


def _persist_command_acceptance(
    transaction: Any,
    *,
    identity: _CommandIdentity,
    update_id: int,
    state: _CommandAcceptanceState,
    operation: Any,
    now: datetime,
    ingress_binding: Mapping[str, Any] | None = None,
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
        "snapshot_message_ids": list(snapshot.message_ids),
        "created_at": now,
    }
    transaction.set(identity.command_ref, state.command_data)
    transaction.set(identity.context_ref, state.context)
    transaction.set(identity.receipt_ref, receipt)
    if ingress_binding is not None:
        binding_ref = identity.context_ref.collection("ingress_bindings").document(
            identity.operation_id
        )
        transaction.set(binding_ref, dict(ingress_binding))
    return TelegramIntakeResult(
        True,
        identity.context_id,
        snapshot.draft_id,
        snapshot.draft_data.get("dialog_number"),
        outbox_id=identity.outbox_id,
        operation_id=identity.operation_id,
        queue_position=operation.sequence,
        snapshot_message_ids=snapshot.message_ids,
    )


def _bind_text_to_recent_receipt_group(
    client: Any, transaction: Any, *, context_id: str, context: Mapping[str, Any],
    draft_id: str | None, user_id: int, message_id: int | None,
    transport_timestamp: int | None,
) -> dict[str, Any] | None:
    """Return an exact forward-first burst binding when transport evidence agrees."""
    group_id = context.get("latest_receipt_group_id")
    if (not isinstance(group_id, str) or not isinstance(draft_id, str)
            or message_id is None or type(transport_timestamp) is not int):
        return None
    group_ref = client.collection(SESSIONS).document(context_id).collection(
        "receipt_groups"
    ).document(group_id)
    snapshot = group_ref.get(transaction=transaction)
    group = snapshot.to_dict() or {} if snapshot.exists else {}
    start, end = group.get("transport_start_timestamp"), group.get("transport_end_timestamp")
    max_message_id = group.get("max_message_id")
    source_ids = group.get("source_message_ids")
    if not isinstance(source_ids, list):
        source_ids = []
    source_ids = [value for value in source_ids if type(value) is int]
    if type(max_message_id) is not int:
        max_message_id = max(source_ids, default=0)
    distance = (start - transport_timestamp if type(start) is int and transport_timestamp < start
                else transport_timestamp - end if type(end) is int and transport_timestamp > end
                else 0)
    if not (
        group.get("context_id") in (None, context_id)
        and group.get("draft_id") == draft_id
        and group.get("owner_user_id") in (None, 0, user_id)
        and group.get("state") in {"open", "closed"}
        and type(start) is int and type(end) is int and distance <= 1
        and max_message_id < message_id
    ):
        return None
    return {
        "draft_id": draft_id,
        "receipt_group_id": group_id,
        "matched_message_ids": source_ids,
        "state": "collecting",
    }


class _TelegramCommandMixin:
    """Persist idempotent command acceptance and operation registration."""

    @staticmethod
    def _route_control_data(route: Mapping[str, Any]) -> dict[str, Any]:
        """Project route state to metadata safe for the operation ledger."""
        allowed_status = {
            "ok", "success", "invalid_route", "uncertain", "error",
            "not_found", "unconfirmed",
        }
        allowed_kinds = {"pure_switch", "search", "substantive", "save", "retry", "annotation"}
        status = route.get("status")
        kind = route.get("kind")
        if not isinstance(status, str) or status not in allowed_status:
            raise ValueError("route status is invalid")
        if kind is not None and (not isinstance(kind, str) or kind not in allowed_kinds):
            raise ValueError("route kind is invalid")
        safe: dict[str, Any] = {"status": status}
        if kind is not None:
            safe["kind"] = kind
        selection_policy = route.get("selection_policy")
        if selection_policy is not None:
            if selection_policy != "silent_v1" or kind not in {"pure_switch", "substantive"}:
                raise ValueError("selection_policy is invalid")
            safe["selection_policy"] = selection_policy
        for field in (
            "target_doc_id", "conversation_id", "target_conversation_id", "target_draft_id",
            "target_operation_id", "dependency_operation_id",
        ):
            value = route.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field} is invalid")
                safe[field] = value
        retry_request_operation_id = route.get("retry_request_operation_id")
        if retry_request_operation_id is not None:
            if kind != "retry":
                raise ValueError("retry_request_operation_id is only valid for retry routes")
            safe["retry_request_operation_id"] = _identifier(
                retry_request_operation_id, "retry_request_operation_id",
            )
        number = route.get("dialog_number")
        if number is not None:
            if type(number) is not int or number < 1:
                raise ValueError("dialog_number is invalid")
            safe["dialog_number"] = number
        if retry_request_operation_id is not None and number is None:
            raise ValueError("retry work routes require a dialog_number")
        choice = route.get("tool_choice")
        search_mode = route.get("search_protocol") == "agent-v1"
        if choice is not None:
            if kind not in {"substantive", "search"} or not isinstance(choice, Mapping) or set(choice) - {
                "name", "arguments", "link_requested",
            }:
                raise ValueError("tool_choice is invalid")
            name = choice.get("name")
            link_requested = choice.get("link_requested", False)
            try:
                arguments = validate_arguments(name, choice.get("arguments"))
            except (TypeError, ValueError) as exc:
                raise ValueError("tool_choice is invalid") from exc
            if (not isinstance(name, str) or name not in OPERATIONS
                    or not isinstance(link_requested, bool)
                    or (link_requested and name != "get_transcript")
                    or (kind == "search" and (name != "query" or link_requested))
                    or (not search_mode and arguments.get("doc_id") is not None
                        and arguments["doc_id"] != safe.get("target_doc_id"))):
                raise ValueError("tool_choice is invalid")
            safe["tool_choice"] = {
                "name": name, "arguments": arguments,
                "link_requested": link_requested,
            }
        if kind == "search":
            if route.get("search_protocol") != "agent-v1" or choice is None:
                raise ValueError("search route protocol is invalid")
            safe["search_protocol"] = "agent-v1"
        refs = route.get("result_refs")
        if refs is not None:
            if (kind != "search" or not isinstance(refs, list)
                    or any(not isinstance(ref, str) or not ref for ref in refs)):
                raise ValueError("result_refs is invalid")
            safe["result_refs"] = list(refs)
        session_id = route.get("search_session_id")
        if session_id is not None:
            if kind != "search" or not isinstance(session_id, str) or not session_id:
                raise ValueError("search_session_id is invalid")
            safe["search_session_id"] = session_id
        if kind == "annotation":
            payload = route.get("annotation_payload")
            text_digest = route.get("annotation_text_sha256")
            legacy_text = route.get("annotation_text")
            if (payload is None and isinstance(legacy_text, str) and legacy_text.strip()):
                safe["annotation_text"] = legacy_text
            elif (not isinstance(payload, Mapping)
                    or not isinstance(text_digest, str)
                    or len(text_digest) != 64
                    or any(char not in "0123456789abcdef" for char in text_digest)):
                raise ValueError("annotation payload is invalid")
            if isinstance(payload, Mapping):
                inline = payload.get("_inline")
                input_ref = payload.get("_input_ref")
                if (isinstance(inline, str) and set(payload) <= {
                        "_inline", "sha256", "byte_length"
                    }):
                    if (not inline.strip() or type(payload.get("byte_length")) is not int
                        or payload["byte_length"] < 0
                        or not isinstance(payload.get("sha256"), str)):
                        raise ValueError("inline annotation payload is invalid")
                elif (isinstance(input_ref, str)
                    and input_ref.startswith("dialog_operations/telegram-ledger-")
                    and input_ref.endswith("/input_parts/manifest")
                    and type(payload.get("byte_length")) is int
                    and payload["byte_length"] > 0
                    and isinstance(payload.get("sha256"), str)
                    and type(payload.get("max_part_bytes")) is int):
                    safe["annotation_payload"] = dict(payload)
                else:
                    raise ValueError("annotation payload reference is invalid")
            if isinstance(payload, Mapping) and "_inline" in payload:
                safe["annotation_payload"] = dict(payload)
            if isinstance(text_digest, str):
                safe["annotation_text_sha256"] = text_digest
        if route.get("search_protocol") == "agent-v1" and kind != "search":
            if (kind != "substantive" or not isinstance(safe.get("conversation_id"), str)
                    or not safe.get("conversation_id")):
                raise ValueError("search continuation binding is invalid")
            safe["search_protocol"] = "agent-v1"
        return safe

    def _command_for_operation(
        self, transaction: Any, *, context_id: str, operation_id: str,
        operation: Mapping[str, Any],
    ):
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if (len(parts) != 4 or parts[:3] != [SESSIONS, context_id, "commands"]):
            return None, None
        try:
            _identifier(parts[3], "command_id")
        except ValueError:
            return None, None
        command_ref = _document_ref(self.client, input_ref)
        snapshot = command_ref.get(transaction=transaction)
        command = snapshot.to_dict() or {} if snapshot.exists else {}
        if (not snapshot.exists or command.get("context_id") != context_id
                or command.get("operation_id") != operation_id):
            return None, None
        return command_ref, command

    def _retry_work_route_is_pinned(
        self, transaction: Any, *, context_id: str, operation_id: str,
        operation: Mapping[str, Any], route: Mapping[str, Any],
    ) -> bool:
        """Validate a retry work replay against its immutable source command."""
        request_operation_id = operation.get("retry_request_operation_id")
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        pinned = operation.get("route_control")
        number = route.get("dialog_number")
        if (
            operation.get("operation_id") != operation_id
            or not isinstance(request_operation_id, str) or not request_operation_id
            or len(parts) != 4 or parts[:3] != [SESSIONS, context_id, "commands"]
            or not parts[3].isdigit()
            or route.get("kind") != "retry"
            or route.get("status") not in {"ok", "success"}
            or route.get("retry_request_operation_id") != request_operation_id
            or type(number) is not int or number < 1
            or not isinstance(pinned, Mapping) or dict(pinned) != dict(route)
        ):
            return False

        source_ref = _document_ref(self.client, input_ref)
        source_snapshot = source_ref.get(transaction=transaction)
        source_command = source_snapshot.to_dict() or {} if source_snapshot.exists else {}
        source_operation_ref = self.client.collection(SESSIONS).document(context_id).collection(
            "operations"
        ).document(request_operation_id)
        source_operation_snapshot = source_operation_ref.get(transaction=transaction)
        source_operation = (
            source_operation_snapshot.to_dict() or {}
            if source_operation_snapshot.exists else {}
        )
        receipt_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/updates/{parts[3]}",
        )
        receipt_snapshot = receipt_ref.get(transaction=transaction)
        receipt = receipt_snapshot.to_dict() or {} if receipt_snapshot.exists else {}
        source_route = source_operation.get("route_control")
        return bool(
            source_command.get("context_id") == context_id
            and source_command.get("operation_id") == request_operation_id
            and source_command.get("kind") == "freeform"
            and source_command.get("update_id") == int(parts[3])
            and source_operation.get("operation_id") == request_operation_id
            and source_operation.get("input_ref") == input_ref
            and isinstance(source_route, Mapping)
            and source_route.get("status") in {"ok", "success"}
            and source_route.get("kind") == "retry"
            and type(source_route.get("dialog_number")) is int
            and source_route.get("dialog_number") == number
            and receipt.get("accepted") is True
            and receipt.get("context_id") == context_id
            and receipt.get("operation_id") == request_operation_id
        )

    def persist_route(
        self, operation_id: str, route: Mapping[str, Any], *, context_id: str,
        owner: str, fence: int, now: datetime,
    ) -> bool:
        """Persist sanitized route state only under the current operation lease."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(route, Mapping):
            raise TypeError("route must be a mapping")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        safe_route = self._route_control_data(route)
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not context_snapshot.exists or not operation_snapshot.exists:
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if (operation.get("operation_id") != operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"),
                    )):
                return False
            _, command = self._command_for_operation(
                transaction, context_id=context_id,
                operation_id=operation_id, operation=operation,
            )
            if command is None:
                existing_route = operation.get("route_control")
                if (not isinstance(existing_route, Mapping)
                        or not self._retry_work_route_is_pinned(
                            transaction, context_id=context_id,
                            operation_id=operation_id, operation=operation,
                            route=safe_route,
                        )):
                    return False
                return dict(existing_route) == safe_route
            existing_route = operation.get("route_control")
            if isinstance(existing_route, Mapping) and existing_route.get("status") in {
                "ok", "success",
            }:
                return dict(existing_route) == safe_route
            transaction.update(operation_ref, {
                "route_control": safe_route,
                "route_control_recorded_at": now,
            })
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def get_route_control(self, context_id: str, operation_id: str) -> dict[str, Any] | None:
        """Read the allowlisted pinned route for safe operation replay."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        ref = self.client.collection(SESSIONS).document(context_id).collection("operations").document(operation_id)
        snapshot = ref.get()
        if not snapshot.exists:
            return None
        value = (snapshot.to_dict() or {}).get("route_control")
        return dict(value) if isinstance(value, Mapping) else None

    def get_ingress_binding(self, context_id: str, operation_id: str) -> dict[str, Any] | None:
        """Read this operation's server-owned provisional forwarding association."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        snapshot = (self.client.collection(SESSIONS).document(context_id)
                    .collection("ingress_bindings").document(operation_id).get())
        row = snapshot.to_dict() or {} if snapshot.exists else None
        if (not isinstance(row, Mapping) or row.get("context_id") != context_id
                or row.get("operation_id") != operation_id):
            return None
        return dict(row)

    def get_switch_notice_state(self, context_id: str, operation_id: str) -> str | None:
        """Read the durable switch-notice disposition for transport replay."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        ref = self.client.collection(SESSIONS).document(context_id).collection("operations").document(operation_id)
        snapshot = ref.get()
        state = (snapshot.to_dict() or {}).get("switch_notice_state") if snapshot.exists else None
        return state if state in {"pending", "delivered", "failed", "delivery_unknown"} else None

    def set_switch_notice_state(
        self, context_id: str, operation_id: str, state: str, *,
        owner: str, fence: int, now: datetime,
    ) -> bool:
        """Persist terminal notice disposition without changing the pinned target."""
        if state not in {"pending", "delivered", "failed", "delivery_unknown"}:
            raise ValueError("switch notice state is invalid")
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        fence, now = _integer(fence, "fence", minimum=1), _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not context_snapshot.exists or not operation_snapshot.exists:
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if (operation.get("operation_id") != operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"),
                    )):
                return False
            if not isinstance(operation.get("route_control"), Mapping):
                return False
            previous = operation.get("switch_notice_state")
            if previous in {"failed", "delivery_unknown"}:
                return state == previous
            if previous == "delivered" and state != "delivered":
                return False
            if previous == state:
                return True
            transaction.update(operation_ref, {"switch_notice_state": state, "updated_at": now})
            return True

        return _txn(self.client.transaction(max_attempts=5))

    def clear_command_candidate(
        self, context_id: str, operation_id: str, *, owner: str, fence: int,
        now: datetime,
    ) -> bool:
        """Delete only a successfully routed candidate under its current fence.

        The route record and command link are rechecked in this transaction.
        Replays after deletion are idempotent and preserve every other field.
        """
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)

        @firestore.transactional
        def _txn(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            if not context_snapshot.exists or not operation_snapshot.exists:
                return False
            context = context_snapshot.to_dict() or {}
            operation = operation_snapshot.to_dict() or {}
            if (operation.get("operation_id") != operation_id
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence, now=now,
                        expiry=context.get("lock_expires_at"),
                    )):
                return False
            route = operation.get("route_control")
            if (not isinstance(route, Mapping)
                    or route.get("status") not in {"ok", "success"}):
                return False
            command_ref, command = self._command_for_operation(
                transaction, context_id=context_id,
                operation_id=operation_id, operation=operation,
            )
            if command_ref is None or command is None:
                return False
            if "candidate" not in command:
                return True
            if not isinstance(command.get("candidate"), str):
                return False
            transaction.update(command_ref, {"candidate": firestore.DELETE_FIELD})
            return True

        return _txn(self.client.transaction(max_attempts=5))

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
        message_id: int | None,
        transport_timestamp: int | None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Coordinate the all-reads-first freeform command reservation."""
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
        ingress_binding = None
        if command.kind == "freeform" and message_id is not None:
            recent_source = _bind_text_to_recent_receipt_group(
                self.client, transaction, context_id=identity.context_id,
                context=state.context, draft_id=state.snapshot.draft_id,
                user_id=user_id, message_id=message_id,
                transport_timestamp=transport_timestamp,
            )
            ingress_binding = {
                "context_id": identity.context_id,
                "operation_id": identity.operation_id,
                "owner_user_id": user_id,
                "candidate_message_id": message_id,
                "transport_timestamp": transport_timestamp,
                "state": "collecting" if recent_source else "pending",
                "draft_id": recent_source.get("draft_id") if recent_source else None,
                "receipt_group_id": recent_source.get("receipt_group_id") if recent_source else None,
                "matched_message_ids": recent_source.get("matched_message_ids", [])
                if recent_source else [],
                "source_binding_finalized": bool(recent_source),
                "created_at": now,
                "deadline_at": now + timedelta(seconds=1 if recent_source else 5),
                "max_deadline_at": now + timedelta(seconds=30),
            }
        operation = _register_command_operation(
            self.client,
            transaction,
            context_ref=identity.context_ref,
            identity=identity,
            now=now,
            snapshot=state.snapshot,
            action_metadata=(command.action_metadata
                             if isinstance(command.action_metadata, Mapping) else None),
            ingress_binding_id=(identity.operation_id if ingress_binding is not None else None),
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
        if (ingress_binding is not None
                and state.context.get("reserved_operation_id") == identity.operation_id):
            state.context["pending_ingress_operation_id"] = identity.operation_id
        return _persist_command_acceptance(
            transaction,
            identity=identity,
            update_id=update_id,
            state=state,
            operation=operation,
            now=now,
            ingress_binding=ingress_binding,
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
        message_id: int | None = None,
        transport_timestamp: int | None = None,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Accept command text and reserve its FIFO position with a save cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        now = _aware(now, "now")
        if thread_id == 0:
            thread_id = None
        elif thread_id is not None:
            thread_id = _integer(thread_id, "thread_id", minimum=1)
        if topic_name is not None and not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string or None")
        if not isinstance(command, TelegramCommand):
            raise TypeError("command must be a TelegramCommand")
        if message_id is not None:
            message_id = _integer(message_id, "message_id", minimum=1)
        if transport_timestamp is not None:
            transport_timestamp = _integer(transport_timestamp, "transport_timestamp", minimum=1)
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
                message_id=message_id,
                transport_timestamp=transport_timestamp,
                now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))


    def accept_receipt_save_callback(
        self,
        update_id: int,
        *,
        chat_id: int,
        user_id: int,
        thread_id: int | None,
        message_id: int,
        control_token: str,
        now: datetime,
    ) -> TelegramIntakeResult:
        """Consume a current persisted Save control and reserve its exact draft cutoff."""
        from services.connectors.telegram.commands import TelegramCommand

        update_id = _integer(update_id, "update_id")
        chat_id = _integer(chat_id, "chat_id")
        user_id = _integer(user_id, "user_id", minimum=1)
        message_id = _integer(message_id, "message_id", minimum=1)
        thread_id = (None if thread_id == 0 else
                     _integer(thread_id, "thread_id", minimum=1)
                     if thread_id is not None else None)
        control_token = _identifier(control_token, "control_token")
        now = _aware(now, "now")
        context_id = build_context_id(chat_id, thread_id)
        receipt_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}/updates/{update_id}")
        control_ref = _document_ref(
            self.client, f"{SESSIONS}/{context_id}/receipt_controls/{control_token}",
        )
        control_snapshot = control_ref.get()
        control_data = control_snapshot.to_dict() or {} if control_snapshot.exists else {}
        notification_id = control_data.get("notification_id")
        notification_ref = (self.client.collection("notification_outbox").document(notification_id)
                            if isinstance(notification_id, str) else None)
        notification_snapshot = notification_ref.get() if notification_ref is not None else None
        notification = (notification_snapshot.to_dict() or {}
                        if notification_snapshot is not None and notification_snapshot.exists else {})
        context_ref = self.client.collection(SESSIONS).document(context_id)

        @firestore.transactional
        def _txn(transaction: Any) -> TelegramIntakeResult:
            replay_snapshot = receipt_ref.get(transaction=transaction)
            if replay_snapshot.exists:
                return _receipt_result(replay_snapshot.to_dict() or {}, context_id)
            control_now = control_ref.get(transaction=transaction)
            control = control_now.to_dict() or {} if control_now.exists else {}
            context_snapshot = context_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            live_notification_snapshot = (
                notification_ref.get(transaction=transaction)
                if notification_ref is not None else None
            )
            row = (live_notification_snapshot.to_dict() or {}
                   if live_notification_snapshot is not None and live_notification_snapshot.exists else {})
            draft_id = row.get("draft_id")
            draft_ref = (_document_ref(self.client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
                         if isinstance(draft_id, str) else None)
            draft_snapshot = draft_ref.get(transaction=transaction) if draft_ref is not None else None
            draft = (draft_snapshot.to_dict() or {}
                     if draft_snapshot is not None and draft_snapshot.exists else {})

            eligible = _receipt_save_control_matches(
                context_exists=context_snapshot.exists, context=context,
                control=control, row=row, draft=draft, context_id=context_id,
                control_token=control_token, notification_id=notification_id,
                user_id=user_id, message_id=message_id,
            )
            if not eligible:
                transaction.set(receipt_ref, {
                    "update_id": update_id, "context_id": context_id,
                    "accepted": False, "outcome": "receipt_save_rejected",
                    "notification_id": notification_id, "message_id": message_id,
                    "created_at": now,
                })
                return TelegramIntakeResult(False, context_id)

            action_metadata = {
                "action": "receipt_save", "notification_id": notification_id,
                "control_token": control_token, "draft_id": draft_id,
                "dialog_number": row.get("dialog_number"), "card_message_id": message_id,
                "card_revision": row.get("revision"), "user_id": user_id,
            }
            command = TelegramCommand("save", "", action_metadata)
            identity = _prepare_command_identity(
                self.client, context_id=context_id, update_id=update_id, command=command,
            )
            saving_outbox_id = "tgcard-saving-" + hashlib.sha256(
                f"{notification_id}:{identity.operation_id}".encode()
            ).hexdigest()
            saving_outbox = prepare_outbox_record_in_transaction(
                self.client, transaction, saving_outbox_id,
                target_kind=OutboxTargetKind.TELEGRAM_DISPATCH, generation=1,
                schedule_at=now, now=now, context_id=context_id,
                notification_id=notification_id, operation_id=identity.operation_id,
            )
            result = self._accept_command_in_transaction(
                transaction, identity=identity, update_id=update_id,
                chat_id=chat_id, user_id=user_id, thread_id=thread_id,
                topic_name=context.get("topic_name"), command=command, now=now,
                message_id=None, transport_timestamp=None,
            )
            write_prepared_outbox_record(transaction, saving_outbox)
            next_revision = _integer(row.get("revision", 1), "revision", minimum=1) + 1
            row.update({
                "body": f"Dialog #{row.get('dialog_number')}: Saving…",
                "reply_markup": {"inline_keyboard": []},
                "receipt_control_token": None,
                "receipt_control_owner_user_id": None,
                "save_operation_id": result.operation_id,
                "operation_id": result.operation_id,
                "revision": next_revision,
                "desired_status_revision": next_revision,
                "state": row.get("state") if row.get("in_flight_attempt_id") else "pending",
                "delivered_at": None, "updated_at": now,
            })
            transaction.set(notification_ref, row)
            transaction.update(control_ref, {"state": "consumed", "operation_id": result.operation_id,
                                             "consumed_at": now})
            transaction.update(receipt_ref, {
                "notification_id": notification_id, "receipt_group_id": row.get("receipt_group_id"),
                "card_message_id": message_id,
                "additional_outbox_ids": [saving_outbox_id],
            })
            return TelegramIntakeResult(
                result.accepted, result.context_id, result.draft_id, result.dialog_number,
                result.message_id, result.intake_sequence, result.notification_id,
                result.outbox_id, result.operation_id, result.queue_position,
                result.snapshot_message_ids, (saving_outbox_id,),
            )

        return _txn(self.client.transaction(max_attempts=5))
