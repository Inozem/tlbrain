"""Fenced persistence boundary for router-confirmed Telegram saves."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from google.cloud import firestore

from core.integrations.firestore.operation_queue import OPERATIONS_SUBCOLLECTION
from services.connectors.telegram.models import TelegramDraftState
from services.connectors.telegram.repository._common import (
    SESSIONS,
    _aware,
    _document_ref,
    _identifier,
    _integer,
)


@dataclass(slots=True)
class _SaveConfirmationState:
    context_ref: Any
    operation_ref: Any
    draft_ref: Any
    manifest_ref: Any
    context: dict[str, Any]
    operation: dict[str, Any]
    draft: dict[str, Any]
    manifest: dict[str, Any] | None
    draft_id: str
    message_ids: list[int]
    manifest_path: str
    manifest_digest: str


def _manifest_digest(draft_id: str, message_ids: list[int]) -> str:
    """Compute the canonical digest for the captured ordered source IDs."""
    return hashlib.sha256(json.dumps(
        {"draft_id": draft_id, "message_ids": message_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _route_identity_matches(operation: Mapping[str, Any], draft_id: str) -> bool:
    """Check route kind and immutable draft/save identity before target reads."""
    route = operation.get("route_result")
    if route is not None and (
        not isinstance(route, Mapping)
        or route.get("kind") != "save"
        or route.get("target_draft_id") not in (None, draft_id)
    ):
        return False
    return (
        operation.get("target_draft_id") in (None, draft_id)
        and operation.get("save_operation_id") in (None, operation.get("operation_id"))
    )


def _route_targets_match(operation: Mapping[str, Any], draft: Mapping[str, Any]) -> bool:
    """Reject stored route targets that point away from the captured draft."""
    route = operation.get("route_result")
    target_fields = {
        "target_draft_id": draft.get("draft_id"),
        "target_dialog_number": draft.get("dialog_number"),
        "target_conversation_id": draft.get("conversation_id"),
        "target_doc_id": draft.get("doc_id"),
    }
    records = [operation]
    if isinstance(route, Mapping):
        records.append(route)
    return all(
        record.get(field) in (None, expected)
        for record in records
        for field, expected in target_fields.items()
    )


def _load_head_state(
    client: Any,
    transaction: Any,
    *,
    context_ref: Any,
    operation_ref: Any,
    context_id: str,
    operation_id: str,
    owner: str,
    fence: int,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Read and validate the live queue head and its server-bound command."""
    context_snapshot = context_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    if not context_snapshot.exists or not operation_snapshot.exists:
        return None
    context = context_snapshot.to_dict() or {}
    operation = operation_snapshot.to_dict() or {}
    expiry = context.get("lock_expires_at")
    if (
        operation.get("operation_id") != operation_id
        or operation.get("state") != "running"
        or context.get("reserved_operation_id") != operation_id
        or context.get("active_operation_id") != operation_id
        or context.get("lock_owner") != owner
        or context.get("fence") != fence
        or not isinstance(expiry, datetime)
        or expiry <= now
    ):
        return None
    input_ref = operation.get("input_ref")
    path_parts = input_ref.split("/") if isinstance(input_ref, str) else []
    if len(path_parts) != 4 or path_parts[:3] != [SESSIONS, context_id, "commands"]:
        return None
    try:
        _identifier(path_parts[3], "command_id")
    except ValueError:
        return None
    command_ref = _document_ref(client, input_ref)
    command_snapshot = command_ref.get(transaction=transaction)
    if not command_snapshot.exists:
        return None
    command = command_snapshot.to_dict() or {}
    if (
        command.get("context_id") != context_id
        or command.get("operation_id") != operation_id
        or command.get("kind") not in {"freeform", "save"}
    ):
        return None
    return context, operation, command


def _load_seal_state(
    transaction: Any,
    *,
    context_ref: Any,
    operation_ref: Any,
    context: dict[str, Any],
    operation: dict[str, Any],
    command: Mapping[str, Any],
    operation_id: str,
    context_id: str,
) -> _SaveConfirmationState | None:
    """Read the captured draft and manifest and validate their route binding."""
    draft_id = command.get("draft_id")
    message_ids = command.get("cutoff_message_ids")
    if not isinstance(draft_id, str) or not isinstance(message_ids, list) or not message_ids:
        return None
    try:
        _identifier(draft_id, "draft_id")
        captured_ids = [_integer(value, "message_id", minimum=1) for value in message_ids]
    except ValueError:
        return None
    if len(set(captured_ids)) != len(captured_ids) or not _route_identity_matches(operation, draft_id):
        return None

    draft_ref = context_ref.collection("drafts").document(draft_id)
    manifest_ref = draft_ref.collection("save_manifests").document(operation_id)
    draft_snapshot = draft_ref.get(transaction=transaction)
    manifest_snapshot = manifest_ref.get(transaction=transaction)
    if not draft_snapshot.exists:
        return None
    draft = draft_snapshot.to_dict() or {}
    manifest = manifest_snapshot.to_dict() or {} if manifest_snapshot.exists else None
    if draft.get("draft_id") != draft_id or not _route_targets_match(operation, draft):
        return None
    digest = _manifest_digest(draft_id, captured_ids)
    return _SaveConfirmationState(
        context_ref=context_ref,
        operation_ref=operation_ref,
        draft_ref=draft_ref,
        manifest_ref=manifest_ref,
        context=context,
        operation=operation,
        draft=draft,
        manifest=manifest,
        draft_id=draft_id,
        message_ids=captured_ids,
        manifest_path=f"{SESSIONS}/{context_id}/drafts/{draft_id}/save_manifests/{operation_id}",
        manifest_digest=digest,
    )


def _matching_seal(state: _SaveConfirmationState, operation_id: str) -> bool:
    """Check whether the captured draft and manifest already prove this save."""
    draft, manifest = state.draft, state.manifest
    return (
        draft.get("state") == TelegramDraftState.PROCESSING.value
        and draft.get("ordered_message_ids") == state.message_ids
        and draft.get("manifest_ref") == state.manifest_path
        and draft.get("manifest_digest") == state.manifest_digest
        and draft.get("active_save_operation_id") == operation_id
        and draft.get("immutable_save_intent_id") == operation_id
        and manifest is not None
        and manifest.get("draft_id") == state.draft_id
        and manifest.get("ordered_message_ids") == state.message_ids
        and manifest.get("manifest_digest") == state.manifest_digest
        and manifest.get("save_operation_id") == operation_id
    )


def _route_is_bound(state: _SaveConfirmationState, operation_id: str) -> bool:
    """Check whether the existing save route and pending hint already agree."""
    route = state.operation.get("route_result") or {}
    return (
        state.operation.get("target_draft_id") == state.draft_id
        and state.operation.get("save_operation_id") == operation_id
        and route.get("kind") == "save"
        and route.get("target_draft_id") == state.draft_id
        and state.context.get("pending_save_operation_id") == operation_id
    )


def _persist_confirmation(
    transaction: Any,
    *,
    state: _SaveConfirmationState,
    operation_id: str,
    now: datetime,
) -> bool:
    """Persist an exact seal or bind missing route links on a verified legacy seal."""
    if _matching_seal(state, operation_id):
        if _route_is_bound(state, operation_id):
            return True
        route = dict(state.operation.get("route_result") or {})
        route.update({"kind": "save", "target_draft_id": state.draft_id})
        state.operation.update({
            "target_draft_id": state.draft_id,
            "save_operation_id": operation_id,
            "route_result": route,
        })
        state.context["pending_save_operation_id"] = operation_id
        transaction.set(state.operation_ref, state.operation)
        transaction.set(state.context_ref, state.context)
        return True

    if (
        state.draft.get("state") != TelegramDraftState.COLLECTING.value
        or state.draft.get("ordered_message_ids") != state.message_ids
        or state.manifest is not None
        or state.draft.get("manifest_ref") is not None
        or state.draft.get("manifest_digest") is not None
        or state.draft.get("active_save_operation_id") is not None
        or state.draft.get("immutable_save_intent_id") is not None
    ):
        return False

    route = dict(state.operation.get("route_result") or {})
    route.update({"kind": "save", "target_draft_id": state.draft_id})
    state.operation.update({
        "target_draft_id": state.draft_id,
        "save_operation_id": operation_id,
        "route_result": route,
    })
    state.draft.update({
        "state": TelegramDraftState.PROCESSING.value,
        "sealed_at": now,
        "manifest_ref": state.manifest_path,
        "manifest_digest": state.manifest_digest,
        "active_save_operation_id": operation_id,
        "immutable_save_intent_id": operation_id,
    })
    if state.context.get("active_draft_id") == state.draft_id:
        state.context.update({"active_draft_id": None, "draft_status": None})
    state.context["pending_save_operation_id"] = operation_id
    transaction.set(state.manifest_ref, {
        "draft_id": state.draft_id,
        "ordered_message_ids": state.message_ids,
        "manifest_digest": state.manifest_digest,
        "save_operation_id": operation_id,
        "sealed_at": now,
    })
    transaction.set(state.draft_ref, state.draft)
    transaction.set(state.operation_ref, state.operation)
    transaction.set(state.context_ref, state.context)
    return True


class _TelegramSaveCommandMixin:
    """Seal only a candidate confirmed by a trusted structured router result."""

    def confirm_save_command(
        self,
        context_id: str,
        operation_id: str,
        *,
        owner: str,
        fence: int,
        now: datetime,
    ) -> bool:
        """Bind a live queue head to its captured nonempty draft and seal atomically."""
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        fence = _integer(fence, "fence", minimum=1)
        now = _aware(now, "now")
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection(OPERATIONS_SUBCOLLECTION).document(operation_id)

        @firestore.transactional
        def _txn(transaction: firestore.Transaction) -> bool:
            head = _load_head_state(
                self.client, transaction, context_ref=context_ref, operation_ref=operation_ref,
                context_id=context_id, operation_id=operation_id, owner=owner, fence=fence, now=now,
            )
            if head is None:
                return False
            context, operation, command = head
            state = _load_seal_state(
                transaction, context_ref=context_ref, operation_ref=operation_ref,
                context=context, operation=operation, command=command,
                operation_id=operation_id, context_id=context_id,
            )
            if state is None:
                return False
            return _persist_confirmation(
                transaction, state=state, operation_id=operation_id, now=now,
            )

        return _txn(self.client.transaction(max_attempts=5))


__all__ = ["_TelegramSaveCommandMixin"]
