"""Thin Cloud Tasks boundary for durable MCP dialog save recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from core.integrations.firestore import outbox as outbox_repository
from core.integrations.firestore.outbox import (
    OutboxState,
    OutboxTargetKind,
    PAYLOAD_SCHEMA_VERSION,
)
from core.integrations.firestore.operations import get_operation
from core.integrations.firestore import mcp_sessions
from core.integrations.firestore.operations import claim_operation, release_operation
from core.dialogs import completion as dialog_completion
from core.integrations.firestore import folders as folder_repository
from core.config import get_root_folder_id


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,480}$")
_FIELDS = {
    "schema_version", "outbox_id", "target_kind", "generation",
    "operation_id", "context_id", "document_id", "save_intent_id",
}


class InvalidTask(ValueError):
    """The callback body does not match the durable task wire contract."""


class RecoveryBindingUnavailable(RuntimeError):
    """The original MCP completion binding is not durably reconstructable."""


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    status: str


def validate_task_envelope(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) - _FIELDS:
        raise InvalidTask("invalid envelope")
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION or type(payload.get("schema_version")) is not int:
        raise InvalidTask("invalid envelope")
    if payload.get("target_kind") != OutboxTargetKind.MCP_SAVE_RECOVERY.value:
        raise InvalidTask("invalid envelope")
    generation = payload.get("generation")
    if type(generation) is not int or generation < 1:
        raise InvalidTask("invalid envelope")
    for key in ("outbox_id", "operation_id"):
        value = payload.get(key)
        if not isinstance(value, str) or not _ID_RE.fullmatch(value):
            raise InvalidTask("invalid envelope")
    for key in ("context_id", "document_id", "save_intent_id"):
        if key in payload and (not isinstance(payload[key], str) or not _ID_RE.fullmatch(payload[key])):
            raise InvalidTask("invalid envelope")
    return payload


def process_dialog_operation(client, payload: object) -> RecoveryResult:
    """Confirm durable identities, then resume the existing checkpointed save."""
    task = validate_task_envelope(payload)
    record = outbox_repository.get_outbox_record(client, task["outbox_id"])
    if record is None:
        return RecoveryResult("stale")
    if (record.target_kind is not OutboxTargetKind.MCP_SAVE_RECOVERY
            or record.generation != task["generation"]
            or record.operation_id != task["operation_id"]):
        return RecoveryResult("stale")
    for field in ("context_id", "document_id", "save_intent_id"):
        if field in task and getattr(record, field, None) != task[field]:
            return RecoveryResult("stale")
    if record.state is OutboxState.FAILED:
        return RecoveryResult("duplicate")

    operation = get_operation(client, task["operation_id"])
    if operation is None:
        return RecoveryResult("stale")
    if operation.state.is_terminal:
        return RecoveryResult("terminal")
    if operation.origin.value != "mcp" or not operation.save_intent_id:
        return RecoveryResult("stale")
    if ("save_intent_id" in task and task["save_intent_id"] != operation.save_intent_id):
        return RecoveryResult("stale")

    binding = mcp_sessions.get_recovery_binding(
        client, getattr(operation, "origin_operation_ref", None) or "",
        operation_id=operation.operation_id,
        immutable_input_digest=operation.immutable_input_digest,
    )
    if binding is None:
        raise RecoveryBindingUnavailable("MCP recovery binding is unavailable")

    now = datetime.now(timezone.utc)
    owner = "mcp-dialog-recovery-" + uuid4().hex
    lease = claim_operation(client, operation.operation_id, owner=owner, now=now, lease_seconds=300)
    if lease is None:
        return RecoveryResult("in_progress")

    from services.mcp.app.mcp import handler

    try:
        folders = folder_repository.get_all_folders(client)
        root_folder_id = get_root_folder_id()
        progress = handler._advance_add_dialog_save(
            client, operation.operation_id, owner, lease, now, root_folder_id, folders,
        )
        handler._publish_add_dialog_index_work(client)
        dialog_completion.finalize_save(
            client, operation_id=operation.operation_id,
            save_intent_id=progress.save_intent_id, owner=owner, fence=lease.fence,
            origin_binding=binding, now=datetime.now(timezone.utc),
        )
        return RecoveryResult("processed")
    finally:
        try:
            release_operation(
                client, operation.operation_id, owner=owner, fence=lease.fence,
                now=datetime.now(timezone.utc),
            )
        except Exception:
            pass
