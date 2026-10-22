"""Durable MCP transport sessions and typed request replay mappings."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from google.cloud import firestore

from core.integrations.firestore.client import get_client


MCP_SESSIONS_COLLECTION = "mcp_sessions"
SESSION_LIFETIME = timedelta(hours=24)


class SessionUnavailableError(Exception):
    """Raised when a session is missing, expired, or bound to another caller."""


class RequestArgumentsConflictError(Exception):
    """Raised when a typed request ID is reused with different arguments."""


@dataclass(frozen=True)
class MCPSession:
    caller_binding: str
    protocol_version: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class SessionInitialization:
    # The raw ID is returned only to the boundary that must set the HTTP header.
    session_id: str
    session: MCPSession


@dataclass(frozen=True)
class RequestReservation:
    operation_id: str
    replayed: bool
    accepted_at: datetime


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _session_hash(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _session_from(data: dict[str, Any]) -> MCPSession:
    return MCPSession(
        caller_binding=data["caller_binding"],
        protocol_version=data["protocol_version"],
        created_at=data["created_at"],
        expires_at=data["expires_at"],
    )


def create_session(
    client: Any | None,
    *,
    caller_binding: str,
    protocol_version: str,
    now: datetime,
) -> SessionInitialization:
    """Persist a new initialized session, returning its opaque header ID once."""
    now = _aware_utc(now, "now")
    if not isinstance(caller_binding, str) or not caller_binding:
        raise ValueError("caller_binding must be a non-empty string")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("protocol_version must be a non-empty string")

    session_id = secrets.token_urlsafe(32)
    record = MCPSession(
        caller_binding=caller_binding,
        protocol_version=protocol_version,
        created_at=now,
        expires_at=now + SESSION_LIFETIME,
    )
    _db(client).collection(MCP_SESSIONS_COLLECTION).document(_session_hash(session_id)).set({
        "caller_binding": record.caller_binding,
        "protocol_version": record.protocol_version,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    })
    return SessionInitialization(session_id=session_id, session=record)


def get_session(
    client: Any | None,
    session_id: str,
    *,
    caller_binding: str,
    now: datetime,
) -> MCPSession:
    """Return an active session for its trusted caller or fail uniformly."""
    now = _aware_utc(now, "now")
    if not isinstance(caller_binding, str) or not caller_binding:
        raise ValueError("caller_binding must be a non-empty string")
    snapshot = _db(client).collection(MCP_SESSIONS_COLLECTION).document(_session_hash(session_id)).get()
    if not snapshot.exists:
        raise SessionUnavailableError("MCP session is unavailable")
    record = _session_from(snapshot.to_dict() or {})
    if record.caller_binding != caller_binding or record.expires_at <= now:
        raise SessionUnavailableError("MCP session is unavailable")
    return record


def _canonical_arguments_digest(arguments: dict[str, Any]) -> str:
    """Hash a strict JSON object with deterministic UTF-8 serialization."""
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")

    def validate(value: Any) -> None:
        if value is None or type(value) in (str, bool, int):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("arguments must not contain non-finite numbers")
            return
        if type(value) is list:
            for item in value:
                validate(item)
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                validate(item)
            return
        raise ValueError("arguments must contain only JSON values")

    validate(arguments)
    encoded = json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_id_hash(request_id: str | int) -> str:
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise ValueError("request_id must be a string or integer")
    typed_id = ["string", request_id] if isinstance(request_id, str) else ["integer", request_id]
    encoded = json.dumps(typed_id, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_recovery_ref(session_id: str, request_id: str | int) -> str:
    """Return the durable, opaque Firestore path for one MCP request mapping."""
    return f"{MCP_SESSIONS_COLLECTION}/{_session_hash(session_id)}/requests/{_request_id_hash(request_id)}"


def get_recovery_binding(
    client: Any | None,
    origin_operation_ref: str,
    *,
    operation_id: str,
    immutable_input_digest: str,
) -> dict[str, str] | None:
    """Load certificate binding material only from the mapping for this operation."""
    if not isinstance(origin_operation_ref, str):
        return None
    parts = origin_operation_ref.split("/")
    if (len(parts) != 4 or parts[0] != MCP_SESSIONS_COLLECTION or parts[2] != "requests"
            or not re.fullmatch(r"[a-f0-9]{64}", parts[1])
            or not re.fullmatch(r"[a-f0-9]{64}", parts[3])):
        return None
    request_snapshot = (
        _db(client).collection(parts[0]).document(parts[1])
        .collection(parts[2]).document(parts[3]).get()
    )
    if not request_snapshot.exists:
        return None
    mapping = request_snapshot.to_dict() or {}
    if (mapping.get("operation_id") != operation_id
            or not isinstance(immutable_input_digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", immutable_input_digest)):
        return None
    return {
        "context_id": origin_operation_ref,
        "draft_id": parts[3],
        "manifest_digest": immutable_input_digest,
    }


def resolve_or_reserve_request(
    client: Any | None,
    session_id: str,
    *,
    caller_binding: str,
    request_id: str | int,
    arguments: dict[str, Any],
    now: datetime,
) -> RequestReservation:
    """Resolve an exact replay or atomically reserve a fresh operation ID."""
    now = _aware_utc(now, "now")
    if not isinstance(caller_binding, str) or not caller_binding:
        raise ValueError("caller_binding must be a non-empty string")
    session_key = _session_hash(session_id)
    request_key = _request_id_hash(request_id)
    arguments_digest = _canonical_arguments_digest(arguments)
    db = _db(client)
    session_ref = db.collection(MCP_SESSIONS_COLLECTION).document(session_key)
    request_ref = session_ref.collection("requests").document(request_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> RequestReservation:
        session_snapshot = session_ref.get(transaction=transaction)
        if not session_snapshot.exists:
            raise SessionUnavailableError("MCP session is unavailable")
        session_data = session_snapshot.to_dict() or {}
        expiry = session_data.get("expires_at")
        if session_data.get("caller_binding") != caller_binding or not isinstance(expiry, datetime) or expiry <= now:
            raise SessionUnavailableError("MCP session is unavailable")

        request_snapshot = request_ref.get(transaction=transaction)
        if request_snapshot.exists:
            request_data = request_snapshot.to_dict() or {}
            if request_data.get("arguments_digest") != arguments_digest:
                raise RequestArgumentsConflictError("request ID was already used with different arguments")
            return RequestReservation(
                operation_id=request_data["operation_id"], replayed=True,
                accepted_at=request_data["created_at"],
            )

        operation_id = uuid4().hex
        transaction.set(request_ref, {
            "operation_id": operation_id,
            "result_ref": f"dialog_operations/{operation_id}",
            "arguments_digest": arguments_digest,
            "created_at": now,
        })
        return RequestReservation(operation_id=operation_id, replayed=False, accepted_at=now)

    return _txn(db.transaction())


def lookup_request_replay(
    client: Any | None,
    session_id: str,
    *,
    caller_binding: str,
    request_id: str | int,
    arguments: dict[str, Any],
    now: datetime,
) -> RequestReservation | None:
    """Read an existing typed request mapping without reserving a fresh key."""
    now = _aware_utc(now, "now")
    if not isinstance(caller_binding, str) or not caller_binding:
        raise ValueError("caller_binding must be a non-empty string")
    session_key = _session_hash(session_id)
    request_key = _request_id_hash(request_id)
    arguments_digest = _canonical_arguments_digest(arguments)
    db = _db(client)
    session_ref = db.collection(MCP_SESSIONS_COLLECTION).document(session_key)
    request_ref = session_ref.collection("requests").document(request_key)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> RequestReservation | None:
        session_snapshot = session_ref.get(transaction=transaction)
        if not session_snapshot.exists:
            raise SessionUnavailableError("MCP session is unavailable")
        session_data = session_snapshot.to_dict() or {}
        expiry = session_data.get("expires_at")
        if session_data.get("caller_binding") != caller_binding or not isinstance(expiry, datetime) or expiry <= now:
            raise SessionUnavailableError("MCP session is unavailable")
        request_snapshot = request_ref.get(transaction=transaction)
        if not request_snapshot.exists:
            return None
        request_data = request_snapshot.to_dict() or {}
        if request_data.get("arguments_digest") != arguments_digest:
            raise RequestArgumentsConflictError("request ID was already used with different arguments")
        return RequestReservation(
            operation_id=request_data["operation_id"], replayed=True,
            accepted_at=request_data["created_at"],
        )

    return _txn(db.transaction())
