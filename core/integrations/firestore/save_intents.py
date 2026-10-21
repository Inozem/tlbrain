"""Durable, provider-neutral save intent records and create dispatch gate."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.operations import _result_data, _result_from, _valid_lease
from core.operations.models import OperationResult


SAVE_INTENTS_COLLECTION = "dialog_save_intents"
OPERATIONS_COLLECTION = "dialog_operations"


class SaveIntentState(StrEnum):
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SaveIntentRecord:
    save_intent_id: str
    immutable_input_digest: str
    desired_content_hash: str
    desired_parent_id_hash: str
    immutable_title_digest: str | None
    immutable_provider_digest: str | None
    state: SaveIntentState = SaveIntentState.PLANNED
    create_outcome: str | None = None
    dispatch_token: str | None = None
    dispatch_operation_id: str | None = None
    dispatch_owner: str | None = None
    dispatch_fence: int | None = None
    doc_id: str | None = None
    confirmed_result: OperationResult | None = None
    fill_checkpoint_ref: str | None = None
    index_checkpoint_ref: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _digest(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def _from(data: dict[str, Any]) -> SaveIntentRecord:
    return SaveIntentRecord(
        save_intent_id=data["save_intent_id"],
        immutable_input_digest=data["immutable_input_digest"],
        desired_content_hash=data["desired_content_hash"],
        desired_parent_id_hash=data["desired_parent_id_hash"],
        immutable_title_digest=data.get("immutable_title_digest"),
        immutable_provider_digest=data.get("immutable_provider_digest"),
        state=SaveIntentState(data.get("state", SaveIntentState.PLANNED.value)),
        create_outcome=data.get("create_outcome"), dispatch_token=data.get("dispatch_token"),
        dispatch_operation_id=data.get("dispatch_operation_id"), dispatch_owner=data.get("dispatch_owner"),
        dispatch_fence=data.get("dispatch_fence"), doc_id=data.get("doc_id"),
        confirmed_result=_result_from(data.get("confirmed_result")),
        fill_checkpoint_ref=data.get("fill_checkpoint_ref"), index_checkpoint_ref=data.get("index_checkpoint_ref"),
        created_at=data.get("created_at"), updated_at=data.get("updated_at"),
    )


def _identity(data: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(data.get(key) for key in (
        "immutable_input_digest", "desired_content_hash", "desired_parent_id_hash",
        "immutable_title_digest", "immutable_provider_digest",
    ))


def create_save_intent(
    client: Any | None,
    save_intent_id: str,
    *,
    operation_id: str,
    desired_content_hash: str,
    desired_parent_id_hash: str,
    now: datetime,
    owner: str | None = None,
    fence: int | None = None,
) -> SaveIntentRecord | None:
    """Create or return an intent bound to the operation's immutable identity."""
    now = _aware(now)
    if not save_intent_id or "/" in save_intent_id:
        raise ValueError("save_intent_id must be a non-empty Firestore document ID")
    if not desired_content_hash or not desired_parent_id_hash:
        raise ValueError("content and parent hashes are required")
    if (owner is None) != (fence is None):
        raise ValueError("owner and fence must be provided together")
    db = _db(client)
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> SaveIntentRecord | None:
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not operation_snapshot.exists:
            raise ValueError("operation does not exist")
        operation = operation_snapshot.to_dict() or {}
        if owner is not None and not _valid_lease(operation, owner, fence, now):
            return None
        bound = operation.get("save_intent_id")
        if bound is not None and bound != save_intent_id:
            raise ValueError("operation is already bound to a different save intent")
        if (operation.get("desired_content_hash") is not None
                and operation["desired_content_hash"] != desired_content_hash):
            raise ValueError("operation content hash conflicts with save intent identity")
        if (operation.get("desired_parent_id_hash") is not None
                and operation["desired_parent_id_hash"] != desired_parent_id_hash):
            raise ValueError("operation parent hash conflicts with save intent identity")
        identity_data = {
            "immutable_input_digest": operation["immutable_input_digest"],
            "desired_content_hash": desired_content_hash,
            "desired_parent_id_hash": desired_parent_id_hash,
            "immutable_title_digest": _digest(operation.get("title")),
            "immutable_provider_digest": _digest(operation.get("expected_provider") or operation.get("provider")),
        }
        snapshot = intent_ref.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            if _identity(data) != _identity(identity_data):
                raise ValueError("save intent ID already belongs to a different immutable identity")
            operation_updates = {}
            if bound is None:
                operation_updates["save_intent_id"] = save_intent_id
            if operation.get("desired_content_hash") is None:
                operation_updates["desired_content_hash"] = desired_content_hash
            if operation.get("desired_parent_id_hash") is None:
                operation_updates["desired_parent_id_hash"] = desired_parent_id_hash
            if operation_updates:
                operation_updates["updated_at"] = now
                transaction.update(operation_ref, operation_updates)
            return _from(data)
        data = {
            "save_intent_id": save_intent_id, **identity_data, "state": SaveIntentState.PLANNED.value,
            "create_outcome": None, "dispatch_token": None, "dispatch_operation_id": None,
            "dispatch_owner": None, "dispatch_fence": None, "doc_id": None,
            "confirmed_result": None, "fill_checkpoint_ref": None, "index_checkpoint_ref": None,
            "created_at": now, "updated_at": now,
        }
        transaction.set(intent_ref, data)
        transaction.update(operation_ref, {"save_intent_id": save_intent_id, "desired_content_hash": desired_content_hash,
                                            "desired_parent_id_hash": desired_parent_id_hash, "updated_at": now})
        return _from(data)

    return _txn(db.transaction())


def get_save_intent(client: Any | None, save_intent_id: str) -> SaveIntentRecord | None:
    snapshot = _db(client).collection(SAVE_INTENTS_COLLECTION).document(save_intent_id).get()
    return _from(snapshot.to_dict() or {}) if snapshot.exists else None


def dispatch_save_intent(
    client: Any | None,
    save_intent_id: str,
    *,
    operation_id: str,
    owner: str,
    fence: int,
    now: datetime,
    create_callback: Callable[[], OperationResult],
) -> OperationResult | None:
    """Reserve the single create dispatch before calling provider-owned code."""
    now = _aware(now)
    db = _db(client)
    if get_save_intent(db, save_intent_id) is None:
        operation = db.collection(OPERATIONS_COLLECTION).document(operation_id).get()
        if not operation.exists:
            return None
        operation_data = operation.to_dict() or {}
        content_hash = operation_data.get("desired_content_hash") or operation_data.get("immutable_input_digest")
        parent_hash = operation_data.get("desired_parent_id_hash") or _digest(operation_data.get("explicit_folder_id") or "")
        intent = create_save_intent(
            db, save_intent_id, operation_id=operation_id,
            desired_content_hash=content_hash, desired_parent_id_hash=parent_hash, now=now,
            owner=owner, fence=fence,
        )
        if intent is None:
            return None
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _reserve(transaction: firestore.Transaction) -> tuple[str, SaveIntentRecord | None]:
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            raise ValueError("operation and save intent must exist before dispatch")
        op = op_snapshot.to_dict() or {}
        data = intent_snapshot.to_dict() or {}
        if op.get("save_intent_id") != save_intent_id or not _valid_lease(op, owner, fence, now):
            return "stale", None
        state = SaveIntentState(data.get("state", "planned"))
        if state == SaveIntentState.CONFIRMED:
            return "confirmed", _from(data)
        if state != SaveIntentState.PLANNED:
            return "closed", None
        token = uuid.uuid4().hex
        updated = dict(data)
        updated.update({"state": SaveIntentState.DISPATCHED.value, "create_outcome": "dispatched",
                        "dispatch_token": token, "dispatch_operation_id": operation_id,
                        "dispatch_owner": owner, "dispatch_fence": fence, "updated_at": now})
        transaction.set(intent_ref, updated)
        return token, _from(updated)

    reservation, record = _reserve(db.transaction())
    if reservation == "confirmed":
        return record.confirmed_result if record else None
    if reservation in {"stale", "closed"} or record is None:
        return None
    try:
        result = create_callback()
    except BaseException as exc:
        if getattr(exc, "outcome", None) == "rejected":
            _mark_rejected(db, save_intent_id, operation_id, owner, fence, now, reservation)
        else:
            _mark_unknown(db, save_intent_id, operation_id, owner, fence, now, reservation)
        raise
    _confirm(db, save_intent_id, operation_id, owner, fence, now, reservation, result)
    current = get_save_intent(db, save_intent_id)
    return current.confirmed_result if current is not None else None


def _mark_unknown(db: Any, intent_id: str, operation_id: str, owner: str, fence: int, now: datetime, token: str) -> None:
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> None:
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            return
        op, data = op_snapshot.to_dict() or {}, intent_snapshot.to_dict() or {}
        if not _valid_lease(op, owner, fence, now) or data.get("dispatch_token") != token:
            return
        data.update({"state": SaveIntentState.UNKNOWN.value, "create_outcome": "unknown", "updated_at": now})
        transaction.set(intent_ref, data)

    _txn(db.transaction())


def _mark_rejected(db: Any, intent_id: str, operation_id: str, owner: str, fence: int, now: datetime,
                   token: str) -> None:
    """Reopen only a definitively rejected dispatch under its original fence."""
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> None:
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            return
        op, data = op_snapshot.to_dict() or {}, intent_snapshot.to_dict() or {}
        if (op.get("save_intent_id") != intent_id
                or not _valid_lease(op, owner, fence, now)
                or data.get("dispatch_token") != token
                or data.get("state") != SaveIntentState.DISPATCHED.value):
            return
        data.update({
            "state": SaveIntentState.PLANNED.value,
            "create_outcome": "rejected",
            "dispatch_token": None,
            "dispatch_operation_id": None,
            "dispatch_owner": None,
            "dispatch_fence": None,
            "updated_at": now,
        })
        transaction.set(intent_ref, data)

    _txn(db.transaction())


def _confirm(db: Any, intent_id: str, operation_id: str, owner: str, fence: int, now: datetime,
             token: str, result: OperationResult) -> bool:
    if not result.document_id:
        raise ValueError("confirmed save intent requires a document ID")
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            return False
        op, data = op_snapshot.to_dict() or {}, intent_snapshot.to_dict() or {}
        if not _valid_lease(op, owner, fence, now) or data.get("dispatch_token") != token:
            return False
        if data.get("state") == SaveIntentState.CONFIRMED.value:
            if _result_from(data.get("confirmed_result")) != result:
                raise ValueError("contradictory save intent reconciliation")
            return True
        if data.get("state") not in {SaveIntentState.DISPATCHED.value, SaveIntentState.UNKNOWN.value}:
            return False
        data.update({"state": SaveIntentState.CONFIRMED.value, "create_outcome": "confirmed",
                     "doc_id": result.document_id, "confirmed_result": _result_data(result), "updated_at": now})
        transaction.set(intent_ref, data)
        return True

    return _txn(db.transaction())


def reconcile_save_intent(
    client: Any | None,
    save_intent_id: str,
    *,
    owner: str,
    fence: int,
    now: datetime,
    external_result: OperationResult,
    operation_id: str | None = None,
) -> SaveIntentRecord | None:
    """Record a discovered create result under the current operation's fence."""
    now = _aware(now)
    if not external_result.document_id:
        raise ValueError("confirmed save intent requires a document ID")
    db = _db(client)
    current = get_save_intent(db, save_intent_id)
    if current is None or current.dispatch_operation_id is None or current.dispatch_token is None:
        return None
    lease_operation_id = operation_id or current.dispatch_operation_id
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(lease_operation_id)
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> SaveIntentRecord | None:
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            return None
        op, data = op_snapshot.to_dict() or {}, intent_snapshot.to_dict() or {}
        if op.get("save_intent_id") != save_intent_id or not _valid_lease(op, owner, fence, now):
            return None
        state = SaveIntentState(data.get("state", "planned"))
        if state == SaveIntentState.CONFIRMED:
            if _result_from(data.get("confirmed_result")) != external_result:
                raise ValueError("contradictory save intent reconciliation")
            return _from(data)
        if state not in {SaveIntentState.UNKNOWN, SaveIntentState.DISPATCHED}:
            return None
        data.update({"state": SaveIntentState.CONFIRMED.value, "create_outcome": "confirmed",
                     "doc_id": external_result.document_id, "confirmed_result": _result_data(external_result),
                     "updated_at": now})
        transaction.set(intent_ref, data)
        return _from(data)

    return _txn(db.transaction())


def update_save_intent_checkpoints(
    client: Any | None,
    save_intent_id: str,
    *,
    operation_id: str,
    owner: str,
    fence: int,
    now: datetime,
    fill_checkpoint_ref: str | None = None,
    index_checkpoint_ref: str | None = None,
) -> SaveIntentRecord | None:
    """Persist durable fill/index checkpoint references under a current operation lease."""
    now = _aware(now)
    db = _db(client)
    op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> SaveIntentRecord | None:
        op_snapshot = op_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not op_snapshot.exists or not intent_snapshot.exists:
            return None
        op, data = op_snapshot.to_dict() or {}, intent_snapshot.to_dict() or {}
        if op.get("save_intent_id") != save_intent_id or not _valid_lease(op, owner, fence, now):
            return None
        if (SaveIntentState(data.get("state", "planned")) != SaveIntentState.CONFIRMED
                or not data.get("doc_id")):
            return None
        if fill_checkpoint_ref is not None:
            data["fill_checkpoint_ref"] = fill_checkpoint_ref
        if index_checkpoint_ref is not None:
            data["index_checkpoint_ref"] = index_checkpoint_ref
        data["updated_at"] = now
        transaction.set(intent_ref, data)
        return _from(data)

    return _txn(db.transaction())
