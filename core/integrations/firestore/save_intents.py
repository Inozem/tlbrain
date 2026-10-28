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
from core.integrations.firestore.mutations import _resource_ref


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
    desired_parent_id_hash: str | None
    desired_parent_id: str | None
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
        desired_parent_id=data.get("desired_parent_id"),
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
        "desired_parent_id", "immutable_title_digest", "immutable_provider_digest",
    ))


def create_save_intent(
    client: Any | None,
    save_intent_id: str,
    *,
    operation_id: str,
    desired_content_hash: str,
    desired_parent_id_hash: str | None,
    now: datetime,
    owner: str | None = None,
    fence: int | None = None,
) -> SaveIntentRecord | None:
    """Create or return an intent bound to the operation's immutable identity."""
    now = _aware(now)
    if not save_intent_id or "/" in save_intent_id:
        raise ValueError("save_intent_id must be a non-empty Firestore document ID")
    if not desired_content_hash:
        raise ValueError("content hash is required")
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
        if (desired_parent_id_hash is not None and operation.get("desired_parent_id_hash") is not None
                and operation["desired_parent_id_hash"] != desired_parent_id_hash):
            raise ValueError("operation parent hash conflicts with save intent identity")
        identity_data = {
            "immutable_input_digest": operation["immutable_input_digest"],
            "desired_content_hash": desired_content_hash,
            "desired_parent_id_hash": desired_parent_id_hash,
            "desired_parent_id": None,
            "immutable_title_digest": _digest(operation.get("title")),
            "immutable_provider_digest": _digest(operation.get("expected_provider") or operation.get("provider")),
        }
        snapshot = intent_ref.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            stored_identity = list(_identity(data))
            requested_identity = list(_identity(identity_data))
            # An unresolved replay may observe the one-time parent binding made later.
            if requested_identity[2] is None:
                requested_identity[2] = stored_identity[2]
                requested_identity[3] = stored_identity[3]
            if tuple(stored_identity) != tuple(requested_identity):
                raise ValueError("save intent ID already belongs to a different immutable identity")
            operation_updates = {}
            if bound is None:
                operation_updates["save_intent_id"] = save_intent_id
            if operation.get("desired_content_hash") is None:
                operation_updates["desired_content_hash"] = desired_content_hash
            if desired_parent_id_hash is not None and operation.get("desired_parent_id_hash") is None:
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
        updates = {"save_intent_id": save_intent_id, "desired_content_hash": desired_content_hash,
                   "updated_at": now}
        if desired_parent_id_hash is not None:
            updates["desired_parent_id_hash"] = desired_parent_id_hash
        transaction.update(operation_ref, updates)
        return _from(data)

    return _txn(db.transaction())


def bind_save_intent_parent(
    client: Any | None,
    save_intent_id: str,
    *,
    operation_id: str,
    desired_parent_id_hash: str,
    desired_parent_id: str,
    owner: str,
    fence: int,
    now: datetime,
) -> SaveIntentRecord | None:
    """Bind the resolved parent exactly once while the operation lease is current."""
    now = _aware(now)
    if not desired_parent_id_hash or not desired_parent_id:
        raise ValueError("parent ID and hash are required")
    if _digest(desired_parent_id) != desired_parent_id_hash:
        raise ValueError("parent ID does not match its hash")
    db = _db(client)
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> SaveIntentRecord | None:
        operation_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not intent_snapshot.exists:
            return None
        operation = operation_snapshot.to_dict() or {}
        data = intent_snapshot.to_dict() or {}
        if (operation.get("save_intent_id") != save_intent_id
                or not _valid_lease(operation, owner, fence, now)):
            return None
        current = data.get("desired_parent_id_hash")
        op_parent = operation.get("desired_parent_id_hash")
        current_parent_id = data.get("desired_parent_id")
        op_parent_id = operation.get("selected_folder_id")
        if current is not None and current != desired_parent_id_hash:
            raise ValueError("save intent parent is already bound to a different folder")
        if current_parent_id is not None and current_parent_id != desired_parent_id:
            raise ValueError("save intent parent is already bound to a different folder")
        if op_parent is not None and op_parent != desired_parent_id_hash:
            raise ValueError("operation parent conflicts with resolved folder")
        if op_parent_id is not None and op_parent_id != desired_parent_id:
            raise ValueError("operation selected folder conflicts with resolved parent")
        data["desired_parent_id_hash"] = desired_parent_id_hash
        data["desired_parent_id"] = desired_parent_id
        data["updated_at"] = now
        transaction.set(intent_ref, data)
        transaction.update(operation_ref, {"desired_parent_id_hash": desired_parent_id_hash, "updated_at": now})
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


def _completion_document_refs(
    db: Any,
    intent_ref: Any,
    verified_content_hash: str,
) -> tuple[Any, Any, Any, Any] | None:
    """Resolve the intent's document and prepare the related completion references."""
    intent_snapshot = intent_ref.get()
    if not intent_snapshot.exists:
        return None
    doc_id = (intent_snapshot.to_dict() or {}).get("doc_id")
    if not doc_id:
        return None
    transcript_ref = db.collection("transcript_index").document(doc_id)
    receipt_ref = (db.collection("index_completions").document(doc_id)
                   .collection("generations").document(verified_content_hash))
    guard_ref = _resource_ref(db, f"document:{doc_id}")
    return doc_id, transcript_ref, receipt_ref, guard_ref


def _completion_replay_result(
    operation: dict[str, Any],
    intent: dict[str, Any],
    *,
    operation_id: str,
    save_intent_id: str,
    doc_id: str,
    origin_binding: dict[str, str],
) -> tuple[bool, OperationResult | None]:
    """Recognize a lost-response replay and return its exact durable result."""
    if operation.get("state") != "completed":
        return False, None
    certificate = intent.get("completion_certificate") or {}
    stored_result = _result_from(operation.get("result"))
    if (operation.get("save_intent_id") == save_intent_id
            and intent.get("save_intent_id") == save_intent_id
            and _completion_operation_binding_matches(
                operation, intent, operation_id=operation_id, origin_binding=origin_binding,
            )
            and intent.get("doc_id") == doc_id
            and certificate.get("operation_id") == operation_id
            and certificate.get("save_intent_id") == save_intent_id
            and certificate.get("doc_id") == doc_id
            and certificate.get("origin_binding") == origin_binding
            and stored_result is not None and stored_result.document_id == doc_id):
        return True, stored_result
    return True, None


def _completion_operation_binding_matches(
    operation: dict[str, Any], intent: dict[str, Any], *,
    operation_id: str, origin_binding: dict[str, str],
) -> bool:
    """Validate the durable create gate and its explicitly linked retry lineage."""
    if intent.get("dispatch_operation_id") == operation_id:
        stored_binding = operation.get("origin_binding")
        if stored_binding is not None:
            return stored_binding == origin_binding
        return (
            origin_binding.get("context_id") == operation.get("origin_operation_ref")
            and origin_binding.get("manifest_digest") == operation.get("immutable_input_digest")
        )
    authorization = operation.get("retry_authorization") or {}
    return (
        authorization.get("operation_id") == operation_id
        and authorization.get("create_gate_operation_id", authorization.get("previous_operation_id"))
            == intent.get("dispatch_operation_id")
        and authorization.get("context_id") == origin_binding.get("context_id")
        and authorization.get("draft_id") == origin_binding.get("draft_id")
        and authorization.get("manifest_digest") == origin_binding.get("manifest_digest")
        and authorization.get("caller_binding") == operation.get("caller_binding")
        and authorization.get("immutable_input_digest") == operation.get("immutable_input_digest")
        and operation.get("retry_of_operation_id") == authorization.get("previous_operation_id")
        and operation.get("retry_draft_id") == origin_binding.get("draft_id")
        and operation.get("origin_binding") == origin_binding
    )


def _completion_inputs_valid(
    operation: dict[str, Any],
    intent: dict[str, Any],
    transcript: dict[str, Any],
    receipt: dict[str, Any],
    guard: dict[str, Any],
    *,
    operation_id: str,
    save_intent_id: str,
    owner: str,
    fence: int,
    doc_id: str,
    origin_binding: dict[str, str],
    verified_content_hash: str,
    verified_parent_id: str,
    verified_provider: str,
    resource_epoch: int,
    guard_reservation_id: str,
    commit_now: datetime,
) -> bool:
    """Validate operation bindings, indexing evidence, and the active guard reservation."""
    if (not _valid_lease(operation, owner, fence, commit_now)
            or operation.get("save_intent_id") != save_intent_id
            or intent.get("save_intent_id") != save_intent_id
            or not _completion_operation_binding_matches(
                operation, intent, operation_id=operation_id, origin_binding=origin_binding,
            )
            or intent.get("state") != SaveIntentState.CONFIRMED.value
            or intent.get("doc_id") != doc_id
            or not intent.get("confirmed_result")
            or not origin_binding
            or (operation.get("origin_binding") is None and (
                origin_binding.get("context_id") != operation.get("origin_operation_ref")
                or origin_binding.get("manifest_digest") != operation.get("immutable_input_digest")
            ))
            or not isinstance(origin_binding.get("draft_id"), str)
            or not origin_binding.get("draft_id")):
        return False
    bound_parent_id = (intent.get("desired_parent_id")
                       or operation.get("selected_folder_id")
                       or operation.get("explicit_folder_id"))
    expected_provider = operation.get("expected_provider") or operation.get("provider")
    if (operation.get("doc_id") not in (None, doc_id)
            or operation.get("desired_content_hash") != verified_content_hash
            or intent.get("desired_content_hash") != verified_content_hash
            or bound_parent_id != verified_parent_id
            or intent.get("desired_parent_id_hash") != _digest(verified_parent_id)
            or operation.get("desired_parent_id_hash") != _digest(verified_parent_id)
            or (expected_provider is not None and verified_provider != expected_provider)
            or (intent.get("immutable_provider_digest") is not None
                and intent.get("immutable_provider_digest") != _digest(verified_provider))):
        return False
    if (receipt.get("doc_id") != doc_id
            or receipt.get("content_hash") != verified_content_hash
            or receipt.get("version") != verified_content_hash
            or receipt.get("status") != "completed"
            or receipt.get("provider") != verified_provider
            or receipt.get("parent_id") != verified_parent_id
            or receipt.get("parent_id_hash") != _digest(verified_parent_id)
            or transcript.get("status") != "synced"):
        return False
    generation_fields = (
        "doc_id", "root_folder_id", "content_hash", "version", "parent_id",
        "parent_id_hash", "provider", "fence", "index_fence", "resource_epoch",
    )
    if any(transcript.get(field) != receipt.get(field) for field in generation_fields):
        return False
    if (guard.get("finalization_operation_id") != operation_id
            or guard.get("finalization_reservation_id") != guard_reservation_id
            or guard.get("finalization_owner") != owner
            or guard.get("finalization_fence") != fence
            or guard.get("resource_epoch", 0) != resource_epoch
            or guard.get("finalization_expires_at") is None
            or guard.get("finalization_expires_at") <= commit_now):
        return False
    return (receipt.get("resource_epoch") == resource_epoch
            and receipt.get("index_fence") == transcript.get("index_fence"))


def _create_completion_artifacts(
    intent: dict[str, Any],
    receipt: dict[str, Any],
    *,
    operation_id: str,
    save_intent_id: str,
    doc_id: str,
    origin_binding: dict[str, str],
    verified_content_hash: str,
    verified_parent_id: str,
    verified_provider: str,
    resource_epoch: int,
    fence: int,
    certificate_id: str,
    verified_at: datetime,
    commit_now: datetime,
) -> tuple[OperationResult, dict[str, Any], dict[str, Any]] | None:
    """Build the confirmed result, completion certificate, and cleanup authorization."""
    confirmed_result = _result_from(intent.get("confirmed_result"))
    skipped_utterances = receipt.get("skipped_utterances")
    if (confirmed_result is None or confirmed_result.document_id != doc_id
            or not isinstance(skipped_utterances, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in skipped_utterances)):
        return None
    result = OperationResult(
        document_id=confirmed_result.document_id,
        title=confirmed_result.title,
        resource_url=confirmed_result.resource_url,
        skipped_utterances=tuple(skipped_utterances),
    )
    certificate = {
        "certificate_id": certificate_id,
        "save_intent_id": save_intent_id,
        "operation_id": operation_id,
        "doc_id": doc_id,
        "origin_binding": dict(origin_binding),
        "content_hash": verified_content_hash,
        "parent_id_hash": _digest(verified_parent_id),
        "provider": verified_provider,
        "receipt_ref": f"index_completions/{doc_id}/generations/{verified_content_hash}",
        "receipt_version": receipt.get("version"),
        "index_fence": receipt.get("index_fence"),
        "resource_epoch": resource_epoch,
        "fence": fence,
        "verified_at": verified_at,
        "completed_at": commit_now,
    }
    cleanup_authorization = {
        "certificate_id": certificate_id,
        "save_intent_id": save_intent_id,
        "operation_id": operation_id,
        "origin_binding": dict(origin_binding),
    }
    return result, certificate, cleanup_authorization


def _write_completed_save(
    transaction: firestore.Transaction,
    intent_ref: Any,
    operation_ref: Any,
    guard_ref: Any,
    intent: dict[str, Any],
    operation: dict[str, Any],
    guard: dict[str, Any],
    *,
    result: OperationResult,
    certificate: dict[str, Any],
    cleanup_authorization: dict[str, Any],
    verified_parent_id: str,
    commit_now: datetime,
) -> OperationResult:
    """Persist completion evidence and release the matching guard atomically."""
    intent.update({"desired_parent_id": verified_parent_id,
                   "completion_certificate": certificate,
                   "cleanup_authorization": cleanup_authorization,
                   "updated_at": commit_now})
    operation.update({
        "state": "completed", "doc_id": result.document_id, "result": _result_data(result),
        "document_finalized": True, "cleanup_authorization": cleanup_authorization,
        "input_cleanup_state": "pending",
        "finished_at": commit_now, "status_updated_at": commit_now, "updated_at": commit_now,
    })
    for field in ("finalization_reservation_id", "finalization_operation_id",
                  "finalization_owner", "finalization_fence", "finalization_expires_at"):
        guard.pop(field, None)
    guard.update({"mutation_state": "idle", "updated_at": commit_now})
    transaction.set(intent_ref, intent)
    transaction.set(operation_ref, operation)
    transaction.set(guard_ref, guard)
    return result


def finalize_save_completion(
    client: Any | None,
    *,
    operation_id: str,
    save_intent_id: str,
    owner: str,
    fence: int,
    origin_binding: dict[str, str],
    now: datetime,
    verified_content_hash: str,
    verified_parent_id: str,
    verified_provider: str,
    resource_epoch: int,
    guard_reservation_id: str,
    certificate_id: str,
    verified_at: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OperationResult | None:
    """Atomically complete the operation and issue immutable cleanup evidence."""
    now = _aware(now)
    verified_at = _aware(verified_at or now)
    clock = clock or (lambda: datetime.now(timezone.utc))
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    intent_ref = db.collection(SAVE_INTENTS_COLLECTION).document(save_intent_id)
    references = _completion_document_refs(db, intent_ref, verified_content_hash)
    if references is None:
        return None
    doc_id, transcript_ref, receipt_ref, guard_ref = references

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationResult | None:
        commit_now = _aware(clock())
        op_snapshot = operation_ref.get(transaction=transaction)
        intent_snapshot = intent_ref.get(transaction=transaction)
        transcript_snapshot = transcript_ref.get(transaction=transaction)
        receipt_snapshot = receipt_ref.get(transaction=transaction)
        guard_snapshot = guard_ref.get(transaction=transaction)
        if not all((op_snapshot.exists, intent_snapshot.exists, transcript_snapshot.exists,
                    receipt_snapshot.exists, guard_snapshot.exists)):
            return None
        operation = op_snapshot.to_dict() or {}
        intent = intent_snapshot.to_dict() or {}
        transcript = transcript_snapshot.to_dict() or {}
        receipt = receipt_snapshot.to_dict() or {}
        guard = guard_snapshot.to_dict() or {}

        is_replay, replay_result = _completion_replay_result(
            operation, intent, operation_id=operation_id, save_intent_id=save_intent_id,
            doc_id=doc_id, origin_binding=origin_binding,
        )
        if is_replay:
            return replay_result

        if not _completion_inputs_valid(
            operation, intent, transcript, receipt, guard,
            operation_id=operation_id, save_intent_id=save_intent_id, owner=owner,
            fence=fence, doc_id=doc_id, origin_binding=origin_binding,
            verified_content_hash=verified_content_hash, verified_parent_id=verified_parent_id,
            verified_provider=verified_provider, resource_epoch=resource_epoch,
            guard_reservation_id=guard_reservation_id, commit_now=commit_now,
        ):
            return None

        artifacts = _create_completion_artifacts(
            intent, receipt, operation_id=operation_id, save_intent_id=save_intent_id,
            doc_id=doc_id, origin_binding=origin_binding,
            verified_content_hash=verified_content_hash, verified_parent_id=verified_parent_id,
            verified_provider=verified_provider, resource_epoch=resource_epoch, fence=fence,
            certificate_id=certificate_id, verified_at=verified_at, commit_now=commit_now,
        )
        if artifacts is None:
            return None
        result, certificate, cleanup_authorization = artifacts
        return _write_completed_save(
            transaction, intent_ref, operation_ref, guard_ref, intent, operation, guard,
            result=result, certificate=certificate, cleanup_authorization=cleanup_authorization,
            verified_parent_id=verified_parent_id, commit_now=commit_now,
        )

    return _txn(db.transaction())
