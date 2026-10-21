"""Durable Firestore guards for external mutations and publication steps."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from google.cloud import firestore

from core.integrations.firestore.client import get_client
from core.integrations.firestore.operations import (
    OPERATIONS_COLLECTION,
    _result_data,
    _result_from,
    _valid_lease,
)
from core.operations.models import EffectState, OperationEffect, OperationResult


RESOURCE_GUARDS_COLLECTION = "resource_mutations"


class MutationIdentityConflict(ValueError):
    """An effect ID was already reserved for a different immutable identity."""


@dataclass(frozen=True, slots=True)
class ResourceGuard:
    resource: str
    owner_operation_id: str | None = None
    owner_effect_id: str | None = None
    fence: int = 0
    expires_at: datetime | None = None
    resource_epoch: int = 0
    mutation_state: str = "idle"
    unsettled_effect_ids: tuple[str, ...] = ()


def _db(client: Any | None) -> Any:
    return client if client is not None else get_client()


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _doc_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _effect_ref(db: Any, operation_id: str, effect_id: str) -> Any:
    return (
        db.collection(OPERATIONS_COLLECTION).document(operation_id)
        .collection("effects").document(effect_id)
    )


def _resource_ref(db: Any, resource: str) -> Any:
    return db.collection(RESOURCE_GUARDS_COLLECTION).document(_doc_id(resource))


def _effect_from(data: dict[str, Any]) -> OperationEffect:
    result = _result_from(data.get("confirmed_result"))
    return OperationEffect(
        effect_id=data["effect_id"], operation_id=data["operation_id"], kind=data["kind"],
        target_resource=data["target_resource"], desired_payload_digest=data["desired_payload_digest"],
        expected_pre_state=data.get("expected_pre_state"), expected_revision=data.get("expected_revision"),
        state=EffectState(data["state"]), dispatched_at=data.get("dispatched_at"),
        confirmed_result=result, fence=int(data.get("fence", 0)),
        resources=tuple(data.get("resources", ())),
        desired_state_digest=data.get("desired_state_digest"),
        mutation_policy=data.get("mutation_policy"),
    )


def _effect_data(
    *, effect_id: str, operation_id: str, kind: str, resources: tuple[str, ...],
    payload_digest: str, owner: str, fence: int, state: EffectState, now: datetime,
    result: Any = None, expected_pre_state: str | None = None,
    expected_revision: str | None = None,
    desired_state_digest: str | None = None,
    mutation_policy: str | None = None,
) -> dict[str, Any]:
    return {
        "effect_id": effect_id, "operation_id": operation_id, "kind": kind,
        "target_resource": resources[0] if resources else "", "resources": list(resources),
        "desired_payload_digest": payload_digest, "expected_pre_state": expected_pre_state,
        "expected_revision": expected_revision, "state": state.value, "dispatched_at": now,
        "desired_state_digest": desired_state_digest, "mutation_policy": mutation_policy,
        "confirmed_result": _result_data(result) if isinstance(result, OperationResult) else None,
        "result": result, "owner": owner, "fence": fence, "updated_at": now,
    }


def _identity(data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        data.get("operation_id"), data.get("kind"), tuple(data.get("resources", ())),
        data.get("desired_payload_digest"), data.get("expected_pre_state"), data.get("expected_revision"),
        data.get("desired_state_digest"), data.get("mutation_policy"),
    )


def _guard_from(data: dict[str, Any] | None, resource: str) -> ResourceGuard:
    data = data or {}
    return ResourceGuard(
        resource=resource, owner_operation_id=data.get("owner_operation_id"),
        owner_effect_id=data.get("owner_effect_id"), fence=int(data.get("fence", 0)),
        expires_at=data.get("expires_at"), resource_epoch=int(data.get("resource_epoch", 0)),
        mutation_state=data.get("mutation_state", "idle"),
        unsettled_effect_ids=tuple(data.get("unsettled_effect_ids", ())),
    )


def get_resource_guard(client: Any | None, resource: str) -> ResourceGuard:
    snapshot = _resource_ref(_db(client), resource).get()
    return _guard_from(snapshot.to_dict() if snapshot.exists else None, resource)


def get_mutation_effect(client: Any | None, operation_id: str, effect_id: str) -> OperationEffect | None:
    """Read the durable typed identity and state for a mutation effect."""
    snapshot = _effect_ref(_db(client), operation_id, effect_id).get()
    return _effect_from(snapshot.to_dict() or {}) if snapshot.exists else None


def get_mutation_result(client: Any | None, operation_id: str, effect_id: str) -> tuple[bool, Any]:
    """Read a confirmed result while preserving a legitimate stored ``None``."""
    snapshot = _effect_ref(_db(client), operation_id, effect_id).get()
    if not snapshot.exists:
        return False, None
    data = snapshot.to_dict() or {}
    if data.get("state") != EffectState.CONFIRMED.value or "result" not in data:
        return False, None
    return True, data.get("result")


def _locked_guard(data: dict[str, Any] | None) -> bool:
    return bool((data or {}).get("unsettled_effect_ids"))


def dispatch_mutation(
    client: Any | None,
    *,
    operation_id: str,
    effect_id: str,
    owner: str,
    fence: int,
    kind: str,
    resources: Iterable[str],
    payload_digest: str,
    now: datetime,
    mutation_callback: Callable[[], Any],
    expected_pre_state: str | None = None,
    expected_revision: str | None = None,
    desired_state_digest: str | None = None,
    mutation_policy: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Any:
    """Persist an effect and resource epochs before calling external code."""
    clock = clock or (lambda: now)
    now = _aware(clock())
    resources = tuple(sorted(set(resources)))
    if not effect_id or "/" in effect_id or not resources or any(not r for r in resources):
        raise ValueError("effect_id and at least one non-empty resource are required")
    if kind not in {"content_update", "parent_move", "provider_call"}:
        raise ValueError("unsupported mutation kind")
    db = _db(client)
    op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    effect_ref = _effect_ref(db, operation_id, effect_id)
    guard_refs = [(resource, _resource_ref(db, resource)) for resource in resources]
    proposed = _effect_data(
        effect_id=effect_id, operation_id=operation_id, kind=kind, resources=resources,
        payload_digest=payload_digest, owner=owner, fence=fence,
        state=EffectState.DISPATCHED, now=now,
        expected_pre_state=expected_pre_state, expected_revision=expected_revision,
        desired_state_digest=desired_state_digest, mutation_policy=mutation_policy,
    )

    @firestore.transactional
    def _reserve(transaction: firestore.Transaction) -> tuple[str, Any]:
        commit_now = _aware(clock())
        op_snapshot = op_ref.get(transaction=transaction)
        effect_snapshot = effect_ref.get(transaction=transaction)
        guard_snapshots = [(resource, ref, ref.get(transaction=transaction)) for resource, ref in guard_refs]
        if not op_snapshot.exists or not _valid_lease(op_snapshot.to_dict() or {}, owner, fence, commit_now):
            return "stale", None
        if effect_snapshot.exists:
            existing = effect_snapshot.to_dict() or {}
            if _identity(existing) != _identity(proposed):
                raise MutationIdentityConflict("effect ID already belongs to a different immutable identity")
            state = EffectState(existing["state"])
            if state == EffectState.CONFIRMED:
                return "confirmed", existing.get("result")
            if state in {EffectState.DISPATCHED, EffectState.UNKNOWN}:
                return "closed", None
            if state == EffectState.FAILED:
                return "closed", None
        guards = [(resource, ref, snap.to_dict() if snap.exists else {}) for resource, ref, snap in guard_snapshots]
        if any(_locked_guard(data) for _, _, data in guards):
            return "blocked", None
        for resource, ref, data in guards:
            unsettled = tuple(data.get("unsettled_effect_ids", ()))
            updated = dict(data)
            updated.update({
                "resource": resource, "owner_operation_id": operation_id,
                "owner_effect_id": effect_id, "owner": owner, "fence": fence,
                "expires_at": op_snapshot.to_dict().get("lease_expires_at"),
                "resource_epoch": int(data.get("resource_epoch", 0)) + 1,
                "mutation_state": EffectState.DISPATCHED.value,
                "unsettled_effect_ids": list((*unsettled, effect_id)), "updated_at": commit_now,
            })
            transaction.set(ref, updated)
        proposed["dispatched_at"] = commit_now
        proposed["updated_at"] = commit_now
        transaction.set(effect_ref, proposed)
        return "dispatch", None

    reservation, prior_result = _reserve(db.transaction())
    if reservation == "confirmed":
        return prior_result
    if reservation != "dispatch":
        return None
    try:
        result = mutation_callback()
    except BaseException:
        _mark_unknown(db, operation_id, effect_id, _aware(clock()), clock=clock)
        raise
    if result is None:
        _mark_unknown(db, operation_id, effect_id, _aware(clock()), clock=clock)
        return None
    if _confirm(db, operation_id, effect_id, owner, fence, _aware(clock()), result, clock=clock):
        return result
    _mark_unknown(db, operation_id, effect_id, _aware(clock()), clock=clock)
    return None


def _mark_unknown(db: Any, operation_id: str, effect_id: str, now: datetime,
                  *, clock: Callable[[], datetime] | None = None) -> None:
    effect_ref = _effect_ref(db, operation_id, effect_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> None:
        commit_now = _aware(clock()) if clock is not None else now
        snapshot = effect_ref.get(transaction=transaction)
        if not snapshot.exists:
            return
        data = snapshot.to_dict() or {}
        if data.get("state") == EffectState.DISPATCHED.value:
            data.update({"state": EffectState.UNKNOWN.value, "updated_at": commit_now})
            guards = []
            for resource in tuple(data.get("resources", ())):
                ref = _resource_ref(db, resource)
                guards.append((ref, ref.get(transaction=transaction)))
            transaction.set(effect_ref, data)
            for ref, guard_snapshot in guards:
                guard = guard_snapshot.to_dict() or {}
                if effect_id in guard.get("unsettled_effect_ids", ()):
                    guard.update({"mutation_state": EffectState.UNKNOWN.value, "updated_at": commit_now})
                    transaction.set(ref, guard)

    _txn(db.transaction())


def _confirm(db: Any, operation_id: str, effect_id: str, owner: str, fence: int,
             now: datetime, result: Any, *, clock: Callable[[], datetime] | None = None) -> bool:
    effect_ref = _effect_ref(db, operation_id, effect_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        commit_now = _aware(clock()) if clock is not None else now
        snapshot = effect_ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        data = snapshot.to_dict() or {}
        op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
        op_snapshot = op_ref.get(transaction=transaction)
        if not op_snapshot.exists or not _valid_lease(op_snapshot.to_dict() or {}, owner, fence, commit_now):
            return False
        if data.get("owner") != owner or data.get("fence") != fence:
            return False
        if data.get("state") != EffectState.DISPATCHED.value:
            return False
        resources = tuple(data.get("resources", ()))
        refs = [(resource, _resource_ref(db, resource)) for resource in resources]
        snapshots = [(resource, ref, ref.get(transaction=transaction)) for resource, ref in refs]
        data.update({
            "state": EffectState.CONFIRMED.value,
            "confirmed_result": _result_data(result) if isinstance(result, OperationResult) else None,
            "result": result, "updated_at": commit_now,
        })
        transaction.set(effect_ref, data)
        for resource, ref, snap in snapshots:
            guard = snap.to_dict() or {}
            unsettled = [item for item in guard.get("unsettled_effect_ids", ()) if item != effect_id]
            guard.update({"unsettled_effect_ids": unsettled,
                          "mutation_state": EffectState.CONFIRMED.value if not unsettled else guard.get("mutation_state"),
                          "updated_at": commit_now})
            transaction.set(ref, guard)
        return True

    return _txn(db.transaction())


def reconcile_mutation(
    client: Any | None,
    *,
    effect_id: str,
    operation_id: str | None = None,
    owner: str,
    fence: int,
    now: datetime,
    confirmed: bool,
    result: Any = None,
    confirmed_result: Any = None,
    clock: Callable[[], datetime] | None = None,
    **_: Any,
) -> OperationEffect | None:
    """Settle an uncertain effect only under its operation's current lease."""
    now = _aware(now)
    db = _db(client)
    matching_guards = db.collection(RESOURCE_GUARDS_COLLECTION).where(
        "owner_effect_id", "==", effect_id,
    ).stream()
    operation_ids = {
        (snapshot.to_dict() or {}).get("owner_operation_id")
        for snapshot in matching_guards
        if (snapshot.to_dict() or {}).get("owner_operation_id")
    }
    if operation_id is not None:
        if operation_ids and operation_id not in operation_ids:
            return None
        resolved_operation_id = operation_id
    elif len(operation_ids) == 1:
        resolved_operation_id = next(iter(operation_ids))
    else:
        return None
    if not resolved_operation_id:
        return None
    effect_ref = _effect_ref(db, resolved_operation_id, effect_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationEffect | None:
        commit_now = _aware(clock()) if clock is not None else now
        effect_snapshot = effect_ref.get(transaction=transaction)
        if not effect_snapshot.exists:
            return None
        data = effect_snapshot.to_dict() or {}
        operation_ref = db.collection(OPERATIONS_COLLECTION).document(resolved_operation_id)
        operation_snapshot = operation_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not _valid_lease(operation_snapshot.to_dict() or {}, owner, fence, commit_now):
            return None
        if data.get("state") not in {EffectState.DISPATCHED.value, EffectState.UNKNOWN.value}:
            return _effect_from(data) if data.get("state") in {EffectState.CONFIRMED.value, EffectState.FAILED.value} else None
        if data.get("operation_id") != resolved_operation_id:
            return None
        resources = tuple(sorted(data.get("resources", ())))
        refs = [(resource, _resource_ref(db, resource)) for resource in resources]
        snapshots = [(resource, ref, ref.get(transaction=transaction)) for resource, ref in refs]
        final_state = EffectState.CONFIRMED if confirmed else EffectState.FAILED
        final_result = confirmed_result if confirmed_result is not None else result
        data.update({
            "state": final_state.value,
            "confirmed_result": _result_data(final_result) if isinstance(final_result, OperationResult) else None,
            "result": final_result if confirmed else None,
            "updated_at": commit_now,
        })
        transaction.set(effect_ref, data)
        for _, ref, snapshot in snapshots:
            guard = snapshot.to_dict() or {}
            unsettled = [item for item in guard.get("unsettled_effect_ids", ()) if item != effect_id]
            guard.update({"unsettled_effect_ids": unsettled,
                          "mutation_state": final_state.value if not unsettled else guard.get("mutation_state"),
                          "updated_at": commit_now})
            transaction.set(ref, guard)
        return _effect_from(data)

    return _txn(db.transaction())


def fail_unsent_mutation(
    client: Any | None, *, operation_id: str, effect_id: str, owner: str, fence: int,
    now: datetime,
) -> OperationEffect | None:
    """Release a known-unsent reservation only while its original owner/fence remains current.

    Expiry alone permits cleanup; takeover or a guard owned by another effect does not.
    """
    now = _aware(now)
    db = _db(client)
    operation_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    effect_ref = _effect_ref(db, operation_id, effect_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> OperationEffect | None:
        operation_snapshot = operation_ref.get(transaction=transaction)
        effect_snapshot = effect_ref.get(transaction=transaction)
        if not operation_snapshot.exists or not effect_snapshot.exists:
            return None
        operation = operation_snapshot.to_dict() or {}
        data = effect_snapshot.to_dict() or {}
        if (operation.get("lease_owner") != owner or operation.get("fence") != fence
                or data.get("operation_id") != operation_id or data.get("owner") != owner
                or data.get("fence") != fence
                or data.get("state") not in {EffectState.DISPATCHED.value, EffectState.UNKNOWN.value}):
            return None
        resources = tuple(sorted(data.get("resources", ())))
        refs = [(resource, _resource_ref(db, resource)) for resource in resources]
        snapshots = [(resource, ref, ref.get(transaction=transaction)) for resource, ref in refs]
        if any(
            not snapshot.exists
            or (guard := (snapshot.to_dict() or {})).get("owner_operation_id") != operation_id
            or guard.get("owner_effect_id") != effect_id
            or guard.get("owner") != owner
            or guard.get("fence") != fence
            or effect_id not in guard.get("unsettled_effect_ids", ())
            for _, _, snapshot in snapshots
        ):
            return None
        data.update({"state": EffectState.FAILED.value, "result": None, "updated_at": now})
        transaction.set(effect_ref, data)
        for _, ref, snapshot in snapshots:
            guard = snapshot.to_dict() or {}
            unsettled = [item for item in guard.get("unsettled_effect_ids", ()) if item != effect_id]
            guard.update({"unsettled_effect_ids": unsettled,
                          "mutation_state": EffectState.FAILED.value if not unsettled else guard.get("mutation_state"),
                          "updated_at": now})
            transaction.set(ref, guard)
        return _effect_from(data)

    return _txn(db.transaction())


def _lease_transaction(client: Any, operation_id: str, owner: str, fence: int, now: datetime,
                       resource_values: Iterable[str] | None, action: str, value: str | None = None) -> bool:
    db = client
    op_ref = db.collection(OPERATIONS_COLLECTION).document(operation_id)
    resources = tuple(sorted(set(resource_values or ())))
    guard_refs = [(resource, _resource_ref(db, resource)) for resource in resources]

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        op_snapshot = op_ref.get(transaction=transaction)
        guard_snapshots = [(resource, ref, ref.get(transaction=transaction)) for resource, ref in guard_refs]
        if not op_snapshot.exists:
            return False
        op_data = op_snapshot.to_dict() or {}
        if not _valid_lease(op_data, owner, fence, now):
            return False
        guard_data = [
            (resource, ref, snap.to_dict() if snap.exists else {})
            for resource, ref, snap in guard_snapshots
        ]
        if any(_locked_guard(data) for _, _, data in guard_data):
            return False
        if action == "checkpoint":
            refs = list(op_data.get("checkpoint_refs", ()))
            if value and value not in refs:
                refs.append(value)
            op_data.update({"checkpoint_refs": refs, "updated_at": now})
            transaction.set(op_ref, op_data)
        elif action == "release":
            op_data.update({"lease_owner": None, "lease_expires_at": None, "updated_at": now})
            transaction.set(op_ref, op_data)
        elif resources:
            expiry = op_data.get("lease_expires_at")
            for resource, ref, data in guard_data:
                data.update({
                    "resource": resource, "owner_operation_id": operation_id,
                    "owner_effect_id": None, "owner": owner, "fence": fence,
                    "expires_at": expiry, "mutation_state": action,
                    action: {"operation_id": operation_id, "fence": fence, "at": now},
                    "updated_at": now,
                })
                transaction.set(ref, data)
            op_data.update({action: {"resources": list(resources), "at": now}, "updated_at": now})
            transaction.set(op_ref, op_data)
        else:
            op_data.update({action: {"resources": list(resources), "at": now}, "updated_at": now})
            transaction.set(op_ref, op_data)
        return True

    return _txn(db.transaction())


def checkpoint_mutation(client: Any | None, *, operation_id: str, owner: str, fence: int,
                         checkpoint_ref: str, now: datetime) -> bool:
    return _lease_transaction(_db(client), operation_id, owner, fence, _aware(now), None, "checkpoint", checkpoint_ref)


def release_mutation(client: Any | None, *, operation_id: str, owner: str, fence: int, now: datetime) -> bool:
    return _lease_transaction(_db(client), operation_id, owner, fence, _aware(now), None, "release")


def finalize_document(client: Any | None, *, operation_id: str, owner: str, fence: int,
                      resources: Iterable[str], now: datetime) -> bool:
    return _lease_transaction(_db(client), operation_id, owner, fence, _aware(now), resources, "document_finalized")


def publish_index_completion(client: Any | None, *, operation_id: str, owner: str, fence: int,
                             resources: Iterable[str], now: datetime) -> bool:
    return _lease_transaction(_db(client), operation_id, owner, fence, _aware(now), resources, "index_completion_published")
