"""Safe execution and read-back reconciliation for durable provider mutations.

Call this inside a RetryGovernor stage callback. ``MutationConflict`` is a
known terminal decision that callers can record as a safe, non-retryable stage
failure. ``MutationOutcomeUnknown`` means the effect remains guarded, so
callers should terminate or yield visibly instead of treating it as a generic
retryable provider error. Retry budgets and physical call counts are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Iterable

from core.integrations.firestore.mutations import (
    MutationIdentityConflict,
    dispatch_mutation,
    fail_unsent_mutation,
    get_mutation_effect,
    get_mutation_result,
    reconcile_mutation,
)
from core.integrations.firestore.operations import validate_operation_lease
from core.operations.models import EffectState, OperationEffect


class MutationPolicy(StrEnum):
    """Provider concurrency contract used by an effect."""

    DOCS_REVISION = "docs_revision"
    DRIVE_METADATA = "drive_metadata"


@dataclass(frozen=True, slots=True)
class MutationSnapshot:
    """Minimal provider read-back, represented only by safe digests."""

    state_digest: str
    revision: str | None = None


class MutationConflict(RuntimeError):
    """A safe visible conflict for a precondition or settled effect."""

    code = "mutation_conflict"


class MutationOutcomeUnknown(RuntimeError):
    """The request may still land; its durable resource guard remains held."""

    code = "mutation_outcome_unknown"


class MutationNotSent(Exception):
    """Provider callback signal that a rejected write was definitely not sent."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        super().__init__("mutation was not sent")


class _KnownUnsent(Exception):
    """A safe precondition or read failure before provider dispatch."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        super().__init__()


class _SentUnconfirmed(Exception):
    """A sent provider write whose single post-write read-back did not prove it."""


_RESULT_MARKER = "__tlbrain_mutation_result_v1__"


def _unwrap_result(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {_RESULT_MARKER}:
        return value[_RESULT_MARKER]
    return value


def _wrapped_result(value: Any) -> dict[str, Any]:
    return {_RESULT_MARKER: value}


def _check_identity(
    effect: OperationEffect,
    *,
    operation_id: str,
    kind: str,
    resources: tuple[str, ...],
    payload_digest: str,
    desired_state_digest: str,
    expected_pre_state: str,
    expected_revision: str | None,
    policy: MutationPolicy,
) -> None:
    if (
        effect.operation_id != operation_id
        or effect.kind != kind
        or effect.resources != resources
        or effect.desired_payload_digest != payload_digest
        or effect.desired_state_digest != desired_state_digest
        or effect.expected_pre_state != expected_pre_state
        or effect.expected_revision != expected_revision
        or effect.mutation_policy != policy.value
    ):
        raise MutationIdentityConflict("effect ID already belongs to a different immutable identity")


def _safe_snapshot(read_back: Callable[[], MutationSnapshot]) -> MutationSnapshot:
    """Hide provider read errors and reject malformed snapshots without echoing data."""
    try:
        snapshot = read_back()
    except Exception:
        raise _SnapshotUnavailable() from None
    if (
        not isinstance(snapshot, MutationSnapshot)
        or not isinstance(snapshot.state_digest, str)
        or not snapshot.state_digest
        or (snapshot.revision is not None and not isinstance(snapshot.revision, str))
    ):
        raise _SnapshotUnavailable() from None
    return snapshot


class _SnapshotUnavailable(Exception):
    """A read-back failed or returned an invalid value."""


def _safe_snapshot_result(
    callback: Callable[[MutationSnapshot], Any], snapshot: MutationSnapshot,
) -> Any:
    """Hide result extraction details because they may include snapshot contents."""
    try:
        return callback(snapshot)
    except Exception:
        raise _ResultUnavailable() from None


class _ResultUnavailable(Exception):
    """A domain result could not safely be derived from a snapshot."""


def _settle(
    client: Any,
    *,
    operation_id: str,
    effect_id: str,
    owner: str,
    fence: int,
    now: datetime,
    confirmed: bool,
    result: Any = None,
    clock: Callable[[], datetime] | None = None,
) -> OperationEffect | None:
    try:
        return reconcile_mutation(
            client, effect_id=effect_id, operation_id=operation_id, owner=owner,
            fence=fence, now=(clock() if clock is not None else now), confirmed=confirmed, result=result,
            clock=clock,
        )
    except Exception:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None


def _canonical_confirmed_result(
    client: Any, operation_id: str, effect_id: str,
) -> tuple[bool, Any]:
    try:
        effect = get_mutation_effect(client, operation_id, effect_id)
        if effect is None or effect.state != EffectState.CONFIRMED:
            return False, None
        found, result = get_mutation_result(client, operation_id, effect_id)
        return (True, _unwrap_result(result)) if found else (False, None)
    except Exception:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None


def _read_effect(client: Any, operation_id: str, effect_id: str) -> OperationEffect | None:
    """Hide storage errors so database details never reach a caller."""
    try:
        return get_mutation_effect(client, operation_id, effect_id)
    except Exception:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None


def _reconcile_existing(
    client: Any,
    *,
    operation_id: str,
    effect_id: str,
    owner: str,
    fence: int,
    now: datetime,
    desired_state_digest: str,
    read_back: Callable[[], MutationSnapshot],
    result_from_snapshot: Callable[[MutationSnapshot], Any],
    clock: Callable[[], datetime] | None = None,
) -> Any:
    try:
        snapshot = _safe_snapshot(read_back)
    except _SnapshotUnavailable:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None
    if snapshot.state_digest != desired_state_digest:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved")
    try:
        result = _safe_snapshot_result(result_from_snapshot, snapshot)
    except _ResultUnavailable:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None
    settled = _settle(
        client, operation_id=operation_id, effect_id=effect_id, owner=owner,
        fence=fence, now=now, confirmed=True, result=_wrapped_result(result),
        clock=clock,
    )
    if settled is not None and settled.state == EffectState.CONFIRMED:
        found, canonical = _canonical_confirmed_result(client, operation_id, effect_id)
        if found:
            return canonical
    if settled is not None and settled.state == EffectState.FAILED:
        raise MutationConflict("mutation effect was already settled as failed")
    raise MutationOutcomeUnknown("mutation outcome remains unresolved")


def _validate_inputs(
    *,
    operation_id: str,
    effect_id: str,
    owner: str,
    kind: str,
    resources: tuple[str, ...],
    payload_digest: str,
    desired_state_digest: str,
    expected_pre_state: str,
    expected_revision: str | None,
    policy: MutationPolicy,
    now: datetime,
) -> None:
    if (
        not isinstance(operation_id, str) or not operation_id
        or "/" in operation_id
        or not isinstance(effect_id, str) or not effect_id or "/" in effect_id
        or not isinstance(owner, str) or not owner
    ):
        raise ValueError("operation_id and a safe effect_id are required")
    if not isinstance(kind, str) or not kind or not resources or any(
        not isinstance(item, str) or not item for item in resources
    ):
        raise ValueError("kind and at least one non-empty resource are required")
    if kind not in {"content_update", "parent_move", "provider_call"}:
        raise ValueError("unsupported mutation kind")
    if any(not isinstance(value, str) or not value for value in (
        payload_digest, desired_state_digest, expected_pre_state,
    )):
        raise ValueError("payload, desired state, and expected pre-state are required")
    if not isinstance(policy, MutationPolicy):
        raise ValueError("unsupported mutation policy")
    if policy == MutationPolicy.DOCS_REVISION and (
        not isinstance(expected_revision, str) or not expected_revision
    ):
        raise ValueError("Docs mutations require an expected revision")
    if policy == MutationPolicy.DRIVE_METADATA and expected_revision is not None:
        raise ValueError("Drive metadata mutations do not support revisions")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")


def _validate_callbacks(
    *,
    fence: int,
    resources: Iterable[str],
    read_back: Callable[[], MutationSnapshot],
    mutation_callback: Callable[[str | None], Any],
    result_from_snapshot: Callable[[MutationSnapshot], Any] | None,
) -> None:
    if isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
        raise ValueError("fence must be a positive integer")
    if isinstance(resources, (str, bytes, bytearray)):
        raise ValueError("resources must be an iterable of resource identifiers")
    if not callable(read_back) or not callable(mutation_callback):
        raise ValueError("read_back and mutation_callback must be callable")
    if result_from_snapshot is not None and not callable(result_from_snapshot):
        raise ValueError("result_from_snapshot must be callable")


def _check_current_lease_before_dispatch(
    client: Any, *, operation_id: str, owner: str, fence: int,
    now: datetime, clock: Callable[[], datetime] | None,
) -> None:
    """Reject provider entry when the fresh operation lease check fails."""
    try:
        check_time = clock() if clock is not None else now
        if not validate_operation_lease(
            client, operation_id, owner=owner, fence=fence, now=check_time,
        ):
            raise _KnownUnsent()
    except _KnownUnsent:
        raise
    except Exception:
        raise _KnownUnsent() from None


def _invoke_and_verify_post_write(
    *,
    mutation_callback: Callable[[str | None], Any],
    callback_revision: str | None,
    read_back: Callable[[], MutationSnapshot],
    desired_state_digest: str,
    result_from_snapshot: Callable[[MutationSnapshot], Any],
) -> dict[str, Any]:
    """Invoke once, then prove the desired state or recover a lost response."""
    try:
        write_result = mutation_callback(callback_revision)
        write_succeeded = True
    except MutationNotSent as error:
        raise _KnownUnsent(error.error) from None
    except Exception:
        write_result = None
        write_succeeded = False

    try:
        post_snapshot = _safe_snapshot(read_back)
    except _SnapshotUnavailable:
        raise _SentUnconfirmed() from None
    if post_snapshot.state_digest != desired_state_digest:
        raise _SentUnconfirmed()
    if write_succeeded:
        return _wrapped_result(write_result)
    try:
        recovered = _safe_snapshot_result(result_from_snapshot, post_snapshot)
    except _ResultUnavailable:
        raise _SentUnconfirmed() from None
    return _wrapped_result(recovered)


def execute_mutation(
    client: Any,
    *,
    operation_id: str,
    effect_id: str,
    owner: str,
    fence: int,
    kind: str,
    resources: Iterable[str],
    payload_digest: str,
    desired_state_digest: str,
    expected_pre_state: str,
    expected_revision: str | None,
    policy: MutationPolicy,
    now: datetime,
    read_back: Callable[[], MutationSnapshot],
    mutation_callback: Callable[[str | None], Any],
    result_from_snapshot: Callable[[MutationSnapshot], Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Any:
    """Execute one guarded mutation or reconcile its existing durable effect.

    A fresh effect can perform one pre-write read and one post-write read. An
    existing uncertain effect performs at most one read. The callback receives
    the persisted Docs revision, or ``None`` for Drive metadata, which has no
    revision/CAS parameter. An unchanged pre-state after a sent request remains
    unresolved because that request may still land later. Persisted callback
    results must use Firestore-compatible data types.
    """
    _validate_callbacks(
        fence=fence, resources=resources, read_back=read_back,
        mutation_callback=mutation_callback, result_from_snapshot=result_from_snapshot,
    )
    try:
        supplied_resources = tuple(resources)
    except Exception:
        raise ValueError("at least one non-empty resource is required") from None
    _validate_inputs(
        operation_id=operation_id, effect_id=effect_id, owner=owner, kind=kind,
        resources=supplied_resources, payload_digest=payload_digest,
        desired_state_digest=desired_state_digest, expected_pre_state=expected_pre_state,
        expected_revision=expected_revision, policy=policy, now=now,
    )
    clean_resources = tuple(sorted(set(supplied_resources)))
    if result_from_snapshot is None:
        result_from_snapshot = lambda _: None

    existing = _read_effect(client, operation_id, effect_id)
    if existing is not None:
        _check_identity(
            existing, operation_id=operation_id, kind=kind, resources=clean_resources,
            payload_digest=payload_digest, desired_state_digest=desired_state_digest,
            expected_pre_state=expected_pre_state, expected_revision=expected_revision,
            policy=policy,
        )
        if existing.state == EffectState.CONFIRMED:
            found, result = _canonical_confirmed_result(client, operation_id, effect_id)
            if found:
                return result
            raise MutationOutcomeUnknown("confirmed mutation result is unavailable")
        if existing.state in {EffectState.DISPATCHED, EffectState.UNKNOWN}:
            return _reconcile_existing(
                client, operation_id=operation_id, effect_id=effect_id, owner=owner,
                fence=fence, now=now, desired_state_digest=desired_state_digest,
                read_back=read_back, result_from_snapshot=result_from_snapshot,
                clock=clock,
            )
        raise MutationConflict("mutation effect is already settled as failed")

    def reserved_attempt() -> Any:
        """Recheck lease ownership after pre-read and immediately before provider code."""
        try:
            snapshot = _safe_snapshot(read_back)
        except _SnapshotUnavailable:
            raise _KnownUnsent() from None
        if snapshot.state_digest == desired_state_digest:
            try:
                return _wrapped_result(_safe_snapshot_result(result_from_snapshot, snapshot))
            except _ResultUnavailable:
                raise _KnownUnsent() from None
        if snapshot.state_digest != expected_pre_state:
            raise _KnownUnsent()
        if policy == MutationPolicy.DOCS_REVISION and snapshot.revision != expected_revision:
            raise _KnownUnsent()
        _check_current_lease_before_dispatch(
            client, operation_id=operation_id, owner=owner, fence=fence,
            now=now, clock=clock,
        )
        callback_revision = expected_revision if policy == MutationPolicy.DOCS_REVISION else None
        return _invoke_and_verify_post_write(
            mutation_callback=mutation_callback, callback_revision=callback_revision,
            read_back=read_back, desired_state_digest=desired_state_digest,
            result_from_snapshot=result_from_snapshot,
        )

    try:
        dispatch_mutation(
            client, operation_id=operation_id, effect_id=effect_id, owner=owner,
            fence=fence, kind=kind, resources=clean_resources,
            payload_digest=payload_digest, desired_state_digest=desired_state_digest,
            mutation_policy=policy.value, now=now, mutation_callback=reserved_attempt,
            expected_pre_state=expected_pre_state, expected_revision=expected_revision,
            clock=clock,
        )
    except MutationIdentityConflict:
        raise
    except _KnownUnsent as unsent:
        try:
            settled = fail_unsent_mutation(
                client, operation_id=operation_id, effect_id=effect_id, owner=owner,
                fence=fence, now=(clock() if clock is not None else now),
            )
        except Exception:
            settled = None
        if isinstance(unsent.error, Exception):
            raise unsent.error
        if settled is not None and settled.state == EffectState.FAILED:
            raise MutationConflict("mutation precondition could not be verified") from None
        if settled is not None and settled.state == EffectState.CONFIRMED:
            found, result = _canonical_confirmed_result(client, operation_id, effect_id)
            if found:
                return result
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None
    except Exception:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved") from None

    found, result = _canonical_confirmed_result(client, operation_id, effect_id)
    if found:
        return result
    current = _read_effect(client, operation_id, effect_id)
    if current is not None and current.state == EffectState.FAILED:
        raise MutationConflict("mutation effect is already settled as failed")
    if current is not None and current.state in {EffectState.DISPATCHED, EffectState.UNKNOWN}:
        raise MutationOutcomeUnknown("mutation outcome remains unresolved")
    raise MutationConflict("mutation could not be reserved")
