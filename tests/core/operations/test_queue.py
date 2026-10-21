"""Focused offline tests for transactional context operation queues."""

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from core.integrations.firestore.operation_queue import (
    HEARTBEAT_INTERVAL_SECONDS,
    LEASE_SECONDS,
    claim_head,
    complete_head,
    fail_head,
    get_operation,
    heartbeat,
    register_operation,
    release_head,
)
from core.operations.queue import QueueOperationState


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _context(client, context_id: str = "context"):
    return client.collection("contexts").document(context_id)


def _register(client, context, operation_id: str, *, digest: str | None = None, ref: str | None = None):
    return register_operation(
        client, context, operation_id, identity_digest=digest or f"digest:{operation_id}",
        input_ref=ref or f"input:{operation_id}", now=NOW,
    )


def test_registration_assigns_fifo_sequence_and_is_idempotent(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    first = _register(client, context, "op-a")
    replay = _register(client, context, "op-a")
    second = _register(client, context, "op-b")

    assert first.sequence == replay.sequence == 1
    assert second.sequence == 2
    data = context.get().to_dict()
    assert data["next_operation_sequence"] == 3
    assert data["reserved_operation_id"] == "op-a"
    assert data["head_sequence"] == 1
    assert data.get("active_operation_id") is None
    assert firestore_fake.documents["contexts/context/operations/op-a"]["next_operation_id"] == "op-b"

    with pytest.raises(ValueError, match="different immutable identity"):
        _register(client, context, "op-a", digest="digest:changed")


def test_claim_is_head_only_and_competing_worker_cannot_acquire(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    _register(client, context, "op-a")
    _register(client, context, "op-b")

    first = claim_head(client, context, owner="worker-a", now=NOW)
    assert first is not None
    assert first.operation_id == "op-a" and first.sequence == 1
    assert first.fence == 1
    assert first.expires_at == NOW + timedelta(seconds=LEASE_SECONDS)
    assert claim_head(client, context, owner="worker-a", now=NOW) is None
    assert claim_head(client, context, owner="worker-b", now=NOW) is None
    assert context.get().to_dict()["reserved_operation_id"] == "op-a"


def test_heartbeat_expiry_takeover_and_stale_fencing(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    _register(client, context, "op-a")
    lease = claim_head(client, context, owner="old", now=NOW)
    assert lease is not None
    assert HEARTBEAT_INTERVAL_SECONDS == 30

    renewed = heartbeat(
        client, context, operation_id="op-a", owner="old", fence=lease.fence,
        now=NOW + timedelta(seconds=30),
    )
    assert renewed is not None
    assert renewed.expires_at == NOW + timedelta(seconds=150)
    assert heartbeat(
        client, context, operation_id="op-a", owner="old", fence=lease.fence,
        now=renewed.expires_at,
    ) is None

    takeover_at = renewed.expires_at
    takeover = claim_head(client, context, owner="new", now=takeover_at)
    assert takeover is not None and takeover.fence > lease.fence
    assert heartbeat(
        client, context, operation_id="op-a", owner="old", fence=lease.fence,
        now=takeover_at + timedelta(seconds=1),
    ) is None
    assert not release_head(
        client, context, operation_id="op-a", owner="old", fence=lease.fence,
        now=takeover_at + timedelta(seconds=1),
    )
    assert not complete_head(
        client, context, operation_id="op-a", owner="old", fence=lease.fence,
        now=takeover_at + timedelta(seconds=1),
    )


def test_release_preserves_head_then_terminal_advances_and_empty_clears(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    _register(client, context, "op-a")
    _register(client, context, "op-b")

    first = claim_head(client, context, owner="worker", now=NOW)
    assert first is not None
    assert release_head(
        client, context, operation_id="op-a", owner="worker", fence=first.fence,
        now=NOW + timedelta(seconds=1),
    )
    context_data = context.get().to_dict()
    assert context_data["reserved_operation_id"] == "op-a"
    assert context_data.get("active_operation_id") is None
    assert get_operation(context, "op-a").state is QueueOperationState.RUNNING

    first = claim_head(client, context, owner="worker-2", now=NOW + timedelta(seconds=121))
    assert first is not None and first.operation_id == "op-a" and first.fence > 1
    assert complete_head(
        client, context, operation_id="op-a", owner="worker-2", fence=first.fence,
        now=NOW + timedelta(seconds=122),
    )
    context_data = context.get().to_dict()
    assert context_data["reserved_operation_id"] == "op-b"
    assert context_data["head_sequence"] == 2
    assert context_data.get("active_operation_id") is None
    assert get_operation(context, "op-a").state is QueueOperationState.COMPLETED

    second = claim_head(client, context, owner="worker-3", now=NOW + timedelta(seconds=123))
    assert second is not None and second.operation_id == "op-b"
    assert fail_head(
        client, context, operation_id="op-b", owner="worker-3", fence=second.fence,
        now=NOW + timedelta(seconds=124),
    )
    context_data = context.get().to_dict()
    assert context_data.get("reserved_operation_id") is None
    assert context_data.get("head_sequence") is None
    assert context_data.get("active_operation_id") is None
    assert context_data["fence"] == second.fence
    assert get_operation(context, "op-b").state is QueueOperationState.FAILED


def test_context_queues_have_independent_sequence_and_leases(firestore_fake):
    client = firestore_fake.client_class()
    first_context = _context(client, "first")
    second_context = _context(client, "second")
    _register(client, first_context, "op-a")
    _register(client, first_context, "op-b")
    independent = _register(client, second_context, "op-x")

    assert independent.sequence == 1
    assert claim_head(client, first_context, owner="worker-a", now=NOW).operation_id == "op-a"
    other_lease = claim_head(client, second_context, owner="worker-b", now=NOW)
    assert other_lease is not None and other_lease.operation_id == "op-x"


def _synchronize_first_context_reads(context, participant_count: int = 2) -> None:
    """Make each concurrent transaction read the same initial context revision."""
    barrier = Barrier(participant_count)
    count_lock = Lock()
    calls = 0
    original_get = context.get

    def synchronized_get(*, transaction=None, **kwargs):
        nonlocal calls
        if transaction is not None:
            with count_lock:
                should_wait = calls < participant_count
                calls += 1
            if should_wait:
                barrier.wait(timeout=5)
        return original_get(transaction=transaction, **kwargs)

    context.get = synchronized_get


def test_concurrent_registration_assigns_unique_consecutive_sequences(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    _synchronize_first_context_reads(context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_register, client, context, operation_id)
            for operation_id in ("op-a", "op-b")
        ]
        entries = [future.result(timeout=10) for future in futures]

    assert sorted(entry.sequence for entry in entries) == [1, 2]
    context_data = context.get().to_dict()
    assert context_data["next_operation_sequence"] == 3
    assert context_data["reserved_operation_id"] in {"op-a", "op-b"}
    assert context_data["queue_tail_operation_id"] in {"op-a", "op-b"}


def test_concurrent_claims_issue_exactly_one_live_lease(firestore_fake):
    client = firestore_fake.client_class()
    context = _context(client)
    _register(client, context, "op-a")
    _synchronize_first_context_reads(context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(claim_head, client, context, owner=owner, now=NOW)
            for owner in ("worker-a", "worker-b")
        ]
        leases = [future.result(timeout=10) for future in futures]

    assert sum(lease is not None for lease in leases) == 1
    assert context.get().to_dict()["active_operation_id"] == "op-a"

