"""Offline durability and lease tests for dialog operation persistence."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from core.integrations.firestore.operations import (
    claim_operation,
    complete_operation,
    create_operation,
    get_operation,
    release_operation,
)
from core.integrations.firestore.save_intents import (
    create_save_intent,
    dispatch_save_intent,
    get_save_intent,
    reconcile_save_intent,
)
from core.operations.models import OperationOrigin, OperationResult


OPERATION_ID = "synthetic-operation-persistence"
SAVE_INTENT_ID = "synthetic-save-intent"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _new_operation(client, *, operation_id: str = OPERATION_ID) -> None:
    create_operation(
        client,
        operation_id=operation_id,
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:synthetic-input",
        immutable_input_ref="synthetic-input-ref",
        now=NOW,
    )


def test_competing_transactional_claims_have_one_winner(firestore_fake, monkeypatch):
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    firestore_fake.set_clock(lambda: NOW)
    _new_operation(client)
    barrier = threading.Barrier(2)
    reads_lock = threading.Lock()
    first_reads = 0
    original_read = FakeTransaction._read

    def synchronized_read(transaction, reference):
        nonlocal first_reads
        snapshot = original_read(transaction, reference)
        should_wait = False
        with reads_lock:
            if first_reads < 2:
                first_reads += 1
                should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return snapshot

    monkeypatch.setattr(FakeTransaction, "_read", synchronized_read)

    def claim(owner: str):
        return claim_operation(client, OPERATION_ID, owner=owner, now=NOW, lease_seconds=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, ("worker-a", "worker-b")))

    assert sum(outcome is not None for outcome in outcomes) == 1
    assert sum(event["operation"] == "commit" and event.get("phase") == "conflict"
               for event in firestore_fake.trace()) == 1
    record = get_operation(client, OPERATION_ID)
    assert record.lease_owner in {"worker-a", "worker-b"}
    assert record.fence == 1


def test_expired_claim_rejects_stale_release_and_terminal_update(firestore_fake):
    client = firestore_fake.client_class()
    _new_operation(client)
    old = claim_operation(client, OPERATION_ID, owner="old-worker", now=NOW, lease_seconds=5)
    assert old is not None

    reclaimed_at = NOW + timedelta(seconds=6)
    current = claim_operation(client, OPERATION_ID, owner="new-worker", now=reclaimed_at, lease_seconds=30)
    assert current is not None
    assert current.fence > old.fence
    assert release_operation(client, OPERATION_ID, owner="old-worker", fence=old.fence, now=reclaimed_at) is False
    assert complete_operation(
        client, OPERATION_ID, owner="old-worker", fence=old.fence,
        result=OperationResult(document_id="stale-doc"), now=reclaimed_at,
    ) is False

    assert complete_operation(
        client, OPERATION_ID, owner="new-worker", fence=current.fence,
        result=OperationResult(document_id="durable-doc"), now=reclaimed_at,
    ) is True
    assert release_operation(
        client, OPERATION_ID, owner="new-worker", fence=current.fence, now=reclaimed_at,
    ) is False
    final = get_operation(client, OPERATION_ID)
    assert final.state.value == "completed"
    assert final.result.document_id == "durable-doc"


def test_confirmed_effect_replay_returns_saved_result_without_redispatch(firestore_fake):
    client = firestore_fake.client_class()
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    effects = []
    result = OperationResult(document_id="created-once", title="Synthetic")

    def create_callback():
        effects.append("create")
        return result

    first = dispatch_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        now=NOW, create_callback=create_callback,
    )
    replay = dispatch_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        now=NOW, create_callback=create_callback,
    )

    assert first == result
    assert replay == result
    assert effects == ["create"]
    saved = get_save_intent(client, SAVE_INTENT_ID)
    assert saved.state.value == "confirmed"
    assert saved.confirmed_result == result


def test_save_intent_dispatch_gate_is_durable_before_external_create(firestore_fake):
    client = firestore_fake.client_class()
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    intent = create_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID,
        desired_content_hash="sha256:content", desired_parent_id_hash="sha256:parent", now=NOW,
    )
    assert intent.state.value == "planned"
    callback_observations = []

    def external_create():
        durable = get_save_intent(client, SAVE_INTENT_ID)
        callback_observations.append((durable.state.value, durable.dispatch_token))
        return OperationResult(document_id="synthetic-created-doc")

    result = dispatch_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        now=NOW, create_callback=external_create,
    )
    assert callback_observations == [("dispatched", callback_observations[0][1])]
    assert callback_observations[0][1]
    assert result.document_id == "synthetic-created-doc"
    assert get_save_intent(client, SAVE_INTENT_ID).state.value == "confirmed"


def test_crash_after_dispatch_retains_unknown_marker_until_reconciliation(firestore_fake):
    client = firestore_fake.client_class()
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    create_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID,
        desired_content_hash="sha256:content", desired_parent_id_hash="sha256:parent", now=NOW,
    )
    physical_creates = []

    def process_crash_after_remote_create():
        physical_creates.append("remote-create")
        raise RuntimeError("synthetic process loss after dispatch")

    try:
        dispatch_save_intent(
            client, SAVE_INTENT_ID, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
            now=NOW, create_callback=process_crash_after_remote_create,
        )
    except RuntimeError as error:
        assert "synthetic process loss" in str(error)

    marker = get_save_intent(client, SAVE_INTENT_ID)
    assert marker.state.value == "unknown"
    assert marker.dispatch_token
    assert dispatch_save_intent(
        client, SAVE_INTENT_ID, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        now=NOW, create_callback=process_crash_after_remote_create,
    ) is None
    assert physical_creates == ["remote-create"]

    reconciled = reconcile_save_intent(
        client, SAVE_INTENT_ID, owner="worker", fence=lease.fence, now=NOW,
        external_result=OperationResult(document_id="found-after-reconcile"),
    )
    assert reconciled.state.value == "confirmed"
    assert reconciled.confirmed_result.document_id == "found-after-reconcile"
