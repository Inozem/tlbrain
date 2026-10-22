"""Focused contract tests for the canonical operation repository."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.operations import (
    claim_operation,
    complete_operation,
    complete_stage_attempt,
    create_operation,
    fail_operation,
    fail_stage_attempt,
    get_operation,
    get_stage,
    list_stale_operations,
    release_operation,
    renew_operation_lease,
    reserve_stage_attempt,
)
from core.operations.models import OperationError, OperationOrigin, OperationResult


NOW = datetime(2026, 4, 5, tzinfo=timezone.utc)
OP_ID = "repository-test-operation"


def _create(client, **kwargs):
    values = {
        "operation_id": OP_ID,
        "origin": OperationOrigin.MCP,
        "caller_binding": "caller-1",
        "immutable_input_digest": "sha256:input",
        "immutable_input_ref": "input-ref-1",
        "now": NOW,
    }
    values.update(kwargs)
    return create_operation(client, **values)


def test_create_round_trips_immutable_refs_nullable_doc_id_and_timestamps(firestore_fake):
    client = firestore_fake.client_class()
    created = _create(client, origin_operation_ref="request-1")
    loaded = get_operation(client, OP_ID)

    assert loaded == created
    assert loaded.origin_operation_ref == "request-1"
    assert loaded.immutable_input_digest == "sha256:input"
    assert loaded.immutable_input_ref == "input-ref-1"
    assert loaded.doc_id is None
    assert loaded.created_at == loaded.updated_at == loaded.status_updated_at == NOW
    stored = firestore_fake.documents[f"dialog_operations/{OP_ID}"]
    assert "input" not in str(stored.get("raw_input", ""))
    assert f"dialog_operations/{OP_ID}" in firestore_fake.documents


def test_create_replay_requires_same_immutable_identity(firestore_fake):
    client = firestore_fake.client_class()
    first = _create(client)
    replay = _create(client, now=NOW + timedelta(days=1))
    assert replay == first
    with pytest.raises(ValueError, match="different immutable identity"):
        _create(client, immutable_input_digest="sha256:different")
    for changed in ({"title": "different"}, {"provider": "other"}, {"save_intent_id": "other-intent"}):
        with pytest.raises(ValueError, match="different immutable identity"):
            _create(client, **changed)


def test_stale_query_returns_only_bounded_old_pending_and_running_mcp_operations(firestore_fake):
    client = firestore_fake.client_class()
    old = NOW - timedelta(minutes=10)
    _create(client, operation_id="stale-pending", origin_operation_ref="ref-pending", now=old)
    _create(client, operation_id="fresh-pending", origin_operation_ref="ref-fresh", now=NOW)
    _create(client, operation_id="stale-telegram", origin=OperationOrigin.TELEGRAM,
            origin_operation_ref="ref-telegram", now=old)
    _create(client, operation_id="stale-running", origin_operation_ref="ref-running", now=old)
    running = claim_operation(client, "stale-running", owner="worker", now=old, lease_seconds=2)
    assert running is not None
    client.collection("dialog_operations").document("stale-running").update({"updated_at": old})

    records = list_stale_operations(client, cutoff=NOW - timedelta(minutes=5), limit=10)

    assert {record.operation_id for record in records} == {"stale-pending", "stale-running"}
    assert list_stale_operations(
        client, cutoff=NOW - timedelta(minutes=5), limit=1
    )[0].operation_id == "stale-pending"
    with pytest.raises(ValueError, match="limit"):
        list_stale_operations(client, cutoff=NOW, limit=501)


def test_claim_reclaim_fence_renew_and_terminal_state(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    first = claim_operation(client, OP_ID, owner="first", now=NOW, lease_seconds=5)
    assert first is not None and first.fence == 1 and first.lease_owner == "first"
    assert claim_operation(client, OP_ID, owner="busy", now=NOW + timedelta(seconds=1), lease_seconds=5) is None
    renewed = renew_operation_lease(
        client, OP_ID, owner="first", fence=first.fence, now=NOW + timedelta(seconds=1), lease_seconds=10,
    )
    assert renewed is not None and renewed.attempts_total == 0
    assert release_operation(client, OP_ID, owner="first", fence=first.fence, now=NOW + timedelta(seconds=2))
    assert get_operation(client, OP_ID).state.value == "running"
    second = claim_operation(client, OP_ID, owner="second", now=NOW + timedelta(seconds=2), lease_seconds=5)
    assert second is not None and second.fence == 2
    expired = NOW + timedelta(seconds=8)
    third = claim_operation(client, OP_ID, owner="third", now=expired, lease_seconds=5)
    assert third is not None and third.fence == 3
    assert release_operation(client, OP_ID, owner="second", fence=second.fence, now=expired) is False
    assert complete_operation(
        client, OP_ID, owner="third", fence=third.fence,
        result=OperationResult(document_id="doc-1"), now=expired,
    )
    final = get_operation(client, OP_ID)
    assert final.state.value == "completed"
    assert final.doc_id == "doc-1"
    assert final.finished_at == expired
    assert release_operation(client, OP_ID, owner="third", fence=third.fence, now=expired) is False
    assert claim_operation(client, OP_ID, owner="revive", now=expired, lease_seconds=5) is None


def test_stage_attempt_state_and_failure_summary_survive_recreation(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    lease = claim_operation(client, OP_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None

    reserved = reserve_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-1", now=NOW,
    )
    assert reserved is not None
    assert reserved.attempts_started == 1 and reserved.in_flight_attempt_id == "attempt-1"
    assert get_operation(client, OP_ID).attempts_total == 1
    # A process restart sees the unknown attempt and cannot replace it.
    assert get_stage(client, OP_ID, "parse/with arbitrary key") == reserved
    stage_doc_id = hashlib.sha256(f"{OP_ID}\0parse/with arbitrary key".encode()).hexdigest()
    assert f"dialog_operations/{OP_ID}/stages/{stage_doc_id}" in firestore_fake.documents
    assert reserve_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-2", now=NOW,
    ) is None
    assert get_stage(client, OP_ID, "parse/with arbitrary key").failures == 0

    error = OperationError("timeout", "provider", "safe message", True)
    assert fail_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-1", error=error, now=NOW,
        retry_not_before=NOW + timedelta(seconds=10),
    )
    failed = get_stage(client, OP_ID, "parse/with arbitrary key")
    assert failed.failures == 1 and failed.attempts_started == 1
    assert failed.in_flight_attempt_id is None
    assert failed.last_error.code == error.code
    assert failed.last_error.stage == "parse/with arbitrary key"
    assert failed.last_error.occurred_at == NOW
    assert failed.last_error.failures == 1 and failed.last_error.operation_id == OP_ID
    assert failed.last_error.sanitized_message == error.message
    assert get_operation(client, OP_ID).stage_failure_counts == (("parse/with arbitrary key", 1),)

    retried = reserve_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-2", now=NOW + timedelta(seconds=11),
    )
    assert retried is not None and retried.attempts_started == 2
    value = {"checkpoint": "saved"}
    assert complete_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-2", result=value, checkpoint_refs=("checkpoint-ref",), now=NOW + timedelta(seconds=12),
    )
    complete = get_stage(client, OP_ID, "parse/with arbitrary key")
    assert complete.result == value and complete.checkpoint_refs == ("checkpoint-ref",)
    assert reserve_stage_attempt(
        client, OP_ID, "parse/with arbitrary key", owner="worker", fence=lease.fence,
        attempt_id="attempt-3", now=NOW + timedelta(seconds=13),
    ) is None


def test_fail_operation_is_fenced_terminal_and_round_trips_safe_error(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    old = claim_operation(client, OP_ID, owner="old", now=NOW, lease_seconds=5)
    assert old is not None
    current = claim_operation(client, OP_ID, owner="current", now=NOW + timedelta(seconds=6), lease_seconds=30)
    assert current is not None and current.fence > old.fence
    error = OperationError(
        "provider_timeout", "provider", "sanitized timeout", True,
        service="drive", stage="upload", sanitized_message="sanitized timeout",
        occurred_at=NOW + timedelta(seconds=6), failures=3,
    )
    assert fail_operation(
        client, OP_ID, owner="old", fence=old.fence, error=error, now=NOW + timedelta(seconds=6),
    ) is False
    assert fail_operation(
        client, OP_ID, owner="current", fence=current.fence, error=error, now=NOW + timedelta(seconds=7),
    ) is True
    failed = get_operation(client, OP_ID)
    assert failed.state.value == "failed"
    assert failed.finished_at == NOW + timedelta(seconds=7)
    assert failed.status_updated_at == failed.finished_at
    assert failed.error.code == "provider_timeout"
    assert failed.error.service == "drive" and failed.error.stage == "upload"
    assert failed.error.sanitized_message == "sanitized timeout"
    assert failed.error.occurred_at == NOW + timedelta(seconds=6)
    assert failed.error.failures == 3 and failed.error.operation_id == OP_ID
    assert claim_operation(client, OP_ID, owner="revive", now=failed.finished_at, lease_seconds=10) is None


def test_legacy_errors_are_enriched_from_stage_and_operation_context(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    lease = claim_operation(client, OP_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    stage_key = "context-stage"
    assert reserve_stage_attempt(
        client, OP_ID, stage_key, owner="worker", fence=lease.fence,
        attempt_id="context-attempt", now=NOW,
    ) is not None
    legacy_error = OperationError("timeout", "provider", "sanitized legacy message", True)
    assert fail_stage_attempt(
        client, OP_ID, stage_key, owner="worker", fence=lease.fence,
        attempt_id="context-attempt", error=legacy_error, now=NOW,
    )
    stage_error = get_stage(client, OP_ID, stage_key).last_error
    assert stage_error.service is None
    assert stage_error.stage == stage_key and stage_error.occurred_at == NOW
    assert stage_error.failures == 1 and stage_error.operation_id == OP_ID
    assert stage_error.sanitized_message == "sanitized legacy message"

    fail_time = NOW + timedelta(seconds=1)
    assert fail_operation(
        client, OP_ID, owner="worker", fence=lease.fence, error=legacy_error, now=fail_time,
    )
    operation_error = get_operation(client, OP_ID).error
    assert operation_error.stage == stage_key and operation_error.occurred_at == fail_time
    assert operation_error.failures == 1 and operation_error.operation_id == OP_ID
    assert operation_error.sanitized_message == "sanitized legacy message"
