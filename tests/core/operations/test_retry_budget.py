"""Offline durable retry-budget tests for operation stages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.gemini import llm
from core.integrations.firestore.operations import claim_operation, create_operation
from core.operations.models import OperationOrigin
from core.operations.retries import RetryBudgetExceeded, RetryGovernor, UnknownOutcome


OPERATION_ID = "synthetic-retry-operation"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _new_operation(client):
    create_operation(
        client,
        operation_id=OPERATION_ID,
        origin=OperationOrigin.TELEGRAM,
        caller_binding="synthetic-telegram-user",
        immutable_input_digest="sha256:retry-input",
        immutable_input_ref="retry-input-ref",
        now=NOW,
    )


def _governor(client, fake_clock):
    return RetryGovernor(client, clock=fake_clock, max_failures=3)


def test_failure_budget_survives_recreated_governors_and_caps_physical_executions(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    physical_runs = []

    def fail_once_per_run():
        physical_runs.append(fake_clock.now())
        raise ValueError("synthetic transient provider failure")

    for run in range(3):
        governor = _governor(client, fake_clock)
        with pytest.raises(ValueError, match="synthetic transient"):
            governor.run_stage(OPERATION_ID, "generate", fail_once_per_run, now=fake_clock.now())
        fake_clock.advance(seconds=1)
    recreated = _governor(client, fake_clock)
    with pytest.raises(RetryBudgetExceeded):
        recreated.run_stage(OPERATION_ID, "generate", fail_once_per_run, now=fake_clock.now())

    assert len(physical_runs) == 3
    stage = recreated.get_stage(OPERATION_ID, "generate")
    assert stage.failures == 3
    assert stage.attempts_started == 3


def test_attempt_is_reserved_before_callback_and_unknown_is_reconciled_before_retry(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    events = []
    governor = _governor(client, fake_clock)

    def uncertain_provider_attempt():
        stage = governor.get_stage(OPERATION_ID, "upload")
        events.append(("callback", stage.attempts_started, stage.in_flight_attempt_id))
        raise UnknownOutcome("synthetic timeout after request may have committed")

    with pytest.raises(UnknownOutcome):
        governor.run_stage(OPERATION_ID, "upload", uncertain_provider_attempt, now=fake_clock.now())
    assert events[0][1] == 1
    assert events[0][2]
    in_flight_attempt_id = events[0][2]

    def reconcile_unknown():
        events.append(("reconcile",))
        return None

    governor = _governor(client, fake_clock)
    persisted = governor.get_stage(OPERATION_ID, "upload")
    assert persisted.attempts_started == 1
    assert persisted.in_flight_attempt_id == in_flight_attempt_id
    assert persisted.failures == 0

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            OPERATION_ID, "upload", lambda: events.append(("must-not-dispatch",)),
            reconcile=reconcile_unknown, now=fake_clock.now(),
        )
    assert events[-1] == ("reconcile",)
    assert not any(event[0] == "must-not-dispatch" for event in events)


def test_waiting_yield_status_and_lease_events_do_not_consume_failure_budget(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="worker", now=fake_clock.now(), lease_seconds=60)
    assert lease is not None
    governor = _governor(client, fake_clock)
    governor.record_wait(OPERATION_ID, "generate", until=fake_clock.now() + timedelta(seconds=30))
    governor.record_yield(OPERATION_ID, "generate", reason="synthetic rate limit wait")
    governor.record_status(OPERATION_ID, "generate", status="waiting")
    governor.record_lease_renewal(
        OPERATION_ID, owner=lease.lease_owner, fence=lease.fence, now=fake_clock.now(),
    )
    stage = governor.get_stage(OPERATION_ID, "generate")
    assert stage.failures == 0
    assert stage.attempts_started == 0


def test_permanent_error_stops_early_and_successful_stage_is_retained(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)
    successful_calls = []
    saved = governor.run_stage(
        OPERATION_ID, "parse", lambda: successful_calls.append("parse") or {"checkpoint": "parse-v1"},
        now=fake_clock.now(),
    )
    fake_clock.advance(seconds=1)
    calls = []

    def permanent_failure():
        calls.append("provider")
        raise ValueError("synthetic permanent validation error")

    with pytest.raises(ValueError, match="permanent validation"):
        governor.run_stage(
            OPERATION_ID, "generate", permanent_failure,
            retryable=lambda error: False, now=fake_clock.now(),
        )
    assert calls == ["provider"]
    assert successful_calls == ["parse"]
    assert governor.get_stage(OPERATION_ID, "parse").result == saved
    assert governor.run_stage(
        OPERATION_ID, "parse", lambda: pytest.fail("completed stage must be reused"), now=fake_clock.now(),
    ) == saved


def test_explicit_fresh_operation_budget_can_reuse_same_save_intent(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    first = RetryGovernor(client, clock=fake_clock, max_failures=3)
    calls = []

    def transient_failure():
        calls.append("first-operation")
        raise ValueError("synthetic retryable failure")

    with pytest.raises(ValueError):
        first.run_stage(OPERATION_ID, "create", transient_failure, save_intent_id="shared-intent", now=fake_clock.now())

    second_id = "synthetic-explicit-retry-operation"
    create_operation(
        client, operation_id=second_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:retry-input",
        immutable_input_ref="retry-input-ref", save_intent_id="shared-intent", now=fake_clock.now(),
    )
    second = RetryGovernor(client, clock=fake_clock, max_failures=3)
    with pytest.raises(ValueError):
        second.run_stage(second_id, "create", lambda: calls.append("fresh-operation") or (_ for _ in ()).throw(ValueError("synthetic retryable failure")), save_intent_id="shared-intent", now=fake_clock.now())
    assert second.get_stage(second_id, "create").failures == 1
    assert first.get_stage(OPERATION_ID, "create").failures == 1


def test_durable_execution_makes_at_most_one_physical_provider_attempt(firestore_fake, fake_clock, sdk_fakes):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    for _ in range(3):
        sdk_fakes.gemini.queue("generate_content", error=ValueError("synthetic provider error"))
    # This direct callback seam checks one physical call per durable attempt.
    # T016 will cover the equivalent single-attempt/retry-policy seams through
    # the real integration path.
    physical_attempts = []

    def one_provider_call():
        physical_attempts.append("attempt")
        return llm.make_client().models.generate_content(model="synthetic-model", contents="offline prompt")

    for _ in range(3):
        with pytest.raises(ValueError):
            _governor(client, fake_clock).run_stage(
                OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
            )
    assert physical_attempts == ["attempt"] * 3
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == 3

    with pytest.raises(RetryBudgetExceeded):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
        )
    assert physical_attempts == ["attempt"] * 3
