"""Offline durable retry-budget tests for operation stages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.qdrant import reader as qdrant_reader
from core.integrations.firestore.operations import claim_operation, create_operation, get_operation, get_stage
from core.operations.models import OperationError, OperationOrigin, OperationState
from core.operations.retries import (
    ConfirmedStageFailure,
    ConfirmedStageSuccess,
    ReconciledStageFailure,
    RetryBudgetExceeded,
    RetryGovernor,
    UnknownOutcome,
)
from core.utils import retry as retry_utils
from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY, with_retry


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


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_callback_past_lease_expiry_does_not_commit_or_release_stale_lease(
    firestore_fake, fake_clock, outcome,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = RetryGovernor(client, clock=fake_clock, lease_seconds=5)

    def callback():
        fake_clock.advance(seconds=6)
        if outcome == "failure":
            raise ValueError("synthetic provider failure")
        return "saved"

    with pytest.raises(RuntimeError, match="could not persist stage"):
        governor.run_stage(OPERATION_ID, "slow-callback", callback, now=NOW)

    stage = get_stage(client, OPERATION_ID, "slow-callback")
    operation = get_operation(client, OPERATION_ID)
    assert stage.in_flight_attempt_id is not None
    assert stage.attempts_started == 1 and stage.failures == 0
    assert operation.lease_owner is not None
    assert operation.lease_expires_at == NOW + timedelta(seconds=5)
    assert fake_clock.now() > operation.lease_expires_at


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_short_callback_persists_post_callback_time(firestore_fake, fake_clock, outcome):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)
    completed_at = NOW + timedelta(seconds=2)

    def callback():
        fake_clock.advance(seconds=2)
        if outcome == "failure":
            raise ValueError("synthetic provider failure")
        return "saved"

    if outcome == "failure":
        with pytest.raises(ValueError, match="synthetic provider failure"):
            governor.run_stage(
                OPERATION_ID, "short-callback", callback, now=NOW,
                retryable=lambda error: False,
            )
    else:
        assert governor.run_stage(OPERATION_ID, "short-callback", callback, now=NOW) == "saved"

    stage = get_stage(client, OPERATION_ID, "short-callback")
    assert stage.updated_at == completed_at
    if outcome == "failure":
        assert stage.last_error.occurred_at == completed_at
        operation = get_operation(client, OPERATION_ID)
        assert operation.state is OperationState.FAILED
        assert operation.finished_at == completed_at


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_expired_reconciliation_stays_in_flight_for_fresh_owner(
    firestore_fake, fake_clock, outcome,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = RetryGovernor(client, clock=fake_clock, lease_seconds=5)
    provider_calls = []

    def uncertain_provider_call():
        provider_calls.append("dispatch")
        raise UnknownOutcome("synthetic uncertain provider result")

    with pytest.raises(UnknownOutcome):
        governor.run_stage(OPERATION_ID, "slow-reconcile", uncertain_provider_call, now=NOW)
    attempt_id = get_stage(client, OPERATION_ID, "slow-reconcile").in_flight_attempt_id
    assert attempt_id

    error = OperationError(
        code="provider_rejected", category="execution", message="Provider rejected the request",
        retryable=True, stage="slow-reconcile",
    )

    def expired_reconcile():
        fake_clock.advance(seconds=6)
        if outcome == "failure":
            return ConfirmedStageFailure(error)
        return ConfirmedStageSuccess("reconciled")

    with pytest.raises(RuntimeError, match="could not persist reconciled stage"):
        governor.run_stage(
            OPERATION_ID, "slow-reconcile", uncertain_provider_call,
            reconcile=expired_reconcile, now=NOW,
        )

    stage = get_stage(client, OPERATION_ID, "slow-reconcile")
    expired_operation = get_operation(client, OPERATION_ID)
    assert stage.in_flight_attempt_id == attempt_id
    assert stage.attempts_started == 1 and stage.failures == 0
    assert expired_operation.lease_owner is not None
    assert expired_operation.lease_expires_at == NOW + timedelta(seconds=5)
    assert fake_clock.now() > expired_operation.lease_expires_at

    committed_at = fake_clock.now() + timedelta(seconds=1)

    def fresh_reconcile():
        fake_clock.advance(seconds=1)
        if outcome == "failure":
            return ConfirmedStageFailure(error)
        return ConfirmedStageSuccess("reconciled")

    if outcome == "failure":
        with pytest.raises(ReconciledStageFailure):
            governor.run_stage(
                OPERATION_ID, "slow-reconcile", uncertain_provider_call,
                reconcile=fresh_reconcile, now=fake_clock.now(),
            )
    else:
        assert governor.run_stage(
            OPERATION_ID, "slow-reconcile", uncertain_provider_call,
            reconcile=fresh_reconcile, now=fake_clock.now(),
        ) == "reconciled"

    recovered = get_stage(client, OPERATION_ID, "slow-reconcile")
    recovered_operation = get_operation(client, OPERATION_ID)
    assert provider_calls == ["dispatch"]
    assert recovered_operation.fence == expired_operation.fence + 1
    assert recovered_operation.lease_owner is None
    assert recovered.attempts_started == 1 and recovered.failures == (outcome == "failure")
    assert recovered.in_flight_attempt_id is None
    assert recovered.updated_at == committed_at
    if outcome == "failure":
        assert recovered.last_error.occurred_at == committed_at
    else:
        assert recovered.result == "reconciled"


def test_max_failures_cannot_exceed_three():
    with pytest.raises(ValueError, match="between 1 and 3"):
        RetryGovernor(max_failures=4)


def test_with_retry_preserves_legacy_default_attempts_and_backoffs(monkeypatch):
    physical_calls = []
    delays = []
    monkeypatch.setattr(retry_utils.time, "sleep", delays.append)

    @with_retry
    def provider_call():
        physical_calls.append("call")
        raise ConnectionError("synthetic outage")

    with pytest.raises(RuntimeError, match="failed after 3 attempts") as raised:
        provider_call()

    assert isinstance(raised.value.__cause__, ConnectionError)
    assert len(physical_calls) == 3
    assert delays == [1, 2]


def test_with_retry_preserves_legacy_rate_limit_backoffs(monkeypatch):
    physical_calls = []
    delays = []
    monkeypatch.setattr(retry_utils.time, "sleep", delays.append)
    monkeypatch.setattr(retry_utils, "_is_rate_limit", lambda error: True)

    @with_retry
    def provider_call():
        physical_calls.append("call")
        raise RuntimeError("synthetic rate limit")

    with pytest.raises(RuntimeError, match="failed after 3 attempts") as raised:
        provider_call()

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert len(physical_calls) == 3
    assert delays == [30, 60]


def test_single_attempt_policy_preserves_original_provider_exception_without_logging_it(monkeypatch, caplog):
    physical_calls = []
    delays = []
    monkeypatch.setattr(retry_utils.time, "sleep", delays.append)
    provider_error = ConnectionError("synthetic private payload")

    @with_retry
    def provider_call():
        physical_calls.append("call")
        raise provider_error

    with pytest.raises(ConnectionError) as raised:
        provider_call(retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)

    assert raised.value is provider_error
    assert physical_calls == ["call"]
    assert delays == []
    assert "synthetic private payload" not in caplog.text


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
    assert get_operation(client, OPERATION_ID).state is OperationState.FAILED


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


def test_process_interruption_preserves_in_flight_attempt_for_reconciliation(firestore_fake, fake_clock):
    class ProcessInterruption(BaseException):
        pass

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)

    with pytest.raises(ProcessInterruption, match="interrupted"):
        governor.run_stage(
            OPERATION_ID, "interrupted",
            lambda: (_ for _ in ()).throw(ProcessInterruption("interrupted")),
            now=fake_clock.now(),
        )

    stage = governor.get_stage(OPERATION_ID, "interrupted")
    assert stage.attempts_started == 1
    assert stage.failures == 0
    assert stage.in_flight_attempt_id is not None


def test_confirmed_reconciliation_failure_is_recorded_before_a_later_retry(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            OPERATION_ID, "reconcile-failure",
            lambda: (_ for _ in ()).throw(UnknownOutcome("provider outcome unknown")),
            now=fake_clock.now(),
        )
    stage = governor.get_stage(OPERATION_ID, "reconcile-failure")
    attempt_id = stage.in_flight_attempt_id
    assert attempt_id is not None

    safe_error = OperationError(
        code="provider_rejected", category="execution", message="Provider rejected the request",
        retryable=True, stage="reconcile-failure",
    )
    callback_calls = []
    with pytest.raises(ReconciledStageFailure, match="Provider rejected") as raised:
        governor.run_stage(
            OPERATION_ID, "reconcile-failure", lambda: callback_calls.append("unexpected"),
            reconcile=lambda: ConfirmedStageFailure(safe_error), now=fake_clock.now(),
        )

    failed = governor.get_stage(OPERATION_ID, "reconcile-failure")
    assert raised.value.error == safe_error
    assert callback_calls == []
    assert failed.failures == 1
    assert failed.in_flight_attempt_id is None
    assert failed.attempts_started == 1

    assert governor.run_stage(
        OPERATION_ID, "reconcile-failure", lambda: callback_calls.append("retry") or "ok",
        now=fake_clock.now(),
    ) == "ok"
    assert callback_calls == ["retry"]
    assert governor.get_stage(OPERATION_ID, "reconcile-failure").attempts_started == 2


def test_permanent_reconciliation_failure_marks_operation_failed(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            OPERATION_ID, "permanent-reconcile-failure",
            lambda: (_ for _ in ()).throw(UnknownOutcome("provider outcome unknown")),
            now=fake_clock.now(),
        )

    safe_error = OperationError(
        code="provider_rejected", category="execution", message="Provider rejected the request",
        retryable=False, stage="permanent-reconcile-failure",
    )
    with pytest.raises(ReconciledStageFailure):
        governor.run_stage(
            OPERATION_ID, "permanent-reconcile-failure", lambda: pytest.fail("must not dispatch"),
            reconcile=lambda: ConfirmedStageFailure(safe_error), now=fake_clock.now(),
        )

    assert governor.get_stage(OPERATION_ID, "permanent-reconcile-failure").failures == 1
    assert get_operation(client, OPERATION_ID).state is OperationState.FAILED


def test_confirmed_reconciliation_success_can_complete_with_none(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    governor = _governor(client, fake_clock)

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            OPERATION_ID, "reconcile-success",
            lambda: (_ for _ in ()).throw(UnknownOutcome("provider outcome unknown")),
            now=fake_clock.now(),
        )

    assert governor.run_stage(
        OPERATION_ID, "reconcile-success", lambda: pytest.fail("must not dispatch"),
        reconcile=lambda: ConfirmedStageSuccess(None), now=fake_clock.now(),
    ) is None
    stage = governor.get_stage(OPERATION_ID, "reconcile-success")
    assert stage.state.value == "completed"
    assert stage.result is None
    assert stage.in_flight_attempt_id is None
    assert governor.run_stage(
        OPERATION_ID, "reconcile-success", lambda: pytest.fail("completed stage must be reused"),
        now=fake_clock.now(),
    ) is None


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
    assert get_operation(client, OPERATION_ID).state is OperationState.FAILED
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
        sdk_fakes.qdrant.queue("scroll", error=ConnectionError("synthetic provider error"))
    physical_attempts = []

    def one_provider_call():
        physical_attempts.append("attempt")
        return qdrant_reader.fetch_utterances_by_order_indexes(
            "doc-retry", "root-retry", [1], retry_policy=SINGLE_ATTEMPT_RETRY_POLICY,
        )

    for _ in range(3):
        with pytest.raises(ConnectionError):
            _governor(client, fake_clock).run_stage(
                OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
            )
    assert physical_attempts == ["attempt"] * 3
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "scroll"]) == 3

    with pytest.raises(RetryBudgetExceeded):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
        )
    assert physical_attempts == ["attempt"] * 3
