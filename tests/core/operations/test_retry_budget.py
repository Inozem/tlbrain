"""Offline durable retry-budget tests for operation stages."""

from __future__ import annotations

import asyncio
import threading
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


def test_run_stage_uses_and_retains_supplied_operation_lease(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="outer-worker", now=NOW, lease_seconds=60)
    assert lease is not None
    governor = _governor(client, fake_clock)

    assert governor.run_stage(
        OPERATION_ID, "external-lease", lambda: "saved", now=NOW,
        owner="outer-worker", fence=lease.fence,
    ) == "saved"
    operation = get_operation(client, OPERATION_ID)
    assert operation.lease_owner == "outer-worker"
    assert operation.fence == lease.fence


def test_run_stage_rejects_stale_supplied_lease_before_callback(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="outer-worker", now=NOW, lease_seconds=60)
    assert lease is not None
    calls = []

    with pytest.raises(RuntimeError, match="lease is unavailable"):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "stale-external-lease", lambda: calls.append("called"), now=NOW,
            owner="wrong-worker", fence=lease.fence,
        )
    assert calls == []


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


def test_async_stage_offloads_store_read_and_keeps_callback_on_origin_loop(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="async-worker", now=NOW, lease_seconds=60)
    assert lease is not None
    governor = _governor(client, fake_clock)
    entered, release = threading.Event(), threading.Event()
    original_call = governor._store_call
    loop_thread = threading.get_ident()
    callback_threads = []

    def blocked_call(name, fallback, *args, **kwargs):
        if name == "get_stage" and not entered.is_set():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("stage read release timed out")
        return original_call(name, fallback, *args, **kwargs)

    governor._store_call = blocked_call

    async def invoke():
        task = asyncio.create_task(governor.run_stage_async(
            OPERATION_ID, "async-offload",
            lambda: callback_threads.append(threading.get_ident()) or "saved",
            owner="async-worker", fence=lease.fence, now=NOW,
        ))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await asyncio.sleep(0)
        finally:
            release.set()
        return await task

    try:
        assert asyncio.run(invoke()) == "saved"
    finally:
        release.set()
    assert callback_threads == [loop_thread]


def test_async_stage_storage_wait_is_cancellable_without_waiting_for_blocked_thread(
    firestore_fake, fake_clock,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="cancel-worker", now=NOW, lease_seconds=60)
    assert lease is not None
    governor = _governor(client, fake_clock)
    entered, release = threading.Event(), threading.Event()
    original_call = governor._store_call

    def blocked_call(name, fallback, *args, **kwargs):
        if name == "validate_lease":
            entered.set()
            if not release.wait(5):
                raise TimeoutError("lease validation release timed out")
        return original_call(name, fallback, *args, **kwargs)

    governor._store_call = blocked_call

    async def invoke():
        task = asyncio.create_task(governor.run_stage_async(
            OPERATION_ID, "cancel-storage", lambda: pytest.fail("callback must not run"),
            owner="cancel-worker", fence=lease.fence, now=NOW,
        ))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            release.set()

    asyncio.run(invoke())


@pytest.mark.parametrize("async_stage", [False, True])
@pytest.mark.parametrize("mode,expected_failures", [("permanent", 1), ("transient", 3)])
def test_terminal_stage_replay_settles_operation_without_new_attempt(
    firestore_fake, fake_clock, monkeypatch, async_stage, mode, expected_failures,
):
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    lease = claim_operation(client, OPERATION_ID, owner="first-owner", now=NOW, lease_seconds=60)
    assert lease is not None
    governor = _governor(client, fake_clock)
    original_commit = FakeTransaction._commit
    interrupted = {"done": False}
    operation_path = f"dialog_operations/{OPERATION_ID}"

    def fail_terminal_settlement_commit(transaction):
        if (not interrupted["done"] and any(
            ref.path == operation_path and isinstance(data, dict)
            and data.get("state") == "failed"
            for _operation, ref, data, _merge in transaction._writes
        )):
            interrupted["done"] = True
            raise OSError("synthetic fault before terminal operation settlement commit")
        return original_commit(transaction)

    monkeypatch.setattr(FakeTransaction, "_commit", fail_terminal_settlement_commit)
    calls = []

    def provider_failure():
        calls.append("provider")
        raise ValueError("permanent provider rejection")

    for attempt in range(expected_failures):
        terminal_attempt = attempt == expected_failures - 1
        if async_stage:
            async def initial():
                expected_error = OSError if terminal_attempt else ValueError
                with pytest.raises(expected_error):
                    await governor.run_stage_async(
                        OPERATION_ID, "terminal-replay", provider_failure,
                        owner="first-owner", fence=lease.fence, now=fake_clock.now(),
                        retryable=lambda _error: mode == "transient",
                    )
            asyncio.run(initial())
        else:
            expected_error = OSError if terminal_attempt else ValueError
            with pytest.raises(expected_error):
                governor.run_stage(
                    OPERATION_ID, "terminal-replay", provider_failure,
                    owner="first-owner", fence=lease.fence, now=fake_clock.now(),
                    retryable=lambda _error: mode == "transient",
                )

    stage = get_stage(client, OPERATION_ID, "terminal-replay")
    assert stage.failures == stage.attempts_started == expected_failures
    assert get_operation(client, OPERATION_ID).state is OperationState.RUNNING
    fake_clock.advance(seconds=61)

    def forbidden_retry():
        calls.append("retry")

    replay_lease = claim_operation(
        client, OPERATION_ID, owner="replay-owner", now=fake_clock.now(), lease_seconds=60,
    )
    assert replay_lease is not None
    if async_stage:
        async def replay():
            with pytest.raises(RetryBudgetExceeded):
                await governor.run_stage_async(
                    OPERATION_ID, "terminal-replay", forbidden_retry,
                    owner="replay-owner", fence=replay_lease.fence, now=fake_clock.now(),
                )
        asyncio.run(replay())
    else:
        with pytest.raises(RetryBudgetExceeded):
            governor.run_stage(
                OPERATION_ID, "terminal-replay", forbidden_retry, owner="replay-owner",
                fence=replay_lease.fence, now=fake_clock.now(),
            )

    assert calls == ["provider"] * expected_failures
    stage = get_stage(client, OPERATION_ID, "terminal-replay")
    assert stage.failures == stage.attempts_started == expected_failures
    assert get_operation(client, OPERATION_ID).state is OperationState.FAILED


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


@pytest.mark.parametrize(
    ("reason", "stage_key", "category", "service"),
    [
        ("billing_disabled", "telegram.finalize.drive", "billing", "google_drive"),
        ("quota_exceeded", "telegram.finalize.docs", "quota", "google_docs"),
        (None, "telegram.finalize.docs", "rate_limited", "google_docs"),
    ],
)
def test_structured_provider_errors_are_bounded_and_never_store_provider_text(
    firestore_fake, fake_clock, reason, stage_key, category, service,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)

    class ProviderFailure(RuntimeError):
        status_code = 429

        def __init__(self, reason):
            super().__init__("synthetic-private-provider-body-secret")
            self.response_json = {"error": {"reason": reason}} if reason else None

    error = ProviderFailure(reason)
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier
    classified_governor = RetryGovernor(
        client, clock=fake_clock, max_failures=3,
        error_classifier=telegram_error_classifier,
    )
    with pytest.raises(ProviderFailure):
        classified_governor.run_stage(
            OPERATION_ID, stage_key, lambda: (_ for _ in ()).throw(error),
            now=fake_clock.now(),
        )

    stage = get_stage(client, OPERATION_ID, stage_key)
    assert stage.failures == 1
    assert stage.last_error.category == category
    assert stage.last_error.service == service
    assert stage.last_error.stage == stage_key
    stored = client.collection("dialog_operations").document(OPERATION_ID).collection(
        "stages").document(stage_key).get().to_dict()
    assert "synthetic-private-provider-body-secret" not in repr(stored)


@pytest.mark.parametrize(
    ("code", "envelope", "category"),
    [
        (403, {"error": {"code": 403, "status": "PERMISSION_DENIED",
                          "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                       "reason": "BILLING_DISABLED"}]}}, "billing"),
        (402, {"error": {"code": 402, "status": "RESOURCE_EXHAUSTED",
                          "message": "private-marker"}}, "billing"),
        (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                          "message": "private-marker"}}, "rate_limited"),
    ],
)
def test_installed_genai_client_error_uses_machine_fields_only(
    firestore_fake, fake_clock, code, envelope, category,
):
    from types import SimpleNamespace
    from google.genai.errors import ClientError
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    error = ClientError(code, envelope)
    assert telegram_error_classifier(error, "telegram-tool-00") == (category, "gemini")
    assert telegram_error_classifier(RuntimeError("unknown"), "telegram-tool-00") == (
        "execution", None,
    )
    assert telegram_error_classifier(SimpleNamespace(code=402), "telegram-tool-00") == (
        "execution", None,
    )


def test_unconfigured_governor_keeps_actual_genai_402_generic(firestore_fake, fake_clock):
    from google.genai.errors import ClientError

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    error = ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})
    with pytest.raises(ClientError):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "legacy-genai-402", lambda: (_ for _ in ()).throw(error),
            now=fake_clock.now(),
        )
    stage = get_stage(client, OPERATION_ID, "legacy-genai-402")
    assert stage.last_error.category == "execution"
    assert stage.last_error.service is None
    assert "private-marker" not in repr(stage.last_error)


def test_unconfigured_governor_preserves_legacy_generic_error_fields(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    with pytest.raises(RuntimeError):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "legacy-stage", lambda: (_ for _ in ()).throw(
                RuntimeError("synthetic billing_disabled body text")), now=fake_clock.now(),
        )
    stage = get_stage(client, OPERATION_ID, "legacy-stage")
    assert stage.last_error.category == "execution"
    assert stage.last_error.service is None


def test_single_attempt_firestore_projection_tags_actual_sdk_failure_but_legacy_does_not(
    firestore_fake, fake_clock, monkeypatch,
):
    from google.api_core.exceptions import ServiceUnavailable
    from tests.fixtures.firestore_fake import FakeQuery
    from core.operations import read
    from core.utils.retry import ProviderBoundaryError, SINGLE_ATTEMPT_RETRY_POLICY
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    secret = "synthetic-firestore-secret-and-source"
    provider_failure = ServiceUnavailable(secret)
    calls = []

    def fail_stream(self, **kwargs):
        calls.append(dict(kwargs))
        raise provider_failure

    monkeypatch.setattr(FakeQuery, "stream", fail_stream)
    governed = RetryGovernor(
        client, clock=fake_clock, error_classifier=telegram_error_classifier,
    )
    with pytest.raises(ProviderBoundaryError) as raised:
        governed.run_stage(
            OPERATION_ID, "telegram-firestore-read",
            lambda: read.list_folders(retry_policy=SINGLE_ATTEMPT_RETRY_POLICY),
            now=fake_clock.now(),
        )
    stage = get_stage(client, OPERATION_ID, "telegram-firestore-read")
    assert calls == [{"retry": None}]
    assert raised.value.original_error is provider_failure
    assert stage.last_error.service == "firestore"
    assert secret not in repr(stage)

    legacy_client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(legacy_client)
    calls.clear()
    with pytest.raises(ServiceUnavailable) as legacy_error:
        _governor(legacy_client, fake_clock).run_stage(
            OPERATION_ID, "legacy-firestore-read", read.list_folders,
            now=fake_clock.now(),
        )
    legacy = get_stage(legacy_client, OPERATION_ID, "legacy-firestore-read")
    assert calls == [{}]
    assert legacy_error.value is provider_failure
    assert legacy.last_error.service is None


def test_single_attempt_speaker_folder_read_marks_lazy_stream_failure(
    firestore_fake, fake_clock, monkeypatch,
):
    from google.api_core.exceptions import ServiceUnavailable
    from tests.fixtures.firestore_fake import FakeQuery
    from core.integrations.firestore.folders import get_folders_by_speaker
    from core.utils.retry import ProviderBoundaryError
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    _new_operation(client)
    secret = "synthetic-speaker-stream-secret"
    provider_failure = ServiceUnavailable(secret)
    calls = []

    def lazy_failure(self, **kwargs):
        calls.append(dict(kwargs))
        def rows():
            raise provider_failure
            yield None
        return rows()

    monkeypatch.setattr(FakeQuery, "stream", lazy_failure)
    governed = RetryGovernor(
        client, clock=fake_clock, error_classifier=telegram_error_classifier,
    )
    with pytest.raises(ProviderBoundaryError) as raised:
        governed.run_stage(
            OPERATION_ID, "telegram-speaker-folders",
            lambda: get_folders_by_speaker("Ada", client, single_attempt=True),
            now=fake_clock.now(),
        )
    stage = get_stage(client, OPERATION_ID, "telegram-speaker-folders")
    assert calls == [{"retry": None}]
    assert raised.value.original_error is provider_failure
    assert stage.last_error.service == "firestore"
    assert secret not in repr(stage)


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

    from core.utils.retry import ProviderBoundaryError
    for _ in range(3):
        with pytest.raises(ProviderBoundaryError) as raised:
            _governor(client, fake_clock).run_stage(
                OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
            )
        assert raised.value.original_error is not None
    assert physical_attempts == ["attempt"] * 3
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "scroll"]) == 3

    with pytest.raises(RetryBudgetExceeded):
        _governor(client, fake_clock).run_stage(
            OPERATION_ID, "provider", one_provider_call, now=fake_clock.now(),
        )
    assert physical_attempts == ["attempt"] * 3
