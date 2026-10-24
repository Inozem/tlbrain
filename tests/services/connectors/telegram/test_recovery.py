"""Telegram-specific recovery across queue, stage, and task boundaries."""

from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxState,
    OutboxTargetKind,
    get_outbox_record,
    record_outbox_delivery_result,
    task_payload,
)
from core.integrations.firestore.operations import create_operation, get_stage
from core.integrations.firestore.operation_queue import (
    QueueOperationRetryStore,
    claim_head,
    get_operation,
)
from core.operations.models import OperationOrigin
from core.operations.outbox import TaskTargetConfig, publish_due_outbox_records
from core.operations.queue import QueueOperationState
from core.operations.retries import (
    ConfirmedStageSuccess,
    RetryBudgetExceeded,
    RetryGovernor,
    UnknownOutcome,
)
from core.operations import outbox as outbox_publisher
from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY, with_retry
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.queue import TelegramQueueWorker
from services.connectors.telegram.repository import SESSIONS, TelegramRepository


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
TARGET = TaskTargetConfig(
    queue_name="telegram",
    url="https://telegram-worker.example.invalid/dispatch",
    service_account_email="worker@example.invalid",
    audience="https://telegram-worker.example.invalid",
    dispatch_deadline=timedelta(seconds=45),
)


class _Clock:
    def __init__(self, current=NOW):
        self.current = current

    def now(self):
        return self.current


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _selector_loop():
    """Use loopback datagrams for asyncio's Windows self-pipe under the socket guard."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    old_socketpair = socket.socketpair
    local_pairs = []

    def local_datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            local_pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_datagram_pair
    try:
        return asyncio.SelectorEventLoop()
    except BaseException:
        for reader, writer in local_pairs:
            reader.close()
            writer.close()
        raise
    finally:
        socket.socketpair = old_socketpair


def _accepted_save(client, update_id=700):
    return TelegramRepository(client=client).accept_command(
        update_id,
        chat_id=42,
        user_id=7,
        command=parse_command("save"),
        now=NOW,
    )


def _stage_operation(client, operation_id):
    create_operation(
        client,
        operation_id=operation_id,
        origin=OperationOrigin.TELEGRAM,
        caller_binding="telegram:42_default",
        immutable_input_digest=f"digest:{operation_id}",
        immutable_input_ref=f"telegram-command:{operation_id}",
        now=NOW,
    )


def test_worker_reconciles_unknown_in_flight_stage_after_crash_without_second_provider_call(
    firestore_fake,
):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client)
    _stage_operation(client, accepted.operation_id)
    provider_calls = []
    reconcile_calls = []
    processor_calls = []
    clock = _Clock()

    def provider_call():
        provider_calls.append(accepted.operation_id)
        raise UnknownOutcome("synthetic lost provider response")

    def processor(context_id, operation, lease):
        processor_calls.append(operation.operation_id)
        governor = RetryGovernor(client, clock=clock)
        if len(processor_calls) == 1:
            return governor.run_stage(
                accepted.operation_id, "save/index", provider_call, now=NOW,
            )
        governor.run_stage(
            accepted.operation_id,
            "save/index",
            provider_call,
            reconcile=lambda: reconcile_calls.append((context_id, operation.operation_id))
            or ConfirmedStageSuccess({"document_id": "synthetic-doc"}),
            now=NOW + timedelta(seconds=1),
        )
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(
        client,
        async_processor(processor),
        clock=lambda: NOW,
        owner_factory=iter(("crashed-worker", "recovery-worker")).__next__,
    )
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))

    assert _run(worker.handle_task(payload)) is False

    stage_before_recovery = get_stage(client, accepted.operation_id, "save/index")
    assert stage_before_recovery.in_flight_attempt_id is not None
    assert stage_before_recovery.attempts_started == 1
    clock.current = NOW + timedelta(seconds=1)
    assert _run(worker.handle_task(payload)) is True

    stage = get_stage(client, accepted.operation_id, "save/index")
    context = client.collection(SESSIONS).document(accepted.context_id).get().to_dict()
    assert provider_calls == [accepted.operation_id]
    assert reconcile_calls == [(accepted.context_id, accepted.operation_id)]
    assert stage.attempts_started == 1 and stage.failures == 0
    assert stage.result == {"document_id": "synthetic-doc"}
    assert context["reserved_operation_id"] is None

    assert _run(worker.handle_task(payload)) is False
    assert provider_calls == [accepted.operation_id]
    assert reconcile_calls == [(accepted.context_id, accepted.operation_id)]
    assert processor_calls == [accepted.operation_id, accepted.operation_id]
    assert get_operation(
        client.collection(SESSIONS).document(accepted.context_id),
        accepted.operation_id,
    ).state is QueueOperationState.COMPLETED
    assert client.collection(SESSIONS).document(accepted.context_id).get().to_dict()[
        "reserved_operation_id"
    ] is None


def async_processor(processor):
    async def invoke(context_id, operation, lease):
        return processor(context_id, operation, lease)

    return invoke


def test_nonterminal_slice_yield_keeps_same_head_for_next_delivery(firestore_fake):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=701)
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))
    outcomes = iter((QueueOperationState.RUNNING, QueueOperationState.COMPLETED))

    async def processor(context_id, operation, lease):
        return next(outcomes)

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    context_ref = client.collection(SESSIONS).document(accepted.context_id)

    assert _run(worker.handle_task(payload)) is False
    after_yield = context_ref.get().to_dict()
    assert after_yield["reserved_operation_id"] == accepted.operation_id
    assert after_yield["active_operation_id"] is None
    assert after_yield["lock_owner"] is None

    assert _run(worker.handle_task(payload)) is True
    assert context_ref.get().to_dict()["reserved_operation_id"] is None


def test_queue_retry_store_reconciles_in_flight_before_repeating_provider(firestore_fake):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=706)
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))
    provider_calls = []
    reconcile_calls = []
    processor_calls = []

    async def processor(context_id, operation, lease, *, retry_governor):
        processor_calls.append(1)
        async def provider():
            provider_calls.append(1)
            raise UnknownOutcome("provider response was lost")
        async def reconcile():
            reconcile_calls.append(1)
            return ConfirmedStageSuccess({"transcript": "checkpointed"})
        result = await retry_governor.run_stage_async(
            operation.operation_id, "telegram.audio.1", provider,
            owner=lease.owner, fence=lease.fence,
            reconcile=reconcile if len(processor_calls) > 1 else None,
        )
        assert result == {"transcript": "checkpointed"}
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payload)) is False
    assert _run(worker.handle_task(payload)) is True
    assert provider_calls == [1]
    assert reconcile_calls == [1]
    stage = next(data for path, data in firestore_fake.documents.items()
                 if path.endswith("/stages/telegram.audio.1"))
    assert stage["attempts_started"] == 1
    assert stage["failures"] == 0
    assert stage["result"] == {"transcript": "checkpointed"}


def test_timed_slice_cancels_processor_and_persists_same_head_wakeup(firestore_fake):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=705)
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    cancelled = []

    async def slow_processor(context_id, operation, lease):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append((context_id, operation.operation_id))

    worker = TelegramQueueWorker(
        client, slow_processor, clock=lambda: NOW,
        heartbeat_interval_seconds=0.005, max_slice_seconds=0.02,
    )

    assert _run(worker.handle_task(payload)) is False
    context = context_ref.get().to_dict()
    assert context["reserved_operation_id"] == accepted.operation_id
    assert context["active_operation_id"] is None
    assert context["lock_owner"] is None
    assert cancelled == [(accepted.context_id, accepted.operation_id)]
    assert get_operation(context_ref, accepted.operation_id).state is QueueOperationState.RUNNING

    continuations = [
        (path.rsplit("/", 1)[-1], data) for path, data in firestore_fake.documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
        and path.rsplit("/", 1)[-1] != accepted.outbox_id
    ]
    assert len(continuations) == 1
    continuation_id, continuation_data = continuations[0]
    assert continuation_data["target_kind"] == OutboxTargetKind.TELEGRAM_DISPATCH.value
    assert continuation_data["generation"] == context["telegram_dispatch_wake_generation"]
    assert continuation_data["state"] == OutboxState.PENDING.value
    continuation_payload = task_payload(get_outbox_record(client, continuation_id))

    async def complete(context_id, operation, lease):
        assert operation.operation_id == accepted.operation_id
        return QueueOperationState.COMPLETED

    assert _run(
        TelegramQueueWorker(client, complete, clock=lambda: NOW).handle_task(continuation_payload)
    )
    assert context_ref.get().to_dict()["reserved_operation_id"] is None


def test_telegram_acceptance_repairs_firestore_to_tasks_gap_and_dispatches_same_operation(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=702)
    enqueue_calls = []

    def enqueue(**kwargs):
        enqueue_calls.append(kwargs)
        if len(enqueue_calls) == 1:
            # Cloud Tasks accepted the deterministic name, but Firestore saw no response.
            raise RuntimeError("synthetic lost enqueue response")
        return False  # The same named task already exists.

    monkeypatch.setattr(outbox_publisher, "enqueue_task", enqueue)
    first = publish_due_outbox_records(
        client, {OutboxTargetKind.TELEGRAM_DISPATCH: TARGET}, now=NOW, limit=5,
    )
    assert first.failed == 1
    assert get_outbox_record(client, accepted.outbox_id).state is OutboxState.RETRY

    repaired = publish_due_outbox_records(
        client,
        {OutboxTargetKind.TELEGRAM_DISPATCH: TARGET},
        now=NOW + timedelta(seconds=1),
        limit=5,
    )
    record = get_outbox_record(client, accepted.outbox_id)
    assert repaired.already_exists == 1
    assert record.state is OutboxState.DELIVERED
    assert [call["task_id"] for call in enqueue_calls] == [record.task_name, record.task_name]
    assert enqueue_calls[0]["body"]["operation_id"] == accepted.operation_id
    assert enqueue_calls[1]["body"] == enqueue_calls[0]["body"]

    processed = []

    async def processor(context_id, operation, lease):
        processed.append((context_id, operation.operation_id))
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(task_payload(record))) is True
    assert processed == [(accepted.context_id, accepted.operation_id)]


def test_delivered_wakeup_tombstone_is_replaced_with_new_generation_for_reserved_head(
    firestore_fake,
):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=703)
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    async def processor(context_id, operation, lease):
        raise AssertionError("a stale task generation must only repair the head wakeup")

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    stale_delivery = {**task_payload(get_outbox_record(client, accepted.outbox_id)), "generation": 2}
    assert _run(worker.handle_task(stale_delivery)) is False

    first_repair = next(
        data for path, data in firestore_fake.documents.items()
        if path.startswith("task_outbox/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
        and path.rsplit("/", 1)[-1] != accepted.outbox_id
    )
    assert first_repair["generation"] == 1
    first_repair_id = next(
        path.rsplit("/", 1)[-1] for path, data in firestore_fake.documents.items()
        if path.startswith("task_outbox/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
        and path.rsplit("/", 1)[-1] != accepted.outbox_id
    )
    record_outbox_delivery_result(client, first_repair_id, delivered=True, now=NOW)
    assert _run(worker.handle_task(stale_delivery)) is False

    repairs = [
        (path.rsplit("/", 1)[-1], data)
        for path, data in firestore_fake.documents.items()
        if path.startswith("task_outbox/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
        and path.rsplit("/", 1)[-1] != accepted.outbox_id
    ]
    assert len(repairs) == 2
    repair_id, repair = next((item for item in repairs if item[1]["generation"] == 2))
    assert repair["target_kind"] == OutboxTargetKind.TELEGRAM_DISPATCH.value
    assert repair["generation"] == 2
    assert repair["state"] == OutboxState.PENDING.value
    assert repair["operation_id"] == accepted.operation_id
    assert repair_id.startswith("tgqueue-")
    assert context_ref.get().to_dict()["reserved_operation_id"] == accepted.operation_id


def test_telegram_stage_failure_budget_is_three_physical_single_attempt_calls(firestore_fake):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=704)
    _stage_operation(client, accepted.operation_id)
    clock = _Clock()
    governor = RetryGovernor(client, clock=clock, max_failures=3)
    physical_calls = []

    @with_retry
    def provider_call():
        physical_calls.append(accepted.operation_id)
        raise ConnectionError("synthetic provider outage")

    for failure_number in range(1, 4):
        clock.current = NOW + timedelta(seconds=failure_number)
        with pytest.raises(ConnectionError):
            governor.run_stage(
                accepted.operation_id,
                "save/index",
                lambda: provider_call(retry_policy=SINGLE_ATTEMPT_RETRY_POLICY),
                now=NOW + timedelta(seconds=failure_number),
            )
        assert get_stage(client, accepted.operation_id, "save/index").failures == failure_number

    clock.current = NOW + timedelta(seconds=4)
    with pytest.raises(RetryBudgetExceeded):
        governor.run_stage(
            accepted.operation_id,
            "save/index",
            lambda: provider_call(retry_policy=SINGLE_ATTEMPT_RETRY_POLICY),
            now=NOW + timedelta(seconds=4),
        )

    assert physical_calls == [accepted.operation_id] * 3
    assert get_stage(client, accepted.operation_id, "save/index").attempts_started == 3


def test_stage_retry_store_updates_stage_without_rewriting_context_queue_document(firestore_fake):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=705)
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="stage-worker", now=NOW)
    assert lease is not None
    before = context_ref.get().to_dict()
    governor = RetryGovernor(
        client,
        clock=_Clock(),
        retry_store=QueueOperationRetryStore(client, context_ref),
    )

    result = governor.run_stage(
        accepted.operation_id,
        "telegram.analysis",
        lambda: {"title": "synthetic"},
        owner=lease.owner,
        fence=lease.fence,
        now=NOW,
    )

    assert result == {"title": "synthetic"}
    assert context_ref.get().to_dict() == before
    assert governor.get_stage(accepted.operation_id, "telegram.analysis").result == result

