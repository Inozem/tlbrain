"""Focused tests for the Telegram single-head queue worker."""

import asyncio
import os
import socket
from datetime import datetime, timedelta, timezone

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
    task_payload,
)
from core.integrations.firestore.operation_queue import register_operation
from core.operations.queue import QueueOperationState
from services.connectors.telegram.queue import TelegramQueueWorker
from services.connectors.telegram.repository import SESSIONS


NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)


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


def _setup(client, context_id="ctx", operation_ids=("op-a",)):
    context_ref = client.collection(SESSIONS).document(context_id)
    payloads = {}
    for operation_id in operation_ids:
        register_operation(
            client, context_ref, operation_id,
            identity_digest=f"digest:{operation_id}", input_ref=f"input:{operation_id}", now=NOW,
        )
        outbox_id = f"task-{context_id}-{operation_id}"
        record = create_outbox_record(
            client, outbox_id, target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1, schedule_at=NOW, now=NOW,
            context_id=context_id, operation_id=operation_id,
        )
        payloads[operation_id] = task_payload(record)
    return context_ref, payloads


def _rows(firestore_fake):
    return {
        path.rsplit("/", 1)[-1]: data
        for path, data in firestore_fake.documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
    }


def test_out_of_order_delivery_wakes_actual_head_and_does_not_process_tail(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-a", "op-b"))
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    assert _run(worker.handle_task(payloads["op-b"])) is False
    assert calls == []
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-a"
    repairs = {
        outbox_id for outbox_id, row in _rows(firestore_fake).items()
        if row.get("operation_id") == "op-a" and outbox_id != payloads["op-a"]["outbox_id"]
    }
    assert len(repairs) == 1
    assert _run(worker.handle_task(payloads["op-b"])) is False
    assert {
        outbox_id for outbox_id, row in _rows(firestore_fake).items()
        if row.get("operation_id") == "op-a" and outbox_id != payloads["op-a"]["outbox_id"]
    } == repairs

    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert calls == ["op-a"]
    context = context_ref.get().to_dict()
    assert context["reserved_operation_id"] == "op-b"
    assert any(row.get("operation_id") == "op-b" for row in _rows(firestore_fake).values())


def test_stale_generation_and_terminal_replay_do_not_call_processor(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    stale = {**payloads["op-a"], "generation": 2}
    assert _run(worker.handle_task(stale)) is False
    assert calls == []
    assert any(row.get("operation_id") == "op-a" for row in _rows(firestore_fake).values())
    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert _run(worker.handle_task(payloads["op-a"])) is False
    assert calls == ["op-a"]
    assert context_ref.get().to_dict()["reserved_operation_id"] is None


def test_terminal_failure_advances_and_wakes_next_operation(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-a", "op-b"))

    async def processor(context_id, operation, lease):
        return QueueOperationState.FAILED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-b"
    assert any(row.get("operation_id") == "op-b" for row in _rows(firestore_fake).values())


def test_same_context_contention_has_one_live_processor_and_expiry_wakeup(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(lease.fence)
        entered.set()
        await release.wait()
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")

    async def run():
        first = asyncio.create_task(worker.handle_task(payloads["op-a"]))
        await entered.wait()
        second = await worker.handle_task(payloads["op-a"])
        assert second is False
        active = context_ref.get().to_dict()
        assert active["active_operation_id"] == "op-a"
        expected = active["lock_expires_at"]
        assert any(
            row.get("operation_id") == "op-a" and row.get("schedule_at") >= expected
            for row in _rows(firestore_fake).values()
        )
        release.set()
        return await first

    assert _run(run()) is True
    assert len(calls) == 1


def test_different_contexts_process_in_parallel(firestore_fake):
    client = firestore_fake.client_class()
    _, first = _setup(client, "ctx-a")
    _, second = _setup(client, "ctx-b")
    entered = 0
    both_entered = asyncio.Event()

    async def processor(context_id, operation, lease):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)

    async def run():
        return await asyncio.gather(
            worker.handle_task(first["op-a"]), worker.handle_task(second["op-a"]),
        )

    assert _run(run()) == [True, True]
    assert entered == 2


def test_heartbeat_renews_lease_and_stale_fence_cannot_settle(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client)
    now = [NOW]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def processor(context_id, operation, lease):
        entered.set()
        await release.wait()
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(
        client, processor, clock=lambda: now[0], owner_factory=lambda: "worker",
        heartbeat_interval_seconds=0.01,
    )

    async def run():
        task = asyncio.create_task(worker.handle_task(payloads["op-a"]))
        await entered.wait()
        now[0] += timedelta(seconds=30)
        await asyncio.sleep(0.03)
        context = context_ref.get().to_dict()
        assert context["lock_expires_at"] == now[0] + timedelta(seconds=120)
        now[0] += timedelta(seconds=121)
        replacement = __import__(
            "core.integrations.firestore.operation_queue", fromlist=["claim_head"]
        ).claim_head(client, context_ref, owner="replacement", now=now[0])
        assert replacement is not None
        release.set()
        assert await task is False
        return replacement

    replacement = _run(run())
    assert context_ref.get().to_dict()["active_operation_id"] == "op-a"
    from core.integrations.firestore.operation_queue import complete_head
    assert complete_head(
        client, context_ref, operation_id="op-a", owner="replacement",
        fence=replacement.fence, now=now[0],
    )
    assert context_ref.get().to_dict()["reserved_operation_id"] is None


def test_processor_exception_releases_matching_lease_and_outbox_has_only_ids(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client)

    async def processor(context_id, operation, lease):
        raise RuntimeError("temporary")

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    try:
        _run(worker.handle_task(payloads["op-a"]))
    except RuntimeError as exc:
        assert str(exc) == "temporary"
    else:
        raise AssertionError("processor exception should propagate")
    context = context_ref.get().to_dict()
    assert context["reserved_operation_id"] == "op-a"
    assert context.get("active_operation_id") is None
    for row in _rows(firestore_fake).values():
        assert "text" not in row and "command" not in row and "message" not in row
