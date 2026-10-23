"""Focused tests for the Telegram single-head queue worker."""

import asyncio
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
    task_payload,
)
from core.integrations.firestore.operation_queue import (
    claim_head,
    fail_head,
    get_operation,
    register_operation,
    release_head,
)
from core.integrations.firestore.operations import (
    claim_operation,
    complete_stage_attempt,
    create_operation,
    get_stage,
    reserve_stage_attempt,
)
from core.operations.queue import QueueOperationState
from core.operations.models import OperationOrigin
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.queue import TelegramQueueWorker
from services.connectors.telegram.repository import SESSIONS, TelegramRepository


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


def test_registration_assigns_contiguous_sequences_and_reserves_only_the_head(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, _ = _setup(client, operation_ids=("op-a", "op-b", "op-c"))

    context = context_ref.get().to_dict()
    operations = [get_operation(context_ref, operation_id) for operation_id in ("op-a", "op-b", "op-c")]

    assert [operation.sequence for operation in operations] == [1, 2, 3]
    assert context["next_operation_sequence"] == 4
    assert context["head_sequence"] == 1
    assert context["reserved_operation_id"] == "op-a"
    assert context["queue_tail_operation_id"] == "op-c"
    assert [
        firestore_fake.documents[
            f"telegram_sessions/ctx/operations/{operation_id}"
        ]["next_operation_id"]
        for operation_id in ("op-a", "op-b", "op-c")
    ] == ["op-b", "op-c", None]


def test_stale_fence_cannot_release_successor_lease(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, _ = _setup(client)
    stale = claim_head(client, context_ref, owner="old-worker", now=NOW)
    assert stale is not None

    successor_now = NOW + timedelta(seconds=121)
    successor = claim_head(client, context_ref, owner="new-worker", now=successor_now)
    assert successor is not None
    assert successor.fence > stale.fence

    assert release_head(
        client, context_ref, operation_id=stale.operation_id,
        owner=stale.owner, fence=stale.fence, now=successor_now,
    ) is False
    context = context_ref.get().to_dict()
    assert context["active_operation_id"] == successor.operation_id
    assert context["lock_owner"] == successor.owner
    assert context["fence"] == successor.fence


def test_stale_fence_cannot_write_stage_checkpoint_after_reclaim(firestore_fake):
    client = firestore_fake.client_class()
    operation_id = "telegram-checkpoint-fence"
    create_operation(
        client,
        operation_id=operation_id,
        origin=OperationOrigin.TELEGRAM,
        caller_binding="telegram:ctx",
        immutable_input_digest="digest:telegram-checkpoint-fence",
        immutable_input_ref="telegram_sessions/ctx/commands/1",
        now=NOW,
    )
    old_lease = claim_operation(
        client, operation_id, owner="old-worker", now=NOW, lease_seconds=5,
    )
    assert old_lease is not None
    stage_key = "save/index"
    reserved = reserve_stage_attempt(
        client, operation_id, stage_key, owner="old-worker",
        fence=old_lease.fence, attempt_id="old-attempt", now=NOW,
    )
    assert reserved is not None

    takeover_time = NOW + timedelta(seconds=6)
    successor = claim_operation(
        client, operation_id, owner="new-worker", now=takeover_time, lease_seconds=30,
    )
    assert successor is not None and successor.fence > old_lease.fence

    assert complete_stage_attempt(
        client, operation_id, stage_key, owner="old-worker",
        fence=old_lease.fence, attempt_id="old-attempt",
        result={"document_id": "stale-doc"}, checkpoint_refs=("stale-ref",),
        now=takeover_time,
    ) is False
    stage = get_stage(client, operation_id, stage_key)
    assert stage.result is None
    assert stage.checkpoint_refs == ()


def test_command_acceptance_persists_predecessor_without_classifying_save_text(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.set({"last_doc_id": "old-doc"})

    save = repository.accept_command(
        100,
        chat_id=42,
        user_id=7,
        command=parse_command("save"),
        now=NOW,
    )
    dependent = repository.accept_command(
        101,
        chat_id=42,
        user_id=7,
        command=parse_command("move last"),
        now=NOW + timedelta(seconds=1),
    )

    assert save.operation_id is not None
    assert dependent.operation_id is not None
    save_lease = claim_head(
        client, context_ref, owner="save-worker", now=NOW + timedelta(seconds=1),
    )
    assert save_lease is not None
    assert save_lease.operation_id == save.operation_id
    assert fail_head(
        client, context_ref, operation_id=save_lease.operation_id,
        owner=save_lease.owner, fence=save_lease.fence,
        now=NOW + timedelta(seconds=2),
    ) is True
    assert context_ref.get().to_dict()["reserved_operation_id"] == dependent.operation_id
    assert get_operation(context_ref, save.operation_id).state is QueueOperationState.FAILED

    dependent_operation = firestore_fake.documents[
        f"{SESSIONS}/{context_id}/operations/{dependent.operation_id}"
    ]
    assert dependent_operation.get("predecessor_operation_id") == save.operation_id
    assert "pending_save_candidate_operation_id" not in dependent_operation
    assert "dependency_operation_id" not in dependent_operation
    assert dependent_operation.get("target_doc_id") is None


@pytest.mark.parametrize("candidate", ["move last", "Перенеси последний", "enlace por favor"])
def test_arbitrary_command_text_preserves_only_queue_lineage_without_classifying_intent(
    firestore_fake, candidate,
):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.set({"last_doc_id": "old-doc"})

    save = repository.accept_command(
        110, chat_id=42, user_id=7, command=parse_command("save"), now=NOW,
    )
    accepted = repository.accept_command(
        111, chat_id=42, user_id=7, command=parse_command(candidate),
        now=NOW + timedelta(seconds=1),
    )

    commands = client.collection(SESSIONS).document(context_id).collection("commands")
    command_data = commands.document("111").get().to_dict()
    operation = client.collection(SESSIONS).document(context_id).collection(
        "operations"
    ).document(accepted.operation_id).get().to_dict()
    assert command_data["candidate"] == candidate
    assert command_data["predecessor_operation_id"] == save.operation_id
    assert operation["predecessor_operation_id"] == save.operation_id
    assert "pending_save_candidate_operation_id" not in command_data
    assert "pending_save_candidate_operation_id" not in operation
    assert "dependency_operation_id" not in operation
    assert operation.get("target_doc_id") is None


def test_all_command_candidates_persist_predecessor_without_save_classification(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"

    candidate = repository.accept_command(
        125, chat_id=42, user_id=7, command=parse_command("guardar"), now=NOW,
    )
    following = repository.accept_command(
        126, chat_id=42, user_id=7, command=parse_command("move last"),
        now=NOW + timedelta(seconds=1),
    )

    context_ref = client.collection(SESSIONS).document(context_id)
    operation = context_ref.collection("operations").document(
        following.operation_id
    ).get().to_dict()
    save_candidate_command = context_ref.collection("commands").document(
        "125"
    ).get().to_dict()
    command = context_ref.collection("commands").document("126").get().to_dict()
    assert candidate.operation_id is not None
    assert save_candidate_command["kind"] == "freeform"
    assert command["kind"] == "freeform"
    assert operation["predecessor_operation_id"] == candidate.operation_id
    assert command["predecessor_operation_id"] == candidate.operation_id
    assert "pending_save_candidate_operation_id" not in operation
    assert "dependency_operation_id" not in operation


def test_independent_command_keeps_candidate_without_dependency(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"

    save = repository.accept_command(
        130, chat_id=42, user_id=7, command=parse_command("save"), now=NOW,
    )
    unrelated = repository.accept_command(
        131, chat_id=42, user_id=7, command=parse_command("summarize this"),
        now=NOW + timedelta(seconds=1),
    )
    later = repository.accept_command(
        132, chat_id=42, user_id=7, command=parse_command("enlace please"),
        now=NOW + timedelta(seconds=2),
    )

    context_ref = client.collection(SESSIONS).document(context_id)
    operation = context_ref.collection(
        "operations"
    ).document(unrelated.operation_id).get().to_dict()
    later_operation = context_ref.collection("operations").document(
        later.operation_id
    ).get().to_dict()
    later_command = context_ref.collection("commands").document("132").get().to_dict()
    assert operation["predecessor_operation_id"] == save.operation_id
    assert "dependency_operation_id" not in operation
    assert "target_doc_id" not in operation
    assert later_command["candidate"] == "enlace please"
    assert "pending_save_candidate_operation_id" not in later_command
    assert later_command["predecessor_operation_id"] == unrelated.operation_id
    assert "pending_save_candidate_operation_id" not in later_operation
    assert later_operation["predecessor_operation_id"] == unrelated.operation_id
    assert "dependency_operation_id" not in later_operation
    assert "target_doc_id" not in later_operation


def test_confirmed_save_settlement_clears_hint_and_keeps_child_snapshot(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.set({"last_doc_id": "old-doc"})
    source = repository.accept_forward(
        139, chat_id=42, message={"message_id": 239, "text": "saved source"}, now=NOW,
    )
    save = repository.accept_command(
        140, chat_id=42, user_id=7, command=parse_command("save"), now=NOW,
    )
    save_lease = claim_head(client, context_ref, owner="save-confirm", now=NOW)
    assert save_lease is not None
    assert repository.confirm_save_command(
        context_id, save.operation_id, owner=save_lease.owner,
        fence=save_lease.fence, now=NOW,
    ) is True
    dependent = repository.accept_command(
        141, chat_id=42, user_id=7, command=parse_command("Перенеси последний"),
        now=NOW + timedelta(seconds=1),
    )
    dependent_operation = context_ref.collection("operations").document(
        dependent.operation_id
    ).get().to_dict()
    accepted_command = context_ref.collection("commands").document("141").get().to_dict()
    assert dependent_operation["pending_save_candidate_operation_id"] == save.operation_id
    assert accepted_command["pending_save_candidate_operation_id"] == save.operation_id
    assert dependent_operation["predecessor_operation_id"] == save.operation_id
    assert accepted_command["draft_id"] is None
    assert accepted_command["cutoff_message_ids"] == []
    assert source.draft_id == save.draft_id
    assert release_head(
        client, context_ref, operation_id=save_lease.operation_id,
        owner=save_lease.owner, fence=save_lease.fence, now=NOW + timedelta(seconds=1),
    ) is True

    async def process_save(context_id, operation, lease):
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(
        client, process_save, clock=lambda: NOW + timedelta(seconds=2),
        owner_factory=lambda: "save-worker",
    )
    save_payload = task_payload(get_outbox_record(client, save.outbox_id))
    assert _run(worker.handle_task(save_payload)) is True

    settled_context = context_ref.get().to_dict()
    assert "pending_save_operation_id" not in settled_context
    assert settled_context["last_doc_id"] == "old-doc"
    assert settled_context["reserved_operation_id"] == dependent.operation_id
    dependent_operation = context_ref.collection("operations").document(
        dependent.operation_id
    ).get().to_dict()
    assert dependent_operation["predecessor_operation_id"] == save.operation_id
    assert dependent_operation["pending_save_candidate_operation_id"] == save.operation_id
    assert context_ref.collection("commands").document("141").get().to_dict() == accepted_command
    assert any(
        row.get("operation_id") == dependent.operation_id
        and outbox_id != dependent.outbox_id
        for outbox_id, row in _rows(firestore_fake).items()
    )

    later = repository.accept_command(
        142, chat_id=42, user_id=7, command=parse_command("Перенеси последний"),
        now=NOW + timedelta(seconds=3),
    )
    later_operation = client.collection(SESSIONS).document(context_id).collection(
        "operations"
    ).document(later.operation_id).get().to_dict()
    assert "pending_save_candidate_operation_id" not in later_operation
    assert "dependency_operation_id" not in later_operation


def test_command_parser_preserves_arbitrary_text_without_reference_classification():
    for text in ("move last", "Перенеси последний", "enlace por favor"):
        command = parse_command(text)
        assert command.kind == "freeform"
        assert command.text == text


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
