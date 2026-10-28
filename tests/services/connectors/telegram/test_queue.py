"""Focused tests for the Telegram single-head queue worker."""

import asyncio
import copy
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest
from google.cloud import firestore

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
    task_payload,
)
from core.integrations.firestore.operation_queue import (
    claim_head,
    complete_head,
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
from services.connectors.telegram.repository._commands import _TelegramCommandMixin


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

    continuation_callbacks = []
    assert release_head(
        client, context_ref, operation_id=stale.operation_id,
        owner=stale.owner, fence=stale.fence, now=successor_now,
        after_release=lambda transaction, context: continuation_callbacks.append(True),
    ) is False
    assert continuation_callbacks == []
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


def test_command_aware_processor_receives_only_owned_command_candidate(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    client.collection(SESSIONS).document(context_id).set({"active_draft_id": "draft-private"})
    client.collection(SESSIONS).document(context_id).collection("drafts").document(
        "draft-private"
    ).set({"state": "collecting", "private_transcript": "must never enter command routing"})
    accepted = repository.accept_command(
        990, chat_id=42, user_id=7,
        command=parse_command("show folders and sync status"), now=NOW,
    )
    operation = get_operation(
        client.collection(SESSIONS).document(context_id), accepted.operation_id,
    )
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))
    seen = []

    async def processor(context_id, operation, lease, *, command):
        seen.append((context_id, command.kind, command.text))
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payload)) is True
    assert seen == [(context_id, "freeform", "show folders and sync status")]
    assert "must never enter command routing" not in repr(seen)
    assert operation.input_ref == f"{SESSIONS}/{context_id}/commands/990"


def test_command_candidate_ownership_mismatch_fails_before_command_processor(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    accepted = repository.accept_command(
        991, chat_id=42, user_id=7, command=parse_command("run a tool"), now=NOW,
    )
    command_ref = client.collection(SESSIONS).document(context_id).collection(
        "commands"
    ).document("991")
    command_ref.update({"operation_id": "foreign-operation"})
    operation = get_operation(
        client.collection(SESSIONS).document(context_id), accepted.operation_id,
    )
    calls = []

    async def processor(context_id, operation, lease, *, command):
        calls.append(command)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    with pytest.raises(ValueError, match="candidate is unavailable"):
        worker._load_command_candidate(context_id, operation)
    assert calls == []


def test_cleared_command_candidate_requires_successful_pinned_route(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = "42_default"
    accepted = repository.accept_command(
        992, chat_id=42, user_id=7, command=parse_command("show folders"), now=NOW,
    )
    context_ref = client.collection(SESSIONS).document(context_id)
    operation_ref = context_ref.collection("operations").document(accepted.operation_id)
    command_ref = context_ref.collection("commands").document("992")
    command_ref.update({"candidate": firestore.DELETE_FIELD})
    worker = TelegramQueueWorker(client, lambda *_args: None, clock=lambda: NOW)
    operation = get_operation(context_ref, accepted.operation_id)

    with pytest.raises(ValueError, match="candidate is unavailable"):
        worker._load_command_candidate(context_id, operation)
    operation_ref.update({"route_control": {
        "status": "ok", "kind": "substantive", "target_doc_id": "doc-pinned",
        "conversation_id": "conversation-pinned", "dialog_number": 2,
    }})
    assert worker._load_command_candidate(context_id, operation) is None

    seen = []

    async def processor(_context_id, _operation, _lease, *, command):
        seen.append(command)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    payload = task_payload(get_outbox_record(client, accepted.outbox_id))
    assert _run(worker.handle_task(payload)) is True
    assert seen == [None]

    operation_ref.update({"route_control": {
        "status": "ok", "kind": "substantive", "target_doc_id": "",
        "conversation_id": "conversation-pinned", "dialog_number": 2,
    }})
    # The queue passes pinned records onward; the command processor validates full shape.
    assert worker._load_command_candidate(context_id, operation) is None


def test_route_control_serializer_keeps_validated_choice_and_rejects_malformed_choice():
    route = {
        "status": "ok", "kind": "substantive", "target_doc_id": "doc-a",
        "conversation_id": "conversation-a", "dialog_number": 1,
        "substantive_text": "private user request",
        "tool_choice": {
            "name": "get_transcript", "arguments": {"doc_id": "doc-a"},
            "link_requested": True,
        },
    }
    saved = _TelegramCommandMixin._route_control_data(route)
    assert saved["tool_choice"] == route["tool_choice"]
    assert "substantive_text" not in saved
    assert "raw command" not in saved

    with pytest.raises(ValueError, match="tool_choice is invalid"):
        _TelegramCommandMixin._route_control_data({
            **route,
            "tool_choice": {"name": "get_transcript", "arguments": {"doc_id": "doc-b"}},
        })

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


def test_failed_confirmed_dependency_is_not_executed_and_delivers_one_durable_notice(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-dependent", "op-later"))
    context_ref.update({
        "chat_id": 42, "last_doc_id": "old-doc", "selected_dialog_number": 99,
        "dialog_number": 99,
        "pending_save_operation_id": "op-dependent",
    })
    operations = context_ref.collection("operations")
    operations.document("op-dependent").update({"dependency_operation_id": "op-save"})
    operations.document("op-later").update({"pending_save_candidate_operation_id": "op-save"})
    operations.document("op-dependent").update({
        "input_ref": f"{SESSIONS}/ctx/commands/op-dependent",
        "target_dialog_number": 17,
    })
    context_ref.collection("commands").document("op-dependent").set({
        "operation_id": "op-dependent", "context_id": "ctx", "dialog_number": 42,
    })

    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    assert fail_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                     fence=save_lease.fence, now=NOW)
    processor_calls = []

    async def processor(context_id, operation, lease):
        processor_calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    assert _run(worker.handle_task(payloads["op-dependent"])) is True
    assert processor_calls == []
    assert context_ref.get().to_dict()["last_doc_id"] == "old-doc"
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-later"
    assert "pending_save_operation_id" not in context_ref.get().to_dict()

    operation = operations.document("op-dependent").get().to_dict()
    assert operation["state"] == "failed"
    assert operation["stage"] == "dependency"
    assert operation["status_updated_at"] == NOW
    assert operation["error"] == {
        "service": "telegram", "stage": "dependency", "code": "dependency_unavailable",
        "category": "dependency",
        "message": "The confirmed prerequisite did not complete successfully.",
        "sanitized_message": "The confirmed prerequisite did not complete successfully.",
        "retryable": False, "occurred_at": NOW, "failures": 0,
        "operation_id": "op-dependent",
    }
    notification_rows = [
        data for path, data in firestore_fake.documents.items()
        if path.startswith("notification_outbox/")
    ]
    assert len(notification_rows) == 1
    notification = notification_rows[0]
    assert notification["kind"] == "error"
    assert notification["dialog_number"] == 17
    assert "Dialog #17" in notification["body"]
    assert "telegram/dependency" in notification["body"]
    assert "op-save" in notification["body"]
    assert "op-dependent" in notification["body"]
    assert "Retry the save" in notification["body"]
    assert "Dialog #42" not in notification["body"]
    dispatch_rows = [
        data for data in _rows(firestore_fake).values()
        if data.get("notification_id") == notification["notification_id"]
    ]
    assert len(dispatch_rows) == 1
    assert dispatch_rows[0]["operation_id"] is None
    assert any(row.get("operation_id") == "op-later" for row in _rows(firestore_fake).values())

    assert _run(worker.handle_task(payloads["op-dependent"])) is False
    assert len([path for path in firestore_fake.documents if path.startswith("notification_outbox/")]) == 1
    assert _run(worker.handle_task(payloads["op-later"])) is True
    assert processor_calls == ["op-later"]

    old_parent = operations.document("op-save").get().to_dict()
    old_dependent = operations.document("op-dependent").get().to_dict()
    retry_operation = register_operation(
        client, context_ref, "op-retry", identity_digest="digest:op-retry",
        input_ref="input:op-retry", now=NOW + timedelta(seconds=1),
    )
    operations.document("op-retry").update({
        "draft_id": "draft-1", "save_intent_id": "intent-1", "doc_id": "doc-1",
    })
    retry_record = create_outbox_record(
        client, "task-ctx-op-retry", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=NOW + timedelta(seconds=1), now=NOW + timedelta(seconds=1),
        context_id="ctx", operation_id=retry_operation.operation_id,
    )
    assert _run(worker.handle_task(task_payload(retry_record))) is True
    assert processor_calls == ["op-later", "op-retry"]
    assert operations.document("op-save").get().to_dict() == old_parent
    assert operations.document("op-dependent").get().to_dict() == old_dependent


def test_dependency_notice_is_reused_by_terminal_status_checks(firestore_fake, monkeypatch):
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.status_checks import TelegramStatusChecks

    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-parent", "op-child"))
    context_ref.update({"chat_id": 55})
    operations = context_ref.collection("operations")
    operations.document("op-child").update({"dependency_operation_id": "op-parent"})
    parent_lease = claim_head(client, context_ref, owner="parent", now=NOW)
    assert parent_lease is not None
    assert fail_head(client, context_ref, operation_id="op-parent", owner=parent_lease.owner,
                     fence=parent_lease.fence, now=NOW)
    processor_calls = []

    async def processor(context_id, operation, lease):
        processor_calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    failed_operation = operations.document("op-child").get().to_dict()
    dependency_notification_id = failed_operation["notification_id"]
    assert dependency_notification_id.startswith("tgdep-")

    class FakeTelegram:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, *, message_thread_id=None):
            self.sent.append((chat_id, text, message_thread_id))
            return {"message_id": 913}

        async def edit_message_text(self, chat_id, message_id, text):
            raise AssertionError("the dependency notice should only be sent once")

    repository = TelegramRepository(client)
    telegram = FakeTelegram()
    publisher = TelegramNotificationPublisher(repository, telegram, clock=lambda: NOW)
    checks = TelegramStatusChecks(
        client, status_reader=lambda _operation_id: None,
        notification_publisher=publisher, clock=lambda: NOW + timedelta(seconds=60),
    )
    checks.terminal_notification_updater = lambda _record: None
    check_records = checks.schedule_checks("ctx", "op-child", NOW)
    document_paths_before_delivery = set(firestore_fake.documents)
    async def direct_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", direct_to_thread)

    assert _run(checks.handle_task(task_payload(check_records[0]))) == "failed"
    delivered_notice = repository.get_notification(dependency_notification_id)
    assert delivered_notice["state"] == "delivered"
    assert delivered_notice["send_attempts_started"] == 1
    assert len(telegram.sent) == 1
    assert operations.document("op-child").get().to_dict()["notification_id"] == dependency_notification_id
    assert set(firestore_fake.documents) == document_paths_before_delivery
    delivered_documents = copy.deepcopy(firestore_fake.documents)

    assert _run(checks.handle_task(task_payload(check_records[1]))) == "no_op"
    assert repository.get_notification(dependency_notification_id)["send_attempts_started"] == 1
    assert len(telegram.sent) == 1
    assert processor_calls == []
    assert firestore_fake.documents == delivered_documents


def test_candidate_snapshot_does_not_block_command_but_completed_dependency_needs_fixed_target(
    firestore_fake,
):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-candidate"))
    context_ref.update({"chat_id": 42})
    operations = context_ref.collection("operations")
    operations.document("op-candidate").update({
        "pending_save_candidate_operation_id": "op-save",
    })
    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    assert fail_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                     fence=save_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW, owner_factory=lambda: "worker")
    assert _run(worker.handle_task(payloads["op-candidate"])) is True
    assert calls == ["op-candidate"]

    context_ref, payloads = _setup(client, "ctx-confirmed", operation_ids=("op-parent", "op-child"))
    context_ref.update({"chat_id": 43, "last_doc_id": "newer-selection"})
    operations = context_ref.collection("operations")
    operations.document("op-parent").update({
        "draft_id": "draft-1", "dialog_number": 2,
        "save_intent_id": "intent-retry", "doc_id": "saved-doc",
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-parent", "target_doc_id": "saved-doc",
    })
    client.collection("dialog_operations").document("op-parent").set({
        "operation_id": "op-parent", "state": "completed",
        "save_intent_id": "intent-retry", "doc_id": "saved-doc",
        "result": {"document_id": "saved-doc"},
    })
    parent_lease = claim_head(client, context_ref, owner="parent", now=NOW)
    assert parent_lease is not None
    assert complete_head(client, context_ref, operation_id="op-parent", owner=parent_lease.owner,
                         fence=parent_lease.fence, now=NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == ["op-candidate", "op-child"]


def test_completed_dependency_without_durable_child_target_fails_closed(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-parent", "op-child"))
    context_ref.update({
        "chat_id": 44, "last_doc_id": "mutable-old-target",
        "pending_save_operation_id": "newer-save",
    })
    operations = context_ref.collection("operations")
    operations.document("op-parent").update({"target_doc_id": "saved-doc"})
    operations.document("op-child").update({
        "dependency_operation_id": "op-parent", "target_doc_id": "mutable-old-target",
    })
    parent_lease = claim_head(client, context_ref, owner="parent", now=NOW)
    assert parent_lease is not None
    from core.integrations.firestore.operation_queue import complete_head
    assert complete_head(client, context_ref, operation_id="op-parent", owner=parent_lease.owner,
                         fence=parent_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    assert context_ref.get().to_dict()["last_doc_id"] == "mutable-old-target"
    assert context_ref.get().to_dict()["pending_save_operation_id"] == "newer-save"
    assert operations.document("op-child").get().to_dict()["error"]["code"] == "dependency_unavailable"


def test_original_confirmed_save_link_uses_exact_shared_result(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-child"))
    context_ref.update({"chat_id": 47, "last_doc_id": "later-selection"})
    operations = context_ref.collection("operations")
    operations.document("op-save").update({"save_operation_id": "shared-save"})
    operations.document("op-child").update({
        "dependency_operation_id": "op-save", "target_doc_id": "saved-doc",
    })
    client.collection("dialog_operations").document("shared-save").set({
        "operation_id": "shared-save", "state": "completed",
        "result": {"document_id": "saved-doc"},
    })
    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    from core.integrations.firestore.operation_queue import complete_head
    assert complete_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                         fence=save_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == ["op-child"]


def test_malformed_linked_save_intent_fails_closed_and_advances_head(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-child", "op-later"))
    context_ref.update({"chat_id": 51})
    operations = context_ref.collection("operations")
    operations.document("op-save").update({
        "save_operation_id": "shared-save", "save_intent_id": ["malformed"],
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-save", "target_doc_id": "saved-doc",
    })
    client.collection("dialog_operations").document("shared-save").set({
        "operation_id": "shared-save", "state": "completed",
        "result": {"document_id": "saved-doc"},
    })
    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    assert complete_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                         fence=save_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    assert operations.document("op-child").get().to_dict()["error"]["code"] == "dependency_unavailable"
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-later"
    assert any(row.get("operation_id") == "op-later" for row in _rows(firestore_fake).values())


def test_original_linked_save_conflicting_parent_document_fails_closed(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-child"))
    context_ref.update({"chat_id": 52})
    operations = context_ref.collection("operations")
    operations.document("op-save").update({
        "save_operation_id": "shared-save", "doc_id": "contradictory-doc",
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-save", "target_doc_id": "saved-doc",
    })
    client.collection("dialog_operations").document("shared-save").set({
        "operation_id": "shared-save", "state": "completed",
        "result": {"document_id": "saved-doc"},
    })
    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    assert complete_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                         fence=save_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    assert operations.document("op-child").get().to_dict()["error"]["code"] == "dependency_unavailable"


def test_failed_retry_parent_uses_verified_draft_dialog_for_notice(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-retry", "op-child", "op-later"))
    context_ref.update({"chat_id": 53, "selected_dialog_number": 99})
    operations = context_ref.collection("operations")
    operations.document("op-retry").update({
        "draft_id": "draft-retry", "dialog_number": 17,
        "save_intent_id": "retry-intent", "doc_id": "saved-doc",
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-retry",
        "input_ref": f"{SESSIONS}/ctx/commands/op-child",
    })
    context_ref.collection("drafts").document("draft-retry").set({
        "draft_id": "draft-retry", "dialog_number": 17,
    })
    context_ref.collection("commands").document("op-child").set({
        "operation_id": "op-child", "context_id": "ctx", "dialog_number": 42,
    })
    retry_lease = claim_head(client, context_ref, owner="retry", now=NOW)
    assert retry_lease is not None
    assert fail_head(client, context_ref, operation_id="op-retry", owner=retry_lease.owner,
                     fence=retry_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    notification = next(
        data for path, data in firestore_fake.documents.items()
        if path.startswith("notification_outbox/")
    )
    assert notification["dialog_number"] == 17
    assert "Dialog #17" in notification["body"]
    assert "Dialog #42" not in notification["body"]
    assert "Dialog #99" not in notification["body"]


def test_malformed_retry_draft_id_omits_notice_attribution_and_advances(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-retry", "op-child", "op-later"))
    context_ref.update({"chat_id": 54, "selected_dialog_number": 99})
    operations = context_ref.collection("operations")
    operations.document("op-retry").update({
        "draft_id": "bad/draft", "dialog_number": 17,
        "save_intent_id": "retry-intent", "doc_id": "saved-doc",
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-retry",
        "input_ref": f"{SESSIONS}/ctx/commands/op-child",
    })
    context_ref.collection("commands").document("op-child").set({
        "operation_id": "op-child", "context_id": "ctx", "dialog_number": 42,
    })
    retry_lease = claim_head(client, context_ref, owner="retry", now=NOW)
    assert retry_lease is not None
    assert fail_head(client, context_ref, operation_id="op-retry", owner=retry_lease.owner,
                     fence=retry_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-later"
    notification = next(
        data for path, data in firestore_fake.documents.items()
        if path.startswith("notification_outbox/")
    )
    assert notification["dialog_number"] is None
    assert "Dialog #17" not in notification["body"]
    assert "Dialog #42" not in notification["body"]
    assert "Dialog #99" not in notification["body"]


def test_contradictory_parent_route_target_fails_closed(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-save", "op-child"))
    context_ref.update({"chat_id": 48})
    operations = context_ref.collection("operations")
    operations.document("op-save").update({
        "save_operation_id": "shared-save", "target_doc_id": "saved-doc",
        "route_result": {"target_doc_id": "contradictory-doc"},
    })
    operations.document("op-child").update({
        "dependency_operation_id": "op-save", "target_doc_id": "saved-doc",
    })
    client.collection("dialog_operations").document("shared-save").set({
        "operation_id": "shared-save", "state": "completed",
        "result": {"document_id": "saved-doc"},
    })
    save_lease = claim_head(client, context_ref, owner="save", now=NOW)
    assert save_lease is not None
    from core.integrations.firestore.operation_queue import complete_head
    assert complete_head(client, context_ref, operation_id="op-save", owner=save_lease.owner,
                         fence=save_lease.fence, now=NOW)
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is True
    assert calls == []
    assert operations.document("op-child").get().to_dict()["error"]["code"] == "dependency_unavailable"


def test_foreign_context_dependency_and_command_ref_cannot_supply_targets(firestore_fake):
    client = firestore_fake.client_class()
    foreign_ref, _ = _setup(client, "foreign", operation_ids=("foreign-parent",))
    foreign_ref.update({"chat_id": 49})
    foreign_ref.collection("commands").document("foreign-command").set({
        "context_id": "foreign", "operation_id": "child", "dialog_number": 6,
    })
    context_ref, payloads = _setup(client, "local", operation_ids=("child",))
    context_ref.update({"chat_id": 50, "selected_dialog_number": 99})
    context_ref.collection("operations").document("child").update({
        "dependency_operation_id": "foreign-parent",
        "target_dialog_number": 17,
        "input_ref": f"{SESSIONS}/foreign/commands/foreign-command",
    })
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["child"])) is True
    assert calls == []
    notice = next(
        data for path, data in firestore_fake.documents.items()
        if path.startswith("notification_outbox/")
    )
    assert notice["dialog_number"] is None
    assert "Dialog #17" not in notice["body"]
    assert "Dialog #99" not in notice["body"]


@pytest.mark.parametrize("dependency_id", ["missing-parent", "op-a", "op-b", "bad id"])
def test_invalid_dependency_lineage_fails_closed_without_processor_effects(firestore_fake, dependency_id):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-a", "op-b"))
    context_ref.update({"chat_id": 45})
    context_ref.collection("operations").document("op-a").update({
        "dependency_operation_id": dependency_id,
    })
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert calls == []
    assert context_ref.collection("operations").document("op-a").get().to_dict()[
        "error"
    ]["code"] == "dependency_unavailable"


def test_pending_dependency_releases_head_without_charging_execution_failures(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-parent", "op-child"))
    context_ref.collection("operations").document("op-child").update({
        "dependency_operation_id": "op-parent",
    })
    context_ref.update({"reserved_operation_id": "op-child", "head_sequence": 2})
    calls = []

    async def processor(context_id, operation, lease):
        calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-child"])) is False
    assert calls == []
    assert get_operation(context_ref, "op-child").state is QueueOperationState.RUNNING
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-child"
    assert context_ref.get().to_dict().get("active_operation_id") is None
    assert not any(path.endswith("/stages/telegram.dependency") for path in firestore_fake.documents)
    assert any(row.get("operation_id") == "op-child" for row in _rows(firestore_fake).values())


def test_stale_lease_cannot_persist_dependency_error_or_advance_head(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-parent", "op-child", "op-later"))
    context_ref.update({"chat_id": 46})
    context_ref.collection("operations").document("op-child").update({
        "dependency_operation_id": "op-parent",
    })
    parent_lease = claim_head(client, context_ref, owner="parent", now=NOW)
    assert parent_lease is not None
    assert fail_head(client, context_ref, operation_id="op-parent", owner=parent_lease.owner,
                     fence=parent_lease.fence, now=NOW)
    outbox_before = _rows(firestore_fake)
    now = [NOW]
    processor_calls = []

    async def processor(context_id, operation, lease):
        processor_calls.append(operation.operation_id)
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: now[0], owner_factory=lambda: "old-worker")

    def expire_and_replace(context_ref, operation_id, sequence):
        now[0] += timedelta(seconds=121)
        replacement = claim_head(client, context_ref, owner="replacement", now=now[0])
        assert replacement is not None
        return "failed", "op-parent", "The confirmed prerequisite did not complete successfully."

    worker._dependency_state = expire_and_replace
    assert _run(worker.handle_task(payloads["op-child"])) is False
    assert processor_calls == []
    context = context_ref.get().to_dict()
    assert context["reserved_operation_id"] == "op-child"
    assert context["active_operation_id"] == "op-child"
    assert context["lock_owner"] == "replacement"
    assert not any(path.startswith("notification_outbox/") for path in firestore_fake.documents)
    assert _rows(firestore_fake) == outbox_before


def test_queue_injected_governor_caps_physical_stage_failures_and_advances_only_on_exhaustion(
    firestore_fake,
):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-a", "op-b"))
    physical_calls = []

    async def processor(context_id, operation, lease, *, retry_governor):
        assert context_id == "ctx"
        assert operation.operation_id == lease.operation_id == "op-a"
        try:
            await retry_governor.run_stage_async(
                "op-a", "telegram.analysis",
                lambda: _fail_provider(physical_calls),
                owner=lease.owner, fence=lease.fence,
            )
        except ConnectionError as error:
            stage = retry_governor.get_stage("op-a", "telegram.analysis")
            error.retryable_stage = stage.last_error.retryable and stage.failures < 3
            raise
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-a"])) is False
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-a"
    assert _run(worker.handle_task(payloads["op-a"])) is False
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-a"
    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert physical_calls == ["provider"] * 3
    stage_rows = [data for path, data in firestore_fake.documents.items()
                  if path.endswith("/stages/telegram.analysis")]
    assert stage_rows[0]["failures"] == 3
    assert stage_rows[0]["attempts_started"] == 3
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-b"


async def _fail_provider(calls):
    calls.append("provider")
    raise ConnectionError("synthetic transient failure")


def test_permanent_stage_error_terminalizes_only_current_queue_head(firestore_fake):
    client = firestore_fake.client_class()
    context_ref, payloads = _setup(client, operation_ids=("op-a", "op-b"))
    context_ref.update({"pending_save_operation_id": "op-a"})
    physical_calls = []

    async def processor(context_id, operation, lease, *, retry_governor):
        try:
            await retry_governor.run_stage_async(
                "op-a", "telegram.classification",
                lambda: _permanent_provider_failure(physical_calls),
                owner=lease.owner, fence=lease.fence,
                retryable=lambda error: not isinstance(error, ValueError),
            )
        except ValueError as error:
            error.retryable_stage = False
            raise
        return QueueOperationState.COMPLETED

    worker = TelegramQueueWorker(client, processor, clock=lambda: NOW)
    assert _run(worker.handle_task(payloads["op-a"])) is True
    assert physical_calls == ["provider"]
    assert context_ref.get().to_dict()["reserved_operation_id"] == "op-b"
    assert "pending_save_operation_id" not in context_ref.get().to_dict()
    stage = next(data for path, data in firestore_fake.documents.items()
                 if path.endswith("/stages/telegram.classification"))
    assert stage["failures"] == 1
    assert stage["attempts_started"] == 1
    assert stage["last_error"]["retryable"] is False


async def _permanent_provider_failure(calls):
    calls.append("provider")
    raise ValueError("synthetic invalid provider input")


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
