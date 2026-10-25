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
from core.integrations.firestore.operations import (
    claim_operation,
    create_operation,
    get_operation as get_shared_operation,
    get_stage,
)
from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
from core.integrations.firestore.operation_queue import (
    QueueOperationRetryStore,
    claim_head,
    get_operation,
)
from core.integrations.firestore.telegram_recovery import FirestoreTelegramQueueRecoveryStore
from core.operations.models import OperationOrigin
from core.operations.outbox import TaskTargetConfig, publish_due_outbox_records
from core.operations.outbox import enqueue_expired_telegram_queue_recoveries
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
from services.connectors.telegram.save import TelegramCompletionCertificate


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


def test_checker_recovery_wakes_only_expired_reserved_heads_and_advances_tombstones(
    firestore_fake,
):
    client = firestore_fake.client_class()
    accepted = _accepted_save(client, update_id=799)
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    context_ref.update({
        "active_operation_id": accepted.operation_id,
        "lock_owner": "expired-owner",
        "lock_expires_at": NOW - timedelta(seconds=1),
        "fence": 1,
    })
    context_ref.collection("operations").document(accepted.operation_id).update({"state": "running"})

    store = FirestoreTelegramQueueRecoveryStore(client)
    assert enqueue_expired_telegram_queue_recoveries(store, now=NOW, limit=10) == 1
    records = [
        (path.rsplit("/", 1)[-1], data)
        for path, data in firestore_fake.documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
    ]
    wake_id, wake = next((item for item in records if item[0] != accepted.outbox_id))
    assert wake["target_kind"] == OutboxTargetKind.TELEGRAM_DISPATCH.value
    assert wake["generation"] == 2
    assert wake["state"] == OutboxState.PENDING.value
    assert context_ref.get().to_dict()["lock_expires_at"] == NOW - timedelta(seconds=1)

    record_outbox_delivery_result(client, wake_id, delivered=True, now=NOW)
    assert enqueue_expired_telegram_queue_recoveries(
        store, now=NOW + timedelta(seconds=1), limit=10,
    ) == 1
    records = [
        data for path, data in firestore_fake.documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
        and data.get("context_id") == accepted.context_id
        and data.get("operation_id") == accepted.operation_id
    ]
    assert sorted(item["generation"] for item in records) == [1, 2, 3]


def test_checker_recovery_null_head_contexts_do_not_starve_lease_less_head(firestore_fake):
    client = firestore_fake.client_class()
    client.collection(SESSIONS).document("a-empty-context").set({
        "reserved_operation_id": None,
        "lock_expires_at": None,
    })
    head_ref = client.collection(SESSIONS).document("z-reserved-context")
    head_ref.set({
        "reserved_operation_id": "operation-z",
        "active_operation_id": None,
        "lock_expires_at": None,
        "fence": 0,
    })
    head_ref.collection("operations").document("operation-z").set({
        "operation_id": "operation-z",
        "sequence": 1,
        "state": "pending",
    })

    recovered = enqueue_expired_telegram_queue_recoveries(
        FirestoreTelegramQueueRecoveryStore(client), now=NOW, limit=1,
    )

    assert recovered == 1
    wakeup = next(
        data for path, data in firestore_fake.documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
    )
    assert wakeup["context_id"] == "z-reserved-context"
    assert wakeup["operation_id"] == "operation-z"


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


def test_explicit_retry_of_unknown_create_keeps_original_gate_and_reconciles_only(
    firestore_fake, sdk_fakes,
):
    from core.dialogs.saving import accept_save, advance_save
    from core.integrations.firestore.operations import fail_operation
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _seed_failed_draft,
        _retry,
    )
    from core.integrations.google_drive.files import DriveCreateError
    from core.operations.models import OperationError
    from tests.fixtures.sdk_fakes import fake_discovery_build

    repository, client, context_id, draft_id, original_id, intent_id = (
        _seed_failed_draft(firestore_fake, update_id=77)
    )
    draft_ref = _document(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    draft_ref.update({"doc_id": None})
    original_shared_ref = client.collection("dialog_operations").document(original_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    # Replace the old hand-seeded core records with a real accepted save so this
    # test crosses the same core create and retry boundaries as production.
    original_shared_ref.delete()
    intent_ref.delete()
    accepted = accept_save(
        client,
        {
            "provider": "telegram", "title": "Recovered dialog",
            "folder_path": ["Client"],
            "utterances": [{"speaker": "Alex", "text": "sealed source text"}],
        },
        operation_id=original_id, caller_binding=f"telegram:{context_id}",
        origin=OperationOrigin.TELEGRAM,
        origin_operation_ref=f"{SESSIONS}/{context_id}/operations/{original_id}",
        now=NOW, save_intent_id=intent_id,
    )
    assert accepted.save_intent_id == intent_id
    original_lease = claim_operation(
        client, original_id, owner="original-save", now=NOW, lease_seconds=600,
    )
    assert original_lease is not None
    sdk_fakes.drive.queue("files.create", error=RuntimeError("lost create response"))
    with pytest.raises(DriveCreateError):
        advance_save(
            client, original_id, owner="original-save", fence=original_lease.fence, now=NOW,
            folders={"folder-a": {"name": "Client", "parent_id": "root"}},
            root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None,
            drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
            governor=RetryGovernor(client, clock=_Clock(NOW)),
        )
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    intent_after_unknown = intent_ref.get().to_dict()
    assert intent_after_unknown["state"] == "unknown"
    assert intent_after_unknown["dispatch_operation_id"] == original_id
    assert fail_operation(
        client, original_id, owner="original-save", fence=original_lease.fence,
        error=OperationError(
            code="create_outcome_unknown", category="execution",
            message="Create outcome is unknown", retryable=False,
        ), now=NOW,
    ) is True
    # The Telegram draft and source manifest are still the original sealed input.
    draft_ref.update({"doc_id": None})
    original_telegram_operation_ref = (
        client.collection(SESSIONS).document(context_id)
        .collection("operations").document(original_id)
    )
    original_telegram_operation_ref.update({"state": "failed", "finished_at": NOW})
    before_intent = intent_ref.get().to_dict()
    reusable = {
        "telegram.preparation": [{"text": "prepared source"}],
        "telegram.analysis": {"title": "Recovered title"},
        "telegram.classification": {"folder_id": "folder-a"},
    }
    old_queue_operation_ref = (
        client.collection(SESSIONS).document(context_id)
        .collection("operations").document(original_id)
    )
    for stage_key, result in reusable.items():
        old_queue_operation_ref.collection("stages").document(stage_key).set({
            "operation_id": original_id, "stage_key": stage_key,
            "state": "completed", "failures": 0, "attempts_started": 1,
            "result": result, "checkpoint_refs": [], "updated_at": NOW,
        })

    retry_acceptance = _retry(repository, context_id, 1, update_id=990)
    assert retry_acceptance.accepted is True
    retry_id = retry_acceptance.operation_id
    retry_shared = client.collection("dialog_operations").document(retry_id).get().to_dict()
    assert retry_shared["doc_id"] is None
    assert retry_shared["immutable_input_digest"] == original_shared_ref.get().to_dict()[
        "immutable_input_digest"
    ]
    assert retry_shared["immutable_input_ref"] == original_shared_ref.get().to_dict()[
        "immutable_input_ref"
    ]
    assert retry_shared["origin_binding"] == {
        "context_id": context_id, "draft_id": draft_id,
        "manifest_digest": repository.get_draft(context_id, draft_id)["manifest_digest"],
    }
    assert intent_ref.get().to_dict() == before_intent
    for stage_key, result in reusable.items():
        stage = (
            client.collection(SESSIONS).document(context_id)
            .collection("operations").document(retry_id)
            .collection("stages").document(stage_key).get().to_dict()
        )
        assert stage["state"] == "completed" and stage["result"] == result
        assert stage["operation_id"] == retry_id
        assert stage["failures"] == 0 and stage["attempts_started"] == 0

    lease = claim_operation(
        client, retry_id, owner="explicit-retry", now=NOW, lease_seconds=600,
    )
    assert lease is not None
    sdk_fakes.drive.queue("files.list", response={"files": []})
    with pytest.raises(UnknownOutcome):
        advance_save(
            client, retry_id, owner="explicit-retry", fence=lease.fence, now=NOW,
            folders={"folder-a": {"name": "Client", "parent_id": "root"}},
            root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None,
            drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
            governor=RetryGovernor(client, clock=_Clock(NOW)),
        )
    create_calls = [row for row in sdk_fakes.trace if row["operation"] == "files.create"]
    marker_queries = [row for row in sdk_fakes.trace if row["operation"] == "files.list"]
    assert len(create_calls) == 1
    assert len(marker_queries) == 1
    assert intent_ref.get().to_dict()["state"] == "unknown"
    assert intent_ref.get().to_dict()["dispatch_operation_id"] == original_id

    # A later positive marker read resolves that same in-flight create stage and
    # lets the retry continue with the original document identity.
    sdk_fakes.drive.queue("files.list", response={"files": [{
        "id": "marker-reconciled-doc",
        "mimeType": "application/vnd.google-apps.document",
        "parents": ["folder-a"], "trashed": False,
        "appProperties": {"tlbrain_save_id": intent_id},
    }]})
    with pytest.raises(RuntimeError):
        advance_save(
            client, retry_id, owner="explicit-retry", fence=lease.fence, now=NOW,
            folders={"folder-a": {"name": "Client", "parent_id": "root"}},
            root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None,
            drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
            governor=RetryGovernor(client, clock=_Clock(NOW)),
        )
    resolved_intent = intent_ref.get().to_dict()
    assert resolved_intent["state"] == "confirmed"
    assert resolved_intent["doc_id"] == "marker-reconciled-doc"
    assert get_shared_operation(client, retry_id).doc_id == "marker-reconciled-doc"
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    assert intent_ref.get().to_dict()["dispatch_operation_id"] == original_id

    from core.integrations.firestore.save_intents import _completion_operation_binding_matches

    binding = retry_shared["origin_binding"]
    assert _completion_operation_binding_matches(
        retry_shared, intent_ref.get().to_dict(), operation_id=retry_id,
        origin_binding=binding,
    ) is True
    foreign_binding = {**binding, "draft_id": "another-draft"}
    assert _completion_operation_binding_matches(
        retry_shared, intent_ref.get().to_dict(), operation_id=retry_id,
        origin_binding=foreign_binding,
    ) is False


def test_delayed_cleanup_resumes_old_authorization_while_newer_selection_survives(
    firestore_fake, monkeypatch,
):
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _retry,
        _seed_failed_draft,
    )
    from services.connectors.telegram.cleanup import TelegramDraftCleanup

    repository, client, context_id, old_draft_id, _original_id, _intent_id = (
        _seed_failed_draft(firestore_fake, update_id=88)
    )
    current = repository.accept_forward(
        188, chat_id=42, thread_id=11,
        message={"message_id": 288, "text": "current selection"}, now=NOW,
    )
    retry = _retry(repository, context_id, 1, update_id=988)
    retry_id = retry.operation_id
    old_draft = repository.get_draft(context_id, old_draft_id)
    intent_id = old_draft["immutable_save_intent_id"]
    manifest_digest = old_draft["manifest_digest"]
    doc_id = old_draft["doc_id"]
    binding = {
        "context_id": context_id, "draft_id": old_draft_id,
        "manifest_digest": manifest_digest,
    }
    certificate_id = f"completion-{retry_id}"
    shared_certificate = {
        "certificate_id": certificate_id, "save_intent_id": intent_id,
        "operation_id": retry_id, "doc_id": doc_id, "origin_binding": binding,
        "receipt_ref": f"index_completions/{doc_id}", "receipt_version": 1,
    }
    shared_authorization = {
        "certificate_id": certificate_id, "save_intent_id": intent_id,
        "operation_id": retry_id, "origin_binding": binding,
    }
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    intent_data = intent_ref.get().to_dict()
    intent_data.update({
        "completion_certificate": shared_certificate,
        "cleanup_authorization": shared_authorization,
    })
    intent_ref.set(intent_data)
    operation_ref = client.collection("dialog_operations").document(retry_id)
    operation_data = operation_ref.get().to_dict()
    operation_data.update({
        "state": "completed", "doc_id": doc_id,
        "cleanup_authorization": shared_authorization,
    })
    operation_ref.set(operation_data)
    certificate = TelegramCompletionCertificate(
        context_id=context_id, draft_id=old_draft_id, manifest_digest=manifest_digest,
        save_intent_id=intent_id, operation_id=retry_id,
        receipt_ref=shared_certificate["receipt_ref"], receipt_version=1,
        receipt_digest=f"receipt-{retry_id}",
        origin_binding=(context_id, old_draft_id, manifest_digest),
    )
    assert repository.commit_completion(certificate) is True

    newer_completed_id = "newer-completed"
    newer_certificate = {
        "context_id": context_id, "draft_id": newer_completed_id,
        "manifest_digest": "newer-manifest",
    }
    _document(client, f"{SESSIONS}/{context_id}/drafts/{newer_completed_id}").set({
        "draft_id": newer_completed_id, "state": "cleaned", "cleanup_state": "cleaned",
        "completion_certificate": newer_certificate, "updated_at": NOW + timedelta(seconds=1),
    })
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.update({
        "last_completion_draft_id": newer_completed_id,
        "last_doc_id": "newer-doc",
        "active_draft_id": current.draft_id,
    })
    client.collection("transcript_index").document(doc_id).set({
        "doc_id": doc_id, "user_facts": {"external-edit": "preserve"},
    })

    original_delete = repository.delete_prepared_page
    first_call = True

    def fail_after_source_cleanup(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            raise RuntimeError("synthetic crash during prepared deletion")
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(repository, "delete_prepared_page", fail_after_source_cleanup)
    cleanup = TelegramDraftCleanup(repository=repository, page_size=1)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _run(cleanup.cleanup(context_id=context_id))
    assert repository.list_messages(context_id, old_draft_id) == []
    assert repository.list_messages(context_id, current.draft_id)

    monkeypatch.setattr(repository, "delete_prepared_page", original_delete)
    assert _run(cleanup.cleanup(context_id=context_id)) is True
    deleted_source_after_resume = list(repository.list_messages(context_id, old_draft_id))
    assert _run(cleanup.cleanup(context_id=context_id)) is False
    assert repository.list_messages(context_id, old_draft_id) == deleted_source_after_resume == []
    context = repository.get_context(context_id)
    assert context["active_draft_id"] == current.draft_id
    assert context["last_completion_draft_id"] == newer_completed_id
    assert context["last_doc_id"] == "newer-doc"
    assert client.collection("transcript_index").document(doc_id).get().to_dict()[
        "user_facts"
    ] == {"external-edit": "preserve"}


@pytest.mark.parametrize(
    ("state", "create_outcome"),
    [("planned", None), ("planned", "rejected")],
)
def test_explicit_retry_accepts_pre_dispatch_or_definitively_rejected_create(
    firestore_fake, state, create_outcome,
):
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _retry,
        _seed_failed_draft,
    )

    repository, client, context_id, draft_id, operation_id, intent_id = (
        _seed_failed_draft(firestore_fake, update_id=97)
    )
    _document(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}").update({"doc_id": None})
    client.collection("dialog_operations").document(operation_id).update({"doc_id": None})
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    intent = intent_ref.get().to_dict()
    intent.update({
        "state": state, "create_outcome": create_outcome,
        "dispatch_token": None, "dispatch_operation_id": None,
        "dispatch_owner": None, "dispatch_fence": None, "doc_id": None,
        "confirmed_result": None,
    })
    intent_ref.set(intent)

    retry = _retry(repository, context_id, 1, update_id=2097)

    assert retry.accepted is True
    assert retry.operation_id
    assert intent_ref.get().to_dict() == intent


def test_explicit_retry_reuses_confirmed_doc_and_fill_checkpoint_without_writes(
    firestore_fake, sdk_fakes,
):
    from core.dialogs.saving import accept_save, advance_save
    from core.integrations.firestore.operations import fail_operation
    from core.operations.models import OperationError
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _retry,
        _seed_failed_draft,
    )
    from tests.fixtures.sdk_fakes import fake_discovery_build

    repository, client, context_id, draft_id, original_id, intent_id = (
        _seed_failed_draft(firestore_fake, update_id=98)
    )
    document_id = f"doc-98"
    draft_ref = _document(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    draft_ref.update({"state": "failed", "doc_id": None})
    shared_ref = client.collection("dialog_operations").document(original_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    shared_ref.delete()
    intent_ref.delete()
    accepted = accept_save(
        client,
        {
            "provider": "telegram", "title": "Recovered dialog",
            "folder_path": ["Client"],
            "utterances": [{"speaker": "Alex", "text": "sealed source text"}],
        },
        operation_id=original_id, caller_binding=f"telegram:{context_id}",
        origin=OperationOrigin.TELEGRAM,
        origin_operation_ref=f"{SESSIONS}/{context_id}/operations/{original_id}",
        now=NOW, save_intent_id=intent_id,
    )
    original_lease = claim_operation(
        client, original_id, owner="original-save", now=NOW, lease_seconds=600,
    )
    assert original_lease is not None
    sdk_fakes.drive.queue("files.create", response={"id": document_id})
    empty_document = {"revisionId": "rev-before", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
    ]}}
    sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-after"})
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "fill-time",
    })
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "post-fill-time",
    })
    original_progress = advance_save(
        client, original_id, owner="original-save", fence=original_lease.fence, now=NOW,
        folders={"folder-a": {"name": "Client", "parent_id": "root"}},
        root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None,
        drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
        governor=RetryGovernor(client, clock=_Clock(NOW)),
    )
    assert original_progress.document_id == document_id
    confirmed_intent = intent_ref.get().to_dict()
    assert confirmed_intent["state"] == "confirmed"
    assert confirmed_intent["dispatch_operation_id"] == original_id
    assert confirmed_intent["doc_id"] == document_id
    assert confirmed_intent["fill_checkpoint_ref"]
    assert fail_operation(
        client, original_id, owner="original-save", fence=original_lease.fence,
        error=OperationError(
            code="later_stage_failed", category="execution",
            message="Later save stage failed", retryable=False,
        ), now=NOW,
    ) is True

    retry = _retry(repository, context_id, 1, update_id=2098)

    assert retry.accepted is True
    retry_operation = client.collection("dialog_operations").document(retry.operation_id).get().to_dict()
    assert retry_operation["immutable_input_ref"] == accepted.immutable_input_ref
    assert retry_operation["immutable_input_digest"] == accepted.immutable_input_digest
    assert retry_operation["doc_id"] == document_id
    assert intent_ref.get().to_dict() == confirmed_intent
    lease = claim_operation(
        client, retry.operation_id, owner="confirmed-retry", now=NOW, lease_seconds=600,
    )
    assert lease is not None
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id,
        "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "post-fill-time",
    })
    progress = advance_save(
        client, retry.operation_id, owner="confirmed-retry", fence=lease.fence, now=NOW,
        folders={"folder-a": {"name": "Client", "parent_id": "root"}},
        root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None,
        drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
        governor=RetryGovernor(client, clock=_Clock(NOW)),
    )

    assert progress.document_id == document_id
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    assert len([row for row in sdk_fakes.trace if row["operation"] == "documents.batchUpdate"]) == 1
    assert intent_ref.get().to_dict()["dispatch_operation_id"] == original_id

    assert fail_operation(
        client, retry.operation_id, owner="confirmed-retry", fence=lease.fence,
        error=OperationError(
            code="later_stage_failed", category="execution",
            message="Later retry stage failed", retryable=False,
        ), now=NOW,
    ) is True
    client.collection(SESSIONS).document(context_id).collection("operations").document(
        retry.operation_id
    ).update({"state": "failed", "finished_at": NOW})
    draft_ref.update({"state": "failed"})

    second_retry = _retry(repository, context_id, 1, update_id=3098)
    assert second_retry.accepted is True
    second_operation = client.collection("dialog_operations").document(
        second_retry.operation_id
    ).get().to_dict()
    assert second_operation["retry_of_operation_id"] == retry.operation_id
    assert second_operation["retry_authorization"]["previous_operation_id"] == retry.operation_id
    assert second_operation["retry_authorization"]["create_gate_operation_id"] == original_id
    assert second_operation["doc_id"] == document_id
    second_lease = claim_operation(
        client, second_retry.operation_id, owner="second-confirmed-retry",
        now=NOW, lease_seconds=600,
    )
    assert second_lease is not None
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id,
        "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "post-fill-time",
    })
    second_progress = advance_save(
        client, second_retry.operation_id, owner="second-confirmed-retry",
        fence=second_lease.fence, now=NOW,
        folders={"folder-a": {"name": "Client", "parent_id": "root"}},
        root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None,
        drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
        governor=RetryGovernor(client, clock=_Clock(NOW)),
    )
    assert second_progress.document_id == document_id
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    assert len([row for row in sdk_fakes.trace if row["operation"] == "documents.batchUpdate"]) == 1

    binding = {
        "context_id": context_id, "draft_id": draft_id,
        "manifest_digest": repository.get_draft(context_id, draft_id)["manifest_digest"],
    }
    certificate_id = f"completion-{second_retry.operation_id}"
    shared_certificate = {
        "certificate_id": certificate_id, "save_intent_id": intent_id,
        "operation_id": second_retry.operation_id, "doc_id": document_id,
        "origin_binding": binding,
        "receipt_ref": f"index_completions/{document_id}", "receipt_version": 1,
    }
    cleanup_authorization = {
        "certificate_id": certificate_id, "save_intent_id": intent_id,
        "operation_id": second_retry.operation_id, "origin_binding": binding,
    }
    completed_intent = intent_ref.get().to_dict()
    completed_intent.update({
        "completion_certificate": shared_certificate,
        "cleanup_authorization": cleanup_authorization,
    })
    intent_ref.set(completed_intent)
    second_shared_ref = client.collection("dialog_operations").document(second_retry.operation_id)
    second_shared = second_shared_ref.get().to_dict()
    second_shared.update({
        "state": "completed", "doc_id": document_id,
        "cleanup_authorization": cleanup_authorization,
    })
    second_shared_ref.set(second_shared)
    from core.integrations.firestore.save_intents import _completion_operation_binding_matches

    assert _completion_operation_binding_matches(
        second_shared, completed_intent,
        operation_id=second_retry.operation_id, origin_binding=binding,
    ) is True
    certificate = TelegramCompletionCertificate(
        context_id=context_id, draft_id=draft_id,
        manifest_digest=binding["manifest_digest"], save_intent_id=intent_id,
        operation_id=second_retry.operation_id,
        receipt_ref=shared_certificate["receipt_ref"], receipt_version=1,
        receipt_digest=f"receipt-{second_retry.operation_id}",
        origin_binding=(context_id, draft_id, binding["manifest_digest"]),
    )
    assert repository.commit_completion(certificate) is True
    assert repository.authorize_cleanup(context_id, certificate) is True


def test_retry_recovers_confirmed_create_before_shared_doc_checkpoint(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.dialogs import saving
    from core.dialogs.saving import accept_save, advance_save
    from core.integrations.firestore.operations import fail_operation
    from core.operations.models import OperationError
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _retry,
        _seed_failed_draft,
    )
    from tests.fixtures.sdk_fakes import fake_discovery_build

    repository, client, context_id, draft_id, original_id, intent_id = (
        _seed_failed_draft(firestore_fake, update_id=98)
    )
    document_id = "doc-before-shared-checkpoint"
    draft_ref = _document(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}")
    draft_ref.update({"state": "failed", "doc_id": None})
    shared_ref = client.collection("dialog_operations").document(original_id)
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    shared_ref.delete()
    intent_ref.delete()
    accept_save(
        client,
        {
            "provider": "telegram", "title": "Recovered dialog",
            "folder_path": ["Client"],
            "utterances": [{"speaker": "Alex", "text": "sealed source text"}],
        },
        operation_id=original_id, caller_binding=f"telegram:{context_id}",
        origin=OperationOrigin.TELEGRAM,
        origin_operation_ref=f"{SESSIONS}/{context_id}/operations/{original_id}",
        now=NOW, save_intent_id=intent_id,
    )
    lease = claim_operation(
        client, original_id, owner="before-shared-checkpoint", now=NOW, lease_seconds=600,
    )
    assert lease is not None
    sdk_fakes.drive.queue("files.create", response={"id": document_id})

    def crash_before_shared_checkpoint(*_args, **_kwargs):
        raise RuntimeError("crash before shared doc checkpoint")

    store_created_doc = saving._store_created_doc
    monkeypatch.setattr(saving, "_store_created_doc", crash_before_shared_checkpoint)
    with pytest.raises(RuntimeError, match="before shared doc checkpoint"):
        advance_save(
            client, original_id, owner="before-shared-checkpoint", fence=lease.fence,
            now=NOW, folders={"folder-a": {"name": "Client", "parent_id": "root"}},
            root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
            suggestion=lambda *_args: None,
            drive=fake_discovery_build("drive", "v3"),
            docs=fake_discovery_build("docs", "v1"),
            governor=RetryGovernor(client, clock=_Clock(NOW)),
        )
    intent_after_crash = intent_ref.get().to_dict()
    assert intent_after_crash["state"] == "confirmed"
    assert intent_after_crash["doc_id"] == document_id
    assert client.collection("dialog_operations").document(original_id).get().to_dict()[
        "doc_id"
    ] is None
    assert draft_ref.get().to_dict()["doc_id"] is None
    assert fail_operation(
        client, original_id, owner="before-shared-checkpoint", fence=lease.fence,
        error=OperationError(
            code="create_checkpoint_interrupted", category="execution",
            message="Create checkpoint interrupted", retryable=False,
        ), now=NOW,
    ) is True
    monkeypatch.setattr(saving, "_store_created_doc", store_created_doc)

    retry = _retry(repository, context_id, 1, update_id=3098)
    assert retry.accepted is True
    assert client.collection("dialog_operations").document(retry.operation_id).get().to_dict()[
        "doc_id"
    ] == document_id
    retry_lease = claim_operation(
        client, retry.operation_id, owner="after-shared-checkpoint",
        now=NOW, lease_seconds=600,
    )
    assert retry_lease is not None
    empty_document = {"revisionId": "rev-before", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
    ]}}
    sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-after"})
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "fill-time",
    })
    sdk_fakes.drive.queue("files.get", response={
        "id": document_id, "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "post-fill-time",
    })
    progress = advance_save(
        client, retry.operation_id, owner="after-shared-checkpoint", fence=retry_lease.fence,
        now=NOW, folders={"folder-a": {"name": "Client", "parent_id": "root"}},
        root_folder_id="root", speaker_index_lookup=lambda _speaker: (),
        suggestion=lambda *_args: None,
        drive=fake_discovery_build("drive", "v3"),
        docs=fake_discovery_build("docs", "v1"),
        governor=RetryGovernor(client, clock=_Clock(NOW)),
    )

    assert progress.document_id == document_id
    assert len([row for row in sdk_fakes.trace if row["operation"] == "files.create"]) == 1
    assert len([row for row in sdk_fakes.trace if row["operation"] == "documents.batchUpdate"]) == 1


@pytest.mark.parametrize(("malformed", "draft_doc_id", "shared_doc_id"), [
    ({"state": "planned", "create_outcome": "confirmed"}, "doc-99", "doc-99"),
    ({"state": "planned", "create_outcome": "rejected", "dispatch_token": None,
      "dispatch_operation_id": None, "dispatch_owner": None, "dispatch_fence": None,
      "doc_id": None, "confirmed_result": None}, "doc-99", "doc-99"),
    ({"state": "unknown", "create_outcome": "unknown", "doc_id": None,
      "confirmed_result": None}, "doc-99", "doc-99"),
    ({"state": "dispatched", "create_outcome": "dispatched", "doc_id": None,
      "confirmed_result": None}, "doc-99", "doc-99"),
    ({"state": "rejected"}, "doc-99", "doc-99"),
    ({"state": "confirmed", "create_outcome": "confirmed", "dispatch_token": None},
     "doc-99", "doc-99"),
    ({"state": "confirmed", "create_outcome": "confirmed", "doc_id": "foreign-doc"},
     "doc-99", "doc-99"),
])
def test_explicit_retry_rejects_inconsistent_intent_binding_write_free(
    firestore_fake, malformed, draft_doc_id, shared_doc_id,
):
    from tests.services.connectors.telegram.test_failed_drafts import (
        _document,
        _retry,
        _seed_failed_draft,
    )

    repository, client, context_id, draft_id, operation_id, intent_id = (
        _seed_failed_draft(firestore_fake, update_id=99)
    )
    intent_ref = client.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    intent_ref.update(malformed)
    _document(client, f"{SESSIONS}/{context_id}/drafts/{draft_id}").update({
        "doc_id": draft_doc_id,
    })
    client.collection("dialog_operations").document(operation_id).update({
        "doc_id": shared_doc_id,
    })
    before = firestore_fake.documents

    retry = _retry(repository, context_id, 1, update_id=2099)

    assert retry.accepted is False
    assert firestore_fake.documents == before


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

