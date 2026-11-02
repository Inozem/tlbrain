"""Focused runtime regressions for immutable numbered save retries."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import asyncio
import hashlib
import inspect
import json
import socket
from datetime import timedelta

import httpx


pytestmark = pytest.mark.offline


@pytest.mark.parametrize("async_stage", [False, True])
def test_parent_bound_governor_uses_fresh_runtime_time(async_stage):
    from core.operations.models import OperationStageRecord, StageState
    from services.connectors.telegram.runtime.saving import _ParentBoundRetryGovernor

    class Clock:
        value = datetime(2026, 10, 30, tzinfo=timezone.utc)

        def __call__(self):
            return self.value

    class Store:
        def __init__(self):
            self.reserved_at = None
            self.completed_at = None

        def validate_lease(self, **_kwargs):
            return True

        def get_stage(self, *, stage_key, **_kwargs):
            return OperationStageRecord(stage_key=stage_key, state=StageState.PENDING)

        def reserve_attempt(self, **kwargs):
            self.reserved_at = kwargs["now"]
            return "attempt-one"

        def complete_attempt(self, **kwargs):
            self.completed_at = kwargs["now"]
            return True

    clock, store = Clock(), Store()
    governor = _ParentBoundRetryGovernor(
        SimpleNamespace(client=object(), max_failures=3, lease_seconds=300,
                        retry_store=store),
        clock=clock, assert_current=lambda: None,
    )
    stale_timestamp = clock.value - timedelta(minutes=10)
    if async_stage:
        async def run():
            return await governor.run_stage_async(
                "operation", "stage", lambda: "done", owner="owner", fence=1,
                now=stale_timestamp,
            )
        assert asyncio.run(run()) == "done"
    else:
        assert governor.run_stage(
            "operation", "stage", lambda: "done", owner="owner", fence=1,
            now=stale_timestamp,
        ) == "done"
    assert store.reserved_at == clock.value
    assert store.completed_at == clock.value


def test_retry_binding_uses_canonical_work_metadata_without_retry_draft_id(firestore_fake):
    """Recognize appended retry work from immutable queue fields and lineage."""
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    client = firestore_fake.client_class()
    context_id, operation_id = "42_11", "tgretry-actual"
    operation = SimpleNamespace(
        operation_id=operation_id,
        input_ref=f"telegram_sessions/{context_id}/commands/301",
    )
    client.collection("telegram_sessions").document(context_id).collection(
        "commands"
    ).document("301").set({"context_id": context_id, "kind": "retry", "text": "retry #1"})
    client.collection("telegram_sessions").document(context_id).collection(
        "operations"
    ).document(operation_id).set({
        "operation_id": operation_id,
        "draft_id": "draft-a",
        "target_draft_id": "draft-a",
        "save_operation_id": operation_id,
        "save_intent_id": "original-save-op",
        "retry_request_operation_id": "retry-request-300",
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1},
    })

    runtime = TelegramSaveRuntime(repository=TelegramRepository(client=client))
    assert runtime._captured_binding(context_id, operation) == ("draft-a", "original-save-op")


def test_retry_binding_rejects_work_that_crosses_the_canonical_draft(firestore_fake):
    """A forged or stale retry row cannot select another sealed draft."""
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    client = firestore_fake.client_class()
    context_id, operation_id = "42_11", "tgretry-stale"
    operation = SimpleNamespace(
        operation_id=operation_id,
        input_ref=f"telegram_sessions/{context_id}/commands/401",
    )
    client.collection("telegram_sessions").document(context_id).collection(
        "commands"
    ).document("401").set({"context_id": context_id, "kind": "retry"})
    client.collection("telegram_sessions").document(context_id).collection(
        "operations"
    ).document(operation_id).set({
        "operation_id": operation_id, "draft_id": "draft-b",
        "target_draft_id": "draft-a", "save_operation_id": operation_id,
        "save_intent_id": "original-save-op", "retry_request_operation_id": "retry-request-400",
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1},
    })

    runtime = TelegramSaveRuntime(repository=TelegramRepository(client=client))
    assert runtime._captured_binding(context_id, operation) is None


def _seed_save_repository_retry(client, *, operation_id="retry-save", draft_id="draft-a"):
    """Persist the canonical minimal retry command, lease, draft, and manifest."""
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository

    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    context_id, intent_id, owner = "42_11", "original-intent", "retry-worker"
    ids = [31]
    digest = hashlib.sha256(json.dumps(
        {"draft_id": draft_id, "message_ids": ids},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": owner, "fence": 3, "lock_expires_at": now + timedelta(minutes=5),
    })
    operation_ref = context_ref.collection("operations").document(operation_id)
    operation_ref.set({
        "operation_id": operation_id, "state": "running",
        "input_ref": f"telegram_sessions/{context_id}/commands/900",
        "draft_id": draft_id, "target_draft_id": draft_id,
        "save_operation_id": operation_id, "save_intent_id": intent_id,
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1},
    })
    context_ref.collection("commands").document("900").set({
        "context_id": context_id, "kind": "retry", "operation_id": operation_id,
        "draft_id": draft_id, "save_intent_id": intent_id, "dialog_number": 1,
    })
    expected_path = (f"telegram_sessions/{context_id}/drafts/{draft_id}/"
                     f"save_manifests/{operation_id}")
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft_ref.set({
        "draft_id": draft_id, "state": "processing", "active_save_operation_id": operation_id,
        "immutable_save_intent_id": intent_id, "manifest_ref": expected_path,
        "manifest_digest": digest, "ordered_message_ids": ids,
    })
    draft_ref.collection("save_manifests").document(operation_id).set({
        "draft_id": draft_id, "save_operation_id": operation_id,
        "manifest_digest": digest, "ordered_message_ids": ids,
    })
    draft_ref.collection("messages").document("31").set({
        "message_id": 31, "intake_sequence": 1, "text": "Original source line.",
    })
    repository = FirestoreTelegramSaveRepository(
        TelegramRepository(client=client), context_id=context_id, operation_id=operation_id,
        owner=owner, fence=3, clock=lambda: now,
    )
    return repository, context_ref, operation_ref, draft_ref, expected_path, intent_id


def test_actual_save_repository_loads_canonical_numbered_retry_without_retry_draft_id(firestore_fake):
    """A legitimate canonical retry loads its original immutable source manifest."""
    client = firestore_fake.client_class()
    repository, *_ = _seed_save_repository_retry(client)

    loaded = asyncio.run(repository.load_sealed_draft("draft-a"), loop_factory=_selector_loop)

    assert loaded is not None
    assert loaded["draft_id"] == "draft-a"
    assert loaded["immutable_save_intent_id"] == "original-intent"
    assert loaded["messages"] == [{
        "message_id": 31, "intake_sequence": 1, "text": "Original source line.",
    }]


@pytest.mark.parametrize("crossing", ["target", "intent", "manifest_digest", "manifest_path", "command", "lease"])
def test_actual_save_repository_rejects_crossed_retry_binding(firestore_fake, crossing):
    """A retry cannot load a draft when any persisted binding or lease has crossed."""
    client = firestore_fake.client_class()
    repository, context_ref, operation_ref, draft_ref, _path, _intent = _seed_save_repository_retry(client)
    if crossing == "target":
        operation_ref.update({"target_draft_id": "draft-b"})
    elif crossing == "intent":
        operation_ref.update({"save_intent_id": "foreign-intent"})
    elif crossing == "manifest_digest":
        draft_ref.update({"manifest_digest": "foreign-digest"})
    elif crossing == "manifest_path":
        draft_ref.update({"manifest_ref": "telegram_sessions/42_11/drafts/draft-b/save_manifests/retry-save"})
    elif crossing == "command":
        context_ref.collection("commands").document("900").update({"draft_id": "draft-b"})
    elif crossing == "lease":
        context_ref.update({"lock_owner": "another-worker"})

    if crossing == "lease":
        with pytest.raises(PermissionError, match="queue lease is stale"):
            asyncio.run(repository.load_sealed_draft("draft-a"), loop_factory=_selector_loop)
        return
    loaded = asyncio.run(repository.load_sealed_draft("draft-a"), loop_factory=_selector_loop)

    assert loaded is None


def test_actual_save_repository_loads_request_bound_retry_with_immutable_source_command(firestore_fake):
    """Accepted retry requests bind the new work while preserving their source command."""
    client = firestore_fake.client_class()
    repository, context_ref, operation_ref, _draft_ref, _path, _intent = _seed_save_repository_retry(client)
    request_id = "retry-request-1"
    operation_ref.update({
        "retry_request_operation_id": request_id,
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1,
                           "retry_request_operation_id": request_id},
    })
    context_ref.collection("commands").document("900").update({
        "kind": "freeform", "operation_id": request_id, "draft_id": "draft-b",
    })
    context_ref.collection("operations").document(request_id).set({
        "operation_id": request_id,
        "input_ref": "telegram_sessions/42_11/commands/900",
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1},
    })
    context_ref.collection("retry_requests").document(request_id).set({
        "accepted": True, "request_operation_id": request_id,
        "context_id": "42_11", "draft_id": "draft-a", "dialog_number": 1,
        "operation_id": "retry-save",
    })

    loaded = asyncio.run(repository.load_sealed_draft("draft-a"), loop_factory=_selector_loop)

    assert loaded is not None
    assert loaded["draft_id"] == "draft-a"
    assert loaded["immutable_save_intent_id"] == "original-intent"


def test_actual_save_repository_rejects_retry_with_crossed_request_binding(firestore_fake):
    """Request based retries must match their canonical accepted retry request."""
    client = firestore_fake.client_class()
    repository, context_ref, operation_ref, _draft_ref, _path, _intent = _seed_save_repository_retry(client)
    request_id = "retry-request-1"
    operation_ref.update({
        "retry_request_operation_id": request_id,
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1,
                           "retry_request_operation_id": request_id},
    })
    command_ref = context_ref.collection("commands").document("900")
    command_ref.update({
        "kind": "freeform", "operation_id": request_id,
    })
    context_ref.collection("operations").document(request_id).set({
        "operation_id": request_id,
        "input_ref": "telegram_sessions/42_11/commands/900",
        "route_control": {"status": "ok", "kind": "retry", "dialog_number": 1},
    })
    context_ref.collection("retry_requests").document(request_id).set({
        "accepted": True, "request_operation_id": request_id,
        "context_id": "42_11", "draft_id": "draft-b", "dialog_number": 1,
        "operation_id": "retry-save",
    })

    loaded = asyncio.run(repository.load_sealed_draft("draft-a"), loop_factory=_selector_loop)

    assert loaded is None


@pytest.mark.parametrize("status, expected", [(400, False), (403, False), (408, True), (429, True), (500, True)])
def test_save_retry_policy_stops_on_definitive_provider_rejection(status, expected):
    """Use the shared stage governor's callback policy for provider HTTP failures."""
    from services.connectors.telegram.runtime.saving import _provider_error_retryable

    error = SimpleNamespace(response=SimpleNamespace(status_code=status))
    assert _provider_error_retryable(error) is expected


def test_genai_client_error_code_is_used_by_provider_retry_policy():
    from google.genai.errors import ClientError
    from services.connectors.telegram.runtime.saving import _provider_error_retryable

    assert _provider_error_retryable(ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})) is False
    assert _provider_error_retryable(ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
    }})) is True


def test_permanent_fill_stage_rejection_terminalizes_shared_save_once(firestore_fake):
    """Exercise shared governor handling for a definitive fill-stage rejection."""
    from core.dialogs.saving import accept_save
    from core.integrations.firestore.operations import claim_operation, get_operation, get_stage
    from core.operations.models import OperationOrigin, OperationState
    from services.connectors.telegram.runtime.saving import _TelegramSaveRetryGovernor

    client = firestore_fake.client_class()
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    accepted = accept_save(
        client,
        {"provider": "telegram", "title": "Sealed", "utterances": [
            {"speaker": "Ada", "text": "Exact source."},
        ]},
        operation_id="telegram-post-doc-failure", caller_binding="telegram:42_11",
        origin=OperationOrigin.TELEGRAM, now=now, save_intent_id="telegram-post-doc-failure",
    )
    operation_id = accepted.operation.operation_id
    lease = claim_operation(
        client, operation_id, owner="telegram-save-test", now=now, lease_seconds=600,
    )
    assert lease is not None

    class PermanentProviderError(RuntimeError):
        response = SimpleNamespace(status_code=400)

    with pytest.raises(PermanentProviderError):
        _TelegramSaveRetryGovernor(
            client, clock=SimpleNamespace(now=lambda: now),
        ).run_stage(
            operation_id, "fill", lambda: (_ for _ in ()).throw(PermanentProviderError()),
            owner="telegram-save-test", fence=lease.fence, now=now,
        )

    assert get_stage(client, operation_id, "fill").failures == 1
    assert get_operation(client, operation_id).state is OperationState.FAILED


def test_shared_save_gemini_402_is_safe_and_terminal_after_one_provider_attempt(firestore_fake):
    from google.genai.errors import ClientError
    from core.dialogs.saving import accept_save
    from core.integrations.firestore.operations import claim_operation, get_stage
    from core.operations.models import OperationOrigin
    from core.operations.retries import RetryBudgetExceeded
    from services.connectors.telegram.runtime.saving import _TelegramSaveRetryGovernor

    client = firestore_fake.client_class()
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    accepted = accept_save(
        client,
        {"provider": "telegram", "title": "Sealed", "utterances": [
            {"speaker": "Ada", "text": "Exact source."},
        ]},
        operation_id="telegram-gemini-payment-required", caller_binding="telegram:42_11",
        origin=OperationOrigin.TELEGRAM, now=now,
        save_intent_id="telegram-gemini-payment-required",
    )
    operation_id = accepted.operation.operation_id
    lease = claim_operation(
        client, operation_id, owner="telegram-save-test", now=now, lease_seconds=600,
    )
    assert lease is not None
    calls = []
    error = ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})
    governor = _TelegramSaveRetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
    )

    def request():
        calls.append(1)
        raise error

    with pytest.raises(ClientError):
        governor.run_stage(
            operation_id, "gemini-analysis", request,
            owner="telegram-save-test", fence=lease.fence, now=now,
        )
    with pytest.raises((RetryBudgetExceeded, RuntimeError)):
        governor.run_stage(
            operation_id, "gemini-analysis", request,
            owner="telegram-save-test", fence=lease.fence, now=now,
        )

    stage = get_stage(client, operation_id, "gemini-analysis")
    assert calls == [1]
    assert stage.failures == 1
    assert stage.last_error.category == "billing"
    assert stage.last_error.service == "gemini"
    assert stage.last_error.retryable is False
    assert "private-marker" not in repr(stage.last_error)


def _selector_loop():
    """Create a selector event loop compatible with Windows test runners."""
    import os

    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    original_socketpair = socket.socketpair

    def local_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reader.bind(("127.0.0.1", 0))
        writer.bind(("127.0.0.1", 0))
        writer.connect(reader.getsockname())
        reader.connect(writer.getsockname())
        return reader, writer

    socket.socketpair = local_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = original_socketpair


def test_default_worker_terminalizes_three_real_metadata_failures_once(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """Run the real default save path through three metadata provider failures."""
    from google.genai import types
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.runtime.factory import build_default_queue_worker
    from services.connectors.telegram.runtime import saving as saving_runtime_module

    settlement_logs = []
    original_error_log = saving_runtime_module.logger.error

    def capture_error_log(message, *args, **kwargs):
        extra = kwargs.get("extra", {})
        if extra.get("event") == "telegram_save_settlement":
            settlement_logs.append(dict(extra))
        return original_error_log(message, *args, **kwargs)

    monkeypatch.setattr(saving_runtime_module.logger, "error", capture_error_log)

    class FakeClock:
        def __init__(self):
            self.value = datetime(2026, 10, 30, tzinfo=timezone.utc)

        def __call__(self):
            return self.value

        def advance(self):
            self.value += timedelta(minutes=10)

    clock = FakeClock()
    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control-queue")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.pytest.invalid/internal/dispatch")
    monkeypatch.setenv("TELEGRAM_DELIVERY_URL", "https://telegram.pytest.invalid/internal/deliver")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://task-worker.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "pytest-task-worker@synthetic-project.invalid")

    original_factory = build_default_queue_worker

    def build_clock_configured_worker(client):
        worker = original_factory(client)
        worker.clock = clock
        worker.processor.clock = clock
        worker.processor.save_handler.clock = clock
        return worker

    monkeypatch.setattr(
        "services.connectors.telegram.runtime.factory.build_default_queue_worker",
        build_clock_configured_worker,
    )
    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response),
        ), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)

    route = {"kind": "save", "target_ref": None, "target_operation_id": None,
             "dialog_number": None, "result_refs": [], "substantive_text": None,
             "tool_choice": None, "certainty": "certain"}
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="route_command", args=route),
        )]))],
    ))
    secret_marker = "synthetic-provider-secret-marker"
    for _ in range(3):
        sdk_fakes.gemini.queue(
            "generate_content", error=RuntimeError(secret_marker),
        )

    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    async def webhook(update):
        return await post("/telegram/webhook", update, secret=True)

    async def deliver_task(body):
        return await post("/internal/deliver", body)

    async def dispatch(body):
        return await post("/internal/dispatch", body)

    forward = {"update_id": 15501, "message": {
        "message_id": 61, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000},
        "text": "The exact forwarded source line.",
    }}
    assert asyncio.run(webhook(forward), loop_factory=_selector_loop).status_code == 200
    forward_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    ack_task = next(body for body in forward_tasks if body.get("notification_id"))
    assert asyncio.run(deliver_task(ack_task), loop_factory=_selector_loop).status_code == 200

    save_update = {"update_id": 15502, "message": {
        "message_id": 62, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(webhook(save_update), loop_factory=_selector_loop).status_code == 200
    command_tasks = [item["body"] for item in task_requests
                     if item["body"].get("target_kind") == "telegram_dispatch"]
    save_task = next(body for body in reversed(command_tasks) if not body.get("notification_id"))
    context_id = save_task["context_id"]

    responses = []
    observed_bindings = []
    for attempt in range(3):
        if attempt:
            clock.advance()
            canonical = task_requests[-1]["body"]
            assert canonical.get("operation_id") == save_task["operation_id"]
            save_task = canonical
        response = asyncio.run(dispatch(save_task), loop_factory=_selector_loop)
        responses.append(response)
        current_row = (db.collection("telegram_sessions").document(context_id)
                       .collection("operations").document(save_task["operation_id"])
                       .get().to_dict())
        current_draft = (db.collection("telegram_sessions").document(context_id)
                         .collection("drafts").document(current_row["target_draft_id"])
                         .get().to_dict())
        observed_bindings.append((current_row["operation_id"], current_draft["draft_id"],
                                  current_draft["manifest_digest"]))
        if attempt < 2:
            assert response.status_code == 200, response.text
            assert response.json() == {"status": "noop"}

    assert responses[-1].status_code == 200, responses[-1].text
    assert responses[-1].json() == {"status": "processed"}
    assert len(set(observed_bindings)) == 1
    metadata_calls = [event for event in sdk_fakes.trace
                      if event["boundary"] == "gemini"
                      and event["operation"] == "generate_content"
                      and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_calls) == 3
    operation_id = save_task["operation_id"]
    queue_row = (db.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(operation_id).get().to_dict())
    draft_id = queue_row["target_draft_id"]
    draft = (db.collection("telegram_sessions").document(context_id)
             .collection("drafts").document(draft_id).get().to_dict())
    assert queue_row["state"] == "failed"
    assert draft["state"] == "failed"
    assert draft["dialog_number"] == 1
    assert draft["ordered_message_ids"] == [61]
    assert isinstance(draft["manifest_digest"], str) and draft["manifest_digest"]
    manifest = (db.collection("telegram_sessions").document(context_id)
                .collection("drafts").document(draft_id).collection("save_manifests")
                .document(operation_id).get().to_dict())
    assert manifest["manifest_digest"] == draft["manifest_digest"]
    assert manifest["ordered_message_ids"] == [61]
    saved_source = (db.collection("telegram_sessions").document(context_id)
                    .collection("drafts").document(draft_id).collection("messages")
                    .document("61").get().to_dict())
    assert saved_source["text"] == "The exact forwarded source line."
    assert not db.collection("dialog_operations").document(operation_id).get().exists
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
    assert not db.collection(SAVE_INTENTS_COLLECTION).document(operation_id).get().exists
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]
    error_messages = [payload for method, payload in sent
                      if method == "sendMessage" and "Error in gemini/metadata" in payload.get("text", "")]
    assert len(error_messages) == 1
    error_text = error_messages[0]["text"]
    assert "Dialog #1" in error_text and operation_id in error_text
    assert "administrator" in error_text and "logs" in error_text
    assert secret_marker not in error_text
    settlement = next(row for row in settlement_logs if row.get("outcome") == "failed")
    assert settlement["operation_id"] == operation_id
    assert settlement["context_id"] == context_id
    assert settlement["service"] == "gemini" and settlement["stage"] == "metadata"
    assert settlement["failures"] == 3 and settlement["error_code"]
    assert secret_marker not in str(settlement)
    error_rows = [row for path, row in firestore_fake.documents.items()
                  if path.startswith("notification_outbox/")
                  and row.get("operation_id") == operation_id and row.get("kind") == "error"]
    assert len(error_rows) == 1
    assert secret_marker not in caplog.text

    duplicate = asyncio.run(dispatch(save_task), loop_factory=_selector_loop)
    assert duplicate.status_code == 200
    assert len([payload for method, payload in sent
                if method == "sendMessage" and "Error in gemini/metadata" in payload.get("text", "")]) == 1


def test_default_app_numbered_retry_saves_failed_dialog_without_replacing_newer_draft(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Retry the failed numbered dialog through the real queue and managed index gates."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.folders import upsert_folder
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.runtime.factory import build_default_queue_worker
    from services.connectors.telegram.runtime import saving as saving_runtime_module

    settlement_logs = []
    original_info_log = saving_runtime_module.logger.info

    def capture_info_log(message, *args, **kwargs):
        extra = kwargs.get("extra", {})
        if extra.get("event") == "telegram_save_settlement":
            settlement_logs.append(dict(extra))
        return original_info_log(message, *args, **kwargs)

    monkeypatch.setattr(saving_runtime_module.logger, "info", capture_info_log)

    class FakeClock:
        def __init__(self):
            self.value = datetime.now(timezone.utc)

        def __call__(self):
            return self.value

        def advance(self):
            self.value += timedelta(minutes=10)

    clock = FakeClock()
    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    for key, value in {
        "TELEGRAM_BOT_TOKEN": "123456:synthetic-token",
        "TELEGRAM_CONTROL_QUEUE": "pytest-telegram-control-queue",
        "TELEGRAM_CONTROL_URL": "https://telegram.pytest.invalid/internal/dispatch",
        "TELEGRAM_DELIVERY_URL": "https://telegram.pytest.invalid/internal/deliver",
        "CLOUD_TASKS_OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
        "OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
        "DIALOG_OPERATIONS_QUEUE": "pytest-dialog-operations-queue",
        "VECTOR_SYNC_URL": "https://vector-sync.pytest.invalid",
        "ROOT_FOLDER_URL": "https://drive.google.com/drive/folders/pytest-synthetic-root",
        "GOOGLE_REFRESH_TOKEN": "pytest-synthetic-refresh",
        "GOOGLE_CLIENT_ID": "pytest-synthetic-client",
        "GOOGLE_CLIENT_SECRET": "pytest-synthetic-secret",
    }.items():
        monkeypatch.setenv(key, value)

    original_factory = build_default_queue_worker

    def build_clock_configured_worker(client):
        worker = original_factory(client)
        worker.clock = clock
        worker.processor.clock = clock
        worker.processor.save_handler.clock = clock
        return worker

    monkeypatch.setattr(
        "services.connectors.telegram.runtime.factory.build_default_queue_worker",
        build_clock_configured_worker,
    )
    task_requests = []
    enqueue_fake = lambda **kwargs: task_requests.append(kwargs)
    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []

    def bot_response(request):
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response),
        ), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    upsert_folder("pytest-synthetic-root", "Root", None, db)
    upsert_folder("folder-client", "Client", "pytest-synthetic-root", db)

    def route_response(kind, **values):
        route = {"kind": kind, "target_ref": None, "target_operation_id": None,
                 "dialog_number": None, "result_refs": [], "substantive_text": None,
                 "tool_choice": None, "certainty": "certain", **values}
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                function_call=types.FunctionCall(name="route_command", args=route),
            )]))],
        ))

    route_response("save")
    for _ in range(3):
        sdk_fakes.gemini.queue("generate_content", error=RuntimeError("synthetic metadata outage"))

    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    async def webhook(update):
        return await post("/telegram/webhook", update, secret=True)

    async def dispatch(task):
        return await post("/internal/dispatch", task)

    async def deliver(task):
        return await post("/internal/deliver", task)

    source_a = "The exact forwarded source line for dialog A."
    forward_a = {"update_id": 15501, "message": {
        "message_id": 61, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": source_a,
    }}
    assert asyncio.run(webhook(forward_a), loop_factory=_selector_loop).status_code == 200
    ack_a = next(item["body"] for item in task_requests
                 if item["body"].get("target_kind") == "telegram_dispatch"
                 and item["body"].get("notification_id"))
    assert asyncio.run(deliver(ack_a), loop_factory=_selector_loop).status_code == 200

    save_update = {"update_id": 15502, "message": {
        "message_id": 62, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(webhook(save_update), loop_factory=_selector_loop).status_code == 200
    save_task = next(item["body"] for item in reversed(task_requests)
                     if item["body"].get("target_kind") == "telegram_dispatch"
                     and not item["body"].get("notification_id"))
    context_id = save_task["context_id"]

    for attempt in range(3):
        if attempt:
            clock.advance()
            save_task = next(item["body"] for item in reversed(task_requests)
                             if item["body"].get("target_kind") == "telegram_dispatch"
                             and item["body"].get("operation_id") == save_task["operation_id"]
                             and not item["body"].get("notification_id"))
        failed_response = asyncio.run(dispatch(save_task), loop_factory=_selector_loop)
        assert failed_response.status_code == 200, failed_response.text
        if attempt < 2:
            assert failed_response.json() == {"status": "noop"}
    assert failed_response.json() == {"status": "processed"}

    session = db.collection("telegram_sessions").document(context_id)
    original = session.collection("operations").document(save_task["operation_id"]).get().to_dict()
    draft_a_id = original["target_draft_id"]
    draft_a_before = session.collection("drafts").document(draft_a_id).get().to_dict()
    assert original["state"] == "failed"
    assert draft_a_before["state"] == "failed" and draft_a_before["dialog_number"] == 1
    assert draft_a_before["ordered_message_ids"] == [61]
    original_intent_id = draft_a_before["immutable_save_intent_id"]
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
    assert not db.collection("dialog_operations").document(save_task["operation_id"]).get().exists
    assert not db.collection(SAVE_INTENTS_COLLECTION).document(original_intent_id).get().exists

    # A newer source starts collecting while dialog #1 remains failed and retryable.
    source_b = "A newer forwarded source that must remain separate."
    forward_b = {"update_id": 15503, "message": {
        "message_id": 63, "date": 1793320020,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 45, "first_name": "Bea"},
                           "date": 1793310020}, "text": source_b,
    }}
    assert asyncio.run(webhook(forward_b), loop_factory=_selector_loop).status_code == 200
    ack_b = next(item["body"] for item in reversed(task_requests)
                 if item["body"].get("target_kind") == "telegram_dispatch"
                 and item["body"].get("notification_id"))
    assert asyncio.run(deliver(ack_b), loop_factory=_selector_loop).status_code == 200
    context_before_retry = session.get().to_dict()
    draft_b_id = context_before_retry["active_draft_id"]
    draft_b_before = session.collection("drafts").document(draft_b_id).get().to_dict()
    assert draft_b_id != draft_a_id and draft_b_before["state"] == "collecting"
    assert draft_b_before["dialog_number"] == 2

    route_response("retry", dialog_number=1)
    retry_text = "Please retry dialog 1."
    retry_update = {"update_id": 15504, "message": {
        "message_id": 64, "date": 1793320030,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7}, "text": retry_text,
    }}
    assert asyncio.run(webhook(retry_update), loop_factory=_selector_loop).status_code == 200
    retry_request_task = next(item["body"] for item in reversed(task_requests)
                              if item["body"].get("target_kind") == "telegram_dispatch"
                              and not item["body"].get("notification_id"))
    request_operation_id = retry_request_task["operation_id"]
    accepted_retry = asyncio.run(dispatch(retry_request_task), loop_factory=_selector_loop)
    assert accepted_retry.status_code == 200, accepted_retry.text
    assert accepted_retry.json() == {"status": "processed"}
    retry_relation = session.collection("retry_requests").document(request_operation_id).get().to_dict()
    assert retry_relation["accepted"] is True
    assert retry_relation["draft_id"] == draft_a_id
    retry_work_task = next(item["body"] for item in reversed(task_requests)
                           if item["body"].get("target_kind") == "telegram_dispatch"
                           and item["body"].get("operation_id") == retry_relation["operation_id"]
                           and not item["body"].get("notification_id"))
    assert "retry_draft_id" not in retry_work_task
    retry_row = session.collection("operations").document(retry_work_task["operation_id"]).get().to_dict()
    retry_request_row = session.collection("operations").document(request_operation_id).get().to_dict()
    assert retry_row["input_ref"] == retry_request_row["input_ref"]
    retry_source_command = session.collection("commands").document(
        retry_row["input_ref"].rsplit("/", 1)[-1]).get().to_dict()
    assert retry_source_command["operation_id"] == request_operation_id
    assert retry_row["route_control"] == {
        "status": "ok", "kind": "retry", "dialog_number": 1,
        "retry_request_operation_id": request_operation_id,
    }

    # The retry metadata comes only after the retry route; complete it with the
    # same native document, index receipt, and finalizer path as an initial save.
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Dialog A saved", "folder_path": ["Client"]}),
        )]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    sdk_fakes.drive.queue("files.create", response={"id": "doc-numbered-retry"})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": "doc-numbered-retry", "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })

    retry_slice = asyncio.run(dispatch(retry_work_task), loop_factory=_selector_loop)
    assert retry_slice.status_code == 200, retry_slice.text
    assert retry_slice.json() == {"status": "noop"}
    retry_row = session.collection("operations").document(retry_work_task["operation_id"]).get().to_dict()
    assert retry_row["state"] == "running"
    assert retry_row["draft_id"] == retry_row["target_draft_id"] == draft_a_id
    assert retry_row["save_operation_id"] == retry_work_task["operation_id"]
    assert retry_row["save_intent_id"] == original_intent_id
    core_operation = db.collection("dialog_operations").document(retry_work_task["operation_id"]).get().to_dict()
    assert core_operation["doc_id"] == "doc-numbered-retry"
    index_task = next(item for item in task_requests
                      if item["url"].endswith("/sync/doc/doc-numbered-retry"))
    fill_events = [event for event in sdk_fakes.trace
                   if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]
    assert len(fill_events) == 1
    reader_text = fill_events[0]["arguments"]["body"]["requests"][0]["insertText"]["text"]
    assert source_a in reader_text and source_b not in reader_text
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: ("folder-client", False, "Dialog A saved"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", lambda _doc_id: reader_text)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc", lambda _doc_id, **_kwargs: reader_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: reader_text)
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        transport = httpx.ASGITransport(app=vector_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector.test") as client:
            return await client.post("/sync/doc/doc-numbered-retry", json=index_task["body"])

    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["status"] == "ok"
    receipt = db.collection("index_completions").document("doc-numbered-retry").collection(
        "generations").document(core_operation["desired_content_hash"]).get().to_dict()
    assert receipt["status"] == "completed"
    wake_kwargs = next(item for item in task_requests
                       if item["body"].get("target_kind") == "telegram_dispatch"
                       and item["body"].get("operation_id") == retry_work_task["operation_id"]
                       and item["body"].get("outbox_id") != retry_work_task.get("outbox_id"))
    wake_record = get_outbox_record(db, wake_kwargs["body"]["outbox_id"])
    assert wake_record is not None
    wake_task = task_payload(wake_record)
    completed = asyncio.run(dispatch(wake_task), loop_factory=_selector_loop)
    assert completed.status_code == 200, completed.text
    assert completed.json() == {"status": "processed"}

    retry_row = session.collection("operations").document(retry_work_task["operation_id"]).get().to_dict()
    draft_a_after = session.collection("drafts").document(draft_a_id).get().to_dict()
    draft_b_after = session.collection("drafts").document(draft_b_id).get().to_dict()
    context_after = session.get().to_dict()
    original_after = session.collection("operations").document(save_task["operation_id"]).get().to_dict()
    assert retry_row["state"] == "completed"
    assert retry_row["doc_id"] == "doc-numbered-retry"
    assert draft_a_after["state"] == "cleaned" and draft_a_after["dialog_number"] == 1
    assert original_after["state"] == "failed"
    assert draft_b_after["state"] == "collecting"
    assert draft_b_after["ordered_message_ids"] == [63]
    assert context_after["active_draft_id"] == draft_b_id
    assert context_after["selected_dialog_number"] == context_before_retry["selected_dialog_number"] == 2
    assert context_after["last_doc_id"] == "doc-numbered-retry"
    draft_b_message = session.collection("drafts").document(draft_b_id).collection(
        "messages").document("63").get().to_dict()
    assert draft_b_message["text"] == source_b
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    final_msgs = [payload for method, payload in sent
                  if method == "sendMessage"
                  and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]
    assert len(final_msgs) == 1
    completed_log = next(row for row in settlement_logs if row.get("outcome") == "completed")
    assert completed_log["operation_id"] == retry_work_task["operation_id"]
    assert completed_log["context_id"] == context_id
    assert completed_log["doc_id"] == "doc-numbered-retry"
    assert completed_log["service"] == "telegram" and completed_log["stage"] == "save"
    metadata_calls = [event for event in sdk_fakes.trace
                      if event["boundary"] == "gemini"
                      and event["operation"] == "generate_content"
                      and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_calls) == 4
    for metadata_call in metadata_calls:
        metadata_prompt = json.loads(metadata_call["arguments"]["contents"])
        assert metadata_prompt["user_request"] == "save this dialog"
        assert metadata_prompt["prepared_dialogue"] == [{"speaker": "Ada", "text": source_a}]
        prompt_text = json.dumps(metadata_prompt, ensure_ascii=False)
        assert source_b not in prompt_text and retry_text not in prompt_text
    requests_before_duplicate = dict(firestore_fake.documents)
    duplicate_request = asyncio.run(dispatch(retry_request_task), loop_factory=_selector_loop)
    duplicate_work = asyncio.run(dispatch(retry_work_task), loop_factory=_selector_loop)
    duplicate_wake = asyncio.run(dispatch(wake_task), loop_factory=_selector_loop)
    assert duplicate_request.status_code == duplicate_work.status_code == duplicate_wake.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    assert len([payload for method, payload in sent
                if method == "sendMessage"
                and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]) == 1
    assert len([path for path in firestore_fake.documents if "/retry_requests/" in path]) == len(
        [path for path in requests_before_duplicate if "/retry_requests/" in path]
    )
    assert len([path for path in firestore_fake.documents
                if path.startswith(f"telegram_sessions/{context_id}/operations/")]) == len(
        [path for path in requests_before_duplicate
         if path.startswith(f"telegram_sessions/{context_id}/operations/")]
    )


@pytest.mark.parametrize("folder_mode", ["explicit", "automatic", "unassigned"])
@pytest.mark.parametrize("final_delivery_failure", [
    "unavailable", "rate_limited", "rate_limited_exhausted",
])
def test_committed_save_recovers_cleanup_after_source_and_request_are_gone(
    firestore_fake, sdk_fakes, monkeypatch, folder_mode, final_delivery_failure,
):
    """Resume genuine core completion after cleanup committed partially and input vanished."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Dialog saved", "folder_path":
                             ["Client"] if folder_mode == "explicit" else None}),
        )]))],
    ))
    if folder_mode == "automatic":
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                text='{"folder_name":"Client","confidence":0.95}',
            )]))],
        ))
    elif folder_mode == "unassigned":
        for _ in range(2):
            sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                    text='{"folder_name":"Client","confidence":0.2}',
                )]))],
            ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    doc_id = "doc-committed-recovery"
    target_folder_id = "folder-unassigned" if folder_mode == "unassigned" else "folder-client"
    if folder_mode == "unassigned":
        sdk_fakes.drive.queue("files.list", response={"files": []})
        sdk_fakes.drive.queue("files.create", response={"id": target_folder_id})
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": [target_folder_id], "modifiedTime": modified,
        })

    source = "The forwarded source that must survive until completion authorization."
    forward = {"update_id": 17701, "message": {
        "message_id": 71, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": source,
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack = next(item["body"] for item in runtime.task_requests
               if item["body"].get("target_kind") == "telegram_dispatch"
               and item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    save_update = {"update_id": 17702, "message": {
        "message_id": 72, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    work = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    for _ in range(8):
        work = next(item["body"] for item in reversed(runtime.task_requests)
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and item["body"].get("operation_id") == work["operation_id"]
                    and not item["body"].get("notification_id"))
        result = asyncio.run(runtime.dispatch(work), loop_factory=_selector_loop)
        assert result.status_code == 200, result.text
        core = runtime.db.collection("dialog_operations").document(work["operation_id"]).get().to_dict()
        if core and core.get("doc_id") == doc_id:
            break
        runtime.clock.advance()
    else:
        pytest.fail("real save did not reach core completion")
    assert core["doc_id"] == doc_id
    assert core["selected_folder_id"] == target_folder_id
    if folder_mode == "unassigned":
        folder_rows = runtime.db.collection("folders").document(target_folder_id).get().to_dict()
        assert folder_rows["name"] == "_unassigned"
    index_task = next(item for item in runtime.task_requests
                      if item["url"].endswith(f"/sync/doc/{doc_id}"))
    reader_text = next(event["arguments"]["body"]["requests"][0]["insertText"]["text"]
                       for event in sdk_fakes.trace
                       if event["boundary"] == "docs"
                       and event["operation"] == "documents.batchUpdate")
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: (target_folder_id, False, "Dialog saved"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", lambda _doc_id: reader_text)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc", lambda _doc_id, **_kwargs: reader_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: reader_text)
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        transport = httpx.ASGITransport(app=vector_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector.test") as client:
            return await client.post(f"/sync/doc/{doc_id}", json=index_task["body"])

    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200 and indexed.json()["status"] == "ok"
    wake_request = next(item["body"] for item in runtime.task_requests
                        if item["body"].get("target_kind") == "telegram_dispatch"
                        and item["body"].get("operation_id") == work["operation_id"]
                        and item["body"].get("outbox_id") != work.get("outbox_id"))
    wake_record = get_outbox_record(runtime.db, wake_request["outbox_id"])
    wake = task_payload(wake_record)
    context_id = work["context_id"]
    session = runtime.db.collection("telegram_sessions").document(context_id)
    row = session.collection("operations").document(work["operation_id"]).get().to_dict()
    draft_id = row["target_draft_id"]
    draft_before = session.collection("drafts").document(draft_id).get().to_dict()
    draft_path = f"telegram_sessions/{context_id}/drafts/{draft_id}"
    source_path = f"telegram_sessions/{context_id}/drafts/{draft_id}/messages/71"

    firestore_fake.fail_next("commit", path=draft_path, phase="before")
    missing_local = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert missing_local.status_code == 503
    core_after_local_failure = runtime.db.collection("dialog_operations").document(
        work["operation_id"]).get().to_dict()
    assert core_after_local_failure["state"] == "completed"
    assert core_after_local_failure["doc_id"] == doc_id
    draft_after_local_failure = session.collection("drafts").document(draft_id).get().to_dict()
    assert "completion_certificate" not in draft_after_local_failure
    assert draft_after_local_failure.get("completion_certificate_ref") is None
    assert session.collection("drafts").document(draft_id).collection(
        "messages").document("71").get().exists
    session.collection("operations").document(work["operation_id"]).collection(
        "agent_requests").document("request").delete()
    status_task = next(item["body"] for item in runtime.task_requests
                       if item["body"].get("target_kind") == "telegram_status_check")
    runtime.app.state.telegram_status_check_worker.clock = runtime.clock
    runtime.clock.value += timedelta(seconds=60)
    status_response = asyncio.run(
        runtime.post("/internal/status-check", status_task), loop_factory=_selector_loop,
    )
    assert status_response.status_code == 200, status_response.text
    assert session.collection("operations").document(work["operation_id"]).get().to_dict()["state"] == "running"
    interim_notifications = [
        snapshot.to_dict() or {} for snapshot in runtime.db.collection("notification_outbox").stream()
    ]
    assert not any(
        row.get("kind") in {"final", "error"} and row.get("operation_id") == work["operation_id"]
        for row in interim_notifications
    )
    from services.connectors.telegram.queue import resolve_current_head_wakeup
    canonical_wake = resolve_current_head_wakeup(runtime.db, context_id)
    if canonical_wake is not None:
        wake = task_payload(canonical_wake)

    firestore_fake.fail_next("batch_commit", path=source_path, phase="after")
    partial = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert partial.status_code in {200, 503}
    local_before = session.collection("drafts").document(draft_id).get().to_dict()["completion_certificate"]
    assert local_before["receipt_digest"]
    if folder_mode == "unassigned":
        assert local_before["classification_outcome"] == "unassigned"
    intent_id = draft_before["immutable_save_intent_id"]
    intent = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(intent_id).get().to_dict()
    committed_grant = intent["completion_certificate"]
    assert committed_grant["doc_id"] == doc_id

    # The provider-independent index can advance the same generation after the
    # core grant; recovery must continue to trust the pinned receipt identity.
    receipt_ref = runtime.db.collection("index_completions").document(doc_id).collection(
        "generations").document(committed_grant["content_hash"])
    receipt = receipt_ref.get().to_dict()
    receipt_ref.update({"index_fence": receipt["index_fence"] + 1,
                        "resource_epoch": receipt["resource_epoch"] + 1})
    index_ref = runtime.db.collection("transcript_index").document(doc_id)
    index_row = index_ref.get().to_dict()
    index_ref.update({"status": "synced", "content_hash": "later-generation-hash",
                      "version": f"{index_row['version']}-later",
                      "index_fence": receipt["index_fence"] + 1,
                      "resource_epoch": receipt["resource_epoch"] + 1})
    source_snapshot = session.collection("drafts").document(draft_id).collection(
        "messages").document("71").get()
    assert not source_snapshot.exists

    if final_delivery_failure.startswith("rate_limited"):
        for _ in range(3 if final_delivery_failure == "rate_limited_exhausted" else 1):
            runtime.bot_responses.append(httpx.Response(429, json={
                "ok": False, "error_code": 429, "parameters": {"retry_after": 7},
            }))
    else:
        runtime.bot_responses.append(httpx.Response(503, json={"error": "temporarily unavailable"}))
    resumed = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert resumed.status_code in {200, 503}, resumed.text
    draft_after_first_retry = session.collection("drafts").document(draft_id).get().to_dict()
    assert draft_after_first_retry["state"] == "cleaned"
    assert draft_after_first_retry["completion_certificate"] == local_before
    first_row = session.collection("operations").document(work["operation_id"]).get().to_dict()
    if first_row["state"] != "completed":
        from services.connectors.telegram.queue import resolve_current_head_wakeup
        current_wake = resolve_current_head_wakeup(runtime.db, context_id)
        if current_wake is not None:
            wake = task_payload(current_wake)
    from services.connectors.telegram.repository import TelegramRepository
    notification_repo = TelegramRepository(runtime.db)
    if final_delivery_failure.startswith("rate_limited"):
        final_id = first_row["notification_id"]
        first_final = notification_repo.get_notification(final_id)
        assert first_final["last_outcome"] == "rate_limited"
        assert first_final["send_attempts_started"] == 1
        early_duplicate = asyncio.run(runtime.app.state.telegram_notification_publisher.publish(final_id),
                                      loop_factory=_selector_loop)
        assert early_duplicate == "busy"
        assert notification_repo.get_notification(final_id)["send_attempts_started"] == 1
        runtime.clock.value += timedelta(seconds=7)
        if final_delivery_failure == "rate_limited_exhausted":
            assert asyncio.run(runtime.app.state.telegram_notification_publisher.publish(final_id),
                               loop_factory=_selector_loop) == "retry"
            runtime.clock.value += timedelta(seconds=7)
            assert asyncio.run(runtime.app.state.telegram_notification_publisher.publish(final_id),
                               loop_factory=_selector_loop) == "failed"
            terminal = notification_repo.get_notification(final_id)
            assert terminal["state"] == "failed" and terminal["send_attempts_started"] == 3
            status_response = asyncio.run(
                runtime.post("/internal/status-check", status_task), loop_factory=_selector_loop,
            )
            assert status_response.status_code == 200
            assert notification_repo.get_notification(final_id)["send_attempts_started"] == 3
            terminal_rows = [
                snapshot.to_dict() or {}
                for snapshot in runtime.db.collection("notification_outbox").stream()
                if (snapshot.to_dict() or {}).get("operation_id") == work["operation_id"]
                and (snapshot.to_dict() or {}).get("kind") in {"final", "error"}
            ]
            assert len(terminal_rows) == 1
            assert terminal_rows[0]["notification_id"] == final_id
        else:
            assert asyncio.run(runtime.app.state.telegram_notification_publisher.publish(final_id),
                               loop_factory=_selector_loop) == "delivered"
    else:
        runtime.clock.advance()
    delivered = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert delivered.status_code == 200, delivered.text
    row_after = session.collection("operations").document(work["operation_id"]).get().to_dict()
    draft_after = session.collection("drafts").document(draft_id).get().to_dict()
    if final_delivery_failure == "rate_limited_exhausted":
        assert row_after["state"] == "failed"
        core_after = runtime.db.collection("dialog_operations").document(work["operation_id"]).get().to_dict()
        assert core_after["state"] == "completed" and core_after["doc_id"] == doc_id
    else:
        assert row_after["state"] == "completed" and row_after["doc_id"] == doc_id
    if final_delivery_failure == "rate_limited":
        final_after = notification_repo.get_notification(final_id)
        assert final_after["state"] == "delivered"
        assert final_after["send_attempts_started"] == 2
    assert draft_after["state"] == "cleaned"
    assert draft_after["completion_certificate"] == local_before
    assert draft_after["completion_certificate"]["classification_outcome"] == local_before.get(
        "classification_outcome")
    assert session.get().to_dict()["last_doc_id"] == doc_id
    accepted_finals = [payload for (method, payload), (response_method, status, ok)
                       in zip(runtime.sent, runtime.bot_results)
                       if method == response_method == "sendMessage" and status == 200 and ok
                       and "Dialog #1: Saved and indexed." in payload.get("text", "")]
    assert len(accepted_finals) == (0 if final_delivery_failure == "rate_limited_exhausted" else 1)
    assert draft_after["dialog_number"] == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == (
        2 if folder_mode == "unassigned" else 1
    )
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]) == 1
    metadata_calls = [event for event in sdk_fakes.trace
                      if event["boundary"] == "gemini"
                      and event["operation"] == "generate_content"
                      and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_calls) == 1

    from core.integrations.firestore.operations import get_operation
    from services.connectors.telegram.app.terminal_notifications import (
        _load_completion_certificate,
    )
    core_operation = get_operation(runtime.db, work["operation_id"])
    assert _load_completion_certificate(
        runtime.db, core_operation, work["operation_id"], draft_after,
    ) is not None
    foreign_local = dict(draft_after)
    foreign_local["completion_certificate"] = {
        **draft_after["completion_certificate"], "context_id": "foreign-context",
    }
    assert _load_completion_certificate(
        runtime.db, core_operation, work["operation_id"], foreign_local,
    ) is None
    intent_ref = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(intent_id)
    valid_intent = intent_ref.get().to_dict()
    valid_core = runtime.db.collection("dialog_operations").document(
        work["operation_id"])
    intent_ref.update({"completion_certificate": {
        **valid_intent["completion_certificate"], "certificate_id": "",
    }})
    assert _load_completion_certificate(
        runtime.db, core_operation, work["operation_id"], draft_after,
    ) is None
    intent_ref.set(valid_intent)
    assert valid_core.get().to_dict()["cleanup_authorization"] == intent.get("cleanup_authorization")


def test_recovery_reads_classification_outcome_from_context_stage_checkpoint(firestore_fake):
    """Restore the classification note from its durable Telegram stage result."""
    from core.dialogs.classification import FolderSelectionResult
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from services.connectors.telegram.save import TelegramClassifiedDialog, _checkpoint_classified
    from services.connectors.telegram.repository import TelegramRepository

    client = firestore_fake.client_class()
    context_id, operation_id = "42_11", "save-recovery-classification"
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner="test-owner", fence=1, clock=lambda: datetime.now(timezone.utc),
    )
    checkpoint = _checkpoint_classified(TelegramClassifiedDialog(
        payload={"title": "Already classified"},
        selection=FolderSelectionResult("folder-a", ("speaker-a", "speaker-b")),
    ))
    stored = ledger._store_payload("save-stage-classification", checkpoint)
    stage_ref = (client.collection("telegram_sessions").document(context_id)
                 .collection("operations").document(operation_id)
                 .collection("stages").document("telegram.classification"))
    stage_ref.set({"stage_key": "telegram.classification", "state": "completed",
                   "result": {"__telegram_save_stage_payload__": stored}})

    runtime = TelegramSaveRuntime(repository=TelegramRepository(client))
    assert runtime._recovered_classification_outcome(ledger, operation_id) == "ambiguous"


def test_temporary_payload_cleanup_resumes_part_deletion_and_rejects_missing_authority(
    firestore_fake,
):
    """A partial multipart purge resumes from tombstones; an uncertified purge is inert."""
    import hashlib
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository
    from services.connectors.telegram.runtime.save_cleanup import TemporarySavePayloadCleanup

    client = firestore_fake.client_class()
    context_id, draft_id, operation_id, intent_id = "42_11", "draft-cleanup", "save-cleanup", "save-cleanup"
    now = datetime.now(timezone.utc)
    digest = "d" * 64
    manifest = {"draft_id": draft_id, "save_operation_id": operation_id,
                "manifest_digest": digest, "ordered_message_ids": [77]}
    session = client.collection("telegram_sessions").document(context_id)
    session.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": "worker", "fence": 1, "lock_expires_at": now + timedelta(minutes=5),
    })
    op_ref = session.collection("operations").document(operation_id)
    op_ref.set({
        "operation_id": operation_id, "state": "running", "target_draft_id": draft_id,
        "draft_id": draft_id, "save_operation_id": operation_id, "save_intent_id": intent_id,
        "route_control": {"status": "ok", "kind": "save"},
    })
    draft_ref = session.collection("drafts").document(draft_id)
    cert = {
        "context_id": context_id, "draft_id": draft_id, "manifest_digest": digest,
        "save_intent_id": intent_id, "operation_id": operation_id,
    }
    draft_ref.set({
        **manifest, "cleanup_state": "authorized",
        "immutable_save_intent_id": intent_id,
        "cleanup_authorization": {
            "save_intent_id": intent_id, "operation_id": operation_id,
            "origin_binding": {"context_id": context_id, "draft_id": draft_id,
                               "manifest_digest": digest},
        },
    })
    draft_ref.collection("save_manifests").document(operation_id).set(manifest)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner="worker", fence=1, clock=lambda: now,
    )
    repository = FirestoreTelegramSaveRepository(
        TelegramRepository(client), context_id=context_id, operation_id=operation_id,
        owner="worker", fence=1, clock=lambda: now, ledger=ledger,
    )
    descriptor = asyncio.run(repository.store_retry_stage_payload(
        operation_id, "analysis", {"text": "x" * 700_000},
    ), loop_factory=_selector_loop)
    assert "_input_ref" in descriptor
    op_ref.collection("stages").document("telegram.analysis").set({
        "stage_key": "telegram.analysis", "state": "completed",
        "result": {"__telegram_save_stage_payload__": descriptor},
    })
    conversation_id = "telegram-global-" + hashlib.sha256(
        f"{context_id}\0{operation_id}".encode(),
    ).hexdigest()
    ledger.save_request(conversation_id, operation_id, "original private save request")
    request_ref = op_ref.collection("agent_requests").document("request")
    assert request_ref.get().to_dict()["request"]["_inline"] == "original private save request"

    denied = TemporarySavePayloadCleanup(client, context_id=context_id)
    assert denied.cleanup(cert) is False
    assert op_ref.collection("stages").document("telegram.analysis").get().to_dict()[
        "result"]["__telegram_save_stage_payload__"] == descriptor

    cleaner = TemporarySavePayloadCleanup(
        client, context_id=context_id, batch_size=1,
        authorize=lambda ctx, candidate: ctx == context_id and candidate == cert,
    )
    part_path = (descriptor["_input_ref"].replace("/manifest", "/part-000000"))
    firestore_fake.fail_next("batch_commit", path=part_path, phase="after")
    with pytest.raises(RuntimeError, match="injected after batch_commit failure"):
        cleaner.cleanup(cert)
    tombstone = op_ref.collection("stages").document("telegram.analysis").get().to_dict()["result"]
    assert tombstone["purged"] is True and tombstone["purged_input_ref"] == descriptor["_input_ref"]
    assert request_ref.get().to_dict()["request"]["purged"] is True
    with pytest.raises(PermissionError, match="allocation lost its queue lease"):
        asyncio.run(repository.store_retry_stage_payload(
            operation_id, "analysis", {"text": "x" * 700_000},
        ), loop_factory=_selector_loop)
    allocation_mid_cleanup = next(iter(op_ref.collection("save_payloads").stream())).to_dict()
    assert allocation_mid_cleanup["cleanup_state"] == "purging"
    assert cleaner.cleanup(cert) is True
    collection = client.collection("dialog_operations").document(
        descriptor["_input_ref"].split("/")[1],
    ).collection("input_parts")
    assert list(collection.where("index", ">=", 0).limit(10).stream()) == []
    assert collection.document("manifest").get().exists
    allocation_docs = list(op_ref.collection("save_payloads").stream())
    assert len(allocation_docs) == 1
    assert allocation_docs[0].to_dict()["cleanup_state"] == "purged"
    allocation_ref = allocation_docs[0].reference
    purged_allocation = allocation_docs[0].to_dict()
    allocation_ref.update({"unexpected_raw_field": "must not survive"})
    assert cleaner.cleanup(cert) is False
    allocation_ref.set(purged_allocation, merge=False)
    stage_ref = op_ref.collection("stages").document("telegram.analysis")
    clean_tombstone = stage_ref.get().to_dict()["result"]
    stage_ref.update({"result": {**clean_tombstone, "_inline": "must not survive"}})
    assert cleaner.cleanup(cert) is False
    stage_ref.update({"result": clean_tombstone})
    from core.integrations.firestore.input_parts import read_input_parts, write_input_parts
    foreign_ref = "dialog_operations/foreign-temporary-payload/input_parts/manifest"
    foreign_payload = b"foreign sealed payload must remain"
    write_input_parts(client, foreign_ref, foreign_payload)
    clean_request_tombstone = request_ref.get().to_dict()["request"]
    inline_value = "valid inline digest with a foreign reference attached"
    inline_bytes = ledger._json_bytes(inline_value)
    hybrid = {
        "_inline": inline_value,
        "_input_ref": foreign_ref,
        "sha256": hashlib.sha256(inline_bytes).hexdigest(),
        "byte_length": len(inline_bytes),
        "max_part_bytes": 512 * 1024,
    }
    stage_ref.update({"result": {"__telegram_save_stage_payload__": hybrid}})
    request_ref.update({"request": hybrid})
    assert cleaner.cleanup(cert) is False
    assert read_input_parts(client, foreign_ref) == foreign_payload
    stage_ref.update({"result": clean_tombstone})
    request_ref.update({"request": clean_request_tombstone})
    assert cleaner.cleanup(cert) is True


def test_cleanup_finds_fenced_allocation_when_stage_descriptor_was_never_published(
    firestore_fake,
):
    """A write that completes before stage publication remains discoverable after certification."""
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository
    from services.connectors.telegram.runtime.save_cleanup import TemporarySavePayloadCleanup

    client = firestore_fake.client_class()
    context_id, draft_id, operation_id, intent_id = "42_11", "draft-orphan", "save-orphan", "save-orphan"
    now = datetime.now(timezone.utc)
    digest = "e" * 64
    manifest = {"draft_id": draft_id, "save_operation_id": operation_id,
                "manifest_digest": digest, "ordered_message_ids": [88]}
    session = client.collection("telegram_sessions").document(context_id)
    session.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": "worker", "fence": 4, "lock_expires_at": now + timedelta(minutes=5),
    })
    op_ref = session.collection("operations").document(operation_id)
    op_ref.set({
        "operation_id": operation_id, "state": "running", "target_draft_id": draft_id,
        "draft_id": draft_id, "save_operation_id": operation_id, "save_intent_id": intent_id,
        "route_control": {"status": "ok", "kind": "save"},
    })
    draft_ref = session.collection("drafts").document(draft_id)
    cert = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest,
            "save_intent_id": intent_id, "operation_id": operation_id}
    draft_ref.set({
        **manifest, "cleanup_state": "authorized",
        "immutable_save_intent_id": intent_id,
        "cleanup_authorization": {
            "save_intent_id": intent_id, "operation_id": operation_id,
            "origin_binding": {"context_id": context_id, "draft_id": draft_id,
                               "manifest_digest": digest},
        },
    })
    draft_ref.collection("save_manifests").document(operation_id).set(manifest)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner="worker", fence=4, clock=lambda: now,
    )
    repository = FirestoreTelegramSaveRepository(
        TelegramRepository(client), context_id=context_id, operation_id=operation_id,
        owner="worker", fence=4, clock=lambda: now, ledger=ledger,
    )
    value = {"text": "y" * 700_000}
    payload = ledger._json_bytes(value)
    allocation_before_write = repository._stage_payload_allocation(
        "save-stage-analysis", hashlib.sha256(payload).hexdigest(), len(payload),
    )
    part_path = allocation_before_write["input_ref"].replace("/manifest", "/part-000000")
    firestore_fake.fail_next("commit", path=part_path, phase="after")
    with pytest.raises(RuntimeError, match="injected after commit failure"):
        asyncio.run(repository.store_retry_stage_payload(
            operation_id, "analysis", value,
        ), loop_factory=_selector_loop)
    assert list(op_ref.collection("stages").stream()) == []
    allocation = next(iter(op_ref.collection("save_payloads").stream())).to_dict()
    assert allocation["cleanup_state"] == "allocated"
    assert allocation["input_ref"] == allocation_before_write["input_ref"]
    parts = client.collection("dialog_operations").document(
        allocation["input_ref"].split("/")[1],
    ).collection("input_parts")
    assert len(list(parts.where("index", ">=", 0).limit(10).stream())) == 1
    assert not parts.document("manifest").get().exists

    # This component test isolates the storage boundary. Production grant validity
    # is exercised by the real default-app completion integration above.
    cleaner = TemporarySavePayloadCleanup(
        client, context_id=context_id,
        authorize=lambda ctx, candidate: ctx == context_id and candidate == cert,
    )
    assert cleaner.cleanup(cert) is True
    allocation_after = next(iter(op_ref.collection("save_payloads").stream())).to_dict()
    assert allocation_after["cleanup_state"] == "purged"
    assert list(parts.where("index", ">=", 0).limit(10).stream()) == []
    assert not op_ref.collection("stages").stream()


@pytest.mark.parametrize("takeover", [True, False], ids=["takeover", "expiry"])
def test_large_ledger_input_write_rejects_stale_lease_at_transaction_commit(
    firestore_fake, monkeypatch, takeover,
):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    context_id, operation_id = "42_11", "ledger-takeover-race"
    now = datetime.now(timezone.utc)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": "old-owner", "fence": 1,
        "lock_expires_at": now + timedelta(minutes=5),
    })
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "running",
    })
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner="old-owner", fence=1, clock=lambda: now,
    )
    payload = ledger._json_bytes({"text": "x" * 8_000})
    digest = hashlib.sha256(payload).hexdigest()

    original_commit = FakeTransaction._commit
    injected = {"done": False}

    def takeover_before_commit(transaction):
        if not injected["done"] and transaction._writes:
            injected["done"] = True
            if takeover:
                context_ref.update({
                    "lock_owner": "new-owner", "fence": 2,
                    "lock_expires_at": now + timedelta(minutes=10),
                })
            else:
                context_ref.update({"lock_expires_at": now - timedelta(seconds=1)})
        return original_commit(transaction)

    monkeypatch.setattr(FakeTransaction, "_commit", takeover_before_commit)
    with pytest.raises(PermissionError, match="lease is stale"):
        ledger._store_payload("model", {"text": "x" * 8_000})

    context_after = context_ref.get().to_dict()
    assert context_after["lock_owner"] == ("new-owner" if takeover else "old-owner")
    assert context_after["fence"] == (2 if takeover else 1)
    parts = client.collection("dialog_operations").document(
        "telegram-ledger-" + hashlib.sha256(
            f"{context_id}\0{operation_id}\0model\0{digest}".encode()
        ).hexdigest(),
    ).collection("input_parts")
    assert list(parts.stream()) == []


def test_retry_stage_part_write_rejects_allocation_cleanup_race(firestore_fake, monkeypatch):
    import hashlib
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.save_repository import FirestoreTelegramSaveRepository
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    context_id, draft_id, operation_id, intent_id = "42_11", "draft-race", "save-race", "save-race"
    now = datetime.now(timezone.utc)
    digest = "f" * 64
    manifest = {"draft_id": draft_id, "save_operation_id": operation_id,
                "manifest_digest": digest, "ordered_message_ids": [99]}
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": "worker", "fence": 1,
        "lock_expires_at": now + timedelta(minutes=5),
    })
    operation_ref = context_ref.collection("operations").document(operation_id)
    operation_ref.set({
        "operation_id": operation_id, "state": "running", "target_draft_id": draft_id,
        "draft_id": draft_id, "save_operation_id": operation_id, "save_intent_id": intent_id,
    })
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft_ref.set({**manifest, "immutable_save_intent_id": intent_id})
    draft_ref.collection("save_manifests").document(operation_id).set(manifest)
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner="worker", fence=1, clock=lambda: now,
    )
    repository = FirestoreTelegramSaveRepository(
        TelegramRepository(client), context_id=context_id, operation_id=operation_id,
        owner="worker", fence=1, clock=lambda: now, ledger=ledger,
    )
    value = {"text": "z" * 700_000}
    payload = ledger._json_bytes(value)
    allocation_data = repository._stage_payload_allocation(
        "save-stage-analysis", hashlib.sha256(payload).hexdigest(), len(payload),
    )
    allocation_ref = operation_ref.collection("save_payloads").document(
        allocation_data["allocation_id"],
    )
    part_path = allocation_data["input_ref"].replace("/manifest", "/part-000000")
    original_commit = FakeTransaction._commit
    cleanup = {"done": False}

    def cleanup_before_part_commit(transaction):
        if (not cleanup["done"]
                and any(ref.path == part_path for _, ref, _, _ in transaction._writes)):
            cleanup["done"] = True
            allocation_ref.update({"cleanup_state": "purging"})
        return original_commit(transaction)

    monkeypatch.setattr(FakeTransaction, "_commit", cleanup_before_part_commit)
    with pytest.raises(PermissionError, match="allocation is no longer writable"):
        asyncio.run(repository.store_retry_stage_payload(
            operation_id, "analysis", value,
        ), loop_factory=_selector_loop)

    assert allocation_ref.get().to_dict()["cleanup_state"] == "purging"
    parts = client.collection("dialog_operations").document(
        allocation_data["input_ref"].split("/")[1],
    ).collection("input_parts")
    assert list(parts.where("index", ">=", 0).stream()) == []
    assert not parts.document("manifest").get().exists


def _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch, *,
                                      heartbeat_interval_seconds=None):
    """Build the real default Telegram queue app with only external boundaries faked."""
    from google.genai import types
    from services.connectors.telegram.app.main import create_app
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.runtime.factory import build_default_queue_worker
    from core.integrations.firestore.folders import upsert_folder

    class FakeClock:
        def __init__(self):
            self.value = datetime.now(timezone.utc)

        def __call__(self):
            return self.value

        def advance(self):
            self.value += timedelta(minutes=10)

    clock = FakeClock()
    db = firestore_fake.client_class()
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: db)
    monkeypatch.setattr("services.connectors.telegram.app.main._allowed_user_ids", lambda: {7})
    for key, value in {
        "TELEGRAM_BOT_TOKEN": "123456:synthetic-token",
        "TELEGRAM_CONTROL_QUEUE": "pytest-telegram-control-queue",
        "TELEGRAM_CONTROL_URL": "https://telegram.pytest.invalid/internal/dispatch",
        "TELEGRAM_DELIVERY_URL": "https://telegram.pytest.invalid/internal/deliver",
        "CLOUD_TASKS_OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
        "OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
        "DIALOG_OPERATIONS_QUEUE": "pytest-dialog-operations-queue",
        "VECTOR_SYNC_URL": "https://vector-sync.pytest.invalid",
        "ROOT_FOLDER_URL": "https://drive.google.com/drive/folders/pytest-synthetic-root",
        "GOOGLE_REFRESH_TOKEN": "pytest-synthetic-refresh",
        "GOOGLE_CLIENT_ID": "pytest-synthetic-client",
        "GOOGLE_CLIENT_SECRET": "pytest-synthetic-secret",
    }.items():
        monkeypatch.setenv(key, value)

    original_factory = build_default_queue_worker
    workers = []

    def build_clock_configured_worker(client):
        worker = original_factory(client)
        worker.clock = clock
        worker.processor.clock = clock
        worker.processor.save_handler.clock = clock
        if heartbeat_interval_seconds is not None:
            worker.heartbeat_interval_seconds = heartbeat_interval_seconds
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        "services.connectors.telegram.runtime.factory.build_default_queue_worker",
        build_clock_configured_worker,
    )
    task_requests = []

    def enqueue_fake(**kwargs):
        task_requests.append(kwargs)

    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_fake)
    monkeypatch.setattr("core.operations.outbox.enqueue_task", enqueue_fake)
    sent = []
    bot_responses = []
    bot_results = []
    audio_responses = {}
    get_file_responses = {}
    audio_requests = []
    audio_request_hook = {"callback": None}
    audio_response_hook = {"callback": None}

    async def before_telegram_request(request):
        callback = audio_request_hook["callback"]
        is_audio_rpc = request.method == "GET" or request.url.path.endswith("/getFile")
        if callback is not None and is_audio_rpc and inspect.iscoroutinefunction(callback):
            await callback(request)

    async def after_telegram_response(response):
        callback = audio_response_hook["callback"]
        request = response.request
        is_audio_rpc = request.method == "GET" or request.url.path.endswith("/getFile")
        if callback is not None and is_audio_rpc:
            result = callback(response)
            if inspect.isawaitable(result):
                await result

    def bot_response(request):
        if request.method == "GET" or request.url.path.endswith("/getFile"):
            audio_requests.append(request)
            callback = audio_request_hook["callback"]
            if callback is not None and not inspect.iscoroutinefunction(callback):
                callback(request)
            if request.url.path.endswith("/getFile"):
                file_id = json.loads(request.content).get("file_id", "synthetic-audio-file")
                if isinstance(get_file_responses, dict):
                    if file_id in get_file_responses:
                        return get_file_responses.pop(file_id)
                if get_file_responses:
                    return get_file_responses.pop(0)
                return httpx.Response(200, json={"ok": True, "result": {
                    "file_id": file_id, "file_path": "voice/synthetic.ogg",
                }})
            if audio_responses:
                if request.url.path in audio_responses:
                    return audio_responses.pop(request.url.path)
                if isinstance(audio_responses, list):
                    return audio_responses.pop(0)
            return httpx.Response(200, content=b"synthetic-audio", headers={"content-type": "audio/ogg"})
        payload = json.loads(request.content)
        sent.append((request.url.path.rsplit("/", 1)[-1], payload))
        if bot_responses:
            response = bot_responses.pop(0)
        else:
            response = httpx.Response(200, json={"ok": True, "result": {"message_id": 900 + len(sent)}})
        bot_results.append((request.url.path.rsplit("/", 1)[-1], response.status_code,
                            response.json().get("ok") is True))
        return response

    original_init = TelegramClient.__init__

    def telegram_client_init(self, token, *, http_client=None, timeout=30.0):
        return original_init(self, token, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(bot_response),
            event_hooks={"request": [before_telegram_request],
                         "response": [after_telegram_response]},
        ), timeout=timeout)

    monkeypatch.setattr(TelegramClient, "__init__", telegram_client_init)
    upsert_folder("pytest-synthetic-root", "Root", None, db)
    upsert_folder("folder-client", "Client", "pytest-synthetic-root", db)
    app = create_app(
        webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
        oidc_verifier=lambda _token: type("Identity", (), {"authorized": True})(),
    )
    original_notification_runtime = __import__(
        "services.connectors.telegram.app.main", fromlist=["_notification_runtime"],
    )._notification_runtime

    def build_clocked_notification_runtime():
        publisher = original_notification_runtime()
        publisher.clock = clock
        return publisher

    monkeypatch.setattr(
        "services.connectors.telegram.app.main._notification_runtime",
        build_clocked_notification_runtime,
    )

    async def post(path, payload, *, secret=False):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            headers = ({"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"}
                       if secret else {"Authorization": "Bearer fake-task"})
            return await client.post(path, headers=headers, json=payload)

    def route_response(kind, **values):
        route = {"kind": kind, "target_ref": None, "target_operation_id": None,
                 "dialog_number": None, "result_refs": [], "substantive_text": None,
                 "tool_choice": None, "certainty": "certain", **values}
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                function_call=types.FunctionCall(name="route_command", args=route),
            )]))],
        ))

    return SimpleNamespace(
        app=app, clock=clock, db=db, task_requests=task_requests, sent=sent, post=post,
        worker=lambda: workers[-1] if workers else None,
        bot_responses=bot_responses, bot_results=bot_results,
        audio_responses=audio_responses, audio_requests=audio_requests,
        get_file_responses=get_file_responses,
        audio_request_hook=audio_request_hook,
        audio_response_hook=audio_response_hook,
        webhook=lambda update: post("/telegram/webhook", update, secret=True),
        dispatch=lambda task: post("/internal/dispatch", task),
        deliver=lambda task: post("/internal/deliver", task), route_response=route_response,
    )


def test_default_worker_audio_transcription_failure_uses_exact_stage_evidence(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """Three real audio SDK failures settle the per-message stage once and safely."""
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    secret = "synthetic-gemini-error-secret"
    for _ in range(3):
        sdk_fakes.gemini.queue("generate_content", error=RuntimeError(secret))
    source = {
        "update_id": 16601,
        "message": {
            "message_id": 71, "date": 1793320000,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
            "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                               "date": 1793310000},
            "voice": {"file_id": "synthetic-audio-file", "mime_type": "audio/ogg"},
        },
    }
    assert asyncio.run(runtime.webhook(source), loop_factory=_selector_loop).status_code == 200
    ack = next(item["body"] for item in runtime.task_requests if item["body"].get("notification_id"))
    ack_response = asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop)
    assert ack_response.status_code == 200, (ack_response.text, runtime.sent, runtime.bot_results)
    save_update = {"update_id": 16602, "message": {
        "message_id": 72, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    for attempt in range(3):
        if attempt:
            runtime.clock.advance()
            task = runtime.task_requests[-1]["body"]
        response = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
        assert response.status_code == 200, response.text
    operation = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                 .collection("operations").document(task["operation_id"]).get().to_dict())
    draft = (runtime.db.collection("telegram_sessions").document(task["context_id"])
             .collection("drafts").document(operation["target_draft_id"]).get().to_dict())
    context_ref = runtime.db.collection("telegram_sessions").document(task["context_id"])
    stage = QueueOperationRetryStore(runtime.db, context_ref).get_stage(
        operation_id=task["operation_id"], stage_key="telegram.audio.71",
    )
    assert stage.failures == stage.attempts_started == 3
    assert operation["state"] == "failed" and draft["state"] == "failed"
    assert draft["ordered_message_ids"] == [71]
    assert (runtime.db.collection("telegram_sessions").document(task["context_id"])
            .collection("drafts").document(draft["draft_id"])
            .collection("messages").document("71").get().exists)
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]
    assert not [event for event in sdk_fakes.trace if event["boundary"] == "docs"]
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"
                and "Infer a concise dialog title" in str(event["arguments"])]
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == 4
    assert len([request for request in runtime.audio_requests if request.method == "GET"]) == 3
    error_messages = [payload["text"] for method, payload in runtime.sent
                      if method == "sendMessage" and "Error in gemini/transcription" in payload.get("text", "")]
    assert len(error_messages) == 1
    error_text = error_messages[0]
    assert task["operation_id"] in error_text and "failures=3" in error_text
    assert "administrator" in error_text and "logs" in error_text
    assert secret not in error_text and secret not in caplog.text
    duplicate = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
    assert duplicate.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"]) == 4


@pytest.mark.parametrize("pause_at", ["get_file", "download"])
def test_default_worker_parent_takeover_at_audio_http_boundary_fences_next_rpc(
    firestore_fake, sdk_fakes, monkeypatch, pause_at,
):
    """A takeover during either real Telegram audio HTTP request stops stale work."""
    import threading
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    source = {"update_id": 16611, "message": {
        "message_id": 81, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000},
        "voice": {"file_id": "fenced-audio-file", "mime_type": "audio/ogg"},
    }}
    assert asyncio.run(runtime.webhook(source), loop_factory=_selector_loop).status_code == 200
    ack = next(item["body"] for item in runtime.task_requests if item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    save_update = {"update_id": 16612, "message": {
        "message_id": 82, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    entered, release = threading.Event(), threading.Event()

    def pause_at_http(request):
        is_get_file = request.url.path.endswith("/getFile")
        is_download = request.method == "GET"
        if ((pause_at == "get_file" and is_get_file)
                or (pause_at == "download" and is_download)):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test audio HTTP release timed out")

    runtime.audio_request_hook["callback"] = pause_at_http
    responses = []

    def dispatch_from_thread():
        responses.append(asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop))

    worker = threading.Thread(target=dispatch_from_thread, daemon=True)
    worker.start()
    try:
        assert entered.wait(5), "audio request did not reach the actual HTTP boundary"
        context = runtime.db.collection("telegram_sessions").document(task["context_id"])
        before = context.get().to_dict()
        replacement_fence = before["fence"] + 1
        context.update({
            "lock_owner": "replacement-audio-worker", "fence": replacement_fence,
            "lock_expires_at": runtime.clock() + timedelta(minutes=5),
        })
    finally:
        release.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert responses and responses[0].status_code == 200
    context_after = runtime.db.collection("telegram_sessions").document(task["context_id"]).get().to_dict()
    assert context_after["lock_owner"] == "replacement-audio-worker"
    assert context_after["fence"] == replacement_fence
    from services.connectors.telegram.queue import QueueOperationState
    operation = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                 .collection("operations").document(task["operation_id"]).get().to_dict())
    assert operation["state"] == QueueOperationState.RUNNING.value
    stage = QueueOperationRetryStore(
        runtime.db, runtime.db.collection("telegram_sessions").document(task["context_id"]),
    ).get_stage(operation_id=task["operation_id"], stage_key="telegram.audio.81")
    assert stage is None or stage.failures == 0
    request_methods = [request.method for request in runtime.audio_requests]
    if pause_at == "get_file":
        assert request_methods == ["POST"]
    else:
        assert request_methods == ["POST", "GET"]
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "generate_content"
                and event["arguments"].get("model") == "gemini-3.5-transcribe"]
    assert (runtime.db.collection("telegram_sessions").document(task["context_id"])
            .collection("drafts").document(operation["target_draft_id"])
            .collection("messages").document("81").get().exists)


def test_default_worker_saves_two_real_audio_transcripts_in_intake_order(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Real Telegram downloads and Gemini transcription reach verified cleanup."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from tests.fixtures import sdk_fakes as sdk_module

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    first_text = "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435 transcript / English sentence."
    second_text = "English phrase mixed with \u05e9\u05dc\u05d5\u05dd."
    first_bytes, second_bytes = b"first-audio-bytes", b"second-audio-bytes"
    for transcript in (first_text, second_text):
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=transcript)]))],
        ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Audio and text saved", "folder_path": ["Client"]}),
        )]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    doc_id = "doc-audio-intake-order"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-audio-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-audio-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"audio-readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })
    original_execute = sdk_module._Request.execute
    inserted_document_text = {"value": None}

    def real_docs_boundary_roundtrip(request, **kwargs):
        result = original_execute(request, **kwargs)
        if request.boundary == "docs" and request.operation == "documents.batchUpdate":
            inserted_document_text["value"] = request.args["body"]["requests"][0]["insertText"]["text"]
        if (request.boundary == "docs" and request.operation == "documents.get"
                and request.args.get("documentId") == doc_id
                and inserted_document_text["value"] is not None):
            return {
                "revisionId": "rev-audio-filled",
                "body": {"content": [{"paragraph": {"elements": [{
                    "textRun": {"content": inserted_document_text["value"]},
                }]}}]},
            }
        return result

    monkeypatch.setattr(sdk_module._Request, "execute", real_docs_boundary_roundtrip)
    runtime.get_file_responses.update({
        "audio-91": httpx.Response(200, json={"ok": True, "result": {
            "file_id": "audio-91", "file_path": "voice/first.ogg",
        }}),
        "audio-92": httpx.Response(200, json={"ok": True, "result": {
            "file_id": "audio-92", "file_path": "voice/second.ogg",
        }}),
    })
    runtime.audio_responses.update({
        "/file/bot123456:synthetic-token/voice/first.ogg": httpx.Response(
            200, content=first_bytes, headers={"content-type": "audio/ogg"},
        ),
        "/file/bot123456:synthetic-token/voice/second.ogg": httpx.Response(
            200, content=second_bytes, headers={"content-type": "audio/ogg"},
        ),
    })

    import threading
    second_download_finished = threading.Event()
    completed_download_paths = []

    async def delay_first_download(request):
        if request.method == "GET" and request.url.path.endswith("/voice/first.ogg"):
            assert await asyncio.to_thread(second_download_finished.wait, 5)

    def record_download_completion(response):
        if response.request.method != "GET":
            return
        completed_download_paths.append(response.request.url.path)
        if response.request.url.path.endswith("/voice/second.ogg"):
            second_download_finished.set()

    runtime.audio_request_hook["callback"] = delay_first_download
    runtime.audio_response_hook["callback"] = record_download_completion

    # Hold the first real SDK transcription response until the second request
    # arrives, forcing completion order to differ from the sealed intake order.
    first_waiting, second_finished = threading.Event(), threading.Event()
    original_async_generate = sdk_module._AsyncGeminiModels.generate_content
    active = 0
    max_active = 0
    completed_markers = []

    async def reverse_audio_completion(self, **kwargs):
        nonlocal active, max_active
        contents = kwargs.get("contents")
        audio = contents.inline_data.data if contents is not None else None
        active += 1
        max_active = max(max_active, active)
        try:
            if audio == second_bytes:
                assert await asyncio.to_thread(first_waiting.wait, 5)
            response = await original_async_generate(self, **kwargs)
            if audio == first_bytes:
                first_waiting.set()
                await asyncio.to_thread(second_finished.wait, 5)
                completed_markers.append(audio)
                return response
            if audio == second_bytes:
                completed_markers.append(audio)
                second_finished.set()
                return response
            return response
        finally:
            active -= 1

    monkeypatch.setattr(sdk_module._AsyncGeminiModels, "generate_content", reverse_audio_completion)

    source_updates = [
        {"update_id": 16621, "message": {
            "message_id": 91, "date": 1793320000,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
            "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                               "date": 1793310000},
            "voice": {"file_id": "audio-91", "mime_type": "audio/ogg"},
        }},
        {"update_id": 16622, "message": {
            "message_id": 92, "date": 1793320010,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
            "forward_origin": {"type": "user", "sender_user": {"id": 45, "first_name": "Bea"},
                               "date": 1793310010},
            "voice": {"file_id": "audio-92", "mime_type": "audio/ogg"},
        }},
        {"update_id": 16623, "message": {
            "message_id": 93, "date": 1793320020,
            "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
            "forward_origin": {"type": "user", "sender_user": {"id": 46, "first_name": "Cy"},
                               "date": 1793310020},
            "text": "Source text follows the audios.",
        }},
    ]
    for source in source_updates:
        assert asyncio.run(runtime.webhook(source), loop_factory=_selector_loop).status_code == 200
        ack = next(item["body"] for item in reversed(runtime.task_requests)
                   if item["body"].get("notification_id"))
        assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    intake_context = runtime.db.collection("telegram_sessions").document("77_default")
    intake_draft_id = intake_context.get().to_dict()["active_draft_id"]
    intake_draft = intake_context.collection("drafts").document(intake_draft_id).get().to_dict()
    assert intake_draft["ordered_message_ids"] == [91, 92, 93]
    assert intake_context.collection("drafts").document(intake_draft_id).collection(
        "messages").document("93").get().to_dict()["text"] == "Source text follows the audios."
    save_update = {"update_id": 16624, "message": {
        "message_id": 94, "date": 1793320030,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    for _ in range(8):
        response = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
        assert response.status_code == 200, response.text
        core = runtime.db.collection("dialog_operations").document(task["operation_id"]).get().to_dict()
        if core and core.get("state") == "failed":
            pytest.fail(f"happy audio workflow failed: {core.get('stage')}")
        if core and core.get("doc_id"):
            break
        runtime.clock.advance()
        task = next(item["body"] for item in reversed(runtime.task_requests)
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and item["body"].get("operation_id") == task["operation_id"]
                    and not item["body"].get("notification_id"))
    else:
        pytest.fail("real audio save did not reach core completion")
    assert max_active == 2 and completed_markers == [second_bytes, first_bytes]
    assert completed_download_paths == [
        "/file/bot123456:synthetic-token/voice/second.ogg",
        "/file/bot123456:synthetic-token/voice/first.ogg",
    ]
    audio_events = [event for event in sdk_fakes.trace
                    if event["boundary"] == "gemini"
                    and event["operation"] == "generate_content"
                    and event["arguments"].get("model") == "gemini-3.5-transcribe"]
    assert len(audio_events) == 2
    audio_config = audio_events[0]["arguments"]["config"].audio_transcription_config
    assert audio_config.mode == "VERBATIM" and audio_config.language_codes == []
    metadata_event = next(event for event in sdk_fakes.trace
                          if event["boundary"] == "gemini"
                          and event["operation"] == "generate_content"
                          and "Infer a concise dialog title" in str(event["arguments"]))
    assert sdk_fakes.trace.index(audio_events[0]) < sdk_fakes.trace.index(metadata_event)
    assert sdk_fakes.trace.index(audio_events[1]) < sdk_fakes.trace.index(metadata_event)
    metadata = json.loads(metadata_event["arguments"]["contents"])
    assert metadata["prepared_dialogue"] == [
        {"speaker": "Ada", "text": first_text},
        {"speaker": "Bea", "text": second_text},
        {"speaker": "Cy", "text": "Source text follows the audios."},
    ]
    assert metadata["prepared_dialogue"][0]["text"] == first_text
    assert metadata["prepared_dialogue"][1]["text"] == second_text
    get_file_requests = [request for request in runtime.audio_requests
                         if request.url.path.endswith("/getFile")]
    download_requests = [request for request in runtime.audio_requests if request.method == "GET"]
    assert len(get_file_requests) == len(download_requests) == 2
    for request in get_file_requests:
        file_id = json.loads(request.content)["file_id"]
        expected_path = {"audio-91": "/voice/first.ogg", "audio-92": "/voice/second.ogg"}[file_id]
        assert any(item.url.path.endswith(expected_path) for item in download_requests)
    assert core["doc_id"] == doc_id
    index_task = next(item for item in runtime.task_requests if item["url"].endswith(f"/sync/doc/{doc_id}"))
    document_text = inserted_document_text["value"]
    assert first_text in document_text and second_text in document_text
    assert document_text.index(first_text) < document_text.index(second_text)
    from core.integrations.firestore.transcripts import load_index
    from core.parsing.parser import parse_document
    _document_metadata, parsed_utterances = parse_document(document_text)
    assert [row["text"] for row in parsed_utterances] == [
        first_text, second_text, "Source text follows the audios.",
    ]
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        transport = httpx.ASGITransport(app=vector_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector.test") as client:
            return await client.post(f"/sync/doc/{doc_id}", json=index_task["body"])

    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200 and indexed.json()["status"] == "ok"
    index_record = load_index(doc_id)
    assert index_record["status"] == "synced"
    assert len(index_record["utterance_hashes"]) == 3
    receipt = runtime.db.collection("index_completions").document(doc_id).collection(
        "generations").document(core["desired_content_hash"]).get().to_dict()
    assert receipt["status"] == "completed"
    wake_request = next(item["body"] for item in runtime.task_requests
                        if item["body"].get("target_kind") == "telegram_dispatch"
                        and item["body"].get("operation_id") == task["operation_id"]
                        and item["body"].get("outbox_id") != task.get("outbox_id"))
    wake = task_payload(get_outbox_record(runtime.db, wake_request["outbox_id"]))
    completed = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert completed.status_code == 200 and completed.json()["status"] == "processed"
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    retry_store = QueueOperationRetryStore(
        runtime.db, runtime.db.collection("telegram_sessions").document(task["context_id"]),
    )
    for stage_key in (
        "telegram.audio.91", "telegram.audio.92",
        "telegram.preparation", "telegram.analysis", "telegram.classification",
    ):
        stage = retry_store.get_stage(operation_id=task["operation_id"], stage_key=stage_key)
        assert stage is not None and stage.failures == 0
    metadata_calls = [event for event in sdk_fakes.trace
                      if event["boundary"] == "gemini"
                      and event["operation"] == "generate_content"
                      and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_calls) == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    session = runtime.db.collection("telegram_sessions").document(task["context_id"])
    operation = session.collection("operations").document(task["operation_id"]).get().to_dict()
    draft = session.collection("drafts").document(operation["target_draft_id"]).get().to_dict()
    assert draft["state"] == "cleaned"
    assert draft["completion_certificate"]["receipt_ref"].startswith(f"index_completions/{doc_id}/")
    assert list(session.collection("drafts").document(draft["draft_id"])
                .collection("messages").stream()) == []
    final_messages = [payload for method, payload in runtime.sent
                      if method == "sendMessage" and "Saved and indexed" in payload.get("text", "")]
    assert len(final_messages) == 1


def test_pending_save_survives_more_than_three_canonical_work_slices(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """A committed Doc remains pending across ordinary slice yields until real indexing."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from tests.fixtures import sdk_fakes as sdk_module

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    from services.connectors.telegram.repository import TelegramRepository
    runtime.route_response("save")
    source = "Exact pending source with unicode \u05e9\u05dc\u05d5\u05dd and blank lines."
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Pending save", "folder_path": ["Client"]}),
        )]))],
    ))
    doc_id = "doc-pending-slices"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(16):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(12))):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })

    original_execute = sdk_module._Request.execute
    written = {"text": None}

    def roundtrip(request, **kwargs):
        result = original_execute(request, **kwargs)
        if request.boundary == "docs" and request.operation == "documents.batchUpdate":
            written["text"] = request.args["body"]["requests"][0]["insertText"]["text"]
        if (request.boundary == "docs" and request.operation == "documents.get"
                and request.args.get("documentId") == doc_id and written["text"] is not None):
            return {"revisionId": "rev-filled", "body": {"content": [
                {"paragraph": {"elements": [{"textRun": {"content": written["text"]}}]}}
            ]}}
        return result

    monkeypatch.setattr(sdk_module._Request, "execute", roundtrip)
    source_update = {"update_id": 16801, "message": {
        "message_id": 101, "date": 1793320000, "chat": {"id": 77, "type": "private"},
        "from": {"id": 7}, "forward_origin": {"type": "user", "sender_user": {
            "id": 44, "first_name": "Ada"}, "date": 1793310000}, "text": source,
    }}
    assert asyncio.run(runtime.webhook(source_update), loop_factory=_selector_loop).status_code == 200
    ack = next(item["body"] for item in runtime.task_requests if item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    intake_session = runtime.db.collection("telegram_sessions").document("77_default")
    from services.connectors.telegram.repository import TelegramRepository
    mapped_a_number = TelegramRepository(client=runtime.db).bind_document(
        "77_default", "doc-A-mapped", "conversation-A-mapped",
        metadata={"title": "Legitimate mapped transcript A"}, now=runtime.clock.value,
    )
    assert mapped_a_number >= 2
    save_update = {"update_id": 16802, "message": {
        "message_id": 102, "date": 1793320010, "chat": {"id": 77, "type": "private"},
        "from": {"id": 7}, "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    operation_id, context_id = task["operation_id"], task["context_id"]

    for _ in range(12):
        result = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
        assert result.status_code == 200, result.text
        core = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
        if core and core.get("doc_id") == doc_id:
            break
        runtime.clock.advance()
        task = next(item["body"] for item in reversed(runtime.task_requests)
                    if item["body"].get("target_kind") == "telegram_dispatch"
                    and item["body"].get("operation_id") == operation_id
                    and not item["body"].get("notification_id"))
    else:
        pytest.fail("save did not create its real Doc")

    session = runtime.db.collection("telegram_sessions").document(context_id)
    operation = session.collection("operations").document(operation_id).get().to_dict()
    core_row = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
    assert session.get().to_dict().get("reserved_operation_id") == operation_id
    assert operation["state"] == "running"
    assert core_row["state"] == "running" and core_row["stage"] == "index_dispatch"
    draft_ref = session.collection("drafts").document(operation["target_draft_id"])
    draft = draft_ref.get().to_dict()
    assert draft["state"] in {"processing", "indexing"}
    assert core["doc_id"] == doc_id
    assert draft_ref.collection("messages").document("101").get().to_dict()["text"] == source
    from core.parsing.parser import parse_document
    assert [row["text"] for row in parse_document(written["text"])[1]] == [source]
    prepared_before = list(draft_ref.collection("prepared").stream())
    assert prepared_before
    metadata_events = [event for event in sdk_fakes.trace if event["boundary"] == "gemini"
                       and event["operation"] == "generate_content"
                       and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_events) == 1
    trace_before_slices = len(sdk_fakes.trace)
    request_ref = session.collection("operations").document(operation_id).collection(
        "agent_requests").document("request")
    private_request_before = request_ref.get().to_dict()
    assert private_request_before and not private_request_before.get("purged")
    assert "save this dialog" in json.dumps(private_request_before, ensure_ascii=False)
    prepared_rows_before = [snapshot.to_dict() for snapshot in prepared_before]
    consumed_outbox_ids = {task["outbox_id"]}
    context_before = session.get().to_dict()
    prior_generation = task["generation"]
    prior_fence = context_before["fence"]

    retry_store = QueueOperationRetryStore(runtime.db, session)
    for _ in range(5):
        current_context = session.get().to_dict()
        generation = current_context["telegram_dispatch_wake_generation"]
        assert generation > prior_generation
        continuation = next(item["body"] for item in reversed(runtime.task_requests)
                            if item["body"].get("target_kind") == "telegram_dispatch"
                            and item["body"].get("operation_id") == operation_id
                            and item["body"].get("context_id") == context_id
                            and item["body"].get("generation") == generation
                            and item["body"].get("outbox_id") not in consumed_outbox_ids
                            and not item["body"].get("notification_id"))
        continuation_record = get_outbox_record(runtime.db, continuation["outbox_id"])
        assert continuation_record is not None
        assert continuation_record.generation == generation
        assert continuation["outbox_id"] not in consumed_outbox_ids
        consumed_outbox_ids.add(continuation["outbox_id"])
        runtime.clock.value = max(runtime.clock.value, continuation_record.schedule_at)
        result = asyncio.run(runtime.dispatch(task_payload(continuation_record)),
                             loop_factory=_selector_loop)
        assert result.status_code == 200, result.text
        row = session.collection("operations").document(operation_id).get().to_dict()
        core_row = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
        assert row["state"] == "running"
        assert session.get().to_dict().get("reserved_operation_id") == operation_id
        assert core_row["state"] == "running" and core_row["stage"] == "index_dispatch"
        assert all(count == 0 for _, count in core_row["stage_failure_counts"])
        prior_generation = generation
        prior_fence = current_context["fence"]
        assert session.get().to_dict()["fence"] > prior_fence
        assert session.collection("drafts").document(operation["target_draft_id"]).get().to_dict()["state"] == "processing"
        assert draft_ref.collection("messages").document("101").get().to_dict()["text"] == source
        assert [snapshot.to_dict() for snapshot in draft_ref.collection("prepared").stream()] == prepared_rows_before
        for stage_key in ("telegram.preparation", "telegram.analysis", "telegram.classification"):
            stage = retry_store.get_stage(operation_id=operation_id, stage_key=stage_key)
            assert stage is not None and stage.failures == 0
        assert request_ref.get().to_dict() == private_request_before
        metadata_calls = [event for event in sdk_fakes.trace
                          if event["boundary"] == "gemini"
                          and event["operation"] == "generate_content"
                          and "Infer a concise dialog title" in str(event["arguments"])]
        assert len(metadata_calls) == 1
        assert not any((item.to_dict() or {}).get("operation_id") == operation_id
                       and (item.to_dict() or {}).get("kind") == "final"
                       for item in runtime.db.collection("notification_outbox").stream())
        assert not any(event["boundary"] == "drive" and event["operation"] == "files.create"
                       for event in sdk_fakes.trace[trace_before_slices:])

    # B is still the reserved pending head; C captures B as predecessor while A is last_doc.
    assert session.get().to_dict()["last_doc_id"] == "doc-A-mapped"
    following_update = {"update_id": 16803, "message": {
        "message_id": 103, "date": 1793320020, "chat": {"id": 77, "type": "private"},
        "from": {"id": 7}, "text": "Can you share the transcript from that save?",
    }}
    assert asyncio.run(runtime.webhook(following_update), loop_factory=_selector_loop).status_code == 200
    following_operation_id = session.get().to_dict()["queue_tail_operation_id"]
    following_operation = session.collection("operations").document(following_operation_id).get().to_dict()
    assert following_operation["predecessor_operation_id"] == operation_id
    assert session.get().to_dict()["last_doc_id"] == "doc-A-mapped"
    following_ack = next(item["body"] for item in reversed(runtime.task_requests)
                         if item["body"].get("notification_id"))

    from core.integrations.firestore.outbox import get_outbox_record
    index_task = next(item for item in runtime.task_requests
                      if item["url"].endswith(f"/sync/doc/{doc_id}"))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=vector_app),
                                     base_url="http://vector.test") as client:
            return await client.post(f"/sync/doc/{doc_id}", json=index_task["body"])

    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200 and indexed.json()["status"] == "ok", indexed.text
    indexed_upsert = next(event for event in reversed(sdk_fakes.trace)
                          if event["boundary"] == "qdrant" and event["operation"] == "upsert")
    indexed_payloads = [point.payload for point in indexed_upsert["arguments"]["points"]
                        if point.payload.get("doc_id") == doc_id]
    assert len(indexed_payloads) == 1 and source in indexed_payloads[0]["text"]
    sdk_fakes.qdrant.queue("scroll", response=(
        [SimpleNamespace(payload=payload) for payload in indexed_payloads], None,
    ))
    runtime.route_response(
        "substantive", target_ref="last", target_operation_id=operation_id,
        dialog_number=None,
        substantive_text="Read the transcript from that saved dialog.",
        tool_choice={"name": "get_transcript", "arguments": {"doc_id": doc_id},
                     "link_requested": True},
    )
    from google.genai import types
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=f"Here is the transcript for Dialog #{draft['dialog_number']}: {source}",
        )]))],
    ))
    final_generation = session.get().to_dict()["telegram_dispatch_wake_generation"]
    final_request = next(item["body"] for item in reversed(runtime.task_requests)
                         if item["body"].get("target_kind") == "telegram_dispatch"
                         and item["body"].get("operation_id") == operation_id
                         and item["body"].get("context_id") == context_id
                         and item["body"].get("generation") == final_generation
                         and item["body"].get("outbox_id") not in consumed_outbox_ids
                         and not item["body"].get("notification_id"))
    final_record = get_outbox_record(runtime.db, final_request["outbox_id"])
    assert final_record is not None
    runtime.clock.value = max(runtime.clock.value, final_record.schedule_at)
    final = asyncio.run(runtime.dispatch(task_payload(final_record)), loop_factory=_selector_loop)
    assert final.status_code == 200, final.text
    assert session.collection("operations").document(operation_id).get().to_dict()["state"] == "completed"
    assert draft_ref.get().to_dict()["state"] == "cleaned"
    assert len([payload for method, payload in runtime.sent if method == "sendMessage"
                and "Saved and indexed" in payload.get("text", "")]) == 1
    assert session.get().to_dict()["last_doc_id"] == doc_id
    consumed_following_ids = set()
    trace_before_following = len(sdk_fakes.trace)
    ack_delivery = asyncio.run(runtime.deliver(following_ack), loop_factory=_selector_loop)
    assert ack_delivery.status_code == 200, ack_delivery.text
    following_after = session.collection("operations").document(following_operation_id).get().to_dict()
    for _ in range(8):
        if following_after["state"] == "completed":
            break
        assert following_after["state"] in {"pending", "running"}, following_after
        context_generation = session.get().to_dict()["telegram_dispatch_wake_generation"]
        candidates = [item["body"] for item in reversed(runtime.task_requests)
                     if item["body"].get("target_kind") == "telegram_dispatch"
                     and item["body"].get("operation_id") == following_operation_id
                     and item["body"].get("context_id") == context_id
                     and item["body"].get("generation") == context_generation
                     and item["body"].get("outbox_id") not in consumed_following_ids
                     and not item["body"].get("notification_id")]
        assert candidates
        following_task = candidates[0]
        following_record = get_outbox_record(runtime.db, following_task["outbox_id"])
        assert following_record is not None
        assert following_record.generation == following_task["generation"]
        assert following_task["outbox_id"] not in consumed_following_ids
        consumed_following_ids.add(following_task["outbox_id"])
        runtime.clock.value = max(runtime.clock.value, following_record.schedule_at)
        following_result = asyncio.run(runtime.dispatch(task_payload(following_record)),
                                       loop_factory=_selector_loop)
        assert following_result.status_code == 200, following_result.text
        following_after = session.collection("operations").document(following_operation_id).get().to_dict()
    if following_after["state"] != "completed":
        pytest.fail(f"following predecessor-bound operation did not complete: {following_after}")
    assert following_after["predecessor_operation_id"] == operation_id
    route_control = following_after["route_control"]
    assert route_control["status"] == "ok"
    assert route_control["target_doc_id"] == doc_id
    assert route_control["dialog_number"] == draft["dialog_number"]
    assert route_control["target_operation_id"] == operation_id
    assert route_control["dependency_operation_id"] == operation_id
    assert route_control["tool_choice"]["name"] == "get_transcript"
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    turn = FirestoreAgentLedger(
        runtime.db, context_id=context_id, operation_id=following_operation_id,
        owner="read-only-check", fence=0, clock=runtime.clock,
    ).load_turn(route_control["conversation_id"], following_operation_id)
    assert turn is not None and source in json.dumps(turn.get("tool_results", {}),
                                                     ensure_ascii=False)
    transcript_calls = [event for event in sdk_fakes.trace[trace_before_following:]
                        if event["boundary"] == "qdrant" and event["operation"] == "scroll"]
    assert transcript_calls
    from google.genai import types as genai_types
    tool_loop_events = [event for event in sdk_fakes.trace[trace_before_following:]
                        if event["boundary"] == "gemini"
                        and event["operation"] == "generate_content"
                        and isinstance(event["arguments"].get("contents"), list)
                        and any(isinstance(message, dict) and message.get("role") == "model"
                                for message in event["arguments"]["contents"])]
    assert tool_loop_events
    for event in tool_loop_events:
        for message in event["arguments"]["contents"]:
            if isinstance(message, dict) and message.get("role") in {"model", "user"}:
                genai_types.Content.model_validate(message)
    assert any(method == "sendMessage" and source in payload.get("text", "")
               for method, payload in runtime.sent)


def test_default_worker_saves_501_unicode_source_rows_over_one_mib(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """The default save path preserves 501 long UTF-8 rows through real cleanup."""
    from google.genai import types
    from tests.fixtures import sdk_fakes as sdk_module
    from tests.fixtures.firestore_fake import FakeTransaction, FakeWriteBatch

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Large exact source", "folder_path": ["Client"]}),
        )]))],
    ))
    doc_id = "doc-501-long-rows"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(32):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for index in range(40):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": f"large-read-{index}",
        })

    transaction_observations = []
    batch_observations = []
    original_transaction_commit = FakeTransaction._commit
    original_batch_commit = FakeWriteBatch.commit
    from google.cloud.firestore_v1 import _helpers as firestore_helpers
    from core.integrations.firestore.input_parts import MAX_PART_BYTES

    def inspect_transaction(transaction):
        prepared = [(reference.path, data) for _, reference, data, _ in transaction._writes
                    if "/prepared/" in reference.path]
        input_rows = [(reference.path, data) for _, reference, data, _ in transaction._writes
                      if ("/messages/" in reference.path or "/prepared/" in reference.path
                          or "/input_parts/" in reference.path)]
        if prepared or input_rows:
            def sdk_doc_bytes(row):
                path, data = row
                document_path = (
                    "projects/pytest-synthetic-project/databases/(default)/documents/" + path
                )
                return firestore_helpers.pbs_for_set_no_merge(document_path, data)[0].update._pb.ByteSize()

            largest_input = max(input_rows, key=sdk_doc_bytes, default=(None, {}))
            payload_lengths = [len(data["payload"]) for path, data in input_rows
                               if "/input_parts/part-" in path and isinstance(data, dict)
                               and isinstance(data.get("payload"), bytes)]
            transaction_observations.append({
                "prepared_rows": len(prepared),
                "max_input_document_bytes": max(map(sdk_doc_bytes, input_rows), default=0),
                "max_input_document_path": largest_input[0],
                "max_input_document_keys": sorted(largest_input[1]) if isinstance(largest_input[1], dict) else [],
                "max_input_part_payload_bytes": max(payload_lengths, default=0),
            })
        return original_transaction_commit(transaction)

    def inspect_batch(batch, **kwargs):
        delete_count = sum(operation == "delete" for operation, *_ in batch._writes)
        if delete_count:
            batch_observations.append(delete_count)
        return original_batch_commit(batch, **kwargs)

    monkeypatch.setattr(FakeTransaction, "_commit", inspect_transaction)
    monkeypatch.setattr(FakeWriteBatch, "commit", inspect_batch)

    original_execute = sdk_module._Request.execute
    inserted = {"text": None}

    def actual_docs_roundtrip(request, **kwargs):
        result = original_execute(request, **kwargs)
        if request.boundary == "docs" and request.operation == "documents.batchUpdate":
            inserted["text"] = request.args["body"]["requests"][0]["insertText"]["text"]
        if (request.boundary == "docs" and request.operation == "documents.get"
                and request.args.get("documentId") == doc_id and inserted["text"] is not None):
            return {"revisionId": "rev-filled", "body": {"content": [
                {"paragraph": {"elements": [{"textRun": {"content": inserted["text"]}}]}}
            ]}}
        return result

    monkeypatch.setattr(sdk_module._Request, "execute", actual_docs_roundtrip)
    sources = []
    for index in range(501):
        word = "ש" * 22 + f"{index:04d}"
        sources.append("  " + " ".join([word] * 25) + "\n\n  " + " ".join([word] * 25))
    assert sum(len(source) for source in sources) < 1_000_000
    assert sum(len(source.encode("utf-8")) for source in sources) > 1024 * 1024
    assert all(len(source.split()) >= 50 for source in sources)

    async def intake_all_sources():
        transport = httpx.ASGITransport(app=runtime.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            for index, source in enumerate(sources):
                update = {"update_id": 19000 + index, "message": {
                    "message_id": 5000 + index, "date": 1793320000 + index,
                    "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
                    "forward_origin": {"type": "user", "sender_user": {
                        "id": 44, "first_name": "Ada"}, "date": 1793310000},
                    "text": source,
                }}
                response = await client.post(
                    "/telegram/webhook", headers={
                        "X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret",
                    }, json=update,
                )
                assert response.status_code == 200, response.text
                if index == 0:
                    ack = next(item["body"] for item in runtime.task_requests
                               if item["body"].get("notification_id"))
                    delivered = await client.post(
                        "/internal/deliver", headers={"Authorization": "Bearer fake-task"},
                        json=ack,
                    )
                    assert delivered.status_code == 200, delivered.text

    asyncio.run(intake_all_sources(), loop_factory=_selector_loop)
    session = runtime.db.collection("telegram_sessions").document("77_default")
    context = session.get().to_dict()
    draft_id = context["active_draft_id"]
    draft_ref = session.collection("drafts").document(draft_id)
    assert draft_ref.get().to_dict()["ordered_message_ids"] == list(range(5000, 5501))
    assert len(list(draft_ref.collection("messages").stream())) == 501

    save_update = {"update_id": 19502, "message": {
        "message_id": 6000, "date": 1793320600, "chat": {"id": 77, "type": "private"},
        "from": {"id": 7}, "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    operation_id = task["operation_id"]
    for _ in range(12):
        response = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
        assert response.status_code == 200, response.text
        core = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
        if core and core.get("doc_id") == doc_id:
            break
        continuation = next(item["body"] for item in reversed(runtime.task_requests)
                            if item["body"].get("target_kind") == "telegram_dispatch"
                            and item["body"].get("operation_id") == operation_id
                            and item["body"].get("outbox_id") != task.get("outbox_id")
                            and not item["body"].get("notification_id"))
        from core.integrations.firestore.outbox import get_outbox_record, task_payload
        record = get_outbox_record(runtime.db, continuation["outbox_id"])
        assert record is not None
        runtime.clock.value = max(runtime.clock.value, record.schedule_at)
        task = task_payload(record)
    else:
        pytest.fail("501-row save did not create its real Doc")

    metadata_event = next(event for event in sdk_fakes.trace
                          if event["boundary"] == "gemini"
                          and event["operation"] == "generate_content"
                          and "Infer a concise dialog title" in str(event["arguments"]))
    metadata = json.loads(metadata_event["arguments"]["contents"])
    assert len(metadata["prepared_dialogue"]) == 501
    assert [row["text"] for row in metadata["prepared_dialogue"]] == sources
    assert metadata["trusted_context"]["word_capacity"] == 521
    assert draft_ref.get().to_dict()["context_capacity"] == 521
    assert all(len(row["text"].split()) >= 50 for row in metadata["prepared_dialogue"])
    prepared_rows = sorted(
        (snapshot.to_dict() for snapshot in draft_ref.collection("prepared").stream()),
        key=lambda row: row["order_index"],
    )
    assert len(prepared_rows) == 501
    assert [row["order_index"] for row in prepared_rows] == list(range(501))
    assert [row["source_message_ids"] for row in prepared_rows] == [[5000 + i] for i in range(501)]
    assert transaction_observations
    assert max(item["prepared_rows"] for item in transaction_observations) <= 400
    largest_observation = max(transaction_observations, key=lambda item: item["max_input_document_bytes"])
    assert largest_observation["max_input_document_bytes"] < 1024 * 1024, largest_observation
    assert max(item["max_input_part_payload_bytes"] for item in transaction_observations) <= MAX_PART_BYTES
    operation_ref = session.collection("operations").document(operation_id)
    allocations_before_cleanup = [snapshot.to_dict()
                                  for snapshot in operation_ref.collection("save_payloads").stream()]
    assert allocations_before_cleanup
    assert max(row["byte_length"] for row in allocations_before_cleanup) > 1024 * 1024
    stage_results_before_cleanup = {}
    for snapshot in operation_ref.collection("stages").stream():
        row = snapshot.to_dict() or {}
        if (isinstance(row.get("result"), dict)
                and "__telegram_save_stage_payload__" in row["result"]):
            stage_results_before_cleanup[snapshot.id] = row["result"]
    assert stage_results_before_cleanup
    allocated_input_refs = []
    for allocation in allocations_before_cleanup:
        assert allocation["cleanup_state"] == "allocated"
        input_ref = allocation["input_ref"]
        allocated_input_refs.append(input_ref)
        part_operation_id = input_ref.split("/")[1]
        parts = runtime.db.collection("dialog_operations").document(
            part_operation_id,
        ).collection("input_parts")
        manifest = parts.document("manifest").get().to_dict()
        assert manifest and manifest["ready"] is True
        assert manifest["byte_length"] == allocation["byte_length"]
        assert list(parts.where("index", ">=", 0).stream())

    from core.parsing.parser import parse_document
    _metadata, parsed_rows = parse_document(inserted["text"])
    assert [row["text"] for row in parsed_rows] == sources
    index_task = next(item for item in runtime.task_requests
                      if item["url"].endswith(f"/sync/doc/{doc_id}"))
    for _ in range(2048):
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[]))],
        ))
    from services.vector_sync.app.main import app as vector_app

    async def sync_large_document():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=vector_app),
                                     base_url="http://vector.test") as client:
            return await client.post(f"/sync/doc/{doc_id}", json=index_task["body"])

    indexed = asyncio.run(sync_large_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200 and indexed.json()["status"] == "ok", indexed.text
    final_request = next(item["body"] for item in reversed(runtime.task_requests)
                         if item["body"].get("target_kind") == "telegram_dispatch"
                         and item["body"].get("operation_id") == operation_id
                         and item["body"].get("outbox_id") != task.get("outbox_id")
                         and not item["body"].get("notification_id"))
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    final_record = get_outbox_record(runtime.db, final_request["outbox_id"])
    assert final_record is not None
    runtime.clock.value = max(runtime.clock.value, final_record.schedule_at)
    finalized = asyncio.run(runtime.dispatch(task_payload(final_record)), loop_factory=_selector_loop)
    assert finalized.status_code == 200, finalized.text
    draft = draft_ref.get().to_dict()
    assert draft["state"] == "cleaned"
    assert draft.get("completion_certificate")
    assert list(draft_ref.collection("messages").stream()) == []
    assert list(draft_ref.collection("prepared").stream()) == []
    allocations_after_cleanup = [snapshot.to_dict()
                                 for snapshot in operation_ref.collection("save_payloads").stream()]
    assert len(allocations_after_cleanup) == len(allocations_before_cleanup)
    assert all(row["cleanup_state"] == "purged" for row in allocations_after_cleanup)
    for input_ref in allocated_input_refs:
        part_operation_id = input_ref.split("/")[1]
        parts = runtime.db.collection("dialog_operations").document(
            part_operation_id,
        ).collection("input_parts")
        assert list(parts.where("index", ">=", 0).stream()) == []
        assert parts.document("manifest").get().exists
    for stage_id in stage_results_before_cleanup:
        stage_after_cleanup = operation_ref.collection("stages").document(stage_id).get().to_dict()
        result = stage_after_cleanup["result"]
        assert result.get("purged") is True
        assert "__telegram_save_stage_payload__" not in result
    assert batch_observations and max(batch_observations) <= 400


def test_default_worker_parent_takeover_during_docs_read_stops_old_executor(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A live takeover during a physical Docs read fences later RPC and local writes."""
    import threading
    from google.genai import types
    from tests.fixtures import sdk_fakes as sdk_module

    runtime = _default_telegram_runtime_harness(
        firestore_fake, sdk_fakes, monkeypatch, heartbeat_interval_seconds=0.01,
    )
    doc_id = "parent-fence-slow-read"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    sdk_fakes.docs.queue("documents.get", response=empty_document)
    entered, release, read_finished = threading.Event(), threading.Event(), threading.Event()
    original_execute = sdk_module._Request.execute

    def blocked_execute(request, **kwargs):
        if (request.boundary == "docs" and request.operation == "documents.get"
                and request.args.get("documentId") == doc_id and not entered.is_set()):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test Docs read release timed out")
        try:
            return original_execute(request, **kwargs)
        finally:
            if request.boundary == "docs" and request.operation == "documents.get":
                read_finished.set()

    monkeypatch.setattr(sdk_module._Request, "execute", blocked_execute)
    forward = {"update_id": 16501, "message": {
        "message_id": 71, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": "Source line for fence test.",
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack_task = next(row["body"] for row in runtime.task_requests
                    if row["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack_task), loop_factory=_selector_loop).status_code == 200
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Parent fence test", "folder_path": ["Client"]}),
        )]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    save_update = {"update_id": 16502, "message": {
        "message_id": 72, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(row["body"] for row in reversed(runtime.task_requests)
                if row["body"].get("target_kind") == "telegram_dispatch"
                and not row["body"].get("notification_id"))

    async def dispatch_with_takeover():
        dispatch = asyncio.create_task(runtime.dispatch(task))
        if not await asyncio.to_thread(entered.wait, 5):
            calls = [(event["boundary"], event["operation"]) for event in sdk_fakes.trace]
            response = await dispatch if dispatch.done() else None
            row = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                   .collection("operations").document(task["operation_id"]).get().to_dict())
            raise AssertionError(f"Docs get did not reach the blocking seam: calls={calls}, "
                                 f"response={getattr(response, 'status_code', None)}, row={row}")
        assert runtime.worker() is not None
        context = runtime.db.collection("telegram_sessions").document(task["context_id"])
        before = context.get().to_dict()
        expiry_before = before["lock_expires_at"]
        runtime.clock.value += timedelta(seconds=1)
        await asyncio.sleep(0.06)
        renewed = context.get().to_dict()
        assert renewed["lock_expires_at"] > expiry_before
        context.update({"lock_owner": "replacement-owner", "fence": renewed["fence"] + 1,
                        "lock_expires_at": runtime.clock() + timedelta(minutes=5)})
        replacement = context.get().to_dict()
        release.set()
        response = await asyncio.wait_for(dispatch, timeout=10)
        assert await asyncio.to_thread(read_finished.wait, 5)
        for _ in range(100):
            core = runtime.db.collection("dialog_operations").document(task["operation_id"]).get()
            if not (core.to_dict() or {}).get("lease_owner"):
                break
            await asyncio.sleep(0.01)
        assert not (core.to_dict() or {}).get("lease_owner"), "save worker did not unwind its global lease"
        return response, replacement

    try:
        response, replacement = asyncio.run(dispatch_with_takeover(), loop_factory=_selector_loop)
    finally:
        release.set()

    assert response.status_code == 200, response.text
    context_after = runtime.db.collection("telegram_sessions").document(task["context_id"]).get().to_dict()
    assert context_after["lock_owner"] == replacement["lock_owner"] == "replacement-owner"
    assert context_after["fence"] == replacement["fence"]
    from core.integrations.firestore.operations import get_stage
    assert get_stage(runtime.db, task["operation_id"], "fill").failures == 0
    doc_calls = [(event["operation"], event["arguments"].get("documentId")) for event in sdk_fakes.trace
                 if event["boundary"] == "docs"]
    assert doc_calls.count(("documents.get", doc_id)) == 1
    assert ("documents.batchUpdate", doc_id) not in doc_calls
    assert len([event for event in sdk_fakes.trace if event["boundary"] == "drive"
                and event["operation"] == "files.create"]) == 1
    operation = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                 .collection("operations").document(task["operation_id"]).get().to_dict())
    draft = (runtime.db.collection("telegram_sessions").document(task["context_id"])
             .collection("drafts").document(operation["target_draft_id"]).get().to_dict())
    assert draft["state"] != "cleaned"
    assert draft.get("completion_certificate") is None
    assert operation.get("failures", 0) == 0
    source = (runtime.db.collection("telegram_sessions").document(task["context_id"])
              .collection("drafts").document(operation["target_draft_id"])
              .collection("messages").document("71").get().to_dict())
    assert source["text"] == "Source line for fence test."


def test_default_worker_heartbeats_while_parent_stage_storage_read_is_blocked(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """The real queue worker renews its context lease during synchronous stage storage."""
    import threading
    from tests.fixtures.firestore_fake import FakeDocumentReference

    runtime = _default_telegram_runtime_harness(
        firestore_fake, sdk_fakes, monkeypatch, heartbeat_interval_seconds=0.01,
    )
    entered, release, read_finished = threading.Event(), threading.Event(), threading.Event()
    original_get = FakeDocumentReference.get
    blocked_path = {"value": None}

    def blocked_get(reference, *, transaction=None, **kwargs):
        if (reference.path == blocked_path["value"] and transaction is None
                and not entered.is_set()):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("stage read release timed out")
        try:
            return original_get(reference, transaction=transaction, **kwargs)
        finally:
            if reference.path == blocked_path["value"]:
                read_finished.set()

    monkeypatch.setattr(FakeDocumentReference, "get", blocked_get)
    forward = {"update_id": 16511, "message": {
        "message_id": 81, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": "Source line for storage heartbeat.",
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack_task = next(row["body"] for row in runtime.task_requests if row["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack_task), loop_factory=_selector_loop).status_code == 200
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", error=RuntimeError("planned metadata stop"))
    save_update = {"update_id": 16512, "message": {
        "message_id": 82, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7}, "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(row["body"] for row in reversed(runtime.task_requests)
                if row["body"].get("target_kind") == "telegram_dispatch"
                and not row["body"].get("notification_id"))
    blocked_path["value"] = (
        f"telegram_sessions/{task['context_id']}/operations/{task['operation_id']}"
        "/stages/telegram.preparation"
    )
    context = runtime.db.collection("telegram_sessions").document(task["context_id"])

    async def dispatch_while_storage_paused():
        dispatch = asyncio.create_task(runtime.dispatch(task))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            expiry_before = context.get().to_dict()["lock_expires_at"]
            runtime.clock.value += timedelta(seconds=1)
            deadline = asyncio.get_running_loop().time() + 5
            expiry_after = expiry_before
            while expiry_after <= expiry_before and asyncio.get_running_loop().time() < deadline:
                assert not read_finished.is_set(), "blocked stage read finished before lease renewal"
                expiry_after = context.get().to_dict()["lock_expires_at"]
                if expiry_after <= expiry_before:
                    await asyncio.sleep(0.01)
            assert expiry_after > expiry_before, "queue worker did not renew its context lease"
            assert not read_finished.is_set(), "stage read must remain blocked through lease renewal"
            release.set()
            response = await asyncio.wait_for(dispatch, timeout=10)
            assert await asyncio.to_thread(read_finished.wait, 5)
            return response
        finally:
            release.set()
            if not dispatch.done():
                await asyncio.wait_for(dispatch, timeout=10)

    response = asyncio.run(dispatch_while_storage_paused(), loop_factory=_selector_loop)
    assert response.status_code == 200, response.text
    prep = (runtime.db.collection("telegram_sessions").document(task["context_id"])
            .collection("operations").document(task["operation_id"])
            .collection("stages").document("telegram.preparation").get().to_dict())
    assert prep["state"] == "completed" and prep["failures"] == 0
    operation = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                 .collection("operations").document(task["operation_id"]).get().to_dict())
    analysis = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                .collection("operations").document(task["operation_id"])
                .collection("stages").document("telegram.analysis").get().to_dict())
    assert analysis["state"] == "failed" and analysis["failures"] == 1
    assert operation["state"] == "running"


def test_default_worker_reconstructs_real_preparation_after_fenced_writer_takeover(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """The fresh queue owner rebuilds preparation after the old prepared-row TX is fenced."""
    import threading
    from tests.fixtures.firestore_fake import FakeTransaction

    runtime = _default_telegram_runtime_harness(
        firestore_fake, sdk_fakes, monkeypatch, heartbeat_interval_seconds=0.01,
    )
    entered, release, old_commit_finished = (
        threading.Event(), threading.Event(), threading.Event(),
    )
    original_commit = FakeTransaction._commit
    blocked = {"done": False}
    paused_transaction = {"value": None}
    old_prepared_writes = []
    old_denial = {"type": None}
    preparation_metadata_commits = []
    target_draft_path = {"value": None}

    def pause_old_prepared_write(transaction):
        prepared_write = any(
            "/prepared/0" in reference.path for _operation, reference, _data, _merge
            in transaction._writes
        )
        if prepared_write and not blocked["done"]:
            blocked["done"] = True
            paused_transaction["value"] = transaction
            target_draft_path["value"] = next(
                reference.path.rsplit("/prepared/", 1)[0]
                for _operation, reference, _data, _merge in transaction._writes
                if "/prepared/" in reference.path
            )
            old_prepared_writes.extend(
                dict(data) for _operation, reference, data, _merge in transaction._writes
                if "/prepared/" in reference.path
            )
            entered.set()
            if not release.wait(5):
                raise TimeoutError("old prepared write release timed out")
        try:
            metadata_write = any(
                reference.path == target_draft_path["value"]
                and "preparation_digest" in data and "prepared_count" in data
                for _, reference, data, _ in transaction._writes if isinstance(data, dict)
            )
            result = original_commit(transaction)
            if metadata_write:
                preparation_metadata_commits.append(1)
            return result
        except Exception as error:
            if transaction is paused_transaction["value"]:
                old_denial["type"] = type(error).__name__
            raise
        finally:
            if transaction is paused_transaction["value"]:
                old_commit_finished.set()

    monkeypatch.setattr(FakeTransaction, "_commit", pause_old_prepared_write)
    forward = {"update_id": 16521, "message": {
        "message_id": 91, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000},
        "text": "First source line. Second source line.",
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack_task = next(row["body"] for row in runtime.task_requests if row["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack_task), loop_factory=_selector_loop).status_code == 200
    runtime.route_response("save")
    save_update = {"update_id": 16522, "message": {
        "message_id": 92, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(row["body"] for row in reversed(runtime.task_requests)
                if row["body"].get("target_kind") == "telegram_dispatch"
                and not row["body"].get("notification_id"))
    context_ref = runtime.db.collection("telegram_sessions").document(task["context_id"])
    sdk_fakes.gemini.queue("generate_content", error=RuntimeError("planned analysis stop"))
    old_results = []

    def run_old_owner():
        try:
            old_results.append(asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop))
        except Exception as error:
            old_results.append(error)

    old_thread = threading.Thread(target=run_old_owner)
    old_thread.start()
    try:
        assert entered.wait(5), "real prepared-row transaction did not reach pause seam"
        runtime.clock.value += timedelta(minutes=6)
        new_response = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
    finally:
        release.set()
        old_thread.join(timeout=5)

    assert not old_thread.is_alive(), "old queue writer did not unwind"
    assert old_results, "old dispatch did not return"
    assert old_commit_finished.wait(5)
    assert old_denial["type"] in {"Aborted", "PermissionError"}
    operation = context_ref.collection("operations").document(task["operation_id"]).get().to_dict()
    draft_ref = context_ref.collection("drafts").document(operation["target_draft_id"])
    prepared = [snapshot.to_dict() for snapshot in draft_ref.collection("prepared").stream()]
    assert [row["order_index"] for row in prepared] == list(range(len(prepared)))
    assert len(prepared) == 1
    assert [row["source_message_ids"] for row in prepared] == [[91]]
    assert "First source line." in prepared[0]["text"]
    assert "Second source line." in prepared[0]["text"]
    assert prepared == old_prepared_writes
    prep_stage = (context_ref.collection("operations").document(task["operation_id"])
                 .collection("stages").document("telegram.preparation").get().to_dict())
    assert prep_stage["state"] == "completed"
    assert prep_stage["attempts_started"] == 1 and prep_stage["failures"] == 0
    analysis_stage = (context_ref.collection("operations").document(task["operation_id"])
                      .collection("stages").document("telegram.analysis").get().to_dict())
    assert analysis_stage["failures"] == 1
    assert preparation_metadata_commits == [1]
    assert new_response.status_code == 200, new_response.text


def test_default_app_retry_reuses_real_doc_after_fill_403(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """A permanent fill read failure retains the created Doc and retries it once."""
    from google.genai import types
    from googleapiclient.errors import HttpError
    from httplib2 import Response
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Dialog A saved", "folder_path": ["Client"]}),
        )]))],
    ))
    doc_id = "doc-created-before-provider-failure"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    provider_secret = "synthetic-drive-provider-response-secret"
    provider_error = HttpError(
        Response({"status": "403", "reason": "Forbidden"}),
        f"provider response contains {provider_secret}".encode(),
    )
    sdk_fakes.docs.queue("documents.get", error=provider_error)

    source = "The exact forwarded source line for dialog A."
    forward = {"update_id": 15501, "message": {
        "message_id": 61, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": source,
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    forward_ack = next(item["body"] for item in runtime.task_requests
                       if item["body"].get("target_kind") == "telegram_dispatch"
                       and item["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(forward_ack), loop_factory=_selector_loop).status_code == 200
    save_update = {"update_id": 15502, "message": {
        "message_id": 62, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    original_task = next(item["body"] for item in reversed(runtime.task_requests)
                         if item["body"].get("target_kind") == "telegram_dispatch"
                         and not item["body"].get("notification_id"))
    context_id = original_task["context_id"]
    failed = asyncio.run(runtime.dispatch(original_task), loop_factory=_selector_loop)
    assert failed.status_code == 200, failed.text
    assert failed.json() == {"status": "processed"}

    session = runtime.db.collection("telegram_sessions").document(context_id)
    original = session.collection("operations").document(original_task["operation_id"]).get().to_dict()
    draft_id = original["target_draft_id"]
    draft_before_retry = session.collection("drafts").document(draft_id).get().to_dict()
    assert original["state"] == "failed"
    assert draft_before_retry["state"] == "failed"
    assert draft_before_retry["ordered_message_ids"] == [61]
    assert draft_before_retry["immutable_save_intent_id"] == original_task["operation_id"]
    source_row = session.collection("drafts").document(draft_id).collection(
        "messages").document("61").get().to_dict()
    assert source_row["text"] == source
    core_before_retry = runtime.db.collection("dialog_operations").document(
        original_task["operation_id"]).get().to_dict()
    assert core_before_retry["state"] == "failed"
    assert core_before_retry["doc_id"] == doc_id
    intent_before_retry = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(
        original_task["operation_id"]).get().to_dict()
    assert intent_before_retry["doc_id"] == doc_id
    from core.integrations.firestore.operations import get_stage
    fill_stage = get_stage(runtime.db, original_task["operation_id"], "fill")
    assert fill_stage.failures == 1
    create_events = [event for event in sdk_fakes.trace
                     if event["boundary"] == "drive" and event["operation"] == "files.create"]
    get_events = [event for event in sdk_fakes.trace
                  if event["boundary"] == "docs" and event["operation"] == "documents.get"]
    assert len(create_events) == len(get_events) == 1
    assert sdk_fakes.trace.index(create_events[0]) < sdk_fakes.trace.index(get_events[0])
    error_messages = [payload["text"] for method, payload in runtime.sent
                      if method == "sendMessage" and "Error in core/fill" in payload.get("text", "")]
    assert len(error_messages) == 1
    assert "Dialog #1" in error_messages[0] and original_task["operation_id"] in error_messages[0]
    assert "provider" not in error_messages[0].lower()
    assert "Automatic retries have stopped" not in error_messages[0]
    assert provider_secret not in error_messages[0] and provider_secret not in caplog.text

    # Retry is explicitly routed through the model and accepted against dialog #1.
    runtime.route_response("retry", dialog_number=1)
    retry_text = "Please retry dialog 1."
    retry_update = {"update_id": 15504, "message": {
        "message_id": 64, "date": 1793320030,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7}, "text": retry_text,
    }}
    assert asyncio.run(runtime.webhook(retry_update), loop_factory=_selector_loop).status_code == 200
    retry_request_task = next(item["body"] for item in reversed(runtime.task_requests)
                              if item["body"].get("target_kind") == "telegram_dispatch"
                              and not item["body"].get("notification_id"))
    accepted = asyncio.run(runtime.dispatch(retry_request_task), loop_factory=_selector_loop)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {"status": "processed"}
    request_id = retry_request_task["operation_id"]
    retry_relation = session.collection("retry_requests").document(request_id).get().to_dict()
    assert retry_relation is not None
    assert retry_relation["accepted"] is True and retry_relation["draft_id"] == draft_id
    retry_task = next(item["body"] for item in reversed(runtime.task_requests)
                      if item["body"].get("target_kind") == "telegram_dispatch"
                      and item["body"].get("operation_id") == retry_relation["operation_id"]
                      and not item["body"].get("notification_id"))
    retry_row = session.collection("operations").document(retry_task["operation_id"]).get().to_dict()
    assert retry_row["save_intent_id"] == original_task["operation_id"]
    retry_request_row = session.collection("operations").document(request_id).get().to_dict()
    assert retry_row["input_ref"] == retry_request_row["input_ref"]
    assert retry_row["route_control"] == {
        "status": "ok", "kind": "retry", "dialog_number": 1,
        "retry_request_operation_id": request_id,
    }

    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })

    retried = asyncio.run(runtime.dispatch(retry_task), loop_factory=_selector_loop)
    assert retried.status_code == 200, retried.text
    assert retried.json() == {"status": "noop"}
    core_operation = runtime.db.collection("dialog_operations").document(
        retry_task["operation_id"]).get().to_dict()
    assert core_operation["doc_id"] == doc_id
    assert core_operation["immutable_input_ref"] == core_before_retry["immutable_input_ref"]
    assert core_operation["immutable_input_digest"] == core_before_retry["immutable_input_digest"]
    index_task = next(item for item in runtime.task_requests
                      if item["url"].endswith(f"/sync/doc/{doc_id}"))
    fill_events = [event for event in sdk_fakes.trace
                   if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]
    assert len(fill_events) == 1
    reader_text = fill_events[0]["arguments"]["body"]["requests"][0]["insertText"]["text"]
    assert source in reader_text and retry_text not in reader_text
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: ("folder-client", False, "Dialog A saved"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", lambda _doc_id: reader_text)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc", lambda _doc_id, **_kwargs: reader_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: reader_text)
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        transport = httpx.ASGITransport(app=vector_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector.test") as client:
            return await client.post(f"/sync/doc/{doc_id}", json=index_task["body"])

    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]))],
    ))
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["status"] == "ok"
    receipt = runtime.db.collection("index_completions").document(doc_id).collection(
        "generations").document(core_operation["desired_content_hash"]).get().to_dict()
    assert receipt["status"] == "completed"
    wake_kwargs = next(item for item in runtime.task_requests
                       if item["body"].get("target_kind") == "telegram_dispatch"
                       and item["body"].get("operation_id") == retry_task["operation_id"]
                       and item["body"].get("outbox_id") != retry_task.get("outbox_id"))
    wake = task_payload(get_outbox_record(runtime.db, wake_kwargs["body"]["outbox_id"]))
    completed = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert completed.status_code == 200, completed.text
    assert completed.json() == {"status": "processed"}
    retry_after = session.collection("operations").document(retry_task["operation_id"]).get().to_dict()
    draft_after = session.collection("drafts").document(draft_id).get().to_dict()
    assert retry_after["state"] == "completed" and retry_after["doc_id"] == doc_id
    assert draft_after["state"] == "cleaned" and draft_after["dialog_number"] == 1
    original_after = session.collection("operations").document(
        original_task["operation_id"]).get().to_dict()
    intent_after = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(
        original_task["operation_id"]).get().to_dict()
    assert original_after["state"] == "failed"
    assert original_after["target_draft_id"] == draft_id
    assert intent_before_retry["dispatch_operation_id"] == original_task["operation_id"]
    assert intent_before_retry["dispatch_token"]
    assert intent_after["dispatch_operation_id"] == intent_before_retry["dispatch_operation_id"]
    assert intent_after["dispatch_token"] == intent_before_retry["dispatch_token"]
    assert intent_after["doc_id"] == core_before_retry["doc_id"] == doc_id
    final_messages = [payload for method, payload in runtime.sent
                      if method == "sendMessage"
                      and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]
    assert len(final_messages) == 1
    metadata_calls = [event for event in sdk_fakes.trace
                      if event["boundary"] == "gemini"
                      and event["operation"] == "generate_content"
                      and "Infer a concise dialog title" in str(event["arguments"])]
    assert len(metadata_calls) == 1
    metadata_prompt = json.loads(metadata_calls[0]["arguments"]["contents"])
    assert metadata_prompt["user_request"] == "save this dialog"
    assert metadata_prompt["prepared_dialogue"] == [{"speaker": "Ada", "text": source}]
    prompt_text = json.dumps(metadata_prompt, ensure_ascii=False)
    assert retry_text not in prompt_text and provider_secret not in prompt_text

    requests_before_duplicate = dict(firestore_fake.documents)
    duplicate_request = asyncio.run(runtime.dispatch(retry_request_task), loop_factory=_selector_loop)
    duplicate_work = asyncio.run(runtime.dispatch(retry_task), loop_factory=_selector_loop)
    duplicate_wake = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert duplicate_request.status_code == duplicate_work.status_code == duplicate_wake.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]) == 1
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage"
                and "Dialog #1: Saved and indexed. Folder: Client." in payload.get("text", "")]) == 1
    assert len([path for path in firestore_fake.documents
                if path.startswith(f"telegram_sessions/{context_id}/operations/")]) == len(
        [path for path in requests_before_duplicate
         if path.startswith(f"telegram_sessions/{context_id}/operations/")]
    )


def test_default_sealed_finalizer_sdk_failures_exhaust_then_numbered_retry_reuses_doc(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """Three real finalizer reads settle once; numbered replay certifies the same Doc and intent."""
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "Dialog Finalize", "folder_path": ["Client"]}),
        )]))],
    ))
    doc_id = "doc-finalizer-sealed-retry"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for index in range(16):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": f"initial-{index}",
            "trashed": False,
        })

    source = "A sealed source line that must remain private."
    forward = {"update_id": 18001, "message": {
        "message_id": 101, "date": 1793320000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310000}, "text": source,
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack = next(row["body"] for row in runtime.task_requests
               if row["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    save_update = {"update_id": 18002, "message": {
        "message_id": 102, "date": 1793320010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    original_task = next(row["body"] for row in reversed(runtime.task_requests)
                         if row["body"].get("target_kind") == "telegram_dispatch"
                         and not row["body"].get("notification_id"))
    context_id, operation_id = original_task["context_id"], original_task["operation_id"]
    first = asyncio.run(runtime.dispatch(original_task), loop_factory=_selector_loop)
    assert first.status_code == 200, first.text

    index_task = next(row for row in runtime.task_requests
                      if row["url"].endswith(f"/sync/doc/{doc_id}"))
    fill_event = next(event for event in sdk_fakes.trace
                      if event["boundary"] == "docs"
                      and event["operation"] == "documents.batchUpdate")
    inserted_text = fill_event["arguments"]["body"]["requests"][0]["insertText"]["text"]
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: ("folder-client", False, "Dialog Finalize"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc",
                        lambda _doc_id: inserted_text)
    from core.integrations.google_docs import reader as docs_reader
    original_read_google_doc = docs_reader.read_google_doc
    monkeypatch.setattr(docs_reader, "read_google_doc", lambda _doc_id, **_kwargs: inserted_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: inserted_text)
    from services.vector_sync.app.main import app as vector_app

    async def sync_document():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=vector_app), base_url="http://vector.test",
        ) as client:
            return await client.post(index_task["url"].replace(
                "https://vector-sync.pytest.invalid", "",
            ), json=index_task["body"])

    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text='{"summary":"Sealed dialog","facts":[]}',
        )]))],
    ))
    for _ in range(5):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)],
        ))
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text

    wake_request = next(
        row["body"] for row in runtime.task_requests
        if row["body"].get("target_kind") == "telegram_dispatch"
        and row["body"].get("operation_id") == operation_id
        and row["body"].get("outbox_id") != original_task.get("outbox_id")
    )
    wake = task_payload(get_outbox_record(runtime.db, wake_request["outbox_id"]))

    # The finalizer must go back through the real Docs SDK after the managed receipt exists.
    sdk_fakes.reset()
    monkeypatch.setattr(docs_reader, "read_google_doc", original_read_google_doc)
    provider_secret = "synthetic-finalizer-docs-body-secret"
    for _ in range(3):
        sdk_fakes.docs.queue(
            "documents.get", error=ConnectionError(provider_secret + " private source"),
        )
    for _ in range(3):
        settled = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)

    session = runtime.db.collection("telegram_sessions").document(context_id)
    operation_row = session.collection("operations").document(operation_id).get().to_dict()
    draft_id = operation_row["target_draft_id"]
    draft = session.collection("drafts").document(draft_id).get().to_dict()
    common = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
    intent = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(operation_id).get().to_dict()
    stage_key = f"telegram.finalize.{operation_id}.docs"
    finalizer_stage = next(
        row.to_dict() for row in session.collection("operations").document(operation_id)
        .collection("stages").stream()
        if row.to_dict().get("stage_key") == stage_key
    )
    finalizer_calls = [event for event in sdk_fakes.trace
                       if event["boundary"] == "docs"
                       and event["operation"] == "documents.get"]
    assert len(finalizer_calls) == 3
    assert finalizer_stage["failures"] == finalizer_stage["attempts_started"] == 3
    assert operation_row["state"] == "failed" and common["state"] == "failed"
    assert common["doc_id"] == intent["doc_id"] == doc_id
    assert draft["dialog_number"] == 1 and draft["state"] == "failed"
    assert draft["immutable_save_intent_id"] == operation_id
    assert intent["dispatch_operation_id"] == operation_id and intent["dispatch_token"]
    assert settled.status_code == 200
    terminal_messages = [payload["text"] for method, payload in runtime.sent
                         if method == "sendMessage"
                         and "google_docs/finalization" in payload.get("text", "")]
    assert len(terminal_messages) == 1
    assert "failures=3" in terminal_messages[0] and operation_id in terminal_messages[0]
    assert "Automatic retries have stopped" in terminal_messages[0]
    assert provider_secret not in terminal_messages[0] and source not in terminal_messages[0]
    assert provider_secret not in caplog.text and source not in caplog.text
    duplicate = asyncio.run(runtime.dispatch(wake), loop_factory=_selector_loop)
    assert duplicate.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs"
                and event["operation"] == "documents.get"]) == 3

    # Canonical dialog-number retry reuses the same intent and already-created document.
    runtime.route_response("retry", dialog_number=1)
    retry_update = {"update_id": 18003, "message": {
        "message_id": 103, "date": 1793320020,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "retry dialog 1",
    }}
    assert asyncio.run(runtime.webhook(retry_update), loop_factory=_selector_loop).status_code == 200
    retry_request = next(row["body"] for row in reversed(runtime.task_requests)
                         if row["body"].get("target_kind") == "telegram_dispatch"
                         and not row["body"].get("notification_id"))
    assert asyncio.run(runtime.dispatch(retry_request), loop_factory=_selector_loop).status_code == 200
    retry_relation = session.collection("retry_requests").document(
        retry_request["operation_id"]).get().to_dict()
    assert retry_relation["accepted"] and retry_relation["draft_id"] == draft_id
    retry_task = next(row["body"] for row in reversed(runtime.task_requests)
                      if row["body"].get("target_kind") == "telegram_dispatch"
                      and row["body"].get("operation_id") == retry_relation["operation_id"]
                      and not row["body"].get("notification_id"))
    for index in range(4):
        sdk_fakes.docs.queue("documents.get", response={
            "revisionId": f"retry-read-{index}", "body": {"content": [
                {"paragraph": {"elements": [{"textRun": {"content": line + "\n"}}]}}
                for line in inserted_text.splitlines()
            ]},
        })
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": f"retry-{index}",
            "trashed": False,
        })
    retried = asyncio.run(runtime.dispatch(retry_task), loop_factory=_selector_loop)
    assert retried.status_code == 200, retried.text
    retry_core = runtime.db.collection("dialog_operations").document(
        retry_task["operation_id"]).get().to_dict()
    retry_intent = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(operation_id).get().to_dict()
    retry_draft = session.collection("drafts").document(draft_id).get().to_dict()
    assert retry_core["doc_id"] == doc_id
    assert retry_core["save_intent_id"] == operation_id
    assert retry_intent["doc_id"] == doc_id
    assert retry_intent["completion_certificate"]["operation_id"] == retry_task["operation_id"]
    assert retry_draft["state"] == "cleaned" and retry_draft["dialog_number"] == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive"
                and event["operation"] == "files.create"]) == 0
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs"
                and event["operation"] == "documents.batchUpdate"]) == 0
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage"
                and "Dialog #1: Saved and indexed." in payload.get("text", "")]) == 1


def test_default_save_oauth_failure_before_create_resumes_same_intent_once(
    firestore_fake, sdk_fakes, monkeypatch, caplog,
):
    """A real token endpoint error before create is charged without unknown-create gating."""
    import requests
    from google.genai import types
    from google.oauth2.credentials import Credentials
    from core.integrations.firestore.operations import get_stage
    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
    from tests.fixtures.sdk_fakes import install
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    runtime.route_response("save")
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text=json.dumps({"title": "OAuth Resume", "folder_path": ["Client"]}),
        )]))],
    ))

    original_refresh = next(
        original for obj, name, original, _replacement in install().values()
        if obj is Credentials and name == "refresh"
    )
    monkeypatch.setattr(Credentials, "refresh", original_refresh)
    token_calls = []

    class SyntheticSession:
        def __init__(self):
            self.adapters = {}

        def mount(self, prefix, adapter):
            self.adapters[prefix] = adapter

        def request(self, method, url, **_kwargs):
            token_calls.append((method, url))
            response = requests.Response()
            response.status_code = 503 if len(token_calls) == 1 else 200
            response.url = url
            response.headers["content-type"] = "application/json"
            response._content = (
                b'{"error":"temporarily_unavailable","secret":"synthetic-token-secret"}'
                if len(token_calls) == 1
                else b'{"access_token":"synthetic-access-token","expires_in":3600,"token_type":"Bearer"}'
            )
            return response

        def close(self):
            pass

    session = SyntheticSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    doc_id = "doc-after-oauth-recovery"
    sdk_fakes.drive.queue("files.create", response={"id": doc_id})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(8):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for index in range(8):
        sdk_fakes.drive.queue("files.get", response={
            "id": doc_id, "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": f"oauth-retry-{index}",
            "trashed": False,
        })

    forward = {"update_id": 18011, "message": {
        "message_id": 111, "date": 1793320100,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "forward_origin": {"type": "user", "sender_user": {"id": 44, "first_name": "Ada"},
                           "date": 1793310100}, "text": "OAuth recovery source line.",
    }}
    assert asyncio.run(runtime.webhook(forward), loop_factory=_selector_loop).status_code == 200
    ack = next(row["body"] for row in runtime.task_requests if row["body"].get("notification_id"))
    assert asyncio.run(runtime.deliver(ack), loop_factory=_selector_loop).status_code == 200
    save_update = {"update_id": 18012, "message": {
        "message_id": 112, "date": 1793320110,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7}, "text": "save this dialog",
    }}
    assert asyncio.run(runtime.webhook(save_update), loop_factory=_selector_loop).status_code == 200
    task = next(row["body"] for row in reversed(runtime.task_requests)
                if row["body"].get("target_kind") == "telegram_dispatch"
                and not row["body"].get("notification_id"))
    failed = asyncio.run(runtime.dispatch(task), loop_factory=_selector_loop)
    assert failed.status_code == 200
    context_id, operation_id = task["context_id"], task["operation_id"]
    queue_op = (runtime.db.collection("telegram_sessions").document(context_id)
                .collection("operations").document(operation_id).get().to_dict())
    draft_id = queue_op["target_draft_id"]
    common = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
    intent = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(operation_id).get().to_dict()
    create_stage = get_stage(runtime.db, operation_id, "create")
    assert create_stage.failures == 1
    assert create_stage.last_error.service == "google_drive"
    assert token_calls == [("POST", "https://oauth2.googleapis.com/token")]
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]
    assert common["state"] == "running" and common["doc_id"] is None
    assert intent["state"] not in {"unknown", "dispatched"}
    assert queue_op["state"] == "running"

    runtime.clock.advance()
    replay = next((row["body"] for row in reversed(runtime.task_requests)
                   if row["body"].get("target_kind") == "telegram_dispatch"
                   and row["body"].get("operation_id") == operation_id
                   and row["body"].get("outbox_id") != task.get("outbox_id")), task)
    resumed = asyncio.run(runtime.dispatch(replay), loop_factory=_selector_loop)
    assert resumed.status_code == 200
    common_after = runtime.db.collection("dialog_operations").document(operation_id).get().to_dict()
    intent_after = runtime.db.collection(SAVE_INTENTS_COLLECTION).document(operation_id).get().to_dict()
    create_calls = [event for event in sdk_fakes.trace
                    if event["boundary"] == "drive" and event["operation"] == "files.create"]
    assert len(create_calls) == 1
    assert common_after["doc_id"] == intent_after["doc_id"] == doc_id
    assert intent_after["state"] == "confirmed"
    assert get_stage(runtime.db, operation_id, "create").failures == 1
    assert len(token_calls) >= 2
    assert all(url == "https://oauth2.googleapis.com/token" for _, url in token_calls)
    persisted = repr({"common": common_after, "intent": intent_after,
                      "create_stage": get_stage(runtime.db, operation_id, "create"),
                      "queue_operation": queue_op})
    assert "synthetic-token-secret" not in persisted
    assert "synthetic-access-token" not in persisted
    assert "OAuth recovery source line." not in persisted
    assert "synthetic-token-secret" not in caplog.text
    assert "OAuth recovery source line." not in caplog.text
