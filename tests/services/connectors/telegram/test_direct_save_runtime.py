"""Focused direct-save completion binding coverage."""

from datetime import datetime, timezone
import hashlib
import asyncio
import json

import pytest


def _direct_save_selector_loop():
    """Keep a timer alive through asyncio.run executor shutdown on Windows."""
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    loop = _selector_loop()

    def tick():
        if not loop.is_closed():
            loop.call_later(0.01, tick)

    loop.call_soon(tick)
    return loop


def test_direct_completion_metadata_renders_the_valid_root_folder():
    from services.connectors.telegram.runtime.runtime import _direct_completion_paragraph

    assert _direct_completion_paragraph({
        "dialog_number": 2, "folder_path": [], "title": "Verified title",
    }) == "Saved as dialog #2 in /. Title: Verified title."


def test_verified_lifecycle_includes_title_and_distinguishes_root_from_unknown():
    from services.connectors.telegram.notifications import (
        _VerifiedTelegramLifecycle, build_lifecycle_body,
    )

    verified = dict(
        dialog_number=2, operation_id="operation-2", service="telegram",
        stage="save", reason="completed", completion_confirmed=True,
        title="Verified title",
    )
    root = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **verified, folder_path="/",
    ))
    unknown = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **verified, folder_path=None,
    ))

    assert "Saved and indexed. Folder: /. Title: Verified title." in root
    assert "Saved and indexed. Title: Verified title." in unknown


def test_completed_direct_default_date_and_title_follow_accepted_date_after_timezone_change(
    monkeypatch,
):
    from services.connectors.telegram.runtime.saving import (
        _normalize_completed_direct_arguments,
    )

    arguments = {"provider": "manual", "utterances": [
        {"speaker": "Ada", "text": "An accepted conversation."},
    ]}
    accepted_at = datetime(2026, 10, 30, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setenv("TZ", "UTC")
    accepted = _normalize_completed_direct_arguments(
        arguments, operation_date="2026-10-30", accepted_at=accepted_at,
    )
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    replayed = _normalize_completed_direct_arguments(
        arguments, operation_date="2026-10-30", accepted_at=accepted_at,
    )

    assert accepted.dialog_date == replayed.dialog_date == "2026-10-30"
    assert accepted.title == replayed.title == "TLBrain dialog 2026-10-30"
    assert "dialog_date" not in arguments and "title" not in arguments


def test_direct_save_binding_creates_numbered_conversation_under_parent_fence(
    firestore_fake,
):
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.queue import claim_head
    from services.connectors.telegram.repository import TelegramRepository

    now = datetime.now(timezone.utc)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    existing = repository.accept_forward(
        700, chat_id=77, thread_id=None,
        message={"message_id": 70, "text": "existing message",
                 "transcription_state": "not_required"}, now=now,
    )
    context_id = existing.context_id
    number_a = repository.bind_document(
        context_id, "doc-a", "conversation-a",
        metadata={"title": "Doc A"}, now=now,
    )
    repository.ensure_conversation(
        context_id, "conversation-a", kind="document", doc_id="doc-a",
        dialog_number=number_a, now=now,
    )
    session_ref = client.collection("telegram_sessions").document(context_id)
    session_ref.update({"active_conversation_id": "conversation-a",
                        "selected_dialog_number": number_a})
    accepted = repository.accept_command(
        701, chat_id=77, thread_id=None, user_id=7,
        command=parse_command("Discuss Doc A"), now=now,
    )
    context_id = accepted.context_id
    lease = claim_head(client, session_ref, owner="direct-save-worker", now=now)
    assert lease is not None and lease.operation_id == accepted.operation_id

    number_b = repository.bind_document(
        context_id, "doc-b", "conversation-b",
        metadata={"title": "Doc B", "provider": "manual"}, now=now,
        operation_id=accepted.operation_id, owner=lease.owner, fence=lease.fence,
        ensure_document_conversation=True,
    )

    assert number_b == 3
    context = session_ref.get().to_dict()
    assert context["active_conversation_id"] == "conversation-a"
    assert context["selected_dialog_number"] == number_a
    assert context["last_doc_id"] == "doc-b"
    assert repository.get_conversation(context_id, "conversation-b").doc_id == "doc-b"
    assert repository.get_conversation(context_id, "conversation-b").dialog_number == number_b
    dialogs = session_ref.collection("dialogs")
    assert dialogs.document(str(number_a)).get().to_dict()["doc_id"] == "doc-a"
    assert dialogs.document(str(number_b)).get().to_dict()["doc_id"] == "doc-b"


def test_direct_save_binding_rejects_stale_parent_fence_without_partial_state(
    firestore_fake,
):
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.queue import claim_head
    from services.connectors.telegram.repository import TelegramRepository

    now = datetime.now(timezone.utc)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        702, chat_id=77, thread_id=None, user_id=7,
        command=parse_command("Discuss Doc A"), now=now,
    )
    context_id = accepted.context_id
    session_ref = client.collection("telegram_sessions").document(context_id)
    lease = claim_head(client, session_ref, owner="direct-save-worker", now=now)
    assert lease is not None and lease.operation_id == accepted.operation_id

    with pytest.raises(PermissionError):
        repository.bind_document(
            context_id, "doc-b", "conversation-b", metadata={"title": "Doc B"},
            now=now, operation_id=accepted.operation_id, owner=lease.owner,
            fence=lease.fence + 1, ensure_document_conversation=True,
        )

    assert session_ref.collection("dialogs").document("1").get().exists is False
    assert repository.get_conversation(context_id, "conversation-b") is None
    assert session_ref.get().to_dict().get("last_doc_id") is None


def test_direct_save_rejects_completed_child_with_foreign_parent_provenance(
    firestore_fake,
):
    from core.dialogs.saving import accept_save
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.queue import claim_head
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id

    now = datetime.now(timezone.utc)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        703, chat_id=77, thread_id=None, user_id=7,
        command=parse_command("Discuss Doc A"), now=now,
    )
    context_id = accepted.context_id
    session_ref = client.collection("telegram_sessions").document(context_id)
    lease = claim_head(client, session_ref, owner="direct-save-worker", now=now)
    assert lease is not None and lease.operation_id == accepted.operation_id
    arguments = {
        "provider": "manual", "title": "Doc B",
        "utterances": [{"speaker": "Ada", "text": "A new conversation."}],
        "folder_path": ["Client"],
    }
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    ledger.save_turn(
        "telegram-doc-a", accepted.operation_id,
        model_response={"text": "", "parts": []},
        tool_calls=[{"id": "call-0", "name": "add_dialog",
                     "arguments": arguments, "ordinal": 0, "round": 0}],
    )
    saver = TelegramSaveRuntime(repository=repository, clock=lambda: now)
    input_ref, input_digest = saver._persist_direct_tool_input(
        context_id=context_id, operation_id=accepted.operation_id,
        call_ordinal=0, lease=lease, arguments=arguments,
    )
    child_id = direct_add_dialog_operation_id(accepted.operation_id, 0, arguments)
    accept_save(
        client, arguments, operation_id=child_id,
        caller_binding="telegram:foreign-context:foreign-parent",
        origin="telegram", now=now, origin_operation_ref="foreign/input",
    )
    child_ref = client.collection("dialog_operations").document(child_id)
    child_ref.update({"state": "completed", "doc_id": "foreign-doc"})

    with pytest.raises(PermissionError, match="parent tool input"):
        saver._completed_dialog_result(
            child_id, context_id, accepted.operation_id, 0, lease,
            input_ref=input_ref, input_digest=input_digest, arguments=arguments,
        )

    expected_conversation = "telegram-dialog-" + hashlib.sha256(child_id.encode()).hexdigest()
    assert repository.get_conversation(context_id, expected_conversation) is None
    binding_key = hashlib.sha256(b"foreign-doc").hexdigest()
    assert session_ref.collection("document_bindings").document(binding_key).get().exists is False
    assert session_ref.get().to_dict().get("last_doc_id") is None


def test_direct_folder_classification_normalizes_speakers_and_recovers_interrupted_stage(
    firestore_fake, sdk_fakes,
):
    from core.integrations.firestore.folders import increment_folder_speakers, upsert_folder
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import ReconciledStageFailure, RetryGovernor
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.queue import claim_head
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    now = datetime.now(timezone.utc)
    client = firestore_fake.client_class()
    upsert_folder("pytest-synthetic-root", "Root", None, client)
    upsert_folder("folder-client", "Client", "pytest-synthetic-root", client)
    increment_folder_speakers("folder-client", ["Ada"], db=client)
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        704, chat_id=77, thread_id=None, user_id=7,
        command=parse_command("Discuss Doc A"), now=now,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="direct-save-worker", now=now)
    assert lease is not None
    stage_key = "telegram.direct-classification.0"
    store = QueueOperationRetryStore(client, context_ref)
    store.reserve_attempt(
        operation_id=accepted.operation_id, stage_key=stage_key,
        owner=lease.owner, fence=lease.fence, attempt_id="interrupted-classification",
        now=now,
    )
    runtime = TelegramSaveRuntime(repository=repository, clock=lambda: now)
    governor = RetryGovernor(client, retry_store=store)
    arguments = {
        "provider": "manual", "utterances": [
            {"speaker": " Ada:: ", "text": "An indexed conversation."},
        ],
    }

    with pytest.raises(ReconciledStageFailure):
        runtime._direct_folder_decision(
            arguments, operation_id=accepted.operation_id, call_ordinal=0,
            lease=lease, retry_governor=governor,
        )
    classified, decision = runtime._direct_folder_decision(
        arguments, operation_id=accepted.operation_id, call_ordinal=0,
        lease=lease, retry_governor=governor,
    )

    assert classified["folder_id"] == "folder-client"
    assert decision["folder_id"] == "folder-client"
    assert not [event for event in sdk_fakes.trace
                if event["operation"] == "generate_content"]


@pytest.mark.parametrize(
    "folder_mode", ["explicit", "automatic", "unassigned", "ambiguous"],
)
@pytest.mark.parametrize("flow_mode", ["route_tool", "agent"])
def test_global_direct_save_waits_for_real_index_and_replays(
    firestore_fake, sdk_fakes, monkeypatch,
    folder_mode, flow_mode,
):
    import httpx
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.input_parts import cleanup_completed_input_parts
    from services.vector_sync.app.main import app as vector_app
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    accepted_timezone_replay = folder_mode == "explicit" and flow_mode == "route_tool"
    if accepted_timezone_replay:
        monkeypatch.setenv("TZ", "UTC")
        runtime.clock.value = datetime(2026, 10, 30, 23, 30, tzinfo=timezone.utc)
    target_folder_id = "folder-unassigned" if folder_mode == "unassigned" else "folder-client"
    if folder_mode == "ambiguous":
        from core.integrations.firestore.folders import increment_folder_speakers, upsert_folder

        upsert_folder("folder-client-b", "Client B", "pytest-synthetic-root", runtime.db)
        increment_folder_speakers("folder-client", ["Ada"], db=runtime.db)
        increment_folder_speakers("folder-client-b", ["Ada"], db=runtime.db)
    arguments = {
        "provider": "manual",
        "utterances": [{"speaker": "Ada", "text": "A directly saved conversation."}],
    }
    if not accepted_timezone_replay:
        arguments["title"] = "Direct Doc B"
    if folder_mode == "explicit":
        arguments["folder_path"] = ["Client"]
    runtime.route_response("substantive", substantive_text="Create a new conversation.",
                           **({"tool_choice": {"name": "add_dialog", "arguments": arguments}}
                              if flow_mode == "route_tool" else {}))
    if flow_mode == "agent":
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                function_call=types.FunctionCall(name="add_dialog", args=arguments),
            )]))],
        ))
    if folder_mode in {"automatic", "ambiguous"}:
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                text='{"folder_name":"Client","confidence":0.95}',
            )]))],
        ))
    if folder_mode == "unassigned":
        for _ in range(2):
            sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                    text='{"folder_name":"Client","confidence":0.2}',
                )]))],
            ))
        sdk_fakes.drive.queue("files.list", response={"files": []})
        sdk_fakes.drive.queue("files.create", response={"id": target_folder_id})
    sdk_fakes.drive.queue("files.create", response={"id": "direct-doc-b"})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": "direct-doc-b", "mimeType": "application/vnd.google-apps.document",
            "parents": [target_folder_id], "modifiedTime": modified,
        })

    update = {"update_id": 17001, "message": {
        "message_id": 91,
        "date": int(runtime.clock.value.timestamp()) if accepted_timezone_replay else 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Create a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    first = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
    assert first.status_code == 200, first.text
    session = runtime.db.collection("telegram_sessions").document(task["context_id"])
    global_conversation_id = "telegram-global-" + hashlib.sha256(
        f"{task['context_id']}\0{task['operation_id']}".encode()).hexdigest()
    ledger = FirestoreAgentLedger(
        runtime.db, context_id=task["context_id"], operation_id=task["operation_id"],
        owner="not-needed-for-read", fence=0, clock=runtime.clock,
    )
    operation = session.collection("operations").document(task["operation_id"]).get().to_dict()
    assert operation["state"] == "running"
    assert not session.get().to_dict().get("last_doc_id")
    assert not session.collection("document_bindings").stream()
    pending_turn = ledger.load_turn(global_conversation_id, task["operation_id"])
    assert pending_turn["tool_calls"][0]["name"] == "add_dialog"
    assert not pending_turn["tool_results"]
    index_task = next(item for item in runtime.task_requests
                      if item["url"].startswith("https://vector-sync.pytest.invalid/sync/doc/"))
    core_operation = runtime.db.collection("dialog_operations").document(
        index_task["body"]["operation_id"]).get().to_dict()
    assert core_operation["state"] == "running"
    fill = next(event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate")
    reader_text = fill["arguments"]["body"]["requests"][0]["insertText"]["text"]
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: (target_folder_id, False, "Direct Doc B"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", lambda _doc_id: reader_text)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc",
                        lambda _doc_id: reader_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: reader_text)

    async def sync_document():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=vector_app), base_url="http://vector.test",
        ) as client:
            return await client.post(index_task["url"].replace(
                "https://vector-sync.pytest.invalid", "",
            ), json=index_task["body"])

    sdk_fakes.gemini.queue(
        "generate_content", response=types.GenerateContentResponse(candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(
                text='{"summary":"Direct conversation","facts":[]}',
            )]))
        ]),
    )
    for _ in range(5):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)],
        ))
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text
    if flow_mode == "agent":
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                text="The direct-doc-b save is complete.",
            )]))],
        ))
    wake = next(item for item in runtime.task_requests
                if item["body"].get("target_kind") == "telegram_dispatch"
                and item["body"].get("operation_id") == task["operation_id"]
                and item["body"].get("outbox_id") != task["outbox_id"])
    wake_record = get_outbox_record(runtime.db, wake["body"]["outbox_id"])
    core_operation_ref = runtime.db.collection("dialog_operations").document(
        index_task["body"]["operation_id"])

    # Simulate a single binding transaction outage after the real Vector Sync
    # receipt commits. The parent must stay pending and recover after the
    # accepted core input parts have been cleaned up.
    binding_path = f"telegram_sessions/{task['context_id']}/dialogs/1"
    firestore_fake.fail_next("commit", path=binding_path,
                             exception=RuntimeError("synthetic bind transaction outage"))
    interrupted = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                              loop_factory=_direct_save_selector_loop)
    assert interrupted.status_code == 503, interrupted.text
    assert session.collection("operations").document(task["operation_id"]).get().to_dict()["state"] == "running"
    assert not session.collection("document_bindings").stream()
    assert not ledger.load_turn(global_conversation_id, task["operation_id"])["tool_results"]
    core_operation = core_operation_ref.get().to_dict()
    assert core_operation["state"] == "completed"
    accepted_completion_ref = core_operation["index_completion_ref"]
    core_operation_ref.update({"index_completion_ref": "index_completions/direct-doc-b/foreign"})
    untrusted = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                            loop_factory=_direct_save_selector_loop)
    assert untrusted.status_code == 200, untrusted.text
    assert session.collection("operations").document(task["operation_id"]).get().to_dict()["state"] == "running"
    assert not session.collection("document_bindings").stream()
    assert not ledger.load_turn(global_conversation_id, task["operation_id"])["tool_results"]
    core_operation_ref.update({"index_completion_ref": accepted_completion_ref})
    core_operation = core_operation_ref.get().to_dict()
    input_ref = core_operation["immutable_input_ref"]
    input_operation_id = input_ref.split("/")[1]
    input_parts = runtime.db.collection("dialog_operations").document(
        input_operation_id).collection("input_parts")
    assert any(row.id.startswith("part-") for row in input_parts.stream())
    assert cleanup_completed_input_parts(runtime.db, operation_limit=20, part_limit=100) > 0
    assert input_parts.document("manifest").get().exists
    assert not any(row.id.startswith("part-") for row in input_parts.stream())

    before_timezone_replay = len(sdk_fakes.trace)
    if accepted_timezone_replay:
        monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    completed = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                            loop_factory=_direct_save_selector_loop)
    assert completed.status_code == 200, completed.text
    queue_row = session.collection("operations").document(task["operation_id"]).get().to_dict()
    context = session.get().to_dict()
    turns = ledger.load_turn(global_conversation_id, task["operation_id"])
    assert queue_row["state"] == "completed", {
        "queue": {key: queue_row.get(key) for key in (
            "state", "error", "failure_reason", "route_control", "reply_state",
        )},
        "sent": runtime.sent,
    }
    result = turns["tool_results"]["0"]
    assert result["provider"] == "manual"
    if accepted_timezone_replay:
        assert result["title"] == "TLBrain dialog 2026-10-30"
        assert len(sdk_fakes.trace) == before_timezone_replay
    expected_path = ["_unassigned"] if folder_mode == "unassigned" else ["Client"]
    assert result["folder_path"] == expected_path
    if folder_mode == "unassigned":
        assert "No confident folder match" in result["notice"]
    elif folder_mode == "ambiguous":
        assert "Speaker matches were ambiguous" in result["notice"]
    assert result["status"] == "synced" and result["indexed"] is True
    assert result["doc_id"] == "direct-doc-b"
    assert context["last_doc_id"] == "direct-doc-b"
    binding = next(item.to_dict() for item in session.collection("document_bindings").stream())
    assert binding["doc_id"] == "direct-doc-b" and binding["dialog_number"] == 1
    conversation = session.collection("conversations").document(binding["conversation_id"]).get().to_dict()
    assert conversation["kind"] == "document" and conversation["dialog_number"] == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == (
        2 if folder_mode == "unassigned" else 1
    )
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]) == 1
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage" and "direct-doc-b" in payload.get("text", "")]) == 1
    counts_after_completion = (
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "gemini" and event["operation"] == "generate_content"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "drive" and event["operation"] == "files.create"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]),
        len([payload for method, payload in runtime.sent
             if method == "sendMessage" and "direct-doc-b" in payload.get("text", "")]),
    )
    duplicate = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                            loop_factory=_direct_save_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    counts_after_duplicate = (
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "gemini" and event["operation"] == "generate_content"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "drive" and event["operation"] == "files.create"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]),
        len([payload for method, payload in runtime.sent
             if method == "sendMessage" and "direct-doc-b" in payload.get("text", "")]),
    )
    assert counts_after_duplicate == counts_after_completion


@pytest.mark.parametrize(("status", "expected_failures"), [(403, 1), (503, 3)])
def test_global_direct_save_terminal_provider_failure_settles_from_child_stage(
    firestore_fake, sdk_fakes, monkeypatch, status, expected_failures,
):
    from googleapiclient.errors import HttpError
    from httplib2 import Response
    from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "Direct Failure Doc",
        "folder_path": ["Client"],
        "utterances": [{"speaker": "Ada", "text": "A source line."}],
    }
    runtime.route_response(
        "substantive", substantive_text="Create a new conversation.",
        tool_choice={"name": "add_dialog", "arguments": arguments},
    )
    sdk_fakes.drive.queue("files.create", response={"id": "direct-failure-doc"})
    provider_secret = "synthetic-google-provider-body-secret"
    for _ in range(expected_failures):
        sdk_fakes.docs.queue("documents.get", error=HttpError(
            Response({"status": str(status), "reason": "Forbidden" if status == 403 else "Unavailable"}),
            provider_secret.encode(),
        ))
    update = {"update_id": 17101 + status, "message": {
        "message_id": 191, "date": 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Create a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))

    session = runtime.db.collection("telegram_sessions").document(task["context_id"])
    child_id = direct_add_dialog_operation_id(task["operation_id"], 0, arguments)
    from core.integrations.firestore.operations import get_stage
    queue_ref = session.collection("operations").document(task["operation_id"])
    child_ref = runtime.db.collection("dialog_operations").document(child_id)
    attempts = 1 if status == 403 else 3
    for attempt in range(attempts):
        if attempt:
            runtime.clock.advance()
        settled = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
        assert settled.status_code == 200, settled.text
        current_queue = queue_ref.get().to_dict()
        expected_state = "failed" if attempt + 1 == attempts else "running"
        assert current_queue["state"] == expected_state
        assert get_stage(runtime.db, child_id, "fill").failures == attempt + 1
        if expected_state == "running":
            assert not [payload for method, payload in runtime.sent
                        if method == "sendMessage" and "Error in google_docs/fill"
                        in payload.get("text", "")]
            assert not [payload for method, payload in runtime.sent
                        if method == "sendMessage" and "Please clarify" in payload.get("text", "")]
    queue = queue_ref.get().to_dict()
    child = child_ref.get().to_dict()
    assert queue["state"] == "failed"
    assert child["doc_id"] == "direct-failure-doc"
    assert child["state"] in {"failed", "running"}
    assert child.get("error") is not None or get_stage(runtime.db, child_id, "fill").last_error is not None
    provider_reads = [event for event in sdk_fakes.trace
                      if event["boundary"] == "docs" and event["operation"] == "documents.get"]
    assert len(provider_reads) == expected_failures
    messages = [payload["text"] for method, payload in runtime.sent
                if method == "sendMessage" and "Error in google_docs/fill" in payload.get("text", "")]
    assert len(messages) == 1
    assert f"failures={expected_failures}" in messages[0]
    assert task["operation_id"] in messages[0]
    if status == 403:
        assert "failures=1" in messages[0]
        assert "Automatic retries have stopped" not in messages[0]
    else:
        assert "Automatic retries have stopped" in messages[0]
    assert "Please clarify" not in messages[0]
    assert provider_secret not in messages[0]

    duplicate = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
    assert duplicate.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.get"]) == expected_failures
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage" and "Error in google_docs/fill" in payload.get("text", "")]) == 1


def test_global_direct_save_rejects_pure_invalid_input_safely_before_child_acceptance(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "", "folder_path": ["Client"],
        "utterances": [{"speaker": "Ada", "text": "A source line."}],
    }
    runtime.route_response(
        "substantive", substantive_text="Create a new conversation.",
        tool_choice={"name": "add_dialog", "arguments": arguments},
    )
    update = {"update_id": 17403, "message": {
        "message_id": 193, "date": 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Create a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))

    response = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)

    assert response.status_code == 200, response.text
    assert runtime.db.collection("telegram_sessions").document(task["context_id"]).collection(
        "operations").document(task["operation_id"]).get().to_dict()["state"] == "failed"
    assert not [row for path, row in firestore_fake.documents.items()
                if path.startswith("dialog_operations/")]
    error_texts = [payload["text"] for method, payload in runtime.sent
                   if method == "sendMessage" and "input is invalid" in payload.get("text", "")]
    assert len(error_texts) == 1
    assert "core/" not in error_texts[0] and "failures=" not in error_texts[0]


def test_global_agent_function_call_blank_title_refuses_once_without_save_work(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from services.connectors.telegram.errors import TELEGRAM_SAVE_INPUT_REFUSAL
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "", "folder_path": ["Client"],
        "utterances": [{"speaker": "Ada", "text": "A source line."}],
    }
    runtime.route_response(
        "substantive", substantive_text="Create a new conversation.",
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="add_dialog", args=arguments),
        )]))],
    ))
    update = {"update_id": 17404, "message": {
        "message_id": 194, "date": 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Create a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))

    response = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)

    assert response.status_code == 200, response.text
    queue_ref = (runtime.db.collection("telegram_sessions").document(task["context_id"])
                 .collection("operations").document(task["operation_id"]))
    assert queue_ref.get().to_dict()["state"] == "failed"
    assert not [path for path in firestore_fake.documents if path.startswith("dialog_operations/")]
    replies = [payload["text"] for method, payload in runtime.sent
               if method == "sendMessage" and payload.get("text") == TELEGRAM_SAVE_INPUT_REFUSAL]
    assert len(replies) == 1
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] in {"drive", "docs"}]
    calls_before_duplicate = len([event for event in sdk_fakes.trace
                                  if event["boundary"] == "gemini"
                                  and event["operation"] == "generate_content"])
    duplicate = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
    assert duplicate.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini"
                and event["operation"] == "generate_content"]) == calls_before_duplicate
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage" and payload.get("text") == TELEGRAM_SAVE_INPUT_REFUSAL]) == 1


def test_docsession_function_call_blank_title_refuses_without_losing_selected_doc_a(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from core.integrations.firestore.client import COLLECTION_NAME
    from services.connectors.telegram.errors import TELEGRAM_SAVE_INPUT_REFUSAL
    from services.connectors.telegram.repository import TelegramRepository
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "", "folder_path": ["Client"],
        "utterances": [{"speaker": "Ada", "text": "A source line."}],
    }
    runtime.route_response(
        "substantive", target_ref="doc-a",
        substantive_text="Discuss Doc A and record a new conversation.",
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="add_dialog", args=arguments),
        )]))],
    ))
    update = {"update_id": 17405, "message": {
        "message_id": 195, "date": 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Discuss Doc A and record a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    repository = TelegramRepository(client=runtime.db)
    context_id = task["context_id"]
    runtime.db.collection(COLLECTION_NAME).document("doc-a").set({
        "source_file": "Existing Doc A", "parent_id": "folder-client",
        "date": "2026-10-31", "status": "synced",
    })
    context_ref = runtime.db.collection("telegram_sessions").document(context_id)
    number_a = repository.bind_document(
        context_id, "doc-a", "conversation-a", metadata={"title": "Existing Doc A"},
        now=runtime.clock(),
    )
    repository.ensure_conversation(
        context_id, "conversation-a", kind="document", doc_id="doc-a",
        dialog_number=number_a, now=runtime.clock(),
    )
    context_ref.update({"active_conversation_id": "conversation-a",
                        "selected_dialog_number": number_a, "last_doc_id": "doc-a"})

    response = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)

    assert response.status_code == 200, response.text
    queue = context_ref.collection("operations").document(task["operation_id"]).get().to_dict()
    assert queue["state"] == "failed"
    selected = context_ref.get().to_dict()
    assert selected["active_conversation_id"] == "conversation-a"
    assert selected["selected_dialog_number"] == number_a
    assert selected["last_doc_id"] == "doc-a"
    assert not [path for path in firestore_fake.documents if path.startswith("dialog_operations/")]
    replies = [payload["text"] for method, payload in runtime.sent
               if method == "sendMessage" and payload.get("text") == TELEGRAM_SAVE_INPUT_REFUSAL]
    assert len(replies) == 1
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] in {"drive", "docs"}]
    calls_before_duplicate = len([event for event in sdk_fakes.trace
                                  if event["boundary"] == "gemini"
                                  and event["operation"] == "generate_content"])
    duplicate = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
    assert duplicate.status_code == 200
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini"
                and event["operation"] == "generate_content"]) == calls_before_duplicate
    assert len([payload for method, payload in runtime.sent
                if method == "sendMessage" and payload.get("text") == TELEGRAM_SAVE_INPUT_REFUSAL]) == 1


@pytest.mark.parametrize(("status", "attempts"), [(403, 1), (503, 3)])
def test_docsession_direct_save_failure_keeps_selected_document_and_typed_pending(
    firestore_fake, sdk_fakes, monkeypatch, status, attempts,
):
    from google.genai import types
    from googleapiclient.errors import HttpError
    from httplib2 import Response
    from core.integrations.firestore.client import COLLECTION_NAME
    from core.integrations.firestore.folders import upsert_folder
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.tools import direct_add_dialog_operation_id
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "Direct Doc B", "folder_path": ["Client"],
        "utterances": [{"speaker": "Ada", "text": "A source line."}],
    }
    runtime.route_response(
        "substantive", target_ref="doc-a",
        substantive_text="Discuss Doc A and record a new conversation.",
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="add_dialog", args=arguments),
        )]))],
    ))
    sdk_fakes.drive.queue("files.create", response={"id": "docsession-failure-b"})
    provider_secret = "synthetic-docsession-provider-body-secret"
    for _ in range(attempts):
        sdk_fakes.docs.queue("documents.get", error=HttpError(
            Response({"status": str(status), "reason": "Forbidden" if status == 403 else "Unavailable"}),
            provider_secret.encode(),
        ))
    update = {"update_id": 17503 + status, "message": {
        "message_id": 194, "date": 1793321000,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Discuss Doc A and record a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    repository = TelegramRepository(client=runtime.db)
    context_id = task["context_id"]
    runtime.db.collection(COLLECTION_NAME).document("doc-a").set({
        "source_file": "Existing Doc A", "parent_id": "folder-client",
        "date": "2026-10-31", "status": "synced",
    })
    context_ref = runtime.db.collection("telegram_sessions").document(context_id)
    number_a = repository.bind_document(
        context_id, "doc-a", "conversation-a", metadata={"title": "Existing Doc A"},
        now=runtime.clock(),
    )
    repository.ensure_conversation(
        context_id, "conversation-a", kind="document", doc_id="doc-a",
        dialog_number=number_a, now=runtime.clock(),
    )
    context_ref.update({"active_conversation_id": "conversation-a",
                        "selected_dialog_number": number_a, "last_doc_id": "doc-a"})
    child_id = direct_add_dialog_operation_id(task["operation_id"], 0, arguments)
    from core.integrations.firestore.operations import get_stage
    queue_ref = context_ref.collection("operations").document(task["operation_id"])
    for attempt in range(attempts):
        if attempt:
            runtime.clock.advance()
        result = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
        assert result.status_code == 200, result.text
        expected = "failed" if attempt + 1 == attempts else "running"
        assert queue_ref.get().to_dict()["state"] == expected
        assert get_stage(runtime.db, child_id, "fill").failures == attempt + 1
        if expected == "running":
            assert not [payload for method, payload in runtime.sent
                        if method == "sendMessage" and "Error in google_docs/fill"
                        in payload.get("text", "")]
            assert not [payload for method, payload in runtime.sent
                        if method == "sendMessage" and "Please clarify" in payload.get("text", "")]
        selection = context_ref.get().to_dict()
        assert selection["active_conversation_id"] == "conversation-a"
        assert selection["selected_dialog_number"] == number_a
        assert selection["last_doc_id"] == "doc-a"
    terminal = [payload["text"] for method, payload in runtime.sent
                if method == "sendMessage" and "Error in google_docs/fill" in payload.get("text", "")]
    assert len(terminal) == 1
    assert f"failures={attempts}" in terminal[0]
    assert "Please clarify" not in terminal[0]
    assert provider_secret not in terminal[0]
def test_document_function_call_pending_replay_preserves_selected_doc_a(
    firestore_fake, sdk_fakes, monkeypatch,
):
    import httpx
    from google.genai import types
    from core.integrations.firestore.outbox import get_outbox_record, task_payload
    from core.integrations.firestore.client import COLLECTION_NAME
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.vector_sync.app.main import app as vector_app
    from tests.services.connectors.telegram.test_save_runtime import (
        _default_telegram_runtime_harness,
    )
    from tests.services.connectors.telegram.test_production_runtime import _selector_loop

    runtime = _default_telegram_runtime_harness(firestore_fake, sdk_fakes, monkeypatch)
    arguments = {
        "provider": "manual", "title": "Direct Doc B",
        "utterances": [{"speaker": "Ada", "text": "A directly saved conversation."}],
    }
    from core.integrations.firestore.folders import increment_folder_speakers, upsert_folder
    upsert_folder("folder-client-b", "Client B", "pytest-synthetic-root", runtime.db)
    increment_folder_speakers("folder-client", ["Ada"], db=runtime.db)
    increment_folder_speakers("folder-client-b", ["Ada"], db=runtime.db)
    runtime.route_response(
        "substantive", target_ref="doc-a",
        substantive_text="Discuss Doc A and record a new conversation.",
    )
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            function_call=types.FunctionCall(name="add_dialog", args=arguments),
        )]))],
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text='{"folder_name":"Client","confidence":0.95}',
        )]))],
    ))
    sdk_fakes.drive.queue("files.create", response={"id": "direct-doc-b"})
    empty_document = {"revisionId": "rev-empty", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}},
    ]}}
    for _ in range(12):
        sdk_fakes.docs.queue("documents.get", response=empty_document)
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-filled"})
    for modified in ("fill-time", "post-fill-time", *(f"readback-{i}" for i in range(10))):
        sdk_fakes.drive.queue("files.get", response={
            "id": "direct-doc-b", "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-client"], "modifiedTime": modified,
        })

    update = {"update_id": 17002, "message": {
        "message_id": 92, "date": 1793321010,
        "chat": {"id": 77, "type": "private"}, "from": {"id": 7},
        "text": "Discuss Doc A and record a new conversation.",
    }}
    assert asyncio.run(runtime.webhook(update), loop_factory=_selector_loop).status_code == 200
    task = next(item["body"] for item in reversed(runtime.task_requests)
                if item["body"].get("target_kind") == "telegram_dispatch"
                and not item["body"].get("notification_id"))
    repository = TelegramRepository(client=runtime.db)
    context_id = task["context_id"]
    runtime.db.collection(COLLECTION_NAME).document("doc-a").set({
        "source_file": "Existing Doc A", "parent_id": "folder-client",
        "date": "2026-10-31", "status": "synced",
    })
    context_ref = runtime.db.collection("telegram_sessions").document(context_id)
    number_a = repository.bind_document(
        context_id, "doc-a", "conversation-a", metadata={"title": "Existing Doc A"},
        now=runtime.clock(),
    )
    repository.ensure_conversation(
        context_id, "conversation-a", kind="document", doc_id="doc-a",
        dialog_number=number_a, now=runtime.clock(),
    )
    repository.append_turn(
        context_id, "conversation-a", turn_id="prior-a-turn",
        operation_id="prior-a-operation", role="user",
        parts=[{"text": "Existing Doc A history."}], now=runtime.clock(),
    )
    context_ref.update({"active_conversation_id": "conversation-a",
                        "selected_dialog_number": number_a, "last_doc_id": "doc-a"})

    first = asyncio.run(runtime.dispatch(task), loop_factory=_direct_save_selector_loop)
    assert first.status_code == 200, first.text
    queue_row = context_ref.collection("operations").document(task["operation_id"]).get().to_dict()
    assert queue_row["state"] == "running"
    assert not context_ref.collection("document_bindings").document(
        hashlib.sha256(b"direct-doc-b").hexdigest()).get().exists
    # The saved function call remains pending with no persisted tool result.
    parent_lease_owner = context_ref.get().to_dict()["lock_owner"]
    parent_fence = context_ref.get().to_dict()["fence"]
    ledger = FirestoreAgentLedger(
        runtime.db, context_id=context_id, operation_id=task["operation_id"],
        owner=parent_lease_owner, fence=parent_fence, clock=runtime.clock,
    )
    turns = ledger.load_turn("conversation-a", task["operation_id"])
    assert turns["tool_calls"][0]["name"] == "add_dialog"
    assert not turns["tool_results"]
    index_task = next(item for item in runtime.task_requests
                      if item["url"].startswith("https://vector-sync.pytest.invalid/sync/doc/"))
    core_operation = runtime.db.collection("dialog_operations").document(
        index_task["body"]["operation_id"]).get().to_dict()
    fill = next(event for event in sdk_fakes.trace
                if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate")
    reader_text = fill["arguments"]["body"]["requests"][0]["insertText"]["text"]
    monkeypatch.setattr("services.vector_sync.app.main.get_file_parent_folder_id",
                        lambda _doc_id: ("folder-client", False, "Direct Doc B"))
    monkeypatch.setattr("services.vector_sync.app.main.read_google_doc", lambda _doc_id: reader_text)
    monkeypatch.setattr("core.integrations.google_docs.reader.read_google_doc",
                        lambda _doc_id: reader_text)
    monkeypatch.setattr("core.indexing.processor.read_google_doc", lambda _doc_id: reader_text)

    async def sync_document():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=vector_app), base_url="http://vector.test",
        ) as client:
            return await client.post(index_task["url"].replace(
                "https://vector-sync.pytest.invalid", "",
            ), json=index_task["body"])

    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[types.Part(
            text='{"summary":"Direct conversation","facts":[]}',
        )]))
    ]))
    for _ in range(5):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)],
        ))
    indexed = asyncio.run(sync_document(), loop_factory=_selector_loop)
    assert indexed.status_code == 200, indexed.text
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text="The new conversation is saved.",
        )]))],
    ))
    model_calls_before_wake = len([event for event in sdk_fakes.trace
                                   if event["boundary"] == "gemini"
                                   and event["operation"] == "generate_content"])
    wake = next(item for item in runtime.task_requests
                if item["body"].get("target_kind") == "telegram_dispatch"
                and item["body"].get("operation_id") == task["operation_id"]
                and item["body"].get("outbox_id") != task["outbox_id"])
    wake_record = get_outbox_record(runtime.db, wake["body"]["outbox_id"])
    binding_path = f"telegram_sessions/{context_id}/dialogs/{number_a + 1}"
    firestore_fake.fail_next(
        "commit", path=binding_path,
        exception=RuntimeError("synthetic Docsession binding outage"),
    )
    interrupted = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                              loop_factory=_direct_save_selector_loop)
    assert interrupted.status_code == 503, interrupted.text
    assert context_ref.collection("operations").document(task["operation_id"]).get().to_dict()[
        "state"] == "running"
    assert not context_ref.collection("document_bindings").document(
        hashlib.sha256(b"direct-doc-b").hexdigest()).get().exists
    assert not ledger.load_turn("conversation-a", task["operation_id"])["tool_results"]
    assert runtime.db.collection("dialog_operations").document(core_operation["operation_id"]).get().to_dict()[
        "state"] == "completed"
    completed = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                            loop_factory=_direct_save_selector_loop)
    assert completed.status_code == 200, completed.text
    queue_row = context_ref.collection("operations").document(task["operation_id"]).get().to_dict()
    turns = ledger.load_turn("conversation-a", task["operation_id"])
    assert queue_row["state"] == "completed"
    assert turns["tool_results"]["0"]["doc_id"] == "direct-doc-b"
    final_bodies = [payload["text"] for method, payload in runtime.sent
                    if method == "sendMessage" and "The new conversation is saved."
                    in payload.get("text", "")]
    assert len(final_bodies) == 1
    assert f"#{number_a + 1}" in final_bodies[0]
    assert "Client" in final_bodies[0]
    assert final_bodies[0].count("Speaker matches were ambiguous") == 1
    context = context_ref.get().to_dict()
    assert context["active_conversation_id"] == "conversation-a"
    assert context["selected_dialog_number"] == number_a
    assert context["last_doc_id"] == "direct-doc-b"
    binding = context_ref.collection("document_bindings").document(
        hashlib.sha256(b"direct-doc-b").hexdigest()).get().to_dict()
    assert binding["dialog_number"] == number_a + 1
    conversation_b = repository.get_conversation(context_id, binding["conversation_id"])
    assert conversation_b.doc_id == "direct-doc-b"
    assert repository.list_turns(context_id, binding["conversation_id"]) == []
    history_a = repository.list_turns(context_id, "conversation-a")
    assert [(turn["role"], turn["operation_id"]) for turn in history_a] == [
        ("user", "prior-a-operation"),
        ("user", task["operation_id"]),
        ("model", task["operation_id"]),
    ]
    assert history_a[0]["parts"] == [{"text": "Existing Doc A history."}]
    serialized_a = json.dumps([turn["parts"] for turn in history_a], sort_keys=True)
    assert "A directly saved conversation." not in serialized_a
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini"
                and event["operation"] == "generate_content"]) == model_calls_before_wake + 1
    counts_after_completion = (
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "gemini" and event["operation"] == "generate_content"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "drive" and event["operation"] == "files.create"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]),
        len([payload for method, payload in runtime.sent if method == "sendMessage"]),
    )
    duplicate = asyncio.run(runtime.dispatch(task_payload(wake_record)),
                            loop_factory=_direct_save_selector_loop)
    assert duplicate.status_code == 200, duplicate.text
    counts_after_duplicate = (
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "gemini" and event["operation"] == "generate_content"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "drive" and event["operation"] == "files.create"]),
        len([event for event in sdk_fakes.trace
             if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"]),
        len([payload for method, payload in runtime.sent if method == "sendMessage"]),
    )
    assert counts_after_duplicate == counts_after_completion
