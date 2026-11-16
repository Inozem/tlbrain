"""Legacy replay, durable payload, and public protocol compatibility tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from tests.fixtures.telegram_contextual import (
    BASE_TIME,
    make_contextual_scenario,
    seed_history,
)


EXPECTED_ADDITIVE_LEGACY_DEFAULTS = {
    "context": {"pending_clarification_id": None, "clarification_generation": 0},
    "command": {
        "message_id": None, "message_date": None, "reply_to_message_id": None,
        "observed_clarification_id": None,
        "observed_clarification_generation": None,
        "observed_presentation_id": None,
        "observed_question_delivery_ref": None,
    },
    "route_control": {"routing_protocol": None},
    "conversation": {"last_presented_list_id": None},
}


def test_native_opaque_checkpoint_and_search_history_survive_ledger_restart(
    firestore_fake, sdk_fakes,
):
    """Persist real native SDK parts and verified search exchanges through the ledger."""
    from google.genai import types

    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import _calls_from
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.runtime import _GovernedModel
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client = firestore_fake.client_class()
    scenario = make_contextual_scenario(unrelated_count=0)
    seeded = seed_history(
        client, context_id=scenario.context_id,
        operation_id=scenario.operation_id,
        conversation_id="legacy-search-session",
        now=BASE_TIME,
    )
    ledger = seeded["ledger"]
    native_part = types.Part(
        function_call=types.FunctionCall(
            id="call-native-legacy-001",
            name="get_transcript",
            args={"doc_id": "doc-b"},
        ),
        thought_signature=b"opaque-native-signature\x00legacy",
    )
    native_response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[native_part]))],
    )
    expected_wire = {
        "candidates": [{"content": {"parts": [{
            "function_call": {
                "id": "call-native-legacy-001",
                "name": "get_transcript",
                "args": {"doc_id": "doc-b"},
            },
            "thought_signature": "b3BhcXVlLW5hdGl2ZS1zaWduYXR1cmUAbGVnYWN5",
        }]}}]
    }
    sdk_fakes.gemini.queue("generate_content", response=native_response)
    native_model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-native-sdk-key"))
    governed_model = _GovernedModel(
        native_model, None, scenario.operation_id, None,
        lambda: BASE_TIME, lambda: True,
    )
    serialized_response = governed_model.generate(
        contents=[{"role": "user", "parts": [{"text": "retrieve the quoted passage"}]}],
        mode="tool_loop",
    )
    ledger.save_request("legacy-search-session", scenario.operation_id,
                        "retrieve the quoted passage")
    ledger.save_turn(
        "legacy-search-session", scenario.operation_id,
        model_response=serialized_response, tool_calls=_calls_from(serialized_response),
    )
    legacy_result = {
        "doc_id": "doc-b", "segments": [{"order_index": 2,
                                             "quote": "The quoted monthly price is €42."}],
        "opaque": {"provider_key": "unchanged", "bytes": [0, 255, 1]},
    }
    ledger.save_tool_result("legacy-search-session", scenario.operation_id, 0, legacy_result)
    durable_before_replay = deepcopy(firestore_fake.documents)

    restarted = FirestoreAgentLedger(
        client, context_id=scenario.context_id, operation_id=scenario.operation_id,
        owner=seeded["owner"], fence=seeded["fence"], clock=lambda: BASE_TIME,
    )
    checkpoint = restarted.load_turn("legacy-search-session", scenario.operation_id)
    assert checkpoint["model_response"] == expected_wire
    assert checkpoint["tool_calls"][0]["name"] == "get_transcript"
    assert checkpoint["tool_calls"][0]["arguments"] == {"doc_id": "doc-b"}
    assert checkpoint["tool_results"]["0"] == legacy_result
    assert restarted.load_request("legacy-search-session", scenario.operation_id) == (
        "retrieve the quoted passage"
    )
    exchange = restarted.load_search_exchange(
        seeded["references"][0], conversation_id="legacy-search-session",
        operation_id=scenario.operation_id,
        checkpoint_conversation_id="private-search-checkpoint",
        call_id="call-prices", ordinal=0,
    )
    assert exchange["call"]["thought_signature"] == "opaque-signature"
    assert exchange["result"] == scenario.exchanges[0]["result"]
    assert firestore_fake.documents == durable_before_replay


def test_absent_v2_fields_have_explicit_legacy_read_defaults_without_rewrite(firestore_fake):
    """Old context/operation documents remain legacy records and are not backfilled."""
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.models import TelegramContext
    from dataclasses import fields

    client = firestore_fake.client_class()
    scenario = make_contextual_scenario(unrelated_count=0)
    seeded = seed_history(client, context_id=scenario.context_id,
                          operation_id=scenario.operation_id, now=BASE_TIME)
    context_ref = client.collection("telegram_sessions").document(scenario.context_id)
    operation_ref = context_ref.collection("operations").document(scenario.operation_id)
    context_ref.update({"active_conversation_id": seeded["conversation_id"],
                        "last_doc_id": "doc-a"})
    legacy_route = {
        "status": "ok", "kind": "substantive",
        "tool_choice": {"name": "list_folders", "arguments": {}, "link_requested": False},
    }
    legacy_command_id = "88421"
    legacy_command_text = "A persisted legacy command"
    legacy_input_ref = (
        f"telegram_sessions/{scenario.context_id}/commands/{legacy_command_id}"
    )
    operation_ref.update({"route_control": legacy_route, "input_ref": legacy_input_ref})
    command_ref = context_ref.collection("commands").document(legacy_command_id)
    command_ref.set({
        "operation_id": scenario.operation_id,
        "context_id": scenario.context_id,
        "user_id": 17,
        "kind": "freeform",
        "candidate": legacy_command_text,
        "created_at": BASE_TIME,
    })
    conversation_ref = context_ref.collection("conversations").document(
        seeded["conversation_id"],
    )
    conversation_ref.update({"last_doc_id": "doc-a"})
    before_context = context_ref.get().to_dict()
    before_operation = operation_ref.get().to_dict()
    before_command = command_ref.get().to_dict()
    before_conversation = conversation_ref.get().to_dict()
    assert "routing_protocol" not in before_operation["route_control"]
    assert "pending_clarification_id" not in before_context
    assert "clarification_generation" not in before_context

    # These are expected defaults for future additive fields, not assertions that
    # a v2 reader exists today. Current legacy readers must not rewrite the record.
    repository = TelegramRepository(client=client)
    from google.cloud import firestore

    @firestore.transactional
    def read_legacy_command(transaction):
        return repository._command_for_operation(
            transaction, context_id=scenario.context_id,
            operation_id=scenario.operation_id, operation=before_operation,
        )

    loaded_command_ref, loaded_command = read_legacy_command(
        client.transaction(read_only=True),
    )
    assert loaded_command_ref is not None and loaded_command_ref.id == legacy_command_id
    assert loaded_command["context_id"] == scenario.context_id
    assert loaded_command["operation_id"] == scenario.operation_id
    assert loaded_command["candidate"] == legacy_command_text
    loaded_context = repository.get_context(scenario.context_id)
    assert loaded_context["last_doc_id"] == "doc-a"
    model_context = TelegramContext(**{
        field.name: loaded_context[field.name]
        for field in fields(TelegramContext) if field.name in loaded_context
    })
    assert model_context.status_message_id is None
    loaded_route = repository.get_route_control(scenario.context_id, scenario.operation_id)
    assert loaded_route == legacy_route
    assert loaded_route.get("routing_protocol", EXPECTED_ADDITIVE_LEGACY_DEFAULTS[
        "route_control"]["routing_protocol"]) is None
    loaded_conversation = repository.get_conversation(scenario.context_id, seeded["conversation_id"])
    assert loaded_conversation is not None
    if isinstance(loaded_conversation, dict) and "last_presented_list_id" in loaded_conversation:
        assert loaded_conversation["last_presented_list_id"] == (
            EXPECTED_ADDITIVE_LEGACY_DEFAULTS["conversation"]["last_presented_list_id"]
        )
    for field_name, expected in EXPECTED_ADDITIVE_LEGACY_DEFAULTS["context"].items():
        if hasattr(model_context, field_name):
            assert getattr(model_context, field_name) == expected
    assert set(EXPECTED_ADDITIVE_LEGACY_DEFAULTS["context"]).isdisjoint(before_context)
    assert set(EXPECTED_ADDITIVE_LEGACY_DEFAULTS["command"]).isdisjoint(before_command)
    assert set(EXPECTED_ADDITIVE_LEGACY_DEFAULTS["conversation"]).isdisjoint(before_conversation)
    assert context_ref.get().to_dict() == before_context
    assert operation_ref.get().to_dict() == before_operation
    assert command_ref.get().to_dict() == before_command
    assert conversation_ref.get().to_dict() == before_conversation


def test_telegram_registry_catalog_is_frozen_and_excludes_adapter_only_import_tool():
    from services.connectors.telegram.tools import get_tool_catalog

    catalog = get_tool_catalog()
    encoded = json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        "0044bf0706ac3146dd4b156c40203e5afb46866d825d8f96f71320c21cfb5122"
    )
    names = [tool["name"] for tool in catalog["tools"]]
    assert len(names) == 13
    assert "import_all_transcripts" not in names
    assert "present_search_answer" not in names
    assert "set_clarification" not in names


def test_mcp_tools_list_matches_complete_public_jsonrpc_baseline():
    from services.mcp.app import main

    baseline = Path(__file__).resolve().parents[3] / "fixtures" / "mcp_baseline.json"
    expected = json.loads(baseline.read_text(encoding="utf-8"))
    request_body = {"jsonrpc": "2.0", "id": "baseline-tools", "method": "tools/list"}

    async def request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as client:
            return await client.post("/mcp", json=request_body)

    import asyncio
    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == expected
    assert response.json()["id"] == "baseline-tools"
    assert len(response.json()["result"]["tools"]) == 14


def test_candidate_cleared_pinned_legacy_write_route_replays_first_effect_without_model_selection(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A replay after candidate cleanup runs its pinned first effect with stable IDs."""
    import asyncio

    from googleapiclient.errors import HttpError
    import httplib2

    from core.integrations.firestore.client import FOLDERS_COLLECTION
    from core.integrations.firestore.operation_queue import claim_head
    from core.operations.effects import get_effect_allocation
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime

    now = BASE_TIME + timedelta(minutes=1)
    root_id, child_folder_id = "t003-root", "t003-created-folder"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:synthetic-token")
    _set_synthetic_drive_oauth(monkeypatch)
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        93001, chat_id=93001, user_id=17,
        command=parse_command("Create the folder Archive"), now=now,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="t003-runtime-worker", now=now)
    operation_ref = context_ref.collection("operations").document(accepted.operation_id)
    operation = operation_ref.get().to_dict()
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    global_conversation_id = "telegram-global-" + hashlib.sha256(
        f"{accepted.context_id}\0{accepted.operation_id}".encode("utf-8")
    ).hexdigest()
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    ledger.save_request(
        global_conversation_id, accepted.operation_id,
        "Create the folder Archive",
    )
    assert ledger.load_request(global_conversation_id, accepted.operation_id) == (
        "Create the folder Archive"
    )
    assert ledger.load_turn(global_conversation_id, accepted.operation_id) is None
    target_route = {
        "status": "ok", "kind": "substantive",
        "tool_choice": {"name": "create_folder", "arguments": {"name": "Archive"},
                        "link_requested": False},
    }
    assert repository.persist_route(
        accepted.operation_id, target_route, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    command_ref = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("commands").document(operation["input_ref"].rsplit("/", 1)[-1])
    assert command_ref.get().to_dict()["candidate"] == "Create the folder Archive"
    assert repository.clear_command_candidate(
        accepted.context_id, accepted.operation_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    assert "candidate" not in command_ref.get().to_dict()
    assert repository.get_route_control(accepted.context_id, accepted.operation_id) == target_route
    assert ledger.load_request(global_conversation_id, accepted.operation_id) == (
        "Create the folder Archive"
    )
    client.collection(FOLDERS_COLLECTION).document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    effect_id = "telegram-effect-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    intent_payload = json.dumps({"name": "Archive", "parent_id": root_id},
                                sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    intent_digest = "sha256:" + hashlib.sha256(intent_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", intent_digest)).encode("utf-8")
    ).hexdigest()

    # Drive is absent on preflight, then accepts the stable ID already allocated
    # in Firestore. The real folder operation performs its own durable read-back.
    drive_404 = HttpError(httplib2.Response({"status": "404"}), b"synthetic missing")
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [child_folder_id]})
    sdk_fakes.drive.queue("files.get", error=drive_404)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.create", response={"id": child_folder_id})
    sdk_fakes.drive.queue("files.get", response={
        "id": child_folder_id, "name": "Archive",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [root_id], "trashed": False,
        "appProperties": {"tlbrain_effect_intent": marker},
    })
    sent = []

    def telegram_response(request):
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 93101}})

    async_http = httpx.AsyncClient(transport=httpx.MockTransport(telegram_response))
    telegram = TelegramClient("123456:synthetic-token", http_client=async_http)
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=TelegramSaveRuntime(repository=repository),
        client=telegram, clock=lambda: now, root_folder_id=root_id,
    )

    async def invoke_runtime():
        try:
            return await runtime(
                accepted.context_id, operation, lease,
                command=None,
            )
        finally:
            await async_http.aclose()

    from core.operations.queue import QueueOperationState

    result = asyncio.run(invoke_runtime())
    allocation = get_effect_allocation(client, child_id, effect_id)
    assert result == QueueOperationState.COMPLETED, sdk_fakes.trace
    assert allocation["resource_id"] == child_folder_id
    assert allocation["operation_id"] == child_id
    assert allocation["effect_id"] == effect_id
    assert operation_ref.get().to_dict()["route_control"] == target_route
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "gemini"
                and event["operation"] == "generate_content"]
    drive_events = [row for row in sdk_fakes.trace if row["boundary"] == "drive"]
    create_events = [row for row in drive_events if row["operation"] == "files.create"]
    assert len(create_events) == 1
    create_event = create_events[0]
    assert create_event["arguments"]["body"]["id"] == child_folder_id
    assert create_event["arguments"]["body"]["appProperties"][
        "tlbrain_effect_intent"
    ] == marker
    assert sent and sent[-1][1]["text"] == json.dumps(
        child_folder_id, ensure_ascii=False, default=str,
    )


def test_legacy_switch_notice_replay_keeps_exact_body_revision_and_message_id(
    firestore_fake,
):
    """The actual publisher and MockTransport preserve the durable legacy notice."""
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository._common import NOTIFICATIONS

    client = firestore_fake.client_class()
    scenario = make_contextual_scenario(unrelated_count=0)
    context_ref = client.collection("telegram_sessions").document(scenario.context_id)
    context_ref.set({"chat_id": scenario.chat.chat_id,
                     "thread_id": scenario.topic.thread_id,
                     "fence": 1,
                     "created_at": BASE_TIME,
                     "updated_at": BASE_TIME}, merge=True)
    calls = []

    def response(request):
        calls.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content)))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 915}})

    async_http = httpx.AsyncClient(transport=httpx.MockTransport(response))
    telegram = TelegramClient("123456:synthetic-token", http_client=async_http)
    repository = TelegramRepository(client=client)
    publisher = TelegramNotificationPublisher(repository, telegram, clock=lambda: BASE_TIME)

    async def run():
        first = await publisher.send_switch_notice(
            context_id=scenario.context_id, operation_id="legacy-switch-operation",
            notification_id="legacy-switch-notice", target_doc_id="doc-b", dialog_number=7,
        )
        before_replay = client.collection(NOTIFICATIONS).document(
            "legacy-switch-notice",
        ).get().to_dict()
        replay_repository = TelegramRepository(client=client)
        replay_publisher = TelegramNotificationPublisher(
            replay_repository, telegram, clock=lambda: BASE_TIME,
        )
        replay = await replay_publisher.send_switch_notice(
            context_id=scenario.context_id, operation_id="legacy-switch-operation",
            notification_id="legacy-switch-notice", target_doc_id="doc-b", dialog_number=7,
        )
        after_replay = client.collection(NOTIFICATIONS).document(
            "legacy-switch-notice",
        ).get().to_dict()
        await async_http.aclose()
        return first, replay, before_replay, after_replay

    import asyncio
    first, replay, before_replay, after_replay = asyncio.run(run())
    row = repository.get_notification("legacy-switch-notice")
    assert first == replay == "confirmed"
    assert row["body"] == "Starting to process dialog #7."
    assert row["state"] == "delivered"
    assert row["message_id"] == 915
    assert row["revision"] == 1
    assert after_replay == before_replay
    assert calls == [("sendMessage", {"chat_id": scenario.chat.chat_id,
                                      "text": "Starting to process dialog #7.",
                                      "message_thread_id": scenario.topic.thread_id})]


def test_legacy_route_metadata_accepts_save_retry_and_annotation_records(firestore_fake):
    """Persist historic route kinds through the real accepted-command/repository path."""
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository

    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    cases = [
        (77001, 77001, "save dialog 1", {
            "status": "ok", "kind": "save", "target_draft_id": "draft-legacy-save",
        }, {"status": "ok", "kind": "save", "target_draft_id": "draft-other-save"}),
        (77002, 77002, "retry dialog 2", {
            "status": "ok", "kind": "retry", "dialog_number": 2,
            "retry_request_operation_id": "retry-request-legacy",
        }, {"status": "ok", "kind": "retry", "dialog_number": 3,
            "retry_request_operation_id": "retry-request-legacy"}),
        (77003, 77003, "note this", {
            "status": "ok", "kind": "annotation", "target_doc_id": "doc-b",
            "annotation_text": "Preserve this exact legacy annotation €42.",
        }, {"status": "ok", "kind": "annotation", "target_doc_id": "doc-c",
            "annotation_text": "Preserve this exact legacy annotation €42."}),
    ]
    for update_id, chat_id, text, route, changed_route in cases:
        accepted = repository.accept_command(
            update_id, chat_id=chat_id, user_id=17,
            command=parse_command(text), now=BASE_TIME,
        )
        context_ref = client.collection("telegram_sessions").document(accepted.context_id)
        lease = claim_head(client, context_ref, owner=f"legacy-worker-{update_id}",
                           now=BASE_TIME + timedelta(seconds=update_id - 77000))
        operation = context_ref.collection("operations").document(accepted.operation_id)
        row = operation.get().to_dict()
        assert row["input_ref"].startswith(
            f"telegram_sessions/{accepted.context_id}/commands/"
        )
        assert repository.persist_route(
            accepted.operation_id, route, context_id=accepted.context_id,
            owner=lease.owner, fence=lease.fence,
            now=BASE_TIME + timedelta(seconds=update_id - 77000),
        )
        replay_repository = TelegramRepository(client=client)
        pinned = replay_repository.get_route_control(accepted.context_id, accepted.operation_id)
        assert pinned == route
        assert "routing_protocol" not in pinned
        assert "routing_protocol" not in operation.get().to_dict().get("route_control", {})
        before_replay_state = operation.get().to_dict()
        assert replay_repository.get_route_control(
            accepted.context_id, accepted.operation_id,
        ) == pinned
        assert operation.get().to_dict() == before_replay_state
        assert not repository.persist_route(
            accepted.operation_id, changed_route,
            context_id=accepted.context_id, owner=lease.owner, fence=lease.fence,
            now=BASE_TIME + timedelta(seconds=update_id - 77000),
        )
        assert replay_repository.get_route_control(accepted.context_id, accepted.operation_id) == pinned
        assert operation.get().to_dict() == before_replay_state


def _executor_for_folder_write(client, *, now, chat_id, monkeypatch):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor

    _set_synthetic_drive_oauth(monkeypatch)
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        chat_id, chat_id=chat_id, user_id=17,
        command=parse_command("Create the folder Archive"), now=now,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner=f"executor-worker-{chat_id}", now=now)
    operation = context_ref.collection("operations").document(accepted.operation_id).get().to_dict()
    execution_context = {
        "context_id": accepted.context_id,
        "operation_id": accepted.operation_id,
        "owner": lease.owner,
        "fence": lease.fence,
        "lease": lease,
        "clock": lambda: now,
    }
    executor = TelegramRegistryExecutor(
        client=client, context=execution_context, save_handler=object(),
        execution_context=execution_context,
    )
    return accepted, operation, lease, repository, executor


def _http_404():
    from googleapiclient.errors import HttpError
    import httplib2

    return HttpError(httplib2.Response({"status": "404"}), b"synthetic missing")


def _set_synthetic_drive_oauth(monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")


def _folder_marker(parent_operation_id: str, child_id: str, root_id: str) -> tuple[str, str]:
    effect_id = "telegram-effect-" + hashlib.sha256(
        f"{parent_operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    intent_payload = json.dumps({"name": "Archive", "parent_id": root_id},
                                sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    intent_digest = "sha256:" + hashlib.sha256(intent_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", intent_digest)).encode("utf-8")
    ).hexdigest()
    return effect_id, marker


def test_confirmed_shared_folder_result_replays_from_checkpoint_without_provider_calls(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """The real executor returns its durable confirmed result before SDK work on replay."""
    from core.integrations.firestore.client import FOLDERS_COLLECTION
    from core.operations.effects import get_effect_allocation

    now = BASE_TIME + timedelta(minutes=2)
    root_id, folder_id = "t003-confirmed-root", "t003-confirmed-folder"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection(FOLDERS_COLLECTION).document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, _lease, _repository, executor = _executor_for_folder_write(
        client, now=now, chat_id=93201, monkeypatch=monkeypatch,
    )
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    effect_id, marker = _folder_marker(accepted.operation_id, child_id, root_id)
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [folder_id]})
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.create", response={"id": folder_id})
    sdk_fakes.drive.queue("files.get", response={
        "id": folder_id, "name": "Archive",
        "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
        "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
    })

    first = executor.execute(name="create_folder", arguments={"name": "Archive"}, call_ordinal=0)
    allocation = get_effect_allocation(client, child_id, effect_id)
    assert allocation["operation_id"] == child_id
    assert allocation["effect_id"] == effect_id
    from core.integrations.firestore.mutations import get_mutation_effect
    from core.integrations.firestore.operations import get_operation
    from core.operations.models import EffectState, OperationState
    confirmed_effect = get_mutation_effect(client, child_id, effect_id)
    completed_child = get_operation(client, child_id)
    assert confirmed_effect is not None and confirmed_effect.state == EffectState.CONFIRMED
    assert completed_child is not None and completed_child.state == OperationState.COMPLETED
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor
    replay_executor = TelegramRegistryExecutor(
        client=client, context=executor.execution_context,
        save_handler=object(), execution_context=executor.execution_context,
    )
    before_replay = sdk_fakes.trace
    replay = replay_executor.execute(name="create_folder", arguments={"name": "Archive"}, call_ordinal=0)

    assert first == replay == folder_id
    assert allocation["resource_id"] == folder_id
    assert sdk_fakes.trace == before_replay
    assert len([event for event in sdk_fakes.trace if event["operation"] == "files.create"]) == 1


def test_unknown_shared_folder_effect_retries_only_readback_and_keeps_identity(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A lost Drive response leaves the real shared effect guarded and read-back-only."""
    from core.integrations.firestore.client import FOLDERS_COLLECTION
    from core.operations.effects import get_effect_allocation

    now = BASE_TIME + timedelta(minutes=3)
    root_id = "t003-unknown-root"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection(FOLDERS_COLLECTION).document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, _lease, _repository, executor = _executor_for_folder_write(
        client, now=now, chat_id=93301, monkeypatch=monkeypatch,
    )
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    effect_id, _marker = _folder_marker(accepted.operation_id, child_id, root_id)
    sdk_fakes.drive.queue("files.generateIds", response={"ids": ["t003-unknown-folder"]})
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.create", error=TimeoutError("synthetic lost response"))
    sdk_fakes.drive.queue("files.get", error=_http_404())
    from core.integrations.firestore.mutations import get_mutation_effect, get_resource_guard
    from core.integrations.firestore.operations import get_operation
    from core.operations.models import EffectState
    from core.operations.mutations import MutationOutcomeUnknown
    with pytest.raises(MutationOutcomeUnknown):
        executor.execute(name="create_folder", arguments={"name": "Archive"}, call_ordinal=0)

    allocation = get_effect_allocation(client, child_id, effect_id)
    assert allocation["resource_id"] == "t003-unknown-folder"
    child = get_operation(client, child_id)
    effect = get_mutation_effect(client, child_id, effect_id)
    guard = get_resource_guard(client, f"tree:{root_id}")
    assert child is not None
    assert effect is not None and effect.state == EffectState.UNKNOWN
    assert effect.effect_id == effect_id and effect.operation_id == child_id
    assert guard.mutation_state == EffectState.UNKNOWN.value
    assert effect_id in guard.unsettled_effect_ids
    initial_generates = [event for event in sdk_fakes.trace
                         if event["boundary"] == "drive"
                         and event["operation"] == "files.generateIds"]
    before_reconcile = sdk_fakes.trace
    # The same validated executor retry reaches shared mutation reconciliation;
    # that layer must read the existing effect and must not invoke the callback.
    sdk_fakes.drive.queue("files.get", error=_http_404())
    with pytest.raises(MutationOutcomeUnknown):
        executor.execute(name="create_folder", arguments={"name": "Archive"}, call_ordinal=0)
    assert get_effect_allocation(client, child_id, effect_id) == allocation
    final_trace = sdk_fakes.trace
    assert len([event for event in final_trace
                if event["boundary"] == "drive" and event["operation"] == "files.create"]) == 1
    assert len([event for event in final_trace
                if event["boundary"] == "drive" and event["operation"] == "files.generateIds"]
               ) == len(initial_generates) == 1
    new_gets = [event for event in final_trace[len(before_reconcile):]
                if event["boundary"] == "drive" and event["operation"] == "files.get"]
    assert new_gets
