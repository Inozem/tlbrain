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
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [child_folder_id]})
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=drive_404)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
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


def _context_v2_worker(client, *, chat_id: int, now=None):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    now = now or BASE_TIME + timedelta(seconds=1)
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        chat_id, chat_id=chat_id, user_id=17,
        command=parse_command("What did we decide about the price?"), now=BASE_TIME,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner=f"context-v2-worker-{chat_id}", now=now)
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    return repository, accepted, context_ref, lease, ledger, now


def test_context_and_intent_checkpoints_are_immutable_and_integrity_checked(firestore_fake):
    """Retained codec records freeze source input and clarification intent."""
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    repository, accepted, _context_ref, lease, ledger, now = _context_v2_worker(
        client, chat_id=88101,
    )
    conversation_id = "source-conversation-v2"
    context_value = {"history": "История🙂" * 900, "refs": [{"doc_id": "doc-b"}]}
    context_ref_data = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation_id, value=context_value,
    )
    context_doc = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).collection(
        "routing_checkpoints",
    ).document("context")
    assert "_input_ref" in context_doc.get().to_dict()["payload"]
    intent_value = {"slots": {"doc_id": "doc-b"}, "history_ref": "exchange-1"}
    intent_ref_data = ledger.save_routing_checkpoint(
        "intent", actor_id=17, conversation_id=conversation_id, value=intent_value,
    )
    durable = deepcopy(firestore_fake.documents)
    assert ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation_id, value=context_value,
    ) == context_ref_data
    assert firestore_fake.documents == durable
    restarted = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    assert restarted.save_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation_id, value=context_value,
    ) == context_ref_data
    assert firestore_fake.documents == durable
    assert restarted.load_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation_id,
        reference=context_ref_data,
    ) == {"reference": context_ref_data, "value": context_value}
    assert restarted.load_routing_checkpoint(
        "intent", actor_id=17, conversation_id=conversation_id,
    )["value"] == intent_value
    with pytest.raises(ValueError, match="immutable"):
        ledger.save_routing_checkpoint(
            "context", actor_id=17, conversation_id=conversation_id,
            value={"history": "changed"},
        )
    intent_doc = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).collection(
        "routing_checkpoints",
    ).document("intent")
    damaged = intent_doc.get().to_dict()
    damaged["payload"]["_inline"]["value"]["slots"]["doc_id"] = "tampered"
    intent_doc.set(damaged)
    with pytest.raises(ValueError, match="does not verify"):
        restarted.load_routing_checkpoint(
            "intent", actor_id=17, conversation_id=conversation_id,
        )


def test_terminal_context_and_intent_reads_survive_candidate_cleanup_and_expired_lease(
    firestore_fake,
):
    from core.integrations.firestore.operation_queue import complete_head
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    repository, accepted, context_ref, lease, ledger, now = _context_v2_worker(
        client, chat_id=88210,
    )
    conversation = "terminal-source"
    context_value = {"history": "x" * 12_000, "ordered": ["one", "two"]}
    intent_value = {"action": "move_transcript", "slots": {"doc_id": "doc-b"}}
    context_reference = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation, value=context_value,
    )
    intent_reference = ledger.save_routing_checkpoint(
        "intent", actor_id=17, conversation_id=conversation, value=intent_value,
    )

    ordinary_route = {"status": "ok", "kind": "substantive"}
    assert repository.persist_route(
        accepted.operation_id, ordinary_route, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    assert repository.clear_command_candidate(
        accepted.context_id, accepted.operation_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    assert complete_head(
        client, context_ref, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    context_ref.update({"lock_expires_at": now - timedelta(seconds=1)})

    reader = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence,
        clock=lambda: now + timedelta(days=1),
    )
    before_reads = deepcopy(firestore_fake.documents)
    assert reader.load_routing_checkpoint(
        "context", actor_id=17, conversation_id=conversation,
        reference=context_reference,
    )["value"] == context_value
    assert reader.load_routing_checkpoint(
        "intent", actor_id=17, conversation_id=conversation,
        reference=intent_reference,
    )["value"] == intent_value
    assert firestore_fake.documents == before_reads


def test_unsupported_metadata_route_markers_fail_closed_without_writes(firestore_fake):
    from services.connectors.telegram.repository import TelegramRepository

    client = firestore_fake.client_class()
    repository, accepted, _context_ref, lease, _ledger, now = _context_v2_worker(
        client, chat_id=88301,
    )
    operation_ref = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id)

    before = deepcopy(firestore_fake.documents)
    for marker in (
        {"routing_protocol": "context-v2", "schema_version": 1},
        {"context_ref": "synthetic-ref", "context_digest": "a" * 64},
        {"decision_ref": "synthetic-ref", "decision_digest": "a" * 64},
    ):
        with pytest.raises(ValueError):
            repository._route_control_data({"status": "error", **marker})
    assert firestore_fake.documents == before

    # An explicit null protocol remains the ordinary legacy default.
    assert repository._route_control_data({
        "status": "error", "routing_protocol": None,
    }) == {"status": "error"}

    unsupported = {
        "routing_protocol": "context-v2", "schema_version": 1,
        "status": "error", "actor_id": 17,
        "context_conversation_id": "synthetic-conversation",
        "result_category": "model_error",
    }
    with pytest.raises(ValueError):
        repository.persist_route(
            accepted.operation_id, unsupported, context_id=accepted.context_id,
            owner=lease.owner, fence=lease.fence, now=now,
        )
    assert firestore_fake.documents == before

    # An already stored, otherwise valid wrapper must not be returned as a pin
    # or overwritten by an ordinary route.
    operation_ref.update({
        "route_control": unsupported,
    })
    with pytest.raises(ValueError):
        repository.get_route_control(accepted.context_id, accepted.operation_id)

    # Nor may an unsupported write overwrite that pin or trigger downstream work.
    pinned = operation_ref.get().to_dict()["route_control"]
    with pytest.raises(ValueError):
        repository.persist_route(
            accepted.operation_id,
            {"status": "error"},
            context_id=accepted.context_id,
            owner=lease.owner,
            fence=lease.fence,
            now=now,
        )
    assert operation_ref.get().to_dict()["route_control"] == pinned


def test_context_and_intent_checkpoints_keep_json_copy_isolated(firestore_fake):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88201,
    )
    conversation = "source-v2"
    with pytest.raises(ValueError, match="kind"):
        ledger.save_routing_checkpoint(
            "decision", actor_id=17, conversation_id=conversation, value={"obsolete": True},
        )
    with pytest.raises(ValueError, match="kind"):
        ledger.load_routing_checkpoint(
            "decision", actor_id=17, conversation_id=conversation,
        )
    inputs = {
        "context": {"history": "original", "refs": ["r1"]},
        "intent": {"slots": {"doc_id": "doc-a"}},
    }
    references = {
        kind: ledger.save_routing_checkpoint(
            kind, actor_id=17, conversation_id=conversation, value=value,
        ) for kind, value in inputs.items()
    }
    inputs["context"]["history"] = "changed after save"
    for kind in inputs:
        loaded = ledger.load_routing_checkpoint(
            kind, actor_id=17, conversation_id=conversation,
        )
        assert loaded["reference"] == references[kind]
        if kind == "context":
            assert loaded["value"] == {"history": "original", "refs": ["r1"]}
        else:
            assert loaded["value"] == inputs[kind]
    with pytest.raises(ValueError, match="scope"):
        ledger.save_routing_checkpoint(
            "intent", actor_id=17, conversation_id="other-source", value={"x": 1},
        )
    with pytest.raises(PermissionError, match="actor"):
        ledger.save_routing_checkpoint(
            "intent", actor_id=18, conversation_id=conversation, value={"x": 1},
        )
    missing = {
        **references["intent"], "ref": f"telegram_sessions/{accepted.context_id}/operations/{accepted.operation_id}/routing_checkpoints/intent",
    }
    client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("intent").delete()
    with pytest.raises(ValueError, match="absent"):
        ledger.load_routing_checkpoint(
            "intent", actor_id=17, conversation_id=conversation, reference=missing,
        )


def test_context_intent_reject_cross_source_conversation(firestore_fake):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88218,
    )
    context_ref = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-A", value={"history": "A"},
    )
    with pytest.raises(ValueError, match="scope changed"):
        ledger.save_routing_checkpoint(
            "intent", actor_id=17, conversation_id="source-B", value={"intent": "B"},
        )
    intent_doc = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("intent")
    assert not intent_doc.get().exists
    assert ledger.load_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-A", reference=context_ref,
    )["value"] == {"history": "A"}


def test_context_intent_first_publication_race_has_one_source_scope(firestore_fake):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from tests.fixtures.firestore_fake import FakeFirestoreClient
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    part_barrier = Barrier(2)

    class BarrierTransaction:
        def __init__(self, transaction):
            self._transaction = transaction
            self._waited_for_part = False

        def __getattr__(self, name):
            return getattr(self._transaction, name)

        def _commit(self):
            if not self._waited_for_part and any(
                "/input_parts/part-" in reference.path
                for _operation, reference, _data, _merge in self._transaction._writes
            ):
                self._waited_for_part = True
                part_barrier.wait(timeout=10)
            return self._transaction._commit()

    class BarrierFirestoreClient(FakeFirestoreClient):
        def transaction(self, max_attempts=5, read_only=False):
            return BarrierTransaction(super().transaction(
                max_attempts=max_attempts, read_only=read_only,
            ))

    client = BarrierFirestoreClient()
    repository, accepted, _context, lease, _ledger, now = _context_v2_worker(
        client, chat_id=88219,
    )
    ledgers = {
        kind: FirestoreAgentLedger(
            client, context_id=accepted.context_id, operation_id=accepted.operation_id,
            owner=lease.owner, fence=lease.fence, clock=lambda: now,
        ) for kind in ("context", "intent")
    }

    def publish(kind, conversation):
        try:
            reference = ledgers[kind].save_routing_checkpoint(
                kind, actor_id=17, conversation_id=conversation,
                value={"history": "x" * 12_000, "kind": kind},
            )
            return "published", reference
        except ValueError as exc:
            return "rejected", str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        context_future = pool.submit(publish, "context", "source-A")
        intent_future = pool.submit(publish, "intent", "source-B")
        outcomes = [context_future.result(timeout=15), intent_future.result(timeout=15)]
    assert sorted(outcome[0] for outcome in outcomes) == ["published", "rejected"]
    rejected = next(outcome[1] for outcome in outcomes if outcome[0] == "rejected")
    assert "scope changed" in rejected
    canonical = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints")
    assert sum(canonical.document(kind).get().exists for kind in ("context", "intent")) == 1
    assert repository.get_route_control(accepted.context_id, accepted.operation_id) is None


def test_context_v2_exact_inline_threshold_and_multiple_input_parts(firestore_fake):
    from core.integrations.firestore.input_parts import MAX_PART_BYTES
    from services.connectors.telegram.repository._common import _routing_json_bytes

    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88202,
    )
    for kind, target_length in (("context", 4096), ("intent", 4097)):
        base_envelope = {
            "routing_protocol": "context-v2", "schema_version": 1, "kind": kind,
            "context_id": accepted.context_id, "operation_id": accepted.operation_id,
            "actor_id": 17, "conversation_id": "threshold-source", "value": {"history": ""},
        }
        base_size = len(_routing_json_bytes(base_envelope))
        value = {"history": "x" * (target_length - base_size)}
        assert len(_routing_json_bytes({**base_envelope, "value": value})) == target_length
        reference = ledger.save_routing_checkpoint(
            kind, actor_id=17, conversation_id="threshold-source", value=value,
        )
        checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
            "operations",
        ).document(accepted.operation_id).collection("routing_checkpoints").document(kind).get().to_dict()
        assert ("_inline" in checkpoint["payload"]) is (target_length == 4096)
        assert ledger.load_routing_checkpoint(
            kind, actor_id=17, conversation_id="threshold-source", reference=reference,
        )["value"] == value
    _repository, large_accepted, _context, _lease, large_ledger, _now = _context_v2_worker(
        client, chat_id=88222,
    )
    large_value = {"history": "漢🙂" * 190_000, "references": ["ref-1", "ref-2"]}
    large_ref = large_ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="large-source", value=large_value,
    )
    row = client.collection("telegram_sessions").document(large_accepted.context_id).collection(
        "operations",
    ).document(large_accepted.operation_id).collection("routing_checkpoints").document("context").get().to_dict()
    payload_ref = row["payload"]["_input_ref"]
    assert row["payload"]["byte_length"] > MAX_PART_BYTES * 2
    manifest = client.collection("dialog_operations").document(
        payload_ref.split("/")[1],
    ).collection("input_parts").document("manifest").get().to_dict()
    assert manifest["part_count"] >= 3
    assert large_ledger.load_routing_checkpoint(
        "context", actor_id=17, conversation_id="large-source", reference=large_ref,
    )["value"] == large_value


@pytest.mark.parametrize("value", [b"binary", float("nan"), ("tuple", "is not JSON")])
def test_context_v2_rejects_non_json_values_before_writing(firestore_fake, value):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88217,
    )
    with pytest.raises(ValueError, match="JSON-safe"):
        ledger.save_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", value={"value": value},
        )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    assert not checkpoint.get().exists


@pytest.mark.parametrize("field,value", [
    ("routing_protocol", "future-v9"), ("schema_version", True),
    ("kind", "intent"), ("context_id", "other-context"),
    ("operation_id", "other-operation"), ("actor_id", 18),
    ("conversation_id", "other-source"), ("ref", "telegram_sessions/guessed"),
    ("digest", "A" * 64), ("byte_length", True),
])
def test_context_v2_supplied_checkpoint_references_fail_closed(firestore_fake, field, value):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88203,
    )
    reference = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-v2", value={"history": "x"},
    )
    broken = dict(reference)
    broken[field] = value
    with pytest.raises(ValueError):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", reference=broken,
        )
    with pytest.raises(ValueError, match="shape"):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2",
            reference={**reference, "injected": "not allowed"},
        )
    with pytest.raises(ValueError):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", reference="bare-reference",
        )
    assert ledger.load_routing_checkpoint(
        "intent", actor_id=17, conversation_id="source-v2",
    ) is None


@pytest.mark.parametrize("header,corrupt_value", [("schema_version", True), ("actor_id", 17.0)])
def test_context_v2_stored_headers_reject_bool_float_aliases(
    firestore_fake, header, corrupt_value,
):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88204,
    )
    reference = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-v2", value={"x": "small"},
    )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    damaged = checkpoint.get().to_dict()
    damaged[header] = corrupt_value
    checkpoint.set(damaged)
    with pytest.raises(ValueError, match="reference binding"):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", reference=reference,
        )


def test_context_v2_envelope_identity_types_are_strict_after_rehash(firestore_fake):
    from services.connectors.telegram.repository._common import _routing_json_bytes

    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88205,
    )
    ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-v2", value={"x": "small"},
    )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    row = checkpoint.get().to_dict()
    envelope = row["payload"]["_inline"]
    envelope["schema_version"] = True
    encoded = _routing_json_bytes(envelope)
    digest = hashlib.sha256(encoded).hexdigest()
    row.update({"digest": digest, "byte_length": len(encoded)})
    row["payload"].update({"sha256": digest, "byte_length": len(encoded), "_inline": envelope})
    checkpoint.set(row)
    with pytest.raises(ValueError, match="envelope binding"):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2",
        )


@pytest.mark.parametrize("damage", [
    "missing_part", "reordered_part", "corrupt_part", "not_ready", "wrong_input_ref",
    "wrong_max_part_bytes",
])
def test_context_v2_multipart_tampering_is_rejected(firestore_fake, damage):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88206,
    )
    reference = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-v2", value={"history": "z" * 12_000},
    )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    row = checkpoint.get().to_dict()
    payload_ref = row["payload"]["_input_ref"]
    data_op = payload_ref.split("/")[1]
    parts = client.collection("dialog_operations").document(data_op).collection("input_parts")
    manifest = parts.document("manifest")
    if damage == "missing_part":
        parts.document("part-000000").delete()
    elif damage == "reordered_part":
        part = parts.document("part-000000").get().to_dict()
        part["index"] = 1
        parts.document("part-000000").set(part)
    elif damage == "corrupt_part":
        part = parts.document("part-000000").get().to_dict()
        part["payload"] = part["payload"][:-1] + b"X"
        parts.document("part-000000").set(part)
    elif damage == "not_ready":
        manifest.update({"ready": False})
    elif damage == "wrong_input_ref":
        row["payload"]["_input_ref"] = "dialog_operations/guessed/input_parts/manifest"
        checkpoint.set(row)
    else:
        row["payload"]["max_part_bytes"] = 123
        checkpoint.set(row)
    with pytest.raises(ValueError):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", reference=reference,
        )


@pytest.mark.parametrize("damage", ["content", "digest", "length", "mode"])
def test_context_v2_inline_tampering_is_rejected(firestore_fake, damage):
    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88207,
    )
    reference = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="source-v2", value={"x": "small"},
    )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    row = checkpoint.get().to_dict()
    if damage == "content":
        row["payload"]["_inline"]["value"]["x"] = "tampered"
    elif damage == "digest":
        row["payload"]["sha256"] = "0" * 64
    elif damage == "length":
        row["payload"]["byte_length"] += 1
    else:
        row["payload"]["_input_ref"] = "dialog_operations/guessed/input_parts/manifest"
    checkpoint.set(row)
    with pytest.raises(ValueError):
        ledger.load_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", reference=reference,
        )


def test_ordinary_successful_route_pin_remains_immutable(firestore_fake):
    client = firestore_fake.client_class()
    repository, accepted, _context, lease, _ledger, now = _context_v2_worker(
        client, chat_id=88208,
    )
    failed = {"status": "error", "kind": "substantive"}
    assert repository.persist_route(
        accepted.operation_id, failed, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    pinned = {"status": "ok", "kind": "substantive", "target_doc_id": "doc-a"}
    assert repository.persist_route(
        accepted.operation_id, pinned, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    before_replay = deepcopy(firestore_fake.documents)
    assert repository.persist_route(
        accepted.operation_id, pinned, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    assert firestore_fake.documents == before_replay
    changed = {**pinned, "target_doc_id": "doc-b"}
    assert not repository.persist_route(
        accepted.operation_id, changed, context_id=accepted.context_id,
        owner=lease.owner, fence=lease.fence, now=now,
    )
    assert repository.get_route_control(accepted.context_id, accepted.operation_id) == pinned

@pytest.mark.parametrize("invalid_identity", ["owner", "fence", "expired"])
def test_context_v2_save_requires_current_live_lease(firestore_fake, invalid_identity):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    _repository, accepted, _context, lease, ledger, now = _context_v2_worker(
        client, chat_id=88212,
    )
    if invalid_identity == "owner":
        ledger = FirestoreAgentLedger(
            client, context_id=accepted.context_id, operation_id=accepted.operation_id,
            owner="foreign-owner", fence=lease.fence, clock=lambda: now,
        )
    elif invalid_identity == "fence":
        ledger = FirestoreAgentLedger(
            client, context_id=accepted.context_id, operation_id=accepted.operation_id,
            owner=lease.owner, fence=lease.fence + 1, clock=lambda: now,
        )
    else:
        ledger.clock = lambda: now + timedelta(days=1)
    with pytest.raises(PermissionError, match="lease is stale"):
        ledger.save_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", value={"x": "y"},
        )
    path = f"telegram_sessions/{accepted.context_id}/operations/{accepted.operation_id}/routing_checkpoints/context"
    assert not client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).collection(
        "routing_checkpoints",
    ).document("context").get().exists


@pytest.mark.parametrize("failure_point", ["part", "manifest"])
def test_context_v2_part_write_failure_never_publishes_checkpoint(firestore_fake, failure_point):
    from services.connectors.telegram.repository._common import _routing_json_bytes

    client = firestore_fake.client_class()
    _repository, accepted, _context, _lease, ledger, _now = _context_v2_worker(
        client, chat_id=88213,
    )
    value = {"history": "a" * 12_000}
    envelope = {
        "routing_protocol": "context-v2", "schema_version": 1, "kind": "context",
        "context_id": accepted.context_id, "operation_id": accepted.operation_id,
        "actor_id": 17, "conversation_id": "source-v2", "value": value,
    }
    digest = hashlib.sha256(_routing_json_bytes(envelope)).hexdigest()
    data_operation = "telegram-ledger-" + hashlib.sha256(
        f"{accepted.context_id}\0{accepted.operation_id}\0routing-context\0{digest}".encode()
    ).hexdigest()
    input_parts = f"dialog_operations/{data_operation}/input_parts"
    fail_path = f"{input_parts}/part-000000" if failure_point == "part" else f"{input_parts}/manifest"
    client.fail_next("commit", path=fail_path, exception=RuntimeError("synthetic write fault"))
    with pytest.raises(RuntimeError, match="synthetic write fault"):
        ledger.save_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2", value=value,
        )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    assert not checkpoint.get().exists
    written_parts = [path for path in firestore_fake.documents
                     if path.startswith("dialog_operations/") and "/input_parts/part-" in path]
    assert bool(written_parts) is (failure_point == "manifest")
    assert not any(path.endswith("/input_parts/manifest")
                   for path in firestore_fake.documents)
    operation = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).get().to_dict()
    assert "route_control" not in operation


def test_context_v2_lease_loss_during_multipart_preparation_does_not_publish(firestore_fake):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    _repository, accepted, _context, lease, _ledger, now = _context_v2_worker(
        client, chat_id=88214,
    )
    calls = 0
    def lease_clock():
        nonlocal calls
        calls += 1
        return now if calls < 6 else now + timedelta(days=1)
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lease_clock,
    )
    with pytest.raises(PermissionError, match="lease is stale"):
        ledger.save_routing_checkpoint(
            "context", actor_id=17, conversation_id="source-v2",
            value={"history": "a" * 1_100_000},
        )
    checkpoint = client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations",
    ).document(accepted.operation_id).collection("routing_checkpoints").document("context")
    assert not checkpoint.get().exists
    written_parts = [path for path in firestore_fake.documents
                     if path.startswith("dialog_operations/") and "/input_parts/part-" in path]
    assert written_parts
    assert not any(path.endswith("/input_parts/manifest")
                   for path in firestore_fake.documents)


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
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
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
    from core.operations.write import _new_resolved_write_targets
    from services.connectors.telegram.runtime.tools import stable_effect_id
    legacy_capability = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id,
        effect_id=stable_effect_id(accepted.operation_id, 0),
        arguments={"name": "Archive"}, parent_id=root_id, root_id=root_id,
        intent_digest="d" * 64,
    )
    with pytest.raises(ValueError, match="binding changed"):
        replay_executor.execute(
            name="create_folder", arguments={"name": "Archive"}, call_ordinal=0,
            resolved_targets=legacy_capability,
        )


def test_telegram_executor_persists_and_checks_private_resolved_target_binding(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor, stable_effect_id
    from core.operations.write import _new_resolved_write_targets

    now = BASE_TIME + timedelta(minutes=4)
    root_id, folder_id = "t009-target-root", "t009-target-folder"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection("folders").document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, _lease, _repository, executor = _executor_for_folder_write(
        client, now=now, chat_id=93401, monkeypatch=monkeypatch,
    )
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    arguments = {"name": "Archive"}
    capability = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id,
        effect_id=stable_effect_id(accepted.operation_id, 0), arguments=arguments,
        parent_id=root_id, root_id=root_id, intent_digest="c" * 64,
    )
    allocator_payload = json.dumps(
        {"name": "Archive", "parent_id": root_id, "root_id": root_id,
         "intent_digest": "c" * 64},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    allocator_intent = "sha256:" + hashlib.sha256(allocator_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, stable_effect_id(accepted.operation_id, 0),
                   "drive_folder", allocator_intent)).encode("utf-8")
    ).hexdigest()
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [folder_id]})
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.create", response={"id": folder_id})
    sdk_fakes.drive.queue("files.get", response={
        "id": folder_id, "name": "Archive",
        "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
        "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
    })

    result = executor.execute(
        name="create_folder", arguments=arguments, call_ordinal=0,
        resolved_targets=capability,
    )
    checkpoint_ref = (client.collection("telegram_sessions").document(accepted.context_id)
                      .collection("operations").document(accepted.operation_id)
                      .collection("agent_checkpoints").document("tool-0"))
    checkpoint = checkpoint_ref.get().to_dict()
    assert result == folder_id
    from core.operations.effects import get_effect_allocation
    allocation = get_effect_allocation(client, child_id, stable_effect_id(accepted.operation_id, 0))
    assert allocation["resource_id"] == folder_id
    assert allocation["intent_digest"] == allocator_intent
    assert checkpoint["validated_input"]["arguments"] == arguments
    assert checkpoint["resolved_targets_binding"]["parent_id"] == root_id
    assert "resolved_targets" not in checkpoint["validated_input"]["arguments"]

    replay = TelegramRegistryExecutor(
        client=client, context=executor.execution_context, save_handler=object(),
        execution_context=executor.execution_context,
    )
    trace = sdk_fakes.trace
    assert replay.execute(
        name="create_folder", arguments=arguments, call_ordinal=0,
        resolved_targets=capability,
    ) == folder_id
    assert replay.reconcile(
        name="create_folder", arguments=arguments, call_ordinal=0,
        resolved_targets=capability,
    ) == folder_id
    assert replay.reconcile(name="create_folder", arguments=arguments,
                            call_ordinal=0) is None
    assert sdk_fakes.trace == trace
    from dataclasses import replace
    with pytest.raises(ValueError, match="binding changed"):
        replay.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets=replace(capability, intent_digest="d" * 64),
        )
    with pytest.raises(ValueError, match="not trusted"):
        replay.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets={"parent_id": root_id},
        )
    from core.operations import registry
    with pytest.raises(registry.OperationValidationError):
        replay.execute(
            name="create_folder", arguments={**arguments, "resolved_targets": {"parent_id": root_id}},
            call_ordinal=0,
        )
    with pytest.raises(ValueError, match="binding changed"):
        replay.execute(name="create_folder", arguments=arguments, call_ordinal=0)
    with pytest.raises(ValueError, match="do not match"):
        replay.execute(
            name="create_folder", arguments={"name": "Different"}, call_ordinal=0,
            resolved_targets=capability,
        )
    assert sdk_fakes.trace == trace


def test_telegram_claim_loser_rechecks_cached_target_binding(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.operations import claim_operation, complete_operation
    from core.operations.models import OperationResult
    from core.operations.write import _new_resolved_write_targets
    from services.connectors.telegram.runtime.tools import stable_effect_id
    from tests.fixtures.firestore_fake import FakeDocumentReference

    now = BASE_TIME + timedelta(minutes=5)
    root_id = "t009-race-root"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection("folders").document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, _lease, _repository, executor = _executor_for_folder_write(
        client, now=now, chat_id=93501, monkeypatch=monkeypatch,
    )
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    arguments = {"name": "Archive"}
    effect_id = stable_effect_id(accepted.operation_id, 0)
    capability = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id, effect_id=effect_id,
        arguments=arguments, parent_id=root_id, root_id=root_id, intent_digest="f" * 64,
    )
    checkpoint_ref = (client.collection("telegram_sessions").document(accepted.context_id)
                      .collection("operations").document(accepted.operation_id)
                      .collection("agent_checkpoints").document("tool-0"))
    original_get = FakeDocumentReference.get
    injected = False

    def get_with_racing_checkpoint(reference, *, transaction=None, **kwargs):
        nonlocal injected
        snapshot = original_get(reference, transaction=transaction, **kwargs)
        if reference.path != checkpoint_ref.path or transaction is not None or injected:
            return snapshot
        assert not snapshot.exists
        injected = True
        from core.integrations.firestore.operations import get_operation
        child = get_operation(client, child_id)
        assert child is not None
        raced = claim_operation(
            client, child_id, owner="t009-racing-worker", now=now, lease_seconds=120,
        )
        assert raced is not None
        digest = hashlib.sha256(json.dumps(
            {"name": "create_folder", "arguments": arguments}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        checkpoint_ref.set({
            "operation_id": child_id,
            "validated_input": {"name": "create_folder", "arguments": arguments,
                                "digest": digest},
            "resolved_targets_binding": {"operation_name": "create_folder",
                                          "operation_id": child_id,
                                          "effect_id": effect_id, "intent_digest": "0" * 64},
            "result": "racing-result",
        }, merge=True)
        complete_operation(
            client, child_id, owner="t009-racing-worker", fence=raced.fence,
            result=OperationResult(title="racing-result"), now=now,
        )
        return snapshot

    monkeypatch.setattr(FakeDocumentReference, "get", get_with_racing_checkpoint)
    with pytest.raises(ValueError, match="checkpoint binding changed"):
        executor.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets=capability,
        )
    assert sdk_fakes.trace == []
    assert injected


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
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
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
# T016: metadata-only contextual checkpoints are decisions, not executed tools.
def _t016_context_snapshot(repository, accepted, command, *, actor_id, conversation_id):
    import hashlib
    from services.connectors.telegram.context import build_conversation_history
    from services.connectors.telegram.runtime.routing_context import (
        BoundRef, RoutingContextSnapshot,
    )

    operation = repository.client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).get().to_dict()
    command_ref = BoundRef(
        ref=operation["input_ref"],
        digest=hashlib.sha256(command.text.encode("utf-8")).hexdigest(),
        context_id=accepted.context_id, actor_id=actor_id,
        conversation_id=conversation_id, operation_id=accepted.operation_id,
        payload={"text": command.text},
    )
    canonical_turns = repository.list_turns(accepted.context_id, conversation_id)
    projected_history = build_conversation_history(
        canonical_turns, conversation_id=conversation_id,
        omit_operation_id=accepted.operation_id,
    )
    history_bytes = json.dumps(
        projected_history, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    history_ref = BoundRef(
        ref=(f"telegram_sessions/{accepted.context_id}/conversations/"
             f"{conversation_id}/turns"),
        digest=hashlib.sha256(history_bytes).hexdigest(),
        context_id=accepted.context_id, actor_id=actor_id,
        conversation_id=conversation_id, operation_id=accepted.operation_id,
        payload={"turns": projected_history},
    )
    snapshot = RoutingContextSnapshot(
        context_id=accepted.context_id, actor_id=actor_id,
        operation_id=accepted.operation_id,
        selected_conversation_id=conversation_id,
        selected_conversation_kind="chat", accepted_command=command.text,
        accepted_command_ref=command_ref, full_history_ref=history_ref,
    )
    snapshot_value = history_ref.to_dict()["payload"]
    assert snapshot_value == {"turns": projected_history}
    assert snapshot.accepted_command_ref.payload == {"text": command.text}
    parts = operation["input_ref"].split("/")
    assert len(parts) == 4 and parts[:3] == [
        "telegram_sessions", accepted.context_id, "commands",
    ]
    command_record = (
        repository.client.collection(parts[0]).document(parts[1])
        .collection(parts[2]).document(parts[3]).get()
    ).to_dict()
    assert command_record is not None
    assert command_record.get("candidate") == command.text
    assert snapshot.full_history_ref.ref == (
        f"telegram_sessions/{accepted.context_id}/conversations/{conversation_id}/turns"
    )
    assert snapshot.full_history_ref.digest == hashlib.sha256(history_bytes).hexdigest()
    return snapshot


def test_t016_metadata_only_application_route_has_no_synthetic_tool_result(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.operation_queue import claim_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, RoutingDecision,
    )
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client = firestore_fake.client_class()
    scenario = make_contextual_scenario(unrelated_count=0)
    repository = TelegramRepository(client=client)
    command = parse_command("Explain the current discussion")
    accepted = repository.accept_command(
        99106, chat_id=scenario.chat.chat_id, thread_id=scenario.topic.thread_id,
        user_id=scenario.actor.user_id, command=command, now=BASE_TIME,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="t016-metadata-worker", now=BASE_TIME)
    assert lease is not None
    source_conversation = "t016-source-conversation"
    checkpoint_conversation = "t016-fixed-checkpoint-conversation"
    repository.ensure_conversation(
        accepted.context_id, source_conversation, kind="chat",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    repository.append_turn(
        accepted.context_id, source_conversation, turn_id="t016-input",
        operation_id=accepted.operation_id, role="user", parts=[{"text": command.text}],
        now=BASE_TIME, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence,
    )
    snapshot = _t016_context_snapshot(
        repository, accepted, command, actor_id=scenario.actor.user_id,
        conversation_id=source_conversation,
    )
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
    )
    context_checkpoint = ledger.save_routing_checkpoint(
        "context", actor_id=scenario.actor.user_id,
        conversation_id=source_conversation, value=snapshot.to_dict(),
    )
    decision = RoutingDecision(
        disposition=Disposition.DELEGATE, action=Action.SUBSTANTIVE,
        substantive_text=command.text, certainty=Certainty.CERTAIN,
    )
    ordinary_summary = {"status": "ok", "kind": "substantive"}
    assert repository.persist_route(
        accepted.operation_id, ordinary_summary,
        context_id=accepted.context_id, owner=lease.owner, fence=lease.fence,
        now=BASE_TIME,
    )
    persisted_summary = repository.get_route_control(
        accepted.context_id, accepted.operation_id,
    )
    metadata_route = {"contextual": {
        "schema_version": 1, "context_ref": context_checkpoint["ref"],
        "context_digest": context_checkpoint["digest"],
        "context_id": accepted.context_id, "operation_id": accepted.operation_id,
        "actor_id": scenario.actor.user_id, "accepted_command": command.text,
        "accepted_command_ref": snapshot.accepted_command_ref.to_dict(),
        "full_history_ref": snapshot.full_history_ref.to_dict(),
        "source_conversation_id": source_conversation,
        "checkpoint_conversation_id": checkpoint_conversation,
        "decision": decision.to_dict(), "resolved": {}, "fixed_plan": None,
        "summary": persisted_summary,
    }}
    ledger.save_application_route(
        checkpoint_conversation, accepted.operation_id, route=metadata_route,
        tool_calls=[], application_route_round=None,
    )

    from google.genai import types
    from services.connectors.telegram.runtime.tools import TelegramRegistryExecutor
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(
            text="The discussion is about the quoted price.",
        )]))],
    ))
    model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t016-metadata-key"))
    executor_context = {
        "context_id": accepted.context_id, "operation_id": accepted.operation_id,
        "owner": lease.owner, "fence": lease.fence, "lease": lease,
        "clock": lambda: BASE_TIME,
    }
    executor = TelegramRegistryExecutor(
        client=client, context=executor_context, save_handler=object(),
        execution_context=executor_context,
    )
    agent = TelegramAgent(model=model, ledger=ledger, tool_executor=executor)
    before = ledger.load_turn(checkpoint_conversation, accepted.operation_id)
    assert before["model_response"] is None
    assert before["tool_calls"] == []
    assert before.get("tool_results", {}) == {}
    try:
        result = agent.run(
            conversation_id=checkpoint_conversation,
            delivery_id=accepted.operation_id,
            history=[{"role": "user", "parts": [{"text": command.text}]}],
        )
    except Exception as error:
        pytest.fail(f"metadata-only checkpoint was treated as an executed tool: {error!r}")

    restored = ledger.load_turn(checkpoint_conversation, accepted.operation_id)
    assert result["status"] == "ok"
    assert result["answer"] == "The discussion is about the quoted price."
    assert sdk_fakes.trace and not any(
        row.get("operation") in {"files.create", "files.update"}
        for row in sdk_fakes.trace
    )
    assert restored["application_route"] == metadata_route
    assert restored["model_response"] is not None
    assert restored["model_response"]["_telegram_tool_loop_rounds"] == 1
    assert len(restored["model_response"]["rounds"]) == 1
    answer_round = restored["model_response"]["rounds"][0]
    assert answer_round["call_start"] == 0 and answer_round["call_count"] == 0
    answer_response = answer_round["response"]
    assert len(answer_response["candidates"]) == 1
    answer_parts = answer_response["candidates"][0]["content"]["parts"]
    assert len(answer_parts) == 1
    answer_part = answer_parts[0]
    assert answer_part.get("text") == result["answer"]
    assert not any("function_call" in part or "function_response" in part for part in answer_parts)
    sdk_payloads = [
        row for row in sdk_fakes.trace
        if row.get("boundary") == "gemini" and row.get("operation") == "generate_content"
    ]
    assert len(sdk_payloads) == 1
    sdk_contents = repr(sdk_payloads[0].get("arguments", {}).get("contents", ()))
    assert "function_response" not in sdk_contents
    assert "<application-tool-result-data>" not in sdk_contents
    assert "accepted_command_ref" not in sdk_contents
    assert "full_history_ref" not in sdk_contents
    assert "saved-routing-decision" not in sdk_contents.lower()
    assert "function_response" not in repr(restored["model_response"])
    assert "<application-tool-result-data>" not in repr(restored["model_response"])
    assert restored["tool_calls"] == []
    assert restored.get("tool_results", {}) == {}
    with pytest.raises(ValueError, match="metadata route cannot become"):
        agent.dispatch_tool_choice(
            name="query", arguments={"query": "new executable choice"},
            target_doc_id=None, conversation_id=checkpoint_conversation,
            delivery_id=accepted.operation_id,
            executor_factory=lambda *_args: executor,
        )


def _t016_seed_prepared_clarification(
    firestore_fake, *, complete, joint=False, resolved_parameters=None,
):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.models import (
        ClarificationState, PendingClarification, build_clarification_id,
    )
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.repository._clarifications import build_clarification_request
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition, FolderSelector,
        RoutingDecision,
    )
    from tests.fixtures.telegram_contextual import make_contextual_scenario

    client = firestore_fake.client_class()
    scenario = make_contextual_scenario(unrelated_count=0)
    command = parse_command("Please create a folder")
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        99116, chat_id=scenario.chat.chat_id, thread_id=scenario.topic.thread_id,
        user_id=scenario.actor.user_id, command=command, now=BASE_TIME,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="t016-clarification-worker", now=BASE_TIME)
    assert lease is not None
    source_conversation = "t016-source-chat"
    target_checkpoint = "t016-fixed-checkpoint-conversation"
    repository.ensure_conversation(
        accepted.context_id, source_conversation, kind="chat",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    repository.append_turn(
        accepted.context_id, source_conversation, turn_id="t016-source-command",
        operation_id=accepted.operation_id, role="user", parts=[{"text": command.text}],
        now=BASE_TIME, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence,
    )
    operation = context_ref.collection("operations").document(accepted.operation_id).get().to_dict()
    snapshot = _t016_context_snapshot(
        repository, accepted, command, actor_id=scenario.actor.user_id,
        conversation_id=source_conversation,
    )
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
    )
    context_checkpoint = ledger.save_routing_checkpoint(
        "context", actor_id=scenario.actor.user_id, conversation_id=source_conversation,
        value=snapshot.to_dict(),
    )
    clarification_id = build_clarification_id(accepted.context_id, accepted.operation_id)
    prefix = f"telegram_sessions/{accepted.context_id}/clarifications/{clarification_id}/"
    pending = PendingClarification(
        clarification_id=clarification_id, context_id=accepted.context_id,
        actor_id=scenario.actor.user_id, initiating_operation_id=accepted.operation_id,
        origin_conversation_id=source_conversation,
        original_command_ref=context_checkpoint["ref"],
        original_command_digest=context_checkpoint["digest"], action="create_folder",
        resolved_parameters=(resolved_parameters if resolved_parameters is not None
                             else {"parent_id": "pytest-synthetic-root"}),
        missing_slots=("folder_name",),
        evidence_snapshot_ref=context_checkpoint["ref"],
        evidence_snapshot_digest=context_checkpoint["digest"],
        state=ClarificationState.WAITING, created_at=BASE_TIME, updated_at=BASE_TIME,
        questions_ref=prefix + "questions", answers_ref=prefix + "answers",
    )
    request = build_clarification_request(
        pending, context_checkpoint,
        question={"slot": "folder_name", "text": "\u041a\u0430\u043a \u043d\u0430\u0437\u0432\u0430\u0442\u044c \u043f\u0430\u043f\u043a\u0443?", "options": []},
        prompt_body="Как назвать папку?",
    )
    intent_checkpoint = ledger.save_routing_checkpoint(
        "intent", actor_id=scenario.actor.user_id, conversation_id=source_conversation,
        value=request,
    )
    decision = RoutingDecision(
        disposition=Disposition.CLARIFY, action=Action.CREATE_FOLDER,
        substantive_text=command.text,
        folder_selector=FolderSelector(kind="root_default"),
        missing_slots=("folder_name",), certainty=Certainty.UNCERTAIN,
    )
    ordinary_summary = {"status": "uncertain", "kind": "substantive"}
    if joint:
        persisted_summary = repository._route_control_data(ordinary_summary)
    else:
        assert repository.persist_route(
            accepted.operation_id, ordinary_summary,
            context_id=accepted.context_id, owner=lease.owner, fence=lease.fence,
            now=BASE_TIME,
        )
        persisted_summary = repository.get_route_control(
            accepted.context_id, accepted.operation_id,
        )
    # This is valid fixture state, not evidence that the production publisher
    # currently writes the route summary and application checkpoint atomically.
    metadata_route = {"contextual": {
        "schema_version": 1,
        "context_ref": context_checkpoint["ref"],
        "context_digest": context_checkpoint["digest"],
        "context_id": accepted.context_id,
        "operation_id": accepted.operation_id,
        "actor_id": scenario.actor.user_id,
        "accepted_command": command.text,
        "accepted_command_ref": snapshot.accepted_command_ref.to_dict(),
        "full_history_ref": snapshot.full_history_ref.to_dict(),
        "source_conversation_id": source_conversation,
        "checkpoint_conversation_id": target_checkpoint,
        "decision": decision.to_dict(),
        "resolved": pending.to_dict()["resolved_parameters"],
        "fixed_plan": None,
        "intent_ref": intent_checkpoint["ref"],
        "intent_digest": intent_checkpoint["digest"],
        "summary": persisted_summary,
    }}
    if joint:
        prepared = ledger.prepare_application_route(
            target_checkpoint, accepted.operation_id, route=metadata_route,
            tool_calls=[], application_route_round=None,
        )
        assert repository.persist_route(
            accepted.operation_id, ordinary_summary,
            context_id=accepted.context_id, owner=lease.owner, fence=lease.fence,
            now=BASE_TIME, application_checkpoint=prepared,
        )
    else:
        ledger.save_application_route(
            target_checkpoint, accepted.operation_id, route=metadata_route,
            tool_calls=[], application_route_round=None,
        )
    completion = repository.prepare_clarification_completion(intent_checkpoint)
    if complete:
        assert repository.complete_with_clarification(
            accepted.context_id, accepted.operation_id, owner=lease.owner,
            fence=lease.fence, completion=completion, now=BASE_TIME,
            clock=lambda: BASE_TIME,
        )
    return (client, repository, accepted, operation, lease, completion,
            metadata_route, intent_checkpoint)


def test_joint_clarification_preserves_frozen_known_folder_path(
    firestore_fake, sdk_fakes,
):
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    (client, _repository, accepted, _operation, lease, _completion,
     route, _intent_reference) = _t016_seed_prepared_clarification(
        firestore_fake, complete=False, joint=True,
        resolved_parameters={
            "parent_id": "pytest-synthetic-root",
            "folder_path": ["Projects", "Archive"],
        },
    )
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
    )
    restored = ledger.load_turn(
        "t016-fixed-checkpoint-conversation", accepted.operation_id,
    )
    assert restored["application_route"]["contextual"]["resolved"] == {
        "parent_id": "pytest-synthetic-root",
        "folder_path": ["Projects", "Archive"],
    }
    assert restored["tool_calls"] == []


def test_t016_stored_clarification_without_intent_fails_before_agent_execution(
    firestore_fake, sdk_fakes,
):
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    (client, _repository, accepted, _operation, lease, _completion,
     route, _intent_reference) = _t016_seed_prepared_clarification(
        firestore_fake, complete=False,
    )
    route["contextual"].pop("intent_ref")
    route["contextual"].pop("intent_digest")
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
    )
    turn_ref = ledger._ref(accepted.operation_id)
    turn = turn_ref.get().to_dict()
    turn["application_route"] = ledger._store_payload("application-route", route)
    turn_ref.set(turn)
    constructed = []
    agent = TelegramAgent(model=object(), ledger=ledger, tool_executor=object())
    with pytest.raises(ValueError, match="persisted application checkpoint is invalid"):
        agent.dispatch_tool_choice(
            name="query", arguments={"query": "must not run"}, target_doc_id=None,
            conversation_id="t016-fixed-checkpoint-conversation",
            delivery_id=accepted.operation_id,
            executor_factory=lambda *_args: constructed.append(True),
        )
    assert constructed == []


def _t016_clarification_publication_snapshot(client, accepted, durable_documents):
    import hashlib

    from services.connectors.telegram.repository._common import NOTIFICATIONS
    from core.integrations.firestore.outbox import OUTBOX_COLLECTION

    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    operation = context_ref.collection("operations").document(accepted.operation_id).get().to_dict()
    result = operation["result"]

    def read_path(path):
        segments = path.split("/")
        assert len(segments) >= 2 and len(segments) % 2 == 0
        reference = client.collection(segments[0]).document(segments[1])
        for index in range(2, len(segments), 2):
            reference = reference.collection(segments[index]).document(segments[index + 1])
        return reference.get().to_dict()

    question = read_path(result["question_ref"])
    notification = client.collection(NOTIFICATIONS).document(
        result["prompt_notification_id"],
    ).get().to_dict()
    outbox = client.collection(OUTBOX_COLLECTION).document(
        result["prompt_outbox_id"],
    ).get().to_dict()
    assert question["question_ref"] == result["question_ref"]
    assert question["question_payload_digest"] == result["request_digest"]
    assert notification["body"] == "Как назвать папку?"
    assert question["prompt_digest"] == hashlib.sha256(
        notification["body"].encode("utf-8")
    ).hexdigest()
    assert question.get("message_id") is None
    assert notification.get("message_id") is None
    assert notification.get("operation_id") is None
    assert outbox.get("message_id") is None
    assert outbox.get("delivery_id") is None
    clarification_id = result["clarification_id"]
    question_prefix = (
        f"telegram_sessions/{accepted.context_id}/clarifications/"
        f"{clarification_id}/questions/"
    )
    question_paths = sorted(
        path for path in durable_documents if path.startswith(question_prefix)
    )
    notification_rows = {
        path: value for path, value in durable_documents.items()
        if path.startswith(f"{NOTIFICATIONS}/")
        and value.get("context_id") == accepted.context_id
        and value.get("clarification_id") == clarification_id
    }
    notification_ids = {
        value.get("notification_id") for value in notification_rows.values()
    }
    outbox_rows = {
        path: value for path, value in durable_documents.items()
        if path.startswith(f"{OUTBOX_COLLECTION}/")
        and value.get("notification_id") in notification_ids
    }
    assert len(question_paths) == 1, question_paths
    assert len(notification_rows) == 1, notification_rows
    assert len(outbox_rows) == 1, outbox_rows
    return {
        "result": operation["result"], "question": question,
        "notification": notification, "outbox": outbox,
        "question_paths": question_paths,
        "notification_paths": sorted(notification_rows),
        "outbox_paths": sorted(outbox_rows),
    }


@pytest.mark.parametrize("completed_before_restart", [False, True])
def test_t016_clarification_restart_reuses_prepared_question_without_model(
    firestore_fake, sdk_fakes, completed_before_restart,
):
    import asyncio
    import httpx
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    (client, repository, accepted, operation, lease, completion,
     metadata_route, intent_checkpoint) = _t016_seed_prepared_clarification(
        firestore_fake, complete=completed_before_restart,
    )
    async def telegram_handler(_request):
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99117}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(telegram_handler))
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=TelegramSaveRuntime(repository=repository),
        client=TelegramClient("synthetic-t016-token", http_client=http_client),
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t016-clarification-key")),
        clock=lambda: BASE_TIME + (timedelta(days=3) if completed_before_restart
                                   else timedelta(seconds=1)),
    )
    model_trace_before_restart = list(sdk_fakes.trace)
    operation_row = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).get().to_dict()
    before_completion_snapshot = (
        _t016_clarification_publication_snapshot(
            client, accepted, firestore_fake.documents,
        )
        if completed_before_restart else None
    )
    try:
        if completed_before_restart:
            # T011 already committed the real queue result. A duplicate update
            # must reuse that terminal outcome without reviving its old lease.
            from services.connectors.telegram.commands import parse_command

            context_row = client.collection("telegram_sessions").document(
                accepted.context_id,
            ).get().to_dict()
            duplicate = repository.accept_command(
                99116,
                chat_id=context_row["chat_id"],
                thread_id=context_row.get("thread_id"),
                user_id=17,
                command=parse_command("Please create a folder"),
                now=BASE_TIME + timedelta(days=3),
            )
            assert duplicate.operation_id == accepted.operation_id
            assert operation_row["state"] == "completed"
            assert operation_row["result_type"] == "clarification"
            restored = repository.prepare_clarification_completion(intent_checkpoint)
            assert restored.request == completion.request
            assert restored.pending.clarification_id == completion.pending.clarification_id
            assert before_completion_snapshot["result"] == operation_row["result"]
            assert sdk_fakes.trace == model_trace_before_restart
        else:
            from services.connectors.telegram.queue import TelegramQueueCompletion

            result = asyncio.run(
                runtime(accepted.context_id, operation, lease, command=None),
            )
            assert isinstance(result, TelegramQueueCompletion), (
                "a pre-completion restart must return the same trusted prepared outcome"
            )
            assert result.clarification.request == completion.request
            assert result.clarification.pending.clarification_id == completion.pending.clarification_id
            assert sdk_fakes.trace == model_trace_before_restart
            # Simulate the generic queue's actual T011 completion boundary only
            # after the processor has returned the prepared repository value.
            assert repository.complete_with_clarification(
                accepted.context_id, accepted.operation_id, owner=lease.owner,
                fence=lease.fence, completion=result.clarification,
                now=BASE_TIME + timedelta(seconds=2), clock=lambda: BASE_TIME + timedelta(seconds=2),
            )
    finally:
        asyncio.run(http_client.aclose())

    completed_row = client.collection("telegram_sessions").document(
        accepted.context_id,
    ).collection("operations").document(accepted.operation_id).get().to_dict()
    assert completed_row["state"] == "completed"
    assert completed_row["result_type"] == "clarification"
    assert repository.get_route_control(accepted.context_id, accepted.operation_id) is not None
    assert metadata_route["contextual"]["intent_ref"] == intent_checkpoint["ref"]
    after_completion_snapshot = _t016_clarification_publication_snapshot(
        client, accepted, firestore_fake.documents,
    )
    if completed_before_restart:
        assert after_completion_snapshot == before_completion_snapshot
    assert after_completion_snapshot["result"] == completed_row["result"]
    contextual = metadata_route["contextual"]
    assert contextual["source_conversation_id"] != contextual["checkpoint_conversation_id"]
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
    )
    fixed_turn = ledger.load_turn(
        contextual["checkpoint_conversation_id"], accepted.operation_id,
    )
    assert fixed_turn["application_route"] == metadata_route


def test_t016_governed_cached_success_cannot_drop_a_trusted_target_binding(
    firestore_fake, sdk_fakes, monkeypatch,
):
    import json
    from types import SimpleNamespace

    from core.integrations.firestore.client import FOLDERS_COLLECTION
    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor
    from core.operations.write import _new_resolved_write_targets
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor
    from services.connectors.telegram.runtime.tools import stable_effect_id

    now = BASE_TIME + timedelta(minutes=8)
    root_id, folder_id = "t016-cache-root", "t016-cache-folder"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection(FOLDERS_COLLECTION).document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, lease, repository, inner = _executor_for_folder_write(
        client, now=now, chat_id=93616, monkeypatch=monkeypatch,
    )
    operation_id = accepted.operation_id
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    effect_id = stable_effect_id(operation_id, 0)
    arguments = {"name": "Archive"}
    capability = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id,
        effect_id=effect_id, arguments=arguments,
        parent_id=root_id, root_id=root_id, intent_digest="a" * 64,
    )
    allocator_payload = json.dumps(
        {"name": "Archive", "parent_id": root_id, "root_id": root_id,
         "intent_digest": "a" * 64},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    allocator_intent = "sha256:" + hashlib.sha256(allocator_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", allocator_intent)).encode("utf-8")
    ).hexdigest()
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [folder_id]})
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.create", response={"id": folder_id})
    sdk_fakes.drive.queue("files.get", response={
        "id": folder_id, "name": "Archive",
        "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
        "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
    })

    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    governor = RetryGovernor(
        client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(client, context_ref),
    )
    actual = governor.run_stage(
        operation_id, "telegram-tool-00",
        lambda: inner.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets=capability,
        ),
        now=now, owner=lease.owner, fence=lease.fence,
    )
    assert actual == folder_id
    child_checkpoint = context_ref.collection("operations").document(operation_id).collection(
        "agent_checkpoints",
    ).document("tool-0").get().to_dict()
    assert child_checkpoint["resolved_targets_binding"]["parent_id"] == root_id

    governed = _GovernedExecutor(
        inner, governor, operation_id, lease, lambda: now, lambda: True,
    )
    trace = list(sdk_fakes.trace)
    # The ordinary governed caller has no typed capability to prove it is the
    # same fixed target. It must not inherit the cached success produced under
    # capability A without revalidating that binding.
    from core.operations.retries import ReconciledStageFailure, UnknownOutcome

    try:
        cached = governed.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
        )
    except (PermissionError, ValueError, UnknownOutcome, ReconciledStageFailure):
        cached = None
    assert sdk_fakes.trace == trace
    assert repository.get_route_control(accepted.context_id, operation_id) is None

    # A legacy None-to-None child remains a positive reconciliation control.
    second_folder = "t016-cache-folder-reconcile"
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{operation_id}\0{1}".encode("utf-8")
    ).hexdigest()
    effect_id = stable_effect_id(operation_id, 1)
    second_arguments = {"name": "Archive-Reconcile"}
    legacy_intent = "sha256:" + hashlib.sha256(json.dumps(
        {"name": "Archive-Reconcile", "parent_id": root_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", legacy_intent)).encode("utf-8")
    ).hexdigest()
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [second_folder]})
    sdk_fakes.drive.queue("files.get", response={
        "id": root_id, "name": "TLBrain",
        "mimeType": "application/vnd.google-apps.folder", "parents": [], "trashed": False,
    })
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response={
        "id": root_id, "name": "TLBrain",
        "mimeType": "application/vnd.google-apps.folder", "parents": [], "trashed": False,
    })
    sdk_fakes.drive.queue("files.create", response={"id": second_folder})
    sdk_fakes.drive.queue("files.get", response={
        "id": second_folder, "name": "Archive-Reconcile",
        "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
        "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
    })

    def lose_ack_after_real_child_write():
        result = inner.execute(
            name="create_folder", arguments=second_arguments, call_ordinal=1,
        )
        assert result == second_folder
        raise UnknownOutcome("synthetic lost acknowledgement after confirmed child")

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            operation_id, "telegram-tool-01", lose_ack_after_real_child_write,
            now=now, owner=lease.owner, fence=lease.fence,
        )
    before_reconcile = list(sdk_fakes.trace)
    try:
        reconciled = governed.execute(
            name="create_folder", arguments=second_arguments, call_ordinal=1,
        )
    except Exception as error:
        pytest.fail(
            "governed reconciliation failed to recover the saved trusted binding: "
            f"{type(error).__name__}"
        )
    assert reconciled == second_folder
    assert sdk_fakes.trace == before_reconcile
    second_checkpoint = context_ref.collection("operations").document(operation_id).collection(
        "agent_checkpoints",
    ).document("tool-1").get().to_dict()
    assert second_checkpoint.get("resolved_targets_binding") is None
    assert cached is None, "unbound governor invocation adopted a typed-target success"


def _t016_seed_real_folder_application(
    firestore_fake, sdk_fakes, monkeypatch, *, seed_completed=True,
):
    from core.integrations.firestore.client import FOLDERS_COLLECTION
    from core.integrations.firestore.operation_queue import claim_head
    from core.operations.write import _new_resolved_write_targets
    from services.connectors.telegram.agent import TelegramAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
    from services.connectors.telegram.runtime.routing_context import (
        Action, Certainty, Disposition,
        FolderSelector, FolderSelectorKind, RoutingContextSnapshot, RoutingDecision,
    )
    from services.connectors.telegram.runtime.tools import (
        TelegramRegistryExecutor, stable_effect_id,
    )

    now = BASE_TIME + timedelta(minutes=9)
    root_id, folder_id = "t016-agent-root", "t016-agent-folder"
    monkeypatch.setenv("ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{root_id}")
    client = firestore_fake.client_class()
    client.collection(FOLDERS_COLLECTION).document(root_id).set({
        "folder_id": root_id, "name": "TLBrain", "parent_id": None,
    })
    accepted, _operation, lease, repository, executor = _executor_for_folder_write(
        client, now=now, chat_id=93617, monkeypatch=monkeypatch,
    )
    command = parse_command("Create the folder Archive")
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    repository.ensure_conversation(
        accepted.context_id, "t016-source-chat", kind="chat",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=now,
    )
    repository.append_turn(
        accepted.context_id, "t016-source-chat", turn_id="t016-agent-input",
        operation_id=accepted.operation_id, role="user",
        parts=[{"text": command.text}], now=now,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence,
    )
    snapshot = _t016_context_snapshot(
        repository, accepted, command, actor_id=17, conversation_id="t016-source-chat",
    )
    ledger = FirestoreAgentLedger(
        client, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    context_checkpoint = ledger.save_routing_checkpoint(
        "context", actor_id=17, conversation_id="t016-source-chat",
        value=snapshot.to_dict(),
    )
    decision = RoutingDecision(
        disposition=Disposition.EXECUTE, action=Action.CREATE_FOLDER,
        substantive_text=command.text,
        folder_selector=FolderSelector(kind=FolderSelectorKind.ROOT_DEFAULT),
        folder_name="Archive",
        certainty=Certainty.CERTAIN,
    )
    decision.validate_command_spans(command.text)
    assert RoutingContextSnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert RoutingDecision.from_dict(decision.to_dict()) == decision
    summary = {
        "status": "ok", "kind": "substantive",
        "tool_choice": {
            "name": "create_folder", "arguments": {"name": "Archive"},
            "link_requested": False,
        },
    }
    if seed_completed:
        assert repository.persist_route(
            accepted.operation_id, summary, context_id=accepted.context_id,
            owner=lease.owner, fence=lease.fence, now=now,
        )
        persisted_summary = repository.get_route_control(
            accepted.context_id, accepted.operation_id,
        )
    else:
        persisted_summary = repository._route_control_data(summary)
        assert repository._route_control_data(persisted_summary) == persisted_summary
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{accepted.operation_id}\0{0}".encode("utf-8")
    ).hexdigest()
    effect_id = stable_effect_id(accepted.operation_id, 0)
    arguments = {"name": "Archive"}
    intent_digest = "b" * 64
    capability = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id,
        effect_id=effect_id, arguments=arguments, parent_id=root_id,
        root_id=root_id, intent_digest=intent_digest,
    )
    allocator_payload = json.dumps(
        {"name": "Archive", "parent_id": root_id, "root_id": root_id,
         "intent_digest": intent_digest},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    allocator_intent = "sha256:" + hashlib.sha256(allocator_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", allocator_intent)).encode("utf-8")
    ).hexdigest()
    folder_result = None
    if seed_completed:
        sdk_fakes.drive.queue("files.generateIds", response={"ids": [folder_id]})
        root_metadata = {"id": root_id, "name": "TLBrain",
                         "mimeType": "application/vnd.google-apps.folder",
                         "parents": [], "trashed": False}
        sdk_fakes.drive.queue("files.get", response=root_metadata)
        sdk_fakes.drive.queue("files.get", error=_http_404())
        sdk_fakes.drive.queue("files.get", error=_http_404())
        sdk_fakes.drive.queue("files.get", response=root_metadata)
        sdk_fakes.drive.queue("files.create", response={"id": folder_id})
        sdk_fakes.drive.queue("files.get", response={
            "id": folder_id, "name": "Archive",
            "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
            "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
        })
        folder_result = executor.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets=capability,
        )
        assert folder_result == folder_id

    checkpoint_conversation = "t016-fixed-agent-checkpoint"
    application_route = {
        "name": "create_folder", "arguments": arguments, "ordinal": 0,
        "target_doc_id": None, "call_id": f"application-{accepted.operation_id}-0",
        "contextual": {
            "schema_version": 1, "context_ref": context_checkpoint["ref"],
            "context_digest": context_checkpoint["digest"], "actor_id": 17,
            "context_id": accepted.context_id, "operation_id": accepted.operation_id,
            "accepted_command": command.text,
            "accepted_command_ref": snapshot.accepted_command_ref.to_dict(),
            "full_history_ref": snapshot.full_history_ref.to_dict(),
            "source_conversation_id": "t016-source-chat",
            "checkpoint_conversation_id": checkpoint_conversation,
            "decision": decision.to_dict(),
            "resolved": {"parent_id": root_id, "root_id": root_id,
                         "intent_digest": intent_digest},
            "summary": persisted_summary,
            "fixed_plan": {"steps": [{
                "ordinal": 0, "name": "create_folder", "arguments": arguments,
                "target_doc_id": None, "operation_id": child_id, "effect_id": effect_id,
                "resolved_targets": {"parent_id": root_id, "root_id": root_id,
                                      "intent_digest": intent_digest},
            }]},
        },
    }
    application_call = {
        "id": application_route["call_id"], "name": "create_folder",
        "arguments": arguments, "ordinal": 0, "round": -1,
    }
    if seed_completed:
        ledger.save_application_route(
            checkpoint_conversation, accepted.operation_id, route=application_route,
            tool_calls=[application_call], application_route_round=None,
        )
        ledger.save_tool_result(
            checkpoint_conversation, accepted.operation_id, 0, folder_result,
        )
        assert repository.get_route_control(accepted.context_id, accepted.operation_id) == (
            persisted_summary
        )
    agent = TelegramAgent(model=None, ledger=ledger, tool_executor=executor)
    return agent, ledger, executor, repository, (
        accepted.operation_id, checkpoint_conversation, application_route,
        arguments, folder_result, persisted_summary, capability,
    )


def test_t016_saved_application_dispatch_requires_exact_action_arguments_and_target(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from core.integrations.firestore.client import COLLECTION_NAME

    agent, ledger, executor, _repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch,
    )
    operation_id, conversation_id, _route, arguments, folder_id, _summary, _capability = seeded
    executor.client.collection(COLLECTION_NAME).document("doc-b").set({
        "doc_id": "doc-b", "meeting_id": "t016-meeting-b",
        "source_file": "T016 target B", "parent_id": "t016-agent-root",
        "status": "synced",
    })
    trace = list(sdk_fakes.trace)
    assert agent.dispatch_tool_choice(
        name="create_folder", arguments=arguments, target_doc_id=None,
        conversation_id=conversation_id, delivery_id=operation_id,
        executor_factory=lambda _target, _delivery, _context: executor,
    ) == folder_id
    assert sdk_fakes.trace == trace
    with pytest.raises(ValueError, match="persisted tool choice"):
        agent.dispatch_tool_choice(
            name="create_folder", arguments={"name": "Different"}, target_doc_id=None,
            conversation_id=conversation_id, delivery_id=operation_id,
            executor_factory=lambda _target, _delivery, _context: executor,
        )
    rejected_action = False
    try:
        agent.dispatch_tool_choice(
            name="move_transcript",
            arguments={"doc_id": "doc-b", "folder_path": ["Archive"]},
            target_doc_id="doc-b",
            conversation_id=conversation_id, delivery_id=operation_id,
            executor_factory=lambda _target, _delivery, _context: executor,
        )
    except ValueError as error:
        rejected_action = "persisted tool choice" in str(error)
    rejected_target = False
    try:
        agent.dispatch_tool_choice(
            name="create_folder", arguments=arguments, target_doc_id="doc-b",
            conversation_id=conversation_id, delivery_id=operation_id,
            executor_factory=lambda _target, _delivery, _context: executor,
        )
    except ValueError as error:
        rejected_target = "persisted tool choice" in str(error)
    new_drive_mutations = [
        row for row in sdk_fakes.trace[len(trace):]
        if row.get("boundary") == "drive"
        and row.get("operation") in {"files.create", "files.update"}
    ]
    assert not new_drive_mutations
    assert rejected_action, "a different registered action escaped the saved application plan"
    assert rejected_target, "a different fixed target escaped the saved application plan"
    assert agent.dispatch_tool_choice(
        name="create_folder", arguments=arguments, target_doc_id=None,
        conversation_id=conversation_id, delivery_id=operation_id,
        executor_factory=lambda _target, _delivery, _context: executor,
    ) == folder_id


def test_t016_saved_application_write_rejects_extra_native_ordinal_before_child_allocation(
    firestore_fake, sdk_fakes, monkeypatch,
):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from core.integrations.google_drive.folder_service import FolderNameConflict
    from core.integrations.firestore.client import COLLECTION_NAME
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    agent, ledger, executor, _repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch,
    )
    operation_id, conversation_id, _route, arguments, _folder_id, _summary, _capability = seeded
    executor.client.collection(COLLECTION_NAME).document("doc-b").set({
        "doc_id": "doc-b", "meeting_id": "t016-meeting-b",
        "source_file": "T016 target B", "parent_id": "t016-agent-root",
        "status": "synced",
    })

    trace_before_extra = list(sdk_fakes.trace)
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(
                id="t016-extra-ordinal", name="create_folder", args=arguments,
            ),
        )]))],
    ))
    agent.model = GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t016-extra-key"))
    result = None
    failure = None
    try:
        result = agent.run(
            conversation_id=conversation_id, delivery_id=operation_id,
            history=[{"role": "user", "parts": [{"text": "Create the folder Archive"}]}],
            target_doc_id=None,
        )
    except (ValueError, PermissionError, FolderNameConflict) as caught:
        failure = caught
    new_drive_mutations = [
        row for row in sdk_fakes.trace[len(trace_before_extra):]
        if row.get("boundary") == "drive"
        and row.get("operation") in {"files.create", "files.update"}
    ]
    assert not new_drive_mutations
    from core.integrations.firestore.operations import get_operation
    from core.integrations.firestore.mutations import get_mutation_effect
    from services.connectors.telegram.runtime.tools import stable_effect_id

    extra_child_id = "telegram-tool-" + hashlib.sha256(
        f"{operation_id}\0{1}".encode("utf-8")
    ).hexdigest()
    extra_child = get_operation(executor.client, extra_child_id)
    assert extra_child is None, (
        "the unapproved ordinal allocated a child before its saved-plan rejection; "
        f"result={result!r}, original_failure={failure!r}"
    )
    assert get_mutation_effect(
        executor.client, extra_child_id, stable_effect_id(operation_id, 1),
    ) is None
    if failure is None:
        assert result is not None and result["status"] in {"ok", "failed"}


def test_t016_governed_typed_execute_reconcile_and_cache_preserve_exact_binding(
    firestore_fake, sdk_fakes, monkeypatch,
):
    import inspect
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    from core.integrations.firestore.operation_queue import QueueOperationRetryStore
    from core.operations.retries import RetryGovernor, UnknownOutcome
    from core.operations.write import _new_resolved_write_targets
    from services.connectors.telegram.runtime.runtime import _GovernedExecutor
    from services.connectors.telegram.runtime.tools import stable_effect_id

    _agent, ledger, inner, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch,
    )
    operation_id, _conversation, _route, arguments, folder_id, _summary, capability = seeded
    context_ref = repository.client.collection("telegram_sessions").document(ledger.context_id)
    now = BASE_TIME + timedelta(minutes=9)
    governor = RetryGovernor(
        repository.client, clock=SimpleNamespace(now=lambda: now),
        retry_store=QueueOperationRetryStore(repository.client, context_ref),
    )
    governed = _GovernedExecutor(
        inner, governor, operation_id, SimpleNamespace(owner=ledger.owner, fence=ledger.fence),
        lambda: now, lambda: True,
    )
    assert "resolved_targets" in inspect.signature(governed.execute).parameters, (
        "governed execution must preserve the inner executor's trusted target capability"
    )
    trace = list(sdk_fakes.trace)
    assert governed.execute(
        name="create_folder", arguments=arguments, call_ordinal=0,
        resolved_targets=capability,
    ) == folder_id
    assert governed.execute(
        name="create_folder", arguments=arguments, call_ordinal=0,
        resolved_targets=capability,
    ) == folder_id
    assert sdk_fakes.trace == trace
    with pytest.raises(ValueError, match="binding changed"):
        governed.execute(
            name="create_folder", arguments=arguments, call_ordinal=0,
            resolved_targets=replace(capability, intent_digest="d" * 64),
        )
    assert sdk_fakes.trace == trace

    # Seed a real typed child through the governor, lose only its ACK, then
    # require the same typed binding for the governed reconciliation readback.
    root_id, reconciled_folder = "t016-agent-root", "t016-agent-reconciled-folder"
    child_id = "telegram-tool-" + hashlib.sha256(
        f"{operation_id}\0{1}".encode("utf-8")
    ).hexdigest()
    effect_id = stable_effect_id(operation_id, 1)
    arguments_1 = {"name": "Archive-Reconcile"}
    capability_1 = _new_resolved_write_targets(
        operation_name="create_folder", operation_id=child_id,
        effect_id=effect_id, arguments=arguments_1, parent_id=root_id,
        root_id=root_id, intent_digest="c" * 64,
    )
    allocator_payload = json.dumps(
        {"name": "Archive-Reconcile", "parent_id": root_id, "root_id": root_id,
         "intent_digest": "c" * 64},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    allocator_intent = "sha256:" + hashlib.sha256(allocator_payload.encode("utf-8")).hexdigest()
    marker = "v1:" + hashlib.sha256(
        "\0".join((child_id, effect_id, "drive_folder", allocator_intent)).encode("utf-8")
    ).hexdigest()
    sdk_fakes.drive.queue("files.generateIds", response={"ids": [reconciled_folder]})
    root_metadata = {"id": root_id, "name": "TLBrain",
                     "mimeType": "application/vnd.google-apps.folder",
                     "parents": [], "trashed": False}
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", error=_http_404())
    sdk_fakes.drive.queue("files.get", response=root_metadata)
    sdk_fakes.drive.queue("files.create", response={"id": reconciled_folder})
    sdk_fakes.drive.queue("files.get", response={
        "id": reconciled_folder, "name": "Archive-Reconcile",
        "mimeType": "application/vnd.google-apps.folder", "parents": [root_id],
        "trashed": False, "appProperties": {"tlbrain_effect_intent": marker},
    })

    def write_then_lose_ack():
        result = inner.execute(
            name="create_folder", arguments=arguments_1, call_ordinal=1,
            resolved_targets=capability_1,
        )
        assert result == reconciled_folder
        raise UnknownOutcome("synthetic ACK loss after committed typed child")

    with pytest.raises(UnknownOutcome):
        governor.run_stage(
            operation_id, "telegram-tool-01", write_then_lose_ack,
            now=now, owner=ledger.owner, fence=ledger.fence,
        )
    before_reconcile = list(sdk_fakes.trace)
    assert governed.execute(
        name="create_folder", arguments=arguments_1, call_ordinal=1,
        resolved_targets=capability_1,
    ) == reconciled_folder
    assert sdk_fakes.trace == before_reconcile


def _t016_save_real_native_round(sdk_fakes, ledger, conversation_id, operation_id):
    from google.genai import types
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.runtime.runtime import _GovernedModel
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    native_round = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(
            text="Previously accepted native round.",
            thought_signature=b"opaque-t016-signature",
        )]),
    )])
    sdk_fakes.gemini.queue("generate_content", response=native_round)
    serialized = _GovernedModel(
        GeminiAgent(client=FakeGeminiClient(api_key="synthetic-t016-native-key")),
        None, operation_id, None,
        lambda: BASE_TIME + timedelta(minutes=10), lambda: True,
    ).generate(
        contents=[{"role": "user", "parts": [{"text": "Create the folder Archive"}]}],
        mode="tool_loop",
    )
    model_rounds = {
        "_telegram_tool_loop_rounds": 1,
        "rounds": [{"response": serialized, "call_start": 0, "call_count": 0}],
    }
    ledger.save_turn(
        conversation_id, operation_id, model_response=model_rounds, tool_calls=[],
    )
    restored = ledger.load_turn(conversation_id, operation_id)
    assert restored["model_response"] == model_rounds
    assert restored["tool_calls"] == []
    return model_rounds


@pytest.mark.parametrize(
    "failure_boundary", [
        "checkpoint_commit", "lease_loss", "projection_conflict", "partial_multipart",
    ],
)
def test_t016_joint_publication_fault_preserves_native_round_and_neither_half(
    firestore_fake, sdk_fakes, monkeypatch, failure_boundary,
):
    from core.integrations.firestore.input_parts import MAX_PART_BYTES

    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, application_route, _arguments, _folder_id, summary, _capability = seeded
    accepted_context = ledger.context_id
    if failure_boundary == "partial_multipart":
        from dataclasses import replace
        from services.connectors.telegram.runtime.routing_context import RoutingDecision

        decision = RoutingDecision.from_dict(application_route["contextual"]["decision"])
        large_decision = replace(
            decision, substantive_text=decision.substantive_text + " "
            + "x" * (MAX_PART_BYTES + 8192),
        )
        application_route = {
            **application_route,
            "contextual": {
                **application_route["contextual"],
                "decision": large_decision.to_dict(),
            },
        }
    assert application_route["contextual"]["summary"] == summary
    assert repository._route_control_data(summary) == summary
    serialized_native_round = _t016_save_real_native_round(
        sdk_fakes, ledger, conversation_id, operation_id,
    )
    prepare = getattr(ledger, "prepare_application_route", None)
    assert callable(prepare), (
        "joint publication requires the planned trusted prepared application checkpoint"
    )
    partial_part_path = None
    if failure_boundary == "partial_multipart":
        route_digest = hashlib.sha256(ledger._json_bytes(application_route)).hexdigest()
        data_operation = "telegram-ledger-" + hashlib.sha256(
            f"{accepted_context}\0{operation_id}\0application-route\0{route_digest}".encode()
        ).hexdigest()
        partial_part_path = (
            f"dialog_operations/{data_operation}/input_parts/part-000001"
        )
        firestore_fake.fail_next(
            "commit", path=partial_part_path, phase="before",
            exception=RuntimeError("T016 partial application-route multipart fault"),
        )
        with pytest.raises(RuntimeError, match="partial application-route multipart fault"):
            prepare(
                conversation_id, operation_id, route=application_route,
                tool_calls=[{
                    "id": application_route["call_id"], "name": application_route["name"],
                    "arguments": application_route["arguments"], "ordinal": 0, "round": -1,
                }],
                application_route_round=None,
            )
        assert firestore_fake.documents.get(
            partial_part_path.rsplit("/", 1)[0] + "/part-000000",
        ) is not None
        assert partial_part_path not in firestore_fake.documents
        assert partial_part_path.rsplit("/", 1)[0] + "/manifest" not in firestore_fake.documents
        assert any(
            row.get("path") == partial_part_path and row.get("operation") == "commit"
            and row.get("phase") == "before"
            and row.get("outcome") == "failed"
            for row in firestore_fake.trace()
        ), "partial multipart commit fault was not reached"
        assert repository.get_route_control(accepted_context, operation_id) is None
        restored = ledger.load_turn(conversation_id, operation_id)
        assert restored["application_route"] is None
        assert restored["model_response"] == serialized_native_round
        assert restored["model_response"]["rounds"][0]["call_start"] == 0
        assert restored["model_response"]["rounds"][0]["call_count"] == 0
        return
    prepared = prepare(
        conversation_id, operation_id, route=application_route,
        tool_calls=[{
            "id": application_route["call_id"], "name": application_route["name"],
            "arguments": application_route["arguments"], "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    checkpoint_path = (
        f"telegram_sessions/{accepted_context}/operations/{operation_id}"
        "/agent_checkpoints/turn"
    )
    if failure_boundary == "checkpoint_commit":
        firestore_fake.fail_next(
            "commit", path=checkpoint_path, phase="before",
            exception=RuntimeError("T016 joint publication checkpoint fault"),
        )
        with pytest.raises(RuntimeError, match="joint publication checkpoint fault"):
            repository.persist_route(
                operation_id, summary,
                context_id=accepted_context, owner=ledger.owner, fence=ledger.fence,
                now=BASE_TIME + timedelta(minutes=10), application_checkpoint=prepared,
            )
    elif failure_boundary == "lease_loss":
        from core.integrations.firestore.operation_queue import release_head

        context_ref = repository.client.collection("telegram_sessions").document(
            accepted_context,
        )
        assert release_head(
            repository.client, context_ref, operation_id=operation_id,
            owner=ledger.owner, fence=ledger.fence,
            now=BASE_TIME + timedelta(minutes=10),
        )
        try:
            published = repository.persist_route(
                operation_id, summary,
                context_id=accepted_context, owner=ledger.owner, fence=ledger.fence,
                now=BASE_TIME + timedelta(minutes=10), application_checkpoint=prepared,
            )
        except (PermissionError, ValueError):
            published = False
        assert published is False, "lost queue lease published a contextual route pair"
    else:
        mismatched_summary = {
            **summary,
            "tool_choice": {"name": "create_folder", "arguments": {"name": "Other"}},
        }
        try:
            repository.persist_route(
                operation_id, mismatched_summary,
                context_id=accepted_context, owner=ledger.owner, fence=ledger.fence,
                now=BASE_TIME + timedelta(minutes=10), application_checkpoint=prepared,
            )
        except (ValueError, PermissionError):
            pass
        else:
            assert repository.get_route_control(accepted_context, operation_id) is None
    assert repository.get_route_control(accepted_context, operation_id) is None
    restored = ledger.load_turn(conversation_id, operation_id)
    assert restored["application_route"] is None
    assert restored["tool_calls"] == []
    assert restored["model_response"] == serialized_native_round
    assert restored["model_response"]["rounds"][0]["call_start"] == 0
    assert restored["model_response"]["rounds"][0]["call_count"] == 0
    assert restored["model_response"]["rounds"][0]["response"]["candidates"][0][
        "content"]["parts"][0]["thought_signature"] == "b3BhcXVlLXQwMTYtc2lnbmF0dXJl"


def test_t016_joint_publication_success_retains_native_round_and_full_pin(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    native = _t016_save_real_native_round(sdk_fakes, ledger, conversation_id, operation_id)
    prepare = getattr(ledger, "prepare_application_route", None)
    assert callable(prepare), (
        "joint publication requires the planned trusted prepared application checkpoint"
    )
    prepared = prepare(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    context_id = ledger.context_id
    application_calls = [{
        "id": route["call_id"], "name": route["name"],
        "arguments": arguments, "ordinal": 0, "round": -1,
    }]
    assert repository.persist_route(
        operation_id, summary, context_id=context_id,
        owner=ledger.owner, fence=ledger.fence,
        now=BASE_TIME + timedelta(minutes=10), application_checkpoint=prepared,
    )
    published_summary = repository.get_route_control(context_id, operation_id)
    restored = ledger.load_turn(conversation_id, operation_id)
    assert published_summary == summary
    assert restored["application_route"] == route
    assert restored["application_route_round"] is None
    assert restored["model_response"] == native
    assert restored["tool_calls"] == application_calls
    assert restored.get("tool_results", {}) == {}
    native_round = restored["model_response"]["rounds"][0]
    assert native_round["call_start"] == 0 and native_round["call_count"] == 0
    assert native_round["response"]["candidates"][0]["content"]["parts"][0][
        "thought_signature"
    ] == "b3BhcXVlLXQwMTYtc2lnbmF0dXJl"
    assert route["contextual"]["checkpoint_conversation_id"] == conversation_id
    assert route["contextual"]["source_conversation_id"] != conversation_id
    ledger.save_turn(
        conversation_id, operation_id, model_response=native,
        tool_calls=application_calls,
    )
    after_native_round = ledger.load_turn(conversation_id, operation_id)
    assert after_native_round["application_route"] == route
    assert after_native_round["model_response"] == native
    assert after_native_round["tool_calls"] == application_calls
    assert after_native_round["model_response"]["rounds"][0]["call_count"] == 0


def test_t016_legacy_summary_pin_is_never_enriched_by_prepared_route(
    firestore_fake, sdk_fakes, monkeypatch,
):
    _agent, ledger, _executor, repository, seeded = _t016_seed_real_folder_application(
        firestore_fake, sdk_fakes, monkeypatch, seed_completed=False,
    )
    operation_id, conversation_id, route, arguments, _folder, summary, _capability = seeded
    legacy_pin = {"status": "ok", "kind": "substantive", "target_doc_id": "doc-b"}
    assert repository.persist_route(
        operation_id, legacy_pin, context_id=ledger.context_id,
        owner=ledger.owner, fence=ledger.fence, now=BASE_TIME + timedelta(minutes=10),
    )
    prepare = getattr(ledger, "prepare_application_route", None)
    assert callable(prepare), (
        "legacy pins require the planned trusted prepared application checkpoint"
    )
    prepared = prepare(
        conversation_id, operation_id, route=route,
        tool_calls=[{
            "id": route["call_id"], "name": route["name"],
            "arguments": arguments, "ordinal": 0, "round": -1,
        }],
        application_route_round=None,
    )
    try:
        published = repository.persist_route(
            operation_id, summary, context_id=ledger.context_id,
            owner=ledger.owner, fence=ledger.fence,
            now=BASE_TIME + timedelta(minutes=10), application_checkpoint=prepared,
        )
    except (ValueError, PermissionError):
        published = False
    assert published is False
    assert repository.get_route_control(ledger.context_id, operation_id) == legacy_pin
    legacy_turn = ledger.load_turn(conversation_id, operation_id)
    assert legacy_turn is None or legacy_turn.get("application_route") is None
