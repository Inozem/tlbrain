"""Focused acceptance checks for the contextual Telegram fixture builders."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json

import pytest

from tests.fixtures.telegram_contextual import (
    BASE_TIME,
    CORPUS_SIZES,
    EnumerationBlocked,
    capture_sdk_calls,
    guard_firestore_enumeration,
    make_contextual_scenario,
    make_transcript_corpus,
    script_provider_outcome,
    seed_history,
    seed_transcript_corpus,
)


def test_scenario_is_deterministic_isolated_and_keeps_display_order_separate():
    first = make_contextual_scenario(unrelated_count=1)
    second = make_contextual_scenario(unrelated_count=1)

    assert first.context_id == "42017_73"
    assert first.context_id == second.context_id
    assert first.selected_dialog_id == "doc-a"
    assert first.explicit_dialog_id == "doc-b"
    assert first.local_dialog_number_2_id == "doc-c"
    assert first.retrieval_order == ("doc-b", "doc-a", "doc-c")
    assert first.shown_order == ("doc-c", "doc-b", "doc-a")
    assert first.documents["doc-b"]["automatic_title"] == "Planning notes"
    assert "price" not in first.documents["doc-b"]["automatic_title"].lower()
    assert "€42" in first.exchanges[0]["result"]["segments"][0]["quote"]
    assert "Ignore prior rules" in first.exchanges[1]["result"]["segments"][0]["quote"]
    assert first.prior_list_id != first.new_list_id
    assert first.prior_list["list_id"] == first.prior_list_id
    assert first.new_list["list_id"] == first.new_list_id
    assert first.prior_list["items"] != first.new_list["items"]
    assert first.active_snapshot() == second.active_snapshot()

    first.documents["doc-a"]["transcript"] = "mutated"
    assert second.documents["doc-a"]["transcript"] != "mutated"
    assert first.topic.thread_id == 73

    from tests.fixtures.telegram_contextual import actor, chat, topic
    customized = make_contextual_scenario(
        actor_value=actor(user_id=18),
        chat_value=chat(chat_id=42018), topic_value=topic(thread_id=74),
        unrelated_count=1,
    )
    assert customized.actor.actor_id == "actor-18"
    assert customized.actor.user_id == 18
    assert customized.context_id == "42018_74"


def test_numeric_actor_is_accepted_by_real_repository_intake(firestore_fake):
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository

    scenario = make_contextual_scenario(unrelated_count=0)
    repository = TelegramRepository(client=firestore_fake.client_class())
    accepted = repository.accept_command(
        901, chat_id=scenario.chat.chat_id, user_id=scenario.actor.user_id,
        thread_id=scenario.topic.thread_id, command=parse_command("prices"), now=BASE_TIME,
    )
    assert accepted.context_id == scenario.context_id


def test_corpus_count_changes_only_unrelated_rows_and_sizes_are_available():
    variants = [make_contextual_scenario(unrelated_count=count) for count in CORPUS_SIZES]
    assert [len(item.unrelated_transcripts) for item in variants] == [1, 324, 10_000]
    assert variants[0].active_snapshot() == variants[1].active_snapshot() == variants[2].active_snapshot()
    assert len(make_transcript_corpus(10_000)) == 10_000
    assert make_transcript_corpus(2) == make_transcript_corpus(2)


def test_canonical_corpus_seed_is_visible_to_real_transcript_and_store_interfaces(firestore_fake):
    from core.integrations.firestore.client import COLLECTION_NAME, FOLDERS_COLLECTION
    from core.integrations.firestore.transcripts import get_transcript_record
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.store import TelegramRuntimeStore

    scenario = make_contextual_scenario(unrelated_count=1)
    client = firestore_fake.client_class()
    client.collection(FOLDERS_COLLECTION).document("root").set({"name": "TLBrain"})
    rows = seed_transcript_corpus(client, 3)
    repository = TelegramRepository(client=client)
    store = TelegramRuntimeStore(repository, root_folder_id="root", context_id=scenario.context_id)
    candidates = {row["doc_id"]: row for row in store.existing_dialog_candidates()}

    assert all(client.collection(COLLECTION_NAME).document(row["doc_id"]).get().exists for row in rows)
    record = get_transcript_record(rows[0]["doc_id"], db=client)
    assert record["parent_id"] == "root"
    assert record["status"] == "synced"
    assert set(row["doc_id"] for row in rows) <= candidates.keys()
    assert candidates[rows[0]["doc_id"]]["title"] == rows[0]["source_file"]
    assert scenario.active_snapshot() == make_contextual_scenario(unrelated_count=3).active_snapshot()


def test_history_and_large_unicode_exchange_roundtrip_with_real_repository_and_ledger(firestore_fake):
    client = firestore_fake.client_class()
    source = "Оpaque 🧭 evidence: €42; 東京 — complete ordered utterance.\n" * 500
    exchanges = ({
        "call": {"id": "large-call", "name": "query", "arguments": {"query": "price"},
                 "ordinal": 0, "round": 0, "thought_signature": bytes(range(256)) * 20},
        "result": {"segments": [{"doc_id": "doc-b", "order_index": 2, "text": source}],
                   "total_matches": 1, "returned_segments": 1, "truncated": False},
        "source_refs": [{"doc_id": "doc-b", "order_index": 2}],
    },)
    seeded = seed_history(client, context_id="42017_73", now=BASE_TIME, exchanges=exchanges)
    reference = seeded["references"][0]

    # Simulate a process restart by creating fresh repository and ledger objects.
    from services.connectors.telegram.repository import TelegramRepository
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    repository = TelegramRepository(client=client)
    ledger = FirestoreAgentLedger(
        client, context_id="42017_73", operation_id=seeded["operation_id"],
        owner=seeded["owner"], fence=seeded["fence"], clock=lambda: BASE_TIME,
    )
    conversation = repository.get_conversation("42017_73", seeded["conversation_id"])
    turns = repository.list_turns("42017_73", seeded["conversation_id"])
    exchange = ledger.load_search_exchange(
        reference, conversation_id=seeded["conversation_id"],
        operation_id=seeded["operation_id"],
        checkpoint_conversation_id="private-search-checkpoint",
        call_id="large-call", ordinal=0,
    )

    assert conversation.conversation_id == seeded["conversation_id"]
    assert [turn["role"] for turn in turns] == ["user", "tool", "model"]
    assert [turn["turn_sequence"] for turn in turns] == [1, 2, 3]
    assert exchange["call"]["thought_signature"] == bytes(range(256)) * 20
    assert exchange["result"]["segments"][0]["text"] == source
    assert exchange["result"]["segments"][0]["order_index"] == 2

    from services.connectors.telegram.context import build_conversation_history
    history = build_conversation_history(
        turns, conversation_id=seeded["conversation_id"],
        search_exchange_loader=ledger.load_search_exchange,
    )
    assert [turn["role"] for turn in history] == ["user", "user", "model"]
    assert history[0]["parts"][0]["text"] == "What did we decide about the quoted price?"
    hydrated_text = history[1]["parts"][0]["text"]
    tool_projection = json.loads(hydrated_text.removeprefix("Untrusted prior tool data: "))
    exchange_text = tool_projection["parts"][0]["text"]
    hydrated_exchange = json.loads(
        exchange_text.removeprefix("Untrusted prior search tool exchange data: ")
    )
    assert hydrated_exchange["result"]["segments"][0]["text"] == source
    assert history[2]["parts"][0]["text"] == "The verified quote was €42 per month."

    with pytest.raises(ValueError, match="binding"):
        ledger.load_search_exchange(
            reference, conversation_id="other-conversation",
            operation_id=seeded["operation_id"],
            checkpoint_conversation_id="private-search-checkpoint",
            call_id="large-call", ordinal=0,
        )
    with pytest.raises(ValueError, match="binding"):
        FirestoreAgentLedger(
            client, context_id="other-context", operation_id=seeded["operation_id"],
            owner=seeded["owner"], fence=seeded["fence"], clock=lambda: BASE_TIME,
        ).load_search_exchange(
            reference, conversation_id=seeded["conversation_id"],
            operation_id=seeded["operation_id"],
            checkpoint_conversation_id="private-search-checkpoint",
            call_id="large-call", ordinal=0,
        )
    for binding in (
        {"operation_id": "other-operation"},
        {"call_id": "different-call"},
        {"ordinal": 1},
    ):
        arguments = {
            "conversation_id": seeded["conversation_id"],
            "operation_id": seeded["operation_id"],
            "checkpoint_conversation_id": "private-search-checkpoint",
            "call_id": "large-call", "ordinal": 0,
        }
        arguments.update(binding)
        with pytest.raises(ValueError, match="binding"):
            ledger.load_search_exchange(reference, **arguments)


def test_history_builder_supports_empty_history_custom_scope_and_rejects_mismatch(firestore_fake):
    from services.connectors.telegram.models import build_context_id

    client = firestore_fake.client_class()
    default_topic_context = build_context_id(42019, None)
    empty = seed_history(
        client, context_id=default_topic_context, operation_id="op-default-topic",
        conversation_id="empty-search", exchanges=(), turns=(),
        chat_id=42019, thread_id=None,
    )
    assert empty["chat_id"] == 42019 and empty["thread_id"] is None
    assert empty["repository"].list_turns(default_topic_context, "empty-search") == []
    configured_context = build_context_id(42020, None)
    configured = seed_history(
        client, context_id=configured_context, operation_id="op-configured-history",
        conversation_id="configured-search", exchanges=(), turns=(
            {"role": "user", "parts": [{"text": "Custom question 🛰️"}]},
            {"role": "model", "parts": [{"text": "Custom answer €9"}]},
        ), chat_id=42020, thread_id=None,
    )
    configured_turns = configured["repository"].list_turns(
        configured_context, "configured-search",
    )
    assert [turn["parts"][0]["text"] for turn in configured_turns] == [
        "Custom question 🛰️", "Custom answer €9",
    ]
    with pytest.raises(ValueError, match="does not match"):
        seed_history(client, context_id="42019_default", chat_id=42019,
                     thread_id=73, operation_id="op-wrong-scope")

    conflict = client.collection("telegram_sessions").document(default_topic_context)
    conflict.update({"chat_id": 999})
    with pytest.raises(ValueError, match="conflicting transport identity"):
        seed_history(client, context_id=default_topic_context,
                     operation_id="op-conflicting-scope", chat_id=42019, thread_id=None)


def test_history_context_isolation_and_stale_lease_rejection(firestore_fake):
    client = firestore_fake.client_class()
    first = seed_history(client, context_id="42017_73", now=BASE_TIME,
                         exchanges=make_contextual_scenario(unrelated_count=0).exchanges[:1])
    second = seed_history(client, context_id="42018_73", now=BASE_TIME,
                          operation_id="op-contextual-other",
                          conversation_id="other-search", exchanges=())
    first_exchange = first["ledger"].load_search_exchange(
        first["references"][0], conversation_id=first["conversation_id"],
        operation_id=first["operation_id"],
        checkpoint_conversation_id="private-search-checkpoint",
        call_id="call-prices", ordinal=0,
    )
    assert first_exchange["context_id"] == "42017_73"
    assert second["repository"].get_conversation("42018_73", "other-search") is not None
    assert first["repository"].get_conversation("42017_73", "other-search") is None

    client.collection("telegram_sessions").document("42017_73").update({"fence": 2})
    with pytest.raises(PermissionError, match="stale"):
        first["ledger"].save_search_exchange(
            first["conversation_id"], checkpoint_conversation_id="private-search-checkpoint",
            call={"id": "next-call", "name": "query", "arguments": {"query": "x"},
                  "ordinal": 1, "round": 1}, canonical_call_id="next-call",
            result={"text": "large synthetic history " * 500},
        )


def test_sdk_capture_records_real_agent_contents_and_function_schema_as_copied(sdk_fakes):
    from core.integrations.gemini.agent import GeminiAgent
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    sdk_fakes.gemini.queue("generate_content", response={"text": "synthetic response"})
    contents = [{"role": "user", "parts": [{"text": "actual prompt"}]}]
    declaration = {
        "name": "query", "description": "Search verified transcripts",
        "parameters_json_schema": {
            "type": "OBJECT", "properties": {"query": {"type": "STRING"}},
            "required": ["query"],
        },
    }
    with capture_sdk_calls() as calls:
        client = FakeGeminiClient(api_key="pytest-synthetic-key")
        GeminiAgent(client=client).generate(contents=contents, mode="routing", tools=[declaration])
        contents[0]["parts"][0]["text"] = "caller mutation"
        declaration["parameters_json_schema"]["properties"]["query"]["type"] = "BOOLEAN"

    event = next(row for row in calls if row["operation"] == "generate_content")
    assert event["arguments"]["contents"][0]["parts"][0]["text"] == "actual prompt"
    tool = event["arguments"]["config"].tools[0]
    native_declaration = tool.function_declarations[0]
    assert native_declaration.name == "query"
    assert native_declaration.parameters_json_schema["properties"]["query"]["type"] == "STRING"


def test_firestore_enumeration_guard_is_opt_in_and_allows_targeted_get(firestore_fake):
    from core.integrations.firestore.client import COLLECTION_NAME

    client = firestore_fake.client_class()
    client.collection(COLLECTION_NAME).document("doc-a").set({"title": "A"})
    with guard_firestore_enumeration() as events:
        assert client.collection(COLLECTION_NAME).document("doc-a").get().to_dict()["title"] == "A"
        with pytest.raises(EnumerationBlocked, match=COLLECTION_NAME):
            client.collection(COLLECTION_NAME).where("parent_id", "==", "root").select(
                ["source_file", "parent_id"],
            ).stream()
        with pytest.raises(EnumerationBlocked, match="dialogs"):
            client.collection("telegram_sessions").document("42017_73").collection("dialogs").stream()
        with pytest.raises(EnumerationBlocked, match="telegram_sessions"):
            client.collection("telegram_sessions").stream()
    assert events[0]["collection"] == COLLECTION_NAME
    assert events[0]["filters"] == (("parent_id", "==", "root"),)

    with pytest.raises(RuntimeError, match="synthetic exit"):
        with guard_firestore_enumeration():
            client.collection("safe_collection").stream()
            raise RuntimeError("synthetic exit")
    assert len(client.collection(COLLECTION_NAME).stream()) == 1

    with guard_firestore_enumeration(allow=True) as search_events:
        rows = client.collection(COLLECTION_NAME).where("title", "==", "A").stream()
    assert len(rows) == 1
    assert search_events[0]["filters"] == (("title", "==", "A"),)

    with guard_firestore_enumeration() as outer:
        with guard_firestore_enumeration() as inner:
            with pytest.raises(EnumerationBlocked):
                client.collection(COLLECTION_NAME).where("title", "==", "A").stream()
        with pytest.raises(EnumerationBlocked):
            client.collection(COLLECTION_NAME).stream()
    assert outer and inner


def test_enumeration_guard_allows_scoped_history_list_and_real_exchange_hydration(firestore_fake):
    from services.connectors.telegram.context import build_conversation_history

    client = firestore_fake.client_class()
    seeded = seed_history(client, context_id="42017_73", now=BASE_TIME)
    repository = seeded["repository"]
    with guard_firestore_enumeration() as events:
        turns = repository.list_turns("42017_73", seeded["conversation_id"])
        history = build_conversation_history(
            turns, conversation_id=seeded["conversation_id"],
            search_exchange_loader=seeded["ledger"].load_search_exchange,
        )

    assert [turn["role"] for turn in turns] == ["user", "tool", "tool", "model"]
    assert [turn["role"] for turn in history] == ["user", "user", "user", "model"]
    assert len(events) == 1
    assert events[0]["collection"].endswith("/turns")


@pytest.mark.parametrize("outcome", ["complete", "lost", "partial"])
def test_provider_outcome_scripts_use_sdk_boundaries_and_distinct_readbacks(
    firestore_fake, sdk_fakes, outcome,
):
    from googleapiclient.discovery import build
    from core.integrations.firestore.client import COLLECTION_NAME

    client = firestore_fake.client_class()
    registry_path = f"{COLLECTION_NAME}/doc-a"
    client.collection(COLLECTION_NAME).document("doc-a").set({"parent_id": "nested-a"})
    before_trace = len(sdk_fakes.trace)
    expected = script_provider_outcome(outcome, registry_path=registry_path)
    drive = build("drive", "v3", credentials=None, cache_discovery=False)
    update_request = drive.files().update(
        fileId="doc-a", addParents="archive", removeParents="nested-a", fields="id,parents",
    )
    if outcome == "lost":
        with pytest.raises(TimeoutError, match="response lost"):
            update_request.execute()
    else:
        assert update_request.execute()["parents"] == ["archive"]
    readback = drive.files().get(fileId="doc-a", fields="id,name,parents,trashed").execute()
    assert readback == expected["readback"]

    registry = client.collection(COLLECTION_NAME).document("doc-a")
    if outcome == "partial":
        with pytest.raises(RuntimeError, match="registry write failed"):
            registry.set({"parent_id": "archive"}, merge=True)
        assert registry.get().to_dict()["parent_id"] == "nested-a"
    else:
        registry.set({"parent_id": "archive"}, merge=True)
        assert registry.get().to_dict()["parent_id"] == "archive"

    trace = sdk_fakes.trace[before_trace:]
    assert [row["operation"] for row in trace
            if row["boundary"] in {"drive", "docs"} and row["operation"] != "build"] == [
        "files.update", "files.get",
    ]
    update_event = next(row for row in trace if row["operation"] == "files.update")
    assert update_event["arguments"]["addParents"] == "archive"
    assert expected["drive_parent"] == "archive"
    assert expected["registry_parent"] == ("nested-a" if outcome == "partial" else "archive")
