from datetime import datetime, timezone

import pytest

from services.connectors.telegram.repository import TelegramRepository
from services.connectors.telegram.search_sessions import run_search, select_search_result


NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)


@pytest.fixture
def repository(firestore_fake):
    return TelegramRepository(firestore_fake.client_class())


def test_search_passes_all_filters_and_preserves_retrieval_metadata(repository):
    filters = {
        "query": "leadership notes",
        "keywords": "strategy OR hiring",
        "folder_path": ["Team", "Leadership"],
        "date_from": "2026-01-01",
        "date_to": "2026-06-30",
    }
    metadata = {
        "truncated": True,
        "total_matches": 5,
        "returned_segments": 1,
        "limit_reason": "result_limit",
        "suggestion": "Narrow the date range",
    }
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return ([{
            "doc_id": "doc-a", "folder_path": "Team/Leadership",
            "dialog_date": "2026-04-02", "segments": [],
        }], metadata)

    title_calls = []

    def title_lookup(doc_ids):
        title_calls.append(doc_ids)
        return {"doc-a": "Planning.txt"}

    result = run_search(
        context_id="ctx-1", operation_id="operation-1", repository=repository,
        now=NOW, retrieve=retrieve, title_lookup=title_lookup,
        **filters,
    )

    assert calls == [filters]
    assert title_calls == [["doc-a"]]
    assert result["metadata"] == metadata
    assert result["status"] == "ok"
    assert result["results"] == [{
        "doc_id": "doc-a", "title": "Planning.txt", "date": "2026-04-02",
        "folder_path": "Team/Leadership",
    }]
    conversation = repository.get_conversation("ctx-1", result["conversation_id"])
    assert conversation.kind.value == "search"
    assert conversation.search_filters == filters
    assert conversation.draft_id is conversation.doc_id is conversation.dialog_number is None


def test_empty_search_is_distinct_from_retrieval_failure(repository):
    failed_calls = []
    title_calls = []

    def fail_retrieval(**kwargs):
        failed_calls.append(kwargs)
        raise RuntimeError("private backend detail")

    no_results = run_search(
        context_id="ctx-2", operation_id="empty", query="no match", repository=repository,
        now=NOW, retrieve=lambda **_: ([], {"truncated": False, "total_matches": 0,
                                            "returned_segments": 0}),
        title_lookup=lambda doc_ids: title_calls.append(doc_ids) or {},
    )
    failed = run_search(
        context_id="ctx-2", operation_id="failed", query="broken", repository=repository,
        now=NOW, folder_path=["Archive"], date_from="2025-01-01",
        retrieve=fail_retrieval,
    )
    malformed_metadata = run_search(
        context_id="ctx-2", operation_id="bad-metadata", query="bad metadata",
        repository=repository, now=NOW,
        retrieve=lambda **_: ([{"doc_id": "doc-a"}], "private malformed metadata"),
        title_lookup=lambda _doc_ids: {},
    )

    assert no_results["status"] == "no_results"
    assert title_calls == []
    assert no_results["metadata"]["total_matches"] == 0
    assert failed["status"] == "error"
    assert failed["error"] == "Search unavailable."
    assert failed["metadata"] is None
    assert malformed_metadata["status"] == "error"
    assert malformed_metadata["error"] == "Search unavailable."
    assert malformed_metadata["metadata"] is None
    malformed_turn = next(
        turn for turn in repository.list_turns("ctx-2", malformed_metadata["conversation_id"])
        if turn["role"] == "tool"
    )
    assert malformed_turn["parts"] == [{
        "type": "search_result", "status": "error", "results": [],
        "metadata": None, "error": "Search unavailable.",
    }]
    assert "Archive" not in str(failed["error"])
    assert failed_calls == [{
        "query": "broken", "keywords": None, "folder_path": ["Archive"],
        "date_from": "2025-01-01", "date_to": None,
    }]
    assert repository.get_conversation("ctx-2", failed["conversation_id"]).search_filters == {
        "query": "broken", "keywords": None, "folder_path": ["Archive"],
        "date_from": "2025-01-01", "date_to": None,
    }


def test_operations_create_separate_sessions_and_replay_is_idempotent(repository):
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [], {"truncated": False, "total_matches": 0, "returned_segments": 0}

    first = run_search(
        context_id="ctx-3", operation_id="operation-a", query="one", repository=repository,
        now=NOW, retrieve=retrieve, title_lookup=lambda _doc_ids: {},
    )
    replay = run_search(
        context_id="ctx-3", operation_id="operation-a", query="one", repository=repository,
        now=NOW, retrieve=retrieve, title_lookup=lambda _doc_ids: {},
    )
    second = run_search(
        context_id="ctx-3", operation_id="operation-b", query="two", repository=repository,
        now=NOW, retrieve=retrieve, title_lookup=lambda _doc_ids: {},
    )

    assert first == replay
    assert first["conversation_id"] != second["conversation_id"]
    assert len(calls) == 2
    assert len(repository.list_turns("ctx-3", first["conversation_id"])) == 2
    assert len(repository.list_turns("ctx-3", second["conversation_id"])) == 2
    assert {turn["operation_id"] for turn in repository.list_turns(
        "ctx-3", first["conversation_id"]
    )} == {"operation-a"}


def test_search_cards_do_not_bind_until_selection_and_selection_is_session_scoped(repository):
    result = run_search(
        context_id="ctx-4", operation_id="operation", query="find", repository=repository,
        now=NOW,
        retrieve=lambda **_: ([{
            "doc_id": "doc-unbound", "folder_path": "Notes/Meetings",
            "dialog_date": "2026-03-05", "segments": [],
        }], {"truncated": False, "total_matches": 1, "returned_segments": 1}),
        title_lookup=lambda doc_ids: {doc_id: "Meeting notes" for doc_id in doc_ids},
    )
    session_ref = repository.client.collection("telegram_sessions").document("ctx-4")

    assert result["results"][0] == {
        "doc_id": "doc-unbound", "title": "Meeting notes", "date": "2026-03-05",
        "folder_path": "Notes/Meetings",
    }
    assert list(session_ref.collection("document_bindings").stream()) == []
    assert list(session_ref.collection("dialogs").stream()) == []
    with pytest.raises(ValueError, match="not a result"):
        select_search_result(
            context_id="ctx-4", conversation_id=result["conversation_id"],
            doc_id="other-doc", repository=repository, now=NOW,
        )

    selected = select_search_result(
        context_id="ctx-4", conversation_id=result["conversation_id"],
        doc_id="doc-unbound", repository=repository, now=NOW,
    )
    replay = select_search_result(
        context_id="ctx-4", conversation_id=result["conversation_id"],
        doc_id="doc-unbound", repository=repository, now=NOW,
    )

    assert replay == selected
    assert selected["dialog_number"] == 1
    assert repository.get_conversation("ctx-4", selected["conversation_id"]).kind.value == "document"
    assert len(list(session_ref.collection("document_bindings").stream())) == 1
    assert len(list(session_ref.collection("dialogs").stream())) == 1
    assert list(session_ref.collection("drafts").stream()) == []


def test_selection_fails_closed_on_malformed_persisted_result_rows(repository):
    result = run_search(
        context_id="ctx-5", operation_id="operation", query="find", repository=repository,
        now=NOW,
        retrieve=lambda **_: ([{"doc_id": "doc-a", "folder_path": "Notes"}], {
            "truncated": False, "total_matches": 1, "returned_segments": 1,
        }),
        title_lookup=lambda doc_ids: {"doc-a": "Notes.txt"},
    )
    tool_turn = next(
        turn for turn in repository.list_turns("ctx-5", result["conversation_id"])
        if turn["role"] == "tool"
    )
    repository.client.collection("telegram_sessions").document("ctx-5").collection(
        "conversations"
    ).document(result["conversation_id"]).collection("turns").document(
        tool_turn["turn_id"]
    ).update({"parts": [{"type": "search_result", "results": [None, "malformed"]}]})

    with pytest.raises(ValueError, match="not a result"):
        select_search_result(
            context_id="ctx-5", conversation_id=result["conversation_id"],
            doc_id="doc-a", repository=repository, now=NOW,
        )
