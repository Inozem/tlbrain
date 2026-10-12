"""Offline behavior baselines for retrieval against the existing Qdrant schema."""

from google.genai import types
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import Record, ScoredPoint

from core.retrieval import pipeline, run, search, transcripts


ROOT_ID = "pytest-synthetic-root"
COLLECTION = "tldv_vectors"


def _embedding(sdk_fakes, values=None):
    sdk_fakes.gemini.queue(
        "embed_content",
        response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=values or [0.25] * 768)]
        ),
    )


def _query_response(points=()):
    return QueryResponse(points=list(points))


def _point(doc_id, score, *, point_id=1, parent_id="folder-child", **payload):
    return ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={"doc_id": doc_id, "parent_id": parent_id, **payload},
    )


def _query_events(sdk_fakes):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == "query_points"]


def _must_by_key(qdrant_filter):
    return {condition.key: condition for condition in qdrant_filter.must}


def _seed_folders(firestore_fake):
    client = firestore_fake.client_class()
    folders = client.collection("folders")
    folders.document("folder-team").set({"name": "Team", "parent_id": ROOT_ID})
    folders.document("folder-child").set({"name": "Child", "parent_id": "folder-team"})


def test_search_operations_keep_separate_types_and_folder_date_filters(sdk_fakes, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "23")
    sdk_fakes.sparse.set_vectors("review plan", [3, 8], [0.2, 0.7])
    _embedding(sdk_fakes)
    sdk_fakes.qdrant.queue("query_points", response=_query_response())
    search.search_summaries_and_facts(
        "review", folder_ids=["folder-team", "folder-child"],
        date_from="2026-01-02", date_to="2026-02-03", top_k=23,
    )
    sdk_fakes.qdrant.queue("query_points", response=_query_response())
    search.keyword_search_utterances(
        "review plan", folder_ids=["folder-team", "folder-child"],
        date_from="2026-01-02", date_to="2026-02-03",
    )
    _embedding(sdk_fakes)
    sdk_fakes.qdrant.queue("query_points", response=_query_response())
    search.search_user_facts(
        "review", folder_ids=["folder-team", "folder-child"],
        date_from="2026-01-02", date_to="2026-02-03",
    )
    _embedding(sdk_fakes)
    sdk_fakes.qdrant.queue("query_points", response=_query_response())
    search.search_summaries_for_doc("doc-pinned", "review", top_k=5)

    events = _query_events(sdk_fakes)
    assert [event["using"] for event in events] == ["dense", "bm25", "dense", "dense"]
    assert [event["limit"] for event in events] == [23, 10, 10, 5]
    semantic, keyword, user_fact, per_doc = events

    semantic_must = _must_by_key(semantic["query_filter"])
    assert semantic_must["type"].match.any == ["summary", "fact"]
    assert semantic_must["root_folder_id"].match.value == ROOT_ID
    assert semantic_must["parent_id"].match.any == ["folder-team", "folder-child"]
    assert semantic_must["dialog_date_num"].range.gte == 20260102
    assert semantic_must["dialog_date_num"].range.lte == 20260203
    assert keyword["query"].indices == [3, 8]
    assert keyword["query"].values == [0.2, 0.7]
    assert _must_by_key(keyword["query_filter"])["type"].match.value == "utterance"
    assert _must_by_key(user_fact["query_filter"])["type"].match.value == "user_fact"
    assert _must_by_key(user_fact["query_filter"])["parent_id"].match.any == ["folder-team", "folder-child"]
    assert _must_by_key(user_fact["query_filter"])["dialog_date_num"].range.gte == 20260102
    assert _must_by_key(user_fact["query_filter"])["dialog_date_num"].range.lte == 20260203
    assert "parent_id" not in _must_by_key(per_doc["query_filter"])
    assert "dialog_date_num" not in _must_by_key(per_doc["query_filter"])
    assert _must_by_key(per_doc["query_filter"])["doc_id"].match.value == "doc-pinned"
    assert all(event["collection_name"] == COLLECTION and event["with_payload"] for event in events)


def test_retrieval_expands_folder_filters_merges_ranges_and_caps_documents(
    sdk_fakes, firestore_fake, monkeypatch
):
    _seed_folders(firestore_fake)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    for query in ("topics", "topics", "topics"):
        _embedding(sdk_fakes)

    parallel_hits = [
        _point("doc-a", 0.91, point_id=1, type="summary", covered_range=[1, 3], order_index=4),
        _point("doc-b", 0.82, point_id=2, type="summary", covered_range=[4, 5], order_index=7),
        _point("doc-low", 0.12, point_id=3, type="summary", covered_range=[0, 1], order_index=2),
        _point("doc-d", 0.71, point_id=4, type="summary", covered_range=[2, 2], order_index=5),
    ]
    # The first two operations run concurrently; identical replies avoid depending on FIFO order.
    sdk_fakes.qdrant.queue("query_points", response=_query_response(parallel_hits))
    sdk_fakes.qdrant.queue("query_points", response=_query_response(parallel_hits))
    sdk_fakes.qdrant.queue("query_points", response=_query_response())  # separate user_fact query

    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="a3", payload={"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 3, "speaker": "B", "text": "third"}),
        Record(id="a2", payload={"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 2, "speaker": "A", "text": "second"}),
        Record(id="a2-copy", payload={"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 2, "speaker": "A duplicate", "text": "duplicate"}),
        Record(id="a1", payload={"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 1, "speaker": "A", "text": "first"}),
    ], None))
    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="b7", payload={"doc_id": "doc-b", "parent_id": "folder-child", "dialog_date": "2026-01-16", "order_index": 7, "speaker": "C", "text": "seven"}),
    ], None))
    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="d5", payload={"doc_id": "doc-d", "parent_id": "folder-child", "dialog_date": "2026-01-17", "order_index": 5, "speaker": "D", "text": "five"}),
    ], None))

    segments, metadata = run.run_retrieval(
        "topics", folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    )

    assert metadata == {
        "truncated": True,
        "total_matches": 4,
        "returned_segments": 3,
        "other_matches": [{"doc_id": "doc-low", "parent_id": "folder-child", "score": 0.0}],
        "suggestion": "Narrow down by period or folder for more precise results.",
    }
    assert [segment["doc_id"] for segment in segments] == ["doc-a", "doc-b", "doc-d"]
    assert segments[0] == {
        "doc_id": "doc-a",
        "folder_path": "Team/Child",
        "dialog_date": "2026-01-15",
        "segments": [{
            "range": [1, 6],
            "dialog": [
                {"speaker": "A", "text": "first", "order_index": 1},
                {"speaker": "A", "text": "second", "order_index": 2},
                {"speaker": "B", "text": "third", "order_index": 3},
            ],
        }],
    }

    query_events = _query_events(sdk_fakes)
    assert len(query_events) == 3
    for event in query_events[:2]:
        conditions = _must_by_key(event["query_filter"])
        assert conditions["parent_id"].match.any == ["folder-team", "folder-child"]
        assert conditions["dialog_date_num"].range.gte == 20260101
        assert conditions["dialog_date_num"].range.lte == 20260131
        assert conditions["root_folder_id"].match.value == ROOT_ID
    assert {event["using"] for event in query_events[:2]} == {"dense", "bm25"}
    assert query_events[2]["using"] == "dense"
    assert _must_by_key(query_events[2]["query_filter"])["type"].match.value == "user_fact"
    scroll_events = [event["arguments"] for event in sdk_fakes.trace
                     if event["boundary"] == "qdrant" and event["operation"] == "scroll"]
    assert [event["scroll_filter"].must[1].match.value for event in scroll_events] == ["doc-a", "doc-b", "doc-d"]
    assert scroll_events[0]["scroll_filter"].must[2].range.gte == 1
    assert scroll_events[0]["scroll_filter"].must[2].range.lte == 6


def test_user_facts_are_searched_separately_and_can_pin_below_threshold(
    sdk_fakes, monkeypatch
):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    for _ in range(4):
        _embedding(sdk_fakes)
    ordinary_hits = []
    sdk_fakes.qdrant.queue("query_points", response=_query_response(ordinary_hits))
    sdk_fakes.qdrant.queue("query_points", response=_query_response(ordinary_hits))
    sdk_fakes.qdrant.queue("query_points", response=_query_response([
        _point("doc-pinned", 0.99, point_id=10, type="user_fact")
    ]))
    sdk_fakes.qdrant.queue("query_points", response=_query_response([
        _point("doc-pinned", 0.2, point_id=11, type="fact", covered_range=[0, 1])
    ]))
    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="u0", payload={"doc_id": "doc-pinned", "parent_id": "folder", "dialog_date": "2026-03-01", "order_index": 0, "speaker": "Speaker", "text": "raw utterance"}),
    ], None))

    segments, metadata = run.run_retrieval("remember this")
    assert metadata == {"truncated": False, "total_matches": 1, "returned_segments": 1}
    assert segments[0]["segments"][0]["dialog"] == [
        {"speaker": "Speaker", "text": "raw utterance", "order_index": 0}
    ]

    events = _query_events(sdk_fakes)
    assert len(events) == 4
    assert {event["using"] for event in events[:2]} == {"dense", "bm25"}
    assert _must_by_key(events[2]["query_filter"])["type"].match.value == "user_fact"
    assert _must_by_key(events[3]["query_filter"])["doc_id"].match.value == "doc-pinned"
    assert "parent_id" not in _must_by_key(events[3]["query_filter"])
    assert "dialog_date_num" not in _must_by_key(events[3]["query_filter"])


def test_empty_retrieval_and_transcript_defaults_are_stable(sdk_fakes, firestore_fake, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    _seed_folders(firestore_fake)
    for _ in range(3):
        _embedding(sdk_fakes)
        sdk_fakes.qdrant.queue("query_points", response=_query_response())

    empty_result = ([], {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": "No data found for the given period or folder.",
    })
    assert run.run_retrieval(
        "no matches", folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    ) == empty_result
    for event in _query_events(sdk_fakes):
        conditions = _must_by_key(event["query_filter"])
        assert conditions["parent_id"].match.any == ["folder-team", "folder-child"]
        assert conditions["root_folder_id"].match.value == ROOT_ID
        assert conditions["dialog_date_num"].range.gte == 20260101
        assert conditions["dialog_date_num"].range.lte == 20260131
    assert {event["using"] for event in _query_events(sdk_fakes)[:2]} == {"dense", "bm25"}
    assert _must_by_key(_query_events(sdk_fakes)[2]["query_filter"])["type"].match.value == "user_fact"

    assert transcripts.get_transcripts() == ([], {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": "Provide doc_id or folder_path.",
    })
    assert not [event for event in sdk_fakes.trace if event["boundary"] == "qdrant" and event["operation"] == "scroll"]

    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="old", payload={
            "doc_id": "doc-old", "parent_id": "folder-team", "dialog_date": "2026-01-10",
            "order_index": 1, "speaker": "Older", "text": "older transcript",
        }),
        Record(id="new", payload={
            "doc_id": "doc-new", "parent_id": "folder-child", "dialog_date": "2026-01-20",
            "order_index": 3, "speaker": "Newer", "text": "newer transcript",
        }),
    ], None))
    transcript_segments, transcript_metadata = transcripts.get_transcripts(
        folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    )
    assert transcript_segments == [{
        "doc_id": "doc-new",
        "folder_path": "Team/Child",
        "dialog_date": "2026-01-20",
        "segments": [{
            "range": [3, 3],
            "dialog": [{"speaker": "Newer", "text": "newer transcript", "order_index": 3}],
        }],
    }]
    assert transcript_metadata == {
        "truncated": True,
        "total_matches": 2,
        "returned_segments": 1,
        "suggestion": "Use limit or narrow down the period to retrieve more transcripts.",
    }
    scroll_event = next(event["arguments"] for event in sdk_fakes.trace
                        if event["boundary"] == "qdrant" and event["operation"] == "scroll")
    transcript_filters = _must_by_key(scroll_event["scroll_filter"])
    assert transcript_filters["parent_id"].match.any == ["folder-team", "folder-child"]
    assert transcript_filters["root_folder_id"].match.value == ROOT_ID
    assert transcript_filters["dialog_date_num"].range.gte == 20260101
    assert transcript_filters["dialog_date_num"].range.lte == 20260131


def test_range_merge_deduplication_and_order_are_explicit():
    assert pipeline.merge_ranges([
        {"doc_id": "doc", "covered_range": [4, 7]},
        {"doc_id": "doc", "covered_range": [1, 3]},
        {"doc_id": "doc", "center_index": 9},
    ]) == {"doc": [[1, 11]]}
    assert pipeline.dedup_and_sort([
        {"doc_id": "doc", "order_index": 3, "text": "third"},
        {"doc_id": "doc", "order_index": 1, "text": "first"},
        {"doc_id": "doc", "order_index": 3, "text": "duplicate"},
    ]) == [
        {"doc_id": "doc", "order_index": 1, "text": "first"},
        {"doc_id": "doc", "order_index": 3, "text": "third"},
    ]
