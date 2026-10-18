"""Offline characterization of the existing Qdrant reader calls."""

from google.genai import types
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import Record, ScoredPoint

from core.integrations.qdrant import reader
from core.retrieval import pipeline, search, transcripts


ROOT_ID = "pytest-synthetic-root"
COLLECTION = "tldv_vectors"


def _embedding(sdk_fakes, values=None):
    sdk_fakes.gemini.queue(
        "embed_content",
        response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=values or [0.25] * 768)]
        ),
    )


def _point(payload, score=0.8, point_id=1):
    return ScoredPoint(id=point_id, version=1, score=score, payload=payload)


def _query_events(sdk_fakes):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == "query_points"]


def _scroll_events(sdk_fakes):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == "scroll"]


def _conditions(qdrant_filter):
    return {condition.key: condition for condition in qdrant_filter.must}


def test_search_read_calls_keep_filters_vectors_limits_and_hit_shapes(sdk_fakes):
    sdk_fakes.sparse.set_vectors("needle", [4, 9], [0.4, 0.9])
    for _ in range(3):
        _embedding(sdk_fakes)

    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[_point({
        "doc_id": "doc-semantic", "parent_id": "folder-a", "type": "summary",
        "center_index": 5, "covered_range": [3, 7],
    }, point_id=1)]))
    semantic = search.search_summaries_and_facts(
        "needle", folder_ids=["folder-a", "folder-b"],
        date_from="2026-02-03", date_to="2026-02-08", top_k=13,
    )
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[_point({
        "doc_id": "doc-keyword", "parent_id": "folder-b", "order_index": 1,
    }, score=0.7, point_id=2)]))
    keyword = search.keyword_search_utterances(
        "needle", folder_ids=["folder-a", "folder-b"],
        date_from="2026-02-03", date_to="2026-02-08",
    )
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[_point({
        "doc_id": "doc-user-fact",
    }, score=0.6, point_id=3)]))
    user_facts = search.search_user_facts(
        "needle", folder_ids=["folder-a", "folder-b"],
        date_from="2026-02-03", date_to="2026-02-08", top_k=6,
    )
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[_point({
        "doc_id": "doc-pinned", "type": "fact", "parent_id": "folder-c",
    }, score=0.5, point_id=4)]))
    per_doc = search.search_summaries_for_doc("doc-pinned", "needle", top_k=4)

    events = _query_events(sdk_fakes)
    assert [event["using"] for event in events] == ["dense", "bm25", "dense", "dense"]
    assert [event["limit"] for event in events] == [13, 10, 6, 4]
    assert [event["collection_name"] for event in events] == [COLLECTION] * 4
    assert all(event["with_payload"] is True for event in events)
    assert all("offset" not in event and "with_vectors" not in event for event in events)
    assert events[0]["query"] == [0.25] * 768
    assert events[1]["query"].indices == [4, 9]
    assert events[1]["query"].values == [0.4, 0.9]

    semantic_filter = _conditions(events[0]["query_filter"])
    assert set(semantic_filter) == {"type", "root_folder_id", "parent_id", "dialog_date_num"}
    assert semantic_filter["type"].match.any == ["summary", "fact"]
    assert semantic_filter["root_folder_id"].match.value == ROOT_ID
    assert semantic_filter["parent_id"].match.any == ["folder-a", "folder-b"]
    assert (semantic_filter["dialog_date_num"].range.gte,
            semantic_filter["dialog_date_num"].range.lte) == (20260203, 20260208)

    keyword_filter = _conditions(events[1]["query_filter"])
    assert set(keyword_filter) == {"type", "root_folder_id", "parent_id", "dialog_date_num"}
    assert keyword_filter["type"].match.value == "utterance"
    assert keyword_filter["root_folder_id"].match.value == ROOT_ID
    assert keyword_filter["parent_id"].match.any == ["folder-a", "folder-b"]
    assert (keyword_filter["dialog_date_num"].range.gte,
            keyword_filter["dialog_date_num"].range.lte) == (20260203, 20260208)

    user_filter = _conditions(events[2]["query_filter"])
    assert set(user_filter) == {"type", "root_folder_id", "parent_id", "dialog_date_num"}
    assert user_filter["type"].match.value == "user_fact"
    assert user_filter["root_folder_id"].match.value == ROOT_ID
    assert user_filter["parent_id"].match.any == ["folder-a", "folder-b"]
    assert (user_filter["dialog_date_num"].range.gte,
            user_filter["dialog_date_num"].range.lte) == (20260203, 20260208)

    per_doc_filter = _conditions(events[3]["query_filter"])
    assert set(per_doc_filter) == {"type", "root_folder_id", "doc_id"}
    assert per_doc_filter["type"].match.any == ["summary", "fact"]
    assert per_doc_filter["root_folder_id"].match.value == ROOT_ID
    assert per_doc_filter["doc_id"].match.value == "doc-pinned"

    assert semantic == [{
        "doc_id": "doc-semantic", "parent_id": "folder-a", "type": "summary",
        "center_index": 5, "covered_range": [3, 7], "score": 0.8,
    }]
    assert keyword == [{
        "doc_id": "doc-keyword", "parent_id": "folder-b", "covered_range": [0, 3], "score": 0.7,
    }]
    assert user_facts == [{"doc_id": "doc-user-fact", "score": 0.6}]
    assert per_doc == [{
        "doc_id": "doc-pinned", "parent_id": "folder-c", "type": "fact",
        "center_index": None, "covered_range": None, "score": 0.5,
    }]


def test_pipeline_fetch_keeps_its_narrow_range_filter_and_payloads(sdk_fakes):
    first = {"doc_id": "doc-range", "type": "utterance", "order_index": 2, "text": "two"}
    second = {"doc_id": "doc-range", "type": "utterance", "order_index": 8, "text": "eight"}
    sdk_fakes.qdrant.queue("scroll", response=([Record(id="two", payload=first)], "ignored-next"))
    sdk_fakes.qdrant.queue("scroll", response=([Record(id="empty", payload=None), Record(id="eight", payload=second)], None))

    assert pipeline.fetch_utterances("doc-range", [[1, 3], [7, 9]]) == [first, second]
    events = _scroll_events(sdk_fakes)
    assert len(events) == 2
    assert [event["collection_name"] for event in events] == [COLLECTION] * 2
    assert [event["limit"] for event in events] == [1000, 1000]
    assert all(event["with_payload"] is True and event["with_vectors"] is False for event in events)
    assert all("offset" not in event for event in events)
    for event, expected_range in zip(events, [(1, 3), (7, 9)]):
        conditions = _conditions(event["scroll_filter"])
        assert set(conditions) == {"type", "doc_id", "order_index"}
        assert conditions["type"].match.value == "utterance"
        assert conditions["doc_id"].match.value == "doc-range"
        assert (conditions["order_index"].range.gte,
                conditions["order_index"].range.lte) == expected_range


def test_transcript_scrolls_keep_filters_and_follow_pagination_offsets(sdk_fakes):
    doc_payload = {"doc_id": "doc-transcript", "order_index": 1, "text": "doc row"}
    folder_payload = {"doc_id": "doc-folder", "order_index": 4, "text": "folder row"}
    sdk_fakes.qdrant.queue("scroll", response=([Record(id="doc-row", payload=doc_payload)], "cursor-doc"))
    sdk_fakes.qdrant.queue("scroll", response=([Record(id="doc-row-2", payload=None)], None))
    assert transcripts._scroll_by_doc_id("doc-transcript") == [doc_payload]

    sdk_fakes.qdrant.queue("scroll", response=([], "cursor-folder"))
    sdk_fakes.qdrant.queue("scroll", response=([Record(id="folder-row", payload=folder_payload)], None))
    assert transcripts._scroll_by_folder(
        ["folder-a", "folder-b"], "2026-04-05", "2026-04-09"
    ) == [folder_payload]

    events = _scroll_events(sdk_fakes)
    assert len(events) == 4
    assert all(event["collection_name"] == COLLECTION for event in events)
    assert all(event["limit"] == 1000 for event in events)
    assert all(event["with_payload"] is True and event["with_vectors"] is False for event in events)
    assert [event["offset"] for event in events] == [None, "cursor-doc", None, "cursor-folder"]

    doc_filter = _conditions(events[0]["scroll_filter"])
    assert set(doc_filter) == {"type", "doc_id", "root_folder_id"}
    assert doc_filter["type"].match.value == "utterance"
    assert doc_filter["doc_id"].match.value == "doc-transcript"
    assert doc_filter["root_folder_id"].match.value == ROOT_ID
    assert events[1]["scroll_filter"] == events[0]["scroll_filter"]

    folder_filter = _conditions(events[2]["scroll_filter"])
    assert set(folder_filter) == {"type", "parent_id", "root_folder_id", "dialog_date_num"}
    assert folder_filter["type"].match.value == "utterance"
    assert folder_filter["parent_id"].match.any == ["folder-a", "folder-b"]
    assert folder_filter["root_folder_id"].match.value == ROOT_ID
    assert (folder_filter["dialog_date_num"].range.gte,
            folder_filter["dialog_date_num"].range.lte) == (20260405, 20260409)
    assert events[3]["scroll_filter"] == events[2]["scroll_filter"]


def test_writer_fetch_uses_doc_root_indexes_and_returns_sorted_payloads(sdk_fakes):
    later = {"doc_id": "doc-write", "root_folder_id": ROOT_ID, "order_index": 6}
    earlier = {"doc_id": "doc-write", "root_folder_id": ROOT_ID, "order_index": 2}
    sdk_fakes.qdrant.queue("scroll", response=([
        Record(id="later", payload=later), Record(id="earlier", payload=earlier),
    ], None))

    assert reader.fetch_utterances_by_order_indexes(
        "doc-write", ROOT_ID, [6, 2, 9]
    ) == [earlier, later]
    [event] = _scroll_events(sdk_fakes)
    assert event["collection_name"] == COLLECTION
    assert event["limit"] == 3
    assert event["with_payload"] is True and event["with_vectors"] is False
    assert "offset" not in event
    conditions = _conditions(event["scroll_filter"])
    assert set(conditions) == {"doc_id", "root_folder_id", "order_index", "type"}
    assert conditions["doc_id"].match.value == "doc-write"
    assert conditions["root_folder_id"].match.value == ROOT_ID
    assert conditions["order_index"].match.any == [6, 2, 9]
    assert conditions["type"].match.value == "utterance"
