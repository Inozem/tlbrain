"""Offline behavior baselines for retrieval over plain mapping results."""

from threading import Barrier
from types import SimpleNamespace

from core.retrieval import pipeline, run, search, transcripts


ROOT_ID = "pytest-synthetic-root"


def _seed_folders(firestore_fake):
    folders = firestore_fake.client_class().collection("folders")
    folders.document("folder-ancestor").set({"name": "Ancestor", "parent_id": "drive-root"})
    folders.document(ROOT_ID).set({"name": "Configured Root", "parent_id": "folder-ancestor"})
    folders.document("folder-team").set({"name": "Team", "parent_id": ROOT_ID})
    folders.document("folder-child").set({"name": "Child", "parent_id": "folder-team"})


def test_empty_folder_paths_keep_no_result_defaults(monkeypatch):
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("empty folder path must not load folder data or scroll Qdrant")

    monkeypatch.setattr(transcripts, "get_root_folder_id", unexpected_call)
    monkeypatch.setattr(transcripts, "get_all_folders", unexpected_call)
    monkeypatch.setattr(transcripts, "_scroll_by_folder", unexpected_call)
    assert transcripts.get_transcripts(folder_path=[]) == ([], {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": "Folder not found: ",
    })
    assert transcripts.get_transcripts() == ([], {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": "Provide doc_id or folder_path.",
    })


def test_search_operations_pass_plain_arguments_to_reader(monkeypatch):
    calls = []
    monkeypatch.setattr(search, "embed", lambda texts: [[0.25] * 768 for _ in texts])

    class SparseValues(list):
        def tolist(self):
            return list(self)

    sparse_embedding = SimpleNamespace(
        indices=SparseValues([3, 8]), values=SparseValues([0.2, 0.7])
    )
    bm25_model = SimpleNamespace(embed=lambda _texts: iter([sparse_embedding]))
    monkeypatch.setattr(search, "get_bm25_model", lambda: bm25_model)

    def record(name):
        def call(*args):
            calls.append((name, args))
            return []
        return call

    monkeypatch.setattr(search.reader, "search_summaries_and_facts", record("semantic"))
    monkeypatch.setattr(search.reader, "keyword_search_utterances", record("keyword"))
    monkeypatch.setattr(search.reader, "search_user_facts", record("user_facts"))
    monkeypatch.setattr(search.reader, "search_summaries_for_doc", record("pinned"))

    search.search_summaries_and_facts(
        "review", ["folder-team", "folder-child"], "2026-01-02", "2026-02-03", 23,
    )
    search.keyword_search_utterances(
        "review plan", folder_ids=["folder-team"], date_from="2026-01-02", date_to="2026-02-03",
    )
    search.search_user_facts(
        "review", folder_ids=["folder-child"], date_from="2026-01-02", date_to="2026-02-03",
    )
    search.search_summaries_for_doc("doc-pinned", "review", top_k=5)

    assert [name for name, _ in calls] == ["semantic", "keyword", "user_facts", "pinned"]
    assert calls[0][1] == ([0.25] * 768, ["folder-team", "folder-child"], "2026-01-02", "2026-02-03", 23)
    assert calls[1][1] == ([3, 8], [0.2, 0.7], 10, ["folder-team"], "2026-01-02", "2026-02-03")
    assert calls[2][1] == ([0.25] * 768, 10, ["folder-child"], "2026-01-02", "2026-02-03")
    assert calls[3][1] == ([0.25] * 768, "doc-pinned", 5)


def test_retrieval_expands_folder_filters_merges_ranges_and_caps_documents(
    monkeypatch, firestore_fake
):
    _seed_folders(firestore_fake)
    monkeypatch.setattr(run, "get_retrieval_top_k", lambda: 15)
    monkeypatch.setattr(run, "get_retrieval_score_threshold", lambda: 0.5)
    search_calls = []
    concurrent_searches = Barrier(2, timeout=2)

    def semantic(**kwargs):
        concurrent_searches.wait()
        search_calls.append(("semantic", kwargs))
        return [
            {"doc_id": "doc-a", "parent_id": "folder-child", "score": 0.91, "covered_range": [1, 3]},
            {"doc_id": "doc-b", "parent_id": "folder-child", "score": 0.82, "covered_range": [4, 5]},
            {"doc_id": "doc-low", "parent_id": "folder-child", "score": 0.12, "covered_range": [0, 1]},
            {"doc_id": "doc-d", "parent_id": "folder-child", "score": 0.71, "covered_range": [2, 2]},
        ]

    def keyword(**kwargs):
        concurrent_searches.wait()
        search_calls.append(("keyword", kwargs))
        return [
            {"doc_id": "doc-a", "parent_id": "folder-child", "score": 0.5, "covered_range": [1, 6]},
            {"doc_id": "doc-low", "parent_id": "folder-child", "score": 0.1, "covered_range": [0, 1]},
        ]

    monkeypatch.setattr(run, "search_summaries_and_facts", semantic)
    monkeypatch.setattr(run, "keyword_search_utterances", keyword)
    monkeypatch.setattr(run, "search_user_facts", lambda *_a, **_kw: [])
    fetched = {
        "doc-a": [
            {"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 3, "speaker": "B", "text": "third"},
            {"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 2, "speaker": "A", "text": "second"},
            {"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 2, "speaker": "duplicate", "text": "duplicate"},
            {"doc_id": "doc-a", "parent_id": "folder-child", "dialog_date": "2026-01-15", "order_index": 1, "speaker": "A", "text": "first"},
        ],
        "doc-b": [{"doc_id": "doc-b", "parent_id": "folder-child", "dialog_date": "2026-01-16", "order_index": 7, "speaker": "C", "text": "seven"}],
        "doc-d": [{"doc_id": "doc-d", "parent_id": "folder-child", "dialog_date": "2026-01-17", "order_index": 5, "speaker": "D", "text": "five"}],
    }
    fetch_calls = []

    def fetch(doc_id, ranges):
        fetch_calls.append((doc_id, ranges))
        return fetched[doc_id]

    monkeypatch.setattr(run, "fetch_utterances", fetch)

    segments, metadata = run.run_retrieval(
        "topics", folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    )

    assert sorted(name for name, _ in search_calls) == ["keyword", "semantic"]
    for _, args in search_calls:
        assert args["folder_ids"] == ["folder-team", "folder-child"]
        assert args["date_from"] == "2026-01-01"
        assert args["date_to"] == "2026-01-31"
    assert fetch_calls == [("doc-a", [[1, 6]]), ("doc-b", [[4, 5]]), ("doc-d", [[2, 2]])]
    assert [segment["doc_id"] for segment in segments] == ["doc-a", "doc-b", "doc-d"]
    assert metadata == {
        "truncated": True, "total_matches": 4, "returned_segments": 3,
        "other_matches": [{"doc_id": "doc-low", "parent_id": "folder-child", "score": 0.0}],
        "suggestion": "Narrow down by period or folder for more precise results.",
    }
    assert segments[0] == {
        "doc_id": "doc-a", "folder_path": "Team/Child", "dialog_date": "2026-01-15",
        "segments": [{"range": [1, 6], "dialog": [
            {"speaker": "A", "text": "first", "order_index": 1},
            {"speaker": "A", "text": "second", "order_index": 2},
            {"speaker": "B", "text": "third", "order_index": 3},
        ]}],
    }


def test_user_facts_are_separate_and_pin_summaries_below_threshold(monkeypatch):
    monkeypatch.setattr(run, "get_all_folders", lambda: {})
    monkeypatch.setattr(run, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(run, "get_retrieval_top_k", lambda: 15)
    monkeypatch.setattr(run, "get_retrieval_score_threshold", lambda: 0.5)
    calls = []
    monkeypatch.setattr(run, "search_summaries_and_facts", lambda **_kw: [])
    monkeypatch.setattr(run, "keyword_search_utterances", lambda **_kw: [])
    user_fact_calls = []

    def user_facts(*args, **kwargs):
        user_fact_calls.append((args, kwargs))
        return [{"doc_id": "doc-pinned", "score": 0.99}]

    monkeypatch.setattr(run, "search_user_facts", user_facts)

    def pinned(doc_id, query, top_k):
        calls.append((doc_id, query, top_k))
        return [{"doc_id": doc_id, "parent_id": "folder", "score": 0.2, "covered_range": [0, 1]}]

    monkeypatch.setattr(run, "search_summaries_for_doc", pinned)
    monkeypatch.setattr(run, "fetch_utterances", lambda *_args: [
        {"doc_id": "doc-pinned", "parent_id": "folder", "dialog_date": "2026-03-01", "order_index": 0, "speaker": "Speaker", "text": "raw utterance"}
    ])

    segments, metadata = run.run_retrieval("remember this")
    assert user_fact_calls == [(('remember this',), {"folder_ids": None, "date_from": None, "date_to": None})]
    assert calls == [("doc-pinned", "remember this", 5)]
    assert metadata == {"truncated": False, "total_matches": 1, "returned_segments": 1}
    assert segments[0]["segments"][0]["dialog"] == [
        {"speaker": "Speaker", "text": "raw utterance", "order_index": 0}
    ]


def test_empty_retrieval_keeps_default_metadata_and_folder_arguments(monkeypatch, firestore_fake):
    _seed_folders(firestore_fake)
    monkeypatch.setattr(run, "get_retrieval_top_k", lambda: 15)
    monkeypatch.setattr(run, "get_retrieval_score_threshold", lambda: 0.5)
    calls = []

    def empty_search(name):
        def search_stub(*args, **kwargs):
            calls.append((name, args, kwargs))
            return []
        return search_stub

    monkeypatch.setattr(run, "search_summaries_and_facts", empty_search("semantic"))
    monkeypatch.setattr(run, "keyword_search_utterances", empty_search("keyword"))
    monkeypatch.setattr(run, "search_user_facts", empty_search("user_facts"))
    assert run.run_retrieval(
        "no matches", folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    ) == ([], {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": "No data found for the given period or folder.",
    })
    assert [name for name, _, _ in calls] == ["semantic", "keyword", "user_facts"]
    for _, _, kwargs in calls:
        assert kwargs["folder_ids"] == ["folder-team", "folder-child"]
        assert kwargs["date_from"] == "2026-01-01"
        assert kwargs["date_to"] == "2026-01-31"


def test_transcripts_use_mapping_results_and_keep_folder_date_behavior(monkeypatch, firestore_fake):
    _seed_folders(firestore_fake)
    scroll_calls = []
    monkeypatch.setattr(transcripts, "_scroll_by_folder", lambda folder_ids, date_from, date_to: (
        scroll_calls.append((folder_ids, date_from, date_to)) or [
            {"doc_id": "doc-old", "parent_id": "folder-team", "dialog_date": "2026-01-10", "order_index": 1, "speaker": "Older", "text": "older"},
            {"doc_id": "doc-new", "parent_id": "folder-child", "dialog_date": "2026-01-20", "order_index": 3, "speaker": "Newer", "text": "newer"},
        ]
    ))
    segments, metadata = transcripts.get_transcripts(
        folder_path=["Team"], date_from="2026-01-01", date_to="2026-01-31"
    )
    assert scroll_calls == [(["folder-team", "folder-child"], "2026-01-01", "2026-01-31")]
    assert segments == [{
        "doc_id": "doc-new", "folder_path": "Team/Child", "dialog_date": "2026-01-20",
        "segments": [{"range": [3, 3], "dialog": [{"speaker": "Newer", "text": "newer", "order_index": 3}]}],
    }]
    assert metadata == {
        "truncated": True, "total_matches": 2, "returned_segments": 1,
        "suggestion": "Use limit or narrow down the period to retrieve more transcripts.",
    }


def test_direct_transcript_lookup_uses_mapping_utterances(monkeypatch, firestore_fake):
    _seed_folders(firestore_fake)
    monkeypatch.setattr(transcripts, "_scroll_by_doc_id", lambda _doc_id: [{
        "doc_id": "doc-direct", "parent_id": "folder-child", "dialog_date": "2026-01-20",
        "order_index": 0, "speaker": "Speaker", "text": "direct transcript",
    }])
    segments, metadata = transcripts.get_transcripts(doc_id="doc-direct")
    assert segments[0]["folder_path"] == "Team/Child"
    assert metadata == {"truncated": False, "total_matches": 1, "returned_segments": 1}


def test_missing_direct_transcript_skips_folder_loading(monkeypatch):
    monkeypatch.setattr(transcripts, "_scroll_by_doc_id", lambda _doc_id: [])

    def unexpected_folder_load():
        raise AssertionError("a missing document must not load folder data")

    monkeypatch.setattr(transcripts, "get_all_folders", unexpected_folder_load)
    monkeypatch.setattr(transcripts, "get_root_folder_id", unexpected_folder_load)
    segments, metadata = transcripts.get_transcripts(doc_id="missing-doc")
    assert segments == []
    assert metadata["limit_reason"] == "no_results"


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
