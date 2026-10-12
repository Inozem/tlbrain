from qdrant_client.http.models import QueryResponse

from core.integrations.fastembed import bm25


def test_model_is_lazy_shared_and_initialized_once(monkeypatch):
    from core.retrieval import search
    from services.vector_sync.app import processor

    constructions = []

    class Model:
        pass

    def construct(**kwargs):
        constructions.append(kwargs)
        return Model()

    monkeypatch.setattr(bm25, "_bm25_model", None)
    monkeypatch.setattr(bm25, "SparseTextEmbedding", construct)

    assert bm25._bm25_model is None
    assert search.get_bm25_model is bm25.get_bm25_model
    assert processor.get_bm25_model is bm25.get_bm25_model
    assert constructions == []

    first = bm25.get_bm25_model()
    assert bm25.get_bm25_model() is first
    assert constructions == [{"model_name": "Qdrant/bm25"}]


def test_keyword_search_keeps_sparse_embed_call_and_vector(sdk_fakes, monkeypatch):
    from core.retrieval.search import keyword_search_utterances

    monkeypatch.setattr(bm25, "_bm25_model", None)
    sdk_fakes.sparse.set_vectors("sprint review", [11, 4], [0.75, 0.2])
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[]))

    assert keyword_search_utterances("sprint review") == []

    sparse_calls = [event for event in sdk_fakes.trace if event["boundary"] == "sparse"]
    assert [event["operation"] for event in sparse_calls] == ["client", "embed"]
    assert sparse_calls[1]["arguments"]["text"] == "sprint review"
    query = next(
        event["arguments"]
        for event in sdk_fakes.trace
        if event["boundary"] == "qdrant" and event["operation"] == "query_points"
    )
    assert query["using"] == "bm25"
    assert query["query"].indices == [11, 4]
    assert query["query"].values == [0.75, 0.2]
