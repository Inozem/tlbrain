from typing import Any

from core.config import get_retrieval_top_k
from core.retrieval.pipeline import dedup_and_sort, fetch_all_utterances
from core.retrieval.search import search_summaries_and_facts
from core.retrieval.segments import build_segments

_MAX_RESULT_DOCS = 3


def run_retrieval(
    query: str,
    client_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hits = search_summaries_and_facts(
        query=query,
        client_name=client_name,
        date_from=date_from,
        date_to=date_to,
        top_k=get_retrieval_top_k(),
    )

    if not hits:
        return [], {
            "truncated": False,
            "total_matches": 0,
            "returned_segments": 0,
            "limit_reason": "no_results",
            "suggestion": "Нет данных за выбранный период или клиента",
        }

    # Pick top _MAX_RESULT_DOCS documents by best hit score
    doc_best_score: dict[str, float] = {}
    doc_client: dict[str, str] = {}
    for hit in hits:
        doc_id = hit["doc_id"]
        if hit["score"] > doc_best_score.get(doc_id, -1):
            doc_best_score[doc_id] = hit["score"]
            doc_client[doc_id] = hit["client_name"] or ""

    sorted_docs = sorted(doc_best_score.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = {doc_id for doc_id, _ in sorted_docs[:_MAX_RESULT_DOCS]}
    other_docs = sorted_docs[_MAX_RESULT_DOCS:]

    result_segments = []
    for doc_id in top_doc_ids:
        utterances = dedup_and_sort(fetch_all_utterances(doc_id))
        if not utterances:
            continue
        full_range = [utterances[0]["order_index"], utterances[-1]["order_index"]]
        result_segments.append(build_segments(doc_id, [full_range], utterances))

    meta: dict[str, Any] = {
        "truncated": len(other_docs) > 0,
        "total_matches": len(doc_best_score),
        "returned_segments": len(result_segments),
    }
    if other_docs:
        meta["other_matches"] = [
            {"doc_id": doc_id, "client_name": doc_client[doc_id], "score": round(score, 4)}
            for doc_id, score in other_docs
        ]
        meta["suggestion"] = "Уточните период или клиента для более точного поиска"

    return result_segments, meta
