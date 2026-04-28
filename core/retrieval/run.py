from typing import Any

from core.config import get_retrieval_top_k
from core.retrieval.pipeline import dedup_and_sort, fetch_utterances, merge_ranges
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

    top_hits = [h for h in hits if h["doc_id"] in top_doc_ids]
    merged_by_doc = merge_ranges(top_hits)

    result_segments = []
    for doc_id, doc_ranges in merged_by_doc.items():
        utterances = dedup_and_sort(fetch_utterances(doc_id, doc_ranges))
        result_segments.append(build_segments(doc_id, doc_ranges, utterances))

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
