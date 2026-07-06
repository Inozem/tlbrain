import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.config import get_retrieval_score_threshold, get_retrieval_top_k
from core.google_drive.firestore import expand_subtree, get_all_folders, resolve_folder_path as _resolve_path
from core.retrieval.pipeline import dedup_and_sort, fetch_utterances, merge_ranges
from core.retrieval.search import keyword_search_utterances, search_summaries_and_facts, search_summaries_for_doc, search_user_facts
from core.retrieval.segments import build_segments, make_folder_path_resolver

logger = logging.getLogger(__name__)

_MAX_RESULT_DOCS = 3


def run_retrieval(
    query: str,
    keywords: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Resolve folder_path → subtree-expanded folder_ids; load folders for path display
    folder_ids: list[str] | None = None
    all_folders = get_all_folders()
    if folder_path:
        terminal_ids = _resolve_path(folder_path)
        if not terminal_ids:
            raise ValueError(f"Folder not found: {'/'.join(folder_path)}")
        folder_ids = [fid for tid in terminal_ids for fid in expand_subtree(tid)]

    resolve_path = make_folder_path_resolver(all_folders)

    # Stage 1: parallel semantic + keyword search
    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            search_summaries_and_facts,
            query=query,
            folder_ids=folder_ids,
            date_from=date_from,
            date_to=date_to,
            top_k=get_retrieval_top_k(),
        )
        keyword_future = executor.submit(
            keyword_search_utterances,
            query=keywords if keywords else query,
            folder_ids=folder_ids,
            date_from=date_from,
            date_to=date_to,
        )

    semantic_hits = semantic_future.result()
    keyword_hits = keyword_future.result()

    # Pin: documents with user_facts matching the query bypass the score threshold
    pinned_hits: list[dict[str, Any]] = []
    user_fact_hits = search_user_facts(query, folder_ids=folder_ids, date_from=date_from, date_to=date_to)
    if user_fact_hits:
        hits_by_doc: dict[str, int] = {}
        for h in user_fact_hits:
            doc_id = h["doc_id"]
            if doc_id:
                hits_by_doc[doc_id] = hits_by_doc.get(doc_id, 0) + 1
        for doc_id, hit_count in hits_by_doc.items():
            extra = search_summaries_for_doc(doc_id, query, top_k=min(hit_count * 5, 20))
            pinned_hits.extend(extra)
        logger.info("query=%r user_fact_pins=%d", query, len(hits_by_doc))

    logger.info(
        "query=%r semantic_hits=%d keyword_hits=%d",
        query, len(semantic_hits), len(keyword_hits),
    )

    threshold = get_retrieval_score_threshold()
    semantic_hits = [h for h in semantic_hits if h["score"] >= threshold]

    # Pinned hits are appended after threshold filtering — they are never filtered out
    semantic_hits = semantic_hits + pinned_hits

    if not semantic_hits and not keyword_hits:
        return [], {
            "truncated": False,
            "total_matches": 0,
            "returned_segments": 0,
            "limit_reason": "no_results",
            "suggestion": "No data found for the given period or folder.",
        }

    # Pick top _MAX_RESULT_DOCS documents by best semantic score;
    # keyword-only docs get score 0 and fill remaining slots if any.
    doc_best_score: dict[str, float] = {}
    doc_parent_id: dict[str, str] = {}
    for hit in semantic_hits:
        doc_id = hit["doc_id"]
        if hit["score"] > doc_best_score.get(doc_id, -1):
            doc_best_score[doc_id] = hit["score"]
            doc_parent_id[doc_id] = hit.get("parent_id") or ""
    for hit in keyword_hits:
        doc_id = hit["doc_id"]
        if doc_id not in doc_best_score:
            doc_best_score[doc_id] = 0.0
            doc_parent_id[doc_id] = hit.get("parent_id") or ""

    sorted_docs = sorted(doc_best_score.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = {doc_id for doc_id, _ in sorted_docs[:_MAX_RESULT_DOCS]}
    other_docs = sorted_docs[_MAX_RESULT_DOCS:]

    all_hits = semantic_hits + keyword_hits
    top_hits = [h for h in all_hits if h["doc_id"] in top_doc_ids]
    merged_by_doc = merge_ranges(top_hits)

    result_segments = []
    for doc_id, doc_ranges in merged_by_doc.items():
        utterances = dedup_and_sort(fetch_utterances(doc_id, doc_ranges))
        result_segments.append(build_segments(doc_id, doc_ranges, utterances, resolve_path))

    meta: dict[str, Any] = {
        "truncated": len(other_docs) > 0,
        "total_matches": len(doc_best_score),
        "returned_segments": len(result_segments),
    }
    if other_docs:
        meta["other_matches"] = [
            {"doc_id": doc_id, "parent_id": doc_parent_id[doc_id], "score": round(score, 4)}
            for doc_id, score in other_docs
        ]
        meta["suggestion"] = "Narrow down by period or folder for more precise results."

    return result_segments, meta
