import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from core.config import get_retrieval_score_threshold, get_retrieval_top_k, get_root_folder_id
from core.integrations.firestore.folders import get_all_folders
from core.integrations.google_drive.folders import (
    expand_subtree,
    make_folder_path_resolver,
    resolve_folder_path,
)
from core.retrieval.pipeline import dedup_and_sort, fetch_utterances, merge_ranges
from core.retrieval.search import keyword_search_utterances, search_summaries_and_facts, search_summaries_for_doc, search_user_facts
from core.retrieval.segments import build_segments

logger = logging.getLogger(__name__)

_MAX_RESULT_DOCS = 3


def _resolve_folder_context(
    folder_path: list[str] | None,
) -> tuple[list[str] | None, Callable[[str], str]]:
    all_folders = get_all_folders()
    root_folder_id = get_root_folder_id()
    folder_ids = None
    if folder_path:
        terminal_ids = resolve_folder_path(folder_path, all_folders, root_folder_id)
        if not terminal_ids:
            raise ValueError(f"Folder not found: {'/'.join(folder_path)}")
        folder_ids = [
            folder_id
            for terminal_id in terminal_ids
            for folder_id in expand_subtree(terminal_id, all_folders)
        ]
    return folder_ids, make_folder_path_resolver(all_folders, root_folder_id)


def _search_in_parallel(
    query: str,
    keywords: str | None,
    folder_ids: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    return semantic_future.result(), keyword_future.result()


def _find_pinned_summaries(
    query: str,
    folder_ids: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> list[dict[str, Any]]:
    pinned_hits: list[dict[str, Any]] = []
    user_fact_hits = search_user_facts(
        query, folder_ids=folder_ids, date_from=date_from, date_to=date_to
    )
    if user_fact_hits:
        hits_by_doc: dict[str, int] = {}
        for hit in user_fact_hits:
            doc_id = hit["doc_id"]
            if doc_id:
                hits_by_doc[doc_id] = hits_by_doc.get(doc_id, 0) + 1
        for doc_id, hit_count in hits_by_doc.items():
            pinned_hits.extend(
                search_summaries_for_doc(doc_id, query, top_k=min(hit_count * 5, 20))
            )
        logger.info("query=%r user_fact_pins=%d", query, len(hits_by_doc))
    return pinned_hits


def _select_documents(
    semantic_hits: list[dict[str, Any]], keyword_hits: list[dict[str, Any]]
) -> tuple[set[str], list[tuple[str, float]], dict[str, str], int]:
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

    ranked_docs = sorted(doc_best_score.items(), key=lambda item: item[1], reverse=True)
    top_doc_ids = {doc_id for doc_id, _ in ranked_docs[:_MAX_RESULT_DOCS]}
    return top_doc_ids, ranked_docs[_MAX_RESULT_DOCS:], doc_parent_id, len(doc_best_score)


def _build_retrieval_result(
    semantic_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    resolve_path: Callable[[str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    top_doc_ids, other_docs, doc_parent_id, total_matches = _select_documents(
        semantic_hits, keyword_hits
    )
    all_hits = semantic_hits + keyword_hits
    top_hits = [hit for hit in all_hits if hit["doc_id"] in top_doc_ids]
    merged_by_doc = merge_ranges(top_hits)

    result_segments = []
    for doc_id, doc_ranges in merged_by_doc.items():
        utterances = dedup_and_sort(fetch_utterances(doc_id, doc_ranges))
        result_segments.append(build_segments(doc_id, doc_ranges, utterances, resolve_path))

    metadata: dict[str, Any] = {
        "truncated": len(other_docs) > 0,
        "total_matches": total_matches,
        "returned_segments": len(result_segments),
    }
    if other_docs:
        metadata["other_matches"] = [
            {"doc_id": doc_id, "parent_id": doc_parent_id[doc_id], "score": round(score, 4)}
            for doc_id, score in other_docs
        ]
        metadata["suggestion"] = "Narrow down by period or folder for more precise results."
    return result_segments, metadata


def run_retrieval(
    query: str,
    keywords: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folder_ids, resolve_path = _resolve_folder_context(folder_path)
    semantic_hits, keyword_hits = _search_in_parallel(
        query, keywords, folder_ids, date_from, date_to
    )
    pinned_hits = _find_pinned_summaries(query, folder_ids, date_from, date_to)

    logger.info(
        "query=%r semantic_hits=%d keyword_hits=%d",
        query, len(semantic_hits), len(keyword_hits),
    )
    threshold = get_retrieval_score_threshold()
    semantic_hits = [hit for hit in semantic_hits if hit["score"] >= threshold]
    semantic_hits = semantic_hits + pinned_hits

    if not semantic_hits and not keyword_hits:
        return [], {
            "truncated": False,
            "total_matches": 0,
            "returned_segments": 0,
            "limit_reason": "no_results",
            "suggestion": "No data found for the given period or folder.",
        }

    return _build_retrieval_result(semantic_hits, keyword_hits, resolve_path)
