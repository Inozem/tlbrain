from collections.abc import Callable
from typing import Any

from core.config import get_root_folder_id
from core.integrations.firestore.folders import get_all_folders
from core.integrations.google_drive.folders import (
    expand_subtree,
    make_folder_path_resolver,
    resolve_folder_path,
)
from core.integrations.qdrant import reader
from core.retrieval.pipeline import dedup_and_sort
from core.retrieval.segments import build_segments


def get_transcripts(
    doc_id: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolve_path = None
    if doc_id is not None:
        utterances, resolve_path = _load_doc_transcripts(doc_id)
    elif folder_path is not None:
        folder_result = _load_folder_transcripts(folder_path, date_from, date_to)
        if folder_result is None:
            return [], _empty_metadata(f"Folder not found: {'/'.join(folder_path)}")
        utterances, resolve_path = folder_result
    else:
        return [], _empty_metadata("Provide doc_id or folder_path.")

    if not utterances:
        return [], _empty_metadata("No data found for the given period or folder.")

    sorted_docs = _group_and_sort_documents(utterances)
    total_docs = len(sorted_docs)
    top_docs = sorted_docs[:limit]
    truncated = total_docs > limit
    result_segments = _build_selected_segments(top_docs, resolve_path)
    meta = _build_metadata(total_docs, len(result_segments), truncated)

    return result_segments, meta


def _load_doc_transcripts(
    doc_id: str,
) -> tuple[list[dict[str, Any]], Callable[[str], str] | None]:
    utterances = _scroll_by_doc_id(doc_id)
    if not utterances:
        return utterances, None
    return utterances, _load_doc_path_resolver()


def _load_doc_path_resolver() -> Callable[[str], str]:
    all_folders = get_all_folders()
    return make_folder_path_resolver(all_folders, get_root_folder_id())


def _load_folder_transcripts(
    folder_path: list[str],
    date_from: str | None,
    date_to: str | None,
) -> tuple[list[dict[str, Any]], Callable[[str], str]] | None:
    if not folder_path:
        return None

    all_folders = get_all_folders()
    root_folder_id = get_root_folder_id()
    terminal_ids = resolve_folder_path(folder_path, all_folders, root_folder_id)
    if not terminal_ids:
        return None

    folder_ids = [
        folder_id
        for terminal_id in terminal_ids
        for folder_id in expand_subtree(terminal_id, all_folders)
    ]
    resolve_path = make_folder_path_resolver(all_folders, root_folder_id)
    utterances = _scroll_by_folder(folder_ids, date_from, date_to)
    return utterances, resolve_path


def _group_and_sort_documents(
    utterances: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    docs: dict[str, list[dict[str, Any]]] = {}
    for utterance in utterances:
        docs.setdefault(utterance["doc_id"], []).append(utterance)
    return sorted(
        docs.items(),
        key=lambda item: item[1][0].get("dialog_date", "") if item[1] else "",
        reverse=True,
    )


def _build_selected_segments(
    documents: list[tuple[str, list[dict[str, Any]]]],
    resolve_path: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    result_segments = []
    for doc_id, doc_utterances in documents:
        sorted_utterances = dedup_and_sort(doc_utterances)
        if not sorted_utterances:
            continue
        min_idx = sorted_utterances[0]["order_index"]
        max_idx = sorted_utterances[-1]["order_index"]
        result_segments.append(
            build_segments(doc_id, [[min_idx, max_idx]], sorted_utterances, resolve_path)
        )
    return result_segments


def _empty_metadata(suggestion: str) -> dict[str, Any]:
    return {
        "truncated": False,
        "total_matches": 0,
        "returned_segments": 0,
        "limit_reason": "no_results",
        "suggestion": suggestion,
    }


def _build_metadata(total_docs: int, returned_segments: int, truncated: bool) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "truncated": truncated,
        "total_matches": total_docs,
        "returned_segments": returned_segments,
    }
    if truncated:
        meta["suggestion"] = "Use limit or narrow down the period to retrieve more transcripts."
    return meta


def _scroll_by_doc_id(doc_id: str) -> list[dict[str, Any]]:
    return reader.fetch_utterances_by_doc_id(doc_id)


def _scroll_by_folder(
    folder_ids: list[str],
    date_from: str | None,
    date_to: str | None,
) -> list[dict[str, Any]]:
    return reader.fetch_utterances_by_folder(folder_ids, date_from, date_to)
