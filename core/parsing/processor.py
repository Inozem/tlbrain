import logging
from typing import Any, Generator

from core.gemini.llm import GeminiEmptyResponseError, generate_summary_and_facts
from core.parsing.windowing import generate_windows


def _date_to_num(dialog_date: str) -> int | None:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for range filtering."""
    try:
        return int(dialog_date.replace("-", ""))
    except (ValueError, AttributeError):
        return None

logger = logging.getLogger(__name__)


def build_utterance_payloads(
    utterances: list[dict[str, Any]],
    doc_id: str,
    version: str,
    client_name: str,
    dialog_date: str,
    root_folder_id: str,
) -> list[dict[str, Any]]:
    dialog_date_num = _date_to_num(dialog_date)
    return [
        {
            "type": "utterance",
            "root_folder_id": root_folder_id,
            "doc_id": doc_id,
            "version": version,
            "client_name": client_name,
            "dialog_date": dialog_date,
            "dialog_date_num": dialog_date_num,
            "speaker": u["speaker"],
            "text": u["text"],
            "order_index": u["order_index"],
        }
        for u in utterances
    ]


def iter_windows(
    utterances: list[dict[str, Any]],
    doc_id: str,
    version: str,
    client_name: str,
    dialog_date: str,
    root_folder_id: str,
    existing_summary_keys: set[str] | None = None,
    allowed_center_indexes: set[int] | None = None,
    skipped_utterances: list[int] | None = None,
) -> Generator[tuple[dict[str, Any], list[dict[str, Any]]], None, None]:
    """Generator: yield (summary_dict, facts_list) one window at a time."""
    if existing_summary_keys is None:
        existing_summary_keys = set()

    dialog_date_num = _date_to_num(dialog_date)

    for window in generate_windows(utterances):
        center_index = window["center_index"]

        if allowed_center_indexes is not None and center_index not in allowed_center_indexes:
            continue

        summary_key = f"{doc_id}:{center_index}:{version}"

        if summary_key in existing_summary_keys:
            logger.debug("skip existing summary_key=%s", summary_key)
            continue

        try:
            summary_text, facts_list = generate_summary_and_facts(window["utterances"])
        except GeminiEmptyResponseError:
            logger.warning("Skipping window center_index=%d for %s: Gemini returned no content", center_index, doc_id)
            if skipped_utterances is not None:
                skipped_utterances.extend(u["order_index"] for u in window["utterances"])
            continue

        summary = {
            "type": "summary",
            "root_folder_id": root_folder_id,
            "summary_id": summary_key,
            "text": summary_text,
            "doc_id": doc_id,
            "center_index": center_index,
            "covered_range": window["covered_range"],
            "client_name": client_name,
            "dialog_date": dialog_date,
            "dialog_date_num": dialog_date_num,
            "version": version,
        }

        facts = [
            {
                "type": "fact",
                "text": fact_text,
                "root_folder_id": root_folder_id,
                "doc_id": doc_id,
                "summary_id": summary_key,
                "center_index": center_index,
                "covered_range": window["covered_range"],
                "client_name": client_name,
                "dialog_date": dialog_date,
                "dialog_date_num": dialog_date_num,
                "version": version,
            }
            for fact_text in facts_list
        ]

        yield summary, facts
