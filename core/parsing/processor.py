import logging
from typing import Any

from core.gemini.llm import generate_summary_and_facts
from core.parsing.windowing import generate_windows


def _date_to_num(dialog_date: str) -> int | None:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for range filtering."""
    try:
        return int(dialog_date.replace("-", ""))
    except (ValueError, AttributeError):
        return None

logger = logging.getLogger(__name__)


def process_document(
    utterances: list[dict[str, Any]],
    doc_id: str,
    version: str,
    client_name: str,
    dialog_date: str,
    root_folder_id: str,
    existing_summary_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Orchestrates: utterances → windows → summaries → facts.

    Returns (utterance_payloads, summaries, facts).
    Idempotency: windows whose summary_key is in existing_summary_keys are skipped.
    Fail-soft: a summary failure skips that window; a facts failure skips only facts.
    Utterances are always returned regardless of summary/facts errors.
    """
    if existing_summary_keys is None:
        existing_summary_keys = set()

    dialog_date_num = _date_to_num(dialog_date)
    utterance_payloads = _build_utterance_payloads(
        utterances, doc_id, version, client_name, dialog_date, dialog_date_num, root_folder_id
    )

    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []

    for window in generate_windows(utterances):
        center_index = window["center_index"]
        summary_key = f"{doc_id}:{center_index}:{version}"

        if summary_key in existing_summary_keys:
            logger.debug("skip existing summary_key=%s", summary_key)
            continue

        window_utterances = window["utterances"]

        try:
            summary_text, facts_list = generate_summary_and_facts(window_utterances)
        except Exception:
            logger.warning("llm failed for summary_key=%s, skipping window", summary_key, exc_info=True)
            continue

        summaries.append({
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
        })

        for fact_text in facts_list:
            facts.append({
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
            })

    return utterance_payloads, summaries, facts


def _build_utterance_payloads(
    utterances: list[dict[str, Any]],
    doc_id: str,
    version: str,
    client_name: str,
    dialog_date: str,
    dialog_date_num: int | None,
    root_folder_id: str,
) -> list[dict[str, Any]]:
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
