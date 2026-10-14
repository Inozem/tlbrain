from typing import Any


def _date_to_num(dialog_date: str) -> int | None:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for range filtering."""
    try:
        return int(dialog_date.replace("-", ""))
    except (ValueError, AttributeError):
        return None


def build_utterance_payloads(
    utterances: list[dict[str, Any]],
    doc_id: str,
    version: str,
    parent_id: str,
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
            "parent_id": parent_id,
            "dialog_date": dialog_date,
            "dialog_date_num": dialog_date_num,
            "speaker": u["speaker"],
            "text": u["text"],
            "order_index": u["order_index"],
        }
        for u in utterances
    ]
