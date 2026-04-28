from typing import Any


def build_segments(
    doc_id: str,
    merged_ranges: list[list[int]],
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Groups sorted, deduped utterances into segments based on merged_ranges.
    Returns the final structure for MCP context output.
    """
    client_name = utterances[0]["client_name"] if utterances else ""
    dialog_date = utterances[0]["dialog_date"] if utterances else ""

    segments = []
    for start, end in merged_ranges:
        dialog = [
            {
                "speaker": u["speaker"],
                "text": u["text"],
                "order_index": u["order_index"],
            }
            for u in utterances
            if start <= u["order_index"] <= end
        ]
        if dialog:
            segments.append({
                "range": [start, end],
                "dialog": dialog,
            })

    return {
        "doc_id": doc_id,
        "client_name": client_name,
        "dialog_date": dialog_date,
        "segments": segments,
    }
