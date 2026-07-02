from typing import Any, Callable


def make_folder_path_resolver(all_folders: dict) -> Callable[[str], str]:
    """Build a resolver that walks parent_id chain to produce a human-readable path."""
    def resolve(parent_id: str) -> str:
        path_parts: list[str] = []
        fid = parent_id
        visited: set[str] = set()
        while fid and fid not in visited:
            visited.add(fid)
            data = all_folders.get(fid, {})
            name = data.get("name", "")
            if name:
                path_parts.append(name)
            fid = data.get("parent_id", "")
        return "/".join(reversed(path_parts))
    return resolve


def build_segments(
    doc_id: str,
    merged_ranges: list[list[int]],
    utterances: list[dict[str, Any]],
    resolve_path: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Groups sorted, deduped utterances into segments based on merged_ranges.
    Returns the final structure for MCP context output.
    """
    parent_id = utterances[0].get("parent_id", "") if utterances else ""
    dialog_date = utterances[0]["dialog_date"] if utterances else ""
    folder_path = resolve_path(parent_id) if resolve_path and parent_id else parent_id

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
        "folder_path": folder_path,
        "dialog_date": dialog_date,
        "segments": segments,
    }
