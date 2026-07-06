from typing import Any

from core.config import get_root_folder_id
from core.google_drive.firestore import aggregate_transcripts_by_folder, get_all_folders


def list_folders() -> dict[str, Any]:
    """Build folder tree with transcript aggregates from Firestore."""
    root_folder_id = get_root_folder_id()
    all_folders = get_all_folders()
    aggregates = aggregate_transcripts_by_folder()

    children: dict[str, list[str]] = {}
    for fid, data in all_folders.items():
        pid = data.get("parent_id")
        if pid:
            children.setdefault(pid, []).append(fid)

    unassigned_folder_id: str | None = None
    for fid, data in all_folders.items():
        if data.get("name") == "_unassigned":
            unassigned_folder_id = fid
            break

    def build_node(folder_id: str) -> dict[str, Any]:
        data = all_folders[folder_id]
        agg = aggregates.get(folder_id, {})
        node: dict[str, Any] = {
            "name": data["name"],
            "dialog_count": agg.get("count", 0),
            "last_dialog_date": agg.get("last_date") or None,
            "last_dialog_doc_id": agg.get("last_doc_id") or None,
            "children": [
                build_node(cid)
                for cid in sorted(
                    children.get(folder_id, []),
                    key=lambda x: all_folders[x].get("name", ""),
                )
            ],
        }
        return node

    root_folders = sorted(
        [fid for fid, data in all_folders.items() if data.get("parent_id") == root_folder_id],
        key=lambda x: all_folders[x].get("name", ""),
    )

    result: dict[str, Any] = {"folders": [build_node(fid) for fid in root_folders]}

    if unassigned_folder_id:
        unassigned_count = aggregates.get(unassigned_folder_id, {}).get("count", 0)
        if unassigned_count > 0:
            result["suggestion"] = (
                f"{unassigned_count} transcript(s) are unassigned. "
                f"Call get_transcript with folder_path=[\"_unassigned\"] to review them, "
                f"then move each using move_transcript(doc_id=..., folder_path=[...])."
            )

    return result
