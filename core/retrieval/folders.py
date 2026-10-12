from typing import Any

from core.config import get_root_folder_id
from core.google_drive.firestore import aggregate_transcripts_by_folder, get_all_folders
from core.integrations.google_drive.folders import build_folder_tree


def list_folders() -> dict[str, Any]:
    """Build folder tree with transcript aggregates from Firestore."""
    root_folder_id = get_root_folder_id()
    all_folders = get_all_folders()
    aggregates = aggregate_transcripts_by_folder()
    return build_folder_tree(all_folders, root_folder_id, aggregates)
