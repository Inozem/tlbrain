"""Transitional Firestore facade for current Drive workflows.

Canonical transcript, folder, and sync-state operations live under
``core.integrations.firestore``. This module keeps only operations still used by
runtime Drive workflows, along with compatibility wrappers for current callers.
Historical migration sources may contain older imports; they are not supported by
the runtime facade.
"""

from google.cloud import firestore

from core.config import get_root_folder_id
from core.integrations.firestore.client import get_client
from core.integrations.firestore.transcripts import (
    aggregate_transcripts_by_folder,
    delete_queued_placeholder,
    mark_download_error,
    mark_downloading,
    write_queued,
)
from core.integrations.firestore import folders as folder_repository
from core.integrations.firestore import transcripts as transcript_repository
from core.integrations.google_drive.folders import (
    expand_subtree as _expand_subtree,
    resolve_folder_path as _resolve_folder_path,
)


def _get_db() -> firestore.Client:
    return get_client()


def upsert_folder(folder_id: str, name: str, parent_id: str | None) -> None:
    """Register or update a folder in folders/{folder_id}."""
    folder_repository.upsert_folder(folder_id, name, parent_id, _get_db())


def get_folder_by_id(folder_id: str) -> dict | None:
    """Return folders/{folder_id} data, or None if not found."""
    return folder_repository.get_folder_by_id(folder_id, _get_db())


def get_all_folders() -> dict[str, dict]:
    """Return {folder_id: data} for all registered folders."""
    return folder_repository.get_all_folders(_get_db())


def expand_subtree(folder_id: str) -> list[str]:
    """Return folder_id plus all its descendants from the folders collection."""
    return _expand_subtree(folder_id, get_all_folders())


def orphan_folder_children(folder_id: str) -> int:
    """Clear parent_id for all subfolders that are direct children of folder_id."""
    return folder_repository.orphan_folder_children(folder_id, _get_db())


def orphan_transcript_children(folder_id: str) -> int:
    """Clear parent_id for all transcript_index docs that are direct children of folder_id."""
    return transcript_repository.orphan_transcript_children(folder_id, _get_db())


def delete_folder(folder_id: str) -> None:
    """Remove a folder record from the folders collection."""
    folder_repository.delete_folder(folder_id, _get_db())


def resolve_folder_path(path: list[str]) -> list[str]:
    """Resolve a folder path, returning all matching IDs at its final segment."""
    if not path:
        return []

    root_folder_id = get_root_folder_id()
    all_folders = get_all_folders()
    return _resolve_folder_path(path, all_folders, root_folder_id)
