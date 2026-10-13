"""Folder use cases coordinating Google Drive and Firestore."""

from typing import Any

from core.config import get_root_folder_id
from core.integrations.firestore import folders as folder_repository
from core.integrations.firestore.transcripts import aggregate_transcripts_by_folder
from core.integrations.google_drive import client as drive_client
from core.integrations.google_drive.folders import (
    build_folder_tree,
    expand_subtree,
    resolve_folder_path,
)


class FolderOperationError(Exception):
    """Base class for expected folder-domain failures."""


class FolderNotFound(FolderOperationError):
    def __init__(self, path: list[str], role: str):
        self.path = path
        self.role = role
        super().__init__("/".join(path))


class AmbiguousFolderPath(FolderOperationError):
    def __init__(self, path: list[str], role: str, count: int):
        self.path = path
        self.role = role
        self.count = count
        super().__init__("/".join(path))


class FolderNameConflict(FolderOperationError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(name)


class FolderSubtreeMove(FolderOperationError):
    pass


def get_or_create_folder(drive: Any, parent_id: str, name: str) -> str:
    """Return the first matching Drive folder, creating it when absent.

    ``drive`` is supplied by the caller so connector-specific authentication and
    transport policy remain owned by that caller.
    """
    results = drive.files().list(
        q=f"name={name!r} and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false",
        fields="files(id)",
        pageSize=1,
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = drive.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return folder["id"]


def get_unassigned_folder_id(drive: Any, root_folder_id: str) -> str:
    """Get or create the root ``_unassigned`` folder and register its Drive ID."""
    all_folders = folder_repository.get_all_folders()
    for folder_id, data in all_folders.items():
        if data.get("name") == "_unassigned" and data.get("parent_id") == root_folder_id:
            return folder_id

    folder_id = get_or_create_folder(drive, root_folder_id, "_unassigned")
    folder_repository.upsert_folder(folder_id, "_unassigned", root_folder_id)
    return folder_id


def _resolve(path: list[str], role: str) -> str:
    root_id = get_root_folder_id()
    matches = resolve_folder_path(path, folder_repository.get_all_folders(), root_id)
    if not matches:
        raise FolderNotFound(path, role)
    if len(matches) > 1:
        raise AmbiguousFolderPath(path, role, len(matches))
    return matches[0]


def list_folders() -> dict[str, Any]:
    """Build the folder tree and transcript aggregates from Firestore."""
    root_folder_id = get_root_folder_id()
    all_folders = folder_repository.get_all_folders()
    aggregates = aggregate_transcripts_by_folder()
    return build_folder_tree(all_folders, root_folder_id, aggregates)


def create_folder(name: str, parent_path: list[str] | None = None) -> str:
    """Create a sibling-unique folder and register it under its Drive ID."""
    parent_id = _resolve(parent_path, "parent") if parent_path else get_root_folder_id()
    if folder_repository.folder_name_exists(name, parent_id):
        raise FolderNameConflict(name)
    folder_id = drive_client.create_folder(name, parent_id)
    folder_repository.upsert_folder(folder_id, name, parent_id)
    return folder_id


def move_folder(folder_path: list[str], new_parent_path: list[str] | None = None) -> tuple[str, str]:
    """Move one uniquely addressed folder and update its Firestore parent."""
    folder_id = _resolve(folder_path, "folder")
    new_parent_id = (
        _resolve(new_parent_path, "parent")
        if new_parent_path
        else get_root_folder_id()
    )
    all_folders = folder_repository.get_all_folders()
    if new_parent_id in expand_subtree(folder_id, all_folders):
        raise FolderSubtreeMove()

    folder_data = folder_repository.get_folder_by_id(folder_id) or {}
    name = folder_data.get("name", "")
    drive_client.move_folder_in_drive(folder_id, new_parent_id)
    folder_repository.upsert_folder(folder_id, name, new_parent_id)
    return folder_id, new_parent_id


def rename_folder(folder_path: list[str], new_name: str) -> str:
    """Rename one uniquely addressed folder and update its Firestore record."""
    folder_id = _resolve(folder_path, "folder")
    folder_data = folder_repository.get_folder_by_id(folder_id) or {}
    parent_id = folder_data.get("parent_id", "")
    if folder_repository.folder_name_exists(new_name, parent_id):
        raise FolderNameConflict(new_name)
    drive_client.rename_folder(folder_id, new_name)
    folder_repository.upsert_folder(folder_id, new_name, parent_id)
    return folder_id

