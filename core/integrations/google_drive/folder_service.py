"""Folder use cases coordinating Google Drive and Firestore."""

from typing import Any, Callable
import re

from core.config import get_root_folder_id
from core.integrations.firestore import folders as folder_repository
from core.integrations.firestore.transcripts import aggregate_transcripts_by_folder
from core.integrations.google_drive import client as drive_client
from core.integrations.google_drive.folders import (
    build_folder_tree,
    expand_subtree,
    resolve_folder_path,
)


def resolve_registered_folder_path(path: list[str], db=None) -> list[str]:
    """Resolve a path against the registered folder snapshot, including duplicates."""
    if not path:
        return []
    root_folder_id = get_root_folder_id()
    all_folders = folder_repository.get_all_folders() if db is None else folder_repository.get_all_folders(db)
    return resolve_folder_path(path, all_folders, root_folder_id)


def expand_registered_subtree(folder_id: str, db=None) -> list[str]:
    """Expand a registered folder subtree in breadth-first repository order."""
    all_folders = folder_repository.get_all_folders() if db is None else folder_repository.get_all_folders(db)
    return expand_subtree(folder_id, all_folders)


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


_STABLE_INTENT_MARKER = re.compile(r"v1:[0-9a-f]{64}\Z")


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


def _resolve(path: list[str], role: str, db=None) -> str:
    root_id = get_root_folder_id()
    all_folders = folder_repository.get_all_folders() if db is None else folder_repository.get_all_folders(db)
    matches = resolve_folder_path(path, all_folders, root_id)
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


def create_folder(
    name: str, parent_path: list[str] | None = None, *,
    folder_id: str | None = None, intent_marker: str | None = None, db=None,
) -> str:
    """Create a sibling-unique folder and register it under its Drive ID."""
    parent_id = (_resolve(parent_path, "parent") if db is None else _resolve(parent_path, "parent", db)) \
        if parent_path else get_root_folder_id()
    stable = folder_id is not None or intent_marker is not None
    if stable and (folder_id is None or intent_marker is None):
        raise ValueError("stable folder creation requires both folder_id and intent_marker")
    if stable and (
        not isinstance(folder_id, str) or not folder_id
        or len(folder_id) > 256
        or not all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in folder_id)
        or not isinstance(intent_marker, str) or not _STABLE_INTENT_MARKER.fullmatch(intent_marker)
    ):
        raise ValueError("invalid stable folder creation identity")
    try:
        if stable:
            duplicate = (folder_repository.folder_name_exists(name, parent_id, exclude_id=folder_id)
                         if db is None else
                         folder_repository.folder_name_exists(name, parent_id, db, exclude_id=folder_id))
        else:
            duplicate = (folder_repository.folder_name_exists(name, parent_id) if db is None
                         else folder_repository.folder_name_exists(name, parent_id, db))
    except Exception:
        if stable:
            raise RuntimeError("Drive folder registration outcome is unresolved") from None
        raise
    if duplicate:
        raise FolderNameConflict(name)
    if stable:
        folder_id = drive_client.create_folder(name, parent_id, folder_id=folder_id, intent_marker=intent_marker)
    else:
        folder_id = drive_client.create_folder(name, parent_id)
    if stable:
        try:
            if db is None:
                folder_repository.upsert_folder(folder_id, name, parent_id)
            else:
                folder_repository.upsert_folder(folder_id, name, parent_id, db)
        except Exception:
            raise RuntimeError("Drive folder registration outcome is unresolved") from None
    else:
        if db is None:
            folder_repository.upsert_folder(folder_id, name, parent_id)
        else:
            folder_repository.upsert_folder(folder_id, name, parent_id, db)
    return folder_id


def prepare_create_folder(
    name: str, parent_path: list[str] | None = None, *, exclude_id: str | None = None, db=None,
) -> str:
    """Resolve the immutable parent and enforce the sibling-name constraint."""
    parent_id = (_resolve(parent_path, "parent") if db is None else _resolve(parent_path, "parent", db)) \
        if parent_path else get_root_folder_id()
    validate_folder_name_available(name, parent_id, exclude_id=exclude_id, db=db)
    return parent_id


def validate_folder_name_available(
    name: str, parent_id: str, *, exclude_id: str | None = None, db=None,
) -> None:
    """Raise the canonical sibling conflict before a create or rename write."""
    duplicate = (folder_repository.folder_name_exists(name, parent_id, exclude_id=exclude_id)
                 if db is None else folder_repository.folder_name_exists(name, parent_id, db, exclude_id=exclude_id))
    if duplicate:
        raise FolderNameConflict(name)


def move_folder(folder_path: list[str], new_parent_path: list[str] | None = None, *, db=None) -> tuple[str, str]:
    """Move one uniquely addressed folder and update its Firestore parent."""
    folder_id, new_parent_id = prepare_move_folder(folder_path, new_parent_path, db=db)
    return apply_move_folder(folder_id, new_parent_id, db=db)


def prepare_move_folder(
    folder_path: list[str], new_parent_path: list[str] | None = None, *, db=None,
) -> tuple[str, str]:
    """Resolve a move's stable folder and parent IDs and reject subtree moves."""
    folder_id = _resolve(folder_path, "folder") if db is None else _resolve(folder_path, "folder", db)
    if new_parent_path:
        new_parent_id = (_resolve(new_parent_path, "parent") if db is None
                         else _resolve(new_parent_path, "parent", db))
    else:
        new_parent_id = get_root_folder_id()
    all_folders = folder_repository.get_all_folders() if db is None else folder_repository.get_all_folders(db)
    if new_parent_id in expand_subtree(folder_id, all_folders):
        raise FolderSubtreeMove()
    return folder_id, new_parent_id


def apply_move_folder(
    folder_id: str, new_parent_id: str, *, db=None,
    before_bookkeeping: Callable[[], None] | None = None,
) -> tuple[str, str]:
    """Apply a move to previously resolved folder IDs."""
    folder_data = (folder_repository.get_folder_by_id(folder_id) if db is None
                   else folder_repository.get_folder_by_id(folder_id, db)) or {}
    name = folder_data.get("name", "")
    drive_client.move_folder_in_drive(folder_id, new_parent_id)
    if before_bookkeeping is not None:
        before_bookkeeping()
    if db is None:
        folder_repository.upsert_folder(folder_id, name, new_parent_id)
    else:
        folder_repository.upsert_folder(folder_id, name, new_parent_id, db)
    return folder_id, new_parent_id


def rename_folder(folder_path: list[str], new_name: str) -> str:
    """Rename one uniquely addressed folder and update its Firestore record."""
    folder_id = _resolve(folder_path, "folder")
    return apply_rename_folder(folder_id, new_name)


def apply_rename_folder(
    folder_id: str, new_name: str, *, validate_name: bool = True, db=None,
    before_bookkeeping: Callable[[], None] | None = None,
) -> str:
    """Rename a previously resolved folder after validating sibling uniqueness."""
    folder_data = (folder_repository.get_folder_by_id(folder_id) if db is None
                   else folder_repository.get_folder_by_id(folder_id, db)) or {}
    parent_id = folder_data.get("parent_id", "")
    if validate_name:
        duplicate = (folder_repository.folder_name_exists(new_name, parent_id) if db is None
                     else folder_repository.folder_name_exists(new_name, parent_id, db))
        if duplicate:
            raise FolderNameConflict(new_name)
    drive_client.rename_folder(folder_id, new_name)
    if before_bookkeeping is not None:
        before_bookkeeping()
    if db is None:
        folder_repository.upsert_folder(folder_id, new_name, parent_id)
    else:
        folder_repository.upsert_folder(folder_id, new_name, parent_id, db)
    return folder_id

