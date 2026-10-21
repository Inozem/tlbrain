"""Google Drive file metadata and mutation operations."""

import logging

from core.integrations.google_drive.client import build_drive_service, build_drive_service_rw

logger = logging.getLogger(__name__)


def get_file_metadata(file_id: str) -> dict:
    """Return the minimal current metadata needed by guarded write read-back."""
    service = build_drive_service()
    result = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,parents,trashed,appProperties",
    ).execute()
    if not isinstance(result, dict) or result.get("id") != file_id:
        raise RuntimeError("Drive metadata is unavailable")
    return result


def get_file_metadata_if_exists(file_id: str) -> dict | None:
    """Return None only when the Drive resource is absent; hide provider details."""
    from googleapiclient.errors import HttpError
    try:
        return get_file_metadata(file_id)
    except HttpError as exc:
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", None)
        if status == 404:
            return None
        raise RuntimeError("Drive metadata is unavailable") from None
    except Exception:
        raise RuntimeError("Drive metadata is unavailable") from None


def get_file_parent_folder_id(doc_id: str) -> tuple[str | None, bool, str | None]:
    """Return (parent_folder_id, is_trashed, file_name) for a Drive file."""
    service = build_drive_service()
    file = service.files().get(fileId=doc_id, fields="name,parents,trashed").execute()
    logger.info("Drive file metadata for %s: trashed=%s, parents=%s", doc_id, file.get("trashed"), file.get("parents"))
    if file.get("trashed"):
        return None, True, None
    parents = file.get("parents", [])
    return (parents[0] if parents else None), False, file.get("name")


def move_file_to_folder(doc_id: str, new_folder_id: str) -> None:
    """Move a Drive file to new_folder_id, removing all current parents."""
    service = build_drive_service_rw()
    file = service.files().get(fileId=doc_id, fields="parents").execute()
    old_parents = ",".join(file.get("parents", []))
    service.files().update(
        fileId=doc_id,
        addParents=new_folder_id,
        removeParents=old_parents,
        fields="id",
    ).execute()
    logger.info("Moved file %s to folder %s", doc_id, new_folder_id)


def rename_file(file_id: str, new_name: str) -> None:
    service = build_drive_service_rw()
    service.files().update(
        fileId=file_id,
        body={"name": new_name},
        fields="id",
    ).execute()
    logger.info("Renamed file %s to '%s'", file_id, new_name)
