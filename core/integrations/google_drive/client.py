"""Google Drive API client and folder mutation operations."""

import logging
import os
from typing import Any

import google_auth_httplib2
import httplib2
from googleapiclient.discovery import build

from core.integrations.google_auth.credentials import credentials

logger = logging.getLogger(__name__)

SCOPES_RO = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_RW = ["https://www.googleapis.com/auth/drive"]


def _build_http(credentials) -> google_auth_httplib2.AuthorizedHttp:
    return google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))


def _build_credentials(scopes: list[str]):
    if os.environ.get("GOOGLE_REFRESH_TOKEN"):
        logger.info("Drive auth: using user OAuth (GOOGLE_REFRESH_TOKEN)")
    return credentials(scopes)


def build_drive_service():
    logger.info("Building Google Drive service")
    return build("drive", "v3", http=_build_http(_build_credentials(SCOPES_RO)))


def build_drive_service_rw():
    return build("drive", "v3", http=_build_http(_build_credentials(SCOPES_RW)))


def create_folder(name: str, parent_id: str) -> str:
    """Create a Drive folder and return its stable Drive ID."""
    service = build_drive_service_rw()
    folder = service.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    logger.info("Created folder %r under %s: %s", name, parent_id, folder["id"])
    return folder["id"]


def move_folder_in_drive(folder_id: str, new_parent_id: str) -> None:
    """Move a Drive folder to ``new_parent_id``, removing its current parents."""
    service = build_drive_service_rw()
    file = service.files().get(fileId=folder_id, fields="parents").execute()
    old_parents = ",".join(file.get("parents", []))
    service.files().update(
        fileId=folder_id,
        addParents=new_parent_id,
        removeParents=old_parents,
        fields="id,parents",
    ).execute()
    logger.info("Moved folder %s → parent %s", folder_id, new_parent_id)


def rename_folder(folder_id: str, new_name: str) -> None:
    """Rename a Drive folder."""
    service = build_drive_service_rw()
    service.files().update(
        fileId=folder_id,
        body={"name": new_name},
        fields="id",
    ).execute()
    logger.info("Renamed folder %s to '%s'", folder_id, new_name)

