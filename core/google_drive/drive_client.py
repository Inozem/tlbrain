import logging
import os
from typing import Any

import google_auth_httplib2
import httplib2
from googleapiclient.discovery import build

from core.config import get_root_folder_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCOPES_RO = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_RW = ["https://www.googleapis.com/auth/drive"]


def _build_http(credentials) -> google_auth_httplib2.AuthorizedHttp:
    return google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))


def _build_credentials(scopes: list[str]):
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if refresh_token:
        logger.info("Drive auth: using user OAuth (GOOGLE_REFRESH_TOKEN)")
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
        )
        creds.refresh(Request())
        return creds
    raise RuntimeError("GOOGLE_REFRESH_TOKEN is not set — Drive access requires user OAuth credentials")


def build_drive_service():
    logger.info("Building Google Drive service")
    return build("drive", "v3", http=_build_http(_build_credentials(SCOPES_RO)))


def build_drive_service_rw():
    return build("drive", "v3", http=_build_http(_build_credentials(SCOPES_RW)))


def list_client_folders() -> list[dict[str, str]]:
    """Return all client folder names from ROOT_FOLDER. Each item: {id, name}."""
    service = build_drive_service()
    root_folder_id = get_root_folder_id()
    folders = service.files().list(
        q=f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)",
    ).execute()["files"]
    logger.info("Client folders found: %s", len(folders))
    return folders


def scan_root_folder() -> list[dict[str, Any]]:
    logger.info("Starting Google Drive scan")

    service = build_drive_service()
    root_folder_id = get_root_folder_id()

    folders = service.files().list(
        q=f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)",
    ).execute()["files"]

    logger.info("Client folders found: %s", len(folders))

    results: list[dict[str, Any]] = []

    for folder in folders:
        folder_id = folder["id"]
        client_name = folder["name"]

        logger.info(
            "Scanning folder: %s (%s)",
            client_name,
            folder_id,
        )

        files = []
        page_token = None
        while True:
            kwargs = dict(
                q=(
                    f"'{folder_id}' in parents "
                    f"and mimeType='application/vnd.google-apps.document' "
                    f"and trashed=false"
                ),
                fields="nextPageToken,files(id,name,createdTime,modifiedTime)",
                pageSize=1000,
            )
            if page_token:
                kwargs["pageToken"] = page_token
            response = service.files().list(**kwargs).execute()
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info(
            "Files found in %s: %s",
            client_name,
            len(files),
        )

        for file in files:
            results.append(
                {
                    "doc_id": file["id"],
                    "name": file["name"],
                    "client_name": client_name,
                    "createdTime": file["createdTime"],
                    "modifiedTime": file["modifiedTime"],
                }
            )

    logger.info("Total files found: %s", len(results))

    return results


def get_start_page_token() -> str:
    service = build_drive_service()
    return service.changes().getStartPageToken().execute()["startPageToken"]


def get_drive_changes(page_token: str) -> tuple[list[dict], str]:
    """Fetch all changes since page_token. Returns (changes, new_token)."""
    service = build_drive_service()
    changes = []
    while page_token:
        response = service.changes().list(
            pageToken=page_token,
            fields="nextPageToken,newStartPageToken,changes(removed,fileId,file(id,name,mimeType,modifiedTime,parents,trashed))",
            spaces="drive",
            includeItemsFromAllDrives=False,
        ).execute()
        changes.extend(response.get("changes", []))
        page_token = response.get("nextPageToken")
    return changes, response["newStartPageToken"]


def get_file_parent_folder_id(doc_id: str) -> tuple[str | None, bool]:
    """Return (parent_folder_id, is_trashed) for a Drive file.

    is_trashed=True means the file is in trash and should be treated as deleted.
    """
    service = build_drive_service()
    file = service.files().get(fileId=doc_id, fields="parents,trashed").execute()
    logger.info("Drive file metadata for %s: trashed=%s, parents=%s", doc_id, file.get("trashed"), file.get("parents"))
    if file.get("trashed"):
        return None, True
    parents = file.get("parents", [])
    return (parents[0] if parents else None), False


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


def rename_folder(folder_id: str, new_name: str) -> None:
    service = build_drive_service_rw()
    service.files().update(
        fileId=folder_id,
        body={"name": new_name},
        fields="id",
    ).execute()
    logger.info("Renamed folder %s to '%s'", folder_id, new_name)


def create_client_folder(client_name: str) -> tuple[str, bool]:
    """Ensure ROOT_FOLDER/{client_name}/ exists in Drive.

    Returns (folder_id, created) — created=False if folder already existed.
    Drive is the source of truth; existing folder is not an error.
    """
    service = build_drive_service_rw()
    root_folder_id = get_root_folder_id()

    existing = service.files().list(
        q=(
            f"'{root_folder_id}' in parents"
            f" and mimeType='application/vnd.google-apps.folder'"
            f" and name='{client_name}'"
            f" and trashed=false"
        ),
        fields="files(id)",
    ).execute()["files"]

    if existing:
        logger.info("Client folder already exists: %s (%s)", client_name, existing[0]["id"])
        return existing[0]["id"], False

    folder = service.files().create(
        body={
            "name": client_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [root_folder_id],
        },
        fields="id",
    ).execute()

    folder_id = folder["id"]
    logger.info("Created client folder: %s (%s)", client_name, folder_id)
    return folder_id, True
