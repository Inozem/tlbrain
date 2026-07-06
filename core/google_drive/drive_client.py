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


def scan_root_folder() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Top-down BFS from ROOT_FOLDER.

    Returns (live_folders, live_docs):
      live_folders — [{id, name, parent_id}] for every folder found (excluding ROOT)
      live_docs    — [{doc_id, name, parent_id, createdTime, modifiedTime}]
    Cycle detection via visited set; depth is unlimited.
    """
    service = build_drive_service()
    root_folder_id = get_root_folder_id()

    live_folders: list[dict[str, Any]] = []
    live_docs: list[dict[str, Any]] = []
    visited: set[str] = {root_folder_id}
    queue: list[tuple[str, str]] = [(root_folder_id, root_folder_id)]

    logger.info("Starting Drive scan from root %s", root_folder_id)

    while queue:
        folder_id, parent_id = queue.pop(0)

        page_token = None
        while True:
            kwargs: dict[str, Any] = {
                "q": (
                    f"'{folder_id}' in parents"
                    " and mimeType='application/vnd.google-apps.folder'"
                    " and trashed=false"
                ),
                "fields": "nextPageToken,files(id,name)",
                "pageSize": 1000,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = service.files().list(**kwargs).execute()
            for sub in response.get("files", []):
                sub_id = sub["id"]
                if sub_id in visited:
                    logger.warning("Cycle detected: skipping folder %s", sub_id)
                    continue
                visited.add(sub_id)
                live_folders.append({"id": sub_id, "name": sub["name"], "parent_id": folder_id})
                queue.append((sub_id, folder_id))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        page_token = None
        while True:
            kwargs = {
                "q": (
                    f"'{folder_id}' in parents"
                    " and mimeType='application/vnd.google-apps.document'"
                    " and trashed=false"
                ),
                "fields": "nextPageToken,files(id,name,createdTime,modifiedTime)",
                "pageSize": 1000,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = service.files().list(**kwargs).execute()
            for file in response.get("files", []):
                live_docs.append({
                    "doc_id": file["id"],
                    "name": file["name"],
                    "parent_id": folder_id,
                    "createdTime": file["createdTime"],
                    "modifiedTime": file["modifiedTime"],
                })
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    logger.info("Drive scan complete — folders=%d docs=%d", len(live_folders), len(live_docs))
    return live_folders, live_docs


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


def get_file_parent_folder_id(doc_id: str) -> tuple[str | None, bool, str | None]:
    """Return (parent_folder_id, is_trashed, file_name) for a Drive file.

    is_trashed=True means the file is in trash and should be treated as deleted.
    """
    service = build_drive_service()
    file = service.files().get(fileId=doc_id, fields="name,parents,trashed").execute()
    logger.info("Drive file metadata for %s: trashed=%s, parents=%s", doc_id, file.get("trashed"), file.get("parents"))
    if file.get("trashed"):
        return None, True, None
    parents = file.get("parents", [])
    return (parents[0] if parents else None), False, file.get("name")


def get_folder_info(folder_id: str) -> tuple[str | None, str | None]:
    """Return (name, parent_folder_id) of a Drive folder — source of truth for client_name.

    The parent lets the sync path verify the folder is a direct child of ROOT (a valid
    client folder) without consulting the clients collection, which can be stale mid-rename.
    """
    service = build_drive_service()
    folder = service.files().get(fileId=folder_id, fields="name,parents").execute()
    parents = folder.get("parents", [])
    return folder.get("name"), (parents[0] if parents else None)


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


def rename_folder(folder_id: str, new_name: str) -> None:
    service = build_drive_service_rw()
    service.files().update(
        fileId=folder_id,
        body={"name": new_name},
        fields="id",
    ).execute()
    logger.info("Renamed folder %s to '%s'", folder_id, new_name)


def create_folder(name: str, parent_id: str) -> str:
    """Create a new folder in Drive under parent_id. Returns the new folder_id."""
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
    """Move folder_id to new_parent_id in Drive."""
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
