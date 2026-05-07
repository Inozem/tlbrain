import logging
from typing import Any

import google.auth
from googleapiclient.discovery import build

from core.config import get_root_folder_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCOPES_RO = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_RW = ["https://www.googleapis.com/auth/drive"]


def build_drive_service():
    logger.info("Building Google Drive service")

    credentials, _ = google.auth.default(scopes=SCOPES_RO)

    return build("drive", "v3", credentials=credentials)


def build_drive_service_rw():
    credentials, _ = google.auth.default(scopes=SCOPES_RW)
    return build("drive", "v3", credentials=credentials)


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

        files = service.files().list(
            q=(
                f"'{folder_id}' in parents "
                f"and mimeType='application/vnd.google-apps.document'"
            ),
            fields="files(id,name,createdTime,modifiedTime)",
        ).execute()["files"]

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
