import logging
import os
import re
from typing import Any

import google.auth
from googleapiclient.discovery import build


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def build_drive_service():
    logger.info("Building Google Drive service")

    credentials, _ = google.auth.default(scopes=SCOPES)

    return build("drive", "v3", credentials=credentials)


def get_root_folder_id() -> str:
    root_folder_url = os.getenv("ROOT_FOLDER_URL", "")

    logger.info("Reading ROOT_FOLDER_URL")

    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", root_folder_url)

    if not match:
        logger.error("Invalid ROOT_FOLDER_URL")
        raise ValueError("Invalid ROOT_FOLDER_URL")

    root_folder_id = match.group(1)

    logger.info("Resolved root folder id: %s", root_folder_id)

    return root_folder_id


def scan_root_folder() -> list[dict[str, Any]]:
    logger.info("Starting Google Drive scan")

    service = build_drive_service()
    root_folder_id = get_root_folder_id()

    folders = service.files().list(
        q=f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'",
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
                f"and mimeType="
                f"'application/vnd.openxmlformats-officedocument.wordprocessingml.document'"
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
