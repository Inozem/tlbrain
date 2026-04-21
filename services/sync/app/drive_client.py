import os
import re
from typing import Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def build_drive_service():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    credentials = Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES,
    )

    return build("drive", "v3", credentials=credentials)


def get_root_folder_id() -> str:
    root_folder_url = os.getenv("ROOT_FOLDER_URL", "")

    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", root_folder_url)

    if not match:
        raise ValueError("Invalid ROOT_FOLDER_URL")

    return match.group(1)


def scan_root_folder() -> list[dict[str, Any]]:
    service = build_drive_service()
    root_folder_id = get_root_folder_id()

    folders = service.files().list(
        q=f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'",
        fields="files(id,name)",
    ).execute()["files"]

    results: list[dict[str, Any]] = []

    for folder in folders:
        folder_id = folder["id"]
        client_name = folder["name"]

        files = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'",
            fields="files(id,name,createdTime,modifiedTime)",
        ).execute()["files"]

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

    return results
