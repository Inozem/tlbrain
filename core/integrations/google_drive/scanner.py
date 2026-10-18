"""Read-only scans of the configured Google Drive folder tree."""

import logging
from typing import Any

from core.config import get_root_folder_id
from core.integrations.google_drive.client import build_drive_service

logger = logging.getLogger(__name__)


def scan_root_folder() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Top-down BFS from ROOT_FOLDER, returning live folders and Google Docs."""
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
