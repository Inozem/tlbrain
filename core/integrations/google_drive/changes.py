"""Read-only access to the Google Drive Changes API."""

from core.integrations.google_drive.client import build_drive_service


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
