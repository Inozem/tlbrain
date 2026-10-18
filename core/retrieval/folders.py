from typing import Any

def list_folders() -> dict[str, Any]:
    """Compatibility wrapper for the canonical Google Drive folder service."""
    from core.integrations.google_drive.folder_service import list_folders as service_list_folders

    return service_list_folders()
