"""Google Drive API client and folder mutation operations."""

import logging
import os
import re
from typing import Any

import google_auth_httplib2
import httplib2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from core.integrations.google_auth.credentials import credentials
from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy, SINGLE_ATTEMPT_RETRY_POLICY

logger = logging.getLogger(__name__)

SCOPES_RO = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_RW = ["https://www.googleapis.com/auth/drive"]


def _build_http(credentials, *, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY) -> google_auth_httplib2.AuthorizedHttp:
    if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY:
        from core.integrations.google_auth.transport import SingleAttemptHttp
        return google_auth_httplib2.AuthorizedHttp(
            credentials, http=SingleAttemptHttp(timeout=60), max_refresh_attempts=0,
        )
    return google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))


def _build_credentials(scopes: list[str], *, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY):
    if os.environ.get("GOOGLE_REFRESH_TOKEN"):
        logger.info("Drive auth: using user OAuth (GOOGLE_REFRESH_TOKEN)")
    return credentials(scopes, retry_policy=retry_policy)


def _build_trusted_credentials(scopes: list[str], service_name: str):
    """Attribute only trusted one-attempt credential failures to their API service."""
    from core.utils.retry import ProviderBoundaryError, SINGLE_ATTEMPT_RETRY_POLICY
    try:
        return _build_credentials(scopes, retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
    except Exception as error:
        raise ProviderBoundaryError(service_name, error) from error


def build_drive_service(*, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY):
    logger.info("Building Google Drive service")
    kwargs = {"num_retries": 0} if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY else {}
    auth_credentials = (
        _build_trusted_credentials(SCOPES_RO, "google_drive")
        if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY
        else _build_credentials(SCOPES_RO, retry_policy=retry_policy)
    )
    return build("drive", "v3", http=_build_http(
        auth_credentials, retry_policy=retry_policy,
    ), **kwargs)


def build_drive_service_rw(*, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY):
    kwargs = {"num_retries": 0} if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY else {}
    auth_credentials = (
        _build_trusted_credentials(SCOPES_RW, "google_drive")
        if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY
        else _build_credentials(SCOPES_RW, retry_policy=retry_policy)
    )
    return build("drive", "v3", http=_build_http(
        auth_credentials, retry_policy=retry_policy,
    ), **kwargs)


_FOLDER_MIME = "application/vnd.google-apps.folder"
_INTENT_MARKER = re.compile(r"v1:[0-9a-f]{64}\Z")
_DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


def _is_drive_id(value: Any) -> bool:
    return isinstance(value, str) and _DRIVE_ID.fullmatch(value) is not None


def generate_folder_id() -> str:
    """Ask Drive for a pre-generated ID to use in a later folder create."""
    try:
        service = build_drive_service_rw()
        result = service.files().generateIds(count=1, space="drive", type="files", fields="ids").execute()
    except Exception:
        raise RuntimeError("Drive folder ID allocation failed") from None
    if not isinstance(result, dict) or not isinstance(result.get("ids"), list):
        raise RuntimeError("Drive folder ID allocation failed")
    ids = result["ids"]
    if len(ids) != 1 or not _is_drive_id(ids[0]):
        raise RuntimeError("Drive folder ID allocation failed")
    return ids[0]


def create_folder(
    name: str, parent_id: str, *, folder_id: str | None = None, intent_marker: str | None = None,
) -> str:
    """Create a Drive folder and return its stable Drive ID."""
    if (folder_id is None) != (intent_marker is None):
        raise ValueError("stable folder creation requires both folder_id and intent_marker")
    if folder_id is not None:
        if (not _is_drive_id(folder_id)
                or not isinstance(intent_marker, str) or not _INTENT_MARKER.fullmatch(intent_marker)):
            raise ValueError("invalid stable folder creation identity")
        try:
            service = build_drive_service_rw()
        except Exception:
            raise RuntimeError("Drive folder creation outcome is unresolved") from None
        body = {"name": name, "mimeType": _FOLDER_MIME, "parents": [parent_id]}
        body.update({"id": folder_id, "appProperties": {"tlbrain_effect_intent": intent_marker}})
        try:
            folder = service.files().create(body=body, fields="id").execute()
        except HttpError as exc:
            if getattr(exc, "status_code", None) != 409 and getattr(getattr(exc, "resp", None), "status", None) != 409:
                raise RuntimeError("Drive folder creation outcome is unresolved") from None
            try:
                found = service.files().get(
                    fileId=folder_id, fields="id,mimeType,name,parents,trashed,appProperties",
                ).execute()
            except Exception:
                raise RuntimeError("Drive folder creation outcome is unresolved") from None
            if not _matches_planned_folder(found, folder_id, name, parent_id, intent_marker):
                raise RuntimeError("Drive folder creation conflicts with the persisted intent") from None
            return folder_id
        except Exception:
            raise RuntimeError("Drive folder creation outcome is unresolved") from None
        if not isinstance(folder, dict) or folder.get("id") != folder_id:
            raise RuntimeError("Drive folder creation outcome is unresolved")
        return folder_id
    service = build_drive_service_rw()
    body = {"name": name, "mimeType": _FOLDER_MIME, "parents": [parent_id]}
    folder = service.files().create(
        body=body,
        fields="id",
    ).execute()
    logger.info("Created folder %r under %s: %s", name, parent_id, folder["id"])
    return folder["id"]


def _matches_planned_folder(file: Any, folder_id: str, name: str, parent_id: str, marker: str) -> bool:
    return (
        isinstance(file, dict)
        and file.get("id") == folder_id
        and file.get("mimeType") == _FOLDER_MIME
        and file.get("name") == name
        and isinstance(file.get("parents"), list)
        and len(file["parents"]) == 1
        and file["parents"][0] == parent_id
        and file.get("trashed") is False
        and isinstance(file.get("appProperties"), dict)
        and file["appProperties"].get("tlbrain_effect_intent") == marker
    )


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

