"""Google Drive file metadata and mutation operations."""

import logging

from core.integrations.google_drive.client import build_drive_service, build_drive_service_rw
from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy

logger = logging.getLogger(__name__)

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
SAVE_INTENT_PROPERTY = "tlbrain_save_id"


class DriveCreateError(RuntimeError):
    """Safe create failure with an outcome classification for recovery logic."""

    def __init__(self, outcome: str):
        if outcome not in {"rejected", "unknown"}:
            raise ValueError("invalid Drive create outcome")
        self.outcome = outcome
        message = "Drive document creation was rejected" if outcome == "rejected" else "Drive document creation outcome is unknown"
        super().__init__(message)


def create_native_doc(service, name: str, parent_id: str, save_intent_id: str) -> str:
    """Create one native Doc carrying its immutable save-intent marker.

    The caller supplies the service so it can control the physical request and
    decide how to recover when its outcome is uncertain.
    """
    body = {
        "name": name,
        "mimeType": GOOGLE_DOC_MIME_TYPE,
        "parents": [parent_id],
        "appProperties": {SAVE_INTENT_PROPERTY: save_intent_id},
    }
    try:
        result = service.files().create(body=body, fields="id").execute()
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", None)
        # 408/429 do not establish whether Drive persisted the request.
        outcome = "rejected" if isinstance(status, int) and 400 <= status < 500 and status not in {408, 429} else "unknown"
        raise DriveCreateError(outcome) from None
    if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"].strip():
        raise DriveCreateError("unknown")
    return result["id"]


def _escape_drive_query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_native_docs_by_save_intent(service, save_intent_id: str, expected_parent_id: str) -> list[dict]:
    """Return every valid native Doc matching a save-intent marker.

    This is a read-only recovery query. An empty list is only an observation;
    callers must decide whether another create attempt is safe.
    """
    query_value = _escape_drive_query_literal(save_intent_id)
    query = f"appProperties has {{ key='{SAVE_INTENT_PROPERTY}' and value='{query_value}' }}"
    valid = []
    page_token = None
    while True:
        kwargs = {
            "q": query,
            "spaces": "drive",
            "pageSize": 1000,
            "fields": "nextPageToken,files(id,mimeType,parents,trashed,appProperties)",
        }
        if page_token:
            kwargs["pageToken"] = page_token
        try:
            response = service.files().list(**kwargs).execute()
        except Exception:
            raise RuntimeError("Drive marker lookup is unavailable") from None
        if not isinstance(response, dict) or not isinstance(response.get("files"), list):
            raise RuntimeError("Drive marker lookup returned malformed metadata")
        next_token = response.get("nextPageToken")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            raise RuntimeError("Drive marker lookup returned malformed metadata")
        for candidate in response["files"]:
            if not isinstance(candidate, dict):
                raise RuntimeError("Drive marker lookup returned malformed metadata")
            props = candidate.get("appProperties")
            if (
                isinstance(candidate.get("id"), str)
                and candidate["id"].strip()
                and candidate.get("mimeType") == GOOGLE_DOC_MIME_TYPE
                and candidate.get("trashed") is False
                and candidate.get("parents") == [expected_parent_id]
                and isinstance(props, dict)
                and props.get(SAVE_INTENT_PROPERTY) == save_intent_id
            ):
                valid.append(candidate)
        page_token = next_token
        if not page_token:
            return valid


def read_native_doc_metadata(service, file_id: str) -> dict:
    """Read metadata for a known file ID without accessing document content."""
    try:
        result = service.files().get(
            fileId=file_id,
            fields="id,mimeType,parents,trashed,appProperties,modifiedTime",
        ).execute()
    except Exception:
        raise RuntimeError("Drive document metadata is unavailable") from None
    if not isinstance(result, dict) or result.get("id") != file_id:
        raise RuntimeError("Drive document metadata is unavailable")
    return result


def get_file_metadata(file_id: str, *, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY) -> dict:
    """Return the minimal current metadata needed by guarded write read-back."""
    service = build_drive_service(retry_policy=retry_policy)
    result = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,parents,trashed,appProperties",
    ).execute()
    if not isinstance(result, dict) or result.get("id") != file_id:
        raise RuntimeError("Drive metadata is unavailable")
    return result


def get_file_metadata_if_exists(file_id: str) -> dict | None:
    """Return None only when the Drive resource is absent; hide provider details."""
    from googleapiclient.errors import HttpError
    try:
        return get_file_metadata(file_id)
    except HttpError as exc:
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", None)
        if status == 404:
            return None
        raise RuntimeError("Drive metadata is unavailable") from None
    except Exception:
        raise RuntimeError("Drive metadata is unavailable") from None


def get_file_parent_folder_id(doc_id: str) -> tuple[str | None, bool, str | None]:
    """Return (parent_folder_id, is_trashed, file_name) for a Drive file."""
    service = build_drive_service()
    file = service.files().get(fileId=doc_id, fields="name,parents,trashed").execute()
    logger.info("Drive file metadata for %s: trashed=%s, parents=%s", doc_id, file.get("trashed"), file.get("parents"))
    if file.get("trashed"):
        return None, True, None
    parents = file.get("parents", [])
    return (parents[0] if parents else None), False, file.get("name")


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
