"""Firestore storage for Drive synchronization state."""

from google.cloud import firestore

from core.integrations.firestore.client import (
    DRIVE_SYNC_TOKEN_DOC,
    TOKENS_COLLECTION,
    get_client,
)


def get_drive_sync_token() -> str | None:
    """Return the stored Drive changes page token, if present."""
    document = get_client().collection(TOKENS_COLLECTION).document(DRIVE_SYNC_TOKEN_DOC).get()
    if not document.exists:
        return None
    return document.to_dict().get("page_token")


def set_drive_sync_token(token: str) -> None:
    """Replace the stored Drive cursor document."""
    get_client().collection(TOKENS_COLLECTION).document(DRIVE_SYNC_TOKEN_DOC).set({
        "page_token": token,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
