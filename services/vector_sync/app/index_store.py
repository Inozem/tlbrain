"""Compatibility exports for transcript index Firestore operations."""

from core.integrations.firestore.client import COLLECTION_NAME, get_client
from core.integrations.firestore.transcripts import (
    delete_index,
    list_all_index_ids,
    load_index,
    save_index,
    update_index,
)


def get_db():
    """Compatibility helper retained for legacy consumers."""
    return get_client()
