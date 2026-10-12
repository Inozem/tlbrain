"""Firestore client factory and collection identifiers."""

from google.cloud import firestore


COLLECTION_NAME = "transcript_index"
CLIENTS_COLLECTION = "clients"
FOLDERS_COLLECTION = "folders"
TOKENS_COLLECTION = "tokens"
DRIVE_SYNC_TOKEN_DOC = "drive_sync"


def get_client() -> firestore.Client:
    """Create a Firestore client when requested, using the SDK defaults."""
    return firestore.Client()
