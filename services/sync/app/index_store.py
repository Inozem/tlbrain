import logging
from typing import Any

from google.cloud import firestore


logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcript_index"


def get_db():
    return firestore.Client()


def load_index(doc_id: str) -> dict[str, Any] | None:
    db = get_db()

    doc = db.collection(COLLECTION_NAME).document(doc_id).get()

    if not doc.exists:
        return None

    logger.info("Index loaded: %s", doc_id)

    return doc.to_dict()


def save_index(doc_id: str, payload: dict[str, Any]) -> None:
    db = get_db()

    db.collection(COLLECTION_NAME).document(doc_id).set(payload)

    logger.info("Index saved: %s", doc_id)
