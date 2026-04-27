import logging
from datetime import datetime, timezone, timedelta

from google.cloud import firestore

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcript_index"
STALE_SYNCING_MINUTES = 15


def _get_db() -> firestore.Client:
    return firestore.Client()


def acquire_for_syncing(doc_id: str) -> bool:
    db = _get_db()
    ref = db.collection(COLLECTION_NAME).document(doc_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        if snapshot.to_dict().get("status") != "imported":
            return False
        transaction.update(ref, {
            "status": "syncing",
            "syncing_started_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Acquired for syncing: %s", doc_id)
    return result


def mark_synced(doc_id: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "synced",
        "synced_at": firestore.SERVER_TIMESTAMP,
        "syncing_started_at": None,
        "error": None,
    })
    logger.info("Marked synced: %s", doc_id)


def mark_error(doc_id: str, error: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "error",
        "error": error,
        "syncing_started_at": None,
    })
    logger.info("Marked error: %s — %s", doc_id, error)


def recover_stale_syncing() -> list[str]:
    db = _get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)

    syncing_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "syncing"))
        .stream()
    )

    recovered = []
    for doc in syncing_docs:
        started_at = doc.to_dict().get("syncing_started_at")
        if started_at and started_at < cutoff:
            db.collection(COLLECTION_NAME).document(doc.id).update({
                "status": "imported",
                "syncing_started_at": None,
            })
            recovered.append(doc.id)
            logger.info("Recovered stale syncing: %s", doc.id)

    return recovered
