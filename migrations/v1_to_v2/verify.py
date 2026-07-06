#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path


def _load_env() -> None:
    env_file = Path(__file__).parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

from google.cloud import firestore as fs
from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

from core.google_drive.firestore import CLIENTS_COLLECTION, COLLECTION_NAME, FOLDERS_COLLECTION
from core.qdrant.client import get_client as get_qdrant_client
from core.qdrant.schema import get_collection_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PLACEHOLDER_STATUSES = {"queued", "downloading"}


def _get_db() -> fs.Client:
    return fs.Client()


def verify() -> bool:
    db = _get_db()
    ok = True

    docs_missing_parent: list[str] = []
    docs_with_client_name: list[str] = []
    used_parent_ids: set[str] = set()

    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}
        if data.get("status") in PLACEHOLDER_STATUSES:
            continue
        if not data.get("parent_id"):
            docs_missing_parent.append(doc.id)
        else:
            used_parent_ids.add(data["parent_id"])
        if "client_name" in data:
            docs_with_client_name.append(doc.id)

    if docs_missing_parent:
        logger.error("FAIL  %d transcript_index docs missing parent_id: %s", len(docs_missing_parent), docs_missing_parent[:10])
        ok = False
    else:
        logger.info("OK    all transcript_index docs have parent_id")

    if docs_with_client_name:
        logger.error("FAIL  %d transcript_index docs still have client_name: %s", len(docs_with_client_name), docs_with_client_name[:10])
        ok = False
    else:
        logger.info("OK    no transcript_index docs have client_name")

    all_folder_ids = {doc.id for doc in db.collection(FOLDERS_COLLECTION).stream()}
    orphan_ids = used_parent_ids - all_folder_ids
    if orphan_ids:
        logger.error("FAIL  parent_ids not in folders/: %s", orphan_ids)
        ok = False
    else:
        logger.info("OK    all parent_ids exist in folders/ (%d folders registered)", len(all_folder_ids))

    clients_remaining = [doc.id for doc in db.collection(CLIENTS_COLLECTION).stream()]
    if clients_remaining:
        logger.error("FAIL  clients/ not empty (%d): %s", len(clients_remaining), clients_remaining[:10])
        ok = False
    else:
        logger.info("OK    clients/ collection is empty")

    qdrant = get_qdrant_client()
    collection = get_collection_name()
    stale_count = qdrant.count(
        collection_name=collection,
        count_filter=Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key="client_name"))]),
    ).count
    if stale_count:
        logger.error("FAIL  Qdrant has %d points still carrying client_name", stale_count)
        ok = False
    else:
        logger.info("OK    Qdrant has no points with client_name")

    logger.info("--- %s ---", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
