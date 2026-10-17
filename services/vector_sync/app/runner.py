import logging

from google.cloud import firestore as firestore_module

from core.integrations.firestore.folders import upsert_folder
from core.config import get_root_folder_id
from core.integrations.google_drive.scanner import scan_root_folder
from core.integrations.firestore.transcripts import (
    delete_index,
    list_all_index_ids,
    list_imported,
    load_index,
    save_index,
)
from core.indexing.processor import process_one

logger = logging.getLogger(__name__)


def run_sync():
    live_folders, live_docs = scan_root_folder()
    root_folder_id = get_root_folder_id()

    for folder in live_folders:
        upsert_folder(folder["id"], folder["name"], folder["parent_id"])
    logger.info("Upserted %d folder(s) from Drive", len(live_folders))

    unchanged = 0
    for file in live_docs:
        doc_id = file["doc_id"]
        existing = load_index(doc_id)

        if existing and existing.get("modifiedTime") == file["modifiedTime"]:
            unchanged += 1
            continue

        save_index(doc_id, {
            "error": None,
            **(existing or {}),
            "doc_id": doc_id,
            "parent_id": file["parent_id"],
            "modifiedTime": file["modifiedTime"],
            "root_folder_id": root_folder_id,
            "status": "imported",
            "status_changed_at": firestore_module.SERVER_TIMESTAMP,
        })

        logger.info("Marked imported: %s parent=%s", doc_id, file["parent_id"])

    logger.info("Drive scan — found=%d unchanged=%d marked_imported=%d", len(live_docs), unchanged, len(live_docs) - unchanged)

    imported = list_imported()
    stats = {"processed": 0, "skipped": 0, "not_acquired": 0, "error": 0}

    for doc in imported:
        doc_id = doc["doc_id"]
        parent_id = doc.get("parent_id", "")
        try:
            result = process_one(doc_id, parent_id, root_folder_id)
            stats[result] = stats.get(result, 0) + 1
        except Exception:
            stats["error"] += 1

    drive_ids = {f["doc_id"] for f in live_docs}
    indexed_ids = set(list_all_index_ids())

    for doc_id in indexed_ids - drive_ids:
        delete_index(doc_id)
        logger.info("Deleted from index: %s", doc_id)

    logger.info(
        "Sync complete — processed=%d skipped=%d not_acquired=%d error=%d",
        stats["processed"], stats["skipped"], stats["not_acquired"], stats["error"],
    )

    return {
        "files_found": len(live_docs),
        "unchanged": unchanged,
        "imported_found": len(imported),
        **stats,
    }
