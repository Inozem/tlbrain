import logging

from core.google_drive.firestore import list_imported, recover_stale_syncing
from core.config import get_root_folder_id
from core.google_drive.drive_client import scan_root_folder
from services.vector_sync.app.index_store import (
    delete_index,
    list_all_index_ids,
    load_index,
    save_index,
)
from services.vector_sync.app.processor import process_one

logger = logging.getLogger(__name__)


def run_sync():
    recover_stale_syncing()

    files = scan_root_folder()
    root_folder_id = get_root_folder_id()

    unchanged = 0
    for file in files:
        doc_id = file["doc_id"]
        existing = load_index(doc_id)

        if existing and existing.get("modifiedTime") == file["modifiedTime"]:
            unchanged += 1
            continue

        save_index(doc_id, {
            **(existing or {}),
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": file["modifiedTime"],
            "root_folder_id": root_folder_id,
            "status": "imported",
        })

        logger.info("Marked imported: %s client=%s", doc_id, file["client_name"])

    logger.info("Drive scan — found=%d unchanged=%d marked_imported=%d", len(files), unchanged, len(files) - unchanged)

    imported = list_imported()
    stats = {"processed": 0, "skipped": 0, "not_acquired": 0, "error": 0}

    for doc in imported:
        doc_id = doc["doc_id"]
        client_name = doc.get("client_name", "")
        try:
            result = process_one(doc_id, client_name, root_folder_id)
            stats[result] = stats.get(result, 0) + 1
        except Exception:
            stats["error"] += 1

    drive_ids = {f["doc_id"] for f in files}
    indexed_ids = set(list_all_index_ids())

    for doc_id in indexed_ids - drive_ids:
        delete_index(doc_id)
        logger.info("Deleted from index: %s", doc_id)

    logger.info(
        "Sync complete — processed=%d skipped=%d not_acquired=%d error=%d",
        stats["processed"], stats["skipped"], stats["not_acquired"], stats["error"],
    )

    return {
        "files_found": len(files),
        "unchanged": unchanged,
        "imported_found": len(imported),
        **stats,
    }
