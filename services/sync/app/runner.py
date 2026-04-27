import logging

from core.google_drive.firestore import list_imported, recover_stale_syncing
from services.sync.app.drive_client import get_root_folder_id, scan_root_folder
from services.sync.app.index_store import (
    delete_index,
    list_all_index_ids,
    load_index,
    save_index,
)
from services.sync.app.processor import process_one

logger = logging.getLogger(__name__)


def run_sync():
    recover_stale_syncing()

    files = scan_root_folder()
    root_folder_id = get_root_folder_id()

    for file in files:
        doc_id = file["doc_id"]
        existing = load_index(doc_id)

        if existing and existing.get("modifiedTime") == file["modifiedTime"]:
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
        "imported_found": len(imported),
        **stats,
    }
