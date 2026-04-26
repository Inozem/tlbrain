import logging

from services.sync.app.docx_parser import extract_text_from_docx_bytes
from services.sync.app.drive_client import (
    download_file_bytes,
    scan_root_folder,
)
from services.sync.app.hashing import sha256_text
from services.sync.app.index_store import (
    load_index,
    save_index,
    delete_index,
    list_all_index_ids,
)

logger = logging.getLogger(__name__)


def is_valid_docx(name: str) -> bool:
    if not name.lower().endswith(".docx"):
        return False
    if name.startswith("~$"):
        return False
    if name.startswith("."):
        return False
    return True


def run_sync():
    files = scan_root_folder()

    stats = {
        "new": 0,
        "updated": 0,
        "skipped": 0,
        "ignored": 0,
        "deleted": 0,
    }

    processed = []
    valid_files = []

    for file in files:
        if not is_valid_docx(file["name"]):
            logger.info("Ignored file: %s (%s)", file["name"], file["doc_id"])
            stats["ignored"] += 1
            continue
        valid_files.append(file)

    for file in valid_files:
        doc_id = file["doc_id"]
        modified_time = file["modifiedTime"]

        existing = load_index(doc_id)

        if existing and existing.get("modifiedTime") == modified_time:
            stats["skipped"] += 1
            continue

        file_bytes = download_file_bytes(doc_id)
        raw_text = extract_text_from_docx_bytes(file_bytes)
        content_hash = sha256_text(raw_text)

        if existing and existing.get("content_hash") == content_hash:
            save_index(doc_id, {**existing, "modifiedTime": modified_time})
            stats["skipped"] += 1
            continue

        save_index(doc_id, {
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": modified_time,
            "content_hash": content_hash,
            "text_preview": raw_text[:1000],
            "indexed": True,
        })

        if existing:
            stats["updated"] += 1
            action = "updated"
        else:
            stats["new"] += 1
            action = "new"

        logger.info(
            "File %s: %s | client=%s chars=%d",
            action,
            doc_id,
            file["client_name"],
            len(raw_text),
        )

        processed.append({
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "action": action,
            "chars": len(raw_text),
            "preview": raw_text[:300],
        })

    drive_ids = {f["doc_id"] for f in valid_files}
    indexed_ids = set(list_all_index_ids())

    for doc_id in indexed_ids - drive_ids:
        delete_index(doc_id)
        stats["deleted"] += 1
        logger.info("File deleted from index: %s", doc_id)

    logger.info(
        "Sync complete — new=%d updated=%d skipped=%d deleted=%d ignored=%d",
        stats["new"],
        stats["updated"],
        stats["skipped"],
        stats["deleted"],
        stats["ignored"],
    )

    return {
        "files_found": len(files),
        "new": stats["new"],
        "updated": stats["updated"],
        "skipped": stats["skipped"],
        "deleted": stats["deleted"],
        "ignored": stats["ignored"],
        "processed": processed,
    }
