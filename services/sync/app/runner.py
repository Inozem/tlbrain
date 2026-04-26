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


def run_sync():
    files = scan_root_folder()

    stats = {
        "new": 0,
        "updated": 0,
        "skipped": 0,
    }

    processed = []

    for file in files:
        doc_id = file["doc_id"]
        modified_time = file["modifiedTime"]

        existing = load_index(doc_id)

        # Fast skip if modifiedTime unchanged
        if existing and existing.get("modifiedTime") == modified_time:
            stats["skipped"] += 1
            continue

        file_bytes = download_file_bytes(doc_id)

        raw_text = extract_text_from_docx_bytes(file_bytes)

        content_hash = sha256_text(raw_text)

        # modifiedTime changed but content same
        if existing and existing.get("content_hash") == content_hash:
            payload = {
                **existing,
                "modifiedTime": modified_time,
            }

            save_index(doc_id, payload)

            stats["skipped"] += 1
            continue

        payload = {
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": modified_time,
            "content_hash": content_hash,
            "text_preview": raw_text[:1000],
            "indexed": True,
        }

        save_index(doc_id, payload)

        if existing:
            stats["updated"] += 1
            action = "updated"
        else:
            stats["new"] += 1
            action = "new"

        processed.append(
            {
                "doc_id": doc_id,
                "client_name": file["client_name"],
                "action": action,
                "chars": len(raw_text),
                "preview": raw_text[:300],
            }
        )

    deleted = 0

    drive_ids = {file["doc_id"] for file in files}
    indexed_ids = set(list_all_index_ids())

    for doc_id in indexed_ids - drive_ids:
        delete_index(doc_id)
        deleted += 1

    return {
        "files_found": len(files),
        "new": stats["new"],
        "updated": stats["updated"],
        "skipped": stats["skipped"],
        "deleted": deleted,
        "processed": processed,
    }
