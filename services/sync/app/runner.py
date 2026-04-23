from services.sync.app.docx_parser import extract_text_from_docx_bytes
from services.sync.app.drive_client import (
    download_file_bytes,
    scan_root_folder,
)
from services.sync.app.hashing import sha256_text
from services.sync.app.index_store import save_index


def run_sync():
    files = scan_root_folder()

    indexed_total = 0

    for file in files:
        doc_id = file["doc_id"]

        file_bytes = download_file_bytes(doc_id)

        raw_text = extract_text_from_docx_bytes(file_bytes)

        content_hash = sha256_text(raw_text)

        payload = {
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": file["modifiedTime"],
            "content_hash": content_hash,
            "text_preview": raw_text[:1000],
            "indexed": True,
        }

        save_index(doc_id, payload)

        indexed_total += 1

    return {
        "files_found": len(files),
        "indexed_total": indexed_total,
        "files": files,
    }
