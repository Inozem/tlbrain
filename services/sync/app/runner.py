from services.sync.app.drive_client import scan_root_folder
from services.sync.app.hashing import sha256_text
from services.sync.app.index_store import save_index


def run_sync():
    files = scan_root_folder()

    for file in files:
        doc_id = file["doc_id"]

        pseudo_content = file["name"] + file["modifiedTime"]

        payload = {
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": file["modifiedTime"],
            "content_hash": sha256_text(pseudo_content),
            "indexed": True,
        }

        save_index(doc_id, payload)

    return {
        "files_found": len(files),
        "indexed_total": len(files),
        "files": files,
    }
