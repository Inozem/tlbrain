from services.sync.app.drive_client import scan_root_folder


def run_sync():
    files = scan_root_folder()

    return {
        "files_found": len(files),
        "files": files,
    }
