from __future__ import annotations

import importlib


def test_runner_deletes_only_index_ids_missing_from_live_drive(app_root, monkeypatch):
    monkeypatch.syspath_prepend(str(app_root / "services" / "vector_sync" / "app"))
    runner = importlib.import_module("services.vector_sync.app.runner")
    deleted = []
    monkeypatch.setattr(
        runner,
        "scan_root_folder",
        lambda: ([], [{"doc_id": "live", "modifiedTime": "same"}]),
    )
    monkeypatch.setattr(runner, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(runner, "load_index", lambda _doc_id: {"modifiedTime": "same"})
    monkeypatch.setattr(runner, "list_imported", lambda: [])
    monkeypatch.setattr(runner, "list_all_index_ids", lambda: ["live", "removed-a", "removed-b"])
    monkeypatch.setattr(runner, "delete_index", deleted.append)

    result = runner.run_sync()

    assert set(deleted) == {"removed-a", "removed-b"}
    assert result == {
        "files_found": 1,
        "unchanged": 1,
        "imported_found": 0,
        "processed": 0,
        "skipped": 0,
        "not_acquired": 0,
        "error": 0,
    }
