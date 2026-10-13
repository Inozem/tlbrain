"""Offline MCP contract tests for folder mutation cycle rejection."""

from __future__ import annotations

import pytest

from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


def _run(coroutine):
    import sniffio

    context_token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
    finally:
        sniffio.current_async_library_cvar.reset(context_token)
        coroutine.close()
    raise AssertionError("in-process MCP coroutine unexpectedly suspended")


@pytest.mark.offline
@pytest.mark.parametrize(
    "new_parent_path",
    [["Parent"], ["Parent", "Child"]],
    ids=("self", "descendant"),
)
def test_move_folder_rejects_self_or_descendant_without_side_effects(
    monkeypatch, new_parent_path
):
    paths = {
        ("Parent",): ["parent-id"],
        ("Parent", "Child"): ["child-id"],
    }
    side_effects = []
    from core.integrations.google_drive import folder_service

    monkeypatch.setattr(
        folder_service, "_resolve", lambda path, role: paths[tuple(path)][0]
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_all_folders",
        lambda: {
            "parent-id": {"name": "Parent", "parent_id": "root-id"},
            "child-id": {"name": "Child", "parent_id": "parent-id"},
        },
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_folder_by_id",
        lambda folder_id: {"name": "Parent"},
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "move_folder_in_drive",
        lambda *args: side_effects.append(("drive", args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: side_effects.append(("firestore", args)),
    )
    request = JSONRPCRequest(
        jsonrpc="2.0",
        id="move-cycle",
        method="tools/call",
        params={
            "name": "move_folder",
            "arguments": {
                "folder_path": ["Parent"],
                "new_parent_path": new_parent_path,
            },
        },
    )

    response = _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))

    assert response["error"] == {
        "code": -32602,
        "message": "Cannot move a folder into its own subtree.",
    }
    assert response["id"] == "move-cycle"
    assert side_effects == []


@pytest.mark.offline
@pytest.mark.parametrize(
    "operation,arguments,error,expected",
    [
        (
            "create_folder",
            {"name": "New"},
            lambda service: service.FolderNotFound(["Missing"], "parent"),
            {"code": -32602, "message": "Parent folder not found: Missing"},
        ),
        (
            "move_folder",
            {"folder_path": ["Same"]},
            lambda service: service.AmbiguousFolderPath(["Same"], "folder", 2),
            {"code": -32602, "message": "Ambiguous path: Same matches 2 folders. Provide a more specific path."},
        ),
        (
            "rename_folder",
            {"folder_path": ["Old"], "new_name": "New"},
            lambda service: service.FolderNameConflict("New"),
            {
                "code": -32602,
                "message": "Folder 'New' already exists in this location.",
                "details": "Use list_folders to see the existing folder hierarchy.",
            },
        ),
    ],
)
def test_folder_domain_errors_keep_existing_mcp_messages(
    monkeypatch, operation, arguments, error, expected
):
    from core.integrations.google_drive import folder_service

    monkeypatch.setattr(
        folder_service,
        operation,
        lambda *args: (_ for _ in ()).throw(error(folder_service)),
    )
    request = JSONRPCRequest(
        jsonrpc="2.0",
        id="folder-error",
        method="tools/call",
        params={"name": operation, "arguments": arguments},
    )

    response = _run(handler.handle_mcp_request(request.model_dump(exclude_none=True)))

    assert response["error"] == expected
