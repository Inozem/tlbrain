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
    monkeypatch.setattr(handler, "resolve_folder_path", lambda path: paths[tuple(path)])
    monkeypatch.setattr(
        handler, "expand_subtree", lambda folder_id: ["parent-id", "child-id"]
    )
    monkeypatch.setattr(
        handler,
        "get_root_folder_id",
        lambda: "root-id",
    )
    monkeypatch.setattr(
        handler,
        "get_folder_by_id",
        lambda folder_id: {"name": "Parent"},
    )
    monkeypatch.setattr(
        handler,
        "move_folder_in_drive",
        lambda *args: side_effects.append(("drive", args)),
    )
    monkeypatch.setattr(
        handler,
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
