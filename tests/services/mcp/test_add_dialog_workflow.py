from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
from google.genai import types
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import Record, ScoredPoint

from tests.fixtures.sdk_fakes import FakeCloudTasksClient
from tests.services.mcp.test_add_dialog import (
    BASELINE,
    _install_document_boundary,
    _payload,
    _post,
    _rpc,
    _selector_loop,
)


@pytest.fixture(autouse=True)
def connectors_disabled(monkeypatch):
    # conftest supplies synthetic credentials; these integrations must remain off.
    for name in (
        "ALLOWED_EMAIL",
        "K_SERVICE",
        "TLDV_API_KEY",
        "TL_DV_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_USER_IDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "pytest-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "pytest-client-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "pytest-refresh-token")


def _post_asgi(app, path: str, *, content: bytes = b""):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://vector-sync.test"
        ) as client:
            return await client.post(path, content=content)

    return asyncio.run(request(), loop_factory=_selector_loop)


def _rpc_payload(response) -> dict:
    """Decode a normal tool result or correlated in-progress error payload."""
    body = response.json()
    for item in body.get("result", {}).get("content", []):
        if item.get("type") == "text":
            try:
                parsed = json.loads(item["text"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and parsed.get("operation_id"):
                return parsed
    error_data = body.get("error", {}).get("data", {})
    if error_data.get("operation_id"):
        return error_data
    raise AssertionError(f"response did not carry an operation_id: {body!r}")


def _call(session_id: str, request_id, name: str, arguments: dict):
    return _post(
        _rpc("tools/call", request_id, {"name": name, "arguments": arguments}),
        headers={"mcp-session-id": session_id},
    )


def test_add_dialog_offline_workflow_replays_dispatch_and_becomes_queryable(
    monkeypatch, sdk_fakes, firestore_fake,
):
    files, documents, created_docs = _install_document_boundary(monkeypatch, sdk_fakes)
    folder_id = "pytest-workflow-folder"
    folder_name = "Workflow Projects"
    folder_path = [folder_name]
    files[folder_id] = {
        "id": folder_id,
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": ["pytest-synthetic-root"],
        "trashed": False,
        "appProperties": {},
    }
    firestore_fake.client_class().collection("folders").document(folder_id).set({
        "folder_id": folder_id,
        "name": folder_name,
        "parent_id": "pytest-synthetic-root",
        "speakers": {},
    })

    initialized = _post(
        _rpc("initialize", "workflow-initialize"),
        headers={"accept": "text/event-stream"},
    )
    assert initialized.status_code == 200
    session_id = initialized.headers["mcp-session-id"]
    listed = _post(
        _rpc("tools/list", "workflow-tools"),
        headers={"mcp-session-id": session_id},
    )
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["result"]["tools"]
    actual_tools = listed.json()["result"]["tools"]
    assert len(actual_tools) == 14
    expected_legacy = [tool for tool in baseline if tool["name"] != "add_dialog"]
    actual_legacy = [tool for tool in actual_tools if tool["name"] != "add_dialog"]
    for tool_set in (actual_legacy, expected_legacy):
        sync_status = next(tool for tool in tool_set if tool["name"] == "sync_status")
        properties = sync_status["inputSchema"]["properties"]
        operation_id = properties.pop("operation_id", None)
        if operation_id is not None:
            assert operation_id.get("type") == "string"
            assert "operation_id" not in sync_status["inputSchema"].get("required", [])
    assert actual_legacy == expected_legacy

    original_create_task = FakeCloudTasksClient.create_task
    lost_ack = True

    def accept_then_lose_ack(self, *, parent, task, **kwargs):
        nonlocal lost_ack
        result = original_create_task(self, parent=parent, task=task, **kwargs)
        url = task.get("http_request", {}).get("url", "")
        if lost_ack and url.startswith(os.environ["VECTOR_SYNC_URL"]):
            lost_ack = False
            raise TimeoutError("synthetic lost Cloud Tasks acknowledgement")
        return result

    monkeypatch.setattr(FakeCloudTasksClient, "create_task", accept_then_lose_ack)
    arguments = {
        "provider": "chatgpt",
        "utterances": [
            {"speaker": "Ada", "text": "The first offline workflow utterance."},
            {"speaker": "Grace", "text": "The second offline workflow utterance."},
        ],
        "title": "Offline workflow conversation",
        "dialog_date": "2025-04-03",
        "folder_path": folder_path,
    }
    request_id = 31031  # JSON-RPC number IDs stay typed across the replay.
    request = _rpc("tools/call", request_id, {"name": "add_dialog", "arguments": arguments})
    first = _post(request, headers={"mcp-session-id": session_id})
    assert first.status_code == 200
    first_body = first.json()
    first_payload = _rpc_payload(first)
    assert first_body["id"] == request_id
    assert first_payload.get("status") in {"created", "indexing"}
    operation_id = first_payload["operation_id"]
    doc_id = first_payload["doc_id"]
    assert operation_id and doc_id

    replay = _post(request, headers={"mcp-session-id": session_id})
    assert replay.status_code == 200
    replay_payload = _rpc_payload(replay)
    assert replay.json()["id"] == request_id
    assert replay_payload["operation_id"] == operation_id
    assert replay_payload["doc_id"] == doc_id
    assert replay_payload.get("status") in {"created", "indexing"}
    assert first_payload["folder_path"] == folder_path
    assert replay_payload["folder_path"] == folder_path

    in_progress = _rpc_payload(_call(
        session_id, "workflow-operation-status-before-sync", "sync_status",
        {"operation_id": operation_id},
    ))
    assert in_progress["operation_id"] == operation_id
    assert in_progress["operation_status"] in {"pending", "running"}
    assert in_progress["stage"]
    assert in_progress["doc_id"] == doc_id

    assert created_docs == {doc_id}
    assert len([
        event for event in sdk_fakes.trace
        if event["boundary"] == "drive" and event["operation"] == "files.create"
        and event["arguments"].get("body", {}).get("mimeType") == "application/vnd.google-apps.document"
    ]) == 1
    assert sum(
        event["boundary"] == "tasks" and event["operation"] == "create_task"
        and event["arguments"]["task"]["http_request"]["url"].startswith(os.environ["VECTOR_SYNC_URL"])
        for event in sdk_fakes.trace
    ) == 1
    assert files[doc_id]["mimeType"] == "application/vnd.google-apps.document"
    assert files[doc_id]["parents"] == [folder_id]
    assert files[doc_id]["appProperties"]["tlbrain_save_id"]
    document_text = documents[doc_id]["text"]
    assert document_text.index("2025-04-03") < document_text.index("chatgpt")
    assert document_text.index("chatgpt") < document_text.index("The first offline workflow utterance.")
    assert document_text.index("The first offline workflow utterance.") < document_text.index("The second offline workflow utterance.")
    index_before_sync = firestore_fake.client_class().collection("transcript_index").document(doc_id).get()
    assert index_before_sync.exists
    assert index_before_sync.to_dict()["provider"] == "chatgpt"
    assert index_before_sync.to_dict()["status"] in {"queued", "downloading", "imported", "syncing"}
    assert index_before_sync.to_dict()["status"] != "synced"

    task_events = [
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
        and event["arguments"]["task"]["http_request"]["url"].startswith(os.environ["VECTOR_SYNC_URL"])
    ]
    task = task_events[0]["arguments"]["task"]
    task_request = task["http_request"]
    task_body = task_request["body"]
    if isinstance(task_body, str):
        task_body = task_body.encode("utf-8")
    task_json = json.loads(task_body)
    assert task_json["doc_id"] == doc_id
    vector_path = "/sync/doc/" + doc_id
    assert task_request["url"].endswith(vector_path)

    # This is the supported post-Doc lost-ack case, not an HTTP request timeout:
    # the fake accepted one task and then withheld its acknowledgement.
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part(text=json.dumps({"summary": "Workflow conversation summary.", "facts": []}))
        ]))
    ]))
    for _ in range(3):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)]
        ))

    from services.vector_sync.app.main import app as vector_sync_app

    sync_response = _post_asgi(vector_sync_app, vector_path, content=task_body)
    assert sync_response.status_code == 200
    assert sync_response.json() == {"status": "ok", "result": "processed"}
    indexed = firestore_fake.client_class().collection("transcript_index").document(doc_id).get()
    assert indexed.to_dict()["status"] == "synced"
    assert indexed.to_dict()["provider"] == "chatgpt"

    sdk_fakes.gemini.queue(
        "embed_content",
        response=types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.25] * 768)]),
    )
    sdk_fakes.gemini.queue(
        "embed_content",
        response=types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.25] * 768)]),
    )
    synthetic_payload = {
        "doc_id": doc_id,
        "parent_id": folder_id,
        "type": "summary",
        "center_index": 1,
        "covered_range": [0, 1],
    }
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[ScoredPoint(
        id="synthetic-summary",
        version=1,
        score=0.95,
        payload=synthetic_payload,
    )]))
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[]))
    sdk_fakes.qdrant.queue("query_points", response=QueryResponse(points=[]))
    sdk_fakes.qdrant.queue("scroll", response=([Record(
        id="synthetic-utterance",
        payload={
            "doc_id": doc_id,
            "type": "utterance",
            "order_index": 0,
            "text": "The first offline workflow utterance.",
        },
    )], None))
    query_response = _call(session_id, "workflow-query", "query", {"query": "offline workflow"})
    assert query_response.status_code == 200
    query_payload = _payload(query_response)
    assert doc_id in json.dumps(query_payload)
    assert "The first offline workflow utterance." in json.dumps(query_payload)
    assert any(
        event["boundary"] == "qdrant" and event["operation"] == "query_points"
        for event in sdk_fakes.trace
    )

    status = _rpc_payload(_call(
        session_id, "workflow-operation-status", "sync_status", {"operation_id": operation_id}
    ))
    assert status["operation_id"] == operation_id
    assert status["operation_status"] == "completed"
    assert status["result"]["doc_id"] == doc_id

    aggregate = _payload(_call(session_id, "workflow-aggregate-status", "sync_status", {}))
    assert {key: value for key, value in aggregate.items() if key != "suggestion"} == {
        "total": 1,
        "queued": 0,
        "downloading": 0,
        "imported": 0,
        "syncing": 0,
        "synced": 1,
        "error": 0,
    }
