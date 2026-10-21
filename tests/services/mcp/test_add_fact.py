import json
import logging
from types import SimpleNamespace

import pytest
from google.genai import types

from core.indexing.user_facts import (
    AddFactDocumentNotFound,
    AddFactInputError,
    AddFactTranscriptNotSynced,
)
from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest
from core.integrations.firestore.effects import get_effect_allocation
from core.integrations.qdrant import client as qdrant_client, writer as qdrant_writer


def _request(arguments, request_id="add-fact-test"):
    return JSONRPCRequest(
        jsonrpc="2.0",
        id=request_id,
        method="tools/call",
        params={"name": "add_fact", "arguments": arguments},
    )


def test_add_fact_adapter_returns_domain_result_and_logs_document(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=handler.__name__)
    received = []
    monkeypatch.setattr(handler.write_operations, "add_fact", lambda doc_id, text: received.append((doc_id, text)) or "trimmed-doc")

    response = handler.handle_tools_call(_request({"doc_id": " doc ", "text": " fact "}))

    assert received == [(" doc ", " fact ")]
    assert response["id"] == "add-fact-test"
    assert json.loads(response["result"]["content"][0]["text"]) == {
        "status": "ok", "doc_id": "trimmed-doc",
    }
    assert "tool call: add_fact" in caplog.text


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (AddFactInputError("doc_id and text are required"), "doc_id and text are required"),
        (AddFactDocumentNotFound("missing"), "Document not found: missing"),
        (
            AddFactTranscriptNotSynced("syncing"),
            "Document is not synced yet (status=syncing). Try again after sync completes.",
        ),
    ],
)
def test_add_fact_adapter_maps_domain_validation_to_invalid_params(monkeypatch, error, message):
    monkeypatch.setattr(handler.write_operations, "add_fact", lambda *_args: (_ for _ in ()).throw(error))

    response = handler.handle_tools_call(_request({"doc_id": "doc", "text": "fact"}))

    assert response == {
        "jsonrpc": "2.0",
        "id": "add-fact-test",
        "error": {"code": -32602, "message": message},
    }


def test_add_fact_adapter_maps_unexpected_failure_to_internal_error(monkeypatch):
    monkeypatch.setattr(handler.write_operations, "add_fact", lambda *_args: (_ for _ in ()).throw(RuntimeError("storage failed")))

    response = handler.handle_tools_call(_request({"doc_id": "doc", "text": "fact"}))

    assert response == {
        "jsonrpc": "2.0",
        "id": "add-fact-test",
        "error": {"code": -32603, "message": "Failed to add fact", "details": "storage failed"},
    }


def test_sessionless_add_fact_reuses_legacy_point_through_the_real_write_guard(
    monkeypatch, firestore_fake, sdk_fakes,
):
    from core.operations import write as write_operations

    client = firestore_fake.client_class()
    client.collection("transcript_index").document("doc-1").set({
        "status": "synced", "parent_id": "folder-a", "dialog_date": "2026-01-02",
    })
    monkeypatch.setattr(write_operations.firestore_client, "get_client", lambda: client)
    monkeypatch.setattr(qdrant_client, "_client", None)

    point_id = qdrant_writer.user_fact_point_id("doc-1", "remember this")
    payload = {
        "type": "user_fact", "doc_id": "doc-1", "text": "remember this",
        "root_folder_id": "pytest-synthetic-root", "parent_id": "folder-a",
        "dialog_date": "2026-01-02", "dialog_date_num": 20260102,
    }
    sdk_fakes.qdrant.queue("retrieve", response=[])
    sdk_fakes.qdrant.queue("retrieve", response=[])
    sdk_fakes.qdrant.queue("retrieve", response=[SimpleNamespace(payload=payload)])
    sdk_fakes.qdrant.queue("retrieve", response=[SimpleNamespace(payload=payload)])
    sdk_fakes.qdrant.queue("retrieve", response=[SimpleNamespace(payload=payload)])
    sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    ))

    first = handler.handle_tools_call(_request(
        {"doc_id": " doc-1 ", "text": " remember this "}, request_id="first-write",
    ))
    second = handler.handle_tools_call(_request(
        {"doc_id": " doc-1 ", "text": " remember this "}, request_id="second-write",
    ))

    expected_payload = {"status": "ok", "doc_id": "doc-1"}
    assert first["id"] == "first-write"
    assert second["id"] == "second-write"
    assert json.loads(first["result"]["content"][0]["text"]) == expected_payload
    assert json.loads(second["result"]["content"][0]["text"]) == expected_payload

    operations = list(client.collection("dialog_operations").stream())
    operation_ids = [snapshot.id for snapshot in operations]
    assert len(operation_ids) == 2
    assert operation_ids[0] != operation_ids[1]
    assert [snapshot.to_dict()["doc_id"] for snapshot in operations] == ["doc-1", "doc-1"]
    assert [get_effect_allocation(client, operation_id, "mcp-write")["resource_id"]
            for operation_id in operation_ids] == [point_id, point_id]

    trace = sdk_fakes.trace
    upserts = [event for event in trace if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
    embeddings = [event for event in trace if event["boundary"] == "gemini" and event["operation"] == "embed_content"]
    retrieves = [event for event in trace if event["boundary"] == "qdrant" and event["operation"] == "retrieve"]
    assert len(upserts) == 1
    assert len(embeddings) == 1
    assert len(retrieves) == 5
    assert all(event["arguments"]["ids"] == [point_id] for event in retrieves)
    stored_points = upserts[0]["arguments"]["points"]
    assert len(stored_points) == 1
    assert stored_points[0].id == point_id
    assert stored_points[0].payload == payload
    gemini_clients = [event for event in trace if event["boundary"] == "gemini" and event["operation"] == "client"]
    assert len(gemini_clients) == 1
    retry_options = gemini_clients[0]["arguments"]["http_options"].retry_options
    assert retry_options.attempts == 1
