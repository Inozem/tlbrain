import json

import pytest

from core.indexing.user_facts import (
    AddFactDocumentNotFound,
    AddFactInputError,
    AddFactTranscriptNotSynced,
)
from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


def _request(arguments):
    return JSONRPCRequest(
        jsonrpc="2.0",
        id="add-fact-test",
        method="tools/call",
        params={"name": "add_fact", "arguments": arguments},
    )


def test_add_fact_adapter_returns_domain_result_and_logs_document(monkeypatch, caplog):
    received = []
    monkeypatch.setattr(handler, "add_user_fact", lambda doc_id, text: received.append((doc_id, text)) or "trimmed-doc")

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
    monkeypatch.setattr(handler, "add_user_fact", lambda *_args: (_ for _ in ()).throw(error))

    response = handler.handle_tools_call(_request({"doc_id": "doc", "text": "fact"}))

    assert response == {
        "jsonrpc": "2.0",
        "id": "add-fact-test",
        "error": {"code": -32602, "message": message},
    }


def test_add_fact_adapter_maps_unexpected_failure_to_internal_error(monkeypatch):
    monkeypatch.setattr(handler, "add_user_fact", lambda *_args: (_ for _ in ()).throw(RuntimeError("storage failed")))

    response = handler.handle_tools_call(_request({"doc_id": "doc", "text": "fact"}))

    assert response == {
        "jsonrpc": "2.0",
        "id": "add-fact-test",
        "error": {"code": -32603, "message": "Failed to add fact", "details": "storage failed"},
    }
