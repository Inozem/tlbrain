from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import httpx
import httplib2
import pytest
from googleapiclient.errors import HttpError

from services.mcp.app import main


BASELINE = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_baseline.json"


@pytest.fixture(autouse=True)
def open_mcp(monkeypatch):
    monkeypatch.delenv("ALLOWED_EMAIL", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("TLDV_API_KEY", raising=False)
    monkeypatch.delenv("TL_DV_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "pytest-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "pytest-client-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "pytest-refresh-token")


def _rpc(method: str, request_id: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _selector_loop():
    """Avoid Python 3.12 Windows' blocking TCP fallback self-pipe accept."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()

    import socket

    old_socketpair = socket.socketpair
    local_pairs = []

    def local_datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            local_pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_datagram_pair
    try:
        try:
            return asyncio.SelectorEventLoop()
        except BaseException:
            for reader, writer in local_pairs:
                reader.close()
                writer.close()
            raise
    finally:
        socket.socketpair = old_socketpair


def _post(body: dict, headers: dict | None = None):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://mcp.test"
        ) as client:
            return await client.post("/mcp", json=body, headers=headers or {})

    return asyncio.run(request(), loop_factory=_selector_loop)


def _initialized_session() -> str:
    response = _post(_rpc("initialize", "initialize-add-dialog"))
    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id
    return session_id


def _call_add_dialog(arguments: dict, request_id: str = "add-dialog", *, session_id: str | None = None):
    session_id = session_id or _initialized_session()
    return _post(
        _rpc("tools/call", request_id, {"name": "add_dialog", "arguments": arguments}),
        headers={"mcp-session-id": session_id},
    )


def _payload(response) -> dict:
    return json.loads(response.json()["result"]["content"][0]["text"])


def _mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


def _has_safe_error_category(fields) -> bool:
    return any(
        isinstance(item.get("category"), str) and item["category"].strip()
        or isinstance(item.get("code"), str) and item["code"].strip()
        for item in fields
    )


def _install_document_boundary(
    monkeypatch, sdk_fakes, *, create_error: BaseException | None = None,
    fill_error: BaseException | None = None,
):
    """Model Drive file metadata and Docs text only at the SDK request boundary."""
    from tests.fixtures import sdk_fakes as sdk_fake_module

    files: dict[str, dict] = {}
    documents: dict[str, dict] = {}
    created_docs: set[str] = set()
    next_id = 0
    revision_counter = 0
    modified_counter = 0
    original_execute = sdk_fake_module._Request.execute
    root_id = os.environ["ROOT_FOLDER_URL"].rsplit("/", 1)[-1]
    files[root_id] = {
        "id": root_id,
        "name": "Synthetic root",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [],
        "trashed": False,
        "appProperties": {},
    }

    def execute(request, **kwargs):
        nonlocal next_id, revision_counter, modified_counter
        args = request.args
        if request.boundary == "drive" and request.operation == "files.create":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            if create_error is not None:
                raise create_error
            requested_id = args.get("body", {}).get("id")
            if (
                requested_id
                and args.get("body", {}).get("mimeType") == "application/vnd.google-apps.document"
            ):
                raise RuntimeError("Native Google Docs IDs are provider-assigned")
            next_id += 1
            file_id = requested_id or f"pytest-add-dialog-doc-{next_id}"
            modified_counter += 1
            files[file_id] = {
                "id": file_id,
                "modifiedTime": f"2025-04-03T00:00:{modified_counter:02d}Z",
                "trashed": False,
                **args.get("body", {}),
            }
            if files[file_id].get("mimeType") == "application/vnd.google-apps.document":
                created_docs.add(file_id)
                documents[file_id] = {"text": "", "revisionId": "pytest-docs-revision-0"}
            return {"id": file_id, "modifiedTime": files[file_id]["modifiedTime"]}
        if request.boundary == "drive" and request.operation == "files.get":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            file_data = files[args["fileId"]]
            return dict(file_data)
        if request.boundary == "drive" and request.operation == "files.update":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            file_data = files[args["fileId"]]
            file_data.update(args.get("body", {}))
            if args.get("addParents"):
                file_data["parents"] = [args["addParents"]]
            modified_counter += 1
            file_data["modifiedTime"] = f"2025-04-03T00:00:{modified_counter:02d}Z"
            return {"id": args["fileId"], "modifiedTime": file_data["modifiedTime"]}
        if request.boundary == "drive" and request.operation == "files.list":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            listed_folders = [
                dict(file_data)
                for file_data in files.values()
                if file_data.get("mimeType") == "application/vnd.google-apps.folder"
            ]
            if "name='_unassigned'" in args.get("q", ""):
                listed_folders = [
                    item for item in listed_folders
                    if item.get("name") == "_unassigned"
                    and os.environ["ROOT_FOLDER_URL"].rsplit("/", 1)[-1] in item.get("parents", [])
                ]
            return {"files": listed_folders}
        if request.boundary == "docs" and request.operation == "documents.batchUpdate":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            if fill_error is not None:
                raise fill_error
            doc_id = args["documentId"]
            body = args["body"]
            current = documents[doc_id]
            write_control = args.get("writeControl") or body.get("writeControl") or {}
            required_revision = write_control.get("requiredRevisionId")
            if required_revision is not None and required_revision != current["revisionId"]:
                raise RuntimeError("Stale Docs revision in test boundary")
            text = current["text"]
            for operation in body.get("requests", []):
                insertion = operation.get("insertText")
                if insertion:
                    text += insertion["text"]
            revision_counter += 1
            revision = f"pytest-docs-revision-{revision_counter}"
            documents[doc_id] = {"text": text, "revisionId": revision}
            modified_counter += 1
            files[doc_id]["modifiedTime"] = f"2025-04-03T00:00:{modified_counter:02d}Z"
            return {"writeControl": {"requiredRevisionId": revision}}
        if request.boundary == "docs" and request.operation == "documents.get":
            sdk_fake_module._record(request.boundary, request.operation, **args)
            doc_id = args["documentId"]
            document = documents[doc_id]
            text = document["text"]
            return {
                "documentId": doc_id,
                "revisionId": document["revisionId"],
                "body": {"content": [{
                    "startIndex": 0,
                    "endIndex": len(text) + 1,
                    "paragraph": {"elements": [{
                        "startIndex": 1,
                        "endIndex": len(text) + 1,
                        "textRun": {"content": text},
                    }]},
                }]},
            }
        return original_execute(request, **kwargs)

    monkeypatch.setattr(sdk_fake_module._Request, "execute", execute)
    return files, documents, created_docs


def test_tools_list_preserves_legacy_schemas_and_exposes_add_dialog_schema():
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    response = _post(_rpc("tools/list", "add-dialog-tools"))
    assert response.status_code == 200
    actual = response.json()["result"]["tools"]
    assert len(actual) == 14
    add_dialog = next((tool for tool in actual if tool["name"] == "add_dialog"), None)
    assert add_dialog is not None

    legacy = [tool for tool in actual if tool["name"] != "add_dialog"]
    expected_legacy = [
        tool for tool in expected["result"]["tools"] if tool["name"] != "add_dialog"
    ]
    assert {tool["name"] for tool in legacy} == {tool["name"] for tool in expected_legacy}
    for tools in (legacy, expected_legacy):
        for tool in tools:
            if tool["name"] == "sync_status":
                properties = tool["inputSchema"]["properties"]
                selector = properties.pop("operation_id", None)
                if selector is not None:
                    assert selector.get("type") == "string"
                    assert "operation_id" not in tool["inputSchema"].get("required", [])
    assert legacy == expected_legacy

    schema = add_dialog["inputSchema"]
    assert schema == {
        "type": "object",
        "required": ["provider", "utterances"],
        "additionalProperties": False,
        "properties": {
            "provider": {"type": "string", "minLength": 1},
            "utterances": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["speaker", "text"],
                    "additionalProperties": False,
                    "properties": {
                        "speaker": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1},
                    },
                },
            },
            "title": {"type": "string", "minLength": 1},
            "dialog_date": {"type": "string", "format": "date"},
            "folder_path": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    assert "enum" not in schema["properties"]["provider"]


@pytest.mark.parametrize("raw_id", [None, True, 7.5])
def test_add_dialog_rejects_invalid_raw_request_ids_without_side_effects(raw_id, firestore_fake, sdk_fakes):
    from services.mcp.app.mcp.handler import handle_mcp_request

    response = asyncio.run(handle_mcp_request({
        "jsonrpc": "2.0", "id": raw_id, "method": "tools/call",
        "params": {"name": "add_dialog", "arguments": {
            "provider": "manual", "utterances": [{"speaker": "A", "text": "words"}],
        }},
    }), loop_factory=_selector_loop)
    assert response["id"] is None
    assert response["error"]["code"] == -32602
    assert not list(firestore_fake.client_class().collection("mcp_sessions").stream())
    assert not list(firestore_fake.client_class().collection("dialog_operations").stream())
    assert sdk_fakes.trace == []


def test_add_dialog_unexpected_pre_reservation_failure_is_sanitized(monkeypatch, firestore_fake, caplog):
    from services.mcp.app.mcp import handler

    session_id = _initialized_session()
    monkeypatch.setattr(handler, "get_root_folder_id", lambda: (_ for _ in ()).throw(
        RuntimeError("SECRET_SENTINEL provider URL_SENTINEL https://private.invalid/raw")
    ))
    response = _call_add_dialog({
        "provider": "manual", "utterances": [{"speaker": "A", "text": "words"}],
    }, "sanitized-preflight-failure", session_id=session_id)
    assert response.status_code == 200
    serialized = response.text
    assert "SECRET_SENTINEL" not in serialized
    assert "URL_SENTINEL" not in serialized
    assert "private.invalid/raw" not in serialized
    assert "SECRET_SENTINEL" not in caplog.text
    assert "URL_SENTINEL" not in caplog.text
    assert "private.invalid/raw" not in caplog.text
    assert response.json()["error"]["code"] == -32603
    assert not list(firestore_fake.client_class().collection("dialog_operations").stream())


def test_add_dialog_replay_survives_changed_explicit_folder_lookup(monkeypatch, sdk_fakes, firestore_fake):
    from services.mcp.app.mcp import handler

    client = firestore_fake.client_class()
    client.collection("folders").document("replay-folder").set({
        "folder_id": "replay-folder", "name": "Replay folder", "parent_id": "pytest-synthetic-root",
    })
    _install_document_boundary(monkeypatch, sdk_fakes)
    session_id = _initialized_session()
    arguments = {
        "provider": "manual", "utterances": [{"speaker": "A", "text": "Stable replay."}],
        "title": "Stable replay", "dialog_date": "2026-10-22", "folder_path": ["Replay folder"],
    }
    first = _call_add_dialog(arguments, "folder-replay", session_id=session_id)
    assert first.status_code == 200
    first_payload = _payload(first)
    assert first_payload["status"] in {"indexing", "synced"}

    def unavailable(_client):
        raise RuntimeError("SECRET_FOLDER_LOOKUP")

    monkeypatch.setattr(handler.folder_repository, "get_all_folders", unavailable)
    replay = _call_add_dialog(arguments, "folder-replay", session_id=session_id)
    assert replay.status_code == 200
    replay_payload = _payload(replay)
    assert replay_payload["operation_id"] == first_payload["operation_id"]
    assert replay_payload["doc_id"] == first_payload["doc_id"]
    assert "SECRET_FOLDER_LOOKUP" not in replay.text


@pytest.mark.parametrize(
    "arguments",
    [
        {"utterances": [{"speaker": "A", "text": "words"}]},
        {"provider": "chatgpt"},
        {"provider": " ", "utterances": [{"speaker": "A", "text": "words"}]},
        {"provider": "chatgpt\nclaude", "utterances": [{"speaker": "A", "text": "words"}]},
        {"provider": 17, "utterances": [{"speaker": "A", "text": "words"}]},
        {"provider": "chatgpt", "utterances": []},
        {"provider": "chatgpt", "utterances": [{"speaker": " ", "text": "words"}]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": " "}]},
        {"provider": "chatgpt", "utterances": [{"speaker": "::", "text": "words"}]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": 5}]},
        {"provider": "chatgpt", "utterances": [{"text": "words"}]},
        {"provider": "chatgpt", "utterances": ["A :: words"]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "title": " "},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "dialog_date": "2025-02-30"},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "folder_path": [" "]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "folder_path": "Folder"},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "folder_path": [7]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "folder_path": ["missing-explicit-folder"]},
        {"provider": "chatgpt", "utterances": [{"speaker": "A", "text": "words"}], "unknown": True},
    ],
)
def test_invalid_add_dialog_arguments_return_invalid_params_without_side_effects(
    arguments, sdk_fakes, firestore_fake,
):
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog(arguments, "invalid-add-dialog")
    assert response.status_code == 200
    assert response.json()["id"] == "invalid-add-dialog"
    assert response.json()["error"]["code"] == -32602
    assert not any(
        event["boundary"] == "drive" and event["operation"] == "files.create"
        for event in sdk_fakes.trace
    )
    assert not any(event["boundary"] == "gemini" for event in sdk_fakes.trace)
    assert list(firestore_fake.client_class().collection("folders").stream()) == []


def test_ambiguous_explicit_folder_path_returns_invalid_params_without_creating_a_folder(
    sdk_fakes, firestore_fake,
):
    client = firestore_fake.client_class()
    for folder_id in ("duplicate-a", "duplicate-b"):
        client.collection("folders").document(folder_id).set({
            "folder_id": folder_id,
            "name": "Duplicate",
            "parent_id": "pytest-synthetic-root",
        })
    before = {snapshot.id: snapshot.to_dict() for snapshot in client.collection("folders").stream()}
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog({
        "provider": "chatgpt",
        "utterances": [{"speaker": "A", "text": "words"}],
        "folder_path": ["Duplicate"],
    }, "ambiguous-path")
    assert response.json()["id"] == "ambiguous-path"
    assert response.json()["error"]["code"] == -32602
    assert {snapshot.id: snapshot.to_dict() for snapshot in client.collection("folders").stream()} == before
    assert not any(
        event["boundary"] == "drive" and event["operation"] == "files.create"
        for event in sdk_fakes.trace
    )
    assert not any(event["boundary"] == "gemini" for event in sdk_fakes.trace)


def test_fresh_invalid_explicit_folder_path_does_not_reserve_request(monkeypatch, firestore_fake):
    from core.integrations.firestore import mcp_sessions

    session_id = _initialized_session()
    response = _call_add_dialog({
        "provider": "manual", "utterances": [{"speaker": "A", "text": "words"}],
        "folder_path": ["missing-folder"],
    }, "invalid-folder-no-reservation", session_id=session_id)
    assert response.json()["error"]["code"] == -32602
    request_documents = (
        firestore_fake.client_class().collection(mcp_sessions.MCP_SESSIONS_COLLECTION)
        .document(mcp_sessions._session_hash(session_id)).collection("requests").stream()
    )
    assert list(request_documents) == []


def test_explicit_empty_folder_path_selects_root_and_reports_accepted_index_task(
    monkeypatch, sdk_fakes, firestore_fake,
):
    files, documents, created_docs = _install_document_boundary(monkeypatch, sdk_fakes)
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog({
        "provider": "my-custom-origin-v2",
        "utterances": [{"speaker": "Ada", "text": "A distinct conversation."}],
        "title": "A review conversation",
        "dialog_date": "2025-04-03",
        "folder_path": [],
    }, "root-add-dialog")
    assert response.status_code == 200
    payload = _payload(response)
    assert payload["provider"] == "my-custom-origin-v2"
    assert payload["status"] == "indexing"
    assert payload["operation_id"]
    assert payload["doc_id"]
    assert payload["folder_path"] == []
    assert not {"url", "link", "webViewLink"} & payload.keys()

    drive_creates = [
        event for event in sdk_fakes.trace
        if event["boundary"] == "drive" and event["operation"] == "files.create"
    ]
    docs_writes = [
        event for event in sdk_fakes.trace
        if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"
    ]
    assert len(drive_creates) == 1
    assert docs_writes
    assert drive_creates[0]["arguments"]["body"]["parents"] == ["pytest-synthetic-root"]
    assert drive_creates[0]["arguments"]["body"]["mimeType"] == "application/vnd.google-apps.document"
    assert payload["doc_id"] in created_docs
    assert files[payload["doc_id"]]["modifiedTime"]
    assert files[payload["doc_id"]]["appProperties"]["tlbrain_save_id"]
    assert "A distinct conversation." in documents[payload["doc_id"]]["text"]
    assert payload["provider"] in documents[payload["doc_id"]]["text"]
    assert documents[payload["doc_id"]]["revisionId"]
    inserted_text = "".join(
        request["insertText"]["text"]
        for event in docs_writes
        for request in event["arguments"]["body"].get("requests", [])
        if "insertText" in request
    )
    assert inserted_text.count("A distinct conversation.") == 1
    assert not any(event["boundary"] == "gemini" for event in sdk_fakes.trace)

    task_events = [
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
    ]
    vector_url = os.environ["VECTOR_SYNC_URL"]
    index_tasks = [
        event for event in task_events
        if event["arguments"]["task"]["http_request"]["url"].startswith(vector_url)
    ]
    assert len(index_tasks) == 1
    task_body = json.loads(index_tasks[0]["arguments"]["task"]["http_request"]["body"])
    assert task_body["schema_version"] == 1
    assert task_body["target_kind"] == "index_dispatch"
    assert task_body["document_id"] == payload["doc_id"]
    assert task_body["operation_id"] == payload["operation_id"]
    assert task_body["save_intent_id"]

    index_record = firestore_fake.client_class().collection("transcript_index").document(payload["doc_id"]).get()
    assert index_record.exists
    assert index_record.to_dict()["provider"] == payload["provider"]
    assert index_record.to_dict()["status"] in {"queued", "downloading", "imported", "syncing"}
    assert index_record.to_dict()["status"] != "synced"
    assert not any(event["boundary"] in {"qdrant", "gemini", "sparse"} for event in sdk_fakes.trace)


def test_omitted_folder_path_uses_shared_unassigned_classification(
    monkeypatch, sdk_fakes,
):
    files, documents, created_docs = _install_document_boundary(monkeypatch, sdk_fakes)
    response = _call_add_dialog({
        "provider": "custom-source",
        "utterances": [{"speaker": "Ada", "text": "Automatically classified conversation."}],
        "title": "Automatic classification",
        "dialog_date": "2025-04-03",
    }, "automatic-folder-add-dialog")

    assert response.status_code == 200
    payload = _payload(response)
    assert payload["status"] == "indexing"
    assert payload["folder_path"] == ["_unassigned"]
    assert payload["doc_id"] in created_docs
    parent_id = files[payload["doc_id"]]["parents"][0]
    assert files[parent_id]["name"] == "_unassigned"
    assert "Automatically classified conversation." in documents[payload["doc_id"]]["text"]


def test_identical_content_with_distinct_request_ids_creates_distinct_documents_in_one_session(
    monkeypatch, sdk_fakes,
):
    _install_document_boundary(monkeypatch, sdk_fakes)
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    session_id = _initialized_session()
    args = {
        "provider": "manual",
        "utterances": [{"speaker": "Ada", "text": "Same content does not mean same dialog."}],
        "folder_path": [],
    }
    first = _payload(_call_add_dialog(args, "same-dialog-first", session_id=session_id))
    second = _payload(_call_add_dialog(args, "same-dialog-second", session_id=session_id))
    assert first["operation_id"] != second["operation_id"]
    assert first["doc_id"] != second["doc_id"]


def test_definitive_provider_rejection_returns_safe_correlated_error_without_raw_details(
    monkeypatch, sdk_fakes,
):
    provider_error = HttpError(
        httplib2.Response({"status": "403"}),
        b'{"error":"SECRET_SENTINEL URL_SENTINEL https://provider.invalid/raw"}',
        uri="https://drive.google.com/drive/v3/files",
    )
    _install_document_boundary(monkeypatch, sdk_fakes, create_error=provider_error)
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog({
        "provider": "manual",
        "utterances": [{"speaker": "Ada", "text": "A provider rejection case."}],
        "title": "Rejected conversation",
        "dialog_date": "2025-04-03",
        "folder_path": [],
    }, "provider-rejection")
    body = response.json()
    assert body["id"] == "provider-rejection"
    assert "error" in body
    serialized = json.dumps(body)
    assert "SECRET_SENTINEL" not in serialized
    assert "URL_SENTINEL" not in serialized
    assert "provider.invalid/raw" not in serialized
    diagnostic_fields = [item for item in _mappings(body["error"])]
    assert any(item.get("operation_id") for item in diagnostic_fields)
    assert any(item.get("service") for item in diagnostic_fields)
    assert any(item.get("stage") for item in diagnostic_fields)
    assert _has_safe_error_category(diagnostic_fields)
    assert any(item.get("sanitized_message") or item.get("message") for item in diagnostic_fields)


def test_vector_sync_dispatch_failure_reports_created_document_and_safe_error(
    monkeypatch, sdk_fakes,
):
    files, documents, created_docs = _install_document_boundary(monkeypatch, sdk_fakes)
    from tests.fixtures.sdk_fakes import FakeCloudTasksClient

    original_create_task = FakeCloudTasksClient.create_task
    vector_url = os.environ["VECTOR_SYNC_URL"]

    def fail_vector_sync(self, *, parent, task, **kwargs):
        url = task.get("http_request", {}).get("url", "")
        if url.startswith(vector_url):
            from tests.fixtures import sdk_fakes as sdk_fake_module

            sdk_fake_module._record("tasks", "create_task", parent=parent, task=task)
            raise RuntimeError("SECRET_SENTINEL dispatch rejected URL_SENTINEL https://queue.invalid/raw")
        return original_create_task(self, parent=parent, task=task, **kwargs)

    monkeypatch.setattr(FakeCloudTasksClient, "create_task", fail_vector_sync)
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog({
        "provider": "claude",
        "utterances": [{"speaker": "Grace", "text": "The document exists after dispatch failure."}],
        "title": "Dispatch failure conversation",
        "dialog_date": "2025-04-03",
        "folder_path": [],
    }, "dispatch-rejection")
    body = response.json()
    assert body["id"] == "dispatch-rejection"
    payload = _payload(response)
    assert payload["status"] == "created"
    assert payload["doc_id"] in created_docs
    assert documents[payload["doc_id"]]["text"]
    assert payload["folder_path"] == []
    serialized = json.dumps(body)
    assert "SECRET_SENTINEL" not in serialized
    assert "URL_SENTINEL" not in serialized
    assert "queue.invalid/raw" not in serialized
    diagnostic_fields = list(_mappings(payload))
    assert any(item.get("operation_id") for item in diagnostic_fields)
    assert any(item.get("service") for item in diagnostic_fields)
    assert any(item.get("stage") for item in diagnostic_fields)
    assert _has_safe_error_category(diagnostic_fields)
    assert any(item.get("sanitized_message") or item.get("message") for item in diagnostic_fields)


def _seed_index_backlog(client, count: int):
    from core.integrations.firestore.outbox import create_outbox_record

    schedule_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        create_outbox_record(
            client, f"backlog-index-{index}", target_kind="index_dispatch", generation=1,
            schedule_at=schedule_at, now=schedule_at, operation_id=f"backlog-operation-{index}",
        )


def test_add_dialog_status_uses_own_outbox_when_record_is_outside_sweep_limit(
    monkeypatch, sdk_fakes, firestore_fake,
):
    from core.integrations.firestore.outbox import get_outbox_record
    from core.integrations.firestore.save_intents import get_save_intent
    from core.integrations.firestore.operations import get_operation

    _install_document_boundary(monkeypatch, sdk_fakes)
    client = firestore_fake.client_class()
    _seed_index_backlog(client, 11)
    response = _call_add_dialog({
        "provider": "manual", "utterances": [{"speaker": "A", "text": "Backlog status."}],
        "title": "Backlog status", "dialog_date": "2026-10-22", "folder_path": [],
    }, "backlog-status")
    payload = _payload(response)
    operation = get_operation(client, payload["operation_id"])
    intent = get_save_intent(client, operation.save_intent_id)
    own_record = get_outbox_record(client, intent.index_checkpoint_ref)
    assert own_record.state.value == "pending"
    assert payload["status"] == "created"


def test_add_dialog_status_ignores_unrelated_outbox_dispatch_failure(
    monkeypatch, sdk_fakes, firestore_fake,
):
    from tests.fixtures.sdk_fakes import FakeCloudTasksClient

    _install_document_boundary(monkeypatch, sdk_fakes)
    client = firestore_fake.client_class()
    _seed_index_backlog(client, 1)
    original_create_task = FakeCloudTasksClient.create_task
    did_fail = False
    vector_url = os.environ["VECTOR_SYNC_URL"]

    def fail_backlog_once(self, *, parent, task, **kwargs):
        nonlocal did_fail
        if not did_fail and task.get("http_request", {}).get("url", "").startswith(vector_url):
            did_fail = True
            raise RuntimeError("unrelated dispatch failure")
        return original_create_task(self, parent=parent, task=task, **kwargs)

    monkeypatch.setattr(FakeCloudTasksClient, "create_task", fail_backlog_once)
    response = _call_add_dialog({
        "provider": "manual", "utterances": [{"speaker": "A", "text": "Own status."}],
        "title": "Own status", "dialog_date": "2026-10-22", "folder_path": [],
    }, "own-status")
    assert did_fail
    assert _payload(response)["status"] == "indexing"


def test_add_dialog_lease_release_exception_is_absent_from_logs(monkeypatch, sdk_fakes, caplog):
    from services.mcp.app.mcp import handler

    _install_document_boundary(monkeypatch, sdk_fakes)

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("SECRET_RELEASE_SENTINEL provider-data")

    monkeypatch.setattr(handler, "release_operation", fail_release)
    response = _call_add_dialog({
        "provider": "manual", "utterances": [{"speaker": "A", "text": "Release log safety."}],
        "title": "Release log safety", "dialog_date": "2026-10-22", "folder_path": [],
    }, "safe-release-log")
    assert response.status_code == 200
    assert response.json()["id"] == "safe-release-log"
    assert "SECRET_RELEASE_SENTINEL" not in caplog.text
    assert "provider-data" not in caplog.text


def test_docs_fill_rejection_preserves_known_document_and_safe_error(monkeypatch, sdk_fakes):
    fill_error = HttpError(
        httplib2.Response({"status": "403"}),
        b'{"error":"SECRET_SENTINEL URL_SENTINEL https://docs.invalid/raw"}',
        uri="https://docs.googleapis.com/v1/documents",
    )
    files, _, created_docs = _install_document_boundary(monkeypatch, sdk_fakes, fill_error=fill_error)
    tools = _post(_rpc("tools/list", "confirm-add-dialog-tool")).json()["result"]["tools"]
    assert "add_dialog" in {tool["name"] for tool in tools}
    response = _call_add_dialog({
        "provider": "manual",
        "utterances": [{"speaker": "Ada", "text": "Docs fill will be rejected."}],
        "title": "Docs rejection conversation",
        "dialog_date": "2025-04-03",
        "folder_path": [],
    }, "docs-fill-rejection")
    body = response.json()
    assert body["id"] == "docs-fill-rejection"
    serialized = json.dumps(body)
    assert "SECRET_SENTINEL" not in serialized
    assert "URL_SENTINEL" not in serialized
    assert "docs.invalid/raw" not in serialized
    created_ids = created_docs
    assert len(created_ids) == 1
    known_doc_id = next(iter(created_ids))
    result = body.get("result", {})
    checked = _payload(response) if result else body["error"]
    checked_fields = list(_mappings(checked))
    assert any(item.get("doc_id") == known_doc_id for item in checked_fields)
    assert any(item.get("folder_path") == [] for item in checked_fields)
    vector_url = os.environ["VECTOR_SYNC_URL"]
    assert not any(
        event["boundary"] == "tasks"
        and event["operation"] == "create_task"
        and event["arguments"]["task"]["http_request"]["url"].startswith(vector_url)
        for event in sdk_fakes.trace
    )
    assert any(item.get("operation_id") for item in checked_fields)
    assert any(item.get("service") for item in checked_fields)
    assert any(item.get("stage") for item in checked_fields)
    assert _has_safe_error_category(checked_fields)
    assert any(item.get("sanitized_message") or item.get("message") for item in checked_fields)
    if result:
        result_payload = json.loads(result["content"][0]["text"])
        assert result_payload.get("status") == "error"
        assert result_payload.get("status") not in {"indexing", "synced"}


def test_document_boundary_supports_canonical_writer_reader_and_metadata_helpers(
    monkeypatch, sdk_fakes,
):
    files, documents, _created_docs = _install_document_boundary(monkeypatch, sdk_fakes)
    from googleapiclient.discovery import build

    from core.integrations.google_docs import reader, writer
    from core.integrations.google_drive.files import get_file_metadata

    drive = build("drive", "v3")
    docs = build("docs", "v1")
    empty_file = drive.files().create(body={
        "name": "Empty first",
        "mimeType": "application/vnd.google-apps.document",
        "parents": ["pytest-synthetic-root"],
        "appProperties": {"tlbrain_save_id": "synthetic-save-id"},
    }, fields="id,modifiedTime").execute()
    assert reader.read_google_doc(empty_file["id"]) == ""
    assert get_file_metadata(empty_file["id"])["modifiedTime"] == empty_file["modifiedTime"]
    empty_document = docs.documents().get(documentId=empty_file["id"]).execute()
    assert empty_document["revisionId"] == "pytest-docs-revision-0"
    with pytest.raises(RuntimeError, match="Stale Docs revision"):
        docs.documents().batchUpdate(
            documentId=empty_file["id"],
            body={
                "requests": [{"insertText": {"location": {"index": 1}, "text": "stale"}}],
                "writeControl": {"requiredRevisionId": "stale-revision"},
            },
        ).execute()
    docs.documents().batchUpdate(
        documentId=empty_file["id"],
        body={
            "requests": [{"insertText": {
                "location": {"index": 1}, "text": "Ada :: Read after an empty document."
            }}],
            "writeControl": {"requiredRevisionId": empty_document["revisionId"]},
        },
    ).execute()
    assert reader.read_google_doc(empty_file["id"]) == "Ada :: Read after an empty document."
    filled_file = get_file_metadata(empty_file["id"])
    assert filled_file["modifiedTime"] != empty_file["modifiedTime"]
    assert get_file_metadata(empty_file["id"])["modifiedTime"] == filled_file["modifiedTime"]

    doc_id, modified_time = writer.create_google_doc(
        drive, docs, "pytest-synthetic-root", "Boundary smoke", "Ada :: Hello", [(0, 3)]
    )
    assert reader.read_google_doc(doc_id) == "Ada :: Hello"
    metadata = get_file_metadata(doc_id)
    assert metadata["id"] == doc_id
    assert metadata["mimeType"] == "application/vnd.google-apps.document"
    assert metadata["parents"] == ["pytest-synthetic-root"]
    assert metadata["modifiedTime"] != modified_time
    assert documents[doc_id]["revisionId"]
    assert files[doc_id]["name"] == "Boundary smoke"


def test_no_argument_sync_status_keeps_legacy_aggregate_suggestion_and_is_read_only(
    firestore_fake, sdk_fakes,
):
    client = firestore_fake.client_class()
    for doc_id, status in (
        ("queued", "queued"),
        ("downloading", "downloading"),
        ("imported", "imported"),
        ("syncing", "syncing"),
        ("unassigned", "synced"),
        ("error", "error"),
    ):
        data = {"status": status, "client_name": "_unassigned" if doc_id == "unassigned" else "Assigned"}
        client.collection("transcript_index").document(doc_id).set(data)
    client.collection("mcp_sessions").document("session-status-like").set({"status": "synced"})
    client.collection("dialog_operations").document("operation-status-like").set({"status": "error"})
    client.collection("task_outbox").document("outbox-status-like").set({"status": "queued"})
    before = firestore_fake.documents
    committed_writes_before = [
        entry for entry in firestore_fake.trace() if entry.get("outcome") == "committed"
    ]

    response = _post(_rpc("tools/call", "legacy-sync-status", {"name": "sync_status", "arguments": {}}))
    payload = _payload(response)
    assert {key: value for key, value in payload.items() if key != "suggestion"} == {
        "total": 6, "queued": 1, "downloading": 1, "imported": 1,
        "syncing": 1, "synced": 1, "error": 1,
    }
    assert payload["suggestion"] == (
        'Assigning transcripts to the correct folder improves search accuracy. '
        '1 transcript(s) are currently unassigned — call '
        'list_recent_transcripts(folder_path=["_unassigned"]) to see them, '
        "then move each one using move_transcript(doc_id='...', folder_path=[...])."
    )
    assert firestore_fake.documents == before
    committed_writes_after = [
        entry for entry in firestore_fake.trace() if entry.get("outcome") == "committed"
    ]
    assert committed_writes_after == committed_writes_before
    assert not any(event["boundary"] in {"drive", "docs", "tasks", "qdrant", "gemini"} for event in sdk_fakes.trace)


def test_no_argument_sync_status_omits_suggestion_when_every_dialog_is_assigned(firestore_fake):
    client = firestore_fake.client_class()
    client.collection("transcript_index").document("assigned").set({"status": "synced", "client_name": "Acme"})
    response = _post(_rpc("tools/call", "legacy-sync-status-no-suggestion", {
        "name": "sync_status", "arguments": {},
    }))
    assert _payload(response) == {
        "total": 1, "queued": 0, "downloading": 0, "imported": 0,
        "syncing": 0, "synced": 1, "error": 0,
    }


def test_add_dialog_missing_operation_after_publish_returns_in_progress(monkeypatch):
    from types import SimpleNamespace

    from services.mcp.app.mcp import handler

    context = SimpleNamespace(identity=SimpleNamespace(caller_binding="caller"))
    reservation = SimpleNamespace(operation_id="missing-operation")
    prepared = (context, {"provider": "google"}, object(), None, {}, None, reservation)
    monkeypatch.setattr(handler, "_prepare_add_dialog", lambda *_args: (prepared, None))

    saving_operation = SimpleNamespace(state=SimpleNamespace(value="saving"))
    operation_results = iter((saving_operation, None))
    monkeypatch.setattr(handler, "get_operation", lambda *_args: next(operation_results))
    monkeypatch.setattr(handler, "claim_operation", lambda *_args, **_kwargs: SimpleNamespace(fence=1))
    monkeypatch.setattr(handler, "_advance_add_dialog_save", lambda *_args: object())
    monkeypatch.setattr(handler, "_publish_add_dialog_index_work", lambda *_args: None)
    monkeypatch.setattr(handler, "release_operation", lambda *_args, **_kwargs: None)

    response = handler._process_add_dialog(SimpleNamespace(id="missing-operation-request"), {})

    assert response["error"]["code"] == -32010
    assert response["error"]["data"]["operation_id"] == "missing-operation"
