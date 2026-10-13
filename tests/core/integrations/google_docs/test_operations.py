from __future__ import annotations

import pytest

from core.integrations.google_docs import client, reader, writer


def _set_oauth(monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-secret")


def test_docs_client_uses_existing_user_grant_and_default_discovery_transport(
    monkeypatch, sdk_fakes
):
    _set_oauth(monkeypatch)
    scopes = ["https://www.googleapis.com/auth/documents.readonly"]

    service = client.build_docs_service(scopes)

    assert service is not None
    build = next(entry for entry in sdk_fakes.trace if entry["operation"] == "build")
    assert build["arguments"]["api"] == "docs"
    assert build["arguments"]["version"] == "v1"
    assert "http" not in build["arguments"]
    assert sdk_fakes.trace[0]["arguments"]["scopes"] is None
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("auth", "credentials.refresh"),
        ("drive", "build"),
    ]


def test_reader_uses_strict_oauth_and_extracts_nonempty_paragraphs(monkeypatch, sdk_fakes):
    _set_oauth(monkeypatch)
    sdk_fakes.docs.queue("documents.get", response={"body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "First\n"}}]}},
        {"paragraph": {"elements": [{"textRun": {"content": "  \n"}}]}},
        {"paragraph": {"elements": [{"textRun": {"content": "Second"}}]}},
        {"table": {}},
    ]}})

    assert reader.read_google_doc("doc-1") == "First\nSecond"
    auth = next(entry for entry in sdk_fakes.trace if entry["boundary"] == "auth")
    assert auth["operation"] == "credentials.refresh"
    assert auth["arguments"]["scopes"] is None
    assert reader.SCOPES == ["https://www.googleapis.com/auth/documents.readonly"]
    assert sdk_fakes.trace[-1]["operation"] == "documents.get"


def test_reader_refuses_adc_fallback(monkeypatch, sdk_fakes):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_REFRESH_TOKEN is not set"):
        reader.read_google_doc("doc-1")

    assert sdk_fakes.trace == []


def test_legacy_reader_module_forwards_to_canonical_reader():
    from core.google_drive import docs_reader as legacy_reader

    assert legacy_reader.read_google_doc is reader.read_google_doc
    assert legacy_reader._build_docs_service is reader._build_docs_service
    assert legacy_reader._extract_text is reader._extract_text


def test_writer_creates_then_fills_doc_with_shifted_bold_ranges(sdk_fakes):
    sdk_fakes.drive.queue(
        "files.create",
        response={"id": "doc-1", "modifiedTime": "synthetic-time"},
    )
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    drive = __import__("googleapiclient.discovery", fromlist=["build"]).build("drive", "v3")
    docs = __import__("googleapiclient.discovery", fromlist=["build"]).build("docs", "v1")

    result = writer.create_google_doc(
        drive, docs, "folder-1", "Meeting", "Alex :: Hello", [(0, 4)]
    )

    assert result == ("doc-1", "synthetic-time")
    operations = [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace]
    assert operations == [
        ("drive", "build"),
        ("drive", "build"),
        ("drive", "files.create"),
        ("docs", "documents.batchUpdate"),
    ]
    create = sdk_fakes.trace[2]["arguments"]
    assert create == {
        "body": {
            "name": "Meeting",
            "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-1"],
        },
        "fields": "id,modifiedTime",
    }
    update = sdk_fakes.trace[3]["arguments"]
    assert update["documentId"] == "doc-1"
    assert update["body"]["requests"] == [
        {"insertText": {"location": {"index": 1}, "text": "Alex :: Hello"}},
        {"updateTextStyle": {
            "range": {"startIndex": 1, "endIndex": 5},
            "textStyle": {"bold": True},
            "fields": "bold",
        }},
    ]
