from __future__ import annotations

import pytest

from core.integrations.google_docs import client, reader, writer
from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY


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


def test_reader_preserves_encoded_native_paragraphs_exactly(monkeypatch, sdk_fakes):
    from core.dialogs.models import normalize_dialog_input
    from core.dialogs.formatting import format_dialog
    from core.indexing.hashing import sha256_text
    from datetime import datetime, timezone

    _set_oauth(monkeypatch)
    normalized = normalize_dialog_input(
        {
            "provider": "custom",
            "dialog_date": "2026-01-09",
            "utterances": [
                {"speaker": "Ada", "text": '  first\n\n  line  "quoted" ' + r"\\n" + "\U0001f600"},
                {"speaker": "Lin", "text": "second\u2028line\u2029next  "},
            ],
        },
        accepted_at=datetime(2026, 1, 9, tzinfo=timezone.utc),
    )
    expected, _ranges = format_dialog(normalized)
    content = []
    for index, line in enumerate(expected.split("\n")):
        runs = [{"textRun": {"content": line[: max(1, len(line) // 2)]}}]
        runs.append({"textRun": {"content": line[max(1, len(line) // 2):] + "\n"}})
        content.append({"paragraph": {"elements": runs}})
    sdk_fakes.docs.queue("documents.get", response={"body": {"content": [
        {"sectionBreak": {"sectionStyle": {}}},
        *content,
    ]}})

    actual = reader.read_google_doc("encoded-doc")

    assert actual == expected
    assert sha256_text(actual) == sha256_text(expected)


def test_reader_keeps_legacy_projection_and_requires_supported_encoded_structure():
    legacy = {"body": {"content": [
        {"paragraph": {"elements": [{"textRun": {
            "content": "DATE: 2026-01-09\nPROVIDER: tldv\n---\n"
            "TEXT_ENCODING: json-string-v1\n"
        }}]}},
        {"paragraph": {"elements": [{"textRun": {"content": "  \n"}}]}},
        {"paragraph": {"elements": [{"textRun": {"content": "A :: literal\n"}}]}},
    ]}}
    assert reader._extract_text(legacy) == (
        "DATE: 2026-01-09\nPROVIDER: tldv\n---\n"
        "TEXT_ENCODING: json-string-v1\nA :: literal"
    )

    malformed_encoded = {"body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content":
            "DATE: 2026-01-09\nPROVIDER: custom\n"
            "TEXT_ENCODING: json-string-v1\n---\nA :: \\\"ok\\\"\n"
        }}]}},
        {"table": {}},
    ]}}
    with pytest.raises(ValueError, match="Unsupported or malformed"):
        reader._extract_text(malformed_encoded)


def test_reader_consumes_explicit_retry_policy_at_decorated_boundary(monkeypatch, sdk_fakes):
    _set_oauth(monkeypatch)
    sdk_fakes.docs.queue("documents.get", response={"body": {"content": []}})

    assert reader.read_google_doc("doc-1", retry_policy=SINGLE_ATTEMPT_RETRY_POLICY) == ""
    get_event = next(entry for entry in sdk_fakes.trace if entry["operation"] == "documents.get")
    assert get_event["arguments"] == {"documentId": "doc-1"}


def test_reader_refuses_adc_fallback(monkeypatch, sdk_fakes):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_REFRESH_TOKEN is not set"):
        reader.read_google_doc("doc-1")

    assert sdk_fakes.trace == []


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
