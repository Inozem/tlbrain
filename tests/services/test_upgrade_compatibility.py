from __future__ import annotations

import pytest

from core.integrations.google_docs import writer
from services.connectors.tldv import formatting


@pytest.mark.offline
def test_tldv_formatting_output_writes_through_canonical_docs_writer(sdk_fakes):
    meeting = {
        "id": "meeting-compatibility",
        "name": "Contract meeting",
        "happenedAt": "2025-02-03T10:15:00Z",
    }
    utterances = [
        {"speaker": "Alex::", "text": "First line."},
        {"speaker": "Alex", "text": "Second line."},
    ]
    expected_text = (
        "DATE: 2025-02-03\n"
        "TIME: 10:15\n"
        "PROVIDER: tldv\n"
        "---\n"
        "Alex :: First line.\n\n"
        "Alex :: Second line."
    )
    expected_speaker_ranges = [(48, 52), (69, 73)]

    text, speaker_ranges = formatting._format_transcript(meeting, utterances)

    assert (text, speaker_ranges) == (expected_text, expected_speaker_ranges)

    sdk_fakes.drive.queue(
        "files.create",
        response={"id": "doc-compatibility", "modifiedTime": "2025-02-03T10:20:00Z"},
    )
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    drive = __import__("googleapiclient.discovery", fromlist=["build"]).build("drive", "v3")
    docs = __import__("googleapiclient.discovery", fromlist=["build"]).build("docs", "v1")

    result = writer.create_google_doc(
        drive, docs, "folder-alex", meeting["name"], text, speaker_ranges
    )

    assert result == ("doc-compatibility", "2025-02-03T10:20:00Z")
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("drive", "build"),
        ("drive", "build"),
        ("drive", "files.create"),
        ("docs", "documents.batchUpdate"),
    ]
    create_call = next(
        entry for entry in sdk_fakes.trace
        if entry["boundary"] == "drive" and entry["operation"] == "files.create"
    )
    assert create_call["arguments"] == {
        "body": {
            "name": "Contract meeting",
            "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-alex"],
        },
        "fields": "id,modifiedTime",
    }

    update_call = next(
        entry for entry in sdk_fakes.trace
        if entry["boundary"] == "docs" and entry["operation"] == "documents.batchUpdate"
    )
    requests_body = update_call["arguments"]["body"]["requests"]
    assert update_call["arguments"] == {
        "documentId": "doc-compatibility",
        "body": {
            "requests": [
                {"insertText": {"location": {"index": 1}, "text": expected_text}},
                {"updateTextStyle": {
                    "range": {"startIndex": 49, "endIndex": 53},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }},
                {"updateTextStyle": {
                    "range": {"startIndex": 70, "endIndex": 74},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }},
            ]
        },
    }
    assert requests_body[0]["insertText"]["text"] == expected_text
