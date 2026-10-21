"""Canonical dialog encoding, UTF-16 styling ranges, and parser compatibility."""

from __future__ import annotations

import json

import pytest


def _format_dialog(normalized, *, dialog_time: str = ""):
    from core.dialogs.formatting import format_dialog

    return format_dialog(normalized, dialog_time=dialog_time)


def _normalize(arguments, accepted_at):
    from core.dialogs.models import normalize_dialog_input

    return normalize_dialog_input(arguments, accepted_at=accepted_at)


def _parse_document(text: str):
    from core.parsing.parser import parse_document

    return parse_document(text)


def _is_valid_format(text: str):
    from core.parsing.parser import is_valid_format

    return is_valid_format(text)


def test_declared_json_string_encoding_decodes_exact_text_and_metadata():
    original = "  lead\r\n\t\nline\n\n\"quoted\" " + r"\n" + " :: \U0001f600  "
    encoded = json.dumps(original, ensure_ascii=False)
    document = (
        "DATE: 2026-01-09\nTIME: 14:20\nPROVIDER: custom\n"
        "TEXT_ENCODING: json-string-v1\n---\n"
        f"\U0001f9d1\u200d\U0001f4bb :: {encoded}\n"
    )

    metadata, utterances = _parse_document(document)
    assert metadata == {"date": "2026-01-09", "time": "14:20", "provider": "custom"}
    assert utterances == [
        {"speaker": "\U0001f9d1\u200d\U0001f4bb", "text": original, "order_index": 0}
    ]
    assert _is_valid_format(document)


@pytest.mark.parametrize(
    "body",
    [
        "Speaker :: {not-json}",
        "Speaker :: 42",
        "Speaker :: null",
        "Speaker :: []",
    ],
)
def test_invalid_declared_encoded_body_is_a_format_error(body: str):
    document = f"DATE: 2026-01-09\nPROVIDER: custom\nTEXT_ENCODING: json-string-v1\n---\n{body}"
    with pytest.raises(ValueError):
        _parse_document(document)
    assert not _is_valid_format(document)


def test_unknown_declared_encoding_is_a_format_error():
    document = "DATE: 2026-01-09\nPROVIDER: custom\nTEXT_ENCODING: base64\n---\nSpeaker :: eA=="
    with pytest.raises(ValueError):
        _parse_document(document)
    assert not _is_valid_format(document)


def test_legacy_transcript_without_encoding_keeps_literal_body_and_optional_time():
    document = (
        'DATE: 2026-01-09\nPROVIDER: tldv\n---\nA :: literal "quote" and '
        + r"\n"
        + "\nIgnored continuation\nB :: second"
    )
    assert _parse_document(document) == (
        {"date": "2026-01-09", "provider": "tldv"},
        [
            {"speaker": "A", "text": 'literal "quote" and ' + r"\n", "order_index": 0},
            {"speaker": "B", "text": "second", "order_index": 1},
        ],
    )
    assert _is_valid_format(document)


def test_canonical_formatter_uses_json_encoding_and_utf16_ranges(sdk_fakes):
    from datetime import datetime, timezone

    normalized = _normalize(
        {
            "provider": "manual",
            "dialog_date": "2026-01-09",
            "utterances": [
                {"speaker": "\U0001f600 Alice", "text": "body starts \U0001f600\n\ntrailing  "},
                {
                    "speaker": "\U0001f9d1\u200d\U0001f4bb Bob",
                    "text": "literal " + r"\n" + ", \"quotes\", ::, \t\r\n",
                },
            ],
        },
        datetime(2026, 1, 9, tzinfo=timezone.utc),
    )
    text, ranges = _format_dialog(normalized, dialog_time="07:08")

    assert "DATE: 2026-01-09" in text
    assert "TIME: 07:08" in text
    assert "PROVIDER: manual" in text
    assert "TEXT_ENCODING: json-string-v1" in text
    assert "---" in text
    assert "\U0001f600 Alice :: " in text
    assert json.dumps("body starts \U0001f600\n\ntrailing  ", ensure_ascii=False) in text
    assert "\U0001f600" in text
    assert _parse_document(text) == (
        {"date": "2026-01-09", "time": "07:08", "provider": "manual"},
        [
            {
                "speaker": "\U0001f600 Alice",
                "text": "body starts \U0001f600\n\ntrailing  ",
                "order_index": 0,
            },
            {
                "speaker": "\U0001f9d1\u200d\U0001f4bb Bob",
                "text": "literal " + r"\n" + ", \"quotes\", ::, \t\r\n",
                "order_index": 1,
            },
        ],
    )
    encoded = text.encode("utf-16-le")
    expected_labels = ["\U0001f600 Alice", "\U0001f9d1\u200d\U0001f4bb Bob"]
    assert len(ranges) == len(expected_labels)
    for (start, end), label in zip(ranges, expected_labels):
        assert encoded[start * 2 : end * 2].decode("utf-16-le") == label

    # The canonical writer applies the Docs API's one-based index shift.
    sdk_fakes.drive.queue("files.create", response={"id": "formatted-doc", "modifiedTime": "t"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    from googleapiclient.discovery import build
    from core.integrations.google_docs.writer import create_google_doc

    create_google_doc(build("drive", "v3"), build("docs", "v1"), "folder", "title", text, ranges)
    update = next(event for event in sdk_fakes.trace if event["operation"] == "documents.batchUpdate")
    styles = [
        request["updateTextStyle"]["range"]
        for request in update["arguments"]["body"]["requests"][1:]
    ]
    assert styles == [{"startIndex": start + 1, "endIndex": end + 1} for start, end in ranges]


def test_formatter_round_trips_through_real_reader_and_parser(monkeypatch, sdk_fakes):
    from datetime import datetime, timezone

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh")
    original = "# heading\n- item\n| x | y |\n\tcode\r\n\n  end  \U0001f600"
    normalized = _normalize(
        {
            "provider": "custom",
            "dialog_date": "2026-01-09",
            "utterances": [{"speaker": "\U0001f9d1\u200d\U0001f4bb Speaker", "text": original}],
        },
        datetime(2026, 1, 9, tzinfo=timezone.utc),
    )
    text, ranges = _format_dialog(normalized)
    assert ranges
    assert "\nTIME: \n" in text
    # Project formatted text into a Docs response and exercise the real reader.
    paragraphs = [
        {"paragraph": {"elements": [{"textRun": {"content": line + "\n"}}]}}
        for line in text.split("\n")
    ]
    sdk_fakes.docs.queue(
        "documents.get",
        response={"body": {"content": paragraphs}},
    )
    from core.integrations.google_docs.reader import read_google_doc

    restored = read_google_doc("round-trip-doc")
    projected = "\n".join(line for line in text.split("\n") if line.strip())
    assert restored == projected
    assert _parse_document(restored)[1] == [
        {"speaker": "\U0001f9d1\u200d\U0001f4bb Speaker", "text": original, "order_index": 0}
    ]


def test_tldv_formatter_output_and_codepoint_ranges_remain_legacy():
    from services.connectors.tldv.formatting import _format_transcript

    text, ranges = _format_transcript(
        {"happenedAt": "2026-01-09T14:20:00Z"},
        [{"speaker": "\U0001f600 A::B", "text": "first"}, {"speaker": "B", "text": "second"}],
    )
    header = "DATE: 2026-01-09\nTIME: 14:20\nPROVIDER: tldv\n---\n"
    first = "\U0001f600 AB :: first"
    second = "B :: second"
    assert text == header + first + "\n\n" + second
    assert ranges == [
        (len(header), len(header) + len("\U0001f600 AB")),
        (
            len(header + first + "\n\n"),
            len(header + first + "\n\n") + len("B"),
        ),
    ]
    assert "TEXT_ENCODING:" not in text
