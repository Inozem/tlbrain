"""Contract tests for the JSON log formatter used by services."""

from __future__ import annotations

import json
import logging

import pytest

from core.utils.logging import configure_logging


def _read_json_lines(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def test_configure_logging_emits_json_with_severity_message_and_extras(capsys):
    configure_logging()
    logger = logging.getLogger("logging.contract")

    logger.warning(
        "processed document %s",
        "synthetic-doc-17",
        extra={"stage": "index", "result": "success", "doc_id": "synthetic-doc-17"},
    )

    [event] = _read_json_lines(capsys.readouterr().err)
    assert event == {
        "severity": "WARNING",
        "message": "processed document synthetic-doc-17",
        "stage": "index",
        "result": "success",
        "doc_id": "synthetic-doc-17",
    }


@pytest.mark.parametrize(
    ("level", "message", "stage", "result", "identifier_key", "identifier"),
    [
        (logging.INFO, "stage completed", "import", "success", "meeting_id", "synthetic-meeting-ok"),
        (logging.INFO, "stage skipped", "sync", "skip", "doc_id", "synthetic-doc-skip"),
        (logging.ERROR, "stage failed", "download", "error", "meeting_id", "synthetic-meeting-error"),
        (logging.WARNING, "stage retrying", "index", "retry", "doc_id", "synthetic-doc-retry"),
    ],
)
def test_json_records_preserve_stage_result_and_document_or_meeting_id(
    capsys, level, message, stage, result, identifier_key, identifier
):
    configure_logging()
    logger = logging.getLogger("logging.lifecycle")

    logger.log(
        level,
        message,
        extra={"stage": stage, "result": result, identifier_key: identifier},
    )

    [event] = _read_json_lines(capsys.readouterr().err)
    assert event["severity"] == logging.getLevelName(level)
    assert event["message"] == message
    assert event["stage"] == stage
    assert event["result"] == result
    assert event[identifier_key] == identifier
