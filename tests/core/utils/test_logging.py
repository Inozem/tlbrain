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


def test_json_records_preserve_safe_correlation_and_status_fields(capsys):
    configure_logging()
    logger = logging.getLogger("logging.safe_extras")
    secret = "synthetic-secret-marker"

    logger.warning(
        "stage completed",
        extra={
            "stage": "index",
            "result": "success",
            "doc_id": "synthetic-doc-safe",
            "meeting_id": "synthetic-meeting-safe",
            "request_id": "synthetic-request-safe",
            "response_status": 207,
            "token_count": 4,
            "processing_state": "complete",
            "access_token": secret,
            "client_secret": secret,
        },
    )

    [event] = _read_json_lines(capsys.readouterr().err)
    assert event["stage"] == "index"
    assert event["result"] == "success"
    assert event["doc_id"] == "synthetic-doc-safe"
    assert event["meeting_id"] == "synthetic-meeting-safe"
    assert event["request_id"] == "synthetic-request-safe"
    assert event["response_status"] == 207
    assert event["token_count"] == 4
    assert event["processing_state"] == "complete"
    assert "access_token" not in event
    assert "client_secret" not in event
    assert secret not in json.dumps(event)


def test_nested_sensitive_extra_key_is_omitted_and_safe_siblings_survive(capsys):
    configure_logging()
    logger = logging.getLogger("logging.nested_extras")
    secret = "synthetic-opaque-nested-secret"
    request_id = "synthetic-nested-request"

    logger.error(
        "proxy request failed",
        extra={
            "proxy": {
                "access_token": secret,
                "request_id": request_id,
                "status": 502,
            }
        },
    )

    [event] = _read_json_lines(capsys.readouterr().err)
    assert secret not in json.dumps(event)
    assert "access_token" not in event["proxy"]
    assert event["proxy"]["request_id"] == request_id
    assert event["proxy"]["status"] == 502
