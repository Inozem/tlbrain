import json
import logging

import pytest


def _json_response(types, value):
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[types.Part(text=json.dumps(value))]))
    ])


def test_trusted_single_attempt_policy_configures_sdk_and_wrapper(sdk_fakes):
    from google.genai import types
    from core.integrations.gemini.llm import call_gemini_json
    from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY

    sdk_fakes.gemini.queue(
        "generate_content", response=_json_response(types, {"folder_name": "Client", "confidence": 0.9}),
    )

    assert call_gemini_json("classify", retry_policy=SINGLE_ATTEMPT_RETRY_POLICY) == {
        "folder_name": "Client", "confidence": 0.9,
    }
    events = sdk_fakes.trace
    client = next(event for event in events if event["operation"] == "client")
    assert client["arguments"]["http_options"].retry_options.attempts == 1
    calls = [event for event in events if event["operation"] == "generate_content"]
    assert len(calls) == 1
    assert calls[0]["arguments"]["model"] == "gemini-2.5-flash"


def test_trusted_single_attempt_does_not_retry_or_log_provider_error(sdk_fakes, caplog):
    from core.integrations.gemini.llm import call_gemini_json
    from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY

    sdk_fakes.gemini.queue(
        "generate_content", error=ConnectionError("synthetic-provider-secret"),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(ConnectionError):
        call_gemini_json("classify", retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)

    assert len([event for event in sdk_fakes.trace
                if event["operation"] == "generate_content"]) == 1
    assert "synthetic-provider-secret" not in caplog.text


def test_legacy_call_keeps_default_client_and_retry_arguments(sdk_fakes):
    from google.genai import types
    from core.integrations.gemini.llm import call_gemini_json

    sdk_fakes.gemini.queue(
        "generate_content", response=_json_response(types, {"folder_name": "Client", "confidence": 0.9}),
    )

    assert call_gemini_json("legacy") == {"folder_name": "Client", "confidence": 0.9}
    client = next(event for event in sdk_fakes.trace if event["operation"] == "client")
    assert "http_options" not in client["arguments"]
