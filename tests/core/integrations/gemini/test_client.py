import json
from importlib import reload

import pytest

from core.integrations import gemini
from core.integrations.gemini import client


def test_package_and_client_import_do_not_construct_client(sdk_fakes):
    reload(gemini)
    reload(client)

    assert sdk_fakes.trace == []


def test_shared_factory_uses_api_key_and_does_not_cache_clients(sdk_fakes, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-client-test-key")

    first = client.make_client()
    second = client.make_client()

    assert first is not second
    client_events = [event for event in sdk_fakes.trace if event["operation"] == "client"]
    assert [event["arguments"]["api_key"] for event in client_events] == [
        "synthetic-client-test-key",
        "synthetic-client-test-key",
    ]


def test_single_attempt_factory_only_adds_http_retry_configuration(monkeypatch):
    from google.genai import types

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-client-test-key")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(client.genai, "Client", Client)

    client.make_client()
    client.make_client(single_attempt=True)

    assert calls[0] == {
        "api_key": "synthetic-client-test-key",
        "http_options": types.HttpOptions(api_version="v1"),
    }
    assert calls[1]["api_key"] == calls[0]["api_key"]
    assert calls[1]["http_options"] == types.HttpOptions(
        api_version="v1",
        retry_options=types.HttpRetryOptions(attempts=1)
    )


@pytest.mark.parametrize("single_attempt", [False, True])
def test_real_sdk_uses_v1_for_structured_requests_with_local_http(
    monkeypatch, single_attempt
):
    import httpx
    from google.genai import types
    from core.integrations.gemini.llm import _generate_structured
    from tests.fixtures.sdk_fakes import _REAL_GENAI_CLIENT

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-client-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1beta")
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{
                    "content": {
                        "parts": [{"text": '{"summary":"parsed","facts":["fixture"]}'}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-3.5-flash-lite",
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def local_client(**kwargs):
        options = kwargs["http_options"]
        kwargs["http_options"] = types.HttpOptions(
            api_version=options.api_version,
            retry_options=options.retry_options,
            httpx_client=http_client,
        )
        return _REAL_GENAI_CLIENT(**kwargs)

    monkeypatch.setattr(client.genai, "Client", local_client)

    sdk_client = client.make_client(single_attempt=single_attempt)
    try:
        result = _generate_structured(sdk_client, "test prompt")
    finally:
        sdk_client.close()

    assert result == {"summary": "parsed", "facts": ["fixture"]}
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/gemini-3.5-flash-lite:generateContent"
    body = json.loads(requests[0].read())
    generation_config = body["generationConfig"]
    assert generation_config["temperature"] == 0
    assert generation_config["topP"] == 1
    assert generation_config["responseFormat"] == {
        "text": {
            "mimeType": "APPLICATION_JSON",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "facts"],
            },
        },
    }
    assert "responseMimeType" not in generation_config
    assert "responseSchema" not in generation_config
    assert "responseJsonSchema" not in generation_config


def test_real_sdk_folder_classification_uses_nullable_v1_schema_and_one_attempt(
    monkeypatch,
):
    import httpx
    from google.genai import types
    from core.integrations.gemini import client as gemini_client
    from core.integrations.gemini import llm
    from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
    from tests.fixtures.sdk_fakes import _REAL_GENAI_CLIENT

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-client-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1beta")
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": '{"folder_name":null,"confidence":0.2}'}]}}],
        })

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def local_client(**kwargs):
        options = kwargs["http_options"]
        kwargs["http_options"] = types.HttpOptions(
            api_version=options.api_version,
            retry_options=options.retry_options,
            httpx_client=http_client,
        )
        return _REAL_GENAI_CLIENT(**kwargs)

    monkeypatch.setattr(gemini_client.genai, "Client", local_client)

    try:
        result = llm.call_gemini_json("classify", retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
    finally:
        http_client.close()

    assert result == {"folder_name": None, "confidence": 0.2}
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/gemini-3.5-flash-lite:generateContent"
    body = json.loads(requests[0].content)
    generation_config = body["generationConfig"]
    assert generation_config["responseFormat"]["text"] == {
        "mimeType": "APPLICATION_JSON",
        "schema": {
            "type": "object",
            "properties": {
                "folder_name": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["folder_name", "confidence"],
        },
    }
    assert "responseMimeType" not in generation_config
    assert "responseSchema" not in generation_config
    assert "responseJsonSchema" not in generation_config


def test_real_sdk_trusted_structured_request_does_not_retry_transient_http_error(
    monkeypatch,
):
    import httpx
    from google.genai.errors import ServerError
    from google.genai import types
    from core.integrations.gemini import client as gemini_client
    from core.integrations.gemini import llm
    from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
    from tests.fixtures.sdk_fakes import _REAL_GENAI_CLIENT

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-client-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"code": 503, "message": "synthetic unavailable"}})

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))

    def local_client(**kwargs):
        options = kwargs["http_options"]
        kwargs["http_options"] = types.HttpOptions(
            api_version=options.api_version,
            retry_options=options.retry_options,
            httpx_client=http_client,
        )
        return _REAL_GENAI_CLIENT(**kwargs)

    monkeypatch.setattr(gemini_client.genai, "Client", local_client)
    try:
        with pytest.raises(ServerError) as exc_info:
            llm.call_gemini_json("classify", retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
    finally:
        http_client.close()

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/gemini-3.5-flash-lite:generateContent"
    assert exc_info.value.code == 503
