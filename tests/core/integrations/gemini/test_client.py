from importlib import reload

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

    assert calls[0] == {"api_key": "synthetic-client-test-key"}
    assert calls[1]["api_key"] == calls[0]["api_key"]
    assert calls[1]["http_options"] == types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)
    )
