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
