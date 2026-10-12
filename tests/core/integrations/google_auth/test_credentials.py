from __future__ import annotations

import importlib

import google_auth_httplib2
import pytest
from google.auth.credentials import AnonymousCredentials
from google.oauth2.credentials import Credentials

from core.google_drive import docs_reader, drive_client
from core.integrations.google_auth.credentials import credentials


_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _set_oauth(monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")


def test_strict_oauth_raises_without_adc_and_keeps_error_and_logs_secret_free(
    monkeypatch, sdk_fakes, caplog
):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as error:
        credentials(_DRIVE_SCOPES)

    assert str(error.value) == (
        "GOOGLE_REFRESH_TOKEN is not set — Drive access requires user OAuth credentials"
    )
    assert sdk_fakes.trace == []
    assert "synthetic" not in str(error.value)
    assert "synthetic" not in caplog.text


def test_adc_fallback_is_opt_in_and_receives_exact_scopes(monkeypatch, sdk_fakes):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"]

    result = credentials(scopes, allow_adc_fallback=True)

    assert isinstance(result, AnonymousCredentials)
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("auth", "default")
    ]
    assert sdk_fakes.trace[0]["arguments"]["scopes"] == scopes


def test_oauth_precedes_adc_and_refreshes_with_the_supplied_scopes(monkeypatch, sdk_fakes):
    _set_oauth(monkeypatch)
    scopes = ["https://www.googleapis.com/auth/documents.readonly"]

    result = credentials(scopes, allow_adc_fallback=True)

    assert isinstance(result, Credentials)
    assert result.refresh_token == "synthetic-refresh-token"
    assert result.client_id == "synthetic-client-id"
    assert result.client_secret == "synthetic-client-secret"
    assert result.scopes == scopes
    assert result.token == "pytest-synthetic-access-token"
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("auth", "credentials.refresh")
    ]


def test_drive_and_docs_keep_their_transport_policies(monkeypatch, sdk_fakes):
    _set_oauth(monkeypatch)

    drive_client.build_drive_service()
    drive_build = next(entry for entry in sdk_fakes.trace if entry["operation"] == "build")
    assert isinstance(drive_build["arguments"]["http"], google_auth_httplib2.AuthorizedHttp)
    assert drive_build["arguments"]["http"].http.timeout == 60

    sdk_fakes.reset()
    docs_reader._build_docs_service()
    docs_build = next(entry for entry in sdk_fakes.trace if entry["operation"] == "build")
    assert docs_build["arguments"]["api"] == "docs"
    assert "credentials" in docs_build["arguments"]
    assert "http" not in docs_build["arguments"]


def test_import_enables_adc_and_keeps_default_discovery_transport(
    app_root, monkeypatch, sdk_fakes
):
    monkeypatch.syspath_prepend(str(app_root / "services" / "connectors" / "tldv"))
    import_main = importlib.import_module("services.connectors.tldv.import_service.app.main")
    assert sdk_fakes.trace == []
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)

    import_main._build_drive()
    import_main._build_docs()

    builds = [entry for entry in sdk_fakes.trace if entry["operation"] == "build"]
    assert [entry["arguments"]["api"] for entry in builds] == ["drive", "docs"]
    assert all("credentials" in entry["arguments"] for entry in builds)
    assert all("http" not in entry["arguments"] for entry in builds)
    default_calls = [entry for entry in sdk_fakes.trace if entry["operation"] == "default"]
    assert len(default_calls) == 2
    assert all(default_call["arguments"]["scopes"] == import_main._SCOPES for default_call in default_calls)
