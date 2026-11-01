from __future__ import annotations

import importlib
import json
from urllib.parse import parse_qs

import google_auth_httplib2
import google.auth.transport.requests
import pytest
from google.auth.credentials import AnonymousCredentials
from google.oauth2.credentials import Credentials

from core.integrations.google_drive import client as drive_client
from core.integrations.google_auth.credentials import credentials
from core.integrations.google_docs import reader as docs_reader


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


def test_oauth_precedes_adc_and_refreshes_without_user_credential_scopes(monkeypatch, sdk_fakes):
    _set_oauth(monkeypatch)
    scopes = ["https://www.googleapis.com/auth/documents.readonly"]

    result = credentials(scopes, allow_adc_fallback=True)

    assert isinstance(result, Credentials)
    assert result.refresh_token == "synthetic-refresh-token"
    assert result.client_id == "synthetic-client-id"
    assert result.client_secret == "synthetic-client-secret"
    assert result.scopes is None
    assert result.token == "pytest-synthetic-access-token"
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("auth", "credentials.refresh")
    ]


@pytest.mark.parametrize(
    "scopes",
    [
        ["https://www.googleapis.com/auth/drive.readonly"],
        ["https://www.googleapis.com/auth/documents.readonly"],
        [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
        ],
    ],
    ids=["drive", "docs", "import-combined"],
)
def test_real_oauth_refresh_sends_unscoped_refresh_request(
    monkeypatch, sdk_fakes, scopes
):
    _set_oauth(monkeypatch)
    from tests.fixtures.sdk_fakes import install

    original_refresh = next(
        original
        for obj, name, original, _replacement in install().values()
        if obj is Credentials and name == "refresh"
    )
    monkeypatch.setattr(Credentials, "refresh", original_refresh)

    requests = []

    class Response:
        def __init__(self, data, status):
            self.data = data
            self.status = status
            self.headers = {"content-type": "application/json"}

    class FakeRequest:
        def __call__(self, method, url, body=None, headers=None, **kwargs):
            del headers, kwargs
            form = parse_qs(body.decode("ascii"))
            requests.append((method, url, form))
            if "scope" in form:
                return Response(json.dumps({"error": "invalid_scope"}).encode(), 400)
            return Response(
                json.dumps(
                    {"access_token": "synthetic-refreshed-access-token", "expires_in": 3600}
                ).encode(),
                200,
            )

    monkeypatch.setattr(
        google.auth.transport.requests, "Request", lambda: FakeRequest()
    )

    result = credentials(scopes, allow_adc_fallback=True)

    assert result.token == "synthetic-refreshed-access-token"
    assert len(requests) == 1
    method, url, form = requests[0]
    assert method == "POST"
    assert url == "https://oauth2.googleapis.com/token"
    assert form == {
        "refresh_token": ["synthetic-refresh-token"],
        "client_id": ["synthetic-client-id"],
        "client_secret": ["synthetic-client-secret"],
        "grant_type": ["refresh_token"],
    }


def test_single_attempt_oauth_refresh_uses_real_sdk_with_one_synthetic_http_failure(
    monkeypatch, sdk_fakes,
):
    """Exercise Credentials.refresh through requests without endpoint retries."""
    import requests
    from google.auth.exceptions import RefreshError
    from requests.adapters import HTTPAdapter
    from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
    from tests.fixtures.sdk_fakes import install

    _set_oauth(monkeypatch)
    original_refresh = next(
        original
        for obj, name, original, _replacement in install().values()
        if obj is Credentials and name == "refresh"
    )
    monkeypatch.setattr(Credentials, "refresh", original_refresh)
    send_calls = []

    class SyntheticSession:
        def __init__(self):
            self.adapters = {}

        def mount(self, prefix, adapter):
            self.adapters[prefix] = adapter

        def request(self, method, url, **_kwargs):
            send_calls.append((method, url))
            response = requests.Response()
            response.status_code = 503
            response.url = url
            response.headers["content-type"] = "application/json"
            response._content = b'{"error":"temporarily_unavailable"}'
            return response

        def close(self):
            pass

    synthetic_session = SyntheticSession()
    monkeypatch.setattr(requests, "Session", lambda: synthetic_session)

    def no_transport(_adapter, _request, **_kwargs):
        pytest.fail("synthetic session should not dispatch a live HTTP transport")

    monkeypatch.setattr(HTTPAdapter, "send", no_transport)
    with pytest.raises(RefreshError):
        credentials(_DRIVE_SCOPES, retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)

    assert len(send_calls) == 1
    assert send_calls[0][0] == "POST"
    assert send_calls[0][1] == "https://oauth2.googleapis.com/token"
    assert synthetic_session.adapters["https://"].max_retries.total == 0
    assert not any("synthetic" in str(event) for event in sdk_fakes.trace)


@pytest.mark.parametrize("mode", ["unauthorized", "protocol"])
def test_authorized_single_attempt_transport_dispatches_once(monkeypatch, mode):
    import requests
    from google_auth_httplib2 import AuthorizedHttp
    from core.integrations.google_auth.transport import SingleAttemptHttp

    class FakeCredentials:
        refreshes = 0

        def before_request(self, request, method, url, headers):
            headers["authorization"] = "Bearer synthetic"

        def refresh(self, request):
            self.refreshes += 1

    class SyntheticSession:
        def __init__(self):
            self.calls = []

        def request(self, **kwargs):
            self.calls.append(kwargs)
            if mode == "protocol":
                raise requests.ConnectionError("synthetic protocol failure")
            response = requests.Response()
            response.status_code = 401
            response.reason = "unauthorized"
            response.url = kwargs["url"]
            response._content = b""
            return response

        def close(self):
            pass

    session = SyntheticSession()
    http = SingleAttemptHttp(timeout=60)
    http._session.close()
    http._session = session
    credentials = FakeCredentials()
    authorized = AuthorizedHttp(credentials, http=http, max_refresh_attempts=0)
    if mode == "protocol":
        with pytest.raises(requests.ConnectionError):
            authorized.request("https://synthetic.invalid/rpc")
    else:
        response, _body = authorized.request("https://synthetic.invalid/rpc")
        assert response.status == 401
    assert len(session.calls) == 1
    assert credentials.refreshes == 0


def test_single_attempt_docs_client_attributes_real_oauth_failure_without_response_text(
    monkeypatch, sdk_fakes, firestore_fake, fake_clock, caplog,
):
    import requests
    from google.auth.exceptions import RefreshError
    from requests.adapters import HTTPAdapter
    from core.integrations.google_docs.client import build_docs_service
    from core.utils.retry import ProviderBoundaryError, SINGLE_ATTEMPT_RETRY_POLICY
    from core.integrations.firestore.operations import create_operation, get_stage
    from core.operations.models import OperationOrigin
    from core.operations.retries import RetryGovernor
    from tests.fixtures.sdk_fakes import install
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    _set_oauth(monkeypatch)
    original_refresh = next(
        original
        for obj, name, original, _replacement in install().values()
        if obj is Credentials and name == "refresh"
    )
    monkeypatch.setattr(Credentials, "refresh", original_refresh)
    secret = "synthetic-oauth-response-secret"
    calls = []

    class SyntheticSession:
        def __init__(self):
            self.adapters = {}

        def mount(self, prefix, adapter):
            self.adapters[prefix] = adapter

        def request(self, method, url, **_kwargs):
            calls.append((method, url))
            response = requests.Response()
            response.status_code = 503
            response.url = url
            response.headers["content-type"] = "application/json"
            response._content = (f'{{"error":"temporarily_unavailable","detail":"{secret}"}}'
                                 .encode())
            return response

        def close(self):
            pass

    session = SyntheticSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(HTTPAdapter, "send", lambda *_args, **_kwargs: pytest.fail(
        "synthetic OAuth session must not open an HTTP transport",
    ))
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    create_operation(
        client, operation_id="oauth-boundary-stage", origin=OperationOrigin.TELEGRAM,
        caller_binding="synthetic-telegram", immutable_input_digest="sha256:oauth",
        immutable_input_ref="oauth-input", now=fake_clock.now(),
    )
    governor = RetryGovernor(
        client, clock=fake_clock, error_classifier=telegram_error_classifier,
    )
    with pytest.raises(ProviderBoundaryError) as raised:
        governor.run_stage(
            "oauth-boundary-stage", "telegram.finalize.docs",
            lambda: build_docs_service(
                ["https://www.googleapis.com/auth/documents"],
                retry_policy=SINGLE_ATTEMPT_RETRY_POLICY,
            ),
            now=fake_clock.now(),
        )
    assert isinstance(raised.value.original_error, RefreshError)
    assert len(calls) == 1 and calls[0][0] == "POST"
    assert session.adapters["https://"].max_retries.total == 0
    assert telegram_error_classifier(raised.value, "telegram.classification") == (
        "execution", "google_docs",
    )
    assert secret not in str(raised.value)
    stored = get_stage(client, "oauth-boundary-stage", "telegram.finalize.docs")
    assert stored is not None
    assert stored.failures == 1
    assert stored.last_error.service == "google_docs"
    assert secret not in repr(stored) and secret not in caplog.text
    assert not any(secret in str(event) for event in sdk_fakes.trace)


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
