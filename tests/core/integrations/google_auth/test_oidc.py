from __future__ import annotations

import pytest

from core.integrations.google_auth import oidc


EMAIL = "task-worker@example.test"


@pytest.fixture
def verifier_config(monkeypatch):
    monkeypatch.setenv("OIDC_AUDIENCE", "https://callback.example.test/internal/dialog-operations")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", EMAIL)


def test_verifies_with_exact_audience_and_requires_verified_expected_email(monkeypatch, verifier_config):
    calls = []
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: "request")

    def verify(token, request, *, audience):
        calls.append((token, request, audience))
        return {"email": EMAIL, "email_verified": True}

    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", verify)

    identity = oidc.verify_google_id_token("signed-token")

    assert calls == [("signed-token", "request", "https://callback.example.test/internal/dialog-operations")]
    assert identity == oidc.OIDCIdentity(EMAIL, True)


@pytest.mark.parametrize("error", [ValueError("bad signature"), ValueError("expired"), ValueError("wrong issuer")])
def test_rejects_signature_expiry_and_issuer_failures(monkeypatch, verifier_config, error):
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    assert oidc.verify_google_id_token("token") is None


def test_rejects_missing_token_and_unverified_email(monkeypatch, verifier_config):
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *_args, **_kwargs: {"email": EMAIL, "email_verified": 1})
    assert oidc.verify_google_id_token(None) is None
    assert oidc.verify_google_id_token("token") is None


def test_reports_a_valid_token_from_another_service_account_as_unauthorized(monkeypatch, verifier_config):
    monkeypatch.setattr("google.auth.transport.requests.Request", lambda: object())
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *_args, **_kwargs: {"email": "other@example.test", "email_verified": True})
    assert oidc.verify_google_id_token("token") == oidc.OIDCIdentity("other@example.test", False)


def test_missing_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("OIDC_AUDIENCE", raising=False)
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", EMAIL)
    with pytest.raises(oidc.OIDCConfigurationError):
        oidc.verify_google_id_token("token")
