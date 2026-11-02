"""Offline tests for deploy-only gcloud Firestore authentication."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from infra.deploy.connectors.telegram import webhook_admin


pytestmark = pytest.mark.offline


def test_deployment_client_uses_active_gcloud_token_and_explicit_project(
    monkeypatch,
):
    from google.cloud import firestore
    from google.oauth2.credentials import Credentials

    token = "ya29.synthetic-access-token"
    observed = {}
    monkeypatch.setattr(webhook_admin, "_find_gcloud", lambda: r"C:\Program Files\Google Cloud\gcloud.cmd")
    monkeypatch.setenv("PATH", r"C:\gcloud tools")
    monkeypatch.setenv("CLOUDSDK_CONFIG", r"C:\Users\tester\.config\gcloud")
    monkeypatch.setenv("CLOUDSDK_ACTIVE_CONFIG_NAME", "deploy")
    monkeypatch.setenv("CLOUDSDK_CORE_ACCOUNT", "operator@example.test")
    monkeypatch.setenv(
        "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT",
        "deployment@example.test",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-bot-token-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "11,22")
    monkeypatch.setenv("TLDV_API_KEY", "synthetic-tldv-secret")
    monkeypatch.setenv("TL_DV_API_KEY", "synthetic-tldv-alias-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-gemini-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-google-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-oauth-secret")
    original_environment = os.environ.copy()

    def run(args, **kwargs):
        observed.update(args=args, **kwargs)
        return subprocess.CompletedProcess(args, 0, stdout=token + "\n", stderr="")

    def forbid_adc(*_args, **_kwargs):
        pytest.fail("deployment auth must not consult ADC")

    monkeypatch.setattr(webhook_admin.subprocess, "run", run)
    monkeypatch.setattr("google.auth.default", forbid_adc)
    monkeypatch.setattr(
        firestore,
        "Client",
        lambda **kwargs: observed.setdefault("client", kwargs) or kwargs,
    )

    client = webhook_admin.create_deployment_firestore_client(project="deployment-project")

    assert client is observed["client"]
    assert observed["client"]["project"] == "deployment-project"
    assert isinstance(observed["client"]["credentials"], Credentials)
    assert observed["client"]["credentials"].token == token
    assert observed["args"] == [
        r"C:\Program Files\Google Cloud\gcloud.cmd",
        "auth", "print-access-token", "--quiet",
    ]
    assert observed["timeout"] == webhook_admin._GCLOUD_TIMEOUT_SECONDS
    assert observed["capture_output"] is True
    assert observed["shell"] is (os.name == "nt")
    assert observed["env"]["PATH"] == r"C:\gcloud tools"
    assert observed["env"]["CLOUDSDK_CONFIG"] == r"C:\Users\tester\.config\gcloud"
    assert observed["env"]["CLOUDSDK_ACTIVE_CONFIG_NAME"] == "deploy"
    assert observed["env"]["CLOUDSDK_CORE_ACCOUNT"] == "operator@example.test"
    assert observed["env"]["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] == "deployment@example.test"
    assert not (webhook_admin._DEPLOY_AUTH_EXCLUDED_ENV & observed["env"].keys())
    assert os.environ.copy() == original_environment
    assert token not in repr(observed["args"])


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess([], 1, stdout="", stderr="synthetic-secret"),
        subprocess.CompletedProcess([], 0, stdout="not a token", stderr="synthetic-secret"),
        subprocess.TimeoutExpired("gcloud", 20, output="synthetic-secret"),
        OSError("synthetic-secret"),
    ],
)
def test_cli_failure_is_redacted_and_never_constructs_firestore(monkeypatch, failure):
    monkeypatch.setattr(webhook_admin, "_find_gcloud", lambda: "gcloud")
    if isinstance(failure, subprocess.CompletedProcess):
        monkeypatch.setattr(webhook_admin.subprocess, "run", lambda *_args, **_kwargs: failure)
    else:
        def fail(*_args, **_kwargs):
            raise failure
        monkeypatch.setattr(webhook_admin.subprocess, "run", fail)
    monkeypatch.setattr(
        webhook_admin,
        "_gcloud_environment",
        lambda: {"PATH": "synthetic-path"},
    )
    from google.cloud import firestore

    monkeypatch.setattr(firestore, "Client", lambda **_kwargs: pytest.fail("Firestore called"))

    with pytest.raises(RuntimeError) as raised:
        webhook_admin.create_deployment_firestore_client(project="explicit-project")

    assert "synthetic-secret" not in str(raised.value)
    assert "gcloud authentication is unavailable" == str(raised.value)


def test_missing_cli_fails_before_firestore(monkeypatch):
    monkeypatch.setattr(webhook_admin, "_find_gcloud", lambda: None)

    with pytest.raises(RuntimeError, match="gcloud authentication is unavailable"):
        webhook_admin.create_deployment_firestore_client(project="explicit-project")


def test_finder_prefers_windows_cmd_and_uses_which(monkeypatch):
    monkeypatch.setattr(webhook_admin, "os", SimpleNamespace(name="nt"))
    calls = []

    def which(name):
        calls.append(name)
        return r"C:\Program Files\Google Cloud\gcloud.cmd" if name == "gcloud.cmd" else None

    monkeypatch.setattr(webhook_admin.shutil, "which", which)

    assert webhook_admin._find_gcloud() == r"C:\Program Files\Google Cloud\gcloud.cmd"
    assert calls == ["gcloud.cmd"]


def test_empty_or_malformed_cli_token_fails(monkeypatch):
    monkeypatch.setattr(webhook_admin, "_find_gcloud", lambda: "gcloud")
    monkeypatch.setattr(
        webhook_admin.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="\n", stderr=""),
    )

    with pytest.raises(RuntimeError, match="gcloud authentication is unavailable"):
        webhook_admin.create_deployment_firestore_client(project="explicit-project")
