"""Shared, offline-safe pytest setup for the service repository."""

from __future__ import annotations

import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from google.cloud import firestore
from google.cloud import firestore_v1
from google.cloud.firestore_v1 import client as firestore_client_module

from tests.fixtures.firestore_fake import FakeFirestoreClient, firestore_fake as _firestore_fake
from tests.fixtures.sdk_fakes import install as _install_sdk_fakes, restore as _restore_sdk_fakes, sdk_fakes as _sdk_fakes


_SYNTHETIC_ENV = {
    "GOOGLE_CLOUD_PROJECT": "pytest-synthetic-project",
    "GCP_PROJECT": "pytest-synthetic-project",
    "GOOGLE_API_KEY": "pytest-synthetic-google-key",
    "GEMINI_API_KEY": "pytest-synthetic-gemini-key",
    "QDRANT_URL": "http://127.0.0.1:6333",
    "QDRANT_API_KEY": "pytest-synthetic-qdrant-key",
    "QDRANT_COLLECTION": "tldv_vectors",
    "ROOT_FOLDER_URL": "https://drive.google.com/drive/folders/pytest-synthetic-root",
    "REGION": "pytest-synthetic-region",
    "TLDV_API_KEY": "pytest-synthetic-tldv-key",
    "TL_DV_API_KEY": "pytest-synthetic-tldv-key",
    "TZ": "UTC",
    "VECTOR_SYNC_URL": "https://vector-sync.pytest.invalid",
    "VECTOR_SYNC_QUEUE": "pytest-vector-sync-queue",
    "TLDV_IMPORT_SERVICE_URL": "https://tldv-import.pytest.invalid",
    "TLDV_IMPORT_QUEUE": "pytest-tldv-import-queue",
    "SYNC_CHECKER_URL": "https://sync-checker.pytest.invalid",
    "OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
    "OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
    "TASK_OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
    "TASK_OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
    "CLOUD_TASKS_OIDC_AUDIENCE": "https://task-worker.pytest.invalid",
    "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
    "INTERNAL_DIALOG_OPERATIONS_URL": "https://dialog-operations.pytest.invalid/internal/dialog-operations",
    "DIALOG_OPERATIONS_QUEUE": "pytest-dialog-operations-queue",
    "TELEGRAM_CONTROL_QUEUE": "pytest-telegram-control-queue",
    "CLOUD_TASKS_QUEUE": "pytest-cloud-tasks-queue",
    "TASK_SERVICE_ACCOUNT_EMAIL": "pytest-task-worker@synthetic-project.invalid",
}
_ENV_PREFIXES = (
    "GOOGLE_", "GCP_", "GEMINI_", "QDRANT_", "TLDV_", "TL_DV_", "TELEGRAM_",
    "OIDC_", "INTERNAL_", "QUEUE_", "TASK_", "CLOUD_TASKS_",
)
_ENV_SUFFIXES = (
    "_URL", "_URI", "_AUDIENCE", "_SERVICE_ACCOUNT", "_SERVICE_ACCOUNT_EMAIL",
    "_QUEUE", "_QUEUE_NAME",
)
_EXPLICIT_ENV_KEYS = {
    "ROOT_FOLDER_URL", "REGION", "TZ", "VECTOR_SYNC_URL", "VECTOR_SYNC_QUEUE",
    "TLDV_IMPORT_SERVICE_URL", "TLDV_IMPORT_QUEUE", "TLDV_RECONCILIATION_URL",
    "SYNC_CHECKER_URL",
}
_saved_environment: dict[str, str] | None = None
_saved_socket_methods: dict[str, Any] | None = None
_saved_firestore_constructors: dict[str, Any] | None = None
_saved_sdk_boundaries: dict[str, Any] | None = None


def _is_isolated_environment_key(name: str) -> bool:
    """Match inherited settings that must be synthetic or absent in tests."""
    return (
        name.startswith(_ENV_PREFIXES)
        or name in _EXPLICIT_ENV_KEYS
        or name.endswith(_ENV_SUFFIXES)
    )


def _is_loopback(address: object) -> bool:
    """Allow local socket traffic used by test servers and Windows asyncio."""
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if isinstance(host, str) and host.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def _apply_synthetic_environment() -> None:
    for name in tuple(os.environ):
        if _is_isolated_environment_key(name):
            os.environ.pop(name, None)
    os.environ.update(_SYNTHETIC_ENV)


def _guard_python_sockets() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def guarded_connect(sock: socket.socket, address: object) -> object:
        if not _is_loopback(address):
            raise RuntimeError(f"pytest offline guard blocked socket connection to {address!r}")
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not _is_loopback(address):
            raise RuntimeError(f"pytest offline guard blocked socket connection to {address!r}")
        return original_connect_ex(sock, address)

    def guarded_sendto(
        sock: socket.socket, data: bytes, *args: object, **kwargs: object
    ) -> int:
        address = kwargs.get("address", args[-1] if args else None)
        if not _is_loopback(address):
            raise RuntimeError(f"pytest offline guard blocked socket send to {address!r}")
        return original_sendto(sock, data, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.socket.sendto = guarded_sendto


class _FakeClock:
    """A deterministic UTC clock local to one test."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("fake clock start must be timezone-aware")
        self._current = start.astimezone(timezone.utc)

    @property
    def current_time(self) -> datetime:
        return self._current

    def now(self) -> datetime:
        return self._current

    def __call__(self) -> datetime:
        return self._current

    def advance(self, *, seconds: float = 0) -> datetime:
        self._current += timedelta(seconds=seconds)
        return self._current


def _restore_test_boundaries() -> None:
    global _saved_environment, _saved_socket_methods, _saved_firestore_constructors, _saved_sdk_boundaries

    if _saved_environment is not None:
        for name in tuple(os.environ):
            if _is_isolated_environment_key(name):
                os.environ.pop(name, None)
        os.environ.update(_saved_environment)
        _saved_environment = None

    if _saved_socket_methods is not None:
        for name, method in _saved_socket_methods.items():
            setattr(socket.socket, name, method)
        _saved_socket_methods = None

    if _saved_firestore_constructors is not None:
        firestore.Client = _saved_firestore_constructors["firestore.Client"]
        firestore_v1.Client = _saved_firestore_constructors["firestore_v1.Client"]
        firestore_client_module.Client = _saved_firestore_constructors["client.Client"]
        _saved_firestore_constructors = None

    if _saved_sdk_boundaries is not None:
        _restore_sdk_fakes(_saved_sdk_boundaries)
        _saved_sdk_boundaries = None


def pytest_configure(config: pytest.Config) -> None:
    """Protect collection imports and test execution before pytest collects tests."""
    global _saved_environment, _saved_socket_methods, _saved_firestore_constructors, _saved_sdk_boundaries

    if (
        _saved_environment is not None
        or _saved_socket_methods is not None
        or _saved_firestore_constructors is not None
        or _saved_sdk_boundaries is not None
    ):
        _restore_test_boundaries()
    _saved_environment = {
        name: value
        for name, value in os.environ.items()
        if _is_isolated_environment_key(name)
    }
    _saved_socket_methods = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "sendto": socket.socket.sendto,
    }
    _saved_firestore_constructors = {
        "firestore.Client": firestore.Client,
        "firestore_v1.Client": firestore_v1.Client,
        "client.Client": firestore_client_module.Client,
    }
    try:
        _saved_sdk_boundaries = _install_sdk_fakes()
        _sdk_fakes.begin_run()
        _apply_synthetic_environment()
        _guard_python_sockets()
        firestore.Client = FakeFirestoreClient
        firestore_v1.Client = FakeFirestoreClient
        firestore_client_module.Client = FakeFirestoreClient
    except BaseException:
        _restore_test_boundaries()
        raise


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore caller state even when collection or test execution fails."""
    _restore_test_boundaries()


@pytest.fixture(scope="session")
def app_root() -> Path:
    """Return the Python application root (the parent of tests/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Reset synthetic cloud/service settings for each test."""
    for name in tuple(os.environ):
        if _is_isolated_environment_key(name):
            monkeypatch.delenv(name, raising=False)
    for name, value in _SYNTHETIC_ENV.items():
        monkeypatch.setenv(name, value)
    return dict(_SYNTHETIC_ENV)


@pytest.fixture(autouse=True)
def _offline_test_boundaries(synthetic_environment: dict[str, str]) -> None:
    """Reset synthetic environment for every test; socket guard spans collection."""
    del synthetic_environment
    _firestore_fake.reset()
    _sdk_fakes.capture_bootstrap()
    _sdk_fakes.reset()


@pytest.fixture
def firestore_fake() -> Any:
    """Return isolated per-test Firestore fake state and its control API."""
    _firestore_fake.reset()
    return _firestore_fake


@pytest.fixture
def sdk_fakes() -> Any:
    """Return per-test programmable external SDK boundary fakes and trace."""
    _sdk_fakes.reset()
    return _sdk_fakes


@pytest.fixture
def fake_clock() -> _FakeClock:
    """Return an isolated clock starting at a fixed UTC instant."""
    return _FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
