"""Shared, offline-safe pytest setup for the service repository."""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from typing import Any

import pytest
from google.cloud import firestore
from google.cloud import firestore_v1
from google.cloud.firestore_v1 import client as firestore_client_module

from tests.fixtures.firestore_fake import FakeFirestoreClient, firestore_fake as _firestore_fake


_SYNTHETIC_ENV = {
    "GOOGLE_APPLICATION_CREDENTIALS": "C:/pytest/synthetic-service-account.json",
    "GOOGLE_CLOUD_PROJECT": "pytest-synthetic-project",
    "GCP_PROJECT": "pytest-synthetic-project",
    "GOOGLE_API_KEY": "pytest-synthetic-google-key",
    "GEMINI_API_KEY": "pytest-synthetic-gemini-key",
    "QDRANT_URL": "http://127.0.0.1:6333",
    "QDRANT_API_KEY": "pytest-synthetic-qdrant-key",
    "TLDV_API_KEY": "pytest-synthetic-tldv-key",
    "TL_DV_API_KEY": "pytest-synthetic-tldv-key",
}
_ENV_PREFIXES = ("GOOGLE_", "GCP_", "GEMINI_", "QDRANT_", "TLDV_", "TL_DV_")
_saved_environment: dict[str, str] | None = None
_saved_socket_methods: dict[str, Any] | None = None
_saved_firestore_constructors: dict[str, Any] | None = None


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
        if name.startswith(_ENV_PREFIXES):
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


def _restore_test_boundaries() -> None:
    global _saved_environment, _saved_socket_methods, _saved_firestore_constructors

    if _saved_environment is not None:
        for name in tuple(os.environ):
            if name.startswith(_ENV_PREFIXES):
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


def pytest_configure(config: pytest.Config) -> None:
    """Protect collection imports and test execution before pytest collects tests."""
    global _saved_environment, _saved_socket_methods, _saved_firestore_constructors

    if (
        _saved_environment is not None
        or _saved_socket_methods is not None
        or _saved_firestore_constructors is not None
    ):
        _restore_test_boundaries()
    _saved_environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(_ENV_PREFIXES)
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
        if name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in _SYNTHETIC_ENV.items():
        monkeypatch.setenv(name, value)
    return dict(_SYNTHETIC_ENV)


@pytest.fixture(autouse=True)
def _offline_test_boundaries(synthetic_environment: dict[str, str]) -> None:
    """Reset synthetic environment for every test; socket guard spans collection."""
    del synthetic_environment
    _firestore_fake.reset()


@pytest.fixture
def firestore_fake() -> Any:
    """Return isolated per-test Firestore fake state and its control API."""
    _firestore_fake.reset()
    return _firestore_fake
