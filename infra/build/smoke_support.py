"""Offline startup and benign-route checks for assembled service artifacts.

This module is copied beside deployment validation tools and deliberately has
no dependency on the source checkout.  The artifact's assembly report is the
only source of the runtime target.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_LOADED_ARTIFACTS: set[Path] = set()
_SERVICE_KINDS = {
    "mcp": "asgi",
    "vector_sync": "asgi",
    "tldv_import": "asgi",
    "sync_checker": "function",
    "tldv_webhook": "function",
    "tldv_reconciliation": "function",
}


class _EmptyFirestore:
    """Small Firestore-shaped response for harmless empty scans."""

    def collection(self, _name: str) -> "_EmptyFirestore":
        return self

    def document(self, _name: str) -> "_EmptyFirestore":
        return self

    def where(self, *_args: Any, **_kwargs: Any) -> "_EmptyFirestore":
        return self

    def select(self, *_args: Any, **_kwargs: Any) -> "_EmptyFirestore":
        return self

    def stream(self) -> list[Any]:
        return []

    def get(self) -> Any:
        return SimpleNamespace(exists=False, to_dict=lambda: None)


class _SmokeRequest:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def get_json(self, silent: bool = False) -> dict[str, Any]:
        return self._payload


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _artifact_import_paths(artifact: Path) -> list[str]:
    """Keep the artifact and interpreter-provided modules on the import path."""
    allowed_roots = {
        Path(value).resolve()
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (value := sysconfig.get_path(key))
    }
    allowed_roots.update(
        Path(value).resolve()
        for value in (sys.prefix, sys.base_prefix)
        if value
    )
    paths = [str(artifact)]
    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry).resolve()
        except (OSError, RuntimeError):
            continue
        if any(_inside(candidate, root) for root in allowed_roots):
            value = str(candidate)
            if value not in paths:
                paths.append(value)
    return paths


def _evict_previous_artifacts() -> None:
    if not _LOADED_ARTIFACTS:
        return
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            resolved = Path(filename).resolve()
        except (OSError, RuntimeError):
            continue
        if any(_inside(resolved, root) for root in _LOADED_ARTIFACTS):
            sys.modules.pop(name, None)


def _install_boundary_fakes(service: str) -> None:
    """Patch SDK boundaries while retaining the installed SDK modules."""
    if service == "vector_sync":
        qdrant_client = importlib.import_module("qdrant_client")

        class OfflineQdrantClient:
            __tlbrain_external_fake__ = True

            def __init__(self, *_args: Any, **_kwargs: Any):
                pass

            def get_collections(self) -> Any:
                return SimpleNamespace(collections=[])

            def create_collection(self, **_kwargs: Any) -> None:
                return None

            def get_collection(self, *_args: Any, **_kwargs: Any) -> Any:
                return SimpleNamespace(payload_schema={})

            def create_payload_index(self, **_kwargs: Any) -> None:
                return None

        _replace_boundary(qdrant_client, "QdrantClient", OfflineQdrantClient)

    if service in {"sync_checker", "tldv_reconciliation"}:
        firestore = importlib.import_module("google.cloud.firestore")
        _replace_boundary(
            firestore, "Client", lambda *_args, **_kwargs: _EmptyFirestore()
        )

        tasks_v2 = importlib.import_module("google.cloud.tasks_v2")

        class OfflineCloudTasksClient:
            def queue_path(self, project: str, region: str, queue: str) -> str:
                return f"projects/{project}/locations/{region}/queues/{queue}"

            def task_path(self, project: str, region: str, queue: str, task: str) -> str:
                return f"projects/{project}/locations/{region}/queues/{queue}/tasks/{task}"

            def create_task(self, **_kwargs: Any) -> dict[str, Any]:
                return {}

        _replace_boundary(tasks_v2, "CloudTasksClient", OfflineCloudTasksClient)
        google_auth = importlib.import_module("google.auth")
        _replace_boundary(
            google_auth, "default", lambda **_kwargs: (None, "smoke-project")
        )

    if service == "sync_checker":
        discovery = importlib.import_module("googleapiclient.discovery")

        class _EmptyDriveRequest:
            def execute(self) -> dict[str, Any]:
                return {"files": [], "changes": [], "startPageToken": "smoke-token"}

        class _EmptyDriveResource:
            def list(self, **_kwargs: Any) -> _EmptyDriveRequest:
                return _EmptyDriveRequest()

            def getStartPageToken(self, **_kwargs: Any) -> _EmptyDriveRequest:
                return _EmptyDriveRequest()

        class _EmptyDrive:
            def files(self) -> _EmptyDriveResource:
                return _EmptyDriveResource()

            def changes(self) -> _EmptyDriveResource:
                return _EmptyDriveResource()

        _replace_boundary(
            discovery, "build", lambda *_args, **_kwargs: _EmptyDrive()
        )

    if service == "tldv_reconciliation":
        requests = importlib.import_module("requests")

        class _EmptyResponse:
            status_code = 200
            content = b"{}"

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {}

        _replace_boundary(requests, "get", lambda *_args, **_kwargs: _EmptyResponse())


def _replace_boundary(module: Any, name: str, replacement: Any) -> None:
    original = getattr(module, name, None)
    if not callable(original):
        module_name = getattr(module, "__name__", type(module).__name__)
        raise RuntimeError(
            f"Required SDK boundary {module_name}.{name} is missing or not callable"
        )
    setattr(module, name, replacement)


def _configure_environment() -> dict[str, str | None]:
    values = {
        "ROOT_FOLDER_URL": "https://drive.google.com/drive/folders/smoke-root",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "QDRANT_API_KEY": "offline-smoke-key",
        "QDRANT_COLLECTION": "offline-smoke-collection",
        "VECTOR_SYNC_URL": "http://127.0.0.1:8080",
        "VECTOR_SYNC_QUEUE": "offline-smoke-queue",
        "REGION": "us-central1",
        "GOOGLE_CLOUD_PROJECT": "smoke-project",
        "TLDV_API_KEY": "offline-smoke-key",
    }
    remove = (
        "ALLOWED_EMAIL",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "TLDV_IMPORT_SERVICE_URL",
    )
    prior = {key: os.environ.get(key) for key in (*values, *remove)}
    os.environ.update(values)
    for key in remove:
        os.environ.pop(key, None)
    return prior


def _restore_environment(prior: dict[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def _exercise_asgi(app: Any) -> None:
    messages: list[dict[str, Any]] = []
    request_pending = True

    async def receive() -> dict[str, Any]:
        nonlocal request_pending
        if request_pending:
            request_pending = False
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"smoke.test")],
        "client": ("127.0.0.1", 12345),
        "server": ("smoke.test", 80),
    }
    _run_coroutine(app(scope, receive, send))
    response_start = next(
        (message for message in messages if message.get("type") == "http.response.start"),
        None,
    )
    if response_start is None or response_start.get("status") != 200:
        status = response_start.get("status") if response_start is not None else "missing"
        raise RuntimeError(f"ASGI health route returned HTTP {status}")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message.get("type") == "http.response.body"
    )
    try:
        payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ASGI health route did not return JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"ASGI health route returned unexpected payload: {payload!r}")


def _run_coroutine(coroutine: Any) -> Any:
    """Drive the smoke coroutine without opening a local event-loop socket."""
    value: Any = None
    while True:
        try:
            yielded = coroutine.send(value)
        except StopIteration as finished:
            return finished.value
        value = None
        if yielded is None:
            continue
        if getattr(yielded, "done", lambda: False)():
            value = yielded.result()
            continue
        raise RuntimeError(f"ASGI smoke needs unsupported async scheduling: {yielded!r}")


def _exercise_health_function(health: Any) -> None:
    result = health()
    if asyncio.iscoroutine(result):
        result = _run_coroutine(result)
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise RuntimeError(f"Health function returned unexpected payload: {result!r}")


def run_smoke(service: str, artifact: str | Path) -> None:
    """Import a service from its assembled artifact and run its benign route."""
    artifact_path = Path(artifact).resolve()
    if service not in {
        "mcp",
        "vector_sync",
        "tldv_import",
        "sync_checker",
        "tldv_webhook",
        "tldv_reconciliation",
    }:
        raise ValueError(f"Unsupported smoke service: {service}")
    report_path = artifact_path / "assembly-report.json"
    if not artifact_path.is_dir() or not report_path.is_file():
        raise FileNotFoundError(f"Artifact or assembly report is missing: {artifact_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("service") != service:
        raise ValueError(f"Artifact is for {report.get('service')!r}, not {service!r}")
    runtime = report.get("runtime")
    if isinstance(runtime, dict):
        # Retain support for component fixtures that predate the assembled
        # report shape.
        kind = runtime.get("kind")
        target = runtime.get("target")
    else:
        # Production assembly reports expose only the target. Runtime kind is
        # a stable property of the six supported service entrypoints.
        kind = _SERVICE_KINDS.get(service)
        target = report.get("runtime_target")
    if kind not in {"asgi", "function"} or not isinstance(target, str):
        raise ValueError("Assembly report has no supported runtime target")
    if ":" in target:
        module_name, symbol_name = target.split(":", 1)
    elif kind == "function":
        module_name, symbol_name = "main", target
    else:
        raise ValueError("ASGI assembly target must use module:symbol form")

    prior_env = _configure_environment()
    prior_path = sys.path[:]
    _evict_previous_artifacts()
    _LOADED_ARTIFACTS.add(artifact_path)
    sys.path[:] = _artifact_import_paths(artifact_path)
    try:
        _install_boundary_fakes(service)
        module = importlib.import_module(module_name)
        handler = getattr(module, symbol_name, None)
        if kind == "asgi":
            if handler is None:
                # Minimal component artifacts expose the same benign health
                # callable without installing an ASGI framework application.
                _exercise_health_function(getattr(module, "health"))
            else:
                _run_coroutine(_exercise_asgi(handler))
        else:
            if handler is None:
                _exercise_health_function(getattr(module, "health"))
                return
            if service == "sync_checker":
                payload = {"doc_id": "offline-smoke-document"}
            elif service == "tldv_webhook":
                payload = {"data": {"meetingId": "offline-smoke-meeting"}}
            else:
                payload = {"full_scan": True}
            result = handler(_SmokeRequest(payload))
            if not isinstance(result, tuple) or len(result) < 2 or result[1] != 200:
                raise RuntimeError(f"Function smoke returned unexpected result: {result!r}")
    finally:
        sys.path[:] = prior_path
        _restore_environment(prior_env)

