"""Programmable fakes at external SDK boundaries used by offline tests.

SDK packages and their models stay real. Only constructors and outbound operations
are replaced; unexpected response-bearing calls fail loudly until scripted.
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import google.auth
from google.auth import _helpers as google_auth_helpers
import google.auth._default as google_auth_default
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_httplib2
import grpc
import httpx
import httplib2
import requests
from fastembed import SparseTextEmbedding as RealSparseTextEmbedding
from google.genai import client as genai_client_module
from googleapiclient import discovery as _discovery_module
from googleapiclient.discovery import build as _REAL_DISCOVERY_BUILD
from google.api_core.exceptions import AlreadyExists
from google.cloud import tasks_v2
from google.cloud.tasks_v2.services import cloud_tasks as _tasks_cloud_module
from google.cloud.tasks_v2.services.cloud_tasks import client as tasks_client_module
from google.cloud.tasks_v2 import CloudTasksClient as RealCloudTasksClient
from google.oauth2.credentials import Credentials
from qdrant_client import qdrant_client as qdrant_client_module
from fastembed.sparse import sparse_text_embedding as sparse_embedding_module


_REAL_AUTH_DEFAULT = google_auth_default.default
_REAL_GENAI_CLIENT = genai_client_module.Client
_REAL_QDRANT_CLIENT = qdrant_client_module.QdrantClient
_REAL_TASKS_CLIENT = tasks_client_module.CloudTasksClient
_REAL_SPARSE_EMBEDDING = sparse_embedding_module.SparseTextEmbedding

_MISSING = object()


def _copy(value: Any) -> Any:
    if isinstance(value, (BaseException, type)) or callable(value):
        return value
    if isinstance(value, (httpx.BaseTransport, httpx.AsyncBaseTransport, httplib2.Http, google_auth_httplib2.AuthorizedHttp)):
        return value
    if isinstance(value, dict):
        return {_copy(key): _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, set):
        return {_copy(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise TypeError(f"Cannot safely copy SDK fake value of type {type(value).__name__}") from exc


@dataclass
class _State:
    trace: list[dict[str, Any]] = field(default_factory=list)
    queues: dict[tuple[str, str], deque[Any]] = field(default_factory=lambda: defaultdict(deque))
    sparse: dict[str, tuple[list[int], list[float]]] = field(default_factory=dict)
    tasks: set[str] = field(default_factory=set)
    anonymous_task_counter: int = 0
    qdrant_collections: dict[str, dict[str, Any]] = field(default_factory=dict)
    docs_revisions: dict[str, str] = field(default_factory=dict)
    docs_revision_counter: int = 0
    clients: list[Any] = field(default_factory=list)


_STATE = _State()


def _record(boundary: str, operation: str, **arguments: Any) -> None:
    _STATE.trace.append({"boundary": boundary, "operation": operation, "arguments": _copy(arguments)})


def _script(boundary: str, operation: str, default: Any = _MISSING) -> Any:
    q = _STATE.queues[(boundary, operation)]
    if q:
        value = q.popleft()
        if isinstance(value, BaseException):
            raise value
        return _copy(value)
    if default is _MISSING:
        raise RuntimeError(f"No scripted {boundary}.{operation} response")
    return _copy(default)


class _Controls:
    def reset(self) -> None:
        _STATE.trace.clear()
        _STATE.queues.clear()
        _STATE.sparse.clear()
        _STATE.tasks.clear()
        _STATE.anonymous_task_counter = 0
        _STATE.qdrant_collections.clear()
        _STATE.docs_revisions.clear()
        _STATE.docs_revision_counter = 0
        _STATE.clients.clear()

    @property
    def trace(self) -> list[dict[str, Any]]:
        return _copy(_STATE.trace)

    def queue(self, boundary: str, operation: str, *, response: Any = _MISSING, error: BaseException | None = None) -> None:
        if (response is _MISSING) == (error is None):
            raise ValueError("provide exactly one of response or error")
        _STATE.queues[(boundary, operation)].append(error if error is not None else _copy(response))

    def set_sparse_vector(self, text: str, indices: list[int], values: list[float]) -> None:
        _STATE.sparse[text] = (list(indices), list(values))


class SDKFakes(_Controls):
    def __init__(self) -> None:
        self._bootstrap_trace: list[dict[str, Any]] = []
        self._captured_bootstrap = False
        self.drive = _Boundary(self, "drive")
        self.docs = _Boundary(self, "docs")
        self.gemini = _Boundary(self, "gemini")
        self.qdrant = _Boundary(self, "qdrant")
        self.tasks = _Boundary(self, "tasks")
        self.sparse = _Boundary(self, "sparse")

    def begin_run(self) -> None:
        super().reset()
        self._bootstrap_trace.clear()
        self._captured_bootstrap = False

    def capture_bootstrap(self) -> None:
        if not self._captured_bootstrap:
            self._bootstrap_trace = [event for event in _STATE.trace if event["boundary"] == "qdrant"]
            self._captured_bootstrap = True

    @property
    def trace(self) -> list[dict[str, Any]]:
        return _copy(_STATE.trace)

    @property
    def bootstrap_trace(self) -> list[dict[str, Any]]:
        return _copy(self._bootstrap_trace)

    def reset(self) -> None:
        super().reset()


class _Boundary:
    def __init__(self, api: SDKFakes, name: str) -> None:
        self.api, self.name = api, name

    def queue(self, operation: str, *, response: Any = _MISSING, error: BaseException | None = None) -> None:
        self.api.queue(self.name, operation, response=response, error=error)

    def set_vectors(self, text: str, indices: list[int], values: list[float]) -> None:
        self.api.set_sparse_vector(text, indices, values)


sdk_fakes = SDKFakes()


class _Request:
    def __init__(self, boundary: str, operation: str, args: dict[str, Any]) -> None:
        self.boundary, self.operation, self.args = boundary, operation, _copy(args)

    def execute(self, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(f"Unsupported execute options: {', '.join(kwargs)}")
        _record(self.boundary, self.operation, **self.args)
        if self.boundary == "docs" and self.operation == "documents.batchUpdate":
            write_control = self.args.get("body", {}).get("writeControl") or {}
            required_revision = write_control.get("requiredRevisionId")
            document_id = self.args["documentId"]
            current_revision = _STATE.docs_revisions.get(document_id)
            if required_revision is not None and current_revision is not None and required_revision != current_revision:
                raise RuntimeError(
                    f"Stale Docs revision for {document_id}: required {required_revision}, current {current_revision}"
                )
            response = _script(self.boundary, self.operation)
            response_control = response.get("writeControl", {}) if isinstance(response, dict) else {}
            new_revision = response_control.get("requiredRevisionId")
            if new_revision is None and isinstance(response, dict):
                new_revision = response.get("revisionId")
            if new_revision is None:
                _STATE.docs_revision_counter += 1
                new_revision = f"pytest-docs-revision-{_STATE.docs_revision_counter}"
            _STATE.docs_revisions[document_id] = new_revision
            return response
        response = _script(self.boundary, self.operation)
        if self.boundary == "docs" and self.operation == "documents.get" and isinstance(response, dict):
            revision = response.get("revisionId")
            if revision is not None:
                _STATE.docs_revisions[self.args["documentId"]] = revision
        return response


class _Resource:
    def __init__(self, boundary: str, name: str) -> None:
        self.boundary, self.name = boundary, name

    def __getattr__(self, operation: str):
        supported = {
            "files": {
                "list": {"q", "fields", "pageSize", "pageToken", "orderBy", "spaces", "includeItemsFromAllDrives", "supportsAllDrives", "corpora", "driveId"},
                "get": {"fileId", "fields", "supportsAllDrives", "acknowledgeAbuse"},
                "create": {"body", "fields", "supportsAllDrives", "supportsTeamDrives"},
                "generateIds": {"count", "space", "type", "fields"},
                "update": {"fileId", "body", "addParents", "removeParents", "fields", "supportsAllDrives", "supportsTeamDrives"},
            },
            "changes": {
                "getStartPageToken": {"driveId", "supportsAllDrives", "supportsTeamDrives"},
                "list": {"pageToken", "fields", "spaces", "includeItemsFromAllDrives", "supportsAllDrives", "includeRemoved", "includeCorpusRemovals", "restrictToMyDrive", "includeTeamDriveItems", "supportsTeamDrives", "teamDriveId", "pageSize"},
            },
            "documents": {
                "get": {"documentId", "includeTabsContent", "suggestionsViewMode"},
                "batchUpdate": {"documentId", "body"},
            },
        }
        allowed_options = supported.get(self.name, {}).get(operation)
        if allowed_options is None:
            raise AttributeError(operation)
        def call(**kwargs: Any) -> _Request:
            extra = kwargs.keys() - allowed_options
            if extra:
                raise TypeError(f"Unsupported {self.name}.{operation} options: {', '.join(sorted(extra))}")
            return _Request(self.boundary, f"{self.name}.{operation}", kwargs)
        return call


class _DiscoveryService:
    def __init__(self, api: SDKFakes, name: str, version: str, kwargs: dict[str, Any]) -> None:
        self.api, self.name, self.version = api, name, version
        _record("drive", "build", api=name, version=version, **kwargs)

    def files(self): return _Resource("drive", "files")
    def changes(self): return _Resource("drive", "changes")
    def documents(self): return _Resource("docs", "documents")


def fake_discovery_build(api: str, version: str, **kwargs: Any) -> _DiscoveryService:
    if api not in {"drive", "docs"}:
        raise NotImplementedError(f"Unsupported discovery API: {api}")
    if kwargs.keys() - {"http", "credentials", "cache_discovery", "num_retries"}:
        raise TypeError(f"Unsupported discovery options: {', '.join(kwargs.keys() - {'http', 'credentials', 'cache_discovery', 'num_retries'})}")
    return _DiscoveryService(sdk_fakes, api, version, kwargs)


def _refresh_credentials(self: Credentials, request: Any) -> None:
    _record("auth", "credentials.refresh", request_type=type(request).__name__, scopes=self.scopes, token_uri=self.token_uri)
    self.token = "pytest-synthetic-access-token"
    # google-auth compares naive UTC datetimes in this installed SDK.
    self.expiry = google_auth_helpers.utcnow() + timedelta(hours=1)


def _auth_default(*, scopes=None, **kwargs: Any):
    if kwargs:
        raise TypeError(f"Unsupported google.auth.default options: {', '.join(kwargs)}")
    from google.auth.credentials import AnonymousCredentials
    _record("auth", "default", scopes=scopes)
    return AnonymousCredentials(), "pytest-synthetic-project"


class _GeminiModels:
    def embed_content(self, **kwargs: Any):
        extra = kwargs.keys() - {"model", "contents", "config"}
        if extra:
            raise TypeError(f"Unsupported Gemini embed_content options: {', '.join(sorted(extra))}")
        _record("gemini", "embed_content", **kwargs)
        return _script("gemini", "embed_content")

    def generate_content(self, **kwargs: Any):
        extra = kwargs.keys() - {"model", "contents", "config", "safety_settings", "system_instruction", "tools", "tool_config", "labels", "cached_content"}
        if extra:
            raise TypeError(f"Unsupported Gemini generate_content options: {', '.join(sorted(extra))}")
        _record("gemini", "generate_content", **kwargs)
        response = _script("gemini", "generate_content")
        return _adapt_scripted_route_response(response, kwargs)


def _member(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _model_replace(value: Any, name: str, item: Any) -> Any:
    if isinstance(value, dict):
        result = dict(value)
        result[name] = item
        return result
    copier = getattr(value, "model_copy", None)
    if callable(copier):
        return copier(update={name: item})
    clone = copy.copy(value)
    setattr(clone, name, item)
    return clone


def _routing_prompt(contents: Any) -> dict[str, Any] | None:
    """Extract only the JSON route prompt from the actual SDK request contents."""
    safe = _json_safe_fake(contents)
    items = safe if isinstance(safe, list) else [safe]
    for item in items:
        parts = item.get("parts", []) if isinstance(item, dict) else []
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else None
            if not isinstance(text, str):
                continue
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                continue
            if (isinstance(value, dict) and isinstance(value.get("command"), str)
                    and isinstance(value.get("context"), dict)
                    and isinstance(value.get("documents"), list)
                    and isinstance(value.get("response_schema"), dict)):
                return value
    return None


def _json_safe_fake(value: Any) -> Any:
    """Small SDK fake serializer supporting mappings and actual genai model objects."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, dict):
        return {key: _json_safe_fake(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_fake(item) for item in value]
    converter = getattr(value, "to_json_dict", None)
    if callable(converter):
        return _json_safe_fake(converter())
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return _json_safe_fake(dumper(mode="json", exclude_none=False))
        except TypeError:
            return _json_safe_fake(dumper())
    if hasattr(value, "__dict__"):
        return {key: _json_safe_fake(item) for key, item in vars(value).items()
                if not key.startswith("_")}
    return value


def _route_declaration_present(config: Any) -> bool:
    """Require the fresh route_command declaration, not a generic agent call."""
    for tool in _member(config, "tools", []) or []:
        declarations = _member(tool, "function_declarations", []) or []
        for declaration in declarations:
            if _member(declaration, "name") != "route_command":
                continue
            schema = _json_safe_fake(_member(declaration, "parameters_json_schema"))
            if not isinstance(schema, dict):
                schema = _json_safe_fake(_member(declaration, "parameters"))
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
            if (isinstance(props, dict) and "action" in props and "disposition" in props
                    and "target_ref" not in props
                    and {"action", "disposition", "substantive_text", "document_selector"}
                    <= required):
                return True
    return False


def _rewrite_route_call(parts: Any, *, prompt: dict[str, Any]) -> tuple[Any, bool]:
    if not isinstance(parts, (tuple, list)):
        return parts, False
    try:
        from services.connectors.telegram.sessions import _valid_persisted_route
        from tests.fixtures.semantic_routing import legacy_route_fixture_to_semantic
    except Exception:
        return parts, False
    legacy_fields = {
        "kind", "target_ref", "dialog_number", "result_refs", "substantive_text",
        "tool_choice", "certainty", "target_operation_id",
    }
    context = prompt.get("context", {})
    active_conversation_id = context.get("active_conversation_id")
    active_cards = [card for card in prompt.get("documents", [])
                    if isinstance(card, dict)
                    and card.get("conversation_id") == active_conversation_id
                    and isinstance(card.get("doc_id"), str)]
    active_doc_id = active_cards[0]["doc_id"] if len({card["doc_id"] for card in active_cards}) == 1 else None
    active_numbers = {card.get("dialog_number") for card in active_cards
                      if type(card.get("dialog_number")) is int}
    active_dialog_number = next(iter(active_numbers)) if len(active_numbers) == 1 else None
    captured_draft = context.get("captured_collecting_draft")
    source_bound = bool(
        isinstance(captured_draft, dict)
        or context.get("forwarding_source_bound") is True
    )
    source_bound_dialog_number = None
    if isinstance(captured_draft, dict):
        source_bound_dialog_number = captured_draft.get("dialog_number")
    elif source_bound:
        number = context.get("forwarding_dialog_number")
        source_bound_dialog_number = number if type(number) is int else None
    converted, changed = [], False
    known_doc_ids = [card.get("doc_id") for card in prompt.get("documents", [])
                     if isinstance(card, dict) and isinstance(card.get("doc_id"), str)]
    for part in parts:
        call = _member(part, "function_call")
        args = _member(call, "args") if call is not None else None
        if (_member(call, "name") == "route_command" and isinstance(args, dict)
                and set(args) == legacy_fields):
            _normalized, legacy_status = _valid_persisted_route(args)
            if legacy_status in {"ok", "uncertain"}:
                semantic = legacy_route_fixture_to_semantic(
                    args, prompt["command"], active_doc_id=active_doc_id,
                    active_dialog_number=active_dialog_number,
                    source_bound_draft=source_bound,
                    source_bound_dialog_number=source_bound_dialog_number,
                    known_doc_ids=known_doc_ids,
                )
                call = _model_replace(call, "args", semantic)
                part = _model_replace(part, "function_call", call)
                changed = True
        converted.append(part)
    return (type(parts)(converted) if isinstance(parts, tuple) else converted), changed


def _adapt_scripted_route_response(response: Any, kwargs: dict[str, Any]) -> Any:
    """Migrate only valid legacy route_command test replies using the real prompt."""
    config = kwargs.get("config")
    if not _route_declaration_present(config):
        return response
    prompt = _routing_prompt(kwargs.get("contents"))
    if prompt is None:
        return response
    candidates = _member(response, "candidates")
    if isinstance(candidates, (tuple, list)):
        rewritten_candidates, changed = [], False
        for candidate in candidates:
            content = _member(candidate, "content")
            parts = _member(content, "parts")
            new_parts, part_changed = _rewrite_route_call(parts, prompt=prompt)
            if part_changed:
                content = _model_replace(content, "parts", new_parts)
                candidate = _model_replace(candidate, "content", content)
                changed = True
            rewritten_candidates.append(candidate)
        if changed:
            return _model_replace(response, "candidates", rewritten_candidates)
        return response
    parts = _member(response, "parts")
    rewritten, changed = _rewrite_route_call(parts, prompt=prompt)
    return _model_replace(response, "parts", rewritten) if changed else response


class _GeminiFiles:
    def upload(self, **kwargs: Any):
        extra = kwargs.keys() - {"file", "config"}
        if extra:
            raise TypeError(f"Unsupported Gemini files.upload options: {', '.join(sorted(extra))}")
        _record("gemini", "files.upload", **kwargs)
        return _script("gemini", "files.upload")

    def delete(self, **kwargs: Any):
        extra = kwargs.keys() - {"name", "config"}
        if extra:
            raise TypeError(f"Unsupported Gemini files.delete options: {', '.join(sorted(extra))}")
        _record("gemini", "files.delete", **kwargs)
        return _script("gemini", "files.delete")


class _AsyncGeminiModels:
    """Async facade sharing the existing Gemini response queues and trace."""

    def __init__(self, sync_models: _GeminiModels) -> None:
        self._sync = sync_models

    async def generate_content(self, **kwargs: Any):
        return self._sync.generate_content(**kwargs)


class _AsyncGeminiFiles:
    """Async facade sharing the existing Gemini file queues and trace."""

    def __init__(self, sync_files: _GeminiFiles) -> None:
        self._sync = sync_files

    async def upload(self, **kwargs: Any):
        return self._sync.upload(**kwargs)

    async def delete(self, **kwargs: Any):
        return self._sync.delete(**kwargs)


class FakeGeminiClient:
    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        if kwargs.keys() - {"http_options"}:
            raise TypeError(f"Unsupported Gemini client options: {', '.join(kwargs)}")
        _record("gemini", "client", api_key=api_key, **kwargs)
        self.models = _GeminiModels()
        self.files = _GeminiFiles()
        self.aio = SimpleNamespace(
            models=_AsyncGeminiModels(self.models),
            files=_AsyncGeminiFiles(self.files),
        )


class FakeQdrantClient:
    def __init__(self, *, url: str, api_key: str | None = None, **kwargs: Any) -> None:
        if kwargs.keys() - {"check_compatibility"}:
            raise TypeError(f"Unsupported Qdrant client options: {', '.join(kwargs)}")
        _record("qdrant", "client", url=url, api_key=api_key, **kwargs)
        _STATE.clients.append(self)

    def get_collections(self):
        _record("qdrant", "get_collections")
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in _STATE.qdrant_collections])

    def create_collection(self, **kwargs: Any):
        allowed = {"collection_name", "vectors_config", "sparse_vectors_config", "shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload", "hnsw_config", "optimizers_config", "wal_config", "quantization_config", "timeout", "strict_mode_config", "metadata", "payload"}
        extra = kwargs.keys() - allowed
        if extra:
            raise TypeError(f"Unsupported Qdrant create_collection options: {', '.join(sorted(extra))}")
        _record("qdrant", "create_collection", **kwargs)
        _STATE.qdrant_collections[kwargs["collection_name"]] = {"payload_schema": {}}
        return True

    def get_collection(self, collection_name: str, **kwargs: Any):
        if kwargs.keys() - {"timeout", "consistency"}:
            raise TypeError(f"Unsupported Qdrant get_collection options: {', '.join(sorted(kwargs.keys() - {'timeout', 'consistency'}))}")
        _record("qdrant", "get_collection", collection_name=collection_name, **kwargs)
        if collection_name in _STATE.qdrant_collections:
            return SimpleNamespace(payload_schema=dict(_STATE.qdrant_collections[collection_name]["payload_schema"]))
        return _script("qdrant", "get_collection")

    def create_payload_index(self, **kwargs: Any):
        allowed = {"collection_name", "field_name", "field_schema", "field_type", "wait", "ordering", "timeout"}
        extra = kwargs.keys() - allowed
        if extra:
            raise TypeError(f"Unsupported Qdrant create_payload_index options: {', '.join(sorted(extra))}")
        _record("qdrant", "create_payload_index", **kwargs)
        name = kwargs["collection_name"]
        if name in _STATE.qdrant_collections:
            _STATE.qdrant_collections[name]["payload_schema"][kwargs["field_name"]] = kwargs["field_schema"]
            return True
        return _script("qdrant", "create_payload_index")

    def __getattr__(self, operation: str):
        if operation not in {"upsert", "delete", "set_payload", "scroll", "query_points", "retrieve"}:
            raise AttributeError(operation)
        def call(*args: Any, **kwargs: Any):
            if args:
                raise TypeError(f"Fake Qdrant {operation} accepts keyword arguments only")
            allowed_options = {
                "upsert": {"collection_name", "points", "wait", "ordering", "shard_key_selector", "update_filter", "update_mode", "timeout"},
                "delete": {"collection_name", "points_selector", "wait", "ordering", "shard_key_selector", "timeout"},
                "set_payload": {"collection_name", "payload", "points", "key", "wait", "ordering", "shard_key_selector", "timeout"},
                "scroll": {"collection_name", "scroll_filter", "limit", "order_by", "offset", "with_payload", "with_vectors", "consistency", "shard_key_selector", "timeout"},
                "query_points": {"collection_name", "query", "using", "prefetch", "query_filter", "search_params", "limit", "offset", "with_payload", "with_vectors", "score_threshold", "lookup_from", "consistency", "shard_key_selector", "timeout"},
                "retrieve": {"collection_name", "ids", "with_payload", "with_vectors", "consistency", "shard_key_selector", "timeout"},
            }[operation]
            extra = kwargs.keys() - allowed_options
            if extra:
                raise TypeError(f"Unsupported Qdrant {operation} options: {', '.join(sorted(extra))}")
            _record("qdrant", operation, args=args, **kwargs)
            if operation in {"upsert", "delete", "set_payload"}:
                return _script("qdrant", operation, True)
            return _script("qdrant", operation)
        return call


class _SparseVector:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices, self.values = _Array(indices), _Array(values)


class _Array(list):
    def tolist(self): return list(self)


class FakeSparseTextEmbedding:
    def __init__(self, model_name: str = "Qdrant/bm25", **kwargs: Any) -> None:
        if model_name != "Qdrant/bm25" or kwargs:
            raise TypeError("Only SparseTextEmbedding(model_name='Qdrant/bm25') is supported")
        _record("sparse", "client", model_name=model_name)

    def query_embed(self, texts: list[str]):
        for text in texts:
            _record("sparse", "query_embed", text=text)
            stable_index = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
            indices, values = _STATE.sparse.get(text, ([stable_index], [1.0]))
            yield _SparseVector(list(indices), list(values))

    def embed(self, texts: list[str]):
        for text in texts:
            _record("sparse", "embed", text=text)
            stable_index = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
            indices, values = _STATE.sparse.get(text, ([stable_index], [1.0]))
            yield _SparseVector(list(indices), list(values))


class FakeCloudTasksClient:
    def __init__(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"Unsupported Cloud Tasks options: {', '.join(kwargs)}")
        _record("tasks", "client")

    @staticmethod
    def queue_path(project: str, location: str, queue: str) -> str:
        return RealCloudTasksClient.queue_path(project, location, queue)

    @staticmethod
    def task_path(project: str, location: str, queue: str, task: str) -> str:
        return RealCloudTasksClient.task_path(project, location, queue, task)

    def create_task(self, *, parent: str, task: dict[str, Any] | tasks_v2.Task, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Unsupported create_task options: {', '.join(kwargs)}")
        _record("tasks", "create_task", parent=parent, task=task)
        injected = _STATE.queues[("tasks", "create_task")]
        configured = injected.popleft() if injected else _MISSING
        if isinstance(configured, BaseException):
            raise configured
        name = task.get("name") if isinstance(task, dict) else task.name
        if name and name in _STATE.tasks:
            raise AlreadyExists("Task already exists")
        if name:
            _STATE.tasks.add(name)
        else:
            _STATE.anonymous_task_counter += 1
            name = f"{parent}/tasks/fake-anonymous-{_STATE.anonymous_task_counter}"
        if configured is not _MISSING:
            return _copy(configured)
        return tasks_v2.Task(name=name)


def _guard_request_send(name: str, original: Any):
    def guarded(self, *args: Any, **kwargs: Any):
        request = args[0] if args else kwargs.get("request")
        url = getattr(request, "url", None)
        if name.startswith("httpx."):
            if url is None:
                raise RuntimeError("pytest offline guard blocked outbound request")
            selected = self._transport_for_url(url)
            if isinstance(selected, (httpx.ASGITransport, httpx.MockTransport)):
                return original(self, *args, **kwargs)
            if _is_loopback_host(getattr(url, "host", None)):
                return original(self, *args, **kwargs)
        elif name == "requests.Session.send":
            if _is_loopback_host(urlsplit(str(url)).hostname):
                return original(self, *args, **kwargs)
        raise RuntimeError("pytest offline guard blocked outbound request")
    return guarded


def _guard_async_request_send(original: Any):
    async def guarded(self, request: Any, *args: Any, **kwargs: Any):
        url = getattr(request, "url", None)
        if url is None:
            raise RuntimeError("pytest offline guard blocked outbound request")
        selected = self._transport_for_url(url)
        if isinstance(selected, (httpx.ASGITransport, httpx.MockTransport)):
            return await original(self, request, *args, **kwargs)
        if _is_loopback_host(getattr(url, "host", None)):
            return await original(self, request, *args, **kwargs)
        raise RuntimeError("pytest offline guard blocked outbound request")
    return guarded


def _guard_requests_request(original: Any):
    def guarded(self, method: Any, url: Any, *args: Any, **kwargs: Any):
        if _is_loopback_host(urlsplit(str(url)).hostname):
            return original(self, method, url, *args, **kwargs)
        raise RuntimeError("pytest offline guard blocked outbound request")
    return guarded


def _guard_httplib2_request(original: Any):
    def guarded(self, *args: Any, **kwargs: Any):
        uri = args[0] if args else kwargs.get("uri")
        if uri and _is_loopback_host(urlsplit(str(uri)).hostname):
            return original(self, *args, **kwargs)
        raise RuntimeError("pytest offline guard blocked outbound request")
    return guarded


def _guard_grpc_channel(name: str, original: Any):
    def guarded(*args: Any, **kwargs: Any):
        target = args[0] if args else kwargs.get("target")
        if target is None:
            raise RuntimeError("pytest offline guard blocked outbound gRPC channel")
        hostport = str(target).split("://")[-1].split("/")[-1]
        if hostport.startswith("[") and "]" in hostport:
            host = hostport[1:hostport.index("]")]
        else:
            host = hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport
        if _is_loopback_host(host):
            return original(*args, **kwargs)
        raise RuntimeError("pytest offline guard blocked outbound gRPC channel")
    return guarded


def _is_loopback_host(host: Any) -> bool:
    if not isinstance(host, str) or not host:
        return False
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_ACTIVE_PATCHES: dict[str, tuple[Any, str, Any, Any]] | None = None
_CACHE_SNAPSHOTS: dict[int, dict[tuple[str, str], tuple[Any, str, Any]]] = {}
_RESTORED_PATCHES: dict[int, dict[str, tuple[Any, str, Any, Any]]] = {}


def _apply_patch(obj: Any, name: str, replacement: Any) -> None:
    setattr(obj, name, replacement)


def install() -> dict[str, tuple[Any, str, Any, Any]]:
    """Install idempotent, rollback-safe SDK and network boundaries."""
    global _ACTIVE_PATCHES
    if _ACTIVE_PATCHES is not None:
        return _ACTIVE_PATCHES

    originals: dict[str, tuple[Any, str, Any, Any]] = {}
    cache_snapshot: dict[tuple[str, str], tuple[Any, str, Any]] = {}
    cache_symbols = {
        "core.integrations.qdrant.client": ("_client", "_single_attempt_client"),
        "core.integrations.fastembed.bm25": ("_bm25_model",),
    }
    for module_name, attrs in cache_symbols.items():
        module = sys.modules.get(module_name)
        if module is not None:
            for attr in attrs:
                if hasattr(module, attr):
                    cache_snapshot[(module_name, attr)] = (module, attr, getattr(module, attr))
    _CACHE_SNAPSHOTS[id(originals)] = cache_snapshot
    real_to_fake = {
        id(_REAL_AUTH_DEFAULT): _auth_default,
        id(_REAL_GENAI_CLIENT): FakeGeminiClient,
        id(_REAL_QDRANT_CLIENT): FakeQdrantClient,
        id(_REAL_TASKS_CLIENT): FakeCloudTasksClient,
        id(_REAL_SPARSE_EMBEDDING): FakeSparseTextEmbedding,
        id(_REAL_DISCOVERY_BUILD): fake_discovery_build,
    }

    def patch(obj: Any, name: str, replacement: Any, old: Any) -> None:
        if old is replacement:
            return
        key = f"{id(obj)}:{name}"
        if key not in originals:
            originals[key] = (obj, name, old, replacement)
        _apply_patch(obj, name, replacement)

    try:
        # All SDK aliases are matched by identity so local wrappers remain untouched.
        for module_name, module in tuple(sys.modules.items()):
            if module is None or module_name == __name__:
                continue
            try:
                attributes = tuple(vars(module).items())
            except TypeError:
                continue
            for name, value in attributes:
                replacement = real_to_fake.get(id(value))
                if replacement is not None and not name.startswith("Fake"):
                    patch(module, name, replacement, value)

        patch(Credentials, "refresh", _refresh_credentials, Credentials.refresh)

        for obj, attr, guard_name in (
            (httpx.Client, "send", "httpx.Client.send"),
            (requests.Session, "request", "requests.Session.request"),
            (requests.Session, "send", "requests.Session.send"),
            (httplib2.Http, "request", "httplib2.Http.request"),
            (grpc, "insecure_channel", "grpc.insecure_channel"),
            (grpc, "secure_channel", "grpc.secure_channel"),
            (grpc.aio, "insecure_channel", "grpc.aio.insecure_channel"),
            (grpc.aio, "secure_channel", "grpc.aio.secure_channel"),
        ):
            original = getattr(obj, attr)
            if guard_name.startswith("httpx.") or guard_name == "requests.Session.send":
                replacement = _guard_request_send(guard_name, original)
            elif guard_name == "requests.Session.request":
                replacement = _guard_requests_request(original)
            elif guard_name == "httplib2.Http.request":
                replacement = _guard_httplib2_request(original)
            else:
                replacement = _guard_grpc_channel(guard_name, original)
            patch(obj, attr, replacement, original)

        async_send = httpx.AsyncClient.send
        patch(httpx.AsyncClient, "send", _guard_async_request_send(async_send), async_send)

        _ACTIVE_PATCHES = originals
        return originals
    except BaseException:
        restore(originals)
        raise


def restore(originals: dict[str, tuple[Any, str, Any, Any]] | None) -> None:
    global _ACTIVE_PATCHES
    if not originals:
        if _ACTIVE_PATCHES is None:
            return
        originals = _ACTIVE_PATCHES
    if _RESTORED_PATCHES.get(id(originals)) is originals:
        return

    for obj, name, old, replacement in reversed(list(originals.values())):
        if getattr(obj, name, None) is replacement:
            setattr(obj, name, old)

    # Imports during collection capture the active fakes. Restore only those exact
    # fake identities; unrelated wrappers and monkeypatches are left intact.
    fake_to_real = {
        id(_auth_default): _REAL_AUTH_DEFAULT,
        id(FakeGeminiClient): _REAL_GENAI_CLIENT,
        id(FakeQdrantClient): _REAL_QDRANT_CLIENT,
        id(FakeCloudTasksClient): _REAL_TASKS_CLIENT,
        id(FakeSparseTextEmbedding): _REAL_SPARSE_EMBEDDING,
        id(fake_discovery_build): _REAL_DISCOVERY_BUILD,
    }
    for module_name, module in tuple(sys.modules.items()):
        if module is None or module_name == __name__:
            continue
        try:
            attributes = tuple(vars(module).items())
        except TypeError:
            continue
        for name, value in attributes:
            real = fake_to_real.get(id(value))
            if real is not None and not name.startswith("Fake") and name not in {"fake_discovery_build", "_auth_default"}:
                setattr(module, name, real)

    cache_snapshot = _CACHE_SNAPSHOTS.pop(id(originals), {})
    for module, attr, value in cache_snapshot.values():
        setattr(module, attr, value)
    for module_name, attr, fake_type in (
        ("core.integrations.qdrant.client", "_client", FakeQdrantClient),
        ("core.integrations.qdrant.client", "_single_attempt_client", FakeQdrantClient),
        ("core.integrations.fastembed.bm25", "_bm25_model", FakeSparseTextEmbedding),
    ):
        module = sys.modules.get(module_name)
        if module is not None and (module_name, attr) not in cache_snapshot:
            value = getattr(module, attr, None)
            if isinstance(value, fake_type):
                setattr(module, attr, None)

    if _ACTIVE_PATCHES is originals:
        _ACTIVE_PATCHES = None
    _RESTORED_PATCHES[id(originals)] = originals

