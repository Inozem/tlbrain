from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import grpc
import pytest
from google.auth import _helpers as google_auth_helpers
from google.auth.transport.requests import Request
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from google.oauth2.credentials import Credentials
from qdrant_client.models import Distance, SparseVectorParams, VectorParams
from qdrant_client.http.models.models import QueryResponse, ScoredPoint

from core.integrations.gemini import embeddings, llm
from core.google_drive import drive_client
from core.google_drive import docs_reader
from core.integrations.qdrant import client as qdrant_client
from core.integrations.qdrant import setup as qdrant_setup
from core.integrations.cloud_tasks.client import enqueue_task
from tests.fixtures.sdk_fakes import FakeQdrantClient, fake_discovery_build
from tests.fixtures import sdk_fakes as sdk_fakes_module


def test_drive_boundary_keeps_real_oauth_and_transport_construction(sdk_fakes, monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-secret")
    sdk_fakes.drive.queue("files.list", response={"files": [{"id": "folder-1", "name": "Ada"}]})

    credentials = drive_client._build_credentials(drive_client.SCOPES_RO)
    assert isinstance(credentials, Credentials)
    assert credentials.refresh_token == "synthetic-refresh"
    assert credentials.token_uri == "https://oauth2.googleapis.com/token"
    assert credentials.token == "pytest-synthetic-access-token"
    assert credentials.valid is True
    headers = {}
    credentials.before_request(Request(), "GET", "https://drive.googleapis.com/", headers)
    assert headers["authorization"] == "Bearer pytest-synthetic-access-token"
    assert isinstance(Request(), Request)
    http = drive_client._build_http(credentials)
    assert http.http.timeout == 60

    service = drive_client.build_drive_service()
    assert service.files().list(fields="files(id,name)").execute()["files"][0]["id"] == "folder-1"
    build = next(e for e in sdk_fakes.trace if e["operation"] == "build")
    assert build["arguments"]["api"] == "drive"
    assert build["arguments"]["http"].http.timeout == 60
    assert sdk_fakes.trace[-1]["operation"] == "files.list"


def test_credential_refresh_expiry_uses_google_auth_utc_clock(sdk_fakes, monkeypatch):
    utc_now = datetime(2026, 1, 15, 12, 0, 0)
    monkeypatch.setattr(google_auth_helpers, "utcnow", lambda: utc_now)

    class LocalClock:
        @staticmethod
        def now():
            # Simulate a machine whose local wall clock is five hours behind UTC.
            return utc_now - timedelta(hours=5)

    monkeypatch.setattr(sdk_fakes_module, "datetime", LocalClock)
    credentials = Credentials(
        token=None,
        refresh_token="synthetic-refresh",
        client_id="synthetic-client",
        client_secret="synthetic-secret",
        token_uri="https://oauth2.googleapis.com/token",
    )

    credentials.refresh(Request())

    assert credentials.expiry == utc_now + timedelta(hours=1)
    assert credentials.valid is True
    headers = {}
    credentials.before_request(Request(), "GET", "https://drive.googleapis.com/", headers)
    assert headers["authorization"] == "Bearer pytest-synthetic-access-token"
    assert sum(e["operation"] == "credentials.refresh" for e in sdk_fakes.trace) == 1


def test_drive_without_user_oauth_still_fails(sdk_fakes, monkeypatch):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_REFRESH_TOKEN is not set"):
        drive_client._build_credentials(drive_client.SCOPES_RO)
    assert sdk_fakes.trace == []
    with pytest.raises(TypeError, match="Unsupported files.list options"):
        fake_discovery_build("drive", "v3").files().list(unexpected_option=True)


def test_docs_reader_uses_real_reader_and_docs_service_boundary(sdk_fakes, monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-secret")
    sdk_fakes.docs.queue("documents.get", response={"body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "Meeting notes\n"}}]}}
    ]}})
    assert docs_reader.read_google_doc("doc-1") == "Meeting notes"
    build = next(e for e in sdk_fakes.trace if e["operation"] == "build")
    assert build["arguments"]["api"] == "docs"
    assert build["arguments"]["version"] == "v1"
    assert sdk_fakes.trace[-1]["operation"] == "documents.get"


def test_gemini_consumers_use_scripted_real_response_models(sdk_fakes):
    sdk_fakes.gemini.queue(
        "embed_content",
        response=types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.25, 0.5])]),
    )
    assert embeddings.embed(["hello"]) == [[0.25, 0.5]]
    embedding_call = next(e for e in sdk_fakes.trace if e["operation"] == "embed_content")
    assert embedding_call["arguments"]["model"] == "gemini-embedding-2"
    assert embedding_call["arguments"]["config"].output_dimensionality == 768

    sdk_fakes.gemini.queue(
        "generate_content",
        response=types.GenerateContentResponse(candidates=[
            types.Candidate(content=types.Content(parts=[
                types.Part(text='{"folder_name":"Ada","confidence":0.9}')
            ]))
        ]),
    )
    assert llm.call_gemini_json("choose") == {"folder_name": "Ada", "confidence": 0.9}
    generated = [e for e in sdk_fakes.trace if e["operation"] == "generate_content"][-1]
    assert generated["arguments"]["config"].response_mime_type == "application/json"


def test_gemini_empty_and_error_responses_remain_observable(sdk_fakes):
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]
    ))
    with pytest.raises(llm.GeminiEmptyResponseError):
        llm._generate_structured(llm.make_client(), "prompt")
    sdk_fakes.gemini.queue("embed_content", error=ValueError("scripted Gemini failure"))
    with pytest.raises(ValueError, match="scripted Gemini failure"):
        embeddings.make_client().models.embed_content(model="test", contents=["x"])


def test_qdrant_startup_uses_real_models_and_queries_are_scripted(sdk_fakes, monkeypatch):
    import importlib

    monkeypatch.setenv("QDRANT_COLLECTION", "component-test")
    qdrant_client._client = None
    vector_main = importlib.reload(importlib.import_module("services.vector_sync.app.main"))
    startup = sdk_fakes.trace
    assert [e["operation"] for e in startup[:3]] == ["client", "get_collections", "create_collection"]
    assert vector_main.ensure_collection is qdrant_setup.ensure_collection
    create = next(e for e in startup if e["operation"] == "create_collection")
    assert isinstance(create["arguments"]["vectors_config"]["dense"], VectorParams)
    assert create["arguments"]["vectors_config"]["dense"].distance == Distance.COSINE
    assert isinstance(create["arguments"]["sparse_vectors_config"]["bm25"], SparseVectorParams)
    assert FakeQdrantClient is not None

    expected = QueryResponse(points=[ScoredPoint(id="point-1", version=1, score=0.8, payload={"doc_id": "d1"})])
    sdk_fakes.qdrant.queue("query_points", response=expected)
    result = qdrant_client.get_client().query_points(collection_name="component-test", query=[1.0], limit=1)
    assert result.points[0].payload["doc_id"] == "d1"
    assert sdk_fakes.trace[-1]["arguments"]["limit"] == 1
    with pytest.raises(TypeError, match="Unsupported Qdrant query_points options"):
        qdrant_client.get_client().query_points(collection_name="component-test", unexpected_option=True)


def test_sparse_vectors_are_configurable_and_deterministic(sdk_fakes):
    from core.retrieval import search

    sdk_fakes.sparse.set_vectors("known phrase", [8, 3], [0.5, 0.25])
    sparse = search._get_bm25_model()
    first = next(sparse.embed(["known phrase"]))
    second = next(sparse.embed(["known phrase"]))
    assert first.indices.tolist() == second.indices.tolist() == [8, 3]
    assert first.values.tolist() == second.values.tolist() == [0.5, 0.25]
    assert [e["operation"] for e in sdk_fakes.trace if e["boundary"] == "sparse"][-2:] == ["embed", "embed"]
    list(sparse.query_embed(["known phrase"]))
    assert sdk_fakes.trace[-1]["operation"] == "query_embed"


def test_cloud_tasks_real_paths_named_duplicates_and_anonymous_tasks(sdk_fakes, monkeypatch):
    monkeypatch.setenv("REGION", "test-region")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "override-project")
    assert enqueue_task("queue", "https://worker.invalid/task", "named", {"x": 1}) is True
    assert enqueue_task("queue", "https://worker.invalid/task", "named", {"x": 1}) is False
    assert enqueue_task("queue", "https://worker.invalid/task") is True
    assert enqueue_task("queue", "https://worker.invalid/task", body={}) is True
    task_calls = [e for e in sdk_fakes.trace if e["operation"] == "create_task"]
    assert len(task_calls) == 4
    parent = "projects/override-project/locations/test-region/queues/queue"
    assert all(call["arguments"]["parent"] == parent for call in task_calls)
    named_path = f"{parent}/tasks/named"
    assert task_calls[0]["arguments"]["task"]["name"] == named_path
    request = task_calls[0]["arguments"]["task"]["http_request"]
    from google.cloud import tasks_v2
    assert request["http_method"] == tasks_v2.HttpMethod.POST
    assert request["url"] == "https://worker.invalid/task"
    assert request["headers"] == {"Content-Type": "application/json"}
    assert request["body"] == b'{"x": 1}'
    assert task_calls[2]["arguments"]["task"].get("name") is None
    assert "headers" not in task_calls[2]["arguments"]["task"]["http_request"]
    empty_request = task_calls[3]["arguments"]["task"]["http_request"]
    assert empty_request["headers"] == {"Content-Type": "application/json"}
    assert empty_request["body"] == b"{}"

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path("pytest-synthetic-project", "test-region", "queue")
    request = {"http_request": {"http_method": tasks_v2.HttpMethod.POST, "url": "https://worker.invalid/"}}
    first = client.create_task(parent=parent, task=request)
    second = client.create_task(parent=parent, task=request)
    assert first.name.startswith(f"{parent}/tasks/fake-anonymous-")
    assert second.name.startswith(f"{parent}/tasks/fake-anonymous-")
    assert first.name != second.name


def test_google_auth_default_and_tasks_errors_are_scriptable(sdk_fakes, monkeypatch):
    import google.auth

    creds, project = google.auth.default(scopes=["synthetic-scope"])
    assert isinstance(creds, AnonymousCredentials)
    assert project == "pytest-synthetic-project"
    monkeypatch.setenv("REGION", "test-region")
    sdk_fakes.tasks.queue("create_task", error=RuntimeError("scripted queue failure"))
    with pytest.raises(RuntimeError, match="scripted queue failure"):
        enqueue_task("queue", "https://worker.invalid/task")


def test_http_boundary_allows_mock_transport_and_blocks_external_session(sdk_fakes):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=transport) as client:
        assert client.get("https://example.invalid/").json() == {"ok": True}
    import requests
    with httpx.Client() as harmless:
        with pytest.raises(RuntimeError, match="offline guard"):
            harmless.get("https://example.invalid/path?secret=synthetic")
    with httpx.Client(mounts={"https://mounted.invalid": transport}) as client:
        assert client.get("https://mounted.invalid/").json() == {"ok": True}
    with pytest.raises(RuntimeError, match="offline guard"):
        requests.Session().get("https://example.invalid/")
    session = requests.Session()
    prepared = session.prepare_request(requests.Request("GET", "https://example.invalid/"))
    with pytest.raises(RuntimeError, match="offline guard"):
        session.send(prepared)
    async_client = httpx.AsyncClient()
    coroutine = async_client.send(async_client.build_request("GET", "https://example.invalid/"))
    with pytest.raises(RuntimeError, match="offline guard"):
        coroutine.send(None)
    coroutine.close()
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(_async_mock_response))
    coroutine = async_client.get("https://example.invalid/")
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    assert completed.value.value.json() == {"ok": True}
    coroutine.close()
    with pytest.raises(RuntimeError, match="offline guard"):
        grpc.insecure_channel("example.invalid:443")
    with pytest.raises(RuntimeError, match="offline guard"):
        grpc.aio.secure_channel("example.invalid:443", grpc.ssl_channel_credentials())
    with pytest.raises(RuntimeError, match="offline guard"):
        grpc.aio.insecure_channel("example.invalid:443")
    with pytest.raises(RuntimeError, match="offline guard") as blocked:
        requests.Session().get("https://example.invalid/path?private=synthetic")
    assert "example.invalid" not in str(blocked.value)
    assert "private" not in str(blocked.value)


async def _async_mock_response(request):
    return httpx.Response(200, json={"ok": True})


def test_sdk_aliases_are_patched_by_identity_and_install_is_repeatable(sdk_fakes):
    import google.auth._default
    from google.genai import client as genai_client
    from qdrant_client import qdrant_client as qdrant_impl
    from google.cloud.tasks_v2.services import cloud_tasks
    from google.cloud.tasks_v2.services.cloud_tasks import client as tasks_impl
    from fastembed.sparse import sparse_text_embedding
    from tests.fixtures import sdk_fakes as fake_module

    assert google.auth._default.default is fake_module._auth_default
    assert genai_client.Client is fake_module.FakeGeminiClient
    assert qdrant_impl.QdrantClient is FakeQdrantClient
    assert cloud_tasks.CloudTasksClient is fake_module.FakeCloudTasksClient
    assert tasks_impl.CloudTasksClient is fake_module.FakeCloudTasksClient
    assert sparse_text_embedding.SparseTextEmbedding is fake_module.FakeSparseTextEmbedding
    assert isinstance(sdk_fakes, fake_module.SDKFakes)
    from tests import conftest
    assert fake_module.install() is conftest._saved_sdk_boundaries


def test_reset_separates_bootstrap_trace_and_clears_per_test_trace(sdk_fakes):
    from tests.fixtures.sdk_fakes import FakeQdrantClient

    sdk_fakes.begin_run()
    FakeQdrantClient(url="http://127.0.0.1:6333", api_key="synthetic")
    sdk_fakes.capture_bootstrap()
    assert sdk_fakes.trace[0]["operation"] == "client"
    assert sdk_fakes.bootstrap_trace[0]["operation"] == "client"
    sdk_fakes.reset()
    assert sdk_fakes.trace == []
    assert sdk_fakes.bootstrap_trace[0]["operation"] == "client"
    sdk_fakes.begin_run()
    assert sdk_fakes.trace == []
    assert sdk_fakes.bootstrap_trace == []


def test_trace_copy_keeps_transport_leaves_and_rejects_uncloneable_mutable_values():
    from tests.fixtures import sdk_fakes as fake_module

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    source = {"payload": {"values": [1]}, "transport": transport}
    copied = fake_module._copy(source)
    source["payload"]["values"].append(2)
    assert copied is not source
    assert copied["payload"]["values"] == [1]
    assert copied["transport"] is transport

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy unavailable")

    with pytest.raises(TypeError, match="Cannot safely copy"):
        fake_module._copy({"nested": [Uncopyable()]})


def test_install_and_configure_roll_back_partial_patches(sdk_fakes, monkeypatch):
    from tests import conftest
    from tests.fixtures import sdk_fakes as fake_module
    import google.auth._default
    from google.genai import client as genai_client
    from qdrant_client import qdrant_client as qdrant_impl
    from google.cloud.tasks_v2.services.cloud_tasks import client as tasks_impl
    from fastembed.sparse import sparse_text_embedding

    original_default = fake_module._REAL_AUTH_DEFAULT
    original_client = fake_module._REAL_GENAI_CLIENT
    original_qdrant = fake_module._REAL_QDRANT_CLIENT
    original_tasks = fake_module._REAL_TASKS_CLIENT
    original_sparse = fake_module._REAL_SPARSE_EMBEDDING
    active = conftest._saved_sdk_boundaries
    original_httpx_send = next(old for obj, name, old, _ in active.values() if obj is httpx.Client and name == "send")
    original_requests_send = next(old for obj, name, old, _ in active.values() if obj is __import__("requests").Session and name == "send")
    original_grpc_aio = next(old for obj, name, old, _ in active.values() if obj is grpc.aio and name == "insecure_channel")
    original_apply = fake_module._apply_patch

    def fail_during_install(obj, name, replacement):
        original_apply(obj, name, replacement)
        if obj is grpc.aio and name == "insecure_channel":
            raise RuntimeError("injected install failure")

    monkeypatch.setattr(fake_module, "_apply_patch", fail_during_install)
    with pytest.raises(RuntimeError, match="injected install failure"):
        conftest.pytest_configure(None)
    assert google.auth._default.default is original_default
    assert genai_client.Client is original_client
    assert qdrant_impl.QdrantClient is original_qdrant
    assert tasks_impl.CloudTasksClient is original_tasks
    assert sparse_text_embedding.SparseTextEmbedding is original_sparse
    assert httpx.Client.send is original_httpx_send
    import requests
    assert requests.Session.send is original_requests_send
    assert grpc.aio.insecure_channel is original_grpc_aio
    assert conftest._saved_sdk_boundaries is None

    monkeypatch.setattr(fake_module, "_apply_patch", original_apply)
    conftest.pytest_configure(None)
    active = conftest._saved_sdk_boundaries
    assert active is not None
    from core.integrations.qdrant import client as qdrant_module
    from core.retrieval import search
    qdrant_cache_before = qdrant_module._client
    sparse_cache_before = search._bm25_model
    fake_module.restore(active)
    assert qdrant_module._client is qdrant_cache_before
    assert search._bm25_model is sparse_cache_before
    fake_module.restore(active)
    assert qdrant_module._client is qdrant_cache_before
    assert search._bm25_model is sparse_cache_before
    assert google.auth._default.default is original_default
    assert genai_client.Client is original_client
    assert qdrant_impl.QdrantClient is original_qdrant
    assert tasks_impl.CloudTasksClient is original_tasks
    assert sparse_text_embedding.SparseTextEmbedding is original_sparse
    conftest.pytest_configure(None)
    assert conftest._saved_sdk_boundaries is not None

