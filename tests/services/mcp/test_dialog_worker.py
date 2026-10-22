from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from core.integrations.firestore.outbox import OutboxState, OutboxTargetKind
from core.operations.models import OperationOrigin, OperationState

from services.mcp.app import dialog_worker, main, oauth
from tests.services.mcp.test_contract import _request


TASK = {
    "schema_version": 1,
    "outbox_id": "recovery-1",
    "target_kind": "mcp_save_recovery",
    "generation": 2,
    "operation_id": "operation-1",
    "save_intent_id": "save-1",
}


def test_envelope_is_strict_and_rejects_text_and_unknown_fields():
    assert dialog_worker.validate_task_envelope(TASK) == TASK
    for invalid in (
        {**TASK, "text": "do this"},
        {**TASK, "tool_arguments": {}},
        {**TASK, "generation": True},
        {**TASK, "schema_version": 1.0},
        {**TASK, "operation_id": "bad/id"},
    ):
        with pytest.raises(dialog_worker.InvalidTask):
            dialog_worker.validate_task_envelope(invalid)


def test_worker_checks_outbox_generation_target_and_operation(monkeypatch):
    record = SimpleNamespace(
        target_kind=OutboxTargetKind.MCP_SAVE_RECOVERY, generation=2,
        operation_id="operation-1", state=OutboxState.PENDING,
    )
    monkeypatch.setattr(dialog_worker.outbox_repository, "get_outbox_record", lambda *_: record)
    monkeypatch.setattr(dialog_worker, "get_operation", lambda *_: None)
    assert dialog_worker.process_dialog_operation(object(), TASK).status == "stale"
    record.generation = 1
    assert dialog_worker.process_dialog_operation(object(), TASK).status == "stale"


def test_worker_noops_terminal_and_rejects_missing_completion_binding(monkeypatch):
    record = SimpleNamespace(
        target_kind=OutboxTargetKind.MCP_SAVE_RECOVERY, generation=2,
        operation_id="operation-1", save_intent_id="save-1", state=OutboxState.DELIVERED,
    )
    monkeypatch.setattr(dialog_worker.outbox_repository, "get_outbox_record", lambda *_: record)
    operation = SimpleNamespace(
        operation_id="operation-1", origin_operation_ref=None,
        origin=OperationOrigin.MCP, save_intent_id="save-1", state=OperationState.COMPLETED,
    )
    monkeypatch.setattr(dialog_worker, "get_operation", lambda *_: operation)
    assert dialog_worker.process_dialog_operation(object(), TASK).status == "terminal"

    operation.state = OperationState.RUNNING
    operation.immutable_input_digest = "a" * 64
    with pytest.raises(dialog_worker.RecoveryBindingUnavailable):
        dialog_worker.process_dialog_operation(object(), TASK)


def test_worker_advances_only_matching_operation_through_existing_checkpoints(monkeypatch):
    record = SimpleNamespace(
        target_kind=OutboxTargetKind.MCP_SAVE_RECOVERY, generation=2,
        operation_id="operation-1", save_intent_id="save-1", state=OutboxState.DELIVERED,
    )
    operation = SimpleNamespace(
        operation_id="operation-1", origin=OperationOrigin.MCP,
        save_intent_id="save-1", origin_operation_ref="mcp_sessions/session-hash/requests/request-hash",
        immutable_input_digest="a" * 64, state=OperationState.RUNNING,
    )
    monkeypatch.setattr(dialog_worker.outbox_repository, "get_outbox_record", lambda *_: record)
    monkeypatch.setattr(dialog_worker, "get_operation", lambda *_: operation)
    recovery_binding = {
        "context_id": operation.origin_operation_ref,
        "draft_id": "request-hash",
        "manifest_digest": operation.immutable_input_digest,
    }
    monkeypatch.setattr(
        dialog_worker.mcp_sessions, "get_recovery_binding",
        lambda *_args, **_kwargs: recovery_binding,
    )
    monkeypatch.setattr(dialog_worker, "claim_operation", lambda *_args, **_kwargs: SimpleNamespace(fence=7))
    monkeypatch.setattr(dialog_worker, "release_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dialog_worker.folder_repository, "get_all_folders", lambda *_: {})
    monkeypatch.setattr(dialog_worker, "get_root_folder_id", lambda: "root")
    advanced = []
    from services.mcp.app.mcp import handler
    monkeypatch.setattr(handler, "_advance_add_dialog_save", lambda *_args: advanced.append("advance") or SimpleNamespace(save_intent_id="save-1"))
    monkeypatch.setattr(handler, "_publish_add_dialog_index_work", lambda *_: advanced.append("publish"))
    monkeypatch.setattr(dialog_worker.dialog_completion, "finalize_save", lambda *_args, **kwargs: advanced.append(kwargs["origin_binding"]))

    assert dialog_worker.process_dialog_operation(object(), TASK).status == "processed"
    assert advanced == ["advance", "publish", recovery_binding]


def test_internal_route_bypasses_mcp_oauth_and_redacts_recovery_blocker(monkeypatch):
    monkeypatch.setattr(oauth, "request_identity", lambda *_: (_ for _ in ()).throw(AssertionError("OAuth called")))
    monkeypatch.setattr(main, "verify_google_id_token", lambda _token: SimpleNamespace(authorized=True))
    monkeypatch.setattr(main, "process_dialog_operation", lambda *_: (_ for _ in ()).throw(
        dialog_worker.RecoveryBindingUnavailable("private details")
    ))
    response = _request("post", "/internal/dialog-operations", json=TASK,
                        headers={"Authorization": "Bearer synthetic-token"})
    assert response.status_code == 503
    assert response.json() == {"error": "Recovery unavailable"}
    assert "private details" not in response.text


def test_internal_route_rejects_invalid_body_after_auth(monkeypatch):
    monkeypatch.setattr(main, "verify_google_id_token", lambda _token: SimpleNamespace(authorized=True))
    response = _request("post", "/internal/dialog-operations", json={**TASK, "dialog_text": "secret"},
                        headers={"Authorization": "Bearer synthetic-token"})
    assert response.status_code == 400
    assert "secret" not in response.text


@pytest.mark.parametrize(
    ("identity", "expected_status"),
    [(None, 401), (SimpleNamespace(authorized=False), 403)],
)
def test_internal_route_maps_invalid_and_wrong_identity_without_details(monkeypatch, identity, expected_status):
    monkeypatch.setattr(main, "verify_google_id_token", lambda _token: identity)
    response = _request("post", "/internal/dialog-operations", json=TASK,
                        headers={"Authorization": "Bearer synthetic-token"})
    assert response.status_code == expected_status
    assert "synthetic-token" not in response.text


def test_internal_route_rejects_missing_bearer(monkeypatch):
    called = []
    monkeypatch.setattr(main, "verify_google_id_token", lambda *_: called.append(True))
    response = _request("post", "/internal/dialog-operations", json=TASK)
    assert response.status_code == 401
    assert called == []


def test_add_dialog_readiness_uses_only_configured_callback_and_preserves_health(monkeypatch):
    monkeypatch.setenv("INTERNAL_DIALOG_OPERATIONS_URL", "https://configured.test/internal/dialog-operations")
    assert main._dialog_recovery_ready()
    monkeypatch.setenv("INTERNAL_DIALOG_OPERATIONS_URL", "http://untrusted.test/internal/dialog-operations")
    assert not main._dialog_recovery_ready()
    monkeypatch.setenv("INTERNAL_DIALOG_OPERATIONS_URL", "https://configured.test/internal/dialog-operations")
    for names in (("DIALOG_OPERATIONS_QUEUE",),
                  ("CLOUD_TASKS_OIDC_AUDIENCE", "TASK_OIDC_AUDIENCE", "OIDC_AUDIENCE"),
                  ("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "TASK_OIDC_SERVICE_ACCOUNT_EMAIL",
                   "CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL", "TASK_SERVICE_ACCOUNT_EMAIL",
                   "OIDC_SERVICE_ACCOUNT_EMAIL")):
        original = {name: os.environ.get(name) for name in names}
        for name in names:
            monkeypatch.delenv(name, raising=False)
        assert not main._dialog_recovery_ready()
        for name, value in original.items():
            if value is not None:
                monkeypatch.setenv(name, value)
    assert _request("get", "/").json() == {"status": "ok"}


def test_add_dialog_readiness_blocks_before_handler_but_other_tools_remain(monkeypatch):
    monkeypatch.setattr(oauth, "request_identity", lambda *_: SimpleNamespace(caller_binding="caller"))
    monkeypatch.setattr(main, "_dialog_recovery_ready", lambda: False)
    called = []
    async def handle(body):
        called.append(body)
        return {}
    monkeypatch.setattr(main, "handle_mcp_request", handle)
    add = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add_dialog"}}
    response = _request("post", "/mcp", json=add)
    assert response.status_code == 503
    assert called == []
    other = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "query"}}
    assert _request("post", "/mcp", json=other).status_code == 200
    assert called == [other]
