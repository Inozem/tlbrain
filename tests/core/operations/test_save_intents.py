"""Focused offline tests for durable save-intent identity and checkpoints."""

from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.operations import claim_operation, create_operation, get_operation
from core.integrations.firestore.save_intents import (
    create_save_intent,
    dispatch_save_intent,
    get_save_intent,
    reconcile_save_intent,
    update_save_intent_checkpoints,
)
from core.operations.models import OperationOrigin, OperationResult


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _operation(client, operation_id: str, *, save_intent_id: str | None = None) -> None:
    create_operation(
        client, operation_id=operation_id, origin=OperationOrigin.MCP,
        caller_binding="caller", immutable_input_digest="sha256:input-a",
        immutable_input_ref="input-ref", provider="provider-a", title="Meeting",
        save_intent_id=save_intent_id, now=NOW,
    )


def _intent(client, intent_id: str, operation_id: str):
    return create_save_intent(
        client, intent_id, operation_id=operation_id,
        desired_content_hash="sha256:content", desired_parent_id_hash="sha256:parent", now=NOW,
    )


def test_intent_create_is_idempotent_but_rejects_conflicting_identity(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    first = _intent(client, "intent-a", "operation-a")
    replay = _intent(client, "intent-a", "operation-a")
    assert replay == first
    with pytest.raises(ValueError, match="hash conflicts|different immutable identity"):
        create_save_intent(
            client, "intent-a", operation_id="operation-a", desired_content_hash="sha256:other",
            desired_parent_id_hash="sha256:parent", now=NOW,
        )
    _operation(client, "operation-b")
    distinct = _intent(client, "intent-b", "operation-b")
    assert distinct.save_intent_id != first.save_intent_id
    assert distinct.desired_content_hash == first.desired_content_hash


def test_existing_operation_hashes_are_immutable_and_matching_prebound_link_is_filled(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-prebound", save_intent_id="intent-prebound")
    created = _intent(client, "intent-prebound", "operation-prebound")
    operation = get_operation(client, "operation-prebound")
    assert operation.save_intent_id == "intent-prebound"
    assert operation.desired_content_hash == created.desired_content_hash
    assert operation.desired_parent_id_hash == created.desired_parent_id_hash

    _operation(client, "operation-conflict", save_intent_id="intent-prebound")
    client.collection("dialog_operations").document("operation-conflict").update({
        "desired_content_hash": "sha256:changed",
    })
    with pytest.raises(ValueError, match="content hash conflicts"):
        _intent(client, "intent-prebound", "operation-conflict")


def test_unknown_create_can_be_reconciled_under_retry_operation_lease(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    first_lease = claim_operation(client, "operation-a", owner="first", now=NOW, lease_seconds=30)
    creates = []

    def lost_create():
        creates.append("create")
        raise RuntimeError("unknown create result")

    with pytest.raises(RuntimeError, match="unknown create result"):
        dispatch_save_intent(
            client, "intent-a", operation_id="operation-a", owner="first", fence=first_lease.fence,
            now=NOW, create_callback=lost_create,
        )
    _operation(client, "operation-b", save_intent_id="intent-a")
    _intent(client, "intent-a", "operation-b")
    retry_lease = claim_operation(client, "operation-b", owner="retry", now=NOW, lease_seconds=30)
    assert retry_lease is not None
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-b", owner="retry", fence=retry_lease.fence,
        now=NOW, create_callback=lambda: (creates.append("second-create") or OperationResult(document_id="wrong")),
    ) is None
    reconciled = reconcile_save_intent(
        client, "intent-a", owner="retry", fence=retry_lease.fence, now=NOW,
        external_result=OperationResult(document_id="found-doc"), operation_id="operation-b",
    )
    assert reconciled.state.value == "confirmed"
    assert reconciled.doc_id == "found-doc"
    assert reconciled.dispatch_operation_id == "operation-a"
    assert creates == ["create"]


def test_stale_dispatch_does_not_auto_create_or_bind_missing_intent(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-stale")
    lease = claim_operation(client, "operation-stale", owner="old", now=NOW, lease_seconds=2)
    assert lease is not None
    expired_at = NOW + timedelta(seconds=3)
    assert dispatch_save_intent(
        client, "intent-never-created", operation_id="operation-stale", owner="old",
        fence=lease.fence, now=expired_at,
        create_callback=lambda: pytest.fail("expired worker dispatched"),
    ) is None
    assert get_save_intent(client, "intent-never-created") is None
    operation = get_operation(client, "operation-stale")
    assert operation.save_intent_id is None
    assert operation.desired_content_hash is None
    assert operation.desired_parent_id_hash is None


def test_checkpoints_require_confirmed_result_with_document_id(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    lease = claim_operation(client, "operation-a", owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    assert update_save_intent_checkpoints(
        client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
        now=NOW, fill_checkpoint_ref="fill/too-early",
    ) is None
    assert get_save_intent(client, "intent-a").fill_checkpoint_ref is None


def test_missing_document_id_never_confirms_or_reopens_create_gate(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    lease = claim_operation(client, "operation-a", owner="worker", now=NOW, lease_seconds=30)
    calls = []
    with pytest.raises(ValueError, match="requires a document ID"):
        dispatch_save_intent(
            client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
            now=NOW, create_callback=lambda: (calls.append("create") or OperationResult()),
        )
    assert get_save_intent(client, "intent-a").state.value == "dispatched"
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
        now=NOW, create_callback=lambda: calls.append("second-create"),
    ) is None
    assert calls == ["create"]

    # An unknown result remains unknown after invalid reconciliation input.
    _operation(client, "operation-unknown")
    _intent(client, "intent-unknown", "operation-unknown")
    unknown_lease = claim_operation(client, "operation-unknown", owner="worker2", now=NOW, lease_seconds=30)
    with pytest.raises(RuntimeError, match="unknown"):
        dispatch_save_intent(
            client, "intent-unknown", operation_id="operation-unknown", owner="worker2",
            fence=unknown_lease.fence, now=NOW,
            create_callback=lambda: (_ for _ in ()).throw(RuntimeError("unknown")),
        )
    with pytest.raises(ValueError, match="requires a document ID"):
        reconcile_save_intent(
            client, "intent-unknown", owner="worker2", fence=unknown_lease.fence,
            now=NOW, external_result=OperationResult(),
        )
    assert get_save_intent(client, "intent-unknown").state.value == "unknown"
    reconciled = reconcile_save_intent(
        client, "intent-unknown", owner="worker2", fence=unknown_lease.fence,
        now=NOW, external_result=OperationResult(document_id="found"),
    )
    assert reconciled.doc_id == "found"


def test_retry_operation_reuses_confirmed_intent_without_resetting_gate(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    lease = claim_operation(client, "operation-a", owner="worker", now=NOW, lease_seconds=30)
    result = OperationResult(document_id="doc-a", title="Meeting")
    calls = []
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
        now=NOW, create_callback=lambda: (calls.append(1) or result),
    ) == result
    _operation(client, "operation-retry", save_intent_id="intent-a")
    retry = _intent(client, "intent-a", "operation-retry")
    assert retry.state.value == "confirmed"
    assert retry.confirmed_result == result
    retry_lease = claim_operation(client, "operation-retry", owner="retry-worker", now=NOW, lease_seconds=30)
    assert retry_lease is not None
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-retry", owner="retry-worker", fence=retry_lease.fence,
        now=NOW, create_callback=lambda: (calls.append(2) or result),
    ) == result
    assert calls == [1]


def test_stale_fence_cannot_dispatch_reconcile_or_write_checkpoints(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    lease = claim_operation(client, "operation-a", owner="old", now=NOW, lease_seconds=2)
    assert lease is not None
    later = NOW + timedelta(seconds=3)
    current = claim_operation(client, "operation-a", owner="new", now=later, lease_seconds=30)
    assert current is not None
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-a", owner="old", fence=lease.fence,
        now=later, create_callback=lambda: pytest.fail("stale worker dispatched"),
    ) is None
    # Reserve under the current fence, then verify stale reconciliation cannot mutate it.
    with pytest.raises(RuntimeError, match="lost"):
        dispatch_save_intent(
            client, "intent-a", operation_id="operation-a", owner="new", fence=current.fence,
            now=later, create_callback=lambda: (_ for _ in ()).throw(RuntimeError("lost")),
        )
    assert get_save_intent(client, "intent-a").state.value == "unknown"
    assert reconcile_save_intent(
        client, "intent-a", owner="old", fence=lease.fence, now=later,
        external_result=OperationResult(document_id="doc-a"),
    ) is None
    assert update_save_intent_checkpoints(
        client, "intent-a", operation_id="operation-a", owner="old", fence=lease.fence,
        now=later, fill_checkpoint_ref="fill/1",
    ) is None
    saved = get_save_intent(client, "intent-a")
    assert saved.state.value == "unknown"
    assert saved.fill_checkpoint_ref is None


def test_checkpoints_preserve_confirmed_create_metadata(firestore_fake):
    client = firestore_fake.client_class()
    _operation(client, "operation-a")
    _intent(client, "intent-a", "operation-a")
    lease = claim_operation(client, "operation-a", owner="worker", now=NOW, lease_seconds=30)
    result = OperationResult(document_id="doc-a")
    assert dispatch_save_intent(
        client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
        now=NOW, create_callback=lambda: result,
    ) == result
    saved = update_save_intent_checkpoints(
        client, "intent-a", operation_id="operation-a", owner="worker", fence=lease.fence,
        now=NOW, fill_checkpoint_ref="fill/1", index_checkpoint_ref="index/1",
    )
    assert saved.state.value == "confirmed"
    assert saved.confirmed_result == result
    assert saved.doc_id == "doc-a"
    assert saved.fill_checkpoint_ref == "fill/1"
    assert saved.index_checkpoint_ref == "index/1"
    with pytest.raises(ValueError, match="contradictory"):
        reconcile_save_intent(
            client, "intent-a", owner="worker", fence=lease.fence, now=NOW,
            external_result=OperationResult(document_id="different-doc"),
        )
