"""Durable resource allocation tests for external effect resources."""

from datetime import datetime, timezone

import pytest
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from core.integrations.firestore.effects import (
    AllocationConflict,
    allocate_effect_resource,
    get_effect_allocation,
)
from core.integrations.firestore.operations import claim_operation, create_operation
from core.integrations.firestore.mutations import dispatch_mutation
from core.operations.effects import allocate_fact_id, allocate_folder_id
from core.operations.models import OperationOrigin
from core.indexing import user_facts


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _operation(client):
    create_operation(
        client, operation_id="resource-allocation", origin=OperationOrigin.MCP,
        caller_binding="caller", immutable_input_digest="sha256:input",
        immutable_input_ref="input-ref", now=NOW,
    )
    lease = claim_operation(client, "resource-allocation", owner="worker", now=NOW, lease_seconds=60)
    assert lease is not None
    return lease


def test_allocation_is_reused_without_regenerating_and_identity_change_fails(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    generated = []
    kwargs = dict(
        client=client, operation_id="resource-allocation", effect_id="fact-effect",
        intent_digest="sha256:" + "a" * 64, owner="worker", fence=lease.fence, now=NOW,
    )
    first = allocate_fact_id(**kwargs, id_factory=lambda: generated.append("first") or "6f11649d-013d-4d97-8a71-1dc816cb880f")
    replay = allocate_fact_id(**kwargs, id_factory=lambda: generated.append("second") or "4c5cda69-4a6c-4db6-b221-1c6b5e56fb5e")
    assert first == replay == "6f11649d-013d-4d97-8a71-1dc816cb880f"
    assert generated == ["first"]
    with pytest.raises(AllocationConflict):
        allocate_fact_id(**{**kwargs, "intent_digest": "sha256:" + "b" * 64}, id_factory=lambda: generated.append("bad"))
    assert generated == ["first"]
    record = get_effect_allocation(client, "resource-allocation", "fact-effect")
    assert record["resource_id"] == first
    assert "text" not in record and "command" not in record


def test_folder_allocation_exposes_stable_intent_marker(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    result = allocate_folder_id(
        client, operation_id="resource-allocation", effect_id="folder-effect",
        intent_digest="sha256:" + "c" * 64, owner="worker", fence=lease.fence, now=NOW,
        id_factory=lambda: "generated-drive-folder-id",
    )
    assert result["resource_id"] == "generated-drive-folder-id"
    assert result["intent_marker"].startswith("v1:")


def test_transactional_competing_candidate_returns_persisted_winner(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    args = dict(
        client=client, operation_id="resource-allocation", effect_id="folder-effect",
        kind="drive_folder", intent_digest="sha256:" + "d" * 64,
        owner="worker", fence=lease.fence, now=NOW,
    )
    winner = allocate_effect_resource(**args, candidate="first-drive-id")
    competing = allocate_effect_resource(**args, candidate="second-drive-id")
    assert competing["resource_id"] == winner["resource_id"] == "first-drive-id"


def test_overlapping_allocations_choose_one_persisted_id(firestore_fake):
    from tests.fixtures.firestore_fake import FakeTransaction

    client = firestore_fake.client_class()
    lease = _operation(client)
    barrier = threading.Barrier(2)
    factories_barrier = threading.Barrier(2)
    factories_used = []
    original_read = FakeTransaction._read

    def synchronized_candidate(transaction, reference):
        snapshot = original_read(transaction, reference)
        if reference.path.endswith("/allocations/resource") and not snapshot.exists:
            barrier.wait(timeout=5)
        return snapshot

    FakeTransaction._read = synchronized_candidate
    try:
        candidates = iter((
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        ))

        def allocate():
            candidate = next(candidates)
            def factory():
                factories_barrier.wait(timeout=5)
                factories_used.append(candidate)
                return candidate
            return allocate_fact_id(
                client, operation_id="resource-allocation", effect_id="racing-effect",
                intent_digest="sha256:" + "e" * 64, owner="worker", fence=lease.fence,
                now=NOW, id_factory=factory,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: allocate(), range(2)))
    finally:
        FakeTransaction._read = original_read
    assert len(set(factories_used)) == 2
    assert results[0] == results[1]


def test_after_commit_failure_reuses_persisted_folder_id_on_restart(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    allocation_path = "dialog_operations/resource-allocation/effects/lost-effect/allocations/resource"
    firestore_fake.fail_next(
        "commit", path=allocation_path, phase="after", exception=RuntimeError("private storage detail"),
    )
    generated = []
    kwargs = dict(
        client=client, operation_id="resource-allocation", effect_id="lost-effect",
        intent_digest="sha256:" + "f" * 64, owner="worker", fence=lease.fence, now=NOW,
    )
    with pytest.raises(RuntimeError, match="resource ID allocation is unresolved"):
        allocate_folder_id(**kwargs, id_factory=lambda: generated.append(1) or "lost-drive-id")
    replay = allocate_folder_id(**kwargs, id_factory=lambda: generated.append(2) or "wrong-drive-id")
    assert replay["resource_id"] == "lost-drive-id"
    assert generated == [1]


def test_new_allocation_rejects_changed_kind_docs_kind_and_stale_lease_before_factory(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    generated = []
    common = dict(
        client=client, operation_id="resource-allocation", effect_id="guarded-effect",
        intent_digest="sha256:" + "1" * 64, owner="worker", fence=lease.fence, now=NOW,
    )
    allocate_fact_id(**common, id_factory=lambda: "33333333-3333-4333-8333-333333333333")
    with pytest.raises(AllocationConflict):
        allocate_folder_id(**common, id_factory=lambda: generated.append("kind"))
    with pytest.raises(ValueError, match="invalid effect resource allocation"):
        allocate_effect_resource(
            client, operation_id="resource-allocation", effect_id="docs-effect", kind="native_docs",
            intent_digest="sha256:" + "2" * 64, owner="worker", fence=lease.fence, now=NOW,
            candidate="docs-id",
        )
    with pytest.raises(AllocationConflict):
        allocate_fact_id(
            client, operation_id="resource-allocation", effect_id="stale-effect",
            intent_digest="sha256:" + "3" * 64, owner="worker", fence=lease.fence,
            now=NOW + timedelta(minutes=2), id_factory=lambda: generated.append("stale"),
        )
    assert generated == []


def test_allocation_subdocument_survives_t020_parent_effect_writes(firestore_fake):
    client = firestore_fake.client_class()
    lease = _operation(client)
    point_id = allocate_fact_id(
        client, operation_id="resource-allocation", effect_id="shared-effect",
        intent_digest="sha256:" + "4" * 64, owner="worker", fence=lease.fence, now=NOW,
        id_factory=lambda: "44444444-4444-4444-8444-444444444444",
    )
    dispatch_mutation(
        client, operation_id="resource-allocation", effect_id="shared-effect", owner="worker",
        fence=lease.fence, kind="provider_call", resources=("provider:synthetic",),
        payload_digest="sha256:mutation", now=NOW, mutation_callback=lambda: "done",
    )
    assert get_effect_allocation(client, "resource-allocation", "shared-effect")["resource_id"] == point_id


def test_allocated_fact_ids_exist_before_upsert_and_are_stable_per_effect(
    firestore_fake, sdk_fakes, monkeypatch,
):
    client = firestore_fake.client_class()
    lease = _operation(client)
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc: {"status": "synced"})
    monkeypatch.setattr(user_facts, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(user_facts, "embed", lambda _texts: [[0.25, 0.5]])
    generated = iter((
        "55555555-5555-4555-8555-555555555555",
        "66666666-6666-4666-8666-666666666666",
    ))

    def add(effect_id):
        point_id = allocate_fact_id(
            client, operation_id="resource-allocation", effect_id=effect_id,
            intent_digest="sha256:" + "5" * 64, owner="worker", fence=lease.fence,
            now=NOW, id_factory=lambda: next(generated),
        )
        assert get_effect_allocation(client, "resource-allocation", effect_id)["resource_id"] == point_id
        user_facts.add_user_fact("doc-id", "Repeated fact", point_id=point_id)
        return point_id

    first = add("fact-effect-a")
    replay = add("fact-effect-a")
    distinct = add("fact-effect-b")
    points = [
        point for event in sdk_fakes.trace
        if event["boundary"] == "qdrant" and event["operation"] == "upsert"
        for point in event["arguments"]["points"]
    ]
    assert [point.id for point in points] == [first, replay, distinct]
    assert first == replay and first != distinct
