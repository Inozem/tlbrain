"""Offline shared-guard tests for durable external mutations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from google.api_core.exceptions import Aborted

from core.integrations.firestore.operations import claim_operation, create_operation
from core.integrations.firestore.mutations import (
    checkpoint_mutation,
    dispatch_mutation,
    finalize_document,
    get_resource_guard,
    publish_index_completion,
    reconcile_mutation,
    release_mutation,
)
from core.operations.models import OperationOrigin


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
OPERATION_ID = "synthetic-mutation-operation"


def _setup(client):
    create_operation(
        client, operation_id=OPERATION_ID, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:mutation-input",
        immutable_input_ref="mutation-input-ref", now=NOW,
    )
    lease = claim_operation(client, OPERATION_ID, owner="worker", now=NOW, lease_seconds=30)
    assert lease is not None
    return lease


@pytest.mark.parametrize("effect_kind,resource", [
    ("content_update", "document:doc-a"),
    ("parent_move", "document:doc-a"),
    ("provider_call", "provider:synthetic"),
])
def test_resource_epoch_and_unsettled_effect_precede_mutation_callback(firestore_fake, effect_kind, resource):
    client = firestore_fake.client_class()
    lease = _setup(client)
    observations = []

    def mutation_callback():
        guard = get_resource_guard(client, resource)
        observations.append((guard.resource_epoch, guard.unsettled_effect_ids))
        return "synthetic-mutation-result"

    assert dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id=f"effect-{effect_kind}",
        owner="worker", fence=lease.fence, kind=effect_kind, resources=(resource,),
        payload_digest="sha256:desired-state", now=NOW, mutation_callback=mutation_callback,
    ) == "synthetic-mutation-result"
    assert observations[0][0] == 1
    assert observations[0][1] == (f"effect-{effect_kind}",)


def test_document_and_subtree_resources_share_guards_with_finalize_and_index_publication(firestore_fake):
    client = firestore_fake.client_class()
    lease = _setup(client)
    resources = ("document:root", "document:child-a", "document:child-b")
    callback_events = []
    publication_results = []

    def mutate_subtree():
        callback_events.append("subtree-provider-mutation")
        assert all(
            get_resource_guard(client, resource).unsettled_effect_ids == ("subtree-effect",)
            for resource in resources
        )
        publication_results.append(finalize_document(
            client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
            resources=("document:child-a",), now=NOW,
        ))
        publication_results.append(publish_index_completion(
            client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
            resources=("document:child-b",), now=NOW,
        ))
        return "done"

    dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="subtree-effect", owner="worker", fence=lease.fence,
        kind="parent_move", resources=resources, payload_digest="sha256:new-parent", now=NOW,
        mutation_callback=mutate_subtree,
    )
    assert callback_events == ["subtree-provider-mutation"]
    assert publication_results == [False, False]


def test_unknown_effect_and_stale_lease_block_finalize_and_redispatch_until_reconciled(firestore_fake):
    client = firestore_fake.client_class()
    lease = _setup(client)
    resource = "document:doc-unknown"
    physical_calls = []

    def uncertain_mutation():
        physical_calls.append("provider-call")
        raise TimeoutError("synthetic unknown result")

    with pytest.raises(TimeoutError, match="synthetic unknown"):
        dispatch_mutation(
            client, operation_id=OPERATION_ID, effect_id="unknown-effect", owner="worker", fence=lease.fence,
            kind="content_update", resources=(resource,), payload_digest="sha256:new-content", now=NOW,
            mutation_callback=uncertain_mutation,
        )
    assert get_resource_guard(client, resource).unsettled_effect_ids == ("unknown-effect",)
    assert finalize_document(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        resources=(resource,), now=NOW,
    ) is False
    assert dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="unknown-effect", owner="worker", fence=lease.fence,
        kind="content_update", resources=(resource,), payload_digest="sha256:new-content", now=NOW,
        mutation_callback=uncertain_mutation,
    ) is None
    assert physical_calls == ["provider-call"]

    reconciled = reconcile_mutation(
        client, effect_id="unknown-effect", owner="worker", fence=lease.fence,
        now=NOW, confirmed=False,
    )
    assert reconciled.state.value == "failed"
    assert get_resource_guard(client, resource).unsettled_effect_ids == ()


def test_expired_lease_cannot_finalize_or_redispatch_after_reclaim(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    create_operation(
        client, operation_id=OPERATION_ID, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:mutation-input",
        immutable_input_ref="mutation-input-ref", now=fake_clock.now(),
    )
    old = claim_operation(client, OPERATION_ID, owner="old-worker", now=fake_clock.now(), lease_seconds=5)
    assert old is not None
    resource = "document:stale-lease"
    calls = []

    def uncertain_mutation():
        calls.append("first-dispatch")
        raise TimeoutError("synthetic uncertain write")

    with pytest.raises(TimeoutError):
        dispatch_mutation(
            client, operation_id=OPERATION_ID, effect_id="stale-effect", owner="old-worker", fence=old.fence,
            kind="content_update", resources=(resource,), payload_digest="sha256:stale", now=fake_clock.now(),
            mutation_callback=uncertain_mutation,
        )
    fake_clock.advance(seconds=6)
    current = claim_operation(client, OPERATION_ID, owner="new-worker", now=fake_clock.now(), lease_seconds=30)
    assert current is not None and current.fence > old.fence
    assert finalize_document(
        client, operation_id=OPERATION_ID, owner="old-worker", fence=old.fence,
        resources=(resource,), now=fake_clock.now(),
    ) is False
    assert dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="stale-effect", owner="old-worker", fence=old.fence,
        kind="content_update", resources=(resource,), payload_digest="sha256:stale", now=fake_clock.now(),
        mutation_callback=uncertain_mutation,
    ) is None
    assert calls == ["first-dispatch"]


def test_checkpoint_and_release_require_current_owner_and_fence(firestore_fake):
    client = firestore_fake.client_class()
    lease = _setup(client)
    assert checkpoint_mutation(
        client, operation_id=OPERATION_ID, owner="intruder", fence=lease.fence,
        checkpoint_ref="synthetic-checkpoint", now=NOW,
    ) is False
    assert release_mutation(
        client, operation_id=OPERATION_ID, owner="intruder", fence=lease.fence, now=NOW,
    ) is False
    assert checkpoint_mutation(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        checkpoint_ref="synthetic-checkpoint", now=NOW,
    ) is True
    assert release_mutation(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence, now=NOW,
    ) is True
    assert checkpoint_mutation(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        checkpoint_ref="stale-checkpoint", now=NOW + timedelta(seconds=1),
    ) is False


def test_independent_resources_can_progress_while_other_effect_is_unsettled(firestore_fake):
    client = firestore_fake.client_class()
    lease = _setup(client)
    calls = []

    def uncertain():
        calls.append("blocked-resource")
        raise TimeoutError("synthetic unknown result")

    with pytest.raises(TimeoutError):
        dispatch_mutation(
            client, operation_id=OPERATION_ID, effect_id="effect-a", owner="worker", fence=lease.fence,
            kind="content_update", resources=("document:a",), payload_digest="sha256:a", now=NOW,
            mutation_callback=uncertain,
        )
    independent = dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="effect-b", owner="worker", fence=lease.fence,
        kind="content_update", resources=("document:b",), payload_digest="sha256:b", now=NOW,
        mutation_callback=lambda: calls.append("independent-resource") or "done",
    )
    assert independent == "done"
    assert calls == ["blocked-resource", "independent-resource"]


def test_effect_ids_are_scoped_to_the_operation(firestore_fake):
    client = firestore_fake.client_class()
    first_lease = _setup(client)
    second_id = "second-mutation-operation"
    create_operation(
        client, operation_id=second_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:second-input",
        immutable_input_ref="second-input-ref", now=NOW,
    )
    second_lease = claim_operation(client, second_id, owner="worker", now=NOW, lease_seconds=30)
    assert second_lease is not None

    assert dispatch_mutation(
        client, operation_id=OPERATION_ID, effect_id="same-effect-id", owner="worker",
        fence=first_lease.fence, kind="content_update", resources=("document:first",),
        payload_digest="sha256:first", now=NOW, mutation_callback=lambda: "first-result",
    ) == "first-result"
    assert dispatch_mutation(
        client, operation_id=second_id, effect_id="same-effect-id", owner="worker",
        fence=second_lease.fence, kind="content_update", resources=("document:second",),
        payload_digest="sha256:second", now=NOW, mutation_callback=lambda: "second-result",
    ) == "second-result"

    for operation_id, expected in ((OPERATION_ID, "first-result"), (second_id, "second-result")):
        effect = client.collection("dialog_operations").document(operation_id).collection("effects").document(
            "same-effect-id",
        ).get()
        assert effect.exists and effect.to_dict()["result"] == expected


def test_publications_write_shared_resource_records_and_serialize_transactions(firestore_fake):
    client = firestore_fake.client_class()
    lease = _setup(client)
    resource = "document:published"
    guard_ref = client.collection("resource_mutations").document(hashlib.sha256(resource.encode()).hexdigest())

    concurrent = client.transaction()
    concurrent._begin()
    guard_ref.get(transaction=concurrent)

    assert finalize_document(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        resources=(resource,), now=NOW,
    ) is True
    with pytest.raises(Aborted):
        concurrent._commit()

    assert publish_index_completion(
        client, operation_id=OPERATION_ID, owner="worker", fence=lease.fence,
        resources=(resource,), now=NOW,
    ) is True
    guard = guard_ref.get().to_dict()
    assert guard["document_finalized"]["operation_id"] == OPERATION_ID
    assert guard["index_completion_published"]["operation_id"] == OPERATION_ID
    assert guard["mutation_state"] == "index_completion_published"
