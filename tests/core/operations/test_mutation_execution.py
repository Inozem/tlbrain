from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.integrations.firestore.mutations import dispatch_mutation, get_resource_guard
from core.integrations.firestore.operations import claim_operation, create_operation
from core.operations.models import OperationOrigin
from core.operations.mutations import (
    MutationConflict,
    MutationNotSent,
    MutationOutcomeUnknown,
    MutationPolicy,
    MutationSnapshot,
    execute_mutation,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
OPERATION_ID = "mutation-execution-operation"


def _lease(client, *, now=NOW, owner="worker", lease_seconds=30, operation_id=OPERATION_ID):
    create_operation(
        client, operation_id=operation_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:input",
        immutable_input_ref="input-ref", now=NOW,
    )
    lease = claim_operation(
        client, operation_id, owner=owner, now=now, lease_seconds=lease_seconds,
    )
    assert lease is not None
    return lease


def _execute(client, lease, **overrides):
    snapshots = iter((MutationSnapshot("sha256:old"), MutationSnapshot("sha256:desired")))
    arguments = {
        "operation_id": OPERATION_ID,
        "effect_id": "effect-a",
        "owner": "worker",
        "fence": lease.fence,
        "kind": "content_update",
        "resources": ("document:a",),
        "payload_digest": "sha256:payload",
        "desired_state_digest": "sha256:desired",
        "expected_pre_state": "sha256:old",
        "expected_revision": None,
        "policy": MutationPolicy.DRIVE_METADATA,
        "now": NOW,
        "read_back": lambda: next(snapshots),
        "mutation_callback": lambda revision: "updated",
    }
    arguments.update(overrides)
    return execute_mutation(client, **arguments)


def _install_firestore_map_name_boundary(monkeypatch):
    """Reject Firestore's reserved ``__.*__`` map field names at writes."""
    from tests.fixtures.firestore_fake import FakeTransaction

    original_set = FakeTransaction.set

    def validate_map(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
                    raise ValueError("Firestore map field names may not match __.*__")
                validate_map(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                validate_map(child)

    def strict_set(self, reference, document_data, merge=False):
        validate_map(document_data)
        return original_set(self, reference, document_data, merge)

    monkeypatch.setattr(FakeTransaction, "set", strict_set)


def test_confirmed_replay_reuses_result_including_none_without_provider_calls(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    calls = []
    args = {
        "read_back": lambda: calls.append("read") or MutationSnapshot(
            "sha256:old" if calls.count("read") == 1 else "sha256:desired"
        ),
        "mutation_callback": lambda revision: calls.append(("write", revision)) or None,
    }

    assert _execute(client, lease, **args) is None
    assert _execute(client, lease, **args) is None
    assert calls == ["read", ("write", None), "read"]


def test_successful_result_storage_uses_firestore_legal_map_fields(monkeypatch, firestore_fake):
    _install_firestore_map_name_boundary(monkeypatch)
    client = firestore_fake.client_class()
    lease = _lease(client)
    calls = []
    args = {
        "read_back": lambda: calls.append("read") or MutationSnapshot(
            "sha256:old" if calls.count("read") == 1 else "sha256:desired"
        ),
        "mutation_callback": lambda revision: calls.append(("write", revision)) or "folder-created",
    }

    assert _execute(client, lease, **args) == "folder-created"
    from core.integrations.firestore.mutations import get_mutation_result
    found, stored = get_mutation_result(client, OPERATION_ID, "effect-a")
    assert found and stored == {"tlbrain_mutation_result_v1": "folder-created"}
    assert _execute(client, lease, **args) == "folder-created"
    assert calls == ["read", ("write", None), "read"]


def test_successful_none_result_is_durably_confirmed_and_replayed(monkeypatch, firestore_fake):
    _install_firestore_map_name_boundary(monkeypatch)
    client = firestore_fake.client_class()
    lease = _lease(client)
    calls = []
    args = {
        "read_back": lambda: calls.append("read") or MutationSnapshot(
            "sha256:old" if calls.count("read") == 1 else "sha256:desired"
        ),
        "mutation_callback": lambda revision: calls.append(("write", revision)) or None,
    }

    assert _execute(client, lease, **args) is None
    from core.integrations.firestore.mutations import get_mutation_result
    found, stored = get_mutation_result(client, OPERATION_ID, "effect-a")
    assert found and stored == {"tlbrain_mutation_result_v1": None}
    assert _execute(client, lease, **args) is None
    assert calls == ["read", ("write", None), "read"]


def test_confirmed_legacy_result_envelope_remains_readable_with_exact_match(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    _execute(client, lease)
    effect_ref = client.collection("dialog_operations").document(OPERATION_ID).collection(
        "effects",
    ).document("effect-a")

    effect_ref.update({"result": {"__tlbrain_mutation_result_v1__": "legacy-result"}})
    assert _execute(client, lease, mutation_callback=lambda _: pytest.fail("confirmed replay must not write")) == "legacy-result"

    effect_ref.update({"result": {"__tlbrain_mutation_result_v1__": "literal", "extra": True}})
    assert _execute(client, lease, mutation_callback=lambda _: pytest.fail("confirmed replay must not write")) == {
        "__tlbrain_mutation_result_v1__": "literal",
        "extra": True,
    }


def test_immutable_identity_conflict_is_rejected_before_provider_calls(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    _execute(client, lease)
    calls = []

    with pytest.raises(ValueError, match="different immutable identity"):
        _execute(client, lease, resources=("document:other",),
                 read_back=lambda: calls.append("read"),
                 mutation_callback=lambda _: calls.append("write"))
    assert calls == []


def test_initial_ledger_read_failure_is_safe_and_skips_provider_callbacks(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    lease = _lease(client)
    from core.operations import mutations as mutation_module

    monkeypatch.setattr(
        mutation_module, "get_mutation_effect",
        lambda *_: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_DETAILS")),
    )
    calls = []
    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(client, lease, read_back=lambda: calls.append("read"),
                 mutation_callback=lambda _: calls.append("write"))
    assert calls == []
    assert "SYNTHETIC_PRIVATE_DETAILS" not in str(error.value)


def test_reservation_storage_failure_is_safe_without_settlement_or_provider_calls(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    lease = _lease(client)
    from core.integrations.firestore import mutations as repository
    from core.operations import mutations as mutation_module

    settlements = []
    monkeypatch.setattr(
        mutation_module, "dispatch_mutation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_STORAGE_DETAILS")),
    )
    monkeypatch.setattr(mutation_module, "reconcile_mutation", lambda *args, **kwargs: settlements.append(args))
    calls = []
    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(client, lease, read_back=lambda: calls.append("read"),
                 mutation_callback=lambda _: calls.append("write"))
    assert calls == []
    assert settlements == []
    assert repository.get_mutation_effect(client, OPERATION_ID, "effect-a") is None
    assert "SYNTHETIC_STORAGE_DETAILS" not in str(error.value)


@pytest.mark.parametrize("snapshot", [None, MutationSnapshot("")])
def test_malformed_snapshot_fails_safely_before_write(firestore_fake, snapshot):
    client = firestore_fake.client_class()
    lease = _lease(client)
    writes = []
    with pytest.raises(MutationConflict):
        _execute(client, lease, read_back=lambda: snapshot,
                 mutation_callback=lambda _: writes.append("write"))
    assert writes == []


def test_snapshot_result_callback_details_are_sanitized(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationConflict) as error:
        _execute(
            client, lease, read_back=lambda: MutationSnapshot("sha256:desired"),
            result_from_snapshot=lambda _: (_ for _ in ()).throw(
                RuntimeError("SYNTHETIC_PRIVATE_DETAILS")
            ),
            mutation_callback=lambda _: pytest.fail("already desired; no write"),
        )
    assert "SYNTHETIC_PRIVATE_DETAILS" not in str(error.value)


def test_invalid_required_inputs_are_rejected_before_storage_or_provider_calls(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    lease = _lease(client)
    calls = []
    from core.operations import mutations as mutation_module

    storage_reads = []
    monkeypatch.setattr(
        mutation_module, "get_mutation_effect",
        lambda *_: storage_reads.append("read") or pytest.fail("validation reached storage"),
    )
    invalid_cases = (
        {"operation_id": 7},
        {"operation_id": "bad/operation"},
        {"effect_id": ""},
        {"effect_id": "bad/effect"},
        {"resources": ("document:a", 3)},
        {"resources": "document:a"},
        {"resources": b"document:a"},
        {"payload_digest": None},
        {"desired_state_digest": 4},
        {"expected_pre_state": ""},
        {"policy": MutationPolicy.DOCS_REVISION, "expected_revision": 7},
        {"fence": True},
        {"fence": 0},
        {"read_back": None},
        {"mutation_callback": "not callable"},
        {"result_from_snapshot": "not callable"},
    )
    for invalid in invalid_cases:
        with pytest.raises(ValueError):
            arguments = {
                "read_back": lambda: calls.append("read"),
                "mutation_callback": lambda _: calls.append("write"),
            }
            arguments.update(invalid)
            _execute(client, lease, **arguments)
    assert calls == []
    assert storage_reads == []


def test_valid_resource_generators_remain_supported(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    snapshots = iter((MutationSnapshot("sha256:old"), MutationSnapshot("sha256:desired")))
    assert _execute(
        client, lease, resources=(resource for resource in ("document:a", "document:b")),
        read_back=lambda: next(snapshots),
        mutation_callback=lambda _: "verified",
    ) == "verified"
    assert get_resource_guard(client, "document:a").resource_epoch == 1
    assert get_resource_guard(client, "document:b").resource_epoch == 1


def test_fresh_already_desired_state_skips_write_after_reservation(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    calls = []

    result = _execute(
        client, lease,
        read_back=lambda: calls.append("read") or MutationSnapshot("sha256:desired"),
        mutation_callback=lambda _: calls.append("write"),
        result_from_snapshot=lambda _: "already done",
    )
    assert result == "already done"
    assert calls == ["read"]
    assert get_resource_guard(client, "document:a").resource_epoch == 1


def test_docs_revision_is_forwarded_and_stale_revision_never_writes(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    observed = []
    docs_snapshots = iter((
        MutationSnapshot("sha256:old", revision="rev-7"),
        MutationSnapshot("sha256:desired", revision="rev-8"),
    ))

    assert _execute(
        client, lease, policy=MutationPolicy.DOCS_REVISION,
        expected_revision="rev-7",
        read_back=lambda: next(docs_snapshots),
        mutation_callback=lambda revision: observed.append(revision) or "saved",
    ) == "saved"
    assert observed == ["rev-7"]

    stale_client = firestore_fake.client_class()
    stale_id = "docs-stale-operation"
    stale_lease = _lease(stale_client, operation_id=stale_id)
    writes = []
    with pytest.raises(MutationConflict):
        _execute(
            stale_client, stale_lease, operation_id=stale_id, policy=MutationPolicy.DOCS_REVISION,
            expected_revision="rev-7",
            read_back=lambda: MutationSnapshot("sha256:old", revision="rev-8"),
            mutation_callback=lambda revision: writes.append(revision),
        )
    assert writes == []
    assert get_resource_guard(stale_client, "document:a").unsettled_effect_ids == ()


def test_expired_lease_during_pre_read_cannot_enter_provider_callback(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    lease = _lease(client, now=fake_clock.now(), lease_seconds=5)
    writes = []

    def read():
        fake_clock.advance(seconds=6)
        return MutationSnapshot("sha256:old")

    with pytest.raises(MutationConflict):
        _execute(
            client, lease, now=fake_clock.now(), clock=fake_clock.now,
            read_back=read, mutation_callback=lambda _: writes.append("write"),
        )
    assert writes == []
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()


def test_pre_read_takeover_prevents_stale_unsent_effect_from_releasing_guard(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    old = _lease(client, now=fake_clock.now(), lease_seconds=5)
    writes = []

    def read():
        fake_clock.advance(seconds=6)
        current = claim_operation(
            client, OPERATION_ID, owner="new-worker", now=fake_clock.now(), lease_seconds=30,
        )
        assert current is not None and current.fence > old.fence
        return MutationSnapshot("sha256:old")

    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, old, now=fake_clock.now(), clock=fake_clock.now,
            read_back=read, mutation_callback=lambda _: writes.append("write"),
        )
    assert writes == []
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


def test_expired_lease_after_remote_write_is_not_confirmed(firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    lease = _lease(client, now=fake_clock.now(), lease_seconds=5)
    writes = []
    reads = []

    def read():
        reads.append("read")
        return MutationSnapshot("sha256:old" if len(reads) == 1 else "sha256:desired")

    def write(_revision):
        writes.append("write")
        fake_clock.advance(seconds=6)
        return "provider-result"

    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease, now=fake_clock.now(), clock=fake_clock.now,
            read_back=read, mutation_callback=write,
        )
    assert writes == ["write"]
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


@pytest.mark.parametrize("post_snapshot", [
    MutationSnapshot("sha256:old"),
    MutationSnapshot("sha256:third-state"),
])
def test_success_response_without_desired_post_readback_stays_unresolved(firestore_fake, post_snapshot):
    client = firestore_fake.client_class()
    lease = _lease(client)
    reads = iter((MutationSnapshot("sha256:old"), post_snapshot))
    writes = []

    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease, read_back=lambda: next(reads),
            mutation_callback=lambda _: writes.append("write") or "provider-said-success",
        )
    assert writes == ["write"]
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


def test_post_write_readback_unavailable_stays_unresolved(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    reads = iter((MutationSnapshot("sha256:old"), RuntimeError("secret read failure")))
    calls = []

    def read():
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(client, lease, read_back=read,
                 mutation_callback=lambda _: calls.append("write") or "success")
    assert calls == ["write"]
    assert "secret" not in str(error.value)
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


def test_explicit_known_not_sent_rejection_settles_without_readback_retry(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    reads = []
    writes = []

    with pytest.raises(MutationConflict):
        _execute(
            client, lease,
            read_back=lambda: reads.append("read") or MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: writes.append("write") or (_ for _ in ()).throw(
                MutationNotSent("secret provider rejection")
            ),
        )
    assert reads == ["read"]
    assert writes == ["write"]
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()


def test_drive_metadata_never_receives_revision_and_unknown_prestate_stays_guarded(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    observed = []
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease,
            read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda revision: observed.append(revision) or (_ for _ in ()).throw(
                TimeoutError("secret-token https://private.invalid/payload")
            ),
        )
    assert observed == [None]
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


def test_uncertain_write_readback_desired_confirms_without_second_write(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    writes = []
    reads = []

    snapshots = iter((MutationSnapshot("sha256:old"), MutationSnapshot("sha256:desired")))

    def read():
        reads.append("read")
        return next(snapshots)

    def write(_revision):
        writes.append("write")
        raise TimeoutError("secret provider detail")

    assert _execute(
        client, lease, read_back=read, mutation_callback=write,
        result_from_snapshot=lambda _: "reconciled",
    ) == "reconciled"
    assert reads == ["read", "read"]
    assert writes == ["write"]
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()


def test_unknown_effect_rejects_changed_desired_digest_before_callbacks(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease,
            read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(TimeoutError("uncertain")),
        )
    calls = []
    with pytest.raises(ValueError, match="different immutable identity"):
        _execute(
            client, lease, desired_state_digest="sha256:other-desired",
            read_back=lambda: calls.append("read"),
            mutation_callback=lambda _: calls.append("write"),
        )
    assert calls == []


def test_reservation_race_identity_error_never_reconciles_other_effect(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    lease = _lease(client)
    from core.operations import mutations as mutation_module

    original_get = mutation_module.get_mutation_effect
    created = False

    def stale_absence_then_create_other_identity(db, operation_id, effect_id):
        nonlocal created
        if not created:
            created = True
            dispatch_mutation(
                db, operation_id=operation_id, effect_id=effect_id, owner="worker",
                fence=lease.fence, kind="content_update", resources=("document:other",),
                payload_digest="sha256:other", now=NOW,
                mutation_callback=lambda: "other-result", expected_pre_state="sha256:old",
                desired_state_digest="sha256:other-desired", mutation_policy="drive_metadata",
            )
            return None
        return original_get(db, operation_id, effect_id)

    monkeypatch.setattr(mutation_module, "get_mutation_effect", stale_absence_then_create_other_identity)
    calls = []
    with pytest.raises(ValueError, match="different immutable identity"):
        _execute(
            client, lease, resources=("document:other",),
            read_back=lambda: calls.append("read"),
            mutation_callback=lambda _: calls.append("write"),
        )
    assert calls == []
    effect = original_get(client, OPERATION_ID, "effect-a")
    assert effect.state.value == "confirmed"
    found, result = mutation_module.get_mutation_result(client, OPERATION_ID, "effect-a")
    assert found and result == "other-result"


def test_unresolved_effect_blocks_same_resource_after_takeover_but_independent_work_proceeds(
    firestore_fake, fake_clock,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    old = _lease(client, now=fake_clock.now(), lease_seconds=5)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, old,
            read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(TimeoutError("uncertain")),
        )
    fake_clock.advance(seconds=6)
    current = claim_operation(
        client, OPERATION_ID, owner="new-worker", now=fake_clock.now(), lease_seconds=30,
    )
    assert current is not None and current.fence > old.fence

    other_id = "independent-mutation-operation"
    create_operation(
        client, operation_id=other_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:other",
        immutable_input_ref="other-ref", now=fake_clock.now(),
    )
    other_lease = claim_operation(
        client, other_id, owner="other-worker", now=fake_clock.now(), lease_seconds=30,
    )
    assert other_lease is not None
    blocked_calls = []
    with pytest.raises(MutationConflict):
        execute_mutation(
            client, operation_id=other_id, effect_id="blocked-effect", owner="other-worker",
            fence=other_lease.fence, kind="content_update", resources=("document:a",),
            payload_digest="sha256:blocked", desired_state_digest="sha256:blocked-desired",
            expected_pre_state="sha256:blocked-old", expected_revision=None,
            policy=MutationPolicy.DRIVE_METADATA, now=fake_clock.now(),
            read_back=lambda: blocked_calls.append("read"),
            mutation_callback=lambda _: blocked_calls.append("write"),
        )
    assert blocked_calls == []

    independent_snapshots = iter((
        MutationSnapshot("sha256:other-old"), MutationSnapshot("sha256:other-desired"),
    ))
    assert execute_mutation(
        client, operation_id=other_id, effect_id="independent-effect", owner="other-worker",
        fence=other_lease.fence, kind="content_update", resources=("document:b",),
        payload_digest="sha256:other-payload", desired_state_digest="sha256:other-desired",
        expected_pre_state="sha256:other-old", expected_revision=None,
        policy=MutationPolicy.DRIVE_METADATA, now=fake_clock.now(),
        read_back=lambda: next(independent_snapshots),
        mutation_callback=lambda _: "done",
    ) == "done"

    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, old, owner="worker",
            read_back=lambda: MutationSnapshot("sha256:desired"),
            mutation_callback=lambda _: pytest.fail("stale owner must not settle"),
        )
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, current, owner="new-worker",
            read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: pytest.fail("must not redispatch"),
        )
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)

    # The new lease can settle once a bounded read-back proves the desired state.
    assert _execute(
        client, current, owner="new-worker",
        read_back=lambda: MutationSnapshot("sha256:desired"),
        mutation_callback=lambda _: pytest.fail("must not redispatch"),
        result_from_snapshot=lambda _: "reconciled",
    ) == "reconciled"
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()


def test_verified_reconciliation_releases_guard_for_next_conflicting_write(monkeypatch, firestore_fake):
    _install_firestore_map_name_boundary(monkeypatch)
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease,
            read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(TimeoutError("uncertain")),
        )

    assert _execute(
        client, lease,
        read_back=lambda: MutationSnapshot("sha256:desired"),
        mutation_callback=lambda _: pytest.fail("reconciliation must not redispatch"),
        result_from_snapshot=lambda _: "verified-result",
    ) == "verified-result"
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()

    next_operation_id = "next-conflicting-mutation-operation"
    create_operation(
        client, operation_id=next_operation_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller", immutable_input_digest="sha256:next-input",
        immutable_input_ref="next-input-ref", now=NOW,
    )
    next_lease = claim_operation(
        client, next_operation_id, owner="next-worker", now=NOW, lease_seconds=30,
    )
    assert next_lease is not None
    snapshots = iter((MutationSnapshot("sha256:desired"), MutationSnapshot("sha256:second-desired")))
    assert execute_mutation(
        client, operation_id=next_operation_id, effect_id="effect-b", owner="next-worker",
        fence=next_lease.fence, kind="content_update", resources=("document:a",),
        payload_digest="sha256:second-payload", desired_state_digest="sha256:second-desired",
        expected_pre_state="sha256:desired", expected_revision=None,
        policy=MutationPolicy.DRIVE_METADATA, now=NOW,
        read_back=lambda: next(snapshots),
        mutation_callback=lambda _: "second-result",
    ) == "second-result"
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ()



def test_failed_effect_during_reconciliation_is_never_reported_as_success(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease, read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(TimeoutError("uncertain")),
        )

    from core.operations import mutations as mutation_module

    original_reconcile = mutation_module.reconcile_mutation

    def settle_failed(*args, **kwargs):
        kwargs["confirmed"] = False
        kwargs.pop("result", None)
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(mutation_module, "reconcile_mutation", settle_failed)
    with pytest.raises(MutationConflict):
        _execute(
            client, lease, read_back=lambda: MutationSnapshot("sha256:desired"),
            result_from_snapshot=lambda _: "must not escape",
        )
    effect = client.collection("dialog_operations").document(OPERATION_ID).collection(
        "effects",
    ).document("effect-a").get().to_dict()
    assert effect["state"] == "failed"


def test_reconciliation_storage_failure_is_safe_and_does_not_redispatch(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationOutcomeUnknown):
        _execute(
            client, lease, read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(TimeoutError("uncertain")),
        )
    from core.operations import mutations as mutation_module

    monkeypatch.setattr(
        mutation_module, "reconcile_mutation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_DETAILS")),
    )
    callbacks = []
    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(
            client, lease,
            read_back=lambda: MutationSnapshot("sha256:desired"),
            mutation_callback=lambda _: callbacks.append("must not write"),
            result_from_snapshot=lambda _: "reconciled",
        )
    assert callbacks == []
    assert "SYNTHETIC_PRIVATE_DETAILS" not in str(error.value)
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)


def test_result_persistence_failure_after_sent_write_is_safe_and_replay_reads_only(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    lease = _lease(client)
    from core.integrations.firestore import mutations as repository

    original_confirm = repository._confirm
    writes = []
    snapshots = iter((MutationSnapshot("sha256:old"), MutationSnapshot("sha256:desired")))

    def fail_confirmation(*_args, **_kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAILS")

    monkeypatch.setattr(repository, "_confirm", fail_confirmation)
    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(
            client, lease, read_back=lambda: next(snapshots),
            mutation_callback=lambda _: writes.append("write") or "provider-result",
        )
    assert writes == ["write"]
    assert "SYNTHETIC_PRIVATE_DETAILS" not in str(error.value)
    assert get_resource_guard(client, "document:a").unsettled_effect_ids == ("effect-a",)

    monkeypatch.setattr(repository, "_confirm", original_confirm)
    assert _execute(
        client, lease, read_back=lambda: MutationSnapshot("sha256:desired"),
        mutation_callback=lambda _: writes.append("blind-redispatch"),
        result_from_snapshot=lambda _: "reconciled",
    ) == "reconciled"
    assert writes == ["write"]


def test_secret_provider_details_do_not_escape_unknown_error(firestore_fake):
    client = firestore_fake.client_class()
    lease = _lease(client)
    with pytest.raises(MutationOutcomeUnknown) as error:
        _execute(
            client, lease, read_back=lambda: MutationSnapshot("sha256:old"),
            mutation_callback=lambda _: (_ for _ in ()).throw(
                RuntimeError("token=secret https://private.invalid/path")
            ),
        )
    assert "secret" not in str(error.value)
    assert "private.invalid" not in str(error.value)
