from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore import mcp_sessions
from core.integrations.firestore import operations as operation_repository
from core.operations.models import OperationOrigin


NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def test_create_session_persists_only_hashed_id_and_fixed_lifetime(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="2025-06-18", now=NOW,
    )

    assert initialized.session.expires_at == NOW + timedelta(hours=24)
    assert initialized.session.created_at == NOW
    docs = client.collection(mcp_sessions.MCP_SESSIONS_COLLECTION)
    stored = docs.document(mcp_sessions._session_hash(initialized.session_id)).get().to_dict()
    assert stored == {
        "caller_binding": "principal:alice",
        "protocol_version": "2025-06-18",
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=24),
    }
    assert initialized.session_id not in repr(stored)
    assert initialized.session_id not in firestore_fake.documents


def test_active_session_requires_matching_caller_and_is_not_sliding(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )

    assert mcp_sessions.get_session(
        client, initialized.session_id, caller_binding="principal:alice", now=NOW + timedelta(hours=23),
    ) == initialized.session
    with pytest.raises(mcp_sessions.SessionUnavailableError):
        mcp_sessions.get_session(
            client, initialized.session_id, caller_binding="principal:bob", now=NOW,
        )
    with pytest.raises(mcp_sessions.SessionUnavailableError):
        mcp_sessions.get_session(
            client, initialized.session_id, caller_binding="principal:alice", now=NOW + timedelta(hours=24),
        )


def test_request_replay_is_typed_canonical_and_independent_by_id(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="anonymous:install", protocol_version="v1", now=NOW,
    )
    kwargs = {
        "caller_binding": "anonymous:install",
        "arguments": {"z": [1, True], "a": "Р“В©"},
        "now": NOW,
    }
    first = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, request_id=7, **kwargs,
    )
    canonical_replay = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, request_id=7, arguments={"a": "Р“В©", "z": [1, True]},
        caller_binding=kwargs["caller_binding"], now=NOW,
    )
    string_id = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, request_id="7", **kwargs,
    )
    other_id = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, request_id=8, **kwargs,
    )
    new_session = mcp_sessions.create_session(
        client, caller_binding="anonymous:install", protocol_version="v1", now=NOW,
    )
    fresh_session = mcp_sessions.resolve_or_reserve_request(
        client, new_session.session_id, request_id=7, **kwargs,
    )

    assert canonical_replay == mcp_sessions.RequestReservation(
        first.operation_id, replayed=True, accepted_at=NOW,
    )
    assert canonical_replay.accepted_at == first.accepted_at == NOW
    request_record = (
        client.collection(mcp_sessions.MCP_SESSIONS_COLLECTION)
        .document(mcp_sessions._session_hash(initialized.session_id))
        .collection("requests")
        .document(mcp_sessions._request_id_hash(7))
        .get()
        .to_dict()
    )
    assert request_record["result_ref"] == f"dialog_operations/{first.operation_id}"
    assert string_id.operation_id != first.operation_id
    assert other_id.operation_id != first.operation_id
    assert fresh_session.operation_id != first.operation_id
    assert len({first.operation_id, string_id.operation_id, other_id.operation_id, fresh_session.operation_id}) == 4


def test_read_only_request_lookup_returns_persisted_acceptance_time(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    first = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, caller_binding="principal:alice", request_id="same",
        arguments={"value": 1}, now=NOW,
    )
    replay = mcp_sessions.lookup_request_replay(
        client, initialized.session_id, caller_binding="principal:alice", request_id="same",
        arguments={"value": 1}, now=NOW + timedelta(minutes=5),
    )
    assert replay == mcp_sessions.RequestReservation(
        first.operation_id, replayed=True, accepted_at=NOW,
    )
    assert mcp_sessions.lookup_request_replay(
        client, initialized.session_id, caller_binding="principal:alice", request_id="fresh",
        arguments={"value": 1}, now=NOW,
    ) is None


def test_recovery_binding_uses_only_matching_durable_request_reference(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    reservation = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, caller_binding="principal:alice", request_id=5,
        arguments={"title": "private title"}, now=NOW,
    )
    ref = mcp_sessions.request_recovery_ref(initialized.session_id, 5)
    immutable_digest = "a" * 64
    operation_repository.create_operation(
        client, operation_id=reservation.operation_id, origin=OperationOrigin.MCP,
        origin_operation_ref=ref, caller_binding="principal:alice",
        immutable_input_digest=immutable_digest,
        immutable_input_ref=f"dialog_operations/{reservation.operation_id}/input_parts/manifest",
        now=NOW,
    )
    operation = operation_repository.get_operation(client, reservation.operation_id)

    assert mcp_sessions.get_recovery_binding(
        client, ref, operation_id=reservation.operation_id,
        immutable_input_digest=operation.immutable_input_digest,
    ) == {
        "context_id": operation.origin_operation_ref,
        "draft_id": mcp_sessions._request_id_hash(5),
        "manifest_digest": operation.immutable_input_digest,
    }
    assert mcp_sessions.get_recovery_binding(
        client, ref, operation_id="other-operation", immutable_input_digest=immutable_digest,
    ) is None


@pytest.mark.parametrize("ref", [
    "mcp_sessions/session/requests/request",
    "mcp_sessions/" + "a" * 64 + "/requests/" + "b" * 64 + "/extra",
    "mcp_sessions/" + "A" * 64 + "/requests/" + "b" * 64,
])
def test_recovery_binding_rejects_malformed_reference(firestore_fake, ref):
    client = firestore_fake.client_class()
    assert mcp_sessions.get_recovery_binding(
        client, ref, operation_id="operation", immutable_input_digest="a" * 64,
    ) is None


def test_recovery_binding_rejects_missing_mapping_and_malformed_operation_digest(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    reservation = mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, caller_binding="principal:alice", request_id=6,
        arguments={"title": "private title"}, now=NOW,
    )
    ref = mcp_sessions.request_recovery_ref(initialized.session_id, 6)
    assert mcp_sessions.get_recovery_binding(
        client, mcp_sessions.request_recovery_ref(initialized.session_id, 7),
        operation_id=reservation.operation_id,
        immutable_input_digest="a" * 64,
    ) is None
    assert mcp_sessions.get_recovery_binding(
        client, ref, operation_id=reservation.operation_id,
        immutable_input_digest="not-a-sha256-digest",
    ) is None


def test_changed_arguments_conflict_without_writing(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    mcp_sessions.resolve_or_reserve_request(
        client, initialized.session_id, caller_binding="principal:alice", request_id=1,
        arguments={"value": 1}, now=NOW,
    )
    before = firestore_fake.trace()

    with pytest.raises(mcp_sessions.RequestArgumentsConflictError):
        mcp_sessions.resolve_or_reserve_request(
            client, initialized.session_id, caller_binding="principal:alice", request_id=1,
            arguments={"value": 2}, now=NOW,
        )

    assert firestore_fake.trace() == before


@pytest.mark.parametrize("bad_id", [True, 7.0])
def test_request_id_rejects_bool_and_float_before_writing(firestore_fake, bad_id):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    before = firestore_fake.trace()

    with pytest.raises(ValueError, match="request_id"):
        mcp_sessions.resolve_or_reserve_request(
            client, initialized.session_id, caller_binding="principal:alice", request_id=bad_id,
            arguments={}, now=NOW,
        )

    assert firestore_fake.trace() == before


@pytest.mark.parametrize("arguments", [{"value": float("nan")}, {1: "not-json"}, {"value": object()}])
def test_non_json_arguments_are_rejected_before_writing(firestore_fake, arguments):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    before = firestore_fake.trace()

    with pytest.raises(ValueError):
        mcp_sessions.resolve_or_reserve_request(
            client, initialized.session_id, caller_binding="principal:alice", request_id="x",
            arguments=arguments, now=NOW,
        )

    assert firestore_fake.trace() == before


@pytest.mark.parametrize("failure", ["unknown", "mismatched", "expired"])
def test_invalid_session_never_creates_request_mapping(firestore_fake, failure):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )
    session_id = "unknown" if failure == "unknown" else initialized.session_id
    binding = "principal:bob" if failure == "mismatched" else "principal:alice"
    current = NOW + timedelta(hours=24) if failure == "expired" else NOW
    before = firestore_fake.trace()

    with pytest.raises(mcp_sessions.SessionUnavailableError):
        mcp_sessions.resolve_or_reserve_request(
            client, session_id, caller_binding=binding, request_id=1, arguments={}, now=current,
        )

    assert firestore_fake.trace() == before


def test_concurrent_duplicate_requests_converge_on_one_operation(firestore_fake):
    client = firestore_fake.client_class()
    initialized = mcp_sessions.create_session(
        client, caller_binding="principal:alice", protocol_version="v1", now=NOW,
    )

    def reserve(_):
        return mcp_sessions.resolve_or_reserve_request(
            client, initialized.session_id, caller_binding="principal:alice", request_id="same",
            arguments={"value": [1, 2]}, now=NOW,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(16)))

    assert len({result.operation_id for result in results}) == 1
    assert sum(not result.replayed for result in results) == 1
