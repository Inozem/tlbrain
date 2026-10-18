from datetime import datetime, timezone

import pytest
from google.api_core.exceptions import Aborted, NotFound
from google.cloud import firestore
from google.cloud import firestore_v1
from google.cloud.firestore_v1 import client as firestore_client_module
from google.cloud.firestore_v1.field_path import FieldPath
from google.cloud.firestore_v1 import _helpers

from core.integrations.firestore import transcripts as transcript_repository
from tests.fixtures.firestore_fake import FakeFirestoreClient


def test_replace_merge_and_update_field_path_semantics(firestore_fake):
    db = firestore.Client()
    ref = db.collection("records").document("one")
    ref.set({"profile": {"name": "Ada", "role": "dev"}, "a.b": 1, "extra": True})
    ref.set({"profile": {"name": "Grace"}}, merge=True)
    ref.update({"profile.name": "Katherine", "profile.team": "compiler", "a.b": 2})
    data = ref.get().to_dict()
    assert data == {
        "profile": {"name": "Katherine", "role": "dev", "team": "compiler"},
        "a.b": 1,
        "a": {"b": 2},
        "extra": True,
    }

    ref.set({"profile": {"name": "Lin"}})
    assert ref.get().to_dict() == {"profile": {"name": "Lin"}}
    ref.set({"profile": {}}, merge=True)
    assert ref.get().to_dict() == {"profile": {}}
    ref.set({"profile": {"count": 2, "keep": True}})
    ref.set({"profile": {"count": firestore.Increment(3)}}, merge=True)
    assert ref.get().to_dict() == {"profile": {"count": 5, "keep": True}}


def test_transforms_missing_update_snapshots_and_input_isolation(firestore_fake):
    db = firestore.Client()
    ref = db.collection("records").document("one")
    with pytest.raises(NotFound):
        ref.update({"x": 1})
    source = {"count": 2, "nested": {"left": 1, "gone": True}}
    ref.set(source)
    source["nested"]["left"] = 99
    snap = ref.get()
    exposed = snap.to_dict()
    exposed["nested"]["left"] = 100
    assert snap.to_dict()["nested"]["left"] == 1
    ref.update({
        "count": firestore.Increment(3),
        "nested.gone": firestore.DELETE_FIELD,
        "nested.right": firestore.Increment(4),
        "updated": firestore.SERVER_TIMESTAMP,
        "missing": firestore.DELETE_FIELD,
    })
    data = ref.get().to_dict()
    assert data["count"] == 5
    assert data["nested"] == {"left": 1, "right": 4}
    assert data["updated"].tzinfo is not None
    assert "missing" not in data
    with pytest.raises(ValueError):
        ref.set({"gone": firestore.DELETE_FIELD})
    with pytest.raises(ValueError):
        ref.set({"nested": {"gone": firestore.DELETE_FIELD}})


def test_updates_use_sdk_validation_and_materialize_nested_transforms(firestore_fake):
    ref = firestore.Client().collection("records").document("one")
    ref.set({"profile": {"name": "Ada", "keep": True}, "plain": 1})
    ref.update({"profile": {"name": "Grace"}})
    assert ref.get().to_dict()["profile"] == {"name": "Grace"}

    ref.set({"profile": {"name": "Grace", "keep": True}, "count": 2})
    ref.update({"profile": {"updated": firestore.SERVER_TIMESTAMP}, "count": firestore.Increment(3)})
    data = ref.get().to_dict()
    assert set(data["profile"]) == {"updated"}
    assert data["profile"]["updated"].tzinfo is not None
    assert data["count"] == 5
    assert not any(isinstance(value, (firestore.Increment,)) for value in data.values())

    for invalid in ({}, {"a": 1, "a.b": 2}, {"profile": {"name": firestore.DELETE_FIELD}}):
        with pytest.raises((ValueError, TypeError)):
            ref.update(invalid)
    with pytest.raises((TypeError, ValueError)):
        ref.update({"bad": object()})
    with pytest.raises((TypeError, ValueError)):
        ref.set({"bad": object()})
    unsupported = (
        firestore.ArrayUnion(["x"]),
        firestore.ArrayRemove(["x"]),
        firestore.Maximum(1),
        firestore.Minimum(1),
    )
    for transform in unsupported:
        with pytest.raises(NotImplementedError):
            ref.update({"items": transform})
        with pytest.raises(NotImplementedError):
            ref.set({"items": transform}, merge=True)


def test_explicit_merge_mask_accepts_sdk_fieldpath_and_applies_transforms(firestore_fake):
    ref = firestore.Client().collection("records").document("one")
    ref.set({"profile": {"count": 2, "keep": True}})
    ref.set({"profile": {"count": firestore.Increment(3)}}, merge=[FieldPath("profile", "count")])
    assert ref.get().to_dict() == {"profile": {"count": 5, "keep": True}}
    with pytest.raises(ValueError):
        ref.set({"profile": {"count": 1}}, merge=["profile", "profile.count"])
    with pytest.raises(ValueError):
        ref.set({"profile": {"count": 1}}, merge=["missing"])


def test_queries_are_immutable_and_support_real_filters_projection_limit(firestore_fake):
    db = firestore.Client()
    col = db.collection("records")
    col.document("a").set({"score": 1, "profile": {"group": "x"}, "other": 1})
    col.document("b").set({"score": 2, "profile": {"group": "x"}, "other": 2})
    col.document("c").set({"score": 3, "profile": {"group": "y"}, "other": 3})
    query = col.where(filter=firestore.FieldFilter("profile.group", "==", "x"))
    filtered = query.where("score", ">", 1)
    assert [doc.id for doc in query.stream()] == ["a", "b"]
    result = filtered.select(["score"]).limit(1).stream()
    assert [(doc.id, doc.to_dict()) for doc in result] == [("b", {"score": 2})]
    assert filtered.limit(0).stream() == []


def test_batch_faults_target_document_paths_and_preserve_atomicity(firestore_fake):
    db = firestore.Client()
    first = db.collection("records").document("first")
    second = db.collection("records").document("second")
    first.set({"value": 1})
    second.set({"value": 2})

    def batch_update():
        batch = db.batch()
        batch.update(first, {"value": 10})
        batch.update(second, {"value": 20})
        return batch

    firestore_fake.fail_next("batch_commit", path="records/second", phase="before")
    with pytest.raises(RuntimeError):
        batch_update().commit()
    assert first.get().to_dict() == {"value": 1}
    assert second.get().to_dict() == {"value": 2}

    firestore_fake.fail_next("batch_commit", path="records/second", phase="after")
    with pytest.raises(RuntimeError):
        batch_update().commit()
    assert first.get().to_dict() == {"value": 10}
    assert second.get().to_dict() == {"value": 20}


def test_real_constructor_aliases_are_patched(firestore_fake):
    assert firestore.Client is FakeFirestoreClient
    assert firestore_v1.Client is FakeFirestoreClient
    assert firestore_client_module.Client is FakeFirestoreClient


def test_batch_commit_is_atomic_and_transaction_conflicts_retry_with_real_decorator(firestore_fake):
    db = firestore.Client()
    one = db.collection("records").document("one")
    absent = db.collection("records").document("absent")
    one.set({"value": 1})
    batch = db.batch()
    batch.update(one, {"value": 2})
    batch.update(absent, {"value": 3})
    with pytest.raises(NotFound):
        batch.commit()
    assert one.get().to_dict() == {"value": 1}

    attempts = []

    @firestore.transactional
    def change(transaction):
        attempts.append(1)
        snapshot = one.get(transaction=transaction)
        if len(attempts) == 1:
            one.update({"outside": True})
        transaction.update(one, {"value": snapshot.to_dict()["value"] + 1})

    change(db.transaction())
    assert len(attempts) == 2
    assert one.get().to_dict() == {"value": 2, "outside": True}


def test_real_transaction_decorator_exhaustion_and_rollback(firestore_fake):
    db = firestore.Client()
    ref = db.collection("records").document("one")
    ref.set({"value": 1})
    calls = []

    @firestore.transactional
    def always_conflict(transaction):
        calls.append(1)
        snapshot = ref.get(transaction=transaction)
        ref.update({"outside": len(calls)})
        transaction.update(ref, {"value": snapshot.to_dict()["value"] + 1})

    with pytest.raises(ValueError, match="Failed to commit transaction in 2 attempts"):
        always_conflict(db.transaction(max_attempts=2))
    assert len(calls) == 2
    assert ref.get().to_dict() == {"value": 1, "outside": 2}

    @firestore.transactional
    def raises_after_staging(transaction):
        transaction.update(ref, {"value": 99})
        raise RuntimeError("abort staged writes")

    with pytest.raises(RuntimeError, match="abort staged writes"):
        raises_after_staging(db.transaction())
    assert ref.get().to_dict()["value"] == 1

    @firestore.transactional
    def read_after_write(transaction):
        transaction.update(ref, {"value": 7})
        ref.get(transaction=transaction)

    with pytest.raises(_helpers.ReadAfterWriteError):
        read_after_write(db.transaction())
    assert ref.get().to_dict()["value"] == 1


def test_commit_faults_distinguish_uncommitted_and_lost_acknowledgment(firestore_fake):
    db = firestore.Client()
    ref = db.collection("records").document("one")
    firestore_fake.fail_next("set", path="records/one", phase="before")
    with pytest.raises(RuntimeError):
        ref.set({"value": 1})
    assert not ref.get().exists
    firestore_fake.fail_next("set", path="records/one", phase="after")
    with pytest.raises(RuntimeError):
        ref.set({"value": 2})
    assert ref.get().to_dict() == {"value": 2}
    assert [entry["outcome"] for entry in firestore_fake.trace()] == ["failed", "committed", "failed"]


def test_injected_aborted_commit_uses_real_sdk_retry(firestore_fake):
    db = firestore.Client()
    ref = db.collection("records").document("one")
    attempts = []
    firestore_fake.fail_next("commit", phase="before", exception=Aborted("injected contention"))

    @firestore.transactional
    def create(transaction):
        attempts.append(1)
        transaction.set(ref, {"attempt": len(attempts)})

    create(db.transaction(max_attempts=3))
    assert len(attempts) == 2
    assert ref.get().to_dict() == {"attempt": 2}


def test_per_test_state_reset_and_real_consumer_smoke(firestore_fake):
    assert firestore_fake.documents == {}
    assert transcript_repository.ensure_imported("smoke", "parent", "source") is True
    assert transcript_repository.acquire_for_syncing("arbitrary-new") is True
    transcript_repository.save_index("smoke", {"nested": {"a": 1}})
    transcript_repository.update_index("smoke", {"nested.b": 2})
    assert transcript_repository.load_index("smoke") == {"nested": {"a": 1, "b": 2}}


def test_clock_faults_and_revision_change_for_deleted_missing_document(firestore_fake):
    db = firestore.Client()
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)
    ref = db.collection("records").document("one")
    ref.set({"at": firestore.SERVER_TIMESTAMP})
    assert ref.get().to_dict()["at"] == now
    with pytest.raises(ValueError, match="timezone-aware"):
        firestore_fake.set_clock(lambda: datetime(2025, 1, 2))
        ref.set({"bad_clock": firestore.SERVER_TIMESTAMP})
    firestore_fake.set_clock(lambda: now)
    ref.delete()

    # Missing-document reads carry a revision, so create/delete ABA is visible.
    tx = db.transaction()
    tx._begin()
    assert not ref.get(transaction=tx).exists
    ref.set({"value": 1})
    ref.delete()
    with pytest.raises(Aborted):
        tx._commit()
