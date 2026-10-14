"""Characterize the existing skip paths and transactional sync claim."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from core.integrations.firestore import transcripts as transcript_repository
from core.parsing.parser import parse_document
from services.vector_sync.app import processor
from core.indexing.hashing import sha256_text, sha256_utterance


DOC_ID = "synthetic-skip-claim-doc"
PARENT_ID = "synthetic-skip-claim-parent"
NEW_PARENT_ID = "synthetic-skip-claim-parent-new"
ROOT_ID = "pytest-synthetic-skip-claim-root"
INDEX_PATH = f"transcript_index/{DOC_ID}"
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _raw(app_root, case="valid_full") -> str:
    cases = json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))["cases"]
    return cases[case]["raw"]


def _index_record(raw: str, *, parent_id: str = PARENT_ID) -> dict:
    metadata, utterances = parse_document(raw)
    content_hash = sha256_text(raw)
    return {
        "doc_id": DOC_ID,
        "status": "imported",
        "parent_id": parent_id,
        "parent_id_hash": sha256_text(parent_id),
        "content_hash": content_hash,
        "version": content_hash,
        "dialog_date": metadata.get("date", ""),
        "provider": metadata.get("provider", ""),
        "utterance_hashes": {
            str(item["order_index"]): sha256_utterance(item["order_index"], item["speaker"], item["text"])
            for item in utterances
        },
        "skipped_utterances": [],
        "speakers": sorted({item["speaker"] for item in utterances if item.get("speaker")}),
    }


def _seed(firestore_fake, raw: str, *, parent_id: str = PARENT_ID) -> dict:
    record = _index_record(raw, parent_id=parent_id)
    firestore_fake.client_class().collection("transcript_index").document(DOC_ID).set(record)
    return record


def _embedding_events(sdk_fakes):
    return [event for event in sdk_fakes.trace if
            (event["boundary"] == "gemini" and event["operation"] in {"embed_content", "generate_content"})
            or event["boundary"] == "sparse"]


def _qdrant_events(sdk_fakes, operation: str | None = None):
    return [event for event in sdk_fakes.trace if event["boundary"] == "qdrant"
            and (operation is None or event["operation"] == operation)]


def _record_effects(monkeypatch, names, effects):
    for name in names:
        original = getattr(processor, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            effects.append((_name, args, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(processor, name, wrapped)


def test_repeated_unchanged_processing_only_marks_synced(app_root, sdk_fakes, firestore_fake, monkeypatch):
    raw = _raw(app_root)
    initial = _seed(firestore_fake, raw)
    effects = []
    _record_effects(monkeypatch, ["mark_synced", "update_index", "set_payload_parent_id", "set_payload_dialog_date"], effects)

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw) == "skipped"
    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw) == "skipped"

    final = firestore_fake.documents[INDEX_PATH]
    assert final["status"] == "synced"
    assert final["content_hash"] == initial["content_hash"]
    assert final["parent_id"] == PARENT_ID
    assert final["parent_id_hash"] == sha256_text(PARENT_ID)
    assert [name for name, _, _ in effects] == ["mark_synced", "mark_synced"]
    assert len(_qdrant_events(sdk_fakes)) == 0
    assert _embedding_events(sdk_fakes) == []


def test_parent_only_change_updates_payload_and_parent_checkpoint_without_embeddings(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    raw = _raw(app_root)
    _seed(firestore_fake, raw)
    effects = []
    _record_effects(monkeypatch, ["set_payload_parent_id", "update_index", "mark_synced"], effects)

    assert processor.process_one(DOC_ID, NEW_PARENT_ID, ROOT_ID, raw) == "skipped"

    final = firestore_fake.documents[INDEX_PATH]
    assert final["status"] == "synced"
    assert final["parent_id"] == NEW_PARENT_ID
    assert final["parent_id_hash"] == sha256_text(NEW_PARENT_ID)
    assert final["content_hash"] == sha256_text(raw)
    assert [name for name, _, _ in effects] == ["set_payload_parent_id", "update_index", "mark_synced"]
    assert effects[0][1] == (DOC_ID, ROOT_ID, NEW_PARENT_ID)
    assert effects[1][1][1] == {"parent_id": NEW_PARENT_ID, "parent_id_hash": sha256_text(NEW_PARENT_ID)}
    payload_calls = _qdrant_events(sdk_fakes, "set_payload")
    assert len(payload_calls) == 1
    assert payload_calls[0]["arguments"]["payload"] == {"parent_id": NEW_PARENT_ID}
    assert _qdrant_events(sdk_fakes, "upsert") == []
    assert _qdrant_events(sdk_fakes, "delete") == []
    assert _embedding_events(sdk_fakes) == []


def test_date_only_change_updates_metadata_and_qdrant_date_without_embeddings(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    old_raw = _raw(app_root)
    new_raw = old_raw.replace("DATE: 2026-01-12", "DATE: 2026-01-13", 1)
    initial = _seed(firestore_fake, old_raw)
    assert _index_record(new_raw)["utterance_hashes"] == initial["utterance_hashes"]
    effects = []
    _record_effects(monkeypatch, ["set_payload_dialog_date", "update_index", "mark_synced"], effects)

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new_raw) == "skipped"

    final = firestore_fake.documents[INDEX_PATH]
    assert final["status"] == "synced"
    assert final["dialog_date"] == "2026-01-13"
    assert final["content_hash"] == sha256_text(new_raw)
    assert final["parent_id"] == PARENT_ID
    assert final["parent_id_hash"] == sha256_text(PARENT_ID)
    assert [name for name, _, _ in effects] == ["set_payload_dialog_date", "update_index", "mark_synced"]
    assert effects[0][1] == (DOC_ID, ROOT_ID, "2026-01-13", 20260113)
    assert effects[1][1][1]["dialog_date"] == "2026-01-13"
    assert effects[1][1][1]["parent_id_hash"] == sha256_text(PARENT_ID)
    date_calls = _qdrant_events(sdk_fakes, "set_payload")
    assert len(date_calls) == 1
    assert date_calls[0]["arguments"]["payload"] == {"dialog_date": "2026-01-13", "dialog_date_num": 20260113}
    assert _qdrant_events(sdk_fakes, "upsert") == []
    assert _qdrant_events(sdk_fakes, "delete") == []
    assert _embedding_events(sdk_fakes) == []


def test_claim_fresh_stale_cutoff_and_missing_document_use_transactional_sdk(
    firestore_fake, sdk_fakes, monkeypatch,
):
    firestore_fake.set_clock(lambda: NOW)

    class FrozenDatetime:
        @staticmethod
        def now(_tz=None):
            return NOW

    monkeypatch.setattr(transcript_repository, "datetime", FrozenDatetime)
    ref = firestore_fake.client_class().collection("transcript_index").document(DOC_ID)
    fresh_record = {
        "doc_id": DOC_ID,
        "status": "syncing",
        "status_changed_at": NOW - timedelta(minutes=15) + timedelta(seconds=1),
    }
    ref.set(fresh_record)
    effects = []

    def record_mark_error(*args, **kwargs):
        effects.append((args, kwargs))

    monkeypatch.setattr(processor, "mark_error", record_mark_error)

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, "synthetic text is never read") == "not_acquired"
    assert firestore_fake.documents[INDEX_PATH] == fresh_record
    assert effects == []
    assert _qdrant_events(sdk_fakes) == []
    assert _embedding_events(sdk_fakes) == []
    assert not any(event["operation"] == "commit" and event.get("outcome") == "committed" for event in firestore_fake.trace())

    ref.update({"status_changed_at": NOW - timedelta(minutes=15)})
    assert transcript_repository.acquire_for_syncing(DOC_ID) is True
    assert firestore_fake.documents[INDEX_PATH]["status"] == "syncing"
    assert firestore_fake.documents[INDEX_PATH]["status_changed_at"] == NOW
    assert [event for event in firestore_fake.trace() if event["operation"] == "commit" and event.get("outcome") == "committed"]

    missing_id = "synthetic-missing-claim-doc"
    assert transcript_repository.acquire_for_syncing(missing_id) is True
    assert firestore_fake.documents[f"transcript_index/{missing_id}"] == {
        "doc_id": missing_id, "status": "syncing", "status_changed_at": NOW,
    }


def test_competing_claims_conflict_retry_and_only_one_wins(firestore_fake, monkeypatch):
    from tests.fixtures.firestore_fake import FakeTransaction

    firestore_fake.set_clock(lambda: NOW)

    class FrozenDatetime:
        @staticmethod
        def now(_tz=None):
            return NOW

    monkeypatch.setattr(transcript_repository, "datetime", FrozenDatetime)
    barrier = threading.Barrier(2)
    first_reads_lock = threading.Lock()
    first_reads = 0
    original_read = FakeTransaction._read

    def synchronized_read(transaction, reference):
        nonlocal first_reads
        snapshot = original_read(transaction, reference)
        should_wait = False
        with first_reads_lock:
            if first_reads < 2:
                first_reads += 1
                should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return snapshot

    monkeypatch.setattr(FakeTransaction, "_read", synchronized_read)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(transcript_repository.acquire_for_syncing, DOC_ID) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(results) == [False, True]
    assert firestore_fake.documents[INDEX_PATH]["status"] == "syncing"
    trace = firestore_fake.trace()
    assert any(event["operation"] == "commit" and event.get("phase") == "conflict" for event in trace)
    assert sum(event["operation"] == "commit" and event.get("outcome") == "committed" for event in trace) == 1
    # One initial failed commit plus the winner's commit and the loser's fresh retry read.
    assert sum(event["operation"] == "commit" and event.get("phase") == "conflict" for event in trace) == 1


def test_processing_failure_is_marked_error_before_original_exception_reraises(
    app_root, firestore_fake, monkeypatch,
):
    raw = _raw(app_root)
    failure = ValueError("synthetic load-index failure")

    def fail_load(_doc_id):
        raise failure

    monkeypatch.setattr(processor, "load_index", fail_load)
    with pytest.raises(ValueError, match="synthetic load-index failure") as raised:
        processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw)

    assert raised.value is failure
    final = firestore_fake.documents[INDEX_PATH]
    assert final["status"] == "error"
    assert final["error"] == str(failure)
    assert final["error_stage"] == "vector_sync"
    assert final["status_changed_at"] is not None
    assert any(event["operation"] == "commit" and event.get("outcome") == "committed"
               for event in firestore_fake.trace())

