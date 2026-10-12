"""Offline characterization of the current incremental index recovery path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from google.genai import types
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from core.integrations.qdrant import writer
from services.vector_sync.app import processor
from services.vector_sync.app.hashing import sha256_utterance


DOC_ID = "synthetic-incremental-doc"
PARENT_ID = "synthetic-incremental-parent"
ROOT_ID = "pytest-incremental-root"
OTHER_DOC = "synthetic-neighbor-doc"
OTHER_ROOT = "synthetic-neighbor-root"
INDEX_PATH = f"transcript_index/{DOC_ID}"


def _response(summary: str, facts: list[str]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part(text=json.dumps({"summary": summary, "facts": facts}))
        ]))
    ])


def _raw(lines: list[str]) -> str:
    return "DATE: 2026-01-12\nTIME: 09:00\nPROVIDER: tldv\n---\n" + "\n".join(lines) + "\n"


def _inputs(
    app_root: Path,
    sdk_fakes,
    firestore_fake,
    *,
    insertion_and_deletion: bool = True,
    prior_skipped: list[int] | None = None,
    queue_responses: bool = True,
) -> tuple[str, str, dict[str, str]]:
    old = json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))["cases"]["valid_full"]["raw"]
    old_lines = old.split("---\n", 1)[1].strip().splitlines()
    # One edit, one insertion at order 8, and deletion of the old trailing order 9.
    if insertion_and_deletion:
        new_lines = old_lines[:7] + [
            "Fixture Speaker B :: We can track changes in a revised table.",
            "Fixture Speaker A :: The inserted check uses invented sample data.",
        ]
    else:
        new_lines = old_lines.copy()
        new_lines[7] = "Fixture Speaker B :: We can track changes in a revised table."
    new = _raw(new_lines)
    old_hashes = {
        str(i): sha256_utterance(i, line.split(" :: ", 1)[0], line.split(" :: ", 1)[1])
        for i, line in enumerate(old_lines)
    }
    new_hashes = {
        str(i): sha256_utterance(i, line.split(" :: ", 1)[0], line.split(" :: ", 1)[1])
        for i, line in enumerate(new_lines)
    }
    firestore_fake.client_class().collection("transcript_index").document(DOC_ID).set({
        "doc_id": DOC_ID,
        "status": "imported",
        "parent_id": PARENT_ID,
        "parent_id_hash": hashlib.sha256(PARENT_ID.encode()).hexdigest(),
        "content_hash": hashlib.sha256(old.encode()).hexdigest(),
        "version": hashlib.sha256(old.encode()).hexdigest(),
        "dialog_date": "2026-01-12",
        "provider": "tldv",
        "utterance_hashes": old_hashes,
        "skipped_utterances": [0, 8] if prior_skipped is None else prior_skipped,
        "speakers": ["Previous synthetic speaker label"],
    })
    # The fake records SDK selectors/arguments; it does not model Qdrant persistence.
    writer.upsert_user_facts([{
        "type": "user_fact", "doc_id": DOC_ID, "text": "Invented user note.", "root_folder_id": ROOT_ID,
    }], [[0.5, 0.25]])
    writer.upsert_utterances([{
        "type": "utterance", "doc_id": OTHER_DOC, "root_folder_id": ROOT_ID,
        "order_index": 8, "text": "Neighbor document point.",
    }], [[0.1]])
    writer.upsert_summaries([{
        "type": "summary", "doc_id": DOC_ID, "root_folder_id": OTHER_ROOT,
        "summary_id": "synthetic-neighbor-summary", "center_index": 8, "text": "Neighbor root point.",
    }], [[0.1]])
    if queue_responses:
        _queue_generation(sdk_fakes)
    return old, new, new_hashes


def _queue_generation(sdk_fakes, *, empty_affected_window: bool = False) -> None:
    if empty_affected_window:
        empty_response = types.GenerateContentResponse(candidates=[
            types.Candidate(finish_reason=types.FinishReason.SAFETY)
        ])
        sdk_fakes.gemini.queue("generate_content", response=empty_response)
    else:
        sdk_fakes.gemini.queue("generate_content", response=_response("Affected six summary.", ["Invented fact for six."]))
    sdk_fakes.gemini.queue("generate_content", response=_response("Affected nine summary.", []))
    for _ in range(3):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)]
        ))


def _qdrant_events(sdk_fakes, operation: str):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == operation]


def _install_failure(monkeypatch, firestore_fake, boundary: str):
    targets = {
        "metadata": ("update_index", lambda fields: fields == {"dialog_date": "2026-01-12", "provider": "tldv"}, 1),
        "changed_hash_1": ("update_index", lambda fields: any(k.startswith("utterance_hashes.") for k in fields), 1),
        "changed_hash_2": ("update_index", lambda fields: any(k.startswith("utterance_hashes.") for k in fields), 2),
        "content": ("update_index", lambda fields: "content_hash" in fields, 1),
        "full_hash_map": ("update_index", lambda fields: "utterance_hashes" in fields, 1),
        "speakers": ("update_index", lambda fields: "speakers" in fields, 1),
        "mark_synced": ("mark_synced", lambda _fields: True, 1),
        "skipped": ("update_skipped_utterances", lambda _fields: True, 1),
        "utterance_delete": ("delete_utterances_by_order_indexes", lambda _fields: True, 1),
        "summary_delete": ("delete_summaries_by_center_indexes", lambda _fields: True, 1),
        "utterance_upsert": ("upsert_utterances", lambda _fields: True, 1),
        "summary_upsert": ("upsert_summaries", lambda _fields: True, 1),
        "fact_upsert": ("upsert_facts", lambda _fields: True, 1),
    }
    name, matches, target_count = targets[boundary]
    original = getattr(processor, name)
    calls = 0
    error = ValueError(f"synthetic {boundary} failure")

    def injected(*args, **kwargs):
        nonlocal calls
        candidate = args[1] if name == "update_index" and len(args) > 1 else None
        if matches(candidate):
            calls += 1
            if calls == target_count:
                if name == "update_index":
                    firestore_fake.fail_next("update", path=INDEX_PATH, phase="before", exception=error)
                    return original(*args, **kwargs)
                raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, name, injected)
    return name, original, error


def _delete_scope(event: dict) -> Filter:
    assert event["collection_name"] == "tldv_vectors"
    selector = event["points_selector"]
    assert isinstance(selector, Filter)
    return selector


def test_incremental_diff_scopes_checkpoints_and_write_order(app_root, sdk_fakes, firestore_fake, monkeypatch):
    _old, new, new_hashes = _inputs(app_root, sdk_fakes, firestore_fake)
    effects: list[tuple[str, object]] = []
    dotted_snapshots: list[dict] = []
    names = ("update_index", "delete_utterances_by_order_indexes", "delete_summaries_by_center_indexes",
             "upsert_utterances", "upsert_summaries", "upsert_facts", "update_skipped_utterances", "mark_synced")
    for name in names:
        original = getattr(processor, name)
        def wrap(*args, _name=name, _original=original, **kwargs):
            detail = args[1] if _name in {"update_index", "update_skipped_utterances"} else (args[2] if _name.endswith("indexes") else None)
            effects.append((_name, detail))
            result = _original(*args, **kwargs)
            if _name == "update_index" and any(key.startswith("utterance_hashes.") for key in detail):
                dotted_snapshots.append(dict(firestore_fake.documents[INDEX_PATH]["utterance_hashes"]))
            return result
        monkeypatch.setattr(processor, name, wrap)
    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new) == "processed"

    deletes = _qdrant_events(sdk_fakes, "delete")
    assert len(deletes) == 2
    utterance_selector, summary_selector = map(_delete_scope, deletes)
    must = lambda selector, key: next(item for item in selector.must if item.key == key).match
    assert must(utterance_selector, "doc_id") == MatchValue(value=DOC_ID)
    assert must(utterance_selector, "root_folder_id") == MatchValue(value=ROOT_ID)
    assert must(utterance_selector, "type") == MatchValue(value="utterance")
    assert set(must(utterance_selector, "order_index").any) == {7, 8, 9}
    assert must(summary_selector, "doc_id") == MatchValue(value=DOC_ID)
    assert must(summary_selector, "root_folder_id") == MatchValue(value=ROOT_ID)
    assert must(summary_selector, "type") == MatchAny(any=["summary", "fact"])
    assert set(must(summary_selector, "center_index").any) == {5, 6, 7, 8, 9, 10, 11}
    # Selector arguments exclude the seeded user_fact and points with a different doc/root.
    assert must(utterance_selector, "doc_id").value == DOC_ID
    assert must(summary_selector, "root_folder_id").value == ROOT_ID

    upserts = _qdrant_events(sdk_fakes, "upsert")
    changed = next(call for call in upserts if [p.payload["type"] for p in call["points"]] == ["utterance", "utterance"])
    assert [point.payload["order_index"] for point in changed["points"]] == [7, 8]
    operations = [event["operation"] for event in sdk_fakes.trace if event["boundary"] == "qdrant" and event["operation"] in {"delete", "upsert"}]
    delete_start = max(i for i, operation in enumerate(operations) if operation == "delete") - 1
    assert operations[delete_start:delete_start + 2] == ["delete", "delete"]
    assert operations[delete_start + 2:] == ["upsert"] * (len(operations) - delete_start - 2)
    assert len(operations) - delete_start - 2 >= 3
    assert [point.payload["center_index"] for call in upserts for point in call["points"] if point.payload.get("type") == "summary"] == [8, 6]

    index = firestore_fake.documents[INDEX_PATH]
    assert index["utterance_hashes"] == new_hashes
    assert index["utterance_hashes"]["0"] == sha256_utterance(0, "Fixture Speaker A", "We will review the sample schedule.")
    assert index["skipped_utterances"] == [0]
    assert index["status"] == "synced"
    assert index["content_hash"] == hashlib.sha256(new.encode()).hexdigest()
    assert index["version"] == hashlib.sha256(new.encode()).hexdigest()
    assert index["parent_id"] == PARENT_ID
    assert index["speakers"] == ["Fixture Speaker A", "Fixture Speaker B"]
    assert [name for name, _ in effects] == [
        "update_index", "delete_utterances_by_order_indexes", "delete_summaries_by_center_indexes",
        "upsert_utterances", "update_index", "update_index", "upsert_summaries", "upsert_facts",
        "update_skipped_utterances", "update_index", "mark_synced", "update_index", "update_index",
    ]
    assert set(effects[1][1]) == {7, 8, 9}
    assert set(effects[2][1]) == {5, 6, 7, 8, 9, 10, 11}
    assert effects[4][1] == {"utterance_hashes.7": new_hashes["7"]}
    assert effects[5][1] == {"utterance_hashes.8": new_hashes["8"]}
    assert len(dotted_snapshots) == 2
    assert dotted_snapshots[0]["0"] == sha256_utterance(0, "Fixture Speaker A", "We will review the sample schedule.")
    assert dotted_snapshots[0]["8"] != new_hashes["8"]
    assert dotted_snapshots[1]["0"] == dotted_snapshots[0]["0"]
    assert dotted_snapshots[1]["7"] == new_hashes["7"]
    assert effects[-5][1] == [0]
    assert effects[-4][1]["content_hash"] == index["content_hash"]
    assert effects[-3][0] == "mark_synced"
    assert effects[-1][1] == {"speakers": index["speakers"]}


@pytest.mark.parametrize("boundary", [
    "metadata", "utterance_delete", "summary_delete", "utterance_upsert", "changed_hash_1", "changed_hash_2",
    "summary_upsert", "fact_upsert", "skipped", "content", "mark_synced", "full_hash_map", "speakers",
])
def test_incremental_faults_restart_at_the_current_checkpoint_frontier(
    app_root, sdk_fakes, firestore_fake, monkeypatch, boundary,
):
    _old, new, _new_hashes = _inputs(app_root, sdk_fakes, firestore_fake)
    name, original, failure = _install_failure(monkeypatch, firestore_fake, boundary)
    with pytest.raises(ValueError, match=f"{boundary} failure"):
        processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new)
    failed = firestore_fake.documents[INDEX_PATH]
    assert failed["status"] == "error"
    assert failed["error"] == str(failure)

    monkeypatch.setattr(processor, name, original)
    trace_before_retry = len(sdk_fakes.trace)
    _queue_generation(sdk_fakes)
    result = processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new)
    retry_events = sdk_fakes.trace[trace_before_retry:]
    # Per-utterance checkpoints precede summaries. Once both have committed, a retry
    # sees no changed utterances and currently skips window regeneration.
    # A deleted trailing key remains in the old map until the final full-map write,
    # so most retries still detect it and rebuild. After the map checkpoint, speakers
    # failure has no remaining diff and the retry skips.
    content_checkpoint_written = boundary in {"mark_synced", "full_hash_map", "speakers"}
    if content_checkpoint_written:
        assert result == "skipped"
        assert not any(event["boundary"] == "qdrant" and event["operation"] in {"delete", "upsert"}
                       for event in retry_events)
    else:
        assert result == "processed"
        assert any(event["boundary"] == "qdrant" and event["operation"] == "delete"
                   for event in retry_events)
    assert firestore_fake.documents[INDEX_PATH]["status"] == "synced"
    if boundary in {"mark_synced", "full_hash_map", "speakers"}:
        # The content checkpoint precedes these writes. The early in-sync path skips
        # them on retry, leaving the current partial map/speaker state observable.
        if boundary in {"mark_synced", "full_hash_map"}:
            assert "9" in firestore_fake.documents[INDEX_PATH]["utterance_hashes"]
        else:
            assert "9" not in firestore_fake.documents[INDEX_PATH]["utterance_hashes"]
        assert firestore_fake.documents[INDEX_PATH]["speakers"] == ["Previous synthetic speaker label"]


def test_dotted_hash_before_window_failure_can_make_retry_skip_that_window(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    _old, new, new_hashes = _inputs(app_root, sdk_fakes, firestore_fake, insertion_and_deletion=False)
    name, original, failure = _install_failure(monkeypatch, firestore_fake, "summary_upsert")
    with pytest.raises(ValueError, match="summary_upsert failure"):
        processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new)
    after_failure = firestore_fake.documents[INDEX_PATH]
    assert after_failure["utterance_hashes"]["7"] == new_hashes["7"]
    assert after_failure["content_hash"] != hashlib.sha256(new.encode()).hexdigest()

    monkeypatch.setattr(processor, name, original)
    trace_before_retry = len(sdk_fakes.trace)
    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new) == "skipped"
    retry_events = sdk_fakes.trace[trace_before_retry:]
    assert not any(event["boundary"] == "qdrant" and event["operation"] in {"delete", "upsert"}
                   for event in retry_events)
    # This records only the attempted selector/write trace. The fake does not model
    # which summary points Qdrant persisted before the injected failure.


def test_incremental_merge_keeps_outside_skips_and_adds_empty_window_indices(
    app_root, sdk_fakes, firestore_fake,
):
    _old, new, _new_hashes = _inputs(
        app_root, sdk_fakes, firestore_fake,
        prior_skipped=[0, 0, 8], queue_responses=False,
    )
    _queue_generation(sdk_fakes, empty_affected_window=True)

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, new) == "processed"

    # Center 6 reanalyzes utterances 4..8. Prior skip 8 is removed, duplicate
    # outside skip 0 survives once, and the safety response adds 4..8.
    assert firestore_fake.documents[INDEX_PATH]["skipped_utterances"] == [0, 4, 5, 6, 7, 8]

