"""Offline checkpoint and recovery characterization for full reindexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from google.genai import types
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from core.integrations.qdrant import writer
from core.indexing import processor


DOC_ID = "synthetic-doc-main"
PARENT_ID = "synthetic-folder-ops"
ROOT_ID = "pytest-synthetic-root"
INDEX_PATH = f"transcript_index/{DOC_ID}"


def _response(summary: str, facts: list[str]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part(text=json.dumps({"summary": summary, "facts": facts}))
        ]))
    ])


def _prepare_inputs(app_root: Path, sdk_fakes, firestore_fake) -> str:
    transcript = json.loads(
        (app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8")
    )["cases"]["valid_full"]
    firestore_fake.client_class().collection("transcript_index").document(DOC_ID).set({
        "doc_id": DOC_ID,
        "status": "imported",
        "parent_id": PARENT_ID,
        "version": "old-synthetic-version",
        "content_hash": "old-synthetic-content-hash",
    })

    # This point is synthetic and versionless, as user-created facts are.
    writer.upsert_user_facts([{
        "type": "user_fact",
        "doc_id": DOC_ID,
        "text": "Remember the invented Tuesday review.",
        "root_folder_id": ROOT_ID,
    }], [[0.5, 0.25]])
    sdk_fakes.gemini.queue("generate_content", response=_response("Window zero summary.", ["The review is Tuesday."]))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window three summary.", []))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window six summary.", []))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window nine summary.", []))
    for _ in range(5):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)]
        ))
    return transcript["raw"]


def _install_effect_recorder(monkeypatch):
    effects: list[tuple[str, object]] = []
    names = (
        "update_index", "upsert_utterances", "upsert_summaries", "upsert_facts",
        "delete_old_versions", "update_skipped_utterances", "mark_synced",
    )
    originals = {name: getattr(processor, name) for name in names}

    def wrap(name):
        original = originals[name]

        def recorded(*args, **kwargs):
            detail = args[0] if name == "update_index" and args else None
            fields = args[1] if name == "update_index" and len(args) > 1 else None
            effects.append((name, fields if name == "update_index" else detail))
            return original(*args, **kwargs)

        return recorded

    for name in names:
        monkeypatch.setattr(processor, name, wrap(name))
    return effects, originals


def _expected_version(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _upsert_events(sdk_fakes):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == "upsert"]


def _delete_events(sdk_fakes):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == "delete"]


def test_full_reindex_freezes_effect_order_and_persisted_checkpoints(
    app_root, sdk_fakes, firestore_fake, monkeypatch,
):
    raw = _prepare_inputs(app_root, sdk_fakes, firestore_fake)
    effects, _ = _install_effect_recorder(monkeypatch)

    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw) == "processed"

    version = _expected_version(raw)
    assert [name for name, _ in effects] == [
        "update_index",                 # dialog_date + provider metadata checkpoint
        "upsert_utterances",
        "upsert_summaries", "upsert_facts", "upsert_summaries",
        "upsert_summaries", "upsert_summaries",
        "delete_old_versions",
        "update_skipped_utterances",
        "update_index",                 # parent/content hash/version checkpoint
        "mark_synced",
        "update_index",                 # utterance_hashes checkpoint
        "update_index",                 # speakers checkpoint
    ]
    assert effects[0] == ("update_index", {"dialog_date": "2026-01-12", "provider": "tldv"})
    assert effects[9] == ("update_index", {
        "parent_id": PARENT_ID,
        "parent_id_hash": "0bcf77c60f7b290aad8540b7672b0288e91c6605aaa327fd31c2279da3c6f677",
        "content_hash": version,
        "version": version,
    })
    assert effects[11] == ("update_index", {"utterance_hashes": {
        "0": "52dc41703dbd24e6d7f4e65e5795fb6874c6fa0c4b9dbf7bab760d1bfbdf40cd",
        "1": "b6c81b7725e929a617d430e66180797f9c62f5a2676803406ca2431fbe45abe6",
        "2": "3d2f108a757a970ad55625ef47db4010c35ed230588e0fc833083b487f5e90dc",
        "3": "d1d323553c8b7e99809c347e313818ac367c416b8a863dac5d35c4ad4fcb5dd3",
        "4": "8eb5dc89b8580288780b3eb33ca731760464a91df1d65236a36fd98c1cd04de0",
        "5": "24fe0a74f2b9b0dfaff274f16eb6144137859e709a66698c067fcc555b58128f",
        "6": "06c45440b76ee0e6351e4269472d64ff7b8b18264d60c73193abc72c82214221",
        "7": "bb67c1ca79142b27d8893517d489b212fa957f5ca1389d7ce669334e07ca355b",
        "8": "4e600bdac560b43ce4c3daa775c33c5aa9dde8a39366d50edd54da6fc481f683",
        "9": "f4650c87d9f4f2e26d26ac56a0e5d7220ff56be6872182e2e956bafb5d094ef0",
    }})
    assert effects[12] == ("update_index", {"speakers": ["Fixture Speaker A", "Fixture Speaker B"]})

    upsert_calls = _upsert_events(sdk_fakes)
    assert [[point.payload["type"] for point in call["points"]] for call in upsert_calls] == [
        ["user_fact"], ["utterance"] * 10, ["summary"], ["fact"],
        ["summary"], ["summary"], ["summary"],
    ]
    points = [point for call in upsert_calls for point in call["points"]]
    utterance_points = points[1:11]
    assert len(utterance_points) == 10
    assert [point.payload["order_index"] for point in utterance_points] == list(range(10))
    assert [(point.payload["speaker"], point.payload["text"]) for point in utterance_points] == [
        ("Fixture Speaker A", "We will review the sample schedule."),
        ("Fixture Speaker B", "The draft has three sections."),
        ("Fixture Speaker A", "Each section needs a clear owner."),
        ("Fixture Speaker B", "We can track changes in a small table."),
        ("Fixture Speaker A", "The first review is on Tuesday."),
        ("Fixture Speaker B", "Notes should include the decision."),
        ("Fixture Speaker A", "A second pass can check the wording."),
        ("Fixture Speaker B", "We will keep the example data invented."),
        ("Fixture Speaker A", "The final copy can stay brief."),
        ("Fixture Speaker B", "We will record the next step."),
    ]
    assert utterance_points[0].payload == {
        "type": "utterance", "root_folder_id": ROOT_ID, "doc_id": DOC_ID,
        "version": version, "parent_id": PARENT_ID, "dialog_date": "2026-01-12",
        "dialog_date_num": 20260112, "speaker": "Fixture Speaker A",
        "text": "We will review the sample schedule.", "order_index": 0,
    }
    assert all(point.payload["version"] == version for point in points[1:])
    assert all(point.payload["root_folder_id"] == ROOT_ID for point in points[1:])
    summary_points = [point for point in points if point.payload["type"] == "summary"]
    assert [point.payload["center_index"] for point in summary_points] == [0, 3, 6, 9]
    fact_point = next(point for point in points if point.payload["type"] == "fact")
    assert fact_point.payload == {
        "type": "fact", "text": "The review is Tuesday.", "root_folder_id": ROOT_ID,
        "doc_id": DOC_ID, "summary_id": f"{DOC_ID}:0:{version}", "center_index": 0,
        "covered_range": [0, 2], "parent_id": PARENT_ID, "dialog_date": "2026-01-12",
        "dialog_date_num": 20260112, "version": version,
    }

    deletion = _delete_events(sdk_fakes)[0]
    assert deletion["collection_name"] == "tldv_vectors"
    assert deletion["points_selector"] == Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
            FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
            FieldCondition(key="type", match=MatchAny(any=["utterance", "summary", "fact"])),
        ],
        must_not=[FieldCondition(key="version", match=MatchValue(value=version))],
    )
    assert "user_fact" not in deletion["points_selector"].must[2].match.any
    assert points[0].payload["type"] == "user_fact"  # cleanup selector leaves this type outside its scope

    index = firestore_fake.documents[INDEX_PATH]
    assert index["status"] == "synced"
    assert index["dialog_date"] == "2026-01-12"
    assert index["provider"] == "tldv"
    assert index["parent_id"] == PARENT_ID
    assert index["parent_id_hash"] == "0bcf77c60f7b290aad8540b7672b0288e91c6605aaa327fd31c2279da3c6f677"
    assert index["content_hash"] == version
    assert index["version"] == version
    assert index["utterance_hashes"] == effects[11][1]["utterance_hashes"]
    assert index["speakers"] == ["Fixture Speaker A", "Fixture Speaker B"]
    assert index["skipped_utterances"] == []


@pytest.mark.parametrize("failure_boundary", ["utterance_write", "content_checkpoint", "old_cleanup"])
def test_full_reindex_failure_is_marked_and_retry_recovers(
    app_root, sdk_fakes, firestore_fake, monkeypatch, failure_boundary,
):
    raw = _prepare_inputs(app_root, sdk_fakes, firestore_fake)
    effects, originals = _install_effect_recorder(monkeypatch)
    failure = ValueError(f"synthetic {failure_boundary} failure")

    if failure_boundary == "utterance_write":
        sdk_fakes.qdrant.queue("upsert", error=failure)
    elif failure_boundary == "old_cleanup":
        sdk_fakes.qdrant.queue("delete", error=failure)
    else:
        real_update_index = originals["update_index"]
        injected = False

        def fail_content_checkpoint(doc_id, fields):
            nonlocal injected
            if "content_hash" in fields and not injected:
                injected = True
                firestore_fake.fail_next(
                    "update", path=INDEX_PATH, phase="before", exception=failure
                )
            return real_update_index(doc_id, fields)

        monkeypatch.setattr(processor, "update_index", fail_content_checkpoint)

    with pytest.raises(ValueError, match=failure_boundary):
        processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw)

    failed_index = firestore_fake.documents[INDEX_PATH]
    assert failed_index["status"] == "error"
    assert failed_index["error"] == str(failure)
    assert failed_index["error_stage"] == "vector_sync"
    firestore_updates = [event for event in firestore_fake.trace()
                         if event.get("path") == INDEX_PATH and event["operation"] == "update"]
    assert any(event.get("outcome") == "committed" and event.get("path") == INDEX_PATH
               for event in firestore_updates)
    assert not any(name == "mark_synced" for name, _ in effects)
    assert not any(name == "update_index" and isinstance(detail, dict) and "utterance_hashes" in detail
                   for name, detail in effects)
    if failure_boundary == "content_checkpoint":
        assert any(event.get("outcome") == "failed" and event.get("phase") == "before"
                   for event in firestore_updates)

    monkeypatch.setattr(processor, "update_index", originals["update_index"])
    _prepare_retry_responses(sdk_fakes)
    assert processor.process_one(DOC_ID, PARENT_ID, ROOT_ID, raw) == "processed"
    recovered = firestore_fake.documents[INDEX_PATH]
    assert recovered["status"] == "synced"
    assert recovered["error"] is None
    assert recovered["version"] == _expected_version(raw)
    assert len(recovered["utterance_hashes"]) == 10
    assert recovered["speakers"] == ["Fixture Speaker A", "Fixture Speaker B"]
    assert len(_delete_events(sdk_fakes)) == (1 if failure_boundary == "utterance_write" else 2)
    assert all("user_fact" not in event["points_selector"].must[2].match.any
               for event in _delete_events(sdk_fakes))


def _prepare_retry_responses(sdk_fakes):
    sdk_fakes.gemini.queue("generate_content", response=_response("Window zero summary.", ["The review is Tuesday."]))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window three summary.", []))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window six summary.", []))
    sdk_fakes.gemini.queue("generate_content", response=_response("Window nine summary.", []))
    for _ in range(5):
        sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.125] * 768)]
        ))
