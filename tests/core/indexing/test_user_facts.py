import pytest

from core.indexing import user_facts


DOC_ID = "synthetic-doc-fact"


def test_add_user_fact_builds_versionless_payload_and_embeds_before_upsert(monkeypatch):
    events = []
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda doc_id: {
        "status": "synced",
        "version": "v2.0.0",
        "parent_id": "synthetic-folder-ops",
        "dialog_date": "2026-01-12",
    })
    monkeypatch.setattr(user_facts, "get_root_folder_id", lambda: "pytest-synthetic-root")
    monkeypatch.setattr(user_facts, "embed", lambda texts: events.append(("embed", texts)) or [[0.25, 0.5]])
    monkeypatch.setattr(user_facts, "upsert_user_facts", lambda payloads, vectors: events.append(("upsert", payloads, vectors)))

    result = user_facts.add_user_fact(f" {DOC_ID} ", "  Remember Tuesday.  ")

    assert result == DOC_ID
    assert events == [
        ("embed", ["Remember Tuesday."]),
        ("upsert", [{
            "type": "user_fact",
            "doc_id": DOC_ID,
            "text": "Remember Tuesday.",
            "root_folder_id": "pytest-synthetic-root",
            "parent_id": "synthetic-folder-ops",
            "dialog_date": "2026-01-12",
            "dialog_date_num": 20260112,
        }], [[0.25, 0.5]]),
    ]
    assert "version" not in events[1][1][0]


@pytest.mark.parametrize("doc_id,text", [(" ", "fact"), ("doc", " ")])
def test_add_user_fact_rejects_empty_trimmed_inputs(doc_id, text):
    with pytest.raises(user_facts.AddFactInputError, match="doc_id and text are required"):
        user_facts.add_user_fact(doc_id, text)


def test_add_user_fact_requires_existing_synced_transcript(monkeypatch):
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc_id: None)
    with pytest.raises(user_facts.AddFactDocumentNotFound, match=f"Document not found: {DOC_ID}"):
        user_facts.add_user_fact(DOC_ID, "Fact")

    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc_id: {"status": "syncing"})
    with pytest.raises(user_facts.AddFactTranscriptNotSynced, match="status=syncing"):
        user_facts.add_user_fact(DOC_ID, "Fact")


@pytest.mark.parametrize(("dialog_date", "expected"), [("", 0), ("invalid", None)])
def test_add_user_fact_date_conversion(monkeypatch, dialog_date, expected):
    captured = []
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc_id: {
        "status": "synced", "dialog_date": dialog_date,
    })
    monkeypatch.setattr(user_facts, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(user_facts, "embed", lambda _texts: [[1.0]])
    monkeypatch.setattr(user_facts, "upsert_user_facts", lambda payloads, _vectors: captured.extend(payloads))

    user_facts.add_user_fact(DOC_ID, "Fact")

    assert captured[0]["dialog_date_num"] == expected


def test_add_user_fact_preserves_explicit_null_payload_fields(monkeypatch):
    captured = []
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc_id: {
        "status": "synced", "dialog_date": None, "parent_id": None,
    })
    monkeypatch.setattr(user_facts, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(user_facts, "embed", lambda _texts: [[1.0]])
    monkeypatch.setattr(user_facts, "upsert_user_facts", lambda payloads, _vectors: captured.extend(payloads))

    user_facts.add_user_fact(DOC_ID, "Fact")

    assert captured[0]["dialog_date"] is None
    assert captured[0]["parent_id"] is None
    assert captured[0]["dialog_date_num"] == 0


def test_add_user_fact_defaults_missing_payload_fields_to_empty_strings(monkeypatch):
    captured = []
    monkeypatch.setattr(user_facts, "get_transcript_record", lambda _doc_id: {"status": "synced"})
    monkeypatch.setattr(user_facts, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(user_facts, "embed", lambda _texts: [[1.0]])
    monkeypatch.setattr(user_facts, "upsert_user_facts", lambda payloads, _vectors: captured.extend(payloads))

    user_facts.add_user_fact(DOC_ID, "Fact")

    assert captured[0]["dialog_date"] == ""
    assert captured[0]["parent_id"] == ""
    assert captured[0]["dialog_date_num"] == 0
