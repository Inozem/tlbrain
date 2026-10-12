import json
from copy import deepcopy

from google.genai import types
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, PointStruct, SparseVector

from core.qdrant import writer
from services.mcp.app.mcp.handler import handle_tools_call
from services.mcp.app.mcp.schemas import JSONRPCRequest


COLLECTION = "tldv_vectors"
DOC_ID = "synthetic-doc-main"
ROOT_ID = "pytest-synthetic-root"


def _event(sdk_fakes, operation):
    return [event["arguments"] for event in sdk_fakes.trace
            if event["boundary"] == "qdrant" and event["operation"] == operation][-1]


def test_upserts_preserve_point_ids_vectors_and_payloads(sdk_fakes):
    utterance = {
        "type": "utterance", "root_folder_id": ROOT_ID, "doc_id": DOC_ID, "version": "v2.0.0",
        "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12", "dialog_date_num": 20260112,
        "speaker": "Fixture Speaker A", "text": "We will review the sample schedule.", "order_index": 4,
    }
    utterance_snapshot = deepcopy(utterance)
    sparse = SparseVector(indices=[2, 7], values=[0.25, 0.75])
    writer.upsert_utterances([utterance], [sparse])
    utterance_event = _event(sdk_fakes, "upsert")
    utterance_point = utterance_event["points"][0]
    assert isinstance(utterance_point, PointStruct)
    assert utterance_point.id == "16fbfc2d-5564-5c64-96ea-e447785fc56d"
    assert utterance_point.vector == {"bm25": sparse}
    assert utterance_point.payload == utterance_snapshot
    assert utterance == utterance_snapshot

    summary = {
        "type": "summary", "root_folder_id": ROOT_ID, "summary_id": "synthetic-doc-main:3:v2.0.0",
        "text": "Review schedule and owners.", "doc_id": DOC_ID, "center_index": 3,
        "covered_range": [1, 5], "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12",
        "dialog_date_num": 20260112, "version": "v2.0.0",
    }
    summary_snapshot = deepcopy(summary)
    writer.upsert_summaries([summary], [[0.125, -0.25]])
    summary_point = _event(sdk_fakes, "upsert")["points"][0]
    assert isinstance(summary_point, PointStruct)
    assert summary_point.id == "a3ba3653-928a-5796-976e-f807239ad83c"
    assert summary_point.vector == {"dense": [0.125, -0.25]}
    assert summary_point.payload == summary_snapshot
    assert summary == summary_snapshot

    fact = {
        "type": "fact", "text": "Review is Tuesday.", "root_folder_id": ROOT_ID, "doc_id": DOC_ID,
        "summary_id": "synthetic-doc-main:3:v2.0.0", "center_index": 3, "covered_range": [1, 5],
        "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12", "dialog_date_num": 20260112,
        "version": "v2.0.0",
    }
    fact_snapshot = deepcopy(fact)
    writer.upsert_facts([fact], [[0.5, 0.75]])
    fact_point = _event(sdk_fakes, "upsert")["points"][0]
    assert isinstance(fact_point, PointStruct)
    assert fact_point.id == "b7cc1956-ed48-53d4-9ad9-d2a4ae374a9b"
    assert fact_point.vector == {"dense": [0.5, 0.75]}
    assert fact_point.payload == fact_snapshot
    assert fact == fact_snapshot

    user_fact = {"type": "user_fact", "doc_id": DOC_ID, "text": "Remember Tuesday.", "root_folder_id": ROOT_ID}
    user_fact_snapshot = deepcopy(user_fact)
    writer.upsert_user_facts([user_fact], [[1.0, 0.0]])
    user_fact_point = _event(sdk_fakes, "upsert")["points"][0]
    assert isinstance(user_fact_point, PointStruct)
    assert user_fact_point.id == "6133a223-d885-5d4d-a547-340651258c3d"
    assert user_fact_point.vector == {"dense": [1.0, 0.0]}
    assert user_fact_point.payload == user_fact_snapshot
    assert user_fact == user_fact_snapshot

    second_version = {**utterance, "version": "v3.0.0"}
    second_version_snapshot = deepcopy(second_version)
    writer.upsert_utterances([second_version], [sparse])
    assert second_version == second_version_snapshot
    upserts = [event["arguments"] for event in sdk_fakes.trace
               if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
    assert len(upserts) == 5
    for call in upserts:
        assert set(call) == {"args", "collection_name", "points"}
        assert call["args"] == ()
        assert call["collection_name"] == COLLECTION
        assert len(call["points"]) == 1
    assert [call["points"][0].id for call in upserts] == [
        "16fbfc2d-5564-5c64-96ea-e447785fc56d",
        "a3ba3653-928a-5796-976e-f807239ad83c",
        "b7cc1956-ed48-53d4-9ad9-d2a4ae374a9b",
        "6133a223-d885-5d4d-a547-340651258c3d",
        "16fbfc2d-5564-5c64-96ea-e447785fc56d",
    ]
    assert upserts[4]["points"][0].vector == {"bm25": sparse}
    assert upserts[4]["points"][0].payload == second_version_snapshot


def test_empty_upserts_do_not_write(sdk_fakes):
    writer.upsert_utterances([], [])
    writer.upsert_summaries([], [])
    writer.upsert_facts([], [])
    writer.upsert_user_facts([], [])
    assert not [event for event in sdk_fakes.trace if event["operation"] == "upsert"]


def test_delete_and_payload_mutation_filters_keep_exact_scope(sdk_fakes):
    writer.delete_old_versions(DOC_ID, "v3.0.0", ROOT_ID)
    writer.delete_by_doc_id(DOC_ID, ROOT_ID)
    writer.delete_utterances_by_order_indexes(DOC_ID, ROOT_ID, [1, 4, 7])
    writer.delete_summaries_by_center_indexes(DOC_ID, ROOT_ID, [0, 6])
    writer.set_payload_parent_id(DOC_ID, ROOT_ID, "synthetic-folder-new")
    writer.set_payload_dialog_date(DOC_ID, ROOT_ID, "2026-02-03", 20260203)
    writer.set_payload_dialog_date(DOC_ID, ROOT_ID, "", None)
    writer.delete_utterances_by_order_indexes(DOC_ID, ROOT_ID, [])
    writer.delete_summaries_by_center_indexes(DOC_ID, ROOT_ID, [])

    events = [event for event in sdk_fakes.trace if event["boundary"] == "qdrant" and event["operation"] != "client"]
    assert [event["operation"] for event in events] == [
        "delete", "delete", "delete", "delete", "set_payload", "set_payload", "set_payload"
    ]
    selectors = [event["arguments"]["points_selector"] for event in events if event["operation"] == "delete"]
    for event in events:
        arguments = event["arguments"]
        assert arguments["args"] == ()
        if event["operation"] == "delete":
            assert set(arguments) == {"args", "collection_name", "points_selector"}
        else:
            assert set(arguments) == {"args", "collection_name", "payload", "points"}
    assert selectors[0] == Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
            FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
            FieldCondition(key="type", match=MatchAny(any=["utterance", "summary", "fact"])),
        ],
        must_not=[FieldCondition(key="version", match=MatchValue(value="v3.0.0"))],
    )
    assert "user_fact" not in selectors[0].must[2].match.any
    assert selectors[1] == Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
    ])
    assert selectors[2] == Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
        FieldCondition(key="order_index", match=MatchAny(any=[1, 4, 7])),
        FieldCondition(key="type", match=MatchValue(value="utterance")),
    ])
    assert selectors[3] == Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
        FieldCondition(key="center_index", match=MatchAny(any=[0, 6])),
        FieldCondition(key="type", match=MatchAny(any=["summary", "fact"])),
    ])
    parent_update, date_update, empty_date_update = [event["arguments"] for event in events if event["operation"] == "set_payload"]
    expected_target = Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=ROOT_ID)),
    ])
    assert parent_update["collection_name"] == COLLECTION
    assert parent_update["payload"] == {"parent_id": "synthetic-folder-new"}
    assert parent_update["points"] == expected_target
    assert date_update["payload"] == {"dialog_date": "2026-02-03", "dialog_date_num": 20260203}
    assert date_update["points"] == expected_target
    assert empty_date_update["payload"] == {"dialog_date": ""}
    assert empty_date_update["points"] == expected_target
    assert all(event["arguments"].get("collection_name") == COLLECTION for event in events)


def test_add_fact_handler_upserts_versionless_user_fact_with_real_embedding(sdk_fakes, firestore_fake):
    doc_id = "synthetic-doc-fact"
    firestore_fake.client_class().collection("transcript_index").document(doc_id).set({
        "status": "synced",
        "version": "v2.0.0",
        "content_hash": "frozen-content-hash",
        "parent_id": "synthetic-folder-ops",
    })
    sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)]
    ))
    request = JSONRPCRequest(
        jsonrpc="2.0",
        id="add-fact-1",
        method="tools/call",
        params={"name": "add_fact", "arguments": {"doc_id": f" {doc_id} ", "text": "  Remember Tuesday.  "}},
    )

    response = handle_tools_call(request)
    assert response["id"] == "add-fact-1"
    assert "error" not in response
    assert json.loads(response["result"]["content"][0]["text"]) == {"status": "ok", "doc_id": doc_id}
    trace = sdk_fakes.trace
    embed_call = next(event["arguments"] for event in trace if event["boundary"] == "gemini" and event["operation"] == "embed_content")
    assert embed_call["model"] == "gemini-embedding-2"
    assert embed_call["contents"] == ["Remember Tuesday."]
    assert embed_call["config"].output_dimensionality == 768
    point = _event(sdk_fakes, "upsert")["points"][0]
    gemini_events = [(index, event) for index, event in enumerate(trace)
                     if event["boundary"] == "gemini" and event["operation"] == "embed_content"]
    upsert_events = [(index, event) for index, event in enumerate(trace)
                     if event["boundary"] == "qdrant" and event["operation"] == "upsert"]
    assert len(gemini_events) == 1
    assert len(upsert_events) == 1
    assert gemini_events[0][0] < upsert_events[0][0]
    assert set(upsert_events[0][1]["arguments"]) == {"args", "collection_name", "points"}
    assert upsert_events[0][1]["arguments"]["args"] == ()
    assert upsert_events[0][1]["arguments"]["collection_name"] == COLLECTION
    assert len(upsert_events[0][1]["arguments"]["points"]) == 1
    assert point.id == "6d9c7799-86d9-5bce-b080-372ee731a624"
    assert point.vector == {"dense": [0.25] * 768}
    assert point.payload == {
        "type": "user_fact",
        "doc_id": doc_id,
        "text": "Remember Tuesday.",
        "root_folder_id": "pytest-synthetic-root",
        "parent_id": "synthetic-folder-ops",
        "dialog_date": "",
        "dialog_date_num": 0,
    }
    assert "version" not in point.payload
