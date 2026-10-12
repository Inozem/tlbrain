import hashlib
import json
from copy import deepcopy

import pytest
from google.genai import types

from core.integrations.gemini import embeddings, llm
from core.parsing import processor
from core.parsing.parser import parse_document


@pytest.fixture
def transcript_cases(app_root):
    return json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))["cases"]


def _utterances(count=10):
    return [
        {"speaker": "Speaker A" if index % 2 == 0 else "Speaker B", "text": f"Line {index}.", "order_index": index}
        for index in range(count)
    ]


def _generation_response(summary, facts):
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[
            types.Part(text=json.dumps({"summary": summary, "facts": facts}))
        ]))
    ])


def test_sha256_goldens_lock_exact_utf8_text_and_utterance_inputs(transcript_cases):
    from services.vector_sync.app.hashing import sha256_text, sha256_utterance

    full = transcript_cases["valid_full"]["raw"]
    changed = transcript_cases["valid_minimally_changed"]["raw"]
    assert sha256_text(full) == "215dc0bd8ee293f333847e6669ce76bdafdec48ce39f4b80ab851df13c31d3b3"
    assert sha256_text(changed) == "1240514f71644a0d4c4d063475d273e3cfb695898a2cf9b4e30472f5f7545964"
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_text("line\n") == "c73b73af8851e9e91bc6b4dc12e7dace0a2bfb931c1d0b8b36ef367319f58cd1"
    assert sha256_text("caf\u00e9 \U0001f9e0") == "b617b1833dc911f285a15a11d6f1e372055cf351d2e3544538ad7886dd5f97db"
    assert sha256_text("synthetic-folder-ops") == "0bcf77c60f7b290aad8540b7672b0288e91c6605aaa327fd31c2279da3c6f677"

    original = (4, "Fixture Speaker A", "The first review is on Tuesday.")
    assert sha256_utterance(*original) == "8eb5dc89b8580288780b3eb33ca731760464a91df1d65236a36fd98c1cd04de0"
    assert sha256_utterance(5, original[1], original[2]) == "6c8f455561ef3cbf16a2719ca52554dbb36a8cc4301985b8e9923a125071d40f"
    assert sha256_utterance(original[0], original[1], "The first review is on Wednesday.") == "7ff79fe987739b48a53ab75d8521d090ae36f5119e9eeb18eef5985481219f88"


def test_utterance_payload_has_frozen_fields_and_date_conversion(transcript_cases):
    raw = transcript_cases["valid_full"]["raw"]
    _, utterances = parse_document(raw)
    original_utterances = deepcopy(utterances)
    payload_input = utterances[:2]
    original_payload_input = deepcopy(payload_input)
    payloads = processor.build_utterance_payloads(
        payload_input,
        doc_id="synthetic-doc-main",
        version="v2.0.0",
        parent_id="synthetic-folder-ops",
        dialog_date="2026-01-12",
        root_folder_id="pytest-synthetic-root-id",
    )
    assert payloads == [
        {
            "type": "utterance",
            "root_folder_id": "pytest-synthetic-root-id",
            "doc_id": "synthetic-doc-main",
            "version": "v2.0.0",
            "parent_id": "synthetic-folder-ops",
            "dialog_date": "2026-01-12",
            "dialog_date_num": 20260112,
            "speaker": "Fixture Speaker A",
            "text": "We will review the sample schedule.",
            "order_index": 0,
        },
        {
            "type": "utterance",
            "root_folder_id": "pytest-synthetic-root-id",
            "doc_id": "synthetic-doc-main",
            "version": "v2.0.0",
            "parent_id": "synthetic-folder-ops",
            "dialog_date": "2026-01-12",
            "dialog_date_num": 20260112,
            "speaker": "Fixture Speaker B",
            "text": "The draft has three sections.",
            "order_index": 1,
        },
    ]
    assert utterances == original_utterances
    assert payload_input == original_payload_input
    assert processor.build_utterance_payloads([], "doc", "v2", "parent", "2026-01-12", "root") == []
    assert processor.build_utterance_payloads(
        [{"speaker": "Speaker A", "text": "Text", "order_index": 0}],
        "doc", "v2", "parent", "not-a-date", "root",
    )[0]["dialog_date_num"] is None
    assert processor.build_utterance_payloads(
        [{"speaker": "Speaker A", "text": "Text", "order_index": 0}],
        "doc", "v2", "parent", "", "root",
    )[0]["dialog_date_num"] is None


def test_iter_windows_preserves_centers_payloads_and_generation_configuration(sdk_fakes, transcript_cases):
    raw = transcript_cases["valid_full"]["raw"]
    _, utterances = parse_document(raw)
    original_utterances = deepcopy(utterances)
    sdk_fakes.gemini.queue("generate_content", response=_generation_response("Center three summary.", ["Review is Tuesday."]))
    sdk_fakes.gemini.queue("generate_content", response=_generation_response("Center six summary.", []))
    sdk_fakes.gemini.queue("generate_content", response=_generation_response("Center nine summary.", []))

    generated = list(processor.iter_windows(
        utterances,
        doc_id="synthetic-doc-main",
        version="v2.0.0",
        parent_id="synthetic-folder-ops",
        dialog_date="2026-01-12",
        root_folder_id="pytest-synthetic-root-id",
        existing_summary_keys={"synthetic-doc-main:0:v2.0.0"},
    ))
    assert generated == [
        (
            {
                "type": "summary", "root_folder_id": "pytest-synthetic-root-id",
                "summary_id": "synthetic-doc-main:3:v2.0.0", "text": "Center three summary.",
                "doc_id": "synthetic-doc-main", "center_index": 3, "covered_range": [1, 5],
                "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12",
                "dialog_date_num": 20260112, "version": "v2.0.0",
            },
            [{
                "type": "fact", "text": "Review is Tuesday.", "root_folder_id": "pytest-synthetic-root-id",
                "doc_id": "synthetic-doc-main", "summary_id": "synthetic-doc-main:3:v2.0.0",
                "center_index": 3, "covered_range": [1, 5], "parent_id": "synthetic-folder-ops",
                "dialog_date": "2026-01-12", "dialog_date_num": 20260112, "version": "v2.0.0",
            }],
        ),
        (
            {
                "type": "summary", "root_folder_id": "pytest-synthetic-root-id",
                "summary_id": "synthetic-doc-main:6:v2.0.0", "text": "Center six summary.",
                "doc_id": "synthetic-doc-main", "center_index": 6, "covered_range": [4, 8],
                "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12",
                "dialog_date_num": 20260112, "version": "v2.0.0",
            },
            [],
        ),
        (
            {
                "type": "summary", "root_folder_id": "pytest-synthetic-root-id",
                "summary_id": "synthetic-doc-main:9:v2.0.0", "text": "Center nine summary.",
                "doc_id": "synthetic-doc-main", "center_index": 9, "covered_range": [7, 9],
                "parent_id": "synthetic-folder-ops", "dialog_date": "2026-01-12",
                "dialog_date_num": 20260112, "version": "v2.0.0",
            },
            [],
        ),
    ]
    assert utterances == original_utterances
    calls = [event["arguments"] for event in sdk_fakes.trace if event["operation"] == "generate_content"]
    assert len(calls) == 3
    expected_dialogues = [
        "Fixture Speaker B: The draft has three sections.\n"
        "Fixture Speaker A: Each section needs a clear owner.\n"
        "Fixture Speaker B: We can track changes in a small table.\n"
        "Fixture Speaker A: The first review is on Tuesday.\n"
        "Fixture Speaker B: Notes should include the decision.",
        "Fixture Speaker A: The first review is on Tuesday.\n"
        "Fixture Speaker B: Notes should include the decision.\n"
        "Fixture Speaker A: A second pass can check the wording.\n"
        "Fixture Speaker B: We will keep the example data invented.\n"
        "Fixture Speaker A: The final copy can stay brief.",
        "Fixture Speaker B: We will keep the example data invented.\n"
        "Fixture Speaker A: The final copy can stay brief.\n"
        "Fixture Speaker B: We will record the next step.",
    ]
    expected_prompt_hashes = [
        "4cc5b73bb6fd9ba8c72f2c6e8caaca667a939a269e0ca4b1986b0bb5e2e435d6",
        "1cd106cfd129704972174125fbe81e05b0a54c6845acc61e758ddf65234afffa",
        "cf7c5ab976ef2265f26b254671a6a81421f3448fb2fe91a715156cc7f3e4ab7d",
    ]
    for call, expected_dialogue, expected_prompt_hash in zip(calls, expected_dialogues, expected_prompt_hashes):
        assert call["model"] == "gemini-2.5-flash"
        assert call["contents"].split("Dialogue:\n", 1)[1] == expected_dialogue
        assert hashlib.sha256(call["contents"].encode("utf-8")).hexdigest() == expected_prompt_hash
        assert "Always reply in English regardless of the dialogue language." in call["contents"]
        assert call["config"].temperature == 0
        assert call["config"].top_p == 1
        assert call["config"].response_mime_type == "application/json"
        assert call["config"].response_schema == {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "facts"],
        }


def test_window_filters_empty_input_and_non_iso_date_payload(sdk_fakes):
    assert list(processor.iter_windows([], "doc", "v2", "parent", "2026-01-01", "root")) == []
    assert sdk_fakes.trace == []

    sdk_fakes.gemini.queue("generate_content", response=_generation_response("", []))
    result = list(processor.iter_windows(
        _utterances(1), "doc", "v2", "parent", "20260101", "root", allowed_center_indexes={0}
    ))
    assert result == [({
        "type": "summary", "root_folder_id": "root", "summary_id": "doc:0:v2", "text": "",
        "doc_id": "doc", "center_index": 0, "covered_range": [0, 0], "parent_id": "parent",
        "dialog_date": "20260101", "dialog_date_num": 20260101, "version": "v2",
    }, [])]


def test_empty_gemini_window_is_recorded_and_later_windows_continue(sdk_fakes):
    utterances = _utterances()
    original_utterances = deepcopy(utterances)
    empty_response = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)])
    sdk_fakes.gemini.queue("generate_content", response=empty_response)
    sdk_fakes.gemini.queue("generate_content", response=empty_response)
    sdk_fakes.gemini.queue("generate_content", response=_generation_response("After empty response.", []))
    skipped = [9]
    result = list(processor.iter_windows(
        utterances, "doc", "v1", "parent", "2026-04-05", "root",
        allowed_center_indexes={0, 3, 6}, skipped_utterances=skipped,
    ))
    assert skipped == [9, 0, 1, 2, 1, 2, 3, 4, 5]
    assert result == [({
        "type": "summary", "root_folder_id": "root", "summary_id": "doc:6:v1",
        "text": "After empty response.", "doc_id": "doc", "center_index": 6,
        "covered_range": [4, 8], "parent_id": "parent", "dialog_date": "2026-04-05",
        "dialog_date_num": 20260405, "version": "v1",
    }, [])]
    assert utterances == original_utterances
    assert len([e for e in sdk_fakes.trace if e["boundary"] == "gemini" and e["operation"] == "generate_content"]) == 3


@pytest.mark.parametrize("failure", [
    ValueError("scripted Gemini failure"),
    json.JSONDecodeError("malformed response", "not json", 0),
])
def test_generation_failures_propagate_without_skipping_or_mutation(sdk_fakes, failure):
    utterances = _utterances(1)
    original_utterances = deepcopy(utterances)
    skipped = [7]
    if isinstance(failure, json.JSONDecodeError):
        response = types.GenerateContentResponse(candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(text="not json")]))
        ])
        sdk_fakes.gemini.queue("generate_content", response=response)
    else:
        sdk_fakes.gemini.queue("generate_content", error=failure)

    with pytest.raises(type(failure)) as exc_info:
        list(processor.iter_windows(
            utterances, "doc", "v1", "parent", "2026-04-05", "root",
            allowed_center_indexes={0}, skipped_utterances=skipped,
        ))
    if isinstance(failure, json.JSONDecodeError):
        assert exc_info.value.msg == "Expecting value"
        assert exc_info.value.doc == "not json"
        assert exc_info.value.pos == 0
    else:
        assert exc_info.value is failure
    assert skipped == [7]
    assert utterances == original_utterances
    assert len([e for e in sdk_fakes.trace if e["boundary"] == "gemini" and e["operation"] == "generate_content"]) == 1


def test_generation_caps_facts_at_ten(sdk_fakes):
    facts = [f"Fact {index}" for index in range(11)]
    sdk_fakes.gemini.queue("generate_content", response=_generation_response("Many facts.", facts))
    summary, returned = llm.generate_summary_and_facts(_utterances(1))
    assert summary == "Many facts."
    assert returned == [f"Fact {index}" for index in range(10)]


def test_embeddings_use_gemini_embedding_2_with_768_dimensions(sdk_fakes):
    values = [0.125] * 768
    sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=values)]
    ))
    assert embeddings.embed(["Synthetic embedding input."]) == [values]
    call = next(event["arguments"] for event in sdk_fakes.trace if event["operation"] == "embed_content")
    assert call["model"] == "gemini-embedding-2"
    assert call["contents"] == ["Synthetic embedding input."]
    assert call["config"].output_dimensionality == 768
