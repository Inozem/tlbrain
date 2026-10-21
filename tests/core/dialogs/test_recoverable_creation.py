"""Offline contract for recoverable native Google Doc creation and filling.

These tests intentionally import the future seams inside helpers so the test
suite can collect before the dialog creation/fill implementation is added.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from googleapiclient.errors import HttpError

from core.integrations.firestore.operations import claim_operation, create_operation
from core.integrations.firestore.save_intents import (
    create_save_intent,
    dispatch_save_intent,
    get_save_intent,
    reconcile_save_intent,
)
from core.operations.models import OperationOrigin, OperationResult


NOW = datetime(2026, 4, 2, tzinfo=timezone.utc)
MARKER_KEY = "tlbrain_save_id"
SAVE_ID = "save-intent-a"
DOC_ID = "native-doc-a"
TITLE = "Synthetic dialog title"
PARENT_ID = "synthetic-folder"
CONTENT = "DATE: 2026-04-02\nPROVIDER: manual\n---\nAda :: hello\n"


def _reconcile(drive, *, save_intent_id: str = SAVE_ID, expected_parent_id: str = PARENT_ID):
    # The component seam searches Drive by the immutable marker and validates
    # native Doc type and parent before adopting a candidate.
    from tests.fixtures.sdk_fakes import fake_discovery_build
    from core.integrations.google_docs.creation import reconcile_google_doc_by_marker

    return reconcile_google_doc_by_marker(
        fake_discovery_build("drive", "v3"), save_intent_id=save_intent_id,
        expected_parent_id=expected_parent_id,
    )


def _create(drive, *, save_intent_id: str = SAVE_ID):
    # The component seam creates only a native Doc carrying the save marker.
    from tests.fixtures.sdk_fakes import fake_discovery_build
    from core.integrations.google_docs.creation import create_marked_google_doc

    return create_marked_google_doc(
        fake_discovery_build("drive", "v3"),
        save_intent_id=save_intent_id,
        parent_id=PARENT_ID,
        title=TITLE,
    )


def _fill(*, document_id: str = DOC_ID, expected_revision_id: str = "rev-1"):
    # The component seam performs revision-guarded insert and readback recovery.
    from tests.fixtures.sdk_fakes import fake_discovery_build
    from core.integrations.google_docs.fill import fill_google_doc_safely

    return fill_google_doc_safely(
        fake_discovery_build("docs", "v1"),
        drive=fake_discovery_build("drive", "v3"),
        document_id=document_id,
        text=CONTENT,
        speaker_ranges=[(31, 34)],
        expected_revision_id=expected_revision_id,
    )


def _candidate(document_id: str, *, marker: str = SAVE_ID, parent: str = PARENT_ID, mime: str = "application/vnd.google-apps.document"):
    return {
        "id": document_id,
        "name": TITLE,
        "mimeType": mime,
        "parents": [parent],
        "trashed": False,
        "appProperties": {MARKER_KEY: marker},
    }


def _drive_search(sdk_fakes, candidates):
    sdk_fakes.drive.queue("files.list", response={"files": candidates})


def _new_intent(firestore_fake, *, operation_id: str = "operation-a"):
    client = firestore_fake.client_class()
    create_operation(
        client,
        operation_id=operation_id,
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:immutable-dialog",
        immutable_input_ref="synthetic-input-ref",
        provider="manual",
        title=TITLE,
        now=NOW,
    )
    create_save_intent(
        client,
        SAVE_ID,
        operation_id=operation_id,
        desired_content_hash="sha256:dialog-content",
        desired_parent_id_hash="sha256:folder",
        now=NOW,
    )
    return client


def test_planned_intent_resumes_before_dispatch_and_creates_once(firestore_fake, sdk_fakes):
    client = _new_intent(firestore_fake)
    assert sdk_fakes.trace == []
    crashed_lease = claim_operation(client, "operation-a", owner="crashed-worker", now=NOW, lease_seconds=30)
    assert crashed_lease is not None
    resumed_at = NOW + timedelta(seconds=31)
    lease = claim_operation(client, "operation-a", owner="resumed-worker", now=resumed_at, lease_seconds=30)
    assert lease is not None
    assert lease.fence > crashed_lease.fence
    assert sdk_fakes.trace == []
    sdk_fakes.drive.queue("files.create", response={"id": DOC_ID, "modifiedTime": "synthetic-time"})

    result = dispatch_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-a",
        owner="resumed-worker",
        fence=lease.fence,
        now=resumed_at,
        create_callback=lambda: _create(sdk_fakes.drive),
    )

    assert result.document_id == DOC_ID
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 1
    assert get_save_intent(client, SAVE_ID).state.value == "confirmed"
    assert dispatch_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-a",
        owner="resumed-worker",
        fence=lease.fence,
        now=resumed_at,
        create_callback=lambda: pytest.fail("confirmed intent dispatched a second create"),
    ) == result
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 1


def test_definitive_drive_rejection_is_distinct_from_unknown_timeout(sdk_fakes):
    rejection = HttpError(SimpleNamespace(status=403, reason="Forbidden"), b"synthetic rejection")
    sdk_fakes.drive.queue("files.create", error=rejection)

    with pytest.raises(Exception) as rejected:
        _create(sdk_fakes.drive)

    assert getattr(rejected.value, "outcome", None) == "rejected"
    assert [event["operation"] for event in sdk_fakes.trace if event["operation"].startswith("files.")] == ["files.create"]

    sdk_fakes.drive.queue("files.create", error=TimeoutError("provider response unavailable"))
    with pytest.raises(Exception) as unknown:
        _create(sdk_fakes.drive)

    assert getattr(unknown.value, "outcome", None) == "unknown"
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 2


def test_creation_reconciles_only_a_valid_marker_match(sdk_fakes):
    _drive_search(
        sdk_fakes,
        [
            _candidate("wrong-parent", parent="other-folder"),
            _candidate("wrong-type", mime="text/plain"),
            _candidate("trashed-doc") | {"trashed": True},
            _candidate(DOC_ID),
        ],
    )

    result = _reconcile(sdk_fakes.drive)

    assert result.state == "confirmed"
    assert result.document_id == DOC_ID
    search = next(event for event in sdk_fakes.trace if event["operation"] == "files.list")
    assert MARKER_KEY in search["arguments"]["q"]
    assert SAVE_ID in search["arguments"]["q"]
    assert "name" not in search["arguments"]["q"]
    assert "fullText" not in search["arguments"]["q"]
    assert [event for event in sdk_fakes.trace if event["operation"].startswith("files.")] == [search]


def test_creation_with_no_marker_match_stays_unknown_and_does_not_create(sdk_fakes):
    _drive_search(sdk_fakes, [])

    result = _reconcile(sdk_fakes.drive)

    assert result.state == "creation_outcome_unknown"
    assert result.document_id is None
    assert [event["operation"] for event in sdk_fakes.trace if event["operation"].startswith("files.")] == ["files.list"]


def test_multiple_marker_matches_are_an_invariant_error_without_deletion(sdk_fakes):
    _drive_search(sdk_fakes, [_candidate("doc-a"), _candidate("doc-b")])

    with pytest.raises(ValueError, match="multiple"):
        _reconcile(sdk_fakes.drive)

    assert [event["operation"] for event in sdk_fakes.trace if event["operation"].startswith("files.")] == ["files.list"]
    assert not any(event["operation"] == "files.delete" for event in sdk_fakes.trace)


def test_create_dispatch_persists_marker_with_parent_and_title(sdk_fakes):
    sdk_fakes.drive.queue("files.create", response={"id": DOC_ID, "modifiedTime": "synthetic-time"})

    result = _create(sdk_fakes.drive)

    assert result.document_id == DOC_ID
    create = next(event for event in sdk_fakes.trace if event["operation"] == "files.create")
    body = create["arguments"]["body"]
    assert body["mimeType"] == "application/vnd.google-apps.document"
    assert body["parents"] == [PARENT_ID]
    assert body["name"] == TITLE
    assert body["appProperties"] == {MARKER_KEY: SAVE_ID}
    assert "id" not in body


@pytest.mark.parametrize("matches", [[], [_candidate("doc-a"), _candidate("doc-b")]])
def test_manual_retry_reuses_intent_and_unknown_gate_after_marker_reconciliation(
    firestore_fake, sdk_fakes, matches
):
    client = firestore_fake.client_class()
    create_operation(
        client,
        operation_id="operation-a",
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:immutable-dialog",
        immutable_input_ref="synthetic-input-ref",
        provider="manual",
        title=TITLE,
        now=NOW,
    )
    create_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-a",
        desired_content_hash="sha256:dialog-content",
        desired_parent_id_hash="sha256:folder",
        now=NOW,
    )
    first_lease = claim_operation(client, "operation-a", owner="first", now=NOW, lease_seconds=30)
    assert first_lease is not None
    sdk_fakes.drive.queue("files.create", error=RuntimeError("provider outcome unavailable"))

    with pytest.raises(Exception) as create_error:
        dispatch_save_intent(
            client,
            SAVE_ID,
            operation_id="operation-a",
            owner="first",
            fence=first_lease.fence,
            now=NOW,
            create_callback=lambda: _create(sdk_fakes.drive),
        )
    assert getattr(create_error.value, "outcome", None) == "unknown"
    assert "provider outcome unavailable" not in str(create_error.value)

    _drive_search(sdk_fakes, matches)
    if matches:
        with pytest.raises(ValueError, match="multiple"):
            _reconcile(sdk_fakes.drive)
    else:
        unresolved = _reconcile(sdk_fakes.drive)
        assert unresolved.state == "creation_outcome_unknown"
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 1

    create_operation(
        client,
        operation_id="operation-b",
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:immutable-dialog",
        immutable_input_ref="synthetic-input-ref",
        provider="manual",
        title=TITLE,
        save_intent_id=SAVE_ID,
        now=NOW,
    )
    create_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-b",
        desired_content_hash="sha256:dialog-content",
        desired_parent_id_hash="sha256:folder",
        now=NOW,
    )
    retry_lease = claim_operation(client, "operation-b", owner="retry", now=NOW, lease_seconds=30)
    assert retry_lease is not None

    assert dispatch_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-b",
        owner="retry",
        fence=retry_lease.fence,
        now=NOW,
        create_callback=lambda: pytest.fail("unknown create gate was reopened by manual retry"),
    ) is None
    intent = get_save_intent(client, SAVE_ID)
    assert intent.state.value == "unknown"
    assert intent.dispatch_operation_id == "operation-a"
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 1
    assert [event["operation"] for event in sdk_fakes.trace].count("files.list") == 1
    assert firestore_fake.documents["dialog_save_intents/" + SAVE_ID]["desired_content_hash"] == "sha256:dialog-content"


def test_unknown_create_adopts_one_marker_match_and_retry_returns_same_doc(firestore_fake, sdk_fakes):
    client = _new_intent(firestore_fake)
    first_lease = claim_operation(client, "operation-a", owner="first", now=NOW, lease_seconds=30)
    assert first_lease is not None
    sdk_fakes.drive.queue("files.create", error=TimeoutError("provider response unavailable"))

    with pytest.raises(Exception) as create_error:
        dispatch_save_intent(
            client,
            SAVE_ID,
            operation_id="operation-a",
            owner="first",
            fence=first_lease.fence,
            now=NOW,
            create_callback=lambda: _create(sdk_fakes.drive),
        )
    assert getattr(create_error.value, "outcome", None) == "unknown"
    assert "provider response unavailable" not in str(create_error.value)

    retry_at = NOW + timedelta(seconds=31)
    create_operation(
        client,
        operation_id="operation-b",
        origin=OperationOrigin.MCP,
        caller_binding="synthetic-caller",
        immutable_input_digest="sha256:immutable-dialog",
        immutable_input_ref="synthetic-input-ref",
        provider="manual",
        title=TITLE,
        save_intent_id=SAVE_ID,
        now=retry_at,
    )
    create_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-b",
        desired_content_hash="sha256:dialog-content",
        desired_parent_id_hash="sha256:folder",
        now=retry_at,
    )
    retry_lease = claim_operation(client, "operation-b", owner="retry", now=retry_at, lease_seconds=30)
    assert retry_lease is not None
    _drive_search(sdk_fakes, [_candidate(DOC_ID)])

    recovered = _reconcile(sdk_fakes.drive)
    assert recovered.state == "confirmed"
    assert recovered.document_id == DOC_ID
    confirmed = reconcile_save_intent(
        client,
        SAVE_ID,
        owner="retry",
        fence=retry_lease.fence,
        now=retry_at,
        external_result=OperationResult(document_id=DOC_ID, title=TITLE),
        operation_id="operation-b",
    )
    assert confirmed.state.value == "confirmed"
    assert confirmed.dispatch_operation_id == "operation-a"

    result = dispatch_save_intent(
        client,
        SAVE_ID,
        operation_id="operation-b",
        owner="retry",
        fence=retry_lease.fence,
        now=retry_at,
        create_callback=lambda: pytest.fail("marker-recovered intent dispatched a second create"),
    )

    assert result.document_id == DOC_ID
    assert get_save_intent(client, SAVE_ID).doc_id == DOC_ID
    assert [event["operation"] for event in sdk_fakes.trace].count("files.create") == 1
    assert [event["operation"] for event in sdk_fakes.trace].count("files.list") == 1


def test_lost_docs_insert_response_reads_back_exact_text_without_retrying_insert(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-1", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )
    sdk_fakes.docs.queue(
        "documents.batchUpdate",
        error=RuntimeError("Docs response unavailable"),
    )
    sdk_fakes.docs.queue(
        "documents.get",
        response={
            "revisionId": "rev-2",
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": CONTENT + "\n"}}]}}
                ]
            },
        },
    )
    sdk_fakes.drive.queue(
        "files.get",
        response={"id": DOC_ID, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "2026-04-02T00:00:00Z"},
    )

    result = _fill()

    assert result.state == "confirmed"
    assert result.document_id == DOC_ID
    assert result.revision_id == "rev-2"
    assert result.modified_time == "2026-04-02T00:00:00Z"
    operations = [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"]
    assert operations == [
        "documents.get",
        "documents.batchUpdate",
        "documents.get",
        "files.get",
    ]
    update = next(event["arguments"] for event in sdk_fakes.trace if event["operation"] == "documents.batchUpdate")
    assert update["writeControl"]["requiredRevisionId"] == "rev-1"
    assert update["body"]["requests"][0]["insertText"]["text"] == CONTENT


def test_revision_conflict_is_reread_and_reported_without_blind_overwrite(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-1", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )
    sdk_fakes.docs.queue("documents.batchUpdate", error=RuntimeError("revision conflict"))
    sdk_fakes.docs.queue(
        "documents.get",
        response={
            "revisionId": "rev-foreign",
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": "foreign edit\n"}}]}}
                ]
            },
        },
    )

    result = _fill()

    assert result.state == "revision_conflict"
    assert result.document_id == DOC_ID
    operations = [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"]
    assert operations == [
        "documents.get",
        "documents.batchUpdate",
        "documents.get",
    ]
    update = next(event["arguments"] for event in sdk_fakes.trace if event["operation"] == "documents.batchUpdate")
    assert update["writeControl"]["requiredRevisionId"] == "rev-1"


def test_exact_doc_confirms_without_write_and_reads_modified_time(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-exact", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": CONTENT + "\n"}}]}}
        ]}},
    )
    sdk_fakes.drive.queue(
        "files.get",
        response={"id": DOC_ID, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "modified"},
    )

    result = _fill()

    assert (result.state, result.revision_id, result.modified_time) == ("confirmed", "rev-exact", "modified")
    assert [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"] == ["documents.get", "files.get"]


def test_document_text_preserves_desired_trailing_newline_after_structural_newline():
    from core.integrations.google_docs.fill import _document_text

    text, revision = _document_text({
        "revisionId": "rev-exact",
        "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": CONTENT + "\n"}}]}}
        ]},
    })

    assert text == CONTENT
    assert text.endswith("\n")
    assert revision == "rev-exact"


def test_empty_doc_is_inserted_once_with_guard_and_post_fill_metadata(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-1", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )
    sdk_fakes.docs.queue("documents.batchUpdate", response={"revisionId": "rev-2"})
    sdk_fakes.drive.queue(
        "files.get",
        response={"id": DOC_ID, "mimeType": "application/vnd.google-apps.document", "modifiedTime": "modified"},
    )

    result = _fill()

    assert (result.state, result.revision_id, result.modified_time) == ("confirmed", "rev-2", "modified")
    updates = [event for event in sdk_fakes.trace if event["operation"] == "documents.batchUpdate"]
    assert len(updates) == 1
    update = updates[0]["arguments"]
    assert update["writeControl"]["requiredRevisionId"] == "rev-1"
    assert update["body"]["requests"][0]["insertText"] == {"location": {"index": 1}, "text": CONTENT}
    assert update["body"]["requests"][1]["updateTextStyle"]["range"] == {"startIndex": 32, "endIndex": 35}


def test_foreign_content_before_write_conflicts_without_mutation(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-foreign", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "foreign text\n"}}]}}
        ]}},
    )

    result = _fill()

    assert result.state == "revision_conflict"
    assert [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"] == ["documents.get"]


def test_empty_doc_with_changed_revision_conflicts_without_mutation(sdk_fakes):
    sdk_fakes.docs.queue(
        "documents.get",
        response={"revisionId": "rev-newer", "body": {"content": [
            {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
        ]}},
    )

    result = _fill()

    assert result.state == "revision_conflict"
    assert [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"] == ["documents.get"]


def test_unresolved_docs_insert_response_remains_unknown_without_retry(sdk_fakes):
    empty_doc = {"revisionId": "rev-1", "body": {"content": [
        {"paragraph": {"elements": [{"textRun": {"content": "\n"}}]}}
    ]}}
    sdk_fakes.docs.queue("documents.get", response=empty_doc)
    sdk_fakes.docs.queue("documents.batchUpdate", error=RuntimeError("Docs response unavailable"))
    sdk_fakes.docs.queue("documents.get", response=empty_doc)

    result = _fill()

    assert result.state == "fill_outcome_unknown"
    operations = [event["operation"] for event in sdk_fakes.trace if event["operation"] != "build"]
    assert operations == ["documents.get", "documents.batchUpdate", "documents.get"]

