"""Contract tests for Telegram save and retention.

The future-facing boundary is ``TelegramSaveWorkflow`` in
``services.connectors.telegram.save``. It receives a repository, an audio
transcriber implementing ``process_messages(messages, *, draft_id, analyze)``,
and injectable preprocessing, analysis, classification, save, and completion
seams. ``TelegramDraftCleanup`` in ``services.connectors.telegram.cleanup``
loads authoritative completion certificates and draft manifests from its
repository before deleting source pages. These tests describe observable
workflow behavior without prescribing a Firestore schema.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import os
import socket
from datetime import datetime, timezone

import pytest


pytestmark = pytest.mark.offline


def _value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _selector_loop():
    """Use loopback datagrams for asyncio's Windows self-pipe fallback."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    old_socketpair = socket.socketpair
    local_pairs = []

    def local_datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            local_pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_datagram_pair
    try:
        return asyncio.SelectorEventLoop()
    except BaseException:
        for reader, writer in local_pairs:
            reader.close()
            writer.close()
        raise
    finally:
        socket.socketpair = old_socketpair


def _save_type():
    return importlib.import_module(
        "services.connectors.telegram.save"
    ).TelegramSaveWorkflow


def _cleanup_type():
    return importlib.import_module(
        "services.connectors.telegram.cleanup"
    ).TelegramDraftCleanup


def _message(message_id, text, *, sequence=None, audio=False):
    row = {
        "message_id": message_id,
        "intake_sequence": message_id if sequence is None else sequence,
        "text": text,
        "received_at": datetime(2026, 10, 20, tzinfo=timezone.utc),
        "original_date": datetime(2026, 10, 19, tzinfo=timezone.utc),
    }
    if audio:
        row["audio_file_id"] = f"synthetic-file-{message_id}"
    return row


class _Repository:
    def __init__(self, *, draft_id="draft-original", messages=None):
        messages = messages or [_message(1, "sealed source", audio=True)]
        self.primary_draft_id = draft_id
        self.active_draft_id = draft_id
        self.drafts = {draft_id: {
            "draft_id": draft_id,
            "status": "sealed",
            "messages": copy.deepcopy(messages),
            "source_manifest": copy.deepcopy(messages),
            "manifest_digest": f"manifest:{draft_id}:v1",
        }}
        self.guard = {"resource_version": 7, "current": True}
        self.certificate = None
        self.cleanup_authorization = None
        self.checkpoints = {}
        self.deleted = []
        self.deleted_pages = []
        self.doc_text = "saved document before independent edit"
        self.doc_writes = []
        self.commit_error = None
        self.previous_dialogue_context = []
        self.previous_context_reads = []

    @property
    def draft(self):
        """Compatibility accessor for the original draft used by older tests."""
        return self.drafts[self.primary_draft_id]

    def add_draft(self, draft, *, make_active=True):
        self.drafts[draft["draft_id"]] = copy.deepcopy(draft)
        if make_active:
            self.active_draft_id = draft["draft_id"]

    async def load_previous_dialogue_context(self, draft_id):
        self.previous_context_reads.append(draft_id)
        return copy.deepcopy(self.previous_dialogue_context)

    async def load_sealed_draft(self, draft_id):
        draft = self.drafts.get(draft_id)
        assert draft is not None
        return copy.deepcopy(draft)

    async def checkpoint_transcription(self, draft_id, message_id, text):
        assert draft_id in self.drafts
        self.checkpoints[message_id] = text

    async def load_current_guard(self, draft_id):
        return copy.deepcopy(self.guard)

    async def verify_save_receipt(self, receipt, *, draft_id, manifest_digest, guard):
        return (
            receipt.get("verified") is True
            and receipt.get("draft_id") == draft_id
            and receipt.get("manifest_digest") == manifest_digest
            and guard == self.guard
            and guard.get("current") is True
        )

    async def commit_completion(self, certificate):
        if self.commit_error:
            raise self.commit_error
        self.certificate = copy.deepcopy(certificate)
        return True

    async def load_completion_certificate(self, context_id):
        return copy.deepcopy(self.certificate)

    async def authorize_cleanup(self, context_id, certificate):
        draft = self.drafts.get(_value(certificate, "draft_id"))
        if (draft is None or _value(certificate, "context_id") != context_id
                or _value(certificate, "draft_id") != draft["draft_id"]
                or _value(certificate, "manifest_digest") != draft["manifest_digest"]
                or self.certificate != certificate):
            return False
        if self.cleanup_authorization is None:
            self.cleanup_authorization = copy.deepcopy(certificate)
        return self.cleanup_authorization == certificate

    async def finalize_cleanup(self, context_id, certificate):
        if self.cleanup_authorization != certificate:
            return False
        draft = self.drafts.get(_value(certificate, "draft_id"))
        if draft is not None:
            draft["cleanup_state"] = "cleaned"
        if self.active_draft_id == _value(certificate, "draft_id"):
            self.active_draft_id = None
        return True

    async def load_draft(self, context_id, draft_id):
        draft = self.drafts.get(draft_id)
        return copy.deepcopy(draft) if draft is not None else None

    async def list_manifest_pages(self, draft_id, *, cursor, limit):
        draft = self.drafts.get(draft_id)
        if draft is None:
            return [], None
        messages = draft["messages"]
        offset = cursor or 0
        page = messages[offset:offset + limit]
        next_cursor = offset + len(page) if offset + len(page) < len(messages) else None
        return copy.deepcopy(page), next_cursor

    async def delete_draft_page(self, draft_id, messages):
        self.deleted_pages.append((draft_id, copy.deepcopy(messages)))
        self.deleted.extend((draft_id, row["message_id"]) for row in messages)
        draft = self.drafts.get(draft_id)
        if draft is not None:
            removed = {row["message_id"] for row in messages}
            draft["messages"] = [
                row for row in draft["messages"] if row["message_id"] not in removed
            ]

    async def list_prepared_pages(self, context_id, draft_id, *, limit):
        return []

    async def delete_prepared_page(self, context_id, draft_id, page):
        return None


class _Audio:
    def __init__(self, repository, events, *, fail=False, transcripts=None):
        self.repository = repository
        self.events = events
        self.fail = fail
        self.transcripts = transcripts or {}

    async def process_messages(self, messages, *, draft_id, analyze):
        self.events.append("audio:start")
        for row in sorted(messages, key=lambda item: item["intake_sequence"]):
            if row.get("audio_file_id"):
                if self.fail:
                    raise RuntimeError("synthetic transcription failure")
                transcript = self.transcripts.get(row["message_id"], "synthetic transcript")
                await self.repository.checkpoint_transcription(
                    draft_id, row["message_id"], transcript
                )
                row["transcription"] = transcript
        self.events.append("audio:checkpointed")
        return await analyze(messages)


class _Pipeline:
    def __init__(self, events, *, receipt=None, fail_save=False):
        self.events = events
        self.receipt = receipt
        self.fail_save = fail_save
        self.contexts = []
        self.saved_inputs = []

    async def preprocess(self, messages):
        self.events.append("preprocess")
        return list(messages)

    async def analyze(self, messages, *, context):
        self.events.append("analyze")
        self.contexts.append(copy.deepcopy(context))
        return {"items": [row["text"] for row in messages], "context": context}

    async def classify(self, result, *, draft_id):
        self.events.append("classify")
        return {"draft_id": draft_id, **result}

    async def save(self, result, *, draft_id, manifest_digest, intent_id, operation_id):
        self.events.append("save")
        self.saved_inputs.append(copy.deepcopy(result))
        if self.fail_save:
            raise RuntimeError("synthetic save failure")
        return copy.deepcopy(self.receipt)


def _workflow(repository, events, pipeline, *, audio=None):
    return _save_type()(
        repository=repository,
        audio_transcriber=audio or _Audio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=pipeline.analyze,
        classify=pipeline.classify,
        save=pipeline.save,
    )


def _receipt(repository, *, verified=True):
    return {
        "verified": verified,
        "draft_id": repository.draft["draft_id"],
        "manifest_digest": repository.draft["manifest_digest"],
        "resource_version": repository.guard["resource_version"],
        "save_intent_id": "intent-1",
        "operation_id": "operation-1",
        "receipt_ref": "receipt-ref-1",
        "receipt_version": 3,
    }


def _committed_certificate(repository, *, draft_id=None, manifest_digest=None):
    """Return a complete synthetic completion certificate for cleanup contracts."""
    draft_id = draft_id or repository.draft["draft_id"]
    manifest_digest = manifest_digest or repository.draft["manifest_digest"]
    return {
        "context_id": "context-1",
        "certificate_id": "certificate-1",
        "draft_id": draft_id,
        "manifest_digest": manifest_digest,
        "save_intent_id": "intent-1",
        "operation_id": "operation-1",
        "doc_id": "doc-1",
        "content_hash": "content-digest-1",
        "parent_id_hash": "parent-doc-1",
        "provider": "synthetic-provider",
        "receipt_ref": "receipt-ref-1",
        "receipt_version": 3,
        "index_fence": 11,
        "resource_epoch": "epoch-1",
        "fence": 7,
        "origin_binding": ("context-1", draft_id, manifest_digest),
        "verified_at": "2026-10-22T10:00:00Z",
        "completed_at": "2026-10-22T10:00:01Z",
    }


def test_audio_barrier_precedes_ordered_preprocessing_analysis_and_classification():
    messages = [
        _message(30, "third", sequence=3, audio=True),
        _message(10, "first", sequence=1),
        _message(20, "second", sequence=2, audio=True),
    ]
    repository = _Repository(messages=messages)
    original_manifest = copy.deepcopy(repository.draft["source_manifest"])
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)

    _run(workflow.save(draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"))

    assert events.index("audio:checkpointed") < events.index("preprocess")
    assert events.index("audio:checkpointed") < events.index("analyze") < events.index("classify")
    assert repository.checkpoints == {20: "synthetic transcript", 30: "synthetic transcript"}
    assert [row["message_id"] for row in pipeline.contexts[0]["messages"]] == [10, 20, 30]
    assert repository.draft["source_manifest"] == original_manifest


def test_topic_hint_and_explicit_folder_reach_classifier_without_changing_save_payload():
    from core.dialogs.classification import FolderSelectionResult
    from services.connectors.telegram.save import TelegramClassifiedDialog

    repository = _Repository()
    repository.context_id = "telegram-context-1"
    repository.get_context = lambda _context_id: {"topic_name": "Planning"}
    events = []
    pipeline = _Pipeline(events)
    calls = []

    async def classify(result, *, draft_id, topic_hint=None, folder_path=None, folder_id=None):
        calls.append((result, draft_id, topic_hint, folder_path, folder_id))
        return TelegramClassifiedDialog(
            payload={"title": "Planning", "utterances": []},
            selection=FolderSelectionResult(folder_id="folder-7"),
        )

    async def analyze(messages, *, context):
        return {"title": "Planning", "folder_path": ["Clients", "Planning"], "utterances": []}

    async def save(result, *, draft_id, manifest_digest, intent_id, operation_id):
        pipeline.saved_inputs.append(copy.deepcopy(result))
        return {
            "verified": True, "draft_id": draft_id, "manifest_digest": manifest_digest,
            "save_intent_id": intent_id, "operation_id": operation_id,
            "receipt_ref": "receipt-ref", "receipt_version": 1,
        }

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=_Audio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=analyze,
        classify=classify,
        save=save,
    )
    result = _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1",
    ))

    assert calls[0][2:] == ("Planning", ["Clients", "Planning"], None)
    assert pipeline.saved_inputs[0]["title"] == "Planning"
    assert "classification_outcome" not in pipeline.saved_inputs[0]
    assert _value(result, "classification_outcome") is None


def test_topic_hint_is_advisory_and_fallback_outcome_survives_completion():
    from core.dialogs.classification import FolderSelectionResult
    from services.connectors.telegram.save import TelegramClassifiedDialog

    repository = _Repository()
    repository.context_id = "telegram-context-1"
    repository.get_context = lambda _context_id: {"topic_name": "Shared topic"}
    second = copy.deepcopy(repository.draft)
    second.update({"draft_id": "draft-second", "manifest_digest": "manifest:draft-second:v1"})
    repository.add_draft(second, make_active=False)
    events = []
    pipeline = _Pipeline(events)
    seen = []

    async def classify(result, *, draft_id, topic_hint=None):
        seen.append((draft_id, topic_hint, result))
        return TelegramClassifiedDialog(
            payload={"title": result["title"], "utterances": []},
            selection=FolderSelectionResult(folder_id=None, requires_unassigned=True),
        )

    async def save(result, *, draft_id, manifest_digest, intent_id, operation_id):
        pipeline.saved_inputs.append(copy.deepcopy(result))
        return {
            "verified": True, "draft_id": draft_id, "manifest_digest": manifest_digest,
            "save_intent_id": intent_id, "operation_id": operation_id,
            "receipt_ref": "receipt-ref", "receipt_version": 1,
        }

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=_Audio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=lambda messages, *, context: {"title": context["draft_id"]},
        classify=classify,
        save=save,
    )
    first_certificate = _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1",
    ))
    second_certificate = _run(workflow.save(
        draft_id="draft-second", intent_id="intent-2", operation_id="operation-2",
    ))

    assert [(row[0], row[1]) for row in seen] == [
        ("draft-original", "Shared topic"), ("draft-second", "Shared topic"),
    ]
    assert [row[2]["title"] for row in seen] == ["draft-original", "draft-second"]
    assert _value(first_certificate, "classification_outcome") == "unassigned"
    assert _value(second_certificate, "classification_outcome") == "unassigned"


def test_ambiguous_selection_saves_only_payload_and_commits_bounded_outcome():
    from core.dialogs.classification import FolderSelectionResult
    from services.connectors.telegram.save import TelegramClassifiedDialog

    repository = _Repository()
    events = []
    pipeline = _Pipeline(events)

    async def classify(result, *, draft_id):
        return TelegramClassifiedDialog(
            payload={"title": "Planning", "utterances": []},
            selection=FolderSelectionResult(
                folder_id="folder-7", speaker_candidate_ids=("speaker-1", "speaker-2"),
            ),
        )

    async def save(result, *, draft_id, manifest_digest, intent_id, operation_id):
        pipeline.saved_inputs.append(copy.deepcopy(result))
        return {
            "verified": True, "draft_id": draft_id, "manifest_digest": manifest_digest,
            "save_intent_id": intent_id, "operation_id": operation_id,
            "receipt_ref": "receipt-ref", "receipt_version": 1,
        }

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=_Audio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=lambda messages, *, context: {"title": "Planning", "utterances": []},
        classify=classify,
        save=save,
    )
    certificate = _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1",
    ))

    saved_payload = pipeline.saved_inputs[0]
    assert saved_payload["title"] == "Planning"
    assert saved_payload["utterances"] == []
    assert "classification_outcome" not in saved_payload
    assert len(saved_payload) == 4  # Payload plus the workflow's provider/date fields.
    assert _value(certificate, "classification_outcome") == "ambiguous"
    assert _value(repository.certificate, "classification_outcome") == "ambiguous"


def test_explicit_folder_id_requires_classifier_support():
    from services.connectors.telegram.save import TelegramSaveError

    repository = _Repository()
    events = []
    pipeline = _Pipeline(events)

    async def classify(result, *, draft_id):
        return result

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=_Audio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=lambda messages, *, context: {"title": "Planning", "folder_id": "folder-7"},
        classify=classify,
        save=pipeline.save,
    )

    with pytest.raises(TelegramSaveError, match="Telegram save failed during classification"):
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1",
        ))


@pytest.mark.parametrize("failure", ["transcription", "save", "completion"])
def test_failure_preserves_raw_manifest_and_never_authorizes_cleanup(failure):
    repository = _Repository()
    original = copy.deepcopy((repository.draft["messages"], repository.draft["source_manifest"]))
    events = []
    pipeline = _Pipeline(
        events,
        receipt=_receipt(repository),
        fail_save=failure == "save",
    )
    audio = _Audio(repository, events, fail=failure == "transcription")
    if failure == "completion":
        repository.commit_error = RuntimeError("synthetic completion commit failure")
    workflow = _workflow(repository, events, pipeline, audio=audio)

    with pytest.raises(RuntimeError) as raised:
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
        ))

    assert (repository.draft["messages"], repository.draft["source_manifest"]) == original
    assert repository.certificate is None
    assert repository.deleted == []
    expected_stage = {
        "transcription": "transcription",
        "save": "save",
        "completion": "completion",
    }[failure]
    assert workflow.status["stage"] == expected_stage
    assert workflow.status["state"] == "failed"
    assert workflow.status["error_code"]
    assert getattr(raised.value, "stage") == expected_stage
    assert "synthetic source" not in str(raised.value)


@pytest.mark.parametrize("receipt_ok,guard_current", [(False, True), (True, False)])
def test_save_requires_verified_receipt_and_current_resource_guard(receipt_ok, guard_current):
    repository = _Repository()
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository, verified=receipt_ok))
    if not guard_current:
        repository.guard["current"] = False
    workflow = _workflow(repository, events, pipeline)

    with pytest.raises((RuntimeError, ValueError)):
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
        ))

    assert repository.certificate is None
    assert repository.draft["messages"]


@pytest.mark.parametrize("identity_field", ["save_intent_id", "operation_id"])
def test_save_rejects_receipt_bound_to_different_intent_or_operation(identity_field):
    repository = _Repository()
    events = []
    receipt = _receipt(repository)
    receipt[identity_field] = "foreign-identity"
    pipeline = _Pipeline(events, receipt=receipt)
    workflow = _workflow(repository, events, pipeline)

    with pytest.raises(RuntimeError, match="completion"):
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
        ))

    assert repository.certificate is None
    assert workflow.status["stage"] == "completion"


def test_injected_telegram_save_error_sets_safe_failed_workflow_status():
    repository = _Repository()
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)
    error_type = importlib.import_module(
        "services.connectors.telegram.save"
    ).TelegramSaveError

    async def fail_classification(result, *, draft_id):
        raise error_type("unsafe customer stage", "private customer detail")

    workflow.classify = fail_classification
    with pytest.raises(error_type) as raised:
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
        ))

    assert workflow.status["stage"] == "classification"
    assert workflow.status["state"] == "failed"
    assert workflow.status["error_code"] == "telegramsaveerror_failed"
    assert "private customer detail" not in repr(workflow.status)
    assert raised.value.stage == "classification"
    assert raised.value.error_code == "telegramsaveerror_failed"
    assert "private customer detail" not in str(raised.value)


def test_success_commits_certificate_bound_to_exact_draft_manifest_intent_and_operation():
    repository = _Repository()
    manifest = copy.deepcopy(repository.draft["source_manifest"])
    manifest_digest = repository.draft["manifest_digest"]
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)

    result = _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
    ))

    certificate = repository.certificate
    assert certificate is not None
    assert _value(certificate, "draft_id") == "draft-original"
    assert _value(certificate, "manifest_digest") == manifest_digest
    assert _value(certificate, "save_intent_id") == "intent-1"
    assert _value(certificate, "operation_id") == "operation-1"
    assert _value(certificate, "receipt_ref") == "receipt-ref-1"
    assert _value(certificate, "receipt_version") == 3
    assert _value(certificate, "context_id") == "telegram-context-1"
    assert _value(certificate, "origin_binding") == (
        "telegram-context-1", "draft-original", manifest_digest
    )
    assert repository.draft["source_manifest"] == manifest
    assert repository.draft["manifest_digest"] == manifest_digest
    assert result is not None
    assert pipeline.saved_inputs[0]["provider"] == "telegram"
    assert pipeline.saved_inputs[0]["dialog_date"] == "2026-10-19"


def test_dialog_date_uses_received_at_when_first_message_has_no_original_date():
    message = _message(1, "first source")
    message["original_date"] = None
    message["received_at"] = datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc)
    repository = _Repository(messages=[message])
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)

    _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
    ))

    assert pipeline.saved_inputs[0]["provider"] == "telegram"
    assert pipeline.saved_inputs[0]["dialog_date"] == "2026-10-20"


@pytest.mark.parametrize(
    "certificate_change",
    ["absent", "foreign_draft", "foreign_manifest"],
)
def test_cleanup_without_exact_authoritative_certificate_deletes_nothing(certificate_change):
    repository = _Repository()
    if certificate_change != "absent":
        certificate = _committed_certificate(repository)
        if certificate_change == "foreign_draft":
            certificate["draft_id"] = "draft-foreign"
        else:
            certificate["manifest_digest"] = "manifest-foreign"
        repository.certificate = certificate
    cleanup = _cleanup_type()(repository=repository, page_size=1)

    _run(cleanup.cleanup(context_id="context-1"))

    assert repository.deleted == []
    assert repository.draft["messages"]


def test_cleanup_returns_false_when_finalization_rejects_certificate():
    repository = _Repository()
    certificate = _committed_certificate(repository)
    repository.certificate = certificate

    async def reject_finalization(_context_id, _certificate):
        return False

    repository.finalize_cleanup = reject_finalization
    cleanup = _cleanup_type()(repository=repository, page_size=1)

    assert _run(cleanup.cleanup(context_id="context-1")) is False
    assert repository.deleted == [("draft-original", 1)]


def test_cleanup_missing_prepared_page_methods_cannot_finalize():
    repository = _Repository()
    repository.certificate = _committed_certificate(repository)
    repository.list_prepared_pages = None
    cleanup = _cleanup_type()(repository=repository, page_size=1)

    with pytest.raises(RuntimeError, match="prepared-page cleanup"):
        _run(cleanup.cleanup(context_id="context-1"))

    assert repository.deleted == []
    assert repository.draft.get("cleanup_state") is None


def test_restartable_paged_cleanup_only_removes_certified_draft_and_preserves_new_draft():
    messages = [_message(index, f"source {index}") for index in range(1, 5)]
    repository = _Repository(messages=messages)
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)
    _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
    ))
    # The committed certificate survives a process crash before cleanup starts.
    committed_certificate = copy.deepcopy(repository.certificate)
    newer_draft = {
        "draft_id": "draft-newer",
        "status": "collecting",
        "messages": [_message(99, "new collecting source")],
    }
    repository.add_draft(newer_draft)
    unchanged_newer_draft = copy.deepcopy(newer_draft)
    cleanup = _cleanup_type()(repository=repository, page_size=2)

    # The first call models a process stopping after its first durable page.
    repository.delete_draft_page = _partial_delete(repository, stop_after=2)
    with pytest.raises(RuntimeError, match="synthetic cleanup crash"):
        _run(cleanup.cleanup(context_id="telegram-context-1"))
    repository.delete_draft_page = _Repository.delete_draft_page.__get__(repository)
    assert repository.certificate == committed_certificate

    _run(cleanup.cleanup(context_id="telegram-context-1"))
    first_cleanup_deletes = list(repository.deleted)
    _run(cleanup.cleanup(context_id="telegram-context-1"))

    assert set(repository.deleted) == {("draft-original", item) for item in (1, 2, 3, 4)}
    assert len(repository.deleted) == len(set(repository.deleted))
    assert repository.active_draft_id == "draft-newer"
    assert repository.drafts["draft-newer"] == unchanged_newer_draft
    assert repository.drafts["draft-newer"]["messages"] == [
        _message(99, "new collecting source")
    ]
    assert repository.draft["messages"] == []
    assert first_cleanup_deletes == repository.deleted


def _partial_delete(repository, *, stop_after):
    original_delete = repository.delete_draft_page
    calls = 0

    async def delete_page(draft_id, messages):
        nonlocal calls
        calls += 1
        await original_delete(draft_id, messages)
        if calls == 1 and stop_after:
            raise RuntimeError("synthetic cleanup crash")

    return delete_page


def test_cleanup_does_not_rewrite_or_erase_independent_saved_doc_edit():
    repository = _Repository()
    events = []
    pipeline = _Pipeline(events, receipt=_receipt(repository))
    workflow = _workflow(repository, events, pipeline)

    _run(workflow.save(
        draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
    ))

    assert repository.certificate is not None
    assert _value(repository.certificate, "draft_id") == "draft-original"
    repository.doc_text = "independent user edit after save commit"
    cleanup = _cleanup_type()(repository=repository, page_size=1)

    _run(cleanup.cleanup(context_id="telegram-context-1"))

    assert repository.doc_text == "independent user edit after save commit"
    assert repository.doc_writes == []
    assert repository.deleted == [("draft-original", 1)]
    assert repository.draft["messages"] == []


def test_new_draft_analysis_context_excludes_previous_saved_draft_content():
    prior_message = _message(1, "prior saved private content")
    repository = _Repository(
        draft_id="draft-original", messages=[prior_message]
    )
    events = []
    prior_pipeline = _Pipeline(events, receipt={
        "verified": True,
        "draft_id": "draft-original",
        "manifest_digest": repository.draft["manifest_digest"],
        "save_intent_id": "prior-intent",
        "operation_id": "prior-operation",
        "receipt_ref": "prior-receipt-ref",
        "receipt_version": 1,
    })
    prior_workflow = _workflow(
        repository, events, prior_pipeline, audio=_Audio(repository, events)
    )

    _run(prior_workflow.save(
        draft_id="draft-original",
        intent_id="prior-intent",
        operation_id="prior-operation",
    ))

    prior_certificate = copy.deepcopy(repository.certificate)
    assert prior_certificate is not None
    assert _value(prior_certificate, "draft_id") == "draft-original"
    assert _value(prior_certificate, "save_intent_id") == "prior-intent"
    assert _value(prior_certificate, "operation_id") == "prior-operation"

    repository.previous_dialogue_context = [
        {"role": "user", "content": "prior saved private content"}
    ]
    repository.add_draft({
        "draft_id": "draft-new",
        "status": "sealed",
        "messages": [_message(2, "new draft only")],
        "source_manifest": [_message(2, "new draft only")],
        "manifest_digest": "manifest:draft-new:v1",
    })
    assert repository.active_draft_id == "draft-new"

    new_pipeline = _Pipeline(events, receipt={
        "verified": True,
        "draft_id": "draft-new",
        "manifest_digest": repository.drafts["draft-new"]["manifest_digest"],
        "save_intent_id": "new-intent",
        "operation_id": "new-operation",
        "receipt_ref": "new-receipt-ref",
        "receipt_version": 2,
    })
    new_workflow = _workflow(
        repository, events, new_pipeline, audio=_Audio(repository, events)
    )

    _run(new_workflow.save(
        draft_id="draft-new",
        intent_id="new-intent",
        operation_id="new-operation",
    ))

    context = new_pipeline.contexts[0]
    assert repository.previous_context_reads == []
    assert "prior saved private content" not in repr(context)
    assert "new draft only" in repr(context)
    assert [row["text"] for row in context["messages"]] == ["new draft only"]
    assert repository.active_draft_id == "draft-new"
    assert repository.drafts["draft-original"]["messages"] == [prior_message]


def test_new_draft_classification_uses_only_its_source_after_prior_doc_move_and_rename():
    from services.connectors.telegram.save import TelegramClassifiedDialog
    from core.dialogs.classification import FolderSelectionResult

    repository = _Repository(messages=[_message(1, "prior private source")])
    repository.context_id = "telegram-context-1"
    repository.get_context = lambda _context_id: {"topic_name": "Current topic"}
    repository.last_doc_id = "doc-old"
    repository.last_saved_document = {
        "title": "renamed prior title", "folder_path": ["old", "folder"],
    }
    repository.previous_dialogue_context = [
        {"role": "user", "content": "prior private source"},
    ]
    repository.add_draft({
        "draft_id": "draft-new", "status": "sealed",
        "messages": [_message(2, "new source only")],
        "source_manifest": [_message(2, "new source only")],
        "manifest_digest": "manifest:draft-new:v1",
    })
    seen = []

    async def classify(result, *, draft_id, topic_hint=None, folder_path=None):
        seen.append((draft_id, copy.deepcopy(result), topic_hint, folder_path))
        return TelegramClassifiedDialog(
            payload={"title": result["title"], "utterances": []},
            selection=FolderSelectionResult(folder_id=f"folder-{draft_id}"),
        )

    async def save(result, *, draft_id, manifest_digest, intent_id, operation_id):
        return {
            "verified": True, "draft_id": draft_id, "manifest_digest": manifest_digest,
            "save_intent_id": intent_id, "operation_id": operation_id,
            "receipt_ref": f"receipt-{draft_id}", "receipt_version": 1,
        }

    def workflow_for(draft_id):
        events = []
        return _save_type()(
            repository=repository,
            audio_transcriber=_Audio(repository, events),
            context_id="telegram-context-1",
            preprocess=lambda messages: list(messages),
            analyze=lambda messages, *, context: {
                "title": f"title-{draft_id}",
                "source": [row["text"] for row in messages],
                "folder_path": ["fresh", draft_id],
            },
            classify=classify,
            save=save,
        )

    _run(workflow_for("draft-original").save(
        draft_id="draft-original", intent_id="intent-old", operation_id="op-old",
    ))
    _run(workflow_for("draft-new").save(
        draft_id="draft-new", intent_id="intent-new", operation_id="op-new",
    ))

    assert seen[0][1]["source"] == ["prior private source"]
    assert seen[1] == (
        "draft-new",
        {"title": "title-draft-new", "source": ["new source only"],
         "folder_path": ["fresh", "draft-new"]},
        "Current topic",
        ["fresh", "draft-new"],
    )
    assert "prior private source" not in repr(seen[1])
    assert "renamed prior title" not in repr(seen[1])
    assert "old" not in repr(seen[1][3])


def test_workflow_governs_and_reuses_firestore_safe_preparation_and_classification_results(
    firestore_fake,
):
    from core.dialogs.classification import FolderSelectionResult
    from core.integrations.firestore.operation_queue import (
        QueueOperationRetryStore, claim_head, register_operation,
    )
    from core.operations.retries import RetryGovernor
    from google.cloud.firestore_v1._helpers import encode_value
    from services.connectors.telegram.models import PreparedUtterance
    from services.connectors.telegram.save import TelegramClassifiedDialog
    from services.connectors.telegram.repository import SESSIONS

    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("telegram-context-1")
    register_operation(client, context_ref, "governed-save", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="save-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(), retry_store=QueueOperationRetryStore(client, context_ref),
    )
    repository = _Repository()
    events = []
    receipt = _receipt(repository)
    receipt["operation_id"] = "governed-save"
    pipeline = _Pipeline(events, receipt=receipt)
    classified_calls = []

    async def classify(result, *, draft_id):
        classified_calls.append(draft_id)
        return TelegramClassifiedDialog(
            payload={"title": "Planning", "utterances": []},
            selection=FolderSelectionResult(
                folder_id="folder-1", speaker_candidate_ids=("speaker-1", "speaker-2"),
            ),
        )

    class ForwardingAudio(_Audio):
        async def process_messages(
            self, messages, *, draft_id, analyze, retry_governor=None,
            operation_id=None, operation_lease=None,
        ):
            self.received = (retry_governor, operation_id, operation_lease)
            return await analyze(messages)

    async def preprocess(_messages):
        return [PreparedUtterance(
            order_index=0, speaker="Speaker 1", reliable_author_key=None,
            text="synthetic transcript", word_count=2, source_message_ids=(1,),
        )]

    audio = ForwardingAudio(repository, events)
    workflow = _save_type()(
        repository=repository,
        audio_transcriber=audio,
        context_id="telegram-context-1",
        preprocess=preprocess,
        analyze=pipeline.analyze,
        classify=classify,
        save=pipeline.save,
    )
    kwargs = {
        "draft_id": "draft-original", "intent_id": "intent-1",
        "operation_id": "governed-save", "retry_governor": governor,
        "operation_lease": lease,
    }

    first = _run(workflow.save(**kwargs))
    second = _run(workflow.save(**kwargs))

    assert audio.received == (governor, "governed-save", lease)
    assert classified_calls == ["draft-original"]
    assert _value(first, "classification_outcome") == "ambiguous"
    assert _value(second, "classification_outcome") == "ambiguous"
    assert pipeline.saved_inputs[0]["title"] == "Planning"
    assert pipeline.saved_inputs[1]["title"] == "Planning"
    preparation = governor.get_stage("governed-save", "telegram.preparation")
    analysis = governor.get_stage("governed-save", "telegram.analysis")
    classification = governor.get_stage("governed-save", "telegram.classification")
    assert preparation.attempts_started == analysis.attempts_started == classification.attempts_started == 1
    assert isinstance(preparation.result, list)
    assert all(isinstance(row, dict) for row in preparation.result)
    assert isinstance(preparation.result[0]["source_message_ids"], list)
    assert isinstance(classification.result, dict)
    assert classification.result["__telegram_classified_dialog__"] is True
    encode_value(preparation.result)
    assert isinstance(analysis.result, dict)
    encode_value(analysis.result)
    encode_value(classification.result)


@pytest.mark.parametrize(
    "bad_stage,expected_stage",
    [
        ("classification", "telegram.classification"),
        ("preparation", "telegram.preparation"),
        ("preparation_opaque", "telegram.preparation"),
    ],
)
def test_governed_malformed_stage_outputs_are_recorded_as_permanent_failures(
    firestore_fake, bad_stage, expected_stage,
):
    from core.integrations.firestore.operation_queue import (
        QueueOperationRetryStore, claim_head, register_operation,
    )
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import SESSIONS

    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("telegram-context-1")
    register_operation(client, context_ref, "governed-invalid", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="save-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(), retry_store=QueueOperationRetryStore(client, context_ref),
    )
    repository = _Repository()
    events = []
    receipt = _receipt(repository)
    receipt["operation_id"] = "governed-invalid"
    pipeline = _Pipeline(events, receipt=receipt)
    callback_calls = []

    async def preprocess(messages):
        callback_calls.append("preparation")
        if bad_stage == "preparation":
            return "malformed"
        if bad_stage == "preparation_opaque":
            return [{"nested": [object()]}]
        return list(messages)

    async def classify(result, *, draft_id):
        callback_calls.append("classification")
        return object()

    class ForwardingAudio(_Audio):
        async def process_messages(
            self, messages, *, draft_id, analyze, retry_governor=None,
            operation_id=None, operation_lease=None,
        ):
            return await analyze(messages)

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=ForwardingAudio(repository, events),
        context_id="telegram-context-1",
        preprocess=preprocess,
        analyze=pipeline.analyze,
        classify=classify,
        save=pipeline.save,
    )

    with pytest.raises(Exception):
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1",
            operation_id="governed-invalid", retry_governor=governor,
            operation_lease=lease,
        ))

    stage = governor.get_stage("governed-invalid", expected_stage)
    assert stage.state.value == "failed"
    assert stage.attempts_started == stage.failures == 1
    assert stage.in_flight_attempt_id is None
    assert stage.last_error is not None and stage.last_error.retryable is False
    expected_calls = ["preparation"]
    if bad_stage == "classification":
        expected_calls.append("classification")
    assert callback_calls == expected_calls


@pytest.mark.parametrize(
    "error_factory,expected_retryability",
    [
        (lambda: ConnectionError("temporary provider outage"), [True, True, False]),
        (lambda: ValueError("permanent provider rejection"), [False]),
    ],
)
def test_workflow_provider_failures_follow_shared_stage_budget_and_permanent_errors(
    firestore_fake, error_factory, expected_retryability,
):
    from core.integrations.firestore.operation_queue import (
        QueueOperationRetryStore, claim_head, register_operation,
    )
    from core.operations.retries import RetryGovernor
    from services.connectors.telegram.repository import SESSIONS

    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("telegram-context-1")
    register_operation(client, context_ref, "governed-failure", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="save-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(), retry_store=QueueOperationRetryStore(client, context_ref),
    )
    repository = _Repository()
    events = []
    receipt = _receipt(repository)
    receipt["operation_id"] = "governed-failure"
    pipeline = _Pipeline(events, receipt=receipt)
    provider_calls = []

    async def analyze(messages, *, context):
        provider_calls.append(1)
        raise error_factory()

    class ForwardingAudio(_Audio):
        async def process_messages(
            self, messages, *, draft_id, analyze, retry_governor=None,
            operation_id=None, operation_lease=None,
        ):
            self.received = (retry_governor, operation_id, operation_lease)
            return await analyze(messages)

    workflow = _save_type()(
        repository=repository,
        audio_transcriber=ForwardingAudio(repository, events),
        context_id="telegram-context-1",
        preprocess=pipeline.preprocess,
        analyze=analyze,
        classify=pipeline.classify,
        save=pipeline.save,
    )
    kwargs = {
        "draft_id": "draft-original", "intent_id": "intent-1",
        "operation_id": "governed-failure", "retry_governor": governor,
        "operation_lease": lease,
    }
    observed_retryability = []

    for _ in expected_retryability:
        with pytest.raises(Exception) as raised:
            _run(workflow.save(**kwargs))
        observed_retryability.append(raised.value.retryable_stage)

    stage = governor.get_stage("governed-failure", "telegram.analysis")
    assert observed_retryability == expected_retryability
    assert stage.failures == len(expected_retryability)
    assert stage.attempts_started == len(expected_retryability)
    assert provider_calls == [1] * len(expected_retryability)


@pytest.mark.parametrize(
    "stage_factory,expected_retryable",
    [
        (lambda: type("Stage", (), {"last_error": None, "failures": 0})(), True),
        (lambda: type("Stage", (), {"last_error": type("Failure", (), {"retryable": False})(), "failures": 1})(), False),
        (lambda: type("Stage", (), {"last_error": type("Failure", (), {"retryable": True})(), "failures": 3})(), False),
    ],
)
def test_governed_stage_retryability_uses_only_persisted_failure_state(stage_factory, expected_retryable):
    from services.connectors.telegram.save import _run_telegram_stage

    class Governor:
        max_failures = 3

        async def run_stage_async(self, *_args, **_kwargs):
            raise OSError("stage result persistence failed")

        def get_stage(self, *_args):
            return stage_factory()

    error = OSError("stage result persistence failed")
    governor = Governor()
    async def raise_stage_error(*_args, **_kwargs):
        raise error

    # Use the same error instance from run_stage_async so the workflow marker is observable.
    governor.run_stage_async = raise_stage_error
    with pytest.raises(OSError):
        _run(_run_telegram_stage(
            governor, "operation", type("Lease", (), {"owner": "worker", "fence": 1})(),
            "telegram.analysis", lambda: None,
        ))

    assert error.telegram_stage_retryable is expected_retryable


def test_governed_stage_state_read_error_remains_retryable():
    from services.connectors.telegram.save import _run_telegram_stage

    error = OSError("stage result persistence failed")

    class Governor:
        async def run_stage_async(self, *_args, **_kwargs):
            raise error

        def get_stage(self, *_args):
            raise OSError("stage read failed")

    with pytest.raises(OSError):
        _run(_run_telegram_stage(
            Governor(), "operation", type("Lease", (), {"owner": "worker", "fence": 1})(),
            "telegram.analysis", lambda: None,
        ))

    assert error.telegram_stage_retryable is True
