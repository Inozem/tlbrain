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

import pytest


pytestmark = pytest.mark.offline


def _value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _run(awaitable):
    return asyncio.run(awaitable)


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
        if self.fail_save:
            raise RuntimeError("synthetic save failure")
        return copy.deepcopy(self.receipt)


def _workflow(repository, events, pipeline, *, audio=None):
    return _save_type()(
        repository=repository,
        audio_transcriber=audio or _Audio(repository, events),
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

    with pytest.raises(RuntimeError):
        _run(workflow.save(
            draft_id="draft-original", intent_id="intent-1", operation_id="operation-1"
        ))

    assert (repository.draft["messages"], repository.draft["source_manifest"]) == original
    assert repository.certificate is None
    assert repository.deleted == []


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
    assert repository.draft["source_manifest"] == manifest
    assert repository.draft["manifest_digest"] == manifest_digest
    assert result is not None


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
        _run(cleanup.cleanup(context_id="context-1"))
    repository.delete_draft_page = _Repository.delete_draft_page.__get__(repository)
    assert repository.certificate == committed_certificate

    _run(cleanup.cleanup(context_id="context-1"))
    first_cleanup_deletes = list(repository.deleted)
    _run(cleanup.cleanup(context_id="context-1"))

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

    _run(cleanup.cleanup(context_id="context-1"))

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
        "status": "collecting",
        "messages": [_message(2, "new draft only")],
        "source_manifest": [_message(2, "new draft only")],
        "manifest_digest": "manifest:draft-new:v1",
    })
    assert repository.active_draft_id == "draft-new"

    new_pipeline = _Pipeline(events, receipt={
        "verified": True,
        "draft_id": "draft-new",
        "manifest_digest": repository.drafts["draft-new"]["manifest_digest"],
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
