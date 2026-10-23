from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from google.genai import types

from core.documents import operations as document_operations
from core.integrations.firestore.effects import get_effect_allocation
from core.integrations.firestore.mutations import get_mutation_effect, get_resource_guard
from core.integrations.firestore.operations import claim_operation, create_operation
from core.integrations.google_drive import folder_service
from core.operations import write
from core.operations.mutations import MutationConflict, MutationOutcomeUnknown
from core.operations.models import OperationOrigin


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(client, fake_clock, operation_id="write-test-operation"):
    create_operation(
        client, operation_id=operation_id, origin=OperationOrigin.MCP,
        caller_binding="synthetic", immutable_input_digest="sha256:" + "a" * 64,
        immutable_input_ref="synthetic-input", now=fake_clock.now(),
    )
    lease = claim_operation(
        client, operation_id, owner="write-worker", now=fake_clock.now(), lease_seconds=60,
    )
    assert lease is not None
    return write.WriteContext(
        operation_id=operation_id, effect_id="tool-ordinal-0", owner="write-worker",
        fence=lease.fence, client=client, clock=fake_clock,
    )


def _configure_firestore(monkeypatch, firestore_fake, fake_clock):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    monkeypatch.setattr(write.firestore_client, "get_client", lambda: client)
    monkeypatch.setattr(write, "get_root_folder_id", lambda: "root-id")
    return client


def test_move_transcript_uses_real_guard_and_replays_without_resolving_path(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"id": "doc-1", "name": "title", "parents": ["old"]},
             "record": {"status": "synced", "parent_id": "old", "source_file": "title"},
             "qparent": "old"}
    resolutions, writes, provider_reads = [], [], []
    monkeypatch.setattr(document_operations, "resolve_move_target",
                        lambda path, _db=None: resolutions.append(path) or "folder-new")
    monkeypatch.setattr(write.drive_files, "get_file_metadata",
                        lambda _id: provider_reads.append("drive") or state["drive"])
    monkeypatch.setattr(write.transcripts, "get_transcript_record",
                        lambda _id, _db=None: provider_reads.append("firestore") or state["record"])
    monkeypatch.setattr(write.qdrant_reader, "get_parent_ids_by_doc_ids",
                        lambda *_args: {"doc-1": [state["qparent"]]})
    monkeypatch.setattr(write.folder_repository, "get_folder_id_by_name", lambda *_args: None)

    def apply(doc_id, parent_id, root_id, *, single_attempt, db=None, **_kwargs):
        writes.append((doc_id, parent_id, root_id, single_attempt))
        state["drive"]["parents"] = [parent_id]
        state["record"]["parent_id"] = parent_id
        state["record"]["status"] = "imported"
        state["qparent"] = parent_id
        return document_operations.MoveTranscriptResult([])

    monkeypatch.setattr(document_operations, "apply_move_transcript", apply)
    result = write.move_transcript("doc-1", ["new-path"], context=context)
    assert result.unassigned_count == 0
    assert writes == [("doc-1", "folder-new", "root-id", True)]
    assert get_mutation_effect(client, context.operation_id, context.effect_id).state.value == "confirmed"

    reads_before_replay = len(provider_reads)
    monkeypatch.setattr(document_operations, "resolve_move_target",
                        lambda _path, _db=None: pytest.fail("confirmed replay must use persisted target IDs"))
    replay = write.move_transcript("doc-1", ["new-path"], context=context)
    assert replay == result
    assert resolutions == [["new-path"]]
    assert len(writes) == 1
    assert len(provider_reads) == reads_before_replay

    with pytest.raises(ValueError, match="different arguments"):
        write.move_transcript("doc-1", ["changed-path"], context=context)
    assert len(writes) == 1


def test_folder_only_transcript_move_updates_selected_doc_payload_without_embedding(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {
        "drive": {"doc-selected": {"id": "doc-selected", "name": "title", "parents": ["old"]}},
        "records": {"doc-selected": {"status": "synced", "parent_id": "old"}},
        "parents": {"doc-selected": ["old"]},
    }
    state["drive"]["doc-next"] = {"id": "doc-next", "name": "next title", "parents": ["next-old"]}
    state["records"]["doc-next"] = {"status": "synced", "parent_id": "next-old"}
    state["parents"]["doc-next"] = ["next-old"]
    payload_updates = []
    monkeypatch.setattr(document_operations, "resolve_move_target", lambda *_args: "folder-new")
    monkeypatch.setattr(write.drive_files, "get_file_metadata",
                        lambda doc_id: state["drive"][doc_id])
    monkeypatch.setattr(write.transcripts, "get_transcript_record",
                        lambda doc_id, _db=None: state["records"][doc_id])
    monkeypatch.setattr(write.qdrant_reader, "get_parent_ids_by_doc_ids",
                        lambda doc_ids, _root: {doc_id: state["parents"][doc_id] for doc_id in doc_ids})

    def set_payload_parent_id(doc_id, root_id, folder_id, **_kwargs):
        payload_updates.append((doc_id, root_id, folder_id))
        state["parents"][doc_id] = [folder_id]

    monkeypatch.setattr(document_operations, "move_file_to_folder",
                        lambda doc_id, folder_id: state["drive"][doc_id].update(parents=[folder_id]))
    monkeypatch.setattr(document_operations.transcripts, "move_transcript_record",
                        lambda doc_id, folder_id, _db=None: state["records"][doc_id].update(
                            parent_id=folder_id, status="imported"))
    monkeypatch.setattr(document_operations, "set_payload_parent_id", set_payload_parent_id)
    monkeypatch.setattr(document_operations.folders, "get_folder_id_by_name", lambda *_args: None)

    from core.integrations.gemini import embeddings
    monkeypatch.setattr(embeddings, "embed", lambda *_args, **_kwargs: pytest.fail(
        "folder-only move must not regenerate embeddings"))

    result = write.move_transcript("doc-selected", ["new-folder"], context=context)

    assert result.unassigned_count == 0
    assert payload_updates == [("doc-selected", "root-id", "folder-new")]
    assert state["records"]["doc-selected"]["parent_id"] == "folder-new"
    assert state["parents"]["doc-selected"] == ["folder-new"]
    assert state["drive"]["doc-next"]["parents"] == ["next-old"]
    assert state["records"]["doc-next"]["parent_id"] == "next-old"
    assert state["parents"]["doc-next"] == ["next-old"]


def test_rename_transcript_partial_cross_store_failure_remains_unknown(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"id": "doc-1", "name": "old", "parents": ["folder"]},
             "record": {"status": "synced", "parent_id": "folder", "source_file": "old"}}
    writes = []
    monkeypatch.setattr(write.transcripts, "get_transcript_record", lambda _id, _db=None: state["record"])
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: state["drive"])

    def apply(doc_id, title, _db=None, **_kwargs):
        writes.append((doc_id, title))
        state["drive"]["name"] = title
        raise RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_RESPONSE")

    monkeypatch.setattr(document_operations, "apply_rename_transcript", apply)
    with pytest.raises(MutationOutcomeUnknown) as error:
        write.rename_transcript("doc-1", "new", context=context)
    assert "SYNTHETIC_PRIVATE_PROVIDER_RESPONSE" not in str(error.value)
    assert writes == [("doc-1", "new")]
    assert get_resource_guard(client, "document:doc-1").unsettled_effect_ids == (context.effect_id,)


def test_lost_rename_response_is_reconciled_without_a_second_provider_write(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"id": "doc-1", "name": "old", "parents": ["folder"]},
             "record": {"status": "synced", "parent_id": "folder", "source_file": "old"}}
    writes, reads = [], []
    monkeypatch.setattr(write.transcripts, "get_transcript_record",
                        lambda _id, _db=None: reads.append("firestore") or state["record"])
    monkeypatch.setattr(write.drive_files, "get_file_metadata",
                        lambda _id: reads.append("drive") or state["drive"])

    def apply(doc_id, title, _db=None, **_kwargs):
        writes.append((doc_id, title))
        raise TimeoutError("SYNTHETIC_LOST_RESPONSE")

    monkeypatch.setattr(document_operations, "apply_rename_transcript", apply)
    with pytest.raises(MutationOutcomeUnknown):
        write.rename_transcript("doc-1", "new", context=context)
    assert writes == [("doc-1", "new")]
    reads_before_reconcile = len(reads)
    state["drive"]["name"] = "new"
    state["record"]["source_file"] = "new"
    assert write.rename_transcript("doc-1", "new", context=context) is None
    assert len(reads) - reads_before_reconcile == 2
    assert writes == [("doc-1", "new")]
    assert get_mutation_effect(client, context.operation_id, context.effect_id).state.value == "confirmed"


def test_expired_after_drive_rename_skips_stale_firestore_bookkeeping(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"name": "old", "parents": ["folder"]},
             "record": {"status": "synced", "parent_id": "folder", "source_file": "old"}}
    bookkeeping = []
    monkeypatch.setattr(write.transcripts, "get_transcript_record",
                        lambda _id, _db=None: state["record"])
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: state["drive"])

    def slow_rename(_doc_id, title):
        state["drive"]["name"] = title
        fake_clock.advance(seconds=61)

    monkeypatch.setattr(document_operations, "rename_file", slow_rename)
    monkeypatch.setattr(write.transcripts, "update_index",
                        lambda *_args, **_kwargs: bookkeeping.append("update"))

    with pytest.raises(MutationOutcomeUnknown):
        write.rename_transcript("doc-1", "new", context=context)
    assert bookkeeping == []
    assert get_resource_guard(client, "document:doc-1").unsettled_effect_ids == (context.effect_id,)


def test_move_folder_success_reserves_tree_folder_and_descendant_document(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"id": "folder-a", "name": "A", "parents": ["old-parent"]},
             "folder": {"name": "A", "parent_id": "old-parent"}}
    monkeypatch.setattr(folder_service, "prepare_move_folder", lambda *_args, **_kwargs: ("folder-a", "new-parent"))
    monkeypatch.setattr(folder_service, "expand_registered_subtree", lambda _id, _db=None: ["folder-a"])
    monkeypatch.setattr(write, "_document_ids_in_folders", lambda _ids, _db=None: ["doc-a"])
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: state["drive"])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id", lambda _id, _db=None: state["folder"])
    monkeypatch.setattr(write.folder_repository, "get_all_folders", lambda _db=None: {"folder-a": state["folder"]})

    def apply(folder_id, parent_id, *, db=None, **_kwargs):
        state["drive"]["parents"] = [parent_id]
        state["folder"]["parent_id"] = parent_id
        return folder_id, parent_id

    monkeypatch.setattr(folder_service, "apply_move_folder", apply)
    assert write.move_folder(["A"], ["Parent"], context=context) == ("folder-a", "new-parent")
    for resource in ("tree:root-id", "folder:folder-a", "document:doc-a"):
        assert get_resource_guard(client, resource).resource_epoch == 1


def test_competing_move_folder_plan_winner_supplies_read_and_apply_targets(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {
        "drive": {"folder-loser": {"parents": ["old"]}, "folder-winner": {"parents": ["old"]}},
        "folder": {"folder-loser": {"name": "Loser", "parent_id": "old"},
                   "folder-winner": {"name": "Winner", "parent_id": "old"}},
    }
    monkeypatch.setattr(folder_service, "prepare_move_folder",
                        lambda *_args, **_kwargs: ("folder-loser", "new"))
    monkeypatch.setattr(folder_service, "expand_registered_subtree", lambda folder_id, _db=None: [folder_id])
    monkeypatch.setattr(write, "_document_ids_in_folders",
                        lambda folder_ids, _db=None: ["doc-" + folder_ids[0].split("-")[-1]])
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda folder_id: state["drive"][folder_id])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id",
                        lambda folder_id, _db=None: state["folder"][folder_id])

    real_persist = write.persist_write_plan
    def persist_winner(*args, **kwargs):
        winner = dict(kwargs["plan"])
        metadata = dict(winner["metadata"])
        metadata["folder_id"] = "folder-winner"
        metadata["descendants"] = ["folder-winner"]
        metadata["docs"] = ["doc-winner"]
        metadata["resources"] = ["document:doc-winner", "folder:folder-winner", "tree:root-id"]
        before = {"drive_parent": "old", "firestore_parent": "old",
                  "descendants": ["folder-winner"], "docs": ["doc-winner"]}
        metadata["desired_state"] = {**before, "drive_parent": "new", "firestore_parent": "new"}
        winner["metadata"] = metadata
        winner["expected_pre_state"] = write.digest(before)
        kwargs["plan"] = winner
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(write, "persist_write_plan", persist_winner)
    applied = []
    def apply(folder_id, parent_id, *, db=None, before_bookkeeping=None):
        applied.append(folder_id)
        state["drive"][folder_id]["parents"] = [parent_id]
        state["folder"][folder_id]["parent_id"] = parent_id
        return folder_id, parent_id
    monkeypatch.setattr(folder_service, "apply_move_folder", apply)

    assert write.move_folder(["loser"], ["new"], context=context) == ("folder-winner", "new")
    assert applied == ["folder-winner"]
    assert get_resource_guard(client, "document:doc-winner").resource_epoch == 1
    assert get_resource_guard(client, "document:doc-loser").resource_epoch == 0


def test_expired_write_context_cannot_reach_domain_apply(monkeypatch, firestore_fake, fake_clock):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    fake_clock.advance(seconds=61)
    applied = []
    monkeypatch.setattr(write.transcripts, "get_transcript_record", lambda _id, _db=None: {"source_file": "old"})
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: {"name": "old", "parents": ["folder"]})
    monkeypatch.setattr(document_operations, "apply_rename_transcript", lambda *_a: applied.append("write"))
    with pytest.raises(ValueError, match="current operation lease"):
        write.rename_transcript("doc-1", "new", context=context)
    assert applied == []


def test_move_folder_guards_descendant_documents_and_rejects_changed_membership(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    state = {"drive": {"id": "folder-a", "name": "A", "parents": ["old-parent"]},
             "folder": {"name": "A", "parent_id": "old-parent"}, "docs": ["doc-a"]}
    reads, writes = [], []
    monkeypatch.setattr(folder_service, "prepare_move_folder", lambda *_args, **_kwargs: ("folder-a", "new-parent"))
    monkeypatch.setattr(folder_service, "expand_registered_subtree", lambda _id, _db=None: ["folder-a"])
    monkeypatch.setattr(write, "_document_ids_in_folders", lambda _ids, _db=None: list(state["docs"]))
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: state["drive"])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id", lambda _id, _db=None: state["folder"])
    monkeypatch.setattr(write.folder_repository, "get_all_folders", lambda _db=None: {"folder-a": state["folder"]})
    monkeypatch.setattr(write.transcripts, "stream_index_records", lambda **_kwargs: [])

    def apply(folder_id, parent_id, *, db=None, **_kwargs):
        writes.append((folder_id, parent_id))
        state["drive"]["parents"] = [parent_id]
        state["folder"]["parent_id"] = parent_id
        return folder_id, parent_id

    monkeypatch.setattr(folder_service, "apply_move_folder", apply)

    # A newly discovered affected document between planning and reservation
    # changes the pre-state, so the callback never writes with incomplete locks.
    def docs_with_change(_ids, _db=None):
        reads.append("membership")
        return ["doc-a"] if len(reads) == 1 else ["doc-a", "doc-new"]

    monkeypatch.setattr(write, "_document_ids_in_folders", docs_with_change)
    with pytest.raises(MutationConflict):
        write.move_folder(["A"], ["Parent"], context=context)
    assert writes == []
    assert get_resource_guard(client, "document:doc-new").resource_epoch == 0


def test_rename_folder_and_create_folder_use_guarded_drive_and_registry_state(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    state = {"drive": {"id": "folder-a", "name": "Old", "parents": ["parent"]},
             "folder": {"name": "Old", "parent_id": "parent"}}
    path_resolutions, sibling_checks, rename_calls = [], [], []
    monkeypatch.setattr(folder_service, "resolve_registered_folder_path",
                        lambda _path, _db=None: path_resolutions.append("resolve") or ["folder-a"])
    monkeypatch.setattr(folder_service, "expand_registered_subtree", lambda _id, _db=None: ["folder-a"])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id", lambda _id, _db=None: state["folder"])
    monkeypatch.setattr(write.folder_repository, "folder_name_exists",
                        lambda *_a, **_k: sibling_checks.append("check") or False)
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: state["drive"])
    def rename_drive(_id, name):
        rename_calls.append(name)
        state["drive"].update(name=name)
    monkeypatch.setattr(write.drive_client, "rename_folder", rename_drive)

    def register(folder_id, name, parent_id, _db=None):
        state["folder"].update(name=name, parent_id=parent_id)

    monkeypatch.setattr(write.folder_repository, "upsert_folder", register)
    rename_context = _context(client, fake_clock, "rename-folder")
    assert write.rename_folder(["Old"], "New", context=rename_context) == "folder-a"
    assert state["drive"]["name"] == state["folder"]["name"] == "New"
    counts = (len(path_resolutions), len(sibling_checks), len(rename_calls))
    monkeypatch.setattr(folder_service, "resolve_registered_folder_path",
                        lambda *_a, **_k: pytest.fail("confirmed replay must not resolve the old path"))
    monkeypatch.setattr(write.folder_repository, "folder_name_exists",
                        lambda *_a, **_k: pytest.fail("confirmed replay must not revalidate sibling names"))
    assert write.rename_folder(["Old"], "New", context=rename_context) == "folder-a"
    assert (len(path_resolutions), len(sibling_checks), len(rename_calls)) == counts
    monkeypatch.setattr(write.folder_repository, "folder_name_exists", lambda *_a, **_k: False)

    context = _context(client, fake_clock, "create-folder")
    allocated = []
    real_allocate = write.allocate_folder_id

    def allocate(*args, **kwargs):
        kwargs["id_factory"] = lambda: "folder-created"
        result = real_allocate(*args, **kwargs)
        allocated.append(get_effect_allocation(client, context.operation_id, context.effect_id))
        return result

    monkeypatch.setattr(write, "allocate_folder_id", allocate)
    monkeypatch.setattr(folder_service, "prepare_create_folder", lambda *_args, **_kwargs: "parent")
    created = {"drive": None, "folder": None}
    monkeypatch.setattr(write.drive_files, "get_file_metadata_if_exists", lambda _id: created["drive"])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id", lambda _id, _db=None: created["folder"])

    def create(name, parent, *, folder_id, intent_marker):
        assert allocated and allocated[0]["resource_id"] == folder_id
        created["drive"] = {"id": folder_id, "name": name, "mimeType": "application/vnd.google-apps.folder",
                            "parents": [parent], "trashed": False,
                            "appProperties": {"tlbrain_effect_intent": intent_marker}}
        return folder_id

    monkeypatch.setattr(write.drive_client, "create_folder", create)
    monkeypatch.setattr(write.folder_repository, "upsert_folder",
                        lambda folder_id, name, parent, _db=None: created.update(folder={"name": name, "parent_id": parent}))
    assert write.create_folder("Created", context=context) == "folder-created"
    assert created["folder"] == {"name": "Created", "parent_id": "parent"}


def test_fresh_rename_folder_to_its_current_name_keeps_legacy_conflict(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    monkeypatch.setattr(folder_service, "resolve_registered_folder_path",
                        lambda _path, _db=None: ["folder-a"])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id",
                        lambda _id, _db=None: {"name": "Same", "parent_id": "parent"})
    monkeypatch.setattr(write.folder_repository, "folder_name_exists",
                        lambda *_args, **_kwargs: True)
    writes = []
    monkeypatch.setattr(write.drive_client, "rename_folder", lambda *_args: writes.append("drive"))

    with pytest.raises(folder_service.FolderNameConflict, match="Same"):
        write.rename_folder(["Same"], "Same", context=context)
    assert writes == []
    assert get_resource_guard(client, "tree:root-id").resource_epoch == 0


@pytest.mark.parametrize(
    ("write_type", "document_id"),
    [("rename_transcript", "doc-1"), ("move_transcript", "doc-1"), ("create_folder", None)],
)
def test_internal_completion_records_only_transcript_document_ids(
    firestore_fake, fake_clock, write_type, document_id,
):
    client = firestore_fake.client_class()
    firestore_fake.set_clock(fake_clock)
    context = _context(client, fake_clock, "completion-" + write_type)
    internal = write.WriteContext(
        operation_id=context.operation_id, effect_id=context.effect_id, owner=context.owner,
        fence=context.fence, client=client, clock=fake_clock, _internal=True,
    )
    metadata = {"write_type": write_type, "doc_id": "doc-1", "folder_id": "folder-a"}

    write._finish(internal, metadata)

    record = client.collection("dialog_operations").document(context.operation_id).get().to_dict()
    assert record["doc_id"] == document_id
    assert record["result"].get("document_id") == document_id


def test_create_folder_rechecks_sibling_uniqueness_inside_tree_guard(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    records, drive_folders, created = {}, {}, []
    monkeypatch.setattr(folder_service, "prepare_create_folder", lambda *_args, **_kwargs: "parent")
    monkeypatch.setattr(write.folder_repository, "folder_name_exists",
                        lambda name, parent, _db=None, *, exclude_id=None: any(
                            key != exclude_id and record["name"] == name and record["parent_id"] == parent
                            for key, record in records.items()))
    monkeypatch.setattr(write.drive_files, "get_file_metadata_if_exists",
                        lambda folder_id: drive_folders.get(folder_id))
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id",
                        lambda folder_id, _db=None: records.get(folder_id))
    monkeypatch.setattr(write.folder_repository, "upsert_folder",
                        lambda folder_id, name, parent, _db=None: records.update(
                            {folder_id: {"name": name, "parent_id": parent}}))
    real_allocate = write.allocate_folder_id
    ids = iter(("first-folder", "second-folder"))
    monkeypatch.setattr(write, "allocate_folder_id", lambda *args, **kwargs: real_allocate(
        *args, **{**kwargs, "id_factory": lambda: next(ids)}))

    def create(name, parent, *, folder_id, intent_marker):
        created.append(folder_id)
        drive_folders[folder_id] = {
            "id": folder_id, "name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent], "trashed": False,
            "appProperties": {"tlbrain_effect_intent": intent_marker},
        }
        return folder_id
    monkeypatch.setattr(write.drive_client, "create_folder", create)

    first = _context(client, fake_clock, "create-sibling-first")
    second = _context(client, fake_clock, "create-sibling-second")
    assert write.create_folder("Same", context=first) == "first-folder"
    with pytest.raises(folder_service.FolderNameConflict):
        write.create_folder("Same", context=second)
    assert created == ["first-folder"]
    assert get_resource_guard(client, "tree:root-id").unsettled_effect_ids == ()


def test_rename_folder_rechecks_sibling_uniqueness_inside_tree_guard(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    folder = {"name": "Old", "parent_id": "parent"}
    drive = {"id": "folder-a", "name": "Old", "parents": ["parent"]}
    checks, writes = [], []
    monkeypatch.setattr(folder_service, "resolve_registered_folder_path",
                        lambda _path, _db=None: ["folder-a"])
    monkeypatch.setattr(folder_service, "expand_registered_subtree",
                        lambda _folder, _db=None: ["folder-a"])
    monkeypatch.setattr(write, "_document_ids_in_folders", lambda _ids, _db=None: [])
    monkeypatch.setattr(write.folder_repository, "get_folder_by_id",
                        lambda _id, _db=None: folder)
    def name_exists(*_args, **_kwargs):
        checks.append("check")
        return len(checks) > 1
    monkeypatch.setattr(write.folder_repository, "folder_name_exists", name_exists)
    monkeypatch.setattr(write.drive_files, "get_file_metadata", lambda _id: drive)
    monkeypatch.setattr(write.drive_client, "rename_folder",
                        lambda *_args: writes.append("drive"))

    with pytest.raises(folder_service.FolderNameConflict):
        write.rename_folder(["Old"], "Taken", context=context)
    assert len(checks) == 2
    assert writes == []
    assert get_resource_guard(client, "tree:root-id").unsettled_effect_ids == ()


def test_add_fact_allocates_stable_point_before_write_and_confirmed_replay_skips_embed(
    monkeypatch, firestore_fake, fake_clock,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    point_id = str(uuid4())
    payload = {"type": "user_fact", "doc_id": "doc-1", "text": "remember this",
               "root_folder_id": "root-id", "parent_id": "folder-a", "dialog_date": "2025-01-01",
               "dialog_date_num": 20250101}
    record = {"status": "synced", "parent_id": "folder-a", "dialog_date": "2025-01-01"}
    points, writes = {}, []
    monkeypatch.setattr(write.user_facts, "prepare_user_fact", lambda *_a, **_k: ("doc-1", "remember this", payload))
    monkeypatch.setattr(write.transcripts, "get_transcript_record", lambda _id, _db=None: record)
    monkeypatch.setattr(write.qdrant_reader, "get_user_fact_payload", lambda _id: points.get(_id))
    real_allocate = write.allocate_fact_id
    def allocate(*args, **kwargs):
        kwargs["id_factory"] = lambda: point_id
        result = real_allocate(*args, **kwargs)
        assert get_effect_allocation(client, context.operation_id, context.effect_id)["resource_id"] == point_id
        return result
    monkeypatch.setattr(write, "allocate_fact_id", allocate)

    def apply(fact_payload, *, point_id, single_attempt, **_kwargs):
        assert get_effect_allocation(client, context.operation_id, context.effect_id)["resource_id"] == point_id
        assert single_attempt is True
        writes.append(point_id)
        points[point_id] = fact_payload

    monkeypatch.setattr(write.user_facts, "apply_user_fact_payload", apply)
    assert write.add_fact("doc-1", " remember this ", context=context) == "doc-1"
    assert write.add_fact("doc-1", " remember this ", context=context) == "doc-1"
    assert writes == [point_id]


def test_fact_embedding_expiry_stops_before_qdrant_upsert(
    monkeypatch, firestore_fake, fake_clock, sdk_fakes,
):
    client = _configure_firestore(monkeypatch, firestore_fake, fake_clock)
    context = _context(client, fake_clock)
    payload = {"type": "user_fact", "doc_id": "doc-1", "text": "remember this",
               "root_folder_id": "root-id", "parent_id": "folder-a",
               "dialog_date": "2025-01-01", "dialog_date_num": 20250101}
    monkeypatch.setattr(write.user_facts, "prepare_user_fact",
                        lambda *_args, **_kwargs: ("doc-1", "remember this", payload))
    monkeypatch.setattr(write.transcripts, "get_transcript_record",
                        lambda _id, _db=None: {"status": "synced", "parent_id": "folder-a"})
    monkeypatch.setattr(write.qdrant_reader, "get_user_fact_payload", lambda _id: None)
    from core.integrations.gemini import embeddings
    real_embed = embeddings.embed

    def slow_embed(texts, **kwargs):
        vectors = real_embed(texts, **kwargs)
        fake_clock.advance(seconds=61)
        return vectors

    monkeypatch.setattr(write.user_facts, "embed", slow_embed)
    sdk_fakes.gemini.queue("embed_content", response=types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    ))

    with pytest.raises(write.WriteError):
        write.add_fact("doc-1", "remember this", context=context)
    assert len([event for event in sdk_fakes.trace
                if event["boundary"] == "gemini" and event["operation"] == "embed_content"]) == 1
    assert not [event for event in sdk_fakes.trace
                if event["boundary"] == "qdrant" and event["operation"] == "upsert"]

