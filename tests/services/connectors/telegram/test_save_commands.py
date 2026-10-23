"""Fenced persistence tests for structured-router save confirmation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from core.integrations.firestore.operation_queue import claim_head
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.repository import SESSIONS, TelegramRepository


pytestmark = pytest.mark.offline
NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _setup(firestore_fake, *, with_source=True):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    if with_source:
        forward = repository.accept_forward(
            1, chat_id=42, message={"message_id": 101, "text": "captured source"}, now=NOW,
        )
        draft_id = forward.draft_id
    else:
        forward = None
        draft_id = None
    accepted = repository.accept_command(
        2, chat_id=42, user_id=7, command=parse_command('"save"'), now=NOW,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="router-worker", now=NOW)
    assert lease is not None
    return client, repository, accepted, context_ref, lease, draft_id


def _all_documents(firestore_fake):
    return deepcopy(firestore_fake.documents)


def test_confirmation_seals_only_the_captured_candidate_and_replays_idempotently(firestore_fake):
    client, repository, accepted, context_ref, lease, draft_id = _setup(firestore_fake)
    context_ref.update({"active_draft_id": "newer", "draft_status": "collecting",
                        "selected_dialog_number": 9, "selection_sequence": 12,
                        "pending_save_operation_id": "stale-save-operation"})
    context_ref.collection("drafts").document("newer").set({
        "draft_id": "newer", "state": "collecting", "ordered_message_ids": [202],
    })
    stale_child = repository.accept_command(
        3, chat_id=42, user_id=7, command=parse_command("keep this snapshot"), now=NOW,
    )
    stale_child_command_ref = context_ref.collection("commands").document("3")
    stale_child_snapshot = stale_child_command_ref.get().to_dict()
    assert "pending_save_candidate_operation_id" not in stale_child_snapshot
    assert repository.confirm_save_command(
        accepted.context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    draft = repository.get_draft(accepted.context_id, draft_id)
    assert draft["state"] == "processing"
    assert draft["ordered_message_ids"] == [101]
    assert draft["active_save_operation_id"] == lease.operation_id
    assert draft["immutable_save_intent_id"] == lease.operation_id
    assert context_ref.get().to_dict()["active_draft_id"] == "newer"
    assert context_ref.get().to_dict()["selected_dialog_number"] == 9
    assert context_ref.get().to_dict()["pending_save_operation_id"] == lease.operation_id
    assert stale_child_command_ref.get().to_dict() == stale_child_snapshot
    assert stale_child.operation_id != lease.operation_id
    operation = context_ref.collection("operations").document(lease.operation_id).get().to_dict()
    assert operation["route_result"] == {"kind": "save", "target_draft_id": draft_id}
    assert operation["target_draft_id"] == draft_id
    assert context_ref.get().to_dict()["pending_save_operation_id"] == lease.operation_id
    following = repository.accept_command(
        4, chat_id=42, user_id=7, command=parse_command("move last"), now=NOW,
    )
    candidate = context_ref.collection("operations").document(following.operation_id).get().to_dict()
    assert candidate["predecessor_operation_id"] == stale_child.operation_id
    assert candidate["pending_save_candidate_operation_id"] == lease.operation_id

    manifest_ref = context_ref.collection("drafts").document(draft_id).collection(
        "save_manifests"
    ).document(lease.operation_id)
    before = manifest_ref.get().to_dict()
    assert repository.confirm_save_command(
        accepted.context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    assert manifest_ref.get().to_dict() == before
    assert repository.get_draft(accepted.context_id, draft_id)["sealed_at"] == NOW


@pytest.mark.parametrize("failure", [
    "wrong_owner", "wrong_fence", "expired", "changed_sources", "route_conflict",
    "terminal_operation", "non_head_operation", "foreign_input_ref", "mismatched_command",
    "manifest_conflict", "target_draft_conflict", "target_dialog_conflict",
    "target_conversation_conflict", "target_doc_conflict",
])
def test_rejected_confirmation_is_write_free(firestore_fake, failure):
    client, repository, accepted, context_ref, lease, draft_id = _setup(firestore_fake)
    confirm_now = NOW
    owner, fence = lease.owner, lease.fence
    if failure == "wrong_owner":
        owner = "other-worker"
    elif failure == "wrong_fence":
        fence += 1
    elif failure == "expired":
        confirm_now = lease.expires_at + timedelta(seconds=1)
    elif failure == "changed_sources":
        context_ref.collection("drafts").document(draft_id).update({"ordered_message_ids": [999]})
    elif failure == "route_conflict":
        context_ref.collection("operations").document(lease.operation_id).update({
            "route_result": {"kind": "freeform", "target_draft_id": draft_id},
        })
    elif failure == "terminal_operation":
        context_ref.collection("operations").document(lease.operation_id).update({"state": "completed"})
    elif failure == "non_head_operation":
        child = repository.accept_command(
            9, chat_id=42, user_id=7, command=parse_command("queued"), now=NOW,
        )
        context_ref.update({"reserved_operation_id": child.operation_id})
    elif failure == "foreign_input_ref":
        context_ref.collection("operations").document(lease.operation_id).update({
            "input_ref": "telegram_sessions/foreign_context/commands/2",
        })
    elif failure == "mismatched_command":
        context_ref.collection("commands").document("2").update({"operation_id": "foreign-op"})
    elif failure == "manifest_conflict":
        context_ref.collection("drafts").document(draft_id).collection(
            "save_manifests"
        ).document(lease.operation_id).set({"manifest_digest": "foreign-manifest"})
    elif failure.startswith("target_"):
        field, value = {
            "target_draft_conflict": ("target_draft_id", "foreign-draft"),
            "target_dialog_conflict": ("target_dialog_number", 999),
            "target_conversation_conflict": ("target_conversation_id", "foreign-conversation"),
            "target_doc_conflict": ("target_doc_id", "foreign-doc"),
        }[failure]
        context_ref.collection("operations").document(lease.operation_id).update({
            "route_result": {"kind": "save", field: value},
        })
    before = _all_documents(firestore_fake)

    assert repository.confirm_save_command(
        accepted.context_id, lease.operation_id, owner=owner, fence=fence, now=confirm_now,
    ) is False
    assert firestore_fake.documents == before


def test_empty_candidate_confirmation_creates_no_draft_or_manifest(firestore_fake):
    client, repository, accepted, context_ref, lease, _ = _setup(firestore_fake, with_source=False)
    before = _all_documents(firestore_fake)

    assert repository.confirm_save_command(
        accepted.context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is False
    assert firestore_fake.documents == before


def test_matching_legacy_seal_binds_missing_route_without_rewriting_progress(firestore_fake):
    client, repository, accepted, context_ref, lease, draft_id = _setup(firestore_fake)
    draft_ref = context_ref.collection("drafts").document(draft_id)
    draft = draft_ref.get().to_dict()
    digest = hashlib.sha256(json.dumps(
        {"draft_id": draft_id, "message_ids": [101]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    manifest_ref = draft_ref.collection("save_manifests").document(lease.operation_id)
    manifest_path = (
        f"{SESSIONS}/{accepted.context_id}/drafts/{draft_id}/save_manifests/{lease.operation_id}"
    )
    draft.update({
        "state": "processing", "sealed_at": NOW, "manifest_ref": manifest_path,
        "manifest_digest": digest, "active_save_operation_id": lease.operation_id,
        "immutable_save_intent_id": lease.operation_id, "doc_id": "existing-doc",
        "prepared_count": 3, "required_audio_count": 1, "completed_audio_ids": [101],
        "context_capacity": 25, "preparation_digest": "preserve-preparation",
    })
    manifest = {
        "draft_id": draft_id, "ordered_message_ids": [101], "manifest_digest": digest,
        "save_operation_id": lease.operation_id, "sealed_at": NOW,
    }
    draft_ref.set(draft)
    manifest_ref.set(manifest)
    context_ref.update({"pending_save_operation_id": "stale-save-operation"})
    operation_ref = context_ref.collection("operations").document(lease.operation_id)
    operation_ref.update({
        "target_doc_id": "existing-doc",
        "target_dialog_number": draft["dialog_number"],
        "target_conversation_id": draft["conversation_id"],
    })
    before_draft = draft_ref.get().to_dict()
    before_manifest = manifest_ref.get().to_dict()

    assert repository.confirm_save_command(
        accepted.context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    assert draft_ref.get().to_dict() == before_draft
    assert manifest_ref.get().to_dict() == before_manifest
    operation = operation_ref.get().to_dict()
    assert operation["route_result"] == {"kind": "save", "target_draft_id": draft_id}
    assert operation["target_doc_id"] == "existing-doc"
    assert context_ref.get().to_dict()["pending_save_operation_id"] == lease.operation_id

