"""Certificate-scoped removal of temporary Telegram save payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.integrations.firestore.input_parts import MAX_PART_BYTES
from services.connectors.telegram.runtime.ledger import _INLINE_CHECKPOINT_VALUE_BYTES
from services.connectors.telegram.save_checkpoints import stage_payload_ref

_STAGE_KINDS = {
    "telegram.preparation": "preparation",
    "telegram.analysis": "analysis",
    "telegram.classification": "classification",
}
_ALLOCATION_KINDS = {f"save-stage-{kind}" for kind in _STAGE_KINDS.values()}
_ALLOCATION_FIELDS = {
    "allocation_id", "context_id", "operation_id", "draft_id", "manifest_digest",
    "save_intent_id", "kind", "input_ref", "sha256", "byte_length",
    "max_part_bytes", "cleanup_state",
}


def _data_operation(context_id: str, operation_id: str, kind: str, digest: str) -> str:
    identity = f"{context_id}\0{operation_id}\0{kind}\0{digest}"
    return "telegram-ledger-" + hashlib.sha256(identity.encode()).hexdigest()


def _allocation_document_id(kind: str, digest: str) -> str:
    return hashlib.sha256(f"{kind}\0{digest}".encode()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    def encode(item: Any) -> Any:
        if isinstance(item, bytes):
            import base64
            return {"__telegram_bytes__": base64.b64encode(item).decode("ascii")}
        raise TypeError(f"unsupported durable agent value: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=encode).encode("utf-8")


class TemporarySavePayloadCleanup:
    """Purge only checkpoint payloads named by an authorized sealed save."""

    def __init__(self, client: Any, *, context_id: str, authorize: Any = None,
                 batch_size: int = 400):
        self.client, self.context_id = client, context_id
        self.authorize = authorize
        self.batch_size = min(max(int(batch_size), 1), 400)

    def cleanup(self, certificate: Mapping[str, Any]) -> bool:
        """Authorize once, collect proof-scoped work, then apply bounded cleanup."""
        scope = self._authorized_cleanup_scope(certificate)
        if scope is None:
            return False
        pending, payload_refs, allocations, complete = self._collect_cleanup_mutations(scope)
        self._apply_tombstones(pending)
        self._mark_allocations_purging(allocations)
        if not self._delete_parts(payload_refs):
            return False
        if not self._delete_allocation_parts(allocations):
            return False
        self._mark_allocations_purged(allocations)
        return complete

    def _authorized_cleanup_scope(
        self, certificate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Load the immutable manifest and exact cleanup grant before scanning records."""
        context_id = certificate.get("context_id")
        draft_id = certificate.get("draft_id")
        digest = certificate.get("manifest_digest")
        intent_id = certificate.get("save_intent_id")
        operation_id = certificate.get("operation_id")
        if (context_id != self.context_id or not all(isinstance(value, str) and value
                for value in (draft_id, digest, intent_id, operation_id))):
            return None
        if not callable(self.authorize) or self.authorize(context_id, certificate) is not True:
            return None
        session = self.client.collection("telegram_sessions").document(context_id)
        draft_ref = session.collection("drafts").document(draft_id)
        draft = draft_ref.get().to_dict() or {}
        manifest = draft_ref.collection("save_manifests").document(operation_id).get().to_dict() or {}
        authorization = draft.get("cleanup_authorization") or {}
        expected_binding = {
            "context_id": context_id, "draft_id": draft_id, "manifest_digest": digest,
        }
        if (draft.get("cleanup_state") not in {"authorized", "cleaned"}
                or draft.get("manifest_digest") != digest
                or manifest.get("manifest_digest") != digest
                or manifest.get("draft_id") != draft_id
                or authorization.get("save_intent_id") != intent_id
                or authorization.get("operation_id") != operation_id
                or authorization.get("origin_binding") != expected_binding):
            return None
        operation_ids = self._proven_operations(
            session, draft_ref, draft_id, digest, intent_id, manifest,
        )
        if operation_id not in operation_ids:
            return None
        ordered_ids = manifest.get("ordered_message_ids")
        allowed_audio = ({f"telegram.audio.{item}" for item in ordered_ids if type(item) is int}
                         if isinstance(ordered_ids, list) else set())
        return {
            "context_id": context_id, "draft_id": draft_id, "digest": digest,
            "intent_id": intent_id, "operation_id": operation_id,
            "session": session, "operation_ids": operation_ids,
            "allowed_audio": allowed_audio,
        }

    def _collect_cleanup_mutations(
        self, scope: Mapping[str, Any],
    ) -> tuple[list[tuple[Any, str, Any]], list[str], list[tuple[Any, dict[str, Any]]], bool]:
        """Collect verified stage, request, and allocation tombstones without writing."""
        context_id = scope["context_id"]
        draft_id = scope["draft_id"]
        digest = scope["digest"]
        intent_id = scope["intent_id"]
        session = scope["session"]
        operation_ids = scope["operation_ids"]
        allowed_audio = scope["allowed_audio"]
        pending: list[tuple[Any, str, Any]] = []
        payload_refs: list[str] = []
        allocations: list[tuple[Any, dict[str, Any]]] = []
        complete = True
        for source_operation_id in operation_ids:
            op_ref = session.collection("operations").document(source_operation_id)
            for allocation_ref in op_ref.collection("save_payloads").stream():
                allocation = allocation_ref.to_dict() or {}
                if not self._verify_allocation(
                    allocation, allocation_ref.id, context_id=context_id,
                    source_operation_id=source_operation_id, draft_id=draft_id,
                    digest=digest, intent_id=intent_id,
                ):
                    complete = False
                    continue
                allocations.append((allocation_ref.reference, allocation))
            for stage_ref in op_ref.collection("stages").stream():
                row = stage_ref.to_dict() or {}
                key = stage_ref.id
                if row.get("state") != "completed" or row.get("stage_key") != key:
                    continue
                if key in _STAGE_KINDS:
                    kind = "save-stage-" + _STAGE_KINDS[key]
                    result = row.get("result")
                    if isinstance(result, Mapping) and result.get("purged") is True:
                        if not self._verify_tombstone(result, context_id, operation_ids, kind):
                            complete = False
                        elif isinstance(result.get("purged_input_ref"), str):
                            payload_refs.append(result["purged_input_ref"])
                        continue
                    descriptor = stage_payload_ref(result)
                    if descriptor is not None:
                        verified = self._verify_any_owner(
                            descriptor, context_id, operation_ids, kind,
                        )
                        if verified:
                            verified_digest, payload_owner = verified
                            tombstone = {
                                "purged": True, "purged_sha256": verified_digest,
                                "purged_byte_length": descriptor["byte_length"],
                                "purged_kind": kind, "purged_owner": payload_owner,
                            }
                            if isinstance(descriptor, Mapping) and isinstance(descriptor.get("_input_ref"), str):
                                payload_refs.append(descriptor["_input_ref"])
                                tombstone["purged_input_ref"] = descriptor["_input_ref"]
                            pending.append((stage_ref.reference, "result", tombstone))
                        else:
                            complete = False
                    else:
                        complete = False
                elif key in allowed_audio:
                    pending.append((stage_ref.reference, "result", {
                        "purged": True, "purged_kind": "audio-transcript",
                    }))
            request_ref = op_ref.collection("agent_requests").document("request")
            request = request_ref.get().to_dict() or {}
            expected_conversation = "telegram-global-" + hashlib.sha256(
                f"{context_id}\0{source_operation_id}".encode(),
            ).hexdigest()
            if (request.get("delivery_id") == source_operation_id
                    and request.get("conversation_id") == expected_conversation):
                descriptor = request.get("request")
                if isinstance(descriptor, Mapping) and descriptor.get("purged") is True:
                    if not self._verify_tombstone(descriptor, context_id, operation_ids, "request"):
                        complete = False
                    elif isinstance(descriptor.get("purged_input_ref"), str):
                        payload_refs.append(descriptor["purged_input_ref"])
                    continue
                verified = self._verify_any_owner(descriptor, context_id, [source_operation_id], "request")
                if verified:
                    verified_digest, payload_owner = verified
                    tombstone = {
                        "purged": True, "purged_sha256": verified_digest,
                        "purged_byte_length": descriptor["byte_length"],
                        "purged_kind": "request", "purged_owner": payload_owner,
                    }
                    if isinstance(descriptor, Mapping) and isinstance(descriptor.get("_input_ref"), str):
                        payload_refs.append(descriptor["_input_ref"])
                        tombstone["purged_input_ref"] = descriptor["_input_ref"]
                    pending.append((request_ref, "request", tombstone))
                else:
                    complete = False
        return pending, payload_refs, allocations, complete

    def _proven_operations(self, session: Any, draft_ref: Any, draft_id: str,
                           digest: str, intent_id: str, current: Mapping[str, Any]) -> list[str]:
        ids: set[str] = set()
        wanted_ids = current.get("ordered_message_ids")
        for snap in draft_ref.collection("save_manifests").stream():
            row = snap.to_dict() or {}
            source_id = row.get("save_operation_id")
            if (row.get("draft_id") != draft_id or row.get("manifest_digest") != digest
                    or row.get("ordered_message_ids") != wanted_ids or source_id != snap.id):
                continue
            queue = session.collection("operations").document(source_id).get().to_dict() or {}
            route = queue.get("route_control")
            if (queue.get("operation_id") != source_id
                    or (queue.get("target_draft_id") or queue.get("draft_id")) != draft_id
                    or queue.get("save_operation_id") != source_id
                    or (queue.get("save_intent_id") or queue.get("save_operation_id") or "") != intent_id
                    or not isinstance(route, Mapping)
                    or route.get("status") not in {"ok", "success"}
                    or route.get("kind") not in {"save", "retry"}
                    or (route.get("kind") == "retry"
                        and (type(route.get("dialog_number")) is not int
                             or route.get("dialog_number") < 1))):
                continue
            ids.add(source_id)
        return sorted(ids)

    def _verify_descriptor(self, descriptor: Any, context_id: str,
                           operation_id: str, kind: str) -> str | None:
        if not isinstance(descriptor, Mapping):
            return None
        digest, length = descriptor.get("sha256"), descriptor.get("byte_length")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or type(length) is not int or length < 0):
            return None
        if "_inline" in descriptor:
            if (set(descriptor) != {"_inline", "sha256", "byte_length"}
                    or length > _INLINE_CHECKPOINT_VALUE_BYTES):
                return None
            try:
                payload = _json_bytes(descriptor["_inline"])
            except (TypeError, ValueError):
                return None
            return digest if len(payload) == length and hashlib.sha256(payload).hexdigest() == digest else None
        if (set(descriptor) != {"_input_ref", "sha256", "byte_length", "max_part_bytes"}
                or length <= _INLINE_CHECKPOINT_VALUE_BYTES):
            return None
        expected_id = _data_operation(context_id, operation_id, kind, digest)
        expected_ref = f"dialog_operations/{expected_id}/input_parts/manifest"
        if descriptor.get("_input_ref") != expected_ref or descriptor.get("max_part_bytes") != MAX_PART_BYTES:
            return None
        manifest = (self.client.collection("dialog_operations").document(expected_id)
                    .collection("input_parts").document("manifest").get().to_dict() or {})
        if (manifest.get("input_ref") != expected_ref or manifest.get("ready") is not True
                or manifest.get("sha256") != digest or manifest.get("byte_length") != length):
            return None
        return digest

    def _verify_any_owner(self, descriptor: Any, context_id: str,
                          operation_ids: list[str], kind: str) -> tuple[str, str] | None:
        for owner in operation_ids:
            verified = self._verify_descriptor(descriptor, context_id, owner, kind)
            if verified is not None:
                return verified, owner
        return None

    def _verify_tombstone(self, tombstone: Mapping[str, Any], context_id: str,
                          operation_ids: list[str], kind: str) -> bool:
        allowed = {
            "purged", "purged_sha256", "purged_byte_length",
            "purged_kind", "purged_owner",
        }
        if isinstance(tombstone.get("purged_input_ref"), str):
            allowed.add("purged_input_ref")
        if (set(tombstone) != allowed or tombstone.get("purged") is not True
                or tombstone.get("purged_kind") != kind):
            return False
        digest = tombstone.get("purged_sha256")
        length = tombstone.get("purged_byte_length")
        owner = tombstone.get("purged_owner")
        input_ref = tombstone.get("purged_input_ref")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or type(length) is not int or length < 0
                or owner not in operation_ids):
            return False
        if input_ref is None:
            return True
        descriptor = {
            "_input_ref": input_ref, "sha256": digest, "byte_length": length,
            "max_part_bytes": MAX_PART_BYTES,
        }
        return self._verify_descriptor(descriptor, context_id, owner, kind) is not None

    def _verify_allocation(self, allocation: Mapping[str, Any], document_id: str, *,
                           context_id: str, source_operation_id: str, draft_id: str,
                           digest: str, intent_id: str) -> bool:
        if set(allocation) != _ALLOCATION_FIELDS:
            return False
        kind, payload_digest = allocation.get("kind"), allocation.get("sha256")
        if (kind not in _ALLOCATION_KINDS
                or allocation.get("allocation_id") != document_id
                or document_id != _allocation_document_id(kind, payload_digest)
                or allocation.get("context_id") != context_id
                or allocation.get("operation_id") != source_operation_id
                or allocation.get("draft_id") != draft_id
                or allocation.get("manifest_digest") != digest
                or allocation.get("save_intent_id") != intent_id
                or allocation.get("cleanup_state") not in {"allocated", "purging", "purged"}
                or not isinstance(payload_digest, str) or len(payload_digest) != 64
                or any(char not in "0123456789abcdef" for char in payload_digest)
                or type(allocation.get("byte_length")) is not int
                or allocation["byte_length"] <= _INLINE_CHECKPOINT_VALUE_BYTES
                or allocation.get("max_part_bytes") != MAX_PART_BYTES):
            return False
        expected_ref = f"dialog_operations/{_data_operation(context_id, source_operation_id, kind, payload_digest)}/input_parts/manifest"
        if allocation.get("input_ref") != expected_ref:
            return False
        header = self._part_manifest(expected_ref)
        if header is not None and (
                header.get("input_ref") != expected_ref
                or header.get("sha256") != payload_digest
                or header.get("byte_length") != allocation["byte_length"]
                or header.get("ready") is not True):
            return False
        return True

    def _part_manifest(self, input_ref: str) -> dict[str, Any] | None:
        parts = input_ref.split("/")
        if (len(parts) != 4 or parts[0] != "dialog_operations"
                or parts[2:] != ["input_parts", "manifest"]):
            return {}
        snapshot = (self.client.collection("dialog_operations").document(parts[1])
                    .collection("input_parts").document("manifest").get())
        return (snapshot.to_dict() or {}) if snapshot.exists else None

    def _mark_allocations_purging(self, allocations: list[tuple[Any, dict[str, Any]]]) -> None:
        pending = [(ref, {**row, "cleanup_state": "purging"})
                   for ref, row in allocations if row.get("cleanup_state") != "purged"]
        for offset in range(0, len(pending), self.batch_size):
            batch = self.client.batch()
            for ref, row in pending[offset:offset + self.batch_size]:
                batch.set(ref, row, merge=False)
            batch.commit()

    def _mark_allocations_purged(self, allocations: list[tuple[Any, dict[str, Any]]]) -> None:
        pending = [(ref, {**row, "cleanup_state": "purged"}) for ref, row in allocations]
        for offset in range(0, len(pending), self.batch_size):
            batch = self.client.batch()
            for ref, row in pending[offset:offset + self.batch_size]:
                batch.set(ref, row, merge=False)
            batch.commit()

    def _delete_allocation_parts(self, allocations: list[tuple[Any, dict[str, Any]]]) -> bool:
        for _ref, allocation in allocations:
            input_ref = allocation["input_ref"]
            parts = input_ref.split("/")
            collection = (self.client.collection("dialog_operations").document(parts[1])
                          .collection("input_parts"))
            header = collection.document("manifest").get()
            if header.exists:
                row = header.to_dict() or {}
                if (row.get("input_ref") != input_ref
                        or row.get("sha256") != allocation["sha256"]
                        or row.get("byte_length") != allocation["byte_length"]
                        or row.get("ready") is not True):
                    return False
            while True:
                page = list(collection.where("index", ">=", 0).limit(self.batch_size).stream())
                if not page:
                    break
                if any((part.to_dict() or {}).get("input_ref") != input_ref for part in page):
                    return False
                batch = self.client.batch()
                for part in page:
                    batch.delete(part.reference)
                batch.commit()
        return True

    def _apply_tombstones(self, pending: list[tuple[Any, str, Any]]) -> None:
        for offset in range(0, len(pending), self.batch_size):
            batch = self.client.batch()
            for ref, field, tombstone in pending[offset:offset + self.batch_size]:
                batch.update(ref, {field: tombstone})
            batch.commit()

    def _delete_parts(self, manifest_refs: list[str]) -> bool:
        for manifest_ref in sorted(set(manifest_refs)):
            parts = manifest_ref.split("/")
            if (len(parts) != 4 or parts[0] != "dialog_operations"
                    or parts[2:] != ["input_parts", "manifest"]):
                return False
            collection = self.client.collection("dialog_operations").document(parts[1]).collection("input_parts")
            header = collection.document("manifest").get().to_dict() or {}
            if header.get("input_ref") != manifest_ref or header.get("ready") is not True:
                return False
            while True:
                parts = list(collection.where("index", ">=", 0).limit(self.batch_size).stream())
                if not parts:
                    break
                if any((part.to_dict() or {}).get("input_ref") != manifest_ref for part in parts):
                    return False
                batch = self.client.batch()
                for part in parts:
                    batch.delete(part.reference)
                batch.commit()
        return True
