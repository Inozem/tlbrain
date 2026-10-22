import hashlib
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from google.cloud import firestore
from qdrant_client.models import SparseVector

from core.indexing.hashing import sha256_text, sha256_utterance
from core.indexing.payloads import build_utterance_payloads
from core.indexing.summaries import iter_windows
from core.integrations.fastembed.bm25 import get_bm25_model
from core.integrations.firestore.transcripts import (
    acquire_for_syncing,
    load_index,
    mark_error,
    mark_synced,
    update_index,
    update_skipped_utterances,
)
from core.integrations.gemini.embeddings import embed, make_client
from core.integrations.firestore import index_completions
from core.integrations.firestore.mutations import (
    acquire_index_guard, begin_index_effect, release_index_guard,
    renew_index_guard, settle_index_effect, get_unsettled_index_effects,
    settle_reconciled_index_effect,
)
from core.integrations.firestore.operations import get_operation
from core.integrations.firestore.save_intents import get_save_intent, SaveIntentState
from core.integrations.google_docs.reader import read_google_doc
from core.integrations.qdrant.writer import (
    delete_old_versions,
    delete_summaries_by_center_indexes,
    delete_utterances_by_order_indexes,
    set_payload_dialog_date,
    set_payload_parent_id,
    upsert_facts,
    upsert_summaries,
    upsert_utterances,
)
from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
from core.operations.retries import (
    RetryGovernor, UnknownOutcome, ConfirmedStageSuccess, ConfirmedStageFailure,
)
from core.operations.models import OperationError
from core.parsing.parser import parse_document
from core.parsing.windowing import generate_windows

logger = logging.getLogger(__name__)
_INDEX_LEASE_SECONDS = 300


@dataclass
class _IndexClaim:
    doc_id: str
    owner: str
    fence: int
    resource_epoch: int
    client: object
    sequence: int = 0


_current_index_claim: ContextVar[_IndexClaim | None] = ContextVar("current_index_claim", default=None)


def _readback_descriptor(kind: str, args: tuple, kwargs: dict) -> dict:
    """Persist only point IDs, selectors, and payload digests for reconciliation."""
    from core.integrations.qdrant.writer import _point_id
    if kind in {"upsert_utterances", "upsert_summaries", "upsert_facts"}:
        payloads = args[0]
        key_type = {"upsert_utterances": "utterance", "upsert_summaries": "summary", "upsert_facts": "fact"}[kind]
        points = []
        for payload in payloads:
            if key_type == "utterance":
                key = str(payload["order_index"])
            elif key_type == "summary":
                key = payload["summary_id"]
            else:
                key = f"{payload['summary_id']}:{payload['text']}"
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            points.append({"id": _point_id(key_type, payload["doc_id"], key), "payload_digest": digest})
        return {"action": "upsert", "points": points}
    doc_id = args[0]
    if kind in {"delete_utterances", "delete_summaries"}:
        return {"action": "delete", "kind": kind, "doc_id": doc_id,
                "root_folder_id": args[1], "indexes": list(args[2])}
    if kind == "delete_old_versions":
        return {"action": "delete_old_versions", "doc_id": doc_id,
                "new_version": args[1], "root_folder_id": args[2]}
    if kind in {"set_parent", "set_dialog_date"}:
        fields = {"parent_id": args[2]} if kind == "set_parent" else {"dialog_date": args[2]}
        if kind == "set_dialog_date" and args[3] is not None:
            fields["dialog_date_num"] = args[3]
        return {"action": "set_payload", "kind": kind, "doc_id": doc_id,
                "root_folder_id": args[1], "fields": fields}
    return {"action": "unverifiable"}


def _readback_confirms(intent: dict) -> bool:
    """Read Qdrant state for one durable effect; lack of evidence remains blocking."""
    from core.integrations.qdrant.client import get_client
    from core.integrations.qdrant.schema import get_collection_name
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    client = get_client()
    collection = get_collection_name()
    descriptor = intent.get("readback", {})
    action = descriptor.get("action")
    if action == "upsert":
        points = descriptor.get("points", [])
        if not points:
            return True
        found = client.retrieve(collection_name=collection, ids=[p["id"] for p in points], with_payload=True, with_vectors=False)
        payloads = {str(point.id): point.payload or {} for point in found}
        for expected in points:
            payload = payloads.get(expected["id"])
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest() if payload is not None else None
            if digest != expected["payload_digest"]:
                return False
        return True
    if action not in {"delete", "delete_old_versions", "set_payload"}:
        return False
    must = [FieldCondition(key="doc_id", match=MatchValue(value=descriptor["doc_id"])),
            FieldCondition(key="root_folder_id", match=MatchValue(value=descriptor["root_folder_id"]))]
    if action == "delete":
        if not descriptor.get("indexes"):
            return True
        kind = descriptor["kind"]
        if kind == "delete_utterances":
            must.extend([FieldCondition(key="type", match=MatchValue(value="utterance")),
                         FieldCondition(key="order_index", match=MatchAny(any=descriptor["indexes"]))])
        else:
            must.extend([FieldCondition(key="type", match=MatchAny(any=["summary", "fact"])),
                         FieldCondition(key="center_index", match=MatchAny(any=descriptor["indexes"]))])
    elif action == "delete_old_versions":
        must.append(FieldCondition(key="type", match=MatchAny(any=["utterance", "summary", "fact"])))
    elif action == "set_payload":
        must.append(FieldCondition(key="type", match=MatchAny(any=["utterance", "summary", "fact", "user_fact"])))
    filt = Filter(must=must)
    if action == "delete_old_versions":
        filt.must_not = [FieldCondition(key="version", match=MatchValue(value=descriptor["new_version"]))]
    offset = None
    all_points = []
    while True:
        points, offset = client.scroll(collection_name=collection, scroll_filter=filt, limit=256,
                                       offset=offset, with_payload=True, with_vectors=False)
        all_points.extend(points)
        if offset is None:
            break
    if action.startswith("delete"):
        return not all_points
    fields = descriptor["fields"]
    return all(all((point.payload or {}).get(key) == value for key, value in fields.items()) for point in all_points)


def _guarded_qdrant_call(kind: str, callback, *args, **kwargs):
    claim = _current_index_claim.get()
    if claim is None:
        return callback(*args, **kwargs)
    now = datetime.now(timezone.utc)
    if not renew_index_guard(
        claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
        now=now, lease_seconds=_INDEX_LEASE_SECONDS,
    ):
        raise RuntimeError("index resource lease was lost before Qdrant batch")
    claim.sequence += 1
    effect_id = f"index-{claim.fence}-{claim.sequence}"
    digest = hashlib.sha256(repr((args, sorted(kwargs.items()))).encode("utf-8")).hexdigest()
    if not begin_index_effect(
        claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
        effect_id=effect_id, kind=kind, payload_digest=digest, now=now,
        readback=_readback_descriptor(kind, args, kwargs),
    ):
        raise RuntimeError("could not reserve Qdrant batch under current index fence")
    try:
        kwargs.setdefault("retry_policy", SINGLE_ATTEMPT_RETRY_POLICY)
        result = callback(*args, **kwargs)
    except BaseException as error:
        known_rejection = getattr(error, "outcome", None) == "rejected"
        settle_index_effect(
            claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
            effect_id=effect_id, now=datetime.now(timezone.utc), confirmed=False,
            known_not_applied=known_rejection,
        )
        if not known_rejection:
            raise UnknownOutcome("Qdrant batch outcome is unknown") from error
        raise
    if not settle_index_effect(
        claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
        effect_id=effect_id, now=datetime.now(timezone.utc), confirmed=True,
    ):
        raise RuntimeError("Qdrant batch completed but could not be settled under its index fence")
    return result


def _renew_current_index_guard() -> None:
    claim = _current_index_claim.get()
    if claim is None:
        return
    if not renew_index_guard(
        claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
        now=datetime.now(timezone.utc), lease_seconds=_INDEX_LEASE_SECONDS,
    ):
        raise RuntimeError("index resource lease expired during indexing")


def _validate_managed_reference(reference: dict | None, doc_id: str, parent_id: str):
    if reference is None:
        return None
    required = {"schema_version", "outbox_id", "target_kind", "generation", "operation_id", "document_id", "save_intent_id"}
    if (not isinstance(reference, dict) or set(reference) != required
            or reference.get("schema_version") != 1 or reference.get("target_kind") != "index_dispatch"
            or reference.get("document_id") != doc_id):
        raise ValueError("invalid managed index reference")
    for field in ("outbox_id", "operation_id", "save_intent_id"):
        value = reference.get(field)
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"managed index reference {field} is invalid")
    if reference.get("generation") is None or isinstance(reference.get("generation"), bool):
        raise ValueError("managed index reference generation is invalid")
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.outbox import OutboxTargetKind, get_outbox_record
    client = get_client()
    operation = get_operation(client, reference["operation_id"])
    intent = get_save_intent(client, reference["save_intent_id"])
    outbox = get_outbox_record(client, reference["outbox_id"])
    if (operation is None or intent is None or operation.save_intent_id != intent.save_intent_id
            or outbox is None or outbox.target_kind != OutboxTargetKind.INDEX_DISPATCH
            or outbox.generation != reference["generation"]
            or outbox.operation_id != reference["operation_id"]
            or outbox.document_id != doc_id
            or outbox.save_intent_id != reference["save_intent_id"]
            or operation.doc_id not in (None, doc_id) or intent.doc_id != doc_id
            or intent.index_checkpoint_ref != reference["outbox_id"]
            or intent.state != SaveIntentState.CONFIRMED
            or operation.desired_content_hash != intent.desired_content_hash
            or intent.desired_parent_id_hash != sha256_text(parent_id)
            or not operation.immutable_input_ref or not intent.fill_checkpoint_ref
            or not intent.index_checkpoint_ref):
        raise ValueError("managed index reference does not match its confirmed save intent")
    return operation, intent


def verify_current_generation(doc_id: str, parent_id: str, root_folder_id: str, raw_text: str) -> bool:
    """Verify the reader projection against both transcript metadata and its receipt."""
    from core.integrations.firestore.client import get_client

    record = load_index(doc_id) or {}
    content_hash = sha256_text(raw_text)
    receipt = index_completions.get_completion(get_client(), doc_id, content_hash)
    if receipt is None or receipt.get("status") != "completed" or record.get("status") != "synced":
        return False
    metadata, _ = parse_document(raw_text)
    parent_hash = sha256_text(parent_id)
    identity = {
        "doc_id": doc_id, "root_folder_id": root_folder_id,
        "content_hash": content_hash, "version": content_hash,
        "parent_id": parent_id, "parent_id_hash": parent_hash,
        "provider": metadata.get("provider", ""),
        "fence": record.get("fence", 0), "index_fence": record.get("index_fence", 0),
        "resource_epoch": record.get("resource_epoch", 0),
    }
    return all(record.get(key) == value and receipt.get(key) == value for key, value in identity.items())


def process_one(
    doc_id: str, parent_id: str, root_folder_id: str, raw_text: str | None = None,
    *, managed_reference: dict | None = None,
) -> str:
    """Process one document and publish completion under a fenced index claim."""
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.transcripts import mark_error_if_index_fence

    client = get_client()
    binding = _validate_managed_reference(managed_reference, doc_id, parent_id)
    if binding is not None:
        operation, intent = binding
        if operation.state.value in {"completed", "failed"}:
            return "managed_terminal"
    claim_owner = f"index-worker-{uuid4().hex}"
    acquired = acquire_index_guard(
        client, doc_id=doc_id, owner=claim_owner, now=datetime.now(timezone.utc),
        lease_seconds=_INDEX_LEASE_SECONDS,
        root_folder_id=root_folder_id,
        operation_fence=binding[0].fence if binding is not None else 0,
        allow_unsettled_index_effects=binding is not None,
    )
    if acquired is None:
        return "not_acquired"
    index_fence, resource_epoch = acquired
    claim = _IndexClaim(doc_id, claim_owner, index_fence, resource_epoch, client)
    token = _current_index_claim.set(claim)
    try:
        if _has_unsettled_index_effect(client, doc_id) and not _reconcile_unknown_effects(claim):
            raise UnknownOutcome("prior Qdrant batch remains unverified")
        existing = load_index(doc_id)
        existing_utterance_hashes = (existing or {}).get("utterance_hashes")
        if raw_text is None:
            raw_text = read_google_doc(doc_id)
        content_hash = sha256_text(raw_text)
        if binding is not None and content_hash != binding[1].desired_content_hash:
            raise ValueError("managed save intent content hash does not match the reader projection")
        parent_id_hash = sha256_text(parent_id)
        text_in_sync = existing.get("content_hash") == content_hash if existing else False
        parent_in_sync = existing.get("parent_id_hash") == parent_id_hash if existing else False
        fast_result = _process_fast_skip_or_parent_only(
            doc_id, root_folder_id, parent_id, parent_id_hash,
            existing_utterance_hashes, text_in_sync, parent_in_sync,
            has_pending_windows=bool((existing or {}).get("pending_window_centers")),
        )
        if fast_result is not None:
            result = fast_result
        elif existing_utterance_hashes:
            if not parent_in_sync:
                _guarded_qdrant_call("set_parent", set_payload_parent_id, doc_id, root_folder_id, parent_id)
            result = _process_incremental(
                doc_id, parent_id, root_folder_id, raw_text, existing,
                existing_utterance_hashes, content_hash, parent_id_hash,
            )
        else:
            result = _process_full(
                doc_id, parent_id, root_folder_id, raw_text, existing,
                text_in_sync, parent_in_sync, content_hash, parent_id_hash,
            )

        current = load_index(doc_id) or {}
        metadata, _ = parse_document(raw_text)
        provider = metadata.get("provider") or current.get("provider")
        if not provider:
            raise ValueError("provider metadata is required to publish index completion")
        if binding is not None and binding[1].immutable_provider_digest is not None:
            if hashlib.sha256(provider.encode("utf-8")).hexdigest() != binding[1].immutable_provider_digest:
                raise ValueError("managed save intent provider does not match document metadata")
        if binding is None:
            owner, operation_fence = claim_owner, 0
            manifest_ref = f"legacy-index/{doc_id}"
            checkpoints = [f"transcript_index/{doc_id}/index/{index_fence}"]
        else:
            operation, intent = binding
            if intent.desired_parent_id_hash != parent_id_hash:
                raise ValueError("managed save intent parent hash does not match the route parent")
            owner, operation_fence = claim_owner, operation.fence
            manifest_ref = operation.immutable_input_ref
            checkpoints = [intent.fill_checkpoint_ref, intent.index_checkpoint_ref,
                           f"transcript_index/{doc_id}/index/{index_fence}"]
        receipt = {
            "doc_id": doc_id, "root_folder_id": root_folder_id,
            "content_hash": content_hash, "version": current.get("version", content_hash),
            "parent_id": parent_id, "parent_id_hash": parent_id_hash, "provider": provider,
            "work_manifest_ref": manifest_ref, "checkpoint_refs": checkpoints,
            "status": "completed", "skipped_utterances": current.get("skipped_utterances", []),
            "completed_at": datetime.now(timezone.utc), "error": None, "owner": owner,
            "fence": operation_fence, "index_fence": index_fence, "resource_epoch": resource_epoch,
        }
        _renew_current_index_guard()
        if not index_completions.publish_completion(client, receipt, now=datetime.now(timezone.utc)):
            raise RuntimeError("index completion became stale before publication")
        return result
    except Exception as e:
        mark_error_if_index_fence(doc_id, str(e), index_fence)
        logger.exception("Failed: %s", doc_id)
        raise
    finally:
        release_index_guard(
            client, doc_id=doc_id, owner=claim_owner, fence=index_fence,
            now=datetime.now(timezone.utc),
        )
        _current_index_claim.reset(token)


def process_managed_one(
    doc_id: str, parent_id: str, root_folder_id: str, raw_text: str,
    managed_reference: dict,
) -> str:
    """Run the managed indexing stage under its durable three-failure budget."""
    from core.integrations.firestore.client import get_client

    binding = _validate_managed_reference(managed_reference, doc_id, parent_id)
    if binding is None:
        raise ValueError("managed index reference is required")
    operation, intent = binding
    if operation.state.value in {"completed", "failed"}:
        return "managed_terminal"
    client = get_client()
    stage_key = "managed_index"
    governor = RetryGovernor(client, max_failures=3, lease_seconds=_INDEX_LEASE_SECONDS)

    def reconcile():
        receipt = index_completions.get_completion(client, doc_id, intent.desired_content_hash)
        if (receipt is not None and receipt.get("status") == "completed"
                and receipt.get("work_manifest_ref") == operation.immutable_input_ref
                and receipt.get("checkpoint_refs", [])[:2]
                == [intent.fill_checkpoint_ref, intent.index_checkpoint_ref]
                and verify_current_generation(doc_id, parent_id, root_folder_id, raw_text)):
            return ConfirmedStageSuccess("processed")
        # An earlier attempt may have dispatched a Qdrant batch. Until a read-back
        # reconciliation proves its effect, keep the stage in-flight and blocked.
        if _has_unsettled_index_effect(client, doc_id):
            owner = f"index-reconcile-{uuid4().hex}"
            acquired = acquire_index_guard(
                client, doc_id=doc_id, owner=owner, now=datetime.now(timezone.utc),
                lease_seconds=_INDEX_LEASE_SECONDS, root_folder_id=root_folder_id,
                operation_fence=operation.fence, allow_unsettled_index_effects=True,
            )
            if acquired is None:
                return None
            fence, epoch = acquired
            claim = _IndexClaim(doc_id, owner, fence, epoch, client)
            token = _current_index_claim.set(claim)
            try:
                confirmed = _reconcile_unknown_effects(claim)
            finally:
                release_index_guard(client, doc_id=doc_id, owner=owner, fence=fence,
                                    now=datetime.now(timezone.utc))
                _current_index_claim.reset(token)
            if not confirmed:
                return None
        return ConfirmedStageFailure(OperationError(
            code="index_attempt_incomplete", category="execution",
            message="Index attempt did not publish a completion receipt", retryable=True,
            stage=stage_key,
        ))

    def execute():
        result = process_one(
            doc_id, parent_id, root_folder_id, raw_text=raw_text,
            managed_reference=managed_reference,
        )
        if result == "not_acquired":
            raise RuntimeError("managed index resource is temporarily unavailable")
        return result

    return governor.run_stage(
        operation.operation_id, stage_key,
        execute,
        retryable=lambda _error: True,
        reconcile=reconcile,
        checkpoint_refs=(intent.fill_checkpoint_ref, intent.index_checkpoint_ref),
    )


def _has_unsettled_index_effect(client, doc_id: str) -> bool:
    return bool(get_unsettled_index_effects(client, doc_id))


def _reconcile_unknown_effects(claim: _IndexClaim) -> bool:
    intents = get_unsettled_index_effects(claim.client, claim.doc_id)
    for effect_id, intent in intents.items():
        if intent.get("state") not in {"unknown", "dispatched"}:
            return False
        if not renew_index_guard(
            claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
            now=datetime.now(timezone.utc), lease_seconds=_INDEX_LEASE_SECONDS,
            allow_unsettled_index_effects=True,
        ) or not _readback_confirms(intent):
            return False
        if not settle_reconciled_index_effect(
            claim.client, doc_id=claim.doc_id, owner=claim.owner, fence=claim.fence,
            effect_id=effect_id, now=datetime.now(timezone.utc),
        ):
            return False
    return not _has_unsettled_index_effect(claim.client, claim.doc_id)


def _process_fast_skip_or_parent_only(
    doc_id: str,
    root_folder_id: str,
    parent_id: str,
    parent_id_hash: str,
    existing_utterance_hashes: dict | None,
    text_in_sync: bool,
    parent_in_sync: bool,
    *,
    has_pending_windows: bool = False,
) -> str | None:
    if not existing_utterance_hashes:
        return None
    if has_pending_windows:
        return None
    if text_in_sync and parent_in_sync:
        logger.info("Skipped (in sync): %s", doc_id)
        return "skipped"
    if text_in_sync and not parent_in_sync:
        _guarded_qdrant_call("set_parent", set_payload_parent_id, doc_id, root_folder_id, parent_id)
        update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash})
        logger.info("Skipped (parent_id updated): %s", doc_id)
        return "skipped"
    return None


def _upsert_utterance_payloads(utterance_payloads: list[dict]) -> None:
    _renew_current_index_guard()
    bm25_embeddings = list(get_bm25_model().embed([u["text"] for u in utterance_payloads]))
    sparse_vectors = [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in bm25_embeddings
    ]
    _guarded_qdrant_call("upsert_utterances", upsert_utterances, utterance_payloads, sparse_vectors)


def _upsert_summary_fact_windows(
    utterances: list[dict],
    doc_id: str,
    version: str,
    parent_id: str,
    dialog_date: str,
    root_folder_id: str,
    *,
    allowed_center_indexes: set[int] | None = None,
) -> tuple[int, int, list]:
    embed_client = make_client()
    summaries_count = 0
    facts_count = 0
    skipped = []
    if allowed_center_indexes is None:
        windows = iter_windows(
            utterances, doc_id, version, parent_id, dialog_date, root_folder_id,
            skipped_utterances=skipped,
        )
    else:
        windows = iter_windows(
            utterances, doc_id, version, parent_id, dialog_date, root_folder_id,
            allowed_center_indexes=allowed_center_indexes,
            skipped_utterances=skipped,
        )
    for summary, facts in windows:
        _renew_current_index_guard()
        summary_vector = embed([summary["text"]], client=embed_client)[0]
        _guarded_qdrant_call("upsert_summaries", upsert_summaries, [summary], [summary_vector])
        if facts:
            _renew_current_index_guard()
            fact_vectors = embed([f["text"] for f in facts], client=embed_client)
            _guarded_qdrant_call("upsert_facts", upsert_facts, facts, fact_vectors)
        summaries_count += 1
        facts_count += len(facts)
    return summaries_count, facts_count, skipped


def _process_incremental(
    doc_id: str,
    parent_id: str,
    root_folder_id: str,
    raw_text: str,
    existing: dict,
    existing_utterance_hashes: dict,
    content_hash: str,
    parent_id_hash: str,
) -> str:
    metadata, utterances = parse_document(raw_text)
    del raw_text
    dialog_date = metadata.get("date", "")
    provider = metadata.get("provider", "")
    version = content_hash

    new_hashes = {
        str(u["order_index"]): sha256_utterance(u["order_index"], u["speaker"], u["text"])
        for u in utterances
    }
    old_keys = set(existing_utterance_hashes.keys())
    new_keys = set(new_hashes.keys())
    changed_str_keys = {
        k for k in (old_keys | new_keys)
        if existing_utterance_hashes.get(k) != new_hashes.get(k)
    }
    changed_indexes = [int(k) for k in changed_str_keys]
    pending_centers = set((existing or {}).get("pending_window_centers") or [])

    if not changed_indexes and not pending_centers:
        if dialog_date != (existing or {}).get("dialog_date", ""):
            try:
                dialog_date_num = int(dialog_date.replace("-", "")) if dialog_date else None
            except (ValueError, AttributeError):
                dialog_date_num = None
            _guarded_qdrant_call("set_dialog_date", set_payload_dialog_date, doc_id, root_folder_id, dialog_date, dialog_date_num)
        update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash, "content_hash": content_hash, "version": version, "dialog_date": dialog_date, "provider": provider})
        logger.info("Incremental skip (no utterance changes): %s", doc_id)
        return "skipped"

    affected_centers: set[int] = set()
    for i in changed_indexes:
        for c in range(i - 2, i + 3):
            if c >= 0:
                affected_centers.add(c)
    affected_centers.update(pending_centers)

    # Reserve all affected centers before changing Qdrant or advancing any
    # transcript checkpoint. A retry can then reconstruct these windows even
    # when the utterance hashes already match the new document.
    update_index(doc_id, {"pending_window_centers": sorted(affected_centers)})
    update_index(doc_id, {"dialog_date": dialog_date, "provider": provider})

    if changed_indexes:
        _guarded_qdrant_call("delete_utterances", delete_utterances_by_order_indexes, doc_id, root_folder_id, changed_indexes)
    _guarded_qdrant_call("delete_summaries", delete_summaries_by_center_indexes, doc_id, root_folder_id, list(affected_centers))

    new_changed_keys = changed_str_keys & new_keys
    changed_utterances = [u for u in utterances if str(u["order_index"]) in new_changed_keys]
    utterance_payloads = build_utterance_payloads(
        changed_utterances, doc_id, version, parent_id, dialog_date, root_folder_id
    )
    if utterance_payloads:
        _upsert_utterance_payloads(utterance_payloads)
        for u in utterance_payloads:
            i_str = str(u["order_index"])
            update_index(doc_id, {f"utterance_hashes.{i_str}": new_hashes[i_str]})

    summaries_count, facts_count, skipped = _upsert_summary_fact_windows(
        utterances, doc_id, version, parent_id, dialog_date, root_folder_id,
        allowed_center_indexes=affected_centers,
    )

    reanalyzed = {
        u["order_index"]
        for w in generate_windows(utterances)
        if w["center_index"] in affected_centers
        for u in w["utterances"]
    }
    existing_skipped = [i for i in ((existing or {}).get("skipped_utterances") or [])
                        if i not in reanalyzed]
    update_skipped_utterances(doc_id, existing_skipped + skipped)
    update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash, "content_hash": content_hash, "version": version})
    update_index(doc_id, {"utterance_hashes": new_hashes})

    speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
    update_index(doc_id, {"speakers": speakers})
    update_index(doc_id, {"pending_window_centers": firestore.DELETE_FIELD})
    logger.info(
        "Incremental sync: %s | changed=%d summaries=%d facts=%d",
        doc_id, len(changed_indexes), summaries_count, facts_count,
    )
    return "processed"


def _process_full(
    doc_id: str,
    parent_id: str,
    root_folder_id: str,
    raw_text: str,
    existing: dict | None,
    text_in_sync: bool,
    parent_in_sync: bool,
    content_hash: str,
    parent_id_hash: str,
) -> str:
    if existing and text_in_sync and parent_in_sync:
        logger.info("Skipped unchanged: %s", doc_id)
        return "skipped"

    metadata, utterances = parse_document(raw_text)
    del raw_text
    dialog_date = metadata.get("date", "")
    provider = metadata.get("provider", "")
    version = content_hash

    update_index(doc_id, {"dialog_date": dialog_date, "provider": provider})
    utterance_payloads = build_utterance_payloads(
        utterances, doc_id, version, parent_id, dialog_date, root_folder_id
    )
    _upsert_utterance_payloads(utterance_payloads)

    summaries_count, facts_count, skipped = _upsert_summary_fact_windows(
        utterances, doc_id, version, parent_id, dialog_date, root_folder_id,
    )

    _guarded_qdrant_call("delete_old_versions", delete_old_versions, doc_id, version, root_folder_id)
    update_skipped_utterances(doc_id, skipped)
    update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash, "content_hash": content_hash, "version": version})

    utterance_hashes = {
        str(u["order_index"]): sha256_utterance(u["order_index"], u["speaker"], u["text"])
        for u in utterance_payloads
    }
    update_index(doc_id, {"utterance_hashes": utterance_hashes})
    speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
    update_index(doc_id, {"speakers": speakers})
    logger.info(
        "Processed: %s | utterances=%d summaries=%d facts=%d",
        doc_id, len(utterance_payloads), summaries_count, facts_count,
    )
    return "processed"
