import logging

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
from core.parsing.parser import parse_document
from core.parsing.windowing import generate_windows

logger = logging.getLogger(__name__)


def process_one(doc_id: str, parent_id: str, root_folder_id: str, raw_text: str | None = None) -> str:
    """Process one document and return its sync result."""
    if not acquire_for_syncing(doc_id):
        logger.info("Not acquired (already syncing): %s", doc_id)
        return "not_acquired"

    try:
        existing = load_index(doc_id)
        existing_utterance_hashes = (existing or {}).get("utterance_hashes")
        if raw_text is None:
            raw_text = read_google_doc(doc_id)

        content_hash = sha256_text(raw_text)
        parent_id_hash = sha256_text(parent_id)
        text_in_sync = existing.get("content_hash") == content_hash if existing else False
        parent_in_sync = existing.get("parent_id_hash") == parent_id_hash if existing else False

        fast_result = _process_fast_skip_or_parent_only(
            doc_id, root_folder_id, parent_id, parent_id_hash,
            existing_utterance_hashes, text_in_sync, parent_in_sync,
        )
        if fast_result is not None:
            return fast_result

        if existing_utterance_hashes and not parent_in_sync:
            set_payload_parent_id(doc_id, root_folder_id, parent_id)

        if existing_utterance_hashes:
            return _process_incremental(
                doc_id, parent_id, root_folder_id, raw_text, existing,
                existing_utterance_hashes, content_hash, parent_id_hash,
            )

        return _process_full(
            doc_id, parent_id, root_folder_id, raw_text, existing,
            text_in_sync, parent_in_sync, content_hash, parent_id_hash,
        )
    except Exception as e:
        mark_error(doc_id, str(e))
        logger.exception("Failed: %s", doc_id)
        raise


def _process_fast_skip_or_parent_only(
    doc_id: str,
    root_folder_id: str,
    parent_id: str,
    parent_id_hash: str,
    existing_utterance_hashes: dict | None,
    text_in_sync: bool,
    parent_in_sync: bool,
) -> str | None:
    if not existing_utterance_hashes:
        return None
    if text_in_sync and parent_in_sync:
        mark_synced(doc_id)
        logger.info("Skipped (in sync): %s", doc_id)
        return "skipped"
    if text_in_sync and not parent_in_sync:
        set_payload_parent_id(doc_id, root_folder_id, parent_id)
        update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash})
        mark_synced(doc_id)
        logger.info("Skipped (parent_id updated): %s", doc_id)
        return "skipped"
    return None


def _upsert_utterance_payloads(utterance_payloads: list[dict]) -> None:
    bm25_embeddings = list(get_bm25_model().embed([u["text"] for u in utterance_payloads]))
    sparse_vectors = [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in bm25_embeddings
    ]
    upsert_utterances(utterance_payloads, sparse_vectors)


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
        summary_vector = embed([summary["text"]], client=embed_client)[0]
        upsert_summaries([summary], [summary_vector])
        if facts:
            fact_vectors = embed([f["text"] for f in facts], client=embed_client)
            upsert_facts(facts, fact_vectors)
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

    if not changed_indexes:
        if dialog_date != (existing or {}).get("dialog_date", ""):
            try:
                dialog_date_num = int(dialog_date.replace("-", "")) if dialog_date else None
            except (ValueError, AttributeError):
                dialog_date_num = None
            set_payload_dialog_date(doc_id, root_folder_id, dialog_date, dialog_date_num)
        update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash, "content_hash": content_hash, "version": version, "dialog_date": dialog_date, "provider": provider})
        mark_synced(doc_id)
        logger.info("Incremental skip (no utterance changes): %s", doc_id)
        return "skipped"

    update_index(doc_id, {"dialog_date": dialog_date, "provider": provider})

    affected_centers: set[int] = set()
    for i in changed_indexes:
        for c in range(i - 2, i + 3):
            if c >= 0:
                affected_centers.add(c)

    delete_utterances_by_order_indexes(doc_id, root_folder_id, changed_indexes)
    delete_summaries_by_center_indexes(doc_id, root_folder_id, list(affected_centers))

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
    mark_synced(doc_id)
    update_index(doc_id, {"utterance_hashes": new_hashes})

    speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
    update_index(doc_id, {"speakers": speakers})
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
        mark_synced(doc_id)
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

    delete_old_versions(doc_id, version, root_folder_id)
    update_skipped_utterances(doc_id, skipped)
    update_index(doc_id, {"parent_id": parent_id, "parent_id_hash": parent_id_hash, "content_hash": content_hash, "version": version})
    mark_synced(doc_id)

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
