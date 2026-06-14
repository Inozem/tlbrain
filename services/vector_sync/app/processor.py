import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from core.gemini.embeddings import embed, make_client
from core.google_drive.docs_reader import read_google_doc
from core.google_drive.firestore import acquire_for_syncing, get_client_folder_id, update_client_speakers, mark_error, mark_synced, update_skipped_utterances
from core.parsing.parser import parse_document
from core.parsing.processor import build_utterance_payloads, iter_windows
from core.parsing.windowing import generate_windows
from core.qdrant.writer import (
    delete_old_versions,
    delete_summaries_by_center_indexes,
    delete_utterances_by_order_indexes,
    set_payload_client_name,
    upsert_facts,
    upsert_summaries,
    upsert_utterances,
)
from services.vector_sync.app.hashing import sha256_text, sha256_utterance
from services.vector_sync.app.index_store import load_index, update_index

logger = logging.getLogger(__name__)

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def process_one(doc_id: str, client_name: str, root_folder_id: str, folder_id: str | None = None, raw_text: str | None = None) -> str:
    """
    Full processing cycle for one document.
    Returns: "processed" | "skipped" | "not_acquired"
    """
    if not acquire_for_syncing(doc_id):
        logger.info("Not acquired (already syncing): %s", doc_id)
        return "not_acquired"

    try:
        existing = load_index(doc_id)
        existing_utterance_hashes = (existing or {}).get("utterance_hashes")
        stored_client_name = (existing or {}).get("client_name", "")
        stored_drive_folder = (existing or {}).get("drive_folder")

        client_name_changed = bool(client_name and client_name != stored_client_name)
        if client_name_changed:
            set_payload_client_name(doc_id, root_folder_id, client_name)
            stored_speakers = (existing or {}).get("speakers", [])
            # Decrement the previous client only if its record still exists: during a folder
            # rename the old record is deleted, and a blind decrement would resurrect a
            # phantom. Counts self-assemble from each doc's increment into the new client.
            if stored_speakers and stored_client_name and get_client_folder_id(stored_client_name):
                update_client_speakers(stored_client_name, stored_speakers, delta=-1)
            if stored_speakers and client_name != "_unassigned":
                update_client_speakers(client_name, stored_speakers)
            logger.info("Updated client_name in Qdrant: %s → %s (%s)", stored_client_name, client_name, doc_id)

        # Keep drive_folder fresh — manual Drive moves don't update it otherwise.
        if folder_id and folder_id != stored_drive_folder:
            update_index(doc_id, {"drive_folder": folder_id})

        if raw_text is None:
            raw_text = read_google_doc(doc_id)
        content_hash = sha256_text(raw_text + client_name)

        if existing_utterance_hashes:
            # --- Incremental path ---
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
                update_index(doc_id, {"client_name": client_name, "content_hash": content_hash, "version": version})
                mark_synced(doc_id)
                logger.info("Incremental skip (no changes): %s", doc_id)
                return "skipped"

            update_index(doc_id, {
                "dialog_date": dialog_date,
                "provider": provider,
            })

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
                changed_utterances, doc_id, version, client_name, dialog_date, root_folder_id
            )
            if utterance_payloads:
                bm25_embeddings = list(_get_bm25_model().embed([u["text"] for u in utterance_payloads]))
                sparse_vectors = [
                    SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                    for e in bm25_embeddings
                ]
                upsert_utterances(utterance_payloads, sparse_vectors)
                for u in utterance_payloads:
                    i_str = str(u["order_index"])
                    update_index(doc_id, {f"utterance_hashes.{i_str}": new_hashes[i_str]})

            embed_client = make_client()
            summaries_count = 0
            facts_count = 0
            skipped = []
            for summary, facts in iter_windows(
                utterances, doc_id, version, client_name, dialog_date, root_folder_id,
                allowed_center_indexes=affected_centers,
                skipped_utterances=skipped,
            ):
                summary_vector = embed([summary["text"]], client=embed_client)[0]
                upsert_summaries([summary], [summary_vector])
                if facts:
                    fact_vectors = embed([f["text"] for f in facts], client=embed_client)
                    upsert_facts(facts, fact_vectors)
                summaries_count += 1
                facts_count += len(facts)

            reanalyzed = {
                u["order_index"]
                for w in generate_windows(utterances)
                if w["center_index"] in affected_centers
                for u in w["utterances"]
            }
            existing_skipped = [i for i in ((existing or {}).get("skipped_utterances") or [])
                                if i not in reanalyzed]
            update_skipped_utterances(doc_id, existing_skipped + skipped)
            update_index(doc_id, {"client_name": client_name, "content_hash": content_hash, "version": version})
            mark_synced(doc_id)
            update_index(doc_id, {"utterance_hashes": new_hashes})

            speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
            prev_speakers = set((existing or {}).get("speakers", []))
            new_speakers = [s for s in speakers if s not in prev_speakers]
            update_index(doc_id, {"speakers": speakers, "speakers_indexed": True})
            if new_speakers and client_name != "_unassigned":
                update_client_speakers(client_name, new_speakers)

            logger.info(
                "Incremental sync: %s | changed=%d summaries=%d facts=%d",
                doc_id, len(changed_indexes), summaries_count, facts_count,
            )
            return "processed"

        else:
            # --- Full reindex path ---
            if existing and existing.get("content_hash") == content_hash:
                mark_synced(doc_id)
                logger.info("Skipped unchanged: %s", doc_id)
                return "skipped"

            metadata, utterances = parse_document(raw_text)
            del raw_text
            dialog_date = metadata.get("date", "")
            provider = metadata.get("provider", "")
            version = content_hash

            update_index(doc_id, {
                "dialog_date": dialog_date,
                "provider": provider,
            })

            utterance_payloads = build_utterance_payloads(
                utterances, doc_id, version, client_name, dialog_date, root_folder_id
            )
            bm25_embeddings = list(_get_bm25_model().embed([u["text"] for u in utterance_payloads]))
            sparse_vectors = [
                SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in bm25_embeddings
            ]
            upsert_utterances(utterance_payloads, sparse_vectors)

            embed_client = make_client()
            summaries_count = 0
            facts_count = 0
            skipped = []
            for summary, facts in iter_windows(
                utterances, doc_id, version, client_name, dialog_date, root_folder_id,
                skipped_utterances=skipped,
            ):
                summary_vector = embed([summary["text"]], client=embed_client)[0]
                upsert_summaries([summary], [summary_vector])
                if facts:
                    fact_vectors = embed([f["text"] for f in facts], client=embed_client)
                    upsert_facts(facts, fact_vectors)
                summaries_count += 1
                facts_count += len(facts)

            delete_old_versions(doc_id, version, root_folder_id)
            update_skipped_utterances(doc_id, skipped)
            update_index(doc_id, {"client_name": client_name, "content_hash": content_hash, "version": version})
            mark_synced(doc_id)

            utterance_hashes = {
                str(u["order_index"]): sha256_utterance(u["order_index"], u["speaker"], u["text"])
                for u in utterance_payloads
            }
            update_index(doc_id, {"utterance_hashes": utterance_hashes})

            speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
            prev_speakers = set((existing or {}).get("speakers", []))
            new_speakers = [s for s in speakers if s not in prev_speakers]
            update_index(doc_id, {"speakers": speakers, "speakers_indexed": True})
            if new_speakers and client_name != "_unassigned":
                update_client_speakers(client_name, new_speakers)

            logger.info(
                "Processed: %s | utterances=%d summaries=%d facts=%d",
                doc_id, len(utterance_payloads), summaries_count, facts_count,
            )
            return "processed"

    except Exception as e:
        mark_error(doc_id, str(e))
        logger.exception("Failed: %s", doc_id)
        raise
