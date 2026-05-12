import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from core.gemini.embeddings import embed, make_client
from core.google_drive.docs_reader import read_google_doc
from core.google_drive.firestore import acquire_for_syncing, update_client_speakers, mark_error, mark_synced
from core.parsing.parser import parse_document
from core.parsing.processor import build_utterance_payloads, iter_windows
from core.qdrant.writer import delete_old_versions, upsert_facts, upsert_summaries, upsert_utterances
from services.vector_sync.app.hashing import sha256_text
from services.vector_sync.app.index_store import load_index, update_index

logger = logging.getLogger(__name__)

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def process_one(doc_id: str, client_name: str, root_folder_id: str) -> str:
    """
    Full processing cycle for one document.
    Returns: "processed" | "skipped" | "not_acquired"
    """
    if not acquire_for_syncing(doc_id):
        logger.info("Not acquired (status != imported): %s", doc_id)
        return "not_acquired"

    try:
        raw_text = read_google_doc(doc_id)
        content_hash = sha256_text(raw_text + client_name)

        existing = load_index(doc_id)
        if existing and existing.get("content_hash") == content_hash:
            mark_synced(doc_id)
            logger.info("Skipped unchanged: %s", doc_id)
            return "skipped"

        metadata, utterances = parse_document(raw_text)
        del raw_text
        dialog_date = metadata.get("date", "")
        provider = metadata.get("provider", "")
        source_file = metadata.get("source_file", "")
        version = content_hash

        update_index(doc_id, {
            "dialog_date": dialog_date,
            "provider": provider,
            "source_file": source_file,
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

        for summary, facts in iter_windows(
            utterances, doc_id, version, client_name, dialog_date, root_folder_id
        ):
            summary_vector = embed([summary["text"]], client=embed_client)[0]
            upsert_summaries([summary], [summary_vector])

            if facts:
                fact_vectors = embed([f["text"] for f in facts], client=embed_client)
                upsert_facts(facts, fact_vectors)

            summaries_count += 1
            facts_count += len(facts)

        delete_old_versions(doc_id, version, root_folder_id)
        update_index(doc_id, {"content_hash": content_hash, "version": version})
        mark_synced(doc_id)

        speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
        prev_speakers = set((existing or {}).get("speakers", []))
        new_speakers = [s for s in speakers if s not in prev_speakers]
        update_index(doc_id, {"speakers": speakers, "speakers_indexed": True})
        if new_speakers:
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
