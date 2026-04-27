import logging

from core.gemini.embeddings import embed
from core.google_drive.docs_reader import read_google_doc
from core.google_drive.firestore import acquire_for_syncing, mark_error, mark_synced
from core.parsing.parser import parse_document
from core.parsing.processor import process_document as _process_doc
from core.qdrant.writer import delete_old_versions, upsert_facts, upsert_summaries, upsert_utterances
from services.sync.app.hashing import sha256_text
from services.sync.app.index_store import load_index, update_index

logger = logging.getLogger(__name__)


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
        content_hash = sha256_text(raw_text)

        existing = load_index(doc_id)
        if existing and existing.get("content_hash") == content_hash:
            mark_synced(doc_id)
            logger.info("Skipped unchanged: %s", doc_id)
            return "skipped"

        metadata, utterances = parse_document(raw_text)
        dialog_date = metadata.get("date", "")
        version = content_hash

        utterance_payloads, summaries, facts = _process_doc(
            utterances, doc_id, version, client_name, dialog_date, root_folder_id
        )

        summary_vectors = embed([s["text"] for s in summaries]) if summaries else []
        facts_vectors = embed([str(f["facts"]) for f in facts]) if facts else []

        upsert_utterances(utterance_payloads)
        upsert_summaries(summaries, summary_vectors)
        upsert_facts(facts, facts_vectors)

        delete_old_versions(doc_id, version, root_folder_id)

        update_index(doc_id, {"content_hash": content_hash, "version": version})
        mark_synced(doc_id)

        logger.info(
            "Processed: %s | utterances=%d summaries=%d facts=%d",
            doc_id, len(utterance_payloads), len(summaries), len(facts),
        )
        return "processed"

    except Exception as e:
        mark_error(doc_id, str(e))
        logger.exception("Failed: %s", doc_id)
        raise
