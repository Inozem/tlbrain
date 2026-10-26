"""Isolated Telegram search sessions backed by canonical transcript retrieval."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from core.operations import read
from core.integrations.firestore.transcripts import get_transcript_record
from services.connectors.telegram.repository import TelegramRepository


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _result_doc_ids(segments: Any) -> list[str]:
    """Return unique document IDs in retrieval order for targeted title reads."""
    if not isinstance(segments, (list, tuple)):
        return []
    doc_ids: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        doc_id = segment.get("doc_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def _transcript_titles(doc_ids: list[str]) -> dict[str, str]:
    """Read titles only for Docs returned by retrieval."""
    titles: dict[str, str] = {}
    for doc_id in doc_ids:
        try:
            record = get_transcript_record(doc_id)
        except Exception:
            continue
        title = record.get("source_file") if isinstance(record, Mapping) else None
        if isinstance(title, str) and title:
            titles[doc_id] = title
    return titles


def _cards(
    segments: Any,
    *,
    titles: Mapping[str, str],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(segments, (list, tuple)):
        return cards
    for result in segments:
        if not isinstance(result, Mapping):
            continue
        doc_id = result.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        title = result.get("title") or result.get("source_file") or titles.get(doc_id) or doc_id
        date = result.get("dialog_date") or result.get("date") or ""
        folder_path = result.get("folder_path")
        card: dict[str, Any] = {
            "doc_id": doc_id,
            "title": title if isinstance(title, str) else doc_id,
            "date": date if isinstance(date, str) else "",
        }
        if isinstance(folder_path, str):
            card["folder_path"] = folder_path
        elif isinstance(folder_path, (list, tuple)) and all(
            isinstance(part, str) for part in folder_path
        ):
            card["folder_path"] = list(folder_path)
        cards.append(card)
    return cards


def _stored_outcome(repository: Any, context_id: str, conversation_id: str) -> dict[str, Any] | None:
    turns = repository.list_turns(context_id, conversation_id)
    for turn in turns:
        if turn.get("role") != "tool" or not isinstance(turn.get("parts"), list):
            continue
        for part in turn["parts"]:
            if isinstance(part, Mapping) and part.get("type") == "search_result":
                return {
                    "status": part.get("status"),
                    "kind": "search",
                    "conversation_id": conversation_id,
                    "results": part.get("results", []),
                    "metadata": part.get("metadata"),
                    "error": part.get("error"),
                }
    return None


def _prepare_search_session(
    repository: Any,
    *,
    context_id: str,
    conversation_id: str,
    filters: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any] | None:
    """Return a persisted replay, or create the search conversation."""
    existing = repository.get_conversation(context_id, conversation_id)
    if existing is not None:
        if existing.kind.value != "search" or existing.search_filters != filters:
            raise ValueError("operation ID is already bound to a different search")
        return _stored_outcome(repository, context_id, conversation_id)
    repository.ensure_conversation(
        context_id, conversation_id, kind="search", search_filters=filters, now=now,
    )
    return None


def _retrieve_search_results(
    retrieve: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
    title_lookup: Callable[[list[str]], Mapping[str, str]],
    *,
    filters: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Run canonical retrieval and construct safe cards, without raising details."""
    metadata: dict[str, Any] | None = None
    try:
        segments, retrieved_metadata = retrieve(**filters)
        if not isinstance(retrieved_metadata, dict):
            raise TypeError("retrieval metadata must be a mapping")
        metadata = retrieved_metadata
        doc_ids = _result_doc_ids(segments)
        try:
            titles = title_lookup(doc_ids) if doc_ids else {}
        except Exception:
            titles = {}
        cards = _cards(segments, titles=titles if isinstance(titles, Mapping) else {})
    except Exception:
        return "error", [], metadata, "Search unavailable."
    return ("ok" if cards else "no_results"), cards, metadata, None


def _persist_search_result(
    repository: Any,
    *,
    context_id: str,
    conversation_id: str,
    operation_id: str,
    cards: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    error: str | None,
    status: str,
    now: datetime,
) -> None:
    """Persist one immutable result turn with stable references and turn ID."""
    result_part = {
        "type": "search_result",
        "status": status,
        "results": cards,
        "metadata": metadata,
        "error": error,
    }
    repository.append_turn(
        context_id, conversation_id,
        turn_id=_stable_id("search-result", context_id, operation_id),
        operation_id=operation_id, role="tool", parts=[result_part],
        source_refs=[{"doc_id": card["doc_id"]} for card in cards], now=now,
    )


def run_search(
    *,
    context_id: str,
    operation_id: str,
    query: str,
    repository: Any | None = None,
    now: datetime | None = None,
    keywords: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    retrieve: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    title_lookup: Callable[[list[str]], Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Run canonical retrieval and persist a replay-safe, separate search session."""
    repository = repository or TelegramRepository()
    now = now or datetime.now(timezone.utc)
    retrieve = retrieve or read.query
    title_lookup = title_lookup or _transcript_titles
    conversation_id = _stable_id("search", context_id, operation_id)
    filters = {
        "query": query,
        "keywords": keywords,
        "folder_path": folder_path,
        "date_from": date_from,
        "date_to": date_to,
    }
    replay = _prepare_search_session(
        repository, context_id=context_id, conversation_id=conversation_id,
        filters=filters, now=now,
    )
    if replay is not None:
        return replay

    repository.append_turn(
        context_id, conversation_id,
        turn_id=_stable_id("search-query", context_id, operation_id),
        operation_id=operation_id, role="user", parts=[{"text": query}], now=now,
    )
    result_status, cards, metadata, error = _retrieve_search_results(
        retrieve, title_lookup, filters=filters,
    )
    _persist_search_result(
        repository, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, cards=cards, metadata=metadata, error=error,
        status=result_status, now=now,
    )
    return {
        "status": result_status,
        "kind": "search",
        "conversation_id": conversation_id,
        "results": cards,
        "metadata": metadata,
        "error": error,
    }


def select_search_result(
    *,
    context_id: str,
    conversation_id: str,
    doc_id: str,
    repository: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Select only a persisted search result and bind its existing Doc."""
    repository = repository or TelegramRepository()
    now = now or datetime.now(timezone.utc)
    conversation = repository.get_conversation(context_id, conversation_id)
    if conversation is None or conversation.kind.value != "search":
        raise ValueError("search session does not exist")
    outcome = _stored_outcome(repository, context_id, conversation_id)
    rows = outcome.get("results") if isinstance(outcome, Mapping) else None
    matching_cards = [
        card for card in rows
        if isinstance(card, Mapping) and card.get("doc_id") == doc_id
    ] if isinstance(rows, (tuple, list)) else []
    if not matching_cards:
        raise ValueError("document is not a result in this search session")
    card = matching_cards[0]
    document_conversation_id = _stable_id("document", context_id, doc_id)
    number = repository.bind_document(
        context_id, doc_id, document_conversation_id,
        metadata={key: value for key, value in card.items() if key != "doc_id"}, now=now,
    )
    repository.ensure_conversation(
        context_id, document_conversation_id, kind="document", doc_id=doc_id,
        dialog_number=number, now=now,
    )
    return {
        "status": "ok",
        "doc_id": doc_id,
        "dialog_number": number,
        "conversation_id": document_conversation_id,
    }


__all__ = ["run_search", "select_search_result"]
