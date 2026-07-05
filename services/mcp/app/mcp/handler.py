import logging
import os
import time

from services.mcp.app.mcp.schemas import (
    JSONRPCRequest,
    JSONRPCResponse,
    TLBrainPayload,
)
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result,
)
from core.config import get_root_folder_id
from core.gemini.embeddings import embed
from core.retrieval.run import run_retrieval
from core.retrieval.transcripts import get_transcripts
from core.retrieval.folders import list_folders
from core.google_drive.drive_client import create_folder, move_file_to_folder, move_folder_in_drive, rename_file
from core.google_drive.firestore import (
    expand_subtree,
    folder_name_exists,
    get_all_folders,
    get_folder_by_id,
    get_sync_status,
    get_transcript_record,
    get_unassigned,
    list_transcripts,
    move_transcript_record,
    resolve_folder_path,
    update_transcript_source_file,
    upsert_folder,
)
from core.qdrant.writer import set_payload_parent_id, upsert_user_facts

logger = logging.getLogger(__name__)


def build_jsonrpc_error(
    request_id,
    code: int,
    message: str,
    details: str | None = None,
) -> dict:
    error = {
        "code": code,
        "message": message,
    }

    if details is not None:
        error["details"] = details

    return JSONRPCResponse(
        id=request_id,
        error=error,
    ).model_dump(exclude_none=True)


async def handle_mcp_request(request_dict: dict) -> dict:
    try:
        request = JSONRPCRequest(**request_dict)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=None,
            code=-32600,
            message="Invalid Request",
            details=str(e),
        )

    method = request.method

    if method == "initialize":
        return handle_initialize(request)

    if method == "notifications/initialized":
        return {}

    if method == "tools/list":
        return handle_tools_list(request)

    if method == "tools/call":
        return handle_tools_call(request)

    return build_jsonrpc_error(
        request_id=request.id,
        code=-32601,
        message="Method not found",
    )


def handle_initialize(request: JSONRPCRequest) -> dict:
    return build_jsonrpc_result(
        request.id,
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "tlbrain",
                "version": os.environ.get("VERSION", "latest"),
            },
        },
    )


def handle_tools_list(request: JSONRPCRequest) -> dict:
    return build_jsonrpc_result(
        request.id,
        {
            "tools": [
                {
                    "name": "query",
                    "description": (
                        "Search through conversation transcripts using semantic + keyword search. "
                        "Translate 'query' to English — summaries and facts are indexed in English. "
                        "Use 'keywords' for names, brands, or terms that must match exactly as spoken."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query in English (used for semantic search over summaries and facts)",
                            },
                            "keywords": {
                                "type": "string",
                                "description": "Optional exact-match terms in the original language of the conversation (used for BM25 keyword search over utterances). Use for names, brands, or specific terms.",
                            },
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional folder path from root, e.g. [\"Clients - Active\", \"Acme Corp\"]. Searches the entire subtree.",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_transcript",
                    "description": "Retrieve full conversation transcripts without semantic search. Use when you need the complete text of a specific dialog or the most recent dialogs in a folder.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Specific document ID. If provided, all other params are ignored.",
                            },
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Folder path from root, e.g. [\"Acme Corp\"]. Returns transcripts from the entire subtree.",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max number of most recent transcripts to return (default: 1)",
                            },
                        },
                    },
                },
                {
                    "name": "list_folders",
                    "description": "List all folders in the knowledge base as a tree with dialog counts and last dialog dates. Call this first to discover the folder hierarchy before querying. If the response contains a 'suggestion' field, present it to the user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "import_all_transcripts",
                    "description": "Trigger a full import of all transcripts from connected providers. Only transcripts not yet in the database will be imported. Use for initial onboarding or after a long offline period. Import in small batches (default 10), then review and assign transcripts before importing the next batch.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Max number of transcripts to import in this batch (default: 10)",
                            },
                            "since": {
                                "type": "string",
                                "description": "ISO date to import from (e.g. 2025-01-01). If not set, imports all transcripts.",
                            },
                        },
                    },
                },
                {
                    "name": "move_transcript",
                    "description": "Move a transcript to a different folder. Updates Google Drive, resets the record for reindexing, and removes old vectors from the search index.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to move",
                            },
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target folder path from root, e.g. [\"Clients - Active\", \"Acme Corp\"]",
                            },
                        },
                        "required": ["doc_id", "folder_path"],
                    },
                },
                {
                    "name": "move_folder",
                    "description": "Move a folder to a different location in the hierarchy. Updates Google Drive and the folder index. Documents inside are not reindexed — their parent_id remains stable.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Path of the folder to move, e.g. [\"Clients - Active\", \"Acme Corp\"]",
                            },
                            "new_parent_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Path of the new parent folder, e.g. [\"Clients - Past\"]. Empty or omitted = move to root.",
                            },
                        },
                        "required": ["folder_path"],
                    },
                },
                {
                    "name": "create_folder",
                    "description": "Create a new folder in Google Drive and register it in the knowledge base.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Folder name",
                            },
                            "parent_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Parent folder path from root, e.g. [\"Clients - Active\"]. Empty or omitted = create at root level.",
                            },
                        },
                        "required": ["name"],
                    },
                },
                {
                    "name": "sync_changes",
                    "description": "Sync recent changes from Google Drive. Use doc_id only when a forced resync of a specific document is needed.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Resync a specific document by ID",
                            },
                        },
                    },
                },
                {
                    "name": "sync_status",
                    "description": "Show the current sync status: how many transcripts are in each stage (queued, downloading, imported, syncing, synced, error) and how many are unassigned. Use to diagnose stuck imports or check overall system health.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "add_fact",
                    "description": "Manually add a fact to a specific transcript. Use when semantic search missed an important detail — pins the document to relevant future queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to attach the fact to",
                            },
                            "text": {
                                "type": "string",
                                "description": "The fact to remember, in English",
                            },
                        },
                        "required": ["doc_id", "text"],
                    },
                },
                {
                    "name": "list_recent_transcripts",
                    "description": "List transcripts sorted by date descending. Use to find recent recordings without knowing the folder. Supports optional folder_path and date range filters, and pagination.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by folder path, e.g. [\"Clients - Active\", \"Acme Corp\"] (optional)",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, inclusive lower bound (optional)",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, inclusive upper bound (optional)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Page size (default: 20)",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Pagination offset (default: 0)",
                            },
                        },
                    },
                },
                {
                    "name": "rename_transcript",
                    "description": "Rename a transcript: updates the file name in Google Drive and the title visible in list_recent_transcripts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to rename",
                            },
                            "new_title": {
                                "type": "string",
                                "description": "New title for the transcript",
                            },
                        },
                        "required": ["doc_id", "new_title"],
                    },
                },
            ]
        },
    )


def handle_tools_call(request: JSONRPCRequest) -> dict:
    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "query":
        return _handle_query(request, arguments)

    if tool_name == "get_transcript":
        return _handle_get_transcript(request, arguments)

    if tool_name == "list_folders":
        return _handle_list_folders(request)

    if tool_name == "import_all_transcripts":
        return _handle_sync_tldv_all(request, arguments)

    if tool_name == "move_transcript":
        return _handle_move_transcript(request, arguments)

    if tool_name == "move_folder":
        return _handle_move_folder(request, arguments)

    if tool_name == "create_folder":
        return _handle_create_folder(request, arguments)

    if tool_name == "sync_changes":
        return _handle_sync_changes(request, arguments)

    if tool_name == "sync_status":
        return _handle_sync_status(request)

    if tool_name == "add_fact":
        return _handle_add_fact(request, arguments)

    if tool_name == "list_recent_transcripts":
        return _handle_list_recent_transcripts(request, arguments)

    if tool_name == "rename_transcript":
        return _handle_rename_transcript(request, arguments)

    return build_jsonrpc_error(
        request_id=request.id,
        code=-32602,
        message="Invalid tool",
    )


def _handle_query(request: JSONRPCRequest, arguments: dict) -> dict:
    query = arguments.get("query", "")
    keywords = arguments.get("keywords") or None
    folder_path = arguments.get("folder_path") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None

    t0 = time.monotonic()
    try:
        segments, meta = run_retrieval(
            query=query,
            keywords=keywords,
            folder_path=folder_path,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Retrieval failed",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    docs_returned = len({s["doc_id"] for s in segments})
    logger.info(
        "tool call: query",
        extra={
            "tool": "query",
            "query": query,
            "folder_path": folder_path,
            "hits_total": meta.get("total_matches", 0),
            "docs_returned": docs_returned,
            "segments_returned": meta.get("returned_segments", 0),
            "truncated": meta.get("truncated", False),
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(TLBrainPayload(segments=segments, meta=meta).model_dump(exclude_none=True))
    return build_jsonrpc_result(request.id, content)


def _handle_get_transcript(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id") or None
    folder_path = arguments.get("folder_path") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None
    limit = arguments.get("limit") or 1

    t0 = time.monotonic()
    try:
        segments, meta = get_transcripts(
            doc_id=doc_id,
            folder_path=folder_path,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Transcript retrieval failed",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    docs_returned = len({s["doc_id"] for s in segments})
    logger.info(
        "tool call: get_transcript",
        extra={
            "tool": "get_transcript",
            "folder_path": folder_path,
            "hits_total": meta.get("total_matches", 0),
            "docs_returned": docs_returned,
            "segments_returned": meta.get("returned_segments", 0),
            "truncated": meta.get("truncated", False),
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(TLBrainPayload(segments=segments, meta=meta).model_dump(exclude_none=True))
    return build_jsonrpc_result(request.id, content)


def _handle_list_folders(request: JSONRPCRequest) -> dict:
    t0 = time.monotonic()
    try:
        result = list_folders()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to list folders",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: list_folders",
        extra={"tool": "list_folders", "folders": len(result.get("folders", [])), "latency_ms": latency_ms},
    )

    content = build_mcp_content(result)
    return build_jsonrpc_result(request.id, content)




def _handle_sync_tldv_all(request: JSONRPCRequest, arguments: dict) -> dict:
    reconciliation_url = os.environ.get("TLDV_RECONCILIATION_URL", "")
    if not reconciliation_url:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="TLDV_RECONCILIATION_URL is not configured",
        )

    limit = arguments.get("limit") or 10
    since = arguments.get("since") or None
    t0 = time.monotonic()
    try:
        import httpx
        body = {"limit": limit, "full_scan": True}
        if since:
            body["since"] = since
        resp = httpx.post(reconciliation_url, json=body, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to trigger sync",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    queued = result.get("queued", 0)
    remaining = result.get("remaining", 0)
    logger.info(
        "tool call: import_all_transcripts",
        extra={"tool": "import_all_transcripts", "latency_ms": latency_ms, "queued": queued, "remaining": remaining},
    )

    payload: dict = {
        "status": "ok",
        "queued": queued,
    }

    if queued > 0:
        payload["suggestion"] = (
            f"Import started for {queued} transcript(s). "
            f"While they are downloading, check two things: "
            f"1. Call list_recent_transcripts(folder_path=[\"_unassigned\"]) — transcripts the system could not assign, move them manually via move_transcript. "
            f"2. Call list_folders to verify transcripts that were assigned automatically went to the correct folder. "
            f"The more accurately transcripts are assigned, the better the system will detect folders for future imports."
        )
    content = build_mcp_content(payload)
    return build_jsonrpc_result(request.id, content)


def _handle_move_transcript(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id", "").strip()
    folder_path = arguments.get("folder_path") or []

    if not doc_id or not folder_path:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="doc_id and folder_path are required",
        )

    t0 = time.monotonic()
    try:
        terminal_ids = resolve_folder_path(folder_path)
        if not terminal_ids:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Folder not found: {'/'.join(folder_path)}",
                details="Use list_folders to see available folders.",
            )
        if len(terminal_ids) > 1:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Ambiguous folder path: {'/'.join(folder_path)} matches {len(terminal_ids)} folders.",
                details="Use a more specific path.",
            )
        new_folder_id = terminal_ids[0]

        root_folder_id = get_root_folder_id()
        move_file_to_folder(doc_id, new_folder_id)
        move_transcript_record(doc_id, new_folder_id)
        set_payload_parent_id(doc_id, root_folder_id, new_folder_id)

        unassigned = get_unassigned()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to move transcript",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: move_transcript",
        extra={"tool": "move_transcript", "doc_id": doc_id, "folder_path": folder_path, "latency_ms": latency_ms},
    )

    payload: dict = {
        "status": "ok",
        "doc_id": doc_id,
        "folder_path": folder_path,
        "unassigned_remaining": unassigned["count"],
    }
    if unassigned["count"] > 0:
        payload["unassigned_transcripts"] = unassigned["transcripts"]
        payload["suggestion"] = (
            f"{unassigned['count']} transcript(s) are still unassigned. "
            f"Show each one using get_transcript(doc_id='...') and move it using move_transcript."
        )
    content = build_mcp_content(payload)
    return build_jsonrpc_result(request.id, content)


def _handle_move_folder(request: JSONRPCRequest, arguments: dict) -> dict:
    folder_path = arguments.get("folder_path") or []
    new_parent_path = arguments.get("new_parent_path") or []

    if not folder_path:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="folder_path is required",
        )

    t0 = time.monotonic()
    try:
        terminal_ids = resolve_folder_path(folder_path)
        if not terminal_ids:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Folder not found: {'/'.join(folder_path)}",
            )
        if len(terminal_ids) > 1:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Ambiguous path: {'/'.join(folder_path)} matches {len(terminal_ids)} folders. Provide a more specific path.",
            )
        folder_id = terminal_ids[0]

        if new_parent_path:
            parent_ids = resolve_folder_path(new_parent_path)
            if not parent_ids:
                return build_jsonrpc_error(
                    request_id=request.id,
                    code=-32602,
                    message=f"Parent folder not found: {'/'.join(new_parent_path)}",
                )
            if len(parent_ids) > 1:
                return build_jsonrpc_error(
                    request_id=request.id,
                    code=-32602,
                    message=f"Ambiguous parent path: {'/'.join(new_parent_path)} matches {len(parent_ids)} folders.",
                )
            new_parent_id = parent_ids[0]
        else:
            new_parent_id = get_root_folder_id()

        # Cycle check: new_parent must not be in the subtree of folder_id
        subtree = expand_subtree(folder_id)
        if new_parent_id in subtree:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message="Cannot move a folder into its own subtree.",
            )

        folder_data = get_folder_by_id(folder_id) or {}
        move_folder_in_drive(folder_id, new_parent_id)
        upsert_folder(folder_id, folder_data.get("name", ""), new_parent_id)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to move folder",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: move_folder",
        extra={"tool": "move_folder", "folder_path": folder_path, "new_parent_path": new_parent_path, "latency_ms": latency_ms},
    )

    content = build_mcp_content({"status": "ok", "folder_id": folder_id, "new_parent_id": new_parent_id})
    return build_jsonrpc_result(request.id, content)


def _handle_create_folder(request: JSONRPCRequest, arguments: dict) -> dict:
    name = arguments.get("name", "").strip()
    parent_path = arguments.get("parent_path") or []

    if not name:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="name is required",
        )

    t0 = time.monotonic()
    try:
        if parent_path:
            parent_ids = resolve_folder_path(parent_path)
            if not parent_ids:
                return build_jsonrpc_error(
                    request_id=request.id,
                    code=-32602,
                    message=f"Parent folder not found: {'/'.join(parent_path)}",
                )
            if len(parent_ids) > 1:
                return build_jsonrpc_error(
                    request_id=request.id,
                    code=-32602,
                    message=f"Ambiguous parent path: {'/'.join(parent_path)} matches {len(parent_ids)} folders.",
                )
            parent_id = parent_ids[0]
        else:
            parent_id = get_root_folder_id()

        if folder_name_exists(name, parent_id):
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Folder '{name}' already exists in this location.",
                details="Use list_folders to see the existing folder hierarchy.",
            )

        folder_id = create_folder(name, parent_id)
        upsert_folder(folder_id, name, parent_id)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to create folder",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: create_folder",
        extra={"tool": "create_folder", "folder_name": name, "parent_path": parent_path, "latency_ms": latency_ms},
    )

    content = build_mcp_content({"status": "ok", "folder_id": folder_id, "name": name})
    return build_jsonrpc_result(request.id, content)


def _handle_sync_changes(request: JSONRPCRequest, arguments: dict) -> dict:
    checker_url = os.environ.get("SYNC_CHECKER_URL", "")
    if not checker_url:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="SYNC_CHECKER_URL is not configured",
        )

    t0 = time.monotonic()
    try:
        import httpx
        body: dict = {}
        if arguments.get("doc_id"):
            body["doc_id"] = arguments["doc_id"].strip()
        resp = httpx.post(checker_url, json=body or None, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to trigger sync",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: sync_changes",
        extra={"tool": "sync_changes", "latency_ms": latency_ms, "queued": result.get("queued", 0)},
    )

    content = build_mcp_content({"status": "ok", **result})
    return build_jsonrpc_result(request.id, content)


def _handle_sync_status(request: JSONRPCRequest) -> dict:
    t0 = time.monotonic()
    try:
        status = get_sync_status()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to get sync status",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: sync_status",
        extra={"tool": "sync_status", "latency_ms": latency_ms, "total": status.get("total", 0)},
    )

    unassigned = status.pop("_unassigned_count", 0)
    if unassigned > 0:
        status["suggestion"] = (
            f"Assigning transcripts to the correct folder improves search accuracy. "
            f"{unassigned} transcript(s) are currently unassigned — call list_recent_transcripts(folder_path=[\"_unassigned\"]) to see them, "
            f"then move each one using move_transcript(doc_id='...', folder_path=[...])."
        )

    content = build_mcp_content(status)
    return build_jsonrpc_result(request.id, content)


def _handle_add_fact(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id", "").strip()
    text = arguments.get("text", "").strip()

    if not doc_id or not text:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="doc_id and text are required",
        )

    t0 = time.monotonic()
    try:
        record = get_transcript_record(doc_id)
        if not record:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Document not found: {doc_id}",
            )
        if record.get("status") != "synced":
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Document is not synced yet (status={record.get('status')}). Try again after sync completes.",
            )

        dialog_date = record.get("dialog_date", "")
        parent_id = record.get("parent_id", "")
        root_folder_id = get_root_folder_id()

        payload = {
            "type": "user_fact",
            "doc_id": doc_id,
            "text": text,
            "root_folder_id": root_folder_id,
            "parent_id": parent_id,
            "dialog_date": dialog_date,
            "dialog_date_num": int(dialog_date.replace("-", "")) if dialog_date else 0,
        }

        vector = embed([text])[0]
        upsert_user_facts([payload], [vector])
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to add fact",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: add_fact",
        extra={"tool": "add_fact", "doc_id": doc_id, "latency_ms": latency_ms},
    )

    content = build_mcp_content({"status": "ok", "doc_id": doc_id})
    return build_jsonrpc_result(request.id, content)


def _handle_list_recent_transcripts(request: JSONRPCRequest, arguments: dict) -> dict:
    folder_path = arguments.get("folder_path") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None
    limit = int(arguments.get("limit") or 20)
    offset = int(arguments.get("offset") or 0)

    folder_ids = None
    if folder_path:
        terminal_ids = resolve_folder_path(folder_path)
        if terminal_ids:
            folder_ids = [fid for tid in terminal_ids for fid in expand_subtree(tid)]

    t0 = time.monotonic()
    try:
        result = list_transcripts(folder_ids=folder_ids, date_from=date_from, date_to=date_to, limit=limit, offset=offset)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to list transcripts",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: list_recent_transcripts",
        extra={
            "tool": "list_recent_transcripts",
            "folder_path": folder_path,
            "total": result["total"],
            "returned": result["returned"],
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(result)
    return build_jsonrpc_result(request.id, content)


def _handle_rename_transcript(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id", "").strip()
    new_title = arguments.get("new_title", "").strip()

    if not doc_id or not new_title:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="doc_id and new_title are required",
        )

    record = get_transcript_record(doc_id)
    if not record:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=f"Document not found: {doc_id}",
        )

    t0 = time.monotonic()
    try:
        rename_file(doc_id, new_title)
        update_transcript_source_file(doc_id, new_title)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to rename transcript",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: rename_transcript",
        extra={"tool": "rename_transcript", "doc_id": doc_id, "new_title": new_title, "latency_ms": latency_ms},
    )

    content = build_mcp_content({"status": "ok", "doc_id": doc_id, "new_title": new_title})
    return build_jsonrpc_result(request.id, content)
