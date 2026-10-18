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
from core.indexing.user_facts import (
    AddFactDocumentNotFound,
    AddFactInputError,
    AddFactTranscriptNotSynced,
    add_user_fact,
)
from core.retrieval.run import run_retrieval
from core.retrieval.transcripts import get_transcripts
from core.integrations.google_drive import folder_service
from core.integrations.google_drive.folder_service import (
    expand_registered_subtree as expand_subtree,
    resolve_registered_folder_path as resolve_folder_path,
)
from core.integrations.firestore.transcripts import (
    get_sync_status,
    list_transcripts,
)
from core.documents import operations as document_operations

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


def _folder_domain_error(request: JSONRPCRequest, error: Exception, operation: str) -> dict:
    from core.integrations.google_drive.folder_service import (
        AmbiguousFolderPath,
        FolderNameConflict,
        FolderNotFound,
        FolderSubtreeMove,
    )

    if isinstance(error, FolderNotFound):
        message = (
            f"Parent folder not found: {'/'.join(error.path)}"
            if error.role == "parent"
            else f"Folder not found: {'/'.join(error.path)}"
        )
    elif isinstance(error, AmbiguousFolderPath):
        if error.role == "parent":
            message = f"Ambiguous parent path: {'/'.join(error.path)} matches {error.count} folders."
        else:
            message = f"Ambiguous path: {'/'.join(error.path)} matches {error.count} folders."
            if operation == "move":
                message += " Provide a more specific path."
    elif isinstance(error, FolderNameConflict):
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=f"Folder '{error.name}' already exists in this location.",
            details="Use list_folders to see the existing folder hierarchy.",
        )
    elif isinstance(error, FolderSubtreeMove):
        message = "Cannot move a folder into its own subtree."
    else:
        raise error
    return build_jsonrpc_error(request_id=request.id, code=-32602, message=message)


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
                    "name": "rename_folder",
                    "description": "Rename a folder in the knowledge base. Updates Google Drive and the folder index. Documents inside are not reindexed.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Path of the folder to rename, e.g. [\"Clients - Active\", \"Old Name\"]",
                            },
                            "new_name": {
                                "type": "string",
                                "description": "New name for the folder",
                            },
                        },
                        "required": ["folder_path", "new_name"],
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

    if tool_name == "rename_folder":
        return _handle_rename_folder(request, arguments)

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
        result = folder_service.list_folders()
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
        outcome = document_operations.move_transcript(doc_id, folder_path)
    except document_operations.FolderPathNotFound as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=str(e),
            details="Use list_folders to see available folders.",
        )
    except document_operations.AmbiguousFolderPathError as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=str(e),
            details="Use a more specific path.",
        )
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
        "unassigned_remaining": outcome.unassigned_count,
    }
    if outcome.unassigned_count > 0:
        payload["unassigned_transcripts"] = outcome.unassigned_transcripts
        payload["suggestion"] = (
            f"{outcome.unassigned_count} transcript(s) are still unassigned. "
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
        folder_id, new_parent_id = folder_service.move_folder(folder_path, new_parent_path)
    except folder_service.FolderOperationError as e:
        return _folder_domain_error(request, e, "move")
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
        folder_id = folder_service.create_folder(name, parent_path)
    except folder_service.FolderOperationError as e:
        return _folder_domain_error(request, e, "create")
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


def _handle_rename_folder(request: JSONRPCRequest, arguments: dict) -> dict:
    folder_path = arguments.get("folder_path") or []
    new_name = arguments.get("new_name", "").strip()

    if not folder_path or not new_name:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="folder_path and new_name are required",
        )

    t0 = time.monotonic()
    try:
        folder_id = folder_service.rename_folder(folder_path, new_name)
    except folder_service.FolderOperationError as e:
        return _folder_domain_error(request, e, "rename")
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to rename folder",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: rename_folder",
        extra={"tool": "rename_folder", "folder_path": folder_path, "new_name": new_name, "latency_ms": latency_ms},
    )

    content = build_mcp_content({"status": "ok", "folder_id": folder_id, "new_name": new_name})
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
    t0 = time.monotonic()
    try:
        doc_id = add_user_fact(arguments.get("doc_id", ""), arguments.get("text", ""))
    except AddFactInputError as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=str(e),
        )
    except AddFactDocumentNotFound as e:
        return build_jsonrpc_error(request_id=request.id, code=-32602, message=str(e))
    except AddFactTranscriptNotSynced as e:
        return build_jsonrpc_error(request_id=request.id, code=-32602, message=str(e))
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
        if not terminal_ids:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Folder not found: {'/'.join(folder_path)}",
                details="Use list_folders to see available folders.",
            )
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

    t0 = time.monotonic()
    try:
        document_operations.rename_transcript(doc_id, new_title)
    except document_operations.TranscriptNotFound as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=str(e),
        )
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
