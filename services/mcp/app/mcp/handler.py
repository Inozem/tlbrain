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
from core.retrieval.run import run_retrieval
from core.retrieval.transcripts import get_transcripts
from core.retrieval.clients import list_clients
from core.google_drive.drive_client import create_client_folder, move_file_to_folder
from core.google_drive.firestore import (
    COLLECTION_NAME,
    create_client,
    get_client_folder_id,
    get_sync_status,
    move_transcript_record,
    get_unassigned,
)
from core.qdrant.writer import delete_by_doc_id
from core.config import get_root_folder_id

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
                "version": "1.0.0",
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
                    "description": "Search through client conversation transcripts. Always translate the query to English before calling this tool — the index is stored in English and translation ensures the best retrieval quality.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query in English",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "client_name": {
                                "type": "string",
                                "description": "Client name filter",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_transcript",
                    "description": "Retrieve full conversation transcripts without semantic search. Use when you need the complete text of a specific dialog or the most recent dialogs for a client.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Specific document ID. If provided, all other params are ignored.",
                            },
                            "client_name": {
                                "type": "string",
                                "description": "Filter by client name",
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
                    "name": "list_clients",
                    "description": "List all clients in the knowledge base with their dialog count and last dialog date. Call this first to discover available clients before querying. If the response contains a 'suggestion' field, present it to the user.",
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
                    "description": "Move a transcript to a different client folder. Updates Google Drive, resets the record for reindexing, and removes old vectors from the search index.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to move",
                            },
                            "new_client_name": {
                                "type": "string",
                                "description": "Target client name (folder will be created if it doesn't exist)",
                            },
                        },
                        "required": ["doc_id", "new_client_name"],
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
                    "name": "create_client",
                    "description": "Create a new client: makes a folder in Google Drive and registers the client in the database. Returns an error if a client with this name already exists.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "client_name": {
                                "type": "string",
                                "description": "Client name (used as folder name in Drive)",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional description",
                            },
                        },
                        "required": ["client_name"],
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

    if tool_name == "list_clients":
        return _handle_list_clients(request)

    if tool_name == "import_all_transcripts":
        return _handle_sync_tldv_all(request, arguments)

    if tool_name == "move_transcript":
        return _handle_move_transcript(request, arguments)

    if tool_name == "sync_status":
        return _handle_sync_status(request)

    if tool_name == "create_client":
        return _handle_create_client(request, arguments)

    return build_jsonrpc_error(
        request_id=request.id,
        code=-32602,
        message="Invalid tool",
    )


def _handle_query(request: JSONRPCRequest, arguments: dict) -> dict:
    query = arguments.get("query", "")
    client_name = arguments.get("client_name") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None

    t0 = time.monotonic()
    try:
        segments, meta = run_retrieval(
            query=query,
            client_name=client_name,
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
            "client_name": client_name,
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
    client_name = arguments.get("client_name") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None
    limit = arguments.get("limit") or 1

    t0 = time.monotonic()
    try:
        segments, meta = get_transcripts(
            doc_id=doc_id,
            client_name=client_name,
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
            "client_name": client_name,
            "hits_total": meta.get("total_matches", 0),
            "docs_returned": docs_returned,
            "segments_returned": meta.get("returned_segments", 0),
            "truncated": meta.get("truncated", False),
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(TLBrainPayload(segments=segments, meta=meta).model_dump(exclude_none=True))
    return build_jsonrpc_result(request.id, content)


def _handle_list_clients(request: JSONRPCRequest) -> dict:
    t0 = time.monotonic()
    try:
        clients = list_clients()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to list clients",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: list_clients",
        extra={
            "tool": "list_clients",
            "docs_returned": len(clients),
            "latency_ms": latency_ms,
        },
    )

    payload: dict = {"clients": clients}
    unassigned = next((c for c in clients if c["client_name"] == "_unassigned"), None)
    if unassigned and unassigned.get("dialog_count", 0) > 0:
        payload["suggestion"] = (
            f"{unassigned['dialog_count']} transcript(s) are unassigned. "
            f"Ask the user to review and assign them to a client. "
            f"Show each one using get_transcript(doc_id='...') and move it using move_transcript(doc_id='...', new_client_name='...')."
        )
    content = build_mcp_content(payload)
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
            f"While they are downloading, check two things via list_clients: "
            f"1. Transcripts in `_unassigned` — the system could not detect the client, assign them manually via move_transcript. "
            f"2. Transcripts that were assigned automatically — verify they went to the correct client. "
            f"The more accurately transcripts are assigned, the better the system will detect clients for future imports."
        )
    content = build_mcp_content(payload)
    return build_jsonrpc_result(request.id, content)


def _handle_move_transcript(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id", "").strip()
    new_client_name = arguments.get("new_client_name", "").strip()

    if not doc_id or not new_client_name:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="doc_id and new_client_name are required",
        )

    t0 = time.monotonic()
    try:
        from google.cloud import firestore as _fs
        db = _fs.Client()
        doc = db.collection(COLLECTION_NAME).document(doc_id).get()
        if not doc.exists:
            return build_jsonrpc_error(
                request_id=request.id,
                code=-32602,
                message=f"Transcript not found: {doc_id}",
            )
        root_folder_id = get_root_folder_id()

        new_folder_id = get_client_folder_id(new_client_name)
        if not new_folder_id:
            new_folder_id, _ = create_client_folder(new_client_name)
        move_file_to_folder(doc_id, new_folder_id)
        delete_by_doc_id(doc_id, root_folder_id)
        move_transcript_record(doc_id, new_client_name, new_folder_id)

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
        extra={
            "tool": "move_transcript",
            "doc_id": doc_id,
            "new_client_name": new_client_name,
            "latency_ms": latency_ms,
        },
    )

    payload: dict = {
        "status": "ok",
        "doc_id": doc_id,
        "new_client_name": new_client_name,
        "unassigned_remaining": unassigned["count"],
    }
    if unassigned["count"] > 0:
        payload["unassigned_transcripts"] = unassigned["transcripts"]
        payload["suggestion"] = (
            f"{unassigned['count']} transcript(s) are still unassigned. "
            f"Show each one using get_transcript(doc_id='...') and move it using move_transcript(doc_id='...', new_client_name='...')."
        )
    content = build_mcp_content(payload)
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
            f"Assigning transcripts to the correct client improves search accuracy and helps the system detect clients automatically in future imports. "
            f"{unassigned} transcript(s) are currently unassigned — ask the user to assign them via move_transcript."
        )

    content = build_mcp_content(status)
    return build_jsonrpc_result(request.id, content)


def _handle_create_client(request: JSONRPCRequest, arguments: dict) -> dict:
    client_name = arguments.get("client_name", "").strip()
    description = arguments.get("description") or None

    if not client_name:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="client_name is required",
        )

    t0 = time.monotonic()
    try:
        folder_id, folder_created = create_client_folder(client_name)
        registered = create_client(client_name, folder_id, description)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to create client",
            details=str(e),
        )

    if not registered:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=f"Client already exists: {client_name}",
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: create_client",
        extra={
            "tool": "create_client",
            "client_name": client_name,
            "folder_created": folder_created,
            "latency_ms": latency_ms,
        },
    )

    status = "ok" if folder_created else "registered_from_drive"
    content = build_mcp_content({"status": status, "client_name": client_name})
    return build_jsonrpc_result(request.id, content)
