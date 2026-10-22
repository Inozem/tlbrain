import logging
import json
import os
import time
import uuid
from datetime import datetime, timezone

from services.mcp.app.mcp.schemas import (
    ADD_DIALOG_INPUT_SCHEMA,
    JSONRPCRequest,
    JSONRPCResponse,
    TLBrainPayload,
    validate_add_dialog_schema,
)
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result,
)
from core.indexing.user_facts import (
    AddFactDocumentNotFound,
    AddFactInputError,
    AddFactTranscriptNotSynced,
)
from core.operations import read as read_operations
from core.operations import sync as sync_operations
from core.operations import write as write_operations
from core.integrations.google_drive import folder_service
from core.integrations.google_drive.folder_service import FolderNotFound
from core.documents import operations as document_operations
from core.dialogs import completion as dialog_completion
from core.dialogs import saving as dialog_saving
from core.dialogs.classification import (
    FolderSelectionRequest,
    classify_dialog_folder,
    FolderPathNotFoundError,
    AmbiguousFolderPathError,
)
from core.dialogs.models import normalize_dialog_input
from core.integrations.firestore import folders as folder_repository
from core.integrations.firestore import mcp_sessions
from core.integrations.firestore import save_intents as save_intent_repository
from core.integrations.firestore import outbox as outbox_repository
from core.integrations.firestore.client import get_client
from core.integrations.firestore.operations import (
    claim_operation,
    complete_operation,
    fail_operation,
    get_operation,
    release_operation,
)
from core.integrations.google_drive.client import build_drive_service_rw
from core.integrations.google_docs.client import build_docs_service
from core.operations.models import OperationError, OperationResult, OperationOrigin
from core.operations import outbox as operation_outbox
from core.operations.outbox import TaskTargetConfig
from core.integrations.firestore.outbox import OutboxTargetKind
from core.operations.registry import OperationValidationError, validate_arguments
from core.config import get_root_folder_id
from services.mcp.app.mcp.request_context import get_request_context

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
    # Pydantic's compatibility parsing is retained for legacy tools. add_dialog
    # needs the original JSON-RPC ID type for its durable typed replay key.
    raw_id = request_dict.get("id") if isinstance(request_dict, dict) else None
    params = request_dict.get("params") if isinstance(request_dict, dict) else None
    if (isinstance(request_dict, dict) and request_dict.get("method") == "tools/call"
            and isinstance(params, dict) and params.get("name") == "add_dialog"
            and type(raw_id) not in (str, int)):
        return {
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32602, "message": "add_dialog requires a string or integer request ID"},
        }
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
                {
                    "name": "add_dialog",
                    "description": "Save a new dialog transcript and queue it for indexing.",
                    "inputSchema": ADD_DIALOG_INPUT_SCHEMA,
                },
            ]
        },
    )


def handle_tools_call(request: JSONRPCRequest) -> dict:
    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "add_dialog":
        return _handle_add_dialog(request, arguments)

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


def _add_dialog_payload(operation_id: str, operation, provider: str, *, status: str) -> dict:
    payload = {
        "operation_id": operation_id,
        "doc_id": operation.doc_id,
        "folder_path": list(operation.selected_folder_path or ()),
        "provider": provider,
        "status": status,
    }
    if operation.error is not None:
        error = operation.error
        payload["error"] = {
            "operation_id": operation_id,
            "service": error.service or "dialog_save",
            "stage": error.stage or operation.stage or "save",
            "code": error.code,
            "category": error.category,
            "sanitized_message": error.sanitized_message or error.message,
            "retryable": error.retryable,
        }
    return payload


def _add_dialog_in_progress(request: JSONRPCRequest, operation_id: str, operation) -> dict:
    stage = (operation.stage if operation is not None else None) or "accepted"
    details = {
        "operation_id": operation_id,
        "stage": stage,
        "retry_same_request": True,
    }
    if operation is not None and operation.doc_id:
        details.update({
            "doc_id": operation.doc_id,
            "folder_path": list(operation.selected_folder_path or ()),
            "service": "google_docs" if stage == "fill" else "dialog_save",
            "category": "in_progress",
            "code": "save_stage_in_progress",
            "sanitized_message": "The save is still being reconciled.",
        })
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "error": {
            "code": -32010,
            "message": "Operation is still in progress",
            "data": details,
        },
    }


def _map_add_dialog_failure(request: JSONRPCRequest, operation_id: str, operation, provider: str) -> dict:
    """Render durable failed state with its sanitized diagnostics and known document."""
    if operation.doc_id:
        return build_jsonrpc_result(request.id, build_mcp_content(
            _add_dialog_payload(operation_id, operation, operation.provider or provider, status="error")
        ))
    error = operation.error
    return {
        "jsonrpc": "2.0", "id": request.id,
        "error": {
            "code": -32603, "message": "Dialog save failed",
            "data": {
                "operation_id": operation_id,
                "stage": error.stage if error else operation.stage or "save",
                "service": error.service if error else "dialog_save",
                "code": error.code if error else "save_failed",
                "category": error.category if error else "execution",
                "sanitized_message": error.sanitized_message if error else "Dialog save failed",
            },
        },
    }


def _handle_add_dialog(request: JSONRPCRequest, arguments: dict) -> dict:
    """Keep implementation failures for this new tool inside its safe error boundary."""
    try:
        return _process_add_dialog(request, arguments)
    except Exception:
        logger.error("add_dialog request failed outside a checkpointed stage")
        return build_jsonrpc_error(
            request_id=request.id, code=-32603,
            message="Dialog save could not be completed",
        )


def _prepare_add_dialog(request: JSONRPCRequest, arguments: dict):
    """Validate the request and reserve its typed, session-bound replay key."""
    context, error_response = _require_add_dialog_session(request)
    if error_response is not None:
        return None, error_response

    validated, error_response = _validate_add_dialog_arguments(request, arguments)
    if error_response is not None:
        return None, error_response

    client = get_client()
    now = datetime.now(timezone.utc)
    reservation, error_response = _lookup_add_dialog_replay(
        request, context, validated, client, now,
    )
    if error_response is not None:
        return None, error_response
    if reservation is not None:
        return _prepare_add_dialog_replay(context, validated, client, now, reservation), None

    root_folder_id = get_root_folder_id()
    folders = folder_repository.get_all_folders(client)
    error_response = _validate_add_dialog_folder_path(
        request, validated, folders, root_folder_id,
    )
    if error_response is not None:
        return None, error_response

    reservation, error_response = _reserve_add_dialog_request(
        request, context, validated, client, now,
    )
    if error_response is not None:
        return None, error_response
    return (
        context, validated, client, root_folder_id, folders,
        reservation.accepted_at, reservation,
    ), None


def _require_add_dialog_session(request: JSONRPCRequest):
    """Require an initialized caller-bound session before any add_dialog effects."""
    context = get_request_context()
    if (context is None or context.session_state != "active" or not context.session_id
            or context.session is None or context.identity is None
            or not getattr(context.identity, "caller_binding", None)):
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602,
            message="An initialized MCP session is required for add_dialog",
        )
    return context, None


def _validate_add_dialog_arguments(request: JSONRPCRequest, arguments: dict):
    """Validate and normalize add_dialog input before session replay lookup."""
    try:
        validate_add_dialog_schema(arguments)
        validated = validate_arguments("add_dialog", arguments)
        normalize_dialog_input(validated, accepted_at=datetime.now(timezone.utc))
    except (OperationValidationError, TypeError, ValueError):
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602, message="Invalid add_dialog arguments",
        )
    return validated, None


def _lookup_add_dialog_replay(request, context, validated, client, now):
    """Look up the typed request key before checking mutable folder state."""
    try:
        reservation = mcp_sessions.lookup_request_replay(
            client, context.session_id,
            caller_binding=context.identity.caller_binding,
            request_id=request.id, arguments=validated, now=now,
        )
    except mcp_sessions.RequestArgumentsConflictError:
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602,
            message="This request ID was already used with different arguments",
        )
    except (mcp_sessions.SessionUnavailableError, ValueError):
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602,
            message="The initialized MCP session is unavailable",
        )
    return reservation, None


def _prepare_add_dialog_replay(context, validated, client, now, reservation):
    """Keep an exact replay bound to its durable operation despite current folder changes."""
    root_folder_id = None
    folders = None
    try:
        root_folder_id = get_root_folder_id()
        folders = folder_repository.get_all_folders(client)
    except Exception:
        pass
    return context, validated, client, root_folder_id, folders, now, reservation


def _validate_add_dialog_folder_path(request, validated, folders, root_folder_id):
    """Reject a fresh request whose selected folder path is no longer unambiguous."""
    if "folder_path" in validated:
        try:
            classify_dialog_folder(
                FolderSelectionRequest(
                    title=validated.get("title", ""), utterances=(),
                    folder_path=tuple(validated["folder_path"]),
                ), folders, root_folder_id,
                speaker_index_lookup=lambda _speaker: (), suggestion=lambda *_args: None,
            )
        except (FolderPathNotFoundError, AmbiguousFolderPathError):
            return build_jsonrpc_error(
                request_id=request.id, code=-32602,
                message="folder_path does not identify exactly one folder under the configured root",
            )
    return None


def _reserve_add_dialog_request(request, context, validated, client, now):
    """Reserve a new caller-bound request after its mutable folder path is valid."""
    try:
        reservation = mcp_sessions.resolve_or_reserve_request(
            client, context.session_id,
            caller_binding=context.identity.caller_binding,
            request_id=request.id, arguments=validated, now=now,
        )
    except mcp_sessions.RequestArgumentsConflictError:
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602,
            message="This request ID was already used with different arguments",
        )
    except (mcp_sessions.SessionUnavailableError, ValueError):
        return None, build_jsonrpc_error(
            request_id=request.id, code=-32602,
            message="The initialized MCP session is unavailable",
        )
    return reservation, None


def _build_add_dialog_folder_callbacks(client, folders):
    """Build folder classification callbacks using the current caller's index."""
    def speaker_index_lookup(speaker: str):
        return [snapshot.id for snapshot in folder_repository.get_folders_by_speaker(speaker, client)]

    def suggestion(stage, folder_ids, title, utterances, topic_hint):
        from core.integrations.gemini.llm import call_gemini_json
        names = [folders.get(folder_id, {}).get("name", folder_id) for folder_id in folder_ids]
        sample = "\n".join(f"{item.speaker}: {item.text}" for item in utterances)
        prompt = (
            "Select the best matching existing folder for this dialog. "
            "Return a folder_name from the candidate list and confidence from 0 to 1. "
            f"Stage: {stage}\nCandidates: {names!r}\nTitle: {title}\n"
            f"Topic: {topic_hint or ''}\nDialog:\n{sample}"
        )
        result = call_gemini_json(prompt)
        return result.get("folder_name"), result.get("confidence", 0)
    return speaker_index_lookup, suggestion


def _advance_add_dialog_save(client, operation_id, owner, lease, now,
                             root_folder_id, folders):
    """Advance durable dialog save stages and checkpoint provider rejections."""
    drive = build_drive_service_rw()
    docs = build_docs_service(["https://www.googleapis.com/auth/documents"])
    speaker_index_lookup, suggestion = _build_add_dialog_folder_callbacks(client, folders)
    try:
        progress = dialog_saving.advance_save(
            client, operation_id, owner=owner, fence=lease.fence, now=now,
            folders=folders, root_folder_id=root_folder_id,
            speaker_index_lookup=speaker_index_lookup, suggestion=suggestion,
            drive=drive, docs=docs,
            ensure_unassigned=lambda root: folder_service.get_unassigned_folder_id(drive, root),
        )
    except Exception as save_error:
        _record_add_dialog_provider_rejection(
            client, operation_id, owner, lease, save_error,
        )
        raise

    return progress


def _record_add_dialog_provider_rejection(client, operation_id, owner, lease, save_error):
    """Persist sanitized terminal state when a provider definitively rejects a save."""
    status = getattr(save_error, "status_code", None)
    if status is None:
        status = getattr(getattr(save_error, "resp", None), "status", None)
    rejected_create = (
        type(save_error).__name__ == "DriveCreateError"
        and getattr(save_error, "outcome", None) == "rejected"
    )
    if rejected_create or (isinstance(status, int) and 400 <= status < 500 and status not in {408, 429}):
        failed = get_operation(client, operation_id)
        fail_operation(
            client, operation_id, owner=owner, fence=lease.fence,
            error=OperationError(
                code="provider_rejected", category="provider", message="Dialog save was rejected",
                sanitized_message="The provider rejected the save request.", retryable=False,
                service=("google_docs" if failed and failed.stage == "fill" else "google_drive"),
                stage=failed.stage if failed else "save",
            ),
            now=datetime.now(timezone.utc),
        )


def _publish_add_dialog_index_work(client):
    """Publish due index outbox records using the configured task target."""
    target = TaskTargetConfig(
        queue_name=os.environ["VECTOR_SYNC_QUEUE"], url=os.environ["VECTOR_SYNC_URL"],
        service_account_email=(os.environ.get("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL")
                               or os.environ.get("TASK_SERVICE_ACCOUNT_EMAIL")),
        audience=(os.environ.get("CLOUD_TASKS_OIDC_AUDIENCE")
                  or os.environ.get("TASK_OIDC_AUDIENCE")),
    )
    operation_outbox.publish_due_outbox_records(
        client, {OutboxTargetKind.INDEX_DISPATCH: target},
        now=datetime.now(timezone.utc), limit=10,
    )


def _finalize_add_dialog_save(request, context, validated, client, operation_id,
                              owner, lease, now, progress):
    """Finalize the save certificate and report this operation's index outbox state."""
    operation = get_operation(client, operation_id)
    if operation is None:
        return None, None, None, None

    binding_material = json.dumps(
        ["string", context.session_id, type(request.id).__name__, request.id],
        separators=(",", ":"), ensure_ascii=False,
    )
    import hashlib
    origin_binding = {
        "context_id": hashlib.sha256(context.session_id.encode("utf-8")).hexdigest(),
        "draft_id": hashlib.sha256(binding_material.encode("utf-8")).hexdigest(),
        "manifest_digest": hashlib.sha256(json.dumps(
            validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
    }
    try:
        completed = dialog_completion.finalize_save(
            client, operation_id=operation_id, save_intent_id=progress.save_intent_id,
            owner=owner, fence=lease.fence, origin_binding=origin_binding, now=now,
        )
    except Exception:
        completed = None
    intent = save_intent_repository.get_save_intent(client, progress.save_intent_id)
    own_index_outbox = (
        outbox_repository.get_outbox_record(client, intent.index_checkpoint_ref)
        if intent is not None and intent.index_checkpoint_ref else None
    )
    dispatch_undelivered = own_index_outbox is None or own_index_outbox.state.value != "delivered"
    dispatch_failed = own_index_outbox is not None and own_index_outbox.state.value in {"retry", "failed"}
    return dispatch_failed, dispatch_undelivered, completed, get_operation(client, operation_id)


def _advance_publish_add_dialog(request, context, validated, client, root_folder_id,
                                folders, operation_id, owner, lease, now):
    """Advance save checkpoints, dispatch indexing work, then finalize its certificate."""
    progress = _advance_add_dialog_save(
        client, operation_id, owner, lease, now, root_folder_id, folders,
    )
    _publish_add_dialog_index_work(client)
    return _finalize_add_dialog_save(
        request, context, validated, client, operation_id, owner, lease,
        now, progress,
    )


def _accept_add_dialog_operation(request, context, validated, client, reservation):
    """Create the validated save once and preserve its operation on uncertain acceptance."""
    operation_id = reservation.operation_id
    operation = get_operation(client, operation_id)
    if operation is not None:
        return operation, None

    try:
        dialog_saving.accept_save(
            client, validated, operation_id=operation_id,
            caller_binding=context.identity.caller_binding, origin=OperationOrigin.MCP,
            now=reservation.accepted_at,
        )
    except Exception:
        operation = get_operation(client, operation_id)
        return operation, _add_dialog_in_progress(request, operation_id, operation)
    return get_operation(client, operation_id), None


def _map_add_dialog_terminal_state(request, operation_id, operation, provider):
    """Render a persisted failed or completed operation before attempting a lease."""
    if operation is None:
        return _add_dialog_in_progress(request, operation_id, None)
    if operation.state.value == "failed":
        return _map_add_dialog_failure(request, operation_id, operation, provider)
    if operation.state.value == "completed":
        return build_jsonrpc_result(request.id, build_mcp_content(
            _add_dialog_payload(operation_id, operation, operation.provider or provider, status="synced")
        ))
    return None


def _claim_add_dialog_lease(client, operation_id, now):
    """Claim the save operation under a fresh owner token for this request attempt."""
    owner = "mcp-add-dialog-" + uuid.uuid4().hex
    lease = claim_operation(client, operation_id, owner=owner, now=now, lease_seconds=300)
    return owner, lease


def _map_add_dialog_advance_result(request, operation_id, provider, result):
    """Map persisted save, index-dispatch, and completion-certificate outcomes."""
    dispatch_failed, dispatch_undelivered, completed, operation = result
    if operation is None:
        return _add_dialog_in_progress(request, operation_id, None)
    if operation.state.value == "failed":
        if operation.doc_id:
            return _map_add_dialog_failure(request, operation_id, operation, provider)
        return _add_dialog_in_progress(request, operation_id, operation)
    if completed is not None:
        return build_jsonrpc_result(request.id, build_mcp_content(
            _add_dialog_payload(operation_id, operation, operation.provider or provider, status="synced")
        ))
    if not operation.doc_id:
        return _add_dialog_in_progress(request, operation_id, operation)
    if dispatch_failed:
        payload = _add_dialog_payload(
            operation_id, operation, operation.provider or provider, status="created",
        )
        payload["error"] = {
            "operation_id": operation_id, "service": "cloud_tasks", "stage": "index_dispatch",
            "code": "cloud_tasks_enqueue_failed", "category": "dispatch",
            "sanitized_message": "Document was created; indexing dispatch will be retried.",
            "retryable": True,
        }
        return build_jsonrpc_result(request.id, build_mcp_content(payload))
    status = "created" if dispatch_undelivered else "indexing"
    return build_jsonrpc_result(request.id, build_mcp_content(
        _add_dialog_payload(operation_id, operation, operation.provider or provider, status=status)
    ))


def _map_add_dialog_advance_exception(request, operation_id, provider, client):
    """Map a failed checkpoint safely, or report the durable operation as in progress."""
    operation = get_operation(client, operation_id)
    if operation is not None and operation.state.value == "failed":
        return _map_add_dialog_failure(request, operation_id, operation, provider)
    return _add_dialog_in_progress(request, operation_id, operation)


def _release_add_dialog_lease(client, operation_id, owner, lease):
    """Release a claimed lease without logging provider or user supplied details."""
    try:
        release_operation(
            client, operation_id, owner=owner, fence=lease.fence,
            now=datetime.now(timezone.utc),
        )
    except Exception:
        logger.error("add_dialog lease release failed; preserving the mapped operation result")


def _run_claimed_add_dialog(request, context, validated, client, root_folder_id,
                            folders, operation_id, owner, lease, now):
    """Advance a leased save and always release its lease after response mapping."""
    try:
        result = _advance_publish_add_dialog(
            request, context, validated, client, root_folder_id, folders,
            operation_id, owner, lease, now,
        )
        return _map_add_dialog_advance_result(
            request, operation_id, validated["provider"], result,
        )
    except Exception:
        return _map_add_dialog_advance_exception(
            request, operation_id, validated["provider"], client,
        )
    finally:
        _release_add_dialog_lease(client, operation_id, owner, lease)


def _process_add_dialog(request: JSONRPCRequest, arguments: dict) -> dict:
    """Orchestrate replay preparation, terminal mapping, and one leased save attempt."""
    prepared, error_response = _prepare_add_dialog(request, arguments)
    if error_response is not None:
        return error_response
    context, validated, client, root_folder_id, folders, now, reservation = prepared
    operation_id = reservation.operation_id
    operation, error_response = _accept_add_dialog_operation(
        request, context, validated, client, reservation,
    )
    if error_response is not None:
        return error_response
    response = _map_add_dialog_terminal_state(
        request, operation_id, operation, validated["provider"],
    )
    if response is not None:
        return response

    owner, lease = _claim_add_dialog_lease(client, operation_id, now)
    if lease is None:
        return _add_dialog_in_progress(request, operation_id, operation)
    return _run_claimed_add_dialog(
        request, context, validated, client, root_folder_id, folders,
        operation_id, owner, lease, now,
    )


def _handle_query(request: JSONRPCRequest, arguments: dict) -> dict:
    query = arguments.get("query", "")
    keywords = arguments.get("keywords") or None
    folder_path = arguments.get("folder_path") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None

    t0 = time.monotonic()
    try:
        segments, meta = read_operations.query(
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
        segments, meta = read_operations.get_transcript(
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
        result = read_operations.list_folders()
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
        outcome = write_operations.move_transcript(doc_id, folder_path)
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
        folder_id, new_parent_id = write_operations.move_folder(folder_path, new_parent_path)
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
        folder_id = write_operations.create_folder(name, parent_path)
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
        folder_id = write_operations.rename_folder(folder_path, new_name)
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
    t0 = time.monotonic()
    try:
        result = sync_operations.sync_changes(arguments.get("doc_id"))
    except sync_operations.SyncConfigurationError:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="SYNC_CHECKER_URL is not configured",
        )
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
        status = read_operations.sync_status()
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

    status = dict(status)
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
        doc_id = write_operations.add_fact(arguments.get("doc_id", ""), arguments.get("text", ""))
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

    t0 = time.monotonic()
    try:
        result = read_operations.list_recent_transcripts(
            folder_path=folder_path,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
    except FolderNotFound as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message=f"Folder not found: {'/'.join(e.path)}",
            details="Use list_folders to see available folders.",
        )
    except read_operations.TranscriptListingError as e:
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
        write_operations.rename_transcript(doc_id, new_title)
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
