"""Canonical Telegram repository port required by the session flow."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping

from google.cloud import firestore
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.integrations.firestore.transcripts import get_transcript_record
from services.connectors.telegram.repository import SESSIONS, TelegramRepository


class TelegramRuntimeStore:
    """Expose repository and transcript metadata using session-flow methods."""

    def __init__(self, repository: TelegramRepository, *, root_folder_id: str | None = None,
                 context_id: str | None = None, operation_id: str | None = None):
        self.repository = repository
        self.client = repository.client
        self.root_folder_id = root_folder_id
        self.context_id = context_id
        self.operation_id = operation_id
        self.pending_search: tuple[str, Any, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.repository, name)

    def _dialog_rows(self, context_id: str) -> list[dict[str, Any]]:
        ref = self.client.collection(SESSIONS).document(context_id).collection("dialogs")
        return [snapshot.to_dict() or {} for snapshot in ref.stream()]

    def _metadata(self, doc_id: str) -> dict[str, Any] | None:
        try:
            record = get_transcript_record(doc_id, db=self.client)
        except TypeError:
            record = get_transcript_record(doc_id)
        except Exception:
            return None
        if not isinstance(record, Mapping):
            return None
        allowed = ("title", "source_file", "date", "dialog_date", "created_at",
                   "folder", "folder_id", "folder_path", "transcript_id", "parent_id")
        return {key: record[key] for key in allowed if key in record}

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        if not isinstance(doc_id, str) or not doc_id:
            return None
        context_ids = ([self.context_id] if self.context_id else
                       [context.id for context in self.client.collection(SESSIONS).stream()])
        for context_id in context_ids:
            for row in self._dialog_rows(context_id):
                if row.get("doc_id") == doc_id:
                    meta = self._metadata(doc_id) or {}
                    return {**meta, "doc_id": doc_id,
                            "conversation_id": row.get("conversation_id"),
                            "dialog_number": row.get("dialog_number")}
        metadata = self._metadata(doc_id)
        if metadata is None:
            return None
        conversation_id = "document-" + hashlib.sha256(
            f"{self.context_id or ''}\0{doc_id}".encode("utf-8")
        ).hexdigest()
        return {**metadata, "doc_id": doc_id, "conversation_id": conversation_id}

    def existing_dialog_candidates(self) -> tuple[dict[str, Any], ...]:
        # Enumerate canonical transcript index metadata. Listing never allocates
        # a dialog number or reads transcript/history content.
        rows: list[dict[str, Any]] = []
        from core.integrations.firestore.client import COLLECTION_NAME
        from core.integrations.firestore.folders import FOLDERS_COLLECTION
        folder_records = {snapshot.id: snapshot.to_dict() or {}
                          for snapshot in self.client.collection(FOLDERS_COLLECTION).stream()}
        scoped_folder_ids = {self.root_folder_id} if self.root_folder_id else set()
        changed = True
        while changed:
            changed = False
            for folder_id, folder in folder_records.items():
                if folder.get("parent_id") in scoped_folder_ids and folder_id not in scoped_folder_ids:
                    scoped_folder_ids.add(folder_id)
                    changed = True
        index = self.client.collection(COLLECTION_NAME)
        for snapshot in index.stream():
            metadata = snapshot.to_dict() or {}
            if not self.root_folder_id or metadata.get("parent_id") not in scoped_folder_ids:
                continue
            doc_id = snapshot.id
            folder_names: list[str] = []
            parent_id = metadata.get("parent_id")
            seen: set[str] = set()
            while parent_id in folder_records and parent_id not in seen:
                seen.add(parent_id)
                folder = folder_records[parent_id]
                name = folder.get("name")
                if isinstance(name, str):
                    folder_names.append(name)
                parent_id = folder.get("parent_id")
            folder_names.reverse()
            if self.root_folder_id in scoped_folder_ids and folder_names:
                root_name = folder_records.get(self.root_folder_id, {}).get("name")
                if isinstance(root_name, str) and folder_names[0] == root_name:
                    folder_names = folder_names[1:]
            conversation_id = "document-" + hashlib.sha256(
                f"{self.context_id or ''}\0{doc_id}".encode("utf-8")
            ).hexdigest()
            rows.append({**{key: metadata[key] for key in (
                "source_file", "date", "dialog_date", "created_at", "parent_id",
            ) if key in metadata}, "doc_id": doc_id,
                "title": metadata.get("source_file", doc_id),
                "folder_id": metadata.get("parent_id"),
                "folder_path": folder_names,
                "conversation_id": conversation_id})
        # Previously bound dialogs provide the canonical stable number and
        # conversation ID, overriding only the binding metadata.
        context_ids = ([self.context_id] if self.context_id else
                       [context.id for context in self.client.collection(SESSIONS).stream()])
        for context_id in context_ids:
            for dialog in self._dialog_rows(context_id):
                doc_id = dialog.get("doc_id")
                if not isinstance(doc_id, str) or not doc_id:
                    continue
                metadata = self._metadata(doc_id)
                if metadata is None:
                    continue
                folder_id = metadata.get("folder_id", metadata.get("parent_id"))
                if self.root_folder_id and folder_id not in scoped_folder_ids:
                    continue
                candidate = next((old for old in rows if old.get("doc_id") == doc_id), None)
                if candidate is None:
                    candidate = {**metadata, "doc_id": doc_id,
                                 "title": metadata.get("title", metadata.get("source_file", doc_id)),
                                 "conversation_id": dialog.get("conversation_id"),
                                 "dialog_number": dialog.get("dialog_number")}
                    rows.append(candidate)
                else:
                    candidate["conversation_id"] = dialog.get("conversation_id")
                    candidate["dialog_number"] = dialog.get("dialog_number")
        return tuple(rows)

    def resolve_dialog_number(self, context_id: str, number: int) -> str | None:
        snapshot = (self.client.collection(SESSIONS).document(context_id)
                    .collection("dialogs").document(str(number)).get())
        row = snapshot.to_dict() or {} if snapshot.exists else {}
        doc_id = row.get("doc_id")
        return doc_id if isinstance(doc_id, str) else None

    def get_or_assign_dialog_number(self, context_id: str, doc_id: str) -> int:
        key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        ref = (self.client.collection(SESSIONS).document(context_id)
               .collection("document_bindings").document(key))
        row = ref.get().to_dict() or {}
        if row.get("doc_id") == doc_id and type(row.get("dialog_number")) is int:
            return row["dialog_number"]
        conversation_id = "document-" + hashlib.sha256(
            f"{context_id}\0{doc_id}".encode("utf-8")
        ).hexdigest()
        metadata = self._metadata(doc_id) or {}
        now = datetime.now(timezone.utc)
        number = self.repository.bind_document(
            context_id, doc_id, conversation_id, metadata=metadata, now=now,
            update_last_doc=False,
        )
        self.repository.ensure_conversation(
            context_id, conversation_id, kind="document", doc_id=doc_id,
            dialog_number=number, now=now,
        )
        return number

    def create_search_session(self, *, context_id: str, query: str,
                              result_refs: list[str], error: str | None,
                              search_filters: Mapping[str, Any] | None = None) -> str:
        from services.connectors.telegram.search_sessions import run_search
        operation_id = self.operation_id or ("search-" + hashlib.sha256(
            f"{context_id}\0{query}".encode("utf-8")).hexdigest())
        pending = self.pending_search
        retrieve = None
        if (pending and isinstance(pending[0], Mapping)
                and pending[0].get("query") == query
                and (search_filters is None or dict(pending[0]) == dict(search_filters))):
            segments, metadata = pending[1], pending[2]
            retrieve = lambda **kwargs: (segments, metadata)
        filters = dict(search_filters or {})
        result = run_search(context_id=context_id, operation_id=operation_id,
                            query=query, repository=self.repository,
                            keywords=filters.get("keywords"),
                            folder_path=filters.get("folder_path"),
                            date_from=filters.get("date_from"),
                            date_to=filters.get("date_to"),
                            now=datetime.now(timezone.utc), retrieve=retrieve)
        self.pending_search = None
        return str(result.get("conversation_id", ""))

    def publish_search_selection(self, operation_id: str, *, conversation_id: str,
                                 owner: str, fence: int, now: datetime) -> bool:
        """Select this operation's persisted search session under its queue fence."""
        context_id = self.context_id
        if (not isinstance(context_id, str) or not isinstance(conversation_id, str)
                or not conversation_id):
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        conversation_ref = context_ref.collection("conversations").document(conversation_id)

        @firestore.transactional
        def commit(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            conversation_snapshot = conversation_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            conversation = (conversation_snapshot.to_dict() or {}
                            if conversation_snapshot.exists else {})
            route_control = operation.get("route_control")
            route_session = (route_control.get("search_session_id")
                             if isinstance(route_control, Mapping)
                             and route_control.get("kind") == "search" else
                             route_control.get("conversation_id")
                             if isinstance(route_control, Mapping)
                             and route_control.get("search_protocol") == "agent-v1" else None)
            if (not context_snapshot.exists or not operation_snapshot.exists
                    or operation.get("operation_id") != operation_id
                    or operation.get("context_id") not in (None, context_id)
                    or operation.get("state") != "running"
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence,
                        now=now, expiry=context.get("lock_expires_at"))):
                return False
            if (not isinstance(route_control, Mapping)
                    or route_session != conversation_id
                    or not conversation_snapshot.exists
                    or conversation.get("conversation_id") != conversation_id
                    or conversation.get("kind") != "search"):
                raise ValueError("search selection binding is invalid")
            if (context.get("active_conversation_id") == conversation_id
                    and context.get("selected_dialog_number") is None):
                return True
            transaction.update(context_ref, {
                "active_conversation_id": conversation_id,
                "selected_dialog_number": None,
                "selection_sequence": int(context.get("selection_sequence", 0)) + 1,
                "updated_at": now,
            })
            return True

        return commit(self.client.transaction(max_attempts=5))

    def publish_selection(self, operation_id: str, *, conversation_id: str,
                          dialog_number: int, owner: str, fence: int,
                          now: datetime) -> bool:
        context_id = self.context_id
        if not isinstance(context_id, str):
            return False
        context_ref = self.client.collection(SESSIONS).document(context_id)
        operation_ref = context_ref.collection("operations").document(operation_id)
        dialog_ref = context_ref.collection("dialogs").document(str(dialog_number))

        @firestore.transactional
        def commit(transaction: Any) -> bool:
            context_snapshot = context_ref.get(transaction=transaction)
            operation_snapshot = operation_ref.get(transaction=transaction)
            dialog_snapshot = dialog_ref.get(transaction=transaction)
            context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
            operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
            dialog = dialog_snapshot.to_dict() or {} if dialog_snapshot.exists else {}
            if (not context_snapshot.exists or not operation_snapshot.exists
                    or not dialog_snapshot.exists or operation.get("state") != "running"
                    or operation.get("operation_id") != operation_id
                    or dialog.get("conversation_id") != conversation_id
                    or type(dialog.get("dialog_number")) is not int
                    or dialog.get("dialog_number") != dialog_number
                    or not validate_context_queue_lease(
                        context, operation_id, owner=owner, fence=fence,
                        now=now, expiry=context.get("lock_expires_at"))):
                return False
            if context.get("active_conversation_id") == conversation_id:
                return True
            transaction.update(context_ref, {
                "active_conversation_id": conversation_id,
                "selected_dialog_number": dialog_number,
                "selection_sequence": int(context.get("selection_sequence", 0)) + 1,
                "updated_at": now,
            })
            return True

        return commit(self.client.transaction(max_attempts=5))
