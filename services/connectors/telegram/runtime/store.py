"""Canonical Telegram repository port required by the session flow."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit

from google.cloud import firestore
from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.integrations.firestore.transcripts import get_transcript_record
from services.connectors.telegram.repository import SESSIONS, TelegramRepository
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.context import build_conversation_history
from services.connectors.telegram.conversations import ConversationKind, build_conversation
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from services.connectors.telegram.runtime.routing_context import (
    BoundRef, CommandSpan, DocumentEvidence, ExplicitReferenceKind,
    RoutingContextSnapshot, SourceKind, json_digest,
)


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

    def _addressed_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Read one transcript record directly, propagating storage failures."""
        record = get_transcript_record(doc_id, db=self.client)
        if record is None:
            return None
        if not isinstance(record, Mapping):
            raise ValueError("addressed transcript metadata is malformed")
        if record.get("doc_id") is not None and record.get("doc_id") != doc_id:
            raise ValueError("addressed transcript metadata has a different document ID")
        allowed = ("title", "source_file", "date", "dialog_date", "created_at",
                   "folder", "folder_id", "folder_path", "transcript_id", "parent_id")
        return {key: self._json_metadata_value(record[key])
                for key in allowed if key in record}

    @classmethod
    def _json_metadata_value(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None or type(value) in (str, bool, int, float):
            return value
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("addressed metadata keys must be strings")
            return {key: cls._json_metadata_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_metadata_value(item) for item in value]
        raise TypeError("addressed metadata must contain JSON-safe values")

    @staticmethod
    def _accepted_input_digest(command: Mapping[str, Any], text: str) -> str:
        encoded = json.dumps(
            {"kind": command.get("kind"), "text": text,
             "action_metadata": command.get("action_metadata")},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _accepted_payload_projection(
        cls, command: Mapping[str, Any], text: str,
    ) -> dict[str, Any]:
        """Freeze accepted command hints as the same JSON-safe projection on capture/replay."""
        keys = (
            "draft_id", "cutoff_message_ids", "cutoff_intake_sequence",
            "predecessor_operation_id", "pending_save_candidate_operation_id",
            "action_metadata", "observed_clarification_id",
            "observed_clarification_generation", "observed_question_delivery_ref",
        )
        return {
            "text": text, "kind": command.get("kind", "freeform"),
            **{key: cls._json_metadata_value(command[key]) for key in keys if key in command},
        }

    def _dialog_binding(self, context_id: str, number: int) -> dict[str, Any] | None:
        if type(number) is not int or number < 1:
            raise ValueError("local dialog number is invalid")
        snapshot = (self.client.collection(SESSIONS).document(context_id)
                    .collection("dialogs").document(str(number)).get())
        if not snapshot.exists:
            return None
        row = snapshot.to_dict() or {}
        if (type(row.get("dialog_number")) is not int or row.get("dialog_number") != number
                or not isinstance(row.get("doc_id"), str) or not row.get("doc_id")
                or not isinstance(row.get("conversation_id"), str)
                or not row.get("conversation_id")):
            raise ValueError("local dialog binding is malformed")
        binding_key = hashlib.sha256(row["doc_id"].encode("utf-8")).hexdigest()
        binding = (self.client.collection(SESSIONS).document(context_id)
                   .collection("document_bindings").document(binding_key).get())
        if binding.exists:
            linked = binding.to_dict() or {}
            if (type(linked.get("dialog_number")) is not int
                    or linked.get("doc_id") != row["doc_id"]
                    or linked.get("dialog_number") != number
                    or linked.get("conversation_id") != row["conversation_id"]):
                raise ValueError("local dialog document binding is inconsistent")
        return row

    @staticmethod
    def _reference_value(span: CommandSpan) -> str | int | None:
        if span.kind is ExplicitReferenceKind.ID:
            if span.parsed_value != span.raw_text or not span.raw_text.strip():
                raise ValueError("explicit document ID does not match its accepted span")
            if any(char.isspace() for char in span.raw_text) or "/" in span.raw_text:
                raise ValueError("explicit document ID span is malformed")
            return span.parsed_value
        if span.kind is ExplicitReferenceKind.LOCAL_NUMBER:
            normalized = unicodedata.normalize("NFKC", span.raw_text)
            match = re.fullmatch(r"(?:no\.?\s*|#\s*)?(\d+)", normalized, flags=re.IGNORECASE)
            if match is None or int(match.group(1)) != span.parsed_value:
                raise ValueError("local dialog number does not match its accepted span")
            return span.parsed_value
        if span.kind is ExplicitReferenceKind.LAST_SAVED:
            if span.parsed_value != span.raw_text or not span.raw_text.strip():
                raise ValueError("last-saved selector does not match its accepted span")
            return None
        if span.parsed_value != span.raw_text:
            raise ValueError("explicit URL does not match its accepted span")
        parsed = urlsplit(span.raw_text)
        if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.port is not None):
            return None
        host = parsed.hostname.lower()
        match = None
        if host == "docs.google.com":
            match = re.fullmatch(r"/document/d/([A-Za-z0-9_-]+)(?:/.*)?", parsed.path)
        elif host == "drive.google.com":
            match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)(?:/.*)?", parsed.path)
        return match.group(1) if match else None

    def _routing_scope(self, actor_id: int, ledger: FirestoreAgentLedger):
        context_ref = self.client.collection(SESSIONS).document(self.context_id)
        operation_ref = context_ref.collection("operations").document(self.operation_id)
        operation_snapshot = operation_ref.get()
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        command = ledger._routing_command()
        input_ref = operation.get("input_ref")
        if (not operation_snapshot.exists or operation.get("operation_id") != self.operation_id
                or operation.get("context_id") not in (None, self.context_id)
                or not isinstance(input_ref, str)
                or command.get("user_id") != actor_id
                or command.get("context_id") != self.context_id
                or command.get("operation_id") != self.operation_id):
            raise PermissionError("accepted command scope is invalid")
        parts = input_ref.split("/")
        if (len(parts) != 4 or parts[:3] != [SESSIONS, self.context_id, "commands"]
                or not parts[3]):
            raise PermissionError("accepted command reference is invalid")
        return context_ref, operation_ref, operation, command, input_ref

    def _validate_captured_snapshot(
        self, snapshot: RoutingContextSnapshot, *, actor_id: int,
        input_ref: str, operation: Mapping[str, Any],
        command: Mapping[str, Any], conversation_id: str,
    ) -> None:
        accepted = snapshot.accepted_command_ref
        history = snapshot.full_history_ref
        if (snapshot.context_id != self.context_id or snapshot.actor_id != actor_id
                or snapshot.operation_id != self.operation_id
                or snapshot.selected_conversation_id != conversation_id):
            raise ValueError("routing snapshot scope is invalid")
        accepted_digest = hashlib.sha256(snapshot.accepted_command.encode("utf-8")).hexdigest()
        expected_payload = self._accepted_payload_projection(command, snapshot.accepted_command)
        if (accepted.ref != input_ref or accepted.digest != accepted_digest
                or accepted.context_id != self.context_id or accepted.actor_id != actor_id
                or accepted.conversation_id != conversation_id
                or accepted.operation_id != self.operation_id
                or accepted.to_dict()["payload"] != expected_payload
                or operation.get("identity_digest") != self._accepted_input_digest(
                    command, snapshot.accepted_command)):
            raise ValueError("captured accepted-command reference is inconsistent")
        expected_history_ref = (
            f"{SESSIONS}/{self.context_id}/conversations/{conversation_id}/turns"
        )
        if (history.ref != expected_history_ref or history.context_id != self.context_id
                or history.actor_id != actor_id or history.conversation_id != conversation_id
                or history.operation_id != self.operation_id
                or not isinstance(history.payload, Mapping)
                or set(history.payload) != {"turns", "conversation"}):
            raise ValueError("captured history reference is inconsistent")
        turns = history.payload.get("turns")
        if not isinstance(turns, (list, tuple)) or history.digest != json_digest(turns):
            raise ValueError("captured history payload digest is invalid")
        selected = history.payload.get("conversation")
        if not isinstance(selected, Mapping) or set(selected) != {
            "kind", "doc_id", "draft_id", "dialog_number", "search_filters",
        }:
            raise ValueError("captured selected conversation metadata is malformed")
        try:
            selected_kind = ConversationKind(selected.get("kind"))
            build_conversation(
                conversation_id, selected_kind, doc_id=selected.get("doc_id"),
                draft_id=selected.get("draft_id"), dialog_number=selected.get("dialog_number"),
                search_filters=selected.get("search_filters"), now=datetime.now(timezone.utc),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("captured selected conversation metadata is invalid") from exc
        if (snapshot.selected_conversation_kind != selected_kind.value
                or snapshot.selected_doc_id != selected.get("doc_id")):
            raise ValueError("captured selected conversation binding is inconsistent")
        projected = build_conversation_history(
            turns, conversation_id=conversation_id, search_exchange_loader=None,
        )
        if json_digest(projected) != history.digest:
            raise ValueError("captured history is not a complete projected conversation")

    def _load_routing_checkpoint(
        self, ledger: FirestoreAgentLedger, *, actor_id: int, input_ref: str,
        operation: Mapping[str, Any], command: Mapping[str, Any],
        operation_ref: Any,
    ) -> RoutingContextSnapshot | None:
        checkpoint = operation_ref.collection("routing_checkpoints").document("context").get()
        if not checkpoint.exists:
            return None
        row = checkpoint.to_dict() or {}
        fields = {
            "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
            "actor_id", "conversation_id", "ref", "digest", "byte_length", "payload",
        }
        if set(row) != fields or not isinstance(row.get("conversation_id"), str):
            raise ValueError("stored routing context header is malformed")
        header = {key: row[key] for key in fields - {"payload"}}
        loaded = ledger.load_routing_checkpoint(
            "context", actor_id=actor_id,
            conversation_id=header["conversation_id"], reference=header,
        )
        if (not isinstance(loaded, Mapping)
                or not isinstance(loaded.get("reference"), Mapping)
                or not isinstance(loaded.get("value"), Mapping)
                or dict(loaded["reference"]) != header):
            raise ValueError("stored routing context payload is malformed")
        snapshot = RoutingContextSnapshot.from_dict(loaded["value"])
        self._validate_captured_snapshot(
            snapshot, actor_id=actor_id, input_ref=input_ref, operation=operation,
            command=command, conversation_id=header["conversation_id"],
        )
        if "candidate" in command and (
            not isinstance(command.get("candidate"), str)
            or command["candidate"] != snapshot.accepted_command
        ):
            raise ValueError("accepted command candidate changed after capture")
        return snapshot

    def _verified_exchange_evidence(
        self, exchanges: list[Mapping[str, Any]], *, actor_id: int,
        conversation_id: str, search_filters: Mapping[str, Any] | None,
    ) -> dict[str, list[DocumentEvidence]]:
        from services.connectors.telegram.runtime.search_agent import (
            _arguments_preserve_search_bounds, _confirmed_result_documents,
            _evidence_is_scoped_read,
        )

        if search_filters is None:
            return {}
        evidence: dict[str, list[DocumentEvidence]] = {}
        for item in exchanges:
            exchange = item["exchange"]
            result = exchange.get("result")
            reference = item["reference"]
            payload = reference.get("payload") if isinstance(reference, Mapping) else None
            call_id = item["call_id"]
            ordinal = item["ordinal"]
            operation_id = item["operation_id"]
            checkpoint_id = item["checkpoint_conversation_id"]
            if (not isinstance(result, Mapping) or not isinstance(payload, Mapping)
                    or type(ordinal) is not int or ordinal < 0
                    or exchange.get("context_id") != self.context_id
                    or exchange.get("conversation_id") != conversation_id
                    or exchange.get("operation_id") != operation_id
                    or exchange.get("checkpoint_conversation_id") != checkpoint_id
                    or exchange.get("call_id") != call_id or exchange.get("ordinal") != ordinal):
                raise ValueError("verified search exchange scope is inconsistent")
            call = exchange.get("call")
            expected_call_id = call.get("id") if isinstance(call, Mapping) else None
            if not expected_call_id:
                expected_call_id = f"telegram-call-{operation_id}-{ordinal}"
            if (not isinstance(call, Mapping) or expected_call_id != call_id
                    or call.get("ordinal") != ordinal):
                raise ValueError("verified search call identity is inconsistent")
            name, arguments = call.get("name"), call.get("arguments")
            filters = search_filters or {}
            if (name not in {"query", "get_transcript", "list_recent_transcripts"}
                    or not _evidence_is_scoped_read(name, arguments)
                    or not _arguments_preserve_search_bounds(filters, arguments)
                    or result.get("error") is not None or result.get("is_error") is True):
                continue
            digest = payload.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("verified search reference digest is invalid")
            source_ref = item.get("turn_ref")
            turn_payload = item.get("turn")
            if not isinstance(source_ref, str) or not isinstance(turn_payload, Mapping):
                raise ValueError("verified exchange has no canonical history anchor")
            source = BoundRef(
                ref=source_ref, digest=item["turn_digest"], context_id=self.context_id,
                actor_id=actor_id, conversation_id=conversation_id,
                operation_id=operation_id, checkpoint_id=checkpoint_id,
                call_id=call_id, ordinal=ordinal,
                payload={"turn": turn_payload, "reference": reference,
                         "exchange_digest": digest},
            )
            for row in _confirmed_result_documents(result):
                if not isinstance(row, Mapping) or row.get("doc_id") is None:
                    continue
                doc_id = row.get("doc_id")
                if not isinstance(doc_id, str) or not doc_id or "/" in doc_id:
                    raise ValueError("verified search result document ID is malformed")
                metadata = self._addressed_metadata(doc_id)
                if metadata is None:
                    continue
                evidence.setdefault(doc_id, []).append(DocumentEvidence(
                    doc_id=doc_id, source_kind=SourceKind.VERIFIED_EXCHANGE,
                    source=source, context_id=self.context_id, actor_id=actor_id,
                    conversation_id=conversation_id, operation_id=operation_id,
                    fragment_refs=(source,),
                    metadata={**metadata, "doc_id": doc_id},
                ))
        return evidence

    def _capture_selected_history(
        self, *, context_ref: Any, actor_id: int, search_exchange_loader: Any,
    ) -> tuple[str, ConversationKind, dict[str, Any], BoundRef, list[Mapping[str, Any]]]:
        """Read one canonical selected conversation and freeze its full history."""
        context_snapshot = context_ref.get()
        context = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        if not context_snapshot.exists:
            raise ValueError("routing context record is missing or malformed")
        active = context.get("active_conversation_id")
        if active is not None and (not isinstance(active, str) or not active):
            raise ValueError("active conversation pointer is malformed")
        stable_chat = "telegram-chat-" + hashlib.sha256(
            self.context_id.encode("utf-8")).hexdigest()
        conversation_id = active or stable_chat
        conversation = self.repository.get_conversation(self.context_id, conversation_id)
        if conversation is None and active is not None:
            raise ValueError("selected conversation is missing")
        if conversation is None:
            kind = ConversationKind.CHAT
            last_turn_sequence = 0
            projection = {"kind": kind.value, "doc_id": None, "draft_id": None,
                          "dialog_number": None, "search_filters": None}
        else:
            kind = ConversationKind(conversation.kind)
            if conversation.conversation_id != conversation_id:
                raise ValueError("selected conversation binding is corrupt")
            build_conversation(
                conversation_id, kind, draft_id=conversation.draft_id,
                doc_id=conversation.doc_id, dialog_number=conversation.dialog_number,
                search_filters=conversation.search_filters, now=conversation.created_at,
            )
            projection = {"kind": kind.value, "doc_id": conversation.doc_id,
                          "draft_id": conversation.draft_id,
                          "dialog_number": conversation.dialog_number,
                          "search_filters": self._json_metadata_value(conversation.search_filters)}
            conversation_snapshot = (context_ref.collection("conversations")
                                     .document(conversation_id).get())
            stored = conversation_snapshot.to_dict() or {} if conversation_snapshot.exists else {}
            last_turn_sequence = stored.get("last_turn_sequence")
            if (not conversation_snapshot.exists
                    or stored.get("conversation_id") != conversation_id
                    or stored.get("kind") != kind.value
                    or stored.get("doc_id") != conversation.doc_id
                    or stored.get("draft_id") != conversation.draft_id
                    or stored.get("dialog_number") != conversation.dialog_number
                    or stored.get("search_filters") != conversation.search_filters
                    or type(last_turn_sequence) is not int or last_turn_sequence < 0):
                raise ValueError("selected conversation record is corrupt")
        rows = [] if conversation is None else self.repository.list_turns(
            self.context_id, conversation_id,
        )
        if any(not isinstance(turn, Mapping) or turn.get("conversation_id") != conversation_id
               for turn in rows):
            raise ValueError("selected conversation history is foreign or malformed")
        if conversation is not None and [turn.get("turn_sequence") for turn in rows] != list(
            range(1, last_turn_sequence + 1)
        ):
            raise ValueError("selected conversation history is incomplete")
        exchanges: list[Mapping[str, Any]] = []

        def hydrate(reference: Any, *, conversation_id: str, operation_id: str,
                    checkpoint_conversation_id: str, call_id: str, ordinal: int) -> Mapping[str, Any]:
            exchange = search_exchange_loader(
                reference, conversation_id=conversation_id, operation_id=operation_id,
                checkpoint_conversation_id=checkpoint_conversation_id,
                call_id=call_id, ordinal=ordinal,
            )
            if not isinstance(reference, Mapping) or not isinstance(exchange, Mapping):
                raise ValueError("verified search exchange is malformed")
            matching_turns = []
            for turn in rows:
                if not isinstance(turn, Mapping):
                    continue
                for part in turn.get("parts", ()):
                    if (isinstance(part, Mapping)
                            and part.get("type") == "search_exchange_ref"
                            and part.get("reference") == reference
                            and part.get("call_id") == call_id
                            and part.get("ordinal") == ordinal
                            and part.get("checkpoint_conversation_id") == checkpoint_conversation_id):
                        matching_turns.append(turn)
            if len(matching_turns) != 1:
                raise ValueError("verified exchange is not anchored to one canonical history turn")
            turn = matching_turns[0]
            turn_payload = self._json_metadata_value(turn)
            turn_id = turn.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError("verified exchange history turn ID is malformed")
            exchanges.append({"reference": reference, "exchange": exchange,
                              "conversation_id": conversation_id, "operation_id": operation_id,
                              "checkpoint_conversation_id": checkpoint_conversation_id,
                              "call_id": call_id, "ordinal": ordinal,
                              "turn": turn_payload,
                              "turn_ref": (f"{SESSIONS}/{self.context_id}/conversations/"
                                           f"{conversation_id}/turns/{turn_id}"),
                              "turn_digest": json_digest(turn_payload)})
            return exchange

        history = build_conversation_history(
            rows, conversation_id=conversation_id, omit_operation_id=self.operation_id,
            search_exchange_loader=hydrate,
        )
        reference = BoundRef(
            ref=f"{SESSIONS}/{self.context_id}/conversations/{conversation_id}/turns",
            digest=json_digest(history), context_id=self.context_id, actor_id=actor_id,
            conversation_id=conversation_id, operation_id=self.operation_id,
            payload={"turns": history, "conversation": projection},
        )
        return conversation_id, kind, projection, reference, exchanges

    def _project_command_references(
        self, command_text: str, references: tuple[CommandSpan, ...],
    ) -> tuple[tuple[CommandSpan, ...], tuple[str | int | None, ...], int | None,
               str | None, str | None]:
        spans = tuple(references)
        if any(not isinstance(span, CommandSpan) for span in spans):
            raise TypeError("explicit references must be validated CommandSpan values")
        values = tuple(self._reference_value(span) for span in spans)
        for span in spans:
            if span.end > len(command_text) or command_text[span.start:span.end] != span.raw_text:
                raise ValueError("explicit reference span does not match accepted command")
        local = [span for span in spans if span.kind is ExplicitReferenceKind.LOCAL_NUMBER]
        ids = [index for index, span in enumerate(spans)
               if span.kind is ExplicitReferenceKind.ID]
        urls = [span for span in spans if span.kind is ExplicitReferenceKind.URL]
        return (spans, values, local[0].parsed_value if len(local) == 1 else None,
                values[ids[0]] if len(ids) == 1 else None,
                urls[0].parsed_value if len(urls) == 1 else None)

    def _collect_routing_evidence(
        self, *, supplied: Mapping[str, Any] | None, selected_doc_id: str | None,
        selected_conversation_id: str, selected_kind: ConversationKind,
        history_ref: BoundRef, accepted_ref: BoundRef,
        references: tuple[CommandSpan, ...], reference_values: tuple[str | int | None, ...],
        verified_exchanges: list[Mapping[str, Any]], actor_id: int, operation_id: str,
        search_filters: Mapping[str, Any] | None,
    ) -> dict[str, tuple[DocumentEvidence, ...]]:
        evidence: dict[str, list[DocumentEvidence]] = {}
        if supplied is not None and not isinstance(supplied, Mapping):
            raise TypeError("document_evidence must be an object")
        for doc_id, raw_items in (supplied or {}).items():
            entries = (raw_items,) if isinstance(raw_items, DocumentEvidence) else tuple(raw_items)
            evidence.setdefault(doc_id, []).extend(entries)

        if isinstance(selected_doc_id, str) and selected_doc_id:
            metadata = self._addressed_metadata(selected_doc_id)
            if metadata is None:
                raise ValueError("selected document metadata is missing")
            evidence.setdefault(selected_doc_id, []).append(DocumentEvidence(
                doc_id=selected_doc_id, source_kind=SourceKind.ACTIVE_DOCUMENT,
                source=history_ref, context_id=self.context_id, actor_id=actor_id,
                conversation_id=selected_conversation_id, operation_id=operation_id,
                metadata={**metadata, "doc_id": selected_doc_id},
            ))

        for span, value in zip(references, reference_values):
            if span.kind is ExplicitReferenceKind.ID:
                doc_id = value
                metadata = self._addressed_metadata(doc_id)
                if metadata is not None:
                    evidence.setdefault(doc_id, []).append(DocumentEvidence(
                        doc_id=doc_id, source_kind=SourceKind.EXPLICIT_REFERENCE,
                        source=accepted_ref, context_id=self.context_id, actor_id=actor_id,
                        conversation_id=selected_conversation_id, operation_id=operation_id,
                        command_span=span, metadata={**metadata, "doc_id": doc_id},
                    ))
            elif span.kind is ExplicitReferenceKind.LOCAL_NUMBER:
                binding = self._dialog_binding(self.context_id, value)
                if binding is None:
                    continue
                doc_id = binding["doc_id"]
                metadata = self._addressed_metadata(doc_id)
                binding_payload = {"dialog_number": value,
                                   "conversation_id": binding["conversation_id"],
                                   "doc_id": doc_id}
                source = BoundRef(
                    ref=f"{SESSIONS}/{self.context_id}/dialogs/{value}",
                    digest=json_digest(binding_payload), context_id=self.context_id,
                    actor_id=actor_id, conversation_id=binding["conversation_id"],
                    operation_id=operation_id, payload=binding_payload,
                )
                evidence.setdefault(doc_id, []).append(DocumentEvidence(
                    doc_id=doc_id, source_kind=SourceKind.LOCAL_NUMBER, source=source,
                    context_id=self.context_id, actor_id=actor_id,
                    conversation_id=binding["conversation_id"], operation_id=operation_id,
                    command_span=span,
                    metadata={**(metadata or {}), "doc_id": doc_id},
                ))
            elif isinstance(value, str) and span.kind is ExplicitReferenceKind.URL:
                metadata = self._addressed_metadata(value)
                if metadata is not None:
                    evidence.setdefault(value, []).append(DocumentEvidence(
                        doc_id=value, source_kind=SourceKind.EXPLICIT_REFERENCE,
                        source=accepted_ref, context_id=self.context_id, actor_id=actor_id,
                        conversation_id=selected_conversation_id, operation_id=operation_id,
                        command_span=span, metadata={**metadata, "doc_id": value},
                    ))

        filters = ((search_filters or {}) if selected_kind is ConversationKind.SEARCH else None)
        for doc_id, proofs in self._verified_exchange_evidence(
            verified_exchanges, actor_id=actor_id,
            conversation_id=selected_conversation_id, search_filters=filters,
        ).items():
            evidence.setdefault(doc_id, []).extend(proofs)
        return {key: tuple(items) for key, items in evidence.items()}

    def capture_routing_context(
        self,
        *,
        actor_id: int,
        owner: str,
        fence: int,
        search_exchange_loader: Any,
        clock: Any = None,
        explicit_references: tuple[CommandSpan, ...] = (),
        document_evidence: Mapping[str, Any] | None = None,
        presentation_refs: tuple[BoundRef, ...] = (),
        confirmed_predecessor_saves: tuple[BoundRef, ...] = (),
        clarification_id: str | None = None,
        clarification_generation: int | None = None,
    ) -> RoutingContextSnapshot:
        """Capture or replay the immutable, context-scoped command input."""
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("routing capture context is missing")
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("routing capture operation is missing")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("routing capture actor is invalid")
        if not callable(search_exchange_loader):
            raise TypeError("a trusted search exchange loader is required")
        now = clock or (lambda: datetime.now(timezone.utc))
        ledger = FirestoreAgentLedger(
            self.client, context_id=self.context_id, operation_id=self.operation_id,
            owner=owner, fence=fence, clock=now,
        )
        context_ref, operation_ref, operation, command, input_ref = self._routing_scope(actor_id, ledger)
        saved = self._load_routing_checkpoint(
            ledger, actor_id=actor_id, input_ref=input_ref,
            operation=operation, command=command, operation_ref=operation_ref,
        )
        if saved is not None:
            return saved

        candidate = command.get("candidate")
        if not isinstance(candidate, str):
            raise ValueError("accepted command candidate is missing")
        accepted_command = parse_command(candidate).text
        if not accepted_command or candidate != accepted_command:
            raise ValueError("accepted command candidate is not normalized")
        if operation.get("identity_digest") != self._accepted_input_digest(command, accepted_command):
            raise ValueError("accepted command does not match its immutable input identity")

        (selected_conversation_id, conversation_kind, conversation_projection,
         history_ref, verified_exchanges) = self._capture_selected_history(
            context_ref=context_ref, actor_id=actor_id,
            search_exchange_loader=search_exchange_loader,
        )
        accepted_ref = BoundRef(
            ref=input_ref,
            digest=hashlib.sha256(accepted_command.encode("utf-8")).hexdigest(),
            context_id=self.context_id, actor_id=actor_id,
            conversation_id=selected_conversation_id, operation_id=self.operation_id,
            payload=self._accepted_payload_projection(command, accepted_command),
        )
        spans, span_values, local_number, parsed_id, parsed_url = self._project_command_references(
            accepted_command, tuple(explicit_references),
        )
        selected_doc_id = conversation_projection["doc_id"]
        all_evidence = self._collect_routing_evidence(
            supplied=document_evidence, selected_doc_id=selected_doc_id,
            selected_conversation_id=selected_conversation_id, selected_kind=conversation_kind,
            history_ref=history_ref, accepted_ref=accepted_ref, references=spans,
            reference_values=span_values, verified_exchanges=verified_exchanges,
            actor_id=actor_id, operation_id=self.operation_id,
            search_filters=conversation_projection["search_filters"],
        )
        snapshot = RoutingContextSnapshot(
            context_id=self.context_id, actor_id=actor_id,
            operation_id=self.operation_id,
            selected_conversation_id=selected_conversation_id,
            selected_conversation_kind=conversation_kind.value,
            selected_doc_id=selected_doc_id,
            accepted_command=accepted_command,
            accepted_command_ref=accepted_ref, full_history_ref=history_ref,
            explicit_references=spans, local_number=local_number,
            parsed_id=parsed_id, parsed_url=parsed_url,
            document_evidence=all_evidence,
            presentation_refs=tuple(presentation_refs),
            confirmed_predecessor_saves=tuple(confirmed_predecessor_saves),
            clarification_id=clarification_id,
            clarification_generation=clarification_generation,
        )
        self._validate_captured_snapshot(
            snapshot, actor_id=actor_id, input_ref=input_ref, operation=operation,
            command=command, conversation_id=selected_conversation_id,
        )
        ledger.save_routing_checkpoint(
            "context", actor_id=actor_id, conversation_id=selected_conversation_id,
            value=snapshot.to_dict(),
        )
        return snapshot

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
