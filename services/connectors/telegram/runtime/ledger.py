"""Durable agent checkpoint storage fenced by the Telegram queue lease."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.operation_queue import validate_context_queue_lease
from core.integrations.firestore.input_parts import MAX_PART_BYTES, read_input_parts, write_input_parts
from services.connectors.telegram.repository import SESSIONS
from services.connectors.telegram.repository._common import (
    ROUTING_INLINE_CHECKPOINT_BYTES,
    ROUTING_PROTOCOL,
    ROUTING_SCHEMA_VERSION,
    ROUTING_KINDS,
    _document_ref,
    _identifier,
    _routing_checkpoint_path,
    _read_routing_checkpoint_value,
    _routing_json_bytes,
    _routing_reference_data,
)

_INLINE_CHECKPOINT_VALUE_BYTES = ROUTING_INLINE_CHECKPOINT_BYTES
_PREPARED_APPLICATION_ROUTE_TOKEN = object()


class PreparedApplicationRoute:
    """Factory-issued immutable application checkpoint for joint route publication."""

    __slots__ = ("context_id", "operation_id", "conversation_id", "delivery_id",
                 "owner", "fence", "_client", "route_payload", "route_digest",
                 "_clock", "_stored_route_json", "_stored_calls_json", "route_round", "_token")

    def __init__(self, token: Any, *, context_id: str, operation_id: str,
                 conversation_id: str, delivery_id: str, owner: str, fence: int,
                 client: Any, clock: Any, route_payload: bytes, route_digest: str,
                 stored_route: Mapping[str, Any], stored_calls: Mapping[str, Any],
                 route_round: int | None):
        if token is not _PREPARED_APPLICATION_ROUTE_TOKEN:
            raise TypeError("prepared application routes are factory-issued")
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "delivery_id", delivery_id)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "fence", fence)
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "route_payload", bytes(route_payload))
        object.__setattr__(self, "route_digest", route_digest)
        object.__setattr__(self, "_stored_route_json", FirestoreAgentLedger._json_bytes(dict(stored_route)))
        object.__setattr__(self, "_stored_calls_json", FirestoreAgentLedger._json_bytes(dict(stored_calls)))
        object.__setattr__(self, "route_round", route_round)
        object.__setattr__(self, "_token", token)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("prepared application routes are immutable")

    @property
    def stored_route(self) -> dict[str, Any]:
        return FirestoreAgentLedger._json_value(self._stored_route_json)

    @property
    def stored_calls(self) -> dict[str, Any]:
        return FirestoreAgentLedger._json_value(self._stored_calls_json)


def _response_rounds(value: Any) -> list[Any]:
    if isinstance(value, Mapping) and value.get("_telegram_tool_loop_rounds") == 1:
        rounds = value.get("rounds")
        return list(rounds) if isinstance(rounds, list) else []
    return [value]


def _prefix_compatible(previous: Any, current: Any) -> bool:
    old_rounds, new_rounds = _response_rounds(previous), _response_rounds(current)
    if len(new_rounds) < len(old_rounds):
        return False
    if len(old_rounds) == 1 and not (
        isinstance(previous, Mapping)
        and previous.get("_telegram_tool_loop_rounds") == 1
    ):
        current_first = new_rounds[0]
        if isinstance(current_first, Mapping) and "response" in current_first:
            current_first = current_first["response"]
        return previous == current_first
    return old_rounds == new_rounds[:len(old_rounds)]


class FirestoreAgentLedger:
    """Persist immutable model rounds and tool results under one queue operation."""

    def __init__(self, client: Any, *, context_id: str, operation_id: str,
                 owner: str, fence: int, clock: Any):
        self.client, self.context_id, self.operation_id = client, context_id, operation_id
        self.owner, self.fence, self.clock = owner, fence, clock

    def _ref(self, delivery_id: str):
        if delivery_id != self.operation_id:
            raise ValueError("agent delivery does not match queue operation")
        return (self.client.collection(SESSIONS).document(self.context_id)
                .collection("operations").document(self.operation_id)
                .collection("agent_checkpoints").document("turn"))

    def _request_ref(self, delivery_id: str):
        if delivery_id != self.operation_id:
            raise ValueError("agent delivery does not match queue operation")
        return (self.client.collection(SESSIONS).document(self.context_id)
                .collection("operations").document(self.operation_id)
                .collection("agent_requests").document("request"))

    def checkpoint_conversation_id(self, delivery_id: str) -> str | None:
        """Return the existing operation checkpoint binding without loading payloads."""
        turn = self._ref(delivery_id).get().to_dict() or {}
        request = self._request_ref(delivery_id).get().to_dict() or {}
        values = {
            row.get("conversation_id") for row in (turn, request)
            if isinstance(row.get("conversation_id"), str)
        }
        if len(values) > 1:
            raise ValueError("agent checkpoint conversation binding changed")
        return next(iter(values), None)

    def _assert_current(self, transaction: Any) -> None:
        context_ref = self.client.collection(SESSIONS).document(self.context_id)
        operation_ref = context_ref.collection("operations").document(self.operation_id)
        context_snapshot = context_ref.get(transaction=transaction)
        operation_snapshot = operation_ref.get(transaction=transaction)
        context_data = context_snapshot.to_dict() or {} if context_snapshot.exists else {}
        operation_data = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        now = self.clock()
        if (not operation_snapshot.exists or operation_data.get("state") != "running"
                or operation_data.get("operation_id") != self.operation_id
                or not validate_context_queue_lease(
                    context_data, self.operation_id, owner=self.owner, fence=self.fence,
                    now=now, expiry=context_data.get("lock_expires_at"))):
            raise PermissionError("Telegram operation lease is stale")

    def _assert_live(self) -> None:
        context = self.client.collection(SESSIONS).document(self.context_id).get()
        operation = (self.client.collection(SESSIONS).document(self.context_id)
                    .collection("operations").document(self.operation_id).get())
        context_data = context.to_dict() or {} if context.exists else {}
        operation_data = operation.to_dict() or {} if operation.exists else {}
        if (not operation.exists or operation_data.get("state") != "running"
                or not validate_context_queue_lease(
                    context_data, self.operation_id, owner=self.owner, fence=self.fence,
                    now=self.clock(), expiry=context_data.get("lock_expires_at"))):
            raise PermissionError("Telegram operation lease is stale")

    def _write_input_document(
        self, reference: Any, data: dict[str, Any], *, write_assertion: Any = None,
    ) -> None:
        """Commit one input part or manifest while the queue fence is current."""
        @firestore.transactional
        def _txn(transaction: Any) -> None:
            self._assert_current(transaction)
            if write_assertion is not None:
                write_assertion(transaction)
            transaction.set(reference, data)

        _txn(self.client.transaction(max_attempts=5))

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        def encode(item: Any) -> Any:
            if isinstance(item, bytes):
                return {"__telegram_bytes__": base64.b64encode(item).decode("ascii")}
            raise TypeError(f"unsupported durable agent value: {type(item).__name__}")
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=encode).encode("utf-8")

    @staticmethod
    def _json_value(payload: bytes) -> Any:
        def decode(item: Any) -> Any:
            if isinstance(item, dict):
                if set(item) == {"__telegram_bytes__"}:
                    return base64.b64decode(item["__telegram_bytes__"], validate=True)
                return {key: decode(value) for key, value in item.items()}
            if isinstance(item, list):
                return [decode(value) for value in item]
            return item
        return decode(json.loads(payload.decode("utf-8")))

    def _store_payload(
        self, kind: str, value: Any, *, write_assertion: Any = None,
    ) -> dict[str, Any]:
        payload = self._json_bytes(value)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) <= _INLINE_CHECKPOINT_VALUE_BYTES:
            return {"_inline": value, "sha256": digest, "byte_length": len(payload)}
        self._assert_live()
        data_operation = "telegram-ledger-" + hashlib.sha256(
            f"{self.context_id}\0{self.operation_id}\0{kind}\0{digest}".encode()
        ).hexdigest()
        input_ref = f"dialog_operations/{data_operation}/input_parts/manifest"
        write_input_parts(
            self.client, input_ref, payload,
            write_document=lambda reference, data: self._write_input_document(
                reference, data, write_assertion=write_assertion,
            ),
        )
        return {"_input_ref": input_ref, "sha256": digest,
                "byte_length": len(payload), "max_part_bytes": MAX_PART_BYTES}

    def _load_payload(self, stored: Any, *, transaction: Any = None,
                      kind: str | None = None) -> Any:
        if not isinstance(stored, Mapping):
            return stored
        # Preserve the original native checkpoint decoder. Contextual route
        # envelopes receive stricter, route-specific verification below.
        if "_inline" in stored:
            return deepcopy(stored["_inline"])
        input_ref = stored.get("_input_ref")
        if isinstance(input_ref, str):
            payload = read_input_parts(self.client, input_ref, transaction=transaction)
            if (hashlib.sha256(payload).hexdigest() != stored.get("sha256")
                    or len(payload) != stored.get("byte_length")):
                raise ValueError("agent payload parts do not verify")
            return self._json_value(payload)
        return dict(stored)

    def _load_application_route(self, stored: Any) -> Any:
        """Decode legacy routes compatibly and verify contextual envelopes strictly."""
        if not isinstance(stored, Mapping):
            return stored
        keys = set(stored)
        inline_shape = keys == {"_inline", "sha256", "byte_length"}
        parts_shape = keys == {"_input_ref", "sha256", "byte_length", "max_part_bytes"}
        if inline_shape:
            value = deepcopy(stored["_inline"])
            payload = self._json_bytes(value)
            digest, length = stored.get("sha256"), stored.get("byte_length")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or type(length) is not int or length != len(payload)
                    or hashlib.sha256(payload).hexdigest() != digest
                    or len(payload) > _INLINE_CHECKPOINT_VALUE_BYTES):
                raise ValueError("contextual application route integrity is invalid")
        elif parts_shape:
            value = self._load_payload(stored, kind="application-route")
            digest, length = stored.get("sha256"), stored.get("byte_length")
            expected_operation = "telegram-ledger-" + hashlib.sha256(
                f"{self.context_id}\0{self.operation_id}\0application-route\0{digest}".encode()
            ).hexdigest()
            if (stored.get("_input_ref")
                    != f"dialog_operations/{expected_operation}/input_parts/manifest"
                    or type(length) is not int or length <= _INLINE_CHECKPOINT_VALUE_BYTES
                    or type(stored.get("max_part_bytes")) is not int
                    or stored.get("max_part_bytes") != MAX_PART_BYTES):
                raise ValueError("contextual application route parts scope is invalid")
        else:
            # Actual historical non-contextual route encodings remain readable.
            value = self._load_payload(stored, kind="application-route")
            if isinstance(value, Mapping) and "contextual" in value:
                raise ValueError("contextual application route lacks verified storage")
            return value
        return value

    def _routing_command(self, transaction: Any = None) -> dict[str, Any]:
        """Resolve this operation's accepted command, including after candidate cleanup."""
        operation_ref = (self.client.collection(SESSIONS).document(self.context_id)
                         .collection("operations").document(self.operation_id))
        operation_snapshot = operation_ref.get(transaction=transaction)
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        input_ref = operation.get("input_ref")
        parts = input_ref.split("/") if isinstance(input_ref, str) else []
        if (operation.get("context_id") not in (None, self.context_id)
                or operation.get("operation_id") != self.operation_id
                or len(parts) != 4
                or parts[:3] != [SESSIONS, self.context_id, "commands"]):
            raise PermissionError("routing checkpoint has no accepted command")
        try:
            _identifier(parts[3], "command_id")
        except ValueError:
            raise PermissionError("routing checkpoint command reference is invalid") from None
        command_snapshot = _document_ref(self.client, input_ref).get(transaction=transaction)
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        if (not command_snapshot.exists or command.get("context_id") != self.context_id
                or command.get("operation_id") != self.operation_id
                or type(command.get("user_id")) is not int or command["user_id"] <= 0):
            raise PermissionError("routing checkpoint command binding is invalid")
        return command

    def _routing_command_ref(self) -> str:
        """Return the accepted command ref for a stored contextual checkpoint."""
        operation_ref = (self.client.collection(SESSIONS).document(self.context_id)
                         .collection("operations").document(self.operation_id))
        snapshot = operation_ref.get()
        operation = snapshot.to_dict() or {} if snapshot.exists else {}
        input_ref = operation.get("input_ref")
        if not isinstance(input_ref, str):
            raise ValueError("routing checkpoint has no accepted command ref")
        return input_ref

    def _check_routing_sibling_scope(
        self, transaction: Any, *, actor_id: int, conversation_id: str,
    ) -> dict[str, Any]:
        """Read both supported headers and ensure one operation scope."""
        references = {}
        for kind in sorted(ROUTING_KINDS):
            checkpoint_ref = _document_ref(
                self.client, _routing_checkpoint_path(self.context_id, self.operation_id, kind),
            )
            snapshot = checkpoint_ref.get(transaction=transaction)
            if not snapshot.exists:
                references[kind] = None
                continue
            row = snapshot.to_dict() or {}
            if set(row) != {
                "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
                "actor_id", "conversation_id", "ref", "digest", "byte_length", "payload",
            }:
                raise ValueError("routing checkpoint sibling header is malformed")
            header = _routing_reference_data({key: row.get(key) for key in (
                "routing_protocol", "schema_version", "kind", "context_id",
                "operation_id", "actor_id", "conversation_id", "ref", "digest",
                "byte_length",
            )})
            if (header["kind"] != kind or header["context_id"] != self.context_id
                    or header["operation_id"] != self.operation_id
                    or header["actor_id"] != actor_id
                    or header["conversation_id"] != conversation_id):
                raise ValueError("routing checkpoint actor/conversation scope changed")
            references[kind] = header
        return references

    def save_routing_checkpoint(
        self, kind: str, *, actor_id: int, conversation_id: str, value: Any,
    ) -> dict[str, Any]:
        """Publish one immutable, operation-owned context-v2 routing artifact."""
        if not isinstance(kind, str) or kind not in ROUTING_KINDS:
            raise ValueError("routing checkpoint kind is invalid")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("routing checkpoint actor is invalid")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("routing checkpoint conversation is invalid")
        self._assert_live()
        value_copy = deepcopy(value)
        envelope = {
            "routing_protocol": ROUTING_PROTOCOL,
            "schema_version": ROUTING_SCHEMA_VERSION,
            "kind": kind,
            "context_id": self.context_id,
            "operation_id": self.operation_id,
            "actor_id": actor_id,
            "conversation_id": conversation_id,
            "value": value_copy,
        }
        payload = _routing_json_bytes(envelope)
        digest = hashlib.sha256(payload).hexdigest()
        byte_length = len(payload)
        reference = {
            "routing_protocol": ROUTING_PROTOCOL,
            "schema_version": ROUTING_SCHEMA_VERSION,
            "kind": kind,
            "context_id": self.context_id,
            "operation_id": self.operation_id,
            "actor_id": actor_id,
            "conversation_id": conversation_id,
            "ref": _routing_checkpoint_path(self.context_id, self.operation_id, kind),
            "digest": digest,
            "byte_length": byte_length,
        }
        checkpoint_ref = _document_ref(self.client, reference["ref"])

        # Fast retry path: validate lease and command, then preserve the original row.
        @firestore.transactional
        def find_existing(transaction: Any) -> dict[str, Any] | None:
            self._assert_current(transaction)
            command = self._routing_command(transaction)
            if command.get("user_id") != actor_id:
                raise PermissionError("routing checkpoint actor does not own command")
            self._check_routing_sibling_scope(
                transaction, actor_id=actor_id, conversation_id=conversation_id,
            )
            snapshot = checkpoint_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            row = snapshot.to_dict() or {}
            if set(row) != set(reference) | {"payload"}:
                raise ValueError("routing checkpoint record shape is invalid")
            stored_ref = _routing_reference_data({key: row.get(key) for key in reference})
            if stored_ref != reference:
                raise ValueError("routing checkpoint is immutable")
            return reference

        existing = find_existing(self.client.transaction(max_attempts=5))
        if existing is not None:
            _read_routing_checkpoint_value(self.client, existing)
            return existing

        kind_key = "routing-" + kind
        stored_payload = self._store_payload(
            kind_key, envelope, write_assertion=self._assert_current,
        )
        # Recheck fence, accepted command and canonical row after parts are ready.
        @firestore.transactional
        def publish(transaction: Any) -> dict[str, Any]:
            self._assert_current(transaction)
            command = self._routing_command(transaction)
            if command.get("user_id") != actor_id:
                raise PermissionError("routing checkpoint actor does not own command")
            self._check_routing_sibling_scope(
                transaction, actor_id=actor_id, conversation_id=conversation_id,
            )
            snapshot = checkpoint_ref.get(transaction=transaction)
            if snapshot.exists:
                row = snapshot.to_dict() or {}
                if set(row) != set(reference) | {"payload"}:
                    raise ValueError("routing checkpoint is immutable")
                stored_ref = _routing_reference_data({key: row.get(key) for key in reference})
                if stored_ref != reference:
                    raise ValueError("routing checkpoint is immutable")
                return reference
            transaction.set(checkpoint_ref, {**reference, "payload": stored_payload})
            return reference

        published = publish(self.client.transaction(max_attempts=5))
        _read_routing_checkpoint_value(self.client, published)
        return published

    def load_routing_checkpoint(
        self, kind: str, *, actor_id: int, conversation_id: str,
        reference: Any = None,
    ) -> dict[str, Any] | None:
        """Read and verify an operation checkpoint without requiring a live lease."""
        if not isinstance(kind, str) or kind not in ROUTING_KINDS:
            raise ValueError("routing checkpoint kind is invalid")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("routing checkpoint actor is invalid")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("routing checkpoint conversation is invalid")
        command = self._routing_command()
        if command.get("user_id") != actor_id:
            raise ValueError("routing checkpoint actor does not own command")
        self._check_routing_sibling_scope(
            None, actor_id=actor_id, conversation_id=conversation_id,
        )
        if reference is None:
            path = _routing_checkpoint_path(self.context_id, self.operation_id, kind)
            snapshot = _document_ref(self.client, path).get()
            if not snapshot.exists:
                return None
            row = snapshot.to_dict() or {}
            reference = {key: row.get(key) for key in (
                "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
                "actor_id", "conversation_id", "ref", "digest", "byte_length",
            )}
        ref = _routing_reference_data(reference)
        if (ref["context_id"] != self.context_id
                or ref["operation_id"] != self.operation_id
                or ref["kind"] != kind or ref["actor_id"] != actor_id
                or ref["conversation_id"] != conversation_id):
            raise ValueError("routing checkpoint reference scope is invalid")
        value = _read_routing_checkpoint_value(self.client, ref)
        return {"reference": ref, "value": value}

    def save_search_exchange(
        self, conversation_id: str, *, checkpoint_conversation_id: str,
        call: Mapping[str, Any], canonical_call_id: str, result: Any,
    ) -> dict[str, Any]:
        """Store one complete search tool exchange with immutable call binding."""
        ordinal = call.get("ordinal")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("search call ordinal is invalid")
        call_id = canonical_call_id
        name = call.get("name")
        arguments = call.get("arguments")
        if (not isinstance(call_id, str) or not call_id
                or not isinstance(name, str) or not isinstance(arguments, Mapping)):
            raise ValueError("search call identity is invalid")
        kind = f"search-history-{ordinal:08d}"
        envelope = {
            "context_id": self.context_id,
            "conversation_id": conversation_id,
            "checkpoint_conversation_id": checkpoint_conversation_id,
            "operation_id": self.operation_id,
            "call_id": call_id,
            "ordinal": ordinal,
            "provenance": "application" if call.get("round") == -1 else "model",
            "call": dict(call),
            "result": result,
        }
        return {
            "version": 1, "context_id": self.context_id,
            "conversation_id": conversation_id, "operation_id": self.operation_id,
            "checkpoint_conversation_id": checkpoint_conversation_id,
            "call_id": call_id, "ordinal": ordinal, "payload_kind": kind,
            "payload": self._store_payload(kind, envelope,
                                           write_assertion=self._assert_current),
        }

    def load_search_exchange(
        self, reference: Any, *, conversation_id: str, operation_id: str,
        checkpoint_conversation_id: str, call_id: str, ordinal: int,
        transaction: Any = None,
    ) -> dict[str, Any]:
        """Hydrate an immutable search exchange after verifying every binding."""
        if (not isinstance(reference, Mapping) or type(reference.get("version")) is not int
                or reference.get("version") != 1
                or reference.get("context_id") != self.context_id
                or reference.get("conversation_id") != conversation_id
                or reference.get("operation_id") != operation_id
                or reference.get("checkpoint_conversation_id") != checkpoint_conversation_id
                or reference.get("call_id") != call_id
                or type(reference.get("ordinal")) is not int
                or reference.get("ordinal") != ordinal):
            raise ValueError("search history reference binding is invalid")
        kind = reference.get("payload_kind")
        payload_ref = reference.get("payload")
        expected_kind = f"search-history-{ordinal:08d}"
        if kind != expected_kind or not isinstance(payload_ref, Mapping):
            raise ValueError("search history reference is invalid")
        digest = payload_ref.get("sha256")
        byte_length = payload_ref.get("byte_length")
        if (not isinstance(digest, str) or len(digest) != 64
                or type(byte_length) is not int or byte_length < 0):
            raise ValueError("search history payload integrity metadata is invalid")
        input_ref = payload_ref.get("_input_ref")
        if input_ref is not None:
            data_operation = "telegram-ledger-" + hashlib.sha256(
                f"{self.context_id}\0{operation_id}\0{kind}\0{digest}".encode()
            ).hexdigest()
            if input_ref != f"dialog_operations/{data_operation}/input_parts/manifest":
                raise ValueError("search history payload reference is not bound")
        value = self._load_payload(payload_ref, transaction=transaction, kind=kind)
        encoded = self._json_bytes(value)
        if hashlib.sha256(encoded).hexdigest() != digest or len(encoded) != byte_length:
            raise ValueError("search history payload integrity check failed")
        if (not isinstance(value, Mapping) or value.get("context_id") != self.context_id
                or value.get("conversation_id") != conversation_id
                or value.get("checkpoint_conversation_id") != checkpoint_conversation_id
                or value.get("operation_id") != operation_id
                or value.get("call_id") != call_id or value.get("ordinal") != ordinal):
            raise ValueError("search history payload binding is invalid")
        return dict(value)

    def load_turn(self, conversation_id: str, delivery_id: str) -> dict[str, Any] | None:
        ref = self._ref(delivery_id)
        snapshot = ref.get()
        value = snapshot.to_dict() or {} if snapshot.exists else None
        if value is not None and value.get("conversation_id") != conversation_id:
            raise ValueError("agent checkpoint conversation binding changed")
        if value is not None:
            value.setdefault("application_route", None)
            value.setdefault("application_route_round", None)
            value["model_response"] = self._load_payload(value.get("model_response"), kind="model")
            value["tool_calls"] = self._load_payload(value.get("tool_calls", []), kind="calls")
            if value.get("application_route") is not None:
                value["application_route"] = self._load_application_route(
                    value["application_route"],
                )
                self._verify_contextual_application_route(
                    conversation_id, delivery_id, value["application_route"],
                    value.get("tool_calls", []), value.get("model_response"),
                    value.get("application_route_round"),
                )
            raw_results = value.get("tool_results", {})
            restored = {str(key): self._load_payload(result, kind=f"tool-{key}")
                        for key, result in raw_results.items()}
            result_rows = self._ref(delivery_id).collection("tool_results").stream()
            for result_snapshot in result_rows:
                row = result_snapshot.to_dict() or {}
                ordinal = row.get("ordinal")
                if type(ordinal) is int:
                    restored[str(ordinal)] = self._load_payload(
                        row.get("result"), kind=f"tool-{ordinal}",
                    )
            value["tool_results"] = restored
        return value

    def _verify_contextual_application_route(
        self, checkpoint_conversation_id: str, delivery_id: str,
        application_route: Any, tool_calls: Any, model_response: Any,
        application_route_round: Any,
    ) -> None:
        """Verify both durable halves and source bindings before a contextual read."""
        from services.connectors.telegram.repository._commands import _TelegramCommandMixin
        from services.connectors.telegram.runtime.routing_context import RoutingContextSnapshot

        if not isinstance(application_route, Mapping) or "contextual" not in application_route:
            return  # Actual pre-context legacy application routes remain compatible.
        contextual, executable = _TelegramCommandMixin._validate_joint_contextual_shape(
            application_route, context_id=self.context_id,
            operation_id=self.operation_id,
            checkpoint_conversation_id=checkpoint_conversation_id,
        )
        context_path = _routing_checkpoint_path(self.context_id, self.operation_id, "context")
        if contextual.get("context_ref") != context_path:
            raise ValueError("contextual source path is invalid")
        header_snapshot = _document_ref(self.client, context_path).get()
        header_row = header_snapshot.to_dict() or {} if header_snapshot.exists else {}
        ref_data = _routing_reference_data({key: header_row.get(key) for key in (
            "routing_protocol", "schema_version", "kind", "context_id", "operation_id",
            "actor_id", "conversation_id", "ref", "digest", "byte_length",
        )})
        if (ref_data["kind"] != "context"
                or ref_data["context_id"] != self.context_id
                or ref_data["operation_id"] != self.operation_id
                or type(ref_data["actor_id"]) is not int
                or ref_data["actor_id"] != contextual["actor_id"]
                or ref_data["conversation_id"] != contextual["source_conversation_id"]
                or ref_data["digest"] != contextual.get("context_digest")):
            raise ValueError("contextual source reference scope changed")
        snapshot = RoutingContextSnapshot.from_dict(
            _read_routing_checkpoint_value(self.client, ref_data),
        )
        snapshot_dict = snapshot.to_dict()
        if (snapshot.context_id != self.context_id
                or snapshot.operation_id != self.operation_id
                or type(snapshot.actor_id) is not int
                or snapshot.actor_id != contextual["actor_id"]
                or snapshot.selected_conversation_id != contextual["source_conversation_id"]
                or snapshot.accepted_command != contextual["accepted_command"]
                or _routing_json_bytes(snapshot.accepted_command_ref.to_dict())
                != _routing_json_bytes(contextual.get("accepted_command_ref"))
                or _routing_json_bytes(snapshot.full_history_ref.to_dict())
                != _routing_json_bytes(contextual.get("full_history_ref"))):
            raise ValueError("contextual source snapshot does not match stored references")
        accepted = snapshot.accepted_command_ref
        if accepted.ref != self._routing_command_ref():
            raise ValueError("contextual bound references are invalid")

        operation_ref = (self.client.collection(SESSIONS).document(self.context_id)
                         .collection("operations").document(self.operation_id))
        operation_snapshot = operation_ref.get()
        operation = operation_snapshot.to_dict() or {} if operation_snapshot.exists else {}
        summary = contextual.get("summary")
        from services.connectors.telegram.repository._commands import _TelegramCommandMixin
        if (not operation_snapshot.exists or operation.get("input_ref") != accepted.ref
                or not isinstance(operation.get("identity_digest"), str)
                or not isinstance(summary, Mapping)
                or _TelegramCommandMixin._route_control_data(summary)
                != operation.get("route_control")):
            raise ValueError("contextual decision lacks its matching ordinary route")
        command_snapshot = _document_ref(self.client, operation["input_ref"]).get()
        command = command_snapshot.to_dict() or {} if command_snapshot.exists else {}
        if (not command_snapshot.exists
                or command.get("context_id") != self.context_id
                or command.get("operation_id") != self.operation_id
                or type(command.get("user_id")) is not int
                or command["user_id"] != snapshot.actor_id
                or ("candidate" in command
                    and command.get("candidate") != snapshot.accepted_command)):
            raise ValueError("contextual accepted command record is invalid")
        identity = hashlib.sha256(json.dumps({
            "kind": command.get("kind"), "text": snapshot.accepted_command,
            "action_metadata": command.get("action_metadata"),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if identity != operation.get("identity_digest"):
            raise ValueError("contextual accepted command identity changed")
        _TelegramCommandMixin._validate_joint_routing_snapshot(
            snapshot, client=self.client, context_id=self.context_id,
            operation_id=self.operation_id, actor_id=snapshot.actor_id,
            input_ref=operation.get("input_ref"), operation=operation,
            command=command, conversation_id=snapshot.selected_conversation_id,
        )
        _TelegramCommandMixin._validate_joint_application_projection(application_route, summary)
        decision = __import__(
            "services.connectors.telegram.runtime.routing_context",
            fromlist=["RoutingDecision"],
        ).RoutingDecision.from_dict(contextual["decision"])
        decision.validate_command_spans(snapshot.accepted_command)
        if (decision.disposition.value == "clarify") != ("intent_ref" in contextual):
            raise ValueError("clarification route is missing its prepared intent")
        request = self._request_ref(delivery_id).get().to_dict() or {}
        if request and request.get("conversation_id") != checkpoint_conversation_id:
            raise ValueError("agent request conversation binding changed")

        if executable:
            if (type(application_route.get("ordinal")) is not int
                    or application_route["ordinal"] < 0
                    or not isinstance(application_route.get("name"), str)
                    or not isinstance(application_route.get("arguments"), Mapping)
                    or not isinstance(tool_calls, list)):
                raise ValueError("contextual executable route is malformed")
            matching = [call for call in tool_calls if isinstance(call, Mapping)
                        and type(call.get("ordinal")) is int
                        and call.get("ordinal") == application_route["ordinal"]]
            if (len(matching) != 1
                    or matching[0].get("name") != application_route["name"]
                    or matching[0].get("arguments") != application_route["arguments"]
                    or ("call_id" in application_route
                        and matching[0].get("id") != application_route["call_id"])):
                raise ValueError("contextual route has no matching application call")
        from services.connectors.telegram.agent import _contextual_call_provenance_valid
        if not _contextual_call_provenance_valid({
            "application_route": application_route,
            "application_route_round": application_route_round,
            "model_response": model_response, "tool_calls": tool_calls,
        }):
            raise ValueError("contextual calls lack native or declared application provenance")
        _TelegramCommandMixin._validate_joint_intent_binding(
            self.client, contextual, snapshot, ref_data,
        )

    def _verify_contextual_intent(
        self, contextual: Mapping[str, Any], snapshot: Any, context_reference: Mapping[str, Any],
    ) -> None:
        from services.connectors.telegram.repository._commands import _TelegramCommandMixin

        _TelegramCommandMixin._validate_joint_intent_binding(
            self.client, contextual, snapshot, context_reference,
        )

    def save_request(self, conversation_id: str, delivery_id: str, request: str) -> None:
        if not isinstance(request, str):
            raise TypeError("agent request must be text")
        ref = self._request_ref(delivery_id)
        stored = self._store_payload("request", request)
        transaction = self.client.transaction()
        @firestore.transactional
        def commit(tx: Any) -> None:
            self._assert_current(tx)
            snapshot = ref.get(transaction=tx)
            row = snapshot.to_dict() or {} if snapshot.exists else {}
            if row.get("conversation_id") not in (None, conversation_id):
                raise ValueError("agent checkpoint conversation binding changed")
            prior = self._load_payload(row.get("request"), kind="request") if row.get("request") is not None else request
            if prior != request:
                raise ValueError("agent request is immutable")
            tx.set(ref, {"conversation_id": conversation_id, "delivery_id": delivery_id,
                         "request": stored}, merge=True)
        commit(transaction)

    def load_request(self, conversation_id: str, delivery_id: str) -> str | None:
        row = self._request_ref(delivery_id).get().to_dict() or {}
        if not row:
            return None
        if row.get("conversation_id") != conversation_id:
            raise ValueError("agent checkpoint conversation binding changed")
        value = self._load_payload(row.get("request"), kind="request")
        return value if isinstance(value, str) else None

    def save_turn(self, conversation_id: str, delivery_id: str, *,
                  model_response: Any, tool_calls: list[Any],
                  application_route: Mapping[str, Any] | None = None) -> None:
        ref = self._ref(delivery_id)
        stored_response = self._store_payload("model", model_response)
        stored_calls = self._store_payload("calls", tool_calls)
        stored_route = (
            self._store_payload(
                "application-route", dict(application_route),
                write_assertion=self._assert_current,
            ) if application_route is not None else None
        )
        transaction = self.client.transaction()
        @firestore.transactional
        def commit(tx: Any) -> None:
            self._assert_current(tx)
            snapshot = ref.get(transaction=tx)
            prior = snapshot.to_dict() or {} if snapshot.exists else {}
            if prior.get("conversation_id") not in (None, conversation_id):
                raise ValueError("agent checkpoint conversation binding changed")
            prior_calls = self._load_payload(prior.get("tool_calls", []), kind="calls")
            if prior_calls and prior_calls != tool_calls[:len(prior_calls)]:
                raise ValueError("persisted agent calls are immutable")
            prior_response = self._load_payload(prior.get("model_response"), kind="model")
            if prior_response is not None and not _prefix_compatible(prior_response, model_response):
                raise ValueError("persisted agent model rounds are immutable")
            prior_stored_route = prior.get("application_route")
            prior_route = (self._load_payload(prior_stored_route, kind="application-route")
                           if prior_stored_route is not None else None)
            if (prior_route is not None and application_route is not None
                    and prior_route != dict(application_route)):
                raise ValueError("persisted application route is immutable")
            row = {**prior, "conversation_id": conversation_id, "delivery_id": delivery_id,
                   "model_response": stored_response, "tool_calls": stored_calls}
            route_to_preserve = (prior_stored_route if prior_stored_route is not None
                                 else stored_route)
            if route_to_preserve is not None:
                row["application_route"] = route_to_preserve
                row["application_route_round"] = prior.get("application_route_round")
            tx.set(ref, row, merge=False)
        commit(transaction)

    def save_application_route(
        self, conversation_id: str, delivery_id: str, *, route: Mapping[str, Any],
        tool_calls: list[Any], application_route_round: int | None,
    ) -> None:
        """Persist trusted app-route provenance without inventing model content."""
        prepared = self._prepare_application_route(
            conversation_id, delivery_id, route=route, tool_calls=tool_calls,
            application_route_round=application_route_round,
        )
        ref = self._ref(delivery_id)
        stored_calls = prepared.stored_calls
        stored_route = prepared.stored_route
        transaction = self.client.transaction()

        @firestore.transactional
        def commit(tx: Any) -> None:
            self._assert_current(tx)
            snapshot = ref.get(transaction=tx)
            prior = snapshot.to_dict() or {} if snapshot.exists else {}
            if prior.get("conversation_id") not in (None, conversation_id):
                raise ValueError("agent checkpoint conversation binding changed")
            existing_stored_route = prior.get("application_route")
            existing_route = (self._load_payload(existing_stored_route, kind="application-route")
                              if existing_stored_route is not None else None)
            expected_route = self._json_value(prepared.route_payload)
            if existing_route is not None and existing_route != expected_route:
                raise ValueError("persisted application route is immutable")
            prior_calls = self._load_payload(prior.get("tool_calls", []), kind="calls")
            if prior_calls and prior_calls != tool_calls[:len(prior_calls)]:
                raise ValueError("persisted agent calls are immutable")
            prior_round = prior.get("application_route_round")
            if ("application_route_round" in prior
                    and prior_round != application_route_round):
                raise ValueError("persisted application route is immutable")
            tx.set(ref, {
                "conversation_id": conversation_id,
                "delivery_id": delivery_id,
                "application_route": existing_stored_route or stored_route,
                "application_route_round": application_route_round,
                "tool_calls": stored_calls,
            }, merge=True)

        commit(transaction)

    def prepare_application_route(
        self, conversation_id: str, delivery_id: str, *, route: Mapping[str, Any],
        tool_calls: list[Any], application_route_round: int | None,
    ) -> PreparedApplicationRoute:
        """Stage a verified route payload before a transaction publishes its pair."""
        return self._prepare_application_route(
            conversation_id, delivery_id, route=route, tool_calls=tool_calls,
            application_route_round=application_route_round,
        )

    def _prepare_application_route(
        self, conversation_id: str, delivery_id: str, *, route: Mapping[str, Any],
        tool_calls: list[Any], application_route_round: int | None,
    ) -> PreparedApplicationRoute:
        self._ref(delivery_id)
        self._assert_live()
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("agent checkpoint conversation is invalid")
        if not isinstance(route, Mapping) or not isinstance(tool_calls, list):
            raise TypeError("application route preparation is invalid")
        if (application_route_round is not None
                and (type(application_route_round) is not int or application_route_round < 0)):
            raise ValueError("application route round is invalid")
        route_copy = deepcopy(dict(route))
        calls_copy = deepcopy(tool_calls)
        payload = self._json_bytes(route_copy)
        digest = hashlib.sha256(payload).hexdigest()
        stored_calls = self._store_payload("calls", calls_copy)
        stored_route = self._store_payload(
            "application-route", route_copy, write_assertion=self._assert_current,
        )
        return PreparedApplicationRoute(
            _PREPARED_APPLICATION_ROUTE_TOKEN, context_id=self.context_id,
            operation_id=self.operation_id, conversation_id=conversation_id,
            delivery_id=delivery_id, owner=self.owner, fence=self.fence,
            client=self.client, clock=self.clock,
            route_payload=payload, route_digest=digest, stored_route=stored_route,
            stored_calls=stored_calls, route_round=application_route_round,
        )

    def save_tool_result(self, conversation_id: str, delivery_id: str,
                         ordinal: int, result: Any) -> None:
        parent_ref = self._ref(delivery_id)
        ref = parent_ref.collection("tool_results").document(f"result-{ordinal:08d}")
        stored_result = self._store_payload(f"tool-{ordinal}", result)
        transaction = self.client.transaction()
        @firestore.transactional
        def commit(tx: Any) -> None:
            self._assert_current(tx)
            parent_snapshot = parent_ref.get(transaction=tx)
            parent = parent_snapshot.to_dict() or {} if parent_snapshot.exists else {}
            snapshot = ref.get(transaction=tx)
            row = snapshot.to_dict() or {} if snapshot.exists else {}
            if parent.get("conversation_id") != conversation_id:
                raise ValueError("agent checkpoint conversation binding changed")
            if row.get("ordinal") not in (None, ordinal):
                raise ValueError("persisted tool result is immutable")
            if row.get("result") is not None and row.get("result") != stored_result:
                raise ValueError("persisted tool result is immutable")
            tx.set(ref, {"ordinal": ordinal, "result": deepcopy(stored_result)}, merge=False)
        commit(transaction)
