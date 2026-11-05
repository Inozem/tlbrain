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

_INLINE_CHECKPOINT_VALUE_BYTES = 4 * 1024


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

    def _load_payload(self, stored: Any) -> Any:
        if not isinstance(stored, Mapping):
            return stored
        if "_inline" in stored:
            return deepcopy(stored["_inline"])
        input_ref = stored.get("_input_ref")
        if isinstance(input_ref, str):
            payload = read_input_parts(self.client, input_ref)
            if (hashlib.sha256(payload).hexdigest() != stored.get("sha256")
                    or len(payload) != stored.get("byte_length")):
                raise ValueError("agent payload parts do not verify")
            return self._json_value(payload)
        return dict(stored)

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
    ) -> dict[str, Any]:
        """Hydrate an immutable search exchange after verifying every binding."""
        if (not isinstance(reference, Mapping) or reference.get("version") != 1
                or reference.get("context_id") != self.context_id
                or reference.get("conversation_id") != conversation_id
                or reference.get("operation_id") != operation_id
                or reference.get("checkpoint_conversation_id") != checkpoint_conversation_id
                or reference.get("call_id") != call_id
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
        value = self._load_payload(payload_ref)
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
            value["model_response"] = self._load_payload(value.get("model_response"))
            value["tool_calls"] = self._load_payload(value.get("tool_calls", []))
            if value.get("application_route") is not None:
                value["application_route"] = self._load_payload(
                    value["application_route"],
                )
            raw_results = value.get("tool_results", {})
            restored = {str(key): self._load_payload(result)
                        for key, result in raw_results.items()}
            result_rows = self._ref(delivery_id).collection("tool_results").stream()
            for result_snapshot in result_rows:
                row = result_snapshot.to_dict() or {}
                ordinal = row.get("ordinal")
                if type(ordinal) is int:
                    restored[str(ordinal)] = self._load_payload(row.get("result"))
            value["tool_results"] = restored
        return value

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
            prior = self._load_payload(row.get("request")) if row.get("request") is not None else request
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
        value = self._load_payload(row.get("request"))
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
            prior_calls = self._load_payload(prior.get("tool_calls", []))
            if prior_calls and prior_calls != tool_calls[:len(prior_calls)]:
                raise ValueError("persisted agent calls are immutable")
            prior_response = self._load_payload(prior.get("model_response"))
            if prior_response is not None and not _prefix_compatible(prior_response, model_response):
                raise ValueError("persisted agent model rounds are immutable")
            prior_stored_route = prior.get("application_route")
            prior_route = (self._load_payload(prior_stored_route)
                           if prior_stored_route is not None else None)
            if (prior_route is not None and application_route is not None
                    and prior_route != dict(application_route)):
                raise ValueError("persisted application route is immutable")
            row = {"conversation_id": conversation_id, "delivery_id": delivery_id,
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
        ref = self._ref(delivery_id)
        stored_calls = self._store_payload("calls", tool_calls)
        stored_route = self._store_payload(
            "application-route", dict(route), write_assertion=self._assert_current,
        )
        transaction = self.client.transaction()

        @firestore.transactional
        def commit(tx: Any) -> None:
            self._assert_current(tx)
            snapshot = ref.get(transaction=tx)
            prior = snapshot.to_dict() or {} if snapshot.exists else {}
            if prior.get("conversation_id") not in (None, conversation_id):
                raise ValueError("agent checkpoint conversation binding changed")
            existing_stored_route = prior.get("application_route")
            existing_route = (self._load_payload(existing_stored_route)
                              if existing_stored_route is not None else None)
            expected_route = deepcopy(dict(route))
            if existing_route is not None and existing_route != expected_route:
                raise ValueError("persisted application route is immutable")
            prior_calls = self._load_payload(prior.get("tool_calls", []))
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
