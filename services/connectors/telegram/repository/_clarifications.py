"""Addressed clarification reads and transaction-scoped guard snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Mapping

from core.integrations.firestore.operation_queue import validate_context_queue_lease
from services.connectors.telegram.models import (
    ClarificationState,
    PendingClarification,
)
from services.connectors.telegram.repository._common import SESSIONS, _document_ref


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,480}$")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid clarification {field}")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"invalid clarification {field}")
    return value


def _aware(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("invalid clarification now")
    return value


def _strict_scoped_record(
    data: Mapping[str, Any], *, context_id: str, clarification_id: str, actor_id: int,
) -> PendingClarification | None:
    if (data.get("context_id") != context_id
            or data.get("clarification_id") != clarification_id
            or type(data.get("actor_id")) is not int
            or data.get("actor_id") != actor_id
            or data.get("protocol") is None):
        return None
    if data.get("protocol") != "clarification-v1":
        raise ValueError("unsupported clarification protocol")
    return PendingClarification.from_dict(data)


def _read_waiting_target(
    repository: Any, transaction: Any, *, context_id: str, clarification_id: str,
    actor_id: int, generation: int, revision: int, pointer_id: Any,
    pointer_generation: Any,
) -> PendingClarification | None:
    if (pointer_id != clarification_id or type(pointer_generation) is not int
            or pointer_generation != generation):
        return None
    ref = _document_ref(
        repository.client,
        f"{SESSIONS}/{context_id}/clarifications/{clarification_id}",
    )
    snapshot = ref.get(transaction=transaction)
    if not snapshot.exists:
        return None
    clarification = _strict_scoped_record(
        snapshot.to_dict() or {}, context_id=context_id,
        clarification_id=clarification_id, actor_id=actor_id,
    )
    if (clarification is None or clarification.state is not ClarificationState.WAITING
            or clarification.generation != generation
            or clarification.revision != revision):
        return None
    return clarification


def _accepted_scope(
    repository: Any, transaction: Any, *, context_id: str, operation_id: str,
    actor_id: int, owner: str, fence: int, now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    context_ref = _document_ref(repository.client, f"{SESSIONS}/{context_id}")
    operation_ref = context_ref.collection("operations").document(operation_id)
    context_snapshot = context_ref.get(transaction=transaction)
    operation_snapshot = operation_ref.get(transaction=transaction)
    if not context_snapshot.exists or not operation_snapshot.exists:
        return None
    context = context_snapshot.to_dict() or {}
    operation = operation_snapshot.to_dict() or {}
    if (operation.get("operation_id") != operation_id
            or operation.get("state") != "running"
            or not validate_context_queue_lease(
                context, operation_id, owner=owner, fence=fence, now=now,
                expiry=context.get("lock_expires_at"),
            )):
        return None
    _command_ref, command = repository._command_for_operation(
        transaction, context_id=context_id, operation_id=operation_id,
        operation=operation,
    )
    if (command is None or command.get("context_id") != context_id
            or command.get("operation_id") != operation_id
            or type(command.get("user_id")) is not int
            or command.get("user_id") != actor_id):
        return None
    return context, operation, command


@dataclass(frozen=True, slots=True)
class _ClarificationGuard:
    """Transaction-read snapshots; never a capability to write or execute."""

    context: Mapping[str, Any]
    operation: Mapping[str, Any]
    accepted_command: Mapping[str, Any]
    clarification: PendingClarification | None


class _TelegramClarificationMixin:
    def get_clarification(
        self, context_id: str, clarification_id: str, *, actor_id: int,
    ) -> PendingClarification | None:
        context_id = _identifier(context_id, "context_id")
        clarification_id = _identifier(clarification_id, "clarification_id")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        ref = _document_ref(
            self.client,
            f"{SESSIONS}/{context_id}/clarifications/{clarification_id}",
        )
        snapshot = ref.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return _strict_scoped_record(
            data, context_id=context_id, clarification_id=clarification_id,
            actor_id=actor_id,
        )

    def get_pending_clarification(
        self, context_id: str, *, actor_id: int,
    ) -> PendingClarification | None:
        context_id = _identifier(context_id, "context_id")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        context_ref = _document_ref(self.client, f"{SESSIONS}/{context_id}")
        snapshot = context_ref.get()
        if not snapshot.exists:
            return None
        context = snapshot.to_dict() or {}
        clarification_id = context.get("pending_clarification_id")
        generation = context.get("clarification_generation", 0)
        if clarification_id is None:
            return None
        if (not isinstance(clarification_id, str) or not _IDENTIFIER.fullmatch(clarification_id)
                or type(generation) is not int or generation < 1):
            return None
        clarification = self.get_clarification(
            context_id, clarification_id, actor_id=actor_id,
        )
        if (clarification is None or clarification.state is not ClarificationState.WAITING
                or clarification.generation != generation):
            return None
        return clarification

    def _read_clarification_guard(
        self,
        transaction: Any,
        *,
        context_id: str,
        operation_id: str,
        actor_id: int,
        owner: str,
        fence: int,
        now: datetime,
        clarification_id: str | None = None,
        expected_generation: int | None = None,
        expected_revision: int | None = None,
    ) -> _ClarificationGuard | None:
        """Read accepted operation scope and optional waiting record in a transaction.

        This method performs no writes. A successful result only describes records
        observed under the supplied transaction and does not grant effect authority.
        """
        context_id = _identifier(context_id, "context_id")
        operation_id = _identifier(operation_id, "operation_id")
        if transaction is None:
            raise ValueError("clarification guard requires a transaction")
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("invalid clarification actor_id")
        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid clarification owner")
        _positive_integer(fence, "fence")
        now = _aware(now)
        if clarification_id is None:
            if expected_generation is not None or expected_revision is not None:
                raise ValueError("invalid clarification expectations")
        else:
            clarification_id = _identifier(clarification_id, "clarification_id")
            _positive_integer(expected_generation, "expected_generation")
            _positive_integer(expected_revision, "expected_revision")

        scope = _accepted_scope(
            self, transaction, context_id=context_id, operation_id=operation_id,
            actor_id=actor_id, owner=owner, fence=fence, now=now,
        )
        if scope is None:
            return None
        context, operation, command = scope

        pointer_id = context.get("pending_clarification_id")
        pointer_generation = context.get("clarification_generation", 0)
        if clarification_id is None:
            if pointer_id is not None:
                return None
            if type(pointer_generation) is not int or pointer_generation < 0:
                return None
            clarification = None
        else:
            clarification = _read_waiting_target(
                self, transaction, context_id=context_id,
                clarification_id=clarification_id, actor_id=actor_id,
                generation=expected_generation, revision=expected_revision,
                pointer_id=pointer_id, pointer_generation=pointer_generation,
            )
            if clarification is None:
                return None
        return _ClarificationGuard(
            context=dict(context), operation=dict(operation),
            accepted_command=dict(command), clarification=clarification,
        )
