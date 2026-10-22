"""Save one sealed Telegram draft through the shared dialog pipeline."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from core.config import get_default_dialog_date
from services.connectors.telegram.preprocessing import (
    context_word_capacity,
    preprocess_messages,
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is missing")
    return value


def _ordered(messages: Sequence[Any]) -> list[Any]:
    rows = list(messages)
    for row in rows:
        message_id = _get(row, "message_id")
        sequence = _get(row, "intake_sequence")
        if type(message_id) is not int or type(sequence) is not int:
            raise ValueError("sealed message ordering is malformed")
    return sorted(rows, key=lambda row: (_get(row, "intake_sequence"), _get(row, "message_id")))


def _dialog_date(messages: Sequence[Any]) -> str:
    if not messages:
        raise ValueError("sealed draft has no source messages")
    first = messages[0]
    for value in (_get(first, "original_date"), _get(first, "received_at")):
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                continue
            return get_default_dialog_date(value).isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    continue
                return get_default_dialog_date(parsed).isoformat()
            except ValueError:
                try:
                    return date.fromisoformat(value).isoformat()
                except ValueError:
                    continue
    raise ValueError("sealed source date is missing or malformed")


def _prepared_row(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    fields = (
        "order_index", "speaker", "reliable_author_key", "text", "word_count",
        "source_message_ids",
    )
    if all(hasattr(item, field) for field in fields):
        return {
            "order_index": item.order_index,
            "speaker": item.speaker,
            "reliable_author_key": item.reliable_author_key,
            "text": item.text,
            "word_count": item.word_count,
            "source_message_ids": tuple(item.source_message_ids),
        }
    return item


def _call_supports(callable_: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


@dataclass(frozen=True, slots=True)
class TelegramCompletionCertificate:
    """Immutable local record binding the verified save to its source draft."""

    context_id: str
    draft_id: str
    manifest_digest: str
    save_intent_id: str
    operation_id: str
    receipt_ref: str
    receipt_version: Any
    receipt_digest: str
    origin_binding: tuple[str, str, str]
    provider: str = "telegram"
    dialog_date: str = ""


@dataclass(frozen=True, slots=True)
class _SealedInput:
    context_id: str
    draft_id: str
    manifest_digest: str
    dialog_date: str
    messages: tuple[Any, ...]


class TelegramSaveError(RuntimeError):
    """Safe workflow error carrying a user-visible stage and bounded code."""

    def __init__(self, stage: str, error_code: str) -> None:
        self.stage = stage
        self.error_code = error_code
        super().__init__(f"Telegram save failed during {stage} ({error_code})")


class TelegramSaveWorkflow:
    """Coordinate transcription, preparation, analysis, shared save, and receipt."""

    def __init__(
        self,
        *,
        repository: Any,
        audio_transcriber: Any,
        context_id: str | None = None,
        preprocess: Callable[[Sequence[Any]], Any] = preprocess_messages,
        analyze: Callable[..., Any],
        classify: Callable[..., Any],
        save: Callable[..., Any],
    ) -> None:
        self.repository = repository
        self.audio_transcriber = audio_transcriber
        self.context_id = context_id
        self.preprocess = preprocess
        self.analyze = analyze
        self.classify = classify
        self.save_pipeline = save
        self.status: dict[str, str | None] = {
            "stage": "idle", "state": "idle", "error_code": None,
        }

    def _stage(self, stage: str) -> None:
        self.status = {"stage": stage, "state": "running", "error_code": None}

    async def _load_sealed_input(self, draft_id: str) -> _SealedInput:
        draft = await _await(self.repository.load_sealed_draft(draft_id))
        if not isinstance(draft, Mapping):
            raise ValueError("sealed draft is missing")
        if draft.get("draft_id") != draft_id:
            raise ValueError("sealed draft identity is invalid")
        state = draft.get("status", draft.get("state"))
        sealed = state == "sealed" or bool(
            draft.get("sealed_at") and draft.get("manifest_ref")
        )
        if not sealed:
            raise ValueError("draft is not sealed")

        digest = _safe_id(draft.get("manifest_digest"), "manifest digest")
        messages = draft.get("messages")
        manifest = draft.get("source_manifest")
        if (not isinstance(messages, Sequence) or isinstance(messages, (str, bytes))
                or not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes))):
            raise ValueError("sealed source manifest is malformed")
        if not messages or len(messages) != len(manifest):
            raise ValueError("sealed source manifest is incomplete")

        ordered = _ordered(messages)
        manifest_ordered = _ordered(manifest)
        message_identity = lambda rows: [
            (_get(row, "message_id"), _get(row, "intake_sequence")) for row in rows
        ]
        if message_identity(ordered) != message_identity(manifest_ordered):
            raise ValueError("sealed source manifest does not match draft messages")

        context_id = (
            self.context_id
            or _get(draft, "context_id")
            or getattr(self.repository, "context_id", None)
        )
        context_id = _safe_id(context_id, "context ID")
        return _SealedInput(
            context_id=context_id,
            draft_id=draft_id,
            manifest_digest=digest,
            dialog_date=_dialog_date(ordered),
            messages=tuple(ordered),
        )

    async def _analyze_prepared(
        self,
        messages: Sequence[Any],
        *,
        draft_id: str,
        context_id: str,
        dialog_date: str,
    ) -> Any:
        self._stage("preparation")
        prepared = await _await(self.preprocess(messages))
        if not isinstance(prepared, Sequence) or isinstance(prepared, (str, bytes)):
            raise ValueError("prepared utterances are malformed")
        prepared_rows = [_prepared_row(row) for row in prepared]
        context = {
            "provider": "telegram",
            "context_id": context_id,
            "draft_id": draft_id,
            "dialog_date": dialog_date,
            "messages": prepared_rows,
            "word_capacity": context_word_capacity(len(prepared_rows)),
        }
        self._stage("analysis")
        return await _await(self.analyze(prepared_rows, context=context))

    async def _transcribe_analyze_and_classify(
        self,
        sealed: _SealedInput,
    ) -> tuple[Any, Any]:
        self._stage("transcription")
        guard_before = await _await(
            self.repository.load_current_guard(sealed.draft_id)
        )
        if guard_before is None:
            raise ValueError("current resource guard is missing")

        async def analyze_after_transcription(rows: Sequence[Any]) -> Any:
            return await self._analyze_prepared(
                rows,
                draft_id=sealed.draft_id,
                context_id=sealed.context_id,
                dialog_date=sealed.dialog_date,
            )

        result = await _await(self.audio_transcriber.process_messages(
            sealed.messages,
            draft_id=sealed.draft_id,
            analyze=analyze_after_transcription,
        ))
        self._stage("classification")
        classified = await _await(self.classify(result, draft_id=sealed.draft_id))
        return classified, guard_before

    async def _save_and_complete(
        self,
        classified: Any,
        sealed: _SealedInput,
        *,
        intent_id: str,
        operation_id: str,
        guard_before: Any,
    ) -> TelegramCompletionCertificate:
        self._stage("save")
        save_args = {
            "draft_id": sealed.draft_id,
            "manifest_digest": sealed.manifest_digest,
            "intent_id": intent_id,
            "operation_id": operation_id,
        }
        provider_args = {}
        if _call_supports(self.save_pipeline, "provider"):
            provider_args["provider"] = "telegram"
        if _call_supports(self.save_pipeline, "dialog_date"):
            provider_args["dialog_date"] = sealed.dialog_date
        save_input = _with_save_metadata(classified, sealed.dialog_date)
        receipt = await _await(self.save_pipeline(
            save_input, **save_args, **provider_args,
        ))
        if not isinstance(receipt, Mapping):
            raise ValueError("save receipt is missing")

        self._stage("completion")
        if (receipt.get("save_intent_id") != intent_id
                or receipt.get("operation_id") != operation_id):
            raise ValueError("save receipt identity does not match request")
        guard_after = await _await(
            self.repository.load_current_guard(sealed.draft_id)
        )
        if guard_after != guard_before:
            raise ValueError("current resource guard changed")
        verified = await _await(self.repository.verify_save_receipt(
            receipt,
            draft_id=sealed.draft_id,
            manifest_digest=sealed.manifest_digest,
            guard=guard_after,
        ))
        if verified is not True:
            raise ValueError("save receipt could not be verified")

        receipt_ref = _safe_id(receipt.get("receipt_ref"), "receipt reference")
        receipt_payload = json.dumps(
            dict(receipt), sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        certificate = TelegramCompletionCertificate(
            context_id=sealed.context_id,
            draft_id=sealed.draft_id,
            manifest_digest=sealed.manifest_digest,
            save_intent_id=_safe_id(intent_id, "save intent ID"),
            operation_id=_safe_id(operation_id, "operation ID"),
            receipt_ref=receipt_ref,
            receipt_version=receipt.get("receipt_version"),
            receipt_digest=hashlib.sha256(receipt_payload).hexdigest(),
            origin_binding=(sealed.context_id, sealed.draft_id, sealed.manifest_digest),
            dialog_date=sealed.dialog_date,
        )
        committed = await _await(self.repository.commit_completion(certificate))
        if committed is False:
            raise ValueError("completion certificate was not committed")
        return certificate

    async def save(
        self,
        *,
        draft_id: str,
        intent_id: str,
        operation_id: str,
    ) -> Any:
        self._stage("load")
        try:
            sealed = await self._load_sealed_input(draft_id)
            classified, guard_before = await self._transcribe_analyze_and_classify(sealed)
            certificate = await self._save_and_complete(
                classified,
                sealed,
                intent_id=intent_id,
                operation_id=operation_id,
                guard_before=guard_before,
            )
            self.status = {"stage": "complete", "state": "completed", "error_code": None}
            return certificate
        except TelegramSaveError as error:
            stage = self.status["stage"] or "load"
            code = _error_code(error)
            self.status = {
                "stage": stage,
                "state": "failed",
                "error_code": code,
            }
            raise TelegramSaveError(stage, code) from None
        except Exception as error:
            stage = self.status["stage"] or "load"
            code = _error_code(error)
            self.status = {"stage": stage, "state": "failed", "error_code": code}
            raise TelegramSaveError(stage, code) from error


async def _await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _with_save_metadata(value: Any, dialog_date: str) -> Any:
    if isinstance(value, Mapping):
        result = dict(value)
        result["provider"] = "telegram"
        result["dialog_date"] = dialog_date
        return result
    return {"classified_result": value, "provider": "telegram", "dialog_date": dialog_date}


def _error_code(error: Exception) -> str:
    """Return a bounded exception category without exposing exception text."""
    name = type(error).__name__.lower()
    return "invalid_input" if isinstance(error, (ValueError, TypeError, KeyError)) else f"{name[:40]}_failed"
