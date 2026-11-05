"""Save one sealed Telegram draft through the shared dialog pipeline."""

from __future__ import annotations

import hashlib
import asyncio
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from core.config import get_default_dialog_date
from core.operations.models import OperationError
from core.operations.retries import ConfirmedStageFailure, ConfirmedStageSuccess, UnknownOutcome
from core.dialogs.classification import FolderSelectionResult
from services.connectors.telegram.context import build_save_analysis_context
from services.connectors.telegram.preprocessing import preprocess_messages
from services.connectors.telegram.save_checkpoints import STAGE_PAYLOAD_REF_FIELD, stage_payload_ref


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
            "source_message_ids": list(item.source_message_ids),
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
    classification_outcome: str | None = None
    skipped_utterance_count: int = 0
    doc_id: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramClassifiedDialog:
    """Normalized save payload paired with transient provider-neutral outcome."""

    payload: Any
    selection: FolderSelectionResult
    skipped_utterance_count: int = 0


def _checkpoint_classified(value: Any) -> Any:
    if not isinstance(value, TelegramClassifiedDialog):
        return _checkpoint_analysis(value)
    return {
        "telegram_classified_dialog": True,
        "payload": _checkpoint_analysis(value.payload),
        "selection": {
            "folder_id": value.selection.folder_id,
            "speaker_candidate_ids": list(value.selection.speaker_candidate_ids),
            "requires_unassigned": value.selection.requires_unassigned,
        },
        "skipped_utterance_count": value.skipped_utterance_count,
    }


def _restore_classified(value: Any) -> Any:
    if not isinstance(value, Mapping) or not (
        value.get("telegram_classified_dialog") is True
        or value.get("__telegram_classified_dialog__") is True
    ):
        return value
    selection = value.get("selection") or {}
    return TelegramClassifiedDialog(
        payload=value.get("payload"),
        selection=FolderSelectionResult(
            folder_id=selection.get("folder_id"),
            speaker_candidate_ids=tuple(selection.get("speaker_candidate_ids", ())),
            requires_unassigned=bool(selection.get("requires_unassigned", False)),
        ),
        skipped_utterance_count=max(0, int(value.get("skipped_utterance_count", 0))),
    )


def _checkpoint_analysis(value: Any) -> Any:
    """Normalize analysis output to Firestore-native values before persistence.

    Analysis providers return JSON-shaped payloads in the Telegram save flow.
    Normalize mappings and sequences recursively, and fail clearly if an
    injected provider returns an opaque model that cannot be safely restored.
    """
    if value is None or isinstance(value, (str, bool, int, float, bytes, datetime)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("analysis result mapping keys must be strings")
        return {key: _checkpoint_analysis(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_analysis(item) for item in value]
    raise TypeError("analysis result must contain only Firestore-native values")


@dataclass(frozen=True, slots=True)
class _SealedInput:
    context_id: str
    draft_id: str
    manifest_digest: str
    dialog_date: str
    messages: tuple[Any, ...]


class TelegramSaveError(RuntimeError):
    """Safe workflow error carrying a user-visible stage and bounded code."""

    def __init__(self, stage: str, error_code: str, *, retryable_stage: bool | None = None) -> None:
        self.stage = stage
        self.error_code = error_code
        self.retryable_stage = retryable_stage
        super().__init__(f"Telegram save failed during {stage} ({error_code})")


class _NoSupportedContentError(TelegramSaveError):
    """Trusted connector refusal for a sealed input with no saveable content."""

    def __init__(self) -> None:
        super().__init__("preparation", "no_supported_content", retryable_stage=False)


class TelegramSavePending(RuntimeError):
    """Control signal for a durably accepted save that is awaiting indexing."""

    def __init__(self, *, stage: str = "indexing") -> None:
        self.stage = stage
        super().__init__("Telegram save is still running")


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
        self._skipped_content_count = 0

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
        operation_id: str,
        retry_governor: Any,
        operation_lease: Any,
    ) -> Any:
        self._stage("preparation")

        async def prepare_for_checkpoint():
            prepared = await _await(self.preprocess(messages))
            if not isinstance(prepared, Sequence) or isinstance(prepared, (str, bytes)):
                if retry_governor is not None and operation_lease is not None:
                    raise ValueError("prepared utterances are malformed")
                return prepared
            rows = [_prepared_row(row) for row in prepared]
            persist_prepared = getattr(self.repository, "save_prepared_utterances", None)
            if callable(persist_prepared):
                from services.connectors.telegram.models import PreparedUtterance
                persisted_rows = [PreparedUtterance(
                    order_index=row["order_index"], speaker=row["speaker"],
                    reliable_author_key=row.get("reliable_author_key"), text=row["text"],
                    word_count=row["word_count"],
                    source_message_ids=tuple(row["source_message_ids"]),
                ) for row in rows]
                audio_rows = [row for row in messages if _get(row, "audio_file_id")]
                completed_audio_ids = [
                    _get(row, "message_id") for row in audio_rows
                    if isinstance(_get(row, "transcription"), str)
                    or (getattr(_get(row, "transcription_state"), "value",
                                _get(row, "transcription_state")) == "completed"
                        and isinstance(_get(row, "transcription_text"), str))
                ]
                preparation_digest = hashlib.sha256(json.dumps(
                    rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                await _await(persist_prepared(
                    draft_id, persisted_rows, required_audio_count=len(audio_rows),
                    completed_audio_ids=completed_audio_ids,
                    context_capacity=max(30, len(rows) + 20),
                    preparation_digest=preparation_digest,
                ))
            if retry_governor is not None and operation_lease is not None:
                rows = [_checkpoint_analysis(row) for row in rows]
                return await self._pack_large_stage_result("preparation", rows, operation_id)
            return rows

        async def reconcile_preparation():
            return ConfirmedStageSuccess(await prepare_for_checkpoint())

        prepared = await _run_telegram_stage(retry_governor, operation_id, operation_lease,
            "telegram.preparation", prepare_for_checkpoint,
            reconcile=reconcile_preparation)
        prepared = await self._unpack_large_stage_result(prepared)
        if not isinstance(prepared, Sequence) or isinstance(prepared, (str, bytes)):
            raise ValueError("prepared utterances are malformed")
        prepared_rows, contentless_source_count = _project_prepared_utterances(prepared, messages)
        self._skipped_content_count = max(
            self._skipped_content_count, contentless_source_count,
        )
        if self._skipped_content_count:
            await _record_skipped_content_count(
                self, self._skipped_content_count,
            )
        if not prepared_rows:
            self._stage("preparation")
            raise _NoSupportedContentError()
        context = build_save_analysis_context(
            prepared_rows, context_id=context_id, draft_id=draft_id,
            dialog_date=dialog_date,
        )
        self._stage("analysis")
        analyze = lambda: self.analyze(prepared_rows, context=context)
        if retry_governor is None or operation_lease is None:
            return await _await(analyze())

        async def analyze_for_checkpoint():
            value = _checkpoint_analysis(await _await(analyze()))
            return await self._pack_large_stage_result("analysis", value, operation_id)

        result = await _run_telegram_stage(retry_governor, operation_id, operation_lease,
            "telegram.analysis", analyze_for_checkpoint)
        return await self._unpack_large_stage_result(result)

    async def _pack_large_stage_result(self, stage: str, value: Any, operation_id: str) -> Any:
        store = getattr(self.repository, "store_retry_stage_payload", None)
        if not callable(store):
            return value
        ref = await _await(store(operation_id, stage, value))
        return {STAGE_PAYLOAD_REF_FIELD: ref}

    async def _unpack_large_stage_result(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload_ref = stage_payload_ref(value)
        if payload_ref is None:
            return value
        load = getattr(self.repository, "load_retry_stage_payload", None)
        if not callable(load):
            raise ValueError("prepared checkpoint payload is unavailable")
        return await _await(load(payload_ref))

    async def _transcribe_analyze_and_classify(
        self,
        sealed: _SealedInput,
        *,
        operation_id: str,
        retry_governor: Any,
        operation_lease: Any,
    ) -> tuple[Any, Any, str | None]:
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
                operation_id=operation_id,
                retry_governor=retry_governor,
                operation_lease=operation_lease,
            )

        audio_args = {"draft_id": sealed.draft_id, "analyze": analyze_after_transcription}
        if retry_governor is not None and operation_lease is not None:
            audio_args.update(retry_governor=retry_governor, operation_id=operation_id,
                              operation_lease=operation_lease)
        result = await _await(self.audio_transcriber.process_messages(sealed.messages, **audio_args))
        self._stage("classification")
        topic_hint = None
        load_context = getattr(self.repository, "get_context", None)
        if callable(load_context):
            context = await _await(load_context(sealed.context_id))
            topic_hint = _get(context, "topic_name")
        folder_path = _get(result, "folder_path")
        folder_id = _get(result, "folder_id")
        if folder_path is not None and folder_id is not None:
            raise ValueError("specify either folder_path or folder_id, not both")
        classify_args = {"draft_id": sealed.draft_id}
        assert_current = getattr(self.repository, "_assert_current", None)
        if callable(assert_current) and _call_supports(self.classify, "assert_current"):
            classify_args["assert_current"] = assert_current
        if folder_path is not None and not _call_supports(self.classify, "folder_path"):
            raise TypeError("classify callback does not support explicit folder_path")
        if folder_id is not None and not _call_supports(self.classify, "folder_id"):
            raise TypeError("classify callback does not support explicit folder_id")
        for name, value in (
            ("topic_hint", topic_hint),
            ("folder_path", folder_path),
            ("folder_id", folder_id),
        ):
            if value is not None and _call_supports(self.classify, name):
                classify_args[name] = value

        async def classify_for_checkpoint():
            classified = await _await(self.classify(result, **classify_args))
            if retry_governor is None or operation_lease is None:
                return classified
            return await self._pack_large_stage_result(
                "classification", _checkpoint_classified(classified), operation_id,
            )

        classified = await _run_telegram_stage(retry_governor, operation_id, operation_lease,
            "telegram.classification", classify_for_checkpoint,
            retryable=lambda error: not isinstance(error, (ValueError, TypeError, KeyError)))
        if retry_governor is not None and operation_lease is not None:
            classified = await self._unpack_large_stage_result(classified)
            classified = _restore_classified(classified)
        outcome = None
        selection = None
        if isinstance(classified, TelegramClassifiedDialog):
            selection = classified.selection
            outcome = (
                "unassigned" if selection.requires_unassigned
                else "ambiguous" if selection.speaker_candidates_ambiguous
                else None
            )
            classified = classified.payload
        classified, skipped_count = _project_contentless_utterances(
            classified, allowed_skip_count=self._skipped_content_count,
        )
        self._skipped_content_count = max(self._skipped_content_count, skipped_count)
        if self._skipped_content_count:
            await _record_skipped_content_count(
                self, self._skipped_content_count,
            )
        if skipped_count and not _utterances(classified):
            self._stage("preparation")
            raise _NoSupportedContentError()
        if self._skipped_content_count:
            classified = TelegramClassifiedDialog(
                payload=classified, selection=selection or FolderSelectionResult(folder_id=None),
                skipped_utterance_count=self._skipped_content_count,
            )
        return classified, guard_before, outcome

    async def _save_and_complete(
        self,
        classified: Any,
        sealed: _SealedInput,
        *,
        intent_id: str,
        operation_id: str,
        guard_before: Any,
        classification_outcome: str | None = None,
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
        skipped_count = max(
            self._skipped_content_count,
            classified.skipped_utterance_count
            if isinstance(classified, TelegramClassifiedDialog) else 0,
        )
        try:
            receipt = await _await(self.save_pipeline(
                save_input, **save_args, **provider_args,
            ))
        except Exception as error:
            error.telegram_stage_retryable = _core_save_retryable(
                getattr(self.repository, "client", None), operation_id, error,
            )
            raise
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

        doc_id = (receipt.get("doc_id") if isinstance(receipt, Mapping) else None)
        if not isinstance(doc_id, str) or not doc_id:
            load_doc_id = getattr(self.repository, "load_completed_doc_id", None)
            if callable(load_doc_id):
                doc_id = await _await(load_doc_id(sealed.draft_id, sealed.manifest_digest))

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
            doc_id=doc_id if isinstance(doc_id, str) and doc_id else None,
            dialog_date=sealed.dialog_date,
            classification_outcome=classification_outcome,
            skipped_utterance_count=skipped_count,
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
        retry_governor: Any = None,
        operation_lease: Any = None,
    ) -> Any:
        self._stage("load")
        self._skipped_content_count = 0
        try:
            sealed = await self._load_sealed_input(draft_id)
            classified, guard_before, classification_outcome = await self._transcribe_analyze_and_classify(
                sealed, operation_id=operation_id, retry_governor=retry_governor,
                operation_lease=operation_lease,
            )
            certificate = await self._save_and_complete(
                classified,
                sealed,
                intent_id=intent_id,
                operation_id=operation_id,
                guard_before=guard_before,
                classification_outcome=classification_outcome,
            )
            self.status = {"stage": "complete", "state": "completed", "error_code": None}
            return certificate
        except TelegramSavePending as pending:
            self.status = {
                "stage": pending.stage,
                "state": "running",
                "error_code": None,
            }
            raise
        except TelegramSaveError as error:
            if isinstance(error, _NoSupportedContentError):
                stage, code = "preparation", "no_supported_content"
            else:
                stage = self.status["stage"] or "load"
                code = _error_code(error)
            self.status = {
                "stage": stage,
                "state": "failed",
                "error_code": code,
            }
            raise TelegramSaveError(
                stage, code, retryable_stage=error.retryable_stage,
            ) from None
        except Exception as error:
            stage = self.status["stage"] or "load"
            code = _error_code(error)
            self.status = {"stage": stage, "state": "failed", "error_code": code}
            raise TelegramSaveError(stage, code,
                retryable_stage=getattr(error, "telegram_stage_retryable", None)) from error


async def _await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _project_prepared_utterances(prepared: Sequence[Any], messages: Sequence[Any]) -> tuple[list[Any], int]:
    """Filter cached blank rows and count sealed sources confirmed contentless."""
    contentless_ids = set()
    for message in messages:
        message_id = _get(message, "message_id")
        if (type(message_id) is int and not _get(message, "audio_file_id")
                and not any(
                    isinstance(_get(message, field), str) and _get(message, field).strip()
                    for field in ("text", "caption", "transcription", "transcription_text")
                )):
            contentless_ids.add(message_id)
    retained = []
    for item in prepared:
        row = _prepared_row(item)
        if not isinstance(_get(row, "text"), str):
            raise ValueError("prepared utterance text is malformed")
        if _get(row, "text").strip():
            retained.append(row)
            continue
        source_ids = _get(row, "source_message_ids")
        if (not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes))
                or not source_ids or any(type(source_id) is not int for source_id in source_ids)
                or any(source_id not in contentless_ids for source_id in source_ids)):
            raise ValueError("blank prepared utterance has no matching contentless source")
    return retained, len(contentless_ids)


async def _record_skipped_content_count(workflow: Any, count: int) -> None:
    """Persist a deterministic skip count through the invocation-bound repository port."""
    record = getattr(workflow.repository, "record_skipped_utterance_count", None)
    if callable(record):
        await _await(record(count))


def _utterances(value: Any) -> list[Any]:
    rows = _get(value, "utterances")
    return list(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []


def _project_contentless_utterances(
    classified: Any, *, allowed_skip_count: int = 0,
) -> tuple[Any, int]:
    """Filter only blank normalized rows while preserving validated source data."""
    wrapper = classified if isinstance(classified, TelegramClassifiedDialog) else None
    payload = wrapper.payload if wrapper is not None else classified
    if not isinstance(payload, Mapping) or payload.get("provider") != "telegram":
        return classified, 0
    if "utterances" not in payload:
        return classified, 0
    rows = _utterances(payload)
    if not isinstance(payload.get("utterances"), Sequence
    ) or isinstance(payload.get("utterances"), (str, bytes)):
        raise ValueError("classified utterances are malformed")
    retained = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"speaker", "text"}:
            raise ValueError("classified utterance is malformed")
        if not isinstance(row.get("speaker"), str) or not row["speaker"].strip():
            raise ValueError("classified speaker is malformed")
        if not isinstance(row.get("text"), str):
            raise ValueError("classified utterance text is malformed")
        if row["text"].strip():
            retained.append(dict(row))
    skipped = len(rows) - len(retained)
    if skipped > allowed_skip_count:
        raise ValueError("blank classification rows do not match contentless source messages")
    if skipped == 0:
        return classified, 0
    projected = dict(payload) if isinstance(payload, Mapping) else payload
    if isinstance(projected, dict):
        projected["utterances"] = retained
    else:
        raise ValueError("classified utterances are malformed")
    if wrapper is not None:
        return TelegramClassifiedDialog(
            projected, wrapper.selection, wrapper.skipped_utterance_count + skipped,
        ), skipped
    return projected, skipped


async def _run_telegram_stage(
    governor: Any,
    operation_id: str,
    lease: Any,
    stage_key: str,
    callback: Callable[[], Any],
    *,
    retryable: Callable[[BaseException], bool] | None = None,
    reconcile: Callable[[], Any] | None = None,
) -> Any:
    if governor is None or lease is None:
        return await _await(callback())
    try:
        return await governor.run_stage_async(
            operation_id, stage_key, callback, owner=lease.owner, fence=lease.fence,
            retryable=retryable or (lambda error: not isinstance(error, (ValueError, TypeError, KeyError))),
            reconcile=reconcile or (lambda: ConfirmedStageFailure(OperationError(
                code="unknown_outcome",
                category="execution",
                message="Telegram stage outcome was not durably recorded",
                retryable=True,
                stage=stage_key,
            ))),
        )
    except Exception as error:
        if isinstance(error, UnknownOutcome):
            error.telegram_stage_retryable = True
            raise
        try:
            stage = await asyncio.to_thread(governor.get_stage, operation_id, stage_key)
            last_error = stage.last_error
            error.telegram_stage_retryable = (
                True if last_error is None else bool(
                    last_error.retryable and stage.failures < governor.max_failures
                )
            )
        except Exception:
            # A failed read cannot establish a permanent or exhausted failure.
            error.telegram_stage_retryable = True
        raise


def _with_save_metadata(value: Any, dialog_date: str) -> Any:
    if isinstance(value, TelegramClassifiedDialog):
        value = value.payload
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


def _core_save_retryable(client: Any, operation_id: str, error: Exception) -> bool:
    """Read shared-save outcome state without adding Telegram attempt accounting."""
    explicit = getattr(error, "telegram_stage_retryable", None)
    if isinstance(explicit, bool):
        return explicit
    if client is not None:
        try:
            from core.integrations.firestore.operations import get_operation, get_stage
            operation = get_operation(client, operation_id)
            if operation is not None:
                stage = get_stage(client, operation_id, operation.stage) if operation.stage else None
                if operation.error is not None:
                    return bool(operation.error.retryable and (operation.error.failures or 0) < 3)
                if stage is not None and stage.last_error is not None:
                    return bool(stage.last_error.retryable and stage.failures < 3)
                if operation.state.value == "failed":
                    return False
        except Exception:
            pass
    return not isinstance(error, (ValueError, TypeError, KeyError))
