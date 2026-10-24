"""Telegram audio download, transcription, and analysis orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from google.genai import types

from core.integrations.gemini.client import make_client
from core.integrations.gemini.transcription import transcribe_part
from core.operations.models import OperationError
from core.operations.retries import (
    ConfirmedStageFailure,
    ConfirmedStageSuccess,
    UnknownOutcome,
)


logger = logging.getLogger(__name__)

_SUPPORTED_AUDIO_MIME_TYPES = frozenset({
    "audio/aac",
    "audio/aiff",
    "audio/alaw",
    "audio/flac",
    "audio/l16",
    "audio/m4a",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/webm",
    "audio/mulaw",
    "audio/x-m4a",
    "audio/x-wav",
})
_DEFAULT_MAX_AUDIO_BYTES = 20 * 1024 * 1024
_DEFAULT_INLINE_REQUEST_LIMIT_BYTES = 18 * 1024 * 1024
_MAX_CONCURRENT_TRANSCRIPTIONS = 2


class TelegramAudioInputError(ValueError):
    """Raised when a Telegram audio item is malformed or unsupported."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _wire_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return value


def _request_size(part: types.Part, config: types.GenerateContentConfig) -> int:
    body = {
        "contents": _wire_model(types.UserContent(parts=[part])),
        "generationConfig": _wire_model(config),
    }
    return len(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _with_transcription(message: Any, transcription: str) -> Any:
    if isinstance(message, Mapping):
        result = dict(message)
        result["transcription"] = transcription
        return result
    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"transcription": transcription})
    copy = getattr(message, "_replace", None)
    if callable(copy):
        return copy(transcription=transcription)
    raise TelegramAudioInputError("Unsupported Telegram message record")


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class TelegramAudioTranscriber:
    """Transcribe audio messages, checkpoint each source, then analyze once."""

    def __init__(
        self,
        *,
        telegram_client: Any,
        repository: Any,
        gemini_client_factory: Callable[..., Any] = make_client,
        context_id: str | None = None,
        max_audio_bytes: int = _DEFAULT_MAX_AUDIO_BYTES,
        inline_request_limit_bytes: int = _DEFAULT_INLINE_REQUEST_LIMIT_BYTES,
    ) -> None:
        if max_audio_bytes <= 0 or inline_request_limit_bytes <= 0:
            raise ValueError("Audio size limits must be positive")
        self.telegram_client = telegram_client
        self.repository = repository
        self.gemini_client_factory = gemini_client_factory
        self.context_id = context_id
        self.max_audio_bytes = max_audio_bytes
        self.inline_request_limit_bytes = inline_request_limit_bytes

    async def process_messages(
        self,
        messages: Sequence[Any],
        *,
        draft_id: str,
        analyze: Callable[[list[Any]], Any],
        retry_governor: Any = None,
        operation_id: str | None = None,
        operation_lease: Any = None,
    ) -> Any:
        ordered = sorted(messages, key=lambda item: _get(item, "intake_sequence", 0))
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TRANSCRIPTIONS)

        async def run_one(message: Any) -> tuple[int, Any]:
            async with semaphore:
                message_id = _get(message, "message_id")
                state = _get(message, "transcription_state")
                state = getattr(state, "value", state)
                saved_text = _get(message, "transcription_text")
                if state == "completed" and isinstance(saved_text, str):
                    saved_text = await self._reconcile_completed_transcript(
                        message_id,
                        saved_text,
                        retry_governor=retry_governor,
                        operation_id=operation_id,
                        operation_lease=operation_lease,
                    )
                    return message_id, _with_transcription(message, saved_text)
                try:
                    transcript = await self._transcribe_with_checkpoint_policy(
                        message,
                        draft_id,
                        message_id,
                        retry_governor=retry_governor,
                        operation_id=operation_id,
                        operation_lease=operation_lease,
                    )
                    return message_id, _with_transcription(message, transcript)
                except Exception as error:
                    await self._checkpoint_terminal_failure(
                        draft_id,
                        message_id,
                        error,
                    )
                    raise

        outcomes = await asyncio.gather(*(run_one(message) for message in ordered), return_exceptions=True)
        failure = next((item for item in outcomes if isinstance(item, BaseException)), None)
        if failure is not None:
            raise failure

        rows = [row for _message_id, row in outcomes]
        result = analyze(rows)
        return await result if hasattr(result, "__await__") else result

    async def _reconcile_completed_transcript(
        self,
        message_id: int,
        saved_text: str,
        *,
        retry_governor: Any,
        operation_id: str | None,
        operation_lease: Any,
    ) -> str:
        """Reuse a completed checkpoint and reconcile a matching in-flight stage."""
        if retry_governor is not None and operation_id and operation_lease is not None:
            stage_key = f"telegram.audio.{message_id}"
            stage_record = retry_governor.get_stage(operation_id, stage_key)
            if stage_record.in_flight_attempt_id is not None:
                return await retry_governor.run_stage_async(
                    operation_id, stage_key, lambda: saved_text,
                    owner=operation_lease.owner, fence=operation_lease.fence,
                    reconcile=lambda: ConfirmedStageSuccess(saved_text),
                )
        return saved_text

    async def _transcribe_with_checkpoint_policy(
        self,
        message: Any,
        draft_id: str,
        message_id: int,
        *,
        retry_governor: Any,
        operation_id: str | None,
        operation_lease: Any,
    ) -> str:
        """Run transcription with governed retries when an operation lease exists."""
        if retry_governor is not None and operation_id and operation_lease is not None:
            return await self._transcribe_governed_stage(
                message,
                draft_id,
                message_id,
                retry_governor=retry_governor,
                operation_id=operation_id,
                operation_lease=operation_lease,
            )
        return await self._transcribe_and_checkpoint(message, draft_id, message_id)

    async def _transcribe_and_checkpoint(
        self,
        message: Any,
        draft_id: str,
        message_id: int,
    ) -> str:
        """Transcribe one audio message and persist its transcript checkpoint."""
        transcript = await self._transcribe_message(message)
        if self.context_id is None:
            await _await_if_needed(self.repository.checkpoint_transcription(
                draft_id, message_id, transcript,
            ))
        else:
            await _await_if_needed(self.repository.checkpoint_transcription(
                self.context_id, draft_id, message_id, transcript,
            ))
        return transcript

    async def _transcribe_governed_stage(
        self,
        message: Any,
        draft_id: str,
        message_id: int,
        *,
        retry_governor: Any,
        operation_id: str,
        operation_lease: Any,
    ) -> str:
        """Execute the transcription checkpoint as a retry-governed stage."""
        stage_key = f"telegram.audio.{message_id}"
        try:
            return await retry_governor.run_stage_async(
                operation_id,
                stage_key,
                lambda: self._transcribe_and_checkpoint(message, draft_id, message_id),
                owner=operation_lease.owner,
                fence=operation_lease.fence,
                retryable=lambda error: not isinstance(error, (ValueError, TypeError, KeyError)),
                reconcile=lambda: ConfirmedStageFailure(OperationError(
                    code="unknown_outcome",
                    category="execution",
                    message="Transcription outcome was not checkpointed",
                    retryable=True,
                    stage=stage_key,
                )),
            )
        except Exception as error:
            self._mark_governed_error_retryability(
                error,
                retry_governor=retry_governor,
                operation_id=operation_id,
                stage_key=stage_key,
            )
            raise

    @staticmethod
    def _mark_governed_error_retryability(
        error: Exception,
        *,
        retry_governor: Any,
        operation_id: str,
        stage_key: str,
    ) -> None:
        """Mark whether a governed failure can retry before terminal checkpointing."""
        if isinstance(error, UnknownOutcome):
            error.telegram_stage_retryable = True
            return
        try:
            stage_record = retry_governor.get_stage(operation_id, stage_key)
            last_error = stage_record.last_error
            error.telegram_stage_retryable = (
                True if last_error is None else bool(
                    last_error.retryable
                    and stage_record.failures < retry_governor.max_failures
                )
            )
        except Exception:
            error.telegram_stage_retryable = True

    async def _checkpoint_terminal_failure(
        self,
        draft_id: str,
        message_id: int,
        error: Exception,
    ) -> None:
        """Persist a terminal transcription failure while suppressing retryable ones."""
        failure_checkpoint = getattr(
            self.repository, "checkpoint_transcription_failure", None,
        )
        if (self.context_id is None or not callable(failure_checkpoint)
                or getattr(error, "telegram_stage_retryable", False)):
            return
        error_code = (
            "unsupported_audio" if isinstance(error, TelegramAudioInputError)
            else "download_failed" if isinstance(error, FileNotFoundError)
            else "transcription_failed"
        )
        try:
            await _await_if_needed(failure_checkpoint(
                self.context_id, draft_id, message_id, error_code,
            ))
        except Exception:
            logger.warning("Telegram transcription failure checkpoint failed")

    async def _transcribe_message(self, message: Any) -> str:
        file_id = _get(message, "audio_file_id")
        if not isinstance(file_id, str) or not file_id:
            raise TelegramAudioInputError("Audio message has no Telegram file ID")

        download = None
        client = None
        remote_name = None
        temporary_file = None
        try:
            download = await self.telegram_client.download_audio(file_id)
            data = _get(download, "data")
            mime_type = _get(download, "mime_type")
            declared_size = _get(download, "size")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TelegramAudioInputError("Telegram audio payload is malformed")
            data = bytes(data)
            if declared_size is not None and (
                isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                raise TelegramAudioInputError("Telegram audio size is malformed")
            if declared_size is not None and declared_size > self.max_audio_bytes:
                raise TelegramAudioInputError("Telegram audio exceeds the supported size")
            if len(data) > self.max_audio_bytes:
                raise TelegramAudioInputError("Telegram audio exceeds the supported size")
            if not data:
                raise TelegramAudioInputError("Telegram audio is empty")
            if not isinstance(mime_type, str) or mime_type.lower() not in _SUPPORTED_AUDIO_MIME_TYPES:
                raise TelegramAudioInputError("Telegram audio MIME type is unsupported")

            client = self.gemini_client_factory(single_attempt=True)
            config = _transcription_config()
            inline_part = types.Part(inline_data=types.Blob(data=data, mime_type=mime_type.lower()))
            if _request_size(inline_part, config) <= self.inline_request_limit_bytes:
                part = inline_part
            else:
                temporary_file = tempfile.TemporaryFile(mode="w+b")
                temporary_file.write(data)
                temporary_file.seek(0)
                uploaded = await client.aio.files.upload(
                    file=temporary_file,
                    config=types.UploadFileConfig(mime_type=mime_type.lower()),
                )
                remote_name = _get(uploaded, "name")
                uri = _get(uploaded, "uri")
                if not isinstance(remote_name, str) or not remote_name:
                    remote_name = None
                if remote_name is None or not isinstance(uri, str) or not uri:
                    raise RuntimeError("Gemini Files upload returned an invalid resource")
                part = types.Part(file_data=types.FileData(file_uri=uri, mime_type=mime_type.lower()))

            return await transcribe_part(client, part)
        finally:
            if temporary_file is not None:
                temporary_file.close()
            if client is not None and remote_name:
                try:
                    await client.aio.files.delete(name=remote_name)
                except Exception:
                    logger.warning("Gemini temporary audio cleanup failed")
            if download is not None:
                try:
                    await self.telegram_client.cleanup_download(download)
                except Exception:
                    logger.warning("Telegram temporary audio cleanup failed")


def _transcription_config() -> types.GenerateContentConfig:
    from core.integrations.gemini.transcription import transcription_config

    return transcription_config()
