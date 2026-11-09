"""Audio transcription boundary contracts for the Telegram connector.

The future-facing public API used here is
``TelegramAudioTranscriber.process_messages(messages, *, draft_id, analyze)``.
Messages are small synthetic records with ``message_id``, ``intake_sequence``
and ``audio_file_id`` fields. Results are associated with their source message
and delivered to analysis in intake order; no Firestore document schema is
prescribed by these tests.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib
import json
import logging
import os
import re
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.integrations.firestore.operation_queue import (
    QueueOperationRetryStore, claim_head, register_operation,
)
from core.operations.retries import (
    RetryBudgetExceeded, RetryGovernor, UnknownOutcome, ReconciledStageFailure,
)
from services.connectors.telegram.repository import SESSIONS


pytestmark = pytest.mark.offline

MODEL = "gemini-3.5-transcribe"
BOT_TOKEN = "synthetic-telegram-bot-token"
TEMP_URL = f"https://files.telegram.test/file/bot{BOT_TOKEN}/synthetic.ogg"
AUDIO_BYTES = b"synthetic-private-audio-bytes"


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _service(telegram, repository, gemini_factory, **kwargs):
    cls = importlib.import_module(
        "services.connectors.telegram.audio"
    ).TelegramAudioTranscriber
    return cls(
        telegram_client=telegram,
        repository=repository,
        gemini_client_factory=gemini_factory,
        **kwargs,
    )


def _message(message_id, *, intake_sequence=None, file_id=None):
    return {
        "message_id": message_id,
        "intake_sequence": message_id if intake_sequence is None else intake_sequence,
        "audio_file_id": file_id or f"file-{message_id}",
    }


def _download(data=AUDIO_BYTES, *, mime_type="audio/ogg", size=None):
    return {
        "data": data,
        "mime_type": mime_type,
        "size": len(data) if size is None else size,
        "temporary_url": TEMP_URL,
    }


def _camel_case(name):
    return re.sub(r"_([a-z])", lambda match: match.group(1).upper(), name)


def _wire_value(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {_camel_case(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def _wire_request(call):
    from google.genai import types

    return {
        "model": call["model"],
        "contents": _wire_value(types.UserContent(parts=[call["contents"]])),
        "generationConfig": _wire_value(call["config"]),
    }


def _wire_parts(contents):
    if isinstance(contents, dict):
        if any(key in contents for key in ("inlineData", "fileData", "text")):
            return [contents]
        if "parts" in contents:
            return _wire_parts(contents["parts"])
        return [part for value in contents.values() for part in _wire_parts(value)]
    if isinstance(contents, (list, tuple)):
        return [part for value in contents for part in _wire_parts(value)]
    return []


def _serialized_request(call):
    request = _wire_request(call)
    # The model is selected in the request URL; this counts the serialized body.
    body = {"contents": request["contents"], "generationConfig": request["generationConfig"]}
    return json.dumps(body, separators=(",", ":")).encode()


class _Telegram:
    def __init__(self, downloads):
        self.downloads = downloads
        self.requested = []
        self.cleaned = []

    async def download_audio(self, file_id):
        self.requested.append(file_id)
        item = self.downloads[file_id]
        if isinstance(item, BaseException):
            raise item
        return item

    async def cleanup_download(self, item):
        self.cleaned.append(item)


class _Repository:
    def __init__(self, source_messages=None):
        self.draft = {
            "status": "processing",
            "messages": source_messages or [],
            "source_manifest": [
                {"message_id": item["message_id"], "source": item["text"]}
                for item in (source_messages or [])
            ],
        }
        self.checkpoints = {}

    async def checkpoint_transcription(self, draft_id, message_id, text):
        self.checkpoints[message_id] = text

    async def load_draft(self, draft_id):
        return self.draft


class _Models:
    def __init__(self, transcriptions=None, *, delay_by_audio=None):
        self.transcriptions = transcriptions or {}
        self.delay_by_audio = delay_by_audio or {}
        self.file_payloads = {}
        self.calls = []
        self.started_markers = []
        self.finished_markers = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    @staticmethod
    def _find_payload(value):
        if isinstance(value, dict):
            inline = value.get("inline_data", value.get("inlineData"))
            if inline is not None:
                data = inline.get("data") if isinstance(inline, dict) else inline.data
                if isinstance(data, str):
                    try:
                        data = base64.b64decode(data)
                    except Exception:
                        pass
                return data
            file_data = value.get("file_data", value.get("fileData"))
            if file_data is not None:
                return (file_data.get("file_uri", file_data.get("fileUri"))
                        if isinstance(file_data, dict) else file_data.file_uri)
            return next((found for child in value.values()
                         if (found := _Models._find_payload(child)) is not None), None)
        if isinstance(value, (list, tuple)):
            return next((found for child in value
                         if (found := _Models._find_payload(child)) is not None), None)
        if hasattr(value, "inline_data") and value.inline_data is not None:
            data = value.inline_data.data
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data)
                except Exception:
                    pass
            return data
        if hasattr(value, "file_data") and value.file_data is not None:
            return value.file_data.file_uri
        if hasattr(value, "parts"):
            return _Models._find_payload(value.parts)
        return None

    async def generate_content(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            marker = self._find_payload(kwargs["contents"])
            self.calls.append(kwargs)
            self.started_markers.append(marker)
            delay = self.delay_by_audio.get(marker, 0)
            if delay:
                await asyncio.sleep(delay)
            resolved_marker = self.file_payloads.get(marker, marker)
            text = self.transcriptions[resolved_marker]
            self.finished_markers.append(marker)
            return SimpleNamespace(text=text)
        finally:
            with self._lock:
                self.active -= 1


class _GeminiFactory:
    def __init__(self, models, files=None):
        self.models = models
        self.files = files or _Files()
        self.options = []
        self.clients = []

    def __call__(self, **kwargs):
        self.options.append(kwargs)
        self.models.file_payloads = self.files.payload_by_uri
        client = SimpleNamespace(aio=SimpleNamespace(models=self.models, files=self.files))
        self.clients.append(client)
        return client


class _Files:
    def __init__(self):
        self.uploaded = []
        self.deleted = []
        self.fail_delete = False
        self.payload_by_uri = {}
        self.upload_sources = []

    async def upload(self, *, file, config=None):
        if hasattr(file, "read"):
            position = file.tell() if hasattr(file, "tell") else None
            data = file.read()
            if position is not None and hasattr(file, "seek"):
                file.seek(position)
        elif isinstance(file, Path):
            data = file.read_bytes()
        else:
            data = Path(file).read_bytes()
        mime_type = (
            config.get("mime_type", config.get("mimeType")) if isinstance(config, dict)
            else getattr(config, "mime_type", getattr(config, "mimeType", None))
        ) if config is not None else None
        index = len(self.uploaded) + 1
        file_ref = SimpleNamespace(name=f"files/resource-{index}", uri=f"https://gemini.test/files/{index}")
        self.uploaded.append((data, mime_type, file_ref))
        self.payload_by_uri[file_ref.uri] = data
        self.upload_sources.append(file)
        return file_ref

    async def delete(self, *, name, config=None):
        self.deleted.append(name)
        if self.fail_delete:
            raise RuntimeError("synthetic cleanup error")


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _selector_loop():
    """Use loopback datagrams for Windows asyncio self-pipe creation in the sandbox."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    old_socketpair = socket.socketpair
    local_pairs = []

    def local_datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            local_pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_datagram_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = old_socketpair


def _config_wire(config):
    return _wire_value(config)


def test_telegram_audio_uses_transcribe_verbatim_auto_detection_and_one_attempt():
    models = _Models({AUDIO_BYTES: "synthetic English transcript"})
    factory = _GeminiFactory(models)
    telegram = _Telegram({"file-1": _download()})
    repository = _Repository()
    service = _service(telegram, repository, factory)

    _run(service.process_messages([_message(1)], draft_id="draft-1", analyze=lambda rows: rows))

    assert factory.options == [{"single_attempt": True}]
    request = models.calls[0]
    assert request["model"] == MODEL
    assert set(request) == {"model", "contents", "config"}
    wire_contents = _wire_value(request["contents"])
    parts = _wire_parts(wire_contents)
    assert len(parts) == 1
    assert set(parts[0]) == {"inlineData"}
    assert parts[0]["inlineData"]["data"] == base64.b64encode(AUDIO_BYTES).decode("ascii")
    assert parts[0]["inlineData"]["mimeType"] == "audio/ogg"
    config = _config_wire(request["config"])
    audio_config = config.get("audioTranscriptionConfig", config.get("audio_transcription_config"))
    assert audio_config["mode"] == "VERBATIM"
    assert audio_config.get("languageCodes", audio_config.get("language_codes", [])) == []
    assert set(audio_config) <= {"mode", "languageCodes", "language_codes"}
    assert set(config) <= {"audioTranscriptionConfig", "audio_transcription_config"}
    assert "thinkingConfig" not in config and "thinking_config" not in config
    assert not ({"temperature", "topP", "topK", "top_p", "top_k"} & set(config))


def test_two_concurrent_transcriptions_keep_message_checkpoints_and_intake_order():
    audios = [b"audio-one", b"audio-two", b"audio-three", b"audio-four"]
    message_ids = [10, 20, 30, 40]
    transcripts = ["transcript one", "transcript two", "transcript three", "transcript four"]
    models = _Models(
        dict(zip(audios, transcripts)),
        delay_by_audio={audios[0]: 0.08, audios[1]: 0.001, audios[2]: 0.02, audios[3]: 0.001},
    )
    factory = _GeminiFactory(models)
    telegram = _Telegram({f"file-{mid}": _download(data) for mid, data in zip(message_ids, audios)})
    repository = _Repository()
    analyzed = []
    checkpoints_when_analyzed = []
    analyze_calls = 0

    async def analyze(rows):
        nonlocal analyze_calls
        analyze_calls += 1
        checkpoints_when_analyzed.append(dict(repository.checkpoints))
        analyzed.extend(rows)
        return rows

    result = _run(_service(telegram, repository, factory).process_messages(
        [
            _message(30, intake_sequence=3),
            _message(10, intake_sequence=1),
            _message(40, intake_sequence=4),
            _message(20, intake_sequence=2),
        ],
        draft_id="draft-2",
        analyze=analyze,
    ))

    assert models.max_active == 2
    expected = dict(zip(message_ids, transcripts))
    assert repository.checkpoints == expected
    assert analyze_calls == 1
    assert checkpoints_when_analyzed == [expected]
    assert len(models.finished_markers) == 4
    assert set(models.finished_markers) == set(audios)
    assert models.finished_markers != models.started_markers
    assert [_field(row, "message_id") for row in analyzed] == [10, 20, 30, 40]
    assert [_field(row, "transcription") for row in analyzed] == transcripts
    assert result is not None


@pytest.mark.parametrize(
    ("download", "limit", "error"),
    [
        (_download(size=101), 100, "size"),
        (_download(mime_type="application/octet-stream"), 1000, "MIME"),
        (FileNotFoundError("synthetic Telegram file unavailable"), 1000, "unavailable"),
    ],
)
def test_invalid_audio_stops_before_analysis_and_keeps_draft_recoverable(
    download, limit, error,
):
    telegram = _Telegram({"file-1": download})
    source_message = {"message_id": 1, "intake_sequence": 1, "text": "sealed source text", "audio_file_id": "file-1"}
    repository = _Repository([source_message])
    original_source = copy.deepcopy((repository.draft["messages"], repository.draft["source_manifest"]))
    factory = _GeminiFactory(_Models({AUDIO_BYTES: "must not be used"}))
    analyzed = []
    service = _service(telegram, repository, factory, max_audio_bytes=limit)

    with pytest.raises((ValueError, FileNotFoundError), match=error):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-recoverable", analyze=lambda rows: analyzed.append(rows)
        ))

    assert analyzed == []
    assert (repository.draft["messages"], repository.draft["source_manifest"]) == original_source
    assert telegram.requested == ["file-1"]
    assert factory.models.calls == []


def test_inline_or_files_api_uses_serialized_wire_body_at_limit_boundaries():
    audio = b"tiny-audio"
    probe_models = _Models({audio: "ok"})
    _run(_service(
        _Telegram({"file-1": _download(audio)}), _Repository(), _GeminiFactory(probe_models),
        inline_request_limit_bytes=10_000,
    ).process_messages([_message(1)], draft_id="draft-probe", analyze=lambda rows: rows))
    body_size = len(_serialized_request(probe_models.calls[0]))
    body = _wire_request(probe_models.calls[0])
    probe_part = _wire_parts(body["contents"])[0]
    assert probe_part["inlineData"]["data"] == base64.b64encode(audio).decode()
    assert probe_part["inlineData"]["mimeType"] == "audio/ogg"
    assert body["generationConfig"]["audioTranscriptionConfig"]["mode"] == "VERBATIM"

    for name, limit, expects_inline in (
        ("below", body_size - 1, False),
        ("at", body_size, True),
        ("above", body_size + 1, True),
    ):
        models = _Models({audio: "ok"})
        files = _Files()
        _run(_service(
            _Telegram({"file-1": _download(audio)}), _Repository(),
            _GeminiFactory(models, files), inline_request_limit_bytes=limit,
        ).process_messages([_message(1)], draft_id=f"draft-{name}", analyze=lambda rows: rows))
        request = _wire_request(models.calls[0])
        part = _wire_parts(request["contents"])[0]
        if expects_inline:
            assert "inlineData" in part
            assert part["inlineData"]["data"] == base64.b64encode(audio).decode()
            assert files.uploaded == []
        else:
            assert "fileData" in part
            uploaded_ref = files.uploaded[0][2]
            assert part["fileData"]["fileUri"] == uploaded_ref.uri
            assert files.deleted == [uploaded_ref.name]


@pytest.mark.parametrize("fail_generation", [False, True])
def test_temporary_download_and_gemini_file_are_cleaned_after_success_or_error(fail_generation):
    files = _Files()
    models = _Models({AUDIO_BYTES: "ok"})
    telegram = _Telegram({"file-1": _download()})
    if fail_generation:
        async def fail(**kwargs):
            raise RuntimeError("synthetic generation error")
        models.generate_content = fail
    factory = _GeminiFactory(models, files)
    # A tiny limit forces the temporary Files API path.
    service = _service(
        telegram, _Repository(), factory,
        inline_request_limit_bytes=10,
    )
    call = service.process_messages(
        [_message(1)], draft_id="draft-cleanup", analyze=lambda rows: rows,
    )

    if fail_generation:
        with pytest.raises(RuntimeError, match="synthetic generation error"):
            _run(call)
    else:
        _run(call)

    assert len(factory.files.uploaded) == 1
    assert factory.files.deleted == [factory.files.uploaded[0][2].name]
    assert len(telegram.cleaned) == 1
    for source in factory.files.upload_sources:
        if hasattr(source, "closed"):
            assert source.closed
        elif isinstance(source, (str, Path)):
            assert not Path(source).exists()


def test_parent_loss_stops_transcription_but_deletes_owned_gemini_upload():
    files = _Files()
    models = _Models({AUDIO_BYTES: "must not be generated"})
    telegram = _Telegram({"file-1": _download()})
    factory = _GeminiFactory(models, files)
    checks = 0

    def assert_current():
        nonlocal checks
        checks += 1
        if checks == 5:
            raise PermissionError("synthetic stale parent lease")

    service = _service(
        telegram, _Repository(), factory, inline_request_limit_bytes=1,
        assert_current=assert_current,
    )
    with pytest.raises(PermissionError, match="stale parent"):
        _run(service._transcribe_message(_message(1)))

    assert len(files.uploaded) == 1
    assert files.deleted == [files.uploaded[0][2].name]
    assert models.calls == []
    assert len(telegram.cleaned) == 1


def test_audio_boundary_logs_exclude_tokens_urls_audio_and_transcript(caplog):
    transcript_secret = "synthetic private transcript"
    models = _Models({AUDIO_BYTES: transcript_secret})
    factory = _GeminiFactory(models)
    telegram = _Telegram({"file-1": _download()})
    service = _service(telegram, _Repository(), factory)
    caplog.set_level(logging.DEBUG)

    _run(service.process_messages(
        [_message(1)], draft_id="draft-log", analyze=lambda rows: rows,
    ))

    traces = caplog.text
    for secret in (
        BOT_TOKEN, TEMP_URL, AUDIO_BYTES.decode(),
        base64.b64encode(AUDIO_BYTES).decode(), transcript_secret,
    ):
        assert secret not in traces


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        (b"english", "English transcript: hello"),
        (b"mixed", "\u0422\u0430\u043a, I, I mean\u2014\u043d\u0443, we should check, check the draft."),
        (b"russian", "Russian transcript: \u043f\u0440\u0438\u0432\u0435\u0442"),
    ],
)
def test_transcript_text_is_passed_through_unchanged(audio, expected):
    """Synthetic text round-trips unchanged; this does not test provider quality."""
    models = _Models({audio: expected})
    telegram = _Telegram({"file-1": _download(audio)})
    repository = _Repository()
    analyzed = []
    service = _service(telegram, repository, _GeminiFactory(models))

    _run(service.process_messages(
        [_message(1)], draft_id="draft-language", analyze=lambda rows: analyzed.extend(rows)
    ))

    assert repository.checkpoints[1] == expected
    assert _field(analyzed[0], "transcription") == expected


@pytest.mark.parametrize(
    "mime_type",
    ["audio/aiff", "audio/aac", "audio/opus", "audio/l16", "audio/alaw", "audio/mulaw"],
)
def test_alternate_gemini_supported_mime_types_are_accepted(mime_type):
    models = _Models({AUDIO_BYTES: "ok"})
    telegram = _Telegram({"file-1": _download(mime_type=mime_type)})
    _run(_service(telegram, _Repository(), _GeminiFactory(models)).process_messages(
        [_message(1)], draft_id="draft-mime", analyze=lambda rows: rows,
    ))
    assert len(models.calls) == 1


def test_default_audio_size_limit_matches_telegram_download_ceiling():
    too_large = b"x" * (20 * 1024 * 1024 + 1)
    telegram = _Telegram({"file-1": _download(too_large)})
    models = _Models()
    with pytest.raises(ValueError, match="size"):
        _run(_service(telegram, _Repository(), _GeminiFactory(models)).process_messages(
            [_message(1)], draft_id="draft-size", analyze=lambda rows: rows,
        ))
    assert models.calls == []


def test_completed_transcription_resumes_without_redownloading_or_regenerating():
    message = {
        **_message(1),
        "transcription_state": "completed",
        "transcription_text": "already checkpointed verbatim",
    }
    telegram = _Telegram({})
    models = _Models()
    repository = _Repository()
    analyzed = []

    result = _run(_service(telegram, repository, _GeminiFactory(models)).process_messages(
        [message], draft_id="draft-resume",
        analyze=lambda rows: (analyzed.extend(rows), rows)[1],
    ))

    assert telegram.requested == []
    assert models.calls == []
    assert repository.checkpoints == {}
    assert _field(analyzed[0], "transcription") == "already checkpointed verbatim"
    assert result is not None


def test_durable_audio_stage_reuses_completed_message_transcript_without_second_provider_call(
    firestore_fake,
):
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("audio-context")
    now = datetime.now(timezone.utc)
    register_operation(client, context_ref, "audio-operation", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="audio-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(),
        retry_store=QueueOperationRetryStore(client, context_ref),
    )
    repository = _Repository()
    models = _Models({AUDIO_BYTES: "durably checkpointed transcript"})
    telegram = _Telegram({"file-1": _download()})
    service = _service(telegram, repository, _GeminiFactory(models))
    results = []

    async def analyze(rows):
        results.append(_field(rows[0], "transcription"))
        return rows

    def process():
        return _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=analyze,
            retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
        ))

    process()
    process()
    assert len(models.calls) == 1
    assert repository.checkpoints == {1: "durably checkpointed transcript"}
    assert results == ["durably checkpointed transcript"] * 2


def test_audio_transcript_checkpoint_reconciles_an_in_flight_stage_without_provider_call(
    firestore_fake,
):
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("audio-context")
    now = datetime.now(timezone.utc)
    register_operation(client, context_ref, "audio-operation", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="audio-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(), retry_store=QueueOperationRetryStore(client, context_ref),
    )
    with pytest.raises(UnknownOutcome):
        _run(governor.run_stage_async(
            "audio-operation", "telegram.audio.1",
            lambda: (_ for _ in ()).throw(UnknownOutcome("lost response")),
            owner=lease.owner, fence=lease.fence,
        ))

    message = {
        **_message(1),
        "transcription_state": "completed",
        "transcription_text": "checkpointed before worker loss",
    }
    models = _Models()
    service = _service(_Telegram({}), _Repository(), _GeminiFactory(models))
    analyzed = []
    _run(service.process_messages(
        [message], draft_id="draft-audio", analyze=lambda rows: analyzed.extend(rows),
        retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
    ))

    stage = governor.get_stage("audio-operation", "telegram.audio.1")
    assert getattr(stage.state, "value", stage.state) == "completed"
    assert models.calls == []
    assert _field(analyzed[0], "transcription") == "checkpointed before worker loss"


def test_unknown_audio_outcome_reconciles_as_retryable_failure_without_user_failure_checkpoint(
    firestore_fake,
):
    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("audio-context")
    now = datetime.now(timezone.utc)
    register_operation(client, context_ref, "audio-operation", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="audio-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    class Repository(_Repository):
        def __init__(self):
            super().__init__()
            self.failure_checkpoints = []
            self.context_checkpoints = []

        def checkpoint_transcription_failure(self, context_id, draft_id, message_id, error_code):
            self.failure_checkpoints.append((context_id, draft_id, message_id, error_code))

        async def checkpoint_transcription(self, context_id, draft_id, message_id, transcript):
            self.context_checkpoints.append((context_id, draft_id, message_id, transcript))

    repository = Repository()
    governor = RetryGovernor(
        client, clock=Clock(), retry_store=QueueOperationRetryStore(client, context_ref),
    )
    service = _service(_Telegram({}), repository, _GeminiFactory(_Models()), context_id="audio-context")
    provider_calls = []

    async def unknown(_message):
        provider_calls.append("unknown")
        raise UnknownOutcome("provider response was lost")

    service._transcribe_message = unknown
    with pytest.raises(UnknownOutcome):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
            retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
        ))
    assert repository.failure_checkpoints == []
    assert repository.context_checkpoints == []

    with pytest.raises(ReconciledStageFailure):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
            retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
        ))
    stage = governor.get_stage("audio-operation", "telegram.audio.1")
    assert stage.in_flight_attempt_id is None
    assert stage.failures == 1
    assert stage.last_error.retryable is True
    assert repository.failure_checkpoints == []

    async def recovered(_message):
        provider_calls.append("recovered")
        return "recovered transcript"

    service._transcribe_message = recovered
    _run(service.process_messages(
        [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
        retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
    ))
    assert provider_calls == ["unknown", "recovered"]
    assert repository.failure_checkpoints == []
    assert repository.context_checkpoints == [
        ("audio-context", "draft-audio", 1, "recovered transcript"),
    ]
    assert governor.get_stage("audio-operation", "telegram.audio.1").failures == 1


def test_audio_gemini_402_is_terminal_and_does_not_repeat_provider_call(firestore_fake):
    from google.genai.errors import ClientError
    from services.connectors.telegram.runtime.save_fencing import telegram_error_classifier

    client = firestore_fake.client_class()
    context_ref = client.collection(SESSIONS).document("audio-context")
    now = datetime.now(timezone.utc)
    register_operation(client, context_ref, "audio-operation", identity_digest="d", input_ref="i", now=now)
    lease = claim_head(client, context_ref, owner="audio-worker", now=now)
    assert lease is not None

    class Clock:
        def now(self):
            return now

    governor = RetryGovernor(
        client, clock=Clock(),
        retry_store=QueueOperationRetryStore(client, context_ref),
        error_classifier=telegram_error_classifier,
    )
    service = _service(
        _Telegram({}), _Repository(), _GeminiFactory(_Models()), context_id="audio-context",
    )
    provider_calls = []
    error = ClientError(402, {"error": {
        "code": 402, "status": "RESOURCE_EXHAUSTED", "message": "private-marker",
    }})

    async def fail(_message):
        provider_calls.append(1)
        raise error

    service._transcribe_message = fail
    with pytest.raises(ClientError):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
            retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
        ))
    with pytest.raises(RetryBudgetExceeded):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
            retry_governor=governor, operation_id="audio-operation", operation_lease=lease,
        ))

    stage = governor.get_stage("audio-operation", "telegram.audio.1")
    assert provider_calls == [1]
    assert stage.failures == 1
    assert stage.last_error.category == "billing"
    assert stage.last_error.service == "gemini"
    assert stage.last_error.retryable is False
    assert "private-marker" not in repr(stage.last_error)


def test_audio_persistence_error_preserves_checkpoint_without_failure_checkpoint():
    class Repository(_Repository):
        def __init__(self):
            super().__init__()
            self.failure_checkpoints = []
            self.context_checkpoints = []

        def checkpoint_transcription_failure(self, context_id, draft_id, message_id, error_code):
            self.failure_checkpoints.append((context_id, draft_id, message_id, error_code))

        async def checkpoint_transcription(self, context_id, draft_id, message_id, transcript):
            self.context_checkpoints.append((context_id, draft_id, message_id, transcript))

    repository = Repository()

    class Governor:
        max_failures = 3

        async def run_stage_async(self, _operation_id, _stage_key, callback, **_kwargs):
            await callback()
            raise OSError("stage result persistence failed")

        def get_stage(self, _operation_id, _stage_key):
            return SimpleNamespace(last_error=None, failures=0, in_flight_attempt_id="attempt-1")

    service = _service(
        _Telegram({}), repository, _GeminiFactory(_Models()), context_id="audio-context",
    )
    provider_calls = []

    async def transcribe(_message):
        provider_calls.append(1)
        return "transcript already checkpointed"

    service._transcribe_message = transcribe
    with pytest.raises(OSError):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-audio", analyze=lambda rows: rows,
            retry_governor=Governor(), operation_id="audio-operation",
            operation_lease=SimpleNamespace(owner="audio-worker", fence=1),
        ))

    assert provider_calls == [1]
    assert repository.context_checkpoints == [
        ("audio-context", "draft-audio", 1, "transcript already checkpointed"),
    ]
    assert repository.failure_checkpoints == []


def test_malformed_gemini_file_reference_fails_and_deletes_known_resource():
    class MalformedFiles(_Files):
        async def upload(self, *, file, config=None):
            self.uploaded.append((b"synthetic", "audio/ogg", SimpleNamespace(name="files/known", uri=None)))
            return self.uploaded[-1][2]

    files = MalformedFiles()
    models = _Models()
    service = _service(
        _Telegram({"file-1": _download()}), _Repository(), _GeminiFactory(models, files),
        inline_request_limit_bytes=1,
    )

    with pytest.raises(RuntimeError, match="invalid resource"):
        _run(service.process_messages(
            [_message(1)], draft_id="draft-malformed-ref", analyze=lambda rows: rows,
        ))

    assert files.deleted == ["files/known"]
    assert models.calls == []


def test_context_checkpoint_supports_synchronous_repository_success_and_failure():
    class SyncRepository(_Repository):
        def __init__(self):
            super().__init__()
            self.calls = []

        def checkpoint_transcription(self, context_id, draft_id, message_id, text):
            self.calls.append(("completed", context_id, draft_id, message_id, text))

        def checkpoint_transcription_failure(self, context_id, draft_id, message_id, error_code):
            self.calls.append(("failed", context_id, draft_id, message_id, error_code))

    repository = SyncRepository()
    models = _Models({AUDIO_BYTES: "verbatim"})
    success = _service(
        _Telegram({"file-1": _download()}), repository, _GeminiFactory(models),
        context_id="42_5",
    )
    _run(success.process_messages(
        [_message(1)], draft_id="draft-context", analyze=lambda rows: rows,
    ))

    failure = _service(
        _Telegram({"file-2": FileNotFoundError("private provider detail")}),
        repository, _GeminiFactory(_Models()), context_id="42_5",
    )
    with pytest.raises(FileNotFoundError):
        _run(failure.process_messages(
            [_message(2, file_id="file-2")], draft_id="draft-context",
            analyze=lambda rows: rows,
        ))

    assert repository.calls == [
        ("completed", "42_5", "draft-context", 1, "verbatim"),
        ("failed", "42_5", "draft-context", 2, "download_failed"),
    ]
