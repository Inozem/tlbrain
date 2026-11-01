"""Telegram webhook application with lazy runtime configuration."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.utils.logging import configure_logging

from .terminal_notifications import (
    terminal_lifecycle_status as _terminal_lifecycle_status,
    update_terminal_notification as _update_terminal_notification,
)

configure_logging()
logger = logging.getLogger(__name__)
_SAFE_TECHNICAL_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_DEFAULT_RUNTIME = object()


def _exception_types(exc: BaseException, *, limit: int = 4) -> list[str]:
    """Return a bounded exception type chain without exposing exception text."""
    result: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(result) < limit and id(current) not in seen:
        seen.add(id(current))
        result.append(type(current).__name__[:80])
        current = current.__cause__ or current.__context__
    return result


def _log_webhook_failure(stage: str, exc: BaseException, update_id: int | None = None) -> None:
    details = {"event": "webhook_failure", "stage": stage,
               "error_type": type(exc).__name__[:80], "error_chain": _exception_types(exc)}
    if update_id is not None:
        details["update_id"] = update_id
    logger.error("Telegram webhook processing failed", extra=details)


def _firestore_secret() -> str | None:
    from core.integrations.firestore.client import get_client

    snapshot = get_client().collection("tokens").document("telegram_webhook").get()
    if not snapshot.exists:
        return None
    data = snapshot.to_dict() or {}
    value = data.get("secret") or data.get("webhook_secret")
    return value if isinstance(value, str) and value else None


def _allowed_user_ids() -> frozenset[int]:
    from core.config import get_connector_capabilities

    get_connector_capabilities()
    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
    return frozenset(int(part.strip()) for part in raw.split(",") if part.strip())


def _default_intake() -> Any:
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.intake import TelegramIntake
    from services.connectors.telegram.repository import TelegramRepository

    client = get_client()
    return TelegramIntake(
        repository=TelegramRepository(client=client),
        publisher=_DurableOutboxPublisher(client),
        allowed_user_ids=_allowed_user_ids(),
    )


class _DurableOutboxPublisher:
    """Publish one stored Telegram dispatch row through its configured OIDC target."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def publish(self, outbox_id: str) -> None:
        from core.integrations.cloud_tasks.client import enqueue_task
        from core.integrations.firestore.outbox import (
            OutboxState, OutboxTargetKind, get_outbox_record,
            record_outbox_delivery_result, task_payload,
        )
        record = get_outbox_record(self.client, outbox_id)
        if record is None or record.target_kind is not OutboxTargetKind.TELEGRAM_DISPATCH:
            raise RuntimeError("accepted Telegram outbox row is unavailable")
        if record.state is OutboxState.DELIVERED:
            return
        queue_name = os.environ.get("TELEGRAM_CONTROL_QUEUE")
        target_url = (os.environ.get("TELEGRAM_DELIVERY_URL") if record.notification_id
                      else os.environ.get("TELEGRAM_CONTROL_URL")) or os.environ.get(
                          "TELEGRAM_CONTROL_URL")
        expected_path = (
            "/internal/deliver" if record.notification_id and
            os.environ.get("TELEGRAM_DELIVERY_URL") else "/internal/dispatch"
        )
        target = _telegram_task_target(target_url, expected_path=expected_path)
        if not queue_name or target is None:
            raise RuntimeError("Telegram task target is not configured")
        now = datetime.now(timezone.utc)
        try:
            enqueue_task(
                queue_name=target.queue_name, url=target.url, task_id=record.task_name,
                body=task_payload(record), schedule_time=record.schedule_at,
                service_account_email=target.service_account_email,
                audience=target.audience, dispatch_deadline=target.dispatch_deadline,
            )
        except Exception:
            record_outbox_delivery_result(
                self.client, outbox_id, delivered=False, now=now,
                safe_error="cloud_tasks_enqueue_failed",
            )
            raise RuntimeError("Telegram task enqueue failed") from None
        result = record_outbox_delivery_result(self.client, outbox_id, delivered=True, now=now)
        if result is None:
            raise RuntimeError("Telegram outbox row disappeared during enqueue")


def _publish_head_wakeup(client: Any, outbox_id: str) -> None:
    _DurableOutboxPublisher(client).publish(outbox_id)


async def _dispatch_worker_for(app: FastAPI, record: Any) -> Any | JSONResponse:
    """Resolve the queue worker only after authenticated task and first-ack checks."""
    worker = app.state.telegram_queue_worker
    if worker is not None:
        return worker
    if not app.state.telegram_managed_runtime:
        return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)
    try:
        worker = await _offload_sync(_default_queue_worker)
        app.state.telegram_queue_worker = worker
        return worker
    except Exception as exc:
        logger.error(
            "Telegram runtime initialization failed",
            extra={"event": "telegram_runtime_failure", "stage": "initialization",
                   "error_type": type(exc).__name__[:80],
                   **_safe_operation_log_ids(record)},
        )
        return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)


async def _schedule_dispatch_status_checks(app: FastAPI, record: Any, publisher: Any) -> None:
    """Schedule lifecycle checks for an accepted operation before queue execution."""
    if not record.operation_id or not (
        app.state.telegram_managed_runtime or record.outbox_id.startswith("tgcommand-")
    ):
        return
    status_worker = app.state.telegram_status_check_worker
    if status_worker is None:
        status_worker = await _offload_sync(
            _status_check_runtime, publisher, app.state.telegram_managed_runtime,
        )
        app.state.telegram_status_check_worker = status_worker
    schedule = getattr(status_worker, "schedule_checks", None)
    if schedule is None:
        return
    accepted_at = (
        await _offload_sync(_accepted_operation_at, record.context_id, record.operation_id)
        if app.state.telegram_managed_runtime else None
    ) or record.created_at or record.schedule_at
    records = await _offload_sync(
        schedule, record.context_id, record.operation_id, accepted_at,
    )
    await _offload_sync(_enqueue_status_check_records, records)


async def _publish_ready_head_wakeup(app: FastAPI, record: Any) -> Any:
    """Publish a canonical queue continuation after managed worker execution."""
    if not app.state.telegram_managed_runtime:
        return None
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.queue import resolve_current_head_wakeup

    client = get_client()
    wakeup = await _offload_sync(resolve_current_head_wakeup, client, record.context_id)
    if wakeup is not None:
        await _offload_sync(_publish_head_wakeup, client, wakeup.outbox_id)
    return client


async def _publish_terminal_status(
    app: FastAPI, record: Any, publisher: Any, client: Any,
) -> JSONResponse | None:
    """Update and deliver terminal status only after managed pending work settles."""
    if not record.operation_id:
        return None
    managed_pending = (
        await _offload_sync(_managed_operation_pending, client, record)
        if app.state.telegram_managed_runtime else False
    )
    lifecycle_id = (
        None if managed_pending else await _offload_sync(_update_terminal_notification, record)
    )
    if not lifecycle_id:
        return None
    if publisher is None:
        publisher = await _offload_sync(_notification_runtime)
        app.state.telegram_notification_publisher = publisher
    delivery = await publisher.publish(lifecycle_id)
    if delivery not in {"delivered", "failed", "delivery_unknown", "not_found"}:
        return JSONResponse({"error": "Lifecycle delivery pending"}, status_code=503)
    return None


def _safe_operation_log_ids(record: Any) -> dict[str, str]:
    """Return only validated canonical identifiers for failure log context."""
    result: dict[str, str] = {}
    for field in ("operation_id", "context_id"):
        value = getattr(record, field, None)
        if isinstance(value, str) and _SAFE_TECHNICAL_ID.fullmatch(value):
            result[field] = value
    return result


_TASK_FIELDS = frozenset({
    "schema_version", "outbox_id", "target_kind", "generation", "operation_id",
    "context_id", "document_id", "save_intent_id", "notification_id",
})
_SYNC_POOL = ThreadPoolExecutor(thread_name_prefix="telegram-internal")


async def _offload_sync(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run blocking auth/storage SDK work off-loop without relying on self-pipe wakeups."""
    future = _SYNC_POOL.submit(function, *args, **kwargs)
    while not future.done():
        await asyncio.sleep(0.01)
    return future.result()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_task_record(payload: dict[str, Any], target: str) -> Any:
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.outbox import OutboxTargetKind, get_outbox_record, task_payload

    record = get_outbox_record(get_client(), payload["outbox_id"])
    if record is None or record.target_kind is not OutboxTargetKind(target):
        return None
    canonical = task_payload(record)
    return record if payload == canonical else None


def _managed_operation_pending(client: Any, record: Any) -> bool:
    """Defer lifecycle fallback while managed queue work still owns its final reply."""
    from services.connectors.telegram.repository import SESSIONS

    snapshot = (client.collection(SESSIONS).document(record.context_id)
                .collection("operations").document(record.operation_id).get())
    row = snapshot.to_dict() or {} if snapshot.exists else {}
    return row.get("state") not in {"completed", "failed"}


def _first_ack_state(record: Any) -> str | None:
    """Return the durable first-ack state for save commands; None means no draft barrier."""
    if not record.operation_id or not record.context_id:
        return None
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.operation_queue import get_operation
    from services.connectors.telegram.repository import NOTIFICATIONS, SESSIONS
    from services.connectors.telegram.repository._common import _document_ref

    client = get_client()
    context_ref = client.collection(SESSIONS).document(record.context_id)
    operation = get_operation(context_ref, record.operation_id)
    if operation is None or not operation.input_ref:
        return "unavailable"
    command = _document_ref(client, operation.input_ref).get()
    if not command.exists:
        return "unavailable"
    draft_id = (command.to_dict() or {}).get("draft_id")
    if not draft_id:
        return None
    rows = list(client.collection(NOTIFICATIONS).where("context_id", "==", record.context_id)
                .where("draft_id", "==", draft_id).where("kind", "==", "draft_ack").limit(1).stream())
    if not rows:
        return "unavailable"
    return (rows[0].to_dict() or {}).get("state", "unavailable")


def _accepted_operation_at(context_id: str, operation_id: str) -> Any:
    """Read the immutable queue acceptance time for managed head wakeups too."""
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.repository import SESSIONS

    operation_ref = (get_client().collection(SESSIONS).document(context_id)
                     .collection("operations").document(operation_id))
    snapshot = operation_ref.get()
    row = snapshot.to_dict() or {} if snapshot.exists else {}
    if (row.get("operation_id") != operation_id
            or row.get("context_id", context_id) != context_id):
        return None
    return row.get("created_at")


def _notification_runtime() -> Any:
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository import TelegramRepository

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("notification runtime is not configured")
    return TelegramNotificationPublisher(TelegramRepository(get_client()), TelegramClient(token))


def _default_queue_worker() -> Any:
    """Build the production queue processor only after an authenticated task arrives."""
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.runtime.factory import build_default_queue_worker

    return build_default_queue_worker(get_client())


def _status_check_runtime(notification_publisher: Any = None, managed_runtime: bool = False) -> Any:
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.status_checks import TelegramStatusChecks

    publisher = notification_publisher or _notification_runtime()
    return TelegramStatusChecks(
        get_client(), notification_publisher=publisher, managed_runtime=managed_runtime,
    )


def _telegram_task_target(callback_url: str | None, *, expected_path: str) -> Any:
    """Build a Cloud Tasks target only from deploy-matched Telegram OIDC config."""
    from urllib.parse import urlsplit
    from core.operations.outbox import TaskTargetConfig

    queue_name = os.environ.get("TELEGRAM_CONTROL_QUEUE")
    audience = os.environ.get("CLOUD_TASKS_OIDC_AUDIENCE") or os.environ.get("TASK_OIDC_AUDIENCE")
    service_account = (
        os.environ.get("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL")
        or os.environ.get("TASK_OIDC_SERVICE_ACCOUNT_EMAIL")
        or os.environ.get("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL")
        or os.environ.get("TASK_SERVICE_ACCOUNT_EMAIL")
    )
    expected_audience = os.environ.get("OIDC_AUDIENCE")
    expected_account = os.environ.get("OIDC_SERVICE_ACCOUNT_EMAIL")
    try:
        parsed = urlsplit(callback_url or "")
        valid_url = (
            parsed.scheme == "https" and bool(parsed.hostname)
            and parsed.username is None and parsed.password is None
            and parsed.path == expected_path
            and not parsed.query and not parsed.fragment
        )
        if not (
            valid_url and queue_name and queue_name.strip()
            and audience and audience.strip() and audience == expected_audience
            and service_account and service_account.strip()
            and service_account == expected_account
        ):
            return None
        return TaskTargetConfig(
            queue_name=queue_name,
            url=parsed.geturl(),
            service_account_email=service_account,
            audience=audience,
        )
    except (TypeError, ValueError):
        return None


def _telegram_status_check_target() -> Any:
    """Build the status-check callback only from deploy-matched Telegram OIDC config."""
    from urllib.parse import urlsplit

    target = _telegram_task_target(
        os.environ.get("TELEGRAM_CONTROL_URL"), expected_path="/internal/dispatch",
    )
    if target is None:
        return None
    parsed = urlsplit(target.url)
    return target.__class__(
        queue_name=target.queue_name,
        url=parsed._replace(path="/internal/status-check").geturl(),
        service_account_email=target.service_account_email,
        audience=target.audience,
        dispatch_deadline=target.dispatch_deadline,
    )


def _enqueue_status_check_records(records: Any) -> None:
    """Enqueue scheduled status rows directly; the outbox remains the retry source."""
    from core.integrations.cloud_tasks.client import enqueue_task
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.outbox import (
        OutboxState,
        record_outbox_delivery_result,
        task_payload,
    )

    target = _telegram_status_check_target()
    if target is None:
        raise RuntimeError("Telegram status check task target is not configured")
    client = get_client()
    now = datetime.now(timezone.utc)
    for record in records or ():
        if record.state is OutboxState.DELIVERED:
            continue
        try:
            enqueue_task(
                queue_name=target.queue_name,
                url=target.url,
                task_id=record.task_name,
                body=task_payload(record),
                schedule_time=record.schedule_at,
                service_account_email=target.service_account_email,
                audience=target.audience,
                dispatch_deadline=target.dispatch_deadline,
            )
        except Exception:
            try:
                record_outbox_delivery_result(
                    client, record.outbox_id, delivered=False, now=now,
                    safe_error="cloud_tasks_enqueue_failed",
                )
            except Exception:
                pass
            raise RuntimeError("Telegram status check task enqueue failed") from None
        result = record_outbox_delivery_result(
            client, record.outbox_id, delivered=True, now=now,
        )
        if result is None:
            raise RuntimeError("Telegram status check outbox row disappeared")


def create_app(
    *, intake: Any = None, webhook_secret: str | None = None,
    queue_worker: Any = _DEFAULT_RUNTIME, notification_publisher: Any = None,
    status_check_worker: Any = None,
    oidc_verifier: Any = None,
    telegram_client: Any = None,
    telegram_enabled: bool | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.telegram_intake = intake
    app.state.telegram_webhook_secret = webhook_secret
    app.state.telegram_managed_runtime = queue_worker is _DEFAULT_RUNTIME
    app.state.telegram_queue_worker = None if queue_worker is _DEFAULT_RUNTIME else queue_worker
    app.state.telegram_notification_publisher = notification_publisher
    app.state.telegram_status_check_worker = status_check_worker
    app.state.telegram_oidc_verifier = oidc_verifier
    app.state.telegram_client = telegram_client
    app.state.telegram_enabled = telegram_enabled

    async def authenticate(request: Request):
        authorization = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            verifier = app.state.telegram_oidc_verifier
            if verifier is None:
                from core.integrations.google_auth.oidc import verify_google_id_token
                verifier = verify_google_id_token
            identity = await _offload_sync(verifier, token.strip())
        except Exception as exc:
            from core.integrations.google_auth.oidc import OIDCConfigurationError
            if isinstance(exc, OIDCConfigurationError):
                return JSONResponse({"error": "Authentication unavailable"}, status_code=503)
            return JSONResponse({"error": "Authentication unavailable"}, status_code=503)
        if identity is None:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        if not identity.authorized:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        return None

    async def deployment_identity(request: Request):
        """Verify the same audience while authorizing only the dedicated deployer."""
        authorization = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            return None, JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            verifier = app.state.telegram_oidc_verifier
            if verifier is None:
                from core.integrations.google_auth.oidc import verify_google_id_token
                verifier = verify_google_id_token
            identity = await _offload_sync(verifier, token.strip())
        except Exception:
            return None, JSONResponse({"error": "Authentication unavailable"}, status_code=503)
        if identity is None:
            return None, JSONResponse({"error": "Unauthorized"}, status_code=401)
        deployer_email = os.environ.get("TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL", "")
        if not deployer_email or getattr(identity, "email", None) != deployer_email:
            return None, JSONResponse({"error": "Forbidden"}, status_code=403)
        return identity, None

    async def task_payload_from(request: Request):
        try:
            payload = json.loads(
                (await request.body()).decode("utf-8"), object_pairs_hook=_unique_json_object,
            )
        except Exception:
            return None, JSONResponse({"error": "Invalid task"}, status_code=400)
        if not isinstance(payload, dict) or set(payload) - _TASK_FIELDS:
            return None, JSONResponse({"error": "Invalid task"}, status_code=400)
        if (type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1
                or type(payload.get("generation")) is not int
                or payload["generation"] < 1 or not isinstance(payload.get("outbox_id"), str)
                or not payload["outbox_id"]):
            return None, JSONResponse({"error": "Invalid task"}, status_code=400)
        if any(not isinstance(value, str) or not value for key, value in payload.items()
               if key in _TASK_FIELDS - {"schema_version", "generation"}):
            return None, JSONResponse({"error": "Invalid task"}, status_code=400)
        return payload, None

    @app.get("/")
    async def health() -> dict[str, Any]:
        try:
            from core.config import get_connector_capabilities

            enabled = get_connector_capabilities().telegram_enabled
        except Exception as exc:
            _log_webhook_failure("capability_config", exc)
            enabled = False
        return {"status": "ok", "telegram_enabled": enabled}

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        enabled = app.state.telegram_enabled
        if enabled is False or (enabled is None and (
            not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_ALLOWED_USER_IDS")
        )):
            return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        secret = app.state.telegram_webhook_secret
        if secret is None:
            try:
                secret = await _offload_sync(_firestore_secret)
            except Exception as exc:
                _log_webhook_failure("secret_load", exc)
                return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not secret or not hmac.compare_digest(supplied, secret):
            logger.warning(
                "Telegram webhook request denied",
                extra={"event": "webhook_denied", "stage": "authorization"},
            )
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        logger.info("Telegram webhook request received", extra={"event": "webhook_receipt"})
        try:
            update = await request.json()
        except Exception:
            logger.warning(
                "Telegram webhook request malformed",
                extra={"event": "webhook_malformed", "stage": "json"},
            )
            return JSONResponse({"detail": "malformed JSON"}, status_code=400)
        if not isinstance(update, dict):
            logger.warning(
                "Telegram webhook request malformed",
                extra={"event": "webhook_malformed", "stage": "json_object"},
            )
            return JSONResponse({"detail": "JSON object required"}, status_code=400)
        raw_update_id = update.get("update_id")
        update_id = raw_update_id if type(raw_update_id) is int and raw_update_id >= 0 else None
        intake_instance = app.state.telegram_intake
        if intake_instance is None:
            try:
                intake_instance = await _offload_sync(_default_intake)
                app.state.telegram_intake = intake_instance
            except Exception as exc:
                _log_webhook_failure("intake_init", exc, update_id)
                return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        try:
            await _offload_sync(intake_instance.handle_update, update)
        except Exception as exc:
            _log_webhook_failure("handle_update", exc, update_id)
            return JSONResponse({"detail": "intake unavailable"}, status_code=503)
        details = {"event": "webhook_completion"}
        if update_id is not None:
            details["update_id"] = update_id
        logger.info("Telegram webhook request completed", extra=details)
        return {"ok": True}

    @app.post("/internal/disable-webhook")
    async def disable_webhook(request: Request):
        _identity, auth_error = await deployment_identity(request)
        if auth_error is not None:
            return auth_error
        client = app.state.telegram_client
        if client is None:
            try:
                from services.connectors.telegram.client import TelegramClient
                token = os.environ.get("TELEGRAM_BOT_TOKEN")
                if not token:
                    return JSONResponse({"error": "Webhook disable unavailable"}, status_code=503)
                client = TelegramClient(token)
            except Exception:
                return JSONResponse({"error": "Webhook disable unavailable"}, status_code=503)
        try:
            result = client.delete_webhook()
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict) or result.get("ok") is not True or result.get("result") is not True:
                raise RuntimeError("delete webhook was not confirmed")
        except Exception:
            return JSONResponse({"error": "Webhook disable failed"}, status_code=502)
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                try:
                    closed = close()
                    if asyncio.iscoroutine(closed):
                        await closed
                except Exception:
                    pass
        return {"status": "disabled"}

    @app.post("/internal/dispatch")
    async def dispatch(request: Request):
        auth_error = await authenticate(request)
        if auth_error is not None:
            return auth_error
        payload, error = await task_payload_from(request)
        if error is not None:
            return error
        if payload.get("target_kind") != "telegram_dispatch":
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        record = None
        try:
            record = await _offload_sync(_load_task_record, payload, "telegram_dispatch")
            if record is None:
                return {"status": "noop"}
            publisher = app.state.telegram_notification_publisher
            if payload.get("notification_id"):
                if publisher is None:
                    publisher = await _offload_sync(_notification_runtime)
                    app.state.telegram_notification_publisher = publisher
                delivery_state = await publisher.publish(payload["notification_id"])
                if delivery_state != "delivered":
                    if delivery_state in {"failed", "delivery_unknown"}:
                        return {"status": delivery_state}
                    return JSONResponse({"error": "Notification delivery pending"}, status_code=503)
                if not record.operation_id:
                    return {"status": "delivered"}
            if record.operation_id:
                ack_state = await _offload_sync(_first_ack_state, record)
                if ack_state not in {None, "delivered"}:
                    if ack_state in {"failed", "delivery_unknown"}:
                        return {"status": ack_state}
                    return JSONResponse({"error": "First acknowledgement pending"}, status_code=503)
            worker = await _dispatch_worker_for(app, record)
            if isinstance(worker, JSONResponse):
                return worker
            await _schedule_dispatch_status_checks(app, record, publisher)
            settled = await worker.handle_task(payload)
            client = await _publish_ready_head_wakeup(app, record)
            lifecycle_pending = await _publish_terminal_status(app, record, publisher, client)
            if lifecycle_pending is not None:
                return lifecycle_pending
        except (ValueError, KeyError) as exc:
            logger.error(
                "Telegram runtime execution failed",
                extra={"event": "telegram_runtime_failure", "stage": "execution",
                       "error_type": type(exc).__name__[:80],
                       **_safe_operation_log_ids(record)},
            )
            return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)
        except Exception as exc:
            logger.error(
                "Telegram runtime execution failed",
                extra={"event": "telegram_runtime_failure", "stage": "execution",
                       "error_type": type(exc).__name__[:80],
                       **_safe_operation_log_ids(record)},
            )
            return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)
        return {"status": "processed" if settled else "noop"}

    @app.post("/internal/status-check")
    async def status_check(request: Request):
        auth_error = await authenticate(request)
        if auth_error is not None:
            return auth_error
        payload, error = await task_payload_from(request)
        if error is not None:
            return error
        if payload.get("target_kind") != "telegram_status_check":
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        try:
            record = await _offload_sync(_load_task_record, payload, "telegram_status_check")
            if record is None:
                return {"status": "noop"}
            worker = app.state.telegram_status_check_worker
            if worker is None:
                worker = await _offload_sync(
                    _status_check_runtime, app.state.telegram_notification_publisher,
                    app.state.telegram_managed_runtime,
                )
                app.state.telegram_status_check_worker = worker
            handler = worker.handle_task
            if asyncio.iscoroutinefunction(handler):
                result = await handler(payload)
            else:
                result = await _offload_sync(handler, payload)
                if asyncio.iscoroutine(result):
                    result = await result
        except (ValueError, KeyError):
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Status check unavailable"}, status_code=503)
        return {"status": result}

    @app.post("/internal/deliver")
    async def deliver(request: Request):
        auth_error = await authenticate(request)
        if auth_error is not None:
            return auth_error
        payload, error = await task_payload_from(request)
        if error is not None:
            return error
        if payload.get("target_kind") != "telegram_dispatch" or not payload.get("notification_id"):
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        try:
            record = await _offload_sync(_load_task_record, payload, "telegram_dispatch")
            if record is None:
                return {"status": "noop"}
            publisher = app.state.telegram_notification_publisher
            if publisher is None:
                publisher = await _offload_sync(_notification_runtime)
                app.state.telegram_notification_publisher = publisher
            result = await publisher.publish(payload["notification_id"])
        except (ValueError, KeyError):
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Delivery unavailable"}, status_code=503)
        if result not in {"delivered", "failed", "delivery_unknown", "not_found"}:
            return JSONResponse({"error": "Notification delivery pending"}, status_code=503)
        return {"status": result}

    return app


app = create_app()


__all__ = ["app", "create_app"]
