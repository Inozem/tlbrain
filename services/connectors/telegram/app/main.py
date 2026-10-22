"""Telegram webhook application with lazy runtime configuration."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .terminal_notifications import (
    terminal_lifecycle_status as _terminal_lifecycle_status,
    update_terminal_notification as _update_terminal_notification,
)


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

    class DurableOutboxPublisher:
        def __init__(self, client: Any) -> None:
            self.client = client

        def publish(self, outbox_id: str) -> None:
            from core.integrations.cloud_tasks.client import enqueue_task
            from core.integrations.firestore.outbox import (
                OutboxState,
                OutboxTargetKind,
                get_outbox_record,
                record_outbox_delivery_result,
                task_payload,
            )
            from core.operations.outbox import TaskTargetConfig

            record = get_outbox_record(self.client, outbox_id)
            if record is None or record.target_kind is not OutboxTargetKind.TELEGRAM_DISPATCH:
                raise RuntimeError("accepted Telegram outbox row is unavailable")
            if record.state is OutboxState.DELIVERED:
                return
            queue_name = os.environ.get("TELEGRAM_CONTROL_QUEUE")
            target_url = (
                os.environ.get("TELEGRAM_DELIVERY_URL") if record.notification_id
                else os.environ.get("TELEGRAM_CONTROL_URL")
            ) or os.environ.get("TELEGRAM_CONTROL_URL")
            if not queue_name or not target_url:
                raise RuntimeError("Telegram task target is not configured")
            target = TaskTargetConfig(queue_name=queue_name, url=target_url)
            now = datetime.now(timezone.utc)
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
                record_outbox_delivery_result(
                    self.client, outbox_id, delivered=False, now=now,
                    safe_error="cloud_tasks_enqueue_failed",
                )
                raise RuntimeError("Telegram task enqueue failed") from None
            result = record_outbox_delivery_result(
                self.client, outbox_id, delivered=True, now=now,
            )
            if result is None:
                raise RuntimeError("Telegram outbox row disappeared during enqueue")

    client = get_client()
    return TelegramIntake(
        repository=TelegramRepository(client=client),
        publisher=DurableOutboxPublisher(client),
        allowed_user_ids=_allowed_user_ids(),
    )


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


def _first_ack_state(record: Any) -> str | None:
    """Return the durable first-ack state for save commands; None means no draft barrier."""
    if not record.operation_id or not record.context_id:
        return None
    from core.integrations.firestore.client import get_client
    from core.integrations.firestore.operation_queue import get_operation
    from services.connectors.telegram.repository import NOTIFICATIONS, SESSIONS

    client = get_client()
    context_ref = client.collection(SESSIONS).document(record.context_id)
    operation = get_operation(context_ref, record.operation_id)
    if operation is None or not operation.input_ref:
        return "unavailable"
    command = client.document(operation.input_ref).get()
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


def _notification_runtime() -> Any:
    from core.integrations.firestore.client import get_client
    from services.connectors.telegram.client import TelegramClient
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.repository import TelegramRepository

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("notification runtime is not configured")
    return TelegramNotificationPublisher(TelegramRepository(get_client()), TelegramClient(token))


def create_app(
    *, intake: Any = None, webhook_secret: str | None = None,
    queue_worker: Any = None, notification_publisher: Any = None,
    oidc_verifier: Any = None,
) -> FastAPI:
    app = FastAPI()
    app.state.telegram_intake = intake
    app.state.telegram_webhook_secret = webhook_secret
    app.state.telegram_queue_worker = queue_worker
    app.state.telegram_notification_publisher = notification_publisher
    app.state.telegram_oidc_verifier = oidc_verifier

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
        except (ValueError, Exception):
            enabled = False
        return {"status": "ok", "telegram_enabled": enabled}

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        secret = app.state.telegram_webhook_secret
        if secret is None:
            try:
                secret = await _offload_sync(_firestore_secret)
            except Exception:
                return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not secret or not hmac.compare_digest(supplied, secret):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        try:
            update = await request.json()
        except Exception:
            return JSONResponse({"detail": "malformed JSON"}, status_code=400)
        if not isinstance(update, dict):
            return JSONResponse({"detail": "JSON object required"}, status_code=400)
        intake_instance = app.state.telegram_intake
        if intake_instance is None:
            try:
                intake_instance = await _offload_sync(_default_intake)
                app.state.telegram_intake = intake_instance
            except Exception:
                return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        try:
            await _offload_sync(intake_instance.handle_update, update)
        except Exception:
            return JSONResponse({"detail": "intake unavailable"}, status_code=503)
        return {"ok": True}

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
            worker = app.state.telegram_queue_worker
            if worker is None:
                return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)
            # Queue worker isolates each synchronous Firestore operation from its heartbeat loop.
            settled = await worker.handle_task(payload)
            if record.operation_id:
                lifecycle_id = await _offload_sync(_update_terminal_notification, record)
                if lifecycle_id:
                    if publisher is None:
                        publisher = await _offload_sync(_notification_runtime)
                        app.state.telegram_notification_publisher = publisher
                    delivery = await publisher.publish(lifecycle_id)
                    if delivery not in {"delivered", "failed", "delivery_unknown", "not_found"}:
                        return JSONResponse({"error": "Lifecycle delivery pending"}, status_code=503)
        except (ValueError, KeyError):
            return JSONResponse({"error": "Invalid task"}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Dispatch unavailable"}, status_code=503)
        return {"status": "processed" if settled else "noop"}

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
