"""Telegram webhook application with lazy runtime configuration."""

from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


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
            target_url = os.environ.get("TELEGRAM_CONTROL_URL")
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


def create_app(*, intake: Any = None, webhook_secret: str | None = None) -> FastAPI:
    app = FastAPI()
    app.state.telegram_intake = intake
    app.state.telegram_webhook_secret = webhook_secret

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
                secret = _firestore_secret()
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
                intake_instance = _default_intake()
                app.state.telegram_intake = intake_instance
            except Exception:
                return JSONResponse({"detail": "webhook unavailable"}, status_code=503)
        try:
            intake_instance.handle_update(update)
        except Exception:
            return JSONResponse({"detail": "intake unavailable"}, status_code=503)
        return {"ok": True}

    return app


app = create_app()


__all__ = ["app", "create_app"]
