"""Focused runtime tests for the real default Telegram outbox publisher."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest


NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)
CONTROL_URL = "https://telegram.pytest.invalid/internal/dispatch"
DELIVERY_URL = "https://telegram.pytest.invalid/internal/deliver"
TASK_IDENTITY = "telegram-task@synthetic-project.iam.gserviceaccount.com"


def _record(client, outbox_id: str, *, notification_id: str | None = None):
    outbox = importlib.import_module("core.integrations.firestore.outbox")
    return outbox.create_outbox_record(
        client,
        outbox_id,
        target_kind=outbox.OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1,
        schedule_at=NOW,
        now=NOW,
        context_id="telegram-context",
        notification_id=notification_id,
    )


def _runtime(monkeypatch, firestore_fake, *, enqueue):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-telegram-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "pytest-telegram-control")
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", CONTROL_URL)
    monkeypatch.setenv("TELEGRAM_DELIVERY_URL", DELIVERY_URL)
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://telegram.pytest.invalid")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", TASK_IDENTITY)
    monkeypatch.setenv("OIDC_AUDIENCE", "https://telegram.pytest.invalid")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", TASK_IDENTITY)
    client = firestore_fake.client_class()
    firestore_client = importlib.import_module("core.integrations.firestore.client")
    monkeypatch.setattr(firestore_client, "get_client", lambda: client)
    cloud_tasks = importlib.import_module("core.integrations.cloud_tasks.client")
    monkeypatch.setattr(cloud_tasks, "enqueue_task", enqueue)
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    intake = app_module._default_intake()
    return app_module, intake, client


def test_default_intake_publisher_sends_matching_oidc_for_dispatch_and_delivery(
    monkeypatch, firestore_fake,
):
    calls = []

    def enqueue_task(**kwargs):
        calls.append(kwargs)
        return True

    _app, intake, client = _runtime(monkeypatch, firestore_fake, enqueue=enqueue_task)
    _record(client, "telegram-command-outbox")
    _record(client, "telegram-notification-outbox", notification_id="telegram-notification")

    intake.publisher.publish("telegram-command-outbox")
    intake.publisher.publish("telegram-notification-outbox")
    intake.publisher.publish("telegram-command-outbox")
    monkeypatch.delenv("TELEGRAM_DELIVERY_URL")
    _record(client, "telegram-notification-fallback", notification_id="telegram-notification-fallback")
    intake.publisher.publish("telegram-notification-fallback")

    assert [call["url"] for call in calls] == [CONTROL_URL, DELIVERY_URL, CONTROL_URL]
    assert [call["service_account_email"] for call in calls] == [TASK_IDENTITY] * 3
    assert [call["audience"] for call in calls] == [
        "https://telegram.pytest.invalid",
    ] * 3
    outbox = importlib.import_module("core.integrations.firestore.outbox")
    assert outbox.get_outbox_record(client, "telegram-command-outbox").state is outbox.OutboxState.DELIVERED
    assert outbox.get_outbox_record(client, "telegram-notification-outbox").state is outbox.OutboxState.DELIVERED


@pytest.mark.parametrize("misconfiguration", ["missing", "mismatched"])
def test_default_intake_publisher_fails_closed_without_matching_oidc(
    monkeypatch, firestore_fake, misconfiguration,
):
    calls = []
    _app, intake, client = _runtime(monkeypatch, firestore_fake, enqueue=lambda **kwargs: calls.append(kwargs))
    _record(client, "telegram-command-outbox")
    if misconfiguration == "missing":
        monkeypatch.delenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL")
        monkeypatch.delenv("TASK_OIDC_SERVICE_ACCOUNT_EMAIL", raising=False)
        monkeypatch.delenv("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL", raising=False)
        monkeypatch.delenv("TASK_SERVICE_ACCOUNT_EMAIL", raising=False)
    else:
        monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "different@synthetic-project.iam.gserviceaccount.com")

    with pytest.raises(RuntimeError, match="task target is not configured"):
        intake.publisher.publish("telegram-command-outbox")
    assert calls == []
    outbox = importlib.import_module("core.integrations.firestore.outbox")
    assert outbox.get_outbox_record(client, "telegram-command-outbox").state is outbox.OutboxState.PENDING
