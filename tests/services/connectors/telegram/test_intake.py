"""Contract tests for Telegram intake before its production modules exist.

The test-facing boundary is ``TelegramRepository(client=...)`` in
``services.connectors.telegram.repository``. It owns Firestore transactions for
receipt deduplication, draft/message persistence, and operation reservations.
``TelegramIntake(repository, publisher, classifier, semantic_router)`` parses
updates and publishes the durable outbox item returned by the repository.
"""

from __future__ import annotations

import asyncio
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import pytest
from core.integrations.firestore.operation_queue import claim_head, complete_head


pytestmark = pytest.mark.offline


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _repository(firestore_fake):
    repository_type = importlib.import_module(
        "services.connectors.telegram.repository"
    ).TelegramRepository
    client = firestore_fake.client_class()
    return repository_type(client=client), client


def _read_doc(client, path):
    parts = path.split("/")
    reference = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        reference = reference.collection(parts[index]).document(parts[index + 1])
    snapshot = reference.get()
    return snapshot.to_dict() if snapshot.exists else None


def _paths_with_prefix(firestore_fake, prefix):
    return [path for path in firestore_fake.documents if path.startswith(prefix)]


def _direct_child_paths(firestore_fake, parent):
    prefix = parent.rstrip("/") + "/"
    return [
        path for path in firestore_fake.documents
        if path.startswith(prefix) and "/" not in path[len(prefix):]
    ]


def _message_path(context_id, draft_id, message_id):
    return f"telegram_sessions/{context_id}/drafts/{draft_id}/messages/{message_id}"


def _intake(repository, publisher, classifier=None, semantic_router=None):
    intake_type = importlib.import_module(
        "services.connectors.telegram.intake"
    ).TelegramIntake
    return intake_type(
        repository=repository,
        publisher=publisher,
        classifier=classifier,
        semantic_router=semantic_router,
    )


def _forward(update_id: int, *, chat_id: int = 42, user_id: int = 7, message_id: int | None = None):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id or update_id + 100,
            "date": 1_798_000_000,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id},
            "forward_origin": {"type": "user", "sender_user": {"id": 91}},
            "text": f"synthetic forwarded message {update_id}",
        },
    }


def _command(update_id: int, command: str):
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 100,
            "date": 1_798_000_000,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 7},
            "text": command,
        },
    }


def _value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


class _Publisher:
    def __init__(self):
        self.outbox_ids = []
        self.failures = 0

    def publish(self, outbox_id):
        self.outbox_ids.append(outbox_id)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("synthetic publish gap")


class _Classifier:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("forwards must not be classified during intake")


def test_forward_atomically_persists_collecting_draft_message_and_draft_ack(
    firestore_fake, sdk_fakes,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    classifier = _Classifier()
    semantic_router = _Classifier()
    intake = _intake(repository, publisher, classifier, semantic_router)

    accepted = intake.handle_update(_forward(1))

    assert _value(accepted, "accepted") is True
    assert _value(accepted, "draft_id")
    assert _value(accepted, "dialog_number") > 0
    assert _value(accepted, "message_id") == 101
    assert _value(accepted, "notification_id")
    assert _value(accepted, "outbox_id")

    context_id = _value(accepted, "context_id")
    update_record = _read_doc(client, f"telegram_sessions/{context_id}/updates/1")
    draft = _read_doc(client, f"telegram_sessions/{context_id}/drafts/{_value(accepted, 'draft_id')}")
    draft_id = _value(accepted, "draft_id")
    message = _read_doc(client, _message_path(context_id, draft_id, 101))
    notification = _read_doc(client, f"notification_outbox/{_value(accepted, 'notification_id')}")
    outbox = _read_doc(client, f"task_outbox/{_value(accepted, 'outbox_id')}")
    assert update_record is not None
    assert draft["state"] == "collecting"
    assert draft["dialog_number"] == _value(accepted, "dialog_number")
    assert message["message_id"] == 101
    assert message["text"] == "synthetic forwarded message 1"
    assert notification["kind"] == "draft_ack"
    assert str(_value(accepted, "dialog_number")) in str(notification["body"])
    assert outbox["notification_id"] == _value(accepted, "notification_id")
    assert publisher.outbox_ids == [_value(accepted, "outbox_id")]
    assert classifier.calls == []
    assert semantic_router.calls == []
    assert not any(event["boundary"] == "gemini" for event in sdk_fakes.trace)


def test_distinct_forwards_reuse_the_unique_collecting_draft_and_dialog_number(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())

    first = intake.handle_update(_forward(2))
    second = intake.handle_update(_forward(3))
    context_id = _value(first, "context_id")

    assert _value(first, "draft_id") == _value(second, "draft_id")
    assert _value(first, "dialog_number") == _value(second, "dialog_number")
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/drafts")) == 1
    draft_id = _value(first, "draft_id")
    assert _read_doc(client, _message_path(context_id, draft_id, 102))["message_id"] == 102
    assert _read_doc(client, _message_path(context_id, draft_id, 103))["message_id"] == 103


def test_channel_forward_persists_original_channel_metadata(firestore_fake):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    update = _forward(4)
    update["message"]["forward_origin"] = {
        "type": "channel",
        "chat": {"id": -1001234567890, "title": "Original channel"},
        "message_id": 765,
        "date": 1_798_000_000,
    }

    accepted = intake.handle_update(update)

    context_id = _value(accepted, "context_id")
    draft_id = _value(accepted, "draft_id")
    message = _read_doc(client, _message_path(context_id, draft_id, 104))
    assert message["source_chat_id"] == -1001234567890
    assert message["source_message_id"] == 765
    assert message["reliable_author_key"] == "telegram-chat:-1001234567890"
    assert message["display_speaker"] == "Original channel"


def test_duplicate_update_reuses_all_records_and_repairs_same_outbox_after_gap(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    publisher.failures = 1
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    update = _forward(8)

    with pytest.raises(RuntimeError, match="synthetic publish gap"):
        intake.handle_update(update)
    replay = intake.handle_update(update)
    context_id = _value(replay, "context_id")
    draft_id = _value(replay, "draft_id")
    notification_id = _value(replay, "notification_id")
    outbox_id = _value(replay, "outbox_id")
    draft = _read_doc(client, f"telegram_sessions/{context_id}/drafts/{draft_id}")
    update_record = _read_doc(client, f"telegram_sessions/{context_id}/updates/8")
    notification = _read_doc(client, f"notification_outbox/{notification_id}")
    outbox = _read_doc(client, f"task_outbox/{outbox_id}")
    assert draft is not None
    assert update_record["message_id"] == 108
    assert notification["notification_id"] == notification_id
    assert outbox["outbox_id"] == outbox_id
    assert publisher.outbox_ids == [outbox_id, outbox_id]
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/updates")) == 1
    assert len(_paths_with_prefix(
        firestore_fake,
        f"telegram_sessions/{context_id}/drafts/{draft_id}/messages/",
    )) == 1


@pytest.mark.parametrize(("chat_type", "user_id"), [("group", 7), ("private", 999)])
def test_disallowed_chat_or_sender_is_acknowledged_without_persisting_content(
    firestore_fake, chat_type, user_id
):
    repository, client = _repository(firestore_fake)
    intake_type = importlib.import_module(
        "services.connectors.telegram.intake"
    ).TelegramIntake
    intake = intake_type(
        repository=repository,
        publisher=_Publisher(),
        classifier=_Classifier(),
        semantic_router=_Classifier(),
        allowed_user_ids={7},
    )
    update = _forward(9, user_id=user_id)
    update["message"]["chat"]["type"] = chat_type

    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=update,
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert not any("/drafts/" in path for path in firestore_fake.documents)
    assert not any("/messages/" in path for path in firestore_fake.documents)
    assert all(update["message"]["text"] not in str(value) for value in firestore_fake.documents.values())


def test_invalid_webhook_secret_is_rejected_before_intake_or_persistence(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-synthetic-secret"},
                json=_forward(10),
            )

    response = asyncio.run(request())
    assert response.status_code == 403
    assert publisher.outbox_ids == []
    assert firestore_fake.documents == {}


def test_malformed_json_is_rejected(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(intake=intake, webhook_secret="synthetic-webhook-secret")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                content="{invalid",
            )

    response = asyncio.run(request())
    assert response.status_code == 400
    assert firestore_fake.documents == {}


def test_publisher_failure_returns_service_unavailable(firestore_fake):
    repository, _ = _repository(firestore_fake)
    publisher = _Publisher()
    publisher.failures = 1
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(intake=intake, webhook_secret="synthetic-webhook-secret")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=_forward(11),
            )

    response = asyncio.run(request())
    assert response.status_code == 503
    assert len(publisher.outbox_ids) == 1
    assert any(path.endswith("/updates/11") for path in firestore_fake.documents)


def test_health_returns_only_safe_capability_status(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-value")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "7")
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(webhook_secret="synthetic-webhook-secret")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.get("/")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "telegram_enabled": True}
    assert "secret-token-value" not in response.text
    assert "synthetic-webhook-secret" not in response.text


def test_command_reservation_blocks_forwards_until_all_fifo_heads_finish(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())

    first = intake.handle_update(_command(20, "Сохрани"))
    second = intake.handle_update(_command(21, "Расскажи подробнее"))
    context_id = _value(first, "context_id")
    assert _value(first, "queue_position") == 1
    assert _value(second, "queue_position") == 2
    context = client.collection("telegram_sessions").document(context_id)
    assert context.get().to_dict()["reserved_operation_id"] == _value(first, "operation_id")

    first_lease = claim_head(client, context, owner="pytest-worker", now=NOW)
    assert first_lease is not None
    assert first_lease.operation_id == _value(first, "operation_id")
    assert complete_head(
        client, context, operation_id=first_lease.operation_id, owner="pytest-worker",
        fence=first_lease.fence, now=NOW,
    ) is True
    state = context.get().to_dict()
    assert state["reserved_operation_id"] == _value(second, "operation_id")
    assert _value(intake.handle_update(_forward(22)), "accepted") is False

    second_lease = claim_head(client, context, owner="pytest-worker-2", now=NOW)
    assert second_lease is not None
    assert second_lease.operation_id == _value(second, "operation_id")
    assert complete_head(
        client, context, operation_id=second_lease.operation_id, owner="pytest-worker-2",
        fence=second_lease.fence, now=NOW,
    ) is True
    assert context.get().to_dict()["reserved_operation_id"] is None
    accepted = intake.handle_update(_forward(23))
    assert _value(accepted, "accepted") is True


def test_concurrent_command_and_forward_are_decided_by_repository_transaction(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    command = _command(30, "Сохрани")
    forward = _forward(31)

    with ThreadPoolExecutor(max_workers=2) as executor:
        command_future = executor.submit(intake.handle_update, command)
        forward_future = executor.submit(intake.handle_update, forward)
        command_result = command_future.result(timeout=10)
        forward_result = forward_future.result(timeout=10)

    if _value(command_result, "accepted"):
        assert _value(forward_result, "accepted") is False
        context_id = _value(command_result, "context_id")
        assert not any(
            path.endswith("/messages/131")
            for path in firestore_fake.documents
            if path.startswith(f"telegram_sessions/{context_id}/drafts/")
        )
    else:
        assert _value(forward_result, "accepted") is True
        context_id = _value(forward_result, "context_id")
        draft_id = _value(forward_result, "draft_id")
        assert _read_doc(client, _message_path(context_id, draft_id, 131))["message_id"] == 131


def test_concurrent_forward_and_save_are_snapshot_included_or_rejected(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    forward = _forward(41)
    save = _command(40, "Сохрани")

    with ThreadPoolExecutor(max_workers=2) as executor:
        forward_future = executor.submit(intake.handle_update, forward)
        save_future = executor.submit(intake.handle_update, save)
        forward_result = forward_future.result(timeout=10)
        save_result = save_future.result(timeout=10)

    accepted = _value(forward_result, "accepted")
    message_id = forward["message"]["message_id"]
    snapshot_ids = _value(save_result, "snapshot_message_ids")
    assert accepted is (message_id in snapshot_ids)
    context_id = _value(save_result, "context_id")
    if accepted:
        assert snapshot_ids.count(message_id) == 1
        draft_id = _value(forward_result, "draft_id")
        assert _read_doc(client, _message_path(context_id, draft_id, message_id)) is not None
    else:
        assert not any(
            path.endswith(f"/messages/{message_id}")
            for path in firestore_fake.documents
            if path.startswith(f"telegram_sessions/{context_id}/drafts/")
        )
