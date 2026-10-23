"""Fast durable intake for forwarded Telegram messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.contexts import (
    message_topic_name,
    normalize_thread_id,
    topic_service_event,
)
from services.connectors.telegram.repository import TelegramRepository


def _integer(value: Any) -> int | None:
    return value if type(value) is int else None


def _telegram_date(value: Any) -> datetime | None:
    stamp = _integer(value)
    return datetime.fromtimestamp(stamp, tz=timezone.utc) if stamp is not None else None


def normalize_forward(message: Mapping[str, Any]) -> dict[str, Any]:
    """Map Telegram's forwarded message shape to repository source fields."""
    origin = message.get("forward_origin")
    origin = origin if isinstance(origin, Mapping) else {}
    origin_type = origin.get("type")
    source_chat_id = source_message_id = source_user_id = None
    display_speaker = None
    reliable_author_key = None

    if origin_type == "user":
        sender = origin.get("sender_user")
        sender = sender if isinstance(sender, Mapping) else {}
        source_user_id = _integer(sender.get("id"))
        name = " ".join(str(sender.get(key)).strip() for key in ("first_name", "last_name")
                        if sender.get(key))
        display_speaker = name or (str(sender.get("username")) if sender.get("username") else None)
        if source_user_id is not None:
            reliable_author_key = f"telegram-user:{source_user_id}"
    elif origin_type == "chat":
        sender = origin.get("sender_chat")
        sender = sender if isinstance(sender, Mapping) else {}
        source_chat_id = _integer(sender.get("id"))
        display_speaker = sender.get("title") if isinstance(sender.get("title"), str) else None
        if source_chat_id is not None:
            reliable_author_key = f"telegram-chat:{source_chat_id}"
        source_message_id = _integer(origin.get("message_id"))
    elif origin_type == "channel":
        channel = origin.get("chat")
        channel = channel if isinstance(channel, Mapping) else {}
        channel_id = _integer(channel.get("id"))
        source_chat_id = channel_id
        display_speaker = channel.get("title") if isinstance(channel.get("title"), str) else None
        if channel_id is not None:
            reliable_author_key = f"telegram-chat:{channel_id}"
        source_message_id = _integer(origin.get("message_id"))
    elif origin_type == "hidden_user":
        display_speaker = origin.get("sender_user_name") if isinstance(origin.get("sender_user_name"), str) else None
    elif origin_type == "sender_user_name":
        display_speaker = origin.get("sender_user_name") if isinstance(origin.get("sender_user_name"), str) else None

    audio = message.get("voice") or message.get("audio")
    audio = audio if isinstance(audio, Mapping) else {}
    normalized: dict[str, Any] = {
        "message_id": _integer(message.get("message_id")),
        "original_date": _telegram_date(origin.get("date")) or _telegram_date(message.get("date")),
        "source_chat_id": source_chat_id,
        "source_message_id": source_message_id,
        "source_user_id": source_user_id,
        "reliable_author_key": reliable_author_key,
        "display_speaker": display_speaker,
        "text": message.get("text") if isinstance(message.get("text"), str) else None,
        "caption": message.get("caption") if isinstance(message.get("caption"), str) else None,
        "audio_file_id": audio.get("file_id") if isinstance(audio.get("file_id"), str) else None,
        "audio_file_unique_id": audio.get("file_unique_id") if isinstance(audio.get("file_unique_id"), str) else None,
        "audio_mime_type": audio.get("mime_type") if isinstance(audio.get("mime_type"), str) else None,
        "audio_file_size": _integer(audio.get("file_size")),
        "transcription_state": "pending" if audio else "not_required",
    }
    return normalized


class TelegramIntake:
    """Validate and durably accept an update, then publish its outbox row."""

    def __init__(
        self,
        repository: TelegramRepository,
        publisher: Any,
        classifier: Any = None,
        semantic_router: Any = None,
        allowed_user_ids: set[int] | frozenset[int] | None = None,
        *,
        clock: Any = None,
    ) -> None:
        self.repository = repository
        self.publisher = publisher
        self.classifier = classifier
        self.semantic_router = semantic_router
        self.allowed_user_ids = frozenset(allowed_user_ids) if allowed_user_ids is not None else None
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def handle_update(self, update: Mapping[str, Any]) -> Any:
        context = self._validated_update_context(update)
        if context is None:
            return None
        message, update_id, chat_id, user_id, thread_id = context

        handled, result = self._route_topic_event(
            message, update_id=update_id, chat_id=chat_id, thread_id=thread_id
        )
        if handled:
            return result

        if "forward_origin" in message and not isinstance(message.get("forward_origin"), Mapping):
            return None
        if "forward_origin" not in message:
            return self._handle_command(
                message,
                update_id=update_id,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        return self._handle_forward(
            message, update_id=update_id, chat_id=chat_id, thread_id=thread_id
        )

    def _validated_update_context(
        self, update: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], int, int, int, int | None] | None:
        if not isinstance(update, Mapping):
            return None
        message = update.get("message")
        if not isinstance(message, Mapping):
            return None
        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, Mapping) or not isinstance(sender, Mapping):
            return None
        chat_id = _integer(chat.get("id"))
        user_id = _integer(sender.get("id"))
        update_id = _integer(update.get("update_id"))
        if (chat.get("type") != "private" or chat_id is None or user_id is None
                or (self.allowed_user_ids is not None and user_id not in self.allowed_user_ids)):
            return None
        if update_id is None:
            return None

        raw_thread_id = message.get("message_thread_id")
        thread_id = normalize_thread_id(raw_thread_id)
        # Missing and explicit null both describe a chat's default context.
        # Any other unusable supplied value must not fall through to it.
        if raw_thread_id is not None and thread_id is None:
            return None

        return message, update_id, chat_id, user_id, thread_id

    def _route_topic_event(
        self,
        message: Mapping[str, Any],
        *,
        update_id: int,
        chat_id: int,
        thread_id: int | None,
    ) -> tuple[bool, Any]:
        topic_event = topic_service_event(message)
        if topic_event is not None:
            if thread_id is None:
                return True, None
            event_kind, topic_name = topic_event
            return True, self.repository.record_topic_event(
                update_id,
                chat_id=chat_id,
                thread_id=thread_id,
                event_kind=event_kind,
                topic_name=topic_name,
                now=self.clock(),
            )
        return False, None

    def _handle_command(
        self,
        message: Mapping[str, Any],
        *,
        update_id: int,
        chat_id: int,
        user_id: int,
        thread_id: int | None,
    ) -> Any:
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        accepted = self.repository.accept_command(
            update_id,
            chat_id=chat_id,
            user_id=user_id,
            thread_id=thread_id,
            topic_name=message_topic_name(message),
            command=parse_command(text),
            now=self.clock(),
        )
        self._publish_if_accepted(accepted)
        return accepted

    def _handle_forward(
        self,
        message: Mapping[str, Any],
        *,
        update_id: int,
        chat_id: int,
        thread_id: int | None,
    ) -> Any:
        normalized = normalize_forward(message)
        if normalized["message_id"] is None:
            return None
        accepted = self.repository.accept_forward(
            update_id,
            chat_id=chat_id,
            thread_id=thread_id,
            topic_name=message_topic_name(message),
            message=normalized,
            now=self.clock(),
        )
        self._publish_if_accepted(accepted)
        return accepted

    def _publish_if_accepted(self, result: Any) -> None:
        outbox_id = result.outbox_id if hasattr(result, "outbox_id") else result.get("outbox_id")
        is_accepted = result.accepted if hasattr(result, "accepted") else result.get("accepted")
        if is_accepted and outbox_id:
            self.publisher.publish(outbox_id)


__all__ = ["TelegramIntake", "normalize_forward"]
