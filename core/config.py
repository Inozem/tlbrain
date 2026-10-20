import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class ConnectorCapabilities:
    telegram_enabled: bool
    tldv_enabled: bool


def get_connector_capabilities() -> ConnectorCapabilities:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    allowed_ids = os.environ.get("TELEGRAM_ALLOWED_USER_IDS")

    if (token is None) != (allowed_ids is None):
        raise ValueError("Telegram connector configuration is incomplete")

    telegram_enabled = token is not None
    if telegram_enabled:
        if not token.strip() or not allowed_ids.strip():
            raise ValueError("Telegram connector configuration must not be empty")
        for entry in allowed_ids.split(","):
            user_id = entry.strip()
            if not re.fullmatch(r"[0-9]+", user_id) or not user_id.strip("0"):
                raise ValueError("Telegram allowed user IDs must be positive decimal IDs")

    return ConnectorCapabilities(
        telegram_enabled=telegram_enabled,
        tldv_enabled=bool(os.environ.get("TLDV_API_KEY")),
    )


def get_installation_timezone() -> ZoneInfo:
    timezone_name = os.environ.get("TZ", "UTC")
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("TZ must name a valid IANA timezone") from None


def get_default_dialog_date(accepted_at: datetime) -> date:
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        raise ValueError("accepted_at must be timezone-aware")
    return accepted_at.astimezone(get_installation_timezone()).date()


def get_root_folder_id() -> str:
    url = os.environ["ROOT_FOLDER_URL"]
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError(f"Cannot extract folder ID from ROOT_FOLDER_URL: {url}")
    return match.group(1)


def get_retrieval_top_k() -> int:
    return int(os.environ["RETRIEVAL_TOP_K"])


def get_retrieval_score_threshold() -> float:
    return float(os.environ["RETRIEVAL_SCORE_THRESHOLD"])
