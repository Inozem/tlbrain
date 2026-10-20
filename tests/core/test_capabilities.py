from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.config import (
    get_connector_capabilities,
    get_default_dialog_date,
    get_installation_timezone,
    get_retrieval_score_threshold,
    get_retrieval_top_k,
    get_root_folder_id,
)


@pytest.mark.parametrize(
    ("telegram_enabled", "tldv_enabled"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_connector_capabilities_are_independent(
    monkeypatch, telegram_enabled, tldv_enabled
):
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_USER_IDS",
        "TLDV_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    if telegram_enabled:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-telegram-token")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "123456, 789012")
    if tldv_enabled:
        monkeypatch.setenv("TLDV_API_KEY", "synthetic-tldv-key")

    capabilities = get_connector_capabilities()

    assert capabilities.telegram_enabled is telegram_enabled
    assert capabilities.tldv_enabled is tldv_enabled


@pytest.mark.parametrize(
    ("present_name", "missing_name"),
    [
        ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USER_IDS"),
        ("TELEGRAM_ALLOWED_USER_IDS", "TELEGRAM_BOT_TOKEN"),
    ],
)
def test_partial_telegram_configuration_is_rejected_without_leaking_token(
    monkeypatch, present_name, missing_name
):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("TLDV_API_KEY", raising=False)
    secret = "synthetic-telegram-secret"
    monkeypatch.setenv(present_name, secret if present_name == "TELEGRAM_BOT_TOKEN" else "123456")
    monkeypatch.delenv(missing_name, raising=False)

    with pytest.raises(ValueError) as exc_info:
        get_connector_capabilities()

    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    ("token", "allowed_ids"),
    [
        ("   ", "123456"),
        ("synthetic-token", "   "),
    ],
)
def test_whitespace_only_telegram_values_are_rejected(
    monkeypatch, token, allowed_ids
):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", allowed_ids)
    monkeypatch.delenv("TLDV_API_KEY", raising=False)

    with pytest.raises(ValueError):
        get_connector_capabilities()


@pytest.mark.parametrize(
    "allowed_ids",
    ["", "123456,,789012", "not-a-number", "0", "-1", "12.3", "12, ", "１２３"],
)
def test_invalid_telegram_allowlist_is_rejected_without_leaking_token(
    monkeypatch, allowed_ids
):
    secret = "synthetic-telegram-secret"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", allowed_ids)
    monkeypatch.delenv("TLDV_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        get_connector_capabilities()

    assert secret not in str(exc_info.value)


def test_telegram_allowlist_entries_may_have_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", " 123456 , 789012 ")
    monkeypatch.delenv("TLDV_API_KEY", raising=False)

    assert get_connector_capabilities().telegram_enabled is True


@pytest.mark.parametrize("api_key", [None, ""])
def test_empty_or_absent_tldv_key_disables_connector(monkeypatch, api_key):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    if api_key is None:
        monkeypatch.delenv("TLDV_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TLDV_API_KEY", api_key)

    assert get_connector_capabilities().tldv_enabled is False


def test_installation_timezone_defaults_to_utc(monkeypatch):
    monkeypatch.delenv("TZ", raising=False)

    assert get_installation_timezone() == ZoneInfo("UTC")


def test_invalid_installation_timezone_raises_value_error(monkeypatch):
    monkeypatch.setenv("TZ", "invalid/synthetic-zone")

    with pytest.raises(ValueError):
        get_installation_timezone()


def test_default_dialog_date_uses_installation_timezone_at_date_boundary(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    accepted_at = datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)

    assert get_default_dialog_date(accepted_at) == date(2024, 12, 31)


def test_core_config_consumers_do_not_require_tldv_configuration(monkeypatch):
    monkeypatch.delenv("TLDV_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_FOLDER_URL", "https://drive.example/folders/synthetic-root")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "17")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.42")

    assert get_root_folder_id() == "synthetic-root"
    assert get_retrieval_top_k() == 17
    assert get_retrieval_score_threshold() == pytest.approx(0.42)
