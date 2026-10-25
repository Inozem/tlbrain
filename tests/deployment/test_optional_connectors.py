"""Offline deployment contract for optional Telegram and TL;DV connectors.

These routes run in synthetic checkouts with fake gcloud and Docker commands;
no credentials or provider calls are used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deploy_routes import _run_route


TOP_ROUTES = ("deploy.sh", "local_deploy.sh")
DIRECT_MCP_ROUTES = ("core/deploy_mcp.sh", "core/local_deploy_mcp.sh")
ROUTES = TOP_ROUTES + DIRECT_MCP_ROUTES


def _connector_environment(telegram: bool, tldv: bool) -> dict[str, str]:
    return {
        "TELEGRAM_BOT_TOKEN": "synthetic-telegram-token" if telegram else "",
        "TELEGRAM_ALLOWED_USER_IDS": "101,202" if telegram else "",
        "TLDV_API_KEY": "synthetic-tldv-key" if tldv else "",
    }


def _cloud_commands(events: list[list[str]]) -> list[str]:
    return [event[1] for event in events if event[0] == "gcloud"]


def _cloud_argv_events(events: list[list[str]]) -> list[list[str]]:
    return [event for event in events if event[0] == "gcloud-argv"]


def _is_mcp_deploy(event: list[str]) -> bool:
    return len(event) > 3 and event[1:4] == ["run", "deploy", "synthetic-mcp"]


def _is_telegram_service_deploy(event: list[str]) -> bool:
    return len(event) > 3 and event[1:3] == ["run", "deploy"] and "telegram" in event[3].lower()


def _is_service_config(event: list[str], service: str) -> bool:
    return len(event) > 3 and event[1] == "run" and (
        (event[2] == "deploy" and event[3] == service)
        or (event[2:4] == ["services", "update"] and len(event) > 4 and event[4] == service)
    )


def _deploy_mcp_command(commands: list[str]) -> str:
    return next(command for command in commands if command.startswith("run deploy synthetic-mcp "))


def _assert_telegram_secret_absent(result, events: list[list[str]]) -> None:
    token = "synthetic-telegram-token"
    deployment_argv = [
        " ".join(event) for event in events
        if event[0] in {"gcloud-argv", "docker-argv"}
    ]
    assert all(token not in argv for argv in deployment_argv)
    assert token not in result.stdout
    assert token not in result.stderr


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("telegram,tldv", [(False, False), (False, True), (True, False), (True, True)])
def test_optional_connector_capabilities_are_selected_by_nonblank_configuration(
    tmp_path: Path, route: str, telegram: bool, tldv: bool,
) -> None:
    result, events, _root = _run_route(
        tmp_path, route, env_overrides=_connector_environment(telegram, tldv),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    _assert_telegram_secret_absent(result, events)

    commands = _cloud_commands(events)
    mcp_command = _deploy_mcp_command(commands)
    telegram_deploys = [event for event in _cloud_argv_events(events) if _is_telegram_service_deploy(event)]
    if route in TOP_ROUTES and telegram:
        assert telegram_deploys
        # The allowlist is configuration, while the bot token remains in Firestore.
        configured_allowlist = [
            arg for event in telegram_deploys for arg in event
            if "TELEGRAM_ALLOWED_USER_IDS" in arg
        ]
        assert configured_allowlist
        assert all("TELEGRAM_ALLOWED_USER_IDS=101,202" in arg for arg in configured_allowlist)
    else:
        assert not telegram_deploys
    assert "TELEGRAM_BOT_TOKEN=synthetic-telegram-token" not in mcp_command

    tldv_lookups = [command for command in commands if "functions describe" in command and "tldv" in command]
    tldv_deploys = [command for command in commands if "functions deploy" in command and "tldv" in command]
    tldv_urls = [line for line in result.stdout.splitlines() if "https://synthetic.example/" in line and "tldv" in line]
    if tldv:
        assert tldv_lookups or tldv_deploys
        if route in TOP_ROUTES:
            assert len(tldv_deploys) == 2
    else:
        assert not tldv_lookups
        assert not tldv_deploys
        assert not tldv_urls

    if route in TOP_ROUTES:
        assert any(command.startswith("run deploy synthetic-vector-sync ") for command in commands)
        assert any(command.startswith("functions deploy synthetic-sync-checker ") for command in commands)
        assert mcp_command


@pytest.mark.parametrize("route", ROUTES)
def test_both_optional_connectors_off_keeps_core_services_available(tmp_path: Path, route: str) -> None:
    result, events, _root = _run_route(tmp_path, route, env_overrides=_connector_environment(False, False))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    _assert_telegram_secret_absent(result, events)
    commands = _cloud_commands(events)
    assert any(command.startswith("run deploy synthetic-mcp ") for command in commands)
    if route in TOP_ROUTES:
        assert any(command.startswith("run deploy synthetic-vector-sync ") for command in commands)
        assert any(command.startswith("functions deploy synthetic-sync-checker ") for command in commands)


@pytest.mark.parametrize("route", ROUTES)
def test_whitespace_connector_values_are_treated_as_unset(tmp_path: Path, route: str) -> None:
    environment = {
        "TELEGRAM_BOT_TOKEN": "   ",
        "TELEGRAM_ALLOWED_USER_IDS": " \t ",
        "TLDV_API_KEY": "   ",
    }
    result, events, _root = _run_route(tmp_path, route, env_overrides=environment)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    _assert_telegram_secret_absent(result, events)
    commands = _cloud_commands(events)
    mcp_command = _deploy_mcp_command(commands)
    assert "TELEGRAM_BOT_TOKEN" not in mcp_command
    assert "TELEGRAM_ALLOWED_USER_IDS" not in mcp_command
    assert not any("functions describe" in command and "tldv" in command for command in commands)
    assert not any("functions deploy" in command and "tldv" in command for command in commands)


@pytest.mark.parametrize("route", DIRECT_MCP_ROUTES)
def test_first_mcp_deploy_waits_for_real_status_url_before_self_url_readiness(
    tmp_path: Path, route: str,
) -> None:
    result, events, _root = _run_route(
        tmp_path, route, env_overrides=_connector_environment(False, False),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    _assert_telegram_secret_absent(result, events)

    cloud = _cloud_argv_events(events)
    deploys = [event for event in cloud if _is_mcp_deploy(event)]
    assert deploys, f"no MCP deploy in command trace: {cloud}"
    initial_deploy = deploys[0]
    initial_text = " ".join(initial_deploy)
    assert "INTERNAL_DIALOG_OPERATIONS_URL" not in initial_text
    assert "QUEUE" not in initial_text
    assert "OIDC" not in initial_text

    url_event = next(
        event for event in events
        if event[:2] == ["gcloud-result", "run-service-url"] and event[2] == "synthetic-mcp"
    )
    real_url = url_event[3]
    assert real_url == "https://synthetic.example/synthetic-mcp"
    url_position = events.index(url_event)
    initial_position = events.index(initial_deploy)
    assert initial_position < url_position
    describe_position = next(
        index for index, event in enumerate(events)
        if event[0] == "gcloud-argv" and event[1:4] == ["run", "services", "describe"]
        and len(event) > 4 and event[4] == "synthetic-mcp"
    )
    assert initial_position < describe_position < url_position

    readiness = next(
        event for event in cloud
        if _is_service_config(event, "synthetic-mcp")
        and any("INTERNAL_DIALOG_OPERATIONS_URL=" in arg for arg in event)
    )
    readiness_position = events.index(readiness)
    readiness_text = " ".join(readiness)
    assert readiness_position > url_position
    assert f"INTERNAL_DIALOG_OPERATIONS_URL={real_url}/internal/dialog-operations" in readiness_text
    assert "QUEUE" in readiness_text
    assert "OIDC" in readiness_text
    assert "synthetic-mcp.run.app" not in readiness_text


def _marker_position(events: list[list[str]], marker: str) -> int:
    return next(
        index for index, event in enumerate(events)
        if event[0] == "rollout-marker" and marker in event[1]
    )


def _assert_drain_confirmation_prompt(result) -> None:
    prompt = (result.stdout + "\n" + result.stderr).lower()
    assert "confirm" in prompt
    assert "old workers" in prompt
    assert "external" in prompt and ("i/o" in prompt or "io" in prompt)
    assert re.search(r"old workers.{0,120}(drain|terminat)", prompt, re.DOTALL)
    assert re.search(r"external.{0,120}(drain|terminat)", prompt, re.DOTALL)


def _rollout_config_events(events: list[list[str]], service: str) -> list[int]:
    return [index for index, event in enumerate(events)
            if event[0] == "gcloud-argv" and _is_service_config(event, service)]


def test_rollout_orders_compatible_readers_consumers_drain_then_writers(tmp_path: Path) -> None:
    result, events, _root = _run_route(
        tmp_path,
        "deploy.sh",
        env_overrides=_connector_environment(True, False),
        stdin_answers="y\ny\n",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    _assert_drain_confirmation_prompt(result)

    decoder = _marker_position(events, "rollout_phase:decoder_compatible")
    guards = _marker_position(events, "rollout_phase:guards_compatible")
    drained = _marker_position(events, "rollout_gate:old_workers_drained")
    writers = _marker_position(events, "rollout_phase:writers_enabled")
    assert decoder < guards < drained < writers

    for service in ("synthetic-mcp", "synthetic-vector-sync"):
        configs = _rollout_config_events(events, service)
        assert configs, f"no config/deploy event for {service}"
        assert len(configs) >= 2
        assert configs[0] < decoder < configs[1] < guards

    writer_commands = [
        index for index, event in enumerate(events)
        if event[0] == "gcloud-argv" and _is_service_config(event, "synthetic-mcp")
        and "INTERNAL_DIALOG_OPERATIONS_URL=" in " ".join(event)
    ]
    telegram_commands = [
        index for index, event in enumerate(events)
        if event[0] == "gcloud-argv" and _is_telegram_service_deploy(event)
    ]
    assert writer_commands and telegram_commands
    assert writers < min(writer_commands)
    assert writers < min(telegram_commands)


@pytest.mark.parametrize("answers", ["y\nn\n", "y\n"], ids=["operator-declines", "missing-confirmation"])
def test_rollout_without_drain_confirmation_stops_before_writer_activation(
    tmp_path: Path, answers: str,
) -> None:
    result, events, _root = _run_route(
        tmp_path,
        "deploy.sh",
        env_overrides=_connector_environment(True, False),
        stdin_answers=answers,
    )
    assert result.returncode != 0
    _assert_drain_confirmation_prompt(result)
    _assert_telegram_secret_absent(result, events)
    decoder = _marker_position(events, "rollout_phase:decoder_compatible")
    guards = _marker_position(events, "rollout_phase:guards_compatible")
    assert decoder < guards
    assert not any(event[0] == "rollout-marker" and "rollout_gate:old_workers_drained" in event[1]
                   for event in events)
    assert not any(event[0] == "rollout-marker" and "rollout_phase:writers_enabled" in event[1]
                   for event in events)
    assert not any(
        event[0] == "gcloud-argv" and _is_service_config(event, "synthetic-mcp")
        and "INTERNAL_DIALOG_OPERATIONS_URL=" in " ".join(event)
        for event in events
    )
    assert not any(
        event[0] == "gcloud-argv" and _is_telegram_service_deploy(event)
        for event in events
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "values",
    [
        {"TELEGRAM_BOT_TOKEN": "synthetic-telegram-token", "TELEGRAM_ALLOWED_USER_IDS": ""},
        {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ALLOWED_USER_IDS": "101,202"},
    ],
)
def test_partial_telegram_configuration_fails_before_cloud_or_docker_mutation(
    tmp_path: Path, route: str, values: dict[str, str],
) -> None:
    environment = {**_connector_environment(False, False), **values}
    result, events, _root = _run_route(tmp_path, route, env_overrides=environment)
    assert result.returncode != 0
    _assert_telegram_secret_absent(result, events)
    assert "TELEGRAM_BOT_TOKEN" in result.stdout + result.stderr
    assert "TELEGRAM_ALLOWED_USER_IDS" in result.stdout + result.stderr
    assert not any(event[0] in {"gcloud", "gcloud-argv", "docker", "docker-argv"} for event in events)

