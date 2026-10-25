"""Offline deployment contract for optional Telegram and TL;DV connectors.

These routes run in synthetic checkouts with fake gcloud and Docker commands;
no credentials or provider calls are used.
"""

from __future__ import annotations

import re
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deploy_routes import _bash, _run_route


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
        assert "TLDV_RECONCILIATION_URL=" in " ".join(next(
            event for event in _cloud_argv_events(events) if _is_mcp_deploy(event)
        ))
        if route in TOP_ROUTES:
            assert len(tldv_deploys) == 2
    else:
        assert not tldv_lookups
        assert not tldv_deploys
        assert not tldv_urls
        assert "TLDV_RECONCILIATION_URL" not in " ".join(next(
            event for event in _cloud_argv_events(events) if _is_mcp_deploy(event)
        ))

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
    mcp_argv = next(event for event in _cloud_argv_events(events) if _is_mcp_deploy(event))
    assert "TLDV_RECONCILIATION_URL" not in " ".join(mcp_argv)
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
    assert real_url == "https://synthetic-mcp-uc.a.run.app"
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
    for setting in (
        "DIALOG_OPERATIONS_QUEUE=",
        "CLOUD_TASKS_OIDC_AUDIENCE=",
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=",
        "OIDC_AUDIENCE=",
        "OIDC_SERVICE_ACCOUNT_EMAIL=",
    ):
        assert setting in readiness_text
    assert "synthetic-mcp.run.app" not in readiness_text


@pytest.mark.parametrize("route", DIRECT_MCP_ROUTES)
@pytest.mark.parametrize("allowed_email", ["first@example.invalid,second@example.invalid", ""])
def test_direct_mcp_routes_preserve_allowlist_list_and_open_mode(
    tmp_path: Path, route: str, allowed_email: str,
) -> None:
    result, events, _root = _run_route(
        tmp_path, route,
        env_overrides={**_connector_environment(False, False), "ALLOWED_EMAIL": allowed_email},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    initial = next(event for event in _cloud_argv_events(events) if _is_mcp_deploy(event))
    env_arg = next(arg for arg in initial if arg.startswith("^|^"))
    assert f"ALLOWED_EMAIL={allowed_email}" in env_arg


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


def _run_deploy_helper(tmp_path: Path, helper: str, *args: str, **overrides: str):
    bash = _bash()
    bin_dir = tmp_path / "helper-bin"
    bin_dir.mkdir()
    log = tmp_path / "helper-events.jsonl"
    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "\"$ROUTE_PYTHON\" -c 'import json,os,sys; open(os.environ[\"ROUTE_EVENT_LOG\"],\"a\",encoding=\"utf-8\").write(json.dumps(sys.argv[1:])+\"\\n\")' \"$@\"\n"
        "case \"$*\" in\n"
        "  *'run services describe'*) echo \"${FAKE_RUNTIME_SERVICE_ACCOUNT-mcp-runtime@example-project.iam.gserviceaccount.com}\" ;;\n"
        "  *'projects describe'*) echo '123456789' ;;\n"
        "  *'iam service-accounts describe'*) echo 'NOT_FOUND: service account missing' >&2; exit 1 ;;\n"
        "  *'tasks queues describe'*) if [[ \"${FAKE_QUEUE_EXISTS:-0}\" == 1 ]]; then exit 0; else echo 'NOT_FOUND: queue missing' >&2; exit 1; fi ;;\n"
        "  *'firestore indexes composite create'*'FAKE_INDEX_ERROR=1'*) echo 'PERMISSION_DENIED' >&2; exit 1 ;;\n"
        "  *'firestore indexes composite create'*) echo 'ALREADY_EXISTS: index already exists' >&2; exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    environment = {
        **{name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if name in os.environ},
        "PATH": os.pathsep.join((str(bin_dir), str(Path(sys.executable).parent), os.environ.get("PATH", ""))),
        "ROUTE_PYTHON": sys.executable,
        "ROUTE_EVENT_LOG": str(log),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west1",
        **overrides,
    }
    script = Path(__file__).resolve().parents[2] / "infra" / "deploy" / "core" / "helpers" / helper
    result = subprocess.run([bash, str(script), *args], cwd=script.parents[4], env=environment,
                            text=True, capture_output=True, check=False, timeout=30)
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, events


def test_dialog_operations_helper_uses_confirmed_url_and_scoped_iam(tmp_path: Path) -> None:
    base_url = "https://synthetic-mcp-uc.a.run.app"
    result, events = _run_deploy_helper(tmp_path, "deploy_dialog_operations.sh", base_url)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    output = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    task_invoker = "tlbrain-dialog-operations-invoker@example-project.iam.gserviceaccount.com"
    runtime = "mcp-runtime@example-project.iam.gserviceaccount.com"
    assert output == {
        "INTERNAL_DIALOG_OPERATIONS_URL": base_url + "/internal/dialog-operations",
        "DIALOG_OPERATIONS_QUEUE": "tlbrain-dialog-operations",
        "CLOUD_TASKS_OIDC_AUDIENCE": base_url,
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL": task_invoker,
        "OIDC_AUDIENCE": base_url,
        "OIDC_SERVICE_ACCOUNT_EMAIL": task_invoker,
    }
    queue = next(event for event in events if event[:4] == ["tasks", "queues", "create", "tlbrain-dialog-operations"])
    assert "--max-concurrent-dispatches=2" in queue
    assert "--location" in queue and queue[queue.index("--location") + 1] == "europe-west1"
    assert any(event[:3] == ["tasks", "queues", "describe"] for event in events)
    assert any(event[:4] == ["iam", "service-accounts", "create", "tlbrain-dialog-operations-invoker"] for event in events)

    bindings = [event for event in events if "add-iam-policy-binding" in event]
    role_bindings = {
        (tuple(event[0:3]), next(arg.split("=", 1)[1] for arg in event if arg.startswith("--role=")),
         next(arg.split("=", 1)[1] for arg in event if arg.startswith("--member=")))
        for event in bindings
    }
    assert (("run", "services", "add-iam-policy-binding"), "roles/run.invoker", f"serviceAccount:{task_invoker}") in role_bindings
    assert (("iam", "service-accounts", "add-iam-policy-binding"), "roles/iam.serviceAccountTokenCreator", "serviceAccount:service-123456789@gcp-sa-cloudtasks.iam.gserviceaccount.com") in role_bindings
    assert (("tasks", "queues", "add-iam-policy-binding"), "roles/cloudtasks.enqueuer", f"serviceAccount:{runtime}") in role_bindings
    assert (("iam", "service-accounts", "add-iam-policy-binding"), "roles/iam.serviceAccountUser", f"serviceAccount:{runtime}") in role_bindings
    assert any(event[:3] == ["projects", "add-iam-policy-binding", "example-project"]
               and any(arg.endswith("roles/datastore.user") for arg in event) for event in events)
    trace = " ".join(" ".join(event) for event in events)
    assert "roles/owner" not in trace and "roles/editor" not in trace
    assert "roles/cloudtasks.admin" not in trace and "roles/datastore.owner" not in trace
    assert not any("remove-iam-policy-binding" in event or "set-iam-policy" in event for event in events)


def test_dialog_operations_rejects_unconfirmed_or_non_base_urls_before_cloud_calls(tmp_path: Path) -> None:
    result, events = _run_deploy_helper(
        tmp_path, "deploy_dialog_operations.sh", "https://untrusted.invalid/internal/dialog-operations",
    )
    assert result.returncode != 0
    assert events == []


def test_dialog_operations_uses_default_compute_identity_when_service_account_is_empty(tmp_path: Path) -> None:
    result, events = _run_deploy_helper(
        tmp_path, "deploy_dialog_operations.sh", "https://synthetic-mcp-uc.a.run.app",
        FAKE_RUNTIME_SERVICE_ACCOUNT="",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    runtime = "123456789-compute@developer.gserviceaccount.com"
    assert any(event[:3] == ["tasks", "queues", "add-iam-policy-binding"]
               and any(arg.endswith(f"serviceAccount:{runtime}") for arg in event) for event in events)
    assert any(event[:3] == ["projects", "add-iam-policy-binding", "example-project"]
               and any(arg.endswith(f"serviceAccount:{runtime}") for arg in event)
               and "--role=roles/datastore.user" in event
               for event in events)
    assert any(event[:3] == ["iam", "service-accounts", "add-iam-policy-binding"]
               and any(arg.endswith(f"serviceAccount:{runtime}") for arg in event)
               and "--role=roles/iam.serviceAccountUser" in event
               for event in events)


def test_firestore_index_helper_adds_declared_indexes_and_exemptions_idempotently(tmp_path: Path) -> None:
    result, events = _run_deploy_helper(tmp_path, "deploy_firestore_indexes.sh")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    composites = [event for event in events if event[:5] == ["firestore", "indexes", "composite", "create", "--project"]]
    assert len(composites) == 6
    as_text = [" ".join(event) for event in composites]
    assert any("--collection-group telegram_sessions" in text and "reserved_operation_id" in text and "lock_expires_at" in text for text in as_text)
    assert any("--collection-group task_outbox" in text and "target_kind" in text and "schedule_at" in text for text in as_text)
    assert any("--collection-group dialog_operations" in text and "origin" in text and "updated_at" in text for text in as_text)
    assert any("--collection-group drafts" in text and "cleanup_state" in text and "updated_at" in text for text in as_text)
    exemptions = [event for event in events if event[:4] == ["firestore", "indexes", "fields", "update"]]
    assert len(exemptions) == 9
    assert {event[4] for event in exemptions} == {
        "text", "caption", "transcription_text", "parts", "raw_command_envelope", "body", "payload",
    }
    assert all("--disable-indexes" in event and "--collection-group" in " ".join(event) for event in exemptions)
    assert all("--project example-project" in " ".join(event) for event in events)
    trace = " ".join(" ".join(event) for event in events)
    assert "delete" not in trace and "clear" not in trace and "indexes import" not in trace
    assert "ALREADY_EXISTS" not in result.stderr

