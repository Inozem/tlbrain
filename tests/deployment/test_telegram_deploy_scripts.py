"""Direct offline contract tests for standalone Telegram deployment scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "synthetic-telegram-token-do-not-log"


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash route execution is unavailable on native Windows")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _fixture(tmp_path: Path):
    root = tmp_path / "checkout"
    shutil.copytree(APP_ROOT / "infra" / "deploy" / "connectors", root / "infra" / "deploy" / "connectors")
    shutil.copytree(APP_ROOT / "infra" / "docker", root / "infra" / "docker")
    (root / "infra" / "build").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "events.jsonl"
    py = sys.executable
    _write(root / "infra" / "build" / "preflight.sh", """#!/usr/bin/env bash
set -euo pipefail
"$REAL_PYTHON" -c 'import json,os,sys; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["preflight",*sys.argv[1:]])+"\\n")' "$@"
"$REAL_PYTHON" -c 'import json,os; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["child-env","preflight",bool(os.environ.get("TELEGRAM_BOT_TOKEN")),bool(os.environ.get("TELEGRAM_ALLOWED_USER_IDS"))])+"\\n")'
if [[ "$1" == --verify-mode ]]; then printf 'sha256:%064d\\n' 7; fi
""", True)
    _write(bindir / "gcloud", """#!/usr/bin/env bash
set -euo pipefail
"$REAL_PYTHON" -c 'import json,os,sys; a=sys.argv[1:]; f=open(os.environ["EVENT_LOG"],"a"); f.write(json.dumps(["gcloud",*a])+"\\n"); f.write(json.dumps(["child-env","gcloud",bool(os.environ.get("TELEGRAM_BOT_TOKEN")),bool(os.environ.get("TELEGRAM_ALLOWED_USER_IDS"))])+"\\n");
if a[:4]==["run","services","describe",os.environ["TELEGRAM_SERVICE_NAME"]] and os.environ.get("SERVICE_MISSING"): print("NOT_FOUND",file=sys.stderr); raise SystemExit(1)
elif a[:4]==["run","services","describe",os.environ["TELEGRAM_SERVICE_NAME"]] and "status.url" in " ".join(a): print("https://telegram.synthetic.invalid")
elif a[:4]==["run","services","describe",os.environ["TELEGRAM_SERVICE_NAME"]] and "serviceAccountName" in " ".join(a): print("telegram-runtime@synthetic-project.iam.gserviceaccount.com")
elif a[:2]==["projects","describe"]: print("123456789012")
elif a[:3]==["iam","service-accounts","describe"]: raise SystemExit(1)
elif a[:2]==["config","get-value"]: print("operator@example.test")
elif a[:2]==["auth","print-identity-token"]: print("synthetic-deployer-id-token")
if "--env-vars-file" in a:
 p=a[a.index("--env-vars-file")+1]; d=json.load(open(p)); mode=oct(os.stat(p).st_mode & 0o777); f.write(json.dumps(["env-file",d.get("TELEGRAM_BOT_TOKEN")=="synthetic-telegram-token-do-not-log",sorted(d),mode,p])+"\\n")' "$@"
""", True)
    _write(bindir / "docker", """#!/usr/bin/env bash
set -euo pipefail
"$REAL_PYTHON" -c 'import json,os,sys; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["docker",*sys.argv[1:]])+"\\n")' "$@"
"$REAL_PYTHON" -c 'import json,os; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["child-env","docker",bool(os.environ.get("TELEGRAM_BOT_TOKEN")),bool(os.environ.get("TELEGRAM_ALLOWED_USER_IDS"))])+"\\n")'
""", True)
    _write(bindir / "python3", """#!/usr/bin/env bash
set -euo pipefail
"$REAL_PYTHON" -c 'import json,os,sys; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["python-argv",bool(os.environ.get("TELEGRAM_BOT_TOKEN")),bool(os.environ.get("TELEGRAM_ALLOWED_USER_IDS")),*sys.argv[1:]])+"\\n")' "$@"
if [[ -n "${TELEGRAM_DISABLE_URL:-}" ]]; then
  "$REAL_PYTHON" -c 'import json,os; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["disable",os.environ.get("TELEGRAM_DISABLE_URL"),bool(os.environ.get("TELEGRAM_DISABLE_ID_TOKEN"))])+"\\n")'
  [[ "${DISABLE_FAIL:-0}" == 1 ]] && exit 1
  exit 0
fi
case "$*" in
  *register_telegram_webhook*)
    "$REAL_PYTHON" -c 'import json,os; open(os.environ["EVENT_LOG"],"a").write(json.dumps(["webhook",bool(os.environ.get("TELEGRAM_BOT_TOKEN")),os.environ.get("TELEGRAM_WEBHOOK_URL"),os.environ.get("PYTHONPATH")])+"\\n"); print("Telegram webhook registration succeeded")'
    ;;
  *) exec "$REAL_PYTHON" "$@" ;;
esac
""", True)
    env = {
        **{k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if k in os.environ},
        "PATH": os.pathsep.join((str(bindir), str(Path(py).parent), os.environ.get("PATH", ""))),
        "REAL_PYTHON": py,
        "EVENT_LOG": str(log),
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "VERSION": "v-synthetic-1",
        "TMPDIR": str(tmp_path),
        "TELEGRAM_SERVICE_NAME": "synthetic-telegram",
        "TELEGRAM_BOT_TOKEN": TOKEN,
        "TELEGRAM_ALLOWED_USER_IDS": "101,202",
        "TZ": "Asia/Jerusalem",
        "SKIP_CONFIRM": "1",
    }
    return root, log, env


def _run(tmp_path: Path, script: str, overrides: dict[str, str] | None = None):
    root, log, env = _fixture(tmp_path)
    env.update(overrides or {})
    outside = tmp_path / "outside-checkout"
    outside.mkdir()
    command = [_bash(), str(root / "infra" / "deploy" / "connectors" / script)]
    if script == "telegram_deploy_common.sh":
        command.append("off")
    result = subprocess.run(
        command,
        cwd=outside, env=env, text=True, capture_output=True, timeout=90, check=False,
    )
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, events, root


@pytest.mark.parametrize("script,local", [("deploy_telegram.sh", False), ("local_deploy_telegram.sh", True)])
def test_standalone_deploy_has_telegram_gate_iam_callbacks_and_webhook_last(tmp_path: Path, script: str, local: bool):
    result, events, root = _run(tmp_path, script)
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    deploy = next(event for event in events if event[0] == "gcloud" and event[1:3] == ["run", "deploy"])
    assert deploy[3] == "synthetic-telegram"
    for flag, value in (("--min", "0"), ("--max", "2"), ("--concurrency", "8"), ("--timeout", "540")):
        assert deploy[deploy.index(flag) + 1] == value
    assert "--allow-unauthenticated" in deploy
    gate = next(event for event in events if event[0] == "preflight")
    if local:
        build = next(i for i, event in enumerate(events) if event[:2] == ["docker", "build"])
        gate_index = events.index(gate)
        push = next(i for i, event in enumerate(events) if event[:2] == ["docker", "push"])
        assert build < gate_index < push < events.index(deploy)
        assert "--image-mode" in gate
        dockerfile = root / "infra" / "docker" / "Dockerfile.telegram"
        assert dockerfile.as_posix() in [Path(arg).as_posix() for arg in events[build]]
    else:
        assert "--verify-mode" in gate
        required_checks = [gate[index + 1] for index, arg in enumerate(gate[:-1]) if arg == "--required-check"]
        assert required_checks == ["compile", "imports", "startup"]
        assert "cache" not in required_checks
        assert deploy[deploy.index("--image") + 1] == "inozem/tlbrain-telegram@sha256:" + "0" * 63 + "7"
        assert not any(event[0] == "docker" for event in events)

    update = next(event for event in events if event[0] == "gcloud" and event[1:4] == ["run", "services", "update"])
    settings = " ".join(update)
    for setting in (
        "TELEGRAM_CONTROL_URL=https://telegram.synthetic.invalid/internal/dispatch",
        "TELEGRAM_DELIVERY_URL=https://telegram.synthetic.invalid/internal/deliver",
        "TELEGRAM_CONTROL_QUEUE=tlbrain-telegram-control",
        "CLOUD_TASKS_OIDC_AUDIENCE=https://telegram.synthetic.invalid",
        "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=tlbrain-telegram-task-invoker@synthetic-project.iam.gserviceaccount.com",
        "OIDC_AUDIENCE=https://telegram.synthetic.invalid",
        "OIDC_SERVICE_ACCOUNT_EMAIL=tlbrain-telegram-task-invoker@synthetic-project.iam.gserviceaccount.com",
        "TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL=tlbrain-telegram-deployer@synthetic-project.iam.gserviceaccount.com",
        "TZ=Asia/Jerusalem",
    ):
        assert setting in settings
    assert events.index(deploy) < next(i for i, e in enumerate(events) if e[0] == "gcloud" and "status.url" in " ".join(e)) < events.index(update)
    webhook = next(event for event in events if event[0] == "webhook")
    assert events.index(update) < events.index(webhook)
    assert webhook[1:3] == [True, "https://telegram.synthetic.invalid/telegram/webhook"]
    assert Path(webhook[3].split(os.pathsep)[0]).resolve() == root.resolve()
    for queue, limit in (("tlbrain-dialog-operations", "2"), ("tlbrain-telegram-control", "4")):
        assert any(e[0] == "gcloud" and e[1:4] == ["tasks", "queues", "update"] and e[4] == queue
                   and f"--max-concurrent-dispatches={limit}" in e for e in events)
    cloud = [e for e in events if e[0] == "gcloud"]
    assert any(e[1:4] == ["iam", "service-accounts", "create"] and e[4] == "tlbrain-telegram-task-invoker" for e in cloud)
    assert any(e[1:4] == ["iam", "service-accounts", "create"] and e[4] == "tlbrain-telegram-deployer" for e in cloud)
    assert any(e[1:4] == ["run", "services", "add-iam-policy-binding"] and e[4] == "synthetic-telegram" for e in cloud)
    assert any("roles/iam.serviceAccountTokenCreator" in " ".join(e) for e in cloud)
    assert any("tlbrain-telegram-deployer@synthetic-project.iam.gserviceaccount.com" in " ".join(e)
               and "user:operator@example.test" in " ".join(e)
               and "roles/iam.serviceAccountTokenCreator" in " ".join(e) for e in cloud)
    assert any("roles/iam.serviceAccountUser" in " ".join(e) for e in cloud)
    queue_grants = [e for e in cloud if "roles/cloudtasks.enqueuer" in " ".join(e)]
    assert len(queue_grants) == 2
    assert all("serviceAccount:telegram-runtime@synthetic-project.iam.gserviceaccount.com" in " ".join(e) for e in queue_grants)
    assert any("roles/datastore.user" in " ".join(e) for e in cloud)
    assert not any("roles/owner" in " ".join(e) or "roles/editor" in " ".join(e) for e in cloud)
    env_file = next(event for event in events if event[0] == "env-file")
    assert env_file[1:3] == [True, ["TELEGRAM_ALLOWED_USER_IDS", "TELEGRAM_BOT_TOKEN"]]
    unrelated_env = [event for event in events if event[0] == "child-env"]
    assert unrelated_env
    assert all(event[2:4] == [False, False] for event in unrelated_env)
    python_calls = [event for event in events if event[0] == "python-argv"]
    env_writer = next(event for event in python_calls if "json.dump" in " ".join(map(str, event)))
    webhook_writer = next(event for event in python_calls if "register_telegram_webhook" in " ".join(map(str, event)))
    assert env_writer[1:3] == [True, True]
    assert webhook_writer[1:3] == [True, False]
    if os.name == "nt":
        assert "chmod 600" in (root / "infra" / "deploy" / "connectors" / "telegram_deploy_common.sh").read_text(encoding="utf-8")
    else:
        assert env_file[3] == "0o600"
    assert TOKEN not in json.dumps(events) + result.stdout + result.stderr
    assert not list(root.parent.glob("tlbrain-telegram-env.*"))


@pytest.mark.parametrize("failure", [False, True])
def test_off_confirms_runtime_delete_before_removing_credentials(tmp_path: Path, failure: bool):
    result, events, _root = _run(tmp_path, "telegram_deploy_common.sh", {
        "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ALLOWED_USER_IDS": "",
        "DISABLE_FAIL": "1" if failure else "0",
    })
    disable_index = next(i for i, event in enumerate(events) if event[0] == "disable")
    token_command = next(event for event in events
                         if event[0] == "gcloud" and event[1:3] == ["auth", "print-identity-token"])
    assert "--include-email" in token_command
    removes = [i for i, event in enumerate(events)
               if event[0] == "gcloud" and "--remove-env-vars" in event]
    assert bool(result.returncode) is failure
    assert events[disable_index][1:] == ["https://telegram.synthetic.invalid/internal/disable-webhook", True]
    assert bool(removes) is not failure
    if removes:
        assert disable_index < removes[0]
        assert "TELEGRAM_BOT_TOKEN,TELEGRAM_ALLOWED_USER_IDS" in events[removes[0]]
    assert "synthetic-deployer-id-token" not in result.stdout + result.stderr + json.dumps(events)


def test_off_skips_cleanly_when_runtime_service_is_absent(tmp_path: Path):
    result, events, _root = _run(tmp_path, "telegram_deploy_common.sh", {
        "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ALLOWED_USER_IDS": "", "SERVICE_MISSING": "1",
    })
    assert result.returncode == 0
    assert "absent" in result.stdout
    assert not any(event[0] in {"disable", "gcloud"} and "--remove-env-vars" in event for event in events)


@pytest.mark.parametrize("script", ["deploy_telegram.sh", "local_deploy_telegram.sh"])
@pytest.mark.parametrize("overrides", [
    {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_ALLOWED_USER_IDS": " \t "},
    {"TELEGRAM_BOT_TOKEN": " \t ", "TELEGRAM_ALLOWED_USER_IDS": "101"},
])
def test_partial_or_blank_config_fails_before_preflight_docker_or_gcloud(tmp_path: Path, script: str, overrides: dict[str, str]):
    result, events, _root = _run(tmp_path, script, overrides)
    assert result.returncode != 0
    assert "TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS" in result.stderr
    assert not any(event[0] in {"gcloud", "docker", "preflight", "webhook"} for event in events)
    assert TOKEN not in result.stdout + result.stderr + json.dumps(events)
