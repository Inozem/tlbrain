"""Offline checks for the TL;DV Import image validation gates."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
IMPORT_SCRIPTS = {
    "registry": APP_ROOT / "infra/deploy/connectors/tldv/deploy_import.sh",
    "local": APP_ROOT / "infra/deploy/connectors/tldv/local_deploy_import.sh",
}
REPOSITORY = "inozem/tlbrain-tldv-import"
DIGEST = "sha256:" + "b" * 64


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash integration scenarios are disabled on Windows unless the runner explicitly enables them")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _sandbox(
    tmp_path: Path, mode: str, *, gate_fails: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bash = _bash()
    repo = tmp_path / "repo"
    deploy_dir = repo / "infra" / "deploy" / "connectors" / "tldv"
    build_dir = repo / "infra" / "build"
    bin_dir = tmp_path / "bin"
    deploy_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    bin_dir.mkdir()
    script = deploy_dir / IMPORT_SCRIPTS[mode].name
    shutil.copyfile(IMPORT_SCRIPTS[mode], script)

    preflight = build_dir / "preflight.sh"
    preflight.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'preflight:%s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
        "if [[ \"${GATE_FAIL:-0}\" == 1 ]]; then exit 8; fi\n"
        "if [[ \"${1:-}\" == --verify-mode ]]; then printf '%s\\n' \"${VERIFY_RESULT}\"; fi\n",
        encoding="utf-8",
    )
    preflight.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'docker:%s\\n' \"$*\" >> \"$EVENT_LOG\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gcloud:%s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
        "case \"$*\" in *'services describe'*) printf 'https://service.example\\n' ;; esac\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)

    event_log = tmp_path / "events.log"
    environment = os.environ.copy()
    environment.update({
        "PATH": str(bin_dir) + os.pathsep + environment.get("PATH", ""),
        "EVENT_LOG": str(event_log),
        "GATE_FAIL": "1" if gate_fails else "0",
        "VERIFY_RESULT": DIGEST,
        "VERSION": "v2026.10.11",
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "ROOT_FOLDER_URL": "https://synthetic.example/root",
        "GEMINI_API_KEY": "synthetic-gemini",
        "TLDV_API_KEY": "synthetic-tldv",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh",
        "GOOGLE_CLIENT_ID": "synthetic-client",
        "GOOGLE_CLIENT_SECRET": "synthetic-secret",
    })
    result = subprocess.run([bash, str(script)], env=environment, text=True, capture_output=True, check=False)
    runner_output = (result.stdout + result.stderr).replace("\x00", "")
    if any(marker in runner_output for marker in ("E_ACCESSDENIED", "NtCreateDirectoryObject", "0xC0000022")):
        pytest.skip("MXC blocks child Bash with E_ACCESSDENIED / NtCreateDirectoryObject")
    events = event_log.read_text(encoding="utf-8").splitlines() if event_log.exists() else []
    return result, events, environment["PATH"].split(os.pathsep)


def test_registry_deploy_verifies_selected_version_and_pins_returned_digest(tmp_path: Path) -> None:
    result, events, _path = _sandbox(tmp_path, "registry")

    assert result.returncode == 0, result.stderr
    assert events[0] == (
        "preflight:--verify-mode --repository inozem/tlbrain-tldv-import --version v2026.10.11 "
        "--service tldv_import --target app.main:app --platform linux/amd64 "
        "--required-check compile --required-check imports --required-check startup"
    )
    deploy = next(event for event in events if event.startswith("gcloud:run deploy "))
    assert f"--image {REPOSITORY}@{DIGEST}" in deploy
    assert f"{REPOSITORY}:v2026.10.11" not in deploy
    assert 0 < next(
        index for index, event in enumerate(events) if event.startswith("gcloud:run services describe")
    )
    assert not any(event.startswith("docker:") for event in events)


def test_registry_verification_failure_prevents_cloud_commands(tmp_path: Path) -> None:
    result, events, _path = _sandbox(tmp_path, "registry", gate_fails=True)

    assert result.returncode != 0
    assert events[0].startswith("preflight:--verify-mode ")
    assert not any(event.startswith("gcloud:") for event in events)
    assert not any(event.startswith("docker:") for event in events)


def test_local_gate_runs_after_amd64_build_and_before_push_and_deploy(tmp_path: Path) -> None:
    result, events, _path = _sandbox(tmp_path, "local")

    assert result.returncode == 0, result.stderr
    assert events[0] == (
        "docker:build --platform linux/amd64 -f "
        f"{tmp_path / 'repo' / 'infra' / 'docker' / 'Dockerfile.tldv_import'} "
        f"-t gcr.io/synthetic-project/tlbrain-tldv-import {tmp_path / 'repo'}"
    )
    assert events[1] == (
        "preflight:--image-mode --image gcr.io/synthetic-project/tlbrain-tldv-import "
        "--service tldv_import --platform linux/amd64"
    )
    assert events[2] == "docker:push gcr.io/synthetic-project/tlbrain-tldv-import"
    assert any(event.startswith("gcloud:run deploy ") for event in events)
    deploy = next(event for event in events if event.startswith("gcloud:run deploy "))
    assert "--image gcr.io/synthetic-project/tlbrain-tldv-import" in deploy


def test_local_gate_failure_prevents_push_and_cloud_commands(tmp_path: Path) -> None:
    result, events, _path = _sandbox(tmp_path, "local", gate_fails=True)

    assert result.returncode != 0
    assert events[0].startswith("docker:build --platform linux/amd64 ")
    assert events[1].startswith("preflight:--image-mode ")
    assert not any(event.startswith("docker:push ") for event in events)
    assert not any(event.startswith("gcloud:") for event in events)


def test_import_scripts_keep_existing_settings_and_no_cache_requirement() -> None:
    registry = IMPORT_SCRIPTS["registry"].read_text(encoding="utf-8")
    local = IMPORT_SCRIPTS["local"].read_text(encoding="utf-8")
    for source in (registry, local):
        assert 'PROJECT_ID=${PROJECT_ID:-tlbrain-prod}' in source
        assert 'REGION=${REGION:-europe-west1}' in source
        assert 'TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}' in source
        assert 'TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}' in source
        assert 'VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}' in source
        assert 'VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}' in source
        assert 'TLDV_IMPORT_MAX_INSTANCES=${TLDV_IMPORT_MAX_INSTANCES:-2}' in source
        assert 'GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}"' in source
        assert '--max-instances "${TLDV_IMPORT_MAX_INSTANCES}"' in source
        assert "--required-check cache" not in source
        assert "set -euo pipefail" in source
    assert 'docker build --platform linux/amd64' in local


def test_tldv_and_top_level_wrapper_routes_prompts_tokens_and_order_are_preserved() -> None:
    connector = APP_ROOT / "infra/deploy/connectors/deploy_tldv.sh"
    local_connector = APP_ROOT / "infra/deploy/connectors/local_deploy_tldv.sh"
    top_level = APP_ROOT / "infra/deploy/deploy.sh"
    local_top_level = APP_ROOT / "infra/deploy/local_deploy.sh"
    connector_source = connector.read_text(encoding="utf-8")
    local_connector_source = local_connector.read_text(encoding="utf-8")
    top_source = top_level.read_text(encoding="utf-8")
    local_top_source = local_top_level.read_text(encoding="utf-8")

    for source, import_script in (
        (connector_source, 'bash "${SCRIPT_DIR}/tldv/deploy_import.sh"'),
        (local_connector_source, 'bash "${SCRIPT_DIR}/tldv/local_deploy_import.sh"'),
    ):
        assert 'if [ -z "${GOOGLE_REFRESH_TOKEN}" ]; then' in source
        assert 'if [ "${SKIP_CONFIRM:-}" != "1" ]; then' in source
        assert 'read -p "Continue? (y/n): " CONFIRM' in source
        assert "gcloud config set project \"${PROJECT_ID}\"" in source
        assert "gcloud tasks queues create \"${TLDV_IMPORT_QUEUE}\"" in source
        assert source.index(import_script) < source.index('bash "${SCRIPT_DIR}/tldv/deploy_webhook.sh"')
        assert source.index('bash "${SCRIPT_DIR}/tldv/deploy_webhook.sh"') < source.index(
            'bash "${SCRIPT_DIR}/tldv/deploy_reconciliation.sh"'
        )

    assert 'read -p "Continue deploy? (y/n): " CONFIRM' in top_source
    assert 'bash infra/deploy/setup_tokens.sh' in top_source
    assert 'export $(grep -v \'^#\' .env | xargs)' in top_source
    assert 'SKIP_CONFIRM=1 bash infra/deploy/connectors/deploy_tldv.sh' in top_source
    assert top_source.index('bash infra/deploy/core/deploy_vector_sync.sh') < top_source.index(
        'SKIP_CONFIRM=1 bash infra/deploy/connectors/deploy_tldv.sh'
    ) < top_source.index('bash infra/deploy/core/deploy_sync_checker.sh') < top_source.index(
        'bash infra/deploy/core/deploy_mcp.sh'
    )
    assert 'read -p "Continue deploy? (y/n): " CONFIRM' in local_top_source
    assert 'SKIP_CONFIRM=1 bash infra/deploy/connectors/local_deploy_tldv.sh' in local_top_source
    assert local_top_source.index('bash infra/deploy/core/local_deploy_vector_sync.sh') < local_top_source.index(
        'SKIP_CONFIRM=1 bash infra/deploy/connectors/local_deploy_tldv.sh'
    ) < local_top_source.index('bash infra/deploy/core/deploy_sync_checker.sh') < local_top_source.index(
        'bash infra/deploy/core/local_deploy_mcp.sh'
    )
