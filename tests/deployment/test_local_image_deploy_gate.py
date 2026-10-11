"""Offline checks for the locally built image validation gate."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "mcp": ROOT / "infra" / "deploy" / "core" / "local_deploy_mcp.sh",
    "vector_sync": ROOT / "infra" / "deploy" / "core" / "local_deploy_vector_sync.sh",
}


def _script(service: str) -> str:
    return SCRIPTS[service].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("service", "image_var", "image_ref", "runtime_service", "dockerfile"),
    [
        ("mcp", "MCP_IMAGE", "gcr.io/${PROJECT_ID}/${MCP_SERVICE_NAME}", "mcp", "Dockerfile.mcp"),
        ("vector_sync", "SYNC_IMAGE", "gcr.io/${PROJECT_ID}/${VECTOR_SYNC_SERVICE_NAME}", "vector_sync", "Dockerfile.vector_sync"),
    ],
)
def test_local_script_builds_validates_pushes_and_deploys_same_amd64_image(
    service: str, image_var: str, image_ref: str, runtime_service: str, dockerfile: str,
) -> None:
    script = _script(service)
    build_at = script.index("docker build")
    gate_at = script.index("--image-mode")
    push_at = script.index("docker push")
    deploy_at = script.index("gcloud run deploy")
    gate = script[gate_at:push_at]

    assert script.startswith("#!/usr/bin/env bash\n\nset -euo pipefail")
    assert build_at < gate_at < push_at < deploy_at
    assert f'{image_var}="{image_ref}"' in script
    assert "docker build --platform linux/amd64" in script
    assert f'-f "${{REPO_ROOT}}/infra/docker/{dockerfile}"' in script
    assert f'-t "${{{image_var}}}"' in script[build_at:gate_at]
    assert 'bash "${REPO_ROOT}/infra/build/preflight.sh" --image-mode' in script
    assert f'--image "${{{image_var}}}"' in gate
    assert f"--service {runtime_service}" in gate
    assert "--platform linux/amd64" in gate
    assert f'docker push "${{{image_var}}}"' in script[push_at:deploy_at]
    assert f'--image "${{{image_var}}}"' in script[deploy_at:]


def test_local_deployment_settings_and_vector_queue_behavior_are_preserved() -> None:
    mcp = _script("mcp")
    vector = _script("vector_sync")
    assert '--max-instances "${MCP_MAX_INSTANCES}"' in mcp
    assert 'VECTOR_SYNC_URL="${VECTOR_SYNC_URL}"' in mcp
    assert 'VERSION="${VERSION}"' in mcp
    assert "--timeout 3600" in vector
    assert "--concurrency 1" in vector
    assert '--max-instances "${VECTOR_SYNC_MAX_INSTANCES}"' in vector
    assert 'gcloud tasks queues create "${VECTOR_SYNC_QUEUE}"' in vector
    assert 'gcloud tasks queues update "${VECTOR_SYNC_QUEUE}"' in vector
    for name in (
        "ROOT_FOLDER_URL", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
        "QDRANT_COLLECTION", "GOOGLE_REFRESH_TOKEN", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
    ):
        assert f'{name}="${{{name}}}"' in mcp
        assert f'{name}="${{{name}}}"' in vector


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash integration scenarios are disabled on Windows unless explicitly enabled")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _run_sandbox(tmp_path: Path, service: str, *, gate_fails: bool = False) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bash = _bash()
    repo = tmp_path / "repo"
    deploy_dir = repo / "infra" / "deploy" / "core"
    build_dir = repo / "infra" / "build"
    bin_dir = tmp_path / "bin"
    deploy_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    bin_dir.mkdir()
    script = deploy_dir / SCRIPTS[service].name
    shutil.copyfile(SCRIPTS[service], script)

    preflight = build_dir / "preflight.sh"
    preflight.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gate:%s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
        "if [[ \"${GATE_FAIL:-0}\" == 1 ]]; then exit 9; fi\n",
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
        "PROJECT_ID": "synthetic-project",
        "ROOT_FOLDER_URL": "https://synthetic.example/root",
        "GEMINI_API_KEY": "synthetic-gemini",
        "QDRANT_URL": "https://synthetic.example/qdrant",
        "QDRANT_API_KEY": "synthetic-qdrant",
        "QDRANT_COLLECTION": "synthetic-collection",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh",
        "GOOGLE_CLIENT_ID": "synthetic-client",
        "GOOGLE_CLIENT_SECRET": "synthetic-secret",
        "VERSION": "synthetic-version",
    })
    try:
        result = subprocess.run([bash, str(script)], env=environment, text=True, capture_output=True, check=False)
    except PermissionError:
        pytest.skip("this Windows host blocks Bash child processes")
    events = event_log.read_text(encoding="utf-8").splitlines() if event_log.exists() else []
    return result, events


@pytest.mark.parametrize("service", ["mcp", "vector_sync"])
def test_local_image_gate_precedes_push_and_deploy(tmp_path: Path, service: str) -> None:
    result, events = _run_sandbox(tmp_path, service)
    assert result.returncode == 0, result.stderr
    assert len(events) >= 4
    assert events[0].startswith("docker:build --platform linux/amd64")
    assert events[1].startswith("gate:--image-mode")
    assert events[2].startswith("docker:push ")
    assert any(event.startswith("gcloud:run deploy ") for event in events)


@pytest.mark.parametrize("service", ["mcp", "vector_sync"])
def test_local_image_gate_failure_stops_before_push_or_cloud_commands(tmp_path: Path, service: str) -> None:
    result, events = _run_sandbox(tmp_path, service, gate_fails=True)
    assert result.returncode != 0
    assert events[0].startswith("docker:build --platform linux/amd64")
    assert events[1].startswith("gate:--image-mode")
    assert not any(event.startswith("docker:push ") for event in events)
    assert not any(event.startswith("gcloud:") for event in events)
