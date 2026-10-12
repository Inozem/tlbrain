"""Offline tests for the ordinary image deployment evidence gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "mcp": ROOT / "infra" / "deploy" / "core" / "deploy_mcp.sh",
    "vector_sync": ROOT / "infra" / "deploy" / "core" / "deploy_vector_sync.sh",
}
DIGEST = "sha256:" + "a" * 64


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip(
            "Bash integration scenarios are disabled on Windows unless the runner explicitly enables them"
        )
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _sandbox(tmp_path: Path, service: str, *, verifier_fails: bool = False) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
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

    verifier = build_dir / "preflight.sh"
    verifier.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$VERIFY_ARGS_LOG\"\n"
        "if [[ \"${VERIFY_FAIL:-0}\" == 1 ]]; then exit 1; fi\n"
        "printf '%s\\n' \"${VERIFY_RESULT}\"\n",
        encoding="utf-8",
    )
    verifier.chmod(0o755)
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$GCLOUD_LOG\"\n"
        "case \"$*\" in *'services describe'*) printf 'https://service.example\\n' ;; esac\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    args_log = tmp_path / "verifier-args.txt"
    gcloud_log = tmp_path / "gcloud.log"
    environment = os.environ.copy()
    environment.update({
        "PATH": str(bin_dir) + os.pathsep + environment.get("PATH", ""),
        "VERSION": "v2026.10.11",
        "VERIFY_RESULT": DIGEST,
        "VERIFY_FAIL": "1" if verifier_fails else "0",
        "VERIFY_ARGS_LOG": str(args_log),
        "GCLOUD_LOG": str(gcloud_log),
        "ROOT_FOLDER_URL": "https://drive.example/root",
        "GEMINI_API_KEY": "synthetic-gemini",
        "QDRANT_URL": "https://qdrant.example",
        "QDRANT_API_KEY": "synthetic-qdrant",
        "QDRANT_COLLECTION": "synthetic",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh",
        "GOOGLE_CLIENT_ID": "synthetic-client",
        "GOOGLE_CLIENT_SECRET": "synthetic-secret",
    })
    result = subprocess.run([bash, str(script)], env=environment, text=True, capture_output=True, check=False)
    verifier_args = args_log.read_text(encoding="utf-8").splitlines() if args_log.exists() else []
    gcloud_calls = gcloud_log.read_text(encoding="utf-8").splitlines() if gcloud_log.exists() else []
    return result, verifier_args, gcloud_calls


@pytest.mark.parametrize(
    ("service", "repository", "runtime_service", "target"),
    [
        ("mcp", "inozem/tlbrain-mcp", "mcp", "services.mcp.app.main:app"),
        ("vector_sync", "inozem/tlbrain-vector-sync", "vector_sync", "services.vector_sync.app.main:app"),
    ],
)
def test_deploy_verifies_selected_tag_then_uses_returned_digest(
    tmp_path: Path, service: str, repository: str, runtime_service: str, target: str,
) -> None:
    result, verifier_args, gcloud_calls = _sandbox(tmp_path, service)

    assert result.returncode == 0, result.stderr
    assert verifier_args == [
        "--verify-mode", "--repository", repository, "--version", "v2026.10.11",
        "--service", runtime_service, "--target", target, "--platform", "linux/amd64",
        "--required-check", "compile", "--required-check", "imports",
        "--required-check", "startup", "--required-check", "cache",
    ]
    deploy = next(call for call in gcloud_calls if call.startswith("run deploy "))
    assert f"--image {repository}@{DIGEST}" in deploy
    assert f"{repository}:v2026.10.11" not in deploy


@pytest.mark.parametrize("service", ["mcp", "vector_sync"])
def test_verification_failure_stops_before_any_cloud_command(tmp_path: Path, service: str) -> None:
    result, verifier_args, gcloud_calls = _sandbox(tmp_path, service, verifier_fails=True)

    assert result.returncode != 0
    assert "--verify-mode" in verifier_args
    assert gcloud_calls == []


@pytest.mark.parametrize(
    ("service", "repository", "runtime_service", "target", "image_var", "digest_var"),
    [
        ("mcp", "inozem/tlbrain-mcp", "mcp", "services.mcp.app.main:app", "MCP_IMAGE", "MCP_DIGEST"),
        ("vector_sync", "inozem/tlbrain-vector-sync", "vector_sync", "services.vector_sync.app.main:app", "SYNC_IMAGE", "SYNC_DIGEST"),
    ],
)
def test_each_script_declares_registry_gate_before_digest_deploy(
    service: str, repository: str, runtime_service: str, target: str, image_var: str, digest_var: str,
) -> None:
    script = SCRIPTS[service].read_text(encoding="utf-8")
    verify_at = script.index("--verify-mode")
    deploy_at = script.index("gcloud run deploy")
    verify_block = script[verify_at:deploy_at]

    assert script.startswith("#!/usr/bin/env bash\n\nset -euo pipefail")
    assert verify_at < deploy_at
    assert f'--repository "{repository}"' in verify_block
    assert '--version "${VERSION}"' in verify_block
    assert f'--service {runtime_service}' in verify_block
    assert f'--target "{target}"' in verify_block
    assert "--platform linux/amd64" in verify_block
    for check in ("compile", "imports", "startup", "cache"):
        assert f"--required-check {check}" in verify_block
    assert f'{image_var}="{repository}@${{{digest_var}}}"' in verify_block
    deploy_block = script[deploy_at:]
    assert f'--image "${{{image_var}}}"' in deploy_block
    assert f'{repository}:${{VERSION}}' not in deploy_block


def test_deploy_settings_are_preserved() -> None:
    mcp = SCRIPTS["mcp"].read_text(encoding="utf-8")
    vector = SCRIPTS["vector_sync"].read_text(encoding="utf-8")
    assert '--max-instances "${MCP_MAX_INSTANCES}"' in mcp
    assert 'VECTOR_SYNC_URL="${VECTOR_SYNC_URL}"' in mcp
    assert 'VERSION="${VERSION}"' in mcp
    assert "--timeout 3600" in vector
    assert "--concurrency 1" in vector
    assert '--max-instances "${VECTOR_SYNC_MAX_INSTANCES}"' in vector
    assert 'gcloud tasks queues create "${VECTOR_SYNC_QUEUE}"' in vector
    assert 'gcloud tasks queues update "${VECTOR_SYNC_QUEUE}"' in vector
