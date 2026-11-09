"""Offline checks that common Sync Checker deployment ignores TL;DV config."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
SYNC_CHECKER_SCRIPT = APP_ROOT / "infra/deploy/core/deploy_sync_checker.sh"
FINGERPRINT = "a" * 64


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash integration scenarios are disabled on Windows unless the runner explicitly enables them")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def test_sync_checker_deploy_ignores_tldv_only_environment(tmp_path: Path) -> None:
    bash = _bash()
    repo = tmp_path / "repo"
    deploy_dir = repo / "infra" / "deploy" / "core"
    build_dir = repo / "infra" / "build"
    bin_dir = tmp_path / "bin"
    deploy_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    bin_dir.mkdir()
    script = deploy_dir / SYNC_CHECKER_SCRIPT.name
    shutil.copyfile(SYNC_CHECKER_SCRIPT, script)

    preflight = build_dir / "preflight.sh"
    preflight.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'preflight:%s:tldv-key=%s:tldv-url=%s\\n' \"$*\" "
        "\"${TLDV_API_KEY+present}\" \"${TLDV_RECONCILIATION_URL+present}\" >> \"$EVENT_LOG\"\n"
        f"if [[ \"${{1:-}}\" != --verify-source ]]; then printf 'Artifact fingerprint: {FINGERPRINT}\\n'; fi\n",
        encoding="utf-8",
    )
    preflight.chmod(0o755)

    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gcloud:%s\\n' \"$*\" >> \"$EVENT_LOG\"\n"
        "if [[ \"$*\" == *'services describe'* ]]; then printf 'https://sync.test/run\\n'; fi\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)

    event_log = tmp_path / "events.log"
    environment = os.environ.copy()
    environment.update({
        "PATH": str(bin_dir) + os.pathsep + environment.get("PATH", ""),
        "EVENT_LOG": str(event_log),
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "ROOT_FOLDER_URL": "https://synthetic.example/root",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh",
        "GOOGLE_CLIENT_ID": "synthetic-client",
        "GOOGLE_CLIENT_SECRET": "synthetic-secret",
        "TLDV_API_KEY": "synthetic-tldv-secret",
        "TLDV_RECONCILIATION_URL": "https://stale-tldv.test/run",
    })
    result = subprocess.run([bash, str(script)], env=environment, text=True, capture_output=True, check=False)
    runner_output = (result.stdout + result.stderr).replace("\x00", "")
    if any(marker in runner_output for marker in ("E_ACCESSDENIED", "NtCreateDirectoryObject", "0xC0000022")):
        pytest.skip("MXC blocks child Bash with E_ACCESSDENIED / NtCreateDirectoryObject")

    assert result.returncode == 0, runner_output
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert any("tldv-key=:tldv-url=" in event for event in events if event.startswith("preflight:"))
    deploy = next(event for event in events if event.startswith("gcloud:functions deploy "))
    assert "TLDV_API_KEY" not in deploy
    assert "TLDV_RECONCILIATION_URL" not in deploy
    assert "synthetic-tldv-secret" not in deploy
    assert "stale-tldv.test" not in deploy
    assert any(event.startswith("gcloud:scheduler jobs create http ") for event in events)
