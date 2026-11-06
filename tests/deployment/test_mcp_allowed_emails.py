"""Keep the email allowlist intact through both MCP deployment scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _bash_executable():
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        return str(git_bash) if git_bash.is_file() else None
    return shutil.which("bash")


def _bash_path(path: Path) -> str:
    value = path.as_posix()
    if os.name == "nt" and len(value) >= 3 and value[1:3] == ":/":
        return f"/{value[0].lower()}/{value[3:]}"
    return value


@pytest.mark.parametrize("script_name", ["deploy_mcp.sh", "local_deploy_mcp.sh"])
@pytest.mark.parametrize("allowed_email", ["", "first@example.test", "first@example.test,second@example.test"])
@pytest.mark.parametrize(
    ("region", "queue", "expected_region", "expected_queue"),
    [
        ("synthetic-region", "custom-sync-queue", "synthetic-region", "custom-sync-queue"),
        ("", "", "europe-west1", "tlbrain-vector-sync-queue"),
        (None, None, "europe-west1", "tlbrain-vector-sync-queue"),
    ],
)
def test_mcp_deploy_preserves_runtime_environment(
    tmp_path, script_name, allowed_email, region, queue, expected_region, expected_queue,
):
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash scenarios require an explicitly configured Windows runner")
    bash = _bash_executable()
    if not bash:
        pytest.skip("bash is unavailable")
    repo = tmp_path / "repo"
    scripts = repo / "infra/deploy/core"
    build = repo / "infra/build"
    scripts.mkdir(parents=True)
    build.mkdir(parents=True)
    helper_dir = scripts / "helpers"
    helper_dir.mkdir()
    script = scripts / script_name
    shutil.copyfile(ROOT / "infra/deploy/core" / script_name, script)
    shutil.copyfile(
        ROOT / "infra/deploy/core/helpers/deploy_dialog_operations.sh",
        helper_dir / "deploy_dialog_operations.sh",
    )
    (build / "preflight.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'sha256:%064d\\n' 0\n", encoding="utf-8",
    )
    args_log = tmp_path / "deploy-args.txt"
    environment = os.environ.copy()
    environment.update({
        "ALLOWED_EMAIL": allowed_email,
        "PROJECT_ID": "synthetic-project",
        "VERSION": "synthetic-version",
        "ROOT_FOLDER_URL": "https://drive.example/root", "GEMINI_API_KEY": "synthetic-gemini",
        "QDRANT_URL": "https://qdrant.example", "QDRANT_API_KEY": "synthetic-qdrant",
        "QDRANT_COLLECTION": "synthetic-collection", "GOOGLE_REFRESH_TOKEN": "synthetic-refresh",
        "GOOGLE_CLIENT_ID": "synthetic-client", "GOOGLE_CLIENT_SECRET": "synthetic-secret",
        "DEPLOY_ARGS_LOG": _bash_path(args_log),
    })
    if region is None:
        environment.pop("REGION", None)
    else:
        environment["REGION"] = region
    if queue is None:
        environment.pop("VECTOR_SYNC_QUEUE", None)
    else:
        environment["VECTOR_SYNC_QUEUE"] = queue
    for optional_config in (
        "TLDV_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USER_IDS",
    ):
        environment.pop(optional_config, None)
    if os.name == "nt":
        git_bin = str(Path(bash).parent)
        environment["PATH"] = git_bin + os.pathsep + environment.get("PATH", "")
    harness = r'''
set -euo pipefail
gcloud() {
    if [[ "${1:-} ${2:-}" == "run deploy" ]]; then
        printf '%s\n' "$@" > "$DEPLOY_ARGS_LOG"
    elif [[ "$*" == *"serviceAccountName"* ]]; then
        printf 'synthetic-runtime@synthetic-project.iam.gserviceaccount.com\n'
    elif [[ "${1:-} ${2:-}" == "projects describe" ]]; then
        printf '123456789012\n'
    else
        printf 'https://service.example\n'
    fi
}
docker() { :; }
export -f gcloud docker
bash "$1"
'''
    result = subprocess.run(
        [bash, "-c", harness, "mcp-runtime-env-test", _bash_path(script)],
        env=environment, text=True, capture_output=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    args = args_log.read_text(encoding="utf-8").splitlines()
    values = [args[index + 1] for index, arg in enumerate(args) if arg == "--set-env-vars"]
    assert len(values) == 1
    assert values[0].startswith("^|^")
    deployed_variables = dict(
        item.split("=", 1) for item in values[0][len("^|^"):].split("|")
    )
    payload_names = [item.split("=", 1)[0] for item in values[0][len("^|^"):].split("|")]
    assert payload_names.count("REGION") == 1
    assert payload_names.count("VECTOR_SYNC_QUEUE") == 1
    assert deployed_variables["ALLOWED_EMAIL"] == allowed_email
    assert deployed_variables["REGION"] == expected_region
    assert deployed_variables["VECTOR_SYNC_QUEUE"] == expected_queue
    for key in ("ROOT_FOLDER_URL", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "VERSION"):
        assert deployed_variables[key] == environment[key]
