"""Integration proofs that failed source gates cannot reach publish commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = APP_ROOT / "infra" / "build"
WORKFLOW = APP_ROOT / ".github" / "workflows" / "build.yml"
GATES = {
    "offline-regression",
    "source-validation",
    "image-validation",
    "native-bootstrap-smoke",
}
PUBLISH_JOBS = {
    "build-mcp": "mcp",
    "build-vector-sync": "vector_sync",
    "build-tldv-import": "tldv_import",
}
REQUIRED_PATHS = {
    "core/**", "services/**", "infra/**", "tests/**", "migrations/**",
    "requirements*.txt", "**/requirements*.txt", "pytest.ini", ".github/workflows/**",
}


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash deployment-script execution is unavailable on native Windows")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _bash_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt" and len(value) >= 3 and value[1:3] == ":/":
        return "/" + value[0].lower() + value[2:]
    return value


def _new_checkout(tmp_path: Path, case: str | None = None) -> Path:
    root = tmp_path / "synthetic-checkout"
    root.mkdir()
    shutil.copytree(APP_ROOT / "core", root / "core", ignore=shutil.ignore_patterns(
        ".env", ".env.*", ".build", "__pycache__", ".pytest_cache", ".venv*",
    ))
    shutil.copytree(APP_ROOT / "services", root / "services", ignore=shutil.ignore_patterns(
        ".env", ".env.*", ".build", "__pycache__", ".pytest_cache", ".venv*",
    ))
    shutil.copytree(APP_ROOT / "infra" / "build", root / "infra" / "build", ignore=shutil.ignore_patterns(
        ".env", ".env.*", "__pycache__", ".pytest_cache",
    ))
    shutil.copytree(APP_ROOT / "infra" / "deploy" / "core", root / "infra" / "deploy" / "core")
    (root / "infra" / "docker").mkdir(parents=True)
    shutil.copy2(APP_ROOT / "infra" / "docker" / "Dockerfile.vector_sync", root / "infra" / "docker" / "Dockerfile.vector_sync")
    shutil.copy2(APP_ROOT / "requirements-common.txt", root / "requirements-common.txt")
    assert not list(root.rglob(".env"))
    assert not list(root.rglob(".env.*"))

    manifest_path = root / "infra" / "build" / "service-manifests.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["services"]["sync_checker"]["entry_roots"] = ["synthetic/main.py"]
    manifest["services"]["sync_checker"]["path_remaps"] = {"synthetic": ""}
    manifest["services"]["vector_sync"] = {
        "entry_roots": ["app/main.py"],
        "runtime": {"kind": "asgi", "target": "app.main:app"},
        "path_remaps": {},
        "module_aliases": {},
        "requirements_roots": ["requirements.txt"],
        "extra_modules": [],
        "resources": [],
        "model_preparation": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    synthetic_entry = root / "synthetic" / "main.py"
    synthetic_entry.parent.mkdir()
    synthetic_entry.write_text("def checker(request):\n    return ({}, 200)\n", encoding="utf-8")
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text(
        "import qdrant_client\n"
        "async def app(scope, receive, send):\n"
        "    await receive()\n"
        "    await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'application/json')]})\n"
        "    await send({'type': 'http.response.body', 'body': b'{\"status\":\"ok\"}', 'more_body': False})\n",
        encoding="utf-8",
    )
    (root / "qdrant_client.py").write_text(
        "class QdrantClient:\n    def __init__(self, *args, **kwargs):\n        pass\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text("", encoding="utf-8")

    if case:
        if case == "missing-module":
            source = synthetic_entry
            source.write_text("import core.missing_module\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
        elif case == "missing-provider":
            source = synthetic_entry
            source.write_text("import unknown_vendor.sdk\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
        elif case == "missing-resource":
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["services"]["sync_checker"]["resources"] = [
                {"source_path": "assets/missing.bin", "runtime_path": "data/missing.bin"}
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif case == "missing-sdk-symbol":
            source = synthetic_entry
            source.write_text("from functions_framework import missing_sdk_symbol\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
        elif case == "startup-failure":
            source = synthetic_entry
            source.write_text("raise RuntimeError('synthetic startup failure')\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
    return root


def _install_fakes(root: Path, tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gcloud\\t%s\\n' \"$*\" >> \"$DEPLOY_EVENT_LOG\"\n"
        "case \"$*\" in *'services describe'*) printf 'https://synthetic.invalid/service\\n' ;; esac\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'docker\\t%s\\n' \"$*\" >> \"$DEPLOY_EVENT_LOG\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    gate = root / "infra" / "build" / "route_gate.py"
    gate.write_text(
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[2]\n"
        "args = sys.argv[1:]\n"
        "def option(name): return args[args.index(name) + 1]\n"
        "service = option('--service')\n"
        "image_mode = '--image-mode' in args\n"
        "artifact = root / '.build' / 'vector_sync-preflight' if image_mode else Path(option('--output'))\n"
        "build = root / 'infra' / 'build'\n"
        "def run(command): return subprocess.run(command, cwd=root, capture_output=True, text=True)\n"
        "assembled = run([sys.executable, str(build / 'assemble.py'), '--service', service, '--source-root', str(root), '--output', str(artifact)])\n"
        "if assembled.returncode: print(assembled.stdout + assembled.stderr); raise SystemExit(assembled.returncode)\n"
        "result = root / '.build' / (service + '-route-validation.json')\n"
        "validated = run([sys.executable, '-I', str(build / 'validate.py'), '--service', service, '--artifact', str(artifact), '--offline', '--result', str(result)])\n"
        "if validated.returncode: print(validated.stdout + validated.stderr); raise SystemExit(validated.returncode)\n"
        "if image_mode and __import__('os').environ.get('PREFLIGHT_FAILURE') == 'missing-validation-record': result.unlink(missing_ok=True)\n"
        "if not result.is_file() or json.loads(result.read_text()).get('status') != 'passed': raise SystemExit('missing validation record')\n",
        encoding="utf-8",
    )
    (root / "infra" / "build" / "preflight.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "case \"${PREFLIGHT_FAILURE:-}\" in\n"
        "  pip-conflict) echo 'synthetic pip check: package-a 2.0 conflicts with dependency-b 2.1' >&2; exit 25 ;;\n"
        "  bootstrap-failure) echo 'synthetic bootstrap failure: runtime dependency setup failed' >&2; exit 24 ;;\n"
        "esac\n"
        "\"$ROUTE_PYTHON\" \"${BASH_SOURCE[0]%/*}/route_gate.py\" \"$@\"\n",
        encoding="utf-8",
    )
    (root / "infra" / "build" / "preflight.sh").chmod(0o755)
    return fake_bin, log


def _run_deployment(
    tmp_path: Path,
    script: str,
    *,
    source_case: str | None = None,
    preflight_failure: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[tuple[str, str]], Path]:
    bash = _bash()
    root = _new_checkout(tmp_path, source_case)
    fake_bin, log = _install_fakes(root, tmp_path)
    environment = {
        **{name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if name in os.environ},
        "PATH": os.pathsep.join((str(fake_bin), str(Path(sys.executable).parent), os.environ.get("PATH", ""))),
        "ROUTE_PYTHON": sys.executable,
        "DEPLOY_EVENT_LOG": str(log),
        "PREFLIGHT_FAILURE": preflight_failure or "",
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "VERSION": "v-synthetic-42",
        "ROOT_FOLDER_URL": "https://synthetic.invalid/root",
        "GEMINI_API_KEY": "synthetic-gemini-key",
        "QDRANT_URL": "https://synthetic.invalid/qdrant",
        "QDRANT_API_KEY": "synthetic-qdrant-key",
        "QDRANT_COLLECTION": "synthetic-collection",
        "GOOGLE_CLIENT_ID": "synthetic-client-id",
        "GOOGLE_CLIENT_SECRET": "synthetic-client-secret",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh-token",
        "SYNC_CHECKER_NAME": "synthetic-sync-checker",
        "VECTOR_SYNC_SERVICE_NAME": "synthetic-vector-sync",
        "VECTOR_SYNC_QUEUE": "synthetic-vector-queue",
    }
    command = [bash, _bash_path(root / "infra" / "deploy" / "core" / script)]
    completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, check=False, timeout=180)
    events = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            kind, payload = line.split("\t", 1)
            events.append((kind, payload))
    return completed, events, root


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    [
        ("missing-module", "missing_module"),
        ("missing-provider", "unknown_vendor.sdk"),
        ("missing-resource", "assets/missing.bin"),
        ("missing-sdk-symbol", "required external import functions_framework.missing_sdk_symbol is missing"),
        ("startup-failure", "validation failed"),
    ],
)
def test_real_function_deploy_script_stops_on_source_validation_failures(
    tmp_path: Path, case: str, diagnostic: str,
) -> None:
    result, events, _root = _run_deployment(
        tmp_path, "deploy_sync_checker.sh", source_case=case,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert diagnostic in result.stdout + result.stderr
    assert not any(kind == "gcloud" and payload.startswith("functions deploy ") for kind, payload in events)


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        ("pip-conflict", "synthetic pip check: package-a 2.0 conflicts"),
        ("bootstrap-failure", "synthetic bootstrap failure: runtime dependency setup failed"),
        ("missing-validation-record", "missing validation record"),
    ],
)
def test_real_image_deploy_script_stops_on_preflight_failures(
    tmp_path: Path, failure: str, diagnostic: str,
) -> None:
    result, events, _root = _run_deployment(
        tmp_path,
        "local_deploy_vector_sync.sh",
        preflight_failure=failure,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert diagnostic in result.stdout + result.stderr
    assert any(kind == "docker" and payload.startswith("build ") for kind, payload in events)
    assert not any(kind == "docker" and payload.startswith("push ") for kind, payload in events)
    assert not any(kind == "gcloud" and payload.startswith("run deploy ") for kind, payload in events)


def test_workflow_guards_all_image_publish_jobs_and_uses_gated_artifacts() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    triggers = workflow["on"]
    assert set(triggers) == {"pull_request", "push"}
    assert triggers["pull_request"]["branches"] == ["main"]
    assert set(triggers["pull_request"]["paths"]) == REQUIRED_PATHS
    assert triggers["push"]["branches"] == ["main"]
    assert set(triggers["push"]["paths"]) == REQUIRED_PATHS
    assert triggers["push"]["tags"] == ["v*"]

    jobs = workflow["jobs"]
    for name, service in PUBLISH_JOBS.items():
        job = jobs[name]
        assert set(job["needs"]) == GATES
        assert job["if"] == "github.event_name == 'push'"
        steps = job["steps"]
        download = next(step for step in steps if step.get("uses") == "actions/download-artifact@v4")
        assert download["with"]["name"] == f"gated-image-{service}"
        publish = next(step for step in steps if step.get("name") == "Load, tag, and push gated image")
        command = publish["run"]
        assert "docker load" in command and "docker push" in command
        assert "docker build" not in command
        assert not any(step.get("uses") == "docker/build-push-action@v5" for step in steps)
