"""Offline contract tests for the isolated deployment validator.

The validation scripts are intentionally loaded only inside test bodies: the
suite can be collected while the implementation is still being developed.
All application inputs below are synthetic temporary artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = APP_ROOT / "infra" / "build"
VALIDATOR = VALIDATION_DIR / "validate.py"
SMOKE_SUPPORT = VALIDATION_DIR / "smoke_support.py"

_ENTRY_PATHS = {
    "mcp": "app/main.py",
    "vector_sync": "app/main.py",
    "tldv_import": "app/main.py",
    "sync_checker": "main.py",
    "tldv_webhook": "main.py",
    "tldv_reconciliation": "main.py",
}
_RUNTIME_TARGETS = {
    "mcp": "app.main:app",
    "vector_sync": "app.main:app",
    "tldv_import": "app.main:app",
    "sync_checker": "main:checker",
    "tldv_webhook": "main:tldv_webhook",
    "tldv_reconciliation": "main:tldv_reconciliation",
}
_RUNTIME_KINDS = {
    "mcp": "asgi",
    "vector_sync": "asgi",
    "tldv_import": "asgi",
    "sync_checker": "function",
    "tldv_webhook": "function",
    "tldv_reconciliation": "function",
}


def _artifact(
    tmp_path: Path,
    service: str,
    entry_source: str,
    *,
    extra_sources: dict[str, str] | None = None,
    requirements: str = "",
) -> Path:
    """Create a minimal assembled artifact and schema-v1 assembly report."""
    artifact = tmp_path / f"artifact-{service}"
    sources = {_ENTRY_PATHS[service]: entry_source, **(extra_sources or {})}
    (artifact / ".requirements").mkdir(parents=True)
    (artifact / "requirements.txt").write_text(requirements, encoding="utf-8")

    source_records = []
    for relative, contents in sources.items():
        path = artifact / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        source_records.append(
            {
                "source_path": relative,
                "runtime_path": relative,
                "sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
            }
        )

    report = {
        "schema_version": 1,
        "service": service,
        "runtime": {
            "target": _RUNTIME_TARGETS[service],
            "kind": _RUNTIME_KINDS[service],
        },
        "source_files": source_records,
    }
    (artifact / "assembly-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return artifact.resolve()


def _external_harness(tmp_path: Path) -> tuple[Path, Path]:
    """Stage both standalone runner files outside the checkout."""
    validation_dir = tmp_path / "external-validation"
    validation_dir.mkdir()
    validator = validation_dir / "validate.py"
    smoke_support = validation_dir / "smoke_support.py"
    shutil.copyfile(VALIDATOR, validator)
    shutil.copyfile(SMOKE_SUPPORT, smoke_support)
    return validator.resolve(), smoke_support.resolve()


def _external_smoke_support(tmp_path: Path) -> Path:
    """Stage only smoke_support for its independently deployable tests."""
    validation_dir = tmp_path / "external-smoke-support"
    validation_dir.mkdir()
    smoke_support = validation_dir / "smoke_support.py"
    shutil.copyfile(SMOKE_SUPPORT, smoke_support)
    return smoke_support.resolve()


def _success_entry_source(service: str) -> str:
    source = "import app_internal.qualified\nfrom app_internal import VALUE\n"
    if service == "vector_sync":
        source = "from qdrant_client import QdrantClient\nclient = QdrantClient()\n" + source
    return (
        source
        + "def deferred_import():\n"
        "    from app_internal import deferred\n"
        "    return deferred.VALUE\n"
        "assert VALUE == 4\n"
        "assert app_internal.qualified.VALUE == 3\n"
        "assert deferred_import() == 2\n"
    )


def _run_validator(
    service: str,
    artifact: Path,
    result_path: Path,
    *,
    cwd: Path,
    validator: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(validator.resolve()),
            "--service",
            service,
            "--artifact",
            str(artifact.resolve()),
            "--offline",
            "--result",
            str(result_path.resolve()),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _tree_fingerprint(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_success_result(
    completed: subprocess.CompletedProcess[str],
    result_path: Path,
    service: str,
    artifact: Path,
) -> dict[str, object]:
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["service"] == service
    assert result["artifact_fingerprint"]
    assert result["target"] == _RUNTIME_TARGETS[service]
    assert result["python_version"].startswith("3.12")
    assert result["platform"]
    assert isinstance(result["installed_versions"], dict)
    assert result["checks"]
    assert result["status"] == "passed"
    return result


def test_validator_files_are_external_absolute_paths_and_only_two_harness_files(
    tmp_path: Path,
) -> None:
    # Keep this assertion inside the test so pytest collection works before
    # validate.py and smoke_support.py have been added.
    validator, smoke_support = _external_harness(tmp_path)
    assert validator.is_absolute()
    assert smoke_support.is_absolute()
    assert {path.name for path in validator.parent.iterdir()} == {
        "validate.py",
        "smoke_support.py",
    }


@pytest.mark.parametrize("service", tuple(_ENTRY_PATHS))
def test_validator_imports_each_service_entrypoint_offline(
    tmp_path: Path, service: str
) -> None:
    artifact = _artifact(
        tmp_path,
        service,
        _success_entry_source(service),
        extra_sources={
            "app_internal/__init__.py": "VALUE = 4\n",
            "app_internal/qualified.py": "VALUE = 3\n",
            "app_internal/deferred.py": "VALUE = 2\n",
        },
    )
    result_path = (tmp_path / "external-results" / f"{service}.json").resolve()
    result_path.parent.mkdir()
    validator, _ = _external_harness(tmp_path)
    before = _tree_fingerprint(artifact)

    completed = _run_validator(
        service, artifact, result_path, cwd=tmp_path, validator=validator
    )

    _assert_success_result(completed, result_path, service, artifact)
    assert _tree_fingerprint(artifact) == before
    assert not list(artifact.rglob("__pycache__"))
    assert not list(artifact.rglob("*.pyc"))


def test_validator_ignores_poisoned_host_paths_and_writes_result_outside_artifact(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        tmp_path,
        "mcp",
        "import harmless_poison\nfrom app_internal import VALUE\nassert VALUE == 7\n",
        extra_sources={
            "harmless_poison.py": "VALUE = 'artifact source wins'\n",
            "app_internal.py": "VALUE = 7\n",
        },
    )
    poison_dir = tmp_path / "poison"
    poison_dir.mkdir()
    (poison_dir / "harmless_poison.py").write_text(
        "raise RuntimeError('host path was used')\n", encoding="utf-8"
    )
    result_path = (tmp_path / "external-results" / "result.json").resolve()
    result_path.parent.mkdir()
    validator, _ = _external_harness(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison_dir)
    user_site = (
        poison_dir
        / "user-site"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "site-packages"
    )
    user_site.mkdir(parents=True)
    (user_site / "harmless_poison.py").write_text(
        "raise RuntimeError('user site was used')\n", encoding="utf-8"
    )
    env["PYTHONUSERBASE"] = str(poison_dir / "user-site")

    completed = _run_validator(
        "mcp", artifact, result_path, cwd=poison_dir, validator=validator, env=env
    )

    _assert_success_result(completed, result_path, "mcp", artifact)
    assert result_path.is_relative_to(tmp_path.resolve())
    assert not result_path.is_relative_to(artifact)


@pytest.mark.parametrize(
    ("service", "entry_source", "extra_sources", "diagnostic"),
    [
        (
            "mcp",
            "import missing_sdk\n",
            {},
            "missing_sdk",
        ),
        (
            "mcp",
            "from app_internal import MISSING_NAME\n",
            {"app_internal.py": "VALUE = 1\n"},
            "MISSING_NAME",
        ),
        (
            "mcp",
            "import missing_internal\n",
            {},
            "missing_internal",
        ),
    ],
    ids=("missing-sdk", "missing-named-import", "missing-internal-module"),
)
def test_validator_fails_for_missing_sdk_or_application_import(
    tmp_path: Path,
    service: str,
    entry_source: str,
    extra_sources: dict[str, str],
    diagnostic: str,
) -> None:
    artifact = _artifact(
        tmp_path, service, entry_source, extra_sources=extra_sources
    )
    result_path = (tmp_path / "external-results" / "failure.json").resolve()
    result_path.parent.mkdir()
    validator, _ = _external_harness(tmp_path)
    before = _tree_fingerprint(artifact)

    completed = _run_validator(
        service, artifact, result_path, cwd=tmp_path, validator=validator
    )

    assert completed.returncode != 0
    assert diagnostic.lower() in (completed.stdout + completed.stderr).lower()
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["artifact_fingerprint"]
    assert _tree_fingerprint(artifact) == before
    assert not list(artifact.rglob("__pycache__"))
    assert not list(artifact.rglob("*.pyc"))


@pytest.mark.parametrize("service", tuple(_ENTRY_PATHS))
def test_smoke_support_runs_offline_startup_and_benign_route(
    tmp_path: Path, service: str
) -> None:
    """Exercise component smokes separately via ``-k smoke_support``."""
    marker = (tmp_path / f"{service}-route-ran.txt").resolve()
    entry_source = "from app_internal import VALUE\nassert VALUE == 4\n"
    if service == "vector_sync":
        entry_source = (
            "from qdrant_client import QdrantClient\n"
            "assert getattr(QdrantClient, '__tlbrain_external_fake__', False) is True\n"
            "client = QdrantClient()\n"
            + entry_source
        )
    entry_source += (
        "from pathlib import Path\n"
        "def health():\n"
        f"    Path({str(marker)!r}).write_text('route ran', encoding='utf-8')\n"
        "    return {'status': 'ok'}\n"
    )
    artifact = _artifact(
        tmp_path,
        service,
        entry_source,
        extra_sources={"app_internal.py": "VALUE = 4\n"},
    )
    smoke_support = _external_smoke_support(tmp_path)
    # run_smoke owns startup and benign-route behavior; its return shape is
    # deliberately outside this test contract.
    # Import the future helper only at test time and from its absolute path.
    # The helper stays outside the application artifact and is never discovered
    # through cwd, PYTHONPATH, or the checkout as an application module.
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('smoke_support', {str(smoke_support)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        f"module.run_smoke({service!r}, {str(artifact)!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"{service}: {completed.stdout}{completed.stderr}"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "route ran"

