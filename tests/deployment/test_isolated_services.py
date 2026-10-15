"""Compose real service assembly with the isolated offline validator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = APP_ROOT / "infra" / "build"
ASSEMBLER = BUILD_TOOLS / "assemble.py"
VALIDATOR = BUILD_TOOLS / "validate.py"
SMOKE_SUPPORT = BUILD_TOOLS / "smoke_support.py"
MANIFEST = BUILD_TOOLS / "service-manifests.json"
SERVICES = (
    "mcp",
    "vector_sync",
    "tldv_import",
    "sync_checker",
    "tldv_webhook",
    "tldv_reconciliation",
)
EXPECTED_TARGETS = {
    "mcp": "services.mcp.app.main:app",
    "vector_sync": "services.vector_sync.app.main:app",
    "tldv_import": "app.main:app",
    "sync_checker": "checker",
    "tldv_webhook": "tldv_webhook",
    "tldv_reconciliation": "tldv_reconciliation",
}
EXPECTED_ENTRIES = {
    "mcp": "services/mcp/app/main.py",
    "vector_sync": "services/vector_sync/app/main.py",
    "tldv_import": "app/main.py",
    "sync_checker": "main.py",
    "tldv_webhook": "main.py",
    "tldv_reconciliation": "main.py",
}
FORBIDDEN_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "tests", "test", "fixtures", "fixture",
    "dist", "build", "cache", ".cache",
}


def _external_harness(destination: Path) -> Path:
    destination.mkdir(parents=True)
    shutil.copyfile(VALIDATOR, destination / "validate.py")
    shutil.copyfile(SMOKE_SUPPORT, destination / "smoke_support.py")
    return destination / "validate.py"


def _assemble(service: str, artifact: Path, source_root: Path = APP_ROOT) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--service", service,
            "--output", str(artifact),
            "--source-root", str(source_root),
        ],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, f"{service}: {completed.stdout}\n{completed.stderr}"


def _validate(service: str, artifact: Path, validator: Path, result_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(validator.resolve()),
            "--service", service,
            "--artifact", str(artifact.resolve()),
            "--offline",
            "--result", str(result_path.resolve()),
        ],
        cwd=result_path.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _assert_runtime_only(artifact: Path, report: dict[str, object], service: str) -> None:
    files = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
    assert "requirements.txt" in files
    assert "assembly-report.json" in files
    assert EXPECTED_ENTRIES[service] in files
    for relative in files:
        parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
        assert not any(part in FORBIDDEN_PARTS or part.startswith(".env") for part in parts), relative
        assert relative.casefold() not in {"requirements-dev.txt", "readme.md", ".dockerignore", ".gitignore"}
    source_files = report["source_files"]
    requirements_graph = report["requirements_graph"]
    assert isinstance(source_files, list) and source_files
    assert isinstance(requirements_graph, dict)
    assert requirements_graph["roots"]
    assert requirements_graph["files"]
    assert all(item["runtime_path"] in files for item in requirements_graph["files"])
    assert all(item["runtime_path"] in files for item in source_files)
    assert all(item["source_path"] for item in source_files)
    assert all(item["runtime_path"] for item in source_files)
    requirements = (artifact / "requirements.txt").read_text(encoding="utf-8")
    assert all(root in requirements for root in requirements_graph["roots"])


def test_all_six_current_assembled_services_pass_offline_runtime_validation(tmp_path: Path) -> None:
    """Validate each actual assembled source tree with the external harness."""
    assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    build_root = APP_ROOT / ".build"
    build_root.mkdir(exist_ok=True)
    assert not build_root.is_symlink()
    output_root = build_root / f"test-isolated-services-{uuid.uuid4().hex}"
    assert output_root.parent.resolve() == build_root.resolve()
    validator = _external_harness(tmp_path / "external-validation")
    results = tmp_path / "external-results"
    results.mkdir()

    try:
        for service in SERVICES:
            config = manifest["services"][service]
            assert config["runtime"]["target"] == EXPECTED_TARGETS[service]
            artifact = output_root / service
            _assemble(service, artifact)
            report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
            assert report["service"] == service
            assert report["runtime_target"] == EXPECTED_TARGETS[service]
            _assert_runtime_only(artifact, report, service)
            source_paths = {record["source_path"] for record in report["source_files"]}
            assert set(config["entry_roots"]) <= source_paths
            expected_requirement_roots = {
                ".requirements/" + root for root in config["requirements_roots"]
            }
            assert expected_requirement_roots <= set(report["requirements_graph"]["roots"])

            # Every manifest remap is reflected in the assembled source records.
            remaps = config.get("path_remaps", {})
            for record in report["source_files"]:
                source = record["source_path"]
                expected = source
                for prefix, replacement in sorted(remaps.items(), key=lambda pair: len(pair[0]), reverse=True):
                    if source == prefix or source.startswith(prefix + "/"):
                        expected = replacement + source[len(prefix):]
                        expected = expected.lstrip("/")
                        break
                assert record["runtime_path"] == expected
            if service in {"sync_checker", "tldv_webhook", "tldv_reconciliation"}:
                assert "main.py" in {record["runtime_path"] for record in report["source_files"]}
            if service == "tldv_reconciliation":
                assert "tldv_client.py" in {record["runtime_path"] for record in report["source_files"]}

            result_path = results / f"{service}.json"
            completed = _validate(service, artifact, validator, result_path)
            assert completed.returncode == 0, f"{service}: {completed.stdout}\n{completed.stderr}"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            assert result["service"] == service
            assert result["target"] == EXPECTED_TARGETS[service]
            assert result["python_version"].startswith("3.12")
            assert result["status"] == "passed"
            assert {check["name"] for check in result["checks"]} >= {
                "runtime_target_and_smoke", "offline_smoke", "artifact_immutability"
            }
    finally:
        resolved_root = output_root.resolve()
        if resolved_root.parent == build_root.resolve() and output_root.exists():
            shutil.rmtree(output_root)


def test_assembler_report_captures_delayed_missing_function_import_and_validator_rejects_it(
    tmp_path: Path,
) -> None:
    """A function-local external import remains required even before invocation."""
    source_root = tmp_path / "synthetic-source"
    build_tools = source_root / "infra" / "build"
    service_root = source_root / "services" / "synthetic_function"
    build_tools.mkdir(parents=True)
    service_root.mkdir(parents=True)
    (service_root / "requirements.txt").write_text("", encoding="utf-8")
    (service_root / "main.py").write_text(
        "def checker(request):\n"
        "    import missing_delayed_runtime_dependency\n"
        "    return ({}, 200)\n",
        encoding="utf-8",
    )
    (build_tools / "dependency-providers.json").write_text(
        '{"schema_version": 1, "providers": []}\n', encoding="utf-8"
    )
    (build_tools / "service-manifests.json").write_text(
        json.dumps({
            "schema_version": 1,
            "services": {
                "sync_checker": {
                    "entry_roots": ["services/synthetic_function/main.py"],
                    "runtime": {"kind": "function", "target": "checker"},
                    "path_remaps": {"services/synthetic_function": ""},
                    "requirements_roots": ["services/synthetic_function/requirements.txt"],
                }
            },
        }),
        encoding="utf-8",
    )
    artifact = source_root / ".build" / "synthetic-artifact"
    _assemble("sync_checker", artifact, source_root)
    report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
    missing = next(
        item for item in report["required_external_imports"]
        if item["name"] == "missing_delayed_runtime_dependency"
    )
    assert missing["dependency_chain"]
    assert missing["source_path"] == "services/synthetic_function/main.py"

    validator = _external_harness(tmp_path / "external-negative-validation")
    result_path = tmp_path / "external-negative-results" / "synthetic.json"
    result_path.parent.mkdir()
    completed = _validate("sync_checker", artifact, validator, result_path)
    assert completed.returncode != 0
    assert "missing_delayed_runtime_dependency" in completed.stdout + completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
