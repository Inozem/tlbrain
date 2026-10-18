"""Focused dependency-boundary checks for assembled service closures."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = APP_ROOT / "infra" / "build" / "assemble.py"
REGISTRY_PATH = APP_ROOT / "infra" / "build" / "dependency-providers.json"
FOLDER_SOURCE = APP_ROOT / "core" / "integrations" / "google_drive" / "folders.py"
RETRY_SOURCE = APP_ROOT / "core" / "utils" / "retry.py"


def _write_project(
    root: Path,
    files: dict[str, str],
    *,
    requirements_roots: list[str] | None = None,
    registry: dict[str, object] | None = None,
) -> None:
    build_root = root / "infra" / "build"
    build_root.mkdir(parents=True)
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "services": {
            "test_service": {
                "entry_roots": ["main.py"],
                "runtime": {"kind": "function", "target": "main"},
                "path_remaps": {},
                "module_aliases": {},
                "requirements_roots": requirements_roots or [],
                "extra_modules": [],
                "resources": [],
                "model_preparation": [],
            }
        },
    }
    (build_root / "service-manifests.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (build_root / "dependency-providers.json").write_text(
        json.dumps(registry or {"schema_version": 1, "providers": [], "optional_imports": []}),
        encoding="utf-8",
    )


def _copy_registry_requirements(root: Path, registry: dict[str, object]) -> None:
    providers = registry.get("providers", [])
    assert isinstance(providers, list)
    for provider in providers:
        assert isinstance(provider, dict)
        requirement_files = provider.get("requirements_files", [])
        assert isinstance(requirement_files, list)
        for relative in requirement_files:
            assert isinstance(relative, str)
            source = APP_ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def _assemble(source_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--service",
            "test_service",
            "--output",
            str(output),
            "--source-root",
            str(source_root),
        ],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _assemble_ok(source_root: Path, output: Path) -> dict[str, object]:
    result = _assemble(source_root, output)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "assembly-report.json").read_text(encoding="utf-8"))


def _artifact_path(source_root: Path, name: str) -> Path:
    return source_root / ".build" / name


def _actual_registry() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_pure_folder_algorithms_assemble_without_sdk_providers(tmp_path: Path) -> None:
    registry = _actual_registry()
    source_root = tmp_path / "folders-checkout"
    source_root.mkdir()
    folder_source = FOLDER_SOURCE.read_text(encoding="utf-8")
    _write_project(
        source_root,
        {"main.py": folder_source, "requirements/service.txt": ""},
        requirements_roots=["requirements/service.txt"],
        registry=registry,
    )
    _copy_registry_requirements(source_root, registry)

    report = _assemble_ok(source_root, _artifact_path(source_root, "folders-artifact"))

    assert report["provider_roots"] == []
    assert report["required_external_imports"] == []
    requirements_files = report["requirements_graph"]["files"]
    assert [entry["runtime_path"] for entry in requirements_files] == [
        ".requirements/requirements/service.txt"
    ]


def test_firestore_only_source_closure_selects_only_firestore_provider(
    tmp_path: Path,
) -> None:
    registry = _actual_registry()
    source_root = tmp_path / "firestore-checkout"
    source_root.mkdir()
    _write_project(
        source_root,
        {
            "main.py": "from google.cloud.firestore import Client\n",
            "requirements/service.txt": "",
        },
        requirements_roots=["requirements/service.txt"],
        registry=registry,
    )
    _copy_registry_requirements(source_root, registry)

    report = _assemble_ok(source_root, _artifact_path(source_root, "firestore-artifact"))

    assert report["provider_roots"] == [
        "core/integrations/firestore/requirements.txt"
    ]
    requirement_files = report["requirements_graph"]["files"]
    staged = {entry["runtime_path"]: entry for entry in requirement_files}
    assert set(staged) == {
        ".requirements/core/integrations/firestore/requirements.txt",
        ".requirements/requirements/service.txt",
    }


def _retry_project(root: Path, required_import: str | None = None) -> dict[str, object]:
    registry = _actual_registry()
    main = "import core.utils.retry\n"
    files = {"main.py": main, "core/utils/retry.py": RETRY_SOURCE.read_text(encoding="utf-8")}
    if required_import:
        files["main.py"] += "import app.required\n"
        files["app/__init__.py"] = ""
        files["app/required.py"] = f"import {required_import}\n"
    files["requirements/service.txt"] = ""
    _write_project(
        root,
        files,
        requirements_roots=["requirements/service.txt"],
        registry=registry,
    )
    _copy_registry_requirements(root, registry)
    return registry


def test_real_retry_optional_sdk_imports_do_not_add_provider_roots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "retry-optional-checkout"
    source_root.mkdir()
    registry = _retry_project(source_root)

    optional_entries = [
        item
        for item in registry["optional_imports"]
        if item["source_file"] == "core/utils/retry.py"
    ]
    assert {item["import_prefix"] for item in optional_entries} == {
        "googleapiclient.errors",
        "qdrant_client.http.exceptions",
        "google.genai.errors",
        "requests.exceptions",
    }
    assert all(item.get("guard", item.get("guard_context")) == "ImportError" for item in optional_entries)

    report = _assemble_ok(source_root, _artifact_path(source_root, "retry-optional-artifact"))

    assert report["provider_roots"] == []
    required_import_names = {
        item["name"] for item in report["required_external_imports"]
    }
    assert not any(
        name.startswith(("googleapiclient", "qdrant_client", "google.genai"))
        for name in required_import_names
    )


@pytest.mark.parametrize(
    ("required_import", "provider_requirement"),
    [
        ("googleapiclient.discovery", "core/integrations/google_docs/requirements.txt"),
        ("qdrant_client", "core/integrations/qdrant/requirements.txt"),
        ("google.genai", "core/integrations/gemini/requirements.txt"),
    ],
)
def test_required_import_elsewhere_still_selects_retry_prefix_provider(
    tmp_path: Path, required_import: str, provider_requirement: str
) -> None:
    source_root = tmp_path / ("retry-required-" + required_import.replace(".", "-"))
    source_root.mkdir()
    _retry_project(source_root, required_import)

    report = _assemble_ok(
        source_root, _artifact_path(source_root, source_root.name + "-artifact")
    )

    assert report["provider_roots"] == [provider_requirement]
    required_import_names = {
        item["name"] for item in report["required_external_imports"]
    }
    assert required_import in required_import_names


def test_current_webhook_artifact_contains_only_its_runtime_integrations() -> None:
    build_root = APP_ROOT / ".build"
    build_root.mkdir(exist_ok=True)
    assert not build_root.is_symlink()
    output = build_root / f"test-minimal-webhook-{uuid.uuid4().hex}"
    assert not output.exists()
    artifact = output / "tldv_webhook"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ASSEMBLER),
                "--service",
                "tldv_webhook",
                "--output",
                str(artifact),
                "--source-root",
                str(APP_ROOT),
            ],
            cwd=APP_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(
            (artifact / "assembly-report.json").read_text(encoding="utf-8")
        )

        assert set(report["provider_roots"]) == {
            "core/integrations/cloud_tasks/requirements.txt",
            "core/integrations/firestore/requirements.txt",
            "core/integrations/google_auth/requirements.txt",
        }
        paths = {
            path.relative_to(artifact).as_posix()
            for path in artifact.rglob("*")
            if path.is_file()
        }
        assert "main.py" in paths
        assert "requirements.txt" in paths
        assert "assembly-report.json" in paths
        assert not any(
            marker in path.casefold()
            for path in paths
            for marker in (
                "google_docs",
                "gemini",
                "qdrant",
                "fastembed",
                "requirements-dev",
                "dev-requirements",
                "/tests/",
                "/test/",
            )
        )
        assert not any(
            any(part.casefold() in {"tests", "test"} for part in PurePosixPath(path).parts)
            for path in paths
        )
        assert not any(
            PurePosixPath(path).name.casefold()
            in {"requirements-dev.txt", "dev-requirements.txt", "test-requirements.txt"}
            for path in paths
        )
    finally:
        resolved_test_dir = output.resolve()
        if (
            resolved_test_dir.parent == build_root.resolve()
            and resolved_test_dir.name.startswith("test-minimal-webhook-")
            and output.is_dir()
            and not output.is_symlink()
        ):
            shutil.rmtree(output)
