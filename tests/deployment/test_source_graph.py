"""Contract tests for the deployment source-graph assembler CLI.

Each test builds a tiny checkout so graph resolution is independent of the
application's installed packages and never imports the synthetic source.
"""

from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = APP_ROOT / "infra" / "build" / "assemble.py"


def _project(
    tmp_path: Path,
    files: dict[str, str],
    *,
    entry_roots: list[str] | None = None,
    module_aliases: dict[str, str] | None = None,
    path_remaps: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Create a minimal synthetic checkout and output directory."""
    source_root = tmp_path / "checkout"
    build_root = source_root / "infra" / "build"
    build_root.mkdir(parents=True)

    for relative_path, contents in files.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "services": {
            "mcp": {
                "entry_roots": entry_roots or ["main.py"],
                "runtime": {},
                "path_remaps": path_remaps or {},
                "module_aliases": module_aliases or {},
                "requirements_roots": [],
                "extra_modules": [],
                "resources": [],
                "model_preparation": [],
            }
        },
    }
    registry = {
        "schema_version": 1,
        "providers": {},
        "optional_imports": {},
    }
    (build_root / "service-manifests.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (build_root / "dependency-providers.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    output = source_root / ".build" / "assembled"
    return source_root, output


def _run(source_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--service",
            "mcp",
            "--output",
            str(output),
            "--source-root",
            str(source_root),
        ],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _set_resources(source_root: Path, resources: list[dict[str, str]]) -> None:
    path = source_root / "infra" / "build" / "service-manifests.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["services"]["mcp"]["resources"] = resources
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _assert_success(source_root: Path, output: Path) -> dict[str, object]:
    result = _run(source_root, output)
    assert result.returncode == 0, result.stderr or result.stdout
    report_path = output / "assembly-report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["service"] == "mcp"
    assert isinstance(report["source_files"], list)
    return report


def _raw_source_entries(report: dict[str, object]) -> list[object]:
    entries = report["source_files"]
    assert isinstance(entries, list)
    return entries


def _source_path(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry.replace("\\", "/").lstrip("./")
    if isinstance(entry, dict):
        for key in ("source_path", "source", "path"):
            value = entry.get(key)
            if isinstance(value, str):
                return value.replace("\\", "/").lstrip("./")
    return None


def _source_paths(report: dict[str, object]) -> set[str]:
    """Read source paths from either a path string or a report entry object."""
    result: set[str] = set()
    for entry in _raw_source_entries(report):
        path = _source_path(entry)
        if path is not None:
            result.add(path)
    return result


def _runtime_paths(report: dict[str, object]) -> set[str]:
    result: set[str] = set()
    for entry in report["source_files"]:  # type: ignore[union-attr]
        if isinstance(entry, dict):
            for key in ("runtime_path", "runtime"):
                value = entry.get(key)
                if isinstance(value, str):
                    result.add(value.replace("\\", "/").lstrip("./"))
                    break
    return result


def test_resolves_absolute_relative_and_function_local_imports(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "import app.entry\n",
            "app/__init__.py": "",
            "app/entry.py": (
                "from app import absolute_dep\n"
                "from . import relative_dep\n"
                "def load():\n"
                "    import app.local_dep\n"
            ),
            "app/absolute_dep.py": "VALUE = 1\n",
            "app/relative_dep.py": "VALUE = 3\n",
            "app/local_dep.py": "VALUE = 2\n",
        },
    )

    report = _assert_success(source_root, output)
    sources = _source_paths(report)
    assert {
        "main.py",
        "app/entry.py",
        "app/absolute_dep.py",
        "app/relative_dep.py",
        "app/local_dep.py",
    } <= sources
    assert (output / "main.py").is_file()
    assert (output / "app" / "entry.py").is_file()
    assert (output / "app" / "absolute_dep.py").is_file()


def test_from_package_submodule_is_resolved_but_package_attribute_is_not(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "from package import submodule, CONSTANT\n",
            "package/__init__.py": "CONSTANT = 42\n",
            "package/submodule.py": "VALUE = 1\n",
        },
    )

    report = _assert_success(source_root, output)
    sources = _source_paths(report)
    assert {"package/__init__.py", "package/submodule.py"} <= sources
    assert not any(path.endswith("package/CONSTANT.py") for path in sources)


def test_namespace_packages_and_every_reachable_parent_init_are_included(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "import namespace.branch.worker\n",
            # namespace/ intentionally has no __init__.py.
            "namespace/branch/__init__.py": "from . import bootstrap\n",
            "namespace/branch/bootstrap.py": "from namespace.shared import helper\n",
            "namespace/branch/worker.py": "from . import bootstrap\n",
            "namespace/shared/__init__.py": "from . import helper\n",
            "namespace/shared/helper.py": "VALUE = 1\n",
        },
    )

    report = _assert_success(source_root, output)
    sources = _source_paths(report)
    assert {
        "namespace/branch/__init__.py",
        "namespace/branch/bootstrap.py",
        "namespace/branch/worker.py",
        "namespace/shared/__init__.py",
        "namespace/shared/helper.py",
    } <= sources
    assert not (output / "namespace" / "__init__.py").exists()


def test_cyclic_imports_terminate_with_each_file_written_once(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "import cycle.left\n",
            "cycle/__init__.py": "",
            "cycle/left.py": "import cycle.right\n",
            "cycle/right.py": "from cycle import left\n",
        },
    )

    report = _assert_success(source_root, output)
    raw_paths = [
        path
        for entry in _raw_source_entries(report)
        if (path := _source_path(entry)) is not None
    ]
    sources = _source_paths(report)
    assert {"cycle/__init__.py", "cycle/left.py", "cycle/right.py"} <= sources
    assert len(raw_paths) == len(set(raw_paths))
    assert (output / "cycle" / "left.py").is_file()
    assert (output / "cycle" / "right.py").is_file()


def test_module_alias_and_source_to_runtime_path_remap_are_applied(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "import legacy_name\n",
            "src/actual.py": "VALUE = 1\n",
        },
        module_aliases={"legacy_name": "src.actual"},
        path_remaps={"src/actual.py": "vendor/actual.py"},
    )

    report = _assert_success(source_root, output)
    assert "src/actual.py" in _source_paths(report)
    assert "vendor/actual.py" in _runtime_paths(report)
    assert (output / "vendor" / "actual.py").is_file()
    assert not (output / "src" / "actual.py").exists()


@pytest.mark.parametrize(
    ("files", "missing_name", "line"),
    [
        (
            {"main.py": "import absent_module\n"},
            "absent_module",
            "1",
        ),
        (
            {
                "main.py": "from present import absent_symbol\n",
                "present.py": "EXISTS = True\n",
            },
            "absent_symbol",
            "1",
        ),
    ],
    ids=("missing-module", "missing-symbol"),
)
def test_missing_dependency_diagnostic_names_service_path_line_and_chain(
    tmp_path: Path,
    files: dict[str, str],
    missing_name: str,
    line: str,
) -> None:
    source_root, output = _project(tmp_path, files)

    result = _run(source_root, output)

    diagnostic = result.stdout + result.stderr
    assert result.returncode != 0
    assert "mcp" in diagnostic
    assert re.search(
        rf"main\.py[^\r\n]{{0,40}}(?:line\s*|:)\s*{line}\b"
        rf"|(?:line\s*|:)\s*{line}\b[^\r\n]{{0,40}}main\.py",
        diagnostic,
        re.IGNORECASE,
    )
    assert missing_name in diagnostic
    # A direct import still has an actionable one-link dependency chain.
    assert "chain" in diagnostic.lower() or "main.py" in diagnostic


def test_transitive_missing_dependency_reports_import_chain(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "import app.entry\n",
            "app/__init__.py": "",
            "app/entry.py": "from app import middle\n",
            "app/middle.py": "import absent.leaf\n",
        },
    )

    result = _run(source_root, output)

    diagnostic = result.stdout + result.stderr
    assert result.returncode != 0
    assert "mcp" in diagnostic
    assert re.search(
        r"app[/\\]middle\.py[^\r\n]{0,40}(?:line\s*|:)\s*1\b"
        r"|(?:line\s*|:)\s*1\b[^\r\n]{0,40}app[/\\]middle\.py",
        diagnostic,
        re.IGNORECASE,
    )
    assert "absent.leaf" in diagnostic
    assert "main.py" in diagnostic
    assert "app/entry.py" in diagnostic


def test_analysis_does_not_execute_synthetic_source(tmp_path: Path) -> None:
    marker = tmp_path / "synthetic-source-was-executed"
    source_root, output = _project(
        tmp_path,
        {
            "main.py": (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "raise RuntimeError('synthetic source executed')\n"
                "import dependency\n"
            ),
            "dependency.py": "VALUE = 1\n",
        },
    )

    report = _assert_success(source_root, output)

    assert "dependency.py" in _source_paths(report)
    assert not marker.exists()


def test_resources_are_staged_and_reported_with_hashes(tmp_path: Path) -> None:
    source_root, output = _project(tmp_path, {"main.py": "VALUE = 1\n", "assets/model.bin": "model bytes"})
    _set_resources(source_root, [{"source_path": "assets/model.bin", "runtime_path": "data/model.bin"}])

    report = _assert_success(source_root, output)

    assert (output / "data" / "model.bin").read_bytes() == b"model bytes"
    assert report["resources"] == [{
        "source_path": "assets/model.bin",
        "runtime_path": "data/model.bin",
        "sha256": __import__("hashlib").sha256(b"model bytes").hexdigest(),
    }]


def test_resource_directory_inventory_excludes_build_and_test_material(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "assets/keep.bin": "keep",
            "assets/.env.local": "secret",
            "assets/tests/test_data.bin": "test data",
            "assets/.build/stale.bin": "stale build",
        },
    )
    _set_resources(source_root, [{"source_path": "assets", "runtime_path": "data"}])

    report = _assert_success(source_root, output)

    assert (output / "data" / "keep.bin").read_bytes() == b"keep"
    assert not (output / "data" / ".env.local").exists()
    assert not (output / "data" / "tests").exists()
    assert not (output / "data" / ".build").exists()
    assert [item["runtime_path"] for item in report["resources"]] == ["data/keep.bin"]


def test_resource_walk_error_fails_with_declared_resource_and_no_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, output = _project(
        tmp_path,
        {"main.py": "VALUE = 1\n", "assets/subdir/file.bin": "asset"},
    )
    _set_resources(source_root, [{"source_path": "assets", "runtime_path": "data"}])
    spec = importlib.util.spec_from_file_location("assemble_under_test", ASSEMBLER)
    assert spec is not None and spec.loader is not None
    assembler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembler_module)
    original_walk = assembler_module.os.walk
    resource_root = source_root / "assets"

    def walk_with_resource_error(top: object, *args: object, **kwargs: object):
        onerror = kwargs.get("onerror")
        if Path(top) == resource_root and onerror is not None:
            onerror(PermissionError("synthetic traversal failure"))
        yield from original_walk(top, *args, **kwargs)

    monkeypatch.setattr(assembler_module.os, "walk", walk_with_resource_error)

    with pytest.raises(assembler_module.AssemblyError, match="cannot enumerate resource 'assets'"):
        assembler_module.assemble("mcp", output, source_root)
    assert not output.exists()


@pytest.mark.parametrize("source_path", ["../outside.txt", ".env.local", "missing.bin"])
def test_missing_traversing_or_excluded_resources_fail_without_publish(tmp_path: Path, source_path: str) -> None:
    source_root, output = _project(tmp_path, {"main.py": "VALUE = 1\n", ".env.local": "secret"})
    _set_resources(source_root, [{"source_path": source_path, "runtime_path": "asset.bin"}])

    result = _run(source_root, output)

    assert result.returncode != 0
    assert not output.exists()


def test_remap_resource_and_portable_case_collisions_fail_before_publish(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {"main.py": "import src.dep\n", "src/dep.py": "VALUE = 1\n", "assets/file": "x"},
        path_remaps={"src/dep.py": "DATA/item.py"},
    )
    _set_resources(source_root, [{"source_path": "assets/file", "runtime_path": "data/ITEM.py"}])

    result = _run(source_root, output)

    assert result.returncode != 0
    assert "collision" in (result.stdout + result.stderr).lower()
    assert not output.exists()


@pytest.mark.parametrize(
    ("path_remaps", "resource_runtime"),
    [
        ({"src/dep.py": "Tests/dep.py"}, None),
        ({}, ".ENV.private/data.bin"),
    ],
)
def test_excluded_runtime_destinations_fail_for_sources_and_resources(
    tmp_path: Path,
    path_remaps: dict[str, str],
    resource_runtime: str | None,
) -> None:
    files = {"main.py": "import src.dep\n", "src/dep.py": "VALUE = 1\n", "assets/file.bin": "asset"}
    source_root, output = _project(tmp_path, files, path_remaps=path_remaps)
    if resource_runtime is not None:
        _set_resources(source_root, [{"source_path": "assets/file.bin", "runtime_path": resource_runtime}])

    result = _run(source_root, output)

    assert result.returncode != 0
    assert "excluded" in (result.stdout + result.stderr).lower()
    assert not output.exists()


def test_output_must_be_inside_source_build_and_failed_rebuild_preserves_output(tmp_path: Path) -> None:
    source_root, output = _project(tmp_path, {"main.py": "VALUE = 1\n"})
    assert _run(source_root, output).returncode == 0
    before = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}

    unsafe = _run(source_root, tmp_path / "outside")
    assert unsafe.returncode != 0
    assert "strict descendant" in (unsafe.stdout + unsafe.stderr)

    _set_resources(source_root, [{"source_path": "main.py", "runtime_path": "main.py"}])
    failed_rebuild = _run(source_root, output)

    assert failed_rebuild.returncode != 0
    after = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before
    assert not list(output.parent.glob(".assemble-stage-*"))
    assert not list(output.parent.glob(".assemble-backup-*"))


def test_report_and_output_bytes_are_deterministic_and_replacement_is_complete(tmp_path: Path) -> None:
    source_root, output = _project(tmp_path, {"main.py": "import dependency\n", "dependency.py": "VALUE = 2\n"})
    assert _run(source_root, output).returncode == 0
    first = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    # Include the generated artifact in the next source analysis; .build remains excluded.
    assert _run(source_root, output).returncode == 0
    second = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert first == second
    report = json.loads((output / "assembly-report.json").read_text(encoding="utf-8"))
    assert report["config_hashes"]["service_manifest_sha256"]
    assert all(entry["sha256"] for entry in report["source_files"])
    assert "runtime_target" in report
    assert not any(".build" in path for path in _source_paths(report))
