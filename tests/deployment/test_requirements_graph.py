"""Contract tests for deployment requirements graph assembly.

Each test invokes the assembler as a subprocess against a tiny synthetic
checkout. The synthetic files are never installed or imported.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = APP_ROOT / "infra" / "build" / "assemble.py"


def _project(
    tmp_path: Path,
    files: dict[str, str],
    *,
    requirements_roots: list[str] | None = None,
    providers: list[dict[str, object]] | None = None,
    optional_imports: list[dict[str, str]] | None = None,
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
                "entry_roots": ["main.py"],
                "runtime": {},
                "path_remaps": {},
                "module_aliases": {},
                "requirements_roots": requirements_roots or [],
                "extra_modules": [],
                "resources": [],
                "model_preparation": [],
            }
        },
    }
    registry = {
        "schema_version": 1,
        "providers": providers or [],
        "optional_imports": optional_imports or [],
    }
    (build_root / "service-manifests.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (build_root / "dependency-providers.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    return source_root, source_root / ".build" / "assembled"


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


def _assert_success(source_root: Path, output: Path) -> None:
    result = _run(source_root, output)
    assert result.returncode == 0, result.stderr or result.stdout
    assert output.is_dir()
    top_level = output / "requirements.txt"
    assert top_level.is_file()
    includes = [
        line
        for line in top_level.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*-[rc]\s+", line)
    ]
    assert includes
    requirements_root = (output / ".requirements").resolve()
    for line in includes:
        target = _include_target(top_level, line)
        assert target.is_file()
        assert target.is_relative_to(requirements_root)


def _staged_requirements(output: Path) -> dict[str, Path]:
    requirements_root = output / ".requirements"
    return {
        path.relative_to(requirements_root).as_posix(): path
        for path in requirements_root.rglob("*")
        if path.is_file()
    }


def _include_target(declaring_file: Path, directive: str) -> Path:
    included = re.sub(r"^\s*-[rc]\s+", "", directive).strip()
    return (declaring_file.parent / included).resolve()


def test_nested_includes_resolve_from_declaring_file_and_are_rewritten(
    tmp_path: Path,
) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/base.txt": "-r nested/runtime.txt\n-c constraints/pins.txt\n",
            "requirements/nested/runtime.txt": "-r ../leaves/leaf.txt\n",
            "requirements/leaves/leaf.txt": "sample-sdk>=2\n",
            "requirements/constraints/pins.txt": "sample-sdk<4\n",
        },
        requirements_roots=["requirements/base.txt"],
    )

    _assert_success(source_root, output)
    staged = _staged_requirements(output)
    assert len(staged) == 4
    assert {path.name for path in staged.values()} == {
        "base.txt",
        "runtime.txt",
        "leaf.txt",
        "pins.txt",
    }

    for staged_file in staged.values():
        for line in staged_file.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*-[rc]\s+", line):
                assert _include_target(staged_file, line).is_file()

    result = _run(source_root, output)
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads((output / "assembly-report.json").read_text(encoding="utf-8"))
    graph = report["requirements_graph"]
    assert len(graph["roots"]) == 1
    assert len(graph["files"]) == 4
    assert all(item["sha256"] for item in graph["files"])


def test_selected_requirements_destination_collision_fails_without_replacing_output(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/base.txt": "sample-sdk>=1\n",
            "assets/duplicate": "resource",
        },
        requirements_roots=["requirements/base.txt"],
    )
    assert _run(source_root, output).returncode == 0
    previous = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    manifest_path = source_root / "infra" / "build" / "service-manifests.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["services"]["mcp"]["resources"] = [{
        "source_path": "assets/duplicate",
        "runtime_path": ".requirements/requirements/base.txt",
    }]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run(source_root, output)

    assert result.returncode != 0
    assert "collision" in (result.stdout + result.stderr).lower()
    assert {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()} == previous


def test_leaf_syntax_and_local_resources_are_preserved_and_staged(
    tmp_path: Path,
) -> None:
    leaf = (
        "sample-sdk[fast,cli]>=2.1,<3; python_version >= '3.11' "
        "--only-binary=:all: \\\n        --hash=sha256:0123456789abcdef\n"
        "https://example.invalid/sample.whl \\\n        --hash=sha256:abcdef0123456789\n"
        "./wheels/local-addon.tar.gz\n"
    )
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/runtime.txt": leaf,
            "requirements/wheels/local-addon.tar.gz": "synthetic archive bytes\n",
        },
        requirements_roots=["requirements/runtime.txt"],
    )

    _assert_success(source_root, output)
    staged = _staged_requirements(output)
    staged_leaf = next(path for path in staged.values() if path.name == "runtime.txt")
    contents = staged_leaf.read_text(encoding="utf-8")
    assert "sample-sdk[fast,cli]>=2.1,<3; python_version >= '3.11'" in contents
    assert "--only-binary=:all:" in contents
    assert "--hash=sha256:0123456789abcdef" in contents
    assert "https://example.invalid/sample.whl" in contents
    assert "--hash=sha256:abcdef0123456789" in contents
    local_resource_lines = [
        line for line in contents.splitlines() if "local-addon.tar.gz" in line
    ]
    assert len(local_resource_lines) == 1
    resource = _include_target(staged_leaf, local_resource_lines[0])
    assert resource.is_file()
    assert resource.read_text(encoding="utf-8") == "synthetic archive bytes\n"


def test_missing_include_fails_with_missing_path_in_diagnostic(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/runtime.txt": "-r absent/child.txt\n",
        },
        requirements_roots=["requirements/runtime.txt"],
    )

    result = _run(source_root, output)
    diagnostic = result.stdout + result.stderr
    assert result.returncode != 0
    assert "absent" in diagnostic or "child.txt" in diagnostic


def test_include_cycle_fails_with_cycle_fact_in_diagnostic(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/first.txt": "-r second.txt\n",
            "requirements/second.txt": "-c first.txt\n",
        },
        requirements_roots=["requirements/first.txt"],
    )

    result = _run(source_root, output)
    diagnostic = (result.stdout + result.stderr).lower()
    assert result.returncode != 0
    assert "cycle" in diagnostic or "cyclic" in diagnostic


def test_conflicting_package_declarations_remain_in_both_source_files(
    tmp_path: Path,
) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": "VALUE = 1\n",
            "requirements/runtime.txt": "-r first.txt\n-r second.txt\n",
            "requirements/first.txt": "sample-sdk==1.0\n",
            "requirements/second.txt": "sample-sdk==2.0\n",
        },
        requirements_roots=["requirements/runtime.txt"],
    )

    _assert_success(source_root, output)
    staged = _staged_requirements(output)
    declarations = "\n".join(
        path.read_text(encoding="utf-8") for path in staged.values()
    )
    assert "sample-sdk==1.0" in declarations
    assert "sample-sdk==2.0" in declarations


def _provider(prefix: str, requirement_file: str) -> dict[str, object]:
    return {
        "import_prefixes": [prefix],
        "requirements_files": [requirement_file],
    }


def _optional(source_file: str, prefix: str, guard: str) -> dict[str, str]:
    return {
        "source_file": source_file,
        "import_prefix": prefix,
        "guard_context": guard,
        "reason": "The synthetic integration is an optional capability.",
    }


def _selected_requirements(output: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in _staged_requirements(output).values()
    )


def test_optional_only_qualified_provider_is_excluded(tmp_path: Path) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": (
                "try:\n"
                "    import acme.optional_sdk.client\n"
                "except ImportError:\n"
                "    optional_sdk = None\n"
            ),
            "requirements/service.txt": "",
            "requirements/optional-sdk.txt": "optional-sdk-dist>=1\n",
        },
        requirements_roots=["requirements/service.txt"],
        providers=[_provider("acme.optional_sdk", "requirements/optional-sdk.txt")],
        optional_imports=[
            _optional("main.py", "acme.optional_sdk", "ImportError")
        ],
    )

    _assert_success(source_root, output)
    assert "optional-sdk-dist" not in _selected_requirements(output)


def test_same_prefix_required_elsewhere_selects_explicit_provider_file(
    tmp_path: Path,
) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": (
                "try:\n"
                "    import acme.optional_sdk.client\n"
                "except ImportError:\n"
                "    optional_sdk = None\n"
                "import app.required\n"
            ),
            "app/__init__.py": "",
            "app/required.py": "import acme.optional_sdk.required_api\n",
            "requirements/service.txt": "",
            "requirements/provider.txt": "mapped-distribution>=7\n",
        },
        requirements_roots=["requirements/service.txt"],
        providers=[_provider("acme.optional_sdk", "requirements/provider.txt")],
        optional_imports=[
            _optional("main.py", "acme.optional_sdk", "ImportError")
        ],
    )

    _assert_success(source_root, output)
    selected = _selected_requirements(output)
    assert "mapped-distribution>=7" in selected
    assert "acme.optional_sdk" not in selected


def test_optional_type_checking_import_is_excluded_by_qualified_context(
    tmp_path: Path,
) -> None:
    source_root, output = _project(
        tmp_path,
        {
            "main.py": (
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    import typing_vendor.sdk.models\n"
            ),
            "requirements/service.txt": "",
            "requirements/typing-sdk.txt": "typing-sdk-dist>=1\n",
        },
        requirements_roots=["requirements/service.txt"],
        providers=[
            _provider("typing_vendor.sdk", "requirements/typing-sdk.txt")
        ],
        optional_imports=[
            _optional("main.py", "typing_vendor.sdk", "TYPE_CHECKING")
        ],
    )

    _assert_success(source_root, output)
    assert "typing-sdk-dist" not in _selected_requirements(output)
