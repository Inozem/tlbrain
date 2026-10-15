"""End-to-end coverage for adding a new statically imported core module."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = APP_ROOT / "infra" / "build" / "assemble.py"


def _checkout(
    tmp_path: Path,
    files: dict[str, str],
    *,
    provider_registry: dict[str, object] | None = None,
    resources: list[dict[str, str]] | None = None,
    requirements_roots: list[str] | None = None,
    runtime: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "checkout"
    build = root / "infra" / "build"
    build.mkdir(parents=True)
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "services": {
            "mcp": {
                "entry_roots": ["main.py"],
                "runtime": runtime or {},
                "path_remaps": {},
                "module_aliases": {},
                "requirements_roots": requirements_roots or [],
                "extra_modules": [],
                "resources": resources or [],
                "model_preparation": [],
            }
        },
    }
    (build / "service-manifests.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = provider_registry or {"schema_version": 1, "providers": [], "optional_imports": []}
    (build / "dependency-providers.json").write_text(json.dumps(registry), encoding="utf-8")
    return root, root / ".build" / "assembled"


def _run(root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--service",
            "mcp",
            "--output",
            str(output),
            "--source-root",
            str(root),
        ],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "infra" / "build" / "service-manifests.json").read_text(encoding="utf-8"))


def test_new_core_module_closes_local_and_provider_graph_without_service_list_edits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "source-import-ran"
    root, output = _checkout(
        tmp_path,
        {
            "main.py": "import app.entry\n",
            "app/__init__.py": "",
            "app/entry.py": "import core.new_feature\n",
            "core/__init__.py": "from . import shared\n",
            "core/new_feature.py": (
                "import provider_sdk.client\n"
                "from core import helper\n"
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
            ),
            "core/shared.py": "VALUE = 1\n",
            "core/helper.py": "from core.shared import VALUE\n",
            "requirements/provider.txt": "-r nested/runtime.txt\n",
            "requirements/nested/runtime.txt": "provider-sdk-dist>=4\n",
            "assets/model.bin": "synthetic model bytes",
        },
        provider_registry={
            "schema_version": 1,
            "providers": [
                {
                    "import_prefixes": ["provider_sdk"],
                    "requirements_files": ["requirements/provider.txt"],
                }
            ],
            "optional_imports": [],
        },
        resources=[{"source_path": "assets/model.bin", "runtime_path": "data/model.bin"}],
    )

    result = _run(root, output)

    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads((output / "assembly-report.json").read_text(encoding="utf-8"))
    source_paths = {entry["source_path"] for entry in report["source_files"]}
    assert {
        "main.py",
        "app/entry.py",
        "core/__init__.py",
        "core/new_feature.py",
        "core/shared.py",
        "core/helper.py",
    } <= source_paths
    assert (output / "core" / "new_feature.py").is_file()
    assert (output / "core" / "__init__.py").is_file()
    assert (output / "core" / "shared.py").is_file()

    edge = next(edge for edge in report["import_edges"] if edge["import"] == "core.new_feature")
    assert edge["source_path"] == "app/entry.py"
    external = next(
        item for item in report["required_external_imports"]
        if item["name"] == "provider_sdk.client"
    )
    assert external["source_path"] == "core/new_feature.py"
    assert external["dependency_chain"] == ["main.py", "app/entry.py", "core/new_feature.py"]

    requirements = output / "requirements.txt"
    assert "provider-sdk-dist" in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output / ".requirements").rglob("*")
        if path.is_file()
    )
    assert report["provider_roots"] == ["requirements/provider.txt"]
    assert "requirements/provider.txt" in requirements.read_text(encoding="utf-8")
    assert (output / "data" / "model.bin").read_bytes() == b"synthetic model bytes"
    assert not marker.exists()

    service = _manifest(root)["services"]["mcp"]
    assert service["entry_roots"] == ["main.py"]
    assert service["extra_modules"] == []


def test_missing_transitive_core_import_fails_with_chain_before_publish(tmp_path: Path) -> None:
    root, output = _checkout(
        tmp_path,
        {
            "main.py": "import app.entry\n",
            "app/__init__.py": "",
            "app/entry.py": "import core.new_feature\n",
            "core/__init__.py": "",
            "core/new_feature.py": "from core import missing_dependency\n",
        },
    )

    result = _run(root, output)
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0
    assert "missing_dependency" in diagnostic
    assert "core/new_feature.py" in diagnostic
    assert "main.py" in diagnostic
    assert "app/entry.py" in diagnostic
    assert "dependency chain" in diagnostic.lower()
    assert not output.exists()


def test_unresolved_external_import_without_provider_fails_actionably(tmp_path: Path) -> None:
    root, output = _checkout(
        tmp_path,
        {
            "main.py": (
                "import app.entry\n"
                "async def app(scope, receive, send):\n"
                "    return None\n"
            ),
            "app/__init__.py": "",
            "app/entry.py": "import unknown_vendor.client\n",
            "requirements/runtime.txt": "",
        },
        requirements_roots=["requirements/runtime.txt"],
        runtime={"kind": "asgi", "target": "main:app"},
    )

    assembled = _run(root, output)
    assert assembled.returncode == 0, assembled.stderr or assembled.stdout
    report = json.loads((output / "assembly-report.json").read_text(encoding="utf-8"))
    external = next(
        item for item in report["required_external_imports"]
        if item["name"] == "unknown_vendor.client"
    )
    assert external["source_path"] == "app/entry.py"
    assert external["dependency_chain"] == ["main.py", "app/entry.py"]

    validation_result_path = tmp_path / "validation-result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(APP_ROOT / "infra" / "build" / "validate.py"),
            "--service",
            "mcp",
            "--artifact",
            str(output),
            "--offline",
            "--result",
            str(validation_result_path),
        ],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0
    assert "unknown_vendor.client" in diagnostic
    validation = json.loads(validation_result_path.read_text(encoding="utf-8"))
    assert validation["status"] == "failed"


def test_missing_declared_resource_fails_without_publishing_artifact(tmp_path: Path) -> None:
    root, output = _checkout(
        tmp_path,
        {"main.py": "import core.new_feature\n", "core/new_feature.py": "VALUE = 1\n"},
        resources=[{"source_path": "assets/model.bin", "runtime_path": "data/model.bin"}],
    )

    result = _run(root, output)
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0
    assert "assets/model.bin" in diagnostic
    assert not output.exists()
