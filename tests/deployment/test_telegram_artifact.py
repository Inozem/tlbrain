"""Telegram runtime artifact and dependency-closure checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = APP_ROOT / "infra" / "build"
ASSEMBLER = BUILD_TOOLS / "assemble.py"
MANIFEST = BUILD_TOOLS / "service-manifests.json"
SERVICE = "telegram"
ENTRY = "services/connectors/telegram/app/main.py"
RUNTIME_TARGET = "services.connectors.telegram.app.main:app"
REQUIREMENTS_ROOT = "services/connectors/telegram/requirements.txt"
FORBIDDEN_PARTS = {
    ".git", ".venv", "venv", "tests", "test", "fixtures", "migrations",
    "infra", "build", "dist", "toolchain", "__pycache__",
}


def _artifact_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_telegram_assembles_to_a_minimal_runtime_artifact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = manifest["services"][SERVICE]
    assert config["entry_roots"] == [ENTRY]
    assert config["runtime"] == {"kind": "asgi", "target": RUNTIME_TARGET}
    assert config["requirements_roots"] == [REQUIREMENTS_ROOT]

    build_root = APP_ROOT / ".build"
    build_root.mkdir(exist_ok=True)
    assert not build_root.is_symlink()
    output = build_root / f"test-telegram-artifact-{uuid.uuid4().hex}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ASSEMBLER),
                "--service",
                SERVICE,
                "--output",
                str(output),
                "--source-root",
                str(APP_ROOT),
            ],
            cwd=APP_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stdout + result.stderr

        report: dict[str, Any] = json.loads(
            (output / "assembly-report.json").read_text(encoding="utf-8")
        )
        assert report["service"] == SERVICE
        assert report["runtime_target"] == RUNTIME_TARGET
        source_paths = {item["runtime_path"] for item in report["source_files"]}
        assert ENTRY in source_paths
        paths = _artifact_paths(output)
        assert ENTRY in paths
        assert "requirements.txt" in paths
        assert "assembly-report.json" in paths
        assert source_paths <= paths

        assert set(report["provider_roots"]) == {
            "core/integrations/cloud_tasks/requirements.txt",
            "core/integrations/firestore/requirements.txt",
            "core/integrations/google_auth/requirements.txt",
        }
        requirement_paths = {
            item["runtime_path"] for item in report["requirements_graph"]["files"]
        }
        assert requirement_paths == {
            ".requirements/core/integrations/cloud_tasks/requirements.txt",
            ".requirements/core/integrations/firestore/requirements.txt",
            ".requirements/core/integrations/google_auth/requirements.txt",
            ".requirements/requirements-common.txt",
            f".requirements/{REQUIREMENTS_ROOT}",
        }
        assert requirement_paths <= paths
        assert "httpx>=0.27" in (
            output / ".requirements" / REQUIREMENTS_ROOT
        ).read_text(encoding="utf-8")

        for path in paths:
            parts = PurePosixPath(path).parts
            folded = tuple(part.casefold() for part in parts)
            assert not any(part in FORBIDDEN_PARTS for part in folded), path
            assert not any(part.startswith(".env") for part in folded), path
            assert PurePosixPath(path).name.casefold() not in {
                "requirements-dev.txt", "dev-requirements.txt", "test-requirements.txt",
            }, path
    finally:
        resolved_output = output.resolve()
        if (
            resolved_output.parent == build_root.resolve()
            and output.exists()
            and not output.is_symlink()
        ):
            shutil.rmtree(output)
