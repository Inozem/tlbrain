"""Cross-cutting checks for service artifacts assembled from the current checkout."""

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

EXPECTED_SERVICES = {
    "mcp": ("asgi", "services.mcp.app.main:app"),
    "vector_sync": ("asgi", "services.vector_sync.app.main:app"),
    "tldv_import": ("asgi", "app.main:app"),
    "sync_checker": ("function", "checker"),
    "tldv_webhook": ("function", "tldv_webhook"),
    "tldv_reconciliation": ("function", "tldv_reconciliation"),
}
EXPECTED_ENTRIES = {
    "mcp": "services/mcp/app/main.py",
    "vector_sync": "services/vector_sync/app/main.py",
    "tldv_import": "app/main.py",
    "sync_checker": "main.py",
    "tldv_webhook": "main.py",
    "tldv_reconciliation": "main.py",
}
IMAGE_SERVICES = {"mcp", "vector_sync", "tldv_import"}
SENSITIVE_NAME_MARKERS = (
    "_secret_", "_secrets_", "_credential_", "_credentials_",
    "_service_account_", "_private_key_", "_privatekey_", "_id_rsa_",
    "_id_ed25519_",
)
PRIVATE_KEY_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".ppk"}
COMMON_CREDENTIAL_FILENAMES = {
    "credentials.json", "service-account.json", "service_account.json",
    "application_default_credentials.json", "authorized_keys",
}
FORBIDDEN_PARTS = {
    ".git", ".venv", "venv", "tests", "test", "fixtures", "migrations",
    "infra", "build", "dist", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "toolchain", "__pycache__",
}


def _looks_credential_like(path: str) -> bool:
    normalized = "_" + "_".join(
        "".join(
            character.lower() if character.isalnum() else "_"
            for character in PurePosixPath(part).stem
        ).strip("_")
        for part in PurePosixPath(path).parts
    ) + "_"
    basename = PurePosixPath(path).name.casefold()
    return (
        basename in COMMON_CREDENTIAL_FILENAMES
        or PurePosixPath(basename).suffix in PRIVATE_KEY_SUFFIXES
        or any(marker in normalized for marker in SENSITIVE_NAME_MARKERS)
        or basename in {"id_rsa", "id_ecdsa", "id_ed25519"}
    )


def _artifact_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_runtime_only(artifact: Path, service: str, report: dict[str, Any]) -> None:
    paths = _artifact_paths(artifact)
    assert "requirements.txt" in paths
    assert "assembly-report.json" in paths
    assert any(path.endswith(".py") for path in paths)

    for path in paths:
        parts = PurePosixPath(path).parts
        folded = tuple(part.casefold() for part in parts)
        assert not any(part in FORBIDDEN_PARTS for part in folded), path
        assert not any(part.startswith(".env") for part in folded), path
        assert not _looks_credential_like(path), path
        assert path.casefold() not in {
            "requirements-dev.txt", "readme.md", ".dockerignore", ".gitignore",
        }

    target_file = EXPECTED_ENTRIES[service]
    assert target_file in paths
    source_paths = {entry["runtime_path"] for entry in report["source_files"]}
    assert target_file in source_paths
    assert all(path in paths for path in source_paths)

    requirement_files = report["requirements_graph"]["files"]
    assert requirement_files
    assert all(item["runtime_path"] in paths for item in requirement_files)
    assert report["requirements_graph"]["roots"]


def test_credential_like_artifact_paths_are_rejected() -> None:
    credential_paths = (
        "service-account.json",
        "credentials.json",
        "application_default_credentials.json",
        "secret.json",
        "secrets.json",
        "config/client_secret.json",
        "config/my_secrets/token.txt",
        "keys/private-key.pem",
        "keys/server.key",
        ".ssh/id_rsa",
        ".ssh/id_ed25519",
    )
    assert all(_looks_credential_like(path) for path in credential_paths)


def test_all_six_current_service_sources_assemble_to_isolated_runtime_artifacts() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    services = manifest["services"]
    assert set(services) == set(EXPECTED_SERVICES)

    build_root = APP_ROOT / ".build"
    build_root.mkdir(exist_ok=True)
    assert not build_root.is_symlink()
    output_root = build_root / f"test-foundation-{uuid.uuid4().hex}"
    assert output_root.parent.resolve() == build_root.resolve()

    try:
        for service, (runtime_kind, runtime_target) in EXPECTED_SERVICES.items():
            config = services[service]
            assert config["runtime"] == {"kind": runtime_kind, "target": runtime_target}
            artifact = output_root / service
            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--service",
                    service,
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
                env=os.environ.copy(),
            )
            assert result.returncode == 0, f"{service}: {result.stdout}\n{result.stderr}"
            assert artifact.is_dir()
            report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
            assert report["service"] == service
            assert report["runtime_target"] == runtime_target
            assert report["source_files"]
            assert report["import_edges"]
            _assert_runtime_only(artifact, service, report)

        assert IMAGE_SERVICES == {name for name, (kind, _target) in EXPECTED_SERVICES.items() if kind == "asgi"}
    finally:
        resolved_root = output_root.resolve()
        if resolved_root.parent == build_root.resolve() and output_root.exists():
            shutil.rmtree(output_root)
