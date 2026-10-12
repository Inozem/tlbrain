"""Cross-cutting checks for service artifacts assembled from the current checkout."""

from __future__ import annotations

import json
import ast
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
EXPECTED_INTEGRATION_REQUIREMENTS = {
    "mcp": {
        "fastembed", "firestore", "gemini", "google_auth", "google_docs", "qdrant",
    },
    "vector_sync": {
        "fastembed", "firestore", "gemini", "google_auth", "google_docs", "qdrant",
    },
    "tldv_import": {
        "cloud_tasks", "firestore", "gemini", "google_auth", "google_docs",
    },
    "sync_checker": {
        "cloud_tasks", "firestore", "google_auth", "google_docs",
    },
    "tldv_webhook": {"cloud_tasks", "firestore", "google_auth"},
    "tldv_reconciliation": {"cloud_tasks", "firestore", "google_auth"},
}
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
    if PurePosixPath(path).as_posix() == "core/integrations/google_auth/credentials.py":
        return False
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
    assert not _looks_credential_like("core/integrations/google_auth/credentials.py")


def test_integration_packages_are_passive_and_own_existing_sdk_bounds() -> None:
    integrations = {
        "google_auth": "google-auth\ngoogle-auth-httplib2",
        "google_drive": "# Google API client dependency is owned by the shared Google Docs API bundle.",
        "google_docs": "google-api-python-client",
        "firestore": "google-cloud-firestore",
        "qdrant": "qdrant-client>=1.9",
        "gemini": "google-genai>=1.0",
        "fastembed": "fastembed",
        "cloud_tasks": "google-cloud-tasks>=2.0.0",
    }
    expected_providers = {
        "google.auth": "google_auth",
        "google.oauth2": "google_auth",
        "google_auth_httplib2": "google_auth",
        "googleapiclient": "google_docs",
        "google.cloud.firestore": "firestore",
        "google.genai": "gemini",
        "qdrant_client": "qdrant",
        "fastembed": "fastembed",
        "google.api_core": "cloud_tasks",
        "google.cloud.tasks_v2": "cloud_tasks",
    }

    for name, declarations in integrations.items():
        package = APP_ROOT / "core" / "integrations" / name
        initializer = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
        assert len(initializer.body) == 1
        assert isinstance(initializer.body[0], ast.Expr)
        assert isinstance(initializer.body[0].value, ast.Constant)
        assert isinstance(initializer.body[0].value.value, str)
        assert (package / "requirements.txt").read_text(encoding="utf-8").strip() == declarations

    registry = json.loads((BUILD_TOOLS / "dependency-providers.json").read_text(encoding="utf-8"))
    actual_providers = {
        prefix: PurePosixPath(requirement_file).parent.name
        for provider in registry["providers"]
        for prefix in provider["import_prefixes"]
        for requirement_file in provider["requirements_files"]
    }
    assert actual_providers == expected_providers
    assert all(not prefix == "google" for prefix in actual_providers)
    assert "../integrations/firestore/requirements.txt" in (
        APP_ROOT / "core" / "google_drive" / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert not (APP_ROOT / "core" / "utils" / "requirements.txt").exists()
    assert (APP_ROOT / "core" / "integrations" / "cloud_tasks" / "requirements.txt").read_text(
        encoding="utf-8"
    ).strip() == "google-cloud-tasks>=2.0.0"


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
            selected_integrations = {
                PurePosixPath(item["runtime_path"]).parent.name
                for item in report["requirements_graph"]["files"]
                if item["runtime_path"].startswith(".requirements/core/integrations/")
            }
            assert selected_integrations == EXPECTED_INTEGRATION_REQUIREMENTS[service]
            _assert_runtime_only(artifact, service, report)

        assert IMAGE_SERVICES == {name for name, (kind, _target) in EXPECTED_SERVICES.items() if kind == "asgi"}
    finally:
        resolved_root = output_root.resolve()
        if resolved_root.parent == build_root.resolve() and output_root.exists():
            shutil.rmtree(output_root)
