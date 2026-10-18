"""Offline checks for the three assembled service runtime images."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = APP_ROOT / "infra" / "build"
DOCKERFILES = {
    "mcp": APP_ROOT / "infra" / "docker" / "Dockerfile.mcp",
    "vector_sync": APP_ROOT / "infra" / "docker" / "Dockerfile.vector_sync",
    "tldv_import": APP_ROOT / "infra" / "docker" / "Dockerfile.tldv_import",
}


def _assemble_module() -> Any:
    path = BUILD_TOOLS / "assemble.py"
    spec = importlib.util.spec_from_file_location("tlbrain_docker_runtime_assemble", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dockerfile_runtime(service: str) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in DOCKERFILES[service].read_text(encoding="utf-8").splitlines()]
    from_lines = [line for line in lines if line.startswith("FROM ")]
    assert from_lines == ["FROM python:3.12-slim AS builder", "FROM python:3.12-slim"]
    runtime_start = lines.index("FROM python:3.12-slim")
    return lines, lines[runtime_start:]


def _assemble_service(service: str, temporary_root: Path) -> tuple[Path, Path]:
    build_root = APP_ROOT / ".build"
    if build_root.is_symlink():
        raise AssertionError("refusing to assemble test artifact through a .build symlink")
    output_root = build_root / f"docker-runtime-{temporary_root.name}"
    if output_root.exists() or output_root.is_symlink():
        raise AssertionError(f"refusing to reuse generated test artifact path: {output_root}")
    output = output_root / service
    _assemble_module().assemble(service, output, APP_ROOT)
    return output, output_root


@pytest.mark.parametrize("service", ["mcp", "vector_sync", "tldv_import"])
def test_dockerfile_and_manifest_describe_the_runtime_artifact(service: str) -> None:
    manifest = json.loads((BUILD_TOOLS / "service-manifests.json").read_text(encoding="utf-8"))
    config = manifest["services"][service]
    lines, runtime = _dockerfile_runtime(service)
    expected_artifact = f"/src/.build/{service}/"
    expected_command = [
        "python", "-m", "uvicorn", config["runtime"]["target"],
        "--host", "0.0.0.0", "--port", "8080",
    ]

    assert f"RUN python infra/build/assemble.py --service {service} --source-root /src --output /src/.build/{service}" in lines
    assert f"COPY --from=builder {expected_artifact} /app/" in runtime
    assert [line for line in runtime if line.startswith("COPY ")] == [
        f"COPY --from=builder {expected_artifact} /app/"
    ]
    assert "WORKDIR /app" in runtime
    assert "RUN pip install --no-cache-dir -r /app/requirements.txt" in runtime
    assert "ENV PORT=8080" in runtime
    assert "CMD " + json.dumps(expected_command) in runtime
    metadata = importlib.import_module("infra.build.image_metadata")
    assert metadata._SERVICE_CMDS[service] == expected_command
    assert config["runtime"]["kind"] == "asgi"
    assert config["model_preparation"] == ([] if service == "tldv_import" else ["Qdrant/bm25"])

    prep_lines = [line for line in runtime if "SparseTextEmbedding" in line or "fastembed" in line.lower()]
    if service == "tldv_import":
        assert not prep_lines
    else:
        assert prep_lines == [
            "RUN python -c \"from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')\""
        ]


@pytest.mark.parametrize("service", ["mcp", "vector_sync", "tldv_import"])
def test_real_image_preflight_validates_runtime_cache_and_binds_evidence(
    service: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    external_temp = tmp_path / "external-temp"
    external_temp.mkdir()
    monkeypatch.setattr(metadata.tempfile, "tempdir", str(external_temp))
    manifest = json.loads((BUILD_TOOLS / "service-manifests.json").read_text(encoding="utf-8"))
    config = manifest["services"][service]
    artifact, generated_root = _assemble_service(service, tmp_path)
    try:
        report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
        runtime_files = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
        source_paths = {record["runtime_path"] for record in report["source_files"]}
        assert source_paths == {path for path in runtime_files if path.endswith(".py")}
        assert {"assembly-report.json", "requirements.txt"} <= runtime_files
        assert not any(
            part.lower() in {"tests", "test", "fixtures", ".venv", "venv", "build", ".build"}
            or part.lower().startswith(".env")
            for path in runtime_files
            for part in Path(path).parts
        )
        assert not any(path.startswith("infra/build/") or path.startswith("tests/") for path in runtime_files)

        requirements = "\n".join(
            (artifact / record["runtime_path"]).read_text(encoding="utf-8")
            for record in report["requirements_graph"]["files"]
        ).lower()
        assert ("fastembed" in requirements) is (service in {"mcp", "vector_sync"})
        assert report["runtime_target"] == config["runtime"]["target"]

        class FakeDocker:
            def __init__(self) -> None:
                self.config = {
                    "image_id": f"sha256:immutable-{service}",
                    "os": "linux",
                    "architecture": "amd64",
                    "rootfs": {"diff_ids": [f"sha256:layer-{service}"]},
                    "config": {
                        "Entrypoint": None,
                        "Cmd": metadata._SERVICE_CMDS[service],
                        "WorkingDir": "/app",
                        "User": None,
                        "Env": ["PYTHONUNBUFFERED=1", "PORT=8080"],
                        "Labels": {},
                    },
                }
                self.image_commands: list[list[str]] = []
                self.validated_services: list[str] = []
                self.cache_probes: list[str] = []
                self.bound_evidence: dict[str, Any] | None = None

            def inspect_image(self, _image: str) -> dict[str, Any]:
                return json.loads(json.dumps(self.config))

            def run(self, image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
                assert image == f"sha256:immutable-{service}"
                self.image_commands.append(command)
                if "assembly-report.json" in command[-1]:
                    return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
                if any(item.endswith("validate.py") for item in command):
                    harness = next(host for host, destination, readonly in mounts if destination == "/tlbrain-validation" and readonly)
                    result_dir = next(host for host, destination, readonly in mounts if destination == "/tlbrain-validation-result" and not readonly)
                    result = result_dir / "result.json"
                    completed = subprocess.run(
                        [sys.executable, "-I", str(harness / "validate.py"), "--service", service,
                         "--artifact", str(artifact), "--offline", "--result", str(result)],
                        check=False, text=True, capture_output=True,
                    )
                    self.validated_services.append(service)
                    return SimpleNamespace(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
                assert "Qdrant/bm25" in command[-1]
                self.cache_probes.append(service)

                class FakeSparseTextEmbedding:
                    def __init__(self, *, model_name: str) -> None:
                        assert model_name == "Qdrant/bm25"

                    def embed(self, _texts: list[str]) -> list[Any]:
                        return [SimpleNamespace(indices=[3], values=[0.75])]

                sys_module = SimpleNamespace(SparseTextEmbedding=FakeSparseTextEmbedding)
                import sys as python_sys

                old_module = python_sys.modules.get("fastembed")
                python_sys.modules["fastembed"] = sys_module
                try:
                    exec(command[-1], {})
                finally:
                    if old_module is None:
                        python_sys.modules.pop("fastembed", None)
                    else:
                        python_sys.modules["fastembed"] = old_module
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def attach_label(self, _image: str, key: str, value: str) -> None:
                assert key == metadata.VALIDATION_LABEL
                self.bound_evidence = json.loads(value)
                self.config["config"]["Labels"][key] = value

        docker = FakeDocker()
        result = metadata.run_image_preflight(
            docker,
            f"{service}:test",
            service,
            {"repository": "offline/test", "run_id": "1", "attempt": "1"},
            build_tools=BUILD_TOOLS,
        )

        assert docker.validated_services == [service]
        needs_cache = service in {"mcp", "vector_sync"}
        assert docker.cache_probes == ([service] if needs_cache else [])
        assert len(docker.image_commands) == 2 + int(needs_cache)
        validator_command = next(command for command in docker.image_commands if any(item.endswith("validate.py") for item in command))
        assert "--offline" in validator_command
        assert docker.bound_evidence is not None
        assert docker.bound_evidence["service"] == service
        assert docker.bound_evidence["target"] == config["runtime"]["target"]
        assert docker.bound_evidence["checks"] == ["compile", "imports", "startup", *(["cache"] if needs_cache else [])]
        assert docker.bound_evidence["status"] == "passed"
        assert docker.bound_evidence["python_version"] == "3.12"
        assert docker.bound_evidence["platform"] == "linux/amd64"
        assert docker.bound_evidence["runtime_fingerprint"] == metadata.runtime_fingerprint(result)
        assert result["image_id"] == f"sha256:immutable-{service}"
        assert result["config"]["Cmd"] == metadata._SERVICE_CMDS[service]
        assert "PORT=8080" in result["config"]["Env"]
    finally:
        shutil.rmtree(generated_root, ignore_errors=True)
