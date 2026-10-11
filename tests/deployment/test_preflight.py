"""Offline contract tests for the preparation-only preflight CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOLS = APP_ROOT / "infra" / "build"
PREFLIGHT = BUILD_TOOLS / "preflight.sh"
RUNNER = BUILD_TOOLS / "bootstrap_runner.py"
VALIDATOR = BUILD_TOOLS / "validate.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("tlbrain_preflight_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validator_module() -> Any:
    spec = importlib.util.spec_from_file_location("tlbrain_preflight_validator_reference", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkout with spaces"
    tools = root / "infra" / "build"
    tools.mkdir(parents=True)
    shutil.copy2(BUILD_TOOLS / "assemble.py", tools / "assemble.py")
    (tools / "service-manifests.json").write_text(json.dumps({
        "schema_version": 1,
        "services": {"mcp": {
            "entry_roots": ["main.py"],
            "runtime": {"kind": "asgi", "target": "main:app"},
            "path_remaps": {}, "module_aliases": {}, "requirements_roots": ["requirements.txt"],
            "extra_modules": [], "resources": [], "model_preparation": [],
        }},
    }), encoding="utf-8")
    (tools / "dependency-providers.json").write_text(json.dumps({
        "schema_version": 1, "providers": {}, "optional_imports": {},
    }), encoding="utf-8")
    (root / "requirements.txt").write_text("# empty synthetic dependency set\n", encoding="utf-8")
    (root / ".env.synthetic").write_text("synthetic-secret=keep-this-value\n", encoding="utf-8")
    (root / "main.py").write_text("class App:\n    pass\napp = App()\n", encoding="utf-8")
    return root, root / ".build" / "assembled package"


def _mock_boundaries(monkeypatch: pytest.MonkeyPatch, module: Any, *, fail_at: str | None = None) -> list[list[str]]:
    calls: list[list[str]] = []
    real_run = subprocess.run
    validator_reference = _validator_module()

    def fake_ensure(*_args: Any, **_kwargs: Any) -> Path:
        return Path(sys.executable)

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        argv = [str(item) for item in args]
        calls.append(argv)
        if any(item.endswith("infra\\build\\assemble.py") or item.endswith("infra/build/assemble.py") for item in argv):
            if fail_at == "assemble":
                return SimpleNamespace(returncode=23)
            return real_run(args, **kwargs)
        if "-m" in argv and "venv" in argv:
            if fail_at == "venv":
                return SimpleNamespace(returncode=24)
            venv_index = argv.index("venv") + 1
            executable = Path(argv[venv_index]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"synthetic verified runtime venv executable")
            return SimpleNamespace(returncode=0)
        if "-m" in argv and "pip" in argv:
            if fail_at == "pip-install" and "install" in argv:
                return SimpleNamespace(returncode=25)
            if fail_at == "pip-check" and "check" in argv:
                return SimpleNamespace(returncode=26)
            return SimpleNamespace(returncode=0)
        if any(str(item).endswith("validate.py") for item in argv):
            if fail_at == "validation":
                return SimpleNamespace(returncode=27)
            artifact = Path(argv[argv.index("--artifact") + 1])
            result = Path(argv[argv.index("--result") + 1])
            result.write_text(json.dumps({
                "status": "passed", "artifact_fingerprint": validator_reference._fingerprint(artifact),
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"unexpected child command: {argv}")

    monkeypatch.setattr(module, "ensure_runtime", fake_ensure)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


def test_cli_requires_explicit_paths_and_rejects_output_outside_build(tmp_path: Path) -> None:
    module = _module()
    root, _output = _checkout(tmp_path)
    with pytest.raises(ValueError, match="strict descendant"):
        module._validate_preflight_paths(root, tmp_path / "outside")
    with pytest.raises(ValueError, match="toolchain cache"):
        module._validate_preflight_paths(root, root / ".build" / "toolchain" / "artifact")
    with pytest.raises(ValueError, match="absolute"):
        module._validate_preflight_paths(Path("relative-root"), Path("relative-output"))


def test_preflight_assembles_once_uses_external_harness_and_gates_on_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    root, output = _checkout(tmp_path)
    calls = _mock_boundaries(monkeypatch, module)
    monkeypatch.chdir(tmp_path)
    env_key = "TLBRAIN_PREFLIGHT_SYNTHETIC_ENV"
    monkeypatch.setenv(env_key, "leave-environment-unchanged")
    environment_before = dict(os.environ)
    source_env_contents = (root / ".env.synthetic").read_bytes()

    assert module.run_preflight("mcp", root, output) == 0
    assert len([call for call in calls if any(item.endswith("assemble.py") for item in call)]) == 1
    assert ["install"] == [call[call.index("install")] for call in calls if "install" in call]
    pip_check = next(call for call in calls if "check" in call)
    validate = next(call for call in calls if any(item.endswith("validate.py") for item in call))
    venv_python = Path(validate[0])
    assert str(venv_python).startswith(str(output.parent.parent) + os.sep) is False
    assert "-I" in pip_check and "-I" in validate
    harness = Path(next(item for item in validate if item.endswith("validate.py")))
    result_path = Path(validate[validate.index("--result") + 1])
    assert harness.is_absolute() and not harness.is_relative_to(root) and not harness.is_relative_to(output)
    assert result_path.is_absolute() and not result_path.is_relative_to(root) and not result_path.is_relative_to(output)
    assert result_path.parent == harness.parent.parent
    assert Path(validate[validate.index("--artifact") + 1]) == Path(os.path.abspath(output))
    printed = capsys.readouterr().out
    assert "Native OS:" in printed
    assert f"Native architecture: {platform.machine()}" in printed
    assert "Artifact deployment target OS: linux" in printed
    assert "Artifact runtime target: main:app" in printed
    assert "Artifact fingerprint:" in printed
    assert (output / "assembly-report.json").is_file()
    assert not any(path.name.lower().startswith(".env") for path in output.rglob("*"))
    assert (root / ".env.synthetic").read_bytes() == source_env_contents
    assert dict(os.environ) == environment_before
    validator_reference = _validator_module()
    assert module._artifact_fingerprint(output) == validator_reference._fingerprint(output)


@pytest.mark.parametrize("fail_at,expected_code", [("assemble", 23), ("venv", 24), ("pip-install", 25), ("pip-check", 26), ("validation", 27)])
def test_preflight_stops_at_first_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str, expected_code: int
) -> None:
    module = _module()
    root, output = _checkout(tmp_path)
    calls = _mock_boundaries(monkeypatch, module, fail_at=fail_at)
    assert module.run_preflight("mcp", root, output) == expected_code
    if fail_at == "assemble":
        assert len(calls) == 1
        assert not output.exists()
    elif fail_at == "venv":
        assert not any("pip" in call for call in calls)
    elif fail_at == "pip-check":
        assert not any(any(item.endswith("validate.py") for item in call) for call in calls)
    elif fail_at == "pip-install":
        assert not any("check" in call for call in calls)
    else:
        assert any("check" in call for call in calls)


def test_preflight_rejects_fingerprint_change_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root, output = _checkout(tmp_path)
    calls = _mock_boundaries(monkeypatch, module)
    original_run = module.subprocess.run

    def mutating_run(args: list[str], **kwargs: Any) -> Any:
        result = original_run(args, **kwargs)
        if any(str(item).endswith("validate.py") for item in args):
            (output / "main.py").write_text("changed after validation\n", encoding="utf-8")
        return result

    monkeypatch.setattr(module.subprocess, "run", mutating_run)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        module.run_preflight("mcp", root, output)
    assert calls


def test_shell_entrypoint_exposes_requested_options() -> None:
    contents = PREFLIGHT.read_text(encoding="utf-8")
    assert "bootstrap_runner.py" in contents
    assert "powershell.exe" in contents
    assert "gcloud info --format='value(basic.python_location)'" in contents
    assert "WindowsApps" in contents
    assert "for candidate_name in python3; do" in contents
    assert "candidate_name in python3 python" not in contents
    assert '"${1:-}" == "--image-mode"' in contents
    assert '"$image_metadata" image-preflight' in contents


def test_image_preflight_runs_runtime_checks_before_finalizing(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    report = {
        "service": "mcp",
        "runtime_target": "services.mcp.app.main:app",
        "source_files": [{"runtime_path": "services/mcp/app/main.py", "source_path": "services/mcp/app/main.py", "sha256": "a" * 64}],
        "resources": [{"runtime_path": "models/config.json", "source_path": "models/config.json", "sha256": "0123456789abcdef" * 4}],
        "requirements_graph": {"roots": ["requirements.txt"], "files": [{"runtime_path": "requirements.txt", "sha256": "def"}]},
    }

    class FakeDocker:
        def __init__(self, fail_at: str | None = None) -> None:
            self.fail_at = fail_at
            self.config = {
                "image_id": "sha256:immutable-image", "os": "linux", "architecture": "amd64",
                "rootfs": {"diff_ids": ["sha256:layer"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"], "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.events: list[str] = []

        def inspect_image(self, _image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.config))

        def run(self, _image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
            assert _image == "sha256:immutable-image"
            if "-c" in command and "assembly-report.json" in command[-1]:
                self.events.append("report")
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            if any(item.endswith("validate.py") for item in command):
                self.events.append("runtime")
                if self.fail_at == "runtime":
                    return SimpleNamespace(returncode=12, stdout="", stderr="")
                result_path = next(host for host, destination, _readonly in mounts if destination == "/tlbrain-validation-result") / "result.json"
                result_path.write_text(json.dumps({"status": "passed", "service": "mcp", "target": report["runtime_target"]}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            self.events.append("cache")
            if self.fail_at == "cache":
                return SimpleNamespace(returncode=13, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def attach_label(self, _image: str, key: str, value: str) -> None:
            self.events.append("finalize")
            self.config["config"]["Labels"][key] = value

    workflow = {"repository": "owner/repo", "run_id": "42", "attempt": "1"}
    docker = FakeDocker()
    result = metadata.run_image_preflight(docker, "mcp:local", "mcp", workflow, build_tools=BUILD_TOOLS)
    assert docker.events == ["report", "runtime", "cache", "finalize"]
    evidence = json.loads(result["config"]["Labels"][metadata.VALIDATION_LABEL])
    assert evidence["checks"] == ["compile", "imports", "startup", "cache"]
    assert evidence["target"] == report["runtime_target"]
    assert evidence["workflow"] == workflow
    assert evidence["source_fingerprint"] == metadata._fingerprint_report_section({"source_files": report["source_files"], "resources": report["resources"]})
    assert evidence["requirements_fingerprint"] == metadata._fingerprint_report_section(report["requirements_graph"])


def test_image_preflight_revalidates_runtime_changed_by_label_attachment() -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    report = {
        "service": "mcp", "runtime_target": "main:app",
        "source_files": [{"runtime_path": "main.py", "source_path": "main.py", "sha256": "a" * 64}],
        "resources": [], "requirements_graph": {"files": []},
    }

    class ChangingRuntimeDocker:
        def __init__(self) -> None:
            self.config = {
                "image_id": "sha256:original", "os": "linux", "architecture": "amd64",
                "rootfs": {"diff_ids": ["sha256:layer-one"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"], "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.events: list[str] = []
            self.runtime_runs = 0
            self.attachments = 0
            self.pinned_images: list[str] = []

        def inspect_image(self, _image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.config))

        def run(self, image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
            self.pinned_images.append(image)
            if "assembly-report.json" in command[-1]:
                self.events.append("report")
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            if any(item.endswith("validate.py") for item in command):
                self.events.append("runtime")
                self.runtime_runs += 1
                result_path = next(host for host, destination, _readonly in mounts if destination == "/tlbrain-validation-result") / "result.json"
                result_path.write_text(json.dumps({"status": "passed", "service": "mcp", "target": "main:app"}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            self.events.append("cache")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def attach_label(self, _image: str, key: str, value: str) -> None:
            self.events.append("finalize")
            self.attachments += 1
            self.config["config"]["Labels"][key] = value
            if self.attachments == 1:
                self.config["rootfs"]["diff_ids"] = ["sha256:layer-two"]
                self.config["image_id"] = "sha256:changed-runtime"

    docker = ChangingRuntimeDocker()
    result = metadata.run_image_preflight(
        docker, "mcp:local", "mcp", {"repository": "o/r", "run_id": "1", "attempt": "1"}, build_tools=BUILD_TOOLS
    )
    label = result["config"]["Labels"][metadata.VALIDATION_LABEL]
    evidence = json.loads(label)
    assert docker.events == ["report", "runtime", "cache", "finalize", "report", "runtime", "cache", "finalize"]
    assert docker.runtime_runs == 2
    assert evidence["runtime_fingerprint"] == metadata.runtime_fingerprint(result)
    assert evidence["source_fingerprint"] == metadata._fingerprint_report_section({"source_files": report["source_files"], "resources": report["resources"]})
    assert evidence["requirements_fingerprint"] == metadata._fingerprint_report_section(report["requirements_graph"])
    assert "sha256:changed-runtime" in docker.pinned_images


@pytest.mark.parametrize("failure", ["runtime", "second_change"])
def test_image_preflight_blocks_failed_or_unstable_runtime_revalidation(failure: str) -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    report = {
        "service": "mcp", "runtime_target": "main:app",
        "source_files": [{"runtime_path": "main.py", "source_path": "main.py", "sha256": "a" * 64}],
        "resources": [], "requirements_graph": {"files": []},
    }

    class UnstableRuntimeDocker:
        def __init__(self) -> None:
            self.config = {
                "image_id": "sha256:original", "os": "linux", "architecture": "amd64",
                "rootfs": {"diff_ids": ["sha256:layer-one"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"], "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.runtime_runs = 0
            self.attachments = 0

        def inspect_image(self, _image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.config))

        def run(self, _image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
            if "assembly-report.json" in command[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            if any(item.endswith("validate.py") for item in command):
                self.runtime_runs += 1
                if failure == "runtime" and self.runtime_runs == 2:
                    return SimpleNamespace(returncode=7, stdout="", stderr="")
                result_path = next(host for host, destination, _readonly in mounts if destination == "/tlbrain-validation-result") / "result.json"
                result_path.write_text(json.dumps({"status": "passed", "service": "mcp", "target": "main:app"}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def attach_label(self, _image: str, key: str, value: str) -> None:
            self.attachments += 1
            self.config["config"]["Labels"][key] = value
            if self.attachments < 3:
                self.config["rootfs"]["diff_ids"] = [f"sha256:layer-{self.attachments + 1}"]
                self.config["image_id"] = f"sha256:changed-{self.attachments}"

    docker = UnstableRuntimeDocker()
    with pytest.raises(RuntimeError):
        metadata.run_image_preflight(
            docker, "mcp:local", "mcp", {"repository": "o/r", "run_id": "1", "attempt": "1"}, build_tools=BUILD_TOOLS
        )
    assert docker.attachments == (1 if failure == "runtime" else 2)
    assert docker.runtime_runs == 2


@pytest.mark.parametrize("fail_at", ["runtime", "cache"])
def test_image_preflight_fails_before_finalization(
    monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    report = {
        "service": "mcp", "runtime_target": "main:app",
        "source_files": [{"runtime_path": "main.py", "source_path": "main.py", "sha256": "a" * 64}],
        "resources": [],
        "requirements_graph": {"files": []},
    }

    class FakeDocker:
        def __init__(self) -> None:
            self.config = {
                "image_id": "sha256:immutable-image", "os": "linux", "architecture": "amd64", "rootfs": {"diff_ids": ["sha256:layer"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"], "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.events: list[str] = []

        def inspect_image(self, _image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.config))

        def run(self, _image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
            assert _image == "sha256:immutable-image"
            if "-c" in command and "assembly-report.json" in command[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            stage = "runtime" if any(item.endswith("validate.py") for item in command) else "cache"
            self.events.append(stage)
            if stage == fail_at:
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            if stage == "runtime":
                result = next(host for host, destination, _readonly in mounts if destination == "/tlbrain-validation-result") / "result.json"
                result.write_text(json.dumps({"status": "passed", "service": "mcp", "target": report["runtime_target"]}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def attach_label(self, *_args: Any) -> None:
            self.events.append("finalize")

    docker = FakeDocker()
    with pytest.raises(RuntimeError):
        metadata.run_image_preflight(docker, "mcp:local", "mcp", {"repository": "o/r", "run_id": "1", "attempt": "1"}, build_tools=BUILD_TOOLS)
    assert "finalize" not in docker.events


def test_image_preflight_pins_checks_to_image_id_and_rejects_tag_movement() -> None:
    metadata = importlib.import_module("infra.build.image_metadata")
    report = {
        "service": "mcp", "runtime_target": "main:app",
        "source_files": [{"runtime_path": "main.py", "source_path": "main.py", "sha256": "a" * 64}],
        "resources": [], "requirements_graph": {"files": []},
    }

    class MovingTagDocker:
        def __init__(self) -> None:
            self.image = {
                "image_id": "sha256:checked-image", "os": "linux", "architecture": "amd64",
                "rootfs": {"diff_ids": ["sha256:checked-layer"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"], "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.moved = False
            self.run_images: list[str] = []
            self.attached = False

        def inspect_image(self, image: str) -> dict[str, Any]:
            if image == "mcp:local" and self.moved:
                changed = json.loads(json.dumps(self.image))
                changed["image_id"] = "sha256:different-image"
                changed["rootfs"]["diff_ids"] = ["sha256:different-layer"]
                return changed
            return json.loads(json.dumps(self.image))

        def run(self, image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> Any:
            self.run_images.append(image)
            assert image == "sha256:checked-image"
            if "assembly-report.json" in command[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            if any(item.endswith("validate.py") for item in command):
                result_path = next(host for host, destination, _readonly in mounts if destination == "/tlbrain-validation-result") / "result.json"
                result_path.write_text(json.dumps({"status": "passed", "service": "mcp", "target": report["runtime_target"]}), encoding="utf-8")
                self.moved = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def attach_label(self, *_args: Any) -> None:
            self.attached = True

    docker = MovingTagDocker()
    with pytest.raises(RuntimeError, match="tag moved"):
        metadata.run_image_preflight(docker, "mcp:local", "mcp", {"repository": "o/r", "run_id": "1", "attempt": "1"}, build_tools=BUILD_TOOLS)
    assert docker.run_images == ["sha256:checked-image", "sha256:checked-image", "sha256:checked-image"]
    assert not docker.attached


@pytest.mark.parametrize("bad_runtime", ["command", "port"])
def test_image_preflight_rejects_service_runtime_mismatch_before_checks(bad_runtime: str) -> None:
    metadata = importlib.import_module("infra.build.image_metadata")

    class BadRuntimeDocker:
        def __init__(self) -> None:
            self.image = {
                "image_id": "sha256:checked-image", "os": "linux", "architecture": "amd64",
                "rootfs": {"diff_ids": ["sha256:checked-layer"]},
                "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS["mcp"].copy(), "WorkingDir": "/app", "User": None, "Env": ["PORT=8080"], "Labels": {}},
            }
            self.runs = 0

        def inspect_image(self, _image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.image))

        def run(self, *_args: Any) -> Any:
            self.runs += 1
            raise AssertionError("runtime checks must not run for an invalid image config")

    docker = BadRuntimeDocker()
    if bad_runtime == "command":
        docker.image["config"]["Cmd"] = ["python", "app.py"]
    else:
        docker.image["config"]["Env"] = ["PORT=9090"]
    with pytest.raises(ValueError, match="image (command|runtime)"):
        metadata.run_image_preflight(docker, "mcp:local", "mcp", {"repository": "o/r", "run_id": "1", "attempt": "1"}, build_tools=BUILD_TOOLS)
    assert docker.runs == 0
