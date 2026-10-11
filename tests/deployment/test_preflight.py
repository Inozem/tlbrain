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
