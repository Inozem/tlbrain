"""Offline runner compatibility through an ordinary Function deploy route.

These tests keep Bash and the production preflight dispatcher in the loop while
replacing the selected bootstrap interpreter and gcloud with local fixtures.
They exercise the Windows Bash fallback and route handoff without installing a
runtime, contacting a registry, or deploying to a cloud project.
"""

from __future__ import annotations

import os
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
RUNNER = APP_ROOT / "infra" / "build" / "bootstrap_runner.py"


def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash route execution is unavailable on native Windows")
    if os.name == "nt":
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _path_for_bash(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt" and len(value) >= 3 and value[1:3] == ":/":
        return "/" + value[0].lower() + value[2:]
    return value


def _copy_route(root: Path) -> Path:
    checkout = root / "synthetic checkout with spaces"
    shutil.copytree(APP_ROOT / "infra", checkout / "infra")
    return checkout


def _windows_git_root(bash_executable: Path) -> Path:
    bash_executable = bash_executable.resolve()
    if bash_executable.parent.name.lower() == "bin" and bash_executable.parent.parent.name.lower() == "usr":
        return bash_executable.parent.parent.parent
    return bash_executable.parent.parent


def _windows_git_path(bash_executable: Path, bin_dir: Path) -> str:
    """Expose only the selected Git for Windows tools to the synthetic route."""
    git_root = _windows_git_root(bash_executable)
    utility_dirs = (
        git_root / "bin",
        git_root / "usr" / "bin",
        git_root / "mingw64" / "bin",
    )
    path_parts = [str(bin_dir), *(str(path) for path in utility_dirs if path.is_dir())]
    return os.pathsep.join(dict.fromkeys(path_parts))


def _runner_module():
    spec = importlib.util.spec_from_file_location("runner_compat_bootstrap", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fakes(root: Path, tmp_path: Path, *, runner_status: int) -> tuple[Path, Path]:
    """Install a fake gcloud and a path-with-spaces interpreter executable."""
    bin_dir = tmp_path / "route tools"
    bin_dir.mkdir()
    interpreter = tmp_path / "Build Area" / "bootstrap python"
    interpreter.parent.mkdir()
    events = tmp_path / "events.log"

    interpreter.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'bootstrap\\t%s\\t%s\\n' \"$0\" \"$*\" >> \"$RUNNER_COMPAT_EVENTS\"\n"
        "if [[ \"$RUNNER_COMPAT_STATUS\" == 0 ]]; then printf 'Artifact fingerprint: %064d\\n' 0; fi\n"
        "exit \"$RUNNER_COMPAT_STATUS\"\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)

    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gcloud\\t%s\\n' \"$*\" >> \"$RUNNER_COMPAT_EVENTS\"\n"
        "case \"$*\" in\n"
        "  'info --format=value(basic.python_location)') printf '%s\\n' \"$RUNNER_COMPAT_INTERPRETER\" ;;\n"
        "  *'services describe'*) printf 'https://synthetic.example/service\\n' ;;\n"
        "  *'functions describe'*) printf 'https://synthetic.example/function\\n' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)

    # Keep PATH limited to fake commands, Bash, and only the real shell utilities
    # the route needs. This forces the gcloud fallback without exposing Python.
    bash_executable = Path(_bash()).resolve()
    if os.name == "nt":
        return events, Path(_windows_git_path(bash_executable, bin_dir))
    else:
        # A tiny isolated PATH lets the actual preflight script run under Bash
        # without inheriting the host's python3 candidate.
        shell_link = bin_dir / "bash"
        shell_link.symlink_to(bash_executable)
        for utility in ("dirname", "sed", "grep", "xargs", "echo"):
            executable = shutil.which(utility)
            if not executable:
                pytest.fail(f"required POSIX shell utility {utility!r} is unavailable")
            utility_link = bin_dir / utility
            utility_link.symlink_to(Path(executable).resolve())
        utility_dirs = []
    path_parts = [str(bin_dir), *(str(directory) for directory in utility_dirs)]
    return events, Path(os.pathsep.join(dict.fromkeys(path_parts)))


@pytest.mark.skipif(os.name != "nt", reason="Windows PATH isolation regression")
@pytest.mark.parametrize("launcher", ["bin", "usr/bin"], ids=["git-bin-bash", "git-usr-bin-bash"])
def test_windows_fake_path_excludes_host_tools_and_keeps_git_utilities(
    launcher: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated inherited PATH cannot expose host fallback tools."""
    selected_bash = Path(_bash()).resolve()
    git_root = next(
        (ancestor for ancestor in selected_bash.parents if (ancestor / "usr" / "bin" / "bash.exe").is_file()),
        None,
    )
    assert git_root is not None, "selected Bash is not inside a Git for Windows installation"
    bash_executable = git_root.joinpath(*launcher.split("/"), "bash.exe")
    assert bash_executable.is_file(), f"selected Git installation is missing {launcher}/bash.exe"

    hostile = tmp_path / "host tools"
    hostile.mkdir()
    for name in ("powershell.exe", "python.exe", "bash.exe"):
        (hostile / name).write_text("sentinel", encoding="utf-8")
    monkeypatch.setenv("PATH", str(hostile))
    monkeypatch.setattr(sys.modules[__name__], "_bash", lambda: str(bash_executable))

    _, isolated_path = _write_fakes(tmp_path, tmp_path, runner_status=0)
    path_value = str(isolated_path)
    normalized_parts = {os.path.normcase(os.path.normpath(part)) for part in path_value.split(os.pathsep)}

    assert os.path.normcase(os.path.normpath(str(hostile))) not in normalized_parts
    for tool in ("powershell.exe", "python.exe", "python3.exe"):
        assert shutil.which(tool, path=path_value) is None
    resolved_bash = Path(shutil.which("bash.exe", path=path_value) or "").resolve()
    assert resolved_bash != (hostile / "bash.exe").resolve()
    assert resolved_bash.is_relative_to(git_root.resolve())
    for utility in ("bash", "dirname", "sed", "grep", "xargs", "echo", "cygpath"):
        assert shutil.which(utility, path=path_value), f"Git utility {utility!r} is missing from isolated PATH"


@pytest.mark.parametrize("refresh_token", [False, True], ids=["fresh-token", "existing-token"])
@pytest.mark.parametrize("runner_status", [0, 17], ids=["preflight-passes", "preflight-fails"])
def test_sync_checker_route_uses_gcloud_bootstrap_fallback_before_cloud_send(
    tmp_path: Path, refresh_token: bool, runner_status: int,
) -> None:
    """The ordinary route preserves a spaced interpreter path and stops on gate failure."""
    bash = _bash()
    checkout = _copy_route(tmp_path)
    preflight = checkout / "infra" / "build" / "preflight.sh"
    if os.name != "nt":
        preflight.chmod(0o644)
        assert preflight.stat().st_mode & 0o111 == 0
    events, isolated_path = _write_fakes(checkout, tmp_path, runner_status=runner_status)
    interpreter = tmp_path / "Build Area" / "bootstrap python"
    environment = {
        **{
            name: os.environ[name]
            for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
            if name in os.environ
        },
        "PATH": str(isolated_path),
        "RUNNER_COMPAT_EVENTS": str(events),
        "RUNNER_COMPAT_INTERPRETER": _path_for_bash(interpreter),
        "RUNNER_COMPAT_STATUS": str(runner_status),
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "SYNC_CHECKER_NAME": "synthetic-sync-checker",
        "VECTOR_SYNC_SERVICE_NAME": "synthetic-vector-sync",
        "VECTOR_SYNC_QUEUE": "synthetic-vector-queue",
        "SYNC_CHECKER_SCHEDULE": "*/15 * * * *",
        "ROOT_FOLDER_URL": "https://synthetic.example/root",
        "GOOGLE_CLIENT_ID": "synthetic-client-id",
        "GOOGLE_CLIENT_SECRET": "synthetic-client-secret",
        "GOOGLE_REFRESH_TOKEN": "",
    }
    if refresh_token:
        (checkout / ".env").write_text(
            "GOOGLE_REFRESH_TOKEN=synthetic-existing-refresh-token\n", encoding="utf-8"
        )

    original_path = os.environ.get("PATH", "")
    completed = subprocess.run(
        [bash, str(checkout / "infra" / "deploy" / "core" / "deploy_sync_checker.sh")],
        cwd=checkout,
        env=environment,
        input="",
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert events.exists(), f"route exited {completed.returncode}; stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    log = events.read_text(encoding="utf-8").splitlines()
    cloud_calls = [line.partition("\t")[2] for line in log if line.startswith("gcloud\t")]
    bootstrap_entries = [line.split("\t", 2)[1:] for line in log if line.startswith("bootstrap\t")]

    assert any(call == "info --format=value(basic.python_location)" for call in cloud_calls)
    assert len(bootstrap_entries) == (1 if runner_status else 2)
    invoked_executable, bootstrap_args = bootstrap_entries[0]
    assert invoked_executable == _path_for_bash(interpreter)
    assert "Build Area" in invoked_executable
    assert _path_for_bash(checkout / "infra" / "build" / "bootstrap_runner.py") in bootstrap_args
    assert "--service sync_checker" in bootstrap_args
    assert _path_for_bash(checkout) in bootstrap_args
    if not runner_status:
        verify_executable, verify_args = bootstrap_entries[1]
        expected_fingerprint = "0" * 64
        expected_artifact = _path_for_bash(checkout / ".build" / "sync_checker")
        assert verify_executable == invoked_executable
        assert "--verify-source" in verify_args
        assert f"--artifact {expected_artifact}" in verify_args
        assert f"--fingerprint {expected_fingerprint}" in verify_args
    assert not any(
        call.lower().startswith("auth ") or "application-default" in call.lower()
        for call in cloud_calls
    )
    assert os.environ.get("PATH", "") == original_path
    assert "docker\t" not in "\n".join(log)

    function_deploys = [call for call in cloud_calls if call.startswith("functions deploy ")]
    uploads = [call for call in cloud_calls if "upload" in call.lower()]
    if runner_status:
        assert completed.returncode != 0
        assert not function_deploys
        assert not uploads
        assert not any(call.startswith("scheduler jobs ") for call in cloud_calls)
    else:
        assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        assert len(function_deploys) == 1
        assert "--runtime python312" in function_deploys[0]
        expected_refresh_token = (
            "synthetic-existing-refresh-token" if refresh_token else ""
        )
        assert f"GOOGLE_REFRESH_TOKEN={expected_refresh_token}" in function_deploys[0]
        assert not uploads


def test_runner_cold_warm_and_corrupt_cache_states_stay_in_private_cache(tmp_path: Path) -> None:
    """Exercise the ordinary runner API across all cache states with its synthetic archive."""
    helper_path = Path(__file__).with_name("test_runner_bootstrap.py")
    spec = importlib.util.spec_from_file_location("runner_compat_bootstrap_fixtures", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    runner = helper._runner()
    archive = helper._fixture_archive("valid-toolchain.tar.gz")
    lock = helper._write_lock(tmp_path)
    downloader = helper._Downloader(archive)
    cache = tmp_path / "private cache with spaces"

    cold = runner.ensure_runtime(
        lock, cache, helper._platform(), downloader=downloader,
        interpreter_probe=helper._fixture_probe,
    )
    warm = runner.ensure_runtime(
        lock, cache, helper._platform(), downloader=downloader,
        interpreter_probe=helper._fixture_probe,
    )
    assert Path(warm) == Path(cold)
    assert len(downloader.calls) == 1
    assert Path(cold).is_relative_to(cache)

    Path(cold).write_text("synthetic corrupt generation", encoding="utf-8")
    repaired = runner.ensure_runtime(
        lock, cache, helper._platform(), downloader=downloader,
        interpreter_probe=helper._fixture_probe,
    )
    assert Path(repaired) == Path(cold)
    assert Path(repaired).read_text(encoding="utf-8") != "synthetic corrupt generation"
    assert len(downloader.calls) == 2
    assert not list(cache.rglob("*.partial"))
    assert sorted(path.name for path in cache.iterdir()) == ["generations"]


def test_successful_host_preflight_report_keeps_host_facts_separate_from_linux_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A synthetic package/startup success reports the host actually running the runner."""
    runner = _runner_module()
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    artifact = root / ".build" / "sync_checker"
    runtime_calls: list[tuple[Path, Path]] = []

    def use_synthetic_runtime(lock_path: Path, cache_path: Path, **_kwargs: object) -> Path:
        runtime_calls.append((lock_path, cache_path))
        return tmp_path / "synthetic-python312"

    def run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        if "assemble.py" in str(argv):
            artifact.mkdir(parents=True)
            (artifact / "assembly-report.json").write_text(
                json.dumps({"runtime_target": "checker"}), encoding="utf-8"
            )
        elif "-m" in argv and "venv" in argv:
            environment = Path(argv[-1])
            venv_python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("synthetic venv launcher", encoding="utf-8")
        elif "--offline" in argv:
            result_path = Path(argv[-1])
            result_path.write_text(json.dumps({
                "status": "passed",
                "artifact_fingerprint": runner._artifact_fingerprint(artifact),
            }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "ensure_runtime", use_synthetic_runtime)
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=run))

    assert runner.run_preflight("sync_checker", root, artifact) == 0
    output = capsys.readouterr().out
    assert len(runtime_calls) == 1
    assert "Native OS: " + runner._native_os_name() in output
    assert "Native architecture: " + runner._native_architecture() in output
    assert "Artifact deployment target OS: linux" in output
    assert "Native OS: linux" not in output or runner._native_os_name() == "linux"
    assert "Artifact runtime target: checker" in output
