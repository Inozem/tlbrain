"""Synthetic offline contracts for the shared dependency environment cache."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
CACHE_MODULE = APP_ROOT / "infra" / "build" / "environment_cache.py"
RUNNER_PATH = APP_ROOT / "infra" / "build" / "bootstrap_runner.py"


def _module(path: Path = CACHE_MODULE) -> Any:
    spec = importlib.util.spec_from_file_location("tlbrain_environment_cache_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("tlbrain_cache_bootstrap", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcesses:
    def __init__(self, *, fail_check: bool = False, fail_install: bool = False, symlink_launcher: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []
        self.creations: list[Path] = []
        self.installs: list[list[str]] = []
        self.checks = 0
        self.fail_check = fail_check
        self.fail_install = fail_install
        self.symlink_launcher = symlink_launcher
        self.os_release: Any = ("Windows", "synthetic-release", "synthetic-build")

    def __call__(self, argv: list[str], **_kwargs: Any) -> Any:
        argv = [str(value) for value in argv]
        self.calls.append(argv)
        self.environments.append(_kwargs.get("env"))
        if "-c" in argv and "ensurepip.version()" in argv[-1]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "version": "3.14.1 (synthetic build)", "implementation": "cpython", "cache_tag": "cpython-314",
                "abiflags": "", "platform": "win-amd64", "sys_platform": "win32", "machine": "AMD64",
                "os_release": self.os_release, "libc": ["", ""], "ensurepip": "25.2", "pip": "25.2",
            }), stderr="")
        if "config" in argv:
            if "debug" in argv:
                return SimpleNamespace(returncode=0, stdout="env_var:\nenv:\nglobal:\nsite:\nuser:\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "venv" in argv:
            destination = Path(argv[-1])
            self.creations.append(destination)
            python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            if self.symlink_launcher and os.name != "nt":
                python.symlink_to(Path(argv[0]).resolve())
            else:
                python.write_bytes(b"synthetic cached interpreter")
            (destination / "lib" / "site-packages").mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0)
        if "install" in argv:
            self.installs.append(argv)
            return SimpleNamespace(returncode=7 if self.fail_install else 0)
        if "check" in argv:
            self.checks += 1
            return SimpleNamespace(returncode=1 if self.fail_check else 0)
        return SimpleNamespace(returncode=0)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "checkout"
    project.mkdir()
    requirements = project / "requirements.txt"
    requirements.write_text('-r "nested/runtime.txt" # runtime\n-c nested/constraints.txt # constraint\n', encoding="utf-8")
    (project / "nested").mkdir()
    (project / "nested" / "runtime.txt").write_text("sample-lib>=1.0; python_version >= '3.12'\n", encoding="utf-8")
    (project / "nested" / "constraints.txt").write_text("sample-lib<3.0\n", encoding="utf-8")
    source = project / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    runtime = tmp_path / "runtime-python.exe"
    runtime.write_bytes(b"synthetic base runtime")
    return project, requirements, runtime


def _prepare(module: Any, tmp_path: Path, processes: FakeProcesses, project: Path, requirements: Path, runtime: Path, *, env: dict[str, str] | None = None) -> tuple[Path | None, int]:
    with module.prepared_environment(
        runtime=runtime, project_root=project, requirements=requirements,
        fresh_parent=tmp_path / "fresh-work", process_runner=processes,
        tree_manifest=_runner()._tree_manifest, verify_manifest=_runner()._verified_runtime_manifest,
        env=env or {"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1"},
    ) as result:
        return result.python, result.status


def test_cold_and_warm_use_one_stable_environment_and_check_every_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    users = []
    for _ in range(2):
        python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
        assert status == 0 and python is not None
        users.append(python)
        (project / "app.py").write_text(f"value = {len(users)}\n", encoding="utf-8")
    assert users[0] == users[1]
    assert len(processes.creations) == 1
    assert len(processes.installs) == 1
    assert processes.checks == 2
    assert not list((tmp_path / "fresh-work").glob("tlbrain-dependencies-*"))


def test_nested_same_size_requirement_runtime_and_pip_config_changes_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    base_env = {"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1"}
    first, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env=base_env)
    constraint = project / "nested" / "constraints.txt"
    constraint.write_text("sample-lib<4.0\n", encoding="utf-8")
    second, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env=base_env)
    third, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env={**base_env, "PIP_INDEX_URL": "https://example.invalid/simple"})
    runtime.write_bytes(b"new selected runtime build")
    fourth, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env={**base_env, "PIP_INDEX_URL": "https://example.invalid/simple"})
    processes.os_release = ("Windows", "new-release", "new-build")
    fifth, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env={**base_env, "PIP_INDEX_URL": "https://example.invalid/simple"})
    assert len({first, second, third, fourth, fifth}) == 5
    assert len(processes.creations) == 5
    assert len(processes.installs) == 5


@pytest.mark.parametrize("input_kind", ["requirements", "runtime"])
def test_changed_cold_input_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_kind: str) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    original = processes.__call__

    def mutate_after_install(argv: list[str], **kwargs: Any) -> Any:
        result = original(argv, **kwargs)
        if "install" in argv:
            if input_kind == "requirements":
                (project / "nested" / "constraints.txt").write_text("sample-lib<5.0\n", encoding="utf-8")
            else:
                runtime.write_bytes(b"runtime changed during install")
        return result

    python, status = _prepare(module, tmp_path, mutate_after_install, project, requirements, runtime)  # type: ignore[arg-type]
    assert python is None and status == 1
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv launchers are symlinks")
def test_standard_posix_venv_launcher_symlink_to_selected_runtime_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses(symlink_launcher=True)
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert status == 0 and python is not None
    assert python.is_symlink()
    assert python.resolve() == runtime.resolve()


def test_config_values_are_digested_without_persisting_raw_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    global_config = project / "global-pip.conf"
    global_config.write_text("[global]\nindex-url = https://user:secret-marker@packages.invalid/simple\n", encoding="utf-8")
    processes = FakeProcesses()
    original = processes.__call__

    def with_secret_config(argv: list[str], **kwargs: Any) -> Any:
        if "config" in argv and "debug" in argv:
            return SimpleNamespace(
                returncode=0,
                stdout=f"env_var:\nenv:\nglobal:\n  {global_config}, exists: True\nsite:\nuser:\n",
                stderr="",
            )
        if "config" in argv:
            return SimpleNamespace(returncode=0, stdout="global.index-url='https://user:secret-marker@packages.invalid/simple'\n", stderr="")
        return original(argv, **kwargs)

    python, status = _prepare(
        module, tmp_path, with_secret_config, project, requirements, runtime,
        env={"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1"},  # type: ignore[arg-type]
    )
    assert status == 0 and python is not None
    cache_root = module._project_cache_root(project.resolve())
    all_cache_bytes = b"".join(path.read_bytes() for path in cache_root.rglob("*") if path.is_file())
    assert b"secret-marker" not in all_cache_bytes


@pytest.mark.parametrize("corruption", ["interpreter", "dependency", "marker"])
def test_corrupt_environment_or_marker_is_rebuilt_before_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str) -> None:
    module = _module()
    runner = _runner()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert status == 0 and python is not None
    if corruption == "interpreter":
        python.write_bytes(b"modified interpreter")
    elif corruption == "dependency":
        (python.parent.parent / "lib" / "site-packages" / "extra.py").write_text("unexpected = True\n", encoding="utf-8")
    else:
        marker = next((tmp_path / "temp").rglob("*.ready.json"))
        marker.write_text("{}", encoding="utf-8")
    repaired, repaired_status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert repaired_status == 0 and repaired is not None
    assert len(processes.creations) == 2
    assert processes.checks == 2


def test_new_source_backed_bytecode_is_normalized_but_original_bytecode_stays_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import py_compile

    module = _module()
    runner = _runner()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert status == 0 and python is not None
    environment = python.parents[1] if os.name == "nt" else python.parents[1]
    source = environment / "lib" / "site-packages" / "recorded.py"
    source.write_text("recorded = True\n", encoding="utf-8")
    marker = next((tmp_path / "temp").rglob("*.ready.json"))
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    ready_payload = {key: value for key, value in metadata.items() if key != "marker_sha256"}
    ready_payload["manifest"] = runner._tree_manifest(environment)
    module._atomic_ready(marker, ready_payload)
    generated = Path(py_compile.compile(str(source), doraise=True))
    _python, warm_status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert warm_status == 0
    assert len(processes.creations) == 1
    assert generated.exists() is False


def test_unsupported_dynamic_requirement_uses_fresh_semantics_each_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    requirements.write_text("https://example.invalid/package.whl\n", encoding="utf-8")
    processes = FakeProcesses()
    first, _ = _prepare(module, tmp_path, processes, project, requirements, runtime)
    second, _ = _prepare(module, tmp_path, processes, project, requirements, runtime)
    installs = [call for call in processes.calls if "install" in call]
    assert first != second
    assert len(processes.creations) == 2
    assert len(installs) == 2
    assert all(call[call.index("-r") + 1] == str(requirements.resolve()) for call in installs)
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_external_constraint_setting_uses_fresh_path_with_original_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    constraint = project / "external-constraints.txt"
    constraint.write_text("sample-lib<3.0\n", encoding="utf-8")
    processes = FakeProcesses()
    env = {"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1", "PIP_CONSTRAINT": str(constraint)}
    first, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env=env)
    second, _ = _prepare(module, tmp_path, processes, project, requirements, runtime, env=env)
    installs = [call for call in processes.calls if "install" in call]
    assert first != second
    assert len(processes.creations) == len(installs) == 2
    assert all(call[call.index("-r") + 1] == str(requirements.resolve()) for call in installs)
    assert all(item and item.get("PIP_CONSTRAINT") == str(constraint) for item in processes.environments)
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_interpolated_include_uses_fresh_semantics_instead_of_resolving_literal_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    requirements.write_text("-r ${RUNTIME_REQUIREMENTS}\n", encoding="utf-8")
    processes = FakeProcesses()
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    installs = [call for call in processes.calls if "install" in call]
    assert status == 0 and python is not None
    assert len(installs) == 1 and installs[0][installs[0].index("-r") + 1] == str(requirements.resolve())
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_non_utf8_unsupported_include_cleans_active_graph_before_revisit(tmp_path: Path) -> None:
    module = _module()
    project, requirements, _runtime = _inputs(tmp_path)
    unsupported = project / "nested" / "unsupported.txt"
    unsupported.write_bytes(b"package-name\n\xff\n")
    requirements.write_text(
        "-r nested/unsupported.txt\n-r nested/unsupported.txt\n",
        encoding="utf-8",
    )
    assert module._read_requirements(requirements, set(), set(), [], project) is False


@pytest.mark.parametrize("kind", ["missing", "cycle"])
def test_requirements_graph_is_guarded_before_any_cache_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    if kind == "missing":
        requirements.write_text("-r absent.txt\n", encoding="utf-8")
    else:
        requirements.write_text("-r nested/runtime.txt\n", encoding="utf-8")
        (project / "nested" / "runtime.txt").write_text("-r ../requirements.txt\n", encoding="utf-8")
    processes = FakeProcesses()
    with pytest.raises(ValueError, match="(?i)(no such file|cycle|requirements)"):
        _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert not processes.creations
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_pip_owns_layer_discovery_and_file_changes_miss_even_with_same_config_override(tmp_path: Path) -> None:
    module = _module()
    global_config = tmp_path / "global-pip.conf"
    site_config = tmp_path / "site-pip.conf"
    env_config = tmp_path / "selected-pip.conf"
    global_config.write_text("[global]\nindex-url = https://global.invalid/simple\n", encoding="utf-8")
    site_config.write_text("[global]\nindex-url = https://site.invalid/simple\n", encoding="utf-8")
    env_config.write_text("[global]\nindex-url = https://user:secret-marker@override.invalid/simple\n", encoding="utf-8")
    outputs = {
        "debug": (
            f"env_var:\n  PIP_CONFIG_FILE='{env_config}'\n  PIP_NO_INDEX='1'\n"
            f"env:\n  global: {env_config}, exists: True\n"
            f"global:\n  {global_config}, exists: True\n    global.index-url: https://global.invalid/simple\n"
            f"site:\n  {site_config}, exists: True\n    global.index-url: https://site.invalid/simple\n",
            "",
        ),
        "list": (
            f":env:.config-file='{env_config}'\n:env:.no-index='1'\n"
            "global.index-url='https://synthetic-secret.invalid/simple'\n",
            "",
        ),
    }

    def run(argv: list[str], **_kwargs: Any) -> Any:
        command = "debug" if "debug" in argv else "list"
        stdout, stderr = outputs[command]
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    env = {"PIP_CONFIG_FILE": str(env_config), "PIP_NO_INDEX": "1"}
    first_supported, first_digest = module._pip_configuration(env, Path("python"), run)
    global_config.write_text("[global]\nindex-url = https://changed.invalid/simple\n", encoding="utf-8")
    second_supported, second_digest = module._pip_configuration(env, Path("python"), run)
    env_config.write_text("[global]\nindex-url = https://changed-override.invalid/simple\n", encoding="utf-8")
    third_supported, third_digest = module._pip_configuration(env, Path("python"), run)
    assert first_supported and second_supported
    assert first_digest != second_digest
    assert third_supported and second_digest != third_digest
    assert "synthetic-secret" not in first_digest and "global.invalid" not in first_digest
    assert "secret-marker" not in first_digest


def test_devnull_pip_layout_without_user_section_is_cacheable() -> None:
    module = _module()
    null_device = os.devnull
    debug_output = (
        f"env_var:\n  PIP_CONFIG_FILE='{null_device}'\n  PIP_NO_INDEX='1'\n"
        f"env:\n  global: {null_device}, exists: True\n"
        "global:\n  C:\\ProgramData\\pip\\pip.ini, exists: False\n"
        "site:\n  C:\\synthetic\\site\\pip.ini, exists: False\n"
    )

    def run(argv: list[str], **_kwargs: Any) -> Any:
        if "debug" in argv:
            stdout = debug_output
        else:
            stdout = f":env:.config-file='{null_device}'\n:env:.no-index='1'\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    supported, digest = module._pip_configuration(
        {"PIP_CONFIG_FILE": null_device, "PIP_NO_INDEX": "1"}, Path("python"), run,
    )
    assert supported and len(digest) == 64


def test_pip_config_unknown_effective_setting_disables_cache(tmp_path: Path) -> None:
    module = _module()

    def run(argv: list[str], **_kwargs: Any) -> Any:
        if "debug" in argv:
            return SimpleNamespace(returncode=0, stdout="env_var:\nenv:\nglobal:\nsite:\nuser:\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="global.find-links='./wheels'\n", stderr="")

    supported, _digest = module._pip_configuration(
        {"PIP_CONFIG_FILE": os.devnull}, Path("python"), run,
    )
    assert not supported


def test_failed_pip_check_does_not_publish_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses(fail_check=True)
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert python is None and status == 1
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_failed_install_does_not_publish_readiness_or_run_consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses(fail_install=True)
    python, status = _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert python is None and status == 7
    assert processes.checks == 0
    assert not list((tmp_path / "temp").rglob("*.ready.json"))


def test_lock_is_kernel_released_and_bounded(tmp_path: Path) -> None:
    module = _module()
    lock = tmp_path / "cache" / "entry.lock"
    with module._key_lock(lock, tmp_path / "cache"):
        with pytest.raises(TimeoutError):
            with module._key_lock(lock, tmp_path / "cache", timeout=0.01):
                pass
    with module._key_lock(lock, tmp_path / "cache", timeout=0.1):
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO fixture")
def test_lock_fifo_is_rejected_without_blocking_or_touching_external_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as os_module

    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    files, cacheable, supported, pip_digest = module._install_inputs(
        requirements, {"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1"}, runtime, processes,
    )
    facts = module._runtime_identity(runtime, processes, {})
    key = module._cache_key(facts, pip_digest, files)
    root = module._project_cache_root(project.resolve())
    _environment, _ready, lock = module._entry_paths(root, key)
    lock.parent.mkdir(parents=True)
    os_module.mkfifo(lock)
    with pytest.raises(RuntimeError, match="regular file"):
        _prepare(module, tmp_path, processes, project, requirements, runtime)


def test_concurrent_callers_serialize_environment_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    processes = FakeProcesses()
    first_using = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def use(label: str) -> None:
        with module.prepared_environment(
            runtime=runtime, project_root=project, requirements=requirements,
            fresh_parent=tmp_path / "fresh-work", process_runner=processes,
            tree_manifest=_runner()._tree_manifest, verify_manifest=_runner()._verified_runtime_manifest,
            env={"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1"},
        ) as result:
            assert result.status == 0 and result.python is not None
            order.append(label)
            if label == "first":
                first_using.set()
                assert release_first.wait(2)

    first = threading.Thread(target=use, args=("first",))
    second = threading.Thread(target=use, args=("second",))
    first.start()
    assert first_using.wait(2)
    second.start()
    time.sleep(0.1)
    assert order == ["first"]
    release_first.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert order == ["first", "second"]
    assert len(processes.creations) == len(processes.installs) == 1
    assert processes.checks == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_cache_root_symlink_is_refused_without_touching_redirect_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    temp = tmp_path / "temp"
    temp.mkdir()
    redirected = tmp_path / "sentinel"
    redirected.mkdir()
    (redirected / "keep").write_text("safe", encoding="utf-8")
    prefix = "tlbrain-deps" if os.name == "nt" else f"tlbrain-deps-{os.getuid()}"
    (temp / prefix).symlink_to(redirected, target_is_directory=True)
    monkeypatch.setattr(module.tempfile, "tempdir", str(temp))
    project, requirements, runtime = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="redirected"):
        _prepare(module, tmp_path, FakeProcesses(), project, requirements, runtime)
    assert (redirected / "keep").read_text(encoding="utf-8") == "safe"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_requirements_symlink_is_rejected_before_metadata_or_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "temp"))
    project, requirements, runtime = _inputs(tmp_path)
    outside = tmp_path / "outside-requirements.txt"
    outside.write_text("sample-lib==1.0\n", encoding="utf-8")
    requirements.unlink()
    requirements.symlink_to(outside)
    processes = FakeProcesses()
    with pytest.raises(ValueError, match="redirected"):
        _prepare(module, tmp_path, processes, project, requirements, runtime)
    assert not processes.calls
    assert outside.read_text(encoding="utf-8") == "sample-lib==1.0\n"
