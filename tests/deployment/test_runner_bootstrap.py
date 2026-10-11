"""Offline contracts for the private CPython runner bootstrap.

The bootstrap implementation is intentionally imported from inside each test
so these contracts collect before the runner and lock file are added. The
small public helper contract exercised here is documented at the call sites;
all downloads and process discovery are faked and all archives are synthetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = APP_ROOT / "infra" / "build"
RUNNER = BUILD_DIR / "bootstrap_runner.py"
POWERSHELL_ADAPTER = BUILD_DIR / "bootstrap_runner.ps1"
LOCK = BUILD_DIR / "runner-toolchain.json"
FIXTURES = Path(__file__).parent / "fixtures" / "toolchain"


PINNED_TOOLCHAINS = {
    "windows-x86_64-none": {
        "os": "windows",
        "libc": "none",
        "target_triple": "x86_64-pc-windows-msvc",
        "executable": "python/python.exe",
        "minimum_key": "os_version_min",
        "minimum_version": "8.1",
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-x86_64-pc-windows-msvc-install_only.tar.gz",
        "sha256": "e90c1b6419da3bd812dd73bb3de40287a21abf153438147639ec5e20375ea93f",
    },
    "macos-x86_64-none": {
        "os": "macos",
        "libc": "none",
        "target_triple": "x86_64-apple-darwin",
        "executable": "python/bin/python3.12",
        "minimum_key": "os_version_min",
        "minimum_version": "10.15",
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-x86_64-apple-darwin-install_only.tar.gz",
        "sha256": "2e31b23f3f1319f707d0e620b48847a0046577541d357276821f9f1b5492e0ba",
    },
    "linux-x86_64-glibc": {
        "os": "linux",
        "libc": "glibc",
        "target_triple": "x86_64-unknown-linux-gnu",
        "executable": "python/bin/python3.12",
        "minimum_key": "libc_version_min",
        "minimum_version": "2.17",
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "sha256": "936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22",
    },
}


def _pinned_toolchain_lock() -> dict[str, Any]:
    """Read the checked-in public build-toolchain pins without loading a runner."""
    return json.loads(LOCK.read_text(encoding="utf-8"))


def test_pinned_toolchain_schema_covers_only_supported_baselines() -> None:
    lock = _pinned_toolchain_lock()
    assert lock["schema_version"] == 1
    assert lock["python_minor"] == "3.12"
    assert set(lock["entries"]) == set(PINNED_TOOLCHAINS)


@pytest.mark.parametrize("key", sorted(PINNED_TOOLCHAINS))
def test_pinned_toolchain_identity_and_reviewed_asset(key: str) -> None:
    entry = _pinned_toolchain_lock()["entries"][key]
    expected = PINNED_TOOLCHAINS[key]
    assert entry["os"] == expected["os"]
    assert entry["architecture"] == "x86_64"
    assert entry["libc"] == expected["libc"]
    assert entry["python_patch"] == "3.12.14"
    assert entry["release"] == "3.12.14+20260901"
    assert entry["target_triple"] == expected["target_triple"]
    assert entry["url"] == expected["url"]
    assert entry["url"].endswith(f"{entry['target_triple']}-install_only.tar.gz")
    assert entry["sha256"] == expected["sha256"]
    assert len(entry["sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in entry["sha256"])
    assert entry["archive_format"] == "tar.gz"
    assert entry["executable"] == expected["executable"]
    executable = Path(entry["executable"])
    assert not executable.is_absolute()
    assert ".." not in executable.parts


@pytest.mark.parametrize("key", sorted(PINNED_TOOLCHAINS))
def test_pinned_toolchain_host_and_platform_bounds(key: str) -> None:
    entry = _pinned_toolchain_lock()["entries"][key]
    expected = PINNED_TOOLCHAINS[key]
    assert entry["host_python_min"] == "3.10"
    assert entry["host_python_max"] == "3.14"
    assert entry[expected["minimum_key"]] == expected["minimum_version"]
    bound_keys = {"os_version_min", "libc_version_min"}
    assert {name for name in bound_keys if name in entry} == {expected["minimum_key"]}


def test_pinned_toolchain_excludes_mutable_or_incompatible_targets() -> None:
    lock_text = LOCK.read_text(encoding="utf-8").lower()
    assert "latest" not in lock_text
    assert "musl" not in lock_text
    assert "x86_64_v2" not in lock_text
    assert "x86_64_v3" not in lock_text
    assert all("aarch64" not in entry["target_triple"] for entry in _pinned_toolchain_lock()["entries"].values())


def _runner() -> Any:
    """Load the future runner by absolute path, without import-time collection."""
    if not RUNNER.is_file():
        raise AssertionError(f"runner implementation is missing: {RUNNER}")
    spec = importlib.util.spec_from_file_location("tlbrain_bootstrap_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_archive(name: str) -> Path:
    path = FIXTURES / name
    assert path.is_file(), f"synthetic fixture archive is missing: {path}"
    return path


def _lock_document(archive: Path | None = None) -> dict[str, object]:
    """Return synthetic lock data; release and digest are deliberately not pins."""
    archive = archive or _fixture_archive("valid-toolchain.tar.gz")
    return {
        "schema_version": 1,
        "python_minor": "3.12",
        "entries": {
            "windows-x86_64-none": {
                "os": "windows",
                "architecture": "x86_64",
                "libc": "none",
                "release": "3.12.99+fixture.1",
                "url": "https://example.invalid/fixture/cpython.tar.gz",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "archive_format": "tar.gz",
                "executable": "python/bin/python.exe",
                "host_python_min": "3.10",
                "host_python_max": "3.14",
            }
        },
    }


def _write_lock(tmp_path: Path, value: dict[str, object] | None = None) -> Path:
    path = tmp_path / "runner-toolchain.json"
    path.write_text(json.dumps(value or _lock_document()), encoding="utf-8")
    return path


class _Downloader:
    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.calls: list[str] = []

    def __call__(self, url: str, destination: Path) -> None:
        self.calls.append(url)
        shutil.copyfile(self.archive, destination)


def _platform() -> dict[str, str]:
    return {"os": "windows", "architecture": "x86_64", "libc": "none"}


def test_lock_schema_accepts_v1_and_selects_keyed_platform(tmp_path: Path) -> None:
    """Contract: load_lock(path), then select_entry(lock, os, arch, libc)."""
    runner = _runner()
    lock = runner.load_lock(_write_lock(tmp_path))
    entry = runner.select_entry(
        lock, os_name="windows", architecture="x86_64", libc="none"
    )
    assert lock["schema_version"] == 1
    assert lock["python_minor"] == "3.12"
    assert entry["release"].startswith("3.12.")
    assert entry["archive_format"] == "tar.gz"
    assert entry["url"].startswith("https://")
    assert len(entry["sha256"]) == 64


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lock: lock.update(schema_version=2),
        lambda lock: lock.update(python_minor="3.10"),
        lambda lock: lock["entries"]["windows-x86_64-none"].update(url="http://example.invalid/a.tar.gz"),
        lambda lock: lock["entries"]["windows-x86_64-none"].update(sha256="not-a-digest"),
        lambda lock: lock["entries"]["windows-x86_64-none"].update(archive_format="zip"),
        lambda lock: lock["entries"]["windows-x86_64-none"].update(executable="../python.exe"),
    ],
    ids=("schema-version", "runtime-minor", "https-required", "sha256", "archive-format", "relative-executable"),
)
def test_lock_schema_rejects_invalid_contract_fields(
    tmp_path: Path, mutate: Any
) -> None:
    runner = _runner()
    document = _lock_document()
    mutate(document)
    with pytest.raises((ValueError, KeyError), match="(?i)(lock|schema|python|https|sha|archive|path|entry)"):
        runner.load_lock(_write_lock(tmp_path, document))


def test_lock_schema_rejects_unavailable_os_architecture_or_libc(tmp_path: Path) -> None:
    runner = _runner()
    lock = runner.load_lock(_write_lock(tmp_path))
    with pytest.raises((ValueError, KeyError), match="(?i)(platform|entry|support|libc|architecture)"):
        runner.select_entry(
            lock, os_name="linux", architecture="aarch64", libc="musl"
        )


def test_lock_schema_enforces_host_python_compatibility_bounds(tmp_path: Path) -> None:
    runner = _runner()
    lock = runner.load_lock(_write_lock(tmp_path))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(host|python|version|compatib)"):
        runner.select_entry(
            lock,
            os_name="windows",
            architecture="x86_64",
            libc="none",
            host_python_version="3.9.18",
        )


def test_portable_runner_cold_then_warm_cache_uses_verified_archive(tmp_path: Path) -> None:
    """Contract: ensure_runtime(lock, cache, platform, downloader=) activates atomically."""
    runner = _runner()
    archive = _fixture_archive("valid-toolchain.tar.gz")
    lock_path = _write_lock(tmp_path)
    downloader = _Downloader(archive)
    cache = tmp_path / "private-cache"
    first = runner.ensure_runtime(
        lock_path, cache, _platform(), downloader=downloader
    )
    assert Path(first).is_file()
    assert downloader.calls == ["https://example.invalid/fixture/cpython.tar.gz"]
    assert not list(cache.rglob("*.partial"))
    second = runner.ensure_runtime(
        lock_path, cache, _platform(), downloader=downloader
    )
    assert Path(second) == Path(first)
    assert len(downloader.calls) == 1


def test_portable_runner_repairs_corrupt_cache_without_exposing_partial_activation(
    tmp_path: Path,
) -> None:
    runner = _runner()
    archive = _fixture_archive("valid-toolchain.tar.gz")
    lock_path = _write_lock(tmp_path)
    downloader = _Downloader(archive)
    cache = tmp_path / "private-cache"
    cache.mkdir()
    (cache / "active").mkdir()
    (cache / "active" / "python.exe").write_text("corrupt", encoding="utf-8")
    executable = runner.ensure_runtime(
        lock_path, cache, _platform(), downloader=downloader
    )
    assert Path(executable).is_file()
    assert len(downloader.calls) == 1
    assert not list(cache.glob("*.partial"))


def test_portable_runner_checksum_failure_precedes_extraction_and_publish(tmp_path: Path) -> None:
    runner = _runner()
    archive = _fixture_archive("corrupt-toolchain.tar.gz")
    document = _lock_document()
    lock_path = _write_lock(tmp_path, document)
    downloader = _Downloader(archive)
    extracted: list[bool] = []
    published: list[bool] = []
    with pytest.raises((ValueError, RuntimeError), match="(?i)(sha|checksum|digest|hash)"):
        runner.ensure_runtime(
            lock_path,
            tmp_path / "cache",
            _platform(),
            downloader=downloader,
            extractor=lambda *_args: extracted.append(True),
            publisher=lambda *_args: published.append(True),
        )
    assert extracted == []
    assert published == []


def test_portable_runner_downloader_failure_precedes_activation_and_cleans_partial_cache(
    tmp_path: Path,
) -> None:
    runner = _runner()
    cache = tmp_path / "private-cache"
    activated: list[bool] = []

    def fail_download(_url: str, _destination: Path) -> None:
        raise RuntimeError("synthetic downloader failure")

    with pytest.raises(RuntimeError, match="synthetic downloader failure"):
        runner.ensure_runtime(
            _write_lock(tmp_path),
            cache,
            _platform(),
            downloader=fail_download,
            publisher=lambda *_args: activated.append(True),
        )

    assert activated == []
    assert not cache.exists() or list(cache.rglob("*")) == []


@pytest.mark.parametrize("archive_name", ["traversal-toolchain.tar.gz", "link-escape-toolchain.tar.gz"])
def test_portable_runner_rejects_archive_path_escape_before_activation(
    tmp_path: Path, archive_name: str
) -> None:
    runner = _runner()
    archive = _fixture_archive(archive_name)
    document = _lock_document(archive)
    lock_path = _write_lock(tmp_path, document)
    activated: list[bool] = []
    with pytest.raises((ValueError, RuntimeError), match="(?i)(archive|path|link|escape|outside|unsafe)"):
        runner.ensure_runtime(
            lock_path,
            tmp_path / "cache",
            _platform(),
            downloader=_Downloader(archive),
            publisher=lambda *_args: activated.append(True),
        )
    assert activated == []


@pytest.mark.parametrize("host_minor", ["3.10", "3.14"])
def test_portable_runner_never_treats_host_python_as_service_runtime(
    tmp_path: Path, host_minor: str
) -> None:
    runner = _runner()
    lock_path = _write_lock(tmp_path)
    runtime = runner.ensure_runtime(
        lock_path,
        tmp_path / "cache",
        _platform(),
        host_python_version=f"{host_minor}.8",
        downloader=_Downloader(_fixture_archive("valid-toolchain.tar.gz")),
    )
    check = runner.check_runtime(runtime)
    assert check["python_minor"] == "3.12"
    assert check["host_python_minor"] == host_minor


def test_portable_runner_reuses_acceptable_native_cpython312(tmp_path: Path) -> None:
    runner = _runner()
    native = tmp_path / "native-python"
    native.write_text("synthetic native Python 3.12 probe", encoding="utf-8")
    downloader = _Downloader(_fixture_archive("valid-toolchain.tar.gz"))
    runtime = runner.ensure_runtime(
        _write_lock(tmp_path),
        tmp_path / "cache",
        _platform(),
        native_interpreter=native,
        interpreter_probe=lambda _path: {"version": "3.12.7", "ssl": True, "venv": True, "pip": True},
        downloader=downloader,
    )
    assert Path(runtime) == native
    assert downloader.calls == []


@pytest.mark.parametrize("failed_check", ["ssl", "venv", "pip"])
def test_portable_runner_requires_ssl_venv_and_pip(tmp_path: Path, failed_check: str) -> None:
    runner = _runner()
    native = tmp_path / "native-python"
    native.write_text("synthetic", encoding="utf-8")
    probe = {"version": "3.12.7", "ssl": True, "venv": True, "pip": True}
    probe[failed_check] = False
    with pytest.raises((ValueError, RuntimeError), match=f"(?i){failed_check}"):
        runner.check_interpreter(native, probe=lambda _path: probe)


def test_discovery_uses_existing_bootstrap_python_without_network_or_oauth(tmp_path: Path) -> None:
    """Contract: discover_bootstrap_interpreter(process_runner, env, platform)."""
    runner = _runner()
    calls: list[list[str]] = []
    refresh_token = tmp_path / ".env"
    refresh_token.write_text("GOOGLE_REFRESH_TOKEN=synthetic-token\n", encoding="utf-8")

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="C:/tools/python.exe\n", stderr="")

    found = runner.discover_bootstrap_interpreter(
        process_runner=run,
        env={"PATH": "synthetic"},
        platform="windows",
        candidates=["C:/tools/python.exe"],
        project_root=tmp_path,
    )
    assert found == "C:/tools/python.exe"
    assert calls == []


def test_discovery_fallback_uses_only_gcloud_python_location(tmp_path: Path) -> None:
    runner = _runner()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="/opt/google/python\n", stderr="")

    found = runner.discover_bootstrap_interpreter(
        process_runner=run,
        env={"PATH": "synthetic"},
        platform="linux",
        candidates=[],
        gcloud_available=True,
    )
    assert found == "/opt/google/python"
    assert calls == [["gcloud", "info", "--format=value(basic.python_location)"]]


def test_discovery_refresh_token_uses_gcloud_fallback_without_setup_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic project token must not trigger OAuth or alter the supplied environment."""
    runner = _runner()
    env = {
        "PATH": "synthetic-path-with-no-bootstrap-tools",
        "PYTHONPATH": "synthetic-pythonpath",
        "TLB_TEST_SENTINEL": "keep-this-value",
    }
    original_env = env.copy()
    process_env = {
        "PATH": "synthetic-process-path",
        "USERPROFILE": "C:/synthetic-user",
        "HOME": "C:/synthetic-home",
        "APPDATA": "C:/synthetic-appdata",
        "LOCALAPPDATA": "C:/synthetic-local-appdata",
    }
    for name, value in process_env.items():
        monkeypatch.setenv(name, value)
    original_process_env = process_env.copy()
    project_env = tmp_path / ".env"
    original_dotenv = "GOOGLE_REFRESH_TOKEN=fake-token-only\n"
    project_env.write_text(original_dotenv, encoding="utf-8")
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="/synthetic/gcloud/python\n", stderr="")

    found = runner.discover_bootstrap_interpreter(
        process_runner=run,
        env=env,
        platform="linux",
        candidates=[],
        gcloud_available=True,
        project_root=tmp_path,
    )

    assert found == "/synthetic/gcloud/python"
    assert calls == [["gcloud", "info", "--format=value(basic.python_location)"]]
    assert env == original_env
    assert env["PATH"] == "synthetic-path-with-no-bootstrap-tools"
    assert {name: os.environ[name] for name in original_process_env} == original_process_env
    assert project_env.read_text(encoding="utf-8") == original_dotenv


@pytest.mark.parametrize(
    ("platform", "command", "stdout"),
    [
        ("windows", "powershell.exe", "C:/tools/python.exe"),
        ("linux", "python3", "/usr/bin/python3"),
    ],
    ids=("powershell-bootstrap", "python3-bootstrap"),
)
def test_discovery_uses_supported_bootstrap_command_before_gcloud(
    platform: str, command: str, stdout: str
) -> None:
    runner = _runner()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    found = runner.discover_bootstrap_interpreter(
        process_runner=run,
        env={"PATH": "synthetic"},
        platform=platform,
        candidates=[command],
        gcloud_available=True,
    )
    assert found == stdout
    assert calls and all(call[0].lower().startswith(command.lower()) for call in calls)
    assert all(call[0] != "gcloud" for call in calls)


def test_discovery_does_not_require_docker_or_windows_tools_on_gcloud_route() -> None:
    runner = _runner()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        assert argv[0] == "gcloud"
        return SimpleNamespace(returncode=0, stdout="/gcloud/python", stderr="")

    runner.discover_bootstrap_interpreter(
        process_runner=run,
        env={"PATH": ""},
        platform="linux",
        candidates=[],
        gcloud_available=True,
    )
    flat = " ".join(" ".join(call) for call in calls).lower()
    assert "docker" not in flat
    assert "powershell" not in flat


@pytest.mark.parametrize(
    ("input_path", "expected"),
    [
        (r"C:\Build Area\Runner\python.exe", r"C:\Build Area\Runner\python.exe"),
        (r"\\server\share\Build Area\python.exe", r"\\server\share\Build Area\python.exe"),
    ],
    ids=("spaces", "unc-path"),
)
def test_windows_paths_preserve_native_and_unc_paths(input_path: str, expected: str) -> None:
    """Contract: normalize_windows_path is host-independent and filesystem-free."""
    runner = _runner()
    assert runner.normalize_windows_path(input_path) == expected


def test_windows_paths_convert_posix_style_cache_without_losing_spaces() -> None:
    runner = _runner()
    result = runner.normalize_windows_path("C:/Build Area/cache/python.exe")
    assert result == r"C:\Build Area\cache\python.exe"


def test_windows_paths_reject_root_escape_deterministically() -> None:
    runner = _runner()
    with pytest.raises((ValueError, RuntimeError), match="(?i)(outside|escape|relative|path)"):
        runner.normalize_windows_path(r"C:\cache\..\..\Windows\python.exe", root=r"C:\cache")


def test_powershell_adapter_forwards_paths_as_arguments_and_propagates_failure(
    tmp_path: Path,
) -> None:
    """Contract: adapter exposes an invocation helper; fake process boundary only."""
    runner = _runner()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=7, stdout="", stderr="actionable synthetic failure")

    with pytest.raises((RuntimeError, SystemExit), match="(?i)(synthetic failure|7|bootstrap)"):
        runner.invoke_powershell_adapter(
            POWERSHELL_ADAPTER,
            [r"C:\Build Area\cache", "--offline"],
            process_runner=run,
        )
    assert len(calls) == 1
    assert r"C:\Build Area\cache" in calls[0]

