"""Private, verified bootstrap for the pinned build CPython runtime.

This module intentionally uses only the Python standard library.  It keeps the
download, unpacking, and activation work inside the caller's cache directory.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import platform as host_platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PLATFORMS = {"windows", "macos", "linux"}
_ARCHITECTURES = {"x86_64"}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def normalize_windows_path(path: os.PathLike[str] | str, root: os.PathLike[str] | str | None = None) -> str:
    """Normalize a Windows path lexically, without consulting the host filesystem."""
    value = os.fspath(path)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid Windows path")
    normalized = ntpath.normpath(value.replace("/", "\\"))
    if root is not None:
        root_value = os.fspath(root)
        if not isinstance(root_value, str) or not root_value or "\x00" in root_value:
            raise ValueError("invalid Windows root path")
        normalized_root = ntpath.normpath(root_value.replace("/", "\\"))
        try:
            common = ntpath.commonpath((normalized, normalized_root))
        except ValueError as exc:
            raise ValueError(f"path is outside root: {value}") from exc
        if ntpath.normcase(common) != ntpath.normcase(normalized_root):
            raise ValueError(f"path escapes root: {value}")
    return normalized


def _platform_name(value: Mapping[str, str] | str | None) -> tuple[str, str]:
    """Return the host OS and architecture used for bootstrap tool discovery."""
    if value is None:
        facts = _current_platform()
        return facts["os"], facts["architecture"]
    if isinstance(value, str):
        facts = _current_platform()
        name = value.lower()
        if name.startswith("win"):
            name = "windows"
        elif name in {"darwin", "osx", "mac"}:
            name = "macos"
        elif name.startswith("linux"):
            name = "linux"
        return name, facts["architecture"]
    facts = _current_platform()
    name = str(value.get("os", facts["os"])).lower()
    if name.startswith("win"):
        name = "windows"
    elif name in {"darwin", "osx", "mac"}:
        name = "macos"
    elif name.startswith("linux"):
        name = "linux"
    architecture = str(value.get("architecture", facts["architecture"])).lower()
    return name, architecture


def discover_bootstrap_interpreter(
    process_runner: Callable[..., Any],
    env: Mapping[str, str],
    platform: Mapping[str, str] | str | None,
    candidates: list[os.PathLike[str] | str] | tuple[os.PathLike[str] | str, ...],
    gcloud_available: bool = False,
    project_root: os.PathLike[str] | str | None = None,
) -> str:
    """Find an already supplied bootstrap Python or ask gcloud for its Python.

    Candidate paths are treated as results of the caller's existing discovery
    process. Only the platform's approved bootstrap command is executed here.
    """
    # Do not use or modify env/project_root. They remain in the signature for
    # the caller's discovery boundary and for backwards-compatible call sites.
    del env, project_root
    os_name, _architecture = _platform_name(platform)
    approved_command = "powershell.exe" if os_name == "windows" else "python3" if os_name in {"linux", "macos"} else None

    def is_store_alias(value: str) -> bool:
        folded = value.replace("/", "\\").casefold()
        return "\\windowsapps\\" in folded or folded.endswith("\\windowsapps")

    for raw_candidate in candidates:
        candidate = os.fspath(raw_candidate)
        basename = ntpath.basename(candidate.replace("/", "\\")).lower()
        if is_store_alias(candidate):
            continue
        # A path supplied by the caller is already an available interpreter;
        # return it verbatim so native spelling and spaces are preserved.
        looks_like_path = ntpath.isabs(candidate) or "/" in candidate or "\\" in candidate
        if looks_like_path and basename in {"python.exe", "python3.exe", "python", "python3", "python3.12"}:
            return candidate
        if approved_command is None or basename != approved_command.lower():
            continue
        if os_name == "windows":
            # Resolve only an Application entry. The WindowsApps filter avoids
            # launching the Microsoft Store execution alias.
            command = (
                "$command = Get-Command -Name python.exe -CommandType Application "
                "-ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch "
                "'(?i)\\\\WindowsApps\\\\' } | Select-Object -First 1; "
                "if ($command) { [Console]::Out.WriteLine($command.Source) }"
            )
            argv = [candidate, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            argv = [candidate, "-I", "-c", "import sys; print(sys.executable)"]
        try:
            result = process_runner(argv, capture_output=True, text=True, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if getattr(result, "returncode", 1) == 0:
            found = str(getattr(result, "stdout", "")).strip()
            if found and not is_store_alias(found):
                return found

    if gcloud_available:
        try:
            result = process_runner(
                ["gcloud", "info", "--format=value(basic.python_location)"],
                capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"gcloud could not report its bootstrap Python location: {exc}") from exc
        if getattr(result, "returncode", 1) != 0:
            detail = str(getattr(result, "stderr", "")).strip()
            raise RuntimeError(f"gcloud bootstrap interpreter discovery failed ({result.returncode}): {detail}")
        found = str(getattr(result, "stdout", "")).strip()
        if found:
            if os_name == "windows" and is_store_alias(found):
                raise RuntimeError("gcloud reported a Microsoft Store Python alias; provide an installed Python executable")
            return found
        raise RuntimeError("gcloud returned an empty bootstrap Python location")
    raise RuntimeError("no bootstrap interpreter found; provide Python or make gcloud available")


def invoke_powershell_adapter(
    adapter_path: os.PathLike[str] | str,
    args: list[os.PathLike[str] | str] | tuple[os.PathLike[str] | str, ...],
    process_runner: Callable[..., Any],
) -> Any:
    """Invoke the PowerShell adapter using an argv vector and report failures."""
    argv = [
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-File", os.fspath(adapter_path), "--",
        *(os.fspath(argument) for argument in args),
    ]
    try:
        result = process_runner(argv, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"PowerShell bootstrap adapter could not start: {exc}") from exc
    if getattr(result, "returncode", 1) != 0:
        detail = str(getattr(result, "stderr", "")).strip() or str(getattr(result, "stdout", "")).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"PowerShell bootstrap adapter failed with exit code {result.returncode}{suffix}")
    return result


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _assert_under(path: Path, root: Path) -> None:
    absolute_path = Path(os.path.abspath(path))
    absolute_root = Path(os.path.abspath(root))
    try:
        common = Path(os.path.commonpath((str(absolute_path), str(absolute_root))))
    except ValueError as exc:
        raise RuntimeError(f"cache path is outside private cache: {path}") from exc
    if common != absolute_root:
        raise RuntimeError(f"cache path is outside private cache: {path}")


def _reject_reparse_point(path: Path, label: str) -> None:
    if _is_reparse_point(path):
        raise RuntimeError(f"refusing symlinked or reparse-point {label}: {path}")


def _version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?(?:[^\s]*)\s*", str(value))
    if not match:
        raise ValueError(f"invalid Python version: {value!r}")
    return int(match[1]), int(match[2]), int(match[3] or 0)


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", value) or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"unsafe {label} path: {value!r}")
    return path


def _validate_entry(key: str, entry: Mapping[str, Any]) -> None:
    required = {
        "os", "architecture", "libc", "release", "url", "sha256",
        "archive_format", "executable", "host_python_min", "host_python_max",
    }
    missing = required.difference(entry)
    if missing:
        raise ValueError(f"lock entry {key!r} missing fields: {', '.join(sorted(missing))}")
    os_name, arch, libc = entry["os"], entry["architecture"], entry["libc"]
    expected_key = f"{os_name}-{arch}-{libc}"
    if key != expected_key or os_name not in _PLATFORMS or arch not in _ARCHITECTURES:
        raise ValueError(f"lock entry has unsupported platform identity: {key!r}")
    if (os_name == "linux" and libc != "glibc") or (os_name != "linux" and libc != "none"):
        raise ValueError(f"lock entry has unsupported libc for platform: {key!r}")
    if not isinstance(entry["release"], str) or not entry["release"].startswith("3.12."):
        raise ValueError(f"lock entry has invalid Python release: {key!r}")
    url = entry["url"]
    parsed_url = urlsplit(url) if isinstance(url, str) else None
    if parsed_url is None or parsed_url.scheme.lower() != "https" or not parsed_url.hostname:
        raise ValueError(f"lock entry URL must use HTTPS: {key!r}")
    if not isinstance(entry["sha256"], str) or not _DIGEST.fullmatch(entry["sha256"]):
        raise ValueError(f"lock entry SHA-256 digest is invalid: {key!r}")
    if entry["archive_format"] != "tar.gz":
        raise ValueError(f"unsupported archive format in lock entry {key!r}")
    _safe_relative(entry["executable"], label="executable")
    low, high = _version(entry["host_python_min"]), _version(entry["host_python_max"])
    if low[:2] < (3, 10) or high[:2] > (3, 14) or low > high:
        raise ValueError(f"invalid host Python compatibility bounds in {key!r}")
    for bound in ("os_version_min", "libc_version_min"):
        if bound in entry:
            _version(entry[bound])
    if "os_version_min" in entry and "libc_version_min" in entry:
        raise ValueError(f"lock entry cannot define both OS and libc minimums: {key!r}")


def load_lock(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load and validate a schema v1 pinned toolchain lock."""
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read runner lock: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("runner lock schema_version must be 1")
    if lock.get("python_minor") != "3.12":
        raise ValueError("runner lock python_minor must be 3.12")
    entries = lock.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("runner lock entries must be a nonempty object")
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"runner lock entry {key!r} must be an object")
        _validate_entry(key, entry)
    return lock


def select_entry(
    lock: Mapping[str, Any], os_name: str, architecture: str, libc: str,
    host_python_version: str | None = None,
) -> dict[str, Any]:
    """Select a pinned entry, enforcing inclusive host bootstrap bounds."""
    key = f"{os_name}-{architecture}-{libc}"
    try:
        entry = lock["entries"][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported platform: {key}") from exc
    if host_python_version is None:
        host_python_version = sys.version.split()[0]
    current = _version(host_python_version)
    if current[:2] < _version(entry["host_python_min"])[:2] or current[:2] > _version(entry["host_python_max"])[:2]:
        raise RuntimeError(f"host Python {host_python_version} is outside supported compatibility bounds")
    return dict(entry)


def _current_platform() -> dict[str, str]:
    system = sys.platform
    if system.startswith("win"):
        name, libc = "windows", "none"
        os_version = host_platform.win32_ver()[0]
        libc_version = ""
    elif system == "darwin":
        name, libc = "macos", "none"
        os_version = host_platform.mac_ver()[0]
        libc_version = ""
    elif system.startswith("linux"):
        detected_libc, libc_version = host_platform.libc_ver()
        detected_libc = detected_libc.lower()
        name, libc = "linux", detected_libc if detected_libc else "unknown"
        os_version = ""
    else:
        name, libc = system, "none"
        os_version = libc_version = ""
    machine = host_platform.machine().lower()
    arch = "x86_64" if machine in {"amd64", "x86_64", "x64"} else machine
    return {
        "os": name, "architecture": arch, "libc": libc,
        "os_version": os_version, "libc_version": libc_version,
    }


def _probe_interpreter(path: Path) -> dict[str, Any]:
    code = (
        "import json, platform, sys\n"
        "libc_name, libc_version = platform.libc_ver()\n"
        "features = {}\n"
        "import ssl, venv\n"
        "try:\n import pip; features['pip'] = True\n"
        "except Exception:\n features['pip'] = False\n"
        "os_name = {'win32': 'windows', 'darwin': 'macos'}.get(sys.platform, 'linux' if sys.platform.startswith('linux') else sys.platform)\n"
        "os_version = platform.win32_ver()[0] if os_name == 'windows' else platform.mac_ver()[0] if os_name == 'macos' else ''\n"
        "print(json.dumps({'version': platform.python_version(), 'ssl': True, 'venv': True, 'pip': features['pip'], 'os': os_name, 'architecture': platform.machine().lower(), 'libc': (libc_name.lower() or 'unknown') if sys.platform.startswith('linux') else 'none', 'libc_version': libc_version, 'os_version': os_version}))"
    )
    result = subprocess.run(
        [str(path), "-I", "-c", code], capture_output=True, text=True,
        timeout=20, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"interpreter probe failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("interpreter probe returned invalid data") from exc


def check_interpreter(path: os.PathLike[str] | str, probe: Callable[[Path], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Check that an interpreter is CPython 3.12 with required stdlib features."""
    candidate = Path(path)
    if not candidate.is_file():
        raise RuntimeError(f"interpreter does not exist: {candidate}")
    facts = dict((probe or _probe_interpreter)(candidate))
    version = facts.get("version")
    if not isinstance(version, str) or _version(version)[:2] != (3, 12):
        raise RuntimeError(f"interpreter must be Python 3.12 (found {version!r})")
    for feature in ("ssl", "venv", "pip"):
        if facts.get(feature) is not True:
            raise RuntimeError(f"interpreter is missing required {feature} support")
    facts["python_minor"] = "3.12"
    return facts


def check_runtime(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Probe a runtime and report the runtime and bootstrap Python versions."""
    candidate = Path(os.path.abspath(path))
    marker_root = None
    marker = None
    for parent in candidate.parents:
        possible = parent / ".bootstrap-runtime.json"
        if possible.is_file():
            marker_root, marker = parent, possible
            break
    managed = any(parent.name == "generations" for parent in candidate.parents)
    if managed:
        if marker_root is None or marker is None:
            raise RuntimeError("managed runtime generation is missing its integrity marker")
        if marker_root.is_symlink():
            raise RuntimeError("managed runtime generation cannot be a symlink")
        try:
            details = json.loads(marker.read_text(encoding="utf-8"))
            if _tree_manifest(marker_root) == details.get("files"):
                facts = dict(details["probe"])
                if facts.get("python_minor") == "3.12" and all(facts.get(k) is True for k in ("ssl", "venv", "pip")):
                    facts["host_python_minor"] = details["host_python_minor"]
                    return facts
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("managed runtime generation failed integrity validation") from exc
        raise RuntimeError("managed runtime generation failed integrity validation")
    facts = check_interpreter(candidate)
    facts["host_python_minor"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    return facts


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "tlbrain-runner-bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    """Hash staged regular files and record directories/links without following them."""
    result: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root).as_posix()
        if relative == ".bootstrap-runtime.json":
            continue
        if item.is_symlink():
            result[relative] = f"link:{stat.S_IMODE(item.lstat().st_mode):04o}:" + os.readlink(item)
        elif item.is_dir():
            result[relative] = f"directory:{stat.S_IMODE(item.lstat().st_mode):04o}"
        elif item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[relative] = f"file:{stat.S_IMODE(item.lstat().st_mode):04o}:" + digest.hexdigest()
        else:
            raise ValueError(f"runtime tree contains a special file: {relative}")
    return result


def _archive_members(archive: Path) -> list[tarfile.TarInfo]:
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            return tar.getmembers()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError(f"invalid tar.gz archive: {exc}") from exc


def _member_path(name: str) -> PurePosixPath:
    return _safe_relative(name.rstrip("/"), label="archive member")


def _link_target(member: tarfile.TarInfo, path: PurePosixPath) -> PurePosixPath:
    raw = member.linkname
    if not raw or "\\" in raw or "\x00" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"unsafe archive link target: {raw!r}")
    target_parts = list(path.parent.parts) if member.issym() else []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not target_parts:
                raise ValueError(f"archive link escapes extraction root: {member.name!r}")
            target_parts.pop()
        else:
            target_parts.append(part)
    if not target_parts:
        raise ValueError(f"unsafe archive link target: {raw!r}")
    return PurePosixPath(*target_parts)


def _extract_safe(archive: Path, destination: Path) -> None:
    members = _archive_members(archive)
    paths: dict[str, tuple[tarfile.TarInfo, PurePosixPath]] = {}
    for member in members:
        path = _member_path(member.name)
        normalized = "/".join(path.parts).casefold()
        if normalized in paths:
            raise ValueError(f"duplicate or colliding archive member: {member.name!r}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported special archive entry: {member.name!r}")
        if member.issym() or member.islnk():
            _link_target(member, path)
        paths[normalized] = (member, path)
    # Reject non-directory ancestors and all traversal through links, regardless
    # of the order in which entries were stored in the tarball.
    for _member, path in paths.values():
        for depth in range(1, len(path.parts)):
            parent = PurePosixPath(*path.parts[:depth])
            found = paths.get("/".join(parent.parts).casefold())
            if found and not found[0].isdir():
                raise ValueError(f"archive path traverses a non-directory: {parent}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        by_name = {m.name.rstrip("/"): m for m in tar.getmembers()}
        # Directories first, then regular files, then links. This prevents a
        # symlink from becoming a parent for a later extraction operation.
        ordered = sorted(paths.values(), key=lambda item: (0 if item[0].isdir() else 1 if item[0].isfile() else 2, len(item[1].parts)))
        directories: list[tuple[Path, int]] = []
        for member, rel in ordered:
            target = destination.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(exist_ok=True)
                directories.append((target, member.mode & 0o777))
            elif member.isfile():
                source = tar.extractfile(by_name[member.name.rstrip("/")])
                if source is None:
                    raise ValueError(f"archive member has no data: {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(target, member.mode & 0o777)
            elif member.issym():
                target.symlink_to(member.linkname)
            else:
                hard_target = _link_target(member, rel)
                source = destination.joinpath(*hard_target.parts)
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"archive hard link target is missing or unsafe: {member.linkname!r}")
                os.link(source, target)
        for directory, mode in sorted(directories, key=lambda pair: len(pair[0].parts), reverse=True):
            os.chmod(directory, mode)


def _publish(staged: Path, generation: Path) -> None:
    """Atomically reveal a complete immutable generation on the same volume."""
    cache_root = generation.parent.parent
    _assert_under(staged, cache_root)
    _assert_under(generation, cache_root)
    _reject_reparse_point(cache_root, "cache root")
    _reject_reparse_point(generation.parent, "generations directory")
    generation.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_point(generation.parent, "generations directory")
    if generation.exists():
        raise FileExistsError(f"runtime generation already exists: {generation}")
    os.replace(staged, generation)


def _platform_facts(value: Mapping[str, str] | str | None) -> dict[str, str]:
    if value is None:
        return _current_platform()
    if isinstance(value, str):
        current = _current_platform()
        current["os"] = value
        return current
    return dict(value)


def _at_least(actual: Any, minimum: Any, label: str) -> bool:
    if not isinstance(actual, str) or not actual:
        raise RuntimeError(f"cannot establish required {label} minimum {minimum}")
    if _version(actual) < _version(minimum):
        return False
    return True


def _meets_minimums(facts: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    if "os_version_min" in entry:
        return _at_least(facts.get("os_version"), entry["os_version_min"], "OS version")
    if "libc_version_min" in entry:
        return _at_least(facts.get("libc_version"), entry["libc_version_min"], "libc version")
    return True


def _native_facts_match(facts: Mapping[str, Any], platform_value: Mapping[str, str]) -> bool:
    aliases = {"amd64": "x86_64", "x64": "x86_64"}
    for key in ("os", "architecture", "libc"):
        actual = facts.get(key)
        if actual is None:
            return False
        expected = platform_value[key]
        if key == "architecture":
            actual, expected = aliases.get(str(actual).lower(), str(actual).lower()), aliases.get(str(expected).lower(), str(expected).lower())
        if actual != expected:
            return False
    return True


def ensure_runtime(
    lock_path: os.PathLike[str] | str,
    cache_dir: os.PathLike[str] | str,
    platform: Mapping[str, str] | str | None = None,
    *,
    host_python_version: str | None = None,
    downloader: Callable[[str, Path], None] | None = None,
    extractor: Callable[[Path, Path], None] | None = None,
    publisher: Callable[..., None] | None = None,
    native_interpreter: os.PathLike[str] | str | None = None,
    interpreter_probe: Callable[[Path], Mapping[str, Any]] | None = None,
) -> Path:
    """Reuse a compatible native 3.12 or activate a verified private runtime."""
    lock = load_lock(lock_path)
    facts = _platform_facts(platform)
    entry = select_entry(lock, facts["os"], facts["architecture"], facts["libc"], host_python_version)
    if not _meets_minimums(facts, entry):
        bound_name = "os_version_min" if "os_version_min" in entry else "libc_version_min"
        raise RuntimeError(f"host does not meet {bound_name}={entry[bound_name]}")
    cache = Path(os.path.abspath(cache_dir))
    _reject_reparse_point(cache, "cache root")
    cache.mkdir(parents=True, exist_ok=True)
    _reject_reparse_point(cache, "cache root")
    _assert_under(cache, cache)
    if native_interpreter is not None:
        try:
            native_facts = check_interpreter(native_interpreter, interpreter_probe)
            if _native_facts_match(native_facts, facts) and _meets_minimums(native_facts, entry):
                return Path(native_interpreter)
        except (OSError, RuntimeError, ValueError, KeyError):
            pass

    platform_key = f"{facts['os']}-{facts['architecture']}-{facts['libc']}"
    generations = cache / "generations"
    generation = generations / f"{platform_key}-{entry['sha256']}"
    _assert_under(generations, cache)
    _assert_under(generation, cache)
    _reject_reparse_point(generations, "generations directory")
    if generations.exists() and not generations.is_dir():
        raise RuntimeError(f"generations path is not a directory: {generations}")
    _reject_reparse_point(generation, "runtime generation")
    executable_rel = _safe_relative(entry["executable"], label="executable")
    generation_executable = generation.joinpath(*executable_rel.parts)
    if not generation.is_symlink() and generation_executable.is_file():
        try:
            marker = json.loads((generation / ".bootstrap-runtime.json").read_text(encoding="utf-8"))
            existing = marker["probe"]
            complete = (
                marker.get("archive_sha256") == entry["sha256"]
                and marker.get("platform_key") == platform_key
                and marker.get("files") == _tree_manifest(generation)
            )
            if (complete and existing.get("python_minor") == "3.12"
                    and all(existing.get(k) is True for k in ("ssl", "venv", "pip"))
                    and _native_facts_match(existing, facts)
                    and _meets_minimums(existing, entry)):
                return generation_executable
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    if generation.exists():
        # This generation failed its integrity check. Keep it quarantined so a
        # failed replacement cannot destroy any other usable generation.
        quarantined = generation.with_name(generation.name + ".corrupt-" + uuid.uuid4().hex)
        _assert_under(quarantined, cache)
        _reject_reparse_point(generations, "generations directory")
        os.replace(generation, quarantined)

    work = Path(tempfile.mkdtemp(prefix="runner-", dir=cache))
    _assert_under(work, cache)
    archive = work / "download.tar.gz.partial"
    staged = work / "tree"
    _assert_under(archive, cache)
    _assert_under(staged, cache)
    try:
        (downloader or _download)(entry["url"], archive)
        digest = _sha256_file(archive)
        if digest != entry["sha256"]:
            raise ValueError("download SHA-256 checksum does not match lock")
        (extractor or _extract_safe)(archive, staged)
        executable = staged.joinpath(*executable_rel.parts)
        try:
            mode = executable.lstat().st_mode
        except OSError as exc:
            raise ValueError(f"required runtime executable is missing: {executable_rel}") from exc
        if not stat.S_ISREG(mode) or (facts["os"] != "windows" and not mode & 0o111):
            raise ValueError("required runtime executable is not a regular executable file")
        probed = check_interpreter(executable, interpreter_probe)
        if not _native_facts_match(probed, facts):
            raise RuntimeError("downloaded runtime platform does not match selected platform")
        if not _meets_minimums(probed, entry):
            bound_name = "os_version_min" if "os_version_min" in entry else "libc_version_min"
            raise RuntimeError(f"downloaded runtime does not meet {bound_name}={entry[bound_name]}")
        host_minor = _version(host_python_version or sys.version.split()[0])
        marker = staged / ".bootstrap-runtime.json"
        marker.write_text(json.dumps({
            "archive_sha256": entry["sha256"],
            "host_python_minor": f"{host_minor[0]}.{host_minor[1]}",
            "platform_key": platform_key,
            "probe": probed,
        }), encoding="utf-8")
        details = json.loads(marker.read_text(encoding="utf-8"))
        details["files"] = _tree_manifest(staged)
        marker.write_text(json.dumps(details), encoding="utf-8")
        (publisher or _publish)(staged, generation)
        result = generation.joinpath(*executable_rel.parts)
        if not result.is_file():
            raise RuntimeError("runtime activation did not publish the required executable")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)

