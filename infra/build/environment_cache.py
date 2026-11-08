"""Private, verified cache for runtime-only dependency environments.

The cache owns eligibility, identity, locking, integrity checks and preparation.
Virtual environments are always created at their final absolute cache path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


_POLICY = "runtime-dependency-env-v1"
_READY_SCHEMA = 1
_LOCK_TIMEOUT = 30.0
_INCLUDE = re.compile(r"^(?P<flag>-r|--requirement|-c|--constraint)(?:\s+|=)(?P<target>.+?)\s*$")
_PACKAGE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?"
    r"(?:\s*;\s*[A-Za-z0-9_.'\"()!<>=\s-]+)?$"
)
_PIP_ENV_ALLOWED = {
    "PIP_CONFIG_FILE", "PIP_NO_INDEX", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
    "PIP_TRUSTED_HOST", "PIP_DISABLE_PIP_VERSION_CHECK", "PIP_TIMEOUT", "PIP_RETRIES",
}
_PIP_CONFIG_KEYS = {
    "no-index", "index-url", "extra-index-url", "trusted-host",
    "disable-pip-version-check", "timeout", "retries",
}
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class Preparation:
    """Result of one preparation attempt, exposed while the key lock is held."""

    def __init__(self, python: Path | None, status: int) -> None:
        self.python = python
        self.status = status


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_redirect(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE)


def _ensure_safe_directory(path: Path, boundary: Path) -> None:
    boundary = Path(os.path.abspath(boundary))
    path = Path(os.path.abspath(path))
    if os.path.commonpath((str(path), str(boundary))) != str(boundary):
        raise RuntimeError("dependency cache path escaped its private root")
    cursor = boundary
    if cursor.exists() and _is_redirect(cursor):
        raise RuntimeError("dependency cache root is redirected")
    if not cursor.exists():
        try:
            cursor.mkdir(parents=True, mode=0o700 if os.name != "nt" else 0o777)
        except FileExistsError:
            pass
    if _is_redirect(cursor) or not cursor.is_dir():
        raise RuntimeError("dependency cache root is redirected or not a directory")
    _check_private_owner(cursor)
    relative = path.relative_to(boundary)
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and _is_redirect(cursor):
            raise RuntimeError("dependency cache directory is redirected")
        if not cursor.exists():
            try:
                cursor.mkdir(mode=0o700 if os.name != "nt" else 0o777)
            except FileExistsError:
                pass
        if _is_redirect(cursor) or not cursor.is_dir():
            raise RuntimeError("dependency cache directory is redirected or not a directory")
        _check_private_owner(cursor)


def _check_private_owner(path: Path) -> None:
    if os.name == "nt":
        return
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise RuntimeError("dependency cache directory is not private to the current user")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_requirements(
    path: Path, active: set[Path], seen: set[Path], files: list[dict[str, str]], graph_root: Path | None = None,
) -> bool:
    """Hash supported static requirements and local recursive -r/-c inputs."""
    absolute = Path(os.path.abspath(path))
    cursor = absolute
    for parent in reversed(absolute.parents):
        if _is_redirect(parent):
            raise ValueError("requirements path is redirected")
    if _is_redirect(cursor):
        raise ValueError("requirements path is redirected")
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("requirements include is missing") from exc
    if graph_root is None:
        graph_root = resolved.parent
    if resolved in active:
        raise ValueError("requirements include cycle")
    if not resolved.is_file():
        raise ValueError("requirements include is not a regular file")
    if resolved in seen:
        return True
    active.add(resolved)
    seen.add(resolved)
    try:
        data = resolved.read_bytes()
        files.append({"path": Path(os.path.relpath(resolved, graph_root)).as_posix(), "sha256": _digest(data)})
        try:
            lines = data.decode("utf-8-sig").splitlines()
        except UnicodeDecodeError:
            return False
        supported = True
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            is_include, target_text, valid_include = _parse_include(line)
            if is_include:
                if not valid_include or target_text is None:
                    supported = False
                    continue
                if "${" in target_text or "%(" in target_text:
                    supported = False
                    continue
                if not target_text or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", target_text):
                    supported = False
                    continue
                target = resolved.parent / target_text
                if _is_redirect(target):
                    raise ValueError("requirements include is redirected")
                if not _read_requirements(target, active, seen, files, graph_root):
                    supported = False
                continue
            if line.startswith(("-", "@")) or "${" in line or "%(" in line or not _PACKAGE.fullmatch(line):
                supported = False
        return supported
    finally:
        active.discard(resolved)


def _parse_include(line: str) -> tuple[bool, str | None, bool]:
    match = _INCLUDE.fullmatch(line)
    if match is None:
        return False, None, False
    target = match.group("target").strip()
    if not target:
        return True, None, False
    quote: str | None = target[0] if target[0] in {"'", '"'} else None
    comment = None
    for index, character in enumerate(target[1:] if quote else target, start=1 if quote else 0):
        if quote:
            if character == quote and (index == 0 or target[index - 1] != "\\"):
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and index > 0 and target[index - 1].isspace():
            comment = index
            break
    if quote is not None:
        return True, None, False
    if comment is not None:
        target = target[:comment].rstrip()
    if target[:1] in {"'", '"'}:
        if len(target) < 2 or target[-1] != target[0]:
            return True, None, False
        target = target[1:-1]
    elif any(character.isspace() for character in target):
        return True, None, False
    if not target or "\x00" in target or "\\" in target:
        return True, None, False
    return True, target, True


def _pip_configuration(
    env: Mapping[str, str], runtime: Path, process_runner: Callable[..., Any],
) -> tuple[bool, str]:
    """Use pip's own public config metadata to digest all loaded sources safely."""
    if any(name.startswith("PIP_") and name not in _PIP_ENV_ALLOWED for name in env):
        return False, "unsupported-pip-environment"
    safe_env = dict(env)
    safe_env.pop("TELEGRAM_BOT_TOKEN", None)
    safe_env.pop("TELEGRAM_ALLOWED_USER_IDS", None)
    try:
        debug = process_runner(
            [str(runtime), "-I", "-X", "utf8", "-m", "pip", "config", "debug"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, env=safe_env, timeout=20,
        )
        listed = process_runner(
            [str(runtime), "-I", "-X", "utf8", "-m", "pip", "config", "list"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, env=safe_env, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "pip-config-probe-failed"
    if debug.returncode or listed.returncode:
        return False, "pip-config-probe-failed"
    try:
        config_keys = _pip_config_keys(listed.stdout)
        if config_keys is None:
            return False, "unsupported-pip-config-format"
        for key in config_keys:
            if not _pip_config_key_supported(key):
                return False, "unsupported-pip-config-setting"
        paths = _config_paths(debug.stdout)
        if paths is None:
            return False, "unsupported-pip-config-debug-format"
        files: list[tuple[str, str]] = []
        for name, exists in paths:
            path = Path(name)
            if _is_null_device(name):
                files.append((os.path.normcase(str(path)), "null-device"))
                continue
            if exists:
                if _is_redirect(path) or not path.is_file():
                    return False, "unsupported-pip-config-file"
                files.append((str(path.resolve()), _hash_file(path)))
            else:
                files.append((name, "missing"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, "unsupported-pip-config-format"
    env_inputs = [(key, env[key]) for key in sorted(_PIP_ENV_ALLOWED) if key in env]
    canonical = {
        "debug": _digest(debug.stdout.encode("utf-8", "surrogatepass")),
        "list": _digest(listed.stdout.encode("utf-8", "surrogatepass")),
        "files": files,
        "env": [(name, _digest(value.encode("utf-8", "surrogatepass"))) for name, value in env_inputs],
    }
    return True, _digest(json.dumps(canonical, sort_keys=True).encode())


def _pip_config_keys(output: str) -> list[str] | None:
    """Parse public ``pip config list`` key names while keeping values opaque."""
    keys = []
    for line in output.splitlines():
        if not line.strip():
            continue
        key, separator, _value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_.:-]+", key.strip()):
            return None
        keys.append(key.strip())
    return keys


def _pip_config_key_supported(key: str) -> bool:
    if key.startswith(":env:."):
        variable = "PIP_" + key[len(":env:."):].replace("-", "_").upper()
        return variable in _PIP_ENV_ALLOWED
    normalized = key.split(".")[-1].replace("_", "-").lower()
    return normalized in _PIP_CONFIG_KEYS


def _is_null_device(name: str) -> bool:
    normalized = os.path.normcase(os.path.abspath(name))
    null_device = os.path.normcase(os.path.abspath(os.devnull))
    if normalized == null_device:
        return True
    return os.name == "nt" and Path(name).name.casefold() == "nul"


def _config_paths(debug_output: str) -> list[tuple[str, bool]] | None:
    """Parse pip-reported source paths while leaving setting values opaque."""
    section = ""
    seen_sections: set[str] = set()
    paths: list[tuple[str, bool]] = []
    for line in debug_output.splitlines():
        if line and not line[0].isspace() and line.endswith(":"):
            section = line[:-1]
            if section not in {"env_var", "env", "global", "site", "user"}:
                return None
            seen_sections.add(section)
            continue
        if not line.strip():
            continue
        if section not in {"env", "global", "site", "user"}:
            continue
        if len(line) >= 4 and line.startswith("    "):
            continue
        if section == "env":
            match = re.fullmatch(r"  (?:global|site|user): (.+), exists: (True|False)", line)
            if match is None:
                match = re.fullmatch(r"  (.+), exists: (True|False)", line)
        else:
            match = re.fullmatch(r"  (.+), exists: (True|False)", line)
        if match is None:
            return None
        paths.append((match.group(1), match.group(2) == "True"))
    required_sections = {"env_var", "env", "global", "site"}
    return paths if required_sections.issubset(seen_sections) and seen_sections.issubset(
        {"env_var", "env", "global", "site", "user"},
    ) else None


def _runtime_identity(runtime: Path, process_runner: Callable[..., Any], env: Mapping[str, str]) -> dict[str, Any]:
    code = (
        "import ensurepip,json,os,pip,platform,sys,sysconfig\n"
        "if sys.platform == 'win32':\n"
        " w=sys.getwindowsversion(); os_release=(w.major,w.minor,w.build,w.platform,w.service_pack,w.product_type,w.service_pack_major,w.service_pack_minor,w.suite_mask)\n"
        "else:\n"
        " u=os.uname(); os_release=(u.sysname,u.release,u.version,u.machine)\n"
        "print(json.dumps({'version':sys.version,'implementation':sys.implementation.name,"
        "'cache_tag':sys.implementation.cache_tag,'abiflags':getattr(sys,'abiflags',''),"
        "'platform':sysconfig.get_platform(),'sys_platform':sys.platform,'machine':sysconfig.get_platform(),"
        "'os_release':os_release,"
        "'libc':platform.libc_ver() if sys.platform.startswith('linux') else ('',''),"
        "'ensurepip':ensurepip.version(),'pip':pip.__version__}))"
    )
    safe_env = dict(env)
    safe_env.pop("TELEGRAM_BOT_TOKEN", None)
    safe_env.pop("TELEGRAM_ALLOWED_USER_IDS", None)
    try:
        before_hash = _hash_file(runtime.resolve(strict=True))
        facts = process_runner(
            [str(runtime), "-I", "-c", code], capture_output=True, text=True,
            check=False, env=safe_env, timeout=20,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("selected Python identity probe timed out") from None
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("could not identify selected Python runtime") from None
    if facts.returncode:
        raise RuntimeError("could not identify selected Python runtime")
    try:
        runtime_facts = json.loads(facts.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("selected Python returned invalid runtime identity") from exc
    resolved = runtime.resolve(strict=True)
    if _hash_file(resolved) != before_hash:
        raise RuntimeError("selected Python changed during identity probe")
    runtime_facts["executable"] = str(resolved)
    runtime_facts["executable_sha256"] = before_hash
    return runtime_facts


@contextlib.contextmanager
def _key_lock(path: Path, boundary: Path, timeout: float = _LOCK_TIMEOUT) -> Iterator[None]:
    _ensure_safe_directory(path.parent, boundary)
    if _is_redirect(path):
        raise RuntimeError("dependency cache lock is redirected")
    if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError("dependency cache lock is not a regular file")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode) or _is_redirect(path):
        os.close(descriptor)
        raise RuntimeError("dependency cache lock is redirected or not a regular file")
    stream = os.fdopen(descriptor, "a+b")
    acquired = False
    started = time.monotonic()
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            while not acquired:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError("timed out waiting for dependency cache lock")
                    time.sleep(0.05)
        else:
            import fcntl
            while not acquired:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError("timed out waiting for dependency cache lock")
                    time.sleep(0.05)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _atomic_ready(path: Path, content: dict[str, Any]) -> None:
    payload = dict(content)
    payload["marker_sha256"] = _digest(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine(environment: Path, ready: Path, cache_root: Path) -> None:
    for item in (environment, ready):
        if not item.exists() and not _is_redirect(item):
            continue
        if _is_redirect(item.parent):
            raise RuntimeError("dependency cache entry parent is redirected")
        if os.path.commonpath((str(item.absolute()), str(cache_root.absolute()))) != str(cache_root.absolute()):
            raise RuntimeError("dependency cache entry escaped private cache root")
        destination = item.with_name(item.name + ".invalid-" + uuid.uuid4().hex)
        os.replace(item, destination)


@contextlib.contextmanager
def prepared_environment(
    *, runtime: os.PathLike[str] | str, project_root: os.PathLike[str] | str,
    requirements: os.PathLike[str] | str, fresh_parent: os.PathLike[str] | str,
    process_runner: Callable[..., Any], tree_manifest: Callable[[Path], dict[str, str]],
    verify_manifest: Callable[[Path, Any], bool], env: Mapping[str, str] | None = None,
) -> Iterator[Preparation]:
    """Prepare a cache hit/miss or uncached fallback, holding a key lock through use."""
    runtime_path = Path(runtime)
    root = Path(project_root).resolve(strict=True)
    requirements_path = _resolve_requirements(requirements)
    run_env = dict(os.environ if env is None else env)
    run_env.pop("TELEGRAM_BOT_TOKEN", None)
    run_env.pop("TELEGRAM_ALLOWED_USER_IDS", None)
    graph_files, cacheable, pip_supported, pip_digest = _install_inputs(
        requirements_path, run_env, runtime_path, process_runner,
    )
    if not cacheable or not pip_supported:
        with _fresh_environment(runtime_path, Path(fresh_parent), requirements_path, process_runner, run_env) as result:
            yield result
        return
    runtime_facts = _runtime_identity(runtime_path, process_runner, run_env)
    key = _cache_key(runtime_facts, pip_digest, graph_files)
    cache_root = _project_cache_root(root)
    environment, ready, lock_path = _entry_paths(cache_root, key)
    fresh_work = Path(fresh_parent)
    fresh_work.mkdir(parents=True, exist_ok=True)
    _ensure_safe_directory(cache_root, cache_root.parent)
    with _key_lock(lock_path, cache_root.parent):
        hit = _ready_hit(ready, environment, key, verify_manifest)
        if hit:
            result = _check_environment(environment, fresh_work, run_env, process_runner, runtime_path, runtime_facts)
        else:
            if environment.exists() or ready.exists() or _is_redirect(environment) or _is_redirect(ready):
                _quarantine(environment, ready, cache_root)
            result = _build_environment(
                runtime_path, requirements_path, fresh_work, run_env, process_runner,
                graph_files, pip_digest, runtime_facts, key, environment, ready,
                tree_manifest,
            )
        yield result


def _resolve_requirements(value: os.PathLike[str] | str) -> Path:
    absolute = Path(os.path.abspath(value))
    for component in (absolute, *absolute.parents):
        if _is_redirect(component):
            raise ValueError("requirements path is redirected")
    try:
        return absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("requirements file is missing") from exc


def _cache_key(runtime: Mapping[str, Any], pip_digest: str, files: list[dict[str, str]]) -> str:
    identity = {"policy": _POLICY, "runtime": runtime, "pip_configuration": pip_digest, "requirements": files}
    return _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())


def _project_cache_root(project_root: Path) -> Path:
    prefix = "tlbrain-deps" if os.name == "nt" else f"tlbrain-deps-{os.getuid()}"
    return Path(tempfile.gettempdir()).resolve() / prefix / _digest(str(project_root).encode())


def _entry_paths(cache_root: Path, key: str) -> tuple[Path, Path, Path]:
    entry_name = key[:24]
    environment = cache_root / entry_name
    return environment, cache_root / f"{entry_name}.ready.json", cache_root / f"{entry_name}.lock"


def _ready_hit(
    ready: Path, environment: Path, key: str, verify_manifest: Callable[[Path, Any], bool],
) -> bool:
    if _is_redirect(ready):
        return False
    try:
        if ready.exists() and not stat.S_ISREG(ready.lstat().st_mode):
            return False
        metadata = json.loads(ready.read_text(encoding="utf-8"))
        marker_hash = metadata.pop("marker_sha256", None)
        expected_hash = _digest(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        return (
            marker_hash == expected_hash
            and metadata.get("schema") == _READY_SCHEMA
            and metadata.get("key") == key
            and environment.is_dir()
            and not _is_redirect(environment)
            and verify_manifest(environment, metadata.get("manifest"))
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _venv_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _valid_venv_launcher(python: Path, runtime: Path) -> bool:
    if not python.is_file():
        return False
    if not _is_redirect(python):
        return True
    if os.name == "nt" or not python.is_symlink():
        return False
    try:
        return python.resolve(strict=True) == runtime.resolve(strict=True)
    except OSError:
        return False


def _check_environment(
    environment: Path, work: Path, env: Mapping[str, str], process_runner: Callable[..., Any],
    runtime: Path, runtime_identity: Mapping[str, Any],
) -> Preparation:
    python = _venv_python(environment)
    if not _valid_venv_launcher(python, runtime) or not _runtime_file_matches(runtime, runtime_identity):
        return Preparation(None, 1)
    checked = process_runner(
        [str(python), "-I", "-m", "pip", "check"],
        cwd=str(work), check=False, env=dict(env),
    )
    return Preparation(None if checked.returncode else python, checked.returncode)


def _build_environment(
    runtime: Path, requirements: Path, work: Path, env: Mapping[str, str],
    process_runner: Callable[..., Any], graph_files: list[dict[str, str]], pip_digest: str,
    runtime_identity: Mapping[str, Any], key: str, environment: Path, ready: Path,
    tree_manifest: Callable[[Path], dict[str, str]],
) -> Preparation:
    created = process_runner(
        [str(runtime), "-I", "-m", "venv", str(environment)],
        cwd=str(work), check=False, env=dict(env),
    )
    if created.returncode:
        return Preparation(None, created.returncode)
    python = _venv_python(environment)
    if not _valid_venv_launcher(python, runtime) or not _runtime_file_matches(runtime, runtime_identity):
        return Preparation(None, 1)
    installed = process_runner(
        [str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", str(requirements)],
        cwd=str(work), check=False, env=dict(env),
    )
    if installed.returncode:
        return Preparation(None, installed.returncode)
    checked = _check_environment(environment, work, env, process_runner, runtime, runtime_identity)
    if checked.status:
        return checked
    if not _cold_inputs_match(requirements, runtime, env, process_runner, graph_files, pip_digest, runtime_identity):
        return Preparation(None, 1)
    _atomic_ready(ready, {"schema": _READY_SCHEMA, "key": key, "manifest": tree_manifest(environment)})
    return checked


def _cold_inputs_match(
    requirements: Path, runtime: Path, env: Mapping[str, str], process_runner: Callable[..., Any],
    files: list[dict[str, str]], pip_digest: str, runtime_identity: Mapping[str, Any],
) -> bool:
    current_files, supported, config_ok, current_pip = _install_inputs(requirements, env, runtime, process_runner)
    if not supported or not config_ok or current_files != files or current_pip != pip_digest:
        return False
    if not _runtime_file_matches(runtime, runtime_identity):
        return False
    try:
        return _runtime_identity(runtime, process_runner, env) == runtime_identity
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return False


def _runtime_file_matches(runtime: Path, identity: Mapping[str, Any]) -> bool:
    try:
        return (str(runtime.resolve(strict=True)) == identity.get("executable")
                and _hash_file(runtime.resolve(strict=True)) == identity.get("executable_sha256"))
    except OSError:
        return False


@contextlib.contextmanager
def _fresh_environment(
    runtime: Path, parent: Path, requirements: Path, process_runner: Callable[..., Any], env: Mapping[str, str],
) -> Iterator[Preparation]:
    """Preserve the former temporary-vs-cacheable behavior for complex inputs."""
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tlbrain-dependencies-", dir=parent) as temporary:
        work = Path(temporary)
        environment = work / "venv"
        created = process_runner([str(runtime), "-I", "-m", "venv", str(environment)], cwd=str(work), check=False, env=dict(env))
        if created.returncode:
            yield Preparation(None, created.returncode)
            return
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError("runtime did not create its runtime-only virtual environment")
        installed = process_runner([str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", str(requirements)], cwd=str(work), check=False, env=dict(env))
        if installed.returncode:
            yield Preparation(None, installed.returncode)
            return
        checked = process_runner([str(python), "-I", "-m", "pip", "check"], cwd=str(work), check=False, env=dict(env))
        yield Preparation(None if checked.returncode else python, checked.returncode)


def _install_inputs(
    requirements: Path, env: Mapping[str, str], runtime: Path, process_runner: Callable[..., Any],
) -> tuple[list[dict[str, str]], bool, bool, str]:
    files: list[dict[str, str]] = []
    cacheable = _read_requirements(requirements, set(), set(), files, requirements.resolve(strict=True).parent)
    try:
        pip_supported, pip_digest = _pip_configuration(env, runtime, process_runner)
    except (OSError, ValueError):
        pip_supported, pip_digest = False, "unsupported-pip-configuration"
    return files, cacheable, pip_supported, pip_digest

