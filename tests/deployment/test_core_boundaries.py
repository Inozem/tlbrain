"""Offline checks for the core/service and integration ownership boundaries."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = APP_ROOT / "core"
INTEGRATION_PACKAGES = (
    "cloud_tasks",
    "fastembed",
    "firestore",
    "gemini",
    "google_auth",
    "google_docs",
    "google_drive",
    "qdrant",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "migrations" not in path.relative_to(APP_ROOT).parts
        and "tests" not in path.relative_to(APP_ROOT).parts
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _service_import_violations(tree: ast.Module, source_name: str) -> list[str]:
    violations: list[str] = []
    module_aliases: set[str] = set()
    dynamic_function_aliases: set[str] = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"importlib", "builtins"}:
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "importlib":
                dynamic_function_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            elif node.module == "builtins":
                dynamic_function_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )

    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Call):
            function = node.func
            dynamic = isinstance(function, ast.Name) and function.id in dynamic_function_aliases
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
                dynamic = dynamic or (
                    function.value.id in module_aliases
                    and function.attr in {"import_module", "__import__"}
                )
            if dynamic:
                value = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                if not isinstance(value, str):
                    violations.append(
                        f"{source_name}:{getattr(node, 'lineno', '?')}: "
                        "unresolvable dynamic import target"
                    )
                else:
                    imported.append(value)

        for name in imported:
            if name == "services" or name.startswith("services."):
                violations.append(
                    f"{source_name}:{getattr(node, 'lineno', '?')}: {name}"
                )
    return violations


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    credential_prefixes = (
        "GOOGLE_",
        "GEMINI_",
        "QDRANT_",
        "MCP_",
        "TLDV_",
        "TL_DV_",
        "TLDV",
    )
    credential_markers = ("API_KEY", "TOKEN", "SECRET", "CREDENTIAL")
    for name in tuple(env):
        if name.startswith(credential_prefixes) or any(
            marker in name.upper() for marker in credential_markers
        ):
            env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=APP_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.offline
def test_live_core_has_no_absolute_or_dynamic_services_imports() -> None:
    violations: list[str] = []
    for path in _python_files(CORE_ROOT):
        violations.extend(
            _service_import_violations(_parse(path), str(path.relative_to(APP_ROOT)))
        )

    assert not violations, "core must not import service modules:\n" + "\n".join(violations)


@pytest.mark.offline
@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module as load_module\nload_module('services.api')",
        "import importlib as loader\nloader.import_module(target)",
        "from builtins import __import__ as load_module\nload_module('services.api')",
        "import services.api",
    ],
)
def test_ast_boundary_detects_aliases_and_unresolvable_dynamic_imports(source: str) -> None:
    assert _service_import_violations(ast.parse(source), "synthetic.py")


@pytest.mark.offline
@pytest.mark.parametrize(
    "package",
    ["core.integrations"]
    + [f"core.integrations.{name}" for name in INTEGRATION_PACKAGES]
    + ["core.integrations.google_drive.folders"],
)
def test_integration_import_does_not_load_sibling_packages(package: str) -> None:
    package_name = package.removeprefix("core.integrations.").split(".")[0]
    siblings = tuple(name for name in INTEGRATION_PACKAGES if name != package_name)
    code = "\n".join(
        [
            "import importlib, sys",
            f"sys.path.insert(0, {str(APP_ROOT)!r})",
            f"importlib.import_module({package!r})",
            f"siblings = {siblings!r}",
            "loaded = [name for name in sys.modules if any(name == 'core.integrations.' + sibling or name.startswith('core.integrations.' + sibling + '.') for sibling in siblings)]",
            "assert not loaded, 'sibling integration packages loaded: ' + repr(loaded)",
        ]
    )
    completed = _run_isolated(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.offline
def test_importing_integration_modules_constructs_no_clients_or_models() -> None:
    module_names = sorted(
        ".".join(path.relative_to(CORE_ROOT).with_suffix("").parts)
        for path in _python_files(CORE_ROOT / "integrations")
        if path.name != "__init__.py"
    )
    code = "\n".join(
        [
            "import importlib, importlib.abc, socket, sys",
            f"sys.path.insert(0, {str(APP_ROOT)!r})",
            "def no_network(*args, **kwargs): raise AssertionError('network access attempted')",
            "socket.create_connection = no_network",
            "socket.socket.connect = no_network",
            "socket.socket.connect_ex = no_network",
            "import google.cloud.firestore as firestore_sdk",
            "from google.cloud import tasks_v2",
            "from google import genai",
            "import google.auth",
            "import google_auth_httplib2",
            "import httplib2",
            "from googleapiclient import discovery",
            "from qdrant_client import QdrantClient",
            "from fastembed import SparseTextEmbedding",
            "calls = []",
            "def spy(label):",
            "    def call(*args, **kwargs): calls.append(label); raise AssertionError(label + ' constructed during import')",
            "    return call",
            "firestore_sdk.Client.__init__ = spy('Firestore')",
            "tasks_v2.CloudTasksClient.__init__ = spy('Cloud Tasks')",
            "genai.Client.__init__ = spy('Gemini')",
            "google.auth.default = spy('Google auth default')",
            "google_auth_httplib2.AuthorizedHttp.__init__ = spy('AuthorizedHttp')",
            "httplib2.Http.__init__ = spy('httplib2.Http')",
            "discovery.build = spy('Drive/Docs discovery')",
            "QdrantClient.__init__ = spy('Qdrant')",
            "SparseTextEmbedding.__init__ = spy('FastEmbed')",
            f"modules = {module_names!r}",
            "for suffix in modules: importlib.import_module('core.' + suffix)",
            "assert calls == [], repr(calls)",
        ]
    )
    completed = _run_isolated(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.offline
def test_parser_windowing_and_folder_algorithms_import_with_unrelated_roots_blocked() -> None:
    code = "\n".join(
        [
            "import importlib.abc, sys",
            f"sys.path.insert(0, {str(APP_ROOT)!r})",
            "blocked = ('services', 'google', 'googleapiclient', 'firebase_admin', 'firebase_functions', 'qdrant_client', 'grpc', 'vertexai', 'fastembed', 'core.integrations.cloud_tasks', 'core.integrations.fastembed', 'core.integrations.firestore', 'core.integrations.gemini', 'core.integrations.google_auth', 'core.integrations.google_docs', 'core.integrations.qdrant')",
            "class RejectImports(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if any(fullname == root or fullname.startswith(root + '.') for root in blocked): raise AssertionError('blocked import attempted: ' + fullname)",
            "sys.meta_path.insert(0, RejectImports())",
            "from core.parsing.parser import parse_document",
            "from core.parsing.windowing import generate_windows",
            "from core.integrations.google_drive.folders import resolve_folder_path, expand_subtree",
            "meta, utterances = parse_document('DATE: 2026-01-01\\nPROVIDER: synthetic\\n---\\nA :: hello\\nB :: world')",
            "assert meta == {'date': '2026-01-01', 'provider': 'synthetic'} and [u['order_index'] for u in utterances] == [0, 1]",
            "assert generate_windows(utterances, anchor_step=1, half_window=1)[0]['covered_range'] == [0, 1]",
            "folders = {'a': {'name': 'same', 'parent_id': 'root'}, 'b': {'name': 'same', 'parent_id': 'root'}, 'c': {'name': 'leaf', 'parent_id': 'a'}}",
            "assert resolve_folder_path(['same'], folders, 'root') == ['a', 'b']",
            "assert expand_subtree('a', folders) == ['a', 'c']",
        ]
    )
    completed = _run_isolated(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _function_nodes(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(_parse(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    } | {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


@pytest.mark.offline
def test_audited_shared_operation_owners_and_compatibility_forwarders() -> None:
    owners = {
        "core/integrations/google_drive/folders.py": {
            "resolve_folder_path",
            "expand_subtree",
            "make_folder_path_resolver",
            "build_folder_tree",
        },
        "core/integrations/google_drive/folder_service.py": {
            "list_folders",
            "get_or_create_folder",
            "resolve_registered_folder_path",
            "expand_registered_subtree",
        },
        "core/integrations/google_drive/scanner.py": {"scan_root_folder"},
        "core/integrations/google_drive/changes.py": {
            "get_start_page_token",
            "get_drive_changes",
        },
        "core/integrations/google_drive/files.py": {
            "get_file_parent_folder_id",
            "move_file_to_folder",
            "rename_file",
        },
        "core/integrations/fastembed/bm25.py": {"get_bm25_model"},
        "core/integrations/firestore/folders.py": {
            "get_all_folders",
            "upsert_folder",
        },
        "core/indexing/processor.py": {"process_one"},
    }
    for relative, expected in owners.items():
        assert expected <= _function_nodes(APP_ROOT / relative).keys(), relative

    allowed_definitions = {
        "resolve_folder_path": {"core/integrations/google_drive/folders.py"},
        "expand_subtree": {"core/integrations/google_drive/folders.py"},
        "make_folder_path_resolver": {"core/integrations/google_drive/folders.py"},
        "build_folder_tree": {"core/integrations/google_drive/folders.py"},
        "list_folders": {
            "core/integrations/google_drive/folder_service.py",
            "core/retrieval/folders.py",
        },
        "get_or_create_folder": {"core/integrations/google_drive/folder_service.py"},
        "get_all_folders": {"core/integrations/firestore/folders.py"},
        "upsert_folder": {"core/integrations/firestore/folders.py"},
        "get_bm25_model": {"core/integrations/fastembed/bm25.py"},
        "process_one": {"core/indexing/processor.py"},
    }
    actual_definitions: dict[str, set[str]] = {name: set() for name in allowed_definitions}
    for root in (CORE_ROOT, APP_ROOT / "services"):
        for path in _python_files(root):
            for name in _function_nodes(path):
                if name in actual_definitions:
                    actual_definitions[name].add(path.relative_to(APP_ROOT).as_posix())
    assert actual_definitions == allowed_definitions

    wrappers = {
        "core/retrieval/folders.py": {"list_folders": "service_list_folders"},
    }
    for relative, expected_calls in wrappers.items():
        functions = _function_nodes(APP_ROOT / relative)
        for name, forwarded_call in expected_calls.items():
            assert forwarded_call in _called_names(functions[name]), f"{relative}:{name}"


@pytest.mark.offline
def test_live_runtime_has_no_legacy_google_drive_package_references() -> None:
    legacy_package = CORE_ROOT / "google_drive"
    assert not legacy_package.exists()
    runtime_files = _python_files(CORE_ROOT) + _python_files(APP_ROOT / "services")
    references = []
    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        if "core.google_drive" in source or "core/google_drive" in source:
            references.append(str(path.relative_to(APP_ROOT)))
    assert not references, "legacy runtime references remain:\n" + "\n".join(references)
