"""The pure Drive folder module must import without external SDKs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.offline
def test_pure_folder_module_imports_without_external_sdk_roots():
    code = "\n".join(
        [
            "import importlib.abc, sys",
            f"sys.path.insert(0, {str(APP_ROOT)!r})",
            "blocked = ('google', 'googleapiclient', 'firebase_admin', 'firebase_functions', 'qdrant_client', 'grpc', 'vertexai', 'fastembed', 'core.config', 'core.integrations.firestore', 'core.google_drive')",
            "class RejectSDKImports(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if any(fullname == root or fullname.startswith(root + '.') for root in blocked):",
            "            raise AssertionError('external SDK import attempted: ' + fullname)",
            "sys.meta_path.insert(0, RejectSDKImports())",
            "import core.integrations.google_drive.folders",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.offline
def test_folder_service_imports_without_gemini_or_qdrant():
    code = "\n".join(
        [
            "import importlib.abc, sys",
            f"sys.path.insert(0, {str(APP_ROOT)!r})",
            "blocked = ('core.integrations.gemini', 'google.generativeai', 'qdrant_client', 'core.integrations.qdrant')",
            "class RejectAIImports(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if any(fullname == root or fullname.startswith(root + '.') for root in blocked):",
            "            raise AssertionError('prohibited integration import attempted: ' + fullname)",
            "sys.meta_path.insert(0, RejectAIImports())",
            "import core.integrations.google_drive.folder_service",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
