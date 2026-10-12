import builtins
import importlib
from unittest.mock import Mock, call

from google.cloud import firestore

from core.integrations.firestore import client as firestore_client
from core.google_drive import firestore as legacy_firestore


def test_collection_and_document_identifiers_are_preserved():
    assert firestore_client.COLLECTION_NAME == "transcript_index"
    assert firestore_client.CLIENTS_COLLECTION == "clients"
    assert firestore_client.FOLDERS_COLLECTION == "folders"
    assert firestore_client.TOKENS_COLLECTION == "tokens"
    assert firestore_client.DRIVE_SYNC_TOKEN_DOC == "drive_sync"
    assert legacy_firestore.COLLECTION_NAME == firestore_client.COLLECTION_NAME
    assert legacy_firestore.CLIENTS_COLLECTION == firestore_client.CLIENTS_COLLECTION
    assert legacy_firestore.FOLDERS_COLLECTION == firestore_client.FOLDERS_COLLECTION
    assert legacy_firestore.TOKENS_COLLECTION == firestore_client.TOKENS_COLLECTION
    assert legacy_firestore.DRIVE_SYNC_TOKEN_DOC == firestore_client.DRIVE_SYNC_TOKEN_DOC


def test_get_client_constructs_fresh_default_client_on_each_call(monkeypatch):
    first = object()
    second = object()
    constructor = Mock(side_effect=[first, second])
    monkeypatch.setattr(firestore, "Client", constructor)

    assert firestore_client.get_client() is first
    assert firestore_client.get_client() is second
    assert constructor.call_args_list == [call(), call()]


def test_legacy_database_helper_forwards_to_factory(monkeypatch):
    first = object()
    second = object()
    constructor = Mock(side_effect=[first, second])
    monkeypatch.setattr(firestore, "Client", constructor)

    assert legacy_firestore._get_db() is first
    assert legacy_firestore._get_db() is second
    assert constructor.call_args_list == [call(), call()]


def test_module_import_is_lazy_and_stays_within_firestore_boundary(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(firestore, "Client", constructor)
    imported_modules = []
    original_import = builtins.__import__

    def recording_import(name, *args, **kwargs):
        imported_modules.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    importlib.reload(firestore_client)
    assert constructor.call_count == 0
    assert [name for name in imported_modules if not name.startswith("_")] == ["google.cloud"]
