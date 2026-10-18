"""Small, stateful Firestore SDK fake for offline service tests.

The public surface intentionally covers the repository's document, query,
batch, and transaction consumers. Unsupported SDK operations fail loudly.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from google.api_core.exceptions import Aborted, NotFound
from google.cloud import firestore
from google.cloud.firestore_v1 import _helpers
from google.cloud.firestore_v1 import field_path as field_path_module


_MISSING = object()


@dataclass
class _Fault:
    operation: str
    path: str | None
    phase: str
    exception: BaseException


@dataclass
class _State:
    documents: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    revisions: dict[tuple[str, str], int] = field(default_factory=dict)
    faults: list[_Fault] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    clock: Callable[[], datetime] = field(default_factory=lambda: lambda: datetime.now(timezone.utc))
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATE = _State()


def _doc_path(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def _clock_now() -> datetime:
    value = _STATE.clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Firestore fake clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _copy(data: Any) -> Any:
    return copy.deepcopy(data)


def _parts(field_path: str) -> tuple[str, ...]:
    if isinstance(field_path, field_path_module.FieldPath):
        if not field_path.parts:
            raise ValueError("Firestore field path cannot be empty")
        return tuple(field_path.parts)
    if not isinstance(field_path, str) or not field_path:
        raise ValueError(f"Invalid Firestore field path: {field_path!r}")
    # SDK FieldPath parsing validates escaping and empty path segments.
    return tuple(field_path_module.parse_field_path(field_path))


def _get_path(data: dict[str, Any], parts: tuple[str, ...]) -> Any:
    node: Any = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_path(data: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    node = data
    for part in parts[:-1]:
        old = node.get(part)
        if not isinstance(old, dict):
            old = {}
            node[part] = old
        node = old
    node[parts[-1]] = _copy(value)


def _delete_path(data: dict[str, Any], parts: tuple[str, ...]) -> None:
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _apply_transform(data: dict[str, Any], path: tuple[str, ...], value: Any, now: datetime) -> None:
    if value is firestore.DELETE_FIELD:
        _delete_path(data, path)
    elif value is firestore.SERVER_TIMESTAMP:
        _set_path(data, path, now)
    elif isinstance(value, firestore.Increment):
        old = _get_path(data, path)
        base = old if isinstance(old, (int, float)) and not isinstance(old, bool) else 0
        delta = value.value
        _set_path(data, path, base + delta)
    else:
        _set_path(data, path, value)


def _materialize_map(data: dict[str, Any], now: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is _helpers._EmptyDict:
            result[key] = {}
        elif isinstance(value, dict):
            result[key] = _materialize_map(value, now)
        else:
            _apply_transform(result, (key,), value, now)
    return result


_SDK_DOCUMENT_PATH = "projects/firestore-fake/databases/(default)/documents/fixture/document"


def _reject_unsupported_transforms(extractor: Any) -> None:
    if extractor.array_removes or extractor.array_unions or extractor.maximums or extractor.minimums:
        raise NotImplementedError("Firestore fake supports only Increment and SERVER_TIMESTAMP transforms")


def _validate_data(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("Firestore document data must be a dictionary")
    # The SDK builders validate field paths, masks, transforms, and Firestore values.
    _helpers.pbs_for_set_no_merge(_SDK_DOCUMENT_PATH, data)


def _apply_set(existing: dict[str, Any] | None, data: dict[str, Any], merge: Any, now: datetime) -> dict[str, Any]:
    if merge is False:
        _validate_data(data)
        extractor = _helpers.DocumentExtractor(data)
        _reject_unsupported_transforms(extractor)
        if extractor.deleted_fields:
            raise ValueError("DELETE_FIELD is only allowed in merge sets or updates")
        result = _materialize_map(extractor.set_fields, now)
        _apply_extractor_transforms(result, extractor, now)
        return result
    # Run the SDK builder first; it handles True and explicit string/FieldPath masks.
    _helpers.pbs_for_set_with_merge(_SDK_DOCUMENT_PATH, data, merge)
    _reject_unsupported_transforms(_helpers.DocumentExtractor(data))
    extractor = _helpers.DocumentExtractorForMerge(data)
    extractor.apply_merge(merge)
    _reject_unsupported_transforms(extractor)
    result = _copy(existing or {})
    mask_paths = [path for path in extractor.merge if path not in extractor.transform_merge]
    for field_path in mask_paths:
        path = _parts(field_path)
        try:
            value = _helpers.get_field_value(extractor.set_fields, field_path)
        except KeyError:
            _delete_path(result, path)
        else:
            _set_path(result, path, _materialize_value(value, now))
    for field_path in extractor.deleted_fields:
        _delete_path(result, _parts(field_path))
    _apply_extractor_transforms(result, extractor, now)
    return result


def _materialize_value(value: Any, now: datetime) -> Any:
    if value is _helpers._EmptyDict:
        return {}
    if isinstance(value, dict):
        return _materialize_map(value, now)
    result: dict[str, Any] = {}
    _apply_transform(result, ("value",), value, now)
    return result["value"]


def _apply_extractor_transforms(data: dict[str, Any], extractor: Any, now: datetime) -> None:
    for path in extractor.deleted_fields:
        _delete_path(data, _parts(path))
    for path in extractor.server_timestamps:
        _apply_transform(data, _parts(path), firestore.SERVER_TIMESTAMP, now)
    for path, amount in extractor.increments.items():
        _apply_transform(data, _parts(path), firestore.Increment(amount), now)


def _apply_update(existing: dict[str, Any] | None, data: dict[str, Any], now: datetime) -> dict[str, Any]:
    if existing is None:
        raise NotFound("Document does not exist")
    if not isinstance(data, dict):
        raise TypeError("Firestore update data must be a dictionary")
    _helpers.pbs_for_update(_SDK_DOCUMENT_PATH, data, None)
    extractor = _helpers.DocumentExtractorForUpdate(data)
    _reject_unsupported_transforms(extractor)
    result = _copy(existing)
    transform_paths = set(extractor.transform_paths)
    for field_path in extractor.top_level_paths:
        if field_path in transform_paths:
            continue
        path = _parts(field_path)
        try:
            value = _helpers.get_field_value(extractor.set_fields, field_path)
        except KeyError:
            _delete_path(result, path)
        else:
            _set_path(result, path, _materialize_value(value, now))
    _apply_extractor_transforms(result, extractor, now)
    return result


class FakeFirestoreClient:
    """Client constructor-compatible fake; all clients share fixture state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise TypeError("FakeFirestoreClient accepts no SDK client options")

    @classmethod
    def reset(cls) -> None:
        with _STATE.lock:
            _STATE.documents.clear()
            _STATE.revisions.clear()
            _STATE.faults.clear()
            _STATE.trace.clear()
            _STATE.clock = lambda: datetime.now(timezone.utc)

    @classmethod
    def set_clock(cls, clock: Callable[[], datetime]) -> None:
        _STATE.clock = clock

    @classmethod
    def fail_next(
        cls,
        operation: str,
        *,
        path: str | None = None,
        phase: str = "before",
        exception: BaseException | None = None,
    ) -> None:
        if phase not in {"before", "after"}:
            raise ValueError("phase must be 'before' or 'after'")
        if operation not in {"set", "update", "delete", "commit", "batch_commit"}:
            raise ValueError(f"Unsupported fault operation: {operation}")
        _STATE.faults.append(_Fault(operation, path, phase, exception or RuntimeError(f"injected {phase} {operation} failure")))

    @classmethod
    def trace(cls) -> list[dict[str, Any]]:
        with _STATE.lock:
            return _copy(_STATE.trace)

    def collection(self, name: str) -> "FakeCollectionReference":
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("collection name must be one non-empty path segment")
        return FakeCollectionReference(name)

    def transaction(self, max_attempts: int = 5, read_only: bool = False) -> "FakeTransaction":
        return FakeTransaction(max_attempts=max_attempts, read_only=read_only)

    def batch(self) -> "FakeWriteBatch":
        return FakeWriteBatch()


class FakeDocumentSnapshot:
    def __init__(self, reference: "FakeDocumentReference", data: dict[str, Any] | None) -> None:
        self.reference = reference
        self.id = reference.id
        self.exists = data is not None
        self._data = _copy(data) if data is not None else None

    def to_dict(self) -> dict[str, Any] | None:
        return _copy(self._data)


class FakeDocumentReference:
    def __init__(self, collection: str, document_id: str) -> None:
        if not isinstance(document_id, str) or not document_id or "/" in document_id:
            raise ValueError("document id must be one non-empty path segment")
        self._key = (collection, document_id)
        self.id = document_id

    @property
    def path(self) -> str:
        return _doc_path(self._key)

    def get(self, *, transaction: "FakeTransaction | None" = None, **kwargs: Any) -> FakeDocumentSnapshot:
        if kwargs:
            raise TypeError(f"Unsupported get options: {', '.join(kwargs)}")
        if transaction is not None:
            return transaction._read(self)
        with _STATE.lock:
            return FakeDocumentSnapshot(self, _STATE.documents.get(self._key))

    def set(self, document_data: dict[str, Any], merge: Any = False, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"Unsupported set options: {', '.join(kwargs)}")
        self._write("set", document_data, merge)

    def update(self, field_updates: dict[str, Any], **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"Unsupported update options: {', '.join(kwargs)}")
        self._write("update", field_updates)

    def delete(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(kwargs)}")
        self._write("delete")

    def _write(self, operation: str, data: Any = None, merge: Any = False) -> None:
        with _STATE.lock:
            _raise_fault(operation, self.path, "before")
            now = _clock_now()
            try:
                _commit_operations([(operation, self, _copy(data), merge)], now)
            except BaseException:
                _STATE.trace.append({"operation": operation, "path": self.path, "outcome": "failed", "phase": "commit"})
                raise
            _STATE.trace.append({"operation": operation, "path": self.path, "outcome": "committed"})
            _raise_fault(operation, self.path, "after")


class FakeCollectionReference:
    def __init__(self, name: str) -> None:
        self.id = name
        self.path = name

    def document(self, document_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self.id, document_id)

    def where(self, field_path: Any = None, op_string: str | None = None, value: Any = None, *, filter: Any = None) -> "FakeQuery":
        if filter is None:
            if field_path is None or op_string is None:
                raise TypeError("where requires a FieldFilter or field, operator, and value")
            filter = firestore.FieldFilter(field_path, op_string, value)
        elif field_path is not None or op_string is not None or value is not None:
            raise ValueError("Cannot combine filter with positional where arguments")
        return FakeQuery(self.id).where(filter=filter)

    def select(self, field_paths: list[str]) -> "FakeQuery":
        return FakeQuery(self.id).select(field_paths)

    def limit(self, count: int) -> "FakeQuery":
        return FakeQuery(self.id).limit(count)

    def stream(self, **kwargs: Any) -> list[FakeDocumentSnapshot]:
        return FakeQuery(self.id).stream(**kwargs)


class FakeQuery:
    def __init__(self, collection: str, filters: tuple[Any, ...] = (), projection: tuple[str, ...] | None = None, limit: int | None = None) -> None:
        self.collection = collection
        self.filters = filters
        self.projection = projection
        self.max_results = limit

    def where(self, field_path: Any = None, op_string: str | None = None, value: Any = None, *, filter: Any = None) -> "FakeQuery":
        if filter is None:
            filter = firestore.FieldFilter(field_path, op_string, value)
        field = getattr(filter, "field_path", None)
        op = getattr(filter, "op_string", None)
        op_name = getattr(op, "name", op)
        if not field or op_name not in {"==", ">", ">=", "<", "<=", "!=", "IS_NULL"}:
            raise NotImplementedError(f"Unsupported Firestore filter: {filter!r}")
        return FakeQuery(self.collection, self.filters + ((field, op_name, _copy(filter.value)),), self.projection, self.max_results)

    def select(self, field_paths: list[str]) -> "FakeQuery":
        if not isinstance(field_paths, (list, tuple)):
            raise TypeError("select expects a list of field paths")
        return FakeQuery(self.collection, self.filters, tuple(field_paths), self.max_results)

    def limit(self, count: int) -> "FakeQuery":
        if not isinstance(count, int) or count < 0:
            raise ValueError("limit must be a non-negative integer")
        return FakeQuery(self.collection, self.filters, self.projection, count)

    def stream(self, **kwargs: Any) -> list[FakeDocumentSnapshot]:
        if kwargs:
            raise TypeError(f"Unsupported query options: {', '.join(kwargs)}")
        if self.max_results == 0:
            return []
        with _STATE.lock:
            keys = sorted((key for key in _STATE.documents if key[0] == self.collection), key=lambda key: key[1])
            result = []
            for key in keys:
                data = _STATE.documents[key]
                if all(_matches(data, field, op, value) for field, op, value in self.filters):
                    if self.projection is not None:
                        selected: dict[str, Any] = {}
                        for path in self.projection:
                            parts = _parts(path)
                            value = _get_path(data, parts)
                            if value is not _MISSING:
                                _set_path(selected, parts, value)
                        data = selected
                    result.append(FakeDocumentSnapshot(FakeDocumentReference(*key), data))
                    if self.max_results is not None and len(result) >= self.max_results:
                        break
            return result


def _matches(data: dict[str, Any], path: str, op: str, expected: Any) -> bool:
    actual = _get_path(data, _parts(path))
    if actual is _MISSING:
        return False
    try:
        return {"==": lambda: actual == expected, "!=": lambda: actual != expected,
                "IS_NULL": lambda: actual is None,
                ">": lambda: actual > expected, ">=": lambda: actual >= expected,
                "<": lambda: actual < expected, "<=": lambda: actual <= expected}[op]()
    except TypeError:
        return False


class FakeTransaction:
    """Minimal transaction shape consumed by the real SDK transactional decorator."""

    def __init__(self, max_attempts: int = 5, read_only: bool = False) -> None:
        self._max_attempts = max_attempts
        self._read_only = read_only
        self._id: bytes | None = None
        self._write_pbs: list[Any] = []
        self._reads: dict[tuple[str, str], int] = {}
        self._writes: list[tuple[str, FakeDocumentReference, Any, Any]] = []
        self._closed = True

    @property
    def in_progress(self) -> bool:
        return self._id is not None

    def _clean_up(self) -> None:
        self._write_pbs = []
        self._reads = {}
        self._writes = []
        self._closed = True
        self._id = None

    def _begin(self, retry_id: bytes | None = None) -> None:
        if self.in_progress:
            raise ValueError("Transaction already begun")
        self._id = retry_id or uuid4().bytes
        self._closed = False

    def _read(self, reference: FakeDocumentReference) -> FakeDocumentSnapshot:
        if not self.in_progress:
            raise ValueError("Transaction has not begun")
        if self._writes:
            raise _helpers.ReadAfterWriteError("Cannot read after a write in a transaction")
        with _STATE.lock:
            self._reads.setdefault(reference._key, _STATE.revisions.get(reference._key, 0))
            return FakeDocumentSnapshot(reference, _STATE.documents.get(reference._key))

    def set(self, reference: FakeDocumentReference, document_data: dict[str, Any], merge: Any = False) -> None:
        self._stage("set", reference, document_data, merge)

    def update(self, reference: FakeDocumentReference, field_updates: dict[str, Any]) -> None:
        self._stage("update", reference, field_updates, False)

    def delete(self, reference: FakeDocumentReference) -> None:
        self._stage("delete", reference, None, False)

    def _stage(self, operation: str, reference: FakeDocumentReference, data: Any, merge: Any) -> None:
        if not self.in_progress:
            raise ValueError("Transaction has not begun")
        if self._read_only:
            raise ValueError("Cannot perform write operation in read-only transaction.")
        self._writes.append((operation, reference, _copy(data), merge))
        self._write_pbs.append((operation, reference.path))

    def _commit(self) -> list[Any]:
        if not self.in_progress:
            raise ValueError("Transaction has no transaction ID")
        with _STATE.lock:
            for key, revision in self._reads.items():
                if _STATE.revisions.get(key, 0) != revision:
                    _STATE.trace.append({"operation": "commit", "path": _doc_path(key), "outcome": "failed", "phase": "conflict"})
                    self._clean_up()
                    raise Aborted(f"Document {_doc_path(key)} changed during transaction")
            _raise_write_fault("commit", self._writes, "before")
            now = _clock_now()
            try:
                _commit_operations([(op, ref, data, merge) for op, ref, data, merge in self._writes], now)
            except BaseException:
                _STATE.trace.append({"operation": "commit", "path": None, "outcome": "failed", "phase": "commit"})
                self._clean_up()
                raise
            for _, ref, _, _ in self._writes:
                _STATE.trace.append({"operation": "commit", "path": ref.path, "outcome": "committed"})
            _raise_write_fault("commit", self._writes, "after")
        self._clean_up()
        return []

    def _rollback(self) -> None:
        if not self.in_progress:
            # The real SDK decorator rolls back after retry exhaustion too.
            self._id = uuid4().bytes
        self._clean_up()


class FakeWriteBatch:
    def __init__(self) -> None:
        self._writes: list[tuple[str, FakeDocumentReference, Any, Any]] = []

    def update(self, reference: FakeDocumentReference, field_updates: dict[str, Any]) -> "FakeWriteBatch":
        self._writes.append(("update", reference, _copy(field_updates), False))
        return self

    def set(self, reference: FakeDocumentReference, document_data: dict[str, Any], merge: Any = False) -> "FakeWriteBatch":
        self._writes.append(("set", reference, _copy(document_data), merge))
        return self

    def delete(self, reference: FakeDocumentReference) -> "FakeWriteBatch":
        self._writes.append(("delete", reference, None, False))
        return self

    def commit(self, **kwargs: Any) -> list[Any]:
        if kwargs:
            raise TypeError(f"Unsupported batch commit options: {', '.join(kwargs)}")
        with _STATE.lock:
            _raise_write_fault("batch_commit", self._writes, "before")
            try:
                _commit_operations(self._writes, _clock_now())
            except BaseException:
                _STATE.trace.append({"operation": "batch_commit", "path": None, "outcome": "failed", "phase": "commit"})
                raise
            for op, ref, _, _ in self._writes:
                _STATE.trace.append({"operation": "batch_commit", "path": ref.path, "write": op, "outcome": "committed"})
            _raise_write_fault("batch_commit", self._writes, "after")
        self._writes.clear()
        return []


def _raise_fault(operation: str, path: str | None, phase: str) -> None:
    for index, fault in enumerate(_STATE.faults):
        if fault.operation == operation and fault.phase == phase and (fault.path is None or fault.path == path):
            _STATE.faults.pop(index)
            _STATE.trace.append({"operation": operation, "path": path, "outcome": "failed", "phase": phase})
            raise fault.exception


def _raise_write_fault(operation: str, writes: list[tuple[str, FakeDocumentReference, Any, Any]], phase: str) -> None:
    _raise_fault(operation, None, phase)
    for _, reference, _, _ in writes:
        _raise_fault(operation, reference.path, phase)


def _commit_operations(writes: list[tuple[str, FakeDocumentReference, Any, Any]], now: datetime) -> None:
    # Build a private working copy, so a failed write never partially changes state.
    working = _copy(_STATE.documents)
    changed: set[tuple[str, str]] = set()
    for operation, ref, data, merge in writes:
        key = ref._key
        existing = working.get(key)
        if operation == "set":
            working[key] = _apply_set(existing, data, merge, now)
        elif operation == "update":
            working[key] = _apply_update(existing, data, now)
        elif operation == "delete":
            working.pop(key, None)
        else:
            raise NotImplementedError(f"Unsupported write operation: {operation}")
        changed.add(key)
    _STATE.documents.clear()
    _STATE.documents.update(working)
    for key in changed:
        _STATE.revisions[key] = _STATE.revisions.get(key, 0) + 1


class FirestoreFake:
    """Fixture facade: reset, fake state, deterministic clock, faults, and trace."""

    client_class = FakeFirestoreClient

    def reset(self) -> None:
        FakeFirestoreClient.reset()

    def set_clock(self, clock: Callable[[], datetime]) -> None:
        FakeFirestoreClient.set_clock(clock)

    def fail_next(self, operation: str, **kwargs: Any) -> None:
        FakeFirestoreClient.fail_next(operation, **kwargs)

    def trace(self) -> list[dict[str, Any]]:
        return FakeFirestoreClient.trace()

    @property
    def documents(self) -> dict[str, dict[str, Any]]:
        with _STATE.lock:
            return {_doc_path(key): _copy(value) for key, value in _STATE.documents.items()}


firestore_fake = FirestoreFake()
