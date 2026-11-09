"""Restartable bounded cleanup for committed Telegram drafts."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any


def _value(record: Any, name: str, default: Any = None) -> Any:
    """Read a completion identity from either a record or a mapping."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


async def _await(value: Any) -> Any:
    """Accept synchronous repositories and async test or adapter seams."""
    return await value if inspect.isawaitable(value) else value


def _prepared_cleanup_methods(repository: Any) -> tuple[Any, Any]:
    """Require both prepared-page operations before cleanup can start deleting."""
    list_page = getattr(repository, "list_prepared_pages", None)
    delete_page = getattr(repository, "delete_prepared_page", None)
    if not callable(list_page) or not callable(delete_page):
        raise RuntimeError("repository must implement prepared-page cleanup")
    return list_page, delete_page


def _accepts(method: Any, name: str) -> bool:
    """Check whether a repository seam accepts an optional identity argument."""
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


class TelegramDraftCleanup:
    """Authorize completed draft cleanup once, then remove children in pages."""

    def __init__(self, *, repository: Any, page_size: int = 200) -> None:
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        self.repository = repository
        self.page_size = min(page_size, 400)

    async def cleanup(self, *, context_id: str, certificate: Any = None) -> bool:
        """Remove source and prepared pages only after durable authorization."""
        if certificate is None:
            certificate = await _await(self.repository.load_completion_certificate(context_id))
        if certificate is None or _value(certificate, "context_id") != context_id:
            return False
        draft_id = _value(certificate, "draft_id")
        if not isinstance(draft_id, str) or not draft_id:
            return False
        _prepared_cleanup_methods(self.repository)
        authorized = await _await(self.repository.authorize_cleanup(context_id, certificate))
        if authorized is not True:
            return False
        await self._delete_source_pages(context_id, draft_id)
        await self._delete_prepared_pages(context_id, draft_id)
        purge = getattr(self.repository, "cleanup_temporary_save_payloads", None)
        if callable(purge) and await _await(purge(context_id, certificate)) is not True:
            return False
        finalized = await _await(self.repository.finalize_cleanup(context_id, certificate))
        return finalized is True

    async def _delete_source_pages(self, context_id: str, draft_id: str) -> None:
        """Delete the first remaining source page until the source set is empty."""
        while True:
            list_kwargs = {"cursor": None, "limit": self.page_size}
            if _accepts(self.repository.list_manifest_pages, "context_id"):
                list_kwargs["context_id"] = context_id
            page, _ = await _await(self.repository.list_manifest_pages(draft_id, **list_kwargs))
            if not page:
                return
            delete_kwargs = {"context_id": context_id} if _accepts(
                self.repository.delete_draft_page, "context_id",
            ) else {}
            await _await(self.repository.delete_draft_page(draft_id, page, **delete_kwargs))

    async def _delete_prepared_pages(self, context_id: str, draft_id: str) -> None:
        """Delete prepared utterances in bounded pages after source checkpoints."""
        list_page, delete_page = _prepared_cleanup_methods(self.repository)
        while True:
            page = await _await(list_page(
                context_id, draft_id, limit=self.page_size,
            ))
            if not page:
                return
            await _await(delete_page(context_id, draft_id, page))
