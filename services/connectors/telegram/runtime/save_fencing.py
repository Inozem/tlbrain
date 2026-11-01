"""Telegram save retry and provider-call fencing adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.operations.retries import RetryGovernor, UnknownOutcome


def _provider_error_retryable(error: BaseException) -> bool:
    """Stop on definitive provider rejections while retrying transient responses."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "resp", None), "status", None)
    if type(status) is int and 400 <= status < 500:
        return status in {408, 429}
    return True


class _TelegramSaveRetryGovernor(RetryGovernor):
    """Use the common stage ledger with Telegram's definitive HTTP failure policy."""

    def run_stage(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("retryable", _provider_error_retryable)
        return super().run_stage(*args, **kwargs)


class _RuntimeClock:
    def __init__(self, clock: Any):
        self._clock = clock

    def now(self) -> datetime:
        return self._clock()


class _ParentBoundRetryGovernor(_TelegramSaveRetryGovernor):
    """Keep stage reservations and callbacks behind the live Telegram queue fence."""

    def __init__(self, source: Any, *, clock: Any, assert_current: Any):
        super().__init__(source.client, clock=_RuntimeClock(clock),
                         max_failures=source.max_failures,
                         lease_seconds=source.lease_seconds,
                         retry_store=source.retry_store)
        self._assert_parent = assert_current

    def _current(self) -> None:
        try:
            self._assert_parent()
        except Exception as error:
            raise UnknownOutcome("Telegram parent queue lease is unavailable") from error

    async def _current_async(self) -> None:
        from services.connectors.telegram.runtime.runtime import _offload
        try:
            await _offload(self._assert_parent)
        except Exception as error:
            raise UnknownOutcome("Telegram parent queue lease is unavailable") from error

    async def run_stage_async(self, operation_id: str, stage_key: str, callback: Any, **kwargs: Any) -> Any:
        kwargs["now"] = self.clock.now()
        await self._current_async()
        reconcile = kwargs.get("reconcile")

        async def guarded_callback() -> Any:
            await self._current_async()
            try:
                result = callback()
                if hasattr(result, "__await__"):
                    result = await result
            except UnknownOutcome:
                raise
            except Exception as error:
                await self._current_async()
                raise error
            await self._current_async()
            return result

        if reconcile is not None:
            async def guarded_reconcile() -> Any:
                await self._current_async()
                try:
                    result = reconcile()
                    if hasattr(result, "__await__"):
                        result = await result
                except UnknownOutcome:
                    raise
                except Exception as error:
                    await self._current_async()
                    raise error
                await self._current_async()
                return result
            kwargs["reconcile"] = guarded_reconcile
        return await super().run_stage_async(operation_id, stage_key, guarded_callback, **kwargs)

    def run_stage(self, operation_id: str, stage_key: str, callback: Any, **kwargs: Any) -> Any:
        kwargs["now"] = self.clock.now()
        self._current()
        reconcile = kwargs.get("reconcile")

        def guarded(function: Any) -> Any:
            def invoke() -> Any:
                self._current()
                try:
                    result = function()
                except UnknownOutcome:
                    raise
                except Exception as error:
                    self._current()
                    raise error
                self._current()
                return result
            return invoke

        if reconcile is not None:
            kwargs["reconcile"] = guarded(reconcile)
        return super().run_stage(operation_id, stage_key, guarded(callback), **kwargs)


class _GuardedGoogleProxy:
    """Check the queue lease immediately around each Google request execution."""

    def __init__(self, value: Any, assert_current: Any):
        self._value, self._assert_current = value, assert_current

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._value, name)
        if not callable(value):
            return value
        if name == "execute":
            def execute(*args: Any, **kwargs: Any) -> Any:
                self._assert_current()
                try:
                    result = value(*args, **kwargs)
                except Exception as error:
                    try:
                        self._assert_current()
                    except Exception as stale:
                        raise UnknownOutcome("Telegram parent queue lease changed during provider request") from stale
                    raise error
                self._assert_current()
                return result
            return execute

        def invoke(*args: Any, **kwargs: Any) -> Any:
            result = value(*args, **kwargs)
            return _GuardedGoogleProxy(result, self._assert_current) if result is not None else result
        return invoke
