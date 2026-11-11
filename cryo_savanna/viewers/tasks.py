# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Background reads for the viewers: functions run in a shared thread pool, results come back on the GUI thread.

A result is dropped when its owner widget is gone, or when a newer request was made on the same
channel (e.g. the user moved on to the next micrograph before the last one was read). A request that
is still waiting for a thread when a newer one is made on its channel is not run at all: stepping
quickly through micrographs or dragging a slider does not make the last one wait behind reads nobody
will see.
"""

from __future__ import annotations

import atexit
import os
import time
import traceback
from typing import Callable

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from qtpy.QtWidgets import QApplication

_POOL: QThreadPool | None = None


def pool() -> QThreadPool:
    global _POOL
    if _POOL is None:
        _POOL = QThreadPool()
        _POOL.setMaxThreadCount(max(2, min(4, (os.cpu_count() or 2) - 1)))
        atexit.register(_POOL.waitForDone, 10000)  # no reads still running while Python shuts down
    return _POOL


class TaskError:
    def __init__(self, error: BaseException):
        self.error = error
        self.trace = traceback.format_exc()

    def __str__(self) -> str:
        e = self.error
        return str(e) if isinstance(e, (OSError, ValueError)) else f"{type(e).__name__}: {e}"


class _Signals(QObject):
    done = Signal(int, object)


class _Job(QRunnable):
    def __init__(self, ticket: int, fn: Callable, signals: _Signals, stale: Callable[[], bool] | None = None):
        super().__init__()
        self.ticket, self.fn, self.signals, self.stale = ticket, fn, signals, stale

    def run(self) -> None:
        if self.stale is not None and self.stale():
            result = None  # superseded while it waited: _deliver drops it (and forgets its callback)
        else:
            try:
                result = self.fn()
            except Exception as e:  # noqa: BLE001 - shown in the viewer
                result = TaskError(e)
        try:
            self.signals.done.emit(self.ticket, result)
        except RuntimeError:  # the owner is gone
            pass


class Tasks(QObject):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._signals = _Signals()
        self._signals.done.connect(self._deliver)
        self._callbacks: dict[int, tuple[Callable | None, str | None]] = {}
        self._latest: dict[str, int] = {}
        self._next = 0

    def run(self, fn: Callable, callback: Callable | None = None, channel: str | None = None) -> int:
        self._next += 1
        ticket = self._next
        self._callbacks[ticket] = (callback, channel)
        stale = None
        if channel is not None:
            self._latest[channel] = ticket
            latest = self._latest  # read in the pool's thread: one dictionary look-up
            stale = lambda: latest.get(channel) != ticket  # noqa: E731
        pool().start(_Job(ticket, fn, self._signals, stale))
        return ticket

    def cancel(self, channel: str) -> None:
        """Drop the result of the pending request on this channel."""
        self._latest[channel] = -1

    @Slot(int, object)
    def _deliver(self, ticket: int, result) -> None:
        callback, channel = self._callbacks.pop(ticket, (None, None))
        if channel is not None and self._latest.get(channel) != ticket:
            return
        if callback is not None:
            callback(result)

    def busy(self) -> bool:
        return bool(self._callbacks)

    def wait(self, timeout: float = 10.0) -> bool:
        """Process events until every request was delivered (tests)."""
        end = time.monotonic() + timeout
        while self._callbacks and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()
        return not self._callbacks
