# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Work that takes a while (deleting scratch folders, reading every job folder), done in a thread while
the window stays alive: it is painted and can be moved, with a note saying what is going on.

``run_busy`` is for work whose result is needed before going on (the caller waits, the window does not
freeze); ``when_done`` for a value that is shown when it is there.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from qtpy.QtCore import QEventLoop, Qt, QTimer
from qtpy.QtWidgets import QProgressDialog, QWidget

QUICK_S = 0.15  # work done within this needs no note (and no flicker of one)
POLL_MS = 30


class _Work(threading.Thread):
    def __init__(self, fn: Callable):
        super().__init__(name="cryo-savanna-busy", daemon=True)
        self.fn = fn
        self.value = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.value = self.fn()
        except BaseException as e:  # noqa: BLE001 - raised again by the caller
            self.error = e

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _Note(QProgressDialog):
    """A note without buttons that stays until the work is done (Esc and the close button do nothing)."""

    def __init__(self, text: str, parent: QWidget | None):
        super().__init__(text, "", 0, 0, parent)
        self.setCancelButton(None)
        self.setWindowTitle("cryo-savanna")
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.finished_work = False

    def reject(self) -> None:
        if self.finished_work:
            super().reject()

    def closeEvent(self, e) -> None:
        if self.finished_work:
            super().closeEvent(e)
        else:
            e.ignore()


def run_busy(fn: Callable, text: str = "Working…", parent: QWidget | None = None):
    """fn() in a thread; returns its result (or raises what it raised) once it is done. Meanwhile the
    windows are painted and a note says ``text``; nothing else can be clicked."""
    work = _Work(fn)
    work.start()
    work.join(QUICK_S)
    if work.is_alive():
        note = _Note(text, parent)
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: loop.quit() if not work.is_alive() else None)
        timer.start(POLL_MS)
        note.show()
        loop.exec()
        timer.stop()
        note.finished_work = True
        note.close()
        note.deleteLater()
        work.join()
    return work.result()


def when_done(owner: QWidget, fn: Callable, done: Callable) -> None:
    """fn() in a thread, then done(result) in the GUI thread (not if ``owner`` is gone by then, nor if fn
    raised). At once when fn is quick."""
    work = _Work(fn)
    work.start()
    work.join(QUICK_S)

    def deliver() -> None:
        if work.error is None:
            done(work.value)

    if not work.is_alive():
        deliver()
        return
    timer = QTimer(owner)  # goes with its owner

    def check() -> None:
        if not work.is_alive():
            timer.stop()
            timer.deleteLater()
            deliver()

    timer.timeout.connect(check)
    timer.start(4 * POLL_MS)
