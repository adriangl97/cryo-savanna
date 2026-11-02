# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Python's cyclic garbage collector, run on the GUI thread only.

Python collects cyclic garbage in whichever thread happens to allocate when a threshold is
passed, often a loader or summary thread. When that frees the Python side of a Qt object that
lives in the GUI thread (a closed window, its menus and actions), PySide hands the deletion to
the GUI thread, which runs it holding the GIL and waits for a Qt lock that the worker holds while
it waits for the GIL: both wait for ever. So automatic collection is switched off, and a timer
collects on the GUI thread instead, by the same thresholds.
"""

from __future__ import annotations

import gc

from qtpy.QtCore import QObject, QTimer

INTERVAL_MS = 500

_guard: "MainThreadGC | None" = None


class MainThreadGC(QObject):
    def __init__(self, parent: QObject, interval_ms: int = INTERVAL_MS):
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    def check(self) -> None:
        """Collect the generations whose thresholds were passed, as the automatic collector would."""
        (c0, c1, c2), (t0, t1, t2) = gc.get_count(), self.threshold
        if c0 > t0:
            gc.collect(2 if c2 > t2 else 1 if c1 > t1 else 0)


def install(app: QObject) -> MainThreadGC:
    """Collect garbage only on the GUI thread (once per process; ``app`` is the QApplication)."""
    global _guard
    if _guard is None:
        _guard = MainThreadGC(app)
    return _guard
