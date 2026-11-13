# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Errors nothing caught: said in the window and kept in a file, not only printed.

An exception in a Qt slot (a button, a timer, a signal from a reading thread) ends that slot and nothing
else: Python prints the traceback to the terminal cryo-savanna was started from, which nobody looks at (or
which is gone: started from a desktop menu). ``install()`` keeps printing it, and also

- appends it, with the time, to ``errors.log`` in cryo-savanna's cache folder (``error_log()``; not in the
  project, which may be read-only or someone else's), and
- emits ``reported`` with one line (the main window shows it in the status bar, with where the log is).

Exceptions in plain threads (threading.excepthook) are handled the same way. The signal is queued to the
GUI thread, so the hook can run in any thread.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

from qtpy.QtCore import QObject, Signal

from cryo_savanna.images.cache import cache_root

LOG_NAME = "errors.log"
MAX_LOG_BYTES = 1024 * 1024  # an older log is started again from its last half

_reporter: "ErrorReporter | None" = None


def error_log() -> Path:
    return cache_root() / LOG_NAME


def one_line(exc: BaseException) -> str:
    """"TypeError: … (job_actions.py:812 in clean)": the error and where it was raised."""
    text = f"{type(exc).__name__}: {exc}".splitlines()[0][:300] if str(exc) else type(exc).__name__
    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        last = tb[-1]
        text += f" ({Path(last.filename).name}:{last.lineno} in {last.name})"
    return text


class ErrorReporter(QObject):
    reported = Signal(str, str)  # one line, the log file ("" when it could not be written)

    def __init__(self, parent: QObject | None = None, log: Path | None = None):
        super().__init__(parent)
        self.log = log if log is not None else error_log()
        self._lock = threading.Lock()
        self._previous = None
        self._previous_thread = None

    def install(self) -> None:
        self._previous, sys.excepthook = sys.excepthook, self._hook
        self._previous_thread, threading.excepthook = threading.excepthook, self._thread_hook

    def uninstall(self) -> None:
        if self._previous is not None:
            sys.excepthook, self._previous = self._previous, None
        if self._previous_thread is not None:
            threading.excepthook, self._previous_thread = self._previous_thread, None

    def _hook(self, exc_type, exc, tb) -> None:
        (self._previous or sys.__excepthook__)(exc_type, exc, tb)  # the terminal, as before
        if isinstance(exc, KeyboardInterrupt):
            return
        self.report(exc_type, exc, tb)

    def _thread_hook(self, args) -> None:
        (self._previous_thread or threading.__excepthook__)(args)
        if args.exc_value is not None and not isinstance(args.exc_value, SystemExit):
            self.report(args.exc_type, args.exc_value, args.exc_traceback)

    def report(self, exc_type, exc, tb) -> None:
        """Keep the traceback and tell the window. Never raises: this runs where errors end up."""
        try:
            text = "".join(traceback.format_exception(exc_type, exc, tb))
            written = self._write(text)
            self.reported.emit(one_line(exc), str(self.log) if written else "")
        except Exception:  # noqa: BLE001 - nothing more can be done about an error here
            pass

    def _write(self, text: str) -> bool:
        try:
            with self._lock:
                self.log.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if self.log.stat().st_size > MAX_LOG_BYTES:
                        old = self.log.read_bytes()[-MAX_LOG_BYTES // 2:]
                        self.log.write_bytes(old[old.find(b"\n") + 1:])
                except OSError:
                    pass
                with open(self.log, "a", encoding="utf-8") as fh:
                    fh.write(f"---- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
            return True
        except OSError:
            return False


def install(app: QObject) -> ErrorReporter:
    """One reporter per process (``app``: the QApplication, which owns it)."""
    global _reporter
    if _reporter is None:
        _reporter = ErrorReporter(app)
        _reporter.install()
    return _reporter
