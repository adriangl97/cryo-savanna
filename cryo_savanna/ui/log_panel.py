# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tail of a job's run.out, run.err and note.txt, refreshed while the job writes to them."""

from __future__ import annotations

import codecs
import os
from pathlib import Path

from qtpy.QtGui import QFontDatabase, QTextCursor
from qtpy.QtWidgets import QPlainTextEdit, QTabWidget

from cryo_savanna.project.snapshot import Stamp, stamp

LOG_FILES = ("run.out", "run.err", "note.txt")
MAX_BYTES = 512 * 1024  # only the end of very long logs is shown
ANCHOR_BYTES = 64  # the last bytes shown, looked at again before text is added to them


def collapse_progress(text: str) -> str:
    """Apply carriage returns the way a terminal would: keep what is written after the last \\r."""
    if "\r" not in text:
        return text
    out = []
    for line in text.split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        out.append(line)
    return "\n".join(out)


def read_tail(path: Path, start: int = 0, max_bytes: int = MAX_BYTES) -> tuple[str, int, bool]:
    """Text of ``path`` from ``start`` (or its last ``max_bytes``). Returns (text, end offset, truncated).

    A character the program has only half written yet (UTF-8: "Å", a progress bar's "█") is left for
    the next read, which starts at the returned offset, instead of being shown as replacement characters.
    """
    text, end, truncated, _, _ = read_more(path, start, b"", max_bytes)
    return text, end, truncated


def read_more(path: Path, start: int, anchor: bytes, max_bytes: int = MAX_BYTES) -> tuple[str, int, bool, bytes, bool]:
    """read_tail for a log shown up to ``start``, in one open of the file: (text, end offset, truncated,
    the bytes just before the end, whether the text follows what was shown).

    ``anchor``: the bytes just before ``start`` as they were when read. If they are not there any more,
    the log was written anew (the job run again) and has already grown past what was shown: it is read
    from its beginning, not added to the old text.
    """
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if start:
            f.seek(start - len(anchor))
            if size < start or f.read(len(anchor)) != anchor:
                start, anchor = 0, b""
        begin = max(start, size - max_bytes)
        f.seek(begin)
        data = f.read(size - begin)
    truncated = begin > start
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    nl = data.find(b"\n") if truncated else -1  # truncated: drop the partial first line
    text = decoder.decode(data[nl + 1:], final=False)
    shown = data[:len(data) - len(decoder.getstate()[0])]
    last = ((b"" if truncated else anchor) + shown)[-ANCHOR_BYTES:]
    return text, begin + len(shown), truncated, last, start > 0


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.setMaximumBlockCount(50000)
        self.path: Path | None = None
        self.stamp: Stamp = None
        self.offset = 0
        self._partial = ""  # last line without its newline yet (progress bars)
        self._anchor = b""  # the bytes just before ``offset``, as they were when read

    def load(self, path: Path | None) -> None:
        self.path, self.stamp, self.offset, self._partial, self._anchor = path, None, 0, "", b""
        self.clear()
        self.refresh()

    def refresh(self) -> bool:
        """Read new text if the file changed. Returns whether anything changed."""
        if self.path is None:
            return False
        st = stamp(self.path)
        if st == self.stamp:
            return False
        old = self.stamp
        self.stamp = st
        if st is None:
            self.setPlainText("")
            self.setPlaceholderText(f"{self.path.name} does not exist (yet).")
            self.offset, self._partial, self._anchor = 0, "", b""
            return True
        at_bottom = self.verticalScrollBar().value() >= self.verticalScrollBar().maximum() - 2
        appended = old is not None and st[1] >= self.offset and self.offset > 0
        try:
            text, end, truncated, anchor, appended = read_more(self.path, self.offset if appended else 0,
                                                               self._anchor if appended else b"")
        except OSError as e:
            self.setPlainText(f"Cannot read {self.path}: {e}")
            return True
        if appended and not truncated:
            self._append(text)
        else:
            self._partial = ""
            prefix = f"[showing the last {MAX_BYTES // 1024} KB of {self.path.name}]\n" if truncated else ""
            self.setPlainText("")
            self._append(prefix + text)
        self.offset, self._anchor = end, anchor
        if at_bottom or not appended:
            self.moveCursor(QTextCursor.End)
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        return True

    def _append(self, text: str) -> None:
        # The last block shows the unfinished last line. Re-render it together with the new text,
        # so "\r" progress updates replace it instead of piling up.
        full = self._partial + text
        cut = full.rfind("\n")
        partial = full[cut + 1:]
        if "\r" in partial:  # only the text after the last carriage return can still be seen
            partial = partial[partial.rstrip("\r").rfind("\r") + 1:]
        self._partial = partial
        cur = self.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(collapse_progress(full))


class LogPanel(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.views: dict[str, LogView] = {}
        for name in LOG_FILES:
            v = LogView()
            self.views[name] = v
            self.addTab(v, name)
        self.job_dir: Path | None = None

    def show_job(self, job_dir: Path | None) -> None:
        if job_dir == self.job_dir:
            self.refresh()
            return
        self.job_dir = job_dir
        for name, v in self.views.items():
            v.load(job_dir / name if job_dir else None)
        self._update_labels()
        if job_dir and self.currentIndex() == 0 and (job_dir / "run.err").exists() and \
                (job_dir / "run.err").stat().st_size > 0 and not (job_dir / "run.out").exists():
            self.setCurrentIndex(1)

    def refresh(self) -> None:
        changed = [v.refresh() for v in self.views.values()]  # every log, not only up to the first change
        if any(changed):
            self._update_labels()

    def _update_labels(self) -> None:
        err = self.views["run.err"]
        has_err = err.stamp is not None and err.stamp[1] > 0
        self.setTabText(1, "run.err ⚠" if has_err else "run.err")
