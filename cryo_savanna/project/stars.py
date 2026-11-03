# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Starred jobs: marked in the list and the tree, and found with the "Starred" filter.

RELION has no such notion, so the stars are cryo-savanna's own: ``.cryo-savanna/stars.json`` in the
project (shared by everyone who opens it, like main_inputs.json), ``{"jobs": ["Refine3D/job011/", ...]}``.
With ``persist=False`` (--readonly) they only last for the session. The file is read again when it
changes, so a star given in another window shows up too.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from cryo_savanna.project.main_inputs import GUI_DIR
from cryo_savanna.project.shared_json import read_json, update_json

FILE = "stars.json"


def _jobs(data: dict) -> set[str]:
    jobs = data.get("jobs")
    return {str(j) for j in jobs} if isinstance(jobs, list) else set()


class Stars:
    def __init__(self, project_dir: str | os.PathLike, persist: bool = True):
        self.path = Path(project_dir) / GUI_DIR / FILE
        self.persist = persist
        self._lock = threading.Lock()  # the loader thread reads, the GUI thread changes
        self._stamp: tuple | None = None
        self._jobs: set[str] = set()
        self._session: dict[str, bool] = {}  # changes not written (read-only)

    def _file_stamp(self) -> tuple | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def stamp(self) -> tuple:
        """Changes when the stars change (the file, or a star of this session)."""
        with self._lock:
            return self._file_stamp(), tuple(sorted(self._session.items()))

    def starred(self) -> set[str]:
        with self._lock:
            st = self._file_stamp()
            if st != self._stamp:
                self._stamp = st
                self._jobs = _jobs(read_json(self.path))
            out = set(self._jobs)
            for job, on in self._session.items():
                (out.add if on else out.discard)(job)
            return out

    def set(self, job: str, on: bool) -> None:
        """Star or unstar ``job``. Other sessions write the file too: the change is made to the file as
        it is now (shared_json), keeping theirs."""
        with self._lock:
            if not self.persist:
                self._session[job] = on
                return

            def change(data: dict) -> dict:
                jobs = _jobs(data)
                (jobs.add if on else jobs.discard)(job)
                return {"jobs": sorted(jobs)}

            self._jobs = _jobs(update_json(self.path, change))
            self._stamp = self._file_stamp()
