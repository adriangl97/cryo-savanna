# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The main inputs chosen in cryo-savanna: which input places a job in the job tree, where the default
(project/lineage.py: optimisation set, else particle set, else the job window's order) is not wanted.

RELION has no such notion, so the choices are cryo-savanna's own: ``.cryo-savanna/main_inputs.json``
in the project (shared by everyone who opens it), ``{"jobs": {"Class3D/job012/": "<input node>"}}``.
With ``persist=False`` (--readonly) they only last for the session. The file is read again when it
changes, so a choice made in another window shows up too.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from cryo_savanna.project.shared_json import update_json

GUI_DIR = ".cryo-savanna"
FILE = "main_inputs.json"


class MainInputs:
    def __init__(self, project_dir: str | os.PathLike, persist: bool = True):
        self.path = Path(project_dir) / GUI_DIR / FILE
        self.persist = persist
        self._lock = threading.Lock()  # the loader thread reads, the GUI thread changes
        self._stamp: tuple | None = None
        self._jobs: dict[str, str] = {}
        self._session: dict[str, str | None] = {}  # changes not written (read-only), None = back to default

    def _file_stamp(self) -> tuple | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def stamp(self) -> tuple:
        """Changes when the choices change (the file, or a choice of this session)."""
        with self._lock:
            return self._file_stamp(), tuple(sorted(self._session.items(), key=lambda kv: kv[0]))

    def choices(self) -> dict[str, str]:
        """job -> chosen main input node."""
        with self._lock:
            st = self._file_stamp()
            if st != self._stamp:
                self._stamp = st
                try:
                    data = json.loads(self.path.read_text())
                    self._jobs = {str(k): str(v) for k, v in data.get("jobs", {}).items()}
                except (OSError, ValueError, AttributeError):
                    self._jobs = {}
            out = dict(self._jobs)
            for job, node in self._session.items():
                if node is None:
                    out.pop(job, None)
                else:
                    out[job] = node
            return out

    def set(self, job: str, node: str | None) -> None:
        """Make ``node`` the main input of ``job``; None goes back to the default. Other sessions write
        the file too: the change is made to the file as it is now (shared_json), keeping theirs."""
        with self._lock:
            if not self.persist:
                self._session[job] = node
                return

            def change(data: dict) -> dict:
                jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
                current = {str(k): str(v) for k, v in jobs.items()}
                if node is None:
                    current.pop(job, None)
                else:
                    current[job] = node
                return {"jobs": dict(sorted(current.items()))}

            self._jobs = update_json(self.path, change)["jobs"]
            self._stamp = self._file_stamp()
