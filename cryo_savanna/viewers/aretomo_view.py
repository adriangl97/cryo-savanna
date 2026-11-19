# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The Tomograms view of an AreTomo3 job (jobdefs/aretomo_jobs.py): the tomogram viewer, with the alignment
thickness (-AlignZ) of the tilt series shown above it, to change for that tilt series alone.

"Save" writes the thicknesses into the job's options (job.star, tomo_align_z); continuing the job aligns
and reconstructs again only the tilt series whose thickness is not the one they were aligned with (the
others keep their alignment and tomogram). The table gets the thickness AreTomo3 measured and the one each
tilt series was aligned with.
"""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

from cryo_savanna import aretomo_runner as A
from cryo_savanna.jobdefs import aretomo_jobs as J
from cryo_savanna.stamps import stamp as _stamp
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.tomo_viewer import TomogramViewer


def _px(v: int | None) -> str:
    return "" if v is None else ("measured" if v == 0 else str(v))


class AretomoTomogramsView(QWidget):
    def __init__(self, project_dir: Path, job: str, star: str, cache=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job = Path(project_dir), job
        self.tasks = Tasks(self)
        self.settings: J.AlignZSettings | None = None
        self.dirty = False  # thicknesses not saved
        self._stamp: object = None
        self._syncing = False

        self.viewer = TomogramViewer(self.project_dir, star, cache=cache)
        self.viewer.set_loaded = lambda _ts: self._fill_columns()
        self.viewer.table.selectionModel().currentRowChanged.connect(lambda *_: self._show_current())

        self.title = QLabel("Alignment thickness:")
        self.spin = QSpinBox()
        self.spin.setRange(0, 100000)
        self.spin.setSingleStep(50)
        self.spin.setSpecialValueText("Measured")
        self.spin.setSuffix(" px")
        self.spin.setToolTip("-AlignZ of this tilt series: the height of the volume AreTomo3 aligns it with, in "
                             "pixels of the tilt series (Measured: AreTomo3's own measure of the sample thickness)")
        self.spin.valueChanged.connect(self._on_spin)
        self.job_btn = QPushButton("Job's")
        self.job_btn.setToolTip("Back to the job's alignment thickness for this tilt series")
        self.job_btn.clicked.connect(self._to_job)
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Write the alignment thicknesses into the job's options (job.star), for when the "
                                 "job is continued: only the tilt series whose thickness changed are aligned and "
                                 "reconstructed again")
        self.save_btn.clicked.connect(self.save)
        self.note = QLabel()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.title, self.spin, self.job_btn, self.save_btn):
            bar.addWidget(w)
        bar.addWidget(self.note, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.viewer, 1)
        self._set_enabled(False)
        self.load()

    # -- loading -------------------------------------------------------------------------------
    def _files(self):
        d = self.project_dir / self.job
        return _stamp(d / "job.star"), _stamp(d / A.SUMMARY)

    def refresh(self) -> None:
        self.viewer.refresh()
        if self._files() != self._stamp and not self.dirty:
            self.load()

    def load(self) -> None:
        self._stamp = self._files()
        project, job = self.project_dir, self.job
        self.tasks.run(lambda: J.read_align_z(project, job), self._on_loaded, channel="load")

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.settings = None
            self._set_enabled(False)
            self.note.setText(f"⚠ {result}")
            return
        self.settings, self.dirty = result, False
        self._fill_columns()
        self._show_current()

    # -- showing -------------------------------------------------------------------------------
    def _names(self) -> list[str]:
        return list(self.viewer.set.names) if self.viewer.set is not None else []

    def _current(self) -> str | None:
        v = self.viewer
        return v.set.names[v.current] if v.set is not None and 0 <= v.current < len(v.set) else None

    def _fill_columns(self) -> None:
        st, ts = self.settings, self.viewer.set
        if st is None or ts is None:
            return
        measured = []
        for i, n in enumerate(ts.names):
            px, thick = ts.pixel_size(i), st.measured.get(n)
            measured.append(f"{thick / px:.0f}" if thick and px else "")
        model = self.viewer.model
        model.set_extra("_measured", "Measured thickness (px)", measured, position=1)
        if not st.use_input:
            model.set_extra("_align_z", "Alignment thickness (px)", [self._cell(n) for n in ts.names], position=2)

    def _cell(self, name: str) -> str:
        st = self.settings
        if st is None:
            return ""
        used, want = st.used.get(name), st.wanted(name)
        if name in st.used and used != want:
            return f"{_px(used) or '?'} → {_px(want)}"
        return _px(want)

    def _set_enabled(self, on: bool) -> None:
        for w in (self.spin, self.job_btn):
            w.setEnabled(on)
        self.save_btn.setEnabled(on and self.dirty and self._writable())

    def _show_current(self) -> None:
        st, name = self.settings, self._current()
        if st is None or name is None:
            self._set_enabled(False)
            return
        if st.use_input:
            self.title.setText("Alignment thickness:")
            self._set_enabled(False)
            self.note.setText("This job reconstructs with the alignment of its input: there is no alignment "
                              "thickness to change")
            return
        self.title.setText(f"Alignment thickness of {name}:")
        self._syncing = True
        self.spin.setValue(st.wanted(name))
        self._syncing = False
        self._set_enabled(True)
        self.job_btn.setEnabled(name in st.per_series)
        self._update_note()

    def _update_note(self) -> None:
        st = self.settings
        if st is None or st.use_input:
            return
        name = self._current()
        parts = []
        if name is not None and name in st.used:
            parts.append(f"aligned with {_px(st.used[name]) or 'an unknown thickness'}")
        parts.append(f"job's: {_px(st.job_z)}")
        redo = st.to_align(self._names())
        if redo:
            parts.append(("continue the job to align again: " if not self.dirty else "after Save, a continue "
                          "aligns again: ") + ", ".join(redo))
        self.note.setText("   ·   ".join(parts))

    # -- changing ------------------------------------------------------------------------------
    def set_align_z(self, value: int, name: str | None = None) -> None:
        """The alignment thickness (px, 0: measured) of a tilt series (default: the one shown), not saved yet."""
        st, name = self.settings, name or self._current()
        if st is None or name is None or st.use_input:
            return
        if value == st.job_z:
            st.per_series.pop(name, None)
        else:
            st.per_series[name] = int(value)
        self.dirty = True
        if name == self._current() and self.spin.value() != value:
            self._syncing = True
            self.spin.setValue(value)
            self._syncing = False
        self.job_btn.setEnabled(name in st.per_series)
        self.save_btn.setEnabled(self._writable())
        self._fill_columns()
        self._update_note()

    def _on_spin(self, v: int) -> None:
        if not self._syncing:
            self.set_align_z(v)

    def _to_job(self) -> None:
        if self.settings is not None:
            self.set_align_z(self.settings.job_z)

    # -- saving ---------------------------------------------------------------------------------
    def _writable(self) -> bool:
        d = self.project_dir / self.job
        return os.access(d, os.W_OK) and (d / "job.star").is_file()

    def save(self) -> bool:
        st = self.settings
        if st is None or not self._writable():
            return False
        try:
            J.save_align_z(self.project_dir, self.job, dict(st.per_series))
        except (OSError, ValueError, KeyError, A.Usage) as e:
            QMessageBox.warning(self, "Save", f"Could not write the job's options ({self.job}job.star): {e}")
            return False
        self.dirty = False
        self._stamp = self._files()
        self.save_btn.setEnabled(False)
        self._update_note()
        redo = st.to_align(self._names())
        self.note.setText("Saved. " + (f"Continue the job to align and reconstruct again: {', '.join(redo)}"
                                       if redo else "No tilt series to align again"))
        return True
