# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's own versions of RELION's interactive selection tools.

ClassSelector replaces ``relion_display --gui`` in an interactive Select job (click classes to
select them, as there); TiltExcluder replaces ``relion_python_tomo_exclude_tilt_images`` (napari)
in Exclude tilt images. Both only collect the choice: saving it (the job, its RELION files, its
status) is done by JobActions with project/selection.py, so the result is a normal finished RELION
job.

The windows are built from two panels, ClassSelection and TiltExclusion. A finished job shows its
saved choice in the same panels, read-only (SelectionView, the job's Selection tab); its "Change
selection…" button opens the window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QItemSelectionModel, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.selection import (
    ClassInput,
    SelectionError,
    read_backup_selection,
    read_class_input,
    read_tilt_exclusion,
    read_tilt_input,
    selected_numbers,
)
from cryo_savanna.results.classes import read_model_classes
from cryo_savanna.results.tomo import tilt_figures
from cryo_savanna.viewers.gallery import GalleryItem, GalleryView
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.tomo_viewer import TiltSeriesViewer


class _ToolWindow(QMainWindow):
    """A window with the tool and Save / Cancel; ``saved`` carries what was chosen."""

    saved = Signal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.save_btn = QPushButton("Save selection")
        self.save_btn.setDefault(True)
        self.save_btn.setToolTip("Write the job's files (Ctrl+S)")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        self.message = QLabel()
        self.message.setWordWrap(True)
        QShortcut(QKeySequence.Save, self, activated=lambda: self.save_btn.click())
        self.busy = False

    def footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self.message, 1)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.save_btn)
        return row

    def set_busy(self, busy: bool, text: str = "") -> None:
        """While the job is being written: no second save."""
        self.busy = busy
        self.save_btn.setEnabled(not busy)
        if text:
            self.message.setText(text)


# ---------------------------------------------------------------------------- classes

class ClassSelection(QWidget):
    """The classes of one iteration as a gallery, the selected ones highlighted.

    ``editable``: a click selects or unselects a class (as in relion_display), with All / None /
    Invert, recenter and regroup. Otherwise the panel only shows a selection (set_selected).
    ``loaded`` carries "" once the classes are read, or the error.
    """

    SORTS = [("distribution", "Class distribution"), ("resolution", "Estimated resolution"),
             ("particles", "Particles"), ("score", "Class score")]
    loaded = Signal(str)

    def __init__(self, project_dir: Path, source: str, selected: list[int] | None = None, recenter: bool = False,
                 regroup: int = 0, editable: bool = True, parent=None):
        super().__init__(parent)
        self.project_dir, self.source, self.editable = Path(project_dir), source, editable
        self.initial = set(selected or [])
        self.tasks = Tasks(self)
        self.input: ClassInput | None = None
        self.counts: dict[int, int] = {}
        self.gallery = GalleryView(self.SORTS)
        # a click toggles, as in relion_display; read-only, clicks change nothing (the selection is only shown)
        self.gallery.list.setSelectionMode(QAbstractItemView.MultiSelection if editable
                                           else QAbstractItemView.NoSelection)
        self.gallery.describe = self._describe
        if editable:
            for text, fn in (("All", self._all), ("None", self._none), ("Invert", self._invert)):
                b = QPushButton(text)
                b.clicked.connect(fn)
                self.gallery.bar.insertWidget(0, b)
        self.recenter = QCheckBox("Recenter class averages")
        self.recenter.setToolTip("Move each selected class average to its centre of mass (relion_display --recenter)")
        self.recenter.setChecked(recenter)
        self.regroup = QSpinBox()
        self.regroup.setRange(0, 9999)
        self.regroup.setValue(regroup)
        self.regroup.setSpecialValueText("no regrouping")
        self.regroup.setPrefix("Regroup into ")
        self.regroup.setSuffix(" groups")
        self.regroup.setToolTip("Regroup the selected particles by their scale correction (relion_display --regroup)")
        self.recenter.setEnabled(editable)
        self.regroup.setEnabled(editable)
        opts = QHBoxLayout()
        opts.addWidget(self.recenter)
        opts.addWidget(self.regroup)
        opts.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.gallery, 1)
        lay.addLayout(opts)
        project = self.project_dir

        def read():
            inp = read_class_input(project, source)
            cs = read_model_classes(project, inp.model, project / inp.data)
            return inp, cs

        self.tasks.run(read, self._on_loaded)

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.gallery.status.setText(f"⚠ {result}")
            self.loaded.emit(str(result))
            return
        inp, cs = result
        self.input = inp
        self.recenter.setVisible(inp.writes_class_averages)
        items = []
        for c in cs.classes:
            self.counts[c.number] = c.particles or 0
            label = c.label() + (f"\n{c.particles} ptcls" if c.particles is not None else "")
            items.append(GalleryItem(c.ref, label, f"Class {c.number}",
                                     {"distribution": c.distribution, "resolution": c.resolution,
                                      "particles": c.particles, "score": c.score}, payload=c))
        self.gallery.set_items(items)
        self.set_selected(self.initial)
        self.loaded.emit("")

    def selected(self) -> list[int]:
        return sorted(i.payload.number for i in self.gallery.selected())

    def set_selected(self, numbers) -> None:
        numbers = set(numbers)
        sm = self.gallery.list.selectionModel()
        sm.clearSelection()
        for r, it in enumerate(self.gallery.model.items):
            if it.payload.number in numbers:
                sm.select(self.gallery.model.index(r), QItemSelectionModel.Select)

    def _all(self) -> None:
        self.gallery.list.selectAll()

    def _none(self) -> None:
        self.gallery.list.clearSelection()

    def _invert(self) -> None:
        chosen = set(self.selected())
        self.set_selected(i.payload.number for i in self.gallery.model.items if i.payload.number not in chosen)

    def _describe(self, sel) -> str:
        total = sum(self.counts.values())
        parts = sum(self.counts.get(i.payload.number, 0) for i in sel)
        share = f" ({100 * parts / total:.0f}%)" if total else ""
        return f"{len(sel)} of {len(self.gallery.model.items)} classes selected, {parts} particles{share}"


class ClassSelector(_ToolWindow):
    """Pick classes of one iteration; ``saved`` emits (selected class numbers, recenter, regroup)."""

    def __init__(self, project_dir: Path, source: str, selected: list[int] | None = None, recenter: bool = False,
                 regroup: int = 0, title: str = "", parent=None):
        super().__init__(title or f"Select classes — {source}", parent)
        self.panel = ClassSelection(project_dir, source, selected, recenter, regroup)
        self.gallery, self.recenter, self.regroup = self.panel.gallery, self.panel.recenter, self.panel.regroup
        self.selected, self.set_selected = self.panel.selected, self.panel.set_selected
        self.panel.loaded.connect(self._on_loaded)
        self.gallery.selection_changed.connect(self._update)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(self.panel, 1)
        lay.addLayout(self.footer())
        self.setCentralWidget(central)
        self.save_btn.clicked.connect(self._save)
        self.resize(1100, 800)
        self.message.setText("Reading the classes…")

    @property
    def input(self) -> ClassInput | None:
        return self.panel.input

    def _on_loaded(self, error: str) -> None:
        if error:
            self.message.setText(f"⚠ {error}")
            self.save_btn.setEnabled(False)
            return
        self._update()

    def _update(self) -> None:
        if not self.busy and self.input is not None:
            self.message.setText("Click classes to select them; Save writes the Select job's files.")

    def _save(self) -> None:
        if self.input is None or self.busy:
            return
        sel = self.selected()
        if not sel:
            self.message.setText("Select at least one class.")
            return
        self.saved.emit((sel, self.recenter.isChecked() and self.input.writes_class_averages, self.regroup.value()))


# ---------------------------------------------------------------------------- tilt images

KEEP = "_keep"


class TiltExclusion(QWidget):
    """The tilt-series viewer with a Keep column: excluded tilts red in the table and as lines in the plots.

    ``editable``: keys as in RELION's napari excluder (↑/↓ next image, Enter or Space keeps/excludes
    it, [ and ] previous/next tilt series) and buttons. Otherwise the panel only shows ``kept``.
    """

    def __init__(self, project_dir: Path, source: str, kept: dict[str, list[bool]] | None = None,
                 editable: bool = True, parent=None):
        super().__init__(parent)
        self.project_dir, self.source, self.editable = Path(project_dir), source, editable
        self.kept: dict[str, np.ndarray] = {k: np.array(v, bool) for k, v in (kept or {}).items()}
        self.viewer = TiltSeriesViewer(project_dir, source)
        self.viewer.napari.hide()
        self.viewer.tilt_view.setSortingEnabled(True)
        self.bar = QHBoxLayout()
        if editable:
            self.toggle_btn = QPushButton("Keep / exclude (Enter)")
            self.toggle_btn.clicked.connect(self.toggle_current)
            self.keep_all = QPushButton("Keep all of this series")
            self.keep_all.clicked.connect(self._keep_all)
            self.bar.addWidget(self.toggle_btn)
            self.bar.addWidget(self.keep_all)
            for key, fn in (("Return", self.toggle_current), ("Enter", self.toggle_current),
                            ("Space", self.toggle_current), ("]", lambda: self._step_series(1)),
                            ("[", lambda: self._step_series(-1))):
                QShortcut(QKeySequence(key), self, activated=fn)
            self.viewer.tilt_view.doubleClicked.connect(lambda _: self.toggle_current())
        self.bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.bar)
        lay.addWidget(self.viewer, 1)
        # the viewer shows a series' tilts: add the keep column and sort by tilt angle, as the excluder does
        original = self.viewer._on_tilts

        def on_tilts(name, tilts):
            original(name, tilts)
            if not isinstance(tilts, TaskError):
                self._decorate(name, tilts)
        self.viewer._on_tilts = on_tilts
        original_loaded = self.viewer._on_loaded

        def on_loaded(result):
            original_loaded(result)
            self._update_series_counts()
        self.viewer._on_loaded = on_loaded

    def _name(self) -> str | None:
        v = self.viewer
        return v.set.names[v.series] if v.set is not None and 0 <= v.series < len(v.set) else None

    def mask(self, name: str, n: int) -> np.ndarray:
        m = self.kept.get(name)
        if m is None or len(m) != n:
            m = self.kept[name] = np.ones(n, bool)
        return m

    def _decorate(self, name: str, tilts) -> None:
        m = self.mask(name, len(tilts))
        self.viewer.tilt_model.set_extra(KEEP, "Keep", ["✓" if k else "✗" for k in m], position=0)
        col = 1 + [c for c, _ in self.viewer.tilt_model.columns].index("rlnTomoNominalStageTiltAngle") \
            if tilts.has("rlnTomoNominalStageTiltAngle") else 0
        self.viewer.tilt_view.sortByColumn(col, Qt.AscendingOrder)
        self._paint_rows()
        self._update_series_counts()

    def _paint_rows(self) -> None:
        """Excluded tilts in red (the model's background role)."""
        model = self.viewer.tilt_model
        name = self._name()
        if name is None or model.table is None:
            return
        m = self.mask(name, len(model.table))
        model.row_brushes = {i: QBrush(QColor(250, 200, 200)) for i in np.nonzero(~m)[0]}
        model.dataChanged.emit(model.index(0, 0), model.index(model.rowCount() - 1, model.columnCount() - 1))
        # and in the plots: a red line at each excluded tilt angle
        t = model.table
        if t.has("rlnTomoNominalStageTiltAngle"):
            angles = t.numeric("rlnTomoNominalStageTiltAngle")[~m]
            figs = tilt_figures(t, name)
            for f in figs:
                f.vlines = [(float(a), "") for a in angles]
                if len(angles):
                    f.notes = [f"{len(angles)} excluded"]
            self.viewer.figures.set_figures(figs, columns=3)

    def _update_series_counts(self) -> None:
        v = self.viewer
        if v.set is None:
            return
        vals = []
        for name in v.set.names:
            m = self.kept.get(name)
            vals.append("all" if m is None else f"{int(m.sum())} of {len(m)}")
        v.series_model.set_extra("_kept", "Kept", vals, position=1)

    def toggle_current(self) -> None:
        v = self.viewer
        idx = v.tilt_view.currentIndex()
        name = self._name()
        if not self.editable or not idx.isValid() or name is None or v.tilt_model.table is None:
            return
        row = v.tilt_proxy.mapToSource(idx).row()
        m = self.mask(name, len(v.tilt_model.table))
        m[row] = not m[row]
        v.tilt_model.set_extra(KEEP, "Keep", ["✓" if k else "✗" for k in m])
        self._paint_rows()
        self._update_series_counts()
        nxt = v.tilt_proxy.index(idx.row() + 1, 0)  # on to the next image, as napari's excluder
        if nxt.isValid():
            v.tilt_view.setCurrentIndex(nxt)

    def _keep_all(self) -> None:
        name = self._name()
        if self.editable and name is not None and name in self.kept:
            self.kept[name][:] = True
            self.viewer.tilt_model.set_extra(KEEP, "Keep", ["✓"] * len(self.kept[name]))
            self._paint_rows()
            self._update_series_counts()

    def _step_series(self, step: int) -> None:
        v = self.viewer
        if v.set is None:
            return
        cur = v.series_view.currentIndex()
        nxt = v.series_proxy.index(cur.row() + step, 0) if cur.isValid() else v.series_proxy.index(0, 0)
        if nxt.isValid():
            v.series_view.setCurrentIndex(nxt)

    def result(self) -> dict[str, list[bool]]:
        names = self.viewer.set.names if self.viewer.set is not None else list(self.kept)
        return {n: [bool(x) for x in self.kept[n]] for n in names if n in self.kept}


class TiltExcluder(_ToolWindow):
    """Tick off tilt images; ``saved`` emits {tilt series: [kept?] in the input's row order}."""

    def __init__(self, project_dir: Path, source: str, kept: dict[str, list[bool]] | None = None, title: str = "",
                 parent=None):
        super().__init__(title or f"Exclude tilt images — {source}", parent)
        self.panel = TiltExclusion(project_dir, source, kept)
        self.viewer, self.kept = self.panel.viewer, self.panel.kept
        self.toggle_current, self._step_series, self._keep_all = (self.panel.toggle_current, self.panel._step_series,
                                                                  self.panel._keep_all)
        self.result = self.panel.result
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(self.panel, 1)
        lay.addLayout(self.footer())
        self.setCentralWidget(central)
        self.save_btn.clicked.connect(self._save)
        self.resize(1300, 850)
        self.message.setText("↑/↓ next image, Enter keeps or excludes it, [ and ] previous/next tilt series.")

    def _save(self) -> None:
        if self.busy or self.viewer.set is None:
            return
        res = self.result()
        excluded = sum(len(v) - sum(v) for v in res.values())
        # a series not looked at keeps all its tilts: only excluding every tilt of every series is refused
        if res and all(not any(res.get(n, [True])) for n in self.viewer.set.names):
            self.message.setText("Every tilt image is excluded.")
            return
        self.message.setText(f"Saving: {excluded} tilt images excluded…")
        self.saved.emit(res)


# ---------------------------------------------------------------------------- a job's saved choice

def _stamp(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class SelectionView(QWidget):
    """What an interactive Select or Exclude tilt images job chose, in the selection window's panel but
    read-only. "Change selection…" opens the window (``request_tool("change", job)``); the view
    follows when the job's files are rewritten."""

    FILES = {"classes": "backup_selection.star", "tilts": "selected_tilt_series.star"}

    def __init__(self, project_dir: Path, job: str, kind: str, request_tool=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.kind = Path(project_dir), job, kind
        self.request_tool = request_tool
        self.panel: QWidget | None = None
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.change_btn = QPushButton("Change selection…")
        self.change_btn.setToolTip("Open the selection window to change which "
                                   + ("classes" if kind == "classes" else "tilt images") + " this job keeps")
        self.change_btn.setVisible(request_tool is not None)
        self.change_btn.clicked.connect(lambda: self.request_tool("change", self.job))
        top = QHBoxLayout()
        top.addWidget(self.note, 1)
        top.addWidget(self.change_btn)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.addLayout(top)
        self._stamp = None
        self._build()

    @property
    def _file(self) -> Path:
        return self.project_dir / self.job / self.FILES[self.kind]

    def _build(self) -> None:
        self._stamp = _stamp(self._file)
        if self.panel is not None:
            self.panel.setParent(None)
            self.panel.deleteLater()
            self.panel = None
        try:
            o = JobStar.read(self.project_dir / self.job / "job.star").options
        except (OSError, ValueError, KeyError) as e:
            self.note.setText(f"⚠ Cannot read {self.job}job.star: {e}")
            return
        if self.kind == "classes":
            regroup = int(float(o.get("nr_groups", "0") or 0)) if o.get("do_regroup") == "Yes" else 0
            self.panel = ClassSelection(self.project_dir, o.get("fn_model", ""), recenter=o.get("do_recenter") == "Yes",
                                        regroup=regroup, editable=False)
            self.panel.loaded.connect(self._on_classes)
        else:
            source = o.get("in_tiltseries", "")
            try:
                kept = read_tilt_exclusion(read_tilt_input(self.project_dir, source), self.job)
            except (OSError, ValueError, KeyError, SelectionError) as e:
                self.note.setText(f"⚠ {e}")
                return
            self.panel = TiltExclusion(self.project_dir, source, kept, editable=False)
            self._note(kept is not None, "Tilt images kept by this job: excluded ones are red (✗).")
        self.lay.addWidget(self.panel, 1)

    def _on_classes(self, error: str) -> None:
        if error:
            self.note.setText(f"⚠ {error}")
            return
        n = len(self.panel.input.classes)
        try:
            flags = read_backup_selection(self.project_dir, self.job, n)
        except (OSError, ValueError) as e:
            self.note.setText(f"⚠ {self.job}backup_selection.star: {e}")
            return
        self.panel.set_selected(selected_numbers(self.panel.input, flags))
        self._note(flags is not None, "Classes selected by this job are highlighted.")

    def _note(self, saved: bool, text: str) -> None:
        self.note.setText(text if saved else "No selection saved in this job yet.")

    def refresh(self) -> None:
        """Called on every poll while the tab is shown: rebuilt after the selection was changed."""
        if _stamp(self._file) != self._stamp:
            self._build()
