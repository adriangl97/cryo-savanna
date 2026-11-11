# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Viewers of the pytom jobs (jobdefs/pytom_jobs.py).

Scores (template matching): every tomogram with its status (pending, running, done, failed), run time
and pytom's search statistics (σ of the scores, search space, rotations), as the job goes; for the one
chosen, the histogram of its score map (with the score Gaussian noise passes once, the cut-off pytom's
extraction starts from) and the maximum projection of the map along z, where particles show as spots.

Thresholds (extraction): every tomogram's automatic cut-off and the threshold used, the histogram of its
candidates' scores with the threshold as a line to drag (or typed; one threshold for all tomograms, or
one each), and the tomogram (or its score map) with the particles above the threshold. "Save" rewrites
particles.star and thresholds.star in the job (as the job wrote them at its end).

pytom plots: the SVG plots pytom wrote (extraction histograms, tophat filter, ROC).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna import pytom_runner as P
from cryo_savanna.results import pytom as R
from cryo_savanna.results.curves import Curve, Figure
from cryo_savanna.viewers.image_canvas import ImageCanvas
from cryo_savanna.viewers.plots import FigureGrid
from cryo_savanna.viewers.tasks import TaskError, Tasks

STATUS_COLORS = {"done": (0, 158, 115), "running": (0, 114, 178), "failed": (213, 94, 0), "aborted": (213, 94, 0)}


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _item(text, number: float | None = None) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if number is not None:
        it.setData(Qt.UserRole, number)
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.verticalHeader().hide()
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fmt(v, digits=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


# ---------------------------------------------------------------------------- template matching

class PytomScoresView(QWidget):
    COLUMNS = ["Tomogram", "Status", "Time (min)", "σ (search)", "Search space", "Rotations"]

    def __init__(self, project_dir: Path, job: str, parent=None):
        super().__init__(parent)
        self.project_dir, self.job = Path(project_dir), job
        self.summary_rel = job + P.SUMMARY
        self.tasks = Tasks(self)
        self.summary: R.MatchSummary | None = None
        self.stats: dict[str, dict] = {}
        self.current: str | None = None
        self._stamp = None
        self._map_stamp = None

        self.table = _table(self.COLUMNS)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.figures = FigureGrid()
        self.mip = ImageCanvas(None, controls=False)
        self.mip_label = QLabel("Maximum projection of the score map along z")
        self.info = QLabel("Reading…")
        self.info.setWordWrap(True)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.figures)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.mip_label)
        bl.addWidget(self.mip, 1)
        right.addWidget(box)
        right.setSizes([300, 400])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(right)
        split.setSizes([420, 700])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        lay.addWidget(self.info)
        self.load()

    def refresh(self) -> None:
        if _stamp(self.project_dir / self.summary_rel) != self._stamp:
            self.load()
        elif self.current is not None and self._map_path(self.current) is not None and \
                _stamp(self._map_path(self.current)) != self._map_stamp:
            self._show_map()

    def load(self) -> None:
        self._stamp = _stamp(self.project_dir / self.summary_rel)
        project, rel = self.project_dir, self.summary_rel
        if self._stamp is None:
            self.info.setText("Waiting for the job to write pytom_match.star (its tomograms and their progress)…")
            return

        def read():
            s = R.read_match_summary(project, rel)
            stats = {}
            for i, n in enumerate(s.names):
                js = s.value(i, P.COL_JOB)
                if js and s.value(i, P.COL_STATUS) == "done":
                    stats[n] = R.job_stats(project / js)
            return s, stats

        self.tasks.run(read, self._on_loaded, channel="load")

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        self.summary, self.stats = result
        s = self.summary
        keep = self.current
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(s.names))
        for i, n in enumerate(s.names):
            st = self.stats.get(n, {})
            status = s.value(i, P.COL_STATUS)
            secs = R._float(s.value(i, P.COL_SECONDS))
            cells = [_item(n), _item(status),
                     _item(f"{secs / 60:.1f}" if secs else "—", secs / 60 if secs else None),
                     _item(_fmt(st.get("std")), st.get("std")), _item(_fmt(st.get("search_space"), 3),
                                                                      st.get("search_space")),
                     _item(_fmt(st.get("n_rotations")), st.get("n_rotations"))]
            color = STATUS_COLORS.get(status)
            if color:
                cells[1].setForeground(pg.mkColor(*color))
            for c, it in enumerate(cells):
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)
        counts = s.counts()
        order = ["done", "running", "pending", "failed", "aborted"]
        parts = [f"{counts[k]} {k}" for k in order if counts.get(k)]
        template, mask = s.general.get(P.P + "Template", ""), s.general.get(P.P + "Mask", "")
        self.info.setText(f"{len(s.names)} tomograms: " + ", ".join(parts) +
                          (f"   ·   template {template}, mask {mask}" if template else ""))
        row = s.names.index(keep) if keep in s.names else 0
        if s.names:
            self.table.selectRow(row)
            if keep == s.names[row]:
                self._show_figures()

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.summary is None:
            return
        self.current = self.table.item(rows[0].row(), 0).text()
        self._show_map()

    def _map_path(self, name: str) -> Path | None:
        s = self.summary
        if s is None or name not in s.names:
            return None
        rel = s.value(s.names.index(name), P.COL_SCORES)
        return self.project_dir / rel if rel else None

    def _show_map(self) -> None:
        name, path = self.current, self._map_path(self.current)
        self._map_stamp = _stamp(path) if path is not None else None
        i = self.summary.names.index(name)
        status = self.summary.value(i, P.COL_STATUS)
        if path is None or self._map_stamp is None or status != "done":
            self.mip.set_array(np.zeros((2, 2), np.float32))
            self.mip_label.setText(f"{name}: {status or 'no score map'} (the score map comes when it is done)")
            self._score = None
            self._show_figures()
            return
        self.mip_label.setText(f"{name}: reading the score map…")
        self.tasks.run(lambda p=path: R.read_score_map(p), lambda res, n=name: self._on_map(n, res), channel="map")

    def _on_map(self, name: str, result) -> None:
        if isinstance(result, TaskError):
            self.mip_label.setText(f"⚠ {result}")
            return
        if name != self.current:
            return
        self._score = result
        self.mip.sigma.setValue(4.0)
        self.mip.set_array(result.mip)
        x, y, z = result.peak_at
        self.mip_label.setText(f"{name}: maximum projection of the scores along z   ·   peak {result.peak:.4f} at "
                               f"x {x}, y {y}, z {z}   ·   mean {result.mean:.4f}, σ {result.std:.4f}")
        self._show_figures()

    def _show_figures(self) -> None:
        figs = []
        sc = getattr(self, "_score", None)
        name = self.current
        st = self.stats.get(name or "", {})
        if sc is not None:
            x, counts = R.histogram(sc.sample)
            fig = Figure("hist", f"Scores of {name}", "Score (LCC)", "Voxels (log₁₀)",
                         [Curve("scores", x, np.log10(np.maximum(counts, 0.5)))])
            fig.vlines.append((sc.peak, f"peak {sc.peak:.3f}"))
            cut = R.gaussian_cut_off(st.get("std"), st.get("search_space"))
            if cut:
                fig.vlines.append((cut, f"1 false positive: {cut:.3f}"))
                fig.notes.append("the line: the score Gaussian noise (pytom's σ) passes once in the search")
            figs.append(fig)
        if self.summary is not None and self.stats:
            names = [n for n in self.summary.names if self.stats.get(n, {}).get("std")]
            if names:
                idx = np.arange(len(names), dtype=float)
                figs.append(Figure("std", "σ of the scores per tomogram", "Tomogram (in table order)", "σ",
                                   [Curve("σ", idx, np.array([self.stats[n]["std"] for n in names]), symbols=True)]))
        self.figures.set_figures(figs)


# ---------------------------------------------------------------------------- extraction

class PytomThresholdView(QWidget):
    COLUMNS = ["Tomogram", "Automatic cut-off", "Threshold", "Candidates", "Selected"]

    def __init__(self, project_dir: Path, job: str, summary: str = "", diameter: float | None = None,
                 cache=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.summary_rel = Path(project_dir), job, summary
        self.cache, self.diameter = cache, diameter
        self.tasks = Tasks(self)
        self.sel: R.Selection | None = None
        self.current: str | None = None
        self.dirty = False
        self._stamp = None
        self._syncing = False
        self.viewer = None

        self.same = QCheckBox("Same threshold for all tomograms")
        self.same.toggled.connect(self._on_same)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setRange(0, 10)
        self.spin.setSingleStep(0.005)
        self.spin.setPrefix("Threshold ")
        self.spin.valueChanged.connect(self._on_spin)
        self.reset_btn = QPushButton("Automatic")
        self.reset_btn.setToolTip("Back to pytom's automatic cut-off (of each tomogram)")
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn = QPushButton("Save thresholds")
        self.save_btn.setToolTip("Rewrite particles.star (and thresholds.star) of this job with these thresholds")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.same, self.spin, self.reset_btn):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.save_btn)

        self.table = _table(self.COLUMNS)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Score (rlnLCCmax)")
        self.plot.setLabel("left", "Candidates")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for axis in ("bottom", "left"):
            self.plot.getPlotItem().getAxis(axis).enableAutoSIPrefix(False)
        self.bars = pg.BarGraphItem(x=[], height=[], width=1, brush=pg.mkBrush(0, 114, 178, 160))
        self.plot.addItem(self.bars)
        self.line = pg.InfiniteLine(0, angle=90, movable=True, pen=pg.mkPen((213, 94, 0), width=2),
                                    label="threshold", labelOpts={"position": 0.9, "color": (213, 94, 0)})
        self.line.sigPositionChanged.connect(self._on_line)
        self.auto_line = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine),
                                         label="automatic", labelOpts={"position": 0.75, "color": (120, 120, 120)})
        self.plot.addItem(self.line)
        self.plot.addItem(self.auto_line)
        self.info = QLabel("Reading…")
        self.info.setWordWrap(True)

        left = QSplitter(Qt.Vertical)
        left.addWidget(self.table)
        left.addWidget(self.plot)
        left.setSizes([250, 300])
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(left)
        self.right = QWidget()
        QVBoxLayout(self.right).setContentsMargins(0, 0, 0, 0)
        self.split.addWidget(self.right)
        self.split.setSizes([450, 750])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.split, 1)
        lay.addWidget(self.info)
        self.load()

    # -- loading -------------------------------------------------------------------------------
    def _files(self):
        d = self.project_dir / self.job
        return _stamp(d / P.CANDIDATES), _stamp(d / P.THRESHOLDS)

    def refresh(self) -> None:
        if self._files() != self._stamp and not self.dirty:
            self.load()

    def load(self) -> None:
        self._stamp = self._files()
        if self._stamp[0] is None:
            self.info.setText("The candidates come when the job has extracted every tomogram (candidates.star).")
            return
        project, job, summary, diameter = self.project_dir, self.job, self.summary_rel, self.diameter

        def read():
            sel = R.read_selection(project, job)
            d = diameter
            if not d and summary and (project / summary).is_file():  # the matching job's
                s = R.read_match_summary(project, summary)
                for i in range(len(s.names)):
                    d = R.job_stats(project / s.value(i, P.COL_JOB)).get("particle_diameter") \
                        if s.value(i, P.COL_JOB) else None
                    if d:
                        break
            return sel, d

        self.tasks.run(read, self._on_loaded, channel="load")

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        (self.sel, diameter), self.dirty = result, False
        self.diameter = self.diameter or diameter
        self.save_btn.setEnabled(False)
        all_lcc = np.concatenate([v for v in self.sel.lcc.values()]) if self.sel.lcc else np.empty(0)
        if len(all_lcc):
            hi = float(np.nanmax(all_lcc))
            self.spin.setRange(0, max(1.0, hi * 1.5))
        self._fill_table()
        if self.viewer is None and self.summary_rel and (self.project_dir / self.summary_rel).is_file():
            from cryo_savanna.viewers.tomo_viewer import TomogramViewer
            self.viewer = TomogramViewer(self.project_dir, self.summary_rel, picks=self.job + P.CANDIDATES,
                                         diameter=self.diameter, cache=self.cache)
            self.viewer.table.hide()  # the tomograms are chosen in this view's table
            self.viewer.picks_loaded = self._apply_keep
            self.right.layout().addWidget(self.viewer)
        elif self.viewer is not None:
            self.viewer.load()
        names = self.sel.names
        keep = self.current if self.current in names else (names[0] if names else None)
        if keep is not None:
            self.table.selectRow(names.index(keep))
        self._update_info()

    def _fill_table(self) -> None:
        sel = self.sel
        self.table.blockSignals(True)
        self.table.setRowCount(len(sel.names))
        for i, n in enumerate(sel.names):
            cand, chosen = sel.counts(n)
            a, t = sel.auto.get(n), sel.threshold.get(n)
            for c, it in enumerate([_item(n), _item(_fmt(a), a), _item(_fmt(t), t), _item(cand, cand),
                                    _item(chosen, chosen)]):
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)

    def _update_row(self, name: str, row: int | None = None) -> None:
        i = self.sel.names.index(name) if row is None else row
        cand, chosen = self.sel.counts(name)
        t = self.sel.threshold.get(name)
        self.table.item(i, 2).setText(_fmt(t))
        self.table.item(i, 4).setText(str(chosen))

    def _update_info(self) -> None:
        if self.sel is None:
            return
        total = sum(self.sel.counts(n)[1] for n in self.sel.names)
        cands = sum(self.sel.counts(n)[0] for n in self.sel.names)
        state = "   ·   not saved" if self.dirty else ""
        self.info.setText(f"{total} particles selected of {cands} candidates in {len(self.sel.names)} tomograms"
                          f"{state}   ·   {self.job}{P.PARTICLES}")

    # -- the threshold ----------------------------------------------------------------------------
    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.sel is None:
            return
        self.current = self.table.item(rows[0].row(), 0).text()
        self._show_current()
        if self.viewer is not None:
            self.viewer.select_name(self.current)

    def _shown_lcc(self) -> np.ndarray:
        if self.same.isChecked():
            return np.concatenate(list(self.sel.lcc.values())) if self.sel.lcc else np.empty(0)
        return self.sel.lcc.get(self.current, np.empty(0))

    def _show_current(self) -> None:
        lcc = self._shown_lcc()
        lcc = lcc[np.isfinite(lcc)]
        if len(lcc):
            counts, edges = np.histogram(lcc, bins=max(10, min(80, len(lcc) // 4)))
            self.bars.setOpts(x=(edges[:-1] + edges[1:]) / 2, height=counts, width=float(edges[1] - edges[0]))
        else:
            self.bars.setOpts(x=[], height=[], width=1)
        thr = self.sel.threshold.get(self.current)
        auto = self.sel.auto.get(self.current)
        self.plot.setTitle(("All tomograms" if self.same.isChecked() else self.current) +
                           f"   ·   {self.sel.counts(self.current)[1] if not self.same.isChecked() else ''}")
        self._syncing = True
        if thr is not None:
            self.spin.setValue(thr)
            self.line.setValue(thr)
        self.auto_line.setVisible(auto is not None and not self.same.isChecked())
        if auto is not None:
            self.auto_line.setValue(auto)
        self._syncing = False
        self._update_title()

    def _update_title(self) -> None:
        if self.same.isChecked():
            n = sum(self.sel.counts(k)[1] for k in self.sel.names)
            self.plot.setTitle(f"All tomograms: {n} selected")
        else:
            self.plot.setTitle(f"{self.current}: {self.sel.counts(self.current)[1]} of "
                               f"{self.sel.counts(self.current)[0]} selected")

    def _on_line(self) -> None:
        if not self._syncing:
            self.set_threshold(float(self.line.value()))

    def _on_spin(self, v: float) -> None:
        if not self._syncing:
            self.set_threshold(float(v))

    def _on_same(self, on: bool) -> None:
        if self.sel is None:
            return
        if on and self.current is not None and self.sel.threshold.get(self.current) is not None:
            self.set_threshold(self.sel.threshold[self.current])
        self._show_current()

    def set_threshold(self, value: float) -> None:
        """The threshold of the tomogram shown (of all of them with "Same threshold for all")."""
        if self.sel is None or self.current is None:
            return
        rows = {n: i for i, n in enumerate(self.sel.names)}  # once: the names are listed anew each time asked
        names = list(rows) if self.same.isChecked() else [self.current]
        for n in names:
            self.sel.threshold[n] = value
            self._update_row(n, rows.get(n))
        self._syncing = True
        self.spin.setValue(value)
        self.line.setValue(value)
        self._syncing = False
        self.dirty = True
        self.save_btn.setEnabled(self._writable())
        self._update_title()
        self._update_info()
        if self.viewer is not None and self.viewer.picks is not None:
            self._apply_keep(self.viewer.picks)
            self.viewer._on_classes()  # recount and redraw with the new selection

    def _reset(self) -> None:
        if self.sel is None:
            return
        self.same.blockSignals(True)
        self.same.setChecked(False)
        self.same.blockSignals(False)
        for i, n in enumerate(self.sel.names):
            self.sel.threshold[n] = self.sel.auto.get(n)
            self._update_row(n, i)
        self.dirty = True
        self.save_btn.setEnabled(self._writable())
        self._show_current()
        self._update_info()
        if self.viewer is not None and self.viewer.picks is not None:
            self._apply_keep(self.viewer.picks)
            self.viewer._on_classes()

    def _apply_keep(self, picks) -> None:
        if self.sel is not None:
            picks.keep = {n: self.sel.selected(n) for n in self.sel.lcc}

    # -- saving -----------------------------------------------------------------------------------
    def _writable(self) -> bool:
        d = self.project_dir / self.job
        return os.access(d, os.W_OK) and (d / P.THRESHOLDS).is_file()

    def users(self) -> list[str]:
        """Jobs that read this job's particles.star or optimisation set (their results would not match)."""
        from cryo_savanna.project.pipeline import Pipeline
        try:
            p = Pipeline.read(self.project_dir)
            return [c for c in p.children(self.job) if c != self.job]
        except (OSError, ValueError, KeyError):  # no pipeline, or the job is not in it
            return []

    def save(self, ask: bool = True) -> bool:
        if self.sel is None or not self._writable():
            return False
        users = self.users()
        if ask and users:
            text = ("These jobs already used this job's particles, which will change:\n  " + "\n  ".join(
                u.rstrip("/") for u in users) + "\n\nSave the new thresholds anyway?")
            if QMessageBox.question(self, "Save thresholds", text) != QMessageBox.Yes:
                return False
        try:
            counts = R.save_thresholds(self.project_dir, self.job, dict(self.sel.threshold))
        except (OSError, ValueError, P.Usage) as e:
            QMessageBox.warning(self, "Save thresholds", f"Could not write the particles: {e}")
            return False
        self.dirty = False
        self.save_btn.setEnabled(False)
        self._stamp = self._files()
        self.info.setText(f"Saved: {sum(c[1] for c in counts.values())} particles in {self.job}{P.PARTICLES}")
        return True


# ---------------------------------------------------------------------------- SVG plots

class SvgPlotsView(QWidget):
    def __init__(self, project_dir: Path, job: str, folder: str = P.PLOTS_DIR, parent=None):
        super().__init__(parent)
        from qtpy.QtSvgWidgets import QSvgWidget
        self.project_dir, self.job, self.folder = Path(project_dir), job, folder
        self.files: list[str] = []
        self._stamp = None
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        self.svg = QSvgWidget()
        try:
            self.svg.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        except AttributeError:  # Qt < 6.7: stretched to the widget
            pass
        self.info = QLabel()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.svg)
        split.setSizes([260, 800])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        lay.addWidget(self.info)
        self.refresh()

    def refresh(self) -> None:
        st = _stamp(self.project_dir / self.job / self.folder)
        if st == self._stamp:
            return
        self._stamp = st
        keep = self.files[self.list.currentRow()] if 0 <= self.list.currentRow() < len(self.files) else None
        self.files = R.plot_files(self.project_dir, self.job, self.folder)
        self.list.blockSignals(True)
        self.list.clear()
        for f in self.files:
            self.list.addItem(Path(f).stem.replace("_", " "))
        self.list.blockSignals(False)
        if self.files:
            self.list.setCurrentRow(self.files.index(keep) if keep in self.files else 0)
        else:
            self.info.setText("No plots yet (pytom writes them when matplotlib and seaborn are in its environment).")

    def _show(self, row: int) -> None:
        if 0 <= row < len(self.files):
            self.svg.load(str(self.project_dir / self.files[row]))
            try:
                self.svg.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
            except AttributeError:
                pass
            self.info.setText(self.files[row])
