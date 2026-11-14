# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Micrographs (or movies): a sortable list with the per-micrograph values, and the selected one large.

Optional side panels: the CTF power spectrum with its fit (CtfFind), the motion track (MotionCorr),
and particle picks drawn as circles (AutoPick, ManualPick, or the particles of an Extract job on
their micrographs), with a figure-of-merit threshold and, for particles of several classes, the
classes to show.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.images.cache import ThumbCache
from cryo_savanna.results.curves import motion_figure
from cryo_savanna.results.micrographs import (
    MicrographSet,
    PickSet,
    micrographs_from_names,
    read_micrographs,
    read_picks,
)
from cryo_savanna.viewers.class_filter import ClassFilter
from cryo_savanna.viewers.image_canvas import ImageCanvas
from cryo_savanna.viewers.plots import FigureGrid
from cryo_savanna.viewers.star_table import RowFilter, TableModel, make_table_view
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.stamps import stamp as _stamp


class MicrographViewer(QWidget):
    current_changed = Signal(int)

    def __init__(self, project_dir: Path, star: str = "", cache: ThumbCache | None = None, ctf: bool = False,
                 motion: bool = False, picks: str = "", diameter: float | None = None, box: str = "", parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.star, self.picks_star = star, picks
        self.want_ctf, self.want_motion = ctf, motion
        self.diameter_A, self.box = diameter, box
        self.mics: MicrographSet | None = None
        self.pickset: PickSet | None = None
        self.tasks = Tasks(self)
        self.current = -1
        self._stamps: tuple = ()
        self._wanted_row: int | None = None

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter micrographs")
        self.filter.setClearButtonEnabled(True)
        self.count = QLabel()
        self.model = TableModel(self)
        self.proxy = RowFilter(self)
        self.proxy.setSourceModel(self.model)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = make_table_view()
        self.table.setModel(self.proxy)
        self.table.selectionModel().currentRowChanged.connect(self._on_row)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(self.filter, 1)
        top.addWidget(self.count)
        lv.addLayout(top)
        lv.addWidget(self.table, 1)

        self.canvas = ImageCanvas(cache, disk_cache=True)
        # picks controls
        self.show_picks = QCheckBox("Picks")
        self.show_picks.setChecked(True)
        self.show_picks.toggled.connect(lambda _: self._draw_picks())
        self.circle = QDoubleSpinBox()
        self.circle.setRange(1, 10000)
        self.circle.setDecimals(0)
        self.circle.setPrefix("⌀ ")
        self.circle.setSuffix(" px")
        self.circle.setToolTip("Circle diameter in micrograph pixels")
        self.circle.valueChanged.connect(lambda _: self._draw_picks())
        self.min_fom = QDoubleSpinBox()
        self.min_fom.setRange(-1e6, 1e6)
        self.min_fom.setDecimals(3)
        self.min_fom.setSingleStep(0.05)
        self.min_fom.setValue(-1e6)
        self.min_fom.setPrefix("FOM ≥ ")
        self.min_fom.setSpecialValueText("all FOM")
        self.min_fom.setToolTip("Hide picks with a lower figure of merit (rlnAutopickFigureOfMerit)")
        self.min_fom.valueChanged.connect(lambda _: self._draw_picks())
        self.classes = ClassFilter()
        self.classes.changed.connect(self._on_classes)
        self.pick_info = QLabel()
        pb = QHBoxLayout()
        pb.setContentsMargins(0, 0, 0, 0)
        for w in (self.show_picks, self.circle, self.min_fom, self.classes, self.pick_info):
            pb.addWidget(w)
        pb.addStretch(1)
        self.pick_bar = QWidget()
        self.pick_bar.setLayout(pb)
        self.pick_bar.setVisible(bool(picks))
        self._picks = None

        # side panels: CTF spectrum and motion track
        self.ctf_canvas = ImageCanvas(cache, controls=False)
        self.ctf_canvas.setMinimumWidth(200)
        self.motion_plot = FigureGrid()
        self.motion_plot.setMinimumWidth(280)
        side = QSplitter(Qt.Horizontal)
        side.addWidget(self.canvas)
        side.addWidget(self.ctf_canvas)
        side.addWidget(self.motion_plot)
        self.ctf_canvas.setVisible(ctf)
        self.motion_plot.setVisible(motion)
        for i, stretch in enumerate((3, 1, 1)):
            side.setStretchFactor(i, stretch)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.pick_bar)
        rv.addWidget(side, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([360, 900])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        self.status = QLabel("Reading…")
        lay.addWidget(self.status)
        self.load()

    # ------------------------------------------------------------------ loading
    def _watched(self) -> tuple:
        return tuple(_stamp(self.project_dir / f) for f in (self.star, self.picks_star) if f)

    def load(self) -> None:
        project, star, picks = self.project_dir, self.star, self.picks_star
        self._stamps = self._watched()

        def read():
            ps = read_picks(project, picks) if picks else None
            if star and (project / star).is_file():
                ms = read_micrographs(project, star)
            elif ps is not None:
                ms = micrographs_from_names(project, ps.micrographs)
            else:
                raise ValueError(f"{star or picks}: not found")
            return ms, ps

        self.tasks.run(read, self._on_loaded, channel="load")

    def refresh(self) -> None:
        """Read the lists again when their files changed (a running job adds micrographs)."""
        if self._watched() != self._stamps:
            self.load()

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.status.setText(f"⚠ {result}")
            return
        ms, ps = result
        name = self.mics.names[self.current] if self.mics is not None and 0 <= self.current < len(self.mics) else None
        self.mics, self.pickset = ms, ps
        if ps is not None:
            # only the micrographs that were picked on (a coordinate list may skip some)
            picked = set(ps.micrographs)
            if picked and not picked >= set(ms.names):
                keep = [i for i, n in enumerate(ms.names) if n in picked]
                if keep:
                    ms.table.data = {c: [v[i] for i in keep] for c, v in ms.table.data.items()}
                    ms.table._numeric.clear()
            self.classes.set_counts(ps.classes)
            self._init_circle()
        cols = [(ms.name_col, "Movie" if ms.kind == "movies" else "Micrograph")] + ms.metric_columns()
        self.model.set_table(ms.table, cols)
        self.table.resizeColumnsToContents()
        self.count.setText(f"{len(ms)} {ms.kind}")
        self.status.setText(f"{ms.source or ''}" + (f"   ·   pixel size {ms.pixel_size:g} Å" if ms.pixel_size else ""))
        if ps is not None:
            self._count_picks()
        row = ms.names.index(name) if name in ms.names else 0
        if self._wanted_row is not None and self._wanted_row < len(ms):
            row, self._wanted_row = self._wanted_row, None
        if len(ms):
            self.select(row)

    def _init_circle(self) -> None:
        if self.circle.value() > 1:
            return
        px = (self.mics.pixel_size if self.mics else None) or (self.pickset.pixel_size if self.pickset else None)
        d = None
        if self.diameter_A and px:
            d = self.diameter_A / px
        elif self.box:
            try:
                d = float(self.box)
            except ValueError:
                d = None
        self.circle.setValue(d or 100)

    def _count_picks(self) -> None:
        ps, names, classes = self.pickset, list(self.mics.names), self.classes.selected()
        which = self.classes.describe()

        def count():
            return [ps.count(n, classes) for n in names]

        def done(counts) -> None:
            if isinstance(counts, TaskError) or self.mics is None or list(self.mics.names) != names:
                return
            self.model.set_extra("_picks", "Picks", counts)
            total = sum(c or 0 for c in counts)
            self.count.setText(f"{len(names)} micrographs, {total} picks" + (f" ({which})" if which else ""))

        self.tasks.run(count, done, channel="counts")

    # ------------------------------------------------------------------ selection
    def select_when_loaded(self, row: int) -> None:
        """Select a micrograph now, or once the list is read (the view was just built)."""
        if self.mics is None:
            self._wanted_row = row
        else:
            self.select(row)

    def select(self, row: int) -> None:
        idx = self.proxy.mapFromSource(self.model.index(row, 0))
        if idx.isValid():
            self.table.setCurrentIndex(idx)
            self.table.scrollTo(idx)

    def _on_row(self, cur, _prev) -> None:
        if not cur.isValid() or self.mics is None:
            return
        row = self.proxy.mapToSource(cur).row()
        self.current = row
        ms = self.mics
        self.canvas.set_image(ms.image(row), keep_view=True)
        nxt = self.proxy.index(cur.row() + 1, 0)
        if nxt.isValid():
            self.canvas.prefetch(ms.image(self.proxy.mapToSource(nxt).row()))
        if self.want_ctf:
            ref = ms.ctf_image(row)
            self.ctf_canvas.set_image(ref)
        if self.want_motion:
            path, px = ms.motion_file(row), ms.pixel_size
            if path is not None:
                self.tasks.run(lambda: motion_figure(path, px), self._on_motion, channel="motion")
        if self.pickset is not None:
            ps, name = self.pickset, ms.names[row]
            self._picks = None
            self.canvas.set_points([], [], 1)
            self.tasks.run(lambda: ps.picks(name), self._on_picks, channel="picks")
        self.current_changed.emit(row)

    def _on_motion(self, fig) -> None:
        if isinstance(fig, TaskError) or fig is None:
            self.motion_plot.set_figures([])
            return
        self.motion_plot.set_figures([fig], columns=1)
        if self.motion_plot.plots:
            vb = self.motion_plot.plots[0].getViewBox()
            vb.setAspectLocked(True)
            vb.autoRange()

    def _on_picks(self, picks) -> None:
        if isinstance(picks, TaskError):
            self.pick_info.setText(f"⚠ {picks}")
            return
        self._picks = picks
        has_fom = picks.fom is not None and len(picks)
        self.min_fom.setEnabled(bool(has_fom))
        self._draw_picks()

    def _on_classes(self) -> None:
        """Other classes chosen: the picks on this micrograph, and the counts in the list."""
        self._draw_picks()
        if self.pickset is not None and self.mics is not None:
            self._count_picks()

    def _draw_picks(self) -> None:
        p = self._picks
        if p is None:
            return
        if not self.show_picks.isChecked():
            self.canvas.set_points([], [], 1)
            self.pick_info.setText(f"{len(p)} picks (hidden)")
            return
        in_classes = p.in_classes(self.classes.selected())
        keep = in_classes.copy()
        if p.fom is not None and self.min_fom.value() > self.min_fom.minimum():
            keep &= p.fom >= self.min_fom.value()
        colors = self.classes.colors(p.cls[keep] if p.cls is not None else None)
        self.canvas.set_points(p.x[keep], p.y[keep], self.circle.value(), colors)
        shown, below = int(keep.sum()), int(in_classes.sum() - keep.sum())
        which = self.classes.describe()
        notes = [n for n in (which, f"{below} below the FOM threshold" if below else "") if n]
        self.pick_info.setText(f"{shown} picks" + (f" ({'; '.join(notes)})" if notes else ""))

