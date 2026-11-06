# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomography viewers: tilt series (per-tilt images and values) and tomograms (Z slabs, side views, picks).

Tilt series: the series of a set, the tilts of the selected one (angle, dose, defocus, CTF,
motion, alignment), the selected tilt image (and its CTF spectrum), and its values against the
tilt angle.

Tomograms: the tomograms of a set with their particle counts; an XY slab (the average of a few
sections) at full resolution, read from the file as the slider moves; XZ and YZ side views from a
binned copy, with the slab's position; particles drawn as circles where the slab cuts them (smaller
away from their centre); denoised and raw side by side.

Both open RELION's own napari viewer (relion_python_tomo_view) for the same files.

Warp/M particles (named by their .tomostar, no RELION tomogram set) come with the tomograms Warp
reconstructed: found by themselves, or in a folder chosen with "Tomograms folder…" (kept for the
session, per project).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.images.mrc import ImageRef, MrcReader, contrast_limits
from cryo_savanna.images.volume import load_volume
from cryo_savanna.results.tables import Table
from cryo_savanna.results.tomo import (
    SERIES_COLUMNS,
    TILT_COLUMNS,
    TiltSeriesSet,
    TomoPicks,
    read_tilt_series_set,
    read_tomo_picks,
    series_summary,
    tilt_figures,
    warp_tomogram_set,
)
from cryo_savanna.viewers.class_filter import ClassFilter
from cryo_savanna.viewers.external import launch_detached, napari_command
from cryo_savanna.viewers.image_canvas import ImageCanvas
from cryo_savanna.viewers.plots import FigureGrid
from cryo_savanna.viewers.star_table import RowFilter, TableModel, make_table_view
from cryo_savanna.viewers.tasks import TaskError, Tasks


_WARP_FOLDERS: dict[str, list[str]] = {}  # project -> folders chosen for Warp's tomograms, this session


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class NapariButton(QPushButton):
    """Starts relion_python_tomo_view; reports it if it stops with an error within a few seconds."""

    def __init__(self, project_dir: Path, report, parent=None):
        super().__init__("Open in napari", parent)
        self.project_dir, self.report = project_dir, report
        self.command: list[str] | None = None
        self.proc = None
        self._watch = QTimer(self)
        self._watch.setSingleShot(True)
        self._watch.setInterval(4000)
        self._watch.timeout.connect(self._check)
        self.clicked.connect(self.launch)
        self.set_command(None)

    def set_command(self, cmd: list[str] | None, missing: str = "") -> None:
        self.command = cmd
        self.setEnabled(cmd is not None)
        self.setToolTip(" ".join(cmd) if cmd else missing or
                        "relion_python_tomo_view not found: load the RELION 5 module (it brings napari)")

    def launch(self) -> None:
        if not self.command:
            return
        try:
            self.proc, self.log = launch_detached(self.command, self.project_dir)
        except OSError as e:
            self.report(f"⚠ napari: {e}")
            return
        self.report("Starting napari… (" + " ".join(Path(c).name if i == 0 else c
                                                   for i, c in enumerate(self.command)) + ")")
        self._watch.start()

    def _check(self) -> None:
        if self.proc is None or self.proc.poll() in (None, 0):
            return
        try:
            tail = self.log.read_text(errors="replace").strip().splitlines()[-3:]
        except OSError:
            tail = []
        self.report(f"⚠ napari stopped (exit {self.proc.returncode}): " + " | ".join(tail))


# ---------------------------------------------------------------------------- tilt series

class TiltSeriesViewer(QWidget):
    def __init__(self, project_dir: Path, star: str, cache=None, tool: str = "", tool_arg: str = "",
                 request_tool=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.star = Path(project_dir), star
        self.tasks = Tasks(self)
        self.set: TiltSeriesSet | None = None
        self.tilts: Table | None = None
        self.series = -1
        self._stamp = None

        self.series_model = TableModel(self)
        self.series_proxy = RowFilter(self)
        self.series_proxy.setSourceModel(self.series_model)
        self.series_view = make_table_view()
        self.series_view.setModel(self.series_proxy)
        self.series_view.selectionModel().currentRowChanged.connect(self._on_series)
        self.tilt_model = TableModel(self)
        self.tilt_proxy = RowFilter(self)
        self.tilt_proxy.setSourceModel(self.tilt_model)
        self.tilt_view = make_table_view()
        self.tilt_view.setModel(self.tilt_proxy)
        self.tilt_view.selectionModel().currentRowChanged.connect(self._on_tilt)
        self.canvas = ImageCanvas(cache, disk_cache=True)
        self.ctf = ImageCanvas(cache, controls=False)
        self.ctf.setMinimumWidth(160)
        self.ctf.hide()
        self.figures = FigureGrid()
        self.napari = NapariButton(self.project_dir, self._report)
        self.napari.set_command(napari_command("tilt-series", star))
        self.info = QLabel("Reading…")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.info, 1)
        self.tool_btn = None
        if tool == "tilts" and request_tool is not None:
            self.tool_btn = QPushButton("Exclude tilt images…")
            self.tool_btn.setToolTip("Make an Exclude tilt images job from this set (as RELION's napari excluder does)")
            self.tool_btn.clicked.connect(lambda: request_tool("tilts", tool_arg))
            top.addWidget(self.tool_btn)
        top.addWidget(self.napari)
        lists = QSplitter(Qt.Vertical)
        lists.addWidget(self.series_view)
        lists.addWidget(self.tilt_view)
        lists.setSizes([160, 400])
        images = QSplitter(Qt.Horizontal)
        images.addWidget(self.canvas)
        images.addWidget(self.ctf)
        images.setStretchFactor(0, 3)
        images.setStretchFactor(1, 1)
        right = QSplitter(Qt.Vertical)
        right.addWidget(images)
        right.addWidget(self.figures)
        right.setSizes([500, 260])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(lists)
        split.addWidget(right)
        split.setSizes([420, 800])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        self.load()

    def _report(self, text: str) -> None:
        self.info.setText(text)

    def load(self) -> None:
        project, star = self.project_dir, self.star
        self._stamp = _stamp(project / star)

        def read():
            ts = read_tilt_series_set(project, star)
            summaries = []
            for i in range(len(ts)):
                try:
                    summaries.append(series_summary(ts.tilts(i)))
                except (OSError, ValueError):
                    summaries.append({})
            return ts, summaries

        self.tasks.run(read, lambda r: self._on_loaded(r), channel="load")

    def refresh(self) -> None:
        if _stamp(self.project_dir / self.star) != self._stamp:
            self.load()

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        ts, summaries = result
        keep = self.set.names[self.series] if self.set is not None and 0 <= self.series < len(self.set) else None
        self.set = ts
        cols = [("rlnTomoName", "Tilt series")]
        model_cols = cols + [(c, t) for c, t in SERIES_COLUMNS if ts.table.has(c)]
        self.series_model.set_table(ts.table, model_cols)
        for key in ("Tilts", "Range (°)", "Defocus (µm)", "Max. res. (Å)"):
            vals = [s.get(key) for s in summaries]
            if any(v is not None for v in vals):
                self.series_model.set_extra("_" + key, key, [v if v is not None else "" for v in vals])
        self.series_view.resizeColumnsToContents()
        n_tilts = sum(int(s.get("Tilts", 0)) for s in summaries)
        self.info.setText(f"{len(ts)} tilt series, {n_tilts} tilt images   ·   {self.star}")
        row = ts.names.index(keep) if keep in ts.names else 0
        if len(ts):
            self.select_series(row)

    def select_series(self, row: int) -> None:
        idx = self.series_proxy.mapFromSource(self.series_model.index(row, 0))
        if idx.isValid():
            self.series_view.setCurrentIndex(idx)

    def _on_series(self, cur, _prev) -> None:
        if not cur.isValid() or self.set is None:
            return
        i = self.series_proxy.mapToSource(cur).row()
        self.series = i
        ts, name = self.set, self.set.names[i]
        self.tasks.run(lambda: ts.tilts(i), lambda t: self._on_tilts(name, t), channel="tilts")

    def _on_tilts(self, name: str, tilts) -> None:
        if isinstance(tilts, TaskError):
            self.info.setText(f"⚠ {tilts}")
            self.tilt_model.set_table(None)
            return
        self.tilts = tilts
        image_col = "rlnMicrographName" if tilts.has("rlnMicrographName") else "rlnMicrographMovieName"
        cols = [(c, t) for c, t in TILT_COLUMNS if tilts.has(c)]
        if tilts.has(image_col):
            cols.append((image_col, "Image" if image_col == "rlnMicrographName" else "Movie"))
        self.tilt_model.set_table(tilts, cols)
        self.tilt_view.resizeColumnsToContents()
        self.ctf.setVisible(tilts.has("rlnCtfImage"))
        self.figures.set_figures(tilt_figures(tilts, name), columns=3)
        # start at the untilted image
        angles = tilts.numeric("rlnTomoNominalStageTiltAngle") if tilts.has("rlnTomoNominalStageTiltAngle") else None
        row = int(np.nanargmin(np.abs(angles))) if angles is not None and len(angles) else 0
        if len(tilts):
            self.select_tilt(row)

    def select_tilt(self, row: int) -> None:
        idx = self.tilt_proxy.mapFromSource(self.tilt_model.index(row, 0))
        if idx.isValid():
            self.tilt_view.setCurrentIndex(idx)
            self.tilt_view.scrollTo(idx)

    def _on_tilt(self, cur, _prev) -> None:
        if not cur.isValid() or self.tilts is None or self.set is None:
            return
        row = self.tilt_proxy.mapToSource(cur).row()
        self.canvas.set_image(self.set.tilt_image(self.tilts, row), keep_view=True)
        if self.ctf.isVisible():
            self.ctf.set_image(self.set.ctf_image(self.tilts, row))


# ---------------------------------------------------------------------------- tomograms

class SideView(pg.GraphicsView):
    """XZ or YZ overview of the tomogram, with the slab's position; click to move the slab."""

    def __init__(self, title: str, on_click, parent=None):
        super().__init__(parent, background="k")
        self.view = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.setCentralItem(self.view)
        self.img = pg.ImageItem()
        self.line = pg.InfiniteLine(angle=0, pen=pg.mkPen((240, 200, 40), width=1))
        self.dots = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(60, 220, 60, 160))
        for it in (self.img, self.line, self.dots):
            self.view.addItem(it)
        self.label = pg.TextItem(title, color=(220, 220, 220), anchor=(0, 0))
        self.view.addItem(self.label, ignoreBounds=True)
        self.on_click = on_click
        self.scene().sigMouseClicked.connect(self._click)
        self.setMinimumHeight(60)

    def set_image(self, img: np.ndarray, width: float, height: float, levels) -> None:
        self.img.setImage(img, levels=levels, autoLevels=False)
        self.img.setRect(QRectF(0, 0, width, height))
        self.view.setRange(QRectF(0, 0, width, height), padding=0.02)
        self.label.setPos(0, 0)

    def _click(self, ev) -> None:
        p = self.view.mapSceneToView(ev.scenePos())
        self.on_click(p.x(), p.y())


class TomogramViewer(QWidget):
    def __init__(self, project_dir: Path, star: str, picks: str = "", diameter: float | None = None, cache=None,
                 parent=None):
        super().__init__(parent)
        self.project_dir, self.star, self.picks_rel = Path(project_dir), star, picks
        self.tasks = Tasks(self)
        self.set: TiltSeriesSet | None = None
        self.picks: TomoPicks | None = None
        self.current = -1
        self.shape: tuple[int, int, int] | None = None  # [z, y, x] of the shown tomogram
        self.voxel = 0.0
        self.z = 0
        self.side_data = None  # binned copy for the side views
        self.levels = (0.0, 1.0)
        self._stamp = None

        self.model = TableModel(self)
        self.proxy = RowFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = make_table_view()
        self.table.setModel(self.proxy)
        self.table.selectionModel().currentRowChanged.connect(self._on_row)

        self.kind = QComboBox()
        self.kind.setToolTip("Which tomogram to show")
        self.kind.currentIndexChanged.connect(lambda _: self._show_current())
        self.compare = QCheckBox("Raw beside it")
        self.compare.setToolTip("Show the raw reconstruction next to the denoised one, at the same place")
        self.compare.toggled.connect(lambda _: self._show_current())
        self.compare.hide()
        self.slab = QSpinBox()
        self.slab.setRange(1, 99)
        self.slab.setValue(5)
        self.slab.setPrefix("Slab ")
        self.slab.setSuffix(" sections")
        self.slab.setToolTip("Average this many sections around the slider's position")
        self.slab.valueChanged.connect(lambda _: self._request_slab())
        self.zslider = QSlider(Qt.Horizontal)
        self.zslider.valueChanged.connect(self._on_z)
        self.zlabel = QLabel()
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0, 20)
        self.sigma.setSingleStep(0.5)
        self.sigma.setValue(3.0)
        self.sigma.setPrefix("Contrast ±")
        self.sigma.setSuffix(" σ")
        self.sigma.valueChanged.connect(lambda _: self._request_slab())
        self.show_picks = QCheckBox("Picks")
        self.show_picks.setChecked(bool(picks))
        self.show_picks.setVisible(bool(picks))
        self.show_picks.toggled.connect(lambda _: self._draw_picks())
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(1, 100000)
        self.diameter.setDecimals(0)
        self.diameter.setValue(diameter or 150)
        self.diameter.setPrefix("⌀ ")
        self.diameter.setSuffix(" Å")
        self.diameter.setVisible(bool(picks))
        self.diameter.valueChanged.connect(lambda _: self._draw_picks())
        self.classes = ClassFilter()  # shown for particles of several classes (Class3D, ...)
        self.classes.changed.connect(self._on_classes)
        self.napari = NapariButton(self.project_dir, self._report)
        self.warp = False  # Warp particles: the tomograms are Warp's, found near the particles
        self.searched: list[Path] = []
        self._missing = False  # the info line says a tomogram was not found
        self.folder_btn = QPushButton("Tomograms folder…")
        self.folder_btn.setToolTip("The folder with Warp's tomograms (its processing folder, or its reconstruction folder)")
        self.folder_btn.clicked.connect(self._choose_folder)
        self.folder_btn.hide()
        self.info = QLabel("Reading…")

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.kind, self.compare, self.slab, self.sigma, self.show_picks, self.diameter, self.classes):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.folder_btn)
        bar.addWidget(self.napari)
        zbar = QHBoxLayout()
        zbar.setContentsMargins(0, 0, 0, 0)
        zbar.addWidget(QLabel("Z"))
        zbar.addWidget(self.zslider, 1)
        zbar.addWidget(self.zlabel)

        self.canvas = ImageCanvas(cache, controls=False)
        self.raw = ImageCanvas(cache, controls=False)
        self.raw.view.setXLink(self.canvas.view)
        self.raw.view.setYLink(self.canvas.view)
        self.raw.hide()
        self.xz = SideView("XZ", lambda x, z: self._side_click(x, z))
        self.yz = SideView("YZ", lambda y, z: self._side_click(None, z))
        sides = QSplitter(Qt.Vertical)
        sides.addWidget(self.xz)
        sides.addWidget(self.yz)
        images = QSplitter(Qt.Horizontal)
        images.addWidget(self.canvas)
        images.addWidget(self.raw)
        main = QSplitter(Qt.Vertical)
        main.addWidget(images)
        main.addWidget(sides)
        main.setSizes([600, 220])
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addLayout(bar)
        cv.addLayout(zbar)
        cv.addWidget(main, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(center)
        split.setSizes([300, 900])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        lay.addWidget(self.info)

        self._slab_timer = QTimer(self)
        self._slab_timer.setSingleShot(True)
        self._slab_timer.setInterval(30)
        self._slab_timer.timeout.connect(self._read_slab)
        self.load()

    def _report(self, text: str) -> None:
        self.info.setText(text)

    # ------------------------------------------------------------------ loading
    def _watched(self) -> tuple:
        return tuple(_stamp(self.project_dir / f) for f in (self.star, self.picks_rel) if f)

    def load(self) -> None:
        project, star, picks = self.project_dir, self.star, self.picks_rel
        self._stamps = self._watched()
        folders = list(_WARP_FOLDERS.get(str(project), []))

        def read():
            pk = read_tomo_picks(project, picks) if picks else None
            if star:
                return read_tilt_series_set(project, star), pk, None
            if pk is not None and pk.pixel_size:  # Warp particles: no tomogram set, Warp's tomograms
                ts, searched = warp_tomogram_set(project, pk, picks, folders)
                return ts, pk, searched
            raise ValueError(f"{picks}: no tomograms for these particles")

        self.tasks.run(read, self._on_loaded, channel="load")

    def refresh(self) -> None:
        if self._watched() != self._stamps:
            self.load()

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        ts, picks, searched = result
        self.warp, self.searched = searched is not None, searched or []
        self.folder_btn.setVisible(self.warp)
        keep = self.set.names[self.current] if self.set is not None and 0 <= self.current < len(self.set) else None
        self.set, self.picks = ts, picks
        self.kind.blockSignals(True)
        current_kind = self.kind.currentData()
        self.kind.clear()
        for col, title in ts.tomogram_kinds():
            self.kind.addItem(title, col)
        i = self.kind.findData(current_kind)
        self.kind.setCurrentIndex(max(0, i))
        self.kind.blockSignals(False)
        kinds = [c for c, _ in ts.tomogram_kinds()]
        self.compare.setVisible("rlnTomoReconstructedTomogramDenoised" in kinds and
                                "rlnTomoReconstructedTomogram" in kinds)
        cols = [("rlnTomoName", "Tomogram")] + [(c, t) for c, t in SERIES_COLUMNS if ts.table.has(c)]
        self.model.set_table(ts.table, cols)
        if picks is not None:
            self.classes.set_counts(picks.classes)
            self._count_picks()
        else:
            self.info.setText(f"{len(ts)} tomograms   ·   {self.star}")
        self.table.resizeColumnsToContents()
        if self.warp:
            self.napari.set_command(None, "RELION's napari viewer needs a RELION tomogram set (rlnTomoName)")
        else:
            cmd = napari_command("particles", self.picks_rel, self.star) if self.picks_rel else \
                napari_command("tomograms", self.star)
            self.napari.set_command(cmd)
        row = ts.names.index(keep) if keep in ts.names else 0
        if len(ts):
            self.select(row)

    def select(self, row: int) -> None:
        idx = self.proxy.mapFromSource(self.model.index(row, 0))
        if idx.isValid():
            self.table.setCurrentIndex(idx)

    def _choose_folder(self) -> None:
        start = str(self.searched[0].parent if self.searched else self.project_dir)
        d = QFileDialog.getExistingDirectory(self, "Folder with Warp's tomograms", start)
        if d:
            self.use_folder(d)

    def use_folder(self, folder: str) -> None:
        """Look for Warp's tomograms in this folder first (for this project, this session)."""
        folders = _WARP_FOLDERS.setdefault(str(self.project_dir), [])
        if folder in folders:
            folders.remove(folder)
        folders.insert(0, folder)
        self.load()

    def _on_row(self, cur, _prev) -> None:
        if cur.isValid():
            self.current = self.proxy.mapToSource(cur).row()
            self._show_current()

    def _ref(self, column: str | None = None) -> ImageRef | None:
        if self.set is None or self.current < 0:
            return None
        ref = self.set.tomogram(self.current, column or self.kind.currentData())
        if ref is None and column is None:  # not of the kind chosen (e.g. one not denoised): another one
            ref = next((r for c, _ in self.set.tomogram_kinds()
                        if (r := self.set.tomogram(self.current, c)) is not None), None)
        return ref

    def _show_current(self) -> None:
        ref = self._ref()
        if ref is None:
            self.tasks.cancel("geometry")
            self.tasks.cancel("slab")
            self.shape, self.side_data = None, None
            for c in (self.canvas, self.raw):
                c.set_image(None)
            for sv in (self.xz, self.yz):
                sv.img.prepareGeometryChange()
                sv.img.clear()
            self._draw_picks()  # none: no tomogram to place them in
            if self.warp and self.set is not None and 0 <= self.current < len(self.set):
                looked = ", ".join(str(d) for d in self.searched) or "nowhere: no processing folder found"
                self._missing = True
                self.info.setText(f"No Warp tomogram for {self.set.names[self.current]} (looked in: {looked}). "
                                  "Choose the folder with “Tomograms folder…”")
            else:
                self.info.setText("This tomogram has not been reconstructed")
            return
        if self._missing:  # back to the totals
            self._missing = False
            if self.picks is not None and self.set is not None:
                self._count_picks()
        compare = self.compare.isVisible() and self.compare.isChecked() and \
            self.kind.currentData() == "rlnTomoReconstructedTomogramDenoised"
        self.raw.setVisible(compare)

        def read():
            with MrcReader(ref.path) as r:
                shape, voxel = (r.nz, r.ny, r.nx), r.voxel
            side = load_volume(ref.path, max_size=256)
            return shape, voxel, side

        self.tasks.run(read, lambda res, rf=ref: self._on_geometry(rf, res), channel="geometry")

    def _on_geometry(self, ref: ImageRef, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        shape, voxel, side = result
        new = self.shape != shape
        self.shape, self.side_data = shape, side
        i = self.current
        self.voxel = voxel or ((self.set.pixel_size(i) or 0) * (self.set.binning(i) or 1))
        self.levels = (side.mean - self.sigma.value() * side.std, side.mean + self.sigma.value() * side.std)
        nz, ny, nx = shape
        self.zslider.blockSignals(True)
        self.zslider.setRange(0, nz - 1)
        if new or not 0 <= self.z < nz:
            self.z = nz // 2
        self.zslider.setValue(self.z)
        self.zslider.blockSignals(False)
        lv = contrast_limits(side.data, self.sigma.value())
        self.xz.set_image(side.data.mean(axis=1), nx, nz, lv)  # [z, x]
        self.yz.set_image(side.data.mean(axis=2), ny, nz, lv)  # [z, y]
        self._read_slab(refit=new)

    # ------------------------------------------------------------------ slab
    def _on_z(self, z: int) -> None:
        self.z = z
        self._request_slab()

    def _side_click(self, x, z) -> None:
        if self.shape is not None and 0 <= z < self.shape[0]:
            self.zslider.setValue(int(z))

    def _request_slab(self) -> None:
        self._slab_timer.start()

    def _read_slab(self, refit: bool = False) -> None:
        if self.shape is None:
            return
        refs = [self._ref()]
        if self.raw.isVisible():
            refs.append(self._ref("rlnTomoReconstructedTomogram"))
        nz = self.shape[0]
        half = self.slab.value() // 2
        z0, z1 = max(0, self.z - half), min(nz, self.z + half + 1)
        self.zlabel.setText(f"{self.z + 1} / {nz}" + (f"  ({z0 + 1}–{z1})" if z1 - z0 > 1 else "") +
                            (f"  {(self.z - nz / 2) * self.voxel:+.0f} Å" if self.voxel else ""))
        for line_view in (self.xz, self.yz):
            line_view.line.setValue(self.z + 0.5)

        def read():
            out = []
            for ref in refs:
                with MrcReader(ref.path) as r:
                    out.append(r.planes(z0, z1).mean(axis=0))
            return out

        self.tasks.run(read, lambda imgs: self._on_slab(imgs, refit), channel="slab")
        self._draw_picks()

    def _on_slab(self, imgs, refit: bool) -> None:
        if isinstance(imgs, TaskError):
            self.info.setText(f"⚠ {imgs}")
            return
        for canvas, img in zip((self.canvas, self.raw), imgs):
            canvas.sigma.setValue(self.sigma.value())
            canvas.set_array(img, keep_view=not refit)

    # ------------------------------------------------------------------ picks
    def _count_picks(self) -> None:
        """The Picks column and the totals, for the classes shown."""
        ts, picks, classes = self.set, self.picks, self.classes.selected()
        self.model.set_extra("_picks", "Picks", [picks.count(n, classes) for n in ts.names], position=1)
        unknown = sum(picks.count(k, classes) for k in picks.by_tomo if k not in ts.names)
        note = f"   ·   {unknown} particles in tomograms not in this set" if unknown else ""
        total = sum(picks.count(k, classes) for k in picks.by_tomo)
        which = self.classes.describe()
        self.info.setText(f"{len(ts)} tomograms, {total} particles" + (f" ({which})" if which else "") +
                          f"   ·   {self.picks_rel}{note}")

    def _on_classes(self) -> None:
        if self.picks is not None and self.set is not None:
            self._count_picks()
        self._draw_picks()

    def _shown_picks(self) -> tuple[np.ndarray, np.ndarray | None]:
        """The current tomogram's particles of the classes shown, in its voxels, and their classes."""
        if self.picks is None or self.shape is None or self.set is None or not self.voxel:
            return np.empty((0, 3)), None
        name = self.set.names[self.current]
        xyz = self.picks.voxels(name, self.voxel, self.shape, self.set.binning(self.current))
        keep = self.picks.in_classes(name, self.classes.selected())
        cls = self.picks.class_numbers(name)
        return xyz[keep], cls[keep] if cls is not None and len(cls) == len(keep) else None

    def picks_voxels(self) -> np.ndarray:
        return self._shown_picks()[0]

    def _draw_picks(self) -> None:
        xyz, cls = self._shown_picks()
        if not self.show_picks.isChecked() or not len(xyz):
            for c in (self.canvas, self.raw):
                c.set_points([], [], 1)
            for s in (self.xz, self.yz):
                s.dots.prepareGeometryChange()  # see ImageCanvas.set_points
                s.dots.clear()
            return
        r = self.diameter.value() / 2 / self.voxel
        dz = np.abs(xyz[:, 2] - (self.z + 0.5))
        cut = dz < r  # the slab passes through these particles: circles as big as the cut
        size = 2 * np.sqrt(np.maximum(r * r - dz[cut] ** 2, 0))
        colors = self.classes.colors(cls)
        cut_colors = [c for c, k in zip(colors, cut) if k] if isinstance(colors, list) else colors
        for c in (self.canvas, self.raw):
            c.set_points(xyz[cut, 0], xyz[cut, 1], np.maximum(size, 2), cut_colors)
        if isinstance(colors, list):
            brushes = {c: pg.mkBrush(*c, 160) for c in set(colors)}
            brush = [brushes[c] for c in colors]
        else:
            brush = pg.mkBrush(*colors, 160)
        for s in (self.xz, self.yz):
            s.dots.prepareGeometryChange()
        self.xz.dots.setData(x=xyz[:, 0], y=xyz[:, 2], brush=brush)
        self.yz.dots.setData(x=xyz[:, 1], y=xyz[:, 2], brush=brush)
        self.cut_count = int(cut.sum())
