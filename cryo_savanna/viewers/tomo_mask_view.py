# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The Masks view of a tomogram mask job (jobdefs/tomomask_jobs.py): the masks drawn on the tomograms.

The tomograms of the job are shown as in the tomogram viewer (XY slab, XZ and YZ side views). The mask of
the tomogram chosen in the table is

- a polygon on the XY slab: drag its corners (or all of it), click on an edge to add a corner, right-click
  a corner to remove it;
- in Z, either all of it ("All Z": nothing is cut in Z, whatever the tomogram's thickness), or the slab
  between a bottom and a top plane ("Between two planes"), drawn as lines in the side views: drag a line
  to move the plane, drag an end of it to tilt the plane (XZ: along x, YZ: along y).

What lies outside the mask at the slab's Z is shaded. "Same mask for all tomograms" gives every tomogram
the mask of the one shown, and then each edit goes to all of them; switched off, each tomogram can be
edited on its own from there. Corners and planes are kept from the tomogram's centre, so a mask given to
tomograms of another size stays centred. "Save masks" rewrites the job's masks (mask_runner.save_masks:
only the changed ones; identical masks become links to one file).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QPointF, QRectF, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna import mask_runner as M
from cryo_savanna import pytom_runner as P
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.tomo_viewer import TomogramViewer
from cryo_savanna.stamps import stamp as _stamp

Z_TITLES = {M.Z_ALL: "All Z", M.Z_SLAB: "Between two planes"}
POLY_PEN = (255, 170, 0)
PLANE_PENS = {"bottom": (0, 200, 255), "top": (255, 80, 200)}
SHADE = (220, 60, 30, 90)  # outside the mask
HELP = ("Drag the polygon's corners (click an edge to add one, right-click a corner to remove it). Between two "
        "planes: drag the lines in XZ and YZ (blue: lower Z, pink: upper Z) to move a plane, their ends to tilt it.")


class _Tomograms(TomogramViewer):
    """The tomogram viewer, telling the mask view when a tomogram set or a tomogram was read."""

    after_loaded = None
    after_geometry = None

    def _on_loaded(self, result) -> None:
        super()._on_loaded(result)
        if self.after_loaded is not None and not isinstance(result, TaskError):
            self.after_loaded()

    def _on_geometry(self, ref, result) -> None:
        super()._on_geometry(ref, result)
        if self.after_geometry is not None and not isinstance(result, TaskError):
            self.after_geometry()


class TomoMaskView(QWidget):
    COLUMNS = ["Tomogram", "Size", "Z", "Corners", ""]

    def __init__(self, project_dir: Path, job: str, cache=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.cache = Path(project_dir), job, cache
        self.job_dir = self.project_dir / job
        self.tasks = Tasks(self)
        self.data: dict | None = None  # masks.json
        self.saved: dict[str, str] = {}  # geometry key of each mask as last saved
        self.current: str | None = None
        self.dirty = False
        self.saving = False
        self._stamp = None
        self._syncing = False
        self.viewer: _Tomograms | None = None

        self.same = QCheckBox("Same mask for all tomograms")
        self.same.setToolTip("Give every tomogram the mask of the one shown; while on, each edit goes to all of "
                             "them. Switch it off to edit tomograms one by one from there.")
        self.same.toggled.connect(self._on_same)
        self.zmode = QComboBox()
        for mode, title in Z_TITLES.items():
            self.zmode.addItem(title, mode)
        self.zmode.setToolTip("All Z: the polygon goes through the whole tomogram (thicker tomograms are not cut). "
                              "Between two planes: only the slab between the bottom and top planes.")
        self.zmode.currentIndexChanged.connect(self._on_zmode)
        self.shade = QCheckBox("Shade outside")
        self.shade.setChecked(True)
        self.shade.toggled.connect(lambda _: self._request_shade())
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Back to the whole tomogram (with flat planes at a quarter of its thickness)")
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn = QPushButton("Save masks")
        self.save_btn.setToolTip("Rewrite the job's masks (masks/<rlnTomoName>_mask.mrc) that changed")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.same, QLabel("Z:"), self.zmode, self.shade, self.reset_btn):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.save_btn)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.itemSelectionChanged.connect(self._on_row)
        self.right = QWidget()
        QVBoxLayout(self.right).setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.right)
        split.setSizes([320, 900])
        self.help = QLabel(HELP)
        self.help.setWordWrap(True)
        self.info = QLabel("Reading…")
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        lay.addWidget(self.help)
        lay.addWidget(self.info)

        # the shapes, added to the viewer's views once it is there
        self.poly = pg.PolyLineROI([], closed=True, pen=pg.mkPen(POLY_PEN, width=2),
                                   handlePen=pg.mkPen(POLY_PEN), hoverPen=pg.mkPen((255, 230, 120), width=2))
        self.poly.setZValue(10)
        self.poly.sigRegionChanged.connect(self._on_live)
        self.poly.sigRegionChangeFinished.connect(self._on_poly)
        self.lines: dict[tuple[str, str], pg.LineSegmentROI] = {}
        for view in ("xz", "yz"):
            for side, color in PLANE_PENS.items():
                roi = pg.LineSegmentROI([(0, 0), (1, 0)], pen=pg.mkPen(color, width=2),
                                        hoverPen=pg.mkPen(color, width=3))
                roi.setZValue(10)
                roi.sigRegionChanged.connect(self._on_live)
                roi.sigRegionChangeFinished.connect(lambda _r, v=view, s=side: self._on_line(v, s))
                self.lines[(view, side)] = roi
        self.shade_item = pg.ImageItem()
        self.shade_item.setZValue(1.5)  # over the slab (and its detail), under the picks
        self._live: dict | None = None  # the geometry while a shape is being dragged
        self._shade_timer = QTimer(self)
        self._shade_timer.setSingleShot(True)
        self._shade_timer.setInterval(40)
        self._shade_timer.timeout.connect(self._draw_shade)
        self._pending_lines: set[tuple[str, str]] = set()
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(0)
        self._commit_timer.timeout.connect(self._do_commit)
        self.load()

    # ------------------------------------------------------------------ loading
    def refresh(self) -> None:
        if not self.dirty and not self.saving and _stamp(self.job_dir / M.GEOMETRY) != self._stamp:
            self.load()
        elif self.viewer is not None:
            self.viewer.refresh()

    def load(self) -> None:
        self._stamp = _stamp(self.job_dir / M.GEOMETRY)
        if self._stamp is None:
            self.info.setText("The masks come when the job has run (masks.json in the job).")
            return
        job_dir = str(self.job_dir)
        self.tasks.run(lambda: M.read_geometry(job_dir), self._on_loaded, channel="load")

    def _on_loaded(self, data) -> None:
        if isinstance(data, TaskError):
            self.info.setText(f"⚠ {data}")
            return
        self.data, self.dirty = data, False
        self.saved = {n: (data["files"].get(n) or {}).get("key", "") for n in data["tomograms"]}
        self.save_btn.setEnabled(False)
        self._fill_table()
        star = data.get("in_tomograms", "")
        if self.viewer is None and star and (self.project_dir / star).is_file():
            self.viewer = _Tomograms(self.project_dir, star, cache=self.cache)
            self.viewer.table.hide()  # the tomograms are chosen in this view's table
            self.viewer.after_loaded = self._on_viewer_loaded
            self.viewer.after_geometry = self._place_shapes
            self.viewer.zslider.valueChanged.connect(lambda _: self._request_shade())
            self.viewer.canvas.view.addItem(self.poly)
            self.viewer.canvas.view.addItem(self.shade_item)
            for (view, _side), roi in self.lines.items():
                getattr(self.viewer, view).view.addItem(roi)
            self.right.layout().addWidget(self.viewer)
        elif self.viewer is None:
            self.info.setText(f"⚠ The tomograms of the masks are not there: {star}")
        names = self.names()
        keep = self.current if self.current in names else (names[0] if names else None)
        if keep is not None:
            self.table.selectRow(names.index(keep))
        self._update_info()

    def _on_viewer_loaded(self) -> None:
        col = P.TOMOGRAM_KINDS.get((self.data or {}).get("tomo_kind", ""), "")
        i = self.viewer.kind.findData(col)
        if i >= 0 and i != self.viewer.kind.currentIndex():
            self.viewer.kind.setCurrentIndex(i)
        if self.current is not None:
            self.viewer.select_name(self.current)

    def names(self) -> list[str]:
        return list(self.data["tomograms"]) if self.data else []

    # ------------------------------------------------------------------ the table
    def _fill_table(self) -> None:
        names = self.names()
        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        for i, n in enumerate(names):
            for c in range(len(self.COLUMNS)):
                self.table.setItem(i, c, QTableWidgetItem())
            self._update_row(n, i)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _update_row(self, name: str, row: int | None = None) -> None:
        if row is None:
            row = self.names().index(name)
        t = self.data["tomograms"][name]
        g = t["geometry"]
        changed = M.geometry_key(t["shape"], g) != self.saved.get(name)
        values = [name, "×".join(str(v) for v in t["shape"]), "planes" if g.get("z_mode") == M.Z_SLAB else "all",
                  str(len(g.get("polygon", []))), "edited" if changed else ""]
        for c, v in enumerate(values):
            self.table.item(row, c).setText(v)

    def _update_info(self) -> None:
        if self.data is None:
            return
        n = len(self.data["tomograms"])
        edited = sum(1 for k, t in self.data["tomograms"].items()
                     if M.geometry_key(t["shape"], t["geometry"]) != self.saved.get(k))
        state = f"   ·   {edited} edited, not saved" if edited else ""
        shape = self._shape_note()
        self.info.setText(f"{n} tomogram masks{state}   ·   {self.job}{M.MASK_DIR}/<rlnTomoName>{M.SUFFIX}{shape}")

    def _shape_note(self) -> str:
        if self.viewer is None or self.viewer.shape is None or self.current is None:
            return ""
        nx, ny, nz = self.data["tomograms"][self.current]["shape"]
        if tuple(self.viewer.shape) != (nz, ny, nx):
            z, y, x = self.viewer.shape
            return f"   ·   ⚠ the tomogram shown is {x}×{y}×{z}, its mask {nx}×{ny}×{nz}"
        return ""

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.data is None:
            return
        self.current = self.table.item(rows[0].row(), 0).text()
        self._show_geometry()
        if self.viewer is not None:
            self.viewer.select_name(self.current)

    # ------------------------------------------------------------------ geometry <-> shapes
    def geometry(self, name: str | None = None) -> dict | None:
        name = name or self.current
        if self.data is None or name not in self.data["tomograms"]:
            return None
        return self.data["tomograms"][name]["geometry"]

    def shape(self, name: str | None = None) -> tuple[int, int, int] | None:
        name = name or self.current
        if self.data is None or name not in self.data["tomograms"]:
            return None
        return tuple(self.data["tomograms"][name]["shape"])

    def _show_geometry(self) -> None:
        self._commit_timer.stop()  # an edit not taken yet was of the tomogram shown before
        self._pending_lines.clear()
        g = self.geometry()
        if g is None:
            return
        self._syncing = True
        self.zmode.setCurrentIndex(max(0, self.zmode.findData(g.get("z_mode", M.Z_ALL))))
        self._syncing = False
        self._place_shapes()

    def _place_shapes(self) -> None:
        """The polygon and the plane lines of the current tomogram's mask, in the viewer's voxels."""
        g, shape = self.geometry(), self.shape()
        if g is None or shape is None or self._syncing:
            return
        nx, ny, nz = shape
        cx, cy, cz = nx / 2, ny / 2, nz / 2
        self._syncing = True
        self.zmode.setCurrentIndex(max(0, self.zmode.findData(g.get("z_mode", M.Z_ALL))))
        self.poly.setPos(0, 0)
        self.poly.setPoints([(x + cx, y + cy) for x, y in g["polygon"]], closed=True)
        slab = g.get("z_mode") == M.Z_SLAB
        for side in ("bottom", "top"):
            z0, ax, ay = g[side]
            self._set_line(self.lines[("xz", side)], (0, cz + z0 - ax * cx), (nx, cz + z0 + ax * (nx - cx)))
            self._set_line(self.lines[("yz", side)], (0, cz + z0 - ay * cy), (ny, cz + z0 + ay * (ny - cy)))
        for roi in self.lines.values():
            roi.setVisible(slab)
        self._syncing = False
        self._live = None
        self._update_info()
        self._request_shade()

    @staticmethod
    def _set_line(roi: pg.LineSegmentROI, p1, p2) -> None:
        roi.setPos(0, 0, update=False, finish=False)
        h1, h2 = roi.getHandles()
        roi.movePoint(h1, QPointF(*p1), finish=False)
        roi.movePoint(h2, QPointF(*p2), finish=False)

    @staticmethod
    def _ends(roi: pg.LineSegmentROI) -> list[tuple[float, float]]:
        pts = [roi.mapToParent(h.pos()) for h in roi.getHandles()]
        return [(p.x(), p.y()) for p in pts]

    def _read_polygon(self, shape) -> list[list[float]]:
        nx, ny, _ = shape
        pts = [self.poly.mapToParent(h.pos()) for h in self.poly.getHandles()]
        return [[round(p.x() - nx / 2, 2), round(p.y() - ny / 2, 2)] for p in pts]

    def _read_line(self, g: dict, view: str, side: str, shape) -> None:
        """The plane ``side`` of ``g`` from its line in the XZ or YZ view (z0 and the slope it shows)."""
        nx, ny, nz = shape
        (a1, z1), (a2, z2) = self._ends(self.lines[(view, side)])
        if abs(a2 - a1) < 1e-6:
            return
        c = nx / 2 if view == "xz" else ny / 2
        slope = (z2 - z1) / (a2 - a1)
        z0 = z1 + slope * (c - a1) - nz / 2
        old = list(g[side])
        g[side] = [z0, slope, old[2]] if view == "xz" else [z0, old[1], slope]

    def _from_shapes(self, view: str | None = None, side: str | None = None) -> dict | None:
        """The current geometry with what the shapes show now (the polygon; one plane line if given)."""
        g, shape = self.geometry(), self.shape()
        if g is None:
            return None
        g = copy.deepcopy(g)
        if len(self.poly.getHandles()) >= 3:
            g["polygon"] = self._read_polygon(shape)
        if view is not None:
            self._read_line(g, view, side, shape)
        return g

    # ------------------------------------------------------------------ edits
    def _on_live(self, *_args) -> None:
        if self._syncing or self.current is None:
            return
        roi = self.sender()
        view_side = next((k for k, r in self.lines.items() if r is roi), (None, None))
        self._live = self._from_shapes(*view_side)
        self._request_shade()

    def _on_poly(self, *_args) -> None:
        if not self._syncing:
            self._commit()

    def _on_line(self, view: str, side: str) -> None:
        if not self._syncing:
            self._commit((view, side))

    def _commit(self, line: tuple[str, str] | None = None) -> None:
        """An edit of a shape is taken at the next turn of the event loop: pyqtgraph reports a change as
        finished also in the middle of one (a corner being removed, the points being set), and the shapes
        are placed again from the geometry, which must not happen inside it."""
        if line is not None:
            self._pending_lines.add(line)
        self._commit_timer.start()

    def _do_commit(self) -> None:
        lines, self._pending_lines = self._pending_lines, set()
        g = self._from_shapes()
        if g is None:
            return
        shape = self.shape()
        for view, side in sorted(lines):
            self._read_line(g, view, side, shape)
        self.set_geometry(g)

    def flush(self) -> None:
        """Take a pending edit now (tests)."""
        if self._commit_timer.isActive():
            self._commit_timer.stop()
            self._do_commit()

    def _on_zmode(self, _i: int) -> None:
        if self._syncing or self.current is None:
            return
        g = copy.deepcopy(self.geometry())
        g["z_mode"] = self.zmode.currentData()
        self.set_geometry(g)

    def _targets(self) -> list[str]:
        return self.names() if self.same.isChecked() else [self.current]

    def set_geometry(self, g: dict | None, names: list[str] | None = None) -> None:
        """The mask of the tomogram shown (of all of them with "Same mask for all tomograms")."""
        if g is None or self.data is None or self.current is None:
            return
        g = M.clean_geometry(g)
        rows = {n: i for i, n in enumerate(self.names())}
        for n in names or self._targets():
            self.data["tomograms"][n]["geometry"] = copy.deepcopy(g)
            self._update_row(n, rows[n])
        for c in (2, 3, 4):
            self.table.resizeColumnToContents(c)
        self._changed()
        self._place_shapes()

    def _changed(self) -> None:
        self.dirty = any(M.geometry_key(t["shape"], t["geometry"]) != self.saved.get(n)
                         for n, t in self.data["tomograms"].items())
        self.save_btn.setEnabled(self.dirty and self._writable())
        self._update_info()

    def _on_same(self, on: bool) -> None:
        if on and self.geometry() is not None:
            self.set_geometry(self.geometry(), self.names())

    def _reset(self) -> None:
        shape = self.shape()
        if shape is None:
            return
        g = M.default_geometry(shape, self.zmode.currentData())
        self.set_geometry(g)

    # ------------------------------------------------------------------ the shading
    def _request_shade(self) -> None:
        self._shade_timer.start()

    def _draw_shade(self) -> None:
        shape, viewer = self.shape(), self.viewer
        g = self._live or self.geometry()
        if not self.shade.isChecked() or g is None or viewer is None or viewer.shape is None:
            self.shade_item.hide()
            return
        nx, ny, nz = shape
        z = min(max(viewer.z, 0), nz - 1)
        inside = np.frombuffer(M.Raster(shape, g).plane(z), np.uint8).reshape(ny, nx)
        rgba = np.zeros((ny, nx, 4), np.uint8)
        rgba[inside == 0] = SHADE
        self.shade_item.setImage(rgba, autoLevels=False)
        self.shade_item.setRect(QRectF(0, 0, nx, ny))
        self.shade_item.show()

    def inside_fraction(self) -> float:
        """The part of the current section inside the mask (tests)."""
        shape, g = self.shape(), self.geometry()
        nx, ny, nz = shape
        z = min(max(self.viewer.z if self.viewer is not None else nz // 2, 0), nz - 1)
        return float(np.frombuffer(M.Raster(shape, g).plane(z), np.uint8).mean())

    # ------------------------------------------------------------------ saving
    def _writable(self) -> bool:
        d = self.job_dir
        finished = any((d / f).exists() for f in (P.SUCCESS, P.FAILURE, P.ABORTED))
        return finished and os.access(d, os.W_OK) and (d / M.GEOMETRY).is_file()

    def users(self) -> list[str]:
        """Jobs that read this job's masks (their results would not match the new ones)."""
        from cryo_savanna.project.pipeline import Pipeline
        try:
            p = Pipeline.read(self.project_dir)
            return [c for c in p.children(self.job) if c != self.job]
        except (OSError, ValueError, KeyError):
            return []

    def save(self, ask: bool = True) -> bool:
        if self.data is None or not self.dirty or self.saving:
            return False
        if not self._writable():
            QMessageBox.warning(self, "Save masks", "The masks can be saved once the job has finished running.")
            return False
        users = self.users()
        if ask and users:
            text = ("These jobs already used this job's masks, which will change:\n  " + "\n  ".join(
                u.rstrip("/") for u in users) + "\n\nSave the new masks anyway?")
            if QMessageBox.question(self, "Save masks", text) != QMessageBox.Yes:
                return False
        data, job_dir, project = copy.deepcopy(self.data), str(self.job_dir), str(self.project_dir)
        self.saving = True
        self.save_btn.setEnabled(False)
        self.info.setText("Writing the masks…")
        # the project is named: this process's working directory is wherever cryo-savanna was started
        self.tasks.run(lambda: (M.save_masks(job_dir, data, project_dir=project), data), self._on_saved,
                       channel="save")
        return True

    def _on_saved(self, result) -> None:
        self.saving = False
        if isinstance(result, TaskError):
            QMessageBox.warning(self, "Save masks", f"Could not write the masks: {result}")
            self._changed()
            return
        done, data = result
        self.data["files"] = data["files"]
        self.saved = {n: (data["files"].get(n) or {}).get("key", "") for n in data["tomograms"]}
        self._stamp = _stamp(self.job_dir / M.GEOMETRY)
        for i, n in enumerate(self.names()):
            self._update_row(n, i)
        self._changed()
        counts = {w: sum(1 for v in done.values() if v == w) for w in ("written", "linked", "kept")}
        self.info.setText(f"Saved: {counts['written']} masks written, {counts['linked']} linked to an identical "
                          f"one, {counts['kept']} unchanged   ·   {self.job}{M.MASK_DIR}/")
