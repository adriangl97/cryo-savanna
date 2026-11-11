# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""3D maps: three orthogonal sections (or projections) and a software-rendered isosurface.

Sections cross at one point: clicking in one moves the other two, the sliders step through each.
They come from the map at full resolution (small maps are held in memory; sections of big maps
are read from the file). The surface is ray-cast from a copy binned to ≤192³ (images/render.py),
at reduced size while it is being rotated and in full when the mouse stops. A mask can be drawn
as an outline on the sections, and a local-resolution map can colour the sections and the surface.
"Open in ChimeraX" shows the same map, level, mask and colouring there.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QImage, QLinearGradient, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.images.mrc import contrast_limits
from cryo_savanna.images.render import framing, occupied, render, rotation
from cryo_savanna.images.volume import Volume, colormap, load_volume, outline, projection, resolution_range, section
from cryo_savanna.viewers.external import chimerax_script, find_chimerax, launch_chimerax
from cryo_savanna.viewers.tasks import TaskError, Tasks

PLANES = [("XY", 0), ("XZ", 1), ("YZ", 2)]  # (title, axis it cuts)
MASK_EDGE = (230, 40, 40, 255)


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def plane_axes(axis: int) -> tuple[int, int]:
    """(row axis, column axis) of the plane that cuts ``axis``, in [z, y, x] numbering."""
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


class SliceView(QWidget):
    clicked = Signal(int, float, float)  # axis, row, column (full-map voxels)
    index_changed = Signal(int, int)  # axis, index

    def __init__(self, title: str, axis: int, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.gv = pg.GraphicsView(background="k")
        self.view = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.gv.setCentralItem(self.view)
        self.img = pg.ImageItem()
        self.overlay = pg.ImageItem()
        self.overlay.setZValue(1)
        pen = pg.mkPen((240, 200, 40, 160), width=1)
        self.hline = pg.InfiniteLine(angle=0, pen=pen)
        self.vline = pg.InfiniteLine(angle=90, pen=pen)
        for it in (self.img, self.overlay, self.hline, self.vline):
            self.view.addItem(it)
        self.title = QLabel(title)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.index_changed.emit(self.axis, v))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.title)
        top.addWidget(self.slider, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addLayout(top)
        lay.addWidget(self.gv, 1)
        self.gv.scene().sigMouseClicked.connect(self._on_click)
        self._shape: tuple[int, int] | None = None

    def set_range(self, n: int, index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(index)
        self.slider.blockSignals(False)

    def set_plane(self, img: np.ndarray, levels, full: tuple[int, int], overlay: np.ndarray | None) -> None:
        refit = self._shape != full
        self._shape = full
        self.img.setImage(img, levels=levels, autoLevels=False)
        self.img.setRect(QRectF(0, 0, full[1], full[0]))
        if overlay is None:
            self.overlay.clear()
        else:
            self.overlay.setImage(overlay, autoLevels=False, levels=(0, 255))
            self.overlay.setRect(QRectF(0, 0, full[1], full[0]))
        if refit:
            self.view.setRange(QRectF(0, 0, full[1], full[0]), padding=0.02)

    def set_cross(self, row: float | None, col: float | None) -> None:
        for line, v in ((self.hline, row), (self.vline, col)):
            line.setVisible(v is not None)
            if v is not None:
                line.setValue(v + 0.5)

    def _on_click(self, ev) -> None:
        if self._shape is None or ev.button() != Qt.LeftButton:
            return
        p = self.view.mapSceneToView(ev.scenePos())
        if 0 <= p.x() < self._shape[1] and 0 <= p.y() < self._shape[0]:
            self.clicked.emit(self.axis, p.y(), p.x())


def drag_rotation(rot: np.ndarray, dx: float, dy: float, degrees_per_pixel: float = 0.5) -> np.ndarray:
    """The view after a mouse drag of (dx, dy) screen pixels, rotating as ChimeraX does.

    The map turns about the screen's own axes (vertical for dx, horizontal for dy), and the side
    facing you follows the mouse: drag right and the front moves right, drag down and it moves down.
    ``rot`` maps map coordinates to screen coordinates (x right, y down, z into the screen), so a
    turn of the map on screen is applied on the left.
    """
    a = np.radians(-dx * degrees_per_pixel)  # about the screen's vertical axis
    b = np.radians(dy * degrees_per_pixel)  # about the screen's horizontal axis
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    turn_x = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    turn_y = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    r = turn_x @ turn_y @ rot
    u, _, vt = np.linalg.svd(r)  # keep it a pure rotation after many small steps
    return u @ vt


class SurfaceView(QWidget):
    """The isosurface, rotated with the mouse (drag, as in ChimeraX), zoomed with the wheel.

    One render runs at a time: requests made meanwhile (mouse moves come much faster than renders
    on a slow machine) collapse into one, made with the latest view as soon as the running one is
    shown. While the mouse moves the image is drawn smaller; 250 ms after it stops, in full.
    """

    FAST = 160  # pixels across while rotating
    MIN_FRAME_S = 0.05  # at most 20 frames a second while rotating: each frame crosses the X11 connection

    rendered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self.tasks = Tasks(self)
        self.data: np.ndarray | None = None
        self.color: np.ndarray | None = None
        self.color_range = (0.0, 1.0)
        self.threshold = 0.0
        self.rot = rotation(30.0, -20.0)  # map (x, y, z) → screen (right, down, into the screen)
        self.zoom = 1.0
        self.frame = None
        self.default_frame = None  # (centre, radius) to frame instead of the part above the level
        self.shown = None  # (rot, zoom, frame) of the image on screen, to draw on it in the same projection
        self.image: QImage | None = None
        self.hit_fraction = 0.0
        self.render_time = 0.0  # seconds, last render
        self._drag = None
        self._busy = False
        self._pending: bool | None = None  # a request made while rendering: fast?
        self._bounds = None  # (data, threshold, box above the threshold): reused while rotating
        self._last_start = 0.0
        self._hold = QTimer(self)  # the frame cap's wait; owned, so it dies with the view
        self._hold.setSingleShot(True)
        self._hold.timeout.connect(self._start_held)
        self._held_fast = True
        self._full = QTimer(self)
        self._full.setSingleShot(True)
        self._full.setInterval(250)
        self._full.timeout.connect(lambda: self._request(fast=False))
        self.message = "Loading…"

    def set_volume(self, data: np.ndarray | None, threshold: float, color: np.ndarray | None = None,
                   color_range=(0.0, 1.0), reframe: bool = True) -> None:
        self.data, self.color, self.color_range = data, color, color_range
        self.set_threshold(threshold, reframe)

    def set_threshold(self, threshold: float, reframe: bool = True) -> None:
        self.threshold = threshold
        if reframe or self.frame is None:
            self.frame = None  # computed with the next render
        self._request(fast=False)

    def set_color(self, color: np.ndarray | None, color_range=(0.0, 1.0)) -> None:
        self.color, self.color_range = color, color_range
        self._request(fast=False)

    def view_along(self, axis: str) -> None:
        self.rot = rotation(*{"z": (0.0, 0.0), "y": (0.0, 90.0), "x": (90.0, 0.0)}[axis])
        self._request(fast=False)

    def reset(self) -> None:
        self.rot, self.zoom = rotation(30.0, -20.0), 1.0
        self.frame = None
        self._request(fast=False)

    def _request(self, fast: bool) -> None:
        if self.data is None:
            self.update()
            return
        if fast:
            self._full.start()
        if self._busy:  # remember only the latest; a full request wins over a fast one
            self._pending = fast and (self._pending is None or self._pending)
            return
        self._start(fast)

    def _start(self, fast: bool) -> None:
        side = min(self.width(), self.height())
        size = self.FAST if fast else int(np.clip(side * self.devicePixelRatioF(), self.FAST, 384))
        data, color, crange, thr = self.data, self.color, self.color_range, self.threshold
        rot, zoom = self.rot.copy(), self.zoom
        frame = self.frame if self.frame is not None else self.default_frame
        cached = self._bounds
        bounds = cached[2] if cached is not None and cached[0] is data and cached[1] == thr else None

        def work():
            t = time.perf_counter()
            b = bounds if bounds is not None else (occupied(data, thr),)
            fr = frame if frame is not None else framing(data, thr, b[0])
            r = render(data, thr, rot, size, color, crange, zoom=zoom, frame=fr, bounds=b[0])
            return r, fr, (data, thr, b), time.perf_counter() - t, (rot, zoom, fr)

        self._busy = True
        self._last_start = time.monotonic()
        self.tasks.run(work, self._on_render)

    def _on_render(self, result) -> None:
        self._busy = False
        if isinstance(result, TaskError):
            self.message = str(result)
            self.image = None
        else:
            r, self.frame, self._bounds, self.render_time, self.shown = result
            h, w, _ = r.rgb.shape
            rgba = np.dstack([r.rgb, np.where(r.hit, 255, 0).astype(np.uint8)])  # see-through where no surface
            self.image = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
            self.hit_fraction = float(r.hit.mean())
            self.message = "" if r.hit.any() else "Nothing above this level"
        self.update()
        self.rendered.emit()
        if self._pending is not None:
            fast, self._pending = self._pending, None
            wait = self.MIN_FRAME_S - (time.monotonic() - self._last_start) if fast else 0
            if wait > 0:
                self._busy = True  # requests meanwhile still collapse into this one
                self._held_fast = fast
                self._hold.start(int(wait * 1000))
            else:
                self._start(fast)

    def _start_held(self) -> None:
        fast = self._held_fast
        if self._pending is not None:  # asked for again while waiting: full wins over fast
            fast, self._pending = fast and self._pending, None
        self._start(fast)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().color(QPalette.Base))
        if self.image is not None:
            side = min(self.width(), self.height())
            target = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
            self.paint_layer(p, target, front=False)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(target, self.image)
            self.paint_layer(p, target, front=True)
        if self.message:
            p.setPen(self.palette().color(QPalette.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
        p.end()

    def paint_layer(self, p: QPainter, target: QRectF, front: bool) -> None:
        """Drawn behind the surface (front=False) and over it; nothing here (see AngdistSurface)."""

    def project(self, points: np.ndarray, target: QRectF) -> tuple[np.ndarray, np.ndarray, float]:
        """Screen x, y and depth (into the screen, 0 at the frame's centre) of map points (x, y, z, in
        binned voxels), as the image on screen was rendered; and screen pixels per voxel."""
        rot, zoom, (centre, radius) = self.shown
        half = radius / zoom
        n = self.image.width()
        per_pixel = target.width() / n
        rel = np.asarray(points, float) - np.asarray(centre, float)
        # render() samples u = linspace(-half, half, n) at pixel centres
        x = target.left() + ((rel @ rot[0] + half) / (2 * half) * (n - 1) + 0.5) * per_pixel
        y = target.top() + ((rel @ rot[1] + half) / (2 * half) * (n - 1) + 0.5) * per_pixel
        return np.stack([x, y], axis=-1), rel @ rot[2], (n - 1) / (2 * half) * per_pixel

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self.data is not None:
            self._full.start()

    def mousePressEvent(self, ev) -> None:
        self._drag = ev.position() if hasattr(ev, "position") else ev.pos()

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is None:
            return
        pos = ev.position() if hasattr(ev, "position") else ev.pos()
        d = pos - self._drag
        self._drag = pos
        self.rot = drag_rotation(self.rot, d.x(), d.y())
        self._request(fast=True)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None
        self._request(fast=False)

    def wheelEvent(self, ev) -> None:
        steps = ev.angleDelta().y() / 120
        self.zoom = float(np.clip(self.zoom * (1.15 ** steps), 0.3, 8.0))
        self._request(fast=True)


def colorbar(lo: float, hi: float, width: int = 120, height: int = 12) -> QPixmap:
    pix = QPixmap(width, height)
    p = QPainter(pix)
    g = QLinearGradient(0, 0, width, 0)
    for t in np.linspace(0, 1, 9):
        r, gg, b = colormap(np.array([lo + t * (hi - lo)]), lo, hi)[0]
        g.setColorAt(float(t), QColor(int(r * 255), int(gg * 255), int(b * 255)))
    p.fillRect(0, 0, width, height, g)
    p.end()
    return pix


class VolumeViewer(QWidget):
    """One of the job's maps; optionally a mask outline and local-resolution colouring."""

    loaded = Signal()

    def __init__(self, project_dir: Path, maps: list[str], mask: str = "", color: str = "", masks: list[str] = (),
                 parent=None, follow: Callable[[], list[str]] | None = None,
                 history: Callable[[], list[tuple[int, list[str]]]] | None = None):
        """``follow``: the maps to show now (a running job's latest iteration), asked again on every refresh;
        a new iteration is read keeping the class shown, the view and the level.

        ``history``: (iteration, maps) of every iteration of an iterative job (results.classes.iteration_maps);
        an Iteration menu then offers them next to the final maps (or the latest iteration, when following),
        and lists new iterations as they are written."""
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.follow = follow
        self.history = history
        self.maps = [m for m in (follow() if follow is not None else maps) if m]
        self.final_maps = [] if follow is not None else list(self.maps)
        self._choices: list[tuple[str, list[str] | None]] = []  # Iteration menu: (label, maps); None = follow
        self.mask_rel, self.color_rel = mask, color
        self.mask_maps = set(masks)  # maps that are masks themselves (level 0.5)
        self.tasks = Tasks(self)
        self.vol: Volume | None = None
        self.mask: Volume | None = None
        self.locres: Volume | None = None
        self.color_range = (0.0, 1.0)
        self.pos = [0, 0, 0]  # [z, y, x], full-map voxels
        self._stamp = None
        self._others: dict[str, tuple] = {}  # mask / local-resolution map -> (stamp, Volume): read once
        self.error = ""

        self.iteration = QComboBox()
        self.iteration.setToolTip("The maps of which iteration: the final ones (or the latest, while the job runs), "
                                  "or any earlier iteration")
        self.iteration.setVisible(history is not None)
        self.iteration.currentIndexChanged.connect(lambda _: self._on_iteration())
        self._fill_iterations()
        self.map_combo = QComboBox()
        self.map_combo.addItems([Path(m).name for m in self.maps])
        self.map_combo.setToolTip("\n".join(self.maps))
        self.map_combo.currentIndexChanged.connect(lambda _: self.load())
        self.mode = QComboBox()
        self.mode.addItems(["Sections", "Projections"])
        self.mode.currentIndexChanged.connect(lambda _: self._update_planes())
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0, 20)
        self.sigma.setSingleStep(0.5)
        self.sigma.setValue(3.0)
        self.sigma.setPrefix("Contrast ±")
        self.sigma.setSuffix(" σ")
        self.sigma.valueChanged.connect(lambda _: self._update_planes())
        self.show_mask = QCheckBox("Mask")
        self.show_mask.setToolTip(f"Outline of {mask}" if mask else "")
        self.show_mask.setChecked(bool(mask))
        self.show_mask.setVisible(bool(mask))
        self.show_mask.toggled.connect(lambda _: self._update_planes())
        self.show_color = QCheckBox("Local resolution")
        self.show_color.setToolTip(f"Colour by {color}" if color else "")
        self.show_color.setChecked(bool(color))
        self.show_color.setVisible(bool(color))
        self.show_color.toggled.connect(lambda _: self._color_changed())
        self.bar_lo, self.bar_hi, self.bar = QLabel(), QLabel(), QLabel()
        for w in (self.bar_lo, self.bar, self.bar_hi):
            w.setVisible(bool(color))
        self.chimerax_btn = QPushButton("Open in ChimeraX")
        self.chimerax = find_chimerax()
        self.chimerax_btn.setEnabled(self.chimerax is not None)
        self.chimerax_btn.setToolTip(self.chimerax or "ChimeraX not found: put it on PATH or set CRYO_SAVANNA_CHIMERAX")
        self.chimerax_btn.clicked.connect(self.open_chimerax)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        for w in (self.iteration, self.map_combo, self.mode, self.sigma, self.show_mask, self.show_color, self.bar_lo, self.bar,
                  self.bar_hi):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.chimerax_btn)

        self.slices = [SliceView(t, a) for t, a in PLANES]
        for s in self.slices:
            s.clicked.connect(self._on_click)
            s.index_changed.connect(self._on_index)
        self.surface = SurfaceView()
        self.level = QDoubleSpinBox()
        self.level.setDecimals(4)
        self.level.setRange(-1e9, 1e9)
        self.level.setPrefix("Level ")
        self.level.setKeyboardTracking(False)
        self.level.valueChanged.connect(self._on_level)
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(0, 1000)
        self.level_slider.valueChanged.connect(self._on_level_slider)
        self.level_info = QLabel()
        sbar = QHBoxLayout()
        sbar.setContentsMargins(0, 0, 0, 0)
        sbar.addWidget(QLabel("Surface"))
        sbar.addWidget(self.level)
        sbar.addWidget(self.level_slider, 1)
        sbar.addWidget(self.level_info)
        for text, fn in (("X", lambda: self.surface.view_along("x")), ("Y", lambda: self.surface.view_along("y")),
                         ("Z", lambda: self.surface.view_along("z")), ("Reset", self.surface.reset)):
            b = QPushButton(text)
            b.setToolTip("View along " + text if len(text) == 1 else "Default view")
            b.setMaximumWidth(56)
            b.clicked.connect(fn)
            sbar.addWidget(b)
        surf = QWidget()
        sv = QVBoxLayout(surf)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(1)
        sv.addLayout(sbar)
        sv.addWidget(self.surface, 1)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.addWidget(self.slices[0], 0, 0)
        grid.addWidget(self.slices[1], 0, 1)
        grid.addWidget(self.slices[2], 1, 0)
        grid.addWidget(surf, 1, 1)
        self.info = QLabel("Reading…")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addLayout(grid, 1)
        lay.addWidget(self.info)
        if self.maps:
            self.load()
        else:
            self.info.setText("No map")

    # ------------------------------------------------------------------ loading
    @property
    def map_rel(self) -> str:
        return self.maps[max(0, self.map_combo.currentIndex())]

    def _path(self, rel: str) -> Path:
        return self.project_dir / rel

    def load(self, keep_view: bool = False) -> None:
        rel, mask, color = self.map_rel, self.mask_rel, self.color_rel
        project = self.project_dir
        self._stamp = _stamp(project / rel)

        kept = self._others  # the mask and the local-resolution map stay the same from map to map

        def other(name: str):
            if not name or not (project / name).is_file():
                return None
            st = _stamp(project / name)
            hit = kept.get(name)
            if hit is None or hit[0] != st:
                hit = kept[name] = (st, load_volume(project / name))
            return hit[1]

        def read():
            return load_volume(project / rel), other(mask), other(color)

        self.info.setText("Reading…")
        self.tasks.run(read, lambda r: self._on_loaded(r, keep_view), channel="load")

    # ------------------------------------------------------------------ iterations
    def _following(self) -> bool:
        """Whether the viewer shows (and follows) the latest iteration of a running job."""
        if self.follow is None:
            return False
        i = self.iteration.currentIndex()
        return not self._choices or i < 0 or self._choices[i][1] is None

    def _fill_iterations(self) -> None:
        """The Iteration menu: the final maps (or the latest iteration), then every iteration, newest first."""
        if self.history is None:
            return
        try:
            history = self.history()
        except OSError:
            history = []
        first = [("Latest iteration", None)] if self.follow is not None else \
            [("Final", self.final_maps)] if self.final_maps else []
        choices = first + [(f"Iteration {it}", maps) for it, maps in reversed(history)]
        if choices == self._choices:
            return
        current = self.iteration.currentText()
        self._choices = choices
        self.iteration.blockSignals(True)
        self.iteration.clear()
        self.iteration.addItems([label for label, _ in choices])
        i = self.iteration.findText(current)
        self.iteration.setCurrentIndex(i if i >= 0 else 0)
        self.iteration.blockSignals(False)

    def iterations_offered(self) -> list[str]:
        return [label for label, _ in self._choices]

    def show_iteration(self, label: str) -> bool:
        i = self.iteration.findText(label)
        if i >= 0:
            self.iteration.setCurrentIndex(i)
        return i >= 0

    def _on_iteration(self) -> None:
        i = self.iteration.currentIndex()
        if not 0 <= i < len(self._choices):
            return
        maps = self._choices[i][1]
        maps = [m for m in (self.follow() if maps is None else maps) if m]
        if maps:
            self._set_maps(maps)
            self.load(keep_view=True)

    def refresh(self) -> None:
        self._fill_iterations()  # new iterations are listed as they are written
        if self._following():
            maps = [m for m in self.follow() if m]
            if maps and maps != self.maps:  # a new iteration
                self._set_maps(maps)
                self.load(keep_view=True)
                return
        if self.maps and _stamp(self._path(self.map_rel)) != self._stamp:
            self.load(keep_view=True)

    def _set_maps(self, maps: list[str]) -> None:
        """Show these maps instead, keeping the one at the same place (class 2 of the next iteration)."""
        index = max(0, self.map_combo.currentIndex())
        self.maps = maps
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        self.map_combo.addItems([Path(m).name for m in maps])
        self.map_combo.setCurrentIndex(min(index, len(maps) - 1))
        self.map_combo.blockSignals(False)
        self.map_combo.setToolTip("\n".join(maps))

    def _on_loaded(self, result, keep_view: bool) -> None:
        if isinstance(result, TaskError):
            self.error = str(result)
            self.info.setText(f"⚠ {result}")
            return
        v, m, c = result
        notes = []
        if m is not None and m.shape != v.shape:
            notes.append(f"the mask's box ({m.shape[2]}×{m.shape[1]}×{m.shape[0]}) differs from the map's: not shown")
            m = None
        if c is not None and c.shape != v.shape:
            notes.append("the local-resolution map's box differs from the map's: not shown")
            c = None
        old = self.vol
        self.vol, self.mask, self.locres = v, m, c
        self.error = ""
        self.show_mask.setEnabled(m is not None)
        self.show_color.setEnabled(c is not None)
        if c is not None:
            self.color_range = resolution_range(c.data, m.data if m is not None else None)
            lo, hi = self.color_range
            self.bar_lo.setText(f"{lo:.2f} Å")
            self.bar_hi.setText(f"{hi:.2f} Å")
            self.bar.setPixmap(colorbar(lo, hi))
        if not (keep_view and old is not None and old.shape == v.shape):
            self.pos = [s // 2 for s in v.shape]
        for s in self.slices:
            s.set_range(v.shape[s.axis], self.pos[s.axis])
        # the surface level: 0.5 for masks, else what encloses 1% of the box (as ChimeraX starts)
        is_mask = self.map_rel in self.mask_maps
        level = 0.5 if is_mask else v.level_for_fraction(0.01)
        if keep_view and old is not None:
            level = self.level.value()
        self.level.blockSignals(True)
        self.level.setSingleStep((v.maximum - v.minimum) / 200 or 0.01)
        self.level.setValue(level)
        self.level.blockSignals(False)
        self._sync_slider()
        self.surface.set_volume(v.data, level, *self._surface_color(), reframe=not keep_view)
        self._update_planes()
        z, y, x = v.shape
        text = f"{Path(v.path).name}   {x}×{y}×{z}" + (f"   {v.voxel:g} Å/voxel" if v.voxel else "")
        if v.factor > 1:
            text += f"   (surface and projections binned by {v.factor})"
        if self._following():
            text += "   ·   the latest iteration written; new ones are shown as they come"
        self.info.setText("   ·   ".join([text] + notes))
        self.loaded.emit()

    # ------------------------------------------------------------------ sections
    def _on_click(self, axis: int, row: float, col: float) -> None:
        ra, ca = plane_axes(axis)
        self.pos[ra], self.pos[ca] = int(row), int(col)
        for s in self.slices:
            s.set_range(self.vol.shape[s.axis], self.pos[s.axis])
        self._update_planes([ra, ca])  # the clicked section itself stays

    def _on_index(self, axis: int, index: int) -> None:
        self.pos[axis] = index
        self._update_planes([axis])  # the other two only move their crosshairs

    def _levels(self):
        v = self.vol
        s = self.sigma.value()
        if self.mode.currentIndex() == 1:
            return None
        return (v.minimum, v.maximum) if s <= 0 else (v.mean - s * v.std, v.mean + s * v.std)

    def _update_planes(self, axes: list[int] | None = None) -> None:
        """Show the sections again (only those cutting ``axes``, when given); crosshairs always."""
        v = self.vol
        if v is None:
            return
        projections = self.mode.currentIndex() == 1
        for s in self.slices:
            s.slider.setEnabled(not projections)
            ra, ca = plane_axes(s.axis)
            s.set_cross(None if projections else self.pos[ra], None if projections else self.pos[ca])
        if projections:
            for s in self.slices:
                img = projection(v.data, s.axis)
                ra, ca = plane_axes(s.axis)
                full = (v.shape[ra], v.shape[ca])
                ov = None
                if self.mask is not None and self.show_mask.isChecked():
                    ov = self._overlay(None, projection(self.mask.data, s.axis) / self.mask.data.shape[s.axis] > 0.02,
                                       img.shape)
                s.set_plane(img, contrast_limits(img, self.sigma.value()), full, ov)
            return
        changed = [s for s in self.slices if axes is None or s.axis in axes]
        if v.factor == 1 or self._all_in_memory():  # small map: everything is in memory
            for s in changed:
                self._show_plane(s, self._planes_in_memory(s.axis, self.pos[s.axis]))
            return
        paths = (v.path, self.mask.path if self.mask is not None else None,
                 self.locres.path if self.locres is not None else None)
        for s in changed:
            axis, index = s.axis, self.pos[s.axis]
            self.tasks.run(lambda a=axis, i=index: tuple(section(p, a, i) if p else None for p in paths),
                           lambda r, sv=s: None if isinstance(r, TaskError) else self._show_plane(sv, r),
                           channel=f"plane{axis}")

    def _all_in_memory(self) -> bool:
        return all(v is None or v.full is not None for v in (self.vol, self.mask, self.locres))

    def _planes_in_memory(self, axis: int, index: int):
        def cut(vol: Volume | None):
            if vol is None:
                return None
            return np.take(vol.full if vol.full is not None else vol.data, index, axis=axis)
        return cut(self.vol), cut(self.mask), cut(self.locres)

    def _show_plane(self, s: SliceView, planes) -> None:
        img, mask, locres = planes
        v = self.vol
        ra, ca = plane_axes(s.axis)
        ov = None
        show_mask = mask is not None and self.show_mask.isChecked()
        show_color = locres is not None and self.show_color.isChecked()
        if show_mask or show_color:
            ov = self._overlay(locres if show_color else None, mask, img.shape, edges=show_mask)
        s.set_plane(img, self._levels(), (v.shape[ra], v.shape[ca]), ov)

    def _overlay(self, locres, mask, shape, edges: bool = True) -> np.ndarray:
        rgba = np.zeros(shape + (4,), np.uint8)
        if locres is not None:
            inside = mask > 0.5 if mask is not None else np.isfinite(locres)
            rgba[..., :3] = (colormap(locres, *self.color_range) * 255).astype(np.uint8)
            rgba[..., 3] = np.where(inside, 110, 0)  # the map stays visible underneath
        if mask is not None and edges:
            rgba[outline(mask.astype(np.float32))] = MASK_EDGE
        return rgba

    # ------------------------------------------------------------------ surface
    def _surface_color(self):
        if self.locres is not None and self.show_color.isChecked():
            return self.locres.data, self.color_range
        return None, (0.0, 1.0)

    def _color_changed(self) -> None:
        self.surface.set_color(*self._surface_color())
        self._update_planes()

    def _sync_slider(self) -> None:
        v = self.vol
        if v is None:
            return
        frac = (self.level.value() - v.minimum) / ((v.maximum - v.minimum) or 1.0)
        self.level_slider.blockSignals(True)
        self.level_slider.setValue(int(np.clip(frac, 0, 1) * 1000))
        self.level_slider.blockSignals(False)
        self.level_info.setText(f"{(self.level.value() - v.mean) / v.std:+.1f} σ")

    def _on_level_slider(self, value: int) -> None:
        v = self.vol
        if v is not None:
            self.level.setValue(v.minimum + value / 1000 * (v.maximum - v.minimum))

    def _on_level(self, value: float) -> None:
        self._sync_slider()
        self.surface.set_threshold(value, reframe=False)

    # ------------------------------------------------------------------ ChimeraX
    def chimerax_command(self) -> str:
        mask = str(self._path(self.mask_rel)) if self.mask is not None and self.show_mask.isChecked() else None
        color = str(self._path(self.color_rel)) if self.locres is not None and self.show_color.isChecked() else None
        level = self.level.value() if self.vol is not None else None
        return chimerax_script(str(self._path(self.map_rel)), level, mask, color,
                               self.color_range if color else None)

    def open_chimerax(self) -> None:
        if self.chimerax is None:
            return
        try:
            launch_chimerax(self.chimerax, self.chimerax_command())
        except OSError as e:
            self.info.setText(f"⚠ ChimeraX: {e}")
