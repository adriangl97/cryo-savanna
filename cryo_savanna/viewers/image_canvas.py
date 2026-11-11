# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""One image with zoom and pan: a downsampled preview first, then the zoomed-in region at full detail.

The whole image is shown at about the widget's size (read and downsampled in a worker thread,
cached). Zooming in past the preview's resolution reads just the visible region from the file and
lays it on top, binned to the screen's resolution, so a 4k×4k micrograph never goes to Qt in full.
Overlays (particle picks) are drawn in the image's own pixel coordinates.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, QTimer, Signal
from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from cryo_savanna.images.cache import ThumbCache
from cryo_savanna.images.mrc import ImageRef, bin_mean, contrast_limits, downsample, header, read_image, read_region
from cryo_savanna.viewers.tasks import TaskError, Tasks

SIZES = (256, 512, 1024, 2048)  # preview sizes, so previews of different widgets share the cache


def preview_size(pixels: int) -> int:
    return next((s for s in SIZES if s >= pixels), SIZES[-1])


def load_preview(ref: ImageRef, size: int, cache: ThumbCache | None, disk: bool) -> tuple[np.ndarray, float, tuple]:
    """(preview, factor, full shape): factor = original pixels per preview pixel."""
    def make() -> np.ndarray:
        return downsample(read_image(ref), size)[0]
    img = cache.get(ref, size, make, disk=disk) if cache is not None else make()
    full = _full_shape(ref, img)
    return img, full[1] / img.shape[1], full


def _full_shape(ref: ImageRef, img: np.ndarray) -> tuple[int, int]:
    if ref.path.lower().endswith((".mrc", ".mrcs", ".ctf", ".st", ".map")):
        try:
            return header(ref.path).shape
        except OSError:
            pass
    return img.shape


class ImageCanvas(QWidget):
    loaded = Signal()  # a new image is shown
    hovered = Signal(float, float)  # image pixel under the mouse

    def __init__(self, cache: ThumbCache | None = None, controls: bool = True, disk_cache: bool = False,
                 parent=None):
        super().__init__(parent)
        self.cache, self.disk_cache = cache, disk_cache
        self.tasks = Tasks(self)
        self.ref: ImageRef | None = None
        self.full_shape: tuple[int, int] | None = None
        self.factor = 1.0
        self.preview: np.ndarray | None = None
        self.levels = (0.0, 1.0)
        self.error = ""

        self.gv = pg.GraphicsView(background="k")
        self.view = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.gv.setCentralItem(self.view)
        self.item = pg.ImageItem()
        self.detail = pg.ImageItem()
        self.detail.setZValue(1)
        self.detail.hide()
        self.points = pg.ScatterPlotItem(pxMode=False, brush=None, pen=pg.mkPen((60, 220, 60), width=2))
        self.points.setZValue(2)
        for it in (self.item, self.detail, self.points):
            self.view.addItem(it)
        self.message = pg.TextItem("", color=(220, 220, 220), anchor=(0.5, 0.5))
        self.message.setZValue(3)
        self.view.addItem(self.message, ignoreBounds=True)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0, 20)
        self.sigma.setSingleStep(0.5)
        self.sigma.setValue(3.0)
        self.sigma.setPrefix("Contrast ±")
        self.sigma.setSuffix(" σ")
        self.sigma.setToolTip("Display range: mean ± this many standard deviations (0 = minimum to maximum)")
        self.sigma.valueChanged.connect(self._apply_levels)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Show the whole image")
        self.fit_btn.clicked.connect(self.fit)
        self.info = QLabel()
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.sigma)
        bar.addWidget(self.fit_btn)
        bar.addWidget(self.info, 1)
        self.bar = QWidget()
        self.bar.setLayout(bar)
        self.bar.setVisible(controls)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.bar)
        lay.addWidget(self.gv, 1)

        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(200)
        self._detail_timer.timeout.connect(self._update_detail)
        self.view.sigRangeChanged.connect(lambda *_: self._detail_timer.start())
        self.gv.scene().sigMouseMoved.connect(self._on_mouse)

    # ------------------------------------------------------------------ loading
    def set_image(self, ref: ImageRef | None, keep_view: bool = False) -> None:
        """Show an image file (or one image of a stack); None clears the view."""
        same = ref is not None and ref == self.ref
        self.ref = ref
        self.detail.hide()
        if ref is None:
            self.tasks.cancel("image")
            self._clear_image()
            self.preview = None
            self._show_message("")
            return
        if not same:
            self._show_message("Loading…" if self.preview is None else "")
        size = preview_size(int(max(self.gv.width(), self.gv.height(), 256) * self.devicePixelRatioF()))
        cache, disk = self.cache, self.disk_cache
        self.tasks.run(lambda: load_preview(ref, size, cache, disk), lambda r: self._on_preview(ref, r, keep_view or same),
                       channel="image")

    def prefetch(self, ref: ImageRef) -> None:
        """Read an image into the cache ahead of time (the next micrograph)."""
        size = preview_size(int(max(self.gv.width(), self.gv.height(), 256) * self.devicePixelRatioF()))
        cache, disk = self.cache, self.disk_cache
        if cache is not None:
            # a channel of its own: stepping quickly, only the newest image ahead is read
            self.tasks.run(lambda: load_preview(ref, size, cache, disk), channel="prefetch")

    def set_array(self, img: np.ndarray, keep_view: bool = False) -> None:
        """Show an array that is already in memory (no detail refinement)."""
        self.ref = None
        self.detail.hide()
        self._on_preview(None, (img.astype(np.float32, copy=False), 1.0, img.shape), keep_view)

    def _on_preview(self, ref, result, keep_view: bool) -> None:
        if isinstance(result, TaskError):
            self.error = str(result)
            self._clear_image()
            self.preview = None
            self._show_message(self.error)
            self.info.setText("")
            self.loaded.emit()
            return
        img, factor, full = result
        self.error = ""
        first = self.full_shape != tuple(full) or not keep_view
        self.preview, self.factor, self.full_shape = img, factor, tuple(full)
        self.levels = contrast_limits(img, self.sigma.value())
        self.item.setImage(img, levels=self.levels, autoLevels=False)
        self.item.setRect(QRectF(0, 0, full[1], full[0]))
        self._show_message("")
        if first:
            self.fit()
        else:
            self._detail_timer.start()
        name = ref.path.rsplit("/", 1)[-1] if ref is not None else ""
        if ref is not None and ref.index is not None:
            name = f"{ref.index + 1}@{name}"
        self.info.setText(f"{name}   {full[1]}×{full[0]}" + (f"   (shown at 1/{factor:.1f})" if factor > 1.01 else ""))
        self.loaded.emit()

    def _clear_image(self) -> None:
        # ImageItem.clear() empties its bounds before prepareGeometryChange (as the points do): mark the
        # old image's area first, so the previous image does not stay on screen
        self.item.prepareGeometryChange()
        self.item.clear()
        self.detail.hide()

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))
        if text:
            r = self.view.viewRect()
            self.message.setPos(r.center())

    def fit(self) -> None:
        if self.full_shape:
            ny, nx = self.full_shape
            self.view.setRange(QRectF(0, 0, nx, ny), padding=0.01)

    # ------------------------------------------------------------------ contrast
    def _apply_levels(self) -> None:
        if self.preview is None:
            return
        self.levels = contrast_limits(self.preview, self.sigma.value())
        self.item.setLevels(self.levels)
        if self.detail.isVisible():
            self.detail.setLevels(self.levels)

    # ------------------------------------------------------------------ detail on zoom
    def _visible(self) -> tuple[int, int, int, int, float] | None:
        """Visible part of the image (x0, y0, x1, y1) and image pixels per screen pixel."""
        if self.full_shape is None:
            return None
        ny, nx = self.full_shape
        r = self.view.viewRect()
        width = max(1.0,self.gv.viewport().width() * self.devicePixelRatioF())
        per_px = r.width() / width
        x0, y0 = max(0, int(r.left())), max(0, int(r.top()))
        x1, y1 = min(nx, int(np.ceil(r.right()))), min(ny, int(np.ceil(r.bottom())))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1, per_px

    def _update_detail(self) -> None:
        ref = self.ref
        vis = self._visible()
        if ref is None or vis is None or self.factor <= 1.01:
            self.detail.hide()
            return
        x0, y0, x1, y1, per_px = vis
        if per_px >= self.factor * 0.75:  # the preview is sharp enough at this zoom
            self.detail.hide()
            self.tasks.cancel("detail")
            return
        # a margin around the visible part, so small pans need no new read
        mx, my = (x1 - x0) // 4, (y1 - y0) // 4
        ny, nx = self.full_shape
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(nx, x1 + mx), min(ny, y1 + my)
        k = max(1, int(per_px))

        def load():
            return bin_mean(read_region(ref, y0, y1, x0, x1), k)

        def show(img) -> None:
            if isinstance(img, TaskError) or ref != self.ref:
                return
            self.detail.setImage(img, levels=self.levels, autoLevels=False)
            self.detail.setRect(QRectF(x0, y0, img.shape[1] * k, img.shape[0] * k))
            self.detail.show()

        self.tasks.run(load, show, channel="detail")

    # ------------------------------------------------------------------ overlays
    def set_points(self, x: np.ndarray, y: np.ndarray, diameter, color=(60, 220, 60)) -> None:
        """Circles of the given diameter (image pixels; one for all, or one per point) centred on (x, y),
        in one colour or one per point (a list of colours)."""
        # pyqtgraph drops the old points' bounds before telling Qt the geometry changes, so Qt would
        # repaint only the new area: where the old circles were stays on screen unless something
        # else covers it (an image that failed to load does not)
        self.points.prepareGeometryChange()
        if len(x) == 0:
            self.points.clear()
            return
        size = np.asarray(diameter, float) if np.ndim(diameter) else float(diameter)
        if isinstance(color, list):
            pens = {c: pg.mkPen(c, width=2) for c in set(color)}
            pen = [pens[c] for c in color]
        else:
            pen = pg.mkPen(color, width=2)
        self.points.setData(x=np.asarray(x, float), y=np.asarray(y, float), size=size, symbol="o",
                            pen=pen, brush=None)

    def _on_mouse(self, pos) -> None:
        if self.full_shape is None:
            return
        p = self.view.mapSceneToView(pos)
        ny, nx = self.full_shape
        if 0 <= p.x() < nx and 0 <= p.y() < ny:
            self.hovered.emit(p.x(), p.y())
