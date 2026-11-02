# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The angular distribution in 3D: the map's surface with RELION's .bild cylinders around it.

The surface is the volume viewer's (software ray-cast, rotated and zoomed the same way); the
cylinders are drawn on it in the same projection: those behind the map's centre before the map,
so the map hides them, the others after. Cylinders pointing at you show as discs. Each is drawn
with RELION's own colour (blue: few particles in that direction, red: many).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from cryo_savanna.images.volume import Volume, load_volume
from cryo_savanna.results.bild import AngdistPair, Bild, angdist_pairs, read_bild
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.volume_viewer import SurfaceView


def bild_to_voxels(bild: Bild, vol: Volume) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cylinder ends and radii in the binned map's voxels (x, y, z), and the map's centre there.

    BILD coordinates are Å with voxel 0 at 0 Å (RELION puts the centre at box × pixel size / 2);
    binned voxel j averages full voxels kj … kj+k-1, so it sits at full voxel kj + (k-1)/2.
    """
    k, apix = vol.factor, vol.voxel

    def binned(full):
        return (np.asarray(full, float) - (k - 1) / 2) / k

    nz, ny, nx = vol.shape
    centre = binned(np.array([nx / 2, ny / 2, nz / 2]))
    return binned(bild.start / apix), binned(bild.end / apix), bild.radius / apix / k, centre


class AngdistSurface(SurfaceView):
    """A map's surface with angular-distribution cylinders around it, framed so all of them show."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cylinders = None  # (start, end, radius, QColors), binned voxels

    def set_bild(self, bild: Bild | None, vol: Volume | None) -> None:
        if bild is None or vol is None or not len(bild) or not vol.voxel:
            self.cylinders, self.default_frame = None, None
        else:
            start, end, radius, centre = bild_to_voxels(bild, vol)
            colors = [QColor.fromRgbF(*map(float, c)) for c in bild.color]
            self.cylinders = (start, end, radius, colors)
            reach = float(np.linalg.norm(np.vstack([start, end]) - centre, axis=1).max())
            self.default_frame = (centre, 1.05 * reach + float(radius.max()))
        self.frame = None
        self.update()

    def paint_layer(self, p: QPainter, target: QRectF, front: bool) -> None:
        if self.cylinders is None or self.shown is None:
            return
        start, end, radius, colors = self.cylinders
        (a, da, scale), (b, db, _) = self.project(start, target), self.project(end, target)
        depth = (da + db) / 2
        idx = np.nonzero(depth < 0 if front else depth >= 0)[0]
        idx = idx[np.argsort(-depth[idx], kind="stable")]  # the farthest first
        widths = np.maximum(2 * radius * scale, 1.0)
        # farther is darker, as on the surface (render(): 1 - 0.35 × depth, 0 front .. 1 back)
        reach = self.shown[2][1]
        shade = 100 + np.round(35 * np.clip((depth + reach) / (2 * reach), 0, 1)).astype(int)
        rounded = self.image.width() > self.FAST  # a lighter core makes them look round; not while rotating
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)  # a cylinder pointing at you is a disc
        end_on = np.hypot(*(b - a).T) < 0.5  # Qt draws nothing for a line without length
        for i in idx:
            color = colors[i].darker(int(shade[i]))
            pa, pb = QPointF(*a[i]), QPointF(*b[i])
            for c, w in ((color, widths[i]), (color.lighter(145), 0.4 * widths[i]) if rounded and widths[i] >= 4
                         else (None, 0)):
                if c is None:
                    break
                if end_on[i]:
                    p.setPen(Qt.NoPen)
                    p.setBrush(c)
                    p.drawEllipse(pa, w / 2, w / 2)
                else:
                    pen.setColor(c)
                    pen.setWidthF(float(w))
                    p.setPen(pen)
                    p.drawLine(pa, pb)


class AngdistPanel(QWidget):
    """The job's maps with their .bild files: choose one, set the surface level, rotate."""

    def __init__(self, project_dir: Path, job: str, parent=None):
        super().__init__(parent)
        self.project_dir, self.job = Path(project_dir), job
        self.tasks = Tasks(self)
        self.pairs: list[AngdistPair] | None = None  # None: not looked yet
        self.vol: Volume | None = None
        self.combo = QComboBox()
        self.combo.setToolTip("Map and .bild file (RELION writes one per class and iteration)")
        self.combo.currentIndexChanged.connect(lambda _: self.load())
        self.level = QDoubleSpinBox()
        self.level.setDecimals(4)
        self.level.setRange(-1e9, 1e9)
        self.level.setPrefix("Level ")
        self.level.setKeyboardTracking(False)
        self.level.valueChanged.connect(lambda v: self.surface.set_threshold(v, reframe=False))
        self.surface = AngdistSurface()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.combo, 1)
        bar.addWidget(self.level)
        for text, fn in (("X", lambda: self.surface.view_along("x")), ("Y", lambda: self.surface.view_along("y")),
                         ("Z", lambda: self.surface.view_along("z")), ("Reset", self.surface.reset)):
            btn = QPushButton(text)
            btn.setToolTip("View along " + text if len(text) == 1 else "Default view")
            btn.setMaximumWidth(56)
            btn.clicked.connect(fn)
            bar.addWidget(btn)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.surface, 1)
        lay.addWidget(self.info)
        self.scan()

    def scan(self) -> None:
        """Look for (new) .bild files; the first (final, or the latest iteration) is shown unless another was chosen."""
        pairs = angdist_pairs(self.project_dir, self.job)
        if self.pairs is not None and [(q.map, q.bild) for q in pairs] == [(q.map, q.bild) for q in self.pairs]:
            return
        current = self.combo.currentIndex()
        old = self.pairs or []
        chosen = old[current].bild if 0 < current < len(old) else None
        self.pairs = pairs
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems([q.label for q in pairs])
        keep = next((i for i, q in enumerate(pairs) if q.bild == chosen), 0)
        self.combo.setCurrentIndex(keep)
        self.combo.blockSignals(False)
        if not pairs:
            self.surface.set_bild(None, None)
            self.surface.set_volume(None, 0.0)
            self.surface.message = "No .bild file in this job"
            self.info.setText("RELION writes the angular distribution as .bild files in Refine3D, Class3D "
                              "and MultiBody jobs.")
            self.surface.update()
            return
        self.load(keep_view=self.vol is not None)

    def load(self, keep_view: bool = False) -> None:
        if not self.pairs:
            return
        pair = self.pairs[max(0, self.combo.currentIndex())]
        project = self.project_dir
        self.info.setText(f"Reading {Path(pair.map).name}…")
        self.tasks.run(lambda: (load_volume(project / pair.map), read_bild(project / pair.bild), pair),
                       lambda r: self._on_loaded(r, keep_view), channel="angdist")

    def _on_loaded(self, result, keep_view: bool) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        vol, bild, pair = result
        same_box = self.vol is not None and self.vol.shape == vol.shape
        self.vol = vol
        if not vol.voxel:
            self.info.setText(f"⚠ {Path(pair.map).name} has no pixel size in its header: the .bild file "
                              "(in Å) cannot be placed on it.")
        level = self.level.value() if keep_view and same_box else vol.level_for_fraction(0.01)
        self.level.blockSignals(True)
        self.level.setSingleStep((vol.maximum - vol.minimum) / 200 or 0.01)
        self.level.setValue(level)
        self.level.blockSignals(False)
        self.surface.set_bild(bild, vol)
        self.surface.set_volume(vol.data, level, reframe=not (keep_view and same_box))
        if vol.voxel:
            self.info.setText(f"{Path(pair.bild).name}: {len(bild)} directions with particles, on "
                              f"{Path(pair.map).name}. Longer and redder: more particles (RELION's colours). "
                              "Drag to rotate, wheel to zoom.")

    def refresh(self) -> None:
        self.scan()
