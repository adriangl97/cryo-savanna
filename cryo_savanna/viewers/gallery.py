# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Image galleries: class averages (2D, or central sections of 3D classes) and particle stacks.

Thumbnails are made only for the items that are on screen, a batch per stack file in the thread
pool, each with its own contrast (mean ± 3σ, as RELION's display). Particles are paged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from qtpy.QtCore import QAbstractListModel, QItemSelectionModel, QModelIndex, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.images.mrc import ImageError, ImageRef, contrast_limits, downsample, read_images, to_uint8
from cryo_savanna.viewers.image_canvas import ImageCanvas
from cryo_savanna.viewers.tasks import TaskError, Tasks


@dataclass
class GalleryItem:
    ref: ImageRef
    label: str
    tooltip: str = ""
    keys: dict[str, float] = field(default_factory=dict)  # sort keys
    payload: object = None  # e.g. the ClassInfo


def make_thumbs(path: str, indices: list[int | None], size: int) -> dict[int | None, np.ndarray]:
    """uint8 thumbnails of several images of one file."""
    real = [i for i in indices if i is not None]
    imgs: dict = read_images(path, real) if real else {}
    if None in indices:  # a single image or a volume (its central section)
        imgs[None] = read_images(path, [None])[None]
    out = {}
    for i, img in imgs.items():
        small = downsample(img, size)[0]
        out[i] = np.ascontiguousarray(to_uint8(small, *contrast_limits(small, 3.0)))
    return out


def to_pixmap(a: np.ndarray) -> QPixmap:
    h, w = a.shape
    return QPixmap.fromImage(QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy())


class GalleryModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[GalleryItem] = []
        self.size = 128
        self.tasks = Tasks(self)
        self._pix: dict[int, QPixmap] = {}
        self._failed: set[int] = set()
        self._queued: set[int] = set()
        self._asked: set[int] = set()
        self._generation = 0
        self._placeholder = QPixmap(self.size, self.size)
        self._placeholder.fill(QColor(70, 70, 70))
        self._flush = QTimer(self)
        self._flush.setSingleShot(True)
        self._flush.setInterval(15)
        self._flush.timeout.connect(self._load_queued)

    def set_items(self, items: list[GalleryItem]) -> None:
        self.beginResetModel()
        self.items = items
        self._forget()
        self.endResetModel()

    def set_size(self, size: int) -> None:
        self.size = size
        self._placeholder = QPixmap(size, size)
        self._placeholder.fill(QColor(70, 70, 70))
        self._forget()
        if self.items:
            self.dataChanged.emit(self.index(0), self.index(len(self.items) - 1), [Qt.DecorationRole])

    def _forget(self) -> None:
        self._generation += 1
        self._pix.clear()
        self._failed.clear()
        self._queued.clear()
        self._asked.clear()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.items)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        it = self.items[row]
        if role == Qt.DisplayRole:
            return it.label
        if role == Qt.ToolTipRole:
            return it.tooltip or str(it.ref)
        if role == Qt.DecorationRole:
            pix = self._pix.get(row)
            if pix is None:
                if row not in self._asked and row not in self._failed:
                    self._asked.add(row)
                    self._queued.add(row)
                    self._flush.start()
                return self._placeholder
            return pix
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignHCenter | Qt.AlignTop)
        return None

    def _load_queued(self) -> None:
        rows, self._queued = sorted(self._queued), set()
        by_file: dict[str, list[int]] = {}
        for r in rows:
            by_file.setdefault(self.items[r].ref.path, []).append(r)
        gen, size = self._generation, self.size
        for path, rs in by_file.items():
            for start in range(0, len(rs), 64):
                chunk = rs[start:start + 64]
                idx = [self.items[r].ref.index for r in chunk]
                self.tasks.run(lambda p=path, i=idx: make_thumbs(p, i, size),
                               lambda res, c=chunk, i=idx: self._on_thumbs(gen, c, i, res))

    def _on_thumbs(self, gen: int, rows: list[int], indices: list, result) -> None:
        if gen != self._generation:
            return
        if isinstance(result, TaskError):
            self._failed.update(rows)
            return
        for r, i in zip(rows, indices):
            a = result.get(i)
            if a is None:
                self._failed.add(r)
                continue
            pix = to_pixmap(a)
            if max(a.shape) < self.size:  # small images (48 px classes) fill the thumbnail
                pix = pix.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._pix[r] = pix
            ix = self.index(r)
            self.dataChanged.emit(ix, ix, [Qt.DecorationRole])

    def loaded(self) -> int:
        return len(self._pix)


class GalleryView(QWidget):
    """Thumbnails with sorting, size, paging and multi-selection; double-click opens one image."""

    selection_changed = Signal()

    PAGE = 1000

    def __init__(self, sort_keys: list[tuple[str, str]] | None = None, paged: bool = False, parent=None):
        super().__init__(parent)
        self.all_items: list[GalleryItem] = []
        self.page = 0
        self.paged = paged
        self.model = GalleryModel(self)
        self.list = QListView()
        self.list.setViewMode(QListView.IconMode)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setUniformItemSizes(True)
        self.list.setLayoutMode(QListView.Batched)
        self.list.setBatchSize(200)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setSpacing(4)
        self.list.setWordWrap(True)
        self.list.setModel(self.model)
        self.list.doubleClicked.connect(self._open)
        self.list.selectionModel().selectionChanged.connect(lambda *_: self.selection_changed.emit())

        self.bar = QHBoxLayout()
        self.bar.setContentsMargins(0, 0, 0, 0)
        self.sort = QComboBox()
        self.sort_keys = [("", "Original order")] + list(sort_keys or [])
        for _, title in self.sort_keys:
            self.sort.addItem(title)
        self.sort.setToolTip("Sort by")
        self.sort.currentIndexChanged.connect(lambda _: self._resort())
        self.descending = QCheckBox("Descending")
        self.descending.toggled.connect(lambda _: self._resort())
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(48, 320)
        self.size.setValue(128)
        self.size.setFixedWidth(120)
        self.size.setToolTip("Thumbnail size")
        self.size.sliderReleased.connect(self._resize)
        self.size.valueChanged.connect(lambda _: None if self.size.isSliderDown() else self._resize())
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.page_label = QLabel()
        self.prev_btn.clicked.connect(lambda: self.set_page(self.page - 1))
        self.next_btn.clicked.connect(lambda: self.set_page(self.page + 1))
        for w in (self.prev_btn, self.page_label, self.next_btn):
            w.setVisible(paged)
        self.bar.addWidget(QLabel("Sort:"))
        self.bar.addWidget(self.sort)
        self.bar.addWidget(self.descending)
        self.bar.addSpacing(12)
        self.bar.addWidget(QLabel("Size:"))
        self.bar.addWidget(self.size)
        self.bar.addSpacing(12)
        self.bar.addWidget(self.prev_btn)
        self.bar.addWidget(self.page_label)
        self.bar.addWidget(self.next_btn)
        self.bar.addStretch(1)
        self.status = QLabel()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(self.bar)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.status)
        self._resize()
        self.selection_changed.connect(self._update_status)
        self.describe: Callable[[list[GalleryItem]], str] = lambda sel: f"{len(sel)} selected" if sel else ""

    def set_items(self, items: list[GalleryItem], keep_page: bool = False) -> None:
        selected = {str(i.ref) for i in self.selected()}
        self.all_items = items
        if not keep_page:
            self.page = 0
        self._resort(selected)

    def _sorted(self) -> list[GalleryItem]:
        key = self.sort_keys[self.sort.currentIndex()][0]
        items = list(self.all_items)
        if key:
            sign = -1 if self.descending.isChecked() else 1  # read once, not for each of a million particles

            def k(it: GalleryItem):
                v = it.keys.get(key)
                return (v is None or not math.isfinite(v), sign * (v or 0))
            items.sort(key=k)
        elif self.descending.isChecked():
            items.reverse()
        return items

    def _resort(self, keep: set | None = None) -> None:
        if keep is None:
            keep = {str(i.ref) for i in self.selected()}
        self._items = self._sorted()
        self.set_page(self.page, keep)

    def set_page(self, page: int, keep: set | None = None) -> None:
        n = len(getattr(self, "_items", []))
        pages = max(1, -(-n // self.PAGE)) if self.paged else 1
        self.page = max(0, min(page, pages - 1))
        shown = self._items[self.page * self.PAGE:(self.page + 1) * self.PAGE] if self.paged else self._items
        self.model.set_items(shown)
        self.page_label.setText(f"{self.page * self.PAGE + 1}–{min(n, (self.page + 1) * self.PAGE)} of {n}" if n else "")
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(self.page < pages - 1)
        if keep:
            sm = self.list.selectionModel()
            for r, it in enumerate(shown):
                if str(it.ref) in keep:
                    sm.select(self.model.index(r), QItemSelectionModel.Select)
        self._update_status()

    def _resize(self) -> None:
        s = self.size.value()
        self.list.setIconSize(QSize(s, s))
        self.list.setGridSize(QSize(s + 12, s + 2 * self.fontMetrics().height() + 10))
        self.model.set_size(s)

    def selected(self) -> list[GalleryItem]:
        if not hasattr(self, "_items"):
            return []
        return [self.model.items[i.row()] for i in self.list.selectionModel().selectedIndexes()]

    def _update_status(self) -> None:
        n = len(self.all_items)
        sel = self.selected()
        extra = self.describe(sel)
        self.status.setText(f"{n} images" + (f"   ·   {extra}" if extra else ""))

    def _open(self, index) -> None:
        it = self.model.items[index.row()]
        dlg = ImageDialog(it.ref, it.label, self)
        dlg.show()


class ImageDialog(QDialog):
    def __init__(self, ref: ImageRef, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — {Path(ref.path).name}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.canvas = ImageCanvas()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.canvas)
        self.resize(560, 600)
        self.canvas.set_image(ref)


__all__ = ["GalleryItem", "GalleryModel", "GalleryView", "ImageDialog", "ImageError", "make_thumbs"]
