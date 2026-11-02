# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Which classes' picks to show: all, or any set of them, each in its own colour.

Used by the Picks views (micrographs and tomograms) when the particles carry more than one class
number (rlnClassNumber), as the particles of Class2D, Class3D and InitialModel jobs do.
"""

from __future__ import annotations

import colorsys

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

PICK_COLOR = (60, 220, 60)  # the colour of picks when they are not coloured by class
# distinct colours for the first classes (on grey micrographs and tomograms); further ones by hue
_PALETTE = [(60, 220, 60), (255, 80, 80), (70, 150, 255), (255, 200, 40), (220, 90, 255), (40, 220, 220),
            (255, 140, 30), (255, 120, 200), (170, 230, 60), (150, 120, 255)]


def class_color(number: int) -> tuple[int, int, int]:
    """A fixed colour per class number, the same in every view."""
    i = max(int(number), 1) - 1
    if i < len(_PALETTE):
        return _PALETTE[i]
    h = (i * 0.618034) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _swatch(color: tuple[int, int, int]) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(*color))
    return QIcon(pm)


class ClassFilter(QPushButton):
    """A "Classes" button whose menu lists the classes to tick; hidden for particles of one class."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoDefault(False)
        self.setToolTip("Show the picks of all classes, or only of the ticked ones")
        self.counts: dict[int, int] = {}

        self.list = QListWidget()
        self.list.setMinimumWidth(180)
        self.list.itemChanged.connect(lambda _: self._on_change())
        self.color_by_class = QCheckBox("Colour by class")
        self.color_by_class.setChecked(True)
        self.color_by_class.toggled.connect(lambda _: self._on_change())
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for text, fn in (("All", lambda: self._tick_all(True)), ("None", lambda: self._tick_all(False)),
                         ("Invert", self._invert)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(6, 6, 6, 6)
        pv.addLayout(buttons)
        pv.addWidget(self.color_by_class)
        pv.addWidget(self.list, 1)
        menu = QMenu(self)
        action = QWidgetAction(menu)  # a list in the menu: ticking classes leaves it open
        action.setDefaultWidget(panel)
        menu.addAction(action)
        self.setMenu(menu)
        self.setVisible(False)
        self._update_text()

    def set_counts(self, counts: dict[int, int]) -> None:
        """The classes and their particles; keeps the ticks of classes that were there before."""
        before = {n: self._item_checked(n) for n in self.counts}
        self.counts = dict(sorted(counts.items()))
        self.list.blockSignals(True)
        self.list.clear()
        for n, c in self.counts.items():
            it = QListWidgetItem(_swatch(class_color(n)), f"Class {n}   ({c:,})")
            it.setData(Qt.UserRole, n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if before.get(n, True) else Qt.Unchecked)
            self.list.addItem(it)
        self.list.setFixedHeight(min(18 * 20, max(4, len(self.counts)) * 20 + 4))
        self.list.blockSignals(False)
        self.setVisible(len(self.counts) > 1)
        self._update_text()

    @property
    def active(self) -> bool:
        return len(self.counts) > 1

    def selected(self) -> set[int] | None:
        """The ticked classes, or None for all of them (or when there are no classes to choose)."""
        if not self.active:
            return None
        ticked = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                  if self.list.item(i).checkState() == Qt.Checked}
        return None if len(ticked) == len(self.counts) else ticked

    def set_selected(self, classes: set[int] | None) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if classes is None or it.data(Qt.UserRole) in classes else Qt.Unchecked)
        self.list.blockSignals(False)
        self._on_change()

    def colors(self, cls: np.ndarray | None) -> list[tuple[int, int, int]] | tuple[int, int, int]:
        """One colour for all picks, or one per pick when they are coloured by class."""
        if cls is None or not self.active or not self.color_by_class.isChecked():
            return PICK_COLOR
        table = {n: class_color(n) for n in np.unique(cls).tolist()}
        return [table[n] for n in cls.tolist()]

    def describe(self) -> str:
        """For the pick counts: which classes are shown, when not all."""
        sel = self.selected()
        if sel is None:
            return ""
        if not sel:
            return "no classes"
        return ("class " if len(sel) == 1 else "classes ") + _ranges(sorted(sel))

    # ------------------------------------------------------------------ internals
    def _item_checked(self, number: int) -> bool:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == number:
                return it.checkState() == Qt.Checked
        return True

    def _tick_all(self, on: bool) -> None:
        self.set_selected(None if on else set())

    def _invert(self) -> None:
        sel = self.selected()
        self.set_selected(set() if sel is None else set(self.counts) - sel)

    def _on_change(self) -> None:
        self._update_text()
        self.changed.emit()

    def _update_text(self) -> None:
        sel = self.selected()
        if sel is None:
            self.setText("Classes: all")
        elif len(sel) <= 4:
            self.setText("Classes: " + (_ranges(sorted(sel)) if sel else "none"))
        else:
            self.setText(f"Classes: {len(sel)} of {len(self.counts)}")


def _ranges(nums: list[int]) -> str:
    """[1, 2, 3, 5] -> "1–3, 5"."""
    out, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}–{n}")
            start = None
    return ", ".join(out)
