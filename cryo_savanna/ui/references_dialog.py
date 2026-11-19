# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The reference maps of a 3D classification: the maps of the pipeline and files, ticked. The ticked ones are at
the top, numbered, in the order of their classes (project/references.py)."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

PATH_ROLE = Qt.UserRole + 1


class ReferencesDialog(QDialog):
    def __init__(self, parent, maps: list[str], nodes: list[tuple[str, str]], start_dir: str,
                 relative: Callable[[str], str], file_filter: str = ""):
        """``maps``: the ones chosen now, in order; ``nodes``: (map, text shown) of the pipeline's maps, newest
        first; ``relative``: a chosen file as the form gives it (relative to the project)."""
        super().__init__(parent)
        self.setWindowTitle("Reference maps")
        self.start_dir, self.relative, self.file_filter = start_dir, relative, file_filter
        self.texts: dict[str, str] = dict(nodes)  # map -> what is shown for it
        self.others: list[str] = [m for m, _ in nodes]  # the maps not ticked are listed in this order
        self.order: list[str] = []  # the ticked ones: class 1, 2, ...
        for m in maps:
            self.texts.setdefault(m, m)
            if m not in self.order:
                self.order.append(m)
        self._filling = False

        intro = QLabel("Tick the maps to start from: each one starts a class. The ticked maps are at the top, "
                       "numbered by their class; drag them or use Up / Down to change the order. <i>Number of "
                       "classes</i> is set to how many are ticked.")
        intro.setWordWrap(True)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(lambda _t: self._apply_filter())
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setMinimumSize(520, 280)
        self.list.itemChanged.connect(self._on_ticked)
        self.list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._on_dragged))

        add = QPushButton("Add files…")
        add.setToolTip("Maps that are not outputs of the pipeline's jobs")
        add.clicked.connect(self._add_files)
        up, down = QPushButton("Up"), QPushButton("Down")
        up.setToolTip("An earlier class for the selected ticked map")
        down.setToolTip("A later class for the selected ticked map")
        up.clicked.connect(lambda: self.move(-1))
        down.clicked.connect(lambda: self.move(1))
        side = QVBoxLayout()
        for b in (add, up, down):
            side.addWidget(b)
        side.addStretch(1)
        middle = QHBoxLayout()
        middle.addWidget(self.list, 1)
        middle.addLayout(side)

        self.summary = QLabel()
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(self.filter)
        lay.addLayout(middle, 1)
        lay.addWidget(self.summary)
        lay.addWidget(self.buttons)
        self._fill()

    # ------------------------------------------------------------------ the list
    def _fill(self, current: str | None = None) -> None:
        """The list as the model has it: the ticked maps (numbered), then the others."""
        self._filling = True
        self.list.clear()
        for i, m in enumerate(self.order, start=1):
            self._item(m, f"{i}.  {self.texts[m]}", True)
        for m in self.others:
            if m not in self.order:
                self._item(m, self.texts[m], False)
        if current is not None:
            row = next((i for i in range(self.list.count()) if self.list.item(i).data(PATH_ROLE) == current), -1)
            self.list.setCurrentRow(row)
        self._filling = False
        self._apply_filter()
        n = len(self.order)
        if n >= 2:
            text = f"{n} maps ticked: {n} classes, the first from map 1, and so on."
        elif n == 1:
            text = "1 map ticked: it is the reference map, as usual (Number of classes stays as it is)."
        else:
            text = "No map ticked: the field is cleared."
        self.summary.setText(text)

    def _item(self, path: str, text: str, ticked: bool) -> None:
        item = QListWidgetItem(text)
        item.setData(PATH_ROLE, path)
        item.setToolTip(path)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                      | (Qt.ItemIsDragEnabled if ticked else Qt.NoItemFlags))
        item.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
        self.list.addItem(item)

    def _apply_filter(self) -> None:
        t = self.filter.text().strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(t) and t not in f"{it.text()} {it.data(PATH_ROLE)}".lower())

    # ------------------------------------------------------------------ changes
    def maps(self) -> list[str]:
        """The ticked maps, in the order of their classes."""
        return list(self.order)

    def tick(self, path: str, on: bool = True) -> None:
        """Tick a map (the next class), or untick it; a file not in the list is added."""
        if on and path not in self.order:
            self.texts.setdefault(path, path)
            if path not in self.others:
                self.others.append(path)
            self.order.append(path)
        elif not on and path in self.order:
            self.order.remove(path)
        else:
            return
        self._fill(current=path)

    def move(self, step: int) -> None:
        """The selected ticked map ``step`` classes earlier (-1) or later (+1)."""
        it = self.list.currentItem()
        path = it.data(PATH_ROLE) if it is not None else None
        if path not in self.order:
            return
        i = self.order.index(path)
        j = i + step
        if not 0 <= j < len(self.order):
            return
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self._fill(current=path)

    def _on_ticked(self, item: QListWidgetItem) -> None:
        if self._filling:
            return
        path, on = item.data(PATH_ROLE), item.checkState() == Qt.Checked
        QTimer.singleShot(0, lambda: self.tick(path, on))  # (the list is filled again: not from its own signal)

    def _on_dragged(self) -> None:
        """A ticked map dragged to another place: the classes follow the list; dropped among the others, the
        place it had is kept."""
        dragged = [self.list.item(i).data(PATH_ROLE) for i in range(self.list.count())
                   if self.list.item(i).data(PATH_ROLE) in self.order]
        if dragged != self.order:
            self.order = dragged
        self._fill(current=self.list.currentItem().data(PATH_ROLE) if self.list.currentItem() else None)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Reference maps", self.start_dir, self.file_filter)
        for p in paths:
            self.tick(self.relative(p))
