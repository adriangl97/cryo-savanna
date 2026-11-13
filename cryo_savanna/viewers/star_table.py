# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""STAR tables: a sortable, filterable view of one block, with plots of its columns."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QAbstractProxyModel, QAbstractTableModel, QModelIndex, QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.results.tables import Table, blocks_of, read_compact
from cryo_savanna.viewers.plots import TablePlot
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.stamps import stamp as _stamp

class TableModel(QAbstractTableModel):
    """Columns of a Table (all, or a chosen list with titles); numbers sort as numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table: Table | None = None
        self.columns: list[tuple[str, str]] = []  # (label, title)
        self.extra: dict[str, list] = {}  # computed columns (e.g. picks per micrograph)
        self.row_numbers = True
        self.row_brushes: dict[int, object] = {}  # row -> background (e.g. excluded tilt images)

    def set_table(self, table: Table | None, columns: list[tuple[str, str]] | None = None) -> None:
        self.beginResetModel()
        self.table = table
        self.columns = columns if columns is not None else [(c, c) for c in (table.columns if table else [])]
        self.extra = {}
        self.endResetModel()

    def set_extra(self, label: str, title: str, values: list, position: int | None = None) -> None:
        """Add (at ``position`` among the columns, default last) or update a computed column."""
        if label not in self.extra:
            at = len(self.columns) if position is None else min(position, len(self.columns))
            self.beginInsertColumns(QModelIndex(), at + 1, at + 1)
            self.columns.insert(at, (label, title))
            self.extra[label] = values
            self.endInsertColumns()
        else:
            self.extra[label] = values
            c = 1 + [l for l, _ in self.columns].index(label)
            self.dataChanged.emit(self.index(0, c), self.index(self.rowCount() - 1, c))

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() or self.table is None else len(self.table)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns) + 1

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return "#" if section == 0 else self.columns[section - 1][1]
        return None

    def _value(self, row: int, col: int):
        label = self.columns[col - 1][0]
        if label in self.extra:
            return self.extra[label][row]
        return self.table.data[label][row]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.table is None:
            return None
        r, c = index.row(), index.column()
        if role == Qt.DisplayRole:
            if c == 0:
                return str(r + 1)
            v = self._value(r, c)
            if v is None:
                return ""
            label = self.columns[c - 1][0]
            if label in ("rlnMicrographName", "rlnMicrographMovieName"):
                return Path(v).name
            if label not in self.extra and self.table.numeric(label) is not None:
                return _number(v)
            return str(v)
        if role == Qt.BackgroundRole:
            return self.row_brushes.get(r)
        if role == Qt.ToolTipRole and c > 0:
            return str(self._value(r, c))
        if role == Qt.TextAlignmentRole and c > 0:
            label = self.columns[c - 1][0]
            if label in self.extra or self.table.numeric(label) is not None:
                return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


    # -- for RowFilter: whole columns at once, so sorting and filtering a million rows stay fast --
    def sort_key(self, col: int) -> np.ndarray:
        """Column ``col`` (0 = row number) as one array to sort by: floats (NaN where there is no
        number), or text."""
        n = self.rowCount()
        if col <= 0 or self.table is None:
            return np.arange(n, dtype=np.float64)
        label = self.columns[col - 1][0]
        if label in self.extra:
            vals = self.extra[label]
            try:
                return np.array([np.nan if v is None or v == "" else float(v) for v in vals], dtype=np.float64)
            except (TypeError, ValueError):
                return np.array(["" if v is None else str(v) for v in vals], dtype=str)
        num = self.table.numeric(label)
        if num is not None:
            return np.asarray(num, dtype=np.float64)
        return np.array([str(v) for v in self.table.data[label]], dtype=str)

    def only_numbers(self, col: int) -> bool:
        """Whether column ``col`` shows nothing but numbers (the row numbers, and the columns
        read_compact kept as floats): a filter text with a letter in it cannot be there."""
        if col <= 0 or self.table is None:
            return True
        label = self.columns[col - 1][0]
        vals = None if label in self.extra else self.table.data.get(label)
        return isinstance(vals, np.ndarray) and vals.dtype.kind == "f"

    def column_text(self, col: int) -> list[str]:
        """The text shown in column ``col`` for every row (what a filter looks for)."""
        if col <= 0 or self.table is None:
            return [str(r + 1) for r in range(self.rowCount())]
        label = self.columns[col - 1][0]
        vals = self.extra[label] if label in self.extra else self.table.data[label]
        if label in ("rlnMicrographName", "rlnMicrographMovieName"):
            return [_file_name(str(v)) for v in vals]
        if label not in self.extra and self.table.numeric(label) is not None:
            if isinstance(vals, np.ndarray) and vals.dtype == np.float64:  # read_compact: already numbers
                return _numbers(vals)
            return [_number(v) for v in vals]
        return ["" if v is None else str(v) for v in vals]


def _number(text: str) -> str:
    """A STAR number as RELION's GUI would show it: no trailing zeros ("-60", "37104.8", "0.885")."""
    try:
        x = float(text)
    except ValueError:
        return text
    if not math.isfinite(x):  # RELION writes inf and nan too (e.g. a CTF fit without a limit)
        return str(text).strip() if isinstance(text, str) else str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6g}"


def _file_name(path: str) -> str:
    """Path(path).name, without making a Path of each of a million micrograph names."""
    name = path.rsplit("/", 1)[-1]
    return name if name and name != "." and os.sep == "/" else Path(path).name


def _whole(x: float) -> str:
    return str(int(x)) if -1e15 < x < 1e15 and x == int(x) else f"{x:.6g}"


def _numbers(values: np.ndarray) -> list[str]:
    """_number() of every value of a column of floats, in one pass. A column with few different values
    (class numbers, optics groups, a constant B-factor: most columns of a particle file) is written out
    once per value, not once per row."""
    n = len(values)
    if n > 4096 and len(np.unique(values[:: n // 2048])) <= 512:
        distinct, rows = np.unique(values, return_inverse=True)
        texts = np.empty(len(distinct), dtype=object)
        texts[:] = [_whole(x) for x in distinct.tolist()]
        return texts[rows.reshape(-1)].tolist()
    return [_whole(x) for x in values.tolist()]


FILTER_AT_ONCE_ROWS = 20_000  # tables up to this size are filtered at every letter typed
FILTER_PAUSE_MS = 250
_NUMBER_CHARS = frozenset("0123456789.-+einfa")  # what _number() can show: digits, exponents, inf, nan


class RowFilter(QAbstractProxyModel):
    """Sorts and filters the rows of a TableModel with numpy, a whole column at a time.

    QSortFilterProxyModel compares rows one pair at a time through Python, which takes tens of
    seconds for a few hundred thousand particles (and froze the window meanwhile). Here a sort is
    one argsort, and a filter one pass over the columns' text. mapToSource / mapFromSource keep
    their meaning: a source row is a row of the table. Rows are numbered for the view as the sort
    puts them; NaN and missing values come first ascending, last descending (as before).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = np.empty(0, np.int64)  # view row -> source row
        self._inv = np.empty(0, np.int64)  # source row -> view row (-1: filtered out)
        self._order = np.empty(0, np.int64)  # every source row, in the sort order
        self._sort_col, self._descending = -1, False
        self._needle = ""
        self._typed = ""
        self._pause = QTimer(self)  # big tables: filtered once the typing pauses, not at every letter
        self._pause.setSingleShot(True)
        self._pause.setInterval(FILTER_PAUSE_MS)
        self._pause.timeout.connect(self._filter_typed)

    # -- the source -------------------------------------------------------------------------
    def setSourceModel(self, model) -> None:
        old = self.sourceModel()
        if old is not None:
            for sig, slot in self._connections(old):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self.beginResetModel()
        super().setSourceModel(model)
        for sig, slot in self._connections(model):
            sig.connect(slot)
        self._order = np.arange(model.rowCount(), dtype=np.int64)
        self._sort_order()
        self._apply_filter()
        self.endResetModel()

    def _connections(self, m):
        return [(m.modelAboutToBeReset, self.beginResetModel), (m.modelReset, self._on_reset),
                (m.columnsAboutToBeInserted, self._on_columns_about), (m.columnsInserted, self._on_columns),
                (m.dataChanged, self._on_data), (m.headerDataChanged, self.headerDataChanged)]

    def _on_reset(self) -> None:
        self._order = np.arange(self.sourceModel().rowCount(), dtype=np.int64)
        self._sort_order()
        self._apply_filter()
        self.endResetModel()

    def _on_columns_about(self, parent, first: int, last: int) -> None:
        self.beginInsertColumns(QModelIndex(), first, last)
        if self._sort_col >= first:
            self._sort_col += last - first + 1

    def _on_columns(self, *args) -> None:
        self.endInsertColumns()

    def _on_data(self, top_left, bottom_right, roles=()) -> None:
        n = len(self._rows)
        if n:
            self.dataChanged.emit(self.index(0, top_left.column()), self.index(n - 1, bottom_right.column()), roles)
        if top_left.column() <= self._sort_col <= bottom_right.column() or self._needle:
            self._relayout(refilter=bool(self._needle))  # the values sorted or filtered by changed

    # -- QAbstractProxyModel ----------------------------------------------------------------
    def index(self, row: int, column: int, parent=QModelIndex()):
        if parent.isValid() or not (0 <= row < len(self._rows)) or not (0 <= column < self.columnCount()):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, child=None):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        m = self.sourceModel()
        return 0 if parent.isValid() or m is None else m.columnCount()

    def mapToSource(self, proxy_index):
        if not proxy_index.isValid() or self.sourceModel() is None or proxy_index.row() >= len(self._rows):
            return QModelIndex()
        return self.sourceModel().index(int(self._rows[proxy_index.row()]), proxy_index.column())

    def mapFromSource(self, source_index):
        if not source_index.isValid() or source_index.row() >= len(self._inv):
            return QModelIndex()
        row = int(self._inv[source_index.row()])
        return self.index(row, source_index.column()) if row >= 0 else QModelIndex()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        m = self.sourceModel()
        if m is None or orientation != Qt.Horizontal:
            return None
        return m.headerData(section, orientation, role)

    # -- sorting ----------------------------------------------------------------------------
    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        self._sort_col, self._descending = column, order == Qt.DescendingOrder
        self._relayout(refilter=False)

    def _sort_order(self) -> None:
        m = self.sourceModel()
        n = m.rowCount() if m is not None else 0
        if m is None or self._sort_col < 0 or n == 0:
            self._order = np.arange(n, dtype=np.int64)
            return
        key = m.sort_key(self._sort_col)
        if key.dtype.kind == "f":
            key = np.where(np.isfinite(key), key, -np.inf)  # no value: first ascending, last descending
            self._order = np.argsort(-key if self._descending else key, kind="stable")
        else:
            order = np.argsort(key, kind="stable")
            self._order = order[::-1].copy() if self._descending else order

    # -- filtering --------------------------------------------------------------------------
    def setFilterFixedString(self, text: str) -> None:
        """Show only the rows with ``text`` in them. A table of many rows takes a moment to look
        through: it is filtered when the typing pauses (flush_filter() does it at once)."""
        self._typed = text.lower()
        m = self.sourceModel()
        if m is not None and m.rowCount() > FILTER_AT_ONCE_ROWS:
            self._pause.start()
        else:
            self._filter_typed()

    def flush_filter(self) -> None:
        self._filter_typed()

    def _filter_typed(self) -> None:
        self._pause.stop()
        if self._typed != self._needle:
            self._needle = self._typed
            self._relayout(refilter=True)

    def _matches(self) -> np.ndarray | None:
        """Source rows shown (a boolean per row), or None for all of them."""
        m = self.sourceModel()
        if not self._needle or m is None:
            return None
        n, needle = m.rowCount(), self._needle
        hit = np.zeros(n, bool)
        # a text that cannot be part of a number ("mic0012", "job005/") is not looked for in the
        # number columns: most columns of a particle file, and most of the time a pass takes
        in_numbers = all(ch in _NUMBER_CHARS for ch in self._needle)
        only_numbers = getattr(m, "only_numbers", None)
        for c in range(m.columnCount()):
            if not in_numbers and only_numbers is not None and only_numbers(c):
                continue
            # the texts as they are: a numpy array of them is as wide as the longest (a million names of
            # 60 letters: 240 MB for one column), and is no faster to look through
            hit |= np.fromiter((needle in s.lower() for s in m.column_text(c)), bool, n)
        return hit

    def _apply_filter(self) -> None:
        keep = self._matches()
        self._rows = self._order if keep is None else self._order[keep[self._order]]
        self._inv = np.full(len(self._order), -1, np.int64)
        self._inv[self._rows] = np.arange(len(self._rows))

    def _relayout(self, refilter: bool) -> None:
        """Sort again (and filter again), keeping the selection and the current row with their rows
        (a row the filter now hides drops out of the selection). One layout change, also when the
        number of rows changes: views follow it, and it costs one pass instead of one signal per run
        of rows."""
        if self.sourceModel() is None:
            return
        self.layoutAboutToBeChanged.emit()
        persistent = self.persistentIndexList()
        sources = [int(self._rows[i.row()]) if 0 <= i.row() < len(self._rows) else -1 for i in persistent]
        self._sort_order()
        if refilter:
            self._apply_filter()
        else:  # the same rows, in the new order
            shown = np.zeros(len(self._order), bool)
            shown[self._rows] = True
            self._rows = self._order[shown[self._order]]
            self._inv = np.full(len(self._order), -1, np.int64)
            self._inv[self._rows] = np.arange(len(self._rows))
        for idx, src in zip(persistent, sources):
            row = int(self._inv[src]) if src >= 0 else -1
            self.changePersistentIndex(idx, self.index(row, idx.column()) if row >= 0 else QModelIndex())
        self.layoutChanged.emit()


def make_table_view() -> QTableView:
    v = QTableView()
    v.setSortingEnabled(True)
    v.setSelectionBehavior(QAbstractItemView.SelectRows)
    v.setSelectionMode(QAbstractItemView.SingleSelection)
    v.setAlternatingRowColors(True)
    v.verticalHeader().setVisible(False)
    v.verticalHeader().setDefaultSectionSize(v.fontMetrics().height() + 6)
    v.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    v.horizontalHeader().setStretchLastSection(True)
    v.sortByColumn(0, Qt.AscendingOrder)
    return v


class StarTableView(QWidget):
    """Any STAR file: choose a block, sort, filter rows, plot columns."""

    row_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path: Path | None = None
        self.tasks = Tasks(self)
        self.block = QComboBox()
        self.block.currentIndexChanged.connect(lambda _: self._load_block())
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter rows")
        self.filter.setClearButtonEnabled(True)
        self.plot_btn = QPushButton("Plot columns")
        self.plot_btn.setCheckable(True)
        self.plot_btn.toggled.connect(self._toggle_plot)
        self.info = QLabel()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(QLabel("Block:"))
        bar.addWidget(self.block)
        bar.addWidget(self.filter, 1)
        bar.addWidget(self.plot_btn)
        bar.addWidget(self.info)
        self.model = TableModel(self)
        self.proxy = RowFilter(self)
        self.proxy.setSourceModel(self.model)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.view = make_table_view()
        self.view.setModel(self.proxy)
        self.view.selectionModel().currentRowChanged.connect(
            lambda cur, _: self.row_selected.emit(self.proxy.mapToSource(cur).row()) if cur.isValid() else None)
        self.plot = TablePlot()
        self.plot.row_clicked.connect(self.select_row)
        self.plot.hide()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.plot)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self._stamp = None

    def set_file(self, path: Path) -> None:
        """Show a STAR file. Its block names are found in the background too: that reads the whole
        file, a long wait for a big particle file over a network file system."""
        self.path = path = Path(path)
        self._stamp = _stamp(self.path)
        self.info.setText("Reading…")
        self.tasks.run(lambda: blocks_of(path), lambda names: self._on_blocks(path, names), channel="blocks")

    def _on_blocks(self, path: Path, names) -> None:
        if path != self.path:
            return
        if isinstance(names, TaskError):
            self.info.setText(str(names))
            return
        keep = self.block.currentText()
        self.block.blockSignals(True)
        self.block.clear()
        self.block.addItems([n or "(unnamed)" for n in names])
        i = self.block.findText(keep)
        if i < 0:  # the main table: the biggest-sounding one
            i = next((k for k, n in enumerate(names) if n in ("particles", "micrographs", "movies")), len(names) - 1)
        self.block.setCurrentIndex(max(0, i))
        self.block.blockSignals(False)
        self._load_block()

    def refresh(self) -> None:
        if self.path is not None and _stamp(self.path) != self._stamp:
            self.set_file(self.path)

    def _load_block(self) -> None:
        if self.path is None or self.block.count() == 0:
            return
        name = self.block.currentText()
        name = "" if name == "(unnamed)" else name
        path = self.path
        self.info.setText("Reading…")
        self.tasks.run(lambda: read_compact(path, name), self._on_table, channel="table")

    def _on_table(self, table) -> None:
        if isinstance(table, TaskError) or table is None:
            self.info.setText(str(table) if table is not None else "empty block")
            self.model.set_table(None)
            return
        self.model.set_table(table)
        self.view.resizeColumnsToContents()
        self.info.setText(f"{len(table)} rows × {len(table.columns)} columns")
        if self.plot.isVisible():
            self.plot.set_table(table)

    def _toggle_plot(self, on: bool) -> None:
        self.plot.setVisible(on)
        if on and self.model.table is not None:
            self.plot.set_table(self.model.table)

    def select_row(self, row: int) -> None:
        src = self.model.index(row, 0)
        idx = self.proxy.mapFromSource(src)
        if idx.isValid():
            self.view.selectRow(idx.row())
            self.view.scrollTo(idx)


