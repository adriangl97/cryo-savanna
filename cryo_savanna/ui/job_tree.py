# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Job lineage tree: each job sits under the job that produced its primary input."""

from __future__ import annotations

import time

from qtpy.QtCore import QModelIndex, QSortFilterProxyModel, Qt, Signal
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QTreeView

from cryo_savanna.project.snapshot import JobInfo, ProjectSnapshot
from cryo_savanna.project.summary import JobSummary
from cryo_savanna.ui import style

NAME_ROLE = Qt.UserRole + 1  # job name ("Class2D/job005/") on column 0
SORT_ROLE = Qt.UserRole + 2
COLUMNS = ("Job", "Type", "Status", "Queue", "Summary", "Modified", "Also uses")
COL_JOB, COL_TYPE, COL_STATUS, COL_QUEUE, COL_SUMMARY, COL_MODIFIED, COL_ALSO = range(len(COLUMNS))

STATUS_FILTERS = {
    "All statuses": None,
    "Active (running or scheduled)": {"Running", "Scheduled"},
    "Running": {"Running"},
    "Scheduled": {"Scheduled"},
    "Succeeded": {"Succeeded"},
    "Failed or aborted": {"Failed", "Aborted"},
}


class JobFilterProxy(QSortFilterProxyModel):
    """Text and status filter. Ancestors of matching jobs stay visible so the lineage is readable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRecursiveFilteringEnabled(True)
        self.setSortRole(SORT_ROLE)
        self._text = ""
        self._statuses: set[str] | None = None
        self._hide_dead = False

    def set_filter(self, text: str, statuses: set[str] | None, hide_failed: bool) -> None:
        def change() -> None:
            self._text = text.strip().lower()
            self._statuses = statuses
            self._hide_dead = hide_failed
        self.refilter(change)

    def refilter(self, change=None) -> None:
        """Re-run the filter, around ``change`` (Qt >= 6.9 wants begin/endFilterChange, Qt5 has neither)."""
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()
            if change:
                change()
            self.endFilterChange()
        else:
            if change:
                change()
            self.invalidateFilter()

    def is_active(self) -> bool:
        return bool(self._text) or self._statuses is not None or self._hide_dead

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        m = self.sourceModel()
        name_idx = m.index(row, COL_JOB, parent)
        status = m.index(row, COL_STATUS, parent).data(SORT_ROLE) or ""
        if self._hide_dead and status in ("Failed", "Aborted"):
            return False
        if self._statuses is not None and status not in self._statuses:
            return False
        if self._text:
            hay = " ".join(str(m.index(row, c, parent).data(Qt.DisplayRole) or "")
                           for c in (COL_JOB, COL_TYPE, COL_QUEUE))
            hay += " " + str(name_idx.data(NAME_ROLE) or "")
            if self._text not in hay.lower():
                return False
        return True


class JobTreeView(QTreeView):
    job_selected = Signal(str)

    def __init__(self, type_names: dict[str, str], parent=None):
        super().__init__(parent)
        self.type_names = type_names
        self.source = QStandardItemModel(0, len(COLUMNS), self)
        self.source.setHorizontalHeaderLabels(COLUMNS)
        self.proxy = JobFilterProxy(self)
        self.proxy.setSourceModel(self.source)
        self.setModel(self.proxy)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSortingEnabled(True)
        self.sortByColumn(COL_JOB, Qt.AscendingOrder)
        self.setAnimated(False)  # no animations: they cost bandwidth over X11
        h = self.header()
        h.setStretchLastSection(True)
        for c in (COL_JOB, COL_STATUS, COL_QUEUE, COL_SUMMARY):  # never cut an alias, a status or a summary
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.setColumnHidden(COL_QUEUE, True)  # shown once SLURM is found
        self.queue = None  # slurm.QueueState
        self._items: dict[str, list[QStandardItem]] = {}
        self._structure: tuple | None = None
        self._collapsed: set[str] = set()
        self._snapshot: ProjectSnapshot | None = None
        self.summaries: dict[str, JobSummary] = {}
        self.expanded.connect(lambda i: self._collapsed.discard(self._name(i)))
        self.collapsed.connect(lambda i: self._collapsed.add(self._name(i)))
        self.selectionModel().currentChanged.connect(self._on_current)
        self.clicked.connect(self._on_clicked)

    # ------------------------------------------------------------------ public
    def set_snapshot(self, snap: ProjectSnapshot) -> None:
        self._snapshot = snap
        key = snap.structure_key()
        if key == self._structure:
            for name, info in snap.jobs.items():
                self._fill_row(self._items[name], info, snap)
            self.proxy.refilter()
            return
        selected = self.current_job()
        scroll = self.verticalScrollBar().value()
        first = self._structure is None
        self.source.removeRows(0, self.source.rowCount())
        self._items.clear()

        root = self.source.invisibleRootItem()
        stack = [(r, root) for r in reversed(snap.roots)]
        while stack:  # no recursion: a chain of jobs can be longer than Python's recursion limit
            name, parent = stack.pop()
            info = snap.jobs[name]
            row = [QStandardItem() for _ in COLUMNS]
            self._fill_row(row, info, snap)
            parent.appendRow(row)
            self._items[name] = row
            stack.extend((child, row[0]) for child in reversed(info.children))
        self._structure = key
        self.blockSignals(True)  # keep the user's collapsed set while re-expanding
        self.expandAll()
        for name in self._collapsed & self._items.keys():
            self.collapse(self.proxy.mapFromSource(self._items[name][0].index()))
        self.blockSignals(False)
        if first:
            for c in (COL_TYPE, COL_MODIFIED):
                self.resizeColumnToContents(c)
        if selected and selected in self._items:
            self.select_job(selected, scroll=False)
        self.verticalScrollBar().setValue(scroll)

    def set_queue(self, state) -> None:
        """Show each job's place in the SLURM queue (a slurm.QueueState)."""
        self.queue = state
        self.setColumnHidden(COL_QUEUE, not state.available)
        for name, row in self._items.items():
            self._fill_queue(row, name)
        if self.proxy.is_active():
            self.proxy.refilter()

    def _fill_queue(self, row: list[QStandardItem], name: str) -> None:
        text, details, kind = style.queue_entry(self.queue, name)
        item = row[COL_QUEUE]
        if item.text() != text:
            item.setText(text)
            item.setData(text, SORT_ROLE)
        item.setToolTip(details)
        item.setData(style.QUEUE_COLORS[kind] if kind else None, Qt.ForegroundRole)  # None: the palette's

    def refresh_times(self) -> None:
        if self._snapshot is None:
            return
        now = time.time()
        for name, row in self._items.items():
            info = self._snapshot.jobs[name]
            row[COL_MODIFIED].setText(style.ago(info.modified, now))
            if info.status == "Running":
                row[COL_STATUS].setText(self._status_text(info, now))

    def set_summaries(self, summaries: dict[str, JobSummary]) -> None:
        self.summaries.update(summaries)
        if self._snapshot is None:
            return
        for name, s in summaries.items():
            row = self._items.get(name)
            if row is not None:
                self._fill_summary(row, self._snapshot.jobs[name], s)

    def clear_summaries(self) -> None:
        self.summaries.clear()

    def current_job(self) -> str | None:
        idx = self.currentIndex()
        return self._name(idx) if idx.isValid() else None

    def select_job(self, name: str, scroll: bool = True) -> bool:
        row = self._items.get(name)
        if row is None:
            return False
        idx = self.proxy.mapFromSource(row[0].index())
        if not idx.isValid():  # filtered out
            return False
        self.setCurrentIndex(idx)
        if scroll:
            self.scrollTo(idx)
        return True

    def set_filter(self, text: str, statuses: set[str] | None, hide_failed: bool) -> None:
        self.proxy.set_filter(text, statuses, hide_failed)
        if self.proxy.is_active():
            self.expandAll()

    def visible_jobs(self) -> list[str]:
        out: list[str] = []
        stack = [QModelIndex()]
        while stack:  # depth first, in the list's order, without recursion (long chains of jobs)
            parent = stack.pop()
            kids = [self.proxy.index(r, 0, parent) for r in range(self.proxy.rowCount(parent))]
            if parent.isValid():
                out.append(self._name(parent))
            stack.extend(reversed(kids))
        return out

    # ------------------------------------------------------------------ internals
    def _name(self, idx: QModelIndex) -> str:
        return idx.sibling(idx.row(), COL_JOB).data(NAME_ROLE)

    def _fill_row(self, row: list[QStandardItem], info: JobInfo, snap: ProjectSnapshot) -> None:
        pr = info.process
        job = row[COL_JOB]
        job.setText(style.job_label(pr.name, pr.alias))
        job.setData(pr.name, NAME_ROLE)
        job.setData(pr.job_number or 0, SORT_ROLE)
        job.setIcon(style.status_icon(info.status, info.pending_update))
        job.setToolTip(pr.name + (f"\nalias: {pr.alias}" if pr.has_alias else ""))
        t = row[COL_TYPE]
        t.setText(style.job_type(info, self.type_names))
        t.setData(t.text(), SORT_ROLE)
        t.setToolTip(pr.type_label)
        s = row[COL_STATUS]
        s.setData(info.status, SORT_ROLE)
        s.setForeground(style.status_color(info.status))
        s.setToolTip(style.status_text(info.status, info.pending_update))
        self._fill_summary(row, info, self.summaries.get(pr.name))
        self._fill_queue(row, pr.name)
        m = row[COL_MODIFIED]
        m.setText(style.ago(info.modified))
        m.setData(info.modified or 0.0, SORT_ROLE)
        m.setToolTip(style.timestamp(info.modified))
        a = row[COL_ALSO]
        labels = [style.job_label(p, snap.jobs[p].process.alias) if p in snap.jobs else p for p in info.other_parents]
        a.setText(", ".join(labels))
        a.setData(a.text(), SORT_ROLE)
        a.setToolTip("Other inputs come from:\n" + "\n".join(info.other_parents) if labels else "")

    def _status_text(self, info: JobInfo, now: float | None = None) -> str:
        text = info.status + ("*" if info.pending_update else "")
        s = self.summaries.get(info.name)
        if info.status == "Running" and s is not None and s.progress(now):
            text += " · " + s.progress(now)
        return text

    def _fill_summary(self, row: list[QStandardItem], info: JobInfo, s: JobSummary | None) -> None:
        row[COL_STATUS].setText(self._status_text(info))
        item = row[COL_SUMMARY]
        item.setText(s.text() if s is not None else "")
        item.setData(item.text(), SORT_ROLE)
        item.setToolTip("\n".join(s.lines()) if s is not None else "")

    def _on_current(self, cur: QModelIndex, _prev: QModelIndex) -> None:
        if cur.isValid():
            self.job_selected.emit(self._name(cur))

    def _on_clicked(self, idx: QModelIndex) -> None:
        # Clicking "Also uses" jumps to the first of those jobs
        if idx.column() == COL_ALSO and self._snapshot is not None:
            info = self._snapshot.jobs.get(self._name(idx))
            if info and info.other_parents:
                self.select_job(info.other_parents[0])
