# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Project → Disk usage: how much each job, job type and Trash/ take, cleaning jobs, and emptying Trash/.

The project is measured in a thread (project/disk_usage.py), so the window stays alive on a large project.
Cleaning goes through relion_pipeliner (JobActions.clean_jobs), which moves the files to Trash/: the space
they take comes free only when Trash/ is emptied, here (JobActions.empty_trash), for good.
"""

from __future__ import annotations

import threading
import time

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.project import disk_usage as D
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.ui import style
from cryo_savanna.ui.busy import when_done

JOB_COLUMNS = ("Job", "Type", "Status", "Size", "Files", "Gentle clean moves", "Harsh clean moves")
TYPE_COLUMNS = ("Type", "Jobs", "Size", "Files", "Share")
TRASH_COLUMNS = ("In Trash/", "Size", "Files", "What it is")
OTHER_COLUMNS = ("Path", "Size", "Files")
LISTED = 15  # entries named in a confirmation, the rest counted


class _Item(QTableWidgetItem):
    """A cell that sorts by its key (a size sorts as a number, not as text)."""

    def __init__(self, text: str, key=None):
        super().__init__(text)
        self.key = text.lower() if key is None else key
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)
        if isinstance(key, (int, float)):
            self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        if isinstance(other, _Item) and type(self.key) is type(other.key):
            return self.key < other.key
        return super().__lt__(other)


def _size(u: D.Usage | None, none: str = "") -> _Item:
    return _Item(none, -1) if u is None else _Item(D.human(u.bytes), u.bytes)


def _table(columns: tuple[str, ...], stretch: int = 0) -> QTableWidget:
    """A sortable read-only table; column ``stretch`` takes the width left over."""
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.ExtendedSelection)
    t.verticalHeader().setVisible(False)
    header = t.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Interactive)
    header.setStretchLastSection(False)
    header.setSectionResizeMode(stretch, QHeaderView.Stretch)
    return t


def _fill(table: QTableWidget, rows: list[list[QTableWidgetItem]], sort_column: int) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, item in enumerate(row):
            table.setItem(r, c, item)
    table.setSortingEnabled(True)
    table.sortItems(sort_column, Qt.DescendingOrder)
    table.resizeColumnsToContents()


def _listing(names: list[str]) -> str:
    shown = "\n".join(f"  {n}" for n in names[:LISTED])
    return shown + (f"\n  … and {len(names) - LISTED} more" if len(names) > LISTED else "")


class DiskUsageWindow(QDialog):
    """One per main window (JobActions.show_disk_usage); measured again with Refresh and after a change."""

    def __init__(self, actions, parent: QWidget | None = None):
        super().__init__(parent)
        self.actions = actions
        self.setWindowTitle("Disk usage")
        self.setWindowFlag(Qt.Window, True)
        self.resize(980, 620)
        self.usage: D.ProjectUsage | None = None
        self._cancel: threading.Event | None = None

        self.summary = QLabel("Measuring…")
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setWordWrap(True)
        note = QLabel("RELION's Clean does not delete: it moves the files to Trash/, where they take as much space as "
                      "before. The space comes free when Trash/ is emptied (Trash tab).")
        note.setWordWrap(True)
        note.setEnabled(False)  # greyed: an explanation, not a value

        self.jobs = _table(JOB_COLUMNS)
        self.jobs.setToolTip("Double-click a job to select it in the main window")
        self.jobs.itemSelectionChanged.connect(self._update_buttons)
        self.jobs.itemDoubleClicked.connect(self._select_job)
        self.btn_gentle = QPushButton("Clean (gentle)…")
        self.btn_gentle.clicked.connect(lambda: self.clean(False))
        self.btn_harsh = QPushButton("Clean (harsh)…")
        self.btn_harsh.clicked.connect(lambda: self.clean(True))
        jobs_tab = self._tab(self.jobs, [self.btn_gentle, self.btn_harsh])

        self.types = _table(TYPE_COLUMNS)
        self.types.setSelectionMode(QAbstractItemView.NoSelection)

        self.trash = _table(TRASH_COLUMNS, stretch=3)
        self.trash.itemSelectionChanged.connect(self._update_buttons)
        self.btn_empty = QPushButton("Empty selected…")
        self.btn_empty.setToolTip("Delete the selected entries of Trash/ for good")
        self.btn_empty.clicked.connect(lambda: self.empty(False))
        self.btn_empty_all = QPushButton("Empty Trash/…")
        self.btn_empty_all.setToolTip("Delete everything in Trash/ for good")
        self.btn_empty_all.clicked.connect(lambda: self.empty(True))
        trash_tab = self._tab(self.trash, [self.btn_empty, self.btn_empty_all])

        self.other = _table(OTHER_COLUMNS)
        self.other.setSelectionMode(QAbstractItemView.NoSelection)
        self.other.setToolTip("Files of the project that are neither a job of the pipeline nor in Trash/ "
                              "(links, such as the movies of an Import job, count as links only)")

        self.tabs = QTabWidget()
        self.tabs.addTab(jobs_tab, "Jobs")
        self.tabs.addTab(self.types, "By type")
        self.tabs.addTab(trash_tab, "Trash")
        self.tabs.addTab(self.other, "Other files")

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self.btn_refresh)
        bottom.addWidget(close)

        lay = QVBoxLayout(self)
        lay.addWidget(self.summary)
        lay.addWidget(note)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(bottom)
        self._update_buttons()

    @staticmethod
    def _tab(table: QTableWidget, buttons: list[QPushButton]) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(table, 1)
        row = QHBoxLayout()
        for b in buttons:
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    # ------------------------------------------------------------------ measuring
    def refresh(self) -> None:
        project = self.actions.project_dir
        if project is None:
            return
        if self._cancel is not None:
            self._cancel.set()  # an earlier scan still going: its result is not wanted any more
        cancel = self._cancel = threading.Event()
        self.summary.setText(f"Measuring {project}…")
        self.btn_refresh.setEnabled(False)

        def work() -> D.ProjectUsage:
            return D.scan(project, Pipeline.read(project), cancel)

        def done(usage: D.ProjectUsage) -> None:
            if cancel is self._cancel:
                self._cancel = None
                self.show_usage(usage)

        when_done(self, work, done)

    def closeEvent(self, e) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self._cancel = None
        super().closeEvent(e)

    def show_usage(self, u: D.ProjectUsage) -> None:
        self.usage = u
        self.btn_refresh.setEnabled(True)
        total = u.total
        disk = (f" The file system has {D.human(u.disk_free)} free of {D.human(u.disk_total)}."
                if u.disk_total else "")
        gentle = sum(j.gentle.bytes for j in u.jobs if j.gentle)
        harsh = sum(j.harsh.bytes for j in u.jobs if j.harsh)
        self.summary.setText(
            f"<b>{D.human(total.bytes)}</b> in {total.files:,} files: jobs {D.human(u.jobs_total.bytes)}, "
            f"Trash/ <b>{D.human(u.trash_total.bytes)}</b>, other files {D.human(u.other_total.bytes)}.{disk}<br>"
            f"Cleaning every finished job would move {D.human(gentle)} (gentle) or {D.human(harsh)} (harsh) to "
            f"Trash/. Measured {time.strftime('%H:%M:%S', time.localtime(u.scanned_at))}.")

        names = self.actions.type_names
        rows = []
        for j in u.jobs:
            name = _Item(style.job_label(j.name, j.alias), j.name)
            name.setData(Qt.UserRole, j.name)
            if not j.exists:
                name.setToolTip("Its directory is not there")
            harsh_item = _size(j.harsh, "—")
            if j.no_harsh_clean and j.harsh is not None:
                harsh_item.setText(harsh_item.text() + " (NO_HARSH_CLEAN)")
                harsh_item.setToolTip("The job has a NO_HARSH_CLEAN file: harsh cleaning here leaves it out, "
                                      "as RELION's Clean all does")
            gentle_item = _size(j.gentle, "—")
            if j.gentle is None:
                gentle_item.setToolTip("RELION cleans only jobs that finished successfully")
            rows.append([name, _Item(style.type_name(j.type_label, names)), _Item(j.status), _size(j.usage),
                         _Item(f"{j.usage.files:,}", j.usage.files), gentle_item, harsh_item])
        _fill(self.jobs, rows, 3)

        whole = max(total.bytes, 1)
        type_rows = [[_Item(t), _Item(str(n), n), _size(usage), _Item(f"{usage.files:,}", usage.files),
                      _Item(f"{100 * usage.bytes / whole:.1f} %", usage.bytes)]
                     for t, (n, usage) in u.by_type().items()]
        for label, usage in (("Trash/", u.trash_total), ("Other files", u.other_total)):
            type_rows.append([_Item(label, "~" + label), _Item("", -1), _size(usage),
                              _Item(f"{usage.files:,}", usage.files),
                              _Item(f"{100 * usage.bytes / whole:.1f} %", usage.bytes)])
        _fill(self.types, type_rows, 2)

        trash_rows = []
        for t in u.trash:
            item = _Item(t.path)
            item.setData(Qt.UserRole, t.path)
            trash_rows.append([item, _size(t.usage), _Item(f"{t.usage.files:,}", t.usage.files),
                               _Item(self._what(t))])
        _fill(self.trash, trash_rows, 1)

        _fill(self.other, [[_Item(p), _size(usage), _Item(f"{usage.files:,}", usage.files)] for p, usage in u.other],
              1)
        self._update_buttons()

    @staticmethod
    def _what(t: D.TrashEntry) -> str:
        if t.job is None:
            return ""
        job = t.job.rstrip("/")
        if t.deleted_job and t.in_pipeline:
            return f"the earlier run of {job}, overwritten (can be undeleted instead of the current one)"
        if t.deleted_job:
            return f"deleted job {job} (can be undeleted)"
        if t.in_pipeline:
            return f"files cleaned from {job}"
        return f"files of {job}"

    # ------------------------------------------------------------------ state
    def _writable(self) -> bool:
        return not self.actions.readonly and self.actions.project_dir is not None

    def _selected(self, table: QTableWidget) -> list[str]:
        rows = sorted({i.row() for i in table.selectedItems()})
        return [table.item(r, 0).data(Qt.UserRole) for r in rows]

    def _jobs(self, harsh: bool) -> list[D.JobUsage]:
        if self.usage is None:
            return []
        by_name = {j.name: j for j in self.usage.jobs}
        chosen = [by_name[n] for n in self._selected(self.jobs) if n in by_name]
        return [j for j in chosen if (j.harsh if harsh else j.gentle) is not None]

    def _update_buttons(self) -> None:
        has_pipeliner = self.actions.pipeliner is not None and self.actions.pipeliner.available
        why = ("read-only (--readonly)" if self.actions.readonly else
               "" if has_pipeliner else "relion_pipeliner not found: load RELION 5.1")
        can = self._writable() and has_pipeliner
        self.btn_gentle.setEnabled(can and bool(self._jobs(False)))
        self.btn_harsh.setEnabled(can and bool(self._jobs(True)))
        for b in (self.btn_gentle, self.btn_harsh):
            b.setToolTip(why or "Clean the selected jobs that finished successfully (relion_pipeliner)")
        self.btn_empty.setEnabled(self._writable() and bool(self._selected(self.trash)))
        self.btn_empty_all.setEnabled(self._writable() and self.usage is not None and bool(self.usage.trash))

    def _select_job(self, item: QTableWidgetItem) -> None:
        job = self.jobs.item(item.row(), 0).data(Qt.UserRole)
        if job:
            self.actions.select.emit(job)

    # ------------------------------------------------------------------ changes
    def clean(self, harsh: bool) -> None:
        jobs = self._jobs(harsh)
        skipped = [j for j in jobs if harsh and j.no_harsh_clean]
        jobs = [j for j in jobs if j not in skipped]
        if not jobs:
            if skipped:
                QMessageBox.information(self, "Clean", "The selected jobs have a NO_HARSH_CLEAN file: none is "
                                                       "cleaned harshly.")
            return
        kind = "harsh" if harsh else "gentle"
        moves = sum((j.harsh if harsh else j.gentle).bytes for j in jobs)  # type: ignore[union-attr]
        text = (f"Do a {kind} clean of {len(jobs)} job{'s' if len(jobs) > 1 else ''}?\n\n"
                + _listing([style.job_label(j.name, j.alias) for j in jobs])
                + f"\n\nRELION moves {D.human(moves)} to Trash/. They take as much space there until Trash/ is "
                  "emptied, and can be put back by hand until then.")
        if harsh:
            text += "\n\nHarsh cleaning also moves files needed to continue the jobs (e.g. Extract's particles)."
        if skipped:
            text += "\n\nLeft out (NO_HARSH_CLEAN): " + ", ".join(j.name.rstrip("/") for j in skipped)
        if QMessageBox.question(self, "Clean", text) != QMessageBox.Yes:
            return
        self.actions.clean_jobs([j.name for j in jobs], harsh, done=self.refresh)

    def empty(self, everything: bool) -> None:
        if self.usage is None:
            return
        by_path = {t.path: t for t in self.usage.trash}
        paths = list(by_path) if everything else [p for p in self._selected(self.trash) if p in by_path]
        if not paths:
            return
        entries = [by_path[p] for p in paths]
        size = sum(t.usage.bytes for t in entries)
        text = (f"Delete {len(entries)} entr{'ies' if len(entries) > 1 else 'y'} of Trash/ ({D.human(size)}) for "
                f"good?\n\n" + _listing([f"{t.path}  ({D.human(t.usage.bytes)})" for t in entries])
                + "\n\nThis cannot be undone.")
        deleted = [t.job.rstrip("/") for t in entries if t.deleted_job and t.job]
        if deleted:
            text += ("\n\nThese deleted jobs can then no longer be undeleted: "
                     + ", ".join(deleted[:LISTED]) + (" …" if len(deleted) > LISTED else ""))
        box = QMessageBox(QMessageBox.Warning, "Empty Trash/", text, QMessageBox.Yes | QMessageBox.Cancel, self)
        box.setDefaultButton(QMessageBox.Cancel)
        if box.exec() != QMessageBox.Yes:
            return
        self.actions.empty_trash(paths, done=self.refresh)


def show(actions, parent: QWidget, existing: DiskUsageWindow | None) -> DiskUsageWindow:
    """The window (the open one, raised), measured again."""
    w = existing if existing is not None else DiskUsageWindow(actions, parent)
    w.show()
    w.raise_()
    w.activateWindow()
    w.refresh()
    return w

