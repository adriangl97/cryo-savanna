# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Every flag of the program behind a job's "Additional arguments:" box, to tick and fill in.

The box stays the one source of truth: ticking a flag or typing a value rewrites the box (leaving
whatever else is typed there alone), and typing in the box ticks the flags it names. Flags that the
job's own options already put on the command line show that value; RELION uses the last value given,
so setting one here overrides it.
"""

from __future__ import annotations

import html

from qtpy.QtCore import QObject, QRunnable, Qt, Signal
from qtpy.QtGui import QBrush, QFont, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.jobdefs.flags import Flag, FlagSource, ProgramFlags, parse_args, set_arg

COL_FLAG, COL_VALUE, COL_DEFAULT, COL_HELP = range(4)
FLAG_ROLE = Qt.UserRole + 1


class _FlagsSignals(QObject):
    done = Signal(str, object)  # program, ProgramFlags | None


class FlagsTask(QRunnable):
    """Run a program's --help off the GUI thread (it can take a moment on a slow file system)."""

    def __init__(self, source: FlagSource, program: str):
        super().__init__()
        self.source, self.program = source, program
        self.signals = _FlagsSignals()

    def run(self) -> None:
        try:
            result = self.source.flags(self.program)
        except Exception:  # noqa: BLE001 - a broken --help must not take the GUI down
            result = None
        self.signals.done.emit(self.program, result)


class FlagPanel(QWidget):
    edited = Signal(str)  # the new additional-arguments text
    flag_focused = Signal(object)  # Flag

    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = ""
        self.flags: ProgramFlags | None = None
        self.job_flags: dict[str, str] = {}
        self.job_flags_from = ""
        self.text = ""
        self.items: dict[str, QTreeWidgetItem] = {}
        self._syncing = False
        self._search = ""

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setTextFormat(Qt.RichText)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setVisible(False)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter flags")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(lambda _: self._apply_filter())
        self.only_added = QCheckBox("Only added")
        self.only_added.toggled.connect(lambda _: self._apply_filter())

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Flag", "Value", "Default", "Description"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setEditTriggers(QTreeWidget.NoEditTriggers)  # only the Value column, see _on_clicked
        self.tree.setMinimumHeight(320)
        hdr = self.tree.header()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(COL_FLAG, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_VALUE, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_DEFAULT, QHeaderView.Interactive)
        self.tree.setColumnWidth(COL_VALUE, 130)
        self.tree.setColumnWidth(COL_DEFAULT, 120)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemDoubleClicked.connect(self._on_clicked)
        self.tree.currentItemChanged.connect(self._on_current)

        bar = QHBoxLayout()
        bar.addWidget(self.filter, 1)
        bar.addWidget(self.only_added)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(self.header)
        lay.addWidget(self.notice)
        lay.addLayout(bar)
        lay.addWidget(self.tree, 1)

    # ------------------------------------------------------------------ content
    def set_loading(self, program: str) -> None:
        self.program, self.flags = program, None
        self._rebuild()
        self.header.setText(f"Reading the flags of <b>{html.escape(program)}</b> …")

    def set_flags(self, program: str, flags: ProgramFlags | None) -> None:
        self.program, self.flags = program, flags
        self._rebuild()

    def set_job_flags(self, flags: dict[str, str], source_job: str) -> None:
        """Flags (-> value) that the job's own options put on the command line, seen in ``source_job``."""
        self.job_flags, self.job_flags_from = flags, source_job
        self._rebuild()

    def set_text(self, text: str) -> None:
        """The additional-arguments box changed: tick and fill in the flags it names."""
        self.text = text
        given = parse_args(text)
        self._syncing = True
        for name, item in self.items.items():
            flag: Flag = item.data(COL_FLAG, FLAG_ROLE)
            on = name in given
            item.setCheckState(COL_FLAG, Qt.Checked if on else Qt.Unchecked)
            if not flag.is_switch:
                item.setText(COL_VALUE, (given[name][0] if given[name] else "") if on else "")
            f = item.font(COL_FLAG)
            f.setBold(on)
            item.setFont(COL_FLAG, f)
        self._syncing = False
        unknown = [n for n in given if n not in self.items]
        self.notice.setVisible(bool(unknown) and self.flags is not None)
        if unknown and self.flags is not None:
            self.notice.setText(f"<span style='color:#c0392b'>Not a flag of {html.escape(self.program)}: "
                                f"{html.escape(', '.join(unknown))}</span>")
        self._apply_filter()

    def set_search(self, text: str) -> None:
        """The editor's "Find option" text, applied on top of the panel's own filter."""
        self._search = text.strip().lower()
        self._apply_filter()

    def matches(self, text: str) -> bool:
        text = text.strip().lower()
        return bool(text) and any(text in f"{f.name} {f.help}".lower() for f in (self.flags.flags if self.flags else []))

    def _rebuild(self) -> None:
        self._syncing = True
        self.tree.clear()
        self.items.clear()
        if self.flags is None:
            self.header.setText(self._no_flags_text())
            self._syncing = False
            return
        dim = QBrush(self.palette().color(QPalette.PlaceholderText))
        section_item = None
        for f in self.flags.flags:
            if section_item is None or section_item.text(COL_FLAG) != (f.section or "Options"):
                section_item = QTreeWidgetItem([f.section or "Options"])
                font = QFont(self.tree.font())
                font.setBold(True)
                section_item.setFont(COL_FLAG, font)
                section_item.setFlags(Qt.ItemIsEnabled)
                self.tree.addTopLevelItem(section_item)
                section_item.setFirstColumnSpanned(True)
            item = QTreeWidgetItem([f.name, "", "", f.help])
            item.setData(COL_FLAG, FLAG_ROLE, f)
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            if not f.is_switch:
                flags |= Qt.ItemIsEditable
            item.setFlags(flags)
            item.setCheckState(COL_FLAG, Qt.Unchecked)
            if f.name in self.job_flags:
                v = self.job_flags[f.name]
                item.setText(COL_DEFAULT, f"job: {v}" if v else "job: on")
                item.setToolTip(COL_DEFAULT, f"Set by the job's options: {v or 'on'} (seen in {self.job_flags_from}). "
                                             "A value here overrides it: RELION uses the last one given.")
                item.setForeground(COL_DEFAULT, QBrush(self.palette().color(QPalette.Highlight)))
            else:
                item.setText(COL_DEFAULT, "off" if f.is_switch else ("required" if f.default is None
                                                                      else f.default or "—"))
                item.setForeground(COL_DEFAULT, dim)
            if f.is_switch:
                item.setText(COL_VALUE, "")
                item.setToolTip(COL_VALUE, "an on/off switch: tick it to add it")
            else:
                item.setToolTip(COL_VALUE, "click to type a value")
            item.setToolTip(COL_FLAG, f.help)
            item.setToolTip(COL_HELP, f.help)
            self.tree.addTopLevelItem(item)
            self.items[f.name] = item
        n = len(self.flags.flags)
        self.header.setText(
            f"<b>{n}</b> flags of <b>{html.escape(self.program)}</b>. Tick one to add it to the box above, "
            f"click its value to change it. <span style='color:gray'>From {html.escape(self.flags.source)}"
            + (f"; 'job:' values come from {html.escape(self.job_flags_from)}" if self.job_flags else "")
            + ".</span>")
        self._syncing = False
        self.set_text(self.text)

    def _no_flags_text(self) -> str:
        if not self.program:
            return ""
        p = html.escape(self.program)
        if self.program.startswith("relion_python_") or self.program == "relion_filament_selection":
            return f"This job passes the additional arguments to <b>{p}</b>, a Python tool: run it with " \
                   "<tt>--help</tt> in a terminal to see its options."
        if " " in self.program or not self.program.startswith("relion_"):
            return f"This job passes the additional arguments to <b>{p}</b>."
        return f"The flags of <b>{p}</b> could not be read (is RELION loaded?)."

    # ------------------------------------------------------------------ editing
    def _on_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column == COL_VALUE and item.flags() & Qt.ItemIsEditable and self.isEnabled():
            self.tree.editItem(item, COL_VALUE)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing:
            return
        flag: Flag | None = item.data(COL_FLAG, FLAG_ROLE)
        if flag is None:
            return
        if column == COL_FLAG:
            on = item.checkState(COL_FLAG) == Qt.Checked
            value = self._value_for(flag, item.text(COL_VALUE)) if on else None
        elif column == COL_VALUE:
            typed = item.text(COL_VALUE).strip()
            if not typed and flag.name not in parse_args(self.text):
                return
            value = self._value_for(flag, typed) if typed else None  # typing adds the flag, clearing removes it
        else:
            return
        self.text = set_arg(self.text, flag.name, value, switch=flag.is_switch)
        self.edited.emit(self.text)

    def _value_for(self, flag: Flag, typed: str) -> str:
        if flag.is_switch:
            return ""
        return typed or self.job_flags.get(flag.name) or flag.default or ""

    def _on_current(self, item: QTreeWidgetItem | None, _prev) -> None:
        flag = item.data(COL_FLAG, FLAG_ROLE) if item is not None else None
        if flag is not None:
            self.flag_focused.emit(flag)

    def _apply_filter(self) -> None:
        own = self.filter.text().strip().lower()
        only_added = self.only_added.isChecked()
        section_visible: dict[int, bool] = {}
        current_section = -1
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            flag: Flag | None = item.data(COL_FLAG, FLAG_ROLE)
            if flag is None:
                current_section = i
                section_visible[i] = False
                continue
            hay = f"{flag.name} {flag.help} {flag.section}".lower()
            show = all(t in hay for t in (own, self._search) if t)
            if only_added:
                show = show and item.checkState(COL_FLAG) == Qt.Checked
            item.setHidden(not show)
            if show:
                section_visible[current_section] = True
        for i, visible in section_visible.items():
            self.tree.topLevelItem(i).setHidden(not visible)
