# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The job editor's setting presets (project/presets.py): saving the form's options as a preset, and managing
the presets of a job type (rename, duplicate, delete, and their options one by one)."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.jobdefs.flags import ADVANCED_TAB, EXTRA_ARGS, has_advanced_tab
from cryo_savanna.jobdefs.model import JobVariant, OptionDef
from cryo_savanna.project.presets import PresetError, PresetStore, check_name, is_input
from cryo_savanna.ui.pacing import pace_splitters


def tab_of(variant: JobVariant) -> dict[str, str]:
    """option key -> the label of the editor tab it is on."""
    out = {k: t.label for t in variant.tabs for row in t.rows for k in row.keys}
    if has_advanced_tab(variant):  # the job editor's own tab
        out[EXTRA_ARGS] = ADVANCED_TAB
    return out


def form_order(variant: JobVariant, options: dict[str, OptionDef]) -> list[str]:
    """The option keys as the editor shows them, tab by tab and row by row (then any not on a tab)."""
    placed = [k for t in variant.tabs for row in t.rows for k in row.keys if k in options]
    seen = set(placed)
    return list(dict.fromkeys(placed)) + [k for k in options if k not in seen]


def _label(opt: OptionDef) -> str:
    return opt.label.rstrip(": ").strip() or opt.key


class SavePresetDialog(QDialog):
    """A name and the options to save: those changed from RELION's default are ticked, inputs are not."""

    COLUMNS = ["Option", "Tab", "Value", ""]

    def __init__(self, parent, variant: JobVariant, options: dict[str, OptionDef], values: dict[str, str],
                 existing: list[str], name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Save setting preset")
        self.options, self.values_, self.existing = options, values, existing
        tabs = tab_of(variant)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Class3D, 200 kDa, K=4")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find option")
        self.search.setClearButtonEnabled(True)
        self.only_changed = QCheckBox("Only changed")
        self.count = QLabel()
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        dim = self.palette().color(QPalette.PlaceholderText)
        self.keys: list[str] = []
        for key in form_order(variant, options):
            opt = options[key]
            v = values.get(key, opt.default)
            changed, inp = v != opt.default, is_input(opt)
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(_label(opt))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if changed and not inp else Qt.Unchecked)
            item.setToolTip(f"{key}\n\n{opt.help}" if opt.help else key)
            note = "input" if inp else ("changed" if changed else "")
            cells = [item, QTableWidgetItem(tabs.get(key, "")), QTableWidgetItem(v), QTableWidgetItem(note)]
            for c, it in enumerate(cells):
                if c and not changed:
                    it.setForeground(dim)
                self.table.setItem(row, c, it)
            self.keys.append(key)
        self.table.resizeColumnsToContents()
        self.search.textChanged.connect(self._filter)
        self.only_changed.toggled.connect(self._filter)
        self.table.itemChanged.connect(lambda _: self._update_count())
        all_btn, none_btn = QPushButton("Tick shown"), QPushButton("Untick shown")
        all_btn.clicked.connect(lambda: self._tick_shown(True))
        none_btn.clicked.connect(lambda: self._tick_shown(False))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Name:", self.name)
        bar = QHBoxLayout()
        bar.addWidget(self.search, 1)
        bar.addWidget(self.only_changed)
        bar.addWidget(all_btn)
        bar.addWidget(none_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("The ticked options are saved. Loading the preset sets them, keeps the inputs of the "
                             "form, and sets every other option to RELION's default."))
        lay.addLayout(bar)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.count)
        lay.addWidget(buttons)
        self.resize(760, 560)
        self._update_count()

    def _filter(self) -> None:
        text = self.search.text().strip().lower()
        for row, key in enumerate(self.keys):
            opt = self.options[key]
            hay = f"{key} {opt.label} {opt.help}".lower()
            changed = self.values_.get(key, opt.default) != opt.default
            self.table.setRowHidden(row, bool(text and text not in hay) or (self.only_changed.isChecked() and not changed))

    def _tick_shown(self, on: bool) -> None:
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _update_count(self) -> None:
        self.count.setText(f"{len(self.chosen())} of {len(self.keys)} options ticked")

    def chosen(self) -> dict[str, str]:
        """The ticked options with their values in the form."""
        return {k: self.values_.get(k, self.options[k].default) for row, k in enumerate(self.keys)
                if self.table.item(row, 0).checkState() == Qt.Checked}

    def accept(self) -> None:
        try:
            name = check_name(self.name.text())
        except PresetError as e:
            QMessageBox.warning(self, "Save setting preset", str(e))
            return
        if not self.chosen():
            QMessageBox.warning(self, "Save setting preset", "Tick at least one option to save.")
            return
        if name in self.existing and QMessageBox.question(
                self, "Save setting preset", f"Replace the preset “{name}”?") != QMessageBox.Yes:
            return
        self.name.setText(name)
        super().accept()


class ManagePresetsDialog(QDialog):
    """The presets of one job type: rename, duplicate, delete; and the options of each, edited as text. Every
    change is saved at once."""

    def __init__(self, parent, store: PresetStore, key: str, title: str, options: dict[str, OptionDef],
                 select: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Setting presets — {title}")
        self.store, self.key, self.options = store, key, options
        self.list = QListWidget()
        self.list.currentItemChanged.connect(lambda *_: self._show())
        self.rename_btn, self.dup_btn, self.del_btn = QPushButton("Rename…"), QPushButton("Duplicate"), \
            QPushButton("Delete…")
        self.rename_btn.clicked.connect(self._rename)
        self.dup_btn.clicked.connect(self._duplicate)
        self.del_btn.clicked.connect(self._delete)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Option", "Key", "Value"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_edit)
        self.add_btn, self.remove_btn = QPushButton("Add option…"), QPushButton("Remove option")
        self.add_btn.clicked.connect(self._add_option)
        self.remove_btn.clicked.connect(self._remove_option)
        self.note = QLabel()
        self.note.setWordWrap(True)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.list, 1)
        lb = QHBoxLayout()
        for b in (self.rename_btn, self.dup_btn, self.del_btn):
            lb.addWidget(b)
        ll.addLayout(lb)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.table, 1)
        rb = QHBoxLayout()
        rb.addWidget(self.add_btn)
        rb.addWidget(self.remove_btn)
        rb.addStretch(1)
        rl.addLayout(rb)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([240, 520])
        pace_splitters(split)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addWidget(self.note)
        lay.addWidget(close)
        self.resize(820, 480)
        self._syncing = False
        self._fill(select)

    # -- the presets
    def current(self) -> str | None:
        it = self.list.currentItem()
        return it.text() if it is not None else None

    def _fill(self, select: str | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        names = self.store.names(self.key)
        for n in names:
            self.list.addItem(QListWidgetItem(n))
        self.list.blockSignals(False)
        if names:
            self.list.setCurrentRow(names.index(select) if select in names else 0)
        else:
            self.note.setText("No presets saved for this job type yet: use “Save current settings as…”.")
        self._show()

    def _warn(self, e: Exception) -> None:
        QMessageBox.warning(self, "Setting presets", str(e))

    def _rename(self) -> None:
        name = self.current()
        if name is None:
            return
        new, ok = QInputDialog.getText(self, "Rename preset", "New name:", text=name)
        if not ok or new.strip() == name:
            return
        try:
            self.store.rename(self.key, name, new)
        except PresetError as e:
            self._warn(e)
            return
        self._fill(new.strip())

    def _duplicate(self) -> None:
        name = self.current()
        if name is None:
            return
        try:
            self._fill(self.store.duplicate(self.key, name))
        except PresetError as e:
            self._warn(e)

    def _delete(self) -> None:
        name = self.current()
        if name is None or QMessageBox.question(self, "Delete preset", f"Delete the preset “{name}”?") \
                != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.key, name)
        except PresetError as e:
            self._warn(e)
        self._fill()

    # -- the options of the preset shown
    def _show(self) -> None:
        name = self.current()
        p = self.store.get(self.key, name) if name is not None else None
        for b in (self.rename_btn, self.dup_btn, self.del_btn, self.add_btn, self.remove_btn):
            b.setEnabled(p is not None)
        self._syncing = True
        self.table.setRowCount(0)
        if p is not None:
            dim = self.palette().color(QPalette.PlaceholderText)
            unknown = []
            for key, value in p.options.items():
                opt = self.options.get(key)
                row = self.table.rowCount()
                self.table.insertRow(row)
                label = QTableWidgetItem(_label(opt) if opt is not None else key)
                k = QTableWidgetItem(key)
                for it in (label, k):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                v = QTableWidgetItem(value)
                if opt is None:
                    unknown.append(key)
                    for it in (label, k, v):
                        it.setForeground(dim)
                        it.setToolTip("Not an option of this form (another RELION version?): kept, not loaded")
                elif opt.help:
                    label.setToolTip(opt.help)
                for c, it in enumerate((label, k, v)):
                    self.table.setItem(row, c, it)
            self.table.resizeColumnsToContents()
            saved = f"saved {p.saved.replace('T', ' ')}" if p.saved else ""
            made = f" with RELION {p.relion_version}" if p.relion_version else ""
            self.note.setText(f"{len(p.options)} options, {saved}{made}." + (
                f" Not in this form (kept, not loaded): {', '.join(unknown)}." if unknown else ""))
        self._syncing = False

    def _table_options(self) -> dict[str, str]:
        return {self.table.item(r, 1).text(): self.table.item(r, 2).text() for r in range(self.table.rowCount())}

    def _save_table(self) -> None:
        name = self.current()
        if name is None:
            return
        try:
            self.store.set_options(self.key, name, self._table_options())
        except PresetError as e:
            self._warn(e)
        self._show()

    def _on_edit(self, item: QTableWidgetItem) -> None:
        if not self._syncing and item.column() == 2:
            self._save_table()

    def _add_option(self) -> None:
        have = set(self._table_options())
        choices = {f"{_label(o)}  ({k})": k for k, o in self.options.items() if k not in have}
        if not choices:
            return
        label, ok = QInputDialog.getItem(self, "Add option", "Option:", list(choices), 0, False)
        if not ok:
            return
        key = choices[label]
        options = self._table_options()
        options[key] = self.options[key].default
        name = self.current()
        if name is None:
            return
        try:
            self.store.set_options(self.key, name, options)
        except PresetError as e:
            self._warn(e)
        self._show()
        row = next((r for r in range(self.table.rowCount()) if self.table.item(r, 1).text() == key), None)
        if row is not None:
            self.table.setCurrentCell(row, 2)
            self.table.editItem(self.table.item(row, 2))

    def _remove_option(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._syncing = True
        for r in rows:
            self.table.removeRow(r)
        self._syncing = False
        self._save_table()
