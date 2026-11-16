# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The job editor's "Variations…": choose options and their values, see every combination as a job (with an
alias you can change), then schedule or run them all (project/variations.py does the arithmetic)."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.jobdefs.model import OptionDef
from cryo_savanna.project import variations as V
from cryo_savanna.project.ops import OpsError, PipelineOps
from cryo_savanna.project.pipeline import Pipeline


def _label(opt: OptionDef) -> str:
    return opt.label.rstrip(": ").strip() or opt.key


class VariationRow(QWidget):
    """One option and its values: a list (or a range) typed in, the choices ticked, or input nodes chosen."""

    def __init__(self, dialog: "VariationsDialog", opt: OptionDef, current: str):
        super().__init__()
        self.dialog, self.opt = dialog, opt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        name = QLabel(_label(opt))
        name.setToolTip(f"{opt.key}\n\n{opt.help}" if opt.help else opt.key)
        name.setMinimumWidth(180)
        lay.addWidget(name)
        self.boxes: list[QCheckBox] = []
        self.edit: QLineEdit | None = None
        if V.is_choice(opt):
            for c in V.choices(opt):
                b = QCheckBox(c)
                b.setChecked(c == current)
                b.toggled.connect(lambda _: dialog.update_preview())
                self.boxes.append(b)
                lay.addWidget(b)
            lay.addStretch(1)
        else:
            self.edit = QLineEdit(current)
            numeric = opt.kind == "slider" or opt.min is not None
            self.edit.setPlaceholderText("3, 4, 5  or  2:8:2" if numeric else "values, separated by commas")
            self.edit.setToolTip("A list of values separated by commas (or one per line), or for numbers a range "
                                 "start:stop:step (2:8:2 is 2, 4, 6, 8)")
            self.edit.textChanged.connect(lambda _: dialog.update_preview())
            lay.addWidget(self.edit, 1)
            if opt.kind == "inputnode":
                pick = QPushButton("Choose…")
                pick.setToolTip("Choose input nodes of the pipeline")
                pick.clicked.connect(self._choose_nodes)
                lay.addWidget(pick)
        lay.addWidget(QLabel("Tag:"))
        self.tag = QLineEdit(V.default_tag(opt.key))
        self.tag.setMaximumWidth(90)
        self.tag.setToolTip("What this option is called in the aliases (K3_T2)")
        self.tag.textChanged.connect(lambda _: dialog.update_preview())
        lay.addWidget(self.tag)
        remove = QToolButton()
        remove.setText("✕")
        remove.setToolTip("Do not vary this option")
        remove.clicked.connect(lambda: dialog.remove_row(self))
        lay.addWidget(remove)

    def values(self) -> list[str]:
        if self.edit is None:
            return [b.text() for b in self.boxes if b.isChecked()]
        return V.parse_values(self.opt, self.edit.text())

    def set_text(self, text: str) -> None:
        if self.edit is not None:
            self.edit.setText(text)
        else:
            wanted = {t.strip() for t in text.split(",")}
            for b in self.boxes:
                b.setChecked(b.text() in wanted)

    def variation(self) -> V.Variation:
        return V.Variation(self.opt.key, self.values(), self.tag.text().strip())

    def _choose_nodes(self) -> None:
        from cryo_savanna.ui.job_editor import _name_matches, node_matches
        p = self.dialog.pipeline
        if p is None or self.edit is None:
            return
        nodes = [n for n in reversed(list(p.nodes.values())) if n.producer and node_matches(
            n.type_label, n.type_depth, self.opt.node_type, self.opt.node_type_depth)
            and _name_matches(n.name, self.opt.pattern)]
        dlg = QDialog(self)
        dlg.setWindowTitle(_label(self.opt))
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        chosen = set(V.parse_values(self.opt, self.edit.text())) if self.edit.text().strip() else set()
        for n in nodes:
            pr = p.processes.get(n.producer) if n.producer else None
            it = QListWidgetItem(n.name + (f"   ({pr.alias.rstrip('/')})" if pr is not None and pr.has_alias else ""))
            it.setData(Qt.UserRole, n.name)
            lst.addItem(it)
            it.setSelected(n.name in chosen)
        ok = QPushButton("OK")
        ok.clicked.connect(dlg.accept)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select the inputs to run the job on (Ctrl/Shift for several):"))
        lay.addWidget(lst, 1)
        lay.addWidget(ok)
        dlg.resize(560, 400)
        if dlg.exec() == QDialog.Accepted:
            self.edit.setText(", ".join(i.data(Qt.UserRole) for i in lst.selectedItems()))


class VariationsDialog(QDialog):
    """Rows of option → values; the combinations as a table of jobs; Schedule all / Run all now."""

    def __init__(self, parent, title: str, options: dict[str, OptionDef], values: dict[str, str], base_alias: str,
                 pipeline: Pipeline | None, dirname: str, tabs: dict[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Variations — {title}")
        self.options, self.values_, self.pipeline, self.dirname = options, values, pipeline, dirname
        self.tabs = tabs or {}
        self.schedule_only = True
        self.rows: list[VariationRow] = []
        self._aliases: dict[tuple, str] = {}  # combination -> an alias typed in the table
        self._combos: list[dict[str, str]] = []
        self._filling = False

        self.base = QLineEdit(base_alias)
        self.base.setToolTip("The start of every alias (the editor's alias, else the job type)")
        self.base.textChanged.connect(lambda _: self.update_preview())
        self.rows_box = QVBoxLayout()
        self.rows_box.addStretch(1)
        rows_w = QWidget()
        rows_w.setLayout(self.rows_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_w)
        scroll.setMinimumHeight(120)
        add = QPushButton("Add option…")
        add.clicked.connect(self.ask_option)
        self.table = QTableWidget(0, 1)
        self.table.verticalHeader().hide()
        self.table.itemChanged.connect(self._on_alias_edit)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.schedule_btn, self.run_btn, cancel = QPushButton("Schedule all"), QPushButton("Run all now"), \
            QPushButton("Cancel")
        self.schedule_btn.clicked.connect(lambda: self.finish(True))
        self.run_btn.clicked.connect(lambda: self.finish(False))
        cancel.clicked.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(QLabel("Alias start:"))
        top.addWidget(self.base, 1)
        top.addWidget(add)
        buttons = QHBoxLayout()
        buttons.addWidget(self.info, 1)
        for b in (cancel, self.schedule_btn, self.run_btn):
            buttons.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("One job for every combination of these values; every other option as in the editor."))
        lay.addLayout(top)
        lay.addWidget(scroll, 1)
        lay.addWidget(QLabel("Jobs (double-click an alias to change it):"))
        lay.addWidget(self.table, 2)
        lay.addLayout(buttons)
        self.resize(820, 600)
        self.update_preview()

    # -- the options varied
    def add_option(self, key: str, text: str | None = None) -> VariationRow | None:
        opt = self.options.get(key)
        if opt is None or any(r.opt.key == key for r in self.rows):
            return None
        row = VariationRow(self, opt, self.values_.get(key, opt.default))
        if text is not None:
            row.set_text(text)
        self.rows.append(row)
        self.rows_box.insertWidget(self.rows_box.count() - 1, row)
        self.update_preview()
        return row

    def ask_option(self) -> None:
        have = {r.opt.key for r in self.rows}
        labels = {}
        for k, o in self.options.items():
            if k not in have:
                tab = self.tabs.get(k, "")
                labels[f"{_label(o)}  ({k}{', ' + tab if tab else ''})"] = k
        if not labels:
            return
        label, ok = QInputDialog.getItem(self, "Vary an option", "Option:", list(labels), 0, False)
        if ok:
            row = self.add_option(labels[label])
            if row is not None and row.edit is not None:
                row.edit.setFocus()
                row.edit.selectAll()

    def remove_row(self, row: VariationRow) -> None:
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.update_preview()

    def variations(self) -> list[V.Variation]:
        return [r.variation() for r in self.rows]

    # -- the jobs
    def update_preview(self) -> None:
        try:
            vs = self.variations()
        except V.VariationError as e:
            self._combos = []
            self._show_table([])
            self.info.setText(f"⚠ {e}")
            self._enable(False)
            return
        n = V.count(vs)
        if n > V.MAX_JOBS:
            self._combos = []
            self._show_table([])
            self.info.setText(f"⚠ {n} jobs: at most {V.MAX_JOBS} in one go")
            self._enable(False)
            return
        self._combos = V.combinations(vs)
        self._show_table(vs)
        self.info.setText(self._problem() or (f"{n} job{'s' if n != 1 else ''}" if n else
                                               "Add an option to vary (and its values)."))
        self._enable(bool(n) and not self._problem())

    def _enable(self, on: bool) -> None:
        self.schedule_btn.setEnabled(on)
        self.run_btn.setEnabled(on)

    def _show_table(self, vs: list[V.Variation]) -> None:
        self._filling = True
        keys = [v.key for v in vs if v.values]
        self.table.clear()
        self.table.setColumnCount(1 + len(keys))
        self.table.setHorizontalHeaderLabels(["Alias"] + [_label(self.options[k]) for k in keys])
        self.table.setRowCount(len(self._combos))
        for r, combo in enumerate(self._combos):
            alias = self._aliases.get(self._ckey(combo)) or V.alias_for(self.base.text(), combo, vs, self.options)
            self.table.setItem(r, 0, QTableWidgetItem(alias))
            for c, k in enumerate(keys, 1):
                it = QTableWidgetItem(combo[k])
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, c, it)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, self.table.columnCount()):  # the values: as wide as their option's name
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._filling = False

    @staticmethod
    def _ckey(combo: dict[str, str]) -> tuple:
        return tuple(sorted(combo.items()))

    def _on_alias_edit(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 0 or not 0 <= item.row() < len(self._combos):
            return
        self._aliases[self._ckey(self._combos[item.row()])] = item.text().strip()
        problem = self._problem()
        self.info.setText(problem or f"{len(self._combos)} jobs")
        self._enable(not problem)

    def aliases(self) -> list[str]:
        return [self.table.item(r, 0).text().strip() for r in range(self.table.rowCount())]

    def _problem(self) -> str:
        """What is wrong with the aliases (empty: nothing)."""
        aliases = self.aliases()
        dup = V.duplicates(aliases)
        if dup:
            return f"⚠ The same alias twice: {', '.join(dup)}"
        if self.pipeline is not None:
            for a in aliases:
                try:
                    PipelineOps.check_alias(self.pipeline, f"{self.dirname}/job000/", a)
                except OpsError as e:
                    return f"⚠ {a or '(empty)'}: {e}"
        return ""

    def jobs(self) -> list[tuple[dict[str, str], str]]:
        """(the options varied with their values, the alias) of every job, in order."""
        return list(zip(self._combos, self.aliases()))

    def finish(self, schedule_only: bool) -> None:
        n = len(self._combos)
        if not n or self._problem():
            return
        if n > V.MAX_WITHOUT_ASKING and QMessageBox.question(
                self, "Variations", f"Make {n} jobs?") != QMessageBox.Yes:
            return
        if not schedule_only:
            local = sum(1 for combo, _ in self.jobs() if {**self.values_, **combo}.get("do_queue", "No") != "Yes")
            if local > 1 and QMessageBox.question(
                    self, "Variations", f"{local} of these jobs are not submitted to the queue: they all start at "
                    "once on this machine. Run them all now?") != QMessageBox.Yes:
                return
        self.schedule_only = schedule_only
        self.accept()

