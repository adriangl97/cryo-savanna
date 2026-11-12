# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
#
#   The XMIPP code in RELION's src/filename.cpp (GPL-2.0-or-later):
#   Authors: Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#   Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
"""Job editor: every option of a job type, laid out like RELION's job window.

Tabs, rows and on/off groups come from the job definitions extracted from RELION 5.1.1. A
yes/no option greys out the rows it controls, as in RELION, and in continue mode only the options
RELION lets you change stay editable. Changed values are marked; hovering or focusing an option
shows its help. The result is a JobStar with exactly the options RELION writes.

Below "Additional arguments:" a panel lists every flag of the program that gets them (flag_panel.py).
"""

from __future__ import annotations

import fnmatch
import html
import os
import re
from pathlib import Path

from qtpy.QtCore import QEvent, QObject, Qt, QThreadPool, Signal
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.relion_env import RelionEnv, RelionEnvError
from cryo_savanna.jobdefs.flags import Flag, FlagSource, command_flags, program_for
from cryo_savanna.jobdefs.model import JobVariant, OptionDef, Row
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.scratch import SCRATCH_KEY
from cryo_savanna.project.scratch import base_dir as scratch_base_dir
from cryo_savanna.project.selection import interactive_kind
from cryo_savanna.ui.flag_panel import FlagPanel, FlagsTask

MODE_NEW, MODE_CONTINUE, MODE_OVERWRITE = "new", "continue", "overwrite"
EXTRA_ARGS = "other_args"
# job types whose interactive step cryo-savanna can do itself, and RELION's own tool for it
RELION_SELECTORS = {"relion.select": "relion_display", "relion.excludetilts": "napari"}


def qt_file_filter(pattern: str) -> str:
    """RELION's file pattern ('STAR files (*.star) \\t Maps (*.{mrc,map})') -> a Qt name filter."""
    out = []
    for part in (p.strip() for p in pattern.split("\t")):
        if not part:
            continue

        def expand(m: re.Match) -> str:
            stem, alts = m.group(1), m.group(2).split(",")
            return " ".join(stem + a for a in alts)

        part = re.sub(r"([^\s(]*)\{([^}]*)\}", expand, part)
        out.append(part if "(" in part else f"{part} ({part})")
    out.append("All files (*)")
    return ";;".join(out)


def _before_nth(label: str, n: int) -> str:
    """FileName::beforeNthOf('.', n): the label up to its n-th dot (all of it if it has fewer)."""
    parts = label.split(".")
    return ".".join(parts[:n]) if n < len(parts) else label


def node_matches(node_label: str, node_depth: int, wanted: str, depth: int) -> bool:
    """Whether a pipeline node shows up in RELION's browser for an input asking for ``wanted``.

    RELION files every node under .Nodes/<kind>/, and nodes with a deeper type also under
    .Nodes/<label up to that depth>/; an input browses .Nodes/<wanted up to its depth>/.
    """
    if not wanted:
        return False
    if wanted == "*":  # CLI tools: any file of the pipeline
        return True
    if depth <= 1:
        return node_label.split(".", 1)[0] == wanted.split(".", 1)[0]
    return node_depth > 1 and _before_nth(node_label, node_depth) == _before_nth(wanted, depth)


def _name_matches(name: str, pattern: str) -> bool:
    """A pattern naming a file ("pytom match (*pytom_match.star)") also limits the nodes listed; the
    usual ones (*.star, *.mrc) do not."""
    globs = re.findall(r"\(([^)]*)\)", pattern)
    specific = [g for part in globs for g in part.split() if not re.fullmatch(r"\*\.[\w{},]+", g)]
    return not specific or any(fnmatch.fnmatch(os.path.basename(name), g) for g in specific)


# ---------------------------------------------------------------------------- option widgets

class OptionWidget(QWidget):
    changed = Signal()

    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        super().__init__()
        self.opt = opt
        self.editor = editor
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(4)

    def value(self) -> str:
        raise NotImplementedError

    def set_value(self, v: str) -> None:
        raise NotImplementedError

    def focus_target(self) -> QWidget:
        return self


class ChoiceWidget(OptionWidget):
    """Yes/No options and RELION's radio menus. A value not in the list (e.g. a Schemes $$variable) is kept."""

    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        super().__init__(opt, editor)
        self.combo = QComboBox()
        self.combo.addItems(["Yes", "No"] if opt.kind == "boolean" else list(opt.choices))
        self.combo.currentTextChanged.connect(lambda _: self.changed.emit())
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.layout_.addWidget(self.combo)
        self.layout_.addStretch(1)

    def value(self) -> str:
        return self.combo.currentText()

    def set_value(self, v: str) -> None:
        if self.combo.findText(v) < 0:
            self.combo.addItem(v)
        self.combo.setCurrentText(v)

    def focus_target(self) -> QWidget:
        return self.combo


class TextWidget(OptionWidget):
    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        super().__init__(opt, editor)
        self.edit = QLineEdit()
        self.edit.textChanged.connect(lambda _: self.changed.emit())
        self.layout_.addWidget(self.edit, 1)

    def value(self) -> str:
        return self.edit.text()

    def set_value(self, v: str) -> None:
        if self.edit.text() != v:
            self.edit.setText(v)

    def focus_target(self) -> QWidget:
        return self.edit


class SliderWidget(TextWidget):
    """A number with a slider over RELION's range. Typed text is kept exactly (RELION allows values
    outside the slider range)."""

    STEPS_MAX = 1000

    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        super().__init__(opt, editor)
        self.edit.setMaximumWidth(110)
        self.slider = None
        self._syncing = False
        lo, hi = opt.min, opt.max
        if lo is not None and hi is not None and hi > lo:
            step = opt.step if opt.step and opt.step > 0 else (hi - lo) / 100
            self.n = max(1, min(self.STEPS_MAX, round((hi - lo) / step)))
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, self.n)
            self.slider.valueChanged.connect(self._from_slider)
            self.layout_.addWidget(self.slider, 1)
            self.edit.textChanged.connect(self._to_slider)
        else:
            self.layout_.addStretch(1)

    def _from_slider(self, i: int) -> None:
        if self._syncing:
            return
        v = self.opt.min + (self.opt.max - self.opt.min) * i / self.n
        step = self.opt.step or 0
        if step >= 1 and float(step).is_integer() and float(self.opt.min).is_integer():
            text = str(int(round(v)))
        else:
            text = f"{v:.6g}"
        self._syncing = True
        self.edit.setText(text)
        self._syncing = False

    def _to_slider(self, text: str) -> None:
        if self._syncing or self.slider is None:
            return
        try:
            v = float(text)
        except ValueError:
            return
        self._syncing = True
        self.slider.setValue(round((min(max(v, self.opt.min), self.opt.max) - self.opt.min)
                                   / (self.opt.max - self.opt.min) * self.n))
        self._syncing = False

    def set_value(self, v: str) -> None:
        super().set_value(v)
        self._to_slider(v)


class FileWidget(TextWidget):
    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        super().__init__(opt, editor)
        self.browse = QToolButton()
        self.browse.setText("…")
        self.browse.setToolTip("Browse")
        self.browse.clicked.connect(self._browse)
        self.layout_.addWidget(self.browse)

    def _start_dir(self) -> Path:
        proj = self.editor.project_dir
        d = self.opt.directory
        if d == "CURRENT_ODIR" and self.editor.job:
            return proj / self.editor.job
        if d and (proj / d).is_dir():
            return proj / d
        current = self.value()
        if current and (proj / current).parent.is_dir():
            return (proj / current).parent
        return proj

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.opt.label.rstrip(": "), str(self._start_dir()),
                                              qt_file_filter(self.opt.pattern) if self.opt.pattern else "")
        if path:
            self.set_value(self.editor.relative(path))


class NodeWidget(FileWidget):
    """Input node: outputs of earlier jobs with the right node type (newest first), or any file."""

    def __init__(self, opt: OptionDef, editor: "JobEditor"):
        OptionWidget.__init__(self, opt, editor)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(30)
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit = self.combo.lineEdit()
        self.edit.textChanged.connect(lambda _: self.changed.emit())
        self.layout_.addWidget(self.combo, 1)
        self.browse = QToolButton()
        self.browse.setText("…")
        self.browse.setToolTip("Browse")
        self.browse.clicked.connect(self._browse)
        self.layout_.addWidget(self.browse)
        self.fill(editor.pipeline)

    def fill(self, pipeline: Pipeline | None) -> None:
        current = self.value()
        self.combo.blockSignals(True)
        self.combo.clear()
        if pipeline is not None:
            nodes = [n for n in pipeline.nodes.values() if n.producer and node_matches(
                n.type_label, n.type_depth, self.opt.node_type, self.opt.node_type_depth)
                and _name_matches(n.name, self.opt.pattern)]
            for n in reversed(nodes):  # newest jobs first
                pr = pipeline.processes.get(n.producer)
                alias = f"   ({pr.alias.rstrip('/')})" if pr is not None and pr.has_alias else ""
                self.combo.addItem(n.name + alias, n.name)
        self.combo.setCurrentIndex(-1)
        self.combo.blockSignals(False)
        self.set_value(current)

    def value(self) -> str:
        i = self.combo.currentIndex()
        text = self.edit.text()
        if i >= 0 and text == self.combo.itemText(i):
            return self.combo.itemData(i)  # shown with its alias, stored as the node name
        return text

    def set_value(self, v: str) -> None:
        i = self.combo.findData(v)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        else:
            self.combo.setCurrentIndex(-1)
            self.edit.setText(v)

    def focus_target(self) -> QWidget:
        return self.combo


def make_widget(opt: OptionDef, editor: "JobEditor") -> OptionWidget:
    if opt.kind in ("boolean", "radio"):
        return ChoiceWidget(opt, editor)
    if opt.kind == "slider":
        return SliderWidget(opt, editor)
    if opt.kind == "inputnode":
        return NodeWidget(opt, editor)
    if opt.kind == "filename":
        return FileWidget(opt, editor)
    return TextWidget(opt, editor)


# ---------------------------------------------------------------------------- editor

class _HelpFilter(QObject):
    """Shows an option's help in the editor's help box when it gets focus or the mouse."""

    def __init__(self, editor: "JobEditor", key: str):
        super().__init__(editor)
        self.editor, self.key = editor, key

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.FocusIn, QEvent.Enter):
            self.editor.show_help(self.key)
        return False


def _scratch_base(values: dict[str, str]) -> None:
    """Show the scratch directory chosen, not the job's own savanna_NNN in it (a new one is given
    when the job starts)."""
    if SCRATCH_KEY in values:
        values[SCRATCH_KEY] = scratch_base_dir(values[SCRATCH_KEY])


class JobEditor(QWidget):
    """Edit a new job or continue an existing one. Emits ``submit`` with the job.star to use."""

    submit = Signal(object, str, bool)  # JobStar, alias, schedule_only
    closed = Signal()

    def __init__(self, env: RelionEnv, type_names: dict[str, str], parent=None, flag_source: FlagSource | None = None):
        super().__init__(parent)
        self.env = env
        self.type_names = type_names
        self.flag_source = flag_source or FlagSource()
        self.flag_panel: FlagPanel | None = None
        self.flags_toggle: QToolButton | None = None
        self.show_flags = True  # kept from job to job
        self.flags_program: str | None = None
        self.source_job: str | None = None  # job whose command shows what the options set
        self._flags_pool = QThreadPool(self)
        self._flags_pool.setMaxThreadCount(1)
        self._flag_tasks: set[FlagsTask] = set()
        self.project_dir = Path(".")
        self.pipeline: Pipeline | None = None
        self.variant: JobVariant | None = None
        self.options: dict[str, OptionDef] = {}
        self.widgets: dict[str, OptionWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self.markers: dict[str, QLabel] = {}
        self.rows: list[tuple[int, Row, list[QWidget]]] = []  # (tab index, row, widgets to show/hide)
        self.selector: QComboBox | None = None  # "Selection window" of jobs cryo-savanna can do itself
        self.mode = MODE_NEW
        self.job: str | None = None  # the job being continued
        self.readonly = False
        self.can_run = True
        self.own_scratch = False  # each job a savanna_NNN of its scratch directory (Preferences)

        self.title = QLabel()
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.15)
        self.title.setFont(f)
        self.subtitle = QLabel()
        self.subtitle.setWordWrap(True)
        self.alias = QLineEdit()
        self.alias.setPlaceholderText("optional")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find option")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_search)
        self.only_changed = QCheckBox("Only changed")
        self.only_changed.toggled.connect(lambda _: self._apply_search())
        self.reset = QPushButton("Defaults")
        self.reset.setToolTip("Set every option to RELION's default (with this shell's RELION_* settings and "
                              "the scratch directory of the preferences)")
        self.reset.clicked.connect(self._reset_defaults)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.help = QTextBrowser()
        self.help.setOpenExternalLinks(False)
        self.help.setMinimumHeight(60)

        self.run_btn = QPushButton("Run now")
        self.run_btn.setDefault(True)
        self.schedule_btn = QPushButton("Schedule")
        self.close_btn = QPushButton("Close")
        self.run_btn.clicked.connect(lambda: self._submit(False))
        self.schedule_btn.clicked.connect(lambda: self._submit(True))
        self.close_btn.clicked.connect(self.closed.emit)
        self.status = QLabel()
        self.status.setWordWrap(True)

        top = QGridLayout()
        top.addWidget(self.title, 0, 0, 1, 4)
        top.addWidget(self.subtitle, 1, 0, 1, 4)
        self.alias_label = QLabel("Alias:")
        top.addWidget(self.alias_label, 2, 0)
        top.addWidget(self.alias, 2, 1)
        top.addWidget(self.search, 2, 2)
        top.addWidget(self.only_changed, 2, 3)
        top.addWidget(self.reset, 2, 4)
        top.setColumnStretch(1, 1)
        top.setColumnStretch(2, 1)

        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.close_btn)
        buttons.addWidget(self.schedule_btn)
        buttons.addWidget(self.run_btn)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(self.help)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addLayout(buttons)

    # ------------------------------------------------------------------ loading
    def set_project(self, project_dir: Path, pipeline: Pipeline | None) -> None:
        self.project_dir = Path(project_dir)
        self.update_pipeline(pipeline)

    def update_pipeline(self, pipeline: Pipeline | None) -> None:
        """Refresh the input-node lists (keeps what is typed or chosen)."""
        self.pipeline = pipeline
        for w in self.widgets.values():
            if isinstance(w, NodeWidget):
                w.fill(pipeline)

    def new_job(self, variant: JobVariant, values: dict[str, str] | None = None, note: str = "",
                source_job: str | None = None) -> None:
        """Edit a new job of this type. ``values`` overrides the defaults (remembered settings, a clone
        of ``source_job``)."""
        self.mode, self.job, self.source_job = MODE_NEW, None, source_job
        self._build(variant)
        start = variant.default_values(self.env)
        for k, v in (values or {}).items():
            if k in start:
                start[k] = v
        if self.own_scratch:
            _scratch_base(start)
        self._set_values(start)
        name = self._type_name(variant)
        self.title.setText(f"New job: {name}" + (" (tomo)" if variant.is_tomo else ""))
        self.subtitle.setText(note)
        self.alias.clear()
        self._update_mode_widgets()

    def continue_job(self, job: str, jobstar: JobStar, variant: JobVariant) -> None:
        """Edit the continue-mode options of an existing job."""
        self.mode, self.job, self.source_job = MODE_CONTINUE, job, job
        self._build(variant)
        values, ignored = variant.values_from_jobstar(jobstar, self.env)
        self._set_values(values)
        name = self._type_name(variant)
        self.title.setText(f"Continue {job.rstrip('/')}: {name}")
        note = "Only the options RELION allows to change when continuing can be edited."
        if ignored:
            note += f" Dropped options unknown to RELION 5.1: {', '.join(ignored)}."
        self.subtitle.setText(note)
        self._update_mode_widgets()

    def overwrite_job(self, job: str, jobstar: JobStar, variant: JobVariant, alias: str = "") -> None:
        """Edit every option of an existing job, to run it again in its own directory."""
        self.mode, self.job, self.source_job = MODE_OVERWRITE, job, job
        self._build(variant)
        values, ignored = variant.values_from_jobstar(jobstar, self.env)
        if self.own_scratch:
            _scratch_base(values)  # it runs again in a scratch directory of its own
        self._set_values(values)
        name = self._type_name(variant)
        self.title.setText(f"Overwrite {job.rstrip('/')}: {name}")
        note = (f"Runs as a new job in {job.rstrip('/')}: the current job and its files go to Trash/ first. "
                "Jobs that used its outputs stay, but their results come from the old run.")
        if ignored:
            note += f" Dropped options unknown to RELION 5.1: {', '.join(ignored)}."
        self.subtitle.setText(note)
        self.alias.setText(alias)
        self._update_mode_widgets()

    def _type_name(self, variant: JobVariant) -> str:
        if getattr(variant, "kind", None) in ("match", "extract", "align"):  # pytom, AreTomo3 (jobdefs/)
            return f"{variant.program} (runs as an External job)"
        if getattr(variant, "is_cli", False):
            return f"{variant.program} (CLI tool, runs as an External job)"
        return self.type_names.get(variant.label, variant.dirname)

    def set_permissions(self, readonly: bool, can_run: bool, reason: str = "") -> None:
        self.readonly, self.can_run = readonly, can_run
        self._update_mode_widgets()
        if reason:
            self.status.setText(reason)

    def _update_mode_widgets(self) -> None:
        is_new = self.mode != MODE_CONTINUE  # a new job, or one that replaces an old one
        self.alias.setVisible(is_new)
        self.alias_label.setVisible(is_new)
        self.schedule_btn.setVisible(is_new)
        self.run_btn.setText({MODE_NEW: "Run now", MODE_CONTINUE: "Continue now",
                              MODE_OVERWRITE: "Overwrite now"}[self.mode])
        enabled = not self.readonly and self.can_run and self.variant is not None
        self.run_btn.setEnabled(enabled)
        self.schedule_btn.setEnabled(enabled)
        self.status.setText("")
        self._update_states()

    def _build(self, variant: JobVariant) -> None:
        options = variant.resolved_options(self.env)  # first: a bad RELION_* value leaves the editor as it was
        self.variant, self.options = variant, options
        self.widgets.clear()
        self.labels.clear()
        self.markers.clear()
        self.rows.clear()
        self.flag_panel = self.flags_toggle = None
        self.flags_program = None
        self.selector = None
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        for ti, tab in enumerate(variant.resolved_tabs(self.env)):
            page = QWidget()
            grid = QGridLayout(page)
            grid.setColumnStretch(1, 1)
            grid.setVerticalSpacing(4)
            r = 0
            panel_row = None
            if tab.label == "Running" and variant.label in RELION_SELECTORS:
                grid.addWidget(QLabel("Selection window:"), r, 0, Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(self._make_selector(RELION_SELECTORS[variant.label]), r, 1)
                r += 1
            for row in tab.rows:
                keys = [k for k in row.keys if k in self.options]
                if not keys:
                    continue
                parts: list[QWidget] = []
                if len(keys) == 1:
                    key = keys[0]
                    label = QLabel(self.options[key].label.strip())
                    field = self._field(key)
                else:  # RELION's place2/place3: several options on one line
                    label = QLabel(row.label.strip() or self.options[keys[0]].label.strip())
                    field = QWidget()
                    h = QHBoxLayout(field)
                    h.setContentsMargins(0, 0, 0, 0)
                    for k in keys:
                        h.addWidget(QLabel(self.options[k].label.strip()))
                        h.addWidget(self._field(k), 1)
                    key = keys[0]
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setToolTip(self._tooltip(keys))
                marker = QLabel()
                marker.setFixedWidth(12)
                for k in keys:
                    self.labels[k], self.markers[k] = label, marker
                grid.addWidget(label, r, 0)
                grid.addWidget(field, r, 1)
                grid.addWidget(marker, r, 2)
                parts = [label, field, marker]
                self.rows.append((ti, row, parts))
                r += 1
                if keys == [EXTRA_ARGS] and not getattr(variant, "is_cli", False):
                    panel_row = r
                    grid.addWidget(self._make_flag_panel(), r, 0, 1, 3)
                    r += 1
            grid.setRowStretch(r if panel_row is None else panel_row, 1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, tab.label)
        self.search.clear()
        self.only_changed.setChecked(False)
        self.help.setHtml("<p style='color:gray'>Point at an option to see its help.</p>")

    def _make_selector(self, tool: str) -> QComboBox:
        self.selector = QComboBox()
        self.selector.addItems(["cryo-savanna's selector", f"RELION's ({tool})"])
        self.selector.setToolTip(
            "Where you make the selection when the job runs now. cryo-savanna's selector writes the same files "
            f"as {tool}.\nOnly for an interactive selection; scheduled jobs always run RELION's {tool}.")
        return self.selector

    @property
    def native_selector(self) -> bool:
        """Run now should open cryo-savanna's own selection window instead of RELION's tool."""
        return self.selector is not None and self.selector.currentIndex() == 0

    def _make_flag_panel(self) -> FlagPanel:
        """The flag list under "Additional arguments:", with a button on that row to show or hide it."""
        box = self.widgets[EXTRA_ARGS]
        panel = FlagPanel()
        panel.setVisible(self.show_flags)
        box.edit.textChanged.connect(panel.set_text)
        panel.edited.connect(box.set_value)
        panel.flag_focused.connect(self.show_flag_help)
        toggle = QToolButton()
        toggle.setText("All flags")
        toggle.setCheckable(True)
        toggle.setChecked(self.show_flags)
        toggle.setToolTip("Show every flag of the program that gets these arguments")
        toggle.toggled.connect(self._toggle_flags)
        box.layout_.addWidget(toggle)
        self.flag_panel, self.flags_toggle = panel, toggle
        return panel

    def _toggle_flags(self, on: bool) -> None:
        self.show_flags = on
        self._apply_search()

    def _update_flags_program(self) -> None:
        """Show the flags of the program that gets the extra arguments with the current options."""
        if self.flag_panel is None or self.variant is None:
            return
        program = program_for(self.variant.label, self.values())
        if program == self.flags_program:
            return
        self.flags_program = program
        self.flag_panel.set_job_flags(*self._job_flags(program))
        if self.flag_source.needs_run(program):
            self.flag_panel.set_loading(program)
            task = FlagsTask(self.flag_source, program)
            task.signals.done.connect(self._on_flags_ready)
            self._flag_tasks.add(task)
            self._flags_pool.start(task)
        else:
            self.flag_panel.set_flags(program, self.flag_source.flags(program))

    def _on_flags_ready(self, program: str, flags) -> None:
        self._flag_tasks = {t for t in self._flag_tasks if t.program != program}
        if self.flag_panel is not None and program == self.flags_program:
            self.flag_panel.set_flags(program, flags)
            self._apply_search()

    def wait_flags(self) -> None:
        """Wait for --help runs in progress (tests)."""
        self._flags_pool.waitForDone()
        QApplication.processEvents()

    def _job_flags(self, program: str) -> tuple[dict[str, str], str]:
        """Flags the job's own options put on the command line, from the note.txt of the job being
        continued or cloned, else of the newest job of this type."""
        candidates = [self.source_job] if self.source_job else []
        if self.pipeline is not None and self.variant is not None:
            candidates += [n for n, pr in reversed(self.pipeline.processes.items())
                           if pr.type_label == self.variant.label and n != self.source_job]
        for job in candidates[:20]:
            try:
                note = (self.project_dir / job / "note.txt").read_text(errors="replace")
                extra = JobStar.read(self.project_dir / job / "job.star").options.get(EXTRA_ARGS, "")
            except (OSError, ValueError, KeyError):
                continue
            flags = command_flags(note, extra, program)
            if flags:
                which = "this job" if job == self.job else job.rstrip("/")
                return flags, which
        return {}, ""

    def _field(self, key: str) -> OptionWidget:
        w = make_widget(self.options[key], self)
        w.changed.connect(lambda k=key: self._on_changed(k))
        w.setToolTip(self._tooltip([key]))
        filt = _HelpFilter(self, key)
        for target in {w, w.focus_target()}:
            target.installEventFilter(filt)
        self.widgets[key] = w
        return w

    def _tooltip(self, keys: list[str]) -> str:
        o = self.options[keys[0]]
        tip = f"{', '.join(keys)}   (default: {o.default or '—'})"
        if o.help:
            tip += "\n\n" + o.help.strip()[:600]
        return tip

    # ------------------------------------------------------------------ values
    def refresh_defaults(self) -> None:
        """The defaults changed (Preferences): a new job's options left at the old default take the new one."""
        if self.variant is None:
            return
        try:
            options = self.variant.resolved_options(self.env)
        except RelionEnvError:
            return
        if options.keys() != self.options.keys():
            return  # other queue fields: shown the next time the editor opens
        vals = self.values()
        old, self.options = self.options, options
        if self.mode == MODE_NEW:
            for k in self.widgets:
                if vals[k] == old[k].default and options[k].default != old[k].default:
                    vals[k] = options[k].default
        self._set_values(vals)

    def _set_values(self, values: dict[str, str]) -> None:
        self._loading = True
        for k, w in self.widgets.items():
            w.set_value(values.get(k, self.options[k].default))
        self._loading = False
        self._update_states()
        self._update_flags_program()

    def values(self) -> dict[str, str]:
        """All options, as RELION writes them to job.star."""
        return {k: (self.widgets[k].value() if k in self.widgets else o.default) for k, o in self.options.items()}

    def jobstar(self, job_dir: str | None = None) -> JobStar:
        """The job.star to submit. A CLI tool job needs its directory (outputs go there): the job
        being continued or overwritten, else the one the next new job will get."""
        v = self.variant
        if self.is_cli:
            return v.make_jobstar(self.values(), self.mode == MODE_CONTINUE, job_dir or self.job_dir(), self.pipeline)
        return JobStar(type_label=v.label, is_continue=self.mode == MODE_CONTINUE, is_tomo=v.is_tomo,
                       options=self.values())

    @property
    def is_cli(self) -> bool:
        return getattr(self.variant, "is_cli", False)

    def job_dir(self) -> str:
        """The directory a new job will get, from the latest snapshot's job counter (no read here, on
        the GUI thread: a slow file system would freeze the window). A job added meanwhile makes the
        guess wrong; job_actions then writes the job.star again for the directory the job did get."""
        if self.job and self.mode != MODE_NEW:
            return self.job
        counter = self.pipeline.job_counter if self.pipeline is not None else 1
        return f"{self.variant.dirname}/job{counter:03d}/"

    def changed_options(self) -> dict[str, tuple[str, str]]:
        """key -> (default, value) for every option that differs from RELION's default."""
        vals = self.values()
        return {k: (o.default, vals[k]) for k, o in self.options.items() if vals[k] != o.default}

    def _on_changed(self, key: str) -> None:
        if getattr(self, "_loading", False):
            return
        self._update_states()
        self._update_flags_program()
        if self.only_changed.isChecked():
            self._apply_search()

    def _reset_defaults(self) -> None:
        if self.variant is None:
            return
        keep = {k: self.widgets[k].value() for k in self.widgets
                if self.mode == MODE_CONTINUE and not self._editable(k, self.values())}
        vals = self.variant.default_values(self.env)
        vals.update(keep)
        self._set_values(vals)

    def _editable(self, key: str, values: dict[str, str]) -> bool:
        return self.variant.enabled_for_mode(self.options[key], self.mode == MODE_CONTINUE)

    def _update_states(self) -> None:
        if self.variant is None:
            return
        vals = self.values()
        pal = self.palette()
        dim = pal.color(QPalette.PlaceholderText).name()
        accent = pal.color(QPalette.Highlight).name()
        for _, row, parts in self.rows:
            active = self.variant.group_active(row.group, vals)
            for k in row.keys:
                if k not in self.widgets:
                    continue
                editable = active and self._editable(k, vals) and not self.readonly
                self.widgets[k].setEnabled(editable)
                changed = vals[k] != self.options[k].default
                self.markers[k].setText(f"<span style='color:{accent}'>●</span>" if changed else "")
                self.markers[k].setToolTip(f"changed from the default ({self.options[k].default or 'empty'})"
                                           if changed else "")
                lab = self.labels[k]
                f = lab.font()
                f.setBold(changed)
                lab.setFont(f)
                lab.setStyleSheet("" if active else f"color:{dim}")
        if self.flag_panel is not None:
            self.flag_panel.setEnabled(self.widgets[EXTRA_ARGS].isEnabled())
        if self.selector is not None:
            self.selector.setEnabled(not self.readonly and interactive_kind(self.variant.label, vals) is not None)

    # ------------------------------------------------------------------ search and help
    def _apply_search(self) -> None:
        text = self.search.text().strip().lower()
        only_changed = self.only_changed.isChecked()
        vals = self.values() if only_changed else {}
        counts = [0] * self.tabs.count()
        for ti, row, parts in self.rows:
            keys = [k for k in row.keys if k in self.options]
            hay = " ".join([row.label] + [f"{k} {self.options[k].label} {self.options[k].help}" for k in keys]).lower()
            flags_match = keys == [EXTRA_ARGS] and self.flag_panel is not None and self.flag_panel.matches(text)
            show = (not text or text in hay or flags_match) and \
                (not only_changed or any(vals[k] != self.options[k].default for k in keys))
            for w in parts:
                w.setVisible(show)
            if keys == [EXTRA_ARGS] and self.flag_panel is not None:
                self.flag_panel.setVisible(show and self.show_flags)
                self.flag_panel.set_search(text if flags_match and text not in hay else "")
            counts[ti] += show
        filtering = bool(text) or only_changed
        for ti in range(self.tabs.count()):
            base = self.tabs.tabText(ti).split("  (")[0]
            self.tabs.setTabText(ti, f"{base}  ({counts[ti]})" if filtering else base)
            self.tabs.setTabEnabled(ti, counts[ti] > 0 or not filtering)
        if filtering and counts and counts[self.tabs.currentIndex()] == 0:
            first = next((i for i, c in enumerate(counts) if c), None)
            if first is not None:
                self.tabs.setCurrentIndex(first)

    def show_help(self, key: str) -> None:
        o = self.options.get(key)
        if o is None:
            return
        extra = []
        if o.kind == "slider" and o.min is not None:
            extra.append(f"range {o.min:g} – {o.max:g}" + (f", step {o.step:g}" if o.step else ""))
        if o.kind == "inputnode" and o.node_type:
            extra.append(f"input node type {o.node_type}")
        env = ", ".join(var for _, var in o.env)
        if env:
            extra.append(f"default from {env}" + (" or Preferences" if "RELION_SCRATCH_DIR" in env else ""))
        if key == SCRATCH_KEY and self.own_scratch:
            extra.append("each job gets its own savanna_NNN subdirectory in it")
        name = "" if o.label.strip() == key else html.escape(key) + " · "
        self.help.setHtml(
            f"<b>{html.escape(o.label.strip())}</b> <span style='color:gray'>{name}default "
            f"{html.escape(o.default or '—')}{' · ' + html.escape(' · '.join(extra)) if extra else ''}</span>"
            f"<p style='white-space:pre-wrap'>{html.escape(o.help.strip())}</p>")

    def show_flag_help(self, flag: Flag) -> None:
        bits = [f"flag of {self.flags_program}", flag.section]
        if flag.is_switch:
            bits.append("on/off switch")
        else:
            bits.append(f"default {flag.default}" if flag.default is not None else "no default")
        job_value = self.flag_panel.job_flags.get(flag.name) if self.flag_panel is not None else None
        note = ""
        if job_value is not None:
            note = (f"<p style='color:gray'>The job's options set this ({html.escape(job_value or 'on')} in "
                    f"{html.escape(self.flag_panel.job_flags_from)}). A value here overrides it: RELION uses the "
                    "last one given.</p>")
        self.help.setHtml(f"<b>{html.escape(flag.name)}</b> <span style='color:gray'>"
                          f"{html.escape(' · '.join(b for b in bits if b))}</span>"
                          f"<p style='white-space:pre-wrap'>{html.escape(flag.help)}</p>{note}")

    # ------------------------------------------------------------------ submit
    def relative(self, path: str) -> str:
        """A chosen file relative to the project, as RELION's GUI gives it. Symlinks inside the project
        (a Movies/ link to the raw data) are kept: only the project's own path is resolved."""
        p = Path(os.path.abspath(path))
        for project in (self.project_dir, self.project_dir.resolve()):
            try:
                return str(p.relative_to(Path(os.path.abspath(project))))
            except ValueError:
                pass
        try:
            return str(p.resolve().relative_to(self.project_dir.resolve()))
        except ValueError:
            return str(p)

    def _submit(self, schedule_only: bool) -> None:
        if self.variant is None:
            return
        self.submit.emit(self.jobstar(), self.alias.text().strip() if self.mode != MODE_CONTINUE else "", schedule_only)
