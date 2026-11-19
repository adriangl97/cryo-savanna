# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The job editor's "Advanced parameters" tab: the "Additional arguments:" box, then every flag of the program
that gets it, as rows like those of the other tabs, under the sections of the program's --help.

The box stays the one source of truth: changing a flag's row rewrites the box (leaving whatever else is typed
there alone), and typing in the box fills in the rows of the flags it names. A flag is in bold (and marked)
when the box gives it. Flags the job writes itself are left out (jobdefs/flags.py ``job_flags``): those of
its options, set on the other tabs, and those it always writes. Typed in the box anyway, they override the
job's value (RELION uses the last one given), which the tab says.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QEvent, QObject, QRunnable, Qt, Signal
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import QFrame, QGridLayout, QLabel, QWidget

from cryo_savanna.jobdefs.cli_tools import is_input
from cryo_savanna.jobdefs.flags import Flag, FlagSource, JobFlag, ProgramFlags, parse_args, set_arg
from cryo_savanna.jobdefs.model import OptionDef

NONE_DEFAULTS = ("None", "NULL")  # shown as an empty field (with the default as its placeholder)
FIRST_FLAG_ROW = 3  # the box, the header line, the notice; then the flags


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


_FILE_WORDS = re.compile(r"\b(file|files|mask|map|star|image|model|stack|directory)\b")


def is_file(flag: Flag) -> bool:
    """Whether a flag names a file (a field with a Browse button): an input, or one without a value by default
    whose help names a file, a mask, a map..."""
    if flag.is_switch:
        return False
    return is_input(flag) or ((flag.default or "") in ("", *NONE_DEFAULTS)
                              and bool(_FILE_WORDS.search(flag.help.lower())))


def flag_option(flag: Flag) -> OptionDef:
    """A flag as an option of the form: a switch as Yes/No, a file as a file field, else a text field, with
    the program's default."""
    if flag.is_switch:
        return OptionDef(flag.name, "boolean", flag.name, "No", flag.help)
    default = "" if flag.default is None or flag.default in NONE_DEFAULTS else flag.default
    return OptionDef(flag.name, "filename" if is_file(flag) else "any", flag.name, default, flag.help)


class _Focus(QObject):
    """Tells the panel which flag the mouse or the focus is on (its help is shown below the tabs)."""

    def __init__(self, panel: "AdvancedPanel", flag: Flag):
        super().__init__(panel)
        self.panel, self.flag = panel, flag

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.FocusIn, QEvent.Enter):
            self.panel.flag_focused.emit(self.flag)
        return False


@dataclass
class Section:
    name: str
    head: QLabel
    line: QFrame
    names: list[str]

    def set_visible(self, on: bool) -> None:
        self.head.setVisible(on)
        self.line.setVisible(on)


@dataclass
class FlagRow:
    flag: Flag
    option: OptionDef
    label: QLabel
    widget: QWidget  # an OptionWidget of the job editor
    marker: QLabel

    def set_visible(self, on: bool) -> None:
        for w in (self.label, self.widget, self.marker):
            w.setVisible(on)


class AdvancedPanel(QWidget):
    edited = Signal(str)  # the new additional-arguments text
    flag_focused = Signal(object)  # Flag

    def __init__(self, make_field: Callable[[OptionDef], QWidget], parent=None):
        """``make_field``: the job editor's widget for an option (job_editor.make_widget)."""
        super().__init__(parent)
        self.make_field = make_field
        self.program = ""
        self.flags: ProgramFlags | None = None
        self.set_by_job: dict[str, JobFlag] = {}  # what the job writes itself (left out)
        self.option_label: Callable[[str], str] = lambda k: k
        self.text = ""
        self.rows: dict[str, FlagRow] = {}
        self.sections: list[Section] = []
        self._built: list[QWidget] = []  # what a rebuild replaces
        self._stretch_row = FIRST_FLAG_ROW
        self._syncing = False
        self._editable = True

        self.grid = QGridLayout(self)
        self.grid.setColumnStretch(1, 1)
        self.grid.setVerticalSpacing(4)
        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setTextFormat(Qt.RichText)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.RichText)
        self.notice.setVisible(False)
        self.grid.addWidget(self.header, 1, 0, 1, 3)
        self.grid.addWidget(self.notice, 2, 0, 1, 3)
        self.grid.setRowStretch(self._stretch_row, 1)

    def place_box(self, label: QLabel, field: QWidget, marker: QLabel) -> None:
        """The editor's "Additional arguments:" row, at the top."""
        self.grid.addWidget(label, 0, 0)
        self.grid.addWidget(field, 0, 1)
        self.grid.addWidget(marker, 0, 2)

    # ------------------------------------------------------------------ content
    def set_program(self, program: str, set_by_job: dict[str, JobFlag], option_label: Callable[[str], str],
                    loading: bool = False) -> None:
        """The program that gets the arguments now, and the flags the job writes itself. ``loading``: its
        --help is being read (set_flags follows)."""
        self.program, self.flags = program, None
        self.set_by_job, self.option_label = set_by_job, option_label
        self._rebuild()
        if loading:
            self.header.setText(f"Reading the flags of <b>{html.escape(program)}</b> …")

    def set_flags(self, program: str, flags: ProgramFlags | None) -> None:
        self.program, self.flags = program, flags
        self._rebuild()

    def hidden(self) -> list[str]:
        """The program's flags left out, as the job writes them itself (in the program's order)."""
        return [f.name for f in (self.flags.flags if self.flags else []) if f.name in self.set_by_job]

    def _rebuild(self) -> None:
        self._syncing = True
        for w in self._built:
            w.setParent(None)
            w.deleteLater()
        self._built.clear()
        self.rows.clear()
        self.sections.clear()
        self.grid.setRowStretch(self._stretch_row, 0)
        r = FIRST_FLAG_ROW
        if self.flags is None:
            self.header.setText(self._no_flags_text())
        else:
            dim = self.palette().color(QPalette.PlaceholderText).name()
            for f in self.flags.flags:
                if f.name in self.set_by_job:
                    continue
                section = f.section or "Options"
                if not self.sections or self.sections[-1].name != section:
                    head = QLabel(section)
                    head.setTextFormat(Qt.PlainText)
                    font = QFont(self.font())
                    font.setBold(True)
                    head.setFont(font)
                    head.setContentsMargins(0, 8 if self.sections else 2, 0, 0)
                    line = QFrame()
                    line.setFrameShape(QFrame.HLine)
                    line.setFrameShadow(QFrame.Sunken)
                    self.grid.addWidget(head, r, 0, 1, 3)
                    self.grid.addWidget(line, r + 1, 0, 1, 3)
                    self._built += [head, line]
                    self.sections.append(Section(section, head, line, []))
                    r += 2
                opt = flag_option(f)
                label = QLabel(f.name)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setToolTip(self._tooltip(f, opt))
                w = self.make_field(opt)
                w.setToolTip(self._tooltip(f, opt))
                edit = getattr(w, "edit", None)
                if edit is not None and not opt.default and f.default:
                    edit.setPlaceholderText(f.default)  # "None": what the program takes without the flag
                marker = QLabel()
                marker.setFixedWidth(12)
                w.changed.connect(lambda name=f.name: self._on_changed(name))
                focus = _Focus(self, f)
                for target in {w, w.focus_target(), label}:
                    target.installEventFilter(focus)
                self.grid.addWidget(label, r, 0)
                self.grid.addWidget(w, r, 1)
                self.grid.addWidget(marker, r, 2)
                self._built += [label, w, marker]
                self.rows[f.name] = FlagRow(f, opt, label, w, marker)
                self.sections[-1].names.append(f.name)
                r += 1
            hidden = self.hidden()
            left_out = ""
            if hidden:
                left_out = (f" Not listed: the <span style='text-decoration:underline dotted'>{len(hidden)} flags "
                            "the job writes itself</span> (point here to see them).")
            self.header.setText(
                f"<b>{len(self.rows)}</b> flags of <b>{html.escape(self.program)}</b>, by the sections of its "
                "--help. A changed one is added to the box above, and is in bold."
                f"<span style='color:{dim}'>{left_out} From {html.escape(self.flags.source)}.</span>")
            self.header.setToolTip(self._hidden_tooltip(hidden))
        self._stretch_row = r
        self.grid.setRowStretch(r, 1)
        self._syncing = False
        self.set_editable(self._editable)
        self.set_text(self.text)

    def _tooltip(self, f: Flag, opt: OptionDef) -> str:
        default = "off" if f.is_switch else ("none (required)" if f.default is None else (f.default or "—"))
        return f"{f.name}   (default: {default})\n\n{f.help}"

    def _hidden_tooltip(self, hidden: list[str]) -> str:
        if not hidden:
            return ""
        by_options, always = [], []
        for name in hidden:
            jf = self.set_by_job[name]
            if jf.options:
                labels = dict.fromkeys(self.option_label(k) for k in jf.options)
                by_options.append(f"{name}: {', '.join(labels)}")
            else:
                always.append(name)
        out = ["Flags the job writes itself, not listed here (typed in the box, a value overrides the job's: "
               "RELION uses the last one given)."]
        if by_options:
            out += ["", "Set by the options on the other tabs:"] + ["  " + t for t in by_options]
        if always:
            out += ["", "Always written by the job: " + " ".join(always)]
        return "\n".join(out)

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

    # ------------------------------------------------------------------ the box <-> the rows
    def set_text(self, text: str) -> None:
        """The additional-arguments box changed: fill in the rows of the flags it names."""
        self.text = text
        given = parse_args(text)
        accent = self.palette().color(QPalette.Highlight).name()
        self._syncing = True
        for name, row in self.rows.items():
            on = name in given
            w = row.widget
            if not w.focus_target().hasFocus():  # not under the cursor of the one being typed in
                if row.flag.is_switch:
                    w.set_value("Yes" if on else "No")
                else:
                    w.set_value((given[name][0] if given[name] else "") if on else row.option.default)
            font = row.label.font()
            font.setBold(on)
            row.label.setFont(font)
            row.marker.setText(f"<span style='color:{accent}'>●</span>" if on else "")
            row.marker.setToolTip("in the additional arguments" if on else "")
        self._syncing = False
        notes = []
        if self.flags is not None:
            unknown = [n for n in given if n not in self.rows and n not in self.set_by_job]
            if unknown:
                notes.append(f"<span style='color:#c0392b'>Not a flag of {html.escape(self.program)}: "
                             f"{html.escape(', '.join(unknown))}</span>")
            override = [n for n in given if n in self.set_by_job and self.flags.get(n) is not None]
            if override:
                notes.append("<span style='color:#d9861c'>Also written by the job (from its options or always); "
                             f"the value here wins, as RELION uses the last one given: {html.escape(', '.join(override))}"
                             "</span>")
        self.notice.setText("<br>".join(notes))
        self.notice.setVisible(bool(notes))

    def _on_changed(self, name: str) -> None:
        if self._syncing:
            return
        row = self.rows[name]
        f = row.flag
        if f.is_switch:
            value = "" if row.widget.value() == "Yes" else None
        else:
            typed = row.widget.value().strip()
            value = typed if typed and typed != row.option.default else None  # the default: not passed
        text = set_arg(self.text, name, value, switch=f.is_switch)
        if text != self.text:
            self.text = text
            self.edited.emit(text)

    def set_editable(self, on: bool) -> None:
        """Like the box (continue mode, a read-only project): the flag rows can be changed or not."""
        self._editable = on
        for row in self.rows.values():
            row.widget.setEnabled(on)

    # ------------------------------------------------------------------ search
    def apply_filter(self, text: str, only_changed: bool) -> int:
        """Show the flags that match the editor's "Find option" text (and are in the box, with "Only
        changed"); the number shown."""
        given = parse_args(self.text) if only_changed else {}
        shown = 0
        for sec in self.sections:
            any_shown = False
            for name in sec.names:
                row = self.rows[name]
                show = (not text or text in self._hay(row.flag)) and (not only_changed or name in given)
                row.set_visible(show)
                any_shown |= show
                shown += show
            sec.set_visible(any_shown)
        return shown

    @staticmethod
    def _hay(f: Flag) -> str:
        return f"{f.name} {f.help} {f.section}".lower()

    def matches(self, text: str) -> bool:
        """Whether a flag listed here matches the "Find option" text."""
        return bool(text) and any(text in self._hay(r.flag) for r in self.rows.values())

    def update_pipeline(self, fill: Callable[[QWidget], None]) -> None:
        """Refresh the lists of the input fields (``fill(widget)``)."""
        for row in self.rows.values():
            fill(row.widget)
