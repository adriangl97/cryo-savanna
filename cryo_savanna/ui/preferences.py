# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Preferences (kept in QSettings) and the Help menu's keyboard shortcuts and About boxes."""

from __future__ import annotations

import os

from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.images.cache import cache_root, cache_size_mb, clear_cache
from cryo_savanna.project import scratch
from cryo_savanna.ui.style import THEMES

# key, default, what it is
DEFAULTS = {
    "poll_s": 2.0,  # seconds between looks at the project
    "queue_poll_s": 20.0,  # seconds between looks at the SLURM queue (0 = off)
    "cache_mb": 2048,  # micrograph previews kept on disk
    "chimerax": "",  # ChimeraX to use instead of the one on PATH
    "theme": "system",  # colours of the window: system, light or dark (style.THEMES)
    "editor": "window",  # where the job editor opens (EDITOR_PLACES)
    "scratch_dir": "",  # default "Copy particles to scratch directory" of new jobs (as $RELION_SCRATCH_DIR)
    "scratch_own": False,  # each job a savanna_NNN subdirectory of its scratch directory (off: as RELION's GUI)
}

SCRATCH_VAR = scratch.ENV_VAR

EDITOR_PLACES = {"window": "In its own window", "docked": "Docked in the main window"}
CHOICES = {"theme": THEMES, "editor": EDITOR_PLACES}  # preferences with a fixed set of values

# keys handled inside the viewers (not QActions, so not found by shortcuts_html's scan)
VIEWER_KEYS = [
    ("Micrographs, tilts, tables", "↑ / ↓", "Previous / next row"),
    ("Images", "Wheel / drag", "Zoom / pan; zoomed in, the region is read again at full detail"),
    ("3D surface", "Drag / wheel", "Rotate (as in ChimeraX) / zoom"),
    ("Galleries", "Ctrl+click, Shift+click", "Select several; double-click opens one image"),
    ("Select classes", "Click", "Select or unselect a class (as relion_display)"),
    ("Exclude tilt images", "Enter or Space", "Keep / exclude the tilt image, then go to the next"),
    ("Exclude tilt images", "[ / ]", "Previous / next tilt series"),
    ("Selection windows", "Ctrl+S", "Save the selection"),
]


def setting(settings: QSettings, key: str):
    """A preference with its type (QSettings gives strings back on some platforms)."""
    default = DEFAULTS[key]
    v = settings.value(key, default)
    if isinstance(default, bool):
        return str(v).lower() not in ("false", "0", "no")
    if isinstance(default, float):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    if isinstance(default, int):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default
    v = str(v or "")
    if key in CHOICES and v not in CHOICES[key]:
        return default
    return v


_set_here: str | None = None  # the CRYO_SAVANNA_CHIMERAX value these preferences put in the environment


def apply_environment(settings: QSettings) -> None:
    """Preferences that work through the environment: a ChimeraX chosen here (a CRYO_SAVANNA_CHIMERAX
    set in the shell wins)."""
    global _set_here
    current = os.environ.get("CRYO_SAVANNA_CHIMERAX")
    if current is not None and current != _set_here:
        return  # the shell's
    chimerax = setting(settings, "chimerax")
    if chimerax:
        os.environ["CRYO_SAVANNA_CHIMERAX"] = _set_here = chimerax
    elif current is not None:
        del os.environ["CRYO_SAVANNA_CHIMERAX"]
        _set_here = None


def apply_relion_defaults(settings: QSettings, env: RelionEnv) -> None:
    """Preferences that act as RELION_* variables for the job editor's defaults: the scratch
    directory, as $RELION_SCRATCH_DIR (one set in the shell wins)."""
    env.set_default(SCRATCH_VAR, setting(settings, "scratch_dir"))


class PreferencesDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None, env: RelionEnv | None = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.settings = settings
        form = QFormLayout()
        self.theme = QComboBox()
        for key, text in THEMES.items():
            self.theme.addItem(text, key)
        self.theme.setCurrentIndex(self.theme.findData(setting(settings, "theme")))
        self.theme.setToolTip("Light or dark colours; as the system follows its dark mode where Qt can tell")
        form.addRow("Theme", self.theme)
        self.editor = QComboBox()
        for key, text in EDITOR_PLACES.items():
            self.editor.addItem(text, key)
        self.editor.setCurrentIndex(self.editor.findData(setting(settings, "editor")))
        self.editor.setToolTip("Where New job, Clone, Continue and the Job editor action (Ctrl+E) open the job editor")
        form.addRow("Job editor", self.editor)
        self.poll = QDoubleSpinBox()
        self.poll.setRange(0.5, 600)
        self.poll.setSuffix(" s")
        self.poll.setValue(setting(settings, "poll_s"))
        self.poll.setToolTip("How often cryo-savanna looks for changes made by RELION (--poll)")
        form.addRow("Look for changes every", self.poll)
        self.queue = QDoubleSpinBox()
        self.queue.setRange(0, 3600)
        self.queue.setSuffix(" s")
        self.queue.setSpecialValueText("never")
        self.queue.setValue(setting(settings, "queue_poll_s"))
        self.queue.setToolTip("How often squeue is asked (--queue-poll); never = no SLURM monitor")
        form.addRow("Look at the SLURM queue every", self.queue)
        self.cache = QSpinBox()
        self.cache.setRange(0, 1_000_000)
        self.cache.setSuffix(" MB")
        self.cache.setValue(setting(settings, "cache_mb"))
        self.cache.setToolTip("Micrograph previews kept on disk, for all projects together; the oldest go first")
        clear = QPushButton("Empty it now")
        clear.clicked.connect(self._clear_cache)
        self.cache_info = QLabel()
        self._cache_label()
        row = QHBoxLayout()
        row.addWidget(self.cache)
        row.addWidget(clear)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Image cache", holder)
        form.addRow("", self.cache_info)
        self.chimerax = QLineEdit(setting(settings, "chimerax"))
        self.chimerax.setPlaceholderText("chimerax on PATH")
        self.chimerax.setToolTip("ChimeraX to start for Open in ChimeraX ($CRYO_SAVANNA_CHIMERAX, if set, wins)")
        form.addRow("ChimeraX", self.chimerax)
        self.scratch = QLineEdit(setting(settings, "scratch_dir"))
        shell = env.get(SCRATCH_VAR) if env is not None and env.from_shell(SCRATCH_VAR) else None
        self.scratch.setPlaceholderText(f"${SCRATCH_VAR} = {shell} (wins)" if shell is not None else "none")
        self.scratch.setToolTip("Default of 'Copy particles to scratch directory' for new jobs ($RELION_SCRATCH_DIR, "
                                "if set in the shell, wins)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_scratch)
        row = QHBoxLayout()
        row.addWidget(self.scratch)
        row.addWidget(browse)
        row.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Scratch directory", holder)
        self.scratch_own = QCheckBox("A subdirectory of its own for each job (savanna_NNN)")
        self.scratch_own.setChecked(setting(settings, "scratch_own"))
        self.scratch_own.setToolTip(
            "Each job started with a scratch directory copies its particles to <directory>/savanna_NNN, with a "
            "random 3-digit number no other job of the project uses, so jobs sharing a scratch disk never wipe "
            "each other's relion_volatile/. RELION removes only relion_volatile/ of a job that finishes well: "
            "the copies of failed or aborted jobs stay until you delete them. Off: the directory as typed, as "
            "in RELION's own GUI.")
        wipe = QPushButton("Wipe savanna folders…")
        wipe.setToolTip("Delete every savanna_NNN folder directly in the scratch directory, after showing them. "
                        "Jobs running with their scratch in one of them will crash.")
        wipe.clicked.connect(self._wipe_scratch)
        row = QHBoxLayout()
        row.addWidget(self.scratch_own)
        row.addStretch(1)
        row.addWidget(wipe)
        row.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("", holder)
        self._env = env
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("<span style='color:gray'>Command-line options (--poll, --queue-poll) win over these "
                             "for that session.</span>"))
        lay.addWidget(buttons)

    def _cache_label(self) -> None:
        self.cache_info.setText(f"<span style='color:gray'>{cache_root()} — {cache_size_mb():.0f} MB now</span>")

    def _browse_scratch(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Scratch directory", self.scratch.text().strip())
        if d:
            self.scratch.setText(d)

    def scratch_base(self) -> str:
        """The scratch directory new jobs get: $RELION_SCRATCH_DIR from the shell, else the one typed here."""
        env = self._env
        if env is not None and env.from_shell(SCRATCH_VAR):
            return (env.get(SCRATCH_VAR) or "").strip()
        return self.scratch.text().strip()

    def _wipe_scratch(self) -> None:
        base = self.scratch_base()
        title = "Wipe savanna folders"
        if not base:
            QMessageBox.information(self, title, "No scratch directory is set: there is nothing to search.")
            return
        dirs = scratch.own_dirs(base)
        if not dirs:
            QMessageBox.information(self, title, f"No savanna_NNN folders in {base}.")
            return
        if WipeScratchDialog(base, dirs, self).exec() != QDialog.Accepted:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            errors = scratch.wipe(dirs)
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(self, title, f"Deleted {len(dirs) - len(errors)} of {len(dirs)} folders. "
                                "These could not be deleted:\n\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, title, f"Deleted {len(dirs)} folder{'s' if len(dirs) != 1 else ''} "
                                    f"from {base}.")

    def _clear_cache(self) -> None:
        clear_cache()  # only cryo-savanna's own files: the root may hold other things
        self._cache_label()

    def accept(self) -> None:
        s = self.settings
        s.setValue("poll_s", self.poll.value())
        s.setValue("queue_poll_s", self.queue.value())
        s.setValue("cache_mb", self.cache.value())
        s.setValue("chimerax", self.chimerax.text().strip())
        s.setValue("scratch_dir", self.scratch.text().strip())
        s.setValue("scratch_own", self.scratch_own.isChecked())
        s.setValue("theme", self.theme.currentData())
        s.setValue("editor", self.editor.currentData())
        super().accept()


class WipeScratchDialog(QDialog):
    """Shows the savanna_NNN folders that would be deleted, and the risk; Cancel is the default."""

    def __init__(self, base: str, dirs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wipe savanna folders")
        n = len(dirs)
        icon = QLabel()
        icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxWarning).pixmap(40, 40))
        self.warning = text = QLabel(
            f"<b>Delete {n} savanna_NNN folder{'s' if n != 1 else ''} in {base}, and everything in them?</b>"
            "<p>Jobs that are running now with their scratch directory in one of these folders <b>will "
            "crash</b>: RELION reads their particles from there. Jobs queued to start later are not affected "
            "(RELION makes the folder again). Wipe them only when no job using scratch is running.</p>"
            "<p>This cannot be undone.</p>")
        text.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(icon, 0, Qt.AlignTop)
        top.addWidget(text, 1)
        self.paths = QPlainTextEdit("\n".join(str(d) for d in dirs))
        self.paths.setReadOnly(True)
        self.paths.setLineWrapMode(QPlainTextEdit.NoWrap)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.delete = buttons.addButton(f"Delete {n} folder{'s' if n != 1 else ''}", QDialogButtonBox.DestructiveRole)
        self.delete.setAutoDefault(False)
        cancel = buttons.button(QDialogButtonBox.Cancel)
        cancel.setDefault(True)
        cancel.setFocus()
        self.delete.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(QLabel("Folders to delete:"))
        lay.addWidget(self.paths, 1)
        lay.addWidget(buttons)
        self.resize(620, 420)


def shortcuts_html(window) -> str:
    """Every keyboard shortcut of the window's actions, and the keys of the viewers."""
    rows = []
    seen = set()
    for a in window.findChildren(type(window.actions.act_clone)):
        keys = a.shortcut().toString(QKeySequence.NativeText)
        text = a.text().replace("&", "").rstrip("…")
        if keys and (text, keys) not in seen:
            seen.add((text, keys))
            rows.append((text, keys))
    rows.sort(key=lambda r: r[0].lower())
    out = ["<h3>Menus and toolbar</h3><table cellpadding=3>"]
    out += [f"<tr><td><b>{k}</b></td><td>{t}</td></tr>" for t, k in rows]
    out.append("</table><h3>Viewers</h3><table cellpadding=3>")
    out += [f"<tr><td><b>{k}</b></td><td>{w}: {t}</td></tr>" for w, k, t in VIEWER_KEYS]
    out.append("</table>")
    return "\n".join(out)


class TextDialog(QDialog):
    def __init__(self, title: str, html: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setHtml(html)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addWidget(buttons)
        self.resize(560, 620)
        self.setAttribute(Qt.WA_DeleteOnClose)
