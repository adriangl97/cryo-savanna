# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Preferences (kept in QSettings) and the Help menu's keyboard shortcuts and About boxes."""

from __future__ import annotations

import os

from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.images.cache import cache_root, cache_size_mb, clear_cache

# key, default, what it is
DEFAULTS = {
    "poll_s": 2.0,  # seconds between looks at the project
    "queue_poll_s": 20.0,  # seconds between looks at the SLURM queue (0 = off)
    "cache_mb": 2048,  # micrograph previews kept on disk
    "chimerax": "",  # ChimeraX to use instead of the one on PATH
}

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
    return str(v or "")


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


class PreferencesDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.settings = settings
        form = QFormLayout()
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

    def _clear_cache(self) -> None:
        clear_cache()  # only cryo-savanna's own files: the root may hold other things
        self._cache_label()

    def accept(self) -> None:
        s = self.settings
        s.setValue("poll_s", self.poll.value())
        s.setValue("queue_poll_s", self.queue.value())
        s.setValue("cache_mb", self.cache.value())
        s.setValue("chimerax", self.chimerax.text().strip())
        super().accept()


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
