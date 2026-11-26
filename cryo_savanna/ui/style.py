# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Colours, icons and small formatting helpers shared by the views."""

from __future__ import annotations

import time

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap

from cryo_savanna.relion_consts import job_type_for_label

# Preferences → Theme: key, what the menu says. Always one of these, set by cryo-savanna: Qt can tell the
# system's dark mode on some desktops only (GNOME), so it is not followed
THEMES = {"light": "Light", "dark": "Dark"}

# Fusion's own colours, written out: the style's palette follows the system's dark mode on recent Qt
_ROLES = {  # role: (light, dark)
    "Window": ("#efefef", "#323232"),
    "WindowText": ("#000000", "#e6e6e6"),
    "Base": ("#ffffff", "#262626"),
    "AlternateBase": ("#f7f7f7", "#2d2d2d"),
    "ToolTipBase": ("#ffffdc", "#3c3c3c"),
    "ToolTipText": ("#000000", "#e6e6e6"),
    "PlaceholderText": ("#808080", "#8c8c8c"),
    "Text": ("#000000", "#e6e6e6"),
    "Button": ("#efefef", "#3a3a3a"),
    "ButtonText": ("#000000", "#e6e6e6"),
    "BrightText": ("#ffffff", "#ffffff"),
    "Light": ("#ffffff", "#505050"),
    "Midlight": ("#cacaca", "#444444"),
    "Mid": ("#b8b8b8", "#5a5a5a"),
    "Dark": ("#9f9f9f", "#202020"),
    "Shadow": ("#767676", "#141414"),
    "Highlight": ("#308cc6", "#2f7fc9"),
    "HighlightedText": ("#ffffff", "#ffffff"),
    "Link": ("#0000ff", "#5aa9f0"),
    "LinkVisited": ("#ff00ff", "#c58af0"),
}
_DISABLED = {  # role: (light, dark), greyed out
    "WindowText": ("#bebebe", "#7a7a7a"),
    "Text": ("#bebebe", "#7a7a7a"),
    "ButtonText": ("#bebebe", "#7a7a7a"),
    "Base": ("#efefef", "#2c2c2c"),
    "Highlight": ("#919191", "#4a4a4a"),
    "HighlightedText": ("#ffffff", "#9a9a9a"),
}


def theme_palette(theme: str) -> QPalette:
    """The light or dark palette of Preferences → Theme."""
    k = 1 if theme == "dark" else 0
    pal = QPalette()
    for role, colors in _ROLES.items():
        pal.setColor(getattr(QPalette, role), QColor(colors[k]))
    for role, colors in _DISABLED.items():
        pal.setColor(QPalette.Disabled, getattr(QPalette, role), QColor(colors[k]))
    return pal


def is_dark(pal: QPalette) -> bool:
    return pal.color(QPalette.Window).lightness() < 128


def apply_theme(app, theme: str) -> None:
    """Preferences → Theme for the whole application, open windows included (anything but "dark" is light)."""
    theme = "dark" if theme == "dark" else "light"
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):  # Qt 6.8: title bars and system dialogs follow too
        scheme = Qt.ColorScheme
        hints.setColorScheme(scheme.Dark if theme == "dark" else scheme.Light)
    app.setPalette(theme_palette(theme))
    from cryo_savanna.viewers import apply_plot_theme
    apply_plot_theme(app.palette())


def input_tint(pal: QPalette) -> QColor:
    """Background of a row of the List that the open job editor takes inputs from: INPUT_COLOR over Base."""
    base, c, a = pal.color(QPalette.Base), INPUT_COLOR, 0.35 if is_dark(pal) else 0.22
    return QColor(*(round(x * a + y * (1 - a)) for x, y in ((c.red(), base.red()), (c.green(), base.green()),
                                                             (c.blue(), base.blue()))))


def input_text(labels: list[str]) -> str:
    """What a job is to the open job editor: its fields that take this job's outputs."""
    return "Input of the job editor: " + ", ".join(labels)


def excluded_row_color(pal: QPalette) -> QColor:
    """Background of excluded rows (tilt images): red, light or dark to keep the text readable."""
    return QColor(110, 40, 40) if is_dark(pal) else QColor(250, 200, 200)

# Readable on both light and dark Fusion palettes
STATUS_COLORS = {
    "Succeeded": QColor("#2e9d4f"),
    "Running": QColor("#1f7ad0"),
    "Scheduled": QColor("#8a63c8"),
    "Failed": QColor("#d43c3c"),
    "Aborted": QColor("#d9861c"),
}
UNKNOWN_COLOR = QColor("#888888")
# jobs whose outputs the open job editor takes as inputs (List and Tree): not a status colour, not the selection's
INPUT_COLOR = QColor("#c93f9a")
STAR = "★"  # starred jobs (project/stars.py)
STAR_COLOR = QColor("#e0a100")

_icons: dict[tuple[str, bool], QIcon] = {}


def status_color(status: str) -> QColor:
    return STATUS_COLORS.get(status, UNKNOWN_COLOR)


def status_icon(status: str, pending: bool = False) -> QIcon:
    """Filled dot in the status colour; a ring when the pipeline has not caught up yet."""
    key = (status, pending)
    if key not in _icons:
        pm = QPixmap(14, 14)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        c = status_color(status)
        if pending:
            p.setPen(c)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(2.5, 2.5, 9, 9))
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(QRectF(2, 2, 10, 10))
        p.end()
        _icons[key] = QIcon(pm)
    return _icons[key]


QUEUE_COLORS = {"queued": QColor("#d9861c"), "running": STATUS_COLORS["Running"], "lost": STATUS_COLORS["Failed"]}


def queue_entry(state, job: str) -> tuple[str, str, str]:
    """(short text, details, kind) of a job in the SLURM queue (a slurm.QueueState); kind is
    "queued", "running", "lost" (ended in SLURM, RELION not told) or "" (not in the queue)."""
    if state is None:
        return "", "", ""
    q = state.live.get(job)
    if q is not None:
        return q.short(), q.describe(), "queued" if q.pending else "running"
    a = state.lost.get(job)
    if a is not None:
        return a.short(), lost_text(a), "lost"
    return "", "", ""


def slurm_elapsed(state, job: str) -> float | None:
    """How long a job running in SLURM has run (its SLURM job's time used), else None."""
    q = state.live.get(job) if state is not None else None
    return q.elapsed_seconds() if q is not None else None


def lost_details(a) -> str:
    bits = [f"exit code {a.exit_code}"] if a.exit_code else []
    if a.elapsed:
        bits.append(f"after {a.elapsed}")
    if a.max_rss:
        bits.append(f"max. memory {a.max_rss}")
    return ", ".join(bits)


def lost_text(a) -> str:
    details = lost_details(a)
    return (f"SLURM job {a.id} ended: {a.state}" + (f" ({details})" if details else "") +
            ". RELION was not told (no exit file): cryo-savanna writes it, so the job is shown as failed "
            "(aborted when cancelled). In a read-only window it stays Running: use Job → Mark as failed "
            "in a window that may change the project.")


def held_text(status: str, held) -> str:
    """What a run still going does with a scheduled job (a pipeliner.Held), for the list and the cards: "waiting
    for a GPU", "waiting for threads", "waiting for input", "after job012" (its run starts it once that job, the
    one it is at, has ended), "in a run" (about to be started); "" when no run holds it."""
    if status != "Scheduled" or held is None:
        return ""
    gpus, threads = held.waiting_for_gpus, getattr(held, "waiting_for_threads", False)
    if gpus and threads:
        return "waiting for GPU/threads"
    if gpus or threads:
        return "waiting for a GPU" if gpus else "waiting for threads"
    if held.waiting:
        return "waiting for input"
    after = getattr(held, "after", "")
    if not held.current and after:
        return f"after {after.rstrip('/').split('/')[-1]}"
    return "in a run"


def status_text(status: str, pending: bool) -> str:
    return f"{status} (pipeline not updated yet)" if pending else status


def job_label(name: str, alias: str) -> str:
    """Tree/graph text for a job: alias when there is one, else Type/jobNNN."""
    if alias not in ("", "None"):
        return f"{alias.rstrip('/')}  ({name.rstrip('/').split('/')[-1]})"
    return name.rstrip("/")


def type_name(type_label: str, names: dict[str, str]) -> str:
    jt = job_type_for_label(type_label)
    if jt is None:
        return type_label
    return names.get(jt.label, jt.dirname)


def job_type(info, names: dict[str, str]) -> str:
    """Type of a job as shown in the tree, the graph and the overview: with the program it runs
    for External jobs ("External: relion_image_handler")."""
    t = type_name(info.process.type_label, names)
    return f"{t}: {info.program}" if info.program else t


def ago(t: float | None, now: float | None = None) -> str:
    if t is None:
        return ""
    d = max(0.0, (now or time.time()) - t)
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        return f"{int(d // 3600)} h ago"
    if d < 86400 * 60:
        return f"{int(d // 86400)} d ago"
    return time.strftime("%Y-%m-%d", time.localtime(t))


def timestamp(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else ""
