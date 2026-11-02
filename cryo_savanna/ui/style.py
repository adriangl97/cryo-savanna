# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Colours, icons and small formatting helpers shared by the views."""

from __future__ import annotations

import time

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap

from cryo_savanna.relion_consts import job_type_for_label

# Readable on both light and dark Fusion palettes
STATUS_COLORS = {
    "Succeeded": QColor("#2e9d4f"),
    "Running": QColor("#1f7ad0"),
    "Scheduled": QColor("#8a63c8"),
    "Failed": QColor("#d43c3c"),
    "Aborted": QColor("#d9861c"),
}
UNKNOWN_COLOR = QColor("#888888")

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
