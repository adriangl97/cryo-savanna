# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The results of a CtfRefine job: a summary, the per-particle CTF changes, the aberration images.

* ``CtfRefineSummary``: what the job fitted, how the CTF values of the particles changed, and per
  optics group the beam tilt, trefoil, 4th-order and magnification values before and after.
* ``CtfChangesView``: histograms / scatter of the new values and of the changes, the particles of a
  micrograph coloured by a value (as RELION's per-micrograph plots in logfile.pdf), and the mean
  change per micrograph.
* ``AberrationsView``: per optics group, the phase differences the data show in Fourier space next
  to the fitted model and what the fit leaves (RELION's plots in logfile.pdf, from its .mrc files).
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.child import external_environ
from cryo_savanna.images.mrc import ImageRef, header, read_image
from cryo_savanna.results import ctfrefine as C
from cryo_savanna.results.tables import Table
from cryo_savanna.viewers.plots import TablePlot
from cryo_savanna.viewers.tasks import TaskError, Tasks

DIVERGING = "CET-D1"  # blue - white - red, for signed values
SEQUENTIAL = "viridis"


def _stamp(path: Path | None):
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _fmt(v: float | None, digits: int = 3) -> str:
    """A number for the summary: ``digits`` decimals without trailing zeros, small ones in e-notation."""
    if v is None or not np.isfinite(v):
        return "–"
    if v != 0 and abs(v) < 10.0 ** -digits:
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _signed(v: float | None, digits: int = 3) -> str:
    s = _fmt(v, digits)
    return ("+" + s) if v is not None and np.isfinite(v) and v > 0 else s


def open_with_system(path: Path) -> str:
    """Open a file with the desktop's viewer; an error message, or ""."""
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, str(path)], start_new_session=True, env=external_environ(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return f"{opener}: {e}"
    return ""


# ---------------------------------------------------------------------------- summary

CHANGE_ROWS = [  # (column, label): the statistics table, in this order
    (C.D_DEFOCUS, "Defocus change"), (C.D_ASTIG, "Astigmatism change"), (C.D_ANGLE, "Astigmatism angle change"),
    (C.D_PHASE, "Phase shift change"), (C.D_BFACTOR, "B-factor change"), (C.D_SCALE, "Scale factor change"),
    (C.DEFOCUS, "Defocus"), (C.ASTIG, "Astigmatism"), ("rlnCtfBfactor", "B-factor"),
    ("rlnCtfScalefactor", "Scale factor"), ("rlnPhaseShift", "Phase shift"),
]
MAG_WARNING = 0.005  # as relion_ctf_refine: an overall scale this far from 1 changes the pixel size


def summary_html(options: dict[str, str], groups: list[C.OpticsGroup], changes: C.ParticleChanges | None,
                 output: str, input_star: str, done: int | None = None) -> str:
    """The summary page (HTML for a QTextBrowser; colours left to the palette)."""
    h = html.escape
    out: list[str] = []
    t = changes.table if changes else None
    n_parts = len(t) if t is not None else 0
    n_mics = len(set(t.column("rlnMicrographName"))) if t is not None and t.has("rlnMicrographName") else 0
    if changes is None:
        out.append("<p><b>No refined particles yet.</b>"
                   + (f" Per-micrograph fits done for {done} micrograph{'s' if done != 1 else ''}." if done else "")
                   + "</p>")
    else:
        line = f"<p><b>{n_parts:,}</b> particles" + (f" on <b>{n_mics:,}</b> micrographs" if n_mics else "")
        line += f" in <tt>{h(output)}</tt>"
        if input_star:
            line += f", from <tt>{h(input_star)}</tt>"
            if changes.matched < n_parts:
                line += f" ({changes.matched:,} of them found there)"
        out.append(line + ".</p>")

    fits = C.fitted(options)
    if fits:
        res = C.min_resolution(options)
        out.append("<h3>Fitted</h3><table cellspacing=0 cellpadding=3>")
        for what, how in fits:
            out.append(f"<tr><td>{h(what)}</td><td>{h(how)}</td></tr>")
        out.append("</table>")
        if res:
            out.append(f"<p>Fits from {res:g} Å to the resolution of the reference.</p>")

    if t is not None:
        rows = []
        for col, label in CHANGE_ROWS:
            s = C.stat(t, col)
            if s is None or (s.sd == 0 and col != C.D_DEFOCUS):  # not fitted (B-factor 0, scale 1, ...)
                continue
            unit = C.UNITS.get(col, "")
            change = col.endswith("Change")
            f = _signed if change else _fmt
            rows.append(f"<tr><td>{h(label)}{f' ({unit})' if unit else ''}</td><td align=right>{f(s.mean)}</td>"
                        f"<td align=right>{_fmt(s.sd)}</td><td align=right>{f(s.median)}</td>"
                        f"<td align=right>{_fmt(s.median_abs) if change else ''}</td>"
                        f"<td align=right>{f(s.lo)} … {f(s.hi)}</td></tr>")
        if rows:
            out.append("<h3>Particles</h3><table cellspacing=0 cellpadding=3 border=0>"
                       "<tr><th align=left></th><th>Mean</th><th>SD</th><th>Median</th><th>Median |Δ|</th>"
                       "<th>1–99 %</th></tr>" + "".join(rows) + "</table>")
        if not changes.has_changes and t.has(C.DEFOCUS):
            out.append("<p>The input particles were not found, so the changes are not shown.</p>")

    for g in groups:
        out.append(_group_html(g))
    return "\n".join(out)


def _group_html(g: C.OpticsGroup) -> str:
    h = html.escape
    out = [f"<h3>Optics group {g.number}: {h(g.name)}</h3>"]
    facts = []
    if g.pixel_size:
        facts.append(f"{g.pixel_size:g} Å/px")
    if g.box:
        facts.append(f"box {g.box} px")
    if g.voltage:
        facts.append(f"{g.voltage:g} kV")
    if g.cs is not None:
        facts.append(f"Cs {g.cs:g} mm" + (f" (was {g.cs_before:g})" if g.cs_before is not None and g.cs_before != g.cs
                                           else ""))
    if facts:
        out.append("<p>" + " · ".join(facts) + "</p>")
    rows = []
    if g.tilt_after is not None:
        b = g.tilt_before or (0.0, 0.0)
        for i, axis in enumerate("XY"):
            rows.append(f"<tr><td>Beam tilt {axis} (mrad)</td><td align=right>{_fmt(b[i])}</td>"
                        f"<td align=right>{_fmt(g.tilt_after[i])}</td>"
                        f"<td align=right>{_signed(g.tilt_after[i] - b[i])}</td></tr>")
    if g.mag_after is not None:
        mb = g.mag_before or C.Magnification(np.eye(2))  # no matrix: the nominal pixel size
        for label, fn, unit, digits in (("Magnification anisotropy", lambda m: m.anisotropy, " (%)", 3),
                                        ("Overall magnification", lambda m: m.mean, "", 5)):
            before, after = fn(mb), fn(g.mag_after)
            rows.append(f"<tr><td>{label}{unit}</td><td align=right>{_fmt(before, digits)}</td>"
                        f"<td align=right>{_fmt(after, digits)}</td>"
                        f"<td align=right>{_signed(after - before, digits)}</td></tr>")
    if rows:
        out.append("<table cellspacing=0 cellpadding=3><tr><th></th><th>Before</th><th>After</th><th>Change</th></tr>"
                   + "".join(rows) + "</table>")
    if g.mag_after is not None:
        m = g.mag_after.matrix
        out.append(f"<p>Magnification matrix [[{m[0, 0]:.5f}, {m[0, 1]:.5f}], [{m[1, 0]:.5f}, {m[1, 1]:.5f}]]</p>")
        if abs(g.mag_after.mean - 1) > MAG_WARNING:
            out.append(f"<p><b>⚠ The overall magnification differs from the nominal pixel size by "
                       f"{(g.mag_after.mean - 1) * 100:+.2f} %:</b> this changes the pixel size of the "
                       f"reconstruction (relion_ctf_refine warns about it too).</p>")
    for title, zs in (("Antisymmetric (odd) Zernike coefficients", g.odd),
                      ("Symmetric (even) Zernike coefficients", g.even)):
        if not zs:
            continue
        out.append(f"<p><b>{title}</b></p><table cellspacing=0 cellpadding=3>"
                   "<tr><th align=left>Term</th><th align=left></th><th>Before</th><th>After</th><th>Change</th></tr>")
        for z in zs:
            change = (z.after - (z.before or 0.0)) if z.after is not None else None
            out.append(f"<tr><td>{z.label}</td><td>{h(z.name)}</td><td align=right>{_fmt(z.before, 4)}</td>"
                       f"<td align=right>{_fmt(z.after, 4)}</td><td align=right>{_signed(change, 4)}</td></tr>")
        out.append("</table>")
        if zs is g.odd:
            tre = _pair(zs, 3, 3)
            if tre is not None:
                out.append(f"<p>Trefoil magnitude {_fmt(tre, 4)}</p>")
        else:
            tet = _pair(zs, 4, 4)
            if tet is not None:
                out.append(f"<p>Tetrafoil magnitude {_fmt(tet, 4)}</p>")
    return "\n".join(out)


def _pair(zs: list[C.Zernike], n: int, m: int) -> float | None:
    """|(Z(n,-m), Z(n,m))| after the fit."""
    a = next((z.after for z in zs if z.n == n and z.m == -m), None)
    b = next((z.after for z in zs if z.n == n and z.m == m), None)
    if a is None or b is None:
        return None
    return float(np.hypot(a, b))


class CtfRefineSummary(QWidget):
    def __init__(self, project_dir: Path, job: str, star: str = "", input_star: str = "",
                 options: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.star, self.input = Path(project_dir), job, star, input_star
        self.options = dict(options or {})
        self.tasks = Tasks(self)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.log_btn = QPushButton("Open logfile.pdf")
        self.log_btn.setToolTip("RELION's plots of the fits, in the system's PDF viewer")
        self.log_btn.clicked.connect(self._open_log)
        self.info = QLabel()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.log_btn)
        bar.addWidget(self.info, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.text, 1)
        self._stamp = None
        self._load()

    def _paths(self) -> tuple[Path | None, Path | None]:
        out = self.project_dir / self.star if self.star else self.project_dir / self.job / C.OUTPUT
        inp = self.project_dir / self.input if self.input else None
        return out, inp

    def _watch(self):
        out, inp = self._paths()
        return _stamp(out), _stamp(inp), _stamp(self.project_dir / self.job), _stamp(self._log())

    def _log(self) -> Path:
        return self.project_dir / self.job / "logfile.pdf"

    def refresh(self) -> None:
        if self._watch() != self._stamp:
            self._load()

    def _load(self) -> None:
        self._stamp = self._watch()
        self.log_btn.setEnabled(self._log().is_file())
        out, inp = self._paths()
        options, job_dir, project = self.options, self.project_dir / self.job, self.project_dir

        def read():
            if not out.is_file():
                return summary_html(options, [], None, "", "", C.micrographs_done(job_dir))
            groups = C.read_optics(out, inp)
            changes = C.particle_changes(out, inp)
            rel = lambda p: os.path.relpath(p, project) if p else ""  # noqa: E731
            return summary_html(options, groups, changes, rel(out), rel(inp) if changes and changes.input_star else "")

        self.tasks.run(read, self._on_html, channel="summary")

    def _on_html(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        bar = self.text.verticalScrollBar()
        pos = bar.value()
        self.text.setHtml(result)
        bar.setValue(pos)

    def _open_log(self) -> None:
        err = open_with_system(self._log())
        self.info.setText(err)


# ---------------------------------------------------------------------------- per-particle changes

HISTOGRAMS = [C.D_DEFOCUS, C.D_ASTIG, C.D_ANGLE, C.D_PHASE, "rlnCtfBfactor", "rlnCtfScalefactor"]
MAP_VALUES = [(C.D_DEFOCUS, "Defocus change"), (C.DEFOCUS, "Defocus"), (C.D_ASTIG, "Astigmatism change"),
              (C.ASTIG, "Astigmatism"), ("rlnCtfBfactor", "B-factor"), ("rlnCtfScalefactor", "Scale factor"),
              (C.D_PHASE, "Phase shift change"), ("rlnPhaseShift", "Phase shift")]


def _varies(t: Table, col: str) -> bool:
    v = t.numeric(col) if t.has(col) else None
    if v is None:
        return False
    v = v[np.isfinite(v)]
    return len(v) > 0 and float(v.max() - v.min()) > 0


def default_histograms(t: Table) -> list[str]:
    cols = [c for c in HISTOGRAMS if _varies(t, c)]
    return cols[:4] or [c for c in (C.DEFOCUS, C.ASTIG) if t.has(c)]


class MicrographMap(QWidget):
    """The particles of one micrograph at their coordinates, coloured by a value; below, that value's
    mean (± SD) on every micrograph, where a click picks the micrograph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table: Table | None = None
        self.summary: C.MicrographSummary | None = None
        self.mic = QComboBox()
        self.mic.setMinimumContentsLength(28)
        self.mic.setToolTip("Micrograph")
        self.value = QComboBox()
        self.value.setToolTip("Colour the particles by")
        self.prev_btn, self.next_btn = QPushButton("◀"), QPushButton("▶")
        for b, step in ((self.prev_btn, -1), (self.next_btn, 1)):
            b.setFixedWidth(32)
            b.clicked.connect(lambda _=False, s=step: self.step(s))
        for w in (self.mic, self.value):
            w.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.mic.currentIndexChanged.connect(lambda _: self.redraw_map())
        self.value.currentIndexChanged.connect(lambda _: self.redraw())
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.prev_btn, self.mic, self.next_btn, QLabel("Colour:"), self.value):
            bar.addWidget(w)
        bar.addStretch(1)
        self.grid = pg.GraphicsLayoutWidget()
        self.map = self.grid.addPlot(row=0, col=0)
        self.map.setAspectLocked(True)
        self.map.invertY(True)  # as the micrograph is shown
        self.map.setLabel("bottom", "X (px)")
        self.map.setLabel("left", "Y (px)")
        self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get(DIVERGING), interactive=False,
                                    width=14)
        self.grid.addItem(self.cbar, row=0, col=1)
        self.overview = self.grid.addPlot(row=1, col=0, colspan=2)
        self.overview.setLabel("bottom", "Micrograph")
        self.overview.showGrid(x=False, y=True, alpha=0.2)
        self.grid.ci.layout.setRowStretchFactor(0, 3)
        self.grid.ci.layout.setRowStretchFactor(1, 1)
        self.info = QLabel()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.info)
        self._groups: dict[str, np.ndarray] = {}
        self._level_cache: dict[str, tuple[float, float, bool]] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._project: Path | None = None
        self._pixel_size: float | None = None

    @staticmethod
    def prepare(t: Table) -> tuple:
        """What set_table works out from every particle (the values per micrograph, and each micrograph's
        particles): a second per million particles, so the view does it in the thread that read them."""
        labels = [c for c, _ in MAP_VALUES if t.has(c)]
        summary = C.per_micrograph(t, labels)
        names = t.column("rlnMicrographName") if t.has("rlnMicrographName") else []
        groups: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            groups.setdefault(n, []).append(i)
        return summary, {k: np.array(v) for k, v in groups.items()}

    def set_table(self, t: Table, project_dir: Path | None = None, prepared: tuple | None = None) -> None:
        """``prepared``: prepare(t), when the caller made it already (off the window's thread)."""
        self._project = project_dir
        keep_mic, keep_val = self.mic.currentData(), self.value.currentData()
        self.table = t
        self._level_cache.clear()
        self.summary, self._groups = prepared if prepared is not None else self.prepare(t)
        for w in (self.mic, self.value):
            w.blockSignals(True)
        self.value.clear()
        for c, label in MAP_VALUES:
            if _varies(t, c) or (c == C.DEFOCUS and t.has(c)):
                unit = C.UNITS.get(c, "")
                self.value.addItem(label + (f" ({unit})" if unit else ""), c)
        i = self.value.findData(keep_val)
        self.value.setCurrentIndex(max(0, i))
        self.mic.clear()
        if self.summary is not None:
            for k, name in enumerate(self.summary.names):
                self.mic.addItem(f"{os.path.basename(name)}  ({self.summary.particles[k]})", name)
        i = self.mic.findData(keep_mic)
        self.mic.setCurrentIndex(max(0, i))
        for w in (self.mic, self.value):
            w.blockSignals(False)
        self.redraw()

    def step(self, d: int) -> None:
        n = self.mic.count()
        if n:
            self.mic.setCurrentIndex((self.mic.currentIndex() + d) % n)

    def _levels(self, col: str) -> tuple[float, float, bool]:
        """Colour range of a value over all particles (so micrographs compare), and whether it is signed.
        Worked out once per value: a percentile over every particle, asked for at each micrograph shown."""
        if col not in self._level_cache:
            self._level_cache[col] = self._find_levels(col)
        return self._level_cache[col]

    def _find_levels(self, col: str) -> tuple[float, float, bool]:
        v = self.table.numeric(col)
        v = v[np.isfinite(v)] if v is not None else np.empty(0)
        if not len(v):
            return 0.0, 1.0, False
        signed = col.endswith("Change")
        if signed:
            m = float(np.percentile(np.abs(v), 98)) or 1.0
            return -m, m, True
        lo, hi = (float(x) for x in np.percentile(v, [2, 98]))
        return (lo, hi if hi > lo else lo + 1.0, False)

    def redraw(self) -> None:
        self.redraw_map()
        self.redraw_overview()

    def redraw_map(self) -> None:
        self.map.clear()
        t, col, name = self.table, self.value.currentData(), self.mic.currentData()
        if t is None or not col or name not in self._groups:
            self.info.setText("No micrographs in the particle file." if t is not None else "")
            return
        rows = self._groups[name]
        x = t.numeric("rlnCoordinateX")
        y = t.numeric("rlnCoordinateY")
        v = t.numeric(col)
        if x is None or y is None or v is None:
            self.info.setText("No particle coordinates in the file.")
            return
        lo, hi, signed = self._levels(col)
        cmap = pg.colormap.get(DIVERGING if signed else SEQUENTIAL)
        self.cbar.setColorMap(cmap)
        self.cbar.setLevels((lo, hi))
        vals = v[rows]
        norm = np.clip((vals - lo) / (hi - lo), 0, 1)
        colors = cmap.map(np.nan_to_num(norm, nan=0.5), mode="qcolor")
        sp = pg.ScatterPlotItem(x=x[rows], y=y[rows], size=9, pen=pg.mkPen((60, 60, 60, 160), width=0.5),
                                brush=[pg.mkBrush(c) for c in colors], data=rows)
        sp.setToolTip(None)
        self.map.addItem(sp)
        size = self._micrograph_size(name)
        xs, ys = x[rows], y[rows]
        if size and len(rows) and np.nanmax(xs) <= size[0] * 1.02 and np.nanmax(ys) <= size[1] * 1.02:
            w, h = size  # the micrograph's edges, when the coordinates are in its pixels
            self.map.addItem(pg.PlotCurveItem([0, w, w, 0, 0], [0, 0, h, h, 0],
                                              pen=pg.mkPen((150, 150, 150), style=Qt.DashLine)))
            self.map.setRange(xRange=(0, w), yRange=(0, h), padding=0.02)
        else:
            self.map.autoRange()
        finite = vals[np.isfinite(vals)]
        unit = C.UNITS.get(col, "")
        text = f"{len(rows)} particles on {os.path.basename(name)}"
        if len(finite):
            text += (f"   ·   {self.value.currentText()}: mean {finite.mean():.4g}, SD {finite.std():.3g}"
                     f"{' ' + unit if unit else ''}")
        self.info.setText(text)
        self._mark_overview()

    def _micrograph_size(self, name: str) -> tuple[int, int] | None:
        if name in self._sizes:
            return self._sizes[name]
        size = None
        if self._project is not None:
            try:
                hd = header(self._project / name)
                size = (hd.nx, hd.ny)
            except (OSError, ValueError):
                size = None
        self._sizes[name] = size
        return size

    def redraw_overview(self) -> None:
        self.overview.clear()
        s, col = self.summary, self.value.currentData()
        if s is None or not col or col not in s.mean:
            return
        m, sd = s.mean[col], s.sd[col]
        idx = np.arange(1, len(m) + 1, dtype=float)
        ok = np.isfinite(m)
        unit = C.UNITS.get(col, "")
        self.overview.setLabel("left", self.value.currentText().split(" (")[0] + (f" ({unit})" if unit else ""))
        if col.endswith("Change"):
            self.overview.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen((150, 150, 150), style=Qt.DotLine)))
        self.overview.addItem(pg.ErrorBarItem(x=idx[ok], y=m[ok], height=2 * np.nan_to_num(sd[ok]),
                                              pen=pg.mkPen((150, 150, 150))))
        pts = pg.ScatterPlotItem(x=idx[ok], y=m[ok], size=6, pen=None, brush=pg.mkBrush(0, 114, 178, 200),
                                 data=np.nonzero(ok)[0])
        pts.sigClicked.connect(self._clicked)
        self.overview.addItem(pts)
        self._marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen((213, 94, 0), width=2), brush=None)
        self.overview.addItem(self._marker)
        self._mark_overview()

    def _clicked(self, _item, points, *args) -> None:
        if len(points):
            self.mic.setCurrentIndex(int(points[0].data()))

    def _mark_overview(self) -> None:
        s, col, k = self.summary, self.value.currentData(), self.mic.currentIndex()
        marker = getattr(self, "_marker", None)
        if marker is None or s is None or col not in s.mean or k < 0:
            return
        y = s.mean[col][k]
        marker.setData([k + 1], [y] if np.isfinite(y) else [np.nan])


class CtfChangesView(QWidget):
    def __init__(self, project_dir: Path, star: str, input_star: str = "", parent=None):
        super().__init__(parent)
        self.project_dir, self.star, self.input = Path(project_dir), star, input_star
        self.tasks = Tasks(self)
        self.plot = TablePlot()
        self.map = MicrographMap()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(self.map)
        split.setSizes([550, 550])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self._stamp = None
        self._load()

    def _paths(self) -> tuple[Path, Path | None]:
        return self.project_dir / self.star, (self.project_dir / self.input if self.input else None)

    def refresh(self) -> None:
        out, inp = self._paths()
        if (_stamp(out), _stamp(inp)) != self._stamp:
            self._load()

    def _load(self) -> None:
        out, inp = self._paths()
        self._stamp = (_stamp(out), _stamp(inp))
        self.plot.info.setText("Reading…")

        def read():
            ch = C.particle_changes(out, inp)
            return ch, (MicrographMap.prepare(ch.table) if ch is not None else None)

        self.tasks.run(read, self._on_changes, channel="changes")

    def _on_changes(self, result) -> None:
        ch, prepared = (result, None) if isinstance(result, TaskError) else result
        if isinstance(ch, TaskError) or ch is None:
            self.plot.info.setText(f"⚠ {ch}" if ch is not None else "No particles in the file.")
            return
        t = ch.table
        scatter = (C.DEFOCUS, C.D_DEFOCUS) if ch.has_changes else ("rlnCoordinateX", C.DEFOCUS)
        self.plot.set_table(t, default_histograms(t), scatter)
        self.map.set_table(t, self.project_dir, prepared)


# ---------------------------------------------------------------------------- aberration images

RANGES = [("±1 rad", 1.0), ("±π rad", np.pi)]  # as RELION's plots: small phases, and the full circle


class AberrationsView(QWidget):
    """Observed phase differences, the fitted model and the residual, per optics group."""

    def __init__(self, project_dir: Path, job: str, star: str = "", parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.star = Path(project_dir), job, star
        self.tasks = Tasks(self)
        self.group = QComboBox()
        self.group.setToolTip("Optics group")
        self.group.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.group.currentIndexChanged.connect(lambda _: self._load_images())
        self.range = QComboBox()
        for label, _ in RANGES:
            self.range.addItem(label)
        self.range.setToolTip("Colour range of the phase images (the displacement images: ±1 px)")
        self.range.currentIndexChanged.connect(lambda _: self._draw())
        self.zoom = QComboBox()
        self.zoom.addItems(["Whole image", "Inner half"])
        self.zoom.setToolTip("Show all frequencies, or up to half of Nyquist")
        self.zoom.currentIndexChanged.connect(lambda _: self._draw())
        self.info = QLabel()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("Optics group:"), self.group, QLabel("Range:"), self.range, self.zoom):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.info)
        self.grid = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.grid, 1)
        self.sets: list[C.AberrationSet] = []
        self.images: list[tuple[C.AberrationRow, np.ndarray | None, np.ndarray | None]] = []
        self.pixel_sizes: dict[int, float] = {}
        self._stamp = None
        self._scan()

    def _watch(self):
        return _stamp(self.project_dir / self.job)

    def refresh(self) -> None:
        if self._watch() != self._stamp:
            self._scan()

    def _scan(self) -> None:
        self._stamp = self._watch()
        project, job = self.project_dir, self.job
        out = project / (self.star or (job + C.OUTPUT))

        def read():
            sets = C.aberration_images(project, job)
            groups = C.read_optics(out) if out.is_file() else []
            return sets, {g.number: g.pixel_size for g in groups if g.pixel_size}, {g.number: g.name for g in groups}

        self.tasks.run(read, self._on_sets, channel="scan")

    def _on_sets(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        self.sets, self.pixel_sizes, names = result
        keep = self.group.currentData()
        self.group.blockSignals(True)
        self.group.clear()
        for s in self.sets:
            self.group.addItem(f"{s.group}: {names.get(s.group, '')}".rstrip(": "), s.group)
        i = self.group.findData(keep)
        self.group.setCurrentIndex(max(0, i))
        self.group.blockSignals(False)
        if not self.sets:
            self.grid.clear()
            self.info.setText("No aberration images yet (relion_ctf_refine writes them at the end).")
            return
        self._load_images()

    def _load_images(self) -> None:
        k = self.group.currentIndex()
        if k < 0 or k >= len(self.sets):
            return
        s, project = self.sets[k], self.project_dir

        def read():
            out = []
            for row in s.rows:
                obs = read_image(ImageRef(str(project / row.observed.path))) if row.observed else None
                fit = read_image(ImageRef(str(project / row.fit.path))) if row.fit else None
                out.append((row, obs, fit))
            return out

        self.tasks.run(read, self._on_images, channel="images")

    def _on_images(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        self.images = result
        self._draw()

    def _draw(self) -> None:
        self.grid.clear()
        if not self.images:
            return
        g = self.group.currentData()
        angpix = self.pixel_sizes.get(g)
        phase_range = RANGES[max(0, self.range.currentIndex())][1]
        inner = self.zoom.currentIndex() == 1
        cmap = pg.colormap.get(DIVERGING)
        for r, (row, obs, fit) in enumerate(self.images):
            kind = (row.observed or row.fit).kind
            lim = phase_range if kind == "phase" else 1.0
            label = pg.LabelItem(f"<b>{html.escape(row.title)}</b>", justify="left")
            self.grid.addItem(label, row=2 * r, col=0, colspan=4)
            panels = [("Observed", obs), (row.fit.title if row.fit else "Fit", fit)]
            if obs is not None and fit is not None and obs.shape == fit.shape:
                panels.append(("Residual (observed − fit)", C.residual(obs, fit, kind)))
            for c, (title, img) in enumerate(panels):
                p = self.grid.addPlot(row=2 * r + 1, col=c, title=title)
                p.setAspectLocked(True)
                p.hideButtons()
                for axis in ("bottom", "left"):
                    p.getAxis(axis).enableAutoSIPrefix(False)
                if img is None:
                    continue
                item = pg.ImageItem(img, levels=(-lim, lim))
                item.setColorMap(cmap)
                ny, nx = img.shape
                if angpix:  # spatial frequency in 1/Å, the origin in the middle
                    fx, fy = 1.0 / (nx * angpix), 1.0 / (ny * angpix)
                    item.setRect(QRectF(-nx / 2 * fx, -ny / 2 * fy, nx * fx, ny * fy))
                    p.setLabel("bottom", "Spatial frequency (1/Å)")
                    half = (nx / 4 * fx, ny / 4 * fy) if inner else (nx / 2 * fx, ny / 2 * fy)
                else:
                    item.setRect(QRectF(-nx / 2, -ny / 2, nx, ny))
                    p.setLabel("bottom", "Fourier pixels")
                    half = (nx / 4, ny / 4) if inner else (nx / 2, ny / 2)
                p.addItem(item)
                p.setRange(xRange=(-half[0], half[0]), yRange=(-half[1], half[1]), padding=0)
            if obs is not None or fit is not None:
                unit = "rad" if kind == "phase" else "px"
                bar = pg.ColorBarItem(values=(-lim, lim), colorMap=cmap, interactive=False, width=12, label=unit)
                self.grid.addItem(bar, row=2 * r + 1, col=3)
        lay = self.grid.ci.layout
        for c in range(3):  # the three panels the same size, the colour bars narrow
            lay.setColumnStretchFactor(c, 1)
            lay.setColumnPreferredWidth(c, 300)
        lay.setColumnStretchFactor(3, 0)
        n = sum(1 for _, o, f in self.images if o is not None or f is not None)
        self.info.setText(f"{n} aberration{'s' if n != 1 else ''} fitted"
                          + ("" if angpix else "   ·   pixel size unknown: axes in Fourier pixels"))
