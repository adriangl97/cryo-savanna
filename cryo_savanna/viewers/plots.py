# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Plots: fixed figures (FSC, Guinier, convergence), histograms/scatter of STAR columns, angular distribution."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from cryo_savanna.results.curves import Figure
from cryo_savanna.results.tables import Table

# colour-blind-safe series colours (Okabe-Ito), then pyqtgraph's hues
SERIES = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233),
          (120, 120, 120), (240, 228, 66)]


def series_pen(i: int, dashed: bool = False):
    color = SERIES[i] if i < len(SERIES) else pg.intColor(i, hues=max(9, i + 1))
    return pg.mkPen(color, width=2, style=Qt.DashLine if dashed else Qt.SolidLine)


class ResolutionAxis(pg.AxisItem):
    """x in 1/Å, labelled in Å."""

    def tickStrings(self, values, scale, spacing):
        return [f"{1 / v:.1f}" if v > 0 else "" for v in values]


class FigureGrid(pg.GraphicsLayoutWidget):
    """Several figures in a grid (two columns when there is room)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plots: list[pg.PlotItem] = []
        self.figures: list[Figure] = []

    def set_figures(self, figures: list[Figure], columns: int = 2) -> None:
        ranges = {f.key: p.getViewBox().viewRange() for f, p in zip(self.figures, self.plots)
                  if not p.getViewBox().autoRangeEnabled()[0]}
        self.clear()
        self.plots, self.figures = [], list(figures)
        cols = 1 if len(figures) == 1 else columns
        for k, fig in enumerate(figures):
            axes = {"bottom": ResolutionAxis("bottom")} if fig.x_resolution else {}
            title = fig.title + (f"<br><span style='font-size:9pt;color:#555'>{' · '.join(fig.notes)}</span>"
                                 if fig.notes else "")
            p = self.addPlot(row=k // cols, col=k % cols, title=title, axisItems=axes)
            p.setLabel("bottom", fig.xlabel)
            p.setLabel("left", fig.ylabel)
            for axis in ("bottom", "left"):
                p.getAxis(axis).enableAutoSIPrefix(False)  # 1/Å values would otherwise read "×0.001"
            p.showGrid(x=True, y=True, alpha=0.2)
            if len(fig.curves) > 1 and len(fig.curves) <= 12:
                p.addLegend(offset=(-10, 10))
            for i, c in enumerate(fig.curves):
                kw = dict(symbol="o", symbolSize=5, symbolBrush=series_pen(i).color()) if c.symbols else {}
                p.plot(c.x, c.y, pen=series_pen(i, c.dashed), name=c.label, connect="finite", **kw)
            for y, label in fig.hlines:
                p.addItem(pg.InfiniteLine(y, angle=0, pen=pg.mkPen((150, 150, 150), style=Qt.DotLine),
                                          label=label, labelOpts={"position": 0.95, "color": (90, 90, 90)}))
            for x, label in fig.vlines:
                p.addItem(pg.InfiniteLine(x, angle=90, pen=pg.mkPen((200, 40, 40), style=Qt.DashLine),
                                          label=label, labelOpts={"position": 0.9, "color": (200, 40, 40)}))
            if fig.key in ranges:
                p.getViewBox().setRange(xRange=ranges[fig.key][0], yRange=ranges[fig.key][1], padding=0)
            self.plots.append(p)


class TablePlot(QWidget):
    """Histograms of some numeric columns, or one column against another; a clicked point emits its row."""

    row_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table: Table | None = None
        self.default_columns: list[str] = []
        self.mode = QComboBox()
        self.mode.addItems(["Histograms", "Scatter"])
        self.xcol, self.ycol = QComboBox(), QComboBox()
        self.cols = QComboBox()
        self.cols.setToolTip("Column to show (Histograms: the default set, or one column)")
        for w in (self.mode, self.xcol, self.ycol, self.cols):
            w.currentIndexChanged.connect(lambda _: self.redraw())
        self.xcol.setMinimumContentsLength(18)
        self.ycol.setMinimumContentsLength(18)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.mode)
        self.lbl_cols = QLabel("Column:")
        self.lbl_x, self.lbl_y = QLabel("X:"), QLabel("Y:")
        for w in (self.lbl_cols, self.cols, self.lbl_x, self.xcol, self.lbl_y, self.ycol):
            bar.addWidget(w)
        bar.addStretch(1)
        self.info = QLabel()
        bar.addWidget(self.info)
        self.grid = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.grid, 1)
        self.scatter: pg.ScatterPlotItem | None = None

    def set_table(self, table: Table, columns: list[str] | None = None, scatter: tuple[str, str] | None = None) -> None:
        keep = (self.mode.currentIndex(), self.cols.currentText(), self.xcol.currentText(), self.ycol.currentText())
        first = self.table is None
        self.table = table
        numeric = table.numeric_columns()
        self.default_columns = [c for c in (columns or []) if c in numeric] or numeric[:4]
        for w in (self.mode, self.xcol, self.ycol, self.cols):
            w.blockSignals(True)
        self.cols.clear()
        self.cols.addItems(["(default set)"] + numeric)
        self.xcol.clear()
        self.xcol.addItems(["(row number)"] + numeric)
        self.ycol.clear()
        self.ycol.addItems(numeric)
        if first:
            if scatter and all(c in numeric for c in scatter):
                self.xcol.setCurrentText(scatter[0])
                self.ycol.setCurrentText(scatter[1])
            elif len(numeric) > 1:
                self.xcol.setCurrentIndex(1)
                self.ycol.setCurrentIndex(1)
        else:
            self.mode.setCurrentIndex(keep[0])
            for w, t in zip((self.cols, self.xcol, self.ycol), keep[1:]):
                i = w.findText(t)
                if i >= 0:
                    w.setCurrentIndex(i)
        for w in (self.mode, self.xcol, self.ycol, self.cols):
            w.blockSignals(False)
        self.redraw()

    def redraw(self) -> None:
        self.grid.clear()
        self.scatter = None
        t = self.table
        if t is None:
            return
        hist = self.mode.currentIndex() == 0
        for w in (self.lbl_cols, self.cols):
            w.setVisible(hist)
        for w in (self.lbl_x, self.xcol, self.lbl_y, self.ycol):
            w.setVisible(not hist)
        if hist:
            cols = self.default_columns if self.cols.currentIndex() <= 0 else [self.cols.currentText()]
            ncol = 1 if len(cols) == 1 else 2
            for k, c in enumerate(cols):
                v = t.numeric(c)
                v = v[np.isfinite(v)] if v is not None else np.empty(0)
                p = self.grid.addPlot(row=k // ncol, col=k % ncol, title=c.removeprefix("rln"))
                p.setLabel("left", "Count")
                if len(v):
                    y, x = np.histogram(v, bins=min(80, max(10, int(np.sqrt(len(v))))))
                    p.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 114, 178, 120), pen=series_pen(0))
                    p.setTitle(f"{c.removeprefix('rln')}<br><span style='font-size:9pt;color:#555'>"
                               f"median {np.median(v):.4g} · mean {v.mean():.4g} · n={len(v)}</span>")
            self.info.setText(f"{len(t)} rows")
            return
        ycol = self.ycol.currentText()
        y = t.numeric(ycol) if ycol else None
        if y is None:
            return
        x = np.arange(1, len(y) + 1, dtype=float) if self.xcol.currentIndex() <= 0 else t.numeric(self.xcol.currentText())
        p = self.grid.addPlot()
        p.setLabel("bottom", self.xcol.currentText().removeprefix("rln"))
        p.setLabel("left", ycol.removeprefix("rln"))
        p.showGrid(x=True, y=True, alpha=0.2)
        ok = np.isfinite(x) & np.isfinite(y)
        rows = np.nonzero(ok)[0]
        step = max(1, len(rows) // 200_000)  # a scatter of millions of particles: every n-th one
        rows = rows[::step]
        self.scatter = pg.ScatterPlotItem(x=x[rows], y=y[rows], data=rows, size=5, pen=None,
                                          brush=pg.mkBrush(0, 114, 178, 150))
        self.scatter.sigClicked.connect(self._clicked)
        p.addItem(self.scatter)
        self.info.setText(f"{len(rows)} points" + (f" (every {step}th row)" if step > 1 else "") +
                          ("   ·   click a point to show its row" if self.receivers_connected() else ""))

    def receivers_connected(self) -> bool:
        try:
            return self.receivers(self.row_clicked) > 0
        except (TypeError, AttributeError):
            return False

    def _clicked(self, item, points, *args) -> None:
        if len(points):
            self.row_clicked.emit(int(points[0].data()))


class AnglesPlot(QWidget):
    """Particles per (rot, tilt), as RELION's angular distribution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Rot (°)")
        self.plot.setLabel("left", "Tilt (°)")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.getViewBox().invertY(False)
        self.bar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"), interactive=False, label="Particles")
        self.bar.setImageItem(self.img, insert_in=self.plot.getPlotItem())
        self.info = QLabel()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.info)
        lay.addWidget(self.plot, 1)

    def set_histogram(self, counts: np.ndarray, n: int, source: str) -> None:
        from qtpy.QtCore import QRectF
        self.img.setImage(counts, autoLevels=False)
        hi = float(counts.max()) or 1.0
        self.bar.setLevels((0, hi))
        self.img.setRect(QRectF(-180, 0, 360, 180))
        empty = (counts == 0).sum() / counts.size
        self.info.setText(f"{n} particles from {source}   ·   {100 * empty:.0f}% of the (rot, tilt) bins are empty")
