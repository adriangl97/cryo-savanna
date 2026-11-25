# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Result viewers: micrographs (+ CTF, motion, picks), class galleries, particle grids, plots, STAR tables, files.

Drawing is plain raster (pyqtgraph without OpenGL): images are downsampled to the size they are
shown at before they reach Qt, and file reads run in a thread pool, so the views stay smooth over
X11 forwarding. pyqtgraph pans and zooms by drawing the whole view again, so every pyqtgraph view's
drags and wheel turns are paced (ui/pacing.py): a quick pan sends a few frames, not one per mouse move.
"""

import pyqtgraph as pg

from cryo_savanna.ui.pacing import FramePacer

pg.setConfigOptions(useOpenGL=False, antialias=False, imageAxisOrder="row-major", background="w", foreground="k")

_graphics_view_init = pg.GraphicsView.__init__


def _paced_init(self, *args, **kwargs) -> None:  # every pyqtgraph view (PlotWidget, GraphicsLayoutWidget, ...)
    _graphics_view_init(self, *args, **kwargs)
    FramePacer(self.viewport())  # after __init__: it sets the viewport


if getattr(pg.GraphicsView.__init__, "__name__", "") != "_paced_init":  # once, also if this module is reloaded
    pg.GraphicsView.__init__ = _paced_init


def apply_plot_theme(pal) -> None:
    """Plots in the colours of the palette (Preferences → Theme): the ones made from now on and the
    open ones. Image views keep their black background."""
    from qtpy.QtGui import QPalette
    from qtpy.QtWidgets import QApplication

    pg.setConfigOptions(background=pal.color(QPalette.Base), foreground=pal.color(QPalette.Text))
    for w in QApplication.allWidgets():
        if not isinstance(w, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
            continue
        w.setBackground("default")
        for item in w.scene().items():
            if isinstance(item, pg.AxisItem):
                item.setPen()
                item.setTextPen()
            elif isinstance(item, pg.LabelItem) and item.opts.get("color") is None:
                item.setText(item.text)
