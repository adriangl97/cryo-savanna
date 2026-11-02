# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Result viewers: micrographs (+ CTF, motion, picks), class galleries, particle grids, plots, STAR tables, files.

Drawing is plain raster (pyqtgraph without OpenGL): images are downsampled to the size they are
shown at before they reach Qt, and file reads run in a thread pool, so the views stay smooth over
X11 forwarding.
"""

import pyqtgraph as pg

pg.setConfigOptions(useOpenGL=False, antialias=False, imageAxisOrder="row-major", background="w", foreground="k")
