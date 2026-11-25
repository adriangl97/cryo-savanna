# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Drags and wheel turns paced to the frames drawn (ui/pacing.py), offscreen."""

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from qtpy.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel, QSplitter, QWidget  # noqa: E402

from cryo_savanna.ui import pacing  # noqa: E402


class Recorder(QWidget):
    def __init__(self):
        super().__init__()
        self.events = []

    def mousePressEvent(self, e):
        self.events.append(("press", e.position().x() if hasattr(e, "position") else e.x()))

    def mouseReleaseEvent(self, e):
        self.events.append(("release", e.position().x() if hasattr(e, "position") else e.x()))

    def mouseMoveEvent(self, e):
        self.events.append(("move", e.position().x() if hasattr(e, "position") else e.x()))

    def wheelEvent(self, e):
        self.events.append(("wheel", e.angleDelta().y()))


def _mouse(w, kind, x, buttons=Qt.LeftButton):
    button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
    p = QPointF(x, 5)
    QApplication.sendEvent(w, QMouseEvent(kind, p, QPointF(w.mapToGlobal(QPoint(x, 5))), button, buttons,
                                          Qt.NoModifier))


def _wheel(w, dy):
    QApplication.sendEvent(w, QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(), QPoint(0, dy),
                                          Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))


def test_drag_moves_and_wheel_steps_are_gathered(qtbot):
    w = Recorder()
    qtbot.addWidget(w)
    pacer = pacing.FramePacer(w)
    _mouse(w, QEvent.MouseButtonPress, 0)
    for x in range(1, 6):  # arrive together: only the latest is handed over
        _mouse(w, QEvent.MouseMove, x)
    for _ in range(3):
        _wheel(w, 40)
    assert w.events == [("press", 0)]
    qtbot.waitUntil(lambda: len(w.events) == 3)
    assert w.events[1:] == [("move", 5), ("wheel", 120)]
    _mouse(w, QEvent.MouseMove, 9)
    _mouse(w, QEvent.MouseButtonRelease, 9)  # the move held back first, then the release at once
    assert w.events[3:] == [("move", 9), ("release", 9)]
    assert not pacer._timer.isActive()


def test_hovering_is_not_held(qtbot):
    w = Recorder()
    w.setMouseTracking(True)
    qtbot.addWidget(w)
    pacing.FramePacer(w, wheel=False)
    _mouse(w, QEvent.MouseMove, 3, buttons=Qt.NoButton)
    _wheel(w, 120)  # wheel=False: not paced
    assert w.events == [("move", 3), ("wheel", 120)]


def test_every_pyqtgraph_view_and_splitter_is_paced(qtbot):
    import pyqtgraph as pg

    import cryo_savanna.viewers  # noqa: F401 - paces pyqtgraph's views
    for view in (pg.GraphicsView(), pg.PlotWidget(), pg.GraphicsLayoutWidget()):
        qtbot.addWidget(view)
        assert any(isinstance(c, pacing.FramePacer) for c in view.viewport().children())
    split = QSplitter()
    qtbot.addWidget(split)
    for _ in range(3):
        split.addWidget(QLabel("x"))
    pacing.pace_splitters(split)
    pacing.pace_splitters(split)  # once only
    for i in (1, 2):
        assert sum(isinstance(c, pacing.FramePacer) for c in split.handle(i).children()) == 1
