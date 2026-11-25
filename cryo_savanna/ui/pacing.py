# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Drags and wheel turns paced to how fast the window can be drawn, so they do not pile up over X11.

Some widgets are drawn again in full at every mouse move of a drag or every step of the wheel: a
pyqtgraph view (it pans and zooms by drawing everything again, not by scrolling), a splitter (both
sides are laid out and drawn again). Over X11 each such frame is sent whole, and a fine-grained
wheel, a touchpad or a quick drag asks for many more frames than the connection can take: the view
lags behind the mouse. ``FramePacer`` holds the events that come too soon after the last frame and
hands them over together (the latest position of a drag, the sum of the wheel's steps), so the view
ends where it would have and only the frames in between are left out. Frames are spaced by at least
as long as the last one took to draw, which over X11 includes handing it to the connection.

Clicks, presses and releases are never held: the moves held back are handed over before them.
Moves without a button pressed (hovering) pass at once.
"""

from __future__ import annotations

import time

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QMouseEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QSplitter, QWidget

MIN_FRAME_S = 0.03  # at least this long between two frames of a drag or of the wheel

_FLUSH_FIRST = (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.Leave,
                QEvent.FocusOut, QEvent.Hide)


def _copy_mouse(e: QMouseEvent) -> QMouseEvent:
    if hasattr(e, "clone"):  # Qt 6
        return e.clone()
    return QMouseEvent(e.type(), e.localPos(), e.windowPos(), e.screenPos(), e.button(), e.buttons(), e.modifiers())


def _merged_wheel(old: QWheelEvent | None, e: QWheelEvent) -> QWheelEvent:
    angle, pixels = e.angleDelta(), e.pixelDelta()
    if old is not None:
        angle, pixels = angle + old.angleDelta(), pixels + old.pixelDelta()
    return QWheelEvent(e.position(), e.globalPosition(), pixels, angle, e.buttons(), e.modifiers(), e.phase(),
                       e.inverted())


class FramePacer(QObject):
    """Paces the drags and wheel turns of one widget (see the module). It goes with the widget."""

    def __init__(self, widget: QWidget, wheel: bool = True):
        super().__init__(widget)
        self.widget = widget
        self.wheel = wheel  # also the wheel (a splitter handle has none to pace)
        self._move: QMouseEvent | None = None  # the latest move of a drag, held back
        self._turn: QWheelEvent | None = None  # the wheel's steps held back, summed
        self._next = 0.0  # time.monotonic() from which the next events may be handed over
        self._sent: float | None = None  # when events were handed over whose frame is not drawn yet
        self._sending = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.release)
        widget.installEventFilter(self)

    def eventFilter(self, obj, e) -> bool:
        if obj is not self.widget or self._sending:
            return False
        t = e.type()
        if t == QEvent.MouseMove and e.buttons() != Qt.NoButton:
            self._move = _copy_mouse(e)
            self._schedule()
            return True
        if t == QEvent.Wheel and self.wheel:
            self._turn = _merged_wheel(self._turn, e)
            self._schedule()
            return True
        if t in _FLUSH_FIRST:
            self.release()
        elif t == QEvent.Paint and self._sent is not None:
            QTimer.singleShot(0, self._drawn)  # after this paint: the frame is drawn and handed to the connection
        return False

    def _schedule(self) -> None:
        if not self._timer.isActive():  # after the events already waiting, so that they are gathered too
            self._timer.start(max(0, round((self._next - time.monotonic()) * 1000)))

    def release(self) -> None:
        """Hand over the events held back, now."""
        self._timer.stop()
        events = [e for e in (self._move, self._turn) if e is not None]
        self._move = self._turn = None
        if not events:
            return
        self._sent = time.monotonic()
        self._next = self._sent + MIN_FRAME_S  # until the frame is drawn and its time known
        self._sending = True
        try:
            for e in events:
                QApplication.sendEvent(self.widget, e)
        finally:
            self._sending = False

    def _drawn(self) -> None:
        if self._sent is None:
            return
        took = time.monotonic() - self._sent
        self._next = self._sent + max(MIN_FRAME_S, 2 * took)  # as long again idle: the connection catches up
        self._sent = None
        if self._timer.isActive():
            self._timer.start(max(0, round((self._next - time.monotonic()) * 1000)))


def pace_splitters(root: QWidget) -> None:
    """Pace the handles of the splitters in ``root`` (and ``root`` itself, if it is one); each only once."""
    splitters = root.findChildren(QSplitter)
    if isinstance(root, QSplitter):
        splitters.append(root)
    for s in splitters:
        for i in range(1, s.count()):  # handle 0 is never shown
            h = s.handle(i)
            if h is not None and not h.property("paced"):
                h.setProperty("paced", True)
                FramePacer(h, wheel=False)
