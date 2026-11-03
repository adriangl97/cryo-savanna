# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Job graph view: every job and every input/output link (layout in graph_layout.py).

Main-data-path links are solid. Other links (references, masks, ...) are dashed and drawn for
the selected job only, unless all links are switched on. Painting is plain QPainter on a raster
viewport, and jobs become plain status-coloured blocks when zoomed far out, so it stays usable
over X11 forwarding.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
)

from cryo_savanna.project.snapshot import JobInfo, ProjectSnapshot
from cryo_savanna.project.summary import JobSummary, summary_lines
from cryo_savanna.ui import style
from cryo_savanna.ui.graph_layout import ARROW, GraphLayout, layout_graph


# ---------------------------------------------------------------------------- items

PAD_L, PAD_R, PAD_V = 15.0, 10.0, 6.0  # text padding inside a card (left includes the status strip)
MIN_CARD_W = 230.0


class CardFonts:
    """Fonts and line height of the job cards, derived from the view's font."""

    def __init__(self, base: QFont):
        self.header = QFont(base)
        self.header.setBold(True)
        self.body = QFont(base)
        self.small = QFont(base)
        self.small.setPointSizeF(max(6.0, base.pointSizeF() * 0.9))
        self.fm_header, self.fm_body, self.fm_small = (QFontMetrics(f) for f in (self.header, self.body, self.small))
        self.line = max(self.fm_header.height(), self.fm_body.height()) + 1


def card_lines(info: JobInfo, type_names: dict[str, str]) -> tuple[str, str | None, int]:
    """(header, alias or None, number of summary lines) of a job's card."""
    pr = info.process
    header = style.job_type(info, type_names)
    alias = pr.alias.rstrip("/").split("/", 1)[-1] if pr.has_alias else None
    return header, alias, summary_lines(pr.type_label)


def card_size(info: JobInfo, type_names: dict[str, str], fonts: CardFonts) -> tuple[float, float]:
    """Width fits the header and the full alias; height fits every line the card can show."""
    header, alias, n_summary = card_lines(info, type_names)
    w = max(MIN_CARD_W, fonts.fm_header.horizontalAdvance(header) + PAD_L + PAD_R,
            fonts.fm_body.horizontalAdvance(alias) + PAD_L + PAD_R if alias else 0)
    lines = 2 + (alias is not None) + n_summary  # header, [alias], job number + status, summary
    return float(w), lines * fonts.line + 2 * PAD_V


class JobNodeItem(QGraphicsItem):
    """A job card: type, alias, job number and status (with progress while running), summary."""

    def __init__(self, name: str, view: "JobGraphView", size: tuple[float, float]):
        super().__init__()
        self.name = name
        self.view = view
        self.w, self.h = size
        self.info: JobInfo | None = None
        self.summary: JobSummary | None = None
        self.queue_text = ""  # place in the SLURM queue
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self._hover = False

    def set_info(self, info: JobInfo) -> None:
        self.info = info
        self._update_tooltip()
        self.update()

    def set_summary(self, summary: JobSummary | None) -> None:
        self.summary = summary
        self._update_tooltip()
        self.update()

    def _update_tooltip(self) -> None:
        pr = self.info.process
        tip = [pr.name, pr.type_label, style.status_text(self.info.status, self.info.pending_update)]
        if self.summary is not None:
            tip += self.summary.lines()
        if self.queue_text:
            tip.append(getattr(self, "_queue_details", self.queue_text))
        self.setToolTip("\n".join(tip))

    def job_line(self) -> str:
        info = self.info
        number = info.process.name.rstrip("/").split("/")[-1]
        text = f"{number} · {info.status}" + ("*" if info.pending_update else "")
        if info.status == "Running" and self.summary is not None and self.summary.progress():
            text += " · " + self.summary.progress()
        if self.queue_text:
            text += " · " + self.queue_text
        return text

    def set_queue(self, text: str, details: str) -> None:
        if text != self.queue_text:
            self.queue_text = text
            self._queue_details = details
            self._update_tooltip()
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-self.w / 2 - 2, -self.h / 2 - 2, self.w + 4, self.h + 4)

    def paint(self, p: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        pal = self.view.palette()
        lod = QStyleOptionGraphicsItem.levelOfDetailFromTransform(p.worldTransform())
        r = QRectF(-self.w / 2, -self.h / 2, self.w, self.h)
        color = style.status_color(self.info.status if self.info else "")
        selected = self.isSelected()
        if lod < 0.35:  # too small to read: a block in the status colour, like a minimap
            p.setPen(QPen(pal.color(QPalette.Highlight), 3 / lod) if selected else Qt.NoPen)
            p.setBrush(color)
            p.drawRect(r)
            return
        if selected:
            pen = QPen(pal.color(QPalette.Highlight), 3)
        elif self._hover:
            pen = QPen(pal.color(QPalette.Text), 1.5)
        else:
            pen = QPen(pal.color(QPalette.Mid), 1)
        p.setPen(pen)
        p.setBrush(pal.color(QPalette.Base))
        p.drawRoundedRect(r, 6, 6)
        strip = QRectF(r.left(), r.top(), 7, r.height())
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(strip, 3, 3)
        if self.info and self.info.pending_update:
            p.setBrush(pal.color(QPalette.Base))
            p.drawRect(QRectF(strip.left() + 2, strip.top() + 4, 3, strip.height() - 8))
        if self.info is None:
            return

        fonts = self.view.fonts
        x, width = r.left() + PAD_L, r.width() - PAD_L - PAD_R
        y = r.top() + PAD_V
        header, alias, n_summary = card_lines(self.info, self.view.type_names)
        dim = pal.color(QPalette.PlaceholderText)

        def line(text: str, font: QFont, fm: QFontMetrics, pen, elide: bool = True) -> None:
            nonlocal y
            p.setFont(font)
            p.setPen(pen)
            shown = fm.elidedText(text, Qt.ElideRight, int(width)) if elide else text
            p.drawText(QRectF(x, y, width, fonts.line), Qt.AlignLeft | Qt.AlignVCenter, shown)
            y += fonts.line

        line(header, fonts.header, fonts.fm_header, pal.color(QPalette.Text))
        if alias is not None:
            line(alias, fonts.body, fonts.fm_body, pal.color(QPalette.Text), elide=False)  # card is sized to fit it
        line(self.job_line(), fonts.small, fonts.fm_small, color if self.info.status != "Succeeded" else dim)
        for text in (self.summary.lines() if self.summary is not None else [])[:n_summary]:
            line(text, fonts.small, fonts.fm_small, dim)

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            for edge in self.view.edges_of(self.name):
                edge.set_highlight(bool(value))
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, parent: str, child: str, pts: list[tuple[float, float]], heights: tuple[float, float],
                 primary: bool, view: "JobGraphView"):
        """``pts``: centre of the parent, bend points, centre of the child. ``heights``: of both cards."""
        super().__init__()
        self.parent_job, self.child_job, self.primary, self.view = parent, child, primary, view
        self.setZValue(0)
        path, self.arrow = self._path(pts, heights)
        self.setPath(path)
        self.set_highlight(False)

    @staticmethod
    def _path(pts: list[tuple[float, float]], heights: tuple[float, float]) -> tuple[QPainterPath, QPolygonF]:
        # From the bottom of the parent to the top of the child, through the layout's bend points (if any),
        # as one smooth curve
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        start, end = QPointF(x0, y0 + heights[0] / 2), QPointF(x1, y1 - heights[1] / 2 - ARROW)
        chain = [start] + [QPointF(x, y) for x, y in pts[1:-1]] + [end]
        path = QPainterPath(start)
        for a, b in zip(chain, chain[1:]):
            dy = (b.y() - a.y()) / 2
            path.cubicTo(QPointF(a.x(), a.y() + dy), QPointF(b.x(), b.y() - dy), b)
        path.lineTo(end.x(), end.y() + 1)
        tip = QPointF(end.x(), end.y() + ARROW)
        arrow = QPolygonF([QPointF(end.x() - 4, end.y()), tip, QPointF(end.x() + 4, end.y())])
        return path, arrow

    def boundingRect(self) -> QRectF:
        return super().boundingRect().united(self.arrow.boundingRect()).adjusted(-2, -2, 2, 2)

    def shape(self) -> QPainterPath:
        return self._shape

    def set_highlight(self, on: bool) -> None:
        self.highlighted = on
        self.setVisible(self.primary or on or self.view.show_all_links)
        pal = self.view.palette()
        c = QColor(pal.color(QPalette.Highlight) if on else pal.color(QPalette.Text))
        if not on:
            c.setAlphaF(0.55 if self.primary else 0.35)
        pen = QPen(c, 2.2 if on else 1.3)
        if not self.primary:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self._color = c
        stroker = QPainterPathStroker()
        stroker.setWidth(8)
        self._shape = stroker.createStroke(self.path())
        self.setZValue(0.5 if on else 0)
        self.update()

    def paint(self, p: QPainter, option, widget=None) -> None:
        p.setPen(self.pen())
        p.setBrush(Qt.NoBrush)
        p.drawPath(self.path())
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawPolygon(self.arrow)


# ---------------------------------------------------------------------------- view

class JobGraphView(QGraphicsView):
    job_selected = Signal(str)
    context_requested = Signal(object, QPoint)  # job name (or None), global position

    def __init__(self, type_names: dict[str, str], parent=None):
        super().__init__(parent)
        self.type_names = type_names
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.nodes: dict[str, JobNodeItem] = {}
        self.edges: list[EdgeItem] = []
        self._edges_by_job: dict[str, list[EdgeItem]] = {}
        self._structure: tuple | None = None
        self._syncing = False
        self.show_all_links = False
        self.queue = None  # slurm.QueueState
        self.graph_layout: GraphLayout | None = None
        self.fonts = CardFonts(self.font())
        self.summaries: dict[str, JobSummary] = {}
        self.scene().selectionChanged.connect(self._on_selection)

    def set_snapshot(self, snap: ProjectSnapshot) -> None:
        key = snap.structure_key()
        if key == self._structure:
            for name, info in snap.jobs.items():
                self.nodes[name].set_info(info)
            return
        selected = self.selected_job()
        first = self._structure is None
        self._syncing = True
        scene = self.scene()
        scene.clear()
        self.nodes.clear()
        self.edges.clear()
        self._edges_by_job.clear()
        sizes = {name: card_size(info, self.type_names, self.fonts) for name, info in snap.jobs.items()}
        lay = self.graph_layout = layout_graph(snap, sizes)
        for (a, b), mids in lay.routes.items():
            e = EdgeItem(a, b, [lay.pos[a], *mids, lay.pos[b]], (sizes[a][1], sizes[b][1]), (a, b) in lay.primary, self)
            files = [n for n in snap.jobs[b].process.inputs
                     if n in snap.pipeline.nodes and snap.pipeline.nodes[n].producer == a]
            e.setToolTip("\n".join(f"{n}  ({snap.pipeline.nodes[n].type_label})" for n in files))
            scene.addItem(e)
            self.edges.append(e)
            self._edges_by_job.setdefault(a, []).append(e)
            self._edges_by_job.setdefault(b, []).append(e)
        for name, info in snap.jobs.items():
            item = JobNodeItem(name, self, sizes[name])
            item.set_info(info)
            item.set_summary(self.summaries.get(name))
            item.set_queue(*style.queue_entry(self.queue, name)[:2])
            item.setPos(*lay.pos[name])
            scene.addItem(item)
            self.nodes[name] = item
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-40, -40, 40, 40))
        self._structure = key
        self._syncing = False
        if selected in self.nodes:
            self.select_job(selected)
        if first:
            self.initial_view()

    def set_summaries(self, summaries: dict[str, JobSummary]) -> None:
        self.summaries.update(summaries)
        for name, s in summaries.items():
            if name in self.nodes:
                self.nodes[name].set_summary(s)

    def set_queue(self, state) -> None:
        """Add each job's place in the SLURM queue to its card (a slurm.QueueState)."""
        self.queue = state
        for name, item in self.nodes.items():
            item.set_queue(*style.queue_entry(state, name)[:2])

    def refresh_running(self) -> None:
        """Repaint running jobs, whose cards show elapsed time."""
        for item in self.nodes.values():
            if item.info is not None and item.info.status == "Running":
                item.update()

    def clear_summaries(self) -> None:
        self.summaries.clear()

    def set_show_all_links(self, on: bool) -> None:
        self.show_all_links = on
        for e in self.edges:
            e.setVisible(e.primary or on or e.highlighted)

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.PaletteChange:  # Preferences → Theme: the cards are painted from the palette,
            for edge in self.edges:  # the links keep a colour
                edge.set_highlight(edge.highlighted)

    def set_dimmed(self, visible: set[str] | None) -> None:
        """Fade jobs not in ``visible`` (the tree filter); None shows everything normally."""
        for name, item in self.nodes.items():
            item.setOpacity(1.0 if visible is None or name in visible else 0.2)
        for e in self.edges:
            on = visible is None or (e.parent_job in visible and e.child_job in visible)
            e.setOpacity(1.0 if on else 0.15)

    def edges_of(self, name: str) -> list[EdgeItem]:
        return self._edges_by_job.get(name, [])

    def selected_job(self) -> str | None:
        sel = [i for i in self.scene().selectedItems() if isinstance(i, JobNodeItem)]
        return sel[0].name if sel else None

    def select_job(self, name: str) -> None:
        item = self.nodes.get(name)
        if item is None or (item.isSelected() and len(self.scene().selectedItems()) == 1):
            return
        self._syncing = True
        self.scene().clearSelection()
        item.setSelected(True)
        self._syncing = False
        if not self.mapToScene(self.viewport().rect()).boundingRect().contains(item.sceneBoundingRect()):
            self.centerOn(item)

    def fit(self) -> None:
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            return
        self.fitInView(r.adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)
        if self.transform().m11() > 1.0:  # never blow small graphs up
            self.resetTransform()
            self.centerOn(r.center())

    def initial_view(self) -> None:
        """Whole graph if it stays readable, else readable size around the selected (or first) job."""
        self.fit()
        if self.transform().m11() < 0.6:
            self.resetTransform()
            self.scale(0.75, 0.75)
            item = self.nodes.get(self.selected_job() or "") or next(iter(self.nodes.values()), None)
            if item is not None:
                self.centerOn(item)

    def zoom(self, factor: float) -> None:
        s = self.transform().m11() * factor
        if 0.05 <= s <= 4.0:
            self.scale(factor, factor)

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.zoom(1.15 ** steps)

    def contextMenuEvent(self, e) -> None:
        item = self.itemAt(e.pos())
        while item is not None and not isinstance(item, JobNodeItem):
            item = item.parentItem()
        self.context_requested.emit(item.name if item is not None else None, e.globalPos())

    def _on_selection(self) -> None:
        if self._syncing:
            return
        name = self.selected_job()
        if name:
            self.job_selected.emit(name)
