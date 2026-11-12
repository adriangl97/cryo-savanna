# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Job information panel: overview with linked inputs/outputs, the job's place in the SLURM queue
and its SLURM history, the job's options (read-only), and the job's result viewers.

Result viewers (micrographs, classes, plots, files, ...) are extra tabs chosen by
``results.registry``. A tab's viewer is built the first time it is shown, kept while the same
job stays selected, and can be opened in its own window."""

from __future__ import annotations

import html
from collections.abc import Callable

from qtpy.QtCore import Qt, QThreadPool, QUrl, Signal
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QHeaderView,
    QMainWindow,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.relion_env import RelionEnv, RelionEnvError
from cryo_savanna.jobdefs.cli_catalog import POSITIONAL
from cryo_savanna.jobdefs.cli_tools import CliCatalog, cli_program_of
from cryo_savanna.jobdefs.model import JobDefs, JobVariant
from cryo_savanna.jobdefs.aretomo_jobs import AretomoCatalog, aretomo_kind_of
from cryo_savanna.jobdefs.pytom_jobs import PytomCatalog, decode, pytom_kind_of
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.snapshot import ProjectSnapshot, stamp
from cryo_savanna.results.registry import MICROGRAPHS, TABLE_PLOT, JobContext, ViewSpec, context_for, views_for
from cryo_savanna.ui import style
from cryo_savanna.ui.flag_panel import FlagsTask

FIXED_TABS = 2  # Overview, Options; the viewers follow


class JobInfoPanel(QTabWidget):
    job_link = Signal(str)  # a link to another job was clicked
    tool_requested = Signal(str, str)  # a viewer asks for a selection tool (kind, file or job)
    main_input_requested = Signal(str, str)  # make this input node the job's main input ("": the default)

    def __init__(self, defs: JobDefs, env: RelionEnv, type_names: dict[str, str], parent=None):
        super().__init__(parent)
        self.defs, self.env, self.type_names = defs, env, type_names
        self.overview = QTextBrowser()
        self.overview.setOpenLinks(False)
        self.overview.anchorClicked.connect(self._on_link)
        self.addTab(self.overview, "Overview")

        opts = QWidget()
        lay = QVBoxLayout(opts)
        lay.setContentsMargins(0, 0, 0, 0)
        self.only_changed = QCheckBox("Only options changed from their defaults")
        self.only_changed.toggled.connect(lambda _: self._fill_options())
        lay.addWidget(self.only_changed)
        self.options = QTreeWidget()
        self.options.setColumnCount(3)
        self.options.setHeaderLabels(["Option", "Value", "Default"])
        self.options.setUniformRowHeights(True)
        self.options.setAlternatingRowColors(True)
        self.options.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.options.header().resizeSection(0, 320)
        lay.addWidget(self.options)
        self.addTab(opts, "Options")

        self._snap: ProjectSnapshot | None = None
        self.summaries: dict = {}  # job name -> JobSummary, shared with the main window
        self._job: str | None = None
        self._jobstar: JobStar | None = None
        self._jobstar_error = ""
        self._jobstar_stamp = None  # job.star's (mtime, size) when read: a Continue writes it again
        # the forms of External jobs made from the "Additional external jobs" menu, as in the editor
        # (the main window gives the ones of its JobActions, so a --help is read once)
        self.cli: CliCatalog | None = None
        self.pytom = PytomCatalog(defs, env)
        self.aretomo = AretomoCatalog(defs, env)
        self._flags_pool = QThreadPool(self)
        self._flag_tasks: set[FlagsTask] = set()
        self.queue = None  # slurm.QueueState
        self.history: dict[str, list | str] = {}  # job -> its SLURM jobs (slurm.AcctJob), or an error
        self.main_inputs = None  # project.main_inputs.MainInputs: the main inputs chosen in cryo-savanna
        self.overview.setHtml("<p>Select a job.</p>")

        self.cache = None  # images.cache.ThumbCache of the project, set by the main window
        self._ctx: JobContext | None = None
        self._specs: list[ViewSpec] = []
        self._job_dir_stamp = None  # of a running job: its viewers are looked for again when it changes
        self.windows: list[ViewerWindow] = []
        self.popout = QToolButton()
        self.popout.setText("⧉")
        self.popout.setToolTip("Open this viewer in its own window")
        self.popout.setAutoRaise(True)
        self.popout.clicked.connect(self.pop_out)
        self.setCornerWidget(self.popout, Qt.TopRightCorner)
        self.popout.setEnabled(False)
        self.currentChanged.connect(self._on_tab)

    def set_queue(self, state) -> None:
        old = style.queue_entry(self.queue, self._job) if self._job else None
        self.queue = state
        if self._job and self._snap is not None and style.queue_entry(state, self._job) != old:
            self._refresh_overview()

    def set_history(self, job: str, rows) -> None:
        self.history[job] = rows
        if job == self._job and self._snap is not None:
            self._refresh_overview()

    def _refresh_overview(self) -> None:
        scroll = self.overview.verticalScrollBar().value()
        self.overview.setHtml(self._overview_html())
        self.overview.verticalScrollBar().setValue(scroll)

    def keep_job(self, snap: ProjectSnapshot) -> None:
        """A new snapshot in which the job shown is as it was: nothing is read or drawn again."""
        self._snap = snap

    def show_job(self, snap: ProjectSnapshot, name: str | None) -> None:
        reload_options = name != self._job or snap.jobs.get(name) is None
        self._snap, self._job = snap, name
        if name is None or name not in snap.jobs:
            self.overview.setHtml("<p>Select a job.</p>")
            self.options.clear()
            self._set_viewers(None)
            return
        path = snap.project_dir / name / "job.star"
        st = stamp(path)  # the same job with a new job.star (continued, overwritten): read it again
        if reload_options or self._jobstar is None or st != self._jobstar_stamp:
            self._jobstar_stamp = st
            try:
                self._jobstar, self._jobstar_error = JobStar.read(path), ""
            except (OSError, ValueError, KeyError) as e:
                self._jobstar, self._jobstar_error = None, str(e)
            self._fill_options()
        self._refresh_overview()
        info = snap.jobs[name]
        self._set_viewers(context_for(snap.project_dir, info, snap.pipeline,
                                      self._jobstar.options if self._jobstar else {}))

    # ------------------------------------------------------------------ result viewers
    def _set_viewers(self, ctx: JobContext | None) -> None:
        """Tabs for the job's viewers; kept as they are while the job and its viewers stay the same."""
        specs = views_for(ctx) if ctx is not None else []
        same_job = ctx is not None and self._ctx is not None and ctx.name == self._ctx.name \
            and ctx.project_dir == self._ctx.project_dir
        self._ctx = ctx
        if same_job and [(v.key, v.args) for v in specs] == [(v.key, v.args) for v in self._specs]:
            return
        current = self.tabText(self.currentIndex()) if self.currentIndex() >= FIXED_TABS else None
        self.blockSignals(True)
        while self.count() > FIXED_TABS:
            w = self.widget(FIXED_TABS)
            self.removeTab(FIXED_TABS)
            w.deleteLater()
        self._specs = specs
        for spec in specs:
            holder = QWidget()
            holder.setLayout(QVBoxLayout())
            holder.layout().setContentsMargins(2, 2, 2, 2)
            holder.view = None
            self.addTab(holder, spec.title)
        titles = [self.tabText(i) for i in range(self.count())]
        if current in titles:  # the same viewer for the next job of a kind (Micrographs → Micrographs)
            self.setCurrentIndex(titles.index(current))
        elif current is not None:
            self.setCurrentIndex(0)
        self.blockSignals(False)
        self._on_tab(self.currentIndex())

    def _on_tab(self, index: int) -> None:
        self.popout.setEnabled(index >= FIXED_TABS)
        if index < FIXED_TABS or self._ctx is None:
            return
        holder = self.widget(index)
        if getattr(holder, "view", None) is None:
            from cryo_savanna.viewers.views import build_view
            spec = self._specs[index - FIXED_TABS]
            holder.view = build_view(spec, self._ctx, self.cache, request_tool=self.tool_requested.emit)
            holder.layout().addWidget(holder.view)
            if spec.kind == TABLE_PLOT and hasattr(holder.view, "plot"):
                star = spec.args.get("star")
                holder.view.plot.row_clicked.connect(lambda row, s=star: self._show_micrograph(s, row))

    def _show_micrograph(self, star: str, row: int) -> None:
        """A point clicked in a plot of the micrographs: show that micrograph."""
        for spec in self._specs:
            if spec.kind == MICROGRAPHS and spec.args.get("star") == star and not spec.args.get("picks"):
                view = self.show_view(spec.title)
                if view is not None:
                    view.select_when_loaded(row)
                return

    def current_view(self) -> QWidget | None:
        i = self.currentIndex()
        return getattr(self.widget(i), "view", None) if i >= FIXED_TABS else None

    def view_titles(self) -> list[str]:
        return [self.tabText(i) for i in range(FIXED_TABS, self.count())]

    def show_view(self, title: str) -> QWidget | None:
        """Switch to a viewer tab by title (and build it); its widget, or None."""
        for i in range(FIXED_TABS, self.count()):
            if self.tabText(i) == title:
                self.setCurrentIndex(i)
                self._on_tab(i)
                return self.widget(i).view
        return None

    def refresh_viewers(self) -> None:
        """Called on every poll: the visible viewer and the open windows re-read changed files."""
        if self._ctx is not None and self._ctx.status == "Running":
            # outputs of a running job appear one by one; looked for only when files were added
            st = stamp(self._ctx.job_dir)
            if st != self._job_dir_stamp:
                self._job_dir_stamp = st
                self._set_viewers(self._ctx)
        v = self.current_view()
        if v is not None and hasattr(v, "refresh"):
            v.refresh()
        for w in list(self.windows):
            w.refresh()

    def pop_out(self) -> None:
        i = self.currentIndex()
        if i < FIXED_TABS or self._ctx is None:
            return
        spec = self._specs[i - FIXED_TABS]
        label = style.job_label(self._ctx.name, self._snap.jobs[self._ctx.name].process.alias) \
            if self._snap is not None and self._ctx.name in self._snap.jobs else self._ctx.name
        w = ViewerWindow(spec, self._ctx, self.cache, f"{spec.title} — {label}", self.window(),
                         request_tool=self.tool_requested.emit)
        w.show()
        self.windows.append(w)
        w.destroyed.connect(lambda *_, ww=w: self.windows.remove(ww) if ww in self.windows else None)

    # ------------------------------------------------------------------ overview
    def _link(self, job: str) -> str:
        info = self._snap.jobs.get(job) if self._snap else None
        label = style.job_label(job, info.process.alias) if info else job
        return f'<a href="job:{html.escape(job)}">{html.escape(label)}</a>'

    def _overview_html(self) -> str:
        snap, name = self._snap, self._job
        info = snap.jobs[name]
        pr = info.process
        pipe = snap.pipeline
        color = style.status_color(info.status).name()
        rows = [
            ("Type", f"{html.escape(style.job_type(info, self.type_names))} "
                     f"<span style='color:gray'>({html.escape(pr.type_label)})</span>"),
            ("Status", f"<b style='color:{color}'>{html.escape(info.status)}</b>"
                       + (" &nbsp;<i>finished on disk; the pipeline is updated by the next RELION GUI or "
                          "pipeliner that checks it</i>" if info.pending_update else "")),
            ("Directory", html.escape(str(snap.project_dir / name))),
        ]
        q_text, q_details, q_kind = style.queue_entry(self.queue, name)
        if q_text:
            rows.insert(2, ("Queue", f"<b style='color:{style.QUEUE_COLORS[q_kind].name()}'>{html.escape(q_text)}"
                                     f"</b><br><span style='color:gray'>{html.escape(q_details)}</span>"))
        if pr.has_alias:
            rows.insert(0, ("Alias", html.escape(pr.alias)))
        summary = self.summaries.get(name)
        if summary is not None:
            for label, items in (("Result", summary.result), ("Settings", summary.settings),
                                 ("Quality", summary.quality)):
                if items:
                    rows.append((label, html.escape(" · ".join(items))))
            if info.status == "Running" and summary.progress():
                rows.append(("Progress", html.escape(summary.progress())))
        if info.modified:
            rows.append(("Last change", f"{style.timestamp(info.modified)} ({style.ago(info.modified)})"))
        if self._jobstar is not None:
            flags = [x for x, on in (("tomo", self._jobstar.is_tomo), ("continue", self._jobstar.is_continue)) if on]
            if flags:
                rows.append(("job.star", ", ".join(flags)))
        elif self._jobstar_error:
            rows.append(("job.star", f"<span style='color:#d43c3c'>{html.escape(self._jobstar_error)}</span>"))

        out = [f"<h3 style='margin-bottom:4px'>{html.escape(style.job_label(pr.name, pr.alias))}</h3>",
               "<table cellspacing=0 cellpadding=2>"]
        out += [f"<tr><td style='color:gray;padding-right:12px'>{k}</td><td>{v}</td></tr>" for k, v in rows]
        out.append("</table>")

        def node_table(title: str, nodes: list[str], linked: Callable[[str], str]) -> None:
            out.append(f"<h4 style='margin-bottom:2px'>{title} ({len(nodes)})</h4>")
            if not nodes:
                out.append("<p style='color:gray'>none</p>")
                return
            out.append("<table cellspacing=0 cellpadding=2>")
            for n in nodes:
                node = pipe.nodes.get(n)
                kind = html.escape(node.type_label if node else "?")
                exists = (snap.project_dir / n).exists()
                missing = "" if exists else " <span style='color:#d9861c'>(missing)</span>"
                out.append(f"<tr><td>{html.escape(n)}{missing}<br><span style='color:gray'>{kind}</span></td>"
                           f"<td style='padding-left:12px'>{linked(n)}</td></tr>")
            out.append("</table>")

        chosen = (self.main_inputs.choices() if self.main_inputs is not None else {}).get(name)
        chosen = chosen if chosen in pr.inputs else None
        main = chosen or next((n for n in pr.inputs if pipe.nodes.get(n) is not None
                               and pipe.nodes[n].producer == info.primary_parent), None)

        def producer(n: str) -> str:
            node = pipe.nodes.get(n)
            if node is None or node.producer is None:
                return "<span style='color:gray'>not made by a job</span>"
            text = "from " + self._link(node.producer)
            if n == main and node.producer == info.primary_parent:
                text += " <b>(main input)</b>"
                if chosen:
                    text += (" <span style='color:gray'>chosen here ·</span> "
                             "<a href='main:' title='Place the job by its optimisation set, else its particles'>"
                             "use the default</a>")
            elif len(pr.inputs) > 1:
                text += (f" · <a href=\"main:{html.escape(n)}\" title='Place this job under that job in the "
                         "list and the tree'>make main input</a>")
            return text

        def consumers(n: str) -> str:
            node = pipe.nodes.get(n)
            if node is None or not node.consumers:
                return ""
            return "used by " + ", ".join(self._link(c) for c in node.consumers)

        node_table("Inputs", pr.inputs, producer)
        node_table("Outputs", pr.outputs, consumers)
        if info.children:
            out.append("<h4 style='margin-bottom:2px'>Continued by</h4><p>"
                       + ", ".join(self._link(c) for c in info.children) + "</p>")
        out += self._history_html(name)
        return "\n".join(out)

    def _history_html(self, name: str) -> list[str]:
        rows = self.history.get(name)
        if rows is None or rows == []:
            return []
        if isinstance(rows, str):
            return [f"<h4 style='margin-bottom:2px'>SLURM</h4><p style='color:gray'>{html.escape(rows)}</p>"]
        out = [f"<h4 style='margin-bottom:2px'>SLURM jobs ({len(rows)})</h4>",
               "<table cellspacing=0 cellpadding=2><tr style='color:gray'><td>Job</td><td>State</td><td>Start</td>"
               "<td>Elapsed</td><td>Nodes</td><td>CPUs</td><td>Max. memory</td><td>Exit</td></tr>"]
        for a in rows:
            color = style.QUEUE_COLORS["lost"].name() if a.ended_badly else (
                style.QUEUE_COLORS["running"].name() if a.active else "")
            state = f"<span style='color:{color}'>{html.escape(a.state)}</span>" if color else html.escape(a.state)
            cells = [a.id, None, a.start.replace("T", " "), a.elapsed, a.nodes, a.cpus, a.max_rss, a.exit_code]
            out.append("<tr>" + "".join(f"<td style='padding-right:10px'>{state if c is None else html.escape(c)}</td>"
                                        for c in cells) + "</tr>")
        out.append("</table>")
        return out

    def _on_link(self, url: QUrl) -> None:
        s = url.toString()
        if s.startswith("job:"):
            self.job_link.emit(s[4:])
        elif s.startswith("main:") and self._job:
            self.main_input_requested.emit(self._job, s[5:])

    # ------------------------------------------------------------------ options
    def _variant_for(self, js: JobStar, kind: str | None, program: str | None) -> JobVariant:
        """The form the editor uses for this job: pytom's, AreTomo3's or the CLI tool's for External jobs
        made from those menus, else the job type's."""
        if kind is not None:
            return self.pytom.variant(kind)
        aretomo = aretomo_kind_of(js)
        if aretomo is not None:
            return self.aretomo.variant(aretomo)
        if program is not None and self.cli is not None:
            v = self.cli.variant(program, js.is_tomo)
            if v is not None:
                return v
        return self.defs.variant_for_jobstar(js)

    def _read_flags_first(self, program: str) -> bool:
        """Start reading a CLI tool's --help off the GUI thread if its form needs it (the options are
        shown when it is done). True while that is going on."""
        if self.cli is None or program in POSITIONAL or not self.cli.flag_source.needs_run(program):
            return False
        if not any(t.program == program for t in self._flag_tasks):
            task = FlagsTask(self.cli.flag_source, program)
            task.signals.done.connect(self._on_flags_ready)
            self._flag_tasks.add(task)
            self._flags_pool.start(task)
        return True

    def _on_flags_ready(self, program: str, _flags) -> None:
        self._flag_tasks = {t for t in self._flag_tasks if t.program != program}
        if self._jobstar is not None and cli_program_of(self._jobstar) == program:
            self._fill_options()

    def wait_flags(self) -> None:
        """Wait for --help runs in progress (tests)."""
        self._flags_pool.waitForDone()
        QApplication.processEvents()

    def _fill_options(self) -> None:
        self.options.clear()
        js = self._jobstar
        if js is None:
            return
        kind = pytom_kind_of(js)
        program = cli_program_of(js) if kind is None and aretomo_kind_of(js) is None else None
        if program is not None and self._read_flags_first(program):
            QTreeWidgetItem(self.options, [f"Reading the options of {program} (its --help)…"])
            return
        try:  # a pytom job or a CLI tool: its own form, as in the editor, not External's fields
            variant = self._variant_for(js, kind, program)
        except KeyError as e:
            QTreeWidgetItem(self.options, [f"Unknown job type: {e}"])
            return
        except RelionEnvError as e:
            QTreeWidgetItem(self.options, [f"⚠ {e}"])
            return
        own_form = getattr(variant, "is_cli", False)  # its values are read from External's arguments
        try:
            values, ignored = variant.values_from_jobstar(js, self.env)
            opts = variant.resolved_options(self.env)
        except RelionEnvError as e:  # a RELION_* value RELION itself would refuse: say so, show the rest
            QTreeWidgetItem(self.options, [f"⚠ {e}"])
            return
        only_changed = self.only_changed.isChecked()
        bold = QFont(self.options.font())
        bold.setBold(True)
        dim = self.palette().color(QPalette.PlaceholderText)
        for tab in variant.resolved_tabs(self.env):
            tab_item = QTreeWidgetItem([tab.label])
            for row in tab.rows:
                active = variant.group_active(row.group, values)
                for key in row.keys:
                    o = opts.get(key)
                    if o is None:
                        continue
                    value = values.get(key, "")
                    changed = (own_form or key in js.options) and value != o.default
                    if only_changed and not changed:
                        continue
                    it = QTreeWidgetItem(tab_item, [o.label.strip().rstrip(":").strip() or key, value, o.default])
                    tip = f"{key}\n\n{o.help}" if o.help else key
                    for c in range(3):
                        it.setToolTip(c, tip)
                    if changed:
                        it.setFont(1, bold)
                    if not active:
                        for c in range(3):
                            it.setForeground(c, dim)
            if tab_item.childCount():
                self.options.addTopLevelItem(tab_item)
        if ignored:
            if kind:  # --key value pairs in "Additional arguments" that this form does not have
                given = decode(js.options.get("other_args", ""))[0]
                extra = QTreeWidgetItem(["Not options of this form"])
            else:
                given = js.options
                extra = QTreeWidgetItem([f"Not defined in RELION {self.defs.version}"])
            for k in ignored:
                QTreeWidgetItem(extra, [k, given.get(k, ""), ""])
            self.options.addTopLevelItem(extra)
        self.options.expandAll()
        self.options.resizeColumnToContents(1)


class ViewerWindow(QMainWindow):
    """A viewer in its own window (another screen, or bigger than the panel)."""

    def __init__(self, spec: ViewSpec, ctx: JobContext, cache, title: str, parent=None, request_tool=None):
        super().__init__(parent, Qt.Window)
        from cryo_savanna.viewers.views import build_view
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.view = build_view(spec, ctx, cache, request_tool=request_tool)
        self.setCentralWidget(self.view)
        self.resize(1100, 750)

    def refresh(self) -> None:
        if hasattr(self.view, "refresh"):
            self.view.refresh()
