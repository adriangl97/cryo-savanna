# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Main window: job tree and graph, job info, result viewers and logs, the job editor, and the SLURM monitor."""

from __future__ import annotations

import sys
import time
from html import escape as html_escape
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, QSettings, Qt, QThread, QThreadPool, QTimer, Signal, Slot
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QToolButton,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna import COPYRIGHT, LICENSE_NOTICE, __version__
from cryo_savanna.env.relion_env import RelionEnv, relion_version, version_warning
from cryo_savanna.images.cache import ThumbCache
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.main_inputs import MainInputs
from cryo_savanna.project.rebuild import find_job_dirs
from cryo_savanna.project.slurm import QueueState, SlurmClient, SlurmError, SlurmMonitor
from cryo_savanna.project.snapshot import ProjectLoader, ProjectSnapshot, job_type_names
from cryo_savanna.project.stars import Stars
from cryo_savanna.project.summary import JobSummary, SummaryReader
from cryo_savanna.ui import style
from cryo_savanna.ui.job_actions import JobActions
from cryo_savanna.ui.job_graph import JobGraphView
from cryo_savanna.ui.job_info import JobInfoPanel
from cryo_savanna.ui.job_tree import STATUS_FILTERS, JobTreeView
from cryo_savanna.ui.log_panel import LogPanel
from cryo_savanna.ui.rebuild_dialog import ask_generate, run_rebuild
from cryo_savanna.ui.preferences import (
    PreferencesDialog,
    TextDialog,
    apply_environment,
    apply_relion_defaults,
    setting,
    shortcuts_html,
)


def _intervals(poll_s: float, queue_poll_s: float) -> tuple[float, float]:
    """The polling intervals as used, whatever the settings say: at least 0.5 s for the project,
    at least 5 s for squeue (0 = off), as on the command line."""
    return max(0.5, poll_s), (0.0 if queue_poll_s <= 0 else max(5.0, queue_poll_s))


class _CallSignals(QObject):
    done = Signal(object, object)  # tag, result


class _Call(QRunnable):
    """fn() in a thread pool; the result (or the exception) comes back with ``tag``."""

    def __init__(self, tag, fn):
        super().__init__()
        self.tag, self.fn = tag, fn
        self.signals = _CallSignals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as e:  # noqa: BLE001 - reported in the GUI
            result = e
        self.signals.done.emit(self.tag, result)


class LoaderWorker(QObject):
    """Runs ProjectLoader.refresh off the GUI thread, so slow filesystems never freeze the window."""

    loaded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, loader: ProjectLoader):
        super().__init__()
        self.loader = loader

    @Slot(bool)
    def refresh(self, force: bool) -> None:
        try:
            snap = self.loader.refresh(force)
            if snap is not None:
                self.loaded.emit(snap)
        except Exception as e:  # keep polling; show the error in the status bar
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


class SummaryWorker(QObject):
    """Reads job summaries (counts, resolution, ...) off the GUI thread and sends them in batches."""

    ready = Signal(object, object)  # project directory, dict job name -> JobSummary

    def __init__(self, project_dir: Path):
        super().__init__()
        self.project_dir = project_dir
        self.reader = SummaryReader(project_dir)
        self.generation = 0  # set from the GUI thread; a pass stops when a newer snapshot is queued

    @Slot(object, int, bool)
    def update(self, snap: ProjectSnapshot, generation: int, force: bool) -> None:
        if force:
            self.reader.clear()
        batch: dict[str, JobSummary] = {}
        last = time.monotonic()
        # running jobs first, then newest first: what the user most likely looks at
        order = sorted(snap.jobs.values(), key=lambda i: (i.status != "Running", -(i.process.job_number or 0)))
        for info in order:
            if generation != self.generation:
                break
            if not self.reader.needs_update(info):
                continue
            batch[info.name] = self.reader.summarize(info, snap.pipeline)
            if time.monotonic() - last > 0.25:
                self.ready.emit(self.project_dir, batch)
                batch, last = {}, time.monotonic()
        if batch:
            self.ready.emit(self.project_dir, batch)


class MainWindow(QMainWindow):
    request_refresh = Signal(bool)
    request_summaries = Signal(object, int, bool)

    def __init__(self, project_dir: Path, tomo: bool | None = None, readonly: bool = False,
                 poll_s: float | None = None, defs: JobDefs | None = None, env: RelionEnv | None = None,
                 settings: QSettings | None = None, flag_source=None, queue_poll_s: float | None = None,
                 slurm: SlurmClient | None = None):
        super().__init__()
        self.settings = settings if settings is not None else QSettings("cryo-savanna", "cryo-savanna")
        apply_environment(self.settings)
        style.apply_theme(QApplication.instance(), setting(self.settings, "theme"))  # before the plots are made
        # None: from the preferences (the command line's --poll / --queue-poll win)
        poll_s = setting(self.settings, "poll_s") if poll_s is None else poll_s
        queue_poll_s = setting(self.settings, "queue_poll_s") if queue_poll_s is None else queue_poll_s
        poll_s, queue_poll_s = _intervals(poll_s, queue_poll_s)
        self.defs = defs or JobDefs.load()
        self.env = env or RelionEnv()
        apply_relion_defaults(self.settings, self.env)
        # tomo None: the tomography job menu if the project has tomography jobs (decided on loading)
        self.tomo, self.readonly, self.poll_s = bool(tomo), readonly, poll_s
        self.tomo_auto = tomo is None
        self._tomo_decided = False
        self.flag_source = flag_source
        self.queue_poll_s = queue_poll_s
        self.slurm_client = slurm or SlurmClient()
        self.slurm_monitor: SlurmMonitor | None = None
        self.queue_state = QueueState()
        self._queue_busy = False
        self._queue_pool = QThreadPool(self)
        self._queue_pool.setMaxThreadCount(1)
        self._queue_tasks: set[_Call] = set()
        self.type_names = job_type_names(self.defs)
        self.snapshot: ProjectSnapshot | None = None
        self.current: str | None = None
        self._busy = False
        self._thread: QThread | None = None
        self._worker: LoaderWorker | None = None
        self._summary_thread: QThread | None = None
        self._summary_worker: SummaryWorker | None = None
        self._retired: list[tuple[QThread, QObject | None]] = []  # stopped threads still reading (_stop_worker)
        self._generation = 0
        self._force_summaries = False
        self.summaries: dict[str, JobSummary] = {}
        self._error = ""
        self._select_when_loaded: str | None = None
        self.relion_version: str | None = None  # of the loaded RELION, once relion_pipeliner --version answered
        self.version_warning: str | None = None
        self._version_call: _Call | None = None

        self._build_ui()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self.tree.refresh_times)
        self.clock.timeout.connect(self.graph.refresh_running)
        self.clock.start(30_000)
        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self._poll_queue)
        self._restore_settings()
        self.open_project(project_dir)
        self._check_version()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.tree = JobTreeView(self.type_names)
        self.graph = JobGraphView(self.type_names)
        self.info = JobInfoPanel(self.defs, self.env, self.type_names)
        self.info.summaries = self.summaries
        self.logs = LogPanel()

        # filter row above the job views
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter jobs (name, alias, type, queue)")
        self.search.setClearButtonEnabled(True)
        self.status_filter = QComboBox()
        self.status_filter.addItems(list(STATUS_FILTERS))
        self.hide_failed = QCheckBox("Hide failed/aborted")
        self.starred_only = QCheckBox(f"{style.STAR} Starred")
        self.starred_only.setToolTip("Only starred jobs (and the jobs they come from); Job → Star marks one")
        self.search.textChanged.connect(self._apply_filter)
        self.status_filter.currentIndexChanged.connect(self._apply_filter)
        self.hide_failed.toggled.connect(self._apply_filter)
        self.starred_only.toggled.connect(self._apply_filter)
        filt = QHBoxLayout()
        filt.setContentsMargins(0, 0, 0, 0)
        filt.addWidget(self.search, 1)
        filt.addWidget(self.status_filter)
        filt.addWidget(self.starred_only)
        filt.addWidget(self.hide_failed)

        self.views = QTabWidget()
        self.views.addTab(self.tree, "List")
        self.views.addTab(self.graph, "Tree")
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 0, 4)
        lv.addLayout(filt)
        lv.addWidget(self.views)

        self.right = QSplitter(Qt.Vertical)
        self.right.addWidget(self.info)
        self.right.addWidget(self.logs)
        self.right.setSizes([400, 300])
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(left)
        self.split.addWidget(self.right)
        self.split.setSizes([700, 600])
        self.setCentralWidget(self.split)

        self.actions = JobActions(self, self.defs, self.env, self.type_names, self.tomo, self.readonly,
                                  flag_source=self.flag_source)
        self.actions.changed.connect(self._force_refresh)
        self.actions.select.connect(self._select_soon)
        self.actions.message.connect(lambda m: self.statusBar().showMessage(m, 6000))
        self.actions.star.connect(self.set_star)

        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_open = self._action(tb, "Open project…", QKeySequence.Open, self._choose_project)
        self.act_refresh = self._action(tb, "Refresh", QKeySequence.Refresh, self._force_refresh)
        tb.addSeparator()
        new_btn = QToolButton()
        new_btn.setText("New job")
        new_btn.setMenu(self.actions.new_menu)
        new_btn.setPopupMode(QToolButton.InstantPopup)
        tb.addWidget(new_btn)
        for a in (self.actions.act_clone, self.actions.act_continue, self.actions.act_run):
            tb.addAction(a)
        tb.addSeparator()
        self.act_tree = self._action(tb, "List", "Ctrl+1", lambda: self.views.setCurrentWidget(self.tree))
        self.act_graph = self._action(tb, "Tree", "Ctrl+2", lambda: self.views.setCurrentWidget(self.graph))
        self.act_fit = self._action(tb, "Fit tree", "Ctrl+0", self._fit_graph)
        self.all_links = QCheckBox("All links")
        self.all_links.setToolTip("Also draw links to references, masks and other secondary inputs of every job\n"
                                  "(otherwise only for the selected job)")
        self.all_links.toggled.connect(self.graph.set_show_all_links)
        tb.addWidget(self.all_links)
        self._action(None, "Find", QKeySequence.Find, lambda: (self.search.setFocus(), self.search.selectAll()))
        act_quit = self._action(None, "Quit", QKeySequence.Quit, self.close)

        self.act_prefs = self._action(None, "Preferences…", "Ctrl+,", self.show_preferences)  # no standard key on Linux
        self.act_editor = self.actions.act_editor
        self.act_editor.setShortcut(QKeySequence("Ctrl+E"))
        self.addAction(self.act_editor)
        self.act_popout = self._action(None, "Open viewer in its own window", "Ctrl+Shift+P", self.info.pop_out)
        self.act_shortcuts = self._action(None, "Keyboard shortcuts", QKeySequence.HelpContents, self.show_shortcuts)
        self.act_about = self._action(None, "About cryo-savanna", None, self.show_about)

        mb = self.menuBar()
        project = mb.addMenu("Project")
        for a in (self.act_open, self.act_refresh, None, self.actions.act_import, self.actions.act_undelete, None,
                  self.act_prefs, None, act_quit):
            project.addSeparator() if a is None else project.addAction(a)
        mb.addMenu(self.actions.job_menu)
        mb.addMenu(self.actions.cli_menu)
        view = mb.addMenu("View")
        for a in (self.act_tree, self.act_graph, self.act_fit):
            view.addAction(a)
        self.act_tomo = QAction("Tomography job menu", self)
        self.act_tomo.setCheckable(True)
        self.act_tomo.setChecked(self.tomo)
        self.act_tomo.setToolTip("New job offers RELION's tomography jobs (like relion --tomo); chosen "
                                 "automatically when the project has tomography jobs")
        self.act_tomo.toggled.connect(self._set_tomo)
        job_items = self.actions.job_menu.actions()  # right below "New job"
        self.actions.job_menu.insertAction(job_items[1] if len(job_items) > 1 else None, self.act_tomo)
        view.addAction(self.act_tomo)
        view.addAction(self.act_editor)
        view.addAction(self.act_popout)
        help_menu = mb.addMenu("Help")
        help_menu.addAction(self.act_shortcuts)
        help_menu.addAction(self.act_about)

        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            lambda pos: self.actions.context_menu(self.tree._name(self.tree.indexAt(pos)) if self.tree.indexAt(pos).isValid()
                                                  else None, self.tree.viewport().mapToGlobal(pos)))
        self.graph.context_requested.connect(self.actions.context_menu)

        self.status_project = QLabel()
        self.status_state = QLabel()
        self.statusBar().addWidget(self.status_project, 1)
        self.statusBar().addPermanentWidget(self.status_state)

        self.tree.job_selected.connect(self._on_job)
        self.graph.job_selected.connect(self._on_graph_job)
        self.info.job_link.connect(self.select_job)
        self.info.tool_requested.connect(self.actions.request_tool)
        self.info.main_input_requested.connect(self.set_main_input)

    def _action(self, tb: QToolBar | None, text: str, shortcut, slot) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        self.addAction(a)
        if tb is not None:
            tb.addAction(a)
        return a

    # ------------------------------------------------------------------ preferences and help
    def show_preferences(self) -> None:
        dlg = PreferencesDialog(self.settings, self, self.env)
        if dlg.exec() != PreferencesDialog.Accepted:
            return
        s = self.settings
        self.poll_s, self.queue_poll_s = _intervals(setting(s, "poll_s"), setting(s, "queue_poll_s"))
        self.poll_timer.start(int(self.poll_s * 1000))
        if self.queue_poll_s > 0 and self.slurm_client.available:
            self.queue_timer.start(int(self.queue_poll_s * 1000))
        else:
            self.queue_timer.stop()
        if self.info.cache is not None:
            self.info.cache.max_disk = setting(s, "cache_mb") * 1024 * 1024
        apply_environment(s)
        apply_relion_defaults(s, self.env)
        self.actions.editor.own_scratch = setting(s, "scratch_own")
        self.actions.editor.refresh_defaults()  # an open editor shows the new scratch directory
        style.apply_theme(QApplication.instance(), setting(s, "theme"))
        self.actions.set_editor_place(setting(s, "editor"))
        self._update_status()

    def show_shortcuts(self) -> None:
        TextDialog("Keyboard shortcuts", shortcuts_html(self), self).show()

    def show_about(self) -> None:
        import qtpy

        exe = self.env.which("relion_pipeliner")
        # never asked here: --version can take long on a busy node (the background check fills it in)
        version = self.relion_version or ("still asking" if self._version_call is not None else "?")
        rows = [("cryo-savanna", __version__), ("Job definitions", f"RELION {self.defs.version}"),
                ("relion_pipeliner", f"{exe} ({version})" if exe else "not on PATH"),
                ("Qt", f"{qtpy.API_NAME} {qtpy.QT_VERSION}"), ("Project", str(self.loader.project_dir))]
        notice = html_escape(LICENSE_NOTICE).replace("\n", "<br>")
        html = "<h3>cryo-savanna</h3><p>A GUI for RELION 5.1 projects.</p><table cellpadding=3>" + "".join(
            f"<tr><td style='color:gray'>{k}</td><td>{html_escape(v)}</td></tr>" for k, v in rows) + "</table>" + (
            f"<p>{html_escape(COPYRIGHT)}<br>{notice}</p>")
        TextDialog("About cryo-savanna", html, self).show()

    def _set_tomo(self, tomo: bool) -> None:
        self.tomo = tomo
        self.actions.set_tomo(tomo)
        self._update_title()

    def _update_title(self) -> None:
        d = self.loader.project_dir if getattr(self, "loader", None) else ""
        self.setWindowTitle(f"cryo-savanna {__version__} — {d}" + (" [tomo]" if self.tomo else "")
                            + (" [read-only]" if self.readonly else ""))

    def _fit_graph(self) -> None:
        self.views.setCurrentWidget(self.graph)
        self.graph.fit()

    # ------------------------------------------------------------------ project loading
    def open_project(self, project_dir: Path) -> None:
        # main inputs chosen and stars given here are the project's (read-only: for this session only)
        root = Path(project_dir).resolve()
        loader = ProjectLoader(project_dir, self.defs, main_inputs=MainInputs(root, persist=not self.readonly),
                               stars=Stars(root, persist=not self.readonly))
        if not loader.is_project() and not self._generate_pipeline(loader.project_dir):
            if self.snapshot is None:
                self._update_status()
            return
        self._stop_worker()
        self.loader = loader
        self.snapshot = None
        self.current = None
        self._thread = QThread(self)
        self._worker = LoaderWorker(loader)
        self._worker.moveToThread(self._thread)
        self.request_refresh.connect(self._worker.refresh)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self.actions.set_project(loader.project_dir)
        self.info.main_inputs = loader.main_inputs
        self._tomo_decided = not self.tomo_auto
        self.summaries.clear()
        self.tree.clear_summaries()
        self.graph.clear_summaries()
        self._summary_thread = QThread(self)
        self._summary_worker = SummaryWorker(loader.project_dir)
        self._summary_worker.moveToThread(self._summary_thread)
        self.request_summaries.connect(self._summary_worker.update)
        self._summary_worker.ready.connect(self._on_summaries)
        self._summary_thread.start()
        self._update_title()
        self.info.cache = ThumbCache(loader.project_dir, max_disk_mb=setting(self.settings, "cache_mb"))
        self.slurm_monitor = SlurmMonitor(loader.project_dir, self.slurm_client, persist=not self.readonly)
        self._show_queue(QueueState())
        self.info.history.clear()
        self.actions.set_slurm(self.slurm_monitor)
        self.refresh(force=True)
        self.poll_timer.start(int(self.poll_s * 1000))
        if self.queue_poll_s > 0 and self.slurm_client.available:
            self.queue_timer.start(int(self.queue_poll_s * 1000))
        else:
            self.queue_timer.stop()

    def load_now(self) -> None:
        """Load synchronously (tests and first paint)."""
        snap = self.loader.refresh(force=True)
        if snap is not None:
            self._on_loaded(snap)

    def load_summaries_now(self) -> None:
        """Read all job summaries synchronously (tests and screenshots)."""
        reader = SummaryReader(self.loader.project_dir)
        self._on_summaries(self.loader.project_dir,
                           {n: reader.summarize(i, self.snapshot.pipeline) for n, i in self.snapshot.jobs.items()})

    def _force_refresh(self) -> None:
        self._force_summaries = True
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        if self._busy or self._worker is None:
            return
        self._busy = True
        self.request_refresh.emit(force)

    def _poll(self) -> None:
        self.refresh()
        self.logs.refresh()
        self.info.refresh_viewers()

    @Slot(object)
    def _on_loaded(self, snap: ProjectSnapshot) -> None:
        if snap.project_dir != self.loader.project_dir:
            return
        self.snapshot = snap
        self._error = ""
        if not self._tomo_decided:
            self._tomo_decided = True
            if any(j.process.is_tomo_type for j in snap.jobs.values()) != self.tomo:
                self.act_tomo.setChecked(not self.tomo)  # → _set_tomo
        self.tree.set_snapshot(snap)
        self.graph.set_snapshot(snap)
        self.actions.set_snapshot(snap)
        self._apply_filter()
        if self._summary_worker is not None:
            self._generation += 1
            self._summary_worker.generation = self._generation
            self.request_summaries.emit(snap, self._generation, self._force_summaries)
            self._force_summaries = False
        if self.current not in snap.jobs:
            self.current = None
        if self._select_when_loaded in snap.jobs:
            self.select_job(self._select_when_loaded)
            self._select_when_loaded = None
        elif self.current is None and snap.jobs:
            self.select_job(list(snap.jobs)[-1])  # newest job, like RELION's GUI
        else:
            self.info.show_job(snap, self.current)
            self.logs.refresh()
        if self.queue_state.checked_at == 0 and self.queue_timer.isActive():
            self._poll_queue()  # first look at the queue as soon as the jobs are known
        self._update_status()

    # ------------------------------------------------------------------ RELION version
    def _check_version(self) -> None:
        """Ask the loaded RELION its version (off the GUI thread: --version can be slow on a busy node)."""
        exe = self.env.which("relion_pipeliner")
        if exe is None:
            return
        self._version_call = _Call(("version",), lambda: relion_version(exe))
        self._version_call.signals.done.connect(self._on_version)
        QThreadPool.globalInstance().start(self._version_call)

    @Slot(object, object)
    def _on_version(self, tag, result) -> None:
        self._version_call = None
        self.set_relion_version(None if isinstance(result, Exception) else result)

    def set_relion_version(self, version: str | None) -> None:
        """Warn once per RELION version that differs from the one cryo-savanna was made for; mark it in the status bar."""
        self.relion_version = version
        self.version_warning = version_warning(version, self.defs.version)
        self._update_status()
        if self.version_warning and self.settings.value("relion_version_warned") != version:
            self.settings.setValue("relion_version_warned", version)
            box = QMessageBox(QMessageBox.Warning, "Different RELION version", self.version_warning,
                              QMessageBox.Ok, self)
            box.setObjectName("version_warning")  # macOS shows no title on message boxes
            box.open()

    # ------------------------------------------------------------------ SLURM
    def _poll_queue(self) -> None:
        if self._queue_busy or self.slurm_monitor is None or self.snapshot is None:
            return
        self._queue_busy = True
        monitor, statuses = self.slurm_monitor, {n: j.status for n, j in self.snapshot.jobs.items()}
        self._start(_Call(("queue", monitor), lambda: monitor.poll(statuses)))

    def queue_now(self) -> QueueState:
        """Look at the queue synchronously (tests)."""
        state = self.slurm_monitor.poll({n: j.status for n, j in self.snapshot.jobs.items()})
        self._show_queue(state)
        return state

    def _request_history(self, job: str) -> None:
        """SLURM history (sacct) of a job that went to the queue, for the overview."""
        monitor = self.slurm_monitor
        if monitor is None or not self.slurm_client.available:
            return
        if not (monitor.mapper.history(job) or monitor.mapper.goes_to_queue(job)):
            return
        self._start(_Call(("history", monitor, job), lambda: monitor.history(job)))

    def history_now(self, job: str) -> None:
        self.info.set_history(job, self.slurm_monitor.history(job))

    def _start(self, call: _Call) -> None:
        call.signals.done.connect(self._on_call)
        self._queue_tasks.add(call)
        self._queue_pool.start(call)

    @Slot(object, object)
    def _on_call(self, tag, result) -> None:
        self._queue_tasks = {t for t in self._queue_tasks if t.tag is not tag}
        if tag[1] is not self.slurm_monitor:  # another project by now
            if tag[0] == "queue":
                self._queue_busy = False
            return
        if tag[0] == "queue":
            self._queue_busy = False
            if isinstance(result, Exception):
                result = QueueState(available=True, error=str(result), checked_at=time.time())
            before = style.queue_entry(self.queue_state, self.current) if self.current else None
            self._show_queue(result)
            if self.current and style.queue_entry(result, self.current) != before:
                self._request_history(self.current)  # it moved in the queue: show the new SLURM state
        elif tag[0] == "history":
            job = tag[2]
            if isinstance(result, SlurmError):
                result = f"sacct: {result}"
            elif isinstance(result, Exception):
                result = f"{type(result).__name__}: {result}"
            self.info.set_history(job, result)

    def _show_queue(self, state: QueueState) -> None:
        self.queue_state = state
        self.tree.set_queue(state)
        self.graph.set_queue(state)
        self.info.set_queue(state)
        self.actions.set_queue(state)
        self.actions.mark_lost(state.lost)  # ended in SLURM without telling RELION: failed (aborted) now
        self._update_status()

    @Slot(object, object)
    def _on_summaries(self, project_dir: Path, batch: dict[str, JobSummary]) -> None:
        if project_dir != self.loader.project_dir:  # queued by the reader of the project open before
            return
        self.summaries.update(batch)
        self.tree.set_summaries(batch)
        self.graph.set_summaries(batch)
        if self.current in batch and self.snapshot is not None:
            self.info.show_job(self.snapshot, self.current)

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:  # the loader of the project open before
            return
        self._error = msg
        self._update_status()

    @Slot()
    def _on_finished(self) -> None:
        self._busy = False

    def _stop_worker(self, wait_ms: int = 3000) -> None:
        """Stop the loader and summary threads of the project open now. A thread still reading after
        ``wait_ms`` (a big file over a slow file system) finishes on its own: it is kept, with its
        worker, until then, as destroying either while it runs aborts the program."""
        if self._summary_thread is not None:
            self._generation += 1
            if self._summary_worker is not None:
                self._summary_worker.generation = self._generation  # stop a running pass
            try:
                self.request_summaries.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._retire(self._summary_thread, self._summary_worker, wait_ms)
            self._summary_thread = None
            self._summary_worker = None
        if self._thread is not None:
            try:
                self.request_refresh.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._retire(self._thread, self._worker, wait_ms)
            self._thread = None
            self._worker = None
            self._busy = False

    def _retire(self, thread: QThread, worker: QObject | None, wait_ms: int) -> None:
        thread.quit()
        if thread.wait(wait_ms):
            return
        entry = (thread, worker)
        self._retired.append(entry)
        thread.finished.connect(lambda: self._retired.remove(entry) if entry in self._retired else None)

    def background_busy(self) -> bool:
        """Whether a background read is still going: a loader or summary thread, an squeue/sacct call,
        a --help run or a viewer's read. All of them only read."""
        from cryo_savanna.viewers import tasks
        threads = any(t.isRunning() for t in self.findChildren(QThread))
        pools = [self._queue_pool, self.actions.editor._flags_pool, self.actions._look_pool] + (
            [tasks._POOL] if tasks._POOL else [])
        return threads or any(p.activeThreadCount() for p in pools)

    def _update_status(self) -> None:
        snap = self.snapshot
        if snap is None:
            self.status_project.setText("No project")
            return
        counts: dict[str, int] = {}
        for j in snap.jobs.values():
            counts[j.status] = counts.get(j.status, 0) + 1
        parts = [f"{len(snap.jobs)} jobs"] + [f"{n} {s.lower()}" for s, n in counts.items() if s != "Succeeded"]
        self.status_project.setText(f"{snap.project_dir}   ·   " + ", ".join(parts))
        pipeliner = self.env.which("relion_pipeliner")
        state = [f"updated {time.strftime('%H:%M:%S', time.localtime(snap.loaded_at))}",
                 f"polling every {self.poll_s:g} s"] + (["read-only"] if self.readonly else [])
        if not pipeliner:
            state.append("relion_pipeliner not found")
        q = self.queue_state
        if q.available and q.checked_at:
            queued, running = q.counts()
            text = f"SLURM: {queued} queued, {running} running"
            if q.lost:
                text += f", {len(q.lost)} ended without telling RELION"
            if q.error:
                text = f"SLURM: ⚠ {q.error}"
            state.append(text)
        if self.version_warning:
            state.insert(0, f"⚠ RELION {self.relion_version} (made for {self.defs.version})")
        if self._error:
            state.insert(0, f"⚠ {self._error}")
        self.status_state.setText("   ·   ".join(state))
        tip = f"relion_pipeliner: {pipeliner or 'not on PATH'}"
        self.status_state.setToolTip(tip + (f"\n\n{self.version_warning}" if self.version_warning else ""))

    def set_main_input(self, job: str, node: str) -> None:
        """Place ``job`` under the producer of ``node`` in the list and the tree ("": the default again)."""
        try:
            self.loader.main_inputs.set(job, node or None)
        except OSError as e:
            QMessageBox.warning(self, "Main input", f"Cannot save the choice: {e}")
            return
        self.statusBar().showMessage(f"{job.rstrip('/')}: main input {node or 'back to the default'}", 6000)
        self._force_refresh()

    def set_star(self, job: str, on: bool) -> None:
        try:
            self.loader.stars.set(job, on)
        except OSError as e:
            QMessageBox.warning(self, "Star", f"Cannot save the star: {e}")
            self.actions.update_state()  # the action back as it was
            return
        self.statusBar().showMessage(f"{job.rstrip('/')}: {'starred' if on else 'star removed'}", 6000)
        self.refresh()  # a new stars stamp: a quick rebuild, no new read of the pipeline

    # ------------------------------------------------------------------ selection
    def _select_soon(self, name: str) -> None:
        """Select a job now, or as soon as it shows up in the pipeline."""
        if self.snapshot is not None and name in self.snapshot.jobs:
            self.select_job(name)
        else:
            self._select_when_loaded = name

    def select_job(self, name: str) -> None:
        if not self.tree.select_job(name):  # hidden by the filter: show it anyway
            self._on_job(name)

    @Slot(str)
    def _on_job(self, name: str) -> None:
        if self.snapshot is None or name not in self.snapshot.jobs:
            return
        self.current = name
        self.actions.set_current(name)
        self.graph.select_job(name)
        self.info.show_job(self.snapshot, name)
        self.logs.show_job(self.snapshot.project_dir / name)
        if name not in self.info.history:
            self._request_history(name)

    @Slot(str)
    def _on_graph_job(self, name: str) -> None:
        self.select_job(name)

    def _apply_filter(self) -> None:
        statuses = STATUS_FILTERS[self.status_filter.currentText()]
        self.tree.set_filter(self.search.text(), statuses, self.hide_failed.isChecked(), self.starred_only.isChecked())
        self.graph.set_dimmed(None if not self.tree.proxy.is_active() else set(self.tree.visible_jobs()))

    def _generate_pipeline(self, folder: Path) -> bool:
        """A folder without default_pipeline.star: if it has job folders, offer to generate one from them.
        True when the folder has a pipeline now."""
        jobs = find_job_dirs(folder)
        if not jobs:
            QMessageBox.warning(self, "Not a RELION project", f"{folder} has no default_pipeline.star.")
            return False
        if ask_generate(folder, len(jobs), self.readonly, self, other_folder=False) != "generate":
            return False
        return run_rebuild(folder, self.defs, new=True, type_names=self.type_names, parent=self) is not None

    def _choose_project(self) -> None:
        start = str(self.loader.project_dir) if self.snapshot else str(Path.cwd())
        d = QFileDialog.getExistingDirectory(self, "Open RELION project", start)
        if d:
            self.open_project(Path(d))

    # ------------------------------------------------------------------ settings
    def _restore_settings(self) -> None:
        s = self.settings
        if s.value("geometry"):
            self.restoreGeometry(s.value("geometry"))
        else:
            self.resize(1400, 850)
        for key, splitter in (("split", self.split), ("right", self.right)):
            if s.value(key):
                splitter.restoreState(s.value(key))
        self.views.setCurrentIndex(int(s.value("view", 0)))
        if s.value("state"):
            self.restoreState(s.value("state"))  # where the job editor dock was
        if s.value("editor_geometry"):  # and its own window
            self.actions.editor_window.placed = self.actions.editor_window.restoreGeometry(s.value("editor_geometry"))
        self.actions.set_editor_place(setting(s, "editor"))
        self.actions.editor.own_scratch = setting(s, "scratch_own")
        self.actions.hide_editor()

    def closeEvent(self, e) -> None:
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("split", self.split.saveState())
        s.setValue("right", self.right.saveState())
        s.setValue("view", self.views.currentIndex())
        s.setValue("state", self.saveState())
        if self.actions.editor_window.placed:
            s.setValue("editor_geometry", self.actions.editor_window.saveGeometry())
        s.sync()  # now: the program may end without the usual clean-up (app.run_gui)
        self.poll_timer.stop()
        self.queue_timer.stop()
        self.clock.stop()
        self.hide()  # the window goes at once; what follows can take a moment
        if not self.actions.wait(5000):  # a change to the project: always let it finish writing
            print("cryo-savanna: waiting for the last change to the project to be written…", file=sys.stderr,
                  flush=True)
            self.actions.wait(-1)
        self._queue_pool.waitForDone(2000)
        self._stop_worker(2000)  # reads still going after this are simply dropped (run_gui)
        super().closeEvent(e)
