# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna entry point.

Opens the project browser. ``--print-tree`` prints the RELION environment and the job lineage
instead, without a display (useful on a login node without X11).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryo_savanna import COPYRIGHT, LICENSE_NOTICE, __version__
from cryo_savanna.env import child
from cryo_savanna.env.relion_env import RelionEnv, relion_version, version_warning
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.lineage import primary_parents
from cryo_savanna.project.main_inputs import MainInputs
from cryo_savanna.project.pipeline import Pipeline

STATUS_MARK = {"Succeeded": " ", "Running": ">", "Scheduled": "~", "Failed": "!", "Aborted": "x"}


def print_tree(p: Pipeline, project: Path, defs: JobDefs) -> None:
    """Indented job lineage: each job under the producer of its primary input."""
    primary_parent = primary_parents(p, project, defs, main_inputs=MainInputs(project, persist=False).choices())
    kids: dict[str | None, list[str]] = {}
    for name, parent in primary_parent.items():
        kids.setdefault(parent, []).append(name)

    stack = [(root, 0) for root in reversed(kids.get(None, []))]
    while stack:  # no recursion: a chain of jobs can be longer than Python's recursion limit
        name, depth = stack.pop()
        pr = p.processes[name]
        extra = [x for x in p.parents(name) if x not in (primary_parent[name], name)]
        links = f"  (+ {', '.join(extra)})" if extra else ""
        alias = f"  [{pr.alias}]" if pr.has_alias else ""
        print(f"{STATUS_MARK.get(pr.status, '?')} {'  ' * depth}{name}{alias}{links}")
        stack.extend((k, depth + 1) for k in reversed(kids.get(name, [])))
    print("\nstatus: ' ' succeeded, '>' running, '~' scheduled, '!' failed, 'x' aborted")


def print_info(project: Path, tomo: bool) -> int:
    env = RelionEnv()
    defs = JobDefs.load()
    print(f"cryo-savanna v{__version__} (job definitions from RELION {defs.version})")
    exe = env.which("relion_pipeliner")
    version = relion_version(exe)
    print(f"relion_pipeliner: {exe or 'NOT FOUND (load RELION 5.1)'}" + (f" (RELION {version})" if version else ""))
    warning = version_warning(version, defs.version)
    if warning:
        print("WARNING: " + warning.replace("\n\n", "\n         "))
    tpl = env.get("RELION_QSUB_TEMPLATE")
    print(f"queue template:   {tpl or '(RELION_QSUB_TEMPLATE not set)'}")
    for ex in env.qsub_extras():
        print(f"queue field:      {ex.label} = {ex.default!r}")
    pipeline = Pipeline.read(project) if (project / "default_pipeline.star").exists() else None
    # as in the window: the tomography job menu if the project has tomography jobs
    tomo = tomo or (pipeline is not None and any(pr.is_tomo_type for pr in pipeline.processes.values()))
    print(f"job menu ({'tomo' if tomo else 'SPA'}): {', '.join(label for label, _ in defs.menu(tomo))}\n")
    if pipeline is None:
        print(f"{project} has no default_pipeline.star (not a RELION project)")
        from cryo_savanna.project.rebuild import find_job_dirs
        if find_job_dirs(project):
            print("It has RELION job folders: open it in the window to generate default_pipeline.star from them.")
        return 1
    print_tree(pipeline, project, defs)
    return 0


def run_gui(project: Path, tomo: bool | None, readonly: bool, poll_s: float | None,
            queue_poll_s: float | None = None) -> int:
    # Plain raster rendering: no OpenGL over X11 forwarding (for cryo-savanna only: env/child.py)
    child.override("QT_XCB_GL_INTEGRATION", "none")
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError as e:  # pip install without the [gui] extra
        print(f"cryo-savanna: the window needs Qt ({e}). Install it with environment.yml, or "
              "pip install 'cryo-savanna[gui]'; --print-tree works without it.", file=sys.stderr)
        return 1

    from cryo_savanna.ui import errors, gc_guard
    from cryo_savanna.ui.main_window import MainWindow
    from cryo_savanna.ui.start_project import choose_project

    app = QApplication.instance() or QApplication(sys.argv[:1])
    gc_guard.install(app)  # garbage collected in a reading thread can deadlock with the GUI thread
    app.setApplicationName("cryo-savanna")
    app.setStyle("Fusion")
    chosen = choose_project(project, readonly)  # not a project: start one here, open another, or close
    if chosen is None:
        return 0
    project = chosen
    reporter = errors.install(app)  # an error nothing caught: in the status bar and a log, not only the terminal
    win = MainWindow(project, tomo=tomo, readonly=readonly, poll_s=poll_s, queue_poll_s=queue_poll_s)
    reporter.reported.connect(win.show_error)
    win.show()
    rc = app.exec()
    if win.background_busy():
        # A read that is still going (summaries of big files over a slow file system, squeue, a
        # viewer) would make Qt abort ("QThread: Destroyed while thread is still running") or wait for
        # it. Every change to the project has been written (closeEvent) and the settings saved; the
        # pipeliner and the programs started from here run in their own sessions. Nothing is lost.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cryo-savanna", description="GUI for RELION 5.1 projects",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # keeps --version lines
    ap.add_argument("--project", type=Path, default=Path.cwd(), help="RELION project directory (default: current)")
    ap.add_argument("--tomo", action="store_true",
                    help="show the tomography job types (default: if the project has tomography jobs)")
    ap.add_argument("--readonly", action="store_true", help="only look: no new jobs, no changes to the project")
    ap.add_argument("--poll", type=float, default=None, metavar="S",
                    help="seconds between checks for changes made by RELION (default: Preferences, 2)")
    ap.add_argument("--queue-poll", type=float, default=None, metavar="S",
                    help="seconds between looks at the SLURM queue (squeue); 0 turns it off "
                         "(default: Preferences, 20)")
    ap.add_argument("--print-tree", action="store_true", help="print the environment and job tree, no window")
    ap.add_argument("--version", action="version", version=f"cryo-savanna v{__version__}\n{COPYRIGHT}\n{LICENSE_NOTICE}")
    args = ap.parse_args(argv)
    project = args.project.resolve()
    if args.print_tree:
        return print_info(project, args.tomo)
    queue_poll = None if args.queue_poll is None else 0.0 if args.queue_poll <= 0 else max(5.0, args.queue_poll)
    poll = None if args.poll is None else max(0.5, args.poll)
    return run_gui(project, True if args.tomo else None, args.readonly, poll, queue_poll)


if __name__ == "__main__":
    sys.exit(main())
