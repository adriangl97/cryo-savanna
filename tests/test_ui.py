# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Project browser windows, offscreen (QT_QPA_PLATFORM=offscreen is set in conftest.py)."""

import dataclasses
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings, Qt  # noqa: E402
from qtpy.QtGui import QKeySequence  # noqa: E402
from qtpy.QtWidgets import QHeaderView, QToolBar  # noqa: E402

from cryo_savanna.project.lock import is_locked  # noqa: E402
from cryo_savanna.project.snapshot import ProjectLoader  # noqa: E402
from cryo_savanna.ui.graph_layout import NODE_H, NODE_W, layout_graph  # noqa: E402
from cryo_savanna.ui.job_graph import PAD_L, PAD_R, card_lines  # noqa: E402
from cryo_savanna.ui.job_tree import COL_SUMMARY  # noqa: E402
from cryo_savanna.ui.log_panel import LogView, collapse_progress  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402
from tests.fixtures.make_fixtures import add_job, make_large_project  # noqa: E402
from tests.test_snapshot import NEW_JOB, tree_files  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    def make(project, poll_s=1000.0):
        w = MainWindow(project, poll_s=poll_s, settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
        qtbot.addWidget(w)
        w.load_now()
        return w
    return make


def test_window_shows_all_jobs(window, spa_project):
    w = window(spa_project)
    assert len(w.tree.visible_jobs()) == 18
    assert len(w.graph.nodes) == 18
    assert w.current == "Class2D/job018/"  # newest job selected, like RELION's GUI
    assert "Class2D/job018" in w.info.overview.toPlainText()


def test_selection_is_shared_by_tree_graph_and_panels(window, spa_project):
    w = window(spa_project)
    w.graph.select_job("PostProcess/job013/")
    w.graph._on_selection()
    assert w.tree.current_job() == "PostProcess/job013/"
    assert w.current == "PostProcess/job013/"
    text = w.info.overview.toPlainText()
    assert "Refine3D/job011/run_half1_class001_unfil.mrc" in text and "(main input)" in text
    assert w.logs.views["run.out"].toPlainText().startswith("Running PostProcess/job013/")
    # the mask link is only drawn because PostProcess is selected
    mask_edge = next(e for e in w.graph.edges if (e.parent_job, e.child_job) == ("MaskCreate/job012/", "PostProcess/job013/"))
    assert mask_edge.isVisible()
    w.select_job("Import/job001/")
    assert not mask_edge.isVisible()


def test_info_links_select_jobs(window, spa_project):
    w = window(spa_project)
    w.select_job("PostProcess/job013/")
    w.info.job_link.emit("MaskCreate/job012/")
    assert w.current == "MaskCreate/job012/"
    assert w.graph.selected_job() == "MaskCreate/job012/"


def test_filter_keeps_ancestors_and_dims_graph(window, spa_project):
    w = window(spa_project)
    w.search.setText("postprocess")
    visible = w.tree.visible_jobs()
    assert "PostProcess/job013/" in visible and "Import/job001/" in visible
    assert "Class2D/job018/" not in visible
    assert w.graph.nodes["Class2D/job018/"].opacity() < 1
    w.search.clear()
    w.hide_failed.setChecked(True)
    visible = w.tree.visible_jobs()
    assert "LocalRes/job017/" not in visible and "Class2D/job018/" not in visible
    assert len(visible) == 16


def test_status_changes_reach_the_list_with_and_without_a_filter(window, spa_copy):
    from cryo_savanna.ui.job_tree import COL_STATUS, SORT_ROLE

    w = window(spa_copy)
    job = "Polish/job015/"  # Running; its only child is Scheduled
    row = w.tree._items[job]
    assert row[COL_STATUS].data(SORT_ROLE) == "Running"
    unchanged = w.tree._items["Import/job001/"]
    (spa_copy / job / "RELION_JOB_EXIT_FAILURE").touch()  # finished on disk: the same structure, a new status
    w.load_now()
    assert w.tree._items[job] is row and row[COL_STATUS].data(SORT_ROLE) == "Failed"  # filled again in place
    assert w.tree._items["Import/job001/"] is unchanged
    w.status_filter.setCurrentText("Failed or aborted")
    assert job in w.tree.visible_jobs()
    (spa_copy / job / "RELION_JOB_EXIT_FAILURE").unlink()
    (spa_copy / job / "RELION_JOB_EXIT_SUCCESS").touch()
    w.load_now()  # with a filter on, a job whose status changed is filtered again
    assert row[COL_STATUS].data(SORT_ROLE) == "Succeeded" and job not in w.tree.visible_jobs()


def test_star_marks_and_filters_jobs(window, spa_copy):
    from cryo_savanna.project.stars import Stars
    from cryo_savanna.ui import style
    from cryo_savanna.ui.job_tree import COL_JOB

    w = window(spa_copy)
    w.select_job("PostProcess/job013/")
    assert w.actions.act_star.isEnabled() and not w.actions.act_star.isChecked()
    w.actions.act_star.trigger()
    w.load_now()
    assert Stars(spa_copy).starred() == {"PostProcess/job013/"}
    assert w.actions.act_star.isChecked()
    row = w.tree._items["PostProcess/job013/"]
    assert row[COL_JOB].text().startswith(style.STAR)
    assert not w.tree._items["MaskCreate/job012/"][COL_JOB].text().startswith(style.STAR)
    assert w.graph.nodes["PostProcess/job013/"].info.starred

    w.starred_only.setChecked(True)  # the starred job and the jobs it comes from
    visible = w.tree.visible_jobs()
    assert "PostProcess/job013/" in visible and "Import/job001/" in visible
    assert "MaskCreate/job012/" not in visible and "Class2D/job018/" not in visible
    assert w.graph.nodes["Class2D/job018/"].opacity() < 1
    w.search.setText(style.STAR)  # the star is in the text, so the search finds it too
    w.starred_only.setChecked(False)
    assert "PostProcess/job013/" in w.tree.visible_jobs()
    assert "MaskCreate/job012/" not in w.tree.visible_jobs()
    w.search.clear()

    w.actions.act_star.trigger()  # and off again
    w.load_now()
    assert Stars(spa_copy).starred() == set()
    assert not row[COL_JOB].text().startswith(style.STAR)


def test_filter_finds_the_program_of_external_jobs(window, spa_copy):
    from cryo_savanna.jobdefs.cli_tools import runner_command
    from tests.fixtures.make_fixtures import JobSpec
    add_job(spa_copy, JobSpec(30, "relion.external", [("flipped.mrc", "DensityMap.mrc")],
                              inputs=["Refine3D/job011/run_class001.mrc"],
                              options={"fn_exe": runner_command("relion_image_handler"),
                                       "in_3dref": "Refine3D/job011/run_class001.mrc",
                                       "other_args": "--i Refine3D/job011/run_class001.mrc --invert_hand"}))
    add_job(spa_copy, JobSpec(31, "relion.external", [], options={"fn_exe": "/home/me/scripts/my_picker.sh"}))
    w = window(spa_copy)
    assert w.snapshot.jobs["External/job030/"].program == "relion_image_handler"
    assert card_lines(w.snapshot.jobs["External/job030/"], w.type_names)[0] == "External: relion_image_handler"
    w.search.setText("image_handler")
    visible = w.tree.visible_jobs()
    assert "External/job030/" in visible and "External/job031/" not in visible
    assert w.graph.nodes["External/job031/"].opacity() < 1 and w.graph.nodes["External/job030/"].opacity() == 1
    w.search.setText("my_picker")
    assert "External/job031/" in w.tree.visible_jobs()

    # the Options tab shows the tool's form (once its --help is read), not External's fields
    w.search.clear()
    w.select_job("External/job030/")
    w.info.wait_flags()
    tree = w.info.options
    rows = {tree.topLevelItem(i).child(j).text(0): tree.topLevelItem(i).child(j).text(1)
            for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())}
    assert rows["--i"] == "Refine3D/job011/run_class001.mrc" and rows["--invert_hand"] == "Yes"
    assert "External executable" not in rows
    w.select_job("External/job031/")  # any other External job: External's fields
    assert tree.topLevelItem(0).child(0).text(1) == "/home/me/scripts/my_picker.sh"


def test_options_tab_marks_changed_values(window, spa_project):
    w = window(spa_project)
    w.select_job("Refine3D/job011/")
    w.info.only_changed.setChecked(True)
    keys = []
    for i in range(w.info.options.topLevelItemCount()):
        tab = w.info.options.topLevelItem(i)
        keys += [(tab.child(j).text(0), tab.child(j).text(1)) for j in range(tab.childCount())]
    assert ("Symmetry", "D2") in keys
    assert ("Mask diameter (A)", "200") not in keys  # left at its default


def test_window_picks_up_new_job_and_status(qtbot, window, spa_copy):
    w = window(spa_copy, poll_s=0.2)
    add_job(spa_copy, NEW_JOB)
    qtbot.waitUntil(lambda: "Class2D/job019/" in w.graph.nodes, timeout=5000)
    assert "Class2D/job019/" in w.tree.visible_jobs()
    (spa_copy / "Class2D/job019/RELION_JOB_EXIT_SUCCESS").touch()
    qtbot.waitUntil(lambda: w.snapshot.jobs["Class2D/job019/"].status == "Succeeded", timeout=5000)
    assert w.snapshot.jobs["Class2D/job019/"].pending_update


def test_browsing_never_writes(window, spa_copy):
    before = tree_files(spa_copy)
    w = window(spa_copy, poll_s=0.2)
    for name in list(w.snapshot.jobs):
        w.select_job(name)
    w.refresh(force=True)
    w.close()
    assert tree_files(spa_copy) == before
    assert not is_locked(spa_copy)


def test_large_project_opens_fast(window, tmp_path):
    make_large_project(tmp_path / "big", rounds=35)
    t = time.perf_counter()
    w = window(tmp_path / "big")
    w.show()
    elapsed = time.perf_counter() - t
    assert len(w.graph.nodes) == 253
    assert elapsed < 2.0


def test_graph_nodes_do_not_overlap(tmp_path):
    make_large_project(tmp_path / "big", rounds=10)
    pos = layout_graph(ProjectLoader(tmp_path / "big").refresh()).pos
    pts = sorted(pos.values())
    for i, (x, y) in enumerate(pts):
        for x2, y2 in pts[i + 1:]:
            if x2 - x >= NODE_W:
                break
            assert abs(y2 - y) >= NODE_H, (x, y, x2, y2)


def test_collapse_progress():
    assert collapse_progress("a\nb 10%\rb 20%\rb 30%\nc") == "a\nb 30%\nc"
    assert collapse_progress("no carriage returns\n") == "no carriage returns\n"


def test_log_view_follows_appends_and_progress(qtbot, tmp_path):
    log = tmp_path / "run.out"
    log.write_text("start\n")
    v = LogView()
    qtbot.addWidget(v)
    v.load(log)
    assert v.toPlainText() == "start\n"
    with open(log, "a") as f:
        f.write("10%\r")
    assert v.refresh()
    assert v.toPlainText() == "start\n10%"
    with open(log, "a") as f:
        f.write("50%\r100%\ndone\n")
    assert v.refresh()
    assert v.toPlainText() == "start\n100%\ndone\n"
    assert not v.refresh()
    log.write_text("rewritten\n")  # replaced, not appended
    assert v.refresh()
    assert v.toPlainText() == "rewritten\n"


def test_summaries_arrive_from_worker(qtbot, tmp_path, spa_project):
    w = MainWindow(spa_project, poll_s=1000, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: "Refine3D/job011/" in w.summaries and w.snapshot is not None, timeout=5000)
    qtbot.waitUntil(lambda: len(w.summaries) == len(w.snapshot.jobs), timeout=5000)
    row = w.tree._items["Refine3D/job011/"]
    assert row[COL_SUMMARY].text() == "288 particles · 6.00 Å · D2"
    assert w.graph.nodes["Refine3D/job011/"].summary.result == ["288 particles", "6.00 Å"]


def test_cards_show_type_alias_number_and_fit_the_alias(window, spa_copy):
    alias = "a_rather_long_alias_that_must_be_fully_visible"
    add_job(spa_copy, dataclasses.replace(NEW_JOB, alias=alias, status="Succeeded"))
    w = window(spa_copy)
    item = w.graph.nodes["Class2D/job019/"]
    header, shown_alias, _ = card_lines(item.info, w.graph.type_names)
    assert (header, shown_alias) == ("2D classification", alias)
    assert item.job_line().startswith("job019 · Succeeded")
    fonts = w.graph.fonts
    assert item.w >= fonts.fm_body.horizontalAdvance(alias) + PAD_L + PAD_R
    # the tree never cuts the alias either
    assert w.tree.header().sectionResizeMode(0) == QHeaderView.ResizeToContents


def _fake_snap(parents: dict[str, str | None], others: dict[str, list[str]] | None = None):
    from types import SimpleNamespace
    jobs = {n: SimpleNamespace(primary_parent=p, other_parents=(others or {}).get(n, [])) for n, p in parents.items()}
    return SimpleNamespace(jobs=jobs)


def test_tree_layout_keeps_cards_apart_and_parents_above():
    import random
    rnd = random.Random(3)
    parents: dict[str, str | None] = {}
    for i in range(300):  # a few trees, some branching, some long chains
        parents[f"j{i}"] = None if i % 97 == 0 or not parents else f"j{rnd.choice([i - 1, i - 1, rnd.randrange(i)])}"
    sizes = {n: (rnd.choice([150.0, 210.0, 330.0]), rnd.choice([46.0, 60.0, 74.0])) for n in parents}
    lay = layout_graph(_fake_snap(parents), sizes)
    assert set(lay.pos) == set(parents)
    boxes = [(x - sizes[n][0] / 2, y - sizes[n][1] / 2, x + sizes[n][0] / 2, y + sizes[n][1] / 2)
             for n, (x, y) in lay.pos.items()]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    for n, p in parents.items():
        if p is not None:
            assert (p, n) in lay.primary and lay.routes[(p, n)] == []
            assert lay.pos[p][1] + sizes[p][1] / 2 < lay.pos[n][1] - sizes[n][1] / 2  # parent's row above
    kids = {}
    for n, p in parents.items():
        if p is not None:
            kids.setdefault(p, []).append(n)
    for p, ch in kids.items():  # centred over its first and last child
        assert abs(lay.pos[p][0] - (lay.pos[ch[0]][0] + lay.pos[ch[-1]][0]) / 2) < 1e-6


def test_tree_layout_of_long_chains_and_odd_links():
    chain = {f"j{i}": (f"j{i - 1}" if i else None) for i in range(3000)}  # no recursion limit
    lay = layout_graph(_fake_snap(chain))
    assert len({x for x, _ in lay.pos.values()}) == 1
    loop = {"a": "c", "b": "a", "c": "b", "d": "gone", "e": "e"}  # a loop of parents, a deleted parent, itself
    lay = layout_graph(_fake_snap(loop, {"d": ["a", "gone"]}))
    assert set(lay.pos) == set(loop) and ("a", "d") in lay.routes and not any("gone" in k for k in lay.routes)


def test_summaries_of_another_project_are_ignored(window, spa_project, tmp_path):
    from cryo_savanna.project.summary import JobSummary
    w = window(spa_project)
    job = next(iter(w.snapshot.jobs))
    w._on_summaries(tmp_path / "the project open before", {job: JobSummary(result=["old"])})
    assert job not in w.summaries
    w._on_summaries(w.loader.project_dir, {job: JobSummary(result=["new"])})
    assert w.summaries[job].result == ["new"]


def test_polling_intervals_have_minimums():
    from cryo_savanna.ui.main_window import _intervals
    assert _intervals(0.1, 1) == (0.5, 5.0) and _intervals(3, 0) == (3, 0.0) and _intervals(2, 30) == (2, 30)


def test_stop_repeating_runs_follows_the_flag_file(window, spa_copy):
    import json
    w = window(spa_copy)
    assert not w.actions.act_stop_runs.isEnabled()
    runs = spa_copy / ".cryo-savanna" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "r.json").write_text(json.dumps({"id": "r", "jobs": ["Class2D/job006/"], "schedule": "cryo-savanna_r",
                                             "pid": 1, "host": "h", "user": "u", "started": 0.0, "command": [],
                                             "repeat": 4}))
    (spa_copy / "RUNNING_PIPELINER_cryo-savanna_r").touch()
    w._force_refresh()
    w.actions.look_at_project_now()
    assert w.actions.act_stop_runs.isEnabled()
    assert [r.id for r in w.actions.pipeliner.repeating_runs()] == ["r"]


def test_log_panel_reads_every_log_while_run_out_keeps_changing(qtbot, tmp_path):
    from cryo_savanna.ui.log_panel import LogPanel
    (tmp_path / "run.out").write_text("start\n")
    (tmp_path / "run.err").write_text("")
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.show_job(tmp_path)
    with open(tmp_path / "run.out", "a") as f:
        f.write("iteration 2\n")
    with open(tmp_path / "run.err", "a") as f:
        f.write("ERROR: out of memory\n")
    panel.refresh()  # one poll: run.out changed, and run.err is read too
    assert "out of memory" in panel.views["run.err"].toPlainText() and panel.tabText(1) == "run.err ⚠"


def test_window_follows_a_project_with_a_loop_of_parents(window, tmp_path):
    from tests.test_snapshot import make_looped_project
    project = make_looped_project(tmp_path / "looped")
    w = window(project)
    assert sorted(w.tree._items) == sorted(w.snapshot.jobs)
    (project / "Import/job001/RELION_JOB_EXIT_SUCCESS").touch()
    w.load_now()  # only a status changed
    assert w.actions.snapshot is w.snapshot and w.snapshot.jobs["Import/job001/"].status == "Succeeded"


def test_a_relion_value_relion_refuses_is_shown_not_fatal(qtbot, tmp_path, spa_project, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    from cryo_savanna.env.relion_env import RelionEnv
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text, *a: shown.append(text))
    w = MainWindow(spa_project, poll_s=1000.0, env=RelionEnv({"RELION_QUEUE_USE": "maybe"}),
                   settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.select_job("Class2D/job006/")
    assert "Class2D/job006" in w.info.overview.toPlainText()
    assert "RELION_QUEUE_USE" in w.info.options.topLevelItem(0).text(0)
    w.actions.new_job("relion.class2d")
    assert shown and "RELION_QUEUE_USE" in shown[0] and not w.actions.editor_shown()


CLOSE_DURING_SLOW_READ = """
import sys, time
from cryo_savanna.project import summary
real = summary.SummaryReader._summarize
def slow(self, *a, **k):
    time.sleep(12)  # one big particle file over a slow file system: longer than the old wait (10 s)
    return real(self, *a, **k)
summary.SummaryReader._summarize = slow
from qtpy.QtCore import QTimer
from cryo_savanna import app
from cryo_savanna.ui import main_window
show = main_window.MainWindow.show
def show_then_close(self):
    show(self)
    QTimer.singleShot(500, self.close)
main_window.MainWindow.show = show_then_close
sys.exit(app.main(["--project", sys.argv[1]]))
"""


def test_closing_while_a_read_is_going_ends_cleanly(spa_copy, tmp_path):
    """Qt aborted ("QThread: Destroyed while thread is still running", core dump) when the window was
    closed while summaries of big files were still being read over a slow file system."""
    import subprocess
    import sys
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path))
    t = time.time()
    r = subprocess.run([sys.executable, "-c", CLOSE_DURING_SLOW_READ, str(spa_copy)], cwd=Path(__file__).parents[1],
                       env=env, capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Destroyed while thread" not in r.stderr
    assert time.time() - t < 8  # the read is dropped, not waited for


def test_main_input_can_be_changed_from_the_overview(window, spa_copy):
    from qtpy.QtCore import QUrl
    w = window(spa_copy)
    job = "PostProcess/job013/"
    w.select_job(job)
    mask = next(n for n in w.snapshot.pipeline.processes[job].inputs if n.startswith("MaskCreate/"))
    assert "make main input" in w.info.overview.toPlainText()
    w.info._on_link(QUrl(f"main:{mask}"))
    w.load_now()
    assert w.snapshot.jobs[job].primary_parent == "MaskCreate/job012/"
    assert job in w.snapshot.jobs["MaskCreate/job012/"].children
    w.select_job(job)
    assert "use the default" in w.info.overview.toPlainText()
    w.info._on_link(QUrl("main:"))
    w.load_now()
    assert w.snapshot.jobs[job].primary_parent == "Refine3D/job011/"


def test_garbage_is_collected_on_the_gui_thread_only(qapp):
    """Collected in a loader thread, a closed window's menus deadlock with the GUI thread (ui/gc_guard.py)."""
    import gc

    from cryo_savanna.ui import gc_guard
    guard = gc_guard.install(qapp)
    assert gc_guard.install(qapp) is guard and not gc.isenabled()

    class Cycle:
        pass
    a = Cycle()
    a.self = a
    gone = []
    import weakref
    weakref.finalize(a, gone.append, True)
    del a
    for _ in range(guard.threshold[0] + 1):  # pass the first threshold, as allocations do
        Cycle()
    junk = [[] for _ in range(guard.threshold[0] + 1)]
    guard.check()
    assert gone == [True]
    del junk


def chain_project(root: Path, n: int) -> Path:
    from cryo_savanna.project.pipeline import Node, Pipeline, Process
    from cryo_savanna.project.star import write_star
    p = Pipeline(job_counter=n + 1)
    for i in range(1, n + 1):
        name = f"External/job{i:03d}/"
        pr = Process(name, "None", "relion.external", "Succeeded", outputs=[name + "out.star"])
        p.nodes[name + "out.star"] = Node(name + "out.star", "ParticleGroupMetadata.star.relion", 1, producer=name)
        if i > 1:
            prev = f"External/job{i - 1:03d}/out.star"
            pr.inputs.append(prev)
            p.nodes[prev].consumers.append(name)
        p.processes[name] = pr
    root.mkdir(parents=True, exist_ok=True)
    write_star(root / "default_pipeline.star", p.to_blocks(), atomic=False)
    return root


def test_a_chain_longer_than_the_recursion_limit(qtbot, tmp_path, capsys):
    """Each job using the one before: the list and --print-tree walk it without recursion."""
    import sys
    from cryo_savanna.app import print_info
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.snapshot import job_type_names
    from cryo_savanna.ui.job_tree import JobTreeView
    n = sys.getrecursionlimit() + 200
    project = chain_project(tmp_path / "chain", n)
    assert print_info(project, False) == 0
    out = [line for line in capsys.readouterr().out.splitlines() if "External/job" in line]
    assert len(out) == n and out[-1].endswith(f"External/job{n}/") and out[-1].startswith("  " * n)
    tree = JobTreeView(job_type_names(JobDefs.load()))
    qtbot.addWidget(tree)
    tree.set_snapshot(ProjectLoader(project).refresh(force=True))
    assert tree.visible_jobs() == [f"External/job{i:03d}/" for i in range(1, n + 1)]


def test_log_keeps_a_half_written_character_for_the_next_read(qtbot, tmp_path):
    """A program writes "Å" (two bytes in UTF-8) in two goes: it is shown once whole, not as two
    replacement characters."""
    log = tmp_path / "run.out"
    data = "resolution 3.2 Å, done ██\n".encode()
    cut = data.index("Å".encode()) + 1
    log.write_bytes(data[:cut])
    view = LogView()
    qtbot.addWidget(view)
    view.load(log)
    with open(log, "ab") as f:
        f.write(data[cut:])
    os.utime(log, ns=(time.time_ns() + 10**9,) * 2)
    view.refresh()
    assert view.toPlainText() == data.decode()


def test_list_refresh_tells_the_views_only_of_what_changed(window, spa_project):
    """A refresh with nothing new (every poll while a job writes its log) must not touch the rows: Qt
    cannot tell an icon is the same one, and each set made the sorted list work the row out again."""
    from qtpy.QtCore import Qt
    w = window(spa_project)
    snap = w.snapshot
    changed = []
    w.tree.source.dataChanged.connect(lambda top, bottom, roles=(): changed.append((top.row(), list(roles))))
    w.tree.set_snapshot(snap)
    w.tree.set_queue(w.queue_state)
    assert changed == []
    job = "Refine3D/job011/"
    jobs = dict(snap.jobs)
    jobs[job] = dataclasses.replace(snap.jobs[job], status="Failed")
    w.tree.set_snapshot(dataclasses.replace(snap, jobs=jobs))
    assert any(Qt.DecorationRole in roles for _, roles in changed)  # its icon, and only its row
    assert len({row for row, _ in changed}) == 1
    changed.clear()
    w.tree.set_snapshot(snap)  # and back
    assert any(Qt.DecorationRole in roles for _, roles in changed)


def test_selected_job_is_not_read_again_while_another_job_changes(window, spa_project, monkeypatch):
    """A snapshot in which the job shown is as it was (another job wrote to its log) reads none of its files."""
    w = window(spa_project)
    snap = w.snapshot
    shown = []
    monkeypatch.setattr(w.info, "show_job", lambda s, name: shown.append(name))
    same = w.loader._rebuild(snap.pipeline, snap.pipeline_stamp, snap)  # only live state changed
    w._on_loaded(same)
    assert shown == [] and w.info._snap is same
    jobs = dict(same.jobs)
    jobs[w.current] = dataclasses.replace(same.jobs[w.current], modified=(same.jobs[w.current].modified or 0) + 5)
    w._on_loaded(dataclasses.replace(same, jobs=jobs))
    assert shown == [w.current]
    w._on_loaded(w.loader.refresh(force=True))  # the pipeline read again: looked at again
    assert shown == [w.current, w.current]


def test_log_written_anew_is_not_added_to_the_old_text(qtbot, tmp_path):
    """A job run again writes a new run.out; once that is longer than the old one was, it still is not
    text added to the old log."""
    log = tmp_path / "run.out"
    log.write_text("first run, line 1\nfirst run, line 2\n")
    v = LogView()
    qtbot.addWidget(v)
    v.load(log)
    with open(log, "a") as fh:
        fh.write("first run, line 3\n")
    assert v.refresh() and v.toPlainText() == "first run, line 1\nfirst run, line 2\nfirst run, line 3\n"
    log.write_text("second run: a new log, already longer than the first one was\n" * 3)
    assert v.refresh()
    assert v.toPlainText() == "second run: a new log, already longer than the first one was\n" * 3
    with open(log, "a") as fh:
        fh.write("and more\n")
    assert v.refresh() and v.toPlainText().endswith("than the first one was\nand more\n")
    assert v.toPlainText().count("second run") == 3


def test_summaries_fill_the_list_without_measuring_the_columns_at_every_cell(window, qtbot):
    """The fitted columns are measured once per batch of changes, not after every cell (seconds for the
    summaries of a few hundred jobs), and are still as wide as their contents afterwards."""
    from cryo_savanna.project.summary import JobSummary
    import tempfile
    root = Path(tempfile.mkdtemp()) / "big"
    make_large_project(root, rounds=35)
    w = window(root)
    w.show()
    qtbot.waitExposed(w)
    h = w.tree.header()
    resized = []
    h.sectionResized.connect(lambda c, old, new: resized.append(c))
    width = h.sectionSize(COL_SUMMARY)
    t = time.perf_counter()
    w.tree.set_summaries({n: JobSummary(result=["a rather long summary text " + n]) for n in w.snapshot.jobs})
    assert time.perf_counter() - t < 1.5  # was ~4.5 s for these 253 jobs
    assert h.sectionResizeMode(COL_SUMMARY) == QHeaderView.ResizeToContents
    assert h.sectionSize(COL_SUMMARY) > width and resized.count(COL_SUMMARY) <= 2
    assert h.sectionSize(COL_SUMMARY) >= w.tree.sizeHintForColumn(COL_SUMMARY)


def test_graph_paints_only_the_cards_that_changed(window, spa_project):
    w = window(spa_project)
    snap = w.snapshot
    painted = []
    for name, item in w.graph.nodes.items():
        item.update = lambda *a, name=name: painted.append(name)
    w.graph.set_snapshot(snap)
    assert painted == []
    jobs = dict(snap.jobs)
    jobs["Refine3D/job011/"] = dataclasses.replace(snap.jobs["Refine3D/job011/"], status="Failed")
    w.graph.set_snapshot(dataclasses.replace(snap, jobs=jobs))
    assert painted == ["Refine3D/job011/"]


def test_jobs_the_open_editor_takes_inputs_from_are_marked(window, spa_project):
    from cryo_savanna.ui import style
    from cryo_savanna.ui.job_tree import COL_JOB, COL_STATUS
    w = window(spa_project)
    a, ed = w.actions, w.actions.editor
    w.select_job("Class3D/job009/")
    a.clone_job()
    assert a.editor_shown()
    tint = style.input_tint(w.tree.palette())

    def marked() -> dict[str, list[str]]:
        tree = {n for n, row in w.tree._items.items() if row[COL_STATUS].data(Qt.BackgroundRole) == tint}
        graph = {n: item.input_labels for n, item in w.graph.nodes.items() if item.input_labels}
        assert tree == set(graph) == set(w.tree.inputs)
        return graph

    labels = {k: ed.options[k].label.rstrip(":").strip() for k in ("fn_img", "fn_ref")}
    assert marked() == {"Select/job007/": [labels["fn_img"]], "InitialModel/job008/": [labels["fn_ref"]]}
    job = w.tree._items["Select/job007/"][COL_JOB]
    assert job.font().bold() and style.input_text([labels["fn_img"]]) in job.toolTip()
    assert style.input_text([labels["fn_img"]]) in w.graph.nodes["Select/job007/"].toolTip()
    assert not w.tree._items["Class3D/job009/"][COL_JOB].font().bold()  # cloned, not an input

    # as the inputs change: a file in a job's directory (also as an absolute path), none, another node
    ed.widgets["fn_ref"].set_value(f"{spa_project}/Refine3D/job011/run_class001.mrc")
    ed.widgets["fn_img"].set_value("")
    assert marked() == {"Refine3D/job011/": [labels["fn_ref"]]}
    assert w.tree._items["Select/job007/"][COL_STATUS].data(Qt.BackgroundRole) is None
    assert "Input of" not in w.tree._items["Select/job007/"][COL_JOB].toolTip()
    ed.widgets["fn_img"].set_value("Refine3D/job011/run_data.star")
    assert marked() == {"Refine3D/job011/": [labels["fn_img"], labels["fn_ref"]]}
    ed.widgets["fn_ref"].set_value("elsewhere/map.mrc")  # not a job's
    assert marked() == {"Refine3D/job011/": [labels["fn_img"]]}
    w.load_now()  # a new look at the project keeps the marks
    assert marked() == {"Refine3D/job011/": [labels["fn_img"]]}

    a.hide_editor()  # closed: nothing marked
    assert marked() == {} and not w.tree._items["Refine3D/job011/"][COL_JOB].font().bold()
    a.act_editor.trigger()  # shown again
    assert marked() == {"Refine3D/job011/": [labels["fn_img"]]}


def test_a_change_made_while_a_poll_reads_is_read_after_it(qtbot, window, spa_copy, monkeypatch):
    """A change (Mark as failed, abort, a job added) asks for a forced refresh. Asked while a poll was still reading,
    it was dropped: the window kept the status read before the change until the next poll, or, with the same
    stamp, until the pipeline changed again."""
    import threading
    w = window(spa_copy)
    qtbot.waitUntil(lambda: not w._busy, timeout=5000)  # the first read, of the opening
    reading, go, calls = threading.Event(), threading.Event(), []
    real = w.loader.refresh

    def refresh(force=False):
        calls.append(force)
        if len(calls) == 1:  # the poll: reads the pipeline as it was, slowly
            snap = real(force)
            reading.set()
            go.wait(10)
            return snap
        return real(force)
    monkeypatch.setattr(w.loader, "refresh", refresh)
    w.refresh()
    assert reading.wait(10)
    add_job(spa_copy, NEW_JOB)  # the change, made meanwhile
    w._force_refresh()
    go.set()
    qtbot.waitUntil(lambda: "Class2D/job019/" in w.snapshot.jobs, timeout=5000)
    assert calls == [False, True]


def test_a_run_still_going_shows_and_guards_its_jobs(window, spa_copy, monkeypatch):
    """A scheduled job a live run waits to run: shown as such, Stop runs offers that run, and Delete warns (RELION's
    pipeliner looks its jobs up by name at the end of the run, and crashes if one is gone)."""
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.ui.job_tree import COL_STATUS
    from tests.test_pipeliner import _live_run
    proc = _live_run(spa_copy, "a", ["Refine3D/job016/"])
    try:
        w = window(spa_copy)
        w.actions.look_at_project_now()
        job = "Refine3D/job016/"
        assert w.tree._items[job][COL_STATUS].text() == "Scheduled · waiting for input"
        assert w.actions.act_stop_runs.isEnabled()
        w.select_job(job)
        assert "waiting for Polish/job015/shiny.star" in w.info.overview.toPlainText()
        asked = []
        monkeypatch.setattr(QMessageBox, "question", lambda parent, title, text, *a: asked.append(text)
                            or QMessageBox.No)
        w.actions.delete(False)
        assert "in a run that is still going" in asked[0]
        proc.kill()
        proc.wait()
        w.actions.look_at_project_now()
        assert w.tree._items[job][COL_STATUS].text() == "Scheduled"
        assert not w.actions.act_stop_runs.isEnabled()
    finally:
        proc.kill()
        proc.wait()


def test_out_of_date_jobs_show_and_can_be_chosen_to_run_again(window, spa_copy):
    """Extract's particles written again after the jobs that read them ran: those jobs and all after them are out
    of date, in the list, on their cards and in the overview; the dialog offers them, the selected job's first."""
    from qtpy.QtWidgets import QDialog

    from cryo_savanna.ui.job_tree import COL_STATUS
    later = time.time() + 60
    os.utime(spa_copy / "Extract/job005/particles.star", (later, later))
    w = window(spa_copy)
    w.actions.look_at_project_now()
    stale = w.actions.stale
    assert "Class2D/job006/" in stale and stale["Class2D/job006/"].changed == ("Extract/job005/particles.star",)
    assert stale["Refine3D/job011/"].upstream is not None  # further down
    assert "Extract/job005/" not in stale and "Polish/job015/" not in stale  # not after it / not finished
    assert w.tree._items["Class2D/job006/"][COL_STATUS].text().endswith("· out of date")
    assert "out of date" in w.graph.nodes["Class3D/job009/"].job_line()
    w.select_job("Class2D/job006/")
    assert "its input from Extract/job005 was written again" in w.info.overview.toPlainText()
    assert w.actions.act_rerun.isEnabled() == w.actions.pipeliner.available  # (needs relion_pipeliner)
    seen = {}

    def answer(dlg):
        for i in range(dlg.list.count()):
            it = dlg.list.item(i)
            seen[it.data(Qt.UserRole)] = (bool(it.flags() & Qt.ItemIsEnabled), it.checkState() == Qt.Checked)
        return QDialog.Rejected
    w.actions.rerun_out_of_date(answer)
    assert seen["Class2D/job006/"] == (True, True)
    assert seen["Select/job007/"][0] is False  # an interactive selection: by hand
    assert seen["Class3D/job009/"] == (True, True)  # after it: chosen too


def test_cards_of_scheduled_jobs_say_what_their_run_waits_for(window, spa_copy):
    from cryo_savanna.project.pipeliner import Held, RunRecord
    w = window(spa_copy)
    job = "Refine3D/job016/"  # Scheduled
    rec = RunRecord("r1", [job], "cryo-savanna", 1, "host", "me", time.time(), [], gpus=["0"])
    card = w.graph.nodes[job]
    for held, text in ((Held(rec, True, [], ["0"]), "waiting for a GPU"),
                       (Held(rec, True, ["Select/job010/particles.star"]), "waiting for input"),
                       (Held(rec, False), "in a run")):
        w.actions.runs.emit({job: held})
        assert card.job_line().endswith("· Scheduled · " + text)
        assert held.describe() in card.toolTip()
        assert w.tree._status_text(w.snapshot.jobs[job]).endswith(text)  # the same words as the list
    w.actions.runs.emit({})
    assert card.job_line().endswith("· Scheduled")


def _toolbar(w) -> list[str]:
    tb = w.findChild(QToolBar, "main_toolbar")
    texts = [a.text() or getattr(tb.widgetForAction(a), "text", lambda: "")() for a in tb.actions() if a.isVisible()]
    return [t for t in texts if t]


def test_toolbar_has_the_project_then_the_job_actions(window, spa_project):
    w = window(spa_project)
    assert _toolbar(w) == ["Open project…", "Refresh", "New job", "Clone…", "Continue…", "Run scheduled (1)…",
                           "Stop runs…"]
    assert not w.act_readonly.isVisible()  # only when read-only
    # the tree's own controls are next to the List/Tree switch (with the tree only), the other programs in New job
    assert w.all_links is w.graph.controls.all_links
    w.views.setCurrentWidget(w.tree)
    assert not w.graph.controls.isVisibleTo(w)
    w.views.setCurrentWidget(w.graph)
    assert w.graph.controls.isVisibleTo(w) and w.graph.controls.parent() is not w.graph
    w.all_links.setChecked(True)
    assert w.graph.show_all_links
    before = w.graph.transform().m11()
    w.graph.controls.zoom_in.click()
    assert w.graph.transform().m11() > before


def test_graph_zoomed_out_cards_show_type_and_number(window, spa_project):
    from qtpy.QtGui import QImage, QPainter, QTransform
    from qtpy.QtWidgets import QStyleOptionGraphicsItem
    from cryo_savanna.ui.job_graph import COMPACT_LOD, MIN_TEXT
    w = window(spa_project)
    item = w.graph.nodes["Class2D/job006/"]
    img = QImage(400, 300, QImage.Format_ARGB32)
    texts = []
    for lod in (0.34, 0.2, 0.05):  # painted without errors at every zoom
        p = QPainter(img)
        p.setTransform(QTransform.fromScale(lod, lod))
        p.drawText = lambda *a, p=p, lod=lod: texts.append((lod, a[-1]))
        item.paint(p, QStyleOptionGraphicsItem())
        p.end()
    assert (0.34, "2D classification") in texts and (0.34, "job006") in texts  # the compact card's text, in full
    assert not [t for lod, t in texts if lod == 0.05]  # too small: a block
    r = item.boundingRect()
    k, _, _ = item._compact_text(r.adjusted(4, 4, -4, -4), 7.0)
    assert k * COMPACT_LOD >= MIN_TEXT  # larger than the full card's text would be there


def test_graph_wheel_steps_are_drawn_together(window, spa_project, qtbot):
    from qtpy.QtCore import QPoint, QPointF
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication
    w = window(spa_project)
    g = w.graph
    w.views.setCurrentWidget(g)
    factors = []
    zoom = g.zoom
    g.zoom = lambda f: (factors.append(f), zoom(f))
    vp = g.viewport()
    for _ in range(4):  # arrive together: one frame
        QApplication.sendEvent(vp, QWheelEvent(QPointF(50, 50), QPointF(vp.mapToGlobal(QPoint(50, 50))), QPoint(),
                                               QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    qtbot.waitUntil(lambda: not g._zoom_timer.isActive())
    assert factors == [pytest.approx(1.15 ** 4)]


def test_toolbar_tooltips_give_the_shortcut_and_why_not(window, spa_project):
    w = window(spa_project)
    a = w.actions
    assert QKeySequence("Ctrl+D").toString(QKeySequence.NativeText) in a.act_clone.toolTip()
    w.select_job("Polish/job015/")  # running
    assert not a.act_continue.isEnabled() and "Not now: The selected job is running" in a.act_continue.toolTip()
    assert not a.act_stop_runs.isEnabled() and "No run that can be stopped" in a.act_stop_runs.toolTip()
    w.select_job("Import/job001/")
    assert "Not now" not in a.act_clone.toolTip()


def test_readonly_shows_on_the_toolbar(qtbot, tmp_path, spa_project):
    w = MainWindow(spa_project, poll_s=1000.0, readonly=True,
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    assert w.act_readonly.isVisible()
    assert not w.actions.act_new.isEnabled() and "read-only" in w.actions.act_new.toolTip()


def test_list_and_tree_switch(window, spa_project):
    w = window(spa_project)
    assert [w.view_switch.tabText(i) for i in range(w.view_switch.count())] == ["List", "Tree"]
    w.view_switch.setCurrentIndex(1)
    assert w.views.currentWidget() is w.graph
    w.act_tree.trigger()
    assert w.views.currentWidget() is w.tree and w.view_switch.currentIndex() == 0


def test_new_job_menu_search_recent_and_other_programs(window, spa_project, monkeypatch):
    from cryo_savanna.ui.job_actions import JobActions
    started = []  # the menu's entries are made with the window: patched before
    monkeypatch.setattr(JobActions, "new_job", lambda self, key: started.append(key))
    monkeypatch.setattr(JobActions, "new_cli_job", lambda self, p: started.append(p))
    w = window(spa_project)
    a = w.actions
    menu = a.new_menu
    texts = [x.text() for x in menu.actions()]
    assert "Other programs" in texts and "Recent" not in texts
    assert any(x.menu() is a.new_cli_menu for x in menu.actions())  # RELION's command-line tools, a submenu
    menu.search.setText("auto-refine")
    found = [x.text() for x in menu.matches()]
    assert found and all("auto-refine" in t.lower() for t in found)
    menu.search.returnPressed.emit()  # Enter: the first one
    assert started and "refine3d" in started[0]
    menu._on_show()  # opened again: Recent first, the search cleared
    texts = [x.text() for x in menu.actions() if x.text()]
    assert texts[0] == "Recent" and texts[1] == found[0] and menu.search.text() == ""
    menu.search.setText("image_handler")  # the command-line tools are searched too
    assert [x.text() for x in menu.matches()] == ["relion_image_handler"]
    menu.matches()[0].trigger()
    assert started[-1] == "relion_image_handler"
    # Recent is kept in the settings: the same in the next window (and every project)
    assert w.settings.value("recent_new_jobs", "").split("\n")[:2] == ["cli:relion_image_handler", f"job:{started[0]}"]
