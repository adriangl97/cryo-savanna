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

from qtpy.QtCore import QSettings  # noqa: E402
from qtpy.QtWidgets import QHeaderView  # noqa: E402

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
