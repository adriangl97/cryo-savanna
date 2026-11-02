# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's class selection and tilt-image exclusion windows, and the jobs they make."""

import shutil

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings, Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from cryo_savanna.project.jobstar import JobStar  # noqa: E402
from cryo_savanna.project.snapshot import ProjectLoader  # noqa: E402
from cryo_savanna.results import registry as R  # noqa: E402
from cryo_savanna.tools.validate import pipeliner_version  # noqa: E402
from cryo_savanna.viewers.tasks import pool  # noqa: E402

OPT = "Class2D/job006/run_it200_optimiser.star"
TILTS = "CtfFind/job003/tilt_series_ctf.star"

_version = pipeliner_version()
needs_relion = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                  reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")


def settle(qtbot, rounds=6):
    for _ in range(rounds):
        pool().waitForDone(20000)
        QApplication.processEvents()
        qtbot.wait(30)


def test_class_selector(qtbot, spa_project):
    from cryo_savanna.viewers.selectors import ClassSelector
    w = ClassSelector(spa_project, OPT, selected=[3, 7])
    qtbot.addWidget(w)
    w.show()
    settle(qtbot)
    assert len(w.gallery.model.items) == 50 and w.selected() == [3, 7]
    assert "2 of 50 classes selected" in w.gallery.status.text() and "particles" in w.gallery.status.text()
    assert w.recenter.isVisible()  # 2D classes
    w.panel._all()
    assert len(w.selected()) == 50
    w.panel._invert()
    assert w.selected() == []
    got = []
    w.saved.connect(got.append)
    w._save()
    assert got == [] and "at least one" in w.message.text()
    w.set_selected([12, 1])
    w.regroup.setValue(0)
    w._save()
    assert got == [([1, 12], False, 0)]
    # a click toggles, as in relion_display
    rect = w.gallery.list.visualRect(w.gallery.model.index(4))
    qtbot.mouseClick(w.gallery.list.viewport(), Qt.LeftButton, pos=rect.center())
    assert w.selected() == [1, 5, 12]
    qtbot.mouseClick(w.gallery.list.viewport(), Qt.LeftButton, pos=rect.center())
    assert w.selected() == [1, 12]


def test_tilt_excluder(qtbot, tomo_project):
    from cryo_savanna.viewers.selectors import TiltExcluder
    w = TiltExcluder(tomo_project, TILTS)
    qtbot.addWidget(w)
    w.show()
    settle(qtbot)
    v = w.viewer
    assert v.series_model.rowCount() == 5 and v.tilt_model.columns[0] == ("_keep", "Keep")
    # sorted by tilt angle; starts at 0°
    angles = [float(v.tilt_proxy.index(r, 2).data()) for r in range(v.tilt_proxy.rowCount())]  # #, Keep, Tilt
    assert angles == sorted(angles)
    row = v.tilt_proxy.mapToSource(v.tilt_view.currentIndex()).row()
    w.toggle_current()
    assert w.kept["TS_01"][row] == np.False_ and w.kept["TS_01"].sum() == 40
    assert v.tilt_model.extra["_keep"][row] == "✗" and row in v.tilt_model.row_brushes
    assert v.series_model.extra["_kept"][0] == "40 of 41"
    # the next image is selected after a toggle; toggling it twice keeps it
    w.toggle_current()
    assert w.kept["TS_01"].sum() == 39
    w.viewer.tilt_view.setCurrentIndex(v.tilt_proxy.mapFromSource(v.tilt_model.index(row, 0)))
    w.toggle_current()
    assert w.kept["TS_01"].sum() == 40
    w._step_series(1)
    settle(qtbot)
    assert w.panel._name() == "TS_03"
    w._keep_all()
    got = []
    w.saved.connect(got.append)
    w._save()
    assert list(got[0]) == ["TS_01", "TS_03"] and sum(got[0]["TS_01"]) == 40
    # a previous exclusion is shown again
    w2 = TiltExcluder(tomo_project, TILTS, kept={"TS_01": [i > 2 for i in range(41)]})
    qtbot.addWidget(w2)
    w2.show()
    settle(qtbot)
    assert w2.viewer.series_model.extra["_kept"][0] == "38 of 41"


def test_views_offer_the_tools(spa_project, tomo_project):
    def specs(project, job):
        snap = ProjectLoader(project).refresh(True)
        opts = JobStar.read(project / job / "job.star").options
        return {v.title: v for v in R.views_for(R.context_for(project, snap.jobs[job], snap.pipeline, opts))}
    assert specs(spa_project, "Class2D/job006/")["Classes"].args["tool"] == "classes"
    assert specs(tomo_project, "CtfFind/job003/")["Tilt series"].args == {
        "star": "CtfFind/job003/tilt_series_ctf.star", "tool": "tilts", "tool_arg": "CtfFind/job003/tilt_series_ctf.star"}
    # a selection job shows its choice in a Selection tab (with "Change selection…"), first
    sel = specs(spa_project, "Select/job007/")
    assert list(sel)[0] == "Selection" and sel["Selection"].args == {"job": "Select/job007/", "what": "classes"}
    assert "tool" not in sel["Selected classes"].args
    tilts = specs(tomo_project, "ExcludeTiltImages/job004/")
    assert list(tilts)[0] == "Selection" and tilts["Selection"].args["what"] == "tilts"
    assert tilts["Tilt series"].args["tool"] == "tilts"  # its output is a tilt-series set like any other
    assert specs(spa_project, "Select/job010/")["Selection"].args["what"] == "classes"  # 3D classes too
    ctx = R.JobContext(spa_project, "Select/job010/", "relion.select",
                       options={"fn_model": "Class3D/job009/run_it025_optimiser.star", "do_discard": "Yes"})
    assert "Selection" not in [v.title for v in R.views_for(ctx)]  # not an interactive selection


def test_selection_view_shows_the_saved_classes_read_only(qtbot, spa_copy):
    from cryo_savanna.viewers.selectors import SelectionView
    asked = []
    v = SelectionView(spa_copy, "Select/job007/", "classes", request_tool=lambda k, a: asked.append((k, a)))
    qtbot.addWidget(v)
    v.show()
    settle(qtbot)
    panel = v.panel
    assert panel.selected() == [3, 7, 12, 18, 22] and "highlighted" in v.note.text()
    assert "5 of 50 classes selected" in panel.gallery.status.text()
    assert not panel.recenter.isEnabled() and not panel.regroup.isEnabled()
    # a click changes nothing here
    rect = panel.gallery.list.visualRect(panel.gallery.model.index(0))
    qtbot.mouseClick(panel.gallery.list.viewport(), Qt.LeftButton, pos=rect.center())
    assert panel.selected() == [3, 7, 12, 18, 22]
    v.change_btn.click()
    assert asked == [("change", "Select/job007/")]
    # the job's selection is changed (by the window, or by relion_display): the view follows
    v.refresh()
    assert v.panel is panel  # nothing changed yet
    (spa_copy / "Select/job007/backup_selection.star").write_text(
        "\ndata_\n\nloop_\n_rlnSelected #1\n" + "".join("1\n" if i in (1, 2) else "0\n" for i in range(1, 51)))
    v.refresh()
    settle(qtbot)
    assert v.panel is not panel and v.panel.selected() == [1, 2]


def test_selection_view_shows_the_kept_tilts_read_only(qtbot, tomo_project, tmp_path):
    from cryo_savanna.project.selection import read_tilt_input, write_tilt_exclusion
    from cryo_savanna.viewers.selectors import SelectionView
    proj = tmp_path / "tomo"
    shutil.copytree(tomo_project, proj, symlinks=True)
    job = "ExcludeTiltImages/job004/"
    inp = read_tilt_input(proj, TILTS)
    write_tilt_exclusion(inp, job, {"TS_01": [i > 2 for i in range(len(inp.tilts["TS_01"]))]})
    v = SelectionView(proj, job, "tilts", request_tool=lambda k, a: None)
    qtbot.addWidget(v)
    v.show()
    settle(qtbot)
    tv = v.panel.viewer
    assert tv.series_model.extra["_kept"][0] == "38 of 41" and "red" in v.note.text()
    row = tv.tilt_proxy.mapToSource(tv.tilt_view.currentIndex()).row()
    before = list(tv.tilt_model.extra["_keep"])
    v.panel.toggle_current()  # read-only: no toggling, no buttons
    qtbot.keyClick(tv.tilt_view, Qt.Key_Return)
    assert list(tv.tilt_model.extra["_keep"]) == before and tv.tilt_model.extra["_keep"][row] in "✓✗"
    assert not hasattr(v.panel, "toggle_btn")
    assert v.change_btn.isVisible()


# ---------------------------------------------------------------------------- the whole flow


@pytest.fixture
def window(qtbot, tmp_path):
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.project.slurm import SlurmClient
    from cryo_savanna.ui.main_window import MainWindow

    def make(project):
        w = MainWindow(project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat),
                       slurm=SlurmClient(which=lambda p: None))
        qtbot.addWidget(w)
        w.load_now()
        return w
    return make


def wait_tool(w, qtbot):
    qtbot.waitUntil(lambda: bool(w.actions.tools), timeout=5000)
    tool = w.actions.tools[-1]
    settle(qtbot)
    return tool


@needs_relion
def test_select_classes_makes_a_finished_select_job(window, qtbot, spa_copy):
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.selection import read_backup_selection
    w = window(spa_copy)
    w.select_job("Class2D/job006/")
    view = w.info.show_view("Classes")
    settle(qtbot)
    assert view.tool_btn is not None and view.tool_source() == OPT
    view.tool_btn.click()
    tool = wait_tool(w, qtbot)
    tool.set_selected([3, 7])
    tool.save_btn.click()
    w.actions.wait()
    qtbot.waitUntil(lambda: tool not in w.actions.tools, timeout=10000)  # closed (and deleted)
    p = Pipeline.read(spa_copy)
    job = max((n for n, pr in p.processes.items() if n.startswith("Select/")), key=lambda n: p.processes[n].job_number)
    pr = p.processes[job]
    assert pr.status == "Succeeded" and pr.type_label == "relion.select.interactive"
    assert read_backup_selection(spa_copy, job, 50) == [n in (3, 7) for n in range(1, 51)]
    js = JobStar.read(spa_copy / job / "job.star")
    assert js.options["fn_model"] == OPT
    note = (spa_copy / job / "note.txt").read_text()
    assert f"--fn_parts {job}particles.star --fn_imgs {job}class_averages.star" in note

    # change the selection from the Select job's own tab: the same job, rewritten
    w.load_now()
    w.select_job(job)
    sel = w.info.show_view("Selection")
    settle(qtbot)
    assert sel.panel.selected() == [3, 7]  # shown read-only
    sel.change_btn.click()
    tool = wait_tool(w, qtbot)
    assert tool.selected() == [3, 7]
    tool.set_selected([7])
    tool.recenter.setChecked(True)
    tool.save_btn.click()
    w.actions.wait()
    qtbot.waitUntil(lambda: tool not in w.actions.tools, timeout=10000)  # closed (and deleted)
    assert read_backup_selection(spa_copy, job, 50).count(True) == 1
    w.info.refresh_viewers()  # the next poll: the Selection tab shows the new choice
    settle(qtbot)
    assert w.info.current_view().panel.selected() == [7]
    # job.star follows the change, so RELION's GUI continues the job with the same settings
    assert JobStar.read(spa_copy / job / "job.star").options["do_recenter"] == "Yes"
    assert (spa_copy / job / "class_averages.mrcs").exists()
    assert len([n for n in Pipeline.read(spa_copy).processes if n.startswith("Select/")]) == len(
        [n for n in p.processes if n.startswith("Select/")])


@needs_relion
def test_editor_run_of_an_interactive_select_opens_the_tool(window, qtbot, spa_copy):
    from cryo_savanna.project.pipeline import Pipeline
    w = window(spa_copy)
    before = set(Pipeline.read(spa_copy).processes)
    w.actions.new_job(w.defs.variant("relion.select").key)
    ed = w.actions.editor
    ed.widgets["fn_model"].set_value(OPT)
    js = ed.jobstar()
    w.actions._on_submit(js, "", False)
    tool = wait_tool(w, qtbot)
    assert set(Pipeline.read(spa_copy).processes) == before  # nothing is added before saving
    tool.close()


def test_running_tab_chooses_the_selection_window(window, qtbot, spa_copy, tomo_project, monkeypatch):
    w = window(spa_copy)
    w.actions.new_job(w.defs.variant("relion.select").key)
    ed = w.actions.editor
    running = [i for i in range(ed.tabs.count()) if ed.tabs.tabText(i).startswith("Running")][0]
    assert ed.selector is not None and ed.tabs.widget(running).isAncestorOf(ed.selector)
    assert ed.selector.currentText() == "cryo-savanna's selector" and ed.native_selector  # the default
    assert not ed.selector.isEnabled()  # no classes chosen yet: not an interactive selection
    ed.widgets["fn_model"].set_value(OPT)
    assert ed.selector.isEnabled()
    runs = []
    monkeypatch.setattr(w.actions, "_run", lambda title, fn, done=None: runs.append(title))
    w.actions._on_submit(ed.jobstar(), "", False)
    tool = wait_tool(w, qtbot)  # cryo-savanna's window, nothing run
    tool.close()
    assert runs == []
    ed.selector.setCurrentIndex(1)  # RELION's relion_display: a normal run through relion_pipeliner
    assert ed.selector.currentText() == "RELION's (relion_display)" and not ed.native_selector
    w.actions._on_submit(ed.jobstar(), "", False)
    assert runs == ["Run job"]
    ed.selector.setCurrentIndex(0)
    w.actions._on_submit(ed.jobstar(), "", True)  # Schedule: always RELION's tool, when the schedule runs
    assert runs == ["Run job", "Schedule job"]
    ed.widgets["do_discard"].set_value("Yes")  # not an interactive selection any more
    assert not ed.selector.isEnabled()

    w.actions.new_job(w.defs.variant("relion.class2d").key)  # other jobs have no such choice
    assert w.actions.editor.selector is None and not w.actions.editor.native_selector
    w.actions.new_job(w.defs.variant("relion.excludetilts", True).key)
    assert w.actions.editor.selector.currentText() == "cryo-savanna's selector"
    assert w.actions.editor.selector.itemText(1) == "RELION's (napari)"


@needs_relion
def test_exclude_tilts_makes_a_finished_job(window, qtbot, tomo_project, tmp_path):
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.results.tomo import read_tilt_series_set
    proj = tmp_path / "tomo"
    shutil.copytree(tomo_project, proj, symlinks=True)
    w = window(proj)
    w.select_job("CtfFind/job003/")
    view = w.info.show_view("Tilt series")
    settle(qtbot)
    view.tool_btn.click()
    tool = wait_tool(w, qtbot)
    tool.toggle_current()
    tool.toggle_current()
    tool.save_btn.click()
    w.actions.wait()
    qtbot.waitUntil(lambda: tool not in w.actions.tools, timeout=10000)  # closed (and deleted)
    p = Pipeline.read(proj)
    job = max((n for n in p.processes if n.startswith("ExcludeTiltImages/")), key=lambda n: p.processes[n].job_number)
    assert job != "ExcludeTiltImages/job004/" and p.processes[job].status == "Succeeded"
    assert p.processes[job].outputs == [job + "selected_tilt_series.star"]
    s = read_tilt_series_set(proj, job + "selected_tilt_series.star")
    assert len(s.tilts(0)) == 39 and len(s.tilts(1)) == 41


def test_tilt_excluder_drops_a_whole_tilt_series(qtbot, tomo_project):
    """Excluding every tilt of one series drops it; the series not looked at keep all their tilts."""
    from cryo_savanna.viewers.selectors import TiltExcluder
    w = TiltExcluder(tomo_project, TILTS)
    qtbot.addWidget(w)
    w.show()
    settle(qtbot)
    w.kept["TS_01"][:] = False
    got = []
    w.saved.connect(got.append)
    w._save()
    assert got and not any(got[0]["TS_01"])
    for name in w.viewer.set.names:  # every series excluded: refused
        w.kept[name] = np.zeros(len(w.viewer.set.tilts(w.viewer.set.index(name))), bool)
    w._save()
    assert len(got) == 1 and "Every tilt image" in w.message.text()


def test_additional_arguments_for_relions_tool_are_not_dropped_silently(window, spa_copy, qtbot, monkeypatch):
    """RELION passes a Select job's additional arguments to relion_display (--max_nr_parts_per_class, ...),
    which cryo-savanna's window cannot follow: it asks first, and opens nothing on No."""
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.project.jobstar import JobStar
    w = window(spa_copy)
    js = JobStar.read(spa_copy / "Select/job007/job.star")
    js.options["other_args"] = "--max_nr_parts_per_class 100"
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.No)[1])
    w.actions.open_tool("classes", js, "", None, None)
    assert asked and "--max_nr_parts_per_class 100" in asked[0] and not w.actions.tools
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w.actions.open_tool("classes", js, "", None, None)
    assert wait_tool(w, qtbot) is not None
    js.options["other_args"] = ""
    asked.clear()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.No)[1])
    w.actions.open_tool("classes", js, "", None, None)
    assert not asked  # nothing to ask without them
