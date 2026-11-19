# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Job editor and job actions, offscreen."""

import os
import shutil
import stat
import sys
import time

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings, Qt  # noqa: E402
from qtpy.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from cryo_savanna.env.relion_env import RelionEnv  # noqa: E402
from cryo_savanna.jobdefs.flags import FlagSource  # noqa: E402
from cryo_savanna.jobdefs.model import JobDefs  # noqa: E402
from cryo_savanna.project.jobstar import JobStar  # noqa: E402
from cryo_savanna.project.pipeline import Pipeline  # noqa: E402
from cryo_savanna.project.snapshot import job_type_names  # noqa: E402
from cryo_savanna.project.star import write_star  # noqa: E402
from cryo_savanna.tools.validate import EXTERNAL_SCRIPT, pipeliner_version  # noqa: E402
from cryo_savanna.ui import preferences as P  # noqa: E402
from cryo_savanna.jobdefs.flags import ADVANCED_TAB  # noqa: E402
from cryo_savanna.project import references  # noqa: E402
from cryo_savanna.ui.job_editor import JobEditor, NodeWidget, node_matches, qt_file_filter  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402

DEFS = JobDefs.load()
ENV = RelionEnv({})
FLAGS = FlagSource(which=lambda p: None)  # the bundled 5.1.1 --help texts, not whatever RELION is on PATH
_version = pipeliner_version()
needs_relion = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                  reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")


@pytest.fixture
def editor(qtbot, spa_project):
    e = JobEditor(ENV, job_type_names(DEFS), flag_source=FLAGS)
    qtbot.addWidget(e)
    e.set_project(spa_project, Pipeline.read(spa_project))
    return e


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning shown: {a[1:]}"))

    def make(project, **kw):
        w = MainWindow(project, poll_s=1000, env=ENV, flag_source=FLAGS,
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat), **kw)
        qtbot.addWidget(w)
        w.load_now()
        return w
    return make


def test_untouched_editor_writes_relions_defaults(editor):
    for variant in DEFS.variants.values():
        editor.new_job(variant)
        assert editor.jobstar().to_text() == variant.new_jobstar(ENV).to_text(), variant.key


def test_groups_follow_their_switch(editor):
    editor.new_job(DEFS.variant("relion.class2d"))
    assert editor.widgets["do_grad"].value() == "Yes"
    assert editor.widgets["nr_iter_grad"].isEnabled() and not editor.widgets["nr_iter_em"].isEnabled()
    editor.widgets["do_em"].set_value("Yes")
    editor.widgets["do_grad"].set_value("No")
    assert editor.widgets["nr_iter_em"].isEnabled() and not editor.widgets["nr_iter_grad"].isEnabled()
    assert set(editor.changed_options()) == {"do_em", "do_grad"}


def test_continue_mode_only_allows_what_relion_allows(editor, spa_project):
    js = JobStar.read(spa_project / "Class2D/job006/job.star")
    editor.continue_job("Class2D/job006/", js, DEFS.variant("relion.class2d"))
    assert not editor.widgets["fn_img"].isEnabled()  # deactivate in continue mode
    assert editor.widgets["fn_cont"].isEnabled()  # only active in continue mode
    assert editor.widgets["nr_classes"].value() == "50"
    out = editor.jobstar()
    assert out.is_continue and out.options == {**js.options, **{k: v for k, v in out.options.items() if k not in js.options}}


def test_overwrite_mode_edits_everything(editor, spa_project):
    js = JobStar.read(spa_project / "Class2D/job006/job.star")
    editor.overwrite_job("Class2D/job006/", js, DEFS.variant("relion.class2d"), alias="my_classes")
    assert editor.widgets["fn_img"].isEnabled() and not editor.widgets["fn_cont"].isEnabled()  # like a new job
    assert editor.widgets["nr_classes"].value() == "50" and editor.alias.text() == "my_classes"
    assert editor.run_btn.text() == "Overwrite now" and editor.schedule_btn.isVisibleTo(editor)
    assert not editor.jobstar().is_continue and editor.jobstar().options == js.options


def test_overwrite_action_follows_the_status(window, spa_project):
    w = window(spa_project)
    w.select_job("Class2D/job006/")
    a = w.actions
    assert a.act_overwrite.isEnabled() == a.act_continue.isEnabled()
    running = [n for n, i in w.snapshot.jobs.items() if i.status == "Running"][0]
    w.select_job(running)
    assert not a.act_overwrite.isEnabled()


def test_input_lists_offer_matching_outputs(editor):
    editor.new_job(DEFS.variant("relion.refine3d"))
    w = editor.widgets["fn_img"]
    assert isinstance(w, NodeWidget)
    items = [w.combo.itemData(i) for i in range(w.combo.count())]
    assert items[0] == "Class2D/job018/run_it025_data.star"  # newest job first
    assert "Extract/job005/particles.star" in items and "MaskCreate/job012/mask.mrc" not in items
    w.combo.setCurrentIndex(items.index("Select/job010/particles.star"))
    assert editor.values()["fn_img"] == "Select/job010/particles.star"
    w.set_value("my/own.star")
    assert editor.values()["fn_img"] == "my/own.star"



@pytest.mark.parametrize("label", ["TomoOptimisationSet.star.relion.tomo.pytom",  # pytom extraction
                                   "TomoOptimisationSet.star.relion"])
def test_input_lists_follow_outputs_added_when_a_job_finishes(window, tomo_project, tmp_path, label):
    """An External job's outputs (its RELION_OUTPUT_NODES.star) go into the pipeline only when it finishes, a
    status change: the input lists, of the form already open and of new ones, offer them then."""
    from cryo_savanna.project.lock import PipelineLock
    from cryo_savanna.project.pipeline import Node, Process
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    job = "External/job030/"
    out = job + "optimisation_set.star"
    (dst / job).mkdir(parents=True)

    def write(status, outputs):
        p = Pipeline.read(dst)
        p.processes[job] = Process(job, "None", "relion.external", status, outputs=list(outputs))
        for n in outputs:
            p.nodes[n] = Node(n, label, 1, producer=job)
        with PipelineLock(dst) as lock:
            p.write(dst, "default", lock)

    write("Running", [])
    w = window(dst)
    ed = w.actions.editor
    ed.new_job(DEFS.variant("relion.pseudosubtomo:tomo"))

    def items():
        c = ed.widgets["in_optimisation"].combo
        return [c.itemData(i) for i in range(c.count())]
    assert out not in items()
    (dst / out).write_text("")
    write("Succeeded", [out])  # what relion_pipeliner does when the job ends
    w.load_now()
    assert items()[0] == out
    ed.new_job(DEFS.variant("relion.reconstructparticletomo:tomo"))
    assert items()[0] == out


def test_search_filters_rows(editor):
    editor.new_job(DEFS.variant("relion.class3d"))
    editor.search.setText("symmetry")
    visible = [row.keys for _, row, parts in editor.rows if not parts[0].isHidden()]
    assert ("sym_name",) in visible and ("nr_classes",) not in visible


def _tab(editor, label):
    return next(i for i in range(editor.tabs.count()) if editor.tabs.tabText(i).split("  (")[0] == label)


def test_advanced_parameters_tab(editor):
    editor.new_job(DEFS.variant("relion.class3d"))
    adv, box = editor.advanced, editor.widgets["other_args"]
    tabs = [editor.tabs.tabText(i) for i in range(editor.tabs.count())]
    assert tabs[-2:] == ["Running", ADVANCED_TAB] and editor.advanced_tab == len(tabs) - 1
    running = [row.keys for ti, row, _ in editor.rows if ti == _tab(editor, "Running")]
    assert ("other_args",) not in running  # the box moved to the new tab, at its top
    assert box.parent() is adv and adv.program == "relion_refine"
    # the flags the job writes itself are left out: from its options (other tabs) and those it always writes
    hidden = set(adv.hidden())
    assert {"--iter", "--K", "--tau2_fudge", "--ref", "--firstiter_cc", "--norm", "--o", "--j", "--gpu"} <= hidden
    assert not hidden & adv.rows.keys() and len(adv.rows) + len(hidden) == 150
    assert "--iter: Number of iterations" in adv.header.toolTip()
    assert [s.name for s in adv.sections][:2] == ["General options", "Initialisation"]  # as in --help
    # a switch is Yes/No, a value a field with the program's default; changing one writes the box
    adv.rows["--dont_skip_gridding"].widget.set_value("Yes")
    adv.rows["--maskedge"].widget.set_value("7")
    assert box.value() == "--dont_skip_gridding --maskedge 7"
    assert adv.rows["--maskedge"].label.font().bold() and not adv.rows["--verb"].label.font().bold()
    assert editor.changed_options()["other_args"] == ("", "--dont_skip_gridding --maskedge 7")
    adv.rows["--maskedge"].widget.set_value("5")  # back to the default: not passed
    assert box.value() == "--dont_skip_gridding"
    # typing in the box fills in the rows, and names what is not one of them
    box.set_value("--pad 1 my_note --nonsense --maskedge 9")
    assert adv.rows["--maskedge"].widget.value() == "9" and adv.rows["--dont_skip_gridding"].widget.value() == "No"
    assert not adv.notice.isHidden() and "--nonsense" in adv.notice.text() and "--pad" in adv.notice.text()
    adv.rows["--maskedge"].widget.set_value("5")
    assert box.value() == "--pad 1 my_note --nonsense"
    assert editor.jobstar().options["other_args"] == "--pad 1 my_note --nonsense"
    assert "command line: --tau2_fudge" in (editor.show_help("tau_fudge") or editor.help.toPlainText())


def test_advanced_parameters_follow_the_program_and_the_search(editor):
    editor.new_job(DEFS.variant("relion.autopick"))
    assert editor.advanced.program == "relion_autopick"
    editor.widgets["do_amyloid"].set_value("Yes")
    assert editor.advanced.program == "relion_find_amyloid"
    editor.new_job(DEFS.variant("relion.external"))
    assert editor.advanced.flags is None and "your program" in editor.advanced.header.text()
    editor.new_job(DEFS.variant("relion.refine3d"))
    editor.search.setText("skip_gridding")  # a flag, not a job option
    adv = _tab(editor, ADVANCED_TAB)
    assert editor.tabs.isTabEnabled(adv) and editor.tabs.tabText(adv) == f"{ADVANCED_TAB}  (2)"  # it and the box
    shown = [n for n, r in editor.advanced.rows.items() if not r.label.isHidden()]
    assert shown == ["--dont_skip_gridding"] and editor.tabs.currentIndex() == adv
    editor.search.clear()
    editor.advanced.rows["--verb"].widget.set_value("0")
    editor.only_changed.setChecked(True)
    assert [n for n, r in editor.advanced.rows.items() if not r.label.isHidden()] == ["--verb"]
    assert editor.tabs.tabText(adv) == f"{ADVANCED_TAB}  (2)"
    editor.widgets["other_args"].set_value("")
    assert editor.tabs.tabText(adv) == f"{ADVANCED_TAB}  (0)"


def test_advanced_parameters_in_continue_mode(editor, spa_project):
    js = JobStar.read(spa_project / "Class3D/job009/job.star")
    editor.continue_job("Class3D/job009/", js, DEFS.variant("relion.class3d"))
    row = editor.advanced.rows["--maskedge"]
    assert row.widget.isEnabled() == editor.widgets["other_args"].isEnabled()
    editor.set_permissions(readonly=True, can_run=False)
    assert not row.widget.isEnabled()


def test_cli_tools_keep_their_box_on_running(editor):
    from cryo_savanna.jobdefs.cli_tools import CliCatalog
    v = CliCatalog(DEFS, ENV, FLAGS).variant("relion_image_handler")
    editor.new_job(v)
    assert editor.advanced is None and ADVANCED_TAB not in [editor.tabs.tabText(i) for i in range(editor.tabs.count())]


def test_several_reference_maps(editor, spa_copy, tmp_path):
    editor.set_project(spa_copy, Pipeline.read(spa_copy))  # (a list is written into a job's directory below)
    editor.new_job(DEFS.variant("relion.class3d"))
    maps = ["Refine3D/job011/run_class001.mrc", "InitialModel/job008/initial_model.mrc", "External/job099/my_map.mrc"]
    seen = {}

    def hook(dlg):
        seen["nodes"] = [dlg.list.item(i).data(Qt.UserRole + 1) for i in range(dlg.list.count())]
        for m in reversed(maps):
            dlg.tick(m)
        assert dlg.maps() == maps[::-1]  # in the order ticked, at the top of the list, numbered
        assert [dlg.list.item(i).text() for i in range(3)][0].startswith("1.  External/job099/my_map.mrc")
        dlg.list.setCurrentRow(0)
        dlg.move(1)
        dlg.list.setCurrentRow(1)
        dlg.move(1)  # the first map is now the last class
        dlg.tick("Refine3D/job011/run_class001.mrc", False)
        dlg.tick("Refine3D/job011/run_class001.mrc")
        assert dlg.maps() == ["InitialModel/job008/initial_model.mrc", "External/job099/my_map.mrc",
                              "Refine3D/job011/run_class001.mrc"]
        dlg.tick("InitialModel/job008/initial_model.mrc", False)
        dlg.tick("InitialModel/job008/initial_model.mrc")
        assert dlg.maps() == ["External/job099/my_map.mrc", "Refine3D/job011/run_class001.mrc",
                              "InitialModel/job008/initial_model.mrc"]
        dlg.list.setCurrentRow(0)
        dlg.move(1)
        dlg.move(1)  # (the map moved stays selected)
        assert dlg.maps() == maps and "3 classes" in dlg.summary.text()
        return True
    assert editor.open_references(dialog_hook=hook)
    assert "Refine3D/job011/run_class001.mrc" in seen["nodes"]  # the pipeline's maps are offered
    own = references.path_in(editor.job_dir())
    ref, k = editor.widgets["fn_ref"], editor.widgets["nr_classes"]
    assert ref.value() == own and k.value() == "3" and not ref.combo.isEnabled() and not k.isEnabled()
    assert editor.refs_button.isEnabled() and "Change…" in editor.refs_line.text()
    assert editor.reference_maps() == maps and not editor.refs_line.isHidden()
    assert "3 reference maps" in editor.refs_line.text() and "initial_model.mrc" in editor.refs_line.text()
    assert {j: v for j, v in editor.input_jobs().items() if "Reference map" in v}.keys() >= {
        "Refine3D/job011/", "InitialModel/job008/"}
    js = editor.jobstar("Class3D/job042/", values={**editor.values(), "nr_classes": "5"})  # Variations…
    assert js.options["fn_ref"] == "Class3D/job042/references.star" and js.options["nr_classes"] == "3"
    from cryo_savanna.project.presets import PresetStore
    editor.presets = PresetStore(tmp_path / "presets")
    editor.presets.save(editor.variant.key, "K8", {"nr_classes": "8", "tau_fudge": "2"}, "5.1.1")
    assert editor.load_preset("K8")  # keeps the inputs (the list), and so its number of classes
    assert editor.references == maps and editor.widgets["nr_classes"].value() == "3"
    assert editor.widgets["tau_fudge"].value() == "2"
    # a clone of a job with a list: its maps, and a list of its own
    job = "Class3D/job009/"
    references.write(spa_copy, job, maps[:2])
    vals = dict(JobStar.read(spa_copy / job / "job.star").options, fn_ref=references.path_in(job), nr_classes="2")
    editor.new_job(DEFS.variant("relion.class3d"), vals, source_job=job)
    assert editor.references == maps[:2] and editor.widgets["fn_ref"].value() == references.path_in(editor.job_dir())
    editor.set_references(maps[:1])  # one map: as RELION has it
    assert editor.widgets["fn_ref"].value() == maps[0] and editor.reference_maps() == []
    assert editor.widgets["nr_classes"].isEnabled() and editor.refs_line.isHidden()
    editor.widgets["fn_ref"].set_value(references.path_in(job))  # a list typed in the field
    assert editor.references == maps[:2] and editor.widgets["nr_classes"].value() == "2"
    editor._reset_defaults()
    assert editor.references == [] and editor.widgets["fn_ref"].value() == ""
    # only the 3D classification has it
    editor.new_job(DEFS.variant("relion.refine3d"))
    assert editor.refs_button is None


def test_cli_tools_menu_and_form(window, spa_project):
    w = window(spa_project)
    a = w.actions
    assert a.cli_menu in [m.menu() for m in w.menuBar().actions()]
    categories = [m.text() for m in a.cli_menu.actions()]
    assert categories[0] == "Maps and images" and "Tomography" in categories
    a.new_cli_job("relion_image_handler")
    ed = a.editor
    assert ed.is_cli and ed.advanced is None and "relion_image_handler" in ed.title.text()
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())][-1] == "Running"
    ed.widgets["--i"].set_value("Refine3D/job011/run_class001.mrc")
    ed.widgets["--o"].set_value("flipped.mrc")
    ed.widgets["--invert_hand"].set_value("Yes")
    assert set(ed.changed_options()) == {"--i", "--o", "--invert_hand"}
    js = ed.jobstar("External/job099/")
    assert js.type_label == "relion.external" and js.options["in_3dref"] == "Refine3D/job011/run_class001.mrc"
    assert js.options["other_args"] == "--i Refine3D/job011/run_class001.mrc --o External/job099/flipped.mrc --invert_hand"
    ed.search.setText("invert hand")
    shown = [row.keys for _, row, parts in ed.rows if not parts[0].isHidden()]
    assert ("--invert_hand",) in shown and ("--angpix",) not in shown


def test_cli_tool_mpi_keeps_to_the_local_limit(qtbot, tmp_path, spa_project, monkeypatch):
    """As relion_pipeliner refuses RELION's own jobs: no more local MPI processes than RELION_ERROR_LOCAL_MPI."""
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    w = MainWindow(spa_project, poll_s=1000, env=RelionEnv({"RELION_ERROR_LOCAL_MPI": "4"}), flag_source=FLAGS,
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat), readonly=False)
    qtbot.addWidget(w)
    w.load_now()
    a = w.actions
    monkeypatch.setattr(a, "_run", lambda *args, **kw: pytest.fail("the job was added"))
    a.new_cli_job("relion_reconstruct")
    ed = a.editor
    ed.widgets["nr_mpi"].set_value("5")
    a._on_submit(ed.jobstar(), "", True)
    assert len(warned) == 1 and "5 parallel MPI processes" in warned[0] and "RELION_ERROR_LOCAL_MPI" in warned[0]



def test_editor_closes_when_its_job_starts(window, spa_project, monkeypatch):
    from types import SimpleNamespace
    w = window(spa_project)
    a, ed = w.actions, w.actions.editor
    pending = []  # (done, result) of the runs started, as _run would call them once finished

    def fake_run(what, fn, done=None, on_error=None):
        res = SimpleNamespace(job="Class2D/job019/", warnings=[])
        pending.append((done, (res, None if "Schedule" in what else object())))
    monkeypatch.setattr(a, "_run", fake_run)

    def finish():
        done, result = pending.pop()
        done(result)

    a.new_job("relion.class2d")
    ed.schedule_btn.click()
    finish()
    assert a.editor_shown() and "scheduled" in ed.status.text()  # kept open for the next one
    ed.run_btn.click()
    finish()
    assert not a.editor_shown() and not w.act_editor.isChecked()

    w.select_job("Class2D/job018/")  # Continue closes it too
    a.continue_job()
    ed.run_btn.click()
    finish()
    assert not a.editor_shown()

    a.new_job("relion.class2d")
    ed.run_btn.click()
    a.new_job("relion.class3d")  # another job opened while the first one waits for RELION's lock
    finish()
    assert a.editor_shown() and ed.variant.label == "relion.class3d"


def test_helpers():
    assert qt_file_filter("STAR files (*.star) \t Maps (*.{mrc,map})") == \
        "STAR files (*.star);;Maps (*.mrc *.map);;All files (*)"
    assert node_matches("ParticleGroupMetadata.star.relion.refine3d", 1, "ParticleGroupMetadata.star.relion", 1)
    assert not node_matches("Mask3D.mrc.relion", 1, "DensityMap.mrc", 1)
    assert node_matches("DensityMap.mrc.relion.halfmap.refine3d", 1, "DensityMap.mrc.halfmap", 1)  # any map
    tomo_half = "DensityMap.mrc.relion.tomo.halfmap.reconstruct"
    assert node_matches(tomo_half, 99, tomo_half, 99)
    assert not node_matches("DensityMap.mrc.relion.tomo.map.reconstruct", 1, tomo_half, 99)


def test_readonly_disables_changes(window, spa_project):
    w = window(spa_project, readonly=True)
    a = w.actions
    assert not a.new_menu.isEnabled()
    assert not any(x.isEnabled() for x in (a.act_clone, a.act_continue, a.act_delete, a.act_alias, a.act_finished))
    assert not a.editor.run_btn.isEnabled()


def test_clone_opens_editor_with_the_jobs_settings(window, spa_project):
    w = window(spa_project)
    w.select_job("Class3D/job009/")
    w.actions.clone_job()
    assert w.actions.editor_shown()
    js = JobStar.read(spa_project / "Class3D/job009/job.star")
    assert w.actions.editor.jobstar().options == js.options and not w.actions.editor.jobstar().is_continue


@pytest.fixture
def scratch_job(window, spa_copy, tmp_path, monkeypatch):
    """A window on a project whose Class3D/job009 used <tmp>/ssd/savanna_123, and the job.stars its
    submissions would add (relion_pipeliner not run)."""
    w = window(spa_copy)
    job = "Class3D/job009/"
    js = JobStar.read(spa_copy / job / "job.star")
    js.options["scratch_dir"] = str(tmp_path / "ssd" / "savanna_123")
    js.write(spa_copy / job / "job.star")
    w.select_job(job)
    added = []

    class FakePipeliner:
        available = True

        def add_job(self, js, alias, refs=None):
            added.append(js)
            raise RuntimeError("not run")
    monkeypatch.setattr(w.actions, "pipeliner", FakePipeliner())
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **k: None)

    def submit():
        w.actions._on_submit(w.actions.editor.jobstar(), "", False)
        w.actions.wait()
        return added[-1].options["scratch_dir"]
    return w, submit


def test_scratch_directory_is_used_as_typed_by_default(scratch_job, tmp_path):
    w, submit = scratch_job
    assert not P.setting(w.settings, "scratch_own")  # as RELION's own GUI
    w.actions.clone_job()
    assert w.actions.editor.values()["scratch_dir"] == str(tmp_path / "ssd" / "savanna_123")
    assert submit() == str(tmp_path / "ssd" / "savanna_123")


def test_each_new_job_gets_a_scratch_directory_of_its_own(scratch_job, tmp_path, monkeypatch):
    from cryo_savanna.project.scratch import base_dir

    w, submit = scratch_job

    def fake_exec(dlg):
        dlg.scratch_own.setChecked(True)
        dlg.accept()
        return P.PreferencesDialog.Accepted
    monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
    w.show_preferences()
    a = w.actions
    a.clone_job()
    assert a.editor.values()["scratch_dir"] == str(tmp_path / "ssd")  # the directory chosen, not the job's own
    dirs = [submit() for _ in range(5)]
    assert all(base_dir(d) == str(tmp_path / "ssd") and d != str(tmp_path / "ssd" / "savanna_123") for d in dirs)
    a.editor.widgets["scratch_dir"].set_value("")
    assert submit() == ""  # no scratch: left alone


def test_alias_mark_delete_undelete_from_the_menus(window, qtbot, spa_copy, monkeypatch):
    w = window(spa_copy)
    a = w.actions
    w.select_job("Class2D/job006/")
    monkeypatch.setattr(QInputDialog, "getText", lambda *x, **k: ("my classes", True))
    a.set_alias()
    a.wait()
    assert Pipeline.read(spa_copy).processes["Class2D/job006/"].alias == "Class2D/my_classes/"
    assert (spa_copy / "Class2D/my_classes").is_symlink()

    w.select_job("LocalRes/job017/")
    a.mark(failed=False)
    a.wait()
    assert Pipeline.read(spa_copy).processes["LocalRes/job017/"].status == "Succeeded"

    w.select_job("Class2D/job018/")
    a.delete(recursive=False)
    a.wait()
    assert "Class2D/job018/" not in Pipeline.read(spa_copy).processes
    w.load_now()
    qtbot.waitUntil(a.act_undelete.isEnabled)  # Trash/ is looked at in the background
    monkeypatch.setattr(QInputDialog, "getItem", lambda *x, **k: ("Class2D/job018", True))
    a.undelete()
    a.wait()
    assert "Class2D/job018/" in Pipeline.read(spa_copy).processes



def test_delete_warns_when_trash_holds_an_earlier_copy(window, spa_copy, monkeypatch):
    """After an overwrite, Trash/ has the job's earlier version: deleting the job again replaces it."""
    w = window(spa_copy)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    w.select_job("Class2D/job018/")
    w.actions.delete(recursive=False)
    assert "earlier copy" not in asked[-1]
    (spa_copy / "Trash/Class2D/job018").mkdir(parents=True)  # as an overwrite leaves it
    w.actions.delete(recursive=False)
    assert "earlier copy of Class2D/job018" in asked[-1] and "for good" in asked[-1]
    assert "Class2D/job018/" in Pipeline.read(spa_copy).processes  # answered No

# ---------------------------------------------------------------------------- with relion_pipeliner

@needs_relion
def test_new_job_from_the_editor_runs_and_continues(window, qtbot, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    script = project / "ext.sh"
    script.write_text(EXTERNAL_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    w = window(project)
    a = w.actions
    a.new_job("relion.external")
    ed = a.editor
    ed.widgets["fn_exe"].set_value(str(script))
    ed.widgets["param1_label"].set_value("sleep")
    ed.widgets["param1_value"].set_value("1")
    ed.alias.setText("from editor")
    expected = ed.jobstar()
    ed.run_btn.click()
    a.wait()
    p = Pipeline.read(project)
    assert list(p.processes) == ["External/job001/"]
    assert p.processes["External/job001/"].alias == "External/from_editor/"
    # the job.star relion_pipeliner wrote is the one the editor made
    assert (project / "External/job001/job.star").read_text() == expected.to_text()
    deadline = time.time() + 60
    while Pipeline.read(project).processes["External/job001/"].status != "Succeeded" and time.time() < deadline:
        time.sleep(0.5)
    assert Pipeline.read(project).processes["External/job001/"].status == "Succeeded"
    # remembered for the next new job, like RELION's GUI
    a.new_job("relion.external")
    assert ed.widgets["param1_label"].value() == "sleep"

    w.load_now()
    w.select_job("External/job001/")
    a.continue_job()
    ed.run_btn.click()
    a.wait()
    assert JobStar.read(project / "External/job001/job.star").is_continue
    deadline = time.time() + 60
    while Pipeline.read(project).processes["External/job001/"].status == "Running" and time.time() < deadline:
        time.sleep(0.5)

    # overwrite: same number and alias, the old job in Trash/
    w.load_now()
    w.select_job("External/job001/")
    a.overwrite_job()
    assert ed.alias.text() == "from_editor" and ed.run_btn.text() == "Overwrite now"
    ed.widgets["param1_value"].set_value("0")
    ed.schedule_btn.click()
    a.wait()
    p = Pipeline.read(project)
    assert list(p.processes) == ["External/job001/"] and p.job_counter == 2
    assert (p.processes["External/job001/"].status, p.processes["External/job001/"].alias) == \
        ("Scheduled", "External/from_editor/")
    assert JobStar.read(project / "External/job001/job.star").options["param1_value"] == "0"
    assert (project / "Trash/External/job001/job.star").exists()
    shutil.rmtree(project / ".cryo-savanna", ignore_errors=True)


@needs_relion
def test_cli_tool_job_runs_in_the_pipeline(window, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    (project / "in.mrc").write_bytes(b"\0" * 1024)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "relion_image_handler"  # stands in for the real program
    fake.write_text(f"#!{sys.executable}\nimport sys\na = sys.argv[1:]\n"
                    "open(a[a.index('--o') + 1], 'wb').write(bytes([8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0]))\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    w = window(project)
    a = w.actions
    a.new_cli_job("relion_image_handler")
    ed = a.editor
    ed.widgets["--i"].set_value("in.mrc")
    ed.widgets["--o"].set_value("flipped.mrc")
    ed.widgets["--invert_hand"].set_value("Yes")
    ed.run_btn.click()
    a.wait()
    deadline = time.time() + 60
    while Pipeline.read(project).processes["External/job001/"].status not in ("Succeeded", "Failed") \
            and time.time() < deadline:
        time.sleep(0.5)
    p = Pipeline.read(project)
    assert p.processes["External/job001/"].status == "Succeeded", (project / "External/job001/run.err").read_text()
    assert p.processes["External/job001/"].outputs == ["External/job001/flipped.mrc"]
    assert p.nodes["External/job001/flipped.mrc"].type_label == "DensityMap.mrc"

    # the Options tab shows the tool's form, not External's fields
    w.load_now()
    w.select_job("External/job001/")
    rows = _option_rows(w.info.options)
    assert rows["--i"].text(1) == "in.mrc" and rows["--i"].font(1).bold()
    assert rows["--o"].text(1) == "flipped.mrc" and rows["--invert_hand"].text(1) == "Yes"
    assert "External executable" not in rows

    # clone opens the tool's form again, with its values
    a.clone_job()
    assert ed.is_cli and ed.variant.program == "relion_image_handler"
    assert (ed.widgets["--i"].value(), ed.widgets["--o"].value(), ed.widgets["--invert_hand"].value()) == \
        ("in.mrc", "flipped.mrc", "Yes")
    shutil.rmtree(project / ".cryo-savanna", ignore_errors=True)


def _option_rows(tree) -> dict:
    return {tree.topLevelItem(i).child(j).text(0): tree.topLevelItem(i).child(j)
            for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())}


def test_options_tab_reads_a_cli_tools_help_off_the_gui_thread(qtbot, tmp_path):
    """With the tool's --help still to read, the Options tab says so, then shows the tool's form."""
    from cryo_savanna.jobdefs.cli_tools import CliCatalog
    from cryo_savanna.ui.job_info import JobInfoPanel

    fake = tmp_path / "relion_image_handler"  # prints no usable --help: the bundled text is used
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    flags = FlagSource(which=lambda p: str(fake) if p == "relion_image_handler" else None)
    defs = JobDefs.load()
    v = CliCatalog(defs, ENV, FLAGS).variant("relion_image_handler")
    values = v.default_values(ENV)
    values.update({"--i": "in.mrc", "--angpix": "1.06"})
    js = v.make_jobstar(values, False, "External/job020/", None)
    panel = JobInfoPanel(defs, ENV, {})
    qtbot.addWidget(panel)
    panel.cli = CliCatalog(defs, ENV, flags)
    panel._jobstar = js
    panel._fill_options()
    assert "Reading the options of relion_image_handler" in panel.options.topLevelItem(0).text(0)
    panel.wait_flags()
    rows = _option_rows(panel.options)
    assert rows["--i"].text(1) == "in.mrc" and rows["--angpix"].text(1) == "1.06"
    assert rows["--angpix"].font(1).bold() and not rows["--invert_hand"].font(1).bold()
    panel.only_changed.setChecked(True)
    assert set(_option_rows(panel.options)) == {"--i", "--angpix"}


def test_browsed_files_keep_the_projects_symlinks(qtbot, tmp_path):
    """A Movies/ link to the raw data stays Movies/... (as in RELION's GUI), not the raw data's own path."""
    project, raw = tmp_path / "proj", tmp_path / "raw"
    project.mkdir()
    raw.mkdir()
    (raw / "m1.tiff").touch()
    (project / "Movies").symlink_to(raw)
    e = JobEditor(ENV, {}, flag_source=FLAGS)
    qtbot.addWidget(e)
    e.set_project(project, None)
    assert e.relative(str(project / "Movies" / "m1.tiff")) == "Movies/m1.tiff"
    assert e.relative(str(project.resolve() / "Class2D" / ".." / "x.star")) == "x.star"
    assert e.relative(str(raw / "m1.tiff")) == str(raw / "m1.tiff")  # outside the project: as chosen


def test_the_submission_script_preview(editor, tmp_path):
    template = tmp_path / "slurm.sh"
    template.write_text("#!/bin/bash\n#SBATCH --job-name=XXXnameXXX\n#SBATCH --ntasks=XXXmpinodesXXX\n"
                        "#SBATCH --cpus-per-task=XXXthreadsXXX\nsrun XXXcommandXXX\n")
    editor.new_job(DEFS.variants["relion.class3d"])
    link = editor.submit_preview_link
    assert link is not None and not link.isVisibleTo(editor)  # not sent to a queue
    editor.widgets["do_queue"].set_value("Yes")
    editor._update_states()
    assert link.isVisibleTo(editor) or not editor.tabs.isVisible()
    for k, v in (("qsubscript", str(template)), ("nr_mpi", "3"), ("nr_threads", "4"), ("use_gpu", "Yes")):
        editor.widgets[k].set_value(v)
    shown = []
    dlg = editor.preview_submission(shown.append)
    assert shown == [dlg] and dlg.tasks.wait()
    assert "#SBATCH --ntasks=3" in dlg.result.script and "--cpus-per-task=4" in dlg.result.script
    assert f"--job-name={editor.job_dir()}" in dlg.result.script
    assert any("asks for none" in p for p in dlg.result.problems)  # GPUs used, none asked for
    assert "Check these" in dlg.checks.text()
