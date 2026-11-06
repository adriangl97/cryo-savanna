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
from cryo_savanna.ui.flag_panel import COL_DEFAULT, COL_FLAG, COL_VALUE  # noqa: E402
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


def test_search_filters_rows(editor):
    editor.new_job(DEFS.variant("relion.class3d"))
    editor.search.setText("symmetry")
    visible = [row.keys for _, row, parts in editor.rows if not parts[0].isHidden()]
    assert ("sym_name",) in visible and ("nr_classes",) not in visible


def test_all_flags_under_additional_arguments(editor):
    editor.new_job(DEFS.variant("relion.class3d"))
    panel, box = editor.flag_panel, editor.widgets["other_args"]
    assert panel.program == "relion_refine" and len(panel.items) == 150
    # ticking a switch and typing a value write the box
    panel.items["--dont_skip_gridding"].setCheckState(COL_FLAG, Qt.Checked)
    panel.items["--pad"].setText(COL_VALUE, "1")
    assert box.value() == "--dont_skip_gridding --pad 1"
    assert panel.items["--pad"].checkState(COL_FLAG) == Qt.Checked
    assert editor.changed_options()["other_args"] == ("", "--dont_skip_gridding --pad 1")
    # typing in the box ticks the flags, and names the ones the program does not have
    box.set_value("--pad 2 my_note --nonsense")
    assert panel.items["--pad"].text(COL_VALUE) == "2"
    assert panel.items["--dont_skip_gridding"].checkState(COL_FLAG) == Qt.Unchecked
    assert not panel.notice.isHidden() and "--nonsense" in panel.notice.text()
    panel.items["--pad"].setCheckState(COL_FLAG, Qt.Unchecked)
    assert box.value() == "my_note --nonsense"
    assert editor.jobstar().options["other_args"] == "my_note --nonsense"


def test_flags_follow_the_program_and_the_search(editor):
    editor.new_job(DEFS.variant("relion.autopick"))
    assert editor.flag_panel.program == "relion_autopick"
    editor.widgets["do_amyloid"].set_value("Yes")
    assert editor.flag_panel.program == "relion_find_amyloid"
    editor.new_job(DEFS.variant("relion.external"))
    assert editor.flag_panel.flags is None and "your program" in editor.flag_panel.header.text()
    editor.new_job(DEFS.variant("relion.refine3d"))
    editor.search.setText("skip_gridding")  # a flag, not a job option
    running = [i for i in range(editor.tabs.count()) if editor.tabs.tabText(i).startswith("Running")][0]
    assert editor.tabs.isTabEnabled(running) and editor.flag_panel.isVisibleTo(editor)
    shown = [n for n, it in editor.flag_panel.items.items() if not it.isHidden()]
    assert shown == ["--dont_skip_gridding"]


def test_flags_set_by_the_job_show_its_value(editor, spa_copy):
    editor.set_project(spa_copy, Pipeline.read(spa_copy))
    note = spa_copy / "Class3D/job009/note.txt"
    note.write_text(note.read_text().replace("`which relion_class3d` --o Class3D/job009/",
                                             "`which relion_refine_mpi` --o Class3D/job009/run --iter 25 --ctf"))
    editor.new_job(DEFS.variant("relion.class3d"))  # the newest Class3D job's command
    panel = editor.flag_panel
    assert panel.job_flags_from == "Class3D/job009"
    assert panel.items["--iter"].text(COL_DEFAULT) == "job: 25" and panel.items["--ctf"].text(COL_DEFAULT) == "job: on"
    panel.items["--iter"].setCheckState(COL_FLAG, Qt.Checked)
    assert editor.widgets["other_args"].value() == "--iter 25"  # starts from the job's value, to override


def test_cli_tools_menu_and_form(window, spa_project):
    w = window(spa_project)
    a = w.actions
    assert a.cli_menu in [m.menu() for m in w.menuBar().actions()]
    categories = [m.text() for m in a.cli_menu.actions()]
    assert categories[0] == "Maps and images" and "Tomography" in categories
    a.new_cli_job("relion_image_handler")
    ed = a.editor
    assert ed.is_cli and ed.flag_panel is None and "relion_image_handler" in ed.title.text()
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

        def add_job(self, js, alias):
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

    # clone opens the tool's form again, with its values
    w.load_now()
    w.select_job("External/job001/")
    a.clone_job()
    assert ed.is_cli and ed.variant.program == "relion_image_handler"
    assert (ed.widgets["--i"].value(), ed.widgets["--o"].value(), ed.widgets["--invert_hand"].value()) == \
        ("in.mrc", "flipped.mrc", "Yes")
    shutil.rmtree(project / ".cryo-savanna", ignore_errors=True)


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
