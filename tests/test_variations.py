# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Variations of a job: values and combinations (project/variations.py), the dialog, and the jobs it makes
through a stand-in relion_pipeliner (tests/fixtures/fake_pipeliner.py)."""

import os

import pytest

from cryo_savanna.jobdefs.model import JobDefs, OptionDef
from cryo_savanna.project import variations as V

DEFS = JobDefs.load()


def opt(kind="any", key="k", **kw):
    return OptionDef(key=key, kind=kind, label="An option:", default="", **kw)


def test_lists_and_ranges():
    o = opt("slider", min=1, max=100)
    assert V.parse_values(o, "3, 4,5") == ["3", "4", "5"]
    assert V.parse_values(o, "2:8:2") == ["2", "4", "6", "8"]
    assert V.parse_values(o, "8:2:-3") == ["8", "5", "2"]
    assert V.parse_values(o, "0.5:1:0.25, 1") == ["0.5", "0.75", "1"]  # each once
    assert V.parse_values(o, "3\n4\n\n") == ["3", "4"]
    for bad in ("2:8:0", "8:2:1", "three"):
        with pytest.raises(V.VariationError):
            V.parse_values(o, bad)
    with pytest.raises(V.VariationError):
        V.parse_values(o, f"1:{V.MAX_JOBS + 5}:1")


def test_choices_and_text():
    assert V.parse_values(opt("boolean"), "Yes, No") == ["Yes", "No"]
    with pytest.raises(V.VariationError):
        V.parse_values(opt("boolean"), "maybe")
    radio = opt("radio", choices=("v1.0", "amy-v1.0"))
    assert V.choices(radio) == ["v1.0", "amy-v1.0"]
    with pytest.raises(V.VariationError):
        V.parse_values(radio, "v2")
    assert V.parse_values(opt("any"), "C1, D2, 2:4:1") == ["C1", "D2", "2", "3", "4"]  # text: ranges too
    assert V.parse_values(opt("inputnode"), "A/job001/x.star, B/job002/y.star") == ["A/job001/x.star", "B/job002/y.star"]
    assert V.parse_values(opt("filename"), "a:b:c") == ["a:b:c"]  # a file name, not a range


def test_combinations_and_aliases():
    vs = [V.Variation("nr_classes", ["3", "4"]), V.Variation("tau_fudge", ["2", "4"]), V.Variation("unused", [])]
    combos = V.combinations(vs)
    assert combos == [{"nr_classes": "3", "tau_fudge": "2"}, {"nr_classes": "3", "tau_fudge": "4"},
                      {"nr_classes": "4", "tau_fudge": "2"}, {"nr_classes": "4", "tau_fudge": "4"}]
    assert V.count(vs) == 4 and V.combinations([]) == [] and V.count([]) == 0
    options = {"fn_img": opt("inputnode", "fn_img")}
    assert V.alias_for("Class3D", combos[0], vs, options) == "Class3D_K3_T2"
    vs[1].tag = "tau "
    assert V.alias_for("", combos[1], vs, options) == "K3_tau4"
    assert V.alias_for("my run", {"fn_img": "Select/job009/particles.star"},
                       [V.Variation("fn_img", ["x"], "in")], options) == "my_run_inSelect_job009"
    assert V.alias_for("job", {"nr_classes": "3"}, vs, options) == "v_job_K3"  # RELION refuses "job…"
    assert V.alias_for("x", {"sym_name": "C(2)"}, [V.Variation("sym_name", ["C(2)"])], options) == "x_symC_2"
    assert V.duplicates(["a", "b", "a", "", ""]) == ["a"]


# ---------------------------------------------------------------------------- the dialog and the editor

@pytest.fixture
def window(qtbot, tmp_path, spa_copy, monkeypatch):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QMessageBox
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.ui.main_window import MainWindow
    from tests.fixtures.fake_pipeliner import install as install_fake_pipeliner
    install_fake_pipeliner(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning shown: {a[1:]}"))
    w = MainWindow(spa_copy, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    return w


def answer(rows: dict[str, str], schedule_only=True, aliases: dict[int, str] | None = None):
    """A Variations dialog answered: these options with these values, then Schedule all / Run all now."""
    def hook(dlg):
        for key, text in rows.items():
            assert dlg.add_option(key, text) is not None, key
        for row, alias in (aliases or {}).items():
            dlg.table.item(row, 0).setText(alias)
        if dlg._problem() or not dlg.schedule_btn.isEnabled():
            hook.problem = dlg.info.text()
            return False
        dlg.finish(schedule_only)
        return bool(dlg.result())
    hook.problem = ""
    return hook


def test_dialog_shows_every_combination(window):
    from cryo_savanna.ui.variations_dialog import VariationsDialog
    ed = window.actions.editor
    window.actions.new_job("relion.class2d")
    seen = []

    def hook(dlg: VariationsDialog):
        dlg.add_option("nr_classes", "50, 100")
        dlg.add_option("do_center", "Yes, No")  # a Yes/No option: its boxes
        assert dlg.add_option("nr_classes") is None  # not twice
        seen.extend([dlg.table.rowCount(), dlg.aliases(), dlg.info.text()])
        dlg.rows[0].edit.setText("50, lots")
        seen.append(dlg.schedule_btn.isEnabled())
        seen.append(dlg.info.text())
        return False
    assert not ed.open_variations(hook)
    assert seen[0] == 4 and seen[1][0] == "Class2D_K50_do_centerYes" and seen[2] == "4 jobs"
    assert seen[3] is False and "not a number" in seen[4]


def test_schedule_all_makes_one_job_per_combination(window, qtbot):
    a, ed = window.actions, window.actions.editor
    a.new_job("relion.class2d")
    ed.widgets["fn_img"].set_value("Extract/job007/particles.star")
    before = set(a.snapshot.jobs)
    assert ed.open_variations(answer({"nr_classes": "50, 100", "tau_fudge": "2:4:2"}))
    qtbot.waitUntil(lambda: "4 jobs scheduled" in ed.status.text(), timeout=20000)
    window.load_now()
    new = sorted(set(window.snapshot.jobs) - before)
    assert len(new) == 4
    from cryo_savanna.project.jobstar import JobStar
    got = {}
    for j in new:
        info = window.snapshot.jobs[j]
        o = JobStar.read(window.snapshot.project_dir / j / "job.star").options
        assert info.status == "Scheduled" and o["fn_img"] == "Extract/job007/particles.star"
        got[info.process.alias.rstrip("/").split("/")[-1]] = (o["nr_classes"], o["tau_fudge"])
    assert got == {"Class2D_K50_T2": ("50", "2"), "Class2D_K50_T4": ("50", "4"),
                   "Class2D_K100_T2": ("100", "2"), "Class2D_K100_T4": ("100", "4")}


def test_run_all_now_runs_each_in_its_own_pipeliner(window, qtbot, monkeypatch):
    a, ed = window.actions, window.actions.editor
    runs = []
    monkeypatch.setattr(a.pipeliner, "run", lambda jobs, **kw: runs.append(list(jobs)))
    a.new_job("relion.class2d")
    ed.widgets["do_queue"].set_value("Yes")  # to the queue: no question about running them all here
    assert ed.open_variations(answer({"nr_classes": "10, 20, 30"}, schedule_only=False))
    qtbot.waitUntil(lambda: len(runs) == 3, timeout=20000)
    assert all(len(r) == 1 for r in runs) and len({r[0] for r in runs}) == 3


def test_alias_clashes_are_refused_before_anything_is_added(window):
    a, ed = window.actions, window.actions.editor
    a.new_job("relion.class2d")
    before = set(a.snapshot.jobs)
    hook = answer({"nr_classes": "50, 100"}, aliases={1: "Class2D_K50"})  # the same as the first one
    assert not ed.open_variations(hook) and "twice" in hook.problem
    hook = answer({"log_diam_max": "180"}, aliases={0: "LoG_based"})  # an alias of the pipeline (AutoPick's)
    a.new_job("relion.autopick")
    assert not ed.open_variations(hook) and "not unique" in hook.problem
    hook = answer({"nr_classes": "50"}, aliases={0: "jobby"})
    a.new_job("relion.class2d")
    assert not ed.open_variations(hook) and "start with 'job'" in hook.problem
    window.load_now()
    assert set(window.snapshot.jobs) == before


def test_not_in_continue_mode_nor_for_selection_jobs(window):
    a, ed = window.actions, window.actions.editor
    window.select_job("Class2D/job018/")
    a.continue_job()
    assert not ed.variations_btn.isVisibleTo(ed) and not ed.open_variations(answer({}))
    a.new_job("relion.class2d")
    assert ed.variations_btn.isVisibleTo(ed) and ed.variations_btn.isEnabled()
    a.new_job("relion.select")
    ed.widgets["fn_model"].set_value("Class2D/job018/run_it025_optimiser.star")  # interactive class selection
    assert not ed.open_variations(answer({"do_recenter": "Yes, No"})) and "selection" in ed.status.text()


def test_a_cli_tool_job_star_names_the_directory_each_job_got(window, qtbot):
    a, ed = window.actions, window.actions.editor
    a.new_cli_job("relion_image_handler")
    ed.widgets["--o"].set_value("scaled.mrc")  # written into the job's directory
    assert ed.open_variations(answer({"--multiply_constant": "2, 4"}))
    qtbot.waitUntil(lambda: "2 jobs scheduled" in ed.status.text(), timeout=20000)
    jobs = [w.strip(".") for w in ed.status.text().split(": ", 1)[1].split(", ")]
    from cryo_savanna.project.jobstar import JobStar
    for j in jobs:
        text = JobStar.read(window.snapshot.project_dir / j / "job.star").to_text()
        assert f"{j.rstrip('/')}/scaled.mrc" in text  # its output goes to its own directory
