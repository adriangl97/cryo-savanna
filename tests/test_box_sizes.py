# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Good box sizes (jobdefs/box_sizes.py, EMAN2's list) and the job editor's warning next to a box size."""

import pytest

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import box_sizes as B
from cryo_savanna.jobdefs.cli_tools import CliCatalog
from cryo_savanna.jobdefs.flags import FlagSource
from cryo_savanna.jobdefs.model import JobDefs

DEFS = JobDefs.load()
ENV = RelionEnv({})


def test_the_list_as_copied():
    """Sorted, each once, even, and made of small primes only: a typo in the copy would show here."""
    def rest(n):
        for p in (2, 3, 5, 7, 11, 13):
            while n % p == 0:
                n //= p
        return n
    assert list(B.ALL) == sorted(set(B.ALL)) and len(B.GOOD) == 71 and len(B.GOOD_LARGE) == 84
    assert B.GOOD[0] == 24 and B.GOOD[-1] == 1024 and B.GOOD_LARGE[-1] == 4096
    assert all(n % 2 == 0 and rest(n) == 1 for n in B.ALL)


def test_neighbours_and_warnings():
    assert B.is_good(128) and not B.is_good(130) and B.warning(128) == ""
    assert B.neighbours(130) == (128, 132) and B.neighbours(122) == (120, 128)
    assert B.neighbours(10) == (None, 24) and B.neighbours(5000) == (4096, None)
    w = B.warning(130)
    assert "130 px is not a good box size" in w and "128 px and 132 px" in w and "less well" not in w
    assert "1024 px and 1050 px (less well tested)" in B.warning(1030)
    assert "smallest good box size (24 px)" in B.warning(16)
    assert B.warning(5000) == "" and B.warning(None) == ""  # beyond the list: nothing known


def test_sizes_typed():
    assert B.as_size("130") == 130 and B.as_size(" 130.0 ") == 130
    assert B.as_size("-1") is None and B.as_size("0") is None and B.as_size("") is None
    assert B.as_size("abc") is None and B.as_size("130.5") is None
    # what int() cannot take: no warning, and no error while it is being typed
    assert B.as_size("inf") is None and B.as_size("nan") is None and B.as_size("1e999") is None


def test_every_box_option_is_in_its_form():
    cli = CliCatalog(DEFS, ENV, FlagSource(which=lambda p: None))
    for form, keys in B.BOX_OPTIONS.items():
        if form.startswith("cli:"):
            program = form[4:]
            v = cli.variant(program, program.startswith("relion_tomo"))
        else:
            v = DEFS.variant(form)
        assert v is not None, form
        assert B.box_options(v.key) == keys
        missing = [k for k in keys if k not in v.options]
        assert not missing, (form, missing)
    assert B.box_options("relion.extract:tomo") == B.BOX_OPTIONS["relion.extract"]
    assert B.box_options("relion.class3d") == ()
    assert B.box_options("relion.ctffind:tomo") == ("box",)
    assert B.box_options("relion.ctfrefinetomo:tomo") == B.box_options("relion.framealigntomo") == ("box_size",)


@pytest.fixture
def editor(qtbot, spa_project):
    pytest.importorskip("pytestqt")
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.snapshot import job_type_names
    from cryo_savanna.ui.job_editor import JobEditor
    e = JobEditor(ENV, job_type_names(DEFS), flag_source=FlagSource(which=lambda p: None))
    qtbot.addWidget(e)
    e.set_project(spa_project, Pipeline.read(spa_project))
    return e


def test_editor_warns_under_a_box_size(editor):
    editor.new_job(DEFS.variant("relion.extract"))
    assert editor.box_warning("extract_size") == ""  # 128, RELION's default
    editor.widgets["extract_size"].set_value("130")
    w = editor.box_warning("extract_size")
    assert "128 px and 132 px" in w and B.SOURCE in w
    editor.widgets["extract_size"].set_value("256")
    assert editor.box_warning("extract_size") == ""


def test_only_box_sizes_in_use(editor):
    editor.new_job(DEFS.variant("relion.extract"))
    editor.widgets["rescale"].set_value("130")
    assert editor.box_warning("rescale") == ""  # "Rescale particles?" is No: not used
    editor.widgets["do_rescale"].set_value("Yes")
    assert "128 px and 132 px" in editor.box_warning("rescale")
    editor.search.setText("diameter")  # the row hidden by the search: its warning too
    assert editor.box_warning("rescale") == ""
    editor.search.clear()
    assert editor.box_warning("rescale")


def test_fft_and_estimation_boxes(editor):
    editor.new_job(DEFS.variant("relion.ctffind", True))
    assert editor.box_warning("box") == ""  # 512, RELION's default
    editor.widgets["box"].set_value("500")
    assert "480 px and 512 px" in editor.box_warning("box")
    for form in ("relion.ctfrefinetomo", "relion.framealigntomo"):
        editor.new_job(DEFS.variant(form, True))
        assert editor.box_warning("box_size") == ""  # 128
        editor.widgets["box_size"].set_value("150")
        assert "140 px and 168 px" in editor.box_warning("box_size"), form


def test_run_and_schedule_ask_first(editor, monkeypatch, spa_project):
    from qtpy.QtWidgets import QMessageBox
    from cryo_savanna.project.jobstar import JobStar
    asked, answer, sent = [], [QMessageBox.No], []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or answer[0])
    editor.submit.connect(lambda js, alias, schedule_only: sent.append(schedule_only))
    editor.new_job(DEFS.variant("relion.extract"))
    editor.run_btn.click()  # a good size: not asked
    assert asked == [] and sent == [False]
    editor.widgets["extract_size"].set_value("310")
    editor.run_btn.click()
    assert sent == [False] and "Particle box size (pix): 310 px" in asked[-1] and asked[-1].endswith("Run the job anyway?")
    editor.schedule_btn.click()
    assert sent == [False] and asked[-1].endswith("Schedule the job anyway?")
    answer[0] = QMessageBox.Yes
    editor.schedule_btn.click()
    assert sent == [False, True]
    editor.search.setText("invert")  # hidden by the search: still asked about
    editor.run_btn.click()
    assert len(asked) == 4
    js = JobStar.read(spa_project / "Extract/job005/job.star")
    js.options["extract_size"] = "310"
    editor.continue_job("Extract/job005/", js, DEFS.variant("relion.extract"))
    editor.run_btn.click()  # continuing: the size cannot be changed, so not asked
    assert len(asked) == 4 and sent[-1] is False


def test_other_forms_and_the_help(editor):
    editor.new_job(DEFS.variant("relion.polish"))
    assert editor.box_warning("extract_size") == ""  # -1: as the input particles
    editor.new_job(DEFS.variant("relion.class3d"))
    assert editor.box_warnings == []  # no box size on this form
    cli = CliCatalog(DEFS, ENV, FlagSource(which=lambda p: None)).variant("relion_image_handler")
    editor.new_job(cli)
    editor.widgets["--new_box"].set_value("250")
    assert "240 px and 256 px" in editor.box_warning("--new_box")
    editor.show_help("--new_box")
    assert B.SOURCE in editor.help.toHtml() and "Particle Box Size" in editor.help.toPlainText()
