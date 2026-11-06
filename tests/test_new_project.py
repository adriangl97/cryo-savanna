# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import shutil
import stat
import sys
from pathlib import Path

import pytest

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.project.new_project import NewProjectError, create_project, is_project
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.relion_consts import GUI_PROJECTDIR, LOCK_DIR, TMP_RUNFILES


def _env(bin_dir: Path) -> RelionEnv:
    return RelionEnv({"PATH": str(bin_dir)})


def _fake_relion(bin_dir: Path, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "relion"
    exe.write_text(f"#!{sys.executable}\n{body}")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


# as RELION's GUI with --do_projdir: .gui_projectdir and an empty default_pipeline.star (RELION 4.0's output)
RELION_DO_PROJDIR = '''import sys, pathlib
assert "--do_projdir" in sys.argv and "--idle" in sys.argv, sys.argv
pathlib.Path(".gui_projectdir").touch()
pathlib.Path("default_pipeline.star").write_text(
    "\\n# version 30001\\n\\ndata_pipeline_general\\n\\n_rlnPipeLineJobCounter                       1\\n")
'''


def _check_project(d: Path) -> None:
    assert is_project(d)
    assert (d / GUI_PROJECTDIR).exists() and (d / TMP_RUNFILES).is_dir()
    assert not (d / LOCK_DIR).exists()
    p = Pipeline.read(d)
    assert p.processes == {} and p.job_counter == 1


def test_project_is_made_by_the_installed_relion(tmp_path):
    exe = _fake_relion(tmp_path / "bin", RELION_DO_PROJDIR)
    d = tmp_path / "proj"
    d.mkdir()
    how = create_project(d, env=_env(tmp_path / "bin"))
    assert str(exe) in how and "written by" not in how
    _check_project(d)


def test_project_is_written_here_without_relion(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "Movies").mkdir()  # whatever is there already stays
    how = create_project(d, env=_env(tmp_path / "empty_bin"))
    assert "not found" in how
    _check_project(d)
    assert (d / "Movies").is_dir()


def test_a_failing_relion_falls_back(tmp_path):
    _fake_relion(tmp_path / "bin", "import sys\nsys.exit('cannot open display')\n")
    d = tmp_path / "proj"
    d.mkdir()
    how = create_project(d, env=_env(tmp_path / "bin"))
    assert "exited with 1" in how
    _check_project(d)


def test_an_existing_project_is_left_alone(tmp_path, spa_project):
    before = (spa_project / "default_pipeline.star").read_bytes()
    assert create_project(spa_project, env=_env(tmp_path)) == "already a RELION project"
    assert (spa_project / "default_pipeline.star").read_bytes() == before


def test_unwritable_directory_is_refused(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    try:
        with pytest.raises(NewProjectError, match="not writable"):
            create_project(d, env=_env(tmp_path))
    finally:
        d.chmod(0o755)


@pytest.mark.skipif(shutil.which("relion") is None, reason="needs RELION's GUI on PATH and a display")
def test_with_the_real_relion(tmp_path):  # pragma: no cover - only where RELION is installed
    import os
    if not os.environ.get("CRYO_SAVANNA_TEST_RELION_GUI"):
        pytest.skip("set CRYO_SAVANNA_TEST_RELION_GUI=1 to open RELION's window for a second")
    d = tmp_path / "proj"
    d.mkdir()
    how = create_project(d)
    assert "made by" in how
    _check_project(d)


# -- the startup question --------------------------------------------------------------

qt = pytest.importorskip("qtpy.QtWidgets")


def _run(start: Path, answers, picks=(), readonly=False, create=None):
    from cryo_savanna.ui import start_project as sp
    answers, picks, asked, errors = list(answers), list(picks), [], []

    def ask(folder, ro):
        asked.append((folder, ro))
        return answers.pop(0)

    def make(folder):
        (folder / "default_pipeline.star").write_text("data_pipeline_general\n_rlnPipeLineJobCounter 1\n")
        return "test"

    got = sp.choose_project(start, readonly, ask=ask, pick_folder=lambda f: picks.pop(0),
                            create=create or make, show_error=errors.append)
    return got, asked, errors


def test_a_project_opens_without_a_question(qapp, spa_project):
    got, asked, _ = _run(spa_project, [])
    assert got == spa_project and asked == []


def test_yes_starts_a_project_here(qapp, tmp_path):
    got, asked, _ = _run(tmp_path, ["new"])
    assert got == tmp_path and is_project(tmp_path) and len(asked) == 1


def test_no_close_returns_nothing_and_writes_nothing(qapp, tmp_path):
    got, _, _ = _run(tmp_path, ["quit"])
    assert got is None and list(tmp_path.iterdir()) == []


def test_open_a_different_folder(qapp, tmp_path, spa_project):
    got, _, _ = _run(tmp_path, ["other"], picks=[spa_project])
    assert got == spa_project.resolve() and not is_project(tmp_path)


def test_a_different_folder_that_is_no_project_is_asked_about(qapp, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    got, asked, _ = _run(tmp_path, ["other", "new"], picks=[other])
    assert got == other.resolve() and [a[0] for a in asked] == [tmp_path, other.resolve()]
    assert is_project(other) and not is_project(tmp_path)


def test_cancelling_the_folder_dialog_asks_again(qapp, tmp_path):
    got, asked, _ = _run(tmp_path, ["other", "quit"], picks=[None])
    assert got is None and len(asked) == 2


def test_a_failed_start_is_shown_and_asked_again(qapp, tmp_path):
    def fail(folder):
        raise NewProjectError("disk full")
    got, asked, errors = _run(tmp_path, ["new", "quit"], create=fail)
    assert got is None and errors == ["disk full"] and len(asked) == 2


def test_readonly_never_starts_a_project(qapp, tmp_path):
    got, asked, _ = _run(tmp_path, ["new", "quit"], readonly=True)
    assert got is None and not is_project(tmp_path) and asked[0][1] is True


def test_the_question_offers_the_three_answers(qapp, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.ui import start_project as sp
    seen = {}

    def fake_exec(box):
        seen["labels"] = [b.text() for b in box.buttons()]
        seen["default"] = box.defaultButton().text()
        box.buttons()[0].click()

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    assert sp.ask(tmp_path, False) in (sp.NEW, sp.OTHER, sp.QUIT)
    assert set(seen["labels"]) == {"Yes, start a new project here", "No, open a different folder…",
                                   "No, close cryo-savanna"}
    assert seen["default"] == "Yes, start a new project here"
    sp.ask(tmp_path, True)
    assert "Yes, start a new project here" not in seen["labels"] and len(seen["labels"]) == 2
