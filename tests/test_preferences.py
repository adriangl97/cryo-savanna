# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Preferences, Help → Keyboard shortcuts / About, and the command line's defaults."""

import os

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings  # noqa: E402

from cryo_savanna.env.relion_env import RelionEnv  # noqa: E402
from cryo_savanna.jobdefs.flags import FlagSource  # noqa: E402
from cryo_savanna.project.slurm import SlurmClient  # noqa: E402
from cryo_savanna.ui import preferences as P  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402


def make(qtbot, project, settings, **kw):
    w = MainWindow(project, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None), settings=settings,
                   slurm=SlurmClient(which=lambda p: None), **kw)
    qtbot.addWidget(w)
    return w


def test_preferences_are_used_unless_the_command_line_says_otherwise(qtbot, tmp_path, spa_project):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s.setValue("poll_s", 7.5)
    s.setValue("cache_mb", 100)
    w = make(qtbot, spa_project, s)
    assert w.poll_s == 7.5
    assert w.info.cache.max_disk == 100 * 1024 * 1024
    assert make(qtbot, spa_project, s, poll_s=1.0).poll_s == 1.0  # --poll 1


def test_preferences_dialog_applies_at_once(qtbot, tmp_path, spa_project, monkeypatch):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = make(qtbot, spa_project, s)

    def fake_exec(dlg):
        dlg.poll.setValue(4)
        dlg.cache.setValue(50)
        dlg.accept()
        return P.PreferencesDialog.Accepted
    monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
    w.show_preferences()
    assert w.poll_s == 4 and w.poll_timer.interval() == 4000
    assert w.info.cache.max_disk == 50 * 1024 * 1024
    assert P.setting(s, "poll_s") == 4.0



def test_theme_preference(qtbot, tmp_path, spa_project, monkeypatch):
    import pyqtgraph as pg
    from qtpy.QtGui import QPalette
    from qtpy.QtWidgets import QApplication

    from cryo_savanna.ui import style
    app = QApplication.instance()
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s.setValue("theme", "nonsense")
    assert P.setting(s, "theme") == "system"
    w = make(qtbot, spa_project, s)
    w.load_now()
    plot = pg.PlotWidget()  # an open plot follows the theme
    qtbot.addWidget(plot)
    plot.setTitle("FSC")

    def choose(theme):
        def fake_exec(dlg):
            dlg.theme.setCurrentIndex(dlg.theme.findData(theme))
            dlg.accept()
            return P.PreferencesDialog.Accepted
        monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
        w.show_preferences()
        app.processEvents()  # the widgets take the new palette from a posted event

    try:
        choose("dark")
        assert P.setting(s, "theme") == "dark"
        assert style.is_dark(app.palette()) and style.is_dark(w.palette())
        text = app.palette().color(QPalette.Text)
        assert plot.backgroundBrush().color() == app.palette().color(QPalette.Base)
        assert plot.getPlotItem().getAxis("left").pen().color() == text
        assert text.name() in plot.getPlotItem().titleLabel.item.toHtml()
        assert style.excluded_row_color(app.palette()).lightness() < 128
        assert w.graph.edges and all(e.pen().color().rgb() == text.rgb() for e in w.graph.edges if not e.highlighted)
        style.apply_theme(app, "system")
        w2 = make(qtbot, spa_project, s)  # and at the next start
        app.processEvents()
        assert style.is_dark(w2.palette())
        choose("light")
        assert not style.is_dark(app.palette())
        assert plot.backgroundBrush().color() == app.palette().color(QPalette.Base)
    finally:
        style.apply_theme(app, "system")



def test_job_editor_opens_in_its_own_window_or_docked(qtbot, tmp_path, spa_project, monkeypatch):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = make(qtbot, spa_project, s)
    w.load_now()
    a = w.actions
    assert P.setting(s, "editor") == "window"  # the default
    w.select_job("Class3D/job009/")
    a.clone_job()
    assert a.editor.window() is a.editor_window and a.editor_window.isWindow()
    assert a.editor_shown() and w.act_editor.isChecked() and a.dock.isHidden()
    a.editor.closed.emit()  # its Close button
    assert not a.editor_shown() and not w.act_editor.isChecked()
    w.act_editor.trigger()
    assert a.editor_window.isVisible()

    def choose(place):
        def fake_exec(dlg):
            dlg.editor.setCurrentIndex(dlg.editor.findData(place))
            dlg.accept()
            return P.PreferencesDialog.Accepted
        monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
        w.show_preferences()

    choose("docked")  # the open editor moves into the main window
    assert P.setting(s, "editor") == "docked"
    assert a.editor.window() is w and a.dock.widget() is a.editor
    assert a.editor_shown() and a.editor_window.isHidden() and w.act_editor.isChecked()
    w.act_editor.trigger()
    assert a.dock.isHidden() and not w.act_editor.isChecked()
    w2 = make(qtbot, spa_project, s)  # and at the next start
    assert w2.actions.editor_place == "docked" and not w2.actions.editor_shown()
    choose("window")
    assert a.editor.window() is a.editor_window and not a.editor_shown()
    s.setValue("editor", "floating")  # not a choice
    assert P.setting(s, "editor") == "window"


def test_chimerax_preference_and_the_shell(monkeypatch, tmp_path):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    monkeypatch.delenv("CRYO_SAVANNA_CHIMERAX", raising=False)
    monkeypatch.setattr(P, "_set_here", None)
    s.setValue("chimerax", "/opt/chimerax/bin/ChimeraX")
    P.apply_environment(s)
    assert os.environ["CRYO_SAVANNA_CHIMERAX"] == "/opt/chimerax/bin/ChimeraX"
    s.setValue("chimerax", "")
    P.apply_environment(s)
    assert "CRYO_SAVANNA_CHIMERAX" not in os.environ
    monkeypatch.setenv("CRYO_SAVANNA_CHIMERAX", "/shell/chimerax")  # set in the shell: it wins
    s.setValue("chimerax", "/opt/other")
    P.apply_environment(s)
    assert os.environ["CRYO_SAVANNA_CHIMERAX"] == "/shell/chimerax"


def test_scratch_directory_preference_and_the_shell(qtbot, tmp_path, spa_project, monkeypatch):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s.setValue("scratch_dir", "/ssd/me")
    env = RelionEnv({})
    w = MainWindow(spa_project, env=env, flag_source=FlagSource(which=lambda p: None), settings=s,
                   slurm=SlurmClient(which=lambda p: None))
    qtbot.addWidget(w)
    class3d = w.defs.variant("relion.class3d")
    assert class3d.default_values(env)["scratch_dir"] == "/ssd/me"

    def fake_exec(dlg):
        dlg.scratch.setText("  ")
        dlg.accept()
        return P.PreferencesDialog.Accepted
    monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
    w.show_preferences()
    assert class3d.default_values(env)["scratch_dir"] == ""  # RELION's own default again

    shell = RelionEnv({"RELION_SCRATCH_DIR": "/shell/scratch"})
    s.setValue("scratch_dir", "/ssd/me")
    P.apply_relion_defaults(s, shell)
    assert class3d.default_values(shell)["scratch_dir"] == "/shell/scratch"  # set in the shell: it wins
    assert "/shell/scratch" in P.PreferencesDialog(s, None, shell).scratch.placeholderText()


def test_job_editor_shows_the_scratch_directory_of_the_preferences(qtbot, tmp_path, spa_copy, monkeypatch):
    from cryo_savanna.project.jobstar import JobStar

    remembered = JobStar.read(spa_copy / "Class3D/job009/job.star")  # RELION's .gui file: the last job's settings
    remembered.options["scratch_dir"] = ""
    remembered.options["nr_classes"] = "7"
    remembered.write(spa_copy / ".gui_class3djob.star")
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = make(qtbot, spa_copy, s)
    ed = w.actions.editor

    def set_scratch(path):
        def fake_exec(dlg):
            dlg.scratch.setText(path)
            dlg.accept()
            return P.PreferencesDialog.Accepted
        monkeypatch.setattr(P.PreferencesDialog, "exec", fake_exec)
        w.show_preferences()

    w.actions.new_job("relion.class3d")
    assert ed.values()["scratch_dir"] == ""
    set_scratch("/ssd/me")  # the editor is open: it follows
    assert ed.values()["scratch_dir"] == "/ssd/me" and ed.values()["nr_classes"] == "7"
    w.actions.new_job("relion.class3d")  # the preference wins over the remembered empty one
    assert ed.values()["scratch_dir"] == "/ssd/me" and ed.values()["nr_classes"] == "7"
    ed.widgets["scratch_dir"].set_value("/typed/here")
    set_scratch("/ssd/other")  # typed in the editor: kept
    assert ed.values()["scratch_dir"] == "/typed/here"
    set_scratch("")  # no preference: RELION's remembered settings, as in its own GUI
    w.actions.new_job("relion.class3d")
    assert ed.values()["scratch_dir"] == ""

    shell = RelionEnv({"RELION_SCRATCH_DIR": "/shell/scratch"})
    remembered.options["scratch_dir"] = "/last/job"
    remembered.write(spa_copy / ".gui_class3djob.star")
    w2 = MainWindow(spa_copy, env=shell, flag_source=FlagSource(which=lambda p: None), settings=s,
                    slurm=SlurmClient(which=lambda p: None))
    qtbot.addWidget(w2)
    w2.actions.new_job("relion.class3d")  # set in the shell: RELION's own rules (the remembered value)
    assert w2.actions.editor.values()["scratch_dir"] == "/last/job"


def test_wipe_savanna_folders_shows_them_and_asks_first(qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QDialog, QMessageBox

    base = tmp_path / "ssd"
    for name in ("savanna_010", "savanna_011", "keep_me"):
        (base / name).mkdir(parents=True)
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = P.PreferencesDialog(s, None, RelionEnv({}))
    qtbot.addWidget(dlg)
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: told.append(text))
    shown = []

    def answer(result):
        def fake_exec(confirm):
            shown.append(confirm.paths.toPlainText())
            assert "will crash" in confirm.warning.text()
            assert confirm.delete.text().startswith("Delete ") and not confirm.delete.isDefault()
            return result
        monkeypatch.setattr(P.WipeScratchDialog, "exec", fake_exec)

    dlg._wipe_scratch()  # no scratch directory set
    assert "No scratch directory" in told[-1] and not shown

    dlg.scratch.setText(str(base))
    answer(QDialog.Rejected)
    dlg._wipe_scratch()
    assert shown[-1].splitlines() == [str(base / "savanna_010"), str(base / "savanna_011")]
    assert (base / "savanna_010").exists()  # cancelled

    answer(QDialog.Accepted)
    dlg._wipe_scratch()
    assert sorted(x.name for x in base.iterdir()) == ["keep_me"]
    assert "Deleted 2 folders" in told[-1]
    dlg._wipe_scratch()
    assert "No savanna_NNN folders" in told[-1]

    shell_base = tmp_path / "shell"
    (shell_base / "savanna_020").mkdir(parents=True)
    (base / "savanna_030").mkdir()
    shell = P.PreferencesDialog(s, None, RelionEnv({"RELION_SCRATCH_DIR": str(shell_base)}))
    qtbot.addWidget(shell)
    shell.scratch.setText(str(base))
    answer(QDialog.Rejected)
    shell._wipe_scratch()  # the directory jobs get: the shell's
    assert shown[-1] == str(shell_base / "savanna_020")


def test_shortcuts_and_about(qtbot, tmp_path, spa_project):
    w = make(qtbot, spa_project, QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    from qtpy.QtGui import QKeySequence
    html = P.shortcuts_html(w)

    def native(keys):  # as the system writes them (⌘D on macOS)
        return QKeySequence(keys).toString(QKeySequence.NativeText)
    for text in ("Clone", native("Ctrl+D"), "Refresh", "Preferences", native("Ctrl+E"), native("Ctrl+Shift+P"),
                 "Exclude tilt images"):
        assert text in html, text
    w.show_shortcuts()
    w.show_about()
    dialogs = [d for d in w.findChildren(P.TextDialog)]
    assert {d.windowTitle() for d in dialogs} == {"Keyboard shortcuts", "About cryo-savanna"}
    about = next(d for d in dialogs if d.windowTitle().startswith("About")).text.toPlainText()
    assert "RELION 5.1.1" in about and str(spa_project) in about
    w.act_editor.trigger()
    assert w.actions.editor_shown()


def test_command_line_defaults(monkeypatch):
    from cryo_savanna import app
    seen = {}
    monkeypatch.setattr(app, "run_gui", lambda project, tomo, readonly, poll, queue: seen.update(
        tomo=tomo, poll=poll, queue=queue) or 0)
    app.main(["--project", "."])
    assert seen == {"tomo": None, "poll": None, "queue": None}  # from the preferences
    app.main(["--project", ".", "--poll", "0.1", "--queue-poll", "1", "--tomo"])
    assert seen == {"tomo": True, "poll": 0.5, "queue": 5.0}
    app.main(["--project", ".", "--queue-poll", "0"])
    assert seen["queue"] == 0.0


def test_version_warning_compares_the_release():
    from cryo_savanna.env.relion_env import version_warning
    assert version_warning("5.1.1-commit-0a1b2c", "5.1.1") is None
    assert version_warning("5.1.1", "5.1.1") is None
    assert version_warning(None, "5.1.1") is None  # unknown: the status bar says relion_pipeliner is missing
    for other in ("5.1.2-commit-abc", "5.0.0", "5.1", "6.0-beta"):
        text = version_warning(other, "5.1.1")
        assert text and other in text and "5.1.1" in text, other


def test_relion_version_from_the_program(tmp_path):
    from cryo_savanna.env.relion_env import relion_version
    exe = tmp_path / "relion_pipeliner"
    exe.write_text("#!/bin/sh\necho 'RELION version: 5.2.0-commit-123abc '\necho 'Precision: BASE=double'\n")
    exe.chmod(0o755)
    assert relion_version(str(exe)) == "5.2.0-commit-123abc"
    assert relion_version(None) is None
    assert relion_version(str(tmp_path / "missing")) is None


def test_other_relion_version_is_warned_once(qtbot, tmp_path, spa_project):
    from qtpy.QtWidgets import QMessageBox
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = make(qtbot, spa_project, s)
    w.load_now()

    def boxes():
        return [b for b in w.findChildren(QMessageBox) if b.objectName() == "version_warning"]

    w.set_relion_version("5.1.1-commit-0a1b2c")
    assert w.version_warning is None and not boxes() and "⚠" not in w.status_state.text()
    w.set_relion_version("5.2.0-commit-123abc")
    assert "⚠ RELION 5.2.0-commit-123abc (made for 5.1.1)" in w.status_state.text()
    assert "deleting and undeleting" in w.status_state.toolTip()
    assert len(boxes()) == 1 and "5.2.0-commit-123abc" in boxes()[0].text()
    w.load_now()  # the mark stays when the project is read again
    assert "⚠ RELION 5.2.0" in w.status_state.text()
    w2 = make(qtbot, spa_project, s)  # next start with the same RELION: marked, but no dialog again
    w2.load_now()
    w2.set_relion_version("5.2.0-commit-123abc")
    assert "⚠ RELION 5.2.0" in w2.status_state.text()
    assert not [b for b in w2.findChildren(QMessageBox) if b.objectName() == "version_warning"]


def test_print_tree_warns_about_another_relion(tmp_path, spa_project, monkeypatch, capsys):
    from cryo_savanna import app
    exe = tmp_path / "bin" / "relion_pipeliner"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\necho 'RELION version: 5.0.1-commit-0'\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    assert app.main(["--project", str(spa_project), "--print-tree"]) == 0
    out = capsys.readouterr().out
    assert "(RELION 5.0.1-commit-0)" in out and "WARNING: cryo-savanna was made for RELION 5.1.1" in out


def test_print_tree_shows_the_tomography_menu_for_a_tomography_project(tomo_project, capsys):
    from cryo_savanna import app
    assert app.main(["--project", str(tomo_project), "--print-tree"]) == 0
    assert "job menu (tomo)" in capsys.readouterr().out


def test_slow_work_runs_in_a_thread_with_the_window_alive(qtbot):
    """run_busy: the result (or the error) of work done in a thread, with the event loop running
    meanwhile; when_done: a value shown once it is there."""
    import threading
    import time

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QWidget

    from cryo_savanna.ui.busy import run_busy, when_done
    main = threading.get_ident()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)

    def slow():
        time.sleep(0.6)
        return threading.get_ident()

    assert run_busy(slow, "Working…") != main and len(ticks) > 5  # the event loop ran meanwhile
    assert run_busy(lambda: 7) == 7  # quick work: at once
    with pytest.raises(ValueError, match="no"):
        run_busy(lambda: (_ for _ in ()).throw(ValueError("no")))
    owner, got = QWidget(), []
    qtbot.addWidget(owner)
    when_done(owner, lambda: 1, got.append)
    assert got == [1]
    when_done(owner, lambda: time.sleep(0.4) or 2, got.append)
    assert got == [1]
    qtbot.waitUntil(lambda: got == [1, 2], timeout=3000)
    timer.stop()
