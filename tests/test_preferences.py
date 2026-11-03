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
