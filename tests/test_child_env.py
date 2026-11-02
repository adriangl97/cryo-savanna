# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Programs cryo-savanna starts get the user's environment, not cryo-savanna's own Qt/Python settings."""

import os
import stat
import subprocess
import time
from pathlib import Path

from cryo_savanna.env import child
from cryo_savanna.env.child import external_environ, override
from cryo_savanna.viewers.external import launch_chimerax

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_values_are_given_back():
    env = {"QT_XCB_GL_INTEGRATION": "none", "CRYO_SAVANNA_ORIG_QT_XCB_GL_INTEGRATION": "__unset__",
           "PYTHONPATH": "/opt/cryo-savanna:/home/me/lib", "CRYO_SAVANNA_ORIG_PYTHONPATH": "/home/me/lib", "PATH": "/bin"}
    out = external_environ(env, prefix="/envs/cryo-savanna")
    assert "QT_XCB_GL_INTEGRATION" not in out  # ChimeraX/napari need OpenGL
    assert out["PYTHONPATH"] == "/home/me/lib" and out["PATH"] == "/bin"
    assert not any(k.startswith("CRYO_SAVANNA_ORIG_") for k in out)
    # the user had set it themselves: they keep their value
    env = {"QT_XCB_GL_INTEGRATION": "xcb_egl", "CRYO_SAVANNA_ORIG_QT_XCB_GL_INTEGRATION": "xcb_egl",
           "CRYO_SAVANNA_ORIG_PYTHONPATH": "__unset__", "PYTHONPATH": "/opt/cryo-savanna"}
    out = external_environ(env, prefix="/envs/cryo-savanna")
    assert out["QT_XCB_GL_INTEGRATION"] == "xcb_egl" and "PYTHONPATH" not in out


def test_values_set_by_the_app_are_given_back(monkeypatch):
    monkeypatch.setattr(child, "_originals", {})
    env = {"PATH": "/bin"}
    override("QT_XCB_GL_INTEGRATION", "none", env)
    assert env["QT_XCB_GL_INTEGRATION"] == "none"
    assert "QT_XCB_GL_INTEGRATION" not in external_environ(env, prefix="/x")
    env2 = {"QT_XCB_GL_INTEGRATION": "glx"}
    override("QT_XCB_GL_INTEGRATION", "none", env2)  # already set by the user: left alone
    assert env2["QT_XCB_GL_INTEGRATION"] == "glx"


def test_own_qt_plugin_paths_are_dropped(tmp_path):
    own = tmp_path / "env"
    env = {"QT_PLUGIN_PATH": str(own / "lib/qt6/plugins"), "QT_QPA_PLATFORM_PLUGIN_PATH": "/usr/lib/qt/plugins"}
    out = external_environ(env, prefix=str(own))
    assert "QT_PLUGIN_PATH" not in out and out["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/usr/lib/qt/plugins"


def test_launcher_records_the_originals(tmp_path):
    """bin/cryo-savanna saves the user's values before setting its own."""
    fake_env = tmp_path / "env"
    (fake_env / "bin").mkdir(parents=True)
    py = fake_env / "bin" / "python"
    py.write_text("#!/bin/sh\nenv\n")
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    env = {"PATH": os.environ["PATH"], "CRYO_SAVANNA_ENV": str(fake_env), "PYTHONPATH": "/home/me/lib"}
    out = subprocess.run([str(ROOT / "bin" / "cryo-savanna")], env=env, capture_output=True, text=True,
                         check=True).stdout
    seen = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    assert seen["QT_XCB_GL_INTEGRATION"] == "none"
    assert seen["CRYO_SAVANNA_ORIG_QT_XCB_GL_INTEGRATION"] == "__unset__"
    assert seen["CRYO_SAVANNA_ORIG_PYTHONPATH"] == "/home/me/lib"
    restored = external_environ(seen, prefix=str(fake_env))
    assert "QT_XCB_GL_INTEGRATION" not in restored and restored["PYTHONPATH"] == "/home/me/lib"
    short = subprocess.run([str(ROOT / "bin" / "savanna")], env=env, capture_output=True, text=True, check=True)
    assert short.stdout == out  # bin/savanna is the same launcher


def test_chimerax_gets_the_users_environment(tmp_path, monkeypatch):
    """The bug: ChimeraX inherited QT_XCB_GL_INTEGRATION=none and showed an empty, undrawn window."""
    fake = tmp_path / "chimerax"
    dump = tmp_path / "env.txt"
    fake.write_text(f"#!/bin/sh\nenv > {dump}.tmp\necho \"$1\" >> {dump}.tmp\nmv {dump}.tmp {dump}\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("QT_XCB_GL_INTEGRATION", "none")
    monkeypatch.setenv("CRYO_SAVANNA_ORIG_QT_XCB_GL_INTEGRATION", "__unset__")
    script = launch_chimerax(str(fake), 'open "/p/map.mrc"\n')
    for _ in range(250):
        if dump.exists():
            break
        time.sleep(0.02)
    text = dump.read_text()
    assert "QT_XCB_GL_INTEGRATION" not in text and "CRYO_SAVANNA_ORIG_" not in text
    assert text.strip().endswith(str(script)) and script.read_text() == 'open "/p/map.mrc"\n'


def test_pipeliner_runs_relion_with_the_users_environment(monkeypatch, tmp_path):
    from cryo_savanna.project.pipeliner import Pipeliner
    monkeypatch.setenv("QT_XCB_GL_INTEGRATION", "none")
    monkeypatch.setenv("CRYO_SAVANNA_ORIG_QT_XCB_GL_INTEGRATION", "__unset__")
    assert "QT_XCB_GL_INTEGRATION" not in Pipeliner(tmp_path).environ
    assert Pipeliner(tmp_path, environ={"A": "1"}).environ == {"A": "1"}  # an explicit one is used as given


def test_qt_api_from_qtpy_does_not_reach_relions_python_tools(tmp_path):
    """qtpy writes QT_API=pyside6 into cryo-savanna's environment; RELION's napari tools (their own qtpy,
    another binding) then stopped with "No Qt bindings could be found"."""
    import sys
    code = (
        "import os, json\n"
        "import cryo_savanna\n"
        "import qtpy\n"
        "os.environ['QT_SOMETHING_NEW'] = '1'\n"
        "from cryo_savanna.env.child import external_environ\n"
        "e = external_environ()\n"
        "print(json.dumps({'inside': os.environ.get('QT_API'), 'child': e.get('QT_API'),"
        " 'new': e.get('QT_SOMETHING_NEW'), 'kept': e.get('QT_SCALE_FACTOR')}))\n"
    )
    import json
    base = {k: v for k, v in os.environ.items() if k != "QT_API"}
    base["QT_SCALE_FACTOR"] = "2"  # the user's own Qt setting: kept
    base["PYTHONPATH"] = str(ROOT)
    out = json.loads(subprocess.run([sys.executable, "-c", code], env=base, capture_output=True, text=True,
                                    check=True).stdout)
    assert out == {"inside": "pyside6", "child": None, "new": None, "kept": "2"}
    base["QT_API"] = "pyqt5"  # the user chose a binding: children keep it
    out = json.loads(subprocess.run([sys.executable, "-c", code], env=base, capture_output=True, text=True,
                                    check=True).stdout)
    assert out["child"] == "pyqt5"
