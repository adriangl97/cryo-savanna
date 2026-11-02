# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The environment for programs cryo-savanna starts: the user's own, without cryo-savanna's settings.

cryo-savanna switches Qt's OpenGL off for itself (``QT_XCB_GL_INTEGRATION=none``, smooth over X11),
its launcher puts cryo-savanna on ``PYTHONPATH``, and qtpy writes the Qt binding it picked into
``QT_API`` (``pyside6``). Programs it starts must not inherit any of them: ChimeraX and napari are
Qt programs that need OpenGL (with ``none`` their 3D window shows only what was behind it), they
run their own Python, and RELION's napari tools use qtpy with their own binding (given
``QT_API=pyside6`` they look only for PySide6 and stop: "No Qt bindings could be found").
Every child process gets ``external_environ()``.

The user's values are taken from, in this order: what the launcher (bin/cryo-savanna) recorded in
``CRYO_SAVANNA_ORIG_<NAME>`` before changing anything (``__unset__`` when there was none); what
``override()`` recorded; the environment as it was when cryo-savanna was imported (this module is
imported by ``cryo_savanna/__init__.py``, before qtpy).
"""

from __future__ import annotations

import os
import sys

ORIG_PREFIX = "CRYO_SAVANNA_ORIG_"
UNSET = "__unset__"
OWN = ("QT_XCB_GL_INTEGRATION", "PYTHONPATH", "QT_API")  # what cryo-savanna (or qtpy) sets for itself
QT_PATHS = ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH")  # point at cryo-savanna's own Qt, if set by its env

_originals: dict[str, str | None] = {}
_startup: dict[str, str | None] = {name: os.environ.get(name) for name in OWN}  # before qtpy
_startup_qt = {k: v for k, v in os.environ.items() if k.startswith(("QT_", "PYSIDE", "SHIBOKEN", "PYQTGRAPH"))}
QT_PREFIXES = ("QT_", "PYSIDE", "SHIBOKEN", "PYQTGRAPH")


def override(name: str, value: str, environ: dict | None = None) -> None:
    """Set a variable for cryo-savanna itself (if not set already), remembering the user's value."""
    env = os.environ if environ is None else environ
    if name in env:
        return
    _originals.setdefault(name, None)
    env[name] = value


def external_environ(environ: dict | None = None, prefix: str | None = None) -> dict[str, str]:
    """A copy of the environment as the user had it, for ChimeraX, RELION and other programs."""
    env = dict(os.environ if environ is None else environ)
    prefix = sys.prefix if prefix is None else prefix
    for name in OWN:
        key = ORIG_PREFIX + name
        if key in env:
            original: str | None = env[key]
            original = None if original == UNSET else original
        elif name in _originals:
            original = _originals[name]
        elif environ is None:  # this process's own environment: as it was before cryo-savanna ran
            original = _startup[name]
        else:
            continue
        if original is None:
            env.pop(name, None)
        else:
            env[name] = original
    for key in [k for k in env if k.startswith(ORIG_PREFIX)]:
        del env[key]
    if environ is None:  # any other Qt setting made inside cryo-savanna (by Qt or a library) is undone too
        for key in [k for k in env if k.startswith(QT_PREFIXES) and k not in OWN]:
            if key not in _startup_qt:
                del env[key]
            else:
                env[key] = _startup_qt[key]
    real_prefix = os.path.realpath(prefix)
    for name in QT_PATHS:  # plugins of cryo-savanna's Qt would not load in another program's Qt
        value = env.get(name)
        if value and all(os.path.realpath(p).startswith(real_prefix) for p in value.split(os.pathsep) if p):
            del env[name]
    return env
