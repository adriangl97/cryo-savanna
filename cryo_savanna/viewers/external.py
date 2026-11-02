# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""External programs for what is better done there: ChimeraX for maps, RELION's napari viewers for tomography.

RELION's own viewer for tilt series, tomograms and particles is ``relion_python_tomo_view`` (napari),
from the loaded RELION module; RELION's GUI starts it the same way for tilt-series nodes.

ChimeraX is looked for in ``$CRYO_SAVANNA_CHIMERAX``, then on PATH (``chimerax``, ``ChimeraX``), then
in the usual macOS application folder. It gets a command script (.cxc) that opens the map at the
level shown here, the mask as a mesh, and colours the surface by local resolution.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cryo_savanna.env.child import external_environ

CHIMERAX_NAMES = ("chimerax", "ChimeraX")
TEMP_PREFIX = "cryo-savanna_"
TEMP_KEEP_S = 86400  # the programs read their script or write their log long after starting
LOCRES_PALETTE = "blue:cyan:green:yellow:red"  # the same order as the slices and surface here


def find_chimerax(environ: dict | None = None) -> str | None:
    env = os.environ if environ is None else environ
    explicit = env.get("CRYO_SAVANNA_CHIMERAX")
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else shutil.which(explicit, path=env.get("PATH"))
    for name in CHIMERAX_NAMES:
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            return found
    if sys.platform == "darwin":
        apps = sorted(glob.glob("/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX"))
        if apps:
            return apps[-1]
    return None


def chimerax_script(map_path: str, level: float | None = None, mask: str | None = None,
                    color: str | None = None, color_range: tuple[float, float] | None = None) -> str:
    """ChimeraX commands: the map as a surface, the mask as a mesh, colour by local resolution."""
    def q(p: str) -> str:
        return '"' + str(p).replace('"', '\\"') + '"'
    lines = [f"open {q(map_path)}", "volume #1 style surface" + (f" level {level:.6g}" if level is not None else "")]
    n = 1
    if mask:
        n += 1
        lines += [f"open {q(mask)}", f"volume #{n} style mesh level 0.5 color #e0303080"]
    if color:
        n += 1
        lines += [f"open {q(color)}", f"volume #{n} hide"]
        rng = f" range {color_range[0]:.3g},{color_range[1]:.3g}" if color_range else ""
        lines.append(f"color sample #1 map #{n} palette {LOCRES_PALETTE}{rng}")
    lines.append("view")
    return "\n".join(lines) + "\n"


def _temp_file(suffix: str, kind: str = "") -> tuple[int, str]:
    """A new temporary file for a program started here; this user's older ones (a day) are removed first,
    so ChimeraX scripts and napari logs do not pile up in the temporary directory."""
    tmp = tempfile.gettempdir()
    try:
        entries = list(os.scandir(tmp))
    except OSError:
        entries = []
    now = time.time()
    for e in entries:
        if not (e.name.startswith(TEMP_PREFIX) and e.name.endswith((".cxc", ".log"))):
            continue
        try:
            st = e.stat(follow_symlinks=False)
            if st.st_uid == os.getuid() and now - st.st_mtime > TEMP_KEEP_S and e.is_file(follow_symlinks=False):
                os.unlink(e.path)
        except OSError:
            pass
    return tempfile.mkstemp(prefix=TEMP_PREFIX + kind, suffix=suffix)


def launch_chimerax(exe: str, script: str, environ: dict | None = None) -> Path:
    """Start ChimeraX on its own (it outlives cryo-savanna) with the script; returns the script file.

    It gets the user's environment, not cryo-savanna's (which switches Qt's OpenGL off): see env/child.py.
    """
    fd, name = _temp_file(".cxc")
    with os.fdopen(fd, "w") as fh:
        fh.write(script)
    subprocess.Popen([exe, name], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     env=external_environ(environ))
    return Path(name)


NAPARI_VIEW = "relion_python_tomo_view"


def napari_command(kind: str, star: str, tomograms: str | None = None, environ: dict | None = None) -> list[str] | None:
    """The relion_python_tomo_view command for "tilt-series", "tomograms" or "particles"; None if not on PATH."""
    env = external_environ(environ)
    exe = shutil.which(NAPARI_VIEW, path=env.get("PATH"))
    if exe is None:
        return None
    if kind == "particles":
        cmd = [exe, "particles", "--particle-star-file", star]
        return cmd + (["--tilt-series-star-file", tomograms] if tomograms else [])
    return [exe, kind, "--tilt-series-star-file", star]


def launch_detached(cmd: list[str], cwd: str | os.PathLike, environ: dict | None = None) -> tuple[subprocess.Popen, Path]:
    """Start a viewer on its own, in the project directory, with its output in a log file."""
    fd, log = _temp_file(".log", "napari_")
    with os.fdopen(fd, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True, stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, env=external_environ(environ))
    return proc, Path(log)
