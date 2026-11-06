# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Starting a new RELION project in a directory, as RELION's GUI does.

RELION's GUI, started in a directory without ``.gui_projectdir``, asks "Do you want to start a new
project here?". On yes it creates ``.gui_projectdir`` and ``.TMP_runfiles/``, and then writes an
empty ``default_pipeline.star`` (GuiMainWindow's constructor, RELION 5.0). The installed RELION does
it here: ``relion --do_projdir`` makes the project without asking and ``--idle 1`` closes its window
again after a second. Without a ``relion`` on PATH, or if it fails, the same files are written here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from cryo_savanna.env.child import external_environ
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.relion_consts import GUI_PROJECTDIR, TMP_RUNFILES

PIPELINE_FILE = "default_pipeline.star"


class NewProjectError(RuntimeError):
    pass


def is_project(directory: str | os.PathLike) -> bool:
    """A RELION project: the directory has a default_pipeline.star."""
    return (Path(directory) / PIPELINE_FILE).is_file()


def create_project(directory: str | os.PathLike, env: RelionEnv | None = None, timeout_s: float = 60) -> str:
    """Make ``directory`` a new, empty RELION project. Returns how it was made, for the log."""
    directory = Path(directory)
    if is_project(directory):
        return "already a RELION project"
    if not directory.is_dir():
        raise NewProjectError(f"{directory} is not a directory")
    if not os.access(directory, os.W_OK | os.X_OK):
        raise NewProjectError(f"{directory} is not writable")
    env = env or RelionEnv()
    relion = env.which("relion")
    how = ""
    if relion:
        try:
            r = subprocess.run([relion, "--do_projdir", "--idle", "1", "--refresh", "1"], cwd=directory,
                               env=external_environ(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout_s)
            if is_project(directory):
                how = f"made by {relion}"
            else:
                how = f"{relion} exited with {r.returncode} without making {PIPELINE_FILE}; written by cryo-savanna"
        except (OSError, subprocess.TimeoutExpired) as e:
            how = f"{relion} failed ({e}); written by cryo-savanna"
    else:
        how = "relion not found on PATH; written by cryo-savanna"
    try:
        (directory / GUI_PROJECTDIR).touch()
        (directory / TMP_RUNFILES).mkdir(exist_ok=True)  # only RELION's interactive "Yes" makes it
        if not is_project(directory):
            with PipelineLock(directory, message="cryo-savanna new project") as lock:
                Pipeline().write(directory, lock=lock)
    except OSError as e:
        raise NewProjectError(f"could not start a project in {directory}: {e}") from e
    return how
