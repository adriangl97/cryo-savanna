# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The runners of the "Additional external jobs" in a project's .cryo-savanna/ folder.

RELION runs ``python3 .cryo-savanna/<runner>.py ...``: a copy of cryo-savanna's runner, so that the jobs
keep working when cryo-savanna is moved or updated. pytom_runner.py, aretomo_runner.py and mask_runner.py
import runner_common.py from beside them, so it always goes in first.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from cryo_savanna import runner_common

COMMON = str(Path(runner_common.__file__).resolve())
COMMON_IN_PROJECT = ".cryo-savanna/runner_common.py"


def install_file(src: str | os.PathLike, dst: Path) -> None:
    """Copy ``src`` to ``dst`` in one step (a job starting meanwhile sees the old file or the new one, never
    half of it); nothing when it is the same already."""
    data = Path(src).read_bytes()
    try:
        if dst.read_bytes() == data:
            return
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def install_runner(project_dir: str | os.PathLike, runner: str | os.PathLike, runner_in_project: str) -> Path:
    """Put runner_common.py, then the runner, into the project; the runner's path there."""
    install_file(COMMON, Path(project_dir) / COMMON_IN_PROJECT)
    dst = Path(project_dir) / runner_in_project
    install_file(runner, dst)
    return dst
