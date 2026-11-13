# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
"""RELION's pipeline lock protocol (PipeLine::read/write with DO_LOCK).

RELION takes the lock by creating the directory ``.relion_lock`` (mkdir is atomic, also on
NFS) and writing ``.relion_lock/lock_<name>_pipeline.star`` inside it. It releases the lock by
removing that file and the directory. Holding the same lock here makes our few direct
pipeline writes safe against the original GUI and relion_pipeliner.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

from cryo_savanna.relion_consts import LOCK_DIR


class LockError(RuntimeError):
    pass


def lock_file_name(pipeline_name: str = "default") -> str:
    # FileName::afterLastOf("/") returns the whole name when there is no slash
    base = pipeline_name.rsplit("/", 1)[-1]
    return f"lock_{base}_pipeline.star"


def is_locked(project_dir: str | os.PathLike) -> bool:
    return (Path(project_dir) / LOCK_DIR).is_dir()


class PipelineLock:
    """Context manager holding RELION's pipeline lock.

    RELION waits 3 s between attempts and gives up after 20; the defaults here match.
    """

    def __init__(
        self,
        project_dir: str | os.PathLike,
        pipeline_name: str = "default",
        message: str = "cryo-savanna",
        wait_s: float = 3.0,
        max_tries: int = 20,
    ):
        self.project_dir = Path(project_dir)
        self.dir = self.project_dir / LOCK_DIR
        self.file = self.dir / lock_file_name(pipeline_name)
        self.message = message
        self.wait_s = wait_s
        self.max_tries = max_tries
        self.held = False

    def acquire(self) -> None:
        tries = 0
        while True:
            try:
                os.mkdir(self.dir, 0o700)
                break
            except FileExistsError:
                tries += 1
                if tries > self.max_tries:
                    raise LockError(
                        f"Waited {self.max_tries * self.wait_s:.0f} s for {self.dir} to disappear. "
                        f"If no RELION process is running, remove {self.file} and {self.dir}."
                    ) from None
                time.sleep(self.wait_s)
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EPERM):
                    raise LockError(f"No write permission to {self.project_dir}; open it read-only.") from e
                if e.errno == errno.EROFS:
                    raise LockError(f"{self.project_dir} is on a read-only file system.") from e
                raise
        try:
            self.file.write_text(self.message + "\n")
        except OSError:
            os.rmdir(self.dir)
            raise
        self.held = True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        # already gone: someone removed a lock they took for stale (as LockError's message tells them to),
        # while the change it guarded went through; nothing is left to release
        try:
            self.file.unlink(missing_ok=True)
        finally:
            try:
                os.rmdir(self.dir)
            except FileNotFoundError:
                pass

    def __enter__(self) -> "PipelineLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
