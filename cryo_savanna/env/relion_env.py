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
"""The RELION_* environment of the loaded RELION module.

RELION's GUI reads its queue settings, defaults and executable paths from environment
variables when it initialises a job (RelionJob::initialise in src/pipeline_jobs.cpp). This
module reads the same variables and parses them the way RELION does, so cryo-savanna shows the
same defaults as the original GUI started from the same shell.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Mapping

# Compiled-in fallbacks (pipeline_jobs.h)
DEFAULT_QSUB_EXTRA_COUNT = 2
DEFAULT_WARNING_LOCAL_MPI = 32
DEFAULT_MPIRUN = "mpirun"

# Variables RELION reads with `const char x = atoi(...)`: the value is truncated to a signed char.
_CHAR_ATOI_VARS = {"RELION_QSUB_NRMPI", "RELION_QSUB_NRTHREADS", "RELION_QSUB_EXTRA_COUNT"}

_RELEASE = re.compile(r"\d+(\.\d+)*")


def relion_version(exe: str | None, timeout: float = 20) -> str | None:
    """The version a RELION program reports with --version ("5.1.1-commit-0a1b2c"), or None."""
    if exe is None:
        return None
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELION_")}
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("RELION version:"):
            return line.split(":", 1)[1].strip()
    return None


def version_warning(installed: str | None, supported: str) -> str | None:
    """Why the loaded RELION may not match cryo-savanna, or None if it is the release cryo-savanna was made for.

    Only the release number counts ("5.1.1-commit-0a1b2c" is 5.1.1). An unknown version (no
    relion_pipeliner, or no answer to --version) is no warning: the status bar says so already.
    """
    if installed is None:
        return None
    m = _RELEASE.match(installed)
    if m and m.group() == supported:
        return None
    return (f"cryo-savanna was made for RELION {supported}, but the loaded RELION is {installed}.\n\n"
            f"Adding and running jobs still goes through this RELION's relion_pipeliner, but the job forms "
            f"show the options of RELION {supported}, and cryo-savanna itself writes a few changes the way "
            f"RELION {supported} does: aliases, marking jobs finished or failed, deleting and undeleting "
            f"jobs, and the files of Select and Exclude tilt images. Check them before relying on them, "
            f"or use RELION's own GUI for these.")


class RelionEnvError(ValueError):
    pass


def float_to_string(f: float) -> str:
    """RELION's floatToString(F) with default width/precision: a float32 printed with %g."""
    f32 = struct.unpack("f", struct.pack("f", float(f)))[0]
    return "%g" % f32


def _atoi(s: str) -> int:
    m = re.match(r"\s*([+-]?\d+)", s)
    return int(m.group(1)) if m else 0


def _to_signed_char(v: int) -> int:
    return ((v + 128) % 256) - 128


def _text_to_integer(name: str, s: str) -> int:
    m = re.match(r"\s*([+-]?\d+)", s)
    if not m:
        raise RelionEnvError(f"{name}={s!r} is not an integer (RELION would refuse to start)")
    return int(m.group(1))


def _text_to_bool(name: str, s: str) -> bool:
    low = s.strip().lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    raise RelionEnvError(f"{name}={s!r} is not true/false/yes/no (RELION would refuse to start)")


@dataclass(frozen=True)
class QsubExtra:
    index: int
    label: str  # RELION_QSUB_EXTRA<i>: text shown in the GUI
    default: str  # RELION_QSUB_EXTRA<i>_DEFAULT
    help: str  # RELION_QSUB_EXTRA<i>_HELP

    @property
    def key(self) -> str:
        return f"qsub_extra{self.index}"


class RelionEnv:
    """Snapshot of the RELION_* environment variables."""

    def __init__(self, environ: Mapping[str, str] | None = None):
        src = os.environ if environ is None else environ
        self.vars: dict[str, str] = {k: v for k, v in src.items() if k.startswith("RELION_")}
        self._path = src.get("PATH", "")

    def get(self, name: str) -> str | None:
        return self.vars.get(name)

    # -- typed accessors (same parsing as RELION) ------------------------------------------
    def integer(self, name: str) -> int | None:
        s = self.vars.get(name)
        if s is None:
            return None
        if name in _CHAR_ATOI_VARS:
            return _to_signed_char(_atoi(s))
        return _text_to_integer(name, s)

    def boolean(self, name: str) -> bool | None:
        s = self.vars.get(name)
        return None if s is None else _text_to_bool(name, s)

    @property
    def allow_change_minimum_dedicated(self) -> bool:
        v = self.integer("RELION_ALLOW_CHANGE_MINIMUM_DEDICATED")
        return True if v is None else v != 0

    @property
    def error_local_mpi(self) -> int:
        v = self.integer("RELION_ERROR_LOCAL_MPI")
        return DEFAULT_WARNING_LOCAL_MPI if v is None else v

    @property
    def mpirun(self) -> str:
        return self.vars.get("RELION_MPIRUN", DEFAULT_MPIRUN)

    def qsub_extras(self) -> list[QsubExtra]:
        """Extra queue fields, exactly as RelionJob::initialise builds them."""
        count = self.integer("RELION_QSUB_EXTRA_COUNT")
        if count is None:
            count = DEFAULT_QSUB_EXTRA_COUNT
        out = []
        for i in range(1, count + 1):
            label = self.vars.get(f"RELION_QSUB_EXTRA{i}")
            if label is None:
                continue
            help_ = self.vars.get(
                f"RELION_QSUB_EXTRA{i}_HELP",
                f"Extra option to pass to the qsub template script. Any occurrences of XXXextra{i}XXX "
                "will be changed by this value.",
            )
            out.append(QsubExtra(i, label, self.vars.get(f"RELION_QSUB_EXTRA{i}_DEFAULT", ""), help_))
        return out

    # -- RELION binaries ------------------------------------------------------------------
    def which(self, program: str) -> str | None:
        return shutil.which(program, path=self._path)

    @property
    def has_relion(self) -> bool:
        return self.which("relion_pipeliner") is not None
