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
"""Command-line flags of the program behind a job's "Additional arguments:" box.

RELION appends that box to one command of the job (RelionJob::getCommands* in pipeline_jobs.cpp);
``program_for`` follows the same branches. The flags come from that program's --help, run from the
loaded RELION and parsed here; without RELION on PATH, the 5.1.1 texts bundled in jobdefs/help/ are
used (tools/bundle_help.py). Only the serial program is asked: the flags of the _mpi one are the same
for job use, and an _mpi binary started outside mpirun can hang on a cluster login node.

RELION's IOParser uses the last value when a flag is given twice, and the extra arguments come last,
so a flag here can also override a value the job's options set.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources

from cryo_savanna.env.child import external_environ
from cryo_savanna.jobdefs.cli_catalog import FLAG_TOOLS
from cryo_savanna.jobdefs.model import DEFAULT_VERSION

# RELION C++ programs that receive a job's additional arguments (their --help can be parsed).
JOB_PROGRAMS = {
    "relion_import", "relion_run_motioncorr", "relion_run_ctffind", "relion_manualpick", "relion_autopick",
    "relion_find_amyloid", "relion_preprocess", "relion_star_handler",
    "relion_class_ranker", "relion_display", "relion_refine", "relion_mask_create", "relion_particle_subtract",
    "relion_postprocess", "relion_motion_refine", "relion_ctf_refine", "relion_tomo_import_coordinates",
    "relion_align_tiltseries", "relion_tomo_reconstruct_tomogram", "relion_tomo_subtomo", "relion_tomo_refine_ctf",
    "relion_tomo_align", "relion_tomo_reconstruct_particle",
}
KNOWN_PROGRAMS = JOB_PROGRAMS | set(FLAG_TOOLS)  # programs whose --help is read (and bundled)
NO_LIVE_HELP = {"relion_manualpick", "relion_display"}  # GUI programs: use the bundled text
HELP_RETRY_ARGS = {"relion_run_motioncorr": ["--use_own"]}  # stops before printing its help otherwise
HIDDEN_FLAGS = {"--version", "--theme", "--list_themes", "--pipeline_control"}

_HEADER = "+++ RELION: command line arguments"
_SECTION = re.compile(r"^=+\s*(.*?)\s*=+\s*$")
_FLAG = re.compile(r"^\s*(--\S+)(?:\s+\((.*?)\))?\s+:\s?(.*)$")


@dataclass(frozen=True)
class Flag:
    name: str  # "--iter"
    default: str | None  # None: no default (RELION calls it compulsory); "false" for on/off switches
    help: str
    section: str

    @property
    def is_switch(self) -> bool:
        return self.default == "false"


@dataclass
class ProgramFlags:
    program: str
    flags: list[Flag]
    source: str  # where the list came from, shown in the panel

    def get(self, name: str) -> Flag | None:
        return next((f for f in self.flags if f.name == name), None)


def parse_help(text: str) -> list[Flag]:
    """Flags in the output of a RELION program's --help (IOParser::writeUsage)."""
    flags: list[Flag] = []
    section = ""
    started = False
    for line in text.splitlines():
        if not started:
            started = line.startswith(_HEADER)
            continue
        if line.startswith("#") or not line.strip():
            continue
        m = _SECTION.match(line)
        if m:
            section = m.group(1)
            continue
        m = _FLAG.match(line)
        if m:
            name, default, usage = m.groups()
            if name not in HIDDEN_FLAGS:
                flags.append(Flag(name, default, usage.strip(), section))
        elif line.startswith(" " * 20) and flags:  # a usage text that went on to the next line
            last = flags[-1]
            flags[-1] = Flag(last.name, last.default, f"{last.help} {line.strip()}", last.section)
        else:
            break  # end of the usage (MPI or error messages follow)
    return flags


# ---------------------------------------------------------------------------- which program

def program_for(label: str, values: dict[str, str]) -> str:
    """The program a job of this type passes its additional arguments to, for these option values.

    The name of a RELION program (serial version), or a description for other tools.
    """
    def yes(k: str) -> bool:
        return values.get(k, "") == "Yes"

    simple = {
        "relion.import": "relion_import",
        "relion.motioncorr": "relion_run_motioncorr",
        "relion.ctffind": "relion_run_ctffind",
        "relion.manualpick": "relion_manualpick",
        "relion.extract": "relion_preprocess",
        "relion.class2d": "relion_refine",
        "relion.class3d": "relion_refine",
        "relion.refine3d": "relion_refine",
        "relion.initialmodel": "relion_refine",
        "relion.multibody": "relion_refine",
        "relion.maskcreate": "relion_mask_create",
        "relion.joinstar": "relion_star_handler",
        "relion.subtract": "relion_particle_subtract",
        "relion.postprocess": "relion_postprocess",
        "relion.polish": "relion_motion_refine",
        "relion.ctfrefine": "relion_ctf_refine",
        "relion.aligntiltseries": "relion_align_tiltseries",
        "relion.reconstructtomograms": "relion_tomo_reconstruct_tomogram",
        "relion.pseudosubtomo": "relion_tomo_subtomo",
        "relion.ctfrefinetomo": "relion_tomo_refine_ctf",
        "relion.framealigntomo": "relion_tomo_align",
        "relion.reconstructparticletomo": "relion_tomo_reconstruct_particle",
        "relion.denoisetomo": "relion_python_tomo_denoise",
        "relion.picktomo": "relion_python_tomo_pick",
        "relion.excludetilts": "relion_python_tomo_exclude_tilt_images",
        "dynamight": "DynaMight",
        "modelangelo": "ModelAngelo",
        "relion.external": "your program",
    }
    if label in simple:
        return simple[label]
    if label == "relion.autopick":
        return "relion_find_amyloid" if yes("do_amyloid") else "relion_autopick"
    if label == "relion.select":
        if yes("do_filaments"):
            return "relion_filament_selection"
        if yes("do_remove_duplicates") or yes("do_select_values") or yes("do_discard") or yes("do_split"):
            return "relion_star_handler"
        return "relion_class_ranker" if yes("do_class_ranker") else "relion_display"
    if label == "relion.localres":
        return "relion_postprocess" if yes("do_relion_locres") else "ResMap"
    if label == "relion.importtomo":
        return "relion_tomo_import_coordinates" if yes("do_coords") else "relion_python_tomo_import"
    return ""


# ---------------------------------------------------------------------------- flag lists

class FlagSource:
    """--help of the loaded RELION's programs, or the bundled texts. Cached per program binary."""

    def __init__(self, version: str = DEFAULT_VERSION, which=shutil.which, timeout: float = 15):
        self.version = version
        self.which = which
        self.timeout = timeout
        self._cache: dict[tuple, ProgramFlags | None] = {}

    def _key(self, program: str) -> tuple:
        path = None if program in NO_LIVE_HELP else self.which(program)
        try:
            return (program, path, os.stat(path).st_mtime_ns if path else None)
        except OSError:
            return (program, None, None)

    def needs_run(self, program: str) -> bool:
        """Whether ``flags`` would start the program (worth doing off the GUI thread)."""
        if program not in KNOWN_PROGRAMS:
            return False
        key = self._key(program)
        return key[1] is not None and key not in self._cache

    def flags(self, program: str) -> ProgramFlags | None:
        if program not in KNOWN_PROGRAMS:
            return None
        key = self._key(program)
        if key not in self._cache:
            path = key[1]
            self._cache[key] = (self._live(program, path) if path else None) or self._bundled(program)
        return self._cache[key]

    def _live(self, program: str, path: str) -> ProgramFlags | None:
        for extra in ([], HELP_RETRY_ARGS.get(program)):
            if extra is None:
                break
            text = self._run([path, *extra, "--help"])
            flags = parse_help(text)
            if flags:
                return ProgramFlags(program, flags, f"{path} --help")
        return None

    def _run(self, cmd: list[str]) -> str:
        env = {k: v for k, v in external_environ().items() if k != "DISPLAY"}
        # some programs write files when started: keep them out of the project
        with tempfile.TemporaryDirectory(prefix="cryo-savanna_help_") as tmp:
            try:
                r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=self.timeout,
                                   stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                return ""
        return r.stdout

    def _bundled(self, program: str) -> ProgramFlags | None:
        try:
            text = resources.files("cryo_savanna.jobdefs").joinpath(f"help/{self.version}/{program}.txt").read_text()
        except (FileNotFoundError, OSError):
            return None
        flags = parse_help(text)
        return ProgramFlags(program, flags, f"RELION {self.version} --help bundled with cryo-savanna "
                                            "(load RELION to read the installed version)") if flags else None


# ---------------------------------------------------------------------------- the arguments text

_RAW_TOKEN = re.compile(r"""(?:[^\s"']+|"[^"]*"?|'[^']*'?)+""")


_PLAIN_WORD = re.compile(r"[\w@%+=:,./-]+")  # what the shell passes on as it is (shlex's safe set)
_GLOB_WORD = re.compile(r"[\w@%+=:,./*?\[\]-]+")  # the same, with * ? [ ] left for the shell


def shell_word(value: str, glob: bool = False) -> str:
    """``value`` as one word of the command RELION runs, which it runs through the shell (system()):
    a plain value as it is, and one with characters the shell acts on ($, *, ;, &, |, spaces, ...)
    quoted, so the program gets the value shown in the form. With ``glob``, * ? [ ] are left to the
    shell (a list of files typed as a pattern). A value typed with its own quotes is kept as typed."""
    if not value:
        return "''"
    if (_GLOB_WORD if glob else _PLAIN_WORD).fullmatch(value):
        return value
    if ("'" in value or '"' in value) and _RAW_TOKEN.fullmatch(value):
        try:  # its quotes must close and make it one word ("it's" is quoted as a whole instead)
            if len(shlex.split(value)) == 1:
                return value
        except ValueError:
            pass
    return shlex.quote(value)


def _unquote(raw: str) -> str:
    try:
        parts = shlex.split(raw)
    except ValueError:
        return raw
    return parts[0] if len(parts) == 1 else raw


def _is_flag(token: str) -> bool:
    return token.startswith("--") and len(token) > 2


@dataclass
class _Group:
    flag: str | None  # None: text before the first flag
    raw: list[str] = field(default_factory=list)  # the flag's value tokens as typed (quotes kept)


def _groups(text: str) -> list[_Group]:
    groups = [_Group(None)]
    for m in _RAW_TOKEN.finditer(text):
        tok = m.group(0)
        if _is_flag(tok):
            groups.append(_Group(tok))
        else:
            groups[-1].raw.append(tok)
    return groups


def _join(groups: list[_Group]) -> str:
    out = []
    for g in groups:
        out += ([g.flag] if g.flag else []) + g.raw
    return " ".join(out)


def parse_args(text: str) -> dict[str, list[str]]:
    """Flags in an additional-arguments text -> the words after each (unquoted). The last one wins."""
    return {g.flag: [_unquote(t) for t in g.raw] for g in _groups(text) if g.flag}


def set_arg(text: str, flag: str, value: str | None, switch: bool = False) -> str:
    """The text with ``flag`` set to ``value`` or removed (None). A switch takes no value.

    Only the flag and its own value (the first word after it) change; everything else stays as typed.
    A value with spaces is quoted as one argument.
    """
    groups = _groups(text)
    own = 0 if switch else 1  # how many of the words after the flag are its value
    present = [g for g in groups if g.flag == flag]
    if value is None:
        for g in present:  # the words after a removed flag stay where they were
            g.flag, g.raw = None, g.raw[own:]
    else:
        raw = [] if switch else [shell_word(value)]
        if present:
            for g in present[:-1]:
                g.flag, g.raw = None, g.raw[own:]
            present[-1].raw = raw + present[-1].raw[own:]
        else:
            groups.append(_Group(flag, raw))
    return _join(groups)


def command_flags(note: str, other_args: str = "", program: str = "") -> dict[str, str]:
    """Flags in the last command(s) a job ran (from its note.txt) -> value, without its extra arguments.

    With ``program``, only the commands that run it (or its _mpi version) count.
    """
    blocks = note.split(" ++++ with the following command(s):")
    if len(blocks) < 2:
        return {}
    body = blocks[-1].split(" ++++", 1)[0]
    extra = parse_args(other_args)
    seen: dict[str, list[str]] = {}
    for line in body.splitlines():
        if program and not re.search(rf"\b{re.escape(program)}(_mpi)?\b", line):
            continue
        for g in _groups(line):
            if g.flag:
                seen.setdefault(g.flag, []).append(_unquote(g.raw[0]) if g.raw else "")
    # a flag also in the extra arguments: the job's own value is the first one, if it has one
    return {f: (v[0] if f in extra else v[-1]) for f, v in seen.items() if f not in extra or len(v) > 1}
