#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The command-line flags each RELION job type writes itself, and the job options behind each one.

Reads the RelionJob::getCommands*Job functions of src/pipeline_jobs.cpp (and the helper
getTomoInputCommmand they call): every flag in a string added to a command (``command += " --iter " + ...``)
is a flag the job writes. Its options are the ``joboptions["..."]`` named in that statement and in the
conditions it sits under, also through local variables (``float val = joboptions["range_tilt"]...;
if (val > 0) command += ...``). ``direct`` keeps the statement and its innermost condition only: what the
job editor names as the option that sets the flag. A flag with no option at all is one the job always writes
(``--o``, ``--norm --scale`` of the classifications).

The job editor's "Advanced parameters" tab leaves these flags out: they are set on the job's other tabs, or
by the job itself.

Usage (after extract.py, whose job definitions it reads to know each job type's options):
    python tools/extract_jobdefs/command_flags.py /path/to/relion-5.1.1 \
        -o cryo_savanna/jobdefs/command-flags-5.1.1.json --version 5.1.1

(the path may also be src/pipeline_jobs.cpp itself)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from extract import NOTICE, ExtractError, _skip_literal, extract_function  # noqa: E402

from cryo_savanna.relion_consts import JOB_TYPES  # noqa: E402

# getCommands<name>Job -> the job type label (RelionJob::getCommands dispatches on PROC_* in this order)
BUILDERS = {
    "Import": "relion.import", "Motioncorr": "relion.motioncorr", "Ctffind": "relion.ctffind",
    "Manualpick": "relion.manualpick", "Autopick": "relion.autopick", "Extract": "relion.extract",
    "Select": "relion.select", "Class2D": "relion.class2d", "Inimodel": "relion.initialmodel",
    "Class3D": "relion.class3d", "Autorefine": "relion.refine3d", "MultiBody": "relion.multibody",
    "Maskcreate": "relion.maskcreate", "Joinstar": "relion.joinstar", "Subtract": "relion.subtract",
    "Postprocess": "relion.postprocess", "Localres": "relion.localres", "DynaMight": "dynamight",
    "ModelAngelo": "modelangelo", "Motionrefine": "relion.polish", "Ctfrefine": "relion.ctfrefine",
    "External": "relion.external", "TomoImport": "relion.importtomo",
    "TomoAlignTiltSeries": "relion.aligntiltseries", "TomoReconstructTomograms": "relion.reconstructtomograms",
    "TomoDenoiseTomograms": "relion.denoisetomo", "TomoPickTomograms": "relion.picktomo",
    "TomoExcludeTiltImages": "relion.excludetilts", "TomoSubtomo": "relion.pseudosubtomo",
    "TomoCtfRefine": "relion.ctfrefinetomo", "TomoAlign": "relion.framealigntomo",
    "TomoReconPart": "relion.reconstructparticletomo",
}
HELPERS = ("getTomoInputCommmand",)  # (RELION's spelling) helpers that add to the command they are given

_FLAG = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")
_OPTION = re.compile(r'joboptions\s*\[\s*"([^"]+)"\s*\]')
_COMMAND_VAR = re.compile(r"^command\w*$")


# ---------------------------------------------------------------------------- tokens

@dataclass
class Tok:
    kind: str  # "str", "id", "punct"
    text: str


def tokens(body: str) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(body):
        c = body[i]
        j = _skip_literal(body, i)
        if j != i:
            if c == '"':
                out.append(Tok("str", body[i + 1: j - 1]))
            elif c == "'":
                out.append(Tok("punct", body[i:j]))
            i = j  # (a comment: skipped)
            continue
        if c.isspace():
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < len(body) and (body[j].isalnum() or body[j] == "_"):
                j += 1
            out.append(Tok("id", body[i:j]))
            i = j
            continue
        two = body[i:i + 2]
        if two in ("+=", "==", "!=", "<=", ">=", "&&", "||", "::", "->", "<<", ">>"):
            out.append(Tok("punct", two))
            i += 2
            continue
        out.append(Tok("punct", c))
        i += 1
    return out


def _text(toks: list[Tok]) -> str:
    return " ".join(f'"{t.text}"' if t.kind == "str" else t.text for t in toks)


# ---------------------------------------------------------------------------- statements

@dataclass
class Use:
    direct: set[str] = field(default_factory=set)
    all: set[str] = field(default_factory=set)
    needs: set[str] | None = None  # a helper's flag: written only for jobs with these options (see assemble)


class Builder:
    """Walks one function body: statements, the conditions they sit under, and the flags they add."""

    def __init__(self, toks: list[Tok], helpers: dict[str, dict[str, Use]] | None = None):
        self.toks = toks
        self.i = 0
        self.vars: dict[str, set[str]] = {}  # local variable -> the options it was computed from
        self.flags: dict[str, Use] = {}
        self.helpers = helpers or {}

    # -- helpers --------------------------------------------------------------------------------
    def peek(self, k: int = 0) -> Tok | None:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def is_(self, text: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.kind != "str" and t.text == text

    def refs(self, toks: list[Tok]) -> set[str]:
        out = set(_OPTION.findall(_text(toks)))
        for t in toks:
            if t.kind == "id" and t.text in self.vars:
                out |= self.vars[t.text]
        return out

    def paren(self) -> list[Tok]:
        """The tokens inside the parenthesis starting here (moves past it)."""
        assert self.is_("("), _text(self.toks[self.i:self.i + 10])
        depth, start = 0, self.i
        while self.i < len(self.toks):
            t = self.toks[self.i]
            if t.kind == "punct" and t.text == "(":
                depth += 1
            elif t.kind == "punct" and t.text == ")":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return self.toks[start + 1: self.i - 1]
            self.i += 1
        raise ExtractError("unbalanced parentheses")

    # -- walking --------------------------------------------------------------------------------
    def run(self) -> dict[str, Use]:
        assert self.is_("{")
        self.i += 1
        self.block([])
        return self.flags

    def block(self, conds: list[set[str]]) -> None:
        while self.peek() is not None and not self.is_("}"):
            self.statement(conds)
        self.i += 1  # the closing brace

    def body(self, conds: list[set[str]]) -> None:
        if self.is_("{"):
            self.i += 1
            self.block(conds)
        else:
            self.statement(conds)

    def statement(self, conds: list[set[str]]) -> None:
        if self.is_("if"):
            self.i += 1
            chain = [self.refs(self.paren())]
            self.body(conds + [chain[0]])
            while self.is_("else"):
                if self.is_("if", 1):
                    self.i += 2
                    r = self.refs(self.paren())
                    self.body(conds + [set().union(*chain) | r])  # under this test, and the ones before failing
                    chain.append(r)
                else:
                    self.i += 1
                    self.body(conds + [set().union(*chain)])
                    break
            return
        if self.is_("for") or self.is_("while"):
            self.i += 1
            self.body(conds + [self.refs(self.paren())])
            return
        if self.is_("{"):
            self.i += 1
            self.block(conds)
            return
        start, depth = self.i, 0
        while self.i < len(self.toks):
            t = self.toks[self.i]
            if t.kind == "punct" and t.text in "([{":
                depth += 1
            elif t.kind == "punct" and t.text in ")]}":
                depth -= 1
            elif t.kind == "punct" and t.text == ";" and depth == 0:
                break
            self.i += 1
        self.simple(self.toks[start:self.i], conds)
        self.i += 1  # the semicolon

    def simple(self, stmt: list[Tok], conds: list[set[str]]) -> None:
        if not stmt:
            return
        assigned, op, k = self._assigned(stmt)
        own = self.refs(stmt)
        if assigned is not None and not _COMMAND_VAR.match(assigned):  # float val = joboptions[...]...
            value = self.refs(stmt[k + 1:])  # (val = val / 3.: what val was computed from)
            self.vars[assigned] = (self.vars.get(assigned, set()) if op == "+=" else set()) | value
        direct = own | (conds[-1] if conds else set())
        every = own.union(*conds) if conds else set(own)
        names = {t.text for t in stmt if t.kind == "id"}
        for helper in names & self.helpers.keys():
            for flag, use in self.helpers[helper].items():
                self._add(flag, use.direct | direct, use.all | every, needs=use.direct)
        if assigned is None or not _COMMAND_VAR.match(assigned):
            return
        for t in stmt:
            if t.kind == "str":
                for flag in _FLAG.findall(t.text):
                    self._add(flag, direct, every)

    @staticmethod
    def _assigned(stmt: list[Tok]) -> tuple[str | None, str, int]:
        """The variable a statement assigns to (``command += ...``, ``std::string command2 = ...``), how
        (``=`` or ``+=``), and where that operator is."""
        for k, t in enumerate(stmt):
            if t.kind == "punct" and t.text in ("=", "+="):
                before = [x for x in stmt[:k] if x.kind == "id"]
                return (before[-1].text if before else None), t.text, k
            if t.kind == "punct" and t.text in ("(", "["):
                break
        return None, "", -1

    def _add(self, flag: str, direct: set[str], every: set[str], needs: set[str] | None = None) -> None:
        use = self.flags.get(flag)
        if use is None:
            use = self.flags[flag] = Use(needs=set(needs) if needs is not None else None)
        elif needs is None:
            use.needs = None  # also written by the job's own code
        elif use.needs is not None:
            use.needs &= needs
        use.direct |= direct
        use.all |= every


# ---------------------------------------------------------------------------- the file

def _body(function: str) -> list[Tok]:
    return tokens(function[function.index("{"):])


def scan(src: str) -> dict[str, dict[str, Use]]:
    """Job type label -> flag -> the options behind it."""
    helpers: dict[str, dict[str, Use]] = {}
    for name in HELPERS:
        found = extract_function(src, rf"^std::string RelionJob::{name}\(")
        if len(found) != 1:
            raise ExtractError(f"helper {name} not found")
        helpers[name] = Builder(_body(found[0])).run()
    out: dict[str, dict[str, Use]] = {}
    for function in extract_function(src, r"^bool RelionJob::getCommands\w+Job\("):
        m = re.match(r"bool RelionJob::getCommands(\w+)Job\(", function)
        assert m is not None
        label = BUILDERS.get(m.group(1))
        if label is None:
            raise ExtractError(f"no job type for getCommands{m.group(1)}Job: add it to BUILDERS")
        out[label] = Builder(_body(function), helpers).run()
    missing = set(BUILDERS.values()) - out.keys()
    if missing:
        raise ExtractError(f"no command builder found for {sorted(missing)}")
    unknown = set(BUILDERS.values()) - {t.label for t in JOB_TYPES}
    if unknown:
        raise ExtractError(f"not job types of relion_consts.py: {sorted(unknown)}")
    return out


def assemble(found: dict[str, dict[str, Use]], version: str, options: dict[str, set[str]]) -> dict:
    """``options``: label -> the options of its job type (all variants). A helper writes flags for several
    kinds of input; those of options a job type does not have (in_manifolds for a 3D refinement) are left out."""
    jobs = {}
    for label in sorted(found):
        own = options.get(label)
        if own is None:
            raise ExtractError(f"{label} is not a job type of the job definitions")
        jobs[label] = {flag: {"direct": sorted(use.direct & own), "all": sorted(use.all & own)}
                       for flag, use in sorted(found[label].items()) if use.needs is None or use.needs <= own}
    return {
        "format": 1,
        "relion_version": version,
        "generated_by": "tools/extract_jobdefs/command_flags.py",
        "notice": NOTICE.replace("(src/pipeline_jobs.cpp, src/gui_jobwindow.cpp, src/gui_mainwindow.cpp; ",
                                 "(src/pipeline_jobs.cpp; "),
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("relion_src", type=Path, help="RELION source tree, or its src/pipeline_jobs.cpp")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--version", required=True, help="RELION version string to record, e.g. 5.1.1")
    ap.add_argument("--jobdefs", type=Path, help="the job definitions of that version (made by extract.py; "
                    "default: cryo_savanna/jobdefs/relion-<version>.json)")
    args = ap.parse_args(argv)
    path = args.relion_src if args.relion_src.is_file() else args.relion_src / "src" / "pipeline_jobs.cpp"
    defs = json.loads((args.jobdefs or REPO / "cryo_savanna" / "jobdefs" / f"relion-{args.version}.json").read_text())
    options: dict[str, set[str]] = {}
    for v in defs["variants"].values():
        options.setdefault(v["label"], set()).update(v["options"])
    data = assemble(scan(path.read_text(encoding="utf-8", errors="replace")), args.version, options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    n = sum(len(v) for v in data["jobs"].values())
    print(f"wrote {args.output}: {len(data['jobs'])} job types, {n} flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
