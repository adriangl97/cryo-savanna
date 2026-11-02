#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Extract RELION job definitions (options + GUI layout) from a RELION source tree.

The RelionJob::initialise* functions (src/pipeline_jobs.cpp), the JobWindow::initialise*
functions (src/gui_jobwindow.cpp) and the job menu of the main window (src/gui_mainwindow.cpp)
are copied verbatim into a small C++ harness (stubs.h + harness_main.inc), compiled and run.
That gives exact defaults, ranges, help texts, tomo variants, tabs, groups and toggles without
parsing C++ by hand.

The harness runs twice: once with no RELION_* environment variables, and once with sentinel
values for all of them. Comparing the two runs shows which option fields come from which
environment variable; those are stored as "env" bindings and resolved at runtime.

Usage:
    python tools/extract_jobdefs/extract.py /path/to/relion-5.1.1 \
        -o cryo_savanna/jobdefs/relion-5.1.1.json --version 5.1.1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from cryo_savanna.relion_consts import JOB_TYPES  # noqa: E402

# Environment variables read by the initialisers that are not plain strings.
# Values are sentinels chosen so they cannot be confused with compiled-in defaults.
NUMERIC_ENV = {
    "RELION_MPI_MAX": "97",
    "RELION_THREAD_MAX": "89",
    "RELION_QSUB_NRMPI": "7",
    "RELION_QSUB_NRTHREADS": "5",
    "RELION_MINIMUM_DEDICATED": "11",
}
BOOL_ENV = {"RELION_QUEUE_USE": "yes"}
# Handled at runtime by cryo-savanna itself (dynamic extra queue fields), or not used while
# initialising jobs. Left unset in both runs.
SKIP_ENV = {"RELION_QSUB_EXTRA_COUNT", "RELION_MPIRUN", "RELION_ERROR_LOCAL_MPI"}
FORCED_STATE_ENV = {"RELION_ALLOW_CHANGE_MINIMUM_DEDICATED": "0"}


class ExtractError(RuntimeError):
    pass


# Written into the JSON: what it is derived from, and RELION's notice in full, as it asks (see NOTICE.md)
NOTICE = ("Derived from RELION (src/pipeline_jobs.cpp, src/gui_jobwindow.cpp, src/gui_mainwindow.cpp; "
          "Author: Sjors H.W. Scheres, MRC Laboratory of Molecular Biology), free software under the GNU "
          "General Public License version 2 or (at your option) any later version. See NOTICE.md.\n"
          "RELION's notice, kept as it asks:\n"
          "Author: \"Sjors H.W. Scheres\"\n"
          "MRC Laboratory of Molecular Biology\n"
          "\n"
          "This program is free software; you can redistribute it and/or modify\n"
          "it under the terms of the GNU General Public License as published by\n"
          "the Free Software Foundation; either version 2 of the License, or\n"
          "(at your option) any later version.\n"
          "\n"
          "This program is distributed in the hope that it will be useful,\n"
          "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
          "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
          "GNU General Public License for more details.\n"
          "\n"
          "This complete copyright notice must be included in any revised version of the\n"
          "source code. Additional authorship citations may be added, but existing\n"
          "author citations must be preserved.")


# ---------------------------------------------------------------------------- C++ text utils


def _skip_literal(src: str, i: int) -> int:
    """Return index just past the string/char literal or comment starting at i (or i if none)."""
    c = src[i]
    if c in "\"'":
        q = c
        i += 1
        while i < len(src):
            if src[i] == "\\":
                i += 2
                continue
            if src[i] == q:
                return i + 1
            i += 1
        raise ExtractError("unterminated literal")
    if src.startswith("//", i):
        # A backslash at the end of the line continues the comment (RELION has a few of those)
        j = src.find("\n", i)
        while j > 0 and src[j - 1] == "\\":
            j = src.find("\n", j + 1)
        return len(src) if j < 0 else j
    if src.startswith("/*", i):
        j = src.find("*/", i + 2)
        if j < 0:
            raise ExtractError("unterminated comment")
        return j + 2
    return i


def extract_function(src: str, signature_regex: str) -> list[str]:
    """Return the full text of every function whose definition matches signature_regex."""
    out = []
    for m in re.finditer(signature_regex, src, flags=re.M):
        i = src.index("{", m.end())
        depth = 0
        j = i
        while j < len(src):
            k = _skip_literal(src, j)
            if k != j:
                j = k
                continue
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(src[m.start(): j + 1])
                    break
            j += 1
        else:
            raise ExtractError(f"unbalanced braces after {m.group(0)!r}")
    return out


def header_defines(h: str) -> str:
    """Constant part of pipeline_jobs.h: #defines, option lists and process-type maps."""
    start = h.index("#define SRC_PIPELINE_JOBS_H_") + len("#define SRC_PIPELINE_JOBS_H_")
    a_end = h.index("static std::string get_proc_label(int type)")
    b_start = h.index("// Status a Process may have")
    b_end = h.index("struct gui_layout")
    text = h[start:a_end] + "\n" + h[b_start:b_end]
    return re.sub(r"^#include.*$", "", text, flags=re.M)


def menu_block(main_cpp: str) -> str:
    start = main_cpp.index("nr_browse_tabs = 0;")
    end = main_cpp.index("browser->callback(", start)
    block = main_cpp[start:end]
    block = re.sub(r"(\w+)\[nr_browse_tabs\]->initialise\(", r"harness_init(\1[nr_browse_tabs], browser, ", block)
    return (
        "void build_menu(bool _do_tomo)\n{\n"
        "\tFl_Hold_Browser *browser = new Fl_Hold_Browser(0, 0, 0, 0);\n"
        "\tFl_Group *browse_grp[128];\n\tJobWindow *gui_jobwindows[128];\n\tint nr_browse_tabs;\n"
        + block
        + "\n}\n"
    )


def build_source(relion: Path) -> str:
    src = relion / "src"
    pj_h = (src / "pipeline_jobs.h").read_text()
    pj_cpp = (src / "pipeline_jobs.cpp").read_text()
    jw_cpp = (src / "gui_jobwindow.cpp").read_text()
    mw_cpp = (src / "gui_mainwindow.cpp").read_text()

    stubs = (HERE / "stubs.h").read_text().replace("// @@PIPELINE_JOBS_H_DEFINES@@", header_defines(pj_h))
    stubs = stubs.replace("#pragma once", "")

    parts = [stubs]
    parts += extract_function(pj_cpp, r"^void RelionJob::initialise\(int _job_type\)")
    parts += extract_function(pj_cpp, r"^void RelionJob::addTomoInputOptions\(")
    job_inits = extract_function(pj_cpp, r"^void RelionJob::initialise\w+Job\(\)")
    win_inits = extract_function(jw_cpp, r"^void JobWindow::initialise\w+Window\(\)")
    if len(job_inits) < 30 or len(win_inits) < 30:
        raise ExtractError(f"found only {len(job_inits)} job and {len(win_inits)} window initialisers")
    parts += job_inits
    parts += extract_function(jw_cpp, r"^void JobWindow::setupTabs\(int nr_tabs\)")
    parts += extract_function(jw_cpp, r"^void JobWindow::setupRunTab\(\)")
    parts += extract_function(jw_cpp, r"^void JobWindow::placeTomoInput\(")
    parts += extract_function(jw_cpp, r"^void JobWindow::initialise\(int my_job_type, bool _is_tomo\)")
    parts += win_inits
    parts.append((HERE / "harness_main.inc").read_text())
    parts.append(menu_block(mw_cpp))
    return "\n\n".join(parts)


def env_names(relion: Path) -> list[str]:
    names = set()
    for f in ("pipeline_jobs.cpp", "gui_jobwindow.cpp"):
        names.update(re.findall(r'getenv\s*\(\s*"(RELION_[A-Z0-9_]+)"', (relion / "src" / f).read_text()))
    return sorted(names)


# ---------------------------------------------------------------------------- compile and run

def compile_harness(source: str, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp = build_dir / "harness.cpp"
    cpp.write_text(source)
    exe = build_dir / "harness"
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise ExtractError("no C++ compiler found (set CXX)")
    res = subprocess.run([cxx, "-std=c++17", "-O0", "-w", "-o", str(exe), str(cpp)], capture_output=True, text=True)
    if res.returncode != 0:
        raise ExtractError(f"harness failed to compile:\n{res.stderr[-6000:]}")
    return exe


def run_harness(exe: Path, env: dict[str, str]) -> dict:
    base = {k: v for k, v in os.environ.items() if not k.startswith("RELION_")}
    base.update(env)
    res = subprocess.run([str(exe)], capture_output=True, text=True, env=base)
    if res.returncode != 0:
        raise ExtractError(f"harness failed:\n{res.stderr[-4000:]}")
    return json.loads(res.stdout)


# ---------------------------------------------------------------------------- post-processing

def variant_key(label: str, is_tomo: bool) -> str:
    return f"{label}:tomo" if is_tomo else label


def shortest_float32(x: float) -> float | int:
    """Shortest decimal that maps to the same float32 (0.100000001 -> 0.1)."""
    target = struct.pack("f", x)
    for prec in range(1, 10):
        y = float(f"{x:.{prec}g}")
        if struct.pack("f", y) == target:
            return int(y) if y.is_integer() else y
    return x


def find_env_bindings(clean: dict, sentinel: dict, string_env: dict[str, str]) -> dict:
    """Map (variant, option) -> {field: env var} by diffing clean and sentinel runs."""
    by_value = {v: k for k, v in string_env.items()}
    num_by_value = {float(v): k for k, v in NUMERIC_ENV.items()}
    bindings: dict[tuple[int, bool, str], dict] = {}
    for vc, vs in zip(clean["variants"], sentinel["variants"]):
        assert (vc["proc_id"], vc["is_tomo"]) == (vs["proc_id"], vs["is_tomo"])
        for key, oc in vc["options"].items():
            os_ = vs["options"].get(key)
            if os_ is None:
                raise ExtractError(f"option {key} only exists without environment variables")
            b = {}
            for field in ("default", "min", "max"):
                if field not in oc or oc[field] == os_[field]:
                    continue
                val = os_[field]
                if isinstance(val, str) and val in by_value:
                    b[field] = by_value[val]
                elif oc["kind"] == "boolean" and field == "default" and val == "Yes":
                    b[field] = next(iter(BOOL_ENV))
                else:
                    try:
                        b[field] = num_by_value[float(val)]
                    except (KeyError, ValueError):
                        raise ExtractError(f"cannot attribute change of {key}.{field}: {oc[field]!r} -> {val!r}")
            if oc.get("forced_inactive") != os_.get("forced_inactive"):
                b["forced_inactive"] = next(iter(FORCED_STATE_ENV))
            if b:
                bindings[(vc["proc_id"], vc["is_tomo"], key)] = b
    return bindings


def assemble(clean: dict, bindings: dict, version: str, commit: str | None) -> dict:
    proc = {p["proc_id"]: p for p in clean["proc_types"]}

    # cross-check with relion_consts
    ours = {(t.proc_id, t.label, t.dirname) for t in JOB_TYPES}
    theirs = {(p["proc_id"], p["label"], p["dirname"]) for p in clean["proc_types"]}
    if ours != theirs:
        raise ExtractError(
            "cryo_savanna/relion_consts.py JOB_TYPES differs from the RELION source:\n"
            f"  only in relion_consts: {sorted(ours - theirs)}\n  only in source: {sorted(theirs - ours)}"
        )

    variants = {}
    for v in clean["variants"]:
        p = proc[v["proc_id"]]
        for key, opt in v["options"].items():
            for f in ("min", "max", "step"):
                if f in opt:
                    opt[f] = shortest_float32(opt[f])
            b = bindings.get((v["proc_id"], v["is_tomo"], key))
            if b:
                opt["env"] = b
        variants[variant_key(p["label"], v["is_tomo"])] = {
            "label": p["label"],
            "proc_id": v["proc_id"],
            "dirname": p["dirname"],
            "is_tomo": v["is_tomo"],
            "options": v["options"],
            "tabs": v["tabs"],
            "groups": v["groups"],
            "missing": v["missing"],
        }

    menus = {}
    for mode, entries in clean["menus"].items():
        menus[mode] = [
            {"menu_label": e["menu_label"], "variant": variant_key(proc[e["proc_id"]]["label"], e["is_tomo"])}
            for e in entries
        ]
    return {
        "format": 1,
        "relion_version": version,
        "relion_commit": commit,
        "generated_by": "tools/extract_jobdefs/extract.py",
        "notice": NOTICE,
        "proc_types": clean["proc_types"],
        "menus": menus,
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("relion_src", type=Path, help="RELION source tree (contains src/pipeline_jobs.cpp)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--version", required=True, help="RELION version string to record, e.g. 5.1.1")
    ap.add_argument("--build-dir", type=Path, default=HERE / "build")
    args = ap.parse_args(argv)

    exe = compile_harness(build_source(args.relion_src), args.build_dir)
    names = [n for n in env_names(args.relion_src) if n not in SKIP_ENV]
    string_env = {
        n: f"@@{n}@@" for n in names if n not in NUMERIC_ENV and n not in BOOL_ENV and n not in FORCED_STATE_ENV
    }
    clean = run_harness(exe, {})
    sentinel_env = dict(string_env)
    sentinel_env.update({k: v for k, v in NUMERIC_ENV.items() if k in names})
    sentinel_env.update({k: v for k, v in BOOL_ENV.items() if k in names})
    sentinel_env.update({k: v for k, v in FORCED_STATE_ENV.items() if k in names})
    sentinel = run_harness(exe, sentinel_env)
    bindings = find_env_bindings(clean, sentinel, string_env)

    commit = None
    if (args.relion_src / ".git").exists():
        r = subprocess.run(["git", "-C", str(args.relion_src), "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip() or None

    data = assemble(clean, bindings, args.version, commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

    n_opts = sum(len(v["options"]) for v in data["variants"].values())
    n_env = len(bindings)
    missing = {k: v["missing"] for k, v in data["variants"].items() if v["missing"]}
    print(f"wrote {args.output}: {len(data['variants'])} variants, {n_opts} options, {n_env} env-bound options")
    print(f"menus: spa={len(data['menus']['spa'])} tomo={len(data['menus']['tomo'])}")
    if missing:
        print(f"placed but undefined options (RELION GUI warns about these too): {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
