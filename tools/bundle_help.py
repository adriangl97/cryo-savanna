# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bundle the --help texts of the programs that jobs and CLI tools run (cryo_savanna/jobdefs/help/<version>/).

They are the fallback for the "Additional arguments" flag list and the CLI tool forms when RELION
is not loaded. Most come from the --help dumps of tools/validate.py; relion_run_motioncorr stops
before printing its help (it insists on --use_own or --use_motioncor2), and relion_manualpick and
relion_display are not run (they open windows), so those three are rebuilt from RELION's source in
the format of IOParser::writeUsage.

Every text is scrubbed of the machine it was captured on (``scrub``): RELION's install and build
directories, the MPI host name, and the build's commit and precision lines. ``--scrub DIR...`` does
only that, to the texts in those directories (e.g. tests/data/help-<version>/, copied from the dumps).

    python tools/bundle_help.py <help dump dir> <relion src dir> [version]
    python tools/bundle_help.py --scrub <dir> [<dir>...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cryo_savanna.jobdefs.flags import KNOWN_PROGRAMS  # noqa: E402

FROM_SOURCE = {
    "relion_run_motioncorr": ("motioncorr_runner.cpp", "MotioncorrRunner::read"),
    "relion_manualpick": ("manualpicker.cpp", "ManualPicker::read"),
    "relion_display": ("displayer.cpp", "Displayer::read"),
}

_SOURCE = re.compile(r"(?:/[^\s/:()]+)+?/(src/)")  # where RELION was built: /scratch/u/relion/src/x.cpp -> src/x.cpp
_ABS_PROGRAM = re.compile(r"(?:/[^\s/:()]+)+/(relion\w*)")  # /opt/x/bin/relion_refine -> relion_refine
_HOST = re.compile(r"(runs on host\s*=\s*)\S+")
_VERSION = re.compile(r"^(RELION version: \d+(?:\.\d+)*)\S*", re.M)


def scrub(text: str) -> str:
    """A --help capture without details of the machine it ran on: program paths become bare names,
    MPI host names "<host>", "RELION version: 5.1.1-commit-..." the release, no "Precision:" line."""
    text = _SOURCE.sub(r"\1", text)
    text = _ABS_PROGRAM.sub(r"\1", text)
    text = _HOST.sub(r"\1<host>", text)
    text = _VERSION.sub(r"\1", text)
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("Precision:"))


_STR = r'"((?:[^"\\]|\\.)*)"'
_CALL = re.compile(r"parser\.(addSection|getOption|checkOption)\s*\(\s*((?:" + _STR + r"\s*,?\s*)+)")


def usage_from_source(cpp: Path, function: str) -> str:
    """The --help text of the options a C++ read() function declares (string-literal arguments only)."""
    src = cpp.read_text()
    start = src.index(f"void {function}(")
    end = src.index("\n}", start)
    body = src[start:end]
    sections: list[tuple[str, list[tuple[str, str, str | None]]]] = []
    for m in _CALL.finditer(body):
        kind = m.group(1)
        args = [a.encode().decode("unicode_escape") for a in re.findall(_STR, m.group(2))]
        if kind == "addSection":
            sections.append((args[0], []))
        elif kind == "checkOption":
            sections[-1][1].append((args[0], args[1] if len(args) > 1 else "", "false"))
        else:
            sections[-1][1].append((args[0], args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else None))
    lines = [f"# generated from src/{cpp.name} ({function})",
             "+++ RELION: command line arguments (with defaults for optional ones between parantheses) +++"]
    for name, opts in sections:
        lines.append(f"====== {name} ===== ")
        # compulsory options first, like IOParser::writeUsageOneSection
        for flag, usage, default in [o for o in opts if o[2] is None] + [o for o in opts if o[2] is not None]:
            aux = f"  {flag}" + (f" ({default})" if default is not None else "")
            lines.append(f"{aux:>35} : {usage}")
    lines.append(f"{'--version':>35} : Print RELION version and exit")
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1:2] == ["--scrub"]:
        for d in sys.argv[2:]:
            for f in sorted(Path(d).glob("*.txt")):
                f.write_text(scrub(f.read_text()))
        return
    dumps, src = Path(sys.argv[1]), Path(sys.argv[2])
    version = sys.argv[3] if len(sys.argv) > 3 else "5.1.1"
    out = ROOT / "cryo_savanna" / "jobdefs" / "help" / version
    out.mkdir(parents=True, exist_ok=True)
    for prog in sorted(KNOWN_PROGRAMS):
        if prog in FROM_SOURCE:
            cpp, fn = FROM_SOURCE[prog]
            text = usage_from_source(src / "src" / cpp, fn)
        else:
            f = dumps / f"{prog}.txt"
            if not f.exists():
                print(f"no --help dump for {prog}")
                continue
            text = f.read_text()
        (out / f"{prog}.txt").write_text(scrub(text))
        print(f"{prog}: {text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
