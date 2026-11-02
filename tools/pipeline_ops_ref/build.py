# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Build ops_ref against a RELION build directory (the one that built relion_pipeliner).

    python tools/pipeline_ops_ref/build.py RELION_BUILD_DIR -o OUTPUT

Compile and link flags are taken from the build's own pipeliner target, so the program uses
exactly the same RELION code as that relion_pipeliner.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make_vars(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if " = " in line and not line.startswith("#"):
            k, v = line.split(" = ", 1)
            out[k.strip()] = v.strip()
    return out


def build(build_dir: Path, output: Path) -> Path:
    target = build_dir / "src/apps/CMakeFiles/pipeliner.dir"
    flags = make_vars(target / "flags.make")
    link = (target / "link.txt").read_text().split()
    compiler = link[0]
    # link.txt: <compiler> <flags...> CMakeFiles/pipeliner.dir/pipeliner.cpp.o -o ../../bin/relion_pipeliner <libs...>
    i_obj = next(i for i, a in enumerate(link) if a.endswith("pipeliner.cpp.o"))
    i_out = link.index("-o", i_obj)
    link_flags = link[1:i_obj]
    libs = [str((build_dir / "src/apps" / a).resolve()) if a.startswith("..") else a for a in link[i_out + 2:]]
    cmd = [compiler, *shlex.split(flags.get("CXX_FLAGS", "")), *shlex.split(flags.get("CXX_DEFINES", "")),
           *shlex.split(flags.get("CXX_INCLUDES", "")), str(HERE / "ops_ref.cpp"), *link_flags, "-o", str(output), *libs]
    # compiler wrappers such as mpicxx look for the real compiler next to themselves
    env = {**os.environ, "PATH": f"{Path(compiler).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(cmd, check=True, env=env)
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("ops_ref"))
    args = ap.parse_args(argv)
    print(build(args.build_dir.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
