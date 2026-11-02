# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Installing with conda/mamba, and the licence notices."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_conda_environments_install_cryo_savanna():
    """`mamba env create -f <file>` installs cryo-savanna itself (pip, from the checkout) with both commands."""
    import tomllib
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"cryo-savanna": "cryo_savanna.app:main", "savanna": "cryo_savanna.app:main"}
    for name in ("environment.yml", "environment.lock.yml", "environment-pyqt5.yml"):
        text = (ROOT / name).read_text()
        assert "  - pip:\n      - .  #" in text, name
        assert "conda-forge" in text and "name: cryo-savanna" in text, name


def test_licence_notices_are_in_place():
    """GPL-2.0-or-later: the licence text, the notices, a header in every source file, the RELION notice."""
    import json
    assert "GNU GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text() and "Version 2, June 1991" in (
        ROOT / "LICENSE").read_text()
    notice = (ROOT / "NOTICE.md").read_text()
    assert "This complete copyright notice must be included" in notice and "Bilbao-Castro" in notice
    out = subprocess.run(["git", "ls-files", "*.py", "*.sh", "*.cpp", "*.h", "*.inc", "*.lua", "bin/"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    missing = [f for f in out.stdout.split() if (ROOT / f).is_file() and not (ROOT / f).is_symlink()
               and "SPDX-License-Identifier: GPL-2.0-or-later" not in (ROOT / f).read_text()]
    assert not missing, missing
    # the runner is copied into users' projects, which get shared: it carries the licence notice itself
    runner = (ROOT / "cryo_savanna/cli_runner.py").read_text()
    assert "either version 2 of the License" in runner and "gnu.org/licenses" in runner
    assert "Redistribution and use in source and binary forms" in notice and "Alister Burt" in notice  # BSD-3
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "GPL-2.0-or-later" and project["license-files"] == ["LICENSE", "NOTICE.md"]
    derived = [f for f in out.stdout.split() if (ROOT / f).is_file() and not (ROOT / f).is_symlink()
               and "Derived from RELION" in (ROOT / f).read_text()]
    listed = [f for f in derived if f.startswith("cryo_savanna/") and f.rsplit("/", 1)[-1] not in notice]
    assert not listed, f"derived from RELION but not in NOTICE.md: {listed}"
    # RELION asks for its complete notice "in any revised version of the source code"
    ported = [f for f in derived if "Derived from RELION" in "\n".join((ROOT / f).read_text().splitlines()[:5])]
    without = [f for f in ported if "This complete copyright notice must be included" not in (ROOT / f).read_text()]
    assert not without, f"derived from RELION without RELION's notice: {without}"
    assert "Bilbao-Castro" in (ROOT / "cryo_savanna/project/star.py").read_text()
    for f in ("cryo_savanna/ui/job_editor.py", "cryo_savanna/project/ops.py", "cryo_savanna/project/selection.py"):
        assert "Sorzano" in (ROOT / f).read_text(), f  # XMIPP code in filename.cpp, multidim_array.h, transformations.h
    assert "Bilbao-Castro" in (ROOT / "tools/extract_jobdefs/stubs.h").read_text()  # strings.cpp's conversions
    assert "src/strings.cpp" in notice.split("`tools/extract_jobdefs/stubs.h`", 1)[1].split("\n", 1)[0]
    assert "transformations.h" in notice and "Jasenko Zivanov" in notice
    # the ported tilt-image exclusion keeps the BSD notice itself, also when copied on its own
    assert "Redistribution and use in source and binary forms" in (ROOT / "cryo_savanna/project/selection.py").read_text()
    defs = json.loads((ROOT / "cryo_savanna" / "jobdefs" / "relion-5.1.1.json").read_text())
    assert "Scheres" in defs["notice"] and "NOTICE.md" in defs["notice"]
    assert "This complete copyright notice must be included" in defs["notice"]  # RELION's notice in full


def test_help_texts_carry_no_details_of_the_machine_they_came_from():
    """tools/bundle_help.py scrubs install and build paths, host names and the build's commit."""
    import re
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from bundle_help import scrub
    raw = ("# /opt/x/relion/bin/relion_refine_mpi\nRELION version: 5.1.1-commit-0a1b2c \nPrecision: BASE=double\n"
           " + Leader      (0) runs on host            = node7\nin: /scratch/u/relion/src/image.h, line 276\n"
           "/opt/x/relion/bin/relion_refine_mpi(+0x8b) [0x5f]\n")
    assert scrub(raw) == ("# relion_refine_mpi\nRELION version: 5.1.1 \n + Leader      (0) runs on host            = <host>\n"
                          "in: src/image.h, line 276\nrelion_refine_mpi(+0x8b) [0x5f]\n")
    leaks = re.compile(r"/[\w.-]+/(?:[\w.-]+/)*(?:bin/relion|src/)|runs on host\s*=\s*(?!<host>)\S|commit-[0-9a-f]{6}|"
                       r"^Precision:", re.M)
    for d in (ROOT / "tests/data/help-5.1.1", ROOT / "cryo_savanna/jobdefs/help/5.1.1"):
        bad = [f.name for f in sorted(d.glob("*.txt")) if leaks.search(f.read_text())]
        assert not bad, bad
