# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's JSON files in a project, changed by several sessions at once."""

import subprocess
import sys
from pathlib import Path

from cryo_savanna.project.main_inputs import MainInputs
from cryo_savanna.project.shared_json import read_json, update_json

ROOT = Path(__file__).resolve().parents[1]

_ADD = """
import sys
from pathlib import Path
from cryo_savanna.project.shared_json import update_json
path, me = Path(sys.argv[1]), sys.argv[2]
for i in range(40):
    update_json(path, lambda d: {**d, "n": d.get("n", 0) + 1, me + str(i): i})
"""


def test_changes_made_at_the_same_time_are_all_kept(tmp_path):
    path = tmp_path / ".cryo-savanna" / "x.json"
    procs = [subprocess.Popen([sys.executable, "-c", _ADD, str(path), f"p{k}"], cwd=ROOT) for k in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    data = read_json(path)
    assert data["n"] == 160 and len(data) == 161
    assert not list(path.parent.glob("*.tmp"))


def test_main_input_chosen_in_another_window_is_kept(tmp_path):
    a, b = MainInputs(tmp_path), MainInputs(tmp_path)
    b.choices()  # b read the file earlier
    a.set("Class3D/job009/", "Extract/job007/particles.star")
    b.set("Refine3D/job011/", "Select/job010/particles.star")
    assert MainInputs(tmp_path).choices() == {"Class3D/job009/": "Extract/job007/particles.star",
                                              "Refine3D/job011/": "Select/job010/particles.star"}
    b.set("Class3D/job009/", None)
    assert a.choices() == {"Refine3D/job011/": "Select/job010/particles.star"}


def test_unreadable_contents_start_afresh(tmp_path):
    path = tmp_path / "y.json"
    path.write_text("[1, 2")
    assert update_json(path, lambda d: {**d, "k": 1}) == {"k": 1}
