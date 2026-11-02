# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the job-definition extractor (tools/extract_jobdefs).

The full extraction runs only when RELION_SRC points at a RELION source tree; it then checks
that the committed JSON is up to date with that source.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "extract_jobdefs"))
import extract  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_extract_function_handles_tricky_cpp():
    src = r'''
void A::f()
{
	place("x"); //comment with a backslash continues \
	on the next line { not code
	std::string s = "brace { in string \" with escape";
	char c = '}';
	/* block { comment */
	if (1) { g(); }
}

void A::g()
{
}
'''
    funcs = extract.extract_function(src, r"^void A::f\(\)")
    assert len(funcs) == 1
    assert funcs[0].rstrip().endswith("if (1) { g(); }\n}")


def test_shortest_float32():
    assert extract.shortest_float32(0.100000001) == 0.1
    assert extract.shortest_float32(2.0) == 2
    assert extract.shortest_float32(0.05000000074505806) == 0.05


@pytest.mark.skipif(not os.environ.get("RELION_SRC"), reason="set RELION_SRC to a RELION 5.1.1 source tree")
@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_committed_json_is_up_to_date(tmp_path):
    out = tmp_path / "jobdefs.json"
    assert extract.main([os.environ["RELION_SRC"], "-o", str(out), "--version", "5.1.1",
                         "--build-dir", str(tmp_path / "build")]) == 0
    committed = json.loads((REPO / "cryo_savanna/jobdefs/relion-5.1.1.json").read_text())
    fresh = json.loads(out.read_text())
    assert fresh["variants"] == committed["variants"]
    assert fresh["menus"] == committed["menus"]
