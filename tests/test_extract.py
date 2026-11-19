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


def test_command_flags_of_tricky_cpp():
    """tools/extract_jobdefs/command_flags.py: the flags a command builder adds and the options behind them."""
    import command_flags
    src = r'''
bool RelionJob::getCommandsDemoJob(std::string &outputname, std::vector<std::string> &commands)
{
	std::string command = "`which relion_demo` --o " + outputname;
	error_message = "ERROR: --not_a_flag in a message";
	if (joboptions["do_a"].getBoolean())
	{
		command += " --a " + joboptions["a_value"].getString();
		if (joboptions["do_b"].getBoolean()) command += " --b";
	}
	else if (joboptions["do_c"].getBoolean())
		command += " --c";
	else
		command += " --d --e ";
	float val = joboptions["range"].getNumber(error_message);
	val = (val < 0.) ? 0. : val;
	if (val > 0.) command += " --range " + floatToString(val / 3.);
	val = joboptions["other"].getNumber(error_message);
	command += " --other " + floatToString(val); // " --in_comment"
	return true;
}
'''
    b = command_flags.Builder(command_flags.tokens(src[src.index("{"):]))
    flags = {f: (sorted(u.direct), sorted(u.all)) for f, u in b.run().items()}
    assert flags == {
        "--o": ([], []),
        "--a": (["a_value", "do_a"], ["a_value", "do_a"]),
        "--b": (["do_b"], ["do_a", "do_b"]),  # direct: its own test only
        "--c": (["do_a", "do_c"], ["do_a", "do_c"]),  # an else-if: after do_a failed
        "--d": (["do_a", "do_c"], ["do_a", "do_c"]), "--e": (["do_a", "do_c"], ["do_a", "do_c"]),
        "--range": (["range"], ["range"]),  # through a local variable
        "--other": (["other"], ["other"]),  # the variable given a new value
    }


@pytest.mark.skipif(not os.environ.get("RELION_SRC"), reason="set RELION_SRC to a RELION 5.1.1 source tree")
def test_committed_command_flags_are_up_to_date(tmp_path):
    import command_flags
    out = tmp_path / "flags.json"
    assert command_flags.main([os.environ["RELION_SRC"], "-o", str(out), "--version", "5.1.1"]) == 0
    committed = json.loads((REPO / "cryo_savanna/jobdefs/command-flags-5.1.1.json").read_text())
    assert json.loads(out.read_text())["jobs"] == committed["jobs"]
