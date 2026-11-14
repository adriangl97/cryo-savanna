# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""runner_common.py: the helpers the runners of the "Additional external jobs" share, and their copies in a project."""

import subprocess
import sys

import pytest

from cryo_savanna import aretomo_runner, mask_runner, pytom_runner, runner_common
from cryo_savanna.jobdefs import aretomo_jobs, pytom_jobs, runner_files, tomomask_jobs

RUNNERS = [(pytom_jobs, "pytom_runner.py", "match"), (aretomo_jobs, "aretomo_runner.py", "align"),
           (tomomask_jobs, "mask_runner.py", "masks")]


@pytest.mark.parametrize("jobs, runner, mode", RUNNERS)
def test_each_runner_goes_into_the_project_with_runner_common(tmp_path, jobs, runner, mode):
    dst = jobs.install_runner(tmp_path)
    assert dst == tmp_path / ".cryo-savanna" / runner
    common = tmp_path / runner_files.COMMON_IN_PROJECT
    assert common.read_bytes() == open(runner_common.__file__, "rb").read()
    # no other runner is needed beside it: only runner_common.py
    assert sorted(p.name for p in dst.parent.iterdir()) == sorted({runner, "runner_common.py"})


@pytest.mark.parametrize("jobs, runner, mode", RUNNERS)
def test_a_copied_runner_runs_on_its_own(tmp_path, jobs, runner, mode):
    """Run as RELION runs it, from the project's copy (python -I: nothing from cryo-savanna's checkout or the
    environment): it finds runner_common.py beside it, says what is wrong, and leaves a failure exit file."""
    jobs.install_runner(tmp_path)
    job = tmp_path / "External/job001"
    job.mkdir(parents=True)
    r = subprocess.run([sys.executable, "-I", f".cryo-savanna/{runner}", mode, "--o", "External/job001/"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert f"cryo-savanna {runner[:-3]}: no --j in the arguments" in r.stderr
    assert (job / runner_common.FAILURE).exists()


def test_install_replaces_a_changed_copy_and_keeps_an_equal_one(tmp_path):
    dst = tmp_path / "x.py"
    runner_files.install_file(runner_common.__file__, dst)
    first = dst.stat().st_ino
    runner_files.install_file(runner_common.__file__, dst)
    assert dst.stat().st_ino == first  # the same already: left alone
    dst.write_text("old")
    runner_files.install_file(runner_common.__file__, dst)
    assert dst.read_bytes() == open(runner_common.__file__, "rb").read()
    assert [p.name for p in tmp_path.iterdir()] == ["x.py"]  # no temporary file left


def test_pytom_runner_keeps_the_names_older_runners_use():
    """aretomo_runner.py and mask_runner.py copied into projects before runner_common.py import pytom_runner.py
    and use these from it; a newer pytom_runner.py in the same project must still have them."""
    used = ["write_star", "_fail", "SUCCESS", "FAILURE", "ABORTED", "_show_tail", "_link", "_f", "write_output_nodes",
            "text", "say", "number", "loop_block", "_touch", "_rel", "Usage", "TOMO_NAME", "yes", "words",
            "tomogram_rows", "tomogram_of", "run_parallel", "num", "mrc_header", "load_modules", "gpu_ids",
            "TOMO_MASK_SUFFIX", "Supervisor", "Aborted"]
    for name in used:
        assert getattr(pytom_runner, name) is getattr(runner_common, name), name
    assert pytom_runner.parse(["match", "--o", "a", "--j", "2"]) == ("match", "a/", 2, {})  # pytom's modes
    assert pytom_runner.parse(["align", "--o", "a", "--j", "1", "--x", "y"], ("align",)) == ("align", "a/", 1, {"x": "y"})


def test_each_runner_says_its_own_name(capsys):
    for module, who in ((pytom_runner, "pytom"), (aretomo_runner, "AreTomo3"), (mask_runner, "tomogram masks")):
        runner_common.WHO = module.WHO
        module.say("hello")
        assert capsys.readouterr().out == f" ++++ {who}: hello\n"
    runner_common.WHO = "cryo-savanna"


def test_supervisor_names_what_to_check_when_a_program_cannot_start(tmp_path):
    sup = runner_common.Supervisor(str(tmp_path), hint=" (check X)")
    with pytest.raises(runner_common.Usage, match=r"cannot start /no/such/program: .* \(check X\)$"):
        sup.start(["/no/such/program"], str(tmp_path / "log"))
