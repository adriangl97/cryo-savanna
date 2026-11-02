# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI tools: forms from --help, External job.star round trips, and cli_runner (no Qt)."""

import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cryo_savanna import cli_runner
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.cli_catalog import CATEGORIES, POSITIONAL, cli_programs, describe
from cryo_savanna.jobdefs.cli_tools import (RUNNER, RUNNER_IN_PROJECT, CliCatalog, cli_program_of, install_runner,
                                            output_path, runner_command)
from cryo_savanna.jobdefs.flags import JOB_PROGRAMS, FlagSource
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

ENV = RelionEnv({})
CATALOG = CliCatalog(JobDefs.load(), ENV, FlagSource(which=lambda p: None))
HELP = Path(__file__).parent / "data" / "help-5.1.1"


def test_catalog():
    assert set(POSITIONAL) == cli_runner.POSITIONAL_TOOLS
    assert not set(cli_programs()) & JOB_PROGRAMS  # only programs no GUI job runs
    assert all(describe(p)[0] in CATEGORIES for p in cli_programs())
    dumped = {f.stem for f in HELP.glob("relion_*.txt") if not f.stem.endswith("_mpi")}
    left_out = dumped - set(cli_programs()) - JOB_PROGRAMS
    assert left_out == {"relion_pipeliner", "relion_schemer", "relion_star_plottable", "relion_external_reconstruct",
                        "relion_prepare_subtomo", "relion_tomo_test"}  # documented in cli_catalog.py


def test_every_tool_has_a_form():
    for program in cli_programs():
        v = CATALOG.variant(program)
        assert v is not None and v.label == "relion.external", program
        assert v.tabs[-1].label == "Running"
        js = v.new_jobstar(ENV)
        assert cli_program_of(js) == program and js.options["fn_exe"] == runner_command(program)
        values, _ = v.values_from_jobstar(js, ENV)
        assert values == v.default_values(ENV), program  # untouched form <-> job.star
    assert len(CATALOG.variant("relion_image_handler").flags) == 77


def test_flags_to_arguments_and_back():
    v = CATALOG.variant("relion_image_handler")
    values = v.default_values(ENV)
    values.update({"--i": "Refine3D/job010/run_class001.mrc", "--o": "flipped.mrc", "--invert_hand": "Yes",
                   "--angpix": "1.06", "other_args": "--my_extra 3"})
    js = v.make_jobstar(values, False, "External/job020/", None)
    assert js.options["other_args"] == ("--i Refine3D/job010/run_class001.mrc --o External/job020/flipped.mrc "
                                        "--angpix 1.06 --invert_hand --my_extra 3")
    back, _ = v.values_from_jobstar(js, ENV)
    assert back == values
    # with a STAR file as input, image_handler's --o is a suffix: passed as typed
    values["--i"] = "Extract/job005/particles.star"
    assert "--o flipped.mrc" in v.make_jobstar(values, False, "External/job020/", None).options["other_args"]



def test_values_reach_the_program_as_typed():
    """RELION runs the command through the shell: what it would act on is quoted, so the program gets the
    value shown in the form; a list of files keeps its pattern for the shell; an open quote is no crash."""
    v = CATALOG.variant("relion_image_handler")
    values = v.default_values(ENV) | {"--i": "Movies/*.mrc", "--o": "out_$USER.mrc", "--angpix": "it's"}
    args = v.make_jobstar(values, False, "External/job020/", None).options["other_args"]
    assert "--i 'Movies/*.mrc'" in args and "--o 'External/job020/out_$USER.mrc'" in args
    assert "--angpix 'it'\"'\"'s'" in args
    back, _ = v.values_from_jobstar(JobStar("relion.external", options={"other_args": args}), ENV)
    assert (back["--i"], back["--o"], back["--angpix"]) == ("Movies/*.mrc", "out_$USER.mrc", "it's")
    merge = CATALOG.variant("relion_merge_particles")
    o = merge.make_jobstar(merge.default_values(ENV) | {"inputs": "Select/job*/particles.star it's"}, False,
                           "External/job009/", None).options
    # the pattern is left to the shell, the open quote is quoted (and no ValueError)
    assert o["other_args"].startswith("Select/job*/particles.star 'it'\"'\"'s' ")


def test_an_output_in_another_folder_reads_back_as_typed():
    """Only the job's own directory is taken off an output when the form is read back from job.star."""
    v = CATALOG.variant("relion_image_handler")
    for typed in ("flipped.mrc", "Maps/flipped.mrc", "Refine3D/job011/flipped.mrc"):
        values = v.default_values(ENV) | {"--i": "a.mrc", "--o": typed}
        js = v.make_jobstar(values, False, "External/job020/", None)
        assert v.values_from_jobstar(js, ENV)[0]["--o"] == typed

def test_outputs_go_to_the_job_directory():
    assert output_path("map.mrc", "file", "External/job003/") == "External/job003/map.mrc"
    assert output_path("./map.mrc", "file", "External/job003/") == "External/job003/map.mrc"
    assert output_path("Maps/map.mrc", "file", "External/job003/") == "Maps/map.mrc"
    assert output_path("", "dir", "External/job003/") == "External/job003/"
    assert output_path("", "file", "External/job003/") is None
    assert output_path("Out/", "dir", "External/job003/") == "External/job003/Out/"
    v = CATALOG.variant("relion_reconstruct")
    assert "--o External/job007/relion.mrc" in v.new_jobstar(ENV).options["other_args"].replace("job001", "job007")


def test_mpi_and_queue_settings():
    v = CATALOG.variant("relion_reconstruct")
    assert v.has_mpi and "nr_mpi" in v.options
    values = v.default_values(ENV)
    values.update({"nr_mpi": "4", "--j": "2", "do_queue": "Yes", "queuename": "gpu"})
    o = v.make_jobstar(values, False, "External/job002/", None).options
    assert (o["param1_label"], o["param1_value"], o["nr_threads"], o["do_queue"], o["queuename"]) == \
        ("mpi", "4", "2", "Yes", "gpu")
    back, _ = v.values_from_jobstar(JobStar("relion.external", options=o), ENV)
    assert back["nr_mpi"] == "4" and back["queuename"] == "gpu"
    assert not CATALOG.variant("relion_align_symmetry").has_mpi


def test_positional_tool():
    v = CATALOG.variant("relion_star_printtable")
    values = v.default_values(ENV)
    values.update({"starfile": "Extract/job005/particles.star", "labels": "rlnDefocusU rlnDefocusV"})
    o = v.make_jobstar(values, False, "External/job004/", None).options
    assert o["other_args"] == "Extract/job005/particles.star data_particles rlnDefocusU rlnDefocusV"
    assert (o["param1_label"], o["param1_value"]) == ("stdout", "table.txt")
    back, _ = v.values_from_jobstar(JobStar("relion.external", options=o), ENV)
    assert back == values
    merge = CATALOG.variant("relion_merge_particles")
    vals = merge.default_values(ENV) | {"inputs": "a.star b.star"}
    assert merge.make_jobstar(vals, False, "External/job009/", None).options["other_args"] == \
        "a.star b.star External/job009/particles.star"


def test_inputs_link_the_job_into_the_pipeline(spa_project):
    p = Pipeline.read(spa_project)
    v = CATALOG.variant("relion_reconstruct")
    values = v.default_values(ENV) | {"--i": "Extract/job005/particles.star"}
    o = v.make_jobstar(values, False, "External/job020/", p).options
    assert o["in_part"] == "Extract/job005/particles.star"
    v = CATALOG.variant("relion_image_handler")
    values = v.default_values(ENV) | {"--i": "Refine3D/job011/run_class001.mrc", "--o": "x.mrc"}
    o = v.make_jobstar(values, False, "External/job020/", p).options
    assert o["in_3dref"] == "Refine3D/job011/run_class001.mrc" and o["in_part"] == ""


# ---------------------------------------------------------------------------- cli_runner

def tool(bin_dir: Path, name: str, body: str) -> None:
    f = bin_dir / name
    f.write_text(f"#!{sys.executable}\nimport os, sys, time\nargs = sys.argv[1:]\n{body}\n")
    f.chmod(f.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def job(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "External/job001").mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)
    return tmp_path, bin_dir


def run_runner(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, RUNNER, *args], capture_output=True, text=True, timeout=60)


WRITE_O = """
o = args[args.index('--o') + 1]
open(o, 'wb').write(bytes([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0]))
open(os.path.join(os.path.dirname(o), 'particles.star'), 'w').write('data_particles\\nloop_\\n_rlnImageName\\n')
print('pipeline_control', args[args.index('--pipeline_control') + 1])
"""


def test_runner_registers_outputs_and_succeeds(job):
    root, bin_dir = job
    tool(bin_dir, "relion_image_handler", WRITE_O)
    r = run_runner("relion_image_handler", "--o", "External/job001/", "--in_3dref", "Refine3D/job010/map.mrc",
                   "--j", "1", "--i", "Refine3D/job010/map.mrc", "--o", "External/job001/flipped.mrc",
                   "--pipeline_control", "External/job001/")
    assert r.returncode == 0, r.stderr
    assert "pipeline_control External/job001/.cli/" in r.stdout  # its own exit files stay private
    jd = root / "External/job001"
    assert (jd / "RELION_JOB_EXIT_SUCCESS").exists() and not (jd / ".cli/RELION_JOB_EXIT_SUCCESS").exists()
    nodes = (jd / "RELION_OUTPUT_NODES.star").read_text()
    assert "External/job001/flipped.mrc DensityMap.mrc" in nodes
    assert "External/job001/particles.star ParticleGroupMetadata.star.relion" in nodes


def test_runner_reports_failure(job):
    root, bin_dir = job
    tool(bin_dir, "relion_image_handler", "sys.exit(3)")
    r = run_runner("relion_image_handler", "--o", "External/job001/", "--j", "1", "--i", "x")
    assert r.returncode == 3
    assert (root / "External/job001/RELION_JOB_EXIT_FAILURE").exists()
    assert not (root / "External/job001/RELION_OUTPUT_NODES.star").exists()


def test_runner_passes_an_abort_on(job):
    root, bin_dir = job
    tool(bin_dir, "relion_reconstruct", """
ctl = args[args.index('--pipeline_control') + 1]
while not os.path.exists(ctl + 'RELION_JOB_ABORT_NOW'):
    time.sleep(0.1)
open(ctl + 'RELION_JOB_EXIT_ABORTED', 'w').close()
sys.exit(1)
""")
    p = subprocess.Popen([sys.executable, RUNNER, "relion_reconstruct", "--o", "External/job001/", "--j", "1"])
    time.sleep(1)
    (root / "External/job001/RELION_JOB_ABORT_NOW").touch()
    assert p.wait(timeout=30) != 0
    assert (root / "External/job001/RELION_JOB_EXIT_ABORTED").exists()


def test_runner_passes_a_stop_signal_on(job):
    """Killed itself (scancel, kill): the program, in its own session, is stopped too, and RELION is told."""
    root, bin_dir = job
    tool(bin_dir, "relion_reconstruct", """
import signal
signal.signal(signal.SIGTERM, lambda *a: (open('stopped', 'w').close(), sys.exit(143)))
while True:
    time.sleep(0.1)
""")
    p = subprocess.Popen([sys.executable, RUNNER, "relion_reconstruct", "--o", "External/job001/", "--j", "1"])
    time.sleep(1)
    p.terminate()
    assert p.wait(timeout=30) == 128 + signal.SIGTERM
    assert (root / "External/job001/RELION_JOB_EXIT_ABORTED").exists()
    deadline = time.time() + 10
    while not (root / "stopped").exists() and time.time() < deadline:
        time.sleep(0.1)
    assert (root / "stopped").exists()  # the program got the signal too


def test_runner_mpi_and_stdout(job, monkeypatch):
    root, bin_dir = job
    tool(bin_dir, "fake_mpirun", "import subprocess\nprint('mpirun', args[:2])\nsys.exit(subprocess.call(args[2:]))")
    tool(bin_dir, "relion_reconstruct_mpi", "print('MPI', ' '.join(args))")
    monkeypatch.setenv("RELION_MPIRUN", "fake_mpirun")
    r = run_runner("relion_reconstruct", "--o", "External/job001/", "--mpi", "3", "--j", "1", "--i", "a.star")
    assert r.returncode == 0, r.stderr
    assert "mpirun ['-n', '3']" in r.stdout and "MPI --i a.star --pipeline_control External/job001/.cli/" in r.stdout
    tool(bin_dir, "relion_star_printtable", "print('row', ' '.join(args))")
    r = run_runner("relion_star_printtable", "--o", "External/job001/", "--stdout", "table.txt", "--j", "1",
                   "a.star", "data_particles")
    assert r.returncode == 0, r.stderr
    assert (root / "External/job001/table.txt").read_text() == "row a.star data_particles\n"  # no pipeline_control
    assert "External/job001/table.txt ProcessData.txt" in (root / "External/job001/RELION_OUTPUT_NODES.star").read_text()


def test_runner_keeps_to_the_local_mpi_limit(job, monkeypatch):
    """CLI tools run MPI locally: no more processes than RELION allows for local jobs."""
    root, bin_dir = job
    tool(bin_dir, "relion_reconstruct_mpi", "print('MPI')")
    monkeypatch.setenv("RELION_MPIRUN", "fake_mpirun")
    monkeypatch.setenv("RELION_ERROR_LOCAL_MPI", "4")
    r = run_runner("relion_reconstruct", "--o", "External/job001/", "--mpi", "5", "--j", "1", "--i", "a.star")
    assert r.returncode == 2 and "RELION_ERROR_LOCAL_MPI environment variable (4)" in r.stderr
    assert (root / "External/job001/RELION_JOB_EXIT_FAILURE").exists()
    monkeypatch.delenv("RELION_ERROR_LOCAL_MPI")
    assert cli_runner.local_mpi_limit() == 32


def test_runner_kills_a_program_that_ignores_the_abort(monkeypatch):
    sent = []
    monkeypatch.setattr(cli_runner.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    grace, kill = cli_runner.ABORT_GRACE_S, cli_runner.KILL_GRACE_S
    signals = cli_runner.stop(7, 100.0, [], 100.0 + grace / 2)
    assert signals == [] and sent == []
    signals = cli_runner.stop(7, 100.0, signals, 101.0 + grace)
    signals = cli_runner.stop(7, 100.0, signals, 102.0 + grace)  # once
    assert sent == [(7, cli_runner.signal.SIGTERM)]
    signals = cli_runner.stop(7, 100.0, signals, 101.0 + grace + kill)
    signals = cli_runner.stop(7, 100.0, signals, 200.0 + grace + kill)
    assert sent == [(7, cli_runner.signal.SIGTERM), (7, cli_runner.signal.SIGKILL)] and len(signals) == 2


def test_runner_rejects_a_command_not_from_relion(job):
    r = run_runner("relion_image_handler", "--i", "x")
    assert r.returncode == 2 and "unexpected argument" in r.stderr


def test_runner_always_leaves_an_exit_file(job):
    """Whatever goes wrong, RELION gets RELION_JOB_EXIT_FAILURE (else it shows the job running for ever)."""
    root, bin_dir = job
    tool(bin_dir, "relion_star_printtable", "print('row')")
    failure = root / "External/job001/RELION_JOB_EXIT_FAILURE"
    r = run_runner("relion_star_printtable", "--o", "External/job001/", "--stdout", "no/such/dir.txt", "--j", "1", "a")
    assert r.returncode == 2 and "cannot write the output to no/such/dir.txt" in r.stderr and failure.exists()
    failure.unlink()
    r = run_runner("relion_star_printtable", "--o", "External/job001/", "--mpi", "two", "--j", "1", "a")
    assert r.returncode == 2 and "not a whole number" in r.stderr and failure.exists()
    failure.unlink()
    (root / "External/job001/RELION_OUTPUT_NODES.star").mkdir()  # cannot be written: an unexpected error
    r = run_runner("relion_star_printtable", "--o", "External/job001/", "--j", "1", "a")
    assert r.returncode == 1 and "Traceback" in r.stderr and failure.exists()


def test_runner_is_the_projects_copy(tmp_path):
    """Jobs name the runner relative to the project, so they outlive a moved or updated cryo-savanna."""
    assert runner_command("relion_image_handler") == f"python3 {RUNNER_IN_PROJECT} relion_image_handler"
    dst = install_runner(tmp_path)
    assert dst == tmp_path / RUNNER_IN_PROJECT and dst.read_bytes() == Path(RUNNER).read_bytes()
    dst.write_text("old")
    install_runner(tmp_path)
    assert dst.read_bytes() == Path(RUNNER).read_bytes()  # an older copy is replaced


def test_continuing_a_cli_job_puts_the_runner_into_the_project(tmp_path):
    from cryo_savanna.project.pipeliner import Pipeliner
    (tmp_path / "External/job001").mkdir(parents=True)
    pl = Pipeliner(tmp_path, exe="/usr/bin/true", environ={"PATH": "/usr/bin:/bin"})
    plain = JobDefs.load().variant("relion.external").new_jobstar(ENV)
    plain.options["fn_exe"] = "my_script.sh"
    pl.continue_job("External/job001/", plain)
    assert not (tmp_path / RUNNER_IN_PROJECT).exists()  # only CLI tool jobs need it
    pl.continue_job("External/job001/", CATALOG.variant("relion_image_handler").new_jobstar(ENV))
    assert (tmp_path / RUNNER_IN_PROJECT).read_bytes() == Path(RUNNER).read_bytes()


def test_cli_tool_jobs_end_to_end_with_a_stand_in_pipeliner(tmp_path, monkeypatch):
    """The validation's CLI tool check (tools/validate.py), against a pipeliner that builds External job
    commands as RELION does: the job runs through the project's runner, registers its output, and a
    second job on that output is linked to the first. On a real RELION: tests/test_against_relion.py."""
    from cryo_savanna.tools.validate import PipelinerChecks
    from tests.fixtures.fake_pipeliner import install as install_fake_pipeliner
    install_fake_pipeliner(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    checks = PipelinerChecks(tmp_path / "out", JobDefs.load())
    checks.check("cli_tool", checks.t_cli_tool)
    result = checks.results["cli_tool"]
    assert result["ok"], result.get("error")
    first, second = result["jobs"]
    assert result["parents_of_second"] == [first]
    assert result["relion_command"].startswith(f"python3 {RUNNER_IN_PROJECT} relion_image_handler --o {first} --j 1 ")
    assert result["relion_command"].endswith(f"--invert_hand --pipeline_control {first}")
