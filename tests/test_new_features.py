# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Time left of running jobs, out-of-date jobs, continuing after the SLURM time limit, the submission script
preview and the local GPU queue."""
import os
import time
from pathlib import Path

import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Process
from cryo_savanna.project.snapshot import JobInfo
from cryo_savanna.project.summary import JobSummary, SummaryReader


# ---------------------------------------------------------------------------- time left
def _iterations(job: Path, files: list[tuple[str, float]]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    for name, t in files:
        (job / name).write_text("")
        os.utime(job / name, (t, t))


def test_a_running_class3d_says_how_long_it_has_left(tmp_path):
    job = tmp_path / "Class3D" / "job010"
    now = time.time()
    # the first run's iterations, days before: not the pace of the continued run
    _iterations(job, [(f"run_it{i:03d}_model.star", now - 900_000 + 60 * i) for i in range(6)])
    _iterations(job, [(f"run_ct5_it{i:03d}_model.star", now - 600 * (10 - i)) for i in range(5, 11)])
    JobStar("relion.class3d", True, False, {"nr_iter": "25"}).write(job / "job.star")
    pr = Process("Class3D/job010/", "None", "relion.class3d", "Running")
    s = SummaryReader(tmp_path).summarize(JobInfo(pr, None, [], status="Running"), None)
    assert s.iteration == 10 and s.total_iterations == 25
    assert s.seconds_per_iteration == pytest.approx(600, abs=1)
    assert s.time_left(now) == pytest.approx(15 * 600, abs=2)
    assert "~2 h 30 min left" in s.progress(now)
    assert s.progress(now + 15 * 600) .endswith("finishing")


def test_vdam_files_every_few_iterations_and_refine3d_without_a_last_iteration(tmp_path):
    now = time.time()
    job = tmp_path / "Class2D" / "job011"
    _iterations(job, [(f"run_it{i:03d}_model.star", now - 100 * (100 - i)) for i in (0, 10, 20, 30)])
    JobStar("relion.class2d", False, False, {"do_grad": "Yes", "nr_iter_grad": "100"}).write(job / "job.star")
    s = SummaryReader(tmp_path).summarize(
        JobInfo(Process("Class2D/job011/", "None", "relion.class2d", "Running"), None, [], status="Running"), None)
    assert s.seconds_per_iteration == pytest.approx(100, abs=1) and s.time_left(now) == pytest.approx(0, abs=2)
    ref = tmp_path / "Refine3D" / "job012"
    _iterations(ref, [(f"run_it{i:03d}_half1_model.star", now - 45 * (4 - i)) for i in range(5)])
    JobStar("relion.refine3d", False, False, {}).write(ref / "job.star")
    s = SummaryReader(tmp_path).summarize(
        JobInfo(Process("Refine3D/job012/", "None", "relion.refine3d", "Running"), None, [], status="Running"), None)
    assert s.time_left(now) is None and s.progress(now).endswith("~45 s/iter")
    assert JobSummary(iteration=3).progress() == "iter 3"  # nothing written yet to tell the pace


# ---------------------------------------------------------------------------- out of date
def _chain(tmp_path: Path, t0: float):
    """Import → Extract → Class2D → Select, each job finished a minute after the one before."""
    from cryo_savanna.project.pipeline import Node, Pipeline
    p = Pipeline(job_counter=5)
    jobs = [("Import/job001/", "relion.importtomo", "movies.star"), ("Extract/job002/", "relion.extract", "particles.star"),
            ("Class2D/job003/", "relion.class2d", "run_it025_optimiser.star"), ("Select/job004/", "relion.select",
                                                                                "particles.star")]
    prev = None
    for i, (name, label, out) in enumerate(jobs):
        pr = Process(name, "None", label, "Succeeded")
        node = name + out
        p.nodes[node] = Node(node, "ParticleGroupMetadata.star.relion", producer=name)
        pr.outputs.append(node)
        if prev is not None:
            pr.inputs.append(prev)
            p.nodes[prev].consumers.append(name)
        p.processes[name] = pr
        d = tmp_path / name
        d.mkdir(parents=True)
        t = t0 + 60 * i
        for f in (out, "RELION_JOB_EXIT_SUCCESS"):
            (d / f).write_text("")
            os.utime(d / f, (t, t))
        prev = node
    return p


def test_jobs_after_one_that_ran_again_are_out_of_date(tmp_path):
    from cryo_savanna.project.staleness import StalenessChecker, downstream
    t0 = time.time() - 3600
    p = _chain(tmp_path, t0)
    checker = StalenessChecker(tmp_path)
    assert checker.check(p) == {}
    # Extract continued: its particles written again (and its exit file) after Class2D ran
    for f in ("particles.star", "RELION_JOB_EXIT_SUCCESS"):
        os.utime(tmp_path / "Extract/job002" / f, (t0 + 600, t0 + 600))
    stale = checker.check(p)
    assert list(stale) == ["Class2D/job003/", "Select/job004/"]
    assert stale["Class2D/job003/"].changed == ("Extract/job002/particles.star",)
    assert "Extract/job002 was written again" in stale["Class2D/job003/"].text().replace("input from ", "")
    assert stale["Select/job004/"].upstream == "Class2D/job003/"
    assert downstream(p, "Class2D/job003/", stale) == ["Class2D/job003/", "Select/job004/"]
    assert downstream(p, "Select/job004/", stale) == ["Select/job004/"]
    # Class2D run again since: up to date, and so is what depends on it... but Select ran before that
    for f in ("run_it025_optimiser.star", "RELION_JOB_EXIT_SUCCESS"):
        os.utime(tmp_path / "Class2D/job003" / f, (t0 + 700, t0 + 700))
    stale = checker.check(p)
    assert list(stale) == ["Select/job004/"] and stale["Select/job004/"].changed


def test_a_job_whose_input_job_is_overwritten_is_out_of_date(tmp_path):
    from cryo_savanna.project.staleness import StalenessChecker
    p = _chain(tmp_path, time.time() - 3600)
    (tmp_path / "Extract/job002/particles.star").unlink()  # its files in Trash/, the new job scheduled
    p.processes["Extract/job002/"].status = "Scheduled"
    stale = StalenessChecker(tmp_path).check(p)
    assert stale["Class2D/job003/"].missing == ("Extract/job002/particles.star",)
    assert "being run again" in stale["Class2D/job003/"].text()
    p.processes["Extract/job002/"].status = "Succeeded"  # finished, the file missing: cannot tell why
    assert "Class2D/job003/" not in StalenessChecker(tmp_path).check(p)


def test_interactive_jobs_are_not_run_again_on_their_own():
    from cryo_savanna.project.pipeliner import rerun_refused
    assert rerun_refused(JobStar("relion.manualpick", False, False, {}))
    assert rerun_refused(JobStar("relion.select", False, False, {"fn_model": "Class2D/job003/run_it025_optimiser.star"}))
    assert not rerun_refused(JobStar("relion.select", False, False, {"fn_model": "x", "do_split": "Yes"}))
    assert not rerun_refused(JobStar("relion.class2d", False, False, {}))


def _fake_project(tmp_path: Path):
    """A project, a stand-in relion_pipeliner (tests/fixtures/fake_pipeliner.py) and an External job's script."""
    import stat

    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.pipeliner import Pipeliner
    from cryo_savanna.project.star import write_star
    from cryo_savanna.tools.validate import EXTERNAL_SCRIPT
    from tests.fixtures.fake_pipeliner import install
    install(tmp_path / "bin")
    project = tmp_path / "p"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    script = project / "ext.sh"
    script.write_text(EXTERNAL_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    environ = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return project, Pipeliner(project, exe=str(tmp_path / "bin/relion_pipeliner"), environ=environ)


def _external(project: Path, in_mic: str = "") -> JobStar:
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": str(project / "ext.sh"), "do_queue": "No", "nr_threads": "1", "in_mic": in_mic,
                       "param1_label": "sleep", "param1_value": "0"})
    return js


def _wait(project: Path, jobs: list[str], timeout: float = 60) -> None:
    from cryo_savanna.project.pipeline import Pipeline
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = Pipeline.read(project)
        if all(p.processes[j].status in ("Succeeded", "Failed", "Aborted") for j in jobs):
            return
        time.sleep(0.2)
    raise AssertionError({j: Pipeline.read(project).processes[j].status for j in jobs})


def test_out_of_date_jobs_are_run_again_in_order(tmp_path, monkeypatch):
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.staleness import StalenessChecker
    monkeypatch.setenv("CRYO_SAVANNA_POLL_S", "0.05")
    project, pl = _fake_project(tmp_path)
    a = pl.add_job(_external(project)).job
    b = pl.add_job(_external(project, f"{a}micrographs.star"), alias="second").job
    pl.run([a, b], sec_wait_after=1)
    _wait(project, [a, b])
    p = Pipeline.read(project)
    assert p.processes[b].status == "Succeeded" and StalenessChecker(project).check(p) == {}
    later = time.time() + 30
    os.utime(project / a / "micrographs.star", (later, later))  # A's output written again
    assert list(StalenessChecker(project).check(p)) == [b]
    jobs, rec = pl.rerun([b])
    assert jobs == [b] and rec.jobs == [b]
    _wait(project, [b])
    p = Pipeline.read(project)
    assert p.processes[b].status == "Succeeded" and p.processes[b].alias.endswith("second/")
    assert (project / "Trash" / b / "job.star").exists()  # the old run, as for an overwrite
    os.utime(project / a / "micrographs.star", None)
    assert StalenessChecker(project).check(p) == {}


def test_jobs_run_again_come_after_the_jobs_they_read_from(tmp_path):
    from cryo_savanna.project.staleness import dependency_order
    p = _chain(tmp_path, time.time())
    # Class2D moved to the end of the pipeline (as after an undelete): it still runs before Select
    p.processes["Class2D/job003/"] = p.processes.pop("Class2D/job003/")
    assert dependency_order(p, ["Select/job004/", "Class2D/job003/", "Extract/job002/"]) == \
        ["Extract/job002/", "Class2D/job003/", "Select/job004/"]


# ---------------------------------------------------------------------------- the local GPU queue
def test_the_gpus_a_job_takes_on_this_machine():
    from cryo_savanna.project.gpu_queue import GpuQueue, job_gpu_ids
    assert job_gpu_ids(JobStar("relion.class3d", False, False, {"use_gpu": "Yes", "gpu_ids": "0:1,2:3"})) == [0, 1, 2, 3]
    assert job_gpu_ids(JobStar("relion.class3d", False, False, {"use_gpu": "No", "gpu_ids": "0"})) is None
    assert job_gpu_ids(JobStar("relion.class3d", False, False, {"use_gpu": "Yes", "do_queue": "Yes"})) is None
    assert job_gpu_ids(JobStar("relion.refine3d", False, False, {"use_gpu": "Yes", "gpu_ids": ""})) == []  # every GPU
    assert job_gpu_ids(JobStar("relion.motioncorr", False, False, {"do_own_motioncor": "No", "gpu_ids": "1"})) == [1]
    assert job_gpu_ids(JobStar("relion.motioncorr", False, False, {"do_own_motioncor": "Yes", "gpu_ids": "1"})) is None
    assert job_gpu_ids(JobStar("relion.autopick", False, False, {"use_gpu": "Yes", "do_log": "Yes"})) is None
    assert job_gpu_ids(JobStar("relion.postprocess", False, False, {})) is None
    q = GpuQueue(count=lambda: 4)
    two = [JobStar("relion.class2d", False, False, {"use_gpu": "Yes", "gpu_ids": "1"}),
           JobStar("relion.class3d", False, False, {"use_gpu": "Yes", "gpu_ids": "0:1"})]
    assert q.gpus_of(two, {}) == ["1", "0"]
    assert q.gpus_of(two, {"CUDA_VISIBLE_DEVICES": "2,3"}) == ["3", "2"]  # as the machine numbers them
    every = [JobStar("relion.refine3d", False, False, {"use_gpu": "Yes", "gpu_ids": ""})]
    assert q.gpus_of(every, {}) == ["0", "1", "2", "3"] and q.gpus_of(every, {"CUDA_VISIBLE_DEVICES": "5"}) == ["5"]


def test_two_commands_on_one_gpu_run_one_after_the_other(tmp_path):
    import subprocess
    import sys

    from cryo_savanna.project.gpu_queue import WAITER
    code = ("import sys, time; open(sys.argv[1], 'a').write('%f start\\n' % time.time()); time.sleep(1.0); "
            "open(sys.argv[1], 'a').write('%f end\\n' % time.time())")
    procs = []
    for name in ("a", "b"):
        procs.append(subprocess.Popen([sys.executable, "-I", str(WAITER), "--locks", str(tmp_path / "locks"), "--gpus", "0",
                                       "--label", name, "--", sys.executable, "-c", code, str(tmp_path / name)],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        time.sleep(0.3)
    outs = [p.communicate(timeout=60)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    (a0, a1), (b0, b1) = ([float(line.split()[0]) for line in (tmp_path / n).read_text().splitlines()] for n in "ab")
    assert b0 >= a1 - 0.05  # b started once a had ended and let go of GPU 0
    assert "waiting: GPU 0 in use by a" in outs[1]


def test_a_run_waits_for_its_gpus_and_can_be_stopped(tmp_path, monkeypatch):
    from cryo_savanna.project.gpu_queue import GpuQueue
    from cryo_savanna.project.pipeline import Pipeline
    monkeypatch.setenv("CRYO_SAVANNA_POLL_S", "0.05")
    project, pl = _fake_project(tmp_path)
    q = GpuQueue(tmp_path / "locks")
    monkeypatch.setattr(q, "gpus_of", lambda jobstars, environ: ["0"])  # as if they all took GPU 0
    pl.gpu_queue = q
    long = _external(project)
    long.options["param1_value"] = "3"
    a = pl.add_job(long).job
    b = pl.add_job(_external(project)).job
    c = pl.add_job(_external(project)).job
    ra = pl.run([a], sec_wait_after=1)
    deadline = time.time() + 20
    while Pipeline.read(project).processes[a].status != "Running" and time.time() < deadline:
        time.sleep(0.1)
    rb, rc = pl.run([b], sec_wait_after=1), pl.run([c], sec_wait_after=1)
    assert ra.gpus == rb.gpus == ["0"]
    time.sleep(0.5)
    held = pl.held_jobs()
    assert held[b].waiting_for_gpus and held[b].stoppable and "waiting for GPU 0" in held[b].describe()
    assert not held[a].waiting_for_gpus
    assert {r.id for r, kind in pl.stoppable_runs()} >= {rb.id, rc.id}
    pl.end_run(rc)  # stopped while it waits: its job stays scheduled
    assert rc.is_alive() is False and Pipeline.read(project).processes[c].status == "Scheduled"
    _wait(project, [a, b])
    p = Pipeline.read(project)
    assert p.processes[a].status == p.processes[b].status == "Succeeded"
    assert os.path.getmtime(project / b / "run.out") >= os.path.getmtime(project / a / "RELION_JOB_EXIT_SUCCESS") - 1


# ---------------------------------------------------------------------------- submission script preview
TEMPLATE = """#!/bin/bash
#SBATCH --job-name=XXXnameXXX
#SBATCH --partition=XXXqueueXXX
#SBATCH --ntasks=XXXmpinodesXXX
#SBATCH --cpus-per-task=XXXthreadsXXX
#SBATCH --gres=gpu:XXXextra1XXX
#SBATCH --time=XXXextra2XXX
#SBATCH --exclusive
#SBATCH -o XXXoutfileXXX
#SBATCH -e XXXerrfileXXX
srun --mpi=pmix XXXcommandXXX
"""


def _sinfo(cmd):
    if cmd[-1] == "%P":
        return "gpu*\ncpu\n"
    return "2-00:00:00\n" if cmd[cmd.index("-p") + 1] == "gpu" else "infinite\n"


def _values(**kw):
    v = {"nr_mpi": "5", "nr_threads": "6", "min_dedicated": "24", "queuename": "gpu", "qsub": "sbatch",
         "qsub_extra1": "4", "qsub_extra2": "1-12:00:00", "use_gpu": "Yes", "gpu_ids": "0:1:2:3"}
    v.update(kw)
    return v


def test_the_submission_script_is_filled_in_as_relion_does():
    from cryo_savanna.project.submit_script import COMMAND_PLACEHOLDER, fill
    script = fill(TEMPLATE, _values(), "Refine3D/job012/", 2)
    assert "#SBATCH --job-name=Refine3D/job012/" in script and "#SBATCH --ntasks=5" in script
    assert "--cpus-per-task=6" in script and "--gres=gpu:4" in script and "--time=1-12:00:00" in script
    assert "-o Refine3D/job012/run.out" in script and "-e Refine3D/job012/run.err" in script
    assert f"srun --mpi=pmix {COMMAND_PLACEHOLDER}" in script
    two = fill("XXXnodesXXX XXXdedicatedXXX XXXcoresXXX\nmpirun XXXcommandXXX", _values(), "J/job001/", 2, ["a", "b"])
    assert two.splitlines() == ["2 24 30", "mpirun a", "mpirun b"]  # one line per command, as RELION


def test_what_is_wrong_with_a_submission_script():
    from cryo_savanna.project.submit_script import check, fill, slurm_seconds
    which = {"sbatch": "/usr/bin/sbatch", "sinfo": "/usr/bin/sinfo"}.get

    def problems(gpu_ids=(0, 1, 2, 3), extra_count=2, template=TEMPLATE, **kw):
        v = _values(**kw)
        script = fill(template, v, "Refine3D/job012/", extra_count)
        return check(script, template, v, list(gpu_ids) if gpu_ids is not None else None, _sinfo, which)

    found, notes = problems()
    assert found == [] and notes == []  # 5 × 6 = 30 cores: no XXXnodesXXX / XXXdedicatedXXX in this template
    assert any("30 cores is not a multiple of the 24" in n
               for n in problems(template=TEMPLATE + "#SBATCH --nodes=XXXnodesXXX\n")[1])
    assert any("Partition gpux does not exist" in p for p in problems(queuename="gpux")[0])
    assert any("longer than partition gpu allows (2-00:00:00)" in p for p in problems(qsub_extra2="3-00:00:00")[0])
    assert any("not a time SLURM reads" in p for p in problems(qsub_extra2="12h")[0])
    assert any("--time=: no value" in p for p in problems(qsub_extra2="")[0])
    assert any("Not filled in: XXXextra2XXX" in p for p in problems(extra_count=1)[0])
    assert any("uses GPUs 0, 1, 2, 3, but the script asks for 2" in p for p in problems(qsub_extra1="2")[0])
    no_gres = TEMPLATE.replace("#SBATCH --gres=gpu:XXXextra1XXX\n", "")
    assert any("asks for none" in p for p in problems(template=no_gres)[0])
    assert any("does not use" in n for n in problems(gpu_ids=None)[1])
    assert any("sbatchx is not on PATH" in p for p in problems(qsub="sbatchx")[0])
    assert (slurm_seconds("90"), slurm_seconds("1:30:00"), slurm_seconds("2-12"), slurm_seconds("UNLIMITED")) == \
        (5400, 5400, 216000, float("inf"))


def test_the_preview_has_the_commands_of_the_last_run(tmp_path):
    from cryo_savanna.project.submit_script import preview
    (tmp_path / "t.sh").write_text(TEMPLATE)
    job = tmp_path / "Refine3D/job012"
    job.mkdir(parents=True)
    (job / "note.txt").write_text(" ++++ Executing new job on Thu Mar  5 10:00:00 2026\n ++++ with the following "
                                  "command(s): \n`which relion_refine_mpi` --o Refine3D/job012/run --auto_refine\n ++++ \n")
    pv = preview(str(tmp_path / "t.sh"), _values(), "Refine3D/job012/", 2, tmp_path, [0, 1, 2, 3], _sinfo,
                 {"sbatch": "x", "sinfo": "y"}.get)
    assert "srun --mpi=pmix `which relion_refine_mpi` --o Refine3D/job012/run --auto_refine" in pv.script
    assert "last run" in pv.notes[0]
    assert preview(str(tmp_path / "none.sh"), _values(), "J/job001/", 2, tmp_path, None).problems[0].startswith(
        "The submission script")


def test_no_pace_from_files_written_together_and_never_more_left_than_the_iterations_take():
    s = JobSummary(iteration=25, total_iterations=25, seconds_per_iteration=600, iteration_written=time.time())
    assert s.time_left(time.time() - 86400) == 0 and s.progress().endswith("finishing")  # all iterations done
    s = JobSummary(iteration=5, total_iterations=25, seconds_per_iteration=60, iteration_written=time.time() + 86400)
    assert s.time_left() == 20 * 60  # the file server's clock a day ahead


def test_the_waiter_starts_with_another_pythons_environment(tmp_path):
    """RELION's module may set PYTHONHOME / PYTHONPATH for its own Python: the waiter still starts, and the
    command it becomes gets them."""
    import subprocess
    import sys

    from cryo_savanna.project.gpu_queue import GpuQueue
    cmd = GpuQueue(tmp_path / "locks").command(
        [sys.executable, "-I", "-c", "import os; print(os.environ['PYTHONHOME'])"], ["0"], "test")
    env = dict(os.environ, PYTHONHOME="/nonexistent/python2.7", PYTHONPATH="/nonexistent")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("/nonexistent/python2.7")


def test_only_relion_jobs_on_a_gpu_make_a_run_wait(monkeypatch):
    """nvidia-smi's compute processes count only when they are part of a RELION job (native or External)."""
    from cryo_savanna import gpu_wait as W
    table = {
        10: (1, "/bin/sh -c `which relion_refine_mpi` --o Class3D/job012/run >> Class3D/job012/run.out"),
        11: (10, "mpirun -n 5 relion_refine_mpi --o Class3D/job012/run"),
        12: (11, "/opt/relion/bin/relion_refine_mpi --o Class3D/job012/run --gpu 0"),
        20: (1, "python3 .cryo-savanna/aretomo_runner.py align --o External/job020/ --j 1"),
        21: (20, "/opt/AreTomo3/AreTomo3 -InPrefix External/job020/aretomo/TS_01"),  # started by the runner
        30: (1, "jupyter-lab"),
        31: (30, "/home/me/env/bin/python -m ipykernel_launcher"),  # someone's notebook
        40: (1, "/home/me/relion_tools/train.py"),  # "relion" in a path is not a RELION program
    }
    assert [p for p in table if W.is_relion_job(p, table)] == [10, 11, 12, 20, 21]
    smi = {"--query-gpu=index,uuid": "0, GPU-a\n1, GPU-b\n2, GPU-c\n",
           "--query-compute-apps=gpu_uuid,pid": "GPU-a, 12\nGPU-b, 31\nGPU-c, 21\nGPU-c, 40\n"}
    monkeypatch.setattr(W, "_nvidia_smi", lambda args: smi[args[0]])
    assert W.busy_gpus(lambda: table) == {"0", "GPU-a", "2", "GPU-c"}  # not GPU 1: only the notebook is on it


# ---------------------------------------------------------------------------- time left of other jobs
def test_relions_progress_bar_gives_the_time_left_of_the_step(tmp_path):
    job = tmp_path / "CtfRefine" / "job014"
    job.mkdir(parents=True)
    JobStar("relion.ctfrefine", False, False, {}).write(job / "job.star")
    out = job / "run.out"
    out.write_text(" + Fitting per-particle defocus ...\n000/??? sec ~~(,_,\">\r  1.50/  6.00 min ...........~~(,_,\">\r"
                   "  3.00/  6.00 min ......................~~(,_,\">")
    t = time.time() - 30
    os.utime(out, (t, t))
    info = JobInfo(Process("CtfRefine/job014/", "None", "relion.ctfrefine", "Running"), None, [], status="Running")
    s = SummaryReader(tmp_path).summarize(info, None)
    assert s.step and s.time_left() == pytest.approx(180 - 30, abs=2)
    assert s.progress_count() == "" and s.progress_time().endswith("~2 min left in this step")
    # the step ended, and something else was written since: no estimate
    out.write_text(out.read_text() + "\r  6.00/  6.00 min ..................................~~(,_,\">\n Done.\n")
    assert SummaryReader(tmp_path).summarize(info, None).time_left() is None
    out.write_text("  0.17/ 77.71 hrs ~~(,_,\">\n")  # as RELION writes long ones
    assert SummaryReader(tmp_path).summarize(info, None).time_left() == pytest.approx(77.54 * 3600, rel=1e-3)


def test_aretomo3_says_how_many_tilt_series_are_done_and_how_long_the_rest_take(tmp_path):
    from cryo_savanna.project.star import StarBlock, write_star
    job = tmp_path / "External" / "job020"
    job.mkdir(parents=True)
    rows = [["TS_01", "done", "600"], ["TS_02", "done", "0"], ["TS_03", "done", "1200"], ["TS_04", "running", "0"],
            ["TS_05", "running", "0"], ["TS_06", "pending", "0"], ["TS_07", "pending", "0"]]
    write_star(job / "aretomo3.star", [StarBlock("global", ["rlnTomoName", "savannaAretomoStatus",
                                                            "savannaAretomoSeconds"], rows)])
    t = time.time()
    os.utime(job / "aretomo3.star", (t, t))
    JobStar("relion.external", False, True, {}).write(job / "job.star")
    info = JobInfo(Process("External/job020/", "None", "relion.external", "Running"), None, [], status="Running")
    s = SummaryReader(tmp_path).summarize(info, None)
    assert s.progress_count() == "3/7 tilt series"  # TS_02 done before this run: not the pace
    assert s.time_left(t) == pytest.approx((2 + 0.5 * 2) * 900 / 2, abs=1)  # two at a time, 15 min each
    assert not s.step and s.progress_time(t).endswith("~22 min left")


def test_a_running_jobs_card_has_its_iteration_and_times_on_lines_of_their_own(qtbot):
    from cryo_savanna.project.snapshot import JobInfo as Info
    from cryo_savanna.project.summary import progress_lines
    from cryo_savanna.ui.job_graph import CardFonts, card_progress_lines, card_size
    from qtpy.QtWidgets import QWidget
    fonts = CardFonts(QWidget().font())
    run = Info(Process("Class3D/job009/", "None", "relion.class3d", "Running"), None, [], status="Running")
    done = Info(Process("Class3D/job009/", "None", "relion.class3d", "Succeeded"), None, [], status="Succeeded")
    ctf = Info(Process("CtfRefine/job014/", "None", "relion.ctfrefine", "Running"), None, [], status="Running")
    assert (card_progress_lines(run), card_progress_lines(done), card_progress_lines(ctf)) == (2, 0, 1)
    assert card_size(run, {}, fonts)[1] == card_size(done, {}, fonts)[1] + 2 * fonts.line
    assert progress_lines("relion.external", "AreTomo3 alignment and reconstruction") == 2
    assert progress_lines("relion.external", "relion_image_handler") == 1
