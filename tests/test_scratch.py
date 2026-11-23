# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""A scratch directory of its own for a job whose scratch directory another job uses."""

import random
from pathlib import Path

import pytest

from cryo_savanna.project import scratch
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeliner import Pipeliner
from cryo_savanna.project.scratch import (
    SCRATCH_KEY,
    InUse,
    base_dir,
    dirs_on_host,
    in_use,
    normalized,
    number_of,
    own_dirs,
    settle,
    unique_dir,
    wipe,
)
from tests.test_pipeliner import _live_run


def test_base_dir_strips_only_the_suffix_it_adds():
    assert base_dir("/scratch/me/savanna_042") == "/scratch/me"
    assert base_dir("/scratch/me/savanna_042/") == "/scratch/me"
    assert base_dir("/savanna_042") == "/"
    assert base_dir("/scratch/me/042") == "/scratch/me/042"  # the user's own directory
    assert base_dir("/scratch/me/savanna_0421") == "/scratch/me/savanna_0421"
    assert base_dir("") == ""
    assert number_of("/scratch/savanna_007") == 7 and number_of("/scratch/007") is None


def test_unique_dir_avoids_numbers_in_use_and_directories_on_disk(tmp_path):
    (tmp_path / "savanna_001").mkdir()
    taken = set(range(1000)) - {1, 2}
    assert unique_dir(str(tmp_path), taken) == str(tmp_path / "savanna_002")
    assert unique_dir(str(tmp_path) + "/", taken) == str(tmp_path / "savanna_002")
    # a cloned value: a new number next to it, not inside it
    assert unique_dir(str(tmp_path / "savanna_555"), taken) == str(tmp_path / "savanna_002")
    with pytest.raises(RuntimeError):
        unique_dir(str(tmp_path), taken | {2})
    picks = {unique_dir("/s", set(), exists=lambda p: False, rng=random.Random(i)) for i in range(20)}
    assert len(picks) > 1 and all(p.startswith("/s/savanna_") and len(p) == len("/s/savanna_000") for p in picks)


def _table(*commands: str):
    return lambda: {100 + i: (1, c) for i, c in enumerate(("ps -A -ww -o pid=,ppid=,args=",) + commands)}


def test_the_scratch_directories_of_this_machines_processes_are_found(tmp_path):
    (tmp_path / "ssd").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "ssd")
    table = _table(f"relion_refine_mpi --o Class2D/job010/run --scratch_dir {tmp_path}/link/ --j 4",
                   f"bash -c relion_refine --scratch_dir='{tmp_path}/other' >> Class3D/job011/run.out",
                   "relion_refine --scratch_dir=rel/dir --o x",
                   "relion_refine --o Refine3D/job012/run --pool 3")
    assert dirs_on_host(tmp_path / "project", table) == {
        str(tmp_path / "ssd"), str(tmp_path / "other"), str(tmp_path / "project/rel/dir")}
    assert dirs_on_host(tmp_path, lambda: {}) is None  # ps could not be read


def test_jobs_without_scratch_or_sent_to_a_queue_are_left_as_they_are(tmp_path):
    busy = InUse(unknown=True)
    for opts in ({SCRATCH_KEY: ""}, {SCRATCH_KEY: "  "}, {SCRATCH_KEY: "$$scratch"}, {},
                 {SCRATCH_KEY: "/ssd", "do_queue": "Yes"}, {SCRATCH_KEY: "/ssd/savanna_007", "do_queue": "Yes"}):
        before = dict(opts)
        assert settle(opts, tmp_path, busy) is None and opts == before


def test_a_free_directory_is_used_as_chosen(tmp_path):
    use = in_use(tmp_path, [{SCRATCH_KEY: "/elsewhere"}], _table("relion_refine --scratch_dir /ssd/savanna_001"))
    opts = {SCRATCH_KEY: "/ssd", "do_queue": "No"}
    assert settle(opts, tmp_path, use) is None and opts[SCRATCH_KEY] == "/ssd"
    opts = {SCRATCH_KEY: "/ssd/savanna_123"}  # left from an earlier start: the directory itself again
    assert settle(opts, tmp_path, use) is None and opts[SCRATCH_KEY] == "/ssd"


def test_a_directory_in_use_gets_a_savanna_folder(tmp_path):
    ssd = tmp_path / "ssd"
    use = in_use(tmp_path, [{SCRATCH_KEY: f"{ssd}/savanna_005"}], _table(f"relion_refine --scratch_dir {ssd}/"))
    for _ in range(30):
        opts = {SCRATCH_KEY: str(ssd)}
        got = settle(opts, tmp_path, InUse(set(use.dirs), set(use.numbers)), exists=lambda d: False)
        assert got == opts[SCRATCH_KEY] and base_dir(got) == str(ssd) and number_of(got) not in (None, 5)
    # its own savanna_NNN from before is kept when free; another job's is not
    opts = {SCRATCH_KEY: f"{ssd}/savanna_042"}
    assert settle(opts, tmp_path, InUse(set(use.dirs), set(use.numbers))) == f"{ssd}/savanna_042"
    opts = {SCRATCH_KEY: f"{ssd}/savanna_005"}
    assert number_of(settle(opts, tmp_path, InUse(set(use.dirs), set(use.numbers)), exists=lambda d: False)) != 5
    # two jobs started together that both need one get two
    both = InUse(set(use.dirs), set(use.numbers))
    a, b = (settle({SCRATCH_KEY: str(ssd)}, tmp_path, both, exists=lambda d: False) for _ in range(2))
    assert a != b


def test_queued_jobs_of_the_project_do_not_count_and_an_unknown_host_counts_as_in_use(tmp_path):
    use = in_use(tmp_path, [{SCRATCH_KEY: "/ssd", "do_queue": "Yes"}], _table())
    assert settle({SCRATCH_KEY: "/ssd"}, tmp_path, use) is None
    use = in_use(tmp_path, [], lambda: {})
    assert use.unknown and settle({SCRATCH_KEY: "/ssd"}, tmp_path, use, exists=lambda d: False).startswith("/ssd/savanna_")


def test_jobs_get_their_scratch_directory_when_they_start(spa_copy, monkeypatch):
    """Pipeliner.run (here, what it does before relion_pipeliner starts): the job.star of a job whose scratch
    directory a job of a run still going has, or a process of this machine, is changed; a free one is kept."""
    def set_scratch(job, value, queue="No"):
        js = JobStar.read(spa_copy / job / "job.star")
        js.options.update({SCRATCH_KEY: value, "do_queue": queue})
        js.write(spa_copy / job / "job.star")
        return JobStar.read(spa_copy / job / "job.star")

    monkeypatch.setattr(scratch, "process_table", _table("relion_refine_mpi --scratch_dir /ssd/b --o x"))
    set_scratch("Refine3D/job016/", "/ssd/a")  # Scheduled, in a run still going (waits for its GPUs, say)
    set_scratch("Polish/job015/", "/ssd/d")  # Running
    proc = _live_run(spa_copy, "a", ["Refine3D/job016/"])
    try:
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        jobs = {"Class3D/job009/": set_scratch("Class3D/job009/", "/ssd/a/"),
                "Class2D/job006/": set_scratch("Class2D/job006/", "/ssd/b"),
                "InitialModel/job008/": set_scratch("InitialModel/job008/", "/ssd/d"),
                "Refine3D/job011/": set_scratch("Refine3D/job011/", "/ssd/c/savanna_001")}
        given = pl._own_scratch(jobs)
        assert sorted(given) == ["Class2D/job006/", "Class3D/job009/", "InitialModel/job008/"]
        assert [base_dir(given[j]) for j in sorted(given)] == ["/ssd/b", "/ssd/a", "/ssd/d"]
        for job, want in (*given.items(), ("Refine3D/job011/", "/ssd/c")):
            assert JobStar.read(spa_copy / job / "job.star").options[SCRATCH_KEY] == want
        queued = {"Class3D/job009/": set_scratch("Class3D/job009/", "/ssd/a", queue="Yes")}
        assert pl._own_scratch(queued) == {}
        assert JobStar.read(spa_copy / "Class3D/job009/job.star").options[SCRATCH_KEY] == "/ssd/a"
    finally:
        proc.kill()
        proc.wait()
    assert normalized("~/x", "/p") == normalized(str(Path.home() / "x"), "/")


def test_own_dirs_finds_only_savanna_folders_directly_in_the_scratch_directory(tmp_path):
    base = tmp_path / "ssd"
    (base / "savanna_001" / "relion_volatile").mkdir(parents=True)
    (base / "savanna_001" / "relion_volatile" / "particles.mrcs").write_bytes(b"x")
    (base / "savanna_002").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (base / "savanna_003").symlink_to(elsewhere)  # deleting through it would delete elsewhere
    (base / "savanna_004").write_text("a file")
    for other in ("savanna_1234", "savanna_ab1", "relion_volatile", "mine"):
        (base / other).mkdir()
    (base / "mine" / "savanna_005").mkdir()  # not directly in it
    found = own_dirs(base)
    assert found == [base / "savanna_001", base / "savanna_002"]
    assert own_dirs(tmp_path / "missing") == []
    assert wipe(found) == []
    assert sorted(x.name for x in base.iterdir()) == ["mine", "relion_volatile", "savanna_003", "savanna_004",
                                                      "savanna_1234", "savanna_ab1"]
    assert elsewhere.exists() and (base / "mine" / "savanna_005").exists()
    errors = wipe([base / "savanna_002"])  # gone already
    assert len(errors) == 1 and "savanna_002" in errors[0]
