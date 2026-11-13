# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""A scratch directory of its own for every new job."""

import random

import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.scratch import (
    SCRATCH_KEY,
    assign,
    base_dir,
    number_of,
    own_dirs,
    taken_numbers,
    unique_dir,
    wipe,
)


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


def test_assign_skips_jobs_without_scratch(tmp_path):
    for value in ("", "  ", "$$scratch"):
        opts = {SCRATCH_KEY: value}
        assert assign(opts, tmp_path, None, []) is None and opts[SCRATCH_KEY] == value
    assert assign({}, tmp_path, None, []) is None


def test_numbers_of_the_projects_jobs_are_not_reused(spa_copy):
    p = Pipeline.read(spa_copy)
    job = "Class3D/job009/"
    js = JobStar.read(spa_copy / job / "job.star")
    js.options[SCRATCH_KEY] = "/scratch/me/savanna_123"
    js.write(spa_copy / job / "job.star")
    labels = [p.processes[job].type_label]
    assert taken_numbers(spa_copy, p, labels) == {123}
    assert taken_numbers(spa_copy, p, []) == set()  # only the job types with a scratch option are read
    # a label RELION added to (relion.class3d.tomo, relion.autopick.log) is still that job type
    p.processes[job].type_label = labels[0] + ".tomo"
    assert taken_numbers(spa_copy, p, labels) == {123}
    for _ in range(50):
        opts = {SCRATCH_KEY: "/scratch/me"}
        got = assign(opts, spa_copy, p, labels, exists=lambda d: False)
        assert got == opts[SCRATCH_KEY] and got.startswith("/scratch/me/savanna_") and number_of(got) != 123


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
