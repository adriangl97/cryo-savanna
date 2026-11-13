# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fixes and speed-ups of the repository audit: each must give what the slower code gave."""
from pathlib import Path

import numpy as np
import pytest

from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lineage import primary_parents
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.star import format_row, loop_columns, loop_rows


def test_a_pass_over_the_pipeline_reads_each_jobstar_once(spa_project):
    p = Pipeline.read(spa_project)
    asked: list[Path] = []

    def read(path):
        asked.append(path)
        return JobStar.read(path)

    parents = primary_parents(p, spa_project, JobDefs.load(), read)
    assert parents == primary_parents(p, spa_project, JobDefs.load())
    assert len(asked) == len(set(asked)) == len(p.processes)


def test_a_missing_jobstar_is_asked_for_once_too(spa_project, tmp_path):
    p = Pipeline.read(spa_project)
    asked = []

    def read(path):
        asked.append(path)
        return JobStar.read(path)  # tmp_path has no job directories: OSError every time

    parents = primary_parents(p, tmp_path, JobDefs.load(), read)
    assert len(asked) == len(p.processes) and parents["PostProcess/job013/"] is not None


STAR = """
data_optics

loop_
_rlnOpticsGroup #1
1

data_particles

loop_
_rlnX #1
_rlnName #2
_rlnY #3
  1.0  a/b.mrc   2
# a comment
4 'quoted name' 5

loop_
_under score 6
7 short
8
9 n 10 # more
11 it's 12

data_after
loop_
_rlnX #1
99
"""


def test_columns_are_read_only_as_far_as_wanted(tmp_path):
    f = tmp_path / "p.star"
    f.write_text(STAR)
    cols, data = loop_columns(f, "particles", ["rlnY", "rlnX", "rlnNope"])
    assert cols == ["rlnX", "rlnName", "rlnY"]
    assert data == {"rlnY": ["2", "5", "6", "", "", "10", "12"], "rlnX": ["1.0", "4", "_under", "7", "8", "9", "11"]}
    assert loop_columns(f, "particles", ["rlnName"])[1]["rlnName"] == \
        ["a/b.mrc", "quoted name", "score", "short", "", "n", "it's"]
    assert loop_columns(f, "particles", ["rlnNope"]) == (cols, {})
    assert loop_rows(f, "particles") == (7, [])
    assert loop_rows(f, "particles", "rlnX") == (7, ["1.0", "4", "_under", "7", "8", "9", "11"])
    assert loop_rows(f, "particles", "rlnNope") is None


def test_rows_of_text_are_written_as_before():
    from cryo_savanna.project.star import format_value
    row = ["1.000000", "a b", "", "#x", "plain", "'q", 3, 1.5, None, True]
    assert format_row(row) == "".join(f"{format_value(v):>10} " for v in row) + "\n"
    assert format_row(["1.000000", "a b", ""]) == '  1.000000      "a b"         "" \n'
    assert format_row(["_label", "x"]).startswith('  "_label" ')  # a row must not read as a label


def test_filter_texts_are_those_shown():
    pytest.importorskip("qtpy")
    from cryo_savanna.viewers.star_table import _file_name, _number, _numbers
    rng = np.random.default_rng(0)
    odd = [np.nan, np.inf, -np.inf, 1e15, -1e15, 1e14, 0.5, -0.0, 1234567.8, 999999.5, 1e-5, 0.1 + 0.2]
    columns = [np.round(rng.uniform(0, 4000, 6000), 6), rng.integers(1, 50, 6000).astype(float),  # many / few values
               rng.choice(odd, 6000), np.array(odd), np.array([]), np.zeros(5000)]
    for values in columns:
        assert _numbers(values) == [_number(v) for v in values]
    for name in ("a/b.mrc", "b.mrc", "a/b/", "a/.", ".", "..", "a/..", "", "/", "a//b", "/abs/p.mrc", "x/."):
        assert _file_name(name) == Path(name).name


def test_an_image_that_cannot_be_read_gets_no_thumbnail(tmp_path):
    pytest.importorskip("qtpy")
    import mrcfile
    from cryo_savanna.viewers.gallery import make_thumbs
    with mrcfile.new(tmp_path / "map.mrc") as m:
        m.set_data(np.ones((8, 8, 8), np.float32))
    data = (tmp_path / "map.mrc").read_bytes()
    (tmp_path / "map.mrc").write_bytes(data[:1024 + 3 * 64 * 4])  # being written: its central section is not there yet
    assert make_thumbs(str(tmp_path / "map.mrc"), [None], 32) == {}


def test_mirror_warning_counts_a_template_used_as_it_is(tmp_path, monkeypatch, capsys):
    import mrcfile
    from cryo_savanna import pytom_runner as P
    for name in ("t.mrc", "m.mrc"):
        with mrcfile.new(tmp_path / name) as m:
            m.set_data(np.ones((8, 8, 8), np.float32))
            m.voxel_size = 10.0
    monkeypatch.chdir(tmp_path)
    opts = {"fn_template": "t.mrc", "fn_mask": "m.mrc", "mask_source": P.MASK_FILE, "do_prep_template": "No",
            "template_mirror": "Yes", "do_prep_mask": "No"}
    P.prepare_template_and_mask(None, opts, "job/", 10.0)  # nothing is prepared: nothing is mirrored
    assert "may not match" not in capsys.readouterr().out


def test_a_slab_is_averaged_as_before_one_section_at_a_time(tmp_path):
    import mrcfile

    from cryo_savanna.images.mrc import MrcReader
    rng = np.random.default_rng(3)
    for dtype in (np.float32, np.int16, np.float16):
        vol = (rng.normal(0, 40, (9, 12, 10))).astype(dtype)
        path = tmp_path / f"{np.dtype(dtype).name}.mrc"
        with mrcfile.new(path) as m:
            m.set_data(vol)
        with MrcReader(path) as r:
            for z0, z1 in ((0, 9), (2, 7), (4, 5), (8, 9)):
                want = r.planes(z0, z1).mean(axis=0)
                got = r.mean_planes(z0, z1)
                assert got.dtype == want.dtype == np.float32 and np.array_equal(got, want)


def test_a_lock_removed_by_hand_does_not_fail_the_change(tmp_path):
    """LockError tells the user to remove a stale lock: a change that held it meanwhile still ends well."""
    import shutil

    from cryo_savanna.project.lock import PipelineLock, is_locked
    with PipelineLock(tmp_path) as lock:
        assert is_locked(tmp_path)
        shutil.rmtree(lock.dir)
    assert not lock.held and not is_locked(tmp_path)
    with PipelineLock(tmp_path) as lock:  # the file only: the directory is still removed
        lock.file.unlink()
    assert not is_locked(tmp_path)


def test_working_directories_of_slurm_jobs_are_resolved_once(tmp_path, monkeypatch):
    import os

    from cryo_savanna.project import slurm
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    (tmp_path / "link").symlink_to(project)
    mapper = slurm.QueueMapper(project, persist=False)
    asked = []
    real = os.path.realpath
    monkeypatch.setattr(slurm.os.path, "realpath", lambda p, *a, **k: (asked.append(str(p)), real(p, *a, **k))[1])
    queue = [slurm.QueueJob(str(i), "RUNNING", name="Class2D/job001", workdir=str(w))
             for i, w in enumerate((project, tmp_path / "link", other, project, other))]
    for _ in range(3):
        found = mapper.map({"Class2D/job001/"}, queue)
        assert [mapper.here(q) for q in queue] == [True, True, False, True, False]
        assert found["Class2D/job001/"].id == "3"
    assert sorted(asked) == sorted(str(w) for w in (project, tmp_path / "link", other))



def test_micrographs_of_particles_are_found_as_numpy_finds_them():
    from cryo_savanna.results import ctfrefine as C
    from cryo_savanna.results.tables import Table
    rng = np.random.default_rng(5)
    names = [f"Mics/m{i:03d}.mrc" for i in rng.integers(0, 40, 3000)] + ["b.mrc", "B.mrc", "a b.mrc", "é.mrc", "b.mrc"]
    want_names, want_inv = np.unique(np.asarray(names, dtype=object).astype(str), return_inverse=True)
    got_names, got_inv = C._unique_names(names)
    assert got_names == [str(n) for n in want_names] and np.array_equal(got_inv, want_inv.reshape(-1))
    assert C._unique_names([])[0] == [] and len(C._unique_names([])[1]) == 0
    values = rng.normal(size=len(names))
    values[::7] = np.nan
    s = C.per_micrograph(Table("particles", ["rlnMicrographName", "v"], {"rlnMicrographName": names, "v": values}), ["v"])
    assert s.names == got_names and s.particles.sum() == len(names)
    k = got_names.index("b.mrc")
    mine = values[[i for i, n in enumerate(names) if n == "b.mrc"]]
    assert s.particles[k] == 2 and s.mean["v"][k] == pytest.approx(np.nanmean(mine))

