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


def test_cache_keeps_the_last_results_and_makes_each_once():
    import threading

    from cryo_savanna.stamps import Cache, stamp
    c = Cache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # used: "b" is now the oldest
    c.put("c", 3)
    assert "b" not in c and c.get("a") == 1 and c.get("c") == 3 and len(c) == 2
    assert c.get("b") is None and c.get("b", 7) == 7
    made = []
    assert c.get_or_make("none", lambda: made.append(1)) is None  # None is a result too: kept
    assert c.get_or_make("none", lambda: made.append(1)) is None and made == [1]
    # two threads asking for the same slow result at once: made once
    slow, started = Cache(2), threading.Event()

    def make():
        started.set()
        made.append("slow")
        threading.Event().wait(0.2)
        return "result"

    got = []
    threads = [threading.Thread(target=lambda: got.append(slow.get_or_make("k", make, one_at_a_time=True)))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == ["result", "result"] and made.count("slow") == 1
    c.clear()
    assert len(c) == 0 and stamp(None) is None and stamp("/nonexistent/x") is None


def test_stamp_changes_with_the_file(tmp_path):
    from cryo_savanna.project.snapshot import stamp as old_name
    from cryo_savanna.stamps import stamp
    f = tmp_path / "f"
    assert stamp(f) is None
    f.write_text("a")
    st = stamp(f)
    assert st == (f.stat().st_mtime_ns, 1) and old_name is stamp
    f.write_text("ab")
    assert stamp(f) != st and stamp(str(f)) == stamp(f)


def test_uncaught_errors_are_kept_and_told(qtbot, tmp_path, capsys, monkeypatch):
    import sys
    import threading

    from cryo_savanna.ui.errors import ErrorReporter
    # Python's own hooks underneath, not pytest's (which would take the errors raised here for the test's)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    rep = ErrorReporter(log=tmp_path / "cache" / "errors.log")
    told = []
    rep.reported.connect(lambda text, log: told.append((text, log)))
    before = sys.excepthook, threading.excepthook
    rep.install()
    try:
        def clean():
            raise TypeError("unsupported operand\nsecond line")

        try:
            clean()
        except TypeError:
            sys.excepthook(*sys.exc_info())  # as Qt does for an exception in a slot
        thread = threading.Thread(target=lambda: 1 / 0)
        thread.start()
        thread.join()
        qtbot.waitUntil(lambda: len(told) == 2)
    finally:
        rep.uninstall()
    assert (sys.excepthook, threading.excepthook) == before
    assert told[0][0].startswith("TypeError: unsupported operand (test_audit.py:") and told[0][0].endswith(" in clean)")
    assert told[0][1] == str(rep.log) and told[1][0].startswith("ZeroDivisionError")
    text = rep.log.read_text()
    assert text.count("Traceback") == 2 and "second line" in text and "ZeroDivisionError" in text
    assert "TypeError" in capsys.readouterr().err  # still printed, as before
    # a log that cannot be written: still told
    blocked = ErrorReporter(log=tmp_path / "cache" / "errors.log" / "below a file")
    blocked.reported.connect(lambda text, log: told.append((text, log)))
    blocked.report(ValueError, ValueError("x"), None)
    assert told[-1] == ("ValueError: x", "")


def test_names_that_look_like_numbers_stay_as_written(tmp_path, qtbot):
    from cryo_savanna.results.tables import Table, read_compact
    from cryo_savanna.viewers.star_table import TableModel
    star = tmp_path / "t.star"
    star.write_text("data_global\n\nloop_\n_rlnTomoName #1\n_rlnTomoSizeX #2\n_rlnNote #3\n"
                    "010 4000 a\n002 4000.0 b\n001 0500 c\n")
    for t in (read_compact(star, "global"), Table.read(star, "global")):
        assert list(t.column("rlnTomoName")) == ["010", "002", "001"] and t.numeric("rlnTomoName") is None
        assert t.numeric("rlnTomoSizeX").tolist() == [4000.0, 4000.0, 500.0]  # numbers are still numbers
        assert "rlnTomoName" not in t.numeric_columns()
        model = TableModel()
        model.set_table(t)
        assert [model.data(model.index(r, 1)) for r in range(3)] == ["010", "002", "001"]
        assert [model.data(model.index(r, 2)) for r in range(3)] == ["4000", "4000", "500"]
        assert model.sort_key(1).tolist() == [10.0, 2.0, 1.0]  # sorted by number, shown as written
        assert model.sort_key(3).tolist() == ["a", "b", "c"]


def test_picks_are_counted_once_per_coordinate_file(tmp_path, monkeypatch):
    from cryo_savanna.results import micrographs as M
    coords = tmp_path / "AutoPick" / "job001"
    coords.mkdir(parents=True)
    rows = []
    for i, n in enumerate((3, 0, 5)):
        f = coords / f"m{i}_autopick.star"
        f.write_text("data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n" + "1 2\n" * n)
        rows.append(f"m{i}.mrc AutoPick/job001/m{i}_autopick.star")
    rows.append("m3.mrc AutoPick/job001/gone_autopick.star")
    (coords / "autopick.star").write_text("data_coordinate_files\n\nloop_\n_rlnMicrographName #1\n"
                                          "_rlnMicrographCoordinates #2\n" + "\n".join(rows) + "\n")
    opened = []
    real = M.loop_rows
    monkeypatch.setattr(M, "loop_rows", lambda path, *a, **k: (opened.append(Path(path).name), real(path, *a, **k))[1])
    M._COUNTS.clear()

    def counts():
        ps = M.read_picks(tmp_path, "AutoPick/job001/autopick.star")  # the list read again: a new PickSet
        return [ps.count(m) for m in ps.micrographs]

    assert counts() == [3, 0, 5, 0] and len(opened) == 3  # the file that is gone is not opened
    opened.clear()
    assert counts() == [3, 0, 5, 0] and opened == []
    (coords / "m1_autopick.star").write_text("data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n1 2\n1 2\n")
    assert counts() == [3, 2, 5, 0] and opened == ["m1_autopick.star"]


def test_every_kind_of_external_job_is_known_in_one_place(tmp_path):
    """jobdefs/external_jobs.py: what pipeliner, lineage, snapshot, the forms and the viewers ask."""
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs import external_jobs as X
    from cryo_savanna.jobdefs.cli_tools import external_program
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.results import registry as R
    env, defs = RelionEnv({}), JobDefs.load()
    forms = X.Forms(defs, env, FlagSource(which=lambda program: None))  # the bundled --help texts
    made = {
        X.CLI: ("relion_image_handler", {"--i": "Refine3D/job010/run_class001.mrc"}, None, "cli_runner.py"),
        X.PYTOM: ("match", {"in_tomograms": "./Tomograms/job006/tomograms.star"}, "Tomograms/job006/tomograms.star",
                  "pytom_runner.py"),
        X.ARETOMO: ("align", {"in_tiltseries": "CtfFind/job003/tilt_series_ctf.star"},
                    "CtfFind/job003/tilt_series_ctf.star", "aretomo_runner.py"),
        X.TOMOMASK: ("masks", {"in_tomograms": "Tomograms/job006/tomograms.star"}, "Tomograms/job006/tomograms.star",
                     "mask_runner.py"),
    }
    assert set(made) == {f.name for f in X.FAMILIES} == set(forms.catalogs)
    for family, (kind, values, main, runner) in made.items():
        variant = forms.variant(family, kind)
        assert variant is not None and getattr(forms, family) is forms.catalogs[family]
        js = variant.make_jobstar({**variant.default_values(env), **values}, False, "External/job001/", None)
        found = X.kind_of(js)
        assert found is not None and (found[0].name, found[1]) == (family, kind) and X.family_of(js) == family
        assert X.main_input(js) == main
        assert forms.variant_for(js) is variant
        title = X.program_title(js.options["fn_exe"])
        assert title == external_program(js.options["fn_exe"]) == found[0].title(kind) and title
        project = tmp_path / family
        assert X.install_runner(js, project) == project / ".cryo-savanna" / runner
        assert (project / ".cryo-savanna" / runner).is_file()
        # its viewers: the family's own (none for a CLI tool), then the files
        views = [v.title for v in R.views_for(R.JobContext(project, "External/job001/", "relion.external", "Succeeded",
                                                           [], js.options))]
        assert views[-1] == "Files"
        own = {X.PYTOM: "Scores", X.TOMOMASK: "Masks"}.get(family)  # (AreTomo3's come with its files)
        assert own is None or (own in views and family in R.EXTERNAL_VIEWS)
        assert family == X.CLI or family in R.EXTERNAL_VIEWS
    # any other job: none of this
    other = defs.variant("relion.external").new_jobstar(env)
    other.options["fn_exe"] = "/opt/tools/my_script.sh --fast"
    assert X.kind_of(other) is None and X.main_input(other) is None and forms.variant_for(other) is None
    assert X.install_runner(other, tmp_path / "none") is None and not (tmp_path / "none").exists()
    assert X.program_title(other.options["fn_exe"]) == "my_script.sh" and X.program_title("") == ""
    class3d = defs.variant("relion.class3d").new_jobstar(env)
    class3d.options["fn_exe"] = "python3 .cryo-savanna/pytom_runner.py match"  # not an External job: not ours
    assert X.kind_of(class3d) is None

