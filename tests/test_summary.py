# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import shutil
import time

import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.star import StarError, loop_rows, read_star_head
from cryo_savanna.project.summary import SummaryReader, duration, parse_ctime, summary_lines


def summaries(project):
    snap = ProjectLoader(project).refresh()
    reader = SummaryReader(project)
    return {n: reader.summarize(i, snap.pipeline) for n, i in snap.jobs.items()}


def test_spa_summaries(spa_project):
    s = summaries(spa_project)
    assert s["Import/job001/"].lines() == ["24 movies", "0.885 Å/px · 200 kV"]
    assert s["MotionCorr/job002/"].result == ["24 micrographs"]
    assert s["MotionCorr/job002/"].quality[0].startswith("motion ")
    assert s["CtfFind/job003/"].quality[0].startswith("CTF max res ")
    assert s["AutoPick/job004/"].result[0].endswith("· 24 micrographs")
    assert s["AutoPick/job004/"].settings == ["LoG"]
    assert s["Extract/job005/"].lines() == ["960 particles", "box 256 → 64 px · 3.54 Å/px"]  # rescaled
    assert s["Class2D/job006/"].lines() == ["960 particles", "K=50 · T=2"]
    assert s["Select/job007/"].result == ["600 particles", "5 classes"]
    assert s["Class3D/job009/"].lines() == ["600 particles · 12.50 Å", "K=4 · T=4 · C1"]
    assert s["Refine3D/job011/"].lines() == ["288 particles · 6.00 Å", "D2"]
    assert s["PostProcess/job013/"].lines() == ["3.20 Å", "B-factor -112 Å²"]
    assert s["MaskCreate/job012/"].lines() == []


def test_tomo_summaries(tomo_project):
    s = summaries(tomo_project)
    assert s["Import/job001/"].lines() == ["5 tilt series", "0.675 Å/px"]
    assert s["CtfFind/job003/"].quality[0].startswith("CTF max res ")  # from the per-tilt-series files
    assert s["Tomograms/job006/"].lines() == ["5 tomograms", "10 Å/px"]
    assert s["Picks/job008/"].result == ["147 particles"]
    assert s["Extract/job009/"].lines() == ["135 particles", "box 640 → 64 px · 6.75 Å/px"]  # binning 10
    assert s["Refine3D/job011/"].settings == ["C6"]


def test_running_job_shows_progress(spa_copy):
    with PipelineLock(spa_copy) as lock:
        p = Pipeline.read(spa_copy)
        p.processes["Class3D/job009/"].status = "Running"
        p.write(spa_copy, lock=lock)
    (spa_copy / "Class3D/job009/RELION_JOB_EXIT_SUCCESS").unlink()  # still running on disk too
    s = summaries(spa_copy)
    c3d = s["Class3D/job009/"]
    assert (c3d.iteration, c3d.total_iterations) == (25, 25)
    assert c3d.started == parse_ctime("Tue Oct 28 10:00:00 2026")
    assert c3d.progress(now=c3d.started + 3725) == "iter 25/25 · 1 h 02 min"
    polish = s["Polish/job015/"]  # running, not iterative: elapsed time only
    assert polish.iteration is None and polish.progress(now=polish.started + 600) == "10 min"


def test_odd_iteration_counts_and_unexpected_errors_leave_an_empty_summary(spa_copy, monkeypatch, capsys):
    js = JobStar.read(spa_copy / "Class3D/job009/job.star")
    js.options["nr_iter"] = "inf"
    js.write(spa_copy / "Class3D/job009/job.star")
    with PipelineLock(spa_copy) as lock:
        p = Pipeline.read(spa_copy)
        p.processes["Class3D/job009/"].status = "Running"
        p.write(spa_copy, lock=lock)
    (spa_copy / "Class3D/job009/RELION_JOB_EXIT_SUCCESS").unlink()
    assert summaries(spa_copy)["Class3D/job009/"].total_iterations is None
    monkeypatch.setattr(SummaryReader, "_settings", lambda *a: 1 / 0)
    snap = ProjectLoader(spa_copy).refresh()
    reader = SummaryReader(spa_copy)
    assert all(reader.summarize(i, snap.pipeline).text() == "" for i in snap.jobs.values())
    assert "ZeroDivisionError" in capsys.readouterr().err
    assert not reader.needs_update(snap.jobs["Extract/job005/"])  # not tried again until the job changes


def test_finished_jobs_are_not_read_again(spa_project):
    snap = ProjectLoader(spa_project).refresh()
    reader = SummaryReader(spa_project)
    info = snap.jobs["Extract/job005/"]
    reader.summarize(info, snap.pipeline)
    assert not reader.needs_update(info)
    info.status = "Running"
    assert reader.needs_update(info)


def test_summary_lines_reserved_per_type():
    assert summary_lines("relion.motioncorr") == 3
    assert summary_lines("relion.class3d") == 2
    assert summary_lines("relion.maskcreate") == 0
    assert summary_lines("relion.autopick.log") == 2  # subtype labels count as their type


def test_time_helpers():
    assert parse_ctime("Tue Oct  7 09:05:03 2026") == time.mktime((2026, 10, 7, 9, 5, 3, 0, 0, -1))
    assert parse_ctime("garbage") is None
    assert duration(59) == "0 min"
    assert duration(3600 * 26 + 60) == "1 d 2 h"


def test_loop_rows_and_head(tmp_path):
    f = tmp_path / "x.star"
    f.write_text("data_general\n_rlnFinalResolution 3.2\n\ndata_parts\nloop_\n_rlnA #1\n_rlnB #2\n1 a\n2 'b c'\n\n"
                 "data_broken\nloop_\n_rlnA\n1 2 3\n")
    assert loop_rows(f, "general") is None  # a list block
    assert loop_rows(f, "parts") == (2, [])
    assert loop_rows(f, "parts", "rlnB") == (2, ["a", "b c"])
    assert loop_rows(f, "parts", "rlnC") is None
    assert loop_rows(f, "missing") is None
    assert read_star_head(f, ["general"])["general"].get("rlnFinalResolution") == "3.2"
    with pytest.raises(StarError):  # reading everything hits the broken block; the head read never does
        read_star_head(f, ["broken"])


def test_extract_before_output_uses_job_options(tomo_project, tmp_path):
    shutil.copytree(tomo_project, tmp_path / "tomo", symlinks=True)
    (tmp_path / "tomo/Extract/job009/particles.star").unlink()
    s = summaries(tmp_path / "tomo")["Extract/job009/"]
    assert s.settings == ["box 512 Å → 64 px", "8 Å/px"]  # from the job options


def test_extract_without_rescaling_shows_one_box(spa_copy):
    star = spa_copy / "Extract/job005/particles.star"
    star.write_text(star.read_text().replace("3.540000", "0.885000"))  # optics: 64 px at 0.885 Å/px
    js = JobStar.read(spa_copy / "Extract/job005/job.star")
    js.options.update(do_rescale="No", extract_size="64")
    js.write(spa_copy / "Extract/job005/job.star")
    s = summaries(spa_copy)["Extract/job005/"]
    assert s.settings == ["box 64 px", "0.885 Å/px"]
