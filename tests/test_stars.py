# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Starred jobs: the project's stars.json and the loader (the window: test_ui.py)."""

import json

from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.stars import Stars


def test_stars_are_kept_in_the_project_and_shared(tmp_path):
    a, b = Stars(tmp_path), Stars(tmp_path)
    a.set("Refine3D/job011/", True)
    b.set("Class3D/job009/", True)  # another window: both stars kept
    assert Stars(tmp_path).starred() == {"Refine3D/job011/", "Class3D/job009/"}
    a.set("Class3D/job009/", False)
    assert b.starred() == {"Refine3D/job011/"}  # read again when the file changes
    assert json.loads((tmp_path / ".cryo-savanna" / "stars.json").read_text()) == {"jobs": ["Refine3D/job011/"]}
    (tmp_path / ".cryo-savanna" / "stars.json").write_text("{not json")
    assert Stars(tmp_path).starred() == set()


def test_readonly_stars_last_for_the_session(tmp_path):
    Stars(tmp_path).set("Refine3D/job011/", True)
    ro = Stars(tmp_path, persist=False)
    ro.set("Class3D/job009/", True)
    ro.set("Refine3D/job011/", False)
    assert ro.starred() == {"Class3D/job009/"}
    assert Stars(tmp_path).starred() == {"Refine3D/job011/"}


def test_a_star_is_picked_up_without_reading_the_pipeline_again(spa_copy):
    stars = Stars(spa_copy)
    loader = ProjectLoader(spa_copy, stars=stars)
    first = loader.refresh()
    assert not any(j.starred for j in first.jobs.values())
    assert loader.refresh() is None
    stars.set("PostProcess/job013/", True)
    snap = loader.refresh()
    assert snap is not None and snap.pipeline is first.pipeline
    assert [n for n, j in snap.jobs.items() if j.starred] == ["PostProcess/job013/"]

