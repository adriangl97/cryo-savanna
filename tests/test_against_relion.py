# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks against a real RELION 5.1 relion_pipeliner. Skipped when the one on PATH is missing or another version.

On a workstation or HPC: `module load relion/5.1 && pytest tests/test_against_relion.py`.
"""

import pytest

from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.tools.validate import PipelinerChecks, check_defaults, pipeliner_version

_version = pipeliner_version()
pytestmark = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")


@pytest.mark.parametrize("mode", ["current", "clean"])
def test_jobstar_defaults_match_relion(tmp_path, mode):
    results = check_defaults(tmp_path, JobDefs.load(), mode)
    bad = {k: v for k, v in results.items() if not v["ok"]}
    assert not bad
    assert all(v["identical_file"] for v in results.values())


def test_pipeliner_flows(tmp_path):
    results = PipelinerChecks(tmp_path, JobDefs.load()).run_all()
    failed = {k: v for k, v in results.items() if not v["ok"]}
    assert not failed
