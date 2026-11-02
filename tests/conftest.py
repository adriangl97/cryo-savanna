# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Qt tests run without a display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# thumbnails of the test projects go to a temporary directory, not ~/.cache
os.environ.setdefault("CRYO_SAVANNA_CACHE", tempfile.mkdtemp(prefix="cryo_savanna_cache_"))

from tests.fixtures.make_fixtures import make_spa_project, make_tomo_project


@pytest.fixture(autouse=True)
def _gc_on_main_thread(request):
    """As in the app (ui/gc_guard.py): garbage is collected on the GUI thread only, also between tests,
    never in a loader or summary thread (that can deadlock with the GUI thread)."""
    try:
        from cryo_savanna.ui import gc_guard
        gc_guard.install(request.getfixturevalue("qapp"))
    except ImportError:  # no Qt: nothing runs in Qt threads
        pass
    yield
    import gc
    gc.collect()


@pytest.fixture(scope="session")
def spa_project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("projects") / "spa"
    make_spa_project(root)
    return root


@pytest.fixture(scope="session")
def tomo_project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("projects") / "tomo"
    make_tomo_project(root)
    return root


@pytest.fixture
def spa_copy(spa_project, tmp_path) -> Path:
    """A private copy of the SPA project that a test may change."""
    dst = tmp_path / "spa"
    shutil.copytree(spa_project, dst, symlinks=True)
    return dst
