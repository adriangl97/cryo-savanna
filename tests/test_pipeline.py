# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import os
import threading
import time

import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import LockError, PipelineLock, is_locked, lock_file_name
from cryo_savanna.project.pipeline import Pipeline, PipelineError
from cryo_savanna.project.star import format_star


def test_read_spa_fixture(spa_project):
    p = Pipeline.read(spa_project)
    assert p.job_counter == 19
    assert len(p.processes) == 18
    assert p.roots() == ["Import/job001/"]
    assert p.parents("Extract/job005/") == ["CtfFind/job003/", "AutoPick/job004/"]
    assert p.children("Extract/job005/") == ["Class2D/job006/", "Class2D/job018/"]
    assert p.parents("PostProcess/job013/") == ["MaskCreate/job012/", "Refine3D/job011/"]
    statuses = {pr.status for pr in p.processes.values()}
    assert statuses == {"Succeeded", "Running", "Scheduled", "Failed", "Aborted"}
    ref = p.find("Refine3D/first_refine")
    assert ref is not None and ref.name == "Refine3D/job011/" and ref.display_name == "Refine3D/first_refine/"
    assert p.find("Class2D/job006") is p.processes["Class2D/job006/"]
    assert p.processes["Class2D/job006/"].job_number == 6
    assert p.nodes["PostProcess/job013/postprocess.star"].kind == "ProcessData"
    assert p.warnings == []


def test_read_tomo_fixture(tomo_project):
    p = Pipeline.read(tomo_project)
    assert p.processes["Import/job001/"].type_label == "relion.importtomo"
    assert p.processes["Import/job001/"].is_tomo_type
    assert not p.processes["Refine3D/job011/"].is_tomo_type  # shared type, tomo only via job.star
    assert JobStar.read(tomo_project / "Refine3D/job011").is_tomo


def test_round_trip(spa_project):
    text = (spa_project / "default_pipeline.star").read_text()
    p = Pipeline.parse(text)
    assert format_star(p.to_blocks()) == text


def test_legacy_pipeline_formats():
    # RELION 3.1: integer type/status columns; pre-5.0 node type labels
    text = """
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                       3

# version 30001

data_pipeline_processes

loop_
_rlnPipeLineProcessName #1
_rlnPipeLineProcessAlias #2
_rlnPipeLineProcessType #3
_rlnPipeLineProcessStatus #4
Import/job001/       None            0            2
Class2D/job002/       None            8            3

# version 30001

data_pipeline_nodes

loop_
_rlnPipeLineNodeName #1
_rlnPipeLineNodeTypeLabel #2
Import/job001/particles.star ParticlesData.star.relion
Class2D/job002/run_it025_optimiser.star OptimiserData.star.relion.class2d

# version 30001

data_pipeline_input_edges

loop_
_rlnPipeLineEdgeFromNode #1
_rlnPipeLineEdgeProcess #2
Import/job001/particles.star Class2D/job002/

# version 30001

data_pipeline_output_edges

loop_
_rlnPipeLineEdgeProcess #1
_rlnPipeLineEdgeToNode #2
Import/job001/ Import/job001/particles.star
Class2D/job002/ Class2D/job002/run_it025_optimiser.star
"""
    p = Pipeline.parse(text)
    assert p.processes["Import/job001/"].type_label == "relion.import"
    assert p.processes["Class2D/job002/"].status == "Failed"
    assert p.nodes["Import/job001/particles.star"].type_label == "ParticleGroupMetadata.star.relion"
    assert p.parents("Class2D/job002/") == ["Import/job001/"]


def test_empty_pipeline():
    p = Pipeline.parse("\n# version 50001\n\ndata_pipeline_general\n\n_rlnPipeLineJobCounter 1\n")
    assert p.processes == {} and p.job_counter == 1


def test_missing_general_block():
    with pytest.raises(PipelineError):
        Pipeline.parse("data_pipeline_processes\nloop_\n_rlnPipeLineProcessName\n")


def test_dangling_edges_become_warnings():
    p = Pipeline.parse(
        "data_pipeline_general\n_rlnPipeLineJobCounter 2\n"
        "data_pipeline_input_edges\nloop_\n_rlnPipeLineEdgeFromNode\n_rlnPipeLineEdgeProcess\nx.star A/job001/\n"
    )
    assert len(p.warnings) == 1


def test_read_retries_on_garbage(tmp_path, monkeypatch):
    path = tmp_path / "default_pipeline.star"
    path.write_text("data_pipeline_processes\n")  # half-written: no general block yet
    good = (Pipeline(job_counter=4).to_blocks())

    def fix_later():
        time.sleep(0.2)
        path.write_text(format_star(good))

    threading.Thread(target=fix_later).start()
    p = Pipeline.read(tmp_path, retries=6, delay_s=0.1)
    assert p.job_counter == 4


def test_write_requires_lock(tmp_path):
    p = Pipeline(job_counter=2)
    with pytest.raises(PipelineError):
        p.write(tmp_path)
    with PipelineLock(tmp_path) as lock:
        p.write(tmp_path, lock=lock)
    assert Pipeline.read(tmp_path).job_counter == 2
    assert (tmp_path / ".pipeline_has_changed").exists()


def test_write_keeps_permissions(tmp_path):
    path = tmp_path / "default_pipeline.star"
    path.write_text(format_star(Pipeline(job_counter=1).to_blocks()))
    os.chmod(path, 0o664)
    with PipelineLock(tmp_path) as lock:
        Pipeline(job_counter=5).write(tmp_path, lock=lock)
    assert (path.stat().st_mode & 0o777) == 0o664


def test_lock_protocol(tmp_path):
    assert lock_file_name() == "lock_default_pipeline.star"
    with PipelineLock(tmp_path, message="test") as lock:
        assert is_locked(tmp_path)
        assert (tmp_path / ".relion_lock" / "lock_default_pipeline.star").read_text() == "test\n"
        with pytest.raises(LockError):
            PipelineLock(tmp_path, wait_s=0.01, max_tries=2).acquire()
        assert lock.held
    assert not is_locked(tmp_path)


def test_lock_waits_for_release(tmp_path):
    first = PipelineLock(tmp_path)
    first.acquire()
    threading.Timer(0.15, first.release).start()
    with PipelineLock(tmp_path, wait_s=0.05, max_tries=20):
        pass
