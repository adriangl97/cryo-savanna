# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Pipeline operations: cryo-savanna's Python versions against RELION's own code.

The comparisons need the reference program from tools/pipeline_ops_ref (built against a RELION
5.1 build): set RELION_OPS_REF to its path. Each case runs the operation with RELION in one copy
of a project and with cryo-savanna in another, then compares the two directory trees file by file
(pipeline, deleted_pipeline.star, .Nodes/, Trash/, alias symlinks).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import is_locked
from cryo_savanna.project.ops import OpsError, PipelineOps, decompose
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import AddResult, Pipeliner, PipelinerError
from cryo_savanna.project.selection import SelectionError

OPS_REF = os.environ.get("RELION_OPS_REF")
needs_ref = pytest.mark.skipif(not OPS_REF, reason="set RELION_OPS_REF to a built tools/pipeline_ops_ref program")


def tree(root: Path) -> dict[str, str]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if rel.startswith(".relion_lock"):
                continue
            if p.is_symlink():
                out[rel] = "link:" + os.readlink(p)
            elif p.is_dir():
                out[rel] = "dir"
            elif p.stat().st_size < 200_000:
                out[rel] = p.read_bytes().decode("utf-8", "replace")
            else:
                out[rel] = f"size:{p.stat().st_size}"
    return out


def ref(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([OPS_REF, *args], cwd=project, capture_output=True, text=True, timeout=60)


def py(project: Path, op: str, *args: str) -> str:
    """Same command line as ops_ref, run with PipelineOps. Returns the error message, if any."""
    ops = PipelineOps(project)
    try:
        if op in ("mark_finished", "mark_failed"):
            ops.mark_finished(args[0], failed=op == "mark_failed")
        elif op == "set_alias":
            ops.set_alias(args[0], args[1])
        elif op in ("delete", "delete_recursive"):
            ops.delete(args[0], recursive=op == "delete_recursive")
        elif op == "undelete":
            ops.undelete(args[0])
    except OpsError as e:
        return str(e)
    return ""


def compare(spa_project: Path, tmp_path: Path, *steps: tuple[str, ...]) -> tuple[Path, Path, list[str]]:
    a, b = tmp_path / "relion", tmp_path / "cryo-savanna"
    shutil.copytree(spa_project, a, symlinks=True)
    shutil.copytree(spa_project, b, symlinks=True)
    errors = []
    for step in steps:
        r = ref(a, *step)
        msg = py(b, *step)
        assert (r.returncode != 0) == bool(msg), (step, r.stderr, msg)
        if msg:
            assert msg.strip() in r.stderr, (step, r.stderr, msg)
            errors.append(msg)
        ta, tb = tree(a), tree(b)
        assert ta.keys() == tb.keys(), (step, sorted(ta.keys() ^ tb.keys()))
        diff = [k for k in ta if ta[k] != tb[k]]
        assert not diff, (step, diff, ta[diff[0]][:2000], tb[diff[0]][:2000])
    assert not is_locked(b)
    return a, b, errors


# ---------------------------------------------------------------------------- against RELION

@needs_ref
@pytest.mark.parametrize("job,alias", [
    ("Class2D/job006/", "my 2D classes"),  # spaces become underscores
    ("Refine3D/job011/", "refined"),  # replaces an existing alias and its symlink
])
def test_set_alias(spa_project, tmp_path, job, alias):
    compare(spa_project, tmp_path, ("set_alias", job, alias))


@needs_ref
def test_remove_alias(spa_project, tmp_path):
    """Like RELION, except that the alias symlink goes too (RELION keeps it; see project/ops.py)."""
    a, b = tmp_path / "relion", tmp_path / "cryo-savanna"
    shutil.copytree(spa_project, a, symlinks=True)
    shutil.copytree(spa_project, b, symlinks=True)
    link = Pipeline.read(a).processes["Refine3D/job011/"].alias.rstrip("/")
    assert ref(a, "set_alias", "Refine3D/job011/", "").returncode == 0
    assert py(b, "set_alias", "Refine3D/job011/", "") == ""
    ta, tb = tree(a), tree(b)
    assert ta.keys() - tb.keys() == {link} and not tb.keys() - ta.keys()
    assert not [k for k in tb if ta[k] != tb[k]]


@needs_ref
@pytest.mark.parametrize("alias", ["x", "job_new", "bad/name", "first_refine"])
def test_set_alias_refused(spa_project, tmp_path, alias):
    _, _, errors = compare(spa_project, tmp_path, ("set_alias", "Refine3D/job016/", alias))
    assert errors


@needs_ref
@pytest.mark.parametrize("op,job", [
    ("mark_finished", "Polish/job015/"),  # running job: unwritten outputs are dropped
    ("mark_failed", "LocalRes/job017/"),
    ("mark_finished", "Class3D/job009/"),  # refine type: last iteration outputs are added
    ("mark_finished", "Class2D/job018/"),  # aborted refine type
])
def test_mark_finished(spa_project, tmp_path, op, job):
    compare(spa_project, tmp_path, (op, job))


@needs_ref
@pytest.mark.parametrize("op", ["mark_finished", "mark_failed"])
def test_mark_a_refine_job_that_never_started(spa_project, tmp_path, op):
    """A deliberate difference (project/ops.py): RELION refuses both and leaves the job Running, whatever it was;
    here marking it failed works (nothing to add), and marking it finished is refused without changing it."""
    job = "Refine3D/job016/"  # Scheduled, no iteration written
    a, b = tmp_path / "relion", tmp_path / "cryo-savanna"
    shutil.copytree(spa_project, a, symlinks=True)
    shutil.copytree(spa_project, b, symlinks=True)
    assert ref(a, op, job).returncode != 0
    assert Pipeline.read(a).processes[job].status == "Running"  # RELION's trap: only Delete is left
    msg = py(b, op, job)
    want = ("Failed", "") if op == "mark_failed" else ("Scheduled", "cannot be marked as finished")
    assert Pipeline.read(b).processes[job].status == want[0] and want[1] in msg


@needs_ref
@pytest.mark.parametrize("job", ["Class2D/job018/", "Select/job010/", "Refine3D/job011/"])
def test_delete_and_undelete(spa_project, tmp_path, job):
    """Delete matches RELION exactly. Undelete matches it except for the links RELION drops."""
    a, b, _ = compare(spa_project, tmp_path, ("delete", job))
    assert ref(a, "undelete", f"Trash/{job}deleted_pipeline.star").returncode == 0
    assert py(b, "undelete", f"Trash/{job}deleted_pipeline.star") == ""
    ta, tb = tree(a), tree(b)
    assert ta.keys() == tb.keys()
    assert [k for k in ta if ta[k] != tb[k]] in ([], ["default_pipeline.star"])
    assert edges(Pipeline.read(b)) == edges(Pipeline.read(spa_project))  # back to how it was
    if job != "Class2D/job018/":  # a job with a parent and children
        assert edges(Pipeline.read(a)) < edges(Pipeline.read(spa_project))  # RELION lost some


@needs_ref
def test_recursive_delete_removes_every_downstream_job(spa_project, tmp_path):
    a, b = tmp_path / "relion", tmp_path / "cryo-savanna"
    shutil.copytree(spa_project, a, symlinks=True)
    shutil.copytree(spa_project, b, symlinks=True)
    assert ref(a, "delete_recursive", "Class3D/job009/").returncode == 0
    deleted = PipelineOps(b).delete("Class3D/job009/", recursive=True)
    left_a, left_b = Pipeline.read(a).processes, Pipeline.read(b).processes
    assert set(left_b) < set(left_a)  # RELION's loop stops early and leaves some behind
    assert "Polish/job015/" in left_a and "Polish/job015/" not in left_b
    assert not any(c in left_b for n in deleted for c in Pipeline.read(spa_project).children(n))
    # the jobs both deleted are in the same place
    both = set(Pipeline.read(spa_project).processes) - set(left_a)
    assert all((b / "Trash" / n).is_dir() for n in both)


def edges(p: Pipeline) -> set[tuple[str, str, str]]:
    return {("in", n, pr.name) for pr in p.processes.values() for n in pr.inputs} | \
        {("out", pr.name, n) for pr in p.processes.values() for n in pr.outputs}


# ---------------------------------------------------------------------------- without RELION

@pytest.mark.parametrize("status", ["Scheduled", "Running"])
def test_a_refine_job_that_never_started_can_be_marked_failed(spa_copy, status):
    """No _optimiser.star yet (Scheduled, or left Running by RELION's own refusal): marked failed as it is, its
    outputs kept so the jobs after it stay linked (also through an Overwrite); marking it finished is refused
    and changes nothing."""
    from cryo_savanna.project.lock import PipelineLock
    job = "Class3D/job009/"  # used by Select/job010/ and Refine3D/job011/
    for f in (spa_copy / job).glob("run_*optimiser.star"):
        f.unlink()  # as if it never wrote an iteration
    with PipelineLock(spa_copy) as lock:
        p = Pipeline.read(spa_copy)
        p.processes[job].status = status
        p.write(spa_copy, lock=lock)
    before = (spa_copy / "default_pipeline.star").read_text()
    p = Pipeline.read(spa_copy)
    outputs, children = p.processes[job].outputs, p.children(job)
    ops = PipelineOps(spa_copy)
    with pytest.raises(OpsError, match="cannot be marked as finished"):
        ops.mark_finished(job)
    assert (spa_copy / "default_pipeline.star").read_text() == before
    ops.mark_finished(job, failed=True)
    p = Pipeline.read(spa_copy)
    assert p.processes[job].status == "Failed"
    assert p.processes[job].outputs == outputs and p.children(job) == children and children


def test_decompose():
    assert decompose("Class2D/job005/run_it025_data.star") == ("Class2D/", "job005/", "run_it025_data.star")
    assert decompose("Class2D/job005/") == ("Class2D/", "job005/", "")
    assert decompose("Import/movies/x.star") is None


def test_delete_plan(spa_project):
    ops = PipelineOps(spa_project)
    procs, _ = ops.delete_plan("Select/job010/", recursive=False)
    assert procs == ["Select/job010/"]
    procs, nodes = ops.delete_plan("Class3D/job009/", recursive=True)
    assert procs[0] == "Class3D/job009/" and "PostProcess/job013/" in procs and "Refine3D/job016/" in procs
    assert "Class3D/job009/run_it025_optimiser.star" in nodes


def test_alias_checks(spa_project):
    p = Pipeline.read(spa_project)
    assert PipelineOps.check_alias(p, "Class2D/job006/", "my run") == "my_run/"
    assert PipelineOps.check_alias(p, "Class2D/job006/", "") == "None"
    for bad in ("x", "jobby", "a/b", "a$b"):
        with pytest.raises(OpsError):
            PipelineOps.check_alias(p, "Class2D/job006/", bad)
    with pytest.raises(OpsError, match="not unique"):
        PipelineOps.check_alias(p, "Refine3D/job016/", "first_refine")


def test_alias_moved_to_another_job_leads_to_it(spa_copy):
    """RELION keeps a removed alias's symlink, so the alias given again to another job led to the first one."""
    ops = PipelineOps(spa_copy)
    ops.set_alias("Class2D/job006/", "best")
    ops.set_alias("Class2D/job006/", "")
    assert not (spa_copy / "Class2D/best").is_symlink()
    os.symlink("../Class2D/job006/", spa_copy / "Class2D/best")  # as RELION leaves it
    assert ops.set_alias("Class2D/job018/", "best") == "Class2D/best/"
    assert os.readlink(spa_copy / "Class2D/best") == "../Class2D/job018/"


def test_alias_refused_over_a_real_directory(spa_copy):
    (spa_copy / "Class2D/best").mkdir()
    with pytest.raises(OpsError, match="already exists"):
        PipelineOps(spa_copy).set_alias("Class2D/job018/", "best")
    assert Pipeline.read(spa_copy).processes["Class2D/job018/"].alias == "None"


def test_undelete_replaces_a_stale_alias_symlink(spa_copy):
    ops = PipelineOps(spa_copy)
    ops.set_alias("Class2D/job018/", "best")
    ops.delete("Class2D/job018/")
    os.symlink("../Class2D/job006/", spa_copy / "Class2D/best")  # left by an alias of another job
    ops.undelete(spa_copy / "Trash/Class2D/job018/deleted_pipeline.star")
    assert os.readlink(spa_copy / "Class2D/best") == "../Class2D/job018/"


def test_undelete_refusal_changes_nothing(spa_copy):
    """A job directory back in the way: Undelete refuses before it makes the alias link (or anything else)."""
    ops = PipelineOps(spa_copy)
    ops.set_alias("Class2D/job018/", "best")
    ops.delete("Class2D/job018/")
    (spa_copy / "Class2D/job018").mkdir()
    before = (spa_copy / "default_pipeline.star").read_text()
    with pytest.raises(OpsError, match="already exists outside"):
        ops.undelete(spa_copy / "Trash/Class2D/job018/deleted_pipeline.star")
    assert not os.path.lexists(spa_copy / "Class2D/best")
    assert (spa_copy / "default_pipeline.star").read_text() == before


def test_in_trash(spa_copy):
    ops = PipelineOps(spa_copy)
    assert ops.in_trash(["Class2D/job018/", "Class2D/job006/"]) == []
    ops.delete("Class2D/job018/")
    assert ops.in_trash(["Class2D/job018/", "Class2D/job006/"]) == ["Class2D/job018/"]


def test_undelete_restores_job(spa_copy):
    ops = PipelineOps(spa_copy)
    ops.delete("Class2D/job018/")
    assert "Class2D/job018/" not in Pipeline.read(spa_copy).processes
    assert ops.trash() == [spa_copy / "Trash/Class2D/job018/deleted_pipeline.star"]
    assert ops.undelete(ops.trash()[0]) == ["Class2D/job018/"]
    assert "Class2D/job018/" in Pipeline.read(spa_copy).processes
    assert (spa_copy / "Class2D/job018/job.star").exists()


def test_job_counter_for_overwrite(spa_copy):
    ops = PipelineOps(spa_copy)
    before = Pipeline.read(spa_copy).job_counter
    assert ops.set_job_counter(6) == before
    assert Pipeline.read(spa_copy).job_counter == 6
    ops.restore_job_counter(before)
    assert Pipeline.read(spa_copy).job_counter == before
    ops.restore_job_counter(3)  # never lowers it
    assert Pipeline.read(spa_copy).job_counter == before


def test_relink_after_overwrite(spa_copy):
    ops = PipelineOps(spa_copy)
    users = Pipeline.read(spa_copy).children("Class2D/job006/")
    assert users
    ops.delete("Class2D/job006/")
    trashed = spa_copy / "Trash/Class2D/job006/deleted_pipeline.star"
    assert not any("Class2D/job006/" in Pipeline.read(spa_copy).parents(u) for u in users)
    saved = spa_copy / "saved_deleted_pipeline.star"
    shutil.copy(trashed, saved)
    ops.undelete(trashed)  # stands in for the new job: same name, same outputs
    p = Pipeline.read(spa_copy)
    for u in users:  # undelete already restores the links; relinking again changes nothing
        p.processes[u].inputs = [n for n in p.processes[u].inputs if p.nodes[n].producer != "Class2D/job006/"]
    with ops._lock("test") as lock:
        p.write(spa_copy, "default", lock)
    assert ops.relink_consumers("Class2D/job006/", saved) == sorted(users)
    assert all("Class2D/job006/" in Pipeline.read(spa_copy).parents(u) for u in users)


# ---------------------------------------------------------------------------- failed overwrites

def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_overwrite_failure_says_where_the_old_job_is(spa_copy, monkeypatch):
    """Any failure while adding the new job (here a timeout) says how to get the old one back."""
    counter = Pipeline.read(spa_copy).job_counter
    pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})

    def add_job(*args, **kwargs):
        raise subprocess.TimeoutExpired("relion_pipeliner", 600)
    monkeypatch.setattr(pl, "add_job", add_job)
    with pytest.raises(PipelinerError, match="Undelete"):
        pl.overwrite_job("Class2D/job018/", JobStar(type_label="relion.class2d", is_continue=False, is_tomo=False,
                                                    options={}))
    assert (spa_copy / "Trash/Class2D/job018").is_dir()
    assert Pipeline.read(spa_copy).job_counter == counter


def test_failed_selection_overwrite_puts_the_old_job_back(spa_copy, monkeypatch, tmp_path):
    job = "Select/job007/"
    before = Pipeline.read(spa_copy)
    files = _files(spa_copy / job)
    pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
    ops = PipelineOps(spa_copy)

    def add_job(js, alias=None):  # stands in for relion_pipeliner: a new job in the old job's directory
        trashed = spa_copy / "Trash" / job
        shutil.copytree(trashed, tmp_path / "old", symlinks=True)
        ops.undelete(trashed / "deleted_pipeline.star")
        shutil.copytree(tmp_path / "old", trashed, symlinks=True)
        (spa_copy / job / "new_job.txt").write_text("the new job")
        return AddResult(job)
    monkeypatch.setattr(pl, "add_job", add_job)

    def write(j):
        raise SelectionError("The selected classes have no particles.")
    with pytest.raises(PipelinerError, match="back as it was"):
        pl.overwrite_done_here(job, JobStar(type_label="relion.select", is_continue=False, is_tomo=False, options={}),
                               write, "relion_display")
    after = Pipeline.read(spa_copy)
    restored = _files(spa_copy / job)
    assert restored == files  # also no deleted_pipeline.star: the job was never deleted
    assert edges(after) == edges(before)
    assert after.processes[job].status == before.processes[job].status
    assert not (spa_copy / "Trash/Select").exists() or not any((spa_copy / "Trash/Select").iterdir())


def test_delete_clears_an_earlier_copy_in_trash_it_cannot_empty(spa_copy):
    """An earlier copy in Trash/ with files this user cannot remove (another user's, or NFS .nfs files):
    the job still goes to Trash/ (the leftover is moved out of the way), never out of both."""
    import stat
    ops = PipelineOps(spa_copy)
    job = "Class2D/job006/"
    busy = spa_copy / "Trash/Class2D/job006/busy"
    busy.mkdir(parents=True)
    (busy / ".nfs000123").write_text("x")
    busy.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert ops.delete(job) == [job]
    finally:
        for d in (spa_copy / "Trash/Class2D").glob(".job006.removed-*/busy"):
            d.chmod(stat.S_IRWXU)
    assert job not in Pipeline.read(spa_copy).processes and not (spa_copy / job).exists()
    assert ops.trash() == [spa_copy / "Trash/Class2D/job006/deleted_pipeline.star"]
    assert not (spa_copy / "Trash/Class2D/job006/busy").exists()
    assert ops.undelete(ops.trash()[0]) == [job]


def test_delete_that_cannot_move_a_job_changes_nothing(spa_copy, monkeypatch):
    ops = PipelineOps(spa_copy)
    procs, _ = ops.delete_plan("Class2D/job006/", recursive=True)
    assert len(procs) > 1
    before = (spa_copy / "default_pipeline.star").read_text()
    real_move = shutil.move

    def move(src, dst):
        if src.rstrip("/").endswith(procs[1].rstrip("/")):
            raise OSError("Permission denied")
        return real_move(src, dst)
    monkeypatch.setattr("cryo_savanna.project.ops.shutil.move", move)
    with pytest.raises(OpsError, match="nothing was deleted"):
        ops.delete("Class2D/job006/", recursive=True)
    assert set(Pipeline.read(spa_copy).processes) == set(Pipeline.parse(before).processes)
    assert all((spa_copy / p).is_dir() for p in procs)
    assert not (spa_copy / procs[0] / "deleted_pipeline.star").exists() and ops.trash() == []
    assert not is_locked(spa_copy)


def test_delete_a_job_whose_directory_is_gone(spa_copy):
    """As in RELION: the job leaves the pipeline even when its directory was removed by hand."""
    ops = PipelineOps(spa_copy)
    shutil.rmtree(spa_copy / "LocalRes/job017")
    assert ops.delete("LocalRes/job017/") == ["LocalRes/job017/"]
    p = Pipeline.read(spa_copy)
    assert "LocalRes/job017/" not in p.processes and "LocalRes/job017/relion_locres.mrc" not in p.nodes
    assert not (spa_copy / "LocalRes/job017").exists() and ops.trash() == []
    assert not is_locked(spa_copy)


def test_undelete_refuses_to_move_into_an_existing_directory(spa_copy):
    ops = PipelineOps(spa_copy)
    ops.delete("Class2D/job018/")
    (spa_copy / "Class2D/job018").mkdir()
    with pytest.raises(OpsError, match="already exists"):
        ops.undelete(spa_copy / "Trash/Class2D/job018/deleted_pipeline.star")
    assert not (spa_copy / "Class2D/job018/job018").exists()
    assert "Class2D/job018/" not in Pipeline.read(spa_copy).processes
