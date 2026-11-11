# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
#
#   The XMIPP code in RELION's src/filename.cpp (GPL-2.0-or-later; decomposePipelineFileName,
#   FileName::beforeNthOf):
#   Authors: Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#   Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
"""Pipeline operations relion_pipeliner has no command-line flags for.

Each method mirrors the PipeLine method RELION's own GUI calls (src/pipeliner.cpp, 5.1.1):
setAliasJob, markAsFinishedJob, deleteJobGetNodesAndProcesses + deleteNodesAndProcesses and
undeleteJob (and the job counter juggling of the GUI's "Overwrite"), including the .Nodes/ entries, the alias symlinks and Trash/. They run while
holding RELION's pipeline lock and re-read the pipeline first, as RELION does, so they are safe
next to the original GUI and running pipeliners. tests/test_ops.py compares every operation with
RELION's own code (tools/pipeline_ops_ref).

Five deliberate differences:
- Delete: RELION runs "rm -rf Trash/<job>; mv <job> Trash/...; rm -rf <job>" after writing the
  pipeline, so when the earlier copy in Trash/ cannot be removed, the mv fails and the job is deleted
  for good. Here that copy is cleared (or moved out of the way) before anything is written, and a
  move that still fails puts the pipeline and the directories back.
- Recursive delete: RELION's loop stops as soon as the last job of a round adds no new
  children, so some downstream jobs stay behind with missing inputs. Here every downstream job
  is deleted, as the confirmation list says.
- Undelete: RELION reads deleted_pipeline.star as a pipeline of its own, so links between the
  restored jobs and the jobs that were not deleted are dropped (the restored job loses its
  parent, its children lose this input). Here every link in that file whose two ends exist
  again is restored. Likewise after an overwrite, the jobs that used the old job's outputs are
  linked to the new job's outputs of the same name (RELION leaves them without that input).
- Mark finished / failed: for a Class2D, Class3D, Refine3D or InitialModel job without any
  _optimiser.star yet, RELION refuses both and sets the job to Running, whatever it was (a
  Scheduled job too); from there it refuses again, Overwrite needs a job that is not running, and
  only Delete is left. Here such a job can be marked failed, and marking it finished is refused
  without changing it. It is marked failed as it is: its outputs stay declared (none was written,
  so all of them would be dropped), so that the jobs scheduled after it stay linked to it, also
  through an Overwrite.
- Aliases: RELION keeps a job's alias symlink when the alias is removed, and when another job of
  the same type later gets that alias, symlink() fails and the alias still leads to the first job.
  Here removing an alias removes its symlink, and a symlink left at an alias's path is replaced.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path

from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import StarBlock, StarError, read_star, write_star
from cryo_savanna.relion_consts import (
    STATUS_FAILED,
    STATUS_SCHEDULED,
    STATUS_SUCCEEDED,
    job_type_for_label,
)

NODES_DIR = ".Nodes"
TRASH_DIR = "Trash"
REFINE_TYPES = {"relion.class2d", "relion.class3d", "relion.refine3d", "relion.initialmodel"}  # markAsFinishedJob
LABEL_OPTIMISER = "OptimiserData.star.relion"
LABEL_PARTS = "ParticleGroupMetadata.star.relion"
LABEL_MAP = "DensityMap.mrc"
ALIAS_FORBIDDEN = '*?()/"\\|#<>&%{}$'

_JOB_DIR = re.compile(r"job\d\d\d")


class OpsError(RuntimeError):
    """An operation was refused, with RELION's own message."""


def decompose(name: str) -> tuple[str, str, str] | None:
    """decomposePipelineFileName: 'Class2D/job005/run.star' -> ('Class2D/', 'job005/', 'run.star')."""
    pos = 0
    while True:
        pos = name.find("/", pos + 1)
        if pos < 0:
            return None
        if _JOB_DIR.match(name, pos + 1):
            end = name.find("/", pos + 6)
            if end < 0:
                end = len(name) - 1
            return name[:pos + 1], name[pos + 1:end + 1], name[end + 1:]


def _base_type(type_label: str) -> str:
    jt = job_type_for_label(type_label)
    return jt.label if jt else type_label


class PipelineOps:
    def __init__(self, project_dir: str | os.PathLike, pipeline_name: str = "default",
                 lock_wait_s: float = 3.0, lock_tries: int = 20):
        self.project_dir = Path(project_dir)
        self.pipeline_name = pipeline_name
        self._lock_args = dict(wait_s=lock_wait_s, max_tries=lock_tries)

    def _lock(self, message: str) -> PipelineLock:
        return PipelineLock(self.project_dir, self.pipeline_name, message=f"cryo-savanna: {message}", **self._lock_args)

    def _read(self) -> Pipeline:
        return Pipeline.read(self.project_dir, self.pipeline_name)

    def _path(self, rel: str) -> Path:
        return self.project_dir / rel

    # ------------------------------------------------------------------ alias
    def set_alias(self, job: str, alias: str) -> str:
        """setAliasJob. Returns the new alias ("None" when removed)."""
        with self._lock("setAliasJob") as lock:
            p = self._read()
            pr = self._process(p, job)
            parts = decompose(pr.name)
            if parts is None:
                raise OpsError(f"invalid pipeline process name: {pr.name}")
            fn_pre = parts[0]
            alias = self.check_alias(p, pr.name, alias)
            link = self._path(fn_pre + alias.rstrip("/"))
            if alias != "None" and link.exists() and not link.is_symlink():
                raise OpsError(f"{fn_pre + alias.rstrip('/')} already exists and is not an alias: "
                               "please provide another one")

            self._delete_node_files(p, pr)
            if pr.alias != "None":
                self._unlink(pr.alias.rstrip("/"))  # RELION keeps it when the alias is removed (see the docstring)
            pr.alias = "None" if alias == "None" else fn_pre + alias
            if pr.alias != "None":
                self._link_alias(pr)
            self._touch_node_files(p, pr)
            p.write(self.project_dir, self.pipeline_name, lock)
            return pr.alias

    @staticmethod
    def check_alias(p: Pipeline, job: str, alias: str) -> str:
        """Validate an alias like setAliasJob; returns it normalised ("my alias" -> "my_alias/")."""
        if alias == "":
            return "None"
        if len(alias) < 2:
            raise OpsError("Alias cannot be less than 2 characters, please provide another one")
        if len(alias) > 2 and alias.startswith("job"):
            raise OpsError("Alias cannot start with 'job', please provide another one")
        if any(c in alias for c in ALIAS_FORBIDDEN):
            raise OpsError('Alias cannot contain following symbols: *, ?, (, ), /, ", \\, |, #, <, >, &, %, {, }, $')
        alias = alias.replace(" ", "_")
        if not alias.endswith("/"):
            alias += "/"
        parts = decompose(job)
        fn_pre = parts[0] if parts else ""
        if any(pr.alias == fn_pre + alias for pr in p.processes.values()):
            raise OpsError("Alias is not unique, please provide another one")
        return alias

    # ------------------------------------------------------------------ mark as finished / failed
    def mark_finished(self, job: str, failed: bool = False) -> None:
        """markAsFinishedJob."""
        with self._lock("markAsFinishedJob") as lock:
            p = self._read()
            pr = self._process(p, job)
            pr.status = STATUS_FAILED if failed else STATUS_SUCCEEDED

            if _base_type(pr.type_label) in REFINE_TYPES:
                root = pr.alias if pr.alias != "None" else pr.name
                opts = self._glob(root + "run_it*optimiser.star")
                opts += self._glob(root + "run_ct?_it???_optimiser.star")
                opts += self._glob(root + "run_ct??_it???_optimiser.star")
                if not opts and not failed:  # left as it is (see the module docstring)
                    raise OpsError("This relion_refine job has not written an iteration yet, so it cannot be marked "
                                   "as finished. Mark it as failed instead (it can then be overwritten), or delete it.")
                if not opts:  # never started: failed as it is, its outputs kept (see the module docstring)
                    p.write(self.project_dir, self.pipeline_name, lock)
                    return
                fn_opt = opts[-1]
                self._add_output(p, pr, fn_opt, LABEL_OPTIMISER)
                self._add_output(p, pr, fn_opt.replace("_optimiser.star", "", 1) + "_data.star", LABEL_PARTS)
                fn_root = fn_opt.replace("_optimiser.star", "", 1)
                if _base_type(pr.type_label) == "relion.refine3d":
                    fn_root += "_half1"
                for fn_map in self._glob(fn_root + "_class???.mrc"):
                    self._add_output(p, pr, fn_map, LABEL_MAP)

            # getOutputNodesFromStarFile
            nodes_star = self._path(pr.name + "RELION_OUTPUT_NODES.star")
            if nodes_star.exists():
                try:
                    block = read_star(nodes_star, ["output_nodes"]).get("output_nodes")
                except (OSError, StarError):
                    block = None
                for rec in (block.records() if block is not None else ()):
                    name = rec.get("rlnPipeLineNodeName", "")
                    if name and name not in p.nodes:
                        depth = int(rec.get("rlnPipeLineNodeTypeLabelDepth", "1") or 1)
                        self._add_output(p, pr, name, rec.get("rlnPipeLineNodeTypeLabel", ""), depth)

            # drop the expected outputs that were never written (and every edge to them)
            missing = {n for n in pr.outputs if not self._path(n).exists()}
            kept, _ = split_pipeline(p, set(), missing)
            kept.write(self.project_dir, self.pipeline_name, lock)

    def _glob(self, pattern: str) -> list[str]:
        # glob(3) as RELION uses it: sorted, paths relative to the project
        hits = glob.glob(pattern, root_dir=self.project_dir)
        return sorted(hits)

    def _add_output(self, p: Pipeline, pr: Process, name: str, label: str, depth: int = 1) -> None:
        """addNewOutputEdge + addNode: alias paths become job paths; new nodes are appended."""
        for other in p.processes.values():
            if other.alias != "None" and other.alias in name:
                name = other.name + name.replace(other.alias, "", 1)
                break
        node = p.nodes.get(name)
        is_new = node is None
        if is_new:
            node = Node(name, label, depth, producer=pr.name)
            p.nodes[name] = node
        self._touch_node_file(p, node, pr.status == STATUS_SCHEDULED)
        if is_new:
            pr.outputs.append(name)

    # ------------------------------------------------------------------ delete / undelete
    def delete_plan(self, job: str, recursive: bool = False, p: Pipeline | None = None) -> tuple[list[str], set[str]]:
        """deleteJobGetNodesAndProcesses: (processes to delete in pipeline order, nodes to delete)."""
        p = p or self._read()
        self._process(p, job)
        procs = [job]
        nodes: set[str] = set()
        i = 0
        while i < len(procs):
            for n in p.processes[procs[i]].outputs:
                nodes.add(n)
                if recursive:
                    for c in p.nodes[n].consumers:
                        if c not in procs:
                            procs.append(c)
            i += 1
        chosen = set(procs)
        order = [n for n in p.processes if n in chosen]
        return order, nodes  # every downstream job (RELION's loop can stop early; see the module docstring)

    def delete(self, job: str, recursive: bool = False) -> list[str]:
        """deleteNodesAndProcesses: move the job(s) to Trash/. Returns the deleted job names.

        Unlike RELION, the earlier copies in Trash/ are cleared before the pipeline is written, and a
        directory that cannot be moved puts everything back: the job never ends up outside both the
        pipeline and Trash/ (RELION's "mv" fails there, and its "rm -rf" then deletes the job).
        A job whose directory is gone (removed by hand) leaves the pipeline too, with nothing to undelete.
        """
        with self._lock("deleteNodesAndProcesses") as lock:
            p = self._read()
            procs, nodes = self.delete_plan(job, recursive, p)
            for name in procs:
                self._clear_trash(name)
            kept, deleted = split_pipeline(p, set(procs), nodes)
            fn_del = self._path(procs[0] + "deleted_pipeline.star")
            if fn_del.parent.is_dir():  # a job directory removed by hand: RELION writes no file there either
                write_star(fn_del, deleted, atomic=False)
            kept.write(self.project_dir, self.pipeline_name, lock)
            moved: list[str] = []
            try:
                for name in procs:
                    jobdir = name.rstrip("/")
                    trash_parent = self._path(f"{TRASH_DIR}/{os.path.dirname(jobdir)}")
                    trash_parent.mkdir(parents=True, exist_ok=True)
                    if os.path.lexists(self._path(jobdir)):
                        shutil.move(str(self._path(jobdir)), str(trash_parent))
                        moved.append(jobdir)
            except (OSError, shutil.Error) as e:
                for jobdir in reversed(moved):  # put back what was moved, and the pipeline as it was
                    shutil.move(str(self._path(f"{TRASH_DIR}/{jobdir}")), str(self._path(jobdir)))
                fn_del.unlink(missing_ok=True)
                p.write(self.project_dir, self.pipeline_name, lock)
                raise OpsError(f"Cannot move {name.rstrip('/')} to {TRASH_DIR}/ ({e}); nothing was deleted.") from e
            for name in procs:
                pr = p.processes[name]
                if pr.alias != "None":
                    self._unlink(pr.alias.rstrip("/"))
                self._delete_node_files(p, pr)
            kept.write(self.project_dir, self.pipeline_name, lock)  # RELION re-reads and writes once more
            return procs

    def _clear_trash(self, job: str) -> None:
        """Remove the earlier copy of ``job`` in Trash/ (RELION's "rm -rf Trash/<job>"). What cannot be
        removed (another user's files, files NFS keeps while they are open) is renamed out of the way,
        to a hidden name that Undelete does not offer; if even that fails, nothing is changed."""
        old = self._path(f"{TRASH_DIR}/{job.rstrip('/')}")
        if not os.path.lexists(old):
            return
        if old.is_symlink() or not old.is_dir():
            old.unlink()
            return
        shutil.rmtree(old, ignore_errors=True)
        if not os.path.lexists(old):
            return
        aside = old.with_name(f".{old.name}.removed-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
        try:
            os.rename(old, aside)
        except OSError as e:
            raise OpsError(f"{TRASH_DIR}/{job.rstrip('/')} holds an earlier copy that cannot be removed ({e}). "
                           "Remove it by hand, then try again; nothing was changed.") from e
        shutil.rmtree(aside, ignore_errors=True)

    def in_trash(self, jobs: list[str]) -> list[str]:
        """Those of ``jobs`` that already have a copy in Trash/ (an overwrite reuses the job's number):
        deleting them again replaces that copy, which is then gone for good (RELION does the same)."""
        return [j for j in jobs if self._path(f"{TRASH_DIR}/{j.rstrip('/')}").exists()]

    def trash(self) -> list[Path]:
        """deleted_pipeline.star files of the jobs in Trash/ (what can be undeleted)."""
        return sorted(self.project_dir.glob(f"{TRASH_DIR}/*/job*/deleted_pipeline.star"))

    def undelete(self, deleted_pipeline: str | os.PathLike) -> list[str]:
        """undeleteJob: put the jobs of a Trash/<Type>/<jobNNN>/deleted_pipeline.star back."""
        fn = Path(deleted_pipeline)
        if not fn.is_absolute():
            fn = self.project_dir / fn
        with self._lock("undeleteJob") as lock:
            p = self._read()
            # importPipeline reads the file as a pipeline of its own: edges to nodes or jobs that
            # are not in it are dropped, exactly as RELION does
            mini = Pipeline.parse(fn.read_text())
            for pr in mini.processes.values():
                if pr.name in p.processes:
                    raise OpsError(f"importPipeline ERROR: cannot import pipeline with non-unique job name: {pr.name}")
            taken = [n for n in mini.processes if os.path.lexists(self._path(n.rstrip("/")))
                     and os.path.lexists(self._path(f"{TRASH_DIR}/{n.rstrip('/')}"))]
            if taken:  # mv would put the copy inside that directory; checked before anything is changed
                raise OpsError(f"Cannot undelete: {', '.join(n.rstrip('/') for n in taken)} already exists outside "
                               f"{TRASH_DIR}/. Move it away first; nothing was changed.")
            for pr in mini.processes.values():
                if pr.alias != "None" and any(o.alias == pr.alias for o in p.processes.values()):
                    pr = replace(pr, alias="None")
                p.processes[pr.name] = pr
                # PipeLine::read makes missing alias symlinks, also when reading this file
                link = self._path(pr.alias.rstrip("/"))
                if pr.alias != "None" and os.path.realpath(link) != os.path.realpath(self._path(pr.name)):
                    self._link_alias(pr)
            for node in mini.nodes.values():
                if node.name not in p.nodes:
                    p.nodes[node.name] = node
            self._restore_links(p, fn)
            restored = []
            for name in mini.processes:
                dest = name.rstrip("/")
                self._path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)
                src = self._path(f"{TRASH_DIR}/{dest}")
                if src.exists():
                    shutil.move(str(src), str(self._path(dest)))
                self._touch_node_files(p, p.processes[name])
                restored.append(name)
            p.write(self.project_dir, self.pipeline_name, lock)
            return restored

    @staticmethod
    def _restore_links(p: Pipeline, fn: Path) -> None:
        """Put back the links of deleted_pipeline.star whose job and node both exist again.

        RELION drops the ones that cross between the restored jobs and the rest of the pipeline.
        """
        blocks = read_star(fn, ["pipeline_input_edges", "pipeline_output_edges"])
        ins = blocks.get("pipeline_input_edges")
        for rec in (ins.records() if ins is not None else ()):
            node, proc = rec.get("rlnPipeLineEdgeFromNode"), rec.get("rlnPipeLineEdgeProcess")
            if node in p.nodes and proc in p.processes and node not in p.processes[proc].inputs:
                p.processes[proc].inputs.append(node)
                if proc not in p.nodes[node].consumers:
                    p.nodes[node].consumers.append(proc)
        outs = blocks.get("pipeline_output_edges")
        for rec in (outs.records() if outs is not None else ()):
            proc, node = rec.get("rlnPipeLineEdgeProcess"), rec.get("rlnPipeLineEdgeToNode")
            if node in p.nodes and proc in p.processes and node not in p.processes[proc].outputs:
                p.processes[proc].outputs.append(node)
                p.nodes[node].producer = proc

    # ------------------------------------------------------------------ overwrite
    def set_job_counter(self, counter: int) -> int:
        """Set the number the next job gets; returns the old one. RELION's GUI overwrites a job by
        deleting it and adding the new job with the counter set back to its number."""
        with self._lock("setJobCounter") as lock:
            p = self._read()
            old, p.job_counter = p.job_counter, counter
            p.write(self.project_dir, self.pipeline_name, lock)
            return old

    def restore_job_counter(self, counter: int) -> None:
        """Undo set_job_counter (never lowers the counter: jobs may have been added meanwhile)."""
        with self._lock("setJobCounter") as lock:
            p = self._read()
            if p.job_counter < counter:
                p.job_counter = counter
                p.write(self.project_dir, self.pipeline_name, lock)

    def relink_consumers(self, job: str, deleted_pipeline: str | os.PathLike) -> list[str]:
        """After an overwrite: jobs that used outputs of the old ``job`` (deleted_pipeline.star of
        its trashed copy) use the new job's outputs of the same name. Returns the relinked jobs.

        An output the new job does not declare yet (e.g. External jobs list theirs when they finish)
        is added as its output, as RELION's addNewInputEdge does for an input inside a job directory.
        """
        fn = Path(deleted_pipeline)
        old = Pipeline.parse(fn.read_text())
        with self._lock("relinkConsumers") as lock:
            p = self._read()
            ins = read_star(fn, ["pipeline_input_edges"]).get("pipeline_input_edges")
            relinked = []
            for rec in (ins.records() if ins is not None else ()):
                node, proc = rec.get("rlnPipeLineEdgeFromNode"), rec.get("rlnPipeLineEdgeProcess")
                if not node or proc == job or proc not in p.processes or job not in p.processes:
                    continue  # (no node: a table without rlnPipeLineEdgeFromNode, written by hand)
                if node not in p.nodes and node.startswith(job) and node in old.nodes:
                    o = old.nodes[node]
                    p.nodes[node] = Node(node, o.type_label, o.type_depth, producer=job)
                    p.processes[job].outputs.append(node)
                if node not in p.nodes or p.nodes[node].producer != job:
                    continue
                if node not in p.processes[proc].inputs:
                    p.processes[proc].inputs.append(node)
                    p.nodes[node].consumers.append(proc)
                    relinked.append(proc)
            if relinked:
                p.write(self.project_dir, self.pipeline_name, lock)
            return sorted(set(relinked))

    # ------------------------------------------------------------------ .Nodes
    def _node_file(self, p: Pipeline, node: Node) -> str:
        """Path of a node below its type folder: through the producing job's alias, if it has one."""
        prod = p.processes.get(node.producer) if node.producer else None
        if prod is not None and prod.alias != "None":
            parts = decompose(node.name)
            if parts is None:
                raise OpsError(f"invalid node name: {node.name}")
            alias = prod.alias if prod.alias.endswith("/") else prod.alias + "/"
            return alias + parts[2]
        return node.name

    def _touch_node_file(self, p: Pipeline, node: Node, even_if_missing: bool) -> None:
        """touchTemporaryNodeFile."""
        if not (self._path(node.name).exists() or even_if_missing):
            return
        fnt = self._node_file(p, node)
        types = [node.type_label.split(".", 1)[0]]
        if node.type_depth > 1:
            parts = node.type_label.split(".")
            types.append(".".join(parts[:node.type_depth]) if node.type_depth < len(parts) else node.type_label)
        for t in types:
            target = self._path(f"{NODES_DIR}/{t}/{fnt}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    def _touch_node_files(self, p: Pipeline, pr: Process) -> None:
        for n in pr.outputs:
            if n in p.nodes:
                self._touch_node_file(p, p.nodes[n], pr.status == STATUS_SCHEDULED)

    def _delete_node_files(self, p: Pipeline, pr: Process) -> None:
        """deleteTemporaryNodeFiles: only the first-level type folder, like RELION."""
        for n in pr.outputs:
            node = p.nodes.get(n)
            if node is None:
                continue
            f = self._path(f"{NODES_DIR}/{node.type_label.split('.', 1)[0]}/{self._node_file(p, node)}")
            for target in (f, f.parent):
                try:
                    os.rmdir(target) if target.is_dir() else target.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------ misc
    def _process(self, p: Pipeline, job: str) -> Process:
        pr = p.processes.get(job)
        if pr is None:
            raise OpsError(f"no such job: {job}")
        return pr

    def _link_alias(self, pr: Process) -> None:
        """The alias symlink of ``pr``. A symlink already at that path is left from an alias that
        was removed (aliases are unique), so it is replaced; RELION's symlink() would fail there and
        leave the alias pointing at the old job."""
        link = self._path(pr.alias.rstrip("/"))
        if link.is_symlink():
            self._unlink(pr.alias.rstrip("/"))
        try:
            os.symlink(f"../{pr.name}", link)
        except OSError:
            pass

    def _unlink(self, rel: str) -> None:
        try:
            os.unlink(self._path(rel))
        except OSError:
            pass


def split_pipeline(p: Pipeline, procs: set[str], nodes: set[str]) -> tuple[Pipeline, list[StarBlock]]:
    """PipeLine::write with deleted processes and nodes: (kept pipeline, blocks of deleted_pipeline.star).

    An edge goes to the deleted pipeline when its job or its node is deleted, even when the
    other end stays (as in RELION's write loop).
    """
    kept = Pipeline(job_counter=p.job_counter, version=p.version)
    for name, pr in p.processes.items():
        if name not in procs:
            kept.processes[name] = replace(pr, inputs=[n for n in pr.inputs if n not in nodes],
                                           outputs=[n for n in pr.outputs if n not in nodes])
    for name, node in p.nodes.items():
        if name not in nodes:
            kept.nodes[name] = node

    full = p.to_blocks()  # same columns and value formatting as a normal write
    gen, proc_b, node_b, in_b, out_b = full
    deleted = [
        gen,
        StarBlock.from_records("pipeline_processes", proc_b.columns, (r for r in proc_b.rows if r[0] in procs)),
        StarBlock.from_records("pipeline_nodes", node_b.columns, (r for r in node_b.rows if r[0] in nodes)),
        StarBlock.from_records("pipeline_input_edges", in_b.columns,
                               (r for r in in_b.rows if r[1] in procs or r[0] in nodes)),
        StarBlock.from_records("pipeline_output_edges", out_b.columns,
                               (r for r in out_b.rows if r[0] in procs or r[1] in nodes)),
    ]
    return kept, deleted
