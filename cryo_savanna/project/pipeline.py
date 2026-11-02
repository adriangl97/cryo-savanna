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
"""Model of a RELION pipeline file (default_pipeline.star).

Reading mirrors PipeLine::read and writing mirrors PipeLine::write in RELION 5.1
(src/pipeliner.cpp). Writes are only meant for the few operations relion_pipeliner cannot do
(project/ops.py) and must happen while holding the pipeline lock.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.star import StarBlock, StarError, parse_star_text, write_star
from cryo_savanna.relion_consts import (
    JOB_TYPE_BY_ID,
    PIPELINE_HAS_CHANGED,
    STATUS_BY_CODE,
    STATUS_LABELS,
    convert_old_node_type_label,
    job_type_for_label,
)

_JOB_RE = re.compile(r"^(?P<dir>[^/]+)/job(?P<num>\d+)/?$")


class PipelineError(RuntimeError):
    pass


@dataclass
class Node:
    name: str
    type_label: str
    type_depth: int = 1
    producer: str | None = None  # name of the process that outputs this node
    consumers: list[str] = field(default_factory=list)  # processes that take it as input

    @property
    def kind(self) -> str:
        """Top-level node kind, e.g. 'ParticleGroupMetadata' or 'DensityMap'."""
        return self.type_label.split(".", 1)[0]


@dataclass
class Process:
    name: str  # e.g. "Class2D/job005/"
    alias: str  # "None" when the job has no alias
    type_label: str  # e.g. "relion.class2d"
    status: str  # one of STATUS_LABELS
    inputs: list[str] = field(default_factory=list)  # node names
    outputs: list[str] = field(default_factory=list)  # node names

    @property
    def has_alias(self) -> bool:
        return self.alias not in ("", "None")

    @property
    def display_name(self) -> str:
        return self.alias if self.has_alias else self.name

    @property
    def job_number(self) -> int | None:
        m = _JOB_RE.match(self.name)
        return int(m.group("num")) if m else None

    @property
    def type_dir(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def is_tomo_type(self) -> bool:
        jt = job_type_for_label(self.type_label)
        return jt is not None and 50 <= jt.proc_id < 99


@dataclass
class Pipeline:
    job_counter: int = 1
    processes: dict[str, Process] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    version: int | None = None
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ reading
    @classmethod
    def read(
        cls,
        project_dir: str | os.PathLike,
        name: str = "default",
        retries: int = 5,
        delay_s: float = 0.3,
    ) -> "Pipeline":
        """Read <project_dir>/<name>_pipeline.star.

        RELION rewrites the file in place, so a read can catch it half-written. Retry when the
        file changes while reading or does not parse.
        """
        path = Path(project_dir) / f"{name}_pipeline.star"
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                st1 = path.stat()
                text = path.read_text(encoding="utf-8", errors="replace")
                st2 = path.stat()
                if (st1.st_mtime_ns, st1.st_size) != (st2.st_mtime_ns, st2.st_size):
                    raise PipelineError("pipeline file changed while reading")
                return cls.parse(text)
            except FileNotFoundError:
                raise
            except (StarError, PipelineError, KeyError, ValueError) as e:
                last_err = e
                time.sleep(delay_s * (attempt + 1))
        raise PipelineError(f"Cannot read {path}: {last_err}")

    @classmethod
    def parse(cls, text: str) -> "Pipeline":
        blocks = parse_star_text(text)
        if "pipeline_general" not in blocks:
            raise PipelineError("missing data_pipeline_general")
        p = cls()
        gen = blocks["pipeline_general"]
        p.version = gen.version
        p.job_counter = int(gen.get("rlnPipeLineJobCounter", "1"))
        if p.job_counter < 0:
            raise PipelineError("rlnPipeLineJobCounter must not be negative")

        nodes = blocks.get("pipeline_nodes")
        if nodes is not None:
            convert = nodes.version is None or nodes.version < 50001
            for rec in nodes.records():
                label = rec.get("rlnPipeLineNodeTypeLabel", "")
                if convert and not label.startswith(("TomoOptimisationSet", "TomoTrajectoryData")):
                    label = convert_old_node_type_label(label)
                depth = int(rec.get("rlnPipeLineNodeTypeLabelDepth", "1"))
                n = Node(rec["rlnPipeLineNodeName"], label, depth)
                p.nodes[n.name] = n

        procs = blocks.get("pipeline_processes")
        if procs is not None:
            for rec in procs.records():
                if "rlnPipeLineProcessTypeLabel" in rec and "rlnPipeLineProcessStatusLabel" in rec:
                    type_label = rec["rlnPipeLineProcessTypeLabel"]
                    status = rec["rlnPipeLineProcessStatusLabel"]
                    if status not in STATUS_LABELS:
                        raise PipelineError(f"unknown process status {status!r}")
                elif "rlnPipeLineProcessType" in rec and "rlnPipeLineProcessStatus" in rec:
                    jt = JOB_TYPE_BY_ID.get(int(rec["rlnPipeLineProcessType"]))
                    type_label = jt.label if jt else rec["rlnPipeLineProcessType"]
                    status = STATUS_BY_CODE[int(rec["rlnPipeLineProcessStatus"])]
                else:
                    raise PipelineError("cannot find type or status in pipeline_processes table")
                pr = Process(rec["rlnPipeLineProcessName"], rec["rlnPipeLineProcessAlias"], type_label, status)
                p.processes[pr.name] = pr

        edges_in = blocks.get("pipeline_input_edges")
        if edges_in is not None:
            for rec in edges_in.records():
                pname, nname = rec["rlnPipeLineEdgeProcess"], rec["rlnPipeLineEdgeFromNode"]
                if pname not in p.processes or nname not in p.nodes:
                    p.warnings.append(f"input edge {nname} -> {pname} refers to unknown process or node")
                    continue
                p.processes[pname].inputs.append(nname)
                p.nodes[nname].consumers.append(pname)

        edges_out = blocks.get("pipeline_output_edges")
        if edges_out is not None:
            for rec in edges_out.records():
                pname, nname = rec["rlnPipeLineEdgeProcess"], rec["rlnPipeLineEdgeToNode"]
                if pname not in p.processes or nname not in p.nodes:
                    p.warnings.append(f"output edge {pname} -> {nname} refers to unknown process or node")
                    continue
                p.processes[pname].outputs.append(nname)
                p.nodes[nname].producer = pname
        return p

    # ------------------------------------------------------------------ writing
    def to_blocks(self) -> list[StarBlock]:
        gen = StarBlock.from_dict("pipeline_general", {"rlnPipeLineJobCounter": self.job_counter})
        procs = StarBlock.from_records(
            "pipeline_processes",
            ["rlnPipeLineProcessName", "rlnPipeLineProcessAlias", "rlnPipeLineProcessTypeLabel",
             "rlnPipeLineProcessStatusLabel"],
            ([pr.name, pr.alias, pr.type_label, pr.status] for pr in self.processes.values()),
        )
        nodes = StarBlock.from_records(
            "pipeline_nodes",
            ["rlnPipeLineNodeName", "rlnPipeLineNodeTypeLabel", "rlnPipeLineNodeTypeLabelDepth"],
            ([n.name, n.type_label, n.type_depth] for n in self.nodes.values()),
        )
        e_in = StarBlock.from_records(
            "pipeline_input_edges",
            ["rlnPipeLineEdgeFromNode", "rlnPipeLineEdgeProcess"],
            ([n, pr.name] for pr in self.processes.values() for n in pr.inputs),
        )
        e_out = StarBlock.from_records(
            "pipeline_output_edges",
            ["rlnPipeLineEdgeProcess", "rlnPipeLineEdgeToNode"],
            ([pr.name, n] for pr in self.processes.values() for n in pr.outputs),
        )
        return [gen, procs, nodes, e_in, e_out]

    def write(self, project_dir: str | os.PathLike, name: str = "default", lock: PipelineLock | None = None) -> None:
        """Write the pipeline. ``lock`` must be a held PipelineLock for this project."""
        if lock is None or not lock.held:
            raise PipelineError("Pipeline.write requires a held PipelineLock")
        project_dir = Path(project_dir)
        write_star(project_dir / f"{name}_pipeline.star", self.to_blocks())
        (project_dir / PIPELINE_HAS_CHANGED).touch()

    # ------------------------------------------------------------------ graph queries
    def parents(self, process_name: str) -> list[str]:
        """Processes whose outputs are inputs of ``process_name`` (unique, input order)."""
        out: list[str] = []
        for n in self.processes[process_name].inputs:
            prod = self.nodes[n].producer
            if prod is not None and prod not in out:
                out.append(prod)
        return out

    def children(self, process_name: str) -> list[str]:
        out: list[str] = []
        for n in self.processes[process_name].outputs:
            for c in self.nodes[n].consumers:
                if c not in out:
                    out.append(c)
        return out

    def roots(self) -> list[str]:
        return [name for name in self.processes if not self.parents(name)]

    def find(self, name_or_alias: str) -> Process | None:
        if name_or_alias in self.processes:
            return self.processes[name_or_alias]
        key = name_or_alias if name_or_alias.endswith("/") else name_or_alias + "/"
        if key in self.processes:
            return self.processes[key]
        for pr in self.processes.values():
            if pr.has_alias and pr.alias.rstrip("/") == name_or_alias.rstrip("/"):
                return pr
        return None
