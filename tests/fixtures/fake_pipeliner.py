# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fake relion_pipeliner and relion_image_handler for tests: External jobs only.

    install(bin_dir)   writes relion_pipeliner, relion_image_handler and python3 into bin_dir

The pipeliner knows --version, --addJobFromStar [--setJobAlias A] (a Scheduled job of any type in
its type's folder, linked to the pipeline nodes its options name; an interactive Select on classes
declares particles.star and, from Class2D, class_averages.star, as RELION does) and --RunJobs
(External jobs, one after the other). It builds an External job's command the way RELION 5.1.1 does
(getCommandsExternalJob: fn_exe --o <job> [--in_* F] [--<label> <value>] --j N <other_args>; then
prepareFinalCommand adds --pipeline_control <job> when the command names a relion_ program), runs
it through the shell with run.out / run.err, and finishes the job from its exit files and
RELION_OUTPUT_NODES.star, as checkProcessCompletion does. The image handler copies --i to --o.
Not RELION: tests/test_against_relion.py and tools/validate_against_relion.sh check the real one.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PIPELINER = r'''
import os, subprocess, sys, time
sys.path.insert(0, %(root)r)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.ops import PipelineOps
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import read_star
from cryo_savanna.relion_consts import job_type_for_label

INPUTS = (("in_mov", "--in_movies"), ("in_mic", "--in_mics"), ("in_part", "--in_parts"),
          ("in_coords", "--in_coords"), ("in_3dref", "--in_3dref"), ("in_mask", "--in_mask"))
EXITS = (("RELION_JOB_EXIT_SUCCESS", "Succeeded"), ("RELION_JOB_EXIT_FAILURE", "Failed"),
         ("RELION_JOB_EXIT_ABORTED", "Aborted"))
args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1]


def add_job(path, alias):
    js = JobStar.read(path)
    label = js.type_label
    with PipelineLock(".") as lock:
        p = Pipeline.read(".")
        name = "%%s/job%%03d/" %% (job_type_for_label(label).dirname, p.job_counter)
        os.makedirs(name, exist_ok=True)
        js.is_continue = False
        js.write(name + "job.star")
        outputs = []
        if label == "relion.select" and js.options.get("fn_model"):
            label = "relion.select.interactive"
            outputs.append((name + "particles.star", "ParticleGroupMetadata.star.relion"))
            if "Class2D/" in js.options["fn_model"]:
                outputs.append((name + "class_averages.star", "ProcessData.star.relion.classaverages"))
        pr = Process(name, "None", label, "Scheduled")
        for key, value in sorted(js.options.items()):
            if key in dict(INPUTS) or value in p.nodes:
                if not value:
                    continue
                p.nodes.setdefault(value, Node(value, "DensityMap.mrc"))
                if value not in pr.inputs:
                    pr.inputs.append(value)
                    p.nodes[value].consumers.append(name)
        for node, node_label in outputs:
            p.nodes[node] = Node(node, node_label, producer=name)
            pr.outputs.append(node)
        p.processes[name] = pr
        p.job_counter += 1
        p.write(".", lock=lock)
    if alias:
        try:
            PipelineOps(".").set_alias(name, alias)
        except Exception as e:  # RELION only warns
            print("WARNING: Failed to set the job alias to %%s (%%s). The job name remains the default." %% (alias, e),
                  file=sys.stderr)


def command(name):
    o = JobStar.read(name + "job.star").options
    cmd = o["fn_exe"] + " --o " + name
    for key, flag in INPUTS:
        if o.get(key):
            cmd += " %%s %%s" %% (flag, o[key])
    for i in range(1, 11):
        if o.get("param%%d_label" %% i):
            cmd += " --%%s %%s" %% (o["param%%d_label" %% i], o["param%%d_value" %% i])
    cmd += " --j " + o.get("nr_threads", "1") + " " + o.get("other_args", "")
    if "relion_" in cmd:
        cmd += " --pipeline_control " + name
    return cmd


def set_status(name, status, outputs=()):
    with PipelineLock(".") as lock:
        p = Pipeline.read(".")
        p.processes[name].status = status
        for node, label in outputs:
            if node not in p.nodes:
                p.nodes[node] = Node(node, label, producer=name)
                p.processes[name].outputs.append(node)
        p.write(".", lock=lock)


def run(name):
    cmd = command(name)
    set_status(name, "Running")
    with open(name + "note.txt", "a") as f:
        f.write(" ++++ Executing new job on %%s\n ++++ with the following command(s): \n%%s\n ++++ \n" %% (time.ctime(), cmd))
    subprocess.run("%%s >> %%srun.out 2>> %%srun.err" %% (cmd, name, name), shell=True)
    status = next((s for marker, s in EXITS if os.path.exists(name + marker)), "Failed")
    outputs = []
    if status == "Succeeded" and os.path.exists(name + "RELION_OUTPUT_NODES.star"):
        block = read_star(name + "RELION_OUTPUT_NODES.star")["output_nodes"]
        outputs = [(r["rlnPipeLineNodeName"], r["rlnPipeLineNodeTypeLabel"]) for r in block.records()]
    set_status(name, status, outputs)


if "--version" in args:
    print("RELION version: 5.1.1-commit-fake")
elif "--addJobFromStar" in args:
    add_job(opt("--addJobFromStar"), opt("--setJobAlias") if "--setJobAlias" in args else "")
elif "--RunJobs" in args:
    for job in opt("--RunJobs").split():
        run(job)
'''

IMAGE_HANDLER = r'''
import shutil, sys
args = sys.argv[1:]
if "--help" in args:
    print(open(%(help)r).read())
    sys.exit(0)
shutil.copy(args[args.index("--i") + 1], args[args.index("--o") + 1])
print("image_handler", " ".join(args))
'''


def install(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    help_text = str(ROOT / "cryo_savanna/jobdefs/help/5.1.1/relion_image_handler.txt")
    for name, body in (("relion_pipeliner", PIPELINER % {"root": str(ROOT)}),
                       ("relion_image_handler", IMAGE_HANDLER % {"help": help_text})):
        f = bin_dir / name
        f.write_text(f"#!{sys.executable}\n{body}")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    py3 = bin_dir / "python3"  # RELION runs the CLI runner with the python3 on PATH
    if not py3.exists():
        py3.symlink_to(sys.executable)
