# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Suggested MPI processes, threads, pooled particles and GPUs (jobdefs/compute_advice.py), what they are made
from (env/machine.py: this machine, or a SLURM node), and the job editor's line with Apply."""

import time

import pytest

from cryo_savanna.env import machine as M
from cryo_savanna.env.machine import Gpu, Specs
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import compute_advice as A
from cryo_savanna.jobdefs.cli_tools import CliCatalog
from cryo_savanna.jobdefs.flags import FlagSource
from cryo_savanna.jobdefs.model import JobDefs

DEFS = JobDefs.load()
ENV = RelionEnv({})


def box(cpus=32, ram=256.0, gpus=4, mem=24.0, queue=False, **kw) -> Specs:
    return Specs("a node" if queue else "this machine", cpus, ram, [Gpu(i, "RTX 3090", mem) for i in range(gpus)],
                 queue=queue, **kw)


def advise(form: str, values: dict, specs: Specs, particles=None, limit=32) -> A.Advice:
    v = DEFS.variant(form)
    vals = v.default_values(ENV) | values
    a = A.advise(v.key, v.resolved_options(ENV), vals, specs, particles, limit)
    assert a is not None
    return a


# ---------------------------------------------------------------------------- relion_refine
def test_refine3d_on_gpus_as_relions_schemes():
    a = advise("relion.refine3d", {"use_gpu": "Yes"}, box(), A.Particles(360, 200_000))
    assert a.values == {"nr_mpi": "5", "nr_threads": "6", "nr_pool": "30"}  # 1 follower per GPU, 5 × 6 <= 32
    assert any("1 follower per GPU" in w for w in a.why)
    assert "box 360 px" in a.basis and "200,000 particles" in a.basis


def test_two_followers_per_gpu_when_memory_and_cpus_allow():
    a = advise("relion.class3d", {"use_gpu": "Yes", "nr_classes": "4"}, box(cpus=64, gpus=2), A.Particles(200, None))
    assert a.values["nr_mpi"] == "5" and a.values["nr_threads"] == "12"  # 2 × 2 followers + leader
    big = advise("relion.class3d", {"use_gpu": "Yes", "nr_classes": "8"}, box(cpus=64, gpus=2), A.Particles(400, None))
    assert big.values["nr_mpi"] == "3"  # 8 classes at box 400 take most of a GPU: one each


def test_gold_standard_needs_an_odd_count():
    one_gpu = advise("relion.refine3d", {"use_gpu": "Yes", "nr_mpi": "4"}, box(gpus=1), A.Particles(128, None))
    assert one_gpu.values["nr_mpi"] == "3"  # two halves: two followers share the GPU
    assert any("odd number of MPI processes" in w for w in one_gpu.warnings)
    three = advise("relion.refine3d", {"use_gpu": "Yes"}, box(cpus=16, gpus=3), A.Particles(128, None))
    assert int(three.values["nr_mpi"]) % 2 == 1 and int(three.values["nr_mpi"]) >= 3


def test_values_that_are_no_whole_number():
    # "inf", "nan" or "1e999" typed in the MPI field: taken as not given, not an error in the editor
    for typed in ("inf", "nan", "1e999", "-inf"):
        a = advise("relion.refine3d", {"use_gpu": "Yes", "nr_mpi": typed}, box(), A.Particles(360, None))
        assert a.values["nr_mpi"] == "5"


def test_vdam_runs_without_mpi():
    a = advise("relion.class2d", {"use_gpu": "Yes", "do_grad": "Yes", "nr_mpi": "5"}, box(gpus=2))
    assert a.values == {"nr_mpi": "1", "nr_threads": "12", "nr_pool": "30"}  # as RELION's Schemes: 12 on 2 GPUs
    assert any("VDAM cannot run with MPI" in w for w in a.warnings)
    ini = advise("relion.initialmodel", {"use_gpu": "No"}, box(cpus=24))
    assert ini.values == {"nr_mpi": "1", "nr_threads": "24", "nr_pool": "3"}


def test_gpu_ids_typed_are_the_gpus_used():
    a = advise("relion.class3d", {"use_gpu": "Yes", "gpu_ids": "0:1"}, box(), A.Particles(128, None))
    assert a.values["gpu_ids"] == "0:1" and a.why[0].startswith("2 of the 4 GPUs")
    assert a.values["nr_mpi"] == "5"  # 2 GPUs, 2 followers each
    vdam = advise("relion.class2d", {"use_gpu": "Yes", "do_grad": "Yes", "gpu_ids": "0:1"}, box())
    assert vdam.values["gpu_ids"] == "0,1"  # one process: its threads over both GPUs
    bad = advise("relion.refine3d", {"use_gpu": "Yes", "gpu_ids": "0:7"}, box())
    assert any("GPU 7 is not on this machine" in w for w in bad.warnings)


def test_memory_warnings():
    small = advise("relion.refine3d", {"use_gpu": "Yes"}, box(mem=8.0), A.Particles(600, None))
    assert any("more than the 8.0 GB of a GPU" in w and "Skip padding" in w for w in small.warnings)
    pre = advise("relion.refine3d", {"use_gpu": "No", "do_preread_images": "Yes"}, box(ram=64.0),
                 A.Particles(400, 1_000_000))
    assert any("Pre-read all particles" in w for w in pre.warnings)
    assert pre.values["nr_mpi"] == "3"  # as few followers as two halves allow
    over = advise("relion.class3d", {"nr_mpi": "9", "nr_threads": "8"}, box())
    assert any("9 MPI × 8 threads = 72 is more than the 32 CPUs" in w for w in over.warnings)


def test_unknown_box_or_gpu_memory_is_one_per_gpu():
    a = advise("relion.class3d", {"use_gpu": "Yes"}, box(mem=None), A.Particles(128, None))  # type: ignore[arg-type]
    assert a.values["nr_mpi"] == "5" and any("GPU memory not known" in w for w in a.why)
    b = advise("relion.class3d", {"use_gpu": "Yes"}, box(), None)
    assert b.values["nr_mpi"] == "5" and any("box size of the particles not known" in w for w in b.why)


# ---------------------------------------------------------------------------- the other programs
def test_cpu_programs():
    assert advise("relion.ctffind", {}, box(cpus=24)).values == {"nr_mpi": "24"}
    assert advise("relion.extract", {}, box(cpus=64)).values == {"nr_mpi": "16"}  # it waits for the disk
    assert advise("relion.ctfrefine", {}, box(cpus=32)).values == {"nr_mpi": "8", "nr_threads": "4"}
    mc = advise("relion.motioncorr", {"do_own_motioncor": "Yes"}, box(cpus=48))
    assert mc.values["nr_mpi"] == "4" and mc.values["nr_threads"] == "12"
    assert advise("relion.maskcreate", {}, box(cpus=8)).values == {"nr_threads": "8"}


def test_what_relion_refuses_with_mpi():
    train = advise("relion.polish", {"do_param_optim": "Yes", "nr_mpi": "4"}, box(cpus=16))
    assert train.values == {"nr_mpi": "1", "nr_threads": "16"} and train.warnings
    assert advise("relion.localres", {"do_relion_locres": "No"}, box()).values == {"nr_mpi": "1"}
    assert advise("relion.localres", {"do_relion_locres": "Yes"}, box(cpus=12)).values == {"nr_mpi": "12"}


def test_one_process_per_gpu():
    mc2 = advise("relion.motioncorr", {"do_own_motioncor": "No", "gpu_ids": "0:1:2"}, box())
    assert mc2.values == {"nr_mpi": "3", "nr_threads": "1", "gpu_ids": "0:1:2"}
    topaz = advise("relion.autopick", {"do_topaz": "Yes", "do_topaz_pick": "Yes"}, box(gpus=2))
    assert topaz.values["nr_mpi"] == "2"
    train = advise("relion.autopick", {"do_topaz": "Yes", "do_topaz_train": "Yes"}, box(cpus=8))
    assert train.values["nr_mpi"] == "1"


def test_local_mpi_limit():
    a = advise("relion.ctffind", {}, box(cpus=128), limit=32)
    assert a.values == {"nr_mpi": "32"} and any("RELION_ERROR_LOCAL_MPI" in w for w in a.why)
    q = advise("relion.ctffind", {}, box(cpus=128, queue=True), limit=32)
    assert q.values == {"nr_mpi": "128"}  # the limit is for local jobs only


def test_queue_dedicated_and_cli_tools():
    a = advise("relion.ctffind", {}, box(cpus=48, queue=True, uses_dedicated=True))
    assert a.values == {"nr_mpi": "48", "min_dedicated": "48"}
    cli = CliCatalog(DEFS, ENV, FlagSource(which=lambda p: None)).variant("relion_reconstruct")
    vals = cli.default_values(ENV)
    local = A.advise(cli.key, cli.options, vals, box(cpus=16), None, 32)
    queued = A.advise(cli.key, cli.options, vals, box(cpus=16, queue=True), None, 32)
    assert local is not None and queued is not None
    assert local.values["nr_mpi"] != "1" and queued.values["nr_mpi"] == "1"  # External jobs: no MPI slots


def test_forms_not_covered():
    v = DEFS.variant("relion.postprocess")
    assert A.advise(v.key, v.resolved_options(ENV), v.default_values(ENV), box()) is None
    assert not A.covered("relion.external", {"nr_threads"})


# ---------------------------------------------------------------------------- the input particles
def test_read_particles(spa_project):
    p = A.read_particles(spa_project, "Extract/job005/particles.star")
    assert p is not None and p.box == 64 and p.count and p.count > 0
    assert A.read_particles(spa_project, "nothing/here.star") is None
    assert A.input_star("relion.refine3d", {"fn_img": "a.star", "fn_cont": ""}) == "a.star"
    assert A.input_star("relion.refine3d", {"fn_img": "a.star", "fn_cont": "b_optimiser.star"}) == "b_optimiser.star"
    assert A.input_star("relion.multibody", {"fn_in": "r_optimiser.star"}) == "r_optimiser.star"
    tomo = {"in_optimisation": "o.star", "in_particles": "p.star", "use_direct_entries": "No"}
    assert A.input_star("relion.refine3d:tomo", tomo) == "o.star"
    assert A.input_star("relion.refine3d:tomo", tomo | {"use_direct_entries": "Yes"}) == "p.star"


# ---------------------------------------------------------------------------- the machine and the queue
def test_nvidia_smi_and_cuda_visible_devices():
    found = M.parse_nvidia_smi("0, GPU-aaa, NVIDIA GeForce RTX 3090, 24576\n1, GPU-bbb, NVIDIA A100-SXM4-80GB, 81920\n")
    assert [(g.index, g.name, g.mem_gb) for g in M.visible_gpus(found, {})] == \
        [(0, "RTX 3090", 24.0), (1, "A100-SXM4-80GB", 80.0)]
    only = M.visible_gpus(found, {"CUDA_VISIBLE_DEVICES": "1"})
    assert [(g.index, g.mem_gb) for g in only] == [(0, 80.0)]  # renumbered from 0
    assert M.visible_gpus(found, {"CUDA_VISIBLE_DEVICES": "GPU-bbb"})[0].mem_gb == 80.0
    assert M.visible_gpus(found, {"CUDA_VISIBLE_DEVICES": ""}) == []


def test_local_specs_without_gpus():
    s = M.local_specs({}, run=lambda cmd: "", which=lambda p: None)
    assert s.cpus >= 1 and s.gpus == [] and s.where == "this machine"


TEMPLATE = """#!/bin/bash
#SBATCH --job-name=XXXnameXXX
#SBATCH --partition=XXXqueueXXX
#SBATCH --ntasks=XXXmpinodesXXX
#SBATCH --cpus-per-task=XXXthreadsXXX
#SBATCH --gres=gpu:XXXextra1XXX
#SBATCH --mem=XXXextra2XXX
#SBATCH --nodes=XXXnodesXXX
mpirun XXXcommandXXX
"""


def test_queue_request_from_the_template():
    r = M.parse_queue_request(TEMPLATE, {"queuename": "gpu", "qsub_extra1": "2", "qsub_extra2": "120G"})
    assert (r.partition, r.gpus, r.mem_gb, r.uses_dedicated, r.is_slurm) == ("gpu", 2, 120.0, True, True)
    assert M.parse_queue_request("#SBATCH -p short\n#SBATCH --gpus-per-node=a100:4\n", {}).gpus == 4
    assert M.parse_queue_request("#SBATCH -p short -G 3\n", {}).gpus == 3
    assert M.parse_queue_request("#SBATCH --gres=gpu:a100:2,tmpfs:10G --mem 64000\n", {}).mem_gb == 62.5
    assert M.parse_queue_request("#SBATCH --gres=gpu:XXXextra5XXX\n", {}).gpus is None  # no such field
    assert not M.parse_queue_request("#$ -pe XXXqueueXXX XXXnodesXXX\n", {}).is_slurm
    # --mem=0 is all the memory of the node: no request of its own (the node's RAM counts), not 0 GB
    assert M.parse_queue_request("#SBATCH -p gpu --mem=0\n", {}).mem_gb is None


def test_sinfo_and_gpu_memory():
    kinds = M.parse_sinfo("64|512000|gpu:a100:4(S:0-1)|10\n48+|256000|gpu:4|2\n32|128000|(null)|5\n")
    assert [(k.cpus, k.gpus, k.gpu_type, k.nodes) for k in kinds] == \
        [(64, 4, "a100", 10), (48, 4, "", 2), (32, 0, "", 5)]
    assert M.gpu_memory_of("a100_80gb") == M.gpu_memory_of("a100-80g") == 80.0
    assert M.gpu_memory_of("a100") == 40.0 and M.gpu_memory_of("rtx_3090") == 24.0
    assert M.gpu_memory_of("mystery") is None


def test_queue_specs():
    calls = []

    def run(cmd):
        calls.append(cmd)
        return {"%P": "cpu*\ngpu\n"}.get(cmd[-1], "64|512000|gpu:a100:4(S:0-1)|10\n32|128000|(null)|4\n")

    s = M.queue_specs(TEMPLATE, {"queuename": "gpu", "qsub_extra1": "2", "qsub_extra2": "100G"}, run,
                      which=lambda p: "/usr/bin/" + p)
    assert isinstance(s, Specs) and s.queue and s.where == "partition gpu"
    assert (s.cpus, len(s.gpus), s.gpu_mem_gb, s.usable_ram_gb) == (64, 2, 40.0, 100.0)  # GPU nodes only
    assert ["sinfo", "-h", "-p", "gpu", "-o", "%c|%m|%G|%D"] in calls
    none = M.queue_specs("#SBATCH --ntasks=4\n", {}, run, which=lambda p: "/usr/bin/" + p)
    assert isinstance(none, Specs) and none.where == "partition cpu"  # the default partition
    assert "sinfo not found" in M.queue_specs(TEMPLATE, {}, run, which=lambda p: None)  # type: ignore[operator]
    assert "no #SBATCH" in M.queue_specs("#$ -pe x 1\n", {}, run, which=lambda p: "/x")  # type: ignore[operator]


def test_specs_cache_looks_once_in_a_thread():
    n = []
    cache = M.SpecsCache(local=lambda: n.append(1) or box())
    ready = []
    first = cache.local(lambda: ready.append(1))
    deadline = time.monotonic() + 5
    while not ready and time.monotonic() < deadline:
        time.sleep(0.01)
    assert first is None and ready == [1]
    assert isinstance(cache.local(), Specs) and n == [1]  # known now: not looked at again


# ---------------------------------------------------------------------------- the job editor
@pytest.fixture
def editor(qtbot, spa_project):
    pytest.importorskip("pytestqt")
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.snapshot import job_type_names
    from cryo_savanna.ui.job_editor import JobEditor
    e = JobEditor(ENV, job_type_names(DEFS), flag_source=FlagSource(which=lambda p: None))
    qtbot.addWidget(e)
    e.specs_cache = M.SpecsCache(local=lambda: box(), queue=lambda t, v: box(cpus=48, gpus=2, queue=True))
    e.set_project(spa_project, Pipeline.read(spa_project))
    return e


def test_editor_suggests_and_applies(editor, qtbot):
    editor.new_job(DEFS.variant("relion.refine3d"))
    editor.widgets["fn_img"].set_value("Extract/job005/particles.star")
    editor.widgets["use_gpu"].set_value("Yes")
    qtbot.waitUntil(lambda: "box 64 px" in editor.compute_hint(), timeout=5000)
    hint = editor.compute_hint()
    assert "Suggested: " in hint and "for this machine" in hint and "href='apply'" in hint
    assert "odd number of MPI processes" in hint  # RELION's default of 1
    assert editor.apply_advice()
    assert editor.widgets["nr_mpi"].value() == "5" and editor.widgets["nr_pool"].value() == "30"
    assert "As suggested" in editor.compute_hint() and "href='apply'" not in editor.compute_hint()
    assert not editor.apply_advice()  # nothing left to change


def test_editor_on_the_queue(editor, qtbot, tmp_path):
    tpl = tmp_path / "sbatch.sh"
    tpl.write_text(TEMPLATE)
    editor.new_job(DEFS.variant("relion.ctffind"))
    editor.widgets["do_queue"].set_value("Yes")
    editor.widgets["qsubscript"].set_value(str(tpl))
    qtbot.waitUntil(lambda: "a node" in editor.compute_hint(), timeout=5000)
    editor.apply_advice()
    assert editor.widgets["nr_mpi"].value() == "48"
    editor.widgets["qsubscript"].set_value(str(tmp_path / "missing.sh"))
    qtbot.waitUntil(lambda: "cannot be read" in editor.compute_hint(), timeout=5000)


def test_editor_has_no_line_on_other_forms(editor):
    editor.new_job(DEFS.variant("relion.postprocess"))
    assert editor.compute_hints == [] and editor.compute_hint() == ""
    editor.new_job(DEFS.variant("relion.class3d"))
    assert len(editor.compute_hints) == 2  # under the GPUs (Compute) and under the threads (Running)
    editor.show_help("nr_mpi")
    assert "SLURM partition" in editor.help.toPlainText()
