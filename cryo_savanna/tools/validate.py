# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check cryo-savanna against a RELION 5.1 installation.

Run on a machine with RELION loaded (module load relion). Nothing is submitted to a queue
and no RELION computation runs; all work happens in a scratch directory.

    python -m cryo_savanna.tools.validate --out ./cryo_savanna_validation

Checks:
  defaults   For every job type (SPA and tomo), let `relion_pipeliner --editJob` build a full
             job.star from a minimal one, and compare it with the job.star cryo-savanna builds
             for the same environment. Done for the current environment and with all RELION_*
             variables removed (compiled-in defaults of this build).
  pipeliner  In a throwaway project: add jobs from job.star with an alias, run one, run a chain,
             continue, abort, failure, gentle clean, and check that relion_pipeliner waits for
             cryo-savanna's pipeline lock. Uses External jobs running a small shell script.
             Then a job of the CLI tools menu, made and run the way the window does it
             (relion_image_handler on a small map, its form from the loaded RELION's --help), and
             a second one on the first one's output: run by RELION through cryo-savanna's
             cli_runner, outputs registered, the two linked in the pipeline.
  help       Save `--help` of every relion_* program on PATH (input for the Advanced flags panel).

Results go to <out>/report.json and <out>/report.txt, and everything is packed into
<out>.tar.gz (for a bug report). FYI, They name the machine: its host name, the value of every RELION_*
variable (queue names, accounts, template and program paths) and where RELION is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

from cryo_savanna.env.relion_env import RelionEnv, relion_version
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.star import write_star

EDIT_SENTINEL = "cryo_savanna_validation"

# Programs that open windows, start napari, or are not RELION command-line tools.
# relion_filament_selection is a Python wrapper (interactive), relion_trace_amyloids.py is the script
# behind relion_python_trace_amyloids, and relion_qsub.csh is the default queue template.
HELP_SKIP = {"relion", "relion_display", "relion_manualpick", "relion_it.py", "relion_maingui", "relion_schemegui",
             "relion_filament_selection", "relion_trace_amyloids.py", "relion_qsub.csh"}
HELP_SKIP_PREFIX = ("relion_python_",)

EXTERNAL_SCRIPT = r"""#!/usr/bin/env bash
# External job used by cryo-savanna's validation: optional sleep (abortable), optional failure,
# writes one micrographs.star output node.
out=""; sleep_s=0; fail=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --o) out="$2"; shift 2;;
    --sleep) sleep_s="$2"; shift 2;;
    --fail) fail="$2"; shift 2;;
    *) shift;;
  esac
done
echo "started $(date +%s.%N)"
for ((i=0; i<sleep_s; i++)); do
  if [[ -e "${out}RELION_JOB_ABORT_NOW" ]]; then touch "${out}RELION_JOB_EXIT_ABORTED"; exit 0; fi
  sleep 1
done
if [[ "$fail" == "1" ]]; then echo "failing on purpose" >&2; touch "${out}RELION_JOB_EXIT_FAILURE"; exit 1; fi
printf 'data_micrographs\nloop_\n_rlnMicrographName\nmic1.mrc\n' > "${out}micrographs.star"
printf '\ndata_output_nodes\nloop_\n_rlnPipeLineNodeName\n_rlnPipeLineNodeTypeLabel\n%smicrographs.star MicrographGroupMetadata.star.relion\n' "$out" > "${out}RELION_OUTPUT_NODES.star"
echo "finished $(date +%s.%N)"
touch "${out}RELION_JOB_EXIT_SUCCESS"
"""


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("RELION_")}


def pipeliner_version() -> str | None:
    """Version of the relion_pipeliner on PATH ("5.1.1-commit-0a1b2c"), or None if there is none."""
    return relion_version(shutil.which("relion_pipeliner"))


def _run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------------------- defaults

def check_defaults(out: Path, defs: JobDefs, mode: str) -> dict:
    env_vars = dict(os.environ) if mode == "current" else _clean_env()
    env = RelionEnv(env_vars)
    work = out / f"defaults_{mode}"
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, variant in sorted(defs.variants.items()):
        fn = key.replace(":", "_")
        src, dst = work / f"{fn}.in.star", work / f"{fn}.relion.star"
        JobStar(variant.label, False, variant.is_tomo, {"other_args": ""}).write(src)
        r = _run(["relion_pipeliner", "--editJob", src.name, "--editJobOut", dst.name,
                  "--editOption", "other_args", "--editValue", EDIT_SENTINEL], cwd=work, env=env_vars)
        if r.returncode != 0 or not dst.exists():
            results[key] = {"ok": False, "error": (r.stderr or r.stdout)[-2000:]}
            continue
        theirs = JobStar.read(dst)
        ours = variant.new_jobstar(env)
        ours.options["other_args"] = EDIT_SENTINEL
        (work / f"{fn}.cryo-savanna.star").write_text(ours.to_text())
        only_theirs = sorted(set(theirs.options) - set(ours.options))
        only_ours = sorted(set(ours.options) - set(theirs.options))
        differ = {k: {"relion": theirs.options[k], "cryo-savanna": ours.options[k]}
                  for k in sorted(set(theirs.options) & set(ours.options)) if theirs.options[k] != ours.options[k]}
        header_ok = (theirs.type_label, theirs.is_tomo) == (ours.type_label, ours.is_tomo)
        results[key] = {
            "ok": header_ok and not (only_theirs or only_ours or differ),
            "header_ok": header_ok,
            "only_in_relion": only_theirs,
            "only_in_cryo-savanna": only_ours,
            "different_values": differ,
            "identical_file": dst.read_text() == ours.to_text(),
        }
    return results


# ------------------------------------------------------------------------------- pipeliner

class PipelinerChecks:
    def __init__(self, out: Path, defs: JobDefs):
        self.proj = out / "pipeliner_project"
        if self.proj.exists():
            shutil.rmtree(self.proj)
        self.proj.mkdir(parents=True)
        self.defs = defs
        self.env = RelionEnv({})
        self.script = self.proj / "validation_external.sh"
        self.script.write_text(EXTERNAL_SCRIPT)
        self.script.chmod(self.script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        write_star(self.proj / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
        self.results: dict[str, dict] = {}
        self.log: list[str] = []

    def pipeline(self) -> Pipeline:
        return Pipeline.read(self.proj)

    def pipeliner(self, *args: str, timeout: float = 180) -> subprocess.CompletedProcess:
        r = _run(["relion_pipeliner", *args], cwd=self.proj, env=_clean_env(), timeout=timeout)
        self.log.append(f"$ relion_pipeliner {' '.join(args)}\n[rc={r.returncode}]\n{r.stdout}{r.stderr}")
        return r

    def external_jobstar(self, name: str, **params: str) -> Path:
        js = self.defs.variant("relion.external").new_jobstar(self.env)
        js.options.update({"fn_exe": str(self.script), "do_queue": "No", "nr_threads": "1"})
        if "in_mic" in params:
            js.options["in_mic"] = params.pop("in_mic")
        for i, (label, value) in enumerate(params.items(), start=1):
            js.options[f"param{i}_label"] = label
            js.options[f"param{i}_value"] = value
        path = self.proj / f"{name}.job.star"
        js.write(path)
        return path

    def add(self, name: str, alias: str | None = None, **params: str) -> subprocess.CompletedProcess:
        args = ["--addJobFromStar", self.external_jobstar(name, **params).name]
        if alias:
            args += ["--setJobAlias", alias]
        return self.pipeliner(*args)

    def check(self, name: str, fn) -> None:
        t0 = time.time()
        try:
            detail = fn() or {}
            self.results[name] = {"ok": True, **detail}
        except AssertionError as e:
            self.results[name] = {"ok": False, "error": str(e)}
        except Exception:
            self.results[name] = {"ok": False, "error": traceback.format_exc()[-2000:]}
        self.results[name]["seconds"] = round(time.time() - t0, 1)

    # individual checks --------------------------------------------------------------------
    def t_add_with_alias(self):
        r = self.add("job1", alias="first_ext", sleep="1")
        assert r.returncode == 0, r.stderr
        p = self.pipeline()
        pr = p.processes.get("External/job001/")
        assert pr is not None, f"processes: {list(p.processes)}"
        assert pr.status == "Scheduled", pr.status
        js = JobStar.read(self.proj / "External/job001/job.star")
        return {"alias": pr.alias, "status": pr.status, "jobstar_is_tomo": js.is_tomo,
                "files": sorted(os.listdir(self.proj / "External/job001"))}

    def t_run_one(self):
        r = self.pipeliner("--RunJobs", "External/job001/", "--schedule", "cryo_savanna_validation", "--sec_wait_after", "1")
        assert r.returncode == 0, r.stderr
        p = self.pipeline()
        pr = p.processes["External/job001/"]
        assert pr.status == "Succeeded", pr.status
        assert "External/job001/micrographs.star" in pr.outputs, pr.outputs
        return {"leftover_files": sorted(f for f in os.listdir(self.proj) if f.startswith(("RUNNING", "."))),
                "note_txt": (self.proj / "External/job001/note.txt").read_text()[-800:]}

    def t_chain(self):
        assert self.add("job2", sleep="2", in_mic="External/job001/micrographs.star").returncode == 0
        assert self.add("job3", sleep="1", in_mic="External/job002/micrographs.star").returncode == 0
        r = self.pipeliner("--RunJobs", "External/job002/ External/job003/", "--schedule", "cryo_savanna_validation",
                           "--sec_wait_after", "1")
        assert r.returncode == 0, r.stderr
        p = self.pipeline()
        st = {n: p.processes[n].status for n in ("External/job002/", "External/job003/")}
        assert all(s == "Succeeded" for s in st.values()), st
        out2 = (self.proj / "External/job002/run.out").read_text()
        out3 = (self.proj / "External/job003/run.out").read_text()
        end2 = float(out2.split("finished ")[1].split()[0])
        start3 = float(out3.split("started ")[1].split()[0])
        node = p.nodes.get("External/job002/micrographs.star")
        return {"status": st, "job3_started_after_job2": start3 >= end2,
                "job002_output_node_producer": node.producer if node else None,
                "parents_of_job003": p.parents("External/job003/")}

    def t_continue(self):
        js = JobStar.read(self.proj / "External/job001/job.star")
        js.is_continue = True
        js.write(self.proj / "External/job001/job.star")
        r = self.pipeliner("--RunJobs", "External/job001/", "--schedule", "cryo_savanna_validation", "--sec_wait_after", "1")
        assert r.returncode == 0, r.stderr
        p = self.pipeline()
        assert p.processes["External/job001/"].status == "Succeeded", p.processes["External/job001/"].status
        return {"run_out_runs": (self.proj / "External/job001/run.out").read_text().count("started")}

    def t_abort(self):
        assert self.add("job4", sleep="60").returncode == 0
        name = "External/job004/"
        proc = subprocess.Popen(["relion_pipeliner", "--RunJobs", name, "--schedule", "cryo_savanna_validation_abort",
                                 "--sec_wait_after", "1"], cwd=self.proj, env=_clean_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 60
            while time.time() < deadline and self.pipeline().processes[name].status != "Running":
                time.sleep(1)
            assert self.pipeline().processes[name].status == "Running", "job never reached Running"
            (self.proj / name / "RELION_JOB_ABORT_NOW").touch()
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
        status = self.pipeline().processes[name].status
        assert status == "Aborted", status
        return {"pipeliner_rc": proc.returncode}

    def t_failure(self):
        assert self.add("job5", fail="1").returncode == 0
        r = self.pipeliner("--RunJobs", "External/job005/", "--schedule", "cryo_savanna_validation", "--sec_wait_after", "1")
        status = self.pipeline().processes["External/job005/"].status
        assert status == "Failed", status
        return {"pipeliner_rc": r.returncode}

    def t_gentle_clean(self):
        r = self.pipeliner("--gentle_clean", "1")
        assert r.returncode == 0, r.stderr
        return {"remaining_files": sorted(os.listdir(self.proj / "External/job001"))}

    def t_lock_interop(self):
        path = self.external_jobstar("job6", sleep="1")
        with PipelineLock(self.proj, message="cryo-savanna validation"):
            proc = subprocess.Popen(["relion_pipeliner", "--addJobFromStar", path.name], cwd=self.proj, env=_clean_env(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(7)
            waited = proc.poll() is None
            added_while_locked = "External/job006/" in self.pipeline().processes
        proc.wait(timeout=60)
        added = "External/job006/" in self.pipeline().processes
        assert waited and not added_while_locked, "relion_pipeliner did not wait for cryo-savanna's lock"
        assert added and proc.returncode == 0, "job was not added after the lock was released"
        return {"relion_waited_for_lock": waited}

    def _cli_job(self, values: dict[str, str]) -> str:
        """Add and run a relion_image_handler job of the CLI tools menu as the window does (project/pipeliner.py,
        the form from the loaded RELION's --help); wait for it to end. Returns the job's name."""
        from cryo_savanna.jobdefs.cli_tools import CliCatalog
        from cryo_savanna.jobdefs.flags import FlagSource
        from cryo_savanna.project.pipeliner import Pipeliner
        variant = CliCatalog(self.defs, self.env, FlagSource()).variant("relion_image_handler")
        assert variant is not None, "no form for relion_image_handler (its --help could not be read)"
        p = self.pipeline()
        vals = variant.default_values(self.env) | {"do_queue": "No"} | values
        predicted = f"External/job{p.job_counter:03d}/"
        js = variant.make_jobstar(vals, False, predicted, p)
        pl = Pipeliner(self.proj, environ=_clean_env())
        job = pl.add_job(js).job
        if job != predicted:  # as the window does: the outputs go to the job's own directory
            variant.make_jobstar(vals, False, job, p).write(self.proj / job / "job.star")
        rec = pl.run([job], sec_wait_after=1)
        deadline = time.time() + 180
        while time.time() < deadline and self.pipeline().processes[job].status in ("Scheduled", "Running"):
            time.sleep(1)
        run_log = self.proj / rec.log
        self.log.append(f"$ {' '.join(rec.command)}\n{run_log.read_text() if run_log.exists() else ''}")
        return job

    def t_cli_tool(self):
        import mrcfile
        import numpy as np
        with mrcfile.new(self.proj / "cli_input.mrc", overwrite=True) as m:
            m.set_data(np.random.default_rng(0).random((16, 16, 16), dtype=np.float32))
            m.voxel_size = 1.5
        first = self._cli_job({"--i": "cli_input.mrc", "--o": "flipped.mrc", "--invert_hand": "Yes"})
        p = self.pipeline()
        pr, d = p.processes[first], self.proj / first
        out = f"{first}flipped.mrc"
        err = (d / "run.err").read_text()[-1500:] if (d / "run.err").exists() else ""
        assert pr.status == "Succeeded", f"{first}: {pr.status}; run.err: {err}"
        assert (self.proj / out).is_file(), f"no {out}"
        assert out in pr.outputs, f"{out} is not an output of {first} (RELION_OUTPUT_NODES.star): {pr.outputs}"
        run_out = (d / "run.out").read_text()
        assert f"--pipeline_control {first}.cli/" in run_out, "the program did not get the runner's own --pipeline_control"
        note = (d / "note.txt").read_text()
        second = self._cli_job({"--i": out, "--o": "again.mrc", "--invert_hand": "Yes"})
        p = self.pipeline()
        assert p.processes[second].status == "Succeeded", f"{second}: {p.processes[second].status}"
        assert p.parents(second) == [first], f"{second} is not linked to {first}: parents {p.parents(second)}"
        assert JobStar.read(self.proj / second / "job.star").options.get("in_3dref") == out
        assert (self.proj / ".cryo-savanna/cli_runner.py").is_file(), "the runner was not copied into the project"
        return {"jobs": [first, second], "outputs_of_first": pr.outputs, "parents_of_second": p.parents(second),
                "relion_command": next((ln for ln in note.splitlines() if "cli_runner.py" in ln), "")[-400:]}

    def run_all(self) -> dict:
        for name in ("add_with_alias", "run_one", "chain", "continue", "abort", "failure", "gentle_clean",
                     "lock_interop", "cli_tool"):
            self.check(name, getattr(self, f"t_{name}"))
        (self.proj.parent / "pipeliner_log.txt").write_text("\n\n".join(self.log))
        return self.results


# ------------------------------------------------------------------------------- help dumps

def dump_help(out: Path, timeout: float = 15) -> dict:
    """--help of the programs next to the relion_pipeliner in use (not other RELIONs on PATH)."""
    hdir = out / "help"
    hdir.mkdir(parents=True, exist_ok=True)
    scratch = out / "help_cwd"
    scratch.mkdir(exist_ok=True)
    bindir = os.path.dirname(os.path.realpath(shutil.which("relion_pipeliner")))
    progs: dict[str, str] = {}
    for f in os.listdir(bindir):
        p = os.path.join(bindir, f)
        if (f.startswith("relion_") or f == "relion") and os.access(p, os.X_OK) and os.path.isfile(p):
            progs[f] = p
    done, skipped, failed = [], [], {}
    for name, path in sorted(progs.items()):
        if name in HELP_SKIP or name.startswith(HELP_SKIP_PREFIX):
            skipped.append(name)
            continue
        try:
            # Some RELION scripts treat --help as a file name and write files: keep them in scratch.
            r = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL, env=_clean_env(), cwd=scratch)
            (hdir / f"{name}.txt").write_text(f"# {path}\n# rc={r.returncode}\n{r.stdout}\n# --- stderr ---\n{r.stderr}")
            done.append(name)
        except subprocess.TimeoutExpired:
            failed[name] = "timeout"
        except OSError as e:
            failed[name] = str(e)
    return {"programs": len(progs), "dumped": done, "skipped": skipped, "failed": failed, "paths": progs}


# ------------------------------------------------------------------------------- main

def environment_info() -> dict:
    info = {"relion_env": {k: v for k, v in os.environ.items() if k.startswith("RELION_")},
            "pipeliner": shutil.which("relion_pipeliner"), "python": sys.version, "hostname": os.uname().nodename}
    for cmd in (["relion_refine", "--version"], ["relion", "--version"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=_clean_env())
            info[" ".join(cmd)] = (r.stdout + r.stderr).strip()[:2000]
        except (OSError, subprocess.TimeoutExpired) as e:
            info[" ".join(cmd)] = f"failed: {e}"
    return info


def summarize(report: dict) -> str:
    lines = [f"cryo-savanna validation against {report['env'].get('pipeliner')}", ""]
    for mode in ("defaults_current", "defaults_clean"):
        res = report.get(mode)
        if res is None:
            continue
        bad = {k: v for k, v in res.items() if not v["ok"]}
        ident = sum(1 for v in res.values() if v.get("identical_file"))
        lines.append(f"[{mode}] {len(res) - len(bad)}/{len(res)} job types match; {ident} byte-identical job.star files")
        for k, v in bad.items():
            lines.append(f"  - {k}: {json.dumps({x: y for x, y in v.items() if x != 'ok'})[:600]}")
    if "pipeliner" in report:
        lines.append("")
        for k, v in report["pipeliner"].items():
            lines.append(f"[pipeliner] {k}: {'OK' if v['ok'] else 'FAILED'}"
                         + ("" if v["ok"] else f" -- {v.get('error', '')[:300]}"))
    if "help" in report:
        h = report["help"]
        lines += ["", f"[help] {len(h['dumped'])} programs dumped, {len(h['skipped'])} skipped, failed: {h['failed']}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("cryo_savanna_validation"))
    ap.add_argument("--only", choices=("defaults", "pipeliner", "help"), action="append")
    args = ap.parse_args(argv)
    if shutil.which("relion_pipeliner") is None:
        print("relion_pipeliner is not on PATH: load RELION 5.1 first (e.g. module load relion/5.1)", file=sys.stderr)
        return 2
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ("defaults", "pipeliner", "help"))
    defs = JobDefs.load()
    report: dict = {"env": environment_info(), "jobdefs_version": defs.version}

    if "defaults" in only:
        print("checking job.star defaults (current environment) ...", flush=True)
        report["defaults_current"] = check_defaults(out, defs, "current")
        print("checking job.star defaults (no RELION_* variables) ...", flush=True)
        report["defaults_clean"] = check_defaults(out, defs, "clean")
    if "pipeliner" in only:
        print("checking relion_pipeliner behaviour (about 2 minutes) ...", flush=True)
        report["pipeliner"] = PipelinerChecks(out, defs).run_all()
    if "help" in only:
        print("saving --help of relion programs ...", flush=True)
        report["help"] = dump_help(out)

    (out / "report.json").write_text(json.dumps(report, indent=1))
    summary = summarize(report)
    (out / "report.txt").write_text(summary)
    with tarfile.open(f"{out}.tar.gz", "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(summary)
    print(f"Full results: {out}/report.json\nFor a bug report: {out}.tar.gz\n"
          "FYI: It holds this machine's host name, its RELION_* settings (queue names, accounts, paths) and "
          "where RELION is installed: look through report.json first if any of that is private.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
