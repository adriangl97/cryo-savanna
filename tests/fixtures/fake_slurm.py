# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fake squeue / sacct / scancel / sbatch for tests: a JSON state file stands in for slurmctld.

    install(bin_dir, state_file)   writes the four commands into bin_dir
    State(state_file)              read and change the queue from a test

Jobs are dicts: id, state, reason, elapsed, limit, nodes, partition, name, workdir, stdout, end,
max_rss, exit_code, submit_line, submitted (epoch seconds). The fake sbatch reads --job-name and
--output from #SBATCH lines (without --job-name the job is named after the script, as in SLURM),
queues the job as PENDING, prints "Submitted batch job N", then runs the script in the background
(RUNNING, then COMPLETED or FAILED) unless FAKE_SBATCH_HOLD is set, which leaves it pending.
sacct knows -j, --name, -S, -X and -o (also SubmitLine and Submit); with FAKE_SACCT_OLD set it
refuses SubmitLine, as SLURM releases before that field do.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

COMMON = r'''
import fcntl, json, os, sys, time
STATE = os.environ.get("FAKE_SLURM_STATE") or %(state)r
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}

def load():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {"next_id": 1000, "jobs": []}

def change(fn):
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = load()
        out = fn(s)
        tmp = STATE + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, STATE)
        return out

def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default
'''

SQUEUE = r'''
args = sys.argv[1:]
jobs = [j for j in load()["jobs"] if j["state"] in ACTIVE]
if "-j" in args:
    want = set(opt(args, "-j").split(","))
    jobs = [j for j in jobs if j["id"] in want]
if "-O" in args:  # only JobID,StdOut is used
    for j in jobs:
        print(j["id"].ljust(40) + j.get("stdout", ""))
    sys.exit(0)
fmt = opt(args, "-o")
codes = {"%i": "id", "%T": "state", "%r": "reason", "%M": "elapsed", "%l": "limit", "%N": "nodes",
         "%P": "partition", "%j": "name", "%Z": "workdir"}
for j in jobs:
    print("\t".join(str(j.get(codes[c], "")) for c in fmt.split("\t")))
'''

SACCT = r'''
args = sys.argv[1:]
jobs = load()["jobs"]
if "-j" in args:
    want = set(opt(args, "-j").split(","))
    jobs = [j for j in jobs if j["id"] in want]
if "--name" in args:
    want = set(opt(args, "--name").split(","))
    jobs = [j for j in jobs if j.get("name") in want]
if "-S" in args and "-j" not in args:  # sacct -S without -j: jobs submitted since then
    since = time.mktime(time.strptime(opt(args, "-S"), "%Y-%m-%dT%H:%M:%S" if "T" in opt(args, "-S") else "%Y-%m-%d"))
    jobs = [j for j in jobs if j.get("submitted", 0) >= since]
fields = opt(args, "-o").split(",")
if os.environ.get("FAKE_SACCT_OLD") and "SubmitLine" in fields:
    print('sacct: error: Invalid field requested: "SubmitLine"', file=sys.stderr)
    sys.exit(1)
keys = {"JobID": "id", "JobName": "name", "State": "state", "Elapsed": "elapsed", "Start": "start", "End": "end",
        "NodeList": "nodes", "ExitCode": "exit_code", "MaxRSS": "", "AllocCPUS": "cpus", "Partition": "partition",
        "WorkDir": "workdir", "SubmitLine": "submit_line", "Submit": "submitted"}
for j in jobs:
    print("|".join(str(j.get(keys[f], "")) if keys[f] else "" for f in fields))
    if j.get("max_rss") and "-X" not in args:
        step = {"JobID": j["id"] + ".batch", "MaxRSS": j["max_rss"], "State": j["state"]}
        print("|".join(str(step.get(f, "")) for f in fields))
'''

SCANCEL = r'''
ids = set(sys.argv[1:])
def cancel(s):
    for j in s["jobs"]:
        if j["id"] in ids and j["state"] in ACTIVE:
            j["state"], j["end"] = "CANCELLED", time.strftime("%%Y-%%m-%%dT%%H:%%M:%%S")
    with open(STATE + ".cancelled", "a") as fh:
        fh.write(" ".join(sorted(ids)) + "\n")
change(cancel)
'''

SBATCH = r'''
import subprocess
script = sys.argv[-1]
text = open(script).read()
def directive(name, default=""):
    for line in text.splitlines():
        if line.startswith("#SBATCH") and ("--" + name + "=") in line:
            return line.split("=", 1)[1].strip()
    return default
def submit(s):
    i = str(s["next_id"])
    s["next_id"] += 1
    s["jobs"].append({"id": i, "state": "PENDING", "reason": "Priority", "elapsed": "0:00", "limit": "1-00:00:00",
                      "nodes": "", "partition": "gpu", "name": directive("job-name", os.path.basename(script)),
                      "workdir": os.getcwd(), "stdout": os.path.join(os.getcwd(), directive("output", "slurm.out")),
                      "end": "Unknown", "exit_code": "0:0", "cpus": "1", "start": "Unknown",
                      "submit_line": " ".join(["sbatch"] + sys.argv[1:]), "submitted": time.time()})
    return i
i = change(submit)
print("Submitted batch job " + i, flush=True)
if not os.environ.get("FAKE_SBATCH_HOLD"):
    runner = ("import json, os, subprocess, sys, time\n" + %(common)r +
              "\ni = sys.argv[1]\n"
              "def st(state, **kw):\n"
              "    def f(s):\n"
              "        for j in s['jobs']:\n"
              "            if j['id'] == i:\n"
              "                j['state'] = state\n"
              "                j.update(kw)\n"
              "    change(f)\n"
              "st('RUNNING', reason='None', nodes='node01', start=time.strftime('%%Y-%%m-%%dT%%H:%%M:%%S'))\n"
              "rc = subprocess.call(['bash', sys.argv[2]])\n"
              "st('COMPLETED' if rc == 0 else 'FAILED', end=time.strftime('%%Y-%%m-%%dT%%H:%%M:%%S'), "
              "exit_code=str(rc) + ':0')\n")
    subprocess.Popen([sys.executable, "-c", runner, i, script], start_new_session=True,
                     stdout=open(directive("output", os.devnull), "a"), stderr=subprocess.STDOUT)
'''


def install(bin_dir: Path, state_file: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    common = COMMON % {"state": str(state_file)}
    for name, body in (("squeue", SQUEUE), ("sacct", SACCT), ("scancel", SCANCEL), ("sbatch", SBATCH)):
        f = bin_dir / name
        text = body % {"common": common} if name == "sbatch" else body.replace("%%", "%") if name == "scancel" else body
        f.write_text(f"#!{sys.executable}\n{common}\n{text}")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)


class State:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self.save({"next_id": 1000, "jobs": []})

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    def save(self, s: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1))
        os.replace(tmp, self.path)

    def add(self, **job) -> dict:
        s = self.load()
        job.setdefault("id", str(s["next_id"]))
        s["next_id"] = max(s["next_id"], int(job["id"].split("_")[0]) + 1)
        defaults = {"state": "PENDING", "reason": "Priority", "elapsed": "0:00", "limit": "2:00:00", "nodes": "",
                    "partition": "gpu", "name": "relion", "workdir": "", "stdout": "", "end": "Unknown",
                    "exit_code": "0:0", "cpus": "4", "start": "Unknown", "submit_line": "", "submitted": time.time()}
        job = {**defaults, **job}
        s["jobs"].append(job)
        self.save(s)
        return job

    def set(self, job_id: str, **kw) -> None:
        s = self.load()
        for j in s["jobs"]:
            if j["id"] == job_id:
                j.update(kw)
        self.save(s)

    def cancelled(self) -> list[str]:
        f = Path(str(self.path) + ".cancelled")
        return f.read_text().split() if f.exists() else []

    def job(self, job_id: str) -> dict:
        return next(j for j in self.load()["jobs"] if j["id"] == job_id)
