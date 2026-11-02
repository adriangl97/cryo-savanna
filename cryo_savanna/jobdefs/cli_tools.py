# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Command-line programs as jobs: a job form from --help, run as a RELION External job.

A CLI tool job is a normal External job (relion.external) whose executable is
"python3 .cryo-savanna/cli_runner.py <program>" and whose "Additional arguments" are the program's own
arguments, so relion_pipeliner runs it, the pipeline links it to its inputs and outputs, and the
original GUI can open, continue and rerun it. The runner is a copy in the project (``install_runner``,
done by project/pipeliner.py before such a job is added or run), named relative to the project as
RELION runs every job from there: a job keeps working when cryo-savanna is moved, updated or
reinstalled, and on compute nodes that see the project but not cryo-savanna's installation. ``CliVariant`` is the form: every flag of the
program as an option (tabs from its --help sections), plus MPI and queue settings.

- Flags: an on/off switch is passed when "Yes"; a value is passed when it differs from the
  program's default (always, for outputs).
- Outputs (flags such as --o that name a file or directory to write) are names inside the job
  directory: "map.mrc" becomes "External/job012/map.mrc"; an empty output directory is the job
  directory itself. A value with a folder in it is passed as typed.
- Inputs that are files of the pipeline are also put in External's input fields (particles,
  micrographs, movies, coordinates, map, mask), so the job hangs under the jobs it reads from.
- MPI: the External parameter "mpi" (cli_runner starts <program>_mpi with $RELION_MPIRUN).
"""

from __future__ import annotations

import os
import re
import shlex
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from cryo_savanna import cli_runner
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.cli_catalog import FLAG_TOOLS, POSITIONAL, PositionalTool, cli_programs
from cryo_savanna.jobdefs.flags import Flag, FlagSource, _RAW_TOKEN, _unquote, shell_word
from cryo_savanna.jobdefs.model import JobDefs, JobVariant, OptionDef, Row, Tab
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(cli_runner.__file__).resolve())  # the source of the copy in each project
RUNNER_IN_PROJECT = ".cryo-savanna/cli_runner.py"
EXTERNAL = "relion.external"
NR_MPI, EXTRA_ARGS, STDOUT = "nr_mpi", "other_args", "stdout"
RUNNING_KEYS = ("do_queue", "queuename", "qsub", "qsubscript", "min_dedicated")
# External's input fields, and the node kinds that go in each
INPUT_SLOTS = (("in_mov", "MicrographMovieGroupMetadata"), ("in_mic", "MicrographGroupMetadata"),
               ("in_part", "ParticleGroupMetadata"), ("in_coords", "MicrographCoordsGroup"),
               ("in_3dref", "DensityMap"), ("in_mask", "Mask3D"))
NONE_DEFAULTS = ("", "None", "NULL")

_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_JOB_DIR = re.compile(r"^External/job\d{3,}/")  # a CLI tool job's own directory (an output named in another stays)
_OUTPUT_NAME = re.compile(r"^--(o|out|odir|oc|o2|o_\w+)$")
_INPUT_NAME = re.compile(r"^--(i|i\d|i_\w+|in|in_\w+|ref\d?|mask\d?|fn_\w+|t|p|mot|opt|ios|ts|tomograms|particles|"
                         r"trajectories|motion|data|model|bodies|map|template|u|angles)$")


# ---------------------------------------------------------------------------- flag kinds

def _is_number(s: str) -> bool:
    return bool(_NUMBER.match(s.strip()))


def output_kind(flag: Flag) -> str:
    """"dir" or "file" for a flag naming something the program writes, "suffix" for an output name
    that is a suffix when the input is a STAR file (image_handler, ctf_toolbox --o), else ""."""
    if flag.is_switch or (flag.default and _is_number(flag.default)):
        return ""
    h = flag.help.lower()
    if "insert this string" in h:
        return "suffix"
    if "to be added to" in h or (flag.default or "").startswith("_"):  # always a suffix
        return ""
    named = bool(_OUTPUT_NAME.match(flag.name))
    described = (h.startswith("output") or "name for output" in h or "rootname for output" in h) and \
        any(w in h for w in ("name", "file", "path", "director", "root", "map", "stack"))
    if not (named or described):
        return ""
    return "dir" if "director" in h or (flag.default or "").endswith("/") else "file"


def is_input(flag: Flag) -> bool:
    if flag.is_switch or output_kind(flag) or (flag.default and _is_number(flag.default)):
        return False
    h = flag.help.lower()
    return bool(_INPUT_NAME.match(flag.name)) or h.startswith(("input", "star file", "tilt sequence", "tilt stack",
                                                              "optimisation set", "particle set", "tomogram",
                                                              "reference map", "image to"))


def _option_for(flag: Flag) -> OptionDef:
    label = flag.name
    out = output_kind(flag)
    if flag.is_switch:
        return OptionDef(flag.name, "boolean", label, "No", flag.help)
    default = "" if flag.default in NONE_DEFAULTS or flag.default is None else flag.default
    if out:
        default = _job_relative(default) if default not in ("./", ".") else ""
        note = {"dir": " [Written in the job directory; empty = the job directory itself.]",
                "file": " [Written in the job directory unless the name has a folder in it.]",
                "suffix": " [With a single input file: written in the job directory unless the name has a "
                          "folder in it. With a STAR file as input: passed as typed.]"}[out]
        return OptionDef(flag.name, "any", label, default, flag.help + note)
    if is_input(flag):
        return OptionDef(flag.name, "inputnode", label, default, flag.help, node_type="*")
    req = " [Required.]" if flag.default is None else ""
    return OptionDef(flag.name, "any", label, default, flag.help + req)


def _job_relative(value: str) -> str:
    return value[2:] if value.startswith("./") else value


def output_path(value: str, kind: str, job_dir: str, star_input: bool = False) -> str | None:
    """The path passed for an output flag (None: not passed)."""
    value = _job_relative(value.strip())
    if not value:
        return job_dir if kind == "dir" else None
    if kind == "suffix" and star_input:
        return value
    if value.startswith("/") or "/" in value.rstrip("/"):
        return value
    return job_dir + value


def _words(text: str) -> list[str]:
    """A list typed in shell syntax as its words; with a quote left open ("it's"), split at spaces."""
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _arg(value: str, glob: bool = False) -> str:
    return shell_word(value, glob)


# ---------------------------------------------------------------------------- the form

@dataclass
class CliVariant(JobVariant):
    program: str = ""
    description: str = ""
    flags: dict[str, Flag] | None = None
    positional: PositionalTool | None = None
    has_mpi: bool = False
    source: str = ""
    external: JobVariant | None = None
    env: RelionEnv | None = None

    @property
    def is_cli(self) -> bool:
        return True

    def resolved_options(self, env: RelionEnv) -> dict[str, OptionDef]:
        return self.options

    def resolved_tabs(self, env: RelionEnv) -> list[Tab]:
        return list(self.tabs)

    def default_values(self, env: RelionEnv) -> dict[str, str]:
        return {k: o.default for k, o in self.options.items()}

    def new_jobstar(self, env: RelionEnv, is_continue: bool = False) -> JobStar:
        return self.make_jobstar(self.default_values(env), is_continue, "External/job001/", None)

    # -- form values -> External job.star ---------------------------------------------------
    def tool_args(self, values: dict[str, str], job_dir: str) -> list[str]:
        """The program's arguments, as written to "Additional arguments"."""
        out: list[str] = []
        if self.positional is not None:
            for a in self.positional.args:
                v = values.get(a.key, "").strip()
                if a.kind == "output":
                    v = output_path(v, "file", job_dir) or ""
                if a.multi:
                    out += [_arg(t, glob=True) for t in _words(v)]  # typed as a shell list of files
                else:
                    out.append(_arg(v))  # '' when empty, so the arguments keep their places
            return out
        for name, flag in self.flags.items():
            v = values.get(name, self.options[name].default)
            if flag.is_switch:
                if v == "Yes":
                    out.append(name)
                continue
            kind = output_kind(flag)
            if kind:
                star_input = values.get("--i", "").strip().endswith(".star")
                path = output_path(v, kind, job_dir, star_input) if kind != "suffix" or v.strip() else None
                if path is not None:
                    out += [name, _arg(path)]
            elif v.strip() and (v != self.options[name].default or flag.default is None):
                out += [name, _arg(v.strip())]
        return out

    def make_jobstar(self, values: dict[str, str], is_continue: bool, job_dir: str,
                     pipeline: Pipeline | None) -> JobStar:
        js = self.external.new_jobstar(self.env, is_continue)
        js.is_tomo = self.is_tomo
        o = js.options
        o["fn_exe"] = runner_command(self.program)
        params: list[tuple[str, str]] = []
        if self.has_mpi and values.get(NR_MPI, "1") not in ("", "1"):
            params.append(("mpi", values[NR_MPI]))
        if self.positional is not None and self.positional.stdout and values.get(STDOUT, "").strip():
            params.append(("stdout", values[STDOUT].strip()))
        for i, (label, value) in enumerate(params, start=1):
            o[f"param{i}_label"], o[f"param{i}_value"] = label, value
        if self.flags and "--j" in self.flags:  # what a queue script reserves (XXXthreadsXXX)
            o["nr_threads"] = values.get("--j", "1") or "1"
        for k in RUNNING_KEYS:
            if k in values:
                o[k] = values[k]
        for k in self.options:
            if k.startswith("qsub_extra"):
                o[k] = values.get(k, "")
        args = self.tool_args(values, job_dir)
        extra = values.get(EXTRA_ARGS, "").strip()
        o[EXTRA_ARGS] = " ".join(args + ([extra] if extra else []))
        if pipeline is not None:
            self._fill_inputs(o, args, pipeline)
        return js

    def _fill_inputs(self, o: dict[str, str], args: list[str], pipeline: Pipeline) -> None:
        """Pipeline files among the arguments go in External's input fields (one per field)."""
        files = []
        for t in args:
            name = _unquote(t)
            if name in pipeline.nodes and pipeline.nodes[name].producer and name not in files:
                files.append(name)
        free = [k for k, _ in INPUT_SLOTS]
        left = []
        for f in files:
            kind = pipeline.nodes[f].kind
            slot = next((k for k, want in INPUT_SLOTS if want == kind and k in free), None)
            if slot is None:
                left.append(f)
            else:
                o[slot] = f
                free.remove(slot)
        for f in left:  # RELION links an input to the existing node by name, whatever the field
            if not free:
                break
            o[free.pop(0)] = f

    # -- External job.star -> form values -----------------------------------------------------
    def values_from_jobstar(self, js: JobStar, env: RelionEnv) -> tuple[dict[str, str], list[str]]:
        values = self.default_values(env)
        o = js.options
        for k in RUNNING_KEYS:
            if k in o and k in values:
                values[k] = o[k]
        for k in values:
            if k.startswith("qsub_extra") and k in o:
                values[k] = o[k]
        for i in range(1, 11):
            label, value = o.get(f"param{i}_label", ""), o.get(f"param{i}_value", "")
            if label == "mpi" and self.has_mpi:
                values[NR_MPI] = value
            elif label == "stdout" and STDOUT in values:
                values[STDOUT] = value
        tokens = [m.group(0) for m in _RAW_TOKEN.finditer(o.get(EXTRA_ARGS, ""))]
        extra = self._read_positional(tokens, values) if self.positional else self._read_flags(tokens, values)
        values[EXTRA_ARGS] = " ".join(extra)
        return values, []

    def _read_flags(self, tokens: list[str], values: dict[str, str]) -> list[str]:
        for name, flag in self.flags.items():  # an output not given was cleared
            if output_kind(flag):
                values[name] = ""
        extra: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            flag = self.flags.get(t)
            if flag is None:
                extra.append(t)
                i += 1
            elif flag.is_switch:
                values[t] = "Yes"
                i += 1
            else:
                v = _unquote(tokens[i + 1]) if i + 1 < len(tokens) else ""
                kind = output_kind(flag)
                if kind:
                    v = _JOB_DIR.sub("", v, count=1) if _JOB_DIR.match(v) else v
                    if kind == "dir" and v in ("", "/"):
                        v = ""
                values[t] = v
                i += 2
        return extra

    def _read_positional(self, tokens: list[str], values: dict[str, str]) -> list[str]:
        args = self.positional.args
        words = [_unquote(t) for t in tokens]
        multi = next((i for i, a in enumerate(args) if a.multi), None)
        before = args if multi is None else args[:multi]
        after = () if multi is None else args[multi + 1:]
        for a in before:
            values[a.key] = words.pop(0) if words else ""
        for a in reversed(after):
            values[a.key] = words.pop() if words else ""
        if multi is not None:
            values[args[multi].key] = " ".join(shlex.quote(w) if " " in w else w for w in words)
            words = []
        for a in args:
            if a.kind == "output" and _JOB_DIR.match(values[a.key]):
                values[a.key] = _JOB_DIR.sub("", values[a.key], count=1)
        return words


def runner_command(program: str) -> str:
    """External's executable for a CLI tool. Through python3, so the runner needs no execute bit."""
    return f"python3 {RUNNER_IN_PROJECT} {program}"


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's cli_runner.py into the project (RUNNER_IN_PROJECT), if it is not there already.

    Replaced in one step (a job starting now reads either copy whole). Jobs made before it existed
    name the runner of the installation that made them, which still works while that is in place.
    """
    dst = Path(project_dir) / RUNNER_IN_PROJECT
    src = Path(RUNNER).read_bytes()
    try:
        if dst.read_bytes() == src:
            return dst
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")  # unique also across hosts (a shared project)
    tmp.write_bytes(src)
    os.replace(tmp, dst)
    return dst


def runner_program(fn_exe: str) -> str | None:
    """The program in an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and any(Path(p.strip("'\"")).name == "cli_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def external_program(fn_exe: str) -> str:
    """What an External job runs: the CLI tool for jobs from that menu, else its executable's name."""
    program = runner_program(fn_exe)
    if program is not None:
        return program
    parts = fn_exe.split()
    return Path(parts[0]).name if parts else ""


def cli_program_of(js: JobStar) -> str | None:
    """The program of a CLI tool job (an External job started through cli_runner), else None."""
    return runner_program(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


# ---------------------------------------------------------------------------- building forms

class CliCatalog:
    """Forms for the CLI tools, built on first use from the loaded RELION's --help."""

    def __init__(self, defs: JobDefs, env: RelionEnv, flag_source: FlagSource):
        self.defs, self.env, self.flag_source = defs, env, flag_source
        self._cache: dict[tuple[str, bool], CliVariant | None] = {}

    def programs(self) -> list[str]:
        return cli_programs()

    def variant(self, program: str, tomo: bool = False) -> CliVariant | None:
        key = (program, tomo)
        if key not in self._cache:
            self._cache[key] = self._build(program, tomo)
        return self._cache[key]

    def _build(self, program: str, tomo: bool) -> CliVariant | None:
        external = self.defs.variant(EXTERNAL, tomo)
        ext_opts = external.resolved_options(self.env)
        options: dict[str, OptionDef] = {}
        tabs: list[Tab] = []
        flags: dict[str, Flag] | None = None
        positional = POSITIONAL.get(program)
        source = ""
        if positional is not None:
            rows = []
            for a in positional.args:
                kind = {"input": "inputnode", "output": "any"}.get(a.kind, "any")
                hint = " [Written in the job directory unless the name has a folder in it.]" if a.kind == "output" else ""
                options[a.key] = OptionDef(a.key, kind, a.label, a.default, a.help + hint, node_type="*")
                rows.append(Row((a.key,)))
            if positional.stdout:
                options[STDOUT] = OptionDef(STDOUT, "any", "Save the output to:", positional.stdout,
                                            "What the program prints is saved to this file in the job directory "
                                            "(empty: only in the job's output log).")
                rows.append(Row((STDOUT,)))
            tabs.append(Tab("Arguments", tuple(rows)))
            source = f"usage: {positional.usage}"
        else:
            pf = self.flag_source.flags(program)
            if pf is None:
                return None
            flags = {f.name: f for f in pf.flags}
            sections: dict[str, list[Row]] = {}
            for f in pf.flags:
                options[f.name] = _option_for(f)
                sections.setdefault(f.section or "Options", []).append(Row((f.name,)))
            tabs += [Tab(_tab_label(s), tuple(rows)) for s, rows in sections.items()]
            source = pf.source
        if flags is None:
            has_mpi = False
        elif self.flag_source.which(program):  # ask the loaded RELION
            has_mpi = self.flag_source.which(program + "_mpi") is not None
        else:
            has_mpi = program in _MPI_PROGRAMS
        running: list[Row] = []
        if has_mpi:
            base = self.defs.variant("relion.class2d").resolved_options(self.env)[NR_MPI]
            options[NR_MPI] = replace(base, help=base.help + " Runs " + program + "_mpi under $RELION_MPIRUN "
                                      "(mpirun). Only for local runs: External jobs cannot ask the queue for MPI slots.")
            running.append(Row((NR_MPI,)))
        for k in RUNNING_KEYS:
            options[k] = ext_opts[k]
            running.append(Row((k,)))
            if k == "qsub":
                for ex in [x for x in ext_opts if x.startswith("qsub_extra")]:
                    options[ex] = ext_opts[ex]
                    running.append(Row((ex,)))
        options[EXTRA_ARGS] = replace(ext_opts[EXTRA_ARGS], help="More arguments for " + program +
                                      ", added after the ones set in this form.")
        running.append(Row((EXTRA_ARGS,)))
        tabs.append(Tab("Running", tuple(running)))
        groups = dict(external.groups)  # queue fields follow "Submit to queue?" as in RELION's windows
        tabs = self._queue_groups(tabs, external)
        category, description = (FLAG_TOOLS.get(program) or (positional.category, positional.description))
        return CliVariant(key=f"cli:{program}" + (":tomo" if tomo else ""), label=EXTERNAL, proc_id=external.proc_id,
                          dirname=external.dirname, is_tomo=tomo, options=options, tabs=tabs, groups=groups,
                          program=program, description=description, flags=flags, positional=positional,
                          has_mpi=has_mpi, source=source, external=external, env=self.env)

    @staticmethod
    def _queue_groups(tabs: list[Tab], external: JobVariant) -> list[Tab]:
        """Give the queue rows the group they have in the External job window."""
        group_of = {r.keys: r.group for t in external.tabs for r in t.rows}
        out = []
        for t in tabs:
            if t.label != "Running":
                out.append(t)
                continue
            out.append(Tab(t.label, tuple(replace(r, group=group_of.get(r.keys)) if r.keys in group_of else r
                                          for r in t.rows)))
        return out


# programs with an _mpi version in RELION 5.1 (used when the loaded RELION cannot be asked)
_MPI_PROGRAMS = {"relion_batchrun", "relion_convert_to_tiff", "relion_flex_analyse", "relion_localsym",
                 "relion_reconstruct", "relion_run_align_tiltseries"}


def _tab_label(section: str) -> str:
    s = section.strip().rstrip(":")
    return s[:1].upper() + s[1:] if s else "Options"
