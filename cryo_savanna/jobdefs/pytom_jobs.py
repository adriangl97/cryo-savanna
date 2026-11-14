# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""pytom-match-pick as jobs ("Additional external jobs"): template matching and candidate extraction.

Both are RELION External jobs (relion.external, tomo), so relion_pipeliner runs them (locally or through
the queue, as any job), the pipeline links them to their inputs and outputs, and RELION's own GUI can
open, continue and rerun them. The executable is "python3 .cryo-savanna/pytom_runner.py match|extract"
(a copy of cryo_savanna/pytom_runner.py in the project, with the runner_common.py it imports, put there
before such a job is added or run).
Every option of the form is written to External's "Additional arguments" as ``--key value`` (shell
quoted), which the runner reads after RELION's ``--j``; the form reads them back from there (clone,
continue, overwrite). Pipeline files among the inputs also go in External's input fields, so the job
hangs under the jobs it reads from.

pytom itself is run by the runner from its conda environment (``conda run -n <env>``): nothing of pytom
is imported here.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from cryo_savanna import pytom_runner as R
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import runner_files
from cryo_savanna.jobdefs.cli_tools import INPUT_SLOTS, RUNNING_KEYS
from cryo_savanna.jobdefs.model import Group, JobDefs, JobVariant, OptionDef, Row, Tab
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(R.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/pytom_runner.py"
EXTERNAL = "relion.external"
MATCH, EXTRACT = R.MATCH, R.EXTRACT
TITLES = {MATCH: "pytom template matching", EXTRACT: "pytom candidate extraction"}
DESCRIPTIONS = {
    MATCH: "Template matching in RELION 5 tomograms with pytom-match-pick (pytom_match_template.py), "
           "on one or more GPUs",
    EXTRACT: "Particles from a pytom template matching job (pytom_extract_candidates.py), with thresholds "
             "chosen interactively, as a RELION 5 particles.star (needs a GPU machine, like matching)",
}
ENV_VAR = "CRYO_SAVANNA_PYTOM_ENV"  # default conda environment of pytom
CUDA_ENV_VAR = "CRYO_SAVANNA_CUDA_MODULE"  # default CUDA module
QUEUE_GROUP = 90


def default_env() -> str:
    return os.environ.get(ENV_VAR, "pytom_tm")


_CUDA_MARKS = ("(default)", "(D)", "<D>", "(L)", "<L>")


def _version_key(name: str) -> list:
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", name) if t]


def pick_cuda_module(listing: str) -> str:
    """The CUDA module from the output of 'module -t avail cuda': the one marked default, else the newest."""
    found, default = [], ""
    for line in listing.splitlines():
        name = line.strip()
        marked = any(m in name for m in _CUDA_MARKS)
        for m in _CUDA_MARKS:
            name = name.replace(m, "")
        name = name.strip()
        if not name or name.endswith(":") or " " in name:
            continue  # a folder of modules, or a message
        base = name.split("/")[0].lower()
        if not base.startswith("cuda") or base.startswith("cudnn") or name.endswith("/"):
            continue
        found.append(name)
        if marked and not default:
            default = name
    return default or (max(found, key=_version_key) if found else "")


@functools.lru_cache(maxsize=1)
def default_cuda_module() -> str:
    """The default for "CUDA module": a CUDA module loaded where cryo-savanna runs, else the default (or
    newest) CUDA module available; empty without environment modules. ${CUDA_ENV_VAR} overrides it."""
    if CUDA_ENV_VAR in os.environ:
        return os.environ[CUDA_ENV_VAR].strip()
    for m in os.environ.get("LOADEDMODULES", "").split(":"):
        base = m.split("/")[0].lower()
        if base.startswith("cuda") and not base.startswith("cudnn"):
            return m
    try:
        r = subprocess.run(["bash", "-c", R.MODULE_INIT + "module -t avail cuda 2>&1"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return pick_cuda_module(r.stdout) if r.returncode == 0 else ""


_warming: threading.Thread | None = None


def warm_defaults() -> None:
    """Start looking up default_cuda_module() in the background (once): "module avail" can take many
    seconds on a cluster, and the first pytom form would otherwise wait for it with the window frozen."""
    global _warming
    if _warming is None:
        _warming = threading.Thread(target=default_cuda_module, name="cryo-savanna-cuda-module", daemon=True)
        _warming.start()


def _warmed_cuda_module() -> str:
    """default_cuda_module(), after a look started by warm_defaults() (not a second one beside it)."""
    if _warming is not None and _warming is not threading.current_thread():
        _warming.join()
    return default_cuda_module()


def _o(key, label, default="", help="", kind="any", **kw) -> OptionDef:
    return OptionDef(key, kind, label, default, help, **kw)


def _yn(key, label, default, help) -> OptionDef:
    return OptionDef(key, "boolean", label, default, help)


def _radio(key, label, choices, default, help) -> OptionDef:
    return OptionDef(key, "radio", label, default, help, choices=tuple(choices))


ENV_OPTIONS = [
    _o("cuda_module", "CUDA module:", "", "Environment module(s) loaded with 'module load' before pytom runs "
       "(e.g. cuda/12.4; several separated by spaces), so that pytom (CuPy) finds CUDA. Empty: no module is "
       "loaded and pytom uses the CUDA on the job's PATH. The default is the CUDA module loaded where "
       "cryo-savanna runs, else the default CUDA module of 'module avail'; empty when there are no CUDA "
       f"modules (${CUDA_ENV_VAR} overrides it)."),
    _o("pytom_env", "pytom conda environment:", "", "Name (or path, with a /) of the conda environment where "
       "pytom-match-pick is installed: its programs run with 'conda run -n <environment>'. Empty: pytom's "
       "programs are on the PATH of the job (a module, or an activated environment). The default comes from "
       f"${ENV_VAR}, else 'pytom_tm', the name pytom's installation guide uses."),
    _o("conda", "conda command:", "", "The conda (or mamba) executable for 'conda run'. Empty: $CONDA_EXE where "
       "the job runs, else 'conda' on its PATH. On a cluster, the compute nodes need it too."),
    _radio("log_level", "pytom log level:", ("info", "debug"), "info",
           "pytom's --log. debug also writes template_psf.mrc and template_convolved.mrc (match)."),
]

TOMO_NAMES = _o("tomo_names", "Only these tomograms:", "", "rlnTomoName of the tomograms to use, separated by "
                "spaces. Empty: all of them.")


def match_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        _o("in_tomograms", "Input tomograms:", "", "The tomograms.star of a RELION 5 Reconstruct tomograms (or "
           "Denoise) job. From it and its tilt-series STAR files the job writes pytom's inputs for every tomogram: "
           "tilt angles (.tlt), accumulated dose and defocus per tilt (.txt), voltage, Cs, amplitude contrast "
           "and the voxel size (rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning).", kind="inputnode",
           node_type="TomogramGroupMetadata.star.relion", pattern="Tomogram set STAR file (*.star)"),
        _radio("tomo_kind", "Search in:", tuple(R.TOMOGRAM_KINDS), "Reconstructed",
               "Which tomogram of each row: the reconstruction (rlnTomoReconstructedTomogram), the denoised one "
               "or a half tomogram. pytom recommends tomograms that were not denoised."),
        TOMO_NAMES,
        _radio("tilt_source", "Tilt angles:", ("Nominal (rlnTomoNominalStageTiltAngle)", "Aligned (rlnTomoYTilt)"),
               "Nominal (rlnTomoNominalStageTiltAngle)",
               "Tilt angles for the missing wedge and the per-tilt weights: the stage angles (as pytom's own "
               "RELION 5 reader takes them), or those refined by the tilt-series alignment."),
        _o("voxel_size", "Tomogram voxel size (Å):", "", "Empty: rlnTomoTiltSeriesPixelSize × "
           "rlnTomoTomogramBinning of each tomogram, else its MRC header."),
        _o("fn_template", "Template map:", "", "The template: a map (.mrc) of the particle, e.g. from the EMDB "
           "or a subtomogram average.", kind="inputnode", node_type="DensityMap.mrc",
           pattern="MRC map (*.mrc)"),
        _yn("do_prep_template", "Prepare the template?", "Yes",
            "Resample the map to the tomogram's voxel size with pytom_create_template.py (low-pass filtered at "
            "2 × the final pixel size unless set below, then Fourier-cropped), into template.mrc in the job. "
            "No: the map is used as it is (its voxel size must be the tomogram's)."),
        _o("template_angpix_in", "Input pixel size (Å):", "", "Pixel size of the map. Empty: from its header."),
        _o("template_angpix", "Final pixel size (Å):", "", "Pixel size of the template. Empty: the tomogram's "
           "voxel size (it must be the same)."),
        _o("template_box", "Final box size (px):", "", "Box of the template, in final pixels. Empty: what the "
           "resampling gives (only a larger box is applied). The mask gets the same box."),
        _yn("template_invert", "Invert contrast?", "Yes", "Multiply the template by -1 (--invert). Yes for "
            "a map with white density and tomograms with dark density (as RELION reconstructs them)."),
        _yn("template_mirror", "Mirror?", "No", "Mirror the template (--mirror), e.g. for a map of the "
            "other hand. A mask from a file must then be mirrored too (Prepare the mask? > Mirror?), or it "
            "no longer matches the template; a created sphere or ellipsoid needs nothing."),
        _yn("template_center", "Center by centre of mass?", "No", "--center"),
        _o("template_lowpass", "Low-pass filter (Å):", "", "Low-pass before resampling. Empty: 2 × final pixel."),
        _radio("mask_source", "Mask:", (R.MASK_FILE, R.MASK_CREATE), R.MASK_CREATE,
               "A mask from a file, or a sphere/ellipsoid made by pytom_create_mask.py. pytom's mask is "
               "around the particle in the template's box (values 0 to 1), not the tomogram mask."),
        _o("fn_mask", "Mask file:", "", "A mask (0 to 1) in the template's box.", kind="inputnode",
           node_type="Mask3D.mrc", pattern="MRC map (*.mrc)"),
        _yn("do_prep_mask", "Prepare the mask?", "No", "Resample the mask to the template's pixel size and box "
            "(pytom_create_template.py, then clipped to 0–1), into mask.mrc in the job."),
        _o("mask_angpix_in", "Input pixel size (Å):", "", "Pixel size of the mask file. Empty: from its header."),
        _o("mask_angpix", "Final pixel size (Å):", "", "Empty: the template's pixel size."),
        _o("mask_box", "Final box size (px):", "", "Empty: the template's box (pytom needs the same box)."),
        _yn("mask_mirror", "Mirror?", "No", "Mirror the mask (--mirror), as the template. Yes when the "
            "template is mirrored, so that the mask still matches it."),
        _o("mask_radius", "Radius (px):", "", "Radius in the template's pixels (the x radius of an ellipsoid). "
           "Empty: half the particle diameter, else 0.4 × the box."),
        _o("mask_minor", "Minor radii (px):", "", "Two numbers: the y and z radii of an ellipsoid. Empty: a sphere."),
        _o("mask_sigma", "Soft edge σ (px):", "", "Gaussian fall-off of the edge (pytom suggests 0.5–1 for "
           "voxels of 10–20 Å). Empty: a hard edge."),
        _o("mask_box_create", "Box size (px):", "", "Empty: the template's box."),
        # Weighting and CTF
        _yn("per_tilt_weighting", "Per-tilt weighting?", "Yes", "Weight each tilt (--per-tilt-weighting): "
            "a fan-shaped missing wedge with the CTF and dose of every tilt, instead of a binary wedge."),
        _yn("use_dose", "Use the accumulated dose?", "Yes", "Dose per tilt from rlnMicrographPreExposure "
            "(--dose-accumulation)."),
        _yn("use_ctf", "Use the CTF?", "Yes", "Defocus per tilt from rlnDefocusU/V (--defocus), with voltage, Cs "
            "and amplitude contrast from the tomogram set."),
        _o("phase_shift", "Phase shift (°):", "", "For a phase plate. Empty or 0: none."),
        _radio("tomogram_ctf_model", "CTF correction of the tomograms:", ("None", "phase-flip"), "None",
               "How the tomograms were CTF-corrected (--tomogram-ctf-model). phase-flip for tomograms reconstructed "
               "from phase-flipped tilts (IMOD strip-based, novaCTF, 3dctf)."),
        _radio("defocus_handedness", "Defocus handedness:", ("0", "1", "-1"), "0",
               "Correct the defocus gradient in each subvolume (--defocus-handedness; only with a volume split). "
               "0: off."),
        _yn("spectral_whitening", "Spectral whitening?", "No", "--spectral-whitening"),
        _o("low_pass", "Low-pass filter (Å):", "", "Applied to tomogram and template (--low-pass). Empty: none."),
        _o("high_pass", "High-pass filter (Å):", "", "Applied to tomogram and template (--high-pass), e.g. 500. "
           "Empty: none."),
        # Picking
        _o("particle_diameter", "Particle diameter (Å):", "", "Sets the angular sampling (Crowther criterion at "
           "2 × voxel, or the low-pass). For an elongated particle, its longest axis. Needed unless an angular "
           "search is given."),
        _o("angular_search", "Angular search:", "", "Overrides the sampling from the diameter: an increment in "
           "degrees, or a .txt file of ZXZ Euler angles in radians (one triple per line)."),
        _o("z_symmetry", "Z-axis rotational symmetry:", "1", "Cn symmetry around the template's z axis: the "
           "search over the last angle is n times shorter."),
        _yn("non_spherical", "Non-spherical mask?", "No", "Needed when the mask is not a sphere (about twice "
            "as slow)."),
        _yn("random_phase", "Random-phase correction?", "No", "Subtract the scores of a phase-randomised "
            "template (STOPGAP's method, -r)."),
        _o("rng_seed", "Random seed:", "", "For the random-phase template. Empty: random."),
        _yn("half_precision", "Half-precision scores?", "No", "Write the score maps as float16."),
        _o("volume_split", "Volume split (x y z):", "1 1 1", "Split each tomogram into subvolumes to fit the GPU "
           "memory (-s)."),
        _o("search_x", "Search range x (px):", "", "Start and end voxel in x, e.g. 0 200. Empty: all."),
        _o("search_y", "Search range y (px):", "", "Start and end voxel in y, e.g. 0 200. Empty: all."),
        _o("search_z", "Search range z (px):", "", "Start and end voxel in z, e.g. 40 160 to leave out the volume above and "
           "below the lamella. Empty: all."),
        _o("tomogram_mask", "Tomogram masks folder:", "", "A folder with a mask per tomogram, of the tomogram's "
           "size, named <rlnTomoName>_mask.mrc: the masks/ folder of a Tomogram masks job (Additional external "
           "jobs), which draws them. Subvolumes without voxels above 0 are not searched; a tomogram without a mask "
           "in the folder stops the job. Also accepted: a path with {name} for the rlnTomoName (e.g. "
           "masks/{name}.mrc), or one mask file for every tomogram. Empty: none.", kind="folder"),
        _o("match_args", "More pytom_match_template.py arguments:", "", "Added to its command line."),
        # Compute
        _o("gpu_ids", "GPUs to use:", "0", "GPU ids separated by colons, as in RELION, e.g. 0:1:2:3. Empty: every GPU the job "
           "sees (CUDA_VISIBLE_DEVICES, nvidia-smi). On a cluster, ids count from 0 among the GPUs the queue "
           "gave the job; ask for them in the submission script (e.g. an extra variable with --gres=gpu:4)."),
        _radio("gpu_mode", "With several GPUs:", (R.GPU_PER_TOMOGRAM, R.GPU_SPLIT), R.GPU_PER_TOMOGRAM,
               "One tomogram per GPU: as many tomograms at a time as GPUs (best for many tomograms; no MPI is "
               "needed, each pytom process gets one GPU). Each tomogram over all GPUs: pytom splits every "
               "search over the GPUs (-g with all of them), one tomogram at a time."),
        _yn("skip_done", "Skip tomograms already done?", "Yes", "When the job runs again (continue), tomograms "
            "with score maps are not searched again: after a failure only the rest run."),
    ] + ENV_OPTIONS
    tabs = [
        Tab("Inputs and template", (
            Row(("in_tomograms",)), Row(("tomo_kind",)), Row(("tomo_names",)), Row(("tilt_source",)),
            Row(("voxel_size",)), Row(("fn_template",)), Row(("do_prep_template",)),
            Row(("template_angpix_in",), group=1), Row(("template_angpix",), group=1),
            Row(("template_box",), group=1), Row(("template_invert",), group=1), Row(("template_mirror",), group=1),
            Row(("template_center",), group=1), Row(("template_lowpass",), group=1),
            Row(("mask_source",)), Row(("fn_mask",), group=2), Row(("do_prep_mask",), group=2),
            Row(("mask_angpix_in",), group=3), Row(("mask_angpix",), group=3), Row(("mask_box",), group=3),
            Row(("mask_mirror",), group=3),
            Row(("mask_radius",), group=4), Row(("mask_minor",), group=4), Row(("mask_sigma",), group=4),
            Row(("mask_box_create",), group=4))),
        Tab("Weighting and CTF", (
            Row(("per_tilt_weighting",)), Row(("use_dose",)), Row(("use_ctf",)), Row(("phase_shift",), group=5),
            Row(("tomogram_ctf_model",)), Row(("defocus_handedness",)), Row(("spectral_whitening",)),
            Row(("low_pass",)), Row(("high_pass",)))),
        Tab("Picking", tuple(Row((k,)) for k in (
            "particle_diameter", "angular_search", "z_symmetry", "non_spherical", "random_phase", "rng_seed", "half_precision", "volume_split", "search_x",
            "search_y", "search_z", "tomogram_mask", "match_args"))),
        Tab("Compute", tuple(Row((k,)) for k in ("gpu_ids", "gpu_mode", "skip_done", "cuda_module", "pytom_env", "conda",
                                                   "log_level"))),
    ]
    groups = {1: Group(1, None, "do_prep_template"), 2: Group(2, None, "mask_source", R.MASK_FILE),
              3: Group(3, 2, "do_prep_mask"), 4: Group(4, None, "mask_source", R.MASK_CREATE),
              5: Group(5, None, "use_ctf")}
    return opts, tabs, groups


def extract_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        _o("in_match", "Template matching job:", "", "The pytom_match.star of a pytom template matching job "
           "(it may still be running: the tomograms it finished are used).", kind="inputnode",
           node_type="ProcessData.star", pattern="pytom match (*pytom_match.star)"),
        TOMO_NAMES,
        _o("n_particles", "Max. particles per tomogram:", "1000", "-n: at most this many peaks per tomogram."),
        _o("tomo_n_particles", "Max. particles of some tomograms:", "", "Another maximum for some tomograms, "
           "as <rlnTomoName>=<number> separated by spaces (e.g. TS_01=3000); the Thresholds view sets it. When "
           "the job is continued, only the tomograms whose settings changed are extracted again."),
        _o("particle_diameter", "Particle diameter (Å):", "", "Peaks closer than half of it to a higher one are "
           "left out. Empty: the matching job's."),
        _radio("cut_off_mode", "Cut-off:", (R.CUT_AUTO, R.CUT_FIXED), R.CUT_AUTO,
               "Automatic: pytom's cut-off from the score statistics, for the number of false positives below. "
               "Fixed: a score (LCC) cut-off."),
        _o("n_false_positives", "False positives per tomogram:", "1", "For the automatic cut-off (values below 1 "
           "make it stricter)."),
        _o("cut_off", "Fixed cut-off (LCC):", "", "Score cut-off for every tomogram."),
        _yn("do_candidates", "Keep candidates below the cut-off?", "Yes",
            "Also extract the peaks a bit below the cut-off, to choose the thresholds interactively in the "
            "Thresholds view of the job (each tomogram, or all at once; particles.star is rewritten there). "
            "Starts with the cut-off above."),
        _o("candidate_factor", "Candidates down to (× cut-off):", "0.7", "The lowest candidate score, as a "
           "fraction of each tomogram's cut-off."),
        _yn("tophat", "Tophat filter?", "No", "Keep only sharp peaks (--tophat-filter)."),
        _o("tophat_connectivity", "Tophat connectivity:", "1", "1–3: the structuring element's connectivity."),
        _o("tophat_bins", "Tophat histogram bins:", "50", "--tophat-bins"),
        _o("plot_bins", "Plot histogram bins:", "20", "--plot-bins, for pytom's extraction plot."),
        _radio("tomogram_mask_mode", "Tomogram mask:", (R.TMASK_JOB, R.TMASK_OTHER, R.TMASK_IGNORE), R.TMASK_JOB,
               "Which tomogram mask limits the peaks: the one used for matching (if any), other masks, or none "
               "(--ignore_tomogram_mask)."),
        _o("tomogram_mask", "Tomogram masks folder:", "", "A folder with a mask per tomogram, named "
           "<rlnTomoName>_mask.mrc (the masks/ folder of a Tomogram masks job); peaks where the mask is not above 0 "
           "are left out. Also accepted: a path with {name} for the rlnTomoName, or one mask file.", kind="folder"),
        _yn("do_roc", "Estimate the ROC?", "No", "Also run pytom_estimate_roc.py (a Gaussian fit of the score "
            "histogram, as a plot)."),
        _o("roc_particles", "ROC: number of peaks:", "", "Empty: 3 × the maximum number of particles."),
        _o("roc_bins", "ROC: histogram bins:", "20", "--bins"),
        _yn("roc_crop", "ROC: crop the plot?", "No", "--crop-plot"),
        _radio("orientations", "Write the angles as:", tuple(R.ORIENTATIONS), "rlnAngleRot/Tilt/Psi",
               "The orientations pytom found, in RELION's convention: as rlnAngleRot/Tilt/Psi (as pytom's "
               "--relion5-compat writes them), as the subtomogram orientation (rlnTomoSubtomogramRot/Tilt/Psi), "
               "or not at all."),
        _o("extract_args", "More pytom_extract_candidates.py arguments:", "", "Added to its command line."),
    ] + ENV_OPTIONS
    tabs = [
        Tab("Input", (Row(("in_match",)), Row(("tomo_names",)))),
        Tab("Extraction", (
            Row(("n_particles",)), Row(("tomo_n_particles",)), Row(("particle_diameter",)), Row(("cut_off_mode",)),
            Row(("n_false_positives",), group=1), Row(("cut_off",), group=2), Row(("do_candidates",)),
            Row(("candidate_factor",), group=3), Row(("tophat",)), Row(("tophat_connectivity",), group=4),
            Row(("tophat_bins",), group=4), Row(("plot_bins",)), Row(("tomogram_mask_mode",)),
            Row(("tomogram_mask",), group=5), Row(("do_roc",)), Row(("roc_particles",), group=6),
            Row(("roc_bins",), group=6), Row(("roc_crop",), group=6))),
        Tab("Output", (Row(("orientations",)), Row(("extract_args",)))),
        Tab("Compute", tuple(Row((k,)) for k in ("cuda_module", "pytom_env", "conda", "log_level"))),
    ]
    groups = {1: Group(1, None, "cut_off_mode", R.CUT_AUTO), 2: Group(2, None, "cut_off_mode", R.CUT_FIXED),
              3: Group(3, None, "do_candidates"), 4: Group(4, None, "tophat"),
              5: Group(5, None, "tomogram_mask_mode", R.TMASK_OTHER), 6: Group(6, None, "do_roc")}
    return opts, tabs, groups


# ---------------------------------------------------------------------------- the form

def _word(value: str) -> str:
    """A shell word without double quotes: job.star keeps "Additional arguments" in double quotes, and
    RELION's STAR reader ends a quoted value at the next one. A value with a single quote in it gets
    backslashes instead of quotes."""
    if "'" not in value:
        return shlex.quote(value)
    return "".join(c if c.isalnum() or c in "@%+=:,./-_" else "\\" + c for c in value)


def encode(values: dict[str, str], keys: list[str]) -> str:
    """The options as "Additional arguments": --key value, shell-quoted."""
    return " ".join(f"--{k} {_word(values.get(k, ''))}" for k in keys)


def quoted_values(values: dict[str, str], keys) -> list[str]:
    """Options whose value has a double quote, which cannot go into job.star's "Additional arguments"."""
    return [k for k in keys if '"' in values.get(k, "")]


def decode(text: str) -> tuple[dict[str, str], list[str]]:
    """--key value pairs from "Additional arguments"; also returns what was not such a pair."""
    try:
        words = shlex.split(text)
    except ValueError:
        words = text.split()
    out, other = {}, []
    i = 0
    while i < len(words):
        w = words[i]
        if w.startswith("--") and i + 1 < len(words):
            out[w[2:]] = words[i + 1]
            i += 2
        else:
            other.append(w)
            i += 1
    return out, other


@dataclass
class PytomVariant(JobVariant):
    kind: str = MATCH
    program: str = ""
    description: str = ""
    own_keys: tuple[str, ...] = ()
    external: JobVariant | None = None
    env: RelionEnv | None = None
    fn_exe: str = ""  # External's executable; empty: pytom_runner.py's (runner_command)

    @property
    def is_cli(self) -> bool:  # the editor: a form of its own, written to an External job.star
        return True

    def resolved_options(self, env: RelionEnv) -> dict[str, OptionDef]:
        return self.options

    def resolved_tabs(self, env: RelionEnv) -> list[Tab]:
        return list(self.tabs)

    def default_values(self, env: RelionEnv) -> dict[str, str]:
        return {k: o.default for k, o in self.options.items()}

    def new_jobstar(self, env: RelionEnv, is_continue: bool = False) -> JobStar:
        return self.make_jobstar(self.default_values(env), is_continue, "External/job001/", None)

    def make_jobstar(self, values: dict[str, str], is_continue: bool, job_dir: str,
                     pipeline: Pipeline | None) -> JobStar:
        assert self.external is not None and self.env is not None  # given by the catalog that makes these
        js = self.external.new_jobstar(self.env, is_continue)
        js.is_tomo = True
        o = js.options
        o["fn_exe"] = self.fn_exe or runner_command(self.kind)
        o["nr_threads"] = values.get("nr_threads", "1") or "1"
        for k in RUNNING_KEYS:
            if k in values:
                o[k] = values[k]
        for k in self.options:
            if k.startswith("qsub_extra"):
                o[k] = values.get(k, "")
        o["other_args"] = encode(values, list(self.own_keys))
        if pipeline is not None:
            self._fill_inputs(o, values, pipeline)
        return js

    def _fill_inputs(self, o: dict[str, str], values: dict[str, str], pipeline: Pipeline) -> None:
        """The job's input files that are pipeline nodes go in External's input fields, so RELION links them."""
        files = []
        for k in self.own_keys:
            if self.options[k].kind != "inputnode":
                continue
            v = values.get(k, "").strip()
            if v and v in pipeline.nodes and pipeline.nodes[v].producer and v not in files:
                if k == "fn_mask" and values.get("mask_source") != R.MASK_FILE:
                    continue
                files.append(v)
        masks = self._masks_node(values, pipeline)
        if masks and masks not in files:
            files.append(masks)
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
        for f in left:
            if free:
                o[free.pop(0)] = f

    def _masks_node(self, values: dict[str, str], pipeline: Pipeline) -> str | None:
        """The output of the Tomogram masks job whose folder is the tomogram masks (so the job hangs under it)."""
        if "tomogram_mask" not in self.own_keys:
            return None
        if self.kind == EXTRACT and values.get("tomogram_mask_mode") != R.TMASK_OTHER:
            return None
        from cryo_savanna.jobdefs.tomomask_jobs import masks_job_of
        return masks_job_of(values.get("tomogram_mask", ""), pipeline)

    def values_from_jobstar(self, js: JobStar, env: RelionEnv) -> tuple[dict[str, str], list[str]]:
        values = self.default_values(env)
        o = js.options
        for k in ("nr_threads",) + RUNNING_KEYS:
            if k in o and k in values:
                values[k] = o[k]
        for k in values:
            if k.startswith("qsub_extra") and k in o:
                values[k] = o[k]
        own, _ = decode(o.get("other_args", ""))
        ignored = []
        for k, v in own.items():
            if k in self.own_keys:
                values[k] = v
            else:
                ignored.append(k)
        return values, ignored


def runner_command(kind: str) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def pytom_kind(fn_exe: str) -> str | None:
    """match / extract for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in (MATCH, EXTRACT) and \
            any(Path(p.strip("'\"")).name == "pytom_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def pytom_kind_of(js: JobStar) -> str | None:
    return pytom_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


MAIN_INPUTS = {MATCH: "in_tomograms", EXTRACT: "in_match"}  # the form's field with the data the job works on


def pytom_main_input(js: JobStar) -> str | None:
    """The file in a pytom job's main input field (its tomograms / its matching job), else None (also for
    jobs that are not pytom's). It places the job in the job tree, whichever External field it went in."""
    kind = pytom_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    value = own.get(MAIN_INPUTS[kind], "").strip()
    while value.startswith("./"):
        value = value[2:]
    return value or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's pytom_runner.py (and the runner_common.py it imports) into the project
    (RUNNER_IN_PROJECT), each replaced in one step."""
    return runner_files.install_runner(project_dir, RUNNER, RUNNER_IN_PROJECT)


def add_running_tab(options: dict[str, OptionDef], tabs: list[Tab], groups: dict[int, Group],
                    ext: dict[str, OptionDef]) -> tuple[list[Tab], dict[int, Group]]:
    """External's "Running" tab (threads, queue, RELION_QSUB_EXTRA fields) after a form's own tabs;
    ``options`` gets its fields (nr_threads: the caller's, else External's)."""
    options.setdefault("nr_threads", ext["nr_threads"])
    running = [Row(("nr_threads",))]
    for k in RUNNING_KEYS:
        options[k] = ext[k]
        running.append(Row((k,), group=None if k == "do_queue" else QUEUE_GROUP))
        if k == "qsub":
            for ex in [x for x in ext if x.startswith("qsub_extra")]:
                options[ex] = ext[ex]
                running.append(Row((ex,), group=QUEUE_GROUP))
    groups = dict(groups)
    groups[QUEUE_GROUP] = Group(QUEUE_GROUP, None, "do_queue", "Yes")
    return list(tabs) + [Tab("Running", tuple(running))], groups


class PytomCatalog:
    """The two forms, built once per RELION environment (its queue settings and RELION_QSUB_EXTRA fields)."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[str, PytomVariant] = {}

    def variant(self, kind: str) -> PytomVariant:
        if kind not in self._cache:
            self._cache[kind] = self._build(kind)
        return self._cache[kind]

    def _build(self, kind: str) -> PytomVariant:
        external = self.defs.variant(EXTERNAL, True)
        ext = external.resolved_options(self.env)
        own, tabs, groups = match_options() if kind == MATCH else extract_options()
        options = {o.key: o for o in own}
        options["pytom_env"] = replace(options["pytom_env"], default=default_env())
        options["cuda_module"] = replace(options["cuda_module"], default=_warmed_cuda_module())
        if kind == EXTRACT:
            options["nr_threads"] = replace(ext["nr_threads"], label="Tomograms at a time:", help=(
                "Extractions run in parallel, each one pytom process. Also the number of threads a queue "
                "script reserves (XXXthreadsXXX). pytom's programs look for a GPU when they start, so the "
                "job needs a machine with a GPU and its NVIDIA driver even to extract: submit it to a GPU "
                "queue (ask for one GPU in the submission script, e.g. --gres=gpu:1) or run it on a GPU "
                "workstation; on a machine without one, every tomogram fails with "
                "'cudaErrorInsufficientDriver'."))
        else:
            options["nr_threads"] = replace(ext["nr_threads"], help=(
                "CPU threads to reserve on the queue (XXXthreadsXXX in the submission script). pytom's matching "
                "runs on the GPUs."))
        tabs, groups = add_running_tab(options, tabs, groups, ext)
        return PytomVariant(key=f"pytom:{kind}", label=EXTERNAL, proc_id=external.proc_id,
                            dirname=external.dirname, is_tomo=True, options=options, tabs=tabs, groups=groups,
                            kind=kind, program=TITLES[kind], description=DESCRIPTIONS[kind],
                            own_keys=tuple(o.key for o in own), external=external, env=self.env)


def last_settings(project_dir: Path, pipeline: Pipeline | None, kind: str,
                  jobs: list[str] | None = None) -> dict[str, str]:
    """CUDA module, pytom environment and conda command of the newest pytom job in the project (any kind),
    so they need typing only once. ``jobs``: the project's pytom jobs, when the caller knows them (the
    window does, from its snapshot): only those are read, not every External job's job.star."""
    if pipeline is None:
        return {}
    for name, pr in reversed(list(pipeline.processes.items())):
        if pr.type_label != EXTERNAL or (jobs is not None and name not in jobs):
            continue
        try:
            js = JobStar.read(Path(project_dir) / name / "job.star")
        except (OSError, ValueError, KeyError):
            continue
        if pytom_kind_of(js) is None:
            continue
        own, _ = decode(js.options.get("other_args", ""))
        return {k: own[k] for k in ("cuda_module", "pytom_env", "conda") if k in own}
    return {}

