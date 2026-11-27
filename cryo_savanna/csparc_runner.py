#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its "CryoSPARC particles to RELION"
# jobs, and travels with it (with runner_common.py, whose helpers it uses).
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""CryoSPARC particles to RELION with pyem's csparc2star.py, as a RELION External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/csparc_runner.py csparc2star

(a copy of this file in the project, beside a copy of runner_common.py). RELION runs it as
``... --o <job dir> [--in_* F]... --j <threads> --in_cs <particles.cs> --<option> <value> ...``. It needs
nothing but the standard library (Python 3.6+); pyem runs in its own conda environment (``conda run -n
<pyem_env>``): first ``python -c HELPER`` (numpy) for what the .cs files say about their micrographs and
poses, then ``csparc2star.py <particles.cs> [<passthrough.cs>...] csparc2star.star <flags>``.

csparc2star.py writes CryoSPARC's paths as they are in the .cs files (relative to the CryoSPARC project:
``J12/extract/<UID>_<name>_particles.mrc``, a Downsample's or Restack's stacks, whatever job wrote them). The
job then makes ``particles.star`` from that, in one of two modes:

- "Use the stacks directly": each stack is linked into the job as ``Stacks/<its CryoSPARC path>.mrcs`` (a
  symbolic link to the CryoSPARC file, absolute: nothing of CryoSPARC's is copied or changed), and
  rlnImageName points at the link. The stacks are found from the paths in the .cs files: in the CryoSPARC
  project folder given (``cs_project``), else in the folders above the particle .cs file (the CryoSPARC
  project's own, or an export's).
- "Re-extract in RELION": rlnMicrographName becomes the RELION micrograph (``in_micrographs``, e.g.
  CtfFind/job003/micrographs_ctf.star) of each particle, the coordinates are in that micrograph's pixels
  (scaled by its size over CryoSPARC's: binning, a Fourier crop or a downsampled micrograph), and the optics
  table is the micrographs' one (same groups, in the same order, as Extract needs) with the particles' image
  pixel size and box, for RELION's Extract with "Re-extract refined particles?".

Micrographs are matched by name: the CryoSPARC name without its UIDs (21 digits and an underscore), its
extension and the suffixes of CryoSPARC's motion correction (``_patch_aligned_doseweighted``, ...), equal to
the RELION one's, else the longest RELION name contained in it (or containing it).

The Y axis (``flip_y``), as pyem's wiki says (asarnow/pyem#133): csparc2star.py's default coordinates are
RELION's, for micrographs the same way up as CryoSPARC's (the re-extracted images are then CryoSPARC's own:
poses and CTF stay); ``--inverty`` keeps CryoSPARC's values (to import the particles into CryoSPARC again);
for micrographs upside-down against CryoSPARC's (CryoSPARC's Patch Motion against RelionCorr or MotionCor3),
``--inverty`` and, for re-extraction, ``--flipy`` (here ``--flipy-pose`` and ``--flipy-ctf``, as the .cs has
poses and CTF; 2D poses are mirrored here). Used directly, the stacks are CryoSPARC's: their poses stay.
"""

import json
import os
import re
import sys

if __package__:  # imported from cryo-savanna
    from cryo_savanna import runner_common as C
else:  # run in a project: the copy of runner_common.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import runner_common as C  # type: ignore[no-redef]

Usage = C.Usage

CSPARC2STAR = "csparc2star"
MODES = (CSPARC2STAR,)
DIRECT, REEXTRACT = "Use the stacks directly", "Re-extract in RELION"
FLIP_DEFAULT = "pyem's default (for RELION)"
FLIP_CRYOSPARC = "--inverty (for CryoSPARC)"
FLIP_UPSIDE_DOWN = "Upside-down micrographs (--inverty --flipy)"
FLIP_CHOICES = (FLIP_DEFAULT, FLIP_CRYOSPARC, FLIP_UPSIDE_DOWN)
PARTICLES = "particles.star"
PYEM_STAR = "csparc2star.star"  # csparc2star.py's own output (CryoSPARC's paths)
INFO = "cs_info.json"  # what HELPER read from the .cs files
LOG = "csparc2star.log"
STACKS = "Stacks"
NODE_PARTICLES = "ParticleGroupMetadata.star.relion"

WHO = "csparc2star"  # say(): " ++++ csparc2star: ..."
say = C.say

UID = re.compile(r"^(?:\d{21}_)+")  # CryoSPARC's UIDs at the start of a file name (pyem: [0-9]{21}_)
# what CryoSPARC's motion correction adds to the movie's name, longest first
MIC_SUFFIXES = ("_patch_aligned_doseweighted", "_rigid_aligned_doseweighted", "_aligned_doseweighted",
                "_patch_aligned", "_rigid_aligned", "_doseweighted", "_aligned")
IMAGE_OPTICS = ("rlnImagePixelSize", "rlnImageSize", "rlnImageDimensionality")
CTF_OPTICS = ("rlnBeamTiltX", "rlnBeamTiltY", "rlnOddZernike", "rlnEvenZernike", "rlnMagMat00", "rlnMagMat01",
              "rlnMagMat10", "rlnMagMat11")
CTF_PARTICLE = ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift", "rlnCtfBfactor",
                "rlnCtfScalefactor", "rlnCtfMaxResolution", "rlnCtfFigureOfMerit")
IMAGE, MIC, OPTICS = "rlnImageName", "rlnMicrographName", "rlnOpticsGroup"
X, Y = "rlnCoordinateX", "rlnCoordinateY"

# run in pyem's environment (numpy): per .cs file its fields and rows, per micrograph its shape and pixel size
HELPER = r"""
import json, sys
import numpy as np
out, allow_pickle, max_header, files = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), sys.argv[4:]

def load(f):
    try:
        return np.load(f, allow_pickle=allow_pickle, max_header_size=max_header)
    except TypeError:  # numpy before 1.24
        return np.load(f, allow_pickle=allow_pickle)

def text(v):
    v = v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return v[1:] if v.startswith(">") else v

info = {"files": [], "micrographs": {}}
for f in files:
    cs = load(f)
    names = list(cs.dtype.names or ())
    info["files"].append({"path": f, "rows": int(cs.shape[0]), "fields": names})
    if "location/micrograph_path" not in names:
        continue
    paths = cs["location/micrograph_path"]
    _, first = np.unique(paths, return_index=True)
    for i in first:
        mic = info["micrographs"].setdefault(text(paths[i]), {})
        if "location/micrograph_shape" in names and "shape" not in mic:
            mic["shape"] = [int(v) for v in cs["location/micrograph_shape"][i]]
        if "location/micrograph_psize_A" in names and "psize" not in mic:
            mic["psize"] = float(cs["location/micrograph_psize_A"][i])
with open(out, "w") as fh:
    json.dump(info, fh)
"""


# ---------------------------------------------------------------------------- the .cs files

def _rel(path):
    p = C._rel(path)
    while p.startswith("./"):
        p = p[2:]
    return p


def _bare(path):
    """A path of a .cs file without the '>' of CryoSPARC's exports."""
    return path[1:] if path.startswith(">") else path


def passthrough_for(cs):
    """The passthrough file CryoSPARC writes beside a job's output: J45_passthrough_particles.cs for
    J45_particles.cs or J45_008_particles.cs (an iteration's), J31_passthrough_particles_selected.cs for
    J31_particles_selected.cs. None when there is none."""
    folder, name = os.path.split(cs)
    m = re.match(r"^(J\d+)_(?:\d+_)?(.+)\.cs$", name)
    if not m or m.group(2).startswith("passthrough_"):
        return None
    path = os.path.join(folder, "%s_passthrough_%s.cs" % (m.group(1), m.group(2)))
    return path if os.path.isfile(path) else None


def cs_files(opts):
    """(the particle .cs, [its passthrough .cs files]), checked."""
    cs = C.text(opts, "in_cs")
    if not cs:
        raise Usage("no particle .cs file given (--in_cs)")
    if not os.path.isfile(cs):
        raise Usage("the particle .cs file is not there: %s" % cs)
    passthroughs = C.words(opts, "passthrough_cs")
    for p in passthroughs:
        if not os.path.isfile(p):
            raise Usage("passthrough .cs file not there: %s" % p)
    if not passthroughs and C.yes(opts, "auto_passthrough"):
        found = passthrough_for(cs)
        if found:
            say("passthrough file of the job: %s" % found)
            passthroughs = [found]
    return cs, passthroughs


def has_3d_poses(fields):
    """The particle .cs carries 3D poses, as csparc2star.py reads them (a refinement, a 3D classification)."""
    return ("alignments3D/class_posterior" in fields or "alignments3D_multi/class_posterior" in fields
            or len([n for n in fields if "class_posterior" in n]) > 1)


def run_helper(sup, opts, conda, job_dir, files):
    """What HELPER reads from the .cs files: {"files": [...], "micrographs": {path: {shape, psize}}}."""
    path = os.path.join(job_dir, INFO)
    max_header = C.number(opts, "max_header_size", int) or 100000
    cmd = conda + ["python", "-c", HELPER, path, "1" if C.yes(opts, "allow_pickle") else "0", str(max_header)]
    rc = sup.run(cmd + files, os.path.join(job_dir, LOG))
    if rc != 0:
        C._show_tail(os.path.join(job_dir, LOG))
        raise Usage("cannot read the .cs files with numpy in pyem's environment (exit code %d): see %s"
                    % (rc, os.path.join(job_dir, LOG)))
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------- csparc2star.py

def pyem_flags(opts, invert_y, flip_pose, flip_ctf):
    """csparc2star.py's flags from the form (not the files)."""
    flags = []
    v = C.number(opts, "boxsize", float, what="Refinement box size")
    if v is not None:
        flags += ["--boxsize", C.num(v)]
    for c in C.words(opts, "classes"):
        try:
            flags += ["--class", str(int(c))]
        except ValueError:
            raise Usage("classes: whole numbers separated by spaces (got %r)" % c)
    v = C.number(opts, "minphic", float, what="Minimum class posterior")
    if v:
        flags += ["--minphic", C.num(v)]
    if C.yes(opts, "noswapxy"):
        flags.append("--noswapxy")
    if C.yes(opts, "invertx"):
        flags.append("--invertx")
    if not invert_y:
        flags.append("--inverty")  # (csparc2star.py's --inverty switches its flip of Y off)
    if flip_pose:
        flags.append("--flipy-pose")
    if flip_ctf:
        flags.append("--flipy-ctf")
    for key, flag in (("transform", "--transform"), ("copy_coords", "--copy-micrograph-coordinates"),
                      ("micrograph_path", "--micrograph-path")):
        if C.text(opts, key):
            flags += [flag, C.text(opts, key)]
    for key, flag in (("cached", "--cached"), ("first10k", "--10k"), ("allow_pickle", "--allow-pickle")):
        if C.yes(opts, key):
            flags.append(flag)
    v = C.number(opts, "max_header_size", int, what="Largest .cs header")
    if v:
        flags += ["--max-header-size", str(v)]
    return flags + ["--loglevel", "INFO"] + C.words(opts, "more_args")


def read_pyem(path):
    """(optics rows, particle columns, particle rows as dicts) of csparc2star.py's STAR file."""
    blocks = C.read_star(path)
    parts = blocks.get("particles")
    if parts is None or not parts.is_loop:
        raise Usage("%s: no data_particles table (did csparc2star.py write a RELION 3.1+ file?)" % path)
    optics = blocks.get("optics")
    return (optics.dicts() if optics is not None else []), list(parts.columns), parts.dicts()


# ---------------------------------------------------------------------------- the stacks

class Stacks(object):
    """Finds CryoSPARC's files from their paths in the .cs files: relative to the CryoSPARC project folder
    given, else to one of the folders above the particle .cs file (the project's, an export's)."""

    def __init__(self, cs_file, project=""):
        self.bases = []
        if project:
            self.bases.append(os.path.abspath(project))
        d = os.path.dirname(os.path.abspath(cs_file))
        while True:
            if d not in self.bases:
                self.bases.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        self.found = {}  # path -> file (None: not found)

    def find(self, path):
        if path in self.found:
            return self.found[path]
        p = _bare(path)
        out = None
        if os.path.isabs(p):
            out = p if os.path.isfile(p) else None
        else:
            for base in self.bases:
                f = os.path.join(base, p)
                if os.path.isfile(f):
                    out = f
                    if base != self.bases[0]:  # the next ones most likely there too
                        self.bases.remove(base)
                        self.bases.insert(0, base)
                    break
        self.found[path] = out
        return out

    def project(self):
        """The folder the paths were found in (the CryoSPARC project's, as far as can be told)."""
        return self.bases[0]


def split_image(name):
    if "@" not in name:
        raise Usage("rlnImageName without an index: %r" % name)
    idx, path = name.split("@", 1)
    return int(idx), path


def link_name(path, found, strip_uid):
    """Where a stack's link goes in the job: Stacks/<its CryoSPARC path>.mrcs (outside the project: its
    whole path)."""
    p = _bare(path)
    rel = os.path.normpath(p)
    if os.path.isabs(rel) or rel.startswith(".."):
        rel = os.path.abspath(found).lstrip("/")
    folder, base = os.path.split(rel)
    if strip_uid:
        base = UID.sub("", base) or base
    stem = base[:-5] if base.endswith(".mrcs") else os.path.splitext(base)[0]
    return os.path.join(STACKS, folder, stem + ".mrcs")


def link(target, at):
    """A symbolic link to the file (absolute: the CryoSPARC project is elsewhere); the file is not touched."""
    target = os.path.abspath(target)
    if os.path.islink(at):
        if os.readlink(at) == target:
            return
        os.remove(at)
    elif os.path.lexists(at):
        raise Usage("%s is in the way of a link to %s (and is not a link itself)" % (at, target))
    folder = os.path.dirname(at)
    if folder:
        os.makedirs(folder, exist_ok=True)
    os.symlink(target, at)


def link_stacks(rows, job_dir, stacks, strip_uid):
    """Links every stack the particles name into the job and points rlnImageName at the links. Returns the
    number of stacks."""
    paths, missing = {}, []
    for r in rows:
        idx, path = split_image(r[IMAGE])
        if path not in paths:
            found = stacks.find(path)
            if found is None:
                missing.append(path)
            paths[path] = [found, idx]
        else:
            paths[path][1] = max(paths[path][1], idx)
    if missing:
        raise Usage("%d of %d particle stacks not found (from %s; give the CryoSPARC project folder): %s"
                    % (len(missing), len(paths), " or ".join(stacks.bases[:3]), " ".join(missing[:5])))
    names, taken = {}, {}  # type: dict[str, str], dict[str, str]
    for path, (found, _) in paths.items():
        name = link_name(path, found, strip_uid)
        if strip_uid and name in taken and taken[name] != found:  # the UID told them apart
            name = link_name(path, found, False)
        taken[name] = found
        names[path] = name
    for path, (found, most) in paths.items():
        try:
            nz = C.mrc_header(found)[2]
        except (OSError, Usage) as e:
            raise Usage("cannot read the stack %s: %s" % (found, e))
        if most > nz:
            raise Usage("%s has %d images, but the particles name image %d of it" % (found, nz, most))
        link(found, os.path.join(job_dir, names[path]))
    for r in rows:
        idx, path = split_image(r[IMAGE])
        r[IMAGE] = "%06d@%s" % (idx, _rel(os.path.join(job_dir, names[path])))
    return len(paths)


def original_images(rows, stacks):
    """rlnImageName at the CryoSPARC stacks themselves (re-extraction: Extract does not read them)."""
    for r in rows:
        idx, path = split_image(r[IMAGE])
        found = stacks.find(path)
        if found is not None:
            r[IMAGE] = "%06d@%s" % (idx, os.path.abspath(found))


# ---------------------------------------------------------------------------- the micrographs

def mic_key(name):
    """A micrograph's name as CryoSPARC and RELION share it: no folder, extension, UIDs or the suffixes of
    CryoSPARC's motion correction."""
    base = os.path.basename(_bare(name))
    stem = os.path.splitext(base)[0]
    stem = UID.sub("", stem) or stem
    for s in MIC_SUFFIXES:
        if stem.endswith(s) and len(stem) > len(s):
            stem = stem[:-len(s)]
            break
    return stem


class Micrographs(object):
    """The RELION micrographs (a micrographs STAR file): their optics, and the one for a CryoSPARC name."""

    def __init__(self, path):
        blocks = C.read_star(path)
        mics = blocks.get("micrographs")
        if mics is None or not mics.is_loop or MIC not in mics.columns:
            raise Usage("%s: no data_micrographs table with rlnMicrographName (not a RELION micrographs STAR "
                        "file?)" % path)
        optics = blocks.get("optics")
        self.path = path
        self.optics_columns = list(optics.columns) if optics is not None else []
        self.optics = optics.dicts() if optics is not None else []
        self.rows = dict((r[MIC], r) for r in mics.dicts())
        self.by_key = {}  # type: dict[str, list[str]]
        for name in self.rows:
            self.by_key.setdefault(mic_key(name), []).append(name)
        self.matched = {}
        self.sizes = {}

    def match(self, cs_name):
        """The RELION micrograph for a CryoSPARC one, else None (also when two would do)."""
        if cs_name in self.matched:
            return self.matched[cs_name]
        key = mic_key(cs_name)
        names = self.by_key.get(key)
        if names is None:
            best, best_len = [], 0
            for k in self.by_key:
                short, long_ = (k, key) if len(k) <= len(key) else (key, k)
                if len(short) < 6 or 2 * len(short) < len(long_) or short not in long_:
                    continue
                if len(k) > best_len:
                    best, best_len = [k], len(k)
                elif len(k) == best_len:
                    best.append(k)
            names = self.by_key[best[0]] if len(best) == 1 else []
        out = names[0] if len(names) == 1 else None
        self.matched[cs_name] = out
        return out

    def optics_group(self, name):
        return self.rows[name].get(OPTICS, "1")

    def pixel_size(self, name):
        group = self.optics_group(name)
        row = next((o for o in self.optics if o.get(OPTICS) == group), None)
        try:
            return float(row["rlnMicrographPixelSize"]) if row else None
        except (KeyError, ValueError):
            return None

    def size(self, name):
        """(nx, ny) of a RELION micrograph, from its MRC header (None: cannot be read)."""
        if name not in self.sizes:
            try:
                nx, ny = C.mrc_header(name)[:2]
                self.sizes[name] = (nx, ny)
            except (OSError, Usage):
                self.sizes[name] = None
        return self.sizes[name]


def remap(rows, mics, info, swapxy, reextract):
    """rlnMicrographName -> the RELION micrograph, the coordinates in its pixels (scaled by its size over
    CryoSPARC's, else by the pixel sizes). Re-extraction leaves out the particles of micrographs not found,
    and gives each particle its micrograph's optics group. Returns the rows kept."""
    cs_mics = info.get("micrographs", {})
    by_base = dict((os.path.basename(k), v) for k, v in cs_mics.items())
    unmatched, unscaled, out = {}, set(), []  # type: dict[str, int], set[str], list[dict]
    factors = {}
    for r in rows:
        cs_name = r.get(MIC, "")
        name = mics.match(cs_name)
        if name is None:
            unmatched[cs_name] = unmatched.get(cs_name, 0) + 1
            if not reextract:
                out.append(r)
            continue
        if cs_name not in factors:
            factors[cs_name] = _factors(cs_name, name, cs_mics.get(cs_name) or by_base.get(os.path.basename(cs_name))
                                        or {}, mics, swapxy)
        f = factors[cs_name]
        if f is None:
            unscaled.add(cs_name)
            continue
        r[MIC] = name
        for col, fac in ((X, f[0]), (Y, f[1])):
            if col in r:
                r[col] = str(int(round(float(r[col]) * fac)))
        if reextract:
            r[OPTICS] = mics.optics_group(name)
        out.append(r)
    if unscaled:
        raise Usage("the size of %d micrograph(s) is not known (no location/micrograph_shape in the .cs files, "
                    "and the RELION micrograph cannot be read), e.g. %s" % (len(unscaled), sorted(unscaled)[0]))
    if unmatched:
        n = sum(unmatched.values())
        say("WARNING: %d particle(s) of %d CryoSPARC micrograph(s) have no micrograph in %s%s: %s"
            % (n, len(unmatched), mics.path, " and are left out" if reextract else
               " and keep CryoSPARC's micrograph name and coordinates",
               " ".join(sorted(unmatched)[:20]) + (" ..." if len(unmatched) > 20 else "")))
    return out


def _factors(cs_name, name, cs_mic, mics, swapxy):
    """(x, y) factors from CryoSPARC's micrograph pixels to RELION's, else None."""
    shape, size = cs_mic.get("shape"), mics.size(name)
    if shape and size and len(shape) >= 2 and shape[0] and shape[1]:
        ny, nx = shape[0], shape[1]  # CryoSPARC's location/micrograph_shape is (y, x)
        return (size[0] / nx, size[1] / ny) if swapxy else (size[1] / ny, size[0] / nx)
    psize, rln = cs_mic.get("psize"), mics.pixel_size(name)
    if psize and rln:
        return (psize / rln, psize / rln)
    return None


def flip_2d(rows):
    """Mirror 2D poses in Y (csparc2star.py's --flipy-pose needs 3D ones): psi and the Y shift change sign."""
    for r in rows:
        for col in ("rlnAnglePsi", "rlnOriginYAngst", "rlnOriginY"):
            if col in r:
                r[col] = repr(-float(r[col]))


def reextract_optics(rows, pyem_optics, mics, keep_ctf):
    """The micrographs' optics table (every group, in its order: Extract finds a particle's group by its
    number in both), with the particles' image pixel size, box and dimensionality (and with the CTF kept, the
    beam tilt, Zernike and magnification matrix) from csparc2star.py's group of their first particle."""
    by_group = dict((o.get(OPTICS), o) for o in pyem_optics)
    # RELION group -> csparc2star.py's optics of its first particle; -> the groups of its particles there
    first, cs_groups = {}, {}  # type: dict[str, dict], dict[str, set]
    for r in rows:
        g = r.get(OPTICS)
        cs = by_group.get(r.get("_cs_group"), pyem_optics[0] if pyem_optics else {})
        first.setdefault(g, cs)
        cs_groups.setdefault(g, set()).add(r.get("_cs_group"))
    for g, groups in cs_groups.items():
        values = set(tuple(by_group.get(c, {}).get(k) for k in IMAGE_OPTICS) for c in groups)
        if len(values) > 1:
            say("WARNING: the particles of RELION optics group %s have several pixel sizes or boxes in CryoSPARC "
                "(%s): the first is used" % (g, "; ".join(" ".join(str(v) for v in t) for t in sorted(values, key=str))))
    extra = [k for k in IMAGE_OPTICS + (CTF_OPTICS if keep_ctf else ())
             if any(k in o for o in first.values()) and k not in mics.optics_columns]
    fallback = next(iter(first.values()), {})  # type: dict
    out = []
    for o in mics.optics:
        o = dict(o)
        cs = first.get(o.get(OPTICS, ""), fallback)
        for k in extra:
            o[k] = cs.get(k, fallback.get(k, "0"))
        out.append(o)
    columns = list(mics.optics_columns) + extra
    if OPTICS not in columns:
        raise Usage("%s: its optics table has no rlnOpticsGroup" % mics.path)
    return columns, out


def suggested_box(optics):
    """The particle box in the micrographs' pixels, as the CryoSPARC particles' (even): Extract's box size."""
    try:
        o = optics[0]
        box = float(o["rlnImageSize"]) * float(o["rlnImagePixelSize"]) / float(o["rlnMicrographPixelSize"])
    except (IndexError, KeyError, ValueError, ZeroDivisionError):
        return None
    return int(round(box / 2.0)) * 2


# ---------------------------------------------------------------------------- the job

def run(argv):
    C.WHO = WHO
    mode, job_dir, _threads, opts = C.parse(argv, MODES)
    cs, passthroughs = cs_files(opts)
    how = C.text(opts, "mode") or DIRECT
    if how not in (DIRECT, REEXTRACT):
        raise Usage("mode: %r is not one of: %s" % (how, ", ".join((DIRECT, REEXTRACT))))
    reextract = how == REEXTRACT
    mics_star = C.text(opts, "in_micrographs")
    if reextract and not mics_star:
        raise Usage("re-extraction needs the RELION micrographs (--in_micrographs)")
    if mics_star and not os.path.isfile(mics_star):
        raise Usage("the micrographs STAR file is not there: %s" % mics_star)
    if mics_star and C.text(opts, "copy_coords"):
        raise Usage("give either the RELION micrographs or csparc2star.py's --copy-micrograph-coordinates")
    if mics_star and C.text(opts, "micrograph_path"):
        raise Usage("give either the RELION micrographs or csparc2star.py's --micrograph-path")
    project = C.text(opts, "cs_project")
    if project and not os.path.isdir(project):
        raise Usage("the CryoSPARC project folder is not there: %s" % project)
    flip = C.text(opts, "flip_y") or FLIP_DEFAULT
    if flip not in FLIP_CHOICES:
        raise Usage("flip_y: %r is not one of: %s" % (flip, ", ".join(FLIP_CHOICES)))
    keep_ctf = C.yes(opts, "keep_ctf") if "keep_ctf" in opts else True
    mics = Micrographs(mics_star) if mics_star else None
    stacks = Stacks(cs, project)

    sup = C.Supervisor(job_dir, hint=" (is pyem's conda environment on the form right?)")
    conda = C.conda_run(opts, "pyem_env")
    info = run_helper(sup, opts, conda, job_dir, [cs] + passthroughs)
    fields = info["files"][0]["fields"] if info.get("files") else []
    say("%s: %d particles%s" % (cs, info["files"][0]["rows"] if info.get("files") else 0,
                                "; passthrough " + " ".join(passthroughs) if passthroughs else ""))

    invert_y = flip == FLIP_DEFAULT  # csparc2star.py's own flip of Y (--inverty leaves it out)
    mirrored = flip == FLIP_UPSIDE_DOWN and reextract  # the re-extracted images: CryoSPARC's upside-down
    if flip == FLIP_UPSIDE_DOWN:
        say("micrographs upside-down against CryoSPARC's: --inverty%s"
            % (", poses and defocus angle mirrored (--flipy)" if reextract else
               " (the stacks are CryoSPARC's: their poses stay)"))
    poses_3d = has_3d_poses(fields)
    flip_pose = mirrored and reextract and poses_3d
    flip_ctf = mirrored and reextract and keep_ctf and "ctf/df_angle_rad" in set(
        f for entry in info.get("files", []) for f in entry["fields"])

    out = os.path.join(job_dir, PYEM_STAR)
    cmd = conda + ["csparc2star.py", cs] + passthroughs + [out] + pyem_flags(opts, invert_y, flip_pose, flip_ctf)
    rc = sup.run(cmd, os.path.join(job_dir, LOG))
    if rc != 0 or not os.path.isfile(out):
        C._show_tail(os.path.join(job_dir, LOG))
        raise Usage("csparc2star.py failed (exit code %d): see %s" % (rc, os.path.join(job_dir, LOG)))
    pyem_optics, columns, rows = read_pyem(out)
    if not rows:
        raise Usage("csparc2star.py wrote no particles (%s)" % out)
    if IMAGE not in columns:
        raise Usage("%s has no rlnImageName (were the passthrough files given?)" % out)
    n_in = len(rows)
    for r in rows:
        r["_cs_group"] = r.get(OPTICS, "1")

    if mics is not None:
        if MIC not in columns or X not in columns:
            raise Usage("%s has no micrograph names or coordinates (location/ fields: are the passthrough "
                        "files given?)" % out)
        rows = remap(rows, mics, info, not C.yes(opts, "noswapxy"), reextract)
        if not rows:
            raise Usage("no particle is on a micrograph of %s (do the micrograph names match?)" % mics_star)
        say("%d of %d micrographs matched to %s" % (sum(1 for v in mics.matched.values() if v),
                                                   len(mics.matched), mics_star))
    elif C.yes(opts, "strip_uid") and MIC in columns:
        for r in rows:
            folder, base = os.path.split(r[MIC])
            r[MIC] = os.path.join(folder, UID.sub("", base) or base)
    if mirrored and reextract and not poses_3d and "rlnAnglePsi" in columns:
        flip_2d(rows)

    if reextract:
        original_images(rows, stacks)
        optics_columns, optics = reextract_optics(rows, pyem_optics, mics, keep_ctf)
        if OPTICS not in columns:
            columns.append(OPTICS)
        if not keep_ctf:
            columns = [c for c in columns if c not in CTF_PARTICLE]
    else:
        n = link_stacks(rows, job_dir, stacks, C.yes(opts, "strip_uid"))
        say("%d particle stacks linked as .mrcs into %s%s/" % (n, job_dir, STACKS))
        optics = pyem_optics
        optics_columns = list(pyem_optics[0].keys()) if pyem_optics else []
    path = job_dir + PARTICLES
    C.write_star(path, [("optics", None, optics_columns, [[o.get(c, "") for c in optics_columns] for o in optics]),
                        ("particles", None, columns, [[r.get(c, "") for c in columns] for r in rows])])
    say("%d particles%s: %s" % (len(rows), "" if len(rows) == n_in else " (of %d)" % n_in, path))
    if reextract:
        box = suggested_box(optics)
        say("for RELION's Extract: micrograph STAR file %s; 'Re-extract refined particles?' Yes, with %s; "
            "%s%s" % (mics_star, path,
                      "'Re-center refined coordinates?' Yes (CryoSPARC's shifts); " if poses_3d else "",
                      "particle box size %d px (CryoSPARC's box in the micrographs' pixels)" % box if box else
                      "the particle box size in the micrographs' pixels"))
        if not keep_ctf:
            say("CryoSPARC's CTF left out: Extract takes the micrographs' (%s)" % mics_star)
    C.write_output_nodes(job_dir, [(path, NODE_PARTICLES)])
    C._touch(os.path.join(job_dir, C.SUCCESS))
    return 0


if __name__ == "__main__":
    C.main(run)
