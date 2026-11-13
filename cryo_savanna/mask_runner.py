#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its tomogram mask jobs, and travels
# with it.
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Tomogram masks (for pytom's --tomogram-mask) as a RELION External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/mask_runner.py masks

(a copy of this file in the project, beside a copy of pytom_runner.py, whose STAR, MRC and job helpers it
uses). RELION runs it as ``... --o <job dir> [--in_* F]... --j <threads> --<option> <value> ...``. It needs
nothing but the standard library (Python 3.6+: RELION may start it with the system python3).

For each tomogram of a RELION 5 tomograms.star, the job writes masks/<rlnTomoName>_mask.mrc (the name
pytom's wrapper looks for in a masks folder: pytom_runner.TOMO_MASK_SUFFIX), a binary mask (int8, 1 inside,
0 outside) of the tomogram's size and voxel size. A mask is

- a polygon in XY (its corners in voxels, from the tomogram's centre), extruded through Z,
- in Z: either every section ("all": the polygon goes through the whole tomogram, however thick it is),
  or the slab between two planes, bottom and top, each z = centre + z0 + ax·(x − cx) + ay·(y − cy) in
  voxels: tilted planes follow a tilted lamella.

The geometry of every tomogram is kept in masks.json; the job's first run makes a starting geometry (the
whole tomogram, or a central slab, from the form) and keeps what is there when it runs again (new
tomograms get the starting one). cryo-savanna's "Masks" view of the job edits it interactively and
rewrites the masks with save_masks(). Only masks whose geometry changed are written again, and masks that
are the same (the same geometry, or the same voxels) are written once: the others are symbolic links to
that file. tomogram_masks.star lists the masks (rlnTomoName, the file), as the job's output node.
"""

import hashlib
import json
import math
import os
import struct
import sys
import traceback

if __package__:  # imported from cryo-savanna
    from cryo_savanna import pytom_runner as B
else:  # run in a project: the copy of pytom_runner.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pytom_runner as B

Usage = B.Usage
text, number, _rel, _touch = B.text, B.number, B._rel, B._touch

MASKS = "masks"
MODES = (MASKS,)
MASK_DIR = "masks"
GEOMETRY = "masks.json"
LIST = "tomogram_masks.star"
NODE_LIST = "ProcessData.star.cryosavanna.tomogram_masks"
SUFFIX = B.TOMO_MASK_SUFFIX
COL_MASK = "savannaTomogramMask"

Z_ALL, Z_SLAB = "all", "slab"
START_ALL, START_SLAB = "Whole tomogram (all Z)", "Central slab"
VERSION = 1
EPS = 1e-6  # voxel centres on a plane are inside it


def say(text_):
    print(" ++++ tomogram masks: " + text_, flush=True)


def mask_path(job_dir, name):
    return os.path.join(job_dir, MASK_DIR, name + SUFFIX)


# ---------------------------------------------------------------------------- geometry

def default_geometry(shape, z_mode=Z_ALL, thickness=None, margin=0.0):
    """The starting mask of a tomogram of ``shape`` (nx, ny, nz): a rectangle ``margin`` voxels inside
    the edges, through all Z or a central slab ``thickness`` voxels thick (default half the tomogram)."""
    nx, ny, nz = shape
    hx, hy = max(nx / 2.0 - margin, 1.0), max(ny / 2.0 - margin, 1.0)
    t = float(thickness) if thickness else nz / 2.0
    return {"polygon": [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], "z_mode": z_mode,
            "bottom": [-t / 2.0, 0.0, 0.0], "top": [t / 2.0, 0.0, 0.0]}


def clean_geometry(g):
    """A geometry with only what makes the mask (rounded, so that equal masks have equal keys)."""
    poly = [[round(float(x), 3), round(float(y), 3)] for x, y in g.get("polygon", [])]
    mode = Z_SLAB if g.get("z_mode") == Z_SLAB else Z_ALL
    out = {"polygon": poly, "z_mode": mode}
    for k in ("bottom", "top"):
        p = list(g.get(k, [0.0, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        out[k] = [round(float(p[0]), 3), round(float(p[1]), 6), round(float(p[2]), 6)]
    return out


def geometry_key(shape, g):
    """What the mask's voxels depend on: two masks with the same key are the same."""
    g = clean_geometry(g)
    if g["z_mode"] == Z_ALL:
        g = {"polygon": g["polygon"], "z_mode": Z_ALL}  # the planes do not matter
    return json.dumps([list(shape), g], sort_keys=True)


def _polygon_runs(poly, y, cx):
    """x intervals [a, b) inside the polygon (even-odd rule) on the line at height y (absolute voxels)."""
    xs = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            xs.append(cx + x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def _half_line(a, r, cx):
    """The x with a·(x − cx) <= r, as (lo, hi)."""
    if a > 0:
        return -math.inf, cx + r / a
    if a < 0:
        return cx + r / a, math.inf
    return (-math.inf, math.inf) if r >= 0 else (math.inf, -math.inf)


class Raster(object):
    """The voxels of a mask, one XY section at a time (bytes of 0/1, x fastest), with voxel (i, j, k)
    inside when its centre (i + ½, j + ½, k + ½) is."""

    def __init__(self, shape, g):
        self.nx, self.ny, self.nz = shape
        self.g = clean_geometry(g)
        self.cx, self.cy, self.cz = self.nx / 2.0, self.ny / 2.0, self.nz / 2.0
        poly = [(x, y + self.cy) for x, y in self.g["polygon"]]
        self.rows = []  # per y: voxel index runs inside the polygon
        for j in range(self.ny):
            runs = []
            for a, b in _polygon_runs(poly, j + 0.5, self.cx):
                i0, i1 = max(0, int(math.ceil(a - 0.5))), min(self.nx, int(math.ceil(b - 0.5)))
                if i1 > i0:
                    runs.append((i0, i1))
            self.rows.append(runs)
        self.ones = b"\x01" * self.nx
        self._all = None

    def plane(self, k):
        """Section k (z) as bytes, ny rows of nx."""
        nx = self.nx
        if self.g["z_mode"] == Z_ALL and self._all is not None:
            return self._all
        out = bytearray(nx * self.ny)
        z = k + 0.5
        bz0, bax, bay = self.g["bottom"]
        tz0, tax, tay = self.g["top"]
        for j, runs in enumerate(self.rows):
            if not runs:
                continue
            lo, hi = -math.inf, math.inf
            if self.g["z_mode"] == Z_SLAB:
                dy = j + 0.5 - self.cy
                # bottom: cz + bz0 + bax·(x−cx) + bay·dy <= z;  top: z <= cz + tz0 + tax·(x−cx) + tay·dy
                l1, h1 = _half_line(bax, z - self.cz - bz0 - bay * dy, self.cx)
                l2, h2 = _half_line(-tax, self.cz + tz0 + tay * dy - z, self.cx)
                lo, hi = max(l1, l2), min(h1, h2)
                if lo > hi:
                    continue
            off = j * nx
            for i0, i1 in runs:
                if lo != -math.inf:
                    i0 = max(i0, int(math.ceil(lo - 0.5 - EPS)))
                if hi != math.inf:
                    i1 = min(i1, int(math.floor(hi - 0.5 + EPS)) + 1)
                if i1 > i0:
                    out[off + i0:off + i1] = self.ones[:i1 - i0]
        out = bytes(out)
        if self.g["z_mode"] == Z_ALL:
            self._all = out
        return out


# ---------------------------------------------------------------------------- MRC

def mrc_int8_header(shape, voxel, dmean, rms, label="cryo-savanna tomogram mask"):
    nx, ny, nz = shape
    h = struct.pack("<10i", nx, ny, nz, 0, 0, 0, 0, nx, ny, nz)
    h += struct.pack("<6f", nx * voxel, ny * voxel, nz * voxel, 90.0, 90.0, 90.0)
    h += struct.pack("<3i", 1, 2, 3)
    h += struct.pack("<3f", 0.0, 1.0, dmean)
    h += struct.pack("<2i", 1, 0)  # ispg 1 (a volume), no extended header
    h += b"\x00" * 8 + b"MRCO" + struct.pack("<i", 20140) + b"\x00" * 84  # extra: EXTTYP, NVERSION
    h += struct.pack("<3f", 0.0, 0.0, 0.0)
    h += b"MAP " + b"\x44\x44\x00\x00" + struct.pack("<f", rms) + struct.pack("<i", 1)
    h += label.encode("ascii")[:80].ljust(800, b"\x00")
    assert len(h) == 1024
    return h


def write_mask(path, shape, voxel, g):
    """The mask as an int8 MRC file at ``path``; returns the SHA-1 of its voxels."""
    raster = Raster(shape, g)
    sha, ones = hashlib.sha1(), 0
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 1024)
        for k in range(shape[2]):
            p = raster.plane(k)
            sha.update(p)
            ones += p.count(1)
            fh.write(p)
        n = float(shape[0] * shape[1] * shape[2])
        mean = ones / n
        fh.seek(0)
        fh.write(mrc_int8_header(shape, voxel or 1.0, mean, math.sqrt(max(mean - mean * mean, 0.0))))
    return sha.hexdigest()


# ---------------------------------------------------------------------------- masks.json

def read_geometry(job_dir):
    path = os.path.join(job_dir, GEOMETRY)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"version": VERSION, "tomograms": {}, "files": {}}
    data.setdefault("tomograms", {})
    data.setdefault("files", {})
    return data


def write_geometry(job_dir, data):
    path = os.path.join(job_dir, GEOMETRY)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _real(path):
    return os.path.realpath(path)


def save_masks(job_dir, data, progress=None, project_dir=None):
    """Write the masks of ``data`` (as read_geometry gives it, with the tomograms' geometry changed) and
    masks.json. Masks whose geometry did not change are left as they are; a mask the same as another one
    (its geometry, or else its voxels) becomes a symbolic link to that one's file. ``progress(name)`` is
    called before each mask is written. ``project_dir``: the project, when it is not the working directory
    (cryo-savanna's Masks view; the job itself runs in the project): tomogram_masks.star names the masks
    from there. Returns {name: "written" | "linked" | "kept"}."""
    d = os.path.join(job_dir, MASK_DIR)
    if not os.path.isdir(d):
        os.makedirs(d)
    tomos, files = data["tomograms"], data["files"]
    keys = {n: geometry_key(t["shape"], t["geometry"]) for n, t in tomos.items()}

    def intact(n):
        f = files.get(n)
        return f is not None and f.get("key") == keys.get(n) and os.path.isfile(mask_path(job_dir, n))

    unchanged = [n for n in tomos if intact(n)]
    changed = [n for n in tomos if n not in unchanged]
    # a changed mask that unchanged ones link to: its file goes to one of them first (the others follow)
    for n in changed:
        p = mask_path(job_dir, n)
        if not os.path.isfile(p) or os.path.islink(p):
            continue
        linkers = [u for u in unchanged if os.path.islink(mask_path(job_dir, u)) and
                   _real(mask_path(job_dir, u)) == _real(p)]
        if linkers:
            heir = mask_path(job_dir, linkers[0])
            os.replace(p, heir)
            for u in linkers[1:]:
                B._link(heir, mask_path(job_dir, u))
    # real files there now, by key and by content
    by_key, by_sha = {}, {}  # by_key: geometry key -> (file, SHA-1 of its voxels)
    for n in unchanged:
        p = mask_path(job_dir, n)
        if not os.path.islink(p):
            sha = files[n].get("sha1")
            by_key.setdefault(keys[n], (p, sha or _file_sha(files, p, job_dir)))
            by_sha.setdefault((tuple(tomos[n]["shape"]), sha), p)
    done = dict((n, "kept") for n in unchanged)
    for n in changed:
        t = tomos[n]
        p = mask_path(job_dir, n)
        if progress is not None:
            progress(n)
        same = by_key.get(keys[n])
        if same is not None and B._link(same[0], p):
            files[n] = {"key": keys[n], "sha1": same[1]}
            done[n] = "linked"
            continue
        tmp = os.path.join(d, ".%s.tmp" % (n + SUFFIX))
        sha = write_mask(tmp, t["shape"], t.get("voxel") or 1.0, t["geometry"])
        same = by_sha.get((tuple(t["shape"]), sha))
        if same is not None and B._link(same, p):
            os.remove(tmp)
            done[n] = "linked"
        else:
            if os.path.islink(p):
                os.remove(p)
            os.replace(tmp, p)
            by_key.setdefault(keys[n], (p, sha))
            by_sha.setdefault((tuple(t["shape"]), sha), p)
            done[n] = "written"
        files[n] = {"key": keys[n], "sha1": sha}
    write_geometry(job_dir, data)
    write_list(job_dir, data, project_dir)
    return done


def _file_sha(files, path, job_dir):
    for n, f in files.items():
        if _real(mask_path(job_dir, n)) == _real(path) and f.get("sha1"):
            return f["sha1"]
    return ""


def write_list(job_dir, data, project_dir=None):
    """tomogram_masks.star: the masks, named from the project (RELION's way; the working directory when the
    job runs, ``project_dir`` when cryo-savanna saves them from wherever it was started)."""
    def rel(path):
        if project_dir is None:
            return _rel(path)
        return os.path.relpath(os.path.abspath(path), os.path.abspath(project_dir))
    rows = [[n, rel(mask_path(job_dir, n))] for n in data["tomograms"]]
    B.write_star(os.path.join(job_dir, LIST), [("global", None, [B.TOMO_NAME, COL_MASK], rows)])


# ---------------------------------------------------------------------------- the job

def run_masks(job_dir, opts):
    star, _, rows = B.tomogram_rows(opts)
    data = read_geometry(job_dir)
    data["version"] = VERSION
    data["in_tomograms"] = star
    data["tomo_kind"] = text(opts, "tomo_kind") or "Reconstructed"
    start = text(opts, "start_z") or START_ALL
    thickness = number(opts, "start_thickness", what="Slab thickness")
    margin = number(opts, "start_margin", what="Margin") or 0.0
    tomos = data["tomograms"]
    for row in rows:
        name = row[B.TOMO_NAME]
        tomogram = B.tomogram_of(row, opts)
        nx, ny, nz, header_voxel = B.mrc_header(tomogram)
        px, binning = B._f(row, "rlnTomoTiltSeriesPixelSize"), B._f(row, "rlnTomoTomogramBinning")
        voxel = px * binning if px and binning else header_voxel
        old = tomos.get(name)
        shape = [nx, ny, nz]
        if old is not None and old.get("shape") == shape:
            old.update(tomogram=tomogram, voxel=voxel)
            continue
        if old is not None:
            say("%s: the tomogram is now %d×%d×%d: its mask starts again" % (name, nx, ny, nz))
        tomos[name] = {"tomogram": tomogram, "shape": shape, "voxel": voxel,
                       "geometry": default_geometry(shape, Z_SLAB if start == START_SLAB else Z_ALL,
                                                    thickness, margin)}
    say("%d tomograms" % len(tomos))
    done = save_masks(job_dir, data, progress=lambda n: say("writing the mask of %s" % n))
    for what in ("written", "linked", "kept"):
        n = sum(1 for v in done.values() if v == what)
        if n:
            say("%d masks %s" % (n, what if what != "linked" else "linked to an identical one"))
    return [(os.path.join(job_dir, LIST), NODE_LIST)]


def run(argv):
    mode, job_dir, _threads, opts = B.parse(argv, MODES)
    B.say = say
    for f in (B.SUCCESS, B.FAILURE, B.ABORTED):
        if os.path.exists(os.path.join(job_dir, f)):
            os.remove(os.path.join(job_dir, f))
    say("cryo-savanna tomogram masks, job %s" % job_dir)
    try:
        nodes = run_masks(job_dir, opts)
    except Usage as e:
        print("ERROR: " + str(e), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, B.FAILURE))
        return 1
    B.write_output_nodes(job_dir, [(_rel(p), t) for p, t in nodes])
    say("done: edit the masks in the job's Masks view in cryo-savanna")
    _touch(os.path.join(job_dir, B.SUCCESS))
    return 0


def main():
    argv = sys.argv[1:]
    try:
        sys.exit(run(argv))
    except Usage as e:
        print("cryo-savanna mask_runner: " + str(e), file=sys.stderr, flush=True)
        B._fail(argv)
        sys.exit(2)
    except Exception:  # anything else: still an exit file, or RELION shows the job running for ever
        traceback.print_exc()
        sys.stderr.flush()
        B._fail(argv)
        sys.exit(1)


if __name__ == "__main__":
    main()
