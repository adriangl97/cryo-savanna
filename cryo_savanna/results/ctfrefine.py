# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What a CtfRefine job (relion_ctf_refine) found, read for its viewers (no Qt).

The job writes ``particles_ctf_refine.star`` (the input particles with new CTF values, and new optics
values in its optics table) and ``logfile.pdf``. Depending on what it fitted it also writes, per
optics group N, 2D phase images of the aberrations in Fourier space (as in RELION 5.1.1's
``src/jaz/single_particle/ctf/``):

* beam tilt and trefoil (antisymmetric): ``beamtilt_delta-phase_per-pixel_optics-group_N.mrc`` (what
  the data show) and ``beamtilt_delta-phase_iter-fit_optics-group_N[_N-3].mrc`` (the fitted model);
* 4th-order (symmetric) aberrations: ``aberr_delta-phase_per-pixel_optics-group_N.mrc`` and
  ``aberr_delta-phase_iter-fit_optics-group_N_N-4.mrc``;
* anisotropic magnification: ``mag_disp_{x,y}_optics-group_N.mrc`` and ``mag_disp_{x,y}_fit_optics-group_N.mrc``.

Per micrograph (in folders that follow the micrograph names) it writes ``*_defocus_fit.star`` or
``*_bfactor_fit.star`` as it goes, which tells how far a running job is.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna.project.star import StarBlock, StarError, loop_columns, read_star_head
from cryo_savanna.results.tables import Table
from cryo_savanna.stamps import Cache, stamp

OUTPUT = "particles_ctf_refine.star"

# ---------------------------------------------------------------------------- what the job fitted

CTF_FITS = [("do_defocus", "Defocus"), ("do_astig", "Astigmatism"), ("do_bfactor", "B-factor"),
            ("do_phase", "Phase shift")]


def fitted(options: dict[str, str]) -> list[tuple[str, str]]:
    """(parameter, how) for what the job estimated, from its job.star, as the command RELION makes:
    anisotropic magnification alone, else the CTF parameters, beam tilt (and trefoil), 4th order."""
    yes = lambda k: options.get(k, "").strip().lower() in ("yes", "true", "1")  # noqa: E731
    if yes("do_aniso_mag"):
        return [("Anisotropic magnification", "per optics group")]
    out = []
    if yes("do_ctf"):
        for key, label in CTF_FITS:
            how = options.get(key, "No")
            if how and how != "No":
                out.append((label, how.lower()))
    if yes("do_tilt"):
        out.append(("Beam tilt", "per optics group"))
        if yes("do_trefoil"):
            out.append(("Trefoil", "per optics group"))
    if yes("do_4thorder"):
        out.append(("4th-order aberrations", "per optics group"))
    return out


def fits_ctf(options: dict[str, str]) -> bool:
    """Whether the job changed per-particle (or per-micrograph) CTF values; True when job.star says nothing."""
    if not options:
        return True
    return any(label in ("Defocus", "Astigmatism", "B-factor", "Phase shift") for label, _ in fitted(options))


def min_resolution(options: dict[str, str]) -> float | None:
    try:
        v = float(options.get("minres", ""))
    except ValueError:
        return None
    return v if v > 0 else None


# ---------------------------------------------------------------------------- optics groups

ODD_NAMES = {(1, -1): "tilt / shift", (1, 1): "tilt / shift", (3, -1): "coma (beam tilt)",
             (3, 1): "coma (beam tilt)", (3, -3): "trefoil", (3, 3): "trefoil"}
EVEN_NAMES = {(0, 0): "piston", (2, -2): "astigmatism", (2, 0): "defocus", (2, 2): "astigmatism",
              (4, -4): "tetrafoil", (4, -2): "secondary astigmatism", (4, 0): "spherical aberration",
              (4, 2): "secondary astigmatism", (4, 4): "tetrafoil"}


def odd_index(i: int) -> tuple[int, int]:
    """(n, m) of RELION's i-th antisymmetric Zernike coefficient (Zernike::oddIndexToMN)."""
    k = int((np.sqrt(1 + 4 * i) - 1.0) / 2.0)
    n = 2 * k + 1
    return n, 2 * (i - (k * k + k)) - n


def even_index(i: int) -> tuple[int, int]:
    """(n, m) of RELION's i-th symmetric Zernike coefficient (Zernike::evenIndexToMN)."""
    k = int(np.sqrt(i))
    return 2 * k, 2 * (i - k * k - k)


def parse_vector(v) -> list[float]:
    """A RELION vector value, "[a,b,c]"."""
    if v is None:
        return []
    s = str(v).strip().strip("[]")
    out = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


@dataclass
class Zernike:
    n: int
    m: int
    name: str
    before: float | None
    after: float | None

    @property
    def label(self) -> str:
        return f"Z({self.n},{self.m:+d})" if self.m else f"Z({self.n},0)"


@dataclass
class Magnification:
    matrix: np.ndarray  # 2x2

    @property
    def singular_values(self) -> tuple[float, float]:
        s = np.linalg.svd(self.matrix, compute_uv=False)
        return float(s[0]), float(s[1])

    @property
    def mean(self) -> float:
        """Overall scale: what RELION calls the mean magnification (1 = the nominal pixel size)."""
        a, b = self.singular_values
        return (a + b) / 2

    @property
    def anisotropy(self) -> float:
        """Difference of the two magnifications, in % (RELION's "magnification anisotropy")."""
        a, b = self.singular_values
        return abs(a - b) * 100.0

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2)))


@dataclass
class OpticsGroup:
    number: int
    name: str
    pixel_size: float | None = None  # Å, of the particle images
    box: int | None = None
    voltage: float | None = None
    cs: float | None = None
    cs_before: float | None = None
    tilt_before: tuple[float, float] | None = None  # mrad
    tilt_after: tuple[float, float] | None = None
    odd: list[Zernike] = field(default_factory=list)
    even: list[Zernike] = field(default_factory=list)
    mag_before: Magnification | None = None
    mag_after: Magnification | None = None

    @property
    def tilt_changed(self) -> bool:
        return self.tilt_after is not None and self.tilt_after != (self.tilt_before or (0.0, 0.0))


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _optics(path: Path | None) -> StarBlock | None:
    if path is None or not path.is_file():
        return None
    try:
        return read_star_head(path, ["optics"]).get("optics")
    except (OSError, StarError, ValueError):
        return None


def _rows(block: StarBlock | None) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if block is None:
        return out
    for k, rec in enumerate(block.records()):
        try:
            num = int(rec.get("rlnOpticsGroup", k + 1))
        except (TypeError, ValueError):
            num = k + 1
        out[num] = rec
    return out


def _tilt(rec: dict | None) -> tuple[float, float] | None:
    if not rec or ("rlnBeamTiltX" not in rec and "rlnBeamTiltY" not in rec):
        return None
    return (_num(rec.get("rlnBeamTiltX")) or 0.0, _num(rec.get("rlnBeamTiltY")) or 0.0)


def _mag(rec: dict | None) -> Magnification | None:
    if not rec or "rlnMagMat00" not in rec:
        return None
    vals = [_num(rec.get(f"rlnMagMat{i}{j}")) for i in (0, 1) for j in (0, 1)]
    if any(v is None for v in vals):
        return None
    return Magnification(np.array(vals, dtype=float).reshape(2, 2))


def _zernikes(before: dict | None, after: dict | None, label: str, index, names) -> list[Zernike]:
    b = parse_vector((before or {}).get(label))
    a = parse_vector((after or {}).get(label))
    out = []
    for i in range(max(len(a), len(b))):
        n, m = index(i)
        out.append(Zernike(n, m, names.get((n, m), ""), b[i] if i < len(b) else None,
                           a[i] if i < len(a) else None))
    return out


def read_optics(output: Path, input_star: Path | None = None) -> list[OpticsGroup]:
    """The optics groups of the job's output, with the values they had in its input."""
    after, before = _rows(_optics(output)), _rows(_optics(input_star))
    groups = []
    for num, rec in sorted(after.items()):
        old = before.get(num)
        g = OpticsGroup(num, str(rec.get("rlnOpticsGroupName", f"opticsGroup{num}")))
        g.pixel_size = _num(rec.get("rlnImagePixelSize"))
        box = _num(rec.get("rlnImageSize"))
        g.box = int(box) if box else None
        g.voltage = _num(rec.get("rlnVoltage"))
        g.cs = _num(rec.get("rlnSphericalAberration"))
        g.cs_before = _num(old.get("rlnSphericalAberration")) if old else None
        g.tilt_before, g.tilt_after = _tilt(old), _tilt(rec)
        g.odd = _zernikes(old, rec, "rlnOddZernike", odd_index, ODD_NAMES)
        g.even = _zernikes(old, rec, "rlnEvenZernike", even_index, EVEN_NAMES)
        g.mag_before, g.mag_after = _mag(old), _mag(rec)
        groups.append(g)
    return groups


# ---------------------------------------------------------------------------- per particle

OUT_COLUMNS = ["rlnImageName", "rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnOpticsGroup",
               "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnCtfBfactor", "rlnCtfScalefactor",
               "rlnPhaseShift"]
IN_COLUMNS = ["rlnImageName", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnCtfBfactor",
              "rlnCtfScalefactor", "rlnPhaseShift"]

# derived columns of the per-particle table
DEFOCUS = "Defocus"  # (U + V) / 2, Å
ASTIG = "Astigmatism"  # |U - V|, Å
D_DEFOCUS = "DefocusChange"
D_ASTIG = "AstigmatismChange"
D_ANGLE = "DefocusAngleChange"
D_PHASE = "PhaseShiftChange"
D_BFACTOR = "CtfBfactorChange"
D_SCALE = "CtfScalefactorChange"

UNITS = {DEFOCUS: "Å", ASTIG: "Å", D_DEFOCUS: "Å", D_ASTIG: "Å", D_ANGLE: "°", D_PHASE: "°",
         D_BFACTOR: "Å²", D_SCALE: "", "rlnDefocusU": "Å", "rlnDefocusV": "Å", "rlnDefocusAngle": "°",
         "rlnCtfBfactor": "Å²", "rlnCtfScalefactor": "", "rlnPhaseShift": "°"}


def _floats(vals: list[str] | None) -> np.ndarray | None:
    if vals is None:
        return None
    try:
        return np.array(vals, dtype=np.float64)
    except ValueError:
        out = np.full(len(vals), np.nan)
        for i, v in enumerate(vals):
            try:
                out[i] = float(v)
            except ValueError:
                pass
        return out


def _wrap(d: np.ndarray, period: float) -> np.ndarray:
    """Into [-period/2, period/2)."""
    return (d + period / 2) % period - period / 2


@dataclass
class ParticleChanges:
    table: Table  # per particle: the output's CTF columns, the derived ones and the changes
    matched: int  # particles found in the input too
    input_star: str = ""

    @property
    def has_changes(self) -> bool:
        return self.table.has(D_DEFOCUS)


def read_particle_changes(output: Path, input_star: Path | None = None) -> ParticleChanges | None:
    """The new CTF values of every particle, and how they changed from the input (matched by image name)."""
    r = loop_columns(output, "particles", OUT_COLUMNS)
    if r is None:
        return None
    data = r[1]
    names = data.get("rlnImageName", [])
    n = len(names)
    cols: dict = {}
    order: list[str] = []

    def put(label: str, values) -> None:
        if values is not None:
            cols[label] = values
            order.append(label)

    for lab in ("rlnMicrographName", "rlnImageName"):
        if lab in data:
            put(lab, data[lab])
    num = {lab: _floats(data.get(lab)) for lab in OUT_COLUMNS[2:]}
    for lab in ("rlnCoordinateX", "rlnCoordinateY", "rlnOpticsGroup"):
        put(lab, num[lab])
    u, v = num["rlnDefocusU"], num["rlnDefocusV"]
    if u is not None and v is not None:
        put(DEFOCUS, (u + v) / 2)
        put(ASTIG, np.abs(u - v))
    for lab in ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnCtfBfactor", "rlnCtfScalefactor",
                "rlnPhaseShift"):
        put(lab, num[lab])

    matched = 0
    r_in = loop_columns(input_star, "particles", IN_COLUMNS) if input_star and input_star.is_file() else None
    if r_in is not None and "rlnImageName" in r_in[1] and n:
        din = r_in[1]
        in_names = din["rlnImageName"]
        if in_names == names:
            idx = np.arange(n)
        else:  # the output is grouped by micrograph: find each particle in the input
            pos = {nm: i for i, nm in enumerate(in_names)}
            idx = np.array([pos.get(nm, -1) for nm in names], dtype=np.int64)
        ok = idx >= 0
        matched = int(ok.sum())

        def before(label: str) -> np.ndarray | None:
            col = _floats(din.get(label))
            if col is None:
                return None
            out = np.full(n, np.nan)
            out[ok] = col[idx[ok]]
            return out

        u0, v0, a0 = before("rlnDefocusU"), before("rlnDefocusV"), before("rlnDefocusAngle")
        if matched and u is not None and v is not None and u0 is not None and v0 is not None:
            put(D_DEFOCUS, (u + v) / 2 - (u0 + v0) / 2)
            put(D_ASTIG, np.abs(u - v) - np.abs(u0 - v0))
            if a0 is not None and num["rlnDefocusAngle"] is not None:
                put(D_ANGLE, _wrap(num["rlnDefocusAngle"] - a0, 180.0))
        for lab, dlab, period in (("rlnPhaseShift", D_PHASE, 360.0), ("rlnCtfBfactor", D_BFACTOR, None),
                                  ("rlnCtfScalefactor", D_SCALE, None)):
            b = before(lab)
            if matched and b is not None and num[lab] is not None:
                d = num[lab] - b
                put(dlab, _wrap(d, period) if period else d)
    return ParticleChanges(Table("particles", order, cols, str(output)), matched,
                           str(input_star) if r_in is not None else "")


@dataclass
class Stat:
    label: str
    n: int
    mean: float
    sd: float
    median: float
    lo: float  # 1st percentile
    hi: float  # 99th percentile
    median_abs: float


def stat(table: Table, label: str) -> Stat | None:
    v = table.numeric(label) if table.has(label) else None
    if v is None:
        return None
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    lo, hi = np.percentile(v, [1, 99])
    return Stat(label, len(v), float(v.mean()), float(v.std()), float(np.median(v)), float(lo), float(hi),
                float(np.median(np.abs(v))))


@dataclass
class MicrographSummary:
    names: list[str]
    particles: np.ndarray
    mean: dict[str, np.ndarray]  # label -> per-micrograph mean
    sd: dict[str, np.ndarray]


def _unique_names(column) -> tuple[list[str], np.ndarray]:
    """The different names of a column, sorted, and for every row which of them it has: what
    np.unique(..., return_inverse=True) gives, without an array of a million names as wide as the longest
    (several hundred MB) and its sort; one look-up per row instead."""
    first: dict[str, int] = {}
    seen = np.fromiter((first.setdefault(str(n), len(first)) for n in column), np.int64, len(column))
    names = sorted(first)
    rank = np.empty(len(names), np.int64)
    rank[[first[n] for n in names]] = np.arange(len(names))
    return names, rank[seen] if len(names) else seen


def per_micrograph(table: Table, labels: list[str]) -> MicrographSummary | None:
    """Mean and spread of some per-particle values on each micrograph."""
    if not table.has("rlnMicrographName"):
        return None
    names, inv = _unique_names(table.column("rlnMicrographName"))
    counts = np.bincount(inv, minlength=len(names)).astype(float)
    mean, sd = {}, {}
    for lab in labels:
        v = table.numeric(lab) if table.has(lab) else None
        if v is None:
            continue
        ok = np.isfinite(v)
        c = np.bincount(inv[ok], minlength=len(names)).astype(float)
        s = np.bincount(inv[ok], weights=v[ok], minlength=len(names))
        s2 = np.bincount(inv[ok], weights=v[ok] ** 2, minlength=len(names))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
            mean[lab] = m
            sd[lab] = np.sqrt(np.maximum(s2 / c - m ** 2, 0))
    return MicrographSummary([str(x) for x in names], counts.astype(int), mean, sd)


# ---------------------------------------------------------------------------- aberration images

@dataclass
class PhaseImage:
    title: str
    path: str  # relative to the project
    kind: str  # "phase" (radians, wraps at ±π) or "displacement"


@dataclass
class AberrationRow:
    title: str  # "Beam tilt / trefoil (antisymmetric)", ...
    observed: PhaseImage | None
    fit: PhaseImage | None
    order: int | None = None  # highest Zernike order of the fit (N), when the file name says


@dataclass
class AberrationSet:
    group: int
    rows: list[AberrationRow] = field(default_factory=list)


_GROUP = r"optics-group_(\d+)"
_PATTERNS = [  # (row, observed/fit, regex, kind)
    ("asym", "obs", re.compile(rf"^beamtilt_delta-phase_per-pixel_{_GROUP}\.mrc$"), "phase"),
    ("asym", "fit", re.compile(rf"^beamtilt_delta-phase_iter-fit_{_GROUP}(?:_N-(\d+))?\.mrc$"), "phase"),
    ("sym", "obs", re.compile(rf"^aberr_delta-phase_per-pixel_{_GROUP}\.mrc$"), "phase"),
    ("sym", "fit", re.compile(rf"^aberr_delta-phase_iter-fit_{_GROUP}(?:_N-(\d+))?\.mrc$"), "phase"),
    ("magx", "obs", re.compile(rf"^mag_disp_x_{_GROUP}\.mrc$"), "displacement"),
    ("magx", "fit", re.compile(rf"^mag_disp_x_fit_{_GROUP}\.mrc$"), "displacement"),
    ("magy", "obs", re.compile(rf"^mag_disp_y_{_GROUP}\.mrc$"), "displacement"),
    ("magy", "fit", re.compile(rf"^mag_disp_y_fit_{_GROUP}\.mrc$"), "displacement"),
]
ROW_TITLES = {"asym": "Antisymmetric aberrations (beam tilt, trefoil)",
              "sym": "Symmetric aberrations (4th order)",
              "magx": "Magnification: X displacement", "magy": "Magnification: Y displacement"}


def aberration_images(project_dir: Path, job: str) -> list[AberrationSet]:
    """The phase (and displacement) images of the job, per optics group, from its directory listing."""
    job_dir = Path(project_dir) / job
    try:
        names = sorted(os.listdir(job_dir))
    except OSError:
        return []
    found: dict[int, dict[str, dict]] = {}
    for name in names:
        for row, which, rx, kind in _PATTERNS:
            mt = rx.match(name)
            if not mt:
                continue
            g = int(mt.group(1))
            order = int(mt.group(2)) if mt.lastindex and mt.lastindex >= 2 and mt.group(2) else None
            slot = found.setdefault(g, {}).setdefault(row, {})
            title = ("Observed" if which == "obs" else "Fit" + (f" (N = {order})" if order else ""))
            img = PhaseImage(title, job + name, kind)
            prev = slot.get(which)
            # of several fits (the job continued with another order), the highest order
            if prev is None or (which == "fit" and (order or 0) >= (slot.get("order") or 0)):
                slot[which] = img
                if which == "fit":
                    slot["order"] = order
            break
    out = []
    for g in sorted(found):
        s = AberrationSet(g)
        for row in ("asym", "sym", "magx", "magy"):
            r = found[g].get(row)
            if r:
                s.rows.append(AberrationRow(ROW_TITLES[row], r.get("obs"), r.get("fit"), r.get("order")))
        out.append(s)
    return out


def residual(observed: np.ndarray, fit: np.ndarray, kind: str) -> np.ndarray:
    """What the fit leaves: for phases, the difference wrapped into [-π, π)."""
    d = observed - fit
    return _wrap(d, 2 * np.pi) if kind == "phase" else d


# ---------------------------------------------------------------------------- progress

FIT_SUFFIXES = ("_defocus_fit.star", "_bfactor_fit.star")


def micrographs_done(job_dir: Path, limit: int = 200_000) -> int:
    """Micrographs a (running) job has finished its per-micrograph fit for."""
    done: set[str] = set()
    for dirpath, _, filenames in os.walk(job_dir):
        for f in filenames:
            for suf in FIT_SUFFIXES:
                if f.endswith(suf):
                    done.add(os.path.join(dirpath, f[: -len(suf)]))
        if len(done) >= limit:
            break
    return len(done)


# ---------------------------------------------------------------------------- shared reads

_changes_cache = Cache(2)  # (output and its stamp, input and its stamp) -> ParticleChanges | None


def _file_key(p: Path | None):
    return None if p is None else (str(p), stamp(p))


def particle_changes(output: Path, input_star: Path | None = None) -> ParticleChanges | None:
    """read_particle_changes, kept while neither file changes: the summary and the plots of a job share it."""
    key = (_file_key(output), _file_key(input_star))
    # one read at a time: the views ask from the viewers' thread pool, and two asking for the same job at
    # once (its summary and its plots) would both read the particles; the second now finds the first's
    return _changes_cache.get_or_make(key, lambda: read_particle_changes(output, input_star), one_at_a_time=True)
