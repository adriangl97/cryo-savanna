# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Software isosurface rendering of a map (no OpenGL): rays cast through the binned volume with numpy.

For each pixel a ray is marched front to back through the volume (orthographic view); the first
sample above the threshold is refined by linear interpolation between the last two samples, and
shaded from the map's gradient there (trilinear, so the surface is smooth). Rays are only marched
through the box around the voxels above the threshold and stop at their first hit, so a 192³ map
renders in ~0.06 s at 320² and ~0.01 s at the size used while rotating. The surface can be
coloured by a second map sampled at the hit points (local resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cryo_savanna.images.volume import colormap

BASE = np.array([0.72, 0.74, 0.80], np.float32)  # surface colour (grey-blue, as ChimeraX maps)
BACKGROUND = np.array([1.0, 1.0, 1.0], np.float32)


def rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Rows: screen x, screen y (down), view direction (into the screen), in volume (x, y, z) coordinates."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (rz @ rx @ ry).astype(np.float64)


def trilinear(vol: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """vol[z, y, x] at fractional positions (clamped to the box)."""
    nz, ny, nx = vol.shape
    x = np.clip(x, 0, nx - 1.001)
    y = np.clip(y, 0, ny - 1.001)
    z = np.clip(z, 0, nz - 1.001)
    x0, y0, z0 = x.astype(np.intp), y.astype(np.intp), z.astype(np.intp)
    fx, fy, fz = x - x0, y - y0, z - z0
    c: np.ndarray | float = 0.0
    for dz, wz in ((0, 1 - fz), (1, fz)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                c = c + vol[z0 + dz, y0 + dy, x0 + dx] * (wz * wy * wx)
    return np.asarray(c)


@dataclass
class Rendered:
    rgb: np.ndarray  # uint8 [h, w, 3]
    hit: np.ndarray  # bool [h, w]
    depth: np.ndarray  # float [h, w], along the view direction (voxels), inf where nothing was hit


def occupied(vol: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Corners (x, y, z) of the box around the voxels above the threshold; None if there are none."""
    above = vol > threshold
    if not above.any():
        return None
    lo, hi = [], []
    for axis in (2, 1, 0):  # x, y, z
        other = tuple(a for a in range(3) if a != axis)
        idx = np.nonzero(above.any(axis=other))[0]
        lo.append(idx[0])
        hi.append(idx[-1])
    return np.array(lo, float), np.array(hi, float)


def framing(vol: np.ndarray, threshold: float,
            bounds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Centre (x, y, z) and radius of the part of the map above the threshold, to frame the view on it."""
    nz, ny, nx = vol.shape
    if bounds is None:
        bounds = occupied(vol, threshold)
    if bounds is None:
        return np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2]), 0.5 * np.sqrt(nx * nx + ny * ny + nz * nz)
    lo, hi = bounds
    radius = 0.45 * float(np.linalg.norm(hi - lo + 1))  # rarely does the map fill the corners of its box
    return (lo + hi) / 2, max(radius, 4.0)


def render(vol: np.ndarray, threshold: float, rot: np.ndarray, size: int = 160, color: np.ndarray | None = None,
           color_range: tuple[float, float] = (0.0, 1.0), step: float = 1.0, zoom: float = 1.0,
           frame: tuple[np.ndarray, float] | None = None,
           bounds: tuple[np.ndarray, np.ndarray] | None = None) -> Rendered:
    """An image (size × size) of the surface vol = threshold, seen along rot[2].

    ``frame`` (centre, radius) is what the image shows (see framing()); by default the whole box.
    Each ray is only marched through ``bounds``, the box around the voxels above the threshold
    (see occupied(); worked out here if not given): rays that miss it cost nothing, the others
    start next to the surface.
    """
    nz, ny, nx = vol.shape
    box_centre = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])
    radius = 0.5 * np.sqrt(nx * nx + ny * ny + nz * nz)
    centre, view_radius = frame if frame is not None else (box_centre, radius)
    # rays start in the plane through the frame's centre, pulled back to cover the whole box
    centre = np.asarray(centre, float)
    centre = centre - rot[2] * float(rot[2] @ (centre - box_centre))
    half = view_radius / zoom
    u = np.linspace(-half, half, size)
    uu, vv = np.meshgrid(u, u)  # [row, col]: rows go down the screen
    origin = centre + uu[..., None] * rot[0] + vv[..., None] * rot[1]  # [h, w, 3] in (x, y, z)
    origin = (origin.reshape(-1, 3) - radius * rot[2]).astype(np.float32)  # t = 0 in front of the box
    d = rot[2].astype(np.float32)
    hit_t = np.full(origin.shape[0], np.inf)
    if bounds is None:
        bounds = occupied(vol, threshold)

    if bounds is not None:
        # where each ray enters and leaves the occupied box (slab test)
        lo, hi = bounds[0] - 0.5, bounds[1] + 0.5
        inv = 1.0 / np.where(np.abs(d) > 1e-9, d, 1e-9)
        t1, t2 = (lo - origin) * inv, (hi - origin) * inv
        t_in = np.maximum(np.minimum(t1, t2).max(axis=1), 0)
        t_out = np.maximum(t1, t2).min(axis=1)
        active = np.nonzero(t_out >= t_in)[0]
        o, t_in, t_out = origin[active], t_in[active].astype(np.float32), t_out[active]
        flat = vol.ravel()
        chunk = 8
        offsets = (np.arange(chunk, dtype=np.float32) * step)[None, :]
        k = 0.0
        while active.size:
            ts = t_in[:, None] + (k + offsets)  # [rays, chunk]
            p = o[:, None, :] + ts[..., None] * d
            ix = np.clip(np.rint(p[..., 0]).astype(np.intp), 0, nx - 1)
            iy = np.clip(np.rint(p[..., 1]).astype(np.intp), 0, ny - 1)
            iz = np.clip(np.rint(p[..., 2]).astype(np.intp), 0, nz - 1)
            above = (flat[(iz * ny + iy) * nx + ix] > threshold) & (ts <= t_out[:, None])
            hit = above.any(axis=1)
            if hit.any():
                rows = np.nonzero(hit)[0]
                hit_t[active[rows]] = ts[rows, above[rows].argmax(axis=1)]
            keep = ~hit & (ts[:, -1] < t_out)
            active, o, t_in, t_out = active[keep], o[keep], t_in[keep], t_out[keep]
            k += chunk * step

    hit = np.isfinite(hit_t)
    rgb = np.empty((origin.shape[0], 3), np.float32)
    rgb[:] = BACKGROUND
    if hit.any():
        idx = np.nonzero(hit)[0]
        t1 = hit_t[idx]
        t0 = t1 - step
        p1 = origin[idx] + t1[:, None] * d
        p0 = origin[idx] + t0[:, None] * d
        v1 = trilinear(vol, p1[:, 0], p1[:, 1], p1[:, 2])
        v0 = trilinear(vol, p0[:, 0], p0[:, 1], p0[:, 2])
        frac = np.clip((threshold - v0) / np.where(np.abs(v1 - v0) > 1e-12, v1 - v0, 1e-12), 0, 1)
        frac = np.where(np.isfinite(v0) & (v0 < threshold), frac, 1.0)
        t = t0 + frac * step
        hit_t[idx] = t
        p = origin[idx] + t[:, None] * d
        h = 0.75
        gx = trilinear(vol, p[:, 0] + h, p[:, 1], p[:, 2]) - trilinear(vol, p[:, 0] - h, p[:, 1], p[:, 2])
        gy = trilinear(vol, p[:, 0], p[:, 1] + h, p[:, 2]) - trilinear(vol, p[:, 0], p[:, 1] - h, p[:, 2])
        gz = trilinear(vol, p[:, 0], p[:, 1], p[:, 2] + h) - trilinear(vol, p[:, 0], p[:, 1], p[:, 2] - h)
        normal = -np.stack([gx, gy, gz], axis=1)  # outward: from dense to empty
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        nv = normal @ rot.T  # in screen coordinates (x right, y down, z into the screen)
        light = np.array([-0.35, -0.45, -0.82])
        light /= np.linalg.norm(light)
        lambert = np.clip(nv @ light, 0, 1)
        half_vec = light + np.array([0, 0, -1.0])
        half_vec /= np.linalg.norm(half_vec)
        spec = np.clip(nv @ half_vec, 0, 1) ** 24
        if color is not None:
            base = colormap(trilinear(color, p[:, 0], p[:, 1], p[:, 2]), *color_range).astype(np.float32)
        else:
            base = np.broadcast_to(BASE, (len(idx), 3))
        depth = t / (2 * radius)  # 0 front .. 1 back
        shade = (0.22 + 0.78 * lambert)[:, None] * base + 0.35 * spec[:, None]
        rgb[idx] = shade * (1.0 - 0.35 * depth[:, None])
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(size, size, 3)
    return Rendered(img, hit.reshape(size, size), hit_t.reshape(size, size))
