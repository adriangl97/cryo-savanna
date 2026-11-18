# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Good box sizes: the sizes (in pixels) for which the FFTs, and so refinement and reconstruction, are fast.

The sizes are EMAN2's list, from the EMAN wiki page "Particle Box Size"
(https://blake.bcm.edu/emanwiki/doku.php?id=eman2:boxsize, last modified 2025-07-04 by Steve Ludtke; CC BY-SA
4.0): sizes up to 1024 for "traditional single particle analysis", and larger ones that it calls "less well
tested, but also probably good". As the page says, a size just above one of these can be much slower (it gives
128 almost 2x faster than 122): when a size is not on the list, take the next larger one. NOTICE.md cites it.

The job editor (ui/job_editor.py) warns next to a box-size option whose value is not on the list, with the
closest good sizes below and above.
"""

from __future__ import annotations

import math
from bisect import bisect_left

SOURCE = "https://blake.bcm.edu/emanwiki/doku.php?id=eman2:boxsize"
SOURCE_NAME = "EMAN2's list of good box sizes"

GOOD = (
    24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 84, 96, 100, 104, 112, 120, 128, 132, 140, 168, 180, 192, 196,
    208, 216, 220, 224, 240, 256, 260, 288, 300, 320, 352, 360, 384, 416, 440, 448, 480, 512, 540, 560, 576, 588,
    600, 630, 640, 648, 672, 686, 700, 720, 750, 756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980,
    1000, 1008, 1024,
)
GOOD_LARGE = (  # "less well tested, but also probably good"
    1050, 1080, 1120, 1134, 1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458, 1470,
    1500, 1512, 1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800, 1890, 1920, 1944, 1960, 2000, 2016,
    2048, 2058, 2100, 2160, 2240, 2250, 2268, 2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688,
    2700, 2744, 2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402, 3430, 3456, 3500,
    3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032, 4050, 4096,
)
ALL = GOOD + GOOD_LARGE

# the options that are a box size in pixels, by form (JobVariant.key without ":tomo"): particle boxes, CtfFind's
# FFT box and the estimation boxes of tomography's CTF refinement and frame alignment. Not those in Å (the pixel
# size is not known in the form).
BOX_OPTIONS: dict[str, tuple[str, ...]] = {
    "relion.extract": ("extract_size", "rescale"),
    "relion.polish": ("extract_size", "rescale"),
    "relion.subtract": ("new_box",),
    "relion.reconstructparticletomo": ("box_size", "crop_size"),
    "relion.pseudosubtomo": ("binned_size_pix",),
    "relion.ctffind": ("box",),  # the boxes CTFFIND takes FFTs of
    "relion.ctfrefinetomo": ("box_size",),  # "Box size for estimation"
    "relion.framealigntomo": ("box_size",),
    "cli:relion_preprocess": ("--extract_size", "--scale"),
    "cli:relion_image_handler": ("--new_box",),
    "cli:relion_particle_subtract": ("--new_box",),
    "cli:relion_motion_refine": ("--scale", "--crop"),
    "cli:relion_reconstruct": ("--newbox",),
    "cli:relion_tomo_subtomo": ("--b", "--crop", "--binned_size_pix"),
    "cli:relion_tomo_reconstruct_particle": ("--b", "--crop"),
}


def box_options(variant_key: str) -> tuple[str, ...]:
    """The box-size options of a form ("relion.extract:tomo" -> those of relion.extract)."""
    return BOX_OPTIONS.get(variant_key.removesuffix(":tomo"), ())


def as_size(text: str) -> int | None:
    """The box size typed (a whole number of pixels), or None when there is none to look at (empty, -1 or 0:
    RELION's "as the input", not a number)."""
    try:
        v = float(text.strip())
    except ValueError:
        return None
    if not math.isfinite(v) or v <= 0 or v != int(v):  # "inf", "nan", "1e999": no whole number
        return None
    return int(v)


def is_good(n: int) -> bool:
    return n in ALL


def neighbours(n: int) -> tuple[int | None, int | None]:
    """The closest good sizes below and above ``n`` (None: none on the list)."""
    i = bisect_left(ALL, n)
    below = ALL[i - 1] if i > 0 else None
    above = ALL[i] if i < len(ALL) and ALL[i] != n else (ALL[i + 1] if i + 1 < len(ALL) else None)
    return below, above


def warning(n: int | None) -> str:
    """Why ``n`` is not a good box size and what to use instead; "" when it is good, or beyond the list."""
    if n is None or is_good(n) or n > ALL[-1]:
        return ""
    below, above = neighbours(n)
    if below is None:
        return f"{n} px is smaller than the smallest good box size ({above} px)."
    if above is None:
        return f"{n} px is not a good box size: the closest good one is {below} px."
    untested = " (less well tested)" if above in GOOD_LARGE else ""
    return (f"{n} px is not a good box size (slow FFTs): the closest good ones are {below} px and "
            f"{above} px{untested}.")
