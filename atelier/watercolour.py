"""Watercolour: transparent colour carried by water and left behind where the water dries.

A wash is a puddle. While it is wet the pigment goes where the water goes, and the water runs
to wherever it evaporates fastest, which is the edge: so a wash left to dry by itself comes out
a little paler inside a thin dark rim. Where the brush laid more water the puddle stands deeper
and dries last, and the pigment gathers there in soft pools; where the shrinking edge stopped
for a while it leaves a faint tide line. Heavy pigment settles into the hollows of the paper,
which is granulation. Colour touched into a wash that is still wet runs through the wet and
stops where the wet stops: its inner edge is soft, its outer edge is the wash's own.

Everything here returns maps (0..1 or densities) for `color.glaze` to lay down.
"""

import numpy as np
from scipy import ndimage

from . import noise


def _box(mask, pad):
    """Slices round the part of `mask` that is set, `pad` px to spare: the work is done in there."""
    ys, xs = np.nonzero(mask > 0.02)
    h, w = mask.shape
    return (slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1)),
            slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1)))


def puddle(mask, sheet, r, ragged=1.5):
    """Where the water lies: the shape the brush filled, its edge wandering a little with the paper."""
    box = _box(mask, 12)
    s = mask[box].shape
    m = noise.warp(mask[box], ragged * (noise.field(s, 30, r) + 0.3 * noise.field(s, 4, r)),
                   ragged * (noise.field(s, 30, r) + 0.3 * noise.field(s, 4, r)))
    soak = 0.08 * (sheet.fiber[box] - 0.5)          # a fibre touching the edge draws the water out a hair
    out = np.zeros_like(mask)
    out[box] = noise.smoothstep(0.3, 0.7, ndimage.gaussian_filter(m, 0.7) + soak)
    return out


def deposit(wet, sheet, r, pool=0.25, rim=0.8, rim_width=3.0, tides=0.2, grain=0.3, scale=150.0):
    """How thickly a puddle over `wet` left its pigment, as a factor on the wash (1 = even).

    pool       how much more the deep places gather
    rim        pigment carried out to the edge as it dried; rim_width its px
    tides      the faint lines left where the shrinking edge paused
    grain      granulation: pigment settled in the hollows of the tooth
    scale      px across the deep and shallow places of the puddle
    """
    box = _box(wet, 4)
    w = wet[box]
    depth = noise.fbm(w.shape, scale, r, octaves=4)
    f = np.clip(1 + pool * depth, 0.4, None)
    d = ndimage.distance_transform_edt(w > 0.5)
    # a deep puddle beside the edge feeds it more: the rim is uneven along its length
    f += rim * np.exp(-d / rim_width) * np.clip(0.8 + 0.35 * noise.field(w.shape, 60, r) + 0.25 * depth, 0.2, None)
    for level in (0.8, 1.5):
        above = depth - level
        f += tides * noise.smoothstep(0, 0.03, above) * np.exp(-np.clip(above, 0, None) / 0.1)
    # pigment settles in the hollows of the tooth and clots a little as it does
    valley = 0.5 - ndimage.gaussian_filter(sheet.tooth[box], 1.2)
    f *= 1 + grain * (1.6 * valley + 0.25 * noise.field(w.shape, 3.0, r))
    out = np.zeros_like(wet)
    out[box] = np.clip(f, 0, None) * w
    return out


def charge(colour, wet, spread, r, streak=0.4):
    """Colour run into a wash while it is still wet: it travels about `spread` px through the wet
    and stops dead at its edge, and the brush that drew it inward leaves streaks square to the edge."""
    box = _box(wet, 4)
    w, c = wet[box], colour[box]
    c = ndimage.gaussian_filter(c * w, spread) / (ndimage.gaussian_filter(w, spread) + 1e-3) * w
    _, (iy, ix) = ndimage.distance_transform_edt(w > 0.5, return_indices=True)
    hairs = 0.75 * noise.field(w.shape, 2.5, r) + 0.25 * noise.field(w.shape, 10, r)
    out = np.zeros_like(wet)
    # each pixel takes the streak of its nearest point on the edge, so streaks run inward from it
    out[box] = c * np.clip(1 + streak * hairs[iy, ix], 0, None)
    return out
