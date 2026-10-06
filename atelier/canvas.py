"""Canvas: cotton duck and linen, woven, sized, and showing through thin paint."""

import numpy as np
from scipy import ndimage

from . import noise
from .color import lin
from .paper import Sheet


def duck(shape, seed, tint="#cbbd9f", thread=4.2):
    """Plain-weave canvas: warp and weft passing over and under, thick here, thin there."""
    r = noise.rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # threads are never straight or even: they wander and swell (slubs)
    wx = xx + 1.2 * noise.field(shape, 60, r)
    wy = yy + 1.2 * noise.field(shape, 60, r)
    fx, fy = wx / thread, wy / thread
    warp = np.cos(np.pi * (fx % 1.0 - 0.5)) ** 2 * (1 + 0.25 * noise.stretched(shape, thread, 40, r))
    weft = np.cos(np.pi * (fy % 1.0 - 0.5)) ** 2 * (1 + 0.25 * noise.stretched((w, h), thread, 40, r).T[:h, :w])
    over = ((np.floor(fx) + np.floor(fy)) % 2).astype(np.float32)   # which thread is on top
    over = ndimage.gaussian_filter(over, 0.8)
    height = warp * over + weft * (1 - over)
    height = height * (0.85 + 0.15 * noise.field(shape, 8, r))
    tooth = (height - height.min()) / (np.ptp(height) + 1e-6)
    col = lin(tint)[None, None, :] * (0.93 + 0.1 * tooth[..., None]) \
        * (1 + 0.02 * noise.fbm(shape, 200, r)[..., None])
    return Sheet(col.astype(np.float32), tooth.astype(np.float32), np.zeros(shape, np.float32),
                 np.ones(shape, np.float32))
