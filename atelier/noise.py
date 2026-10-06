"""Noise: the grain of the world. Every irregularity in the museum starts here."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def rng(seed):
    return np.random.default_rng(seed)


def _unit(a):
    a = a - a.mean()
    return (a / (a.std() + 1e-9)).astype(np.float32)


def field(shape, scale, r):
    """Smooth random field with unit variance and features about `scale` px wide."""
    h, w = shape
    if scale < 1.5:
        return _unit(ndimage.gaussian_filter(r.standard_normal(shape), scale * 0.5))
    gh, gw = int(np.ceil(h / scale)) + 3, int(np.ceil(w / scale)) + 3
    g = r.standard_normal((gh, gw)).astype(np.float32)
    big = np.asarray(Image.fromarray(g).resize((round(gw * scale), round(gh * scale)), Image.BICUBIC))
    o = int(scale)
    return _unit(big[o:o + h, o:o + w])


def stretched(shape, sx, sy, r):
    """Smooth field with features `sx` px across and `sy` px along the vertical."""
    h, w = shape
    k = sy / sx
    g = field((int(np.ceil(h / k)) + 2, w), sx, r)
    return _unit(ndimage.zoom(g, (k, 1), order=1)[:h])


def fbm(shape, scale, r, octaves=5, gain=0.5):
    total, amp, norm = np.zeros(shape, np.float32), 1.0, 0.0
    for _ in range(octaves):
        if scale < 0.8:
            break
        total += amp * field(shape, scale, r)
        norm += amp * amp
        amp *= gain
        scale /= 2
    return total / np.sqrt(norm)


def line1d(n, scale, r, rows=1):
    """`rows` independent smooth 1-D signals of length n, unit variance."""
    k = int(np.ceil(n / scale)) + 4
    g = r.standard_normal((rows, k)).astype(np.float32)
    z = ndimage.zoom(g, (1, (n + 3 * scale) / k), order=3)
    return _unit(z[:, int(scale):int(scale) + n]) if rows > 1 else _unit(z[0, int(scale):int(scale) + n])


def warp(img, dx, dy, order=1):
    """Resample `img` (H,W) at (y+dy, x+dx)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ndimage.map_coordinates(img, [yy + dy, xx + dx], order=order, mode="reflect")


def fibers(shape, r, count, length=(30, 160), width=1, curl=0.12, strength=(0.3, 1.0)):
    """Curved fibre strands drawn into a float map in [0, 1]."""
    h, w = shape
    img = Image.new("F", (w, h), 0.0)
    d = ImageDraw.Draw(img)
    seg = 5.0
    lens = r.uniform(*length, count)
    n = int(lens.max() / seg) + 2
    ang = r.uniform(0, 2 * np.pi, (count, 1)) + np.cumsum(r.normal(0, curl, (count, n)), axis=1)
    x = r.uniform(-20, w + 20, (count, 1)) + np.cumsum(np.cos(ang) * seg, axis=1)
    y = r.uniform(-20, h + 20, (count, 1)) + np.cumsum(np.sin(ang) * seg, axis=1)
    val = r.uniform(*strength, count)
    for i in range(count):
        k = max(2, int(lens[i] / seg))
        d.line(list(zip(x[i, :k], y[i, :k])), fill=float(val[i]), width=width)
    return np.asarray(img, dtype=np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)
