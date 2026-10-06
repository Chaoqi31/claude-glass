"""Grounds: the surfaces that receive the marks. Paper, and later silk, rock, canvas."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import noise
from .color import lin


@dataclass
class Sheet:
    color: np.ndarray   # (H,W,3) linear reflectance of the bare surface
    tooth: np.ndarray   # (H,W) surface height in [0,1]; dry brush only catches the peaks
    fiber: np.ndarray   # (H,W) fibre map in [0,1]; wet ink travels along it
    alpha: np.ndarray   # (H,W) 1 on the sheet, 0 off it


def _norm(a, lo=1, hi=99):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1).astype(np.float32)


def deckle(shape, margin, r, ragged=6.0):
    """Sheet mask with a hand-made deckle edge: soft, thinning, a little fibrous."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    top, left = margin if isinstance(margin, tuple) else (margin, margin)
    d = np.minimum.reduce([yy - top, h - 1 - top - yy, xx - left, w - 1 - left - xx])
    wobble = ragged * (0.7 * noise.field(shape, 40, r) + 0.3 * noise.field(shape, 6, r))
    tufts = noise.fibers(shape, r, count=int((h + w) * 1.2), length=(8, 40), strength=(0.5, 1.0))
    d = d + wobble + 4.0 * ndimage.gaussian_filter(tufts, 0.6)
    return noise.smoothstep(-1.0, 5.0, d)


def washi(shape, seed, tint="#f0ebdf", margin=90, fibre_density=1.0):
    """Kozo paper: long visible fibres, cloudy formation, a few flecks of bark."""
    r = noise.rng(seed)
    h, w = shape
    area = h * w
    fib = noise.fibers(shape, r, count=int(area / 450 * fibre_density), length=(40, 260),
                       curl=0.08, strength=(0.15, 0.8))
    fib = ndimage.gaussian_filter(fib, 0.7)
    fine = noise.fibers(shape, r, count=int(area / 300), length=(8, 40), curl=0.2, strength=(0.1, 0.5))
    fine = ndimage.gaussian_filter(fine, 0.5)
    formation = noise.fbm(shape, max(h, w) / 5, r, octaves=6, gain=0.55)
    grain = noise.field(shape, 1.0, r)

    tooth = _norm(0.35 * noise.field(shape, 1.3, r) + 0.25 * noise.field(shape, 3.5, r)
                  + 0.9 * fib + 0.6 * fine - 0.15 * formation)

    base = lin(tint)
    light = 1.0 + 0.016 * formation + 0.03 * np.clip(fib, 0, 1) + 0.01 * fine + 0.004 * grain
    color = base[None, None, :] * light[..., None]

    # bark flecks (chiri): rare, tiny, warm
    flecks = np.zeros(shape, np.float32)
    n = int(area / 250_000)
    ys, xs = r.integers(0, h, n), r.integers(0, w, n)
    flecks[ys, xs] = r.uniform(0.5, 1.0, n)
    flecks = ndimage.gaussian_filter(flecks, r.uniform(0.8, 1.6)) * 6
    color = color * (1 - np.clip(flecks, 0, 0.35)[..., None] * np.array([0.55, 0.65, 0.8], np.float32))

    alpha = deckle(shape, margin, r) if margin else np.ones(shape, np.float32)
    return Sheet(color.astype(np.float32), tooth, _norm(fib + 0.5 * fine), alpha)


def laid(shape, seed, tint="#eee6d3", margin=0, chain=270, rib=8.5, foxing=10):
    """Rag paper made on a laid mould: the wires leave fine ribs across the sheet and chain lines
    a thumb apart down it; after a few centuries, a scatter of rust-brown fox marks."""
    r = noise.rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    formation = noise.fbm(shape, max(h, w) / 6, r, octaves=6, gain=0.55)
    bend = 3 * noise.line1d(h, 200, r)                    # no chain wire is quite straight
    chains = np.exp(-((((xx + bend[:, None]) % chain) - chain / 2) / 2.5) ** 2)
    ribs = 0.5 + 0.5 * np.cos(2 * np.pi * (yy + 1.5 * noise.field(shape, 60, r)) / rib)
    fib = ndimage.gaussian_filter(noise.fibers(shape, r, count=int(h * w / 900), length=(6, 30), curl=0.25,
                                               strength=(0.1, 0.4)), 0.5)
    tooth = _norm(0.5 * noise.field(shape, 1.4, r) + 0.3 * noise.field(shape, 4, r) + 0.35 * ribs + 0.4 * fib)
    light = (1 + 0.012 * formation - 0.006 * chains + 0.004 * (ribs - 0.5) + 0.006 * fib
             + 0.004 * noise.field(shape, 1.0, r))
    color = lin(tint)[None, None, :] * light[..., None]
    fox = np.zeros(shape, np.float32)
    for _ in range(foxing):
        cy, cx, rad = r.uniform(0, h), r.uniform(0, w), r.uniform(2, 12)
        y0, y1, x0, x1 = (max(0, int(cy - rad * 4)), min(h, int(cy + rad * 4)),
                          max(0, int(cx - rad * 4)), min(w, int(cx + rad * 4)))
        d = np.hypot(yy[y0:y1, x0:x1] - cy, xx[y0:y1, x0:x1] - cx) / rad
        fox[y0:y1, x0:x1] += r.uniform(0.2, 0.6) * (np.exp(-d * d * 2) + 0.3 * np.exp(-d))
    color *= 1 - np.clip(fox, 0, 0.8)[..., None] * np.array([0.18, 0.32, 0.5], np.float32)
    alpha = deckle(shape, margin, r, ragged=1.2) if margin else np.ones(shape, np.float32)
    return Sheet(color.astype(np.float32), tooth, _norm(fib), alpha)
