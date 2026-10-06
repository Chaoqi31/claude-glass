"""Watercolour paper, and the small marks of a pointed brush on it.

A watercolour is painted on rag paper made in a mould and couched between woollen felts. Paper
dried straight from the felts keeps their imprint, a close field of rounded hills with narrow
creases between, and is sold as Rough: a wash settles into the creases, and a dry brush drawn
lightly across touches only the hills, which is how a painter makes sand sparkle or grass bristle.
The sheet is sized so a wash stays where it is put, and wherever it was soaked it dries a little
buckled.

Detail goes on with the point of a round brush. Pressed and lifted in one movement it leaves a
touch shaped like a leaf, fullest a little after it landed; laid on dry paper each touch dries with
its own dark rim, and a mass of them laid one beside another is a tree, a bush, a field of scrub.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import noise
from .color import lin
from .paper import Sheet, deckle


def rough(shape, seed, tint="#f3eee2", margin=72, hill=7.0):
    """Rough rag paper with four deckle edges; its hills are about `hill` px across."""
    r = noise.rng(seed)
    h, w = shape
    felt = noise.field(shape, hill, r) + 0.45 * noise.field(shape, hill / 2, r) + 0.35 * noise.field(shape, hill * 3, r)
    # the felt's knots pressed hollows into the wet sheet: a pebbled surface, creased where the field crosses zero
    hills = np.abs(felt) ** 0.6 + 0.25 * noise.field(shape, 1.2, r)
    fib = ndimage.gaussian_filter(noise.fibers(shape, r, count=h * w // 1600, length=(5, 24), curl=0.3,
                                               strength=(0.1, 0.4)), 0.5)
    tooth = ndimage.gaussian_filter(hills, 0.7) + 0.3 * fib
    lo, hi = np.percentile(tooth, [1, 99])
    tooth = np.clip((tooth - lo) / (hi - lo), 0, 1).astype(np.float32)
    formation = noise.fbm(shape, max(h, w) / 8, r, octaves=5, gain=0.55)
    light = 1 + 0.008 * formation + 0.006 * fib
    color = lin(tint)[None, None, :] * light[..., None]
    fiber = np.clip(fib / (fib.max() + 1e-9) * 2.5, 0, 1).astype(np.float32)
    return Sheet(color.astype(np.float32), tooth, fiber, deckle(shape, margin, r, ragged=5.0))


def touches(shape, rows, belly=0.35):
    """The union of touches of a pointed brush, as a 0..1 mask for `watercolour.puddle`.
    rows (x, y, length, width, angle): each touch lands at (x, y) and travels `length` px along
    `angle` (radians, image coordinates), swelling to `width` px a `belly` of the way along."""
    rows = np.asarray(rows, np.float64)
    h, w = shape
    im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(im)
    t = np.linspace(0, 1, 9)
    half = 0.5 * np.where(t < belly, np.sin(0.5 * np.pi * np.minimum(t / belly, 1)) ** 0.7,
                          np.cos(0.5 * np.pi * np.clip((t - belly) / (1 - belly), 0, 1)) ** 0.9)
    for x, y, L, wd, a in rows:
        c, s = np.cos(a), np.sin(a)
        along, side = t * L, half * wd
        px = np.concatenate([along, along[::-1]])
        py = np.concatenate([side, -side[::-1]])
        draw.polygon(list(zip(x + c * px - s * py, y + s * px + c * py)), fill=255)
    return ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, 0.6)


def cockle(img, sheet, wet, r, buckle=2.5, grain=0.3):
    """The sheet under the gallery light from the upper left: every hill of the tooth with a lit
    and a shaded side, and the slow buckles left wherever the washes soaked the paper (`wet`)."""
    h, w = wet.shape
    soaked = ndimage.gaussian_filter(np.clip(wet, 0, 1), 40)
    bump = (noise.field((h, w), 260, r) + 0.5 * noise.field((h, w), 110, r)) * (0.25 + soaked)
    gy, gx = np.gradient(bump)
    ty, tx = np.gradient(ndimage.gaussian_filter(sheet.tooth, 0.8))
    return img * (1 + buckle * (gx + gy) + grain * (tx + ty))[..., None]
