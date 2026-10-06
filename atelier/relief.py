"""Relief printing: blocks are carved, inked, and pressed into paper.

A block is a raster of the uncut surface (1) and what the gouge took away (0). Ink sits
only on the surface; the paper takes it where its own grain meets the block, so solids
come out a little salted. Each colour is a separate block, and none of them registers
perfectly with the others.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import noise


def cuts(shape, strokes, ss=2):
    """Rasterise gouge cuts. Each cut is (points (N,2), half-widths (N,)); returns 1 where cut."""
    h, w = shape
    im = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    for pts, hw in strokes:
        pts = np.asarray(pts, np.float64) * ss
        hw = np.asarray(hw, np.float64) * ss
        if len(pts) < 2:
            continue
        t = np.gradient(pts, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        left, right = pts + nrm * hw[:, None], pts - nrm * hw[:, None]
        poly = np.vstack([left, right[::-1]])
        d.polygon([tuple(p) for p in poly], fill=255)
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def vcut(path, width, r, taper=0.3, wobble=0.12):
    """A V-gouge cut along `path`: it bites in, runs, and lifts out, never quite steady."""
    path = np.asarray(path, np.float64)
    n = len(path)
    t = np.linspace(0, 1, n)
    body = noise.smoothstep(0, taper, t) * noise.smoothstep(1, 1 - taper, t) ** 0.8
    jitter = 1 + wobble * np.sin(t * r.uniform(6, 14) + r.uniform(0, 6))
    return path, np.maximum(width * body * jitter, 0.0)


def woodgrain(shape, r, along=0.0, scale=1.0):
    """The grain of a cherry block: long streaks and the ghosts of growth rings."""
    h, w = shape
    g = noise.stretched((w, h), 2.5 * scale, 160 * scale, r).T[:h, :w] if along == 0.0 else \
        noise.stretched(shape, 2.5 * scale, 160 * scale, r)
    yy = np.arange(h, dtype=np.float32)[:, None] if along == 0.0 else np.arange(w, dtype=np.float32)[None, :]
    rings = np.sin(yy / (9 * scale) + 3.0 * noise.fbm(shape, 300 * scale, r, octaves=3))
    return 0.7 * g + 0.3 * rings


def bokashi(shape, axis, start, end, lo=0.0, hi=1.0):
    """A wiped gradient on the block: full ink at `start`, none at `end` (fractions of the axis)."""
    h, w = shape
    t = (np.arange(h if axis == 0 else w, dtype=np.float32) / (h if axis == 0 else w) - start) / (end - start)
    ramp = lo + (hi - lo) * (1 - noise.smoothstep(0, 1, t))
    return ramp[:, None] * np.ones((1, w), np.float32) if axis == 0 else ramp[None, :] * np.ones((h, 1), np.float32)


def ink_film(block, r, roller=0.06, squash=0.25, grain=None, grain_strength=0.0):
    """How much ink the block carries: roller streaks, a little extra where ink squeezes at edges."""
    h, w = block.shape
    streak = noise.stretched((w, h), 2.0, 60.0, r).T[:h, :w] if roller else 0.0
    film = 1.0 + roller * streak + 0.05 * noise.fbm(block.shape, 120, r, octaves=4)
    rim = np.clip(ndimage.gaussian_filter(block, 2.0) - ndimage.gaussian_filter(block, 5.0), 0, None)
    film = film + squash * rim * 4
    if grain is not None:
        film = film * (1 + grain_strength * grain)
    return film


def pull(block, film, sheet, r, pressure=1.0, chatter=None, shift=(0.0, 0.0), turn=0.0):
    """Press paper onto the inked block. Returns ink density on the sheet.

    chatter  ink caught by shallow gouge marks in cleared areas (a map), or None
    shift, turn  misregistration of this block against the sheet (px, radians)
    """
    if shift != (0.0, 0.0) or turn:
        c = np.array(block.shape) / 2
        m = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        off = c - m @ c - np.array(shift[::-1])
        block = ndimage.affine_transform(block, m, off, order=1)
        film = ndimage.affine_transform(film, m, off, order=1, cval=1.0)
    # the sheet's own grain decides where it meets the ink; light pressure misses the valleys
    thr = 0.18 + 0.35 * (1 - pressure)
    catch = noise.smoothstep(thr - 0.12, thr + 0.12, sheet.tooth + 0.08 * noise.field(block.shape, 1.2, r))
    dens = block * film * (0.78 + 0.22 * catch)
    if chatter is not None:
        dens += (1 - block) * chatter * catch
    return dens


def emboss(img, blocks, light=(-0.6, -0.8), depth=0.018):
    """Where a block pressed hardest the paper sank a little; show it in raking light."""
    hgt = ndimage.gaussian_filter(-sum(blocks), 1.6)
    gy, gx = np.gradient(hgt)
    shade = 1 + depth * 12 * (gx * light[0] + gy * light[1])
    return img * shade[..., None]
