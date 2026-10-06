"""Gouache: opaque watercolour as the Mughal studio laid it on burnished paper, with shell gold.

The ground was wasli: thin papers pasted up into a card, sized, and rubbed with an agate until
it would hold a line as fine as a hair without spreading. The painter laid each part in flat
body colour and then modelled it in hatching: a squirrel-hair brush of a few hairs drew strokes
side by side along the form, in finer and finer touches of a slightly darker or lighter colour,
until at arm's length they melt into one another. The outline came last, in a darker tone of the
same colour. Gold was ground with gum into a paint and kept in a mussel shell; laid with a brush
and rubbed with an agate it glints wherever its flakes lie flat to the light. Frames were ruled
with a pen against a straightedge, one line for each colour.

Everything here returns coverage (0..1) or colour; `color.cover` lays it on.
"""

import numpy as np
from scipy import ndimage

from . import brush, noise
from .color import lin
from .etching import hand

GOLD = lin("#c6a45b")          # shell gold at rest: warm and grainy
GREEN_GOLD = lin("#bcad6d")    # gold alloyed with silver, a little green
GLINT = lin("#fff2cf")


def body(mask, r, mottle=0.06):
    """A flat coat of body colour over `mask`: opaque, but the brush never lays it quite even,
    and where it ran thin the ground glows faintly through."""
    thin = mottle * np.clip(0.6 * noise.field(mask.shape, 22, r) + 0.4 * noise.field(mask.shape, 5, r), 0, None)
    return mask * (1 - thin)


def strokes(shape, x, y, ang, length, width, alpha, r, bend=0.05, step=0.5):
    """Short strokes of a fine brush centred at (x, y) and turned to `ang`, each about `length`
    px long and `width` px wide, laying `alpha` of colour at its middle. The brush sets down and
    lifts, so a stroke is paler at both ends; the hand bows it a little. -> coverage"""
    acc = np.zeros(shape, np.float32)
    n = len(x)
    if not n:
        return acc
    L = length * r.uniform(0.7, 1.3, n)
    k = max(3, int(np.ceil(1.3 * length / step)))
    s = np.linspace(-1, 1, k)[None]
    t = 0.5 * s * L[:, None]
    bow = bend * L[:, None] * r.normal(0, 1, (n, 1)) * (1 - s * s)
    c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
    lanes = max(1, int(round(width / 0.6)))
    off = ((np.arange(lanes) - (lanes - 1) / 2) * width / lanes)[:, None, None]
    X = x[:, None] + c * t - sn * bow - sn * off
    Y = y[:, None] + sn * t + c * bow + c * off
    V = np.broadcast_to(np.asarray(alpha, np.float32).reshape(-1, 1) * np.sin(np.pi * (s + 1) / 2) ** 0.7
                        * (L[:, None] / (k - 1)) * width * 1.5 / lanes, X.shape)
    buf, o = brush.splat(X.ravel(), Y.ravel(), V.ravel())
    brush.paste(acc, buf, o)
    return 1 - np.exp(-ndimage.gaussian_filter(acc, 0.45))


def hatch(want, angle, r, spacing=2.6, length=12.0, width=1.0, alpha=0.8, bend=0.05):
    """Strokes laid side by side wherever `want` (0..1) asks for colour, each turned to `angle`
    (radians, a number or a map) so that it runs with the form. Where the want is strong they
    close up until they touch; where it is weak they stand apart. -> coverage"""
    h, w = want.shape
    gy, gx = np.mgrid[0:h:spacing, 0:w:spacing].astype(np.float32)
    x = gx.ravel() + r.uniform(0, spacing, gx.size)
    y = gy.ravel() + r.uniform(0, spacing, gy.size)
    xi, yi = np.clip(x.astype(int), 0, w - 1), np.clip(y.astype(int), 0, h - 1)
    p = want[yi, xi]
    keep = r.random(p.size) < p
    n = int(keep.sum())
    a = (angle[yi[keep], xi[keep]] if np.ndim(angle) else np.full(n, angle)) + r.normal(0, 0.07, n)
    return strokes((h, w), x[keep], y[keep], a, length, width,
                   alpha * (0.7 + 0.3 * np.sqrt(p[keep])) * r.uniform(0.8, 1.1, n), r, bend)


def line(shape, paths, r, width=1.1, alpha=1.3, wander=0.3, tremor=0.08, step=0.5):
    """Hairlines along `paths` [(N,2), ...], drawn slowly with a brush of a few hairs: they hold
    their weight, swell a little where the hand pressed and thin where it lifted. -> coverage"""
    acc = np.zeros(shape, np.float32)
    for P in paths:
        C = hand(P, r, wander=wander, tremor=tremor, step=step)
        k = len(C)
        if k < 3:
            continue
        g = np.gradient(C, axis=0)
        g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
        press = np.clip(1 + 0.2 * noise.line1d(k, 90, r) + 0.08 * noise.line1d(k, 12, r), 0.45, 1.5)
        wd = width * press
        lanes = max(1, int(round(width * 1.5 / 0.6)))
        off = (np.arange(lanes) - (lanes - 1) / 2)[:, None] / lanes * wd[None]
        X = C[None, :, 0] - g[None, :, 1] * off
        Y = C[None, :, 1] + g[None, :, 0] * off
        V = np.broadcast_to(alpha * step * wd * 1.5 / lanes, X.shape)
        buf, o = brush.splat(X.ravel(), Y.ravel(), V.ravel())
        brush.paste(acc, buf, o)
    return 1 - np.exp(-ndimage.gaussian_filter(acc, 0.45))


def rule(into, a, b, width, r, load=1.0):
    """A line ruled with a pen along a straightedge from `a` to `b` into the coverage map `into`:
    dead straight, the same width all the way, but carrying more colour where the pen set down
    and as rough at its edges as the fibres of the paper. Ends are square, so ruled frames
    cross a little at the corners."""
    H, W = into.shape
    (ax, ay), (bx, by) = a, b
    pad = width / 2 + 3
    x0, x1 = max(0, int(min(ax, bx) - pad)), min(W, int(np.ceil(max(ax, bx) + pad)))
    y0, y1 = max(0, int(min(ay, by) - pad)), min(H, int(np.ceil(max(ay, by) + pad)))
    if x1 <= x0 or y1 <= y0:
        return into
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    L = np.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / L, (by - ay) / L
    t = (xx - ax) * ux + (yy - ay) * uy
    d = (xx - ax) * -uy + (yy - ay) * ux
    k = max(8, int(L / 4))
    f = np.clip(t / L, 0, 1) * (k - 1)
    ink = load * (0.88 + 0.06 * noise.line1d(k, 60, r) + 0.12 * np.exp(-np.arange(k) / (k * 0.15)))[f.astype(int)]
    half = width / 2 * (1 + 0.04 * noise.line1d(k, 25, r))[f.astype(int)] + 0.1 * noise.field(t.shape, 2.5, r)
    cov = noise.smoothstep(half + 0.55, half - 0.55, np.abs(d)) * noise.smoothstep(-0.6, 0.6, t) \
        * noise.smoothstep(L + 0.6, L - 0.6, t) * ink
    into[y0:y1, x0:x1] = 1 - (1 - into[y0:y1, x0:x1]) * (1 - np.clip(cov, 0, 1))
    return into


def gold(shape, r, tone=GOLD, burnish=0.5, glint=1.0):
    """Shell gold under the gallery lights. Flakes lying every which way read as a warm grainy
    ochre; where the agate pressed them flat they shine, more where the card tilts to the light;
    here and there a single flake flashes. -> (H,W,3) linear colour"""
    grain = noise.field(shape, 1.1, r)
    tilt = noise.fbm(shape, 150, r, octaves=3)
    laid = noise.field(shape, 7, r)              # where the brush left the paint a little thicker
    lum = 0.8 + 0.08 * grain + 0.05 * laid + burnish * (0.24 + 0.2 * tilt)
    flash = noise.smoothstep(2.6, 3.5, noise.field(shape, 0.8, r)) * glint
    return tone * lum[..., None] + GLINT * (0.55 * flash)[..., None]
