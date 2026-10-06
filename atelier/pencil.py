"""Pencil, crayon and pen: marks made with a hard point, and the machine that ruled the paper first.

A pencil point is narrow and hard. It rides over the hills of the paper's surface and leaves its
colour only where it touches: pressed lightly it marks the tops of the grain and the hollows stay
white, pressed hard, or passed over again and again, it crushes the grain and fills them. Colour
laid with a pencil is built of strokes going back and forth one way, and the strokes show: they
overlap a little here and part a little there, each tapers where the hand lifted, and each carries
the faint striations of the point. Graphite is a grey that never quite reaches black; the coloured
crayons of the 1870s were pigment in wax and clay, translucent, so whatever lies under them shows
through.

A steel pen lays its ink with the pressure of the hand: the downstroke spreads the nib and runs
broad, the upstroke is a hair, and the ink thins as the pen runs dry. Iron gall ink goes on
blue-black and over the years turns brown and fades.

Account books were ruled before they were bound, on a machine that drew a gang of pens across the
sheet at once from an inked felt: every line dead straight and the same width, a little heavier
where a pen set down, paler where its felt ran dry.

Strokes accumulate pressure into maps; `catch` turns pressure into what the paper took, and
`color.glaze` lays it down.
"""

import numpy as np
from scipy import ndimage

from . import brush, gouache, noise
from .etching import hand


def lay(into, paths, presses, width, r, step=0.5):
    """Press a point `width` px across along dense paths [(N,2)] with pressures [(N,)]. The point
    has worn a little flat, so it marks nearly evenly across and falls away at its sides, and it is
    never quite even, so every stroke carries faint striations down its length. All the strokes go
    into `into` at once."""
    n = max(2, int(np.ceil(width / 0.45)))
    u = (np.arange(n) + 0.5) / n * 2 - 1
    Xs, Ys, Vs = [], [], []
    for C, p in zip(paths, presses):
        if len(C) < 2:
            continue
        g = np.gradient(C, axis=0)
        g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
        facet = (1 - np.abs(u) ** 3) ** 0.7 * np.clip(1 + 0.15 * r.standard_normal(n), 0.5, 1.4)
        off = (u * width / 2)[:, None]
        Xs.append((C[None, :, 0] - g[None, :, 1] * off).ravel())
        Ys.append((C[None, :, 1] + g[None, :, 0] * off).ravel())
        Vs.append((facet[:, None] * p[None, :] * (width / n) * step).ravel())
    if Xs:
        buf, o = brush.splat(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Vs).astype(np.float32))
        brush.paste(into, ndimage.gaussian_filter(buf, 0.45), o)
    return into


def _resample(C, step):
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(t, s, C[:, 0]), np.interp(t, s, C[:, 1])], 1), t


def _ramp(t, L, down=4.0, up=12.0):
    """Pressure along a stroke of length L: the point lands quickly and lifts slowly."""
    return np.clip(t / down, 0, 1) ** 0.7 * np.clip((L - t) / up, 0, 1) ** 0.8


def line(into, P, width, r, pressure=0.9, wander=0.8, tremor=0.2, lift=(140, 420), corners=False, step=0.5):
    """A contour as a hand draws a long line: through control points P, in lengths of a few
    centimetres, each set down a hair beside the end of the last and pressed a little harder or
    softer."""
    C = hand(P, r, wander, tremor, step, corners)
    if len(C) < 3:
        return into
    C, s = _resample(C, step)
    paths, presses, a = [], [], 0.0
    while a < s[-1] - 2:
        b = min(s[-1], a + r.uniform(*lift))
        if s[-1] - b < 30:
            b = s[-1]
        k = (s >= a) & (s <= b)
        piece = C[k] + r.normal(0, 0.5, 2)
        t = s[k] - a
        paths.append(piece)
        p = pressure * r.uniform(0.85, 1.1) * _ramp(t, b - a, 3.0, 9.0)
        presses.append(p * (1 + 0.1 * noise.line1d(len(t), 60, r)) if len(t) > 8 else p)
        a = b - r.uniform(0, 5)
    return lay(into, paths, presses, width, r, step)


def fill(into, mask, angle, r, width=8.0, spacing=4.5, pressure=0.75, reach=(-4.0, 3.0), band=230.0, bow=0.012,
         tilt=0.012, step=0.6):
    """Colour laid over `mask` (0..1) the way a pencil fills a shape: strokes at about `angle`
    (radians, in image coordinates) `spacing` px apart, alternately one way and back, each landing
    quickly and lifting slowly. A stroke's ends overrun the shape or stop short of it by `reach` px.
    The wrist swings every stroke through the same small arc, and the angle wanders slowly from one
    stroke to the next. A shape wider than the hand's reach is filled in bands about `band` px long,
    and where two bands meet their strokes overlap."""
    ys, xs = np.nonzero(mask > 0.5)
    if not ys.size:
        return into
    y0, x0 = ys.min(), xs.min()
    m = mask[y0:ys.max() + 1, x0:xs.max() + 1] > 0.5
    h, w = m.shape
    ca, sa = np.cos(angle), np.sin(angle)
    half = np.hypot(h, w) / 2 + 2
    offs = np.arange(-half, half, spacing) + r.uniform(0, spacing)
    offs += r.normal(0, spacing * 0.12, offs.size)
    nk = len(offs)
    turn = tilt * noise.line1d(nk + 8, 6, r)[:nk]
    phase = band * (0.5 + 0.2 * noise.line1d(nk + 8, 5, r)[:nk])
    arc = bow * r.choice([-1, 1])
    t = np.arange(-half, half, 1.0)
    X = w / 2 + ca * t[None] - sa * offs[:, None]
    Y = h / 2 + sa * t[None] + ca * offs[:, None]
    ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    inside = np.zeros(X.shape, bool)
    inside[ok] = m[Y[ok].astype(int), X[ok].astype(int)]
    paths, presses = [], []
    for k in range(nk):
        e = np.flatnonzero(np.diff(np.concatenate([[0], inside[k].astype(np.int8), [0]])))
        for a, b in zip(e[::2], e[1::2]):
            if b - a < 3:
                continue
            ta, tb = t[a], t[b - 1]
            cuts = [c for c in np.arange(-half + phase[k], tb, band) if ta + 25 < c < tb - 25]
            ends = [ta] + cuts + [tb]
            for j in range(len(ends) - 1):
                u0 = ends[j] - (r.uniform(*reach) if j == 0 else r.uniform(4, 14))
                u1 = ends[j + 1] + (r.uniform(*reach) if j == len(ends) - 2 else r.uniform(4, 14))
                L = u1 - u0
                if L < 3:
                    continue
                tt = np.arange(0, L, step)
                v = arc * r.uniform(0.7, 1.3) * L * (1 - (2 * tt / L - 1) ** 2) + turn[k] * (tt - L / 2)
                if k % 2:
                    tt, v = tt[::-1], v[::-1]
                along = u0 + tt
                o = offs[k] + v
                C = np.stack([x0 + w / 2 + ca * along - sa * o, y0 + h / 2 + sa * along + ca * o], 1)
                pr = pressure * r.uniform(0.85, 1.1) * _ramp(np.arange(len(tt)) * step, L)
                if len(tt) > 12:
                    pr = pr * (1 + 0.1 * noise.line1d(len(tt), 40, r))
                paths.append(C)
                presses.append(pr)
    return lay(into, paths, presses, width, r, step)


def catch(press, sheet, grip=0.55, soft=0.1):
    """What the paper took from a pressure map: a point pressed with `press` reaches down to
    1 - grip * press of the tooth's height and marks everything standing above that."""
    reach = 1 - grip * press
    return press * noise.smoothstep(reach - soft, reach + soft, sheet.tooth)


def pen(into, P, width, r, load=1.0, slant=0.55, step=0.4):
    """A steel nib through control points P: broad on the downstroke (down the slant of the
    hand), a hairline going up and across, and paler as the pen runs dry. -> ink density"""
    C = brush.path(np.asarray(P, np.float64), step)[:, :2]
    if len(C) < 3:
        return into
    g = np.gradient(C, axis=0)
    g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
    down = np.clip(g[:, 0] * -np.sin(slant) + g[:, 1] * np.cos(slant), 0, 1)
    wd = width * (0.28 + 0.95 * down ** 1.6)
    s = np.arange(len(C)) * step
    ink = load * (0.55 + 0.45 * np.exp(-s / 900)) * (1 + 0.15 * noise.line1d(len(C), 80, r))
    n = max(2, int(np.ceil(width * 1.3 / 0.4)))
    u = ((np.arange(n) + 0.5) / n * 2 - 1)[:, None]
    X = C[None, :, 0] - g[None, :, 1] * u * wd[None] / 2
    Y = C[None, :, 1] + g[None, :, 0] * u * wd[None] / 2
    V = np.broadcast_to(ink[None] * wd[None] / n * step, X.shape)
    buf, o = brush.splat(X.ravel(), Y.ravel(), V.ravel().astype(np.float32))
    brush.paste(into, ndimage.gaussian_filter(buf, 0.5), o)
    return into


# The strokes of a running hand, none of them quite a letter: (x across, y up from the baseline, in
# x-heights), each beginning and ending on the baseline so that they join.
HAND = [
    [(0.12, -0.75), (0.32, -1.0), (0.55, -0.85), (0.62, -0.3), (0.66, 0.0)],                      # a hump
    [(0.15, -1.0), (0.17, -0.35), (0.35, 0.0), (0.6, -0.25), (0.7, -1.0), (0.72, -0.3), (0.85, 0.0)],  # a cup
    [(0.35, -0.5), (0.3, -0.75), (0.12, -0.6), (0.12, -0.2), (0.35, 0.0), (0.55, -0.05)],         # a small loop
    [(0.45, -1.6), (0.5, -2.4), (0.32, -2.5), (0.22, -1.8), (0.2, -0.6), (0.3, 0.0), (0.5, -0.1)],  # tall loop
    [(0.3, -0.9), (0.45, -1.0), (0.45, 0.2), (0.38, 1.4), (0.18, 1.5), (0.18, 1.0), (0.5, 0.2), (0.7, -0.1)],
    [(0.25, -0.6), (0.55, -1.0), (0.3, -1.0), (0.12, -0.6), (0.2, -0.1), (0.45, -0.1), (0.6, -0.65), (0.6, -0.2),
     (0.75, 0.0)],                                                                                  # an oval
    [(0.18, -0.95), (0.22, -0.3), (0.38, 0.0)],                                                      # a stroke
    [(0.3, -1.0), (0.45, -0.8), (0.5, -0.2), (0.62, 0.0)],
]
FIGURES = [
    [(0.5, -1.4), (0.15, -1.2), (0.05, -0.5), (0.3, 0.0), (0.6, -0.5), (0.55, -1.25), (0.45, -1.4)],
    [(0.25, -1.3), (0.45, -1.45), (0.35, -0.6), (0.25, 0.0)],
    [(0.1, -1.1), (0.4, -1.45), (0.65, -1.15), (0.4, -0.6), (0.05, 0.0), (0.4, -0.05), (0.7, 0.05)],
    [(0.1, -1.25), (0.45, -1.45), (0.6, -1.1), (0.3, -0.8), (0.65, -0.45), (0.45, 0.0), (0.05, -0.15)],
    [(0.05, -1.4), (0.65, -1.45), (0.4, -0.7), (0.25, 0.0)],
    [(0.6, -1.2), (0.35, -1.45), (0.1, -1.1), (0.35, -0.8), (0.62, -1.2), (0.45, -0.4), (0.3, 0.3)],
]


def scribble(x, y, length, h, r, slant=0.55, figures=False):
    """A running hand nobody can read, along the baseline y from x for about `length` px: words of
    joined strokes up to the x-height `h` and down again, now and then a loop above the line or
    below it, all leaning with the hand. With `figures`, short groups of upright marks, each made
    on its own, like sums of money. -> [(N,2) control points], one per lift of the pen"""
    out, cx = [], float(x)
    while cx < x + length:
        n = int(r.integers(1, 4) if figures else r.integers(2, 8))
        pts = [] if figures else [(cx - 0.3 * h, y - 0.1 * h)]
        for _ in range(n):
            shape = FIGURES[r.integers(len(FIGURES))] if figures else HAND[r.integers(len(HAND))]
            q = np.array(shape) * (r.uniform(0.85, 1.2), r.uniform(0.85, 1.15))
            q = (q + r.normal(0, 0.07 if figures else 0.04, q.shape)) * h + (cx, y)
            if figures:
                out.append(q)
                cx = q[:, 0].max() + 0.25 * h
            else:
                pts += list(q)
                cx = q[-1][0]
        if not figures:
            out.append(np.array(pts))
        cx += h * (r.uniform(0.8, 1.3) if figures else r.uniform(0.9, 1.8))
    lean = np.tan(0.12 if figures else slant)
    for P in out:
        P[:, 0] -= (P[:, 1] - y) * lean
        P[:, 1] += 0.05 * h * r.standard_normal(len(P))
    return out


def ruling(shape, lines, width, r, load=0.8):
    """Printed ruling: `lines` [((x0, y0), (x1, y1)), ...] drawn by one gang of pens fed from one
    felt, each pen carrying its own share of ink. -> coverage"""
    cov = np.zeros(shape, np.float32)
    for a, b in lines:
        gouache.rule(cov, a, b, width * r.uniform(0.9, 1.1), r, load=load * r.uniform(0.8, 1.1))
    return cov


if __name__ == "__main__":      # a fill comes out even: no stripes where neighbouring strokes pile up or part
    m = np.zeros((300, 400), np.float32)
    m[40:260, 40:360] = 1
    p = fill(np.zeros_like(m), m, -0.9, noise.rng(0), width=8, spacing=4.4, pressure=0.8)[80:220, 80:320]
    lo, mid = np.percentile(p, [5, 50])
    assert lo > 0.5 * mid, (lo, mid)
    print("fill even:", round(float(lo), 2), round(float(mid), 2))
