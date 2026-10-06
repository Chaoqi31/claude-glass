"""Etching: lines drawn with a needle through wax, bitten into copper, inked, wiped and printed.

The needle moves as freely as a pen, but what it leaves is a groove, and a groove prints
only the ink it holds: a line bitten long is deep and black, a line stopped out early stays
fine and grey. Wiping never gets the plate quite clean, so a veil of tone lies over it all,
heavier where the printer's hand passed lightly. The plate goes through the press into
damp paper so hard that its edge stays in the sheet as a ridge: the platemark.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import brush, noise
from .color import glaze, pigment, saturate

INK = pigment("#2b2521")        # warm black, a little brown where it lies thin


def hand(P, r, wander=1.2, tremor=0.3, step=1.5, corners=False):
    """A line as a hand draws it through control points P: a slow wander and a fine tremor.
    With `corners` the line goes straight from point to point and turns sharply at each."""
    P = np.asarray(P, np.float64)[:, :2]
    if corners:
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
        C = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)
    else:
        C = brush.path(P, step)
    if len(C) < 3:
        return C
    t = np.gradient(C, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    k = len(C)
    off = wander * noise.line1d(k, max(8.0, 40 / step), r) + tremor * noise.line1d(k, 3.0, r)
    return C + n * off[:, None]


def _runs(ok):
    e = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    return zip(e[::2], e[1::2])


class Needle:
    """Everything drawn on the plate, as grooves: width in px, depth 0 (barely bitten) to 1."""

    def __init__(self, shape, r, ss=3):
        self.shape, self.r, self.ss = shape, r, ss
        self.strokes = []
        self.hide = np.zeros(shape, bool)       # whatever has been drawn in front: lines stop at its edge

    def line(self, pts, width=1.4, depth=0.8, taper=0.5):
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        h, w = self.shape
        P = np.rint(pts).astype(int)
        vis = ~self.hide[P[:, 1].clip(0, h - 1), P[:, 0].clip(0, w - 1)]
        if not vis.all():
            for a, b in _runs(vis):
                if b - a >= 2:
                    self._add(pts[a:b], width, depth, taper)
            return
        self._add(pts, width, depth, taper)

    def _add(self, pts, width, depth, taper):
        s = np.linspace(0, 1, len(pts))
        # the needle sets down and lifts: thinner at both ends, never quite even between
        w = width * (taper + (1 - taper) * np.sin(np.pi * s) ** 0.35) * (1 + 0.1 * self.r.standard_normal())
        self.strokes.append((depth, pts, np.maximum(w, 0.5)))

    def stroke(self, P, width=1.4, depth=0.8, wander=1.2, tremor=0.3, taper=0.5, corners=False):
        self.line(hand(P, self.r, wander, tremor, corners=corners), width, depth, taper)

    def hatch(self, tone, angle, spacing, width=1.2, depth=0.7, length=(30, 110), bow=0.04, jitter=0.12,
              hook=0.07, lo=0.0):
        """Short parallel strokes wherever `tone` (0 light .. 1 dark) asks for them: a darker tone
        simply lets more of the candidate lines through, so the spacing closes up."""
        r = self.r
        ys, xs = np.nonzero(tone > lo)
        if not ys.size:
            return self
        by, bx = ys.min(), xs.min()
        tone = tone[by:ys.max() + 1, bx:xs.max() + 1]
        h, w = tone.shape
        ca, sa = np.cos(angle), np.sin(angle)
        diag = np.hypot(h, w) + 4
        t = np.arange(-diag / 2, diag / 2, 2.0)
        for i, o in enumerate(np.arange(-diag / 2, diag / 2, spacing)):
            thr = lo + (1 - lo) * ((i * 0.618034 + r.random() * 0.05) % 1.0)
            o = o + r.normal(0, spacing * 0.2)
            x = w / 2 - sa * o + ca * t
            y = h / 2 + ca * o + sa * t
            ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
            if not ok.any():
                continue
            tv = np.zeros_like(t)
            tv[ok] = tone[y[ok].astype(int), x[ok].astype(int)]
            on = ok & (tv > thr)
            if not on.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                j = a
                while j < b - 3:
                    n = int(r.uniform(*length) / 2.0)
                    k = min(b, j + n)
                    if k - j >= 3:
                        self._hatch_stroke(x[j:k] + bx, y[j:k] + by, tv[j:k], width, depth, bow, jitter, hook, angle)
                    j = k + int(r.uniform(0.5, 4))
        return self

    def _hatch_stroke(self, x, y, tv, width, depth, bow, jitter, hook, angle):
        r = self.r
        L = len(x)
        s = np.linspace(-1, 1, L)
        a = angle + r.normal(0, jitter)
        cx, cy = x.mean(), y.mean()
        half = (L - 1)
        u = s * half
        v = bow * half * (1 - s * s) * r.choice([-1, 1]) + 0.35 * noise.line1d(L, 6, r)
        P = np.stack([cx + np.cos(a) * u - np.sin(a) * v, cy + np.sin(a) * u + np.cos(a) * v], 1)
        if L >= 6 and r.random() < hook:            # the hand flicks back at the end of the stroke
            e = P[-1]
            d = P[-1] - P[-4]
            d /= np.hypot(*d) + 1e-9
            q = np.array([-d[1], d[0]]) * r.choice([-1, 1])
            P = np.vstack([P, e + d * 2 + q * 3, e + q * 6 - d * 1])
        k = float(tv.mean())
        self.line(P, width * (0.75 + 0.5 * k), depth * (0.65 + 0.35 * k))

    def patches(self, tone, size, angle, spacing=3.0, width=1.05, depth=0.8, spread=0.35, cross=0.55):
        """Tone laid in small patches of parallel strokes, each patch turned its own way: how an
        etcher builds foliage and rough ground. `angle` is a number or a map; `cross` is the tone
        above which a second set crosses the first."""
        r = self.r
        h, w = tone.shape
        ang = np.broadcast_to(np.asarray(angle, np.float32), tone.shape)
        ys, xs = np.nonzero(tone > 0.03)
        if not ys.size:
            return self
        gy = np.arange(ys.min(), ys.max() + 1, size * 0.8)
        gx = np.arange(xs.min(), xs.max() + 1, size * 0.8)
        for cy in gy:
            for cx in gx:
                x0, y0 = cx + r.normal(0, size * 0.3), cy + r.normal(0, size * 0.3)
                if not (0 <= x0 < w and 0 <= y0 < h):
                    continue
                t0 = float(tone[int(y0), int(x0)])
                if t0 < 0.03:
                    continue
                a = float(ang[int(y0), int(x0)]) + r.normal(0, spread)
                R = size * r.uniform(0.45, 0.75)
                for layer, lo in ((0, 0.0), (1, cross)):
                    k = (t0 - lo) / (1 - lo)
                    if k <= 0:
                        break
                    if layer:
                        a += r.choice([-1, 1]) * r.uniform(0.6, 1.2)
                    d, q = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
                    sp = spacing / max(k, 0.12)
                    for o in np.arange(-R + r.uniform(0, sp), R, sp):
                        half = np.sqrt(max(R * R - o * o, 0)) * r.uniform(0.6, 1.15)
                        if half < 3:
                            continue
                        c = np.array([x0, y0]) + q * (o + r.normal(0, sp * 0.15))
                        tt = np.linspace(-half, half, max(3, int(half)))
                        P = c + d * tt[:, None] + q * (0.06 * half * (1 - (tt / half) ** 2) * r.choice([-1, 1]))[:, None]
                        Pi = P.astype(int)
                        ok = (Pi[:, 0] >= 0) & (Pi[:, 0] < w) & (Pi[:, 1] >= 0) & (Pi[:, 1] < h)
                        ok[ok] &= tone[Pi[ok, 1], Pi[ok, 0]] > 0.03
                        for i0, i1 in _runs(ok):
                            if i1 - i0 >= 3:
                                self.line(P[i0:i1], width, depth * (0.8 + 0.2 * k), taper=0.45)
        return self

    def loops(self, path, size, width=1.2, depth=0.7, turns=1.0, wobble=0.45):
        """A line that runs along `path` in small uneven loops: leaves, bushes, curls of thatch."""
        r = self.r
        step = 0.2
        C = brush.path(np.asarray(path, np.float64), step)
        L = len(C)
        if L < 4:
            return
        tan = np.gradient(C, axis=0)
        tan /= np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-9
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        rad = size * np.exp(wobble * noise.line1d(L, 14 / step, r))
        pitch = 1.4 * rad * np.exp(0.35 * noise.line1d(L, 9 / step, r))
        phase = np.cumsum(2 * np.pi * turns * step / pitch)
        squash = 0.9 + 0.5 * noise.line1d(L, 20 / step, r).clip(-1, 1)
        P = C + nrm * (rad * np.sin(phase))[:, None] + tan * (rad * squash * np.cos(phase))[:, None]
        self.line(P[::3], width, depth, taper=0.6)

    def scallops(self, path, size, width=1.2, depth=0.7, out=1):
        """An edge drawn as a run of small arcs bulging outward, the way foliage is outlined."""
        r = self.r
        C = brush.path(np.asarray(path, np.float64), 0.5)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        pts, pos = [], 0.0
        while pos < s[-1] - 2:
            L = size * np.exp(r.normal(0, 0.4))
            e = min(s[-1], pos + L)
            k = np.linspace(pos, e, max(6, int((e - pos) / 0.7)))
            X, Y = np.interp(k, s, C[:, 0]), np.interp(k, s, C[:, 1])
            d = np.stack([np.gradient(X), np.gradient(Y)], 1)
            d /= np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9
            f = np.sin(np.linspace(0, np.pi, len(k))) ** 0.6 * (e - pos) * r.uniform(0.35, 0.6) * out
            pts.append(np.stack([X - d[:, 1] * f, Y + d[:, 0] * f], 1))
            pos = e
        if pts:
            P = np.vstack(pts)
            k = len(P)
            P = P + np.stack([noise.line1d(k, 30, r), noise.line1d(k, 30, r)], 1) * 0.6
            self.line(P, width, depth, taper=0.7)

    def grooves(self):
        """Rasterise every line, deepest last, at `ss` times the plate's resolution. -> (H,W) 0..1"""
        h, w = self.shape
        ss = self.ss
        im = Image.new("F", (w * ss, h * ss), 0.0)
        dr = ImageDraw.Draw(im)
        for depth, pts, wd in sorted(self.strokes, key=lambda s: s[0]):
            seg = np.hypot(*np.diff(pts, axis=0).T)
            s = np.concatenate([[0], np.cumsum(seg)])
            t = np.arange(0, s[-1] + 0.25, 0.5)
            X, Y, Rr = (np.interp(t, s, a) * ss for a in (pts[:, 0], pts[:, 1], wd / 2))
            for x, y, rr in zip(X, Y, Rr):
                dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=float(depth))
        g = np.asarray(im, np.float32)
        return g.reshape(h, ss, w, ss).mean(axis=(1, 3))


def print_plate(grooves, sheet, box, r, tone=0.02, wipe=0.6, ink=INK):
    """Ink the plate, wipe it, and pull it through the press onto `sheet` at box (y0, x0).

    grooves  (h,w) from Needle.grooves(); box is where the plate's top-left corner lands
    Returns linear RGB of the printed sheet and the plate mask (for the platemark).
    """
    H, W = sheet.alpha.shape
    h, w = grooves.shape
    y0, x0 = box
    # the ink each groove holds, a little grainy where the acid bit roughly
    held = grooves ** 0.8 * (0.93 + 0.14 * noise.field((h, w), 1.5, r))
    dens = saturate(2.6 * ndimage.gaussian_filter(held, 0.45), 3.0)
    # plate tone: what the wiping left, streaked with the movement of the hand
    swirl = noise.warp(noise.stretched((h, w), 40, 260, r), 60 * noise.field((h, w), 400, r), 60 * noise.field((h, w), 400, r))
    film = tone * np.clip(1 + wipe * (0.35 * swirl + 0.65 * noise.fbm((h, w), 240, r, octaves=4)), 0.2, None)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx])
    film += 0.22 * np.exp(-edge / 1.1) * np.clip(0.6 + 0.6 * noise.field((h, w), 25, r), 0, 1)   # ink caught on the bevel
    film += 0.4 * tone * np.exp(-edge / 30.0)
    # old scratches on the copper, and pits where the acid got through the ground
    scr = noise.fibers((h, w), r, count=14, length=(80, 500), width=1, curl=0.01, strength=(0.3, 1.0))
    film += 0.035 * ndimage.gaussian_filter(scr, 0.5)
    film += 0.5 * ndimage.gaussian_filter((r.random((h, w)) < 0.00008).astype(np.float32), 0.6)
    dens = np.maximum(dens, 0) + np.clip(film, 0, None)
    plate = np.zeros((H, W), np.float32)
    D = np.zeros((H, W), np.float32)
    plate[y0:y0 + h, x0:x0 + w] = 1.0
    D[y0:y0 + h, x0:x0 + w] = dens
    # the damp paper takes the ink more fully where its surface is pressed flat
    D *= 0.9 + 0.1 * sheet.tooth
    return glaze(sheet.color, D, ink), plate
