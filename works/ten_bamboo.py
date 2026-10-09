"""Loquats and a White-eye. Colour woodblock print from assembled blocks, with blind embossing.

A loquat bough comes into an album leaf from the right in early summer. Its leaves spring in a loose
rosette from the last stretch of the bough round a cluster of fruit, a young shoot hangs below, and a
white-eye has come down onto the bough. The leaf is printed the way Hu Zhengyan's workshop in Nanjing
printed the Ten Bamboo Studio manual in 1633, from assembled blocks (douban), and there is no keyline.
Every area of one colour was cut as a small block of its own after a brush drawing, brushed with water
colour and wiped on the wood so that it shades within its shape, and the blocks were printed one after
another onto the same damp sheet, never quite in register. Each leaf is the shape of one stroke of a
loaded brush, set down at the stalk and lifted at the tip: a grey-green ground wiped darker toward the
stalk, then a deep blue-green wiped dark over one half, along one side or at the stalk, and the midrib
and a few veins from a block of their own in the darkest green; a leaf turned over shows the rusty felt
of its underside. The fruit are gold, or straw and still green at the shoulder where they are not yet
ripe, wiped orange on the side away from the light, the ripest flushed warmer, and each dried calyx is a
small dark touch of the brush. The white-eye's wing and back are cut as rows of small scale-shaped
feathers, its long wing and tail feathers as tapering strokes edged with fine lines. Thin colour misses
the valleys of the paper, so the pale end of every wipe comes out speckled, and the grain of the wood
shows faintly in the colour. The other veins of the leaves, the thin ring round the bird's eye and the
white feathers of its belly were pressed into the paper from an uninked block and show only as relief.
The title, 一樹金, a whole tree of gold, is the end of a line of Dai Fugu's on a garden at the start of
summer.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from atelier import ink, lettering, noise, paper, plate
from atelier.color import glaze, pigment

TITLE = "Loquats and a White-eye"
DATE = "2026"
MEDIUM = ("Colour woodblock print from assembled blocks (douban) in 131 impressions, with blind embossing "
          "(gonghua), on xuan paper")
AFTER = ("Hu Zhengyan and the Ten Bamboo Studio Manual of Calligraphy and Painting (Shizhuzhai shuhua pu), "
         "Nanjing, 1633")
ROOM = "The Workshop"
YEAR = 1633
PLACE = "Nanjing"
REGION = "East Asia"
NOTE = ("No keyline: every leaf, fruit and feather is a small block of its own, cut after a brush drawing and "
        "wiped so that it shades within its shape. Most of the leaves' veins and the bird's eye-ring and belly are "
        "only pressed into the paper.")

SH, SW = 2600, 2000
SS = 3                                   # shapes are cut at three times the size they print
HEADING, SIGN = "一樹金", "克勞德筆"
INSCRIPTION = lettering.CUT / "ten_bamboo_title.png"
XINGKAI = next(Path("/System/Library/AssetsV2").glob("com_apple_MobileAsset_Font*/*/AssetData/Xingkai.ttc"), None)

LIGHTS = [pigment(c) for c in ("#9cb88c", "#a7bc86", "#93b496")]            # grounds of the leaves
DEEPS = [pigment(c) for c in ("#285d58", "#3a604b", "#2b5850")]             # deep blue-green, grey-green
VEINC, NEW, NEW2 = pigment("#1e3a33"), pigment("#c4ce8e"), pigment("#7fa35e")
FUR, SCORCH, FELT, RUST, VEIN = (pigment(c) for c in ("#c58456", "#b98a4c", "#d9cfa3", "#b98250", "#8c5632"))
GOLD, STRAW, ORANGE, BLUSH = pigment("#f5c54e"), pigment("#eed88c"), pigment("#ec9236"), pigment("#dc6832")
SHOULDER, CALYX, RUSSET = pigment("#aab85e"), pigment("#3d2c22"), pigment("#a8673a")
STALK, BARK, SUMI = pigment("#a76a3d"), pigment("#a89886"), pigment("#3e3530")
KEY = pigment("#2c2826")


# ---------------------------------------------------------------- geometry: lines as the knife follows them

def bez(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def resample(p, step):
    s0 = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.linspace(0, s0[-1], max(3, int(s0[-1] / step) + 1))
    return np.c_[np.interp(s, s0, p[:, 0]), np.interp(s, s0, p[:, 1])]


def normals(p):
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.c_[-t[:, 1], t[:, 0]]


def knife(p, r, amp=0.8, nicks=0.004):
    """An outline as the knife cut it: resampled every 2 px and never quite steady, the hand drifting a
    little off the drawn line and the wood chipping out here and there at the edge."""
    q = resample(p, 2.0)
    n = len(q)
    off = amp * (noise.line1d(n, 16, r) + 0.35 * noise.line1d(n, 3, r))
    for c in r.integers(0, n, r.poisson(n * nicks)):
        k = np.arange(n)
        off -= r.uniform(0.6, 1.6) * np.exp(-((k - c) / r.uniform(1.0, 2.5)) ** 2)
    return q + normals(q) * off[:, None]


def ribbon(path, hw):
    """A band of half-width hw along a path, as a closed outline."""
    nrm = normals(path) * np.broadcast_to(np.asarray(hw, np.float64), len(path))[:, None]
    return np.vstack([path + nrm, (path - nrm)[::-1]])


def brush(path, w, r, entry=1.0, lift=0.4, swell=0.12, amp=0.4):
    """A shape cut after one stroke of the brush along `path`, `w` its half-width: the tip set down in a
    rounded touch `entry` half-widths long and pressed a little just after it, then drawn out and lifted
    over the last `lift` of the way, so that it narrows to a point. Returns the outline, the path, and the
    half-width along it."""
    p = resample(np.asarray(path, np.float64), 1.5)
    n = len(p)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    e = min(entry * w, 0.3 * s[-1])
    head = np.where(s < e, np.sqrt(np.clip(1 - (1 - s / max(e, 1e-6)) ** 2, 0, 1)), 1.0)
    tail = np.clip((1 - s / s[-1]) / lift, 0, 1) ** 0.8
    press = 1 + swell * np.exp(-((s - e - w) / (1.5 * w + 1)) ** 2)
    prof = w * head * tail * press * (1 + 0.06 * noise.line1d(n, 20, r))
    return knife(ribbon(p, prof), r, amp, 0.002), p, prof


def oval(c, a, b, ang=0.0, n=120, wob=None):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = 1.0 if wob is None else wob(th)
    x, y = a * np.cos(th) * rr, b * np.sin(th) * rr
    return np.c_[c[0] + x * np.cos(ang) - y * np.sin(ang), c[1] + x * np.sin(ang) + y * np.cos(ang)]


def inside(poly, pts):
    """Which of the points lie inside the polygon."""
    x, y = pts[:, :1], pts[:, 1:]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    return (((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)).sum(1) % 2 == 1


# ---------------------------------------------------------------- the wood and the wiping

def woodgrain(shape, ang, r, period=12.0):
    """The face of a small block as the colour finds it: fine streaks along the grain and the ghosts of
    growth lines crossing the block, the grain running whichever way the cutter turned the piece."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    n = int(np.hypot(h, w)) + 80
    along, across = xx * c + yy * s + n / 2, -xx * s + yy * c + n / 2
    q = across + 5 * np.interp(along, np.arange(n), noise.line1d(n, 160, r))
    fine = np.interp(q, np.arange(n), noise.line1d(n, 2.2, r)) * noise.smoothstep(-0.8, 1.0, noise.field(shape, 50, r))
    rings = np.cumsum(np.clip(1 + 0.5 * noise.line1d(n, 40, r), 0.3, None)) / period
    ring = np.exp(-((np.interp(q, np.arange(n), rings) % 1 - 0.5) / 0.14) ** 2)
    ring = (ring - ring.mean()) / (ring.std() + 1e-6) * noise.smoothstep(-0.6, 1.0, noise.field(shape, 70, r))
    g = 0.6 * fine + 0.25 * ring
    return (g / (g.std() + 1e-6)).astype(np.float32)


def uneven(shape, r, ang, amp=0.07, scale=24.0):
    """How unevenly a wipe goes: the brush or the cloth leaves the colour lumpy, and dragged along
    `ang` it leaves faint streaks."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = int(np.hypot(h, w)) + 40
    across = -xx * np.sin(ang) + yy * np.cos(ang) + n / 2
    streak = np.interp(across, np.arange(n), noise.line1d(n, 4.0, r))
    return amp * (noise.field(shape, scale, r) + 0.45 * noise.field(shape, scale / 3, r) + 0.2 * streak)


# ---------------------------------------------------------------- blocks

class Block:
    """One small block: the shapes cut on it (fill 1 left standing, 0 cut away), its colour, and how the
    colour is brushed and wiped on it, a function of sheet position. Its grain runs along `turn`."""

    def __init__(self, k, shapes, amount, turn=0.0, **press):
        self.k, self.shapes, self.amount, self.turn, self.press = k, shapes, amount, turn, press

    def cut(self, pad=14):
        pts = np.vstack([p for p, _ in self.shapes])
        x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + pad, (SW, SH))
        im = Image.new("L", ((x1 - x0) * SS, (y1 - y0) * SS), 0)
        d = ImageDraw.Draw(im)
        for p, fill in self.shapes:
            d.polygon([tuple(v) for v in (p - (x0, y0)) * SS], fill=int(255 * fill))
        self.y0, self.x0 = y0, x0
        self.m = np.asarray(im.resize((x1 - x0, y1 - y0), Image.BOX), np.float32) / 255
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        self.amt = np.asarray(self.amount(xx, yy), np.float32) * np.ones_like(self.m)
        return self

    @property
    def at(self):
        return slice(self.y0, self.y0 + self.m.shape[0]), slice(self.x0, self.x0 + self.m.shape[1])


def footprint(polys):
    """The ground an object covers, cut on its own: what stands in front is cleared from every block behind."""
    return Block(None, [(p, 1) for p in polys], lambda x, y: 1.0).cut(pad=4)


def press(img, sheet, b, r, off=1.8, grain=0.11, salt=0.8, wet=0.35, spread=1.8, squash=0.2):
    """One impression. The block is brushed with water colour and wiped, the damp sheet laid on it and
    rubbed. Thin colour reaches only the tops of the fibres, so the pale end of every wipe comes out
    speckled; the grain of the block shows in the colour; and no block lands quite where the last did."""
    at = b.at
    sh = paper.Sheet(sheet.color[at], sheet.tooth[at], sheet.fiber[at], sheet.alpha[at])
    shift = r.normal(0, off, 2)
    m = ndimage.shift(b.m, shift, order=1)
    amt = ndimage.shift(b.amt, shift, order=1, mode="nearest")
    film = amt * (1 + grain * woodgrain(m.shape, b.turn, r)) * (1 + 0.05 * noise.fbm(m.shape, 40, r, octaves=3))
    rim = np.clip(ndimage.gaussian_filter(m, 1.5) - ndimage.gaussian_filter(m, 4.0), 0, None)
    film = film + squash * 3 * rim * amt
    thin = np.clip(1 - film / 0.95, 0, 1)
    thr = 0.16 + 0.42 * thin
    catch = noise.smoothstep(thr - 0.12, thr + 0.12, sh.tooth + 0.08 * noise.field(m.shape, 1.2, r))
    dens = m * film * (1 - salt * thin * (1 - catch))
    dens = ink.bleed(dens, noise.smoothstep(0.01, 0.1, dens), sh, wet=wet, spread=spread, halo=0.05, edge=0.1,
                     soften=0.6, mottle=0.15, whiskers=0.12, seed=int(r.integers(1 << 30)))
    img[at] = glaze(img[at], dens, b.k)
    return shift


def frame(mid, X, Y, prof, other=None):
    """Shape coordinates of sheet points: how far along the line `mid` (0 to 1) and how far out toward the
    edge (-1 to 1), from the nearest point of it; `other` is the half-width on the far side, if it differs."""
    i = cKDTree(mid).query(np.c_[X.ravel(), Y.ravel()])[1].reshape(X.shape)
    nrm = normals(mid)
    lat = (X - mid[i, 0]) * nrm[i, 0] + (Y - mid[i, 1]) * nrm[i, 1]
    w = prof[i] if other is None else np.where(lat > 0, prof[i], other[i])
    return i / (len(mid) - 1), lat / np.maximum(w, 2.0)


# ---------------------------------------------------------------- the leaf

def leaf(r, base, ang, L, wmax, bend=0.0, kind="top", fold=1.0, tone=1.0, wave=0.0, scorch=0.0):
    """A loquat leaf as the print renders it after the brush: one long stroke that leaves the twig on a
    short stalk, widens slowly to past its middle and comes in quickly to an acute tip, a little curved,
    its edge never ruled, with a few shallow teeth where the brush wavered. A ground for the whole leaf,
    wiped darker toward the stalk; a deep green for one half or the whole, wiped dark at the margin, at the
    midrib, at the stalk or along one side; the midrib and a few veins in the darkest green from a block of
    their own. The other veins are left on the uninked block, which presses them up out of the paper."""
    n = 260
    t = np.linspace(0, 1, n)
    mid = (base + np.outer(t * L, unit(ang))
           + np.outer(L * (bend * t ** 2 + wave * np.sin(np.pi * t)), unit(ang + np.pi / 2)))
    nrm = normals(mid)
    st, sharp = r.uniform(0.04, 0.07), r.uniform(0.7, 0.9)
    s = np.clip((t - st) / (1 - st), 0, 1)

    def half(w, pk):
        """One side, stalk to tip: a slow wedge out to its widest past the middle, a quicker curve in."""
        rise = np.sin(0.5 * np.pi * np.clip(s / pk, 0, 1)) ** r.uniform(1.3, 1.8)
        fall = np.cos(0.5 * np.pi * np.clip((s - pk) / (1 - pk), 0, 1)) ** sharp
        width = w * np.where(s < pk, rise, fall) * (1 + 0.05 * noise.line1d(n, 60, r) + 0.02 * noise.line1d(n, 12, r))
        return np.maximum(width, np.where(t < 0.5, 3.0, 0.0))
    pk = r.uniform(0.55, 0.68)
    left, right = half(wmax, pk), half(wmax * fold, pk + r.normal(0, 0.03))
    teeth = r.random() < 0.6

    def margin(side, width):
        """The edge as the knife followed the brush: a few shallow teeth, never evenly spaced."""
        bite = np.zeros(n)
        for c in r.uniform(0.4, 0.92, r.poisson(3) if teeth else 0):
            ph = np.clip((t - c) / r.uniform(0.025, 0.045) + 1, 0, 1)
            bite += np.where((ph > 0) & (ph < 1), r.uniform(2.0, 4.5) * (1 - ph ** 1.6), 0)
        return mid + side * nrm * np.maximum(width - bite, 0)[:, None]
    lm, rm = margin(1, left), margin(-1, right)
    whole = knife(np.vstack([lm, rm[::-1]]), r)
    halves = [knife(np.vstack([lm, mid[::-1]]), r), knife(np.vstack([rm, mid[::-1]]), r)]
    nv = int(L / r.uniform(52, 64))
    lat = []
    for side, w in ((1, left), (-1, right)):
        for tv in np.linspace(0.16, 0.84, nv) + r.normal(0, 0.03, nv):
            i0 = int(tv * (n - 1))
            i1 = min(int((tv + r.uniform(0.07, 0.11)) * (n - 1)), n - 4)
            im = (i0 + i1) // 2
            bow = mid[im] + side * nrm[im] * w[im] * r.uniform(0.38, 0.5)
            p0, p1 = mid[i0], mid[i1] + side * nrm[i1] * w[i1] * r.uniform(0.7, 0.86)
            lat.append(bez(p0, (p0 + bow) / 2, bow, p1, 20))
    rib = brush(mid[3:int(0.93 * n)], r.uniform(1.8, 2.3), r, entry=0, lift=0.9, swell=0, amp=0.25)[0]
    raised = [knife(ribbon(p, np.linspace(1.5, 0.5, 20) * r.uniform(0.75, 1.2)), r, 0.3, 0) for p in lat]
    g = ang + r.normal(0, 0.15)
    at = lambda X, Y: frame(mid, X, Y, left, right)

    if kind == "under":                          # turned over: the rusty felt, the veins standing up from it
        def felt(X, Y):
            tt, ss = at(X, Y)
            return np.clip(0.75 + 0.2 * (1 - tt) + uneven(X.shape, r, g, 0.07), 0, None)

        def rust(X, Y):
            tt, ss = at(X, Y)
            u = (0.75 * (1 - noise.smoothstep(0.0, 0.55, tt + uneven(X.shape, r, g, 0.08, 40)))
                 + 0.35 * (1 - np.abs(ss)) ** 3)
            return np.clip(0.9 * u + uneven(X.shape, r, g, 0.08), 0, None)
        veins = [rib] + [brush(p, r.uniform(0.7, 1.1), r, entry=0, lift=0.85, swell=0, amp=0.2)[0]
                         for p in lat if r.random() < 0.55]
        blocks = [Block(FELT, [(whole, 1)], felt, g, salt=1.0, grain=0.07),
                  Block(RUST, [(whole, 1)], rust, g + 0.4, off=2.4),
                  Block(VEIN, [(v, 1) for v in veins], lambda X, Y: 0.55 + uneven(X.shape, r, g, 0.08), g, wet=0.2)]
        ridge = knife(ribbon(mid[8:234], np.linspace(2.2, 0.6, 226)), r, 0.25, 0)
        return [whole], blocks, raised + [ridge]

    light, dark = (LIGHTS[r.integers(3)], DEEPS[r.integers(3)]) if kind == "top" else (NEW, NEW2)
    mode, shade, inward = r.choice(3, p=[0.5, 0.25, 0.25]), r.choice([1, -1]), r.random() < 0.35

    def ground(X, Y):
        tt, ss = at(X, Y)
        stalkward = 1 - noise.smoothstep(0.0, 0.6, tt + uneven(X.shape, r, g, 0.08, 40))
        return np.clip(tone * (0.68 + 0.35 * stalkward) + uneven(X.shape, r, g, 0.06), 0, None)

    def deep(side):
        def amt(X, Y):
            tt, ss = at(X, Y)
            u = uneven(X.shape, r, g + 0.3, 0.07, 40)
            fade = 1 - 0.5 * noise.smoothstep(0.3, 1.0, tt + u)
            if mode == 0:                        # the half in shadow deep, the other light
                q = np.abs(ss) + u
                across = 1 - noise.smoothstep(0.1, 0.9, q) if inward else noise.smoothstep(0.05, 0.9, q)
                return np.clip((1.0 if side == shade else 0.38) * tone * (0.15 + 0.95 * across) * fade, 0, None)
            if mode == 1:                        # dark at the stalk, wiped out toward the tip
                return np.clip(tone * 1.05 * (1 - noise.smoothstep(0.05, 0.8, tt + u)), 0, None)
            return np.clip(tone * noise.smoothstep(-0.7, 0.95, shade * ss + u) * fade, 0, None)
        return amt
    blocks = [Block(light, [(whole, 1)], ground, g)]
    if mode == 0:
        blocks += [Block(dark, [(h, 1)], deep(sd), g + r.normal(0, 0.3), off=2.2) for h, sd in zip(halves, (1, -1))]
    else:
        blocks.append(Block(dark, [(whole, 1)], deep(0), g + r.normal(0, 0.3), off=2.2))
    keep = r.choice([0.0, 0.15, 0.35])
    printed = [brush(p, r.uniform(0.9, 1.3), r, entry=0, lift=0.85, swell=0, amp=0.2)[0] for p in lat if r.random() < keep]
    strength = 0.75 if kind == "top" else 0.35
    blocks.append(Block(VEINC, [(rib, 1)] + [(v, 1) for v in printed],
                        lambda X, Y: np.clip(strength + uneven(X.shape, r, g, 0.1, 30), 0, None), g, wet=0.2, off=2.0))
    if kind == "new":                            # a young leaf still felted rusty along its edge
        def fur(X, Y):
            tt, ss = at(X, Y)
            return (0.45 * noise.smoothstep(0.7, 1.0, np.abs(ss) + uneven(X.shape, r, g, 0.1))
                    + 0.3 * (1 - noise.smoothstep(0, 0.3, tt)))
        blocks.append(Block(FUR, [(whole, 1)], fur, g))
    if scorch:                                   # an old leaf, browned at the tip
        def brown(X, Y):
            tt, ss = at(X, Y)
            return scorch * noise.smoothstep(0.62, 0.85, tt + uneven(X.shape, r, g, 0.1, 30))
        blocks.append(Block(SCORCH, [(whole, 1)], brown, g))
    return [whole], blocks, raised


# ---------------------------------------------------------------- the fruit, its stalks, the bough

def fruit(r, c, rad, ang, ripe=0.7, face=0.0, kind="ripe"):
    """A loquat, `ang` pointing from its middle to the stalk, a little narrower at the stalk end. A ground
    of gold, or of straw where it is not yet ripe, wiped thin where the light falls; orange wiped onto the
    side turned from the light, as much as the fruit is ripe; green at the shoulder of the unripe ones and a
    warmer blush on the ripest; the dried calyx at the far end a single small dark touch."""
    c = np.asarray(c, np.float64)
    ph = r.uniform(0, 6)
    wob = lambda th: (1 + 0.05 * np.cos(2 * th)) * (1 - 0.06 * np.cos(th)) * (1 + 0.015 * np.sin(5 * th + ph))
    out = knife(oval(c, rad * 1.12, rad, ang, 180, wob), r, 0.6)
    lit = c + 0.45 * rad * unit(np.radians(-130) + r.normal(0, 0.35))
    stem = c + unit(ang) * rad * 0.9
    g = r.uniform(0, np.pi)
    far = lambda X, Y, p: np.hypot(X - p[0], Y - p[1]) / rad

    def ground(X, Y):
        return np.clip(0.5 + 0.4 * noise.smoothstep(0.1, 1.05, far(X, Y, lit)) + uneven(X.shape, r, g, 0.06), 0, None)

    def orange(X, Y):
        return np.clip(ripe * noise.smoothstep(0.45, 1.25, far(X, Y, lit) + uneven(X.shape, r, g + 1, 0.14, 14)), 0, None)
    blocks = [Block(STRAW if kind == "pale" else GOLD, [(out, 1)], ground, g, salt=0.55, grain=0.07),
              Block(ORANGE, [(out, 1)], orange, g + 0.7, off=2.6, salt=0.55, grain=0.07)]
    if kind == "pale":
        def green(X, Y):
            return np.clip(0.8 * noise.smoothstep(0.95, 0.2, far(X, Y, stem) + uneven(X.shape, r, g, 0.12, 16)), 0, None)
        blocks.append(Block(SHOULDER, [(out, 1)], green, g + 0.3, off=2.2))
    if kind == "blush":
        warm = c + 0.55 * rad * unit(np.radians(-130) + r.normal(0, 0.8))

        def blush(X, Y):
            return np.clip(0.65 * noise.smoothstep(1.0, 0.25, far(X, Y, warm) + uneven(X.shape, r, g, 0.12, 16)), 0, None)
        blocks.append(Block(BLUSH, [(out, 1)], blush, g + 1.2, off=2.4))
    if face > 0.15 or r.random() < 0.4:          # the calyx, unless it is turned away: a dab of the brush,
        tip = c - unit(ang) * rad * (0.86 - 0.5 * face)          # often with a smaller one set against it
        a = r.uniform(0, 2 * np.pi)
        d = unit(a) * rad * r.uniform(0.07, 0.1)
        dabs = [bez(tip - d, tip - 0.3 * d + r.normal(0, 1, 2), tip + 0.3 * d, tip + d, 12)]
        if r.random() < 0.65:
            e = tip + d * r.choice([-1, 1]) * 0.8
            d2 = unit(a + r.choice([-1, 1]) * r.uniform(0.9, 1.6)) * rad * r.uniform(0.05, 0.08)
            dabs.append(bez(e, e + 0.33 * d2, e + 0.66 * d2, e + d2, 8))
        touch = [brush(p, rad * r.uniform(0.03, 0.042), r, entry=1.0, lift=0.6, swell=0.2, amp=0.25)[0] for p in dabs]
        blocks.append(Block(CALYX, [(k, 1) for k in touch], lambda X, Y: 0.62 + uneven(X.shape, r, g, 0.1), g, wet=0.3))
    blocks.append(Block(RUSSET, [(out, 1)], lambda X, Y: 0.5 * noise.smoothstep(0.45, 0.1, far(X, Y, stem)), g))
    return [out], blocks, stem


def stalk(r, path, w0, w1):
    """A fruit stalk, thick and felted rusty brown."""
    p = resample(path, 3.0)
    out = knife(ribbon(p, np.linspace(w0, w1, len(p))), r, 0.35, 0)
    g = np.arctan2(*(p[-1] - p[0])[::-1])
    return out, Block(STALK, [(out, 1)], lambda X, Y: 0.8 + uneven(X.shape, r, g, 0.1, 12), g)


def bough(r, path, w0, w1, young=0.7):
    """The bough: a pale grey-brown block for the whole of it, and over it sumi wiped along its under side
    and dragged half dry, with a few fissures of the bark left standing on it; toward the tip the new wood is
    felted rusty like the stalks."""
    p = resample(path, 3.0)
    n = len(p)
    hw = np.linspace(w0, w1, n) * (1 + 0.07 * noise.line1d(n, 25, r))
    out = knife(ribbon(p, hw), r, 0.6)
    g = np.arctan2(*(p[-1] - p[0])[::-1])
    along = noise.line1d(n, 14, r)

    def dark(X, Y):
        tt, ss = frame(p, X, Y, hw)
        k = np.clip(tt * (n - 1), 0, n - 1).astype(int)
        q = ss + 0.3 * along[k] + uneven(X.shape, r, g, 0.12, 10)
        dry = noise.smoothstep(-0.6, 0.6, along[k] + 0.7 * noise.field(X.shape, 7, r))
        return np.clip(0.85 * noise.smoothstep(-0.25, 0.9, q) * (0.45 + 0.55 * dry)
                       * (1 - 0.6 * noise.smoothstep(young - 0.1, young + 0.15, tt)), 0, None)
    lines = []
    nrm = normals(p)
    for _ in range(int(n / 9)):
        i0 = int(r.integers(0, n - 12))
        i1 = min(n - 1, i0 + int(r.integers(8, 40)))
        o = r.uniform(-0.75, 0.6)
        seg = p[i0:i1] + nrm[i0:i1] * (hw[i0:i1] * (o + 0.08 * np.sin(np.linspace(0, 3, i1 - i0))))[:, None]
        if len(seg) > 3:
            lines.append(knife(ribbon(seg, np.sin(np.linspace(0.15, 3.0, len(seg))) * r.uniform(0.7, 1.4)), r, 0.2, 0))

    def fur(X, Y):
        tt = frame(p, X, Y, hw)[0] + uneven(X.shape, r, g, 0.08)
        return 0.8 * noise.smoothstep(young - 0.15, young + 0.2, tt)
    return [out], [Block(BARK, [(out, 1)], lambda X, Y: 0.72 + uneven(X.shape, r, g, 0.06), g),
                   Block(SUMI, [(out, 1)], dark, g, off=2.4, wet=0.25),
                   Block(SUMI, [(q, 1) for q in lines], lambda X, Y: 0.8, g, wet=0.2),
                   Block(FUR, [(out, 1)], fur, g)], p, hw


# ---------------------------------------------------------------- the white-eye

def spline(pts, n=20, closed=True):
    """A Catmull-Rom curve through the points."""
    P = np.asarray(pts, np.float64)
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([P[0], P, P[-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    seg = [0.5 * (2 * P[i] + (P[i + 1] - P[i - 1]) * t + (2 * P[i - 1] - 5 * P[i] + 4 * P[i + 1] - P[i + 2]) * t ** 2
                  + (3 * P[i] - P[i - 1] - 3 * P[i + 1] + P[i + 2]) * t ** 3) for i in range(1, len(P) - 2)]
    return np.vstack(seg + ([] if closed else [P[-2:-1]]))


# the bird drawn facing right in its own units, perched at PERCH; it is turned to face left on the sheet
BODY = [(21, -11), (18, -21), (10, -27.5), (0, -29), (-10, -26.5), (-21, -20.5), (-35, -12.5), (-50, -3),
        (-64, 7), (-76, 18), (-82, 28), (-78, 37), (-66, 44), (-48, 51), (-30, 56), (-14, 53), (-1, 46),
        (9, 34), (16, 20), (20, 7), (21.5, -3)]
PARTING = [(21.5, -3), (13, -1), (4, 4), (-10, 10), (-26, 17), (-44, 25), (-62, 33), (-80, 32)]
WING = [(-3, -13), (-16, -15), (-32, -9), (-48, 1), (-64, 13), (-80, 27), (-96, 39), (-110, 48), (-97, 50),
        (-78, 47), (-58, 41), (-38, 33), (-20, 23), (-8, 11), (-2, -1)]
COVERTS = [(-3, -13), (-16, -15), (-30, -9.5), (-42, -1), (-48, 10), (-44, 24), (-32, 29), (-20, 22), (-9, 10),
           (-2, -1)]
BEND = [(-3, -13), (-2, -1), (-9, 10), (-20, 22), (-31, 29)]          # the front edge of the folded wing
SECONDARIES = [((-28, 26), (-86, 47), 7.5), ((-32, 17), (-92, 43), 8), ((-33, 8), (-95, 37), 8),
               ((-31, -1), (-89, 29), 8)]
PRIMARIES = [((-58, 38), (-104, 50), 6), ((-56, 30), (-110, 49), 6.5), ((-54, 22), (-114, 46), 7)]
TAILS = [((-80, 38), (-127, 88), 7.5), ((-80, 32), (-131, 85.5), 8), ((-79, 26), (-134, 82.5), 8),
         ((-79, 21), (-136, 79), 7.5)]
BILL = [(19, -10.6), (26, -11.4), (33, -10.6), (40, -8.6)]
EYE = np.array([9.5, -15.5])
PERCH = np.array([-30.0, 68.0])
YELLOW, OLIVE, OLIVE2, FLANK = pigment("#f0d03e"), pigment("#a8ad46"), pigment("#6f7c33"), pigment("#d2c8b6")
WINGA, TERT, PRIM, TAILC = pigment("#c4c462"), pigment("#77813e"), pigment("#454b31"), pigment("#4b5238")
LINE, BLACK, BILLC, LEGS = pigment("#37372a"), pigment("#1d1917"), pigment("#3c3b3d"), pigment("#3d3936")


def bird(r, perch, along, scale=2.25, tilt=-0.25):
    """A white-eye on the bough, leaning toward the fruit, facing left, cut after a brush drawing. The
    yellow-olive back on one block wiped deeper down the back and paler into the white of the belly; the
    lemon throat and the feathers under the tail on another; the flank shaded grey. The back and the
    coverts of the wing are rows of small scale-shaped feathers on a block of their own, a few with a dark
    fleck; every long feather of the wing and the tail is a tapering stroke, wiped dark along its shaft,
    and a block of fine lines edges them. The eye is a small dark dot in a thin ring that takes no colour;
    the bill is sharp, and the legs and toes are fine dark strokes that grip the bark."""
    c, s = np.cos(tilt), np.sin(tilt)

    def T(p):
        p = np.asarray(p, np.float64).reshape(-1, 2) - PERCH
        p = np.c_[-p[:, 0], p[:, 1]]
        return perch + scale * np.c_[p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]

    def loc(X, Y):
        q0, q1 = (X - perch[0]) / scale, (Y - perch[1]) / scale
        return -(q0 * c + q1 * s) + PERCH[0], -q0 * s + q1 * c + PERCH[1]
    L = lambda f: (lambda X, Y: f(*loc(X, Y), X.shape))
    g = tilt + np.pi
    stroke = lambda path, w, **k: brush(T(path), w * scale, r, **k)
    down = T([(0, 1)])[0] - T([(0, 0)])[0]

    sp = spline(BODY, 20)                               # the back and the belly ruffle into small feathers
    step = np.hypot(*np.diff(sp, axis=0).T)
    ph = np.r_[0, np.cumsum(step / (r.uniform(6.5, 8.5) * (1 + 0.3 * noise.line1d(len(step), 30, r))))]
    x, y = sp[:, 0], sp[:, 1]
    loose = (noise.smoothstep(26, 40, y) * noise.smoothstep(10, 0, x)
             + noise.smoothstep(-6, -14, y) * noise.smoothstep(-12, -22, x) * noise.smoothstep(-78, -66, x))
    body = knife(T(sp - normals(sp) * (r.uniform(0.9, 1.5) * (1 - (ph % 1) ** 1.4) * loose)[:, None]), r, 0.35, 0)
    wsp, cov = spline(WING, 10), spline(COVERTS, 8)
    wing, coverts = knife(T(wsp), r, 0.35, 0), knife(T(cov), r, 0.3, 0)
    ring = knife(T(oval(EYE, 4.0, 3.8, 0.3, 48)), r, 0.25, 0)
    eye = knife(T(oval(EYE + (0.3, 0), 2.6, 2.5, 0, 32)), r, 0.15, 0)
    lore = stroke([EYE + (3.9, 0.8), EYE + (6.5, 2.2), (19.5, -11.8)], 0.7, entry=0, lift=0.5, swell=0, amp=0.1)[0]
    bill = stroke(bez(*BILL, 20), 3.3, entry=0, lift=0.95, swell=0, amp=0.15)[0]

    def curve(base, tip, bow):
        base, tip = np.asarray(base, np.float64), np.asarray(tip, np.float64)
        d = tip - base
        nr = np.array([-d[1], d[0]]) / np.hypot(*d) * bow
        return bez(base, base + d / 3 + nr, base + 2 * d / 3 + nr, tip, 30)

    def quill(p, w, deep, tip):
        """A feather wiped dark along its shaft and pale toward its edge, and deeper toward its tip."""
        def f(X, Y):
            tt, ss = frame(p, X, Y, np.maximum(w, 1.0))
            return np.clip(deep * (0.35 + 0.65 * noise.smoothstep(1.0, 0.1, np.abs(ss) + uneven(X.shape, r, g, 0.12, 8)))
                           * (1 + tip * noise.smoothstep(0.4, 1.0, tt)), 0, None)
        return f
    feathers = [stroke(curve(b, e, r.normal(0, 1.2)), w, entry=1.2, lift=0.5, swell=0.05) + (PRIM, 0.9, 0.35)
                for b, e, w in PRIMARIES]
    feathers += [stroke(curve(b, e, r.normal(0, 1.2)), w, entry=1.2, lift=0.42, swell=0.05) + (TERT, 0.8, 0.15)
                 for b, e, w in SECONDARIES]
    tails = [stroke(curve(b, e, r.normal(0, 1.5)), w, entry=1.2, lift=0.45, swell=0.05) for b, e, w in TAILS]
    front = [(wing, 0), (body, 0)] + [(f[0], 0) for f in feathers]
    blocks = []
    for i, (q, p, w) in enumerate(tails):              # the tail, behind the body and the wing
        blocks.append(Block(TAILC, [(q, 1)] + [(o[0], 0) for o in tails[i + 1:]] + front, quill(p, w, 0.85, 0.3), g))
    for i, (q, p, w, k, deep, tip) in enumerate(feathers):     # each feather overlaps the ones below it
        over = [(o[0], 0) for o in feathers[i + 1:]] + [(coverts, 0)]
        blocks.append(Block(k, [(q, 1)] + over, quill(p, w, deep, tip), g))
    edges = []
    for q, p, w, k, deep, tip in feathers[len(PRIMARIES):]:    # a fine line down the lower edge of each
        nr = normals(p)
        e = p + np.sign((nr @ down).mean()) * nr * (0.82 * w)[:, None]
        edges.append(brush(e[int(0.2 * len(e)):int(0.96 * len(e))], 0.5 * scale, r, entry=0, lift=0.5, swell=0, amp=0.1)[0])
    shafts = [brush(p[int(0.06 * len(p)):int(0.88 * len(p))], 0.42 * scale, r, entry=0, lift=0.6, swell=0, amp=0.1)[0]
              for q, p, w in tails]

    def crescent(cx, cy, rs):
        """The rounded end of one small feather, the next row overlapping its base."""
        th = np.linspace(np.pi - 1.25, np.pi + 1.25, 12) + r.normal(0, 0.15)
        arc = np.c_[cx + rs * np.cos(th), cy + 0.8 * rs * np.sin(th)]
        return knife(T(ribbon(arc, np.sin(np.linspace(0.2, np.pi - 0.2, 12)) * (0.12 * rs + 0.25))), r, 0.12, 0)
    scales, flecks = [], []
    bend = spline(BEND, 12, closed=False)
    for k, d in enumerate((4.5, 9.5, 15.0, 21.5, 29.0)):  # rows of coverts, larger toward the long feathers
        gap = 4.2 + 0.32 * d
        row = resample(bend - (d, 0), gap)
        row = (row[:-1] + row[1:]) / 2 if k % 2 else row
        row = row + r.normal(0, 0.12 * gap, row.shape)
        for cx, cy in row[inside(cov, row)]:
            scales.append(crescent(cx, cy, 0.55 * gap * r.uniform(0.85, 1.1)))
            if k >= 2 and r.random() < 0.5:
                flecks.append(knife(T(oval((cx + 0.35 * gap, cy), 0.3 * gap, 0.13 * gap, r.normal(0, 0.2), 12)), r, 0.1, 0))
    cand = np.c_[r.uniform(-74, -16, 400), r.uniform(-28, 6, 400)]     # and a few small ones down the back
    cand = cand[inside(sp, cand) & ~inside(wsp, cand) & (cKDTree(sp).query(cand)[0] > 2.6)]
    marks = []
    for q in cand:
        if all(np.hypot(*(q - m)) > 8 for m in marks):
            marks.append(q)
    scales += [crescent(cx, cy, r.uniform(1.5, 2.0)) for cx, cy in marks[:9]]

    ax = np.array(loc(perch[0] + along[0], perch[1] + along[1])) - np.array(loc(perch[0], perch[1]))
    ax = ax / np.hypot(*ax) * np.sign(ax[0])            # along the bark toward the head, and into it
    dn = np.array([-ax[1], ax[0]])
    top = np.array(loc(perch[0], perch[1] - 4.0))
    feet = []
    for u0 in (-7.0, 8.0):                              # the legs out of the belly feathers, the toes round the bark
        F = top + u0 * ax + 0.3 * dn
        feet.append(stroke([F + (2.5, -22), F + (1.2, -9), F], 1.15, entry=0, lift=0.12, swell=0, amp=0.1)[0])
        for toe in (((0, 0), (4, 0.7), (8, 2.8), (9.6, 6.6)), ((0, 0), (3, 1.5), (5.4, 4.6), (5.1, 8.6)),
                    ((0, 0), (-3.6, 0.4), (-6.6, 2.3), (-6.9, 6.0))):
            feet.append(stroke(bez(*[F + u * ax + v * dn for u, v in toe], 14), 0.85, entry=0, lift=0.45,
                               swell=0, amp=0.08)[0])

    yellow = L(lambda x, y, sh: np.clip(0.9 * noise.smoothstep(-26, 2, x + uneven(sh, r, g, 4.0, 14))
                                        * (1 - noise.smoothstep(26, 44, y + uneven(sh, r, g, 4.0, 14)))
                                        * (1 - 0.75 * noise.smoothstep(-15, -23, y + uneven(sh, r, g, 2.0, 10))
                                           * noise.smoothstep(18, 10, x))          # the crown olive, not yellow
                                        + 0.85 * noise.smoothstep(-64, -78, x) * noise.smoothstep(20, 34, y), 0, 1))
    parting = spline(PARTING, 12, closed=False)[::-1]

    def olive(x, y, sh):
        below = np.interp(x, parting[:, 0], parting[:, 1])
        above = noise.smoothstep(below + 5, below - 4, y + uneven(sh, r, g, 3.0, 10))
        return np.clip((0.5 + 0.42 * noise.smoothstep(12, -46, x)) * above + uneven(sh, r, g, 0.07, 14), 0, None)

    def flank(x, y, sh):
        below = np.interp(x, parting[:, 0], parting[:, 1])
        return (0.55 * noise.smoothstep(below - 2, below + 6, y) * noise.smoothstep(54, 28, y + uneven(sh, r, g, 5.0, 12))
                * noise.smoothstep(-2, -24, x) * noise.smoothstep(-88, -70, x))
    billc = L(lambda x, y, sh: 0.95 - 0.4 * noise.smoothstep(-0.5, 1.2, y - np.interp(x, (19, 40), (-10.6, -8.6))))
    blocks += [Block(YELLOW, [(body, 1), (ring, 0), (wing, 0)], yellow, g),
               Block(OLIVE, [(body, 1), (ring, 0), (wing, 0)], L(olive), g, off=2.0),
               Block(FLANK, [(body, 1), (wing, 0)], L(flank), g, salt=0.9),
               Block(WINGA, [(wing, 1)], L(lambda x, y, sh: np.clip(0.75 + uneven(sh, r, g, 0.06, 12), 0, None)), g),
               Block(OLIVE, [(coverts, 1)], L(lambda x, y, sh: np.clip(0.7 + 0.25 * noise.smoothstep(-10, -40, x)
                                                                       + uneven(sh, r, g, 0.08, 8), 0, None)), g),
               Block(OLIVE2, [(q, 1) for q in scales], lambda X, Y: np.clip(0.8 + uneven(X.shape, r, g, 0.1, 10), 0, None),
                     g, wet=0.15, off=1.4),
               Block(LINE, [(q, 1) for q in edges + flecks], lambda X, Y: 0.75, g, wet=0.12, off=1.2),
               Block(LINE, [(q, 1) for q in shafts] + front, lambda X, Y: 0.7, g, wet=0.12, off=1.2),
               Block(BLACK, [(eye, 1)], lambda X, Y: 0.95, g, wet=0.08, salt=0.1, off=0.8),
               Block(LINE, [(lore, 1)], lambda X, Y: 0.8, g, wet=0.1),
               Block(BILLC, [(bill, 1)], billc, g, wet=0.12),
               Block(LEGS, [(f, 1) for f in feet] + [(body, 0)], lambda X, Y: 0.85, g, wet=0.15)]
    scallops = []
    th = np.linspace(-0.9, 0.9, 10)
    for cx, cy in np.array([(-18, 44), (-32, 49), (-46, 48), (-25, 52), (-56, 45), (-40, 43), (-8, 40), (-1, 32)]) \
            + r.normal(0, 1.2, (8, 2)):
        arc = np.c_[cx - 5.5 * np.cos(th), cy + 5.5 * np.sin(th)]
        scallops.append(T(ribbon(arc, np.sin(np.linspace(0.3, 2.8, 10)) * 0.7)))
    raised = [[(ring, 1), (eye, 0)]] + [[(q, 1)] for q in scallops]
    return [body, wing] + [f[0] for f in feathers] + [t[0] for t in tails] + feet, blocks, raised


# ---------------------------------------------------------------- the inscription

def write(ch, px, face, r, S=4):
    """One character, about `px` pixels high, as the brush wrote it and the cutter followed it: each stroke
    swells where the brush was pressed and thins where it was lifted, the edge is chipped here and there,
    and once in a while the wood has broken clean across a stroke."""
    n = int(px * S * 1.5)
    im = Image.new("L", (n, n), 0)
    ImageDraw.Draw(im).text((n / 2, n / 2), ch, 255, font=ImageFont.truetype(str(XINGKAI), int(px * S), index=face),
                            anchor="mm")
    m = np.asarray(im.rotate(r.normal(0, 2.0), Image.BICUBIC)) > 127
    din = ndimage.distance_transform_edt(m)
    sdf = din - ndimage.distance_transform_edt(~m)
    hw = ndimage.grey_dilation(din, size=int(S * px * 0.1) | 1)          # how wide the stroke is about here
    a = noise.smoothstep(-1.2, 1.2, sdf + 0.28 * hw * np.clip(noise.field(m.shape, S * px * 0.22, r), -2, 1.6))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    edge = np.argwhere(m & (din < 1.5))
    gy, gx = np.gradient(ndimage.gaussian_filter(sdf, 2.0))
    for y, x in edge[r.choice(len(edge), r.poisson(5))]:
        a *= noise.smoothstep(0, 1.5, np.hypot(yy - y, xx - x) - r.uniform(0.5, 1.4) * S)
    for y, x in edge[r.choice(len(edge), r.poisson(0.6))]:
        nx, ny = np.array([gx[y, x], gy[y, x]]) / (np.hypot(gx[y, x], gy[y, x]) + 1e-6)
        u, v = (xx - x) * nx + (yy - y) * ny, np.abs((xx - x) * ny - (yy - y) * nx)
        a *= 1 - ((1 - noise.smoothstep(0.25 * S, 0.6 * S, v)) * noise.smoothstep(-2, 0, u)
                  * noise.smoothstep(2.4 * hw[y, x], 2.0 * hw[y, x], u))
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((n // S, n // S), Image.BOX))


def cut():
    """Cut the inscription into atelier/lettering (it needs the Xingkai font of macOS): the title in one
    column, and the signature beside it in smaller characters, starting lower."""
    r = noise.rng(7)
    T, s = 58, 36
    W, H = int(T * 2.9), int(T * 5.0)
    a = np.zeros((H, W), np.float32)

    def put(gl, cx, cy):
        h, w = gl.shape
        y0, x0 = int(cy - h / 2), int(cx - w / 2)
        a[y0:y0 + h, x0:x0 + w] = np.maximum(a[y0:y0 + h, x0:x0 + w], gl)
    y = T * 0.85
    for ch in HEADING:
        put(write(ch, T * r.uniform(0.93, 1.07), 1, r), W - T * 0.85 + r.normal(0, 2.5), y)
        y += T * r.uniform(1.04, 1.16)
    y = T * 1.9
    for ch in SIGN:
        put(write(ch, s * r.uniform(0.95, 1.05), 3, r), W - T * 1.97 + r.normal(0, 1.5), y)
        y += s * r.uniform(1.08, 1.18)
    a = noise.smoothstep(0.3, 0.6, a + 0.12 * noise.field(a.shape, 2.0, r))    # the knife roughens every edge
    Image.fromarray((a * 255).astype(np.uint8)).save(INSCRIPTION)


class Lettering(Block):
    """The inscription, cut on a block of its own and printed in ink."""

    def __init__(self, x0, y0):
        super().__init__(KEY, [], None, 0.0, wet=0.3, salt=0.4, spread=1.6)
        self.x0, self.y0 = x0, y0

    def cut(self, pad=0):
        self.m = np.asarray(Image.open(INSCRIPTION), np.float32) / 255
        self.amt = np.full_like(self.m, 0.92)
        return self


# ---------------------------------------------------------------- the leaf, cut and printed

def emboss(img, raised, light=(-0.55, -0.7)):
    """The uninked block: the damp paper was pressed hard into it, so its shapes stand up from the sheet
    and show only where the raking light catches one side of them and leaves the other in shadow."""
    h = ndimage.gaussian_filter(raised, 1.3) + 0.6 * ndimage.gaussian_filter(raised, 3.5)
    gy, gx = np.gradient(h)
    shade = 1 - 1.0 * (gx * light[0] + gy * light[1])
    return img * np.clip(shade, 0.75, 1.2)[..., None]


ROSETTE = [  # leaves along the last stretch of the bough: angle, length, width, droop, kind, fold, wave,
    # how far back along the bough it springs, how browned its tip is
    (-96, 470, 92, 0.05, "top", 0.6, 0.04, 70, 0), (-127, 720, 106, 0.08, "top", 0.85, 0.03, 48, 0),
    (-150, 780, 112, 0.03, "top", 0.8, 0.05, 28, 0), (178, 880, 112, 0.13, "top", 0.9, 0.02, 6, 0),
    (146, 560, 104, 0.05, "under", 1.0, 0.02, 0, 0), (100, 560, 100, 0.0, "top", 0.5, -0.04, 12, 0.55)]
FORWARD = [(58, 300, 134, 0.2, "top", 0.4, 0.1, 2, 0)]          # a leaf turned toward us, in front of the fruit
SHOOT = [(22, 470, 90, 0.05, "new", 0.85, 0.03, 4, 0), (64, 600, 98, 0.06, "top", 0.75, -0.03, 12, 0),
         (106, 500, 96, 0.03, "top", 0.6, 0.04, 22, 0.4)]
FRUIT = [  # centre, radius, how ripe, how far the calyx end is turned toward us, kind, how near the front
    ((1005, 1462), 68, 0.5, 0.2, "ripe", 1), ((1120, 1492), 84, 0.75, 0.15, "ripe", 4),
    ((893, 1503), 58, 0.25, 0.1, "pale", 0), ((1232, 1452), 56, 0.3, 0.0, "pale", 0),
    ((838, 1628), 70, 0.85, 0.45, "ripe", 3), ((975, 1594), 84, 0.7, 0.5, "blush", 6),
    ((1150, 1662), 72, 0.6, 0.2, "ripe", 5), ((1258, 1598), 60, 0.5, 0.1, "ripe", 2),
    ((906, 1764), 62, 0.9, 0.3, "ripe", 4), ((1046, 1758), 78, 0.8, 0.6, "blush", 7),
    ((1192, 1782), 58, 0.3, 0.0, "pale", 3), ((1086, 1886), 50, 0.6, 0.25, "ripe", 6)]


def scene(r):
    """Everything on the leaf, back to front: (footprint, blocks, raised shapes)."""
    def leaves(path, specs):
        out = []
        for a, L, w, droop, kind, fold, wave, back, scorch in specs:
            bend = droop * np.sign(np.cos(np.radians(a)))
            out.append(leaf(r, path[-1 - back] + r.normal(0, 4, 2), np.radians(a + r.normal(0, 3)),
                            L * r.uniform(0.9, 1.06), w * r.uniform(0.9, 1.1), bend, kind, fold, r.uniform(0.8, 1.1),
                            wave, scorch))
        return out
    path = bez((2080, 740), (1780, 800), (1430, 1000), (1080, 1280), 300)
    C = path[-1]
    P = path[210]
    twig = bez(P, P + (30, 60), P + (70, 130), P + (90, 200), 80)
    items = leaves(path, ROSETTE) + leaves(twig, SHOOT)
    foot, blocks, _, _ = bough(r, twig, 15, 9, young=0.3)
    items.append((foot, blocks, []))
    foot, blocks, p, hw = bough(r, path, 42, 20)
    items.append((foot, blocks, []))
    knot = C + (-30, 110)
    fruits = []
    for c, rad, ripe, face, kind, z in FRUIT:
        c = np.asarray(c, np.float64)
        fruits.append((z, fruit(r, c, rad, np.arctan2(*(knot - c)[::-1]) + r.normal(0, 0.2), ripe, face, kind)))
    o, b = stalk(r, bez(C, C + (-5, 40), knot + (8, -40), knot, 30), 10, 8)
    sfoot, stalks = [o], [b]
    for _, (_, _, stem) in fruits:
        mid = knot + (stem - knot) * 0.5 + r.normal(0, 12, 2)
        o, b = stalk(r, bez(knot, mid, mid, stem, 24), 6.5, 5)
        sfoot.append(o)
        stalks.append(b)
    items.append((sfoot, stalks, []))
    items += [f[:2] + ([],) for z, f in sorted(fruits, key=lambda q: q[0])]
    items += leaves(path, FORWARD)
    i = int(0.45 * (len(p) - 1))
    nrm = normals(p)[i]
    top = p[i] + nrm * hw[i] * (1 if nrm[1] < 0 else -1)
    items.append(bird(r, top + (0, 4), p[min(i + 3, len(p) - 1)] - p[max(i - 3, 0)], 2.25, -0.25))
    return items


def paint(seed=1633):
    r = noise.rng(seed)
    sheet = paper.washi((SH, SW), seed, tint="#f3ead6", margin=(36, 34), fibre_density=0.3)
    sheet.alpha = paper.deckle((SH, SW), (36, 34), r, ragged=0.8)
    cover = np.zeros((SH, SW), np.float32)
    raised = np.zeros((SH, SW), np.float32)
    todo = [Lettering(320, 1990).cut()]
    for foot, blocks, emb in scene(r)[::-1]:            # front to back: each thing is cleared from the blocks behind it
        for b in blocks:
            b.cut()
            b.m *= 1 - cover[b.at]
            todo.append(b)
        for e in emb:
            u = Block(None, [(e, 1)] if isinstance(e, np.ndarray) else e, lambda X, Y: 1.0).cut(pad=4)
            u.m = np.clip(ndimage.shift(u.m * (1 - cover[u.at]), r.normal(0, 1.6, 2), order=1), 0, 1)
            raised[u.at] = np.maximum(raised[u.at], u.m)
        for f in foot:
            u = footprint([f])
            cover[u.at] = np.maximum(cover[u.at], u.m)
    img = sheet.color.copy()
    for b in todo:
        press(img, sheet, b, r, **b.press)
    img = emboss(img, raised)
    return plate.mount(img, sheet, shadow=0.35)


if __name__ == "__main__":
    cut()
