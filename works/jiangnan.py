"""Water Town in Spring. Watercolour woodblock print in the manner of the Jiangsu school.

Across a canal in a water town of Jiangnan, on a still spring morning after rain: white walls and dark
tiled roofs along the far bank, stepping back in rows, the stepped fire walls rising over them, an arched
bridge where a side canal comes in, two boats, a willow and a little blossom. The town is seen a little
from above and almost in elevation, as the Chinese printmakers draw it, with no vanishing point: heights
go straight up the sheet and depth runs up it and a little aside. It is printed the Chinese way, in water
colour and not in oil: the colour is brushed onto each block, gradations brushed in with a second, wetter
brush, and the dampened xuan laid over it and rubbed. The damp sheet takes the colour softly and draws a
hair of it out past the edge of every cut along its fibres; thin colour misses the valleys of the paper,
and the grain of the board shows in every flat. The walls are the bare paper. The water gives the town
back from blocks cut from the same drawing turned upside down, printed paler and wetter and broken by a
few ripples cut across them.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from atelier import ink, noise, paper, plate, relief, watercolour
from atelier.color import glaze, pigment

TITLE = "Water Town in Spring"
DATE = "2026"
MEDIUM = "Watercolour woodblock print (shuiyin muke), twenty-one impressions from twenty blocks, on dampened xuan paper"
AFTER = ("the watercolour woodcuts of the Jiangsu school, Nanjing, 1960s–1980s, and the Ten Bamboo Studio manual, "
         "Nanjing, 1633")
ROOM = "Paper and Water"
YEAR = 1980
PLACE = "Nanjing"
REGION = "East Asia"
NOTE = ("White walls and dark tiles across a canal on a spring morning after rain. The walls are the bare paper; "
        "the reflections are cut from the same drawing turned upside down and printed paler and wetter.")

H, W, M = 1800, 2400, 80
SH, SW = H + 2 * M, W + 2 * M
SS = 2                                      # blocks are cut at twice the size they print
S, KX, KY, COS = 50.0, 0.15, 0.25, 0.96     # px a metre; how far depth runs right and up the sheet; heights seen from a little above
X0, Y0 = -2.0, 1120.0                       # the metre at the sheet's left edge, and the row of the far waterline
BASE, WALK, BX = 1.1, 2.2, 31.7             # the stone bank, the river street on it, the bridge over the side canal

SKY, WARM, WATER = pigment("#c6d0d6"), pigment("#f0e3cf"), pigment("#b4cfc6")
STONE, SHADE, WOOD = pigment("#bab3a6"), pigment("#c2c8cb"), pigment("#a89c8b")
ROOF, CAP, KEY, DARK = pigment("#62676b"), pigment("#4f5357"), pigment("#76797c"), pigment("#4c4f53")
HULL, WILLOW, LEAF = pigment("#9c9184"), pigment("#d4dd8e"), pigment("#a3b862")
PEACH, BLUSH, COAT, STRAW = pigment("#eeb0b4"), pigment("#d98a98"), pigment("#69778a"), pigment("#cdb98e")

NAMES = ("sky", "water", "wall", "stone", "shade", "wood", "roof", "cap", "dark", "hull", "key",
         "willow", "leaf", "peach", "blush", "coat", "straw")


def hand(p, r, amp, step):
    """Resample a path on the block every `step` px and let it waver, as a knife guided by hand does."""
    s0 = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s0[-1] < 1e-6:
        return p
    s = np.linspace(0, s0[-1], max(2, int(s0[-1] / step) + 1))
    q = np.c_[np.interp(s, s0, p[:, 0]), np.interp(s, s0, p[:, 1])]
    t = np.gradient(q, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    k = s / (40.0 * SS)
    wav = amp * (np.sin(k * r.uniform(0.5, 1.5) + r.uniform(0, 7)) + 0.5 * np.sin(k * r.uniform(2, 4) + r.uniform(0, 7)))
    return q + np.c_[-t[:, 1], t[:, 0]] * wav[:, None]


def box(x0, x1, y0, y1, d):
    return [(x0, y0, d), (x1, y0, d), (x1, y1, d), (x0, y1, d)]


class Blocks:
    """The blocks of the print, cut together from one drawing at twice the size they print. Things are cut
    back to front: each first clears its shape from every block, so what stands in front hides what is
    behind it on all of them alike. With `mirror` the drawing is cut upside down for the water, turned
    about the waterline under the thing being cut, `dref` metres out from the far bank."""

    def __init__(self, mirror=False):
        self.im = {n: Image.new("L", (SW * SS, SH * SS), 0) for n in NAMES}
        self.d = {n: ImageDraw.Draw(i) for n, i in self.im.items()}
        self.tone = {n: np.zeros((SH, SW), np.float32) for n in ("roof", "shade", "stone")}
        self.mirror, self.dref, self.kt = mirror, 0.0, 1.0      # kt: how dark the keyline is brushed here

    def P(self, pts):
        """World (metres: x along the far bank, y up from the water, d back from it) to the block (px)."""
        p = np.atleast_2d(np.asarray(pts, np.float64))
        up = COS * p[:, 1] + KY * p[:, 2]
        if self.mirror:
            up = 2 * KY * self.dref - up
        return np.c_[M + S * (p[:, 0] - X0 + KX * p[:, 2]), M + Y0 - S * up] * SS

    def area(self, name, pts, fill=255, screen=False):
        self.d[name].polygon([tuple(v) for v in (pts if screen else self.P(pts))], fill=fill)

    def clear(self, pts, screen=False):
        q = [tuple(v) for v in (pts if screen else self.P(pts))]
        for d in self.d.values():
            d.polygon(q, fill=0)

    def ramp(self, name, pts, a, b, t0, t1, g=1.0, over=False):
        """Cut a shape and brush a gradation into it: t0 of the colour along edge a, t1 along edge b.
        `over` lays it on the same surface as what is already there instead of in front of it."""
        self.area(name, pts)
        q = self.P(pts) / SS
        (x0, y0), (x1, y1) = np.floor(q.min(0)).astype(int) - 1, np.ceil(q.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, SW), min(y1, SH)
        if x1 <= x0 or y1 <= y0:
            return
        im = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(im).polygon([tuple(v - (x0, y0)) for v in q], fill=255, outline=255)
        m = np.asarray(im) > 0
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        def dist(seg):
            (ax, ay), (bx, by) = self.P(seg)[:2] / SS
            nx, ny = by - ay, ax - bx
            return np.abs((xx - ax) * nx + (yy - ay) * ny) / (np.hypot(nx, ny) + 1e-9)

        da, db = dist(a), dist(b)
        u = da / (da + db + 1e-6)
        tone, old = t0 + (t1 - t0) * u ** g, self.tone[name][y0:y1, x0:x1]
        old[m] = np.maximum(old, tone)[m] if over else tone[m]

    def flat(self, name, pts, t):
        self.ramp(name, pts, pts[:2], pts[1:3], t, t)

    def strip(self, name, p, hw, fill=255):
        """A band of half-width hw (px as printed) along a path already on the block."""
        if len(p) < 2:
            return
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        off = np.c_[-t[:, 1], t[:, 0]] * (np.broadcast_to(hw, len(p)) * SS)[:, None]
        self.d[name].polygon([tuple(v) for v in np.vstack([p + off, (p - off)[::-1]])], fill=fill)

    def key(self, pts, r, hw=0.8, name="key", fill=255, breaks=0.006, amp=0.6, screen=False):
        """A line left standing between two knife cuts: never of one width, a little unsteady, and broken
        here and there where the thin ridge of wood gave way. With fill=0 it is a line cut into a block."""
        p = np.asarray(pts, np.float64) if screen else self.P(pts)
        if len(p) < 2:
            return
        p = hand(p, r, amp * SS, 3.0 * SS)
        fill = int(fill * self.kt) if fill and name == "key" else fill
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))] / SS
        if s[-1] < 1.0:
            return
        _, w = relief.vcut(p, hw, r, taper=min(0.3, 5.0 / s[-1]), wobble=0.15)
        w = w * (1 + 0.35 * np.sin(s / r.uniform(12, 30) + r.uniform(0, 7)) * np.sin(s / r.uniform(50, 140) + r.uniform(0, 7)))
        gone = np.zeros(len(p), bool)
        for c in r.uniform(0, s[-1], r.poisson(s[-1] * breaks)):
            gone |= (s > c) & (s < c + r.uniform(1.5, 4.0))
        edges = np.flatnonzero(np.diff(np.r_[1, gone.astype(int), 1]))
        for a, b in zip(edges[::2], edges[1::2]):
            self.strip(name, p[a:b], w[a:b], fill)

    def white(self, pts):
        """A whitewashed wall: it hides what is behind it and takes no colour, but the water gives it back."""
        self.clear(pts)
        self.area("wall", pts)

    def get(self, name):
        return np.asarray(self.im[name].resize((SW, SH), Image.BOX), np.float32) / 255

    def shade(self, name):
        """The brushed tone of a block, carried a pixel past the edges of its shapes."""
        t = self.tone[name]
        return np.where(t > 0, t, ndimage.grey_dilation(t, 3))


FRONT = [("E", 7.0, 2), ("E", 6.0, 1), ("G", 6.5, 2), ("E", 7.5, 2), ("E", 4.5, 1), (None, 6.4, 0), ("G", 6.0, 2),
         ("E", 7.5, 1)]


def town(r):
    """The houses in two rows, back to front and each row left to right. The row on the water is laid
    out by hand, the mouth of the side canal left open in it for the bridge; in the row behind each house
    is its own: one storey or two, the eaves to the canal or the gable, a fire wall at the end or none."""
    def house(kind, x0, L, two, d0, tf, near):
        return dict(gable=kind == "G", x0=x0, x1=x0 + L, d0=d0, D=r.uniform(5.5, 6.5) if kind == "G" else r.uniform(6.5, 8.5),
                    tf=tf, near=near,
                    he=BASE + (5.9 if two else 3.3) + r.uniform(-0.2, 0.3), pitch=r.uniform(0.44, 0.5),
                    fire=r.random() < (0.6 if near > 0.8 else 0.35), fire_r=False, steps=int(r.choice([2, 3])),
                    timber=two and r.random() < 0.45, seed=int(r.integers(1 << 30)))
    out = []
    x, d0 = X0 - KX * 14 - r.uniform(2, 6), 14.0
    while x < X0 + W / S - KX * d0 + 2:                 # the row behind, its houses standing a little in and out
        g = r.random() < 0.35
        L = r.uniform(5.0, 7.0) if g else r.uniform(5.0, 8.5)
        out.append(house("G" if g else "E", x, L, r.random() < 0.75, d0 + r.uniform(-1.0, 3.0), 0.55, 0.45))
        x += L
    x = X0 - 1.0
    for kind, L, two in FRONT:
        if kind:
            out.append(house(kind, x, L, two == 2, WALK, 1.0, 1.0))
        else:
            out[-1]["fire_r"] = True                        # the house on the side canal ends in a fire wall
        x += L
    return out


def cut_house(bl, r, h):
    bl.kt = 0.25 + 0.75 * h["near"]
    (gable if h["gable"] else eaves)(bl, r, h)


def eaves(bl, r, h):
    """A house with its eaves to the canal: the fire wall at its left end, the plain end wall at its right,
    the white front with the shadow of the eaves on it, and the roof, a flat of dark tiles seen from a
    little above, with a few of its rows cut in it and a darker course of tiles along the eave and the ridge."""
    x0, x1, d0, D, he, p, tf, near = (h[k] for k in ("x0", "x1", "d0", "D", "he", "pitch", "tf", "near"))
    hr = he + (D / 2 + 0.6) * p
    if h["fire"]:
        fire_wall(bl, r, h, x0)
    if not h["fire_r"]:
        end = [(x1, BASE, d0), (x1, he, d0), (x1, hr, d0 + D / 2), (x1, he, d0 + D), (x1, BASE, d0 + D)]
        bl.white(end)
        bl.flat("shade", end, 0.34)
        bl.key(end[1:4], r, 0.7)
    front = box(x0, x1, BASE, he, d0)
    bl.white(front)
    shadow(bl, r, x0, x1, he, d0, near)
    openings(bl, r, h)
    bl.key(front[3::-3], r, 0.7 * near + 0.2, breaks=0.01)
    a = np.linspace(x0 - (0 if h["fire"] else 0.3), x1 + (0 if h["fire_r"] else 0.3), 9)
    sag = r.uniform(0.03, 0.12) * np.sin(np.pi * (a - a[0]) / (a[-1] - a[0]))   # old roofs sag between their ends
    eave, ridge = np.c_[a, he - 0.1 - sag, np.full(9, d0 - 0.6)], np.c_[a, hr - 0.5 * sag, np.full(9, d0 + D / 2)]
    roof = np.vstack([eave, ridge[::-1]])
    bl.clear(roof)
    bl.ramp("roof", roof, ridge[[0, -1]], eave[[0, -1]], 0.85 * tf, tf)
    tiles(bl, r, eave, ridge, near)
    bl.key(eave, r, 2.2 * tf, name="cap", fill=int(255 * tf), breaks=0.002, amp=0.3)     # the course of drip tiles
    bl.key(ridge, r, 1.8 * tf, name="cap", fill=int(255 * tf), breaks=0.002, amp=0.3)
    if h["fire_r"]:
        fire_wall(bl, r, h, x1)


def tiles(bl, r, eave, ridge, near):
    """A run of tile rows cut into the roof as fine white lines down the slope, over one stretch of it
    only, the knife skipping here and there. Far off, a roof is left its flat grey."""
    if near < 0.4:
        return
    (ex, ey), (gx, gy) = bl.P(eave[[0, -1]]).T, bl.P(ridge[[0, -1]]).T
    u0 = r.uniform(0.1, 0.5)
    for u in np.arange(u0, min(u0 + r.uniform(0.25, 0.45), 0.95), 0.5 * S * SS / (ex[1] - ex[0])):
        x = ex[0] + u * (ex[1] - ex[0])
        y0 = np.interp(x, gx, gy) + r.uniform(0.02, 0.12) * (ey[0] - gy[0])
        y1 = np.interp(x, ex, ey) - r.uniform(0.04, 0.2) * (ey[0] - gy[0])
        bl.key([(x, y0), (x, y1)], r, 0.45 * near, name="roof", fill=0, breaks=0.02, amp=0.15, screen=True)


def shadow(bl, r, x0, x1, top, d, near):
    """The soft grey of the shadow under the eaves brushed down a white wall and fading, and the damp
    climbing its foot."""
    band = box(x0, x1, top - r.uniform(1.6, 2.2), top, d)
    bl.ramp("shade", band, band[2:], band[:2], 0.55 + 0.1 * near, 0.0, g=0.8)
    foot = box(x0, x1, BASE, BASE + r.uniform(1.0, 1.8), d)
    bl.ramp("shade", foot, foot[:2], foot[2:], 0.18, 0.0, g=1.4, over=True)


def cap(bl, r, p, tf, kick=(0.0, 0.0)):
    """The little roof of dark tiles along the top of a wall, `p` its line on the block, its ends kicking
    up by `kick` metres like a horse's head."""
    q = bl.P(p)
    w = np.full(len(q), 0.13 * S * tf ** 0.5)
    for i, kk in ((0, kick[0]), (-1, kick[1])):
        if kk:                                          # the end of the cap curls up and out
            v = q[i] - q[1 if i == 0 else -2]
            v /= np.linalg.norm(v) + 1e-9
            q = np.insert(q, len(q) if i else 0, q[i] + (v * 0.6 + [0, -1 if not bl.mirror else 1]) * kk * S * SS, axis=0)
            w = np.insert(w, len(w) if i else 0, 0.05 * S)
    bl.strip("cap", q, w, fill=int(255 * tf))


def fire_wall(bl, r, h, xw):
    """A stepped fire wall at the end of a house, rising over its roof in steps toward the ridge and down
    again behind, seen nearly end on: the white ends of its steps one above another, each with a small
    cap of dark tiles, and its side going back in the grey of its own shade."""
    d0, D, he, p, k, tf = (h[n] for n in ("d0", "D", "he", "pitch", "steps", "tf"))
    t, e = 0.32, np.linspace(0, D / 2, k + 1)
    tops = he + 0.55 + (e[1:] + 0.6) * p                    # each step stands clear of the roof under it
    prof = [(0.0, BASE)] + [q for (a, b), y in zip(zip(e[:-1], e[1:]), tops) for q in ((a, y), (b, y))]
    prof += [(D - dd, y) for dd, y in prof[::-1]]
    face = [(xw + t, y, d0 + dd) for dd, y in prof]
    bl.white(face)
    bl.flat("shade", face, 0.38 * tf)
    for j in range(k - 1, -1, -1):                      # the ends of the steps, the deepest first
        bl.white(box(xw - t, xw + t, BASE if j == 0 else tops[j - 1], tops[j], d0 + e[j]))
        bl.key([(xw - t, tops[j], d0 + e[j]), (xw - t, BASE if j == 0 else tops[j - 1], d0 + e[j])], r, 0.5, breaks=0.02)
    bl.key(face[1:-1], r, 0.45, breaks=0.02)
    for y, d in zip(tops, d0 + e[:-1]):                 # the cap on the end of each step, a little roof of its own
        bl.area("cap", [(xw - t - 0.2, y - 0.04, d), (xw - t - 0.13, y - 0.12, d), (xw + t + 0.13, y - 0.12, d),
                        (xw + t + 0.2, y - 0.04, d), (xw + t + 0.04, y + 0.2, d), (xw - t - 0.04, y + 0.2, d)],
                fill=int(255 * tf))


def gable(bl, r, h):
    """A house that turns its gable to the canal: its side wall in shade, the roof running back from the
    canal, and the white gable stepping up over the roof to the ridge and down again, each step capped
    with dark tiles."""
    x0, x1, d0, D, he, p, k, tf, near = (h[n] for n in ("x0", "x1", "d0", "D", "he", "pitch", "steps", "tf", "near"))
    xm = (x0 + x1) / 2
    hr = he + ((x1 - x0) / 2 + 0.2) * p
    side = [(x1, BASE, d0), (x1, BASE, d0 + D), (x1, he, d0 + D), (x1, he, d0)]
    bl.white(side)
    bl.flat("shade", side, 0.34 * tf)
    for q, g0 in (([(x0 - 0.2, he - 0.1, d0), (xm, hr, d0), (xm, hr, d0 + D), (x0 - 0.2, he - 0.1, d0 + D)], 0.95),
                  ([(xm, hr, d0), (x1 + 0.2, he - 0.1, d0), (x1 + 0.2, he - 0.1, d0 + D), (xm, hr, d0 + D)], 0.85)):
        bl.clear(q)
        bl.flat("roof", q, g0 * tf)
    if near > 0.4:
        for dd in np.arange(r.uniform(0.6, 1.2), D - 0.4, r.uniform(1.0, 1.4)):
            if r.random() < 0.55:
                a, b = r.uniform(0.05, 0.3), r.uniform(0.6, 0.95)
                bl.key([(xm + a * (x1 - xm), hr - a * (hr - he), d0 + dd), (xm + b * (x1 - xm), hr - b * (hr - he), d0 + dd)],
                       r, 0.5, name="roof", fill=0, breaks=0.0, amp=0.2)
    bl.key([(xm, hr, d0), (xm, hr, d0 + D)], r, 1.8 * tf, name="cap", fill=int(255 * tf), breaks=0.0, amp=0.3)
    e = np.linspace(0, (x1 - x0) / 2, k + 1)
    tops = he + 0.55 + (e[1:] + 0.2) * p
    prof = [(0.0, BASE)] + [q for (a, b), y in zip(zip(e[:-1], e[1:]), tops) for q in ((a, y), (b, y))]
    prof += [(x1 - x0 - a, y) for a, y in prof[::-1]]
    wall = [(x0 + a, y, d0) for a, y in prof]
    bl.white(wall)
    bl.ramp("shade", wall, [(x0, tops[-1], d0), (x1, tops[-1], d0)], [(x0, BASE + 3, d0), (x1, BASE + 3, d0)],
            0.32 + 0.1 * near, 0.0, g=0.9)
    bl.key(wall[1:-1], r, 0.6 * near + 0.2, breaks=0.01)
    for j, ((a, b), y) in enumerate(zip(zip(e[:-1], e[1:]), tops)):
        kick = 0.12 + 0.04 * j
        cap(bl, r, [(x0 + a - 0.12, y + 0.06, d0 - 0.1), (x0 + b + 0.12, y + 0.06, d0 - 0.1)], tf, (kick, 0.0))
        cap(bl, r, [(x1 - b - 0.12, y + 0.06, d0 - 0.1), (x1 - a + 0.12, y + 0.06, d0 - 0.1)], tf, (0.0, kick))
    openings(bl, r, h)


def openings(bl, r, h):
    """Doors and windows, few and small in the white: the open ones dark, the shutters and the timber
    fronts of upper storeys in a warm grey, the lattice cut fine through them in the near houses."""
    x0, x1, d0, he, near, tf = (h[n] for n in ("x0", "x1", "d0", "he", "near", "tf"))
    if near < 0.3:
        return
    if near < 0.8:                                      # further off, a window is a dark fleck
        for a in r.uniform(x0 + 0.6, x1 - 1.2, r.integers(0, 3)):
            y = BASE + (r.choice([1.0, 4.2]) if he > BASE + 5 else 1.0)
            bl.area("dark", box(a, a + 0.6, y, y + 0.7, d0), fill=int(r.uniform(90, 150) * tf))
        return
    if h["gable"]:
        xm = (x0 + x1) / 2
        door = box(xm - 0.6, xm + 0.6, BASE, BASE + 2.2, d0)
        bl.clear(door)
        bl.area("dark", door, fill=int(r.uniform(150, 200)))
        bl.key(door[:1] + door[3:0:-1], r, 0.6)
        if he > BASE + 5:
            window(bl, r, box(xm - 0.45, xm + 0.45, he - 1.6, he - 0.8, d0))
        return
    a = x0 + r.uniform(0.7, 1.5)
    while a < x1 - 1.6:                                 # the ground floor
        if r.random() < 0.5:
            w = r.uniform(1.0, 1.4)
            q = box(a, a + w, BASE, BASE + 2.15, d0)
            bl.clear(q)
            bl.area("wood" if r.random() < 0.6 else "dark", q, fill=int(r.uniform(140, 210)))
            bl.key(q[:1] + q[3:0:-1], r, 0.6)
            a += w + r.uniform(1.2, 2.6)
        else:
            w = r.uniform(0.8, 1.1)
            window(bl, r, box(a, a + w, BASE + 1.0, BASE + 1.9, d0))
            a += w + r.uniform(1.4, 3.0)
    if he < BASE + 5:
        return
    if h["timber"]:                                     # the upper storey fronted in timber
        q = box(x0 + 0.1, x1 - 0.1, BASE + 3.4, he - 0.3, d0)
        bl.clear(q)
        bl.area("wood", q, fill=int(r.uniform(130, 180)))
        bl.key([q[0], q[1]], r, 0.6)
        for a in np.arange(x0 + r.uniform(0.6, 0.9), x1 - 0.5, r.uniform(0.75, 0.95)):
            for y in np.linspace(BASE + 3.75, he - 0.6, 3):
                bl.key([(a + 0.08, y, d0), (a + 0.6, y, d0)], r, 0.35, name="wood", fill=0, breaks=0.0, amp=0.2)
        return
    for a in np.arange(x0 + r.uniform(0.8, 1.6), x1 - 1.2, r.uniform(2.2, 3.4)):
        window(bl, r, box(a, a + r.uniform(0.7, 1.0), BASE + 4.0, BASE + 4.85, d0))


def window(bl, r, q):
    """A window in a white wall: a dark opening, or a shutter, with its lattice cut through it."""
    bl.clear(q)
    name = "dark" if r.random() < 0.6 else "wood"
    bl.area(name, q, fill=int(r.uniform(130, 210)))
    (a0, y0, d), (a1, y1) = q[0], q[2][:2]
    for a in np.linspace(a0, a1, int(r.integers(3, 5)))[1:-1]:
        bl.key([(a, y0, d), (a, y1, d)], r, 0.3, name=name, fill=0, breaks=0.0, amp=0.1)
    bl.key([(a0, (y0 + y1) / 2, d), (a1, (y0 + y1) / 2, d)], r, 0.3, name=name, fill=0, breaks=0.0, amp=0.1)


def side_canal(bl, r):
    """The side canal coming in under the bridge: its water, and the stone end where it turns away."""
    a, b = BX - 3.2, BX + 3.2
    bl.clear(water := [(a, 0, 0), (b, 0, 0), (b, 0, 11), (a, 0, 11)])
    bl.area("water", water)
    end = [(a, BASE, 11), (b + 4, BASE, 11), (b + 4, BASE, 13.5), (a, BASE, 13.5)]
    bl.clear(end)
    bl.flat("stone", end, 0.15)
    face = box(a, b + 4, 0, BASE, 11)
    bl.clear(face)
    bl.ramp("stone", face, face[2:], face[:2], 0.35, 0.7)


def bank(bl, r, x0, x1, stair=None):
    """The stone bank along the far side of the canal, damp and darker toward the water, its courses cut
    as a few broken lines, the pale flags of the river street on it, and a flight of steps down to the
    water where the boats tie up."""
    walk = [(x0, BASE, 0.0), (x1, BASE, 0.0), (x1, BASE, WALK), (x0, BASE, WALK)]
    bl.clear(walk)
    bl.flat("stone", walk, 0.12)
    face = box(x0, x1, 0.0, BASE, 0.0)
    bl.clear(face)
    bl.ramp("stone", face, face[2:], face[:2], 0.3, 0.75, g=1.6)
    bl.key(face[3:1:-1], r, 0.8, breaks=0.01)
    for y in (0.4, 0.75):
        for a in np.arange(x0 + r.uniform(0, 3), x1 - 1, r.uniform(3, 6)):
            if r.random() < 0.5:
                bl.key([(a, y + 0.03 * r.normal(), 0.0), (min(a + r.uniform(1, 3), x1), y + 0.03 * r.normal(), 0.0)],
                       r, 0.45, breaks=0.03)
    if stair:
        for k in range(4):                              # each step a slab laid proud of the bank
            y0, y1 = BASE * (3 - k) / 4, BASE * (4 - k) / 4
            st = box(stair + 0.45 * k, stair + 2.4, y0, y1, -0.5)
            bl.clear(st)
            bl.ramp("stone", st, st[2:], st[:2], 0.18, 0.4)
            bl.key(st[3:1:-1], r, 0.6)


def bridge(bl, r):
    """A single arch of stone over the mouth of the side canal, its ramps stepping down onto the river
    street on either side. The arch and its reflection close into a ring in the water."""
    Lb, R, crown, c0 = 7.4, 2.6, 4.7, 0.25
    top = lambda x: BASE + 0.45 + (crown - BASE) * np.cos(np.pi / 2 * np.clip(np.abs(x - BX) / Lb, 0, 1)) ** 0.8
    xs = BX + np.linspace(-Lb, Lb, 90)
    near, far = np.c_[xs, top(xs), np.zeros(90)], np.c_[xs, top(xs), np.full(90, WALK - 0.4)]
    deck = np.vstack([near, far[::-1]])
    bl.clear(deck)
    bl.flat("stone", deck, 0.2)
    bl.key(far, r, 0.6, breaks=0.01)
    for x in np.r_[BX - np.arange(1.6, Lb - 0.2, 0.42), BX + np.arange(1.6, Lb - 0.2, 0.42)]:
        bl.key([(x, top(x), 0.2), (x, top(x), WALK - 0.5)], r, 0.4, breaks=0.04, amp=0.2)     # the steps
    figure(bl, r, (BX - 1.4, top(BX - 1.4) - 0.4, 0.9), 1.6, "basket")
    th = np.linspace(0, np.pi, 50)
    arch = np.c_[BX + R * np.cos(th), c0 + R * np.sin(th), np.zeros(50)]
    face = np.vstack([near, [(BX + Lb, BASE, 0), (BX + 3.4, BASE, 0), (BX + 3.4, 0, 0), (BX + R, 0, 0)], arch,
                      [(BX - R, 0, 0), (BX - 3.4, 0, 0), (BX - 3.4, BASE, 0), (BX - Lb, BASE, 0)]])
    bl.clear(face)
    bl.ramp("stone", face, [(BX - 9, crown + 0.5, 0), (BX + 9, crown + 0.5, 0)], [(BX - 9, 0, 0), (BX + 9, 0, 0)],
            0.22, 0.55, g=1.3)
    bl.key(near, r, 0.8, breaks=0.004)
    bl.key(near - [0, 0.45, 0], r, 0.5, breaks=0.02)
    hole = np.vstack([[(BX + R, 0, 0)], arch, [(BX - R, 0, 0)]])
    bl.clear(hole)
    bl.area("dark", hole, fill=120)
    bl.ramp("stone", hole, [(BX - R, c0 + R, 0), (BX + R, c0 + R, 0)], [(BX - R, 0, 0), (BX + R, 0, 0)], 1.0, 0.55, g=0.8)
    for rad in (R, R + 0.38):
        bl.key(np.c_[BX + rad * np.cos(th), c0 + rad * np.sin(th), np.zeros(50)], r, 0.7, breaks=0.004)
    for t in np.linspace(0.12, np.pi - 0.12, 15) + r.normal(0, 0.02, 15):
        bl.key([(BX + R * np.cos(t), c0 + R * np.sin(t), 0), (BX + (R + 0.38) * np.cos(t), c0 + (R + 0.38) * np.sin(t), 0)],
               r, 0.4, breaks=0.0, amp=0.1)


def figure(bl, r, foot, height, kind):
    """A figure no taller than a finger joint, seen from behind or aside: a jacket, dark trousers, and a
    head of dark hair, or the black felt hat of the boatmen of these canals."""
    def at(pts):
        p = np.asarray(pts, float) * height
        return np.c_[foot[0] + p[:, 0], foot[1] + p[:, 1], np.full(len(p), foot[2])]

    def oval(cx, cy, rx, ry, n=14):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]
    j = lambda pts: np.asarray(pts, float) + r.normal(0, 0.006, np.shape(pts))
    lean = 0.12 if kind == "scull" else 0.0
    parts = [("dark", j([[-0.1, 0], [-0.02, 0], [0.0, 0.47], [-0.1, 0.47]])),
             ("dark", j([[0.02, 0], [0.1, 0.0], [0.1, 0.47], [0.0, 0.47]])),
             ("coat", j([[-0.13, 0.42], [0.13, 0.42], [0.11 + lean, 0.8], [0.05 + lean, 0.84], [-0.05 + lean, 0.84],
                         [-0.11 + lean, 0.8]])),
             ("dark", oval(lean, 0.91, 0.062, 0.072))]
    if kind == "scull":
        parts += [("dark", oval(lean, 0.955, 0.1, 0.035)), ("dark", oval(lean, 0.98, 0.065, 0.045))]
    if kind == "basket":
        parts += [("straw", j([[0.1, 0.42], [0.3, 0.42], [0.28, 0.55], [0.12, 0.55]]))]
    for name, q in parts:
        bl.clear(at(q))
        bl.area(name, at(q), fill=230)


def boat(bl, r, x0, dc, length, beam, awnings, sculler=False):
    """A wupeng boat side on, long and low and pointed at both ends, under arched awnings of bamboo matting
    blacked with soot and tung oil. A boatman stands at the stern and sculls it along with a long oar."""
    def at(u, side, y):
        u, side, y = np.broadcast_arrays(*(np.asarray(v, float) for v in (u, side, y)))
        half = beam / 2 * np.sin(np.pi * np.clip(u, 0.002, 0.998)) ** 0.45
        return np.stack([x0 + u * length, y, dc + side * half], -1)
    sheer = lambda u: 0.32 + 0.4 * (2 * np.asarray(u, float) - 1) ** 6
    u = np.linspace(0, 1, 50)
    deck = np.vstack([at(u, -1, sheer(u)), at(u[::-1], 1, sheer(u[::-1]))])
    bl.clear(deck)
    bl.area("hull", deck, fill=110)
    side = np.vstack([at(u, -1, sheer(u)), at(u[::-1], -0.9, 0.0)])
    bl.clear(side)
    bl.area("hull", side, fill=225)
    bl.key(at(u, -1, sheer(u)), r, 0.7)
    bl.key(at(u, 1, sheer(u)), r, 0.5)
    bl.key(at(u, -0.9, 0.03), r, 0.5, breaks=0.03)
    ph = np.linspace(0, np.pi, 16)
    for ua, ub in awnings:
        arc = lambda uu: at(uu, -np.cos(ph), sheer(uu) + 0.05 + 0.62 * np.sin(ph))
        q = np.vstack([bl.P(arc(ua)), bl.P(arc(ub))])
        q = q[ConvexHull(q).vertices]
        bl.clear(q, screen=True)
        bl.area("dark", q, fill=225, screen=True)
        bl.key(arc(ua), r, 0.5)
    if sculler:
        foot = at(0.92, 0, sheer(0.92))
        figure(bl, r, foot, 1.65, "scull")
        bl.key([foot + [0.1, 1.15, 0], at(1.3, 0, 0.0)], r, 0.9, name="dark", breaks=0)


def tress(bl, r, c, length, w0, sway):
    """One hanging tress of the willow, cut as a long tongue on the pale block, often with a narrower
    one on the deeper block inside it, and parted by a cut or two down its length."""
    t = np.linspace(0, 1, 36)
    mid = c[0] + sway * t ** 1.6 + 0.03 * length * noise.line1d(36, 18, r) * t
    w = w0 * np.clip(1 - t, 0, 1) ** 0.8 * (1 + 0.15 * np.sin(np.pi * t))
    y, d = c[1] - length * t, np.full(36, c[2])
    bl.area("willow", np.vstack([np.c_[mid - w / 2, y, d], np.c_[mid + w / 2, y, d][::-1]]), fill=int(r.uniform(150, 235)))
    if r.random() < 0.6:
        n = int(r.uniform(0.55, 0.9) * 36)
        o = r.uniform(-0.2, 0.2) * w0
        bl.area("leaf", np.vstack([np.c_[mid[:n] + o - 0.3 * w[:n], y[:n], d[:n]], np.c_[mid[:n] + o + 0.3 * w[:n], y[:n], d[:n]][::-1]]),
                fill=int(r.uniform(110, 200)))
    if w0 > 0.35:
        f, i0, i1 = r.uniform(-0.2, 0.2), int(r.integers(2, 9)), int(r.integers(22, 32))
        bl.key(np.c_[mid + f * w, y, d][i0:i1], r, 0.45, name="willow", fill=0, breaks=0.02, amp=0.3)


def willow(bl, r, x, d, height, lean, reach):
    """A weeping willow in new leaf on the river street, leaning out over the water. Its colour is brushed
    onto blocks cut in long hanging tongues, a pale yellow-green for the mass and a deeper green for the
    strands inside it, with a few cuts down them to part the strands; the damp sheet feathers every edge.
    The trunk and its arching boughs are printed dark, and thin, from a block of their own."""
    base, top = np.array([x, BASE, d]), np.array([x + lean, BASE + height, d - reach])
    tt = np.linspace(0, 1, 24)[:, None]
    spine = base + tt ** 0.85 * (top - base) + np.sin(np.pi * tt) * [-0.25 * lean, 0, 0]
    wid = (0.3 - 0.2 * tt) * [1, 0, 0]
    bark = np.vstack([spine - wid, (spine + wid)[::-1]])
    bl.clear(bark)
    bl.area("dark", bark, fill=200)
    bl.key(spine[3:] + 0.4 * wid[3:], r, 0.4, name="dark", fill=0, breaks=0.05)      # a split in the bark
    for _ in range(int(r.integers(16, 21))):
        side = np.sign(lean) * (1 if r.random() < 0.65 else -1)
        start = spine[int(r.integers(12, 24))]
        out = np.array([side * r.uniform(2.0, 5.0), r.uniform(0.4, 1.4), r.uniform(-1.0, 0.8)])
        ss = np.linspace(0, 1, 12)[:, None]
        bough = start + ss * out - ss ** 2 * [0, r.uniform(0.8, 1.8), 0]
        bl.key(bough[:6], r, 0.5, name="dark", fill=150, breaks=0.0, amp=0.3)
        for c in bough[r.choice(np.arange(2, 12), int(r.integers(4, 7)), replace=False)]:
            tress(bl, r, c, min(r.uniform(0.45, 1.0) * (c[1] - 0.3), r.uniform(2.5, 6.5)), r.uniform(0.3, 0.75),
                  0.25 * side * r.uniform(0, 1))


def peach(bl, r, x, d, height):
    """A peach tree in flower on the river street: a crooked trunk, thin branches forking upward, and the
    blossom in loose clusters along the twigs, printed in two pinks, the paler brushed on wet so that it
    spreads a little in the damp sheet."""
    def grow(p, dirn, length, w, depth):
        tt = np.linspace(0, 1, 10)[:, None]
        path = p + tt * dirn * length + np.sin(np.pi * tt) * np.array([r.normal(0, 0.12), 0, 0]) * length
        bl.key(path, r, max(0.4, w * S / 2), name="dark", fill=200, breaks=0.0, amp=0.3)
        if depth <= 1:
            for c in path[2::3]:
                bloom(c)
        if depth == 0:
            return
        for _ in range(int(r.integers(2, 4))):
            a = np.clip(np.arctan2(dirn[1], dirn[0]) + r.uniform(-0.9, 0.9), 0.15, np.pi - 0.15)
            grow(path[int(r.integers(4, 10))], np.array([np.cos(a), np.sin(a), 0.0]), length * r.uniform(0.55, 0.8),
                 w * 0.62, depth - 1)

    def bloom(c):
        for _ in range(int(r.integers(1, 3))):
            o = bl.P(c + r.normal(0, 0.15, 3) * [1, 1, 0])[0]
            rad = r.uniform(0.08, 0.15) * S * SS
            a = np.linspace(0, 2 * np.pi, 13, endpoint=False)
            bl.area("peach", o + np.c_[np.cos(a), np.sin(a)] * (rad * (1 + 0.25 * np.cos(5 * a + r.uniform(0, 7))))[:, None],
                    fill=int(r.uniform(130, 255)), screen=True)
            if r.random() < 0.45:
                bl.area("blush", o + np.c_[np.cos(a), np.sin(a)] * rad * r.uniform(0.25, 0.45), fill=200, screen=True)
    grow(np.array([x, BASE, d]), np.array([0.15, 1.0, 0.0]), height * 0.4, 0.12, 3)


def scene(r):
    """Everything to be cut, back to front, each with the waterline it is turned about in the water,
    its own seed, and how to cut it."""
    items = [(0.0, lambda bl, rr, h=h: cut_house(bl, rr, h)) for h in town(r)]
    items.insert(len(items) - sum(1 for k, _, _ in FRONT if k), (0.0, side_canal))
    items += [(0.0, lambda bl, rr: bank(bl, rr, X0 - 2, BX - 3.2, stair=9.5)),
              (0.0, lambda bl, rr: bank(bl, rr, BX + 3.2, X0 + 70)),
              (0.0, bridge),
              (0.0, lambda bl, rr: peach(bl, rr, BX + 5.6, 1.2, 5.2)),
              (0.0, lambda bl, rr: willow(bl, rr, 2.2, 1.0, 8.6, 4.2, 3.0)),
              (-2.0, lambda bl, rr: boat(bl, rr, 6.5, -2.0, 7.0, 1.6, [(0.32, 0.5), (0.53, 0.7)])),
              (-15.0, lambda bl, rr: boat(bl, rr, 14.5, -15.0, 7.5, 1.7, [(0.3, 0.5), (0.53, 0.72)], sculler=True))]
    return [(dref, int(r.integers(1 << 30)), fn) for dref, fn in items]


def ripples(r, wake):
    """Ripples cut across the water with single pushes of the knife: few, for the water is still, more in
    the wake of the moving boat, and longer and further apart toward us."""
    out = []
    breeze = noise.field((SH, SW), 300, r)
    y = M + Y0 + 6
    while y < M + H:
        d = y - M - Y0
        x = M - r.uniform(0, 200)
        while x < M + W:
            L = (10 + 0.1 * d) * r.lognormal(0, 0.5)
            if r.random() < 0.8 * noise.smoothstep(0.0, 1.6, breeze[int(y), int(np.clip(x + L / 2, 0, SW - 1))]):
                tt = np.linspace(0, 1, max(6, int(L / 3)))
                out.append((np.c_[x + L * tt, y + 0.3 * r.normal() * np.sin(np.pi * tt)],
                            (0.55 + 0.004 * d) * r.lognormal(0, 0.3) * np.sin(np.pi * tt) ** 0.7))
            x += L + (24 + 0.4 * d) * r.lognormal(0, 0.6)
        y += (4 + 0.03 * d) * r.uniform(0.6, 1.5)
    if wake is not None:
        (sx, sy) = wake
        for i in range(1, 9):                           # the wake opening out behind the stern
            for side in (-1, 1):
                cx, cy, L = sx + 26 * i, sy + side * 4.5 * i, 14 + 3 * i
                tt = np.linspace(0, 1, 12)
                out.append((np.c_[cx - L / 2 + L * tt, cy + 0.6 * side * np.sin(np.pi * tt)],
                            (1.1 - 0.1 * i) * np.sin(np.pi * tt) ** 0.6))
    return relief.cuts((SH, SW), out)


def board(r, period=13.0):
    """The face of a plywood block as the colour finds it: growth lines waving across the veneer, crowding
    and spreading and fading in and out, with the finer streaks of the fibre between them."""
    rows = np.arange(SH, dtype=np.float32)[:, None] + 45 * noise.fbm((SH, SW), 700, r, octaves=3) + 2 * noise.field((SH, SW), 30, r)
    rings = np.cumsum(np.clip(1 + 0.5 * noise.line1d(SH + 600, 80, r), 0.25, None)) / period
    ph = np.interp(rows + 300, np.arange(SH + 600), rings)
    g = np.exp(-((ph % 1 - 0.5) / 0.15) ** 2) * noise.smoothstep(-1.0, 1.2, noise.stretched((SW, SH), 12, 600, r).T)
    g = g + 0.3 * noise.stretched((SW, SH), 2.0, 220, r).T
    return (g - g.mean()) / g.std()


def brushed(stops, r, waver=10.0, streak=0.02):
    """A gradation brushed onto the block: (row, amount) stops; its edge wavers across the board, and the
    brush leaves faint streaks along its strokes, most where it was half loaded."""
    t = np.arange(SH, dtype=np.float32)[:, None] - M + waver * (noise.line1d(SW, 280, r)[None, :] + 0.3 * noise.field((SH, SW), 70, r))
    a = np.full((SH, SW), stops[0][1], np.float32)
    for (p0, v0), (p1, v1) in zip(stops, stops[1:]):
        a += (v1 - v0) * noise.smoothstep(p0, p1, t)
    a *= 1 + streak * noise.stretched((SW, SH), 14, 420, r).T * (0.4 + 2.4 * np.clip(a * (1 - a), 0, 0.25))
    return np.clip(a * (1 + 0.06 * noise.fbm((SH, SW), 260, r, octaves=3)), 0, None)


def press(img, sheet, pits, board, r, block, k, amount=1.0, grain=0.15, wet=0.5, spread=3.0, halo=0.25, edge=0.2,
          whiskers=0.5, salt=0.25, loose=0.8, pool=None, feather=0.0):
    """One impression. Colour is brushed onto the block, the dampened xuan laid over it and rubbed. Thin
    colour misses the valleys of the sheet, and a flat is salted where the sheet did not quite meet it;
    the damp sheet draws a little of the colour out past the edge of the cut, along its fibres, and it
    dries there with a faint edge."""
    ys, xs = np.nonzero(block > 0.004)
    if len(ys) == 0:
        return img
    pad = int(3 * spread) + 12
    b = slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, SH)), slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, SW))
    sh = paper.Sheet(sheet.color[b], sheet.tooth[b], sheet.fiber[b], sheet.alpha[b])
    blk, amt = block[b], np.broadcast_to(np.asarray(amount, np.float32), block.shape)[b]
    if loose:                                           # a colour block is cut by eye from a tracing, never to the hair
        blk = noise.warp(blk, loose * noise.field(blk.shape, 140, r), loose * noise.field(blk.shape, 140, r))
    if feather:                                         # the colour brushed on thinner toward the edges of the block
        reach = ndimage.distance_transform_edt(blk > 0.5) / (feather * np.clip(1 + 0.5 * noise.field(blk.shape, 25, r), 0.3, None))
        amt = amt * noise.smoothstep(0.0, 1.0, reach)
    if pool:                                            # the wet colour pools and dries unevenly in the flats
        wet_ = (blk > 0.05).astype(np.float32)
        amt = amt * np.where(wet_ > 0, watercolour.deposit(wet_, sh, r, **pool), 1.0)
    film = relief.ink_film(blk, r, roller=0.0, squash=0.06, grain=board[b], grain_strength=grain) * amt
    dens = relief.pull(blk, film, sh, r, pressure=1.0, shift=tuple(r.normal(0, 0.9 if loose else 0.3, 2)))
    dens *= 1 - salt * np.clip(1.3 - amt, 0, 1) * pits[b]
    dens = ink.bleed(dens, noise.smoothstep(0.01, 0.12, dens), sh, wet=wet, spread=spread, halo=halo, edge=edge,
                     soften=0.7, mottle=0.2, whiskers=whiskers, seed=int(r.integers(1 << 30)))
    img[b] = glaze(img[b], dens, k)
    return img


def paint(seed=1980):
    r = noise.rng(seed)
    sheet = paper.washi((SH, SW), seed, tint="#f4f0e6", margin=(40, 38), fibre_density=0.35)
    sheet.alpha = paper.deckle((SH, SW), (40, 38), r, ragged=2.5)
    pits = noise.smoothstep(0.32, 0.14, sheet.tooth)
    boards = [board(r) for _ in range(2)]
    yy = np.arange(SH, dtype=np.float32)[:, None] - M
    xx = np.arange(SW, dtype=np.float32)[None, :] - M
    inside = (noise.smoothstep(-0.5, 0.5, yy) * noise.smoothstep(-0.5, 0.5, H - yy)
              * noise.smoothstep(-0.5, 0.5, xx) * noise.smoothstep(-0.5, 0.5, W - xx))

    real, mirror = Blocks(), Blocks(mirror=True)
    real.d["sky"].rectangle((M * SS, M * SS, (M + W) * SS, (M + H) * SS), fill=255)
    canal = [(X0 - 30, 0, 0), (X0 + 100, 0, 0), (X0 + 100, 0, -90), (X0 - 30, 0, -90)]
    real.clear(canal)
    real.area("water", canal)
    for dref, s, fn in scene(r):
        for bl in (real, mirror):
            bl.dref = dref
            fn(bl, noise.rng(s))
            bl.kt = 1.0

    def amounts(bl):
        a = {n: bl.get(n) * inside for n in NAMES}
        for n in bl.tone:
            a[n] *= bl.shade(n)
        return a
    A, Rf = amounts(real), amounts(mirror)
    rip = ripples(r, real.P([(22.3, 0.0, -15.0)])[0] / SS)
    # the water sways the reflections a little, row by row, more toward us
    sway = (0.4 + 1.6 * noise.smoothstep(Y0, H, yy)) * noise.line1d(SH, 14, r)[:, None] + 0.3 * noise.field((SH, SW), 6, r)
    still = A["water"] * (1 - rip)

    def refl(*parts):
        return noise.warp(sum(w * Rf[n] for n, w in parts), sway, 0 * sway) * still
    fres = 0.12 + 0.88 * noise.smoothstep(Y0 + 640, Y0 + 40, yy) ** 1.3     # the water mirrors best under the far bank
    img = sheet.color.copy()
    ink_ = lambda block, k, **kw: press(img, sheet, pits, boards[int(r.integers(2))], r, block, k, **kw)
    ink_(A["sky"], SKY, amount=brushed([(0, 0.7), (180, 0.35), (420, 0.06), (560, 0.0)], r), grain=0.12, spread=6)
    ink_(A["sky"], WARM, amount=brushed([(200, 0.0), (380, 0.32), (520, 0.3), (620, 0.0)], r), grain=0.1, spread=6)
    walls = noise.warp(Rf["wall"], sway, 0 * sway) * still * fres
    ink_(still, WATER, amount=brushed([(Y0, 0.2), (Y0 + 300, 0.32), (H, 0.6)], r) * (1 - 0.65 * walls),
         grain=0.2, spread=5, pool=dict(pool=0.08, rim=0.0, tides=0.1, grain=0.25, scale=250))
    ink_(refl(("roof", 0.45), ("cap", 0.4), ("dark", 0.4), ("key", 0.2), ("shade", 0.4)) * fres, ROOF, wet=0.9, spread=7,
         halo=0.35, loose=0)
    ink_(refl(("stone", 0.5), ("wood", 0.4), ("hull", 0.45)) * fres, STONE, wet=0.9, spread=7, loose=0)
    ink_(refl(("willow", 0.5), ("leaf", 0.4)) * fres, LEAF, wet=0.9, spread=8, loose=0)
    ink_(refl(("peach", 0.5), ("blush", 0.4)) * fres, PEACH, wet=0.9, spread=8, loose=0)
    ink_(A["stone"], STONE, grain=0.1, spread=3, pool=dict(pool=0.1, rim=0.2, tides=0.0, grain=0.35, scale=80))
    ink_(A["shade"], SHADE, grain=0.08, wet=0.7, spread=5)
    ink_(A["wood"], WOOD, grain=0.06, spread=2.5)
    ink_(A["hull"], HULL, grain=0.15, spread=2.5)
    ink_(A["willow"], WILLOW, grain=0.12, wet=0.95, spread=9, halo=0.45, loose=1.5, whiskers=0.9, feather=7,
         pool=dict(pool=0.35, rim=0.5, tides=0.2, grain=0.3, scale=40))
    ink_(A["peach"], PEACH, grain=0.06, wet=0.85, spread=5, halo=0.4, feather=3, pool=dict(pool=0.3, rim=0.4, grain=0.2, scale=20))
    ink_(A["roof"], ROOF, amount=1 + 0.08 * noise.fbm((SH, SW), 80, r, octaves=3), grain=0.09, spread=2.5, salt=0.45, pool=dict(pool=0.06, rim=0.3, rim_width=2.5, tides=0.0, grain=0.3, scale=90))
    ink_(A["cap"], CAP, amount=0.85, grain=0.08, spread=1.8, wet=0.4, loose=0)
    ink_(A["dark"], DARK, amount=0.85, grain=0.1, spread=2, wet=0.4)
    ink_(A["leaf"], LEAF, grain=0.06, wet=0.8, spread=5, halo=0.35, feather=5)
    ink_(A["blush"], BLUSH, wet=0.5, spread=2)
    ink_(A["coat"], COAT, wet=0.3, spread=1.5)
    ink_(A["straw"], STRAW, wet=0.3, spread=1.5)
    ink_(A["key"], KEY, grain=0.04, wet=0.3, spread=1.5, whiskers=0.4, edge=0.1, halo=0.15, loose=0)
    img = relief.emboss(img, [A["key"], A["roof"]], depth=0.01)
    return plate.mount(img, sheet, shadow=0.35)
