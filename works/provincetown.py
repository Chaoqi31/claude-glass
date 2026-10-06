"""Harbor, August. Acrylic thinned with water, poured into unprimed cotton duck, over charcoal.

In 1952 Helen Frankenthaler laid a length of raw cotton duck on the floor of her studio and poured
paint thinned with turpentine onto it from a coffee can. The colour did not lie on the canvas; it
sank into the weave and became the canvas, and the cloth left bare between the pours was as much a
part of the picture as the colour. In the 1960s she changed to acrylic thinned with water, which
stains cleaner and brighter, and the shapes grew larger and more open.

The canvas is laid flat and a few loose contours are drawn on it in charcoal. Then the paint is
poured, one colour at a time, from a can carried along the canvas: where the hand paused the paint
stood deep and spread far, where it walked on it left an arm, and where it hurried a thin rivulet.
Wet paint runs out through the weave until the cloth has drunk it, faster where the cloth is
thirstier and slower across the soft creases it lies in, so every edge is a front of lobes and
inlets, and points where it crept along a thread; the water runs out ahead of the pigment and
leaves a pale ring round each shape, darker where it stopped. The floor is not quite level, so the
paint standing on the cloth drained to the low side of each shape: it is thinner where it ran off
and darker below the line where its water stood, deepening further down, with a tide line where
the water's edge stood longest. A second pour into one that has begun to dry pushes the first
aside and leaves a pale ring and a dark line of it where its water stopped. Where the sizing lay
thick the cloth all but refused the paint, and the heavier pigments settle into the hollows of the
weave. Drops fly from the can as it swings, and one lands hard enough to splash. Where one colour
crosses another they glaze.
"""

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from atelier import brush, canvas, noise, pencil
from atelier.color import pigment

TITLE = "Harbor, August"
DATE = "2026"
MEDIUM = "Acrylic thinned with water, poured and stained into unprimed cotton duck, over charcoal"
AFTER = "Helen Frankenthaler, the soak-stain paintings, from Mountains and Sea (1952) to The Bay (1963)"
ROOM = "Colour Itself"
YEAR = 1963
PLACE = "New York"
REGION = "Americas"
NOTE = ("Ultramarine, cerulean, coral, sap green and a little ochre poured onto raw cotton duck laid on the "
        "floor, each shape left as the paint ran out, with the bare canvas between them.")

H, W = 2400, 3000
Q = 4                                   # px to a cell of the grid the paint's spread is worked out on
GH, GW = H // Q, W // Q
CANVAS = "#efe6d1"

PAINT = dict(ultramarine=pigment("#2a50d8"), cerulean=pigment("#3aa6d6"), coral=pigment("#ee6a5e"),
             sap=pigment("#8db25a"), ochre=pigment("#d9a23e"))
GRAIN = dict(ultramarine=1.0, cerulean=0.8, coral=0.25, sap=0.35, ochre=0.6)   # how readily each settles out
WATER = np.array([0.05, 0.065, 0.10], np.float32)       # cotton wetted with medium goes a little darker, warmer
CHARCOAL = pigment("#4a4643")
STEPS = ((0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1))
DOWN = np.array([-0.12, 0.3])           # how far the floor falls for a px across and a px down the canvas


def _s(r):
    return int(r.integers(1 << 31))


class Floor:
    """The canvas as the paint finds it on the floor: its sizing is uneven and it lies in a few soft
    creases, so wet paint runs further one way than another, and the floor under it is not level."""

    def __init__(self, r):
        g = (GH, GW)
        crease = np.exp(-(noise.field(g, 65, r) / 0.07) ** 2) * noise.smoothstep(-0.3, 0.6, noise.field(g, 100, r))
        thread = np.exp(-(noise.field(g, 28, r) / 0.12) ** 2) * noise.smoothstep(0.0, 1.0, noise.field(g, 75, r))
        # a crease holds the front back and leaves an inlet; a thirsty run of thread draws it out in a point
        speed = np.exp(0.2 * noise.fbm(g, 120, r, octaves=3) + 0.28 * noise.field(g, 25, r)
                       + 0.1 * noise.field(g, 8, r) - 1.4 * crease + 0.5 * thread)
        idx = np.arange(GH * GW).reshape(GH, GW)
        a, b, c = [], [], []
        for dy, dx in STEPS:
            x0, x1 = max(0, -dx), GW - max(0, dx)
            s0, s1 = speed[:GH - dy, x0:x1], speed[dy:, x0 + dx:x1 + dx]
            along = 0.95 if dy == 0 or dx == 0 else 1.0     # wet cotton wicks a little faster along its threads
            a.append(idx[:GH - dy, x0:x1].ravel())
            b.append(idx[dy:, x0 + dx:x1 + dx].ravel())
            c.append((along * np.hypot(dy, dx) * 0.5 * (1 / s0 + 1 / s1)).ravel())
        self.edges = [np.concatenate(v) for v in (a, b, c)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.low = DOWN[0] * xx + DOWN[1] * yy + 50 * noise.fbm((H, W), 700, r, octaves=3) + 12 * noise.field((H, W), 150, r)

    def reach(self, src):
        """How much further than each point the paint poured at rows (x, y, R) could have run, in px: R,
        less the way it had to go to get there, from whichever source served it best."""
        cell = (np.clip(np.round(src[:, 1] / Q), 0, GH - 1) * GW + np.clip(np.round(src[:, 0] / Q), 0, GW - 1))
        best = np.full(GH * GW, -1.0)
        np.maximum.at(best, cell.astype(int), src[:, 2] / Q)
        node = np.flatnonzero(best > 0)
        top = best[node].max()
        a, b, c = self.edges
        n = GH * GW
        G = sparse.coo_matrix((np.concatenate([c, top - best[node] + 1e-3]),
                               (np.concatenate([a, np.full(len(node), n)]), np.concatenate([b, node]))),
                              shape=(n + 1, n + 1)).tocsr()
        # clamped, not cut: a step here would ring through the spline into the soft fronts
        d = dijkstra(G, directed=False, indices=n, limit=top + 40)[:n].reshape(GH, GW)
        rc = top - np.minimum(d, top + 40)
        return Q * ndimage.zoom(rc, Q, order=3, grid_mode=True, mode="nearest")


def pour(P):
    """Rows (x, y, R) every cell along the paths the can was carried through, each given by control rows
    (x, y, R): R is how far the paint that landed there could run, which is how much of it there was."""
    P = [P] if np.ndim(P[0]) == 1 else P
    return np.vstack([brush.path(np.asarray(p, np.float64), Q) for p in P])


def points(R, r, n, halo):
    """Where the paint crept out along a thread past the rest of its front: n short tapering points
    from places on the front of a broad body of paint, most of them square to the weave, each with
    pigment to its tip."""
    small = R[::8, ::8]
    gy, gx = np.gradient(ndimage.gaussian_filter(small, 2))
    ys, xs = np.nonzero((np.abs(small) < 5) & (ndimage.grey_dilation(small, size=15) > 100))
    rows = []
    for k in r.choice(len(xs), min(n, len(xs)), replace=False):
        y, x = ys[k], xs[k]
        a = np.arctan2(-gy[y, x], -gx[y, x])
        if r.uniform() < 0.7:
            a = np.round(a / (np.pi / 2)) * (np.pi / 2) + r.normal(0, 0.1)
        L, w0 = r.uniform(25, 70), r.uniform(16, 26)
        t = np.linspace(0, 1, 6)
        rows.append(pour(np.stack([8 * x + np.cos(a) * (t * L - 14), 8 * y + np.sin(a) * (t * L - 14),
                                   1.5 * halo + w0 * (1 - t) ** 0.7 + 2], 1)))
    return np.vstack(rows) if rows else np.zeros((0, 3))


def fling(x, y, ang, fan, far, n, size, r):
    """Drops thrown from the can at (x, y) as it swung: n of them fanned `fan` radians about `ang` out
    to `far` px, the heavier landing nearer, the further ones drawn out along their flight.
    -> rows (x, y, radius, heading, stretch)"""
    t = r.uniform(0.1, 1, n)
    a = ang + fan * r.uniform(-0.5, 0.5, n)
    return np.stack([x + far * t * np.cos(a), y + far * t * np.sin(a), size * r.uniform(0.4, 1, n) * (1.2 - 0.7 * t),
                     a, 1 + 2 * t * r.uniform(0.3, 1, n)], 1)


class Box:
    """The colours as they went into the canvas, kept as a density per pigment, and the medium that
    went with them; glazed onto the bare duck at the end, since stains on stains multiply."""

    def __init__(self, sheet, floor, r):
        self.sheet, self.floor, self.r = sheet, floor, r
        self.d = {k: np.zeros((H, W), np.float32) for k in PAINT}
        self.water = np.zeros((H, W), np.float32)
        self.sealed = np.zeros((H, W), np.float32)          # where earlier medium has already sealed the cloth
        self.wet = np.zeros((H, W), np.float32)
        top = ndimage.gaussian_filter(sheet.tooth, 0.8)
        self.valley = (top.mean() - top) / top.std()        # the hollows between the threads
        self.clot = noise.field((H, W), 2.5, r)
        self.bald = noise.smoothstep(1.5, 1.75, noise.fbm((H, W), 170, r, octaves=4))   # where the sizing lay thick
        # the cloth drinks more in some places than others
        self.take = ((1 + 0.05 * noise.fbm((H, W), 120, r, octaves=3))
                     * (1 + 0.02 * noise.stretched((W, H), 3, 200, r).T + 0.02 * noise.stretched((H, W), 3, 200, r)))

    def drain(self, a):
        """How the paint standing on the cloth over `a` drained to the low side of the floor and dried
        there: thinner where it ran off, darker below the line where its water stood and deeper still
        further down, with a tide line where the water's edge stood longest. -> factor on the density"""
        low = self.floor.low
        m, lo = a[::8, ::8], low[::8, ::8]
        # the water stands level with the floor round about it, so it lies deep where the floor falls below that
        d = low - ndimage.zoom(ndimage.gaussian_filter(lo * m, 30) / (ndimage.gaussian_filter(m, 30) + 1e-3), 8,
                               order=1)
        v = d[a > 0.5][::31]
        if v.size < 100:
            return 1.0
        # the water's edge caught on the threads as it drew back, so it wanders
        e = d - np.quantile(v, 0.55) + 8 * noise.field((H, W), 60, self.r) + 3 * noise.field((H, W), 20, self.r) \
            + 1.2 * noise.field((H, W), 5, self.r)
        mark = np.clip(0.2 + 0.8 * noise.field((H, W), 150, self.r), 0, 1)     # no tide line is whole
        ep = np.clip(e, 0, None)
        return 0.7 + noise.smoothstep(-1.5, 1, e) * (0.32 + 0.15 * (1 - np.exp(-ep / 40)) + 0.3 * mark * np.exp(-ep / 4))

    def stain(self, pours, halo=6.0, drops=(), runs=3):
        """One colour poured: `pours` [(paths, load, {pigment: share})], in the order they went down.
        The shape as a whole ends where the last of the water stopped, with `drops` (rows from `fling`)
        thrown beyond it and `runs` points creeping past the rest. Each pour drains to the low side of
        the floor, the heavier later ones less. Every pour after the first went into paint that had
        begun to dry: its water pushes the paint already there aside with a frilled front, leaving a
        pale ring and a dark line of it where the water stopped, and its own colour ends in a darker
        edge."""
        r = self.r
        reach = []
        for i, (P, _, _) in enumerate(pours):
            frill = 1.2 + 2.5 * (i > 0)
            reach.append(self.floor.reach(pour(P)) + 3 * noise.field((H, W), 60, r)
                         + frill * (noise.field((H, W), 26, r) + 0.4 * noise.field((H, W), 7, r)))
        R = np.max(reach, axis=0)
        hw = (halo * np.exp(0.7 * noise.field((H, W), 350, r))).astype(np.float32)
        for x, y, rad, a, s in drops:
            e = rad * s + 40
            sl = (slice(int(np.clip(y - e, 0, H)), int(np.clip(y + e, 0, H))),
                  slice(int(np.clip(x - e, 0, W)), int(np.clip(x + e, 0, W))))
            yy, xx = np.mgrid[sl]
            u, v = (xx - x) * np.cos(a) + (yy - y) * np.sin(a), (yy - y) * np.cos(a) - (xx - x) * np.sin(a)
            R[sl] = np.maximum(R[sl], hw[sl] + 3 * (rad - np.hypot(u / np.where(u > 0, s, 1), v)))
        tips = points(R, r, runs, halo)
        if len(tips):
            R = np.maximum(R, self.floor.reach(tips))
        c = R[::4, ::4]         # where two wet fronts met they ran together
        R += ndimage.zoom(ndimage.grey_closing(c, footprint=np.hypot(*np.mgrid[-4:5, -4:5]) <= 4) - c, 4, order=1)
        R += (0.2 + 0.8 * noise.smoothstep(0.0, 1.5, noise.field((H, W), 500, r))) * noise.field((H, W), 12, r)
        # the water runs out through the threads on its own way, a little ahead here and a lot there
        wet = noise.smoothstep(-0.8, 0.8, R + 0.5 * hw * noise.field((H, W), 70, r)
                               + 1.4 * (self.sheet.tooth - 0.22) + 0.5 * noise.field((H, W), 3, r))
        pig = noise.smoothstep(hw - 1.2, hw + 1.2,
                               R + 0.9 * (self.sheet.tooth - 0.22) + 0.4 * noise.field((H, W), 2.5, r))
        inner = np.clip(R - hw, 0, None)
        C = {k: np.zeros((H, W), np.float32) for p in pours for k in p[2]}
        got = np.zeros((H, W), np.float32)
        deep = np.zeros((H, W), np.float32)     # where paint stood, and the heavy pigments had time to settle
        for i, ((P, load, mix), ri) in enumerate(zip(pours, reach)):
            a = noise.smoothstep(0.6 * hw - 1.5, 0.6 * hw + 1.5, ri)
            f = self.drain(a)
            if not i:
                f = f * (1 - 0.7 * self.bald * (f < 0.8))   # drained off a thickly sized place, it all but went
                deep = np.maximum(deep, a * (f > 0.8))
            else:
                rw = 3 + 6 * noise.smoothstep(-1, 1, noise.field((H, W), 90, r))     # how far its water ran ahead
                ring = noise.smoothstep(-rw - 1.5, -rw + 1.5, ri) * (1 - a)
                line = np.exp(-((ri + rw) / 2.0) ** 2) * got
                keep = (1 - 0.7 * a) * (1 - 0.35 * ring) * (1 + 0.6 * line)
                for k in C:
                    C[k] *= keep
                # heavier, it drained less, and its own colour ends in a darker edge
                f = (0.65 + 0.5 * f) * (1 + 0.35 * np.exp(-np.clip(ri - 0.6 * hw, 0, None) / 4))
                deep = np.maximum(deep, a)
            for k, share in mix.items():
                C[k] += a * load * share * f
            got = np.maximum(got, a)
        _, load, mix = pours[0]
        for k, share in mix.items():                              # drops and points carry the first pour
            C[k] += (1 - got) * load * share
        rim = (0.15 + 0.35 * noise.smoothstep(-1, 1, noise.field((H, W), 120, r))) * np.exp(-inner / 3.0)
        whisper = 0.05 * (wet - pig) * noise.smoothstep(0, hw, R)
        seal = 1 - 0.3 * self.sealed
        body = pig * (self.take + rim) * seal
        for k, share in mix.items():
            self.d[k] += whisper * load * share * seal
        for k in C:
            grain = 1 + (0.03 + GRAIN[k] * (0.05 + 0.1 * deep)) * self.valley + GRAIN[k] * (0.03 + 0.05 * deep) * self.clot
            self.d[k] += C[k] * body * np.clip(grain, 0, None)
        self.water += 0.6 * wet + 1.4 * wet * np.exp(-np.clip(R, 0, None) / 1.8)
        self.sealed = np.maximum(self.sealed, wet - pig)
        self.wet = np.maximum(self.wet, wet)

    def glaze(self, char):
        char = char * (1 - 0.4 * self.wet) + 0.4 * self.wet * ndimage.gaussian_filter(char, 2.0)
        a = self.water[..., None] * WATER + char[..., None] * CHARCOAL
        for k, K in PAINT.items():
            a = a + self.d[k][..., None] * K
        return self.sheet.color * np.exp(-a)


def drawing(sheet, r):
    """The charcoal drawn on the bare canvas before anything was poured: a few loose contours, which
    the paint later followed for a while, or went its own way across."""
    press = np.zeros((H, W), np.float32)
    lines = [([(2250, 560), (1900, 680), (1550, 800), (1200, 830), (900, 790), (650, 640)], 0.85),
             ([(60, 2100), (600, 2040), (1200, 2090), (1700, 2150), (1950, 2060)], 0.8),
             ([(1150, 1380), (1500, 1280), (1900, 1320), (2150, 1480), (2250, 1650)], 0.8),
             ([(100, 450), (450, 330), (850, 330), (1100, 450)], 0.75),
             ([(2350, 1050), (2650, 980), (2900, 1060), (2990, 1250)], 0.75)]
    for P, p in lines:
        pencil.line(press, np.array(P, np.float64), 10.0, r, pressure=p, wander=3.0, tremor=0.4, lift=(300, 900))
    # charcoal dust settles round the threads it catches on
    return 0.6 * ndimage.gaussian_filter(pencil.catch(press, sheet, grip=1.2, soft=0.3), 1.2)


def paint(seed=1963):
    r = noise.rng(seed)
    sheet = canvas.duck((H, W), _s(r), tint=CANVAS, thread=3.0)
    floor = Floor(r)
    box = Box(sheet, floor, r)
    char = drawing(sheet, r)
    # the coral at the left, its tail run down toward the middle
    box.stain([([(-100, 1150, 230), (200, 950, 290), (500, 820, 300), (850, 780, 250), (1100, 830, 130),
                 (1250, 950, 45), (1300, 1080, 20)], 0.4, dict(coral=1.0)),
               ([(100, 900, 120), (400, 760, 150), (700, 700, 110)], 0.9, dict(coral=1.0))],
              halo=6, drops=fling(1100, 600, -0.6, 0.6, 320, 10, 12, r), runs=8)
    box.stain([([(1120, 640, 45), (1220, 600, 70), (1320, 610, 45)], 0.8, dict(ochre=1.0))],
              halo=4, drops=fling(1330, 590, -0.3, 0.9, 200, 6, 9, r), runs=0)
    # the ultramarine: a long pause at the top right, then carried down across the coral in an arm that
    # runs on into the water, and a rivulet dribbled down the right
    box.stain([([[(2950, 100, 160), (2650, 300, 290), (2300, 520, 320), (2000, 780, 240), (1750, 930, 120),
                  (1500, 960, 175), (1250, 930, 110), (1000, 900, 150), (750, 950, 85), (560, 1080, 110),
                  (400, 1300, 50), (300, 1540, 36)],
                 [(2350, 300, 150), (2100, 120, 220), (1800, 0, 200), (1500, -80, 150)],
                 [(2050, 900, 26), (2120, 1080, 18), (2260, 1200, 16), (2250, 1380, 15), (2350, 1560, 18)]], 0.42,
                dict(ultramarine=1.0)),
               ([(2600, 330, 160), (2300, 560, 190), (2050, 740, 120)], 0.65, dict(ultramarine=1.0))],
              halo=5, drops=fling(2850, 520, 0.25, 0.7, 260, 18, 12, r), runs=10)
    # the cerulean along the bottom, the water, with a rivulet up to the ultramarine and a splash
    box.stain([([[(-100, 1760, 250), (250, 1680, 300), (650, 1660, 270), (1050, 1720, 200), (1400, 1810, 160),
                  (1750, 1860, 170), (2100, 1870, 210), (2450, 1840, 220), (2750, 1730, 180), (3100, 1630, 150)],
                 [(1500, 1720, 30), (1520, 1500, 18), (1600, 1250, 16), (1700, 1080, 18), (1720, 980, 22)]],
                0.45, dict(cerulean=1.0)),
               ([(-50, 1800, 100), (250, 1720, 140), (550, 1700, 100), (850, 1740, 60), (1100, 1800, 30)], 0.4,
                dict(cerulean=0.6, sap=0.4))],
              halo=6, drops=np.vstack([[(1150, 1480, 34, 0, 1)], fling(1150, 1480, 0, 2 * np.pi, 110, 12, 11, r)]),
              runs=10)
    # the sap green low on the right, under the end of the water
    box.stain([([(1700, 2500, 240), (2050, 2250, 300), (2450, 2110, 320), (2850, 2040, 320), (3100, 1970, 280)],
                0.35, dict(sap=1.0)),
               ([(2250, 2300, 130), (2500, 2200, 150), (2750, 2170, 100)], 0.6, dict(sap=1.0))],
              halo=7, drops=fling(1650, 2350, 3.3, 0.6, 200, 8, 11, r), runs=8)
    return box.glaze(char)
