"""The White Between. Oil thinned with turpentine on white-primed cotton duck.

Sam Francis lived in Paris through the 1950s, and there the close field of small cells he painted at
first, as in Big Red, opened up. In the big canvases of 1956 to 1958, the Basel Mural and Around the Blues
among them, the cells gathered into clusters with the white of the canvas breathing between them, and in
the Blue Balls of 1960 to 1963 they withdrew to the edges and left the middle empty. He painted with a
three-inch brush and oil thinned with turpentine until it ran, on a canvas standing against the wall, one
colour at a time, letting each dry before the next went on; what the paint did as it ran and dried he left.

This canvas was primed white over the weave. The lightest colours went on first: three veils of cerulean
so thin that they only tint the white, then yellow, cerulean, cobalt and green, then ultramarine and a
deep violet blue, and last most of the yellow, orange and red, which hide a little of the blue they cross
where the blues only glaze. Each cell is a few touches of the brush, pressed and drawn a short way. The
thin paint soaked into the gesso and spread past the brush, its oil a little further than its colour, and
as it dried the pigment drew out to the edge and left a dark rim round a paler middle, with the weave
showing through. Where it stood deepest, at the foot of a cell, it ran straight down, thinning as it went,
and stopped in a bead; the brush threw flecks as it worked. At the left the cells crowd into a mass that
runs down the edge of the canvas, the biggest where there was most room and smaller ones packed round
them; along the top they thin out and drift away to the right, and on the right they gather again in the
corners, with a few cells alone in the white between.
"""

import numpy as np
from scipy import ndimage

from atelier import brush, canvas, noise
from atelier import watercolour as wc
from atelier.color import lin, pigment
from atelier.paper import Sheet

TITLE = "The White Between"
DATE = "2026"
MEDIUM = "Oil thinned with turpentine on white-primed cotton duck"
AFTER = ("Sam Francis, the paintings of 1953–65: Big Red (1953), the Basel Mural (1956–58), Around the Blues "
         "(1957, 1962–63), the Blue Balls series (1960–63) and the first Edge paintings (1964–65)")
ROOM = "Colour Itself"
YEAR = 1961
PLACE = "Paris"
REGION = "Americas"
NOTE = ("Cells of cerulean, cobalt and ultramarine, with touches of yellow, orange, red and green, crowd down "
        "the left of a white canvas, drift along the top and gather in the corners on the right; the thin oil "
        "dried with dark edges round paler middles and ran down in drips.")

H, W = 2250, 3600
GESSO = "#f6f3ec"

PAINT = dict(cerulean=pigment("#2b93d8"), cobalt=pigment("#2a62d2"), ultramarine=pigment("#2341c2"),
             deep=pigment("#1b3a9a"), violet=pigment("#4b2f96"), yellow=pigment("#f5c518"),
             lemon=pigment("#f2df3a"), orange=pigment("#f17a1f"), red=pigment("#de2b2b"),
             crimson=pigment("#b0203f"), green=pigment("#14996e"), lime=pigment("#86c23a"),
             pink=pigment("#f2a2b4"))
GRAIN = dict(cerulean=0.6, cobalt=0.5, ultramarine=0.8, deep=0.5, violet=0.4, yellow=0.2, lemon=0.15,
             orange=0.25, red=0.25, crimson=0.15, green=0.35, lime=0.3, pink=0.1)
LOAD = dict(cerulean=0.75, cobalt=0.7, ultramarine=0.75, deep=0.8, violet=0.7, yellow=0.95, lemon=0.85,
            orange=0.9, red=0.85, crimson=0.8, green=0.75, lime=0.85, pink=0.6)
ORDER = dict(lemon=0, yellow=0, pink=0, cerulean=1, lime=1.5, orange=1.5, green=2, red=2, cobalt=2,
             crimson=2.5, ultramarine=2.5, violet=3, deep=3)
BLUES = ("cerulean", "cobalt", "ultramarine", "deep", "violet")
OIL = pigment("#e9dcbc")
# how far each paint hides what is under it: the cadmiums and cerulean cover, ultramarine and violet glaze
HIDE = dict(cerulean=0.3, cobalt=0.12, ultramarine=0.05, deep=0.05, violet=0.05, yellow=0.6, lemon=0.55,
            orange=0.6, red=0.5, crimson=0.15, green=0.2, lime=0.45, pink=0.45)
TOUCHES = ("yellow", "lemon", "orange", "red", "crimson", "pink")


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def ground(r):
    """White priming brushed over cotton duck: it fills the weave but does not hide it."""
    sheet = canvas.duck((H, W), int(r.integers(1 << 31)), tint=GESSO, thread=3.0)
    col = lin(GESSO)[None, None, :] * (0.95 + 0.07 * sheet.tooth[..., None]) \
        * (1 + 0.012 * noise.fbm((H, W), 600, r, octaves=3)[..., None]
           + 0.006 * noise.stretched((H, W), 40, 400, r)[..., None])
    sheet.color = col.astype(np.float32)
    return sheet


class Painting:
    """The canvas as it is painted, colour after colour, each laid over what is already there."""

    def __init__(self, r):
        self.r = r
        self.sheet = ground(r)
        self.img = self.sheet.color.copy()
        self.streak = (0.7 * noise.stretched((1024, 1024), 10, 140, r)
                       + 0.3 * noise.stretched((1024, 1024), 4, 50, r))

    def _box(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return slice(y0, y1), slice(x0, x1)

    def _lay(self, sl, dens, colour, oil=0.0):
        """Thin paint of `colour` at density `dens` over `sl`: it hides some of what is under it, as
        far as the paint covers, and glazes the rest."""
        a = (HIDE[colour] * (1 - np.exp(-1.5 * dens)))[..., None]
        self.img[sl] = (self.img[sl] * (1 - a) + self.sheet.color[sl] * a) \
            * np.exp(-dens[..., None] * PAINT[colour] - np.asarray(oil)[..., None] * OIL)

    def cell(self, x, y, s, colour, load, ang, spread=8.0, drips=1.0, pale=0.35, grav=0.3, rim=0.9, rim_width=3.5):
        """One cell about `s` px across from its middle at (x, y): a few touches of the brush turned
        about `ang`, round where it was pressed and square where a flat one was drawn along. The thin
        paint dried with a rim `rim` dark and `rim_width` px wide round a middle `pale` lighter, settled
        `grav` heavier at the foot, crept up to `spread` px past the brush, and ran down in drips."""
        r = self.r
        k = 1 if s < 60 else int(np.clip(round(s / 55 + r.uniform(-0.5, 1.0)), 2, 6))
        prims = []
        for i in range(k):
            ox, oy = (0.0, 0.0) if k == 1 else r.normal(0, 0.33 * s, 2)
            a = ang + (0 if i == 0 else r.normal(0, 0.7))
            if k == 1:
                rad, hl = s * r.uniform(0.7, 0.95), s * r.uniform(0, 0.55)
            else:
                rad, hl = s * r.uniform(0.33, 0.55), s * r.uniform(0.1, 0.5)
            prims.append((x + ox, y + oy, a, hl, rad, r.uniform(0.3, 0.7) if r.uniform() < 0.55 else 0))
        reach = max(np.hypot(cx - x, cy - y) + hl + rad for cx, cy, _, hl, rad, _ in prims)
        pad = 1.15 * reach + 0.2 * s + 2.5 * spread + 14
        sl = self._box(x - pad, y - pad, x + pad, y + pad)
        if sl is None:
            return
        yy, xx = np.mgrid[sl].astype(np.float32)
        d, near, st = None, None, None
        o = r.uniform(0, 1024, 2)
        bend = max(30, 0.8 * s)     # the brush never travels quite straight
        wu, wv = 10 * noise.field(xx.shape, bend, r), 10 * noise.field(xx.shape, bend, r)
        for i, (cx, cy, a, hl, rad, cr) in enumerate(prims):
            u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
            v = (yy - cy) * np.cos(a) - (xx - cx) * np.sin(a)
            if cr:      # a flat brush pressed and drawn along: straight sides, rounded corners
                qx, qy = np.abs(u) - hl, np.abs(v) - rad * (1 - cr)
                di = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad * cr
            else:       # a round one, pressed
                di = np.hypot(np.maximum(np.abs(u) - hl, 0), v) - rad
            si = ndimage.map_coordinates(self.streak, [u + wu + o[0] + 97 * i, v + wv + o[1]], order=1,
                                         mode="grid-wrap")
            if d is None:
                d, near, st = di, di, si
            else:
                st = np.where(di < near, si, st)
                near = np.minimum(near, di)
                d = _smin(d, di, 0.15 * rad)
        sh = d.shape
        d = d + 0.09 * s * noise.field(sh, max(6, 0.5 * s), r) + 1.6 * noise.field(sh, 16, r) \
            + 0.6 * noise.field(sh, 5, r) + 2.4 * (self.sheet.tooth[sl] - 0.5)
        m = noise.smoothstep(0.6, -0.6, d)
        if m.max() < 0.5:
            return
        sub = Sheet(self.sheet.color[sl], self.sheet.tooth[sl], self.sheet.fiber[sl], self.sheet.alpha[sl])
        f = wc.deposit(m, sub, r, pool=0.35, rim=rim, rim_width=rim_width, tides=0.2, grain=GRAIN[colour],
                       scale=max(30, 0.7 * s))
        din = ndimage.distance_transform_edt(m > 0.5)
        dout = ndimage.distance_transform_edt(m <= 0.5)
        g = 1 + grav * np.clip((yy - y) / s, -1, 1.2)
        pm = 1 - pale * noise.smoothstep(6, 0.35 * s + 16, din)
        cloud = 1 + 0.12 * noise.fbm(sh, max(20, 0.5 * s), r, octaves=2)
        weave = 1 + 0.3 * (0.5 - self.sheet.tooth[sl])
        dens = load * f * g * pm * cloud * weave * (1 + 0.13 * st)
        # the thinner paint crept out through the gesso a little way, and its oil a little further
        sw = spread * np.clip(0.2 + 0.8 * noise.field(sh, max(20, 0.6 * s), r), 0, None) ** 1.5
        ms = noise.smoothstep(sw + 0.7, sw - 0.7, dout) * (1 - m) * (sw > 0.6)
        tide = np.exp(-((sw - dout) / 1.3) ** 2) * ms
        ow = sw + 2 + 4 * np.clip(noise.field(sh, 40, r), 0, None)
        mo = noise.smoothstep(ow + 0.7, ow - 0.7, dout) * (1 - m)
        dens = dens + load * (0.04 * ms + 0.07 * tide)
        oil = mo * 0.06 * (1 + 0.6 * (self.sheet.tooth[sl] - 0.5)) * np.clip(1 + 0.5 * noise.field(sh, 50, r), 0, None)
        self._lay(sl, dens, colour, oil)
        # paint pooled at the foot of the cell and ran down
        n = r.poisson(drips * (0.15 + s / 250))
        if n:
            cols = np.nonzero((m > 0.5).any(axis=0))[0]
            bot = np.array([np.nonzero(m[:, j] > 0.5)[0].max() for j in cols])
            w = np.exp((bot - bot.max()) / (0.2 * s + 4))
            for j in r.choice(len(cols), n, p=w / w.sum()):
                w0 = np.clip(s * r.uniform(0.04, 0.1), 3, 14)
                L = np.clip(np.exp(r.normal(np.log(90 + 0.8 * s), 0.75)), 25, 900)
                self.drip(sl[1].start + cols[j] + r.uniform(-0.5, 0.5), sl[0].start + bot[j], w0, L, colour,
                          load * r.uniform(0.7, 1.1))

    def drip(self, x0, y0, w0, L, colour, load):
        """Paint that ran straight down from (x0, y0) for L px, thinning as it went, and stopped in a bead."""
        r = self.r
        y1 = y0 + L
        pad = w0 + 12
        sl = self._box(x0 - pad, y0 - w0, x0 + pad, y1 + 2 * w0 + 8)
        if sl is None:
            return
        yy, xx = np.mgrid[sl].astype(np.float32)
        n = sl[0].stop - sl[0].start
        t = np.clip((yy - y0) / L, 0, 1)
        wob = (1.5 * noise.line1d(n, 150, r) + 0.4 * noise.line1d(n, 15, r))[:, None]
        xc = x0 + wob + r.normal(0, 0.004) * (yy - y0)
        swell = np.clip(1 + 0.3 * noise.line1d(n, 40, r), 0.45, None)[:, None]
        hw = 0.5 * w0 * (1 - 0.5 * t ** 0.8) * swell * (1 + 0.8 * np.exp(-np.clip(yy - y0, 0, None) / (1.2 * w0)))
        hw = hw + 0.45 * noise.field(hw.shape, 2.5, r) + 0.8 * (self.sheet.tooth[sl] - 0.5)
        dl = np.where((yy >= y0 - w0) & (yy <= y1), np.abs(xx - xc) - hw, 1e3)
        rb = 0.25 * w0 * r.uniform(1.3, 2.0) + 1.0
        k = int(np.clip(y1 - rb - sl[0].start, 0, n - 1))
        bx, by = x0 + wob[k, 0] + r.normal(0, 0.004) * L, y1 - 0.3 * rb
        db = (np.hypot((xx - bx) / rb, (yy - by) / (1.3 * rb)) - 1) * rb
        d = _smin(dl, db, 0.5 * rb)
        m = noise.smoothstep(0.55, -0.55, d)
        rim = np.exp(-np.clip(-d, 0, None) / 1.3)
        dens = load * m * (0.8 + 0.4 * rim + 0.6 * np.exp(-((yy - by) / (1.6 * rb)) ** 2))
        self._lay(sl, dens, colour)

    def drop(self, x, y, rad, el, a, colour, load):
        """A drop of paint `rad` px round, drawn out `el` times along its flight `a`."""
        r = self.r
        e = rad * el + 4
        sl = self._box(x - e, y - e, x + e, y + e)
        if sl is None:
            return
        yy, xx = np.mgrid[sl].astype(np.float32)
        u, v = (xx - x) * np.cos(a) + (yy - y) * np.sin(a), (yy - y) * np.cos(a) - (xx - x) * np.sin(a)
        d = (np.hypot(u / el, v) - rad) + 0.15 * rad * noise.field(u.shape, max(2, rad), r)
        m = noise.smoothstep(0.5, -0.5, d)
        rim = np.exp(-np.clip(-d, 0, None) / 1.2)
        self._lay(sl, load * m * (1 + 0.5 * rim), colour)

    def fling(self, x, y, ang, fan, far, n, size, colour, load):
        """Paint flung from a brush at (x, y): n drops fanned `fan` about `ang` out to `far` px, the big
        ones nearer, the far ones drawn out with a smaller drop beyond, and the heaviest running down."""
        r = self.r
        t = r.uniform(0.04, 1, n) ** 0.8
        a = ang + fan * r.normal(0, 0.35, n)
        rad = size * r.uniform(0.1, 1, n) ** 2 * (1.25 - 0.7 * t) + 0.8
        el = 1 + 0.8 * t * r.uniform(0.2, 1, n)
        for i in range(n):
            px, py = x + far * t[i] * np.cos(a[i]), y + far * t[i] * np.sin(a[i])
            self.drop(px, py, rad[i], el[i], a[i], colour, load * r.uniform(0.8, 1.3))
            if el[i] > 1.5:
                q = rad[i] * el[i] * 1.7
                self.drop(px + q * np.cos(a[i]), py + q * np.sin(a[i]), 0.35 * rad[i] + 0.4, 1.2, a[i], colour, load)
            if rad[i] > 5 and r.uniform() < 0.4:
                self.drip(px, py, 1.2 * rad[i], r.uniform(20, 160), colour, load)


def kernels():
    """Where the cells gather: polylines of rows (x, y, spread, strength)."""
    return [
        [(-120, -80, 280, 1), (300, 230, 240, 1), (520, 600, 190, 1), (420, 1000, 160, 1), (260, 1400, 140, 1),
         (150, 1800, 150, 1), (40, 2300, 190, 1)],
        [(300, 230, 180, 0.95), (700, 150, 150, 0.9), (1050, 260, 120, 0.8), (1450, 140, 85, 0.65),
         (1850, 280, 65, 0.55), (2250, 160, 45, 0.45)],
        [(3720, 40, 200, 0.95), (3450, 250, 160, 0.85), (3380, 520, 120, 0.7), (3560, 760, 100, 0.55)],
        [(3650, 2200, 180, 0.9), (3400, 2150, 120, 0.7), (3150, 2260, 80, 0.5)],
        [(150, 2050, 140, 0.65), (650, 2200, 130, 0.6), (1100, 2330, 100, 0.5)],
    ]


def cluster_map(r):
    gh, gw = H // 10, W // 10
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32) * 10
    P = np.zeros((gh, gw), np.float32)
    for path in kernels():
        for x, y, s, w in brush.path(np.asarray(path, np.float64), 12):
            P = np.maximum(P, w * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))
    return np.where(P > 0.12, P + 0.18 * noise.fbm((gh, gw), 25, r, octaves=3), 0)


ACCENTS = [  # where the touches of other colours went: (x, y, reach, colours)
    (520, 560, 320, ("yellow", "lemon", "orange")),
    (3450, 300, 260, ("yellow", "red", "orange")),
    (3450, 2250, 240, ("red", "orange", "crimson")),
    (650, 2150, 260, ("orange", "red", "pink")),
    (260, 1650, 260, ("green", "lime")),
    (1000, 280, 220, ("green", "yellow")),
    (380, 1250, 300, ("violet", "crimson")),
]


def plan(r):
    """The cells. Each cluster is filled the way the painter filled it: the biggest cell where there
    is most room, then the next into the room left, overlapping a little or leaving a seam of white,
    down to small ones at the ragged edges; a few strays beyond. -> rows (x, y, size, colour)"""
    P = cluster_map(r)
    gh, gw = P.shape
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * 10
    free = P > 0.35
    cells = []
    while len(cells) < 320:
        room = ndimage.distance_transform_edt(free).ravel() * 10
        top = room.max()
        if top < 46:
            break
        cand = np.flatnonzero(room > 0.6 * top)
        i = cand[r.choice(len(cand), p=room[cand] ** 2 / (room[cand] ** 2).sum())]
        x, y = (i % gw + r.uniform()) * 10, (i // gw + r.uniform()) * 10
        s = float(np.clip(room[i] * r.uniform(1.0, 1.4), 10, 190))
        cells.append((x, y, s))
        free &= (gx - x) ** 2 + (gy - y) ** 2 > (0.65 * s) ** 2
    # later colours laid over the first, anywhere in the clusters
    ys, xs = np.nonzero(P > 0.5)
    for k in r.choice(len(xs), len(cells) // 6, replace=False):
        s = float(np.exp(r.normal(np.log(55), 0.35)))
        cells.append(((xs[k] + r.uniform()) * 10, (ys[k] + r.uniform()) * 10, s))
    # strays round the edges of the clusters
    near = ndimage.binary_dilation(P > 0.35, iterations=7) & (P <= 0.35)
    ys, xs = np.nonzero(near)
    for k in r.choice(len(xs), 18, replace=False):
        s = float(np.exp(r.normal(np.log(18), 0.5)))
        cells.append(((xs[k] + r.uniform()) * 10, (ys[k] + r.uniform()) * 10, s))
    blues = np.array([0.4, 0.27, 0.2, 0.09, 0.04])
    out = []
    for x, y, s in cells:
        col = BLUES[r.choice(5, p=blues / blues.sum())]
        for ax, ay, reach, cols in ACCENTS:
            near = np.exp(-((x - ax) ** 2 + (y - ay) ** 2) / (2 * reach ** 2))
            if s < 120 and r.uniform() < 0.4 * near:
                col = cols[r.integers(len(cols))]
        out.append((x, y, s, col))
    return out


def paint(seed=1961):
    r = noise.rng(seed)
    pic = Painting(r)
    cells = plan(r)
    # touches of other colours among the blues
    cells += [(520, 330, 72, "yellow"), (300, 760, 50, "orange"), (3420, 420, 46, "red"), (3330, 200, 58, "yellow"),
              (210, 1720, 56, "green"), (760, 2170, 60, "pink"), (980, 300, 44, "green"), (560, 2080, 38, "orange")]
    # first three veils of cerulean so thin they only tint the white
    for x, y, s in [(830, 880, 230), (3180, 760, 170), (1350, 380, 150)]:
        pic.cell(x, y, s, "cerulean", 0.09, r.uniform(0, np.pi), spread=0, drips=0, pale=0.1, rim=0.6, rim_width=3)
    # then the colours lightest first; most of the touches went on over the blues, some under them
    late = r.uniform(size=len(cells)) < 0.6
    order = sorted(range(len(cells)), key=lambda k: (ORDER[cells[k][3]] + 3 * (cells[k][3] in TOUCHES and late[k]),
                                                     r.uniform()))
    for k in order:
        x, y, s, col = cells[k]
        thin = r.uniform(0.35, 0.6) if r.uniform() < 0.3 else 1.0       # some went on much thinner
        pic.cell(x, y, s, col, thin * LOAD[col] * np.exp(r.normal(0, 0.3)), r.uniform(0, np.pi),
                 spread=r.uniform(2, 10), drips=1.0 if s > 40 else 0.4, pale=r.uniform(0.15, 0.5),
                 rim=r.uniform(0.3, 1.4), rim_width=r.uniform(2.5, 7))
    # a few cells adrift in the white, which hardly ran
    for x, y, s, col in [(1460, 1330, 50, "cerulean"), (1990, 1540, 14, "cobalt"), (2600, 1690, 88, "cobalt"),
                         (2790, 1830, 34, "red"), (2700, 1935, 20, "yellow"), (1900, 420, 60, "cerulean"),
                         (2650, 300, 28, "cobalt"), (2930, 170, 16, "cerulean"), (3260, 2050, 40, "ultramarine")]:
        pic.cell(x, y, s, col, LOAD[col] * r.uniform(0.8, 1.1), r.uniform(0, np.pi), spread=r.uniform(2, 8),
                 drips=0.3, pale=r.uniform(0.2, 0.45), rim=r.uniform(0.5, 1.2), rim_width=r.uniform(2.5, 6))
    # flecks thrown off the brush as it worked, in bursts round the clusters
    big = [c for c in cells if c[2] > 50]
    for k in r.choice(len(big), 14, replace=False):
        x, y, s, col = big[k]
        a = r.uniform(0, 2 * np.pi)
        cx, cy = x + 1.1 * s * np.cos(a), y + 1.1 * s * np.sin(a)
        n = r.integers(15, 60)
        sx, sy = r.uniform(30, 130, 2)
        for i in range(n):
            pic.drop(cx + r.normal(0, sx), cy + r.normal(0, sy), 0.6 + 2.4 * r.uniform() ** 2.5, r.uniform(1, 1.6),
                     r.uniform(0, np.pi), col, LOAD[col] * r.uniform(0.8, 1.3))
    pic.fling(650, 700, -0.15, 0.5, 700, 18, 10, "ultramarine", 1.0)
    pic.fling(3300, 520, np.pi - 0.3, 0.6, 380, 12, 9, "red", 0.9)
    return pic.img
