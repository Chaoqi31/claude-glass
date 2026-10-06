"""North Rose. Pot-metal glass, painted and fired, leaded into plate tracery.

About 1235 the queen of France, Blanche of Castile, paid for the rose window in the north
transept of Chartres. It is cut through a wall of stone: a flower in the middle, twelve
petals, a ring of squares and lozenges, a rim of half-moons, and between them little
quatrefoils, every opening glazed as its own panel. Her lilies and her castles are all
over it. This rose keeps her heraldry and her plan, and puts twelve white doves in the
petals, all coming down to a rose at the centre.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from atelier import glass, noise
from atelier.color import lin

TITLE = "North Rose"
DATE = "2026"
MEDIUM = "Pot-metal and flashed glass, painted and fired, leaded, in plate tracery"
AFTER = "The north rose of Chartres Cathedral, given by Blanche of Castile, c. 1235"
ROOM = "The Workshop"
YEAR = 1235
PLACE = "Chartres"
REGION = "Europe"
NOTE = ("A north rose never has the sun on it; this is its light on any grey afternoon. Twelve "
        "doves come down to the rose at the centre, among the lilies of France and the castles of "
        "Castile: seven thousand pieces of glass, no two of them the same thickness.")

# The glazier's glass, as it looks against a grey north sky.
BLUE, PALE, RED, WHITE, YELLOW, GREEN = range(1, 7)
GLASSES = [
    glass.Glass(("#2d5fd6", "#2448b8", "#3a70d8", "#2a50c0", "#2440a0", "#3548b8"), thick=0.32, mottle=0.22, streak=0.07, seeds=1.5, weather=0.18),
    glass.Glass(("#7aa0e6", "#8aa8e0", "#6a90d8"), thick=0.25, mottle=0.18, streak=0.06, seeds=1.5, weather=0.18),
    glass.Glass(("#cc1e22", "#b4182c", "#d8321e", "#c02a20"), thick=0.22, mottle=0.2, streak=0.16, seeds=0.8, weather=0.14),
    glass.Glass(("#eef0e0", "#e2eada", "#f0e8cc", "#dde6e0"), thick=0.15, mottle=0.14, streak=0.04, seeds=2.5, weather=0.5),
    glass.Glass(("#f4bc3c", "#e8a830", "#f8cc58"), thick=0.22, mottle=0.18, streak=0.08, seeds=1.2, weather=0.2),
    glass.Glass(("#30a864", "#5aa84a", "#28a07e"), thick=0.25, mottle=0.18, streak=0.06, seeds=1.2, weather=0.2),
]
TRACE, MAT = 2.4, 0.08


# ---- shapes, as signed distance (negative inside), in a local frame: v points away from the centre

def circle(u, v, r):
    return np.hypot(u, v) - r


def box(u, v, a, b):
    qx, qy = np.abs(u) - a, np.abs(v) - b
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def diamond(u, v, a):
    return box((u + v) / np.sqrt(2), (u - v) / np.sqrt(2), a / np.sqrt(2), a / np.sqrt(2))


def cone(u, v, r1, r2, h):
    """A petal: a small circle at the origin and a large one at (0, h), and the tangents between."""
    b = (r1 - r2) / h
    a = np.sqrt(1 - b * b)
    q = np.abs(u)
    k = -b * q + a * v
    return np.where(k < 0, np.hypot(q, v) - r1, np.where(k > a * h, np.hypot(q, v - h) - r2, a * q + b * v - r1))


def half(u, v, r):
    return np.maximum(np.hypot(u, v) - r, -v)


def quatrefoil(u, v, r):
    a, b = r * 0.46, r * 0.54
    return np.minimum.reduce([circle(u - a, v, b), circle(u + a, v, b), circle(u, v - a, b), circle(u, v + a, b)])


def lobed(u, v, r0, rl, n):
    d = circle(u, v, r0)
    for k in range(n):
        t = 2 * np.pi * k / n
        d = np.minimum(d, circle(u - r0 * np.sin(t), v - r0 * np.cos(t), rl))
    return d


def outline(sdf, level, origin, reach):
    """Points 1 px apart round the level set `level` of `sdf`, found along rays from `origin`."""
    phi = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    du, dv = np.sin(phi), np.cos(phi)
    lo, hi = np.zeros_like(phi), np.full_like(phi, reach)
    for _ in range(30):
        mid = (lo + hi) / 2
        inside = sdf(origin[0] + du * mid, origin[1] + dv * mid) < level
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    P = np.stack([origin[0] + du * lo, origin[1] + dv * lo], 1)
    P = np.vstack([P, P[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    t = np.arange(0, s[-1], 1.0)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1), s[-1]


def scrolls(u, v, cell, width=0.9):
    """Scrollwork scratched out of a mat: one spiral in every cell, turning alternate ways."""
    a, b = u / cell, v / cell
    i, j = np.floor(a), np.floor(b)
    fa, fb = a - i - 0.5, b - j - 0.5
    turn = np.where((i + j) % 2 == 0, 1.0, -1.0)
    rho = np.hypot(fa, fb)
    phi = (np.arctan2(fb, fa * turn) % (2 * np.pi)) / (2 * np.pi)
    target = 0.07 + 0.3 * phi
    dd = np.minimum(np.abs(rho - target), np.abs(rho - target - 0.3)) * cell
    return noise.smoothstep(width, width * 0.3, dd) * (rho < 0.47)


# ---- motifs: small cartoons of their own, drawn once for every place they are used

def bez(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    P = [np.asarray(p, np.float64) for p in (p0, p1, p2, p3)]
    return list(map(tuple, (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]))


def mirror(pts):
    return [(-x, y) for x, y in reversed(pts)]


def rot(pts, a):
    c, s = np.cos(a), np.sin(a)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


class Motif:
    """Pieces and paint in [-1, 1]^2, y pointing away from the rose's centre; `rad` is the size
    in px it will be used at, so painted lines can be given in px."""
    S = 640

    def __init__(self, rad, r):
        self.rad, self.r = rad, r
        self.keys = Image.new("I", (self.S, self.S), 0)
        self.kind = Image.new("L", (self.S, self.S), 0)
        self.paint = Image.new("F", (self.S, self.S), 0.0)
        self.dk, self.dc, self.dp = ImageDraw.Draw(self.keys), ImageDraw.Draw(self.kind), ImageDraw.Draw(self.paint)
        self.n = 0

    def P(self, pts, jitter=0.0):
        j = self.r.normal(0, jitter, (len(pts), 2)) if jitter else np.zeros((len(pts), 2))
        return [((x + dx + 1) * self.S / 2, (1 - y - dy) * self.S / 2) for (x, y), (dx, dy) in zip(pts, j)]

    def piece(self, pts, kind, jitter=0.002):
        self.n += 1
        q = self.P(pts, jitter)
        self.dk.polygon(q, fill=self.n)
        self.dc.polygon(q, fill=kind)

    def disc(self, x, y, rad, kind):
        self.piece([(x + rad * np.sin(t), y + rad * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)],
                   kind, jitter=0.003)

    def line(self, pts, width, density=TRACE, taper=0.35):
        """A painted line, `width` px, thinner where the brush set down and lifted."""
        pts = np.asarray(pts, np.float64)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        k = max(2, int(s[-1] * self.S / 2 / 1.5))
        t = np.linspace(0, s[-1], k)
        x, y = np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])
        f = t / max(s[-1], 1e-9)
        wid = width * (taper + (1 - taper) * np.sin(np.pi * np.clip(f, 0, 1)) ** 0.6) * self.r.uniform(0.85, 1.15)
        scale = self.S / (2 * self.rad)            # template px per image px
        for (X, Y), rr in zip(self.P(list(zip(x, y))), wid * scale / 2):
            self.dp.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=density)

    def dot(self, x, y, rad_px, density=TRACE):
        X, Y = self.P([(x, y)])[0]
        rr = rad_px * self.S / (2 * self.rad)
        self.dp.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=density)

    def done(self):
        keys = np.asarray(self.keys, np.int64)
        paint = np.asarray(self.paint, np.float32)
        scale = self.S / (2 * self.rad)
        # a thin mat along the inside of every piece, to model it, and the paint only lives on glass
        edge = ndimage.distance_transform_edt(~glass.boundary(keys)) / scale
        paint = np.maximum(paint, MAT * np.exp(-edge / 2.5)) * (keys > 0)
        paint *= 0.75 + 0.25 * noise.field(paint.shape, 2.0 * scale, self.r)   # the brush was never evenly loaded
        self.a_keys, self.a_kind = keys, np.asarray(self.kind, np.int64)
        self.a_paint = ndimage.gaussian_filter(paint, 0.5 * scale)
        return self


def fleur(rad, r, kind=YELLOW, small=False):
    """The lily of France: side petals springing from under the band and bowing out and down."""
    m = Motif(rad, r)
    j = lambda: r.normal(0, 0.012)
    for s in (1, -1):
        upper = bez((0.06, 0.1), (0.2 + j(), 0.56), (0.66, 0.74 + j()), (0.78, 0.16))
        tip = bez((0.78, 0.16), (0.8, 0.06), (0.74, 0.0), (0.68, 0.06), n=10)
        under = bez((0.68, 0.06), (0.66, 0.36 + j()), (0.44, 0.44), (0.3, 0.06))
        m.piece([(s * x, y) for x, y in upper + tip + under], kind)
        foot = bez((0.12, -0.1), (0.3, -0.12), (0.46, -0.24), (0.44, -0.44)) + \
            bez((0.44, -0.44), (0.36, -0.46), (0.3, -0.32), (0.18, -0.22), n=10) + [(0.08, -0.2)]
        m.piece([(s * x, y) for x, y in foot], kind)
    left = bez((-0.13, 0.06), (-0.32 + j(), 0.34), (-0.16, 0.62 + j()), (0, 0.95))
    m.piece(left + mirror(left), kind)
    stub = bez((-0.09, -0.1), (-0.1, -0.25), (-0.05, -0.35), (0, -0.54))
    m.piece(stub + mirror(stub), kind)
    m.piece([(-0.4, -0.1), (0.4, -0.1), (0.43, -0.01), (0.4, 0.08), (-0.4, 0.08), (-0.43, -0.01)], kind)
    # the painter's lines: a midrib, the edges of the band, a bead on it
    m.line([(0, 0.14), (0.005, 0.5), (0, 0.82)], 1.0 if small else 1.3)
    if small:
        return m.done()
    for s in (1, -1):
        m.line([(s * x, y) for x, y in bez((0.3, 0.22), (0.4, 0.5), (0.62, 0.56), (0.7, 0.24), n=16)], 1.0)
    m.line([(-0.36, 0.045), (0.36, 0.045)], 0.9)
    m.line([(-0.36, -0.065), (0.36, -0.065)], 0.9)
    for x in (-0.22, 0.0, 0.22):
        m.dot(x, -0.01, 1.4)
    return m.done()


def castle(rad, r, kind=YELLOW, gate=BLUE):
    m = Motif(rad, r)

    def tower(x0, x1, y0, y1, merlons):
        w = (x1 - x0) / (2 * merlons - 1)
        top = [(x0, y0), (x1, y0), (x1, y1)]
        for i in reversed(range(merlons)):
            a = x0 + 2 * i * w
            top += [(a + w, y1), (a + w, y1 + 0.12), (a, y1 + 0.12), (a, y1)] if i else [(a + w, y1), (a + w, y1 + 0.12), (a, y1 + 0.12)]
        m.piece(top, kind)

    tower(-0.34, -0.2, -0.6, 0.0, 1)
    tower(0.2, 0.34, -0.6, 0.0, 1)
    tower(-0.68, -0.34, -0.6, 0.22, 2)
    tower(0.34, 0.68, -0.6, 0.22, 2)
    tower(-0.2, 0.2, -0.6, 0.52, 3)
    arch = lambda x, y0, y1, hw: [(x - hw, y0), (x + hw, y0)] + \
        [(x + hw * np.cos(t), y1 + hw * np.sin(t)) for t in np.linspace(0, np.pi, 12)]
    m.piece(arch(0, -0.6, -0.24, 0.1), gate)
    for x, y0, y1 in ((0, 0.14, 0.3), (-0.51, -0.12, 0.02), (0.51, -0.12, 0.02)):
        m.piece(arch(x, y0, y1, 0.045), gate)
    # masonry, drawn in thinner paint
    for y in np.arange(-0.5, 0.5, 0.11):
        for x0, x1, top in ((-0.68, -0.34, 0.22), (-0.34, -0.2, 0.0), (-0.2, 0.2, 0.52), (0.2, 0.34, 0.0), (0.34, 0.68, 0.22)):
            if y < top - 0.03:
                m.line([(x0 + 0.02, y + r.normal(0, 0.004)), (x1 - 0.02, y + r.normal(0, 0.004))], 0.7, density=1.4, taper=0.6)
    m.line([(-0.68, -0.6), (0.68, -0.6)], 1.2)
    return m.done()


def rose(rad, r):
    m = Motif(rad, r)
    for k in range(5):                                 # the barbs show between the petals
        a = 2 * np.pi * (k + 0.5) / 5
        barb = bez((-0.11, 0.5), (-0.11, 0.66), (-0.06, 0.84), (0, 1.0))
        m.piece(rot(barb + mirror(barb), -a), GREEN)
    for k in range(5):
        a = 2 * np.pi * k / 5
        edge = [(0.47 * np.sin(a) + 0.5 * np.sin(a + t), 0.47 * np.cos(a) + 0.5 * np.cos(a + t))
                for t in np.linspace(-1.21, 1.21, 30)]
        m.piece([(0, 0)] + edge, RED)
    for k in range(5):
        a = 2 * np.pi * (k + 0.5) / 5
        m.disc(0.25 * np.sin(a), 0.25 * np.cos(a), 0.17, WHITE)
    m.disc(0, 0, 0.17, YELLOW)
    for k in range(5):
        a = 2 * np.pi * k / 5
        for t in (-0.25, 0, 0.25):
            m.line([(0.52 * np.sin(a + t * 1.2), 0.52 * np.cos(a + t * 1.2)),
                    (0.78 * np.sin(a + t), 0.78 * np.cos(a + t))], 1.0, taper=0.2)
        b = a + np.pi / 5
        m.line([(0.6 * np.sin(b), 0.6 * np.cos(b)), (0.92 * np.sin(b), 0.92 * np.cos(b))], 0.9)
        m.line([(0.3 * np.sin(b) + 0.1 * np.sin(b + 1.5), 0.3 * np.cos(b) + 0.1 * np.cos(b + 1.5)),
                (0.3 * np.sin(b) + 0.1 * np.sin(b - 1.5), 0.3 * np.cos(b) + 0.1 * np.cos(b - 1.5))], 0.8)
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        m.dot(0.09 * np.sin(t), 0.09 * np.cos(t), 1.5)
    m.dot(0, 0, 1.8)
    return m.done()


def dove(rad, r):
    """The dove comes down head first, wings spread, a cross behind its head."""
    m = Motif(rad, r)
    j = lambda s=0.015: r.normal(0, s)
    # no two painted alike: each has its own spread of wing and fan of tail
    span, lift, fan = r.uniform(0.9, 1.02), r.uniform(-0.07, 0.07), r.uniform(0.8, 1.1)
    wing = lambda pts, s: [(s * x * span, y + lift * abs(x)) for x, y in pts]
    m.disc(0, -0.5, 0.27, RED)
    for s in (1, -1):
        lead_edge = bez((-0.1, -0.14), (-0.34, -0.32 + j()), (-0.7, -0.34 + j()), (-0.96, -0.2))
        trail = [(-0.96, -0.2), (-0.88, -0.01), (-0.82, -0.06), (-0.74, 0.13), (-0.66, 0.06), (-0.58, 0.23),
                 (-0.48, 0.15), (-0.38, 0.28), (-0.24, 0.2), (-0.12, 0.14)]
        m.piece(wing(lead_edge + trail, s), WHITE)
        covert = bez((-0.1, -0.14), (-0.3, -0.28), (-0.46, -0.3), (-0.54, -0.28), n=12) + \
            bez((-0.54, -0.28), (-0.5, -0.12), (-0.44, 0.0), (-0.36, 0.08), n=12) + [(-0.12, 0.12)]
        m.piece(wing(covert, s), WHITE)
    tail = [(x * fan, y) for x, y in
            [(-0.1, 0.3), (-0.3, 0.8), (-0.19, 0.86), (-0.07, 0.8), (0, 0.9), (0.07, 0.8), (0.19, 0.86), (0.3, 0.8), (0.1, 0.3)]]
    m.piece(tail, WHITE)
    body = [(0.15 * np.sin(t), 0.02 + 0.37 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
    m.piece(body, WHITE)
    m.disc(0, -0.49, 0.13, WHITE)
    # the cross in the halo, the eye, the beak, the feathers
    for dx, dy in ((0.17, 0.0), (-0.17, 0.0), (0.0, -0.17)):
        m.line([(dx * 0.8, -0.5 + dy * 0.8), (dx * 1.5, -0.5 + dy * 1.5)], 2.6, taper=0.9)
    m.dot(0.035, -0.5, 1.5)
    m.line([(0, -0.6), (0, -0.67)], 1.6, taper=0.2)
    for s in (1, -1):
        for x0, y0 in ((-0.88, -0.1), (-0.78, 0.03), (-0.69, 0.1), (-0.6, 0.16), (-0.5, 0.2), (-0.4, 0.23)):
            m.line(wing([(x0, y0), (x0 * 0.78 + 0.02, y0 - 0.2 - 0.05 * j(1))], s), 0.9, taper=0.3)
        for y in (-0.2, -0.08):
            m.line(wing([(x, y + 0.08 * np.sin(np.pi * (x + 0.12) / 0.4)) for x in np.linspace(-0.14, -0.46, 8)], s), 0.8)
    for x in (-0.12, 0.0, 0.12):
        m.line([(x * 0.5, 0.36), (x * 1.6 * fan, 0.8)], 0.8)
    m.line([(-0.1, -0.3), (0, -0.25), (0.1, -0.3)], 0.8)
    return m.done()


def rosette(rad, r, a=WHITE, b=YELLOW, heart=RED):
    m = Motif(rad, r)
    for k in range(6):
        p = bez((0, 0.12), (0.26, 0.34), (0.22, 0.66), (0, 0.9))
        m.piece(rot(mirror(p) + p, 2 * np.pi * k / 6), a if k % 2 == 0 else b)
        m.line(rot([(0, 0.24), (0, 0.72)], 2 * np.pi * k / 6), 0.9)
    m.disc(0, 0, 0.16, heart)
    m.dot(0, 0, 1.6)
    return m.done()


# ---- the cartoon

class Cartoon:
    def __init__(self, shape, centre, r):
        self.shape, self.c, self.r = shape, centre, r
        self.keys = np.zeros(shape, np.int64)
        self.kind = np.zeros(shape, np.int64)
        self.paint = np.zeros(shape, np.float32)
        self.open = np.zeros(shape, np.float32)
        self.n = 0
        # the mason and the glazier worked by hand: every line wanders a little
        self.jx = 1.0 * noise.field(shape, 28, r)
        self.jy = 1.0 * noise.field(shape, 28, r)

    def element(self, rad, deg, sdf, reach, origin=(0.0, 0.0)):
        """An opening at polar (`rad`, `deg` clockwise from the top): local pixels and its outline."""
        t = np.deg2rad(deg + self.r.normal(0, 0.15))
        cx = self.c[0] + rad * np.sin(t) + self.r.normal(0, 1.2)
        cy = self.c[1] - rad * np.cos(t) + self.r.normal(0, 1.2)
        h, w = self.shape
        x0, x1 = max(0, int(cx - reach)), min(w, int(cx + reach) + 1)
        y0, y1 = max(0, int(cy - reach)), min(h, int(cy + reach) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx + self.jx[y0:y1, x0:x1] - cx, yy + self.jy[y0:y1, x0:x1] - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = dx * np.sin(t) - dy * np.cos(t)
        d = sdf(u, v)
        self.open[y0:y1, x0:x1] = np.maximum(self.open[y0:y1, x0:x1], noise.smoothstep(0.7, -0.7, d))
        sel = d < 2.0                        # the glass runs a little way under the stone
        iy, ix = np.nonzero(sel)
        self.n += 1
        return dict(Y=iy + y0, X=ix + x0, u=u[sel], v=v[sel], d=d[sel], sdf=sdf, origin=origin,
                    reach=reach, base=self.n << 26)

    def set(self, e, m, key, kind):
        self.keys[e["Y"][m], e["X"][m]] = e["base"] + key
        self.kind[e["Y"][m], e["X"][m]] = kind

    def bands(self, e, m, bands, sdf=None, d=None, origin=None, reach=None, seg=64, tag=0):
        """Border strips following an outline from the outside in: [(px, glass, pearls?)].
        Returns the mask left inside them."""
        sdf, origin, reach = sdf or e["sdf"], origin or e["origin"], reach or e["reach"]
        d = e["d"] if d is None else d
        lo = 0.0
        for i, (bw, kind, pearls) in enumerate(bands):
            band = m & (d >= -(lo + bw))
            mid = -(lo + bw / 2)
            Q, L = outline(sdf, mid, origin, reach * 1.5)
            s = cKDTree(Q).query(np.stack([e["u"][band], e["v"][band]], 1))[1].astype(np.float64)
            nseg = max(3, int(round(L / seg)))
            cut = np.floor(s / L * nseg + self.r.random()).astype(np.int64) % nseg
            self.set(e, band, (tag * 16 + i + 1) * 4096 + cut, kind)
            if pearls:
                count = max(6, int(round(L / (bw * 1.3))))
                sp = L / count
                size = bw * 0.38 * (1 + 0.1 * self.r.standard_normal(count + 1))
                dist = np.hypot(s % sp - sp / 2, d[band] - mid)
                clear = noise.smoothstep(0.5, -0.5, dist - size[(s // sp).astype(np.int64) % count])
                self.paint[e["Y"][band], e["X"][band]] = TRACE * 0.9 * (1 - clear)
            m = m & (d < -(lo + bw))
            lo += bw
        return m

    def ground(self, e, m, kind, u0=0.0, v0=0.0, sectors=6, scroll=11.0, tag=0):
        """A plain ground cut in sectors round (u0, v0), with scrollwork scratched through a mat."""
        a = np.arctan2(e["u"][m] - u0, e["v"][m] - v0)
        k = np.floor((a / (2 * np.pi) + 0.5) * sectors + self.r.random()).astype(np.int64) % sectors
        self.set(e, m, (1 << 20) + tag * 64 + k, kind)
        if scroll:
            self.paint[e["Y"][m], e["X"][m]] = MAT * (1 - scrolls(e["u"][m] - u0, e["v"][m] - v0, scroll))

    def motif(self, e, m, mo, u0, v0, turn=0.0, tag=0):
        """Lay a motif's pieces and paint at (u0, v0), turned by `turn` radians; returns what is left."""
        S = mo.S
        du, dv = e["u"][m] - u0, e["v"][m] - v0
        c, s = np.cos(turn), np.sin(turn)
        du, dv = du * c - dv * s, du * s + dv * c
        tx, ty = (du / mo.rad + 1) * S / 2, (1 - dv / mo.rad) * S / 2
        k = ndimage.map_coordinates(mo.a_keys, [ty, tx], order=0, cval=0)
        hit = np.zeros_like(m)
        hit[np.nonzero(m)[0][k > 0]] = True
        iy, ix = (np.clip(np.rint(t[k > 0]).astype(int), 0, S - 1) for t in (ty, tx))
        self.set(e, hit, (2 << 20) + tag * 4096 + k[k > 0], mo.a_kind[iy, ix])
        self.paint[e["Y"][hit], e["X"][hit]] = ndimage.map_coordinates(mo.a_paint, [ty[k > 0], tx[k > 0]], order=1)
        return m & ~hit

    def lattice(self, e, m, cell, band, kinds, angle=np.pi / 4, flower=None):
        """A trellis: fillets crossing on the diagonal, glass between; where they cross, a knot,
        or a four-petalled flower with a white heart. flower = (petal length, width, heart) px"""
        cell_k, band_k, knot_k = kinds
        u, v = e["u"][m], e["v"][m]
        a = (u * np.cos(angle) + v * np.sin(angle)) / cell
        b = (-u * np.sin(angle) + v * np.cos(angle)) / cell
        ra, rb, fa, fb = np.round(a), np.round(b), np.floor(a), np.floor(b)
        on_a, on_b = np.abs(a - ra) * cell < band / 2, np.abs(b - rb) * cell < band / 2
        if not band:
            on_a = on_b = np.zeros_like(a, bool)
        idx = lambda p, q: (p.astype(np.int64) + 512) * 1024 + (q.astype(np.int64) + 512)
        key = np.where(on_a & on_b, idx(ra, rb) + (1 << 21) * 3,
                       np.where(on_a, idx(ra, fb) + (1 << 21) * 4, np.where(on_b, idx(fa, rb) + (1 << 21) * 5, idx(fa, fb) + (1 << 21) * 6)))
        kind = np.where(on_a | on_b, band_k, cell_k)
        kind = np.where(on_a & on_b, knot_k, kind)
        if flower:
            fd, fp, fc = flower                 # petal distance, petal radius, heart radius
            da, db = (a - ra) * cell, (b - rb) * cell
            pet = (np.hypot(np.abs(da) - fd, db) < fp) | (np.hypot(da, np.abs(db) - fd) < fp)
            heart = np.hypot(da, db) < fc
            key = np.where(pet | heart, idx(ra, rb) + (1 << 21) * (3 + 4 * heart), key)
            kind = np.where(heart, WHITE, np.where(pet, band_k, kind))
        i = np.nonzero(m)[0]
        self.keys[e["Y"][i], e["X"][i]] = e["base"] + key
        self.kind[e["Y"][i], e["X"][i]] = kind


def paint(seed=1235):
    H = W = 2600
    C = (W / 2, H / 2)
    R = 1180
    r = noise.rng(seed)
    cart = Cartoon((H, W), C, r)

    # the flower in the middle: twelve red lobes, pearls, and a rose on blue
    r0, rl = 0.128 * R, 0.034 * R
    e = cart.element(0, 0, lambda u, v: lobed(u, v, r0, rl, 12), r0 + rl + 4)
    m = cart.bands(e, e["d"] < 2, [(12, RED, False), (8, WHITE, True)])
    rho = np.hypot(e["u"], e["v"])
    r1 = r0 - 24
    lobes = m & (rho >= r1)
    ang = np.arctan2(e["u"][lobes], e["v"][lobes])
    lobe = np.round(ang / (2 * np.pi / 12)).astype(np.int64) % 12
    cart.set(e, lobes, (3 << 20) + lobe, RED)
    for k in range(12):                              # a small white trefoil in every lobe
        t = 2 * np.pi * k / 12
        lu, lv = e["u"] - r0 * np.sin(t), e["v"] - r0 * np.cos(t)
        tre = np.zeros_like(m)
        for a in (0, 2 * np.pi / 3, 4 * np.pi / 3):
            tre |= np.hypot(lu - 4.5 * np.sin(t + a), lv - 4.5 * np.cos(t + a)) < 5.2
        cart.set(e, m & tre, (3 << 20) + 64 + k, WHITE)
    disc = cart.bands(e, m & (rho < r1), [(8, WHITE, True)], sdf=lambda u, v: np.hypot(u, v) - r1,
                      d=rho - r1, tag=1)
    rest = cart.motif(e, disc, rose(r1 - 20, r), 0, 0)
    cart.ground(e, rest, BLUE, sectors=10)

    for k in range(12):
        # petals: a dove in the head of each, flowers on a trellis in the tail
        tr, hr, h = 0.032 * R, 0.09 * R, (0.41 - 0.228) * R
        e = cart.element(0.228 * R, 30 * k + 15, lambda u, v: cone(u, v, tr, hr, h), h + hr + 4, origin=(0.0, h * 0.7))
        m = cart.bands(e, e["d"] < 2, [(11, RED, False), (7, WHITE, True)])
        rm = hr - 18
        rho = np.hypot(e["u"], e["v"] - h)
        head = m & (rho < rm)
        cart.lattice(e, m & ~head, 32, 0, (BLUE, RED, RED), flower=(5.8, 4.5, 4.0))
        inner = cart.bands(e, head, [(8, RED, False)], sdf=lambda u, v: np.hypot(u, v - h) - rm, d=rho - rm,
                           origin=(0.0, h), reach=rm + 4, tag=1)
        rest = cart.motif(e, inner, dove(rm - 10, r), 0, h)
        cart.ground(e, rest, BLUE, 0, h, sectors=5)

        # the lozenges: castles of Castile on red, lilies of France on blue
        if k % 2 == 0:
            a = 0.12 * R
            e = cart.element(0.64 * R, 30 * k, lambda u, v: diamond(u, v, a), a + 4)
            m = cart.bands(e, e["d"] < 2, [(12, BLUE, False), (8, WHITE, True)])
            rest = cart.motif(e, m, castle((a - 20) / np.sqrt(2) * 1.05, r), 0, 0)
            cart.ground(e, rest, RED, sectors=4, scroll=0)
        else:
            a = 0.088 * R
            e = cart.element(0.64 * R, 30 * k, lambda u, v: box(u, v, a, a), a * 1.42 + 4)
            m = cart.bands(e, e["d"] < 2, [(12, RED, False), (8, WHITE, True)])
            rest = cart.motif(e, m, fleur(a - 22, r), 0, 0)
            cart.ground(e, rest, BLUE, sectors=4)

        # half-moons against the rim, round side in: a trellis, and a rosette in a roundel
        sr, sc, rim = 0.205 * R, 0.99 * R, 0.985 * R
        moon = lambda u, v: np.maximum(np.hypot(u, v) - sr, np.hypot(u, v + sc) - rim)
        e = cart.element(sc, 30 * k, moon, sr + 4, origin=(0.0, -0.1 * R))
        m = cart.bands(e, e["d"] < 2, [(11, RED, False), (7, WHITE, True)])
        v0, rr = -0.105 * R, 0.07 * R
        rho = np.hypot(e["u"], e["v"] - v0)
        rnd = m & (rho < rr)
        cart.lattice(e, m & ~rnd, 24, 6, (BLUE, RED, RED))
        inner = cart.bands(e, rnd, [(8, BLUE, False), (6, WHITE, True)], sdf=lambda u, v: np.hypot(u, v - v0) - rr,
                           d=rho - rr, origin=(0.0, v0), reach=rr + 4, tag=1)
        rest = cart.motif(e, inner, rosette(rr - 17, r), 0, v0)
        cart.ground(e, rest, RED, 0, v0, sectors=6, scroll=0)

        # quatrefoils between the half-moons, a lily in every lobe, as at Chartres
        q = 0.07 * R
        e = cart.element(0.8 * R, 30 * k + 15, lambda u, v: quatrefoil(u, v, q), q + 4)
        m = cart.bands(e, e["d"] < 2, [(8, RED, False), (6, WHITE, True)])
        for j in range(4):
            t = np.pi / 2 * j
            m = cart.motif(e, m, fleur(0.5 * q - 12, r, small=True), 0.46 * q * np.sin(t), 0.46 * q * np.cos(t), turn=t, tag=j)
        heart = m & (np.hypot(e["u"], e["v"]) < 7)
        cart.set(e, heart, (3 << 20), RED)
        cart.ground(e, m & ~heart, BLUE, sectors=4, scroll=8.0)

    # north light: an even grey sky, a little brighter overhead
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sky = (1.0 - 0.14 * yy / H) * (1 + 0.05 * noise.field((H, W), 500, r))
    img, lead, lab = glass.window(cart.keys, cart.kind, GLASSES, cart.paint, sky, r, width=2.3, most=4200)
    print("pieces:", lab.max())

    # the stone: nearly black; the splayed edge of each opening catches a little of the glass light
    open_ = cart.open
    stone_d = ndimage.distance_transform_edt(open_ < 0.5)
    near = ndimage.gaussian_filter(img * open_[..., None], 4) / (ndimage.gaussian_filter(open_, 4)[..., None] + 1e-4)
    rr = np.hypot(xx - C[0], yy - C[1]) / R
    grain = 1 + 0.1 * noise.fbm((H, W), 40, r, octaves=5) + 0.06 * noise.field((H, W), 1.5, r)
    # round the rim, three roll mouldings, each lit on the side that faces the glass
    rolls = sum(np.exp(-((rr - c) * R / wd) ** 2) * g for c, wd, g in ((1.012, 5, 2.2), (1.036, 7, 1.6), (1.062, 6, 1.2)))
    hollows = sum(np.exp(-((rr - c) * R / 4) ** 2) for c in (1.024, 1.049))
    stone = lin("#0c0b0a")[None, None, :] * (grain * (1 + 0.8 * rolls - 0.5 * hollows))[..., None]
    wall = noise.smoothstep(1.075, 1.08, rr)
    course = 118
    joint = np.minimum(np.abs((yy + 30) % course - course / 2), 1e9)
    row = np.floor((yy + 30) / course)
    vj = np.abs((xx + (row % 2) * 97 + row * 31) % 190 - 95)
    joints = noise.smoothstep(2.5, 0.5, np.minimum(course / 2 - joint, vj))
    stone *= ((1 - 0.35 * wall * joints) * (1 - 0.25 * wall))[..., None]
    stone += near * (0.07 * np.exp(-stone_d / 2.0))[..., None]
    out = stone * (1 - open_[..., None]) + img * open_[..., None]
    return glass.halation(out)
