"""Cumulus over the Sayama Hills. Poster colour (gouache) on drawing paper.

An afternoon in early summer in the Sayama Hills, between Tokorozawa and Higashimurayama, where woods of
oak and camphor come down to the rice fields. A cumulus has been building over the hills since noon and
stands above them, lit from the upper left. The rice was planted a few weeks ago and is still short, so
the water between the rows holds the sky and the cloud. A lane runs in from the foot of the picture to a
shrine grove on the left, an old camphor and two oaks, and passes under a small torii into their shade;
day lilies are out in a stand of tall grass by the lane.

It is painted in poster colour, the cheap opaque paint in jars that the background painters of Studio Ghibli
used, on drawing paper, and the paint stops short of the edge of the sheet as it does on their boards. The
sky went on first while the paper was damp, broad strokes of blue and white worked into each other, deepest
at the top. The cloud went into it before it dried, head by head: the shadow first, lavender-grey, bluer
toward the floor; then the lights, opaque white warmed with a little yellow, laid over the top of each head
and blended down into the shadow with a damp brush, so that each sunlit top stands crisp against the blue
and the head behind it, and each shadowed side runs soft into the wet sky. The trees are built the way Kazuo
Oga builds them, from the darkest green up: a dark ground for each mass, then touch upon touch of the point
of the brush, each pass a little lighter and kept to where the light reaches further up the crowns, the last
dragged nearly dry so it catches the grain of the paper. The water is laid level, in strokes of what it
gives back; the rice goes over it tuft by tuft, each with its reflection, and the grass blade by blade, dark
first and the sunlit blades last. Every touch carries the paint's body: thickest where the brush landed,
thinning along its length until it breaks on the tooth, and thin enough at its edges for what is under it to
show.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import dabs, impasto, noise, plate, wash
from atelier.color import lin
from atelier.paper import Sheet

TITLE = "Cumulus over the Sayama Hills"
DATE = "2026"
MEDIUM = "Poster colour (gouache) on drawing paper"
AFTER = ("Kazuo Oga, the background paintings for Studio Ghibli: My Neighbor Totoro (1988), Kiki's Delivery "
         "Service (1989), Only Yesterday (1991), Princess Mononoke (1997)")
ROOM = "Paper and Water"
YEAR = 1988
PLACE = "Sayama Hills, Tokorozawa"
REGION = "East Asia"
NOTE = ("A summer cumulus over the wooded hills where Totoro is set, the young rice in the paddies holding the "
        "sky, and a lane running into the shade of a shrine grove. The trees are built touch upon touch, from "
        "the darkest green to the sunlit tops.")

H, W = 2200, 3400
EDGE = 44                                   # the cut edge of the sheet
PX0, PY0, PX1, PY1 = 150, 140, 3250, 2062   # where the paint reaches, give or take
PW, PH = PX1 - PX0, PY1 - PY0
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
SUN = np.float32([-0.7, -0.55, 0.45])
SUN /= np.linalg.norm(SUN)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette, mixed between its paints."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(np.asarray(t).astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def u2x(u):
    return PX0 + np.asarray(u) * PW


def v2y(v):
    return PY0 + np.asarray(v) * PH


def _s(r):
    return int(r.integers(1 << 31))


# the jars, each list from dark to light
SKY = pal("#1764a6", "#1d71b3", "#2580bf", "#318fc9", "#41a0d3", "#58afdb", "#74bee1", "#93cbe4", "#b3d8e6", "#c7e0e4")
CLOUD = pal("#848bb2", "#8f96ba", "#9ca3c3", "#a5abc9", "#b6bbd3", "#c8cbdc", "#d9dbe3", "#e7e7e7", "#f2f0e9",
            "#faf7ee", "#fffcf2")

# poster colour from the jar: hardly any body, it dries flat and matt
GOUACHE = dict(thick=0.006, grooves=0.02, lips=0.0, land=0.15, lift=0.0, tails=0.5, pickup=0.35, merge=3.0,
               spent=0.55, ends=(0.4, 0.15), taper=0.3, fray=2.0, flatten=0.4)


def support(r):
    """A sheet of white drawing paper, a little warm with age, cut square: a fine even tooth."""
    s = wash.rough((H, W), _s(r), tint="#f3efe4", margin=0, hill=3.2)
    d = np.minimum.reduce([YY - EDGE, H - 1 - EDGE - YY, XX - EDGE, W - 1 - EDGE - XX])
    d = d + 0.8 * noise.field((H, W), 300, r) + 0.25 * noise.field((H, W), 3, r)
    return Sheet(s.color, s.tooth, s.fiber, noise.smoothstep(-0.7, 0.7, d).astype(np.float32))


def reach(sheet, r):
    """Where the painter stopped: a loose rectangle well inside the sheet. The strokes of the sky end
    raggedly at its sides, the grass at its foot runs out in blades, and the last of each stroke is
    dragged thin, so the paint breaks on the tooth of the paper."""
    side = lambda n, *sc: sum(a * noise.line1d(n, s, r) for a, s in sc)
    left = PX0 + side(H, (9, 400), (12, 55), (3, 9))
    right = PX1 + side(H, (9, 400), (12, 55), (3, 9))
    top = PY0 + side(W, (6, 500), (4, 70), (1.5, 9))
    foot = PY1 + side(W, (8, 400), (6, 30), (5, 6))
    d = np.minimum.reduce([XX - left[:, None], right[:, None] - XX, YY - top[None, :], foot[None, :] - YY])
    thin = noise.smoothstep(-2, 22, d)
    catch = noise.smoothstep(0.62 - 0.95 * thin, 0.7 - 0.95 * thin, sheet.tooth + 0.08 * noise.field((H, W), 4, r))
    return (noise.smoothstep(-1.0, 1.0, d) * catch).astype(np.float32)


class Desk:
    """The painting as it goes on: the paper, the paint on it, and how wet it still is."""

    def __init__(self, sheet, r, seed):
        self.r, self.sheet = r, sheet
        self.hand = noise.rng(seed + 1)                 # how full the brush was, touch by touch
        self.img = sheet.color.copy()
        self.height = (0.3 * sheet.tooth).astype(np.float32)
        self.wet = np.zeros((H, W), np.float32)
        self.mottle = (0.7 * noise.field((H, W), 6, r) + 0.3 * noise.field((H, W), 30, r)).astype(np.float32)
        self.grain = noise.field((H, W), 1.3, r)

    def brush(self, paths, width, load, share=None, how=GOUACHE, **kw):
        """Strokes of a loaded brush along `paths`, each carrying the colours of `load` (N, K, 3)."""
        if len(paths):
            impasto.lay(self.img, self.height, paths, width, load, self.r, share=share, wet=self.wet,
                        **{**how, **kw})

    def dry(self):
        self.wet[:] = 0

    def fill(self, mask, colour, wet=True):
        """A coat of colour laid flat over `mask` with a big soft brush: it only shows where the strokes
        that go over it skip."""
        self.img += (colour - self.img) * mask[..., None]
        if wet:
            self.wet = np.maximum(self.wet, mask)

    def touch(self, polys, colours, dry=0.0):
        """Touches of poster colour from the point of a brush, laid in order, each over those before:
        polygons (N, V, 2) whose first and last vertices are where the brush landed and whose middle ones
        are where it lifted, each filled with its colour (N, 3). The paint has body. It is thickest where
        the brush landed and thins along the touch, until the last of it breaks on the tooth of the paper;
        at the edges of a touch, where the hairs carried less, it is thinner too, and what is under it
        shows; and it is never quite even, a shade lighter where it was rubbed thin over the grain. `dry` >
        0: the brush was nearly dry and left paint only on the tooth. The touches are laid at twice the
        size of the plate and the result brought down to it, so that their edges are true."""
        P = np.asarray(polys, np.float32)
        if not len(P):
            return
        x0, y0 = max(int(P[..., 0].min()) - 2, 0), max(int(P[..., 1].min()) - 2, 0)
        x1, y1 = min(int(P[..., 0].max()) + 3, W), min(int(P[..., 1].max()) + 3, H)
        if x1 <= x0 or y1 <= y0:
            return
        sl = (slice(y0, y1), slice(x0, x1))
        up = lambda f: np.repeat(np.repeat(f[sl], 2, 0), 2, 1)
        work = up(self.img)
        tooth, grain, mottle = up(self.sheet.tooth), up(self.grain), up(self.mottle)
        paper = (tooth, noise.smoothstep(0.3, 0.7, tooth + 0.3 * grain), mottle, grain)
        Q = 2 * (P - [x0, y0])
        C = np.broadcast_to(np.asarray(colours, np.float32), (len(P), 3))
        for part in np.array_split(np.arange(len(P)), max(1, min(12, len(P) // 1000))):
            self._touches(work, Q[part], C[part], dry, paper)
        self.img[sl] = work.reshape(y1 - y0, 2, x1 - x0, 2, 3).mean((1, 3))

    def _touches(self, work, Q, colours, dry, paper):
        tooth, hold, mottle, grain = paper
        n, V = Q.shape[:2]
        tags = Image.new("I", work.shape[1::-1], 0)
        mark = ImageDraw.Draw(tags)
        for i, q in enumerate(Q.reshape(n, -1).tolist()):
            mark.polygon(q, fill=i + 1)
        tag = np.asarray(tags, np.int32)                 # which touch is on top at each point
        k = np.maximum(tag - 1, 0)
        # how far along its touch the brush had got there, and how broad the touch is
        s0 = (Q[:, 0] + Q[:, -1]) / 2
        d = (Q[:, V // 2 - 1] + Q[:, V // 2]) / 2 - s0
        length = np.hypot(d[:, 0], d[:, 1]) + 1e-3
        area = 0.5 * np.abs((Q[..., 0] * np.roll(Q[..., 1], -1, 1) - np.roll(Q[..., 0], -1, 1) * Q[..., 1]).sum(1))
        yy, xx = np.mgrid[0:tag.shape[0], 0:tag.shape[1]].astype(np.float32) + 0.5
        t = np.clip(((xx - s0[k, 0]) * d[k, 0] + (yy - s0[k, 1]) * d[k, 1]) / (length[k] ** 2), 0, 1)
        thin, body = self.hand.uniform(0.15, 0.45, n), self.hand.uniform(0.0, 0.1, n)
        edge = ndimage.grey_dilation(tag, 5) != ndimage.grey_erosion(tag, 5)
        broad = noise.smoothstep(4.0, 12.0, 2 * area / length)[k]
        op = (tag > 0) * (1 - thin[k] * noise.smoothstep(0.35, 1.0, t) ** 1.5)
        op *= 1 - noise.smoothstep(0.7, 1.0, t) * (1 - hold)                 # the end breaks on the tooth
        op *= 1 - 0.45 * broad * edge * (1 - hold)                             # thin at the edges
        if dry:
            op *= noise.smoothstep(dry - 0.1, dry + 0.1, tooth + 0.1 * mottle)
        col = colours[k] * (1 + body[k] * (t - 0.3) + 0.05 * mottle + 0.03 * grain + 0.04 * (tooth - 0.5))[..., None]
        work += (col - work) * op[..., None]


def at(P):
    P = np.asarray(P)
    return np.clip(P[..., 1].astype(int), 0, H - 1), np.clip(P[..., 0].astype(int), 0, W - 1)


def loads(colours, tone, r, spread=0.04, step=1.0):
    """The brush for each stroke: the paint for its tone, with the next paint up or down the row of jars
    streaked into it, and a little of a third."""
    N = len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 1)
    d = step / (len(colours) - 1)
    j = np.clip(t + r.choice([-1, 1], N) * d, 0, 1)
    k = np.clip(t + r.choice([-1, 1], N) * 0.5 * d, 0, 1)
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.05, 0.2, N)], 1)
    return np.stack([ramp(colours, t), ramp(colours, j), ramp(colours, k)], 1), share


def strew(spacing, odds, r, box=None):
    """Points about `spacing` apart, never in rows, kept where `odds` (H,W) says so."""
    x0, y0, x1, y1 = box or (0, 0, W, H)
    P = dabs.scatter((int(y1 - y0) + 2 * spacing, int(x1 - x0) + 2 * spacing), spacing, r) - spacing + [x0, y0]
    P = P.astype(np.float32)
    keep = r.random(len(P)) < (odds[at(P)] if np.ndim(odds) else odds)
    return P[keep]


# ---------------------------------------------------------------------------------------------- the sky

def sky_tone(x, y):
    v = (y - PY0) / PH
    return 0.06 + 0.84 * noise.smoothstep(-0.05, 0.6, v) ** 1.15 + 0.04 * noise.smoothstep(0.5, 1.0, (x - PX0) / PW)


def sky(desk, r):
    """The sky, laid while the paper was still damp: long strokes of a broad brush from side to side, row
    under row, cerulean with a little white, deepest at the top and paler and greener toward the hills."""
    desk.fill(noise.smoothstep(v2y(0.66), v2y(0.62), YY), ramp(SKY, sky_tone(XX, YY)))
    rows = []
    y = PY0 - 60
    while y < v2y(0.60):
        w = r.uniform(40, 62)
        x = PX0 - r.uniform(40, 160)
        while x < PX1 + 60:
            L = r.uniform(380, 900)
            a = r.normal(0.03, 0.05)
            pts = np.array([[x, y], [x + L / 2, y + L / 2 * a + r.normal(0, 8)], [x + L, y + L * a + r.normal(0, 12)]])
            rows.append((pts, w))
            x += L * r.uniform(0.55, 0.85)
        y += w * r.uniform(1.0, 1.35)
    order = r.permutation(len(rows))
    paths = [rows[i][0] for i in order]
    w = np.array([rows[i][1] for i in order])
    mid = np.array([p.mean(0) for p in paths])
    load, share = loads(SKY, sky_tone(mid[:, 0], mid[:, 1]), r, spread=0.035)
    desk.brush(paths, w, load, share, pickup=0.5, merge=5.0, spent=0.25, fray=2.0)


# ---------------------------------------------------------------------------------------------- the cloud

# the cloud as heaps of vapour: u, v of the centre, half-width, half-height, the size of a head (px),
# how far off (haze), and the v of the floor it stands on
MASSES = [(0.640, 0.395, 0.185, 0.060, 120, 0.00, 0.47),     # the great cloud's broad base, behind the hills
          (0.630, 0.300, 0.140, 0.085, 130, 0.00, 0.47),     # its body
          (0.612, 0.190, 0.092, 0.075, 108, 0.00, 0.47),     # the tower
          (0.598, 0.098, 0.058, 0.045, 78, 0.00, 0.47),      # the heads boiling up at the top
          (0.702, 0.220, 0.058, 0.060, 84, 0.00, 0.47),      # its right shoulder
          (0.532, 0.262, 0.060, 0.050, 80, 0.00, 0.47),      # and its left
          (0.935, 0.235, 0.070, 0.052, 72, 0.18, 0.31),      # a smaller cumulus off to the right
          (0.468, 0.150, 0.036, 0.020, 40, 0.28, 0.17),      # small clouds drifting in the blue
          (0.795, 0.090, 0.030, 0.016, 34, 0.30, 0.104),
          (0.870, 0.402, 0.090, 0.022, 46, 0.62, 0.432),     # far low clouds along the hills
          (1.000, 0.390, 0.060, 0.020, 40, 0.66, 0.412)]


def heads(r):
    """The heads of vapour: a few big ones filling each mass and many small ones crowding its upper edge,
    where it boils up; the lower a head, the nearer the eye, so that each one's sunlit top stands in
    front of the one above it; and small heads swelling out of the sunlit rims. -> (N, 8): x, y, radius,
    how far forward, haze, floor y, top y of its cloud, whether it is one of the small heads"""
    out = []
    for u, v, a, b, s, haze, floor in MASSES:
        x, y, a, b = u2x(u), v2y(v), a * PW, b * PH
        fl = v2y(floor)
        top = min(v2y(m[1] - m[3]) for m in MASSES if m[6] == floor)
        n = int(0.65 * a * b / (s * s)) + 2
        for t, k in zip(r.uniform(0, 2 * np.pi, n), 0.62 * np.sqrt(r.uniform(0, 1, n))):
            yh = y + b * k * np.sin(t)
            out.append((x + a * k * np.cos(t), yh, 1.45 * s * r.uniform(0.8, 1.2), r.uniform(0, 0.3) * s + 0.7 * (yh - fl),
                        haze, fl, top))
        for t in r.uniform(-0.97 * np.pi, -0.03 * np.pi, int(np.pi * (a + b) / (1.8 * s)) + 1):
            R = s * np.clip(np.exp(r.normal(-0.45, 0.35)), 0.3, 1.0)
            yh = y + (b - 0.6 * R) * np.sin(t)
            out.append((x + (a - 0.6 * R) * np.cos(t), yh, R, 0.35 * s + r.uniform(0, 0.3) * R + 0.7 * (yh - fl),
                        haze, fl, top))
    L = np.array(out, np.float32)
    small = []                                  # small heads swelling out of the sunlit rims
    for x, y, R, z, haze, fl, top in L:
        t = np.arctan2(SUN[1], SUN[0]) + r.uniform(-1.3, 1.3)
        c = np.array([x, y]) + R * np.array([np.cos(t), np.sin(t)])
        if r.random() < 0.45 or c[1] > fl - 20 or haze > 0.4 \
                or (np.hypot(L[:, 0] - c[0], L[:, 1] - c[1]) < 0.97 * L[:, 2]).any():
            continue
        q = R * r.uniform(0.18, 0.34)
        rho = R - 0.45 * q
        small.append((x + rho * np.cos(t), y + rho * np.sin(t), q, z + np.sqrt(max(R * R - rho * rho, 0)) - 0.35 * q,
                      haze, fl, top))
    return np.vstack([np.column_stack([L, np.zeros(len(L))]), np.column_stack([np.array(small, np.float32).reshape(-1, 7),
                                                                               np.ones(len(small))])]).astype(np.float32)


def domes(spheres, r, wobble=0.1, floor=None):
    """Front surfaces of a heap of domes (x, y, radius, z): at each pixel, which dome is nearest the eye,
    how far forward its surface is and which way it faces. -> inside, who, depth, nx, ny"""
    Z = np.full((H, W), -1e5, np.float32)
    who = np.full((H, W), -1, np.int32)
    NX, NY = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    wx, wy = (wobble * noise.fbm((H, W), 60, r, octaves=3) for _ in range(2))
    for i, (x, y, R, z) in enumerate(spheres[:, :4]):
        y0, y1 = max(int(y - 1.25 * R), 0), min(int(y + 1.25 * R) + 1, H)
        x0, x1 = max(int(x - 1.25 * R), 0), min(int(x + 1.25 * R) + 1, W)
        if y1 <= y0 or x1 <= x0:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        dx, dy = (XX[sl] - x) / R + wx[sl], (YY[sl] - y) / R + wy[sl]
        d2 = dx * dx + dy * dy
        ok = d2 < 1
        if floor is not None:
            ok &= YY[sl] < floor[i]
        zz = np.where(ok, z + R * np.sqrt(np.maximum(1 - d2, 0)), -1e5)
        up = zz > Z[sl]
        Z[sl][up], who[sl][up], NX[sl][up], NY[sl][up] = zz[up], i, dx[up], dy[up]
    inside = Z > -1e4
    return inside, who, np.where(inside, Z, 0).astype(np.float32), NX, NY


def carry(f, m, s):
    """`f` averaged over the paint in `m` within about `s` px: carried a little past the edge of `m`."""
    k = (s, s) + (0,) * (np.ndim(f) - 2)
    w = ndimage.gaussian_filter(m, s)
    return ndimage.gaussian_filter(f * (m if np.ndim(f) == 2 else m[..., None]), k) / (
        (w if np.ndim(f) == 2 else w[..., None]) + 1e-4)


def cloud_model(L, r):
    """How the sun falls on the cloud. It comes from the upper left and low, so that it rakes across each
    head and leaves the lower right of it in shadow; and it falls on the whole cloud as on one great
    heap, so that the heads high up and on the left are lit nearly all over, and those low down and on
    the right catch it only along their tops. The shadow is lavender-grey, bluer toward the floor and in
    the creases where a head tucks under its neighbour, lit a little from below by what the ground gives
    back, and from above by the sky. -> inside, tone 0..1, sun (the cap of each head), the light given
    back, haze"""
    sag = 6 * noise.line1d(W, 90, r)
    fl = L[:, 5] + sag[np.clip(L[:, 0].astype(int), 0, W - 1)]
    inside, who, Z, NX, NY = domes(L, r, wobble=0.08, floor=fl)
    m = inside.astype(np.float32)
    k = np.maximum(who, 0)
    rho = np.maximum(np.hypot(NX, NY), 1)
    nl = np.stack([NX / rho, NY / rho, np.sqrt(np.clip(1 - NX * NX - NY * NY, 0, 1))], -1)
    ll = nl @ unit([-0.75, -0.6, 0.28])
    g = np.unique(L[:, 5], return_inverse=True)[1]       # which cloud each head belongs to
    x0 = np.array([(L[g == j, 0] - L[g == j, 2]).min() for j in range(g.max() + 1)])
    x1 = np.array([(L[g == j, 0] + L[g == j, 2]).max() for j in range(g.max() + 1)])
    side = np.clip((XX - x0[g[k]]) / (x1[g[k]] - x0[g[k]]), 0, 1)
    low = np.clip((YY - L[k, 6]) / (L[k, 5] - L[k, 6] + 1), 0, 1)
    heap = np.clip(1.25 - 0.75 * low ** 1.3 - 0.6 * side, 0, 1)
    tucked = noise.smoothstep(0.85, 0.99, ndimage.gaussian_filter(m, 10))
    crease = tucked * (1 - nl[..., 2]) ** 1.1 * noise.smoothstep(-0.3, 0.3, NY)
    zc = np.where(inside, L[k, 3], -1e4).astype(np.float32)          # in the hollow behind the edge of a nearer head
    near = ndimage.maximum_filter(np.where(L[k, 7] > 0, -1e4, zc), size=31)
    hollow = ndimage.gaussian_filter(noise.smoothstep(5, 60, near - zc) * m, 5) * m
    lit = ll + 0.3 * (heap - 0.5) + 0.045 * noise.fbm((H, W), 60, r, octaves=3) + 0.04 * noise.field((H, W), 12, r) \
        - 0.6 * crease - 0.42 * hollow
    sun = noise.smoothstep(-0.1, 0.22, lit)
    half = noise.smoothstep(-0.42, 0.0, lit)
    back = noise.smoothstep(0.15, 0.75, nl @ unit([0.3, 0.85, 0.45])) * (1 - crease) * (1 - half)
    above = noise.smoothstep(0.2, 0.9, -nl[..., 1])
    shade = 0.25 - 0.17 * low - 0.1 * crease + 0.08 * above * (1 - low) + 0.1 * back
    top = (0.8 + 0.2 * noise.smoothstep(0.1, 0.65, lit)) * (0.6 + 0.4 * heap)    # fullest where the sun is square on
    tone = shade + (top - shade) * (0.75 * sun + 0.25 * half)
    return m, tone * m, sun * m, back, L[k, 4] * m


def cloud(desk, r):
    """The cloud, painted into the wet sky in poster colour. Its shadow went on first, lavender-grey
    worked with blue toward the floor; then into it, while wet, the lights, opaque white warmed with a
    little yellow, laid over the cap of each head and blended down into the shadow with a damp brush so
    that the turn is soft. Against the blue, the lit side of a head is cut crisp and its shadowed side
    runs soft into the wet sky."""
    L = heads(r)
    m, tone, sun, back, haze = cloud_model(L, r)
    soft = carry(tone, m, 3)                            # edges between heads lost in the shade, found in the sun
    tone = soft + (tone - soft) * noise.smoothstep(0.4, 0.75, soft)
    tone += (0.02 * noise.field((H, W), 26, r) + 0.008 * noise.field((H, W), 6, r)) * (1 - 0.8 * sun)
    tone += (0.8 - tone) * 0.8 * haze                    # the far clouds, with the air between
    col = ramp(CLOUD, tone)
    col += (lin("#c8bccb") - col) * (0.4 * back)[..., None]
    col += (desk.img - col) * (0.5 * haze)[..., None]
    col = np.where(m[..., None] > 0.5, ndimage.gaussian_filter(col, (0.6, 0.6, 0)), carry(col, m, 5))
    wet = noise.warp(m, 4 * noise.field((H, W), 14, r), 4 * noise.field((H, W), 14, r))
    wet = noise.smoothstep(0.1, 0.9, ndimage.gaussian_filter(wet, 2.2) + 0.12 * noise.field((H, W), 3, r))
    crisp = noise.smoothstep(0.2, 0.55, carry(sun, m, 6)) * (1 - 1.4 * haze)
    a = ndimage.gaussian_filter(m, 0.7) * crisp + wet * (1 - crisp)
    desk.img += (col - desk.img) * a[..., None]


# ---------------------------------------------------------------------------------------------- the woods

# greens, each list from the shade under the trees, blue-black, to the tops in the sun, yellow-green
FOREST = pal("#0c201f", "#112a27", "#16352d", "#1d4232", "#255036", "#2f6039", "#3b703c", "#4a823f", "#5c9342",
             "#71a446", "#8ab54d", "#a5c65a", "#c2d76e")
RIDGE = pal("#1f3f3d", "#264a45", "#2e574c", "#386350", "#436f54", "#507c58", "#5f895c", "#6f9660", "#81a466",
            "#96b36f")
FAR = pal("#4c7375", "#567d7d", "#628784", "#6f928b", "#7c9c91", "#8aa798", "#9ab2a0")


def leaves(x, y, length, width, ang, belly=0.32, m=6, bow=0.0):
    """Touches of the point of a round brush: each lands at (x, y), travels `length` px along `ang`,
    swelling to `width` a `belly` of the way along, and lifts to a point; `bow` curves it to one side by
    that fraction of its length. -> (N, 2m, 2) polygons"""
    t = np.linspace(0, 1, m)
    half = 0.5 * np.where(t < belly, np.sin(0.5 * np.pi * np.minimum(t / belly, 1)) ** 0.5,
                          np.cos(0.5 * np.pi * np.clip((t - belly) / (1 - belly), 0, 1)) ** 0.9)
    along, side = length[:, None] * t, width[:, None] * half
    mid = 4 * np.asarray(bow, np.float32).reshape(-1, 1) * length[:, None] * t * (1 - t)
    px_ = np.concatenate([along, along[:, ::-1]], 1)
    py_ = np.concatenate([mid + side, (mid - side)[:, ::-1]], 1)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return np.stack([x[:, None] + c * px_ - s * py_, y[:, None] + s * px_ + c * py_], -1).astype(np.float32)


def sprigs(P, size, ang, r, k=(2, 4), fan=0.8, slim=(0.34, 0.52)):
    """Each touch a few leaves of the brush point set down from about the same place and fanned round
    `ang`, so its edge is serrated. -> polygons (M, 12, 2), and for each the touch it belongs to"""
    n = len(P)
    idx = np.repeat(np.arange(n), r.integers(k[0], k[1] + 1, n))
    m = len(idx)
    L = size[idx] * r.uniform(0.55, 1.0, m)
    a = ang[idx] + r.uniform(-fan, fan, m)
    off = size[idx] * 0.22
    x = P[idx, 0] + r.normal(0, 1, m) * off - np.cos(a) * L * 0.3
    y = P[idx, 1] + r.normal(0, 1, m) * off - np.sin(a) * L * 0.3
    return leaves(x, y, L, L * r.uniform(*slim, m), a, m=8), idx


def blots(P, size, ang, r, m=16, lobes=(2, 5), rough=0.2, squash=(0.55, 0.85)):
    """Dabs of the brush pressed down and turned a little as it lifted: each an irregular round of about
    `size` px, longer along `ang`, its edge broken into a few lobes. -> polygons (N, m, 2)"""
    n = len(P)
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)[None]
    k = r.integers(lobes[0], lobes[1] + 1, n)[:, None]
    rad = 1 + rough * np.sin(k * th + r.uniform(0, 2 * np.pi, (n, 1))) + 0.13 * r.normal(0, 1, (n, m))
    a = 0.5 * size[:, None] * np.clip(rad, 0.5, 1.5)
    lx, ly = a * np.cos(th), a * r.uniform(*squash, (n, 1)) * np.sin(th)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return np.stack([P[:, :1] + c * lx - s * ly, P[:, 1:] + s * lx + c * ly], -1).astype(np.float32)


def skyline(knots, r, wob=(6, 170), fine=(2.5, 35)):
    """y of a line across the picture through (u, v) knots, never quite smooth."""
    u, v = np.array(knots, np.float32).T
    y = v2y(np.interp((np.arange(W) - PX0) / PW, u, v))
    y = ndimage.gaussian_filter1d(y, 50) + wob[0] * noise.line1d(W, wob[1], r) + fine[0] * noise.line1d(W, fine[1], r)
    return y.astype(np.float32)


def hillside(top, foot, x0, x1, R, r, depth=1.4):
    """The crowns of the woods packed over the face of a hill, between its skyline and its foot, smaller
    toward the top where they are further off, now and then a big old tree among them: domes
    (x, y, radius, z)."""
    out = []
    x = x0
    while x < x1:
        rr = R * np.clip(np.exp(r.normal(0.1, 0.35)), 0.6, 2.2)
        out.append((x, top[int(np.clip(x, 0, W - 1))] + 0.55 * rr, rr))
        x += rr * r.uniform(0.7, 1.3)
    y0 = int(top[max(x0, 0):min(x1, W)].min())
    y1 = int(foot[max(x0, 0):min(x1, W)].max())
    P = dabs.scatter((y1 - y0 + 2 * R, x1 - x0), R * 0.9, r) + [x0, y0 - R]
    for x, y in P:
        xi = int(np.clip(x, 0, W - 1))
        f = (y - top[xi]) / max(foot[xi] - top[xi], 1)
        if 0.1 < f < 1.05:
            out.append((x, y, R * np.clip(np.exp(r.normal(0, 0.3)), 0.55, 2.0) * (0.75 + 0.45 * f)))
    a = np.array(out, np.float32)
    xi = np.clip(a[:, 0].astype(int), 0, W - 1)
    z = (a[:, 1] - top[xi]) * depth + 0.3 * a[:, 2]
    return np.column_stack([a, z]).astype(np.float32)


def unit(n):
    n = np.asarray(n, np.float32)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def leafage(spheres, r, big=(-0.05, -0.55, 0.83), wobble=0.12, local=0.6, gather=60.0, crease=0.5, vary=0.06):
    """How the sun falls on a mass of foliage made of domes: each crown or clump turns its own dome to
    the sun, the whole mass its broad face (`big`, a direction or a map); where one clump tucks in under
    another it is dark, and no two clumps are quite the same green. -> inside, tone 0..1, the way the
    leaves hang (an angle per pixel), which dome is in front"""
    inside, who, Z, NX, NY = domes(spheres, r, wobble)
    nz = np.sqrt(np.clip(1 - NX * NX - NY * NY, 0, 1))
    lam = np.clip(np.stack([NX, NY, nz], -1) @ SUN, 0, 1)
    lb = np.clip(unit(big) @ SUN, 0, 1)
    tucked = noise.smoothstep(0.82, 0.98, ndimage.gaussian_filter(inside.astype(np.float32), gather / 4))
    tone = (local * lam + (1 - local) * lb) * (1 - crease * tucked * (1 - nz) ** 1.5)
    tone += vary * r.normal(0, 1, len(spheres)).astype(np.float32)[np.maximum(who, 0)]
    tone += 0.04 * noise.fbm((H, W), gather, r, octaves=3)
    hang = np.arctan2(NY + 0.9, NX)                      # outward from the clump, and down
    return inside.astype(np.float32), tone.astype(np.float32), hang.astype(np.float32), who


def build(desk, inside, tone, hang, colours, leaf, r, cover, top=1.0, spacing=0.5, k=(2, 4), dry=0.55,
          edge=0.4, box=None, shrink=0.25, jitter=0.015, cool=None, other=None):
    """A mass of foliage the way Oga builds it: its darkest green first, in touches close enough to cover;
    then pass after pass of touches each a little lighter, kept to the part of the mass the light reaches
    (`cover`, the share of it each pass covers), so that each sits on the one before as a clump with a
    crisp serrated edge; the last, the lightest, dragged nearly dry along the very tops. At the edge of
    the mass the touches are sprays of leaves, inside it dabs. Where `cool` (H,W) says so the greens are
    mixed from the `other` row of jars."""
    soft = ndimage.gaussian_filter(inside, leaf * 0.3)
    desk.fill(noise.smoothstep(0.6, 0.9, ndimage.gaussian_filter(inside, leaf * 0.5)), ramp(colours, 0.0), wet=False)
    if box is None:
        ys, xs = np.nonzero(inside > 0.5)
        box = (int(xs.min()) - leaf, int(ys.min()) - leaf, int(xs.max()) + leaf, int(ys.max()) + leaf)
    body = noise.smoothstep(edge - 0.1, edge + 0.1, soft)
    t = tone[body > 0.5]
    n = len(cover)
    for j, frac in enumerate(cover):
        odds = body
        if j:
            lo = np.quantile(t, 1 - frac)
            odds = odds * noise.smoothstep(lo - 0.01, lo + 0.03, tone)
        last = j == n - 1 and dry
        P = strew(leaf * spacing * (1.6 if last else 1.0), odds, r, box)
        if not len(P):
            continue
        c = np.clip(top * j / (n - 1) + r.normal(0, jitter, len(P)), 0, 1)
        size = leaf * r.uniform(0.75, 1.25, len(P)) * (1 - shrink * j / n)
        a = hang[at(P)] + r.normal(0, 0.25, len(P))
        rim = soft[at(P)] < 0.8
        sp, idx = sprigs(P[rim], size[rim], a[rim], r, k)
        bl = blots(P[~rim], size[~rim], a[~rim], r)
        # some touches carry a little lighter paint on the side the brush was turned toward the light
        two = (~rim) & (r.random(len(P)) < 0.4)
        lift = np.array([SUN[0], SUN[1]]) / np.hypot(SUN[0], SUN[1])
        sub = blots(P[two] + lift * 0.15 * size[two, None], 0.55 * size[two], a[two], r)
        col = ramp(colours, c)
        col2 = ramp(colours, np.clip(c[two] + 0.04, 0, 1))
        if cool is not None:
            m = cool[at(P)][:, None]
            col = col * (1 - m) + ramp(other, c) * m
            col2 = col2 * (1 - m[two]) + ramp(other, np.clip(c[two] + 0.04, 0, 1)) * m[two]
        desk.touch(np.concatenate([sp, bl, sub]), np.concatenate([col[rim][idx], col[~rim], col2]),
                   dry=dry if last else 0.0)


RIDGE_COOL = pal("#173634", "#1d403d", "#244b45", "#2b564b", "#356150", "#406d56", "#4d795b", "#5b8660", "#6b9366",
                 "#7ea06c")


def hills(desk, r):
    """The Sayama Hills across the paddies: a far ridge, pale and blue with the air between, and the near
    one, its woods of oak and chestnut a canopy of round crowns, each lit on the side toward the sun,
    some groups of them catching more of it than others, and here and there a stand of cedars, darker."""
    far = skyline([(0.35, 0.425), (0.5, 0.42), (0.64, 0.414), (0.78, 0.418), (0.9, 0.425), (1.05, 0.436)], r,
                  wob=(5, 140))
    near = skyline([(0.3, 0.425), (0.4, 0.44), (0.5, 0.452), (0.6, 0.462), (0.7, 0.452), (0.8, 0.47), (0.9, 0.462),
                    (1.05, 0.474)], r)
    foot = skyline([(0.3, 0.588), (0.6, 0.584), (0.8, 0.586), (1.05, 0.582)], r, wob=(3, 300), fine=(1.5, 40))
    x0, x1 = int(u2x(0.33)), PX1 + 60
    s = hillside(far, near + 60, x0, x1, 30, r, depth=1.0)
    inside, tone, hang, _ = leafage(s, r, local=0.4, crease=0.2, vary=0.04)
    build(desk, inside, tone, hang, FAR, 16, r, (1.0, 0.5, 0.2), top=0.85, dry=0.0, k=(1, 2), jitter=0.01)
    s = hillside(near, foot + 25, x0, x1, 40, r)
    cedar = []                                           # stands of cedar: narrow spires, a dome on a dome
    for u in (0.52 + r.uniform(-0.03, 0.03), 0.86 + r.uniform(-0.03, 0.03)):
        for _ in range(int(r.integers(2, 5))):
            x = u2x(u) + r.normal(0, 45)
            yb = near[int(np.clip(x, 0, W - 1))] + r.uniform(30, 110)
            ht = r.uniform(80, 190)
            for f in np.linspace(0, 1, 7):
                cedar.append((x + r.normal(0, 2), yb - f * ht, 26 * (1 - 0.8 * f) + 4, 400 + 30 * f))
    s = np.vstack([s, np.array(cedar, np.float32)])
    kind = np.r_[np.clip(r.normal(0.25, 0.25, len(s) - len(cedar)), 0, 1), np.ones(len(cedar))].astype(np.float32)
    inside, tone, hang, who = leafage(s, r, local=0.55, crease=0.55, vary=0.07)
    cool = kind[np.maximum(who, 0)]
    tone += 0.14 * noise.fbm((H, W), 300, r, octaves=2) - 0.1 * cool
    build(desk, inside, tone, hang, RIDGE, 15, r, (1.0, 0.72, 0.5, 0.32, 0.17, 0.09), top=1.0, dry=0.45,
          cool=cool, other=RIDGE_COOL)
    return np.minimum(far, near), foot


# the trees of the shrine grove: u of the root, v of the ground, the height and the spread of the tree (as
# fractions of the height and the width of the picture), how far forward
TREES = [(0.13, 0.632, 0.61, 0.34, 420), (0.335, 0.632, 0.47, 0.2, 380), (0.425, 0.626, 0.27, 0.1, 360)]
GROUND = 0.632      # v of the ground the grove stands on
BARK = pal("#2a2622", "#38322b", "#4a4237", "#625a4b", "#7d7562", "#9a917a")


def tree(xb, yb, height, spread, z0, r):
    """A camphor as the painter lays it out before the leaves go on: a short trunk forking into limbs that
    reach up and out, a clump of leaves at the end of each limb and of each side branch, and a few more
    inside the crown so that it is not hollow. -> limbs [(control points, half-width)], clumps
    (x, y, R, z), the crown as one dome"""
    R = spread / 2.9
    fork = np.array([xb + r.normal(0, 0.05) * R, yb - 0.36 * height])
    reach_ = height - 0.36 * height - 1.2 * R
    limbs = [(np.array([[xb, yb + 20], [xb + r.normal(0, 6), (yb + fork[1]) / 2], fork]), 0.034 * spread)]
    groups = []
    n = int(r.integers(5, 8))
    for a in -np.pi / 2 + np.linspace(-1.1, 1.1, n) + r.normal(0, 0.12, n):
        L = reach_ * r.uniform(0.6, 1.1) * (1 - 0.25 * abs(np.cos(a)))
        end = fork + L * np.array([1.4 * np.cos(a), np.sin(a)])
        mid = (fork + end) / 2 + r.normal(0, 0.08 * L, 2)
        limbs.append((np.array([fork, mid, end]), 0.02 * spread * r.uniform(0.8, 1.1)))
        gr = R * r.uniform(0.65, 1.2)
        groups.append((end[0], end[1] - 0.2 * gr, gr, z0 + r.uniform(-0.2, 0.4) * R))
        if r.random() < 0.85:
            s0 = fork + (end - fork) * r.uniform(0.4, 0.65)
            b = a + r.choice([-1, 1]) * r.uniform(0.5, 0.9)
            e2 = s0 + L * r.uniform(0.35, 0.55) * np.array([np.cos(b), np.sin(b)])
            limbs.append((np.array([s0, (s0 + e2) / 2 + r.normal(0, 8, 2), e2]), 0.01 * spread))
            g2 = R * r.uniform(0.55, 0.75)
            groups.append((e2[0], e2[1] - 0.2 * g2, g2, z0 + r.uniform(-0.1, 0.45) * R))
    c = fork + np.array([0, -0.55 * reach_])
    for _ in range(2):
        p = c + r.normal(0, 0.3 * R, 2)
        groups.append((p[0], p[1], R * r.uniform(0.8, 1.0), z0 - 0.5 * R))
    G = np.array(groups)                                 # no clump left hanging in the air on its own
    d = np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1]) - 0.85 * (G[:, None, 2] + G[None, :, 2])
    np.fill_diagonal(d, 1e9)
    groups = [tuple(g) for g, k in zip(G, d.min(1)) if k < 0]
    return limbs, groups, (c[0], c[1], 1.5 * R + 0.3 * reach_, z0)


def billows(groups, r):
    """Smaller clumps swelling out of each big one, over its side toward the eye and the sky."""
    out = [tuple(g) for g in groups]
    for x, y, gr, z in groups:
        for _ in range(int(4 + 5 * (gr / 150) ** 2)):
            th = r.uniform(-np.pi, np.pi)
            el = np.arccos(r.uniform(0.1, 1.0))
            d = np.array([np.sin(el) * np.cos(th), np.sin(el) * np.sin(th) * 0.85 - 0.12, np.cos(el)])
            rr = gr * r.uniform(0.3, 0.5)
            b = np.array([x, y, z]) + 0.85 * gr * d
            out.append((b[0], b[1], rr, b[2] - 0.2 * rr))
    return np.array(out, np.float32)


def bark(desk, limbs, r):
    """Trunks and limbs, dark in the shade of the crowns; down the trunks the bark in ridges, and a lit
    streak of grey-green on the side the sun finds through a gap."""
    paths, wd, tn = [], [], []
    for P, w in limbs:
        paths.append(P)
        wd.append(w)
        tn.append(0.12)
        if w > 25:
            for f in r.uniform(-0.75, 0.6, 5):
                paths.append(P + [f * w + r.normal(0, 2), 0] + r.normal(0, 3, P.shape))
                wd.append(r.uniform(0.08, 0.16) * w)
                tn.append(r.uniform(0.0, 0.3))
            paths.append(P + [-0.5 * w, 0])
            wd.append(0.25 * w)
            tn.append(0.55)
    load, share = loads(BARK, np.array(tn), r, spread=0.05)
    desk.brush(paths, np.array(wd), load, share, pickup=0.2, merge=2.0, spent=0.4, fray=2.5, taper=0.5)


OAK = pal("#16352a", "#1e432f", "#285233", "#336336", "#407539", "#4f883c", "#62993f", "#78ab45", "#91bc4e",
          "#abcb5f", "#c6db7a")


def grove(desk, r):
    """The shrine grove: an old camphor, its crown heaped up dark against the sky, and beside it, a little
    further back, two oaks of a lighter, yellower green; a dark space under them where their trunks stand,
    and at their feet the undergrowth."""
    gl = v2y(GROUND)
    edge = skyline([(-0.1, GROUND + 0.006), (0.3, GROUND + 0.004), (0.4, GROUND - 0.004), (0.46, 0.6)], r,
                   wob=(4, 90), fine=(3, 18))
    shade = noise.smoothstep(-1, 1, edge[None, :] - YY) * noise.smoothstep(-1, 1, YY - v2y(0.42)) \
        * noise.smoothstep(u2x(0.47), u2x(0.43), XX + 120 * (YY - gl) / PH)
    desk.fill(shade, ramp(FOREST, 0.06), wet=False)
    box = (PX0 - 60, int(v2y(0.42)), int(u2x(0.48)), int(gl) + 20)
    P = strew(13, shade * 0.85, r, box)                  # the dark under the trees, in big dull touches
    n = len(P)
    low = noise.smoothstep(v2y(0.5), gl, P[:, 1])
    desk.touch(blots(P, r.uniform(18, 42, n), -np.pi / 2 + r.normal(0, 0.6, n), r),
               ramp(FOREST, np.clip(r.uniform(0.02, 0.16, n) + 0.08 * low, 0, 1)))
    xs = r.uniform(PX0 - 40, u2x(0.45), 14)            # saplings standing in the shade
    paths = [np.array([[x, gl + 5], [x + r.normal(0, 6), (gl + v2y(0.45)) / 2], [x + r.normal(0, 12), v2y(0.45)]])
             for x in xs]
    load, share = loads(BARK, r.uniform(0.0, 0.15, len(xs)), r)
    desk.brush(paths, r.uniform(2.5, 6, len(xs)), load, share, pickup=0.1, spent=0.5, taper=0.6)
    P = strew(30, shade * noise.smoothstep(v2y(0.58), gl, YY) * noise.smoothstep(0.6, 1.2, noise.field((H, W), 60, r)),
              r, box)                                    # sun let through the crowns onto the floor of the grove
    n = len(P)
    desk.touch(blots(P, r.uniform(10, 26, n), r.normal(0, 0.15, n), r, squash=(0.3, 0.5)),
               ramp(FOREST, r.uniform(0.45, 0.7, n)))
    for trees, colours, leaf in ((TREES[1:], OAK, 24), (TREES[:1], FOREST, 30)):
        limbs, groups, whole = [], [], []
        for u, v, h, sp, z in trees:
            lb, g, w = tree(u2x(u), v2y(v), h * PH, sp * PW, z, r)
            limbs += lb
            groups += g
            whole.append(w)
        groups, whole = np.array(groups, np.float32), np.array(whole, np.float32)
        bark(desk, limbs, r)
        s = billows(groups, r)
        _, _, _, BX, BY = domes(whole, r, 0.04)
        _, _, _, GX, GY = domes(groups, r, 0.08)
        nb = unit(np.stack([BX, BY - 0.2, np.sqrt(np.clip(1 - BX * BX - BY * BY, 0, 1)) + 0.25], -1))
        ng = unit(np.stack([GX, GY - 0.1, np.sqrt(np.clip(1 - GX * GX - GY * GY, 0, 1)) + 0.1], -1))
        inside, tone, hang, _ = leafage(s, r, big=0.55 * nb + 0.45 * ng, local=0.4, gather=80, crease=0.7)
        desk.fill(noise.smoothstep(0.55, 0.9, ndimage.gaussian_filter(inside, 12)), ramp(colours, 0.05), wet=False)
        build(desk, inside, tone, hang, colours, leaf, r, (1.0, 0.72, 0.47, 0.27, 0.13, 0.08), top=1.0,
              dry=0.5, k=(2, 5))
    bush = []                                            # the undergrowth at its foot, with gaps between
    x = PX0 - 60
    while x < u2x(0.47):
        rr = r.uniform(25, 70)
        if r.random() < 0.4:
            for dx in r.normal(0, 0.6 * rr, int(r.integers(2, 4))):     # a low shrub, wider than it is tall
                bush.append((x + dx, edge[int(np.clip(x, 0, W - 1))] - r.uniform(0.0, 0.4) * rr, rr * r.uniform(0.6, 1.0),
                             600 + r.uniform(0, 30)))
        x += rr * r.uniform(1.0, 2.2)
    inside, tone, hang, _ = leafage(np.array(bush, np.float32), r, big=(-0.2, -0.8, 0.5), local=0.5, crease=0.5)
    tone *= 0.55 + 0.45 * noise.smoothstep(u2x(0.25), u2x(0.45), XX)
    build(desk, inside, tone, hang, FOREST, 14, r, (1.0, 0.6, 0.35, 0.18, 0.07), top=0.75, dry=0.5, k=(2, 4))
    return edge


# ---------------------------------------------------------------------------------------------- the ground

HORIZON = float(v2y(0.552))     # the eye's height
VPX = float(u2x(0.80))          # where the rows of rice run to
FOCAL = 2360.0                  # px from the eye to the picture
EYE = 4.0                       # m: the eye is on the raised lane
FIELD = pal("#1f3a26", "#2b4a2b", "#3a5f30", "#4b7334", "#5d8838", "#729d3d", "#89b044", "#a0c150", "#b8d062",
            "#cfdc78")
EARTH = pal("#8c7a62", "#a08d70", "#b29f7f", "#c3b08e", "#d1c09e", "#ddcdad", "#e8dbbf")
MUD = lin("#4f5a40")
LILY = pal("#b24a1e", "#c95d23", "#dc7429", "#e98b33", "#f2a446", "#f6bc5e", "#f8d07c")


def bezier(*C, n=700):
    t = np.linspace(0, 1, n)[:, None]
    k = len(C) - 1
    from math import comb
    return sum(comb(k, i) * (1 - t) ** (k - i) * t ** i * np.asarray(c, np.float64) for i, c in enumerate(C))


def lane():
    """The lane: from the foot of the picture it swings out to the right and back to the foot of the grove,
    narrowing as it goes. -> its centre line (n, 2), the normal to the right of it, its half-width"""
    P = bezier((u2x(0.215), v2y(1.06)), (u2x(0.385), v2y(0.87)), (u2x(0.385), v2y(0.70)), (u2x(0.272), v2y(0.633)))
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return (P.astype(np.float32), np.stack([-T[:, 1], T[:, 0]], 1).astype(np.float32),
            (0.3 * (P[:, 1] - HORIZON)).astype(np.float32))


def lane_edges():
    P, N, hw = lane()
    rows = np.arange(H, dtype=np.float32)
    left, right = P - N * hw[:, None], P + N * hw[:, None]
    return (np.interp(rows, left[::-1, 1], left[::-1, 0]).astype(np.float32),
            np.interp(rows, right[::-1, 1], right[::-1, 0]).astype(np.float32))


def polygon_mask(pts, blur=0.8):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in np.asarray(pts, np.float64)], fill=255)
    return ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, blur)


def blades(x, y, L, w, ang, bend, m=6):
    """Blades of grass, each a stroke of the point flicked from its root at (x, y) along `ang`, curving by
    `bend` and tapering to nothing. -> (N, 2m, 2) polygons"""
    t = np.linspace(0, 1, m)[None]
    a = ang[:, None] + bend[:, None] * t
    step = (L / (m - 1))[:, None]
    cx = x[:, None] + np.cumsum(np.cos(a) * step, 1) - np.cos(a[:, :1]) * step
    cy = y[:, None] + np.cumsum(np.sin(a) * step, 1) - np.sin(a[:, :1]) * step
    half = 0.5 * w[:, None] * (1 - t) ** 0.8 + 0.3
    nx, ny = -np.sin(a), np.cos(a)
    left = np.stack([cx + nx * half, cy + ny * half], -1)
    right = np.stack([cx - nx * half, cy - ny * half], -1)
    return np.concatenate([left, right[:, ::-1]], 1).astype(np.float32)


def mirror(desk, top, foot):
    """What the water gives back: the hills mirrored about their foot, and beyond them the sky and the
    cloud mirrored about the horizon, a little darker, and greener with the mud under the water."""
    src = np.where(YY - foot[None, :] < (foot - top)[None, :] + 20, 2 * foot[None, :] - YY, 2 * HORIZON - YY)
    src = np.clip(src, 0, H - 1).astype(int)
    img = ndimage.gaussian_filter(desk.img, (2.0, 2.0, 0))[src, XX.astype(int)]
    close = noise.smoothstep(HORIZON + 60, H, YY)[..., None]
    return (img * (0.9 - 0.16 * close) + MUD * (0.03 + 0.1 * close)).astype(np.float32)


def paddies(desk, r, top, foot, groveline):
    """The paddies, the rice in them only a few weeks planted, so that the water between the rows holds the
    sky: the water laid first in level strokes of what it gives back, then, paddy by paddy from the far
    ones forward, the rows of rice, each tuft a few flicks of the point with its reflection under it,
    and the levee in front of it, a low bank of grass."""
    _, lane_r = lane_edges()
    far = np.maximum(foot, np.where(XX[0] < u2x(0.47), groveline, 0)).astype(np.float32)
    levees = [0.601, 0.622, 0.654, 0.706, 0.786, 0.930]
    lines = [skyline([(-0.1, v + 0.004), (0.5, v), (1.1, v - 0.006)], r, wob=(1.5 + 8 * (v - 0.55), 300),
                     fine=(0.5 + 2 * (v - 0.55), 40)) for v in levees]
    water = noise.smoothstep(-1, 1, YY - far[None, :] - 3) * noise.smoothstep(-1, 1, lines[-1][None, :] - YY) \
        * noise.smoothstep(-1, 1, XX - lane_r[:, None] - 30)
    mir = mirror(desk, top, foot)
    desk.fill(water, mir, wet=False)
    rows, wd, cols = [], [], []
    y = far.min() - 4
    while y < lines[-1].max():
        w = 1.5 + 0.012 * (y - HORIZON)
        x = u2x(0.25) + r.uniform(-60, 0)
        while x < PX1 + 40:
            L = w * r.uniform(15, 40)
            pts = np.array([[x, y + r.normal(0, 0.15 * w)], [x + L / 2, y + r.normal(0, 0.15 * w)],
                            [x + L, y + r.normal(0, 0.15 * w)]])
            q = at(pts)
            if water[q].max() > 0.5:
                rows.append(pts)
                wd.append(w * r.uniform(0.8, 1.2))
                cols.append(mir[q][[0, 2]] * r.uniform(0.95, 1.04))
            x += L * r.uniform(0.6, 0.9)
        y += w * r.uniform(0.6, 0.9)
    desk.brush(rows, np.array(wd), np.array(cols), pickup=0.4, merge=2.0, spent=0.3, fray=1.2, tails=0.3,
               ends=(0.3, 0.2), hide=0.8)
    for i in range(len(levees)):
        rice(desk, r, lines[i - 1] if i else far, lines[i], lane_r)
        levee(desk, r, lines[i], lane_r)
    return lines


def rice(desk, r, y_far, y_near, lane_r, row=0.3, tall=0.15):
    """Rice in one paddy, planted in rows `row` m apart that run away to the vanishing point, each tuft
    `tall` m high. Near, a tuft is a few blades of yellow-green with its reflection hanging under it; far
    off, the rows are threads of green with the sky between."""
    yn, yf = float(np.median(y_near)), float(np.median(y_far))
    a, b = row / EYE, row / (FOCAL * EYE)
    xl = float(np.interp(yn, np.arange(H), lane_r)) + 20
    g0, g1 = (xl - VPX) / (yn - HORIZON), (PX1 + 60 - VPX) / (yn - HORIZON)
    D0, D1 = 1 / (yn - HORIZON - 2), 1 / max(yf - HORIZON + 2, 1)
    lanes = g0 - (g0 % a) + np.cumsum(a * np.clip(r.normal(1, 0.14, int((g1 - g0) / a) + 3), 0.7, 1.35))
    lanes = lanes[lanes < g1]                            # the rows set out by eye, not ruled
    if tall / EYE * (yn - HORIZON) < 9:                 # far off: the rows only, threads of green
        polys, cols = [], []
        ya, yb = HORIZON + 1 / D0, HORIZON + 1 / D1
        for g in lanes:
            xa, xb = VPX + g / D0, VPX + g / D1
            for _ in range(int(r.integers(1, 4))):
                f0, f1 = np.sort(r.uniform(0, 1, 2))
                p0 = np.array([xa + (xb - xa) * f0, ya + (yb - ya) * f0])
                p1 = np.array([xa + (xb - xa) * f1, ya + (yb - ya) * f1])
                wd = max(0.5, 0.25 * tall / EYE * (p0[1] - HORIZON))
                polys.append(np.array([p0 + [-wd, 0], p1 + [-wd * 0.5, 0], p1 + [wd * 0.5, 0], p0 + [wd, 0]]))
                cols.append(ramp(FIELD, r.uniform(0.35, 0.65)))
        if polys:
            P = np.array(polys, np.float32)
            ok = (P[:, :, 0].mean(1) > np.interp(P[:, :, 1].mean(1), np.arange(H), lane_r) + 12)
            desk.touch(P[ok], np.array(cols)[ok])
        return
    nr = len(lanes)                                      # planted by hand: the hills along a row set by eye,
    d = D0 + np.cumsum(b * r.uniform(0.6, 1.4, (nr, int((D1 - D0) / b) + 4)), 1) - b * r.uniform(0, 1, (nr, 1))
    amp = a * np.where(r.random(nr) < 0.25, r.uniform(0.35, 0.8, nr), np.abs(r.normal(0, 0.15, nr)))
    g = lanes[:, None] + amp[:, None] * np.sin(np.pi * (d - D0) / (D1 - D0) * r.uniform(0.5, 1.6, (nr, 1))
                                               + r.uniform(0, 2 * np.pi, (nr, 1)))   # no row quite straight, a few wander
    g, d = g.ravel() + r.normal(0, 0.1 * a, g.size), d.ravel() + r.normal(0, 0.2 * b, d.size)
    keep = (d < D1) & (r.random(d.size) < 0.88)
    g, d = g[keep], d[keep]
    yy, xx = HORIZON + 1 / d, VPX + g / d
    on = (yy > y_far[np.clip(xx.astype(int), 0, W - 1)] + 2) & (xx > np.interp(yy, np.arange(H), lane_r) + 12) \
        & (xx < PX1 + 60)
    order = np.argsort(yy[on])
    xx, yy = xx[on][order], yy[on][order]
    n = len(xx)
    if not n:
        return
    sz = tall / EYE * (yy - HORIZON) * np.clip(np.exp(r.normal(0, 0.38, n)), 0.45, 2.0)
    lean = r.normal(0, 0.2, n)                           # each clump leans its own way
    refl = blades(xx, yy + 0.5, sz * r.uniform(0.5, 0.8, n), np.maximum(0.1 * sz, 1.0),
                  np.pi / 2 - lean + r.normal(0, 0.1, n), r.normal(0, 0.1, n))
    desk.touch(refl, ramp(FIELD, r.uniform(0.0, 0.25, n)))
    idx = np.repeat(np.arange(n), r.integers(3, 8, n))
    m = len(idx)
    ang = -np.pi / 2 + lean[idx] + r.normal(0, 0.28, m)
    tufts = blades(xx[idx] + r.normal(0, 0.05, m) * sz[idx], yy[idx], sz[idx] * r.uniform(0.6, 1.05, m),
                   np.maximum(sz[idx] * r.uniform(0.045, 0.1, m), 0.9), ang,
                   -0.7 * (ang + np.pi / 2 - lean[idx]) + r.normal(0, 0.15, m))
    lit = np.clip(0.5 + 0.3 * r.random(m) - 0.3 * (np.cos(ang) > 0.12) + 0.15 * (np.cos(ang) < -0.12), 0, 1)
    desk.touch(tufts, ramp(FIELD, lit))


def levee(desk, r, line, lane_r):
    """A levee between two paddies: a low bank of grass, its top in the sun, its face in shade, and its
    reflection dark in the water at its foot."""
    th = 0.045 * (float(np.median(line)) - HORIZON) + 2
    line = line + 0.12 * th * noise.line1d(W, 30, r) + 0.06 * th * noise.line1d(W, 7, r)
    top = line - th * (1 + 0.3 * noise.line1d(W, 40, r) + 0.15 * noise.line1d(W, 9, r))
    x = np.arange(W, dtype=np.float32)
    y0, y1 = max(int(top.min()) - 4, 0), min(int((line + 0.7 * th).max()) + 4, H)
    Y = YY[y0:y1]
    ok = (x > np.interp(line, np.arange(H), lane_r) - 10)[None, :]
    img = desk.img[y0:y1]
    refl = noise.smoothstep(-1, 1, Y - line[None, :] + 0.3 * th) * noise.smoothstep(-1, 1, line[None, :] + 0.7 * th - Y)
    img += (ramp(FIELD, 0.05 + 0.05 * noise.field(Y.shape, 20, r)) - img) * (refl * ok * 0.6)[..., None]
    band = noise.smoothstep(-1, 1, Y - top[None, :]) * noise.smoothstep(-1, 1, line[None, :] + 0.2 * th - Y) * ok
    face = noise.smoothstep(line[None, :] - 0.5 * th, line[None, :] - 0.1 * th, Y)
    img += (ramp(FIELD, 0.35 - 0.25 * face + 0.08 * noise.field(Y.shape, 30, r)) - img) * band[..., None]
    xs = x[ok[0]][::max(1, int(4 - th / 10))]
    xs = xs + r.uniform(-2, 2, len(xs))
    n = len(xs)
    yb = line[np.clip(xs.astype(int), 0, W - 1)] - th * r.uniform(0.35, 0.95, n)
    L = th * r.uniform(0.4, 1.2, n)
    bl = blades(xs, yb, L, np.maximum(0.12 * L, 1.0), -np.pi / 2 + r.normal(0.05, 0.35, n), r.normal(0, 0.3, n))
    desk.touch(bl, ramp(FIELD, np.clip(r.normal(0.55, 0.2, n), 0, 1)))


def lane_paint(desk, r):
    """The lane, beaten earth pale in the sun: laid along its length with a brush not quite full so that it
    breaks on the tooth of the paper, two ruts worn in it and a ridge of grass down the middle, the shade of
    the grove lying across its far end."""
    P, N, hw = lane()
    left, right = P - N * hw[:, None], P + N * hw[:, None]
    mask = polygon_mask(np.vstack([left, right[::-1]]), 1.0)
    desk.fill(mask, ramp(EARTH, 0.72 + 0.06 * noise.fbm((H, W), 60, r, octaves=3)), wet=False)
    paths, wd, tn = [], [], []
    for _ in range(700):
        t0 = r.uniform(-0.05, 0.98)
        t1 = min(t0 + r.uniform(0.03, 0.1), 1.0)
        ti = (np.linspace(t0, t1, 6).clip(0, 1) * (len(P) - 1)).astype(int)
        s = r.uniform(-1.05, 1.05)
        paths.append(P[ti] + N[ti] * (hw[ti] * s)[:, None])
        wd.append(float(hw[ti].mean()) * r.uniform(0.04, 0.1))
        rut = np.exp(-((abs(s) - 0.45) / 0.1) ** 2)
        tn.append(0.78 - 0.45 * rut + r.normal(0, 0.07))
    load, share = loads(EARTH, np.array(tn), r, spread=0.04)
    desk.brush(paths, np.array(wd), load, share, pickup=0.25, merge=2.0, spent=0.7, dry=0.3, fray=2.0)
    ti = (r.uniform(0, 1, 260) ** 0.7 * (len(P) - 1)).astype(int)     # grit and small stones in the ruts
    n = len(ti)
    G = P[ti] + N[ti] * (hw[ti] * r.choice([-1, 1], n) * (0.45 + r.normal(0, 0.12, n)))[:, None]
    gs = 0.035 * hw[ti] * r.uniform(0.5, 1.4, n) + 1.5
    desk.touch(blots(G + np.stack([np.zeros(n), 0.3 * gs], 1), gs * 1.1, r.normal(0, 0.2, n), r, squash=(0.4, 0.6)),
               ramp(EARTH, r.uniform(0.0, 0.25, n)))
    desk.touch(blots(G, gs, r.normal(0, 0.2, n), r, squash=(0.45, 0.7)), ramp(EARTH, r.uniform(0.75, 1.0, n)))
    # the grove's shade across the far end of it, and the dapples of sun let through the crowns
    t = np.arange(len(P)) / (len(P) - 1)
    sh = polygon_mask(np.vstack([left[t > 0.8], right[t > 0.8][::-1]]), 3.0)
    spots = noise.smoothstep(1.2, 1.6, noise.stretched((H, W), 40, 9, r)) * r.uniform(0.3, 0.6)
    desk.fill(sh * (1 - spots) * 0.85, ramp(FOREST, 0.18) * 0.6 + lin("#7c7590") * 0.4, wet=False)
    for s0, spread, dens in ((0.0, 0.07, 0.8), (-1.0, 0.06, 1.8), (1.0, 0.06, 1.8)):
        ti = np.sort(r.uniform(0, 1, int(1500 * dens)) ** 0.8 * (len(P) - 1)).astype(int)
        n = len(ti)
        x, y = (P[ti] + N[ti] * (hw[ti] * (s0 + r.normal(0, spread, n)))[:, None]).T
        L = 0.08 * hw[ti] * r.uniform(0.5, 1.5, n) + 2
        bl = blades(x, y, L, np.maximum(0.15 * L, 1.0), -np.pi / 2 + r.normal(0, 0.45, n), r.normal(0, 0.3, n))
        shade = sh[at(np.stack([x, y], 1))]
        desk.touch(bl, ramp(FIELD, np.clip(r.normal(0.55, 0.2, n) - 0.4 * shade, 0, 1)))


def meadow(desk, r, mask, sun, grow=1.0, calm=0.0):
    """Summer grass and weeds the way Oga paints them: a dark ground; on it tussocks, each a sheaf of blades
    flicked up from one root and leaning a little with the wind, the blades on the side toward the sun
    lighter and the tips lightest, the dark ground showing between them; here and there the rosette of a
    broad-leaved weed, and the small white flowers of fleabane. `sun` (H,W) is how much sun reaches it,
    `grow` how tall it stands (a number or a map), `calm` how little it varies from tussock to tussock."""
    desk.fill(mask, ramp(FIELD, 0.03 + (0.1 + 0.3 * calm) * sun + 0.05 * noise.fbm((H, W), 60, r, octaves=3)), wet=False)
    ys, xs = np.nonzero(mask[::10, ::10] > 0.5)
    pts = np.stack([xs * 10 + r.uniform(0, 10, len(xs)), ys * 10 + r.uniform(0, 10, len(xs))], 1)
    clump = noise.smoothstep(-0.4, 1.0, noise.field((H, W), 90, r))
    tall = (lambda P: grow[at(P)]) if np.ndim(grow) else (lambda P: grow)

    def reach(P):
        return np.clip((P[:, 1] - HORIZON) / (H - HORIZON), 0.05, 1)

    for layer in range(3):                               # the back of the grass first, then the front
        keep = r.random(len(pts)) < 0.14 * ((0.12 + 0.88 * clump[at(pts)]) * (1 - calm) + 1.6 * calm)
        P = pts[keep] + r.normal(0, 4, (int(keep.sum()), 2))
        n = len(P)
        if not n:
            continue
        size = (30 + 200 * reach(P) ** 1.6) * r.uniform(0.6, 1.3, n) * tall(P)
        lean = r.normal(0.15, 0.12, n)
        idx = np.repeat(np.arange(n), r.integers(6, 14, n))
        m = len(idx)
        spread = r.uniform(-1, 1, m)
        ang = -np.pi / 2 + lean[idx] + 0.5 * spread
        L = size[idx] * r.uniform(0.45, 1.0, m) * (1 - 0.35 * np.abs(spread))
        bl = blades(P[idx, 0] + spread * size[idx] * 0.07, P[idx, 1] + 0.15 * size[idx], L,
                    np.maximum(L * np.exp(r.normal(np.log(0.05), 0.35, m)), 1.1), ang, 0.5 * spread + r.normal(0.12, 0.2, m))
        s = sun[at(P)][idx]
        t = 0.06 + (0.3 + 0.18 * calm) * s + (0.24 - 0.12 * calm) * s * (spread < -0.15) - 0.1 * (spread > 0.35) \
            + 0.06 * layer + r.normal(0, 0.07 * (1 - 0.5 * calm), m)
        desk.touch(bl, ramp(FIELD, np.clip(t, 0, 1)))
        if layer == 1:                                   # broad-leaved weeds among the grass
            k = r.random(len(pts)) < 0.012 * (1 - calm)
            Q = pts[k]
            q = len(Q)
            sz = (14 + 70 * reach(Q) ** 1.5) * r.uniform(0.7, 1.3, q)
            idx = np.repeat(np.arange(q), r.integers(4, 8, q))
            a = r.uniform(-np.pi, 0, len(idx))
            lv = leaves(Q[idx, 0], Q[idx, 1], sz[idx] * r.uniform(0.7, 1.1, len(idx)), sz[idx] * r.uniform(0.35, 0.5, len(idx)),
                        a, belly=0.45)
            sq = sun[at(Q)][idx]
            desk.touch(lv, ramp(FIELD, np.clip(0.15 + 0.35 * sq + 0.2 * sq * (np.cos(a + 2.2) > 0.3) + r.normal(0, 0.05, len(idx)), 0, 1)))
    keep = r.random(len(pts)) < 0.05                     # tips caught by the sun
    P = pts[keep]
    n = len(P)
    L = (20 + 130 * reach(P) ** 1.6) * r.uniform(0.4, 0.9, n) * tall(P)
    bl = blades(P[:, 0], P[:, 1] - 0.2 * L, L, np.maximum(L * r.uniform(0.03, 0.06, n), 1.0),
                -np.pi / 2 + r.normal(0.15, 0.3, n), r.normal(0.2, 0.3, n))
    desk.touch(bl, ramp(FIELD, np.clip(0.55 + 0.45 * sun[at(P)] + r.normal(0, 0.06, n), 0, 1)), dry=0.4)
    k = (r.random(len(pts)) < 0.02 * (1 - 0.7 * calm)) & (clump[at(pts)] > 0.5)    # fleabane, in a few drifts
    F = pts[k] + r.normal(0, 6, (int(k.sum()), 2))
    n = len(F)
    d = (3 + 9 * reach(F)) * r.uniform(0.8, 1.2, n)
    desk.touch(blots(F, d, r.uniform(0, np.pi, n), r, lobes=(5, 8), rough=0.25, squash=(0.6, 0.9)),
               np.repeat(lin("#f4f1e4")[None], n, 0) * (0.75 + 0.25 * sun[at(F)])[:, None])
    desk.touch(blots(F, 0.38 * d, np.zeros(n), r), np.repeat(lin("#e8c64a")[None], n, 0))


def tall_grass(desk, r, mass, crest, box):
    """A mass of tall summer grass standing up out of the short: a dark ground, then sheaves of long
    blades from the back of the mass forward, each from one root and all leaning a little the same way
    with the wind. The blades that reach the top of the mass are in the sun and the ones low in it in its
    own shade, so that it reads as one heap lit along its crest (`crest`, the y of its top at each x);
    last, the sunlit tips along the crest, dragged nearly dry."""
    desk.fill(noise.smoothstep(0.5, 0.95, ndimage.gaussian_filter(mass, 14)),
              ramp(FIELD, 0.03 + 0.03 * noise.fbm((H, W), 50, r, octaves=3)), wet=False)
    P = strew(13, noise.smoothstep(0.2, 0.8, mass), r, box)
    P = P[np.argsort(P[:, 1])]
    n = len(P)
    near = np.clip((P[:, 1] - HORIZON) / (H - HORIZON), 0.05, 1)
    below = P[:, 1] - crest[np.clip(P[:, 0].astype(int), 0, W - 1)]
    size = np.minimum(70 + 330 * near ** 1.4, below + r.uniform(5, 50, n)) * r.uniform(0.75, 1.05, n)
    lean = r.normal(0.13, 0.07, n)
    idx = np.repeat(np.arange(n), r.integers(5, 11, n))
    m = len(idx)
    fan = r.uniform(-1, 1, m)
    ang = -np.pi / 2 + lean[idx] + 0.38 * fan
    L = size[idx] * r.uniform(0.55, 1.0, m) * (1 - 0.3 * np.abs(fan))
    w = np.clip(L * np.exp(r.normal(np.log(0.042), 0.35, m)), 1.2, 16)
    bl = blades(P[idx, 0] + fan * size[idx] * 0.05, P[idx, 1], L, w, ang, 0.35 * fan + r.normal(0.12, 0.15, m))
    tip = P[idx, 1] + L * np.sin(ang)
    up = np.clip(1 - (tip - crest[np.clip(P[idx, 0].astype(int), 0, W - 1)]) / 160, 0, 1) ** 1.6
    t = 0.1 + 0.26 * up + 0.12 * up * (fan < -0.2) - 0.05 * (fan > 0.4) + r.normal(0, 0.05, m)
    desk.touch(bl, ramp(FIELD, np.clip(t, 0, 0.92)))
    ok = (up > 0.6) & (r.random(m) < 0.14)               # a few tips along the crest, in the sun
    k = int(ok.sum())
    Lt = L[ok] * r.uniform(0.25, 0.45, k)
    tx = P[idx[ok], 0] + fan[ok] * size[idx[ok]] * 0.05 + np.cos(ang[ok]) * (L[ok] - Lt)
    ty = P[idx[ok], 1] + np.sin(ang[ok]) * (L[ok] - Lt)
    desk.touch(blades(tx, ty, Lt, 0.7 * w[ok], ang[ok], 0.2 * fan[ok]),
               ramp(FIELD, np.clip(0.7 + 0.2 * r.random(k), 0, 1)), dry=0.3)


def lily(desk, r, x, y, s, face, tilt):
    """A day lily (yabukanzō) in flower, turned up toward the light and a little away: six tepals curving
    out from a yellow throat, the three inner ones broader and lighter, each with a darker streak of
    red-orange down its middle. `face` is how far the flower is turned toward the eye (1 full on), `tilt`
    the way it leans."""
    th = np.linspace(0, 2 * np.pi, 6, endpoint=False) + r.uniform(0, 1)
    inner = np.arange(6) % 2 == 1
    L = s * np.where(inner, 1.0, 0.85) * r.uniform(0.85, 1.1, 6)
    dx, dy = np.cos(th), face * np.sin(th)
    ang = np.arctan2(dx * np.sin(tilt) + dy * np.cos(tilt), dx * np.cos(tilt) - dy * np.sin(tilt))
    Lp = L * np.clip(np.hypot(dx, dy), 0.4, 1)
    bow = r.normal(0, 0.1, 6) + 0.08
    order = np.argsort(np.where(inner, 1, 0) + 0.1 * np.sin(th))      # the outer tepals first, the back first
    sun_a = np.arctan2(SUN[1], SUN[0])
    tone = 0.45 + 0.22 * np.cos(ang - sun_a) + 0.15 * inner + r.normal(0, 0.04, 6)
    pet = leaves(np.full(6, x), np.full(6, y), Lp, np.where(inner, 0.46, 0.34) * L, ang, belly=0.5, m=8, bow=bow)
    desk.touch(pet[order], ramp(LILY, tone[order]))
    mid = leaves(np.full(6, x), np.full(6, y), 0.55 * Lp, 0.1 * L, ang, belly=0.3, m=6, bow=bow)
    desk.touch(mid[order], ramp(LILY, tone[order] - 0.5))
    desk.touch(blots(np.array([[x, y]]), np.array([0.26 * s]), np.array([tilt]), r), lin("#e9c25a")[None])


def lilies(desk, r, x0, y0, s, n):
    """A clump of day lilies: strap leaves arching out from one root, and above them bare stems, each with
    a flower open and a bud or two."""
    k = int(r.integers(9, 15))
    a = -np.pi / 2 + r.uniform(-1.1, 1.1, k)
    L = s * r.uniform(2.2, 3.6, k)
    lv = blades(np.full(k, x0) + r.normal(0, 0.3 * s, k), np.full(k, y0), L, np.full(k, 0.22 * s), a,
                np.sign(a + np.pi / 2) * r.uniform(0.6, 1.3, k))
    desk.touch(lv, ramp(FIELD, np.clip(0.3 + 0.25 * (a < -np.pi / 2) + r.normal(0, 0.06, k), 0, 1)))
    for j in range(n):
        lean = r.normal(0, 0.18)
        h = s * r.uniform(4.2, 5.6)
        top = np.array([x0 + h * np.sin(lean), y0 - h * np.cos(lean)])
        stem = blades(np.array([x0 + r.normal(0, 0.2 * s)]), np.array([y0]), np.array([h]), np.array([0.11 * s]),
                      np.array([-np.pi / 2 + lean]), np.array([r.normal(0, 0.12)]))
        desk.touch(stem, ramp(FIELD, np.array([0.45])))
        for _ in range(int(r.integers(1, 3))):          # buds, closed, leaning out from the top of the stem
            b = -np.pi / 2 + lean + r.choice([-1, 1]) * r.uniform(0.35, 0.9)
            bud = leaves(np.array([top[0]]), np.array([top[1]]), np.array([s * r.uniform(0.7, 1.0)]),
                         np.array([0.2 * s]), np.array([b]), belly=0.65)
            desk.touch(bud, ramp(LILY, np.array([r.uniform(0.15, 0.4)])) * 0.65 + ramp(FIELD, np.array([0.55])) * 0.35)
        if j < n - 1 or r.random() < 0.5:
            lily(desk, r, top[0] + r.normal(0, 0.1 * s), top[1], s * r.uniform(0.9, 1.15), r.uniform(0.55, 0.85),
                 r.normal(0, 0.35))


def corner(desk, r, groveline):
    """The grass between the lane and the grove: short and calm under the trees and along the lane, and
    standing out of it toward the foot of the picture one mass of tall summer grass, its top falling
    toward the lane, with a clump of day lilies in flower in it."""
    lane_l, _ = lane_edges()
    mask = noise.smoothstep(-1, 1, lane_l[:, None] + 10 - XX) * noise.smoothstep(-1, 1, YY - groveline[None, :] + 4)
    shade = noise.smoothstep(v2y(0.71), v2y(0.645), YY) * noise.smoothstep(u2x(0.1), u2x(0.26), XX)
    x = np.arange(W, dtype=np.float32)
    crest = 1470 + 0.25 * (x - PX0) + 22 * noise.line1d(W, 260, r) + 6 * noise.line1d(W, 40, r)
    side = lane_l - 140 * noise.smoothstep(2000, 1750, np.arange(H, dtype=np.float32)) + 26 * noise.line1d(H, 150, r) \
        + 8 * noise.line1d(H, 30, r)
    mass = mask * noise.smoothstep(-2, 2, YY - crest[None, :]) * noise.smoothstep(-2, 2, side[:, None] - XX)
    desk.fill(mask, ramp(FIELD, 0.3), wet=False)
    meadow(desk, r, mask * (1 - mass), np.clip(1 - 0.7 * shade, 0, 1), grow=0.55, calm=0.75)
    tall_grass(desk, r, mass, crest, (PX0 - 60, 1400, int(lane_l.max()), PY1 + 40))
    sweep(desk, r, PX0 - 60, PY1 + 30, 1, 45, reach=(200, 470))
    lilies(desk, r, u2x(0.155), v2y(0.965), 54, 2)
    lilies(desk, r, u2x(0.205), v2y(0.845), 32, 2)


def near_bank(desk, r, lines):
    """The bank on the right below the last paddy: grass in the sun, and a few tall blades arching in from
    the corner."""
    _, lane_r = lane_edges()
    mask = noise.smoothstep(-1, 1, XX - lane_r[:, None] + 10) * noise.smoothstep(-1, 1, YY - lines[-1][None, :] + 3)
    meadow(desk, r, mask, 0.85 + 0.15 * noise.smoothstep(-0.5, 0.5, noise.field((H, W), 120, r)))
    sweep(desk, r, PX1 + 40, PY1 + 30, -1, 40)


def sweep(desk, r, x0, y0, side, n, reach=(250, 620)):
    """Tall grass close to the eye, growing from just out of the picture and arching in over the corner:
    long blades, dark against what is beyond them, the sun along the edges of some."""
    x = x0 + side * r.uniform(-40, 180, n)
    y = y0 + r.uniform(-20, 40, n)
    L = r.uniform(*reach, n)
    ang = -np.pi / 2 + side * r.uniform(0.0, 0.55, n)
    bend = side * r.uniform(0.3, 1.2, n)
    bl = blades(x, y, L, r.uniform(9, 18, n), ang, bend, m=10)
    desk.touch(bl, ramp(FIELD, np.clip(r.normal(0.18, 0.08, n), 0, 1)))
    lit = r.random(n) < 0.45
    bl = blades(x[lit] - side * 2, y[lit], L[lit] * 0.9, r.uniform(3, 6, int(lit.sum())), ang[lit], bend[lit], m=10)
    desk.touch(bl, ramp(FIELD, np.clip(r.normal(0.6, 0.1, int(lit.sum())), 0, 1)), dry=0.3)


def bar(p0, p1, w0, w1, r, m=6, wob=0.35):
    """A touch laid along from `p0` to `p1` with a brush pressed `w0` px wide at the start and `w1` at
    the end, its edges never quite true. -> polygon (2m, 2)"""
    t = np.linspace(0, 1, m)[:, None]
    c = np.asarray(p0, np.float32) + (np.asarray(p1, np.float32) - p0) * t
    d = np.asarray(p1, np.float32) - p0
    n = np.array([-d[1], d[0]], np.float32) / (np.hypot(*d) + 1e-6)
    half = 0.5 * (w0 + (w1 - w0) * t)
    left = c + n * (half + r.normal(0, wob, (m, 1)))
    right = c - n * (half + r.normal(0, wob, (m, 1)))
    return np.concatenate([left, right[::-1]]).astype(np.float32)


def torii(desk, r):
    """A small torii where the lane goes in under the trees, in their shade, set down in a few touches:
    two posts leaning in a little, each a shade lighter on the side toward the sun, two beams and a strut
    between them, of vermilion gone dull and dark and worn paler in places, the posts darker at their feet
    where the damp rises; the top beam capped in black, turned up at its ends and not quite level."""
    P, N, hw = lane()
    x, y, w = float(P[-1, 0]), float(P[-1, 1]) + 4, float(hw[-1])
    half, ht = 1.15 * w, 2.6 * w
    red = pal("#2b1613", "#40201a", "#552a20", "#6c3424", "#7f4331", "#8e5a49")
    tilt = r.normal(0, 0.012)
    lift = lambda u: y - ht * u
    polys, tones = [], []
    for sx, wd in ((-1, 0.1), (1, 0.09)):              # the posts: the shaded body, then the lit side
        b0, b1 = (x + sx * half, y), (x + sx * half * 0.9 + r.normal(0, 0.3), lift(0.985 + r.normal(0, 0.006)))
        polys += [bar(b0, b1, wd * w * 1.05, wd * w * 0.9, r),
                  bar(np.add(b0, (-0.02 * w, 0)), np.add(b1, (-0.02 * w, 0)), wd * w * 0.5, wd * w * 0.4, r)]
        tones += [r.uniform(0.3, 0.45), r.uniform(0.6, 0.75)]
    for v, l, rr, wd in ((0.74, 1.24, 1.3, 0.055), (0.955, 1.5, 1.54, 0.08)):    # the tie beam and the top beam
        polys.append(bar((x - l * half, lift(v) - tilt * l * half), (x + rr * half, lift(v) + tilt * rr * half),
                         wd * w, wd * w * 0.92, r))
        tones.append(r.uniform(0.45, 0.6))
    polys.append(bar((x + 0.02 * w, lift(0.745)), (x + 0.03 * w, lift(0.94)), 0.05 * w, 0.045 * w, r))
    tones.append(0.35)
    desk.touch(np.array(polys), ramp(red, np.array(tones)))
    f = np.linspace(-1, 1, 7)                            # the black cap, turned up at its ends
    cx = x + 1.64 * half * f
    cy = lift(1.025) - 0.04 * ht * np.abs(f) ** 2.4 + tilt * (cx - x) + r.normal(0, 0.15, 7)
    th = 0.065 * w * (1 - 0.35 * np.abs(f))
    cap = np.concatenate([np.stack([cx, cy - th / 2], 1), np.stack([cx, cy + th / 2], 1)[::-1]]).astype(np.float32)
    desk.touch(cap[None], lin("#221d1b")[None])
    k = 7                                                # worn: paler where the paint has gone, dark at the feet
    q = np.stack([x + r.choice([-1, 1], k) * half * r.uniform(0.9, 1.0, k), lift(r.uniform(0.2, 0.9, k))], 1)
    desk.touch(blots(q, w * r.uniform(0.04, 0.08, k), np.full(k, -np.pi / 2), r, squash=(0.35, 0.6)),
               ramp(red, r.uniform(0.85, 1.0, k)), dry=0.45)
    feet = [bar((x + sx * half, y + 1), (x + sx * half * 0.99, lift(0.15)), 0.11 * w, 0.1 * w, r) for sx in (-1, 1)]
    desk.touch(np.array(feet), np.stack([lin("#251a16"), lin("#2c201b")]), dry=0.2)


def paint(seed=1988):
    r = noise.rng(seed)
    sheet = support(r)
    keep = reach(sheet, r)
    desk = Desk(sheet, r, seed)
    sky(desk, r)

    def part(k):                                         # each passage its own run of chance from here on
        desk.r = noise.rng(seed * 101 + k)
        return desk.r

    cloud(desk, part(1))
    desk.dry()
    near, foot = hills(desk, part(7))
    groveline = grove(desk, part(12))
    lines = paddies(desk, part(2), near, foot, groveline)
    lane_paint(desk, part(3))
    torii(desk, part(4))
    near_bank(desk, part(5), lines)
    corner(desk, part(6), groveline)
    img = sheet.color * (1 - keep[..., None]) + desk.img * keep[..., None]
    ty, tx = np.gradient(ndimage.gaussian_filter(sheet.tooth, 0.8))
    img = img * (1 + 0.07 * (tx + ty))[..., None]
    return plate.mount(img, sheet)
