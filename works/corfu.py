"""A Wall in the Sun, Corfu. Watercolour, with a few touches of body colour, on rough rag paper.

Sargent spent the autumn of 1909 on Corfu, and there and in the summers after it, on the
Simplon and in Venice, the watercolour he had kept for holidays became the place where he
painted most freely. He worked out of doors on rough paper with big round brushes full of
water, and what he painted was the sun: the white walls of the island and the shade of its
olives thrown across them.

The light on the wall is the paper, kept clear of every wash. The shade goes down in one wash
of cobalt and rose with ultramarine to granulate, laid in a few big strokes of a loaded brush
that leave the sun showing between them; while it is wet, ochre is run into its foot where the
ground throws light back up, and it dries with dark rims where the water stood. A second, deeper
wash goes over the thickest of it when the first is dry. The sea is one deep wash, the cypress
nearly opaque paint, the olive leaves flicks of the point and a dry brush dragged over the tooth
of the paper. Nothing is gone over twice.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import brush, noise, pencil, plate, wash
from atelier import watercolour as wc
from atelier.color import lin, pigment
from atelier.paper import Sheet

TITLE = "A Wall in the Sun, Corfu"
DATE = "2026"
MEDIUM = "Watercolour, with a few touches of body colour, on rough rag paper"
AFTER = ("John Singer Sargent, the late watercolours, 1900–1911: Corfu: Lights and Shadows; Olive Trees, Corfu; "
         "Corfu: The Terrace")
ROOM = "Paper and Water"
YEAR = 1909
PLACE = "Corfu"
REGION = "Americas"
NOTE = ("A whitewashed garden wall above the sea at noon, with the shade of an olive thrown across it in one wash "
        "of cobalt and rose. The wall in the sun is the bare paper.")

H, W = 2400, 3120
X0, Y0, X1, Y1 = 150, 140, 2970, 2262        # the part of the sheet the painter worked, give or take
SW, SH = X1 - X0, Y1 - Y0
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
U, V = (XX - X0) / SW, (YY - Y0) / SH

PALETTE = dict(   # the box, as absorbance at unit density
    cerulean=pigment("#7fb6dc"), cobalt=pigment("#5b8ee0"), ultramarine=pigment("#4c58d4"),
    rose=pigment("#e0708c"), light_red=pigment("#d8704a"), burnt_sienna=pigment("#a65a36"),
    yellow_ochre=pigment("#e0b264"), raw_sienna=pigment("#cf9342"), gamboge=pigment("#ecc84a"),
    viridian=pigment("#3f9a88"), indigo=pigment("#3d4a66"), sepia=pigment("#6f5a4a"), graphite=pigment("#6e6e74"))
SHADE = dict(ultramarine=0.22, cobalt=0.24, rose=0.10)       # the shadow, wherever the sun is kept off

HORIZON = 0.215             # v of the sea's edge, which is the eye's height
VP = -3.0                   # u of the point the wall runs away to, on the horizon
END, END_W = 0.72, 0.06     # u of the wall's near end, and the width of the end face
TOP, FOOT = 0.34, 0.90      # v of the wall's top and foot at its near end


def _s(r):
    return int(r.integers(1 << 31))


def toward(v_end, u):
    """v at `u` on the line that runs from (END, v_end) away to the vanishing point."""
    return HORIZON + (v_end - HORIZON) * (u - VP) / (END - VP)


def marks(rows, belly=0.35, start=0.7):
    """The union of strokes of a round brush, as a 0..1 mask: rows (x, y, length, width, angle,
    bend). Each lands at (x, y), travels `length` px along `angle`, swells to `width` a `belly`
    of the way along and lifts to a point; `bend` bows it by that fraction of its length.
    `start` < 1 makes the landing rounder, as a loaded brush pressed down is."""
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    for x, y, L, wd, a, b in np.asarray(rows, np.float64).reshape(-1, 6):
        t = np.linspace(0, 1, int(np.clip(L / 6, 9, 60)))
        half = 0.5 * wd * np.where(t < belly, np.sin(0.5 * np.pi * np.minimum(t / belly, 1)) ** start,
                                   np.cos(0.5 * np.pi * np.clip((t - belly) / (1 - belly), 0, 1)) ** 0.9)
        c, s = np.cos(a), np.sin(a)
        along, mid = t * L, 4 * b * L * t * (1 - t)
        px = np.concatenate([along, along[::-1]])
        py = np.concatenate([mid + half, (mid - half)[::-1]])
        draw.polygon(list(zip(x + c * px - s * py, y + s * px + c * py)), fill=255)
    return ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, 0.6)


def broad(mask, k=11):
    """A shape as a brush lays it: no sliver or chink narrower than about `k` px, its corners round."""
    return noise.smoothstep(0.42, 0.58, ndimage.gaussian_filter(mask, k / 2.5))


class Box:
    """The paint box. Transparent washes multiply, so the order they went on does not change the
    colour: each is kept as a density per pigment and glazed onto the paper at the end."""

    def __init__(self, sheet, r, keep):
        self.sheet, self.r, self.keep = sheet, r, keep
        self.d = {k: np.zeros((H, W), np.float32) for k in PALETTE}
        self.soak = np.zeros((H, W), np.float32)
        self.strokes = {}
        self.opaque = []

    def lay(self, density, **mix):
        d = density * self.keep
        for k, f in mix.items():
            self.d[k] += f * d

    def wash(self, mask, ragged=1.0, **kw):
        """A wash laid on dry paper over `mask`: (where its water lay, the pigment it left)."""
        mask = mask * self.keep
        ys, xs = np.flatnonzero(mask.max(1) > 0.02), np.flatnonzero(mask.max(0) > 0.02)
        wet, dep = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        if len(ys):
            sl = (slice(max(0, ys[0] - 24), ys[-1] + 25), slice(max(0, xs[0] - 24), xs[-1] + 25))
            sheet = Sheet(*(a[sl] for a in (self.sheet.color, self.sheet.tooth, self.sheet.fiber, self.sheet.alpha)))
            wet[sl] = wc.puddle(mask[sl], sheet, self.r, ragged)
            if wet[sl].max() > 0.02:
                dep[sl] = wc.deposit(wet[sl], sheet, self.r, **kw)
            self.soak[sl] = np.maximum(self.soak[sl], wet[sl])
        return wet, dep

    def stroke(self, key, rows, radius, **kw):
        ink, water = self.strokes.setdefault(key, (np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)))
        kw = dict(dict(load=1.0, clumps=5, splay=0.1, head=0.5, bristles=int(np.clip(12 * radius, 24, 160))), **kw)
        brush.stroke(self.sheet, rows, radius, seed=_s(self.r), into=(ink, water), **kw)

    def take(self, key, dry=0.0):
        """The pigment the strokes under `key` left; a dry brush (`dry` > 0) leaves it only on the
        hills of the paper, more of them the harder it was pressed."""
        ink = self.strokes.pop(key)[0]
        return pencil.catch(ink, self.sheet, grip=dry) if dry else ink

    def touch(self, rows, density, ragged=0.5, **mix):
        """Touches of the point, laid on dry paper and left to dry each with its rim."""
        if len(rows):
            _, dep = self.wash(marks(rows), ragged, pool=0.15, rim=0.9, rim_width=1.5, scale=40)
            self.lay(dep * density, **mix)

    def body(self, alpha, colour):
        """Body colour: paint with white in it, which covers what is under it instead of glazing it."""
        self.opaque.append((alpha * self.keep, lin(colour)))

    def glaze(self):
        img = self.sheet.color.copy()
        for k, kk in PALETTE.items():
            img *= np.exp(-self.d[k][..., None] * kk)
        for a, c in self.opaque:
            img = img * (1 - a[..., None]) + c * a[..., None]
        return img


def frame(r):
    """Where the painter stopped: near the edge of the sheet, by eye, a little further here and there."""
    side = lambda n: 7 * noise.line1d(n, 380, r) + 2.5 * noise.line1d(n, 50, r)
    d = np.minimum.reduce([XX - X0 - side(H)[:, None], X1 + side(H)[:, None] - XX,
                           YY - Y0 - side(W)[None, :], Y1 + side(W)[None, :] - YY])
    return noise.smoothstep(-1.5, 1.5, d).astype(np.float32)


class Site:
    """The drawing: the wall and its end, the foot of it, the brink of the terrace, the sea's edge.
    Lines are y per column (or x per row for the wall's end), each a little off the true."""

    def __init__(self, r):
        cols = np.arange(W, dtype=np.float32)
        u = (cols - X0) / SW
        line = lambda n, *scales: sum(a * noise.line1d(n, s, r) for a, s in scales)
        self.near = (u - VP) / (END - VP)                         # 1 at the wall's end, smaller further off
        self.cap = 0.026 * SH * self.near                         # height of the coping
        self.top = Y0 + SH * toward(TOP, u) + line(W, (5, 700), (2.5, 140), (1.2, 40), (0.4, 9))
        self.sky_edge = self.top - 0.35 * self.cap                # we look down on the coping: its top is a sliver
        self.under = self.top + self.cap
        self.foot = Y0 + SH * toward(FOOT, u) + line(W, (3, 260), (2.5, 30))
        self.horizon = Y0 + SH * HORIZON + line(W, (0.8, 400))
        self.brink = Y0 + SH * (0.87 - 0.02 * (u - 0.77)) + line(W, (4, 160), (2, 22))
        self.corner = X0 + END * SW + line(H, (2.5, 400), (0.7, 25))
        self.endx = self.corner + END_W * SW + line(H, (2, 300), (0.7, 25))
        self.xc = xc = int(np.median(self.corner))
        # the end face runs away to the right, so its top and foot rise a little towards the sea
        self.e_top = self.sky_edge[xc] - 0.022 * (cols - xc) + line(W, (0.8, 40))
        self.e_foot = self.foot[xc] - 0.097 * (cols - xc) + line(W, (1.5, 30))
        step = lambda d: noise.smoothstep(-0.7, 0.7, d)
        left = step(self.corner[:, None] - XX)
        self.face = left * step(YY - self.sky_edge[None, :]) * step(self.foot[None, :] - YY)
        inside = step(XX - self.corner[:, None]) * step(self.endx[:, None] - XX)
        self.end = inside * step(YY - self.e_top[None, :]) * step(self.e_foot[None, :] - YY)
        lip = step(XX - self.endx[:, None] + 1) * step(self.endx[:, None] + 0.3 * self.cap[xc] - XX)
        self.end = np.maximum(self.end, lip * step(YY - self.e_top[None, :]) * step(self.e_top[None, :] + self.cap[xc] - YY))
        self.wall = np.clip(self.face + self.end, 0, 1)
        beyond = step(XX - self.endx[:, None])
        self.ground = np.clip(left * step(YY - self.foot[None, :]) + inside * step(YY - self.e_foot[None, :])
                              + beyond * step(YY - self.brink[None, :]), 0, 1)
        self.sea = step(YY - self.horizon[None, :]) * (1 - np.clip(self.wall + self.ground, 0, 1))


def sky(box, r, site):
    """The sky at noon, hot and pale: cerulean and a little cobalt laid side to side in broad strokes
    of the biggest brush, thinning towards the sea, with a breath of ochre and rose low down."""
    hy = site.horizon[None, :]
    _, dep = box.wash(noise.smoothstep(-1, 1, hy + 3 - YY), 2.0, pool=0.2, rim=0.25, rim_width=6, tides=0.1,
                      grain=0.15, scale=300)
    y = Y0 - 20
    while y < site.horizon.mean() + 30:
        R = r.uniform(80, 110)
        box.stroke("sky", [(X0 - 150, y + r.normal(0, 8), 0.9), (W / 2, y + r.normal(0, 14), 1.0),
                           (X1 + 150, y + r.normal(0, 8), 0.8)], R, load=1.0, reach=9000, dryness=0.5, clumps=7)
        y += R * r.uniform(1.0, 1.3)
    ink = ndimage.gaussian_filter(np.clip(box.take("sky"), 0, 1.3), 2)
    v = np.clip((YY - Y0) / (hy - Y0), 0, 1)
    box.lay(dep * (0.45 + 0.55 * ink) * (0.04 + 0.24 * (1 - v) ** 1.3), cerulean=0.65, cobalt=0.35)
    box.lay(dep * noise.smoothstep(0.55, 1.0, v) * 0.06, yellow_ochre=0.6, rose=0.35)


def mountains(box, r, site):
    """The mountains of Epirus across the strait: two flat washes of violet gone pale in the heat,
    each left to dry with its edge."""
    for lift, dens in ((0.044, 0.16), (0.024, 0.26)):
        crest = sum(a * np.abs(noise.line1d(W, s, r)) for a, s in ((1.0, 420), (0.45, 130), (0.12, 30)))
        ridge = site.horizon - SH * lift * (0.45 + 0.55 * crest / crest.max())
        m = noise.smoothstep(-0.8, 0.8, YY - ridge[None, :]) * noise.smoothstep(-0.8, 0.8, site.horizon[None, :] + 2 - YY)
        _, dep = box.wash(m, 0.8, pool=0.15, rim=0.6, rim_width=2.5, grain=0.35, scale=200)
        box.lay(dep * dens, cobalt=0.45, rose=0.35, ultramarine=0.2)


def sea(box, r, site, reserve):
    """The sea in one wash of cobalt and ultramarine with a little viridian, paler and greyer at the
    horizon and deepest under the wall, laid in long overlapping strokes from side to side; where
    the brush rode over the hollows of the paper, a glitter of sun shows in them."""
    m = site.sea * (1 - reserve)
    _, dep = box.wash(m, 1.2, pool=0.12, rim=0.9, rim_width=3, tides=0.1, grain=0.45, scale=180)
    y = site.horizon.mean() - 20
    while y < site.brink.max() + 30:
        R = r.uniform(40, 60)
        a, b = (X0 - 150, X1 + 150) if r.uniform() < 0.5 else (X1 + 150, X0 - 150)
        box.stroke("sea", [(a, y + r.normal(0, 3), 1.0), ((a + b) / 2, y + r.normal(0, 5), 0.9),
                           (b, y + r.normal(0, 3), 0.85)], R, load=1.0, reach=9000, dryness=0.5, clumps=9)
        y += R * r.uniform(1.0, 1.25)
    ink = ndimage.gaussian_filter(np.clip(box.take("sea"), 0, 1.3), 2)
    v = np.clip((YY - site.horizon[None, :]) / (0.4 * SH), 0, 1)
    glitter = noise.smoothstep(1.2, 2.0, noise.stretched((H, W), 300, 14, r)) * noise.smoothstep(0.36, 0.22, box.sheet.tooth)
    box.lay(dep * (0.55 + 0.45 * ink) * (0.5 + 0.6 * v ** 0.7) * (1 - 0.9 * glitter), cobalt=0.52, ultramarine=0.34,
            viridian=0.16, indigo=0.04)
    box.lay(dep * (1 - v) ** 3 * 0.14, rose=0.4, cobalt=0.3)


def centred(x, y, L, wd, a, b):
    """Rows for `marks` of strokes centred on (x, y) rather than landing there."""
    x, y, L, wd, a, b = np.broadcast_arrays(*map(np.asarray, (x, y, L, wd, a, b)))
    return np.stack([x - 0.45 * L * np.cos(a), y - 0.45 * L * np.sin(a), L, wd, a, b], 1).reshape(-1, 6)


def spray(r, x, y, theta, n, leaf, lean=None, k=0.55, wide=(0.17, 0.26)):
    """Leaves along a twig from (x, y) towards `theta`, alternate, smaller towards its tip, and the
    twig itself last. A shadow on a wall leans every leaf towards `lean` by `k`, the way the
    slanting sun draws them out."""
    s, rows = leaf * 0.15, []
    for i in range(n):
        s += leaf * r.uniform(0.45, 0.8)
        th = theta + 0.25 * np.sin(s / (3 * leaf))                  # the twig curves a little
        px, py = x + np.cos(th) * s, y + np.sin(th) * s
        a = th + (1 if i % 2 else -1) * r.uniform(0.35, 0.8)
        if lean is not None:
            a = lean + k * np.angle(np.exp(1j * (a - lean)))
        L = leaf * r.uniform(0.7, 1.2) * (1 - 0.35 * i / n)
        rows.append((px, py, L, L * r.uniform(*wide), a, r.normal(0, 0.07)))
    rows.append((x, y, s, max(3.0, 0.07 * leaf), theta, r.normal(0, 0.04)))
    return rows


def dapple(r, thick, area, A, size=1.0, squash=1.0, n=900, fringe=1.0):
    """The shade of an olive crown as a painter lays it, for one wash: where the crown is thick, big
    strokes of a loaded brush side by side along the slant of the sun `A`, with rounds of light
    left between; where it thins, blades of shade a few at a time, now and then a spray with its
    twig, and out in the sun a stray leaf. -> (rows of strokes, rows of the lights kept clear,
    rows for the deeper second wash)"""
    x0, y0, x1, y1 = area
    x, y = r.uniform(x0, x1, n), r.uniform(y0, y1, n)
    t = thick[np.clip(y, 0, H - 1).astype(int), np.clip(x, 0, W - 1).astype(int)]
    big = r.uniform(0, 1, n) < noise.smoothstep(0.45, 0.95, t)
    L = r.uniform(150, 360, n) * size
    a = A + r.normal(0, 0.14, n)
    wd = L * r.uniform(0.2, 0.33, n) * squash
    bend = r.normal(0, 0.06, n)
    strokes = [centred(x[big], y[big], L[big], wd[big], a[big], bend[big])]
    sel = big & (t > 0.95)
    deep = centred(x[sel], y[sel], 0.7 * L[sel], 0.5 * wd[sel], a[sel], bend[sel])
    across = np.array([-np.sin(A), np.cos(A)])
    for xi, yi, ti in zip(x, y, t):
        u = r.uniform()
        if 0.15 < ti < 0.7 and u < 0.2 * fringe:
            k = int(r.integers(2, 6))
            Lb = r.uniform(60, 170, k) * size
            o = r.normal(0, 1, k) * 40 * size
            g = r.normal(0, 1, k) * 30 * size
            strokes.append(centred(xi + o * across[0] + g * np.cos(A), yi + o * across[1] + g * np.sin(A), Lb,
                                   Lb * r.uniform(0.16, 0.28, k) * squash, A + r.normal(0, 0.22, k), r.normal(0, 0.08, k)))
        elif 0.15 < ti < 0.7 and u < 0.25 * fringe:
            rows = np.array(spray(r, xi, yi, A + r.normal(0, 0.7), int(r.integers(3, 7)), r.uniform(55, 100) * size,
                                  A, 0.55, wide=(0.24, 0.36)))
            rows[:, 3] *= squash
            strokes.append(rows)
        elif -0.25 < ti <= 0.15 and u < 0.05 * fringe:
            k = int(r.integers(1, 3))
            Lb = r.uniform(40, 90, k) * size
            strokes.append(centred(xi + r.normal(0, 20, k), yi + r.normal(0, 20, k), Lb, Lb * r.uniform(0.2, 0.32, k) * squash,
                                   A + r.normal(0, 0.3, k), r.normal(0, 0.08, k)))
    # the rounds of sun let through the crown: the painter went round them, two or three touches each
    m = (t > 0.45) & (r.uniform(0, 1, n) < 0.06)
    k = r.integers(1, 4, n)[m]
    cx, cy, hl = np.repeat(x[m], k), np.repeat(y[m], k), np.repeat(r.uniform(40, 110, n)[m] * size, k)
    j = len(cx)
    lights = centred(cx + r.normal(0, 0.3, j) * hl, cy + r.normal(0, 0.3, j) * hl, hl * r.uniform(0.5, 1, j),
                     hl * r.uniform(0.3, 0.55, j) * squash, A + r.normal(0, 0.5, j), r.normal(0, 0.1, j))
    return np.vstack(strokes), lights, deep


def whitewash(box, r, site):
    """The wall before the shade went on: lime over rubble, never quite white and never quite flat. A
    breath of ochre where the ground's light warms it, a grey-ochre stain where the rain splashed its
    foot."""
    f = site.face
    warm = noise.fbm((H, W), 300, r, octaves=3) + 1.4 * noise.smoothstep(0.62, 0.9, V) - 0.4
    _, dep = box.wash(broad(noise.smoothstep(0.2, 0.6, warm), 25) * f, 2.0, pool=0.2, rim=0.5, rim_width=5, grain=0.3,
                      scale=200)
    box.lay(dep * 0.07, yellow_ochre=0.6, rose=0.25)
    h = 0.035 * SH * site.near * (1 + 0.4 * noise.line1d(W, 90, r))
    _, dep = box.wash(f * noise.smoothstep(-1, 1, YY - (site.foot - h)[None, :]), 1.5, pool=0.3, rim=0.8, rim_width=3,
                      tides=0.3, grain=0.5, scale=80)
    box.lay(dep * 0.12, raw_sienna=0.4, cobalt=0.3, rose=0.15)


def wall_shade(box, r, site):
    """The olive's shade on the wall, in one wash of cobalt, ultramarine and rose: cooler high up,
    where the sky is reflected into it, rosier where rose was dropped in, and warmer at the foot,
    where ochre was run into it wet for the light coming up off the ground. When it was dry, a
    deeper wash went over the thickest of it, and the coping's shadow was drawn along the top in one
    stroke."""
    thick = (1.3 - U / 0.45 - 0.9 * np.clip(V - 0.66, 0, None) / 0.3 + 0.3 * noise.fbm((H, W), 260, r, octaves=3)
             + 0.5 * np.exp(-((U - 0.58) / 0.05) ** 2 - ((V - 0.48) / 0.08) ** 2))
    area = (X0 - 150, site.sky_edge.min() - 40, site.corner.max(), site.foot.max() + 40)
    rows, lights, deep = dapple(r, thick, area, A=1.12)
    first = broad(np.clip(marks(rows) - marks(lights, belly=0.5, start=0.5), 0, 1), 7) * site.face
    wet, dep = box.wash(first, 1.5, pool=0.3, rim=0.9, rim_width=3.0, tides=0.2, grain=0.55, scale=160)
    warm = noise.smoothstep(0.5, 0.85, V)
    cool = 1 - 0.45 * warm
    box.lay(dep * 0.8, ultramarine=SHADE["ultramarine"] * cool, cobalt=SHADE["cobalt"] * cool, rose=SHADE["rose"])
    box.lay(wc.charge(warm * first, wet, 30, r, streak=0.3) * 0.3, yellow_ochre=0.5, rose=0.2, raw_sienna=0.15)
    box.lay(wc.charge(noise.smoothstep(0.5, 0.35, V) * first, wet, 40, r, streak=0.3) * 0.15, cerulean=0.6, cobalt=0.4)
    drop = noise.smoothstep(0.6, 1.4, noise.field((H, W), 120, r)) * first
    box.lay(wc.charge(drop, wet, 20, r, streak=0.2) * 0.25, rose=0.7, cobalt=0.2)
    _, dep = box.wash(broad(marks(deep), 7) * first, 1.0, pool=0.2, rim=1.0, rim_width=2.5, tides=0.1, grain=0.6, scale=90)
    box.lay(dep * 0.5, ultramarine=0.32, rose=0.16, cobalt=0.08)
    xs = np.arange(X0 - 120, site.xc + 1, 60.0)
    xs[-1] = site.xc
    xi = xs.astype(int).clip(0, W - 1)
    p = site.near[xi] ** 1.2 * np.clip(0.75 + 0.4 * noise.line1d(len(xs) + 8, 4, r)[:len(xs)], 0.15, 1)
    box.stroke("coping", np.stack([xs, site.under[xi] + 0.1 * site.cap[xi] + r.normal(0, 1.5, len(xs)), p], 1),
               0.3 * site.cap.max(), load=1.0, dryness=1.3, reach=2000, clumps=6, head=0.3)
    box.lay(box.take("coping") * site.face * 0.5, **SHADE)
    return first


def end_face(box, r, site):
    """The end of the wall, turned from the sun: rose and ochre for the light the ground throws
    into it, cobalt into that while it was wet; the coping's end a shade darker and a thin shadow
    under it."""
    et, ef = site.e_top[None, :], site.e_foot[None, :]
    vv = np.clip((YY - et) / (ef - et), 0, 1)
    wet, dep = box.wash(site.end, 0.8, pool=0.25, rim=0.7, grain=0.5, scale=120)
    box.lay(dep * 0.22, rose=0.4, yellow_ochre=0.6)
    box.lay(dep * (0.5 - 0.4 * vv), ultramarine=0.28, cobalt=0.22, rose=0.08)
    core = noise.smoothstep(0.35 * END_W * SW, 0, XX - site.corner[:, None]) * site.end
    box.lay(wc.charge(core, wet, 8, r) * 0.25, ultramarine=0.4, rose=0.15)
    box.lay(wc.charge(noise.smoothstep(0.4, 1.0, vv) * site.end, wet, 25, r, streak=0.4) * 0.45, yellow_ochre=0.5,
            light_red=0.25, rose=0.15)
    c = site.cap[site.xc]
    box.lay(dep * noise.smoothstep(1, -1, YY - et - c) * 0.2, ultramarine=0.4, rose=0.2)
    box.lay(dep * noise.smoothstep(-1, 1, YY - et - c) * noise.smoothstep(1, -1, YY - et - 1.5 * c) * 0.35,
            ultramarine=0.4, rose=0.15, sepia=0.1)


def cypress_rows(r, cx=0.865, base=0.80, tip=0.06, wmax=0.026, n=1300):
    """A cypress as the brush builds it: a flame of upward strokes, fullest a third of the way up,
    its tips standing out along its edges. -> (rows, which side of the tree each is on (-1 sun,
    +1 shade), the tree's mask)"""
    xb, yb, yt, wm = X0 + cx * SW, Y0 + base * SH, Y0 + tip * SH, wmax * SW
    t = r.uniform(0, 1, n) ** 1.3
    half = wm * np.sin(np.pi * (0.08 + 0.92 * t) ** 0.62) ** 0.8
    sway = 5 * noise.line1d(512, 60, r)
    cen = xb + sway[(t * 511).astype(int)] - 6 * t
    side = r.uniform(-1, 1, n)
    L = (34 + 80 * (1 - t)) * r.uniform(0.7, 1.3, n)
    rows = np.stack([cen + side * half * 0.8, yb - t * (yb - yt), L, L * r.uniform(0.32, 0.5, n),
                     -np.pi / 2 + 0.45 * side + r.normal(0, 0.12, n), r.normal(0, 0.08, n)], 1)
    return rows, side, broad(marks(rows), 5)


def cypress(box, r, rows, side, mask):
    """The cypress below the terrace, standing up out of the olives against the sea and the sky: one
    wash of green with indigo run into its shaded side while wet, then the darks flicked up in flames
    with a brush not quite full, and in the thick of it paint so heavy it is nearly opaque; its sunny
    side left flickering, with a few touches of sun on it."""
    wet, dep = box.wash(mask, 0.8, pool=0.25, rim=0.7, rim_width=2, grain=0.45, scale=60)
    box.lay(dep * 0.8, viridian=0.32, raw_sienna=0.32, gamboge=0.08, indigo=0.05)
    box.lay(wc.charge(broad(marks(rows[side > 0.1]), 5) * mask, wet, 14, r, streak=0.5) * 0.7,
            indigo=0.35, viridian=0.25, burnt_sienna=0.1)
    for x, y, L, wd, a, b in rows[(side > -0.5) & (r.uniform(0, 1, len(side)) < 0.35)]:
        L = 1.3 * L
        box.stroke("flame", [(x, y, 0.9), (x + np.cos(a) * L / 2 + b * L, y + np.sin(a) * L / 2, 0.7),
                             (x + np.cos(a) * L, y + np.sin(a) * L, 0.05)], 0.45 * wd, load=0.8, dryness=1.4,
                   reach=1.5 * L, clumps=4, bristles=40)
    box.lay(box.take("flame") * mask * 0.9, indigo=0.45, viridian=0.3, sepia=0.15)
    core = (np.abs(side - 0.25) < 0.3) & (r.uniform(0, 1, len(side)) < 0.12)
    box.touch(rows[core] * [1, 1, 0.8, 0.5, 1, 1], 1.4, indigo=0.45, sepia=0.3, viridian=0.25)
    sun = (side < -0.4) & (r.uniform(0, 1, len(side)) < 0.5)
    box.touch(rows[sun] * [1, 1, 0.5, 0.6, 1, 1], 0.4, gamboge=0.3, raw_sienna=0.25, viridian=0.1)


def bough(r, x, y, a, length, sag=0.12, n=6, kink=0.12):
    """Control rows (x, y) of a bough that sets off along `a` and bends down under its leaves,
    twisting as an olive's do."""
    pts = [(x, y)]
    for i in range(n):
        a += sag + r.normal(0, kink)
        x, y = x + np.cos(a) * length / n, y + np.sin(a) * length / n
        pts.append((x, y))
    return np.array(pts)


def lobed(r, cx, cy, rx, ry):
    """The outline of a clump of leaves: an ellipse pushed out into a few lobes."""
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    k = r.integers(3, 6)
    rad = 1 + 0.16 * np.sin(k * th + r.uniform(0, 6)) + 0.08 * np.sin((k + 2) * th + r.uniform(0, 6))
    return np.stack([cx + rx * rad * np.cos(th), cy + ry * rad * np.sin(th)], 1), th, rad


def clumps(r, rows, leaf, hang=0.5):
    """Olive foliage as a painter sees it, clumps (cx, cy, rx, ry) of leaves: their outlines run
    together, leaves hang out from their lower edges, and each has a shaded part below and away from
    the sun. -> (mask of the clumps, mask of their shade, rows of the leaves at their edges)"""
    im, sh = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    dm, ds = ImageDraw.Draw(im), ImageDraw.Draw(sh)
    blades = []
    for cx, cy, rx, ry in rows:
        P, th, rad = lobed(r, cx, cy, rx, ry)
        dm.polygon([tuple(p) for p in P], fill=255)
        Q, _, _ = lobed(r, cx + 0.22 * rx, cy + 0.3 * ry, 0.72 * rx, 0.66 * ry)
        ds.polygon([tuple(p) for p in Q], fill=255)
        for j in r.choice(len(th), int(2.4 * (rx + ry) / leaf)):
            t = th[j]
            if np.sin(t) < -0.3 and r.uniform() < 0.75:          # few stand up from the top of a clump
                continue
            out = np.arctan2(rx * np.sin(t), ry * np.cos(t))
            a = np.angle((1 - hang) * np.exp(1j * out) + hang * 1j) + r.normal(0, 0.3)
            L = leaf * r.uniform(0.7, 1.3)
            blades.append((cx + 0.82 * rx * rad[j] * np.cos(t), cy + 0.82 * ry * rad[j] * np.sin(t), L,
                           L * r.uniform(0.18, 0.27), a, r.normal(0, 0.08)))
    g = lambda i: ndimage.gaussian_filter(np.asarray(i, np.float32) / 255, 0.8)
    return g(im), g(sh), np.array(blades)


def foliage(box, r, mass, shade, blades, leaf, silver=1.0, dark=1.0):
    """Paint clumps of olive: one wet wash of grey-green over the lot, with blue and indigo run into
    the shaded parts and a little ochre into the sunny tops; when that was dry, darker leaves touched
    into the shade and hung from the lower edges, and a few strays in the light."""
    m = broad(np.clip(mass + marks(blades), 0, 1), 5)
    wet, dep = box.wash(m, 1.5, pool=0.35, rim=0.9, rim_width=3, tides=0.2, grain=0.5, scale=110)
    box.lay(dep * 0.55, cobalt=0.22, viridian=0.2, raw_sienna=0.2, ultramarine=0.04)
    sh = broad(shade, 9) * m
    box.lay(wc.charge(sh, wet, 20, r, streak=0.3) * 0.6 * dark, indigo=0.25, ultramarine=0.25, viridian=0.2)
    box.lay(wc.charge(noise.smoothstep(0.5, 0.0, shade) * m, wet, 25, r, streak=0.3) * 0.2 * silver,
            raw_sienna=0.35, gamboge=0.2, cobalt=0.1)
    ys, xs = np.nonzero(sh[::7, ::7] > 0.6)
    k = r.uniform(0, 1, len(xs)) < 49 * 0.25 / (leaf * leaf / 12)
    xs, ys = xs[k] * 7 + r.uniform(0, 7, k.sum()), ys[k] * 7 + r.uniform(0, 7, k.sum())
    L = leaf * r.uniform(0.6, 1.1, len(xs))
    inner = centred(xs, ys, L, L * r.uniform(0.2, 0.3, len(xs)), np.pi / 2 + r.normal(0, 0.7, len(xs)),
                    r.normal(0, 0.08, len(xs)))
    pick = r.uniform(0, 1, len(blades))
    box.touch(np.vstack([inner, blades[pick < 0.4]]), 0.7 * dark, indigo=0.3, ultramarine=0.22, viridian=0.25,
              raw_sienna=0.08)
    box.touch(blades[(pick >= 0.4) & (pick < 0.65)], 0.55, viridian=0.3, cobalt=0.2, raw_sienna=0.15)
    box.touch(blades[pick > 0.85], 0.45, cobalt=0.25, viridian=0.15, raw_sienna=0.15)
    return m


def olive(box, r, site):
    """The olive overhead whose shade is on the wall: its crown is in the top corner, out of the
    picture but for its lowest clumps, and its boughs reach out from it and hang more clumps over
    the sea and the top of the wall. The boughs are drawn in dark where they show between the
    leaves, a dry brush is dragged across the clumps so the tooth of the paper sparkles through like
    light through olive leaves, and a few leaves catch the sun in body colour over the dark."""
    boughs = [bough(r, X0 - 90, Y0 + SH * y0, a, SW * L, sag=0.06, n=7, kink=0.22)
              for y0, a, L in ((0.03, 0.02, 0.56), (0.15, 0.3, 0.38), (-0.05, 0.55, 0.34), (0.08, -0.15, 0.3))]
    rows = []
    for _ in range(12):
        rx = r.uniform(220, 360)
        rows.append((X0 + r.uniform(-0.06, 0.24) * SW, Y0 + r.uniform(-0.07, 0.14) * SH, rx, rx * r.uniform(0.5, 0.7)))
    for B in boughs:
        C = brush.path(np.c_[B, np.ones(len(B))], 2.0)
        s = 0.0
        while True:
            s += r.uniform(120, 200)
            i = int(s / 2)
            if i >= len(C):
                break
            rx = r.uniform(110, 210) * (1 - 0.55 * i / len(C))
            rows.append((C[i, 0] + r.normal(0, 25), C[i, 1] + rx * r.uniform(0.15, 0.55), rx, rx * r.uniform(0.5, 0.75)))
    mass, shade, blades = clumps(r, rows, 78)
    m = foliage(box, r, mass, shade, blades, 78)
    for B in boughs:
        box.stroke("bough", [(bx, by, 1 - 0.85 * k / (len(B) - 1)) for k, (bx, by) in enumerate(B)], 17, load=1.0,
                   dryness=1.3, reach=1400)
    box.lay(box.take("bough") * (1 - 0.8 * m), sepia=0.7, indigo=0.45, rose=0.12)
    for cx, cy, rx, ry in rows[::2]:
        a, L = r.uniform(0.2, 1.2), r.uniform(1.0, 1.8) * rx
        box.stroke("drag", [(cx - rx * 0.6, cy - ry * 0.3, 0.6), (cx, cy, 0.8), (cx + np.cos(a) * L / 2, cy + np.sin(a) * L / 2, 0.3)],
                   r.uniform(16, 26), load=0.5, dryness=2.2, reach=L, clumps=7)
    box.lay(box.take("drag", dry=0.55) * 0.5, cobalt=0.3, viridian=0.25, raw_sienna=0.15, indigo=0.1)
    pick = blades[r.uniform(0, 1, len(blades)) < 0.05]
    box.body(marks(pick * [1, 1, 0.8, 0.8, 1, 1]) * 0.7, "#aab8ac")


CROWNS = [(0.775, 0.60, 0.075, 0.32), (0.865, 0.665, 0.07, 0.26), (0.955, 0.59, 0.085, 0.32), (1.03, 0.66, 0.06, 0.26)]


def grove_rows(r):
    """Olives on the slope below the terrace, their crowns cut off by its brink, each a few clumps."""
    rows = []
    for u, v, hw, hh in CROWNS:
        cx, top, w, h = X0 + u * SW, Y0 + v * SH, hw * SW, hh * SH
        for _ in range(5):
            rx = w * r.uniform(0.45, 0.65)
            rows.append((cx + r.uniform(-0.55, 0.55) * w, top + r.uniform(0.25, 0.75) * h, rx, rx * r.uniform(0.6, 0.85)))
    return rows


def ground(box, r, site):
    """The terrace in the sun: one pale wash of ochre and rose, run out nearly dry so the paper
    sparkles through; broad strokes of rose and ochre laid across it side by side; a dry brush of
    light red for the beaten earth; the olive's shade lying across the near ground; weeds along the
    foot of the wall, and the short shadow of the wall's end."""
    g = site.ground
    _, dep = box.wash(g, 1.5, pool=0.25, rim=0.4, rim_width=4, grain=0.35, scale=220)
    skip = noise.smoothstep(0.3, 1.2, noise.stretched((H, W), 400, 40, r)) * noise.smoothstep(0.42, 0.25, box.sheet.tooth)
    box.lay(dep * (1 - 0.85 * skip) * 0.13, yellow_ochre=0.55, rose=0.3, light_red=0.12)
    for k in range(10):
        y, x, L = Y0 + SH * r.uniform(0.7, 1.02), r.uniform(X0 - 300, X1 - 200), r.uniform(400, 1100)
        box.stroke("rose" if k % 2 else "ochre", [(x, y, 0.2), (x + L / 2, y + r.normal(0, 8), 1.0),
                                                  (x + L, y + r.normal(0, 12), 0.1)], r.uniform(30, 60), load=0.9,
                   dryness=1.4, reach=0.9 * L, clumps=8)
    box.lay(ndimage.gaussian_filter(box.take("rose"), 1.5) * g * 0.16, rose=0.5, light_red=0.4)
    box.lay(ndimage.gaussian_filter(box.take("ochre"), 1.5) * g * 0.16, yellow_ochre=0.6, raw_sienna=0.3)
    for _ in range(18):
        y, x, L = Y0 + SH * r.uniform(0.72, 1.0), r.uniform(X0 - 200, X1), r.uniform(200, 600)
        box.stroke("earth", [(x, y, 0.5), (x + L / 2, y + r.normal(0, 6), 0.8), (x + L, y + r.normal(0, 10), 0.4)],
                   r.uniform(25, 50), load=0.45, dryness=2.2, reach=L, clumps=8)
    box.lay(box.take("earth", dry=0.45) * g * 0.4, light_red=0.4, raw_sienna=0.35, sepia=0.1)
    thick = 1.25 - np.hypot(U / 0.55, (V - 1.06) / 0.24) + 0.3 * noise.fbm((H, W), 200, r, octaves=3)
    rows, lights, _ = dapple(r, thick, (X0 - 200, Y0 + 0.6 * SH, X0 + 0.75 * SW, Y1 + 100), A=-0.1, size=1.6,
                             squash=0.5, n=380, fringe=0.4)
    m = broad(np.clip(marks(rows) - marks(lights, belly=0.5, start=0.5), 0, 1), 11) * g
    wet, dep = box.wash(m, 1.5, pool=0.3, rim=0.9, rim_width=3, tides=0.15, grain=0.55, scale=140)
    box.lay(dep * 0.6, **SHADE)
    box.lay(wc.charge(m * noise.smoothstep(0.3, 1.2, noise.field((H, W), 90, r)), wet, 25, r) * 0.3,
            rose=0.5, yellow_ochre=0.3)
    tufts = []
    for x in np.sort(r.uniform(X0, site.xc, 16)):
        y = site.foot[int(x)] + 4
        k = int(r.integers(5, 14))
        L = r.uniform(18, 55, k) * site.near[int(x)]
        tufts.append(np.stack([x + r.normal(0, 18, k), y + r.uniform(-2, 6, k), L, L * r.uniform(0.12, 0.22, k),
                               -np.pi / 2 + r.normal(0, 0.45, k), r.normal(0, 0.15, k)], 1))
    tufts = np.vstack(tufts)
    pick = r.uniform(0, 1, len(tufts))
    box.touch(tufts[pick < 0.5], 0.6, yellow_ochre=0.4, raw_sienna=0.35, viridian=0.1)
    box.touch(tufts[(pick >= 0.5) & (pick < 0.85)], 0.7, viridian=0.3, raw_sienna=0.3, gamboge=0.15)
    box.touch(tufts[pick >= 0.85], 0.9, indigo=0.4, viridian=0.3)
    xc, xe = site.xc, int(site.endx.mean())
    fy, fe = site.foot[xc], site.e_foot[xe]
    im = Image.new("F", (W, H), 0.0)
    ImageDraw.Draw(im).polygon([(xc - 4, fy + 3), (xe, fe + 2), (xe + 0.1 * SW, fe - 0.02 * SH),
                                (xc + 0.05 * SW, fy - 0.015 * SH)], fill=1.0)
    cast = ndimage.gaussian_filter(np.asarray(im), 1.0) * g
    _, dep = box.wash(cast, 1.5, pool=0.2, rim=0.9, rim_width=2.5, grain=0.5, scale=80)
    box.lay(dep * (0.6 + 0.4 * np.clip(1 - (XX - xe) / (0.1 * SW), 0, 1)), **SHADE)


def drawing(sheet, r, site):
    """The few pencil lines the painting went over: the wall, its end, the sea's edge."""
    press = np.zeros((H, W), np.float32)
    xs = np.arange(X0 - 40, site.xc, 18)
    for line, p in ((site.sky_edge, 0.5), (site.foot, 0.45), (site.under, 0.3)):
        pencil.line(press, np.stack([xs, line[xs]], 1), 2.2, r, pressure=p)
    ys = np.arange(int(site.e_top[site.xc]), int(site.foot[site.xc]), 18)
    pencil.line(press, np.stack([site.corner[ys], ys], 1), 2.2, r, pressure=0.5)
    pencil.line(press, np.stack([site.endx[ys], ys], 1), 2.2, r, pressure=0.35)
    xh = np.arange(X0, X1, 24)
    pencil.line(press, np.stack([xh, site.horizon[xh]], 1), 2.0, r, pressure=0.3)
    return pencil.catch(press, sheet, grip=0.7)


def paint(seed=7):
    r = noise.rng(seed)
    sheet = wash.rough((H, W), _s(r), tint="#f6f2e6")
    site = Site(r)
    box = Box(sheet, r, frame(r))
    gmass, gshade, gblades = clumps(r, grove_rows(r), 40, hang=0.35)
    out = np.clip(site.ground + site.wall, 0, 1)
    gx, gy = gblades[:, 0].clip(0, W - 1).astype(int), gblades[:, 1].clip(0, H - 1).astype(int)
    gblades = gblades[(gy < site.brink[gx] - 12) & (out[gy, gx] < 0.5) & (gx > site.endx.max() + 10)]
    gmass, gshade = gmass * (1 - out), gshade * (1 - out)
    gmask = broad(np.clip(gmass + marks(gblades), 0, 1), 5) * (1 - out)
    crows, cside, cmask = cypress_rows(r)
    cmask = cmask * (1 - gmask * (YY > Y0 + 0.63 * SH))
    sky(box, r, site)
    mountains(box, r, site)
    sea(box, r, site, np.clip(ndimage.grey_erosion(cmask, 5) + ndimage.grey_dilation(gmask, 5), 0, 1))
    whitewash(box, r, site)
    wall_shade(box, r, site)
    end_face(box, r, site)
    ground(box, r, site)
    foliage(box, r, gmass, gshade, gblades, 40, silver=1.6, dark=0.6)
    cypress(box, r, crows, cside, cmask)
    olive(box, r, site)
    box.lay(drawing(sheet, r, site), graphite=1.0)
    img = wash.cockle(box.glaze(), sheet, box.soak, r)
    return plate.mount(img, sheet)
