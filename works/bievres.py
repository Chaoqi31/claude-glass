"""Garden Flowers in a Turquoise Vase. Pastel on buff paper.

Around 1900 Redon put away the charcoal blacks he had worked in for twenty years and took up colour,
and in the last fifteen years of his life he drew flowers again and again: bunches brought in from the
garden at Bièvres, outside Paris, standing in a vase on a table that is hardly there. He drew them in
pastel, sticks of nearly pure pigment held together with a little gum, on toned paper. The colour sits
on the tooth of the sheet as a dry powder, matt and very bright. Laid with the side of the stick it
catches on the ridges of the paper and leaves the hollows bare, so that a second colour laid over a first
lets the first show through, and a finger drawn across it drives the powder into the hollows in a soft
bloom. His bouquets float in a field of colour, some flowers sharp and others going back into it.

This sheet was worked from the back to the front. The field went on first, in veils of the side of the
stick laid lightly over one another and rubbed with the flat of the hand: grey-blue and periwinkle on
the left, deepening toward the corner; a glow of gold behind the place where the flowers would go; and
below, where the table would be, no edge, only a warming into terracotta and the violet shadow of the
vase. A second sitting of lighter sticks was dragged across the rubbed colour and left as it fell,
catching on the tooth. The vase is drawn by hand and its two sides do not quite match: long strokes
down its swelling body, deep turquoise, aqua where the light falls and blue on the other side, with
ultramarine, green and violet broken into the glaze, rubbed in places, and the outline restated with
the tip. The bouquet began as a haze behind the bunch, cool on one side and warm on the other, and
the dark of the foliage at its heart. Over that went the sprays, the larkspur and the leaves, and then
the flowers, each laid in broad veils of the side of the stick and drawn over with the tip: the creases
of the poppies' petals, the dark blotch at their hearts with its ring of stamens round the green seed
head, the fringed ring round each anemone's black boss, the ragged florets of the cornflowers. Some
flowers stand in shadow and were drawn with deeper sticks. Those at the back and at the edges were laid
lightly, and the field was drawn back over their far sides and rubbed in with the finger. A little loose
powder fell down the sheet as it was worked.
"""

import numpy as np
from scipy import ndimage

from atelier import brush, noise, paper, plate
from atelier.color import lin

TITLE = "Garden Flowers in a Turquoise Vase"
DATE = "2026"
MEDIUM = "Pastel on buff paper, laid with the side and the tip of the stick and rubbed in places with the hand"
AFTER = ("Odilon Redon, the pastels of flowers, c. 1905–1914: Vase of Flowers; Bouquet in a Chinese Vase; "
         "Wild Flowers in a Long-necked Vase")
ROOM = "The Garden"
YEAR = 1910
PLACE = "Paris"
REGION = "Europe"
NOTE = ("Poppies, anemones, cornflowers, marigolds and larkspur in a turquoise vase, coming out of a field of rubbed "
        "blue and gold with no table under it. A few of the flowers blaze, some stand in shadow, and those at the "
        "edges are given back to the field.")

H, W = 2800, 2200
EDGE = 78                        # the wall shows this far round the sheet
TINT = "#bda686"                 # the paper, an unbleached buff


def sheet_of(r, shape=(H, W), edge=EDGE):
    """Pastel paper: a buff sheet with a coarse, irregular tooth of grains and short fibres, so
    that the stick catches on the grains and skips the hollows between them."""
    fib = ndimage.gaussian_filter(noise.fibers(shape, r, count=shape[0] * shape[1] // 650, length=(5, 24),
                                               curl=0.3, strength=(0.2, 0.8)), 0.6)
    tooth = paper._norm(0.55 * noise.field(shape, 1.2, r) + 0.4 * noise.field(shape, 2.6, r)
                        + 0.2 * noise.field(shape, 7, r) + 0.4 * fib)
    formation = noise.fbm(shape, max(shape) / 5, r, octaves=6, gain=0.55)
    light = 1 + 0.025 * formation + 0.03 * (tooth - 0.5)
    alpha = paper.deckle(shape, edge, r, ragged=1.0) if edge else np.ones(shape, np.float32)
    return paper.Sheet((lin(TINT)[None, None] * light[..., None]).astype(np.float32), tooth, fib, alpha)


def _dab(c, d, r, p):
    """A touch of the blunt end of the stick, pressed and turned: a round mark with a ragged rim."""
    rad = max(1.0, d / 2)
    x0, y0 = int(c[0] - rad) - 3, int(c[1] - rad) - 3
    n = int(2 * rad) + 7
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dx, dy = xx + x0 - c[0], yy + y0 - c[1]
    k = r.integers(3, 6)
    rim = 1 + 0.12 * np.cos(k * np.arctan2(dy, dx) + r.uniform(0, 6.3)) + 0.06 * noise.field((n, n), 2.0, r)
    buf = p * noise.smoothstep(1.0, 0.7, np.hypot(dx, dy) / (rad * rim)) * (1 + 0.15 * noise.field((n, n), 1.5, r))
    return ndimage.gaussian_filter(buf, 0.5), (y0, x0)


def stick(P, width, r, p=0.8, kind="side"):
    """The pressure one stroke of a pastel stick leaves along control points P [(x, y), ...].
    Drawn with its side ("side"), the stick lays a band `width` px across, even over its face and
    falling away fast at its edges; the face is worn unevenly, so the band is streaked along its
    length, and its ends are cut on the slant where the stick met and left the paper at an angle.
    Laid lightly and lifted slowly ("veil"), the band has no ends and hardly an edge.
    Swung so that the band swells and narrows, it lays a leaf ("lens", pointed at both ends) or a
    petal ("petal", narrow at the heart of the flower and round at its end). With its tip ("tip")
    it draws a line, fullest down the middle; a single point is a dab. -> (buffer, (y0, x0))"""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    if len(P) == 1:
        return _dab(P[0], width, r, p)
    C = brush.path(P, 1.0)[:, :2]
    L = len(C)
    if L < 3:
        return None
    g = np.gradient(C, axis=0)
    g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
    tip = kind == "tip"
    n = max(3, int(np.ceil(width / (0.45 if tip else 0.8))))
    u = (np.arange(n) + 0.5) / n * 2 - 1
    t = np.arange(L, dtype=np.float32)[None]
    s = t[0] / (L - 1)
    if tip:
        prof = np.sqrt(np.clip(1 - u * u, 0, 1))
        streak = np.clip(1 + 0.15 * r.standard_normal(n), 0.6, 1.3)
        down, up = 1.5, r.uniform(3, 10)
    else:
        prof = noise.smoothstep(1.0, 0.3 if kind == "veil" else 0.72, np.abs(u)) * (1 + r.uniform(-0.35, 0.35) * u)
        streak = np.clip(1 + 0.7 * ndimage.gaussian_filter1d(r.standard_normal(n), 1.0), 0.1, 1.7)
        down, up = ((r.uniform(2, 6), r.uniform(6, 26)) if kind == "side" else
                    (r.uniform(0.15, 0.35) * L, r.uniform(0.2, 0.45) * L) if kind == "veil" else (1.0, 2.0))
    lean = np.clip(r.uniform(-0.6, 0.6, 2) * width / 2, -L / 4, L / 4) if kind == "side" else np.zeros(2)
    if kind == "lens":
        shape = np.clip(np.sin(np.pi * s), 0, 1) ** 0.7
    elif kind == "petal":
        shape = np.where(s < 0.68, (s / 0.68) ** 0.55, np.sqrt(np.clip(1 - ((s - 0.68) / 0.32) ** 2, 0, 1)))
    else:
        shape = 1.0
    a = (lean[0] * u + abs(lean[0]) + r.uniform(0, 2, n))[:, None]
    b = (L - 1 - lean[1] * u - abs(lean[1]) - r.uniform(0, 2, n))[:, None]
    env = noise.smoothstep(a, a + down, t) * noise.smoothstep(b, b - up, t)
    if L > 12:
        along = p * (1 + 0.12 * noise.line1d(L, 30, r))
        crumb = 1 + (0.12 if tip else 0.3) * noise.line1d(L, 4, r, rows=n)
        hw = width / 2 * shape * (1 + 0.06 * noise.line1d(L, 50, r))
    else:
        along, crumb, hw = np.full(L, p), 1.0, width / 2 * shape * np.ones(L)
    hw = np.maximum(hw, 0.3)
    X = C[None, :, 0] - g[None, :, 1] * u[:, None] * hw[None]
    Y = C[None, :, 1] + g[None, :, 0] * u[:, None] * hw[None]
    V = (prof * streak)[:, None] * env * along[None] * crumb * (2 * hw[None] / n)
    buf, o = brush.splat(X.ravel(), Y.ravel(), V.ravel().astype(np.float32))
    return ndimage.gaussian_filter(buf, 0.5 if tip else 0.7), o


def _box(shape, y0, x0, y1, x1, pad=0):
    return max(0, y0 - pad), max(0, x0 - pad), min(shape[0], y1 + pad), min(shape[1], x1 + pad)


class Pastel:
    """The sheet as it is worked: every sitting of the sticks is laid over what is already there."""

    def __init__(self, sheet, r):
        self.img, self.paper, self.tooth, self.r = sheet.color.copy(), sheet.color, sheet.tooth, r
        self.shade = 0.0             # how far the passage being drawn lies in shadow: its sticks are deeper

    def layer(self, strokes, grip=0.8, soft=0.12, dust=0.0, clip=None):
        """Lay strokes [(P, width, colour, pressure, side), ...]. Pastel is opaque, but it only
        sits where the stick touched: a light stroke catches on the tops of the grains and leaves
        the hollows bare, so whatever was there before (paper or an earlier colour) shows between,
        and pressing harder, or going over a place again, drives it down into the hollows too.
        `dust` lets a little powder fall from the stroke down the sheet."""
        marks = []
        dusk = (1 - 0.5 * self.shade) * DUSK ** self.shade
        for P, wd, c, p, kind in strokes:
            m = stick(P, wd, self.r, p, kind)
            if m is not None:
                marks.append((m[0], m[1], (lin(c) if isinstance(c, str) else np.asarray(c, np.float32)) * dusk))
        if not marks:
            return
        y0, x0, y1, x1 = _box(self.img.shape, min(o[0] for _, o, _ in marks), min(o[1] for _, o, _ in marks),
                              max(o[0] + b.shape[0] for b, o, _ in marks), max(o[1] + b.shape[1] for b, o, _ in marks))
        if y1 <= y0 or x1 <= x0:
            return
        press = np.zeros((y1 - y0, x1 - x0), np.float32)
        col = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        for b, (oy, ox), c in marks:
            oy, ox = oy - y0, ox - x0
            ys, xs, ye, xe = _box(press.shape, oy, ox, oy + b.shape[0], ox + b.shape[1])
            if ye > ys and xe > xs:
                piece = b[ys - oy:ye - oy, xs - ox:xe - ox]
                press[ys:ye, xs:xe] += piece
                col[ys:ye, xs:xe] += piece[..., None] * c
        sl = (slice(y0, y1), slice(x0, x1))
        c = col / (press[..., None] + 1e-6)
        if clip is not None:
            press *= clip[sl]
        reach = 1 - grip * press
        tooth = self.tooth[sl] + 0.08 * noise.field(press.shape, 1.3, self.r)
        cov = np.clip(1.5 * press, 0, 1) * noise.smoothstep(reach - soft, reach + soft, tooth)
        self.img[sl] += (c - self.img[sl]) * cov[..., None]
        if dust:
            self.fall(cov, c, y0, x0, dust)

    def fall(self, cov, c, y0, x0, amount):
        """Loose powder: grains knocked off the stick drift down the sheet and settle in specks."""
        r = self.r
        n = int(amount * cov.sum() / 400)
        if n < 1:
            return
        ys, xs = r.integers(0, cov.shape[0], 4 * n), r.integers(0, cov.shape[1], 4 * n)
        k = r.random(4 * n) < cov[ys, xs]
        ys, xs = ys[k][:n], xs[k][:n]
        if not len(ys):
            return
        Y, X = y0 + ys + r.exponential(16, len(ys)), x0 + xs + r.normal(0, 4, len(ys))
        v = r.uniform(0.4, 1.2, len(ys)).astype(np.float32)
        a, o = brush.splat(X, Y, v)
        cc = [brush.splat(X, Y, v * c[ys, xs, j])[0] for j in range(3)]
        a = ndimage.gaussian_filter(a, 0.55)
        cc = np.stack([ndimage.gaussian_filter(q, 0.55) for q in cc], -1) / (a[..., None] + 1e-6)
        ys0, xs0, ye, xe = _box(self.img.shape, o[0], o[1], o[0] + a.shape[0], o[1] + a.shape[1])
        a = np.clip(1.6 * a, 0, 0.9)[ys0 - o[0]:ye - o[0], xs0 - o[1]:xe - o[1]]
        cc = cc[ys0 - o[0]:ye - o[0], xs0 - o[1]:xe - o[1]]
        self.img[ys0:ye, xs0:xe] += (cc - self.img[ys0:ye, xs0:xe]) * a[..., None]

    def smudge(self, P, width, k=0.6, sigma=5.0, drag=8.0):
        """A finger drawn along P: it pushes the powder down into the hollows and drags the
        colours it passes into one another, leaving a soft bloom with no grain in it."""
        m = stick(P, width, self.r, 1.0, "tip")
        if m is None:
            return
        b, (oy, ox) = m
        pad = int(3 * sigma + drag + width / 3 + 4)
        b = np.pad(b, pad)
        oy, ox = oy - pad, ox - pad
        b = ndimage.gaussian_filter(b, width / 7)
        b = np.clip(b / (b.max() + 1e-9) * 1.4, 0, 1) * k
        y0, x0, y1, x1 = _box(self.img.shape, oy, ox, oy + b.shape[0], ox + b.shape[1])
        if y1 <= y0 or x1 <= x0:
            return
        b = b[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        sub = self.img[y0:y1, x0:x1]
        P = np.asarray(P, np.float64)
        d = P[-1] - P[0]
        d = d / (np.hypot(*d) + 1e-9) * drag
        bl = ndimage.shift(ndimage.gaussian_filter(sub, (sigma, sigma, 0)), (d[1], d[0], 0), order=1, mode="nearest")
        sub += (bl - sub) * b[..., None]

    def rub(self, mask, sigma=8.0, wear=0.45):
        """The flat of the hand passed over a whole passage where `mask` says: the powder is
        driven into the hollows and the strokes run together into one soft tone, and it is wiped
        thin off the tops of the grains, where the paper shows again in specks."""
        self.img += (ndimage.gaussian_filter(self.img, (sigma, sigma, 0)) - self.img) * mask[..., None]
        self.img += (self.paper - self.img) * (wear * mask * noise.smoothstep(0.6, 0.95, self.tooth))[..., None]

    def finish(self, sheet):
        """The tooth still shows through the powder, lit from the upper left."""
        t = ndimage.gaussian_filter(sheet.tooth, 0.8)
        gy, gx = np.gradient(t)
        s = -(gx + gy)
        s /= s.std() + 1e-9
        return self.img * (1 + 0.03 * np.clip(s, -3, 3))[..., None]


LIGHT = -2.3                     # the light comes from the upper left
DUSK = np.float32([0.8, 0.78, 1.0])           # what shadow does to a colour, besides deepening it
BLACK = "#16111c"
BLUES = ("#3a4fc8", "#4462d8", "#5448bc", "#3346b0")
SHEEN = ("#f4682c", "#b01a3a", "#b4547e")     # a poppy's petals: orange in the light, carmine away from it, mauve at the heart


def tint(c, r, s=0.05):
    """A stick of colour `c` (a name, or linear RGB): no two sticks of one name quite match."""
    c = lin(c) if isinstance(c, str) else np.asarray(c, np.float32)
    return np.clip(c * np.exp(r.normal(0, s, 3)), 0, 1).astype(np.float32)


class Head:
    """A flower's face, turned by `rot` and seen at a slant (`squash` < 1 foreshortens it):
    head(rr, a) is the point at `rr` of the radius and angle `a` on the face."""

    def __init__(self, x, y, R, rot=0.0, squash=1.0):
        self.x, self.y, self.R, self.rot, self.sq = x, y, R, rot, squash

    def __call__(self, rr, a):
        u, v = rr * self.R * np.cos(a), rr * self.R * np.sin(a) * self.sq
        c, s = np.cos(self.rot), np.sin(self.rot)
        return self.x + c * u - s * v, self.y + s * u + c * v

    def ray(self, a, r0, r1, bend=0.0, n=4):
        return [self(r0 + (r1 - r0) * k / (n - 1), a + bend * (k / (n - 1)) ** 2) for k in range(n)]

    def arc(self, rr, a0, a1, n=5):
        return [self(rr, a) for a in np.linspace(a0, a1, n)]

    def w(self, a, width, across=True):
        """`width` on the face as it shows for a mark running out along the radius at angle `a`
        (or, with across=False, running round it)."""
        q = (np.sin(a), self.sq * np.cos(a)) if across else (np.cos(a), self.sq * np.sin(a))
        return width * np.hypot(*q)


def _spokes(r, n, jit):
    return r.uniform(0, 2 * np.pi) + np.arange(n) * 2 * np.pi / n + r.normal(0, jit, n)


def poppy(pa, h, r, base, deep, light, line, k=1.0):
    """A field poppy: four crumpled petals, the far ones first, each laid deep at the heart and lit
    at the rim, with the creases of the crumpling drawn in; the near petals' edges drawn where they
    lie over the far ones; at the heart a black blotch round the grey-green seed head and its
    crown, ringed with black stamens."""
    R = h.R
    angs, reach = _spokes(r, 4, 0.15), r.uniform(0.9, 1.05, 4)
    for i in np.argsort(np.sin(angs)):
        a, e = angs[i], reach[i]
        pa.layer([(h.ray(a + da, 0.04, 0.85 * e, r.normal(0, 0.1)), h.w(a, 0.75 * R), tint(deep, r), 0.8 * k, "petal")
                  for da in (-0.3, 0.3)], grip=0.7)
        pa.layer([(h.ray(a + da, 0.12, e * r.uniform(0.93, 1.02), r.normal(0, 0.12)), h.w(a + da, 0.66 * R),
                   tint(base, r), r.uniform(0.8, 0.95) * k, "petal") for da in (-0.42, 0.0, 0.42)], grip=0.65, dust=0.15)
        pa.layer([(h.ray(a + r.uniform(-0.55, 0.55), 0.15, r.uniform(0.55, 0.85), r.normal(0, 0.25)), 0.1 * R,
                   tint(deep, r), 0.55 * k, "lens") for _ in range(3)], grip=0.7)
        lit = np.cos(a - LIGHT)
        pa.layer([(h.ray(a + r.uniform(-0.4, 0.4), 0.25, r.uniform(0.75, 0.95) * e, r.normal(0, 0.15)),
                   h.w(a, 0.22 * R), tint(SHEEN[0] if lit > 0 else SHEEN[1], r), 0.45 * k, "petal") for _ in range(2)]
                 + [(h.ray(a + r.uniform(-0.3, 0.3), 0.08, 0.32), h.w(a, 0.18 * R), tint(SHEEN[2], r), 0.4 * k, "lens")],
                 grip=0.7)
        if lit > -0.3:
            pa.layer([(h.arc(r.uniform(0.76, 0.86) * e, a - 0.55, a + 0.4), h.w(a, 0.15 * R, False), tint(light, r),
                       0.5 * k, "side")], grip=0.7)
        S = [(h.arc(0.97 * e, a0, a0 + r.uniform(0.4, 0.9)), max(3.0, 0.022 * R), tint(line, r), 0.75 * k, "tip")
             for a0 in [a + r.uniform(-0.75, 0.0)]]
        if np.sin(a) > 0:
            S += [(h.ray(a + s * 0.76, 0.28, 0.96 * e, s * 0.06, 4), 4.0, tint(line, r), 0.85 * k, "tip") for s in (-1, 1)]
            S += [(h.ray(a + s * 0.7, 0.4, 0.92 * e, s * 0.06, 4), 3.0, tint(light, r), 0.6 * k, "tip") for s in (-1, 1)]
        S += [(h.ray(a + r.uniform(-0.5, 0.5), 0.3, r.uniform(0.65, 0.9), r.normal(0, 0.2), 5), 2.0, tint(line, r),
               0.4 * k, "tip") for _ in range(2)]
        pa.layer(S)
    # the heart: the dark blotches at the feet of the petals, run together into one ragged round;
    # on it the ring of stamens, fine dark hairs with pale anthers; and in the middle the grey-green
    # seed head with its rayed crown
    pa.layer([(h.ray(a, 0.02, r.uniform(0.19, 0.25), r.normal(0, 0.2), 3), h.w(a, 0.17 * R), tint("#1e0f24", r), 0.9,
               "lens") for a in _spokes(r, 11, 0.15)], grip=0.9)
    sa = _spokes(r, int(np.clip(0.2 * R, 16, 46)), 0.07)
    ends = r.uniform(0.2, 0.31, len(sa))
    pa.layer([(h.ray(a, 0.1, e, r.normal(0, 0.2), 3), max(1.8, 0.01 * R), tint(BLACK, r), 0.85, "tip")
              for a, e in zip(sa, ends)])
    pa.layer([([h(e * r.uniform(0.6, 1.0), a + r.normal(0, 0.05))], max(3.0, r.uniform(0.022, 0.034) * R),
               tint(("#c8bc8a", "#9aa48c", "#e0d49a", "#2a1830")[r.integers(4)], r), 0.95, "dab")
              for a, e in zip(sa, ends) for _ in range(2)])
    pa.layer([([h(0, 0)], 0.17 * R, tint("#869a70", r), 1.0, "dab")], grip=0.95)
    pa.layer([([h(0.03, LIGHT)], 0.08 * R, tint("#b9c89c", r), 0.8, "dab")])
    pa.layer([(h.ray(a, 0.0, 0.075, 0, 2), max(1.8, 0.009 * R), tint("#2c3420", r), 0.85, "tip") for a in _spokes(r, 8, 0.1)])


def anemone(pa, h, r, base, deep, light, line, eye, boss=BLACK, k=1.0):
    """A poppy anemone: six or seven broad round petals of one colour, deep at their feet and
    veined, lit at their tips; in the red and violet ones the feet of the petals are white. At the
    heart a black boss in a fringed ring of stamens, each a hair with a dark head."""
    R = h.R
    n = int(r.integers(6, 8))
    angs, reach = _spokes(r, n, 0.12), r.uniform(0.86, 1.04, n)
    pa.layer([(h.ray(a, 0.05, 0.7 * e), h.w(a, 0.55 * R), tint(deep, r), 0.75 * k, "petal")
              for a, e in zip(angs, reach)], grip=0.7)
    pa.layer([(h.ray(a + da, 0.1, e * r.uniform(0.95, 1.02), r.normal(0, 0.08)), h.w(a, 0.6 * R), tint(base, r),
               r.uniform(0.8, 0.95) * k, "petal") for a, e in zip(angs, reach) for da in (-0.13, 0.13)],
             grip=0.65, dust=0.15)
    pa.layer([(h.arc(0.78 * e, a - 0.24, a + 0.24, 3), h.w(a, 0.16 * R, False), tint(light, r), 0.45 * k, "side")
              for a, e in zip(angs, reach) if np.cos(a - LIGHT) > -0.2], grip=0.7)
    pa.layer([(h.ray(a + r.uniform(-0.25, 0.25), 0.3, r.uniform(0.6, 0.85), r.normal(0, 0.1)), 2.0, tint(deep, r),
               0.35 * k, "tip") for a in angs for _ in range(2)])
    S = []
    for a, e in zip(angs, reach):
        if r.random() < 0.75:
            S.append((h.ray(a + np.pi / n * r.uniform(0.8, 1.0), 0.35, 0.9 * e, 0.05, 3), 2.8, tint(line, r), 0.6 * k,
                      "tip"))
        if r.random() < 0.5:
            S.append((h.arc(0.97 * e, a + r.uniform(-0.35, -0.1), a + r.uniform(0.1, 0.35), 4), 2.6, tint(line, r),
                      0.7 * k, "tip"))
    pa.layer(S)
    if eye:
        pa.layer([(h.ray(a, 0.08, r.uniform(0.34, 0.42)), h.w(a, 0.3 * R), tint(eye, r, 0.03), 0.75, "petal")
                  for a in angs], grip=0.7)
    fa = _spokes(r, int(0.75 * R), 0.06)
    fa = fa[r.random(len(fa)) < 0.8]
    ends = r.uniform(0.22, 0.34, len(fa))
    pa.layer([(h.ray(a, 0.1, e, r.normal(0, 0.12), 3), r.uniform(1.8, 2.8), tint(boss, r), 0.85, "tip")
              for a, e in zip(fa, ends)])
    pa.layer([([h(e, a)], r.uniform(3.0, 5.0), tint(boss, r), 1.0, "dab") for a, e in zip(fa, ends) if r.random() < 0.6])
    pa.layer([([h(r.uniform(0, 0.04), r.uniform(0, 6.3))], r.uniform(0.22, 0.28) * R, tint(boss, r), 1.0, "dab")
              for _ in range(2)], grip=0.9)
    pa.layer([([h(0.06, LIGHT)], 0.06 * R, tint("#6a6a8a", r), 0.6, "dab")])


def cornflower(pa, h, r, k=1.0):
    """A cornflower: a ring of funnelled blue florets with ragged mouths round a knot of purple."""
    R = h.R
    angs = _spokes(r, int(r.integers(9, 13)), 0.12)
    pa.layer([([h(0, 0)], 1.1 * R, tint(BLUES[0], r), 0.5 * k, "dab")], grip=0.7)
    pa.layer([(h.ray(a, 0.2, r.uniform(0.85, 1.05), r.normal(0, 0.1)), h.w(a, 0.42 * R), tint(BLUES[r.integers(4)], r),
               r.uniform(0.85, 1.0) * k, "petal") for a in angs], grip=0.7, dust=0.15)
    pa.layer([(h.ray(a + d, 0.78, r.uniform(1.0, 1.12), d * 1.5, 3), max(2.4, 0.06 * R), tint("#6286ea", r), 0.9 * k,
               "tip") for a in angs for d in (-0.1, 0.0, 0.1)])
    pa.layer([([h(r.uniform(0, 0.2), r.uniform(0, 2 * np.pi))], 0.16 * R, tint("#2e1c5c", r), 1.0, "dab")
              for _ in range(6)])


def marigold(pa, h, r, outer=("#f39a1e", "#f7b52a", "#f08218"), inner=("#e8661a", "#ee7a16", "#d8541a"), k=1.0):
    """A pot marigold: a full head of short petals, ring inside ring, round a brown eye."""
    R = h.R
    pa.layer([([h(0, 0)], 1.75 * R, tint(outer[0], r), 0.7 * k, "dab")], grip=0.7)
    a1 = _spokes(r, int(r.integers(20, 28)), 0.06)
    pa.layer([(h.ray(a, 0.42, r.uniform(0.92, 1.06)), h.w(a, 0.3 * R), tint(outer[r.integers(3)], r), 0.9 * k,
               "petal") for a in a1], grip=0.7, dust=0.15)
    a2 = _spokes(r, int(r.integers(14, 19)), 0.08)
    pa.layer([(h.ray(a, 0.15, r.uniform(0.55, 0.68)), h.w(a, 0.27 * R), tint(inner[r.integers(3)], r), 0.9 * k,
               "petal") for a in a2], grip=0.7)
    pa.layer([(h.ray(a, 0.5, 0.92), 2.4, tint("#b4410e", r), 0.6 * k, "tip") for a in a1[::2] + np.pi / len(a1)])
    pa.layer([(h.ray(a, 0.55, 0.88), 3.2, tint("#f8dc4a", r), 0.75 * k, "tip") for a in a1 if np.cos(a - LIGHT) > 0.5])
    pa.layer([([h(0, 0)], 0.42 * R, tint("#6e3414", r), 1.0, "dab")]
             + [([h(r.uniform(0, 0.16), r.uniform(0, 6.3))], 4.0, "#3a1a0a", 1.0, "dab") for _ in range(5)], grip=0.9)


def daisy(pa, h, r, k=1.0):
    """A marguerite: white rays, those turned from the light gone grey-blue, round a yellow button."""
    R = h.R
    angs = _spokes(r, int(r.integers(16, 23)), 0.06)
    pa.layer([(h.ray(a, 0.2, r.uniform(0.85, 1.05), r.normal(0, 0.1)), h.w(a, 0.24 * R),
               tint("#f8f4ea" if np.cos(a - LIGHT) > -0.3 else "#cfd0dc", r, 0.03), r.uniform(0.85, 1.0) * k, "petal")
              for a in angs], grip=0.75)
    pa.layer([(h.ray(a + 0.06, 0.3, 0.9), 1.8, tint("#8c8a98", r), 0.45 * k, "tip") for a in angs if r.random() < 0.35])
    pa.layer([([h(0, 0)], 0.48 * R, tint("#f2be22", r), 1.0, "dab")], grip=0.9)
    pa.layer([(h.arc(0.14, LIGHT + 2.4, LIGHT + 4.0, 4), 0.12 * R, tint("#d27a18", r), 0.8, "side")])
    pa.layer([([h(r.uniform(0, 0.15), r.uniform(0, 6.3))], 3.0, "#6a4a12", 1.0, "dab") for _ in range(5)])


def geranium(pa, x, y, R, r, k=1.0):
    """A head of geranium: a close dome of small five-petalled florets, scarlet, lit pink on top
    and darker underneath."""
    fl = sorted(((x + 0.85 * R * np.sqrt(q) * np.cos(a), y + 0.75 * R * np.sqrt(q) * np.sin(a))
                 for a, q in zip(r.uniform(0, 2 * np.pi, 30), r.uniform(0, 1, 30))), key=lambda p: p[1])
    pa.layer([([(x, y)], 1.8 * R, tint("#d83a3a", r), 0.6 * k, "dab")], grip=0.7)
    pa.layer([([(fx + 0.1 * R * np.cos(b), fy + 0.1 * R * np.sin(b))], 0.14 * R,
                tint(("#ee3b3c", "#f2564a", "#e42e3e")[r.integers(3)], r), 0.95 * k, "dab")
               for fx, fy in fl for b in r.uniform(0, 6.3) + np.arange(5) * 1.2566], grip=0.75, dust=0.15)
    pa.layer([([(fx + r.normal(0, 3), fy + 0.08 * R)], 0.16 * R, tint("#a8142a", r), 0.75 * k, "dab")
              for fx, fy in fl if fy > y + 0.2 * R], grip=0.7)
    pa.layer([([(fx - 0.05 * R, fy - 0.05 * R)], 0.1 * R, tint("#f7909a", r), 0.7 * k, "dab") for fx, fy in fl[:8]])
    pa.layer([([(fx, fy)], 3.0, "#f8e2d2", 0.9, "dab") for fx, fy in fl])


def button(pa, x, y, R, r, k=1.0):
    """A buttercup: a cup of lemon yellow, a dark line drawn round part of it."""
    pa.layer([([(x, y)], 2 * R, tint("#f5de3c", r), 0.95 * k, "dab")], grip=0.8)
    pa.layer([([(x + 0.35 * R, y + 0.35 * R)], R, tint("#e2a81e", r), 0.7 * k, "dab")])
    a0 = r.uniform(0, 6.3)
    pa.layer([([(x + R * np.cos(a), y + R * np.sin(a)) for a in np.linspace(a0, a0 + r.uniform(1.5, 3.5), 5)], 2.4,
               tint("#4a4420", r), 0.7 * k, "tip")])


def larkspur(pa, P, size, r, k=1.0, pale=False):
    """A spike of larkspur: florets of blue and violet (or, `pale`, of lilac) up a leaning stem,
    smaller toward the top, and a tip of buds."""
    cols = ("#9aa6e6", "#b0a2e2", "#c4b0e6", "#a4b8ea", "#b89ad6") if pale else (*BLUES, "#6a4ec0")
    C = brush.path(np.asarray(P, np.float64), 1.0)[:, :2]
    g = np.gradient(C, axis=0)
    g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
    pa.layer([(P, 4.5, tint("#3f6a32", r), 0.8 * k, "tip")])
    fl, t, side = [], 0.0, 1
    while t < 0.86:
        i, s = int(t * (len(C) - 1)), size * (1 - 0.6 * t)
        if r.random() < 0.9:
            fl.append((C[i, 0] - g[i, 1] * side * 0.3 * s, C[i, 1] + g[i, 0] * side * 0.3 * s, s * r.uniform(0.7, 1.2)))
        t, side = t + s * r.uniform(0.35, 0.7) / len(C), -side
    pa.layer([([(x + 0.28 * s * np.cos(b), y + 0.28 * s * np.sin(b))], 0.5 * s,
                tint(cols[r.integers(5)], r), 0.95 * k, "dab")
               for x, y, s in fl for b in r.uniform(0, 6.3) + np.arange(5) * 1.2566], grip=0.75, dust=0.1)
    pa.layer([([(x - 0.15 * s, y - 0.15 * s)], 0.25 * s, tint("#e4e2f6" if pale else "#8ea4f2", r), 0.7 * k, "dab")
              for x, y, s in fl[::2]])
    pa.layer([([(x, y)], 0.22 * s, tint("#5a4a8a" if pale else "#221848", r), 1.0 * k, "dab") for x, y, s in fl])
    j = int(0.86 * (len(C) - 1))
    pa.layer([([C[j] + (r.normal(0, 4), r.normal(0, 4)), C[-1] + (r.normal(0, 6), r.normal(0, 6))], 0.3 * size,
               tint(cols[0] if pale else "#4a5ac8", r), 0.8 * k, "lens") for _ in range(3)])


def spray(pa, x, y, a, L, r, colour="#f6f2ea", k=1.0):
    """A spray of tiny flowers on hair-fine branching stems, with a few leaves no bigger than
    the flowers; some of the flowers have come away and drift off beyond it."""
    stems, dots, lv = [], [], []

    def branch(x, y, a, L, depth):
        pts = [(x, y)]
        for _ in range(4):
            a += r.normal(0, 0.15)
            x, y = x + np.cos(a) * L / 4, y + np.sin(a) * L / 4
            pts.append((x, y))
        stems.append((pts, max(2.2, 4.0 - 0.6 * depth), tint("#5f6a34", r), 0.75 * k, "tip"))
        if 0 < depth < 3 and r.random() < 0.7:
            px, py = pts[int(r.integers(1, 4))]
            b, l = a + r.choice((-1, 1)) * r.uniform(0.5, 1.1), r.uniform(22, 40)
            lv.append(([(px, py), (px + l * np.cos(b), py + l * np.sin(b))], r.uniform(9, 15),
                       tint(("#2f8a7a", "#3f9a84", "#4a8a5a", "#2a6a66")[r.integers(4)], r), 0.85 * k, "lens"))
        if depth < 3:
            for _ in range(int(r.integers(2, 4))):
                branch(*pts[int(r.integers(1, 5))], a + r.normal(0, 0.6), L * r.uniform(0.4, 0.6), depth + 1)
        else:
            dots.extend(([(x + r.normal(0, 5), y + r.normal(0, 5))], r.uniform(6, 10), tint(colour, r, 0.03),
                         0.95 * k, "dab") for _ in range(int(r.integers(3, 8))))
            if r.random() < 0.35:
                d = r.uniform(30, 120)
                dots.append(([(x + d * np.cos(a) + r.normal(0, 25), y + d * np.sin(a) + r.normal(0, 25))], r.uniform(5, 8),
                             tint(colour, r, 0.03), 0.8 * k, "dab"))

    branch(x, y, a, L, 0)
    pa.layer(stems)
    pa.layer(lv)
    pa.layer(dots)


def leaves(pa, items, r, k=1.0):
    """Leaves, each two swings of the side of the stick, a dark one and over half of it a lighter,
    some with the midrib drawn with the tip: items (x, y, angle, length, width, colour, light)."""
    D, S, V = [], [], []
    for x, y, a, L, wd, c, c2 in items:
        b = r.normal(0, 0.3)
        P = [(x + np.cos(a + b * j / 3) * L * j / 3, y + np.sin(a + b * j / 3) * L * j / 3) for j in range(4)]
        D.append((P, wd, tint(c, r), r.uniform(0.75, 0.95) * k, "lens"))
        if c2 is not None:
            s = r.choice((-1, 1)) * 0.2 * wd
            Q = [(px - s * np.sin(a), py + s * np.cos(a)) for px, py in P]
            S.append((Q, 0.55 * wd, tint(c2, r), r.uniform(0.5, 0.8) * k, "lens"))
        if r.random() < 0.45:
            V.append((P[:3], 2.6, tint(("#0f2a24", "#8cbf5c", "#12303a")[r.integers(3)], r), 0.55 * k, "tip"))
    pa.layer(D, grip=0.75)
    pa.layer(S, grip=0.65)
    pa.layer(V)


def stems(pa, ends, r, mouth, spread=70):
    """A stem from the mouth of the vase to each flower, bowed a little."""
    S = []
    for x, y in ends:
        mx, my = mouth[0] + r.uniform(-spread, spread), mouth[1] + 10
        bow = r.normal(0, 0.16)
        mid = ((mx + x) / 2 + (y - my) * bow, (my + y) / 2 - (x - mx) * bow)
        S.append(([(mx, my), mid, (x, y)], r.uniform(3.5, 6.0), tint(("#3f6a32", "#4c7a36", "#2f5a3a", "#5a7a3a")[r.integers(4)], r),
                  0.8, "tip"))
    pa.layer(S)


def bud(pa, P, r, show=None):
    """A poppy bud nodding on its hooked, hairy stem; `show` is the colour of the petals where it has split."""
    pa.layer([(P, 4.5, tint("#5a7438", r), 0.8, "tip")])
    C = brush.path(np.asarray(P, np.float64), 1.0)[:, :2]
    pa.layer([([C[i], C[i] + r.normal(0, 6, 2)], 1.6, tint("#4a5a30", r), 0.6, "tip") for i in r.integers(0, len(C), 26)])
    d = C[-1] - C[-12]
    d /= np.hypot(*d) + 1e-9
    e = C[-1] + d * 62
    pa.layer([([C[-1], e], 34, tint("#6f8a54", r), 0.9, "lens")], grip=0.8)
    pa.layer([([C[-1] + (5, 4), e + (4, 3)], 14, tint("#3f5a3a", r), 0.7, "lens"),
              ([C[-1] + (-5, -3), C[-1] + d * 40 + (-5, -3)], 9, tint("#a9bd84", r), 0.6, "lens")])
    if show:
        pa.layer([([C[-1] + d * 22, e + d * 6], 12, tint(show, r), 0.9, "lens")])


def fallen(pa, x, y, a, r, c=("#e8432a", "#b2192c", "#f6803c")):
    """A petal dropped on the table, lying cupped, with its shadow under it."""
    ca, sa = np.cos(a), np.sin(a)
    P = [(x, y), (x + 48 * ca - 10 * sa, y + 48 * sa + 10 * ca), (x + 98 * ca, y + 98 * sa)]
    pa.layer([([(px + 8, py + 18) for px, py in P], 66, tint("#6a4a70", r), 0.45, "veil")], grip=0.7)
    pa.layer([(P, 74, tint(c[1], r), 0.85, "petal")], grip=0.75)
    pa.layer([([(px + 6 * sa, py - 6 * ca) for px, py in P], 54, tint(c[0], r), 0.9, "petal")], grip=0.7)
    pa.layer([([(px + 22 * sa, py - 22 * ca) for px, py in P[1:]], 14, tint(c[2], r), 0.6, "lens"),
              ([(x + 14 * ca, y + 14 * sa), (x + 80 * ca + 6 * sa, y + 80 * sa - 6 * ca)], 2.2, tint("#6e0f20", r), 0.5, "tip"),
              ([(x + 12 * ca - 9 * sa, y + 12 * sa + 9 * ca), (x + 68 * ca - 18 * sa, y + 68 * sa + 18 * ca)], 2.0,
               tint("#6e0f20", r), 0.45, "tip")])


TABLE = 2270                       # about where the table would be: no edge, only a warming of the ground
VX, RIM, FOOT = 1075, 1690, 2560   # the vase: its axis at the lip, the lip of its mouth, its foot
GX, GY = 1300, 940                 # the glow: where the light sits behind the flowers
BX, BY = 1090, 1120                # the middle of the bunch


def _mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def veils(C, P0, r, pitch, wd, L, ang, p, jit=0.18, keep=1.0):
    """Strokes of the side of the stick over the whole sheet, on a loose grid so that no place is
    missed: each takes its colour from the design `C` and its pressure from `P0` where it falls,
    and its slant from `ang`(x, y)."""
    S = []
    for y in np.arange(EDGE - 120, H - EDGE + 120, pitch[1]):
        for x in np.arange(EDGE - 120, W - EDGE + 120, pitch[0]):
            if r.random() > keep:
                continue
            x1, y1 = x + r.uniform(-0.5, 0.5) * pitch[0], y + r.uniform(-0.5, 0.5) * pitch[1]
            ix, iy = int(np.clip(x1, 0, W - 1)), int(np.clip(y1, 0, H - 1))
            a, l = ang(ix, iy) + r.normal(0, jit), r.uniform(*L)
            c, s, b = np.cos(a) * l / 2, np.sin(a) * l / 2, r.normal(0, 0.05)
            S.append(([(x1 - c, y1 - s), (x1 - s * b, y1 + c * b), (x1 + c, y1 + s)], r.uniform(*wd),
                      tint(C[iy, ix], r, 0.03), p * P0[iy, ix] * r.uniform(0.8, 1.2), "veil"))
    return [S[i] for i in r.permutation(len(S))]


def ground(pa, r):
    """The field. A design in soft clouds: grey-blue and periwinkle, deepening toward the upper
    left corner; a glow of gold behind the place where the flowers will go, with rose and pale
    green in it; and below, where the table would be, a warming into terracotta, with the violet
    shadow of the vase. It goes on in veils of the side of the stick, laid lightly over one
    another and rubbed with the flat of the hand; then a second sitting of lighter sticks is
    dragged across the rubbed colour and left as it falls, catching on the tooth."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.minimum.reduce([yy - EDGE, H - EDGE - yy, xx - EDGE, W - EDGE - xx])
    clip = noise.smoothstep(0, 30, d + 12 * noise.field((H, W), 60, r) + 5 * noise.field((H, W), 8, r))
    m = [noise.fbm((H, W), s, r, octaves=3) for s in (900, 420, 260, 600, 340)]
    sm = noise.smoothstep
    glow = np.clip(np.exp(-(((xx - GX) / 600) ** 2 + ((yy - GY) / 680) ** 2)) * (1 + 0.22 * m[1]) + 0.05 * m[2], 0, 1)
    far = np.hypot((xx - GX - 500) / 1900, (yy - GY - 300) / 1700) + 0.08 * m[0]
    table = sm(TABLE - 200, TABLE + 200, yy + 90 * m[0] + 25 * m[1])
    C = np.broadcast_to(lin("#7a90ba"), (H, W, 3)).copy()
    C = _mix(C, lin("#8686c6"), 0.8 * sm(-0.4, 1.2, m[1]))                       # periwinkle clouds
    C = _mix(C, lin("#6aa4ac"), 0.7 * sm(0.1, 1.4, m[2]))                        # blue-green clouds
    C = _mix(C, lin("#3d4d96"), 0.8 * sm(0.58, 1.0, far))                        # deepening to the far corner
    C = _mix(C, lin("#ecb44c"), sm(0.14, 0.55, glow))                            # the glow: ochre gold
    C = _mix(C, lin("#f7dd86"), sm(0.5, 0.92, glow))                             # pale gold at its heart
    C = _mix(C, lin("#eaa286"), 0.4 * sm(0.3, 1.4, m[3]) * sm(0.14, 0.5, glow))      # rose in the glow
    C = _mix(C, lin("#c8d09a"), 0.4 * sm(0.4, 1.5, -m[3]) * sm(0.14, 0.5, glow))     # and pale green
    T = _mix(np.broadcast_to(lin("#b9623e"), (H, W, 3)), lin("#8f4a38"), 0.5 * sm(-0.5, 1.5, m[3]))
    T = _mix(T, lin("#d58248"), 0.45 * sm(0.2, 1.4, m[4]))
    lit = np.exp(-(((xx - VX + 420) / 420) ** 2 + ((yy - FOOT - 40) / 120) ** 2))
    T = _mix(T, lin("#dda06a"), 0.3 * lit)                                       # light on the table, left of the foot
    C = _mix(C, T, table)
    cast = np.exp(-(((xx - VX - 330) / 360) ** 2 + ((yy - FOOT + 30) / 95) ** 2))
    side = np.exp(-(((xx - VX - 300) / 190) ** 2 + ((yy - FOOT + 360) / 340) ** 2))
    shadow = np.clip(0.95 * cast + 0.5 * side, 0, 0.9)
    C = _mix(C, lin("#5f4474"), shadow)                                          # the vase's shadow, in violet
    # the sticks press lighter where the blue meets the gold, and toward the edges of the sheet,
    # and there the buff paper is left to do the work
    ring = sm(0.05, 0.2, glow) * sm(0.42, 0.2, glow) * (1 - table)
    bare = sm(0.5, 1.5, -m[4]) * (1 - sm(0.15, 0.45, glow)) * (1 - table)
    P0 = ((1 - 0.3 * ring + 0.4 * sm(0.4, 0.8, glow)) * (1 - 0.6 * bare)
          * (0.4 + 0.6 * sm(-20, 260, d + 110 * m[1] + 40 * m[2])))
    slant = lambda k, a0: (lambda ix, iy: 0.0 + 0.1 * m[k][iy, ix] if table[iy, ix] > 0.5
                           else a0 + 0.45 * m[k][iy, ix])
    S = veils(C, P0, r, (110, 260), (180, 280), (560, 900), slant(0, -1.3), 0.8)
    for k in range(3):
        pa.layer(S[k::3], clip=clip, grip=0.75)
    pa.rub(0.75 + 0.25 * sm(-1.2, 0.6, m[2]), 12)
    S = veils(C, P0, r, (120, 280), (150, 240), (460, 800), slant(3, -1.0), 0.65)
    for k in range(2):
        pa.layer(S[k::2], clip=clip, grip=0.75)
    pa.rub(0.45 + 0.5 * sm(-1.0, 0.8, m[4]), 7)
    # the second sitting: lighter sticks, dragged over the rubbed colour and left unrubbed
    C2 = _mix(C, lin("#9fc0c4"), 0.4 * sm(0.0, 1.2, m[2]) * (1 - sm(0.1, 0.4, glow)) * (1 - table))   # turquoise-grey
    C2 = _mix(C2, lin("#a898c8"), 0.4 * sm(0.3, 1.5, m[0]) * (1 - sm(0.2, 0.5, glow)))                # violet
    C2 = _mix(C2, lin("#f8ecc0"), 0.3 * sm(0.3, 0.8, glow) * (1 - table))                            # cream in the glow
    C2 = _mix(C2, lin("#eeb09a"), 0.3 * sm(0.2, 1.4, m[4]) * sm(0.14, 0.5, glow) * (1 - table))       # and rose
    C2 = _mix(C2, lin("#d98a5c"), 0.3 * table)
    S = veils(C2, P0 * (1 - 0.85 * shadow), r, (150, 300), (100, 200), (320, 660), slant(1, -1.15), 0.6, jit=0.5,
              keep=0.7)
    pa.layer(S[0::2], clip=clip, grip=0.7)
    pa.layer(S[1::2], clip=clip, grip=0.7)
    pa.ground = ndimage.gaussian_filter(pa.img, (20, 20, 0))       # the colour of the field, for losing things in it


def vase(pa, r):
    """The turquoise vase, drawn by hand. Its two sides do not quite match and it leans a hair.
    Long strokes of the side of the stick down its swelling body, deep turquoise, aqua where the
    light falls and going to blue on the other side, with ultramarine, green and violet broken
    into the glaze and a rust-red flower with dark leaves painted on its belly, half lost; the
    strokes rubbed together in places; then the light on its shoulder scumbled in pale aqua,
    its outline drawn brokenly with the tip, and its mouth filled with dark."""
    h = FOOT - RIM
    ys = np.arange(RIM - 60, FOOT + 60)
    wob = {s: 3.5 * noise.line1d(len(ys), 110, r) + 1.2 * noise.line1d(len(ys), 28, r) for s in (-1, 1)}

    def R(y, s):
        """The vase's half-width at height y, on its left (s = -1) or its right (s = 1)."""
        t = (np.asarray(y, np.float64) - RIM) / h + (0.018 if s < 0 else -0.012)
        rad = np.interp(t, [-0.02, 0.03, 0.09, 0.15, 0.25, 0.36, 0.50, 0.68, 0.86, 0.94, 0.97, 1.0],
                        [131, 125, 102, 108, 198, 250, 241, 193, 137, 123, 135, 131])
        return rad * (1.035 if s < 0 else 0.97) + np.interp(y, ys, wob[s])

    axis = lambda y: VX + 0.02 * (np.asarray(y, np.float64) - RIM)
    at = lambda u, y: axis(y) + u * R(y, -1 if u < 0 else 1)
    pt = lambda u, t: (at(u, RIM + t * h), RIM + t * h)

    yy, xx = np.mgrid[RIM - 60:FOOT + 60, 0:W].astype(np.float32)
    inside = np.minimum(xx - (axis(yy) - R(yy, -1)), axis(yy) + R(yy, 1) - xx)
    top = RIM - 22 * np.sqrt(np.clip(1 - ((xx - axis(RIM)) / 131) ** 2, 0, 1))
    bot = FOOT + 15 * np.sqrt(np.clip(1 - ((xx - axis(FOOT)) / 131) ** 2, 0, 1))
    inside = np.minimum.reduce([inside, yy - top, bot - yy]) + 2.5 * noise.field(yy.shape, 7, r)
    clip = np.zeros((H, W), np.float32)
    clip[RIM - 60:FOOT + 60] = noise.smoothstep(-2.0, 2.5, inside)

    xf = axis(FOOT)
    pa.layer([([(xf + 40, FOOT + 6), (xf + 250, FOOT - 4), (xf + 480, FOOT - 22)], 76, tint("#5a3f6e", r), 0.6, "veil"),
              ([(xf + 90, FOOT - 34), (xf + 340, FOOT - 48)], 60, tint("#6a4a7a", r), 0.4, "veil"),
              ([(xf - 110, FOOT + 15), (xf + 60, FOOT + 17), (xf + 210, FOOT + 6)], 24, tint("#3f2c50", r), 0.65, "veil")],
             grip=0.75)
    pa.smudge([(xf + 20, FOOT), (xf + 470, FOOT - 22)], 110, k=0.5, sigma=5, drag=10)

    def lane(u, t0, t1, wd, c, p, kind="side", du=0.0):
        ty = np.linspace(t0, t1, max(3, int((t1 - t0) * h / 70)))
        return ([(at(u + du * (t - t0) / (t1 - t0 + 1e-9), RIM + t * h) + r.normal(0, 2.0), RIM + t * h) for t in ty],
                wd, tint(c, r), p * r.uniform(0.9, 1.1), kind)

    def glaze(u):
        if u < -0.45:
            return ("#1b9a98", "#179094", "#22a49c")
        if u < 0.15:
            return ("#0e7f8c", "#0c7688", "#128886", "#0e7c7c")
        if u < 0.6:
            return ("#0a6078", "#095472", "#0d6874")
        return ("#084062", "#0a3458", "#073a54")

    S = []
    for u in np.linspace(-0.9, 0.9, 10) + r.normal(0, 0.02, 10):
        t, t1 = -0.03, 0.0
        while t1 < 1.0:
            t1 = min(1.0, t + r.uniform(0.3, 0.6))
            cols = glaze(u)
            S.append(lane(u, t, t1, r.uniform(70, 100), cols[r.integers(len(cols))], 1.15))
            t = t1 - r.uniform(0.03, 0.1)
    pa.layer(S[0::2], grip=0.9, clip=clip)                      # pressed in hard: the field must not show through
    pa.layer(S[1::2], grip=0.9, clip=clip)
    pa.layer([lane(u, -0.03, r.uniform(0.09, 0.17), r.uniform(50, 70), ("#0c4652", "#0b3c50", "#0f565e")[r.integers(3)],
                   0.6) for u in np.linspace(-0.8, 0.8, 5)], grip=0.7, clip=clip)     # the neck, under the flowers
    # the glaze: deeper and stranger colours broken into it, some down the form and some round it
    S = []
    for _ in range(44):
        u, t = r.uniform(-0.85, 0.85), r.uniform(0.14, 0.95)
        c = ("#2148ae", "#1f8f6c", "#56469e", "#0a5868", "#1a388c", "#2fb0a4", "#0d6e86")[r.integers(7)]
        if r.random() < 0.5:
            S.append(lane(u, t, min(1.0, t + r.uniform(0.08, 0.25)), r.uniform(26, 64), c, r.uniform(0.4, 0.65),
                          du=r.normal(0, 0.1)))
        else:
            w = r.uniform(0.15, 0.4)
            S.append(([(at(np.clip(u + w * k, -0.95, 0.95), RIM + t * h), RIM + t * h + 16 * (1 - k * k) + r.normal(0, 3))
                       for k in (-1, 0, 1)], r.uniform(36, 70), tint(c, r), r.uniform(0.35, 0.55), "veil"))
    pa.layer(S[0::2], grip=0.7, clip=clip)
    pa.layer(S[1::2], grip=0.7, clip=clip)
    S = [lane(u, 0.58, 1.0, r.uniform(40, 70), ("#0a4460", "#0c3658", "#0a4e5a")[r.integers(3)], 0.5)
         for u in r.uniform(-0.6, 0.9, 6)]                                   # the belly turns under, into shadow
    S += [lane(u, 0.78, 0.99, r.uniform(60, 90), "#b06a48", 0.22, "veil") for u in (-0.5, -0.15)]  # the table, reflected
    pa.layer(S, grip=0.7, clip=clip)
    for _ in range(7):
        u, t, du = r.uniform(-0.75, 0.8), r.uniform(0.15, 0.55), r.normal(0, 0.25)
        pa.smudge([pt(u, t), pt(np.clip(u + du, -0.85, 0.85), t + r.uniform(0.2, 0.4))], r.uniform(50, 90), k=0.55,
                  sigma=4, drag=10)
    # a flower painted on the belly under the glaze, half lost: broad soft touches, no drawing
    fx, fy = pt(0.16, 0.5)
    S = [([(fx - 30 + 150 * np.cos(a) * k, fy + 70 + 190 * np.sin(a) * k) for k in (0.25, 0.65, 1.0)], r.uniform(34, 50),
          tint(("#17337e", "#0c4a50", "#1c2f70")[r.integers(3)], r), 0.6, "lens") for a in (2.2, 1.35, 0.5, 2.9)]
    S += [([(fx + 8 * np.cos(a), fy + 8 * np.sin(a)), (fx + l * np.cos(a), fy + 0.9 * l * np.sin(a))], r.uniform(46, 66),
           tint(("#a83f2c", "#c05432", "#8a2c3c")[r.integers(3)], r), r.uniform(0.6, 0.9), "petal")
          for a, l in zip((-2.5, -1.3, -0.2, 0.9), r.uniform(45, 85, 4))]
    gx, gy = pt(-0.5, 0.72)
    S += [([(gx + 8 * np.cos(a), gy + 8 * np.sin(a)), (gx + 38 * np.cos(a), gy + 36 * np.sin(a))], r.uniform(26, 34),
           tint(("#a83f2c", "#b84c3c")[r.integers(2)], r), 0.5, "petal") for a in _spokes(r, 4, 0.3)]
    pa.layer(S, grip=0.75, clip=clip)
    pa.smudge([(fx - 110, fy + 220), (fx + 40, fy - 50)], 170, k=0.5, sigma=4.0, drag=8)
    pa.smudge([(gx - 30, gy + 40), (gx + 30, gy - 40)], 100, k=0.5, sigma=3.5, drag=7)
    # the light on the shoulder, scumbled with the side of a pale stick; the reflected light down the dark side
    S = [lane(r.uniform(-0.68, -0.34), t, t + r.uniform(0.07, 0.16), r.uniform(24, 46),
              ("#6fd0c2", "#8adaca", "#5cc6ba")[r.integers(3)], r.uniform(0.3, 0.5), du=r.normal(0, 0.06))
         for t in r.uniform(0.2, 0.5, 9)]
    S += [lane(0.86, t, t + r.uniform(0.1, 0.22), r.uniform(12, 20), "#3fa0aa", 0.35) for t in (0.3, 0.48, 0.63)]
    pa.layer(S, grip=0.55, clip=clip)
    pa.layer([lane(-0.52 + r.normal(0, 0.04), t, t + r.uniform(0.04, 0.08), r.uniform(14, 22), "#d4f2e6", 0.5,
                   du=r.normal(0, 0.05)) for t in (0.26, 0.31, 0.38)], grip=0.5, clip=clip)
    # the outline, drawn brokenly with the tip, firmer down the dark side; the foot; the mouth
    S = []
    for s, c, p, wd in ((-1, "#1b6a74", 0.5, 3.5), (1, "#0d2c4c", 0.8, 5.0), (1, "#0d2c4c", 0.5, 3.5)):
        t = r.uniform(0.02, 0.12)
        while t < 0.97:
            t1 = min(0.99, t + r.uniform(0.12, 0.38))
            off = r.normal(0, 2.5)
            S.append(([(axis(y) + s * (R(y, s) + off), y) for y in RIM + h * np.linspace(t, t1, 6)], wd,
                      tint(c, r), p * r.uniform(0.7, 1.1), "tip"))
            t = t1 + r.uniform(0.02, 0.14)
    S += [([(xf - 122, FOOT - 12), (xf, FOOT - 5), (xf + 122, FOOT - 15)], 22, tint("#0b3c54", r), 0.7, "side"),
          ([(xf + 131 * np.cos(a), FOOT - 2 + 14 * np.sin(a)) for a in np.linspace(0.25, np.pi - 0.15, 7)], 4.5,
           tint("#0c2438", r), 0.8, "tip"),
          ([(xf + 131 * np.cos(a), FOOT + 2 + 14 * np.sin(a)) for a in np.linspace(0.1, 1.3, 5)], 3.5,
           tint("#0c2438", r), 0.6, "tip")]
    pa.layer(S)
    xm = axis(RIM)
    pa.layer([([(xm - 129, RIM + 1), (xm, RIM - 1), (xm + 125, RIM + 1)], 42, tint("#0b222c", r), 0.95, "lens"),
              ([(xm - 90, RIM + 4), (xm + 80, RIM)], 26, tint("#081820", r), 0.9, "lens")], grip=0.9)
    lip = lambda a0, a1, n=6: [(xm + 130 * np.cos(a), RIM + 22 * np.sin(a) + r.normal(0, 1.2)) for a in np.linspace(a0, a1, n)]
    pa.layer([(lip(1.75, 2.95), 6, tint("#8fd8ca", r), 0.75, "tip"), (lip(0.75, 1.5), 5, tint("#4fb0aa", r), 0.6, "tip"),
              (lip(0.1, 0.6, 4), 4, tint("#237680", r), 0.6, "tip"),
              (lip(np.pi + 0.3, 2 * np.pi - 0.3, 8), 3.5, tint("#12404c", r), 0.6, "tip")])


POPPIES = dict(vermilion=("#e8432a", "#b2192c", "#f6803c", "#6e0f20"),
               scarlet=("#f05a28", "#c8302a", "#f9a050", "#8a2418"),
               coral=("#ee7a6c", "#d24a52", "#f8b0a0", "#a03040"))
ANEMONES = dict(violet=("#7048b8", "#4a2a88", "#a88ad8", "#2c1a5a", "#f0ecf2"),
                white=("#f4f0ea", "#c9c6d6", "#ffffff", "#8a88a0", None),
                red=("#dc2230", "#a01226", "#f25a4a", "#5a0c1c", "#f4f0ec"),
                blue=("#4a5ad0", "#2c3496", "#8c9ae8", "#1c2060", "#eef0f6"),
                rose=("#e76a8e", "#b83a62", "#f6a6be", "#7a1a3e", "#f6f0f2"))
MARIGOLDS = dict(orange=(("#f39a1e", "#f7b52a", "#f08218"), ("#e8661a", "#ee7a16", "#d8541a")),
                 yellow=(("#f7c22a", "#f9d23a", "#f4b02a"), ("#f19a1e", "#f5a91f", "#e88a1a")))
KEY = [  # kind, x, y, radius, turn, squash, variant, depth (0 at the back), in shadow, lost in the field
    ("poppy", 850, 1215, 225, 0.25, 0.88, "vermilion", 0.9, 0.0, 0.0),
    ("anemone", 1215, 905, 150, 0.1, 0.88, "white", 0.72, 0.0, 0.0),
    ("anemone", 1385, 1330, 165, -0.2, 0.8, "violet", 0.85, 0.3, 0.0),
    ("poppy", 1470, 660, 135, -0.35, 0.55, "scarlet", 0.4, 0.0, 0.3),
    ("poppy", 1535, 1592, 112, 0.9, 0.5, "vermilion", 0.8, 0.55, 0.0),
    ("poppy", 560, 800, 112, 0.6, 0.78, "coral", 0.25, 0.0, 0.6),
    ("anemone", 520, 1440, 108, 0.5, 0.6, "red", 0.8, 0.2, 0.25),
    ("anemone", 930, 560, 98, 0.2, 0.85, "blue", 0.2, 0.0, 0.55),
    ("anemone", 1720, 1215, 74, 0.3, 0.6, "rose", 0.5, 0.0, 0.6),
    ("anemone", 1085, 1130, 105, 0.4, 0.8, "violet", 0.45, 0.75, 0.0),
    ("marigold", 1620, 960, 96, 0.0, 0.85, "orange", 0.6, 0.0, 0.0),
    ("marigold", 842, 1612, 80, 0.4, 0.7, "orange", 0.95, 0.45, 0.0),
    ("marigold", 1178, 1452, 82, 0.0, 0.8, "yellow", 0.92, 0.1, 0.0),
    ("geranium", 400, 1150, 102, 0, 1, None, 0.5, 0.0, 0.35),
    ("daisy", 1010, 760, 70, 0.3, 0.75, None, 0.5, 0.0, 0.0),
    ("daisy", 1800, 1420, 60, -0.4, 0.7, None, 0.6, 0.0, 0.5),
    ("daisy", 480, 980, 62, 0.6, 0.8, None, 0.5, 0.3, 0.0),
    ("daisy", 1190, 1240, 56, 0.2, 0.7, None, 0.88, 0.4, 0.0),
]
SMALL = [  # kind, x, y, radius, in shadow, lost
    ("cornflower", 720, 965, 56, 0.0, 0.0), ("cornflower", 800, 1040, 50, 0.2, 0.0), ("cornflower", 690, 1045, 44, 0.0, 0.3),
    ("cornflower", 1500, 830, 54, 0.0, 0.0), ("cornflower", 1575, 890, 46, 0.0, 0.2),
    ("cornflower", 1040, 1395, 52, 0.3, 0.0), ("cornflower", 975, 1455, 44, 0.5, 0.0),
    ("cornflower", 1280, 1150, 46, 0.5, 0.0), ("cornflower", 330, 880, 40, 0.0, 0.6), ("cornflower", 1250, 640, 40, 0.0, 0.4),
    ("button", 690, 1420, 22, 0.0, 0.0), ("button", 735, 1462, 19, 0.2, 0.0), ("button", 668, 1478, 18, 0.0, 0.0),
    ("button", 1470, 1130, 22, 0.0, 0.0), ("button", 1510, 1160, 18, 0.0, 0.0), ("button", 1450, 1175, 20, 0.2, 0.0),
    ("button", 1060, 640, 20, 0.0, 0.0), ("button", 1100, 600, 17, 0.0, 0.0), ("button", 1850, 1010, 16, 0.0, 0.3),
    ("button", 1900, 1085, 14, 0.0, 0.4), ("button", 300, 1340, 15, 0.0, 0.3), ("button", 352, 1392, 13, 0.0, 0.0),
]
HEART = [(1080, 1340, 280, 255), (940, 1030, 210, 200), (1320, 1100, 210, 215), (740, 1340, 170, 155),
         (1420, 1480, 165, 120), (1075, 1610, 150, 100)]      # the dark of the bunch: rounds of foliage in shadow
HAZE = [(740, 960, 400, 440, ("#3f8f96", "#4a86a8", "#3a7a80", "#5aa39a", "#417f92"), 26),      # behind the left of it, cool
        (560, 1380, 260, 220, ("#3f8f96", "#4a86a8", "#5aa39a"), 9),
        (1570, 1160, 300, 380, ("#dc9a52", "#d98a6a", "#e2a85a", "#c9785a"), 14)]              # behind the right, warm


def _in(blobs, r):
    """A point in one of the rounds `blobs` [(x, y, rx, ry, ...)], and how far out in it (1 at the rim)."""
    b = blobs[r.choice(len(blobs), p=np.array([q[2] * q[3] for q in blobs], float) / sum(q[2] * q[3] for q in blobs))]
    a, q = r.uniform(0, 2 * np.pi), np.sqrt(r.uniform(0, 1))
    return b[0] + q * b[2] * np.cos(a), b[1] + q * b[3] * np.sin(a), q


def lose(pa, x, y, R, r, amount):
    """Draw the field back over the far side of a thing and rub it in, so that it goes back into the ground."""
    a = np.arctan2(y - BY, x - BX)
    S = []
    for da in (-0.8, 0.0, 0.8):
        b = a + da + r.normal(0, 0.2)
        cx, cy = x + 0.6 * R * np.cos(b), y + 0.6 * R * np.sin(b)
        tx, ty = -np.sin(b) * 0.8 * R, np.cos(b) * 0.8 * R
        S.append(([(cx - tx, cy - ty), (cx + tx, cy + ty)], 0.9 * R,
                  tint(pa.ground[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))], r, 0.03), 0.35 + 0.5 * amount, "veil"))
    pa.layer(S, grip=0.7)
    tx, ty = -np.sin(a) * 0.9 * R, np.cos(a) * 0.9 * R
    pa.smudge([(x - tx, y - ty), (x + tx, y + ty)], 1.3 * R, k=0.25 + 0.4 * amount, sigma=3.0, drag=6)


def flower(pa, r, kind, x, y, R, turn, squash, variant, depth, shade, lost):
    """One flower. One in shadow is drawn with deeper sticks; one going back into the field is
    laid lightly, and then the field's own colour is drawn over its far side and rubbed in."""
    k = (0.82 + 0.18 * depth) * (1 - 0.3 * lost)
    h = Head(x, y, R, turn, squash)
    pa.shade = shade
    if kind == "poppy":
        poppy(pa, h, r, *POPPIES[variant], k=k)
    elif kind == "anemone":
        anemone(pa, h, r, *ANEMONES[variant], k=k)
    elif kind == "cornflower":
        cornflower(pa, h, r, k=k)
    elif kind == "marigold":
        marigold(pa, h, r, *MARIGOLDS[variant], k=k)
    elif kind == "daisy":
        daisy(pa, h, r, k=k)
    elif kind == "geranium":
        geranium(pa, x, y, R, r, k=k)
    else:
        button(pa, x, y, R, r, k=k)
    pa.shade = 0.0
    if lost:
        lose(pa, x, y, R, r, lost)


def bouquet(pa, r):
    """The flowers. First a haze of colour behind the bunch, cool on the left and warm on the right,
    rubbed into the field; the stems; over them the dark of the foliage at its heart, rubbed soft
    and drawn into again; the sprays, the larkspur and the buds that escape from it; and then
    leaves and flowers in turn from the back of the bunch to the front, each flower in broad veils
    of the side of the stick and then drawn over with the tip. The ones at the back and at the
    edges are given back to the field. Last, two petals that have dropped on the table."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rnd = lambda bl: np.maximum.reduce([noise.smoothstep(1.15, 0.45, np.hypot((xx - b[0]) / b[2], (yy - b[1]) / b[3]))
                                        for b in bl])
    to = lambda x, y: np.arctan2(y - RIM, x - VX)             # the way a stem runs here, out from the mouth
    S = []
    for b in HAZE:
        for _ in range(b[5]):
            x, y, q = _in([b], r)
            a, l = to(x, y) + r.normal(0, 0.6), r.uniform(260, 520)
            c, s = np.cos(a) * l / 2, np.sin(a) * l / 2
            S.append(([(x - c, y - s), (x + r.normal(0, 20), y + r.normal(0, 20)), (x + c, y + s)], r.uniform(110, 190),
                      tint(b[4][r.integers(len(b[4]))], r), r.uniform(0.4, 0.6) * (1.2 - 0.5 * q), "veil"))
    S = [S[i] for i in r.permutation(len(S))]
    pa.layer(S[0::2], grip=0.75)
    pa.layer(S[1::2], grip=0.75)
    pa.rub(0.85 * rnd(HAZE), 9)
    pa.ground = ndimage.gaussian_filter(pa.img, (20, 20, 0))
    fl = [list(f) for f in KEY] + [[kind, x, y, R, r.uniform(-0.6, 0.6), r.uniform(0.6, 0.95), None, r.uniform(0.3, 0.8),
                                    shade, lost] for kind, x, y, R, shade, lost in SMALL]
    stems(pa, [(f[1], f[2]) for f in fl], r, (VX, RIM))

    def darks(n, wd, L, p, deep, pale):
        S = []
        for _ in range(n):
            x, y, q = _in(HEART, r)
            a, l = to(x, y) + r.normal(0, 0.5), r.uniform(*L)
            c, s = np.cos(a) * l / 2, np.sin(a) * l / 2
            cols = deep if q < 0.6 else pale
            S.append(([(x - c, y - s), (x + r.normal(0, 10), y + r.normal(0, 10)), (x + c, y + s)], r.uniform(*wd),
                      tint(cols[r.integers(len(cols))], r), r.uniform(*p), "lens"))
        pa.layer(S[0::2], grip=0.75)
        pa.layer(S[1::2], grip=0.75)

    darks(190, (60, 110), (130, 260), (0.75, 0.95), ("#10262a", "#14301f", "#151d3c", "#0d1c22"),
          ("#1f4a3a", "#1d4650", "#34481f", "#243560"))
    pa.rub(0.5 * noise.smoothstep(0.3, 0.9, rnd(HEART)), 4, wear=0.15)
    darks(70, (22, 48), (110, 240), (0.55, 0.8), ("#1c3f34", "#1a3c48", "#2c4420", "#1f2c58", "#0c1a20"),
          ("#2a5a48", "#24525c", "#44602a", "#16332e"))                    # and drawn into again, unrubbed
    pa.layer([(P, wd, tint("#4a6e34", r), 0.7, "lens") for P, wd in (                          # blades of grass
        ([(1000, 1320), (760, 900), (560, 620), (430, 500)], 13), ([(1240, 1420), (1560, 1290), (1820, 1300), (1990, 1390)], 12),
        ([(1120, 1300), (1150, 900), (1120, 600), (1060, 420)], 11))])
    larkspur(pa, [(790, 1030), (690, 790), (615, 570), (565, 400)], 44, r, k=0.6, pale=True)
    lose(pa, 615, 545, 250, r, 0.7)
    spray(pa, 850, 880, -2.25, 520, r)
    spray(pa, 1560, 1080, -0.3, 460, r)
    spray(pa, 1480, 1470, 0.45, 340, r, colour="#f3e6b0")
    spray(pa, 640, 1420, 2.6, 300, r, colour="#f1c9c4")
    larkspur(pa, [(1400, 1380), (1500, 1080), (1610, 760), (1730, 380)], 54, r)
    bud(pa, [(1150, 800), (1215, 640), (1272, 525), (1322, 480), (1352, 506)], r, show="#e8432a")
    bud(pa, [(700, 1420), (565, 1362), (440, 1335), (372, 1372), (354, 1425)], r)
    D, Lt = ("#1d4a44", "#24584a", "#2a6a66", "#2f5a2a", "#16332e"), ("#4f9a78", "#6f9a50", "#3f8a84", None, None)
    lv = []
    for x, y, angs, L, z in ((1090, 740, (-2.1, -1.6, -1.2), (140, 210), 0.3),
                             (640, 1240, (2.7, 3.0, 3.4), (160, 220), 0.55), (1490, 1430, (0.0, 0.4, 0.8), (150, 210), 0.6),
                             (1570, 1170, (-0.35, 0.1), (130, 180), 0.4), (770, 770, (-2.5, -2.05), (130, 190), 0.2)):
        lv += [(z + r.uniform(-0.1, 0.1), (x + r.normal(0, 30), y + r.normal(0, 30), a + r.normal(0, 0.08), r.uniform(*L),
                                           r.uniform(40, 58), D[r.integers(5)], Lt[r.integers(5)])) for a in angs]
    for _ in range(34):
        x, y, q = _in(HEART, r)
        if q > 0.45:
            lv.append((r.uniform(0, 0.9), (x, y, to(x, y) + r.normal(0, 0.5), r.uniform(90, 170), r.uniform(34, 56),
                                           D[r.integers(5)], Lt[r.integers(5)] if q > 0.8 else None)))
    fl.sort(key=lambda f: f[7])
    for lo, hi in ((-1, 0.35), (0.35, 0.65), (0.65, 2)):
        leaves(pa, [it for z, it in lv if lo <= z < hi], r)
        for f in fl:
            if lo <= f[7] < hi:
                flower(pa, r, *f)
    leaves(pa, [(930, 1650, 2.3, 200, 54, D[0], Lt[0]), (1010, 1672, 1.9, 150, 46, D[4], None),
                (1225, 1662, 0.9, 190, 52, D[1], Lt[2]), (1150, 1678, 1.3, 140, 44, D[3], None)], r)
    # leaves only outlined, out in the haze
    S = []
    for _ in range(12):
        x, y, q = _in(HAZE, r)
        a, l, w = r.uniform(0, 2 * np.pi), r.uniform(50, 90), r.uniform(0.25, 0.4)
        e = (x + l * np.cos(a), y + l * np.sin(a))
        for s in (-1, 1):
            S.append(([(x, y), (x + l / 2 * np.cos(a) - s * w * l * np.sin(a), y + l / 2 * np.sin(a) + s * w * l * np.cos(a)), e],
                      2.4, tint(("#7a5a4a", "#2f5a66", "#8a6a8a")[r.integers(3)], r), 0.5, "tip"))
    pa.layer(S)
    fallen(pa, 585, 2588, 0.25, r)
    fallen(pa, 762, 2652, -0.7, r, c=("#f05a28", "#c8302a", "#f9a050"))


def paint(seed=1910):
    r = noise.rng(seed)
    sheet = sheet_of(r)
    pa = Pastel(sheet, r)
    ground(pa, r)
    vase(pa, r)
    bouquet(pa, r)
    return plate.mount(pa.finish(sheet), sheet)


if __name__ == "__main__":      # a light stroke catches on the grains and leaves hollows bare; a hard one fills them
    r = noise.rng(0)
    sh = sheet_of(r, (200, 400), edge=0)
    pa = Pastel(sh, r)
    pa.layer([([(20, 60), (380, 60)], 60, "#2040c0", 0.35, "side"), ([(20, 140), (380, 140)], 60, "#2040c0", 1.2, "side")])
    blue = (pa.img[..., 2] - pa.img[..., 0]) > 0.15
    light, hard = blue[50:70, 60:340].mean(), blue[130:150, 60:340].mean()
    assert 0.1 < light < 0.8 < hard, (light, hard)
    print("light stroke covers", round(float(light), 2), "hard stroke covers", round(float(hard), 2))
