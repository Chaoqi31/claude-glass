"""A Pool in the Hills, Noon. Acrylic on cotton duck.

In 1964 David Hockney went to live in Los Angeles, and for the next few years he painted its swimming pools
again and again: blue water under a blank sky, a low house with walls of glass, a palm or two, a diving
board. He painted them in the new acrylic paint, which dries in minutes and lies flat and bright, often on
canvas left bare in a wide border round the picture, like the white edge of a snapshot. The flat parts went
on fast with a roller or a broad brush against masking tape. He took his time over the water: its light he
drew in lines of white and pale blue with a small brush, and the splash in A Bigger Splash, he said, took two
weeks of small brushes and little lines.

This picture is a pool on a terrace in the hills above Hollywood at midday, seen from one end, so that its
side runs away across the canvas at an angle. The border was taped off on raw cotton duck. The sky went on
first with a roller, in a few passes; then the palms, the house, the deck and the water, each in a flat coat
against tape, the copings left as bare canvas between them. Over the dry turquoise went long ragged shapes
of cerulean, and of ultramarine toward the far end, cut in by hand; then the light on the water, drawn with
a round brush in white: wavy lines running down the pool, each about half a swing out of step with the next,
so that they touch and part and close into rounded loops, a stroke lifted every swing or two and the next
set down a little on, the brush pressed into each turn; larger and looser near, smaller and closer where the
pool runs away, with a line of pale blue or lemon beside some of them, and broken rings round the splash.
The board went on next, run out from the deck over the water, and last the splash the diver left: a sheet
of water thrown up from a narrow foot and falling back, laid first in broad thin white so that the blue
shows through, then streaked up with a small brush to a ragged crown, with a thinner veil thrown out to one
side, a few loose streaks and drops, and the water broken white along its foot.
"""

import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import brush, canvas, dabs, noise
from atelier.color import lin

TITLE = "A Pool in the Hills, Noon"
DATE = "2026"
MEDIUM = "Acrylic on cotton duck, rolled and brushed flat against tape, the border and the copings left bare"
AFTER = ("David Hockney, the Los Angeles swimming pools in acrylic on canvas: Picture of a Hollywood Swimming Pool "
         "(1964), Peter Getting Out of Nick's Pool (1966), The Splash, A Little Splash and A Bigger Splash "
         "(1966–67), Portrait of an Artist (Pool with Two Figures) (1972)")
ROOM = "The Garden"
YEAR = 1967
PLACE = "Hollywood Hills, Los Angeles"
REGION = "Europe"
NOTE = ("A pool on a terrace above Hollywood at noon, seen at an angle, with no one in it: a diving board run out "
        "over the water, the splash the diver left, and the light on the water drawn in loose lines of white.")

H, W = 3000, 3000
TOP, BOTTOM, LEFT, RIGHT = 200, 2700, 210, 2790      # the picture, inside the bare border
LIGHT = (-0.55, -0.6, 0.6)
RAW = "#dccdb0"                                      # unprimed cotton duck

# The pool is drawn in one-point perspective: a point X m across and Z m away on the water, `up` m above it,
# lands at (XC + F X / Z, YH + F (EYE - up) / Z). The painter stands on the terrace, EYE m above the water.
XC, YH, F, EYE = 1135.0, 560.0, 2140.0, 4.0
SLOPE = (2540 - XC) / (BOTTOM - YH)                  # the right coping runs at X = SLOPE * EYE
X_END = SLOPE * EYE
X_OUT = X_END + 0.2                                  # outer edge of the coping

ROOF, SOFFIT, FACADE, BACK = 1040, 1068, 1104, 1440  # roof edge, its shadow, the wall and glass, the deck behind
COPE, SHADE, WET = 1512, 1534, 1572                  # far coping (bare canvas), the shaded band, open water
HOUSE_END = 1985
BOARD = dict(z=(5.0, 5.55), up=0.75, thick=0.08, tip=-0.1)    # run out from the deck on the right
SUN = 0.13                                           # how far beyond a thing 1 m up the noon sun throws its shadow
SPLASH = (-0.95, 5.3)                                # where the diver went in, (X, Z)

SKY = "#93d1ec"
FASCIA = "#f6f4ec"
UNDER = "#a59dbd"
GLASS = "#7b8ba7"
MULLION = "#eeebe3"
WALL = "#f0d27c"
DECK = "#ebc9bb"
WATER = "#3ab4c9"
BAND = "#2a55b0"
WHITE = "#f5faf6"


def project(X, Z, up=0.0):
    return XC + F * np.asarray(X) / Z, YH + F * (EYE - up) / np.asarray(Z)


def depth(y):
    """How far away the water lies at canvas height y, in m."""
    return F * EYE / (y - YH)


def left_edge(Z):
    """The X of the left edge of the picture at depth Z."""
    return (LEFT - XC) * Z / F


def fresh(seed, name):
    """Each passage draws on chances of its own, so the others stay as they are when one is changed."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def swell(u, power=1.0):
    """sin(pi u)**power, safe at the ends: what swells from nothing and goes back to nothing."""
    return np.clip(np.sin(np.pi * np.clip(u, 0, 1)), 0, 1) ** power


# ---------------------------------------------------------------------------------------------------------------
# flat colour

def edges(poly):
    """The edges of a convex polygon as (a point on it, its inward normal)."""
    P = np.asarray(poly, np.float64)
    c = P.mean(0)
    out = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        t = (q - p) / np.hypot(*(q - p))
        n = np.array([-t[1], t[0]])
        out.append((p, n if np.dot(c - p, n) > 0 else -n))
    return out


def box(poly, pad=10):
    P = np.asarray(poly, np.float64)
    y0, x0 = np.floor(P.min(0)[::-1] - pad).astype(int)
    y1, x1 = np.ceil(P.max(0)[::-1] + pad).astype(int)
    return slice(max(y0, 0), min(y1, H)), slice(max(x0, 0), min(x1, W))


def coat(poly, cuts, r):
    """Where a flat coat over the convex polygon lies, in its box: -> (slices, coverage). Each edge was
    either taped ('tape': dead straight, but the paint crept a px or two under the tape along the threads)
    or cut in by hand with the brush ('hand': it wavers, a little at a time)."""
    sl = box(poly)
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    yy, xx = np.mgrid[sl[0], sl[1]].astype(np.float32)
    wob = {}
    if "tape" in cuts:
        wob["tape"] = 0.45 * noise.field((h, w), 1.3, r) + 1.6 * np.clip(noise.field((h, w), 4.0, r) - 1.5, 0, None)
    if "hand" in cuts:
        wob["hand"] = 2.4 * noise.field((h, w), 120, r) + 0.9 * noise.field((h, w), 14, r) \
            + 0.35 * noise.field((h, w), 2.0, r)
    d = np.full((h, w), np.inf, np.float32)
    for (p, n), cut in zip(edges(poly), cuts):
        d = np.minimum(d, (xx - p[0]) * n[0] + (yy - p[1]) * n[1] + wob[cut])
    return sl, noise.smoothstep(-0.8, 0.8, d)


def blot(poly, r, wobble=3.0):
    """Where a patch cut in by hand round any polygon lies, in its box: -> (slices, coverage)."""
    sl = box(poly, 14)
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if h < 4 or w < 4:
        return sl, np.zeros((max(h, 0), max(w, 0)), np.float32)
    im = Image.new("F", (w, h), 0.0)
    ImageDraw.Draw(im).polygon([(x - sl[1].start, y - sl[0].start) for x, y in poly], fill=1.0)
    m = ndimage.gaussian_filter(np.asarray(im), 2.5)
    j = wobble / 10 * (0.7 * noise.field((h, w), 30, r) + 0.3 * noise.field((h, w), 6, r))
    return sl, noise.smoothstep(0.42, 0.58, m + j)


def rolled(shape, r, band=210.0):
    """A coat put on with a roller in passes about `band` px wide across the canvas: each a little thinner
    as the roller empties, the laps a little thicker, a ridge at the edge of each pass, and all over the
    fine stipple the nap leaves. -> (tone about 1, relief)"""
    h, w = shape
    k = np.zeros((h, w), np.float32)
    ridge = np.zeros((h, w), np.float32)
    u = np.arange(h, dtype=np.float32)[:, None]
    p = -r.uniform(0, band)
    while p < h:
        off = 5 * noise.line1d(w, 400, r)[None] + p
        bw = band * r.uniform(0.95, 1.05)
        prof = noise.smoothstep(-2, 2, u - off) * noise.smoothstep(-2, 2, off + bw - u)
        load = 1 - r.uniform(0.1, 0.3) * np.linspace(0, 1, w)[None] ** 1.3 + 0.04 * noise.line1d(w, 300, r)[None]
        k += prof * load
        ridge += np.exp(-((u - off) / 1.6) ** 2) + np.exp(-((u - off - bw) / 1.6) ** 2)
        p += bw * r.uniform(0.72, 0.9)
    stipple = 0.65 * noise.field(shape, 1.4, r) + 0.35 * noise.field(shape, 4.0, r)
    tone = 1 + 0.018 * np.tanh(k - 1.2) + 0.012 * stipple
    return tone.astype(np.float32), (0.12 * ridge + 0.06 * stipple).astype(np.float32)


def brushed(shape, r, along=0):
    """A coat laid with a broad flat brush, its strokes running across (along=0) or down (along=1):
    the streaks of the bristles, and each stroke a shade thicker or thinner. -> (tone, relief)"""
    h, w = shape
    if along:
        hair, lay_ = noise.stretched(shape, 1.6, 70, r), noise.stretched(shape, 50, 260, r)
    else:
        hair, lay_ = noise.stretched((w, h), 1.6, 70, r).T, noise.stretched((w, h), 50, 260, r).T
    tone = 1 + 0.011 * hair + 0.014 * lay_ + 0.006 * noise.field(shape, 3, r)
    return tone.astype(np.float32), (0.06 * hair).astype(np.float32)


class Canvas:
    """The picture as it stands: colour in linear light, the height of the surface, and the bare duck."""

    def __init__(self, seed):
        self.seed = seed
        self.duck = canvas.duck((H, W), seed, tint=RAW, thread=3.1)
        self.img = self.duck.color.astype(np.float32).copy()
        self.height = 0.35 * self.duck.tooth
        sl, a = coat([(LEFT, TOP), (RIGHT, TOP), (RIGHT, BOTTOM), (LEFT, BOTTOM)], ["tape"] * 4, fresh(seed, "tape"))
        self.frame = np.zeros((H, W), np.float32)    # inside the tape round the border
        self.frame[sl] = a
        self.grit = noise.field((H, W), 1.2, fresh(seed, "grit"))

    def lay(self, sl, a, colour, tone, relief):
        """Lay a coat of colour with coverage `a` over the slices; thin paint sits a little less on the high
        threads of the weave than in the hollows."""
        a = a * self.frame[sl]
        t = self.duck.tooth[sl]
        col = lin(colour)[None, None] * (tone * (1.012 - 0.024 * t))[..., None]
        img = self.img[sl]
        img += (col - img) * a[..., None]
        hgt = self.height[sl]
        hgt += a * (0.2 * t + relief - 0.6 * hgt)   # the film fills the weave a little, and what lay beneath
        return a

    def flat(self, name, poly, cuts, colour, how="roll", hide=0.985, **kw):
        """A flat coat of one colour over the convex polygon. -> (slices, coverage)"""
        r = fresh(self.seed, name)
        sl, a = coat(poly, cuts, r)
        tone, relief = rolled(a.shape, r, **kw) if how == "roll" else brushed(a.shape, r, **kw)
        return sl, self.lay(sl, a * hide, colour, tone, relief)

    def patch(self, name, poly, colour, hide=0.95, mask=None, wobble=3.0):
        """A patch of colour brushed in by hand over any polygon. -> (slices, coverage)"""
        r = fresh(self.seed, name)
        sl, a = blot(poly, r, wobble)
        if a.size == 0:
            return sl, a
        if mask is not None:
            a = a * mask[sl]
        tone, relief = brushed(a.shape, r)
        return sl, self.lay(sl, a * hide, colour, tone, relief)

    def lines(self, strokes, colour, mask=None, height=0.3, edge=0.25):
        """Lines of one colour from a round brush. Each stroke is an (n, 4) array of rows x, y, half-width
        in px and load (0..1): how much paint the brush left there. The line is as wide as the brush was
        pressed, its edge catches on the threads of the canvas, and where little paint was left it is thin
        enough for what lies beneath to show, and breaks up over the weave. In place."""
        S = np.zeros((H, W), np.float32)
        Lw = np.zeros((H, W), np.float32)
        r = fresh(self.seed, "lines " + colour)
        buf = []

        def flush():
            if not buf:
                return
            X, Y, V, A = (np.concatenate(c) for c in zip(*buf))
            ok = (X > -20) & (X < W + 20) & (Y > -20) & (Y < H + 20)
            if ok.any():
                for acc, val in ((S, V), (Lw, V * A)):
                    b, o = brush.splat(X[ok], Y[ok], val[ok].astype(np.float32))
                    brush.paste(acc, b, o)
            buf.clear()

        count = 0
        for P in strokes:
            if len(P) < 2:
                continue
            C = brush.path(np.asarray(P, np.float64), 0.5)
            if len(C) < 3:
                continue
            x, y, w, a = C.T
            g = np.gradient(C[:, :2], axis=0)
            g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
            wm = max(float(w.max()), 0.4)
            lanes = max(2, int(np.ceil(2.4 * wm / 0.5)))
            u = np.linspace(-1, 1, lanes)[:, None]
            hair = np.clip(1 + 0.3 * r.standard_normal((lanes, 1)), 0.3, 1.8)    # the bristles carry unevenly
            ww = np.maximum(w, 0.2)[None]
            X = x[None] + (-g[:, 1])[None] * u * ww
            Y = y[None] + g[:, 0][None] * u * ww
            V = np.broadcast_to(0.5 * 2 * ww / lanes, X.shape) * hair
            A = np.broadcast_to(np.clip(a, 0, 1)[None], X.shape) * (0.75 + 0.25 * hair)
            buf.append((X.ravel(), Y.ravel(), V.ravel(), A.ravel()))
            count += X.size
            if count > 4_000_000:
                flush()
                count = 0
        flush()
        S = ndimage.gaussian_filter(S, 0.65)
        Lw = ndimage.gaussian_filter(Lw, 0.65)
        c = 1 - np.exp(-2.4 * S)                    # where the brush touched
        t = self.duck.tooth - 0.5
        m = noise.smoothstep(0.3, 0.62, c + edge * t + 0.07 * self.grit)
        load = np.clip(Lw / np.maximum(S, 1e-4), 0, 1)
        m *= noise.smoothstep(0.06, 0.26, load + 0.2 * t + 0.06 * self.grit)   # a dry brush skips the hollows
        alpha = m * (1 - np.exp(-3.2 * Lw)) * self.frame                       # paint over paint builds up
        if mask is not None:
            alpha *= mask
        ys, xs = np.nonzero(alpha.max(1) > 1e-3)[0], np.nonzero(alpha.max(0) > 1e-3)[0]
        if not len(ys):
            return
        sl = slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1)
        a = alpha[sl]
        self.img[sl] += (lin(colour) * (1.01 - 0.02 * self.duck.tooth[sl])[..., None] - self.img[sl]) * a[..., None]
        self.height[sl] += a * (height - 0.5 * (self.height[sl] - 0.35 * self.duck.tooth[sl]))


def stroke(x, y, w, a):
    return np.stack(np.broadcast_arrays(x, y, w, a), 1).astype(np.float64)


def pressure(n, r, start=0.85, lift=0.2, wobble=0.18, scale=30):
    """How hard the brush was pressed along a stroke of n points: set down, varying, lifted at the end."""
    p = np.clip(0.85 + wobble * noise.line1d(n, scale, r), 0.4, 1.15)
    m = max(2, n // 14)
    p[:m] = np.linspace(start, p[m], m)
    f = max(2, int(n * lift))
    p[-f:] *= np.linspace(1, 0.1, f) ** 0.8
    return p


# ---------------------------------------------------------------------------------------------------------------
# the picture: sky, palms, house, deck

def sky(cv):
    cv.flat("sky", [(LEFT, TOP), (RIGHT, TOP), (RIGHT, BACK + 10), (LEFT, BACK + 10)], ["tape"] * 4, SKY, band=230)


def leaf(r, x0, y0, ang, L, wd, droop, n=12):
    """One tapered stroke from (x0, y0) toward `ang`, L px long and `wd` px across at its widest, bowing
    down by `droop` (a fraction of its length) toward its tip."""
    s = np.linspace(0, 1, n)
    bend = droop * L * s ** 2
    x = x0 + L * s * np.cos(ang)
    y = y0 + L * s * np.sin(ang) + bend
    w = wd / 2 * (0.25 + 0.75 * swell(0.12 + 0.88 * s, 0.7)) * np.clip(1.15 - 0.95 * s ** 1.5, 0, 1)
    return stroke(x, y, w, np.clip(1.05 - 0.4 * s ** 3, 0, 1) * r.uniform(0.85, 1.0))


def palm(r, base, top, R, bow, fronds=(15, 20), dead=3, droop=1.0, reach=1.0):
    """One palm: a tall thin trunk, its shadowed side, and the fronds thrown out from the crown, each a
    sheaf of tapered strokes that arch out and fall at the tip, `droop` times as far and `reach` times as
    long as most; `dead` of them hanging brown below."""
    (bx, by), (tx, ty) = base, top
    s = np.linspace(0, 1, 12)
    x = bx + (tx - bx) * s + bow * np.sin(np.pi * s)
    y = by + (ty - by) * s
    w = 0.05 * R * (1 - 0.35 * s) * (1 + 0.15 * s ** 8)
    trunk = [stroke(x, y, w, 1.0)]
    side = [stroke(x + 0.55 * w, y, 0.4 * w, 0.85)[:-1]]
    darks, mids, lights, gone = [], [], [], []
    k = int(r.integers(*fronds))
    for i in range(k):
        phi = r.uniform(0, 2 * np.pi)
        if i < dead:                                 # the old fronds hang down under the crown
            ang = np.pi / 2 + r.normal(0, 0.35)
            gone.append(leaf(r, tx + r.normal(0, 6), ty + 10, ang, R * r.uniform(0.35, 0.6), R * 0.09, 0.05))
            continue
        el = r.uniform(-0.55, 0.75)                  # how far it rises toward or away from the eye
        ang = np.arctan2(-np.sin(el) * 0.9 - 0.1, np.cos(phi) * np.cos(el) + 1e-3)
        L = R * reach * r.uniform(0.7, 1.1) * (0.75 + 0.25 * abs(np.cos(phi)))
        bend = (0.25 + 0.3 * np.cos(el) * abs(np.cos(ang))) * droop
        for j in range(int(r.integers(3, 6))):
            a = ang + r.normal(0, 0.14) + 0.2 * (j - 2) * r.uniform(0.3, 1)
            darks.append(leaf(r, tx, ty - 6, a, L * r.uniform(0.75, 1.05), R * r.uniform(0.07, 0.11),
                              bend * r.uniform(0.8, 1.3)))
        if r.random() < 0.8:
            out = (mids if r.random() < 0.55 else lights)
            out.append(leaf(r, tx, ty - 9, ang + r.normal(0, 0.06), L * r.uniform(0.5, 0.8), R * r.uniform(0.04, 0.07),
                            bend))
    return trunk, side, gone, darks, mids, lights


def palms(cv, which):
    """The palms: on the right a tall one leaning a little toward the house, its small stiff crown over a
    skirt of dead fronds, and a shorter one leaning away with long fronds that arch and hang; on the left
    one more over the roof."""
    r = fresh(cv.seed, "palms " + which)
    stands = {"right": [((2215, BACK + 30), (2148, 340), 150, 18, dict(fronds=(21, 25), dead=6, droop=0.4, reach=0.85)),
                        ((2580, BACK + 30), (2630, 800), 185, -16, dict(fronds=(12, 16), dead=1, droop=1.35,
                                                                        reach=1.05))],
              "left": [((470, ROOF + 40), (452, 735), 150, 6, {})]}[which]
    layers = [[] for _ in range(6)]
    for base, top, R, bow, kind in stands:
        for lay_, more in zip(layers, palm(r, base, top, R, bow, **kind)):
            lay_ += more
    for lay_, col in zip(layers, ("#a29068", "#7c6a48", "#9b7b46", "#22552f", "#3a8a45", "#93c65a")):
        cv.lines(lay_, col, height=0.2)


def house(cv):
    cv.flat("fascia", [(LEFT, ROOF), (2045, ROOF), (2045, SOFFIT), (LEFT, SOFFIT)], ["tape"] * 4, FASCIA)
    cv.flat("soffit", [(LEFT, SOFFIT), (HOUSE_END, SOFFIT), (HOUSE_END, FACADE), (LEFT, FACADE)], ["tape"] * 4,
            UNDER, how="brush")
    cv.flat("glass", [(LEFT, FACADE), (1180, FACADE), (1180, BACK), (LEFT, BACK)], ["tape"] * 4, GLASS, how="brush",
            along=1)
    glass(cv)
    cv.flat("wall", [(1180, FACADE), (HOUSE_END, FACADE), (HOUSE_END, BACK), (1180, BACK)], ["tape"] * 4, WALL)
    for i, x in enumerate((540, 862, 1172)):
        cv.flat(f"mullion{i}", [(x, FACADE), (x + 13, FACADE), (x + 13, BACK), (x, BACK)], ["tape"] * 4, MULLION,
                how="brush", along=1)


def glass(cv):
    """What the glass gives back and what shows through it: the bright sky in its upper part, a curtain
    drawn to one side behind the first pane, the dark of a room, and a few streaks of light across it."""
    r = fresh(cv.seed, "glass")
    cv.flat("glass sky", [(LEFT, FACADE), (1180, FACADE), (1180, 1204), (LEFT, 1226)],
            ["tape", "tape", "hand", "tape"], "#97a8c3", how="brush", along=1)
    cv.flat("curtain", [(LEFT, FACADE + 4), (322, FACADE + 4), (338, BACK), (LEFT, BACK)], ["tape"] * 4, "#f0e2b4",
            how="brush", along=1)
    folds = []
    for x in np.linspace(LEFT + 22, 312, 5) + r.normal(0, 5, 5):
        yy = np.linspace(FACADE + 8, BACK - 4, 3)
        folds.append(stroke(x + np.array([0, r.normal(0, 1.5), r.normal(3, 2)]), yy, r.uniform(7, 12),
                            r.uniform(0.3, 0.45)))
    cv.lines(folds, "#d7bf86", height=0.05)
    cv.flat("room", [(338, 1250), (1180, 1240), (1180, BACK), (338, BACK)], ["hand", "tape", "tape", "tape"],
            "#5e6d88", how="brush", along=1)
    pane = np.zeros((H, W), np.float32)
    pane[FACADE:BACK, 330:1180] = 1
    streaks = []
    for i in range(12):
        x0, y0, L = r.uniform(300, 1150), r.uniform(FACADE + 60, BACK), r.uniform(70, 230)
        a = np.deg2rad(r.uniform(-60, -50))
        s = np.linspace(0, 1, 6)
        streaks.append(stroke(x0 + L * s * np.cos(a), y0 + L * s * np.sin(a), r.uniform(1.5, 3.0) * (1.1 - 0.6 * s),
                              0.55 * (1 - 0.5 * s)))
    cv.lines(streaks, "#d3dbe8", mask=pane, height=0.1)


def deck(cv):
    cv.flat("deck back", [(LEFT, BACK), (RIGHT, BACK), (RIGHT, COPE), (LEFT, COPE)], ["tape"] * 4, DECK, band=200)
    x0 = XC + (X_OUT / EYE) * (COPE - YH)
    x1 = XC + (X_OUT / EYE) * (BOTTOM - YH)
    cv.flat("deck side", [(x0, COPE - 30), (RIGHT, COPE - 30), (RIGHT, BOTTOM), (x1, BOTTOM)], ["tape"] * 4, DECK,
            band=200)


def plants(cv):
    """Two clumps of agave at the end of the house: tapered strokes of grey-green thrown up from the
    ground, the lighter ones laid over the darker."""
    r = fresh(cv.seed, "plants")
    dark, light = [], []
    for cx, R, k in ((1938, 150, 15), (1795, 95, 10)):
        for i in range(k):
            ang = -np.pi / 2 + r.normal(0, 0.62)
            L = R * r.uniform(0.55, 1.0) * (1 - 0.3 * abs(np.sin(ang + np.pi / 2)))
            (dark if r.random() < 0.6 else light).append(
                leaf(r, cx + r.normal(0, 10), BACK + 18, ang, L, r.uniform(16, 26), -0.04))
    cv.lines(dark, "#3f7a52", height=0.3)
    cv.lines(light, "#7fae7a", height=0.3)


# ---------------------------------------------------------------------------------------------------------------
# the water

def water(cv):
    pts = [(LEFT, SHADE), (XC + SLOPE * (SHADE - YH), SHADE), (XC + SLOPE * (BOTTOM - YH), BOTTOM), (LEFT, BOTTOM)]
    sl, a = cv.flat("water", pts, ["tape"] * 4, WATER, how="brush")
    mask = np.zeros((H, W), np.float32)
    mask[sl] = a
    x0, x1 = XC + SLOPE * (SHADE - YH), XC + SLOPE * (WET - YH)
    cv.flat("band", [(LEFT, SHADE), (x0, SHADE), (x1, WET), (LEFT, WET)], ["tape", "tape", "hand", "tape"], BAND,
            how="brush")
    mask[:WET + 3] = 0
    return mask


def threads(r):
    """The lines of the light as they lie on the water: wavy lines running down the pool toward the
    house, each swinging from side to side about its own course and about half a swing out of step with
    its neighbours, so that they touch and part and close into rounded loops; where one swings harder it
    crosses its neighbour, and now and then it curls round on itself. A swing is a little shorter far off.
    -> list of (n, 4) rows: X, Z, the phase of the swing, and how far along the line it has come"""
    gap, m = 0.17, 1500                             # m between the lines; points along each
    zn, zf = depth(BOTTOM) - 0.15, depth(WET) + 0.05
    xa, xb = left_edge(zf) - 0.3, X_END + 0.25
    n = int((xb - xa) / gap) + 1
    Z = np.linspace(zn, zf, m)
    lam = 0.40 * (Z / 4.0) ** -0.3                  # m from one swing to the next
    turn = 2 * np.pi * np.concatenate([[0.0], np.cumsum(np.diff(Z) / lam[1:])])
    step = noise.stretched((m, 8 * n), 48, 320, r)  # how far out of step with its neighbours, changing slowly
    reach = noise.stretched((m, 8 * n), 36, 200, r)
    drift = noise.stretched((m, 8 * n), 48, 500, r)
    idx = np.arange(m)
    out = []
    for i in range(n):
        own, curl, hand, wander, swing = (noise.line1d(m, s, r) for s in (240, 90, 18, 300, 70))
        th = turn + np.pi * i + 1.5 * step[:, 8 * i] + 0.2 * own + r.uniform(-0.1, 0.1)
        A = gap * np.clip(0.47 + 0.1 * reach[:, 8 * i] + 0.1 * swing, 0.15, 0.85)
        X = xa + gap * (i + 0.05 * r.standard_normal()) + gap * (0.2 * drift[:, 8 * i] + 0.1 * wander) \
            + A * np.sin(th) + 0.005 * hand
        Zc = Z + lam / (2 * np.pi) * 0.25 * curl * np.cos(th)    # the hand hurries one side of a swing
        # now and then, partway along a swing, the line runs round a small loop and on
        c = np.floor(th / np.pi)
        for ic in np.flatnonzero(np.diff(c)) + 1:
            if 14 < ic < m - 14 and r.random() < 0.04:
                psi = 2 * np.pi * noise.smoothstep(ic - 12, ic + 12, idx)
                rho = r.uniform(0.035, 0.05)
                X = X + r.choice((-1.0, 1.0)) * rho * (1 - np.cos(psi))
                Zc = Zc + rho * np.sin(psi)
        out.append(np.stack([X, Zc, th, turn], 1))
    return out


def zone(Z):
    """Width of the brush, in px, for a line at depth Z: a small brush near, a smaller one far."""
    return 5.0 * 4.0 / np.maximum(Z, 2.0)


def deeps(cv, mask):
    """The deeper water, brushed in over the turquoise before the light went over it: long ragged shapes
    of cerulean lying across the pool, as the ripples gather, more of them further off, and at the far
    end ultramarine, which runs together there. Each is cut in by hand with the brush, flat."""
    r = fresh(cv.seed, "deeps")
    y0, y1 = WET - 4, BOTTOM + 4
    x0, x1 = LEFT - 4, int(XC + SLOPE * (BOTTOM - YH)) + 6
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    Z = F * EYE / (yy - YH)
    X = (xx - XC) * Z / F
    # the shapes are laid out on the water itself, long across the pool and short in depth
    za, zb, xa, xb = 3.5, depth(WET) + 0.3, left_edge(depth(WET)) - 0.3, X_END + 0.3
    gz, gx = int((zb - za) / 0.02), int((xb - xa) / 0.02)
    long_ = noise.stretched((gx, gz), 12, 55, r).T
    fine = noise.stretched((gx, gz), 4, 14, r).T
    big = noise.field((gz, gx), 70, r)
    iz, ix = (Z - za) / 0.02, (X - xa) / 0.02
    look = lambda g: ndimage.map_coordinates(g, [iz, ix], order=1, mode="nearest")
    far = np.clip((Z - depth(BOTTOM)) / (depth(WET) - depth(BOTTOM)), 0, 1)
    f = look(long_) + 0.35 * look(fine) + 0.5 * look(big) + 2.4 * far ** 1.5
    tone, relief = brushed(f.shape, r)
    sl = slice(y0, y1), slice(x0, x1)
    for lo, colour, hide in ((1.25, "#2c8fcd", 0.93), (2.35, "#2b62bc", 0.95)):
        a = noise.smoothstep(lo - 0.04, lo + 0.04, f) * mask[sl]
        cv.lay(sl, ndimage.gaussian_filter(a, 0.7) * hide, colour, tone, relief)


def clear_of_splash(X, Z, room=0.6):
    return np.hypot(X - SPLASH[0], Z - SPLASH[1]) > room


def light(cv, mask):
    """The light on the water, drawn freely in white with a round brush along the lines of threads(): a
    stroke runs for a swing or two or three and is lifted, and the next is set down just past it, or a
    little short of it and to one side, or further on; the brush is pressed into each turn and eased along
    the runs between. Beside some lines, on one side of them, a thinner one in pale blue, or in lemon
    where the sun is strongest. Round the splash, broken rings."""
    r = fresh(cv.seed, "light")
    white, pale, lemon = [], [], []
    zn, zf = depth(BOTTOM) - 0.3, depth(WET) + 0.1
    xa, xb = left_edge(zf) - 0.5, X_END + 0.5
    sun = noise.field((60, 90), 11, r)              # where the light on the water is strong, and where it fails
    for T in threads(fresh(cv.seed, "threads")):
        side = r.choice((-1.0, 1.0))
        j = int(np.searchsorted(T[:, 3], r.uniform(0, 2.5)))
        while j < len(T) - 12:
            e = min(len(T), int(np.searchsorted(T[:, 3], T[j, 3] + r.uniform(1.2, 3.4) * 2 * np.pi)))
            X, Z, th = T[j:e, 0] + r.normal(0, 0.01), T[j:e, 1], T[j:e, 2]
            u = r.random()
            dt = r.uniform(0.1, 0.35) if u < 0.55 else r.uniform(-0.6, -0.15) if u < 0.85 else r.uniform(1.0, 3.0)
            j = max(j + 12, int(np.searchsorted(T[:, 3], T[e - 1, 3] + dt)))
            n = len(X)
            if n < 12:
                continue
            cz = Z.mean()
            g = sun[int(np.clip((cz - zn) / (zf - zn), 0, 0.999) * 60),
                    int(np.clip((X.mean() - xa) / (xb - xa), 0, 0.999) * 90)]
            if r.random() < 0.02 + 0.15 * noise.smoothstep(0.4, 1.8, -g):    # not drawn: the loops run together
                continue
            x, y = project(X, Z)
            p = pressure(n, r, wobble=0.25, scale=25)
            w = zone(Z) * p * (0.78 + 0.4 * np.sin(th) ** 2) * np.clip(np.exp(0.35 * g + r.normal(0, 0.15)), 0.45, 1.7)
            load = np.clip(0.75 + 0.35 * p, 0, 1) * r.uniform(0.8, 1.0)
            ok = (x > LEFT - 40) & (y > WET - 12) & clear_of_splash(X, Z)
            into = pale if (r.random() < 0.05 or g < -1.2 + 0.4 * r.random()) else white
            for run in np.split(np.arange(n), np.flatnonzero(np.diff(ok.astype(int))) + 1):
                if ok[run[0]] and len(run) > 4:
                    into.append(stroke(x[run], y[run], w[run], load[run]))
            if r.random() < 0.4 and n > 30:
                i0 = int(r.integers(0, n // 2))
                i1 = min(n, i0 + int(r.integers(n // 3, n)))
                if ok[i0:i1].all():
                    gx, gy = np.gradient(x[i0:i1]), np.gradient(y[i0:i1])
                    q = side * 1.8 * w[i0:i1] / (np.hypot(gx, gy) + 1e-9)
                    tgt = lemon if (r.random() < 0.4 and cz < 6.5) else pale
                    tgt.append(stroke(x[i0:i1] - gy * q, y[i0:i1] + gx * q, 0.55 * w[i0:i1],
                                      0.9 * pressure(i1 - i0, r)))
    # the water the splash set moving: broken rings spreading round where the diver went in
    ex, ez = SPLASH
    for rho in (0.42, 0.72, 1.02):
        a = r.uniform(0, 2 * np.pi)
        end = a + 2 * np.pi
        while a < end - 0.3:
            b = min(end, a + r.uniform(0.5, 1.4))
            t = np.linspace(a, b, 40)
            rr = rho * (1 + 0.04 * noise.line1d(40, 10, r)) + 0.01 * np.sin(t * r.uniform(10, 18) + r.uniform(0, 6))
            X, Z = ex + rr * np.cos(t), ez + rr * np.sin(t)
            x, y = project(X, Z)
            (white if r.random() < 0.75 else pale).append(stroke(x, y, 0.7 * zone(Z) * pressure(40, r), 0.9))
            a = b + r.uniform(0.2, 0.7)
    cv.lines(pale, "#a7e2f2", mask=mask, height=0.15)
    cv.lines(lemon, "#eef3a2", mask=mask, height=0.15)
    cv.lines(white, WHITE, mask=mask, height=0.2)


def board(cv):
    """The diving board, run out from the deck on the right over the water: its top, the edge turned to
    the painter in shadow, the stand it is bolted to, and its shadow on the deck."""
    (zn, zf), up, th, tip = BOARD["z"], BOARD["up"], BOARD["thick"], BOARD["tip"]
    lean = SUN * (up - 0.25)
    yn0, yf0 = project(0, zn + lean, 0.25)[1], project(0, zf + lean, 0.25)[1]
    out = lambda y: XC + (X_OUT / EYE) * (y - YH) + 4
    cv.flat("board shadow", [(out(yf0), yf0), (RIGHT, yf0), (RIGHT, yn0), (out(yn0), yn0)], ["hand"] * 4,
            "#c99fae", how="brush")
    (xn, yn), (xf, yf) = project(tip, zn, up), project(tip, zf, up)
    yb = project(tip, zn, up - th)[1]
    sx = project(3.55, zn, up)[0]
    yd = project(0, zn + 0.12, 0.25)[1]
    cv.flat("stand", [(sx, yb - 6), (RIGHT, yb - 6), (RIGHT, yd), (sx, yd)], ["tape"] * 4, "#9aa3b6", how="brush",
            along=1)
    cv.flat("stand side", [(sx, yb - 6), (sx + 26, yb - 6), (sx + 26, yd), (sx, yd)], ["tape"] * 4, "#7b8397",
            how="brush", along=1)
    cv.flat("board", [(xf, yf), (RIGHT, yf), (RIGHT, yn), (xn, yn)], ["tape"] * 4, "#ecd56b", how="brush")
    cv.flat("board edge", [(xn, yn - 1), (RIGHT, yn - 1), (RIGHT, yb), (xn, yb)], ["tape"] * 4, "#bf9e44",
            how="brush")


def splash(cv):
    """The splash, a moment after the dive: a sheet of water thrown up from a narrow foot in a few plumes
    side by side, the middle one tallest, each opening as it rises and falling back over at the top in
    ragged fingers; every part of it on the path a thrown thing takes. First broad strokes of thin white,
    so that the blue shows through, for the body of the sheet, and thinner still for a veil thrown out to
    the left; then short streaks of a small brush up through it, closer toward the top; the fingers of the
    crown, full of paint where the water turns over to fall; a few loose streaks; drops; and the water
    broken white along its foot."""
    r = fresh(cv.seed, "splash")
    ex, ez = SPLASH
    x0, y0 = project(ex, ez)
    k = F / ez                                      # px to the m, in the plane of the splash
    #         where on the foot, lean, m high, share of the strokes
    plumes = [(-0.15, 0.0, 1.45, 0.45), (0.45, 0.08, 1.05, 0.25), (-0.6, -0.1, 1.15, 0.3)]
    body, veil, sheet, blue, drops, foot = [], [], [], [], [], []

    def rise(u, h, d, s0, s1, n=28):
        """A strand from the foot at u (-1..1 across it), thrown h m high at d from upright, from s0 to s1
        (1 at the top of its rise, more as it falls back). -> (x, y, s)"""
        s = np.linspace(s0, s1, n)
        sway = r.uniform(0, 0.03) * np.sin(2 * np.pi * r.uniform(1.0, 2.5) * s + r.uniform(0, 6))
        return x0 + k * (0.22 * u + 2 * h * np.tan(d) * s + sway), y0 - k * h * (2 * s - s * s), s

    def throw():
        """Where a strand leaves the foot, its slant from upright and its height, in one of the plumes."""
        uc, lean, tall, _ = plumes[r.choice(len(plumes), p=[q[3] for q in plumes])]
        v = r.normal(0, 1)
        return float(np.clip(uc + r.normal(0, 0.3), -1, 1)), lean + 0.085 * v, \
            tall * (1 - 0.12 * v * v) * r.uniform(0.8, 1.04)

    def drop(x, y, size):
        drops.append(stroke([x, x + r.normal(0, 1)], [y, y + r.uniform(0.5, 3)], [size, 0.7 * size], [1.0, 0.8]))

    for i in range(80):                             # the body of the sheet, broad and thin
        u, d, h = throw()
        s0, s1 = r.uniform(0, 0.1), r.uniform(0.85, 1.15)
        x, y, s = rise(u, h * r.uniform(0.85, 1.0), d, s0, s1)
        f = (s - s0) / (s1 - s0)
        body.append(stroke(x, y, k * r.uniform(0.03, 0.06) * (0.45 + 0.55 * swell(0.1 + 0.8 * f, 0.6)),
                           r.uniform(0.2, 0.36) * (1 - 0.4 * f)))
    for i in range(14):                             # the veil thrown out to the left, thinner still
        x, y, s = rise(r.uniform(-1.0, -0.6), r.uniform(0.8, 1.05), -0.35 + r.normal(0, 0.05), r.uniform(0.05, 0.2),
                       r.uniform(1.0, 1.3))
        f = (s - s[0]) / (s[-1] - s[0])
        veil.append(stroke(x, y, k * r.uniform(0.025, 0.045) * swell(0.05 + 0.9 * f, 0.4), r.uniform(0.15, 0.25)))
        if i < 5:                                   # and its edge
            sheet.append(stroke(x + r.normal(0, 4), y, r.uniform(0.6, 1.1), 0.5 * (1 - 0.5 * f)))
    for i in range(320):                            # streaks up through the sheet
        u, d, h = throw()
        s0 = r.uniform(0, 1) ** 0.7 * 0.95
        s1 = min(s0 + r.uniform(0.12, 0.55), 1.3)
        x, y, s = rise(u, h, d, s0, s1, 16)
        f = (s - s0) / (s1 - s0)
        wide = r.uniform(2.2, 4.0) if r.random() < 0.2 else r.uniform(0.7, 1.8)
        (blue if r.random() < 0.2 else sheet).append(stroke(x, y, wide * (0.6 + 0.4 * swell(f, 0.5)),
                                                            r.uniform(0.45, 0.95) * pressure(16, r)))
    for uc, lean, tall, share in plumes:            # the crown: fingers of water turning over to fall
        for _ in range(int(20 * share)):
            u, d, h = float(np.clip(uc + r.normal(0, 0.3), -1, 1)), lean + r.normal(0, 0.08), tall * r.uniform(0.75, 1.05)
            for _ in range(int(r.integers(3, 7))):
                s0, s1 = r.uniform(0.6, 0.85), r.uniform(1.25, 1.55) if r.random() < 0.4 else r.uniform(0.98, 1.2)
                dd = d + r.normal(0, 0.025)
                x, y, s = rise(u + r.normal(0, 0.03), h * r.uniform(0.95, 1.03), dd, s0, s1, 16)
                f = (s - s0) / (s1 - s0)
                sheet.append(stroke(x, y, r.uniform(1.0, 3.2) * (0.5 + 0.7 * f ** 2), r.uniform(0.7, 1.0)))
                if r.random() < 0.4:                # drops flung on from the tip
                    for _ in range(int(r.integers(1, 3))):
                        sd = s1 + r.uniform(0.05, 0.3)
                        xx, yy, _ = rise(u, h, dd + r.normal(0, 0.03), sd, sd, 1)
                        drop(xx[0], yy[0], r.uniform(1.2, 3.4))
    for i in range(6):                              # loose streaks, thrown higher
        u, d, h = throw()
        x, y, s = rise(u, h * r.uniform(1.08, 1.25), d + r.normal(0, 0.06), r.uniform(0.3, 0.55),
                       r.uniform(0.95, 1.1))
        sheet.append(stroke(x, y, r.uniform(0.8, 1.3), 0.85 * (1 - 0.5 * s ** 4)))
    for i in range(45):                             # spray over the crown
        u, d, h = throw()
        sd = r.uniform(0.8, 1.5)
        xx, yy, _ = rise(u, h * r.uniform(0.95, 1.2), d, sd, sd, 1)
        drop(xx[0] + r.normal(0, 10), yy[0] + r.normal(0, 8), r.uniform(1.0, 2.8))
    for i in range(110):                            # broken water along the foot, scribbled across
        dX, dZ = r.normal(0, 0.17), r.normal(0, 0.05)
        L = r.uniform(0.04, 0.16)
        t = np.linspace(-0.5, 0.5, 10)
        x, y = project(ex + dX + L * t, ez + dZ + 0.012 * np.sin(t * r.uniform(4, 9) + r.uniform(0, 6)))
        foot.append(stroke(x, y - k * 0.015, r.uniform(1.5, 4.0) * swell(t + 0.5, 0.4) + 0.4, r.uniform(0.6, 1.0)))
    for i in range(140):                            # and where the sheet leaves the water, short and close
        u, d, h = throw()
        x, y, s = rise(u, h * r.uniform(0.3, 0.6), d, 0.0, r.uniform(0.06, 0.22), 8)
        foot.append(stroke(x, y, r.uniform(1.2, 3.0) * (1 - 0.5 * s / s[-1]), r.uniform(0.5, 0.95)))
    cv.lines(veil, "#d8eff3", height=0.05)
    cv.lines(body, "#e2f3f5", height=0.08)
    cv.lines(foot, WHITE, height=0.12)
    cv.lines(sheet, WHITE, height=0.15)
    cv.lines(blue, "#bfe6ef", height=0.1)
    cv.lines(drops, WHITE, height=0.15)


def paint(seed=1967):
    cv = Canvas(seed)
    sky(cv)
    palms(cv, "right")
    palms(cv, "left")
    house(cv)
    deck(cv)
    plants(cv)
    mask = water(cv)
    deeps(cv, mask)
    light(cv, mask)
    board(cv)
    splash(cv)
    h = ndimage.gaussian_filter(cv.height, 0.7)
    return dabs.shine(cv.img, h, light=LIGHT, relief=0.6, gloss=0.006, reach=(0.9, 1.08))


if __name__ == "__main__":
    assert abs(depth(project(0.0, 6.0)[1]) - 6.0) < 1e-6, "depth undoes the projection"
    xs = [project(X_END, z)[0] - (XC + SLOPE * (project(X_END, z)[1] - YH)) for z in (4.5, 7.0)]
    assert max(map(abs, xs)) < 1e-6, "the right coping is a line of constant X"
    sl, a = coat([(10, 10), (60, 10), (60, 40), (10, 40)], ["tape"] * 4, noise.rng(0))
    assert a[25 - sl[0].start, 35 - sl[1].start] > 0.99 and a[0, 0] < 0.01, "a coat covers inside, nothing outside"
    T = threads(noise.rng(1))
    d = np.array([b[:, 0] - a[:, 0] for a, b in zip(T, T[1:])])
    assert (d.min(1) < 0.04).mean() > 0.8 and (d.max(1) > 0.25).mean() > 0.8, "neighbouring lines touch and part"
    print("ok")
