"""The Caliph's Garden by Moonlight. Watercolour and body colour over pen and ink; the colour plate tipped onto a page.

Edmund Dulac came to London from Toulouse in 1904, and for ten years Hodder and Stoughton gave him a
book of pictures every Christmas: the Arabian Nights in 1907, The Sleeping Beauty and Other Fairy
Tales in 1910, Hans Andersen in 1911. He drew each picture in pencil and fine pen, then built the
colour in watercolour, layer over layer, washing the sheet down under the tap between layers so that
the colour came out deep and luminous at once, with a fine grain of the paper's tooth through it. He
loved the night, in his own blue, set off with a little gold and rose, and from Persian and Japanese
painting he took flat silhouettes and patterned masses. The pictures were printed in three colours
and each plate was tipped onto a heavier cream page.

This is the garden of the Caliph's palace in Baghdad, as the Arabian Nights tell of it: the palace of
domes and minarets across the garden at night, its windows lit, a fountain on its axis, cypresses, a
stone pine leaning in from the left and a rose bush with a brass lantern hung in it. Everything was
drawn first, freehand, with a fine nib. The sky went on in four washes of Prussian blue and
ultramarine, each let dry and washed down before the next, so the colour sits in the hollows of the
paper and the tops of its grain show pale; round the moon it was lifted with a damp brush. The palace
is the paper under a pale wash of cobalt that pooled against its edges, its windows washed with
gamboge. The lawns are three washes: a clear blue green, then ultramarine and Prussian blue, which
granulate, carried deeper towards us and under the terrace, and last the long shadows of the
cypresses. The walk is cobalt with a little rose. The roses are rose madder and crimson, open, half
open and in bud; round the lantern the dark was carried up thin and gold run into it. The water of the
fountain, the flowers in the grass and the stars went on last in Chinese white.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from atelier import brush, noise, pencil, plate, wash
from atelier import watercolour as wc
from atelier.color import lin, pigment
from atelier.etching import hand
from atelier.paper import Sheet

TITLE = "The Caliph's Garden by Moonlight"
DATE = "2026"
MEDIUM = "Watercolour and body colour over pen and ink; the colour plate tipped onto a cream page"
AFTER = ("Edmund Dulac, the fairy-tale illustrations of 1907–1916 in watercolour over pen and ink: Stories from the "
         "Arabian Nights (1907); The Sleeping Beauty and Other Fairy Tales (1910); Stories from Hans Andersen (1911)")
ROOM = "Paper and Water"
YEAR = 1910
PLACE = "London"
REGION = "Europe"
NOTE = ("A palace out of the Arabian Nights across its garden at night, the windows lit, with a fountain on the axis, "
        "cypresses, a leaning pine and a rose bush with a brass lantern glowing in it. Every line was drawn freehand "
        "with a fine nib; the blue of the sky is four washes, each washed down before the next.")

H, W = 2900, 2200                       # the page on the wall
PAGE = (84, 72, 2116, 2832)             # the page, x0 y0 x1 y1
PX, PY, PW, PH = 236, 206, 1728, 2360   # the plate tipped onto it
YY, XX = np.mgrid[0:PH, 0:PW].astype(np.float32)

AX, HZ = 812.0, 1000.0                  # the garden's axis, and the height of the eye
MOON = (1180.0, 400.0, 70.0)
YB, YW, YP = 1168.0, 905.0, 840.0       # the palace: its foot, the cornice of the wings, of the pavilion
YT, YF, YS = 1200.0, 1232.0, 1336.0     # the terrace: the parapet's top, its foot, the foot of the steps
BASIN = (AX, 2000.0, 470.0, 170.0)      # the fountain's basin: centre and radii of its rim
BOWL = 1846.0                           # the height of the fountain's bowl
LAMP = (1440.0, 1428.0)                 # the lantern among the roses

PALETTE = dict(   # the box, as absorbance at unit density
    prussian=pigment("#2a6496"), ultramarine=pigment("#4a56d0"), cobalt=pigment("#5b8ee0"),
    indigo=pigment("#3b4766"), viridian=pigment("#3f9a88"), sap=pigment("#7a9848"),
    rose=pigment("#e0708c"), crimson=pigment("#c2385a"), gamboge=pigment("#f0ca48"),
    yellow_ochre=pigment("#e0b264"), raw_sienna=pigment("#cf9342"), burnt_sienna=pigment("#a65a36"),
    sepia=pigment("#6f5a4a"), ink=pigment("#3a302b"))
GRAIN = dict(prussian=0.12, ultramarine=0.5, cobalt=0.4, indigo=0.25, viridian=0.3, sap=0.15, rose=0.05,
             crimson=0.05, gamboge=0.0, yellow_ochre=0.3, raw_sienna=0.25, burnt_sienna=0.35, sepia=0.4, ink=0.0)
LIFT = dict(prussian=0.6, ultramarine=0.9, cobalt=0.85, indigo=0.7, viridian=0.6, sap=0.6, rose=0.5, crimson=0.4,
            gamboge=0.6, yellow_ochre=0.8, raw_sienna=0.7, burnt_sienna=0.7, sepia=0.7, ink=0.0)
WARM = ("gamboge", "raw_sienna", "yellow_ochre", "rose", "crimson")
STONE = dict(cobalt=0.78, prussian=0.08, rose=0.1, yellow_ochre=0.04)     # moonlit stone
SHADE = dict(ultramarine=0.42, prussian=0.28, indigo=0.2, rose=0.1)       # the shadows of the night
GOLD = dict(gamboge=0.62, raw_sienna=0.28, rose=0.1)                      # lamplight
LEAF = dict(indigo=0.42, prussian=0.33, sap=0.15, sepia=0.1)              # the dark of the trees

TRUNK = [(30, 2440, 80), (84, 2210, 72), (62, 1960, 67), (104, 1700, 62), (84, 1450, 58), (142, 1210, 54),
         (132, 990, 48), (182, 830, 44), (230, 712, 40)]
LIMBS = [[(230, 718, 38), (296, 592, 31), (386, 470, 25), (510, 392, 19), (660, 336, 13), (800, 300, 8)],
         [(222, 730, 34), (176, 590, 27), (150, 452, 21), (112, 330, 15), (64, 236, 9)],
         [(140, 1000, 26), (232, 928, 20), (330, 884, 15), (430, 866, 9)]]
CROWNS = [(120, 168, 200, 80), (392, 116, 230, 78), (650, 212, 196, 68), (250, 356, 170, 66), (520, 318, 118, 52),
          (64, 490, 130, 58), (862, 286, 120, 48), (436, 842, 92, 38)]


def _s(r):
    return int(r.integers(1 << 31))


def fill(polys, blur=0.7, discs=()):
    """Polygons (and discs (x, y, radius)) in plate px, filled, as a 0..1 mask."""
    im = Image.new("L", (PW, PH), 0)
    d = ImageDraw.Draw(im)
    for P in polys:
        d.polygon([(float(x), float(y)) for x, y in P], fill=255)
    for x, y, q in discs:
        d.ellipse((x - q, y - q, x + q, y + q), fill=255)
    return ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, blur)


def rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.float64)


def ellipse(cx, cy, rx, ry, a0=0.0, a1=2 * np.pi, n=96):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def runs(ok):
    """(start, stop) of each run of True in a 1-D array."""
    e = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(ok, np.int8), [0]])))
    return list(zip(e[::2], e[1::2]))


def along(P, a, b, step=2.0):
    """The points of the polyline P from arc length a to b, every `step` px."""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    t = np.append(np.arange(a, b, step), b)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)


def arch(cx, foot, w, h, point=0.22, n=18):
    """An opening with a pointed head standing on y=foot, w wide and h tall to its apex: two arcs, each
    struck from a centre `point` of the half-width beyond the middle, meeting at the apex."""
    rr = w / 2
    d = point * rr
    R = rr + d
    top = np.sqrt(R * R - d * d)
    spring = foot - h + top
    phi = np.linspace(np.pi, np.arccos(-d / R), n)
    left = np.stack([cx + d + R * np.cos(phi), spring - R * np.sin(phi)], 1)
    right = np.stack([2 * cx - left[::-1, 0], left[::-1, 1]], 1)
    return np.vstack([[(cx - rr, foot)], left, right[1:], [(cx + rr, foot)]])


ONION = PchipInterpolator([0, 0.1, 0.28, 0.48, 0.66, 0.8, 0.9, 1.0], [0.9, 1.1, 1.2, 1.06, 0.72, 0.36, 0.14, 0.0])


def onion(cx, base, w, h, n=48, f=1.0):
    """An onion dome on a drum w wide at y=base, h tall: it swells past the drum, turns in, and draws up
    to a point. With f < 1, one of the ribs that run up it, that fraction of the way out."""
    t = np.linspace(0, 1, n)
    rad, y = w / 2 * ONION(t), base - t * h
    if f != 1.0:
        return np.stack([cx + f * rad, y], 1)
    return np.vstack([np.stack([cx - rad, y], 1), np.stack([cx + rad[::-1], y[::-1]], 1)])


def urn(cx, foot, h):
    """A garden urn on its plinth: foot, swelling body, a lip."""
    prof = PchipInterpolator([0, 0.12, 0.2, 0.35, 0.6, 0.82, 0.9, 1.0], [0.42, 0.42, 0.22, 0.5, 0.58, 0.4, 0.55, 0.6])
    t = np.linspace(0, 1, 30)
    rad, y = h * 0.5 * prof(t), foot - t * h
    return np.vstack([np.stack([cx - rad, y], 1), np.stack([cx + rad[::-1], y[::-1]], 1)])


def leaf(x, y, L, wd, a, n=12):
    """A pointed leaf from its stalk at (x, y), L long along `a`, wd across, fullest a little below the middle."""
    t = np.linspace(0, 1, n)
    half = 0.5 * wd * np.sin(np.pi * t ** 0.85) ** 0.8
    c, s = np.cos(a), np.sin(a)
    px = np.concatenate([t * L, t[::-1] * L])
    py = np.concatenate([half, -half[::-1]])
    return np.stack([x + c * px - s * py, y + s * px + c * py], 1)


def ribbon(rows, r, step=3.0, knots=0.07):
    """A limb along control rows (x, y, width): -> (outline polygon, one edge, the other, centre, widths)."""
    C = brush.path(np.asarray(rows, np.float64), step)
    xy = C[:, :2]
    w = C[:, 2] * (1 + knots * noise.line1d(len(C), 12, r) + 0.5 * knots * noise.line1d(len(C), 3, r))
    g = np.gradient(xy, axis=0)
    g /= np.hypot(*g.T)[:, None] + 1e-9
    n = np.stack([-g[:, 1], g[:, 0]], 1) * (w / 2)[:, None]
    return np.vstack([xy + n, (xy - n)[::-1]]), xy + n, xy - n, xy, w


class Box:
    """The paint box. Transparent washes multiply, so the order they went on does not change the colour:
    each is kept as a density per pigment and glazed onto the paper at the end. The granulating pigments
    settle into the hollows of the tooth; washing the sheet down lifts the loose pigment off its hills."""

    def __init__(self, sheet, r):
        self.sheet, self.r = sheet, r
        self.d = {k: np.zeros((PH, PW), np.float32) for k in PALETTE}
        self.soak = np.zeros((PH, PW), np.float32)
        self.valley = (1.6 * (0.5 - ndimage.gaussian_filter(sheet.tooth, 0.8))
                       + 0.2 * noise.field((PH, PW), 1.6, r)).astype(np.float32)
        self.hill = noise.smoothstep(0.4, 0.85, sheet.tooth).astype(np.float32)
        # the heavy pigment does not settle grain by grain but in small clots, a few grains across
        clot = noise.smoothstep(-0.2, 1.0, noise.field((PH, PW), 2.2, r) + 0.6 * noise.field((PH, PW), 6, r))
        self.settled = (np.clip(self.valley + 0.25, 0, None) * (0.4 + 0.9 * clot)).astype(np.float32)
        self.strokes, self.opaque = {}, []

    def lay(self, density, **mix):
        for k, f in mix.items():
            g = GRAIN[k]
            self.d[k] += f * density * (np.clip(1 + g * self.valley, 0.15, None) if g else 1.0)

    def wash(self, mask, ragged=1.0, **kw):
        """A wash laid on dry paper over `mask`: (where its water lay, the pigment it left)."""
        ys, xs = np.flatnonzero(mask.max(1) > 0.02), np.flatnonzero(mask.max(0) > 0.02)
        wet, dep = np.zeros((PH, PW), np.float32), np.zeros((PH, PW), np.float32)
        if len(ys):
            sl = (slice(max(0, ys[0] - 24), ys[-1] + 25), slice(max(0, xs[0] - 24), xs[-1] + 25))
            sheet = Sheet(*(a[sl] for a in (self.sheet.color, self.sheet.tooth, self.sheet.fiber, self.sheet.alpha)))
            wet[sl] = wc.puddle(mask[sl], sheet, self.r, ragged)
            if wet[sl].max() > 0.02:
                dep[sl] = wc.deposit(wet[sl], sheet, self.r, grain=0.0, **kw)
            self.soak[sl] = np.maximum(self.soak[sl], wet[sl])
        return wet, dep

    def settle(self, dep, amount, **mix):
        """The granulating part of a wash: what sank into the hollows of the tooth and clotted there."""
        self.lay(dep * self.settled * amount, **mix)

    def scrub(self, amount, where):
        """The sheet washed down under the tap with a soft brush: the loose pigment comes off the hills of
        the tooth, the stain in its hollows stays."""
        f = amount * where * (0.45 + 0.55 * self.hill)
        for k, d in self.d.items():
            if LIFT[k]:
                d *= 1 - LIFT[k] * f

    def stroke(self, key, rows, radius, **kw):
        ink, water = self.strokes.setdefault(key, (np.zeros((PH, PW), np.float32), np.zeros((PH, PW), np.float32)))
        kw = dict(dict(load=1.0, clumps=5, splay=0.1, head=0.5, bristles=int(np.clip(12 * radius, 24, 160))), **kw)
        brush.stroke(self.sheet, rows, radius, seed=_s(self.r), into=(ink, water), **kw)

    def take(self, key):
        return self.strokes.pop(key)[0]

    def banded(self, y0, y1, x0, x1, radius, key="band"):
        """The streaks a big brush leaves going side to side down a wash: 0..~1."""
        y = y0 - radius
        while y < y1 + radius:
            R = radius * self.r.uniform(0.8, 1.2)
            self.stroke(key, [(x0 - 120, y + self.r.normal(0, 6), 0.9), ((x0 + x1) / 2, y + self.r.normal(0, 10), 1.0),
                              (x1 + 120, y + self.r.normal(0, 6), 0.85)], R, load=1.0, reach=9000, dryness=0.5, clumps=7)
            y += R * self.r.uniform(1.0, 1.25)
        return ndimage.gaussian_filter(np.clip(self.take(key), 0, 1.3), 2)

    def body(self, alpha, colour):
        """Body colour: Chinese white in the paint, which covers what is under it instead of glazing it."""
        self.opaque.append((np.clip(alpha, 0, 1), lin(colour) if isinstance(colour, str) else colour))

    def glaze(self):
        img = self.sheet.color.copy()
        for k, kk in PALETTE.items():
            img *= np.exp(-self.d[k][..., None] * kk)
        for a, c in self.opaque:
            img = img * (1 - a[..., None]) + c * a[..., None]
        return img


class Pen:
    """The drawing, made before any colour, freehand with a fine steel nib. A long line goes down in
    lengths of a few centimetres, each set down a hair before or after the end of the last; the nib
    spreads as the hand presses and runs to a hair where it lifts, and a long straight line bows a
    little, as one drawn without a ruler does. A figure of straight sides is drawn a side at a time,
    each side run a little past its corner or stopped just short of it; a curve closed on itself is
    begun anywhere and drawn round in one go or two, whose ends overlap or do not quite meet. Lines
    are kept by depth, so that whatever stands in front can take out the lines of what it hides."""

    def __init__(self, r):
        self.r = r
        self.press = {}

    def line(self, depth, P, width=1.5, pressure=0.75, closed=False, wander=1.0, corners=True):
        r = self.r
        P = np.asarray(P, np.float64)[:, :2]
        into = self.press.setdefault(depth, np.zeros((PH, PW), np.float32))
        if corners and len(P) <= 5:
            Q = np.vstack([P, P[:1]]) if closed else P
            parts = []
            for a, b in zip(Q[:-1], Q[1:]):
                L = np.hypot(*(b - a))
                if L > 1:
                    u = (b - a) / L
                    parts.append(np.array([a - u * r.uniform(-1.5, 3.5), b + u * r.uniform(-1.5, 3.5)]))
        elif closed:
            loop = np.vstack([P, P, P[:1]])
            L = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T).sum()
            t0 = r.uniform(0, L)
            t1 = t0 + L * (1 + r.uniform(-0.012, 0.03))
            if L > 360:
                m = t0 + L * r.uniform(0.4, 0.6)
                parts = [along(loop, t0, m + r.uniform(-2, 6)), along(loop, m, t1)]
            else:
                parts = [along(loop, t0, t1)]
        else:
            parts = [P]
        for C in parts:
            self.stroke(into, C, width, pressure, wander, corners)

    def stroke(self, into, P, width, pressure, wander, corners):
        r = self.r
        C = hand(P, r, wander, 0.25, 0.5, corners)
        if len(C) < 3:
            return
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        L = s[-1]
        if len(P) == 2 and L > 80:
            u = (P[1] - P[0]) / (np.hypot(*(P[1] - P[0])) + 1e-9)
            bow = r.normal(0, 0.002) * L * 4 * (s / L) * (1 - s / L)
            C = C + np.array([-u[1], u[0]]) * bow[:, None]
        paths, presses, a = [], [], 0.0
        while a < L - 1:
            b = min(L, a + r.uniform(60, 220))
            if L - b < 25:
                b = L
            k = (s >= a) & (s <= b)
            if k.sum() > 2:
                t = s[k] - a
                p = pressure * r.uniform(0.8, 1.12) * np.clip(t / 2.5, 0, 1) ** 0.7 * np.clip((b - a - t) / 9, 0, 1) ** 0.8
                if len(t) > 12:
                    p = p * np.clip(1 + 0.25 * noise.line1d(len(t), 50, r), 0.3, None)
                paths.append(C[k] + r.normal(0, 0.35, 2))
                presses.append(p)
            if b >= L:
                break
            a = b - r.uniform(-1.5, 4)
        pencil.lay(into, paths, presses, width * r.uniform(0.85, 1.15), r)

    def ink(self, hidden):
        """The ink, each depth's lines taken out where something in front of them stands."""
        out = np.zeros((PH, PW), np.float32)
        for depth, p in self.press.items():
            out += p * (1 - hidden.get(depth, 0))
        return np.clip(out, 0, 1.3)


def palace(r, pen):
    """The palace across the garden: two long wings with a crenellated parapet and a kiosk at each end, two
    slender towers with onion cupolas, and between them a pavilion with a great pointed arch and an onion
    dome. -> dict of polygons: stone, lit and dark windows and their reveals, the recess of the arch"""
    P = dict(stone=[], lit=[], dark=[], reveal=[], recess=[], domes=[], finials=[], bands=[], towers=[])
    line = lambda Q, closed=False, w=1.5, p=0.75: pen.line(0, Q, w, p, closed)

    def window(cx, foot, w, h, chance, point=0.22):
        cx, foot = cx + r.normal(0, 0.8), foot + r.normal(0, 0.8)
        w, h = w * r.uniform(0.94, 1.06), h * r.uniform(0.97, 1.03)
        A = arch(cx, foot, w, h, point)
        lit = r.uniform() < chance
        P["lit" if lit else "dark"].append(A)
        P["reveal"].append(np.vstack([A[: len(A) // 2 + 1], arch(cx + 0.18 * w, foot, 0.64 * w, h - 0.12 * w, point)[len(A) // 2::-1]]))
        line(A, True, 1.3, 0.7)
        line([(cx, foot), (cx, foot - h + 0.3 * w)], w=1.0, p=0.5)          # the glazing bar
        if h > 60:
            line([(cx - w / 2, foot - 0.45 * h), (cx + w / 2, foot - 0.45 * h)], w=1.0, p=0.45)

    for s in (-1, 1):
        inner, outer = AX + s * 237, AX + s * 800
        x0, x1 = min(inner, outer), max(inner, outer)
        P["stone"].append(rect(x0, YW - 26, x1, YB))
        line([(outer, YB), (outer, YW - 26), (inner, YW - 26)])
        line([(x0, YW), (x1, YW)], w=1.3, p=0.65)
        line([(x0, YW + 8), (x1, YW + 8)], w=1.1, p=0.45)
        line([(x0, 1024), (x1, 1024)], w=1.1, p=0.5)
        P["bands"] += [(x0, x1, YW, 9), (x0, x1, 1024, 6)]
        x, zig = x0 + 9, []
        while x < x1 - 9:                                                   # the crenellation, cut by hand
            M = arch(x, YW - 26, 14 * r.uniform(0.9, 1.1), 13 * r.uniform(0.88, 1.12), point=0.6, n=6)
            P["stone"].append(M)
            zig += list(M)
            x += 25 + r.normal(0, 1.0)
        line(np.array(zig), w=1.1, p=0.55)
        x = inner + s * 52
        while (x - outer) * s < -50:                                        # windows, two storeys
            window(x, YB + 4, 30, 124, 0.62)
            window(x, 1006, 24, 68, 0.3)
            x += s * 70
        kx = outer - s * 34                                                 # a kiosk on the roof at the far end
        P["stone"].append(rect(kx - 24, YW - 74, kx + 24, YW - 26))
        line(rect(kx - 24, YW - 74, kx + 24, YW - 26), True, 1.2, 0.6)
        line(arch(kx, YW - 26, 26, 40, 0.3), True, 1.1, 0.55)
        P["domes"].append((kx, YW - 76, 56, 52))
        P["finials"].append((kx, YW - 128, YW - 152))
    for s in (-1, 1):                                                       # the towers
        cx = AX + s * 205
        P["stone"].append(rect(cx - 32, 612, cx + 32, YB))
        P["towers"].append((cx, 32))
        line([(cx - 32, YB), (cx - 32, 612)])
        line([(cx + 32, YB), (cx + 32, 612)])
        for yb in (700, 612):
            B = rect(cx - 41, yb - 13, cx + 41, yb)
            P["stone"].append(B)
            line(B, True, 1.3, 0.7)
            P["bands"].append((cx - 41, cx + 41, yb, 10))
        for foot, h in ((810, 54), (950, 54), (1102, 62)):
            window(cx, foot, 15, h, 0.5, point=0.3)
        P["domes"].append((cx, 599, 80, 100))
        P["finials"].append((cx, 499, 466))
    P["stone"] += [rect(AX - 172, YP, AX + 172, YB), rect(AX - 182, YP - 16, AX + 182, YP),
                   rect(AX - 108, 772, AX + 108, YP - 16), rect(AX - 118, 760, AX + 118, 772)]
    line([(AX - 172, YB), (AX - 172, YP)])
    line([(AX + 172, YB), (AX + 172, YP)])
    line(rect(AX - 182, YP - 16, AX + 182, YP), True, 1.4, 0.75)
    line([(AX - 108, YP - 16), (AX - 108, 772)])
    line([(AX + 108, YP - 16), (AX + 108, 772)])
    line(rect(AX - 118, 760, AX + 118, 772), True, 1.3, 0.7)
    P["bands"] += [(AX - 182, AX + 182, YP, 11), (AX - 118, AX + 118, 772, 8)]
    for x in (-72, -24, 24, 72):
        window(AX + x, 830, 16, 44, 0.5, point=0.3)
    P["domes"].append((AX, 759, 236, 182))
    P["finials"].append((AX, 577, 528))
    line(rect(AX - 114, 864, AX + 114, YB), True, 1.4, 0.75)                 # the frame of the great arch
    great = arch(AX, YB, 168, 292, point=0.32, n=30)
    P["recess"].append(great)
    line(great, True, 1.5, 0.8)
    door = arch(AX, YB, 70, 152, point=0.28)
    P["lit"].append(door)
    line(door, True, 1.4, 0.75)
    line([(AX, YB), (AX, YB - 120)], w=1.0, p=0.5)
    top = arch(AX, 1000, 52, 74, point=0.32)
    P["lit"].append(top)
    line(top, True, 1.3, 0.7)
    for x in (-136, 136):
        window(AX + x, YB + 4, 30, 124, 0.85)
        window(AX + x, 1006, 24, 68, 0.6)
    for cx, base, w, h in P["domes"]:
        D = onion(cx, base, w, h)
        P["stone"].append(D)
        line(D, True, 1.5, 0.8)
        for f in (-0.5, 0.0, 0.5):
            line(onion(cx, base, w, h, f=f), w=1.1, p=0.45)
    for cx, a, b in P["finials"]:
        line([(cx, a), (cx, b)], w=1.6, p=0.85)
    return P


def terrace(r, pen):
    """The terrace the palace stands on: its floor before the doors, the parapet along its edge, the
    retaining wall below, and the flight of steps down into the garden between two piers with urns."""
    T = dict(floor=[rect(-10, YB, PW + 10, YT)], parapet=[], wall=[], steps=[], risers=[], piers=[])
    line = lambda Q, closed=False, w=1.5, p=0.75: pen.line(1, Q, w, p, closed)
    half, foot_half = 182.0, 198.0
    for s in (-1, 1):
        a, b = sorted((AX + s * (half + 36), AX + s * 2000))
        T["parapet"].append(rect(a, YT, b, YF))
        T["wall"].append(rect(a, YF, b, YS + 6))
        line([(a, YT), (b, YT)], w=1.5, p=0.8)
        line([(a, YT + 8), (b, YT + 8)], w=1.1, p=0.5)
        line([(a, YF), (b, YF)], w=1.4, p=0.75)
        for y in np.arange(YF + 26, YS - 40, 26):
            line([(a, y + r.normal(0, 0.5)), (b, y + r.normal(0, 0.5))], w=1.0, p=0.3)
        px = AX + s * (half + 18)
        Q = rect(px - 20, YT - 26, px + 20, YS + 6)
        T["piers"].append(Q)
        line(Q, True, 1.5, 0.8)
        C = rect(px - 27, YT - 34, px + 27, YT - 26)
        T["piers"].append(C)
        line(C, True, 1.4, 0.75)
        U = urn(px, YT - 34, 58)
        T["piers"].append(U)
        line(U, True, 1.4, 0.75)
    n = 5
    ys = np.linspace(YF - 2, YS, n + 1)
    for k in range(n):
        y0, y1 = ys[k], ys[k + 1]
        hw = half + (foot_half - half) * (k + 0.5) / n
        tread = 6 + 1.6 * k
        T["steps"].append(rect(AX - hw, y0, AX + hw, y0 + tread))
        T["risers"].append(rect(AX - hw, y0 + tread, AX + hw, y1))
        line([(AX - hw, y0), (AX + hw, y0)], w=1.5, p=0.8)
        line([(AX - hw, y0 + tread), (AX + hw, y0 + tread)], w=1.1, p=0.45)
    line([(AX - foot_half, YS), (AX + foot_half, YS)], w=1.4, p=0.7)
    return T


def cypress(r, xb, yb, h, w):
    """A cypress standing at (xb, yb), h tall and w across at its fullest: a flame of dark foliage, fullest
    a fifth of the way up, its edge gently lobed, its tip leaning a little, its foot a ragged fringe of
    foliage on the ground."""
    n = 220
    t = np.linspace(0, 1, n)
    cx = xb + r.normal(0, 0.025) * h * t ** 2 + 2.5 * noise.line1d(n, 50, r) * t
    prof = np.where(t < 0.2, 0.82 + 0.18 * np.sin(0.5 * np.pi * t / 0.2), np.clip((1 - t) / 0.8, 0, 1) ** 0.78)
    tuft = lambda: 1 + 0.07 * np.abs(noise.line1d(n, 7, r)) + 0.02 * noise.line1d(n, 2.5, r)
    hw, y = w / 2 * prof, yb - t * h
    u = np.linspace(0, 1, 16)[1:-1]
    foot = np.stack([cx[0] + hw[0] * (1 - 2 * u), yb + 3 + 6 * np.abs(noise.line1d(14, 2.5, r))], 1)
    return np.vstack([np.stack([cx - hw * tuft(), y], 1), np.stack([cx + hw * tuft(), y], 1)[::-1], foot])


def crown(r, cx, cy, rx, ry):
    """A crown of the stone pine as Dulac paints it: a flat cloud of needles, domed on top and nearly level
    underneath, its edge scalloped in small round lobes. -> (outline polygon, lobes (x, y, radius))"""
    th = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    wob = 1 + 0.1 * noise.line1d(160 + 40, 18, r)[:160]
    below = np.sin(th) > 0
    O = np.stack([cx + rx * wob * np.cos(th), cy + ry * wob * np.sin(th) * np.where(below, 0.5, 1.0)], 1)
    lobes = []
    for i in range(0, 160, 3):
        x, y = O[i]
        q = (r.uniform(0.12, 0.22) if below[i] else r.uniform(0.2, 0.38)) * ry
        k = 0.35 * q
        lobes.append((x - k * np.cos(th[i]), y - k * np.sin(th[i]), q))
    return O, lobes


def pine(r, pen):
    """The stone pine on the left, drawn as Dulac draws a tree by night: a pale trunk and limbs outlined in
    pen and hatched on the side away from the moon, and dark flat crowns of needles held out against the
    stars on twigs that show pale between them. -> (bark polygons, crown polygons, crown lobes, inner clouds)"""
    parts, edges, polys, lobes, inner = [], [], [], [], []
    trunk = ribbon(TRUNK, r, knots=0.07)
    limbs = [ribbon(L, r, knots=0.06) for L in LIMBS]
    for R in [trunk] + limbs:
        parts.append(R[0])
        edges += [R[1], R[2]]
    spines = [(R[3], R[4]) for R in limbs] + [(trunk[3][-60:], trunk[4][-60:])]
    for cx, cy, rx, ry in CROWNS:
        O, L = crown(r, cx, cy, rx, ry)
        polys.append(O)
        lobes += L
        for _ in range(int(r.integers(1, 3))):
            ix, iy = cx + r.uniform(-0.4, 0.4) * rx, cy + r.uniform(0.0, 0.25) * ry
            inner.append(crown(r, ix, iy, r.uniform(0.35, 0.6) * rx, r.uniform(0.45, 0.7) * ry)[0])
        best = min(((np.hypot(*(xy[i] - (cx, cy + 0.5 * ry))), xy[i], w[i]) for xy, w in spines
                    for i in range(0, len(xy), 4)), key=lambda z: z[0])
        _, (sx, sy), w0 = best
        for _ in range(int(r.integers(2, 5))):
            tx, ty = cx + r.uniform(-0.6, 0.6) * rx, cy + 0.3 * ry
            mx, my = (sx + tx) / 2 + r.normal(0, 14), (sy + ty) / 2 + r.uniform(4, 16)
            R = ribbon([(sx, sy, max(4.0, 0.45 * w0)), (mx, my, max(3.5, 0.28 * w0)), (tx, ty, 2.5)], r, step=2.0,
                       knots=0.03)
            parts.append(R[0])
            edges += [R[1], R[2]]
    for E in edges:
        pen.line(3, E[::3], 1.5, 0.8, corners=False, wander=0.4)
    # hatching across the trunk and the big limbs on their shaded side, and a few marks of the bark along them
    for xy, w in [(trunk[3], trunk[4])] + [(R[3], R[4]) for R in limbs]:
        g = np.gradient(xy, axis=0)
        g /= np.hypot(*g.T)[:, None] + 1e-9
        nrm = np.stack([-g[:, 1], g[:, 0]], 1)
        nrm *= np.where(nrm[:, :1] * 0.7 - nrm[:, 1:] * 0.7 > 0, -1.0, 1.0)     # pointing away from the moon
        gate = noise.line1d(len(xy), 30, r)
        for i in range(0, len(xy), 2):
            if gate[i] < -0.2 or w[i] < 12:
                continue
            L = w[i] * r.uniform(0.15, 0.45)
            a = xy[i] + nrm[i] * (w[i] / 2 - 1)
            b = a - nrm[i] * L + g[i] * r.normal(0, 2)
            pen.line(3, [a, (a + b) / 2 + g[i] * 1.5, b], 1.1, 0.6, corners=False, wander=0.2)
        for i in range(0, len(xy) - 30, 22):
            if r.uniform() < 0.5 and w[i] > 20:
                o = r.uniform(-0.3, 0.3) * w[i]
                seg = xy[i:i + int(r.integers(8, 30))] + nrm[i] * o
                pen.line(3, seg, 1.0, 0.45, corners=False, wander=0.3)
    return parts, polys, lobes, inner


# rings of petals, outermost first: count, distance out, half-width, half-length, how far it stands up
RINGS = dict(open=((5, 0.56, 0.5, 0.42, 0.0), (4, 0.34, 0.4, 0.31, 0.1), (3, 0.16, 0.3, 0.22, 0.17),
                   (2, 0.05, 0.17, 0.13, 0.22)),
             half=((4, 0.3, 0.46, 0.52, 0.0), (3, 0.1, 0.26, 0.36, 0.3), (2, 0.03, 0.17, 0.26, 0.42)))


def rose(r, R, kind):
    """One flower in its own frame, its heart at the origin and its stalk below: petals (x, y, half-width,
    half-length, angle, notch, squash) from the back to the front. An open rose is rings of petals round a
    closed heart, each ring standing higher than the one outside it, the petals notched at the lip; it may
    face us or be tipped well over. A half-open rose is a goblet of four upright shells seen from the side,
    the front ones over a heart that stands up out of it in a furl of narrow petals. A bud is an egg of
    three petals wrapped one round another and drawn to a point."""
    if kind == "bud":
        m = r.choice([-1.0, 1.0])
        P = [(0.0, -0.08, 0.33, 0.6, 0.0), (-0.11 * m, 0.03, 0.27, 0.5, -0.25 * m), (0.12 * m, 0.06, 0.26, 0.47, 0.28 * m)]
        return [(x * R, y * R, a * R * r.uniform(0.9, 1.1), b * R * r.uniform(0.9, 1.1),
                 -np.pi / 2 + tilt + r.normal(0, 0.05), -0.12, 1.0) for x, y, a, b, tilt in P], 1.0
    face = r.uniform(0.5, 0.97) if kind == "open" else r.uniform(0.42, 0.55)
    out = []
    for i, (k, dist, a, b, lift) in enumerate(RINGS[kind]):
        k = k + int(r.integers(0, 2)) if kind == "open" and i < 2 else k
        ph = r.uniform(0, 2 * np.pi)
        for j in range(k):
            ang = ph + 2 * np.pi * j / k + r.normal(0, 0.15)
            x, y = dist * R * np.cos(ang), face * dist * R * np.sin(ang) - lift * R
            if kind == "open":
                shape, sq, z = ang, face, i
            else:
                shape, sq = -np.pi / 2 + (0.5 if i else 0.4) * np.cos(ang), 1.0
                y += (0.08 * R * np.sin(ang) if i == 0 else 0)
                z = i + (2.6 * np.sin(ang) if i == 0 else 0)
            out.append((z + r.normal(0, 0.02), (x, y, a * R * r.uniform(0.88, 1.12), b * R * r.uniform(0.88, 1.12),
                                                shape, 0.14, sq)))
    out.sort(key=lambda q: q[0])
    return [p for _, p in out], face


def _petal_q(X, Y, p):
    px, py, a, b, ang, notch, sq = p
    dx, dy = X - px, (Y - py) / sq
    c, s = np.cos(ang), np.sin(ang)
    u, v = dx * c + dy * s, -dx * s + dy * c
    return np.sqrt((u / b) ** 2 + (v / a) ** 2) + notch * np.exp(-(v / (0.22 * a)) ** 2) * np.clip(u / b, 0, None), u


def blooms(r, flowers, pen):
    """The flowers rasterised petal by petal from the front back: each petal is seen where no petal in front
    of it lies, a crease of shadow lies along the edges of the petals in front of it, its lip is left pale,
    and its edge is drawn in pen where it shows. Each flower is turned its own way. -> dict of full-size
    maps (mask, crease, lip, heart, tone, deep), and the sepals under the buds and cups as polygons"""
    out = {k: np.zeros((PH, PW), np.float32) for k in ("mask", "crease", "lip", "heart", "side", "tone", "deep")}
    out["sepals"] = []
    t = np.linspace(0, 2 * np.pi, 90)
    for cx, cy, R, kind, turn, tone, deep in flowers:
        petals, face = rose(r, R, kind)
        c, s = np.cos(turn), np.sin(turn)

        def world(x, y):
            return cx + c * x - s * y, cy + s * x + c * y

        if kind != "open":                                  # the sepals under the flower, and its stalk
            base = (0.42 if kind == "bud" else 0.64) * R
            seps = [(np.pi / 2 + 0.9 + r.normal(0, 0.15), 0.75, 0.2), (np.pi / 2 - 0.9 + r.normal(0, 0.15), 0.7, 0.2)]
            if kind == "bud":                               # and one clasping the bud
                seps.append((-np.pi / 2 + r.choice([-0.55, 0.55]), 0.62, 0.16))
            else:
                seps = [(a, 0.6 * L, wd) for a, L, wd in seps] + [(np.pi / 2 + r.normal(0, 0.2), 0.45, 0.16)]
            for a, L, wd in seps:
                F = leaf(0, base, L * R, wd * R, a)
                out["sepals"].append(np.stack(world(F[:, 0], F[:, 1]), 1))
            sx, sy = world(np.array([0, 0.06 * R, -0.04 * R]), np.array([base, base + 0.7 * R, base + 1.4 * R]))
            pen.line(5, np.stack([sx, sy], 1), 1.2, 0.7, corners=False, wander=0.3)
        ext = 1.5 * R
        x0, x1 = int(max(0, cx - ext)), int(min(PW, cx + ext))
        y0, y1 = int(max(0, cy - ext)), int(min(PH, cy + ext))
        Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        LX, LY = c * (X - cx) + s * (Y - cy), -s * (X - cx) + c * (Y - cy)
        qs = [_petal_q(LX, LY, p) for p in petals]
        ms = [noise.smoothstep(1.05, 0.95, q) for q, _ in qs]
        over = np.zeros_like(X)
        mask, crease, lip = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
        for j in reversed(range(len(petals))):
            vis = ms[j] * (1 - over)
            if over.max() > 0:
                crease += vis * np.clip(2.2 * ndimage.gaussian_filter(over, 0.07 * R) - over, 0, 1)
            q, u = qs[j]
            lip += vis * noise.smoothstep(0.72, 0.97, q) * (u > 0)
            mask = np.maximum(mask, ms[j])
            over = np.maximum(over, ms[j])
            px, py, a, b, ang, notch, sq = petals[j]
            uu = b * np.cos(t) * (1 - notch / 0.14 * 0.12 * np.exp(-(np.sin(t) / 0.22) ** 2) * (np.cos(t) > 0))
            vv = a * np.sin(t)
            ca, sa = np.cos(ang), np.sin(ang)
            ex, ey = px + uu * ca - vv * sa, py + sq * (uu * sa + vv * ca)
            seen = np.ones(len(t), bool)
            for k in range(j + 1, len(petals)):
                seen &= _petal_q(ex, ey, petals[k])[0] > 1.02
            wx, wy = world(ex, ey)
            for a0, a1 in runs(seen):
                if a1 - a0 > 4:
                    pen.line(5, np.stack([wx[a0:a1], wy[a0:a1]], 1), 1.2, 0.7, corners=False, wander=0.2)
        inner = [p for p in petals if p[1] < -0.15 * R] or petals
        hx = np.mean([p[0] for p in inner])
        hy = np.mean([p[1] for p in inner]) - (0.35 * np.mean([p[3] for p in inner]) if kind == "half" else 0)
        rad = {"open": 0.42, "half": 0.3, "bud": 0.2}[kind] * R
        heart = mask * noise.smoothstep(rad, 0.2 * rad, np.hypot(LX - hx, (LY - hy) / (face if kind == "open" else 1)))
        # the side away from the light: the lantern's if it is near, else the moon's
        to = np.subtract(LAMP, (cx, cy))
        if np.hypot(*to) > 520:
            to = np.subtract(MOON[:2], (cx, cy))
        to = to / np.hypot(*to)
        side = mask * np.clip(0.5 - 0.7 * ((X - cx) * to[0] + (Y - cy) * to[1]) / R, 0, 1)
        sl = (slice(y0, y1), slice(x0, x1))
        for key, v in (("mask", mask), ("crease", np.clip(crease, 0, 1)), ("lip", np.clip(lip, 0, 1)), ("heart", heart),
                       ("side", side)):
            out[key][sl] = np.maximum(out[key][sl], v)
        out["tone"][sl] = np.where(mask > 0.05, tone, out["tone"][sl])
        out["deep"][sl] = np.where(mask > 0.05, deep, out["deep"][sl])
    return out


def bush(r, pen):
    """The rose bush in the near corner on the right, with the lantern hung among it: a mound of leaves, the
    leaves at its edge and round the flowers drawn one by one, canes arching out of it with buds at their
    ends, and the flowers, open, half open and in bud, turned this way and that, some of them partly behind
    a leaf. -> dict of polygons and discs, and the flowers (x, y, size, kind, turn, tone, depth of colour)"""
    B = dict(lobes=[], body=[], leaves=[], shade=[], front=[], lit=[], flowers=[], lamp=[], glass=[], iron=[])
    top = lambda x: np.interp(x, [980, 1080, 1180, 1280, 1380, 1500, 1640, 1800],
                              [2160, 1930, 1690, 1550, 1478, 1450, 1410, 1420])
    lx, ly = LAMP
    for _ in range(140):
        x = r.uniform(1000, 1800)
        y = r.uniform(top(x) + 30, PH + 80)
        B["lobes"].append((x, y, r.uniform(45, 110) * (0.6 + 0.6 * (y - 1300) / 1000)))
    for x in np.arange(1000, 1800, 34):
        B["lobes"].append((x + r.normal(0, 8), top(x) + r.uniform(22, 50), r.uniform(30, 52)))
    xs = np.arange(1040, 1810, 10.0)                        # the body of the mound, under its leafy edge
    B["body"] = [np.vstack([np.stack([xs, top(xs) + 85], 1), [(1810, PH + 50), (990, PH + 50)]])]

    def leafset(x, y, a, size):
        """A rose leaf: a stalk with two pairs of leaflets and one at its end."""
        c, s = np.cos(a), np.sin(a)
        out = [leaf(x + c * size * 1.1, y + s * size * 1.1, size, 0.48 * size, a + r.normal(0, 0.15))]
        for k, f in enumerate((0.35, 0.75)):
            px, py = x + c * size * f * 1.1, y + s * size * f * 1.1
            for side in (-1, 1):
                out.append(leaf(px, py, size * (0.72 + 0.2 * k), 0.45 * size, a + side * r.uniform(0.7, 1.1)))
        return out

    canes = []
    for x0, dx, hgt in ((1160, -80, 230), (1260, -40, 250), (1340, 30, 190), (1580, 70, 230), (1690, -30, 170),
                        (1100, -60, 150)):
        y0 = top(x0) + 50
        C = np.array([(x0, y0), (x0 + 0.3 * dx, y0 - 0.6 * hgt), (x0 + dx, y0 - hgt)])
        canes.append(C)
        pen.line(4, brush.path(C, 2.0)[:, :2][::3], 1.3, 0.75, corners=False)
        P = brush.path(C, 4.0)[:, :2]
        for i in range(8, len(P) - 6, 14):
            a = np.arctan2(*(P[min(i + 2, len(P) - 1)] - P[i])[::-1])
            B["leaves"] += leafset(*P[i], a + r.choice([-1, 1]) * r.uniform(0.6, 1.2), r.uniform(20, 28))
    for x in np.arange(1000, 1800, 24):
        if r.uniform() < 0.85:
            B["leaves"] += leafset(x, top(x) + r.uniform(-10, 30), -np.pi / 2 + r.normal(0, 0.9), r.uniform(22, 34))
    for _ in range(110):                                    # leaves deepened one by one inside the mass, when dry
        x = r.uniform(1000, 1760)
        y = r.uniform(top(x) + 50, PH + 30)
        B["shade"] += leafset(x, y, r.uniform(0, 2 * np.pi), r.uniform(20, 32) * (0.8 + 0.5 * (y - 1400) / 900))
    spots = [(1372, 1590, 50, "open"), (1478, 1626, 54, "open"), (1612, 1574, 46, "open"), (1196, 1890, 56, "open"),
             (1430, 1860, 58, "open"), (1650, 1830, 56, "open"), (1330, 2110, 62, "open"), (1600, 2170, 66, "open"),
             (1300, 1528, 40, "half"), (1548, 1500, 38, "half"), (1236, 1668, 42, "half"), (1688, 1488, 36, "half"),
             (1684, 1702, 40, "half"), (1100, 2010, 46, "half"), (1500, 2040, 44, "half"),
             (1524, 1716, 26, "bud"), (1716, 1604, 24, "bud"), (1150, 1760, 27, "bud"), (1392, 1990, 28, "bud"),
             (1690, 1966, 28, "bud"), (1262, 1592, 22, "bud")]
    for C in canes[:4]:
        a = np.arctan2(*(C[-1] - C[-2])[::-1])                              # a bud at the end, along the cane
        spots.append((C[-1][0], C[-1][1], r.uniform(20, 26), "bud", a + np.pi / 2))
    spread = dict(open=0.3, half=0.45, bud=0.35)
    age = dict(open=((0.6, 1.1), (0.0, 0.35)), half=((0.85, 1.2), (0.3, 0.6)), bud=((1.0, 1.3), (0.6, 0.95)))
    for sp in spots:
        x, y, R, kind = sp[:4]
        turn = sp[4] if len(sp) > 4 else r.normal(0, spread[kind])
        (t0, t1), (d0, d1) = age[kind]                      # the older the flower, the paler and pinker
        B["flowers"].append((x, y, R, kind, turn, r.uniform(t0, t1), r.uniform(d0, d1)))
        for _ in range(int(r.integers(1, 3) if kind == "bud" else r.integers(3, 6))):
            a = r.uniform(0, 2 * np.pi)
            B["leaves"] += leafset(x + 0.7 * R * np.cos(a), y + 0.6 * R * np.sin(a), a, r.uniform(0.5, 0.7) * max(R, 34))
        if kind != "bud" and r.uniform() < 0.5:                             # a leaf in front, across its edge
            a = r.uniform(0, 2 * np.pi)
            B["front"] += leafset(x + R * np.cos(a), y + 0.9 * R * np.sin(a),
                                  a + np.pi + r.choice([-1, 1]) * r.uniform(0.9, 1.3), r.uniform(0.38, 0.48) * R)
    near = lambda P: np.hypot(*(P.mean(0) - (lx, ly + 40))) < 250
    B["lit"] = [P for P in B["leaves"] + B["front"] if near(P)]
    for P in B["leaves"]:
        if near(P) or r.uniform() < 0.3:
            pen.line(4, P, 1.1, 0.6, closed=True, corners=False, wander=0.2)
    for P in B["front"]:
        pen.line(6, P, 1.1, 0.65, closed=True, corners=False, wander=0.2)
    # the lantern: a domed cap with a finial, a six-sided body of pierced brass and glass, a knob below
    body = np.array([(lx - 31, ly - 46), (lx + 31, ly - 46), (lx + 27, ly + 46), (lx - 27, ly + 46)])
    cap = onion(lx, ly - 44, 80, 66)
    base = np.vstack([ellipse(lx, ly + 46, 27, 26, 0, np.pi, 24), [(lx - 27, ly + 46)]])
    knob = ellipse(lx, ly + 76, 7, 7)
    B["lamp"] = [body, cap, base, knob]
    B["glass"] = [arch(lx, ly + 36, 30, 72, 0.3), arch(lx - 24, ly + 36, 8, 66, 0.3), arch(lx + 24, ly + 36, 8, 66, 0.3)]
    for P in B["lamp"] + B["glass"]:
        pen.line(6, P, 1.4, 0.85, closed=True, corners=False, wander=0.2)
    for o in (-19, 19):
        pen.line(6, [(lx + o, ly - 46), (lx + 0.88 * o, ly + 46)], 1.3, 0.8)
    pen.line(6, [(lx - 33, ly - 44), (lx + 33, ly - 44)], 1.7, 0.9)
    pen.line(6, [(lx, ly - 110), (lx, ly - 132)], 1.5, 0.85)
    pen.line(6, ellipse(lx, ly - 138, 6, 6), 1.3, 0.85, closed=True)
    crook = brush.path(np.array([(lx + 140, 1800), (lx + 132, 1560), (lx + 124, ly - 190), (lx + 76, ly - 222),
                                 (lx + 14, ly - 196), (lx, ly - 150)]), 2.0)[:, :2]
    for o in (-3.0, 3.0):
        pen.line(6, crook[::3] + (o, 0), 1.3, 0.9, corners=False, wander=0.2)
    B["iron"] = [np.vstack([crook + (-4, 0), (crook + (4, 0))[::-1]])]
    return B


class Garden:
    """Everything drawn, as masks in plate px."""

    def __init__(self, r, pen):
        self.P, self.T = palace(r, pen), terrace(r, pen)
        bark, crowns, lobes, inner = pine(r, pen)
        self.B = B = bush(r, pen)
        P, T = self.P, self.T
        mx, my, mr = MOON
        self.moon = fill([ellipse(mx, my, mr, mr)], 0.8)
        # the pine: its bark and its crowns
        self.bark = fill(bark, 0.8)
        self.crowns = fill(crowns, 0.9, lobes)
        self.inner = fill(inner, 1.0) * self.crowns
        # the rose bush: its flowers, the leaves in front of them, and the lantern
        F = blooms(r, B["flowers"], pen)
        self.front = fill(B["front"] + F["sepals"], 0.7)
        for S in F["sepals"]:
            pen.line(6, S, 1.0, 0.6, closed=True, corners=False, wander=0.2)
        self.lamp, self.glass = fill(B["lamp"], 0.8), fill(B["glass"], 0.8)
        self.roses = F["mask"] * (1 - self.front) * (1 - self.lamp)
        self.crease, self.lip, self.heart, self.side = (F[k] for k in ("crease", "lip", "heart", "side"))
        self.tone, self.deep = F["tone"], F["deep"]
        self.litleaves = fill(B["lit"], 0.7) * (1 - self.roses) * (1 - self.lamp)
        self.iron = fill(B["iron"], 0.7)
        self.mound = np.clip(fill(B["leaves"] + B["body"], 0.7, B["lobes"]) + self.litleaves + self.front, 0, 1)
        self.shadeleaves = fill(B["shade"], 0.7)
        self.bush = np.clip(self.mound + self.roses + self.lamp, 0, 1)
        self.reserve = np.clip(self.bark + self.roses + self.litleaves + self.lamp, 0, 1)
        keep = 1 - self.reserve
        # the palace and the terrace
        self.stone = fill(P["stone"])
        self.lit, self.dark, self.recess = fill(P["lit"]), fill(P["dark"]), fill(P["recess"])
        self.reveal = fill(P["reveal"], 0.6)
        self.floor = fill(T["floor"])
        self.parapet, self.wall = fill(T["parapet"]) * keep, fill(T["wall"]) * keep
        self.steps, self.risers, self.piers = fill(T["steps"]), fill(T["risers"]), fill(T["piers"])
        front = np.clip(self.parapet + self.wall + self.steps + self.risers + self.piers, 0, 1)
        self.facade = front
        self.stone *= (1 - front) * keep
        # the cypresses: a pair at the foot of the terrace either side of the steps, and more on the terrace
        self.trees = [(AX - 300, YS + 24, 820, 112), (AX + 300, YS + 24, 760, 106),
                      (AX - 455, YB + 32, 430, 74), (AX + 448, YB + 32, 396, 70), (AX - 560, YB + 30, 330, 60),
                      (AX + 590, YB + 30, 372, 66), (AX + 700, YB + 28, 300, 58), (AX - 690, YB + 30, 380, 66)]
        self.cyp = [cypress(r, *t) for t in self.trees]
        self.cmask = [fill([C], 0.8) * keep * (1 - front if t[1] < YS else 1) for C, t in zip(self.cyp, self.trees)]
        blobs = [ellipse(x, y, rx, ry) for x, y, rx, ry in
                 ((1700, 990, 120, 140), (1610, 1060, 110, 120), (1690, 1130, 140, 90), (1560, 1130, 90, 70))]
        self.far = fill(blobs, 1.0) * keep
        top = YS - 52 + 10 * noise.line1d(PW, 60, r) - 5 * np.abs(noise.line1d(PW, 12, r))
        self.hedge = (noise.smoothstep(-1, 1, YY - top[None, :]) * noise.smoothstep(1, -1, YY - YS - 30)
                      * noise.smoothstep(-1, 1, np.abs(XX - AX) - 226) * keep)
        # the ground: the walk from the steps to the basin and on towards us, the paving round the basin, the
        # borders of the walk and the lawns. The edges of the walk are as the pen drew them, not true.
        cx, cy, rx, ry = BASIN
        dd = np.clip((np.arange(PH, dtype=np.float32) - HZ) / (YS - HZ), 0, None)
        edges = [AX + s * 118 * dd * (1 + 0.012 * noise.line1d(PH, 500, r)) + 1.6 * np.sqrt(dd) * noise.line1d(PH, 70, r)
                 for s in (-1, 1)]
        self.basin = fill([ellipse(cx, cy, rx, ry)], 0.8)
        self.curb = fill([np.vstack([ellipse(cx, cy, rx, ry, 0, np.pi, 48), ellipse(cx, cy + 36, rx, ry, np.pi, 0, 48)])], 0.8)
        self.water = fill([ellipse(cx, cy, rx - 28, ry - 10)], 0.8)
        self.inwall = self.water * (1 - fill([ellipse(cx, cy + 9, rx - 30, ry - 11)], 0.8))
        self.cast = fill([ellipse(cx - 14, cy + 58, rx + 4, ry + 2)], 1.2) * (1 - self.basin) * (1 - self.curb)
        ring = fill([ellipse(cx, cy + 16, rx + 70, ry + 30)], 1.0)
        straight = (noise.smoothstep(-1.2, 1.2, XX - edges[0][:, None]) * noise.smoothstep(-1.2, 1.2, edges[1][:, None] - XX)
                    * (YY > YS))
        self.ground = (YY > YS - 30).astype(np.float32) * (1 - self.basin) * (1 - self.curb) * keep
        self.walk = np.clip(straight + ring, 0, 1) * self.ground
        self.lawn = self.ground * (1 - self.walk)
        rows = []                                   # the borders: small upright touches just outside each edge
        for s, e in zip((-1, 1), edges):
            y = YS + 6.0
            while y < PH - 2:
                d = (y - HZ) / (YS - HZ)
                L = (5 + 6 * d) * r.uniform(0.7, 1.3)
                rows.append((e[int(y)] + s * (1 + r.uniform(0, 0.6) * L), y + r.uniform(0, 1.5) * d, L,
                             0.6 * L * r.uniform(0.8, 1.2), -np.pi / 2 + s * r.uniform(0.05, 1.0)))
                y += (1.0 + 1.5 * d) * r.uniform(0.6, 1.3)
        self.border = wash.touches((PH, PW), rows) * self.lawn * (1 - ring)
        line = lambda Q, closed=False, w=1.5, p=0.75, c=True: pen.line(2, Q, w, p, closed, corners=c)
        ys = np.arange(YS + 1, PH, 3.0)
        for e in edges:
            xs = e[ys.astype(int)]
            for a, b in runs(ring[ys.astype(int), xs.astype(int).clip(0, PW - 1)] < 0.4):
                if b - a > 3:
                    line(np.stack([xs[a:b], ys[a:b]], 1), w=1.4, p=0.72, c=False)
        E = ellipse(cx, cy + 16, rx + 70, ry + 30, n=480)
        for a, b in runs(straight[E[:, 1].astype(int).clip(0, PH - 1), E[:, 0].astype(int).clip(0, PW - 1)] < 0.5):
            if b - a > 3:
                line(E[a:b], w=1.3, p=0.65, c=False)
        line(ellipse(cx, cy, rx, ry), True, 1.5, 0.8)
        line(ellipse(cx, cy, rx - 28, ry - 10), True, 1.3, 0.65)
        line(ellipse(cx, cy + 9, rx - 30, ry - 11, 1.08 * np.pi, 1.92 * np.pi, 60), w=1.0, p=0.35)
        line(ellipse(cx, cy + 36, rx, ry, 0, np.pi, 48), w=1.5, p=0.8)
        line([(cx - rx, cy), (cx - rx, cy + 36)], w=1.4, p=0.75)
        line([(cx + rx, cy), (cx + rx, cy + 36)], w=1.4, p=0.75)
        # the fountain: a column in the middle of the basin carrying a shallow bowl
        self.column = fill([rect(cx - 15, BOWL + 30, cx + 15, cy + 4), ellipse(cx, cy, 32, 9)], 0.8)
        self.bowl = fill([ellipse(cx, BOWL, 96, 24), ellipse(cx, BOWL, 96, 36, 0, np.pi, 40)], 0.8)
        line(ellipse(cx, BOWL, 96, 24), True, 1.3, 0.7)
        line(ellipse(cx, BOWL, 96, 36, 0, np.pi, 40), w=1.3, p=0.7)
        line([(cx - 15, BOWL + 34), (cx - 15, cy)], w=1.3, p=0.7)
        line([(cx + 15, BOWL + 34), (cx + 15, cy)], w=1.3, p=0.7)
        # the sky: everything above the palace's foot that the palace, the moon and the pale bark do not take
        self.sky = noise.smoothstep(1, -1, YY - YB) * (1 - self.stone) * (1 - self.moon) * (1 - front) * keep
        trees = np.clip(sum(self.cmask) + self.far + self.crowns, 0, 1)
        self.open = self.sky * (1 - trees)


def sky(box, r, g):
    """The night in four washes, each let dry and washed down before the next: Prussian blue with a
    little cobalt, then ultramarine and Prussian laid side to side in broad strokes, deeper at the top,
    then indigo and ultramarine over the upper sky, and last a thin glaze over all. Every wash was thinned
    as it came near the moon, and at the end the colour round it was lifted with a damp brush and a
    breath of ochre run into the lifted place, which turns the blue green."""
    m = g.sky
    v = np.clip(YY / YB, 0, 1)
    mx, my, mr = MOON
    rho = np.hypot(XX - mx, YY - my) / mr + 0.5 * noise.fbm((PH, PW), 90, r, octaves=3)
    thin = 1 - 0.45 * noise.smoothstep(6.5, 1.0, rho)
    deep = noise.fbm((PH, PW), 520, r, octaves=3)
    sponge = lambda s: 0.55 + 0.45 * noise.smoothstep(-1.0, 1.2, noise.fbm((PH, PW), s, r, octaves=2))
    wet, dep = box.wash(m, 2.5, pool=0.25, rim=0.35, rim_width=7, tides=0.2, scale=380)
    box.lay(dep * 0.34 * thin, prussian=0.62, cobalt=0.38)
    box.lay(wc.charge(noise.smoothstep(0.55, 1.0, v) * m, wet, 60, r, streak=0.2) * 0.12, viridian=0.55, prussian=0.45)
    box.scrub(0.4, m * sponge(400))
    band = box.banded(0, YB, 0, PW, 85)
    wet, dep = box.wash(m, 2.0, pool=0.22, rim=0.45, rim_width=5, tides=0.15, scale=300)
    grade = (0.4 + 0.75 * (1 - v) ** 1.2) * (1 + 0.16 * deep) * thin
    box.lay(dep * (0.72 + 0.28 * band) * grade * 0.6, ultramarine=0.48, prussian=0.4, indigo=0.12)
    box.scrub(0.35, m * sponge(300))
    upper = noise.smoothstep(0.78, 0.3, v + 0.1 * deep)
    wet, dep = box.wash(m, 2.0, pool=0.25, rim=0.35, rim_width=6, tides=0.15, scale=260)
    box.lay(dep * upper * 0.42 * (1 + 0.15 * deep) * thin, indigo=0.38, ultramarine=0.47, prussian=0.15)
    box.scrub(0.3, m * sponge(240))
    wet, dep = box.wash(m, 2.0, pool=0.2, rim=0.3, rim_width=5, tides=0.1, scale=220)
    box.lay(dep * 0.17 * (0.7 + 0.3 * (1 - v)) * thin, prussian=0.5, ultramarine=0.5)
    lifted = noise.smoothstep(3.6, 3.0, rho) * m            # where the damp brush went: a soft but definite edge
    box.scrub(0.55, lifted * noise.smoothstep(3.6, 1.2, rho))
    wet, dep = box.wash(lifted, 1.5, pool=0.3, rim=0.3, rim_width=6, scale=120)
    box.lay(dep * noise.smoothstep(3.6, 1.0, rho) * 0.1, yellow_ochre=0.6, gamboge=0.4)


def moon(box, r, g):
    """The moon is the paper, with a wash of pale gamboge and ochre and two faint grey seas."""
    wet, dep = box.wash(g.moon, 0.4, pool=0.2, rim=0.5, rim_width=2, scale=40)
    box.lay(dep * 0.1, gamboge=0.6, yellow_ochre=0.4)
    mx, my, mr = MOON
    seas = noise.smoothstep(0.4, 1.2, noise.fbm((PH, PW), 22, r, octaves=2)) * \
        noise.smoothstep(0.95, 0.4, np.hypot(XX - mx - 12, YY - my + 8) / mr)
    box.lay(wc.charge(seas * g.moon, wet, 6, r) * 0.08, cobalt=0.5, yellow_ochre=0.5)


def specks(r, x, y, size, squash=(0.8, 1.2)):
    """Small round touches of the point of a fine brush, as a 0..1 map."""
    im = np.zeros((PH, PW), np.float32)
    for xx, yy, s in zip(x, y, size):
        k = int(np.ceil(2 * s)) + 2
        x0, y0 = int(xx) - k, int(yy) - k
        if x0 < 0 or y0 < 0 or x0 + 2 * k >= PW or y0 + 2 * k >= PH:
            continue
        gy, gx = np.mgrid[0:2 * k + 1, 0:2 * k + 1].astype(np.float32)
        d = np.hypot(gx + x0 - xx, (gy + y0 - yy) * r.uniform(*squash))
        im[y0:y0 + 2 * k + 1, x0:x0 + 2 * k + 1] += noise.smoothstep(s + 0.6, s - 0.6, d)
    return np.clip(im, 0, 1)


def stars(box, r, g):
    """Stars in Chinese white, the last thing on the sheet: a few hundred specks from the point of a fine
    brush, thicker in the upper sky and few near the moon, with a handful of larger ones."""
    mx, my, mr = MOON
    n = 1400
    x, y = r.uniform(0, PW, n), r.uniform(0, YB, n)
    xi, yi = x.astype(int).clip(0, PW - 1), y.astype(int).clip(0, PH - 1)
    want = g.open[yi, xi] * (1.1 - y / YB) * noise.smoothstep(3.0, 6.0, np.hypot(x - mx, y - my) / mr)
    keep = r.uniform(0, 1, n) < 0.5 * want
    x, y = x[keep], y[keep]
    size = np.clip(r.lognormal(-0.25, 0.45, len(x)), 0.5, 2.6)
    big = r.uniform(0, 1, len(x)) < 0.03
    size[big] = r.uniform(2.6, 3.6, big.sum())
    box.body(specks(r, x, y, size) * 0.9 * g.open, "#fbf5e2")


def palace_washes(box, r, g):
    """The palace: one pale wash of cobalt over all of it, which pooled against the lines and settled into
    the grain, deeper on the side away from the moon; the reveals of the windows and the shadows under
    each cornice and in the great arch; the lit windows in gamboge and raw sienna, the dark ones in
    indigo. No window was painted where a tree stands in front of it."""
    P = g.P
    trees = np.clip(sum(g.cmask) + g.far, 0, 1)
    lit, dark = g.lit * (1 - trees), g.dark * (1 - trees)
    m = np.clip(g.stone * (1 - lit) * (1 - dark), 0, 1)
    wet, dep = box.wash(m, 1.5, pool=0.35, rim=0.9, rim_width=3, tides=0.25, scale=110)
    cool = noise.smoothstep(AX + 400, AX - 700, XX) + 0.4 * noise.smoothstep(950, YB, YY)
    box.lay(dep * (0.24 + 0.14 * cool), **STONE)
    box.settle(dep, 0.06, ultramarine=0.7, cobalt=0.3)
    side = np.zeros((PH, PW), np.float32)
    for cx, base, w, h in P["domes"]:
        side += noise.smoothstep(cx + 0.2 * w, cx - 0.5 * w, XX) * (YY < base) * (YY > base - h - 2) * (np.abs(XX - cx) < 0.65 * w)
    for cx, hw in P["towers"]:
        side += noise.smoothstep(cx + 0.1 * hw, cx - hw, XX) * (np.abs(XX - cx) < hw + 9) * (YY > 590)
    box.lay(dep * np.clip(side, 0, 1) * m * 0.42, **SHADE)
    bands = np.zeros((PH, PW), np.float32)
    for x0, x1, y, k in P["bands"]:
        bands += noise.smoothstep(x0 - 1, x0 + 1, XX) * noise.smoothstep(x1 + 1, x1 - 1, XX) * \
            noise.smoothstep(y - 1, y + 1, YY) * noise.smoothstep(y + k + 1, y + k - 1, YY)
    box.lay(wc.charge(np.clip(bands, 0, 1) * m, wet, 3, r, streak=0.3) * 0.5, **SHADE)
    box.lay(g.reveal * m * 0.35, **SHADE)
    _, dep = box.wash(g.recess * (1 - lit) * (1 - g.facade), 0.8, pool=0.3, rim=0.8, rim_width=2, scale=60)
    box.lay(dep * 0.55, **SHADE)
    box.settle(dep, 0.1, ultramarine=0.7, cobalt=0.3)
    wet, dep = box.wash(lit * (1 - g.facade), 0.6, pool=0.3, rim=0.9, rim_width=1.5, scale=30)
    box.lay(dep * 0.5 * (1 + 0.25 * noise.field((PH, PW), 40, r)), **GOLD)
    head = lit * (1 - np.roll(g.lit, 34, axis=0))
    box.lay(wc.charge(head, wet, 8, r) * 0.32, raw_sienna=0.6, burnt_sienna=0.25, rose=0.15)
    _, dep = box.wash(dark * (1 - g.facade), 0.6, pool=0.3, rim=0.8, rim_width=1.5, scale=30)
    box.lay(dep * 0.85, ultramarine=0.4, indigo=0.45, rose=0.15)


def terrace_washes(box, r, g):
    """The terrace in moonlight: its floor and the tops of the parapet and the treads pale, the faces turned
    to us in a cool half-shadow, the wall below deeper; gold from the door spread over the floor."""
    keep = 1 - g.reserve
    lit = np.clip(g.floor * (1 - g.stone) * (1 - g.facade) + g.steps, 0, 1) * keep
    wet, dep = box.wash(lit, 1.2, pool=0.3, rim=0.7, rim_width=2, tides=0.2, scale=90)
    box.lay(dep * 0.24, **STONE)
    glow = np.exp(-((XX - AX) / 150) ** 2) * noise.smoothstep(YT + 10, YB, YY)
    box.lay(wc.charge(glow * lit, wet, 20, r) * 0.35, **GOLD)
    faces = np.clip(g.parapet + g.piers + g.risers, 0, 1) * keep
    wet, dep = box.wash(faces, 1.2, pool=0.3, rim=0.8, rim_width=2, tides=0.2, scale=90)
    box.lay(dep * 0.3, **STONE)
    box.lay(dep * 0.22, **SHADE)
    box.settle(dep, 0.06, ultramarine=0.7, cobalt=0.3)
    wet, dep = box.wash(g.wall, 1.4, pool=0.3, rim=0.6, rim_width=3, tides=0.2, scale=140)
    box.lay(dep * 0.55, **SHADE)
    box.lay(dep * 0.2, **STONE)
    box.settle(dep, 0.1, ultramarine=0.7, cobalt=0.3)


def trees(box, r, g):
    """The cypresses and the trees at the end of the terrace: a wash of indigo and Prussian blue with a little
    sap green, very dark, laid on dry paper and deepened when dry, with the side toward the moon kept a shade
    lighter; the hedge below the wall the same."""
    allc = np.clip(sum(g.cmask), 0, 1)
    for mk, (xb, yb, h, w) in zip(g.cmask, g.trees):
        wet, dep = box.wash(mk, 0.6, pool=0.2, rim=0.7, rim_width=2.5, tides=0.1, scale=70)
        moonside = noise.smoothstep(xb - 0.1 * w, xb + 0.55 * w, XX)
        box.lay(dep * (1.55 - 0.45 * moonside), **LEAF)
        box.lay(wc.charge(moonside * mk, wet, 6, r, streak=0.6) * 0.22, prussian=0.5, viridian=0.5)
        for _ in range(int(h / 14)):
            t = r.uniform(0.04, 0.85)
            x = xb + r.normal(0, 0.16) * w * (1 - t)
            y = yb - t * h
            L = r.uniform(30, 70) * (1 - 0.6 * t)
            box.stroke("flame", [(x, y, 0.8), (x + r.normal(0, 3), y - L / 2, 0.6), (x + r.normal(0, 5), y - L, 0.05)],
                       r.uniform(3, 6), load=0.8, dryness=1.3, reach=L * 1.5, clumps=3, bristles=30)
    box.lay(box.take("flame") * allc * 0.5, indigo=0.6, sepia=0.2, prussian=0.2)
    wet, dep = box.wash(g.far, 1.0, pool=0.3, rim=0.7, rim_width=3, scale=90)
    box.lay(dep * 1.25, **LEAF)
    wet, dep = box.wash(g.hedge, 0.8, pool=0.3, rim=0.8, rim_width=3, tides=0.2, scale=80)
    box.lay(dep * 1.3, **LEAF)


def pine_washes(box, r, g):
    """The pine: the bark one pale wash of cobalt and sepia, with blue run into the side away from the moon;
    the crowns indigo and Prussian blue, very dark and flat, and gone over again when dry where one cloud of
    needles lies behind another."""
    wet, dep = box.wash(g.bark, 0.5, pool=0.25, rim=0.7, rim_width=2, tides=0.1, scale=80)
    box.lay(dep * 0.36, cobalt=0.5, sepia=0.3, yellow_ochre=0.12, rose=0.08)
    gy, gx = np.gradient(ndimage.gaussian_filter(g.bark, 5))
    away = np.clip(ndimage.gaussian_filter(gx * 0.7 - gy * 0.7, 2) * 18, 0, 1) * g.bark   # the side the moon misses
    box.lay(wc.charge(away, wet, 4, r, streak=0.5) * 0.5, **SHADE)
    wet, dep = box.wash(g.crowns, 0.8, pool=0.25, rim=0.8, rim_width=2.5, tides=0.15, scale=70)
    box.lay(dep * 1.25, indigo=0.42, prussian=0.36, sap=0.14, sepia=0.08)
    box.lay(wc.charge(noise.smoothstep(0.3, 1.2, noise.fbm((PH, PW), 70, r, octaves=2)) * g.crowns, wet, 12, r) * 0.25,
            viridian=0.5, prussian=0.5)
    _, dep = box.wash(g.inner, 0.7, pool=0.2, rim=0.9, rim_width=2, scale=50)
    box.lay(dep * 0.55, indigo=0.6, prussian=0.4)


def ground(box, r, g):
    """The lawns in three washes, each let dry before the next. First a clear blue green of viridian and
    Prussian blue laid side to side over all of it, thinner in the middle distance where the moon lies on
    the grass; then, the sheet washed down, ultramarine and Prussian blue, which granulate, carried deeper
    towards us, along the foot of the terrace and out at the sides; and last the long shadows of the two
    great cypresses laid across the grass and the walk towards us. The walk is a pale wash of cobalt and
    ultramarine with a little rose that pooled and settled into the grain, paler where it goes away into
    the moonlight and deepened towards us with blue run into it while it was wet; its borders are small
    upright touches of the dark of the trees, and the basin throws a short shadow on it."""
    lawn, walk = g.lawn, g.walk
    near = noise.smoothstep(YS + 120, PH, YY)
    foot = np.exp(-((YY - YS) / 120) ** 2)
    side = noise.smoothstep(460, 0, XX) + 0.6 * noise.smoothstep(AX + 360, PW, XX)
    lit = np.exp(-((YY - 1620) / 230) ** 2) * np.exp(-((XX - AX) / 560) ** 2)
    soft = noise.fbm((PH, PW), 360, r, octaves=3)
    band = box.banded(YS - 40, PH, 0, PW, 60)
    wet, dep = box.wash(lawn, 1.8, pool=0.3, rim=0.5, rim_width=4, tides=0.2, scale=220)
    box.lay(dep * (0.5 + 0.16 * band) * (1 - 0.4 * lit) * (1 + 0.08 * soft), viridian=0.48, prussian=0.4, cobalt=0.12)
    box.lay(wc.charge(lit * lawn, wet, 60, r, streak=0.3) * 0.1, sap=0.6, viridian=0.4)
    box.scrub(0.3, lawn * (0.6 + 0.4 * noise.smoothstep(-1, 1.2, noise.fbm((PH, PW), 200, r, octaves=2))))
    deep = np.clip(0.2 + 0.55 * near + 0.5 * foot + 0.35 * side + 0.12 * soft - 0.25 * lit, 0.05, 1.15)
    wet, dep = box.wash(lawn, 1.6, pool=0.35, rim=0.6, rim_width=4, tides=0.25, scale=180)
    box.lay(dep * deep * (0.85 + 0.25 * band) * 0.72, ultramarine=0.55, prussian=0.3, indigo=0.15)
    box.settle(dep * deep, 0.12, ultramarine=0.8, cobalt=0.2)
    wet, dep = box.wash(walk, 2.0, pool=0.32, rim=0.7, rim_width=3, tides=0.25, scale=150)
    away = noise.smoothstep(2100, YS, YY)
    box.lay(dep * (0.34 - 0.12 * away) * (0.85 + 0.3 * band), cobalt=0.6, ultramarine=0.25, rose=0.12, yellow_ochre=0.03)
    box.settle(dep, 0.08, ultramarine=0.7, cobalt=0.3)
    _, dep = box.wash(g.border, 0.6, pool=0.3, rim=0.8, rim_width=1.5, scale=40)
    box.lay(dep * (0.9 + 0.3 * near), **LEAF)
    shade = []
    for xb, yb, h, w in g.trees[:2]:
        L = 1.2 * h
        a = np.deg2rad(110)
        dx, dy = np.cos(a), np.sin(a)
        nrm = np.array([-dy, dx])
        t = np.linspace(0, 1, 60)
        hw = 0.5 * w * (1 + 1.9 * t) * np.clip((1 - t) / 0.8, 0, 1) ** 0.75 * (1 + 0.06 * noise.line1d(60, 6, r))
        mid = np.stack([xb + dx * L * t, yb + dy * L * t], 1)
        shade.append(np.vstack([mid + nrm * hw[:, None], (mid - nrm * hw[:, None])[::-1]]))
    sh = fill(shade, 1.0) * g.ground
    wet, dep = box.wash(sh, 2.0, pool=0.3, rim=0.7, rim_width=3, tides=0.2, scale=120)
    box.lay(dep * walk * 0.5, **SHADE)
    box.lay(dep * (1 - walk) * 0.38, ultramarine=0.5, indigo=0.3, prussian=0.2)
    box.settle(dep, 0.1, ultramarine=0.8, cobalt=0.2)
    _, dep = box.wash(g.cast * g.ground, 1.2, pool=0.3, rim=0.7, rim_width=2, scale=80)
    box.lay(dep * 0.45, **SHADE)
    near = walk * noise.smoothstep(2000, PH, YY + 60 * noise.fbm((PH, PW), 120, r, octaves=2))
    box.lay(wc.charge(near, np.clip(walk, 0, 1), 40, r, streak=0.3) * (0.8 + 0.3 * band) * 0.45, **SHADE)


def fountain(box, r, g):
    """The basin's rim and the bowl pale stone, a wash of cobalt that pooled against the lines; the face of
    the curb towards us, which the moon does not reach, deeper towards its foot, and the inside of the far
    wall dark across the water; the water a deep wash that holds the sky, with the gold of the door broken
    in it; the jet and the falling water Chinese white, dragged and dotted on last."""
    cx, cy, rx, ry = BASIN
    keep = 1 - g.reserve
    rim = np.clip(g.basin - g.water, 0, 1)
    stone = np.clip(rim + g.curb + g.column + g.bowl, 0, 1) * keep
    wet, dep = box.wash(stone, 1.4, pool=0.35, rim=0.9, rim_width=2.5, tides=0.2, scale=90)
    box.lay(dep * 0.24, **STONE)
    box.settle(dep, 0.07, ultramarine=0.7, cobalt=0.3)
    box.lay(dep * noise.smoothstep(cx + 4, cx - 22, XX) * np.clip(g.column + g.bowl, 0, 1) * 0.3, **SHADE)
    _, dep = box.wash(g.curb * keep, 1.2, pool=0.3, rim=0.8, rim_width=2.5, tides=0.2, scale=80)
    box.lay(dep * (0.25 + 0.3 * noise.smoothstep(cy, cy + ry + 36, YY)), **SHADE)
    water = g.water * (1 - g.column) * (1 - g.bowl) * keep
    band = box.banded(cy - ry, cy + ry, cx - rx, cx + rx, 26, "pool")
    wet, dep = box.wash(water, 1.0, pool=0.25, rim=0.7, rim_width=3, tides=0.2, scale=110)
    box.lay(dep * (0.8 + 0.35 * band), prussian=0.45, ultramarine=0.35, indigo=0.2)
    box.settle(dep, 0.1, ultramarine=0.8, cobalt=0.2)
    _, dep2 = box.wash(g.inwall * water, 0.6, pool=0.2, rim=0.6, rim_width=2, scale=40)
    box.lay(dep2 * 0.6, **SHADE)
    streaks = noise.smoothstep(0.4, 1.2, noise.stretched((PH, PW), 5, 40, r)) * np.exp(-((XX - cx) / 80) ** 2) * \
        noise.smoothstep(cy + 40, cy - ry, YY)
    box.scrub(0.8, streaks * water)
    box.lay(wc.charge(streaks * water, wet, 3, r) * 0.45, **GOLD)
    # the jet: a column of white rising from the bowl, opening at its head into a crown of water that falls
    # back in broken arcs, with spray round the head and drops below; ripples where it falls into the basin
    top = 1300.0
    ys = np.linspace(BOWL - 6, top, 14)
    box.stroke("jet", np.stack([cx + 1.2 * noise.line1d(14, 4, r), ys, np.linspace(1.0, 0.5, 14)], 1), 4.2,
               load=1.0, reach=4000, dryness=0.5, clumps=3, bristles=24, head=0.2)
    drops = np.zeros((PH, PW), np.float32)
    lean = r.normal(0, 0.15)                                        # a breath of wind
    for _ in range(28):
        s = r.choice([-1, 1])
        far = r.uniform(20, 125) * (1 + lean * s)
        rise, y0 = r.uniform(8, 38), top + r.uniform(-4, 16)
        land = BOWL - 2 if far < 90 else cy - 0.3 * ry
        x = np.linspace(0, far, 50)
        xm = 0.2 * far                                              # the water goes on up a little before it turns
        py = np.where(x < xm, y0 - rise * (1 - ((xm - x) / xm) ** 2),
                      y0 - rise + (land - y0 + rise) * ((x - xm) / (far - xm)) ** 2)
        px = cx + s * x
        n = int(r.uniform(0.35, 1.0) * len(x))                      # most arcs break into drops before they land
        box.stroke("jet", np.stack([px[:n:3], py[:n:3], np.linspace(0.85, 0.2, len(px[:n:3]))], 1),
                   r.uniform(1.0, 1.8), load=0.7, reach=far * 2, dryness=1.4, clumps=2, bristles=8, head=0.1)
        sel = (r.uniform(0, 1, len(x)) < 0.22) & (np.arange(len(x)) >= 0.6 * n)
        drops[py[sel].astype(int).clip(0, PH - 1), px[sel].astype(int).clip(0, PW - 1)] += r.uniform(0.5, 1, sel.sum())
    k = 160
    sx, sy = cx + r.normal(0, 26, k), top - 18 + r.normal(0, 14, k)
    drops[sy.astype(int).clip(0, PH - 1), sx.astype(int).clip(0, PW - 1)] += r.uniform(0.3, 0.9, k)
    jet = np.clip(box.take("jet") * 1.2, 0, 1) + np.clip(ndimage.gaussian_filter(drops, 0.8) * 3, 0, 1)
    ripple = noise.smoothstep(0.6, 1.4, noise.stretched((PH, PW), 30, 3, r)) * \
        noise.smoothstep(1.0, 0.3, np.hypot((XX - cx) / 150, (YY - cy) / 40)) * water
    box.body(np.clip(jet, 0, 0.92) + 0.5 * ripple, "#eef0ec")


def roses(box, r, g):
    """The rose bush: the mound of leaves a dark wash of indigo, Prussian blue and sap green, a little lighter
    along its top where the moon catches it, with gold run into it round the lantern, and deepened leaf by
    leaf when dry; the leaves the lantern lights an olive gold, and those in front of the flowers a lighter
    green. The flowers are rose madder, the buds deepest with crimson and the open flowers paler, as they
    age; crimson in the creases where one petal lies over another and at the heart, their lips left pale,
    warmed with gold on the side towards the lantern and cooled with blue on the side away from the light."""
    lx, ly = LAMP
    dist = np.hypot(XX - lx, (YY - ly - 30) * 1.1) + 40 * noise.fbm((PH, PW), 60, r, octaves=2)
    lamp = noise.smoothstep(380, 40, dist)
    front = g.front * (1 - g.litleaves)
    leaves = g.mound * (1 - g.litleaves) * (1 - g.roses) * (1 - g.lamp) * (1 - front)
    wet, dep = box.wash(leaves, 0.8, pool=0.3, rim=0.8, rim_width=2.5, tides=0.15, scale=80)
    low = noise.smoothstep(1450, 2300, YY)
    box.lay(dep * (0.95 + 0.3 * low - 0.55 * lamp), **LEAF)
    box.lay(wc.charge(noise.smoothstep(1750, 1420, YY) * leaves, wet, 30, r, streak=0.4) * 0.2, prussian=0.5, viridian=0.5)
    box.lay(wc.charge(lamp * leaves, wet, 25, r, streak=0.4) * 0.45, raw_sienna=0.5, gamboge=0.3, burnt_sienna=0.2)
    _, dep = box.wash(g.shadeleaves * leaves, 0.4, pool=0.25, rim=0.9, rim_width=1.5, scale=30)
    box.lay(dep * (0.55 + 0.25 * low) * (1 - lamp), indigo=0.55, prussian=0.35, sepia=0.1)
    _, dep = box.wash(front, 0.5, pool=0.25, rim=0.9, rim_width=1.5, scale=30)
    box.lay(dep * 0.75, prussian=0.38, viridian=0.32, sap=0.2, indigo=0.1)
    box.lay(dep * (1 - lamp) * 0.25, **LEAF)
    wet, dep = box.wash(g.litleaves, 0.4, pool=0.25, rim=0.9, rim_width=1.5, scale=30)
    box.lay(dep * 0.75, sap=0.32, raw_sienna=0.28, gamboge=0.15, indigo=0.15, prussian=0.1)
    box.lay(dep * (1 - lamp) * 0.6, **LEAF)
    wet, dep = box.wash(g.roses, 0.4, pool=0.25, rim=0.8, rim_width=1.5, scale=30)
    t, d = g.tone, g.deep
    box.lay(dep * (0.42 - 0.3 * g.lip) * t, rose=0.8 - 0.5 * d, crimson=0.2 + 0.5 * d)
    box.lay(g.crease * 0.55 * t, crimson=0.6, rose=0.25, ultramarine=0.15)
    box.lay(wc.charge(g.heart, wet, 4, r) * 0.45 * t, crimson=0.75, rose=0.25)
    box.lay(dep * lamp * (0.38 - 0.3 * g.side), gamboge=0.65, raw_sienna=0.25, rose=0.1)
    box.lay(dep * (0.1 + 0.5 * g.side) * (1 - 0.5 * lamp), **SHADE)


def lantern(box, r, g):
    """The lantern: its pierced brass raw and burnt sienna, deeper on the side away from us, its glass the
    paper with a breath of gamboge, a touch of Chinese white for the flame and points of gold where the
    light comes through the piercings. Round it the dark of the leaves and the grass was carried up thin
    and gold run into the thin place, so that the light seems to spill out over them; then the iron crook
    it hangs from, which stays dark against the light."""
    lx, ly = LAMP
    brass = g.lamp * (1 - g.glass)
    wet, dep = box.wash(brass, 0.4, pool=0.25, rim=0.9, rim_width=1.5, scale=20)
    box.lay(dep * 0.65, raw_sienna=0.45, burnt_sienna=0.35, gamboge=0.2)
    box.lay(wc.charge(noise.smoothstep(lx + 10, lx - 30, XX) * g.lamp, wet, 4, r) * 0.4, sepia=0.6, burnt_sienna=0.4)
    d = np.hypot(XX - lx, (YY - ly) * 1.08) + 16 * noise.fbm((PH, PW), 40, r, octaves=3)
    orb = noise.smoothstep(175, 28, d) ** 1.5 * (1 - brass)
    thin = orb * (0.85 + 0.15 * box.hill)
    for k, dens in box.d.items():
        if k not in WARM:
            dens *= 1 - 0.93 * thin
    box.lay(orb * 0.5 * (1 + 0.15 * noise.field((PH, PW), 30, r)), gamboge=0.58, raw_sienna=0.24, rose=0.18)
    box.lay(noise.smoothstep(330, 60, d) * (1 - brass) * 0.14, gamboge=0.4, raw_sienna=0.3, rose=0.3)
    wet, dep = box.wash(g.glass, 0.3, pool=0.3, rim=0.6, rim_width=1.5, scale=20)
    box.lay(dep * 0.2 * noise.smoothstep(4, 34, np.hypot(XX - lx, YY - ly - 6)), gamboge=0.8, raw_sienna=0.2)
    box.body(specks(r, [lx], [ly + 8], [4.5], squash=(0.5, 0.55)) * 0.9, "#fff6d8")
    xs, ys = [], []
    for f, n in ((0.16, 6), (0.36, 5), (0.56, 3)):
        half = 40 * ONION(f) * 0.72
        xs += list(lx + np.linspace(-half, half, n))
        ys += [ly - 44 - 66 * f] * n
    for k in range(5):
        xs.append(lx + (k - 2) * 9.0)
        ys.append(ly + 56)
    box.body(specks(r, xs, ys, np.full(len(xs), 1.3)) * 0.85, "#f8d677")
    box.lay(g.iron * 1.0, indigo=0.6, sepia=0.4)


def flowers(box, r, g):
    """Small white flowers along the borders of the walk and in the grass on the left, each a few touches of
    Chinese white round a dot of gold, in clumps: bigger and fewer near us, a dust of them further off."""
    xs, ys, ss = [], [], []
    for _ in range(30):
        y = r.uniform(YS + 60, PH - 10)
        depth = (y - HZ) / (PH - HZ)
        half = 118 * (y - HZ) / (YS - HZ)
        if r.uniform() < 0.7:
            x = AX + r.choice([-1, 1]) * half * r.uniform(1.02, 1.12)
        else:
            x = r.uniform(60, AX - half)
        k = int(r.integers(4, 13))
        spread = 10 + 60 * depth
        xs.append(x + r.normal(0, spread, k))
        ys.append(y + r.normal(0, 0.35 * spread, k))
        ss.append(np.full(k, 1.2 + 5.0 * depth ** 1.3) * r.uniform(0.8, 1.2, k))
    x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)
    xi, yi = x.astype(int).clip(0, PW - 1), y.astype(int).clip(0, PH - 1)
    room = np.clip(g.lawn + g.border, 0, 1) * (1 - g.bush) * (1 - g.bark)
    ok = room[yi, xi] > 0.5
    x, y, s = x[ok], y[ok], s[ok]
    petals = np.zeros((PH, PW), np.float32)
    for k in range(5):
        a = 2 * np.pi * k / 5 + r.uniform(0, 2 * np.pi, len(x))
        petals = np.maximum(petals, specks(r, x + 0.55 * s * np.cos(a), y + 0.45 * s * np.sin(a), 0.42 * s))
    box.body(petals * 0.85, "#eeede4")
    box.body(specks(r, x, y, 0.25 * s + 0.3) * 0.85, "#e6b84a")


def drawing(g, pen):
    """The pen lines, each taken out where something drawn in front of it stands."""
    trees = np.clip(sum(g.cmask) + g.far + g.hedge, 0, 1)
    near = np.clip(g.bark + g.crowns + g.bush + g.reserve, 0, 1)
    grow = lambda m, k: ndimage.grey_dilation(np.clip(m, 0, 1), k)
    hidden = {0: grow(g.facade + trees + near, 3),
              1: grow(trees + near, 3),
              2: grow(near, 3),
              3: np.clip(ndimage.grey_erosion(g.bark, 7) + g.crowns + g.bush, 0, 1),
              4: grow(g.roses + g.lamp + g.front + g.iron, 3),
              5: grow(g.front + g.lamp, 3)}
    return pen.ink(hidden)


def paper(r):
    """The paper of the original: a smooth rag paper with a fine tooth, as the plate shows it."""
    s = wash.rough((PH, PW), _s(r), tint="#f4ecd9", margin=0, hill=3.0)
    return Sheet(s.color, s.tooth, s.fiber, np.ones((PH, PW), np.float32))


def page(r):
    """The page of the book: a heavier cream paper trimmed straight, a shade darker towards its edges."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x0, y0, x1, y1 = PAGE
    d = np.minimum.reduce([xx - x0, x1 - xx, yy - y0, y1 - yy])
    tone = (1 + 0.012 * noise.fbm((H, W), 500, r, octaves=4) + 0.01 * noise.field((H, W), 1.5, r)
            - 0.035 * np.exp(-np.clip(d, 0, None) / 70))
    z = np.zeros((H, W), np.float32)
    return Sheet((lin("#ebe0c6")[None, None] * tone[..., None]).astype(np.float32), z, z,
                 noise.smoothstep(-0.8, 0.8, d).astype(np.float32))


def paint(seed=5):
    r = noise.rng(seed)
    sheet = paper(r)
    pen = Pen(r)
    g = Garden(r, pen)
    box = Box(sheet, r)
    sky(box, r, g)
    moon(box, r, g)
    palace_washes(box, r, g)
    terrace_washes(box, r, g)
    trees(box, r, g)
    ground(box, r, g)
    fountain(box, r, g)
    pine_washes(box, r, g)
    roses(box, r, g)
    lantern(box, r, g)
    box.lay(drawing(g, pen), ink=1.0)
    flowers(box, r, g)
    stars(box, r, g)
    img = wash.cockle(box.glaze(), sheet, box.soak, r, buckle=1.2, grain=0.2)
    mount = page(r)
    out = mount.color.copy()
    sl = (slice(PY, PY + PH), slice(PX, PX + PW))
    edge = np.zeros((H, W), np.float32)
    edge[sl] = 1
    lift = ndimage.gaussian_filter(np.roll(edge, (3, 2), (0, 1)), 2.5) * 0.22 * (1 - edge)
    out *= (1 - lift)[..., None]
    out[sl] = img
    return plate.mount(out, mount)
