"""The Bay through the Pines. Colour lithograph poster.

A railway poster for the Côte d'Azur, drawn in the manner Roger Broders used for the
Paris–Lyon–Méditerranée between the wars. From a rocky slope above the sea the view goes out under
an umbrella pine to a deep blue bay. The pine grows out of a red outcrop on the left, its trunk
swinging out over the water and its crown spreading across the top of the sheet toward the sun; a
smaller pine stands on the rocks across the cove. A white town climbs a headland under lavender
mountains, and the sun over the sea gilds the side of every crown that faces it and lays a road of
glitter on the water. In the foreground, in the shade of the pines, agaves and bushes of the maquis
grow among blocks of the red porphyry of the Esterel.

Each colour was drawn on a stone of its own and the stones were printed one after another, seven in
all: yellow, rose, orange, light blue, green, ultramarine and a dark navy. The flats are tusche laid
with a brush, so their edges are drawn and not ruled, and the lighter colours run a little way under
the darker. The rocks are cut into planes, gold on top, orange on the faces turned to the sun and
red on those turned away, and each plane is modelled in greasy crayon, which catches only the peaks
of the stone's grain and breaks into a fine stipple, thicker as the plane turns from the light; navy
lies in the undercuts. The ground in the shade is rose grained with ultramarine. The agaves are pale
blue-green, each leaf folded into a lit half and a dark one, with a margin printed from the yellow
stone alone. The sky is a flat blue, stopped out round the sun and grained with ultramarine crayon
toward the top; the glare round the sun is yellow and rose spattered from a brush, and the glitter
was scraped out of the two blue stones with a needle. No stone fell quite in register, so a hair of
a third colour, or of paper, runs along the joins. The lettering was drawn on the stones with the
picture, letter by letter, and the title given a shadow on the navy stone. The sheet has been folded
for the post and has yellowed a little.
"""

from math import comb

import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate, ndimage, special

from atelier import noise, plate, relief
from atelier.color import glaze, lin, pigment
from atelier.paper import Sheet

TITLE = "The Bay through the Pines"
DATE = "2026"
MEDIUM = "Colour lithograph poster, printed from seven stones on machine-made paper"
AFTER = ("Roger Broders, the travel posters for the Paris–Lyon–Méditerranée railway, 1920s–1930s: Menton, "
         "c. 1923; Agay, 1928; Antibes, c. 1928; Sainte-Maxime; Villefranche-sur-Mer")
ROOM = "The Workshop"
YEAR = 1930
PLACE = "Paris"
REGION = "Europe"
NOTE = ("A new poster for the Côte d'Azur: an umbrella pine leaning out over a deep blue bay, the sun on the "
        "water and a white town on its headland, with agaves and red rocks cut into planes below. Seven stones, "
        "drawn in tusche, crayon and spatter and printed a hair out of register.")

PW, PH = 2000, 3100
SHAPE = (PH, PW)
EDGE = 34                                   # the wall shows this far round the sheet
FRAME = (80, 80, PW - 80, PH - 80)          # the centre of the ruled border round the whole design
PIC = (110, 110, PW - 110, 2440)            # the picture's box
HZ = 860.0                                  # the horizon
SUN = (1165.0, 600.0, 74.0)                 # its centre and radius
LIGHT = np.array([0.45, -0.85, 0.3]) / np.linalg.norm([0.45, -0.85, 0.3])   # toward the sun, for the rocks

PAPER = "#f2e8cf"
INKS = dict(yellow=pigment("#f6d360"), rose=pigment("#e9a6c8"), orange=pigment("#ea6a2a"),
            blue=pigment("#8ccaf2"), green=pigment("#2e6a49"), ultra=pigment("#2a5ed2"), navy=pigment("#262c5c"))
ORDER = ("yellow", "rose", "orange", "blue", "green", "ultra", "navy")

Y, X = np.mgrid[0:PH, 0:PW].astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# drawing on the stone


def raster(polys, ss=2, lines=(), region=(0, 0, PW, PH)):
    """Polygons (and polylines with widths) filled into a mask over the plate, anti-aliased. Only the
    `region` (x0, y0, x1, y1) is drawn."""
    rx0, ry0, rx1, ry1 = region
    im = Image.new("L", ((rx1 - rx0) * ss, (ry1 - ry0) * ss), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([((x - rx0) * ss, (y - ry0) * ss) for x, y in p], fill=255)
    for p, w in lines:
        d.line([((x - rx0) * ss, (y - ry0) * ss) for x, y in p], fill=255, width=max(1, round(w * ss)), joint="curve")
    out = np.zeros(SHAPE, np.float32)
    out[ry0:ry1, rx0:rx1] = np.asarray(im.resize((rx1 - rx0, ry1 - ry0), Image.BOX), np.float32) / 255
    return out


class Layers:
    """Shapes drawn one over another into `n` masks over the `region` of the plate: each shape goes into
    its own mask and, unless told not to cover, blots out whatever earlier shapes put beneath it in
    the others."""

    def __init__(self, n, region, ss=2):
        self.region, self.ss = region, ss
        x0, y0, x1, y1 = region
        self.ims = [Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0) for _ in range(n)]
        self.dr = [ImageDraw.Draw(im) for im in self.ims]

    def put(self, k, pts, cover=True):
        x0, y0 = self.region[:2]
        q = [((x - x0) * self.ss, (y - y0) * self.ss) for x, y in pts]
        for j, d in enumerate(self.dr):
            if cover or j == k:
                d.polygon(q, fill=255 if j == k else 0)

    def masks(self):
        x0, y0, x1, y1 = self.region
        out = []
        for im in self.ims:
            m = np.zeros(SHAPE, np.float32)
            m[y0:y1, x0:x1] = np.asarray(im.resize((x1 - x0, y1 - y0), Image.BOX), np.float32) / 255
            out.append(m)
        return out


def box(mask, pad=8):
    """The slice of the plate where `mask` is drawn."""
    ys, xs = np.nonzero(mask > 0.002)
    if len(ys) == 0:
        return None
    return (slice(max(0, ys.min() - pad), min(PH, ys.max() + pad + 1)),
            slice(max(0, xs.min() - pad), min(PW, xs.max() + pad + 1)))


def ahead(a, dx, dy, fill):
    """`a` looked up `dx`, `dy` pixels away: out[y, x] = a[y + dy, x + dx]."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


class Stone:
    """One grained stone. Grease is all or nothing: the stone takes ink where tusche or crayon lies and
    refuses it everywhere else, so every tone on it is made of how much of the grain the crayon or the
    spatter has covered."""

    def __init__(self, r):
        self.r = r
        g = 0.8 * noise.field(SHAPE, 1.0, r) + 0.45 * noise.field(SHAPE, 2.4, r)
        self.grain = special.ndtr(g / g.std()).astype(np.float32)      # even in [0, 1]: the peaks of the grain
        self.solid = np.zeros(SHAPE, np.float32)
        self.ruled = np.zeros(SHAPE, np.float32)
        self.dots = np.zeros(SHAPE, np.float32)
        self.cut = np.zeros(SHAPE, np.float32)

    def tusche(self, mask):
        """A flat brushed in tusche (its edge is given its waver once the whole stone is drawn)."""
        np.maximum(self.solid, mask, out=self.solid)

    def rule(self, mask):
        """A line drawn in tusche with the ruling pen: straight, but its edge still the brush's."""
        np.maximum(self.ruled, mask, out=self.ruled)

    def crayon(self, tone, angle=0.8, width=3.0, length=5.0, press=0.18, firm=0.1):
        """Shading in litho crayon, strokes laid side by side along `angle`. The crayon only touches the
        peaks of the grain, harder in the middle of each stroke, so a tone comes out as a stipple; bearing
        down hard, it fills the grain solid."""
        sl = box(tone)
        if sl is None:
            return
        t = tone[sl]
        h, w = t.shape
        c, s = np.cos(angle), np.sin(angle)
        yy, xx = Y[sl] - sl[0].start, X[sl] - sl[1].start
        across, along = -xx * s + yy * c, xx * c + yy * s
        n = int(np.hypot(h, w)) + 4
        f = noise.stretched((n, n), width, width * length, self.r).T     # strokes run along the second axis
        f = ndimage.map_coordinates(f, [across + n / 2, along], order=1, mode="wrap")
        p = t + 4 * t * (1 - t) * (press * f + 0.1 * noise.field((h, w), 40, self.r))
        dep = noise.smoothstep(1 - p - firm, 1 - p + firm, self.grain[sl])
        self.dots[sl] = np.maximum(self.dots[sl], dep * (t > 0.003))

    def spatter(self, density, rmin=0.6, rmax=5.0, tail=2.3, flick=(0.35, 0.3)):
        """Tusche flicked from a stiff brush over a screen: drops of every size, most of them tiny, a few
        drawn out along the flick. `density` is the share of the stone they should cover."""
        sl = box(density)
        if sl is None:
            return
        r = self.r
        dens = np.clip(density[sl], 0, 0.95)
        h, w = dens.shape
        rad = np.clip(rmin * r.random(20000) ** (-1 / tail), 0, rmax)
        lam = -np.log(1 - dens) / (np.pi * np.mean(rad ** 2))
        n = r.poisson(lam.sum())
        if n == 0:
            return
        idx = r.choice(lam.size, n, p=(lam / lam.sum()).ravel())
        ys, xs = np.divmod(idx, w)
        ys, xs = ys + r.random(n), xs + r.random(n)
        rad = np.clip(rmin * r.random(n) ** (-1 / tail), 0, rmax)
        ss = 3
        im = Image.new("L", (w * ss, h * ss), 0)
        d = ImageDraw.Draw(im)
        ang = flick[0] + r.normal(0, flick[1], n)
        el = 1 + r.exponential(0.25, n)
        for x, y, q, a, e in zip(xs * ss, ys * ss, rad * ss, ang, el):
            if e > 1.35:                                     # a drop drawn out: two overlapping beads
                dx, dy = np.cos(a) * q * (e - 1), np.sin(a) * q * (e - 1)
                d.ellipse([x - q + dx, y - q + dy, x + q + dx, y + q + dy], fill=255)
                d.ellipse([x - q - dx, y - q - dy, x + q - dx, y + q - dy], fill=255)
            else:
                d.ellipse([x - q, y - q, x + q, y + q], fill=255)
        m = np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255
        self.dots[sl] = np.maximum(self.dots[sl], m)

    def scrape(self, mask):
        """Grease taken off again with a needle or the blade of a knife."""
        np.maximum(self.cut, mask, out=self.cut)

    def clean(self, mask):
        """The stone washed out where a shape drawn later is to print on bare paper."""
        self.solid *= 1 - mask
        self.dots *= 1 - mask

    def drawing(self, amp=1.3):
        """What the stone holds once it is drawn: the tusche flats given the waver of a brush, the crayon
        and the spatter, less what was scraped away."""
        r = self.r
        dx = amp * (noise.field(SHAPE, 70, r) + 0.4 * noise.field(SHAPE, 11, r))
        dy = amp * (noise.field(SHAPE, 70, r) + 0.4 * noise.field(SHAPE, 11, r))
        t = ndimage.map_coordinates(self.solid, [Y + dy, X + dx], order=1, mode="nearest")
        t = noise.smoothstep(0.3, 0.7, np.maximum(t, self.ruled) + 0.1 * noise.field(SHAPE, 1.2, r))
        return np.maximum(t, self.dots) * (1 - self.cut)


def pull(img, drawing, ink, sheet, r, shift, turn):
    """One run through the press: the stone damped and rolled up with ink, the sheet laid to the marks
    (never quite exactly) and pulled. A litho flat is smooth but never dead even, and the ink misses the
    deepest pits of the paper."""
    c = np.array(SHAPE, np.float64) / 2
    m = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    drawing = ndimage.affine_transform(drawing, m, c - m @ c - np.array(shift[::-1]), order=1)
    film = 1 + 0.05 * noise.fbm(SHAPE, 260, r, octaves=4) + 0.03 * noise.field(SHAPE, 3.0, r)
    catch = 0.68 + 0.32 * noise.smoothstep(0.03, 0.2, sheet.tooth)      # the deepest pits of the sheet stay bare
    dens = ndimage.gaussian_filter(np.clip(drawing, 0, 1) * film * catch, 0.55)
    return glaze(img, dens, ink)


# ---------------------------------------------------------------------------------------------------
# the paper


def poster_paper(r):
    """Machine-made poster paper, smooth and a little cloudy, gone cream with age and darker toward the
    edges where the light reached it longest. Trimmed straight, with a nick or two."""
    formation = noise.fbm(SHAPE, 300, r, octaves=5, gain=0.55)
    tooth = 0.6 * noise.field(SHAPE, 1.0, r) + 0.4 * noise.field(SHAPE, 2.6, r) + 0.2 * noise.field(SHAPE, 8, r)
    tooth = noise.smoothstep(-2.2, 2.2, tooth)
    x0, y0, x1, y1 = EDGE, EDGE, PW - EDGE, PH - EDGE
    d = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])
    toned = noise.smoothstep(160, 0, d) * (0.6 + 0.4 * noise.field(SHAPE, 90, r))
    light = 1 + 0.014 * formation + 0.01 * (tooth - 0.5)
    color = lin(PAPER)[None, None] * light[..., None]
    color *= 1 - np.clip(toned, 0, 1)[..., None] * np.array([0.03, 0.06, 0.12], np.float32)
    wob = 0.6 * noise.field(SHAPE, 30, r) + 0.25 * noise.field(SHAPE, 3, r)
    alpha = noise.smoothstep(-0.8, 0.8, d + wob)
    for _ in range(5):                                   # nicks in the trimmed edge
        t, q = r.uniform(0.1, 0.9), r.uniform(3, 9)
        cx, cy = [(x0 + t * (x1 - x0), y0), (x0 + t * (x1 - x0), y1), (x0, y0 + t * (y1 - y0)),
                  (x1, y0 + t * (y1 - y0))][r.integers(4)]
        alpha *= noise.smoothstep(0.6 * q, 1.2 * q, np.hypot(X - cx, Y - cy))
    return Sheet(color.astype(np.float32), tooth.astype(np.float32), tooth.astype(np.float32), alpha)


def folds(img, sheet, r):
    """The poster was folded for the post: in half down its length and in three across. Each crease
    throws a faint ridge of light and shade and has cracked the ink along it."""
    for axis, at in ((1, PW / 2 + r.normal(0, 6)), (0, PH / 3 + r.normal(0, 8)), (0, 2 * PH / 3 + r.normal(0, 8))):
        coord = X if axis == 1 else Y
        line = at + 1.5 * noise.line1d(PH if axis == 1 else PW, 300, r)
        d = coord - (line[:, None] if axis == 1 else line[None, :])
        ridge = 1 + 0.03 * np.tanh(d / 3.0) * np.exp(-(d / 9.0) ** 2) - 0.01 * np.exp(-(d / 2.0) ** 2)
        crack = np.exp(-(d / 1.3) ** 2) * noise.smoothstep(0.55, 0.9, noise.field(SHAPE, 1.6, r) * 0.5 + 0.5)
        img = img * ridge[..., None]
        img = img + (sheet.color * 0.97 - img) * np.clip(crack, 0, 1)[..., None] * 0.7
    return img


# ---------------------------------------------------------------------------------------------------
# the lettering


def arc(cx, cy, rx, ry, a0, a1, n=28):
    a = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(a), cy - ry * np.sin(a)))


# capitals of a lettering artist's sans, as skeletons on a cap height of 1 (y down): advance, strokes
GLYPHS = {
    "A": (0.80, [[(0.07, 0.93), (0.40, 0.07), (0.73, 0.93)], [(0.22, 0.66), (0.58, 0.66)]]),
    "C": (0.76, [arc(0.42, 0.5, 0.34, 0.43, 50, 310)]),
    "D": (0.78, [[(0.08, 0.07), (0.08, 0.93)],
                 [(0.08, 0.07), (0.32, 0.07)] + arc(0.32, 0.5, 0.38, 0.43, 90, -90) + [(0.08, 0.93)]]),
    "E": (0.60, [[(0.54, 0.07), (0.08, 0.07), (0.08, 0.93), (0.54, 0.93)], [(0.08, 0.5), (0.46, 0.5)]]),
    "I": (0.17, [[(0.085, 0.07), (0.085, 0.93)]]),
    "L": (0.56, [[(0.08, 0.07), (0.08, 0.93), (0.52, 0.93)]]),
    "M": (0.92, [[(0.08, 0.93), (0.10, 0.07), (0.46, 0.74), (0.82, 0.07), (0.84, 0.93)]]),
    "N": (0.76, [[(0.08, 0.93), (0.08, 0.07), (0.68, 0.93), (0.68, 0.07)]]),
    "O": (0.90, [arc(0.45, 0.5, 0.37, 0.43, 0, 360, 56)]),
    "P": (0.66, [[(0.08, 0.93), (0.08, 0.07), (0.32, 0.07)] + arc(0.32, 0.29, 0.27, 0.22, 90, -90) + [(0.08, 0.51)]]),
    "R": (0.68, [[(0.08, 0.93), (0.08, 0.07), (0.32, 0.07)] + arc(0.32, 0.29, 0.27, 0.22, 90, -90) + [(0.08, 0.51)],
                 [(0.30, 0.51), (0.62, 0.93)]]),
    "S": (0.64, [arc(0.32, 0.285, 0.24, 0.215, 28, 270) + arc(0.32, 0.715, 0.24, 0.215, 90, -152)]),
    "T": (0.70, [[(0.04, 0.07), (0.66, 0.07)], [(0.35, 0.07), (0.35, 0.93)]]),
    "U": (0.74, [[(0.08, 0.07), (0.08, 0.62)] + arc(0.37, 0.62, 0.29, 0.31, 180, 360) + [(0.66, 0.07)]]),
    "Y": (0.74, [[(0.05, 0.07), (0.37, 0.52), (0.69, 0.07)], [(0.37, 0.52), (0.37, 0.93)]]),
    "Z": (0.66, [[(0.08, 0.07), (0.60, 0.07), (0.06, 0.93), (0.60, 0.93)]]),
    "-": (0.40, [[(0.07, 0.57), (0.33, 0.57)]]),
    "'": (0.20, [[(0.12, -0.02), (0.08, 0.22)]]),
    " ": (0.34, []),
}
ACCENTS = {"Ô": ("O", [(-0.15, -0.07), (0.0, -0.21), (0.15, -0.07)]), "É": ("E", [(-0.08, -0.07), (0.1, -0.21)])}


def lettering(text, x, base, cap, weight, r, track=0.06, width=None):
    """A line of capitals drawn on the stone with a flat brush, each letter set by eye: a little off the
    line, leaning a hair, its strokes not quite all of one weight. Returns (polyline, width) pairs; with
    `width` the line is spaced out to fill it."""
    chars = [ACCENTS.get(c, (c, None)) for c in text]
    adv = [GLYPHS[c][0] for c, _ in chars]
    if width is not None:
        track = (width / cap - sum(adv)) / (len(chars) - 1)
    out = []
    for (c, acc), a in zip(chars, adv):
        lean = r.normal(0, 0.008)
        sc = cap * (1 + r.normal(0, 0.012))
        ox, oy = x + r.normal(0, 0.01) * cap, base - sc + r.normal(0, 0.008) * cap
        parts = list(GLYPHS[c][1]) + ([[(a / 2 + u, v) for u, v in acc]] if acc else [])
        for k, s in enumerate(parts):
            s = np.asarray(s, np.float64)
            px = ox + (s[:, 0] + lean * (1 - s[:, 1])) * sc
            py = oy + s[:, 1] * sc
            w = weight * cap * (1 + r.normal(0, 0.035)) * (0.8 if acc and k == len(parts) - 1 else 1)
            out.append((list(zip(px, py)), w))
        x += (a + track) * cap
    return out


# ---------------------------------------------------------------------------------------------------
# the picture


def ridge_line(x0, x1, base, peaks, r, rough=2.0):
    """A skyline drawn by hand: straight-ish runs between peaks, with small breaks along it. Returns the
    x samples, the line, and which peak owns each sample."""
    xs = np.arange(x0, x1 + 1, 2.0)
    cones = np.array([py + np.where(xs < px, (px - xs) * sl, (xs - px) * sr) for px, py, sl, sr in peaks])
    y = np.minimum(cones.min(0), base)
    y += rough * noise.line1d(len(xs), 9, r) + 2.5 * rough * noise.line1d(len(xs), 60, r)
    return xs, y, cones.argmin(0)


def tufted(cx, cy, a, b, r, spike=(5, 16), rough=0.1, flat=0.7, droop=0.5):
    """The outline of a clump of pine needles: an uneven oval with a flatter underside, and round it the
    tufts at the ends of the twigs, each a fan of needles splayed outward, here a run of long ones and
    there a stretch nearly bare; the ones underneath hang. Returns the outline and the needles, as
    polygons."""
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    rad = 1 + rough * noise.line1d(240, 22, r) + 0.4 * rough * noise.line1d(240, 6, r)
    x, y = cx + a * rad * np.cos(t), cy - b * rad * np.sin(t)
    y = np.minimum(y, cy + flat * b * (1 + 0.25 * noise.line1d(240, 30, r)))
    x, y = np.append(x, x[0]), np.append(y, y[0])
    if spike[1] <= 0:
        return np.stack([x, y], 1), np.zeros((0, 3, 2))
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    u = np.cumsum(r.uniform(5, 13, int(s[-1] / 5) + 2))
    u = u[u < s[-1]]
    px, py = np.interp(u, s, x), np.interp(u, s, y)
    tx, ty = np.interp(u + 2, s, x) - np.interp(u - 2, s, x), np.interp(u + 2, s, y) - np.interp(u - 2, s, y)
    nl = np.hypot(tx, ty) + 1e-9
    ang = np.arctan2(tx / nl + droop, -ty / nl)                 # outward, and pulled down a little
    L = (spike[0] + (spike[1] - spike[0]) * np.clip(r.exponential(0.4, len(u)), 0, 1.3)
         * noise.smoothstep(-1.4, 0.6, noise.line1d(len(u), 7, r)))
    k = 4
    a_ = ang[:, None] + r.uniform(-0.6, 0.6, (len(u), k))
    l_ = L[:, None] * r.uniform(0.55, 1.1, (len(u), k))
    w_ = r.uniform(1.3, 2.4, (len(u), k))
    c, sn = np.cos(a_), np.sin(a_)
    bx, by = px[:, None] - 2 * c, py[:, None] - 2 * sn
    tri = np.stack([np.stack([bx - sn * w_, by + c * w_], -1), np.stack([bx + sn * w_, by - c * w_], -1),
                    np.stack([bx + c * (l_ + 2), by + sn * (l_ + 2)], -1)], -2).reshape(-1, 3, 2)
    return np.stack([x, y], 1), tri


def clumps(pads, sun, r, spike=(5, 16), rough=0.1, flat=0.7, light=(0.75, 0.5, 0.42, 0.15)):
    """Clumps of foliage (cx, cy, a, b, lit) drawn one over another in the order given. Each is dark
    beneath; lit 1, the sky lights the top of it; lit 2, the side that faces `sun` glows as well, where
    the light comes through the needles. Returns the dark, mid, lit and glowing parts as masks
    over the plate, the lighter three carried a little way under the dark so that a slip in register
    shows as a dark hair and not a white one; and last the lighter three together, as drawn."""
    ss = 2
    lab = np.zeros((PH * ss, PW * ss), np.uint8)
    for px, py, pa, pb, lit in pads:
        dx, dy = sun[0] - px, sun[1] - py
        dn = np.hypot(dx, dy)
        dx, dy = dx / dn, dy / dn
        bx0, by0 = max(0, int(px - 1.3 * pa - 50)), max(0, int(py - 1.3 * pb - 50))
        bx1, by1 = min(PW, int(px + 1.3 * pa + 50)), min(PH, int(py + 1.3 * pb + 50))
        if bx1 <= bx0 or by1 <= by0:
            continue
        w, h = bx1 - bx0, by1 - by0

        def poly(ox, oy, sp, scale=1.0):
            out, tri = tufted(px - bx0 + ox, py - by0 + oy, pa * scale, pb * scale, r, sp, rough, flat)
            im = Image.new("L", (w * ss, h * ss), 0)
            d = ImageDraw.Draw(im)
            d.polygon([tuple(q) for q in out * ss], fill=1)
            for t in tri * ss:
                d.polygon([tuple(q) for q in t], fill=1)
            return np.asarray(im, bool)
        win = lab[by0 * ss:by1 * ss, bx0 * ss:bx1 * ss]
        pad = poly(0, 0, spike)
        win[pad] = 1
        if not lit:
            continue
        win[pad & ~poly(-light[3] * pa * dx, light[0] * pb, spike, 1.03)] = 2
        win[pad & ~poly(-light[1] * pa * dx, -0.35 * pb * dy + light[2] * pb, spike, 1.02)] = 3
        if lit > 1:
            win[pad & ~poly(-(0.1 * pa + 7) * dx, -(0.1 * pb + 7) * dy, (max(spike[0] - 1, 0), max(spike[1] - 4, 0)))] = 4
    sl = box(lab[::ss, ::ss].astype(np.float32), pad=4)
    out = []
    for k in (1, 2, 3, 4):
        m = np.zeros(SHAPE, np.float32)
        if sl is not None:
            sub = (lab[sl[0].start * ss:sl[0].stop * ss, sl[1].start * ss:sl[1].stop * ss] == k).astype(np.uint8) * 255
            m[sl] = np.asarray(Image.fromarray(sub).resize((sl[1].stop - sl[1].start, sl[0].stop - sl[0].start),
                                                           Image.BOX), np.float32) / 255
        out.append(m)
    out = [noise.smoothstep(0.2, 0.5, m) for m in out]         # a brush cannot hold a tip finer than this
    core = np.clip(sum(out[1:]), 0, 1)
    whole = np.clip(out[0] + core, 0, 1)
    return out[:1] + [np.clip(ndimage.gaussian_filter(m, 1.5) * 2.2, 0, 1) * whole for m in out[1:]] + [core]


def crown(cx, cy, a, b, r, toward, tiers=((0.5, 3), (0.8, 4), (1.0, 5), (0.85, 5)), hang=()):
    """The clumps of an umbrella pine's crown: a dark mass first, so no sky shows through its heart,
    then tiers of broad flat clumps stacked into a dome, the lower hung in front of the upper so their
    lit tops show against the dark behind; last, the clumps that hang lower at the ends of the limbs.
    The sun is on the `toward` side (+1 right, -1 left), and only the clumps on that side glow."""
    pads = [(cx, cy + 0.05 * b, 0.82 * a, 0.62 * b, 0)]
    T = len(tiers)
    for k, (wf, m) in enumerate(tiers):
        yk = cy - 0.72 * b + k * (1.15 * b / (T - 1))
        half = a * wf
        pw = 1.9 * half / m
        row = []
        for j in range(m):
            u = -half + (j + 0.5) * 2 * half / m + r.normal(0, 0.22) * pw
            pa = 0.7 * pw * r.uniform(0.7, 1.3)
            row.append((cx + u, yk + r.normal(0, 0.13) * b, pa, min(pa * r.uniform(0.45, 0.62), 0.5 * b),
                        2 if u * toward > 0.15 * a else 1))
        r.shuffle(row)
        pads += row
    for hx, hy, ha in hang:
        pads.append((hx, hy, ha, ha * r.uniform(0.4, 0.5), 2 if (hx - cx) * toward > 0 else 1))
    return pads


def bezier(p, n=120):
    p = np.asarray(p, np.float64)
    t = np.linspace(0, 1, n)[:, None]
    k = len(p) - 1
    return sum(comb(k, i) * t ** i * (1 - t) ** (k - i) * p[i] for i in range(k + 1))


def bole(path, limbs, w0, w1, sun, r, nscales=70):
    """An umbrella pine's trunk, a smooth line through `path` from its foot to the fork, and the limbs
    (tip, bend, thickness) it opens into under the crown. The bark toward the sun catches it: a band of
    red, and on the very edge a rim of gold. Returns the whole, the red band, the gold rim, and short
    dark scales across the lit side."""
    path = np.asarray(path, np.float64)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    n = 260
    p = interpolate.CubicSpline(s / s[-1], path)(np.linspace(0, 1, n))
    p[:, 0] += 3 * noise.line1d(n, 30, r)
    hw = np.linspace(w0, w1, n) / 2 * (1 + 0.07 * noise.line1d(n, 18, r))
    fork = path[-1]
    parts = [(p, hw)]
    for tip, mid, f in limbs:
        q = bezier([fork, mid, tip], 90)
        q[:, 0] += 1.5 * noise.line1d(90, 20, r)
        parts.append((q, np.linspace(f * w1, 0.5 * f * w1, 90) / 2 * (1 + 0.08 * noise.line1d(90, 14, r))))
    whole = relief.cuts(SHAPE, parts)
    side = np.sign(sun[0] - path[0, 0])

    def band(off, wid, wave):
        out = []
        for pts, h in parts:
            k = len(pts)
            o = (off + wave * noise.line1d(k, 7, r)) * h
            out.append((pts + np.stack([side * o, np.zeros(k)], 1), h * wid))
        return relief.cuts(SHAPE, out) * whole
    red = band(0.25, 0.68, 0.12)
    gold = band(0.8, 0.28, 0.1)
    scales = []
    pts, h = parts[0]
    for i in r.integers(0, len(pts) - 1, nscales):
        c = pts[i] + np.array([side * h[i] * r.uniform(0.0, 0.6), 0])
        L = h[i] * r.uniform(0.4, 0.9)
        t = np.linspace(-0.5, 0.5, 6)[:, None]
        scales.append((c + t * np.array([L, r.normal(0, 0.25) * L]), r.uniform(1.0, 2.2) * np.cos(t[:, 0] * 2.5)))
    return whole, red, gold, relief.cuts(SHAPE, scales) * whole


def facets(blocks, r, region, ss=2):
    """Porphyry in blocks cut into planes. Each block (cx, cy, a, b, z) is a convex solid: a broad top
    tipped up to the sky, bevels, sides falling away left, right and below, and a face turned toward
    us, each plane at a tilt of its own; z says how near the block stands. At every point the nearest
    block shows, and the plane of it seen there takes the light by how squarely it faces the sun.
    Returns over the plate the tops, the faces in the sun, the faces half turned away and those in
    shade; how far each plane has turned from the front of its block (0 to 1, for the crayon); the
    undercuts where a nearer block shades a farther one; the dark foot of every block; the shadows
    the blocks cast on the ground; and the rock as a whole."""
    rx0, ry0, rx1, ry1 = region
    H, W = ry1 - ry0, rx1 - rx0
    gy, gx = np.mgrid[0:H * ss, 0:W * ss].astype(np.float32)
    gx, gy = rx0 + (gx + 0.5) / ss - 0.5, ry0 + (gy + 0.5) / ss - 0.5
    depth = np.full(gx.shape, -np.inf, np.float32)
    lam = np.zeros(gx.shape, np.float32)
    turn = np.zeros(gx.shape, np.float32)
    bid = np.full(gx.shape, -1, np.int32)
    for i, (cx, cy, a, b, z) in enumerate(blocks):
        th = np.radians(np.r_[np.array([-90, -35, -145, 5, 175, 55, 125, 90]) + r.normal(0, 12, 8), r.uniform(0, 360)])
        ph = np.radians(np.r_[np.array([54, 60, 60, 74, 74, 66, 66, 72]) + r.normal(0, 5, 8), r.uniform(8, 22)])
        rho = np.r_[r.uniform(0.85, 1.12, 8), 4.0]
        sx0, sx1 = max(0, int((cx - 1.5 * a - rx0) * ss)), min(W * ss, int((cx + 1.5 * a - rx0) * ss) + 1)
        sy0, sy1 = max(0, int((cy - 1.5 * b - ry0) * ss)), min(H * ss, int((cy + 1.5 * b - ry0) * ss) + 1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        sub = (slice(sy0, sy1), slice(sx0, sx1))
        u, v = (gx[sub] - cx) / a, (gy[sub] - cy) / b
        c, s, t = (q[:, None, None] for q in (np.cos(th), np.sin(th), np.tan(ph)))
        ws = t * (rho[:, None, None] - u * c - v * s)
        k, wv = ws.argmin(0), ws.min(0)
        zz = z + 0.5 * a * wv
        hit = (wv > 0) & (zz > depth[sub])
        nrm = np.stack([np.tan(ph) * np.cos(th), a / b * np.tan(ph) * np.sin(th), np.ones(9)], 1)
        face = nrm @ LIGHT / np.linalg.norm(nrm, axis=1)
        depth[sub][hit] = zz[hit]
        lam[sub][hit] = face[k][hit]
        turn[sub][hit] = (1 - wv / wv.max())[hit]
        bid[sub][hit] = i

    def down(m):
        return m.reshape(H, ss, W, ss).mean((1, 3)).astype(np.float32)
    rock = bid >= 0
    parts = [down(rock & (lam > 0.7)), down(rock & (lam > 0.4) & (lam <= 0.7)),
             down(rock & (lam > 0.12) & (lam <= 0.4)), down(rock & (lam <= 0.12))]
    cov = down(rock)
    D, B = depth.reshape(H, ss, W, ss).max((1, 3)), bid[::ss, ::ss]
    under = np.zeros((H, W), bool)
    for d in (3, 6, 10, 15):
        dx, dy = round(0.47 * d), round(-0.88 * d)
        Bn, Dn = ahead(B, dx, dy, -1), ahead(D, dx, dy, -np.inf)
        under |= (B >= 0) & (Bn >= 0) & (Bn != B) & (Dn > D + 3)
    foot = np.zeros((H, W), np.float32)
    for d in (1, 2, 3, 4, 5):
        foot = np.maximum(foot, ahead(cov, 0, -d, 0))
    cast = np.zeros((H, W), np.float32)
    for d in (4, 8, 13, 19, 26, 34):
        cast = np.maximum(cast, ahead(cov, round(0.3 * d), -d, 0))
    out = []
    for m in parts + [down(turn), ndimage.gaussian_filter(under.astype(np.float32), 0.6), foot * (1 - cov),
                      cast * (1 - cov), cov]:
        o = np.zeros(SHAPE, np.float32)
        o[ry0:ry1, rx0:rx1] = m
        out.append(o)
    return out


def agave(cx, cy, size, r, tilt=0.5):
    """An agave seen a little from above: a rosette of thick leaves round its heart, the outer ones
    spread low and arching over toward their tips, the inner ones standing up. Each leaf is folded
    along its midrib, so the half turned up to the light is pale and the other half is in shade; both
    have a yellow margin, and the leaf ends in a dark spine. Returns the leaves, farthest first, each a
    list of (layer, polygon): 0 the lit half, 1 the shaded half, 2 and 3 their margins, 4 the spine."""
    ring = []
    for n, el, ln, dr in ((14, (0.05, 0.45), (0.9, 1.1), (0.6, 1.6)), (10, (0.6, 0.9), (0.75, 0.95), (0.3, 0.7)),
                          (6, (1.1, 1.4), (0.45, 0.65), (0.0, 0.2))):
        ring += [(b, r.uniform(*el), size * r.uniform(*ln), r.uniform(*dr))
                 for b in 2 * np.pi * (np.arange(n) + r.uniform(0, 1, n)) / n]
    ct, st = np.cos(tilt), np.sin(tilt)
    t = np.linspace(0, 1, 28)
    mid = len(t) // 2
    leaves = []
    for beta, el, L, droop in ring:
        e = el - droop * t ** 2
        d3 = np.stack([np.cos(e) * np.sin(beta), np.sin(e), np.cos(e) * np.cos(beta)], 1)
        p3 = np.vstack([[0, 0, 0], np.cumsum(d3[:-1], 0) * L / (len(t) - 1)])
        p3 += 0.05 * size * np.array([np.sin(beta), 0, np.cos(beta)])
        m = np.stack([cx + p3[:, 0], cy - (p3[:, 1] * ct - p3[:, 2] * st)], 1)
        depth = p3[mid, 2] * ct + p3[mid, 1] * st
        tg = np.gradient(m, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
        nv = np.stack([-tg[:, 1], tg[:, 0]], 1)
        w = 0.13 * L * (1 - t) ** 0.7 * (0.75 + 0.25 * noise.smoothstep(0, 0.2, t))
        lit = 1 if nv[mid] @ LIGHT[:2] > 0 else -1
        polys = []
        j = (t > 0.03) & (t < 0.9)
        for side, k in ((-lit, 1), (lit, 0)):
            edge = m + side * nv * w[:, None]
            inner = m + side * nv * (0.8 * w)[:, None]
            polys.append((k, np.vstack([m, edge[::-1]])))
            polys.append((k + 2, np.vstack([inner[j], edge[j][::-1]])))
        q = int(0.93 * (len(t) - 1))
        polys.append((4, np.array([m[q] + nv[q] * w[q], m[-1] + tg[-1] * 0.07 * L, m[q] - nv[q] * w[q]])))
        leaves.append((depth, polys))
    leaves.sort(key=lambda q: q[0])
    return [p for _, p in leaves]


def village(hill, coast, r):
    """The old town crowding up the headland: tall houses of every width packed up the slope, each set
    a little forward of or behind its neighbours, their fronts white in the sun, their shaded sides
    lavender, roofs of orange tile or flat terraces, and here and there a dark window; the church and
    its campanile near the top. `hill` and `coast` give the headland's top and its waterline at any x.
    Returns (fronts, sides, roofs, red roofs, windows)."""
    lay = Layers(5, (280, 640, 1140, 1160))
    put = lay.put

    def house(x, yb, fw, fh, red=False, flat=False):
        sw = fw * r.uniform(0.25, 0.45)
        rh = fw * r.uniform(0.16, 0.26)
        put(1, [(x - sw, yb - fh + 0.3 * sw), (x, yb - fh), (x, yb), (x - sw, yb)])
        put(0, [(x, yb - fh), (x + fw, yb - fh), (x + fw, yb), (x, yb)])
        if not flat:
            put(3 if red else 2, [(x - sw - 2, yb - fh + 0.3 * sw), (x - 1, yb - fh + 1), (x + fw + 2, yb - fh + 1),
                                  (x + fw - 0.1 * fw, yb - fh - rh), (x - sw + 0.1 * fw, yb - fh - rh + 0.3 * sw)])
        floors, bays = max(1, int(fh / 17)), max(1, int(fw / 13))
        cells = r.permutation(floors * bays)[:r.choice([0, 0, 1, 1, 1, 2, 2, 3])]
        for c in cells:
            i, j = divmod(c, floors)
            wx = x + fw * (i + 0.5) / bays + r.normal(0, 1.2)
            wy = yb - fh + fh * (j + 0.4) / floors + r.normal(0, 1.2)
            ww, wh = r.uniform(2.6, 4.2), r.uniform(4.0, 7.0)
            put(4, [(wx - ww / 2, wy - wh / 2), (wx + ww / 2, wy - wh / 2), (wx + ww / 2, wy + wh / 2),
                    (wx - ww / 2, wy + wh / 2)], cover=False)

    # the church first, up on the hill: a gabled front and the campanile beside it
    cx, cb = 690.0, 850.0
    house(cx, cb, 58, 62)
    put(1, [(cx + 52, cb - 146), (cx + 62, cb - 150), (cx + 62, cb), (cx + 52, cb)])
    put(0, [(cx + 62, cb - 150), (cx + 86, cb - 150), (cx + 86, cb), (cx + 62, cb)])
    put(2, [(cx + 50, cb - 146), (cx + 60, cb - 152), (cx + 74, cb - 188), (cx + 88, cb - 152), (cx + 88, cb - 148)])
    for j in range(3):
        put(4, [(cx + 70, cb - 136 + 34 * j), (cx + 78, cb - 136 + 34 * j), (cx + 78, cb - 122 + 34 * j),
                (cx + 70, cb - 122 + 34 * j)], cover=False)
    houses = []
    yb = 860.0
    while yb < 1150:
        x = 450 + r.uniform(-50, 50)
        while x < 1100:
            fw, fh = r.uniform(16, 52), r.uniform(24, 66) * (1.4 if r.random() < 0.12 else 1)
            y = yb + r.normal(0, 11)
            if hill(x) + 0.6 * fh + 30 < y < min(coast(x), coast(x + fw)) - 6 and r.random() < 0.8:
                houses.append((x, y, fw, fh, r.random() < 0.3, r.random() < 0.2))
            x += fw + r.uniform(-8, 26)
        yb += r.uniform(12, 30)
    for q in sorted(houses, key=lambda q: q[1]):
        house(*q)
    return lay.masks()


def boat(x, y, hgt):
    """A small sailing boat far out: the mainsail white, the jib in shade, the hull a dark sliver."""
    main = [(x, y - hgt), (x + 1.5, y - 5), (x + 0.55 * hgt, y - 6)]
    jib = [(x - 2, y - 0.82 * hgt), (x - 0.38 * hgt, y - 5), (x - 2, y - 5)]
    hull = [(x - 0.45 * hgt, y - 4), (x + 0.65 * hgt, y - 4), (x + 0.5 * hgt, y + 0.04 * hgt),
            (x - 0.35 * hgt, y + 0.04 * hgt)]
    return raster([main]), raster([jib]), raster([hull])


def paint(seed=1930):
    r = noise.rng(seed)
    sheet = poster_paper(r)
    S = {k: Stone(r) for k in ORDER}
    x0, y0, x1, y1 = PIC
    pic = noise.smoothstep(-0.5, 0.5, np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y]))
    sky = pic * noise.smoothstep(HZ + 0.6, HZ - 0.6, Y)
    sx, sy, sr = SUN
    dsun = np.hypot(X - sx, Y - sy)
    th = np.arctan2(Y - sy, X - sx)
    disc = sr + 1.2 * np.sin(3 * th + 1.0) + 0.8 * np.sin(5 * th + 2.3) + 0.5 * np.sin(9 * th + 0.4)   # cut by hand
    halo = np.clip(dsun - disc, 0, None)

    # ---- the land and the water
    # the headland across the bay, and its town
    hxs, hy, _ = ridge_line(x0, 1170, 1200, [(300, 772, 0.1, 0.22), (700, 860, 0.2, 0.3)], r, rough=1.2)
    hill = lambda x: np.interp(x, hxs, hy)
    coast = lambda x: (np.interp(x, [x0 - 5, 400, 700, 950, 1100, 1165], [1180, 1150, 1100, 1040, 1000, 990])
                       + 3 * np.sin(x / 37.0))
    fronts, sides, roofs, reds, windows = village(hill, coast, r)
    town = np.clip(fronts + sides + roofs + reds, 0, 1)
    head = pic * noise.smoothstep(-0.6, 0.6, Y - hill(X)) * noise.smoothstep(0.6, -0.6, Y - coast(X)) * (X < 1170)
    head = head * (1 - town)
    # trees on the hill round the town, and a few cypresses by the church
    trees = []
    for _ in range(70):
        x = r.uniform(x0, 1130)
        y = r.uniform(hill(x) + 8, coast(x) - 12)
        if town[int(y), int(np.clip(x, 0, PW - 1))] < 0.5:
            a = r.uniform(12, 28)
            trees.append((x, y, a, a * r.uniform(0.5, 0.7), 1))
    trees.sort(key=lambda p: p[1])
    tr = [m * head for m in clumps(trees, SUN, r, spike=(1.5, 5))]
    cypress = raster([[(x, y - h), (x + w, y - 0.3 * h), (x + 0.6 * w, y), (x - 0.6 * w, y), (x - w, y - 0.3 * h)]
                      for x, y, w, h in ((656, 852, 6, 62), (638, 862, 5, 48), (802, 878, 6, 58), (980, 930, 5, 46))])
    # far mountains behind it, and a far cape on the right
    peaks = [(250, 690, 0.4, 0.3), (560, 735, 0.28, 0.45), (820, 790, 0.4, 0.55), (1000, 835, 0.5, 0.8)]
    xs, ry, own = ridge_line(x0, 1120, HZ, peaks, r)
    mts = pic * noise.smoothstep(-0.6, 0.6, Y - np.interp(X, xs, ry, right=HZ + 5)) * (Y < HZ + 2) * (1 - town)
    owner = np.interp(X, xs, own, right=0).round().astype(int)
    pk = np.array(peaks)
    spur = pk[owner, 0] + (Y - pk[owner, 1]) * np.where(owner % 2 == 0, -0.35, 0.25) + 6 * noise.field(SHAPE, 30, r)
    mts_shade = mts * noise.smoothstep(2, -2, X - spur)
    xs2, ry2, _ = ridge_line(1480, x1, HZ, [(1790, 822, 0.12, 0.05)], r, rough=0.8)
    cape = pic * noise.smoothstep(-0.6, 0.6, Y - np.interp(X, xs2, ry2, left=HZ + 5)) * (Y < HZ + 2)

    # the near shore: a beach in the sun at the head of a little cove, porphyry stepping out into the
    # water on the right, a red outcrop on the left where the big pine stands, and below, the slope we
    # stand on, in the shade of the pines, with agaves and bushes of the maquis
    fx = np.array([x0 - 5, 300, 560, 760, 900, 1100, 1300, 1500, 1700, x1 + 5])
    fy = np.array([1830, 1840, 1880, 1930, 1962, 1972, 1950, 1905, 1885, 1875])
    shore = np.interp(X, fx, fy) + 5 * noise.line1d(PW, 30, r)[None, :] + 2 * noise.line1d(PW, 6, r)[None, :]
    land = pic * noise.smoothstep(-0.6, 0.6, Y - shore)
    blocks = [(330, 1990, 300, 170), (700, 2080, 220, 130), (935, 2160, 120, 80), (120, 2120, 100, 130),
              (250, 2290, 260, 200), (590, 2330, 240, 160), (440, 2470, 280, 120),
              (960, 1985, 70, 40), (1045, 1998, 48, 28), (1330, 1950, 90, 55), (1450, 1925, 130, 75),
              (1600, 1905, 110, 70), (1745, 1895, 150, 85), (1880, 1915, 100, 80),
              (1235, 1892, 34, 18), (1530, 1842, 55, 30), (1662, 1818, 40, 22), (1805, 1822, 60, 30),
              (1400, 1870, 26, 14),
              (1700, 2300, 230, 140), (1870, 2190, 130, 120), (1540, 2440, 200, 110), (1880, 2420, 160, 120)]
    blocks = [(cx, cy, a, b, cy + r.uniform(-15, 15)) for cx, cy, a, b in blocks]
    rtop, rlit, rhalf, rshade, turn, under, foot, cast, rock = [m * pic for m in facets(blocks, r, (x0, 1640, x1, y1))]
    lay = Layers(5, (x0, 1700, x1, y1))
    for cx, cy, size in ((1000, 2140, 100), (1790, 2150, 160), (1230, 2395, 330)):
        for leaf in agave(cx, cy, size, r):
            for k, poly in leaf:
                lay.put(k, poly)
    ag_lit, ag_shade, ag_mlit, ag_mshade, ag_spine = [m * pic for m in lay.masks()]
    agv = np.clip(ag_lit + ag_shade + ag_mlit + ag_mshade + ag_spine, 0, 1)
    bushes = []
    for cx, cy, a in ((660, 1985, 80), (1120, 2095, 80), (1450, 2105, 120), (1600, 2135, 85), (925, 2320, 120)):
        bushes += crown(cx, cy, a, 0.62 * a, r, 1 if cx < sx else -1, tiers=((0.6, 2), (1.0, 3), (0.85, 3)))
    mq = [m * pic * (1 - agv) for m in clumps(bushes, SUN, r, spike=(3, 9))]
    fg = np.clip(agv + mq[0] + mq[4], 0, 1)
    rtop, rlit, rhalf, rshade, under, rock = [m * (1 - fg) for m in (rtop, rlit, rhalf, rshade, under, rock)]
    ground = land * (1 - rock) * (1 - fg)
    edge = (np.interp(X, [x0, 700, 1000, 1300, 1600, x1], [2110, 2085, 2060, 2050, 2030, 2020])
            + 12 * noise.line1d(PW, 45, r)[None, :] + 4 * noise.line1d(PW, 6, r)[None, :])
    shady = ground * np.maximum(noise.smoothstep(-0.6, 0.6, Y - edge), cast)
    sand = ground - shady
    foot = foot * ground
    sea = pic * noise.smoothstep(HZ - 0.6, HZ + 0.6, Y) * (1 - land) * (1 - head) * (1 - town) * (1 - rock)
    sail, jib, hull = [np.maximum(a, b) for a, b in zip(boat(780, 1330, 70), boat(1430, 1010, 28))]

    # the pines: the big one out of the outcrop, leaning over the bay with its crown across the top of
    # the sheet; the small one on the rocks across the cove
    big = (crown(200, 520, 190, 100, r, 1, tiers=((0.6, 2), (1.0, 3), (0.8, 3)), hang=((150, 600, 70),))
           + crown(540, 340, 390, 210, r, 1, tiers=((0.45, 3), (0.75, 4), (1.0, 6), (0.9, 5)),
                   hang=((360, 545, 90), (700, 560, 80)))
           + crown(900, 470, 175, 95, r, 1, tiers=((0.6, 2), (1.0, 3), (0.8, 3)), hang=((1000, 545, 55),)))
    small = crown(1700, 1290, 225, 115, r, -1, tiers=((0.5, 2), (0.8, 3), (1.0, 4), (0.8, 3)), hang=((1530, 1370, 55),))
    cs = [m * pic for m in clumps(small, SUN, r, spike=(3, 10))]
    cdark, cmid, clit, cglow, ccore = [np.maximum(a * pic, b) for a, b in zip(clumps(big, SUN, r), cs)]
    crowns, over_sea = np.clip(cdark + ccore, 0, 1), np.clip(cs[0] + cs[4], 0, 1)
    pl = bole([(235, 2520), (275, 2200), (305, 1900), (290, 1600), (318, 1300), (380, 1050), (455, 870), (550, 715),
               (640, 610)],
              [((330, 560), (480, 640), 0.6), ((560, 420), (600, 520), 0.75), ((880, 500), (780, 590), 0.6),
               ((760, 380), (700, 480), 0.5)], 128, 70, SUN, r, nscales=110)
    ps = bole([(1770, 1905), (1800, 1760), (1790, 1620), (1748, 1500), (1690, 1420)],
              [((1560, 1330), (1610, 1400), 0.7), ((1680, 1250), (1680, 1350), 0.65),
               ((1830, 1300), (1760, 1370), 0.6)],
              48, 30, SUN, r, nscales=34)
    whole, red, gold, scales = [np.maximum(a, b) * pic * (1 - crowns) for a, b in zip(pl, ps)]

    # the sun's road on the water: a scatter of short glints, thickest under the sun
    road = sea * (1 - crowns) * np.exp(-((X - sx) / (30 + 0.42 * np.clip(Y - HZ, 0, None))) ** 2)
    marks = []
    py, px = np.divmod(r.choice(PH * PW, int(road.sum() / 45), p=(road / road.sum()).ravel()), PW)
    for x, y in zip(px, py):
        L = r.uniform(3, 9 + 0.035 * (y - HZ))
        t = np.linspace(0, 1, 8)
        marks.append((np.stack([x + L * (t - 0.5), y + 0.5 * r.normal() * np.sin(np.pi * t)], 1),
                      (r.uniform(0.5, 1.1) + 0.0015 * (y - HZ)) * np.sin(np.pi * t) ** 0.5 + 0.15))
    glint = relief.cuts(SHAPE, marks)

    # ---- the stones
    open_sky = sky * (1 - crowns) * (1 - town)
    off = (dsun > disc + 1.0).astype(np.float32)                 # the sun itself is stopped out: bare paper
    reach = 390 + 30 * noise.field(SHAPE, 140, r)            # how far out the blue was stopped off the sun
    # yellow: the glare spattered round the sun, the lit and glowing needles, the hill, warm light on the
    # town, the tops of the rocks and a warmth in their sunny faces, the beach, the margins of the
    # agaves, sparks on the water
    S["yellow"].spatter(np.clip(0.95 * np.exp(-halo / 75.0), 0, 0.9) * sky * off, rmax=3.5)
    S["yellow"].tusche((clit + cglow + mq[2] + mq[3] + tr[2]) * pic)
    S["yellow"].tusche(head)
    S["yellow"].crayon(0.22 * fronts, angle=1.2)
    S["yellow"].tusche(rtop)
    S["yellow"].crayon(0.3 * rlit, angle=1.0)
    S["yellow"].crayon(0.8 * sand, angle=-0.2, length=4)
    S["yellow"].crayon(0.3 * ag_lit, angle=-1.2, length=7)
    S["yellow"].tusche(ag_mlit + ag_mshade)
    S["yellow"].tusche(glint * noise.smoothstep(0.3, 1.0, noise.field(SHAPE, 4, r)))
    # rose: a thin warm haze round the glare; the mountains; the shaded sides of the houses; red roofs;
    # the red faces and the shade of the rocks and their shadows; the slope in shade; a little in the
    # hill to grey its green
    S["rose"].spatter(np.clip(0.3 * np.exp(-halo / 90.0), 0, 0.3) * sky * off * (halo < 320), rmax=2.5)
    S["rose"].tusche(mts + cape)
    S["rose"].tusche(sides + reds)
    S["rose"].tusche(rhalf + rshade + shady)
    S["rose"].crayon((0.3 + 0.5 * turn) * rlit, angle=1.0)
    S["rose"].crayon(0.2 * head, angle=-0.6, length=3)
    # orange: the roofs, the rocks, the glowing rims of the crowns, the beach
    S["orange"].tusche(roofs + reds)
    S["orange"].tusche(rlit + rhalf + rshade)
    S["orange"].crayon((0.35 + 0.3 * turn) * rtop, angle=0.7)
    S["orange"].tusche(cglow + mq[3])
    S["orange"].crayon(0.1 * shady, angle=-0.4, length=4)
    warm = 0.12 + 0.12 * noise.smoothstep(0, 1.2, noise.field(SHAPE, 150, r))
    S["orange"].crayon(warm * sand, angle=-0.2, length=4, press=0.12)
    # light blue: the sky, flat but spattered off toward the sun; the sea; mountains and hill; the shaded
    # sides of the houses, the shade on the rocks and the slope; the crowns; the agaves
    S["blue"].tusche(sky * (1 - cglow) * (1 - town) * noise.smoothstep(reach - 4, reach + 4, dsun))
    S["blue"].spatter(0.92 * noise.smoothstep(sr + 50, reach + 30, dsun) ** 1.3 * open_sky * (dsun < reach + 40), rmax=2.4)
    S["blue"].tusche(np.clip(sea + mts + cape + head + sides + clit + mq[2] + tr[2] + over_sea
                             + ag_lit + ag_shade, 0, 1) * (1 - sail) * (1 - cglow))
    S["blue"].tusche(jib)
    S["blue"].scrape(glint * noise.smoothstep(-0.3, 0.6, noise.field(SHAPE, 5, r)))
    # green: the middle tone of the crowns, the bushes and the trees, the cypresses, the agaves in shade
    S["green"].tusche(cmid + mq[1] + tr[1] + cypress + ag_shade)
    S["green"].crayon(0.5 * ag_mshade, angle=-1.2)
    # ultramarine: the sea, paling in the shallows, toward the horizon and down the sun's road; crayon
    # grain toward the top of the sky; the shade on the mountains, the rocks and the slope
    dl = ndimage.distance_transform_edt((head + land + town + rock) < 0.5)
    deep = noise.smoothstep(4, 80, dl) * (1 - 0.3 * noise.smoothstep(HZ + 300, HZ, Y))
    deep *= 1 - 0.45 * np.exp(-((X - sx) / (25 + 0.3 * np.clip(Y - HZ, 0, None))) ** 2)
    S["ultra"].crayon(np.clip(deep, 0, 1) * sea * (1 - crowns) * (1 - sail - jib), angle=0.03, width=3, length=9,
                      press=0.35)
    S["ultra"].scrape(glint)
    S["ultra"].crayon(0.18 * noise.smoothstep(430, y0, Y) * open_sky, angle=0.35, width=2.0, length=3, press=0.1)
    S["ultra"].crayon(0.5 * mts_shade, angle=1.1)
    S["ultra"].crayon((0.3 + 0.4 * turn) * rshade, angle=0.9)
    S["ultra"].crayon((0.02 + 0.3 * turn) * rhalf, angle=0.9)
    S["ultra"].crayon((0.18 + 0.32 * noise.smoothstep(2050, y1, Y)) * shady, angle=0.5, length=6, press=0.12)
    # navy: the dark of the crowns, the bushes and the trees, the cypresses, the undercuts and the foot of
    # the rocks, the spines of the agaves, windows, the hulls
    S["navy"].tusche(cdark + mq[0] + tr[0] + cypress)
    S["navy"].tusche(under + foot + ag_spine)
    S["navy"].crayon(0.4 * turn ** 2 * rshade, angle=0.9, width=2.5, length=7, press=0.3)
    S["navy"].tusche(windows + hull)

    # the trunks over everything, on stones washed clean under them
    knock = noise.smoothstep(0.5, 0.95, ndimage.gaussian_filter(whole, 1.2))
    for st in S.values():
        st.clean(knock)
    S["yellow"].tusche(gold)
    S["rose"].tusche(red * (1 - gold))
    S["orange"].tusche(red)
    S["navy"].tusche(whole * (1 - red))
    S["navy"].tusche(scales * (1 - gold))

    # the frame, the lettering, and the signature on the slope
    fx0, fy0, fx1, fy1 = FRAME
    S["ultra"].rule(raster([], lines=[([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1), (fx0, fy0)], 12)]))
    name = lettering("CÔTE D'AZUR", 175, 2745, 215, 0.155, r, width=PW - 350)
    title = raster([], lines=name, ss=3, region=(100, 2440, PW - 100, 2800))
    S["ultra"].tusche(title)
    S["navy"].tusche(np.roll(title, (9, 11), (0, 1)) * (1 - title))
    plm = lettering("PARIS-LYON-MÉDITERRANÉE", 235, 2910, 76, 0.15, r, width=PW - 470)
    S["orange"].tusche(raster([], lines=plm, ss=3, region=(100, 2800, PW - 100, 2960)))
    sig = lettering("CLAUDE", 1720, 2422, 20, 0.16, r, track=0.2)
    S["navy"].tusche(raster([], lines=sig, ss=4, region=(1700, 2390, 1880, 2436)))

    img = sheet.color.copy()
    for k in ORDER:
        img = pull(img, S[k].drawing(), INKS[k], sheet, r, r.normal(0, 1.4, 2), r.normal(0, 2e-4))
    img = folds(img, sheet, r)
    return plate.mount(img, sheet, shadow=0.35)
