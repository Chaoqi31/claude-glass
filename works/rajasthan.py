"""The Black Sun. Acrylic on canvas, laid on with the knife and the brush in layers, scumbled, run and scratched.

In the late 1970s S. H. Raza, who had painted the villages and fields of France for twenty years, went back in
his pictures to the India of his childhood, painting in Paris the earth of Rajasthan and Saurashtra as he
remembered it under the summer sun: not the land as it is seen, but its colours and the order under them. The
canvases of these years are square, framed inside their own edges by bands joined at the corners like the
members of a wooden frame, and divided into zones of blazing colour, vermilion, carmine, saffron, orange and
ochre, with a deep green or a blue against them and black to make them burn. Triangles, bands and wedges cross
the zones, drawn by hand. He worked in acrylic, which dries almost as fast as it is laid, so that every layer
lies on dry paint: spread with the knife, brushed, scumbled with a dry brush in one hot colour over another,
run thin so that it drips, and scratched with the point of the brush or the knife down to the colour beneath.

Here the sky is laid in bands, hot on the left and deep on the right, and two triangles stand in it as planes of
colour rather than as things. The large one is divided down its height into a light side and a deep one, with a
carmine wedge along its left side and a black one down the upper part of its right, crossing above its point. A
green one overlaps its foot, and where the two lie over each other they make a third colour, burnt sienna. A
carmine wedge hangs point down into the green one, and where those two cross there is a lozenge of ultramarine.
The bands of the sky run on through the triangles and change colour as they cross them; the red one runs across
both, orange and vermilion in the one and crimson in the other. The sun stands black above, red at its rim.
Below, past a band of green and black, the earth is a saffron room and a carmine one with a pillar of vermilion
between them. Into the wet paint of each went a sheaf of broad strokes going one way, the room's own colour
deepening toward the middle of it, saffron into orange and vermilion, carmine into dark red, and the room's colour
brushed back over its edges; black lines are written across them along the other slope of the triangles.

The canvas went on in this order. A ground of saffron and orange brushed thin, every which way. Then each zone,
first spread broad and flat with a wide knife, then worked again with a small one over a part of it, the patches
going now one way and now another as the hand moved on, with ridges where the knife lifted: some zones left broad
and smooth, others worked over and over. Each edge was painted by hand, the paint of one zone running on over the
next in places and in others stopping short of it, so that a line of the ground shows between. The wedges were
spread with the knife along their length, and the strokes in the rooms laid while the rooms were wet. Then the
frame, each side painted along its length, the left half hot and the right half deep. Over the dry paint,
scumbles of a dry brush, one hot colour over another, the colour beneath glowing through; and last the scratches,
down to the yellow ground, and the runs.
"""

import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Black Sun"
DATE = "2026"
MEDIUM = "Acrylic on canvas, laid on with the knife and the brush in layers, scumbled, run and scratched"
AFTER = ("S. H. Raza, the paintings of Rajasthan and Saurashtra made in Paris from the mid 1970s: Rajasthan (1975), "
         "Rajasthan (1983), Saurashtra (1983), La Terre (1985)")
ROOM = "Colour Itself"
YEAR = 1983
PLACE = "Paris"
REGION = "South Asia"
NOTE = ("Two triangles stand in a banded sky as planes of colour, crossed by the bands and making a third colour "
        "where they overlap, under a black sun; below, a saffron room and a carmine one, each crossed by broad "
        "strokes of its own colour deepened. Acrylic spread with the knife zone by zone, scumbled, and scratched "
        "down to the yellow ground.")

H = W = 2700
PAD = 48                        # the canvas runs on round the stretcher this far, so strokes run off the edges
SH, SW = H + 2 * PAD, W + 2 * PAD
U = W / 1000.0                  # the drawing is made on a square of 1000 units
LIGHT = (-0.6, -0.5, 0.62)


def px(p):
    """Units of the drawing -> px on the canvas."""
    return np.asarray(p, np.float64) * U + PAD


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from deep to light
VERMILION = pal("#c82e16", "#d8381a", "#e2461c", "#ea5420", "#ef6424")
RED = pal("#8e1012", "#a51515", "#b81c18", "#c8251b", "#d6321f")
CARMINE = pal("#6a0a1d", "#820c21", "#981127", "#ad172c", "#c02333")
CRIMSON = pal("#3c0511", "#4d0816", "#5f0b1b", "#720f21")
SIENNA = pal("#5a2210", "#702c12", "#873815", "#9c4519", "#b0541e")
ORANGE = pal("#dd6219", "#e8761c", "#ef881f", "#f39a26", "#f6ab30")
SAFFRON = pal("#ee9e12", "#f3b01a", "#f6c024", "#f8ce38", "#f9da58")
LEMON = pal("#f8d64a", "#fae27a", "#fbeba2")
OCHRE = pal("#a5671b", "#ba7a21", "#ca8d2b", "#d8a03a")
GREEN = pal("#082c1a", "#0a3620", "#0e4829", "#145b31", "#1d6f3a", "#2b8443")
ULTRA = pal("#0e1452", "#131c6a", "#192886", "#20369e")
BLACK = pal("#0e0a0a", "#160f0f", "#201313", "#2a1916")

# how the paint goes on: spread flat with a small knife, ridged where it lifts; spread broad and smooth with a wide
# one; thick acrylic brushed; thinned for the ground; a dry brush dragged over dry paint; broad strokes of a loaded
# flat brush laid into wet paint; lines written with a small brush
KNIFE = dict(thick=0.11, grooves=0.06, lips=0.3, land=0.4, lift=0.75, tails=0.05, pickup=0.15, merge=1.0, spent=0.08,
             ends=(0.02, 0.0), taper=0.0, fray=0.5, flatten=0.9)
SWEEP = dict(KNIFE, thick=0.035, grooves=0.02, lips=0.06, land=0.1, lift=0.12, fray=0.3, flatten=1.0)
BRUSH = dict(thick=0.05, grooves=0.3, lips=0.12, land=0.2, lift=0.25, tails=0.2, pickup=0.25, merge=2.0, spent=0.08,
             ends=(0.3, 0.2), taper=0.08, fray=0.8, flatten=0.6)
THIN = dict(thick=0.02, grooves=0.15, lips=0.02, land=0.1, lift=0.05, tails=0.3, pickup=0.4, merge=4.0, spent=0.2,
            ends=(0.4, 0.3), taper=0.1, fray=1.5, flatten=0.5, hide=0.95)
DRY = dict(BRUSH, thick=0.05, spent=0.5, tails=0.5, pickup=0.1)
WET = dict(thick=0.09, grooves=0.25, lips=0.25, land=0.35, lift=0.5, tails=0.3, pickup=0.3, merge=4.0, spent=0.4,
           ends=(0.08, 0.05), taper=0.35, fray=1.6, flatten=0.7)
LINE = dict(thick=0.3, grooves=0.25, lips=0.1, land=0.5, lift=0.3, tails=0.6, pickup=0.0, merge=1.0, spent=0.55,
            ends=(0.6, 0.3), taper=0.6, fray=0.6, flatten=0.5)

# the drawing, in units: the frame's inner edge, and the field inside the narrow fillet
FL, FT, FR, FB = 46, 54, 954, 948
L2, T2, R2, B2 = 55, 63, 945, 939
M = 30                                                                  # the frame runs on past the canvas edge
BANDS = (T2 - 5, 124, 192, 210, 330, 350, 440, 522)                    # the bands of the sky
MIDS = (488, 512, 497, 520, 482, 506, 494)                             # where the hot half meets the deep, band by band
APEX, FOOT_L, FOOT_R, SPLIT = (318, 118), (96, 522), (640, 522), (446, 522)     # the large triangle, and its division
GAPEX, GFOOT_L, GFOOT_R = (640, 250), (468, 522), (842, 522)          # the green one, over its foot
HANG_L, HANG_R, HANG = (522, T2 - 5), (760, T2 - 5), (652, 338)       # a wedge hanging point down into the green
SUN, SUN_R = (800, 190), 64


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def cut(poly, a, b):
    """The part of a convex polygon on the right of the line from a to b, as the canvas is drawn, y down."""
    s = [(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) for p in poly]
    out = []
    for i, p in enumerate(poly):
        q, j = poly[(i + 1) % len(poly)], (i + 1) % len(poly)
        if s[i] >= 0:
            out.append(tuple(p))
        if s[i] * s[j] < 0:
            t = s[i] / (s[i] - s[j])
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def inside(poly, *shapes):
    """The part of a convex polygon inside every one of some convex shapes. -> polygon, or [] if none"""
    for sh in shapes:
        sh = list(sh)
        if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(sh, sh[1:] + sh[:1])) < 0:
            sh = sh[::-1]
        for a, b in zip(sh, sh[1:] + sh[:1]):
            poly = cut(poly, a, b)
            if len(poly) < 3:
                return []
    return poly


def zone(poly, g, grow=1.5, rough=1.0, pad=40):
    """A zone of the drawing: its signed distance in px (positive inside) over the box round it. The line is drawn
    by hand: every corner set down a little off, the edge wavering, out past the drawn line in one place and short
    of it in another. -> (x0, y0, sd)"""
    P = px(poly) + g.normal(0, 1.2 * rough + 0.8, (len(poly), 2))
    x0, y0 = max(int(P[:, 0].min()) - pad, 0), max(int(P[:, 1].min()) - pad, 0)
    x1, y1 = min(int(P[:, 0].max()) + pad, SW), min(int(P[:, 1].max()) + pad, SH)
    im = Image.new("1", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in P - (x0, y0)], fill=1)
    m = np.asarray(im)
    sd = np.where(m, ndimage.distance_transform_edt(m) - 0.5, 0.5 - ndimage.distance_transform_edt(~m))
    s = sd.shape
    sd += grow + rough * (2.4 * noise.field(s, 350, g) + 1.8 * noise.field(s, 110, g) + 0.9 * noise.field(s, 30, g)
                          + 0.4 * noise.field(s, 6, g))
    return x0, y0, sd.astype(np.float32)


def lanes(pts, angle, w, L, g, overlap=0.4, tilt=0.03, vary=0.15, bow=0.012):
    """Strokes that cover the points `pts` (N, 2) in lanes side by side along `angle`, as a painter fills a band:
    each lane of its own width, laid in one stroke or a few end to end, each overlapping the last and a little
    askew, bowed a little, now one way and now the other. -> paths, half-widths"""
    e = np.array([np.cos(angle), np.sin(angle)])
    f = np.array([-e[1], e[0]])
    u, q = pts @ e, pts @ f
    paths, ws = [], []
    v = q.min() + 0.55 * w
    while v - 0.55 * w < q.max():
        ww = w * np.exp(np.clip(g.normal(0, vary), -2 * vary, 2 * vary))
        band = np.abs(q - v) < ww
        if band.any():
            u0 = u[band].min() - g.uniform(0.2, 0.6) * ww
            u1 = u[band].max() + g.uniform(0.2, 0.6) * ww
            m = max(1, int(round((u1 - u0) / g.uniform(*L))))
            cut_ = np.linspace(u0, u1, m + 1)
            cut_[1:-1] += g.uniform(-0.25, 0.25, m - 1) * (u1 - u0) / m
            for j in range(m):
                s0, s1 = cut_[j] - (g.uniform(0.15, 0.4) * ww if j else 0), cut_[j + 1]
                t = g.normal(0, tilt)
                p0 = e * s0 + f * (v + g.normal(0, 0.06 * ww))
                p1 = p0 + (e * np.cos(t) + f * np.sin(t)) * (s1 - s0)
                P = np.array([p0, (p0 + p1) / 2 + f * g.normal(0, bow * (s1 - s0)), p1])
                paths.append(P[::-1] if g.random() < 0.4 else P)
                ws.append(ww)
        v += 2 * ww * (1 - overlap) * g.uniform(0.85, 1.1)
    return paths, np.array(ws)


def patches(pts, w, aim, g, length=(2.0, 4.5), bow=0.04):
    """Patches of paint spread with the knife over the points `pts` (N, 2): one about every 1.5 w px, each of its
    own width about w px half-wide and `length` half-widths long, heading the way `aim` (a function of the place)
    gives, and laid in no order, so that each lies over some of those before it. -> paths, half-widths"""
    lo, span = pts.min(0), np.ptp(pts, 0)
    seeds = dabs.scatter((span[1] + 1, span[0] + 1), 1.5 * w, g).astype(np.float64) + lo
    seeds = seeds[cKDTree(pts).query(seeds)[0] < 0.5 * w]
    if not len(seeds):
        seeds = pts.mean(0)[None]
    n = len(seeds)
    ww = w * np.exp(np.clip(g.normal(0, 0.4, n), -0.8, 0.8))
    th = aim(seeds)
    e = np.stack([np.cos(th), np.sin(th)], 1) * (ww * g.uniform(*length, n))[:, None] / 2
    f = np.stack([-e[:, 1], e[:, 0]], 1) * g.normal(0, 2 * bow, n)[:, None]
    flip = g.random(n) < 0.5
    paths = [np.array([c + d, c + b, c - d]) if k else np.array([c - d, c + b, c + d])
             for c, d, b, k in zip(seeds, e, f, flip)]
    return paths, ww


def loads(fam, keys, g, accent=None, odds=0.0):
    """The load of each knife or brush: the shade of the paint at `keys` (0 deep to 1 light), the shade next to it,
    and a little of a third, now and then an accent from elsewhere on the palette. -> colours (N, 3, 3), shares"""
    n, N = len(fam), len(keys)
    i = np.clip(np.round(np.asarray(keys) * (n - 1)).astype(int), 0, n - 1)
    j = np.clip(i + g.choice([-1, 1], N), 0, n - 1)
    third = fam[np.clip(i + g.integers(-1, 2, N), 0, n - 1)].copy()
    if accent is not None:
        hit = g.random(N) < odds
        third[hit] = accent[g.integers(0, len(accent), hit.sum())]
    cols = np.stack([fam[i], fam[j], third], 1) * np.exp(g.normal(0, 0.02, (N, 1, 1)))
    share = np.stack([g.uniform(0.65, 0.9, N), g.uniform(0.1, 0.25, N), g.uniform(0.02, 0.08, N)], 1)
    return np.clip(cols, 1e-4, 1).astype(np.float32), share


def arc(p, theta, L, turn, wave=0.0, n=14):
    """A movement of the hand about p, L px long, heading off at `theta` and turning `turn` radians in all, and
    swinging `wave` radians one way and back on the way."""
    s = np.linspace(0, 1, n)
    th = theta + turn * s + wave * np.sin(2 * np.pi * s)
    P = np.concatenate([[[0.0, 0.0]], np.cumsum(np.stack([np.cos(th[1:]), np.sin(th[1:])], 1) * L / (n - 1), 0)])
    return P - P.mean(0) + p


def scratch(rgb, height, ground, base, paths, g, width=1.4, depth=0.6, reveal=0.85):
    """Lines scratched with the point of the brush handle or of the knife through the paint to the ground. The point
    goes in and comes out, presses harder and lighter and here and there rides over; the paint it ploughs up
    stands in a little burr along both sides."""
    for P in paths:
        C = brush.path(np.asarray(P, np.float64), 0.5)
        n = len(C)
        s = np.arange(n) * 0.5
        press = noise.smoothstep(0, 10, s) * noise.smoothstep(s[-1], s[-1] - 16, s) \
            * np.clip(0.75 + 0.3 * noise.line1d(n, 220, g) + 0.2 * noise.line1d(n, 30, g), 0, 1.2)
        R = 2.2 * width + 4
        x0, y0 = int(max(C[:, 0].min() - R, 0)), int(max(C[:, 1].min() - R, 0))
        x1, y1 = int(min(C[:, 0].max() + R + 1, rgb.shape[1])), int(min(C[:, 1].max() + R + 1, rgb.shape[0]))
        if x1 <= x0 or y1 <= y0:
            continue
        im = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(im).line([tuple(p) for p in np.vstack([C[::8], C[-1:]]) - (x0, y0)], fill=255,
                                width=int(2 * R))
        yy, xx = np.nonzero(np.asarray(im))
        Y, X = yy + y0, xx + x0
        d, k = cKDTree(C).query(np.stack([X, Y], 1).astype(np.float64))
        p = press[k]
        wd = width * (0.5 + 0.7 * p)
        cut_ = noise.smoothstep(wd + 0.7, wd - 0.7, d) * noise.smoothstep(0.2, 0.45, p)
        rgb[Y, X] += (ground[Y, X] - rgb[Y, X]) * (reveal * cut_)[:, None]
        burr = 0.35 * depth * np.exp(-((d - 1.7 * wd) / (0.7 * wd)) ** 2) * p
        height[Y, X] = np.maximum(height[Y, X] - depth * cut_ * p, base[Y, X]) + burr


def runs(rgb, height, drips, g):
    """Paint thinned to run, gathered at the foot of a zone and run down over what is below: a thread that wanders
    a little from the plumb and thins as it goes, and mostly stops in a bead."""
    for x0, y0, w, L, col, a in drips:
        y0, y1 = int(y0), int(min(rgb.shape[0] - 1, y0 + L))
        if y1 - y0 < 8 or not 0 <= x0 < rgb.shape[1]:
            continue
        ys = np.arange(y0, y1 + 1)
        t = (ys - y0) / (y1 - y0)
        wave = lambda k, n: np.interp(t, np.linspace(0, 1, n), g.normal(0, k, n))
        xc = x0 + (1.5 + 0.01 * L) * wave(1.0, 6) + w * wave(0.3, 14)
        half = w * (1.2 - 0.6 * t) * np.exp(wave(0.15, 10))
        half = (half + w * np.exp(-((ys - y1 + 1.4 * w) / (1.1 * w)) ** 2)) * np.sqrt(np.clip((y1 - ys) / w, 0, 1))
        xa, xb = int(max(0, xc.min() - 3 * w - 2)), int(min(rgb.shape[1], xc.max() + 3 * w + 3))
        d = np.abs(np.arange(xa, xb)[None] - xc[:, None])
        cov = np.clip(half[:, None] - d + 0.5, 0, 1) * (a * (1 - 0.2 * t))[:, None]
        sub = rgb[y0:y1 + 1, xa:xb]
        sub += (col - sub) * cov[..., None]
        bead = np.sqrt(np.clip(1 - (d / (half[:, None] + 1e-3)) ** 2, 0, 1))
        height[y0:y1 + 1, xa:xb] += 0.6 * half[:, None] * bead * cov


def paint(seed=1983):
    r = noise.rng(seed)
    duck = canvas.duck((SH, SW), seed, tint="#f1ece0", thread=3.2)
    rgb = duck.color.copy()
    height = duck.tooth * 0.35
    tone = noise.field((SH // 16 + 2, SW // 16 + 2), 22, r)      # where a paint went on a little deeper or lighter
    B = impasto.bristles(r)

    def fresh(name):
        """Each passage draws on chances of its own, so the others stay as they are when one is changed."""
        return np.random.default_rng([seed, zlib.crc32(name.encode())])

    def shade(pts, key, drift):
        k0, k1 = key if isinstance(key, tuple) else (key, key)
        t = k0 + (k1 - k0) * np.clip(pts[:, 0] / SW, 0, 1)
        iy = np.clip((pts[:, 1] / 16).astype(int), 0, tone.shape[0] - 1)
        ix = np.clip((pts[:, 0] / 16).astype(int), 0, tone.shape[1] - 1)
        return t + 0.15 * drift * tone[iy, ix]

    def region(name, poly, grow, rough):
        """-> the passage's chances, the zone's window and signed distance, and the points inside it (window px)"""
        g = fresh(name)
        if len(poly) < 3:
            return g, None, None, np.zeros((0, 2))
        x0, y0, sd = zone(poly, g, grow, rough)
        ys, xs = np.nonzero(sd[::4, ::4] > 0)
        return g, (x0, y0), sd, np.stack([xs * 4.0 + 2, ys * 4.0 + 2], 1)

    def lay(g, at, sd, paths, ws, fam, key, how, drift=1.0, accent=None, odds=0.0, dry=0.0):
        """Lay strokes on a copy of the canvas in the zone's window. -> the window, and the painted copy"""
        keys = shade(np.array([p.mean(0) for p in paths]) + at, key, drift) + g.normal(0, 0.07, len(paths))
        cols, share = loads(fam, keys, g, accent, odds)
        win = np.s_[at[1]:at[1] + sd.shape[0], at[0]:at[0] + sd.shape[1]]
        sub, hs = rgb[win].copy(), height[win].copy()
        impasto.lay(sub, hs, paths, ws, cols, g, share=share, wet=np.zeros(sd.shape, np.float32), dry=dry, **how)
        return win, sub, hs

    def put(win, sub, hs, a):
        rgb[win] += (sub - rgb[win]) * a[..., None]
        height[win] += (hs - height[win]) * a

    def fill(name, poly, fam, angle, w, L=(500, 900), how=BRUSH, grow=1.5, rough=1.0, key=0.5, overlap=0.45,
             accent=None, odds=0.08):
        """Paint a band in lanes of strokes along `angle`, w px half-wide, stopped at its hand-drawn edge."""
        g, at, sd, pts = region(name, poly, grow, rough)
        if len(pts):
            paths, ws = lanes(pts, angle, w, L, g, overlap)
            put(*lay(g, at, sd, paths, ws, fam, key, how, accent=accent, odds=odds), np.clip(sd + 0.5, 0, 1))

    def knife(name, poly, fam, key=0.5, angle=0.0, dirs=(0.0, 0.6, -0.6), w=36, worked=0.6, ridge=1.0, grow=1.5,
              rough=1.0, drag=14, drift=1.0, accent=None, odds=0.08, skip=0.12):
        """Lay a zone with the knife: first spread broad and flat over all of it with a wide knife, then worked again
        with a small one, w px half-wide, over a part of it (`worked`, 0 none to 1 all). The hand works one place
        and moves on, so each patch heads along one of `dirs` (off `angle`) as its neighbours do, and the heading
        changes from place to place. Its edge is drawn by hand, and here and there (`drag`, px) the small knife
        runs on over it into the next zone; now and then (`skip`) the knife is nearly dry and leaves the colour
        beneath showing through. `ridge`: how high the paint stands where the small knife lifts."""
        g, at, sd, pts = region(name, poly, grow, rough)
        if not len(pts):
            return
        lo, span = pts.min(0), np.ptp(pts, 0)
        cen = dabs.scatter((span[1] + 1, span[0] + 1), 4.5 * w, g).astype(np.float64) + lo
        head = angle + np.asarray(dirs, np.float64)[g.integers(0, len(dirs), len(cen))]
        tree = cKDTree(cen)
        aim = lambda p: head[tree.query(p)[1]] + g.normal(0, 0.08, len(p))
        a = np.clip(sd + 0.5, 0, 1)
        put(*lay(g, at, sd, *patches(pts, 2.2 * w, aim, g, (1.4, 3.0)), fam, key, SWEEP, drift, accent, odds), a)
        busy = pts[(g.random(len(cen)) < worked)[tree.query(pts)[1]]]
        if len(busy):
            paths, ws = patches(busy, w, aim, g)
            dry = np.where(g.random(len(paths)) < skip, g.uniform(0.2, 0.45, len(paths)), 0.0)
            over = drag * np.clip(1.6 * noise.field(sd.shape, 220, g) - 0.5, 0, 1)
            how = dict(KNIFE, lips=KNIFE["lips"] * ridge, lift=KNIFE["lift"] * ridge)
            put(*lay(g, at, sd, paths, ws, fam, key, how, drift, accent, odds, dry), np.clip(sd + 0.5 + over, 0, 1))

    def scumble(name, poly, fam, key=0.6, angle=0.0, dirs=(0.0,), w=40, catch=0.5, grow=-6, rough=3.0, soft=8.0):
        """A dry brush dragged across dry paint. Its colour catches only on the high places, the ridges of the knife
        and the threads of the canvas where the paint lies thin, and breaks along the drag of the bristles; where
        the hand bore down it catches more, and between, the colour beneath glows through. `catch`: how little it
        leaves."""
        g, at, sd, pts = region(name, poly, grow, rough)
        if not len(pts):
            return
        aim = lambda p: angle + np.asarray(dirs)[g.integers(0, len(dirs), len(p))] + g.normal(0, 0.06, len(p))
        win, sub, hs = lay(g, at, sd, *patches(pts, w, aim, g, (2.5, 6.0)), fam, key, DRY)
        h0 = height[win]
        hp = h0 - ndimage.gaussian_filter(h0, 5)
        hp /= hp.std() + 1e-6
        yy, xx = np.mgrid[0:sd.shape[0], 0:sd.shape[1]].astype(np.float32)
        u, v = xx * np.cos(angle) + yy * np.sin(angle), yy * np.cos(angle) - xx * np.sin(angle)
        drag = ndimage.map_coordinates(B, [u * 0.5 + 17, v * 0.7 + 41], order=1, mode="grid-wrap")
        c = 0.5 * hp + 0.25 * drag + 0.8 * (duck.tooth[win] - 0.5) + 0.45 * noise.field(sd.shape, 140, g)
        a = np.clip(sd / soft + 0.5, 0, 1) * noise.smoothstep(catch - 0.12, catch + 0.12, c)
        rgb[win] += (sub - rgb[win]) * a[..., None]
        height[win] += 0.4 * (hs - height[win]) * a

    def gesture(name, room, specs, fam):
        """Broad strokes of a loaded flat brush laid into the wet paint of a room: each takes up the room's colour
        under it and drags it along, its edges soften into it, and toward its end it runs dry and breaks up over
        it. specs of (x, y, heading, length, turn, half-width px, key), `fam` running from the room's own colour
        to the deepest."""
        g = fresh(name)
        x0, y0, sd = zone(room, g)
        wet = np.zeros((SH, SW), np.float32)
        wet[y0:y0 + sd.shape[0], x0:x0 + sd.shape[1]] = np.clip(sd + 0.5, 0, 1)
        paths = [arc(px((x, y)), th, L * U, tn, 0.15, n=20) for x, y, th, L, tn, _, _ in specs]
        cols, share = loads(fam, np.array([s[6] for s in specs]), g)
        impasto.lay(rgb, height, paths, np.array([s[5] for s in specs], np.float64), cols, g, share=share, wet=wet,
                    **WET)

    def write(name, specs, fam):
        """Lines written with a small brush, swelling and thinning: specs of (x, y, heading, length, turn, swing,
        half-width px)."""
        g = fresh(name)
        paths = [arc(px((x, y)), th, L * U, tn, wv, n=24) for x, y, th, L, tn, wv, _ in specs]
        cols, share = loads(fam, g.uniform(0.2, 0.8, len(specs)), g)
        impasto.lay(rgb, height, paths, np.array([s[-1] for s in specs], np.float64), cols, g, share=share,
                    wet=np.zeros((SH, SW), np.float32), **LINE)

    # the ground: saffron and orange brushed thin over the white priming, in broad lanes across and then every
    # which way, the left half more yellow and the right more orange
    g = fresh("ground")
    yy, xx = np.mgrid[0:SH:24, 0:SW:24]
    paths, ws = lanes(np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64), 0.06, 110, (900, 1600), g,
                      overlap=0.45, tilt=0.03)
    n = 30
    c = np.stack([g.uniform(0, SW, n), g.uniform(0, SH, n)], 1)
    a = g.uniform(-np.pi, np.pi, n)
    d = np.stack([np.cos(a), np.sin(a)], 1) * g.uniform(500, 1100, n)[:, None] / 2
    paths += [np.array([p - q, p + q]) for p, q in zip(c, d)]
    ws = np.concatenate([ws, g.uniform(60, 100, n)])
    mids = np.array([p.mean(0) for p in paths])
    cols, share = loads(np.concatenate([SAFFRON[1:], ORANGE[2:]]),
                        np.clip(mids[:, 0] / SW, 0, 1) * 0.7 + g.uniform(0, 0.3, len(paths)), g)
    impasto.lay(rgb, height, paths, ws, cols, g, share=share, **THIN)
    ground, base = rgb.copy(), height.copy()

    # the sky, band by band, the left half hot and the right deep, each worked its own way: some spread broad and
    # left smooth, some worked over in patches every which way
    side = lambda k, left: rect(L2 - 5, BANDS[k], MIDS[k], BANDS[k + 1]) if left else \
        rect(MIDS[k], BANDS[k], R2 + 5, BANDS[k + 1])
    flat, cross, every = (0.0,), (0.0, 0.55, -0.55), (0.0, 0.7, -0.7, 1.57)
    sky = [("s0l", 0, 1, VERMILION, 0.6, cross, 24, 0.7), ("s0r", 0, 0, CRIMSON, 0.55, (0.0, 0.2), 56, 0.15),
           ("s1l", 1, 1, ORANGE, (0.8, 0.5), (0.0, 0.25), 60, 0.1), ("s1r", 1, 0, RED, 0.55, every, 26, 0.8),
           ("s2l", 2, 1, SAFFRON, 0.7, flat, 16, 0.4), ("s2r", 2, 0, SIENNA, 0.1, flat, 16, 0.4),
           ("s3l", 3, 1, VERMILION, (0.7, 0.45), cross, 44, 0.55), ("s3r", 3, 0, ORANGE, (0.5, 0.8), flat, 64, 0.1),
           ("s4l", 4, 1, ORANGE, 0.85, flat, 18, 0.4), ("s4r", 4, 0, CARMINE, 0.45, flat, 18, 0.4),
           ("s5l", 5, 1, RED, 0.65, cross, 26, 0.6), ("s5r", 5, 0, RED, 0.3, every, 40, 0.7),
           ("s6l", 6, 1, ORANGE, 0.65, (0.0, 1.57, 0.8), 50, 0.5), ("s6r", 6, 0, VERMILION, 0.45, (0.0, -0.8), 28, 0.6)]
    for name, k, left, fam, key, dirs, w, worked in sky:
        knife(name, side(k, left), fam, key, dirs=dirs, w=w, worked=worked, accent=ORANGE, odds=0.08)
    scumble("sc_s3l", rect(L2, 212, 470, 328), ORANGE, key=0.85, angle=-0.4, dirs=(0.0, 0.5))
    scumble("sc_s1r", rect(540, 126, R2, 190), CARMINE, key=0.6, angle=0.6, dirs=(0.0, -0.4), catch=0.55)
    scumble("sc_s5r", rect(600, 352, R2, 438), CRIMSON, key=0.6, angle=-0.5, dirs=(0.0, 0.9), catch=0.45)
    scumble("sc_s6l", rect(L2, 444, 300, 520), VERMILION, key=0.4, angle=0.3, dirs=(0.0, -0.6), catch=0.4)

    # the large triangle, divided down its height, light on the left and deep on the right, and the bands of the sky
    # running on through it in other colours, the red one in orange and vermilion; a carmine wedge along its left
    # side and a black one down the upper part of its right, crossing over its top, each spread with the knife
    # along its length
    slope = lambda p, q: float(np.arctan2(q[1] - p[1], q[0] - p[0]))
    left_t, right_t = [APEX, SPLIT, FOOT_L], [APEX, FOOT_R, SPLIT]
    up, down = slope(FOOT_L, APEX), slope(APEX, FOOT_R)
    hill = [(1, LEMON, 0.5, SAFFRON, 0.4), (2, ORANGE, 0.8, OCHRE, 0.5), (3, SAFFRON, 0.7, ORANGE, 0.45),
            (4, OCHRE, 0.6, SIENNA, 0.6), (5, ORANGE, 0.5, VERMILION, 0.6), (6, LEMON, 0.3, SAFFRON, 0.3)]
    for k, fl, kl, fr, kr in hill:
        band = rect(0, BANDS[k], 1000, BANDS[k + 1])
        w = min(36, 0.4 * (BANDS[k + 1] - BANDS[k]) * U)
        knife(f"hill{k}l", inside(band, left_t), fl, kl, dirs=(0.0, up), w=w, worked=0.25 if k in (1, 6) else 0.6,
              rough=2.0, drag=18)
        knife(f"hill{k}r", inside(band, right_t), fr, kr, dirs=(down, 0.0, down + 1.2), w=w, worked=0.7, rough=2.0,
              drag=18)
    scumble("sc_hill", inside(rect(0, 212, 1000, 328), right_t), SIENNA, key=0.5, angle=down, dirs=(0.0, -0.9),
            catch=0.42, grow=-3, soft=5)
    scumble("sc_hill6", inside(rect(0, 444, 1000, 520), left_t), ORANGE, key=0.5, angle=up, dirs=(0.0, 0.8),
            catch=0.45, grow=-3, soft=5)
    over = lambda p, k: (APEX[0] + (APEX[0] - p[0]) * k, APEX[1] + (APEX[1] - p[1]) * k)
    along = lambda p, k: (APEX[0] + (p[0] - APEX[0]) * k, APEX[1] + (p[1] - APEX[1]) * k)
    top = (APEX[1] - (T2 - 5)) / (FOOT_R[1] - APEX[1])
    outer, inner = (FOOT_L[0] - 34, FOOT_L[1]), (600, 522)
    for name, poly, fam, a in (("ray_l", [APEX, outer, FOOT_L], CARMINE, up),
                               ("ray_l2", [APEX, over(outer, top), over(FOOT_L, top)], CARMINE, up),
                               ("wedge", [APEX, along(FOOT_R, 0.62), along(inner, 0.38)], BLACK, down),
                               ("wedge2", [APEX, over(FOOT_R, top), over(inner, top)], BLACK, down)):
        fill(name, poly, fam, a, 12, L=(300, 700), how=KNIFE, rough=2.6, key=0.55)

    # the wedge hanging point down, in two carmines; the green triangle over the large one's foot, the red band
    # crossing it in crimson; where the green crosses the large triangle, burnt sienna, and where the wedge crosses
    # the green, ultramarine
    hang = [HANG_L, HANG_R, HANG]
    for k, (ya, yb), key, worked in ((0, (0, 192), 0.75, 0.3), (1, (192, 340), 0.3, 0.7)):
        knife(f"hang{k}", inside(rect(0, ya, 1000, yb), hang), CARMINE, key,
              dirs=(0.0, slope(HANG, HANG_L), slope(HANG_R, HANG)), w=34, worked=worked, rough=2.0, drag=16)
    green = [GAPEX, GFOOT_R, GFOOT_L]
    gup, gdown = slope(GFOOT_L, GAPEX), slope(GAPEX, GFOOT_R)
    for k, (ya, yb), fam, key in ((0, (0, 350), GREEN, 0.2), (1, (350, 440), CRIMSON, 0.5),
                                  (2, (440, 522), GREEN, 0.6)):
        knife(f"green{k}", inside(rect(0, ya, 1000, yb), green), fam, key, dirs=(gup, gdown, 0.0), w=34, worked=0.6,
              rough=2.0, drag=16)
    for k, key in ((5, 0.25), (6, 0.55)):
        knife(f"lap{k}", inside(rect(0, BANDS[k], 1000, BANDS[k + 1]), green, right_t), SIENNA, key,
              dirs=(gup, down), w=28, worked=0.7, rough=2.0, drag=12)
    knife("lozenge", inside(hang, green), ULTRA, 0.45, dirs=(gup, gdown), w=20, worked=0.6, rough=1.5, drag=8)
    scumble("sc_green", cut(inside(rect(0, 444, 1000, 520), green), FOOT_R, APEX), SAFFRON, key=0.6, angle=gup,
            dirs=(0.0, -1.2), w=36, grow=-3, soft=4)

    # the sun: a disc of carmine, and over it black spread with the knife across and across again, its edge cut
    # less surely, so that here and there the red shows at its rim
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    disc = [(SUN[0] + SUN_R * np.cos(a), SUN[1] + SUN_R * np.sin(a)) for a in t]
    knife("sun_red", disc, CARMINE, 0.45, -0.3, (0.0, 1.0), 26, 0.3, grow=1.0, drag=0)
    knife("sun", disc, BLACK, 0.35, 0.12, (0.0, 1.1, -0.8), 30, 0.8, grow=-2.5, rough=2.6, drag=0)

    # the horizon: a line of saffron, and a band green to the left and black to the right
    fill("line", rect(L2 - 5, 517, R2 + 5, 528), SAFFRON, 0.0, 16, key=0.8)
    knife("hor_l", rect(L2 - 5, 528, 500, 562), GREEN, 0.45, dirs=flat, w=34, worked=0.5)
    knife("hor_r", rect(500, 528, R2 + 5, 562), BLACK, 0.6, dirs=(0.0, 0.3), w=34, worked=0.6, accent=GREEN,
          odds=0.3)

    # the earth: a saffron room and a carmine one with a pillar of vermilion between, each spread with the knife
    # along the slopes of the triangles. Into the wet paint of each, a sheaf of broad strokes going one way, the
    # room's own colour deepening toward the middle of it, saffron into orange into vermilion, carmine into dark
    # red, and the room's colour brushed back over its edges; black lines written across them along the other
    # slope; a black wedge rising in the corner; and under them bands of the earth, the deep and the light
    # changing sides
    room_l, room_r = rect(L2 - 5, 562, 470, 862), rect(530, 562, R2 + 5, 862)
    sheaf = lambda a, specs: [(x, y, a + da, L, tn, w, key) for x, y, da, L, tn, w, key in specs]
    knife("room_l", room_l, SAFFRON, 0.65, dirs=(1.57, down, up), w=52, worked=0.12, accent=LEMON, odds=0.06)
    gesture("g_l", room_l, sheaf(up, [(205, 735, 0.0, 300, 0.15, 95, 0.65), (270, 700, -0.12, 260, -0.1, 80, 0.9),
                                      (150, 775, 0.15, 210, 0.2, 70, 0.35), (310, 690, 0.0, 160, 0.0, 55, 1.0),
                                      (105, 690, 0.4, 150, -0.2, 60, 0.0), (370, 800, -0.3, 150, 0.2, 60, 0.05)]),
            np.concatenate([SAFFRON[1::-1], ORANGE[3::-1], VERMILION[::-1]]))
    knife("pillar", rect(470, 562, 530, 862), VERMILION, 0.6, 1.57, flat, 30, 0.5)
    knife("room_r", room_r, CARMINE, 0.55, dirs=(gup, gdown, 0.0), w=30, worked=0.65, ridge=1.2, accent=RED,
          odds=0.15)
    gesture("g_r", room_r, sheaf(gdown, [(705, 712, 0.0, 300, -0.15, 95, 0.6), (770, 690, 0.12, 250, 0.1, 80, 0.9),
                                         (645, 748, -0.15, 200, 0.0, 70, 0.35), (725, 650, 0.0, 150, 0.1, 55, 1.0),
                                         (865, 640, -0.4, 140, 0.2, 60, 0.0), (600, 805, 0.3, 150, -0.2, 60, 0.0)]),
            np.concatenate([CARMINE[3:1:-1], RED[2::-1], CRIMSON[3:0:-1]]))
    dark = [(R2 + 5, 650), (R2 + 5, 862), (760, 862)]
    knife("dark", dark, BLACK, 0.5, slope(dark[2], dark[0]), (0.0, 1.57), 30, 0.6, rough=1.5)
    write("black", [(250, 725, down, 320, -0.2, 0.25, 11), (725, 720, gup, 300, 0.2, 0.2, 11)], BLACK)
    fill("e1", rect(L2 - 5, 860, 500, 875), BLACK, 0.0, 16, key=0.6)
    fill("e1r", rect(500, 860, R2 + 5, 875), GREEN, 0.0, 16, key=0.45)
    knife("e2", rect(L2 - 5, 875, 500, B2 + 5), RED, 0.55, dirs=cross, w=32, worked=0.6, accent=VERMILION, odds=0.2)
    knife("e2r", rect(500, 875, R2 + 5, B2 + 5), SAFFRON, (0.9, 0.5), dirs=(0.0, 0.2), w=40, worked=0.2,
          accent=ORANGE, odds=0.2)
    scumble("sc_room", rect(540, 570, 760, 855), VERMILION, key=0.5, angle=gup, dirs=(0.0, 1.0))
    scumble("sc_yellow", rect(80, 780, 450, 858), ORANGE, key=0.7, angle=np.pi / 2, dirs=(0.0, 0.5))
    scumble("sc_earth", rect(L2, 877, 440, B2), ORANGE, key=0.6, angle=0.45, dirs=(0.0, -0.9), catch=0.35)

    # the fillet round the field, then the frame: each side painted along its length, the left half hot and the
    # right half deep, so that the joints at the corners show
    for k, (poly, a) in enumerate((([(FL, FT), (FR, FT), (R2, T2), (L2, T2)], 0.0),
                                   ([(FL, FB), (FR, FB), (R2, B2), (L2, B2)], 0.0),
                                   ([(FL, FT), (L2, T2), (L2, B2), (FL, FB)], np.pi / 2),
                                   ([(FR, FT), (R2, T2), (R2, B2), (FR, FB)], np.pi / 2))):
        fill(f"fillet{k}", poly, SAFFRON if k != 3 else OCHRE, a, 13, key=(0.8, 0.4), grow=1.0)
    frame = [("top_l", [(-M, -M), (500, -M), (500, FT), (FL, FT)], 0.0, VERMILION, 0.55),
             ("top_r", [(500, -M), (1000 + M, -M), (FR, FT), (500, FT)], 0.0, CARMINE, 0.5),
             ("bot_l", [(-M, 1000 + M), (500, 1000 + M), (500, FB), (FL, FB)], 0.0, VERMILION, 0.45),
             ("bot_r", [(500, 1000 + M), (1000 + M, 1000 + M), (FR, FB), (500, FB)], 0.0, CARMINE, 0.6),
             ("left", [(-M, -M), (FL, FT), (FL, FB), (-M, 1000 + M)], np.pi / 2, VERMILION, 0.6),
             ("right", [(1000 + M, -M), (FR, FT), (FR, FB), (1000 + M, 1000 + M)], np.pi / 2, CARMINE, 0.55)]
    for name, poly, a, fam, key in frame:
        fill(name, poly, fam, a, 40, L=(600, 1000), how=KNIFE, key=key, accent=RED, odds=0.15)
    knife("bar_top", [(560, 15), (932, 15), (918, 37), (574, 37)], BLACK, 0.5, dirs=flat, w=18, worked=0.5, drag=0)
    knife("bar_bot", [(84, 963), (436, 963), (422, 984), (98, 984)], ULTRA, 0.6, dirs=flat, w=18, worked=0.5, drag=0)
    scumble("sc_frame", rect(430, -M, 570, FT), CARMINE, key=0.4, grow=0)

    # the scratches, down through everything to the yellow ground: the sides of the green triangle and of the
    # hanging wedge drawn on past their points, the division of the large one, lines along the bands, the pillar
    # and the frame, hatching on the black wedges, a line across the sun, and lines in the rooms along the slopes
    g = fresh("scratch")
    on = lambda p, q, k: [px(q), px(np.array(q) + (np.array(q) - np.array(p)) * k)]
    lines = [on(GFOOT_L, GAPEX, 0.25), on(GFOOT_R, GAPEX, 0.3), on(HANG_L, HANG, 0.3), on(HANG_R, HANG, 0.22),
             px([np.array(APEX) + (np.array(SPLIT) - APEX) * 0.15, np.array(APEX) + (np.array(SPLIT) - APEX) * 0.7])]
    for y in (536, 552):
        lines.append(px([[g.uniform(60, 140), y + g.normal(0, 1)], [g.uniform(860, 940), y + g.normal(0, 1)]]))
    for y in (24, 975):
        lines.append(px([[g.uniform(20, 60), y], [g.uniform(940, 980), y + g.normal(0, 1.5)]]))
    e = (np.array(dark[2]) - dark[0]) / np.hypot(*(np.array(dark[2]) - dark[0]))
    for _ in range(6):
        p = np.array(dark[1]) + (np.array(dark[0]) - dark[1]) * g.uniform(0.1, 0.9)
        lines.append(px([p, p + e * g.uniform(60, 160)]))
    e = (np.array(FOOT_R) - APEX) / np.hypot(*(np.array(FOOT_R) - APEX))
    for _ in range(3):
        p = np.array(APEX) + e * g.uniform(30, 120) + np.array([-e[1], e[0]]) * g.uniform(-4, 4)
        lines.append(px([p, p + e * g.uniform(60, 150)]))
    lines.append(px([[SUN[0] - SUN_R + 8, 208], [SUN[0] + SUN_R + 14, 205]]))
    for y in (896, 921):                    # along the red band of the earth
        lines.append(px([[g.uniform(60, 120), y], [g.uniform(380, 495), y + g.normal(0, 1.5)]]))
    for x in (488, 509):                    # fluting down the pillar
        lines.append(px([[x, g.uniform(570, 600)], [x + g.normal(0, 1.5), g.uniform(820, 856)]]))
    for x in (FL / 2 - 7, FL / 2 + 7, 1000 - FL / 2 - 7, 1000 - FL / 2 + 7):     # down the sides of the frame
        lines.append(px([[x, g.uniform(70, 160)], [x + g.normal(0, 1), g.uniform(820, 930)]]))
    for p, a, L in (((100, 845), up, 190), ((560, 600), gdown, 170), ((330, 600), down, 120)):   # in the rooms
        lines.append(px([p, np.array(p) + L * np.array([np.cos(a), np.sin(a)])]))
    scratch(rgb, height, ground, base, lines, g)

    # runs: thinned paint run down from the sun, from the green of the horizon into the yellow, from the strokes in
    # the rooms, and from the point of the hanging wedge
    g = fresh("runs")
    drips = [(g.uniform(SUN[0] - 40, SUN[0] + 40), SUN[1] + SUN_R - 4, BLACK[:2]) for _ in range(2)] \
        + [(g.uniform(80, 460), 562, GREEN[1:3]) for _ in range(3)] \
        + [(200, 772, VERMILION[1:3]), (772, 822, RED[:2])] \
        + [(HANG[0] + 1, HANG[1] - 4, ULTRA[:2])] + [(g.uniform(560, 740), 562, BLACK[:2]) for _ in range(2)]
    runs(rgb, height, [(*px((x, y)), g.uniform(2.0, 3.6), g.exponential(90) + 50, fam[g.integers(0, len(fam))],
                        g.uniform(0.8, 0.95)) for x, y, fam in drips], g)

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.6, gloss=0.02, reach=(0.76, 1.18))
    img = img + 0.03 * impasto.glints(height, LIGHT)[..., None]
    return img[PAD:PAD + H, PAD:PAD + W]


if __name__ == "__main__":
    r = noise.rng(0)
    x0, y0, sd = zone(rect(100, 100, 200, 160), r)
    c = px((150, 130)).astype(int)
    assert sd[c[1] - y0, c[0] - x0] > 60 and sd[5, 5] < -20, "a zone is positive inside and negative outside"
    paths, ws = lanes(np.stack(np.meshgrid(np.arange(0, 400, 4.0), np.arange(0, 100, 4.0)), -1).reshape(-1, 2), 0.0,
                      20, (150, 250), r)
    ends = np.concatenate([p[[0, -1]] for p in paths])
    assert ends[:, 0].min() < 5 and ends[:, 0].max() > 395, "lanes run the length of the zone"
    assert len(paths) >= 6, "in several lanes of a few strokes each"
    tri = [(0, 0), (10, 10), (-10, 10)]
    piece = inside(rect(-20, 5, 20, 10), tri)
    assert sorted(piece) == sorted([(5.0, 5.0), (10, 10), (-10, 10), (-5.0, 5.0)]), "a band across a triangle"
    assert inside(rect(-20, -9, 20, -1), tri) == [], "and none above its point"
    assert sorted(inside(tri, rect(0, -5, 20, 20))) == sorted([(0, 0), (10, 10), (0, 10)]), "its right half"
    pts = np.stack(np.meshgrid(np.arange(0, 400, 4.0), np.arange(0, 200, 4.0)), -1).reshape(-1, 2)
    paths, ws = patches(pts, 20, lambda p: np.zeros(len(p)), r)
    mids = np.array([p.mean(0) for p in paths])
    assert len(paths) > 100 and mids.min() > -12 and mids.max() < 412, "patches cover the zone and keep to it"
    assert np.allclose([abs(p[-1, 1] - p[0, 1]) for p in paths], 0), "each along the way it is aimed"
    img, h = np.full((60, 200, 3), 0.5, np.float32), np.zeros((60, 200), np.float32)
    gr = np.full((60, 200, 3), 0.9, np.float32)
    scratch(img, h, gr, np.zeros_like(h), [np.array([[10, 30], [190, 30]])], r)
    assert img[30, 20:180, 0].mean() > 0.7 and img[10, 100, 0] == 0.5, "a scratch shows the ground along its line only"
    print("ok")
