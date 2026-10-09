"""The Sea Garden. Gouache on paper, cut and pasted on white paper, mounted on canvas.

In his last years at the Hôtel Régina in Nice, too ill to stand at an easel, Matisse drew with
scissors. His assistants brushed big sheets of paper with Linel gouache, one colour to a sheet, and
he cut into the colour freehand. The pieces were pinned to white paper on the walls, moved about for
days and at last pasted down. He called the walls of leaves and stars round his bed a garden he
could walk in.

The ground here is three sheets of white paper joined edge over edge. Six slabs of colour,
ultramarine, emerald, orange, magenta, lemon and a second ultramarine, were cut by eye, each side in
two or three runs of the scissors that do not quite line up, and pasted turning round a violet slab
set askew in the middle, one lapping over the next, the white left in wedges between them. Over them
go three sea weeds and four stars, each cut from a sheet of another colour. The white weed, the
largest, climbs the blue with its paddles held out to either side and reaches one long one over onto
the violet, where a lemon star lies; a magenta weed drifts along the green and dips into the orange;
a black one rises out of the magenta and leans in toward the middle from the other side. Each stem
zig-zags where its lobes spring, no two lobes are the same length, and one in each of the smaller
weeds turns back against the rest. The brush streaks of the gouache run each piece its own way. The
long curves of the cuts are made of shorter runs, a hair out of line where the blades closed and
opened again, with a nick here and a slit past an inside corner there, and the brittle paint has
flaked a little along the cut. Where the paste missed, the paper stands off the ground; corners have
lifted and throw soft shadows, and where one piece lies over the edge of another the edge shows
through it. The charcoal marks where a slab was first set out, and the holes of the pins, stay
beside it.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import noise, pencil
from atelier.color import lin, pigment
from atelier.paper import Sheet

TITLE = "The Sea Garden"
DATE = "2026"
MEDIUM = "Gouache on paper, cut and pasted on white paper, mounted on canvas"
AFTER = ("Henri Matisse, the cut-outs (gouaches découpées) of his last years at the Hôtel Régina, Nice, 1950–1954: "
         "Beasts of the Sea, 1950; Memory of Oceania, 1952–53; The Snail, 1953; The Sheaf, 1953")
ROOM = "Colour Itself"
YEAR = 1953
PLACE = "Nice"
REGION = "Europe"
NOTE = ("Slabs of pure colour pasted turning round a violet one set askew, and over them three sea weeds and four "
        "stars cut freehand from sheets brushed with other colours, crossing from one ground into the next.")

H = W = 2600
SS = 4                                   # supersampling of a cut edge


# ---------------------------------------------------------------- outlines

def clockwise(C):
    """The same outline, run clockwise as the eye sees it (y down)."""
    x, y = C[:, 0], C[:, 1]
    return C if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0 else C[::-1].copy()


def resample(C, step=1.0):
    """A closed outline with its points `step` px apart."""
    C = np.vstack([C, C[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    t = np.linspace(0, s[-1], max(4, int(s[-1] / step)), endpoint=False)
    return np.stack([np.interp(t, s, C[:, 0]), np.interp(t, s, C[:, 1])], 1)


def slab(corners, r, kink=0.015, bow=0.01):
    """A slab cut by eye, corner to corner: a long side in two or three runs of the scissors, each
    begun a little off the line of the last so the side bends where they meet, and each bowed as
    the blades closed. The corners fall where the runs crossed, never quite square."""
    c = np.asarray(corners, np.float64)
    P = []
    for i in range(len(c)):
        a, b = c[i], c[(i + 1) % len(c)]
        d = b - a
        L = np.hypot(*d)
        nrm = np.array([d[1], -d[0]]) / L
        k = 1 + int(L > 450) + int(L > 1100)
        knots = np.concatenate([[0], np.sort(r.uniform(0.22, 0.78, k - 1)), [1]])
        off = np.concatenate([[0], r.normal(0, kink * L, k - 1), [0]])
        for j in range(k):
            t = np.linspace(0, 1, max(2, int(L * (knots[j + 1] - knots[j]))), endpoint=False)[:, None]
            p0, p1 = a + d * knots[j] + nrm * off[j], a + d * knots[j + 1] + nrm * off[j + 1]
            sag = r.normal(0, bow) * L * (knots[j + 1] - knots[j]) * 4 * t * (1 - t)
            P.append(p0 + (p1 - p0) * t + nrm * sag)
    return np.vstack(P)


def finger(base, angle, length, wb, wm, bend=0.0, peak=0.6, end=1.0, step=2.0):
    """A finger of colour as the scissors go round it: the centre line from `base` along `angle`,
    turning by `bend` radians over its length; `wb` wide at the root, swelling to `wm` at `peak` of
    the way out and closing beyond it in a cap, round for `end` 1, blunter below, sharper above.
    -> centre (N,2), half-width (N,)"""
    n = max(8, int(length / step))
    s = np.linspace(0, 1, n)
    a = angle + bend * s
    C = np.asarray(base, float) + np.cumsum(np.stack([np.cos(a), np.sin(a)], 1), 0) * length / (n - 1)
    C -= C[0] - np.asarray(base, float)
    rise = wb + (wm - wb) * np.sin(0.5 * np.pi * np.clip(s / peak, 0, 1))
    u = np.clip((s - peak) / (1 - peak), 0, 1)
    return C, 0.5 * np.where(s < peak, rise, wm * (1 - u * u) ** (0.5 * end))


def capsule(C, half, lean=0.0):
    """The closed outline of a finger: out along one side, round the tip, back along the other;
    `lean` cuts its left side that much fuller about the middle and its right side as much leaner."""
    g = np.gradient(C, axis=0)
    g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
    nl = np.stack([g[:, 1], -g[:, 0]], 1)
    k = lean * np.sin(np.pi * np.linspace(0, 1, len(C)))
    L, R = C + nl * (half * (1 + k))[:, None], C - nl * (half * (1 - k))[:, None]
    th = np.linspace(0, np.pi, 12)[1:-1]
    a0 = np.arctan2(nl[-1, 1], nl[-1, 0])
    cap = C[-1] + max(half[-1], 0.5) * np.stack([np.cos(a0 - th), np.sin(a0 - th)], 1)
    a1 = np.arctan2(-nl[0, 1], -nl[0, 0])
    foot = C[0] + max(half[0], 0.5) * np.stack([np.cos(a1 - th), np.sin(a1 - th)], 1)
    return np.vstack([L, cap, R[::-1], foot])


def blob(parts, pad=40, k=2, fillet=3.0):
    """The union of closed outlines as one shape, the inner corners filled a little as a hand cuts
    them round, traced back into one outline. -> (N,2) clockwise, about 1 px apart"""
    P = np.vstack(parts)
    x0, y0 = P.min(0) - pad
    w, h = (P.max(0) - P.min(0) + 2 * pad).astype(int) + 1
    im = Image.new("L", (w * k, h * k), 0)
    d = ImageDraw.Draw(im)
    for q in parts:
        d.polygon([tuple(p) for p in (q - [x0, y0]) * k], fill=255)
    m = ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, fillet * k) > 0.5
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))
    C = trace(m).astype(np.float64)
    C = ndimage.gaussian_filter1d(C, 1.5 * k, axis=0, mode="wrap")
    return resample(C / k + [x0, y0], 1.0)


def trace(m):
    """Moore-neighbour tracing of the outer boundary of a binary mask -> (N,2) x, y, in order."""
    m = np.pad(m, 1)
    ys, xs = np.nonzero(m)
    i = np.lexsort((xs, ys))[0]
    start = (ys[i], xs[i])
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # clockwise from up
    out, cur, back = [start], start, 6                      # we entered the start from its left
    for _ in range(4 * m.size):
        for j in range(8):
            k = (back + 1 + j) % 8
            y, x = cur[0] + nb[k][0], cur[1] + nb[k][1]
            if m[y, x]:
                back = (k + 4) % 8
                cur = (y, x)
                break
        if cur == start:
            break
        out.append(cur)
    out = np.array(out, np.float64)
    return np.stack([out[:, 1] - 1, out[:, 0] - 1], 1)


def weed(base, angle, length, width, lobes, r, bend=(0.0, 0.0), sway=0.0, fillet=0.05):
    """A sea weed cut lobe by lobe, the way the scissors drew it. The stem runs from `base` along
    `angle`, swaying by b0 * s + b1 * s * s radians over its length for `bend` (b0, b1), and turning
    `sway` radians away from each lobe where it springs, so that it zig-zags as it climbs; it
    narrows from `width[0]` to `width[1]`. Each lobe is (t, side, turn, length, root, swell, peak,
    curl, end): it springs `t` of the way up the stem on its left (-1) or right (+1), turned `turn`
    radians off the stem's heading, `root` wide where it leaves the stem and `swell` at its widest
    `peak` of the way out, curling back toward the stem's heading by `curl` radians, its tip round
    for `end` 1, blunter below, sharper above; the edge that faces up the weed is cut leaner."""
    n = max(8, int(length / 2))
    s = np.linspace(0, 1, n)
    a = angle + bend[0] * s + bend[1] * s * s
    for t, side, *_ in lobes:
        a -= side * sway * noise.smoothstep(t - 0.04, t + 0.04, s)
    stem = np.cumsum(np.stack([np.cos(a), np.sin(a)], 1), 0) * length / (n - 1)
    stem += np.asarray(base, float) - stem[0]
    parts = [capsule(stem, 0.5 * (width[0] + (width[1] - width[0]) * s))]
    for t, side, turn, L, root, swell, peak, curl, end in lobes:
        j = int(t * (n - 1))
        C, hf = finger(stem[j], a[j] + side * turn, L, root, swell, bend=-side * curl, peak=peak, end=end)
        parts.append(capsule(C, hf, lean=-side * 0.22 + r.uniform(-0.12, 0.12)))
    return blob(parts, fillet=fillet * width[0])


def star(centre, radius, r, arms=5, turn=0.0, fat=0.44, curl=0.3):
    """A starfish cut arm by arm: arms of different length and heading, each bowed its own way and
    cut fuller on one side."""
    parts = []
    for i in range(arms):
        a = turn + 2 * np.pi * (i + r.uniform(-0.1, 0.1)) / arms
        L, f = radius * r.uniform(0.72, 1.08), fat * radius * r.uniform(0.85, 1.15)
        C, hf = finger(centre, a, L, 1.1 * f, f, bend=r.normal(0, curl), peak=0.05, end=1.7)
        parts.append(capsule(C, hf, lean=r.uniform(-0.25, 0.25)))
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    parts.append(np.asarray(centre) + 0.36 * radius * np.stack([np.cos(th), np.sin(th)], 1))
    return blob(parts, fillet=0.04 * radius)


# ---------------------------------------------------------------- the scissors

def scissors(C, r, run=(70, 190), turn=0.3, facet=0.65, jog=1.0, wobble=1.2, nicks=1 / 1500, overcut=0.35):
    """Cut an outline (closed, clockwise, about 1 px apart) as the scissors cut it: in runs as long
    as one closing of the blades, each a little straighter than the line it follows, the next one
    starting a hair to one side; a slow wander of the hand; now and then a nick where the points
    went into the paper, and a short slit past an inside corner where a cut ran on."""
    C = resample(clockwise(C), 1.0)
    n = len(C)
    g = np.roll(C, -2, 0) - np.roll(C, 2, 0)
    ang = np.unwrap(np.arctan2(g[:, 1], g[:, 0]))
    k = np.angle(np.exp(1j * (np.roll(ang, -3) - np.roll(ang, 3))))       # turning over 6 px
    # runs of the blades: end one where it has turned too far or grown long
    bounds, a = [0], 0
    while a < n - 1:
        target, b, acc = r.uniform(*run), a + 1, 0.0
        while b < n - 1 and b - a < target and abs(ang[b] - ang[a]) < turn:
            b += 1
        bounds.append(b)
        a = b
    out = C.copy()
    nrm = np.stack([g[:, 1], -g[:, 0]], 1) / (np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9)   # outward
    off = wobble * noise.line1d(n, 160, r) + 0.25 * wobble * noise.line1d(n, 25, r)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 4:
            continue
        t = np.linspace(0, 1, b - a + 1)[:, None]
        chord = C[a] + (C[b] - C[a]) * t
        out[a:b + 1] = (1 - facet) * C[a:b + 1] + facet * chord
        off[a:b + 1] += r.normal(0, jog) + r.normal(0, jog * 0.5) * (t[:, 0] - 0.5)
    out += nrm * off[:, None]
    # nicks, and slits run past the sharpest inside corners
    extra = []
    for i in np.flatnonzero(r.random(n) < nicks):
        d, w = r.uniform(1.5, 4.0), r.uniform(2, 5)
        extra.append((i, [out[i] - g[i] / np.hypot(*g[i]) * w / 2, out[i] - nrm[i] * d,
                          out[i] + g[i] / np.hypot(*g[i]) * w / 2]))
    inner = np.flatnonzero((k < -0.5) & (k == ndimage.minimum_filter1d(k, 15, mode="wrap")))
    for i in inner:
        if r.random() < overcut:
            # one of the two cuts that met here ran on past the corner, into the paper
            u = C[i] - C[i - 6] if r.random() < 0.5 else C[i] - C[(i + 6) % n]
            u /= np.hypot(*u) + 1e-9
            side = np.array([-u[1], u[0]]) * 0.3
            extra.append((i, [out[i] + side, out[i] + u * r.uniform(3, 9), out[i] - side]))
    if extra:
        pieces, last = [], 0
        for i, pts in sorted(extra, key=lambda e: e[0]):
            pieces += [out[last:i], np.array(pts)]
            last = i + 1
        pieces.append(out[last:])
        out = np.vstack(pieces)
    return out


def coverage(P, pad=24):
    """The area inside an outline, antialiased: -> (patch, (y0, x0))."""
    x0, y0 = np.floor(P.min(0)).astype(int) - pad
    x1, y1 = np.ceil(P.max(0)).astype(int) + pad
    im = Image.new("L", ((x1 - x0) * SS, (y1 - y0) * SS), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in (P - [x0, y0]) * SS], fill=255)
    a = np.asarray(im, np.float32).reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3)) / 255
    return a, (y0, x0)


# ---------------------------------------------------------------- the design

PAINTS = {   # Linel gouache at full strength, as it dried on the sheets, one colour to a sheet
    "orange": "#ee7616", "emerald": "#0d8c64", "ultramarine": "#2440ac", "magenta": "#bf2279",
    "lemon": "#f5cd22", "black": "#1b1a1c", "violet": "#7d5fbe", "white": "#fbfaf4",
}
GROUND = "#f2eee4"          # the white paper everything was pinned to
STOCK = "#f5f2ea"           # the paper the colours were brushed on
LIGHT = np.array([-0.55, -0.65, 0.52]) / np.linalg.norm([-0.55, -0.65, 0.52])
CHARCOAL = pigment("#4a4644")


U = 100.0
SLABS = [  # the grounds, bottom first, cut by eye and pasted turning round a slab set askew in the middle
    ([(0.5, 19.5), (8.0, 18.2), (8.6, 25.7), (0.25, 25.45)], "lemon"),
    ([(0.5, 1.5), (10.5, 0.7), (12.1, 18.9), (1.0, 19.8)], "ultramarine"),
    ([(9.3, 0.6), (25.6, 0.2), (25.0, 8.7), (17.5, 9.1), (10.4, 10.1)], "emerald"),
    ([(16.0, 7.9), (25.7, 9.4), (25.1, 19.7), (16.9, 18.8)], "orange"),
    ([(19.5, 19.2), (25.8, 17.9), (25.7, 25.8), (20.3, 25.6)], "ultramarine"),
    ([(7.3, 18.6), (20.6, 16.9), (21.2, 25.7), (6.6, 25.3)], "magenta"),
    ([(12.4, 9.2), (18.7, 12.2), (15.5, 18.2), (9.4, 14.7)], "violet"),
]
# the lobes of the weeds, each (t, side, turn, length, root, swell, peak, curl, end): see `weed`
WHITE = [  # round paddles, the longest reaching over into the middle
    (0.07, 1, 1.6, 470, 110, 200, 0.6, 0.2, 1.1), (0.15, -1, 1.5, 330, 105, 190, 0.6, 0.45, 1.1),
    (0.3, 1, 1.3, 600, 100, 195, 0.58, 0.5, 1.1), (0.4, -1, 1.25, 330, 95, 180, 0.6, 0.45, 1.1),
    (0.55, 1, 1.15, 400, 90, 175, 0.6, 0.5, 1.1), (0.63, -1, 1.2, 400, 88, 170, 0.6, 0.55, 1.1),
    (0.8, 1, 0.95, 380, 82, 155, 0.6, 0.35, 1.1), (0.88, -1, 1.0, 190, 78, 130, 0.6, 0.3, 1.1),
    (1.0, 1, 0.15, 420, 80, 145, 0.6, -0.2, 1.15),
]
MAGENTA = [  # fingers swept up toward the tip, one turned back against the rest
    (0.05, -1, 1.1, 400, 80, 135, 0.5, 0.45, 1.2), (0.12, 1, 1.05, 330, 78, 125, 0.5, 0.4, 1.2),
    (0.3, -1, 1.0, 480, 75, 135, 0.5, 0.45, 1.2), (0.38, 1, 0.95, 280, 72, 120, 0.5, 0.35, 1.2),
    (0.55, -1, 0.95, 360, 70, 125, 0.5, 0.4, 1.25), (0.62, 1, 1.5, 220, 60, 105, 0.5, -0.35, 1.2),
    (0.78, -1, 0.85, 260, 65, 110, 0.5, 0.35, 1.3), (0.86, 1, 0.8, 200, 60, 100, 0.5, 0.3, 1.3),
    (1.0, -1, 0.1, 280, 62, 105, 0.5, 0.15, 1.3),
]
BLACK = [  # rising the other way, leaning in toward the middle, one finger turned back
    (0.05, 1, 1.1, 380, 80, 130, 0.5, 0.45, 1.25), (0.14, -1, 1.05, 300, 76, 120, 0.5, 0.4, 1.25),
    (0.3, 1, 1.0, 500, 74, 135, 0.5, 0.45, 1.2), (0.42, -1, 1.0, 380, 72, 125, 0.5, 0.4, 1.25),
    (0.58, 1, 0.95, 330, 68, 118, 0.5, 0.4, 1.25), (0.66, -1, 1.45, 200, 58, 100, 0.5, -0.3, 1.2),
    (0.8, 1, 0.85, 260, 62, 108, 0.5, 0.35, 1.3), (0.88, -1, 0.85, 200, 58, 100, 0.5, 0.3, 1.3),
    (1.0, 1, 0.1, 300, 60, 105, 0.5, 0.15, 1.3),
]
FORMS = [  # the sea forms laid over them, bottom first: how each was cut, the hand it was cut with, its colour
    (weed, dict(base=(450, 1880), angle=-1.3, length=1300, width=(150, 90), bend=(0.45, -0.6), sway=0.22,
                lobes=WHITE), 4, "white"),
    (weed, dict(base=(1350, 900), angle=-0.55, length=950, width=(85, 55), bend=(0.1, 0.15), sway=0.18,
                lobes=MAGENTA), 2, "magenta"),
    (weed, dict(base=(2020, 2250), angle=-1.85, length=950, width=(85, 55), bend=(-0.1, 0.25), sway=0.18,
                lobes=BLACK), 3, "black"),
    (star, dict(centre=(1350, 1270), radius=230, turn=0.4), 1, "lemon"),
    (star, dict(centre=(1150, 2230), radius=190, turn=0.3), 2, "white"),
    (star, dict(centre=(320, 480), radius=170, turn=1.0), 1, "orange"),
    (star, dict(centre=(690, 2260), radius=135, turn=0.7), 1, "black"),
]


def design(r):
    """The pieces in the order they were pasted, bottom first: (outline, paint, kind)."""
    P = [(slab(np.array(c) * U, r), paint, "slab") for c, paint in SLABS]
    for i, (cut, kw, hand, paint) in enumerate(FORMS):
        P.append((cut(r=np.random.default_rng([hand, i]), **kw), paint, "form"))
    return P


# ---------------------------------------------------------------- the material

def ground(r):
    """The ground: big sheets of smooth white drawing paper joined edge over edge, a little warm,
    their formation faintly cloudy, a few fibres in them, and the slow buckles they took when they
    were pasted to the canvas. -> (colour, tooth, buckle height)"""
    formation = noise.fbm((H, W), 420, r, octaves=6, gain=0.55)
    fib = ndimage.gaussian_filter(noise.fibers((H, W), r, count=H * W // 2500, length=(6, 30), curl=0.25,
                                               strength=(0.1, 0.5)), 0.6)
    tooth = 0.6 * noise.field((H, W), 1.4, r) + 0.4 * noise.field((H, W), 3.5, r) + 0.6 * fib
    tooth = (tooth - tooth.min()) / (np.ptp(tooth) + 1e-9)
    light = 1 + 0.018 * formation + 0.014 * fib + 0.004 * noise.field((H, W), 1.0, r)
    # three sheets: the two on the left laid over the one on the right, the upper over the lower
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - (1342 + 3 * noise.line1d(H, 500, r))[:, None]
    dy = yy - (1638 + 3 * noise.line1d(W, 500, r))[None, :]
    right = noise.smoothstep(-0.7, 0.7, dx)
    low = noise.smoothstep(-0.7, 0.7, dy) * (1 - right)
    light *= 1 + 0.008 * right - 0.008 * low
    light *= 1 - 0.07 * np.exp(-np.maximum(dx, 0) / 1.6) * right - 0.07 * np.exp(-np.maximum(dy, 0) / 1.6) * low
    del xx, yy, dx, dy
    colour = lin(GROUND)[None, None, :] * light[..., None]
    colour[..., 2] *= 1 - 0.01 * low                       # one of them a shade warmer
    buckle = 6 * noise.fbm((H, W), 700, r, octaves=3, gain=0.5)
    return colour.astype(np.float32), tooth.astype(np.float32), buckle.astype(np.float32)


def gouache(shape, origin, angle, r, brush=60.0, streak=1.0):
    """The paint on a sheet as the assistant brushed it: strokes of a broad flat brush laid side by
    side along `angle`, each overlapping the last a little, its bristles leaving fine lines down its
    length, a ridge where its edge dragged, more paint where it was reloaded and less where it ran
    thin. Computed for the patch `shape` at `origin` (y0, x0) of the picture. -> thickness, about 1"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy += origin[0]
    xx += origin[1]
    ca, sa = np.cos(angle), np.sin(angle)
    u = xx * ca + yy * sa
    v = -xx * sa + yy * ca
    del xx, yy
    u0, v0 = u.min(), v.min()
    nu = int(u.max() - u0) + 2
    pitch = brush * 0.74
    nb = int((v.max() - v0) / pitch) + 4
    c = v0 - pitch + pitch * (np.arange(nb) + r.uniform(-0.06, 0.06, nb))
    width = brush * r.uniform(1.0, 1.1, nb)
    load = r.uniform(0.88, 1.12, nb)
    wander = 2.5 * noise.line1d(nu, 500, r, rows=nb)                 # no stroke is quite straight
    swell = 0.07 * noise.line1d(nu, 260, r, rows=nb)                 # nor carries its paint evenly
    fade = 0.5 + 0.5 * noise.line1d(nu, 140, r, rows=nb)             # and its streaks come and go
    m = int(nb * brush * 3)                                          # the bristle lines, one long run
    lanes = 0.5 * noise.line1d(m, 1.6, r) + 0.35 * noise.line1d(m, 3.5, r) + 0.2 * noise.line1d(m, 9, r)
    grooves = np.zeros(m, np.float32)
    grooves[r.integers(0, m, m // 60)] = r.uniform(0.6, 1.4, m // 60)
    lanes -= ndimage.gaussian_filter1d(grooves, 0.8) * 1.5           # here and there a bristle cut a clean line
    ui = np.clip((u - u0).astype(np.int32), 0, nu - 1)
    del u
    k0 = ((v - c[0]) / pitch).astype(np.int32)
    T, P = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for dk in (-1, 0, 1, 2):
        k = np.clip(k0 + dk, 0, nb - 1)
        cv = c[k] + wander[k, ui]
        d = np.abs(v - cv) / (0.5 * width[k])
        prof = noise.smoothstep(1.0, 0.6, d)
        lane = np.interp(v - cv + k * brush * 3 + brush, np.arange(m, dtype=np.float32), lanes)
        T += prof * (1 + 0.15 * noise.smoothstep(0.6, 0.85, d)) * load[k] * (1 + swell[k, ui]) \
            * (1 + 0.17 * streak * lane * (0.4 + 0.6 * fade[k, ui]))
        P += prof
    T /= 0.45 + 0.55 * np.maximum(P, 0.2)                # a stroke laid over the last drags some of it along
    return T * (1 + 0.06 * noise.field((h, w), 350, r))  # and the sheet was never brushed quite evenly


def lifting(alpha, inside, r, kind, corners):
    """How far a piece stands off what is under it, px: the thickness of the paper, the paste never
    quite flat, a blister or two where it missed, and here and there an edge or a corner that has
    come up."""
    h, w = alpha.shape
    hgt = 0.5 + 0.6 * noise.field((h, w), 160, r) * np.exp(-inside / 90) + 0.3 * noise.field((h, w), 60, r)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    deep = np.argwhere(inside > 60)
    for _ in range(r.poisson(1.5 if kind == "slab" else 0.3) if len(deep) else 0):
        y, x = deep[r.integers(len(deep))]
        s = r.uniform(18, 55)
        hgt += r.uniform(1.0, 2.5) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
    edge = np.argwhere((inside > 0.5) & (inside < 2.0))
    if len(corners):                                    # which corners and tips have come up the most
        mid = np.argwhere(alpha > 0.5).mean(0)
        d = (corners - mid) / (np.linalg.norm(corners - mid, axis=1, keepdims=True) + 1e-9)
        odds = 0.25 + np.clip(d @ [0.7, 0.7], 0, 1)
        odds /= odds.sum()
    for _ in range(r.poisson(2.0 if kind == "slab" else 1.2) if len(edge) else 0):
        if len(corners) and r.random() < 0.7:           # corners and tips come up first
            y, x = corners[r.choice(len(corners), p=odds)]
        else:
            y, x = edge[r.integers(len(edge))]
        A, D, E = r.uniform(3.0, 10.0), r.uniform(25, 60), r.uniform(60, 200)
        hgt += A * np.exp(-inside / D) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * E * E))
    return np.maximum(hgt, 0.3) * (alpha > 0)


def cast(alpha, hgt):
    """The shadow a piece casts on what lies under it, under a light from the upper left: thrown
    further and softer the more it stands off. -> darkness 0..1"""
    out = ndimage.gaussian_filter(alpha, 1.2) * 0.25                 # the edge's own hairline of shadow
    off = -LIGHT[:2] / LIGHT[2]
    for lv, lo, hi in ((0.6, 0.0, 1.5), (3.0, 1.5, 5.0), (7.0, 5.0, 99.0)):
        m = alpha * ((hgt >= lo) & (hgt < hi))
        if m.max() < 0.01 and lv > 0.6:
            continue
        m = np.maximum(m, alpha * (hgt >= hi))                       # higher parts throw this one too
        s = ndimage.shift(m, (off[1] * lv, off[0] * lv), order=1)
        out = np.maximum(out, ndimage.gaussian_filter(s, 0.6 + 0.45 * lv))
    return np.clip(out, 0, 1) * (1 - alpha)


def shading(hgt, k=0.45):
    """The light on a surface standing `hgt` px off flat, relative to flat."""
    gy, gx = np.gradient(ndimage.gaussian_filter(hgt, 1.0) * k)
    return (LIGHT[2] - gx * LIGHT[0] - gy * LIGHT[1]) / np.sqrt(1 + gx * gx + gy * gy) / LIGHT[2]


def turning(C, win):
    """How far a closed outline turns over `win` px either side of each point, radians (clockwise +)."""
    g = np.roll(C, -1, 0) - C
    a = np.arctan2(g[:, 1], g[:, 0])
    d = np.angle(np.exp(1j * (np.roll(a, -1) - a)))
    return ndimage.convolve1d(d, np.ones(2 * win), mode="wrap", origin=-1)


def ends(C, win=30, least=0.9):
    """The corners of an outline and the tips of its fingers: where it turns hardest. -> indices"""
    turn = turning(C, win)
    return np.flatnonzero((turn > least) & (turn == ndimage.maximum_filter1d(turn, 2 * win, mode="wrap")))


def pins(C, kind, r):
    """Where a piece was pinned while it was moved about the wall: near its corners and ends, a few
    holes at each place, for it was pinned more than once."""
    C = np.asarray(C)
    cx, cy = C.mean(0)
    spots = []
    for i in ends(C):
        if r.random() < (0.75 if kind == "slab" else 0.3):
            p = C[i] + (np.array([cx, cy]) - C[i]) / (np.hypot(cx - C[i, 0], cy - C[i, 1]) + 1e-9) * r.uniform(14, 40)
            spots.append(p)
    if kind == "form" or not spots:
        spots.append(np.array([cx, cy]) + r.normal(0, 15, 2))
    holes = []
    for p in spots:
        for _ in range(int(r.choice([1, 1, 2, 2, 3, 4]))):
            holes.append(p + r.normal(0, 5, 2))
    return holes


def prick(img, holes, r):
    """The holes the pins left, through everything they went through: a dark point, the paper pushed
    up round it, catching the light on one side."""
    for x, y in holes:
        rad = r.uniform(0.9, 1.6)
        x0, y0 = int(x) - 6, int(y) - 6
        if x0 < 0 or y0 < 0 or x0 + 13 > W or y0 + 13 > H:
            continue
        yy, xx = np.mgrid[y0:y0 + 13, x0:x0 + 13].astype(np.float32)
        d = np.hypot(xx - x, yy - y)
        core = noise.smoothstep(rad + 0.6, rad - 0.4, d)
        burr = np.exp(-((d - rad - 1.1) / 0.8) ** 2)
        side = ((xx - x) * LIGHT[0] + (yy - y) * LIGHT[1]) / (d + 1e-6)
        f = (1 - 0.72 * core) * (1 + 0.1 * burr * side) * (1 - 0.05 * noise.smoothstep(4.5, 1.0, d))
        img[y0:y0 + 13, x0:x0 + 13] *= f[..., None]


def setout(marks, C, corner, r, holes):
    """Before a slab was pasted its place was set out in charcoal on the ground, a short stroke
    along each side from a corner; the slab was moved after, and the marks and the holes of its
    earlier pinning stay beside it."""
    n = len(C)
    for i in corner:
        if r.random() > 0.45:
            continue
        a, b = C[(i + 45) % n] - C[i], C[i - 45] - C[i]
        a, b = a / np.hypot(*a), b / np.hypot(*b)
        p = C[i] - (a + b) / np.hypot(*(a + b)) * r.uniform(10, 34) + r.normal(0, 6, 2)
        la, lb = r.uniform(30, 80, 2)
        pencil.line(marks, [p + a * la, p + a * la * 0.5, p, p + b * lb * 0.5, p + b * lb], 5.0, r,
                    pressure=r.uniform(0.35, 0.6), wander=1.2, tremor=0.5, corners=True)
        if r.random() < 0.6:
            holes.append(p + (a + b) * r.uniform(12, 30) + r.normal(0, 3, 2))


def paint(seed=1953):
    r = noise.rng(seed)
    img, tooth, buckle = ground(r)
    marks, holes = np.zeros((H, W), np.float32), []
    stack = np.zeros((H, W), np.float32)            # how far the paper stands off the ground, px
    for C, name, kind in design(r):
        C = scissors(C, r)
        alpha, (y0, x0) = coverage(C)
        h, w = alpha.shape
        sl = (slice(max(y0, 0), min(y0 + h, H)), slice(max(x0, 0), min(x0 + w, W)))   # what of it is on the sheet
        on = (slice(sl[0].start - y0, sl[0].stop - y0), slice(sl[1].start - x0, sl[1].stop - x0))
        corner = ends(C)
        if kind == "slab":                  # charcoal on the ground, under whatever is pasted after it
            setout(marks, C, corner, r, holes)
            ys, xs = np.nonzero(marks)
            if len(ys):
                bb = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                img[bb] *= np.exp(-pencil.catch(marks[bb], Sheet(None, tooth[bb], None, None), 0.45)[..., None]
                                  * CHARCOAL)
                marks[:] = 0
        inside = ndimage.distance_transform_edt(alpha > 0.5).astype(np.float32)
        hgt = lifting(alpha, inside, r, kind, C[corner][:, ::-1] - [y0, x0])
        img[sl] *= (1 - 0.42 * cast(alpha, hgt))[on][..., None]
        # the thin paper takes the shape of what it was pasted over: an edge beneath shows through as a ridge
        under = np.zeros((h, w), np.float32)
        under[on] = stack[sl]
        surface = ndimage.gaussian_filter(under, 2.5) + hgt
        T = gouache((h, w), (y0, x0), r.uniform(0, np.pi), r, brush=r.uniform(52, 72), streak=r.uniform(0.7, 1.2))
        k = -np.log(np.clip(lin(PAINTS[name]) / lin(STOCK), 1e-4, 1))
        k = k * (1 + 0.03 * r.normal(0, 1, 3)) * (1 + 0.025 * noise.field((h, w), 300, r))[..., None]
        cov = 1 - np.exp(-2.2 * T)
        # gouache is brittle: along the cut the blades flaked a little of it off the paper
        cov *= 1 - noise.smoothstep(1.7, 2.2, noise.field((h, w), 1.4, r)) * noise.smoothstep(3.0, 1.0, inside)
        if name == "white":           # white over white paper: it covers rather than stains
            col = lin(STOCK) + (lin(PAINTS[name]) - lin(STOCK)) * cov[..., None]
        else:
            col = lin(STOCK) * np.exp(-k * cov[..., None])
        gy, gx = np.gradient(T)
        col *= (shading(surface) * (1 - 0.06 * (gx * LIGHT[0] + gy * LIGHT[1]) / LIGHT[2]))[..., None]
        # the cut edge: a hair of light along the sides that face the lamp
        ay, ax = np.gradient(ndimage.gaussian_filter(alpha, 0.7))
        facing = np.clip(-(ax * LIGHT[0] + ay * LIGHT[1]) / (np.hypot(ax, ay) + 1e-6), 0, 1)
        rim = noise.smoothstep(1.6, 0.4, inside) * facing * (0.12 + 0.03 * np.clip(hgt, 0, 8))
        col += (1 - col) * rim[..., None]
        a = alpha[on]
        img[sl] = img[sl] * (1 - a[..., None]) + col[on] * a[..., None]
        stack[sl] = stack[sl] * (1 - a) + surface[on] * a
        holes += pins(C, kind, r)
    img *= shading(buckle, 0.6)[..., None]
    prick(img, holes, r)
    return img
