"""Youth, with Two Spirals. Tempera on paper, mounted on canvas.

In Stockholm in 1907 Hilma af Klint painted The Ten Largest: ten paintings over three metres high of the
ages of a life, in forms that float and grow on grounds of one strong colour. They are in tempera on sheets
of paper joined together and mounted on canvas.

Here the ground is an orange laid flat with a broad brush, stroke beside stroke and this way and that, and
cut in round every form. Two spirals hold the picture: a pale disc high on the left wound with one ribbon of
blue, and a dark one low on the right wound with a ribbon of cream. Three circles overlap the pale spiral and
one another, and in the middle a great flower spreads six petals from one point; where circles or petals
cross, each shows through the other, painted in the colour of the two together. Round them gather a rosette
of round petals about a dark heart, a white flower inside a drawn ring, buds of two colours on their stems,
garlands of dots and tendrils that curl. Every shape is laid in flat, gone over with strokes that follow it and
drawn round with a smaller brush, so no edge is true and the colours lap a little where they meet. The paint
is opaque and matt, a little chalky where the brush ran thin; the joins of the paper show faintly, and the
sheets have cockled.
"""

import numpy as np
from scipy import ndimage

from atelier import dabs, impasto, noise
from atelier.color import lin

TITLE = "Youth, with Two Spirals"
DATE = "2026"
MEDIUM = "Tempera on paper, the sheets joined and mounted on canvas"
AFTER = "Hilma af Klint, The Ten Largest (De tio största), Stockholm, 1907"
ROOM = "Colour Itself"
YEAR = 1907
PLACE = "Stockholm"
REGION = "Europe"
NOTE = ("Two spirals, a great flower and three circles float on a ground of orange among buds, dots and "
        "tendrils; where forms cross, each shows through the other.")

H, W = 3200, 2400
S = 2                      # the plan of the picture is worked out at half size
LIGHT = (-0.5, -0.6, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
PAINTS = {
    "orange": pal("#bc4b24", "#c5552a", "#cd5f30", "#d46a37", "#da7642"),
    "salmon": pal("#d4734c", "#da815c", "#df8e6b"),
    "lemon": pal("#e8cf78", "#edd98b", "#f1e29f", "#f4e9b3"),
    "yellow": pal("#d99a1e", "#e1a829", "#e8b637", "#edc34b"),
    "white": pal("#e6ddc9", "#ece4d3", "#f1eadd", "#f4efe5"),
    "cream": pal("#eadaa6", "#efe2b8", "#f2e9c9"),
    "pink": pal("#d97f86", "#e0928f", "#e6a49f", "#ecb7b0"),
    "madder": pal("#6e1a2c", "#842336", "#9a2e41", "#ad3b4d"),
    "lavender": pal("#7686c8", "#8494d0", "#93a4d8", "#a4b3e0"),
    "blue": pal("#6f87cf", "#7d93d5", "#8b9fdb", "#9aace0"),
    "ultra": pal("#3c4f9e", "#4659a8", "#5163b0"),
    "navy": pal("#1a203d", "#20284a", "#283157", "#303b64"),
    "primrose": pal("#ecd68a", "#f0de9d", "#f3e5b0"),
    "sky": pal("#aebfe7", "#bbcaeb", "#c8d4ef"),
    "mint": pal("#97bc9c", "#a5c6a8", "#b3d0b4"),
}
KIN = {"orange": "salmon", "salmon": "orange", "lemon": "white", "yellow": "orange", "white": "cream",
       "cream": "lemon", "pink": "white", "madder": "pink", "lavender": "blue", "blue": "lavender", "ultra": "blue", "navy": "lavender",
       "primrose": "lemon", "sky": "white", "mint": "sky"}

# the forms, in full-size px
SPIRALS = [  # disc, ribbon, centre, radius, turns, half-width of the ribbon, which way it winds
    ("lemon", "lavender", (720, 800), 680, 6.0, 20, 1),
    ("navy", "cream", (1840, 2820), 490, 4.75, 17, -1),
]
CIRCLES = [("yellow", (1840, 310), 270), ("white", (1960, 780), 320), ("blue", (1500, 1080), 280)]
# every part of the overlaps of the circles and the pale spiral (bit 8) has its own mixture
MIX = {1: "yellow", 2: "white", 4: "blue", 3: "primrose", 6: "sky", 8: "lemon", 12: "mint"}
RIMS = ("yellow", "lavender", "ultra")          # the line each circle is drawn round with
FAN = ((1240, 1990), -30, [  # centre, where the petals start; each petal's heading, length, half-width, paint
    (-97, 650, 150, "pink"), (-38, 470, 132, "yellow"), (18, 560, 146, "white"), (84, 440, 128, "blue"),
    (148, 600, 152, "pink"), (-152, 470, 128, "white")], "madder")
OUTLINE = {"pink": "white", "white": "yellow", "yellow": "lavender", "blue": "white"}   # each petal's drawn edge
ROSETTE = ((560, 2790), 150, [(122, "pink"), (116, "white"), (126, "yellow"), (118, "pink"), (124, "white"),
                              (114, "yellow")], "navy", 66)
DAISY = ((330, 1800), 190, 9)      # a flower inside a drawn ring
TENDRILS = [  # start, heading (degrees), length, turns of the curl at its tip, which way it turns, paint
    ((300, 1340), 125, 420, 1.5, -1, "white"),
    ((1157, 279), -40, 380, 1.4, 1, "lemon"),
    ((1290, 2430), 95, 360, 1.3, -1, "white"),
    ((1352, 2900), 170, 400, 1.5, 1, "lemon"),
    ((2280, 800), 80, 340, 1.3, 1, "lavender"),
    ((430, 2575), -118, 340, 1.3, -1, "white"),
]
STEMS = [  # start, heading, length, bend (per px), bud length, bud half-width, bud paint, its second paint
    ((2240, 2470), -95, 560, -0.0008, 160, 50, "white", "pink"),
    ((1572, 330), 160, 160, 0.003, 130, 40, "yellow", "white"),
    ((560, 2540), -118, 170, -0.003, 140, 44, "pink", "white"),
]
BRANCHES = [  # stem, how far along it, heading, length, bend, bud length, bud half-width, bud paint, second paint
    (0, 0.38, -62, 140, -0.002, 120, 38, "pink", "madder"),
    (0, 0.7, -150, 150, -0.004, 110, 36, "lemon", "yellow"),
]
GARLANDS = [  # centre, radius, from and to (degrees) for a ring of dots round a form, spacing, dot radius, paints
    ((1840, 2820), 560, 202, 238, 36, 11, ("white",)),
    ((720, 800), 745, 198, 252, 38, 11, ("white", "pink")),
    ((560, 2800), 320, -40, 55, 34, 10, ("lemon", "white")),
]
SWAGS = [  # from, to, sag (px), spacing, dot radius, paints in turn
    ((100, 3075), (1250, 3105), 70, 44, 12, ("yellow", "white", "yellow", "lavender")),
]

# how the paint goes on: tempera dries as it leaves the brush, so it barely drags what lies beneath
TEMPERA = dict(thick=0.04, grooves=0.06, lips=0.03, land=0.15, lift=0.06, tails=0.25, pickup=0.03, merge=0.8,
               spent=0.25, ends=(0.45, 0.3), fray=0.75, flatten=0.4)
LINE = dict(TEMPERA, tails=0.3, merge=0.4, spent=0.5, ends=(0.6, 0.5), taper=0.35, fray=0.8)


def rim(r, R, n=2048):
    """A circle drawn by eye: how far out it runs at each of n angles, a little oval, with a few slow
    swells and a tremor, px."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = R + 0.012 * R * r.normal() * np.cos(2 * (t - r.uniform(0, np.pi)))
    for k in range(3, 9):
        out = out + (0.004 * R + 1.0) / k ** 0.8 * r.normal() * np.sin(k * t + r.uniform(0, 2 * np.pi))
    for k in range(9, 48, 3):
        out = out + 0.45 / np.sqrt(k / 9) * r.normal() * np.sin(k * t + r.uniform(0, 2 * np.pi))
    return out.astype(np.float32)


def reach(table, th):
    """The radius of a drawn circle at angles `th`."""
    return table[(np.mod(th / (2 * np.pi), 1) * len(table)).astype(int) % len(table)]


def petal(dx, dy, a, r0, r1, half, r):
    """A petal laid outward from the heart along `a`, at (dx, dy) from the flower's centre, bowed a little,
    its two sides drawn each on its own. -> mask, and the way round its edge from one side of the heart to
    the other, a few px inside it (relative to the centre)"""
    c, s = np.cos(a), np.sin(a)
    bow = r.normal(0, 0.12) * half
    waves = [[(k, r.normal(), r.uniform(0, 6)) for k in (1, 2, 3)] for _ in "ab"]

    def width(t, sgn):
        side = [1 + sum(0.04 / k * q * np.sin(np.pi * k * t + ph) for k, q, ph in w_) for w_ in waves]
        return half * np.sin(np.pi * t ** 1.36) ** 0.55 * np.where(sgn > 0, *side)

    u, v = dx * c + dy * s, dy * c - dx * s
    t = np.clip((u - r0) / (r1 - r0), 0, 1)
    v = v - bow * np.sin(np.pi * t)
    m = (u > r0) & (u < r1) & (np.abs(v) < width(t, v))
    T = np.linspace(0.15, 0.995, 150)
    T, sgn = np.r_[T, T[::-1]], np.r_[np.ones(150), -np.ones(150)]
    V = sgn * np.maximum(width(T, sgn) - 4, 0) + bow * np.sin(np.pi * T)
    U = r0 + T * (r1 - r0) - 4 * T ** 8
    return m, np.stack([U * c - V * s, U * s + V * c], 1)


def plan(r):
    """The picture as cells of one colour: a key for every point of the plan (half size), and for each key
    its paint, how its strokes run (0 the ground, 1 round its centre, 2 out from it, 3 straight across) and
    that centre."""
    h, w = H // S, W // S
    gy, gx = (np.mgrid[0:h, 0:w].astype(np.float32) + 0.5) * S
    key = np.zeros((h, w), np.int32)
    cells = [dict(paint="orange", mode=0, centre=(W / 2, H / 2), grade=0.0, reach=1.0)]

    def near(cx, cy, R):
        sl = (slice(max(0, int((cy - R) / S)), max(0, min(h, int((cy + R) / S) + 2))),
              slice(max(0, int((cx - R) / S)), max(0, min(w, int((cx + R) / S) + 2))))
        return sl, gx[sl] - cx, gy[sl] - cy

    def put(sl, m, **c):
        cells.append(dict(dict(grade=0.0, reach=1.0), **c))
        key[sl][m] = len(cells) - 1

    def disc(cx, cy, R, **c):
        t = rim(r, R)
        sl, dx, dy = near(cx, cy, 1.1 * R)
        put(sl, np.hypot(dx, dy) < reach(t, np.arctan2(dy, dx)), centre=(cx, cy), **c)

    rims = [rim(r, R) for _, _, _, R, *_ in SPIRALS]
    paint_, _, (cx, cy), R, *_ = SPIRALS[1]
    sl, dx, dy = near(cx, cy, 1.1 * R)
    put(sl, np.hypot(dx, dy) < reach(rims[1], np.arctan2(dy, dx)), paint=paint_, mode=1, centre=(cx, cy))
    # three circles and the pale spiral, each part of their overlaps painted in its own mixture
    tabs = [rim(r, R) for _, _, R in CIRCLES]
    centres = [c for _, c, _ in CIRCLES] + [SPIRALS[0][2]]
    code = np.zeros((h, w), np.int8)
    for i, ((cx, cy), t) in enumerate(zip(centres, tabs + rims[:1])):
        code |= (np.hypot(gx - cx, gy - cy) < reach(t, np.arctan2(gy - cy, gx - cx))).astype(np.int8) << i
    for v in np.unique(code[code > 0]):
        put(np.s_[:, :], code == v, paint=MIX[v], mode=1 if v == 8 else 3,
            centre=centres[max(i for i in range(4) if v >> i & 1 and (i < 3 or v == 8))])
    # the great flower: six petals spread from one point, each seen through the next where they cross, and
    # darkest where three of them meet
    (fx, fy), r0, fan, heart = FAN
    sl, dx, dy = near(fx, fy, 680)
    code, edges = np.zeros(dx.shape, np.int16), []
    for j, (a, L, half, p) in enumerate(fan):
        mask, edge = petal(dx, dy, np.radians(a + r.normal(0, 2)), r0, L, half, r)
        code |= mask.astype(np.int16) << j
        edges.append((edge + [fx, fy], OUTLINE[p]))
    for v in np.unique(code[code > 0]):
        on = [j for j in range(len(fan)) if v >> j & 1]
        name = "+".join(sorted({fan[j][3] for j in on})[:2])
        put(sl, code == v, paint=name if name in PAINTS else heart, mode=2, centre=(fx, fy), grade=0.3,
            reach=fan[on[0]][1])
    disc(fx, fy, 46, paint=heart, mode=1)
    # a rosette of round petals that overlap and show through one another, round a dark heart
    (ox, oy), ring, petals, heart, hr = ROSETTE
    sl, dx, dy = near(ox, oy, ring + 140)
    a0, code = r.uniform(0, 2 * np.pi), np.zeros(dx.shape, np.int16)
    for j, (R, p) in enumerate(petals):
        a = a0 + 2 * np.pi * (j + r.uniform(-0.05, 0.05)) / len(petals)
        px, py, t = ring * np.cos(a), ring * np.sin(a), rim(r, R)
        code |= (np.hypot(dx - px, dy - py) < reach(t, np.arctan2(dy - py, dx - px))).astype(np.int16) << j
    for v in np.unique(code[code > 0]):
        ps = sorted({p for j, (_, p) in enumerate(petals) if v >> j & 1})
        put(sl, code == v, paint=heart if bin(v).count("1") > 2 else "+".join(ps), mode=3, centre=(ox, oy))
    disc(ox, oy, hr, paint=heart, mode=1)
    # the buds at the ends of their stems, each an egg of two colours, one on each side
    stems = []
    for p, a, L, bend, bl, bw, col, col2 in STEMS + [(None, *b_[2:]) for b_ in BRANCHES]:
        if p is None:                   # a side stem, set on a stem already drawn
            j, f = BRANCHES[len(stems) - len(STEMS)][:2]
            p = stems[j][int(f * (len(stems[j]) - 1))]
        n = max(8, int(L / 3))
        ang = np.radians(a) + bend * np.arange(n) * L / n + 0.0015 * np.cumsum(noise.line1d(n, n / 3, r)) * L / n
        C = np.asarray(p, float) + np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * L / n, 0)
        stems.append(C)
        a = ang[-1]
        bx, by = C[-1] - 8 * np.array([np.cos(a), np.sin(a)])
        sl, dx, dy = near(bx, by, bl + 20)
        mask, _ = petal(dx, dy, a, 0, bl, bw, r)
        side = dy * np.cos(a) - dx * np.sin(a) > 0
        for m, pc in ((mask & side, col), (mask & ~side, col2)):
            put(sl, m, paint=pc, mode=2, centre=(bx, by), grade=0.2, reach=bl)
    return key, cells, rims, tabs, edges, stems


def at(P):
    """Where points (x, y) of the canvas fall on the plan."""
    P = np.asarray(P)
    return (np.clip((P[..., 1] / S).astype(int), 0, H // S - 1), np.clip((P[..., 0] / S).astype(int), 0, W // S - 1))


def either(paths, r, odds=0.5):
    """Some strokes go one way, some the other."""
    back = r.random(len(paths)) < odds
    paths[back] = paths[back, ::-1]
    return paths


def tones(fam, t):
    """A paint at tone t (0 its darkest, 1 its lightest), between the tones it was mixed in."""
    k = len(fam) - 1
    x = np.clip(t, 0, 1) * k
    i = np.minimum(x.astype(int), k - 1)
    f = (x - i)[:, None]
    return fam[i] * (1 - f) + fam[i + 1] * f


for _a, _b in (("pink", "white"), ("white", "yellow"), ("pink", "yellow"), ("blue", "white"), ("blue", "pink")):
    # petals seen through petals
    PAINTS[_a + "+" + _b] = (tones(PAINTS[_a], np.linspace(0, 1, 3)) + tones(PAINTS[_b], np.linspace(0, 1, 3))) / 2
    KIN[_a + "+" + _b] = _a


def load(paints, tone, r, streak=0.1, odds=0.12):
    """The load of each brush: its tone of the paint, a slightly lighter or darker one streaked through it,
    and now and then a touch of a paint akin to it."""
    N = len(tone)
    cols = np.empty((N, 3, 3), np.float32)
    for p in np.unique(paints):
        sel = np.nonzero(paints == p)[0]
        fam, t, m = PAINTS[p], tone[sel], len(sel)
        third = tones(fam, t + r.normal(0, 2 * streak, m))
        hit = r.random(m) < odds
        third[hit] = tones(PAINTS[KIN[p]], t[hit])
        cols[sel] = np.stack([tones(fam, t), tones(fam, t + r.normal(0, streak, m)), third], 1)
    return cols, np.stack([r.uniform(0.5, 0.75, N), r.uniform(0.2, 0.4, N), r.uniform(0.03, 0.12, N)], 1)


def cut(paths, key, cell, dist, margin, r):
    """Each stroke stops where it would leave its cell, or come nearer the edge than `margin` px,
    somewhere in the last step. -> the strokes that are left, and the index of each"""
    iy, ix = at(paths)
    ok = (key[iy, ix] == cell[:, None]) & (dist[iy, ix] >= margin[:, None])
    ok[:, 0] = True
    stop = np.where(ok.all(1), paths.shape[1], np.argmin(ok, 1))
    out, keep = [], []
    for i, k in enumerate(stop):
        if k >= 2:
            q = paths[i, :k]
            if k < paths.shape[1]:
                q = np.vstack([q, q[-1] + r.uniform(0, 1) * (paths[i, k] - q[-1])])
            out.append(q)
            keep.append(i)
    return out, np.array(keep, int)


def pulls(C, r, lo=280, hi=620, lap=(10, 28)):
    """A long line drawn in several pulls of the brush, each set down a little before the last one lifted."""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    out, a = [], 0.0
    while a < s[-1] - 20:
        b = a + r.uniform(lo, hi)
        if b > s[-1] - 0.4 * lo:
            b = s[-1]
        m = (s >= a) & (s <= b)
        if m.sum() > 2:
            out.append(C[m])
        a = b - r.uniform(*lap)
    return out


def coil(centre, table, turns, hw, sense, r, step=4.0):
    """The ribbon of a spiral as one path: once round just inside the rim of its disc, then winding in, the
    turns a little uneven, to the middle."""
    cx, cy = centre
    a0 = r.uniform(0, 2 * np.pi)
    th = np.linspace(0, 2 * np.pi * (turns + 1), 40000)
    pitch = (float(table.mean()) - hw) / (turns + 0.55)
    local = pitch * (1 + 0.08 * noise.line1d(len(th), 4000, r)) * (th > 2 * np.pi)
    rho = reach(table, a0 + sense * th) - hw - np.cumsum(local) * (th[1] - th[0]) / (2 * np.pi)
    C = np.stack([cx + rho * np.cos(a0 + sense * th), cy + rho * np.sin(a0 + sense * th)], 1)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    si = np.arange(0, s[-1], step)
    C = np.stack([np.interp(si, s, C[:, 0]), np.interp(si, s, C[:, 1])], 1)
    g = np.gradient(C, axis=0)
    g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
    return C + g[:, ::-1] * [-1, 1] * 1.4 * noise.line1d(len(C), 40, r)[:, None]


def curl(p, a, L, turns, sense, r, kick=0.0):
    """A tendril as the brush drew it: out from `p` along `a` degrees, turning slowly, then winding
    `turns` times into a tight curl at its tip. -> points"""
    n = max(8, int(L / 3))
    s = np.linspace(0, 1, n)
    k = sense * (2 * np.pi * turns * 3.5 / L * s ** 2.5) + kick * np.sin(2 * np.pi * (1.1 * s + r.uniform(0, 1))) \
        + 0.0015 * noise.line1d(n, n / 4, r)
    ang = np.radians(a) + np.cumsum(k) * L / n
    return np.asarray(p, float) + np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * L / n, 0)


def swag(p0, p1, sag, spacing, r):
    """Where the dots of a garland go, hung from p0 to p1 and sagging `sag` px, never quite evenly."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    t = np.linspace(0, 1, 400)[:, None]
    C = p0 + d * t + np.array([-d[1], d[0]]) / np.hypot(*d) * sag * 4 * t * (1 - t)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    k = max(2, int(s[-1] / spacing))
    si = (np.arange(k) + 0.5 + r.uniform(-0.18, 0.18, k)) * s[-1] / k
    return np.stack([np.interp(si, s, C[:, 0]), np.interp(si, s, C[:, 1])], 1)


def paint(seed=1907):
    key, cells, rims, tabs, edges, stems = plan(np.random.default_rng([seed, 0]))
    g, r = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])     # the cells' choices; the strokes
    n, (h, w) = len(cells), key.shape
    gy, gx = (np.mgrid[0:h, 0:w].astype(np.float32) + 0.5) * S
    paints = np.array([c["paint"] for c in cells])
    mode = np.array([c["mode"] for c in cells])
    cx, cy = (np.array([c["centre"][i] for c in cells], np.float32) for i in (0, 1))
    grade = np.array([c["grade"] for c in cells])
    span = np.array([c["reach"] for c in cells])

    # the edges of the cells, how far each point lies inside its own, and which way the edge runs there
    b = np.zeros(key.shape, bool)
    b[:-1] |= key[:-1] != key[1:]
    b[1:] |= key[1:] != key[:-1]
    b[:, :-1] |= key[:, :-1] != key[:, 1:]
    b[:, 1:] |= key[:, 1:] != key[:, :-1]
    dist, (qy, qx) = ndimage.distance_transform_edt(~b, return_indices=True)
    dist = ((dist + 0.5) * S).astype(np.float32)
    normal = np.arctan2(gy - (qy + 0.5) * S, gx - (qx + 0.5) * S)
    c2, s2 = (ndimage.gaussian_filter(f(2 * normal), 1.5) for f in (np.cos, np.sin))
    tangent = (0.5 * np.arctan2(s2, c2) + np.pi / 2).astype(np.float32)
    breadth = 2 * ndimage.maximum(dist, key, np.arange(n))

    # how each cell is painted: which way its strokes run, and how big they are
    wide = np.clip(0.15 * breadth, 7, 20) * g.uniform(0.85, 1.15, n)
    long_ = wide * g.uniform(5, 9, n)
    long_ = np.where(mode == 2, np.minimum(long_, 0.6 * span), long_)
    tone = g.uniform(0.35, 0.65, n)
    radial = np.arctan2(gy - cy[key], gx - cx[key])
    wander = (1.0 + 1.1 * noise.field(key.shape, 700 / S, r) + 0.25 * noise.field(key.shape, 260 / S, r)).astype(np.float32)
    hug = np.exp(-dist / 120) * (key == 0)          # near a form the ground strokes turn to go round it
    ground = 0.5 * np.arctan2((1 - hug) * np.sin(2 * wander) + hug * np.sin(2 * tangent),
                              (1 - hug) * np.cos(2 * wander) + hug * np.cos(2 * tangent))
    m = mode[key]
    rho = np.hypot(gx - cx[key], gy - cy[key])
    level = g.uniform(0, np.pi, n)[key]
    across = (m == 3) | ((m == 1) & (rho < 2.2 * wide[key]))     # round strokes would turn too tight at the middle
    field = (np.select([m == 0, across, m == 1], [ground, level, radial + np.pi / 2], radial)
             + 0.1 * noise.field(key.shape, 150 / S, r)).astype(np.float32)
    drift = 0.12 * noise.field(key.shape, 500 / S, r)          # where a colour runs darker or lighter

    # the paper: a smooth sheet with a little tooth, joined to its neighbours by narrow overlaps
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tooth = 0.6 * noise.field((H, W), 1.3, r) + 0.4 * noise.field((H, W), 4, r)
    lap = np.zeros((H, W), np.float32)
    for x0 in (790, 1585):
        x = x0 + 5 * noise.line1d(H, 900, r)[:, None]
        lap += noise.smoothstep(x - 0.8, x + 0.8, xx) * noise.smoothstep(x + 10, x + 4, xx)
    y = 1655 + 5 * noise.line1d(W, 900, r)[None, :]
    lap += noise.smoothstep(y - 0.8, y + 0.8, yy) * noise.smoothstep(y + 10, y + 4, yy)
    del yy, xx
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = lin("#e4d9c2")
    height = (0.25 * tooth + 0.3 * lap).astype(np.float32)
    wet = np.zeros((H, W), np.float32)

    def coat(where, col, inset):
        """A flat first coat over `where`, stopping a little short of the ground, thin enough that what lies
        beneath warms it here and there."""
        d = ndimage.zoom((ndimage.distance_transform_edt(where) * S).astype(np.float32), S, order=1)[:H, :W]
        a = noise.smoothstep(inset, inset + 2.5, d + 0.8 * noise.field((H, W), 10, r)) \
            * (0.9 + 0.05 * noise.field((H, W), 60, r) + 0.03 * noise.field((H, W), 8, r))
        rgb[:] += (col - rgb) * np.clip(a, 0, 1)[..., None]

    def seeds(spacing, odds):
        """Where strokes start: a shaken honeycomb over the canvas and a little beyond, thinned cell by
        cell by `odds`."""
        P = dabs.scatter((H + 120, W + 120), spacing, r) - 60
        c = key[at(P)]
        keep = r.random(len(P)) < odds[c]
        return P[keep], c[keep]

    def follow(angle, P, L, steps, tilt=0.0):
        return impasto.follow(angle, P / S, L / S, steps, 0.0, tilt) * S

    def lay(paths, width, paint_, tone_, how, **kw):
        cols, share = load(np.asarray(paint_), np.asarray(tone_, float), r)
        impasto.lay(rgb, height, paths, width, cols, r, share=share, wet=wet, **dict(how, **kw))

    # the ground laid in all over, first in a flat coat and then with a broad brush this way and that
    rgb[:] += (PAINTS["orange"][2] * (1 + 0.05 * noise.field((H, W), 400, r))[..., None] - rgb) * 0.94
    P = dabs.scatter((H + 120, W + 120), 40, r) - 60
    P = P[r.random(len(P)) < 0.866 * 40 ** 2 * 3.0 / (2 * 30 * 180)]
    N = len(P)
    paths = either(follow(wander, P, 180 * np.exp(r.normal(0, 0.25, N)), 8, r.normal(0, 0.5, N)), r)
    lay(paths, 30 * np.exp(r.normal(0, 0.12, N)), np.full(N, "orange"), 0.5 + drift[at(P)] + r.normal(0, 0.1, N),
        TEMPERA)
    wet[:] = 0

    # the forms, each laid in flat and then gone over with strokes that follow it, the lightest strokes of a
    # petal toward its tip
    mid = np.stack([tones(PAINTS[p], np.array([0.5]))[0] for p in paints]) * (1 + 0.03 * g.normal(0, 1, (n, 1)))
    coat(key > 0, ndimage.zoom(mid[key], (S, S, 1), order=1)[:H, :W], 1.0)
    P, c = seeds(12, np.where(mode > 0, 0.866 * 12 ** 2 * 3.0 / (2 * wide * long_), 0))
    inner = dist[at(P)] >= 0.9 * wide[c]
    P, c = P[inner], c[inner]
    paths = follow(field, P, long_[c] * np.exp(r.normal(0, 0.25, len(c))), 12,
                   r.normal(0, np.where(mode[c] == 3, 0.4, 0.08)))
    paths, i = cut(either(paths, r), key, c, dist, wide[c] * r.uniform(0.35, 0.7, len(c)), r)
    c, P = c[i], P[i]
    far = np.hypot(P[:, 0] - cx[c], P[:, 1] - cy[c]) / span[c]
    lay(paths, wide[c] * np.exp(r.normal(0, 0.12, len(c))), paints[c],
        tone[c] + grade[c] * (far - 0.5) + drift[at(P)] + r.normal(0, 0.07, len(c)), TEMPERA)

    # the ground gone over again round the forms, stopping short of them
    P, c = seeds(14, np.where(mode == 0, 0.866 * 14 ** 2 * 1.2 / (2 * 20 * 140), 0))
    inner = dist[at(P)] >= 18
    P, c = P[inner], c[inner]
    paths = follow(field, P, 140 * np.exp(r.normal(0, 0.25, len(c))), 10, r.normal(0, 0.15, len(c)))
    paths, i = cut(either(paths, r), key, c, dist, 20 * r.uniform(0.1, 0.5, len(c)), r)
    P = P[i]
    lay(paths, 20 * np.exp(r.normal(0, 0.12, len(i))), np.full(len(i), "orange"),
        0.5 + drift[at(P)] + r.normal(0, 0.1, len(i)), TEMPERA)

    # then every edge drawn round with a smaller brush, now a hair over the next colour, now short of it
    we = np.clip(0.5 * wide, 4, 9)
    Le = r.uniform(110, 240, n)
    band = (dist >= 0.5 * we[key]) & (dist <= 1.5 * we[key])
    iy, ix = np.nonzero(band & (r.random(key.shape) < S * S / (we[key] * 0.45 * Le[key])))
    P = np.stack([(ix + r.random(len(ix))) * S, (iy + r.random(len(iy))) * S], 1)
    c = key[iy, ix]
    paths = follow(tangent, P, Le[c], 16)
    hair = noise.field(key.shape, 120 / S, r)
    py, px = at(paths)
    q = np.stack([(qx[py, px] + 0.5) * S, (qy[py, px] + 0.5) * S], -1)
    v = paths - q
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    depth = we[c][:, None] * (0.92 + 0.28 * np.clip(hair[py, px], -1.5, 1.5))
    paths = np.where(nv > 0.5, q + v / np.maximum(nv, 1e-6) * depth[..., None], paths).astype(np.float32)
    py, px = at(paths)
    ok = key[py, px] == c[:, None]
    ok[:, 1:] &= np.linalg.norm(np.diff(q, axis=1), axis=-1) < 3 * Le[c][:, None] / 15    # a corner: the edge jumps
    seg = np.diff(paths, axis=1)
    ok[:, 2:] &= np.abs(np.angle(np.exp(1j * np.diff(np.arctan2(seg[..., 1], seg[..., 0]), axis=1)))) < 0.5
    stop = np.where(ok.all(1), ok.shape[1], np.argmin(ok, 1))
    keep = np.nonzero(stop >= 4)[0]
    keep = keep[r.permutation(len(keep))]
    c, P = c[keep], P[keep]
    far = np.hypot(P[:, 0] - cx[c], P[:, 1] - cy[c]) / span[c]
    lay([paths[i, :stop[i]][::r.choice([-1, 1])] for i in keep], we[c] * np.exp(r.normal(0, 0.1, len(c))), paints[c],
        np.where(mode[c] == 0, 0.5, tone[c]) + grade[c] * (far - 0.5) + drift[at(P)] + r.normal(0, 0.06, len(c)),
        TEMPERA, fray=0.45)
    wet[:] = 0

    # over the dry paint: tendrils and stems, and the petals of the flower in the ring
    lines, wl, lp = [], [], []
    for p, a, L, turns, sense, col in TENDRILS:
        for C in pulls(curl(p, a, L * 1.3, turns, sense, r, r.choice([-1, 1]) * r.uniform(0.002, 0.004)), r, 220, 420):
            lines.append(C), wl.append(r.uniform(3.6, 4.6)), lp.append(col)
    for C in stems:                    # stopping where the bud sits on it
        lines.append(C[:-4]), wl.append(r.uniform(4.0, 5.0)), lp.append("cream")
    lay(lines, np.array(wl), np.array(lp), r.uniform(0.4, 0.7, len(lines)), LINE)
    strokes, ws, ps, ts = [], [], [], []
    t = np.linspace(0, 1, 6)[:, None]
    (ox, oy), R, k = DAISY
    a0 = r.uniform(0, 2 * np.pi)
    for j in range(k):
        a = a0 + 2 * np.pi * (j + r.uniform(-0.08, 0.08)) / k
        e, f = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        r1 = R * r.uniform(0.74, 0.84)
        strokes.append(np.array([ox, oy]) + e * (34 + t * (r1 - 34)) + f * r.normal(0, 6) * np.sin(np.pi * t))
        ws.append(r.uniform(20, 25)), ps.append("white"), ts.append(r.uniform(0.45, 0.8))
    lay(strokes, np.array(ws), np.array(ps), np.array(ts), dict(TEMPERA, ends=(0.5, 0.75), taper=0.4))

    # the ribbons of the spirals, each one line of colour drawn in several pulls, and the circles drawn round
    for (_, col, centre, R, turns, hw, sense), table in zip(SPIRALS, rims):
        P = pulls(coil(centre, table, turns, hw, sense, r), r)
        lay(P, hw * r.uniform(0.92, 1.08, len(P)), np.full(len(P), col), r.uniform(0.35, 0.6, len(P)),
            dict(LINE, spent=0.25, fray=0.6))
    for (x, y), table, rc in [(c_, t_, rc_) for (_, c_, _), t_, rc_ in zip(CIRCLES, tabs, RIMS)] + \
            [(DAISY[0], rim(r, DAISY[1]), "yellow")]:
        th = r.uniform(0, 2 * np.pi) + np.linspace(0, 2 * np.pi * 1.06, 900)
        rr = reach(table, th) - 6 + 1.5 * noise.line1d(len(th), 30, r)
        P = pulls(np.stack([x + rr * np.cos(th), y + rr * np.sin(th)], 1), r, 300, 700)
        lay(P, r.uniform(5.8, 7.2, len(P)), np.full(len(P), rc), np.full(len(P), 0.15), LINE)
    P, cols = zip(*[(q, col) for edge, col in edges for q in pulls(edge, r, 200, 450)])
    lay(list(P), r.uniform(3.6, 4.6, len(P)), np.array(cols), r.uniform(0.3, 0.6, len(P)), LINE)

    # last the dots, each one touch of the brush: garlands round the forms and hung between them, and the
    # hearts of the small flowers
    def dots(P, size, names):
        cols = np.stack([tones(PAINTS[q], np.array([r.uniform(0.4, 0.75)]))[0] for q in names])
        dabs.lay(rgb, height, np.asarray(P, np.float32), cols, r, size=(size, 0.85 * size), spread=3.0, square=2.1)

    for (ox, oy), R, a0, a1, spacing, size, cols in GARLANDS:
        k = max(2, int(np.radians(a1 - a0) * R / spacing))
        th = np.radians(a0 + (a1 - a0) * (np.arange(k) + 0.5 + r.uniform(-0.18, 0.18, k)) / k)
        dots(np.stack([ox + R * np.cos(th), oy + R * np.sin(th)], 1) + r.normal(0, 2, (k, 2)), size,
             [cols[j % len(cols)] for j in range(k)])
    for p0, p1, sag, spacing, size, cols in SWAGS:
        P = swag(p0, p1, sag, spacing, r)
        dots(P, size, [cols[j % len(cols)] for j in range(len(P))])
    (ox, oy), hr = ROSETTE[0], ROSETTE[4]
    a = np.linspace(0, 2 * np.pi, 9)[:-1] + r.uniform(0, 1)
    dots(np.stack([ox + 0.6 * hr * np.cos(a), oy + 0.6 * hr * np.sin(a)], 1), 8, ["lemon"] * 8)
    a, d = r.uniform(0, 2 * np.pi, 12), 24 * np.sqrt(r.uniform(0, 1, 12))
    dots(np.stack([DAISY[0][0] + d * np.cos(a), DAISY[0][1] + d * np.sin(a)], 1), 9, ["madder"] * 12)
    dots(np.array([FAN[0], DAISY[0]], float), 12, ["lemon", "lemon"])

    # the light falls across the sheet, showing the brushwork faintly, the joins and the slow buckles of the paper
    bumps = 2.5 * noise.fbm((H, W), 420, r, octaves=3)
    img = dabs.shine(rgb, 0.15 * height + bumps + 0.25 * ndimage.gaussian_filter(lap, 0.8), light=LIGHT, relief=1.0,
                     gloss=0, reach=(0.85, 1.15))
    return img * (1 + 0.012 * noise.field((H, W), 1.0, r))[..., None]
