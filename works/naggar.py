"""The Last Light on the Snows. Tempera on canvas.

From 1928 Nicholas Roerich lived at Naggar in the Kullu valley, and from there, as from Darjeeling and Sikkim
before, he painted the Himalayas over and over in tempera on canvas and board: one great mass of snow, or a short
range, standing in a large quiet sky of turquoise and violet, the snow lit rose or gold and modelled in a few
broad planes, the foothills below it plain dark shapes. Tempera dries almost as it leaves the brush, so it cannot
be blended: a colour changes in a band, or in strokes laid beside and over other strokes, and a light paint
dragged nearly dry over a darker one breaks on the weave and lets the darker show between.

Here the sun has left the valley and catches only the snows of one massif: three summits, the highest a little right
of the middle, joined by rounded shoulders of snow. The sky went on first in long level strokes, band by band:
ultramarine, violet, a band of gold and rose cloud, then turquoise down to the snows. Each summit was laid in a few
broad planes, the faces turned to the sun in a rose darker than their light, the faces turned away in violet and
ultramarine, and gone over with strokes down the fall of the slope; where a face rounds over into a shoulder its
colour changes in bands that lie side by side, and the strokes run across them. Then the light, cream, gold and
salmon, was dragged over the sunlit planes with a brush almost dry, thinning out as the snow turns away, and a paler
blue over the shaded ones. The foot of the snows goes down into a band of mist laid in long strokes across, and
veils of it were dragged up over them. Under the mist stand three bands of foothills, blue, violet and plum, the
nearest darkest, each brushed along its slope, with lilac light dragged along its crest; on the middle one stands a
small white stupa. The paint is matt and thin, and the weave of the canvas shows through it.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise, tempera
from atelier.color import lin

TITLE = "The Last Light on the Snows"
DATE = "2026"
MEDIUM = "Tempera on canvas, laid in flat and gone over with a brush almost dry"
AFTER = ("Nicholas Roerich, the Himalayan paintings in tempera, 1924–47: the Himalayas series; Mount of Five "
         "Treasures (Kanchenjunga), 1933; Sacred Himalayas, 1933; the Tibet series; Path to Shambhala, 1933")
ROOM = "Open Air"
YEAR = 1935
PLACE = "Naggar, Kullu Valley"
REGION = "Europe"
NOTE = ("The last of the sun on one massif of snow: three summits laid in a few broad planes, rose and gold where "
        "they face the sun and violet where they turn away, over a band of mist and three dark bands of foothills. "
        "The light is dragged on nearly dry, so it breaks on the canvas.")

H, W = 2160, 3240
S = 2                               # the plan is worked out at half size
F = 6000.0                          # the eye: a long focal length (px), as from far off; where it looks, its level
CX, CY = W / 2, 0.70 * H
Z0, Z1 = 30.0, 66.0                 # the snows are surveyed in slices of depth between these
SUN = np.array([-0.95, 0.15, 0.1]) / np.linalg.norm([-0.95, 0.15, 0.1])  # low in the west: (right, up, away)
KC = 0.4                            # how softly the summits and shoulders run into one another
ROUGH = 0.03                        # how unevenly the snow lies, so that no edge of it runs dead clean

PEAKS = [  # the summits: on the canvas (px) at a depth; the faces they fall in, each its way (azimuth, degrees:
    # 0 right, 90 away, 180 left, 270 toward us) and its slope; how crisp the edges are at the top, and how fast
    # they round off lower down
    (1780, 500, 52.0, ((160, 0.55), (205, 0.95), (265, 1.15), (335, 1.05), (15, 0.85), (95, 0.6)), (0.03, 0.14)),
    (930, 700, 50.0, ((165, 0.75), (225, 0.9), (330, 0.8), (25, 0.5), (95, 0.7)), (0.40, 0.20)),
    (2600, 740, 50.0, ((170, 0.7), (235, 0.85), (335, 0.95), (15, 0.55), (95, 0.7)), (0.55, 0.25)),
]
DOMES = [  # the rounded shoulders and snowfields: on the canvas (px) at a depth, their slopes across and in
    # depth, and how broad their round tops are
    (1430, 740, 52.0, (0.6, 0.8), 0.35),
    (1290, 860, 50.5, (0.6, 0.8), 0.4),
    (2130, 780, 51.5, (0.7, 0.8), 0.4),
    (330, 990, 49.0, (0.45, 0.6), 0.6),
    (3020, 940, 49.0, (0.45, 0.6), 0.6),
    (1650, 893, 48.5, (0.35, 0.3), 0.4),
    (2560, 925, 47.0, (0.4, 0.3), 0.4),
]
MIST = (1190, 40)                   # the top of the mist on the canvas (px), and how far it wanders


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


HILLS = [  # the bands of foothills, far to near: the middle of the crest on the canvas (px), how far it rises and
    # falls, how much higher it stands at the right than at the left, and its colour
    (1400, 22, 30, "#3d47a4"),
    (1595, 40, 120, "#2f2b86"),
    (1880, 50, -260, "#24174d"),
]
RAMP = pal("#2f3796", "#3f4bb0", "#5560c4", "#7470cf", "#9a86d4", "#c895c9", "#eb9fae", "#f4ae9c", "#f7c9a2",
           "#f9e1b4")
HATCH = pal("#6a72c2", "#2c3178", "#2e1f5e")      # dragged over each band dry, toward its foot
CRESTS = pal("#7c82d2", "#6c64c4", "#74539e")     # and the last light along its crest
SKY = [(0.000, "#2e3292"), (0.080, "#4b40aa"), (0.135, "#f1c497"), (0.205, "#3fc0c6"), (0.300, "#66cfcb"),
       (0.420, "#94dcca")]
STREAKS = [  # long streaks of cloud across the sky: from, to (of the width), middle, half-thickness (of the height),
    # and their colours, the upper part's two and the lower part's
    (0.00, 0.60, 0.040, 0.007, ("#4a3fa6", "#5c4cb4", "#24287e")),
    (0.35, 1.00, 0.095, 0.006, ("#5a4cb8", "#6a58c0", "#3a3494")),
    (0.05, 0.75, 0.150, 0.013, ("#f0c66e", "#f7dea0", "#ee9e9a")),
    (0.40, 1.00, 0.180, 0.011, ("#f3b8a4", "#f7d29a", "#d98aa6")),
    (0.00, 0.45, 0.192, 0.008, ("#f6e2b4", "#f2c48a", "#eba2a2")),
    (0.55, 0.98, 0.255, 0.005, ("#8ad8d0", "#a8e4d4", "#5cc8cc")),
    (0.02, 0.40, 0.330, 0.005, ("#8ad8d0", "#a8e4d4", "#5cc8cc")),
]
MISTC = pal("#8a8cd4", "#9696da", "#7f9cd4")      # the mist, in long streaks of these

# how tempera goes on: thin, and dry almost as it leaves the brush, so it drags nothing up from beneath
TEMPERA = dict(thick=0.02, grooves=0.04, lips=0.02, land=0.12, lift=0.04, tails=0.3, pickup=0.0, merge=0.3,
               spent=0.35, ends=(0.5, 0.35), taper=0.25, fray=0.9, flatten=0.0)


def ramp(stops, t):
    """The colour at t (0..1) along `stops`, from dark to light."""
    k = len(stops) - 1
    x = np.clip(t, 0, 1) * k
    i = np.minimum(x.astype(int), k - 1)
    f = (x - i)[..., None]
    L = np.log(stops)
    return np.exp(L[i] * (1 - f) + L[i + 1] * f).astype(np.float32)


def world(X, y, z):
    """The point of the land at depth z that the eye sees at (X, y) on the canvas."""
    return (X - CX) * z / F, (CY - y) * z / F


def peak(q, x, z):
    """A summit: flat faces that meet in crisp edges near the top and round off lower down, as snow lies on
    them. -> height at (x, z), and which face"""
    X, y, zp, fs, (k0, k1) = q
    px, ph = world(X, y, zp)
    dx, dz = x - px, z - zp
    e = np.stack([s * (np.cos(np.radians(a)) * dx + np.sin(np.radians(a)) * dz) for a, s in fs])
    m = e.max(0)
    k = k0 + k1 * np.maximum(m, 0)
    return ph - m - k * np.log(np.exp((e - m) / k).sum(0)), e.argmax(0)


def dome(q, x, z):
    """A rounded shoulder of snow, its flanks falling at their slopes across and in depth. -> height, face"""
    X, y, zp, (sx, sz), rho = q
    px, ph = world(X, y, zp)
    return ph + rho - np.sqrt((sx * (x - px)) ** 2 + (sz * (z - zp)) ** 2 + rho ** 2), np.zeros(x.shape, np.int32)


PIECES = [(peak, q) for q in PEAKS] + [(dome, q) for q in DOMES]


def height(x, z):
    """How high the snows stand at (x, z): the summits and shoulders running softly into one another.
    -> height, which piece is highest there, and which of its faces"""
    m = None
    for i, (kind, q) in enumerate(PIECES):
        h, f = kind(q, x, z)
        if m is None:
            m, acc, best, face = h, np.ones_like(h), np.zeros(h.shape, np.int32), f
            continue
        top = np.maximum(m, h)
        acc = acc * np.exp((m - top) / KC) + np.exp((h - top) / KC)
        up = h > m
        best[up], face[up] = i, f[up]
        m = top
    return m + KC * np.log(acc), best, face


def survey(r):
    """The snows in slices of depth, near to far: in each slice, for each column of the plan, how high the snow
    stands, a little uneven, and which piece and face of the massif it is. -> depths, heights, piece * 64 + face"""
    zs = np.arange(Z0, Z1, 0.01, dtype=np.float32)
    X = ((np.arange(W // S) + 0.5) * S).astype(np.float32)
    grain = 0.05
    rough = noise.field((int((Z1 - Z0) / grain) + 4, int(40 / grain)), 8, r)
    Y = np.empty((len(zs), len(X)), np.float32)
    ident = np.empty((len(zs), len(X)), np.int32)
    for a in range(0, len(zs), 300):
        z = np.repeat(zs[a:a + 300, None], len(X), 1)
        x = (X[None, :] - CX) * z / F
        hgt, piece, face = height(x, z)
        bump = ndimage.map_coordinates(rough, [(z - Z0) / grain, (x + 20) / grain], order=1, mode="nearest")
        Y[a:a + 300] = np.minimum(hgt + ROUGH * bump, 3.0 * (z - Z0) - 1.0)
        ident[a:a + 300] = piece * 64 + face
    return zs, Y, ident


def look(zs, Y):
    """Which slice the eye meets first along the ray through each point of the plan (past the last where it
    meets only sky)."""
    m = np.minimum.accumulate(CY - F * Y / zs[:, None], axis=0)
    rows = ((np.arange(H // S) + 0.5) * S).astype(np.float32)
    first = np.empty((H // S, m.shape[1]), np.int32)
    for c in range(m.shape[1]):
        first[:, c] = np.searchsorted(-m[:, c], -rows, side="left")
    return first


def crest(mid, amp, lean, r):
    """The crest of a band of foothills, for each column of the plan: soft and slowly undulating, as a brush
    draws it. -> y, px"""
    w = W // S
    x = (np.arange(w) + 0.5) * S
    return (mid + amp * (0.8 * noise.line1d(w, 500, r) + 0.35 * noise.line1d(w, 170, r))
            - lean * (x / W - 0.5) + 1.2 * noise.line1d(w, 6, r)).astype(np.float32)


def at(P):
    """Where points (x, y) of the canvas fall on the plan."""
    P = np.asarray(P)
    return (np.clip((P[..., 1] / S).astype(int), 0, H // S - 1), np.clip((P[..., 0] / S).astype(int), 0, W // S - 1))


def either(paths, r, odds=0.5):
    """Some strokes go one way, some the other."""
    back = r.random(len(paths)) < odds
    paths[back] = paths[back, ::-1]
    return paths


def cut(paths, key, cell, dist, margin, r):
    """Each stroke stops where it would leave its cell, or come nearer its edge than `margin` px, somewhere in
    the last step. -> the strokes that are left, and the index of each"""
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


def edges(key):
    """How far each point of the plan lies inside its own cell, px."""
    b = np.zeros(key.shape, bool)
    b[:-1] |= key[:-1] != key[1:]
    b[1:] |= key[1:] != key[:-1]
    b[:, :-1] |= key[:, :-1] != key[:, 1:]
    b[:, 1:] |= key[:, 1:] != key[:, :-1]
    return ((ndimage.distance_transform_edt(~b) + 0.5) * S).astype(np.float32)


def plan(r):
    """The picture worked out at half size: the zone of every point (0 sky, 1 snows, 2 mist, 3 to 5 the bands
    of foothills), the cell its strokes keep to, its colour, the colours dragged over it, which way its strokes
    run, how much sun it has, and where the crests of the foothills run."""
    h, w = H // S, W // S
    gy, gx = (np.mgrid[0:h, 0:w].astype(np.float32) + 0.5) * S
    xs = gx[0]

    # the sky in bands laid across, each edge wavering, and long streaks of cloud
    band = np.zeros((h, w), np.int32)
    for j, (y0, _) in enumerate(SKY[1:], 1):
        edge = y0 * H + 14 * noise.line1d(w, 300, r)[None, :] + 0.012 * H * (xs[None, :] / W - 0.5)
        band[gy > edge] = j
    col = pal(*[c for _, c in SKY])[band]
    for x0, x1, yc, th, hexes in STREAKS:
        c = pal(*hexes)
        q = np.clip((gx - x0 * W) / ((x1 - x0) * W), 0, 1)
        half = th * H * np.clip(np.sin(np.pi * q), 0, 1) ** 0.6 * (1 + 0.4 * noise.line1d(w, 200, r))[None, :]
        off = gy - (yc * H + 12 * noise.line1d(w, 500, r)[None, :])
        inside = (np.abs(off) < half) & (q > 0) & (q < 1)
        col[inside] = np.where((off > 0.4 * half)[inside][:, None], c[2],
                               np.where((r.random(inside.sum()) < 0.5)[:, None], c[0], c[1]))
    hi = col.copy()
    zone = np.zeros((h, w), np.int32)
    cell = np.zeros((h, w), np.int32)
    ang = np.zeros((h, w), np.float32)
    sun = np.zeros((h, w), np.float32)

    # the snows, seen from the eye, and the slope of the snow at each point seen
    zs, Y, ident = survey(r)
    first = look(zs, Y)
    seen = first < len(zs)
    fi = np.minimum(first, len(zs) - 1)
    piece = ident[fi, np.broadcast_to(np.arange(w), fi.shape)]
    z = zs[fi]
    x, y = (gx - CX) * z / F, (CY - gy) * z / F
    hx, hz, e = np.zeros_like(z), np.zeros_like(z), 0.01
    px, pz = x[seen], z[seen]
    hx[seen] = (height(px + e, pz)[0] - height(px - e, pz)[0]) / (2 * e)
    hz[seen] = (height(px, pz + e)[0] - height(px, pz - e)[0]) / (2 * e)
    slope = np.hypot(hx, hz) + 1e-6
    s = (-hx * SUN[0] + SUN[1] - hz * SUN[2]) / np.sqrt(1 + slope ** 2)
    ux, uz = -hx / slope, -hz / slope
    fall = np.arctan2(slope * z + y * uz, ux * z - x * uz)        # the way the snow falls, on the canvas

    top = (MIST[0] + MIST[1] * noise.line1d(w, 160, r) + 10 * noise.line1d(w, 40, r))[None, :]   # the mist
    snows = seen & (gy < top)
    t = np.clip((s - 0.05) / 0.85, 0, 1)
    alt = np.clip((y - 2.0) / 5.0, 0, 1)                 # the higher snows warmer
    # in the last of the sun a face is lit or it is not: rose to cream where it is, violet where it is not
    te = 0.2 + 0.3 * noise.smoothstep(0, 0.15, t) + 0.5 * t * (1 + 0.15 * alt)
    lit = noise.smoothstep(0.4, 0.75, t)                 # where the light goes on dry over a darker coat

    def bands(v):                                        # tempera changes colour in bands, never smoothly
        return np.floor(v * 9 + 0.5) / 9

    col[snows] = ramp(RAMP, bands(te - 0.18 * lit))[snows]
    hi[snows] = ramp(RAMP, bands(te) + 0.12)[snows]
    ang[snows] = fall[snows]
    sun[snows] = lit[snows]
    zone[snows] = 1
    cell[snows] = (1 + piece)[snows]
    veil = (snows * noise.smoothstep(top - 160, top - 10, gy)).astype(np.float32)

    mist = gy >= top
    streak = ndimage.zoom(noise.field((h, w // 40 + 2), 5, r), (1, 40), order=1)[:, :w]
    col[mist], hi[mist], ang[mist] = MISTC[np.digitize(streak, [-0.6, 0.6])][mist], MISTC[1], 0.0
    zone[mist], cell[mist] = 2, 1 << 20

    # the foothills, band over band, each one plain colour
    tops = [crest(mid, amp, lean, r) for mid, amp, lean, _ in HILLS]
    below = np.zeros((h, w), np.float32)                  # how far below the crest of its band, px
    along = np.zeros((h, w), np.float32)
    lite = np.zeros_like(col)
    for b, (_, _, _, c) in enumerate(HILLS):
        m = gy > tops[b][None, :]
        d = gy - tops[b][None, :]
        col[m], hi[m], lite[m] = lin(c), HATCH[b], CRESTS[b]
        zone[m], cell[m] = 3 + b, (1 << 22) + b
        lean = np.arctan(np.gradient(ndimage.gaussian_filter1d(tops[b], 12 / S), S))[None, :]
        along[m] = np.broadcast_to(lean, m.shape)[m]
        ang[m] = (lean * np.exp(-d / 500))[m]
        below[m] = d[m]
    return dict(zone=zone, cell=cell, col=col, hi=hi, lite=lite, ang=ang, along=along, sun=sun, veil=veil,
                below=below, tops=tops)


def paint(seed=1935):
    r = np.random.default_rng([seed, 0])         # the design
    p = np.random.default_rng([seed, 1])         # the hand
    pl = plan(r)
    zone, cell, col, hi, ang = pl["zone"], pl["cell"], pl["col"], pl["hi"], pl["ang"]
    dz, dc = edges(zone), edges(cell)

    sheet = canvas.duck((H, W), seed, tint="#d8d1c0", thread=3.3)
    rgb = sheet.color.copy()
    height = (0.6 * sheet.tooth + 0.15 * noise.fbm((H, W), 10, p, octaves=2)).astype(np.float32)
    # what a nearly dry brush catches on: the threads, and the lumps of the coats beneath
    skin = (0.35 * sheet.tooth + noise.fbm((H, W), 11, p, octaves=3)).astype(np.float32)
    patches = noise.smoothstep(-0.2, 1.2, noise.field((H // S, W // S), 120, p))

    # the first coat, laid flat over each part and stopping at its edges, which waver a little
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    iy = np.clip(((yy + 1.6 * noise.field((H, W), 3, p)) / S).astype(int), 0, H // S - 1)
    ix = np.clip(((xx + 1.6 * noise.field((H, W), 3, p)) / S).astype(int), 0, W // S - 1)
    del yy, xx
    cov = tempera.coat(np.ones((H, W), np.float32), p, angle=0.0, size=(4.0, 14.0), thin=0.06, mottle=0.04)
    rgb += (col[iy, ix] - rgb) * (1 - 0.25 * (1 - cov))[..., None]
    del iy, ix, cov

    def strokes(where, width, length, cover, key, dist, margin=0.5, steps=10, tilt=0.06, colours=None, jitter=0.0,
                dry=0.0, hide=1.0, vary=0.04, spacing=10, field=ang, surface=None, how=TEMPERA):
        """One pass of strokes over the points of the plan where `where` (0..1) asks for them, each running
        along `field` and keeping to its cell of `key`."""
        P = dabs.scatter((H + 120, W + 120), spacing, p) - 60
        iy, ix = at(P)
        L = length * np.exp(p.normal(0, 0.25, len(P)))
        wd = width * np.exp(p.normal(0, 0.12, len(P)))
        keep = (p.random(len(P)) < where[iy, ix] * 0.866 * spacing ** 2 * cover / (2 * wd * L)) & \
               (dist[iy, ix] >= margin * wd)
        P, L, wd = P[keep], L[keep], wd[keep]
        c = key[at(P)]
        paths = either(impasto.follow(field, P / S, L / S, steps, 0.0, p.normal(0, tilt, len(P))) * S, p)
        paths, i = cut(paths, key, c, dist, margin * wd * p.uniform(0.4, 1.0, len(wd)), p)
        P, wd, c = P[i], wd[i], c[i]
        src = colours if colours is not None else col
        # the colour is taken from near where the stroke starts, but never from beyond the edge of its cell
        near = [P + p.normal(0, spread, P.shape) for spread in (jitter, 3 * jitter + 6)]
        base, kin = (src[at(np.where((key[at(q)] == c)[:, None], q, P))] for q in near)
        n = len(P)
        c0 = base * np.exp(p.normal(0, vary, (n, 1)) + p.normal(0, 0.5 * vary, (n, 3)))
        c1 = c0 * np.exp(p.normal(0, 0.4 * vary, (n, 3)))
        loads = np.stack([c0, c1, kin * np.exp(p.normal(0, vary, (n, 1)))], 1).astype(np.float32)
        share = np.stack([p.uniform(0.5, 0.75, n), p.uniform(0.2, 0.4, n), p.uniform(0.03, 0.12, n)], 1)
        surface = surface if surface is not None else height if dry < 0.5 else skin.copy()
        impasto.lay(rgb, surface, paths, wd, loads, p, share=share, dry=dry, hide=hide, **how)

    # the sky, in long strokes laid across, band by band; where two bands meet their strokes cross into each other
    strokes(zone == 0, 18, 420, 2.0, zone, dz, margin=0.45, steps=12, tilt=0.03, jitter=10, vary=0.015)
    # the snows, plane by plane, each in strokes that run down its fall, over a coat darker where the sun is
    strokes(zone == 1, 7, 90, 1.6, cell, dc, margin=0.4, steps=9, tilt=0.08, vary=0.03)
    # the light dragged over the sunlit planes with a brush nearly dry, so it breaks on the weave and the lumps
    strokes((zone == 1) * pl["sun"] * (0.3 + 0.7 * patches), 7, 100, 1.8, cell, dc, margin=0.25, steps=9, tilt=0.08,
            colours=hi, dry=0.84)
    strokes((zone == 1) * pl["sun"] ** 2 * patches, 6, 80, 0.6, cell, dc, margin=0.3, steps=8, tilt=0.08, colours=hi,
            dry=0.6)
    # and a little paler blue dragged over the shaded planes, the light of the sky in them
    strokes((zone == 1) * (1 - pl["sun"]) * patches, 6, 70, 0.6, cell, dc, margin=0.25, steps=8, tilt=0.1, colours=hi,
            dry=0.86)
    # the mist across the foot of the snows, and veils of it dragged up over them
    strokes(zone == 2, 10, 300, 1.6, zone, dz, margin=0.4, steps=12, tilt=0.03, jitter=8, vary=0.025)
    strokes(pl["veil"], 9, 220, 1.4, zone, dz, margin=0.2, steps=12, tilt=0.03, field=np.zeros_like(ang), dry=0.3,
            hide=0.8, colours=np.broadcast_to(MISTC[1], col.shape))
    # the foothills, the far band first, each brushed along its slope, a colour dragged dry over its foot here and
    # there, and the last light along its crest
    for b in range(len(HILLS)):
        strokes(zone == 3 + b, 10, 170, 1.6, cell, dc, margin=0.4, steps=10, tilt=0.06, vary=0.05)
        thick = noise.smoothstep(40, 400, pl["below"]) * patches
        strokes((zone == 3 + b) * thick, 10, 150, 0.9 if b else 0.3, cell, dc, margin=0.3, steps=10, tilt=0.08,
                colours=hi, dry=0.7, surface=height)
        strokes((zone == 3 + b) * noise.smoothstep(40, 5, pl["below"]), 8, 160, 1.2, cell, dc, margin=0.2,
                steps=10, tilt=0.05, colours=pl["lite"], dry=0.45, field=pl["along"], surface=height)
    # a stupa on a knoll of the middle band, whitewashed, the last of the sun on its western side
    top = pl["tops"][1]
    j = 1100 + int(np.argmin(top[1100:1400]))
    stupa(rgb, j * S + 1.0, top[j] + 5.0, p)

    return dabs.shine(rgb, height, light=(-0.5, -0.6, 0.62), relief=0.35, gloss=0.0, reach=(0.9, 1.1))


def stupa(rgb, x, y, r, k=4, size=1.25):
    """A small chorten standing at (x, y): two square steps, the dome, the box over it and the spire, in
    whitewash lit on the left and violet shade on the right, its edges a little unsteady."""
    n = 80
    x0, y0 = int(x) - n // 2, int(y) - n + 8
    v, u = (np.mgrid[0:n * k, 0:n * k] + 0.5) / k
    u = (u + x0 - x + 0.35 * noise.field((n * k, n * k), 6, r)) / size   # a hand, not a rule
    v = (v + y0 - y + 0.35 * noise.field((n * k, n * k), 6, r)) / size
    parts = [  # mask, lit colour, shaded colour
        ((np.abs(u) < 13) & (v > -6) & (v < 0), "#e9cfc4", "#6e68b0"),
        ((np.abs(u) < 10) & (v > -11) & (v <= -6), "#f0d6c9", "#746eb6"),
        (((u / 9.2) ** 2 + ((v + 17.5) / 9.0) ** 2 < 1) & (v <= -10.5), "#f6dfcf", "#7c76bc"),
        ((np.abs(u) < 3.6) & (v > -29.5) & (v <= -25.5), "#f0d4c4", "#6c66ae"),
        ((np.abs(u) < 3.0 * (v + 46) / 16.5) & (v > -46) & (v <= -29.5), "#e4b25e", "#8a6f86"),
        ((u ** 2 + (v + 47.5) ** 2) < 2.4, "#efc46c", "#9a7f7a"),
    ]
    patch = rgb[y0:y0 + n, x0:x0 + n]
    shade = u > 0.5 + 0.08 * (v + 20)                                 # the light comes from the left, in front
    for m, lit, dark in parts:
        a = m.reshape(n, k, n, k).mean((1, 3))
        c = np.where(shade.reshape(n, k, n, k).mean((1, 3))[..., None] > 0.5, lin(dark), lin(lit))
        c = c * (1 + 0.04 * noise.field((n, n), 2, r))[..., None]
        patch += (c - patch) * a[..., None]
