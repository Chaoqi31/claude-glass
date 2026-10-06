"""Sun and Moon. Oil on canvas.

At Louveciennes, outside Paris, in 1912 and 1913, Robert Delaunay painted the Circular Forms: rings and
segments of colour with no object in them, their titles naming only the sun and the moon. He built them on
the contrast of colours laid side by side, as Chevreul had set it out in 1839: orange beside blue, red beside
green, yellow beside violet, each making the other more intense, so that where they meet the edge seems to
tremble. Sonia Delaunay took it further the next year in Electric Prisms.

Here a warm sun fills the left of the canvas and a smaller, cooler moon rises at the upper right, in
crescents. Two more discs crowd in, one cut by the top of the canvas and one by its foot, and a great arc
sweeps across everything, turning to the opposite of each colour it crosses. Each ring was laid in first with
a big brush, round and round, then filled with strokes that go round it in one segment and across it in the
next, some thinned to a film the weave shows through, some loaded. Each edge was cut in by eye with a smaller
brush, now a hair over its neighbour, now just short of it, so no circle is quite true and nothing is
outlined; at some meetings of opposite colours each is flicked across into the other. The paint is matt and
stands in low relief.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Sun and Moon"
DATE = "2026"
MEDIUM = "Oil on canvas, thinned in places and loaded in others"
AFTER = ("Robert Delaunay, Circular Forms (Formes circulaires), Louveciennes, 1912–13; "
         "Sonia Delaunay, Electric Prisms (Prismes électriques), 1914")
ROOM = "Colour Itself"
YEAR = 1913
PLACE = "Louveciennes"
REGION = "Europe"
NOTE = ("Discs of unmixed colour, ringed and quartered so that orange meets blue, red meets green and yellow "
        "meets violet. A warm sun and a smaller, cool moon hold the two ends, and one great arc crosses everything.")

H, W = 2400, 3200
S = 2                      # the plan of the picture is worked out at half size
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
PAINTS = {
    "orange": pal("#e2561e", "#ea6a21", "#f07f27", "#f39434", "#f5a746"),      # cadmium orange
    "red": pal("#c8262b", "#d6352b", "#e0472e", "#e75d3c", "#ed7656"),         # vermilion
    "madder": pal("#b3204a", "#c53a5d", "#d55a77", "#e17e95"),
    "yellow": pal("#eaa612", "#f0b818", "#f4c922", "#f6d835", "#f7e46a"),      # cadmium yellow
    "lemon": pal("#efd51c", "#f3df36", "#f6e866", "#f7ee98"),
    "green": pal("#0b6e4d", "#127f57", "#1c9163", "#2ca371", "#4cb487"),       # emerald
    "viridian": pal("#0d5f55", "#147062", "#1d8573", "#389b88"),
    "blue": pal("#1e4ca5", "#285db8", "#3870c5", "#5087d0", "#719fda"),        # cobalt
    "ultra": pal("#1b2884", "#223397", "#2d41a9", "#3c53b8"),                  # ultramarine
    "deep": pal("#111a5a", "#16216c", "#1d2a80", "#253591"),                   # ultramarine, deep
    "cerulean": pal("#2b82c1", "#4195cd", "#5fa9d7", "#84bee1"),
    "violet": pal("#45348f", "#5442a0", "#6654ae", "#7b6bbb", "#9585c9"),
    "pink": pal("#e5a2ad", "#ebb5bd", "#f0c7cc", "#f4d7d9"),                   # madder let down with white
    "white": pal("#ede3d0", "#f2eadb", "#f5efe4", "#f8f3ea"),                  # warm whites
    "cream": pal("#efdcb5", "#f3e5c8", "#f6ecd8"),
    "skyblue": pal("#a3c0e2", "#b7cfea", "#cbdcf1"),
}
KIN = {"orange": ("red", "yellow"), "red": ("orange", "madder"), "madder": ("red", "pink"), "yellow": ("orange", "lemon"),
       "lemon": ("yellow", "white"), "green": ("viridian", "lemon"), "viridian": ("green", "blue"),
       "blue": ("ultra", "cerulean"), "ultra": ("blue", "violet"), "deep": ("ultra", "violet"), "cerulean": ("blue", "white"),
       "violet": ("ultra", "madder"), "pink": ("madder", "white"), "white": ("cream", "pink"),
       "cream": ("white", "yellow"), "skyblue": ("cerulean", "white")}
OPPOSITE = {"orange": "blue", "blue": "orange", "red": "green", "green": "red", "yellow": "violet",
            "violet": "yellow", "lemon": "violet", "cerulean": "orange", "ultra": "yellow", "deep": "yellow", "viridian": "red",
            "madder": "green", "pink": "red", "white": "red", "cream": "red", "skyblue": "red"}

# The discs: centre x, y, z, rings. Each ring: outer radius, how far its centre sits from the disc's, its own z
# (None for the disc's), and its sectors as (start in degrees, paint): None leaves that part open, "X" lays the
# opposite of whatever lies beneath.
DISCS = [
    # the sun's light spread over the canvas, in pale bands
    (1150, 1320, 0, [(1280, 0, 0, None, [(-35, "white"), (55, "cream"), (145, "white"), (235, "white")]),
                     (1720, 0, 0, None, [(-35, "cream"), (55, "white"), (145, "skyblue"), (235, "white")]),
                     (4000, 0, 0, None, [(-35, "white"), (55, "cream"), (145, "white"), (235, "pink")])]),
    # the moon's
    (2470, 760, 1, [(780, 0, 0, None, [(-150, "white"), (30, "skyblue")]),
                    (1030, 0, 0, None, [(-150, "cream"), (30, "white")])]),
    # the sun: a bright core, then rings broad and narrow, each divided its own way
    (1150, 1320, 3, [(175, -26, 18, None, [(-35, "lemon"), (145, "white")]),
                     (330, -16, 12, None, [(-35, "orange"), (55, "blue"), (145, "orange"), (235, "blue")]),
                     (410, -9, 7, None, [(10, "violet"), (190, "yellow")]),
                     (620, -4, 3, None, [(10, "red"), (100, "green"), (190, "pink"), (280, "green")]),
                     (700, 0, 0, None, [(-80, "pink"), (60, "cerulean"), (130, "cream"), (200, "lemon")]),
                     (940, 0, 0, None, [(-50, "yellow"), (70, "deep"), (130, "orange"), (250, "violet")])]),
    # the moon, in crescents
    (2470, 760, 4, [(240, 70, -55, None, [(-60, "white"), (120, "pink")]),
                    (340, 42, -34, None, [(-60, "green"), (150, "red"), (240, "green")]),
                    (450, 18, -15, None, [(-150, "blue"), (60, "orange"), (150, "blue")]),
                    (560, 0, 0, None, [(-150, "deep"), (-20, "lemon"), (60, "violet")])]),
    # a small disc high up, cut by the edge
    (1950, 205, 5, [(90, 0, 0, None, [(0, "red"), (180, "green")]),
                    (170, 0, 0, None, [(90, "yellow"), (270, "violet")]),
                    (245, 0, 0, None, [(0, "blue"), (180, "orange")])]),
    # one low on the right, cool and pale, cut by the foot of the canvas
    (2420, 2080, 2, [(170, 0, 0, None, [(30, "violet"), (210, "lemon")]),
                     (400, 0, 0, None, [(-60, "lemon"), (30, "skyblue"), (120, "pink"), (210, "white")]),
                     (640, 0, 0, None, [(30, "cerulean"), (150, "violet"), (250, "cerulean")])]),
    # a great arc swept across everything, taking the opposite colour of all it crosses
    (-600, 3100, 6, [(2860, 0, 0, None, [(0, None)]), (2990, 0, 0, None, [(0, "X")])]),
]


def table(r, n=2048, amp=1.0, ks=range(2, 13), fine=0.5):
    """A smooth closed wander round a circle (or along a line), sampled at n points: a few slow
    swells and a fine tremor, px."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros(n, np.float32)
    for k in ks:
        out += amp / k ** 0.8 * r.normal() * np.sin(k * t + r.uniform(0, 2 * np.pi))
    for k in range(13, 70, 2):
        out += fine / np.sqrt(k / 13) * r.normal() * np.sin(k * t + r.uniform(0, 2 * np.pi))
    return out


def plan(r):
    """The picture as cells of one colour: a key for every point of the plan (half size), and for each
    key its paint, the centre its strokes turn about, and the disc, ring and sector it belongs to."""
    yy, xx = (np.mgrid[0:H // S, 0:W // S].astype(np.float32) + 0.5) * S
    key = np.zeros(xx.shape, np.int32)
    cells = [dict(paint="white", centre=(W / 2, H / 2), disc=-1, ring=0, sector=0)]
    items, maps = [], []
    for d, (cx, cy, z, rings) in enumerate(DISCS):
        ring = np.full(xx.shape, -1, np.int16)
        sector = np.zeros(xx.shape, np.int16)
        for i, (R, dx, dy, zr, sectors) in enumerate(rings):
            ox, oy = cx + dx, cy + dy
            rho, th = np.hypot(xx - ox, yy - oy), np.arctan2(yy - oy, xx - ox)
            wob = table(r, amp=0.006 * R + 2.5)            # drawn by eye: a little out of true
            ecc = 0.006 * R * np.cos(2 * (np.linspace(0, 2 * np.pi, 2048, endpoint=False) - r.uniform(0, np.pi)))
            idx = ((th / (2 * np.pi)) % 1 * 2048).astype(int) % 2048
            inside = (rho < R + wob[idx] + ecc[idx]) & (ring < 0)
            ring[inside] = i
            a = np.radians([s for s, _ in sectors])
            # each dividing line is drawn from the centre outward, bowing and wavering a little
            lines = []
            for _ in a:
                bow, tr = r.normal(0, 0.006 * R + 3), table(r, 1024, 2.0, range(1, 6), 0.4)
                ri = np.clip(rho / max(R, 1) * 1023, 0, 1023).astype(int)
                lines.append((bow * np.sin(np.pi * np.clip(rho / R, 0, 1)) + tr[ri]) / np.maximum(rho, 25))
            u = (th - a[0] - lines[0]) % (2 * np.pi)
            s = np.zeros(xx.shape, np.int16)
            for j in range(1, len(a)):
                s += (u >= (a[j] + lines[j] - a[0] - lines[0]) % (2 * np.pi))
            sector[inside] = s[inside]
            items.append((z if zr is None else zr, d, i, (ox, oy)))
        maps.append((ring, sector))
    for z, d, i, centre in sorted(items, key=lambda t: t[0]):
        ring, sector = maps[d]
        sectors = DISCS[d][3][i][4]
        for j, (_, paint) in enumerate(sectors):
            m = (ring == i) & (sector == j)
            if paint is None or not m.any():
                continue
            if paint == "X":
                for k in np.unique(key[m]):
                    mk = m & (key == k)
                    cells.append(dict(paint=OPPOSITE[cells[k]["paint"]], centre=centre, disc=d, ring=i, sector=j))
                    key[mk] = len(cells) - 1
            else:
                cells.append(dict(paint=paint, centre=centre, disc=d, ring=i, sector=j))
                key[m] = len(cells) - 1
    return key, cells


def at(P):
    """Where points (x, y) of the canvas fall on the plan."""
    P = np.asarray(P)
    return (np.clip((P[..., 1] / S).astype(int), 0, H // S - 1), np.clip((P[..., 0] / S).astype(int), 0, W // S - 1))


def either(paths, r, odds=0.5):
    """Some strokes go one way, some the other."""
    back = r.random(len(paths)) < odds
    paths[back] = paths[back, ::-1]
    return paths


def brushes(paints, tone, r, spread=0.1, odds=0.3):
    """The load of each brush, from the paint of its cell: the tone it wants, the tone next to it
    streaked in, and a little of a third, now and then a paint akin to it."""
    N = len(tone)
    cols, share = np.empty((N, 3, 3), np.float32), np.empty((N, 3))
    for p in np.unique(paints):
        sel = np.nonzero(paints == p)[0]
        fam, n, m = PAINTS[p], len(PAINTS[p]), len(sel)
        t = np.clip(tone[sel] + r.normal(0, spread, m), 0, 0.999)
        i = (t * n).astype(int)
        j = np.clip(i + np.where(r.random(m) < t * n - i, 1, -1), 0, n - 1)
        third = fam[np.clip(i + r.choice([-2, 2], m), 0, n - 1)]
        for q in KIN[p]:
            hit = r.random(m) < odds / 2
            third[hit] = PAINTS[q][(t[hit] * len(PAINTS[q])).astype(int)]
        cols[sel] = np.stack([fam[i], fam[j], third], 1)
        share[sel] = np.stack([r.uniform(0.45, 0.7, m), r.uniform(0.2, 0.4, m), r.uniform(0.05, 0.25, m)], 1)
    return cols, share


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


def bands():
    """Every ring of every disc as the band it fills: (disc, ring, centre x, y, inner and outer radius)."""
    out = []
    for d, (cx, cy, z, rings) in enumerate(DISCS):
        for i, (R, dx, dy, zr, sectors) in enumerate(rings):
            inner = rings[i - 1][0] - np.hypot(dx - rings[i - 1][1], dy - rings[i - 1][2]) if i else 0.0
            out.append((d, i, cx + dx, cy + dy, max(inner, 0.0), R))
    return out


def courses(key, owner, r, step=3.0):
    """The ground of each ring laid in courses: brush-wide strokes side by side round it (straight across
    a core, where round ones would turn too tight), each course broken into strokes a few brush-widths
    long that overlap end to end, and run a little past the end of their cell or stop short of it.
    `owner` (disc, ring) per cell. -> paths, half-widths, cells"""
    paths, ws, cs = [], [], []
    for d, i, ox, oy, r0, r1 in bands():
        mine = (owner[:, 0] == d) & (owner[:, 1] == i)
        if not mine.any():
            continue
        band = r1 - r0
        wl = float(np.clip(0.32 * band, 12, 40))
        lines = []
        if r0 < 2.5 * wl:
            a = r.uniform(0, np.pi)
            e, f = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            u = np.arange(-r1, r1, step)
            for v in np.arange(-r1 + r.uniform(0.3, 1.2) * wl, r1, 1.5 * wl):
                lines.append(np.array([ox, oy]) + u[:, None] * e + (v + r.normal(0, 0.1 * wl)) * f)
        else:
            nc = int(np.ceil(band / (1.5 * wl)))
            for k in range(nc):
                rho = r0 + (k + 0.5 + r.uniform(-0.15, 0.15)) * band / nc
                th = r.uniform(0, 2 * np.pi) + np.arange(0, 2 * np.pi, step / rho)
                lines.append(np.stack([ox + rho * np.cos(th), oy + rho * np.sin(th)], 1))
        for pts in lines:
            seen = (pts[:, 0] > -60) & (pts[:, 0] < W + 60) & (pts[:, 1] > -60) & (pts[:, 1] < H + 60)
            kk = np.where(seen, key[at(pts)], -1)
            kk = np.where((kk >= 0) & mine[np.maximum(kk, 0)], kk, -1)
            cuts = np.nonzero(np.diff(kk))[0] + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(kk)]):
                if kk[a] < 0 or (b - a) * step < 0.6 * wl:
                    continue
                cell = kk[a]
                a = int(np.clip(a - r.uniform(-0.4, 0.5) * wl / step, 0, len(pts) - 1))   # past the end, or short
                b = int(np.clip(b + r.uniform(-0.4, 0.5) * wl / step, a + 2, len(pts)))
                L = wl * (r.uniform(7, 12) if d <= 1 else r.uniform(4, 8)) / step
                s0 = a
                while s0 < b - 1:
                    s1 = int(min(b, s0 + L * r.uniform(0.7, 1.3)))
                    if s1 - s0 >= 2:
                        paths.append(pts[np.linspace(s0, s1 - 1, 8).astype(int)])
                        ws.append(wl * r.uniform(0.85, 1.15))
                        cs.append(cell)
                    if s1 >= b:
                        break
                    s0 = int(s1 - 0.3 * wl / step * r.uniform(0.5, 1.5))
    order = r.permutation(len(paths))
    return np.array(paths, np.float32)[order], np.array(ws)[order], np.array(cs)[order]


# how the paint goes on: thinned to a film the weave shows through; middling; loaded
THIN = dict(thick=0.03, grooves=0.15, lips=0.03, land=0.2, lift=0.1, tails=0.4, pickup=0.3, merge=2.5, spent=0.4,
            ends=(0.5, 0.35), fray=1.3, hide=0.72)
MID = dict(thick=0.15, grooves=0.4, lips=0.08, land=0.45, lift=0.3, tails=0.5, pickup=0.35, merge=2.0, spent=0.45,
           ends=(0.45, 0.3), fray=1.2)
LOADED = dict(thick=0.25, grooves=0.55, lips=0.12, land=0.6, lift=0.4, tails=0.6, pickup=0.35, merge=2.5, spent=0.4,
              ends=(0.45, 0.3), fray=1.1)
BODIES = (THIN, MID, LOADED)


def paint(seed=1913):
    key, cells = plan(np.random.default_rng([seed, 0]))
    g, r = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])     # the cells' choices; the strokes
    n, (h, w) = len(cells), key.shape
    gy, gx = (np.mgrid[0:h, 0:w].astype(np.float32) + 0.5) * S
    paints = np.array([c["paint"] for c in cells])

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
    breadth = 2 * ndimage.maximum(dist, key, np.arange(n))     # the widest stroke each cell could take

    # how each cell is painted: which way its strokes run, how big they are, how thick the paint
    ring = np.array([c["ring"] for c in cells])
    sector = np.array([c["sector"] for c in cells])
    flip = (ring + sector) % 2
    mode = np.where(flip == 0, np.where(g.random(n) < 0.7, 0, 2), np.where(g.random(n) < 0.6, 1, 3))
    mode = np.where((mode == 3) & (breadth < 260), 2, mode)         # a narrow ring has no room to lie across
    disc = np.array([c["disc"] for c in cells])
    mode = np.where(disc <= 1, np.where(g.random(n) < 0.5, 3, 0), mode)    # the open bands: across, or round
    pale = np.isin(paints, ["white", "cream", "pink", "skyblue"])
    body = np.where(pale, g.choice(3, n, p=[0.15, 0.4, 0.45]), g.choice(3, n, p=[0.3, 0.45, 0.25]))
    body = np.where(disc <= 1, g.choice(2, n, p=[0.6, 0.4]), body)       # the open bands go on thinner
    hand = np.array([1.2, 1.2, 1.15, 0.9, 0.85, 1.1, 1.0])[disc]          # each disc its own size of brush
    wide = np.clip(0.17 * breadth, 9, 30) * hand * g.uniform(0.85, 1.2, n)
    long_ = wide * np.select([mode == 0, mode == 1, mode == 2], [g.uniform(7, 14, n), g.uniform(3, 8, n),
                                                                 g.uniform(5, 10, n)], g.uniform(8, 16, n))
    long_ = np.where(mode == 1, np.minimum(long_, np.maximum(breadth * 1.1, 3 * wide)),
                     np.where(mode == 0, long_, np.maximum(np.minimum(long_, 1.3 * breadth), wide)))
    tone = np.where(pale, g.uniform(0.35, 0.75, n), g.uniform(0.25, 0.6, n))
    slant = g.choice([-1, 1], n) * g.uniform(0.5, 0.9, n)
    level = g.uniform(0, np.pi, n)
    cx = np.array([c["centre"][0] for c in cells])
    cy = np.array([c["centre"][1] for c in cells])
    radial = np.arctan2(gy - cy[key], gx - cx[key])
    m = mode[key]
    field = np.select([m == 0, m == 1, m == 2], [radial + np.pi / 2, radial, radial + np.pi / 2 + slant[key]],
                      level[key]) + 0.12 * noise.field(key.shape, 150 / S, r)
    field = field.astype(np.float32)

    ground = canvas.duck((H, W), seed, tint="#e2d8c4", thread=3.0)
    rgb, height = ground.color.copy(), ground.tooth * 0.25
    wet = np.zeros((H, W), np.float32)

    def seeds(spacing, odds):
        """Where strokes start: a shaken honeycomb over the canvas and a little beyond, thinned cell
        by cell by `odds` (one per cell)."""
        P = dabs.scatter((H + 120, W + 120), spacing, r) - 60
        c = key[at(P)]
        keep = r.random(len(P)) < odds[c]
        return P[keep], c[keep]

    def follow(angle, P, L, steps, tilt=0.0):
        return impasto.follow(angle, P / S, L / S, steps, 0.0, tilt) * S

    # the first sitting: each ring laid in with a big brush, round and round, over its edges a little, and
    # left to dry; where a cell is to stay thin the paint is a film the weave shows through
    drift = 0.22 * noise.field(key.shape, 500 / S, r)               # where a cell runs darker or lighter
    drift *= np.where(pale, 0.5, 1)[key]
    owner = np.array([(c["disc"], c["ring"]) for c in cells])
    paths, wl, c = courses(key, owner, r)
    P = paths[:, 0]
    cols, share = brushes(paints[c], np.clip(tone[c] + drift[at(P)] + 0.05, 0, 1), r)
    for thin in (False, True):
        sel = np.nonzero((body[c] == 0) == thin)[0]
        impasto.lay(rgb, height, paths[sel], wl[sel], cols[sel], r, share=share[sel], wet=wet, thick=0.03,
                    grooves=0.15, lips=0.03, land=0.2, lift=0.1, tails=0.4, pickup=0.25, spent=0.3, ends=(0.5, 0.35),
                    fray=1.5, hide=0.8 if thin else 0.95, dry=r.uniform(0, 0.08, len(sel)))
    wet[:] = 0

    # the second: every cell filled in strokes of its own direction and size, thin first, loaded last
    cover = np.array([0.9, 2.6, 3.0])[body] * np.where(disc <= 1, 0.25, 1)
    P, c = seeds(12, 0.866 * 12 ** 2 * cover / (2 * wide * long_))
    inner = dist[at(P)] >= 0.9 * wide[c]
    P, c = P[inner], c[inner]
    paths = follow(field, P, long_[c] * np.exp(r.normal(0, 0.25, len(c))), 12, r.normal(0, 0.08, len(c)))
    margin = wide[c] * np.where(mode[c] == 1, r.uniform(0.05, 0.4, len(c)), r.uniform(0.35, 0.7, len(c)))
    paths, i = cut(either(paths, r), key, c, dist, margin, r)
    c, P = c[i], P[i]
    for k, how in enumerate(BODIES):
        sel = np.nonzero(body[c] == k)[0]
        sel = sel[r.permutation(len(sel))]
        cols, share = brushes(paints[c[sel]], tone[c[sel]] + drift[at(P[sel])] + r.normal(0, 0.07, len(sel)), r)
        dry = r.uniform(0, 0.12, len(sel))
        impasto.lay(rgb, height, [paths[j] for j in sel], wide[c[sel]] * np.exp(r.normal(0, 0.12, len(sel))), cols, r,
                    share=share, wet=wet, dry=dry, **how)

    # then each edge cut in with a smaller brush drawn along it, now a hair over the next colour, now short of it
    we = np.clip(0.55 * wide, 4, 10)
    Le = r.uniform(120, 260, n)
    band = (dist >= 0.5 * we[key]) & (dist <= 1.5 * we[key])
    pick = band & (r.random(key.shape) < S * S / (we[key] * 0.45 * Le[key]))
    iy, ix = np.nonzero(pick)
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
    ang = np.arctan2(seg[..., 1], seg[..., 0])
    ok[:, 2:] &= np.abs(np.angle(np.exp(1j * np.diff(ang, axis=1)))) < 0.5
    stop = np.where(ok.all(1), ok.shape[1], np.argmin(ok, 1))
    keep = np.nonzero(stop >= 4)[0]
    edge = [paths[i, :stop[i]][::r.choice([-1, 1])] for i in keep]
    c = c[keep]
    cols, share = brushes(paints[c], tone[c] + drift[at(P[keep])] + r.normal(0, 0.05, len(c)), r)
    impasto.lay(rgb, height, edge, we[c] * np.exp(r.normal(0, 0.1, len(c))), cols, r, share=share, wet=wet,
                **MID)

    # at some meetings of opposite colours each is flicked across into the other, so the edge vibrates
    pure = ("orange", "blue", "red", "green", "yellow", "violet", "cerulean", "lemon", "ultra")
    hot = np.array([[OPPOSITE[a] == b and a in pure and b in pure for b in paints] for a in paints])
    hot &= r.random((n, n)) < 0.45
    hot |= hot.T
    other = key.copy()
    for ax, sh in ((1, 1), (1, -1), (0, 1), (0, -1)):
        nb = np.roll(key, sh, ax)
        other = np.where(other == key, nb, other)
    iy, ix = np.nonzero(b & hot[key, other] & (r.random(key.shape) < S / 13))
    a, o = key[iy, ix], other[iy, ix]
    P = np.stack([(ix + 0.5) * S, (iy + 0.5) * S], 1)
    t = tangent[iy, ix] - np.pi / 2
    e = np.stack([np.cos(t), np.sin(t)], 1)
    e *= np.where(key[at(P + 6 * e)] == o, 1, -1)[:, None]          # toward the other colour
    t = np.arctan2(e[:, 1], e[:, 0]) + r.normal(0, 0.35, len(a))
    e = np.stack([np.cos(t), np.sin(t)], 1)
    back, over = r.uniform(12, 30, len(a)), r.uniform(4, 15, len(a))
    paths = np.stack([P - e * back[:, None] + e * (back + over)[:, None] * f for f in np.linspace(0, 1, 6)], 1)
    order = r.permutation(len(a))
    cols, share = brushes(paints[a], tone[a] + drift[iy, ix] + r.normal(0, 0.05, len(a)), r)
    impasto.lay(rgb, height, paths[order], r.uniform(3.5, 6.5, len(a)), cols[order], r, share=share[order], wet=wet,
                **dict(MID, tails=0.8, spent=0.6, pickup=0.15))
    wet[:] = 0

    # the third, over dry paint: here and there a brush dragged nearly dry across a cell, against the run of
    # its strokes, in a paler tone or a paint akin to it, catching only the ridges and the weave
    scumbled = (r.random(n) < np.where(body == 0, 0.4, 0.15)) & (disc > 1)
    P, c = seeds(16, scumbled * 0.866 * 16 ** 2 * 0.7 / (2 * 1.2 * wide * long_ + 1))
    inner = dist[at(P)] >= 0.8 * wide[c]
    P, c = P[inner], c[inner]
    paths = follow(field, P, long_[c] * r.uniform(0.8, 1.3, len(c)), 10,
                   r.choice([-1, 1], len(c)) * r.uniform(0.4, 0.9, len(c)))
    paths, i = cut(either(paths, r), key, c, dist, 0.6 * wide[c], r)
    c, P = c[i], P[i]
    kin = np.array([KIN[p][1] for p in paints[c]])
    cols, share = brushes(np.where(r.random(len(c)) < 0.5, paints[c], kin), tone[c] + 0.25, r)
    impasto.lay(rgb, height, paths, 1.2 * wide[c], cols, r, share=share, wet=wet, dry=r.uniform(0.5, 0.75, len(c)),
                thick=0.06, grooves=0.2, lips=0.03, land=0.2, lift=0.1, tails=0.6, pickup=0, spent=0.5,
                ends=(0.5, 0.35), fray=1.6)

    height = ndimage.gaussian_filter(height, 0.6)      # the lamp sees the surface a hair softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.74, 1.2))
    return img + 0.04 * impasto.glints(height, LIGHT)[..., None]
