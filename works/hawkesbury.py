"""Noon on the Hawkesbury. Oil on canvas, painted out of doors.

In the summer of 1896 Arthur Streeton went up the Hawkesbury from Sydney with his paints and
stood on the ridges over the river in the heat. He painted standing up, fast, while the light
held: a pale ground, the masses rubbed in thin, and then the paint itself, long loaded strokes
across the sky and the water and short quick ones in the trees and the grass. The lights went
on thick, a stiff paste dragged over the weave, and the shadows stayed thin, so that in the
heat the ground shows through them. At noon nothing casts much shadow; the far ridges go blue
and then violet and then into the glare, and the river takes the colour of the sky.

Here the river comes round below the painter in a broad curve, sweeps back behind a flat of
bleached paddocks and goes off between the sandstone ridges into the haze. Gums stand on the
slope in front, and the rock he stood on is at the lower left.
"""

import numpy as np
from scipy import ndimage, spatial

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Noon on the Hawkesbury"
DATE = "2026"
MEDIUM = "Oil on canvas, painted out of doors; long loaded strokes in the sky and the water, short ones in the trees"
AFTER = ("Arthur Streeton and the Heidelberg School: The Purple Noon's Transparent Might, 1896; Fire's On, 1891; "
         "Golden Summer, Eaglemont, 1889")
ROOM = "Open Air"
YEAR = 1896
PLACE = "Hawkesbury River, New South Wales"
REGION = "Oceania"
NOTE = ("The river from a sandstone ridge at noon, the paddocks on the flat bleached to straw and the far ridges "
        "gone into the haze. The lights are thick and the shadows thin enough for the ground to show.")

H, W = 1700, 3200
F = 2600.0                 # focal length, px
EYE = 182.0                # the painter's eye above the river, m
SKYLINE = 460              # the true horizon, px from the top
HAZE = 6500.0              # m of air that veil the land by about two thirds
SUN = np.array([-0.28, 0.93, -0.24])       # high and a little behind the painter's left shoulder
SUN /= np.linalg.norm(SUN)
LAMP = (-0.6, -0.5, 0.62)  # the gallery light on the paint
LUMA = dabs.LUMA

# the river seen from above, upstream from the painter's feet: across, away (m), half-width,
# and the width of the flats on its left and right banks (looking upstream)
COURSE = np.array([
    (900, 520, 220, 0, 60), (297, 470, 215, 0, 120), (164, 520, 210, 0, 260), (0, 570, 205, 0, 360),
    (-185, 660, 195, 0, 420), (-387, 840, 170, 0, 430), (-550, 1090, 150, 0, 380), (-640, 1420, 135, 0, 330),
    (-520, 1900, 125, 40, 260), (-220, 2700, 115, 120, 120), (80, 3800, 105, 200, 40), (-250, 5500, 95, 150, 60),
    (200, 8000, 88, 60, 140), (-350, 11000, 80, 60, 60), (100, 15000, 70, 0, 0)], np.float64)
SIDE = 420.0               # m from the edge of the flats to the top of the cliffs


def hexes(*cs):
    return np.stack([lin(c) for c in cs]).astype(np.float32)


# the palette, mixed for each passage, every row from dark to light
AIR = lin("#c6c9d6")                                                                   # the heat haze the far land goes into
SKY = hexes("#b4c0d6", "#bec7d9", "#c8cedc", "#d1d4dd", "#d9d8db", "#dfdad5")
SKY_TOUCH = hexes("#cdc5d6", "#e0d3c4", "#b7c5dd", "#d6cfcf")                         # lilac, warm, cobalt, rose
FAR = hexes("#7586ab", "#8090b1", "#8b99b6", "#97a3bc", "#a3acc2")                    # cobalt hills in the haze
FOREST = hexes("#38412f", "#434d39", "#505a43", "#5f684c", "#6f7554", "#83845f")        # olive bush
BLUEGREEN = hexes("#3f514f", "#4b5e59", "#587063")
SHADE = hexes("#384463", "#45506d", "#545e78")                                        # the blue in the gums' shade
ROCK = hexes("#7c675e", "#937663", "#aa8767", "#bd976f", "#cca77e", "#d9b88f")         # sandstone
PADDOCK = hexes("#9d7838", "#b08741", "#c0964b", "#cca559", "#d5b46b", "#ddc380")       # gold
STRAW = hexes("#c4ab73", "#d1ba84", "#dcc895", "#e4d3a6")                             # bleached
GREEN_STRAW = hexes("#8e8b52", "#9e9a5e", "#aea96b")
SAND = hexes("#b39378", "#c2a386", "#cfb295", "#d9c0a5", "#e2cdb4")
SHALLOW = hexes("#6d7aa6", "#8a8ea9", "#a49ea6", "#b8ac9f")                            # water over sand
WATER = hexes("#2f3f7e", "#364585", "#3e4c8c", "#465492", "#505c98", "#5b669e", "#6871a4", "#7880ab", "#8a8fb3",
              "#9da1bc")                                                               # ultramarine to violet
REFLECT = hexes("#273648", "#2e3d4f", "#364556", "#3f4e5d", "#4a5864")                  # the bush in the water
GLARE = hexes("#cfd2de", "#e2e1e2", "#efebe2")
TREE = hexes("#2f3527", "#3a4030", "#474c38", "#565b42", "#686b4c")
SHADOW = hexes("#3a3f5e", "#464a69", "#555874")                                      # what the trees cast at noon
GUM = hexes("#2f3d3b", "#3a4841", "#4b5442", "#5e664b", "#727755", "#87885f", "#9e9a68")  # deep shade to sunlit leaf
TWIG = hexes("#4f463f", "#675a4e", "#7f6f60")
BARK = hexes("#5f5f75", "#7f7c88", "#a39c9d", "#c3b8ad", "#dcd1c2", "#ebe3d4")
BARK_WARM = hexes("#c8987a", "#d9b08f")                                                # new bark, salmon
LEDGE = hexes("#5c4d46", "#7a6253", "#987a5e", "#b38f6a", "#c9a678", "#d9ba8c")
GRASS = hexes("#9a8150", "#b0955c", "#c2a86c", "#d1b97d", "#ddc78f")


class Land:
    """The country round the river: sandstone plateau cut by the river's gorge, the gorge sides
    rising in a long slope to a band of cliff under the rim, flats of river silt on the inside of
    the bends, and far off a higher range. Heights in metres above the river."""

    def __init__(self, r):
        C = brush.path(COURSE, 4.0)
        self.C, self.tree = C, spatial.cKDTree(C[:, :2])
        T = np.gradient(C[:, :2], axis=0)
        self.T = T / np.linalg.norm(T, axis=1, keepdims=True)
        self.box = (-14000.0, 0.0, 25.0)            # the noise maps start here and step 25 m
        self.big, self.mid, self.small = (noise.field((1300, 1120), s, r) for s in (44, 15, 3.5))
        self.range = noise.line1d(2000, 400, r)      # the far range's crest, every 15 m across

    def look(self, a, X, Z):
        x0, z0, step = self.box
        return ndimage.map_coordinates(a, [(Z - z0) / step, (X - x0) / step], order=3, mode="nearest")

    def __call__(self, X, Z):
        """-> heights, and what the ground is at each point: its distance in from the water's
        edge, how far up the gorge side it lies (0 on the flats, 1 on the plateau), and how far
        in from the edge of the flats."""
        shape = X.shape
        X, Z = X.ravel(), Z.ravel()
        d, i = self.tree.query(np.stack([X, Z], 1))
        C, T = self.C[i], self.T[i]
        left = T[:, 0] * (Z - C[:, 1]) - T[:, 1] * (X - C[:, 0]) > 0
        bank = d - C[:, 2]
        u = bank - np.where(left, C[:, 3], C[:, 4])
        s = np.clip(u / SIDE, 0, None)
        rise = 0.75 * noise.smoothstep(0, 0.9, s) + 0.25 * noise.smoothstep(0.84, 0.97, s)
        crest = self.range[np.clip(((X + 15000) / 15).astype(int), 0, 1999)]
        far = 960 * noise.smoothstep(16500, 21000, Z) * (0.82 + 0.2 * np.clip(crest, -1.5, 1.5))
        top = 172 + 16 * self.look(self.big, X, Z) + far
        h = 1.5 + 2.5 * noise.smoothstep(0, 40, bank) + (top - 4) * rise \
            + 120 * rise * (1 - rise) * self.look(self.mid, X, Z) + 3 * self.look(self.small, X, Z) * noise.smoothstep(0, 60, bank)
        h = np.where(bank < 0, 0.0, np.maximum(h, 0.4))
        return h.reshape(shape), bank.reshape(shape), rise.reshape(shape), u.reshape(shape)


def first(y, rows):
    """For each row of the canvas, the nearest sample along each line of sight that stands at or
    above it, which is the one the eye meets there (len(y) where nothing does: the sky)."""
    M = -np.minimum.accumulate(y, 0)
    return np.stack([np.searchsorted(M[:, c], -rows) for c in range(y.shape[1])], 1)


def survey(land, k=2):
    """What the painter sees from the ridge, worked out at 1/k of the canvas: at every pixel, the
    distance to the ground, its height and kind and slope to the sun, and on the water, the row
    of the canvas that shows what the water mirrors there (`glass` 1 where that is land)."""
    h, w, f, top = H // k, W // k, F / k, SKYLINE / k
    Z = np.geomspace(140, 42000, 2000)[:, None]
    X = (np.arange(w)[None, :] + 0.5 - w / 2) * Z / f
    Zg = np.broadcast_to(Z, X.shape)
    hgt, bank, rise, u = land(X, Zg)
    rows = np.arange(h)[:, None] + 0.5
    y = top + f * (EYE - hgt) / Z
    see = first(y, rows[:, 0])
    sky = see == len(Z)
    cc = np.arange(w)[None, :]
    k1 = np.clip(see, 1, len(Z) - 1)
    y0, y1 = y[k1 - 1, cc], y[k1, cc]
    t = np.clip((y0 - rows) / (y0 - y1 + 1e-9), 0, 1)        # where the line of sight crosses the ground
    out = {n_: a[k1 - 1, cc] * (1 - t) + a[k1, cc] * t for n_, a in
           (("Z", Zg), ("X", X), ("h", hgt), ("bank", bank), ("rise", rise), ("u", u))}
    P = ndimage.gaussian_filter(np.stack([out["X"], out["h"], out["Z"]], -1), (1.5, 1.5, 0))
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    n *= np.sign(n[..., 1:2] + 1e-9)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    out["steep"] = np.where(sky, 0, 1 - n[..., 1])
    out["sun"] = np.where(sky, 1, np.clip(n @ SUN, 0, 1))
    # the water mirrors what the line of sight meets as it climbs again, as steeply as it fell:
    # the first ground that stands above it, or else the sky
    iy, ix = np.nonzero(~sky & (out["bank"] < 0))
    zw, kw = out["Z"][iy, ix], see[iy, ix]
    hit, live = np.full(len(iy), -1), np.arange(len(iy))
    for j0 in range(0, len(Z), 16):
        j = np.arange(j0, min(j0 + 16, len(Z)))
        reach = EYE * (Z[j, 0][None, :] / zw[live, None] - 1)
        up = (hgt[j[None, :], ix[live, None]] >= reach) & (j[None, :] > kw[live, None])
        got = up.any(1)
        hit[live[got]] = j[up[got].argmax(1)]
        live = live[~got & (reach[:, -1] < 1100)]
        if not len(live):
            break
    jh = np.maximum(hit, 0)
    mirror, glass = np.broadcast_to(2 * top - rows, (h, w)).copy(), np.zeros((h, w))
    mirror[iy, ix] = np.where(hit >= 0, top + f * (EYE - hgt[jh, ix]) / Z[jh, 0], 2 * top - iy - 0.5)
    glass[iy, ix] = hit >= 0
    out.update(mirror=mirror * k, glass=glass, sky=sky)
    out = {n_: ndimage.zoom(a.astype(np.float32), k, order=1, mode="nearest")[:H, :W] for n_, a in out.items()}
    out["sky"] = out["sky"] > 0.5
    return out


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def sketch(v, land, r):
    """The motif in flat colour before a stroke is laid: what each thing is, lit by the noon sun
    and veiled by the air between. -> colours (H,W,3), the same before the air veils them, the kinds
    of ground (masks), and for the paddocks the tone of each field (bleached above 0.72, a little
    green still below 0.33)."""
    yy = np.arange(H, dtype=np.float32)[:, None] + np.zeros((1, W), np.float32)
    sky, Z, bank, rise, u, steep, sun = (v[k] for k in ("sky", "Z", "bank", "rise", "u", "steep", "sun"))
    water = ~sky & (bank < 0)
    sand = ~sky & ~water & (bank < 16 + 26 * noise.smoothstep(-1, 1.5, noise.field((H, W), 90, r))) & (u < 12)
    flat = ~sky & ~water & ~sand & (u < 30 + 40 * noise.field((H, W), 120, r)) & (rise < 0.12)
    # the cliffs show in broken bands under the rims, where the gums cannot hold
    seam = noise.stretched((W, H), 22, 240, r).T[:H, :W]
    rock = ~sky & ~water & (rise > 0.72) & (rise < 0.985) & (steep > 0.25) & (seam > 0.55 - 0.8 * (steep - 0.25))
    forest = ~sky & ~water & ~sand & ~flat & ~rock
    shallow = water & (bank - u > 120) & (-bank < 14 + 30 * noise.smoothstep(-1, 1.5, noise.field((H, W), 70, r)))
    # the paddocks are fields of their own, some cut and some standing, some a little green still
    seeds = np.stack([r.uniform(-2500, 3000, 900), r.uniform(300, 9000, 900)], 1)
    cell = spatial.cKDTree(seeds).query(np.stack([v["X"][flat], Z[flat]], 1))[1]
    field = np.zeros((H, W), np.float32)
    field[flat] = r.uniform(0.25, 0.95, 900)[cell]
    patch = land.look(land.small, v["X"], Z)               # stands of greyer and greener gums
    light = 0.42 + 0.62 * sun
    pick = lambda pal, t: pal[np.clip((t * len(pal)).astype(int), 0, len(pal) - 1)]
    col = np.zeros((H, W, 3), np.float32)
    col[forest] = pick(FOREST, 0.38 + 0.12 * patch[forest])
    col[rock] = pick(ROCK, 0.6 + 0.1 * patch[rock])
    f = field[flat]
    col[flat] = np.where((f < 0.33)[:, None], pick(GREEN_STRAW, f / 0.33),
                         np.where((f < 0.72)[:, None], pick(PADDOCK, (f - 0.33) / 0.39), pick(STRAW, (f - 0.72) / 0.23)))
    col[sand] = pick(SAND, 0.5 + 0.4 * noise.smoothstep(0, 30, bank[sand]))
    col *= light[..., None]
    t = np.clip(yy / SKYLINE, 0, 1) ** 1.6
    col[sky] = (SKY[0] * (1 - t[..., None]) + SKY[5] * t[..., None])[sky]
    base = col.copy()
    air = (1 - np.exp(-Z / HAZE))[..., None]
    land_ = ~sky & ~water
    col[land_] = (col * (1 - air) + AIR * air)[land_]
    # the water: its own deep colour, and more of what it mirrors the flatter it is seen
    e = np.clip(EYE / Z / 0.45, 0, 1)
    R = (0.1 + 0.8 * (1 - e) ** 2)[..., None]
    mr = np.clip(v["mirror"], 0, H - 1).astype(int)
    seen = col[mr, np.arange(W)[None, :]] * np.where(v["glass"] > 0.5, 0.8, 1.0)[..., None]
    wcol = WATER[2] * (1 - R) + seen * R
    wcol[shallow] = 0.45 * wcol[shallow] + 0.55 * SHALLOW[2]
    base[water] = wcol[water]
    col[water] = (wcol * (1 - 0.6 * air) + AIR * 0.6 * air)[water]
    kinds = dict(sky=sky, water=water, sand=sand, flat=flat, rock=rock, forest=forest, shallow=shallow)
    return col, base, kinds, field


def loads(pal, tone, r, accent=None, odds=0.15, spread=0.08):
    """The brush for each stroke: the pile on the palette for its tone, a neighbouring pile streaked
    into it, and a third, now and then an accent picked up from elsewhere on the palette."""
    n, N = len(pal), len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n
    i = t.astype(int)
    j = np.clip(i + np.where(r.random(N) < t - i, 1, -1), 0, n - 1)
    third = pal[np.clip(i + r.choice([-2, 2], N), 0, n - 1)]
    if accent is not None:
        hit = r.random(N) < odds
        third[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.5, 0.75, N), r.uniform(0.2, 0.4, N), r.uniform(0.02, 0.15, N)], 1)
    return np.stack([pal[i], pal[j], third], 1), share


def lay_in(rgb, sk, sky, tooth, r):
    """The first lay-in, rubbed on thin with a big brush: the masses in their colours, sunk into the
    hollows of the weave, and thinnest in the lights and the sky, so that the pale ground glows up
    through them."""
    drag = noise.stretched((W, H), 14, 300, r).T[:H, :W]
    lum = sk @ LUMA
    a = np.where(sky, 0.48 + 0.12 * drag, 0.88 - 0.25 * noise.smoothstep(0.3, 0.75, lum) + 0.1 * drag)
    a = np.clip(a * (1.1 - 0.35 * tooth), 0, 0.95)
    rgb += (ndimage.gaussian_filter(sk, (5, 9, 0)) - rgb) * a[..., None]


def along(q, smooth, lo, hi):
    """Which way the brush goes: along the lines on which `q` (H,W) holds level, smoothed over
    `smooth` px, and lying level wherever q changes by less than `lo` to `hi` per px. -> angles (H,W)"""
    gy, gx = np.gradient(ndimage.gaussian_filter(q, smooth))
    a = np.arctan2(gx, -gy)
    lev = noise.smoothstep(lo, hi, np.hypot(gx, gy))
    return (0.5 * np.arctan2(lev * np.sin(2 * a), 1 - lev + lev * np.cos(2 * a))).astype(np.float32)


def pieces(path, length, overlap, r):
    """Cut a long line into the strokes a brush lays it in, each starting a little back over the last."""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def walk(x, y, angle, length, bend, n, r, wander=0.0):
    """Paths drawn freehand from (x, y): n points along `angle`, turning `bend` per px and
    wandering a little. All arguments one per path. -> (N, n, 2)"""
    step = (length / (n - 1))[:, None]
    turn = angle[:, None] + bend[:, None] * step * np.arange(n) + np.cumsum(r.normal(0, wander, (len(x), n)), 1)
    d = np.stack([np.cos(turn), np.sin(turn)], -1) * step[..., None]
    d[:, 0] = 0
    return np.stack([x, y], 1)[:, None, :] + np.cumsum(d, 1)


def limb(rows, k, r):
    """A limb as k ribbons of bark side by side, twisting a little as they go up. -> [(path,
    half-widths, side)], side running from -1 on the lit left to 1 on the shaded right."""
    A = brush.path(np.asarray(rows, float), 3.0)
    d = np.gradient(A[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm *= np.sign(nrm[:, :1] + 1e-9)
    t = np.linspace(0, 1, len(A))
    out = []
    for j in range(k):
        f = np.clip(0.8 * ((2 * j + 1) / k - 1) + 0.15 * np.sin(5 * t + r.uniform(0, 6.3)), -0.9, 0.9)
        out.append((A[:, :2] + nrm * (A[:, 2] * f)[:, None], A[:, 2] * 1.3 / k, f))
    return out, A


def foliage(masses, holes, r):
    """Gum leaves in drooping masses: each mass a few strands of short falling touches hung from its
    upper part, thick near the twig and thinning to loose single touches at the tip, with the river
    showing between the strands and through the holes (`holes` (H,W) > 0); the touches as large as
    the mass is. -> paths (N, 5, 2), half-widths, how lit each is (-1 deep in the shade .. 1 in the
    sun), its mass"""
    out = []
    for k, (cx, cy, rx, ry, thick) in enumerate(masses):
        n = r.integers(3, 7)
        for x0, y0, a, b in zip(cx + rx * r.uniform(-0.75, 0.75, n), cy - ry * r.uniform(0.1, 0.55, n),
                                rx * r.uniform(0.22, 0.42, n), ry * r.uniform(0.55, 1.0, n)):
            m = int(thick * np.pi * a * b / 24)
            t = r.uniform(0, 1, m) ** 0.8                     # down the strand, thickest near the top
            sway = a * r.uniform(0.3, 0.9) * np.sin(r.uniform(0, 6.3) + r.uniform(2, 4) * t)
            x, y = x0 + sway + a * r.normal(0, 0.5, m) * (1 - 0.45 * t), y0 + 2 * b * t
            out.append((x, y, -(0.45 * (x - cx) / rx + 0.95 * (y - cy) / ry) + 0.25 * r.normal(0, 1, m), np.full(m, k)))
    x, y, lit, mass = (np.concatenate(z) for z in zip(*out))
    keep = holes[np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1)] < 0
    x, y, lit, mass = x[keep], y[keep], lit[keep], mass[keep]
    size = np.array([(rx * ry) ** 0.5 for _, _, rx, ry, _ in masses])[mass]
    N = len(x)
    L = np.clip(r.uniform(0.12, 0.3, N) * size * np.exp(r.normal(0, 0.25, N)), 10, 50)
    hang = np.pi / 2 + r.normal(0, 0.3, N)
    paths = walk(x - 0.5 * L * np.cos(hang), y - 0.5 * L * np.sin(hang), hang, L, r.normal(0, 0.012, N), 5, r, 0.04)
    return paths, r.uniform(3.2, 7.0, N) * np.clip(size / 100, 0.7, 1.3), lit, mass


def within(paths, mask):
    """A stroke stops where its passage ends: each path is cut back to where it first leaves `mask`
    and laid out again over the same number of points."""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, 4 * n)
    i0 = np.minimum(t.astype(int), n - 2)
    fine = paths[:, i0] * (1 - (t - i0))[None, :, None] + paths[:, i0 + 1] * (t - i0)[None, :, None]
    out = ~mask[np.clip(fine[..., 1].astype(int), 0, H - 1), np.clip(fine[..., 0].astype(int), 0, W - 1)]
    end = np.maximum(t[np.where(out.any(1), np.argmax(out, 1), 4 * n - 1)], 1.0)
    s = np.linspace(0, 1, n)[None, :] * end[:, None]
    j0 = np.minimum(s.astype(int), n - 2)
    g = (s - j0)[..., None]
    return np.take_along_axis(paths, j0[..., None], 1) * (1 - g) + np.take_along_axis(paths, (j0 + 1)[..., None], 1) * g


# the big gum on the slope below, framing the view: trunk and limbs as rows (x, y, half-width) from
# the rock up; its leaves in drooping masses (x, y, rx, ry, how thick) hung from the ends of the limbs
GUM_LIMBS = [
    [(2985, 1800, 64), (2950, 1600, 59), (2885, 1420, 54), (2895, 1220, 49), (2945, 1030, 44), (2915, 850, 39),
     (2850, 700, 33)],
    [(2850, 700, 23), (2770, 585, 18), (2670, 500, 13), (2590, 440, 9), (2510, 400, 5.5)],
    [(2850, 700, 22), (2890, 540, 17), (2870, 380, 13), (2910, 220, 9), (2950, 70, 5.5)],
    [(2915, 850, 18), (3020, 770, 14), (3110, 670, 10), (3210, 590, 7)],
    [(2895, 1220, 17), (2790, 1140, 13), (2690, 1100, 9), (2590, 1085, 5.5)],
]
LEAVES = [(2530, 480, 135, 140, 1.0), (2640, 560, 105, 100, 0.7), (2940, 170, 200, 150, 1.0),
          (2810, 350, 140, 125, 0.85), (3070, 320, 150, 120, 0.9), (3160, 680, 140, 175, 1.0), (3050, 790, 100, 95, 0.6),
          (2600, 1170, 130, 110, 0.9), (2730, 1180, 80, 75, 0.5), (2710, 420, 70, 65, 0.6), (3190, 440, 85, 100, 0.8),
          (2560, 670, 55, 60, 0.45), (3130, 970, 75, 100, 0.7), (2440, 450, 55, 70, 0.5)]
# the rocks in front: the ledge the painter stands on at the left and the one the gum grows from at
# the right, each as its outline against the view and the edge where its top breaks over
LEDGE_TOP = np.array([(-30, 1400), (130, 1372), (270, 1388), (410, 1366), (560, 1405), (700, 1470), (830, 1560),
                      (930, 1680), (980, 1760)], np.float32)
LEDGE_LIP = np.array([(-30, 1500), (160, 1478), (340, 1505), (500, 1530), (640, 1590), (760, 1670), (850, 1790)],
                     np.float32)
ROOT_TOP = np.array([(2330, 1760), (2440, 1640), (2580, 1590), (2760, 1565), (2960, 1548), (3220, 1540)], np.float32)
ROOT_LIP = np.array([(2330, 1800), (2460, 1700), (2600, 1660), (2780, 1632), (2980, 1615), (3220, 1610)], np.float32)


def copse(land, v, r, n=9000):
    """The few trees left standing when the flats were cleared, and the odd one by the water, where
    the eye meets them: -> (x, y of the foot, crown radius px, distance m, by the water)"""
    X, Z = r.uniform(-1800, 2800, n), np.geomspace(450, 9000, n)[r.permutation(n)]
    h, bank, _, u = land(X, Z)
    keep = ((u < 15) & (bank > 30) & (r.random(n) < 0.006)) | ((bank > 8) & (bank < 30) & (r.random(n) < 0.04))
    x, y = W / 2 + F * X / Z, SKYLINE + F * (EYE - h) / Z
    keep &= (x > 0) & (x < W) & (y > 0) & (y < H)
    x, y, Z, bank = x[keep], y[keep], Z[keep], bank[keep]
    seen = np.abs(v["Z"][y.astype(int), x.astype(int)] - Z) < 0.05 * Z
    R = F * r.uniform(5, 9, len(x)) / Z
    return x[seen], y[seen], R[seen], Z[seen], bank[seen] < 30


def paint(seed=1896):
    r = noise.rng(seed)
    land = Land(r)
    v = survey(land)
    sk, base, kind, field = sketch(v, land, r)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#eee3c8", thread=3.0)
    rgb, height = ground.color.copy(), ground.tooth * 0.2
    lay_in(rgb, sk, v["sky"], ground.tooth, r)
    lumb = base @ LUMA
    Z = v["Z"]
    near = np.clip((700 / Z) ** 0.55, 0.3, 1.25)          # how large the marks are there
    air = 1 - np.exp(-Z / HAZE)
    flow = along(np.where(v["sky"], v["h"].max(), v["h"]), 4, 0.05, 0.4)    # the lie of the ground
    shore = along(np.clip(v["bank"], -60, 600), 6, 0.1, 1.0)               # the flats lie along the river
    level = (0.03 * noise.field((H, W), 300, r)).astype(np.float32)

    def strew(mask, gap, scale=None):
        """Where the strokes of a passage start: a shaken honeycomb over the mask, `gap` px apart,
        closer together where `scale` (H,W) says the marks are smaller."""
        g = gap * (0.3 if scale is not None else 1)
        P = dabs.scatter((H + 2 * int(g + 2), W + 2 * int(g + 2)), g, r) - int(g + 2)
        P = P[mask[at(P)]]
        return P if scale is None else P[r.random(len(P)) < (0.3 / scale[at(P)]) ** 2]

    def brushwork(paths, w, pal, tone, veil=0.0, accent=None, odds=0.15, spread=0.08, back=0.5, dry=(0.55, 0.9),
                  **kw):
        """Lay strokes along `paths`, each loaded from `pal` by its `tone` (0 dark, 1 light) and let
        down with the colour of the air by `veil`, in the steps a painter mixes it in. Each brush
        runs out of paint at its own pace, between the bounds of `dry`."""
        if not len(paths):
            return
        N = len(paths)
        rev = r.random(N) < back
        paths[rev] = paths[rev, ::-1]
        cols, share = loads(pal, tone, r, accent, odds, spread)
        a = np.clip(np.floor(np.asarray(veil) * 8 + r.random(N)) / 8, 0, 1)[:, None, None]
        kw = dict(dict(land=0.1, lift=0.15, fray=2.5, ends=(0.35, 0.2), taper=0.35), **kw)
        impasto.lay(rgb, height, paths, w, (cols * (1 - a) + AIR * a).astype(np.float32), r, share=share,
                    spent=r.uniform(*dry, N), **kw)

    def passage(P, angle, L, w, pal, tone, veil=0.0, n=10, bend=0.0, tilt=0.0, inside=None, room=5, **kw):
        """One passage: strokes from the seeds P along `angle`, kept within `inside` (grown by `room` px)."""
        if len(P):
            paths = impasto.follow(angle, P, L, n, bend, tilt)
            if inside is not None:
                paths = within(paths, ndimage.maximum_filter(inside, 2 * room + 1) if room else inside)
            brushwork(paths, w, pal, tone, veil, **kw)

    def span(P, lo, hi, vary=0.3):
        return r.uniform(lo, hi, len(P)) * np.exp(r.normal(0, vary, len(P)))

    def tone_of(P, pal, jitter=0.04):
        """Where the sketch's own colour there falls between the darkest and lightest piles of `pal`."""
        lp = pal @ LUMA
        return np.clip((lumb[at(P)] - lp[0]) / (lp[-1] - lp[0]), 0, 1) + r.normal(0, jitter, len(P))

    def clusters(P, k, reach):
        """Each seed opened into a little group of k strokes laid round it, as a painter dabs a tree's
        crown in a few touches of one mixture. -> seeds, and the index of the group each belongs to"""
        g = np.repeat(np.arange(len(P)), k)
        return P[g] + r.normal(0, 1, (len(g), 2)) * reach[g, None], g

    def outcrop(top, lip, x0, x1):
        """A sandstone ledge in front: its sunlit top in thick warm strokes along the bedding, its face
        under the lip in shade, a few joints, and dry grass along its edge. -> where the rock is"""
        t_, l_ = np.interp(xx[0], *top.T), np.interp(xx[0], *lip.T)
        rock = (yy > t_[None, :] + 3 * noise.field((H, W), 8, r)) & (xx >= x0) & (xx < x1)
        lit = rock & (yy < l_[None, :])
        under = np.where(lit, 0.62, 0.25)[rock] + 0.08 * noise.field((H, W), 40, r)[rock]
        rgb[rock] += (LEDGE[np.clip((under * len(LEDGE)).astype(int), 0, len(LEDGE) - 1)] - rgb[rock]) * 0.9
        P = strew(lit, 38)
        n = len(P)
        passage(P, level, span(P, 60, 170), span(P, 10, 18, 0.2), LEDGE, 0.62 + 0.28 * noise.field((H, W), 90, r)[at(P)],
                n=8, tilt=np.arctan(np.gradient(t_)[at(P)[1]]) * 0.7 + r.normal(0, 0.08, n), accent=BARK_WARM,
                odds=0.25, spread=0.05, thick=0.4, dry=(0.4, 0.8), grooves=0.6, pickup=0.45, merge=3, inside=lit,
                room=4)
        P = strew(rock & ~lit, 30)
        n = len(P)
        bed = np.arctan(np.gradient(l_)[at(P)[1]]) + r.normal(0, 0.1, n)
        passage(P, level, span(P, 30, 85), span(P, 10, 16, 0.2), LEDGE, 0.15 + 0.2 * noise.field((H, W), 80, r)[at(P)],
                n=7, tilt=np.where(r.random(n) < 0.65, bed, np.pi / 2 + r.normal(0, 0.3, n)), accent=SHADE, odds=0.15,
                spread=0.05, thick=0.12, grooves=0.5, pickup=0.6, merge=3, inside=rock, room=2)
        xs = np.arange(x0 - 20.0, min(x1, lip[-2, 0]), 4)
        joints = [np.stack([xs, np.interp(xs, *lip.T) + r.normal(0, 2, len(xs))], 1)] + \
                 [np.stack([x + np.cumsum(r.normal(0, 1.5, 30)), np.interp(x, *lip.T) + np.arange(30) * 4.0], 1)
                  for x in r.uniform(x0 + 40, min(x1, lip[-2, 0]), 6)]
        cut = [j[m] for j in joints for m in pieces(j, 90, -14, r)]
        brushwork(np.stack([c[np.linspace(0, len(c) - 1, 6).astype(int)] for c in cut]), r.uniform(1.6, 3, len(cut)),
                  LEDGE[:2], np.full(len(cut), 0.3), accent=SHADE, odds=0.4, thick=0.05, pickup=0.5, merge=1)
        c = r.uniform(x0 - 10, x1, int((x1 - x0) / 90))
        cn = r.integers(1, 6, len(c))
        bx = np.repeat(c, cn) + r.normal(0, 22, cn.sum())                 # the grass grows in clumps
        by = np.interp(bx, *top.T) + r.uniform(-2, 18, len(bx))
        k = r.integers(3, 10, len(bx))
        lean = np.repeat(r.normal(0, 0.35, len(bx)), k)
        blades = walk(np.repeat(bx, k) + r.normal(0, 5, k.sum()), np.repeat(by, k),
                      -np.pi / 2 + lean + r.normal(0, 0.4, k.sum()), r.uniform(12, 70, k.sum()) * np.exp(r.normal(0, 0.3, k.sum())),
                      r.normal(0, 0.02, k.sum()), 6, r, 0.04)
        brushwork(blades, r.uniform(1.3, 2.8, k.sum()), GRASS, r.uniform(0.25, 1, k.sum()), back=0, thick=0.3,
                  dry=(0.5, 0.9), grooves=0.3, pickup=0.35, merge=1, taper=0.7)
        return rock

    # the sky: thin, in long level strokes dragged from a dry brush so that the ground glows through
    land_top = np.argmin(v["sky"], 0)                    # the first row of land in each column
    above = yy < land_top[None, :] + 14
    sway = (-0.02 + 0.05 * noise.field((H, W), 900, r)).astype(np.float32)
    for gap, L, w_, th in ((130, (500, 1300), (9, 16), 0.03), (150, (250, 700), (6, 11), 0.035)):
        P = strew(above, gap)
        n = len(P)
        t = np.clip(P[:, 1] / SKYLINE, 0, 1)
        passage(P, sway, span(P, *L, 0.4), span(P, *w_, 0.25), SKY, 0.05 + 0.88 * t ** 1.2, n=22,
                bend=r.normal(0, 0.0002, n), tilt=r.normal(0, 0.035, n), accent=SKY_TOUCH, odds=0.4, spread=0.04,
                dry=(0.75, 1.0), thick=th + 0.03 * t, grooves=0.08, lips=0.02, pickup=0.55, merge=5, fray=3,
                ends=(0.8, 0.4), taper=0.5, inside=above, room=0)

    # the far range and the plateaus, blue in the haze: level strokes, thin, soft at every edge
    far = ~v["sky"] & (Z > 13000)
    P = strew(far, 34)
    passage(P, level, span(P, 120, 320), span(P, 5, 9, 0.2), FAR, 0.05 + 0.6 * noise.smoothstep(0.5, 1, v["sun"][at(P)]),
            n=12, spread=0.04, thick=0.05, grooves=0.1, pickup=0.75, merge=6, fray=3, inside=far, room=3)
    mid = ~v["sky"] & ~kind["water"] & (Z > 3000) & (Z <= 13000)
    for k_, pal in (("forest", FOREST), ("rock", ROCK), ("flat", PADDOCK), ("sand", SAND)):
        P = strew(mid & kind[k_], 22)
        passage(P, level + 0.4 * flow, span(P, 40, 120), span(P, 3.5, 6, 0.2), pal, tone_of(P, pal), air[at(P)],
                n=8, accent=BLUEGREEN if k_ == "forest" else None, odds=0.25, thick=0.08, grooves=0.15, pickup=0.6,
                merge=4, inside=mid & kind[k_], room=4)
    # where the ridges meet the sky the heat haze takes them, in thin dry veils dragged level across
    haze = (yy > land_top[None, :] - 50) & (yy < land_top[None, :] + 60) & (v["sky"] | (Z > 4000))
    P = strew(haze, 150)
    passage(P, level, span(P, 300, 900), span(P, 5, 10, 0.2), np.stack([AIR, SKY[4], SKY[5]]), np.full(len(P), 0.5),
            n=16, dry=(0.85, 1.0), thick=0.03, grooves=0.05, pickup=0.6, merge=6, fray=3, inside=haze, room=0)

    # the near ridges: the bush in short quick touches, a few to a crown and mostly upright, olive
    # with blue-green and the blue of the shade in it; the cliffs in level strokes, thick in the sun
    P = strew(kind["forest"] & (Z <= 3000), 44, near)
    Q, g = clusters(P, 4, 12 * near[at(P)])
    inside = kind["forest"][at(Q)]
    Q, g = Q[inside], g[inside]
    n = len(Q)
    sq = near[at(Q)]
    tilt = np.where(r.random(n) < 0.6, -np.pi / 2 + r.normal(0, 0.3, n) - flow[at(Q)], r.normal(0, 0.25, n))
    passage(Q, flow, span(Q, 16, 40, 0.35) * sq, np.maximum(span(Q, 4.5, 8.5, 0.2) * sq, 2.4), FOREST,
            tone_of(P, FOREST, 0.03)[g] + r.normal(0, 0.02, n), air[at(Q)], n=6, tilt=tilt,
            accent=np.concatenate([BLUEGREEN, SHADE]), odds=0.35, spread=0.05, thick=0.1, dry=(0.6, 0.95),
            grooves=0.3, pickup=0.45, merge=2, inside=kind["forest"], room=6)
    P = strew(kind["rock"] & (Z <= 3000), 18, near)
    s = near[at(P)]
    tn = tone_of(P, ROCK)
    passage(P, flow, span(P, 25, 70) * s, np.maximum(span(P, 3, 5.5) * s, 2.2), ROCK, tn, air[at(P)], n=6,
            tilt=r.normal(0, 0.12, len(P)), thick=0.08 + 0.22 * np.clip(tn, 0, 1), grooves=0.45, pickup=0.35,
            inside=kind["rock"], room=3)

    # the flats: the paddocks field by field, gold and straw, in strokes that lie along the river,
    # and the sand at the water's edge laid thick
    P = strew(kind["flat"] & (Z <= 3000), 40, near)
    s, f = near[at(P)], field[at(P)]
    for sel, pal, tn in ((f < 0.33, GREEN_STRAW, f / 0.33), ((f >= 0.33) & (f < 0.72), PADDOCK, (f - 0.33) / 0.39),
                         (f >= 0.72, STRAW, (f - 0.72) / 0.23)):
        Q = P[sel]
        tq = 0.7 * tn[sel] + 0.3 * tone_of(Q, pal)
        passage(Q, shore, span(Q, 70, 240, 0.4) * s[sel], np.maximum(span(Q, 5, 10) * s[sel], 2.5), pal, tq,
                air[at(Q)], n=10, tilt=r.normal(0, 0.04, len(Q)), accent=SAND[2:], odds=0.12, spread=0.05,
                thick=0.12 + 0.18 * np.clip(tq, 0, 1), grooves=0.4, pickup=0.5, merge=3, inside=kind["flat"], room=5)
    P = strew(kind["sand"] & (Z <= 3000), 34, near)
    s = near[at(P)]
    passage(P, shore, span(P, 60, 200, 0.4) * s, np.maximum(span(P, 4, 8) * s, 2.2), SAND, tone_of(P, SAND),
            air[at(P)], n=8, thick=0.3, dry=(0.45, 0.85), grooves=0.4, pickup=0.5, inside=kind["sand"], room=4)

    # the trees left on the flats: a crown of a few dark clumps, the trunk under it, and its shadow
    # pooled on the ground and thrown a little to the right
    tx, ty, tR, tZ, by = copse(land, v, r)
    tv = 1 - np.exp(-tZ / HAZE)
    m = len(tx)
    brushwork(walk(np.repeat(tx - 0.4 * tR, 3), np.repeat(ty + 0.08 * tR, 3) + np.tile([0, 0.12, 0.24], m) * np.repeat(tR, 3),
                   r.normal(0.03, 0.03, 3 * m), np.repeat(tR, 3) * r.uniform(1.2, 2.0, 3 * m), np.zeros(3 * m), 4, r),
              np.repeat(np.maximum(0.16 * tR, 1.4), 3), SHADOW, np.full(3 * m, 0.4), np.repeat(tv, 3), thick=0.05,
              pickup=0.5, merge=2)
    big = tR > 4
    brushwork(walk(tx[big], ty[big], np.full(big.sum(), -np.pi / 2) + r.normal(0, 0.08, big.sum()), 0.9 * tR[big],
                   r.normal(0, 0.01, big.sum()), 4, r), np.maximum(0.1 * tR[big], 1.2), TREE[:2],
              np.full(big.sum(), 0.4), tv[big], back=0, thick=0.08, merge=1)
    c = r.integers(2, 5, m)
    ti = np.repeat(np.arange(m), c)
    cR = tR[ti] * r.uniform(0.45, 0.8, len(ti))
    cx, cy = tx[ti] + r.normal(0, 0.5, len(ti)) * tR[ti], ty[ti] - tR[ti] * np.clip(1 + r.normal(0, 0.35, len(ti)), 0.5, 1.6)
    k = np.maximum(3, (cR * 1.3).astype(int))
    si = np.repeat(np.arange(len(ti)), k)
    sx = cx[si] + r.uniform(-0.8, 0.8, len(si)) * cR[si]
    sy = cy[si] + r.uniform(-0.7, 0.6, len(si)) * cR[si]
    L = np.maximum(cR[si] * r.uniform(0.9, 1.6, len(si)), 4)
    lit = -(sx - tx[ti][si]) / tR[ti][si] - (sy - cy[si]) / cR[si]
    brushwork(walk(sx, sy + L / 2, -np.pi / 2 + r.normal(0, 0.2, len(si)), L, r.normal(0, 0.012, len(si)), 5, r),
              np.maximum(0.3 * cR[si], 1.6), TREE, 0.3 + 0.16 * lit, tv[ti][si], accent=SHADE, odds=0.3, thick=0.14,
              grooves=0.3, pickup=0.3, merge=1.5)

    # the water: long fluid strokes laid level and merged wet into wet, deep ultramarine near and
    # violet as it goes off; under the wooded banks the water holds their dark, in soft upright
    # strokes broken by the light; a few touches of glare laid thick and left
    wet = kind["water"]
    ripple = (level + 0.03 * noise.field((H, W), 240, r)).astype(np.float32)
    swell = 0.1 * noise.field((H, W), 380, r)
    for gap, L, w_, pick in ((100, (450, 1500), (7, 14), 0.85), (110, (250, 800), (3.5, 7), 0.75)):
        P = strew(wet, gap, near)
        s = near[at(P)]
        n = len(P)
        tn = tone_of(P, WATER, 0.02) + swell[at(P)] + r.normal(0, 0.05, n) * (pick < 0.8)
        passage(P, ripple, np.minimum(span(P, *L, 0.45), 1.5 * L[1]) * s, np.maximum(span(P, *w_, 0.25) * s, 2.5),
                WATER, tn, 0.35 * air[at(P)], n=24, bend=r.normal(0, 0.00015, n), tilt=r.normal(0, 0.012, n),
                spread=0.02, thick=0.05 + 0.1 * np.clip(tn, 0, 1), dry=(0.45, 0.9), grooves=0.12, pickup=pick, merge=8,
                fray=2, ends=(0.9, 0.5), taper=0.5, inside=wet, room=4)
    P = strew(kind["shallow"], 40, near)
    s = near[at(P)]
    passage(P, shore, span(P, 80, 260, 0.4) * s, np.maximum(span(P, 4, 8) * s, 2.2), SHALLOW, tone_of(P, SHALLOW),
            0.4 * air[at(P)], n=10, thick=0.1, pickup=0.65, merge=5, inside=kind["shallow"], room=4)
    glass = wet & (v["glass"] > 0.5)
    P = strew(glass, 70, near)
    s = near[at(P)]
    passage(P, ripple, span(P, 200, 600, 0.4) * s, np.maximum(span(P, 6, 12) * s, 2.5), REFLECT,
            tone_of(P, REFLECT, 0.05), 0.4 * air[at(P)], n=16, tilt=r.normal(0, 0.01, len(P)), thick=0.04, pickup=0.7,
            merge=8, ends=(0.9, 0.5), taper=0.5, inside=glass, room=2)
    P = strew(glass, 60, near)
    s = near[at(P)]
    passage(P, level + np.pi / 2, span(P, 40, 130, 0.4) * s, np.maximum(span(P, 6, 12) * s, 2.5), REFLECT,
            0.3 + 0.12 * r.normal(0, 1, len(P)), 0.5 * air[at(P)], n=6, tilt=r.normal(0, 0.04, len(P)), thick=0.03,
            dry=(0.7, 0.95), pickup=0.8, merge=8, ends=(0.9, 0.6), taper=0.5, inside=glass, room=3)
    rb = by & (tR > 3) & wet[np.clip(ty + tR, 0, H - 1).astype(int), np.clip(tx, 0, W - 1).astype(int)]
    rk = np.maximum(2, (tR[rb] * 0.4).astype(int))
    rR = np.repeat(tR[rb], rk)
    rx = np.repeat(tx[rb], rk) + np.concatenate([r.uniform(-0.6, 0.6, kk) for kk in rk]) * rR
    ry = np.repeat(ty[rb], rk) + rR * r.uniform(0.05, 0.3, len(rx))
    brushwork(walk(rx, ry, np.full(len(rx), np.pi / 2) + r.normal(0, 0.05, len(rx)), rR * r.uniform(1.0, 2.0, len(rx)),
                   np.zeros(len(rx)), 5, r), np.maximum(0.45 * rR, 1.8), REFLECT, np.full(len(rx), 0.25),
              np.repeat(tv[rb], rk), back=0, thick=0.04, dry=(0.75, 0.95), pickup=0.7, merge=4)
    P = strew(wet, 160, near)
    s = near[at(P)]
    passage(P, ripple, span(P, 300, 900) * s, np.maximum(span(P, 1.2, 2.5) * s, 1.2), WATER[5:8],
            np.full(len(P), 0.5), 0.35 * air[at(P)], n=16, thick=0.08, dry=(0.85, 1.0), grooves=0.1, pickup=0.5,
            inside=wet, room=0)
    hot = np.argwhere(wet & (yy > 1000) & (xx > 1000) & (xx < 2300))
    spot = hot[r.integers(0, len(hot), 4)]
    k = r.integers(3, 8, 4)
    gp = np.repeat(spot[:, ::-1], k, 0) + r.normal(0, 1, (k.sum(), 2)) * [90, 10]
    gp = gp[wet[at(gp)]]
    brushwork(walk(gp[:, 0], gp[:, 1], r.normal(0, 0.03, len(gp)), r.uniform(10, 45, len(gp)), np.zeros(len(gp)), 5, r),
              r.uniform(1.2, 3.0, len(gp)), GLARE, r.random(len(gp)), accent=WATER[7:], odds=0.5, thick=0.5,
              dry=(0.6, 0.95), grooves=0.4, pickup=0.3, merge=1, ends=(0.8, 0.5), taper=0.6)

    # the rocks in front, and the gum's shadow lying across the one it grows from
    outcrop(LEDGE_TOP, LEDGE_LIP, 0, 990)
    roots = outcrop(ROOT_TOP, ROOT_LIP, 2300, W)
    n = 14
    sx = r.uniform(2450, 3200, n)
    P = np.stack([sx, np.interp(sx, *ROOT_TOP.T) + r.uniform(8, 70, n)], 1)
    passage(P, level, span(P, 60, 220), span(P, 6, 14), SHADOW, np.full(n, 0.3), n=8, tilt=r.normal(0, 0.06, n),
            thick=0.05, pickup=0.6, merge=4, inside=roots, room=0)

    # the gum: the trunk in ribbons of pale bark twisting up out of the rock, deep in shade on the
    # right; the limbs thinning; the twigs drawn with the point; the leaves in drooping masses
    tips = []
    for j, rows in enumerate(GUM_LIMBS):
        ribbons, A = limb(rows, 6 if j == 0 else 3, r)
        tips.append(A[:, :2])
        for path, half, sd in ribbons:
            cut = pieces(path, 160, 40, r)
            mm = len(cut)
            brushwork(np.stack([path[c][np.linspace(0, len(c) - 1, 8).astype(int)] for c in cut]),
                      np.array([half[c].mean() for c in cut]) * r.uniform(0.85, 1.15, mm), BARK,
                      0.72 - 0.75 * np.array([sd[c].mean() for c in cut]), back=0.3, accent=BARK_WARM, odds=0.3,
                      thick=0.22, dry=(0.4, 0.8), grooves=0.5, pickup=0.4, merge=1.5, taper=0.2)
        if j == 0:          # the twist of the bark, in short strokes slanting across the trunk
            d = np.gradient(A[:, :2], axis=0)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) * np.sign(-d[:, 1:2] + 1e-9)
            i, c = r.integers(0, len(A), 170), r.uniform(-0.85, 0.85, 170)
            x0 = A[i, :2] + nrm[i] * (c * A[i, 2])[:, None]
            brushwork(walk(x0[:, 0], x0[:, 1], np.arctan2(d[i, 1], d[i, 0]) + 0.55, A[i, 2] * r.uniform(0.6, 1.4, 170),
                           r.normal(0, 0.01, 170), 5, r), r.uniform(1.5, 3.5, 170), BARK, 0.66 - 0.55 * c,
                      accent=BARK_WARM, odds=0.3, back=0.5, thick=0.2, grooves=0.4, pickup=0.5, merge=1, taper=0.5)
    limbs = np.concatenate(tips)
    tw = []
    for cx_, cy_, rx, ry, _ in LEAVES:
        foot = limbs[np.argmin(np.hypot(limbs[:, 0] - cx_, limbs[:, 1] - cy_ + 0.4 * ry))]
        for _ in range(r.integers(3, 7)):
            tw.append((foot, np.array([cx_ + r.uniform(-0.8, 0.8) * rx, cy_ + r.uniform(-0.7, 0.4) * ry])))
    tw = np.array(tw)
    d = tw[:, 1] - tw[:, 0]
    brushwork(walk(tw[:, 0, 0], tw[:, 0, 1], np.arctan2(d[:, 1], d[:, 0]), np.hypot(*d.T), r.normal(0, 0.004, len(tw)),
                   8, r, 0.09), r.uniform(1.0, 2.2, len(tw)), TWIG, r.uniform(0.2, 0.9, len(tw)), back=0, thick=0.1,
              grooves=0.1, pickup=0.2, merge=0.5, taper=0.7)
    paths, ws, lit, mass = foliage(LEAVES, noise.field((H, W), 30, r) - 0.8, r)
    deep = (r.normal(0, 0.1, len(LEAVES)) - 0.12 * (np.array(LEAVES)[:, 1] > 900))[mass]   # some masses in the others' shade
    brushwork(paths, ws, GUM, 0.5 + 0.34 * lit + deep, accent=np.concatenate([BLUEGREEN, SHADE[:1]]),
              odds=0.2 + 0.4 * (lit < -0.2), back=0.3, spread=0.05, thick=0.12 + 0.12 * (lit > 0.4), dry=(0.55, 0.95),
              grooves=0.25, pickup=0.4, merge=1.5, taper=0.45, ends=(0.5, 0.3))

    height = ndimage.gaussian_filter(height, 1.1)      # the lamp sees the surface a little softer than the brush left it
    img = dabs.shine(rgb, height, light=LAMP, relief=0.42, gloss=0, reach=(0.78, 1.16))
    return img + 0.04 * impasto.glints(height, LAMP)[..., None]
