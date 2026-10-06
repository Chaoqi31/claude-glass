"""Poppies above the Ledges, Appledore. Oil on canvas, painted out of doors in the July sun.

Celia Thaxter kept a small garden beside her cottage on Appledore, the largest of the Isles
of Shoals, a few miles out to sea from Portsmouth: poppies, hollyhocks, cornflowers and the
rest sown thick in a plot fenced against the wind, on an island that is otherwise bare
granite. Childe Hassam came to stay summer after summer, and from 1890 to 1894 he painted her
garden again and again in the full light of high summer: each flower a quick loaded touch or
twist of nearly pure colour, the leaves and grasses round them in short broken strokes of
a dozen greens, the ledges below in planes of warm ochre and lilac, and the sea beyond in
level strokes of blue and violet with the light lying on it in white. He worked on a pale
ground and let it show through, so the whole canvas keeps the brightness of the day.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Poppies above the Ledges, Appledore"
DATE = "2026"
MEDIUM = ("Oil on canvas, on a pale ground left showing: the flowers in loaded touches of pure colour, the leaves "
          "and grasses in short broken strokes, the sea in level ones")
AFTER = "Childe Hassam, the paintings of Celia Thaxter's garden, Appledore, 1890–94"
ROOM = "The Garden"
YEAR = 1890
PLACE = "Appledore, Isles of Shoals"
REGION = "Americas"
NOTE = ("A cottage garden on a granite island in July, poppies and cornflowers running down to the ledges, "
        "hollyhocks against the sea. The flowers went on last, each in a few loaded touches of colour.")

H, W = 2100, 2560
LIGHT = (-0.6, -0.5, 0.62)                                   # the gallery lamp, raking the paint
SUN = np.float32([-0.62, -0.5, 0.6]) / np.float32(np.linalg.norm([-0.62, -0.5, 0.6]))   # high, from the left
LIT = float(np.arctan2(SUN[1], SUN[0]))                      # the way toward the sun, on the canvas
HORIZON = 640.0
ISLANDS = ((140, 1060, 26), (1220, 1410, 9))                 # the far islands: from, to, how high
# the granite: centre, half-length, half-height of each slab, and whether it lies among the flowers
LEDGES = ((2360, 808, 330, 46, 0), (2010, 834, 170, 27, 0), (2300, 1010, 340, 120, 0), (2400, 1200, 280, 120, 0),
          (1960, 1120, 240, 120, 0), (1655, 1140, 140, 70, 0), (1400, 1150, 92, 22, 0), (985, 1050, 300, 62, 0),
          (600, 990, 330, 55, 0), (175, 975, 240, 48, 0), (2170, 1420, 250, 112, 1), (2490, 1610, 170, 95, 1))
HOLLYHOCKS = ((300, 1615, 640, 0.28, 1, 0.42), (275, 1630, 430, 0.13, 1, 0.74),  # a clump: x and y of the foot, the top,
              (245, 1645, 330, 0.0, 3, 0.55), (215, 1660, 560, -0.22, 2, 0.74),   # the lean, the palette in BLOOMS, its tone;
              (240, 1690, 230, -0.08, 1, 0.55))                                     # the farthest first


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
SKY = pal("#bccde2", "#c9d6e5", "#d6dee7", "#e3e5e4", "#eeebe2", "#f5efe4")
SEA = pal("#213f8a", "#2a4b98", "#3457a4", "#4064ad", "#5373b8", "#6b86c2", "#8d9fcc")   # ultramarine, cobalt
VIOLET = pal("#43438c", "#53519a", "#6662a8", "#7b74b4", "#948bc2")
TURQ = pal("#25808e", "#33949f", "#47a8aa", "#62bab2", "#86cbbe")                         # viridian, cerulean
GLITTER = pal("#e6ebf3", "#f2f2ee", "#fbf9f1", "#fffdf6")
STONE = pal("#bfa27e", "#cfb38d", "#dcc39e", "#e6d2b2", "#eee0c8", "#f5ecdb")            # granite in the sun
PINKSTONE = pal("#c39184", "#d0a495", "#dbb6a7", "#e6c9bb", "#efd9cd")
LILAC = pal("#6e6584", "#80779a", "#9289aa", "#a59db8", "#b9b2c7", "#cbc5d3")             # and in the shade
WEED = pal("#4e3e2c", "#64492f", "#7c5a38", "#5f5a33", "#77703f")                        # rockweed at the tide line
DEEP = pal("#1c3027", "#233d2e", "#2b4a36", "#355840")
GREEN = pal("#3c6243", "#4a7348", "#59834f", "#6a9358", "#7da264")
SAP = pal("#7e9a4c", "#92aa56", "#a6b963", "#bac975", "#ccd78c")
BLUEGREEN = pal("#33605c", "#416e67", "#537f76", "#69948a", "#83a99c")
GREY = pal("#6f8676", "#839887", "#99ab99", "#afbeab", "#c5d0bd")                        # the glaucous poppy leaves
OLIVE = pal("#69663a", "#7e7845", "#958c53", "#aaa065")
SHADE = pal("#34416f", "#434f80", "#556192", "#6c74a4")                                  # violet-blue in the shadows
EARTH = pal("#9c7f52", "#b39465", "#c8aa79", "#d8bf91")
SCARLET = pal("#9c1d36", "#b8242f", "#d0322a", "#e1462b", "#ec6436", "#f2884f", "#f6ab78")   # crimson to coral
ROSE = pal("#9c3a62", "#b8527a", "#d0708c", "#e192a4", "#eeb5be", "#f6d4d4")
WHITE = pal("#b8adc4", "#cfc5d0", "#e2dadb", "#efe9e2", "#f8f4eb", "#fefcf5")
MAROON = pal("#4a1430", "#621c3a", "#7c2846", "#963656", "#b04c6a")
BLUE = pal("#1c2a78", "#24399a", "#2f4db8", "#4566cc", "#6683d6", "#8aa0de")             # the cornflowers
HEART = pal("#1a1622", "#262430", "#30342a", "#46502e")
GOLD = pal("#d39a1e", "#e3b22c", "#eec744", "#f6da70")
FAMS = (DEEP, GREEN, SAP, BLUEGREEN, GREY, OLIVE, SHADE, EARTH)
BLOOMS = (SCARLET, ROSE, WHITE, MAROON, BLUE, HEART, GOLD, GREY, SAP)
# the tubes, for colour broken the way the eye mixes it
HUES = dabs.palette(["#f3d23a", "#eaa63a", "#e2492c", "#cf5277", "#9a63ad", "#3c45a2", "#2e62b2", "#46a2c9",
                     "#2a8b7a", "#4aa766", "#93b54a"])
TUBE_WHITE, TUBE_DARK = "#fbf7ec", "#1f2450"


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette, for a whole field of tones."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(t.astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a neighbour on the palette streaked in, and a
    third, now and then an accent picked up from elsewhere."""
    n, N = len(colours), len(tone)
    i = (np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n).astype(int)
    j = np.clip(i + r.choice([-1, 1], N), 0, n - 1)
    third = colours[np.clip(i + r.choice([-1, 1], N), 0, n - 1)]
    if accent is not None:
        hit = r.random(N) < odds
        third[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.03, 0.2, N)], 1)
    return np.stack([colours[i], colours[j], third], 1), share


def mixed(families, which, tone, r, accent=None, odds=0.2, spread=0.06):
    """Loads for strokes drawn from several palettes, `which` saying which one each takes."""
    cols, share = np.empty((len(which), 3, 3), np.float32), np.empty((len(which), 3))
    tone = np.broadcast_to(np.asarray(tone, np.float64), (len(which),))
    for i, colours in enumerate(families):
        sel = which == i
        if sel.any():
            cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds, spread)
    return cols, share


def islands(r):
    """The far islands, low and bare on the horizon: how high they stand above it at each x."""
    x = np.arange(W, dtype=np.float32)
    h = np.zeros(W, np.float32)
    for a, b, top in ISLANDS:
        t = np.clip((x - a) / (b - a), 0, 1)
        lump = 1 + 0.35 * np.sin(np.pi * (t * r.uniform(1.2, 2.5) + r.uniform(0, 1)))
        h = np.maximum(h, top * np.clip(np.sin(np.pi * t), 0, 1) ** 0.4 * lump * (1 + 0.12 * noise.line1d(W, 50, r)))
    return h


def bank(r):
    """The far edge of the garden, y at each x: high on the left by the cottage, then running down to
    the right where the ground falls away to the ledges."""
    x = np.arange(W, dtype=np.float32)
    y = np.interp(x, [0, 450, 800, 1150, 1450, 1700, 2000, 2300, 2560],
                  [1000, 1012, 1062, 1122, 1172, 1198, 1238, 1288, 1318])
    return (y + 24 * noise.line1d(W, 230, r) + 11 * noise.line1d(W, 50, r) + 5 * noise.line1d(W, 12, r)).astype(np.float32)


def ledges(r):
    """The granite: low slabs of it, each split along its joints into a few flat planes and worn round
    on its back, the nearer in front of the farther. -> which slab owns each point (-1 none), how
    squarely the sun falls there (-1..1), the way its plane runs (radians, along the level), how far
    down the slab it lies (0 its back .. 1 its foot), and whether the slab lies among the flowers"""
    own = np.full((H, W), -1, np.int16)
    lit, run, drop = (np.zeros((H, W), np.float32) for _ in range(3))
    amid = np.zeros((H, W), bool)
    for i, (cx, cy, a, b, among) in enumerate(sorted(LEDGES, key=lambda l: l[1])):
        x0, x1 = max(0, int(cx - 1.3 * a)), min(W, int(cx + 1.3 * a))
        y0, y1 = max(0, int(cy - 1.35 * b)), min(H, int(cy + 1.35 * b))
        v, u = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        u, v = (u - cx) / a, (v - cy) / b
        z = np.full(u.shape, -1.0, np.float32)
        for j in range(r.integers(2, 4)):                    # the slab, and blocks split from it
            ou, ov, sz = (0.0, 0.0, 1.0) if j == 0 else (r.uniform(-0.55, 0.55), r.uniform(-0.25, 0.3), r.uniform(0.4, 0.7))
            m = r.integers(6, 10)
            th = r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + r.normal(0, 0.2, m)
            rk = sz * np.exp(r.normal(0, 0.1, m))
            rho = np.max([((u - ou) * np.cos(t) + (v - ov) * np.sin(t)) / k for t, k in zip(th, rk)], 0)
            rho = rho + 0.03 * noise.field(u.shape, 25, r)
            zj = sz * (r.uniform(0.9, 1.15) - 0.7 * rho - 0.3 * np.minimum(rho, 1) ** 2) + r.uniform(-0.1, 0.1) * (u - ou)
            z = np.maximum(z, np.where(rho < 1, zj, -1))
        inside = z > -1
        gy, gx = np.gradient(ndimage.gaussian_filter(np.where(inside, z, 0), 1.5) * b * 1.5)
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        sel = (slice(y0, y1), slice(x0, x1))
        own[sel][inside] = i
        lit[sel][inside] = (nrm @ SUN)[inside]
        run[sel][inside] = np.arctan2(gx, -gy)[inside]
        drop[sel][inside] = np.clip((v[inside] + 1) / 2, 0, 1)
        amid[sel][inside] = bool(among)
    return own, lit, run, drop, amid


def ring(x, y, R, k, phi, m, r, reach=(0.6, 1.0), wide=(0.32, 0.5), curl=0.35):
    """Petals round a heart, each one loaded stroke pulled out from it or pressed in toward it: a flower
    seen face on, tipped by k (1 facing us, less turned away) and turned by phi; the far petals go on
    first. -> [(path, half-width, how lit 0..1)]"""
    th = r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + r.normal(0, 0.3, m)
    c, s = np.cos(phi), np.sin(phi)
    q = np.linspace(0, 1, 4)
    out = []
    for t in th[np.argsort(np.sin(th))]:
        a = t + r.normal(0, curl) * q
        rad = R * (r.uniform(0.05, 0.2) + q * r.uniform(*reach))
        px, py = rad * np.cos(a), rad * np.sin(a) * k
        path = np.stack([x + c * px - s * py, y + s * px + c * py], 1)
        w = R * r.uniform(*wide) * (0.55 + 0.45 * np.hypot(np.sin(t), k * np.cos(t)))
        out.append((path if r.random() < 0.6 else path[::-1], w, 0.5 + 0.5 * np.cos(t + phi - LIT)))
    return out


def twist(x, y, R, k, phi, m, r):
    """A flower in a few twists of a loaded brush laid round its heart, each a short broad arc, so the
    petals come out ruffled and overlapping, never a clean round; tipped by k and turned by phi as in
    `ring`. -> [(path, half-width, how lit)]"""
    c, s = np.cos(phi), np.sin(phi)
    out = []
    for t0 in r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + r.normal(0, 0.4, m):
        t = t0 + np.linspace(0, r.uniform(0.9, 1.8), 5) * r.choice([-1, 1])
        rad = R * r.uniform(0.35, 0.6) * (1 + r.normal(0, 0.08, 5))
        px, py = rad * np.cos(t), rad * np.sin(t) * k
        mid = t[2]
        out.append((np.stack([x + c * px - s * py, y + s * px + c * py], 1), R * r.uniform(0.36, 0.5)
                    * (0.6 + 0.4 * np.hypot(np.cos(mid), k * np.sin(mid))), 0.5 + 0.5 * np.cos(mid + phi - LIT)))
    return out


def cup(x, y, R, phi, r):
    """A flower from the side, a bowl: the far petals standing up, the dim inside between them, and the
    near ones wrapped round in front in a broad curved stroke. -> [(path, half-width, how lit)]"""
    c, s = np.cos(phi), np.sin(phi)
    turn = lambda px, py: np.stack([x + c * px - s * py, y + s * px + c * py], 1)
    q, t = np.linspace(-1, 1, 5), np.linspace(0, 1, 4)
    out = []
    for side in (-1, 1):
        if r.random() < 0.8:
            out.append((turn(side * R * (0.25 + 0.5 * t) + r.normal(0, 0.06 * R), -R * (0.05 + 0.55 * t) * r.uniform(0.7, 1.1))[::-1],
                        R * r.uniform(0.24, 0.34), 0.3 + 0.4 * (side < 0)))
    out.append((turn(q * R * 0.6, -0.15 * R - 0.1 * R * (1 - q * q)), R * 0.2, 0.0))
    for _ in range(r.integers(1, 3)):
        path = turn(q * R * r.uniform(0.7, 0.95) + r.normal(0, 0.05 * R), R * (0.05 + 0.42 * (1 - q * q)) * r.uniform(0.8, 1.2))
        out.append((path[::r.choice([-1, 1])], R * r.uniform(0.3, 0.42), 0.7))
    return out


def poppy(x, y, R, fam, base, r):
    """One poppy, in the few touches it gets: face up, a ragged ring of petals round its dark heart;
    from the side, a bowl; half open or too small to need more, two or three petals twisted together;
    a bud nodding on its hooked stem; or one dropping its last petal.
    -> strokes [(path, half-width, palette, tone)], and its stem or None"""
    phi = r.normal(0, 0.35)
    u = r.random()
    heart = False
    if u > 0.88 and R > 6:                                                   # a bud, nodding
        a = np.pi / 2 + r.normal(0, 0.5)
        bud = np.array([[x, y], [x + 0.55 * R * np.cos(a), y + 0.55 * R * np.sin(a)]])
        out = [(bud, 0.26 * R, 7 + (r.random() < 0.5), r.uniform(0.3, 0.6))]
        if r.random() < 0.4:
            out.append((bud[::-1].copy(), 0.12 * R, fam, base))             # the red showing through the split
        dx = r.normal(0, 0.5 * R)
        return out, np.array([[x + dx, y + R * r.uniform(1.5, 3)], [x + 0.6 * dx, y - 0.5 * R], [x, y - 0.05 * R]])
    if R < 6 or u < 0.22:
        petals = ring(x, y, R, r.uniform(0.5, 1), phi, r.integers(2, 4), r, reach=(0.6, 1.1), wide=(0.38, 0.55), curl=0.9)
    elif u < 0.58:
        petals, heart = ring(x, y, R, r.uniform(0.45, 0.9), phi, r.integers(4, 7), r), r.random() < 0.8
    elif u < 0.84:
        petals = cup(x, y, R, phi, r)
    else:
        petals = ring(x, y, R, r.uniform(0.5, 0.9), phi, 1, r, wide=(0.4, 0.55), curl=0.6)
    out = [(p, w, fam, base + 0.45 * (l - 0.5) + r.normal(0, 0.05)) for p, w, l in petals]
    if heart and R > 7:
        h = np.array([[x - 0.1 * R, y + r.normal(0, 0.05 * R)], [x + 0.1 * R, y + r.normal(0, 0.05 * R)]])
        out.append((h, 0.17 * R, 5, r.uniform(0, 0.6)))
        if R > 22 and r.random() < 0.5:
            out.append((h + [0.02 * R, -0.04 * R], 0.07 * R, 8, 0.8))        # the seed head, a touch of pale green
    if r.random() > 0.45:
        return out, None
    bend, L = r.normal(0, 0.4 * R), R * r.uniform(1.5, 4)
    return out, np.array([[x + bend, y + 0.3 * R + L], [x + 0.5 * bend, y + 0.3 * R + 0.5 * L], [x, y + 0.3 * R]])


def cornflower(x, y, R, r):
    """A cornflower: a ragged little star of blue touches round a dark heart, or, far off or turned
    away, two or three. -> strokes [(path, half-width, palette, tone)]"""
    petals = ring(x, y, R, r.uniform(0.55, 1), r.normal(0, 0.5), r.integers(4, 8) if R > 5 else r.integers(1, 4), r,
                  reach=(0.5, 1.0), wide=(0.16, 0.27), curl=0.4)
    out = [(p, max(w, 1.2), 4, r.uniform(0.3, 0.75) + 0.3 * (l - 0.5)) for p, w, l in petals]
    if R > 6 and r.random() < 0.7:
        out.append((np.array([[x - 0.08 * R, y], [x + 0.08 * R, y + 0.05 * R]]), 0.16 * R, 3, 0.2))
    return out


def hollyhock(x0, foot, top, lean, palette, tone, r):
    """One spike of hollyhock, leaning a little out of its clump and bowed by its own weight: big
    rough leaves at its foot, a few smaller ones up the stalk, the flowers crowding round it, largest
    and widest open below and turned every way, some face on, some half open, some seen as cups from
    the side, and the buds getting smaller to its tip. -> the leaves at its foot, and the stalk with
    the leaves up it [(path, half-width, family in FAMS, tone)]; the flowers [(path, half-width, palette, tone)]"""
    span = foot - top
    t = np.linspace(0, 1, 18)                                                # 0 at the foot .. 1 at the tip
    ys = foot - t * span
    xs = x0 + lean * t * span + (0.6 * lean + r.normal(0, 0.03)) * span * t * t + 5 * np.sin(t * r.uniform(3, 6) + r.uniform(0, 6))
    xat = lambda y: float(np.interp(y, ys[::-1], xs[::-1]))
    q = np.linspace(0, 1, 4)[:, None]

    def leaf(x, y, R, a0, shade):
        """A broad rough leaf, a fan of loaded strokes out from where it joins the stalk."""
        out = []
        for a in a0 + np.linspace(-0.9, 0.9, r.integers(4, 8)) + r.normal(0, 0.15):
            path = np.array([x, y]) + q * R * r.uniform(0.65, 1.0) * np.array([np.cos(a), 0.85 * np.sin(a)])
            out.append((path, R * r.uniform(0.17, 0.26), r.choice([0, 1, 1, 3]),
                        shade + 0.3 * np.cos(a - LIT) + r.normal(0, 0.08)))
        a = a0 - 0.5 * np.sign(np.cos(a0)) + r.normal(0, 0.2)                  # the sun along its upper edge
        out.append((np.array([x, y]) + q * R * 0.8 * np.array([np.cos(a), 0.85 * np.sin(a)]), R * 0.1, 2, r.uniform(0.5, 0.9)))
        return out

    away = lambda side, lo, hi: (0 if side > 0 else np.pi) - side * r.uniform(lo, hi)
    base = []
    for y in np.sort(r.uniform(foot - 0.5 * span, foot - 60, r.integers(7, 11))):
        side = r.choice([-1, 1])
        base += leaf(xat(y), y, r.uniform(60, 100), away(side, -0.3, 0.6), r.uniform(0.15, 0.45))
    green = [(np.stack([xs, ys], 1), 4.5, 0, 0.55)]
    for y in r.uniform(top + 0.1 * span, foot - 0.42 * span, r.integers(7, 11)):
        side = r.choice([-1, 1])
        green += leaf(xat(y), y, r.uniform(20, 38), away(side, 0.2, 0.7), r.uniform(0.3, 0.65))
    flowers = []
    y, last = top + r.uniform(0.05, 0.08) * span, top + r.uniform(0.55, 0.65) * span
    while y < last:
        f = (y - top) / (last - top)                                          # 0 the highest open flower .. 1 the lowest
        R = (18 + 26 * f ** 0.7) * np.exp(r.normal(0, 0.15))
        side = r.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
        bx, by = xat(y) + side * R * r.uniform(0.4, 1.3) + r.normal(0, 0.1 * R), y + r.normal(0, 0.15 * R)
        u, face = r.random(), r.random()
        if side and u < 0.22:                                                 # seen from the side, a cup
            petals = cup(bx, by, 0.9 * R, side * r.uniform(0.6, 1.3), r)
        else:
            k, phi = ((r.uniform(0.35, 0.65), side * r.uniform(0.9, 1.5)) if side and face < 0.45 else   # turned aside
                      (r.uniform(0.35, 0.6), r.normal(0, 0.3)) if face < 0.7 else                         # tipped up or down
                      (r.uniform(0.7, 1.0), r.normal(0, 0.4)))                                            # or face on
            petals = (twist(bx, by, R, k, phi, r.integers(2, 5), r) if u < 0.75 else
                      ring(bx, by, R, k, phi, r.integers(3, 6), r, reach=(0.5, 0.8), wide=(0.5, 0.7), curl=0.25))
        dim = tone - 0.1 * (side > 0)                                         # the side away from the sun
        flowers += [(p, w, palette, dim + 0.4 * (l - 0.5) + r.normal(0, 0.05)) for p, w, l in petals]
        if not (side and u < 0.22):
            flowers.append((np.array([[bx - 0.12 * R, by], [bx + 0.1 * R, by + 0.04 * R]]), 0.22 * R, palette, tone - 0.4))
            if R > 20 and r.random() < 0.6:
                flowers.append((np.array([[bx, by - 0.04 * R], [bx + 0.05 * R, by]]), 0.08 * R, 6, 0.8))   # the pale column
        y += R * (r.uniform(0.55, 1.0) + (r.random() < 0.15) * r.uniform(0.6, 1.2))   # now and then a gap, the stalk showing
    for y in np.linspace(top, top + 0.06 * span, r.integers(5, 9)):          # the buds, smallest at the tip
        R = 4 + 9 * (y - top) / (0.06 * span)
        x = xat(y) + r.normal(0, 0.6 * R)
        flowers.append((np.array([[x, y], [x + r.normal(0, 2), y + R]]), max(0.45 * R, 1.6),
                        8 if r.random() < 0.65 else palette, 0.4 if r.random() < 0.65 else tone))
    return base, green, flowers


def stay(paths, key):
    """A stroke stops where the thing it describes ends: each path is cut back to where `key` (H,W
    labels) first differs from its value at the start, and laid out again over as many points."""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, 4 * n)
    i0 = np.minimum(t.astype(int), n - 2)
    fine = paths[:, i0] * (1 - (t - i0))[None, :, None] + paths[:, i0 + 1] * (t - i0)[None, :, None]
    lab = key[np.clip(fine[..., 1].astype(int), 0, H - 1), np.clip(fine[..., 0].astype(int), 0, W - 1)]
    out = lab != lab[:, :1]
    end = np.maximum(t[np.where(out.any(1), np.argmax(out, 1), 4 * n - 1)], 0.6)
    s = np.linspace(0, 1, n)[None, :] * end[:, None]
    j0 = np.minimum(s.astype(int), n - 2)
    g = (s - j0)[..., None]
    return np.take_along_axis(paths, j0[..., None], 1) * (1 - g) + np.take_along_axis(paths, (j0 + 1)[..., None], 1) * g


def level(f):
    """The direction along the level lines of a field."""
    gy, gx = np.gradient(f)
    return np.arctan2(gx, -gy).astype(np.float32)


def paint(seed=1890):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#eee6d4", thread=3.0)

    # the places: the sky, the far islands, the sea, the granite, and the garden falling to it
    isle = islands(r)
    own, lit, run, drop, amid = ledges(r)
    edge = bank(r)
    rock = own >= 0
    garden = (yy > edge[None]) & ~(rock & amid & (drop < 0.55 + 0.12 * noise.field((H, W), 60, r)))
    rock &= ~garden
    land = rock | garden
    sea = (yy >= HORIZON) & ~land
    sky = yy < HORIZON - isle[None]
    island = ~sky & (yy < HORIZON)
    key = np.select([sky, island, sea, rock], [0, 1, 2, 10 + own], 3).astype(np.int16)
    depth = np.clip((yy - edge[None]) / (H - edge[None]), 0, 1)              # the garden, 0 at its far edge .. 1 at our feet
    s = (0.42 + 1.1 * depth ** 1.15).astype(np.float32)                      # how large things are there
    far = np.clip((yy - HORIZON) / 520, 0, 1)                                # the sea, 0 at the horizon
    ps = (0.45 + 0.55 * far).astype(np.float32)
    shore = ndimage.distance_transform_edt(~land).astype(np.float32)         # px out from the land

    # the design, as the colours add up across the room. A pale sky, a little bluer overhead, with a
    # bank of fair-weather cloud low on the right; the sea deep blue, hazed along the horizon, violet
    # in patches and turquoise over the shallows; the granite cream, ochre and rose in the sun and
    # lilac in the shade, dark with weed at the tide line; the garden in greens by the light on its clumps
    hi = np.clip((HORIZON - yy) / HORIZON, 0, 1)
    want = ramp(SKY, 1 - hi ** 0.5 + 0.05 * noise.fbm((H, W), 420, r, octaves=3))
    want = mix(want, lin("#f4e6d8"), 0.35 * np.exp(-((HORIZON - 70 - yy) / 90) ** 2))     # warmer just above the haze
    cloud = noise.smoothstep(0.55, 1.6, noise.fbm((H, W), 120, r, octaves=4) - 0.7
                             + 1.4 * np.exp(-((yy - 500) / 90) ** 2) * noise.smoothstep(1300, 2300, xx)) * (yy < HORIZON - 20)
    want = mix(want, lin("#f9f6ee"), 0.7 * cloud)
    want = np.where(island[..., None], mix(lin("#a395ab"), lin("#c8b8b0"), noise.smoothstep(HORIZON - 4, HORIZON - 14, yy)), want)
    deep = ramp(SEA, 0.88 - 0.62 * noise.smoothstep(0, 0.4, far) + 0.08 * noise.fbm((H, W), 220, r, octaves=3))
    deep = mix(deep, lin("#a6b1d2"), 0.6 * np.exp(-(yy - HORIZON) / 18))
    deep = mix(deep, ramp(VIOLET, 0.45), 0.45 * noise.smoothstep(0.6, 1.6, noise.field((H, W), 180, r)) * far)
    shallow = np.clip(1.1 * np.exp(-shore / 120) * noise.smoothstep(HORIZON + 200, HORIZON + 380, yy), 0, 1)
    deep = mix(deep, ramp(TURQ, 0.35 + 0.45 * np.exp(-shore / 35)), shallow)
    want = np.where(sea[..., None], deep, want)
    slab = r.uniform(-0.12, 0.12, len(LEDGES) + 1)[own]
    sunlit = noise.smoothstep(-0.2, 0.6, lit)
    pink = 0.6 * noise.smoothstep(-0.3, 1.2, noise.field((H, W), 260, r))
    stone = mix(ramp(LILAC, 0.2 + 0.7 * sunlit + slab),
                mix(ramp(STONE, 0.25 + 0.55 * sunlit + slab), ramp(PINKSTONE, 0.25 + 0.55 * sunlit + slab), pink),
                noise.smoothstep(0.35, 0.65, sunlit))
    below = np.zeros((H, W), bool)
    for k in range(1, 17):
        below |= np.roll(sea, -k, 0)
    tide = ndimage.gaussian_filter((rock & below).astype(np.float32), 2)
    stone = mix(stone, ramp(WEED, 0.5 + 0.3 * noise.field((H, W), 50, r)), 0.85 * tide)
    want = np.where(rock[..., None], stone, want)
    dome = noise.fbm((H, W), 150, r, octaves=4)
    gy, gx = np.gradient(ndimage.gaussian_filter(dome, 3) * 40)
    glit = (-gx * SUN[0] - gy * SUN[1] + SUN[2]) / np.sqrt(gx * gx + gy * gy + 1)
    glit = (glit - glit.mean()) / (glit.std() + 1e-6)
    glight = np.clip(0.48 + 0.12 * dome + 0.1 * glit + 0.05 * noise.fbm((H, W), 40, r, octaves=2) + 0.08 * (1 - depth),
                     0, 1).astype(np.float32)
    green = mix(ramp(np.concatenate([DEEP, GREEN, SAP]), glight), ramp(GREY, 0.7), 0.4 * (1 - depth) ** 2)
    want = np.where(garden[..., None], green, want).astype(np.float32)

    # which way the strokes run: in long easy slants across the sky, level over the sea, along the
    # planes of the granite, and up the garden, leaning to the right
    calm = (-0.25 + 0.35 * noise.field((H, W), 600, r)).astype(np.float32)
    coast = level(ndimage.gaussian_filter(shore, 8))
    flat = (0.025 * noise.field((H, W), 300, r)).astype(np.float32)
    upright = (-np.pi / 2 + 0.38 + 0.3 * noise.field((H, W), 300, r)).astype(np.float32)
    flow = np.select([sky, island, sea, rock], [calm, 0 * calm, flat, run], upright).astype(np.float32)

    # the canvas rubbed over first with thin paint of the colour of each place; the pale ground shows through
    rag = noise.stretched((W, H), 14, 220, r).T          # rubbed on level, as the sky and the sea run
    alpha = np.clip(0.8 + 0.06 * rag + 0.12 * garden, 0, 1)[..., None]
    rgb = (ground.color * (1 - alpha) + want * (1 + 0.04 * rag[..., None]) * alpha).astype(np.float32)
    height = ground.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def broken(P, pure=(0.3, 0.6), odds=0.04, spread=0.12, vivid=0.03):
        """The brush for each stroke: the colour of the place pushed part of the way toward a colour
        nearly pure from the tube let down to its value, now and then all the way, a second such
        colour streaked beside it, and a little of the colour the place adds up to."""
        T = want[at(P)]
        n = len(P)
        lt = np.log(np.clip(T, 1e-4, 1))

        def one():
            tube = np.log(np.clip(dabs.tint(T, HUES, r, TUBE_WHITE, TUBE_DARK, vivid=vivid, spread=spread), 1e-4, 1))
            f = r.uniform(*pure, n)
            f[r.random(n) < odds] = 1
            return np.exp(lt + (tube - lt) * f[:, None])
        share = np.stack([r.uniform(0.55, 0.8, n), r.uniform(0.1, 0.3, n), r.uniform(0.1, 0.25, n)], 1)
        return np.stack([one(), one(), T * np.exp(r.normal(0, 0.04, (n, 1)))], 1).astype(np.float32), share

    def leafage(P, lift=0.0, wander=0.1, odds=0.15):
        """The paint for a stroke of leaves at each seed: the green for its light, and among them the
        violet-blue of the shadows, blue-greens, the grey of the poppy leaves, olive and bare earth."""
        y, x = at(P)
        n = len(P)
        l = glight[y, x] + lift + r.normal(0, wander, n)
        fam = np.digitize(l, [0.33, 0.6])
        tone = np.clip(np.select([fam == 0, fam == 1], [l / 0.33, (l - 0.33) / 0.27], (l - 0.6) / 0.4), 0, 1)
        u = r.random(n)
        fam[(fam == 0) & (u < 0.3)] = 6
        fam[(fam == 1) & (u > 0.72)] = 3
        fam[(fam == 1) & (u < 0.14)] = 4
        fam[(fam == 2) & (u > 0.86)] = 5
        fam[(fam == 2) & (u < 0.06)] = 7
        fam[r.random(n) < 0.4 * noise.smoothstep(0.3, 0.0, depth[y, x])] = 4      # the far garden greyer
        return mixed(FAMS, fam, tone, r, np.concatenate([SAP[2:], BLUEGREEN[1:3], SHADE[2:]]), odds)

    def go(odds, spacing, length, width, load, field=flow, scale=None, tilt=0.0, bend=0.0, n=6, group=(1, 1), flip=0.5,
           dry=0.0, cut=None, **kw):
        """A passage. The strokes go down in small groups, a few laid side by side from one loading
        of the brush at one slant, each drier than the last, as a painter hatches; then the next."""
        P = strew(spacing * np.sqrt(np.mean(group)), odds)
        G = len(P)
        m = r.integers(group[0], group[1] + 1, G)
        g = np.repeat(np.arange(G), m)
        j = np.arange(len(g)) - np.repeat(np.cumsum(m) - m, m)                   # its place in the group
        lean = r.normal(0, tilt, G)[g]
        a = field[at(P)][g] + lean
        k = scale[at(P)][g] if scale is not None else np.ones(len(g))
        wd = r.uniform(*width, G)[g] * r.uniform(0.85, 1.15, len(g)) * k ** 0.8
        side = (j - (m[g] - 1) / 2) * 1.6 * wd
        Q = P[g] + np.stack([-np.sin(a), np.cos(a)], 1) * side[:, None] \
            + np.stack([np.cos(a), np.sin(a)], 1) * (r.normal(0, 0.4, len(g)) * wd)[:, None]
        paths = impasto.follow(field, Q, r.uniform(*length, G)[g] * r.uniform(0.75, 1.25, len(g)) * k, n,
                               r.normal(0, bend, G)[g], lean)
        if cut is not None:
            paths = stay(paths, cut)
        back = (r.random(G) < flip)[g]
        paths[back] = paths[back, ::-1]
        cols, share = load(P)
        d = (r.uniform(*dry, G) if isinstance(dry, tuple) else np.full(G, dry))[g]
        keep = odds[at(Q)] > 0.02                                                 # a group stops at the edge of its passage
        impasto.lay(rgb, height, paths[keep], wd[keep],
                    (cols[g] * np.exp(r.normal(0, 0.03, (len(g), 1, 3))))[keep].astype(np.float32), r,
                    share=share[g][keep], wet=wet, dry=np.clip(d + 0.08 * j, 0, 0.85)[keep], **kw)

    thin = lambda m, k, k0: (m * (k0 / k) ** 2).astype(np.float32)          # fewer, larger strokes toward us
    soft = dict(ends=(0.45, 0.3), taper=0.3, fray=2.2)
    blades = dict(ends=(0.3, 0.8), taper=0.65, fray=1.6, spent=0.75)
    skyf, seaf, rockf, gard = (m.astype(np.float32) for m in (sky, sea, rock, garden))
    rim = level(ndimage.gaussian_filter(rockf, 6))

    def daub(items, families, accent=None, odds=0.2, **kw):
        """Lay a list of (path, half-width, palette, tone) strokes, in order."""
        if items:
            paths, w, f, t = zip(*items)
            lay(list(paths), np.array(w), mixed(families, np.array(f), np.array(t), r, accent, odds), **kw)

    # the sky first, thin and soft: broad strokes of pale blue let down toward cream at the horizon,
    # wet into one another, the weave of the canvas showing through them
    def air(P, swing=0.035):
        """The sky's brush: the colour of the place, a little warmer or cooler streaked into it."""
        T = want[at(P)]
        n = len(P)
        cols = np.stack([T * np.exp(r.normal(0, 0.015, (n, 3))), T * np.exp(r.normal(0, swing, (n, 1)) * np.float32([1, 0.2, -1])),
                         T * np.exp(r.normal(0, 0.03, (n, 3)))], 1)
        share = np.stack([r.uniform(0.55, 0.75, n), r.uniform(0.2, 0.35, n), r.uniform(0.05, 0.15, n)], 1)
        return np.clip(cols, 0, 1).astype(np.float32), share

    airy = dict(grooves=0.1, lips=0.01, land=0.05, lift=0.03, ends=(0.6, 0.5), taper=0.4, fray=3.0, spent=0.8, cut=key)
    go(skyf, 150, (160, 320), (26, 40), air, tilt=0.15, thick=0.02, pickup=0.7, merge=10, hide=0.8, **airy)
    go(skyf, 55, (80, 200), (14, 24), air, tilt=0.25, bend=0.001, group=(1, 2), thick=0.04, pickup=0.6, merge=6,
       dry=(0, 0.1), hide=0.8, **airy)

    # the lay-in of the rest: broad thin strokes of the colour of each place
    go(1 - skyf, 140, (90, 200), (16, 26), lambda P: broken(P, (0.0, 0.2), 0.0, 0.08), tilt=0.2,
       cut=key, thick=0.03, grooves=0.15, lips=0.02, land=0.1, lift=0.05, pickup=0.6, merge=6, spent=0.7, fray=3,
       taper=0.4, hide=0.7)
    height -= 0.8 * (height - 0.2 * ground.tooth)
    wet[:] = 0

    # the second sitting, wet into wet: each thing in its colours
    go(island.astype(np.float32), 7, (20, 70), (2.5, 4.5), lambda P: broken(P, (0.1, 0.3), 0.0, 0.06), tilt=0.03,
       cut=key, thick=0.1, pickup=0.4, merge=1, dry=(0, 0.3), **soft)
    go(thin(seaf, ps, 0.45), 20 * 0.45, (40, 140), (3, 7), lambda P: broken(P, (0.2, 0.5), 0.03, 0.1), scale=ps,
       tilt=0.03, bend=0.0005, group=(1, 3), cut=key, thick=0.1, pickup=0.45, merge=2, dry=(0, 0.3), **soft)
    go(rockf, 22, (25, 60), (6, 11), lambda P: broken(P, (0.2, 0.5), 0.03, 0.1), tilt=0.25, group=(1, 3), cut=key,
       thick=0.16, pickup=0.45, merge=2, dry=(0, 0.3), **soft)
    go(thin(gard, s, 0.42), 22 * 0.42, (16, 40), (6, 10), leafage, scale=s, tilt=0.6, group=(1, 3), thick=0.16,
       pickup=0.45, merge=2, dry=(0, 0.3), **soft)
    wet[:] = 0
    height += 0.3 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2)) * (1 - skyf)    # crusts left between sittings

    # the third, over dry paint: the sea in smaller level strokes nearer the colours of the tubes; the
    # granite plane by plane, cream, ochre and rose in the sun and lilac in the shade, its joints, and the
    # weed along the tide line
    go(thin(seaf, ps, 0.45), 16 * 0.45, (25, 90), (2.5, 5.5), lambda P: broken(P, (0.4, 0.75), 0.06, 0.14), scale=ps,
       tilt=0.03, bend=0.001, group=(1, 2), cut=key, thick=0.14, pickup=0, merge=0, dry=(0.1, 0.45), **soft)
    P = strew(20, rockf)
    n = len(P)
    y, x = at(P)
    L = sunlit[y, x]
    fam = np.where(L > 0.5 + r.normal(0, 0.1, n), (r.random(n) < 0.3 + 0.5 * pink[y, x]).astype(int), 2)
    hw = np.clip(9 * np.exp(r.normal(0, 0.35, n)), 4, 18)
    lay(stay(impasto.follow(run, P, hw * r.uniform(2.5, 5.5, n), 6, r.normal(0, 0.002, n), r.normal(0, 0.25, n)), key), hw,
        mixed((STONE, PINKSTONE, LILAC), fam, np.where(fam == 2, 0.2 + 0.7 * L, 0.35 + 0.55 * L) + slab[y, x]
              + r.normal(0, 0.08, n), r, np.concatenate([LILAC[2:], PINKSTONE[1:], WEED[2:3]]), 0.15),
        thick=0.24, spent=0.5, pickup=0, merge=0, grooves=0.4, dry=r.uniform(0.05, 0.4, n), ends=(0.5, 0.25), taper=0.35,
        fray=2.2)
    P = strew(90, rockf * 0.6)
    n = len(P)
    lay(stay(impasto.follow(run, P, r.uniform(30, 110, n), 6, r.normal(0, 0.004, n), np.pi / 2 + r.normal(0, 0.4, n)), key),
        r.uniform(1.2, 2.4, n), loads(np.concatenate([SHADE[:2], LILAC[:2]]), r.uniform(0, 1, n), r), thick=0.1, pickup=0,
        dry=r.uniform(0.2, 0.5, n), ends=(0.3, 0.6), taper=0.6, fray=1.0)
    P = strew(12, (tide > 0.3) * rockf)
    n = len(P)
    lay(stay(impasto.follow(rim, P, r.uniform(15, 45, n), 5, 0, r.normal(0, 0.15, n)), key), r.uniform(2.5, 5, n),
        loads(WEED, r.uniform(0.1, 0.9, n), r, np.concatenate([TURQ[:2], LILAC[:2]]), 0.15), thick=0.2, pickup=0,
        dry=r.uniform(0.1, 0.4, n), ends=(0.4, 0.3), taper=0.5, fray=1.5)

    # the garden: grasses drawn up in long thin strokes of many greens, leaning to the wind off the sea;
    # the leaves among them in short broken strokes every way; blue-violet in the shadows, and
    # yellow-green where the sun catches the tops
    go(thin(gard, s, 0.42), 16 * 0.42, (30, 90), (2.2, 4.0), leafage, scale=s, tilt=0.3, bend=0.004, flip=0, thick=0.2,
       pickup=0.1, merge=0, dry=(0.0, 0.35), **blades)
    go(thin(gard, s, 0.42), 20 * 0.42, (10, 26), (3.5, 6.5), lambda P: leafage(P, 0.03, 0.14), scale=s, tilt=1.2,
       group=(1, 2), thick=0.22, pickup=0, merge=0, dry=(0.1, 0.45), **soft)
    go(thin(gard * noise.smoothstep(0.42, 0.25, glight), s, 0.42), 24 * 0.42, (15, 45), (3, 6),
       lambda P: mixed(FAMS, np.full(len(P), 6), r.uniform(0.2, 0.8, len(P)), r, DEEP[1:], 0.3), scale=s, tilt=0.5,
       thick=0.18, pickup=0, merge=0, dry=(0.1, 0.4), **soft)
    go(thin(gard * noise.smoothstep(0.55, 0.8, glight), s, 0.42), 26 * 0.42, (12, 40), (2.5, 4.5),
       lambda P: leafage(P, 0.15, 0.1, 0.25), scale=s, tilt=0.4, flip=0, thick=0.24, pickup=0, merge=0, dry=(0.15, 0.5),
       **blades)
    wet[:] = 0

    # the hollyhocks, a clump of them standing up against the sea: the big leaves at their feet, then
    # each stalk with its leaves and its flowers, the farthest first
    petal = dict(thick=0.42, spent=0.4, pickup=0.15, merge=1, grooves=0.5, ends=(0.4, 0.3), taper=0.3, fray=1.3)
    leafy = dict(thick=0.24, pickup=0.3, merge=1, ends=(0.5, 0.3), taper=0.35, fray=1.8, spent=0.55)
    spikes = [hollyhock(*h, r) for h in HOLLYHOCKS]
    feet = sum((b for b, _, _ in spikes), [])
    daub(feet, FAMS, SAP[2:], 0.2, dry=r.uniform(0, 0.3, len(feet)), **leafy)
    for _, green, fl in spikes:
        daub(green, FAMS, SAP[2:], 0.2, dry=r.uniform(0, 0.3, len(green)), **leafy)
        daub(fl, BLOOMS, np.concatenate([WHITE[4:], ROSE[4:], MAROON[2:3]]), 0.2, dry=r.uniform(0, 0.25, len(fl)),
             **{**petal, "ends": (0.6, 0.55), "taper": 0.12, "fray": 1.0, "tails": 0.2, "grooves": 0.3})
    trace = Image.new("L", (W // 4, H // 4), 0)                              # where the clump stands, so nothing behind
    pen = ImageDraw.Draw(trace)                                              # it is painted over it
    for b, g, f in spikes:
        for p, w, *_ in b + g + f:
            pen.line([tuple(v) for v in p / 4], fill=255, width=int(w / 2) + 2)
    clump = np.kron(np.asarray(trace) > 0, np.ones((4, 4), bool))[:H, :W] & (yy < min(h[1] for h in HOLLYHOCKS) - 30)

    # the poppies, strewn thick where they have seeded themselves: a drift down the bank toward the cove
    # and another along the far edge, a few large ones close by; the cornflowers sprinkled among them.
    # The far ones go on first
    toward = np.float32([1700, -770]) / np.hypot(1700, 770)
    river = np.exp(-((xx * toward[1] - (yy - 1950) * toward[0]) / 330) ** 2)
    rim_ = noise.smoothstep(160, 20, yy - edge[None])
    dens = np.clip(0.12 + 0.6 * river + 0.5 * rim_ + 0.35 * noise.fbm((H, W), 200, r, octaves=3), 0, 1) * garden
    kinds = noise.field((H, W), 260, r)
    P = dabs.scatter((H, W), 18, r)
    y, x = at(P)
    R = 26 * s[y, x] ** 1.4 * np.exp(r.normal(0, 0.25, len(P)))
    ok = (r.random(len(P)) < dens[y, x] * np.minimum((8 / R) ** 2, 1)) & ~clump[y, x]
    P, R = P[ok], R[ok]
    n = len(P)
    y, x = at(P)
    k, u = kinds[y, x] + r.normal(0, 0.6, n), r.random(n)
    fam = np.select([k > 1.0, k < -1.1], [2, 1], 0)                          # white drifts, pink ones, and the reds
    base = np.select([fam == 1, fam == 2, u < 0.18, u > 0.88],
                     [r.uniform(0.45, 0.75, n), r.uniform(0.62, 0.85, n), r.uniform(0.18, 0.35, n), r.uniform(0.72, 0.85, n)],
                     r.uniform(0.5, 0.68, n)) - 0.15 * (glight[y, x] < 0.35)
    C = dabs.scatter((H, W), 22, r)
    cy, cx = at(C)
    Rc = 10 * s[cy, cx] ** 1.3 * np.exp(r.normal(0, 0.2, len(C)))
    ok = garden[cy, cx] & ~clump[cy, cx] & (r.random(len(C)) < (0.25 + 0.35 * noise.smoothstep(-0.5, 1.5, noise.field((H, W), 300, r)))[cy, cx]
                           * np.minimum((5 / Rc) ** 2, 1))
    C, Rc = C[ok], Rc[ok]
    order = np.argsort(np.concatenate([P[:, 1], C[:, 1]]))
    items, stems = [], []
    for i in order:
        if i < n:
            out, stem = poppy(P[i, 0], P[i, 1], R[i], fam[i], base[i], r)
            items += out
            if stem is not None:
                stems.append((stem, max(1.0, 0.07 * R[i]), 7, r.uniform(0.2, 0.7)))
        else:
            items += cornflower(C[i - n, 0], C[i - n, 1], Rc[i - n], r)
    daub(stems, BLOOMS, GREEN[1:], 0.3, thick=0.15, pickup=0.2, merge=0.5, ends=(0.3, 0.6), taper=0.5, fray=1.0,
         dry=r.uniform(0.1, 0.4, len(stems)))
    daub(items, BLOOMS, np.concatenate([SCARLET[4:6], ROSE[3:5], WHITE[4:]]), 0.2, dry=r.uniform(0, 0.25, len(items)),
         **petal)
    wet[:] = 0

    # last, over all of it: a few grasses and leaves in front of the flowers; the light on the sea in
    # level touches of white, thickest along the sun's path; the sea white where it laps the ledges;
    # and the sun on the backs of the rocks in thick cream
    go(thin(gard * ~clump, s, 0.42) * 0.25, 16 * 0.42, (30, 90), (2, 3.6), lambda P: leafage(P, 0.05), scale=s, tilt=0.3,
       bend=0.004, flip=0, thick=0.2, pickup=0, merge=0, dry=(0.1, 0.4), **blades)
    rows = noise.smoothstep(0.6, 1.6, noise.stretched((W, H), 8, 140, r).T[:H, :W])
    P = strew(10, seaf * rows * (0.05 + 0.6 * np.exp(-((xx - 650) / 480) ** 2)) * (0.3 + 0.7 * ps))
    n = len(P)
    hw = np.clip(2.0 * np.exp(r.normal(0, 0.4, n)), 1.0, 4.5) * ps[at(P)]
    lay(stay(impasto.follow(flat, P, hw * r.uniform(5, 12, n), 5, r.normal(0, 0.01, n)), key), hw,
        loads(GLITTER, r.uniform(0.2, 1, n), r, SKY[3:], 0.2), thick=0.4, spent=0.7, pickup=0.05, ends=(0.6, 0.5),
        taper=0.6, fray=1.0, land=0.8, lift=0.6, dry=r.uniform(0.1, 0.4, n))
    P = strew(14, seaf * noise.smoothstep(16, 3, shore) * noise.smoothstep(-0.4, 0.8, noise.field((H, W), 90, r))
              * (yy > HORIZON + 120))
    n = len(P)
    lay(stay(impasto.follow(coast, P, r.uniform(15, 50, n), 6, r.normal(0, 0.01, n)), key), r.uniform(2.5, 6, n),
        loads(np.concatenate([GLITTER, TURQ[3:]]), r.uniform(0, 1, n), r), thick=0.35, pickup=0.2, ends=(0.4, 0.2),
        taper=0.6, fray=2.0, hide=0.85, dry=r.uniform(0.2, 0.6, n))
    P = strew(36, rockf * noise.smoothstep(0.75, 0.95, sunlit))
    n = len(P)
    hw = np.clip(8 * np.exp(r.normal(0, 0.35, n)), 4, 16)
    lay(stay(impasto.follow(run, P, hw * r.uniform(1.5, 4, n), 6, r.normal(0, 0.001, n), r.normal(0, 0.2, n)), key), hw,
        loads(np.concatenate([STONE[3:], PINKSTONE[3:]]), r.uniform(0.3, 1, n), r, GOLD[3:], 0.1), thick=0.34,
        dry=r.uniform(0.15, 0.5, n), pickup=0, spent=0.6, ends=(0.5, 0.2), taper=0.35, fray=2.0)

    height = ndimage.gaussian_filter(height, 0.7)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.6, gloss=0, reach=(0.76, 1.15))
    return img + 0.04 * impasto.glints(height, LIGHT)[..., None]