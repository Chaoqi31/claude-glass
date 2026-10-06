"""Birches by the River, October. Oil on canvas, in quick broken strokes of thick paint.

In the autumn of 1895, at Gorki in Tver Governorate, Isaac Levitan painted a birch grove on
the bank of a small river in full gold under a clear sky. He worked fast and fresh. The
leaves are flickering touches of unmixed colour, lemon, amber and copper side by side, with
the sky let through between them; the trunks are drawn with a loaded brush in white, cream
and grey-violet, and marked across in black; the ground is gold over green, the grass
flicked in every way and the fallen leaves pressed flat on it; and the river goes on in long
strokes of cold blue, whose cold makes the gold sing. The grove and the river here are invented,
painted in his manner.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, spatial

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Birches by the River, October"
DATE = "2026"
MEDIUM = "Oil on canvas, in quick broken strokes of thick paint"
AFTER = "Isaac Levitan, Golden Autumn, 1895"
ROOM = "Open Air"
YEAR = 1895
PLACE = "Tver Governorate, Russia"
REGION = "Europe"
NOTE = ("A birch grove on a river bank in October, the leaves in quick touches of lemon, amber and copper, "
        "the trunks drawn with a loaded brush. The river is laid in long strokes of cold blue against the gold.")

H, W = 1960, 3000
HZ, CX = 800, 1500                  # the horizon, and the column straight ahead
F, K = 2000.0, 28000.0              # focal length in px, and that times the height of the eye in metres
LIGHT = (-0.6, -0.5, 0.62)
SUN = np.array([-0.8, -0.6])        # toward the sun, across the canvas
FALL = np.array([1.0, -0.55])       # which way shadows fall over the ground: (across, depth)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
SKY = pal("#5a86c6", "#6890cc", "#769bd1", "#86a7d6", "#97b4da", "#a9c1de", "#bccfe2", "#cfdce4")   # cobalt, cerulean, white
CLOUD = pal("#a3aac3", "#b9bdd0", "#cfcfd8", "#e2e0dc", "#efebe1", "#f8f4ea")
WOODS = pal("#6a7697", "#7682a0", "#8490aa", "#929db4", "#a1aabd")
SPRUCE = pal("#3a5062", "#435a6a", "#4e6572", "#5c717c")
RUSTWOOD = pal("#8a6652", "#9a7358", "#ab835f", "#bd9569")
GREENF = pal("#7b8f58", "#8c9e61", "#9faf6d", "#b3c07f")
STUBBLE = pal("#b09a62", "#c1a96d", "#d0b87b", "#dcc68d")
PLOUGH = pal("#7a6152", "#8a6e5b", "#9a7c66")
FALLOW = pal("#8d8a96", "#9c98a2", "#aba6ac")
RIVER = pal("#30579e", "#3b63a8", "#4770b2", "#557dbb", "#668cc5", "#799ccd", "#8eadd6", "#a6bfde")
GRASS = pal("#324c2e", "#3e5a34", "#4b6839", "#5a773e", "#6b8645", "#7e954d")
SUNGRASS = pal("#8a7a3a", "#9e8b40", "#b29b47", "#c3a951", "#d0b661", "#dcc376")   # grass in the sun, ochre to gold
SHADOW = pal("#3b4f45", "#465a50", "#52655c", "#5f6f6a")
YELLOW = pal("#77622a", "#987b27", "#bf9923", "#dab12b", "#e8c53b", "#f1d652", "#f6e377", "#faeea2")  # umber to lemon
WARM = pal("#6c3418", "#88401c", "#a44e21", "#bd6126", "#d0792c", "#de9338", "#e9ad52")            # copper to amber
ACCENT = pal("#c43f1e", "#dd5a25", "#a0ae4a", "#fff3c6")          # vermilion, scarlet, a leaf still green, a light
BARK = pal("#6c6f88", "#82869f", "#9a9eb4", "#b3b6c8", "#cbcbd3", "#e3e0da", "#f1ebdf", "#f8f2e4")   # grey-violet to cream
BLACK = pal("#181618", "#231f21", "#312b29", "#443b35")
TWIG = pal("#2f2522", "#43332c", "#5c463a", "#76604f")
BANK = pal("#3a3528", "#4b4330", "#5d5236")
OLIVE = pal("#4b5228", "#5e6430", "#747738", "#8b8a44")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ramp(colours, tone):
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(t.astype(int), len(colours) - 2)
    f = (t - i)[..., None]
    return colours[i] * (1 - f) + colours[i + 1] * f


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a neighbour on the palette streaked
    in, and a third, now and then an accent picked up from elsewhere."""
    n, N = len(colours), len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n
    i = t.astype(int)
    j = np.clip(i + np.where(r.random(N) < t - i, 1, -1), 0, n - 1)
    third = colours[np.clip(i + r.choice([-2, 2], N), 0, n - 1)]
    if accent is not None:
        hit = r.random(N) < odds
        third[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.02, 0.2, N)], 1)
    return np.stack([colours[i], colours[j], third], 1), share


def mixed(families, which, tone, r, accent=None, odds=0.2, spread=0.06):
    """Loads for strokes drawn from several palettes, `which` saying which one each takes."""
    cols, share = np.empty((len(which), 3, 3), np.float32), np.empty((len(which), 3))
    for i, colours in enumerate(families):
        sel = which == i
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds, spread)
    return cols, share


def either(paths, r, odds=0.5):
    """Some strokes are laid one way, some the other."""
    back = r.random(len(paths)) < odds
    paths[back] = paths[back, ::-1]
    return paths


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back
    over the last, and lift off now and then (keep < 1). -> index arrays into path"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


# ---- the lie of the land: the picture is a view over flat ground from a rise, so every point
# below the horizon stands for a point on the ground, and things shrink as they go back

def ground(x, y):
    """Where a point of the picture lies on the ground: (across, depth) in metres."""
    Z = K / np.maximum(y - HZ, 0.5)
    return (x - CX) * Z / F, Z


def project(X, Z):
    return CX + F * X / Z, HZ + K / Z


def flat(X, Z, a):
    """The direction in the picture of a line lying on the ground at (X, Z), running at angle `a`
    over the ground (0 across, pi/2 away)."""
    return np.arctan2(-K * np.sin(a), F * (Z * np.cos(a) - X * np.sin(a))).astype(np.float32)


# the river's course over the ground, far to near: (across, depth, half its breadth), in metres
COURSE = [(-30, 1400, 4), (-60, 900, 4.5), (-130, 470, 5), (-55, 290, 5.5), (-75, 205, 6), (-36, 150, 6.5),
          (-38, 100, 7), (-22, 70, 7), (-17, 52, 7.5), (-24, 40, 8), (-34, 30, 8.5), (-46, 22, 9)]

# the birches: (across, depth, height, girth at the foot in metres, lean, how much copper is in the crown)
GROVE = [(4, 60, 17, 0.25, -0.03, 0.6), (-6, 50, 18, 0.26, -0.04, 0.2), (26, 47, 19, 0.28, 0.04, 0.5),
         (8, 45, 20, 0.29, 0.02, 0.7), (0, 40, 22, 0.31, -0.02, 0.15), (-3, 36, 15, 0.19, 0.06, 0.35),
         (13, 34, 21, 0.34, 0.03, 0.55), (6, 29, 22.5, 0.44, -0.015, 0.25)]
# across the river, a copse and a tree by the water: (across, depth, height, copper)
FAR = [(-64, 108, 15, 0.4), (-61, 114, 12, 0.7), (-52, 105, 13, 0.3), (-64, 103, 10, 0.6), (-78, 117, 16, 0.2),
       (-65, 175, 13, 0.5), (-34, 75, 16, 0.4)]
# bushes on both banks: (across, depth, breadth, height, copper)
BUSHES = [(-60, 104, 7, 2.5, 0.6), (-70, 110, 6, 2, 0.4), (-40, 88, 4, 1.8, 0.8), (-46, 130, 5, 2, 0.5),
          (-31, 70, 3, 1.5, 0.7), (-8, 52, 3.5, 1.6, 0.6), (-11, 64, 3, 1.4, 0.5), (-17, 34, 7, 1.8, 0.45)]


# ---- the sky: high and clear, a few clouds drifting over from the west

CLOUDS = [      # each cloud as lobes (x, y, across, up), and the level of the flat floor it sits on
    ([(200, 368, 90, 20), (330, 350, 85, 30), (470, 328, 95, 40), (610, 318, 85, 46), (730, 334, 90, 36),
      (850, 326, 75, 32), (960, 350, 85, 24), (1100, 366, 100, 16), (1260, 378, 80, 10), (600, 372, 220, 18),
      (950, 380, 180, 14)], 392),
    ([(2380, 472, 70, 18), (2510, 458, 80, 26), (2650, 468, 90, 20), (2800, 484, 70, 10)], 494),
    ([(80, 652, 140, 10), (380, 642, 170, 12), (660, 656, 120, 9), (900, 666, 100, 7)], 674),
    ([(1150, 140, 110, 16), (1280, 128, 90, 12), (1390, 146, 70, 8)], 154)]


def puff(x, y, r):
    """How far into the clouds each point of a grid lies (above about 0.55 is cloud), the edges
    torn a little by the wind."""
    x = x + 22 * noise.fbm(x.shape, 12, r, octaves=3)
    y = y + 12 * noise.fbm(x.shape, 10, r, octaves=3)
    m = 0
    for lobes, floor in CLOUDS:
        m = m + sum(np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)) for cx, cy, sx, sy in lobes) \
            * noise.smoothstep(floor + 10, floor - 24, y)
    return m


def sky_field(cloud, r):
    """Which way the sky's strokes run: level, rising and falling a little across the sky, and
    in the clouds turning half round their billows."""
    s = 4
    c = cloud[::s, ::s]
    a = -0.02 + 0.05 * noise.field(c.shape, 900 / s, r)
    gy, gx = np.gradient(ndimage.gaussian_filter(c, 2))
    a = blend(a, np.arctan2(gx, -gy), 0.5 * noise.smoothstep(0.15, 0.5, c))
    return ndimage.zoom(a.astype(np.float32), s, order=1)[:H, :W]


# ---- the land

def patches(spacing, spread, r):
    """Directions in patches, each laid its own way, as a painter hatches a passage block by block."""
    s = 4
    h, w = H // s + 1, W // s + 1
    seeds = dabs.scatter((h, w), spacing / s, r)
    lab = np.full((h, w), -1, np.int32)
    q = np.clip(seeds[:, ::-1].astype(int), 0, [h - 1, w - 1])
    lab[q[:, 0], q[:, 1]] = np.arange(len(seeds))
    _, (cy, cx) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    return ndimage.zoom(r.normal(0, spread, len(seeds))[lab[cy, cx]].astype(np.float32), s, order=0)[:H, :W]


def woods(r):
    """The woods along the horizon: the far ones a low wall of crowns, and nearer, on the left,
    a dark wood of spruce drawn up in points. -> the top of each, column by column"""
    x = np.arange(W, dtype=np.float32)
    far = HZ - 24 - 10 * noise.line1d(W, 200, r) - 7 * np.abs(noise.line1d(W, 26, r))
    spruce = 852 - (24 + 6 * noise.line1d(W, 60, r)) * noise.smoothstep(900, 760, x)
    px = r.uniform(-10, 10)
    while px < 860:
        tall = r.uniform(28, 66) * (1 - 0.65 * noise.smoothstep(560, 860, px))
        spruce = np.minimum(spruce, 852 - tall * np.clip(1 - np.abs(x - px) / (0.28 * tall), 0, None))
        px += r.uniform(0.25, 0.55) * tall
    return far, np.where(spruce < 851, spruce, np.inf)


def fields(r):
    """The fields beyond the river, a patchwork of strips over the ground. -> which field each
    point below the horizon lies in; and for each field its kind (0 green, 1 stubble, 2 ploughland,
    3 fallow), its tone, and the way its furrows run over the ground"""
    s = 2
    yy, xx = np.mgrid[HZ:H:s, 0:W:s].astype(np.float32) + 0.5
    X, Z = ground(xx, yy)
    n = 300
    seeds = np.stack([r.uniform(-2200, 2200, n) / 2.8, np.exp(r.uniform(np.log(140), np.log(1700), n))], 1)
    wob = noise.field(X.shape, 40, r)
    _, k = spatial.cKDTree(seeds).query(np.stack([(X + 12 * wob) / 2.8, Z * (1 + 0.04 * wob)], -1).reshape(-1, 2))
    lab = np.zeros((H, W), np.int32)
    lab[HZ:] = ndimage.zoom(k.reshape(X.shape), s, order=0)[:H - HZ, :W]
    furrow = np.where(r.random(n) < 0.7, 0.0, np.pi / 2) + r.normal(0, 0.12, n)
    return lab, r.choice(4, n, p=[0.36, 0.34, 0.15, 0.15]), r.uniform(0.15, 0.85, n), furrow


def river(r):
    """The river over the ground. -> for each point of the picture, how far inside the water it
    lies in metres (> 0 in it), and the way the water runs across the picture there"""
    C = brush.path(np.asarray(COURSE, float), 0.5)
    s = 2
    yy, xx = np.mgrid[HZ + 2:H:s, 0:W:s].astype(np.float32)
    X, Z = ground(xx, yy)
    d, k = spatial.cKDTree(C[:, :2]).query(np.stack([X.ravel(), Z.ravel()], 1))
    tang = np.gradient(C[:, :2], axis=0)[k]
    inside, run = np.full((H, W), -99, np.float32), np.zeros((H, W), np.float32)
    up = lambda a: ndimage.zoom(a.reshape(X.shape).astype(np.float32), s, order=1)[:H - HZ - 2, :W]
    inside[HZ + 2:] = up(C[k, 2] - d) + 0.5 * noise.field((H - HZ - 2, W), 30, r)
    a = flat(X.ravel(), Z.ravel(), np.arctan2(tang[:, 1], tang[:, 0])).reshape(X.shape)
    run[HZ + 2:] = 0.5 * np.arctan2(up(np.sin(2 * a)), up(np.cos(2 * a)))
    return inside, run


# ---- the birches

def birch(x0, y0, tall, hw, lean, r, low=(0.3, 0.42)):
    """A birch, as it stands in the picture: the trunk as rows (x, y, half-width) from the foot up;
    its limbs, each rising from the trunk and hanging over at the end; and the strands of twigs
    that hang from them. -> trunk (n, 3), limbs [(6, 3)], strands [(4, 2)]"""
    t = np.linspace(0, 1, 40)
    x = x0 + tall * (lean * t + r.normal(0, 0.025) * np.sin(np.pi * t) + r.normal(0, 0.008) * np.sin(2 * np.pi * t)
                     + 0.004 * noise.line1d(40, 8, r))
    girth = hw * (1 - 0.82 * t) ** 1.15 * (1 + 0.18 * noise.smoothstep(0.08, 0, t)) + 0.5
    trunk = brush.path(np.stack([x, y0 - tall * t, girth], 1), 2.0)
    t0, wide = r.uniform(*low), (0.24 if low[0] < 0.2 else 0.17) * tall * r.uniform(0.85, 1.15)
    limbs, strands = [], []
    side = r.choice([-1, 1])
    lop = dict(zip((-1, 1), r.uniform(0.6, 1.3, 2)))                # no crown is the same on both sides

    def hang(B, lam, out):
        a = np.pi / 2 - out * r.uniform(0.05, 0.5) + np.array([0, 0.3, 0.55, 0.7]) * out * r.uniform(0, 0.4)
        strands.append(B + np.cumsum(np.stack([np.cos(a), np.sin(a)], 1) * [[0], [lam / 3], [lam / 3], [lam / 3]], 0))

    for ti in np.sort(r.uniform(t0, 0.97, int(6 + tall / 95))):
        side = -side if r.random() < 0.8 else side
        if r.random() < 0.15:
            continue
        u = (ti - t0) / (1 - t0)
        reach = wide * lop[side] * np.sin(np.pi * (0.12 + 0.88 * u)) ** 0.7 * (1 - 0.55 * u) * r.uniform(0.55, 1.25)
        phi = (0.25 + 0.5 * (1 - u)) * r.uniform(0.8, 1.2)      # from the upright: birch limbs climb, the low ones less
        L = max(reach / np.sin(phi), 0.05 * tall)
        j = int(ti * (len(trunk) - 1))
        s = np.linspace(0, 1, 6)
        P = trunk[j, :2] + L * s[:, None] * [side * np.sin(phi), -np.cos(phi)] \
            + (r.uniform(0.1, 0.55) * L * s ** 2.2)[:, None] * [0.12 * side, 1]
        P[1:] += r.normal(0, 0.035 * L, (5, 2)) * s[1:, None]       # no limb grows straight
        limbs.append(np.concatenate([P, (trunk[j, 2] * (0.45 - 0.3 * s) + 0.6)[:, None]], 1))
        for sk in np.arange(r.uniform(0.12, 0.25), 1.0, r.uniform(0.09, 0.14)):
            B = np.array([np.interp(sk, s, P[:, 0]), np.interp(sk, s, P[:, 1])])
            hang(B, L * r.uniform(0.12, 0.42) * (0.5 + 0.5 * sk), side)
    for ti in np.arange(t0 + 0.03, 0.98, r.uniform(0.03, 0.045)):   # short shoots straight off the trunk, more near the top
        lam = tall * r.uniform(0.04, 0.1) * (1 + 0.6 * (ti - t0) / (1 - t0))
        hang(trunk[int(ti * (len(trunk) - 1)), :2], lam, r.choice([-1, 1]))
    return trunk, limbs, strands


def crown(limbs, strands, cluster, r):
    """Where the leaves of a crown hang, as a density on a quarter-scale grid: thick round the
    strands and the limbs, broken into clumps with the sky between them. -> (density, envelope)"""
    s = 4
    im = Image.new("L", (W // s, H // s), 0)
    pen = ImageDraw.Draw(im)
    for P in strands:
        pen.line([tuple(p / s) for p in P], fill=255, width=max(1, int(cluster / s)))
    for P in limbs:
        pen.line([tuple(p / s) for p in P[1:, :2]], fill=200, width=max(1, int(0.8 * cluster / s)))
    a = np.asarray(im, np.float32) / 255
    soft = ndimage.gaussian_filter(a, cluster / s * 0.35)
    soft /= soft.max() + 1e-6
    clumps = noise.fbm(a.shape, 2.2 * cluster / s, r, octaves=3)
    torn = noise.field(a.shape, max(1.0, 0.5 * cluster / s), r)        # the edge of a clump is never smooth
    dens = noise.smoothstep(0.08, 0.5, soft) * noise.smoothstep(-0.7, 0.45, clumps + 0.6 * soft) \
        * noise.smoothstep(-1.4, 0.2, torn + 2.5 * soft - 0.6)
    env = ndimage.gaussian_filter(a, 1.6 * cluster / s)
    return dens, env / (env.max() + 1e-6)


def stand(X0, Z0, tall_m, girth_m, lean, r):
    """A birch set on the ground at (X0, Z0), drawn to the size its distance gives it."""
    x0, y0 = project(X0, Z0)
    tall = F * tall_m / Z0
    small = tall < 450                                              # far off, the crown starts low and hides the trunk
    trunk, limbs, strands = birch(x0, y0, tall, 0.5 * F * girth_m / Z0, lean, r, (0.05, 0.15) if small else (0.3, 0.42))
    dens, env = crown(limbs, strands, (0.15 if small else 0.065) * tall, r)
    return dict(x=x0, y=y0, tall=tall, Z=Z0, trunk=trunk, limbs=limbs, strands=strands, dens=dens, env=env)


def bush(X0, Z0, breadth, tall_m, r):
    """A low bush on the ground, a mound of twigs and leaves, drawn like a small crown."""
    x0, y0 = project(X0, Z0)
    w, tall = F * breadth / Z0, F * tall_m / Z0
    strands = []
    for _ in range(18):
        u = r.uniform(-0.5, 0.5)
        top = y0 - tall * r.uniform(0.25, 1.0) * np.sqrt(max(0.05, 1 - 4 * u * u))   # ragged, never a dome
        strands.append(np.stack([np.full(4, x0 + u * w) + r.normal(0, 0.05 * w, 4), np.linspace(top, y0, 4)], 1))
    dens, env = crown([], strands, 0.22 * tall, r)
    return dict(x=x0, y=y0, tall=tall, Z=Z0, dens=dens, env=env, trunk=None)


def shadows(r):
    """Where the grove's shadows lie on the grass: each trunk's a long band thrown to the right
    over the ground, the crowns' broken patches beyond it. -> (H,W), 0 in the sun to 1 in shade"""
    s = 2
    im = Image.new("L", (W // s, H // s), 0)
    pen = ImageDraw.Draw(im)
    d = FALL / np.hypot(*FALL)
    for X0, Z0, tall, girth, *_ in GROVE:
        G = np.array([X0, Z0]) + np.arange(0, 2.1 * tall, 0.25)[:, None] * d
        G = G[G[:, 1] > 12]
        x, y = project(G[:, 0], G[:, 1])
        wd = F * girth * 1.25 / G[:, 1]
        for i in range(len(G) - 1):
            pen.line([(x[i] / s, y[i] / s), (x[i + 1] / s, y[i + 1] / s)], fill=255, width=max(1, int(wd[i] / s)))
        for _ in range(int(3 * tall)):
            g = np.array([X0, Z0]) + r.uniform(0.75, 2.1) * tall * d + r.normal(0, 0.22 * tall) * np.array([-d[1], d[0]])
            if g[1] > 12:
                cx, cy = project(*g)
                rho = r.uniform(0.5, 1.6)
                rx, ry = F * rho / g[1] / s, K * rho / g[1] ** 2 / s
                pen.ellipse([cx / s - rx, cy / s - ry, cx / s + rx, cy / s + ry], fill=int(r.uniform(150, 255)))
    a = ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, 1.2)
    return ndimage.zoom(a, s, order=1)[:H, :W]


def paint(seed=1895):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    weave = canvas.duck((H, W), seed, tint="#e6dcc4", thread=3.2)
    X, Z = ground(xx, yy)
    near = np.clip(30 / Z, 0, 1.3) * (yy > HZ)                      # 1 at thirty metres
    vary = noise.field((H, W), 300, r)
    noisy = lambda P: vary[at(P)]

    cloud = ndimage.zoom(puff(xx[::4, ::4], yy[::4, ::4], r), 4, order=1)[:H, :W]
    flow = sky_field(cloud, r)
    far, spruce = woods(r)
    lab, kind, ftone, furrow = fields(r)
    inside, run = river(r)
    water = inside > 0
    sky = yy < np.minimum(far, spruce)[None, :] + 4
    wood = ~sky & (yy < 834)
    pines = ~sky & (yy < 852) & (yy >= spruce[None, :] - 2)
    C = brush.path(np.asarray(COURSE, float), 0.5)
    o = np.argsort(C[:, 1])
    beyond = X < np.interp(Z, C[o, 1], C[o, 0])                     # across the river
    meadow = ~water & (Z < np.where(beyond, 115, 230) * (1 + 0.12 * noise.field((H, W), 120, r)))
    tilled = ~sky & ~wood & ~pines & ~water & ~meadow
    sky_tone = lambda y: 0.14 + 0.78 * np.clip(y / HZ, 0, 1) ** 1.5
    mirror = np.clip(2 * HZ - yy, 0, HZ - 1).astype(int)
    seen = cloud[mirror, xx.astype(int)] * water                  # the clouds as the water gives them back

    # the plan of the picture, which the lay-in rubs on thin
    plan = ramp(SKY, sky_tone(yy) + 0.04 * vary)
    plan += (CLOUD[3] - plan) * noise.smoothstep(0.45, 0.9, cloud)[..., None]
    band = slice(HZ, 1200)
    tint = np.choose(kind[lab[band]][..., None], [ramp(f, ftone[lab[band]]) for f in (GREENF, STUBBLE, PLOUGH, FALLOW)])
    plan[band] = np.where(tilled[band, :, None], tint + (SKY[6] - tint) * np.clip((Z[band] - 150) / 900, 0.1, 0.5)[..., None],
                          plan[band])
    plan = np.where(wood[..., None], ramp(WOODS, 0.55 + 0.1 * vary), plan)
    plan = np.where(pines[..., None], ramp(SPRUCE, 0.4), plan)
    plan = np.where(meadow[..., None], ramp(SUNGRASS, 0.45 + 0.15 * vary) * 0.5 + ramp(GRASS, 0.6) * 0.5, plan)
    plan = np.where(water[..., None], ramp(RIVER, sky_tone(mirror) - 0.2 + 0.3 * seen), plan)
    alpha = np.clip(0.9 + 0.08 * (0.5 - weave.tooth), 0, 1)[..., None]
    rgb = (weave.color * (1 - alpha) + plan * alpha).astype(np.float32)
    height = weave.tooth * 0.25
    wet = np.ones((H, W), np.float32)

    def lay(paths, w, load, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def strew(odds, spacing):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def sized(P, w, L, cover, base):
        """Thin seeds strewn `base` apart so strokes of these sizes cover the ground `cover` times."""
        keep = r.random(len(P)) < (base / np.sqrt(2 * w * L / (cover * 0.866))) ** 2
        return P[keep], w[keep], L[keep]

    def go(field, P, L, n=6, bend=0.0, tilt=0.0, back=0.5):
        return either(impasto.follow(field, P, L, n, bend, tilt), r, back)

    def hazed(load, amount):
        """The air between: the colours of a load paled toward the blue of the horizon by `amount`."""
        return load[0] + (SKY[6] - load[0]) * np.reshape(amount, (-1, 1, 1)), load[1]

    # the sky, thin and calm: broad soft strokes laid level, cobalt overhead letting down to a
    # pale warm blue at the horizon; the clouds modelled softly into it while it is wet, and a
    # little thick white only where their tops catch the sun
    P = strew(sky & (cloud < 0.6), 110)
    n = len(P)
    t = sky_tone(P[:, 1]) + 0.02 * noisy(P) + r.normal(0, 0.015, n)
    calm_sky = np.stack([ramp(SKY, t), ramp(SKY, t + r.choice([-0.04, 0.04], n)), ramp(SKY, t - 0.02)], 1)
    lilac = r.random(n) < 0.12 * np.clip(P[:, 1] / HZ, 0, 1) ** 3
    calm_sky[lilac, 2] = pal("#a9a6c8", "#b7b2cc", "#8fb3c9")[r.integers(0, 3, lilac.sum())]
    lay(go(flow, P, r.uniform(200, 420, n), 10, r.normal(0, 0.0003, n)), r.uniform(26, 38, n),
        (calm_sky, np.stack([r.uniform(0.5, 0.7, n), r.uniform(0.2, 0.4, n), r.uniform(0.05, 0.2, n)], 1)),
        thick=0.025, spent=0.4, grooves=0.06, lips=0, land=0.1, lift=0.05, tails=0.5, pickup=0.7, merge=6, fray=4.0,
        ends=(0.8, 0.7), taper=0.4, hide=0.85)
    P = strew(noise.smoothstep(0.45, 0.65, cloud), 22)
    n = len(P)
    side, up_ = 60 * SUN, np.array([0, -38.0])
    lit = np.tanh(2.5 * (cloud[at(P - side)] - cloud[at(P + side)])) \
        + np.tanh(2 * (cloud[at(P - up_)] - cloud[at(P + up_)]))
    lay(go(flow, P, r.uniform(50, 130, n), 7, r.normal(0, 0.002, n)), r.uniform(10, 16, n),
        loads(CLOUD, 0.45 + 0.3 * lit, r, np.concatenate([SKY[5:7], pal("#c9c2d6")]), 0.2, spread=0.04),
        thick=0.05, spent=0.4, grooves=0.15, lips=0.02, land=0.15, lift=0.1, pickup=0.6, merge=5, hide=0.9)
    crest = P[(lit > 0.6) & (cloud[at(P)] > 0.7) & (r.random(n) < 0.6)]
    m = len(crest)
    lay(go(flow, crest, r.uniform(25, 60, m), 5, r.normal(0, 0.004, m)), r.uniform(6, 10, m),
        loads(CLOUD, np.full(m, 0.9), r, spread=0.05), thick=0.22, spent=0.5, pickup=0.3, merge=2, grooves=0.4)
    P = strew(noise.smoothstep(0.3, 0.45, cloud) * noise.smoothstep(0.75, 0.6, cloud), 26)
    n = len(P)
    lay(go(flow, P, r.uniform(50, 120, n), 7), r.uniform(8, 13, n), loads(SKY, sky_tone(P[:, 1]) + 0.04, r, spread=0.02),
        thick=0.03, grooves=0.06, lips=0, pickup=0.6, merge=5, hide=0.85)
    tooth = weave.tooth * 0.25
    height = np.where(sky & (cloud < 0.7), tooth + 0.3 * ndimage.gaussian_filter(height - tooth, 3), height)  # thin, it sinks in

    # the far woods, a low blue wall of crowns with rust and a little gold in it, and the spruce
    # nearer on the left, dark and drawn up in points
    P = strew(wood.astype(np.float32), 11)
    n = len(P)
    top = noise.smoothstep(14, 0, P[:, 1] - far[at(P)[1]])
    w_ = r.uniform(2.6, 4.6, n)
    crowns = top > 0.4
    lay(go(np.zeros((H, W), np.float32), P, np.where(crowns, w_ * r.uniform(1.2, 2.4, n), r.uniform(18, 40, n)), 5, 0,
           np.where(crowns, r.uniform(0, np.pi, n), r.normal(0, 0.1, n))), w_,
        hazed(loads(WOODS, 0.4 + 0.25 * top + 0.08 * noisy(P), r, np.concatenate([RUSTWOOD, YELLOW[3:5]]), 0.25), 0.3),
        thick=0.06, grooves=0.2, lips=0.05, pickup=0.5, merge=2, hide=0.9)
    P = strew(pines.astype(np.float32), 8)
    n = len(P)
    lay(go(np.full((H, W), -np.pi / 2, np.float32), P, r.uniform(12, 30, n), 5, 0, r.normal(0, 0.3, n)),
        r.uniform(2.4, 4.4, n), hazed(loads(SPRUCE, 0.4 + 0.2 * r.random(n), r, WOODS[:2], 0.2), 0.2),
        thick=0.1, grooves=0.3, lips=0.05, pickup=0.4, merge=1.5)

    # the fields, strips of green, stubble, ploughland and fallow, each laid along its furrows
    P = strew(tilled.astype(np.float32), 15)
    n = len(P)
    y, x = at(P)
    k = lab[y, x]
    lay(go(flat(X, Z, furrow[lab]), P, r.uniform(16, 44, n) * (0.6 + near[y, x] * 4), 5),
        r.uniform(2.2, 3.6, n) * (1 + 3 * near[y, x]),
        hazed(mixed((GREENF, STUBBLE, PLOUGH, FALLOW), kind[k], ftone[k] + 0.06 * noisy(P), r, SUNGRASS[2:5], 0.15),
              np.clip((Z[y, x] - 150) / 900, 0.1, 0.5)),
        thick=0.08, grooves=0.25, lips=0.05, pickup=0.45, merge=2)

    # the river, in long strokes of cold blue laid level: deeper where it gives back the sky
    # overhead, paler far off, the clouds in it a little greyer than themselves
    P = strew(water.astype(np.float32), 9)
    P, w_, L_ = sized(P, 2.6 + 9 * near[at(P)], 26 + 190 * near[at(P)] ** 0.9, 2.2, 9)
    n = len(P)
    y, x = at(P)
    level = blend(np.zeros((H, W), np.float32), run, noise.smoothstep(0.3, 0.1, near)).astype(np.float32)
    which = (r.random(n) < 0.7 * noise.smoothstep(0.5, 0.85, seen[y, x])).astype(int)
    tone = np.where(which, 0.12 + 0.22 * seen[y, x], sky_tone(mirror[y, x]) - 0.24 + 0.2 * seen[y, x]) + 0.04 * noisy(P)
    lay(go(level, P, L_, 8, r.normal(0, 0.0006, n), r.normal(0, 0.04, n)), w_ * np.exp(r.normal(0, 0.2, n)),
        mixed((RIVER, CLOUD), which, tone, r, np.concatenate([SKY[2:5], pal("#9aa3cf")]), 0.2),
        thick=0.08, spent=0.7, pickup=0.55, merge=4, grooves=0.25, lips=0.05, fray=1.5, taper=0.3, ends=(0.4, 0.2))

    # under the far bank the water gives the bank back, dark and broken, and the trees standing
    # near it in gold; then a few level strokes of the blue drawn across them
    rows = np.zeros((H, W), np.float32)
    for i in range(HZ + 1, H):
        rows[i] = (rows[i - 1] + 1) * water[i]
    under = water & (rows < F * 1.4 / Z)
    P = strew(0.85 * under, 8)
    n = len(P)
    sc = 0.4 + 1.6 * near[at(P)]
    lay(go(np.full((H, W), np.pi / 2, np.float32), P, 11 * np.exp(r.normal(0, 0.5, n)) * sc, 4, 0, r.normal(0, 0.06, n)),
        r.uniform(1.6, 3, n) * sc,
        mixed((GRASS, BANK, YELLOW), r.choice(3, n, p=[0.45, 0.35, 0.2]), r.uniform(0.25, 0.5, n), r, RIVER[2:4], 0.3),
        thick=0.14, pickup=0.6, merge=2)
    far_trees = [stand(X0, Z0, t, 0.3, r.normal(0, 0.07), r) for X0, Z0, t, _ in FAR]
    glow = np.zeros((H, W), np.float32)
    for tr in far_trees:
        x0, y0, tall = tr["x"], tr["y"], tr["tall"]
        bx = slice(int(max(0, x0 - 0.6 * tall)), int(min(W, x0 + 0.6 * tall)))
        by = slice(int(y0), int(min(H, y0 + tall)))
        qy = np.clip(((2 * y0 - yy[by, bx]) / 4).astype(int), 0, H // 4 - 1)
        glow[by, bx] = np.maximum(glow[by, bx], tr["dens"][qy, np.clip((xx[by, bx] / 4).astype(int), 0, W // 4 - 1)])
    glow *= water
    P = strew(0.4 * noise.smoothstep(0.12, 0.5, glow), 4)
    n = len(P)
    sc = 0.5 + 1.2 * near[at(P)]
    lay(go(np.full((H, W), np.pi / 2, np.float32), P, r.uniform(8, 24, n) * sc, 4), r.uniform(1.6, 3.2, n) * sc,
        mixed((YELLOW, WARM, GRASS), r.choice(3, n, p=[0.5, 0.25, 0.25]), r.uniform(0.15, 0.32, n), r, RIVER[1:3], 0.6),
        thick=0.14, pickup=0.6, merge=2, dry=r.uniform(0, 0.3, n))
    P = strew(0.5 * (under | (glow > 0.1)), 16)
    n = len(P)
    sc = 0.5 + 1.2 * near[at(P)]
    lay(go(level, P, r.uniform(18, 50, n) * sc, 5, 0, r.normal(0, 0.03, n)), r.uniform(1.2, 2.4, n) * sc,
        loads(RIVER, sky_tone(mirror[at(P)]) - 0.1, r, SKY[4:6], 0.3), thick=0.12, pickup=0.2, dry=r.uniform(0.2, 0.5, n))

    # the meadow, in short strokes lying on the ground and growing smaller as it goes back: the
    # green first, then the gold of the grass in the sun so that the green shows between, quiet
    # in places; under the birches the fallen leaves, and a few scattered over the rest
    behind = noise.smoothstep(1700, 1960, yy - 260 * noise.smoothstep(0, 1, xx / W) + 90 * noise.field((H, W), 140, r))
    shade = np.maximum(shadows(r), 0.85 * behind) * meadow             # and over the near grass, a tree behind the painter's
    lie = patches(220, 0.6, r)                                           # which way the strokes lie, patch by patch
    calm = noise.smoothstep(0.3, 1.1, noise.field((H, W), 600, r))
    upright = np.full((H, W), -np.pi / 2, np.float32)
    level0 = np.zeros((H, W), np.float32)
    hz = lambda y, x: np.clip((Z[y, x] - 60) / 700, 0, 0.25)

    def turf(odds, cover, wide=1.0, length=1.0):
        """Strokes lying on the ground, thinner and shorter as they go back, laid one way in a patch."""
        P = strew(odds, 9)
        nr = np.minimum(near[at(P)], 1.2)
        P, w, L = sized(P, (1.8 + 4.2 * nr) * wide, (20 + 30 * nr) * length, cover(nr), 9)
        y, x = at(P)
        n = len(P)
        a = flat(X[y, x], Z[y, x], lie[y, x] + r.normal(0, 0.2, n))
        return go(level0, P, L * np.exp(r.normal(0, 0.25, n)), 4, r.normal(0, 0.002, n), a), w * np.exp(r.normal(0, 0.2, n)), P, y, x

    paths, w_, P, y, x = turf(meadow.astype(np.float32), lambda nr: 0.75 + 0.8 * nr)
    n = len(P)
    dark = shade[y, x] > r.uniform(0.3, 0.6, n)
    lay(paths, w_, hazed(mixed((GRASS, SHADOW), dark.astype(int), np.where(dark, 0.35, 0.42) + 0.08 * noisy(P)
                               + 0.05 * (1 - calm[y, x]) * r.normal(0, 1, n), r, SUNGRASS[:3], 0.15), hz(y, x)),
        thick=0.14, spent=0.5, grooves=0.4, lips=0.08, pickup=0.45, merge=2, fray=1.4, ends=(0.4, 0.3), taper=0.3)
    paths, w_, P, y, x = turf(np.clip(1.1 - 1.6 * shade, 0, 1) * (1 - 0.35 * calm) * meadow, lambda nr: 0.55 + 0.75 * nr)
    n = len(P)
    lay(paths, w_, hazed(loads(SUNGRASS, 0.52 + 0.12 * noisy(P) + 0.06 * (1 - calm[y, x]) * r.normal(0, 1, n), r,
                               np.concatenate([YELLOW[3:6], WARM[4:6]]), 0.15 * (1 - calm[y, x])), hz(y, x)),
        thick=0.16, spent=0.5, grooves=0.45, lips=0.08, pickup=0.35, merge=2, dry=r.uniform(0, 0.3, n), ends=(0.4, 0.3),
        taper=0.3)
    litter = 0.03 + 0.9 * sum(np.exp(-((X - X0) ** 2 + (Z - Z0) ** 2) / (2 * 5.0 ** 2)) for X0, Z0, *_ in GROVE)
    carpet = np.clip(litter, 0, 1) * noise.smoothstep(-0.9, 0.6, noise.field((H, W), 50, r)) * meadow
    paths, w_, P, y, x = turf(carpet, lambda nr: 0.6 + 0.7 * nr, 1.1, 0.8)
    n = len(P)
    lay(paths, w_, mixed((YELLOW, WARM, SUNGRASS), r.choice(3, n, p=[0.4, 0.1, 0.5]), 0.46 + 0.12 * noisy(P)
                         - 0.35 * shade[y, x] + 0.06 * r.normal(0, 1, n), r, np.concatenate([ACCENT[:1], BANK[1:]]), 0.1),
        thick=0.2, spent=0.4, grooves=0.45, pickup=0.35, merge=2, dry=r.uniform(0, 0.3, n), ends=(0.3, 0.2))
    P = strew(np.maximum(carpet, 0.03 * meadow * (1 - calm)), 8)
    P, w_, L_ = sized(P, 2 + 4.5 * np.minimum(near[at(P)], 1.2), 5 + 9 * np.minimum(near[at(P)], 1.2), 0.15, 8)
    n = len(P)
    y, x = at(P)
    w_ = w_ * np.exp(r.normal(0, 0.3, n))
    lay(go(level0, P, w_ * r.uniform(1.2, 2.3, n), 3, r.normal(0, 0.02, n), flat(X[y, x], Z[y, x], r.uniform(0, np.pi, n))), w_,
        mixed((YELLOW, WARM, SUNGRASS), r.choice(3, n, p=[0.4, 0.3, 0.3]), r.uniform(0.4, 0.85, n) - 0.35 * shade[y, x], r,
              np.concatenate([ACCENT[:1], BANK[1:]]), 0.08),
        thick=0.26, spent=0.2, pickup=0.3, land=0.4, ends=(0.3, 0.2), taper=0.3, merge=1.5)

    # the shadows of the trunks, laid over it all in long cool strokes, broken on the leaves
    fall = FALL / np.hypot(*FALL)
    paths, ws = [], []
    for X0, Z0, tall, girth, *_ in GROVE:
        G = np.array([X0, Z0]) + np.arange(0, 2.1 * tall, 0.25)[:, None] * fall
        G = G[G[:, 1] > 15]
        line = np.stack(project(G[:, 0], G[:, 1]), 1)
        for m in pieces(line, 110, -8, r, keep=0.6):
            paths.append(line[m])
            ws.append(0.45 * F * girth / G[m, 1].mean())
    n = len(paths)
    lay(paths, np.array(ws) * r.uniform(0.8, 1.2, n),
        mixed((SHADOW, GRASS), (r.random(n) < 0.3).astype(int), r.uniform(0.2, 0.5, n), r, SKY[1:3], 0.2),
        thick=0.16, pickup=0.3, merge=2, hide=0.8, dry=r.uniform(0.3, 0.6, n), taper=0.1, ends=(0.2, 0.2))

    # the banks, where the grass breaks off over the water in dark earth
    lip = water ^ ndimage.binary_erosion(water, iterations=2)
    gy, gx = np.gradient(ndimage.gaussian_filter(inside, 3))
    along = np.arctan2(gx, -gy).astype(np.float32)
    P = strew(0.6 * ndimage.binary_dilation(lip, iterations=2), 7)
    n = len(P)
    sc = 0.4 + 1.4 * near[at(P)]
    lay(go(along, P, r.uniform(8, 20, n) * sc, 4, 0, r.normal(0, 0.15, n)), r.uniform(1.6, 3.2, n) * sc,
        mixed((BANK, GRASS, SUNGRASS), r.choice(3, n, p=[0.12, 0.38, 0.5]), r.uniform(0.35, 0.7, n), r),
        thick=0.24, pickup=0.3, merge=1.5)
    wet[:] = 0

    # the birches, each in one go: the leaves behind the trunk, the limbs and the twigs hanging
    # from them, the trunk in ribbons of white, cream and grey-violet with black marks across it,
    # and the leaves in front, in touches of unmixed colour with the sky between them
    quarter = lambda Q: (np.clip(Q[:, 1].astype(int) // 4, 0, H // 4 - 1), np.clip(Q[:, 0].astype(int) // 4, 0, W // 4 - 1))

    def crown_seeds(tr, spacing, stray=False):
        """Where the touches of a crown go, as thick as its leaves hang; or the strays, a few
        single leaves out in its thin edges."""
        q = np.argwhere(tr["dens"] > 0.01)
        (y0, x0), (y1, x1) = np.maximum(q.min(0) * 4 - 16, 0), np.minimum(q.max(0) * 4 + 20, [H, W])
        P = dabs.scatter((y1 - y0, x1 - x0), spacing, r) + [x0, y0]
        d = tr["dens"][quarter(P)]
        keep = r.random(len(P)) < (np.where((d > 0.004) & (d < 0.15), 0.25, 0) if stray else d ** 0.8)
        return P[keep]

    def touches(tr, P, size, copper, darker, hue, olive=0.03):
        """Leaves: short loaded touches every which way, of many sizes, lemon where the crown
        turns to the sun and amber, copper and umber in its depths."""
        n = len(P)
        if not n:
            return
        o, o2 = 0.07 * tr["tall"] * SUN, 0.025 * tr["tall"] * SUN
        lit = np.tanh(3 * (tr["env"][quarter(P - o)] - tr["env"][quarter(P + o)])) \
            + 0.4 * (tr["dens"][quarter(P - o2)] - tr["dens"][quarter(P + o2)])
        tone = 0.56 + 0.32 * lit + 0.1 * noisy(P) - darker + 0.08 * r.normal(0, 1, n)
        warm = (r.random(n) < np.clip(copper + 0.35 * hue[at(P)] + 0.25 * (tone < 0.4), 0.03, 0.97)).astype(int)
        warm[r.random(n) < olive * (1.5 - tone)] = 2                  # a few leaves have not turned
        s = np.clip(size * np.exp(r.normal(0, 0.35, n)), 0.45 * size, 2.3 * size)
        L = s * r.uniform(1.0, 2.6, n)
        a = np.pi / 2 + r.normal(0, 0.9, n)
        d = np.stack([np.cos(a), np.sin(a)], 1)
        bow = (r.normal(0, 0.25, n) * s)[:, None] * np.stack([-d[:, 1], d[:, 0]], 1)
        paths = np.stack([P - d * L[:, None] / 2, P + bow, P + d * L[:, None] / 2], 1)
        cols, share = mixed((YELLOW, WARM, OLIVE), warm, tone, r, ACCENT, 0.12, spread=0.08)
        haze = np.clip((tr["Z"] - 50) / 500, 0, 0.25)
        lay(paths, s, (cols * (1 - haze) + haze * SKY[5], share),
            thick=0.42, spent=0.3, grooves=0.5, lips=0.18, land=0.7, lift=0.45, tails=0.4, pickup=0.15, merge=1.5,
            ends=(0.3, 0.15), taper=0.25, fray=1.0)

    def bark(tr):
        """The trunk in ribbons of paint dragged up and down it, its limbs and twigs, and the black."""
        A = tr["trunk"][: int(0.93 * len(tr["trunk"]))]                  # the leader ends among the leaves
        d = np.gradient(A[:, :2], axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        nrm *= np.sign(nrm[:, :1] + 1e-9)                           # toward the shaded side, on the right
        hw, hw_max = A[:, 2], A[:, 2].max()
        far_off = tr["tall"] < 450
        twigs = [(P[:3, :2], max(1.0, P[:3, 2].mean()), 0) for P in tr["limbs"]] + \
                [(P[2:4, :2], max(0.9, P[2:4, 2].mean()), 1) for P in tr["limbs"]] + \
                [(S[:3], max(0.7, 0.0004 * tr["tall"]), 1) for S in tr["strands"] if r.random() < 0.06]
        n = len(twigs)
        dark = np.array([t[2] for t in twigs])
        lay([t[0] for t in twigs], np.array([t[1] for t in twigs]) * r.uniform(0.9, 1.2, n),
            mixed((BARK, TWIG), dark, np.where(dark, 0.45, 0.6), r),
            thick=0.25, spent=0.6, pickup=0.2, merge=1, taper=0.5, ends=(0.2, 0.5))
        k = 3 if hw_max > 5 else 2 if hw_max > 2 else 1
        paths, ws, tones = [], [], []
        for j in range(k):
            c = (2 * j + 1) / k - 1
            f = np.clip(0.72 * c + 0.1 * np.sin(np.linspace(0, r.uniform(3, 9), len(A)) + r.uniform(0, 6)), -0.9, 0.9)
            line = A[:, :2] + nrm * (hw * f)[:, None]
            for m in pieces(line, 50 + 6 * hw_max, 10, r, keep=0.6 if far_off else 1.0):
                paths.append(line[m] if r.random() < 0.5 else line[m][::-1])
                ws.append(hw[m].mean() * 1.3 / k)
                tones.append(0.9 - 0.8 * ((c + 1) / 2) ** 1.2 - 0.2 * far_off)     # far off, the white is quieter
        n = len(paths)
        lay(paths, np.array(ws) * r.uniform(0.9, 1.15, n),
            loads(BARK, np.array(tones) + r.normal(0, 0.09, n), r, np.concatenate([YELLOW[4:6], WARM[5:], SKY[3:5]]),
                  0.2),
            thick=0.3, spent=0.55, pickup=0.25, land=0.3, merge=1.5, ends=(0.3, 0.3))
        if far_off:
            return
        edge = A[:, :2] + nrm * (0.95 * hw)[:, None]                    # the shadow side drawn down in grey-violet
        cut = pieces(edge, 60 + 5 * hw_max, -10, r, keep=0.55)
        n = len(cut)
        lay([edge[m] for m in cut], np.array([0.14 * hw[m].mean() + 0.6 for m in cut]),
            loads(np.concatenate([BARK[:2], BLACK[3:]]), r.uniform(0.2, 0.8, n), r), thick=0.2, pickup=0.3, merge=1)
        marks, mw = [], []
        for _ in range(int(2.6 * len(A) / (5 + 1.1 * hw_max))):        # the lenticels, thin dashes across the bark
            j = int(0.92 * r.random() ** 1.3 * (len(A) - 1))
            f0 = r.uniform(-1, 1)
            fs = np.clip(f0 + np.array([-0.5, 0, 0.5]) * r.uniform(0.25, 0.9), -1, 1)
            marks.append(A[j, :2] + nrm[j] * (hw[j] * fs)[:, None] + d[j] * r.normal(0, 0.06 * hw[j], 3)[:, None])
            mw.append(r.uniform(0.5, 1.3) * max(1, hw[j] / 9) ** 0.6)
        for b in 0.8 * r.random(r.integers(5, 10)) ** 1.3:             # and here and there the black, in a band
            for _ in range(r.integers(2, 5)):
                j = int(np.clip(b + r.normal(0, 0.012), 0, 1) * (len(A) - 1))
                f0 = r.choice([-1, 1], p=[0.35, 0.65]) * r.uniform(0.6, 1.0)
                fs = np.linspace(f0, f0 - np.sign(f0) * r.uniform(0.4, 1.3), 3)
                marks.append(A[j, :2] + nrm[j] * (hw[j] * fs)[:, None] + d[j] * r.normal(0, 0.1 * hw[j], 3)[:, None])
                mw.append(r.uniform(0.15, 0.32) * hw[j] + 0.6)
        for _ in range(r.integers(4, 9)):                               # the foot, black and fissured
            j = int(r.uniform(0, 0.06) * (len(A) - 1))
            c = r.uniform(-0.8, 0.8)
            marks.append(A[j, :2] + nrm[j] * hw[j] * (c + r.normal(0, 0.15, 3))[:, None]
                         + d[j] * hw[j] * np.array([[0.0], [1.0], [2.0]]) * r.uniform(0.5, 1.3))
            mw.append(r.uniform(0.12, 0.28) * hw[j] + 0.5)
        for P in tr["limbs"]:                                          # under each limb, the dark it leaves
            j = int(np.argmin(np.hypot(*(A[:, :2] - P[0, :2]).T)))
            side = np.sign((P[-1, :2] - P[0, :2]) @ nrm[j])
            marks.append(np.stack([P[0, :2] + side * nrm[j] * 0.95 * hw[j] + 0.3 * hw[j] * d[j],
                                   P[0, :2] + side * nrm[j] * 0.5 * hw[j] - 0.3 * hw[j] * d[j],
                                   P[0, :2] + side * nrm[j] * 0.15 * hw[j] - 0.7 * hw[j] * d[j]]))
            mw.append(max(0.9, 0.22 * hw[j]))
        n = len(marks)
        lay(marks, np.array(mw), loads(BLACK, r.uniform(0.2, 0.7, n), r, TWIG[1:3], 0.2),
            thick=0.3, spent=0.5, pickup=0.35, merge=1, ends=(0.4, 0.2), taper=0.4, fray=0.8)
        m = int(14 + 3 * hw_max)                                        # and the grass grown up over its foot
        P = np.stack([tr["x"] + r.normal(0, 1.8 * hw_max, m), tr["y"] + r.uniform(-2.5, 0.8, m) * hw_max], 1)
        sc = near[at(P)]
        lay(go(upright, P, r.uniform(10, 30, m) * (0.3 + sc), 5, r.normal(0, 0.004, m), r.normal(0, 0.3, m)),
            r.uniform(2, 4.5, m) * (0.4 + 0.6 * sc),
            mixed((GRASS, SUNGRASS), r.integers(0, 2, m), r.uniform(0.35, 0.6, m), r),
            thick=0.24, pickup=0.3, merge=1.5)

    hue = noise.field((H, W), 120, r)
    plants = [(tr, c[3]) for tr, c in zip(far_trees, FAR)] + [(bush(*b[:4], r), b[4]) for b in BUSHES] + \
             [(stand(X0, Z0, t, g, lean, r), copper) for X0, Z0, t, g, lean, copper in GROVE]
    for tr, copper in sorted(plants, key=lambda p: -p[0]["Z"]):    # the farthest first
        size = 6.5 * np.clip(30 / tr["Z"], 0.2, 1.1) ** 0.6
        if tr["trunk"] is None:                                       # a bush: dark at the root, lit on top
            P = crown_seeds(tr, 1.2 * size)
            rise = np.clip((tr["y"] - P[:, 1]) / tr["tall"], 0, 1)
            touches(tr, P, 0.85 * size, copper, 0.3 - 0.4 * rise, hue, olive=0.35)
            continue
        P = crown_seeds(tr, 1.6 * size)
        back = r.random(len(P)) < 0.38
        touches(tr, P[back], size, copper, 0.15, hue)
        bark(tr)
        A, front = tr["trunk"], P[~back]
        off = np.abs(front[:, 0] - np.interp(front[:, 1], A[::-1, 1], A[::-1, 0]))
        hidden = off < 1.5 * np.interp(front[:, 1], A[::-1, 1], A[::-1, 2]) + size     # the trunk shows through the crown
        touches(tr, front[~hidden | (r.random(len(front)) < 0.45)], size, copper, 0.0, hue)
        touches(tr, crown_seeds(tr, 3 * size, stray=True), 0.7 * size, copper, 0.0, hue)

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.72, 1.2))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
