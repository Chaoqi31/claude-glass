"""The Footbridge, Green Harmony. Oil on canvas, in dense layers.

In 1893 Monet bought the marshy meadow across the road from his garden at Giverny, dug a
pond there fed from a branch of the Epte, planted it with water lilies and threw over its
narrow end a wooden footbridge in the Japanese manner, painted green. In the summer of 1899
he painted it a dozen times from the same place on the bank, most of the canvases nearly
square: the bridge arched across the top and cut by both edges, willows and bamboo crowding
behind it, and below it the pond, the lilies lying on it in level drifts and the dark water
between them full of reflected green. Here the foliage goes on in dense strokes of many
greens, sitting over sitting; the reflections are pulled straight down, the pads put in
short level touches and the flowers in a few thick ones; and the bridge is drawn last, over
all of it, in a few loaded strokes, a little crooked, as wood painted by hand is.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Footbridge, Green Harmony"
DATE = "2026"
MEDIUM = ("Oil on canvas, in dense layers of greens and blues: the pads in short level strokes, the reflections "
          "drawn down between them, the bridge in a few loaded strokes")
AFTER = "Claude Monet, the Water Lily Pond with the Japanese footbridge, Giverny, 1899"
ROOM = "The Garden"
YEAR = 1899
PLACE = "Giverny"
REGION = "Europe"
NOTE = ("The green wooden footbridge over Monet's lily pond, willows and bamboo crowding behind it and the "
        "lilies lying on the water below. The bridge went on last, in a few loaded strokes over the leaves.")

H, W = 2280, 2400
LIGHT = (-0.6, -0.5, 0.62)
HORIZON, FOCAL = 690.0, 3900.0         # the eye's level, hidden in the leaves; how steeply the pond falls away from it
K = H - HORIZON
BANK = 1120.0                          # the far edge of the water across the middle
CREST, SAG = (1340.0, 762.0), 1.0e-4   # the top of the deck at its crest; how fast it falls toward the ends
RAIL, MID, FASCIA = 276.0, 142.0, 66.0  # the top rail and the middle rail above the deck, and the depth of the deck
POSTS = (150.0, 615.0, 1098.0, 1588.0, 2056.0)
BANDS = (1178, 1250, 1340, 1455, 1605, 1795, 2025, 2275)    # where the drifts of lilies cross the middle
Z0, Z1 = FOCAL * K / (H + 40 - HORIZON), FOCAL * K / (BANK - 70 - HORIZON)   # depth of the nearest water and the far bank
X1 = (W / 2 + 150) * Z1 / FOCAL
STEP = 20.0                            # the grid laid over the pond as seen from above, in px of its near edge


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
DEEP = pal("#16221e", "#1c2b25", "#22352d", "#2a4036", "#334b3f", "#3d5748")                # viridian and black
GREEN = pal("#2f4a3a", "#395843", "#44654b", "#507253", "#5e7f5d", "#6f8d68")               # emerald, chrome green
SAP = pal("#67703f", "#78824a", "#8a9355", "#9ca462", "#adb372", "#bec285", "#cdcf99")      # yellow-green, cadmium
BLUE = pal("#24303f", "#2d3b4d", "#37475b", "#45566a", "#56677a", "#6a7b8c")                # cobalt, ultramarine
OLIVE = pal("#4d4c30", "#5d5b38", "#6e6a42", "#807a4e", "#918a5e")                          # earth green, ochre
WARM = pal("#7a4c3a", "#93604a", "#a87a5c", "#b98f72", "#c9a586")                           # burnt sienna let down
ROSE = pal("#8a4c5a", "#a2636e", "#b97f86", "#cc9ba0", "#dbb6b6")                           # madder
VIOLET = pal("#3a3452", "#4a4266", "#5c5379", "#6f668b")                                    # the irises
CHINK = pal("#9fb0b8", "#b7c4c6", "#cdd6d2")                                                # sky through the leaves
FAMILIES = (DEEP, GREEN, SAP, BLUE, OLIVE, WARM, ROSE, VIOLET, CHINK)
BRIDGE = pal("#2f5450", "#38625d", "#43706a", "#4f7e77", "#5d8c84", "#6e9b92", "#82aaa0", "#99bab0", "#b2cbc1",
             "#cadcd2")                                                                     # viridian, cobalt, white
WATER = pal("#1f2b25", "#26352d", "#2e4035", "#374b3e", "#415648")                          # the dark of the pond
PADS = pal("#3f5240", "#4b6049", "#587053", "#667e5e", "#768d6a", "#889b79", "#9caa8b", "#b2bda0")
SHEEN = pal("#6f8580", "#7f938e", "#93a59f", "#a8b8b0", "#bccac1", "#cfdad1")               # pads holding the sky
LILAC = pal("#6c6874", "#807b88", "#958f9c", "#aaa4b0", "#bfb9c3")
WINE = pal("#4f3634", "#634440", "#77524c", "#8a6258")                                      # young leaves
PETALS = (pal("#b9b1aa", "#d2ccc3", "#e6e1d7", "#f3efe6"), pal("#b17880", "#c78f94", "#dbaaaa", "#ebc6c0"),
          pal("#b39545", "#cbb057", "#dfc96f"))                                             # white, pink, yellow
DIM = np.float32([0.56, 0.63, 0.7])   # what the water gives back of the green above it


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette, for a whole field of tones."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(t.astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone (0 dark, 1 light), a paint next to it on
    the palette streaked in, and a third, now and then an accent picked up from elsewhere."""
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
    for i, colours in enumerate(families):
        sel = which == i
        if sel.any():
            cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds, spread)
    return cols, share


BOW = np.random.default_rng(1899).normal(0, 3, 16)


def deck(x):
    """The top of the deck at x: highest a little right of the middle, falling away to both ends,
    not quite true, as a carpenter builds and a painter draws."""
    x = np.asarray(x, np.float32)
    d = x - CREST[0]
    return CREST[1] + SAG * d * d * (1 - 0.1 * d / W) + np.interp(x, np.linspace(-200, W + 200, 16), BOW)


def shore(r):
    """The far edge of the water, y for each x: low across the middle and coming forward on both
    sides, where the banks close in."""
    x = np.arange(W, dtype=np.float32)
    return (BANK + 20 * noise.line1d(W, 300, r) + 6 * noise.line1d(W, 45, r)
            + 330 * noise.smoothstep(480, 0, x) ** 1.5 + 170 * noise.smoothstep(2020, 2400, x) ** 1.4).astype(np.float32)


def seen(X, Z):
    """Where a point of the pond (X across, Z deep, both in px of its near edge) falls in the picture."""
    s = FOCAL / Z
    return W / 2 + X * s, HORIZON + K * s


def drifts(r):
    """How thickly the lilies grow over the pond, on a grid laid over it as seen from above:
    level drifts across it, each a little ragged and the far ones broken off, and an open
    channel winding back through them. -> a sampler (X, Z) -> density"""
    gz, gx = int((Z1 - Z0) / STEP) + 2, int(2 * X1 / STEP) + 2
    Z = Z0 + STEP * np.arange(gz, dtype=np.float32)[:, None]
    X = -X1 + STEP * np.arange(gx, dtype=np.float32)[None]
    zb = FOCAL * K / (np.float32(BANDS) - HORIZON)
    gap = np.abs(np.gradient(zb))
    wav = noise.field((gz, gx), 22, r)
    D = np.zeros((gz, gx), np.float32)
    for yb, z, g in zip(BANDS, zb, gap):
        reach = 1.0 if yb > 1900 else noise.smoothstep(-1.4, -0.3, noise.line1d(gx, 70, r))[None]
        D = np.maximum(D, reach * np.exp(-((Z - z + 0.35 * g * wav) / (0.36 * g)) ** 2))
    far = (1 - FOCAL / Z) / 0.63                                            # 0 at the near edge, 1 at the far bank
    D *= 1 - 0.7 * np.exp(-((X + 300 - 170 * far - 120 * wav) / (190 + 90 * far)) ** 2)
    D += 0.16 * noise.field((gz, gx), 6, r)
    return lambda X_, Z_: ndimage.map_coordinates(D, [(Z_ - Z0) / STEP, (X_ + X1) / STEP], order=1, mode="nearest")


def pads(r, grows, edge):
    """Where each lily pad lies (x, y, half-width, half-depth): seeds strewn evenly over the pond
    as seen from above and kept where the drifts grow, then seen from the bank, smaller and
    flatter as they go back. The far ones come first."""
    P = dabs.scatter((int(Z1 - Z0) + 1, int(2 * X1) + 1), 88, r) + np.float32([-X1, Z0])
    P = P[grows(P[:, 0], P[:, 1]) > 0.5 + r.normal(0, 0.06, len(P))]
    x, y = seen(P[:, 0], P[:, 1])
    rx = 84 * FOCAL / P[:, 1] * np.exp(r.normal(0, 0.3, len(P)))
    ry = rx * (y - HORIZON) / FOCAL * r.uniform(0.85, 1.15, len(P))
    ok = (x > -rx) & (x < W + rx) & (y < H + ry) & (y > edge[np.clip(x.astype(int), 0, W - 1)] + ry)
    out = np.column_stack([x, y, rx, ry])[ok].astype(np.float32)
    return out[np.argsort(out[:, 1])]


def leaf(pad, r):
    """One pad in one to five short level touches laid one under another from its far edge to its
    near one, bowed a little to its oval and never quite filling it, so the water shows at its
    edges. -> paths, half-widths, where each lies (v)"""
    cx, cy, rx, ry = pad
    m = int(np.clip(1 + ry / 6, 1, 5))
    v = -0.75 + 1.5 * (np.arange(m) + r.uniform(0.25, 0.75, m)) / m
    q = np.linspace(-1, 1, 5)
    paths = []
    for vk in v:
        span = rx * np.sqrt(1 - vk * vk)
        reach = r.uniform(0.55, 1.0) * span
        x = cx + r.uniform(-0.3, 0.3) * span + q * reach
        y = cy + vk * ry + r.normal(0, 0.04) * q * reach - vk * 0.3 * ry * q * q
        paths.append(np.stack([x, y], 1)[::r.choice([-1, 1])])
    return paths, np.clip(ry / m * r.uniform(0.85, 1.3, m), 1.3, 15), v


def seat(pad, r):
    """The dark of the water just under a pad's near edge, where it sits in the pond."""
    cx, cy, rx, ry = pad
    q = np.linspace(-1, 1, 5)
    return np.stack([cx + q * rx * r.uniform(0.55, 0.85), cy + ry * (0.75 - 0.25 * q * q) + 1.5], 1)


def flower(pad, r):
    """A lily on its pad, a little heap of thick paint: two petals spread on the water, two or
    three short strokes for the cup standing up from them, and now and then a touch of yellow at
    its heart. -> petal paths, half-widths, how high on the flower each lies; heart paths, half-widths"""
    cx, cy, rx, ry = pad
    f = 0.34 * rx
    x0, y0 = cx + r.normal(0, 0.2 * rx), cy - 0.2 * ry
    q = np.linspace(0, 1, 4)
    petals, lift = [], []
    for side in (-1, 1):
        a = (0 if side > 0 else np.pi) + r.normal(0, 0.2) - side * 0.15
        l = f * r.uniform(0.6, 1.0)
        petals.append(np.stack([x0 + side * 0.1 * f + q * l * np.cos(a), y0 + q * l * np.sin(a)], 1))
        lift.append(0.0)
    for _ in range(r.integers(2, 4)):
        a = -np.pi / 2 + r.normal(0, 0.6)
        b = np.array([x0 + r.normal(0, 0.25 * f), y0 - 0.1 * f])
        petals.append(b + q[:, None] * f * r.uniform(0.35, 0.7) * np.array([np.cos(a), np.sin(a)]))
        lift.append(r.uniform(0.4, 1))
    hearts = [np.array([[x0 - 0.1 * f, y0 - 0.35 * f], [x0 + 0.1 * f, y0 - 0.4 * f]]) + r.normal(0, 0.08 * f, 2)] \
        if r.random() < 0.5 else []
    return petals, np.maximum(1.6, f * r.uniform(0.22, 0.32, len(petals))), lift, hearts, \
        np.full(len(hearts), max(1.6, 0.12 * f))


def spans(a, b, r, reach=(300, 700), lap=(30, 110)):
    """How a long line is laid: stroke after stroke, each as far as one loading of the brush
    carries, each overlapping the last (or, with a negative lap, leaving a gap)."""
    out, x = [], a
    while True:
        e = min(b, x + r.uniform(*reach))
        out.append((x, e))
        if e >= b:
            return out
        x = e - r.uniform(*lap)


def along(rise, x0, x1, r, wob=3.5):
    """A stroke along the bridge from x0 to x1, `rise` px above the deck, wavering as a hand does."""
    n = max(4, int(abs(x1 - x0) / 50))
    x = np.linspace(x0, x1, n) + r.normal(0, 3, n)
    k = max(2, int(abs(x1 - x0) / 150) + 2)
    sway = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, k), r.normal(0, 1, k))
    return np.stack([x, deck(x) - rise + wob * sway + r.normal(0, 1.5) + r.normal(0, 0.6, n)], 1)


def post(xp, r, top=RAIL + 18, foot=-6):
    """A post in one stroke pulled down from above the top rail to the deck, leaning a little."""
    y = np.linspace(deck(xp) - top, deck(xp) - foot, 6)
    return np.stack([xp + r.normal(0, 3) * np.linspace(0, 1, 6) + r.normal(0, 0.8, 6), y], 1)


def paint(seed=1899):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#dcd4c2", thread=3.0)
    edge = shore(r)
    water = yy > edge[None]
    grows = drifts(r)
    ys, xs = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    s4 = np.maximum(ys - HORIZON, 1) / K
    drift = ndimage.zoom(grows((xs - W / 2) / s4, FOCAL / s4), 4, order=1)[:H, :W] * water
    near = np.clip((yy - HORIZON) / K, 0, 1)                                   # 1 at the near edge of the pond

    # the places: trees behind, a willow on the left, bamboo on the right, reeds and irises along the banks
    rows, cols = np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32)
    xw = (900 - 260 * noise.smoothstep(250, 1000, rows) + 50 * noise.line1d(H, 250, r))[:, None]
    tip = (1010 - 200 * ((cols - 280) / 700) ** 2 + 70 * noise.line1d(W, 40, r))[None]
    willow = noise.smoothstep(60, -60, xx - xw) * noise.smoothstep(tip + 30, tip - 60, yy)
    bamboo = noise.smoothstep(-70, 70, xx - 1970 - 60 * noise.line1d(H, 300, r)[:, None]) * noise.smoothstep(1080, 880, yy)
    reach = (170 + 50 * noise.line1d(W, 120, r) + 360 * noise.smoothstep(560, 80, cols)
             + 160 * noise.smoothstep(1960, 2380, cols))[None]
    reeds = noise.smoothstep(-25, 25, yy - (edge[None] - reach - 30 * noise.field((H, W), 30, r)))
    kind = np.select([water, reeds > 0.5, willow > 0.5, bamboo > 0.5], [4, 3, 1, 2], 0).astype(np.int8)

    # the light on the leaves: the trees in rough domes lit from the upper left, the willow in
    # hanging cascades, the reeds lit at their tips and in their own shade at the foot
    hang = 0.7 * noise.stretched((H, W), 140, 900, r) + 0.5 * noise.stretched((H, W), 40, 500, r)
    dome = np.where(kind == 1, hang, noise.fbm((H, W), 240, r, octaves=4))
    gy, gx = np.gradient(ndimage.gaussian_filter(dome, 2) * 50)
    Lv = np.float32(LIGHT) / np.linalg.norm(LIGHT)
    lit = (-gx * Lv[0] - gy * Lv[1] + Lv[2]) / np.sqrt(gx * gx + gy * gy + 1)
    lit = (lit - lit.mean()) / lit.std()
    foot = noise.smoothstep(edge[None] - 80, edge[None], yy)
    light = (0.44 + 0.11 * dome + 0.12 * lit + 0.05 * noise.fbm((H, W), 40, r, octaves=2) + 0.08 * (0.5 - xx / W)
             - 0.1 * noise.smoothstep(0.3, 1.5, noise.field((H, W), 500, r)))
    light += np.select([kind == 1, kind == 2, kind == 3],
                       [0.04 + 0.06 * noise.stretched((H, W), 10, 300, r) - 0.12 * noise.smoothstep(500, 1000, yy), 0.04,
                        0.08 - 0.3 * foot + 0.16 * noise.field((H, W), 180, r)], -0.05)
    light = light.astype(np.float32)
    ang = np.pi / 2 + 0.25 * noise.field((H, W), 500, r)                        # upright, more or less
    ang = np.where(kind == 1, np.pi / 2 + 0.05 * noise.field((H, W), 400, r), ang)
    ang = np.where(kind == 2, np.pi / 2 + 0.2 * noise.field((H, W), 200, r), ang)
    ang = np.where(kind == 3, np.pi / 2 + 0.25 * noise.field((H, W), 120, r), ang)
    ang = np.where(water, np.where(drift > 0.5, 0, np.pi / 2), ang).astype(np.float32)
    level = (0.03 * noise.field((H, W), 400, r)).astype(np.float32)
    fall = (np.pi / 2 + 0.03 * noise.field((H, W), 300, r)).astype(np.float32)

    # the design, as the colours add up across the room: the leaves by their light, the water
    # giving them back darker, the drifts a dull green under the pads
    want = ramp(np.concatenate([DEEP, GREEN, SAP]), light)
    mirror = np.clip(2 * (edge[None] + 25) - yy, 0, edge[None] - 2).astype(int)
    want = np.where(water[..., None], want[mirror, xx.astype(int)] * DIM, want)
    want += (ramp(PADS, 0.15 + 0.25 * near) - want) * (0.8 * noise.smoothstep(0.3, 0.8, drift))[..., None]

    rag = noise.stretched((H, W), 14, 300, r)
    alpha = np.clip(0.86 + 0.05 * rag, 0, 1)[..., None]
    rgb = (ground.color * (1 - alpha) + want * (1 + 0.04 * rag[..., None]) * alpha).astype(np.float32)
    height = ground.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def go(P, L, n=6, bend=0.0, tilt=0.0, flip=0.5, field=None):
        paths = impasto.follow(ang if field is None else field, P, L, n, bend, tilt)
        back = r.random(len(P)) < flip
        paths[back] = paths[back, ::-1]
        return paths

    def leafage(P, lift=0.0, wander=0.1, blue=0.2):
        """The paint for a stroke of leaves at each seed: the family for its place and light
        (0 deep, 1 green, 2 sap, 3 cobalt, 4 earth green), and the tone in it."""
        y, x = at(P)
        n, k = len(P), kind[y, x]
        l = light[y, x] + lift + r.normal(0, wander, n)
        fam = np.digitize(l, [0.36, 0.62])
        tone = np.clip(np.select([fam == 0, fam == 1], [l / 0.36, (l - 0.36) / 0.26], (l - 0.62) / 0.38), 0, 1)
        u = r.random(n)
        fam[(fam == 0) & (u < blue)] = 3
        tone[fam == 3] *= 0.7
        fam[(fam == 1) & (u > 0.84) & (k != 1)] = 4
        fam[(k == 3) & (u > 0.72) & (fam != 0)] = 4
        return fam, tone

    def mirrored(P):
        """The place in the leaves a stroke of water gives back, straight above it across the far edge."""
        x = at(P)[1]
        return np.column_stack([P[:, 0], np.clip(2 * (edge[x] + 25) - P[:, 1], 0, edge[x] - 2)])

    # the first sitting, into the wet wash: everything laid in broad and fluid
    P = strew(150, np.ones((H, W)))
    n = len(P)
    T = want[at(P)]
    lay(go(P, r.uniform(160, 340, n), 10, 0, r.normal(0, 0.15, n)), r.uniform(20, 32, n),
        (np.stack([T * np.exp(r.normal(0, 0.08, (n, 3))) for _ in range(3)], 1), None),
        thick=0.015, grooves=0.1, lips=0.02, land=0.1, lift=0.05, pickup=0.7, merge=8, spent=0.85, hide=0.6,
        fray=4, taper=0.5, tails=0.5, ends=(0.7, 0.6))
    height -= 0.85 * (height - 0.2 * ground.tooth)
    wet[:] = 0

    # the second, wet into wet: each thing in its own greens. The trees in short broad touches,
    # broadest in the shadows; the willow pulled down in long strokes; the bamboo and reeds drawn up
    soft = dict(spent=0.6, pickup=0.5, merge=2, grooves=0.6, fray=2.2, taper=0.3, ends=(0.4, 0.3))
    accent = np.concatenate([BLUE[2:], SAP[2:5], OLIVE[2:]])
    P = strew(24, kind == 0)
    n = len(P)
    fam, tone = leafage(P)
    g = np.exp(r.normal(0, 0.3, n)) * (1.35 - 0.7 * np.clip(light[at(P)], 0, 1))
    lay(go(P, r.uniform(22, 50, n) * g, 4, 0, r.normal(0, 0.5, n)), r.uniform(7, 13, n) * g,
        mixed(FAMILIES, fam, tone, r, accent, 0.15), thick=0.18, dry=r.uniform(0, 0.3, n), **soft)
    P = strew(17, kind == 1)
    n = len(P)
    fam, tone = leafage(P)
    L = np.minimum(r.uniform(60, 170, n), tip[0, at(P)[1]] + 40 - P[:, 1])
    lay(go(P, np.maximum(L, 20), 8, r.normal(0, 0.001, n), r.normal(0, 0.03, n), flip=0), r.uniform(3.5, 7, n),
        mixed(FAMILIES, fam, tone, r, accent, 0.15), thick=0.15, dry=r.uniform(0, 0.35, n),
        **{**soft, "spent": 0.75, "taper": 0.6, "ends": (0.3, 0.6)})
    P = strew(16, kind == 2)
    n = len(P)
    fam, tone = leafage(P)
    flick = r.random(n) < 0.5                                                   # leaves, against the canes
    lay(go(P, np.where(flick, r.uniform(20, 45, n), r.uniform(50, 130, n)), 5, 0,
           np.where(flick, r.choice([-0.7, 0.7], n) + r.normal(0, 0.2, n), r.normal(0, 0.06, n))),
        np.where(flick, r.uniform(3, 6, n), r.uniform(3, 5.5, n)), mixed(FAMILIES, fam, tone, r, accent, 0.15),
        thick=0.16, dry=r.uniform(0, 0.3, n), **soft)
    P = strew(15, kind == 3)
    n = len(P)
    fam, tone = leafage(P)
    lay(go(P, r.uniform(30, 80, n), 5, r.normal(0, 0.004, n), np.pi + r.normal(0, 0.25, n), flip=0),
        r.uniform(2.5, 5.5, n), mixed(FAMILIES, fam, tone, r, accent, 0.2), thick=0.16, dry=r.uniform(0, 0.3, n),
        **{**soft, "taper": 0.7, "ends": (0.2, 0.8)})

    # the water: the leaves given back, pulled straight down; and the drifts laid in, level and dull
    down = {**soft, "ends": (0.5, 0.5), "taper": 0.45, "fray": 2.8, "merge": 3}
    P = strew(26, water * noise.smoothstep(0.7, 0.3, drift))
    n = len(P)
    s = near[at(P)]
    fam, tone = leafage(mirrored(P), -0.08, blue=0.06)
    cols, share = mixed(FAMILIES, fam, tone, r, WATER[1:], 0.3)
    lay(go(P, r.uniform(40, 130, n) * (0.35 + 0.65 * s), 6, 0, r.normal(0, 0.03, n), field=fall),
        r.uniform(5, 11, n) * (0.45 + 0.65 * s), (cols * DIM, share), thick=0.12, dry=r.uniform(0, 0.35, n), **down)
    P = strew(15, drift * (0.25 + 0.75 * (1 - near)))
    n = len(P)
    s = near[at(P)]
    lay(go(P, (20 + 70 * s) * r.uniform(0.7, 1.3, n), 6, 0, r.normal(0, 0.04, n), field=level),
        (1.5 + 8 * s * s) * r.uniform(0.8, 1.2, n),
        mixed((PADS, GREEN, DEEP), r.choice(3, n, p=[0.55, 0.3, 0.15]), 0.2 + 0.25 * s + 0.1 * r.normal(0, 1, n), r),
        thick=0.16, dry=r.uniform(0, 0.3, n), **soft)
    P = strew(9, np.abs(yy - edge[None]) < 18)                                # the reeds' feet and their own dark in the water
    n = len(P)
    lay(go(P, r.uniform(18, 45, n), 4, 0, r.normal(0, 0.1, n), field=fall), r.uniform(3, 6, n),
        mixed((DEEP, GREEN, OLIVE), r.choice(3, n, p=[0.6, 0.3, 0.1]), r.uniform(0.2, 0.6, n), r), thick=0.12,
        dry=r.uniform(0, 0.3, n), **soft)
    wet[:] = 0
    height += 0.3 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2))

    # the third, over dry paint: lights and darks restated in the leaves, a few chinks of sky and
    # warm touches, the reflections again
    last = dict(pickup=0, merge=0, fray=2.2, taper=0.35, ends=(0.4, 0.3), spent=0.6, grooves=0.6)
    P = strew(21, kind < 3)
    n = len(P)
    fam, tone = leafage(P, 0.04, 0.16, blue=0.1)
    vert = kind[at(P)] == 1
    g = np.exp(r.normal(0, 0.3, n))
    lay(go(P, np.where(vert, r.uniform(40, 120, n), r.uniform(14, 34, n)) * g, 5, 0,
           np.where(vert, r.normal(0, 0.03, n), r.normal(0, 0.55, n)), flip=0.5 * ~vert),
        np.where(vert, r.uniform(2.5, 5, n), r.uniform(4, 8, n)) * g, mixed(FAMILIES, fam, tone, r, accent, 0.2),
        thick=0.2, dry=r.uniform(0.1, 0.5, n), **last)
    P = strew(18, kind == 3)
    n = len(P)
    fam, tone = leafage(P, 0.05, 0.14)
    hit = r.random(n)
    fam[hit < 0.007], fam[(hit > 0.007) & (hit < 0.014)] = 6, 5
    fam[(hit > 0.02) & (hit < 0.1) & (P[:, 0] < 520)] = 7                      # irises on the left bank
    lay(go(P, r.uniform(25, 70, n), 5, r.normal(0, 0.004, n), np.pi + r.normal(0, 0.3, n), flip=0), r.uniform(2, 4.5, n),
        mixed(FAMILIES, fam, tone, r), thick=0.2, dry=r.uniform(0.1, 0.45, n), **{**last, "taper": 0.7, "ends": (0.2, 0.8)})
    c = np.column_stack([r.uniform(150, 2300, 9), np.zeros(9)])
    c[:, 1] = edge[c[:, 0].astype(int)] - r.uniform(30, 130, 9)
    P = np.concatenate([q + r.normal(0, [28, 12], (r.integers(4, 9), 2)) for q in c])  # flowers along the bank
    n = len(P)
    lay(go(P, r.uniform(6, 14, n), 3, 0, r.normal(0, 0.8, n)), r.uniform(2.5, 4.5, n),
        mixed((ROSE, WARM, PETALS[0]), r.choice(3, n, p=[0.5, 0.2, 0.3]), r.uniform(0.5, 1, n), r), thick=0.3,
        dry=r.uniform(0.1, 0.3, n), **last)
    P = strew(70, (yy < 520) * (kind != 1) * (kind < 3) * noise.smoothstep(0.55, 0.8, light))
    n = len(P)
    lay(go(P, r.uniform(8, 20, n), 4, 0, r.normal(0, 0.8, n)), r.uniform(2.5, 4.5, n),
        loads(CHINK, r.uniform(0.2, 0.9, n), r), thick=0.25, dry=r.uniform(0.1, 0.4, n), **last)
    P = strew(34, water * noise.smoothstep(0.6, 0.3, drift))
    n = len(P)
    s = near[at(P)]
    fam, tone = leafage(mirrored(P), 0.04, 0.15, blue=0.05)
    cols, share = mixed(FAMILIES, fam, tone, r, np.concatenate([BLUE[:3], VIOLET[:2]]), 0.15)
    lay(go(P, r.uniform(30, 100, n) * (0.35 + 0.65 * s), 6, 0, r.normal(0, 0.03, n), field=fall),
        r.uniform(4, 8, n) * (0.45 + 0.65 * s), (cols * DIM, share), thick=0.16, dry=r.uniform(0.25, 0.55, n),
        **{**last, "ends": (0.5, 0.5), "taper": 0.45, "fray": 2.8})
    wet[:] = 0

    # the pads, wet into wet: under the larger ones the dark they sit in, then each pad in a few
    # loaded level strokes, its far side catching the light of the sky; a young leaf here and there
    lily = pads(r, grows, edge)
    N = len(lily)
    big = lily[lily[:, 3] > 6]
    lay([seat(p, r) for p in big], 0.35 * big[:, 3] + 1.2,
        mixed((WATER, BLUE), (r.random(len(big)) < 0.2).astype(int), np.full(len(big), 0.3), r), thick=0.08, hide=0.6,
        dry=0.3, pickup=0.3)
    paths, ws, vs, owner = [], [], [], []
    for i, p in enumerate(lily):
        a, b, c = leaf(p, r)
        paths += a
        ws += list(b)
        vs += list(c)
        owner += [i] * len(a)
    owner, vs = np.array(owner), np.array(vs)
    d = near[at(lily[:, :2])[0], 0]
    glow = noise.field((H, W), 600, r)[at(lily[:, :2])]                           # the sky lies brighter on some drifts
    hue = noise.field((H, W), 500, r)[at(lily[:, :2])][owner]                     # lilac in some, yellow-green in others
    tone = (0.37 + 0.12 * d + 0.16 * glow + r.normal(0, 0.08, N))[owner] - 0.3 * vs + r.normal(0, 0.1, len(vs))
    u = r.random(len(vs))
    fam = np.where((vs < -0.25) & (u < 0.45), 1, np.where(u > 0.9 - 0.15 * (hue > 0.6), 2, 0))
    fam[(u < 0.3 * noise.smoothstep(-0.5, -1.2, hue)) & (vs > -0.3)] = 3
    lay(paths, np.array(ws), mixed((PADS, SHEEN, LILAC, SAP), fam, tone, r, np.concatenate([SHEEN[3:], LILAC[2:]]), 0.15),
        thick=0.36, spent=0.5, pickup=0.15, merge=0.5, grooves=1.0, dry=r.uniform(0.1, 0.4, len(paths)), fray=1.6,
        taper=0.25, ends=(0.35, 0.3))
    young = big[r.random(len(big)) < 0.02]
    lay([across[::r.choice([-1, 1])] for across in
         (np.stack([p[0] + np.linspace(-0.4, 0.4, 4) * p[2], np.full(4, p[1] + 0.35 * p[3])], 1) for p in young)],
        np.clip(0.3 * young[:, 3], 1.5, 8), loads(WINE, np.full(len(young), 0.5), r), thick=0.3, dry=0.3, pickup=0.2)
    wet[:] = 0

    # the flowers, a few heaps of thick paint on the nearer pads
    bloom = lily[(lily[:, 2] > 28) & (r.random(N) < 0.2 * noise.smoothstep(-0.3, 1.2, noise.field((H, W), 300, r)[at(lily[:, :2])])
                                         + 0.06 * d)]
    petals, pw, kinds, ups, hearts, hw = [], [], [], [], [], []
    for p in bloom:
        a, b, c, e, f = flower(p, r)
        petals += a
        pw += list(b)
        ups += list(c)
        kinds += [r.choice(3, p=[0.62, 0.3, 0.08])] * len(a)
        hearts += e
        hw += list(f)
    lay(petals, np.array(pw), mixed(PETALS, np.array(kinds), 0.25 + 0.6 * np.array(ups), r, PETALS[1][1:], 0.3),
        thick=0.45, spent=0.35, pickup=0.35, ends=(0.3, 0.2), dry=r.uniform(0.15, 0.4, len(petals)))
    if hearts:
        lay(hearts, np.array(hw), loads(PETALS[2], np.full(len(hearts), 0.5), r), thick=0.5, spent=0.3, pickup=0.4,
            dry=0.25)
    wet[:] = 0

    # the bridge, last and wet into wet: the shadow under the deck, the far railing seen through
    # the near one, the posts, the rails, the deck, and the light along their tops
    wood = dict(thick=0.5, spent=0.45, pickup=0.15, merge=0.6, grooves=1.0, fray=2.0, taper=0.15, ends=(0.3, 0.2))
    grain = np.concatenate([GREEN[3:], LILAC[1:3], BLUE[3:]])

    def line(rise, w, tone, reach=(250, 600), lap=(30, 110), dry=(0, 0.2), **kw):
        paths = [along(rise, a, b, r) for a, b in spans(-60, W + 60, r, reach, lap)]
        n = len(paths)
        lay(paths, w * r.uniform(0.8, 1.25, n), loads(BRIDGE, tone + r.normal(0, 0.08, n), r, grain, 0.25, 0.05),
            **{**wood, "dry": r.uniform(*dry, n), **kw})

    def posts(xs, w, tone, top=RAIL + 18, foot=-6, **kw):
        lay([post(x, r, top, foot) for x in xs], w * r.uniform(0.85, 1.15, len(xs)),
            loads(BRIDGE, tone + r.normal(0, 0.05, len(xs)), r, grain, 0.25, 0.05), **{**wood, **kw})

    xp = np.float32(POSTS) + r.normal(0, 5, len(POSTS))
    line(-FASCIA - 26, 28, 0.1, hide=0.45, thick=0.05, dry=(0.3, 0.5))
    line(RAIL - 38, 7, 0.3, hide=0.9)
    line(MID - 32, 6, 0.28, hide=0.9)
    posts(xp + 36 + r.normal(0, 3, len(xp)), 7, 0.3, RAIL - 30)
    posts(xp, 15, 0.46)
    posts(xp - 6, 7, 0.78, RAIL + 12, 0, dry=0.15)
    posts(xp + 10, 3.5, 0.12, RAIL + 10, 0, hide=0.7, dry=0.3)
    line(MID, 12, 0.44)
    line(MID + 5, 7, 0.76, (200, 520), (-40, 60))
    line(MID - 9, 3.5, 0.1, hide=0.7, dry=(0.2, 0.5))
    line(RAIL, 15, 0.46)
    line(RAIL + 6, 8, 0.8, (200, 520), (-40, 60))
    line(RAIL - 11, 3.5, 0.1, hide=0.7, dry=(0.2, 0.5))
    line(-FASCIA * 0.3, 19, 0.46, (200, 450))
    line(-FASCIA * 0.66, 18, 0.36, (200, 450))
    line(-FASCIA * 0.9, 8, 0.18, dry=(0.1, 0.4))
    line(-5, 9, 0.8, (200, 520), (-40, 60))
    wet[:] = 0

    # a few leaves over it: willow fronds hanging in front of its left end, reeds before the right
    P = strew(30, (xx < 360) * (yy > deck(xx) - RAIL - 200) * (yy < deck(xx) - 40))
    P = P[r.random(len(P)) < 0.25]
    n = len(P)
    fam, tone = leafage(P, 0.05)
    lay(go(P, r.uniform(80, 200, n), 8, r.normal(0, 0.002, n), r.normal(0, 0.05, n), flip=0, field=fall),
        r.uniform(2, 3.5, n), mixed(FAMILIES, fam, tone, r), thick=0.15, dry=r.uniform(0.35, 0.6, n),
        **{**last, "taper": 0.6})
    P = strew(30, (xx > 2150) * (yy > deck(xx) + 20) * (yy < deck(xx) + 140))
    P = P[r.random(len(P)) < 0.3]
    n = len(P)
    fam, tone = leafage(P, 0.05)
    lay(go(P, r.uniform(60, 140, n), 6, r.normal(0, 0.004, n), np.pi + r.normal(0, 0.25, n), flip=0),
        r.uniform(2, 3.5, n), mixed(FAMILIES, fam, tone, r), thick=0.15, dry=r.uniform(0.3, 0.55, n),
        **{**last, "taper": 0.7, "ends": (0.2, 0.8)})

    height = ndimage.gaussian_filter(height, 0.8)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.6, gloss=0, reach=(0.75, 1.15))
    return img + 0.04 * impasto.glints(height, LIGHT)[..., None]
