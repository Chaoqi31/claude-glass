"""The Lily Pond, Clouds and Willows. Oil on canvas, laid over years.

From 1914 until he died in 1926 Monet painted little but the pond he had dug in his garden at
Giverny, on canvases two metres high, set end to end for the oval rooms of the Orangerie. He
left out the bank and the horizon. There is only the surface of the water, seen close, with
the clouds and the willows upside down in it and the lilies lying on it. He came back to each
panel for years, so the paint is thin in one place and thick in the next: a fluid lay-in, long
strokes down the reflections of the willows, rose and gold dragged dry across the clouds once
the paint beneath had dried, and last the pads in short loaded strokes, flatter and smaller as
they go back, and the flowers in a few touches.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Lily Pond, Clouds and Willows"
DATE = "2026"
MEDIUM = "Oil on canvas, thin and thick: a fluid lay-in, long strokes in the reflections, scumbles dragged over dry paint, the pads in loaded dabs"
AFTER = "Claude Monet, the Water Lilies for the Orangerie, Giverny, 1914–26"
ROOM = "The Garden"
YEAR = 1920
PLACE = "Giverny"
REGION = "Europe"
NOTE = ("The surface of the pond seen close, with no bank and no horizon: clouds and willows reflected in it, "
        "lilies lying on it. The paint went on in sittings, each over the last once it had dried.")

H, W = 1600, 4800
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
BLUE = pal("#1e2b4a", "#283a5e", "#33486e", "#41587e", "#53698d", "#687c9d")      # cobalt, ultramarine
VIOLET = pal("#2a2740", "#36314f", "#433c5e", "#544b6e", "#675d80", "#7c7192")    # cobalt violet
TEAL = pal("#1a3235", "#223f41", "#2c4d4d", "#395c5a", "#4a6d69", "#5f807a")      # viridian with blue
DEEP = pal("#0d1a19", "#12241f", "#183027", "#203c31", "#2b4a3c", "#3a5b4a")      # viridian and black
INDIGO = pal("#121629", "#191e37", "#222946", "#2d3555", "#3a4466", "#4c5679")    # ultramarine, deep
CLOUD = pal("#5f5a78", "#76698a", "#8d7a92", "#a38894", "#b5978e", "#c2a48c", "#ccb194", "#d8c2a8",
            "#e1d2be")                                                            # lavender, rose, gold
GOLD = pal("#9a7a52", "#ad8b5c", "#bf9d69", "#cfae79")                           # yellow ochre, Naples yellow
ROSE = pal("#8f6a74", "#a47b84", "#b88e93", "#c9a2a2")                           # madder let down
PADS = pal("#26382d", "#33473a", "#415745", "#506650", "#62765b", "#768867", "#8b9a76")
SHEEN = pal("#34494b", "#435a5a", "#566d6a", "#697d78", "#7d8f88", "#8e9d96")    # leaves holding the sky
LILAC = pal("#3c3a4c", "#4d4a5e", "#615c70", "#777083", "#8e8596", "#a59baa")    # and the clouds
PETALS = pal("#97606a", "#b07d83", "#c89d9d", "#dcbcb5"), pal("#c5bdb3", "#dad2c6", "#e8e1d5"), \
    pal("#ad8a4a", "#c4a160", "#d8bc80")                                         # rose, white, yellow lilies

# the clouds in the water (x, y, spread across, spread up and down, weight), drawn out level as reflections lie
CLOUDS = np.array([(1550, 380, 230, 120, 0.9), (1850, 330, 280, 140, 1.0), (2150, 470, 250, 150, 1.0),
                   (2450, 600, 260, 140, 0.95), (2700, 760, 220, 130, 0.85), (2300, 760, 180, 110, 0.6),
                   (1950, 600, 160, 100, 0.55), (2900, 900, 180, 110, 0.6), (1400, 520, 140, 90, 0.5),
                   (2050, 650, 120, 80, -0.6), (2550, 420, 110, 70, -0.5),            # bays of blue cut into it
                   (3750, 520, 200, 110, 0.85), (4000, 430, 220, 120, 0.8), (3900, 650, 150, 90, 0.5),
                   (3900, 1180, 170, 80, 0.5), (4150, 1300, 140, 70, 0.4)], np.float32)
# where the lilies drift (x from, x to, y, how deep the drift lies, how thickly it is grown)
DRIFTS = [(60, 1250, 1360, 200, 0.85), (1450, 2400, 1460, 160, 0.75), (2600, 3600, 1300, 170, 0.8),
          (3700, 4760, 1420, 180, 0.85), (250, 950, 860, 100, 0.65), (1600, 2200, 720, 70, 0.55),
          (2600, 3200, 960, 100, 0.6), (3850, 4550, 800, 90, 0.65), (250, 900, 260, 45, 0.5),
          (2050, 2600, 140, 35, 0.4), (3150, 3900, 300, 45, 0.45), (4150, 4600, 230, 35, 0.45)]
TRUNKS = [(3330, 0.04, 46), (830, -0.03, 30), (4560, 0.02, 34)]       # reflected trunks: x, lean, half-width


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
    t = np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n
    i = t.astype(int)
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
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds, spread)
    return cols, share


def clouds(r):
    """How much cloud the water holds at each point (0..1), its edges torn and pulled down in
    streaks by the ripples, which side of it faces the sun, and the billows inside it.
    -> (cloud, lit +1 .. shaded -1, billows)"""
    s = 4
    yy, xx = np.mgrid[0:H:s, 0:W:s].astype(np.float32)
    m = sum(k * np.exp(-0.5 * ((xx - cx) / sx) ** 2 - 0.5 * ((yy - cy) / sy) ** 2) for cx, cy, sx, sy, k in CLOUDS)
    m += 0.3 * noise.fbm(yy.shape, 40, r, octaves=4) + 0.12 * noise.stretched(yy.shape, 6, 60, r)
    c = noise.smoothstep(0.3, 1.0, m)
    lit = np.tanh(4 * (np.roll(c, (-10, -15), (0, 1)) - np.roll(c, (10, 15), (0, 1))))
    up = lambda a: ndimage.zoom(a, s, order=1)[:H, :W].astype(np.float32)
    return up(c), up(lit), up(noise.fbm(c.shape, 30, r, octaves=3))


def willows(r, cloud):
    """How dark the willows lie in the water (0..1): curtains of fronds hanging straight down
    through it, swaying a little and parting here and there, and fainter toward the near edge,
    where the eye sees into the pond more than off it; and the trunks, dark bands. -> (shade, trunk)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx + 50 * noise.field((H, W), 500, r) + 140 * noise.line1d(H, 450, r)[:, None]
    curtain = np.maximum.reduce([noise.smoothstep(1450, 850, x), 0.6 * np.exp(-((x - 3330) / 170) ** 2),
                                 noise.smoothstep(4300, 4750, x)])
    fronds = noise.stretched((H, W), 28, 700, r)
    shade = noise.smoothstep(0.35, 0.75, curtain + 0.35 * fronds * noise.smoothstep(0, 0.4, curtain)) \
        * (1 - 0.3 * (yy / H) ** 2) * noise.smoothstep(0.45, 0.05, cloud)
    trunk = np.zeros((H, W), np.float32)
    y = np.arange(H, dtype=np.float32)
    for x0, lean, hw in TRUNKS:
        xt = x0 + lean * (y - H / 2) + 25 * noise.line1d(H, 300, r)
        half = hw * (1 + 0.15 * noise.line1d(H, 200, r))
        trunk = np.maximum(trunk, noise.smoothstep(1.0, 0.7, np.abs(x - xt[:, None]) / half[:, None]))
    return shade.astype(np.float32), trunk


def pads(r):
    """Where each pad lies (x, y, half-width, half-depth): in drifts, larger and rounder toward
    the near edge, flatter and smaller as they go back."""
    out, lump = [], noise.field((H, W), 260, r)
    for x0, x1, y0, depth, thick in DRIFTS:
        R = 24 + 90 * (y0 / H) ** 1.5
        P = dabs.scatter((int(2.4 * depth) + 1, int(x1 - x0) + 1), 1.25 * R, r) + [x0, y0 - 1.2 * depth]
        u, v = (P[:, 0] - (x0 + x1) / 2) / ((x1 - x0) / 2), (P[:, 1] - y0) / depth
        P = P[(1 - u * u - v * v + 0.35 * lump[at(P)] > 0) & (r.random(len(P)) < thick)]
        near = np.clip(P[:, 1] / H, 0, 1)
        rx = (24 + 90 * near ** 1.5) * np.exp(r.normal(0, 0.22, len(P)))
        out.append(np.column_stack([P, rx, rx * (0.2 + 0.22 * near) * r.uniform(0.85, 1.15, len(P))]))
    return np.concatenate(out).astype(np.float32)


def across(pad, v, at_, reach, tilt):
    """A stroke across a pad at `v` (-1 its far edge, 1 its near one), centred `at_` and reaching
    `reach` of the way along its chord there, bowed to the oval."""
    cx, cy, rx, ry = pad
    span = rx * np.sqrt(1 - v * v)
    q = np.linspace(-1, 1, 5)
    x = cx + at_ * span + q * reach * span
    return np.stack([x, cy + v * ry + tilt * q * reach * span - v * 0.3 * ry * q * q], 1)


def leaf(pad, r):
    """One pad in a handful of short strokes laid across it, more or less level, bowed to its
    oval toward the rim and never quite filling it. -> paths, half-widths, where each lies (v)"""
    ry = pad[3]
    m = int(np.clip(2 + ry / 4, 2, 11))
    v = r.uniform(-0.85, 0.85, m)
    paths = [across(pad, v[k], r.uniform(-0.4, 0.4), r.uniform(0.35, 0.75), r.normal(0, 0.12))[::r.choice([-1, 1])]
             for k in range(m)]
    ws = list(np.clip(ry * r.uniform(0.18, 0.35, m), 1.6, 15))
    if ry > 22:                       # a curl round the rim, as the brush turned with the edge of the leaf
        a = r.uniform(0, 2 * np.pi) + np.linspace(0, r.uniform(0.6, 1.1), 6)
        paths.append(np.stack([pad[0] + 0.8 * pad[2] * np.cos(a), pad[1] + 0.8 * ry * np.sin(a)], 1))
        ws.append(0.28 * ry)
        v = np.append(v, np.sin(a[3]))
    return paths, np.array(ws), v


def seat(pad, r):
    """The dark of the water just under a pad's near edge, where it sits in the pond."""
    cx, cy, rx, ry = pad
    q = np.linspace(-1, 1, 5)
    return np.stack([cx + q * rx * r.uniform(0.6, 0.85), cy + ry * (0.75 - 0.25 * q * q) + 2], 1)


def flower(pad, r):
    """A lily on its pad, a little heap of thick paint: two petals spread outward on the water,
    two or three short strokes for the cup above them, and now and then a touch of yellow at
    its heart. -> petal paths, half-widths, how high on the flower each lies; heart paths, half-widths"""
    cx, cy, rx, ry = pad
    f = 0.22 * rx
    x0, y0 = cx + r.normal(0, 0.2 * rx), cy - 0.2 * ry
    m = r.integers(2, 4)
    q = np.linspace(0, 1, 4)
    petals, lift = [], []
    for side in (-1, 1):                                                   # spread on the water, a little apart
        a = (0 if side > 0 else np.pi) + r.normal(0, 0.2) - side * 0.15
        l = f * r.uniform(0.6, 1.0)
        petals.append(np.stack([x0 + side * 0.1 * f + q * l * np.cos(a), y0 + q * l * np.sin(a)], 1))
        lift.append(0.0)
    for _ in range(m):                                                     # the cup, standing up from them
        a = -np.pi / 2 + r.normal(0, 0.6)
        l = f * r.uniform(0.35, 0.7)
        b = np.array([x0 + r.normal(0, 0.25 * f), y0 - 0.1 * f])
        petals.append(b + q[:, None] * l * np.array([np.cos(a), np.sin(a)]))
        lift.append(r.uniform(0.4, 1))
    hearts = [np.array([[x0 - 0.1 * f, y0 - 0.35 * f], [x0 + 0.1 * f, y0 - 0.4 * f]]) + r.normal(0, 0.08 * f, 2)] \
        if r.random() < 0.6 else []
    return petals, np.maximum(2.0, f * r.uniform(0.18, 0.26, len(petals))), lift, hearts, \
        np.full(len(hearts), max(2.0, 0.12 * f))


def paint(seed=1920):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint="#dcd3c2", thread=3.0)
    cloud, lit, billow = clouds(r)
    shade, trunk = willows(r, cloud)
    level = (0.03 + 0.08 * noise.field((H, W), 700, r)).astype(np.float32)
    fall = (np.pi / 2 + 0.05 * noise.field((H, W), 500, r) + 0.025 * noise.field((H, W), 90, r)).astype(np.float32)
    hue = noise.field((H, W), 700, r)                                       # violet here, blue there
    green = noise.field((H, W), 600, r)                                     # and where the water shows its own green
    sweep = noise.field((H, W), 450, r)                                     # where the hand swept long, where it dabbed
    sparing = np.maximum(noise.smoothstep(-0.7, -1.5, noise.field((H, W), 450, r)),       # where the brush went thinner,
                         0.8 * noise.smoothstep(H - 70, H, np.arange(H, dtype=np.float32))[:, None]    # and the foot
                         * noise.smoothstep(-1, 1, noise.field((H, W), 200, r)))                     # left unfinished
    deep = (np.arange(H, dtype=np.float32) / H)[:, None]                    # 0 far, 1 near

    # the wash: the whole canvas rubbed over thin, darker where the willows will go, mauve where
    # the clouds will, the threads showing through and the paint pooled between them
    open_ = ramp(BLUE, 0.45 - 0.2 * deep + 0.08 * hue)
    under = open_ + (ramp(CLOUD, 0.25 + 0.3 * cloud) - open_) * cloud[..., None]
    under += (ramp(DEEP, 0.4 + 0.1 * hue) - under) * (0.9 * shade)[..., None]
    rag = noise.stretched((W, H), 14, 420, r).T[:H, :W] * (1 - shade) + noise.stretched((H, W), 18, 380, r) * shade
    alpha = np.clip((0.86 - 0.25 * sparing + 0.05 * rag) * (1 + 0.1 * (0.5 - ground.tooth)), 0, 1)[..., None]
    rgb = (ground.color * (1 - alpha) + under * (1 + 0.04 * rag[..., None]) * alpha).astype(np.float32)
    height = ground.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W), thinned
        where the brush went sparing."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)] * (1 - 0.8 * sparing[at(P)])]

    def go(field, P, L, n, bend=0.0, tilt=0.0):
        paths = impasto.follow(field, P, L, n, bend, tilt)
        back = r.random(len(P)) < 0.5
        paths[back] = paths[back, ::-1]
        return paths

    def sky(P, lift=0.0):
        """The sky in the water at each seed: which paint (0 blue, 1 violet, 2 teal, 3 cloud) and its tone."""
        y, x = at(P)
        c, d, n = cloud[y, x], deep[y, 0], len(P)
        which = np.where(r.random(n) < c, 3, np.where(green[y, x] + 0.1 * r.normal(0, 1, n) > 0.7 - 1.1 * d, 2,
                                                       (hue[y, x] + 0.08 * r.normal(0, 1, n) > 0.25).astype(int)))
        tone = np.where(which == 3, 0.2 + 0.4 * c + 0.12 * lit[y, x] + 0.12 * billow[y, x],
                        0.55 - 0.3 * d + 0.1 * hue[y, x])
        return which, tone + lift

    # the first sitting, into the wet wash: the surface laid in, broad and fluid
    P = strew(165, np.ones((H, W)))
    n = len(P)
    which, tone = sky(P, -0.1)
    willow = r.random(n) < shade[at(P)]
    which[willow], tone[willow] = 4, 0.3
    lay(go(np.where(shade > 0.5, fall, level), P, r.uniform(250, 500, n), 10, 0, r.normal(0, 0.12, n) * ~willow),
        r.uniform(22, 36, n), mixed((BLUE, VIOLET, TEAL, CLOUD, DEEP), which, tone, r, INDIGO[2:], 0.3),
        thick=0.015, grooves=0.1, lips=0.02, land=0.1, lift=0.05, pickup=0.7, merge=8, spent=0.85, hide=0.6,
        fray=4, taper=0.5, tails=0.5, ends=(0.7, 0.6))
    height -= 0.85 * (height - 0.2 * ground.tooth)      # thinned with turpentine, it sinks into the canvas as it dries
    wet[:] = 0

    # the second: the reflections. The sky in the water, level, the clouds in short strokes every
    # which way; then the willows straight down through it, wet into wet, and the trunks
    P = strew(44, (1 - shade) ** 2)
    n = len(P)
    which, tone = sky(P, -0.15 * shade[at(P)])
    cl = which == 3
    lilac = cl & (r.random(n) < 0.22)                                      # lavender broken into the clouds
    which[lilac], tone[lilac] = 1, 0.75 + 0.2 * cloud[at(P)][lilac]
    g = sweep[at(P)]
    big = (0.85 + 0.3 * deep[at(P)[0], 0]) * np.exp(0.15 * g)
    slant = np.where(cl, r.normal(0, 0.45, n), r.normal(0, 0.04 + 0.12 * noise.smoothstep(0, -1.5, g), n))
    lay(go(level, P, np.where(cl, r.uniform(40, 120, n), r.uniform(70, 200, n) * np.exp(0.5 * g)) * big, 8,
           r.normal(0, 0.0006, n), slant),
        np.where(cl, r.uniform(11, 20, n), r.uniform(8, 16, n)) * big,
        mixed((BLUE, VIOLET, TEAL, CLOUD), which, tone, r, np.concatenate([ROSE, BLUE[3:]]), 0.08),
        thick=0.1, spent=0.7, pickup=0.7, merge=4, grooves=0.5, dry=r.uniform(0.15, 0.5, n), fray=2.5, taper=0.35,
        hide=0.88)
    P = strew(50, shade)
    n = len(P)
    s = shade[at(P)]
    fam = np.digitize(hue[at(P)] + 0.25 * r.normal(0, 1, n), [0.3, 1.0])    # deep green, indigo, teal
    fringe = r.random(n) > noise.smoothstep(0.15, 0.6, s)
    fam[fringe] = 2 + (r.random(fringe.sum()) < 0.4)                       # teal or blue, near the tone of the water
    lay(go(fall, P, r.uniform(150, 450, n) * (0.6 + 0.4 * s), 12, r.normal(0, 0.0006, n)), r.uniform(8, 16, n),
        mixed((DEEP, INDIGO, TEAL, BLUE), fam, np.where(fringe, 0.46, 0.2 + 0.25 * (1 - s)) + 0.05 * r.normal(0, 1, n), r,
              VIOLET[1:4], 0.2),
        thick=0.18, spent=0.5, pickup=0.55, merge=2, grooves=0.6, taper=0.15, dry=r.uniform(0.05, 0.3, n), fray=2.5)
    P = strew(38, trunk)
    n = len(P)
    lay(go(fall, P, r.uniform(200, 450, n), 12, r.normal(0, 0.0005, n)), r.uniform(8, 14, n),
        mixed((VIOLET, INDIGO), (r.random(n) < 0.6).astype(int), np.full(n, 0.3), r, DEEP[1:4], 0.25),
        thick=0.2, spent=0.6, pickup=0.6, merge=3, dry=r.uniform(0.15, 0.45, n), hide=0.85, fray=2.5)
    wet[:] = 0
    height += 0.35 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2))    # crusts the years left

    # the third, over dry paint: rose and gold dragged across the clouds, veils of sky over the
    # willows and the open water, ultramarine pulled into the edges of the clouds
    P = strew(30, noise.smoothstep(0.3, 0.6, cloud) * noise.smoothstep(-0.4, 0.6, billow))
    n = len(P)
    c = cloud[at(P)]
    lt = lit[at(P)]
    lay(go(level, P, r.uniform(40, 110, n), 8, r.normal(0, 0.002, n), r.normal(0, 0.22, n)), r.uniform(10, 20, n),
        loads(CLOUD, np.minimum(0.32 + 0.4 * c + 0.15 * lt + 0.1 * billow[at(P)], 0.88), r,
              np.concatenate([GOLD[2:], ROSE[2:]]), 0.1 + 0.25 * (lt > 0.3)),
        dry=r.uniform(0.55, 0.85, n), hide=0.85, thick=0.12, land=0.2, lift=0.1, pickup=0, grooves=0.4, ends=(0.5, 0.3))
    P = strew(70, np.maximum(noise.smoothstep(0.4, 0.7, shade), 0.6 * trunk))
    n = len(P)
    lay(go(fall, P, r.uniform(200, 420, n), 12, r.normal(0, 0.0006, n)), r.uniform(10, 20, n),
        mixed((VIOLET, BLUE, TEAL), r.integers(0, 3, n), np.full(n, 0.45), r),
        dry=r.uniform(0.6, 0.85, n), hide=0.5, thick=0.08, land=0.1, lift=0.05, pickup=0, grooves=0.3)
    P = strew(80, (1 - shade) * (1 - cloud))
    n = len(P)
    lay(go(level, P, r.uniform(80, 200, n), 8, 0, r.normal(0, 0.08, n)), r.uniform(9, 16, n),
        mixed((BLUE, VIOLET), r.integers(0, 2, n), 0.62 - 0.3 * deep[at(P)[0], 0], r, CLOUD[2:4], 0.2),
        dry=r.uniform(0.6, 0.8, n), hide=0.7, thick=0.08, land=0.1, lift=0.05, pickup=0, grooves=0.3)
    P = strew(70, noise.smoothstep(0.1, 0.25, cloud) * noise.smoothstep(0.65, 0.4, cloud))
    n = len(P)
    lay(go(level, P, r.uniform(80, 200, n), 8, 0, r.normal(0, 0.3, n)), r.uniform(8, 14, n),
        loads(INDIGO, np.full(n, 0.6), r, BLUE[2:5], 0.4), dry=0.5, hide=0.85, thick=0.1, pickup=0)
    wet[:] = 0

    # the fourth: the lilies, wet into wet. Under each pad the dark it sits in, then the pad in
    # loaded strokes, its far side catching the sky
    lily = pads(r)
    y, x = at(lily[:, :2])
    N = len(lily)
    lay([seat(p, r) for p in lily], 0.35 * lily[:, 3] + 1.5,
        mixed((DEEP, INDIGO), (r.random(N) < 0.4).astype(int), np.full(N, 0.2), r), thick=0.1, hide=0.55, dry=0.3,
        pickup=0.3)
    paths, ws, vs, owner = [], [], [], []
    for i, p in enumerate(lily):
        a, b, c = leaf(p, r)
        paths += a
        ws += list(b)
        vs += list(c)
        owner += [i] * len(a)
    owner, vs = np.array(owner), np.array(vs)
    tone = (0.38 + 0.2 * deep[y, 0] + 0.1 * hue[y, x] - 0.15 * cloud[y, x] + 0.12 * shade[y, x] * deep[y, 0]
            + r.normal(0, 0.1, N))[owner] - 0.22 * vs + r.normal(0, 0.12, len(vs))
    u = r.random(N)
    fam = np.where(u < 0.15 + 0.5 * cloud[y, x], 2, (u < 0.4 + 0.4 * (1 - deep[y, 0])).astype(int))[owner]
    lay(paths, np.array(ws), mixed((PADS, SHEEN, LILAC), fam, tone, r, np.concatenate([CLOUD[3:7], ROSE[1:3], TEAL[1:3]]),
                                   0.35),
        thick=0.36, spent=0.55, pickup=0.45, merge=2, grooves=1.0, dry=r.uniform(0.25, 0.6, len(paths)))
    wet[:] = 0

    # the last: light dragged over the far side of some pads, and the flowers
    lit_pads = lily[(r.random(N) < 0.5) & (lily[:, 3] > 8)]
    hl = [across(p, r.uniform(-0.7, -0.25), r.uniform(-0.3, 0.3), r.uniform(0.3, 0.6), r.normal(0, 0.08))
          for p in lit_pads]
    lay(hl, np.clip(0.3 * lit_pads[:, 3], 2, 10), mixed((SHEEN, CLOUD), r.integers(0, 2, len(hl)), np.full(len(hl), 0.6), r),
        dry=r.uniform(0.45, 0.7, len(hl)), thick=0.15, pickup=0)
    bloom = lily[(lily[:, 2] > 35) & (lily[:, 1] > 600)]
    bloom = bloom[r.permutation(len(bloom))[:16]]
    petals, pw, kinds, ups, hearts, hw = [], [], [], [], [], []
    for p in bloom:
        a, b, c, d, e = flower(p, r)
        petals += a
        pw += list(b)
        ups += list(c)
        kinds += [r.choice(3, p=[0.5, 0.42, 0.08])] * len(a)
        hearts += d
        hw += list(e)
    kinds = np.array(kinds)
    lay(petals, np.array(pw), mixed(PETALS, kinds, 0.25 + 0.6 * np.array(ups), r, PETALS[1][1:], 0.3), thick=0.45,
        spent=0.35, pickup=0.35, ends=(0.3, 0.2), dry=r.uniform(0.2, 0.45, len(petals)))
    lay(hearts, np.array(hw), loads(PETALS[2], np.full(len(hearts), 0.5), r), thick=0.5, spent=0.3, pickup=0.4, dry=0.25)

    height = ndimage.gaussian_filter(height, 0.8)      # the lamp sees the surface a little softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.6, gloss=0, reach=(0.75, 1.15))
    return img + 0.05 * impasto.glints(height, LIGHT)[..., None]
