"""The Outer Harbour, Sunrise. Oil on canvas, painted in one sitting.

In 1872 Monet painted the outer harbour of Le Havre from a window above the quays, at sunrise on
a morning of mist. He worked fast, on a light ground, in paint thinned until it went on almost
like a stain: the sky scrubbed in with a broad brush, the water laid across it in level touches
while it was still wet, the masts and chimneys of the far side drawn into the wet mist with a
smaller brush and lost again, the boats a few dark strokes. The sun and its reflection went on
last, in short strokes of thicker orange. Shown in Paris two years later, the picture gave its
title, Impression, to a critic who used the word to mock the whole exhibition.

Here the sun stands a little left of the middle, over the harbour mouth, in a mist of blue-violet
and lilac. Its rose and orange glow spreads round it and drifts faintly across the upper sky, and
lies pale on the water about its reflection. Chimneys and a crane stand on the left, their smoke
drifting, and on the right ships lie at the quay, a steamer among them, one nearer than the rest.
Three boats are out on the water. The paint is so thin that the weave of the canvas shows.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Outer Harbour, Sunrise"
DATE = "2026"
MEDIUM = ("Oil on canvas, thin paint scrubbed and laid wet into wet over a light ground in one sitting, "
          "the sun and its reflection in thicker paint")
AFTER = "Claude Monet, Impression, Sunrise, Le Havre, 1872"
ROOM = "Open Air"
YEAR = 1872
PLACE = "Le Havre"
REGION = "Europe"
NOTE = ("The outer harbour at dawn, its masts, cranes and chimneys going into the mist, three boats out on the water. "
        "All of it thin and quick, in one sitting; only the orange sun and its reflection are thick.")

H, W = 2130, 2800            # about 48 by 63 cm
QUAY = 1010                  # where the far side of the harbour meets the water
SX, SY, SR = 1060.0, 610.0, 46.0
GROUND = "#d6d0c3"           # a light grey priming
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
MIST = pal("#5b6185", "#686e93", "#767ca0", "#858aad", "#9498b9", "#a4a7c5", "#b5b7d1")   # blue-violet, cobalt let down
LILAC = pal("#6a6582", "#787391", "#87829f", "#9691ad", "#a6a1bb", "#b6b1c8", "#c7c2d5")  # lilac
ROSE = pal("#a3848c", "#b18d8e", "#bf9790", "#cba293", "#d5ae99", "#ddbaa2", "#e4c7ae")   # the sun's glow, rose and orange
SEA = pal("#435478", "#4e6084", "#5b6c90", "#69799c", "#7886a8", "#8893b4", "#99a1c0")    # the water, holding the mist
TEAL = pal("#3f5e6e", "#4a6a7a", "#577786", "#658492", "#74929f")                         # and its own blue-green
DEEP = pal("#1e2742", "#26304d", "#2f3a58", "#3a4564", "#465172")                         # ultramarine, in the ripples
FAR = pal("#464a68", "#525675", "#5f6382", "#6c708f", "#7a7e9c")                          # the shapes in the mist
DARK = pal("#141924", "#1a202d", "#222937", "#2b3341")                                     # the boats
ORANGE = pal("#b9401f", "#d2502a", "#e2602f", "#eb7438", "#f08a4a", "#f2a466")             # the sun

# paint thinned with turpentine: it hardly stands up from the canvas, and drags into the wet paint around it
THIN = dict(thick=0.002, grooves=0.04, lips=0.0, land=0.05, lift=0.0, tails=0.4, pickup=0.65, merge=10.0, spent=0.5,
            ends=(0.5, 0.4), taper=0.3, fray=4.0, flatten=0.0)
# and paint as it comes from the tube
LOADED = dict(thick=0.3, grooves=0.45, lips=0.15, land=0.6, lift=0.35, tails=0.9, pickup=0.15, merge=1.5, spent=0.7,
              ends=(0.15, 0.05), taper=0.5, fray=1.0, flatten=0.8)

SHIPS = [(1560, 2190, 950, [(1720, 320), (1900, 225)]),                  # at the quay: bow, stern, deck, masts (x, top)
         (2330, 2900, 958, [])]
FAINT = [(1610, 600), (2140, 520)]                                        # masts farther off, behind them
CHIMNEYS = [(150, QUAY, 330, 16, 1.0), (300, QUAY, 520, 11, 0.6), (2235, 952, 770, 13, 0.5)]   # x, foot, top, half-width, smoke
MASTS = [(60, 720), (430, 680), (900, 790)]                               # fishing boats on the left
NEAR = (2575, 1112, 250, 10.0, 2390, 2770)                                # a ship nearer in: mast, waterline, top, half-width, bow, stern
BOATS = [(1400, 1570, 265, 0.0, 2), (430, 1330, 175, 0.15, 2), (2070, 1240, 115, 0.35, 1)]   # x, y, length, mist, rowers


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(np.asarray(t).astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.05):
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


def mixed(families, which, tone, r, accent=None, odds=0.2, spread=0.05):
    """Loads for strokes drawn from several palettes, `which` saying which one each takes."""
    cols, share = np.empty((len(which), 3, 3), np.float32), np.empty((len(which), 3))
    for i, colours in enumerate(families):
        sel = which == i
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds, spread)
    return cols, share


def fine(p):
    """A palette with a mixture between each pair of neighbours: strokes loaded from it streak less."""
    return np.concatenate([np.stack([p[:-1], (p[:-1] + p[1:]) / 2], 1).reshape(-1, 3), p[-1:]])


def line(a, b, n=6, bow=0.0):
    """Points from a to b, the middle pushed `bow` px to one side."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = b - a
    q = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return a + q * d + np.array([d[1], -d[0]]) / (np.hypot(*d) + 1e-6) * 4 * bow * q * (1 - q)


def harbour(r):
    """The far side, drawn into the wet mist with a smaller brush: low sheds and quays, the hulls of
    the ships, their masts, a few yards and a little rigging, the chimneys and their smoke, a crane.
    Each mast goes on first as a wide soft stroke the wet sky half takes back, then in two loads of a
    narrower brush. -> {kind: (paths, half-widths, tones)}"""
    out = {k: ([], [], []) for k in ("blocks", "hull", "haze", "uprights", "tops", "spars", "lines", "smoke")}

    def put(kind, path, w, tone):
        out[kind][0].append(path)
        out[kind][1].append(w)
        out[kind][2].append(tone)

    def mast(mx, foot, top, w, tone, yards=0.5):
        """A mast in two or three loads of the brush, each higher one lighter and drier, with a gap
        between where the mist swallowed it; a few yards across it, longer low down."""
        lean = r.normal(0, 0.012)
        xm = lambda y: mx + lean * (foot - y)
        put("haze", line((xm(foot), foot), (xm(top + 30), top + 30), 5, r.normal(0, 4)), 2.4 * w, tone + 0.2)
        cuts = [0.0] + sorted(r.uniform(0.25, 0.8, r.integers(1, 3))) + [1.0]
        for k in range(len(cuts) - 1):
            ya, yb = (foot - (c * (foot - top)) for c in (cuts[k] + (k > 0) * r.uniform(0.04, 0.12), cuts[k + 1]))
            put("uprights" if k == 0 else "tops", line((xm(ya), ya), (xm(yb), yb), 4, r.normal(0, 2)), w * (1 - 0.15 * k),
                tone + 0.12 * k)
        for f in (0.15, 0.35, 0.55):
            if r.random() < 0.6 * yards:
                y = top + f * (foot - top) + r.normal(0, 15)
                span = (25 + 0.15 * (y - top)) * r.uniform(0.7, 1.2) * w / 6
                t = r.normal(0, 0.06)
                put("spars", line((xm(y) - span, y - t * span), (xm(y) + span, y + t * span), 4), 0.5 * w, tone + 0.1)
        return xm

    x = -60.0
    while x < 1000:                                        # sheds and warehouses along the left quay
        bw, h = r.uniform(90, 260), r.uniform(40, 120) * (1.2 - x / 1300)
        put("blocks", line((x, QUAY - 4 - h / 2), (x + bw, QUAY - 4 - h / 2 + r.normal(0, 3)), 4), h / 2 + 4, r.uniform(0.3, 0.7))
        x += bw * r.uniform(0.6, 0.95)
    for x in np.arange(980, 1560, 160.0):                  # the harbour mouth, a low jetty barely there
        put("blocks", line((x, QUAY - 10), (x + r.uniform(150, 260), QUAY - 12 + r.normal(0, 3)), 4), r.uniform(8, 16), 0.85)
    for x in np.arange(1480, 2900, 200.0):                 # the quay behind the ships
        put("blocks", line((x, QUAY - 45), (x + r.uniform(220, 320), QUAY - 45 + r.normal(0, 4)), 4), r.uniform(28, 45), r.uniform(0.35, 0.65))
    for bow, stern, deck, masts in SHIPS:
        put("blocks", line((bow, deck + 22), (stern, deck + 26), 6, r.normal(0, 3)), 30, 0.1)
        put("blocks", line((bow + 30, deck + 44), (stern, deck + 46), 6), 16, 0.02)
        put("spars", line((bow + 10, deck + 2), (bow - 130, deck - 55), 4), 3.4, 0.3)              # the bowsprit
        for mx, top in masts:
            mast(mx, deck, top, 6.5, 0.3, 0.5)
    for mx, top in FAINT:
        mast(mx, QUAY - 50, top, 4.5, 0.55, 0.0)
    for mx, top in MASTS:
        mast(mx, QUAY - 20, top, 5.0, 0.4, 0.25)
    mx, wl, top, w, bow, stern = NEAR                      # the nearer ship, darker and plainer than the rest
    put("hull", line((bow, wl - 20), (stern, wl - 16), 6, r.normal(0, 3)), 22, 0.05)
    put("hull", line((bow + 40, wl - 4), (stern - 10, wl - 2), 6), 9, 0.0)
    put("spars", line((bow + 10, wl - 32), (bow - 120, wl - 80), 4), 4.0, 0.15)
    xm = mast(mx, wl - 30, top, w, 0.05, 1.0)
    for e in (bow - 110, stern - 20):
        put("lines", line((xm(top + 20), top + 20), (e, wl - 34), 5, r.normal(0, 6)), 1.8, 0.25)
    for cx, foot, top, hw, smoke in CHIMNEYS:
        lean = r.normal(0, 4)
        mid = (foot + top) / 2
        put("haze", line((cx, foot), (cx + 2 * lean, top + 20), 4), 2 * hw, 0.4)
        put("uprights", line((cx, foot - 30), (cx + lean, mid), 4), hw, 0.08)
        put("uprights", line((cx + lean, mid + 30), (cx + 2 * lean, top), 4), 0.9 * hw, 0.14)
        for k in range(int(10 * smoke)):                   # the smoke going up and drifting off to the right
            q = np.linspace(0, 1, 7)[:, None]
            reach, rise = r.uniform(250, 650) * smoke, r.uniform(160, 320) * smoke
            p = np.hstack([cx + 2 * lean + reach * q + r.normal(0, 14), top - 10 - rise * np.sqrt(q) + 40 * q * q
                           + 18 * np.sin(np.pi * q * r.uniform(1, 2) + r.uniform(0, 6.3))])
            put("smoke", p + r.normal(0, 4, p.shape), r.uniform(20, 32) + 26 * k / 10, r.uniform(0.22, 0.52))
    cx, top, jx, jy = 600, 690, 830, 520                   # the crane
    put("haze", line((cx, QUAY - 15), (cx + 5, top + 30), 4), 14, 0.4)
    put("uprights", line((cx, QUAY - 15), (cx + 2, top + 120), 4), 7.0, 0.15)
    put("uprights", line((cx + 2, top + 150), (cx + 5, top), 4), 5.5, 0.2)
    put("spars", line((cx, top + 25), (jx, jy), 5, 4), 4.5, 0.2)
    put("lines", line((cx + 5, top - 40), (jx, jy), 4), 1.7, 0.3)
    put("lines", line((jx - 2, jy + 4), (jx - 4, jy + 180), 4), 1.5, 0.3)
    put("haze", line((1330, QUAY - 8), (1332, 790), 4), 14, 0.75)                    # a far beacon at the harbour mouth
    put("uprights", line((1330, QUAY - 8), (1332, 800), 4), 7, 0.6)
    return {k: (v[0], np.array(v[1], np.float32), np.array(v[2], np.float32)) for k, v in out.items()}


def boat(x, y, L, rowers, r):
    """A rowing boat in a few dark strokes: the hull, a pale catch of light along the gunwale, the
    rowers (one standing at the stern to scull, one sitting), the oar, and the broken reflection.
    -> lists of (path, half-width, tone) for the hull and oar, the rowers, the gunwale and the reflection"""
    dark, figs, edge, refl = [], [], [], []
    sag = lambda a, b, yy, bow: line((x + a * L, yy), (x + b * L, yy + r.normal(0, 0.01 * L)), 6, bow)
    dark.append((sag(-0.5, 0.5, y, -0.03 * L), 0.095 * L, 0.3))
    dark.append((sag(-0.42, 0.46, y + 0.045 * L, 0.0), 0.06 * L, 0.1))
    dark.append((line((x - 0.42 * L, y - 0.01 * L), (x - 0.6 * L, y - 0.07 * L), 4), 0.04 * L, 0.2))   # the stem
    edge.append((sag(-0.45, 0.4, y - 0.08 * L, -0.02 * L), 0.02 * L, 0.55))
    xs = x + 0.32 * L                                       # the sculler at the stern
    figs.append((line((xs, y - 0.04 * L), (xs + 0.014 * L, y - 0.31 * L), 5, 0.006 * L), 0.034 * L, 0.15))
    figs.append((line((xs + 0.008 * L, y - 0.29 * L), (xs + 0.022 * L, y - 0.355 * L), 3), 0.023 * L, 0.1))
    dark.append((line((xs - 0.03 * L, y - 0.22 * L), (xs + 0.28 * L, y + 0.14 * L), 4), 0.008 * L + 1.2, 0.2))
    if rowers > 1:                                          # and one sitting forward
        xb = x - 0.2 * L
        figs.append((line((xb - 0.01 * L, y - 0.03 * L), (xb + 0.012 * L, y - 0.125 * L), 3), 0.045 * L, 0.15))
        figs.append((line((xb + 0.008 * L, y - 0.12 * L), (xb + 0.016 * L, y - 0.165 * L), 3), 0.024 * L, 0.1))
    for k in range(4):                                      # the reflection, broken by the ripples
        yy = y + (0.14 + 0.1 * k) * L
        a = r.uniform(-0.45, -0.1) + 0.05 * k
        refl.append((sag(a, a + r.uniform(0.35, 0.8) * (1 - 0.18 * k), yy, r.normal(0, 0.03 * L)),
                     (0.045 - 0.007 * k) * L * r.uniform(0.7, 1.2), 0.3))
    for k in range(3):                                      # and the sculler's, in short wavering touches
        yy = y + (0.1 + 0.08 * k) * L
        refl.append((line((xs - 0.05 * L, yy), (xs + 0.05 * L, yy + r.normal(0, 0.01 * L)), 4, r.normal(0, 0.01 * L)), 0.02 * L, 0.3))
    return dark, figs, edge, refl


def sun(rgb, height, r):
    """The sun, one round touch of thick orange put on with a twist of the brush: heaped toward the
    middle and combed round by the bristles, paler where the twist lifted off, its edge a little
    greyed where it met the wet mist."""
    k = int(1.4 * SR)
    sl = (slice(int(SY) - k, int(SY) + k), slice(int(SX) - k, int(SX) + k))
    yy, xx = np.mgrid[sl].astype(np.float32)
    rho, th = np.hypot(xx - SX, yy - SY) / SR, np.arctan2(yy - SY, xx - SX)
    start = r.uniform(0, 2 * np.pi)
    turn = (th - start) % (2 * np.pi)                       # how far round the brush had gone
    B = impasto.bristles(r)
    look = lambda along, across, o: ndimage.map_coordinates(B, [along + o, across + o], order=1, mode="grid-wrap")
    coil = (rho + 0.5 * turn / (2 * np.pi) + 0.05 * np.cos(3 * th + r.uniform(0, 6.3))) * SR   # wound inward as it went round
    comb = look(turn * np.maximum(rho, 0.15) * SR, coil * 0.8, 300.0)            # streaks running round
    clump = look(turn * np.maximum(rho, 0.15) * SR * 0.3, coil * 0.25, 900.0)
    edge = 1 + 0.035 * np.cos(2 * th + r.uniform(0, 6.3)) + 0.03 * np.cos(3 * th + r.uniform(0, 6.3)) \
        + 0.015 * np.cos(5 * th + r.uniform(0, 6.3)) + 0.03 * clump
    a = noise.smoothstep(1.03, 0.93, rho / edge)
    lift = np.exp(-((rho - 0.5 + 0.12 * turn / (2 * np.pi)) / 0.11) ** 2) * noise.smoothstep(3.0, 5.5, turn)
    col = ORANGE[2] + (ORANGE[1] - ORANGE[2]) * (0.22 * noise.smoothstep(-1, 1, comb + 0.8 * clump))[..., None] \
        + (ORANGE[4] - ORANGE[2]) * (0.4 * lift)[..., None]
    rim = noise.smoothstep(0.75, 1.0, rho / edge)[..., None]
    col = np.exp(np.log(col) * (1 - 0.3 * rim) + np.log(np.clip(rgb[sl], 1e-4, 1)) * 0.3 * rim)
    rgb[sl] += (col - rgb[sl]) * a[..., None]
    heap = 2.6 * np.clip(1 - rho * rho, 0, 1) ** 0.3 + 0.9 * lift + 0.3 * noise.smoothstep(-0.5, 0.5, comb)
    height[sl] += (np.maximum(height[sl], heap + 0.1) - height[sl]) * a


def paint(seed=1872):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint=GROUND, thread=3.0)
    tooth = 0.5 * ground.tooth
    height = tooth.copy()
    Y = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    X = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    deep = np.clip((Y - QUAY) / (H - QUAY), 0, 1)          # 0 at the far quay, 1 at the foot of the canvas
    sway = 28 * noise.line1d(H, 260, r)                    # the sun's road wavers as it comes toward us
    road = np.exp(-((X - SX - sway[:, None]) / (70 + 240 * deep)) ** 2) * (Y > QUAY)
    # the sun's glow: spread round it, though not right up to it, and drifting faintly across the
    # upper sky; and its reflection, lying wide about the sun's on the water
    halo = np.exp(-((X - SX) / 560) ** 2 - ((Y - SY + 40) / 330) ** 2) * (1 - 0.8 * np.exp(-((X - SX) ** 2 + (Y - SY) ** 2) / 170 ** 2))
    upper = noise.smoothstep(650, 80, Y) * noise.smoothstep(-1.2, 1.0, noise.field((H, W), 600, r))
    warm = (np.clip(0.65 * halo + 0.18 * upper, 0, 1) * (Y < QUAY + 20)).astype(np.float32)
    glint = (np.exp(-((X - SX - sway[:, None]) / (90 + 220 * deep)) ** 2) * (Y > QUAY) * (0.7 - 0.4 * deep)).astype(np.float32)

    # first the whole canvas rubbed over with a rag and thin colour, blue-violet above and below, rose
    # about the sun, so that the ground glows through whatever goes over it; the paint pools between the threads
    y = np.arange(H, dtype=np.float32)
    under = ramp(MIST, np.interp(y, [0, 300, 600, 850, QUAY], [0.66, 0.62, 0.56, 0.62, 0.68]))
    under += (ramp(SEA, 0.68 - 0.3 * np.clip((y - QUAY) / (H - QUAY), 0, 1) ** 0.9) - under) \
        * noise.smoothstep(QUAY - 40, QUAY + 40, y)[:, None]
    under = under[:, None] + (ramp(ROSE, 0.7) - under[:, None]) * (0.35 * warm + 0.25 * glint)[..., None]
    rag = noise.stretched((W, H), 30, 220, r).T[:H, :W]
    alpha = np.clip(0.5 + 0.2 * noise.smoothstep(QUAY - 200, QUAY + 100, y)[:, None] + 0.05 * rag
                    + 0.15 * (0.5 - ground.tooth), 0, 1)[..., None]
    rgb = np.ascontiguousarray(ground.color * (1 - alpha) + under * (1 + 0.02 * rag[..., None]) * alpha, np.float32)
    wet = np.full((H, W), 0.5, np.float32)                 # and left to set a little before the brush went over it
    big = noise.field((H, W), 700, r)                      # where the mist lies heavier or lighter
    scrub = noise.field((H, W), 500, r)                    # how the arm swung as it scrubbed
    busy = noise.field((H, W), 380, r)                     # where the water is broken, where it lies calm
    flat = np.zeros((1, 1), np.float32)
    sky, sea = (Y < QUAY + 40).astype(np.float32), (Y > QUAY - 20).astype(np.float32)

    def lay(paths, w, load, how=THIN, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **{**how, **kw})

    def strew(spacing, odds, squash=1.0):
        """Where the strokes of a passage lie: a shaken honeycomb, kept by `odds` (H,W), its rows
        `squash` times closer together than its points, for strokes laid more across than up."""
        P = (dabs.scatter((int(squash * H) + 2 * spacing, W + 2 * spacing), spacing, r) - spacing) / [1, squash]
        return P[r.random(len(P)) < odds[at(P)]]

    def go(P, length, angle, n=8, bend=0.0):
        """Strokes of `length` through each point (their middles), turned `angle` and bowed by `bend`,
        drawn from either end."""
        paths = impasto.follow(flat, P, length, n, bend, angle)
        paths -= (paths[:, -1:] - paths[:, :1]) / 2
        back = r.random(len(P)) < 0.5
        paths[back] = paths[back, ::-1]
        return paths

    def air(P, lift=0.0, spread=0.08):
        """The paint of the sky at each point: darkest about the sun, lighter overhead and down in
        the mist on the water; blue-violet and lilac, and rose where the glow is."""
        y, x = at(P)
        n = len(P)
        tone = np.interp(y, [0, 300, 600, 850, QUAY], [0.66, 0.62, 0.54, 0.62, 0.7]) + 0.05 * big[y, x] + lift
        u = r.random(n)
        rose = 0.7 * warm[y, x]
        lilac = 0.15 + 0.25 * noise.smoothstep(650, 80, y) + 0.2 * noise.smoothstep(700, QUAY, y)
        which = np.where(u < rose, 2, np.where(u < rose + (1 - rose) * lilac, 1, 0))
        return mixed((fine(MIST), fine(LILAC), fine(ROSE)), which, tone + 0.08 * (which == 2), r, MIST[3:6], 0.12, spread)

    def water(P, lift=0.0, spread=0.06):
        """The paint of the water: pale with mist near the far side, deeper toward us, paler down the
        sun's road; lilac, blue-violet, now and then its own blue-green, and rose about the reflection."""
        y, x = at(P)
        n = len(P)
        d = deep[y, x]
        tone = 0.68 - 0.36 * d ** 0.9 + 0.06 * road[y, x] + 0.05 * big[y, x] + lift
        u = r.random(n)
        pink, lil, teal = 0.45 * glint[y, x], 0.4 * (1 - d) ** 1.5 + 0.06, 0.05 + 0.3 * d
        which = np.where(u < pink, 3, np.where(u < pink + lil, 1, np.where(u > 1 - teal, 2, 0)))
        tone = tone + np.array([0.0, -0.04, 0.1, 0.06])[which]    # so that the four lie close in value
        return mixed((fine(SEA), fine(LILAC), fine(TEAL), fine(ROSE)), which, tone, r, SEA[2:5], 0.15, spread)

    # the sky scrubbed in with the broadest brush, this way and that, thin
    P = strew(170, sky, 1.8)
    n = len(P)
    ang = 0.25 * scrub[at(P)] + r.normal(0, 0.2, n) + np.where(r.random(n) < 0.35, r.choice([-0.6, 0.6], n), 0)
    lay(go(P, r.uniform(250, 600, n), ang, 8, r.normal(0, 0.0006, n)), r.uniform(40, 80, n), air(P),
        dry=r.uniform(0.0, 0.15, n), hide=0.75, merge=8)
    # the water laid across while it was wet, in long level strokes
    P = strew(220, sea, 3.5)
    n = len(P)
    lay(go(P, r.uniform(300, 900, n), r.normal(0, 0.025, n), 8, r.normal(0, 0.0003, n)), r.uniform(28, 55, n), water(P),
        dry=r.uniform(0.0, 0.15, n), hide=0.8, merge=8, taper=0.45, ends=(0.6, 0.5), spent=0.7)
    # and the sky worked again, wet into wet, lighter here and darker there
    P = strew(200, sky, 1.6)
    n = len(P)
    ang = 0.2 * scrub[at(P)] + r.normal(0, 0.2, n)
    lay(go(P, r.uniform(200, 500, n), ang, 8, r.normal(0, 0.001, n)), r.uniform(30, 60, n), air(P, r.normal(0, 0.06, n)),
        dry=r.uniform(0.0, 0.2, n), hide=0.4, spent=0.6, merge=6)
    # the glow brushed round the sun, rose into orange, and dragged thin across the upper sky
    P = strew(180, sky * halo, 1.5)
    n = len(P)
    lay(go(P, r.uniform(180, 460, n), r.normal(0, 0.3, n), 8, r.normal(0, 0.001, n)), r.uniform(22, 48, n),
        loads(fine(ROSE), r.uniform(0.45, 0.85, n), r, ORANGE[3:], 0.1), dry=r.uniform(0.15, 0.4, n), hide=0.38, spent=0.7, merge=8)
    P = strew(240, sky * upper, 2.0)
    n = len(P)
    lay(go(P, r.uniform(250, 600, n), r.normal(0, 0.15, n), 8, r.normal(0, 0.0006, n)), r.uniform(20, 45, n),
        loads(fine(ROSE), r.uniform(0.55, 0.9, n), r), dry=r.uniform(0.35, 0.6, n), hide=0.22, spent=0.8, merge=6)
    # last a few dry scrubs high up, catching on the weave
    P = strew(230, sky * noise.smoothstep(800, 200, Y))
    n = len(P)
    ang = r.normal(0, 0.25, n) + r.choice([-0.5, 0.0, 0.5], n)
    lay(go(P, r.uniform(200, 450, n), ang, 8, r.normal(0, 0.001, n)), r.uniform(30, 55, n), air(P, 0.07),
        dry=r.uniform(0.4, 0.6, n), hide=0.5, spent=0.8)

    # the far side drawn into the wet mist
    far = harbour(r)
    for kind, kw in (("blocks", dict(hide=0.65, dry=0.1)), ("hull", dict(hide=0.8, dry=0.1, merge=4, pickup=0.4)),
                     ("haze", dict(hide=0.25, dry=0.15, merge=8, pickup=0.8)),
                     ("uprights", dict(hide=0.6, dry=0.15, merge=5, pickup=0.35, spent=0.85, tails=0.05, fray=2.0)),
                     ("tops", dict(hide=0.4, dry=0.4, merge=5, pickup=0.35, spent=0.95, tails=0.05, fray=2.0)),
                     ("spars", dict(hide=0.4, dry=0.35, merge=4, pickup=0.3, tails=0.05, fray=1.5)),
                     ("lines", dict(hide=0.22, dry=0.45, merge=2, fray=1.0, pickup=0.3, tails=0.05)),
                     ("smoke", dict(hide=0.55, dry=0.2, merge=10, pickup=0.6))):
        paths, w, tone = far[kind]
        lay(paths, w, loads(MIST if kind == "smoke" else FAR, tone, r, LILAC[2:5], 0.25), **kw)
    # the mist laid back over it, most about the harbour mouth and up the masts, so that they go into it
    veil = np.exp(-((Y - QUAY + 110) / 200) ** 2) * (0.45 + 0.55 * np.exp(-((X - SX) / 600) ** 2)) \
        + 0.8 * noise.smoothstep(800, 150, Y) * (Y < QUAY)
    P = strew(140, veil.astype(np.float32), 2.0)
    n = len(P)
    lay(go(P, r.uniform(250, 650, n), r.normal(0, 0.1, n), 8, r.normal(0, 0.0006, n)), r.uniform(30, 60, n), air(P, 0.06),
        dry=r.uniform(0.15, 0.45, n), hide=0.45)

    # their reflections drawn down into the water, faint, the nearer ship's darker
    paths, w, tone = [], [], []
    mx, wl, top, hw, bow, stern = NEAR
    for bow_, stern_, deck, masts in SHIPS + [(bow, stern, wl - 20, [(mx, top)])]:
        near = deck > QUAY
        for x_, top_ in masts:
            for k in range(2 + near):
                y0 = (wl if near else QUAY) + 10 + k * r.uniform(60, 120)
                paths.append(line((x_ + r.normal(0, 3), y0), (x_ + r.normal(0, 6), y0 + r.uniform(50, 130) * (1 + near)), 4, r.normal(0, 4)))
                w.append((hw if near else 6.0) * r.uniform(0.6, 1.0))
                tone.append(0.15 if near else 0.35)
        for k in range(3):
            y0 = (wl + 4 if near else QUAY + 14) + 16 * k
            paths.append(line((bow_ + 40 + r.normal(0, 30), y0), (stern_, y0 + r.normal(0, 3)), 6, r.normal(0, 3)))
            w.append((14 - 3 * k) * (1 + 0.3 * near))
            tone.append((0.0 if near else 0.15) + 0.1 * k)
    for cx, foot, top_, hw_, smoke in CHIMNEYS:
        paths.append(line((cx, QUAY + 8), (cx + r.normal(0, 5), QUAY + 0.4 * (foot - top_)), 5, r.normal(0, 4)))
        w.append(0.8 * hw_)
        tone.append(0.3)
    lay(paths, np.array(w), loads(FAR, np.array(tone), r, SEA[1:4], 0.4), hide=0.45, dry=0.25, merge=8, spent=0.9, pickup=0.5)

    # the water in level touches, wet into wet: long and quiet, then shorter where it is broken,
    # small far off and larger toward us
    P = strew(110, sea, 2.0)
    n = len(P)
    s = 0.45 + 1.1 * deep[at(P)]
    lay(go(P, r.uniform(150, 450, n) * s, r.normal(0, 0.025, n), 7, r.normal(0, 0.001, n)), r.uniform(10, 22, n) * s,
        water(P, 0.0, 0.07), dry=r.uniform(0.1, 0.4, n), hide=0.45, merge=3, taper=0.45, ends=(0.6, 0.5), spent=0.75)
    P = strew(60, sea * noise.smoothstep(-0.8, 0.8, busy) * (0.3 + 0.7 * deep), 1.5)
    n = len(P)
    s = 0.45 + 1.1 * deep[at(P)]
    lay(go(P, r.uniform(50, 200, n) * s, r.normal(0, 0.03, n), 6, r.normal(0, 0.0015, n)), r.uniform(4, 10, n) * s,
        water(P, 0.0, 0.1), dry=r.uniform(0.15, 0.5, n), hide=0.6, merge=2, thick=0.004, grooves=0.08, fray=2, spent=0.8,
        taper=0.5)
    P = strew(200, sea * (1 - deep) * (Y > QUAY + 30), 1.5)       # the mist on the water catching the light
    n = len(P)
    lay(go(P, r.uniform(80, 260, n), r.normal(0, 0.02, n), 5), r.uniform(3, 6, n), water(P, 0.08),
        dry=r.uniform(0.3, 0.6, n), hide=0.4, merge=4, fray=2, taper=0.5)
    P = strew(80, sea * glint, 1.8)                                 # and the glow in it, pale rose about the sun's road
    n = len(P)
    s = 0.45 + 1.1 * deep[at(P)]
    lay(go(P, r.uniform(40, 200, n) * s, r.normal(0, 0.03, n), 5, r.normal(0, 0.0015, n)), r.uniform(3, 8, n) * s,
        loads(fine(ROSE), r.uniform(0.6, 0.95, n), r, ORANGE[4:], 0.1), dry=r.uniform(0.3, 0.55, n), hide=0.4, merge=2,
        thick=0.004, fray=2, spent=0.8, taper=0.5)

    # the ripples: dark dashes gathered where the water is broken, a few strays between, small and
    # faint far off, larger and darker toward us, no two alike; and pale touches among them
    dash, pale = [], []

    def touch(x, y, s, tilt, store, size=1.0):
        d = np.clip((y - QUAY) / (H - QUAY), 0, 1)
        if abs(x - SX - sway[int(np.clip(y, 0, H - 1))]) < 45 + 110 * d and r.random() < 0.85:
            return
        L, hw, a = 85 * s * size * np.exp(r.normal(0, 0.45)), 7 * s * size * np.exp(r.normal(0, 0.3)), tilt + r.normal(0, 0.05)
        e = 0.5 * L * np.array([np.cos(a), np.sin(a)])
        store.append((line((x, y) - e, (x, y) + e, 5, r.normal(0, 0.02 * L))[::r.choice([-1, 1])], hw, d))

    for _ in range(18):
        cy = QUAY + 80 + (H - QUAY - 80) * r.random() ** 0.6
        s = 0.3 + 1.2 * (cy - QUAY) / (H - QUAY)
        cx, tilt = r.uniform(-100, W + 100), r.normal(0, 0.05)
        for _ in range(r.integers(3, 10)):
            touch(cx + r.normal(0, 170 * s), cy + r.normal(0, 28 * s), s, tilt, dash)
        for _ in range(r.integers(1, 4)):
            touch(cx + r.normal(0, 200 * s), cy + r.normal(0, 35 * s), s, tilt, pale, 0.7)
    for _ in range(34):
        y = QUAY + 80 + (H - QUAY - 80) * r.random() ** 0.7
        touch(r.uniform(-50, W + 50), y, 0.3 + 1.2 * (y - QUAY) / (H - QUAY), r.normal(0, 0.05), dash, 0.8)
    for lo, hi, hide, t0, t1 in ((0, 0.35, 0.45, 0.5, 0.9), (0.35, 0.65, 0.65, 0.4, 0.8), (0.65, 1.01, 0.85, 0.25, 0.65)):
        band = [(p, w) for p, w, d in dash if lo <= d < hi]
        if band:
            paths, w = zip(*band)
            n = len(paths)
            lay(list(paths), np.array(w), mixed((DEEP, TEAL), (r.random(n) < 0.5).astype(int), r.uniform(t0, t1, n), r,
                                                SEA[1:3], 0.3),
                dry=r.uniform(0.05, 0.4, n), hide=hide, thick=0.015, grooves=0.15, pickup=0.35, merge=1.5, taper=0.5, land=0.3,
                ends=(0.25, 0.0), fray=2.0, spent=0.9, tails=0.35)
    paths, w, d = zip(*pale)
    n = len(paths)
    lay(list(paths), np.array(w), mixed((fine(LILAC), fine(MIST)), (r.random(n) < 0.5).astype(int), r.uniform(0.75, 0.95, n), r),
        dry=r.uniform(0.25, 0.5, n), hide=0.5, merge=2, thick=0.004, fray=2, spent=0.8, taper=0.5)

    # the boats, a few dark strokes each, fainter as they go back into the mist
    for x, y, L, haze, rowers in BOATS:
        dark, figs, edge, refl = boat(x, y, L, rowers, r)
        for strokes, fam, kw in ((dark, DARK, dict(hide=0.95, dry=0.05, spent=0.4, merge=3, pickup=0.35, fray=1.4, tails=0.3)),
                                 (figs, DARK, dict(hide=0.95, dry=0.0, spent=0.15, merge=2, pickup=0.3, fray=0.8, tails=0.0)),
                                 (edge, FAR, dict(hide=0.55, dry=0.45, spent=0.8, merge=2, pickup=0.3, fray=1.2, tails=0.5)),
                                 (refl, DARK, dict(hide=0.65, dry=0.3, spent=0.8, merge=4, pickup=0.45, fray=1.6, tails=0.6))):
            paths, w, tone = zip(*strokes)
            cols, share = loads(fam, np.array(tone), r, DEEP[:3], 0.3)
            cols = cols + (ramp(MIST, 0.55) - cols) * haze
            kw["hide"] *= 1 - 0.3 * haze
            lay(list(paths), np.array(w), (cols, share), thick=0.03, grooves=0.25, taper=0.45, ends=(0.25, 0.05), **kw)

    # the sun, and its reflection: a column of short loaded strokes, close together under the far
    # side and spreading as they come toward us
    before = height.copy()
    sun(rgb, height, r)
    paths, w, y = [], [], QUAY + 24.0
    while y < H - 70:                                       # row by row down the water, one, two or three strokes to a row
        d = (y - QUAY) / (H - QUAY)
        cx = SX + sway[int(y)] + r.normal(0, 6 + 30 * d)
        for _ in range(r.choice([1, 1, 2, 2, 3])):
            L = r.uniform(20, 80) * (0.5 + 1.4 * d)
            x0, yy, t = cx + r.normal(0, 12 + 45 * d) - L / 2, y + r.normal(0, 2 + 3 * d), r.normal(0, 0.05)
            paths.append(line((x0, yy), (x0 + L, yy + t * L), 5, r.normal(0, 1.5 + 3 * d))[::r.choice([-1, 1])])
            w.append(r.uniform(2.5, 6.5) * (0.6 + 1.0 * d))
        y += r.uniform(6, 16) * (0.6 + 2.2 * d)
    k = len(paths)
    lay(paths, np.array(w), loads(ORANGE, r.uniform(0.2, 0.85, k), r, ORANGE[4:], 0.4, 0.12), how=LOADED,
        dry=r.uniform(0.0, 0.45, k), pickup=0.2, taper=0.6, tails=1.0, fray=1.5, ends=(0.1, 0.0))

    # thinned with turpentine, the paint sinks into the canvas as it dries, thinnest on the tops of
    # the threads; only the orange stands up
    loaded = height - before
    thin = noise.smoothstep(0.6, 0.1, loaded)
    weave = ground.tooth - ndimage.gaussian_filter(ground.tooth, 2.0)
    rgb *= (1 + 0.3 * thin * weave)[..., None]
    height = tooth + 0.25 * (before - tooth) + loaded
    height = ndimage.gaussian_filter(height, 0.5)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.6, gloss=0, reach=(0.86, 1.12))
    return img + 0.05 * impasto.glints(height, LIGHT)[..., None]
