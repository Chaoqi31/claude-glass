"""The Gate in the Snow. Oil on canvas, thick paint laid wet into wet and then over dry.

In the winter of 1868-69 Monet was at Étretat on the Channel coast and painted the snow out of doors. In The
Magpie a black bird sits on a gate in a wattle fence, the only dark in a canvas that is almost all light:
fresh snow under a low sun, and the shadows of the fence lying across it toward us. He painted the
shadows as colour, blue and violet and lilac, not grey, and the lit snow in cream, rose and pale yellow,
in thick strokes laid side by side and over each other, so that the paint has the body of the snow. The
jury of the Salon of 1869 turned it down.

Here a lane comes up through a gap in a wattle fence toward a farmhouse. The sun is low behind the fence
on the left, so everything throws its shadow toward us, where the sun sends it: the hurdles, their
stakes, the gateposts, the trunks of a row of trees, the house, and the bird on the gatepost.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Gate in the Snow"
DATE = "2026"
MEDIUM = "Oil on canvas, thick paint laid wet into wet and then over dry, the last strokes dragged across the crust"
AFTER = "Claude Monet, The Magpie, Étretat, 1868–69"
ROOM = "Open Air"
YEAR = 1869
PLACE = "Étretat"
REGION = "Europe"
NOTE = ("A lane through a wattle fence in fresh snow under a low winter sun, a magpie on the gatepost. "
        "Every shadow falls where the sun throws it, in blue and violet; the lit snow is cream and pale gold.")

H, W = 2000, 2900            # about 89 by 130 cm
HZ, F, CAM = 800.0, 3000.0, 3.0   # the horizon, the eye's reach in px, its height above the snow in metres
CX = W / 2
AZ, EL = np.radians(27.0), np.radians(12.0)   # the sun, low behind the fence on the left
SIN, COS, RUN = np.sin(AZ), np.cos(AZ), np.tan(EL)
SUN = (CX - F * np.tan(AZ), HZ - F * RUN / COS)   # just off the top left corner; the shadows run from under it
GATE = (-2.3, 0.5)           # the gateposts, metres across
LIGHT = (-0.6, -0.5, 0.62)
DRY = 0.002                  # paint left between sittings; a trace of wet, since lay() reads a lane with none at all as black
LUMA = dabs.LUMA
# trees: across, away, trunk radius, height of the fork, reach of the crown (metres)
TREES = [(-6.0, 19.0, 0.24, 5.2, 4.6), (-13.0, 30.0, 0.2, 4.6, 4.0), (-11.5, 41.0, 0.18, 4.4, 3.8),
         (-15.0, 58.0, 0.18, 4.2, 3.8), (-19.5, 80.0, 0.17, 4.0, 3.6),
         (2.8, 64.0, 0.2, 3.6, 3.8), (10.0, 88.0, 0.3, 5.5, 8.0), (17.0, 93.0, 0.3, 5.8, 8.5),
         (24.5, 86.0, 0.28, 5.2, 7.5), (31.0, 80.0, 0.25, 4.6, 6.0)]
HOUSE = dict(at=(8.0, 68.0), turn=np.radians(14), long=17.0, deep=8.0, eaves=3.6, ridge=7.6)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
SKY = pal("#c5c3bf", "#ccc9c2", "#d4d0c5", "#dcd7c7", "#e4ddc8", "#ebe2c6")
HAZE = lin("#dcd6d3")
HILLS = pal("#b0adc0", "#bcb8c8", "#cbc6cf")
LIT = pal("#e7d9c3", "#eee0c8", "#f2e6cf", "#f5ebd8")                # snow in the sun: cream
PEACH = pal("#e8cfbb", "#eed8c4", "#f2dfcc")                          # rose
GOLD = pal("#e8d6aa", "#eedfb6", "#f3e7c4")                           # and where it turns to the sun, pale gold
SHADE = pal("#8a92c3", "#97a0cb", "#a4acd2", "#b2b8d8", "#c0c3dc")   # in shadow: cobalt and violet, let down
LILAC = pal("#a7a2c6", "#b3accc", "#bfb7d2", "#cbc3d8")
COBALT = pal("#8b9dcd", "#9fb0d6")
HURDLE = pal("#383238", "#443c42", "#544950", "#695d64", "#83777e")              # the wattle, against the light
STAKE = pal("#40373a", "#56484b", "#6e5d59", "#9a7b62")
GABLE = pal("#c9a98d", "#d8b99b", "#e3c8a8")                          # plaster in the sun
WALL = pal("#8f879c", "#9d95a8", "#ada3ad", "#bbaea9")                # and in shade, warmer where the snow lights it
ROOF = pal("#a3a6c3", "#afb2cb", "#bbbed3", "#c8c8d9")
BRICK = pal("#87606a", "#9e6e65", "#b5836b")
TIMBER = pal("#5a5163", "#6b6173", "#7d7385")
BARK = pal("#51485a", "#62586a", "#766b7a", "#8d8290")
CROWN = pal("#b2adc1", "#c0bbcb", "#cdc8d5", "#dad5dd", "#e8e0d3")
BIRD = pal("#0e0f15", "#16171f", "#20222d")
SHEEN = pal("#161c2a", "#18232b", "#1f2638")
FLANK = pal("#b3b9cd", "#c6cbdb", "#d8dbe6")
ROSE = pal("#b7a1bf", "#c4abc3", "#d0b7c8")                          # warm light thrown back into the shadows
RIM = pal("#e3cfa7", "#eedfbb", "#f5ead0")


def mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(np.asarray(t).astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def see(X, Y, Z):
    """Where a point of the scene (X across, Y up, Z away, metres) falls on the canvas."""
    return CX + F * np.asarray(X) / Z, HZ + F * (CAM - np.asarray(Y)) / Z


def back(x, y, Z):
    """The point of the scene at depth Z that falls at (x, y) on the canvas. -> X, Y"""
    return (np.asarray(x) - CX) * Z / F, CAM - (np.asarray(y) - HZ) * Z / F


def cast(X, Y, Z):
    """Where the sun throws a point Y metres up onto the snow."""
    k = np.asarray(Y) / RUN
    return X + k * SIN, Z - k * COS


def sheet():
    im = Image.new("F", (W, H), 0.0)
    return im, ImageDraw.Draw(im)


def ground(d, Xs, Zs, fill=1.0):
    """A shape lying on the snow, drawn through its corners (metres)."""
    x, y = see(Xs, 0, np.maximum(np.asarray(Zs), 0.5))
    d.polygon(list(zip(x, y)), fill=fill)


def streak(d, X, Z, rho, fill=1.0):
    """The shadow (or the light) the low sun throws from something small and round `rho` metres across:
    on the snow it lies drawn out toward us, 1/sin(elevation) times as long as it is wide."""
    a = np.linspace(0, 2 * np.pi, 11)[:-1]
    along = rho / np.sin(EL) * np.cos(a)
    ground(d, X + along * SIN + rho * np.sin(a) * COS, Z - along * COS + rho * np.sin(a) * SIN, fill)


def lane(Z):
    """The middle of the lane and its half-width (m), from the foot of the canvas up to the house."""
    return (np.interp(Z, [7, 10, 14.8, 22, 35, 55, 72], [-1.9, -1.6, -0.9, 0.3, 2.6, 5.2, 6.4]),
            np.interp(Z, [7, 15, 70], [1.5, 1.4, 1.15]))


def fence(r):
    """The wattle fence: stakes at uneven spacing on a line a little out of square to the canvas, hurdles
    woven between them to about 0.7 m, each a little higher or lower than the next and sagging between its
    stakes, a cap of snow along the top, and the gap where the lane goes through between two stout posts."""
    X = np.arange(-17, 13, 0.02)
    Z = 14.8 + 0.035 * X + 0.12 * np.sin(0.45 * X + 1.0)
    stakes, x = [], -17 + r.uniform(0, 1.5)
    while x < 13:
        if not GATE[0] - 0.5 < x < GATE[1] + 0.5:
            stakes.append((x, r.uniform(0.035, 0.05), r.uniform(0.82, 1.0), r.normal(0, 0.03)))
        x += r.uniform(1.5, 2.1)
    stakes += [(GATE[0], 0.09, 1.2, -0.02), (GATE[1], 0.085, 1.14, 0.03)]
    sx = np.sort([s[0] for s in stakes])
    i = np.searchsorted(sx, X).clip(1, len(sx) - 1)
    f = np.clip((X - sx[i - 1]) / (sx[i] - sx[i - 1]), 0, 1)        # how far across its hurdle
    top = (0.7 + r.normal(0, 0.04, len(sx) + 1)[i] + r.normal(0, 0.025, len(sx) + 1)[i] * (f - 0.5)
           - 0.035 * np.sin(np.pi * f) + 0.01 * noise.line1d(len(X), 12, r))
    top[(X > GATE[0]) & (X < GATE[1])] = 0
    cap = (0.075 + 0.025 * noise.line1d(len(X), 18, r)) * (top > 0)
    return dict(X=X, Z=Z, top=top, cap=cap, stakes=stakes)


def zf(fen, X):
    return float(np.interp(X, fen["X"], fen["Z"]))


def house():
    """The farmhouse's footprint: front left, front right, back right, back left; and the ends of the ridge."""
    (ax, az), t = HOUSE["at"], HOUSE["turn"]
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    A = np.array([ax, az])
    B, D = A + HOUSE["long"] * u, A + HOUSE["deep"] * v
    return A, B, B + HOUSE["deep"] * v, D, A + HOUSE["deep"] / 2 * v, B + HOUSE["deep"] / 2 * v


def chimneys():
    """Two brick stacks on the ridge, one at the gable end: (X, Z, half-width m, top m)."""
    A, B, C, D, RA, RB = house()
    return [(*(RA + 0.05 * (RB - RA)), 0.3, HOUSE["ridge"] + 1.0), (*(RA + 0.7 * (RB - RA)), 0.26, HOUSE["ridge"] + 0.8)]


def limbs(r):
    """Each tree drawn the way it grows: the trunk to the fork, then boughs that part and part again,
    each shorter and thinner, rising toward the light and bowed a little at the ends by the snow.
    -> [(path (6, 2) px, half-width px, order 0 trunk .. 4 twig, tree)]"""
    out = []
    for t, (X, Z, girth, fork, reach) in enumerate(TREES):
        s = F / Z
        x0, y0 = see(X, 0, Z)
        x1, y1 = see(X + r.normal(0, 0.12), fork, Z)
        q = np.linspace(0, 1, 6)
        out.append((np.stack([x0 + (x1 - x0) * q + r.normal(0, 0.03) * s * np.sin(np.pi * q), y0 + (y1 - y0) * q], 1),
                    girth * s, 0, t))

        def grow(p, a, L, w, k):
            ang = a + r.normal(0, 0.3) * q + 0.07 * k * q ** 2 * np.sign(np.cos(a))
            pts = p + np.cumsum(np.vstack([[0, 0], np.stack([np.cos(ang[:-1]), np.sin(ang[:-1])], 1) * L / 5]), 0)
            out.append((pts, max(w, 0.7), k, t))
            if k < 4:
                for j in range(r.integers(2, 4)):
                    b = ang[-1] + r.choice([-1, 1]) * r.uniform(0.25, 0.7)
                    b += 0.2 * ((-np.pi / 2 - b + np.pi) % (2 * np.pi) - np.pi)   # turning up toward the light
                    grow(pts[-1], b, L * r.uniform(0.6, 0.8) * (1.15 if j == 0 else 1), w * r.uniform(0.45, 0.6), k + 1)

        for j in range(r.integers(3, 5)):
            grow(np.array([x1, y1]), -np.pi / 2 + r.normal(0, 0.55), reach * s * r.uniform(0.35, 0.5), girth * s * 0.6, 1)
    return out


def magpie(fen):
    """The bird on the left gatepost, facing the sun, in the strokes a painter would put it down in: the
    body, the wing over it, the white of the flank and of the shoulder, the head and the bill, the long
    tail hanging behind the post, the legs, and the light along its crown and back.
    -> [(path px, half-width px, what it is)], and how far away it sits"""
    Z = zf(fen, GATE[0])
    fx, fy = see(GATE[0] - 0.01, 1.26, Z)
    s = 1.15 * F / Z / 200          # a little larger than life, as a painter sees what holds his eye
    parts = [([(-20, -26), (-4, -26), (12, -17)], 11.0, "body"),
             ([(-12, -28), (4, -27), (20, -19)], 6.5, "wing"),
             ([(-15, -16), (-4, -13), (8, -12)], 6.5, "flank"),
             ([(-13, -31), (2, -27.5)], 2.6, "shoulder"),
             ([(-21, -34), (-27, -37), (-33, -37.5)], 6.8, "head"),
             ([(-34, -37.5), (-43, -36)], 2.2, "bill"),
             ([(12, -19), (36, -5), (63, 12)], 4.6, "tail"),
             ([(-3, -11), (-4, 0)], 1.3, "legs"),
             ([(-31, -43), (-20, -40), (-6, -37)], 1.2, "rim")]
    out = []
    for pts, w, kind in parts:
        p = np.asarray(pts, np.float32)
        q = np.linspace(0, 1, 6)[:, None]
        path = (p[0] + (p[-1] - p[0]) * q) if len(p) == 2 else \
            ((1 - q) ** 2 * p[0] + 2 * q * (1 - q) * (2 * p[1] - 0.5 * (p[0] + p[2])) + q ** 2 * p[2])
        out.append((np.array([fx, fy]) + s * path, s * w, kind))
    return out, Z


def shadows(fen, bird, r):
    """Where the snow lies in shadow (0 sun .. 1 shade), thrown the way the sun throws it: the hurdles,
    with light coming through the weave here and there, their stakes and the gateposts and the bird; the
    trunks, and the crowns of the trees, broken by the light between the twigs; the house and its stacks."""
    im, d = sheet()
    X, Z, top, cap = fen["X"], fen["Z"], fen["top"], fen["cap"]
    for run in (X < GATE[0], X > GATE[1]):
        cx, cz = cast(X[run], top[run] + cap[run], Z[run])
        ground(d, np.r_[X[run], cx[::-1]], np.r_[Z[run], cz[::-1]])
    across = np.array([COS, SIN])

    def bar(X0, Z0, rad, Y, lean):
        p, q = np.array([X0, Z0]), np.array(cast(X0 + lean, Y, Z0))
        c = np.array([p - rad * across, p + rad * across, q + 0.8 * rad * across, q - 0.8 * rad * across])
        ground(d, c[:, 0], c[:, 1])

    for x0, rad, h, lean in fen["stakes"]:
        bar(x0, zf(fen, x0), rad, h + 0.05, lean * h)
    run = np.nonzero(top > 0)[0]
    for k in r.choice(run, 260):                 # light through the weave
        streak(d, *cast(X[k], r.uniform(0.1, 0.85) * top[k], Z[k]), r.uniform(0.008, 0.02), 0.0)
    for X0, Z0, girth, fork, reach in TREES:
        bar(X0, Z0, girth, fork, 0.0)
    A, B, C, D, RA, RB = house()
    P = np.array([A, B, C, D] + [cast(p[0], HOUSE["eaves"], p[1]) for p in (A, B, C, D)]
                 + [cast(p[0], HOUSE["ridge"], p[1]) for p in (RA, RB)])
    hull = P[ConvexHull(P).vertices]
    ground(d, hull[:, 0], hull[:, 1])
    for x0, z0, hw, h in chimneys():
        bar(x0, z0, hw, h, 0.0)
    strokes, Zb = bird
    for path, w, kind in strokes:
        if kind not in ("rim", "shoulder", "flank"):
            for x, y in path:
                streak(d, *cast(*back(x, y, Zb), Zb), w * Zb / F)
    hard = ndimage.gaussian_filter(np.asarray(im, np.float32), 1.2)
    im, d = sheet()                              # the crowns: twigs heavy with snow, and light between them
    for X0, Z0, girth, fork, reach in TREES:
        n = int(260 * reach)
        p = r.normal(0, 1, (n, 3))
        p *= (r.uniform(0, 1, n) ** 0.5 / np.linalg.norm(p, axis=1))[:, None] * [reach, 0.75 * reach, reach]
        cx, cz = cast(X0 + p[:, 0], fork + 0.9 * reach + p[:, 1], Z0 + p[:, 2])
        for a, b, rho in zip(cx, cz, r.uniform(0.03, 0.09, n)):
            if b > 6.5 and -200 < CX + F * a / b < W + 200:
                streak(d, a, b, rho)
    soft = ndimage.gaussian_filter(np.asarray(im, np.float32), 3.0)
    return np.clip(np.maximum(hard, 0.5 * soft), 0, 1)


def surface(X, Z, r):
    """The snow, in metres: long low drifts across the wind, the lane worn into it with a bank along each
    side, and two wheel ruts down it."""
    h = np.zeros_like(X)
    for lam in r.uniform(1.2, 9.0, 12):
        th = r.normal(0.4, 0.5)
        fade = noise.smoothstep(3, 9, lam * F * CAM / Z ** 2)            # drifts too small to see far off are lost
        h += 0.008 * lam * fade * np.sin(2 * np.pi / lam * (X * np.cos(th) + Z * np.sin(th)) + r.uniform(0, 2 * np.pi))
    mid, hw = lane(Z)
    dd = X - mid
    h += -0.1 * noise.smoothstep(hw, 0.6 * hw, np.abs(dd)) + 0.09 * np.exp(-((np.abs(dd) - hw) / 0.35) ** 2)
    for side in (-1, 1):
        off = side * 0.72 + 0.05 * np.sin(0.3 * Z + side)
        h -= 0.06 * np.exp(-((dd - off) / 0.08) ** 2) * noise.smoothstep(2, 6, 0.08 * F * CAM / Z ** 2)
    return h


def design(r):
    """What each place adds up to, seen from across the room, and the maps the brush goes by."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    below = yy > HZ + 0.5
    Z = np.where(below, F * CAM / np.maximum(yy - HZ, 0.5), 1e4).astype(np.float32)
    X = ((xx - CX) * Z / F).astype(np.float32)
    fen = fence(r)
    bird = magpie(fen)
    sk = limbs(r)
    shade = shadows(fen, bird, r) * below

    # the snow in the sun: slopes turned to it catch more, those turned away fall into their own shade
    h = surface(np.where(below, X, 0), np.where(below, np.minimum(Z, 400), 400), r)
    hy, hx = np.gradient(h)
    hX = hx * F / Z
    dZ = -Z ** 2 / (F * CAM)
    hZ = (hy - hX * (xx - CX) / F * dZ) / dZ
    sun = np.clip(1 - (-hX * SIN + hZ * COS) / RUN, 0, 2.2) * (1 - shade)
    far = noise.smoothstep(25, 160, Z)
    toward = np.exp(-((xx - SUN[0]) / 1800) ** 2) * below                # the snow toward the sun glows gold
    patch = noise.field((H, W), 300, r)
    light = mix(ramp(LIT, 0.55 + 0.25 * patch), ramp(PEACH, 0.5 + 0.3 * patch),
                noise.smoothstep(-0.5, 1.2, noise.field((H, W), 500, r)))
    light = mix(light, ramp(GOLD, 0.6), 0.6 * toward * (0.4 + 0.6 * far))
    light = mix(light, lin("#f7efdc"), 0.5 * noise.smoothstep(1.1, 2.0, sun))
    dark = mix(ramp(SHADE, 0.35 + 0.25 * patch + 0.4 * far), ramp(LILAC, 0.5 + 0.4 * far),
               noise.smoothstep(-0.8, 1.0, noise.field((H, W), 400, r)))
    snow = mix(dark, light, noise.smoothstep(0.0, 1.0, sun))
    snow = mix(snow, HAZE, 0.5 * noise.smoothstep(40, 400, Z))

    # the sky: pale yellow-grey, lightest and warmest toward the sun, a little greyer and cooler away from it
    ds = np.hypot(xx - SUN[0], (yy - SUN[1]) * 1.3)
    sky = ramp(SKY, 0.12 + 0.25 * noise.smoothstep(0, HZ, yy) + 0.65 * np.exp(-ds / 1300))
    sky = mix(sky, lin("#c9c7cb"), 0.45 * noise.smoothstep(800, 2900, xx) * noise.smoothstep(HZ, 0, yy))   # cooler away from it
    sky = mix(sky, lin("#e6e0cf"), 0.55 * np.exp(-((yy - HZ + 40) / 90) ** 2))
    want = np.where(below[..., None], snow, sky)
    crest = HZ - 10 - 22 * noise.smoothstep(-0.8, 1.8, noise.line1d(W, 700, r)) - 5 * noise.line1d(W, 50, r)
    hills = noise.smoothstep(-1.5, 1.5, yy - crest[None]) * ~below
    want = mix(want, mix(ramp(HILLS, 0.5 + 0.3 * noise.field((H, W), 200, r)), HAZE, 0.35), hills)

    # the house
    A, B, C, Dd, RA, RB = house()
    P = lambda p, y: tuple(map(float, see(p[0], y, p[1])))
    e, g = HOUSE["eaves"], HOUSE["ridge"]
    faces = {}
    for name, poly in (("gable", [P(A, 0), P(Dd, 0), P(Dd, e), P(RA, g), P(A, e)]),
                       ("wall", [P(A, 0), P(B, 0), P(B, e), P(A, e)]),
                       ("roof", [P(A - (B - A) * 0.02, e - 0.35), P(B, e - 0.35), P(RB, g + 0.15), P(RA, g + 0.15)])):
        im, d = sheet()
        d.polygon(poly, fill=1.0)
        faces[name] = np.array(im, np.float32)
    faces["wall"] *= 1 - faces["gable"]
    faces["roof"] = np.clip(faces["roof"] - faces["gable"] * (yy > see(0, e - 0.35, A[1])[1]), 0, 1)
    im, d = sheet()
    for cx_, cz_, hw, top in chimneys():
        xa, ya = see(cx_ - hw, g - 0.6, cz_)
        xb, yb = see(cx_ + hw, top, cz_)
        d.rectangle([xa, yb, xb, ya], fill=1.0)
    faces["stack"] = np.asarray(im, np.float32)
    up = noise.smoothstep(see(0, 0, A[1])[1], see(0, e, A[1])[1], yy)       # 0 at the foot of the wall, 1 at the eaves
    hz = noise.field((H, W), 60, r)
    want = mix(want, ramp(WALL, 0.75 - 0.5 * up + 0.08 * hz), faces["wall"])
    want = mix(want, ramp(GABLE, 0.55 + 0.1 * hz), faces["gable"])
    want = mix(want, ramp(ROOF, 0.45 + 0.3 * noise.smoothstep(see(0, e, A[1])[1], see(0, g, RA[1])[1], yy) + 0.08 * hz),
               faces["roof"])
    want = mix(want, ramp(BRICK, 0.55), faces["stack"])

    # the crowns of the trees: twigs heavy with snow, lit where they turn to the sun
    im, d = sheet()
    di, dd = sheet()
    for path, w, k, t in sk:
        if k >= 2:
            s = F / TREES[t][1]
            for x, y in path[::2]:
                rad = 0.3 * s * r.uniform(0.6, 1.3)
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=1.0)
        a = np.arctan2(*(path[-1] - path[0])[::-1]) % np.pi
        dd.line([tuple(q) for q in path], fill=float(a) + 1, width=max(3, int(2 * w)))
    crown = ndimage.gaussian_filter(np.asarray(im, np.float32), 6) * ~below
    lines = np.asarray(di, np.float32)
    idx = ndimage.distance_transform_edt(lines == 0, return_distances=False, return_indices=True)
    twig = (lines[idx[0], idx[1]] - 1).astype(np.float32)                    # the way the nearest branch runs
    glow = np.exp(-np.hypot(xx - SUN[0], yy - SUN[1]) / 1200)
    want = mix(want, ramp(CROWN, 0.5 + 0.2 * noise.field((H, W), 40, r) + 0.35 * glow),
               0.6 * noise.smoothstep(0.15, 0.6, crown))

    # the fence against the light: the faces of the hurdles in their own shadow, the caps of snow along
    # their tops, the stakes and the gateposts
    Xf, Zf, top, cap = fen["X"], fen["Z"], fen["top"], fen["cap"]
    masks = {}
    for name in ("hurdle", "cap", "stake"):
        masks[name] = sheet()
    for run in (Xf < GATE[0], Xf > GATE[1]):
        xb, yb = see(Xf[run], 0, Zf[run])
        xt, yt = see(Xf[run], top[run], Zf[run])
        xc, yc = see(Xf[run], top[run] + cap[run], Zf[run])
        masks["hurdle"][1].polygon(list(zip(np.r_[xb, xt[::-1]], np.r_[yb, yt[::-1]])), fill=1.0)
        masks["cap"][1].polygon(list(zip(np.r_[xt, xc[::-1]], np.r_[yt + 3, yc[::-1]])), fill=1.0)
    for x0, rad, hgt, lean in fen["stakes"]:
        z0 = zf(fen, x0)
        w = rad * F / z0
        xa, ya = see(x0, 0, z0)
        xb, yb = see(x0 + lean * hgt, hgt, z0)
        masks["stake"][1].polygon([(xa - w, ya), (xa + w, ya), (xb + 0.85 * w, yb), (xb - 0.85 * w, yb)], fill=1.0)
    masks = {k: np.asarray(v[0], np.float32) for k, v in masks.items()}
    want = mix(want, ramp(HURDLE, 0.4 + 0.15 * noise.field((H, W), 30, r)), masks["hurdle"])
    want = mix(want, ramp(LIT, 0.6), masks["cap"])
    want = mix(want, ramp(STAKE, 0.35), masks["stake"])
    return dict(want=want.astype(np.float32), X=X, Z=Z, below=below, shade=shade, sun=sun, fen=fen, bird=bird,
                sk=sk, faces=faces, crown=crown, twig=twig, toward=toward, hills=hills, h=h, **masks)


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def valued(c, lum):
    """A colour brought to the value `lum` the way a painter does it: let down with white to go lighter,
    deepened to go darker, its hue kept."""
    lc = (c @ LUMA)[..., None]
    lum = np.asarray(lum)[..., None]
    return np.where(lum >= lc, 1 - (1 - c) * (1 - lum) / (1 - lc + 1e-6), c * lum / (lc + 1e-6)).clip(0, 1)


def paint(seed=1869):
    r = noise.rng(seed)
    D = design(r)
    want, Z, below, shade, sun, faces, fen = (D[k] for k in ("want", "Z", "below", "shade", "sun", "faces", "fen"))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cloth = canvas.duck((H, W), seed, tint="#d8d0bf", thread=3.0)

    # where each passage lies
    sky = (~below).astype(np.float32)
    home = np.clip(faces["wall"] + faces["gable"] + faces["roof"] + faces["stack"], 0, 1)
    fenced = np.clip(D["hurdle"] + D["cap"] + D["stake"], 0, 1)
    zfence = zf(fen, 0.0)
    beyond = below * (Z > zfence) * (1 - home) * (1 - fenced)
    near = below * (Z <= zfence + 0.3) * (1 - fenced)
    crowns = noise.smoothstep(0.12, 0.5, D["crown"]) * (1 - home)
    k = np.clip(F / Z / 200, 0.4, 2.2).astype(np.float32)          # the size of the brush, growing toward us

    # which way the strokes run: in the sky a slant hatching; over the snow level, along the lane where it
    # runs, and down the shadows toward us; up and down the walls and the slope of the roof; along the twigs
    radial = np.arctan2(yy - HZ, xx - SUN[0])
    Zr = np.maximum(F * CAM / np.maximum(yy[:, :1] - HZ, 0.5), 1)
    mid, hw = lane(Zr)
    xm = CX + F * mid / Zr
    along = np.arctan2(1, np.gradient(xm[:, 0]))[:, None]
    inlane = noise.smoothstep(1.2, 0.8, np.abs(D["X"] - mid) / hw) * below
    flow = 0.12 * noise.field((H, W), 300, r)
    flow = blend(flow, along, 0.45 * inlane)
    flow = blend(flow, radial, noise.smoothstep(0.3, 0.7, shade))
    turn = noise.field((H, W), 700, r)                             # the sky brushed a few ways, a patch at a time
    skyflow = blend(blend(0.04 * noise.field((H, W), 300, r), np.full((H, W), -0.4, np.float32), noise.smoothstep(0.3, 0.9, turn)),
                    np.full((H, W), 0.35, np.float32), noise.smoothstep(-0.4, -1.0, turn)).astype(np.float32)
    A, B, C, Dd, RA, RB = house()
    ax_, ay_ = see(A[0], HOUSE["eaves"], A[1])
    rx_, ry_ = see(RA[0], HOUSE["ridge"], RA[1])
    bx_, by_ = see(B[0], 0, B[1])
    fx0, fy0 = see(A[0], 0, A[1])
    flow = np.where(faces["roof"] > 0.5, np.arctan2(ry_ - ay_, rx_ - ax_), flow)
    flow = np.where(faces["wall"] > 0.5, np.arctan2(by_ - fy0, bx_ - fx0), flow)
    flow = np.where((faces["gable"] + faces["stack"]) > 0.5, np.pi / 2, flow)
    flow = np.where(crowns > 0.3, D["twig"], flow)
    flow = np.where(sky * (1 - crowns) * (1 - home) > 0.5, skyflow, flow)
    flow = np.where(D["hurdle"] + D["cap"] > 0.5, 0.0, flow)
    flow = np.where(D["stake"] > 0.5, np.pi / 2, flow).astype(np.float32)

    # the canvas rubbed over first with thin paint of the colour of each place
    rag = noise.stretched((H, W), 12, 90, r)
    rgb = (0.12 * cloth.color + 0.85 * want * (1 + 0.05 * rag[..., None])).astype(np.float32)
    height = cloth.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def broken(P, fams, which, other, pure=(0.2, 0.5), odds=0.1, value=0.05, base=None):
        """The brush for each stroke: the colour of the place pushed part of the way toward a hue of its
        family brought to the same value, a second streaked beside it, and now and then a third of another
        kind of light."""
        T = want[at(P)] if base is None else base
        n = len(P)
        lum = T @ LUMA
        sizes = np.array([len(f) for f in fams])
        offs = np.r_[0, np.cumsum(sizes)[:-1]]
        table = np.concatenate(fams)

        def hue(fi, f):
            c = valued(table[offs[fi] + (r.random(n) * sizes[fi]).astype(int)], lum * np.exp(r.normal(0, value, n)))
            return T + (c - T) * f[:, None]
        c1 = hue(which, r.uniform(*pure, n))
        c2 = hue(np.where(r.random(n) < 0.25, other, which), r.uniform(*pure, n))
        c3 = np.where((r.random(n) < odds)[:, None], hue(other, np.full(n, 0.5)), T * np.exp(r.normal(0, 0.03, (n, 1))))
        share = np.stack([r.uniform(0.5, 0.75, n), r.uniform(0.15, 0.35, n), r.uniform(0.05, 0.2, n)], 1)
        return np.stack([c1, c2, c3], 1).astype(np.float32), share

    FAMS = (LIT, PEACH, GOLD, SHADE, LILAC, COBALT, ROSE)

    def snow(P, pure=(0.2, 0.5), odds=0.12):
        """Snow: in the sun cream, rose or pale gold, in shadow cobalt, violet or lilac."""
        y, x = at(P)
        n = len(P)
        lit = r.random(n) < noise.smoothstep(0.15, 0.85, sun[y, x])
        u = r.random(n)
        gold = (u < 0.1 + 0.45 * D["toward"][y, x]) & (sun[y, x] > 0.7)     # a yellow brought down in value goes olive
        which = np.where(lit, np.where(gold, 2, np.where(u < 0.6, 0, 1)),
                         np.where(u < 0.45, 3, np.where(u < 0.85, 4, 5)))
        other = np.where(lit, r.choice([4, 4, 5, 1], n), r.choice([6, 6, 4], n))
        return broken(P, FAMS, which, other, pure, odds)

    def air(P, pure=(0.15, 0.4)):
        """The sky: yellow-grey, gold toward the sun, a little lilac away from it."""
        y, x = at(P)
        n = len(P)
        g = np.exp(-np.hypot(x - SUN[0], y - SUN[1]) / 1100)
        u = r.random(n)
        which = np.where(u < 0.7 * g ** 1.5, 1, np.where(u < 0.8 - 0.3 * g, 0, 2))
        return broken(P, (SKY, GOLD, np.concatenate([LILAC[2:], pal("#c8c6c8", "#d2cfcd")]), PEACH), which,
                      r.choice([2, 3], n), pure, 0.06, 0.025)

    def loads(colours, tone, accent=None, odds=0.2, spread=0.08):
        """A brush loaded from one palette at `tone` (0 dark .. 1 light), a neighbour streaked in, and now
        and then an accent picked up from elsewhere."""
        m, N = len(colours), len(tone)
        i = (np.clip(tone + r.normal(0, spread, N), 0, 0.999) * m).astype(int)
        j = np.clip(i + r.choice([-1, 1], N), 0, m - 1)
        third = colours[np.clip(i + r.choice([-1, 1], N), 0, m - 1)].copy()
        if accent is not None:
            hit = r.random(N) < odds
            third[hit] = accent[r.integers(0, len(accent), hit.sum())]
        share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.03, 0.2, N)], 1)
        return np.stack([colours[i], colours[j], third], 1).astype(np.float32), share

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def go(odds, region, spacing, length, width, load, scale=None, tilt=0.0, bend=0.0, dry=0.0, n=8, group=(1, 1),
           field=None, **kw):
        """A passage. The strokes go down in small groups, a few laid side by side from one loading of the
        brush at one slant, each drier than the last, as a painter hatches; then the next."""
        P = strew(spacing * np.sqrt(np.mean(group)), odds)
        G = len(P)
        if not G:
            return
        f = flow if field is None else field
        m = r.integers(group[0], group[1] + 1, G)
        own = np.repeat(np.arange(G), m)
        j = np.arange(len(own)) - np.repeat(np.cumsum(m) - m, m)
        lean = r.normal(0, tilt, G)[own]
        a = f[at(P)][own] + lean
        s = scale[at(P)][own] if scale is not None else np.ones(len(own))
        wd = r.uniform(*width, G)[own] * r.uniform(0.85, 1.15, len(own)) * s ** 0.8
        side = (j - (m[own] - 1) / 2) * 1.6 * wd
        Q = P[own] + np.stack([-np.sin(a), np.cos(a)], 1) * side[:, None] \
            + np.stack([np.cos(a), np.sin(a)], 1) * (r.normal(0, 0.4, len(own)) * wd)[:, None]
        paths = impasto.follow(f, Q, r.uniform(*length, G)[own] * r.uniform(0.75, 1.25, len(own)) * s ** 0.9, n,
                               r.normal(0, bend, G)[own], lean)
        flip = (r.random(G) < 0.5)[own]
        paths[flip] = paths[flip, ::-1]
        cols, share = load(P)
        d = (r.uniform(*dry, G) if isinstance(dry, tuple) else np.full(G, dry))[own]
        stay = region[at(Q)] > 0.5
        impasto.lay(rgb, height, paths[stay], wd[stay],
                    (cols[own] * np.exp(r.normal(0, 0.02, (len(own), 1, 3))))[stay].astype(np.float32), r,
                    share=share[own][stay], wet=wet, dry=np.clip(d + 0.08 * j, 0, 0.85)[stay], **kw)

    soft = dict(ends=(0.45, 0.3), taper=0.3, fray=2.2)
    cream = dict(ends=(0.2, 0.08), taper=0.25, fray=1.6, grooves=0.8, lips=0.18, land=0.7, lift=0.45)
    one = np.ones((H, W), np.float32)
    kb = np.maximum(k, 0.5)                                         # the smallest brush, for the distance
    thin = lambda m: (m * (0.5 / kb) ** 1.7).astype(np.float32)     # fewer strokes where they are larger
    glow = np.exp(-np.hypot(xx - SUN[0], yy - SUN[1]) / 1100)
    lit = noise.smoothstep(0.25, 0.8, sun)

    # the lay-in: the whole canvas brushed over in broad thin strokes of the colour of each place
    plain = lambda P: broken(P, FAMS, np.zeros(len(P), int), np.zeros(len(P), int), (0.0, 0.1), 0.0)
    go(one, one, 150, (100, 240), (22, 34), plain, tilt=0.2, thick=0.03, grooves=0.15, lips=0.02, land=0.1, lift=0.05,
       pickup=0.6, merge=6, spent=0.6, fray=3, taper=0.4)
    height -= 0.8 * (height - 0.2 * cloth.tooth)
    wet[:] = DRY

    def pieces(path, size, gap):
        """Cut a line into lumps along it, `size` px long with `gap` px between, the way snow lies on a bough."""
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        out, a = [], r.uniform(0, gap)
        while a < L[-1] - 2:
            b = min(L[-1], a + size * r.uniform(0.6, 1.4))
            q = np.linspace(a, b, 4)
            out.append(np.stack([np.interp(q, L, path[:, 0]), np.interp(q, L, path[:, 1])], 1))
            a = b + gap * r.uniform(0.3, 1.7)
        return out

    def trees(which):
        """The limbs of some of the trees drawn into their snowy crowns, dark against the light, the
        trunks with a line of sun down their left sides, and the snow lying in lumps along each bough."""
        dark, rims, lumps = [], [], []
        for path, w, order, t in D["sk"]:
            Zt = TREES[t][1]
            if t not in which or w < 0.6 or order > (3 if Zt < 45 else 2):
                continue
            haze = noise.smoothstep(18, 110, Zt)
            tan = np.gradient(path, axis=0)
            nrm = np.stack([tan[:, 1], -tan[:, 0]], 1) / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
            nrm *= np.where(nrm[:, 1:] > 0, -1, 1)                   # the side that faces up
            if order == 0:
                nrm *= -np.sign(nrm[:, :1] + 1e-9)                   # a trunk's sunward side is its left
                m = 1 if w < 10 else 2
                for j in range(m):
                    dark.append((path + nrm * ((j + 0.5) / m - 0.5) * w, 1.15 * w / m, 0.2 + 0.5 * haze + 0.1 * j))
                if Zt < 45:
                    rims.append((path + nrm * 0.8 * w, max(1.0, 0.16 * w), haze))
            else:
                dark.append((path, w, 0.25 + 0.55 * haze + 0.1 * order))
                if w >= 1.0:
                    lumps += [(q + nrm[0] * 0.55 * w, 0.6 * w * r.uniform(0.8, 1.3), haze)
                              for q in pieces(path, 4 * w + 6, 1.5 * w + 3)]
        if dark:
            lay([d_[0] for d_ in dark], np.array([d_[1] for d_ in dark]),
                loads(BARK, np.array([d_[2] for d_ in dark]).clip(0, 0.95), CROWN[:2], 0.15), thick=0.16, pickup=0.25,
                dry=0.1, spent=0.5, taper=0.45, ends=(0.3, 0.1), fray=1.0, tails=0.4)
        if rims:
            lay([q[0] for q in rims], np.array([q[1] for q in rims]), loads(RIM, np.array([0.2 + 0.5 * q[2] for q in rims])),
                thick=0.2, pickup=0.3, dry=0.1, spent=0.5, taper=0.3, fray=1.0)
        if lumps:
            lay([q[0] for q in lumps], np.array([q[1] for q in lumps]),
                loads(np.concatenate([CROWN[2:], RIM]), np.array([0.25 + 0.5 * q[2] for q in lumps]) + 0.3 * r.random(len(lumps)),
                      RIM, 0.3), thick=0.3, pickup=0.1, dry=r.uniform(0.0, 0.35, len(lumps)), spent=0.5, taper=0.3,
                ends=(0.5, 0.4), fray=1.2)

    # the second sitting, wet into wet: the sky in broad soft strokes, the distance, the snowy crowns and the
    # trees behind the house, the house, then the snow, small far off and large toward us
    open_sky = sky * (1 - home)
    thinned = dict(thick=0.03, grooves=0.2, lips=0.02, land=0.1, lift=0.05, ends=(0.45, 0.35), taper=0.35, fray=2.8)
    go(open_sky, open_sky, 60, (120, 320), (18, 34), air, tilt=0.1, bend=0.0007, group=(1, 3), field=skyflow,
       pickup=0.6, merge=7, dry=(0, 0.15), **thinned)
    go(D["hills"], D["hills"], 12, (30, 80), (4, 6), air, tilt=0.08, thick=0.08, pickup=0.5, merge=3, field=one * 0.0,
       **soft)
    go(crowns * 0.9, crowns, 12, (10, 26), (3.5, 7), lambda P: loads(CROWN, 0.35 + 0.3 * r.random(len(P)) + 0.3 * glow[at(P)],
                                                                     RIM, 0.25), tilt=0.6, thick=0.14, pickup=0.4, merge=2,
       dry=(0.1, 0.45), **soft)
    behind = [t for t, tr in enumerate(TREES) if tr[1] > HOUSE["at"][1] + 4]
    trees(behind)
    for face, colours, tone, tl in (("wall", WALL, 0.55, 0.3), ("gable", GABLE, 0.55, 0.2), ("roof", ROOF, 0.5, 0.2),
                                    ("stack", BRICK, 0.5, 0.1)):
        m = faces[face]
        go(m, m, 11, (18, 44), (4, 7), lambda P: loads(colours, tone + 0.15 * r.normal(0, 1, len(P))), tilt=tl,
           thick=0.1, pickup=0.5, merge=2, dry=(0, 0.25), **soft)
    go(thin(beyond), beyond, 11, (22, 50), (5, 8), snow, scale=kb, tilt=0.18, group=(1, 3), thick=0.1, pickup=0.45,
       merge=2, dry=(0, 0.3), **soft)
    body = dict(grooves=0.6, lips=0.15, land=0.5, lift=0.35, ends=(0.15, 0.05), taper=0.2, fray=1.4)
    go(thin(near), near, 26, (70, 170), (13, 20), snow, scale=kb, tilt=0.15, bend=0.0015, group=(2, 4), thick=0.24,
       pickup=0.35, merge=2, spent=0.5, dry=(0, 0.15), **body)
    face = D["hurdle"]
    rods, rodw, rodt = [], [], []
    for a0, a1 in (((-60 - CX) * zfence / F, GATE[0] - 0.06), (GATE[1] + 0.06, (W + 60 - CX) * zfence / F)):
        for hgt in np.arange(0.035, 0.75, 0.04):
            x = a0 - r.uniform(0, 1.0)
            while x < a1:
                L = r.uniform(0.5, 1.6)
                Xs = np.linspace(max(x, a0), min(x + L, a1), 7)
                tp = np.interp(Xs, fen["X"], fen["top"])
                if Xs[-1] - Xs[0] > 0.12 and hgt < tp.min() - 0.02:
                    Zs = np.interp(Xs, fen["X"], fen["Z"])
                    xs, ys = see(Xs, hgt + 0.016 * np.sin(Xs * r.uniform(8, 12) + r.uniform(0, 6.3)), Zs)   # in and out of the sails
                    rods.append(np.stack([xs, ys], 1)[::r.choice([-1, 1])])
                    rodw.append(0.018 * F / Zs.mean() * r.uniform(0.8, 1.25))
                    rodt.append(r.uniform(0.05, 0.6) + 0.3 * hgt + (0.4 if r.random() < 0.18 else 0))
                x += L * r.uniform(0.55, 0.9)
    lay(rods, np.array(rodw), loads(HURDLE, np.array(rodt).clip(0, 0.95), STAKE[1:3], 0.15), thick=0.2, pickup=0.25, merge=1,
        dry=r.uniform(0.0, 0.25, len(rods)), spent=0.6, ends=(0.3, 0.2), taper=0.3, fray=0.8, tails=0.5)
    wet[:] = DRY
    crust = 1 - open_sky * (1 - crowns)                              # the sky is thin paint and keeps none
    height += crust * (0.3 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2))    # crusts left between sittings
                       + 0.2 * noise.smoothstep(0.2, 1.8, noise.fbm((H, W), 22, r, octaves=3)))

    # the third, over dry paint: more snow in the crowns, and the sky veiled again, thin and dry, gold toward
    # the sun and a little lilac away from it
    go(crowns * 0.45, crowns, 14, (6, 16), (2.5, 5), lambda P: loads(CROWN, 0.55 + 0.25 * r.random(len(P)) + 0.3 * glow[at(P)],
                                                                     RIM, 0.3), tilt=0.7, thick=0.2, pickup=0, merge=0,
       dry=(0.25, 0.55), **soft)
    go(open_sky * (1 - crowns) * (0.25 + 0.75 * glow), open_sky, 70, (80, 220), (12, 22), lambda P: air(P, (0.3, 0.6)),
       tilt=0.15, group=(1, 3), field=skyflow, pickup=0, merge=0, dry=(0.3, 0.55), hide=0.75, **thinned)

    # the house: the shadow under the eaves, the windows and the door, the light along the verge and the ridge
    def quad(p, du, y0, y1):
        q = [see((p + c * du)[0], y, (p + c * du)[1]) for c, y in ((-1, y0), (1, y0), (1, y1), (-1, y1))]
        return np.array([(float(a), float(b)) for a, b in q])
    u = (B - A) / np.linalg.norm(B - A)
    v = (Dd - A) / np.linalg.norm(Dd - A)
    lines_, ws, tones = [], [], []
    e_, g_ = HOUSE["eaves"], HOUSE["ridge"]
    for f, half, y0, y1 in ((0.13, 0.35, 1.2, 2.0), (0.33, 0.35, 1.2, 2.0), (0.5, 0.45, 0.0, 2.0), (0.68, 0.35, 1.2, 2.0),
                            (0.86, 0.35, 1.2, 2.0)):
        c = quad(A + f * (B - A), half * u, y0, y1)
        x0, x1 = c[:, 0].min(), c[:, 0].max()
        for j in range(2):
            xj = x0 + (j + 0.5) / 2 * (x1 - x0)
            lines_.append(np.array([[xj, c[2:, 1].mean() + 1], [xj, c[:2, 1].mean() - 1]]))
            ws.append(0.3 * (x1 - x0))
            tones.append(0.25 + 0.3 * j)
    c = quad(A + 0.5 * (Dd - A), 0.35 * v, 1.3, 2.1)
    lines_.append(np.array([[c[:, 0].mean(), c[2:, 1].mean()], [c[:, 0].mean(), c[:2, 1].mean()]]))
    ws.append(0.4 * np.ptp(c[:, 0]))
    tones.append(0.5)
    lay(lines_, np.array(ws), loads(TIMBER, np.array(tones), WALL[:2], 0.3), thick=0.16, pickup=0.3, dry=0.1, ends=(0.1, 0.05),
        taper=0.1, fray=0.8)
    at_ = lambda p, y: np.array(see(p[0], y, p[1]), np.float32)
    eave = np.stack([at_(A, e_ - 0.25), at_(B, e_ - 0.25)])
    verge = np.stack([at_(A, e_ - 0.3), at_(RA, g_ + 0.1)])
    ridge = np.stack([at_(RA, g_ + 0.12), at_(RB, g_ + 0.12)])
    trio = lambda *c: np.stack(c)
    lay([eave, eave + [0, 3], verge, ridge, ridge], np.array([4.0, 3.0, 3.2, 2.2, 1.6]),
        (np.stack([trio(WALL[0], WALL[1], ROOF[0]), trio(ROOF[1], ROOF[2], WALL[1]), trio(RIM[1], RIM[2], GOLD[1]),
                   trio(RIM[0], RIM[1], CROWN[3]), trio(RIM[2], GOLD[2], RIM[1])]), np.tile([0.6, 0.3, 0.1], (5, 1))),
        thick=0.22, pickup=0.2, dry=np.array([0.1, 0.3, 0.2, 0.35, 0.5]), spent=0.8, ends=(0.2, 0.1), fray=1.2)
    caps = []
    for cx_, cz_, hw_, top_ in chimneys():
        xa, ya = see(cx_ - hw_, top_, cz_)
        xb, yb = see(cx_ + hw_, top_, cz_)
        caps.append(np.array([[xa - 1, ya], [xb + 1, ya]]))
        xl = xa + 0.3 * (xb - xa)
        caps.append(np.array([[xl, ya + 2], [xl, see(0, g_ - 0.2, cz_)[1]]]))
    lay(caps, np.array([3.0, 0.3 * (caps[0][1, 0] - caps[0][0, 0]), 3.0, 0.3 * (caps[2][1, 0] - caps[2][0, 0])]),
        (np.stack([trio(RIM[2], RIM[1], CROWN[3]), trio(BRICK[2], BRICK[1], GABLE[0])] * 2), np.tile([0.6, 0.3, 0.1], (4, 1))),
        thick=0.25, pickup=0.1, dry=0.15, ends=(0.2, 0.1), fray=0.8)

    # the snow: thick creamy strokes side by side and over each other, nearer the colours of the tubes,
    # laid across the light and down the shadows; then the ruts
    go(thin(beyond) * 0.5, beyond, 11, (18, 40), (4, 7), lambda P: snow(P, (0.35, 0.65), 0.15), scale=kb, tilt=0.35,
       group=(1, 3), thick=0.2, pickup=0, merge=0, dry=(0.1, 0.45), **cream)
    go(thin(near) * 0.8, near, 26, (50, 120), (9, 15), lambda P: snow(P, (0.35, 0.7), 0.18), scale=kb, tilt=0.35,
       bend=0.003, group=(2, 4), thick=0.3, pickup=0, merge=0, dry=(0.0, 0.3), **cream)
    ruts, rw, rl = [], [], []
    for side in (-1, 1):
        zs = np.geomspace(7.0, 68.0, 160)
        mid_, hw_ = lane(zs)
        x_, y_ = see(mid_ + side * 0.72 + 0.05 * np.sin(0.3 * zs + side), 0, zs)
        pts = np.stack([x_, y_], 1)
        cut = np.r_[0, np.sort(r.choice(np.arange(5, 155), 26, replace=False)), 159]
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a >= 2 and r.random() < 0.85:
                q = pts[a:b + 1:max(1, (b - a) // 5)]
                sc = F / zs[(a + b) // 2]
                ruts += [q, q + [0.045 * sc, 0]]
                rw += [max(1.0, 0.035 * sc), max(0.8, 0.025 * sc)]
                rl += [0, 1]
    rl = np.array(rl)
    lay(ruts, np.array(rw), (np.where(rl[:, None, None] == 0, trio(SHADE[1], SHADE[0], LILAC[1])[None],
                                      trio(GOLD[2], RIM[2], LIT[3])[None]).astype(np.float32), np.tile([0.6, 0.3, 0.1], (len(rl), 1))),
        thick=0.2, pickup=0, dry=r.uniform(0.15, 0.5, len(rl)), spent=0.8, ends=(0.2, 0.1), taper=0.5, fray=1.2, tails=0.5)

    near_trees = [t for t in range(len(TREES)) if t not in behind]   # in front of the snow beyond them, behind the fence
    trees(near_trees)
    im, d = sheet()
    for path, w, order, t in D["sk"]:
        if order == 0 and t in near_trees:
            d.line([tuple(q) for q in path], fill=1.0, width=int(2 * w + 120))
    clear = 1 - np.asarray(im, np.float32)                            # the last strokes keep off the trunks

    # the fence: snow lodged here and there in the weave, in its own shadow, and a fleck of the sunlit field
    # through it; the snow drifted against its foot; the caps along the hurdles, lit on top; the stakes and the
    # gateposts, their caps of snow, and a line of light down the side of each the sun reaches
    Xf, Zf, top, cap = fen["X"], fen["Z"], fen["top"], fen["cap"]
    xt, yt = see(Xf, top, Zf)
    pick = r.choice(len(rods), len(rods) // 10, replace=False)
    lay([rods[i][1:5] - [0, 0.6 * rodw[i]] for i in pick], np.array([0.5 * rodw[i] for i in pick]),
        loads(np.concatenate([SHADE[1:3], LILAC[:2]]), r.random(len(pick))), thick=0.22, pickup=0, dry=r.uniform(0.3, 0.6, len(pick)),
        spent=0.7, ends=(0.4, 0.3), taper=0.4, fray=1.2)
    capl, capw, capt = [], [], []              # 0 the lit top of a cap, 1 its shaded face and the lumps hanging from it, 2 the drift
    for run in (Xf < GATE[0], Xf > GATE[1]):
        idx = np.nonzero(run)[0]
        a = idx[0]
        while a < idx[-1] - 3:
            b = min(idx[-1], a + r.integers(35, 90))
            sel = np.arange(a, b + 1, max(1, (b - a) // 5))
            sc = F / Zf[a]
            lump = r.uniform(1.2, 2.0)
            xs, ys = see(Xf[sel], top[sel] + 0.6 * cap[sel], Zf[sel])
            capl += [np.stack([xs, ys], 1), np.stack([xs, ys + 0.6 * cap[a] * sc * lump], 1)]
            capw += [0.45 * cap[a] * sc * lump + 1, 0.3 * cap[a] * sc * lump + 0.8]
            capt += [0, 1]
            if r.random() < 0.35:
                m_ = len(xs) // 2
                drop = r.uniform(0.04, 0.12) * sc
                capl.append(np.array([[xs[m_], ys[m_]], [xs[m_] + r.normal(0, 2), ys[m_] + drop]]))
                capw.append(0.35 * cap[a] * sc + 1)
                capt.append(1)
            xs, ys = see(Xf[sel], 0.02 + r.uniform(0, 0.1), Zf[sel])   # and the drift at its foot
            capl.append(np.stack([xs, ys], 1))
            capw.append(r.uniform(0.02, 0.06) * sc)
            capt.append(2)
            a = b - r.integers(2, 10)
    capt = np.array(capt)
    tops, unders, drift = trio(GOLD[2], RIM[2], LIT[3]), trio(LILAC[2], SHADE[3], LILAC[3]), trio(SHADE[1], LILAC[0], SHADE[2])
    lay(capl, np.array(capw), (np.stack([tops, unders, drift])[capt].astype(np.float32), np.tile([0.55, 0.3, 0.15], (len(capt), 1))),
        thick=0.4, pickup=0.15, dry=np.where(capt == 0, r.uniform(0.0, 0.12, len(capt)), np.where(capt == 2, 0.35, 0.15)),
        spent=0.5, ends=(0.3, 0.15),
        taper=0.3, fray=1.6, grooves=0.7, land=0.8, lift=0.5)
    posts, pw, pt = [], [], []
    for x0, rad, hgt, lean in fen["stakes"]:
        z0 = zf(fen, x0)
        sc = F / z0
        w = rad * sc
        xa, ya = see(x0, 0, z0)
        xb, yb = see(x0 + lean * hgt, hgt, z0)
        m = 2 if rad > 0.06 else 1
        for j in range(m):
            o = ((j + 0.5) / m - 0.5) * 2 * w
            posts.append(np.array([[xb + o, yb + 1], [(xa + xb) / 2 + o, (ya + yb) / 2], [xa + o, ya + 4]]))
            pw.append(1.25 * w / m)
            pt.append(0.25 + 0.3 * j)
        if x0 > 0.3:                                               # its sunward side, which we see right of the lane
            posts.append(np.array([[xb - 0.8 * w, yb + 2], [xa - 0.8 * w, ya - 0.3 * (ya - yb)]]))
            pw.append(max(1.0, 0.22 * w))
            pt.append(0.97)
        posts.append(np.array([[xb - 1.1 * w, yb - 0.3 * w], [xb + 1.1 * w, yb - 0.3 * w]]))   # its cap of snow
        pw.append(0.6 * w + 1.5)
        pt.append(0.99)
        posts.append(np.array([[xb - 1.0 * w, yb + 0.5 * w], [xb + 1.0 * w, yb + 0.5 * w]]))   # shaded on its near side
        pw.append(0.3 * w + 1)
        pt.append(0.96)
    pt = np.array(pt)
    cols = loads(STAKE, pt.clip(0, 0.9))[0]
    cols[pt > 0.98] = tops
    cols[(pt > 0.965) & (pt < 0.98)] = trio(GOLD[1], RIM[1], GOLD[2])
    cols[(pt > 0.955) & (pt < 0.965)] = unders
    lay(posts, np.array(pw), (cols.astype(np.float32), np.tile([0.55, 0.3, 0.15], (len(pt), 1))), thick=0.28, pickup=0.15,
        dry=np.where(pt > 0.95, 0.15, 0.05), spent=0.5, ends=(0.3, 0.2), taper=0.2, fray=1.0)
    wet[:] = DRY

    # the last sitting, over all of it dry: pale gold dragged across the crust of the lit snow, cobalt and
    # violet into the shadows, and the bird
    last = dict(pickup=0, merge=0, **cream)
    go(thin(near) * lit * 0.35, near, 26, (50, 140), (9, 16), lambda P: loads(np.concatenate([GOLD, LIT[2:], PEACH[1:]]),
                                                                            r.random(len(P)), RIM, 0.2),
       scale=kb, tilt=0.35, bend=0.003, thick=0.2, dry=(0.45, 0.7), **last)
    go(thin(near) * noise.smoothstep(0.4, 0.8, shade) * 0.35, near, 26, (40, 120), (7, 12),
       lambda P: loads(np.concatenate([SHADE[:3], COBALT, LILAC[:2]]), r.random(len(P))), scale=kb, tilt=0.25,
       thick=0.2, dry=(0.4, 0.65), **last)
    go(thin(beyond) * lit * 0.25 * clear, beyond * clear, 11, (18, 45), (4, 7), lambda P: loads(np.concatenate([GOLD, LIT[2:]]),
                                                                               r.random(len(P))),
       scale=kb, tilt=0.3, thick=0.16, dry=(0.45, 0.7), **last)
    strokes, Zb = D["bird"]
    kinds = [k_ for _, _, k_ in strokes]
    fam = {"body": BIRD, "wing": np.concatenate([BIRD[1:], SHEEN[:1]]), "flank": FLANK, "shoulder": FLANK[1:],
           "head": BIRD[:2], "bill": BIRD[:1], "tail": np.concatenate([SHEEN, BIRD[1:]]), "legs": BIRD, "rim": RIM}
    cols = np.stack([np.stack([fam[k_][i % len(fam[k_])] for i in (0, 1, 2)]) for k_ in kinds]).astype(np.float32)
    for sharp in (False, True):                                     # the tail and the bill taper to a point
        sel = [i for i, k_ in enumerate(kinds) if (k_ in ("tail", "bill")) == sharp]
        lay([strokes[i][0] for i in sel], np.array([strokes[i][1] for i in sel]),
            (cols[sel], np.tile([0.55, 0.3, 0.15], (len(sel), 1))), thick=0.35, pickup=0.15,
            dry=np.array([0.25 if kinds[i] in ("rim", "shoulder") else 0.0 for i in sel]), spent=0.6 if sharp else 0.35,
            ends=(0.25, 0.1), taper=0.7 if sharp else 0.3, fray=1.0, tails=0.5, lips=0.12)

    img = dabs.shine(rgb, ndimage.gaussian_filter(height, 0.8), light=LIGHT, relief=0.5, gloss=0, reach=(0.78, 1.12))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
