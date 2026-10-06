"""Sunflowers, Late Summer. Oil on a white-primed canvas, thin and thick.

In her last years Joan Mitchell lived on a hill above the Seine at Vétheuil, with a garden where she
grew sunflowers, and painted large canvases primed white. A painting went on in gestures of the whole
arm: masses of strokes, some laid broad and flat with a big brush, some lashed on as lines, some
dragged nearly dry, the colours mostly as they came from the tube, cadmium yellow and orange,
ultramarine, emerald and sap green, a few near-blacks. Paint thinned with turpentine ran down the
canvas from the strokes. Then she painted the white, back over the edges of the colour, to cut it
back and shape it, so that the white is a colour among the others and the masses breathe in it.
Nothing here is drawn. The sunflowers are in the weight of the masses, high on the canvas, and in
their colours.
"""

import zlib

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Sunflowers, Late Summer"
DATE = "2026"
MEDIUM = ("Oil on canvas: thin washes that run, broad loaded strokes and fast dry ones of unmixed colour, "
          "white painted back over them")
AFTER = "Joan Mitchell, La Grande Vallée and the Sunflowers, Vétheuil, 1983–92"
ROOM = "Colour Itself"
YEAR = 1990
PLACE = "Vétheuil"
REGION = "Americas"
NOTE = ("Masses of cadmium yellow and orange, ultramarine and emerald, high on a white canvas, with thin paint "
        "running down from them. The white is painted too, back over the colour, to shape it.")

H, W = 3000, 2400
S = 4                   # the maps that place the strokes are worked out at a quarter of the size
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
YELLOWS = pal("#9c6a22", "#bf8b2e", "#e2930f", "#f0b419", "#f6cb22", "#f8dc34", "#f6e46a", "#efd796")
#             raw sienna, ochre, cadmium yellow deep, medium, light, lemon, Naples yellow
ORANGES = pal("#c8401a", "#da521c", "#e8661f", "#ef7a25", "#f38f30", "#f4a447")    # cadmium orange
OCHRES = pal("#8a5a1e", "#a36e25", "#b9842f", "#c99a3c", "#d6b058", "#e1c27a")     # raw sienna, yellow ochre
ULTRA = pal("#121a5e", "#172377", "#1d2f92", "#2640a8", "#3654b8")                 # ultramarine
COBALT = pal("#24509f", "#2f62b6", "#4677c4", "#6590d0", "#86aadb")
CERULEAN = pal("#2a78ad", "#3b8dc0", "#5aa3cf", "#82bbdc")
PRUSSIAN = pal("#0e1f33", "#142c45", "#1c3b58", "#27506f")
VIOLET = pal("#1d1546", "#2a1f60", "#382a78", "#4a3a8f", "#6353a6", "#8475bb")      # violet-blue
EMERALD = pal("#0a5a44", "#0e7054", "#158762", "#229e72", "#3caa7c", "#6cc09a")
VIRIDIAN = pal("#0b4a45", "#0f5f57", "#16766a", "#24897b")
SAP = pal("#2c4610", "#3a5a15", "#4b6e1b", "#628624", "#7f9f30")
LEAF = pal("#7d9e2e", "#93b43c", "#a9c64e")                                        # sap green let down with yellow
DARK = pal("#120b18", "#1c1124", "#281531", "#371a37")                             # black with violet
MAROON = pal("#3a0e1a", "#521424", "#6a1c2c")
GREYS = pal("#d6d6d1", "#dcdcd8", "#d9dce1", "#e0dbd0", "#e3e1db")                 # white greyed, dragged through
WHITES = pal("#e4e3dd", "#e9eae8", "#edeae1", "#f0eee8", "#f2eee3", "#f4f2ed", "#f7f5ef")   # greyed, cool, warm, bright

# the masses and the places inside them (centre x, y, half-width, half-height, tilt)
SUN = (850, 960, 540, 480, -0.15)         # cadmium yellow and orange, high and to the left
SEA = (1830, 830, 430, 600, 0.12)         # ultramarine, higher still, pressing in from the right
FIELD = (720, 1760, 430, 250, -0.25)      # the greens, under the sun and to the left
CORE = (960, 1030, 320, 260, 0.4)         # where the orange burns
DROOP = (800, 1310, 420, 170, -0.1)       # ochre, on the heavy underside
CROWN = (760, 720, 420, 240, -0.3)        # lemon, where the light falls on it
DEEP = (1520, 980, 200, 380, 0.25)        # violet-blue, hard against the yellow
HIGH = (1900, 420, 300, 220, 0.1)         # cobalt, toward the top
SEAM = (1370, 1150, 170, 330, 0.45)       # emerald, in the seam between the masses

# the big movements that set each passage going: control points (x, y), half-width, tone
KEYS = {
    "yellow": [([(450, 700), (800, 600), (1150, 640)], 70, 0.85), ([(480, 760), (400, 1000), (450, 1250)], 70, 0.55),
               ([(560, 1060), (850, 980), (1200, 1000)], 80, 0.6)],
    "orange": [([(780, 1220), (1000, 1150), (1260, 1120)], 55, 0.5)],
    "ochre": [([(650, 1300), (700, 1450), (740, 1600)], 45, 0.5)],
    "blue": [([(1620, 300), (1600, 750), (1570, 1250)], 85, 0.55), ([(1950, 350), (1920, 800), (1860, 1350)], 80, 0.45),
             ([(2170, 640), (2150, 1000), (2110, 1330)], 55, 0.6)],
    "violet": [([(1560, 620), (1450, 860), (1350, 1080)], 60, 0.4)],
    "emerald": [([(1420, 1150), (1330, 1380), (1200, 1560)], 45, 0.5),
                ([(450, 1700), (700, 1650), (1000, 1720)], 50, 0.6)],
    "sap": [([(500, 1850), (800, 1800), (1050, 1860)], 55, 0.5)],
    "white": [([(1500, 1480), (1650, 1560), (1850, 1580)], 80, 0.5)],
}

YY, XX = np.mgrid[0:H:S, 0:W:S].astype(np.float32)

# how far each kind of movement turns in all, radians (a spread for the straight ones, a range for the curved)
TURN = {"slab": 0.35, "drop": 0.25, "arc": (0.8, 2.2), "hook": (1.2, 2.4), "whip": (0.6, 1.4)}


def at(P):
    P = np.asarray(P)
    return (np.clip((P[..., 1] / S).astype(int), 0, YY.shape[0] - 1),
            np.clip((P[..., 0] / S).astype(int), 0, YY.shape[1] - 1))


def region(m, r, lump=0.3):
    """How deep inside a mass each point lies: 1 at its heart, 0 at its ragged rim and beyond."""
    cx, cy, rx, ry, t = m
    c, s = np.cos(t), np.sin(t)
    u, v = ((XX - cx) * c + (YY - cy) * s) / rx, ((YY - cy) * c - (XX - cx) * s) / ry
    return np.clip(1 - np.hypot(u, v) * np.exp(lump * noise.field(XX.shape, 300 / S, r)), 0, 1)


def places(dens, n, r):
    """n points thrown down by `dens`: thick where it runs high, none where it is zero."""
    d = dens.ravel().astype(np.float64)
    i = r.choice(d.size, n, p=d / d.sum())
    y, x = np.divmod(i, dens.shape[1])
    return np.stack([(x + r.random(n)) * S, (y + r.random(n)) * S], 1)


def gesture(kind, p, theta, L, turn, n=16):
    """One movement of the arm about p, L px long, heading off at `theta` and turning `turn` radians in
    all, in the manner of its kind: straight, a bow, a hook at the end, a whip one way and back."""
    s = np.linspace(0, 1, n)
    k = noise.smoothstep(0.55, 0.95, s) + 0.05 if kind == "hook" else np.ones(n)
    k = np.pi * np.sin(2 * np.pi * s) if kind == "whip" else k / np.trapezoid(k, s)
    th = theta + turn * np.concatenate([[0], np.cumsum((k[1:] + k[:-1]) / 2 * np.diff(s))])
    mid = (th[1:] + th[:-1]) / 2
    P = np.concatenate([[[0.0, 0.0]], np.cumsum(np.stack([np.cos(mid), np.sin(mid)], 1) * L / (n - 1), 0)])
    return P - P.mean(0) + p


def campaign(dens, n, kinds, L, w, steer, r, m=(2, 3), gap=1.2):
    """n bursts of strokes, placed by `dens`: each one movement from `kinds` ({kind: odds}), set going
    by `steer`, L px long and w px half-wide, and gone over m times side by side, as the arm repeats
    itself. -> paths, half-widths, the burst each belongs to"""
    names, odds = list(kinds), np.array(list(kinds.values()), float)
    paths, ws, ids = [], [], []
    for b, c in enumerate(places(dens, n, r)):
        kind = names[r.choice(len(names), p=odds / odds.sum())]
        theta, sign = steer(c, r)
        ww, LL = r.uniform(*w), r.uniform(*L) * (1.1 if kind == "hook" else 1)
        t = TURN[kind]
        turn = r.uniform(*t) * sign if isinstance(t, tuple) else r.normal(0, t)
        if kind in ("arc", "hook"):
            turn = np.sign(turn) * min(abs(turn), LL / (2.5 * ww))      # never tighter than the brush can turn
        k = int(r.integers(m[0], m[1] + 1))
        ahead, across = np.array([np.cos(theta), np.sin(theta)]), np.array([-np.sin(theta), np.cos(theta)])
        for j in range(k):          # each time a little to one side, starting a little sooner or later
            q = c + across * ww * gap * (j - (k - 1) / 2) * r.uniform(0.8, 1.25) + ahead * LL * r.uniform(-0.25, 0.25)
            paths.append(gesture(kind, q, theta + r.normal(0, 0.1), LL * np.exp(r.normal(0, 0.2)),
                                 turn * np.exp(r.normal(0, 0.12))))
            ws.append(ww * np.exp(r.normal(0, 0.1)))
            ids.append(b)
    return paths, np.array(ws), np.array(ids)


def around(m, spread=0.5):
    """Strokes that go round a mass, each burst one way or the other."""
    def steer(c, r):
        ch = r.choice([-1, 1])
        return np.arctan2(c[1] - m[1], c[0] - m[0]) + ch * np.pi / 2 + r.normal(0, spread), ch
    return steer


def falling(lean, spread):
    """Strokes pulled down the canvas, leaning `lean` from the plumb, now and then pushed back up."""
    def steer(c, r):
        return np.pi / 2 + lean + r.normal(0, spread) + np.pi * (r.random() < 0.25), r.choice([-1, 1])
    return steer


def loose(m, depth, spread=0.5):
    """Strokes that go round a mass toward its rim and any way at all in its heart, as a hand works a clump."""
    def steer(c, r):
        ch = r.choice([-1, 1])
        if r.random() < 1.2 - depth[at(c)]:
            return np.arctan2(c[1] - m[1], c[0] - m[0]) + ch * np.pi / 2 + r.normal(0, spread), ch
        return r.uniform(-np.pi, np.pi), ch
    return steer


def along(angle, spread=0.2):
    """Strokes that follow a map of directions (or one direction), either way along it."""
    def steer(c, r):
        a = angle[at(c)] if np.ndim(angle) else angle
        return a + r.normal(0, spread) + np.pi * (r.random() < 0.5), r.choice([-1, 1])
    return steer


def brushes(fam, tone, r, accent=None, odds=0.0):
    """The load of each brush: the paint for its tone (0 dark, 1 light), the paint next to it streaked
    in, and a little of a third, now and then an accent from elsewhere on the palette."""
    n, N = len(fam), len(tone)
    i = (np.clip(tone, 0, 0.999) * n).astype(int)
    j = np.clip(i + r.choice([-1, 1], N), 0, n - 1)
    third = fam[np.clip(i + r.choice([-2, 2], N), 0, n - 1)].copy()
    if accent is not None:
        hit = r.random(N) < odds
        third[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.6, 0.85, N), r.uniform(0.12, 0.3, N), r.uniform(0.02, 0.1, N)], 1)
    return np.stack([fam[i], fam[j], third], 1), share


def run(rgb, height, drips, r):
    """Thinned paint running down from a heavy wet stroke: a thread that wanders a little from the
    plumb, thins as it goes and mostly gathers in a bead where it stops, standing up from the canvas."""
    for x0, y0, w, L, col, a in drips:
        y0, y1 = int(y0), int(min(H - 1, y0 + L))
        if y1 - y0 < 8 or not 0 <= x0 < W:
            continue
        ys = np.arange(y0, y1 + 1)
        t = (ys - y0) / (y1 - y0)
        wave = lambda k, n: np.interp(t, np.linspace(0, 1, n), r.normal(0, k, n))
        xc = x0 + (1.5 + 0.015 * L) * wave(1.0, 6) + w * wave(0.3, 14)
        half = w * (1.25 - 0.6 * t) * np.exp(wave(0.15, 10))
        if r.random() < 0.75:
            half = (half + w * np.exp(-((ys - y1 + 1.4 * w) / (1.1 * w)) ** 2)) * np.sqrt(np.clip((y1 - ys) / w, 0, 1))
        else:                                                    # or runs out to nothing
            half *= np.sqrt(np.clip((y1 - ys) / (0.3 * (y1 - y0) + 1), 0, 1))
        xa, xb = int(max(0, xc.min() - 3 * w - 2)), int(min(W, xc.max() + 3 * w + 3))
        d = np.abs(np.arange(xa, xb)[None] - xc[:, None])
        cov = np.clip(half[:, None] - d + 0.5, 0, 1) * (a * (1 - 0.25 * t))[:, None]
        sub = rgb[y0:y1 + 1, xa:xb]
        sub += (col - sub) * cov[..., None]
        bead = np.sqrt(np.clip(1 - (d / (half[:, None] + 1e-3)) ** 2, 0, 1))
        height[y0:y1 + 1, xa:xb] += 0.7 * half[:, None] * bead * cov


# how the paint goes on, each with the body it has: the priming painted over in white; thinned to a wash
# that lies flat; loaded and brushed out broad; middling; lashed in lines; white cut back wet; flung fast
PRIME = dict(thick=0.006, grooves=0.12, lips=0.03, land=0.2, lift=0.1, tails=0.4, pickup=0.3, merge=3, spent=0.3,
             ends=(0.4, 0.3), taper=0.2, fray=1.5, flatten=0.7, hide=0.95, dry=(0.0, 0.08))
SCRUB = dict(PRIME, thick=0.0008, grooves=0.02, hide=0.7, dry=(0.2, 0.6))     # rubbed thin: the weave shows through
WASH = dict(thick=0.006, grooves=0.05, lips=0.0, land=0.1, lift=0.05, tails=0.6, pickup=0.3, merge=8, spent=0.6,
            ends=(0.5, 0.4), taper=0.4, fray=5.0, flatten=0.2, hide=0.3, dry=(0.1, 0.45))
FLAT = dict(thick=0.08, grooves=0.6, lips=0.12, land=0.5, lift=0.3, tails=0.5, pickup=0.45, merge=3, spent=0.42,
            ends=(0.3, 0.15), taper=0.15, fray=2.0, flatten=0.7, dry=(0.0, 0.1))
MID = dict(thick=0.14, grooves=0.5, lips=0.12, land=0.5, lift=0.3, tails=0.6, pickup=0.45, merge=2.5, spent=0.5,
           ends=(0.35, 0.2), taper=0.25, fray=1.6, flatten=0.6, dry=(0.0, 0.15))
LINE = dict(thick=0.22, grooves=0.4, lips=0.08, land=0.4, lift=0.3, tails=0.8, pickup=0.35, merge=1.5, spent=0.65,
            ends=(0.5, 0.3), taper=0.45, fray=1.2, flatten=0.5)
COAT = dict(thick=0.03, grooves=0.2, lips=0.06, land=0.3, lift=0.15, tails=0.5, pickup=0.45, merge=2.5, spent=0.45,
            ends=(0.35, 0.25), taper=0.2, fray=1.8, flatten=0.6, hide=0.95, dry=(0.0, 0.25))
FLUNG = dict(thick=0.16, grooves=0.6, lips=0.1, land=0.5, lift=0.3, tails=1.0, pickup=0.35, merge=2, spent=0.85,
             ends=(0.4, 0.2), taper=0.35, fray=1.8, flatten=0.5)
TANGLE = {"slab": 0.45, "hook": 0.2, "arc": 0.2, "whip": 0.15}
LASHES = {"whip": 0.5, "slab": 0.3, "hook": 0.2}


def paint(seed=1990):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint="#efece3", thread=3.0)
    rgb, height = ground.color.copy(), ground.tooth * 0.2
    wet = np.zeros((H, W), np.float32)

    def fresh(name):
        """Each passage draws on chances of its own, so the others stay as they are when one is changed."""
        return np.random.default_rng([seed, zlib.crc32(name.encode())])

    def lay(g, paths, w, fam, tone, drip=0.0, accent=None, odds=0.0, **kw):
        """Lay the strokes; then thinned paint gathered at the foot of the heavy ones runs down onto the
        white below, before the next go on."""
        cols, share = brushes(fam, np.asarray(tone, float), g, accent, odds)
        if isinstance(kw.get("dry"), tuple):
            kw["dry"] = g.uniform(*kw["dry"], len(paths))
        impasto.lay(rgb, height, paths, w, cols, g, share=share, wet=wet, **kw)
        drips = []
        for P, ww, c in zip(paths, np.broadcast_to(w, (len(paths),)), cols[:, 0]):
            q = P[np.argmax(P[:, 1])]
            y, x = int(q[1] + 1.5 * ww), int(q[0])
            if ww < 18 or not (0 <= x < W and y < H) or rgb[y, x].min() < 0.6:
                continue
            for _ in range(g.poisson(drip)):
                drips.append((q[0] + g.normal(0, 0.5 * ww), q[1] + 0.75 * ww, g.uniform(1.6, 4.0),
                              min(g.exponential(120) + 25, 420), c, g.uniform(0.8, 1.0)))
        run(rgb, height, drips, g)

    def look(field, paths):
        return field[at(np.array([P.mean(0) for P in paths]))]

    def passage(name, dens, n, kinds, L, w, steer, fam, tone=0.5, key=None, m=(2, 3), drip=0.0, accent=None,
                odds=0.0, how=FLAT, **kw):
        """One passage: bursts of strokes in one paint, their tone `tone` (or a map of tones), each burst a
        little lighter or darker than the next; then its big movements on top."""
        g = fresh(name)
        paths, ws, ids = campaign(dens, n, kinds, L, w, steer, g, m)
        t = (look(tone, paths) if np.ndim(tone) else tone) + g.normal(0, 0.1, ids.max() + 1)[ids] \
            + g.normal(0, 0.04, len(ids))
        if key:
            paths = paths + [np.array(p, float) for p, _, _ in KEYS[key]]
            ws = np.concatenate([ws, [k[1] for k in KEYS[key]]])
            t = np.concatenate([t, [k[2] for k in KEYS[key]]])
        lay(g, paths, ws, fam, t, drip, accent, odds, **{**how, **kw})

    sun, sea, field = region(SUN, r), region(SEA, r), region(FIELD, r)
    core, droop, crown = region(CORE, r), region(DROOP, r), region(CROWN, r)
    deep, high, seam = region(DEEP, r), region(HIGH, r), region(SEAM, r)
    mass = np.maximum.reduce([sun, sea, field])
    hue = noise.field(XX.shape, 250 / S, r)             # where a mass runs to one of its paints, where to another
    lean = 0.9 * noise.field(XX.shape, 600 / S, r)

    # the priming painted over in broad passages of white, warm and cool and greyed here and there, brushed
    # every which way; where the brush ran thin or dry the weave shows through
    passage("prime", 1 - 0.85 * noise.smoothstep(0.1, 0.5, mass), 22, {"slab": 0.9, "arc": 0.1}, (700, 1400),
            (120, 220), along(lean, 0.3), WHITES, 0.5 + 0.3 * noise.field(XX.shape, 500 / S, r), m=(1, 2), how=PRIME)
    passage("scrub", 1 - noise.smoothstep(0.05, 0.3, mass), 10, {"slab": 1}, (600, 1200), (110, 200), along(lean, 0.3),
            WHITES[2:], 0.6, m=(1, 1), how=SCRUB)
    wet[:] = 0

    # the first sitting: thin washes laid in where the masses will go, lying flat
    for i, (dens, fam, n, steer) in enumerate(((sun, YELLOWS, 4, loose(SUN, sun)), (sea, COBALT, 4, falling(0.2, 0.4)),
                                               (field, EMERALD, 3, along(0.0, 0.4)),
                                               (seam, SAP, 1, falling(0.5, 0.3)))):
        passage(f"wash{i}", np.sqrt(dens), n, {"arc": 1, "slab": 1}, (500, 900), (90, 130), steer, fam, 0.6, m=(1, 1),
                how=WASH)
    wet *= 0.4

    # the second: the masses in loaded strokes, wet into wet, each built of the paints next to its own.
    # The sun first, yellows, orange in its heart, ochre on its heavy underside and lemon and Naples where
    # the light falls; then the blues against it; the greens. Inside and round each, the tangle: middling
    # strokes, veils, greyed white dragged through, and small lashes, hooks and whips over all
    def tangle(name, dens, steer, mids, tone, lashes):
        passage(f"{name}_mid", dens ** 1.2, 18, TANGLE, (180, 450), (14, 30), steer, mids, tone, m=(1, 3), drip=0.3,
                how=MID)
        passage(f"{name}_veil", dens, 3, {"slab": 1}, (250, 500), (30, 60), steer, mids, tone, m=(1, 1),
                how=dict(MID, thick=0.02, hide=0.55))
        passage(f"{name}_grey", dens * dens, 3, {"slab": 0.6, "arc": 0.4}, (200, 450), (12, 26), steer, GREYS, 0.5,
                m=(1, 1), how=dict(MID, hide=0.8, pickup=0.6, dry=(0.1, 0.4)))
        passage(f"{name}_lash", dens ** 1.5 + 0.4 * dens * (1 - dens), 10, LASHES, (150, 380), (5, 11), steer, lashes,
                0.5, m=(1, 2), how=LINE)

    passage("yellow", sun * (1 - 0.6 * core), 10, {"slab": 0.7, "arc": 0.2, "hook": 0.1}, (250, 600), (40, 70),
            loose(SUN, sun), YELLOWS, 0.3 + 0.45 * crown, "yellow", accent=ORANGES, odds=0.12)
    passage("orange", core, 5, {"slab": 0.6, "hook": 0.4}, (220, 480), (35, 60), loose(CORE, core), ORANGES, 0.5,
            "orange", accent=YELLOWS, odds=0.15)
    passage("ochre", droop, 6, {"slab": 0.7, "hook": 0.3}, (180, 420), (25, 48), falling(0.9, 0.5), OCHRES, 0.55,
            "ochre", m=(1, 3), drip=1.2, accent=ORANGES, odds=0.2)
    passage("lemon", crown * sun, 4, {"slab": 0.8, "arc": 0.2}, (220, 500), (35, 60), loose(SUN, sun), YELLOWS, 0.85,
            m=(1, 2))
    tangle("sun", sun, loose(SUN, sun), YELLOWS, 0.5 + 0.25 * hue + 0.2 * crown,
           np.concatenate([ORANGES[:4], OCHRES[:3]]))

    passage("blue", sea * (1 - 0.5 * high), 10, {"drop": 0.55, "slab": 0.45}, (300, 800), (40, 80), falling(0.15, 0.3),
            ULTRA, 0.55, "blue", drip=1.0, accent=VIOLET, odds=0.2)
    passage("violet", deep, 5, {"slab": 0.6, "arc": 0.4}, (220, 520), (25, 50), falling(0.6, 0.35), VIOLET, 0.45,
            "violet", m=(1, 3), drip=0.6, accent=ULTRA, odds=0.3)
    passage("cobalt", high, 4, {"slab": 0.8, "arc": 0.2}, (250, 550), (35, 60), falling(0.5, 0.4), COBALT, 0.5,
            m=(1, 2), accent=CERULEAN, odds=0.3)
    tangle("sea", sea, falling(0.25, 0.5), np.concatenate([ULTRA[2:], COBALT, CERULEAN]),
           0.45 + 0.25 * hue + 0.25 * high, np.concatenate([PRUSSIAN, VIOLET[:4]]))

    passage("seam", seam, 5, {"slab": 1}, (200, 480), (20, 45), falling(0.6, 0.5), EMERALD, 0.5, "emerald", m=(1, 2),
            accent=VIRIDIAN, odds=0.25)
    passage("emerald", field, 5, {"slab": 1}, (220, 550), (30, 60), along(-0.2, 0.35), EMERALD, 0.45, m=(2, 3),
            drip=0.8, accent=SAP, odds=0.2)
    passage("sap", field, 4, {"slab": 1}, (220, 500), (30, 60), along(0.15, 0.35), SAP, 0.5, "sap", m=(1, 3), drip=0.8,
            accent=EMERALD, odds=0.2)
    passage("leaf", field * field, 2, {"slab": 1}, (200, 420), (30, 48), along(-0.15, 0.2), LEAF, 0.5, m=(1, 1))
    tangle("field", np.maximum(field, seam), along(-0.15, 0.6), np.concatenate([VIRIDIAN, EMERALD]), 0.5 + 0.3 * hue,
           np.concatenate([SAP[:3], VIRIDIAN[:2], LEAF]))

    # the arm flung out across the warm mass in long fast strokes, breaking up as they run dry
    for i, (fam, dens, steer) in enumerate(((YELLOWS, sun * sun, around(SUN, 0.3)),
                                            (ORANGES, core, around(CORE, 0.3)))):
        g = fresh(f"flung{i}")
        paths, w, ids = campaign(dens, 1, {"whip": 1}, (400, 750), (12, 22), steer, g, m=(1, 2), gap=3)
        lay(g, paths, w, fam, g.uniform(0.3, 0.8, len(paths)), **FLUNG)

    # white, painted back round the colour where it lies: over the ground along its edge, cutting in
    # from outside and taking a little of the colour up
    laid = (np.abs(rgb[::S, ::S] - ground.color[::S, ::S]).sum(-1) > 0.25).astype(np.float32)
    cover = ndimage.gaussian_filter(laid, 60 / S)
    rim = (cover < 0.3) * noise.smoothstep(0.06, 0.25, cover)        # just outside the colour, never in its gaps
    gy, gx = np.gradient(ndimage.gaussian_filter(laid, 60 / S))
    edge = np.arctan2(gy, gx) + np.pi / 2
    passage("rim", rim, 7, {"slab": 1}, (250, 550), (40, 75), along(edge, 0.2), WHITES[2:], 0.5, "white", m=(1, 2),
            how=COAT)
    wet[:] = 0

    # the third, over dry paint: a touch or two of near-black deep in the blue; thin lines lashed across
    # the masses; scumbles dragged over them; loaded strokes of pure colour on top, and white cut back
    passage("dark", deep * sea, 2, {"slab": 1}, (120, 240), (14, 20), falling(0.5, 0.5),
            np.concatenate([DARK, MAROON[1:]]), 0.5, m=(1, 1), dry=0.3, how=dict(LINE, thick=0.12, spent=0.5))
    warm, down = loose(SUN, sun), falling(0.2, 0.4)
    for i, (fam, dens, steer) in enumerate(((ORANGES[1:3], sun * sun * (1 - core), warm),
                                            (VIOLET[1:3], sea * sea, down),
                                            (SAP[:2], field * field, along(-0.1, 0.5)))):
        passage(f"lash{i}", dens, 2, {"whip": 1}, (300, 500), (4, 7), steer, fam, 0.5, m=(1, 1), how=LINE)
    for i, (fam, dens, steer) in enumerate(((YELLOWS, sun * (1 - core), warm), (ORANGES, core, warm),
                                            (COBALT, sea, down),
                                            (GREYS, mass * (1 - mass), along(edge, 0.4)), (CERULEAN, sea * high, down),
                                            (EMERALD, field, along(-0.1, 0.5)))):
        passage(f"scumble{i}", dens, 2 + 2 * (i > 2), {"slab": 0.7, "arc": 0.3}, (200, 500), (40, 65), steer, fam, 0.7,
                m=(1, 2), dry=0.7)
    for i, (fam, dens, n, steer) in enumerate(((YELLOWS, crown * crown * sun, 4, warm), (ORANGES, core, 2, warm),
                                               (ULTRA, sea, 2, down), (EMERALD, field, 1, along(-0.1, 0.4)))):
        passage(f"late{i}", dens, n, {"slab": 0.7, "arc": 0.3 * (i < 3)}, (180, 450), (35, 60), steer, fam, 0.7,
                m=(1, 2), drip=0.6)
    passage("cut", rim * noise.smoothstep(0.14, 0.06, cover), 3, {"slab": 1}, (180, 380), (45, 80), along(edge, 0.3),
            WHITES[3:], 0.6, m=(1, 1), how=dict(COAT, pickup=0, dry=(0.0, 0.2)))

    height = ndimage.gaussian_filter(height, 0.6)      # the lamp sees the surface a hair softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.68, 1.22))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
