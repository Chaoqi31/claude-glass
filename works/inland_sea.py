"""Evening Sails. Colour woodblock print in the shin-hanga manner.

Hiroshi Yoshida cut one set of blocks for his sailing boats on the Inland Sea and printed it again
and again in the colours of different hours. This is the evening impression of a new design. Two
great square sails, sewn from vertical cloths, take the low sun on the side that faces it. Every cloth
was wiped on the block by itself, so no two take the light alike, and the lavender of the side turned
away comes from a block of its own, laid a hair out of register. The sea gives the sails back in
horizontal pieces carried by ripples cut by hand. Only the hull and the rigging have a keyline; the
rest is overprinted colour. Thin ink misses the valleys of the paper, so the pale ends of every
gradation come out salted, and the grain of the cherry and the swirl of the baren show in the flats.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import noise, paper, plate, relief
from atelier.color import glaze, pigment

TITLE = "Evening Sails"
DATE = "2026"
MEDIUM = "Colour woodblock print (shin-hanga) in twenty-five impressions, on kozo paper"
AFTER = "Hiroshi Yoshida, Sailing Boats, from the Inland Sea series, 1926"
ROOM = "The Workshop"
YEAR = 1926
PLACE = "Seto Inland Sea"
REGION = "East Asia"
NOTE = ("Yoshida printed the same blocks again for different hours of the day. This new design is pulled "
        "for the evening: the sails take the last of the sun, and the sea gives them back in pieces.")

H, W, M = 1600, 2400, 80
HZ, WL, WF = 905.0, 1014.0, 912.0       # horizon, the near boat's waterline, the far boat's
SAILS = [(560, 1030, 455, 924, 11, dict(tilt=6.0)),                        # xl, xr, head, foot, cloths
         (1075, 1615, 345, 932, 13, dict(tilt=-4.0)),
         (2092, 2118, 852, 902, 4, dict(belly=0.4, foot=2.0))]
BOW, STERN, DECK = 440, 1700, 948
HUT = (1630, 1694, 15)                  # the mat hut on the after deck: from, to, height
FARHULL = (2074, 2138, 905.0)
OFF = (2.2, -1.0)                       # how far the lavender block sits out of register, px

APRICOT, ROSE, VIOLET, DUSK = pigment("#f6c391"), pigment("#eca5ad"), pigment("#9a9bd6"), pigment("#7a80bd")
GOLD, GLOW, SHADE, SEAM = pigment("#f8e0ae"), pigment("#f3ad7c"), pigment("#9c98d8"), pigment("#b08a86")
SEA_WARM, SEA, SEA2, DEEP = pigment("#f2b49c"), pigment("#7f88d0"), pigment("#7681c9"), pigment("#5c66b0")
WOOD, PLUM, ISLE, ISLE2 = pigment("#b9785a"), pigment("#5a4048"), pigment("#ab9fc6"), pigment("#968bb8")
CLOTH, KEEL = pigment("#d6c3b8"), pigment("#5d4a5e")


def wipe(stops, r, waver=10.0, streak=0.025):
    """A gradation wiped down the block. stops: (row, amount) pairs. The edge of the wipe wavers, and the
    brush leaves faint streaks along its path where the ink is half wiped away."""
    t = np.arange(H, dtype=np.float32)[:, None] + waver * (noise.line1d(W, 300, r)[None, :]
                                                           + 0.35 * noise.field((H, W), 70, r))
    a = np.full((H, W), stops[0][1], np.float32)
    for (p0, v0), (p1, v1) in zip(stops, stops[1:]):
        a += (v1 - v0) * noise.smoothstep(p0, p1, t)
    a *= 1 + streak * noise.stretched((W, H), 3, 140, r).T * 4 * np.clip(a * (1 - a), 0, 0.25)
    return np.clip(a * (1 + 0.025 * noise.field((H, W), 160, r)), 0, None)


def baren(shape, r, n=1200):
    """The rub of the baren: overlapping circling strokes, each ridged by the coil of cord inside it."""
    h, w = shape
    im = Image.new("F", (w, h), 0.0)
    d = ImageDraw.Draw(im)
    for cx, cy, q, a0, span, v in zip(r.uniform(0, w, n), r.uniform(0, h, n), r.uniform(40, 140, n),
                                      r.uniform(0, 360, n), r.uniform(80, 240, n), r.normal(0, 1, n)):
        for k in range(4):
            d.arc((cx - q - 8 * k, cy - q - 8 * k, cx + q + 8 * k, cy + q + 8 * k), a0, a0 + span, fill=float(v), width=2)
    rub = ndimage.gaussian_filter(np.asarray(im), 2.5)
    return rub / rub.std()


def figure(r, period=15.0):
    """The grain of a flat-sawn cherry plank as the ink finds it: fine growth lines along the board, now
    crowded and now spread, waving slowly and fading in and out."""
    rings = np.cumsum(np.clip(1 + 0.4 * noise.line1d(H + 200, 60, r), 0.3, None)) / period
    ph = np.interp(np.arange(H)[:, None] + 100 + 6 * noise.fbm((H, W), 800, r, octaves=2)
                   + 0.8 * noise.field((H, W), 40, r), np.arange(H + 200), rings)
    g = np.exp(-((ph % 1 - 0.5) / 0.12) ** 2) * noise.smoothstep(-0.8, 1.2, noise.stretched((W, H), 8, 400, r).T)
    return (g - g.mean()) / g.std()


def impress(img, press, r, block, amount, ink, grain=0.05, salt=0.3, squash=0.1, rub=0.035):
    """One impression. The block is brushed with pigment and wiped, the damp sheet laid to the kento and
    rubbed with the baren. Thin ink reaches only the tops of the fibres, so the pale ends of a
    gradation come out salted; `salt` is how dry the printer kept this block."""
    sheet, rubs, pits = press
    pad = lambda a: np.pad(a, M)
    full, amt = pad(block), pad(np.clip(amount, 0, None))
    here = noise.smoothstep(0.2, 1.2, noise.field((H, W), 380, r))         # the grain shows here and there
    gr = pad(figure(r) * (0.4 + here))
    film = relief.ink_film(full, r, roller=0.0, squash=squash, grain=gr, grain_strength=grain) * amt
    b = np.roll(rubs[r.integers(len(rubs))], tuple(r.integers(0, 500, 2)), (0, 1))
    dens = relief.pull(full, film, sheet, r, pressure=1.6 + 0.05 * b, shift=tuple(r.normal(0, 0.4, 2)),
                       turn=r.normal(0, 1.5e-4))
    # the damp sheet swells and settles by a fibre's width between pulls, so each pull misses its own specks
    dens *= (1 + rub * b) * (1 - salt * np.clip(1 - 2 * amt, 0, 1) * np.roll(pits, tuple(r.integers(-1, 2, 2)), (0, 1)))
    k = np.pad(ink, ((M, M), (M, M), (0, 0))) if ink.ndim == 3 else ink     # a block brushed in two colours
    return glaze(img, ndimage.gaussian_filter(dens, 0.7), k)


def sail(x, y, xl, xr, yt, yb, n, belly=0.55, foot=16.0, bow=3.0, tilt=0.0):
    """A square sail hung from its yard and filled by the wind, as a function of picture coordinates.
    The cloth turns away toward both leeches, so the panels there look a little narrower; each cloth
    sags between its lacings to the yard and hangs a little differently at the foot. Returns the cloth
    and the across coordinate (0 at the left leech, 1 at the right)."""
    v = np.clip((y - yt) / (yb - yt), 0, 1)
    spread = 0.02 * (xr - xl) * v + bow * np.sin(np.pi * v)
    left, right = xl - spread, xr + spread
    s = (x - left) / (right - left)
    am = belly * (0.55 + 0.45 * v)
    u = np.clip(0.5 + np.arcsin(np.clip((2 * s - 1) * np.sin(am), -1, 1)) / (2 * am), 0, 1)
    sc = np.clip(s, 0, 1)
    head = yt + tilt * (sc - 0.5) + 2.5 * np.abs(np.sin(np.pi * u * n))
    # seeded by the sail itself, so its reflection hangs the same way
    hang = 2.5 * np.interp(u * n, np.arange(n) + 0.5, np.random.default_rng(int(xl)).normal(0, 1, n))
    edge = np.minimum.reduce([x - left, right - x, y - head, yb + foot * np.sin(np.pi * sc) + hang - y])
    return noise.smoothstep(-0.8, 0.8, edge), u


def rig(x, y):
    """The three sails, and the matting of the hut, at picture coordinates: the cloth, and how far
    across it each point lies (the sun is on the left)."""
    a, b, h = HUT
    base = sheer((a + b) / 2, BOW, STERN, DECK) + 2
    top = base - h * np.clip(1 - np.abs((x - (a + b) / 2) / ((b - a) / 2)) ** 4, 0, 1) ** 0.25
    hut = noise.smoothstep(top - 0.7, top + 0.7, y) * noise.smoothstep(base + 0.7, base - 0.7, y) * (top < base - 0.5)
    cloth, across = np.zeros((2,) + x.shape, np.float32)
    for c, u in [sail(x, y, *g[:5], **g[5]) for g in SAILS] + [(hut, (x - a) / (b - a))]:
        across = np.where(c > cloth, u, across)
        cloth = np.maximum(cloth, c)
    return cloth, across


def seams(xl, xr, yt, yb, n, r, belly=0.55, foot=16.0, bow=3.0, tilt=0.0):
    """The seams between the cloths, cut as fine lines that follow the belly of the sail. No two are cut
    alike: one runs heavy, the next is a hair, and each swells, thins and here and there breaks."""
    v = np.linspace(0, 1, 120)
    spread = 0.02 * (xr - xl) * v + bow * np.sin(np.pi * v)
    left, right = xl - spread, xr + spread
    am = belly * (0.55 + 0.45 * v)
    out = []
    for k in range(1, n):
        s = (np.sin(am * (2 * k / n - 1)) + np.sin(am)) / (2 * np.sin(am))
        px = left + s * (right - left) + 0.8 * noise.line1d(120, 14, r)
        top = yt + tilt * (s - 0.5) + 1
        py = top + v * (yb + foot * np.sin(np.pi * s) - top - 1)
        hw = (0.55 * r.lognormal(0, 0.4) * np.clip(1 + 0.6 * noise.line1d(120, 10, r), 0, None)
              * noise.smoothstep(0, 0.04, v) * noise.smoothstep(1, 0.9, v) ** 0.5)
        out.append((np.stack([px, py], 1), hw))
    return out


def sheer(px, xb, xs, top, k=1.0):
    """The top edge of the hull: it sweeps up to the stem and lifts a little at the stern."""
    t = (px - xb) / (xs - xb)
    return (top - 50 * k * noise.smoothstep(0.24, 0.0, t) ** 1.7 - 12 * k * noise.smoothstep(0.86, 1.0, t)
            - 8 * k * (2 * np.clip(t, 0, 1) - 1) ** 2)


def hull(x, y, xb, xs, top, wl, k=1.0):
    """A long low wasen with a raked stem at the bow and a square stern."""
    keel = wl - 30 * k * noise.smoothstep(0.13, 0.0, (x - xb) / (xs - xb)) ** 2
    stem = xb + 58 * k * np.clip((y - top + 50 * k) / (wl - top + 50 * k), 0, 1) ** 1.4
    edge = np.minimum.reduce([x - stem, xs - x, y - sheer(x, xb, xs, top, k), keel - y])
    return noise.smoothstep(-0.8, 0.8, edge)


def cut(px, py, wt, r, amp=1.0, ends=(0.04, 0.4), swell=0.45):
    """One stroke of the V-gouge along a path: it bites in, runs a little unsteadily, rides deeper and
    shallower, and lifts out."""
    n = len(px)
    t = np.linspace(0, 1, n)
    py = py + amp * noise.line1d(n, max(4.0, n / 5), r)
    a, b = r.uniform(*ends, 2)
    hw = (wt * noise.smoothstep(0, a, t) * noise.smoothstep(1, 1 - b, t) ** 0.8
          * np.clip(1 + swell * noise.line1d(n, max(4.0, n / 6), r), 0.25, None))
    return np.stack([px, py], 1), np.clip(hw, 0, None)


def line(p0, p1, w, r, sag=0.0, n=80, taper=0.06):
    """A spar, stay or rope: a cut line from p0 to p1, sagging by `sag` px in the middle."""
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float) + np.array([0.0, sag]) * 4 * t * (1 - t)
    return relief.vcut(pts, w, r, taper=taper, wobble=0.15)


def ripple(x0, y0, L, wt, r, d):
    """One ripple, cut with a single push of the knife: it bites in, swells, and runs out to a hair,
    bowing a little on the way. Half of them are cut from the other end."""
    n = max(8, int(L / 2.5))
    t = np.linspace(0, 1, n)
    hw = (wt * noise.smoothstep(0, r.uniform(0.03, 0.12), t) * (1 - t) ** r.uniform(0.3, 0.8)
          * np.clip(1 + 0.3 * noise.line1d(n, max(4.0, n / 4), r), 0.3, None))
    py = y0 + (0.4 + 0.01 * d) * (r.normal() * np.sin(np.pi * t) + 0.6 * r.normal() * np.sin(2 * np.pi * t))
    return np.stack([x0 + L * t, py], 1), hw if r.random() < 0.5 else hw[::-1]


def ripples(r):
    """Ripples cut by hand: small and crowded toward the horizon, larger and looser toward the viewer,
    gathered where a breath of wind crosses the water and absent where it is glassy. Some run doubled
    and some fork. Part are gouged from the cool water block so the warm first printing shows through;
    the rest stand on a block of their own and print dark."""
    wind = noise.field((H, W), 420, r)
    light, dark = [], []
    d = 2.0
    while d < H - HZ + 10:
        y0 = HZ + d
        x0 = r.uniform(-300, 0)
        while x0 < W + 60:
            L = (15 + 0.45 * d) * r.lognormal(0, 0.5)
            if r.random() < noise.smoothstep(-1.0, 0.8, wind[int(min(y0, H - 1)), int(np.clip(x0 + L / 2, 0, W - 1))]):
                pts, hw = ripple(x0, y0, L, (0.28 + 0.0045 * d) * r.lognormal(0, 0.4), r, d)
                group = [(pts, hw)]
                if r.random() < 0.2:            # doubled: a shorter cut alongside
                    at = x0 + r.uniform(-0.2, 0.5) * L
                    side = r.choice([-1, 1]) * (2 + 0.03 * d) * r.uniform(0.8, 1.4)
                    group.append(ripple(at, y0 + side, L * r.uniform(0.4, 0.8), 0.7 * hw.max(), r, d))
                if r.random() < 0.15:           # forked: a branch leaves it and thins away
                    i = r.integers(len(pts) // 5, len(pts) // 2 + 1)
                    t = np.linspace(0, 1, len(pts) - i)
                    group.append((pts[i:] + np.outer(t ** 1.5, [0, r.choice([-1, 1]) * (1.5 + 0.025 * d)]),
                                  0.7 * hw[i] * (1 - t) ** 0.8))
                if r.random() < 0.6 + 0.2 * min(d / 450, 1):
                    dark += group
                else:
                    light += [(p, np.minimum(h, 1.8)) for p, h in group]    # a gouge this narrow never opens a wide glint
            x0 += L + (14 + 0.6 * d) * r.lognormal(0, 0.7)
        d += (1.5 + 0.05 * d) * r.uniform(0.5, 1.6)
    return relief.cuts((H, W), light), relief.cuts((H, W), dark)


def paint(seed=1926):
    r = noise.rng(seed)
    SH, SW = H + 2 * M, W + 2 * M
    sheet = paper.washi((SH, SW), seed, tint="#f1eadb", margin=(38, 36), fibre_density=0.9)
    sheet.alpha = paper.deckle((SH, SW), (38, 36), r, ragged=2.0)
    # the valleys between the fibres, gathered into the specks that thin ink misses
    pits = noise.smoothstep(0.32, 0.14, sheet.tooth)
    press = sheet, [baren((SH, SW), r) for _ in range(3)], pits
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    col, row = x[0], y[:, 0]

    # a long low island on the right, wooded along its top, and a fainter one far off on the left
    rise = (noise.smoothstep(1840, 2080, col) * (30 + 7 * noise.line1d(W, 230, r))
            + 9 * np.exp(-((col - 2280) / 110) ** 2) + noise.smoothstep(390, 60, col) * (12 + 3 * noise.line1d(W, 90, r)))
    ridge = HZ - rise - 1.2 * np.abs(noise.line1d(W, 5, r)) * (rise > 3)
    distance = np.where(col < 1200, 0.5, 1.0)[None, :]

    def island(px, py):
        top = np.interp(px, col, ridge)
        return noise.smoothstep(top - 0.7, top + 0.7, py) * noise.smoothstep(HZ + 0.6, HZ - 0.6, py)

    # the sea is cut into bands of ripples, finer toward the horizon. Each band carries its piece of a
    # reflection a little to one side, and fine tapered cuts along the troughs between the bands let
    # the water show between the pieces
    Y = [HZ + 1.0]
    while Y[-1] < H + 50:
        Y.append(Y[-1] + (2 + 0.08 * (Y[-1] - HZ)) * r.uniform(0.6, 1.5))
    Y = np.array(Y, np.float32)
    hs, db = np.diff(Y), Y[:-1] - HZ
    j = np.clip(np.searchsorted(Y, row, side="right") - 1, 0, len(db) - 1)
    f = (row - Y[j]) / hs[j] - 0.5
    dx = (((0.4 + 0.035 * db) * r.normal(0, 1, len(db)))[j] + f * hs[j] * r.normal(0, 0.5, len(db))[j]
          + 8 * noise.line1d(H, 220, r) * noise.smoothstep(0, 400, row - HZ))[:, None]     # the columns sway as they go down
    gaps = []
    for yb, h in zip(Y[1:], hs):
        x0 = (BOW - 300 if yb > WL else 1990) - r.uniform(0, 8 * h)
        while x0 < W:
            L = h * r.uniform(2, 9)
            gaps.append(ripple(x0, yb, L, (0.09 + 0.05 * noise.smoothstep(WL, H, yb)) * h * r.lognormal(0, 0.5), r, yb - HZ))
            x0 += L + h * r.lognormal(0, 0.8)
    strip = 1 - relief.cuts((H, W), gaps)

    cloth, across = rig(x, y)
    rc, ru = rig(x + dx, 2 * WL - y)
    fc, fu = rig(x + 0.5 * dx, 2 * WF - y)
    nearside = x < 1900
    rcloth = np.where(nearside, rc * (y > WL + 1), fc * (y > WF + 1)) * strip
    ru = np.where(nearside, ru, fu)
    rcool = noise.smoothstep(0.42, 1.0, ru)
    rturn = noise.smoothstep(0.3, 0.62, ru) * noise.smoothstep(1.0, 0.7, ru)

    boat = hull(x, y, BOW, STERN, DECK, WL)
    sh = sheer(x, BOW, STERN, DECK)
    rboat = hull(x + dx, 2 * WL - y, BOW, STERN, DECK, WL) * (y > WL) * strip
    fboat = hull(x, y, *FARHULL, WF, k=0.12)
    rfboat = hull(x + 0.5 * dx, 2 * WF - y, *FARHULL, WF, k=0.12) * (y > WF) * strip
    isle = island(x, y) * (1 - cloth)
    risle = island(x + dx, 2 * HZ - y) * (y > HZ) * strip
    crest = isle * noise.smoothstep(HZ - 8, HZ - 24, y)

    # every cloth was wiped on the block by itself: its tone, and the height where its gradation falls,
    # are its own, and the side of each sail turned from the sun cools cloth by cloth
    gold, glow, rose, tones, sails, smask, shade, dusk = np.zeros((8, H, W), np.float32)
    wob = noise.field((H, W), 30, r)
    for (xl, xr, yt, yb, n, kw), hz in zip(SAILS, (1.0, 1.0, 0.7)):
        c, u = sail(x, y, xl, xr, yt, yb, n, **kw)
        v = np.clip((y - yt) / (yb - yt), 0, 1)
        k = np.minimum((u * n).astype(int), n - 1)
        uc = (k + 0.5) / n
        cool = 0.7 * noise.smoothstep(0.42, 1.0, uc) + 0.3 * noise.smoothstep(0.42, 1.0, u)
        turn = noise.smoothstep(0.3, 0.62, uc) * noise.smoothstep(1.0, 0.7, uc)
        tone = 1 + r.normal(0, 0.06, n)[k]
        fall = r.normal(0.45, 0.12, n)[k] + 0.03 * wob
        spread = r.uniform(0.15, 0.4, n)[k]
        head = noise.smoothstep(fall + spread, fall - spread, v)
        sails += c
        c = c * hz
        gold += c * 0.5 * (1 - 0.6 * cool) * tone
        glow += c * (0.38 + 0.25 * head) * (1 - 0.95 * cool) * tone
        rose += c * (0.4 * turn + 0.2 * head * (hz == 1.0))
        tones += c * r.choice([0.0, 0.0, 0.0, 0.05, 0.1], n)[k]
        # the lavender block: cut along a seam, wiped in from the far leech so that its cut edge prints
        # hard near the head and is lost toward the foot
        cs, us = sail(x - OFF[0], y - OFF[1], xl, xr, yt, yb, n, **kw)
        vs = np.clip((y - OFF[1] - yt) / (yb - yt), 0, 1)
        edge = np.round(0.5 * n) / n
        blk = cs * noise.smoothstep(-0.7, 0.7, (us - edge) * (xr - xl))
        start = edge - 0.35 + 0.43 * noise.smoothstep(0.35, 0.95, vs) + 0.03 * wob
        smask += blk
        shade += blk * hz * 0.85 * noise.smoothstep(start, 0.97, us) ** 0.7 * (1 - 0.2 * vs)
        dusk += blk * hz * 0.4 * noise.smoothstep(0.6, 1.0, us) ** 2
    hut = np.clip(cloth - sails, 0, 1)
    gold += 0.45 * hut
    glow += 0.3 * hut

    # spars and the keyline of the hull on one block, stays and sheets on another; masts stand behind the sails
    (fl, fr, ft, fb), (ml, mr, mt, mb), (kl, kr, kt, kb) = (g[:4] for g in SAILS)
    cx = [(fl + fr) / 2, (ml + mr) / 2, (kl + kr) / 2]
    t = np.linspace(0, 1, 60)
    masts = relief.cuts((H, W), [(np.stack([c + rake * t + 0.3 * noise.line1d(60, 20, r), b + t * (e - b)], 1),
                                  w * (1 - 0.5 * t))
                                 for c, rake, b, e, w in ((cx[0], -5, DECK, ft - 62, 3.0), (cx[1], 3, DECK, mt - 70, 3.4),
                                                          (cx[2], 0, 905, kt - 9, 0.7))])
    px = np.linspace(BOW + 4, STERN - 2, 200)
    ty = np.linspace(DECK - 50, WL, 40)
    spars = [line((fl - 24, ft - 8), (fr + 24, ft + 4), 2.3, r, sag=-6, taper=0.14),
             line((ml - 26, mt + 2), (mr + 26, mt - 6), 2.5, r, sag=-7, taper=0.14),
             line((kl - 5, kt - 1), (kr + 5, kt - 1), 0.8, r),
             cut(px, sheer(px, BOW, STERN, DECK) + 2.5, 1.3, r, amp=0.3, ends=(0.01, 0.03), swell=0.25),
             cut(px + 30, sheer(px + 30, BOW, STERN, DECK) + 21, 0.9, r, amp=0.4, ends=(0.02, 0.06), swell=0.3),
             cut(px[40:] + 60, sheer(px[40:] + 60, BOW, STERN, DECK) + 42, 0.6, r, amp=0.4, ends=(0.05, 0.2)),
             cut(px + 50, np.full_like(px, WL - 2), 1.8, r, amp=0.3, ends=(0.01, 0.03), swell=0.3),
             cut(BOW + 58 * ((ty - DECK + 50) / (WL - DECK + 50)) ** 1.4, ty, 1.1, r, amp=0.2, ends=(0.02, 0.1), swell=0.2),
             line((BOW + 2, DECK - 50), (BOW - 3, DECK - 64), 2.2, r, taper=0.3),
             line((STERN - 4, DECK - 12), (STERN + 16, WL + 10), 4.5, r, taper=0.2)]
    gear = [line((cx[0] - 5, ft - 58), (BOW + 1, DECK - 60), 0.7, r, sag=12),
            line((cx[1] + 3, mt - 66), (STERN - 8, DECK - 11), 0.7, r, sag=16),
            line((cx[0] - 4, ft - 54), (cx[1] + 3, mt - 62), 0.6, r, sag=8),
            line((fl - 2, fb + 2), (fl - 50, DECK - 12), 0.6, r),
            line((fr + 2, fb + 2), (fr + 20, DECK - 3), 0.55, r),
            line((ml - 2, mb + 2), (ml - 18, DECK - 2), 0.55, r),
            line((mr + 4, mb + 2), (mr + 44, DECK - 7), 0.6, r)]
    spar = np.maximum(np.maximum(masts * (1 - cloth), relief.cuts((H, W), spars)), fboat)
    rope = relief.cuts((H, W), gear)
    seam = relief.cuts((H, W), sum((seams(*g[:5], r, **g[5]) for g in SAILS[:2]), [])) * sails
    gouge, dark = ripples(r)

    # the masks of the blocks
    sky = noise.smoothstep(HZ + 0.6, HZ - 0.6, y) * (1 - cloth)
    sea = noise.smoothstep(HZ - 0.6, HZ + 0.6, y) * (1 - cloth)
    water = sea * (1 - boat) * (1 - fboat)
    plank = noise.smoothstep(sh + 20.4, sh + 21.6, y), noise.smoothstep(sh + 41.4, sh + 42.6, y)
    left = noise.smoothstep(W * 1.1, -W * 0.1, x)
    one = np.ones((H, W), np.float32)
    # the water and the cool side of the reflections are brushed onto one block in their two colours,
    # so no misregistration can open a line of paper between them; the second blue is wiped away
    # toward the reflections for the same reason
    still = wipe([(HZ, 0.3), (HZ + 25, 0.05), (HZ + 150, 0.4), (HZ + 450, 0.65), (H, 0.75)], r) * (1 - rcloth)
    lit = (0.25 + 0.7 * rcool) * rcloth
    pair = (still[..., None] * SEA + lit[..., None] * SHADE) / (still + lit + 1e-6)[..., None]

    img = sheet.color.copy()
    blocks = [
        (cloth, gold, GOLD, 0.07, 0.5),
        (sky, wipe([(0, 0), (360, 0), (740, 0.8), (HZ, 1.0)], r) * (0.8 + 0.3 * left), APRICOT, 0.03, 0.3),
        (sea * (1 - boat), wipe([(HZ, 1.0), (HZ + 220, 0.75), (H, 0.35)], r), SEA_WARM, 0.05, 0.3),
        (sky, wipe([(0, 0), (170, 0), (520, 0.85), (770, 0.75), (HZ, 0.4)], r), ROSE, 0.03, 0.3),
        (cloth, glow, GLOW, 0.07, 0.5),
        (cloth, rose, ROSE, 0.05, 0.5),
        ((isle + 0.3 * risle * (1 - gouge)) * distance,
         wipe([(HZ - 50, 0.75), (HZ - 12, 0.6), (HZ, 0.35), (HZ + 1, 0.3)], r, waver=3), ISLE, 0.05, 0.3),
        (sky, wipe([(0, 0.92), (280, 0.75), (590, 0.2), (720, 0.0)], r), VIOLET, 0.05, 0.3),
        (sky, wipe([(0, 0.35), (240, 0.0)], r), DUSK, 0.05, 0.3),
        (water * (1 - gouge * noise.smoothstep(HZ + 350, HZ + 150, y)), still + lit, pair, 0.05, 0.3),
        (rcloth, 0.4 * (1 - 0.7 * rcool), GLOW, 0.06, 0.4),
        (smask, shade, SHADE, 0.07, 0.5),
        (smask, dusk, DUSK, 0.07, 0.5),
        (tones, one, CLOTH, 0.06, 0.5),
        (rcloth, 0.35 * rturn, ROSE, 0.06, 0.4),
        (water * (1 - gouge) * (1 - ndimage.gaussian_filter(rcloth, 1.5)),
         wipe([(HZ + 100, 0.0), (HZ + 380, 0.3), (H, 0.4)], r), SEA2, 0.05, 0.3),
        (crest * distance, 0.35 * one, ISLE2, 0.03, 0.3),
        (boat, wipe([(DECK - 50, 0.95), (WL, 1.0)], r, waver=2), WOOD, 0.12, 0.4),
        (boat * (1 - plank[0]), 0.45 * one, GLOW, 0.06, 0.4),
        (boat * plank[0], (0.3 + 0.3 * plank[1]) * wipe([(DECK, 0.85), (WL, 1.15)], r, waver=2), DEEP, 0.08, 0.4),
        ((rboat + rfboat) * (1 - gouge), 0.7 * one, KEEL, 0.05, 0.3),
        (dark * sea * (1 - boat), wipe([(HZ, 0.35), (HZ + 200, 0.6), (H, 0.75)], r), DEEP, 0.03, 0.2),
        (seam, 0.4 * noise.smoothstep(-1.6, 0.4, noise.field((H, W), 60, r)), SEAM, 0.0, 0.2),
        (rope, 0.5 * one, PLUM, 0.0, 0.2),
        (spar, 0.75 * one, PLUM, 0.0, 0.2),
    ]
    for block, amount, ink, grain, salt in blocks:
        img = impress(img, press, r, block, amount * one, ink, grain=grain, salt=salt)
    return plate.mount(img, sheet, shadow=0.35)
