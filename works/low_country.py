"""The Low Country. Etching, printed on laid paper.

In the 1640s Rembrandt walked out of Amsterdam along the dykes and drew what he saw with
an etching needle: a cottage under its trees, the flat land going away to a thin line with
a church tower on it, and a great deal of sky. He crowded one side of the plate and left
the other nearly empty, and let the emptiness be the subject.
"""

import numpy as np
from PIL import Image, ImageDraw

from atelier import etching, lettering, noise, paper, plate, relief

TITLE = "The Low Country"
DATE = "2026"
MEDIUM = "Etching, printed on laid paper"
AFTER = "Rembrandt van Rijn, Landscape with a Cottage and a Large Tree, Amsterdam, 1641"
ROOM = "The Workshop"
YEAR = 1641
PLACE = "Amsterdam"
REGION = "Europe"
NOTE = ("A cottage under its trees on one side of the plate, and on the other the flat land "
        "running out to a church tower, a windmill, and a great deal of sky.")

PH, PW = 1480, 2400          # the copper plate, px
HORIZON = 955
EAVE, GROUND = 1045, 1182


def poly(points):
    im = Image.new("L", (PW, PH), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in points], fill=255)
    return np.asarray(im, np.float32) / 255


def runs(ok):
    e = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    return zip(e[::2], e[1::2])


# ---- the far distance: a thin line with a town on it

def distance(nd, r):
    x = 0.0
    while x < 1420:                                     # the horizon, broken
        L = r.uniform(40, 220)
        y = HORIZON + r.normal(0, 1.0)
        nd.stroke([(x, y), (x + L / 2, y + r.normal(0, 0.8)), (x + L, y + r.normal(0, 0.8))], width=0.9, depth=0.35,
                  wander=0.4, tremor=0.2)
        x += L + r.uniform(4, 30)
    x = 0.0
    while x < 1400:                                     # far trees, low along it
        L = r.uniform(20, 120)
        if r.random() < 0.55:
            nd.loops([(x, HORIZON - 3), (x + L, HORIZON - 3 + r.normal(0, 1.5))], r.uniform(1.6, 3.2), width=0.85,
                     depth=0.35)
        x += L + r.uniform(20, 120)
    # the town: roofs of every size, a few trees among them, and a church tower
    gx = 280.0
    while gx < 640:
        gw, gh = r.uniform(5, 16), r.uniform(8, 26)
        kind = r.random()
        if kind < 0.18:                                  # a tree between the houses
            nd.loops([(gx - gw, HORIZON - 4), (gx, HORIZON - gh * 0.9), (gx + gw, HORIZON - 5)], 2.4, width=0.85, depth=0.38)
        else:
            step = gw * r.uniform(0.3, 0.9)
            P = [(gx - gw, HORIZON - 1), (gx - gw, HORIZON - gh * 0.5), (gx - gw + step * 0.3, HORIZON - gh),
                 (gx + gw - step * 0.3 * (kind > 0.6), HORIZON - gh * (1 if kind > 0.6 else 0.97)), (gx + gw, HORIZON - gh * 0.5),
                 (gx + gw, HORIZON - 1)]
            nd.stroke(P, width=0.85, depth=0.38, wander=0.15, tremor=0.12, corners=True)
            if r.random() < 0.5:
                nd.hatch(poly([(gx, HORIZON - gh * 0.55), (gx + gw, HORIZON - gh * 0.5), (gx + gw, HORIZON - 1),
                               (gx, HORIZON - 1)]) * 0.55, np.deg2rad(85), 2.2, width=0.75, depth=0.36, length=(4, 14))
        gx += gw * r.uniform(1.1, 2.2) + r.uniform(0, 14)
    tx = 470
    for side in (-11, 11):
        nd.stroke([(tx + side, HORIZON - 1), (tx + side, HORIZON - 58)], width=1.0, depth=0.45, wander=0.2)
    for y in (HORIZON - 58, HORIZON - 36):
        nd.stroke([(tx - 11, y), (tx + 11, y)], width=0.8, depth=0.4, wander=0.2)
    for side in (-7, 7):                                # a lantern, and a short spire
        nd.stroke([(tx + side, HORIZON - 58), (tx + side, HORIZON - 76)], width=0.8, depth=0.4, wander=0.1)
    nd.stroke([(tx - 8, HORIZON - 76), (tx, HORIZON - 95), (tx + 8, HORIZON - 76)], width=0.85, depth=0.42, corners=True,
              wander=0.1)
    nd.stroke([(tx, HORIZON - 95), (tx, HORIZON - 104)], width=0.7, depth=0.4, wander=0.05)
    nd.hatch(poly([(tx + 2, HORIZON - 57), (tx + 11, HORIZON - 57), (tx + 11, HORIZON - 1), (tx + 2, HORIZON - 1)]) * 0.6,
             np.deg2rad(85), 2.4, width=0.8, depth=0.4, length=(8, 30))
    for mx, s in ((860, 1.0), (1150, 0.7)):             # two windmills
        base = HORIZON - 1
        nd.stroke([(mx - 9 * s, base), (mx - 6 * s, base - 30 * s), (mx, base - 38 * s), (mx + 6 * s, base - 30 * s),
                   (mx + 9 * s, base)], width=0.9, depth=0.45, corners=True, wander=0.2)
        nd.hatch(poly([(mx, base - 36 * s), (mx + 6 * s, base - 30 * s), (mx + 9 * s, base), (mx, base)]) * 0.7,
                 np.deg2rad(80), 2.2, width=0.8, depth=0.45, length=(5, 20))
        hub = np.array([mx + 1, base - 34 * s])
        a0 = r.uniform(0.2, 0.5)
        for k in range(4):
            a = a0 + k * np.pi / 2
            d, q = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            nd.stroke([hub, hub + 46 * s * d], width=0.9, depth=0.45, wander=0.15)
            nd.stroke([hub + 12 * s * d + 5 * s * q, hub + 44 * s * d + 5 * s * q], width=0.7, depth=0.35, wander=0.1)


def willow(nd, r, x, y, s):
    """A pollard willow in the middle distance: a stump of a trunk and a bristling head of shoots."""
    top = y - 20 * s
    nd.stroke([(x, y), (x + r.normal(0, 0.8) * s, top)], width=1.0 + 1.2 * s, depth=0.5 + 0.2 * s, wander=0.3)
    head = poly([(x - 10 * s, top + 2), (x - 13 * s, top - 12 * s), (x - 4 * s, top - 22 * s), (x + 8 * s, top - 21 * s),
                 (x + 15 * s, top - 10 * s), (x + 11 * s, top + 2)])
    nd.hatch(head * 0.7, np.deg2rad(-60), 2.2, width=0.9, depth=0.5 + 0.2 * s, length=(4, 14))
    for _ in range(int(6 + 8 * s)):
        a = -np.pi / 2 + r.normal(0, 0.55)
        L = r.uniform(10, 26) * s
        x0 = x + r.normal(0, 5 * s)
        nd.line(etching.hand([(x0, top - 6 * s), (x0 + np.cos(a) * L, top - 6 * s + np.sin(a) * L)], r, 0.2, 0.2, 1.0),
                0.85, 0.5 + 0.2 * s, taper=0.3)


def meadows(nd, r):
    """The flat land between: fields crowding together as they go away, hedges and a row of
    willows along a ditch, a farm among its trees, cattle, and a sail on a hidden canal."""
    for i in range(30):
        f = (i / 29) ** 2.0
        y = HORIZON + 4 + f * 330
        x = r.uniform(-40, 200)
        right = 1320 - 150 * f
        while x < right:
            L = r.uniform(30, 160) * (0.5 + f)
            if r.random() < (0.55 if f < 0.2 else 0.28):
                nd.stroke([(x, y + r.normal(0, 1)), (x + L * 0.5, y + r.normal(0, 1.2)), (min(x + L, right), y + r.normal(0, 1))],
                          width=0.7 + 0.35 * f, depth=0.28 + 0.2 * f, wander=0.5, tremor=0.25)
            x += L + r.uniform(30, 260) * (0.4 + f)
    for x0, x1, y in ((60, 420, 1000), (620, 860, 1030), (900, 1180, 990), (250, 560, 1075)):   # fields of short grass
        for _ in range(int((x1 - x0) / 6)):
            x = r.uniform(x0, x1)
            yy = y + r.uniform(-8, 8)
            nd.line([(x, yy), (x + r.normal(0, 0.8), yy - r.uniform(3, 7) * (1 + (y - HORIZON) / 150))], 0.8, 0.35)
    for x, y, L in ((180, 985, 170), (930, 975, 120)):   # hedgerows between the fields
        nd.loops([(x, y), (x + L * 0.5, y + r.normal(0, 2)), (x + L, y + r.normal(0, 2))], r.uniform(2.5, 4), width=0.9,
                 depth=0.45)
    for k in range(9):                                   # willows along a ditch, going away
        f = k / 8
        x, y = 120 + 700 * f ** 0.8, 1150 - 150 * f ** 0.7
        willow(nd, r, x + r.normal(0, 6), y, 1.3 - 0.95 * f)
    nd.stroke([(40, 1175), (400, 1100), (780, 1020), (920, 1000)], width=0.9, depth=0.45, wander=1.5)
    # a farm among trees
    fx, fy = 1040, 985
    nd.stroke([(fx - 40, fy), (fx - 36, fy - 22), (fx, fy - 34), (fx + 42, fy - 20), (fx + 44, fy)], width=0.9,
              depth=0.5, corners=True, wander=0.3)
    nd.hatch(poly([(fx, fy - 32), (fx + 42, fy - 20), (fx + 44, fy), (fx + 2, fy)]) * 0.55, np.deg2rad(80), 2.4,
             width=0.8, depth=0.45, length=(6, 20))
    for tx, tw in ((fx - 70, 30), (fx + 70, 38), (fx + 115, 26)):
        nd.loops([(tx - tw, fy - 25), (tx, fy - 45), (tx + tw, fy - 22)], 3.2, width=0.9, depth=0.5)
        nd.hatch(poly([(tx - tw * 0.8, fy - 24), (tx, fy - 42), (tx + tw * 0.8, fy - 22), (tx + tw * 0.6, fy - 4),
                       (tx - tw * 0.6, fy - 4)]) * 0.6, np.deg2rad(-50), 2.4, width=0.8, depth=0.45, length=(6, 18))
    sx, sy = 700, HORIZON - 2                            # a sail, going along a canal nobody can see
    nd.stroke([(sx, sy), (sx + 1, sy - 42)], width=0.9, depth=0.5, wander=0.1)
    nd.stroke([(sx + 1, sy - 40), (sx + 20, sy - 8), (sx + 1, sy - 6)], width=0.8, depth=0.45, corners=True, wander=0.2)
    nd.hatch(poly([(sx + 2, sy - 36), (sx + 18, sy - 9), (sx + 2, sy - 7)]) * 0.5, np.deg2rad(70), 2.2, width=0.7,
             depth=0.4, length=(4, 14))
    for cx, cy, s in ((600, 1012, 1.0), (655, 1018, 0.9), (880, 1002, 0.75)):   # cattle
        body = poly([(cx - 14 * s, cy - 5 * s), (cx + 14 * s, cy - 6 * s), (cx + 15 * s, cy + 4 * s), (cx - 13 * s, cy + 5 * s)])
        nd.hatch(body * 0.85, np.deg2rad(-65), 2.0, width=0.9, depth=0.5, length=(5, 12))
        for lx in (-10, -6, 7, 11):
            nd.line([(cx + lx * s, cy + 4 * s), (cx + lx * s + r.normal(0, 0.5), cy + 11 * s)], 0.9, 0.5)
        nd.stroke([(cx - 14 * s, cy - 3 * s), (cx - 20 * s, cy - 6 * s), (cx - 22 * s, cy)], width=1.0, depth=0.5)


def sky(nd, r):
    for _ in range(4):
        x, y = r.uniform(80, 1000), r.uniform(800, 905)
        L = r.uniform(60, 200)
        nd.stroke([(x, y), (x + L * 0.5, y + r.normal(0, 2)), (x + L, y + r.normal(0, 2))], width=0.75, depth=0.2,
                  wander=1.0, tremor=0.3)
    for bx, by, s in ((720, 360, 1.0), (770, 342, 0.8), (744, 404, 0.9), (1160, 292, 0.7)):
        nd.stroke([(bx - 9 * s, by - 3 * s), (bx - 4 * s, by - 5 * s), (bx, by)], width=1.0, depth=0.6, wander=0.1)
        nd.stroke([(bx, by), (bx + 4 * s, by - 5 * s), (bx + 10 * s, by - 2 * s)], width=1.0, depth=0.6, wander=0.1)


# ---- the cottage

def cottage(nd, r, xx, yy):
    """A farmhouse seen end-on: its gable in the light, its long side and deep roof of thatch
    going away into the shade of the trees."""
    apex, g0, g1 = (1512, 736), (1392, 1050), (1632, 1052)
    rend, eend = (2190, 706), (2280, 1010)
    gable = poly([g0, apex, g1, (1626, 1200), (1402, 1200)])
    roof = poly([apex, rend, eend, g1])
    side = poly([g1, eend, (2282, 1188), (1628, 1200)]) * (1 - roof)
    house = np.clip(gable + roof + side, 0, 1)
    # the gable end, in the sun: upright boards, a loft door, a small window
    for x in np.arange(1410, 1624, 15.0) + r.normal(0, 1.5, 15):
        top = apex[1] + (1050 - apex[1]) * min(1, abs(x - apex[0]) / (apex[0] - g0[0])) + 10
        nd.stroke([(x, top), (x + r.normal(0, 1), 1198)], width=0.85, depth=0.5, wander=0.7)
    gshade = gable * np.clip(0.08 + 0.3 * noise.smoothstep(1080, 1200, yy) + 0.25 * noise.smoothstep(1560, 1628, xx), 0, 1)
    nd.hatch(gshade, np.deg2rad(88), 3.4, width=0.9, depth=0.55, length=(20, 80))
    nd.hatch(np.clip(gshade - 0.3, 0, 1) * 2, np.deg2rad(20), 3.4, width=0.9, depth=0.6, length=(15, 50))
    for box, t in (((1496, 900, 1528, 944), 0.92), ((1470, 1092, 1506, 1128), 0.88)):
        x0, y0, x1, y1 = box
        m = poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        for a in (-50, 40):
            nd.hatch(m * t, np.deg2rad(a), 2.5, width=1.1, depth=0.9, length=(8, 30))
        nd.stroke([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], width=1.2, depth=0.9, corners=True, wander=0.3)
    # the thick edge of the thatch round the gable, and the ridge
    for a, b in ((apex, g0), (apex, g1)):
        P = etching.hand([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), b], r, 2.0, 0.6)
        nd.line(P, 1.8, 0.95)
        n = np.array([-(b[1] - a[1]), b[0] - a[0]], np.float64)
        n /= np.hypot(*n)
        n *= np.sign(n[1]) * -1                          # the thatch edge stands proud, above the boards
        for t in np.linspace(0.04, 0.98, 60):
            p0 = np.array(a) + (np.array(b) - np.array(a)) * t
            L = r.uniform(10, 24)
            nd.line(etching.hand([p0 - n * 2, p0 + n * L * 0.5, p0 + n * L], r, 0.3, 0.3, 1.0), 1.0, 0.8, taper=0.3)
    ridge = [(x, apex[1] + (rend[1] - apex[1]) * (x - apex[0]) / (rend[0] - apex[0]) + 9 * np.sin(np.pi * (x - apex[0])
              / (rend[0] - apex[0])) + r.normal(0, 1.5)) for x in np.linspace(apex[0], rend[0], 24)]
    nd.line(etching.hand(ridge, r, 1.0, 0.6, 1.0), 1.6, 0.9)
    for x, y in ridge[1:-1]:                             # straw standing up along the ridge
        for _ in range(3):
            xj = x + r.normal(0, 8)
            nd.line([(xj, y + 2), (xj + r.normal(0, 3), y - r.uniform(4, 12))], 1.0, 0.8, taper=0.2)
    # the long roof: thatch down the fall, darker toward the trees and in the lee of the eave
    u = np.clip((xx - apex[0]) / (rend[0] - apex[0]), 0, 1)
    ridge_y = apex[1] + (rend[1] - apex[1]) * u
    eave_y = g1[1] + (eend[1] - g1[1]) * np.clip((xx - g1[0]) / (eend[0] - g1[0]), 0, 1)
    v = np.clip((yy - ridge_y) / (eave_y - ridge_y + 1e-3), 0, 1)
    t = (0.14 + 0.45 * u ** 1.2 + 0.3 * noise.smoothstep(0.75, 1.0, v) + 0.16 * noise.fbm((PH, PW), 90, r, octaves=4)
         - 0.12 * noise.smoothstep(0.15, 0.0, v))
    thatch = np.clip(t, 0, 1) * roof
    # the thatch: tufts of straw laid down the slope, thick and dark toward the trees and under
    # the eave, thin where the sun lies on it; and here and there one long straw
    nd.patches(thatch, 40, np.deg2rad(86) + 0.25 * noise.field((PH, PW), 80, r), spacing=3.1, width=1.05, depth=0.78,
               spread=0.18, cross=0.62)
    ridge_at = lambda x: apex[1] + (rend[1] - apex[1]) * (x - apex[0]) / (rend[0] - apex[0])
    eave_at = lambda x: g1[1] + (eend[1] - g1[1]) * (x - g1[0]) / (eend[0] - g1[0])
    for _ in range(60):
        xs = r.uniform(g1[0], rend[0])
        ya = ridge_at(xs) + r.uniform(10, 60)
        yb = min(eave_at(min(xs, eend[0])) - r.uniform(0, 40), ya + r.uniform(60, 180))
        if yb - ya > 30:
            nd.stroke([(xs, ya), (xs + r.normal(4, 3), (ya + yb) / 2), (xs + r.normal(8, 4), yb)], width=1.0, depth=0.8,
                      wander=1.0)
    under = roof * noise.smoothstep(eave_y - 50, eave_y - 8, yy) * (0.5 + 0.4 * u)
    nd.patches(np.clip(under, 0, 1), 22, np.deg2rad(-15), spacing=3.0, width=1.0, depth=0.8, spread=0.25, cross=0.6)
    eave = [(x, g1[1] + (eend[1] - g1[1]) * (x - g1[0]) / (eend[0] - g1[0]) + 5 * np.sin(x / 31) + r.normal(0, 2))
            for x in np.linspace(g1[0], eend[0], 36)]
    nd.line(etching.hand(eave, r, 0.8, 0.8, 1.0), 2.0, 1.0)
    for x, y in eave[1:-1]:
        for _ in range(2):
            xj = x + r.normal(0, 6)
            nd.line([(xj, y - r.uniform(0, 8)), (xj + r.normal(0, 1.5), y + r.uniform(4, 14))], 1.1, 0.9)
    cx, cy = 1880, 770                                   # a chimney through the ridge
    nd.stroke([(cx, cy + 8), (cx, cy - 32), (cx + 24, cy - 33), (cx + 24, cy + 6)], width=1.3, depth=0.85, corners=True,
              wander=0.4)
    nd.hatch(poly([(cx + 9, cy - 32), (cx + 24, cy - 33), (cx + 24, cy + 6), (cx + 9, cy + 6)]) * 0.8, np.deg2rad(84), 2.6,
             width=1.0, depth=0.8, length=(8, 30))
    # the long side in shade: boards, the dark band under the eave, a door with someone in it, a window
    shade = side * np.clip(0.38 + 0.25 * u + 0.5 * noise.smoothstep(eave_y + 45, eave_y + 2, yy), 0, 1)
    nd.patches(shade, 30, np.deg2rad(92), spacing=3.0, width=1.0, depth=0.72, spread=0.12, cross=0.6)
    for y0 in np.arange(0, 150, 14.0):
        P = [(x, eave_y[0, int(x)] + 16 + y0 + r.normal(0, 1)) for x in np.linspace(1636, 2276, 6)]
        if P[0][1] < 1196:
            nd.stroke(P, width=0.9, depth=0.65, wander=0.6)
    door = poly([(1760, 1076), (1822, 1073), (1824, 1197), (1758, 1198)])
    fig = poly([(1782, 1197), (1784, 1146), (1788, 1128), (1798, 1124), (1806, 1130), (1807, 1150), (1811, 1197)])
    for a in (-60, 30, 80):
        nd.hatch(door * (0.97 - 0.6 * fig), np.deg2rad(a), 2.4, width=1.2, depth=0.95, length=(12, 50))
    nd.stroke([(1758, 1198), (1760, 1076), (1822, 1073), (1824, 1197)], width=1.6, depth=1.0, corners=True, wander=0.5)
    win = poly([(1960, 1070), (2020, 1067), (2021, 1108), (1961, 1110)])
    bars = (np.abs(xx - 1990) < 2.5) | (np.abs(yy - 1089) < 2.5)
    for a in (-55, 40):
        nd.hatch(win * 0.92 * ~bars, np.deg2rad(a), 2.4, width=1.1, depth=0.9, length=(8, 30))
    nd.stroke([(1960, 1070), (2020, 1067), (2021, 1108), (1961, 1110), (1960, 1070)], width=1.3, depth=0.9, corners=True,
              wander=0.4)
    nd.stroke([(1402, 1052), (1402, 1200)], width=1.4, depth=0.85, wander=0.5)
    nd.stroke([(1628, 1054), (1628, 1200)], width=1.8, depth=0.95, wander=0.5)
    nd.stroke([(1395, 1200), (1630, 1202), (2000, 1195), (2290, 1190)], width=1.6, depth=0.9, wander=1.5)
    return house


# ---- trees

def crown(clumps, r, jag=0.3):
    """A mass of foliage from clumps (x, y, radius): its mask, and a tone lit from the upper left."""
    mask = np.zeros((PH, PW), np.float32)
    shade = np.zeros((PH, PW), np.float32)
    for cx, cy, rad in clumps:
        y0, y1 = max(0, int(cy - rad * 1.5)), min(PH, int(cy + rad * 1.5))
        x0, x1 = max(0, int(cx - rad * 1.5)), min(PW, int(cx + rad * 1.5))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = (xx - cx) / rad, (yy - cy) / rad
        edge = 1 + jag * (0.6 * noise.field(dx.shape, max(3.0, rad / 3), r) + 0.4 * noise.field(dx.shape, max(2.0, rad / 9), r))
        inside = np.hypot(dx, dy) < edge
        s = noise.smoothstep(-0.9, 0.9, dx * 0.55 + dy * 0.83)
        shade[y0:y1, x0:x1] = np.where(inside, s, shade[y0:y1, x0:x1])
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], inside)
    return mask, shade


def masses(big, r, per=9):
    """Break each big mass of a tree (x, y, radius) into the clusters it is made of."""
    out = []
    for cx, cy, R in big:
        for _ in range(per):
            a, d = r.uniform(0, 2 * np.pi), R * np.sqrt(r.uniform(0, 0.7))
            out.append((cx + d * np.cos(a), cy + d * np.sin(a) * 0.85, R * r.uniform(0.32, 0.55)))
    out.sort(key=lambda c: c[1] - 0.4 * c[0])           # clusters nearer the light overlap the rest
    return out


def foliage(nd, r, clumps, depth=0.8, dark=0.0, leafy=1.0):
    """A crown: masses hatched in small turning patches, every cluster edged with leaves on the
    side the light comes from."""
    mask, shade = crown(clumps, r, jag=0.22)
    mask *= ~nd.hide
    ys, xs = np.nonzero(mask)
    lowright = noise.smoothstep(-0.3, 1.1, ((xs - xs.mean()) * 0.45 + (ys - ys.mean()) * 0.9) / (xs.std() + ys.std()))
    mass = np.zeros_like(mask)
    mass[ys, xs] = lowright
    t = mask * np.clip(0.1 + dark + 0.42 * shade + 0.34 * mass + 0.15 * noise.fbm((PH, PW), 45, r, octaves=3), 0, 0.86)
    turn = np.deg2rad(-48) + 0.7 * noise.field((PH, PW), 160, r)
    nd.patches(t, 24, turn, spacing=2.9, width=1.05, depth=depth, spread=0.4, cross=0.55)
    # leaves along the lit edge of every cluster, where the cluster is not buried
    for i, (cx, cy, rad) in enumerate(clumps):
        phi = np.linspace(np.pi * 0.75, np.pi * 1.9, max(12, int(rad * 1.6)))
        P = np.stack([cx + rad * np.cos(phi), cy + rad * np.sin(phi)], 1)
        ok = np.ones(len(P), bool)
        for ox, oy, orad in clumps[i + 1:]:
            ok &= np.hypot(P[:, 0] - ox, P[:, 1] - oy) > orad * 0.9
        for a, b in runs(ok):
            j = a
            while j < b - 4:
                k = min(b, j + int(r.uniform(14, 40)))
                if r.random() < 0.8 * leafy:
                    if r.random() < 0.6:
                        nd.scallops(P[j:k], r.uniform(5, 9), width=1.05, depth=depth, out=-1)
                    else:
                        nd.loops(P[j:k], r.uniform(2.2, 3.6), width=1.0, depth=depth)
                j = k + int(r.uniform(2, 10))
    # small leaves scattered over the half-lit parts
    for _ in range(int(mask.sum() / 2500 * leafy)):
        k = r.integers(0, ys.size)
        x, y = xs[k], ys[k]
        if 0.12 < t[y, x] < 0.55:
            a = r.uniform(0, 2 * np.pi)
            L = r.uniform(10, 26)
            nd.scallops([(x, y), (x + np.cos(a) * L, y + np.sin(a) * L)], r.uniform(4, 7), width=1.0, depth=depth, out=-1)
    return mask > 0


def trunk(nd, r, pts, width, depth=0.9, taper=0.5):
    """A trunk or limb: its two edges (the shadow side heavier), and bark running along it.
    Returns its silhouette, so what lies behind it can be hidden."""
    C = etching.hand(pts, r, 1.5, 0.5, 2.0)
    tan = np.gradient(C, axis=0)
    tan /= np.hypot(tan[:, 0], tan[:, 1])[:, None]
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    wd = np.linspace(width, width * taper, len(C))[:, None]
    lft, rgt = C - nrm * wd / 2, C + nrm * wd / 2
    nd.line(lft, 1.1, depth)
    nd.line(rgt, 1.9, depth)
    body = poly(np.vstack([lft, rgt[::-1]]))
    for _ in range(int(len(C) * width / 60)):           # bark, crowding toward the shadowed side
        i = int(r.integers(0, len(C) - 4))
        f = r.beta(2.2, 1.2)
        L = int(r.uniform(6, 22))
        seg = C[i:i + L] - C[i] + lft[i] + (rgt[i] - lft[i]) * f
        if len(seg) > 2:
            nd.line(etching.hand(seg, r, 0.4, 0.3, 1.0), 1.0, depth * 0.9)
    ang = float(np.mean(np.arctan2(tan[:, 1], tan[:, 0])))
    nd.hatch(body * 0.5, ang + np.pi / 2 + 0.3, 3.4, width=1.0, depth=depth, length=(4, 14), bow=0.2)
    return body > 0.5


# ---- the ground in front

def foreground(nd, r, xx, yy):
    """A sandy track coming up to the house, rough ground either side, and a dark bank of
    weeds in the near corner that the eye steps over into the picture."""
    track = lambda y: 1510 - (y - GROUND) / (PH - GROUND) * 780 + 60 * np.sin((y - GROUND) / 90)
    half = lambda y: 26 + (y - GROUND) * 0.55
    on_track = np.abs(xx - track(yy)) < half(yy)
    ground = yy > GROUND - 4
    bank_top = lambda x: 1290 + 0.00055 * (x - 80) ** 2 + 18 * np.sin(x / 55)
    bank = (yy > bank_top(xx)) & (xx < 760)
    g = ground * np.clip(0.05 + 0.6 * noise.smoothstep(1750, 2350, xx) + 0.16 * noise.fbm((PH, PW), 90, r, octaves=3)
                         + 0.22 * noise.smoothstep(GROUND + 80, GROUND, yy) * noise.smoothstep(1400, 2100, xx), 0, 1)
    g = np.where(bank, np.clip(0.45 + 0.35 * noise.smoothstep(bank_top(xx), bank_top(xx) + 120, yy)
                               + 0.15 * noise.fbm((PH, PW), 40, r, octaves=3), 0, 0.9), g * (1 - 0.9 * on_track))
    g = g.astype(np.float32)
    nd.hatch(np.clip(g * 1.2, 0, 1), np.deg2rad(-4), 5.0, width=1.0, depth=0.68, length=(40, 170), bow=0.04, jitter=0.06,
             hook=0.05)
    nd.patches(np.clip(g - 0.3, 0, 1) * 1.5, 34, np.deg2rad(-8) + 0.3 * noise.field((PH, PW), 200, r), spacing=3.2,
               width=1.0, depth=0.75, spread=0.25, cross=0.55)
    # the edges of the track, and its ruts
    for side in (-1, 1):
        ys = np.linspace(GROUND + 4, PH + 20, 30)
        nd.stroke(np.stack([track(ys) + side * half(ys) + r.normal(0, 3, 30), ys], 1), width=1.1, depth=0.7, wander=2.5)
    for off in (-0.35, 0.3):
        ys = np.linspace(GROUND + 60, PH + 20, 20)
        nd.stroke(np.stack([track(ys) + off * half(ys), ys], 1), width=0.9, depth=0.55, wander=2.0)
    # weeds standing up against the light along the top of the bank, and a few reeds
    for x in np.arange(0, 760, 7.0) + r.normal(0, 3, 109):
        y = bank_top(x) + r.uniform(0, 12)
        for _ in range(int(r.integers(1, 4))):
            a = -np.pi / 2 + r.normal(0.05, 0.4)
            L = r.uniform(10, 40) * (1.3 - x / 1200)
            nd.line(etching.hand([(x, y), (x + np.cos(a) * L * 0.5 + r.normal(0, 1.5), y + np.sin(a) * L * 0.5),
                                  (x + np.cos(a) * L, y + np.sin(a) * L)], r, 0.4, 0.2, 1.0), 1.05, 0.85, taper=0.2)
    for x in r.uniform(60, 560, 11):
        y = bank_top(x) + r.uniform(5, 20)
        L = r.uniform(50, 120)
        a = -np.pi / 2 + r.normal(0.15, 0.1)
        nd.stroke([(x, y), (x + np.cos(a) * L * 0.6, y + np.sin(a) * L * 0.6), (x + np.cos(a + 0.12) * L, y + np.sin(a + 0.12) * L)],
                  width=1.15, depth=0.85, wander=0.8)
    # a few clumps of grass by the track and under the house
    for _ in range(16):
        cx, cy = r.uniform(900, 2380), r.uniform(GROUND + 15, 1420)
        if abs(cx - track(cy)) < half(cy) + 10:
            continue
        for _ in range(int(r.integers(4, 10))):
            a = -np.pi / 2 + r.normal(0, 0.4)
            L = r.uniform(8, 26)
            x0 = cx + r.normal(0, 6)
            nd.line(etching.hand([(x0, cy), (x0 + np.cos(a) * L * 0.5 + r.normal(0, 1), cy + np.sin(a) * L * 0.5),
                                  (x0 + np.cos(a) * L, cy + np.sin(a) * L)], r, 0.3, 0.2, 1.0), 1.0, 0.75, taper=0.2)


def figure(nd, r, x, y, s=1.0):
    """A man walking with a stick, in a handful of lines."""
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    nd.stroke(P([(-3, -62), (-8, -58), (-6, -52), (2, -52), (4, -58), (-3, -62)]), width=1.1, depth=0.9, wander=0.2)
    nd.stroke(P([(-11, -64), (5, -65)]), width=1.3, depth=0.9, wander=0.2)
    coat = poly(P([(-6, -52), (4, -52), (8, -30), (9, -18), (-10, -18), (-8, -32)]))
    nd.hatch(coat * 0.8, np.deg2rad(70), 2.2, width=1.0, depth=0.9, length=(6, 20))
    nd.stroke(P([(-6, -52), (-8, -32), (-10, -18), (9, -18), (8, -30), (4, -52)]), width=1.1, depth=0.9, corners=True,
              wander=0.3)
    nd.stroke(P([(-4, -18), (-7, 0)]), width=1.2, depth=0.9, wander=0.2)
    nd.stroke(P([(4, -18), (9, -1)]), width=1.2, depth=0.9, wander=0.2)
    nd.stroke(P([(8, -40), (16, -30), (22, 2)]), width=1.0, depth=0.85, wander=0.2)


def paint(seed=1641):
    r = noise.rng(seed)
    nd = etching.Needle((PH, PW), r)
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    # drawn from the front back, so each thing hides what is behind it
    figure(nd, r, 1265, 1330, 1.3)
    shrubs = [(1370, 1160, 44), (1330, 1178, 30), (2040, 1170, 46), (2130, 1180, 40)]
    nd.hide |= foliage(nd, r, masses(shrubs, r, per=5), depth=0.8, dark=0.1)
    # the great tree at the end of the house
    tree = trunk(nd, r, [(2300, 1215), (2292, 1020), (2276, 840), (2262, 700)], 70)
    for limb in ([(2262, 730), (2180, 590), (2090, 470)], [(2266, 712), (2290, 520), (2320, 300)],
                 [(2268, 740), (2350, 640), (2420, 560)]):
        tree |= trunk(nd, r, limb, 30, depth=0.85)
    big = [(2060, 620, 100), (2000, 480, 110), (2080, 330, 140), (2210, 200, 170), (2350, 150, 150), (2420, 350, 160),
           (2370, 560, 150), (2250, 450, 150), (2150, 610, 140), (2300, 760, 120), (2420, 800, 110), (2140, 80, 120),
           (1990, 220, 90), (2420, 980, 90)]
    crown_a = foliage(nd, r, masses(big, r, per=10), dark=0.24)
    nd.hide |= tree | crown_a
    house = cottage(nd, r, xx, yy)
    nd.hide |= house > 0.5
    # a second tree behind the roof, and a lighter one by the gable
    back = [(1720, 700, 90), (1800, 560, 110), (1900, 470, 110), (1760, 420, 80), (1700, 560, 60)]
    nd.hide |= foliage(nd, r, masses(back, r, per=8), depth=0.75, dark=0.05)
    nd.hide |= trunk(nd, r, [(1318, 1204), (1310, 1040), (1298, 900), (1290, 800)], 36)
    for limb in ([(1294, 860), (1236, 760), (1180, 660)], [(1292, 820), (1345, 700), (1390, 600)]):
        nd.hide |= trunk(nd, r, limb, 15, depth=0.85)
    slim = [(1200, 760, 74), (1280, 650, 92), (1375, 610, 72), (1150, 680, 56), (1345, 730, 62), (1250, 540, 64),
            (1320, 500, 56), (1215, 860, 46), (1405, 700, 52), (1290, 430, 50)]
    nd.hide |= foliage(nd, r, masses(slim, r, per=7), depth=0.75, dark=-0.04, leafy=1.2)
    foreground(nd, r, xx, yy)
    distance(nd, r)
    meadows(nd, r)
    sky(nd, r)

    grooves = nd.grooves()
    sig = np.asarray(Image.open(lettering.CUT / "etched_signature.png"), np.float32) / 255
    sig = noise.warp(sig, 0.8 * noise.field(sig.shape, 12, r), 0.8 * noise.field(sig.shape, 12, r))
    sy, sx = 1440 - sig.shape[0] // 2, 1640                 # signed in the plate, in the light of the track
    grooves[sy:sy + sig.shape[0], sx:sx + sig.shape[1]] = np.maximum(grooves[sy:sy + sig.shape[0], sx:sx + sig.shape[1]],
                                                                     sig * 0.7)
    SH, SW = PH + 420, PW + 440
    sheet = paper.laid((SH, SW), seed, margin=(20, 20))
    y0, x0 = 190, 220
    img, pm = etching.print_plate(grooves, sheet, (y0, x0), r)
    img = relief.emboss(img, [pm], light=(-0.6, -0.8), depth=0.05)
    return plate.mount(img, sheet, shadow=0.35)
