"""The Inn in the Flood, Port-Marly. Oil on canvas, painted quickly in light, open strokes.

In March 1876 the Seine came up over the road at Port-Marly, west of Paris, and Alfred Sisley,
who was living at Marly-le-Roi, painted the village standing in the water: the inn at the corner
of the street with its ground floor drowned, the young trees along the towpath rising out of the
flood, the boats going up and down where the carts had been. He painted it several times that
spring, in grey weather and in sun, and it is the calmest of floods. Half the canvas or more is
sky. He worked fast over a pale ground, in clean colour and open strokes: curving touches round
the clouds and short ones across the blue, the water in long level strokes with the reflections
broken into them, the stone of the house in warm ochres and pinks with a cool lilac in its shade.

Here the inn stands at the right with its lit front turned toward the trees along the drowned
road, which go off to the left under the far hills, and a boatman poles a flat boat along the street.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Inn in the Flood, Port-Marly"
DATE = "2026"
MEDIUM = ("Oil on canvas, light open strokes laid wet into wet over a pale ground, the clouds in thicker paint, "
          "the water in long level strokes with the reflections broken into them")
AFTER = "Alfred Sisley, The Flood at Port-Marly, 1876"
ROOM = "Open Air"
YEAR = 1876
PLACE = "Port-Marly"
REGION = "Europe"
NOTE = ("A village street on the Seine under a spring flood: an inn standing in the water, a row of young trees "
        "rising out of it, and over them a wide sky of blue and white cloud, held again in the still water.")

H, W = 2100, 2800                    # about 60 by 81 cm
HZ, F, CAM = 1250.0, 3000.0, 2.0     # the horizon, the eye's reach in px, its height above the flood in metres
CX = W / 2
LIGHT = (-0.6, -0.5, 0.62)           # the lamp in the gallery
SUN = np.float32([-0.6, 0.5, 0.62]) / np.float32(np.linalg.norm([-0.6, 0.5, 0.62]))   # daylight, from the upper left
TURN = np.radians(48)
U1 = np.array([-np.sin(TURN), np.cos(TURN)])      # along the inn's front and the drowned road, away to the left
U2 = np.array([np.cos(TURN), np.sin(TURN)])       # along its end wall, to the right
CORNER = np.array([12.4, 38.0])                   # its corner nearest us, (X, Z) in metres
LONG, DEEP, EAVES, RIDGE = 11.0, 8.0, 6.5, 9.8
DECK = 1300.0                                     # the level base of the clouds, metres up
SKY = 6                                           # the sky chosen, out of many tried
LUMA = dabs.LUMA
BOATS = [(-8.0, 36.0, 4.2, 0.75, True), (-18.0, 50.0, 3.8, 0.3, False)]   # X, Z, length (m), heading off the road, a boatman


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
BLUE = pal("#5578b4", "#6283bc", "#7190c5", "#819dcd", "#93abd5", "#a5b9dc", "#b8c7e3", "#cbd5e9")   # cobalt, cerulean, white
LILAC = pal("#8e8aab", "#9b97b6", "#a9a4c0", "#b7b1ca", "#c5bfd3", "#d3cddd")                       # the clouds in shade
CREAM = pal("#e4dccd", "#ebe3d3", "#f1e9da", "#f5efe2", "#f9f4ea", "#fcf9f2")                       # and in the sun
ROSE = pal("#d6bdb4", "#dfc8bd", "#e7d3c8", "#eedfd4")                                             # warm toward the horizon
HAZE = pal("#c6cbdc", "#d2d4de", "#dcdadc", "#e4dfd9")
HILLS = pal("#7c7d9e", "#8888a8", "#9594b2", "#a2a0bc", "#b0adc6")
MEADOW = pal("#a0a698", "#aeb2a3", "#bcbeaf")
SILT = pal("#99a08d", "#a6ab98", "#b3b6a3", "#c0c1ae", "#cdcbb9")                                   # the flood's own colour
OCHRE = pal("#b19c78", "#c0ab86", "#cdb994", "#d9c7a3", "#e3d4b3")
JADE = pal("#86a69d", "#95b2a8", "#a4bdb3", "#b3c8be")
STONE = pal("#dbbb90", "#e3c69d", "#eacfaa", "#efd9b8", "#f3e2c6", "#f7ead5")                       # the inn's front in the sun
PINK = pal("#d2a28e", "#dbae9a", "#e3baa7", "#eac7b5", "#efd3c3")
SHADE = pal("#66627f", "#716d8b", "#7d7896", "#8a84a0", "#9890a6", "#a69aa4")                       # in shade, warmer low down
TILE = pal("#a8705f", "#b67d6b", "#c38b78", "#cd9a86", "#d6aa96")                                   # old tiles in the sun
TILE_SH = pal("#76606f", "#836b79", "#907783", "#9d838d")
SHUTTER = pal("#7a8a8b", "#879696", "#95a2a0", "#a3aeab")
PANE = pal("#3d3c52", "#4b4a60", "#5b5a70", "#6e6e86", "#8587a0")
BOARD = pal("#5d6a66", "#6b7873", "#7a8681")                                                       # the painted board
SIGN = pal("#7e4a3f", "#91574a", "#a46657", "#c99b84")
BARK = pal("#4b4352", "#584e5e", "#665a6a", "#756877", "#877884")
BARK_LIT = pal("#9b867f", "#ab958b", "#bba598")
TWIGS = pal("#7a6e86", "#887b91", "#96889c", "#a395a6", "#b0a3b0")
BUDS = pal("#a57f78", "#b48e85", "#c39e93")
HULL = pal("#2d2932", "#39333d", "#463e49", "#5a5059")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette."""
    t, g = np.clip(tone, 0, 1) * (len(colours) - 1), np.arange(len(colours))
    return np.stack([np.interp(t, g, colours[:, c]) for c in range(3)], -1).astype(np.float32)


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def valued(c, lum):
    """A colour brought to the value `lum` as a painter does it: let down with white to go lighter,
    deepened to go darker, its hue kept."""
    lc = (c @ LUMA)[..., None]
    lum = np.asarray(lum)[..., None]
    return np.where(lum >= lc, 1 - (1 - c) * (1 - lum) / (1 - lc + 1e-6), c * lum / (lc + 1e-6)).clip(0, 1)


def see(X, Y, Z):
    """Where a point of the scene (X across, Y up from the flood, Z away; metres) falls on the canvas."""
    return CX + F * np.asarray(X) / Z, HZ + F * (CAM - np.asarray(Y)) / Z


def clouds(r):
    """The cumulus, heaped up from a level deck: each a cluster of rounded lobes, some stacked on
    others and leaning with the wind, on a flat base; large overhead, smaller and more crowded toward
    the horizon, lit from the upper left, their edges torn and thinned in places so that the blue shows
    through. Worked at a quarter of the size. -> cloud (0..1), lit (-1..1), how far off (0..1), the way
    round them (radians) and how strongly their edges hold the brush to it (0..1), over the canvas"""
    s = 4
    h, w = int(HZ) // s + 2, W // s + 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) * s
    acc = np.zeros((3, h, w), np.float32)               # cloud, lit, far
    z0, z1 = F * DECK / HZ, 42000.0
    n = 800
    Z = np.sqrt(r.uniform(z0 ** 2, z1 ** 2, n))
    X = r.uniform(-0.7, 0.7, n) * Z
    street = np.cos(2 * np.pi * (0.55 * X + 0.83 * Z) / 5200 + r.uniform(0, 6.3))     # cloud streets along the wind
    keep = r.random(n) < 0.4 + 0.35 * street
    # first, high up and behind the heaps, thin streaks of cloud drawn out along the wind
    hy = yy / HZ
    sub = noise.rng(r.integers(1 << 30))
    streak = noise.smoothstep(0.6, 1.8, noise.stretched((w, h), 9, 110, sub).T[:h, :w] + 0.7 * noise.field((h, w), 30, sub))
    acc[0] = 0.4 * streak * noise.smoothstep(0.05, 0.3, hy) * noise.smoothstep(0.9, 0.55, hy) \
        * noise.smoothstep(-0.2, 0.8, noise.field((h, w), 90, sub))
    acc[1], acc[2] = 0.1 * acc[0], 0.5 * acc[0]
    tex = noise.fbm((256, 256), 10, sub, octaves=4)
    look = lambda u, v: ndimage.map_coordinates(tex, [v, u], order=1, mode="grid-wrap")
    for Xc, Zc in sorted(zip(X[keep], Z[keep]), key=lambda c: -c[1]):
        S = 1000 * np.exp(np.clip(r.normal(0, 0.55), -1.2, 1.3))
        a, b = 0.5 * S * F / Zc * (1 + 1.2 * Zc / z1), S * r.uniform(0.3, 0.55) * F / Zc * (1 - 0.65 * Zc / z1)
        if a < 140 and Zc < 8000:           # no little puffs up in the open blue
            continue
        cx, yb = CX + F * Xc / Zc, HZ - F * DECK / Zc
        x0, x1 = int(max(0, (cx - 2.2 * a) / s)), int(min(w, (cx + 2.2 * a) / s + 2))
        y0, y1 = int(max(0, (yb - 1.7 * b) / s)), int(min(h, (yb + 0.2 * b) / s + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        u = (xx[y0:y1, x0:x1] - cx) / a
        v = (yb - yy[y0:y1, x0:x1]) / b
        o = r.uniform(0, 256, 8)
        m = int(np.clip(3 + S / 600, 3, 7))
        uj = np.r_[r.normal(0, 0.1), r.uniform(-0.8, 0.8, m)]              # a heart to it, and lobes round that
        vj = np.r_[r.uniform(0.05, 0.2), r.uniform(0, 0.5, m) * np.sqrt(1 - np.abs(uj[1:]))]
        aj = np.r_[r.uniform(0.45, 0.6), r.uniform(0.25, 0.5, m)][:, None, None]
        bj = np.r_[r.uniform(0.55, 0.75), r.uniform(0.35, 0.65, m)][:, None, None]
        vp = np.maximum(v, 0)
        uw = u - np.clip(r.normal(0, 0.25), -0.35, 0.35) * vp + 0.18 * look(u * 7 + o[6], v * 7 + o[7])   # the wind shears it
        e = ((uw[None] - uj[:, None, None]) / aj) ** 2 + ((vp[None] - vj[:, None, None]) / bj) ** 2
        j = e.argmin(0)
        e = e.min(0)
        d = 1 - e + 0.4 * look(u * 14 + o[0], v * 12 + o[1]) + 0.2 * look(u * 36 + o[2], v * 30 + o[3])
        alpha = (noise.smoothstep(0.0, 0.35, d) * noise.smoothstep(-0.04, 0.1, v + 0.05 * look(u * 30 + o[4], 0 * u + o[5]))
                 * noise.smoothstep(2.2, 1.9, np.abs(u)) * noise.smoothstep(1.7, 1.45, v))
        alpha *= (r.uniform(0.4, 0.6) if r.random() < 0.12 else r.uniform(0.8, 1.0)) * (1 - 0.55 * Zc / z1)
        aj, bj = aj[:, 0, 0][j], bj[:, 0, 0][j]
        nx, ny = (uw - uj[j]) / aj, (vp - vj[j]) / bj
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        lit = nx * SUN[0] + ny * SUN[1] + nz * SUN[2] - 0.55 * noise.smoothstep(0.3, 0.0, v)
        for q, val in enumerate((1.0, lit, Zc / z1)):   # each over those beyond it
            acc[q, y0:y1, x0:x1] += (val - acc[q, y0:y1, x0:x1]) * alpha
    acc[1:] /= np.maximum(acc[0], 1e-3)
    acc[1] = ndimage.gaussian_filter(acc[1], 1.5)       # the lobes melt a little into each other
    # the brush goes round a cloud as a whole, along the edge of its mass, not round every billow in it
    gy, gx = np.gradient(ndimage.gaussian_filter(acc[0], 7))
    g = np.hypot(gx, gy)
    up = lambda a_: np.pad(ndimage.zoom(a_, s, order=1)[:int(HZ), :W], ((0, H - int(HZ)), (0, 0)))
    cloud, lit, far, c2, s2, gw = (up(a_) for a_ in (*acc, np.cos(2 * np.arctan2(gx, -gy)), np.sin(2 * np.arctan2(gx, -gy)),
                                                     noise.smoothstep(0.1, 0.5, g / (g.max() + 1e-6))))
    return cloud, lit, far, (0.5 * np.arctan2(s2, c2)).astype(np.float32), gw


def inn():
    """The inn in the scene (X, Y, Z metres): its front along the road, its end wall, a hipped roof of
    old tiles with two stacks, the windows with their shutters, the doors and the ground-floor windows
    drowned to their lintels, a painted board along the front and a sign hung out on an iron bracket.
    -> [(name, corners)], in the order they are drawn"""
    A, B, C = CORNER, CORNER + LONG * U1, CORNER + DEEP * U2
    R1, R2 = A + DEEP / 2 * (U1 + U2), A + (LONG - DEEP / 2) * U1 + DEEP / 2 * U2
    p = lambda q, y: (q[0], y, q[1])
    Ae, Be, Ce = A - 0.35 * (U1 + U2), B + 0.35 * (U1 - U2), C + 0.35 * (U2 - U1)
    out = [("front", [p(A, 0), p(B, 0), p(B, EAVES), p(A, EAVES)]), ("end", [p(A, 0), p(C, 0), p(C, EAVES), p(A, EAVES)])]

    def box(base, along, s, half, y0, y1, name):
        q0, q1 = base + (s - half) * along, base + (s + half) * along
        out.append((name, [p(q0, y0), p(q1, y0), p(q1, y1), p(q0, y1)]))

    box(A, U1, 0.22, 0.22, 0, EAVES, "quoin")
    box(A, U2, 0.2, 0.2, 0, EAVES, "quoin_sh")
    box(A, U1, LONG / 2, LONG / 2, 2.62, 2.8, "course")
    box(A, U2, DEEP / 2, DEEP / 2, 2.62, 2.8, "course_sh")
    box(A, U1, LONG / 2, LONG / 2, EAVES - 0.55, EAVES, "eave")
    box(A, U2, DEEP / 2, DEEP / 2, EAVES - 0.5, EAVES, "eave_sh")
    box(A, U1, LONG / 2 + 0.2, LONG / 2 - 0.5, 1.85, 2.45, "board")
    for k, s in enumerate((1.75, 4.25, 6.75, 9.25)):
        box(A, U1, s, 0.62 if k == 1 else 0.5, 0, 1.65 if k == 1 else 1.42, "door" if k == 1 else "pane")
        if k == 2:
            box(A, U1, s, 0.52, 3.05, 4.78, "shut")                    # one pair of shutters closed
        else:
            box(A, U1, s, 1.02, 3.0, 4.82, "shutter")                 # the others folded back either side
            box(A, U1, s, 0.5 - 0.04 * k, 3.05, 4.78, "pane")
    box(A, U2, 4.4, 0.52, 3.05, 4.78, "closed")
    box(A, U2, 2.3, 0.55, 0, 1.4, "pane")
    out += [("roof", [p(Ae, EAVES - 0.12), p((Ae + Be) / 2, EAVES - 0.24), p(Be, EAVES - 0.12), p(R2, RIDGE),
                      p((R1 + R2) / 2, RIDGE - 0.12), p(R1, RIDGE)]),
            ("hip", [p(Ae, EAVES - 0.12), p(Ce, EAVES - 0.12), p(R1, RIDGE)])]
    for c, top in ((R1 + 0.7 * U1, RIDGE + 1.35), (R2 - 0.2 * U1, RIDGE + 1.1)):   # the stacks
        a, b = 0.42 * U1, 0.3 * U2
        out += [("stack", [p(c - a - b, RIDGE - 1.2), p(c + a - b, RIDGE - 1.2), p(c + a - b, top), p(c - a - b, top)]),
                ("stack_sh", [p(c - a - b, RIDGE - 1.2), p(c - a + b, RIDGE - 1.2), p(c - a + b, top), p(c - a - b, top)]),
                ("pot", [p(c - 0.15 * U1 - b, top), p(c + 0.15 * U1 - b, top), p(c + 0.13 * U1 - b, top + 0.45),
                         p(c - 0.13 * U1 - b, top + 0.45)])]
    for s in (3.0, 8.0):                                             # two dormers in the roof
        c = A + s * U1 + 0.6 * U2
        box(c, U1, 0, 0.62, EAVES - 0.1, EAVES + 1.45, "dormer")
        box(c, U1, 0, 0.34, EAVES + 0.2, EAVES + 1.3, "dpane")
        out.append(("cap", [p(c - 0.78 * U1, EAVES + 1.38), p(c + 0.78 * U1, EAVES + 1.38), p(c + 0.4 * U2, EAVES + 2.15)]))
    s0 = A + 0.75 * U1                                               # the sign, hung out over the water
    out += [("iron", [p(s0, 3.25), p(s0 - 1.45 * U2, 3.22), p(s0 - 1.45 * U2, 3.12), p(s0, 3.1)]),
            ("sign", [p(s0 - 0.5 * U2, 3.12), p(s0 - 1.32 * U2, 3.12), p(s0 - 1.32 * U2, 2.45), p(s0 - 0.5 * U2, 2.45)])]
    return out


def trees(r):
    """The young trees along the drowned road, each drawn the way it grows: the trunk to the fork, then
    limbs that part and part again, shorter and thinner each time, rising toward the light, out to
    the finest twigs. -> limbs [(path px, half-width px, order 0 trunk .. 5 twig, tree)],
    feet [(x, waterline y, Z)]"""
    limbs, feet = [], []
    base = CORNER - 5.0 * U2
    q = np.linspace(0, 1, 6)
    for i, t in enumerate(np.arange(9.2, 64.0, 5.3) + r.normal(0, 0.5, 11)):
        X, Z = base + t * U1
        s = F / Z
        x0, y0 = see(X, 0, Z)
        girth, fork, reach = r.uniform(0.11, 0.16), r.uniform(2.3, 3.1), r.uniform(3.2, 4.6)
        full = r.random() < 0.35
        x1, y1 = see(X + r.normal(0, 0.08), fork, Z)
        limbs.append((np.stack([x0 + (x1 - x0) * q + r.normal(0, 0.02) * s * np.sin(np.pi * q), y0 + (y1 - y0) * q], 1),
                      girth * s, 0, i))

        def grow(p, a, L, w, k):
            ang = a + r.normal(0, 0.22) * q + 0.02 * k * q ** 2 * np.sign(np.cos(a))
            pts = p + np.cumsum(np.vstack([[0, 0], np.stack([np.cos(ang[:-1]), np.sin(ang[:-1])], 1) * L / 5]), 0)
            limbs.append((pts, w, k, i))
            if k < 5 and w > 0.25:
                for j in range(r.integers(2, 4 + full)):
                    b = ang[-1] + r.choice([-1, 1]) * r.uniform(0.25, 0.7)
                    b += 0.25 * ((-np.pi / 2 - b + np.pi) % (2 * np.pi) - np.pi)     # turning up toward the light
                    grow(pts[-1] - (pts[-1] - pts[-2]) * r.uniform(0, 0.6), b, L * r.uniform(0.62, 0.78),
                         w * r.uniform(0.5, 0.65), k + 1)

        for j in range(r.integers(3, 5)):
            grow(np.array([x1, y1]), -np.pi / 2 + r.normal(0, 0.45), reach * s * r.uniform(0.38, 0.5), girth * s * 0.62, 1)
        feet.append((x0, y0, Z))
    return limbs, feet


def boats():
    """The boats, each a flat hull in a stroke or two with its gunwale catching the light; one with a
    boatman standing at the stern, poling. -> [(path px, half-width px, what it is)] as seen, and mirrored"""
    out, refl = [], []
    for X, Z, L, turn, man in BOATS:
        d = np.array([U1[0] * np.cos(turn) - U1[1] * np.sin(turn), U1[0] * np.sin(turn) + U1[1] * np.cos(turn)])
        mid, s = np.array([X, Z]), F / Z
        for sign, store in ((1, out), (-1, refl)):
            def seg(a, b, y0, y1):
                t = np.linspace(0, 1, 6)
                q = a + (b - a) * t[:, None]
                return np.stack(see(q[:, 0], sign * (y0 + (y1 - y0) * t), q[:, 1]), 1)
            store.append((seg(mid - L / 2 * d, mid + L / 2 * d, 0.16, 0.18), 0.14 * s, "hull"))
            store.append((seg(mid - 0.46 * L * d, mid + 0.48 * L * d, 0.3, 0.32), 0.04 * s, "side"))
            store.append((seg(mid - 0.42 * L * d, mid + 0.45 * L * d, 0.36, 0.38), 0.03 * s, "gunwale"))
            if man:                                     # standing at the stern, leaning on the pole
                f = mid + 0.34 * L * d
                side = np.array([d[1], -d[0]])
                store.append((seg(f, f + 0.02 * side, 0.4, 0.95), 0.07 * s, "legs"))
                store.append((seg(f + 0.02 * side, f + 0.05 * side - 0.06 * d, 0.95, 1.5), 0.13 * s, "coat"))
                store.append((seg(f + 0.05 * side - 0.06 * d, f + 0.05 * side - 0.08 * d, 1.55, 1.72), 0.075 * s, "head"))
                store.append((seg(f - 0.1 * d + 0.04 * side, f - 0.45 * d, 1.42, 1.15), 0.035 * s, "coat"))
                store.append((seg(f - 0.05 * d, f - 1.3 * d, 1.85, 0.0), 0.016 * s + 0.6, "pole"))
    return out, refl


def plane(P, U, mirror=False):
    """For each pixel, where its ray meets the vertical plane through P along U: how far along it (m)
    and how high above the water, or, mirrored, the point whose reflection the pixel shows."""
    dx = (np.arange(W, dtype=np.float32) - CX) / F
    dy = -(np.arange(H, dtype=np.float32) - HZ) / F
    s = (P[0] - P[1] * dx) / (U[1] * dx - U[0])
    Z = P[1] + s * U[1]
    Y = CAM + dy[:, None] * Z[None]
    return np.broadcast_to(s[None], (H, W)), (-Y if mirror else Y)


def drawn(shapes, names, mirror, r, wobble=4.0):
    """The shapes drawn as the brush draws them, a label to each pixel, their edges wavering."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for name, pts in shapes:
        X, Y, Z = np.array(pts, np.float64).T
        x, y = see(X, -Y if mirror else Y, Z)
        d.polygon(list(zip(x, y)), fill=names.index(name) + 1)
    lab = np.asarray(im)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    jx = wobble * noise.field((H, W), 60, r) + 1.4 * noise.field((H, W), 12, r)
    jy = wobble * noise.field((H, W), 60, r) + 1.4 * noise.field((H, W), 12, r)
    return ndimage.map_coordinates(lab, [yy + jy, xx + jx], order=0, mode="nearest")


NAMES = ["front", "end", "quoin", "quoin_sh", "course", "course_sh", "eave", "eave_sh", "board", "door", "pane",
         "shutter", "closed", "roof", "hip", "stack", "stack_sh", "iron", "sign", "shut", "dormer", "cap", "pot", "dpane"]


def paint_on(img, lab, table, dim=1.0):
    """Fill each labelled place in `img` (in place) with the colour its function gives for those pixels."""
    for name, f in table.items():
        m = lab == NAMES.index(name) + 1
        if m.any():
            img[m] = f(m) * dim


def design(r):
    """What each place adds up to, seen from across the room, and the maps the brush goes by."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n60, n200, n500 = noise.field((H, W), 60, r), noise.field((H, W), 200, r), noise.field((H, W), 500, r)

    # the sky: a clear blue overhead, paler toward the sun beyond the top left, going to a warm haze
    # over the hills; and the clouds in it, cream where the sun is on them and lilac in their shade
    cloud, lit, far, round_, held = clouds(noise.rng(SKY))
    hy = np.clip(yy / HZ, 0, 1)
    toward = np.exp(-np.hypot(xx + 300, (yy + 600) * 1.3) / 1700)
    above = ramp(BLUE, 0.1 + 0.5 * hy ** 1.5 + 0.35 * toward + 0.06 * n500)
    above = mix(above, ramp(HAZE, 0.2 + 0.8 * noise.smoothstep(0.8, 1.0, hy) + 0.1 * n200), noise.smoothstep(0.62, 1.0, hy))
    lt = np.clip(0.42 + 0.55 * lit, 0, 1)
    col = mix(ramp(LILAC, 0.25 + 0.7 * lt + 0.08 * n60), ramp(CREAM, 0.2 + 0.8 * lt), noise.smoothstep(0.45, 0.75, lt))
    col = mix(col, ramp(ROSE, 0.4 + 0.5 * lt), 0.45 * noise.smoothstep(0.2, 0.8, far) * noise.smoothstep(0.3, 0.7, lt))
    col = mix(col, ramp(HAZE, 0.6 + 0.4 * lt), 0.55 * noise.smoothstep(0.25, 1.0, far))
    above = mix(above, col, cloud)

    # the far side of the river: a long low line of hills, meadows below them, trees along the water
    # and a village of small pale houses at their foot
    x = np.arange(W, dtype=np.float32)
    crest = (HZ - 28 - 50 * np.exp(-((x - 620) / 600) ** 2) - 30 * noise.smoothstep(1800, 2700, x)
             - 7 * noise.line1d(W, 260, r) - 3 * noise.line1d(W, 40, r)).astype(np.float32)
    hills = noise.smoothstep(-1.5, 1.5, yy - crest[None]) * (yy < HZ + 8)
    hill = ramp(HILLS, 0.4 + 0.4 * noise.smoothstep(crest[None], HZ, yy) + 0.12 * n60)
    hill = mix(hill, ramp(MEADOW, 0.5 + 0.2 * n60), 0.5 * noise.smoothstep(0.2, 1.2, noise.field((H, W), 40, r))
               * noise.smoothstep(crest[None] + 8, HZ - 4, yy))
    above = mix(above, hill, hills)
    tops = HZ - 4 - 12 * noise.smoothstep(-0.4, 1.6, noise.line1d(W, 18, r) + 0.6 * noise.line1d(W, 90, r))
    above = mix(above, ramp(HILLS, 0.15 + 0.15 * n60), noise.smoothstep(-1, 1, yy - tops[None]) * (yy < HZ + 6))
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    xv = 60.0
    while xv < 1450:
        bw, bh, rh = r.uniform(12, 30), r.uniform(8, 16), r.uniform(4, 8)
        yb = HZ + 3 - r.uniform(0, 5)
        d.rectangle([xv, yb - bh, xv + bw, yb], fill=1)
        d.polygon([(xv - 1, yb - bh), (xv + bw + 1, yb - bh), (xv + bw - 3, yb - bh - rh), (xv + 2, yb - bh - rh)], fill=2)
        xv += bw + r.uniform(2, 60) * (1 + 2 * (r.random() < 0.3))
    village = np.asarray(im)
    above = mix(above, ramp(CREAM, 0.15 + 0.3 * noise.field((H, W), 6, r)), (village == 1) * 0.8)
    above = mix(above, ramp(TILE, 0.55 + 0.2 * noise.field((H, W), 6, r)), (village == 2) * 0.7)

    # the water holds all of it upside down: the sky and the far bank mirrored about the far waterline,
    # everything standing in the flood about its own foot
    wl = HZ + 4.0
    deep = np.clip((yy - wl) / (H - wl), 0, 1)
    refl = ndimage.uniform_filter(above[np.clip(2 * wl - yy, 0, H - 1).astype(int), xx.astype(int)], (3, 61, 1))
    labs = {}
    for mirror in (False, True):
        lab = drawn(inn(), NAMES, mirror, r)
        Yf, Ye = plane(CORNER, U1, mirror)[1], plane(CORNER, U2, mirror)[1]

        def front(m, Yf=Yf):                # bound now: the next pass of the loop sees the reflection
            c = ramp(STONE, 0.52 + 0.12 * n60[m] + 0.1 * n200[m])
            c = mix(c, ramp(PINK, 0.55 + 0.12 * n60[m]), np.clip(0.75 * noise.smoothstep(2.9, 2.5, Yf[m])
                                                                 + 0.4 * noise.smoothstep(0.5, 1.3, n200[m]), 0, 1))
            return mix(c, ramp(OCHRE, 0.3), 0.45 * noise.smoothstep(0.9, 0.0, Yf[m] + 0.3 * n60[m]))   # damp above the water

        def roof(m):
            return mix(ramp(TILE, 0.45 + 0.15 * n60[m] + 0.1 * n200[m]), ramp(OCHRE, 0.5),
                       0.3 * noise.smoothstep(0.6, 1.5, n200[m] * 2 - n60[m]))

        tone = lambda pl, t0, k=0.0: (lambda m: ramp(pl, t0 + k * n60[m]))
        tab = {"front": front, "roof": roof,
               "end": lambda m, Ye=Ye: mix(ramp(SHADE, 0.14 + 0.06 * n60[m] + 0.36 * noise.smoothstep(2.6, 0.0, Ye[m])), PINK[0],
                                    0.25 * noise.smoothstep(2.6, 0.0, Ye[m])),
               "quoin": tone(STONE, 0.85, 0.1), "quoin_sh": tone(SHADE, 0.62), "course": tone(STONE, 0.92),
               "course_sh": tone(SHADE, 0.72), "eave": lambda m: mix(ramp(SHADE, 0.55 + 0.05 * n60[m]), ramp(PINK, 0.4), 0.3),
               "eave_sh": tone(SHADE, 0.25), "board": tone(BOARD, 0.5, 0.2), "door": tone(PANE, 0.12),
               "pane": tone(PANE, 0.22, 0.25), "shutter": tone(SHUTTER, 0.5, 0.15), "closed": tone(SHUTTER, 0.25, 0.1),
               "hip": tone(TILE_SH, 0.45, 0.15), "stack": tone(PINK, 0.45, 0.1), "stack_sh": tone(SHADE, 0.3),
               "iron": tone(PANE, 0.05), "sign": tone(SIGN, 0.35, 0.2), "shut": tone(SHUTTER, 0.55, 0.1),
               "dormer": tone(STONE, 0.7, 0.1), "cap": tone(TILE_SH, 0.55, 0.1), "pot": tone(TILE, 0.3, 0.1),
               "dpane": tone(PANE, 0.3, 0.2)}
        if mirror:
            paint_on(refl, lab, tab, 0.82)
        else:
            labs["inn"], inn_tab = lab, tab

    # the trees: the wood drawn in, and the haze of their twigs over whatever lies behind
    limbs, feet = trees(r)
    wood, twig = ([Image.new("F", (W, H), 0.0) for _ in range(2)] for _ in range(2))
    aim = Image.new("F", (W, H), 0.0)
    for path, w, k, i in limbs:
        for m in (0, 1):
            pts = [tuple(c) for c in (path * [1, -1] + [0, 2 * feet[i][1]] if m else path)]
            if k <= 2:
                ImageDraw.Draw(wood[m]).line(pts, fill=1.0, width=max(1, int(round(2 * w))))
            if k >= 3:
                ImageDraw.Draw(twig[m]).line(pts, fill=1.0, width=max(1, int(round(2 * w))))
        if k >= 1:
            ImageDraw.Draw(aim).line([tuple(c) for c in path], fill=float(np.arctan2(*(path[-1] - path[0])[::-1]) % np.pi + 1),
                                     width=max(3, int(round(2 * w))))
    env = [Image.new("F", (W, H), 0.0) for _ in range(2)]
    for path, w, k, i in limbs:
        if k >= 3:
            rad = 0.5 * F / feet[i][2] * r.uniform(0.5, 1.0)
            for m in (0, 1):
                x_, y_ = path[-1] if not m else path[-1] * [1, -1] + [0, 2 * feet[i][1]]
                ImageDraw.Draw(env[m]).ellipse([x_ - rad, y_ - rad, x_ + rad, y_ + rad], fill=1.0)
    cn = noise.smoothstep(-0.8, 0.8, noise.field((H, W), 14, r))
    crown = [np.maximum(noise.smoothstep(0.02, 0.3, ndimage.gaussian_filter(np.asarray(t_), 4)),
                        0.75 * noise.smoothstep(0.1, 0.7, ndimage.gaussian_filter(np.asarray(e_), 7)) * (0.55 + 0.45 * cn))
             for t_, e_ in zip(twig, env)]
    woods = [ndimage.gaussian_filter(np.asarray(w_), 0.7).clip(0, 1) for w_ in wood]
    tcol = mix(ramp(TWIGS, 0.2 + 0.2 * n60), ramp(BUDS, 0.35), 0.3 * noise.smoothstep(0.0, 1.2, noise.field((H, W), 35, r)))
    refl = mix(refl, tcol * 0.8, 0.7 * crown[1])
    refl = mix(refl, ramp(BARK, 0.25) * 0.9, woods[1])

    # the ripples shift the reflection up and down in bands drawn out across the water, more toward us
    wave = 0.7 * noise.stretched((W, H), 5, 260, r).T[:H, :W] + 0.3 * noise.stretched((W, H), 2.5, 70, r).T[:H, :W]
    dyw = (4 + 42 * deep ** 1.1) * wave
    dxw = (1 + 5 * deep) * noise.stretched((W, H), 8, 120, r).T[:H, :W]
    refl = np.stack([ndimage.map_coordinates(refl[..., c], [np.clip(yy + dyw, 0, H - 1), np.clip(xx + dxw, 0, W - 1)], order=1)
                     for c in range(3)], -1)
    # and where a breath of wind ruffles it, in long bands, the water shows the blue from higher up instead
    ruffle = noise.smoothstep(0.4, 1.4, noise.stretched((W, H), 26, 700, r).T[:H, :W] + 0.3 * noise.field((H, W), 120, r))
    refl = mix(refl, ramp(BLUE, 0.55 - 0.28 * deep + 0.05 * n200), 0.75 * ruffle * noise.smoothstep(0.02, 0.2, deep))
    # and the flood's own pale colour shows more close to, where the eye looks into it rather than along it
    own = mix(ramp(SILT, 0.55 - 0.2 * deep + 0.1 * n200), ramp(OCHRE, 0.5 + 0.15 * n60),
              0.4 * noise.smoothstep(-0.6, 1.0, noise.field((H, W), 300, r)))
    own = mix(own, ramp(JADE, 0.55), 0.25 * noise.smoothstep(0.2, 1.4, n500))
    own = mix(own, ramp(BLUE, 0.3 + 0.1 * n200), 0.45 * deep ** 1.3)
    want = np.where((yy < wl)[..., None], above, mix(own, refl * 0.96, 0.92 - 0.47 * deep ** 0.8))
    paint_on(want, labs["inn"], inn_tab)
    want = mix(want, tcol, 0.6 * crown[0])
    want = mix(want, ramp(BARK, 0.35 + 0.2 * n60), woods[0])
    want = (want * np.exp(0.04 * np.stack([noise.field((H, W), 260, r), noise.field((H, W), 260, r)], -1)
                          @ np.float32([[0.5, -0.1, -0.4], [-0.3, 0.35, -0.05]]))).astype(np.float32)

    # which way the strokes run: across the sky in a slant that flattens toward the horizon and turns
    # round the clouds; level over the water; up the walls in places and along them in others, up the
    # slopes of the roof; out along the twigs
    wind = (-0.32 + 0.3 * noise.field((H, W), 700, r)) * (1 - 0.75 * noise.smoothstep(0.7, 1.0, hy))
    flow = np.where(yy < wl, blend(wind, round_, 0.8 * held), 0.03 * noise.field((H, W), 300, r))
    flow = np.where((yy > crest[None] - 3) & (yy < wl + 2), 0.08 * n60, flow)
    lab = labs["inn"]
    face = lambda *ns: np.isin(lab, [NAMES.index(n) + 1 for n in ns])
    across = noise.field((H, W), 40, r) < -0.2
    up_ = np.pi / 2 + 0.06 * n60
    along1 = np.arctan2(HZ - yy, CX + F * U1[0] / U1[1] - xx)
    along2 = np.arctan2(HZ - yy, CX + F * U2[0] / U2[1] - xx)
    flow = np.where(face("front", "quoin", "eave"), np.where(across, along1, up_), flow)
    flow = np.where(face("end", "quoin_sh", "eave_sh"), np.where(across, along2, up_), flow)
    flow = np.where(face("course", "board"), along1, flow)
    flow = np.where(face("course_sh", "iron", "sign"), along2, flow)
    flow = np.where(face("door", "pane", "dpane", "shutter", "closed", "shut", "dormer", "stack", "stack_sh", "pot"), up_, flow)
    flow = np.where(face("cap"), along1, flow)
    A, B, C = CORNER, CORNER + LONG * U1, CORNER + DEEP * U2
    R1, R2 = A + DEEP / 2 * (U1 + U2), A + (LONG - DEEP / 2) * U1 + DEEP / 2 * U2
    slope = lambda e, g: np.arctan2(*(np.subtract(see(g[0], RIDGE, g[1]), see(e[0], EAVES, e[1])))[::-1])
    flow = np.where(face("roof"), slope((A + B) / 2, (R1 + R2) / 2), flow)
    flow = np.where(face("hip"), slope((A + C) / 2, R1), flow)
    twigs = np.asarray(aim)
    idx = ndimage.distance_transform_edt(twigs == 0, return_distances=False, return_indices=True)
    flow = np.where(crown[0] > 0.2, twigs[idx[0], idx[1]] - 1, flow).astype(np.float32)
    k = np.where(yy < wl, 0.45 + 0.95 * (1 - hy) ** 1.3, 0.3 + 1.15 * deep ** 0.9).astype(np.float32)
    return dict(want=want, cloud=cloud, lit=lit, far=far, crest=crest, wl=wl, deep=deep, inn=lab, crown=crown[0],
                limbs=limbs, feet=feet, flow=flow, k=k)


def quads(shapes, mirror=False):
    """Each shape's corners on the canvas, as seen or as the water shows it. -> [(name, (4, 2) px)]"""
    out = []
    for name, pts in shapes:
        X, Y, Z = np.array(pts, np.float64).T
        c = np.stack(see(X, -Y if mirror else Y, Z), 1)
        out.append((name, c if len(c) == 4 else np.vstack([c, c[-1:]])))      # a triangle as a quad with its apex twice
    return out


def across(c, m, upright, r, wob=0.05):
    """`m` strokes filling a quad (bottom left, bottom right, top right, top left) side by side, each
    run up it (or along it), a little crooked and not quite reaching its ends. -> paths, half-widths"""
    paths, ws = [], []
    for f in (np.arange(m) + 0.5) / m + r.normal(0, 0.06 / m, m):
        if upright:
            a, b, span = c[0] + (c[1] - c[0]) * f, c[3] + (c[2] - c[3]) * f, np.linalg.norm(c[1] - c[0])
        else:
            a, b, span = c[0] + (c[3] - c[0]) * f, c[1] + (c[2] - c[1]) * f, np.linalg.norm(c[3] - c[0])
        q = np.linspace(r.uniform(0.02, 0.1), r.uniform(0.88, 0.98), 5)[:, None]
        paths.append((a + (b - a) * q + r.normal(0, wob * span + 0.3, (5, 2)))[::r.choice([-1, 1])])
        ws.append(span / m * r.uniform(0.55, 0.75) + 0.3)
    return paths, ws


def trim(paths, block):
    """Cut each path where it first runs into `block` (H,W), keeping what comes before it.
    -> the paths left, and which were kept"""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, 4 * (n - 1) + 1)
    i = np.minimum(t.astype(int), n - 2)
    fine = paths[:, i] + (paths[:, i + 1] - paths[:, i]) * (t - i)[None, :, None]
    hit = block[at(fine.reshape(-1, 2))].reshape(N, -1) > 0.5
    stop = np.where(hit.any(1), hit.argmax(1), len(t))
    keep = stop >= 5
    return [fine[q, :stop[q]] for q in np.nonzero(keep)[0]], keep


def paint(seed=1876):
    r = noise.rng(seed)
    D = design(r)
    want, flow, k, cloud, deep, wl = (D[n] for n in ("want", "flow", "k", "cloud", "deep", "wl"))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cloth = canvas.duck((H, W), seed, tint="#e5e0d3", thread=3.0)
    lab = D["inn"]
    face = lambda *ns: np.isin(lab, [NAMES.index(n) + 1 for n in ns]).astype(np.float32)
    house = (lab > 0).astype(np.float32)
    crest = D["crest"][None]
    sky = (yy < crest - 2) * (1 - house)
    bank = ((yy >= crest - 2) & (yy < wl + 2)) * (1 - house)
    water = (yy >= wl + 2) * (1 - house)
    lum = want @ LUMA
    busy = noise.smoothstep(0.012, 0.05, np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, (5, 31))
                                                            - ndimage.uniform_filter(lum, (5, 31)) ** 2, 0)))

    # the canvas rubbed over first with thin paint of the colour of each place, the ground glowing through
    rag = noise.stretched((H, W), 14, 90, r)
    alpha = np.clip(0.78 + 0.06 * rag + 0.12 * (0.5 - cloth.tooth), 0, 1)[..., None]
    rgb = (cloth.color * (1 - alpha) + ndimage.uniform_filter(want, (7, 7, 1)) * (1 + 0.03 * rag[..., None]) * alpha)
    rgb = rgb.astype(np.float32)
    height = cloth.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def broken(P, fams, pure=(0.15, 0.4), odds=0.12, value=0.05):
        """The brush for each stroke: the colour of the place, a second pushed part of the way toward
        one of the colours of `fams` brought to the same value and streaked beside it, now and then a
        third further off."""
        T = want[at(P)]
        n = len(P)
        lu = T @ LUMA
        table = np.concatenate(fams)

        def hue(f):
            c = valued(table[r.integers(0, len(table), n)], lu * np.exp(r.normal(0, value, n)))
            return T + (c - T) * f[:, None]
        c1 = T * np.exp(r.normal(0, 0.02, (n, 3)) + r.normal(0, value, (n, 1)))
        c3 = np.where((r.random(n) < odds)[:, None], hue(np.full(n, 0.7)), T * np.exp(r.normal(0, 0.03, (n, 1))))
        share = np.stack([r.uniform(0.5, 0.75, n), r.uniform(0.15, 0.35, n), r.uniform(0.05, 0.2, n)], 1)
        return np.stack([c1, hue(r.uniform(*pure, n)), c3], 1).clip(0, 1).astype(np.float32), share

    def loads(colours, tone, accent=None, odds=0.2, spread=0.08):
        """A brush loaded from one palette at `tone` (0 dark .. 1 light), a neighbour streaked in, and
        now and then an accent picked up from elsewhere."""
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

    def go(odds, region, spacing, length, width, load, field=None, scale=None, short=None, tilt=0.0, bend=0.0, dry=0.0,
           n=8, group=(1, 1), block=None, **kw):
        """A passage. The strokes go down in small groups, a few laid side by side from one loading of the
        brush at one slant, each drier than the last, as a painter hatches; no two of a size."""
        P = strew(spacing * np.sqrt(np.mean(group)), odds)
        G = len(P)
        if not G:
            return
        f = flow if field is None else field
        m = r.integers(group[0], group[1] + 1, G)
        own = np.repeat(np.arange(G), m)
        N = len(own)
        j = np.arange(N) - np.repeat(np.cumsum(m) - m, m)
        lean = r.normal(0, tilt, G)[own]
        a = f[at(P)][own] + lean
        s = scale[at(P)][own] if scale is not None else np.ones(N)
        wd = r.uniform(*width, G)[own] * np.exp(r.normal(0, 0.18, N)) * s ** 0.8
        side = (j - (m[own] - 1) / 2) * 1.6 * wd
        Q = P[own] + np.stack([-np.sin(a), np.cos(a)], 1) * side[:, None] \
            + np.stack([np.cos(a), np.sin(a)], 1) * (r.normal(0, 0.4, N) * wd)[:, None]
        L = r.uniform(*length, G)[own] * np.exp(r.normal(0, 0.22, N)) * s ** 0.9
        if short is not None:
            L = L * (1 - short[at(Q)])
        paths = impasto.follow(f, Q, np.maximum(L, 2.5 * wd), n, r.normal(0, bend, G)[own], lean)
        stay = region[at(Q)] > 0.5
        if block is not None:                       # a stroke stops where it runs into what is painted after it
            cut, kept = trim(paths, block)
            stay &= kept
            paths = np.empty(N, object)
            paths[kept] = cut
        else:
            paths = list(paths)
        flip = (r.random(G) < 0.5)[own]
        paths = [p[::-1] if fl else p for p, fl, ok in zip(paths, flip, stay) if ok]
        cols, share = load(Q)
        d = (r.uniform(*dry, G) if isinstance(dry, tuple) else np.full(G, dry))[own]
        impasto.lay(rgb, height, paths, wd[stay], (cols * np.exp(r.normal(0, 0.02, (N, 1, 3))))[stay].astype(np.float32),
                    r, share=share[stay], wet=wet, dry=np.clip(d + 0.08 * j, 0, 0.85)[stay], **kw)

    thin = lambda m, k0=0.6: (m * np.clip((k0 / k) ** 1.6, 0, 1)).astype(np.float32)     # fewer, where they are larger
    soft = dict(ends=(0.4, 0.25), taper=0.3, fray=2.0)
    level = (0.02 * noise.field((H, W), 300, r)).astype(np.float32)
    airy = lambda P: broken(P, (BLUE, BLUE[2:]), (0.1, 0.3), 0.08, 0.04)
    puff = lambda P: broken(P, (CREAM, LILAC, ROSE), (0.1, 0.35), 0.12, 0.04)
    flood = lambda P: broken(P, (BLUE[3:], LILAC[2:], OCHRE, CREAM[:3], JADE), (0.12, 0.4), 0.15, 0.05)

    # the first sitting, all of it wet into wet: the sky and the water laid in with a broad brush, thin
    plain = lambda P: broken(P, (BLUE,), (0.0, 0.1), 0.0, 0.02)
    broad = dict(scale=k, block=house, thick=0.03, grooves=0.15, lips=0.02, land=0.1, lift=0.05, pickup=0.6, merge=5,
                 spent=0.6, fray=3, hide=0.7)
    go(sky + bank, sky + bank, 120, (110, 230), (15, 24), plain, tilt=0.2, taper=0.4, **broad)
    go(water, water, 100, (160, 380), (9, 16), plain, field=level, tilt=0.015, taper=0.45, ends=(0.6, 0.5), **broad)

    # the sky: the blue in short strokes slanting across it, every one a little different, then the
    # clouds in broad strokes bent round their heaps, cream in the sun and lilac in their shade, worked
    # into the wet blue at their edges
    go(thin(sky * (1 - noise.smoothstep(0.2, 0.55, cloud))), sky, 24, (50, 130), (6, 12), airy, scale=k, tilt=0.18,
       bend=0.002, group=(1, 3), block=house, thick=0.12, grooves=0.4, pickup=0.45, merge=2, dry=(0, 0.3), spent=0.7, **soft)
    go(thin(sky * noise.smoothstep(0.1, 0.45, cloud)), sky, 30, (60, 150), (10, 18), puff, scale=k, tilt=0.25, bend=0.002,
       group=(1, 2), block=house, thick=0.18, grooves=0.35, pickup=0.6, merge=5, dry=(0, 0.15), spent=0.6, **soft)
    # the far bank in small level touches
    go(bank, bank, 8, (10, 30), (2.0, 3.6), lambda P: broken(P, (HILLS, MEADOW), (0.1, 0.3), 0.06), field=level,
       block=house, tilt=0.12, thick=0.1, pickup=0.4, merge=1.5, dry=(0, 0.3), **soft)
    # the water in long level strokes, small far off and larger toward us, short where a reflection
    # breaks it, carrying the sky, the lilac, the cream of the clouds and the ochre of the flood
    go(thin(water, 0.5), water, 18, (60, 200), (3.5, 7), flood, field=level, scale=k, short=0.7 * busy, tilt=0.02,
       block=house, bend=0.0005, thick=0.08, grooves=0.35, pickup=0.5, merge=2, dry=(0.05, 0.35), ends=(0.6, 0.5),
       taper=0.45, fray=2.2, spent=0.75)

    # the inn: its front in strokes up the wall and along it, its end wall cool, the roof down the slope
    stone = lambda P: broken(P, (STONE, PINK, OCHRE), (0.1, 0.35), 0.15, 0.05)
    cool = lambda P: broken(P, (SHADE, LILAC, PINK[:2]), (0.1, 0.3), 0.12, 0.05)
    tiles = lambda P: broken(P, (TILE, PINK, OCHRE[:3], TILE_SH), (0.15, 0.45), 0.2, 0.08)
    walls = dict(tilt=0.1, group=(1, 3), thick=0.14, pickup=0.45, merge=1.5, dry=(0, 0.3), ends=(0.3, 0.15), taper=0.25,
                 fray=1.4)
    front = face("front", "quoin", "course", "eave", "board", "door", "pane", "shutter", "shut")
    go(front, front, 9, (16, 46), (3.5, 7), stone, **walls)
    end = face("end", "quoin_sh", "course_sh", "eave_sh", "closed", "pane") * (xx > see(CORNER[0], 0, CORNER[1])[0] - 4)
    go(end, end, 8, (14, 40), (3, 6), cool, **{**walls, "dry": (0, 0.12)})
    roof = face("roof", "hip")
    go(roof, roof, 9, (18, 50), (3.5, 7), tiles, **walls)
    go(face("stack", "stack_sh"), face("stack", "stack_sh"), 5, (10, 24), (2.5, 4), stone, **walls)
    wet[:] = 0
    height += 0.25 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2))     # crusts left between sittings

    # the second sitting, over dry paint: the lights of the clouds in thicker creamy strokes, the blue cut
    # back into their edges, and long dry strokes of light dragged across the lower sky
    lit = D["lit"]
    glow = sky * noise.smoothstep(0.45, 0.75, cloud) * noise.smoothstep(0.25, 0.7, lit) * (1 - 0.6 * D["far"])
    go(thin(glow) * 0.5, sky, 18, (40, 100), (6, 12), lambda P: loads(CREAM, 0.55 + 0.4 * r.random(len(P)), ROSE, 0.15),
       scale=k, tilt=0.25, bend=0.003, group=(1, 2), block=house, thick=0.24, grooves=0.45, lips=0.08, land=0.35, lift=0.2,
       pickup=0, merge=0, dry=(0.0, 0.35), ends=(0.3, 0.15), taper=0.3, fray=1.6)
    edge = sky * noise.smoothstep(0.08, 0.3, cloud) * noise.smoothstep(0.6, 0.35, cloud)
    go(thin(edge) * 0.7, sky, 20, (25, 70), (4, 8), airy, scale=k, tilt=0.4, bend=0.003, block=house, thick=0.14, pickup=0,
       merge=0, dry=(0.0, 0.35), **soft)
    go(thin(sky * noise.smoothstep(0.55, 0.95, yy / HZ)) * 0.5, sky, 30, (80, 220), (5, 10),
       lambda P: broken(P, (CREAM, ROSE, HAZE), (0.3, 0.6), 0.1), scale=k, tilt=0.05, block=house, thick=0.08, pickup=0,
       merge=0, dry=(0.45, 0.7), **soft)

    # the water: dark and pale dashes where it is ruffled, small far off and larger toward us
    rough = water * noise.smoothstep(-0.3, 1.0, noise.stretched((W, H), 30, 500, r).T[:H, :W])
    dash = dict(field=level, scale=k, block=house, thick=0.06, pickup=0, merge=0, ends=(0.6, 0.4), taper=0.5, fray=1.8)
    go(thin(rough, 0.5) * (0.4 + 0.9 * deep), water, 16, (25, 80), (2.0, 4.5),
       lambda P: broken(P, (BLUE[1:4], LILAC[:3], JADE[:2]), (0.4, 0.7), 0.2, 0.1), tilt=0.04, dry=(0.1, 0.45), **dash)
    go(thin(water * (1 - 0.6 * busy), 0.5) * 0.4, water, 18, (40, 140), (2.0, 4.0),
       lambda P: broken(P, (CREAM, BLUE[5:], ROSE), (0.4, 0.7), 0.1, 0.06), tilt=0.02, dry=(0.35, 0.65), **dash)

    # the trees, over the dry sky: a haze of twigs scumbled in, then the limbs and the finer twigs drawn
    # over it, the trunks dark with the light down one side
    cr = D["crown"]
    go(cr * 0.9, cr > 0.05, 9, (12, 34), (2.5, 5.5), lambda P: loads(TWIGS, 0.12 + 0.35 * r.random(len(P)), BUDS, 0.3),
       tilt=0.3, thick=0.1, pickup=0, merge=0, dry=(0.25, 0.6), hide=0.8, **soft)
    limbs, feet = D["limbs"], D["feet"]
    for lo, hi, kw in ((3, 5, dict(dry=0.3, thick=0.08, hide=0.85, spent=0.8, taper=0.6, ends=(0.2, 0.0), tails=0.6,
                                   fray=0.6)),
                       (0, 2, dict(dry=0.1, thick=0.16, hide=0.95, spent=0.5, taper=0.45, ends=(0.3, 0.05), tails=0.4,
                                   fray=1.0))):
        sel = [(p, w, kk, i) for p, w, kk, i in limbs if lo <= kk <= hi and w > 0.3]
        tone = np.array([0.08 + 0.07 * kk + 0.25 * (feet[i][2] - 36) / 40 for p, w, kk, i in sel])
        lay([p for p, *_ in sel], np.array([max(0.7, 1.05 * w) for _, w, *_ in sel]),
            loads(BARK, tone.clip(0, 0.9), TWIGS, 0.2), pickup=0.1, merge=0.5, **kw)
    trunks = [(p, w, i) for p, w, kk, i in limbs if kk == 0]
    lay([p - [0.55 * w, 0] for p, w, i in trunks], np.array([0.3 * w + 0.5 for p, w, i in trunks]),
        loads(BARK_LIT, np.full(len(trunks), 0.4)), dry=0.3, thick=0.12, pickup=0, spent=0.7, taper=0.3, fray=0.8)
    # and their reflections, broken into short touches that tremble down the water, shifted further
    # the further down they go
    paths, ws, tones = [], [], []
    for p, w, kk, i in limbs:
        if kk > 2 or w < 0.6:
            continue
        y0 = feet[i][1]
        m = p * [1, -1] + [0, 2 * y0]
        Lp = np.r_[0, np.cumsum(np.hypot(*np.diff(m, axis=0).T))]
        a = r.uniform(0, 6)
        while a < Lp[-1] - 2:
            b = min(Lp[-1], a + r.uniform(6, 24))
            q = np.linspace(a, b, 3)
            seg = np.stack([np.interp(q, Lp, m[:, 0]), np.interp(q, Lp, m[:, 1])], 1)
            dd = max(seg[:, 1].mean() - y0, 0) / 200
            seg[:, 0] += r.normal(0, 0.6 + 3 * dd)
            if seg[:, 1].max() < H - 4 and r.random() < 0.8 - 0.2 * kk:
                paths.append(seg)
                ws.append(w * r.uniform(0.8, 1.4))
                tones.append(0.15 + 0.1 * kk + 0.3 * dd)
            a = b + r.uniform(1, 9) * (1 + dd)
    lay(paths, np.array(ws), loads(BARK, np.clip(tones, 0, 0.9), LILAC[:3], 0.3), dry=r.uniform(0.15, 0.45, len(paths)),
        thick=0.06, pickup=0, hide=0.8, spent=0.6, taper=0.3, ends=(0.4, 0.3), fray=1.4)

    # the inn: the windows, the doors drowned to their lintels, the shutters, the painted board and the
    # sign, each put in with a stroke or two over the wall, the long ones in pieces
    pick = {"pane": (PANE, 0.5), "door": (PANE, 0.3), "shutter": (SHUTTER, 0.5), "closed": (SHUTTER, 0.25),
            "shut": (SHUTTER, 0.6), "board": (BOARD, 0.5), "sign": (SIGN, 0.4), "iron": (PANE, 0.1), "dormer": (STONE, 0.75),
            "cap": (TILE_SH, 0.55), "pot": (TILE, 0.3), "dpane": (PANE, 0.45)}
    paths, ws, fams = [], [], []
    for name, c in quads(inn()):
        if name in pick:
            wide = name in ("board", "iron", "cap")
            span = np.linalg.norm(c[3] - c[0]) if wide else np.linalg.norm(c[1] - c[0])
            glass = name in ("pane", "door", "dpane")
            p_, w_ = across(c, max(1, int(span / (6 if wide else 22 if glass else 14))), not wide, r)
            if glass and np.linalg.norm(c[3] - c[0]) > 12:      # the shadow under the lintel
                top = c[3] + (c[0] - c[3]) * 0.12
                p_.append(np.stack([top, top + (c[2] - c[3])]) + r.normal(0, 0.6, (2, 2)))
                w_.append(0.1 * np.linalg.norm(c[3] - c[0]) + 0.8)
            for q_, wq in zip(p_, w_):
                cuts = np.sort(r.uniform(0, 1, r.integers(0, 3))) if wide else []
                for u0, u1 in zip(np.r_[0, cuts], np.r_[cuts, 1]):     # the long ones in a few pieces
                    t, g = np.linspace(u0, u1, 4) * (len(q_) - 1), np.arange(len(q_))
                    paths.append(np.stack([np.interp(t, g, q_[:, 0]), np.interp(t, g, q_[:, 1])], 1))
                    ws.append(wq * r.uniform(0.8, 1.15))
                    fams.append(name)
    fams = np.array(fams)
    cols = np.empty((len(paths), 3, 3), np.float32)
    share = np.empty((len(paths), 3))
    for name, (pl, t0) in pick.items():
        sel = fams == name
        if sel.any():
            cols[sel], share[sel] = loads(pl, t0 + 0.12 * r.normal(0, 1, sel.sum()), SHADE[:2] if pl is PANE else PANE[2:],
                                          0.15)
    lay(paths, np.array(ws), (cols, share), dry=r.uniform(0.0, 0.3, len(paths)), thick=0.14, pickup=0.15, spent=0.6,
        taper=0.1, ends=(0.1, 0.05), fray=1.0, tails=0.0)
    # their reflections, broken into the water below
    paths, ws, kinds = [], [], []
    for name, c in quads(inn(), True):
        if name in ("pane", "door", "shutter", "shut", "sign", "board"):
            span = np.linalg.norm(c[1] - c[0])
            y0_, y1_ = c[:, 1].min(), c[:, 1].max()
            y = y0_ + r.uniform(0, 6)
            while y < min(y1_, H - 5):
                L = r.uniform(5, 14) * (1 + (y - wl) / 400)
                xa = c[0, 0] + (c[1, 0] - c[0, 0]) * r.uniform(-0.1, 0.2) + r.normal(0, 2 + (y - wl) / 120)
                paths.append(np.array([[xa, y], [xa + span * r.uniform(0.7, 1.1), y + r.normal(0, 1)]]))
                ws.append(L / 2)
                kinds.append(name)
                y += L + r.uniform(2, 12)
    kinds = np.array(kinds)
    cols, share = loads(PANE, np.full(len(paths), 0.35) + 0.1 * r.normal(0, 1, len(paths)), LILAC[:3], 0.4)
    sh = (kinds == "shutter") | (kinds == "shut")
    cols[sh] = loads(SHUTTER, np.full(sh.sum(), 0.3))[0]
    for kind, (pl, t0) in (("sign", (SIGN, 0.3)), ("board", (BOARD, 0.35))):
        sg = kinds == kind
        cols[sg] = loads(pl, np.full(sg.sum(), t0))[0]
    lay(paths, np.array(ws), (cols, share), dry=r.uniform(0.2, 0.5, len(paths)), thick=0.06, pickup=0, hide=0.7,
        spent=0.7, taper=0.3, ends=(0.5, 0.4), fray=1.6)

    # the boats, a few dark strokes each, and their reflections broken beneath them
    seen_, mirrored = boats()
    for strokes, kw in ((mirrored, dict(dry=0.3, hide=0.6, spent=0.8, fray=1.6, tails=0.5)),
                        (seen_, dict(dry=0.05, hide=0.95, spent=0.4, fray=1.0, tails=0.3))):
        paths = [p for p, w, kind in strokes]
        kind = np.array([kd for p, w, kd in strokes])
        tone = np.select([kind == "side", kind == "coat", kind == "head"], [0.55, 0.15, 0.3], 0.2)
        cols, share = loads(HULL, tone + 0.05 * r.normal(0, 1, len(paths)), LILAC[:2], 0.2)
        cols[kind == "gunwale"] = loads(OCHRE, np.full((kind == "gunwale").sum(), 0.75))[0]
        lay(paths, np.array([w for p, w, kd in strokes]), (cols, share), thick=0.12, pickup=0.1, taper=0.4, ends=(0.3, 0.1),
            **kw)

    img = dabs.shine(rgb, ndimage.gaussian_filter(height, 0.8), light=LIGHT, relief=0.5, gloss=0, reach=(0.8, 1.12))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
