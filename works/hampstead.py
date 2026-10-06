"""Summer Cloud, Hampstead. Oil on paper laid down on board.

In the summers of 1821 and 1822, living at Hampstead, Constable went up onto the Heath day after day to
paint the sky, which he called the chief organ of sentiment in a landscape. He worked on sheets of paper
primed with a warm red-brown and pinned in the lid of his paint box, and a study took him an hour or less,
about as long as a cloud keeps its shape. The blue went on first, scrubbed round where the clouds would be,
thin enough for the ground to glow through it. The clouds went over it while it was wet: their shadows in
grey and violet, in strokes that go round each heap of vapour, then the lights, stiff cream and white laid
on the sunlit tops with a hog brush and the palette knife and dragged off where the wind tears at them.
Last he cut the blue back into the edges of the heads, so that one is found sharp against the sky and the
next is lost in it.

Here a summer cumulus towers over the trees of the Heath, its tops in the sun and its underside going
grey and violet, while smaller clouds drift across the blue to the left. At the foot is a strip of dark
trees, and between them a glint of the far country. The sheet has been laid down on board and its edges
show. Two corners are still pinned; at a third the pin has gone and the corner has lifted, and the fourth
is folded under.
"""

import numpy as np
from scipy import ndimage

from atelier import dabs, impasto, noise
from atelier.color import lin

TITLE = "Summer Cloud, Hampstead"
DATE = "2026"
MEDIUM = ("Oil on paper laid down on board, painted out of doors over a red-brown ground, the lights laid on "
          "thick with a stiff brush and the palette knife")
AFTER = "John Constable, the cloud studies at Hampstead, 1821–22"
ROOM = "Open Air"
YEAR = 1822
PLACE = "Hampstead Heath"
REGION = "Europe"
NOTE = ("A summer cumulus towering over the trees of the Heath, painted in under an hour on a sheet of paper. "
        "The lights are thick; where the paint is thin, the red-brown ground shows through.")

H, W = 2090, 2620                     # the board; the sheet on it is about 24 by 30 cm
SHEET = np.float32([[86, 80], [2536, 88], [2530, 2008], [80, 1998]])   # its corners: top left, top right, bottom right, bottom left
HORIZON = 1868
SUN = np.float32([-0.52, -0.62, 0.58])  # toward the sun, high on the left
LIGHT = (-0.6, -0.5, 0.62)              # the gallery lamp
GROUND, PAPER, BOARD = "#8c5e4a", "#e6dcc6", "#4e4236"


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light
SKY = pal("#36639c", "#416ea8", "#4e7cb4", "#5e8bbf", "#7199c7", "#88abd0", "#a1bdd9", "#b8cbdd", "#c9d3da")
HAZE = pal("#b9bfc6", "#c9c8c5", "#d6d0c4")                 # the warm pale low down, toward the heath
CLOUD = pal("#52525f", "#5f5e6c", "#6d6b7a", "#7c7a88", "#8c8996", "#9d99a4", "#afaab1", "#c1bbba",
            "#d3c9c0", "#e2d7c6", "#ece3cf", "#f4eedd")      # from the grey-violet underside to the sunlit tops
WHITE = pal("#f3ead4", "#f8f2e3", "#fbf8ef")                 # lead white, a touch of Naples yellow in some
WARM = pal("#b9a497", "#c8b3a2", "#a89a98")                  # light thrown up warm into the shadows
TREES = pal("#15170f", "#1c1f14", "#242718", "#2d301e", "#373a24", "#44462c")
BROWN = pal("#2a2219", "#37291d", "#4a3624")                 # the ground of the heath, and the trunks
FAR = pal("#6b7c90", "#7f8f9f", "#96a3ae")                    # the far country
GLINT = pal("#c8bd92", "#d6cda4", "#e0d8b4")

# the clouds as heaps of vapour: x, y, half-width, half-height, the size of a head, how far off (haze), floor
MASSES = [(1730, 1360, 740, 160, 150, 0.0, 1508),            # the great cloud: its broad base, in its own shadow
          (1690, 1010, 370, 320, 150, 0.0, 1508),            # its body
          (1640, 650, 290, 290, 120, 0.0, 1508),             # the tower
          (1590, 370, 180, 130, 85, 0.0, 1508),              # the heads boiling up at the top
          (1890, 520, 140, 150, 75, 0.0, 1508),
          (2310, 1090, 340, 270, 130, 0.0, 1508),            # its right flank, running off the sheet
          (1170, 1260, 250, 150, 105, 0.0, 1508),            # and its left, lower
          (520, 830, 160, 70, 60, 0.35, 880),                # small clouds drifting in the blue
          (330, 1280, 210, 75, 66, 0.45, 1336),
          (830, 1560, 180, 55, 52, 0.55, 1596),
          (1060, 540, 110, 48, 44, 0.3, 572)]


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(np.asarray(t).astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def loads(colours, tone, r, spread=0.05, third=None, odds=0.2):
    """The brush for each stroke: the paint for its tone (0 dark, 1 light), with a paint a little
    lighter or darker streaked in, and now and then a third picked up from elsewhere on the palette."""
    N = len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 1)
    j = np.clip(t + r.choice([-1, 1], N) / (len(colours) - 1), 0, 1)
    k = ramp(colours, np.clip(t + r.choice([-1, 1], N) * 0.6 / (len(colours) - 1), 0, 1))
    if third is not None:
        hit = r.random(N) < odds
        k[hit] = third[r.integers(0, len(third), hit.sum())]
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.03, 0.2, N)], 1)
    return np.stack([ramp(colours, t), ramp(colours, j), k], 1), share


def orient(f, bias=0.0, angle=0.0):
    """Directions along the level lines of a field, swung toward `angle` where the field is flat (and
    by `bias` everywhere): the way a brush goes round a form, and straight across where there is none."""
    gy, gx = np.gradient(f)
    a = np.arctan2(gx, -gy)
    m = np.hypot(gx, gy)
    m = m / (np.percentile(m, 99) + 1e-9)
    c, s = m * np.cos(2 * a) + bias * np.cos(2 * angle), m * np.sin(2 * a) + bias * np.sin(2 * angle)
    return (0.5 * np.arctan2(s, c)).astype(np.float32)


def sheet(r):
    """The sheet as it lies on the board, cut by hand so that no side is quite straight, its corners worn
    round, the top left one folded under. -> how far inside its edge each pixel is (px, negative off it), and
    how far in from the fold"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    e = []
    for k in range(4):                                   # top, right, bottom, left
        a, b = SHEET[k], SHEET[(k + 1) % 4]
        t = (b - a) / np.hypot(*(b - a))
        e.append((xx - a[0]) * -t[1] + (yy - a[1]) * t[0])
    d = np.minimum.reduce(e)
    for (a, b), R in zip(((3, 0), (0, 1), (1, 2), (2, 3)), r.uniform(4, 12, 4)):
        q = np.hypot(np.maximum(R - e[a], 0), np.maximum(R - e[b], 0))
        d = np.where((e[a] < R) & (e[b] < R), R - q, d)
    c, v = SHEET[0], np.float32([0.7071, 0.7071])
    u = (xx - c[0]) * v[0] + (yy - c[1]) * v[1]
    fold = u - 40
    d = np.minimum(d, fold)
    return d + 1.2 * noise.field((H, W), 140, r) * noise.smoothstep(0, 8, fold) + 0.4 * noise.field((H, W), 2.5, r), fold


def tooth(r):
    """Wove paper with a little tooth: a fine grit, and short fibres in it. -> height 0..1"""
    t = 0.5 * noise.field((H, W), 1.2, r) + 0.3 * noise.field((H, W), 3.0, r) + 0.2 * noise.field((H, W), 11, r)
    fib = noise.fibers((H, W), r, count=H * W // 3000, length=(6, 28), curl=0.25, strength=(0.2, 0.6))
    return noise.smoothstep(-2.2, 2.4, t + 0.8 * ndimage.gaussian_filter(fib, 0.6)).astype(np.float32)


def cumulus(r):
    """The clouds as the painter thinks of them before he paints: heaps of vapour, each a round head
    swelling out of the one below it, a few big heads and many small ones crowding the tops where the sun
    is on them. -> lobes (N, 7): x, y, radius, how far forward, haze, floor, top of their cloud"""
    out = []
    for x, y, a, b, s, haze, floor in MASSES:
        top = min(m[1] - m[3] for m in MASSES if m[6] == floor)
        n = int(1.5 * a * b / (s * s)) + 2
        for t, u in zip(r.uniform(0, 2 * np.pi, n), 0.7 * np.sqrt(r.uniform(0, 1, n))):
            out.append((x + a * u * np.cos(t), y + b * u * np.sin(t), s * r.uniform(0.8, 1.25), r.uniform(0, 0.3) * s, haze, floor, top))
        for t in r.uniform(-0.97 * np.pi, -0.03 * np.pi, int(np.pi * (a + b) / (2.0 * s)) + 1):   # heads along its upper edge
            R = s * np.clip(np.exp(r.normal(-0.45, 0.35)), 0.3, 1.0)
            out.append((x + (a - 0.6 * R) * np.cos(t), y + (b - 0.6 * R) * np.sin(t), R, 0.35 * s + r.uniform(0, 0.3) * R, haze, floor, top))
    L = np.array(out, np.float32)
    small = []
    for x, y, R, z, haze, floor, top in L:
        t = np.arctan2(SUN[1], SUN[0]) + r.uniform(-1.2, 1.2)
        c = np.array([x, y]) + R * np.array([np.cos(t), np.sin(t)])
        if r.random() < 0.4 or c[1] > floor - 20 or (np.hypot(L[:, 0] - c[0], L[:, 1] - c[1]) < 0.97 * L[:, 2]).any():
            continue                                     # only where its rim is the edge of the cloud
        q = R * r.uniform(0.18, 0.32)
        rho = R - 0.45 * q
        small.append((x + rho * np.cos(t), y + rho * np.sin(t), q, z + np.sqrt(max(R * R - rho * rho, 0)) - 0.35 * q, haze, floor, top))
    return np.vstack([L, np.array(small, np.float32).reshape(-1, 7)])


def model(L, r):
    """The front surface of the clouds, head by head: which head is nearest the eye at each pixel and which
    way the surface faces there. Each head turns its own small dome to the sun, but less than the whole cloud
    turns its great one; heads nearer the sun cast their shadows on those beyond, and the deeper a place lies
    in its cloud the darker it is. -> inside (H,W), the head at each pixel, the surface's depth, its tone
    0 (shadow) .. 1 (full sun)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Z = np.full((H, W), -1e4, np.float32)
    who = np.full((H, W), -1, np.int32)
    NX, NY = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    wx, wy = (0.1 * noise.fbm((H, W), 70, r, octaves=3) for _ in range(2))   # no head is a true circle
    sag = 9 * noise.line1d(W, 90, r)
    for i, (x, y, R, z, haze, floor, top) in enumerate(L):
        y0, y1, x0, x1 = max(int(y - 1.2 * R), 0), min(int(y + 1.2 * R) + 1, H), max(int(x - 1.2 * R), 0), min(int(x + 1.2 * R) + 1, W)
        if y1 <= y0 or x1 <= x0:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        dx, dy = (xx[sl] - x) / R + wx[sl], (yy[sl] - y) / R + wy[sl]
        d2 = dx * dx + dy * dy
        zz = np.where((d2 < 1) & (yy[sl] < floor + sag[x0:x1]), z + R * np.sqrt(np.maximum(1 - d2, 0)), -1e4)
        up = zz > Z[sl]
        Z[sl][up], who[sl][up], NX[sl][up], NY[sl][up] = zz[up], i, dx[up], dy[up]
    inside = Z > -1e3
    depth = np.where(inside, np.maximum(Z, 0), 0).astype(np.float32)
    gy, gx = np.gradient(ndimage.gaussian_filter(depth, 50))
    nb = np.stack([-gx, -gy, np.full_like(gx, 0.5)], -1)
    n = 0.7 * nb / np.linalg.norm(nb, axis=-1, keepdims=True) \
        + 0.3 * np.stack([NX, NY, np.sqrt(np.clip(1 - NX * NX - NY * NY, 0, 1))], -1)
    lit = np.clip(n @ SUN / np.linalg.norm(n, axis=-1), 0, 1)
    # the shadows the heads cast on the cloud, and how much vapour the sunlight has come through to reach each
    # place: march toward the sun over the surface, at a quarter scale
    z4, in4 = Z[::4, ::4], inside[::4, ::4].astype(np.float32)
    h4, w4 = z4.shape
    y4, x4 = np.mgrid[0:h4, 0:w4].astype(np.float32)
    flat = np.hypot(SUN[0], SUN[1])
    block, through = np.full(z4.shape, -1e4, np.float32), np.zeros(z4.shape, np.float32)
    for t in np.arange(6, 700, 6.0):
        i4 = np.clip(np.rint(y4 + SUN[1] / flat * t / 4), 0, h4 - 1).astype(int), np.clip(np.rint(x4 + SUN[0] / flat * t / 4), 0, w4 - 1).astype(int)
        if t < 420:
            block = np.maximum(block, z4[i4] - z4 - t * SUN[2] / flat)
        through += 6 * in4[i4]
    shade = ndimage.zoom(noise.smoothstep(0, 40, block), 4, order=1)[:H, :W]
    sunlit = ndimage.zoom(np.exp(-through / 170), 4, order=1)[:H, :W]
    k = np.maximum(who, 0)
    low = noise.smoothstep(0, 1, (yy - L[k, 6]) / (L[k, 5] - L[k, 6])) ** 1.5 * (L[k, 4] == 0)
    sky = 0.5 - 0.5 * np.clip(NY, -1, 1)                 # in the shadow the tops of the heads still have the sky over them
    tone = 0.04 + 0.8 * noise.smoothstep(0.1, 0.9, 0.6 * sunlit + 0.4 * lit) * (1 - 0.6 * shade) * (1 - 0.35 * low) \
        + 0.28 * sky * (1 - 0.5 * low)
    return inside.astype(np.float32), who, depth, (tone * inside).astype(np.float32)


def heads(L, who, tone, r):
    """The heads the painter can see, and for each the strokes that model its sunlit top: arcs along its upper
    rim, the outermost the brightest, each one further in a little darker, reaching as far round as the sun
    does. On some heads the blue is cut back sharp against the rim; on others the rim is dragged out and lost.
    -> {kind: (paths, half-widths, tones)} for the lights, the knife, the cuts and the drags"""
    sun = np.arctan2(SUN[1], SUN[0])
    out = {k: ([], [], []) for k in ("light", "knife", "cut", "drag")}

    def look(x, y):
        return np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1)

    def put(kind, x, y, th, rho, span, w, lift, keep):
        """An arc round (x, y), kept to the longest run of it where `keep` holds."""
        t = th + np.linspace(-span, span, 12)
        px, py = x + rho * np.cos(t), y + rho * np.sin(t)
        ok = keep(look(px, py))
        best, run, end = 0, 0, 0
        for k, o in enumerate(ok):
            run = run + 1 if o else 0
            if run > best:
                best, end = run, k
        if best < 4:
            return
        P = np.stack([px[end - best + 1:end + 1], py[end - best + 1:end + 1]], 1)
        out[kind][0].append(P[::r.choice([-1, 1])])
        out[kind][1].append(w)
        out[kind][2].append(float(tone[look(P[:, 0], P[:, 1])].mean()) + lift)

    for i in np.argsort(L[:, 3] + L[:, 2]):              # the farthest first
        x, y, R, z, haze, floor, top = L[i]
        mine = lambda q, i=i: who[q] == i
        th = sun + r.normal(0, 0.3)
        sunny = tone[look(x + 0.8 * R * np.cos(th), y + 0.8 * R * np.sin(th))]
        n = (1 if R < 40 else int(r.integers(1, 3)) if R < 90 else int(r.integers(2, 4))) if sunny > 0.35 else int(r.random() < 0.75)
        if sunny <= 0.35:                                # in the shadow a head is turned only by the light of the sky above it
            th = -np.pi / 2 + r.normal(0, 0.4)
        w = R * r.uniform(0.17, 0.27)
        for j in range(n - 1, -1, -1):                   # from the inside out, the lightest last
            put("light", x, y, th + r.normal(0, 0.12), R - w * (1 + 1.3 * j), r.uniform(0.3, 0.65) * (1 - 0.12 * j),
                w * r.uniform(0.8, 1.2), 0.12 * (1 - j / max(n, 1)), mine)
        if R > 30 and sunny > 0.4 and r.random() < 0.55:   # found: the blue cut sharp against it
            b = r.uniform(6, 14)
            put("cut", x, y, th + r.normal(0, 0.3), R + 0.8 * b, r.uniform(0.4, 1.0), b, 0.0, lambda q: who[q] < 0)
        elif R > 30:                                      # lost: its rim dragged out into the blue
            put("drag", x, y, th + r.normal(0, 0.5), R * r.uniform(0.95, 1.05), r.uniform(0.3, 0.7), r.uniform(8, 18), 0.0,
                lambda q: np.ones(len(q[0]), bool))
        if R > 55 and haze == 0 and sunny > 0.7 and r.random() < 0.7:   # the knife on the bigger heads in the sun
            put("knife", x, y, th + r.normal(0, 0.2), R - 1.3 * w, r.uniform(0.25, 0.5), w * r.uniform(0.9, 1.3), 0.25, mine)
    return out


# paint with some body, scrubbed on: it hardly stands up from the paper, and drags into the wet paint near it
SCRUB = dict(thick=0.025, grooves=0.07, lips=0.02, land=0.2, lift=0.05, tails=0.5, pickup=0.4, merge=6.0, spent=0.7,
             ends=(0.3, 0.1), taper=0.3, fray=3.0, flatten=0.3)
# stiff paint from a loaded hog brush
LOADED = dict(thick=0.3, grooves=0.6, lips=0.15, land=0.4, lift=0.35, tails=0.9, pickup=0.25, merge=1.5, spent=0.75,
              ends=(0.35, 0.05), taper=0.45, fray=2.2, flatten=0.8)
# and from the flat of the knife: no bristles, sharp sides, a ridge where it was lifted
KNIFE = dict(thick=0.45, grooves=0.0, lips=0.35, land=0.3, lift=0.7, tails=0.0, pickup=0.3, merge=0.6, spent=0.8,
             ends=(0.15, 0.0), taper=0.05, fray=0.8, flatten=1.0)
# dabs of foliage from a round brush, short and ragged, hardly streaked
LEAF = dict(thick=0.06, grooves=0.08, lips=0.05, land=0.3, lift=0.1, tails=0.2, pickup=0.3, merge=2.0, spent=0.5,
            ends=(0.7, 0.5), taper=0.3, fray=2.5, flatten=0.5)
CLUMPS = [(200, 220, 70), (720, 160, 46), (1180, 260, 96), (1830, 280, 100), (2250, 80, 150), (2450, 140, 66)]   # x, half-width, height


def paint(seed=1822):
    r = noise.rng(seed)
    d, fold = sheet(r)
    on = noise.smoothstep(-0.7, 0.7, d)
    tt = tooth(r)
    Y = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    X = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    big = noise.field((H, W), 320, r)

    # the ground, red-brown, brushed across the sheet some days before and dry: thinner in streaks, and here
    # and there along the edges the brush missed and the paper shows
    streak = noise.stretched((W, H), 5, 260, r).T[:H, :W]
    reach = noise.smoothstep(2, 16, d + 9 * noise.field((H, W), 70, r) - 5)
    a = (reach * np.clip(0.92 - 0.06 * streak - 0.1 * (tt - 0.5), 0, 1))[..., None]
    rgb = lin(PAPER) * (0.95 + 0.08 * tt)[..., None] * (1 - a) + lin(GROUND) * (1 + 0.05 * big + 0.03 * streak)[..., None] * a
    rgb = np.ascontiguousarray(rgb, np.float32)
    height = (0.3 * tt).astype(np.float32)
    wet = np.zeros((H, W), np.float32)
    # the paint stops short of the edges, raggedly, where the sheet was held
    margin = noise.smoothstep(0, 20, d + 18 * noise.field((H, W), 90, r) + 6)

    def lay(paths, w, load, how, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **{**how, **kw})

    def strew(spacing, odds, squash=1.0):
        """Where the strokes of a passage lie: a shaken honeycomb kept by `odds` (H,W), its rows `squash`
        times closer than its points, for strokes laid more across than up."""
        P = ((dabs.scatter((int(squash * H) + 2 * spacing, W + 2 * spacing), spacing, r) - spacing) / [1, squash]).astype(np.float32)
        return P[r.random(len(P)) < (odds * margin)[at(P)]]

    def go(field, P, length, n=8, bend=0.0, tilt=0.0):
        """Strokes along a field of directions, each with its middle on its point, drawn from either end."""
        paths = impasto.follow(field, P, length, n, bend, tilt)
        paths -= (paths[:, -1:] - paths[:, :1]) / 2
        back = r.random(len(P)) < 0.5
        paths[back] = paths[back, ::-1]
        return paths

    def blue(P, lift=0.0):
        """The paint of the sky at each point: deep at the top, paler toward the heath, palest toward the sun."""
        y, x = at(P)
        return 0.08 + 0.74 * noise.smoothstep(80, HORIZON, y) ** 0.9 + 0.12 * noise.smoothstep(1500, 100, x) \
            * noise.smoothstep(1300, 200, y) + 0.05 * big[y, x] + lift

    def veiled(load, P, k=0.6):
        """Far clouds take on the blue of the air between."""
        y, x = at(P)
        h = (k * haze[y, x])[:, None, None]
        return load[0] + (ramp(SKY, blue(P, 0.18))[:, None] - load[0]) * h, load[1]

    L = cumulus(r)
    inside, who, depth, tone = model(L, r)
    k = np.maximum(who, 0)
    haze = np.where(who >= 0, L[k, 4], 0).astype(np.float32)
    size = np.where(who >= 0, L[k, 2], 60).astype(np.float32)     # how big the head is
    fl = L[k, 5]                                                   # and where the floor of its cloud is
    near = ndimage.gaussian_filter(inside, 12)
    mass = ndimage.gaussian_filter(inside, 40)
    wind = orient(mass, 1.5, -0.12)                    # level, rising a little to the right, and round the cloud near it
    broad = orient(ndimage.gaussian_filter(depth, 60), 0.6, -0.45)  # round the masses of the cloud, and slanting
    curl = orient(ndimage.gaussian_filter(depth, 30), 0.6, -0.45)   # round the heads

    # the blue scrubbed in with the broadest brush, round where the clouds will be, thin enough for the ground
    # to glow through it; then worked again wet into wet, lighter and darker, some strokes with more white
    P = strew(170, noise.smoothstep(0.7, 0.25, near) * (Y < HORIZON + 20), 2.0)
    n = len(P)
    lay(go(wind, P, r.uniform(300, 700, n), 8, r.normal(0, 0.0003, n), r.normal(0, 0.05, n)), r.uniform(34, 64, n),
        loads(SKY, blue(P), r, 0.05), SCRUB, dry=r.uniform(0.0, 0.15, n), hide=0.92)
    P = strew(140, noise.smoothstep(0.8, 0.3, near) * (Y < HORIZON + 20), 1.8)
    n = len(P)
    lay(go(wind, P, r.uniform(220, 520, n), 8, r.normal(0, 0.0005, n), r.normal(0, 0.05, n)), r.uniform(18, 40, n),
        loads(SKY, blue(P, r.normal(0.02, 0.05, n)), r, 0.05), SCRUB, dry=r.uniform(0.05, 0.3, n), hide=0.8, spent=0.85,
        taper=0.5, tails=0.8)
    P = strew(140, noise.smoothstep(HORIZON - 700, HORIZON - 100, Y) * noise.smoothstep(0.6, 0.2, near) * (Y < HORIZON + 10), 2.6)
    n = len(P)                                           # and low down, toward the heath, paler and warmer
    lay(go(wind, P, r.uniform(250, 600, n), 8, r.normal(0, 0.0003, n), r.normal(0, 0.03, n)), r.uniform(12, 30, n),
        loads(HAZE, r.uniform(0.35, 0.8, n), r, 0.04), SCRUB, dry=r.uniform(0.2, 0.5, n), hide=0.5, spent=0.85, taper=0.5)

    # the cloud laid in with its shadow colours, grey and violet, going round its masses
    P = strew(72, noise.smoothstep(0.2, 0.45, near), 1.2)
    n = len(P)
    y, x = at(P)
    lens, bend, flat = r.uniform(140, 360, n), r.normal(0, 0.0012, n), ((Y - fl)[y, x] > -260) & (haze[y, x] == 0)
    paths = go(broad, P, lens, 8, bend)
    paths[flat] = go(wind, P[flat], lens[flat], 8, 0.3 * bend[flat])     # and level toward its floor
    lay(paths, r.uniform(18, 40, n),
        veiled(loads(CLOUD, 0.1 + 0.55 * tone[y, x], r, 0.05, WARM, 0.12), P), SCRUB, dry=r.uniform(0.0, 0.1, n), hide=0.94,
        pickup=0.55, merge=5, spent=0.85, taper=0.5, tails=0.8, ends=(0.4, 0.1))
    # the half lights, broad strokes going round the masses, wet into the shadow
    P = strew(44, inside * noise.smoothstep(0.18, 0.4, tone), 1.1)
    n = len(P)
    y, x = at(P)
    lay(go(broad, P, r.uniform(100, 260, n), 7, r.normal(0, 0.0015, n)), np.clip(0.14 * size[y, x], 8, 24) * r.uniform(0.7, 1.3, n),
        veiled(loads(CLOUD, 0.25 + 0.55 * tone[y, x], r, 0.05, WARM, 0.08), P), SCRUB, thick=0.08, grooves=0.3, dry=r.uniform(0.0, 0.2, n),
        hide=0.9, merge=6, pickup=0.55, spent=0.85, taper=0.5, tails=0.8, ends=(0.4, 0.1))
    # the lights laid in broad, loaded strokes over the sunlit side, cream into white
    P = strew(30, inside * noise.smoothstep(0.45, 0.7, tone), 1.1)
    n = len(P)
    y, x = at(P)
    lay(go(curl, P, r.uniform(60, 180, n), 7, r.normal(0, 0.001, n)), r.uniform(12, 28, n),
        veiled(loads(np.vstack([CLOUD[7:], WHITE[:2]]), 1.4 * tone[y, x] - 0.45, r, 0.05), P, 0.4), LOADED,
        thick=0.18, dry=r.uniform(0.0, 0.25, n), pickup=0.4, merge=3)
    # each head modelled by the strokes on its sunlit top, stiff cream and white; the knife on the biggest;
    # the blue cut sharp against some rims, and others dragged out and lost
    ALL = np.vstack([CLOUD, WHITE])
    hd = heads(L, who, tone, r)
    for kind in ("light", "knife", "cut", "drag"):
        paths, w, t = hd[kind]
        n = len(paths)
        if not n:
            continue
        w, t = np.array(w, np.float32), np.clip(np.array(t, np.float32), 0, 1)
        mid = np.array([q[len(q) // 2] for q in paths], np.float32)
        if kind == "light":
            lay(paths, w, veiled(loads(ALL, t, r, 0.04), mid, 0.4), LOADED, thick=0.04 + 0.3 * t ** 2, dry=r.uniform(0.0, 0.3, n), pickup=0.35)
        elif kind == "knife":
            lay(paths, w, loads(WHITE, r.uniform(0.3, 1.0, n), r, 0.1), KNIFE, dry=r.uniform(0.0, 0.2, n))
        elif kind == "cut":
            lay(paths, w, loads(SKY, blue(mid), r, 0.03), SCRUB, thick=0.06, grooves=0.3, merge=1.0, dry=r.uniform(0.0, 0.2, n), hide=0.95)
        else:
            lay(paths, w, veiled(loads(ALL, 0.9 * t, r, 0.05), mid), SCRUB, dry=r.uniform(0.45, 0.65, n), hide=0.7, merge=6)
    # the underside in level strokes along the floor of the cloud, broken and dry
    P = strew(40, inside * (haze == 0) * noise.smoothstep(-170, -50, Y - fl), 2.0)
    n = len(P)
    lay(go(wind, P, r.uniform(150, 420, n), 6, r.normal(0, 0.0006, n), r.normal(0, 0.03, n)), r.uniform(12, 26, n),
        veiled(loads(CLOUD, r.uniform(0.04, 0.25, n), r, 0.04, WARM, 0.1), P), SCRUB, dry=r.uniform(0.1, 0.35, n), hide=0.85, merge=6)
    # and over it, here and there, the warm light thrown up from the sunlit country, dragged on dry
    P = strew(90, inside * (haze == 0) * noise.smoothstep(-320, -170, Y - fl) * noise.smoothstep(-60, -150, Y - fl), 2.0)
    n = len(P)
    lay(go(wind, P, r.uniform(80, 220, n), 6, r.normal(0, 0.0006, n), r.normal(0, 0.04, n)), r.uniform(10, 20, n),
        loads(WARM, r.uniform(0.2, 0.9, n), r, 0.05), SCRUB, dry=r.uniform(0.35, 0.55, n), hide=0.38, merge=6)

    # wisps high on the left, pale paint dragged thin and level with a dry brush
    P = np.stack([r.uniform(200, 1200, 7), r.uniform(200, 620, 7)], 1).astype(np.float32)
    lay(go(wind, P, r.uniform(300, 650, 7), 8, r.normal(0, 0.0004, 7), r.normal(-0.06, 0.04, 7)), r.uniform(18, 36, 7),
        loads(np.vstack([SKY[6:], CLOUD[9:]]), r.uniform(0.3, 1.0, 7), r), SCRUB, dry=r.uniform(0.5, 0.65, 7), hide=0.5, spent=0.9, taper=0.5)

    heath(rgb, height, lay, strew, go, X, Y, r)
    return support(rgb, height, d, on, fold, r)


def heath(rgb, height, lay, strew, go, X, Y, r):
    """The foot of the study: the far country in a few level strokes with a glint of sun on it, and over it
    the trees of the Heath, clump by clump, in dark dabs with the warm ground glowing between them."""
    xs = np.arange(W, dtype=np.float32)
    crowns = np.full(W, HORIZON + 34.0, np.float32)
    for cx, half, h in CLUMPS:                           # each clump a few crowns, lower toward its ends
        for _ in range(int(half / 40) + 2):
            x0 = cx + r.uniform(-0.85, 0.85) * half
            R = 8 + h * r.uniform(0.2, 0.45)
            y0 = HORIZON + 34 - h * np.sqrt(max(1 - ((x0 - cx) / half) ** 2, 0)) + 0.7 * R
            crowns = np.minimum(crowns, np.where(np.abs(xs - x0) < R, y0 - np.sqrt(np.clip(R * R - (xs - x0) ** 2, 0, None)), np.inf))
    crowns += 4 * noise.line1d(W, 14, r) + 2 * noise.line1d(W, 5, r)
    flat = np.zeros((1, 1), np.float32)
    P = strew(30, noise.smoothstep(HORIZON - 16, HORIZON - 6, Y) * noise.smoothstep(HORIZON + 40, HORIZON + 25, Y), 3.0)
    n = len(P)
    lay(go(flat, P, r.uniform(80, 260, n), 5, 0, r.normal(0, 0.02, n)), r.uniform(4, 9, n), loads(FAR, r.uniform(0.2, 0.9, n), r),
        SCRUB, dry=r.uniform(0.1, 0.4, n), hide=0.85)
    P = strew(22, noise.smoothstep(HORIZON + 4, HORIZON + 10, Y) * noise.smoothstep(HORIZON + 26, HORIZON + 18, Y)
              * noise.smoothstep(380, 440, X) * noise.smoothstep(640, 580, X), 3.0)
    n = len(P)
    lay(go(flat, P, r.uniform(30, 120, n), 4, 0, r.normal(0, 0.02, n)), r.uniform(2.5, 6, n), loads(GLINT, r.uniform(0.2, 1.0, n), r),
        LOADED, thick=0.1, dry=r.uniform(0.2, 0.5, n))
    top = crowns[None, :]
    P = strew(15, noise.smoothstep(top + 4, top + 16, Y))     # the masses, in broad dabs every way
    n = len(P)
    lay(go(flat, P, r.uniform(10, 28, n), 4, r.normal(0, 0.02, n), r.uniform(-1.4, 1.4, n)), r.uniform(8, 16, n),
        loads(TREES, r.uniform(0.0, 0.5, n), r, 0.08, BROWN, 0.25), LEAF, dry=r.uniform(0.05, 0.35, n), hide=0.92)
    x = r.uniform(80, 2540, 700)                         # the crowns broken against the sky in small touches
    P = np.stack([x, crowns[x.astype(int)] + r.uniform(-2, 16, 700)], 1).astype(np.float32)
    lay(go(flat, P, r.uniform(4, 12, 700), 4, 0, r.uniform(-1.5, 1.5, 700)), r.uniform(4, 9, 700),
        loads(TREES, r.uniform(0.0, 0.6, 700), r, 0.06, BROWN, 0.1), LEAF, dry=r.uniform(0.0, 0.3, 700), hide=0.95)
    P = strew(30, noise.smoothstep(HORIZON + 60, HORIZON + 90, Y), 2.0)   # the heath at the foot, dark and level
    n = len(P)
    lay(go(flat, P, r.uniform(60, 200, n), 5, 0, r.normal(0, 0.04, n)), r.uniform(8, 18, n), loads(TREES, r.uniform(0.0, 0.4, n), r, 0.08, BROWN, 0.3),
        SCRUB, dry=r.uniform(0.05, 0.3, n), hide=0.9)
    lit = []                                             # and the sun on the left of each crown
    for cx, half, h in CLUMPS:
        x = cx - half * r.uniform(0.2, 0.9, 12)
        lit.append(np.stack([x, crowns[np.clip(x, 0, W - 1).astype(int)] + r.uniform(5, 30, 12)], 1))
    P = np.vstack(lit).astype(np.float32)
    n = len(P)
    lay(go(flat, P, r.uniform(5, 14, n), 4, 0, r.uniform(-1.3, 1.3, n)), r.uniform(3, 6, n), loads(TREES, r.uniform(0.55, 0.9, n), r),
        LEAF, thick=0.12, dry=r.uniform(0.2, 0.5, n))


def support(rgb, height, d, on, fold, r):
    """The sheet laid down on its board: brown millboard round it, a little glue squeezed out at the edge and
    darkened, the sheet's shadow; pins in two corners, the holes of the two that came out, and at one of those
    the corner lifted; the top left corner folded under, a little thicker along the fold."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fib = ndimage.gaussian_filter(noise.fibers((H, W), r, count=H * W // 4000, length=(10, 50), curl=0.15, strength=(0.2, 0.7)), 0.7)
    board = lin(BOARD) * (1 + 0.07 * noise.fbm((H, W), 50, r, octaves=4) + 0.06 * fib)[..., None]
    glue = noise.smoothstep(-9 - 4 * noise.field((H, W), 30, r), -1, d) * (1 - on)
    board *= (1 - 0.2 * glue)[..., None] * np.float32([1.0, 0.97, 0.88])
    c = SHEET[2]                                          # the bottom right corner, where its pin came out, lifted
    u = (xx - c[0]) * -0.7071 + (yy - c[1]) * -0.7071
    flap = noise.smoothstep(70, 60, u) * on
    cast = ndimage.gaussian_filter(np.roll(on, (5, 4), (0, 1)), 3.5) + 0.6 * ndimage.gaussian_filter(np.roll(flap, (13, 11), (0, 1)), 6)
    board *= (1 - 0.5 * np.clip(cast, 0, 1) * (1 - on))[..., None]
    out = rgb * on[..., None] + board * (1 - on[..., None])
    h = height * on + 1.6 * on + 0.15 * fib * (1 - on) + 0.12 * np.maximum(70 - u, 0) * flap \
        + 0.9 * noise.smoothstep(26, 2, fold) * on
    crease = np.exp(-((u - 66) / 1.6) ** 2) * on
    out *= (1 - 0.35 * crease)[..., None]
    for x, y in ((2506, 118), (110, 1966)):               # the pins, their heads a little domed, each with its shadow
        q = np.hypot(xx - x - 5, yy - y - 4)
        out *= (1 - 0.45 * np.exp(-(q / 6) ** 2))[..., None]
        q = np.hypot(xx - x, yy - y)
        a = noise.smoothstep(7.5, 6.0, q)
        out = out * (1 - a[..., None]) + lin("#8b867d") * (0.8 + 0.25 * noise.field((H, W), 3, r))[..., None] * a[..., None]
        h = np.maximum(h, h * (1 - a) + (2 + 5 * np.sqrt(np.clip(1 - (q / 7) ** 2, 0, 1))) * a)
    for x, y in ((116, 110), (2500, 1978)):               # and the holes
        q = np.hypot(xx - x, yy - y)
        out *= (1 - 0.8 * noise.smoothstep(3.0, 1.8, q) + 0.08 * np.exp(-((q - 4.5) / 1.5) ** 2))[..., None]
        h += 0.8 * np.exp(-((q - 4) / 1.5) ** 2) - 1.0 * noise.smoothstep(3.0, 1.8, q)
    img = dabs.shine(out, ndimage.gaussian_filter(h, 0.6), light=LIGHT, relief=0.6, gloss=0, reach=(0.85, 1.12))
    return img + 0.04 * impasto.glints(h, LIGHT)[..., None]
