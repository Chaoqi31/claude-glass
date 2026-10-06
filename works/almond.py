"""Almond Branches against the Sky. Oil on canvas, in thick strokes, the flowers in loaded touches of white.

In February 1890, in the asylum at Saint-Rémy, Van Gogh heard that his brother Theo had a
son, named Vincent after him, and painted for the child's room branches of almond in flower,
the first tree to flower in Provence, against the sky. He painted them as he had learned
from the Japanese prints he collected: seen from below, with no ground and no horizon, the
branches drawn round with a dark line and the flowers set flat on the blue. The sky goes on
in short, thick strokes laid side by side in patches, each patch its own way, and near a
branch they turn and run along it. The boughs are dragged on with a loaded brush in umber,
olive and Prussian blue; the twigs, which grow in short shoots turning at every node, go on
in a few strokes that turn there too. Each flower is a few touches of white, pressed at the
rim of a petal and drawn in toward the heart, with a dab of yellow-green or pink there and a
speck or two for the stamens. His blue has faded a little since, and his whites have yellowed.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Almond Branches against the Sky"
DATE = "2026"
MEDIUM = "Oil on canvas, in thick strokes, the flowers in loaded touches of white"
AFTER = "Vincent van Gogh, Almond Blossom, Saint-Rémy, 1890"
ROOM = "The Garden"
YEAR = 1890
PLACE = "Saint-Rémy-de-Provence"
REGION = "Europe"
NOTE = ("Almond branches seen from below, drawn round in dark paint and set flat against the sky, "
        "as in the Japanese prints he collected. Each flower is a few touches of white from a loaded brush.")

H, W = 2380, 3000
LIGHT = (-0.6, -0.5, 0.62)
SUN = np.array([-0.77, -0.64])          # where the light comes from, across the canvas
ROOT = np.array([520.0, 2900.0])        # where the trunk stands, below the canvas


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
SKY = pal("#5d94a0", "#699ea9", "#74a7b1", "#7fb0b8", "#8ab8be", "#96c0c3", "#a3c8c8")  # cobalt, viridian, white
BLUE = pal("#5f8ba3", "#6a95ab", "#769fb3", "#82a9ba", "#8eb3c0", "#9bbdc6")             # where the cobalt leads
COBALT = pal("#5682a5", "#6590b0")
VIRIDIAN = pal("#72a9a2", "#86b7ad")
PALE = pal("#b2cfcb", "#c0d6d0")
BARK = pal("#252a2e", "#38332a", "#4b4231", "#5c5536", "#6b693e", "#7c7a4c", "#908d66", "#a5a283")  # umber to olive
PRUSSIAN = pal("#131e2e", "#1a293c", "#23354a")
OCHRE = pal("#8f6f3c", "#a8864b", "#b99a5c")
WHITE = pal("#d2d2bc", "#dddac4", "#e6e1cc", "#ece6d3", "#f1ebdb")                        # lead white, yellowed
PINK = pal("#d4aca3", "#dfbdb3", "#e9cfc6", "#f1ded6")                                    # a red lake, let down and faded
HEART = pal("#8b974f", "#a0aa5f", "#b5ba71", "#c6c688", "#d6d4a3")                        # yellow-green
RED = pal("#7f3b31", "#9a5145", "#6a4030")
LEAF = pal("#4c6b3a", "#5d7d41", "#71904b", "#89a35b", "#a1b56f")

# the boughs as drawn, from the trunk out: rows of (x, y, half-width)
BOUGHS = [
    [(690, 2470, 112), (725, 2280, 101), (665, 2080, 94), (735, 1900, 87), (890, 1775, 79), (1065, 1655, 73),
     (1185, 1505, 65)],
    [(1185, 1505, 57), (1400, 1392, 49), (1640, 1402, 43), (1880, 1272, 36), (2160, 1212, 31), (2420, 1062, 26),
     (2700, 1012, 21), (3060, 880, 17)],
    [(1185, 1505, 52), (1172, 1270, 44), (1322, 1080, 38), (1282, 860, 32), (1410, 640, 26), (1392, 420, 21),
     (1500, 200, 16), (1520, -60, 12)],
    [(735, 1890, 52), (560, 1722, 44), (500, 1500, 38), (330, 1362, 31), (242, 1122, 25), (90, 962, 19), (-60, 880, 15)],
    [(890, 1775, 44), (1150, 1822, 39), (1450, 1762, 34), (1750, 1862, 29), (2050, 1802, 25), (2350, 1880, 21),
     (2650, 1800, 17), (3060, 1850, 13)],
]


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


def mixed(families, which, tone, r, accent=None, odds=0.2):
    """Loads for strokes drawn from several palettes, `which` saying which one each takes."""
    cols, share = np.empty((len(which), 3, 3), np.float32), np.empty((len(which), 3))
    for i, colours in enumerate(families):
        sel = which == i
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds)
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


def straight(P, step=3.0):
    """A polyline resampled every `step` px, its corners kept."""
    P = np.asarray(P, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))])
    si = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(si, s, P[:, j]) for j in range(P.shape[1])], 1)


def frame(A):
    """Unit tangents along a line, and normals turned toward the light."""
    d = np.gradient(A[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    return d, n * np.sign((n @ SUN).mean() + 1e-9)


def gnarl(A, r):
    """Old wood is never smooth: the line of a bough wanders off its course and back, and the
    wood swells at its knots and pinches between them."""
    _, n = frame(A)
    A = A.copy()
    A[:, :2] += n * (0.09 * A[:, 2] * noise.line1d(len(A), 80, r))[:, None]
    A[:, 2] *= 1 + 0.13 * noise.line1d(len(A), 35, r)
    return A


def grow(r):
    """The tree. The boughs are drawn by hand; off them grow twigs, and off the twigs sprigs,
    each a run of short straight shoots that turns at every node, one way and then the other,
    as almond wood grows, reaching away from the trunk into whatever sky is still free.
    -> limbs [(rows (n, 3) of x, y, half-width; order 0 bough to 3 sprig)], nodes [(x, y, direction, order, tip)]"""
    s = 8
    room = Image.new("L", (W // s + 2, H // s + 2), 0)
    pen = ImageDraw.Draw(room)

    def claim(A, pad):
        for (x0, y0, w0), (x1, y1, w1) in zip(A[:-4:4], A[4::4]):
            pen.line([(x0 / s, y0 / s), (x1 / s, y1 / s)], fill=255, width=max(1, int(2 * (max(w0, w1) + pad) / s)))

    def space():
        return ndimage.distance_transform_edt(np.asarray(room) == 0) * s

    def free(free_, e):
        if not (0 <= e[0] < W and 0 <= e[1] < H):
            return 300.0
        return free_[int(e[1] / s), int(e[0] / s)]

    def shoot(p, a, w0, L, order, free_):
        pts, k, run, mean = [np.array([p[0], p[1], w0])], 0, 0.0, a
        floor = (0, 6.0, 4.4, 3.2)[order]
        while run < L:
            seg = r.uniform(45, 110) * (0.65 if order == 3 else 1)
            amp = (-1) ** k * r.uniform(0.05, 0.24)
            best, score = 0.0, -1e9
            for dl in (-0.45, -0.22, 0.0, 0.22, 0.45):
                b = mean + dl + amp
                sc = free(free_, pts[-1][:2] + seg * np.array([np.cos(b), np.sin(b)])) - 70 * abs(dl) + r.uniform(0, 25)
                if sc > score:
                    best, score = dl, sc
            mean += 0.5 * best
            b = mean + amp
            e = pts[-1][:2] + seg * np.array([np.cos(b), np.sin(b)])
            if k > 0 and free(free_, e) < 14 + pts[-1][2]:
                break
            run += seg
            pts.append(np.array([e[0], e[1], max(floor, w0 * (1 - 0.6 * run / L))]))
            k += 1
            if not (-120 < e[0] < W + 120 and -120 < e[1] < H + 120):
                break
        return np.array(pts), run

    limbs = [(gnarl(brush.path(np.asarray(b, float), 3.0), r), 0) for b in BOUGHS]
    for A, _ in limbs:
        claim(A, 30)
    free_ = space()
    nodes, i = [], 0
    while i < len(limbs):
        A, order = limbs[i]
        i += 1
        if order == 3:
            continue
        S = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(A[:, :2], axis=0).T))])
        gap = (150, 120, 95)[order]
        side, pos = r.choice([-1, 1]), r.uniform(0.3, 1.0) * gap
        while pos < S[-1] - 30:
            j = np.searchsorted(S, pos)
            x, y, w = A[j]
            pos += gap * r.uniform(0.7, 1.4)
            side = -side
            if not (-60 < x < W + 60 and -60 < y < H + 60) or r.random() > (0.9, 0.75, 0.6)[order]:
                continue
            d = A[min(j + 3, len(A) - 1), :2] - A[max(j - 3, 0), :2]
            t = np.arctan2(d[1], d[0])
            a = t + side * r.uniform(0.5, 1.0)
            away = np.arctan2(y - ROOT[1], x - ROOT[0])
            if np.cos(a - away) < -0.1:
                a = t - side * r.uniform(0.5, 1.0)
            L = r.uniform(*((300, 750), (150, 380), (60, 160))[order])
            w0 = min(w * (0.42, 0.55, 0.65)[order], (18, 11, 6.5)[order])
            pts, run = shoot((x, y), a, w0, L, order + 1, free_)
            if run < (140, 90, 50)[order]:
                continue
            twig = straight(pts)
            limbs.append((twig, order + 1))
            claim(twig, (18, 14, 10)[order])
            free_ = space()
            for k, (x_, y_, _) in enumerate(pts[1:], 1):
                e = pts[min(k + 1, len(pts) - 1), :2] - pts[k - 1, :2]
                nodes.append((x_, y_, np.arctan2(e[1], e[0]), order + 1, k == len(pts) - 1))
    return limbs, nodes


def sky_field(limbs, r):
    """Which way the sky's strokes run: in patches, each laid its own way, and near every branch
    along it, within a reach that grows with its girth. -> (directions, px from the nearest branch's edge)"""
    s = 4
    h, w = H // s + 1, W // s + 1
    ang, girth, spine = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), bool)
    for A, _ in sorted(limbs, key=lambda l: -l[0][0, 2]):
        d = np.gradient(A[:, :2], axis=0)
        i, j = (A[:, 1] / s).astype(int), (A[:, 0] / s).astype(int)
        ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        spine[i[ok], j[ok]] = True
        ang[i[ok], j[ok]] = np.arctan2(d[ok, 1], d[ok, 0])
        girth[i[ok], j[ok]] = A[ok, 2]
    dist, (iy, ix) = ndimage.distance_transform_edt(~spine, return_indices=True)
    edge = dist * s - girth[iy, ix]
    reach = 25 + 2.4 * girth[iy, ix]
    seeds = dabs.scatter((h, w), 200 / s, r)
    lab = np.zeros((h, w), np.int32) - 1
    q = np.clip(seeds[:, ::-1].astype(int), 0, [h - 1, w - 1])
    lab[q[:, 0], q[:, 1]] = np.arange(len(seeds))
    _, (cy, cx) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    far = 0.6 + 0.7 * noise.field((h, w), 800 / s, r) + r.uniform(-0.5, 0.5, len(seeds))[lab[cy, cx]]
    far = 0.5 * np.arctan2(ndimage.gaussian_filter(np.sin(2 * far), 2), ndimage.gaussian_filter(np.cos(2 * far), 2))
    field = blend(far, ang[iy, ix], noise.smoothstep(reach, 0.3 * reach, edge))
    up = lambda a: ndimage.zoom(a, s, order=0)[:H, :W]
    return up(field.astype(np.float32)), up(edge.astype(np.float32))


def sheaves(field, seeds, length, width, k, r, n=8, bend=0.0):
    """Strokes laid side by side in little sheaves: from each seed, k strokes about a brush-width
    apart following the same field, their starts staggered. -> paths (N*k, n, 2), half-widths, seed of each"""
    N = len(seeds)
    a = field[at(seeds)]
    d = np.stack([np.cos(a), np.sin(a)], -1)[:, None, :]
    across = np.stack([-np.sin(a), np.cos(a)], -1)[:, None, :]
    j = np.arange(k) - (k - 1) / 2
    L = np.asarray(length)[:, None] * np.exp(r.normal(0, 0.25, (N, k)))
    w = np.asarray(width)[:, None] * np.exp(r.normal(0, 0.15, (N, k)))
    P = seeds[:, None, :] + across * (1.75 * w * (j + r.normal(0, 0.12, (N, k))))[..., None] \
        + d * (L * r.uniform(-0.2, 0.2, (N, k)))[..., None]
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None].repeat(k, 1)
    return impasto.follow(field, P.reshape(-1, 2), L.ravel(), n, turn.ravel()), w.ravel(), np.repeat(np.arange(N), k)


def bough(A, r):
    """A bough in the strokes of a loaded brush dragged along it: ribbons of umber and olive
    twisting round the wood, dark lines of bark in Prussian blue, lighter paint dragged dry over
    the side toward the light, and its edges. -> [(path, half-width, tone, kind)] with kind 0 body,
    1 bark line, 2 dry light; the shaded edge and the lit one"""
    _, n = frame(A)
    t = np.linspace(0, 1, len(A))
    S = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(A[:, :2], axis=0).T))])
    out = []

    def run(f, half, tone, kind, length, overlap, keep=1.0):
        line = A[:, :2] + n * (A[:, 2] * f)[:, None]
        for m in pieces(line, length, overlap, r, keep):
            out.append((line[m], half[m].mean() * r.uniform(0.85, 1.2), tone + 0.12 * f[m].mean(), kind))

    k = int(np.clip(A[:, 2].max() / 14 + 2, 3, 7))
    for j in range(k):
        c = (2 * j + 1) / k - 1
        f = np.clip(0.8 * c + 0.22 * np.sin(2 * np.pi * r.uniform(0.6, 1.4) * t + r.uniform(0, 2 * np.pi)), -0.92, 0.92)
        run(f, A[:, 2] * 1.35 / k, 0.52, 0, 190, 50)
    for _ in range(int(S[-1] * A[:, 2].mean() / 2600)):
        f = np.clip(r.uniform(-0.85, 0.85) + 0.15 * np.sin(2 * np.pi * t * r.uniform(1, 3) + r.uniform(0, 6)), -0.9, 0.9)
        run(f, np.clip(0.06 * A[:, 2], 2.2, 5.5), 0.1, 1, r.uniform(80, 240), -20, keep=0.5)
    for _ in range(int(S[-1] * A[:, 2].mean() / 5000)):
        f = np.clip(r.uniform(0.15, 0.8) + 0.1 * np.sin(2 * np.pi * t * 2 + r.uniform(0, 6)), -0.9, 0.9)
        run(f, 0.25 * A[:, 2], 0.8, 2, r.uniform(90, 220), -30, keep=0.6)
    return out, A[:, :2] - n * A[:, 2:3], A[:, :2] + n * A[:, 2:3]


def twig(A, r):
    """A twig in a few strokes that run on through the nodes, turning there, and a dark line
    along its shaded side, here and there. -> [(path, half-width)] for the wood, and for the line"""
    _, n = frame(A)
    wood = [(A[m, :2], A[m, 2].mean() * r.uniform(0.95, 1.15)) for m in pieces(A[:, :2], 210, 18, r)]
    edge = A[:, :2] - n * A[:, 2:3] * 0.8
    shade = [(edge[m], max(1.5, 0.26 * A[m, 2].mean())) for m in pieces(edge, 170, 10, r, 0.75) if A[m, 2].mean() > 3.5]
    return wood, shade


def flower(c, R, toward, r):
    """A flower in a few touches of the loaded brush, each pressed at the rim of a petal and drawn
    in toward the heart. Most face the sky, turned and tilted so that some petals foreshorten;
    some are seen from the side, a cup of petals fanning up from the calyx; some are half open.
    -> petal touches [(path, half-width, tone)], heart path or None, stamens [paths], calyx path or None"""
    kind = r.choice(3, p=[0.6, 0.25, 0.15])
    rho = np.array([1.0, 0.75, 0.5, 0.25])
    touches = []
    if kind == 1:
        base = c - 0.35 * R * np.array([np.cos(toward), np.sin(toward)])
        m = r.choice([3, 4])
        for a in toward + np.linspace(-0.75, 0.75, m) + r.normal(0, 0.12, m):
            l = R * r.uniform(0.8, 1.05)
            tip = base + l * np.array([np.cos(a), np.sin(a)])
            touches.append((np.stack([tip, base + 0.6 * (tip - base), base + 0.25 * (tip - base)]), R * r.uniform(0.28, 0.36),
                            0.62 + 0.28 * np.cos(a - np.arctan2(*SUN[::-1]))))
        calyx = np.stack([base + 0.1 * R * np.array([np.cos(toward), np.sin(toward)]),
                          base - 0.35 * R * np.array([np.cos(toward), np.sin(toward)])])
        return touches[::-1] if r.random() < 0.5 else touches, None, [], calyx
    m = r.choice([4, 5, 5, 5, 6]) if kind == 0 else 5
    th = r.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(m) / m + r.normal(0, 0.22, m)
    if kind == 2:
        th = th[r.permutation(m)[:3]]
    tilt, q = r.uniform(0, np.pi), r.uniform(0.42, 1.0)

    def place(u, v):
        v = v * q
        return np.stack([c[0] + u * np.cos(tilt) - v * np.sin(tilt), c[1] + u * np.sin(tilt) + v * np.cos(tilt)], -1)

    for t in th[r.permutation(len(th))]:
        l, curl = R * r.uniform(0.72, 1.12), r.normal(0, 0.18)
        a = t + curl * (1 - rho)
        tip = place(np.cos(t), np.sin(t)) - c
        lit = 0.62 + 0.28 * (tip @ SUN) / (np.hypot(*tip) + 1e-9)       # petals turned to the light are paler
        if R > 45:                                # a broad petal in two touches, side by side
            for side in (-1, 1):
                b = a + side * r.uniform(0.14, 0.22)
                touches.append((place(l * rho * np.cos(b), l * rho * np.sin(b)), R * r.uniform(0.26, 0.32), lit))
        else:
            touches.append((place(l * rho * np.cos(a), l * rho * np.sin(a)), R * r.uniform(0.4, 0.48), lit))
    heart = place(np.array([-0.12, 0.12]) * R, np.array([0.04, -0.04]) * R)
    stamens = []
    for t in r.uniform(0, 2 * np.pi, r.integers(1, 4)):
        rr = R * r.uniform(0.14, 0.26)
        stamens.append(place((rr + np.array([0, 4.0])) * np.cos(t), (rr + np.array([0, 4.0])) * np.sin(t)))
    return touches, heart, stamens, None


def leaf(p, a, r):
    """A young leaf, folded and pointed: two touches drawn out from the node, one each side of
    the fold, meeting at the tip. -> two paths, half-width"""
    q = np.linspace(0, 1, 7)
    b = a + r.normal(0, 0.25) * q
    spine = p + r.uniform(55, 95) * q[:, None] * np.stack([np.cos(b), np.sin(b)], 1)
    w = r.uniform(5, 7.5)
    bulge = (1.1 * w * np.sin(np.pi * q ** 0.8))[:, None] * np.stack([-np.sin(b), np.cos(b)], 1)
    return [spine + bulge, spine - bulge], w


def paint(seed=1890):
    r = noise.rng(seed)
    limbs, nodes = grow(r)
    flow, edge = sky_field(limbs, r)
    ground = canvas.duck((H, W), seed, tint="#e4d9bf", thread=3.3)
    noisy = lambda scale, P: noise.field((H, W), scale, r)[at(P)]

    # the lay-in: the whole canvas rubbed over thin in the blue, the threads showing through
    hue = noise.field((H, W), 700, r)
    alpha = np.clip(0.94 + 0.06 * (0.5 - ground.tooth), 0, 1)[..., None]
    rgb = (ground.color * (1 - alpha) + ramp(SKY, 0.4 + 0.12 * hue) * alpha).astype(np.float32)
    height = ground.tooth * 0.3

    def lay(paths, w, load, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], **kw)

    # the sky: thick strokes side by side in pairs, in patches, longer where they run along a branch
    sweep = noise.field((H, W), 400, r)
    L, w_, k = 125.0, 21.0, 2
    sp = float(np.sqrt(k * L * 2 * w_ / (3.0 * 0.866)))
    P = dabs.scatter((H + 2 * int(sp), W + 2 * int(sp)), sp, r) - sp
    P = P[edge[at(P)] > -6]
    n = len(P)
    near = noise.smoothstep(120, 0, edge[at(P)])
    paths, ws, s = sheaves(flow, P, r.uniform(80, 170, n) * np.exp(0.25 * sweep[at(P)]) * (1 + 0.4 * near),
                           r.uniform(17, 26, n) * (1 - 0.35 * near), k, r, 8, r.normal(0, 0.002, n))
    t = (0.5 + 0.12 * hue[at(P)] + 0.06 * noisy(160, P) + 0.05 * (P[:, 1] / H - P[:, 0] / W) + r.normal(0, 0.06, n))[s]
    cobalt = (r.random(n) < 0.5 * noise.smoothstep(0.2, 1.2, noisy(900, P)))[s].astype(int)
    lay(either(paths, r), ws, mixed((SKY, BLUE), cobalt, t, r, np.concatenate([COBALT, VIRIDIAN, PALE]), 0.18),
        thick=0.24, spent=0.25, pickup=0.6, merge=4, fray=1.0, grooves=0.35)
    height = ndimage.gaussian_filter(height, 0.8)      # the sky's paint settles a little before the branches go on

    # the boughs, dragged on with a loaded brush and drawn round in Prussian blue, a stroke at a time
    edges = []
    for A, order in limbs:
        if order:
            continue
        strokes, shade, lit = bough(A, r)
        for kind, colours, accent, odds, kw in (
                (0, BARK, np.concatenate([PRUSSIAN[1:], OCHRE[:2], VIRIDIAN[:1]]), 0.25, dict(thick=0.3, pickup=0.25)),
                (1, np.concatenate([PRUSSIAN, BARK[:2]]), None, 0, dict(thick=0.25, pickup=0.3, spent=0.7)),
                (2, BARK[4:], np.concatenate([OCHRE[1:], VIRIDIAN, SKY[3:4]]), 0.35,
                 dict(thick=0.2, pickup=0.1, dry=r.uniform(0.45, 0.75, sum(s[3] == 2 for s in strokes))))):
            mine = [s for s in strokes if s[3] == kind]
            tone = np.array([s[2] for s in mine]) + r.normal(0, 0.08, len(mine))
            lay([s[0] for s in mine], np.array([s[1] for s in mine]), loads(colours, tone, r, accent, odds), **kw)
        w0 = A[:, 2]
        edges += [(shade, np.clip(0.13 * w0 + 2.5, 3, 12), 0.92), (lit, np.clip(0.06 * w0 + 1.8, 2, 5), 0.5)]
    cut = [(line[m], wd[m].mean()) for line, wd, keep in edges for m in pieces(line, 150, 25, r, keep)]
    nc = len(cut)
    lay([c for c, _ in cut], np.array([wd for _, wd in cut]) * np.exp(r.normal(0, 0.3, nc)),
        loads(PRUSSIAN, np.full(nc, 0.5), r, BARK[:2], 0.3, spread=0.3), thick=0.3, spent=0.55, lips=0.1, pickup=0.3)

    # the twigs, order by order, in strokes of umber and olive, warmer in the young wood, dark on the shaded side
    for order in (1, 2, 3):
        wood, shade = [], []
        for A, o in limbs:
            if o == order:
                a_, b_ = twig(A, r)
                wood += a_
                shade += b_
        nw = len(wood)
        lay([p for p, _ in wood], np.array([w for _, w in wood]),
            loads(BARK, r.uniform(0.25, 0.5, nw), r, np.concatenate([OCHRE[:2], PRUSSIAN[2:]]), 0.2 + 0.05 * order),
            thick=0.3, spent=0.5, pickup=0.25, land=0.3, taper=0.1)
        ns = len(shade)
        lay([p for p, _ in shade], np.array([w for _, w in shade]), loads(PRUSSIAN, np.full(ns, 0.5), r, BARK[:2], 0.3),
            thick=0.22, spent=0.6, pickup=0.3)

    # the blossom, in clusters along the twigs, thick where the flowers crowd; first a few young
    # leaves, and the specks of bud at the bare nodes
    bloom = noise.smoothstep(-0.5, 1.0, noise.field((H, W), 420, r))
    F, leaves, specks = [], [], []
    for x, y, a, order, tip in nodes:
        if not (20 < x < W - 20 and 20 < y < H - 20):
            continue
        if r.random() < 0.07:
            leaves.append(leaf(np.array([x, y]), a + r.choice([-1, 1]) * r.uniform(0.3, 0.9), r))
        if r.random() > 0.2 + 0.65 * bloom[int(y), int(x)] + 0.2 * tip:
            if r.random() < 0.2:
                b = a + r.choice([-1, 1]) * r.uniform(0.4, 1.2)
                specks.append(np.array([x, y]) + np.outer([4, 4 + r.uniform(5, 10)], [np.cos(b), np.sin(b)]))
            continue
        for _ in range(min(6, r.geometric(0.42))):
            bud_ = r.random() < (0.5 if tip else 0.3)
            R = r.uniform(20, 32) if bud_ else r.uniform(46, 88)
            off = a + r.choice([-1, 1]) * r.uniform(0.2, 1.8)
            c = np.array([x, y]) + r.uniform(0.3, 0.8) * R * np.array([np.cos(off), np.sin(off)])
            if all(np.hypot(*(c - f[0])) > 0.55 * (R + f[1]) for f in F):
                F.append((c, R, bud_, off))
    leaves = [(p, w) for halves, w in leaves for p in halves]
    nl, nk = len(leaves), len(specks)
    lay([p for p, _ in leaves], np.array([w for _, w in leaves]), loads(LEAF, r.uniform(0.3, 0.9, nl), r, HEART, 0.3),
        thick=0.32, spent=0.5, pickup=0.2, taper=0.85, ends=(0.3, 0.0))
    lay(specks, r.uniform(3.5, 5.5, nk), loads(np.concatenate([RED[2:], BARK[2:4], LEAF[:2]]), r.uniform(0.1, 0.9, nk), r),
        thick=0.45, spent=0.3, pickup=0.2)
    touches, tw, tt, tc, hearts, hw, ht, stam, calyx = [], [], [], [], [], [], [], [], []
    for c, R, bud_, off in F:
        d = np.array([np.cos(off), np.sin(off)])
        if bud_:
            touches.append(np.stack([c + d * R, c + 0.2 * d * R, c - 0.45 * d * R]))
            tw.append(0.5 * R)
            tt.append(r.uniform(0.4, 0.8))
            tc.append(r.random() < 0.55)
            calyx.append(np.stack([c - 0.35 * d * R, c - 0.75 * d * R]))
            continue
        p_, h_, s_, k_ = flower(c, R, off, r)
        pink = r.random() < 0.25
        touches += [p for p, _, _ in p_]
        tw += [w for _, w, _ in p_]
        tt += [t for _, _, t in p_]
        tc += [pink] * len(p_)
        if h_ is not None:
            hearts.append(h_)
            hw.append(0.16 * R)
            ht.append(pink or r.random() < 0.15)
        if k_ is not None:
            calyx.append(k_)
        stam += s_
    nt = len(touches)
    cols, share = loads(WHITE, np.array(tt) + r.normal(0, 0.1, nt), r, np.concatenate([PINK[1:], HEART[3:]]), 0.3)
    tc = np.array(tc)
    cols[tc, 1] = PINK[r.integers(1, 4, tc.sum())]
    lay(touches, np.array(tw), (cols, share), thick=0.38, spent=0.2, pickup=0.12, land=0.9, lift=0.5, taper=0.25,
        ends=(0.7, 0.2), merge=2, grooves=0.4, tails=0.4, fray=1.0)
    ht = np.array(ht, bool)
    cols, share = loads(HEART, np.full(len(hearts), 0.6), r, PINK[:2], 0.2)
    cols[ht] = loads(PINK, np.full(ht.sum(), 0.15), r)[0]
    lay(hearts, np.array(hw), (cols, share), thick=0.45, spent=0.3, pickup=0.35, land=0.4, grooves=0.3)
    nc = len(calyx)
    lay(calyx, r.uniform(4.5, 7, nc), loads(np.concatenate([RED[2:], BARK[1:4], LEAF[:1]]), r.uniform(0.2, 0.9, nc), r),
        thick=0.4, spent=0.4, pickup=0.25)
    ns = len(stam)
    lay(stam, r.uniform(1.6, 2.4, ns), loads(np.concatenate([RED[::2], BARK[1:3]]), np.full(ns, 0.5), r, spread=0.3),
        thick=0.5, spent=0.5, pickup=0.2, grooves=0.2, land=0.2)

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.7, 1.16))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
