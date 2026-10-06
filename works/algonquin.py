"""Pine on the Point, Evening. Oil on canvas, over a red ground.

Tom Thomson spent most of the years from 1912 to his death in 1917 in Algonquin Park, north of
Toronto, guiding, fire-ranging and painting from the canoe and the shore: small sketches on wood
panels, each done at one sitting with a loaded brush, the strokes set down once, side by side,
and the wood left bare between them. In the winters, in a shack in Toronto, he worked a few of
them up into canvases, The Jack Pine and The West Wind among them, bolder and flatter in design.
In The Jack Pine a red underpainting shows between the strokes of the sky and round the branches,
where he painted the sky up to the tree and not over it.

This pine is not one of his but is painted his way: the sky in broad level sweeps from lemon
through rose to a pale green, the far hills low and purple, the lake in bands of cold blue and
violet, the granite in a few broad planes from a flat brush, lit on top and in shadow below, and
the tree last, its branches dragged out downwind by the loaded brush.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Pine on the Point, Evening"
DATE = "2026"
MEDIUM = "Oil on canvas, over a red ground"
AFTER = "Tom Thomson in Algonquin Park, 1914–17: The Jack Pine; The West Wind; the oil sketches on wood panel"
ROOM = "Open Air"
YEAR = 1916
PLACE = "Algonquin Park, Ontario"
REGION = "Americas"
NOTE = ("A wind-bent pine on a granite point, dark against a northern lake and an evening sky laid in broad "
        "bands. The red ground is left showing round the tree, as Thomson left it.")

H, W = 2450, 2800
LIGHT = (-0.6, -0.5, 0.62)
LAKE = 1400            # where the far shore meets the water
SKYLINE = 1250         # about where the hills stand against the sky


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
GROUND = pal("#82301c", "#963b24", "#a84d30")                                 # burnt sienna and vermilion
GREEN = pal("#5c6c63", "#6b7a6a", "#7b8972", "#8a977a", "#98a383")            # viridian, ochre and white
ROSE = pal("#8a5d5c", "#986965", "#a5766e", "#b08378", "#b98f82")             # madder and white, dulled
LEMON = pal("#b9963f", "#c3a34b", "#ccb058", "#d3bb66", "#d9c475")            # cadmium lemon and ochre
GLOW = pal("#c27f3c", "#cd9048")
SKYMIX = np.concatenate([GLOW, ROSE[1:4], GREEN[2:5], LEMON[2:4]])
FAR = pal("#332a47", "#3d3352", "#483c5e", "#53466a", "#5f5177")              # the far hills
NEAR = pal("#1c1728", "#221c31", "#29223a", "#312944")
BLUE = pal("#1f304d", "#273a5a", "#304668", "#3b5376", "#476184")
VIOLET = pal("#2d2849", "#373156", "#423b63", "#4e4670", "#5a527d")
SHEEN = pal("#8f93a0", "#a2a3a8", "#b3b0a6", "#a8928f")                       # the sky caught in the water
OCHRE = pal("#7a5733", "#8c653c", "#9c7347", "#aa8053", "#b58c60")
RUST = pal("#4f251a", "#612e1e", "#743a24", "#86472c")
SLATE = pal("#2f2a36", "#3a3442", "#47404f", "#544c5c", "#61596a")            # grey-violet granite
ROSE_ROCK = pal("#7e5047", "#915e52", "#a26c5d", "#b07a69", "#bb8775")         # the granite's own pink
SHADOW = pal("#24223a", "#2c2a44", "#35324f", "#3f3b5b", "#4a4567")            # blue-violet, the rock in shadow
NEEDLES = pal("#0d130f", "#121a14", "#18221a", "#1f2b20", "#283626", "#33422c")
BARK = pal("#161111", "#201716", "#2c1e1a", "#40261b", "#5b311e", "#784025")
PRUSSIAN = pal("#121d2a", "#192736")
OLIVE = pal("#2f371e", "#3e4727")


def at(P):
    return np.clip(P[..., 1].astype(int), 0, H - 1), np.clip(P[..., 0].astype(int), 0, W - 1)


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a neighbour on the palette streaked in,
    and a third, now and then an accent picked up from elsewhere."""
    n, N = len(colours), len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n
    i = t.astype(int)
    j = np.clip(i + np.where(r.random(N) < t - i, 1, -1) * r.choice([1, 1, 2], N), 0, n - 1)
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


def spacing(length, width, cover):
    """Seeds far enough apart that strokes of this size cover the ground about `cover` times."""
    return float(np.sqrt(length * 2 * width / (cover * 0.866)))


def split(paths, inside, fine=6):
    """Break each stroke, to within a few px, where it passes over ground it may not cover, as a
    painter lifts the brush at the tree and sets it down again past it; `inside` says which of
    the strokes' points (N, m, 2) lie on ground it may cover. -> the pieces (M, n, 2), how long
    each is, and the stroke each came from"""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, (n - 1) * fine + 1)
    i = np.minimum(t.astype(int), n - 2)
    f = (t - i)[None, :, None]
    Q = paths[:, i] * (1 - f) + paths[:, i + 1] * f
    edge = np.diff(np.pad(inside(Q).astype(np.int8), ((0, 0), (1, 1))), axis=1)
    out, src = [], []
    for k in range(N):
        for a, b in zip(np.nonzero(edge[k] == 1)[0], np.nonzero(edge[k] == -1)[0]):
            if b - a > 2:
                u = np.linspace(a, b - 1.001, n)
                j = u.astype(int)
                out.append(Q[k, j] + (Q[k, j + 1] - Q[k, j]) * (u - j)[:, None])
                src.append(k)
    out = np.asarray(out, np.float32).reshape(-1, n, 2)
    return out, np.hypot(*np.diff(out, axis=1).transpose(2, 0, 1)).sum(1), np.asarray(src, int)


def ray(P, a, L, bend=0.0, n=8):
    """Strokes from P (N,2) at angles `a`, `L` long, turning `bend` radians per px. -> (N, n, 2)"""
    N = len(P)
    L = np.broadcast_to(np.asarray(L, np.float32), (N,))
    s = np.linspace(0, 1, n)[None, :] * L[:, None]
    th = np.asarray(a)[:, None] + np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None] * s
    step = np.diff(s, axis=1, prepend=0)
    return (P[:, None, :] + np.cumsum(np.stack([np.cos(th) * step, np.sin(th) * step], -1), axis=1)).astype(np.float32)


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back over
    the last, and lifting off now and then (keep < 1). -> index arrays into path"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m[np.linspace(0, len(m) - 1, min(len(m), 10)).astype(int)])
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def ridge(xs, ys, r, fine=4, broad=8):
    return np.interp(np.arange(W), xs, ys) + fine * noise.line1d(W, 40, r) + broad * noise.line1d(W, 170, r)


# ---- the pine: worked out stroke by stroke before anything is painted, so that the sky can be
# laid up to it and the red ground left round it

TRUNK = [(830, 1882, 66), (846, 1700, 58), (872, 1480, 50), (915, 1250, 43), (968, 1010, 36),
         (1025, 780, 29), (1080, 560, 22), (1132, 360, 16), (1182, 190, 10), (1236, 66, 5)]
# each tier: where its branch leaves the trunk, the branch out to its tip (dx, dy), its clumps of
# needles (how far along the branch, how long, how deep) and how far it reaches upwind
TIERS = [(90, [(0, 0), (50, -24), (120, -34), (200, -26)], [(0.4, 170, 46), (0.92, 140, 36)], (60, 30)),
         (235, [(0, 0), (110, -30), (260, -36), (420, -20), (560, 4)], [(0.35, 200, 52), (0.78, 260, 60), (1.0, 150, 40)],
          (110, 40)),
         (470, [(0, 0), (150, -16), (320, -10), (470, 14)], [(0.6, 260, 66), (0.98, 150, 50)], None),
         (700, [(0, 0), (180, -36), (430, -44), (700, -22), (940, 22), (1110, 70)],
          [(0.22, 260, 76), (0.55, 460, 118), (0.9, 330, 86)], (190, 60)),
         (1010, [(0, 0), (160, -6), (380, 16), (600, 54), (780, 108)], [(0.5, 380, 92), (0.9, 260, 70)], (80, 36)),
         (1250, [(0, 0), (150, -12), (320, -4), (470, 30)], [(0.75, 300, 62)], None)]
DEAD = [(1400, [(0, 0), (70, 24), (140, 34)]), (1560, [(0, 0), (-60, 20), (-120, 22)])]
ROOTS = [(-130, 24, 17), (120, 22, 15), (-60, 40, 11), (70, 44, 10)]


def pine(r):
    """The strokes of the tree, in the order they go on. -> list of (kind, paths, half-widths, tones)"""
    A = brush.path(np.asarray(TRUNK, float), 3.0)
    x_at = lambda y: float(np.interp(y, A[::-1, 1], A[::-1, 0]))
    w_at = lambda y: float(np.interp(y, A[::-1, 1], A[::-1, 2]))
    out = []
    # the trunk in ribbons of bark laid in overlapping strokes up its length, warmest on the left
    # where the glow catches it
    d = np.gradient(A[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    for c, half, tone, keep in ((-0.62, 0.48, 0.42, 1), (0.0, 0.48, 0.12, 1), (0.62, 0.48, 0.05, 1), (-0.86, 0.17, 0.9, 0.7)):
        f = c + 0.08 * np.sin(np.linspace(0, 5, len(A)) + r.uniform(0, 6))
        line = A[:, :2] + nrm * (A[:, 2] * f)[:, None]
        cut = pieces(line, 260, 60, r, keep)
        out.append(("bark", [line[m] for m in cut], np.array([half * A[m, 2].mean() for m in cut]),
                    tone + r.normal(0, 0.1, len(cut))))
    x0, y0 = TRUNK[0][:2]
    # the roots, gripping the rock: each leaves the trunk downward and turns out along the stone
    out.append(("bark", [brush.path(np.array([(x0 + 0.2 * dx, y0 - 40), (x0 + 0.55 * dx, y0 + 0.6 * dy), (x0 + dx, y0 + dy)]), 4.0)
                         for dx, dy, _ in ROOTS], np.array([w for _, _, w in ROOTS], float), r.uniform(0.05, 0.4, len(ROOTS))))
    for y0, pts in DEAD:
        x0, w0 = x_at(y0), w_at(y0)
        B = brush.path(np.asarray([(x0 + dx, y0 + dy) for dx, dy in pts], float), 3.0)
        cut = pieces(B, 90, 10, r)
        out.append(("branch", [B[m] for m in cut], np.linspace(0.3 * w0, 3, len(cut) + 1)[:-1], r.uniform(0.2, 0.7, len(cut))))
    for y0, pts, clumps, upwind in TIERS:
        out += tier(np.asarray([(x_at(y0) + dx, y0 + dy) for dx, dy in pts], float), 0.3 * w_at(y0) + 3, clumps, upwind, r)
    return out


def tier(pts, w0, clumps, upwind, r):
    """One tier: the branch, then its clumps of needles dragged out downwind in broad strokes from
    a loaded brush, each clump flat along the top and hanging in lobes below; the branch again
    where it shows between them."""
    B = brush.path(pts, 3.0)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(B, axis=0).T))])
    Ls = s[-1]
    T = np.gradient(B, axis=0)
    ang = np.arctan2(T[:, 1], T[:, 0])
    cut = pieces(B, 170, 30, r)
    wide = np.array([w0 * (1 - 0.72 * s[m].mean() / Ls) for m in cut])
    out = [("branch", [B[m] for m in cut], wide, r.uniform(0.15, 0.7, len(cut)))]

    def site(sa, u):
        i = np.clip(np.searchsorted(s, sa), 0, len(B) - 1)
        a = ang[i]
        return (B[i] + np.stack([-np.sin(a), np.cos(a)], 1) * u[:, None]).astype(np.float32), a

    for f, L, depth in clumps:
        n = 14 + int(L * depth / 2200)
        q = r.uniform(-1, 1, n)                                     # across the clump: -1 its top, 1 its bottom
        sa = np.clip(f * Ls - 0.5 * L, 0.03 * Ls, None) + L * r.uniform(0, 0.5, n) * (1 - 0.3 * np.abs(q))
        P, a = site(sa, depth * np.where(q < 0, 0.55, 0.75) * q)
        out.append(("needles", ray(P, a + 0.1 * q + r.normal(0, 0.05, n), L * r.uniform(0.4, 0.9, n) * (1 - 0.35 * q * q),
                                   0.001 * (q + 0.4), 9), depth / 4.2 * r.uniform(0.6, 1.15, n),
                    0.3 + 0.35 * np.clip(-q - 0.5, 0, 1) + r.normal(0, 0.12, n)))
        # its underside hangs in short lobes, swept back by the wind
        m = 2 + int(L / 110)
        P, a = site(f * Ls + L * r.uniform(-0.35, 0.45, m), np.full(m, 0.5 * depth))
        out.append(("lobe", ray(P, a + r.uniform(0.5, 1.1, m), depth * r.uniform(0.3, 0.6, m), -0.002, 6),
                    depth / 5 * r.uniform(0.7, 1.1, m), r.uniform(0.1, 0.4, m)))
        m = 1 + int(L / 200)
        P, a = site(f * Ls + L * r.uniform(-0.3, 0.35, m), np.full(m, -0.4 * depth))
        out.append(("lobe", ray(P, a - r.uniform(0.3, 0.7, m), depth * r.uniform(0.25, 0.45, m), 0.004, 5),
                    depth / 7 * r.uniform(0.7, 1.1, m), r.uniform(0.4, 0.8, m)))
        # and its tip streams on downwind in a few long strokes dragged out dry
        m = 2 + int(L / 160)
        P, a = site(f * Ls + 0.35 * L + r.uniform(-20, 20, m), depth * r.uniform(-0.4, 0.4, m))
        out.append(("streak", ray(P, a + r.normal(0.05, 0.08, m), r.uniform(70, 190, m), 0.0008, 8),
                    r.uniform(3.5, 6.5, m), r.uniform(0.15, 0.5, m)))
    if upwind:
        # upwind of the trunk only a short clump survives, brushed back toward it
        reach, depth = upwind
        m = int(reach / 16) + 5
        q = r.uniform(-1, 1, m)
        P = (B[0] + np.stack([-reach * r.uniform(0.5, 1.0, m) * (1 - 0.5 * q * q), 0.7 * depth * q - 0.3 * depth], 1)).astype(np.float32)
        out.append(("needles", ray(P, r.normal(0.08 * q, 0.06, m), reach * r.uniform(0.45, 0.9, m), 0.0, 6),
                    depth / 4.5 * r.uniform(0.6, 1.1, m), r.uniform(0.2, 0.6, m)))
    show = [j for j in range(max(1, len(cut) // 2)) if r.random() < 0.7]
    out.append(("branch", [B[cut[j]] for j in show], wide[show] * 0.8, r.uniform(0.1, 0.5, len(show))))
    return out


def silhouette(strokes):
    """Where the tree's paint will lie: each stroke drawn as a fat line, blunt where the brush lands."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for kind, paths, ws, _ in strokes:
        for p, w in zip(paths, ws):
            d.line([tuple(q) for q in p], fill=255, width=max(1, int(round(1.8 * w))), joint="curve")
            q = p[0]
            d.ellipse([q[0] - 0.9 * w, q[1] - 0.9 * w, q[0] + 0.9 * w, q[1] + 0.9 * w], fill=255)
    return np.asarray(im) > 0


# ---- the granite: the point the pine stands on, its top a row of round-shouldered whalebacks
# stepping down into the water, the nearer in front

HUMPS = [(200, 2380, 640, 590), (880, 2340, 560, 470), (1560, 2470, 560, 470), (2230, 2520, 560, 380),
         (2800, 2520, 380, 320)]
TOP = 1700             # nothing of the rock rises above this row
BROW = ([0, 350, 700, 1000, 1300, 1600, 1900, 2200, 2500, 2800], [2060, 2085, 2070, 2110, 2125, 2155, 2210, 2255, 2285, 2300])
LEDGE = ([0, 300, 600, 900, 1150], [2265, 2280, 2320, 2395, 2470])


def granite(r):
    """The rock in three planes: its top, lit by the sky and sloping away to the right; the face
    below it in shadow; and a ledge in front at the lower left catching the light again.
    -> plane of each pixel (-1 off the rock, 0 the top, 1 the face, 2 the ledge), and the brow
    where the top turns down into the face, per column"""
    yy, xx = np.mgrid[TOP:H, 0:W].astype(np.float32)
    warp = 0.05 * noise.field((H - TOP, W), 70, r)
    rho = np.stack([np.abs((xx - cx) / rx) ** 2.8 + ((yy - cy) / ry) ** 2 for cx, cy, rx, ry in HUMPS]).min(0) + warp
    inside = np.maximum.accumulate(rho < 1, axis=0)                 # under the top of any hump, all is rock
    brow = ridge(*BROW, r, 3, 12)
    ledge = np.interp(np.arange(W), *LEDGE, right=H + 100) + 10 * noise.line1d(W, 120, r)
    plane = np.full((H, W), -1, np.int8)
    plane[TOP:] = np.where(inside, np.where(yy < brow[None, :], 0, np.where(yy >= ledge[None, :], 2, 1)), -1)
    return plane, brow


def paint(seed=1916):
    r = noise.rng(seed)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#dccdb0", thread=3.4)
    rgb, height = ground.color.copy(), ground.tooth * 0.3
    sparing = noise.smoothstep(-0.7, -1.6, noise.field((H, W), 360, r))      # where the brush went thinner

    def lay(paths, w, load, rr=None, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r if rr is None else rr, share=load[1],
                        **{"ends": (0.25, 0.1), "fray": 1.2, **kw})

    def strew(mask, sp, keep=None):
        P = dabs.scatter((H + 2 * int(sp), W + 2 * int(sp)), sp, r) - int(sp)
        P = P[mask[at(P)]]
        odds = (1 if keep is None else keep[at(P)]) * (1 - 0.3 * sparing[at(P)])
        return P[r.random(len(P)) < odds]

    def stroke(paths, w, load, allowed, back=0.5, short=1.6, **kw):
        """Lay strokes broken where they would cross ground they may not cover (a mask, or a test
        of points); pieces too short to be a stroke are dropped."""
        w = np.broadcast_to(np.asarray(w, np.float64), (len(paths),))
        paths, L, k = split(paths, allowed if callable(allowed) else lambda Q: allowed[at(Q)])
        ok = L > short * w[k]
        paths, k = paths[ok], k[ok]
        flip = r.random(len(paths)) < back
        paths[flip] = paths[flip, ::-1]
        kw = {key: v[k] if isinstance(v, np.ndarray) else v for key, v in kw.items()}
        lay(paths, w[k], (load[0][k], load[1][k]), **kw)

    # the ground: red, rubbed on with a rag, thinner in streaks where the weave comes through
    rub = noise.fbm((H, W), 300, r, octaves=4)
    red = GROUND[0] + (GROUND[2] - GROUND[0]) * noise.smoothstep(-1.6, 1.6, rub)[..., None]
    wipe = noise.stretched((W, H), 14, 260, r).T[:H, :W] + 0.5 * noise.field((H, W), 200, r)
    a = np.clip(0.92 + 0.04 * rub - 0.3 * noise.smoothstep(1.2, 2.4, wipe), 0.5, 0.97)[..., None]
    rgb = rgb * (1 - a) + red * (0.8 + 0.4 * ground.tooth[..., None]) * a

    tree = pine(noise.rng(seed + 1))
    gap = ndimage.distance_transform_edt(~silhouette(tree))
    free = gap > np.clip(2.5 + 2.5 * noise.field((H, W), 45, r), 0, 6)
    near_tree = free & (gap < 150)

    far = ridge([0, 220, 430, 640, 830, 1010, 1250, 1500, 1760, 2000, 2240, 2500, 2800],
                [1236, 1204, 1182, 1196, 1238, 1272, 1290, 1270, 1242, 1226, 1246, 1214, 1232], r)
    near = ridge([0, 160, 330, 470, 560, 1640, 1760, 1950, 2200, 2450, 2650, 2800],
                 [1338, 1330, 1352, 1380, 1440, 1440, 1372, 1334, 1318, 1300, 1310, 1302], r, 3, 5)
    lake = LAKE + 3 * noise.line1d(W, 300, r)
    plane, brow = granite(r)
    rock = plane >= 0
    shore = np.argmax(rock, axis=0).astype(np.float32)              # the top of the rock, per column
    shade = noise.field((H, W), 300, r)
    scrub = lambda n: dict(thick=0.04, spent=0.8, dry=r.uniform(0.15, 0.4, n), hide=0.9, grooves=0.3, pickup=0.2, fray=2.5)

    # ---- the sky: each band laid in long loaded sweeps from left or right, of every length, a few
    # running half across the canvas; where two bands meet the brush carries both, wet into wet
    flow = (0.03 * noise.field((H, W), 650, r) + 0.015 * noise.field((H, W), 220, r)).astype(np.float32)
    wob = 0.05 * noise.field((H, W), 520, r) + 0.025 * noise.field((H, W), 160, r)
    bars = noise.stretched((W, H), 60, 1200, r).T[:H, :W]           # the rose lies in long bars
    sky = yy < far[None, :] + 45

    def sky_load(P):
        n = len(P)
        t = P[:, 1] / SKYLINE + wob[at(P)]
        rose = 1.5 * np.exp(-((t - 0.5) / 0.14) ** 2) + 0.55 * bars[at(P)] - 1.0     # above 0 in the band of rose
        which = np.where(rose + r.normal(0, 0.35, n) > 0, 1, np.where(t + r.normal(0, 0.09, n) > 0.57, 2, 0))
        tone = np.select([which == 0, which == 1], [0.1 + 0.85 * np.clip(t / 0.5, 0, 1), 0.3 + 0.5 * np.clip((t - 0.35) / 0.35, 0, 1)],
                         0.2 + 0.8 * np.clip((t - 0.55) / 0.4, 0, 1)) + 0.08 * shade[at(P)] + r.normal(0, 0.1, n)
        load = mixed((GREEN, ROSE, LEMON), which, tone, r, SKYMIX, 0.3)
        to_rose, to_split = np.exp(-np.abs(rose) / 0.3), np.exp(-np.abs(t - 0.57) / 0.06) * (which != 1)
        other = np.where(which == 1, np.where(t < 0.5, 0, 2), np.where(to_rose > to_split, 1, 2 - which))
        both = r.random(n) < np.maximum(to_rose, to_split)
        load[0][both, 1] = mixed((GREEN, ROSE, LEMON), other, tone, r)[0][both, 0]
        load[1][both, 1] = r.uniform(0.35, 0.6, both.sum())
        return np.clip(t, 0, 1), load

    sweep = dict(thick=0.1, spent=0.6, pickup=0.7, grooves=0.7, lips=0.0, fray=1.8, ends=(0.3, 0.15), taper=0.55,
                 tails=0.7, land=0.4, merge=5.0)          # thinner, pushed wet into wet, so the bands run together
    P = strew(sky, spacing(700, 45, 2.6))
    n = len(P)
    t, load = sky_load(P)
    stroke(impasto.follow(flow, P, r.uniform(400, 1000, n), 40, 0.0, r.normal(0, 0.02, n)), r.uniform(38, 55, n), load,
           free & sky, short=1.0, **scrub(n))
    T = np.clip(yy / SKYLINE, 0, 1)
    P = strew(sky, spacing(476, 22, 2.6), (16.5 / ((1.15 - 0.4 * T) * (36 - 14 * T))).astype(np.float32))
    n = len(P)
    t, load = sky_load(P)
    L = np.clip(560 * np.exp(r.normal(0, 0.5, n)) * (1.15 - 0.4 * t), 160, 1500)
    stroke(impasto.follow(flow, P, L, 48, r.normal(0, 0.00012, n), r.normal(0, 0.015, n)),
           np.clip((36 - 14 * t) * np.exp(r.normal(0, 0.35, n)), 10, 60), load, free & sky, **sweep)
    P = strew(sky & near_tree, spacing(260, 15, 1.3))
    n = len(P)
    t, load = sky_load(P)
    stroke(impasto.follow(flow, P, r.uniform(150, 400, n), 24, 0.0, r.normal(0, 0.05, n)), r.uniform(12, 20, n), load,
           free & sky, **sweep)

    # ---- the far hills, low and purple: strokes along the ridge and level lower down
    slope = np.arctan(np.gradient(ndimage.gaussian_filter1d(far, 30)))[None, :]
    hill = (yy >= far[None, :]) & (yy < lake[None, :] + 10)
    field = (slope * noise.smoothstep(90, 10, yy - far[None, :]) + 0.02 * noise.field((H, W), 300, r)).astype(np.float32)
    hill_tone = lambda P: 0.2 + 0.5 * np.clip((P[:, 1] - far[at(P)[1]]) / (lake[at(P)[1]] - far[at(P)[1]]), 0, 1) ** 1.5 \
        + 0.1 * shade[at(P)]
    for mask, L, w, sp in ((hill, (90, 220), (10, 15), spacing(150, 12, 3.4)), (hill & near_tree, (40, 110), (8, 12), 22)):
        P = strew(mask, sp)
        n = len(P)
        stroke(impasto.follow(field, P, r.uniform(*L, n), 10, r.normal(0, 0.0015, n)), r.uniform(*w, n),
               loads(FAR, hill_tone(P), r, np.concatenate([NEAR[2:], ROSE[:1]]), 0.12), free & hill, thick=0.2, spent=0.5,
               pickup=0.45, grooves=1.1)
    x = np.arange(-20, W + 20, 60.0)
    x += r.uniform(-15, 15, len(x))
    P = np.stack([x, far[np.clip(x.astype(int), 0, W - 1)] + 12], 1).astype(np.float32)
    n = len(P)
    stroke(impasto.follow(field, P, r.uniform(130, 260, n), 12), r.uniform(10, 13, n), loads(FAR, r.uniform(0.1, 0.35, n), r),
           free & hill, back=0.0, thick=0.2, spent=0.45, pickup=0.35)

    # the near shore, darker, with its fringe of spruce
    land = (yy >= near[None, :]) & (yy < lake[None, :] + 10)
    P = strew(land, spacing(120, 11, 3.2))
    n = len(P)
    stroke(impasto.follow(np.zeros((1, 1), np.float32), P, r.uniform(70, 170, n), 8, r.normal(0, 0.002, n)),
           r.uniform(9, 13, n), loads(NEAR, r.uniform(0.2, 0.8, n), r, FAR[:2], 0.15), free & land, thick=0.22, spent=0.5)
    x = np.sort(r.uniform(0, W, 900))
    x = x[near[x.astype(int)] < lake[x.astype(int)] - 25]
    x = x[noise.line1d(W, 60, r)[x.astype(int)] + r.normal(0, 0.6, len(x)) > -0.2]
    n = len(x)
    P = np.stack([x, near[x.astype(int)] + 14], 1).astype(np.float32)
    tall = r.uniform(18, 62, n) * (0.6 + 0.4 * noise.smoothstep(-1, 1, noise.line1d(W, 120, r)[x.astype(int)]))
    stroke(ray(P, -np.pi / 2 + r.normal(0, 0.06, n), tall, 0.0, 6), r.uniform(4, 7, n), loads(NEAR, r.uniform(0, 0.4, n), r),
           free, back=0.0, thick=0.3, spent=0.6, taper=0.85, ends=(0.2, 0.7), tails=0.2)

    # ---- the lake, in level bands of cold blue and violet, laid in long strokes that run into one
    # another: dark under the far shore where the hills are reflected, a pale band from the sky,
    # deepening toward the rock; and a few glints
    water = (yy >= lake[None, :] - 4) & (yy < shore[None, :] + 30)
    depth = np.clip((yy - lake[None, :]) / (2150 - LAKE), 0, 1)
    bands = noise.stretched((W, H), 60, 1800, r).T[:H, :W]
    calm = (0.012 * noise.field((H, W), 400, r)).astype(np.float32)

    def lake_load(P):
        n = len(P)
        d, b = depth[at(P)], bands[at(P)]
        which = (b + 0.25 * r.normal(0, 1, n) > 0.1).astype(int)
        tone = 0.6 - 0.45 * d + 0.3 * np.exp(-((d - 0.13) / 0.07) ** 2) + 0.12 * shade[at(P)] + 0.12 * (b < -0.9)
        tone[d < 0.03] -= 0.45                                         # the hills reflected under the far shore
        load = mixed((BLUE, VIOLET), which, tone, r, SHEEN, 0.05)
        both = r.random(n) < np.exp(-np.abs(b - 0.1) / 0.35)
        load[0][both, 1] = mixed((BLUE, VIOLET), 1 - which, tone, r)[0][both, 0]
        load[1][both, 1] = r.uniform(0.35, 0.6, both.sum())
        return d, load

    P = strew(water, spacing(650, 36, 2.0))
    d, load = lake_load(P)
    n = len(P)
    stroke(impasto.follow(calm, P, r.uniform(400, 900, n), 32), r.uniform(28, 42, n), load, free & water, short=1.0, **scrub(n))
    P = strew(water, spacing(258, 10, 2.6), (6 / ((0.6 + 0.8 * depth) * (10 + 12 * depth))).astype(np.float32))
    d, load = lake_load(P)
    n = len(P)
    stroke(impasto.follow(calm, P, np.clip(380 * np.exp(r.normal(0, 0.5, n)) * (0.6 + 0.8 * d), 120, 1300), 40,
                          r.normal(0, 0.0002, n)),
           np.clip((10 + 12 * d) * np.exp(r.normal(0, 0.3, n)), 5, 34), load, free & water, thick=0.11, spent=0.6, pickup=0.65,
           grooves=0.8, fray=1.8, ends=(0.3, 0.15), taper=0.55, tails=0.7, merge=4.0)
    P = strew(water & (near_tree | (yy > shore[None, :] - 140)), spacing(200, 11, 1.6))       # and along the shore
    d, load = lake_load(P)
    n = len(P)
    stroke(impasto.follow(calm, P, r.uniform(120, 300, n), 20), r.uniform(9, 14, n), load, free & water, thick=0.16,
           spent=0.6, pickup=0.6, grooves=0.8)
    P = strew(water & (depth < 0.7), 130, noise.smoothstep(-0.3, 1.2, noise.field((H, W), 300, r)))
    n = len(P)
    stroke(ray(P, r.normal(0, 0.01, n), r.uniform(40, 160, n), 0.0, 8), r.uniform(3, 5.5, n),
           loads(SHEEN, r.uniform(0.5, 1.0, n), r), free & water, short=1.0, thick=0.2, spent=0.7, dry=r.uniform(0.1, 0.35, n))

    # ---- the granite in three planes, each in a few broad loaded strokes of a flat brush: the face
    # in blue-violet shadow first, then the top over its upper edge in ochre and rose where the sky
    # lights it, and the ledge lit again in front
    slab = dict(thick=0.14, spent=0.35, tails=0.2, fray=1.2, grooves=1.0, ends=(0.05, 0.1), taper=0.1, land=0.35,
                lift=0.5, lips=0.15, pickup=0.4, merge=2.0)
    slope = np.arctan(np.gradient(ndimage.gaussian_filter1d(0.5 * shore + 0.5 * brow, 80)))
    P = strew(rock, spacing(350, 45, 2.2))
    n = len(P)
    pl = plane[at(P)]
    stroke(ray(P, np.where(pl == 1, 0.95, slope[at(P)[1]]) + r.normal(0, 0.1, n), r.uniform(200, 500, n), 0.0, 12),
           r.uniform(40, 60, n), mixed((OCHRE, SHADOW, OCHRE), pl, np.where(pl == 1, 0.45, 0.6) + r.normal(0, 0.1, n), r,
                                       ROSE_ROCK, 0.3), lambda Q: (plane[at(Q)] == pl[:, None]) & free[at(Q)], short=1.0, **scrub(n))
    for which, sp, L, w, colours, accent in ((1, 90, (200, 450), (36, 58), (SHADOW,), np.concatenate([RUST[1:3], SLATE[3:]])),
                                             (0, 85, (160, 460), (30, 48), (OCHRE, ROSE_ROCK), RUST[2:]),
                                             (2, 70, (150, 400), (30, 45), (OCHRE, ROSE_ROCK), SHADOW[3:])):
        reach = np.zeros((H, W), bool)                                  # each plane's strokes run a little over the last
        reach[TOP:] = ndimage.distance_transform_edt(plane[TOP:] != which) < 22
        P = strew(reach & rock, sp)
        n = len(P)
        x = np.clip(P[:, 0].astype(int), 0, W - 1)
        down = np.clip((P[:, 1] - shore[x]) / np.maximum(brow[x] - shore[x], 1), 0, 1.5)
        if which == 0:
            a_ = slope[x] + r.normal(0, 0.08, n)                     # along the top as it falls away
            tone = 0.75 - 0.3 * down - 0.2 * x / W
        elif which == 1:
            a_ = 0.95 + r.normal(0, 0.3, n)                          # down the face as it falls away
            tone = 0.55 - 0.3 * noise.smoothstep(2150, 2450, P[:, 1]) - 0.15 * x / W
        else:
            a_ = 0.18 + r.normal(0, 0.08, n)                         # along the ledge
            tone = 0.6 - 0.15 * x / W
        tone = tone + r.normal(0, 0.1, n)
        fam = (r.random(n) < 0.4).astype(int) if len(colours) > 1 else np.zeros(n, int)
        stroke(ray(P, a_, r.uniform(*L, n), r.normal(0, 0.0006, n), 10), r.uniform(*w, n),
               mixed(colours, fam, tone, r, accent, 0.25), reach & rock & free, back=0.4, short=1.0, **slab)
    # and up to the foot of the tree in smaller strokes
    P = strew(rock & near_tree, 26)
    n = len(P)
    pl = plane[at(P)]
    stroke(ray(P, np.where(pl == 1, 0.95, slope[at(P)[1]]) + r.normal(0, 0.1, n), r.uniform(50, 140, n), 0.0, 6),
           r.uniform(13, 20, n), mixed((OCHRE, SHADOW, OCHRE), pl, np.where(pl == 1, 0.45, 0.6) + r.normal(0, 0.1, n), r,
                                       ROSE_ROCK, 0.3),
           lambda Q: (plane[at(Q)] == pl[:, None]) & free[at(Q)], back=0.3, short=0.9, **slab)

    # ---- the pine, last, in the darkest paint, with its own hand
    rt = noise.rng(seed + 2)
    for kind, paths, ws, tone in tree:
        k = len(paths)
        if not k:
            continue
        paths = [np.asarray(p, np.float32) for p in paths]
        if kind == "bark":
            lay(paths, ws * rt.uniform(0.9, 1.2, k), loads(BARK, tone, rt, np.concatenate([RUST[2:], PRUSSIAN]), 0.3), rt,
                thick=0.3, spent=0.55, pickup=0.3, ends=(0.2, 0.2), land=0.3, fray=1.6)
        elif kind == "branch":
            lay(paths, ws, loads(BARK, tone, rt, RUST[1:3], 0.35), rt, thick=0.3, spent=0.6, taper=0.45, pickup=0.3)
        elif kind == "streak":
            lay(paths, ws, loads(NEEDLES, tone, rt, OLIVE, 0.25), rt, thick=0.26, spent=0.85, taper=0.5, tails=0.8, pickup=0.3,
                dry=rt.uniform(0.0, 0.35, k))
        elif kind == "needles":
            lay(paths, ws, loads(NEEDLES, tone, rt, np.concatenate([PRUSSIAN, OLIVE, BARK[3:5]]), 0.25), rt,
                thick=0.3, spent=0.4, taper=0.2, tails=0.3, fray=1.6, pickup=0.35, ends=(0.4, 0.25), grooves=1.0)
        else:
            lay(paths, ws, loads(NEEDLES, tone, rt, np.concatenate([OLIVE, BARK[4:5]]), 0.25), rt, thick=0.3, spent=0.45,
                taper=0.3, tails=0.3, ends=(0.3, 0.5), pickup=0.3, fray=1.5)

    height = ndimage.gaussian_filter(height, 0.7)      # the lamp sees the surface a hair softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.8, gloss=0, reach=(0.7, 1.2))
    return img + 0.06 * impasto.glints(height, LIGHT)[..., None]
