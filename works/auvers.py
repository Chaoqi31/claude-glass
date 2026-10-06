"""Wheat Field under a Heavy Sky. Oil on canvas, in thick strokes from a loaded brush.

In May 1890 Van Gogh left the asylum at Saint-Rémy for Auvers-sur-Oise, north of Paris, and
in July, in the last weeks of his life, he painted the wheat on the plateau above the village
on canvases twice as wide as they were high: "immense stretches of wheatfields under troubled
skies", he wrote to his brother. He painted each in a sitting, wet into wet, with a loaded
brush, and gave each part of the picture its own stroke: the ripe wheat in short flicks that
rise and toss as the wind goes over it, the sky in broad strokes that turn, a cart track in
long strokes dragged along it, and the crows in two or three strokes of black apiece. Since
then his chrome yellows have browned a little and his whites have yellowed.

This field is not one of his but is painted his way: the track comes in from the lower left,
swings right and stops in the wheat, and a storm comes on over a low swell of land.
"""

import numpy as np
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Wheat Field under a Heavy Sky"
DATE = "2026"
MEDIUM = "Oil on canvas, in thick strokes from a loaded brush"
AFTER = "Vincent van Gogh, the last wheat fields at Auvers, July 1890: Wheatfield with Crows; Wheatfield under Thunderclouds"
ROOM = "Open Air"
YEAR = 1890
PLACE = "Auvers-sur-Oise"
REGION = "Europe"
NOTE = ("Ripe wheat under a storm, painted in one sitting: the wheat in short upward flicks, the sky in broad "
        "turning strokes, the track in long dragged ones. Each crow is two or three strokes of black.")

H, W = 1860, 3800          # a double square, 50 by 100 cm
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
SKY = pal("#080d19", "#0b1322", "#0e1a2e", "#12223c", "#162b4c", "#1b355d", "#21416f", "#284e81", "#305b92",
          "#3a68a0", "#4675ab", "#5383b3", "#6391ba")                                # Prussian, ultramarine, cobalt
CERULEAN = pal("#6a9dba", "#7fadc2", "#94bbc6")
CLOUD = pal("#6a7f9a", "#8396ad", "#9eafbe", "#b7c3c9", "#ccd3d2", "#dbddd6", "#e5e3d8")  # whites, yellowed a little
CREAM = pal("#e9ddb9", "#e0d0a2")
WHEAT = pal("#5f480f", "#775d13", "#8f7318", "#a4861e", "#b69729", "#c5a735", "#d1b545", "#dbc158")  # chrome, browned
LEMON = pal("#cbb43e", "#d6c04c", "#dfcb5c", "#e6d470")
ORANGE = pal("#984e17", "#ab5c1c", "#bc6c24", "#c97e30")
RUST = pal("#5f2818", "#74331c", "#894123", "#9c522b")                              # red ochre, burnt sienna
OLIVE = pal("#2d2a10", "#403a14", "#544d1a", "#686020")
GREEN = pal("#2c4820", "#3e5f29", "#537832", "#6b903a", "#86a645", "#9fb653")       # viridian and chrome
EARTH = pal("#4f2114", "#652b19", "#7b3820", "#8f4628", "#a05734", "#ae6a45", "#bb8060", "#c6977b", "#cea98f",
            "#d4b7a0")                                                               # red ochre let down with white
BLACK = pal("#06070a", "#0a0c10", "#0f1218", "#0c1522")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


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


def sheaves(field, seeds, length, width, k, r, n=8, bend=0.0, fan=0.0, tilt=0.0):
    """Strokes side by side in little sheaves: from each seed, k strokes about a brush-width apart
    along the same field, their starts staggered and fanning out by `fan`. -> paths, half-widths, seed of each"""
    N = len(seeds)
    a = field[at(seeds)] + tilt
    d = np.stack([np.cos(a), np.sin(a)], -1)[:, None, :]
    across = np.stack([-np.sin(a), np.cos(a)], -1)[:, None, :]
    j = np.arange(k) - (k - 1) / 2
    L = np.asarray(length)[:, None] * np.exp(r.normal(0, 0.25, (N, k)))
    w = np.asarray(width)[:, None] * np.exp(r.normal(0, 0.15, (N, k)))
    P = seeds[:, None, :] + across * (1.8 * w * (j + r.normal(0, 0.12, (N, k))))[..., None] \
        + d * (L * r.uniform(-0.15, 0.15, (N, k)))[..., None]
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None] + fan * j / L
    tl = np.broadcast_to(np.asarray(tilt, np.float32), (N,))[:, None].repeat(k, 1)
    return impasto.follow(field, P.reshape(-1, 2), L.ravel(), n, turn.ravel(), tl.ravel()), w.ravel(), np.repeat(np.arange(N), k)


def spacing(k, length, width, cover):
    """Seeds far enough apart that sheaves of k strokes cover the ground about `cover` times."""
    return float(np.sqrt(k * length * 2 * width / (cover * 0.866)))


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def trim(paths, mask):
    """Cut each stroke where it first leaves the mask."""
    n = paths.shape[1]
    ins = mask[np.clip(paths[..., 1].astype(int), 0, H - 1), np.clip(paths[..., 0].astype(int), 0, W - 1)]
    k = np.where(ins.all(1), n, ins.argmin(1))
    t = np.linspace(0, 1, n)[None, :] * np.maximum(k - 1, 0)[:, None]
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    row = np.arange(len(paths))[:, None]
    return paths[row, i] * (1 - f) + paths[row, np.minimum(i + 1, n - 1)] * f


# ---- the land: a low swell, the field going back to it, and the track

def horizon(r):
    xs = [0, 600, 1200, 1800, 2400, 3000, 3500, 3800]
    ys = [778, 772, 792, 774, 748, 730, 722, 730]
    return np.interp(np.arange(W), xs, ys) + 7 * noise.line1d(W, 260, r) + 3 * noise.line1d(W, 45, r)


TRACK = [(1010, 2000), (1120, 1760), (1330, 1570), (1620, 1430), (1860, 1320), (2000, 1220), (2070, 1130),
         (2090, 1075)]


def track(hz, r):
    """Where each pixel lies on the track: across it (-1 to 1 inside, the sign the side), how far
    along it (0 at the foot of the canvas, 1 where it stops in the wheat), the way it runs there,
    and its half-width."""
    C = brush.path(np.asarray(TRACK, float), 2.0)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    xi = np.clip(C[:, 0].astype(int), 0, W - 1)
    d = np.clip((C[:, 1] - hz[xi]) / (H - hz[xi]), 0, 1.2)
    half = 260 * d ** 0.9 * (0.5 + 0.5 * noise.smoothstep(s[-1], s[-1] - 300, s))     # narrowing as it goes in
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    grid = np.full((H, W), -1, np.int32)
    ok = (C[:, 0] >= 0) & (C[:, 0] < W) & (C[:, 1] >= 0) & (C[:, 1] < H)
    grid[C[ok, 1].astype(int), C[ok, 0].astype(int)] = np.nonzero(ok)[0]
    dist, (iy, ix) = ndimage.distance_transform_edt(grid < 0, return_indices=True)
    i = grid[iy, ix]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (xx - C[i, 0]) * -T[i, 1] + (yy - C[i, 1]) * T[i, 0]
    across = np.sign(v) * dist / half[i] * (1 + 0.1 * noise.field((H, W), 55, r))
    return across.astype(np.float32), (s[i] / s[-1]).astype(np.float32), np.arctan2(T[i, 1], T[i, 0]).astype(np.float32), half[i]


def wheat_field(depth, r):
    """Which way the wheat's strokes run: far off, where the ears are seen from above, in level
    dashes; nearer, rising and leaning with the wind, which comes in gusts."""
    gust = noise.field((H, W), 480, r)
    eddy = noise.field((H, W), 160, r)
    far = 0.14 * eddy - 0.05
    near = -np.pi / 2 + 0.45 + 0.4 * gust + 0.18 * eddy
    a = blend(far, near, noise.smoothstep(0.03, 0.32, depth))
    return np.where(np.sin(a) > 0, a - np.pi, a).astype(np.float32)      # every flick goes up from where it lands


# ---- the sky: a wind from the left, turning in slow eddies, and a few clouds heaped up in it

EDDIES = [(420, 260, 380, 0.8), (1550, 200, 420, -0.7), (2250, 520, 330, 0.6), (3050, 380, 400, -0.8),
          (3600, 120, 300, 0.6)]
# each cloud as lobes (x, y, half-width, half-height): heaped up on top, flatter along the base
CLOUDS = [[(560, 520, 85, 40), (690, 455, 110, 70), (850, 395, 105, 88), (990, 345, 80, 66), (1110, 400, 115, 62),
           (1270, 445, 115, 40), (1430, 470, 95, 26), (760, 520, 120, 38)],
          [(2650, 282, 70, 45), (2760, 238, 80, 62), (2880, 272, 75, 42), (2990, 302, 60, 25)],
          [(3120, 672, 80, 30), (3260, 642, 95, 48), (3400, 612, 70, 45), (3520, 657, 100, 34), (3660, 682, 80, 22)]]


def puff(x, y):
    """How far into the clouds a point lies (above about 0.6 is cloud)."""
    m = 0
    for cx, cy, rx, ry in (l for c in CLOUDS for l in c):
        m = m + np.exp(-0.42 * (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))
    return m


def sky_field(r):
    """The stream the sky's strokes follow: a drift rising a little to the right, turned in slow
    eddies, and parting round the clouds. -> directions, and the puff of the clouds on the same grid"""
    s = 4
    yy, xx = np.mgrid[0:H + s:s, 0:W + s:s].astype(np.float32)
    psi = yy + 0.1 * xx + 55 * noise.field(yy.shape, 110, r)
    for cx, cy, sg, spin in EDDIES:
        psi += spin * 1.2 * sg * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg))
    m = puff(xx, yy)
    psi += 140 * np.tanh(m)                     # the wind goes round a cloud, not through it
    gy, gx = np.gradient(psi)
    a = ndimage.zoom(np.arctan2(-gx, gy).astype(np.float32), s, order=0)[:H, :W]
    return a, ndimage.zoom(m.astype(np.float32), s, order=1)[:H, :W]


def cloud_field(m):
    """How the brush models a cloud: round the edge of each heap, and winding in toward its heart,
    so that the strokes spiral through it rather than ring it."""
    gy, gx = np.gradient(ndimage.gaussian_filter(m, 20))
    return (np.arctan2(-gx, gy) + 0.6 * noise.smoothstep(0.7, 1.5, m)).astype(np.float32)


# ---- the crows: (x, y, span), the near birds over the field, a flock rising away to the right

CROWS = [(1500, 980, 230), (2250, 900, 260), (2750, 1040, 300), (3250, 930, 240), (2000, 690, 175), (2420, 620, 165),
         (2780, 560, 155), (3080, 650, 170), (3400, 560, 150), (3650, 680, 140), (2550, 420, 120), (2900, 380, 110),
         (3250, 330, 100), (3550, 300, 90), (2700, 260, 80), (3100, 200, 75), (3400, 150, 70), (600, 660, 140),
         (880, 860, 190)]


def crow(x, y, span, r):
    """A crow in two or three quick strokes of a loaded brush: a wing either side, each pressed down
    at the body and flicked out to its tip, raised in a V or, now and then, bowed at the shoulder
    and dropped at the tip, the two never quite alike; and often a stab for the body.
    -> paths, half-widths"""
    lean = r.normal(0, 0.2)
    rot = np.array([[np.cos(lean), -np.sin(lean)], [np.sin(lean), np.cos(lean)]])
    lift, bowed = r.uniform(0.15, 0.55), r.random() < 0.35
    paths, ws = [], []
    for side in (-1, 1):
        reach, up = 0.5 * span * r.uniform(0.82, 1.05), lift * r.uniform(0.85, 1.15)
        mid = (0.42, -0.35 * r.uniform(0.85, 1.15)) if bowed else (0.45, -0.45 * up - r.uniform(0.03, 0.12))
        tip = (1.0, r.uniform(-0.05, 0.12)) if bowed else (1.0, -up + r.uniform(-0.06, 0.06))
        pts = np.array([(0.02, 0.03), mid, tip]) * [side * reach, reach]
        P = brush.path(pts @ rot.T + [x, y], 2.0)
        P = P[np.linspace(0, len(P) - 1, 6).astype(int)]
        paths.append(P)
        ws.append(max(3.0, 0.068 * span) * r.uniform(0.85, 1.15))
    if r.random() < 0.7:
        a = np.pi / 2 + lean + r.normal(0, 0.4)
        paths.append(np.array([x, y]) + np.outer(np.linspace(-0.05, 0.1, 4) * span, [np.cos(a), np.sin(a)]))
        ws.append(max(3.2, 0.085 * span))
    return paths, ws


def paint(seed=1890):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#d6c9a6", thread=3.3)        # the bare canvas has yellowed
    rgb, height = ground.color.copy(), ground.tooth * 0.3
    hz = horizon(r)
    depth = np.clip((yy - hz[None, :]) / (H - hz[None, :]), 0, 1)
    land = yy > hz[None, :]
    across, along, course, half = track(hz, r)
    road = (np.abs(across) < 1) & (along < 0.995)
    verge = ~road & (np.abs(across) < 1.32 + 0.12 * noise.field((H, W), 90, r)) & (along < 0.97)
    wheat = land & ~road
    sparing = 0.6 * noise.smoothstep(-0.6, -1.4, noise.field((H, W), 380, r)) * land     # where the brush went thinner
    grain = noise.field((H, W), 140, r)

    def lay(paths, w, load, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], **{"ends": (0.25, 0.1), "fray": 1.4, **kw})

    def strew(mask, sp, keep=None):
        P = dabs.scatter((H + 2 * int(sp), W + 2 * int(sp)), sp, r) - int(sp)
        P = P[mask[at(P)]]
        odds = (1 if keep is None else keep[at(P)]) * (1 - 0.5 * sparing[at(P)])
        return P[r.random(len(P)) < odds]

    def passage(P, field, L, w, k, colours, tone, accent=None, odds=0.2, n=8, bend=0.0, fan=0.0, tilt=0.0,
                back=0.5, within=None, spread=0.06, **kw):
        """One passage of the picture, in sheaves of strokes whose colours follow `tone` (0 dark, 1 light)."""
        if len(P):
            paths, ws, s = sheaves(field, P, L, w, k, r, n, bend, fan, tilt)
            if within is not None:
                paths = trim(paths, within)
            flip = r.random(len(paths)) < back
            paths[flip] = paths[flip, ::-1]
            odds = np.asarray(odds)[s] if np.ndim(odds) else odds
            lay(paths, ws, loads(colours, (tone + r.normal(0, 0.08, len(P)))[s], r, accent, odds, spread), **kw)

    def flicks(mask, cover, k, wide=1.0, long_=1.0):
        """Seeds for the wheat's strokes, closer together far off where the strokes are short and level.
        -> seeds, depth, half-widths, lengths"""
        ds = np.linspace(0, 1, 64)
        hw_, L_ = (7.5 + 7 * ds) * wide, (40 + 120 * ds ** 1.1) * long_
        sp = np.array([spacing(k, l, w, cover) for l, w in zip(L_, hw_)])
        P = strew(mask, sp[0], np.interp(depth, ds, (sp[0] / sp) ** 2).astype(np.float32))
        d = depth[at(P)]
        n = len(P)
        return P, d, np.interp(d, ds, hw_) * r.uniform(0.85, 1.15, n), np.interp(d, ds, L_) * r.uniform(0.8, 1.25, n)

    # the lay-in, rubbed on thin: blue over the sky, ochre over the field, red-brown along the track,
    # the weave showing through and here and there the bare canvas
    under = np.where(land[..., None], WHEAT[1], SKY[5])
    under = np.where(road[..., None], EARTH[3], under)
    a = (0.97 - 0.6 * noise.smoothstep(-1.8, -2.4, noise.field((H, W), 40, r)) * land)[..., None]
    rgb = rgb * (1 - a) + under * (0.7 + 0.6 * ground.tooth[..., None]) * a

    # ---- the sky: broad strokes turning with the wind, near black at the top, paler down to the land
    flow, m = sky_field(r)
    vary = noise.field((H, W), 450, r)
    sky = yy < hz[None, :] + 30
    low = lambda P: np.clip(P[:, 1] / hz[at(P)[1]], 0, 1)
    P = strew(sky, spacing(3, 220, 16, 3.4))
    n = len(P)
    passage(P, flow, r.uniform(150, 300, n), r.uniform(13, 19, n), 3, SKY, 0.03 + 0.8 * low(P) ** 1.6 + 0.08 * vary[at(P)],
            CERULEAN[:1], 0.04, n=12, bend=r.normal(0, 0.0018, n), thick=0.28, spent=0.45, pickup=0.5, grooves=0.7)
    # streaks of darker blue up high, and of paler blue low down where the sky clears
    P = strew(sky & (yy < 0.7 * hz[None, :]), 190)
    n = len(P)
    passage(P, flow, r.uniform(150, 280, n), r.uniform(9, 13, n), 1, SKY[:4], np.full(n, 0.5), n=10,
            bend=r.normal(0, 0.002, n), thick=0.3, spent=0.5)
    P = strew(sky & (yy > 0.6 * hz[None, :]), 230)
    n = len(P)
    passage(P, flow, r.uniform(140, 260, n), r.uniform(12, 16, n), 1, SKY, 0.15 + 0.8 * low(P) ** 1.6 + 0.08 * vary[at(P)],
            CERULEAN, 0.3, n=10, bend=r.normal(0, 0.002, n), thick=0.3, spent=0.5)

    # the clouds, in the thickest white: first the body, greying toward the base, in short strokes
    # that close it up; then the heaped tops lit from the upper left, in long strokes winding through
    m = m + 0.08 * noise.fbm((H, W), 60, r, octaves=3)
    shape = cloud_field(m)
    cloud = m > 0.6
    lit_of = lambda P: puff(P[:, 0] + 50, P[:, 1] + 60) - puff(P[:, 0] - 50, P[:, 1] - 60)
    for cover, L, w, base, only in ((2.6, (60, 120), (13, 17), 0.15, -9.0), (1.2, (110, 200), (15, 20), 0.45, 0.0)):
        P = strew(cloud, spacing(2, np.mean(L), np.mean(w), cover))
        top = np.clip(m[at(P)] - 0.6, 0, 1) + 0.6 * np.tanh(lit_of(P))
        P, top = P[top > only], top[top > only]
        n = len(P)
        o = np.argsort(top)
        passage(P[o], shape, r.uniform(*L, n), r.uniform(*w, n), 2, CLOUD, (base + 0.4 * top)[o],
                np.concatenate([CREAM, CERULEAN[1:]]), 0.25, n=10, bend=r.normal(0, 0.0015, n), back=0.4,
                within=m > 0.5, thick=0.4, spent=0.4, pickup=0.45, land=0.7, grooves=0.6, ends=(0.45, 0.25),
                taper=0.3, tails=0.4)
    # the sky brushed back over their shadowed undersides, so that they sit in it
    P = strew((m > 0.4) & (m < 0.85), 60)
    P = P[(lit_of(P) < -0.05) & (r.random(len(P)) < 0.6)]
    n = len(P)
    passage(P, flow, r.uniform(100, 200, n), r.uniform(12, 16, n), 1, SKY, 0.03 + 0.8 * low(P) ** 1.6 + 0.08 * vary[at(P)],
            CERULEAN[1:], 0.2, n=10, bend=r.normal(0, 0.0018, n), thick=0.28, spent=0.5, pickup=0.55)

    # ---- the wheat: an underlayer of deep ochre and olive, then flicks of gold, rising and tossed
    # by the wind, short level dashes far off and longer flicks near, orange where the ears ripen most
    lie = wheat_field(depth, r)
    warm = noise.field((H, W), 600, r)
    rows = noise.stretched((W, H), 26, 520, r).T[:H, :W]           # far off, the wheat lies in bands
    P, d, w, L = flicks(wheat, 1.4, 1, 1.3, 0.9)
    n = len(P)
    which = (r.random(n) < 0.3 + 0.25 * np.tanh(warm[at(P)])).astype(int)
    load = mixed((np.concatenate([OLIVE[1:], WHEAT[:3]]), ORANGE[:2]), which, 0.4 + 0.15 * grain[at(P)], r)
    lay(impasto.follow(lie, P, L, 8, -r.normal(0.3, 0.3, n) / L, r.normal(0, 0.25, n)), w, load, thick=0.2, spent=0.6,
        taper=0.3, pickup=0.4, grooves=0.5)
    far = depth + 0.04 * grain < 0.17
    zone = noise.field((H, W), 900, r)
    for k, mask, cover in ((1, wheat & far, 1.8), (3, wheat & ~far, 2.5)):
        P, d, w, L = flicks(mask, cover, k)
        n = len(P)
        if k == 1:                                  # far off, level dashes bowed a little either way
            bend = r.normal(0, 0.5, n) / L
        else:                                       # rising, they curve upward
            bend = -r.normal(0.35, 0.35, n) / L * noise.smoothstep(0.1, 0.5, d)
        which = (r.random(n) < 0.15 + 0.3 * noise.smoothstep(0.2, 1.4, warm[at(P)])).astype(int)
        tone = 0.5 + 0.16 * grain[at(P)] + 0.1 * zone[at(P)] + 0.12 * (1 - d) + 0.12 * rows[at(P)] * (d < 0.4) \
            - 0.12 * noise.smoothstep(0.7, 1.0, d)
        paths, ws, s = sheaves(lie, P, L, w, k, r, 8, bend, 0.25, -0.5 * bend * L + r.normal(0, 0.3 * (k == 1), n))
        flip = r.random(len(paths)) < (0.4 if k == 1 else 0.12)
        paths[flip] = paths[flip, ::-1]
        load = mixed((WHEAT, ORANGE), which, tone, r, np.concatenate([RUST[2:], GREEN[3:4], OLIVE[2:]]), 0.12)
        lay(paths, ws, (load[0][s], load[1][s]), thick=0.36, spent=0.7, taper=0.3, tails=0.8, pickup=0.45, grooves=0.55,
            ends=(0.45, 0.3) if k == 1 else (0.25, 0.1))

    # the edge of the land broken: ears standing up into the sky here and there, and the sky brushed
    # down over the far wheat in places
    x0 = np.arange(-20, W + 20, 30.0)
    x0 = (x0 + r.uniform(-12, 12, len(x0)))[r.random(len(x0)) < 0.5]
    n = len(x0)
    P = np.stack([x0, hz[np.clip(x0.astype(int), 0, W - 1)] + r.uniform(-2, 16, n)], 1).astype(np.float32)
    lay(impasto.follow(np.full((1, 1), -np.pi / 2, np.float32), P, r.uniform(18, 42, n), 5, r.normal(0, 0.01, n),
                       0.35 + r.normal(0, 0.35, n)), r.uniform(4.5, 7, n), loads(WHEAT, r.uniform(0.4, 0.95, n), r, LEMON, 0.3),
        thick=0.34, spent=0.7, taper=0.5)
    P = strew((yy > hz[None, :] - 30) & (yy < hz[None, :] + 4), 80)
    n = len(P)
    lay(impasto.follow(np.zeros((1, 1), np.float32), P, r.uniform(50, 130, n), 6, r.normal(0, 0.002, n), r.normal(0, 0.06, n)),
        r.uniform(7, 11, n), loads(np.concatenate([SKY[9:], CERULEAN]), r.uniform(0.2, 0.8, n), r), thick=0.28, spent=0.6,
        pickup=0.5)

    # ---- the track: long strokes of red-brown dragged along it, darker in the ruts, in streaks worn
    # into it lengthways, and the green verges in short strokes laid slantwise, like a fishbone
    P = strew(road, spacing(2, 260, 11, 3.0))
    n = len(P)
    d, u = depth[at(P)], across[at(P)]
    stripes = np.interp(u, np.linspace(-1.2, 1.2, 240), noise.line1d(240, 30, r))
    tone = 0.52 + 0.14 * stripes - 0.14 * np.exp(-((np.abs(u) - 0.55) / 0.2) ** 2) + 0.12 * grain[at(P)]
    passage(P, course, (110 + 300 * d) * r.uniform(0.8, 1.2, n), np.clip(0.06 * half[at(P)], 6, 13) * r.uniform(0.7, 1.3, n),
            2, EARTH, tone, np.concatenate([ORANGE[1:3], OLIVE[2:], RUST]), 0.2, n=10, back=0.5, within=road | verge,
            thick=0.3, spent=0.55, pickup=0.5)
    P = strew(verge | road & (np.abs(across) > 0.88), spacing(1, 40, 7.5, 4.6))
    n = len(P)
    d, u = depth[at(P)], across[at(P)]
    passage(P, course, (20 + 35 * d) * r.uniform(0.8, 1.2, n), np.clip(0.035 * half[at(P)], 6, 9), 1, GREEN,
            0.45 + 0.2 * grain[at(P)] + 0.15 * (np.abs(u) < 1.12), np.concatenate([WHEAT[4:6], RUST[2:]]), 0.15, n=6,
            tilt=np.sign(u) * r.uniform(0.3, 0.9, n), back=0.3, thick=0.32, spent=0.6)

    # ---- the wheat again: red-brown and orange dashes in rows far off; pale flicks over all, some
    # leaning out over the verges; and dark lines among the stalks near at hand
    P = strew(wheat & ~verge & (depth < 0.45), 30, (noise.smoothstep(0.45, 0.0, depth) * (0.5 + 0.5 * (rows > 0.3))).astype(np.float32))
    n = len(P)
    d = depth[at(P)]
    lay(impasto.follow(lie, P, (32 + 60 * d) * r.uniform(0.7, 1.3, n), 5, r.normal(0, 0.003, n), r.normal(0, 0.2, n)),
        (7 + 3 * d) * r.uniform(0.8, 1.2, n), mixed((RUST, ORANGE), (r.random(n) < 0.35).astype(int), np.full(n, 0.5), r),
        thick=0.32, spent=0.5, pickup=0.35)
    P, d, w, L = flicks(land & ~road, 0.45, 2, 0.8, 0.75)
    n = len(P)
    bend = -r.normal(0.4, 0.35, n) / L * noise.smoothstep(0.1, 0.5, d)
    passage(P, lie, L, w, 2, LEMON, 0.4 + 0.3 * grain[at(P)] + 0.15 * (1 - d), np.concatenate([CREAM[:1], ORANGE[2:]]), 0.15,
            n=8, bend=bend, fan=0.25, tilt=-0.5 * bend * L, back=0.1, thick=0.38, spent=0.75, taper=0.35, tails=0.9,
            land=0.8, pickup=0.3)
    P = strew(wheat & ~verge & (depth > 0.35), 90, (noise.smoothstep(0.35, 0.8, depth) * 0.5).astype(np.float32))
    n = len(P)
    d = depth[at(P)]
    lay(impasto.follow(lie, P, (60 + 70 * d) * r.uniform(0.7, 1.3, n), 7, -r.normal(0.3, 0.2, n) / 100),
        r.uniform(3.5, 6, n), loads(np.concatenate([OLIVE[:3], GREEN[:2]]), np.full(n, 0.5), r), thick=0.24, spent=0.8,
        taper=0.6, pickup=0.3)

    # ---- the crows, last, in quick strokes of black from a full brush
    paths, ws = [], []
    for x, y, span in CROWS:
        p_, w_ = crow(x, y, span, r)
        paths += p_
        ws += w_
    n = len(paths)
    lay(paths, np.array(ws), loads(BLACK, r.uniform(0.2, 0.8, n), r, SKY[1:3], 0.2), thick=0.45, spent=0.5, taper=0.75,
        tails=0.3, land=0.7, pickup=0.12, ends=(0.3, 0.45), fray=0.8)

    height = ndimage.gaussian_filter(height, 0.7)      # the lamp sees the surface a hair softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.65, 1.25))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
