"""Olive Trees below the Alpilles. Oil on canvas, in thick strokes from a loaded brush.

In May 1889 Van Gogh went of his own accord into the asylum of Saint-Paul-de-Mausole,
outside Saint-Rémy in Provence, and stayed a year. When he was let out he painted what lay
round the walls: olive groves, cypresses, wheat, the bare limestone of the Alpilles. He drew
with the loaded brush, laying stroke beside stroke along every form, two or three colours
on the brush at once, and went round things in Prussian blue. Since then his chrome yellows
have browned, his red lakes have faded almost to nothing and the bare canvas between the
strokes has yellowed. This is a canvas as they look now.
"""

import numpy as np
from scipy import ndimage, signal

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Olive Trees below the Alpilles"
DATE = "2026"
MEDIUM = "Oil on canvas, in thick strokes from a loaded brush"
AFTER = "Vincent van Gogh, the olive groves at Saint-Rémy, 1889"
ROOM = "Open Air"
YEAR = 1889
PLACE = "Saint-Rémy-de-Provence"
REGION = "Europe"
NOTE = ("Every stroke follows the thing it describes and carries two or three colours, unmixed. "
        "The yellows have browned and the pinks have faded, as they have on his canvases.")

H, W = 2380, 3000
FOOT = 990            # where the hills meet the plain
GROVE = 1150          # where the plain gives way to the floor of the grove
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# earth colours and deep ones, as the canvases show them now; each list from dark to light
SKY = pal("#46659a", "#5a78a6", "#7090b4", "#8aa5c0", "#a6bccb", "#c1cfd0")       # cobalt, ultramarine
HORIZON = pal("#8fadb8", "#a9c0c2", "#c4d0c8", "#dcdcc6")                          # cobalt with viridian, white
CLOUD = pal("#8796ad", "#a2afbd", "#bec5c3", "#d7d5c0", "#e8e1c5", "#f2eacf")      # whites touched with yellow
HILLS = pal("#2c365e", "#3d4874", "#525b88", "#6a6f9a", "#8584ac", "#a29fbd")      # violet and ultramarine
FIELDS = pal("#4b6140", "#5e7347", "#73844f", "#8a9358")
STUBBLE = pal("#a68b4b", "#b99d58", "#c9ae67", "#d5bd7b")
OCHRE = pal("#97713d", "#ad8448", "#c09654", "#ceaa66", "#d8bb7c")                # yellow ochre, raw sienna
ORANGE = pal("#a3632f", "#b6753a", "#c68a4b", "#d19d5e")
GREENS = pal("#4e6b48", "#5f7c4e", "#738c55", "#88995d")                         # viridian, emerald
SHADE = pal("#36427a", "#45528a", "#566197", "#6a71a3")                          # violet-blue shadow
LEAVES = pal("#344d48", "#46615a", "#5a7667", "#6f8a74", "#86997f", "#9eaa8c")
BARK = pal("#383e5c", "#4d4b64", "#665b55", "#7e6c53", "#957f57", "#a8946a")
CYPRESS = pal("#101a1b", "#152423", "#1c302b", "#264034", "#35523d", "#4a6644")
PRUSSIAN = pal("#16223a", "#1d2c44", "#273952")
OCHRE_LIGHT = pal("#c4a46a", "#d2b67c")
PINK = pal("#d3b09a")                            # a red lake, faded: rare
CHROME = pal("#c2a446", "#b3953c")               # chrome yellow, browned
CERULEAN = pal("#8db4b6")
SILVER = pal("#7690a0", "#9c9f55", "#26443e")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def either(paths, r, odds=0.5):
    """Some strokes are laid one way along the form, some the other."""
    back = r.random(len(paths)) < odds
    paths[back] = paths[back, ::-1]
    return paths


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.12):
    """The brush for each stroke: the paint for its tone, a paint near it on the palette
    streaked in, and a third, now and then an accent picked up from elsewhere."""
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


def mixed(families, which, tone, r, accent, odds):
    """Loads for strokes drawn from several palettes, `which` saying which one each takes."""
    cols, share = np.empty((len(which), 3, 3), np.float32), np.empty((len(which), 3))
    for i, colours in enumerate(families):
        sel = which == i
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds)
    return cols, share


def sheaves(field, seeds, length, width, k, r, n=8, bend=0.0, fan=0.0, tilt=0.0):
    """Strokes laid side by side in little sheaves, the way he hatched: from each seed, k
    strokes about a brush-width apart following the same field, their starts staggered and
    fanning out by `fan`. -> paths (N*k, n, 2), half-widths, the seed each came from"""
    N = len(seeds)
    a = field[at(seeds)] + tilt
    d = np.stack([np.cos(a), np.sin(a)], -1)[:, None, :]
    across = np.stack([-np.sin(a), np.cos(a)], -1)[:, None, :]
    j = np.arange(k) - (k - 1) / 2
    L = np.asarray(length)[:, None] * np.exp(r.normal(0, 0.3, (N, k)))
    w = np.asarray(width)[:, None] * np.exp(r.normal(0, 0.18, (N, k)))
    P = seeds[:, None, :] + across * (1.8 * w * (j + r.normal(0, 0.12, (N, k))))[..., None] \
        + d * (L * r.uniform(-0.15, 0.15, (N, k)))[..., None]
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None] + fan * j / L
    tl = np.broadcast_to(np.asarray(tilt, np.float32), (N,))[:, None].repeat(k, 1)
    return impasto.follow(field, P.reshape(-1, 2), L.ravel(), n, turn.ravel(), tl.ravel()), w.ravel(), np.repeat(np.arange(N), k)


def spacing(k, length, width, cover=2.9):
    """Seeds far enough apart that sheaves of k strokes cover the ground about `cover` times."""
    return float(np.sqrt(k * length * 2 * width / (cover * 0.866)))


def angle_of(vx, vy):
    return np.arctan2(vy, vx).astype(np.float32)


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def blobs(shapes, r, lump=0.2):
    """Lumpy ellipses (x, y, rx, ry): how far inside the nearest one each pixel lies (> 0 inside)."""
    out = np.full((H, W), -1, np.float32)
    for cx, cy, rx, ry in shapes:
        x0, x1 = int(max(0, cx - 1.5 * rx)), int(min(W, cx + 1.5 * rx))
        y0, y1 = int(max(0, cy - 1.5 * ry)), int(min(H, cy + 1.5 * ry))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        a = np.arctan2((yy - cy) / ry, (xx - cx) / rx)
        k = r.uniform(0, 2 * np.pi, 3)
        edge = 1 + lump * (0.6 * np.sin(2 * a + k[0]) + 0.3 * np.sin(3 * a + k[1]) + 0.2 * np.sin(5 * a + k[2]))
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], edge - np.hypot((xx - cx) / rx, (yy - cy) / ry))
    return out


def round_field(masses, pad=160, rim=(0.5, 0.95)):
    """How the brush models rounded things (x, y, rx, ry, across): round the edge of each mass,
    and one way, `across`, through its middle. -> (stroke directions (H,W), nearest mass (H,W))"""
    c = np.asarray(masses, np.float32)
    x0, y0 = int(max(0, (c[:, 0] - c[:, 2]).min() - pad)), int(max(0, (c[:, 1] - c[:, 3]).min() - pad))
    x1, y1 = int(min(W, (c[:, 0] + c[:, 2]).max() + pad)), int(min(H, (c[:, 1] + c[:, 3]).max() + pad))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.stack([np.hypot((xx - cx) / rx, (yy - cy) / ry) for cx, cy, rx, ry, _ in c])
    k = d.argmin(0)
    edge = np.arctan2((yy - c[k, 1]) / c[k, 3] ** 2, (xx - c[k, 0]) / c[k, 2] ** 2) + np.pi / 2
    ang, near = np.zeros((H, W), np.float32), np.zeros((H, W), np.int32)
    ang[y0:y1, x0:x1] = blend(c[k, 4], edge, noise.smoothstep(*rim, d.min(0)))
    near[y0:y1, x0:x1] = k
    return ang, near


# ---- the sky: a drift from the left, and the clouds turning over in it

LOBES = np.array([(820, 320, 140, 1.2), (1050, 240, 170, -0.6), (1300, 210, 160, 0.8), (1540, 280, 170, -1.3),
                  (1780, 360, 130, 0.6), (1150, 450, 130, -0.9), (1450, 470, 140, 0.5), (2250, 290, 120, 1.0),
                  (2480, 260, 130, -0.7), (2680, 330, 100, 0.9), (330, 230, 110, -0.8)], np.float32)


def puff(x, y):
    """How far into the clouds a point lies (above about 0.6 is cloud)."""
    m = 0
    for cx, cy, sg, _ in LOBES:
        m = m + np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (1.1 * sg) ** 2))
    return m


def sky_field(r):
    """The stream the sky's strokes follow: a slow drift to the right, wound round the lobes
    of cloud, each turning the other way from its neighbour."""
    s = 4
    yy, xx = np.mgrid[0:H + s:s, 0:W + s:s].astype(np.float32)
    psi = yy + 90 * noise.field(yy.shape, 150, r)
    for cx, cy, sg, spin in LOBES:
        psi += spin * 1.2 * sg * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg))
    gy, gx = np.gradient(psi)
    return ndimage.zoom(angle_of(gy, -gx), s, order=0)[:H, :W]


# ---- the Alpilles: a broken chain of limestone, ribbed by spurs and gullies

def ridge_line(r):
    """The crest: crags and saddles, falling away to the right."""
    xs = [0, 120, 260, 380, 470, 560, 650, 760, 880, 1000, 1110, 1220, 1310, 1390, 1460, 1540, 1620, 1700,
          1790, 1880, 1980, 2100, 2240, 2380, 2520, 2680, 2840, 3000]
    ys = [880, 860, 815, 790, 745, 770, 735, 790, 815, 770, 735, 760, 700, 655, 690, 640, 668, 720,
          705, 760, 800, 790, 830, 850, 845, 875, 885, 900]
    return np.interp(np.arange(W), xs, ys) + 4 * noise.line1d(W, 30, r) + 8 * noise.line1d(W, 140, r)


def ribs(crest, r):
    """Spurs running down from each crag and gullies from each saddle, as lines x(y) over the
    face of the hills. -> list of (x at the crest, x for every row, +1 for a spur or -1 for a gully)"""
    c = ndimage.gaussian_filter1d(crest, 25)
    y = np.arange(H)
    out = []
    for kind, idx in ((1, signal.argrelmin(c, order=60)[0]), (-1, signal.argrelmax(c, order=60)[0])):
        for x0 in idx:
            drift = r.uniform(0.25, 0.7) * r.choice([-1, 1])
            out.append((x0, (x0 + drift * np.clip(y - c[x0], 0, None) + 14 * noise.line1d(H, 90, r)).astype(np.float32), kind))
    return out


def hills_field(crest, lines, top):
    """Strokes in the hills, rows `top` to FOOT+40: down the spurs and gullies and along the
    folds between; and which way each face turns, toward the light on the left of a spur and
    the right of a gully. -> (directions, facing from +1 lit to -1 in shade)"""
    y = np.arange(top, FOOT + 40)
    d = np.stack([np.arange(W, dtype=np.float32)[None, :] - xs[y][:, None] for _, xs, _ in lines])
    k = np.abs(d).argmin(0)
    dist = np.take_along_axis(d, k[None], 0)[0]
    kind = np.array([s for _, _, s in lines], np.float32)[k]
    down = np.arctan2(1, np.stack([np.gradient(xs)[y] for _, xs, _ in lines])[k, np.arange(len(y))[:, None]])
    along = np.arctan(np.gradient(ndimage.gaussian_filter1d(crest, 20)))[None, :]
    t = noise.smoothstep(90, 25, np.abs(dist)) * noise.smoothstep(0, 40, y[:, None] - crest[None, :])
    return blend(along, down, t).astype(np.float32), -np.tanh(dist / 45) * kind


# ---- the trees: limbs as rows of (x, y, half-width) from the ground up; crowns as clumps (x, y, radius)

TREE_A = [[(640, 2000, 92), (668, 1830, 84), (612, 1660, 76), (650, 1480, 70)],
          [(650, 1480, 52), (548, 1330, 42), (452, 1180, 32), (372, 1040, 20)],
          [(650, 1480, 50), (720, 1310, 40), (792, 1160, 30), (900, 1030, 18)],
          [(650, 1480, 36), (640, 1330, 28), (612, 1180, 20), (650, 1030, 12)]]
CROWN_A = [(372, 1010, 165), (520, 880, 150), (700, 790, 175), (880, 905, 160), (1050, 1015, 130),
           (290, 1160, 112), (600, 1060, 140), (830, 1120, 125), (1010, 1190, 96), (455, 745, 110),
           (905, 700, 115), (1130, 880, 92), (200, 1005, 86), (660, 625, 100), (760, 985, 110)]
FORK_A = (650, 1480)
TREE_B = [[(1765, 1660, 60), (1790, 1540, 55), (1752, 1430, 50)],
          [(1752, 1430, 36), (1660, 1320, 28), (1590, 1210, 18)],
          [(1752, 1430, 34), (1840, 1310, 26), (1930, 1205, 16)]]
CROWN_B = [(1590, 1180, 105), (1730, 1090, 118), (1890, 1150, 110), (2020, 1250, 86), (1500, 1300, 78),
           (1680, 1260, 92), (1830, 1290, 88), (1640, 990, 76), (1820, 960, 80), (1980, 1060, 76)]
FORK_B = (1752, 1430)
FAR = [(1240, 1112, 34, 26), (1330, 1105, 26, 20), (2230, 1118, 36, 28), (2770, 1128, 40, 30),
       (2910, 1122, 30, 24), (120, 1100, 30, 24)]
CYP_X, CYP_BASE, CYP_TOP = 2525, 1395, 225
SHADOWS = [(1060, 2060, 640, 70), (980, 2160, 460, 40), (2060, 1690, 380, 42), (2000, 1750, 260, 24),
           (2790, 1432, 270, 24)]
BASES = [(640, 2000, 300), (1765, 1660, 190), (2525, 1400, 130)]      # where the ground turns round a trunk


def cypress_half_width(y):
    """The cypress, a flame: widest a quarter of the way up, drawn to a point."""
    t = np.clip((CYP_BASE - y) / (CYP_BASE - CYP_TOP), 0, 1)
    return 126 * (1 - t) ** 0.8 * (0.55 + 0.45 * noise.smoothstep(0, 0.22, t))


def cypress_axis(y):
    return CYP_X + 22 * np.sin((y - CYP_TOP) / 260.0)


def limb(rows, r, k, twist=0.8):
    """A limb as k ribbons of bark twisting round its axis. -> ribbons [(path, half-width, side)],
    side running from -1 on the lit left of the limb to 1 on its shaded right; and its two
    edges, for the outline."""
    A = brush.path(np.asarray(rows, float), 3.0)
    d = np.gradient(A[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n *= np.sign(n[:, :1] + 1e-9)
    t = np.linspace(0, 1, len(A))
    out = []
    for j in range(k):
        c = (2 * j + 1) / k - 1
        f = np.clip(0.78 * c + 0.2 * np.sin(2 * np.pi * twist * t + r.uniform(0, 2 * np.pi)), -0.92, 0.92)
        out.append((A[:, :2] + n * (A[:, 2] * f)[:, None], A[:, 2] * 1.3 / k, f))
    reach = np.stack([f * A[:, 2] + 1.3 / k * A[:, 2] * s for _, _, f in out for s in (-1, 1)])
    return out, A[:, :2] + n * reach.min(0)[:, None], A[:, :2] + n * reach.max(0)[:, None]


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back
    over the last (with a negative overlap, a little after it), and lift off now and then
    (keep < 1). -> index arrays into path"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def furrows(r, rhythm):
    """The floor of the grove in rows, as he ploughed it with the brush: dashes laid end to end
    along the lie of the land, the rows closer and the dashes smaller as they go back. Where
    `rhythm` (H,W) runs high the dashes turn short and stand across the row; where it runs
    low they draw out long. -> paths (N, 6, 2), half-widths, how near each lies (0..1)"""
    swell = noise.line1d(W + 600, 700, r)          # the long undulation of the land, shared by neighbouring rows
    paths, ws, nears = [], [], []
    y = GROVE - 14.0
    while y < H + 80:
        near = np.clip((y - FOOT) / (H - FOOT), 0, 1)
        w = 3.5 + 11 * near
        wob = noise.line1d(W + 600, 240, r)
        x = -300 + r.uniform(0, 80)
        while x < W + 150:
            beat = rhythm[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]
            L = (22 + 110 * near) * np.exp(r.normal(0, 0.25)) * (0.5 if beat > 0.55 else 1.7 if beat < -0.55 else 1)
            turn = r.choice([-0.6, 0.6]) if beat > 0.55 else r.normal(0, 0.06)
            xs = x + np.linspace(0, L * np.cos(turn), 6)
            i = np.clip((xs + 300).astype(int), 0, W + 599)
            ys = y + (10 + 45 * near) * swell[i] + 4 * near * wob[i] + 0.05 * near * (xs - W / 2) \
                + np.linspace(0, L * np.sin(turn), 6)
            paths.append(np.stack([xs, ys], 1))
            ws.append(w * np.exp(r.normal(0, 0.15)) * (1.15 if beat > 0.55 else 1))
            nears.append(near)
            x += L * np.cos(turn) * r.uniform(0.55, 0.95)
        y += 1.45 * w * r.uniform(0.85, 1.15)
    return np.array(paths, np.float32), np.array(ws), np.array(nears)


def arcs(r):
    """Round each trunk the strokes of the ground turn into arcs, as if the land swelled there."""
    paths, ws = [], []
    for x0, y0, R in BASES:
        for rho in np.arange(0.25 * R, R, 0.09 * R):
            t = np.linspace(r.uniform(0.15, 0.5), np.pi - r.uniform(0.15, 0.5), 80)
            line = np.stack([x0 + 1.3 * rho * np.cos(t), y0 + 0.32 * rho * np.sin(t) + 6], 1)
            for m in pieces(line, 40 + 0.35 * rho, -6, r, keep=0.8):
                paths.append(line[m][np.linspace(0, len(m) - 1, 6).astype(int)])
                ws.append(3.5 + 0.035 * rho)
    return np.array(paths, np.float32), np.array(ws)


def paint(seed=1889):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#e3d8bc", thread=3.3)
    rgb, height = ground.color.copy(), ground.tooth * 0.3
    sparing = noise.smoothstep(-0.5, -1.3, noise.field((H, W), 420, r))    # where the brush went thinner
    noisy = lambda scale, P: noise.field((H, W), scale, r)[at(P)]

    def strew(mask, spacing_):
        """Where the strokes of a passage start: a shaken honeycomb inside the mask, thinned
        here and there below the hills so the canvas shows."""
        P = dabs.scatter((H + 2 * spacing_, W + 2 * spacing_), spacing_, r) - spacing_   # run over the edges
        P = P[mask[at(P)]]
        return P[r.random(len(P)) > 0.5 * sparing[at(P)] * (P[:, 1] > FOOT)]

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], **kw)

    def passage(P, field, L, w, k, colours, tone, accent=None, odds=0.2, n=8, bend=0.0, fan=0.0, tilt=0.0,
                back=0.5, **kw):
        """One passage of the picture, in sheaves of strokes whose colours follow `tone` (0 dark, 1 light)."""
        if len(P):
            paths, ws, s = sheaves(field, P, L, w, k, r, n, bend, fan, tilt)
            tone = tone + r.normal(0, 0.08, len(P))
            odds = np.asarray(odds)[s] if np.ndim(odds) else odds
            lay(either(paths, r, back), ws, loads(colours, tone[s], r, accent, odds, spread=0.06), **kw)

    def draw(lines, w, length=130, overlap=25, keep=0.8):
        """Outlines in Prussian blue, drawn with the brush a stroke at a time, not always closed,
        and taking up some of the wet colour they cross."""
        paths = [line[m] for line in lines for m in pieces(line, length, overlap, r, keep)]
        n = len(paths)
        lay(paths, w * np.exp(r.normal(0, 0.25, n)), loads(PRUSSIAN, np.full(n, 0.5), r, spread=0.3),
            thick=0.3, spent=0.55, lips=0.1, pickup=0.5)

    crest = ridge_line(r)
    behind = yy < crest[None, :]

    # the sky, passage by passage: long sweeps at the upper left, short dabs to the right, level
    # strokes over the hills, and the clouds on top in large curling strokes of the thickest paint
    flow = sky_field(r)
    clear = ndimage.zoom(puff(xx[::8, ::8], yy[::8, ::8]), 8, order=1)[:H, :W] < 0.65
    vary = noise.field((H, W), 500, r)
    over = yy > crest[None, :] - 230                    # the band of sky just over the hills
    sweep = clear & ~over & ((xx + 0.8 * yy < 1700 + 300 * vary) | (yy < 130))
    short = clear & ~over & ~sweep & (xx > 1950 + 200 * vary)
    for mask, L, w, n_, cross in ((sweep, (200, 340), (12, 16), 14, 0), (clear & ~over & ~sweep & ~short, (80, 160),
                                  (10, 14), 12, 0), (short, (40, 80), (9, 13), 6, 0.5)):
        P = strew(mask, spacing(3, np.mean(L), np.mean(w)))
        n = len(P)
        passage(P, flow, r.uniform(*L, n), r.uniform(*w, n), 3, SKY,
                0.15 + 0.75 * np.clip(P[:, 1] / FOOT, 0, 1) ** 1.2 + 0.08 * noisy(300, P), CERULEAN, 0.25, n=n_,
                tilt=cross * r.choice([-1, 1], n) + r.normal(0, 0.15 * (cross > 0), n), thick=0.28)
    P = strew(over & (yy < crest[None, :] + 24), spacing(2, 220, 8.5))
    n = len(P)
    passage(P, np.full((H, W), 0.02, np.float32) + 0.06 * vary, r.uniform(150, 300, n), r.uniform(7, 10, n), 2,
            HORIZON, 0.3 + 0.6 * np.clip((P[:, 1] - crest[at(P)[1]] + 230) / 230, 0, 1), CLOUD[4:], 0.3, n=10,
            back=0.3, thick=0.26)
    P = strew(behind, spacing(3, 155, 15.5, 3.8))
    m = puff(P[:, 0], P[:, 1]) + 0.12 * noise.fbm((H, W), 120, r, octaves=3)[at(P)]
    P, m = P[m > 0.6], m[m > 0.6]
    n = len(P)
    lit = puff(P[:, 0] - 70, P[:, 1] - 90) - puff(P[:, 0] + 70, P[:, 1] + 90)
    passage(P, flow, r.uniform(100, 210, n), r.uniform(13, 18, n), 3, CLOUD,
            0.55 + 0.35 * np.tanh(2.2 * lit) + 0.12 * np.clip(m - 0.8, 0, 1), np.concatenate([CHROME[:1], SKY[3:4]]),
            0.2, n=14, bend=r.normal(0, 0.003, n), thick=0.34, spent=0.4)

    # the Alpilles: strokes down the spurs and gullies and along the folds, violet and blue, the
    # faces toward the light paler and touched with ochre; the crest drawn in broken lines
    lines = ribs(crest, r)
    top = int(crest.min()) - 20
    fold, facing = hills_field(crest, lines, top)
    field = np.zeros((H, W), np.float32)
    field[top:FOOT + 40] = fold
    P = strew(~behind & (yy < FOOT + 25), spacing(3, 100, 8.5, 3.8))
    n = len(P)
    face = facing[np.clip(P[:, 1].astype(int) - top, 0, FOOT + 39 - top), at(P)[1]]
    depth = np.clip((P[:, 1] - crest[at(P)[1]]) / (FOOT - crest[at(P)[1]]), 0, 1)
    passage(P, field, r.uniform(60, 160, n), r.uniform(6, 10, n), 3, HILLS,
            0.45 + 0.35 * face + 0.15 * noise.smoothstep(40, 0, P[:, 1] - crest[at(P)[1]]) + 0.1 * noisy(90, P),
            np.concatenate([OCHRE_LIGHT, FIELDS[2:3]]), 0.08 + 0.5 * (face > 0.4) + 0.2 * (depth > 0.8), n=10,
            thick=0.26, spent=0.5)
    x = np.arange(1150, 2420)
    draw([np.stack([x, crest[x] + 4], 1)], 5.5, keep=0.55)
    gullies = [np.stack([xs[y], y], 1) for x0, xs, _ in lines if 1150 < x0 < 2420
               for y in [np.arange(int(crest[x0]) + 30, FOOT - 40, 6)]]
    draw(gullies, 3.2, length=90, keep=0.4)

    # the plain at their foot: fields in level strokes, green and stubble, then hedges and far olives
    P = strew((yy >= FOOT - 10) & (yy < GROVE + 30), spacing(3, 60, 5.5))
    n = len(P)
    stripe = noise.stretched((W, H), 18, 700, r).T[:H, :W]
    s = stripe[at(P)]
    for sel, colours in ((s < 0.35, FIELDS), (s >= 0.35, STUBBLE)):
        passage(P[sel], (0.03 + 0.05 * stripe).astype(np.float32), r.uniform(30, 90, sel.sum()),
                r.uniform(4, 7, sel.sum()), 3, colours, 0.4 + 0.4 * noisy(120, P)[sel], CHROME, 0.25, n=6, thick=0.22)
    hedge = ((np.abs(stripe - 0.35) < 0.06) & (yy > FOOT + 30) & (yy < GROVE)) | (blobs(FAR, r, 0.3) > 0)
    P = strew(hedge, 9)
    n = len(P)
    lay(impasto.follow(np.full((H, W), -0.3, np.float32), P, r.uniform(14, 30, n), 5, r.normal(0, 0.04, n)),
        r.uniform(3.5, 5.5, n), loads(CYPRESS, np.full(n, 0.7), r, LEAVES[:2], 0.4), thick=0.28)

    # the floor of the grove, in rows and arcs: ochres and oranges, green dashes, violet-blue shadows
    shade = blobs(SHADOWS, r, 0.25) + 0.08 * noise.field((H, W), 40, r)
    lean = np.stack([noise.field((H, W), 380, r) for _ in range(2)])     # where orange and green come through
    for paths, ws in (furrows(r, noise.field((H, W), 300, r))[:2], arcs(r)):
        P = paths[:, 0]
        n = len(P)
        f = lean[:, at(P)[0], at(P)[1]]
        odds = np.clip(np.stack([np.ones(n), 0.45 + 0.45 * f[0], 0.35 + 0.55 * f[1]], 1), 0.03, None)
        which = (odds.cumsum(1) / odds.sum(1, keepdims=True) < r.random((n, 1))).sum(1)
        which[shade[at(P)] + 0.12 * r.normal(0, 1, n) > 0] = 3
        tone = 0.5 + 0.25 * noisy(220, P) + 0.1 * np.clip((P[:, 1] - FOOT) / (H - FOOT), 0, 1)
        lay(either(paths, r, 0.3), ws, mixed((OCHRE, ORANGE, GREENS, SHADE), which, tone, r,
                                            np.concatenate([CHROME, PINK, SHADE[2:3]]), 0.2), thick=0.26)

    # the cypress: flames of dark green licking upward, lighter tongues on the side of the light
    ax, hw = cypress_axis(yy), cypress_half_width(yy) + 1e-3
    rel = (xx - ax) / hw
    cyp = (np.abs(rel) < 1 + 0.15 * noise.field((H, W), 25, r)) & (yy > CYP_TOP + 45) & (yy < CYP_BASE + 30)
    flame = (-np.pi / 2 - 0.45 * np.clip(rel, -1.5, 1.5)
             + 0.45 * np.sin(yy / 95 + 2.6 * rel + 0.8 * noise.field((H, W), 200, r))).astype(np.float32)
    P = strew(cyp, spacing(2, 125, 9, 3.2))
    n = len(P)
    tip = 25 + 1.6 * hw[at(P)]                                             # no flame overshoots the top
    passage(P, flame, np.minimum(r.uniform(70, 170, n), tip), np.minimum(r.uniform(8, 12, n), 1 + 0.12 * tip), 2,
            CYPRESS, 0.36 - 0.28 * rel[at(P)] + 0.12 * noisy(70, P), np.concatenate([PRUSSIAN[1:2], pal("#2f5a45")]),
            0.25, n=9, bend=r.normal(0, 0.004, n), back=0.15, thick=0.32)
    P = strew(cyp & (rel < 0.1), 46)
    n = len(P)
    tip = 20 + 1.3 * hw[at(P)]
    passage(P, flame, np.minimum(r.uniform(50, 110, n), tip), np.minimum(r.uniform(6, 9, n), 1 + 0.12 * tip), 2,
            CYPRESS[3:], np.full(n, 0.6), CHROME[1:], 0.15, n=8, bend=r.normal(0, 0.005, n), back=0.1, thick=0.32)

    # the olives: trunks twisted into ribbons of bark and drawn round, then the crowns clump by clump
    for tree, crown, fork in ((TREE_A, CROWN_A, FORK_A), (TREE_B, CROWN_B, FORK_B)):
        lines = []
        for j, rows in enumerate(tree):
            ribbons, left, right = limb(rows, r, 4 if j == 0 else 3)
            for path, half, side in ribbons:
                cut = pieces(path, 190, 60, r)
                n = len(cut)
                sd = np.array([side[m].mean() for m in cut])
                lay([path[m] for m in cut], np.array([half[m].mean() for m in cut]) * r.uniform(0.85, 1.15, n),
                    loads(BARK, (0.6 if j == 0 else 0.42) - 0.4 * sd, r, pal("#4c5a8c", "#5b7058"), 0.25),
                    thick=0.22, spent=0.5, land=0)
            lines += [right, left]
        draw(lines, 6.5, keep=0.85)
        masses = [(x, y, cr * r.uniform(0.9, 1.25), cr * r.uniform(0.65, 0.9),
                   np.arctan2(y - fork[1], x - fork[0]) + r.uniform(-0.5, 0.5)) for x, y, cr in crown]
        ang, k = round_field(masses)
        P = strew(blobs([m[:4] for m in masses], r, 0.3) > 0, spacing(3, 62, 8.5, 2.8))
        n = len(P)
        c = np.asarray(masses, np.float32)[k[at(P)]]
        lit = -(0.6 * (P[:, 0] - c[:, 0]) / c[:, 2] + 0.8 * (P[:, 1] - c[:, 1]) / c[:, 3])
        low_ = (P[:, 1] - c[:, 1].min()) / np.ptp(c[:, 1])
        bias = r.normal(0, 0.12, len(masses))[k[at(P)]]          # some clumps greyer, some greener
        passage(P, ang, r.uniform(35, 75, n), r.uniform(6.5, 10.5, n), 3, LEAVES,
                0.5 + 0.3 * lit - 0.15 * low_ + bias + 0.08 * noisy(80, P), SILVER, 0.3, n=7,
                bend=r.normal(0, 0.012, n), fan=0.3, thick=0.32, spent=0.8)
        # the dark under each clump, in short strokes of Prussian blue and deep green
        P = P[(lit < -0.45) & (r.random(n) < 0.4)]
        n = len(P)
        lay(either(impasto.follow(ang, P, r.uniform(25, 55, n), 5, r.normal(0, 0.02, n)), r),
            r.uniform(4, 6.5, n), loads(np.concatenate([PRUSSIAN, LEAVES[:1]]), np.full(n, 0.5), r), thick=0.3)
    height = ndimage.gaussian_filter(height, 0.6)      # the lamp sees the surface a hair softer than the brush left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.65, 1.25))
    return img + 0.1 * impasto.glints(height, LIGHT)[..., None]
