"""The Village under the Night Wind. Oil on canvas, in thick strokes from a loaded brush.

At Saint-Rémy in June 1889 Van Gogh painted the sky before dawn as he saw it from his window
and as he remembered it: the morning star, a waning moon, the hills, a village and a cypress.
He laid the sky in thick strokes that run with the wind and go round each star in rings,
drew the currents with lines of Prussian blue, built the cypress of rising flicks of
black-green and olive, and outlined the houses. Since then his chrome yellows have browned
a little and the canvas between the strokes has darkened.

This night is not his but is painted his way, on a canvas as his look now. The wind comes up
off the hills on the left and rolls over into one great eddy above the church, a crescent
moon hangs beside the cypress, and the village sleeps with a few windows lit.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Village under the Night Wind"
DATE = "2026"
MEDIUM = "Oil on canvas, in thick strokes from a loaded brush"
AFTER = "Vincent van Gogh, the night skies at Saint-Rémy, 1889–90: The Starry Night; Road with Cypress and Star"
ROOM = "Open Air"
YEAR = 1889
PLACE = "Saint-Rémy-de-Provence"
REGION = "Europe"
NOTE = ("The wind rolls over the village in currents, round the moon and every star in rings, and a cypress "
        "goes up into it like a flame. The yellows have browned a little, as his have.")

H, W = 2400, 3000          # a no. 30 canvas, 73 by 92 cm
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
NIGHT = pal("#121930", "#172140", "#1d2a53", "#243468", "#2c3f7b", "#344b8a", "#3e5996", "#4b689f",
            "#5c78a5", "#7089aa", "#879daf", "#a0b0b1", "#b9c3b5", "#d0d3bf")       # Prussian, ultramarine, cobalt
HALO = pal("#86a0a4", "#9bb0a6", "#afbfa3", "#c2cca7", "#d2d5ae", "#e0ddba", "#ebe6ca")  # viridian and white, yellowed
WHITE = pal("#dedcc6", "#e8e4cc", "#f0ecd6")
CHROME = pal("#86712a", "#9b832d", "#ae9535", "#c0a643", "#ccb556", "#d6c370")       # chrome yellow, browned
ORANGE = pal("#94592a", "#a76a2f", "#b97d38")
PRUSSIAN = pal("#0e1427", "#131a31", "#19233d")
HILLS = pal("#141b30", "#1b2544", "#23325c", "#2c3f74", "#374d8a", "#465e97", "#5d729f")
FIELDS = pal("#11181b", "#172123", "#1e2b2b", "#263634", "#30423d", "#3d5046", "#4f6050")  # blue-green, the valley at night
CYPRESS = pal("#0b0f0e", "#101614", "#151d19", "#1b2620", "#223127", "#2c3c2e")       # black-green
OLIVE = pal("#363f27", "#47502e", "#5b6237", "#717443")
UMBER = pal("#251e18", "#352a21", "#473828")
WALLS = pal("#3a4459", "#4c5870", "#606d86", "#778399", "#8f99a7", "#a7aeb1")       # walls in the moonlight
WARM_WALLS = pal("#66634f", "#7c7860", "#938d70")
ROOFS = pal("#131929", "#1c2335", "#272e43", "#373d52")
TILES = pal("#45302d", "#583b33", "#6b4839")                                          # a red roof, darkened
WINDOW = pal("#b3822f", "#c49637", "#d2aa47", "#dbbb5e")

MOON = (2030, 560, 92, 275)                           # x, y, radius of the disc, of its halo
# x, y, core, halo, warmth, how soon the halo goes blue, how it whirls
STARS = [(440, 410, 34, 215, 0.8, 1.0, 0.12), (1180, 190, 20, 125, 0.3, 1.4, 0.22), (290, 1010, 15, 100, 0.5, 1.6, 0.12),
         (1820, 1110, 19, 118, 0.4, 1.2, 0.1), (2860, 420, 14, 95, 0.6, 1.5, 0.15), (880, 790, 13, 85, 0.2, 1.8, 0.1),
         (2800, 1060, 16, 105, 0.4, 1.3, 0.18), (1660, 150, 12, 78, 0.3, 1.7, 0.1)]
EDDY = (1280, 560, 240)                              # the great eddy the current rolls up into
CYP = (2460, 2480, 70)                               # the cypress: x of its axis at the foot, the foot, the tip
LICKS = [(-1, 1650, 620, 0.5), (-1, 930, 520, 0.45), (-1, 420, 260, 0.32),
         (1, 1930, 650, 0.42), (1, 1270, 540, 0.5), (1, 690, 380, 0.38)]   # side, tip, reach, how far out
CHURCH = (1000, 1800)


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a paint near it on the palette streaked
    in, and a third, now and then an accent picked up from elsewhere."""
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


def sheaves(field, seeds, length, width, k, r, n=8, bend=0.0, tilt=0.0):
    """Strokes laid side by side in little sheaves: from each seed, k strokes about a brush-width
    apart following the same field, their starts staggered. -> paths, half-widths, seed of each"""
    N = len(seeds)
    a = field[at(seeds)] + tilt
    d = np.stack([np.cos(a), np.sin(a)], -1)[:, None, :]
    across = np.stack([-np.sin(a), np.cos(a)], -1)[:, None, :]
    j = np.arange(k) - (k - 1) / 2
    L = np.asarray(length)[:, None] * np.exp(r.normal(0, 0.3, (N, k)))
    w = np.asarray(width)[:, None] * np.exp(r.normal(0, 0.2, (N, k)))
    P = seeds[:, None, :] + across * (1.8 * w * (j + r.normal(0, 0.12, (N, k))))[..., None] \
        + d * (L * r.uniform(-0.2, 0.2, (N, k)))[..., None]
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None].repeat(k, 1)
    tl = np.broadcast_to(np.asarray(tilt, np.float32), (N,))[:, None].repeat(k, 1)
    return impasto.follow(field, P.reshape(-1, 2), L.ravel(), n, turn.ravel(), tl.ravel()), w.ravel(), np.repeat(np.arange(N), k)


def trim(paths, mask):
    """Cut each stroke where it first leaves the mask, so that none overshoots a silhouette."""
    n = paths.shape[1]
    ins = mask[np.clip(paths[..., 1].astype(int), 0, H - 1), np.clip(paths[..., 0].astype(int), 0, W - 1)]
    k = np.where(ins.all(1), n, ins.argmin(1))
    t = np.linspace(0, 1, n)[None, :] * np.maximum(k - 1, 0)[:, None]
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    row = np.arange(len(paths))[:, None]
    return paths[row, i] * (1 - f) + paths[row, np.minimum(i + 1, n - 1)] * f


def spacing(k, length, width, cover=2.9):
    """Seeds far enough apart that sheaves of k strokes cover the ground about `cover` times."""
    return float(np.sqrt(k * length * 2 * width / (cover * 0.866)))


def blend(a, b, t):
    """Mix two fields of stroke directions, which have no front or back, by `t` toward b."""
    return 0.5 * np.arctan2((1 - t) * np.sin(2 * a) + t * np.sin(2 * b), (1 - t) * np.cos(2 * a) + t * np.cos(2 * b))


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back over
    the last, and lift off now and then (keep < 1). -> index arrays into path"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def rings(cx, cy, r0, r1, w0, w1, r, whirl=0.08, gap=1.8):
    """Strokes laid round and round a light from radius r0 out to r1, each ring about a brush-width
    beyond the last and no ring quite round, every stroke drawn a little outward as it goes, so
    that the rings whirl. -> paths (N, 8, 2), half-widths, how far out each lies (0 at r0, 1 at r1)"""
    paths, ws, fs = [], [], []
    rad = r0
    k = r.uniform(0, 2 * np.pi, 2)
    while rad < r1:
        f = (rad - r0) / max(r1 - r0, 1)
        w = w0 + (w1 - w0) * f
        th = r.uniform(0, 2 * np.pi)
        end = th + 2 * np.pi
        while th < end:
            span = np.clip(r.uniform(4, 15) * w / rad, 0.3, 2.0)
            t = np.linspace(th, th + span, 8)
            rr = rad * (1 + 0.06 * np.sin(2 * t + k[0]) + 0.03 * np.sin(3 * t + k[1])) + r.normal(0, 0.35 * w) \
                + (whirl + r.normal(0, 0.05)) * rad * (t - th)
            paths.append(np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1))
            ws.append(w * np.exp(r.normal(0, 0.2)))
            fs.append(f)
            th += span * r.uniform(0.6, 1.05)
        rad += gap * w * r.uniform(0.7, 1.25)
    return np.array(paths, np.float32), np.array(ws), np.array(fs)


# ---- the sky: a drift to the right, a current rising from the low left that rolls up into one
# great eddy, and the air wound round the moon and every star

def current_y(x):
    return np.interp(x, [-200, 300, 700, 1050, 1300, 1600, 2100, 3200], [1250, 1170, 1060, 930, 855, 860, 890, 900])


def wind(x, y):
    """How far into the current or the eddy a point lies: 1 on the line of the current, less across it."""
    ex, ey, R = EDDY
    band = np.exp(-((y - current_y(x)) / 170) ** 2) * noise.smoothstep(2500, 1900, x)
    return np.maximum(band, np.exp(-((x - ex) ** 2 + (y - ey) ** 2) / (2 * (1.15 * R) ** 2)))


def glow(x, y):
    """The light round the moon and the stars, which the sky's own strokes take up."""
    g = 0
    for sx, sy, _, h, *_ in [MOON, *STARS]:
        g = g + np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (1.25 * h) ** 2)
    return g


def flow(r):
    """The stream the sky's strokes follow (H,W) and its stream function on a 4 px grid."""
    s = 4
    yy, xx = np.mgrid[0:H + s:s, 0:W + s:s].astype(np.float32)
    psi = yy + 70 * noise.field(yy.shape, 140, r)
    psi += 2.6 * 160 * np.tanh((yy - current_y(xx)) / 160) * noise.smoothstep(2600, 1800, xx)
    ex, ey, R = EDDY
    q = (xx - ex) ** 2 + (yy - ey) ** 2
    psi -= 4.5 * R * np.exp(-q / (2 * R * R))
    for x, y, _, h, *_ in [MOON, *STARS]:
        sg = 0.55 * h
        psi += 6 * sg * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sg * sg))
    gy, gx = np.gradient(psi)
    a = np.arctan2(-gx, gy) + 0.3 * np.exp(-q / (2 * (1.2 * R) ** 2))     # in the eddy the strokes wind inward
    return ndimage.zoom(a.astype(np.float32), s, order=0)[:H, :W], psi


# ---- the land

def far_crest(r):
    xs = [0, 300, 650, 1000, 1350, 1700, 2050, 2400, 3000]
    ys = [1450, 1360, 1285, 1340, 1435, 1385, 1330, 1385, 1445]
    return np.interp(np.arange(W), xs, ys) + 9 * noise.line1d(W, 150, r) + 3 * noise.line1d(W, 30, r)


def mid_crest(r):
    xs = [0, 400, 800, 1200, 1600, 2000, 2400, 3000]
    ys = [1540, 1495, 1515, 1560, 1530, 1495, 1525, 1565]
    return np.interp(np.arange(W), xs, ys) + 7 * noise.line1d(W, 170, r)


def front_crest(r):
    xs = [0, 450, 900, 1400, 1800, 2200, 2600, 3000]
    ys = [2235, 2200, 2215, 2185, 2150, 2110, 2100, 2120]
    return np.interp(np.arange(W), xs, ys) + 10 * noise.line1d(W, 180, r)


def contour(crest, depth, yy, wave, r):
    """Strokes along the lie of the land: with the crest just under it, flattening further down, rolling."""
    slope = np.arctan(np.gradient(ndimage.gaussian_filter1d(crest, 30)))[None, :]
    return (slope * np.exp(-np.clip(yy - crest[None, :], 0, None) / depth) + wave * noise.field(yy.shape, 260, r)).astype(np.float32)


def village(r):
    """The houses in rows, back to front, crowding toward the church:
    (scale, [(x, base, width, height, roof, side, tilt, gable end on)])."""
    rows = []
    for y0, s, n in ((1575, 0.5, 18), (1640, 0.62, 18), (1712, 0.78, 16), (1800, 0.95, 14), (1895, 1.15, 11),
                     (1995, 1.35, 8), (2095, 1.55, 5)):
        row = []
        for x in np.sort(np.clip(np.concatenate([r.normal(CHURCH[0] + 60, 430, n - 2), r.uniform(150, 2150, 2)]), 120, 2180)):
            if abs(x - CHURCH[0]) < 160 * s and abs(y0 - CHURCH[1]) < 60:
                continue
            row.append((x, y0 + r.normal(0, 10 * s), s * r.uniform(70, 135), s * r.uniform(45, 72), s * r.uniform(20, 42),
                        s * r.uniform(18, 36), r.normal(0, 0.03), r.random() < 0.4))
        rows.append((s, row))
    return rows


def inside(poly, sp, r):
    """Seeds about sp apart inside a polygon [(x, y), ...]."""
    p = np.asarray(poly, np.float32)
    o = np.floor(p.min(0)) - 2
    w, h = (np.ceil(p.max(0) - o) + 3).astype(int)
    im = Image.new("L", (int(w), int(h)), 0)
    ImageDraw.Draw(im).polygon([tuple(q) for q in p - o], fill=1)
    m = np.asarray(im, bool)
    S = dabs.scatter((int(h), int(w)), sp, r)
    S = S[(S[:, 0] >= 0) & (S[:, 0] < w - 1) & (S[:, 1] >= 0) & (S[:, 1] < h - 1)]
    return S[m[S[:, 1].astype(int), S[:, 0].astype(int)]] + o


def edges(poly, r, step=4.0, wob=1.2):
    """The sides of a polygon as hand-drawn lines, a point every few px."""
    p = np.asarray(poly, np.float32)
    p = np.vstack([p, p[:1]])
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(3, int(np.hypot(*(b - a)) / step))
        d = (b - a) / (np.hypot(*(b - a)) + 1e-6)
        out.append(a + np.linspace(0, 1, n)[:, None] * (b - a) + np.array([-d[1], d[0]]) * wob * noise.line1d(n, 6, r)[:, None])
    return out


# ---- the cypress: one flame from a foot below the frame to a point near the top, its sides
# licking up and out in tongues and cutting back in above each

def cypress(r, x0=2000, x1=2950):
    """The cypress over columns x0..x1: (inside mask (H,W), stroke directions, place across it
    from -1 at its left edge to 1 at its right, its half-width in each row, the lines between
    its tongues, its two sides)"""
    xb, yb, yt = CYP
    y = np.arange(H, dtype=np.float32)
    t = np.clip((yb - y) / (yb - yt), 0, 1)
    axis = xb + 25 * t + 30 * np.sin(1.6 * np.pi * t) + 30 * t * np.sin(3 * np.pi * t) + 6 * noise.line1d(H, 200, r)
    hw = 255 * (1 - t) ** 0.85 * (0.72 + 0.28 * noise.smoothstep(0, 0.15, t)) + 1
    out = np.zeros((2, H), np.float32)
    for side, tip, reach, amp in LICKS:
        u = (y - tip) / reach                   # 0 at the tip of the lick, 1 where it starts, below
        out[(side + 1) // 2] += amp * np.where(u >= 0, np.clip(1 - u, 0, 1) ** 1.6, np.clip(1 + u / 0.32, 0, 1) ** 2)
    xl, xr = axis - hw * (1 + out[0]), axis + hw * (1 + out[1])
    yy, xx = np.mgrid[0:H, x0:x1].astype(np.float32)
    a_, l_, r_ = axis[:, None], xl[:, None], xr[:, None]
    rel = np.where(xx < a_, (xx - a_) / (a_ - l_), (xx - a_) / (r_ - a_))
    up = lambda x_: np.arctan2(-1, -np.gradient(ndimage.gaussian_filter1d(x_, 8)))[:, None]   # the way a line x(y) runs
    ang = blend(up(axis) - 0.25 * np.clip(rel, -1.2, 1.2), np.where(rel < 0, up(xl), up(xr)),
                noise.smoothstep(0.3, 0.95, np.abs(rel)))
    m = np.zeros((H, W), bool)
    m[:, x0:x1] = np.abs(rel) < 1 + 0.06 * noise.field(yy.shape, 50, r) + 0.03 * noise.field(yy.shape, 12, r)
    field, place = np.full((H, W), -np.pi / 2, np.float32), np.zeros((H, W), np.float32)
    field[:, x0:x1] = ang + 0.08 * noise.field(yy.shape, 300, r)
    place[:, x0:x1] = np.clip(rel, -1.5, 1.5)
    lines = []
    for c in (-0.6, -0.2, 0.25, 0.6):
        rows = np.arange(r.uniform(yt + 250, 1300), H, 6.0).astype(int)
        x_ = axis[rows] + c * np.where(c < 0, axis - xl, xr - axis)[rows] + 10 * noise.line1d(len(rows), 40, r)
        lines.append(np.stack([x_, rows], 1).astype(np.float32))
    rows = np.arange(yt + 15, H + 40, 5.0)
    i = np.clip(rows.astype(int), 0, H - 1)
    sides = [np.stack([axis[i] + c * (axis - xl)[i] if c < 0 else axis[i] + c * (xr - axis)[i], rows], 1).astype(np.float32)
             for c in (-0.88, 0.88)]
    return m, field, place, (xr - xl) / 2, lines, sides


def paint(seed=1889):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#cfc4a8", thread=3.3)
    rgb, height = ground.color.copy(), ground.tooth * 0.3

    def lay(paths, w, load, **kw):
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], **{"ends": (0.35, 0.25), "taper": 0.3, "fray": 1.5, **kw})

    def strew(mask, sp):
        P = dabs.scatter((H + 2 * int(sp), W + 2 * int(sp)), sp, r) - int(sp)
        return P[mask[at(P)]]

    def passage(P, field, L, w, k, colours, tone, accent=None, odds=0.2, n=8, bend=0.0, tilt=0.0, back=0.5,
                spread=0.06, within=None, **kw):
        """One passage of the picture, in sheaves of strokes whose colours follow `tone` (0 dark, 1 light),
        kept `within` a mask if given."""
        if len(P):
            paths, ws, s = sheaves(field, P, L, w, k, r, n, bend, tilt)
            if within is not None:
                paths = trim(paths, within)
            flip = r.random(len(paths)) < back
            paths[flip] = paths[flip, ::-1]
            odds = np.asarray(odds)[s] if np.ndim(odds) else odds
            lay(paths, ws, loads(colours, (tone + r.normal(0, 0.08, len(P)))[s], r, accent, odds, spread), **kw)

    def draw(lines, w, length=120, overlap=20, keep=0.8, colours=PRUSSIAN, **kw):
        """Lines drawn with the brush a stroke at a time, not always closed, in Prussian blue unless told otherwise."""
        paths = [line[m] for line in lines for m in pieces(line, length, overlap, r, keep)]
        if paths:
            n = len(paths)
            lay(paths, w * np.exp(r.normal(0, 0.25, n)), loads(colours, np.full(n, 0.5), r, spread=0.3),
                **{"thick": 0.3, "spent": 0.55, "lips": 0.1, "pickup": 0.4, **kw})

    def fill(mask, field, colours, tone, w=(8, 12), L=(50, 120)):
        """Go back over the places a passage left bare with a few short strokes."""
        P = strew(mask & (height < 0.6), 22)
        n = len(P)
        passage(P, field, r.uniform(*L, n), r.uniform(*w, n), 1, colours, tone(P), n=7, thick=0.26)

    def tree(cx, cy, rx, ry, s, tone=0.3):
        """A tree or a bush: dark strokes across its middle and round its edge, the top catching the moon."""
        h_, w_ = int(1.5 * ry) + 2, int(1.5 * rx) + 2
        ly, lx = np.mgrid[-h_:h_ + 1, -w_:w_ + 1].astype(np.float32)
        a = blend(np.float32(r.uniform(-1.3, -0.3)), np.arctan2(ly / ry ** 2, lx / rx ** 2) + np.pi / 2,
                  noise.smoothstep(0.4, 0.9, np.hypot(lx / rx, ly / ry))).astype(np.float32)
        t_ = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        lump = 1 + 0.1 * np.sin(3 * t_ + r.uniform(0, 6)) + 0.06 * np.sin(5 * t_ + r.uniform(0, 6))
        P = inside(np.stack([cx + rx * lump * np.cos(t_), cy + ry * lump * np.sin(t_)], 1), max(6.0, 14 * s), r)
        n = len(P)
        if n:
            o = np.array([cx - w_, cy - h_], np.float32)
            paths = impasto.follow(a, P - o, r.uniform(24, 50, n) * s + 8, 6, r.normal(0, 0.015, n)) + o
            up = np.clip((cy - P[:, 1]) / ry, -1, 1)
            lay(paths, np.maximum(4.0, r.uniform(5.5, 8, n) * s), loads(FIELDS, tone + 0.15 * up, r, HILLS[2:4], 0.3),
                thick=0.32, pickup=0.35)

    # the drawing in thin paint first: sky, land and cypress washed in, the weave showing through,
    # here and there a patch of bare canvas left
    far, front = far_crest(r), front_crest(r)
    cyp, flame, rel, half, edges_, sides = cypress(r)
    under = np.where((yy < far[None, :])[..., None], NIGHT[3] * (1 + 0.25 * noise.field((H, W), 300, r))[..., None],
                     np.where((yy < far[None, :] + 160)[..., None], HILLS[2], FIELDS[1]))
    under[cyp] = CYPRESS[1]
    a = 0.95 * noise.smoothstep(-2.2, -1.5, noise.field((H, W), 150, r))[..., None]
    rgb = rgb * (1 - a) + under * (0.7 + 0.6 * ground.tooth[..., None]) * a

    # ---- the sky, laid in along the wind; the current and the eddy paler, the air round the lights paler still
    field, psi = flow(r)
    flowlines = lambda P, k: np.sin(psi[(P[:, 1] / 4).astype(int).clip(0, psi.shape[0] - 1),
                                        (P[:, 0] / 4).astype(int).clip(0, psi.shape[1] - 1)] / k)
    vary, grain = noise.field((H, W), 420, r), noise.field((H, W), 160, r)
    sky = yy < far[None, :] + 40

    def sky_tone(P):
        x, y = P.T
        b, low = wind(x, y), noise.smoothstep(far[at(P)[1]] - 380, far[at(P)[1]] - 20, y)
        return 0.22 + 0.07 * vary[at(P)] + 0.2 * b + 0.2 * b * flowlines(P, 70) + 0.08 * flowlines(P, 40) \
            + 0.3 * glow(x, y) + 0.32 * low, b, low

    P = strew(sky, spacing(3, 210, 12.5, 2.3))
    tone, b, low = sky_tone(P)
    n = len(P)
    passage(P, field, (120 + 210 * b) * np.exp(0.3 * grain[at(P)]) * (1 - 0.35 * low), (10.5 + 5 * b) * r.uniform(0.8, 1.3, n),
            3, NIGHT, tone, HALO, 0.05 + 0.2 * b, n=12, bend=r.normal(0, 0.002, n), thick=0.28, spent=0.45, pickup=0.55)

    # the current again, in long strokes of paler paint, and its lines drawn in Prussian blue
    band = wind(xx, yy)
    P = strew(sky & (band > 0.5), spacing(2, 240, 11, 0.8))
    n = len(P)
    passage(P, field, r.uniform(180, 340, n), r.uniform(12, 17, n), 2, NIGHT, 0.6 + 0.28 * flowlines(P, 70),
            HALO, 0.3, n=14, thick=0.3, spent=0.5)
    fill(sky, field, NIGHT, lambda P: sky_tone(P)[0])
    ex, ey, R = EDDY
    seeds = [(x_, current_y(x_) + o) for x_ in range(-100, 2100, 300) for o in (-150, 150)]
    seeds += [(ex + q * np.cos(t), ey + q * np.sin(t)) for q, t in ((90, 0.5), (170, 2.6), (240, 4.4), (300, 1.2))]
    seeds += [(x_, y_) for x_, y_ in zip(r.uniform(0, 2300, 30), r.uniform(0, 1250, 30)) if glow(x_, y_) < 0.3]
    lines = impasto.follow(field, np.array(seeds, np.float32), r.uniform(500, 900, len(seeds)), 60)
    draw(list(lines), 4.0, length=150, overlap=-15, keep=0.55)

    # the moon: its halo in rings of pale green and white going out into the blue, the dim rest
    # of its disc, then the crescent in thick chrome yellow, run round with orange
    mx, my, MR, mh = MOON
    paths, ws, f = rings(mx, my, 1.1 * MR, mh, 8.5, 12.5, r, whirl=0.05)
    lay(paths, ws, loads(np.concatenate([NIGHT[6:10], HALO[1:], WHITE[:1]]), (1 - f) ** 1.3 + r.normal(0, 0.06, len(ws)), r,
                         WHITE, 0.2), thick=0.32, pickup=0.35)
    paths, ws, f = rings(mx, my, 0.15 * MR, MR, 7, 9, r, whirl=0.03, gap=1.6)
    lay(paths, ws, loads(np.concatenate([NIGHT[9:11], HALO[:3]]), np.full(len(ws), 0.5), r, HALO[3:5], 0.3), thick=0.3, pickup=0.3)
    th, d, R2 = -0.4, 0.5 * MR, 0.86 * MR          # the dark of the disc: a circle set off toward th
    arcs, rho = [], 0.38 * MR
    while rho < 1.03 * MR:
        a0 = np.arccos(np.clip((rho ** 2 + d ** 2 - R2 ** 2) / (2 * rho * d), -1, 1))
        t0, t1 = th + a0, th + 2 * np.pi - a0
        while t0 < t1 - 0.05:
            span = min(r.uniform(0.5, 0.9), t1 - t0)
            u = np.linspace(t0, t0 + span, 8)
            q = rho + r.normal(0, 2)
            arcs.append(np.stack([mx + q * np.cos(u), my + q * np.sin(u)], 1))
            t0 += span * r.uniform(0.7, 0.95)
        rho += r.uniform(10, 13)
    n = len(arcs)
    lay(np.array(arcs, np.float32), r.uniform(8, 10.5, n), loads(np.concatenate([CHROME[3:], WHITE[:1]]), r.uniform(0.2, 0.8, n),
                                                                r, ORANGE[2:], 0.3), thick=0.52, pickup=0.1)
    a0 = np.arccos(np.clip(((1.03 * MR) ** 2 + d ** 2 - R2 ** 2) / (2 * 1.03 * MR * d), -1, 1))
    u = np.linspace(th + a0, th + 2 * np.pi - a0, 90)
    b0 = np.arccos(np.clip((MR ** 2 - d ** 2 - R2 ** 2) / (2 * d * R2), -1, 1))
    v = np.linspace(th + b0, th + 2 * np.pi - b0, 70)
    draw([np.stack([mx + 1.03 * MR * np.cos(u), my + 1.03 * MR * np.sin(u)], 1),
          np.stack([mx + d * np.cos(th) + R2 * np.cos(v), my + d * np.sin(th) + R2 * np.sin(v)], 1)], 4.0, length=100,
         overlap=10, keep=0.95, colours=ORANGE[:2], pickup=0.2)

    # the stars: a core of chrome yellow, then rings of white and pale green going out into the blue
    ramp = np.concatenate([NIGHT[6:10], HALO, WHITE[:2]])
    for sx, sy, c, h, warm, soon, whirl in STARS:
        paths, ws, f = rings(sx, sy, 0.15 * c, 1.15 * c, 5, 7, r, whirl=whirl + 0.04, gap=1.5)
        lay(paths, ws, loads(CHROME, 0.3 + 0.6 * f, r, WHITE, 0.4 - 0.3 * warm), thick=0.42, pickup=0.2)
        paths, ws, f = rings(sx, sy, 1.15 * c, h, 7, 11.5, r, whirl=whirl)
        lay(paths, ws, loads(ramp, (1 - f) ** soon + r.normal(0, 0.06, len(ws)), r, CHROME[3:], warm * (f < 0.4)),
            thick=0.32, pickup=0.35)
    sx, sy, c, h, *_ = STARS[0]
    t = np.linspace(0, 2 * np.pi, 160)
    draw([np.stack([sx + 1.02 * h * np.cos(t), sy + 1.02 * h * np.sin(t)], 1)], 3.4, length=140, overlap=-20, keep=0.6)

    # ---- the hills, two ranges rolling one before the other, in strokes along their slopes,
    # lighter on the crests, each crest drawn
    mid = mid_crest(r)
    for top, bottom, colours, light, dark, cover in ((far, mid, np.concatenate([HILLS[2:], NIGHT[7:9]]), 0.9, 0.35, 2.4),
                                                    (mid, front, np.concatenate([FIELDS[:4], HILLS[2:5]]), 0.8, 0.1, 2.0)):
        P = strew((yy >= top[None, :] - 5) & (yy < bottom[None, :] + 30), spacing(2, 130, 9.5, cover))
        n = len(P)
        depth = np.clip((P[:, 1] - top[at(P)[1]]) / 220, 0, 1)
        lie = contour(top, 140, yy, 0.12, r)
        passage(P, lie, r.uniform(80, 190, n) * (1 - 0.3 * depth), r.uniform(7.5, 11.5, n), 2,
                colours, light - (light - dark) * depth + 0.1 * grain[at(P)], NIGHT[5:8], 0.15, n=10, thick=0.27, pickup=0.5)
        fill((yy >= top[None, :]) & (yy < bottom[None, :]), lie, colours,
             lambda P: light - (light - dark) * np.clip((P[:, 1] - top[at(P)[1]]) / 220, 0, 1))
        x = np.arange(0, 2300)
        draw([np.stack([x, top[x] + 3], 1)], 4.5, keep=0.65)

    # the village, row by row from the back, trees among the houses: walls in short upright strokes,
    # roofs along their slope, every house outlined and the windows lit; the church in the middle
    def block(polys, s, lines_w, longer=0.0):
        """Fill each (polygon, stroke direction, palette, tone) with strokes, then outline them all."""
        paths, ws, cols, shares = [], [], [], []
        for poly, a, colours, t in polys:
            P = inside(poly, max(5.5, 14 * s), r)
            n = len(P)
            if n:
                paths.append(impasto.follow(np.full((1, 1), a, np.float32), P, r.uniform(18, 40, n) * s + 6 + longer, 5,
                                            r.normal(0, 0.01, n), r.normal(0, 0.12, n)))
                ws.append(np.maximum(3.2, r.uniform(4, 6.5, n) * s))
                c, sh = loads(colours, np.full(n, t) + r.normal(0, 0.05, n), r, NIGHT[4:7], 0.2)
                cols.append(c), shares.append(sh)
        if paths:
            lay(np.concatenate(paths), np.concatenate(ws), (np.concatenate(cols), np.concatenate(shares)), thick=0.3, pickup=0.35)
        draw([l_ for poly, *_ in polys for l_ in edges(poly, r)], lines_w, length=80 * s + 25, overlap=6, keep=0.85)

    for k, (s, row) in enumerate(village(r)):
        for _ in range(int(2 + 1.5 * s)):
            tx = np.clip(r.normal(CHURCH[0], 600), 100, 2150)
            tree(tx, row[0][1] - 30 * s if row else 1600, r.uniform(45, 90) * s, r.uniform(35, 70) * s, s)
        polys, wins = [], []
        for hx, hb, hw_, hh, rh, side, tl, gable in row:
            a0, a1 = np.array([hx, hb]), np.array([hx + hw_, hb + tl * hw_])
            top0, top1 = a0 - [0, hh], a1 - [0, hh]
            apex = (top0 + top1) / 2 - [0, rh + 8 * s]
            polys.append(([a0, a1, top1, top0], -np.pi / 2, WALLS if r.random() < 0.8 else WARM_WALLS, 0.12 + 0.75 * r.random()))
            if gable:
                sd = np.array([side, -0.4 * side])
                polys.append(([a1, a1 + sd, top1 + sd, top1], -np.pi / 2, WALLS, 0.15))
                polys.append(([top1, top1 + sd, apex + sd, apex], np.arctan2(sd[1], sd[0]), ROOFS, 0.5))
                roof = [top0 - [4 * s, 0], top1 + [4 * s, 0], apex]
            else:
                roof = [top0 - [6 * s, 0], top1 + [6 * s, 0], top1 + [-0.18 * hw_, -rh], top0 + [0.18 * hw_, -rh]]
            polys.append((roof, np.arctan2(*(top1 - top0)[::-1]), TILES if r.random() < 0.06 else ROOFS, 0.5))
            for _ in range(r.integers(0, 3) + 1):
                c = a0 + r.uniform(0.18, 0.82) * (a1 - a0) - [0, r.uniform(0.35, 0.6) * hh]
                ln = r.uniform(0.12, 0.2) * hh
                wins.append((np.array([c - [0, ln], c + [0, ln]]), r.random() < 0.7))
        block(polys, s, max(2.2, 3.2 * s))
        if wins:
            n = len(wins)
            lit = np.array([w_[1] for w_ in wins])
            c, sh = loads(WINDOW, np.full(n, 0.6), r)
            c[~lit] = loads(ROOFS, np.full((~lit).sum(), 0.6), r)[0]
            lay([w_[0] for w_ in wins], np.maximum(3.0, r.uniform(3.5, 5.5, n) * s), (c, sh), thick=0.45, pickup=0.1, spent=0.3)
        if k == 3:
            cx, cb = CHURCH
            block([([(cx - 115, cb), (cx + 110, cb + 4), (cx + 110, cb - 92), (cx - 115, cb - 96)], -np.pi / 2, WALLS, 0.7),
                   ([(cx - 125, cb - 96), (cx + 118, cb - 92), (cx + 85, cb - 140), (cx - 92, cb - 144)], 0.0, ROOFS, 0.4),
                   ([(cx + 70, cb + 3), (cx + 128, cb + 4), (cx + 126, cb - 200), (cx + 72, cb - 202)], -np.pi / 2, WALLS, 0.8)],
                  1.0, 3.4, longer=10)
            block([([(cx + 68, cb - 200), (cx + 100, cb - 198), (cx + 99, cb - 600)], -np.pi / 2, WALLS, 0.9),
                   ([(cx + 100, cb - 198), (cx + 130, cb - 198), (cx + 99, cb - 600)], -np.pi / 2, WALLS, 0.3)], 0.7, 3.0)

    # the near rise in front of the village, dark, in rolling strokes, bushes along its crest
    P = strew(yy >= front[None, :] - 5, spacing(2, 130, 10, 2.8))
    n = len(P)
    passage(P, contour(front, 200, yy, 0.2, r), r.uniform(70, 180, n), r.uniform(7, 10.5, n), 2, FIELDS,
            0.45 - 0.35 * np.clip((P[:, 1] - front[at(P)[1]]) / 300, 0, 1) + 0.12 * grain[at(P)],
            np.concatenate([OLIVE[:2], HILLS[2:4]]), 0.3, n=9, thick=0.3)
    for bx in np.arange(-40, 2300, 115) + r.uniform(-30, 30, 21):
        tree(bx, front[int(np.clip(bx, 0, W - 1))] + 30, r.uniform(90, 150), r.uniform(50, 85), 1.3, 0.22)

    # ---- the cypress, in rising strokes of black-green, lighter on the side of the moon, the lines
    # between its tongues drawn dark, its sides laid along the flame, then a few flicks of olive
    P = strew(cyp, spacing(2, 220, 13, 3.4))
    n = len(P)
    up = np.clip((2480 - P[:, 1]) / 2400, 0, 1)
    passage(P, flame, r.uniform(150, 340, n) * (1 - 0.55 * up), np.minimum(r.uniform(11, 15, n), 0.3 * half[at(P)[0]] + 3), 2,
            CYPRESS, 0.3 - 0.25 * rel[at(P)] + 0.1 * grain[at(P)] + 0.15 * np.sin(6 * rel[at(P)] + 2 * grain[at(P)]),
            np.concatenate([OLIVE[:2], UMBER, PRUSSIAN[1:]]),
            0.15 + 0.3 * (rel[at(P)] < -0.45),
            n=10, bend=r.normal(0, 0.002, n), back=0.12, thick=0.34, pickup=0.35, spent=0.5, within=cyp)
    draw(edges_, 3.6, length=200, overlap=30, keep=0.6, colours=CYPRESS[:2])
    draw(sides, 9.0, length=230, overlap=50, keep=1.0, colours=CYPRESS[:4], thick=0.34, spent=0.5, pickup=0.3)
    P = strew(cyp & (rel < 0.1), 90)
    n = len(P)
    passage(P, flame, r.uniform(60, 140, n), r.uniform(5, 7, n), 1, OLIVE[:3], np.full(n, 0.3), UMBER, 0.3, n=8,
            bend=r.normal(0, 0.004, n), back=0.1, thick=0.32, within=cyp)

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.65, 1.25))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
