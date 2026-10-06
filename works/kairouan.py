"""White Domes, Kairouan. Watercolour over pencil on paper, mounted on card.

In April 1914 Klee travelled to Tunisia with August Macke and Louis Moilliet, to Tunis, Hammamet and
Kairouan, and there, he wrote in his diary, colour took hold of him. The watercolours he made on the
journey and after it turn the white towns into a chequer of transparent colour, patch laid beside
patch, with the domes and arches drawn into it and a sun over it.

The chequer is ruled by eye in pencil: bands across the sheet whose edges tilt and drift, cut into
cells by uprights that wander from band to band. Each cell is one wash of a flat brush, laid in a few
strokes side by side on dry paper and left to dry with its own dark rim, the water running to its
lower edge; where two were laid while the other was still wet they ran into each other. Some cells
stop a hair short of their neighbours and some go over them, and where they overlap the colours
multiply. The heart of the sheet is warm and is gone over a second time; towards the edge the cells
grow larger, cooler and paler, and some are left as paper. The white domes are the paper itself.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import noise, pencil, plate, wash
from atelier import watercolour as wc
from atelier.color import lin, pigment
from atelier.paper import Sheet

TITLE = "White Domes, Kairouan"
DATE = "2026"
MEDIUM = "Watercolour over pencil on paper, mounted on card"
AFTER = ("Paul Klee, watercolours after the journey to Tunisia of April 1914: Red and White Domes; "
         "In the Style of Kairouan")
ROOM = "Paper and Water"
YEAR = 1914
PLACE = "Kairouan"
REGION = "Europe"
NOTE = ("A white town in the south as a loose chequer of transparent washes, warm at its heart and cooler "
        "towards the edge, with two domes left as bare paper.")

H, W = 2500, 2500                        # the plate
CARD = (110, 110, 2390, 2390)            # the mount, x0 y0 x1 y1
PX, PY, PW, PH = 290, 290, 1920, 1800    # the sheet glued on it
M = 26                                   # the sheet's arrays reach this far past its edge, for the deckle
CU, CV = 0.47, 0.55                      # the warm heart of the chequer, as a fraction of the sheet

PALETTE = dict(
    gamboge=pigment("#f2c437"), yellow_ochre=pigment("#dfae5e"), light_red=pigment("#dc6c48"),
    burnt_sienna=pigment("#b05a3a"), rose=pigment("#dd6f7c"), cerulean=pigment("#6fb4dc"),
    cobalt=pigment("#4f86dc"), ultramarine=pigment("#4a55cc"), viridian=pigment("#36a08c"),
    indigo=pigment("#3d4a66"), graphite=pigment("#6e6e74"))
GRAIN = dict(gamboge=0.04, yellow_ochre=0.25, light_red=0.2, burnt_sienna=0.3, rose=0.05, cerulean=0.35,
             cobalt=0.3, ultramarine=0.45, viridian=0.15, indigo=0.15, graphite=0.0)
WARM = dict(gold=dict(gamboge=0.75, yellow_ochre=0.25), ochre=dict(yellow_ochre=1.0),
            coral=dict(rose=0.45, light_red=0.35, gamboge=0.2), terracotta=dict(light_red=0.6, burnt_sienna=0.4),
            rose=dict(rose=1.0))
COOL = dict(green=dict(viridian=0.4, yellow_ochre=0.35, gamboge=0.25), turquoise=dict(cerulean=0.55, viridian=0.35, cobalt=0.1),
            blue=dict(cobalt=0.7, ultramarine=0.3), violet=dict(ultramarine=0.7, rose=0.3))
MIX = {**WARM, **COOL}
STRENGTH = dict(gold=0.62, ochre=0.75, coral=0.7, rose=0.56, terracotta=0.72, green=0.45, turquoise=0.42,
                blue=0.42, violet=0.45)
DEEPER = dict(gold="coral", ochre="coral", coral="rose", rose="coral", terracotta="rose", green="turquoise",
              turquoise="blue", blue="violet", violet="blue")
DARK = dict(indigo=0.35, rose=0.35, ultramarine=0.3)
SUN = dict(gamboge=0.6, light_red=0.25, rose=0.15)


def _s(r):
    return int(r.integers(1 << 31))


def bilinear(P, s, v):
    """Points at (s, v) in the quad P (TL, TR, BR, BL): s runs along the top, v down the side."""
    s, v = np.asarray(s, np.float64)[..., None], np.asarray(v, np.float64)[..., None]
    tl, tr, br, bl = P
    return (1 - v) * ((1 - s) * tl + s * tr) + v * ((1 - s) * bl + s * br)


def inside(q, x, y):
    e = np.roll(q, -1, 0) - q
    w = (x - q[:, 0]) * e[:, 1] - (y - q[:, 1]) * e[:, 0]
    return (w <= 0).all() or (w >= 0).all()


def strokes(r, P, flat=110.0):
    """How a hand fills the quad P with a flat brush about `flat` px wide: strokes side by side, mostly
    across, each set down and lifted a little before or past the edge, so the ends of the patch are not
    quite straight and its corners not quite square. -> [(polygon, load)]"""
    P = np.asarray(P, np.float64)
    if r.uniform() < 0.3:
        P = P[[3, 0, 1, 2]]                       # this one was laid in strokes going up
    L = 0.5 * (np.hypot(*(P[1] - P[0])) + np.hypot(*(P[2] - P[3]))) + 1
    D = 0.5 * (np.hypot(*(P[3] - P[0])) + np.hypot(*(P[2] - P[1]))) + 1
    k = max(1, int(round(D / (flat * r.uniform(0.8, 1.25)))))
    side = (P[3] - P[0] + P[2] - P[1]) / (2 * D)
    s, out = np.linspace(0, 1, 28), []
    for i in range(k):
        v0, v1 = i / k - (0.15 / k if i else 0), (i + 1) / k + (0.15 / k if i < k - 1 else 0)
        e = (r.normal(0, 2.5, 2) + (r.uniform(0, 1, 2) < 0.14) * r.uniform(5, 15, 2)
             - (r.uniform(0, 1, 2) < 0.1) * r.uniform(3, 9, 2)) / L
        ss = -e[0] + s * (1 + e.sum())
        top, bot = bilinear(P, ss, v0), bilinear(P, ss, v1)
        bow = 4 * s * (1 - s)
        top += (1 + 4 * (i == 0)) * (noise.line1d(28, 7, r) + 0.3 * noise.line1d(28, 2, r) + r.normal(0, 0.8) * bow)[:, None] * side
        bot += (1 + 4 * (i == k - 1)) * (noise.line1d(28, 7, r) + 0.3 * noise.line1d(28, 2, r) + r.normal(0, 0.8) * bow)[:, None] * side
        out.append((np.vstack([top, bot[::-1]]), r.uniform(0.85, 1.15)))
    return out


def around(polys, pad=40):
    pts = np.vstack(polys)
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    return slice(max(0, y0), min(H, y1)), slice(max(0, x0), min(W, x1))


def raster(polys, sl):
    """Polygons [(points, load)] in plate px drawn over the slices `sl`: (their union, the brush's load).
    Where one stroke went over the edge of the last a little more pigment was left."""
    ys, xs = sl
    size = (xs.stop - xs.start, ys.stop - ys.start)
    top, tot = np.zeros(size[::-1], np.float32), np.zeros(size[::-1], np.float32)
    for P, load in polys:
        im = Image.new("L", size, 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in np.asarray(P) - (xs.start, ys.start)], fill=255)
        a = np.asarray(im, np.float32) / 255
        top, tot = np.maximum(top, a * load), tot + a * load
    m = np.minimum(tot, 1)
    load = ndimage.gaussian_filter(top + 0.2 * (tot - top), 2.0) / (ndimage.gaussian_filter(m, 2.0) + 1e-3)
    return m, load


def broad(mask, k=7):
    """A shape as a brush lays it: no chink narrower than about `k` px, its corners round."""
    return noise.smoothstep(0.42, 0.58, ndimage.gaussian_filter(mask, k / 2.5))


def dome(r, cx, base, rad):
    """A dome sitting on the line y=base: a half round, a little full at its shoulders."""
    t = np.linspace(0, np.pi, 48)
    rr = rad * (1 + 0.02 * noise.line1d(48, 12, r))
    return np.stack([cx - rr * np.cos(t), base - rr * (0.97 * np.sin(t) ** 0.9 + 0.03 * np.sin(t) ** 8)], 1)


def shade(P, cx, base, rad):
    """The shaded side of a dome: its outline from the crown down the side away from the sun, and back
    up a flatter curve inside it."""
    t = np.linspace(np.pi, 0.5 * np.pi, 20)
    inner = np.stack([cx - 0.45 * rad * np.cos(t), base - 0.97 * rad * np.sin(t) ** 0.9], 1)
    return np.vstack([P[P[:, 0] > cx], inner])


def disc(r, cx, cy, rad, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = rad * (1 + 0.02 * noise.line1d(n, 10, r))
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1)


def arch(cx, foot, wd, ht):
    """An arched doorway standing on y=foot: upright sides and a round head."""
    t = np.linspace(0, np.pi, 30)
    head = np.stack([cx - wd / 2 * np.cos(t), foot - ht + wd / 2 - wd / 2 * np.sin(t)], 1)
    return np.vstack([[cx - wd / 2, foot], head, [cx + wd / 2, foot]])


def ribbon(P, w):
    """A line drawn with the point of a brush along the points P, `w` px wide, as a polygon."""
    g = np.gradient(P, axis=0)
    g /= np.hypot(*g.T)[:, None] + 1e-9
    n = np.stack([-g[:, 1], g[:, 0]], 1) * (np.broadcast_to(w, len(P)) / 2)[:, None]
    return np.vstack([P + n, (P - n)[::-1]])


def square(r, cx, cy, w, h):
    a = r.normal(0, 0.08)
    c, s = np.cos(a), np.sin(a)
    q = np.array([(-w, -h), (w, -h), (w, h), (-w, h)]) / 2 + r.normal(0, 0.06 * min(w, h), (4, 2))
    return q @ np.array([[c, s], [-s, c]]) + (cx, cy)


class Box:
    """The paint box: each wash kept as a density per pigment and glazed onto the paper at the end,
    the granulating pigments settling deeper into the hollows of the tooth than the staining ones."""

    def __init__(self, sheet, r):
        self.sheet, self.r = sheet, r
        self.d = {k: np.zeros((H, W), np.float32) for k in PALETTE}
        self.soak = np.zeros((H, W), np.float32)
        self.grain = 1.6 * (0.5 - ndimage.gaussian_filter(sheet.tooth, 1.2)) + 0.25 * noise.field((H, W), 3.0, r)

    def wash(self, mask, sl, ragged, **kw):
        sheet = Sheet(*(a[sl] for a in (self.sheet.color, self.sheet.tooth, self.sheet.fiber, self.sheet.alpha)))
        wet = wc.puddle(mask, sheet, self.r, ragged)
        self.soak[sl] = np.maximum(self.soak[sl], wet)
        return wet, wc.deposit(wet, sheet, self.r, grain=0.0, **kw)

    def lay(self, sl, dep, mix):
        for k, f in mix.items():
            self.d[k][sl] += f * dep * np.clip(1 + GRAIN[k] * self.grain[sl], 0.2, None)

    def glaze(self):
        img = self.sheet.color.copy()
        for k, kk in PALETTE.items():
            img *= np.exp(-self.d[k][..., None] * kk)
        return img


def feather(mask, r):
    """Where two wet washes meet, each runs into the other in soft fingers."""
    f = lambda: 14 * noise.field(mask.shape, 9, r) + 4 * noise.field(mask.shape, 2.5, r)
    return noise.warp(mask, f(), f())


def blossom(dep, wet, r):
    """A bloom: water run back into a drying wash pushes the pigment out into a pale round with a dark
    frilled edge, most often near the wash's edge, where the drying starts."""
    d = ndimage.distance_transform_edt(wet > 0.5)
    yy, xx = np.mgrid[0:wet.shape[0], 0:wet.shape[1]]
    ys, xs = np.nonzero((d > 12) & (d < 40))
    if not len(ys):
        return dep
    i = r.integers(len(ys))
    rad = r.uniform(40, 90) * (1 + 0.1 * noise.field(wet.shape, 12, r) + 0.12 * noise.field(wet.shape, 30, r))
    q = np.hypot(yy - ys[i], xx - xs[i]) - rad
    return dep * (1 - 0.3 * noise.smoothstep(1, -5, q)) + 0.6 * np.median(dep[wet > 0.5]) * np.exp(-(q / 2) ** 2) * wet


def patch(box, r, parts, white=None, ragged=1.6, rim=(0.3, 0.85), k=None, bloom=0.0):
    """One wash on dry paper. `parts` [(strokes, mix, density)] go on wet together, so where two touch
    they bleed into each other and dry with one edge round them both; `white` is paper kept clear.
    The water runs to the lower edge as it dries and leaves a bead there; now and then it blooms."""
    sl = around([p for P, _, _ in parts for p, _ in P])
    ms = [raster(P, sl) for P, _, _ in parts]
    k = k or r.uniform(7, 16)
    masks = [broad(m, k) for m, _ in ms]
    m = np.maximum.reduce(masks)
    if white is not None:
        m = m * (1 - white[sl])
    if m.max() < 0.5:
        return
    wet, dep = box.wash(m, sl, ragged, pool=0.28, rim=r.uniform(*rim), rim_width=r.uniform(2.2, 3.6), tides=0.15,
                        scale=r.uniform(60, 140))
    bead = ndimage.gaussian_filter(wet * (1 - np.roll(wet, -5, 0)), 1.5) * np.clip(noise.field(wet.shape, 40, r), 0, None)
    dep = dep + r.uniform(0, 0.25) * bead * wet
    if r.uniform() < bloom:
        dep = blossom(dep, wet, r)
    load = sum(l * q for (_, l), q in zip(ms, masks)) / (sum(masks) + 1e-3)
    shares = [1.0]
    if len(parts) > 1:
        shares = [wc.charge(feather(q, r), wet, 10, r, streak=0.3) for q in masks]
        tot = sum(shares) + 1e-3
        shares = [s / tot for s in shares]
    for (_, mix, dens), sh in zip(parts, shares):
        box.lay(sl, dep * load * sh * dens, mix)


def grid(r):
    """The chequer, ruled by eye: bands across the sheet whose edges tilt and drift, each cut into cells
    by uprights that wander from band to band, now and then left out so two cells run together or
    doubled so one splits; now and then a cell is cut across, or corner to corner. Cells grow larger
    towards the edge of the sheet. -> (cells, row(k, x) the y of the k-th line between bands)"""
    def cuts(n, lo, hi):
        t = (np.arange(n) + 0.5) / n
        w = r.uniform(0.75, 1.3, n) * (0.6 + 1.1 * np.abs(2 * t - 1) ** 1.5)
        return lo + (hi - lo) * np.concatenate([[0], np.cumsum(w) / w.sum()])
    Y, X = cuts(8, PY + 60, PY + PH - 70), cuts(7, PX + 60, PX + PW - 64)
    tilt = np.clip(r.normal(0, 0.025, 9), -0.04, 0.04)
    wave = 32 * noise.line1d(W, 500, r, rows=9)
    row = lambda k, x: Y[k] + tilt[k] * (x - PX - PW / 2) + wave[k][int(np.clip(x, 0, W - 1))]
    drift, cells, taken = np.zeros(8), [], {}
    for b in range(8):
        drift = np.clip(drift + r.normal(0, 16, 8), -42, 42)
        lean = r.normal(0, 8, 8)
        xt, xb = X + drift - lean / 2, X + drift + lean / 2
        keep = [0] + [j for j in range(1, 7) if r.uniform() > 0.17] + [7]
        for j0, j1 in zip(keep[:-1], keep[1:]):
            q = np.array([(xt[j0], row(b, xt[j0])), (xt[j1], row(b, xt[j1])),
                          (xb[j1], row(b + 1, xb[j1])), (xb[j0], row(b + 1, xb[j0]))])
            parts = [(q, (b == 0, j1 == 7, b == 7, j0 == 0))]
            near = np.hypot((q[:, 0].mean() - PX) / PW - CU, (q[:, 1].mean() - PY) / PH - CV) < 0.3
            if q[1, 0] - q[0, 0] > (200 if near else 300) and r.uniform() < (0.7 if near else 0.4):
                f = r.uniform(0.35, 0.65)
                a, c = bilinear(q, f, 0), bilinear(q, f + r.normal(0, 0.04), 1)
                o = parts[0][1]
                parts = [(np.array([q[0], a, c, q[3]]), (o[0], False, o[2], o[3])),
                         (np.array([a, q[1], q[2], c]), (o[0], o[1], o[2], False))]
            for p, o in parts:
                if min(p[3, 1] - p[0, 1], p[2, 1] - p[1, 1]) < 70:
                    continue                         # too thin a sliver to lay a brush in: left as paper
                if any(a < p[:, 0].mean() < z for a, z in taken.get(b, [])):
                    continue                         # under a tall cell come down from the band above
                if b < 7 and r.uniform() < 0.1:
                    p = p.copy()
                    p[2:, 1] = row(b + 2, p[2, 0]), row(b + 2, p[3, 0])
                    taken.setdefault(b + 1, []).append((p[0, 0] - 20, p[1, 0] + 20))
                    cells.append(dict(q0=p, outer=(o[0], o[1], b == 6, o[3]), band=b))
                elif p[3, 1] - p[0, 1] > 170 and r.uniform() < (0.45 if near else 0.15):
                    f = r.uniform(0.38, 0.62)
                    a, c = bilinear(p, 0, f), bilinear(p, 1, f + r.normal(0, 0.05))
                    cells += [dict(q0=np.array([p[0], p[1], c, a]), outer=(o[0], o[1], False, o[3]), band=b),
                              dict(q0=np.array([a, c, p[2], p[3]]), outer=(False, o[1], o[2], o[3]), band=b)]
                elif r.uniform() < 0.07:
                    cells += [dict(q0=np.array([p[0], p[1], p[2], p[2]]), outer=None, band=b),
                              dict(q0=np.array([p[0], p[2], p[3], p[3]]), outer=None, band=b)]
                else:
                    cells.append(dict(q0=p, outer=o, band=b))
    for i, c in enumerate(cells):
        c["i"] = i
        c["xy"] = c["q0"].mean(0)
        u, v = (c["xy"] - (PX, PY)) / (PW, PH)
        c["rho"] = np.hypot((u - CU) / 0.5, (v - CV) / 0.48)
        c["q"] = c["q0"] + r.normal(0, 1.5, (4, 2)) if c["outer"] is None else inset(r, c["q0"], c["outer"])
    return cells, row


def inset(r, q, outer):
    """Each edge of a cell drawn a little in from its line or a little over it: a hair of paper shows
    between some neighbours, others overlap; the outer edge of the chequer wanders more."""
    g = [r.uniform(-10, 60) if o else [r.uniform(-11, -3), r.uniform(1.5, 5), r.uniform(-1, 1)][r.choice(3, p=[0.4, 0.42, 0.18])]
         + r.normal(0, 6) for o in outer]
    t, rt, b, l = g
    return q + np.array([[l, t], [-rt, t], [-rt, -b], [l, -b]]) + r.normal(0, 3.5, (4, 2))


def seg(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.array([a, (a + b) / 2, b])


def town(r, cells, row):
    """What stands in the chequer: three whitewashed buildings, each a cube with its dome, two domes
    the bare paper and one washed coral, the washes of the sky round them; an arcade; a sun.
    -> (shapes kept white, washes [(polygons, mix, density)], the colours they fix for their cells,
    the outlines the pencil drew, the arches)"""
    at = lambda x, y: next((c for c in cells if inside(c["q0"], x, y)), None)
    fixed, keep, washes, lines, shades = {}, [], [], [], []
    for u, k, rad, sky, white in ((0.6, 3, 0.068, "blue", True), (0.33, 3, 0.052, "turquoise", False),
                                  (0.47, 4, 0.042, "violet", True)):
        x, rad = PX + u * PW, rad * PW
        base = row(k, x) - 2
        D = dome(r, x, base, rad)
        w = 1.3 * rad
        h = r.uniform(0.85, 1.1) * w
        C = np.array([(x - w, base - 1), (x + w, base - 1), (x + w + r.normal(0, 2), base + h),
                      (x - w + r.normal(0, 2), base + h)])
        keep += [D, C]
        lines += [D, seg(C[0], C[3]), seg(C[3], C[2]), seg(C[2], C[1])]
        up = at(x, base - 0.5 * rad)
        if up:
            fixed[up["i"]] = (sky, 0.55)
        if not white:
            washes.append(([(D, 1.0)], MIX["coral"], 0.8))
        elif rad > 0.06 * PW:
            shades.append((D, shade(D, x, base, rad)))
    sx, sy, sr = PX + 0.84 * PW, PY + 0.12 * PH, 0.04 * PW
    S = disc(r, sx, sy, sr)
    keep.append(S)
    washes.append(([(S, 1.0)], SUN, 0.62))
    if at(sx, sy):
        fixed[at(sx, sy)["i"]] = ("blue", 0.5)
    x, arches = PX + r.uniform(0.26, 0.32) * PW, []
    for _ in range(3):
        wd = r.uniform(150, 200)
        cx = x + wd / 2
        foot = row(7, cx) - r.uniform(0, 8)
        arches.append(arch(cx, foot, wd, r.uniform(1.15, 1.35) * (row(7, cx) - row(6, cx))))
        c = at(cx, foot - 30)
        if c:
            fixed[c["i"]] = ("gold", 0.55)                # a sunlit wall, so the doorways in it read dark
        x += wd + r.uniform(20, 60)
    return keep, washes, fixed, lines + arches, arches, shades


def colours(r, cells, fixed):
    """Which colour each cell gets: warm in the heart of the sheet and cooler towards its edge and up in
    the sky, in slow zones of related colour, never quite the same as the cell beside it; towards the
    edge some are left as paper, and now and then one in the town is left white for a wall in the sun."""
    temp, hue = noise.field((64, 64), 24, r), noise.field((64, 64), 16, r)
    for i, (name, d) in fixed.items():
        cells[i]["c"], cells[i]["d"] = name, d
    for c in cells:
        if "c" in c:
            continue
        u, v = (c["xy"] - (PX, PY)) / (PW, PH)
        f = lambda a: a[int(np.clip(v * 63, 0, 63)), int(np.clip(u * 63, 0, 63))]
        if r.uniform() < 0.05 + 0.3 * noise.smoothstep(0.8, 1.3, c["rho"]):
            c["c"] = None
            continue
        t = (1 - 0.75 * noise.smoothstep(0.25, 1.05, c["rho"]) - 0.3 * noise.smoothstep(0.3, 0.05, v)
             + 0.25 * noise.smoothstep(0.7, 0.95, v) + 0.2 * f(temp) + r.normal(0, 0.15))
        fam = list(WARM if t > 0.45 else COOL)
        k = int(np.clip((0.5 + 0.35 * f(hue) + r.normal(0, 0.15)) * len(fam), 0, len(fam) - 1))
        near = {o.get("c") for o in cells if o is not c and np.hypot(*(o["xy"] - c["xy"])) < 260}
        name = next((fam[(k + dk) % len(fam)] for dk in (0, 1, -1, 2, -2) if fam[(k + dk) % len(fam)] not in near),
                    fam[k])
        c["c"] = name
        c["d"] = (STRENGTH[name] * r.uniform(0.75, 1.2) * (1.15 - 0.7 * noise.smoothstep(0.3, 1.15, c["rho"]))
                  * (0.8 if name in COOL else 1.0))


def groups(r, cells, fixed):
    """Cells laid one at a time, but now and then two side by side laid while the first was wet."""
    out = []
    for c in cells:
        if not c["c"]:
            continue
        p = out[-1][-1] if out else None
        if (p and len(out[-1]) == 1 and p["i"] == c["i"] - 1 and p["band"] == c["band"] and p["c"] != c["c"]
                and c["i"] not in fixed and p["i"] not in fixed and r.uniform() < 0.25):
            out[-1].append(c)
        else:
            out.append([c])
    return out


def seconds(r, cells, arches):
    """The second sitting, when the first washes were dry, most of it in the heart of the sheet: part of a
    cell taken down a tone in a deeper colour of its family, now and then spilling over its edge; the
    arcade; a few triangles; and last the dark accents."""
    out, drawn = [], []
    lit = [c for c in cells if c["c"]]
    w = np.array([max(0.02, 1 - c["rho"]) ** 2 for c in lit])
    for c in (lit[i] for i in r.choice(len(lit), 18, replace=False, p=w / w.sum())):
        s0, s1 = r.uniform(-0.06, 0.3), r.uniform(0.7, 1.06)
        v0, v1 = r.uniform(0.3, 0.6), r.uniform(0.95, 1.12)
        if r.uniform() < 0.4:
            s0, s1, v0, v1 = v0 - 0.3, v1, s0 + 0.05, s1
        q = bilinear(c["q0"], [s0, s1, s1, s0], [v0, v0, v1, v1])
        out.append((strokes(r, q), MIX[DEEPER[c["c"]]], r.uniform(0.3, 0.5)))
    at = lambda x, y: next((c for c in cells if inside(c["q0"], x, y)), None)
    for P in arches:
        c = at(*P[len(P) // 2] + (0, 40))
        out.append(([(P, 1.0)], MIX[DEEPER[c["c"] if c and c["c"] else "ochre"]], 0.65))
        wd = r.uniform(4, 6) * (1 + 0.3 * noise.line1d(len(P), 6, r))
        drawn.append(([(ribbon(P, wd), 1.0)], DARK, 0.95))
    for c in (lit[i] for i in r.choice(len(lit), 3, replace=False, p=w / w.sum())):
        a = r.uniform(0.05, 0.35)
        b = a + r.uniform(0.4, 0.6)
        apex = bilinear(c["q0"], (a + b) / 2 + r.normal(0, 0.08), r.uniform(0.15, 0.45))
        q = np.array([apex, apex, bilinear(c["q0"], b, 0.98), bilinear(c["q0"], a, 0.98)])
        out.append((strokes(r, q), MIX[DEEPER[c["c"]]], r.uniform(0.35, 0.5)))
    dark, corners = [], np.vstack([c["q0"] for c in lit])
    u, v = ((corners - (PX, PY)) / (PW, PH)).T
    corners = corners[np.hypot((u - CU) / 0.5, (v - CV) / 0.48) < 0.75]
    for x, y in corners[r.choice(len(corners), 6, replace=False)] + r.normal(0, 7, (6, 2)):
        s = r.uniform(16, 28)
        dark.append(square(r, x, y, s, s * r.uniform(0.85, 1.15)))
    c = lit[r.choice(len(lit), p=w / w.sum())]
    dark.append(square(r, *bilinear(c["q0"], 0.5, 0.0), r.uniform(70, 110), r.uniform(14, 20)))
    return out, drawn + [([(P, 1.0)], DARK, r.uniform(1.1, 1.5)) for P in dark]


def drawing(r, sheet, cells, lines):
    """The pencil the chequer was ruled up with by eye, showing where the washes leave it: a few of the
    lines between cells, run on past the corners, and the outlines of the domes and arches."""
    press = np.zeros((H, W), np.float32)
    for i in r.choice(len(cells), 14, replace=False):
        q = cells[i]["q0"]
        a, b = (q[0], q[1]) if r.uniform() < 0.55 else (q[0], q[3])
        d = (b - a) / (np.hypot(*(b - a)) + 1e-9)
        a, b = a - d * r.uniform(-10, 30), b + d * r.uniform(-10, 30)
        pencil.line(press, np.array([a, (a + b) / 2 + r.normal(0, 1.5, 2), b]), 2.6, r, pressure=r.uniform(0.25, 0.4))
    for P in lines:
        pencil.line(press, P, 2.6, r, pressure=0.38)
    return pencil.catch(press, sheet, grip=1.7)


def paper(r):
    """A small sheet of rag paper with torn edges, glued down on the card."""
    s = wash.rough((PH + 2 * M, PW + 2 * M), _s(r), tint="#f6f1e4", margin=M, hill=6.0)

    def put(a, fill):
        out = np.full((H, W) + a.shape[2:], fill, np.float32)
        out[PY - M:PY + PH + M, PX - M:PX + PW + M] = a
        return out
    return Sheet(put(s.color, 1.0), put(s.tooth, 0.5), put(s.fiber, 0.0), put(s.alpha, 0.0))


def card(r):
    """The mount: buff card cut straight, a shade darker towards its edges."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x0, y0, x1, y1 = CARD
    d = np.minimum.reduce([xx - x0, x1 - xx, yy - y0, y1 - yy])
    tone = 1 + 0.012 * noise.fbm((H, W), 500, r, octaves=4) - 0.03 * np.exp(-np.clip(d, 0, None) / 60)
    z = np.zeros((H, W), np.float32)
    return Sheet((lin("#ddd1b9")[None, None] * tone[..., None]).astype(np.float32), z, z,
                 noise.smoothstep(-0.8, 0.8, d).astype(np.float32))


def paint(seed=3):
    r = noise.rng(seed)
    sheet, mount = paper(r), card(r)
    box = Box(sheet, r)
    cells, row = grid(r)
    keep, washes, fixed, lines, arches, shades = town(r, cells, row)
    colours(r, cells, fixed)
    full = (slice(0, H), slice(0, W))
    white = noise.smoothstep(0.1, 0.4, ndimage.gaussian_filter(raster([(P, 1.0) for P in keep], full)[0], 2.5))
    for g in groups(r, cells, fixed):
        patch(box, r, [(strokes(r, c["q"]), MIX[c["c"]], c["d"]) for c in g], white, bloom=0.05)
    glazes, dark = seconds(r, cells, arches)
    for parts in glazes:
        patch(box, r, [parts], white)
    for parts in washes:
        patch(box, r, [parts], ragged=1.2, k=5, rim=(0.6, 1.0))
    for D, S in shades:      # the shade on a white dome, its inner edge softened with a damp brush
        sl = around([D])
        box.lay(sl, 0.24 * ndimage.gaussian_filter(raster([(S, 1.0)], sl)[0], 12) * broad(raster([(D, 1.0)], sl)[0], 3),
                MIX["blue"])
    for parts in dark:
        patch(box, r, [parts], white, ragged=0.6, k=3, rim=(0.3, 0.5))
    box.lay(full, drawing(r, sheet, cells, lines), dict(graphite=1.0))
    img = wash.cockle(box.glaze(), sheet, box.soak, r)
    lifted = ndimage.gaussian_filter(np.roll(sheet.alpha, (3, 2), (0, 1)), 3) * 0.22
    a = sheet.alpha[..., None]
    return plate.mount(mount.color * (1 - lifted[..., None]) * (1 - a) + img * a, mount)
