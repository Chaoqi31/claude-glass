"""Exotic Garden with a Pale Sun. Oil on canvas, thinly painted.

Henri Rousseau never saw a jungle. His came from the hothouses and the menagerie of the Jardin des
Plantes in Paris and from picture books, and he built them as a gardener plants a bed: one plant
beside another and row behind row, every leaf drawn whole and turned flat to the viewer, each in its
own green, one half of it lighter than the other and the midrib painted between them, often with a
rim of ochre round its edge. The flowers and the fruit are hard-edged and bright, and the light in
them is still. He used more than twenty greens.

This garden is not one of his. A pale sun stands in a clear sky between an orange tree hung with fruit
and a palm, over a row of light trees. Below them the thicket goes back in rows, laurel, dracaena,
rubber plant, fern and aroid, falling back in places into deep dark recesses and standing in others
in planes of lit leaves; in front of it a banana spreads its torn leaves, lotus flowers stand up on
their stalks, pink, white and blue, a white lily and red bracts rise in spikes, and along the foot
grass and agave blades cross, rimmed in ochre. The sky went on first in level strokes, row under row,
and the sun on it in round strokes of cream, the sky's last strokes closing round its edge; then the
dark under-painting of the thicket, rubbed in every way with a big brush, and everything else from the
back to the front, each row covering the feet of the row behind. Every leaf was filled with a small
brush in strokes running out from the midrib or along its length, one half lighter than the other and
a second green worked in among them; where a stroke stopped at the edge it left a little ridge, and now
and then it ran a hair over the line. The paint is thin, and the weave of the canvas shows through it,
most in the sky.
"""

import zlib

import numpy as np
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Exotic Garden with a Pale Sun"
DATE = "2026"
MEDIUM = "Oil on canvas, thinly painted, every leaf drawn round with a small brush"
AFTER = ("Henri Rousseau, the jungle paintings, Paris, 1891–1910: Surprised! (1891), The Snake Charmer (1907), "
         "Exotic Landscape (1908 and 1910), The Dream (1910)")
ROOM = "The Garden"
YEAR = 1910
PLACE = "Paris"
REGION = "Europe"
NOTE = ("A jungle built row behind row, as Rousseau built his from the hothouses of the Jardin des Plantes: "
        "a pale sun between an orange tree and a palm, and lotus and red bracts lit against the dark thicket, "
        "every leaf drawn whole in a green of its own.")

H, W = 2460, 3200
LIGHT = (-0.6, -0.5, 0.62)
SUN = (1660.0, 340.0, 92.0)        # the pale sun in the gap: x, y, radius


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# the greens, each from dark to light: forty-five of them, and no leaf takes one unchanged
GREENS = {
    "yellow": pal("#566a25", "#6a7f2f", "#7f943a", "#95a848", "#abbb5a", "#c0cb72", "#d2d890"),
    "grass": pal("#2b4a26", "#395c2f", "#496e39", "#5b8245", "#6e9452", "#84a663", "#9cb87a"),
    "emerald": pal("#163c2c", "#1e4c37", "#295d44", "#356f52", "#448161", "#589372", "#71a586"),
    "teal": pal("#173b3b", "#204946", "#2a5954", "#366a62", "#447b71", "#588d82", "#71a095"),
    "sage": pal("#3d4d40", "#4a5c4c", "#596c5a", "#6a7d69", "#7e917c", "#94a691", "#adbea8"),
    "olive": pal("#3b401d", "#4a5025", "#5a612f", "#6d733a", "#828647", "#989b58"),
    "deep": pal("#12241b", "#182c22", "#1f362a", "#274133", "#304c3c", "#3a5846"),
}
HAZE = lin("#d3dacb")
RIB = pal("#d9d7a2", "#e6e1b6", "#c9c88f", "#b9bf86")
OCHRE = pal("#c8a446", "#d6b85c", "#b8923a", "#e2cd84")
RUST = pal("#7c3526", "#8f432b", "#6a2c22")
BARK = pal("#3e322a", "#55463a", "#6b5a4a", "#827060", "#9a8a78")
ORANGE = pal("#c9561a", "#dc6e20", "#ea8a2b", "#f3a43c", "#f7bf62")
PINK = pal("#cf7a90", "#e096a8", "#ecb1bd", "#f4ccd2", "#f9e1e0")
WHITE = pal("#d3d1c2", "#e5e1d2", "#f0ece0", "#f8f5ec")
BLUE = pal("#3d5ca6", "#5072b7", "#6b8ac7", "#8ea7d6", "#b3c5e3")
RED = pal("#8c1521", "#ab1c27", "#c7282a", "#da3d2b", "#e75834")
GOLD = pal("#d0901f", "#e2ad2f", "#efc850")
SKY = pal("#9fbcc0", "#b3cac7", "#c9d8cd", "#dde1ca", "#ebe4c6")
SKY_AT = (0, 330, 620, 880, 1150)
SUNLIT = pal("#f9f4e4", "#f3ecd5", "#ece6d2")      # the sun's creams
GLOW = lin("#e6e9d9")                              # the pale the sky turns close round it
WARM = np.float32([1.18, 1.07, 0.72])              # the second green worked into a leaf: a yellower one
COOL = np.float32([0.8, 0.97, 1.16])               # or a bluer
DUSK = lin("#0c2420")                              # the dark a leaf in a recess sinks into
LUMA = np.float32([0.2126, 0.7152, 0.0722])
# where the thicket falls back into a deep recess (below 0) and where a plane of it stands in the light
# (above 0): x, y, reach across, reach up and down, how much
GLADES = ((330, 1260, 300, 290, -1.0), (1850, 1350, 200, 170, -0.9), (2480, 1250, 300, 270, -1.0),
          (2060, 1090, 210, 120, 0.9), (2960, 1180, 230, 160, 0.9), (820, 1060, 220, 110, 0.6))


def glade(x, y):
    return sum(z * np.exp(-((x - x0) / rx) ** 2 - ((y - y0) / ry) ** 2) for x0, y0, rx, ry, z in GLADES)


def fresh(seed, name):
    """Each part of the garden draws on chances of its own, so the rest stays as it is when one is changed."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def haze(c, a):
    return c * (1 - a) + HAZE * a


def ang(v):
    return float(np.arctan2(v[1], v[0]))


def toward(a, b):
    """The signed turn from heading a to heading b, radians."""
    return float(np.angle(np.exp(1j * (b - a))))


class Easel:
    """The canvas as it is being painted: colour, the height of the paint and how thin it lies, with what a
    small brush leaves the same wherever it goes: the unevenness of an edge drawn round with it, the bristle
    marks in its strokes, and the paint mixed a shade warmer or cooler from one load to the next. `dusk`
    sinks whatever is painted into the dark of a recess."""

    def __init__(self, rgb, height, r):
        self.rgb, self.height = rgb, height
        self.thin = np.full(height.shape, 0.55, np.float32)
        self.grain = noise.field(height.shape, 1.8, r)
        self.wash = noise.field(height.shape, 45, r)
        self.B = impasto.bristles(r)
        self.jit = r.uniform(-1, 1, 8192).astype(np.float32)
        self.dusk = 0.0

    def look(self, along, across):
        return impasto._look(self.B, along, across)


def wavers(r, s, L, scale):
    """A hand's slow wavering along a line, of unit size, about `scale` px from one swing to the next."""
    k = np.arange(int(L / scale) + 6) - 2.0
    return np.interp(s / scale, k, r.normal(0, 1, len(k))).astype(np.float32)


def leaf(ez, r, P, half, light, dark, lit=1, phi=0.6, curve=0.3, band=7.0, mottle=0.06, shade=(0.1, 0.06),
         rib=None, veins=None, rim=None, rimside=0, tip=None, teeth=0, serr=0.0, notch=None, slits=(), body=0.25,
         soft=1.5, wob=0.04, sides=(1.0, 1.0), second=0.5):
    """One leaf painted flat-on: the outline drawn round with a small brush, each half filled in strokes
    that follow the leaf, one half lighter than the other and a second green worked in among them; then
    the midrib, the veins and a rim in another colour painted over. In place on the easel.

    P, half   the midrib from the stalk to the tip (n, 2), and the half-width along it (n,), px
    light, dark   the colours of the lit half and of the other; `lit` +1 or -1 picks the side
    phi, curve    the strokes' angle to the midrib, and how far they swing toward the tip near the edge;
              below about 0.45 they run along the leaf, the brush lifted and loaded again every few
              widths, above it each runs from the midrib out to the edge
    band      the width of the brush that fills it, px; `mottle` how much one stroke differs from the next
    shade     how much darker each half runs toward its edge, and toward the stalk
    rib, veins, rim   (colour, width) of the midrib; (colour, spacing, width, alpha) of the veins;
              (colour, width, alpha) of a rim just inside the edge, on both sides or on `rimside` only
    tip       (colour, amount): the colour the leaf turns toward its tip
    teeth, serr   the number of teeth along each edge and how deep they bite
    notch     (depth, width) px of the cleft at the stalk of a heart-shaped leaf
    slits     tears from the edge in toward the midrib along a vein: (side, at px, gap px, depth 0..1)
    body      the height of the paint, px; `soft` the edge's softness, px; `wob` how the hand wavers
    sides     the two halves' widths, unequal on a leaf turned a little away
    second    how much of the second green, yellower or bluer, is worked in"""
    Hc, Wc = ez.height.shape
    C = brush.path(np.column_stack([P, half]), 0.5)
    X, Y, Wd = C[:, 0], C[:, 1], np.maximum(C[:, 2], 0)
    n = len(C)
    T = np.gradient(C[:, :2], axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C[:, :2], axis=0).T))])
    L = sl[-1]
    if L < 2 or Wd.max() < 0.3:
        return
    wl = Wd * sides[0] * (1 + wob * noise.line1d(n, 50, r))
    wr = Wd * sides[1] * (1 + wob * noise.line1d(n, 50, r))
    if teeth:
        for w_ in (wl, wr):
            w_ *= 1 - serr * (1 - (sl / L * teeth + r.uniform(0, 1)) % 1) ** 1.5
    m = max(wl.max(), wr.max()) + 6
    x0, x1 = int(np.floor(X.min() - m)), int(np.ceil(X.max() + m)) + 1
    y0, y1 = int(np.floor(Y.min() - m)), int(np.ceil(Y.max() + m)) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= Wc or y0 >= Hc:
        return
    lab = np.full((y1 - y0, x1 - x0), -1, np.int32)
    lab[np.rint(Y - y0).astype(int), np.rint(X - x0).astype(int)] = np.arange(n)
    dist, (iy, ix) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    k = lab[iy, ix]
    reach = ndimage.maximum_filter1d(np.maximum(wl, wr), 41) + 5.0
    jj, kk = np.nonzero(dist <= reach[k])
    py, px = jj + y0, kk + x0
    ok = (py >= 0) & (py < Hc) & (px >= 0) & (px < Wc)
    jj, kk, py, px = jj[ok], kk[ok], py[ok], px[ok]
    k = k[jj, kk]
    vx, vy = px - X[k], py - Y[k]
    s = sl[k] + vx * T[k, 0] + vy * T[k, 1]
    u = vy * T[k, 0] - vx * T[k, 1]
    au = np.abs(u)
    # how far inside the outline, measured square to it, so the edge is drawn alike wherever it runs
    lw, rw_ = (np.interp(s, sl, w_) for w_ in (wl, wr))
    lg, rg = (np.interp(s, sl, np.gradient(w_, sl)) for w_ in (wl, wr))
    hw = np.where(u > 0, lw, rw_)
    e = (hw - au) / np.sqrt(1 + np.where(u > 0, lg, rg) ** 2)
    e = np.minimum(np.minimum(e, s + 0.5), L - s)
    if notch:
        e = np.minimum(e, s - notch[0] * np.clip(1 - au / notch[1], 0, 1))
    t = np.clip(s / L, 0, 1)
    f = np.clip(au / np.maximum(hw, 0.5), 0, 1)
    ph = phi + curve * f
    d = s * np.cos(ph) + au * np.sin(ph)          # along the strokes
    c = -s * np.sin(ph) + au * np.cos(ph)         # across them
    for side, at, gap, depth in slits:            # a tear, held open wider than a stroke can run over
        c0 = -at * np.sin(phi)
        apart = np.maximum(np.abs(c - c0) - gap / 2, (1 - depth - f) * hw)
        e = np.where(np.sign(u) == side, np.minimum(e, 3 * apart), e)
    # the strokes: lanes a brush wide side by side across the leaf, ragged where one meets the next, each
    # stroke a shade off its neighbours; where it stopped, most on the line, a few a hair over it
    o = r.uniform(0, 4096, 10)
    q = c / band + 0.12 * ez.look(d * 0.15 + o[0], c * 0.05 + o[1]) + 0.07 * ez.look(d * 1.3 + o[2], c * 0.3 + o[3])
    lane = np.floor(q).astype(np.int64)
    fq = q - lane
    along = phi < 0.45
    if along:
        z = d / (band * r.uniform(5, 9)) + ez.jit[lane % 8192]
        seg = np.floor(z).astype(np.int64)
        fd = z - seg
    else:
        seg = (u > 0).astype(np.int64)
        fd = f
    sk = (lane * 131 + seg * 977 + int(o[4])) % 8192
    j1, j2, j3 = ez.jit[sk], ez.jit[(7 * sk + 3001) % 8192], ez.jit[(13 * sk + 5003) % 8192]
    over = np.where(j2 > 0.75, 8.0 * (j2 - 0.75), 0.3 * j2) \
        * np.sqrt(np.clip(np.sin(np.pi * fd) if along else 1 - (2 * fq - 1) ** 2, 0, 1))   # a stroke's end is round
    ee = e + over + 0.5 * ez.look(f * 12 + o[5], s * 1.1 + o[6])
    cov = np.clip(ee / soft + 0.5 + 0.2 * ez.grain[py, px], 0, 1)
    keep = cov > 0.004
    if not keep.any():
        return
    py, px, u, au, s, t, f, d, c, ee, cov, fd, fq, sk, j1, j2, j3 = (
        a[keep] for a in (py, px, u, au, s, t, f, d, c, ee, cov, fd, fq, sk, j1, j2, j3))
    g = ez.look(d * 0.9 + o[7], c * 1.1 + 37.0 * (sk % 97))      # the bristles' streaks, each stroke its own
    half_lit = noise.smoothstep(-1.3, 1.3, u * lit)              # the two halves meet softly under the midrib
    col = dark + (light - dark) * half_lit[:, None]
    hue = WARM if r.random() < 0.5 else COOL
    a1, a2, b1, b2 = r.uniform(0.8, 2.2), r.uniform(0.6, 1.6), r.uniform(0, 6.3), r.uniform(0, 6.3)
    w2 = second * noise.smoothstep(-0.2, 0.8, np.cos(np.pi * a1 * t + b1) * np.cos(np.pi * a2 * f + b2) + 0.45 * j3)
    col = col * (1 + (hue - 1) * w2[:, None])
    run = r.uniform(-0.3, 0.3)                    # each leaf lighter toward its tip, or darker
    t0, f0 = r.uniform(0.25, 0.75), r.uniform(0.2, 0.6)          # and where the light lies fullest on it
    glow = 0.24 * half_lit * np.exp(-(t - t0) ** 2 / 0.08 - (f - f0) ** 2 / 0.15)
    start = np.exp(-(fd / 0.1) ** 2)              # where the brush landed, loaded
    end = np.exp(-((fd - 0.95) / 0.06) ** 2)      # and where it lifted, leaving a ridge
    tone = (1 - shade[0] * f ** 1.5) * (1 - shade[1] * (1 - t) ** 2) * (1 + run * (2 * t - 1) + glow) \
        * (1 + mottle * j1 + 0.035 * g + 0.04 * ez.wash[py, px] - 0.04 * start) \
        * (1 - (0.08 + 0.05 * j2) * np.exp(-np.maximum(ee, 0) / 1.8))
    col = col * tone[:, None]
    if tip is not None:
        col += (tip[0] - col) * (tip[1] * noise.smoothstep(0.45, 1.0, t))[:, None]
    if veins:                                     # each vein its own width and strength, thinning and fading outward
        vc, sp, vw, va = veins
        cv = (c + 0.6 * ez.look(d * 0.3 + o[8], c * 0.05 + o[9])) / sp
        cv = cv + 0.28 * np.interp(cv, np.arange(80) - 40.0, r.normal(0, 1, 80)) + 0.5
        kv = np.floor(cv).astype(np.int64)
        dq = np.abs(cv - kv - 0.5) * sp
        v1, v2 = ez.jit[(kv * 17 + 101) % 8192], ez.jit[(kv * 29 + 707) % 8192]
        wv = vw * (1 + 0.35 * v1) * (1.3 - 0.9 * f)
        a = np.clip((wv / 2 - dq) / 0.7 + 0.5, 0, 1) * va * (0.75 + 0.25 * v2) \
            * noise.smoothstep(0.98, 0.5 + 0.15 * v1, f) * noise.smoothstep(0.02, 0.1, f) \
            * (0.7 + 0.3 * noise.smoothstep(-0.8, 0.4, ez.look(d * 0.7 + o[8], kv * 11.0 + o[9])))
        col += (vc - col) * a[:, None]
    if rib:
        rc, rw = rib
        wv = rw * (1 - t) ** 0.6 * (1 + 0.2 * wavers(r, s, L, 25))
        a = np.clip((wv / 2 + 0.35 - au) / 1.3 + 0.5, 0, 1) * (1 - 0.5 * t ** 1.5) * r.uniform(0.5, 0.9) \
            * (0.6 + 0.4 * noise.smoothstep(-0.8, 0.6, ez.look(s * 0.8 + o[2], u * 0.5 + o[5])))
        col += (rc - col) * a[:, None]
    if rim:
        mc, mw, ma = rim
        mv = mw * np.clip(1 + 0.45 * wavers(r, s, L, 30), 0.3, 2.0)
        a = np.clip((mv - ee) / 0.8 + 0.5, 0, 1) * ma * (1 if not rimside else (np.sign(u) == rimside)) \
            * (0.75 + 0.25 * noise.smoothstep(-0.8, 0.5, ez.look(s * 0.5 + o[3], f * 3 + o[4])))
        col += (mc - col) * a[:, None]
    if ez.dusk:
        col += (DUSK - col) * ez.dusk
    rgb, hgt = ez.rgb, ez.height
    rgb[py, px] += (col - rgb[py, px]) * cov[:, None]
    dome = 1 - (2 * fq - 1) ** 2
    hb = body * (0.75 + 0.3 * g + 0.15 * j1 + 0.35 * dome + (0.9 + 0.5 * j2) * end + 0.3 * start)
    hgt[py, px] += (hb + 0.5 * hgt[py, px] - hgt[py, px]) * cov
    thin = np.clip(0.35 + 0.35 * fd - 0.5 * end + 0.15 * j3, 0.05, 1)
    ez.thin[py, px] += (thin - ez.thin[py, px]) * cov


def spine(p0, a, L, bend=0.0, n=16, sway=0.0, r=None):
    """A midrib or a stalk from p0 heading `a`, turning `bend` radians over its length, swaying a little."""
    s = np.linspace(0, 1, n)
    th = a + bend * s ** 1.4
    if sway and r is not None:
        th = th + sway * np.sin(np.pi * s * r.uniform(0.8, 1.6) + r.uniform(0, 6))
    st = np.stack([np.cos(th), np.sin(th)], 1) * L / (n - 1)
    return np.asarray(p0, float) + np.vstack([[0, 0], np.cumsum(st[:-1], 0)])


def profile(kind, n=16):
    """The half-width of a leaf of each kind along its midrib, 1 at its widest."""
    t = np.linspace(0, 1, n)
    if kind == "blade":
        return (1 - t) ** 0.8 * (1 + 0.25 * np.sin(np.pi * t))
    if kind == "lance":
        return np.sin(np.pi * t ** 0.85) ** 0.9
    if kind == "ovate":
        return np.sin(np.pi * t ** 0.7) ** 0.75
    if kind == "round":
        return np.sin(np.pi * t ** 0.9) ** 0.6
    if kind == "heart":
        tm = 0.28
        return np.where(t < tm, np.sqrt(np.clip(1 - ((t - tm) / tm) ** 2, 0, 1)),
                        np.cos(np.pi / 2 * np.clip((t - tm) / (1 - tm), 0, 1)) ** 0.9)
    if kind == "banana":
        return noise.smoothstep(0, 0.1, t) ** 0.4 * (1 - noise.smoothstep(0.8, 1.0, t)) ** 0.5
    if kind == "petal":
        return np.sin(np.pi * t ** 0.75) ** 0.8
    raise ValueError(kind)


def tints(fam, tone, r, step=1.6, drift=0.04, hz=0.0):
    """The two halves of one leaf: a green of the family at about `tone` (0 its darkest, 1 its lightest),
    its lit half `step` places lighter than the other, both nudged so that no two leaves match."""
    G = GREENS[fam] if isinstance(fam, str) else fam
    n = len(G)
    x = np.clip(tone, 0, 1) * (n - 1) + r.normal(0, 0.35)

    def at(v):
        v = float(np.clip(v, 0, n - 1))
        i = min(int(v), n - 2)
        return G[i] + (G[i + 1] - G[i]) * (v - i)

    j = np.exp(r.normal(0, drift, 3)).astype(np.float32)
    return haze(at(x + step / 2) * j, hz), haze(at(x - step / 2) * j, hz)


def stem(ez, r, P, w0, w1, light, dark, **kw):
    """A stalk, a branch or a trunk: strokes drawn along it, its lit side lighter."""
    P = np.asarray(P, float)
    leaf(ez, r, P, np.linspace(w0, w1, len(P)), light, dark, lit=kw.pop("lit", 1), phi=0.0, curve=0.0,
         band=kw.pop("band", max(3.0, 0.5 * w0)), mottle=0.07, shade=(0.3, 0.0), wob=0.02, **kw)


def disc(ez, r, c, R, light, dark, lit=(-0.62, -0.78), mottle=0.05, band=5.0, body=0.4, squash=1.0, comb=0.3,
         rim=0.16):
    """A round thing painted flat, turning lighter toward one side and darker round its rim on the other:
    an orange, the heart of a lotus. Its outline is drawn round by hand, a little out of true; the strokes
    go round it, `band` px wide, each a shade off the next, the bristles combing them `comb` deep, and stop
    at the edge, a few a hair over it."""
    Hc, Wc = ez.height.shape
    x0, x1 = max(int(c[0] - R - 6), 0), min(int(c[0] + R + 7), Wc)
    y0, y1 = max(int(c[1] - R * squash - 6), 0), min(int(c[1] + R * squash + 7), Hc)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - c[0], (yy - c[1]) / squash
    rad, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    k = np.arange(2, 6)[:, None, None]
    wav = (0.022 / k * r.normal(0, 1, (4, 1, 1)) * np.sin(k * th + r.uniform(0, 6, (4, 1, 1)))).sum(0)
    rr = R * (1 + wav * (R > 4))
    o = r.uniform(0, 4096, 6)
    sh_ = rad.shape
    q = rad / band + 0.15 * ez.look((th * R * 0.3 + o[0]).ravel(), (rad * 0.2 + o[1]).ravel()).reshape(sh_)
    lane = np.floor(q).astype(np.int64)
    z = th * np.maximum(rad, band) / (band * r.uniform(2.5, 4.0)) + 3 * ez.jit[lane % 8192]
    seg = np.floor(z).astype(np.int64)
    sk = (lane * 131 + seg * 977 + int(o[2])) % 8192
    jit, j2 = ez.jit[sk], ez.jit[(7 * sk + 3001) % 8192]
    over = np.where(j2 > 0.75, 7.0 * (j2 - 0.75), 0.25 * j2)
    fray = ez.look((rad * 0.3 + o[3]).ravel(), (th * R + o[4]).ravel()).reshape(sh_)
    e = (rr - rad) * min(1.0, squash + 0.3) + over + 0.4 * fray
    cov = np.clip(e / 1.0 + 0.5 + 0.2 * ez.grain[y0:y1, x0:x1], 0, 1)
    if not (cov > 0.004).any():
        return
    sh = (dx * lit[0] + dy * lit[1]) / R
    a = noise.smoothstep(-0.9, 0.75, sh)
    col = dark + (light - dark) * a[..., None]
    col = col * (1 - rim * noise.smoothstep(0.65, 1.0, rad / R) * noise.smoothstep(0.3, -0.6, sh))[..., None]
    g = ez.look((th * np.maximum(rad, 1) * 0.9 + o[5]).ravel(), (rad * 1.1 + o[1]).ravel()).reshape(sh_)
    col = col * (1 + mottle * jit + 0.03 * g)[..., None]
    if ez.dusk:
        col += (DUSK - col) * ez.dusk
    sub = ez.rgb[y0:y1, x0:x1]
    sub += (col - sub) * cov[..., None]
    hs = ez.height[y0:y1, x0:x1]
    hs += (body * (1 + comb * g + 0.2 * jit) + 0.4 * hs - hs) * cov
    ts = ez.thin[y0:y1, x0:x1]
    ts += (0.3 + 0.1 * jit - ts) * cov


def sprig(ez, r, P, size, n, fam, tone, kind="ovate", hz=0.0, rib=True, veined=0.0, spread=(0.45, 1.0), hang=0.35,
          shape=(0.26, 0.4), step=1.6, ribs=RIB, first=0.15):
    """Leaves set one by one along a twig, alternately, each on a short stalk and angled forward, one
    at the tip; no two the same size, angle, curl or green. `kind` one shape or several to choose from."""
    C = brush.path(np.asarray(P, float), 2.0)
    T = np.gradient(C, axis=0)
    ts = np.clip(np.linspace(first, 1.0, n) + r.normal(0, 0.25 / n, n), 0.05, 1.0)
    ts[-1] = 1.0
    for i, t in enumerate(ts):
        j = min(int(t * (len(C) - 1)), len(C) - 1)
        a0 = ang(T[j])
        a = a0 if i == n - 1 else a0 + (-1) ** (i + (n % 2)) * r.uniform(*spread)
        a += hang * toward(a, np.pi / 2) * r.uniform(0, 1)
        Ll = size * r.uniform(0.7, 1.2) * (0.8 + 0.3 * t)
        k = kind if isinstance(kind, str) else kind[r.integers(len(kind))]
        wd = Ll * r.uniform(*shape) * (1.25 if k == "heart" else 1)
        m = 16 if k == "heart" else 8
        p = C[j] + (0.06 - 0.2 * (k == "heart")) * Ll * np.array([np.cos(a), np.sin(a)])
        Q = spine(p, a, Ll, r.normal(0, 0.3), n=m)
        light, dark = tints(fam, tone + r.normal(0, 0.1), r, step=step, hz=hz)
        rc = haze(ribs[r.integers(len(ribs))], hz)
        rc = light + (rc - light) * np.clip((Ll - 50) / 110, 0.3, 1.0)   # a small leaf's midrib shows faintly
        kw = {}
        how = r.random()                  # not every leaf alike: some midribs dark, some hardly painted
        if how < 0.2:
            rc = dark * 0.72
        if k == "heart":
            kw["notch"] = (0.24 * Ll, 0.45 * wd)
        if r.random() < veined:
            kw["veins"] = ((rc + light) / 2, 0.11 * Ll * r.uniform(0.85, 1.2), 0.008 * Ll + 0.6, 0.3)
        if r.random() < 0.25:             # and some drawn along one edge in a paler green
            kw["rim"], kw["rimside"] = ((light + haze(GREENS["yellow"][5], hz)) / 2, max(1.2, 0.035 * wd + 0.6),
                                        0.6), r.choice([-1, 1])
        leaf(ez, r, Q, wd * profile(k, m), light, dark, lit=r.choice([-1, 1]), phi=0.75 + 0.2 * (k == "heart"),
             curve=0.3, band=max(3.0, 0.3 * wd), rib=(rc, max(0.8, 0.05 * wd)) if rib and how < 0.88 else None,
             shade=(0.2, 0.08), body=0.22, sides=(r.uniform(0.85, 1.1), r.uniform(0.85, 1.1)), **kw)


def orange(ez, r, p, R, hz=0.0):
    """An orange hanging on its little stalk: lighter on one side, darker round the rim on the other, a
    touch of green at the stalk; some riper and redder than others."""
    q = p + np.array([r.normal(0, 0.15 * R), 0.5 * R + R])
    stem(ez, r, [p, (p + q) / 2 + [r.normal(0, 2), 0], q - [0, 0.8 * R]], 2.2, 1.6, haze(GREENS["olive"][4], hz),
         haze(GREENS["olive"][2], hz))
    i = r.uniform(1.2, 3.6)
    disc(ez, r, q, R, haze(ORANGE[min(int(i) + 1, 4)] * 1.05, hz), haze(ORANGE[int(i) - 1], hz),
         lit=(-0.55 + r.normal(0, 0.3), -0.8 + r.normal(0, 0.1)), band=max(3.0, 0.3 * R), mottle=0.04, comb=0.1,
         rim=r.uniform(0.06, 0.18))
    if r.random() < 0.6:
        disc(ez, r, q - [0.05 * R, 0.88 * R], 0.13 * R, haze(GREENS["olive"][3], hz), haze(GREENS["olive"][1], hz),
             body=0.3, band=3.0)


def tree(ez, r, foot, top, w, fam, tone, size, kind="ovate", depth=3, fork=(2, 4), reach=(0.45, 0.72), twig=8,
         fruit=0.0, fruit_r=(20, 28), hz=0.0, spread=0.75, lift=0.3, bend=0.35, shape=(0.26, 0.4), first=0.55,
         fan=None, crown=None, veined=0.0):
    """A tree: the trunk, its branches forking smaller and smaller, then the leaves twig by twig and the
    fruit hanging among them, no two the same size, a leaf hanging down in front of one here and there.
    `fan` spreads the first boughs evenly about the trunk's heading, each `crown` px long."""
    segs = []

    def grow(p, a, L, wd, order):
        P = spine(p, a, L, r.normal(0, bend), n=10, sway=0.06, r=r)
        segs.append((P, wd, order))
        if order == 0:
            return
        k = r.integers(*fork)
        ts = np.sort(r.uniform(first if order == depth else 0.3, 1.0, k))
        for i, t in enumerate(ts):
            q = P[min(int(t * 9 + 0.5), 9)]
            if fan is not None and order == depth:
                na = a + fan * (2 * (i + r.uniform(0.2, 0.8)) / k - 1)
                nl = crown * r.uniform(0.8, 1.15)
            else:
                na = a + r.choice([-1, 1]) * r.uniform(0.25, spread * 1.3)
                nl = L * r.uniform(*reach)
            na += lift * toward(na, -np.pi / 2)
            grow(q, na, nl, wd * r.uniform(0.45, 0.62), order - 1)

    d = np.subtract(top, foot)
    grow(np.asarray(foot, float), ang(d), float(np.hypot(*d)), w, depth)
    for P, wd, order in sorted(segs, key=lambda s: -s[1]):
        light, dark = haze(BARK[3] * np.exp(r.normal(0, 0.05)), hz), haze(BARK[1], hz)
        stem(ez, r, P, wd, max(wd * 0.6, 1.2), light, dark, lit=-1)
    twigs = [s for s in segs if s[2] < depth]
    hung = []
    for k, i in enumerate(r.permutation(len(twigs))):
        P, _, order = twigs[i]
        near = k / max(len(twigs) - 1, 1) - 0.5             # the twigs painted last stand in front, lighter
        f_ = fam if isinstance(fam, str) else fam[r.integers(len(fam))]
        sprig(ez, r, P, size, twig if order <= 1 else max(2, twig // (order + 1)), f_, tone + 0.3 * near, kind, hz,
              shape=shape, first=0.15 if order <= 1 else 0.5, veined=veined)
        if r.random() < fruit:
            R = fruit_r[0] + (fruit_r[1] - fruit_r[0]) * r.random() ** 1.4
            hung.append((P[r.integers(3, 10)] + [0, 4], R, f_))
    held = []
    for p, R, f_ in hung:                         # the fruit hangs in front of the leaves, never two in one place
        c = p + [0, 1.5 * R]
        if any(np.hypot(*(c - c_)) < 0.9 * (R + R_) for c_, R_ in held):
            continue
        held.append((c, R))
        orange(ez, r, p, R, hz)
        if r.random() < 0.22:                     # but a leaf hangs down in front of one here and there
            sprig(ez, r, [p + [r.normal(0, 4), -2], p + [r.normal(0, 0.4 * R), 0.9 * R]], size, 1, f_, tone + 0.2,
                  kind, hz, shape=shape, first=1.0)


def clump(ez, r, base, n, length, width, fam, tone, spread=0.45, lean=0.0, droop=1.0, rim=0.8, hz=0.0,
          rib=0.0, stiff=0.0, kind="blade", step=1.6):
    """Grass, sedge or agave: blades from one root, each its own length, width and curve, crossing one
    another; a rim of ochre or cream along one edge or both."""
    for i in range(n):
        a = -np.pi / 2 + lean + np.clip(r.normal(0, spread), -1.4, 1.4)
        L = length * r.uniform(0.5, 1.15)
        out = np.sign(a + np.pi / 2 + 1e-6)
        bend = out * droop * r.uniform(0.2, 1.2) * (1 - stiff) + r.normal(0, 0.1)
        p0 = np.asarray(base, float) + [r.normal(0, 1.5 * width), r.uniform(0, 0.5 * width)]
        P = spine(p0, a, L, bend, n=12, sway=0.04, r=r)
        wd = width * r.uniform(0.7, 1.25)
        light, dark = tints(fam, tone + r.normal(0, 0.12), r, step=step, hz=hz)
        rc = None
        if r.random() < rim:
            rc = (haze(OCHRE[r.integers(4)], hz), max(1.2, 0.14 * wd), r.uniform(0.75, 0.95))
        leaf(ez, r, P, wd * profile(kind, 12), light, dark, lit=r.choice([-1, 1]), phi=0.04, curve=0.0,
             band=max(3.0, 0.45 * wd), rim=rc, rimside=r.choice([-1, 0, 1]),
             rib=(haze(RIB[r.integers(4)], hz), max(1.0, 0.1 * wd)) if r.random() < rib else None,
             shade=(0.15, 0.1), body=0.3)


def frond(ez, r, base, a, L, bend, fam, tone, leaflet, width, n, spread=1.0, droop=0.6, hz=0.0, kind="lance",
          rachis=None):
    """A palm frond or a fern: the leaflets set along both sides of the midrib, longest in its middle,
    each hanging a little under its own weight."""
    R = spine(base, a, L, bend, n=14, sway=0.03, r=r)
    C = brush.path(R, 3.0)
    T = np.gradient(C, axis=0)
    items = []
    for side in (-1, 1):
        for t in np.clip(np.linspace(0.06, 0.985, n) + r.normal(0, 0.35 / n, n), 0.03, 0.995):
            j = min(int(t * (len(C) - 1)), len(C) - 1)
            a0 = ang(T[j])
            la = a0 + side * spread * (1.0 - 0.45 * t) * r.uniform(0.8, 1.2)
            ll = leaflet * (0.3 + 0.7 * np.sin(np.pi * min(t, 0.999) ** 0.75)) * r.uniform(0.75, 1.2)
            b = droop * toward(la, np.pi / 2) * r.uniform(0.4, 1.0) * ll / leaflet
            items.append((t, C[j], la, ll, b, side))
    rc = rachis if rachis is not None else (haze(GREENS["yellow"][3], hz), haze(GREENS["olive"][2], hz))
    stem(ez, r, R, max(2.0, 0.008 * L + 1.5), 1.2, *rc)
    for k in r.permutation(len(items)):
        t, p, la, ll, b, side = items[k]
        Q = spine(p, la, ll, b, n=8)
        light, dark = tints(fam, tone + r.normal(0, 0.1), r, hz=hz)
        wd = width * r.uniform(0.8, 1.2) * (0.6 + 0.4 * np.sin(np.pi * min(t, 0.999) ** 0.7))
        leaf(ez, r, Q, wd * profile(kind, 8), light, dark, lit=side, phi=0.08, curve=0.1, band=max(2.5, 0.5 * wd),
             rib=(haze(RIB[r.integers(4)], hz), max(0.7, 0.12 * wd)), shade=(0.12, 0.1), body=0.22)


def palm(ez, r, foot, crown, w, fronds, fam, tone, length, leaflet, width, hz=0.0, n=30):
    """A palm: the trunk ringed where old fronds fell, rising in a long curve, and its crown of fronds
    thrown out every way, unevenly: the young ones short and standing up, the old ones long and hanging."""
    mid = (np.asarray(foot) + np.asarray(crown)) / 2 + [r.normal(0, 40), 0]
    P = brush.path(np.array([foot, mid, crown], float), 40.0)
    leaf(ez, r, P, np.linspace(w, 0.65 * w, len(P)), haze(BARK[4], hz), haze(BARK[2], hz), lit=-1, phi=np.pi / 2,
         curve=0.0, band=6.0, mottle=0.08, shade=(0.35, 0.0), veins=(haze(BARK[1], hz), 0.9 * w, 2.4, 0.65), wob=0.02)
    gaps = r.uniform(0.4, 1.6, fronds)              # the fronds not evenly spaced round the crown
    angles = -np.pi - 0.6 + (1.2 + np.pi) * (np.cumsum(gaps) - gaps[0] / 2) / gaps.sum()
    for k in r.permutation(fronds):
        a = angles[k] + r.normal(0, 0.06)
        side = np.sign(np.cos(a) + 1e-6)
        up = max(0.0, -np.sin(a)) * (1 - abs(np.cos(a)))          # how near upright it stands
        old = r.random()
        L = length * (0.5 + 0.65 * old) * (1 - 0.3 * up)
        b = side * (0.3 + 1.5 * old * r.uniform(0.6, 1.0))
        frond(ez, r, crown, a, L, b, fam, tone + r.normal(0, 0.08), leaflet * (0.75 + 0.35 * old) * r.uniform(0.9, 1.1),
              width, int(n * (0.6 + 0.5 * old)), spread=r.uniform(0.7, 1.15), droop=0.3 + 0.9 * old * r.uniform(0.7, 1),
              hz=hz)


def banana(ez, r, foot, h, n, fam, tone, length, width, hz=0.0, old=True):
    """A banana: the soft trunk striped lengthways, and the great leaves rising from its top and arching
    over, torn here and there from the edge to the midrib."""
    top = np.asarray(foot, float) - [r.normal(0, 15), h]
    P = brush.path(np.array([foot, (np.asarray(foot) + top) / 2 + [r.normal(0, 12), 0], top], float), 30.0)
    light, dark = tints("olive", 0.65, r, hz=hz)
    leaf(ez, r, P, np.linspace(0.42 * width, 0.3 * width, len(P)), light, dark, lit=-1, phi=0.0, curve=0.0,
         band=5.0, mottle=0.12, shade=(0.3, 0.0), veins=None, wob=0.02)
    if old:     # an old leaf hanging down the trunk, gone brown
        a = r.choice([-1, 1])
        Q = spine(top + [0, 0.15 * h], np.pi / 2 + a * 0.3, 0.55 * length, -a * 0.4, n=10)
        leaf(ez, r, Q, 0.45 * width * profile("banana", 10), haze(OCHRE[2] * 0.75, hz), haze(BARK[2], hz),
             lit=1, phi=1.3, band=8, mottle=0.12, rib=(haze(OCHRE[1] * 0.8, hz), 4),
             slits=[(r.choice([-1, 1]), r.uniform(40, 0.5 * length), 3, r.uniform(0.5, 1)) for _ in range(5)])
    angles = np.linspace(-np.pi / 2 - 1.1, -np.pi / 2 + 1.1, n) + r.normal(0, 0.15, n)
    for k in np.argsort(-np.abs(angles + np.pi / 2) + r.normal(0, 0.2, n)):
        a = angles[k]
        side = np.sign(a + np.pi / 2 + 1e-6)
        L = length * r.uniform(0.7, 1.1) * (1 - 0.25 * (abs(a + np.pi / 2) < 0.4))
        Q = spine(top + [r.normal(0, 6), r.uniform(0, 30)], a, L, side * r.uniform(0.5, 1.6), n=14, sway=0.04, r=r)
        light, dark = tints(fam, tone + r.normal(0, 0.1), r, step=2.2, hz=hz)
        slits = [(r.choice([-1, 1]), r.uniform(0.2, 0.9) * L, r.uniform(2, 5), r.uniform(0.35, 1.0))
                 for _ in range(r.integers(2, 9))]
        leaf(ez, r, Q, width * r.uniform(0.8, 1.15) * profile("banana", 14), light, dark, lit=r.choice([-1, 1]),
             phi=1.3, curve=0.1, band=r.uniform(8, 12), mottle=0.07, slits=slits, rib=(haze(RIB[r.integers(4)], hz),
             r.uniform(6, 9)), veins=(haze(GREENS["yellow"][5], hz), 13, 1.0, 0.18), shade=(0.15, 0.12), body=0.3)


def lotus(ez, r, c, R, colours, a=-np.pi / 2, hz=0.0, stalk=None):
    """A lotus seen from the side: the stalk, the back petals standing up, the gold heart, and the front
    petals open over it, each pointed petal paler at its foot and flushed at its tip."""
    c = np.asarray(c, float)
    if stalk is not None:
        stem(ez, r, np.array([stalk, (np.asarray(stalk) + c) / 2 + [r.normal(0, 15), 0], c + [0, 0.15 * R]]), 5.0, 4.0,
             haze(GREENS["olive"][4], hz), haze(GREENS["olive"][2], hz))
    fam = colours
    for row, (m, spread, ln, tn) in enumerate(((6, 0.95, 1.0, 0.35), (5, 1.25, 0.9, 0.7), (6, 1.55, 0.95, 0.85))):
        offs = np.linspace(-spread, spread, m) + r.normal(0, 0.08, m)
        for k in r.permutation(m):
            pa = a + offs[k]
            L = R * ln * r.uniform(0.85, 1.12) * (1 - 0.18 * abs(offs[k]) / spread)
            Q = spine(c, pa, L, r.normal(0, 0.15) - 0.25 * np.sign(offs[k]) * (row == 2), n=8)
            i = np.clip(tn * (len(fam) - 1) + r.normal(0, 0.4), 1, len(fam) - 1)
            light = haze(fam[int(np.ceil(i))], hz)
            dark = haze(fam[int(np.floor(i)) - 1], hz)
            leaf(ez, r, Q, R * 0.27 * r.uniform(0.85, 1.15) * profile("petal", 8), light, dark,
                 lit=np.sign(offs[k] + 1e-6) * (-1) ** row, phi=0.12, curve=0.2, band=max(5, 0.16 * R), mottle=0.02,
                 rib=(haze((fam[max(int(i) - 1, 0)] + light) / 2, hz), 1.2), tip=(haze(fam[0], hz), 0.3),
                 shade=(0.1, -0.15), body=0.3)
        if row == 0:
            disc(ez, r, c + [0, -0.12 * R], 0.2 * R, haze(GOLD[2], hz), haze(ORANGE[1], hz), squash=0.6, body=0.4)


def spike(ez, r, foot, h, fam, n, size, hz=0.0, lean=0.0, kind="lance"):
    """A flowering stalk: its bracts, red or white, set alternately up it, smaller toward the top."""
    P = spine(foot, -np.pi / 2 + lean, h, r.normal(0, 0.2), n=10, sway=0.04, r=r)
    stem(ez, r, P, 4.0, 2.0, haze(fam[2] * 0.8, hz), haze(fam[0] * 0.8, hz))
    C = brush.path(P, 2.0)
    T = np.gradient(C, axis=0)
    for i, t in enumerate(np.linspace(0.3, 1.0, n) + r.normal(0, 0.01, n)):
        j = min(int(t * (len(C) - 1)), len(C) - 1)
        last = i == n - 1
        a = ang(T[j]) + (0 if last else (-1) ** i * r.uniform(0.5, 0.95))
        L = size * (1.15 - 0.5 * t) * r.uniform(0.8, 1.2)
        Q = spine(C[j], a, L, (-1) ** i * r.uniform(-0.2, 0.4), n=8)
        k = r.uniform(1.0, len(fam) - 1.01)
        leaf(ez, r, Q, L * r.uniform(0.18, 0.26) * profile(kind, 8), haze(fam[int(k) + 1], hz),
             haze(fam[int(k) - 1], hz),
             lit=(-1) ** i, phi=0.2, band=3, rib=(haze(fam[0], hz), 1.2), shade=(0.15, 0.1), body=0.3)


def heart_plant(ez, r, foot, n, size, fam, tone, hz=0.0, reach=(0.5, 1.0), spread=1.1, lean=0.0):
    """Great heart-shaped leaves on long stalks from one root, each hanging from the end of its stalk,
    the veins and midrib painted pale over two greens."""
    for k in r.permutation(n):
        a = -np.pi / 2 + lean + (k / max(n - 1, 1) - 0.5) * 2 * spread + r.normal(0, 0.12)
        Ls = size * r.uniform(*reach)
        S = spine(foot, a, Ls, r.normal(0, 0.25), n=8)
        stem(ez, r, S, 6.0, 4.0, *tints("olive", 0.6, r, hz=hz))
        p = S[-1]
        la = a + toward(a, np.pi / 2) * r.uniform(0.35, 0.8)
        L = size * r.uniform(0.6, 0.9)
        Q = spine(p - 0.2 * L * np.array([np.cos(la), np.sin(la)]), la, L, r.normal(0, 0.3), n=16)
        light, dark = tints(fam, tone + r.normal(0, 0.08), r, step=2.0, hz=hz)
        wd = L * r.uniform(0.36, 0.44)
        rc = haze(RIB[r.integers(4)], hz)
        leaf(ez, r, Q, wd * profile("heart", 16), light, dark, lit=r.choice([-1, 1]), phi=0.95, curve=0.5,
             band=r.uniform(8, 11), notch=(0.24 * L, 0.45 * wd), rib=(rc, 0.03 * L),
             veins=((rc + 2 * light) / 3, r.uniform(0.09, 0.12) * L, 0.008 * L + 0.8, 0.5),
             shade=(0.18, 0.05), body=0.3, sides=(r.uniform(0.85, 1.1), r.uniform(0.85, 1.1)))


def pinnate(ez, r, foot, h, pairs, size, fam, tone, hz=0.0, lean=0.0):
    """A stem with its leaves in pairs, toothed, each drawn round in ochre and veined in red-brown."""
    P = spine(foot, -np.pi / 2 + lean, h, r.normal(0, 0.25), n=10, sway=0.04, r=r)
    stem(ez, r, P, 5.0, 2.5, *tints("grass", 0.7, r, hz=hz))
    C = brush.path(P, 2.0)
    T = np.gradient(C, axis=0)
    for i, t in enumerate(list(np.linspace(0.18, 0.92, pairs)) + [1.0]):
        j = min(int(t * (len(C) - 1)), len(C) - 1)
        for side in ((-1, 1) if t < 1 else (0,)):
            a = ang(T[j]) + side * r.uniform(0.7, 1.15)
            a += 0.25 * toward(a, np.pi / 2) * r.uniform(0, 1)
            L = size * (1.1 - 0.4 * t) * r.uniform(0.85, 1.1)
            Q = spine(C[j], a, L, r.normal(0, 0.25), n=10)
            light, dark = tints(fam, tone + r.normal(0, 0.1), r, step=1.8, hz=hz)
            wd = L * r.uniform(0.27, 0.34)
            leaf(ez, r, Q, wd * profile("ovate", 10), light, dark, lit=r.choice([-1, 1]), phi=0.8, curve=0.35,
                 band=max(3, 0.2 * wd), teeth=int(L / 16), serr=0.1, rib=(haze(RUST[r.integers(3)], hz), 0.025 * L + 1),
                 veins=(haze(RUST[r.integers(3)], hz), 0.1 * L, 1.4, 0.75),
                 rim=(haze(OCHRE[r.integers(4)], hz), 2.0, 0.9),
                 shade=(0.12, 0.06), body=0.28)


def sky_colour(y):
    y = np.asarray(y, float)
    return np.stack([np.interp(y, SKY_AT, SKY[:, ch]) for ch in range(3)], -1).astype(np.float32)


def sky(ez, r):
    """The sky, laid first over the whole canvas: a thin coat of its colours rubbed on, then long level
    strokes side by side, row under row, blue at the top, paler toward the foot and round the sun; the sun
    on it in round strokes of cream from its middle out; then the sky's last strokes taken round it,
    closing on its edge and dragging a little of its cream."""
    sh = (H, W)
    sx, sy, R = SUN
    yy = np.arange(H, dtype=np.float32)[:, None] + 30 * noise.field(sh, 500, r)
    xx = np.arange(W, dtype=np.float32)[None, :]
    coat = sky_colour(yy)
    coat = coat + (GLOW - coat) * (0.3 * np.exp(-(np.hypot(xx - sx, yy - sy) / (2.6 * R)) ** 2))[..., None]
    ez.rgb[:] = ez.rgb * 0.2 + coat * 0.8

    def mixed(P):
        """The sky's colour where each stroke goes, a shade off, and a second warmer or cooler streaked through it."""
        x, y = P[:, 0], P[:, 1]
        c = sky_colour(y) * np.exp(r.normal(0, 0.022, (len(P), 1))).astype(np.float32)
        c = c + (GLOW - c) * (0.3 * np.exp(-(np.hypot(x - sx, y - sy) / (2.6 * R)) ** 2))[:, None]
        c2 = c * np.where(r.random((len(P), 1)) < 0.5, [1.025, 1.0, 0.97], [0.975, 1.0, 1.03])
        return np.stack([c, c2], 1).astype(np.float32)

    def arcs(n, inner, outer, span):
        """Short strokes curving round the sun's middle, each at a radius of its own."""
        out = []
        for _ in range(n):
            rho = r.uniform(inner, outer)
            th = r.uniform(0, 2 * np.pi) + np.linspace(0, r.uniform(*span), 10)
            rr = rho + 0.03 * rho * np.sin(th * r.uniform(1, 3) + r.uniform(0, 6))
            P = np.stack([sx + rr * np.cos(th), sy + rr * np.sin(th)], 1)
            out.append(P if r.random() < 0.5 else P[::-1])
        return out

    wet = np.zeros(sh, np.float32)
    kw = dict(thick=0.035, grooves=0.08, lips=0.03, land=0.1, lift=0.05, ends=(0.5, 0.3), taper=0.4, fray=1.3,
              flatten=0.3, spent=0.5, pickup=0.25, merge=2.0, tails=0.4, hide=0.95, wet=wet)
    level = []
    for y in np.cumsum(r.uniform(10, 20, 110)) - 12:          # the rows not ruled: some closer, some wider
        if y > 1530:
            break
        x = -r.uniform(0, 300)
        while x < W:
            L = r.uniform(150, 440)
            yb = y + r.normal(0, 2.5)
            P = np.array([[x, yb], [x + L / 2, yb + r.normal(0, 2.0)], [x + L, yb + r.normal(0, 3.0)]])
            level.append(P if r.random() < 0.5 else P[::-1])
            x += L * r.uniform(0.72, 0.95)
    impasto.lay(ez.rgb, ez.height, level, r.uniform(8, 12, len(level)), mixed(np.array([p[1] for p in level])), r,
                **kw)
    sy0, sx0 = int(sy - R), int(sx - R)
    by, bx = np.mgrid[sy0:sy0 + int(2 * R) + 1, sx0:sx0 + int(2 * R) + 1]
    under = np.clip((R - 9 - np.hypot(bx - sx, by - sy)) / 2 + 0.5, 0, 1)[..., None]    # a first thin coat of cream
    box = ez.rgb[sy0:sy0 + int(2 * R) + 1, sx0:sx0 + int(2 * R) + 1]
    box += (SUNLIT[1] - box) * under
    sun = arcs(18, 0.05 * R, 0.5 * R, (0.6, 1.6)) + arcs(26, 0.45 * R, 0.85 * R, (0.5, 1.2)) \
        + arcs(26, R - 10, R - 4, (0.5, 1.1))
    mid = np.array([p[len(p) // 2] for p in sun])
    lightness = noise.smoothstep(-1.0, 1.0, ((mid[:, 0] - sx) * -0.6 + (mid[:, 1] - sy) * -0.8) / R)[:, None]
    c = (SUNLIT[2] + (SUNLIT[0] - SUNLIT[2]) * lightness) * np.exp(r.normal(0, 0.012, (len(sun), 1)))
    c2 = np.repeat(SUNLIT[1][None], len(sun), 0)
    impasto.lay(ez.rgb, ez.height, sun, r.uniform(7, 9, len(sun)), np.stack([c, c2], 1).astype(np.float32), r,
                **{**kw, "thick": 0.03, "grooves": 0.03, "pickup": 0.1, "hide": 0.98})
    ring = arcs(14, R + 4, 1.2 * R, (0.5, 1.0))
    impasto.lay(ez.rgb, ez.height, ring, r.uniform(8, 11, len(ring)), mixed(np.array([p[5] for p in ring])), r,
                **{**kw, "pickup": 0.35, "merge": 2.5})


def ground(ez, r, mask):
    """The dark under-painting of the thicket, before any leaf goes on it: a coat of deep green rubbed in,
    then worked over with a big brush every way in darker and lighter greens, so that where it shows
    between the leaves it reads as more leaves in shadow; darkest where the thicket falls back into a
    recess."""
    sh = (H, W)
    t = np.clip(0.5 + 0.35 * noise.field(sh, 160, r), 0, 1)[..., None]
    col = GREENS["deep"][1] * (1 - t) + GREENS["deep"][3] * t
    m = mask[..., None]
    ez.rgb[:] = ez.rgb * (1 - m) + col * m
    ez.thin[:] = ez.thin * (1 - mask) + 0.4 * mask
    P = dabs.scatter(sh, 60, r)
    P = P[mask[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)] > 0.5]
    n = len(P)
    a = r.uniform(-np.pi, 0, n)
    L = r.uniform(110, 280, n)
    e = np.stack([np.cos(a), np.sin(a)], 1)
    bow = np.stack([-e[:, 1], e[:, 0]], 1) * (r.normal(0, 0.12, n) * L)[:, None]
    paths = np.stack([P, P + e * L[:, None] / 2 + bow, P + e * L[:, None]], 1)
    deep = np.clip(-glade(P[:, 0], P[:, 1]), 0, 1)
    pick = np.clip(r.normal(1.8, 1.0, n) - 1.6 * deep, 0, 4.99)
    G = np.concatenate([GREENS["deep"][:4], GREENS["teal"][1:3]])
    c1 = G[pick.astype(int)] * np.exp(r.normal(0, 0.06, (n, 1)))
    c2 = G[np.clip(pick + r.normal(0, 0.8, n), 0, 5).astype(int)]
    impasto.lay(ez.rgb, ez.height, list(paths), r.uniform(14, 28, n), np.stack([c1, c2], 1).astype(np.float32), r,
                thick=0.04, grooves=0.1, lips=0.03, land=0.15, lift=0.08, ends=(0.4, 0.3), taper=0.4, fray=1.5,
                spent=0.5, pickup=0.3, merge=2.0, tails=0.4, hide=0.9, dry=r.uniform(0, 0.3, n))


def laurel(ez, r, foot, s, fam, tone, hz):
    """One or two leafy stalks, the leaves set alternately up them."""
    kind = ["ovate", "lance", "ovate", "round"][r.integers(4)]
    for _ in range(r.integers(1, 3)):
        h = s * r.uniform(1.6, 2.6)
        P = spine(foot + np.array([r.normal(0, 0.2 * s), 0]), -np.pi / 2 + r.normal(0, 0.35), h, r.normal(0, 0.5), n=8,
                  sway=0.05, r=r)
        stem(ez, r, P, max(2.5, 0.02 * s + 1.5), 1.5, *tints("olive", 0.5, r, hz=hz))
        sprig(ez, r, P, s, r.integers(4, 8), fam, tone, kind, hz, veined=0.15, spread=(0.5, 1.2), hang=0.4,
              shape=(0.28, 0.4), step=1.8, first=0.2)


def rubber(ez, r, foot, s, fam, tone, hz):
    """A rubber plant: a stout stalk and a few great oval leaves held up from it, dark and glossy,
    each with a pale midrib."""
    h = s * r.uniform(1.8, 2.6)
    P = spine(foot, -np.pi / 2 + r.normal(0, 0.2), h, r.normal(0, 0.3), n=8, sway=0.04, r=r)
    stem(ez, r, P, 0.04 * s + 2, 2.5, *tints("olive", 0.4, r, hz=hz))
    sprig(ez, r, P, 1.25 * s, r.integers(3, 6), fam, tone, "round", hz, spread=(0.35, 0.9), hang=0.2,
          shape=(0.34, 0.42), step=1.4, first=0.3)


def dracaena(ez, r, foot, s, fam, tone, hz):
    """A dracaena or a young palm: long narrow leaves thrown out every way from the top of a short stalk,
    the longer ones bowing over."""
    top = np.asarray(foot, float) - [r.normal(0, 0.1 * s), s * r.uniform(0.3, 1.0)]
    stem(ez, r, [foot, (np.asarray(foot) + top) / 2, top], 0.03 * s + 2, 2.0, *tints("olive", 0.45, r, hz=hz))
    n = r.integers(8, 15)
    for a in r.permutation(np.linspace(-np.pi + 0.15, -0.15, n) + r.normal(0, 0.12, n)):
        L = s * r.uniform(1.4, 2.4)
        P = spine(top, a, L, np.sign(np.cos(a)) * r.uniform(0.3, 1.3), n=10, sway=0.03, r=r)
        wd = L * r.uniform(0.06, 0.09)
        light, dark = tints(fam, tone + r.normal(0, 0.1), r, step=1.6, hz=hz)
        leaf(ez, r, P, wd * profile("lance", 10), light, dark, lit=r.choice([-1, 1]), phi=0.1, curve=0.1,
             band=max(3, 0.45 * wd), rib=(light * 1.25, max(1.0, 0.12 * wd)), shade=(0.15, 0.1), body=0.25)


def canna(ez, r, foot, s, fam, tone, hz):
    """Broad pointed leaves rising from the root and arching, wrapped round one another at the foot."""
    n = r.integers(3, 6)
    for a in r.permutation(-np.pi / 2 + np.linspace(-0.7, 0.7, n) + r.normal(0, 0.15, n)):
        L = s * r.uniform(1.6, 2.4)
        P = spine(np.asarray(foot) + [r.normal(0, 8), 0], a, L, np.sign(a + np.pi / 2) * r.uniform(0.2, 0.9), n=12,
                  sway=0.03, r=r)
        wd = L * r.uniform(0.15, 0.2)
        light, dark = tints(fam, tone + r.normal(0, 0.1), r, step=2.0, hz=hz)
        leaf(ez, r, P, wd * profile("lance", 12), light, dark, lit=r.choice([-1, 1]), phi=0.55, curve=0.4,
             band=max(4, 0.25 * wd), rib=(light * 1.2, max(1.5, 0.06 * wd)), veins=(light * 1.12, 0.07 * L, 1.0, 0.3),
             shade=(0.18, 0.1), body=0.28)


def fern(ez, r, foot, s, fam, tone, hz):
    """A fern: fronds rising from the root and arching out, each of many small leaflets."""
    n = r.integers(4, 7)
    for a in r.permutation(-np.pi / 2 + np.linspace(-1.0, 1.0, n) + r.normal(0, 0.12, n)):
        frond(ez, r, foot, a, s * r.uniform(1.6, 2.4), np.sign(np.cos(a)) * r.uniform(0.4, 1.0), fam, tone,
              0.32 * s, 0.045 * s, int(np.clip(s / 7, 14, 28)), spread=1.1, droop=0.3, hz=hz)


def aroid(ez, r, foot, s, fam, tone, hz):
    heart_plant(ez, r, foot, r.integers(2, 5), 1.6 * s, fam, tone, hz=hz, spread=0.9)


def shrub(ez, r, foot, s, fam, tone, hz):
    """A shrub of small leaves on twigs."""
    tree(ez, r, foot, np.asarray(foot) + [r.normal(0, 0.3 * s), -s * r.uniform(1.0, 1.6)], 5, fam, tone, 0.3 * s,
         depth=3, fork=(2, 4), reach=(0.5, 0.75), twig=7, hz=hz, spread=0.9, lift=0.25, first=0.3)


def sedge(ez, r, foot, s, fam, tone, hz):
    clump(ez, r, foot, r.integers(8, 15), 2.2 * s, 0.06 * s + 2, fam, tone, spread=0.45, droop=0.8, rim=0.6, hz=hz)


SPECIES = {"laurel": laurel, "rubber": rubber, "dracaena": dracaena, "canna": canna, "fern": fern, "aroid": aroid,
           "shrub": shrub, "sedge": sedge}


def thicket(ez, r, rows, dim=0.0):
    """The thicket built up row by row from the back: each row a line of plants of a few kinds standing a
    little lower and nearer than the one before and covering its feet, dark and light by turns. It is not
    the same everywhere: in a few places it falls back into a deep recess, few plants and those sunk in
    shadow, where the eye rests; in a few others broad leaves stand together in the light (`glade`).
    A row: (the height its leaves stand at, from x, to x, how many plants, leaf size, kinds and their odds,
    families, tone, haze)."""
    for y, x0, x1, n, size, kinds, fams, tone, hz in rows:
        names, odds = list(kinds), np.array(list(kinds.values()), float)
        for x in np.sort(r.uniform(x0, x1, n))[r.permutation(n)]:
            s = size * r.uniform(0.75, 1.25)
            kind = names[r.choice(len(names), p=odds / odds.sum())]
            fam = fams[r.integers(len(fams))]
            z = float(np.clip(glade(x, y - 0.8 * s), -1, 1))
            if r.random() < -0.3 * z:
                continue
            if r.random() < z:
                kind = ("aroid", "canna", "rubber")[r.integers(3)]
                fam = ("grass", "yellow", "sage")[r.integers(3)]
                s *= 1 + 0.3 * z
            foot = np.array([x, y + r.normal(0, 0.15 * size) + 1.1 * s])
            ez.dusk = max(dim, 0.6 * max(-z, 0.0))
            SPECIES[kind](ez, r, foot, s, fam, tone + 0.3 * z + r.normal(0, 0.07), hz)
            ez.dusk = 0.0


def paint(seed=1910):
    base = canvas.duck((H, W), seed, tint="#dcd3bb", thread=3.4)
    ez = Easel(base.color.copy(), np.zeros((H, W), np.float32), noise.rng(seed))
    g = lambda name: fresh(seed, name)

    sky(ez, g("sky"))
    yy = np.arange(H, dtype=np.float32)[:, None]
    edge = yy + 40 * noise.field((H, W), 220, g("under"))
    ground(ez, g("ground"), noise.smoothstep(1478, 1482, edge).astype(np.float32))

    # the far row: light trees along the foot of the sky, a palm standing up over them
    r = g("far")
    for x in (1020, 1190, 1370, 1550, 1740, 1930, 2110, 2290, 2440):
        x += r.normal(0, 35)
        y = 1040 + r.normal(0, 25)
        h = r.uniform(300, 440)
        tree(ez, r, (x, y + 80), (x + r.normal(0, 20), y - 0.4 * h), 7, ["grass", "yellow", "sage"][r.integers(3)],
             r.uniform(0.62, 0.85), r.uniform(22, 32), depth=3, fork=(2, 4), reach=(0.5, 0.75), twig=8, hz=0.06,
             spread=0.9, lift=0.15, first=0.4)
    palm(ez, r, (1130, 1150), (1150, 610), 10, 9, "teal", 0.6, 360, 120, 6.0, hz=0.1, n=24)

    # the gap in the canopy: a tall tree at the left edge and boughs hanging in from above, in darker greens
    tree(ez, g("tall tree"), (-40, 1500), (150, 380), 36, ["emerald", "teal"], 0.45, 58, depth=4, fork=(3, 5),
         reach=(0.45, 0.65), twig=7, spread=0.7, lift=0.05, first=0.6, kind=["ovate", "round", "lance"], fan=1.1,
         crown=430)
    tree(ez, g("bough"), (420, -160), (1180, 230), 16, ["emerald", "grass"], 0.5, 50, depth=3, fork=(3, 5),
         reach=(0.45, 0.7), twig=7, spread=0.8, lift=-0.2, first=0.3, kind=["ovate", "heart", "ovate"])

    # the big trees: an orange tree on the left, a tree of heart-shaped leaves on the right, a palm
    tree(ez, g("orange tree"), (470, 1750), (540, 800), 24, "emerald", 0.55, 52, depth=4, fork=(3, 5),
         reach=(0.45, 0.65), twig=7, fruit=0.2, fruit_r=(19, 34), spread=0.7, lift=0.05, first=0.7,
         kind=["ovate", "lance", "ovate"], fan=1.4, crown=480, veined=0.2)
    tree(ez, g("right tree"), (3020, 1750), (2930, 760), 34, ["yellow", "grass", "yellow"], 0.6, 95,
         kind=["heart", "ovate", "heart"], depth=3, fork=(3, 5), reach=(0.5, 0.7), twig=6, spread=0.7, lift=0.05,
         first=0.6, shape=(0.36, 0.46), fan=1.2, crown=560, veined=0.7)
    palm(ez, g("palm"), (2620, 1750), (2470, 320), 26, 11, "emerald", 0.5, 700, 215, 9.0, n=34)

    # the depth of the garden: dark rows of every kind of plant, one in front of another
    thicket(ez, g("thicket"), [
        (990, 880, 2480, 18, 70, {"laurel": 2, "dracaena": 1, "shrub": 1}, ["teal", "sage", "emerald"], 0.45, 0.08),
        (1090, -100, 3300, 28, 105, {"laurel": 3, "dracaena": 2, "rubber": 0.5, "fern": 1}, ["deep", "teal", "emerald"],
         0.4, 0.0),
        (1270, -100, 3300, 26, 125, {"laurel": 2, "canna": 2, "dracaena": 2, "aroid": 1, "fern": 1},
         ["deep", "teal", "emerald", "olive"], 0.42, 0.0),
        (1460, -100, 3300, 28, 140, {"laurel": 2, "dracaena": 2, "aroid": 1, "rubber": 0.5, "fern": 1},
         ["deep", "teal", "emerald"], 0.36, 0.0),
    ])
    thicket(ez, g("shadow"), [(1660, 850, 2150, 9, 140, {"aroid": 2, "laurel": 2, "rubber": 1}, ["deep", "teal"], 0.3,
                               0.0)], dim=0.45)

    # in front of it, lit: the bananas, the lotus, a young palm, a fern
    banana(ez, g("banana"), (1140, 1960), 520, 7, "yellow", 0.62, 650, 90)
    dracaena(ez, g("young palm"), np.array([2020, 1900]), 230, "sage", 0.75, 0.0)
    banana(ez, g("banana2"), (2180, 2000), 380, 5, "grass", 0.68, 480, 74)
    fern(ez, g("fern"), np.array([2700, 1980]), 210, "grass", 0.65, 0.0)
    r = g("lotus")
    for c, R, fam, foot in (((1440, 1330), 120, PINK, (1460, 2050)), ((1650, 1490), 95, WHITE, (1640, 2050)),
                            ((1900, 1300), 112, PINK, (1880, 2050)), ((2620, 1460), 88, BLUE, (2600, 2050)),
                            ((590, 1440), 105, PINK, (600, 2050))):
        lotus(ez, r, c, R, fam, a=-np.pi / 2 + r.normal(0, 0.1), stalk=foot)

    # a dark row again, the white lilies and the red spikes standing up in front of it, a lit row, then the
    # foreground: great heart leaves at the left, toothed leaves at the right, agaves and grass along the foot
    thicket(ez, g("near"), [(1800, -100, 3300, 24, 160, {"aroid": 2, "laurel": 2, "dracaena": 1},
                             ["deep", "emerald", "teal"], 0.32, 0.0)])
    for i, (x, h, n, size, lean) in enumerate(((775, 520, 17, 52, -0.06), (880, 380, 11, 44, 0.14))):
        spike(ez, g(f"white{i}"), (x, 2020 + 25 * i), h, WHITE, n, size, lean=lean, kind="petal")
    for i, (x, y, h, n, size, lean) in enumerate(((2250, 2010, 610, 15, 84, -0.1), (2350, 1990, 380, 8, 78, 0.05),
                                                  (2470, 2030, 500, 12, 66, 0.18))):
        spike(ez, g(f"spike{i}"), (x, y), h, RED, n, size, lean=lean)
    thicket(ez, g("near2"), [(2050, -100, 3300, 20, 150, {"fern": 2, "canna": 2, "sedge": 2, "dracaena": 1},
                              ["grass", "yellow", "sage"], 0.6, 0.0)])
    heart_plant(ez, g("hearts"), (200, 2520), 7, 560, "emerald", 0.5, spread=1.0, lean=0.25)
    pinnate(ez, g("pinnate"), (2950, 2500), 820, 6, 250, "grass", 0.5, lean=-0.1)
    clump(ez, g("agave l"), (790, 2540), 18, 620, 44, "sage", 0.9, spread=0.5, droop=0.3, rim=1.0, rib=0.5, stiff=0.6)
    clump(ez, g("agave r"), (2430, 2540), 16, 560, 40, "sage", 0.82, spread=0.5, droop=0.3, rim=1.0, rib=0.5, stiff=0.6)
    r = g("grass")
    for x in np.linspace(940, 2320, 11) + r.normal(0, 45, 11):
        clump(ez, r, (x, 2490), 26, r.uniform(380, 560), 13, ["yellow", "grass", "emerald"][r.integers(3)],
              r.uniform(0.5, 0.85), spread=0.5, droop=1.0, rim=0.75)
    for x in (60, 3150):
        clump(ez, r, (x, 2490), 18, 420, 13, "yellow", 0.6, spread=0.5, droop=1.0, rim=0.75)

    # the weave through the thin paint: its crests hold less of it and show the ground, most where the paint
    # is pale and thin, and the whole surface follows the weave under the light
    crest = noise.smoothstep(0.2, 0.6, base.tooth)
    show = 0.35 * ez.thin * crest * noise.smoothstep(0.0, 0.45, ez.rgb @ LUMA)
    rgb = (ez.rgb + (base.color - ez.rgb) * show[..., None]) * np.float32([1.0, 0.98, 0.92])   # and yellowed a little
    height = ndimage.gaussian_filter(ez.height, 0.6) + 0.6 * base.tooth * (0.5 + ez.thin)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.82, 1.14))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
