"""Irises. Oil on canvas, in thick strokes from a loaded brush, every flower drawn round in blue.

In May 1889, in his first week in the asylum of Saint-Paul-de-Mausole at Saint-Rémy, before
he was let out beyond its walls, Van Gogh painted a bed of irises in the garden there. He
saw it close and from a little above, so that the bed fills the canvas, with a strip of
red earth at the foot and a band of yellow-green meadow at the top. He drew it as he had
learned from the Japanese prints he collected: every leaf and every flower silhouetted and
drawn round with the brush, the colours laid flat inside. Each flower is a few strokes of
ultramarine and violet, curling with the petal, the falls darker and veined, a touch of
yellow at the throat; each leaf is a few long strokes of viridian and blue-green dragged
from the root to the point, crossing and twisting; the earth is short thick touches. Among
the blue there is one white iris. His red lake has faded since, so the violets have gone
bluer, and the canvas has yellowed.

This bed is not his but is painted his way.
"""

import numpy as np
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Irises in the Asylum Garden"
DATE = "2026"
MEDIUM = "Oil on canvas, in thick strokes from a loaded brush"
AFTER = "Vincent van Gogh, Irises, Saint-Rémy, May 1889"
ROOM = "The Garden"
YEAR = 1889
PLACE = "Saint-Rémy-de-Provence"
REGION = "Europe"
NOTE = ("A bed of irises seen close and from above, each flower a few curling strokes of blue drawn round "
        "in a darker one, each leaf a few long strokes of blue-green. One iris is white.")

H, W = 2300, 3000
LIGHT = (-0.6, -0.5, 0.62)
SUN = np.array([-0.77, -0.64])


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each list from dark to light
BLUE = pal("#171c55", "#1e2768", "#26337d", "#2f4091", "#3a4fa3", "#4860b3", "#5a72bf", "#6f86c9", "#879bd2",
           "#a2b2dc", "#bfc8e3")                                                     # ultramarine, cobalt, white
VIOLET = pal("#1e1c52", "#2b2768", "#3a347e", "#4a4592", "#5d58a4", "#7470b4", "#8e8bc4")
INK = pal("#0e1236", "#141a47", "#1b2458", "#222f66")                                # ultramarine and a little black
CHROME = pal("#c98a1b", "#d9a126", "#e4b73c", "#ecca5e")
WHITE = pal("#9fa98f", "#b5bca3", "#cbcfb8", "#dcdcc7", "#e8e6d4", "#f1eee0")          # lead white, a little yellowed
LEAF = pal("#24524c", "#2f655a", "#3c7866", "#4b8a73", "#5d9b80", "#72ac8f", "#88bc9e", "#a0caae", "#b8d6c0")
LEAFY = pal("#8fa860", "#a6b96d", "#bcc77f")                                         # yellow-green, streaked in
EDGE = pal("#12283a", "#17324a", "#1e3d55", "#24485a")
DEEP = pal("#13292b", "#183433", "#1f3f3c", "#28504a", "#2f5c55")
EARTH = pal("#57261a", "#6c3021", "#823d29", "#964b33", "#a75b40", "#b56e55", "#c2846b", "#cc9a85", "#d5b09e")
RED = pal("#b8462a", "#c55a2e")
MAUVE = pal("#6c4552", "#83596a")
MEADOW = pal("#4c6b28", "#5f7f2d", "#738f33", "#88a33a", "#9cb444", "#afc153", "#c0cd66", "#cfd77c")
GRASS = pal("#2f4f2a", "#3d6030", "#4c7236")
SPECK = pal("#e3c53c", "#ebd765", "#efe9cf", "#e39b2d")
HEDGE = pal("#1d3628", "#25422f", "#2f5036", "#3a5e3d", "#486d45", "#5a7c4c")
ORANGE = pal("#b54a17", "#cc5f1d", "#dc7624", "#e68d2e", "#eda43f")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a neighbour on the palette streaked in,
    and a third, now and then an accent picked up from elsewhere."""
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


def sheaves(field, seeds, length, width, k, r, n=8, bend=0.0, tilt=0.0):
    """Strokes side by side in little sheaves: from each seed, k strokes about a brush-width apart
    along the same field, their starts staggered. -> paths, half-widths, seed of each"""
    N = len(seeds)
    a = field[at(seeds)] + tilt
    d = np.stack([np.cos(a), np.sin(a)], -1)[:, None, :]
    across = np.stack([-np.sin(a), np.cos(a)], -1)[:, None, :]
    j = np.arange(k) - (k - 1) / 2
    L = np.asarray(length)[:, None] * np.exp(r.normal(0, 0.25, (N, k)))
    w = np.asarray(width)[:, None] * np.exp(r.normal(0, 0.15, (N, k)))
    P = seeds[:, None, :] + across * (1.8 * w * (j + r.normal(0, 0.12, (N, k))))[..., None] \
        + d * (L * r.uniform(-0.15, 0.15, (N, k)))[..., None]
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,))[:, None].repeat(k, 1)
    tl = np.broadcast_to(np.asarray(tilt, np.float32), (N,))[:, None].repeat(k, 1)
    return impasto.follow(field, P.reshape(-1, 2), L.ravel(), n, turn.ravel(), tl.ravel()), w.ravel(), np.repeat(np.arange(N), k)


def spacing(k, length, width, cover):
    """Seeds far enough apart that sheaves of k strokes cover the ground about `cover` times."""
    return float(np.sqrt(k * length * 2 * width / (cover * 0.866)))


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


def edges(r):
    """Where the meadow stops at the top of the bed, and where the earth starts at its foot, for every x."""
    x = np.arange(W)
    top = 330 + 38 * noise.line1d(W, 420, r) + 10 * noise.line1d(W, 60, r)
    foot = np.interp(x, [0, 500, 1000, 1500, 2000, 2500, 3000], [1790, 1850, 1960, 1900, 1950, 2070, 2110]) \
        + 22 * noise.line1d(W, 220, r) + 7 * noise.line1d(W, 45, r)
    hedge = np.clip(30 + 50 * noise.line1d(W, 520, r) + 12 * noise.line1d(W, 70, r), 0, None) * noise.smoothstep(2500, 1300, x)
    return top, foot, hedge


# ---- the flowers in the round. Each petal is a spine (rho out from the stem, h up it) with a width
# across it; the stem is the z axis, the throat at the origin. Rows from the claw to the tip.

FALL = np.array([(0.03, 0.05), (0.3, 0.1), (0.52, -0.02), (0.62, -0.25), (0.6, -0.5)])
DROOP = np.array([(0, 0), (0, -0.02), (-0.02, -0.12), (-0.05, -0.25), (-0.1, -0.32)])
FALL_SHUT = np.array([(0.03, 0.05), (0.08, 0.3), (0.12, 0.55), (0.11, 0.8), (0.06, 1.0)])
STANDARD = np.array([(0.03, 0.08), (0.17, 0.36), (0.22, 0.62), (0.16, 0.86), (0.05, 1.0)])
STANDARD_SHUT = np.array([(0.02, 0.05), (0.06, 0.35), (0.07, 0.65), (0.04, 0.85), (0.0, 1.0)])
STYLE = np.array([(0.02, 0.06), (0.1, 0.1), (0.18, 0.12), (0.24, 0.11), (0.28, 0.08)])
LAMP = np.array([-0.5, -0.45, 0.75]) / np.linalg.norm([-0.5, -0.45, 0.75])   # y runs away from us, z up


def iris(o, droop, age, r):
    """An iris: three falls arching out and hanging, three standards cupped upright over them,
    and over the claw of each fall a little style arm. `o` is how far it has opened (0 shut, 1
    open), `droop` how far the falls hang, `age` how far it has gone over: the standards flop
    open and the falls hang limp and twisted. No two petals alike: each turns off its place,
    reaches further or less, is longer or shorter, twists along its length and ruffles at its edge.
    -> [(spine (n,3), across (n,3), normal (n,3), kind)], kind 0 fall, 1 standard, 2 style arm"""
    s = np.linspace(0, 1, 16)
    u = np.linspace(0, 1, 5)
    phi0 = r.uniform(0, 2 * np.pi)
    prof = (0.18 + 0.82 * noise.smoothstep(0.02, 0.55, s)) * np.sqrt(np.clip(1 - (np.clip(s - 0.55, 0, 1) / 0.45) ** 2, 0, 1))
    out = []
    for kind, half, length in ((0, 0.35, 0.88), (1, 0.28, 0.8), (2, 0.08, 1.0)):
        if kind == 2 and o < 0.6:
            continue
        for i in range(3):
            if kind == 0:
                d = np.clip(droop + 0.5 * age + r.normal(0, 0.4), -0.3, 1.3)
                shape = o * (FALL * [r.uniform(0.8, 1.3), 1] + d * DROOP) + (1 - o) * FALL_SHUT
            elif kind == 1:
                shape = o * STANDARD * [r.uniform(0.7, 1.5) * (1 + 0.9 * age), r.uniform(0.8, 1.15) * (1 - 0.35 * age)] \
                    + (1 - o) * STANDARD_SHUT
            else:
                shape = STYLE
            phi = phi0 + 2 * np.pi * i / 3 + np.pi / 3 * (kind == 1) + r.normal(0, 0.1 if kind == 2 else 0.3)
            L = length * r.uniform(0.78, 1.18)
            rho = ndimage.gaussian_filter1d(np.interp(s, u, shape[:, 0]), 1.0, mode="nearest") * L
            h = ndimage.gaussian_filter1d(np.interp(s, u, shape[:, 1]), 1.0, mode="nearest") * L
            p = np.stack([rho * np.cos(phi), rho * np.sin(phi), h], 1)
            T = np.gradient(p, axis=0)
            T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
            t0 = np.array([-np.sin(phi), np.cos(phi), 0.0])
            tw = np.clip(r.normal(0, 0.45) + (r.random() < 0.1 + 0.25 * age) * r.choice([-1.0, 1.0]), -1.1, 1.1) * s
            t = t0 * np.cos(tw)[:, None] + np.cross(T, t0) * np.sin(tw)[:, None]
            w = half * prof * (0.5 + 0.5 * o) * (1 - 0.15 * age) * r.uniform(0.8, 1.2) \
                * (1 + 0.12 * s * np.sin(2 * np.pi * r.uniform(1.5, 3) * s + r.uniform(0, 6)))
            out.append((p, t * w[:, None], np.cross(T, t), kind))
    return out


def seen(yaw, tilt, toward, pitch):
    """How a flower is seen: turned on its stem by `yaw`, the stem leaning by `tilt` toward the
    bearing `toward`, and looked down on at `pitch`. -> rows: right, up the canvas, toward the eye"""
    cz, sz = np.cos(yaw), np.sin(yaw)
    k = np.array([-np.sin(toward), np.cos(toward), 0.0])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M = (np.eye(3) + np.sin(tilt) * K + (1 - np.cos(tilt)) * K @ K) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    cam = np.array([[1, 0, 0], [0, np.sin(pitch), np.cos(pitch)], [0, -np.cos(pitch), np.sin(pitch)]])
    return cam @ M


def leaf(base, a0, L, w, r):
    """A sword leaf from its root: rising, bending away from upright as it goes and now and then
    broken and hanging, and turning on itself, so that it narrows where it is seen edge-on and
    beyond shows its other face. -> spine (n,2), half-width across it (n,2), which face shows (n,)"""
    s = np.linspace(0, 1, 28)
    side = np.sign(a0 + np.pi / 2 + 1e-6)
    a = a0 + side * r.uniform(0.1, 0.6 + 1.6 * abs(a0 + np.pi / 2)) * s ** 1.6 + r.normal(0, 0.12) * np.sin(np.pi * s)
    if r.random() < 0.22:
        f = r.uniform(0.4, 0.75)
        a = a + side * r.uniform(0.7, 1.5) * noise.smoothstep(f - 0.04, f + 0.04, s)
    a = np.clip(a + np.pi / 2, -2.1, 2.1) - np.pi / 2         # a broken leaf hangs, but does not curl back on itself
    d = np.stack([np.cos(a), np.sin(a)], 1)
    S = base + np.vstack([[0, 0], np.cumsum(d[:-1] * L / 27, 0)])
    q = np.cos(r.uniform(-0.9, 0.9) + r.normal(0, 1.0) * s)
    half = w * (0.75 + 0.25 * noise.smoothstep(0, 0.12, s)) * (1 - s ** 2.4) ** 0.65 * (0.3 + 0.7 * np.abs(q))
    return S, np.stack([-d[:, 1], d[:, 0]], 1) * half[:, None], np.sign(q)


FAMILIES = {      # each paint: its palette, the accents picked up into it, and how often
    "blue": (BLUE, VIOLET[2:6], 0.2),
    "violet": (VIOLET, BLUE[3:8], 0.3),
    "white": (WHITE, pal("#c4cba9", "#d5d9bd", "#aab8b4"), 0.3),
    "ink": (INK, VIOLET[:2], 0.25),
    "olive": (pal("#3c4a47", "#4b5853", "#5c6860"), INK[2:], 0.2),
    "chrome": (CHROME, ORANGE[2:], 0.35),
    "leaf": (LEAF, np.concatenate([LEAFY, BLUE[4:6]]), 0.14),
    "leafy": (LEAF, LEAFY, 0.45),
    "orange": (ORANGE, CHROME[2:], 0.3),
    "heart": (pal("#6e2410", "#8a3014", "#a63f18"), ORANGE[:2], 0.3),
    "marileaf": (pal("#1f3a26", "#2a4c2e", "#365e36", "#46703f"), LEAF[1:3], 0.2),
    "edge": (EDGE, LEAF[:2], 0.25),
    "stem": (LEAF[1:7], LEAFY, 0.25),
    "spathe": (pal("#3f5f3f", "#4f6e44", "#62804c", "#78905a"), LEAF[3:5], 0.3),
}
EYE_LIGHT = np.array([-0.5, 0.45, 0.75]) / np.linalg.norm([-0.5, 0.45, 0.75])   # right, up, toward the eye


class Strokes:
    """Strokes gathered in the order the brush lays them, each with its paint named by family and tone."""

    def __init__(self):
        self.paths, self.w, self.tone, self.fam, self.dry = [], [], [], [], []

    def add(self, path, w, tone, fam, dry=0.0):
        if len(path) >= 2:
            self.paths.append(np.asarray(path, np.float32))
            self.w.append(w)
            self.tone.append(tone)
            self.fam.append(fam)
            self.dry.append(dry)

    def load(self, r):
        n, fam, tone = len(self.paths), np.array(self.fam), np.array(self.tone)
        cols, share = np.empty((n, 3, 3), np.float32), np.empty((n, 3))
        for f in set(self.fam):
            sel = fam == f
            colours, accent, odds = FAMILIES[f]
            cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds)
        return cols, share


def flower(st, c, R, look, r):
    """An iris in strokes, petal by petal from the back: each petal a few strokes of the loaded
    brush curling along it from the claw, the falls veined dark, a pale stroke where the light
    catches, the petal drawn round, and on each fall that faces us a touch of yellow at the throat.
    `look`: how open, how the falls hang, how old, how it is seen, the paints of falls, standards
    and outline, and how deep in shadow it stands."""
    o, droop, age, view, falls, standards, line, shade = look
    sl = np.linspace(0, 1, 16)
    petals = []
    for p, across, nrm, kind in iris(o, droop, age, r):
        P, A, Nc = p @ view.T, across @ view.T, nrm @ view.T
        cup = (nrm * np.linalg.norm(across, axis=1, keepdims=True)) @ view.T
        petals.append((P[:, 2].mean(), c + R * P[:, :2] * [1, -1], R * A[:, :2] * [1, -1], R * cup[:, :2] * [1, -1], Nc, kind))
    hw = np.clip(0.1 * R, 6, 20)
    for _, S, V, C, Nc, kind in sorted(petals, key=lambda e: e[0]):
        wv = np.linalg.norm(V, axis=1)
        if wv.max() < 1.5:
            continue
        paint_ = falls if kind == 0 else standards
        thin = wv.max() < 0.12 * R
        facing = np.sign(Nc[:, 2] + 1e-6)
        lam = ((Nc * facing[:, None]) @ EYE_LIGHT).mean()
        tone = (0.5, 0.68, 0.72)[kind] + 0.35 * (lam - 0.4) - shade - 0.15 * age
        lat = np.sign(V.mean(0) @ SUN + 1e-6)
        k = int(np.clip(round(2 * wv.max() / (1.5 * hw)), 1, 6))
        bow = (-0.3, 0.3, 0.0)[kind]
        b = min(hw, 0.6 * wv.max() + 2)
        inside = np.clip(1 - 0.75 * b / (wv + 1e-6), 0, 1)[:, None]   # so the strokes keep within the outline
        for f in np.linspace(-1, 1, k) * (1 - 1 / (k + 1)) + r.normal(0, 0.05, k):
            path = (S + f * V * inside + bow * f * f * C)[(sl >= r.uniform(0, 0.12)) & (sl <= r.uniform(0.82, 1.0))]
            st.add(path if r.random() < 0.7 else path[::-1], b * r.uniform(0.85, 1.15), tone + 0.18 * f * lat, paint_)
        if kind == 0 and paint_ != "white" and not thin and facing[2:8].mean() > 0:
            for f in r.uniform(-0.5, 0.5, r.integers(2, 5)):
                st.add((S + f * V + bow * f * f * C)[(sl >= 0.1) & (sl <= r.uniform(0.45, 0.8))], max(1.8, 0.25 * hw),
                       tone - 0.35, paint_, r.uniform(0.1, 0.5))
        if lam > 0.2 and kind < 2:
            f = lat * r.uniform(0.25, 0.55)
            st.add((S + f * V + bow * f * f * C)[(sl >= 0.25) & (sl <= 0.85)], 0.55 * b, tone + 0.4, paint_, 0.3)
        E = (S - lat * V + bow * C)[2:] if thin else np.vstack([(S + V + bow * C)[2:], (S - V + bow * C)[2:][::-1]])
        per = np.hypot(*np.diff(E, axis=0).T).sum()
        pale = line == "olive"            # the white iris is drawn round more lightly
        for m in pieces(E, per * r.uniform(0.35, 0.6), 10, r, keep=(0.95, 0.7)[pale]) if kind < 2 and not (thin and pale) else []:
            st.add(E[m], min(np.clip(0.028 * R, 2.5, 6.5), 0.2 * wv.max() + 1) * (1, 0.65)[pale] * np.exp(r.normal(0, 0.15)),
                   r.uniform(0.2, 0.8), line)
        if kind == 0 and o > 0.5 and facing[2:6].mean() > 0:
            st.add((S + 0.12 * C)[(sl >= 0.04) & (sl <= r.uniform(0.3, 0.42))], np.clip(0.032 * R, 2.8, 6.0),
                   r.uniform(0.3, 0.7), "chrome")
            if r.random() < 0.5:
                f = r.choice([-0.3, 0.3])
                st.add((S + f * V + 0.1 * C)[(sl >= 0.08) & (sl <= 0.3)], np.clip(0.03 * R, 2.5, 5), 0.92, standards, 0.2)


def bud(st, c, R, a, paint_, r):
    """A bud: a spindle of blue wound in on itself, in a green sheath, pointing up."""
    s = np.linspace(0, 1, 14)
    d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    S = c + np.outer(R * 1.1 * s, d) + np.outer(0.08 * R * np.sin(np.pi * s) * r.choice([-1, 1]), n)
    half = 0.2 * R * np.sin(np.pi * np.clip(s, 0, 1) ** 0.75) ** 0.9
    for f in (-0.45, 0.45):
        twist = f + 0.35 * np.sin(2.5 * s + r.uniform(0, 6))
        st.add((S + np.outer(twist * half, n))[3:], max(4, 0.55 * half.max()), r.uniform(0.25, 0.6), paint_)
    for side in (-1, 1):
        e = (S + side * np.outer(half, n))[2:]
        st.add(e, np.clip(0.03 * R, 2, 4), 0.5, "ink" if paint_ != "white" else "olive")
    for side in (-1, 1):
        st.add(S[:7] + side * np.outer(half[:7] * np.linspace(1.1, 0.2, 7), n), max(3.5, 0.4 * half.max()),
               r.uniform(0.2, 0.7), "spathe")


def marigold(st, c, R, r):
    """A marigold on the earth: a few dark leaves, then the head, a heap of short touches of
    orange pressed at the rim and pulled in toward the heart, ragged and uneven, a paler ring
    within and the heart dark; turned a little, so it is seen somewhat from the side."""
    for _ in range(r.integers(10, 17)):      # a low mound of leaves, cut fine
        a = r.uniform(-0.2, np.pi + 0.2) if r.random() < 0.8 else r.uniform(0, 2 * np.pi)
        p0 = c + R * np.array([r.normal(0, 0.45), r.uniform(0.1, 0.6)])
        d = np.array([np.cos(a), 0.6 * np.sin(a)])
        bend = np.array([-d[1], d[0]]) * r.normal(0, 0.3)
        st.add(np.stack([p0, p0 + 0.3 * R * (d + bend), p0 + R * r.uniform(0.35, 0.7) * (d + 2 * bend)]), R * r.uniform(0.07, 0.11),
               r.uniform(0.1, 0.8), "marileaf")
    squash, rot = r.uniform(0.55, 0.9), r.uniform(-0.5, 0.5)
    Q = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    for ring, m, r0, r1, wid, tone in ((0, r.integers(14, 21), (0.75, 1.05), (0.3, 0.5), 0.14, 0.4),
                                       (1, r.integers(8, 13), (0.45, 0.7), (0.05, 0.2), 0.13, 0.75)):
        for t in r.uniform(0, 2 * np.pi, m):
            e = np.array([np.cos(t), np.sin(t) * squash])
            curl = np.array([-e[1], e[0]]) * r.normal(0, 0.15)
            a_, b_ = R * r.uniform(*r0), R * r.uniform(*r1)
            path = c + np.stack([a_ * e, 0.5 * (a_ + b_) * (e + curl), b_ * e]) @ Q.T
            st.add(path, R * wid * r.uniform(0.8, 1.25), tone + 0.25 * (e @ SUN) + r.normal(0, 0.08), "orange")
    for _ in range(2):
        st.add(c + (R * r.normal(0, 0.06, (2, 2)) + [[-0.08 * R, 0], [0.08 * R, 0]]) @ Q.T, 0.13 * R, r.uniform(0.1, 0.6), "heart")


def stem(st, top_, base, w, r):
    """A stem in a stroke or two of green dragged down it, and a dark line along its shaded side."""
    mid = 0.5 * (top_ + base) + [r.normal(0, 0.06) * np.hypot(*(base - top_)), 0]
    S = brush.path(np.stack([top_, mid, base]), 6.0)
    for m in pieces(S, r.uniform(200, 320), 20, r):
        st.add(S[m], w * r.uniform(0.9, 1.1), r.uniform(0.35, 0.7), "stem")
    side = S + [w, 0]
    for m in pieces(side, 200, 10, r, keep=0.6):
        st.add(side[m], max(1.8, 0.3 * w), 0.4, "edge")


def blade(st, S, V, face, w, r):
    """A leaf in a few long strokes dragged from the root to the point, its outline drawn after."""
    wv = np.linalg.norm(V, axis=1)
    k = 3 if w < 40 else 4
    lat = np.sign(V.mean(0) @ SUN + 1e-6)
    up, under = (r.uniform(0.32, 0.5) if r.random() < 0.25 else r.uniform(0.6, 0.92)), r.uniform(0.18, 0.4)
    paint_ = "leafy" if r.random() < 0.3 else "leaf"
    for f in np.linspace(-1, 1, k) * (1 - 1 / (k + 1)) + r.normal(0, 0.05, k):
        line = S + f * V
        for m in pieces(line, r.uniform(380, 560), 40, r):
            tone = (up if face[m].mean() > 0 else under) + 0.12 * f * lat + r.normal(0, 0.03)
            st.add(line[m], max(4.0, wv[m].mean() * 1.25 / k + 1.5), tone, paint_)
    for e in (S + V, S - V):
        for m in pieces(e, r.uniform(300, 460), 15, r, keep=0.85):
            st.add(e[m], np.clip(0.11 * w, 3, 5.5) * np.exp(r.normal(0, 0.15)), r.uniform(0.2, 0.8), "edge")


# the flowers grow in clumps: (x, y, reach, how many, sizes), the far ones smaller
CLUMPS = [(330, 500, 300, 4, (160, 200)), (1000, 450, 330, 4, (150, 195)), (1600, 420, 260, 3, (150, 190)),
          (2550, 470, 380, 5, (165, 210)), (2250, 950, 330, 5, (210, 260)), (650, 1000, 300, 4, (200, 250)),
          (1350, 1150, 250, 3, (220, 265)), (2450, 1560, 400, 5, (260, 330)), (380, 1580, 260, 2, (240, 290))]
WHITE_IRIS = (1860, 760, 270)


def paint(seed=1889):
    r = noise.rng(seed)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#ddd0b0", thread=3.3)
    top, foot, hedge = edges(r)
    grass, soil = yy < top[None, :], yy > foot[None, :]
    noisy = lambda scale, P: noise.field((H, W), scale, r)[at(P)]

    # the lay-in, rubbed on thin, the weave showing through
    t = np.clip(0.45 + 0.22 * noise.fbm((H, W), 90, r, octaves=3) + 0.15 * noise.stretched((H, W), 14, 160, r), 0, 1)[..., None]
    under = np.where(grass[..., None], MEADOW[4], np.where(soil[..., None], EARTH[3], DEEP[2] * (1 - t) + LEAF[4] * t))
    a = (0.86 + 0.1 * ground.tooth)[..., None]
    rgb = (ground.color * (1 - a) + under * (0.85 + 0.3 * ground.tooth[..., None]) * a).astype(np.float32)
    height = ground.tooth * 0.3

    def lay(paths, w, load, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], **{"ends": (0.3, 0.15), "fray": 1.2, **kw})

    def strew(mask, sp):
        P = dabs.scatter((H + 2 * int(sp), W + 2 * int(sp)), sp, r) - int(sp)
        return P[mask[at(P)]]

    def passage(P, field, L, w, k, colours, tone, accent=None, odds=0.2, n=8, bend=0.0, tilt=0.0, back=0.5, **kw):
        if len(P):
            paths, ws, s = sheaves(field, P, L, w, k, r, n, bend, tilt)
            flip = r.random(len(paths)) < back
            paths[flip] = paths[flip, ::-1]
            lay(paths, ws, loads(colours, (tone + r.normal(0, 0.08, len(P)))[s], r, accent, odds), **kw)

    # ---- the meadow: short strokes of yellow-green, level and slanting, tufts standing up in it,
    # and the hedge at the very top in curling strokes of deep green
    lie = (0.25 * noise.field((H, W), 260, r) - 0.1).astype(np.float32)
    P = strew(yy < top[None, :] + 60, spacing(2, 60, 8.5, 2.3))
    n = len(P)
    tuft = noise.field((H, W), 140, r)[at(P)] > 0.7
    passage(P, lie, r.uniform(30, 95, n), r.uniform(6, 11, n), 2, MEADOW,
            0.55 + 0.25 * noisy(300, P) + 0.12 * noisy(60, P) - 0.15 * (P[:, 1] > top[at(P)[1]] - 40), GRASS, 0.2, n=6,
            bend=r.normal(0, 0.006, n), tilt=np.where(tuft, -np.pi / 2 + 0.6 * r.normal(0, 0.4, n), r.normal(0, 0.25, n)),
            thick=0.3, spent=0.55)
    P = strew(yy < top[None, :] + 20, 70)
    n = len(P)
    lay(impasto.follow(np.full((H, W), -1.4, np.float32), P, r.uniform(25, 55, n), 5, r.normal(0, 0.01, n),
                       r.normal(0, 0.35, n)), r.uniform(4, 6.5, n), loads(GRASS, r.uniform(0.2, 0.9, n), r, MEADOW[:2], 0.3),
        thick=0.3, spent=0.6, taper=0.5)
    P = strew(yy < top[None, :] - 15, 95)
    n = len(P)
    lay(impasto.follow(lie, P, r.uniform(9, 20, n), 4, 0.0, r.normal(0, 0.5, n)), r.uniform(4, 6, n),
        loads(SPECK, r.uniform(0, 1, n), r, spread=0.3), thick=0.45, spent=0.3, land=0.8)
    curl = (-0.6 + 2.2 * noise.field((H, W), 300, r)).astype(np.float32)
    P = strew(yy < hedge[None, :] + 10, spacing(2, 70, 9, 2.6))
    n = len(P)
    passage(P, curl, r.uniform(45, 95, n), r.uniform(7, 11, n), 2, HEDGE, 0.45 + 0.25 * noisy(90, P)
            - 0.3 * (P[:, 1] < 0.5 * hedge[at(P)[1]]), pal("#2b4a5e", "#5b7a3e"), 0.2, n=8, bend=r.normal(0, 0.012, n),
            thick=0.3)

    # ---- the earth: short thick touches in rows, red ochre and burnt sienna let down with white,
    # here and there a red, a mauve, a dark
    slope = np.arctan(np.gradient(ndimage.gaussian_filter1d(foot, 80)))[None, :]
    deep_ = np.clip((yy - foot[None, :]) / (H - foot[None, :]), 0, 1)
    rows = (slope * (1 - deep_) - 0.3 * deep_ + 0.12 * noise.field((H, W), 250, r)).astype(np.float32)
    P = strew(yy > foot[None, :] - 40, spacing(2, 100, 12.5, 1.6))
    n = len(P)
    passage(P, rows, r.uniform(60, 140, n), r.uniform(10, 15, n), 2, EARTH, 0.42 + 0.15 * noisy(200, P),
            np.concatenate([RED, MAUVE]), 0.2, n=8, bend=r.normal(0, 0.003, n), thick=0.32, spent=0.55, pickup=0.35)
    P = strew(yy > foot[None, :] - 30, spacing(1, 45, 7.5, 0.7))
    n = len(P)
    passage(P, rows, r.uniform(25, 70, n), r.uniform(5.5, 9.5, n), 1, EARTH,
            0.48 + 0.2 * noisy(160, P) + 0.12 * noisy(40, P) + 0.1 * deep_[at(P)], np.concatenate([RED, MAUVE, EARTH[:2]]), 0.3,
            n=5, bend=r.normal(0, 0.012, n), tilt=r.normal(0, 0.2, n), thick=0.45, spent=0.4, land=0.9, pickup=0.15,
            ends=(0.5, 0.35))

    # ---- the depth of the bed: long dark strokes rising, which the leaves will mostly cover
    rise = (-np.pi / 2 + 0.55 * noise.field((H, W), 500, r)).astype(np.float32)
    below = (yy > top[None, :] + 40 + 50 * noise.field((H, W), 70, r)) & (yy < foot[None, :] + 10)
    P = strew(below & (yy < foot[None, :] - 10), spacing(1, 170, 24, 1.8))
    n = len(P)
    paths, ws, s = sheaves(rise, P, r.uniform(110, 230, n), r.uniform(18, 30, n), 1, r, 8, r.normal(0, 0.004, n),
                           r.normal(0, 0.45, n))
    tone = 0.45 + 0.3 * noisy(250, P) + 0.25 * noise.smoothstep(top[at(P)[1]] + 250, top[at(P)[1]], P[:, 1])
    lay(trim(paths, below), ws, loads(np.concatenate([DEEP[1:], LEAF[1:5]]), tone, r,
                                      np.concatenate([BLUE[1:4], MEADOW[1:3]]), 0.25), thick=0.24, spent=0.6, pickup=0.4)

    # ---- the plants. The flowers are placed clump by clump, each on a stem from a root in the bed;
    # clumps of leaves fan out from those roots and from others between them. Everything is laid
    # from the back of the bed to the front, a flower counting as nearer the higher its stem holds it
    sc = lambda y: 0.7 + 0.45 * np.clip((y - 400) / 1700, 0, 1)
    F = [(*WHITE_IRIS, True)]
    for gx, gy, rad, count, (r0, r1) in CLUMPS:
        placed = 0
        for _ in range(400):
            t, q = r.uniform(0, 2 * np.pi), np.sqrt(r.uniform())
            x, y, R = gx + rad * q * np.cos(t), gy + 0.7 * rad * q * np.sin(t), r.uniform(r0, r1)
            if all(np.hypot(x - f[0], y - f[1]) > 0.55 * (R + f[2]) for f in F):
                F.append((x, y, R, False))
                placed += 1
                if placed == count:
                    break
    items, roots = [], []
    for x, y, R, white in F:
        base = np.array([x + r.normal(0, 60), min(y + r.uniform(380, 720) * sc(y), foot[int(np.clip(x, 0, W - 1))] + 100)])
        roots.append(base)
        near = base[1] + 0.3 * (base[1] - y)
        if white:
            o, age, tilt, toward = 1.0, 0.0, 0.5, -np.pi / 2 - 0.5
        else:
            o, age = [(1.0, 0.0), (r.uniform(0.65, 0.9), 0.0), (r.uniform(0.3, 0.6), 0.0), (1.0, r.uniform(0.5, 1.0))][
                r.choice(4, p=[0.45, 0.15, 0.2, 0.2])]
            tilt, toward = [(abs(r.normal(0, 0.25)), r.uniform(0, 2 * np.pi)),       # upright
                            (r.uniform(0.5, 1.0), -np.pi / 2 + r.normal(0, 0.4)),   # nodding toward us: seen from above
                            (r.uniform(0.6, 1.2), r.choice([0, np.pi]) + r.normal(0, 0.3)),   # bowed to one side
                            (r.uniform(0.3, 0.7), np.pi / 2 + r.normal(0, 0.4))][r.choice(4, p=[0.35, 0.3, 0.2, 0.15])]
        view = seen(r.uniform(0, 2 * np.pi), tilt, toward, r.uniform(0.3, 0.5) + 0.25 * y / H)
        throat = np.array([x, y]) + R * (np.array([0, 0, -0.12]) @ view.T)[:2] * [1, -1]
        falls = "white" if white else ("violet" if r.random() < 0.45 else "blue")
        standards = "white" if white else ("blue" if r.random() < 0.85 else "violet")
        items.append((near - 1, "stem", (throat, base, np.clip(0.05 * R, 5, 10))))
        items.append((near + 700 * white, "flower", (np.array([x, y]), R, (o, r.uniform(0, 1), age, view, falls, standards,
                                                            "olive" if white else "ink", -0.1 if white else r.uniform(0, 0.14)))))
        if r.random() < 0.4 and not white:
            Rb, side = R * r.uniform(0.45, 0.65), r.choice([-1, 1])
            c = np.array([x + side * R * r.uniform(0.5, 1.0), y - R * r.uniform(0.3, 1.0)])
            items.append((near - 2, "stem", (c, throat + [0, 0.8 * R], np.clip(0.04 * R, 3.5, 6))))
            items.append((near + 1, "bud", (c, Rb, -np.pi / 2 + r.normal(0, 0.35), standards)))
    for _ in range(5):
        c = np.array([r.uniform(100, W - 100), r.uniform(380, 640)])
        base = c + [r.normal(0, 40), r.uniform(350, 600)]
        roots.append(base)
        items.append((base[1] - 1, "stem", (c, base, r.uniform(3.5, 5.5))))
        items.append((base[1], "bud", (c, r.uniform(45, 65), -np.pi / 2 + r.normal(0, 0.3), r.choice(["blue", "violet"]))))
    for x in (380, 1250, 2050, 2820):
        x = int(np.clip(x + r.normal(0, 60), 80, W - 80))
        y = foot[x] + (H - foot[x]) * r.uniform(0.3, 0.75)
        for j in range(r.integers(1, 4)):
            items.append((y + 40, "marigold", (np.array([x, y]) + [j * r.uniform(120, 180) * r.choice([-1, 1]), r.normal(0, 40)],
                                               r.uniform(75, 105))))
    G = dabs.scatter((H, W), 360, r)
    G = G[(G[:, 1] > top[at(G)[1]] + 260) & (G[:, 1] < foot[at(G)[1]] + 40)]
    G = [g for g in G if min(np.hypot(*(g - b)) for b in roots) > 170]
    for b in list(roots) + G:
        m, lean = r.integers(3, 6), r.normal(0, 0.2)
        for spread in np.linspace(-0.75, 0.75, m) * r.uniform(0.7, 1.1) + r.normal(0, 0.12, m):
            L = sc(b[1]) * r.uniform(560, 1000) * (1 - 0.3 * abs(spread))
            L = min(L, r.uniform(0.75, 1.35) * (b[1] - top[int(np.clip(b[0], 0, W - 1))]))
            w = sc(b[1]) * r.uniform(40, 58)
            S, V, face = leaf(b, -np.pi / 2 + lean + spread, L, w, r)
            items.append((b[1] + 0.3 * (b[1] - S[14, 1]) + r.normal(0, 15), "leaf", (S, V, face, w)))
    items.sort(key=lambda e: e[0])
    bands = np.array_split(np.arange(len(items)), 6)
    for band in bands:
        green, bloom = Strokes(), Strokes()
        for i in band:
            _, kind, d = items[i]
            if kind == "leaf":
                blade(green, *d, r)
            elif kind == "stem":
                stem(green, *d, r)
            elif kind == "flower":
                flower(bloom, *d, r)
            elif kind == "bud":
                bud(bloom, *d, r)
            else:
                marigold(bloom, *d, r)
        for st, kw in ((green, dict(thick=0.3, spent=0.5, pickup=0.3, grooves=0.35, land=0.4, taper=0.3, fray=0.8, tails=0.4)),
                       (bloom, dict(thick=0.36, spent=0.45, pickup=0.3, grooves=0.45, land=0.7, taper=0.25, ends=(0.4, 0.3),
                                    fray=0.8, tails=0.3))):
            if st.paths:
                lay(st.paths, np.array(st.w), st.load(r), dry=np.array(st.dry), **kw)

    height = ndimage.gaussian_filter(height, 0.7)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.65, 1.25))
    return img + 0.08 * impasto.glints(height, LIGHT)[..., None]
