"""Window on the Harbour, Collioure. Oil on canvas, laid thin and fast in separate strokes of pure colour.

In the summer of 1905 Matisse worked at Collioure, a fishing port on the Mediterranean near the
Spanish border, and Derain joined him there. From his room over the harbour he painted the window
itself, flung open: the casements and the wall round them in colours no wall has, viridian, violet,
vermilion and pink, pots of geraniums and a vine on the sill, and beyond, the boats on a sea of rose
and blue. He worked on a white ground, fast, and laid each colour on thin in its own patch or dash,
not mixed into its neighbour, and he left the canvas bare between them, so that the light of the
picture comes from the canvas itself. A wall is a few columns of broad strokes laid side by side, a
boat a few loaded strokes, a mast one quick line. The colours hold together by their weights and
their places, a strong one small, a pale one wide. That autumn, at the Salon, a critic called the
painters who hung with him wild beasts, fauves.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Window on the Harbour, Collioure"
DATE = "2026"
MEDIUM = "Oil on canvas, laid thin and fast in separate strokes of pure colour, the white ground bare between them"
AFTER = "Henri Matisse, Open Window, Collioure, 1905"
ROOM = "The Garden"
YEAR = 1905
PLACE = "Collioure"
REGION = "Europe"
NOTE = ("A window flung open on the harbour, the casements and the wall round them in viridian, violet, vermilion "
        "and pink. Each colour is laid on thin in its own patch or dash, and the white canvas shows between them.")

H, W = 2400, 2000
LIGHT = (-0.6, -0.5, 0.62)
EX, EY, D = 880.0, 800.0, 2300.0              # where the eye meets the wall square on, and how far back it stands
X0, X1, Y0, Y1 = 560.0, 1430.0, 300.0, 1960.0  # the opening, on the inner face of the wall
T = 300.0                                      # the thickness of the wall
SWING = np.radians([117.0, 137.0])             # how far each casement stands open
STILE, RAIL, FOOT = 0.13, 0.035, 0.06          # a casement's members, as fractions of its width and height
PANES = [(0.035, 0.255), (0.27, 0.49), (0.505, 0.725)]
PANEL = (0.75, 0.94)
RAILS = [(RAIL / 2, RAIL / 2), (0.2625, 0.0075), (0.4975, 0.0075), (0.7375, 0.0125), (1 - FOOT / 2, FOOT / 2)]
HORIZON = 800.0
LEFT, RIGHT, TOP, BELOW, JAMB_L, JAMB_R, SOFFIT, SILL, SKY, HILL, SEA = range(11)
POT = 40
THIN = dict(thick=0.05, grooves=0.14, lips=0.04, land=0.25, lift=0.12, ends=(0.12, 0.05), taper=0.35, fray=1.5,
            flatten=0.3, spent=0.3, pickup=0.15, merge=2.0, tails=0.25, hide=0.94)
LOADED = dict(thick=0.12, grooves=0.3, lips=0.1, land=0.5, lift=0.3, ends=(0.2, 0.1), taper=0.25, fray=1.2, spent=0.25,
              pickup=0.1, merge=1.0, flatten=0.5, tails=0.15, hide=1.0)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each family from dark to light
VIRIDIAN = pal("#185e50", "#1f705e", "#2a836d", "#3a977f", "#52ab93", "#73c0a8")
EMERALD = pal("#2b8a5e", "#3aa070", "#58b585", "#82c99f")
TURQUOISE = pal("#5cbcb8", "#7fcbc6", "#a2dad3", "#c4e8e1", "#def3ee")
COBALT = pal("#27479e", "#3657b0", "#4a6dc0", "#6a8bd0", "#93addf")
ULTRA = pal("#25276f", "#302f85", "#3f3b97", "#5149a6")
VIOLET = pal("#5c3f8f", "#7151a3", "#8a67b6", "#a585c8", "#c0a6da")
MAUVE = pal("#8d4f8f", "#a5629f", "#bc7bb0", "#d098c3", "#e2b8d6")
LILAC = pal("#c6b4dc", "#d6c7e6", "#e5dbef", "#f0eaf5")
ROSE = pal("#d46a78", "#e08793", "#eaa3ab", "#f2bfc1", "#f8d9d6")
SALMON = pal("#e4826a", "#ec9b80", "#f2b39a", "#f6cab4")
VERMILION = pal("#c8361f", "#d6442a", "#e15730", "#ea6c38")
ORANGE = pal("#e8823a", "#ef9a48", "#f4b15e", "#f7c67e")
OCHRE = pal("#bf8c36", "#d0a046", "#ddb45c", "#e8c97c", "#f1dca2")
SAP = pal("#6c9c3a", "#86b247", "#a3c55c", "#c0d77c")
CREAM = pal("#f3e9d4", "#f8f1e2", "#fcf8ee")
MAGENTA = pal("#a8325c", "#bf436c", "#d05a80", "#e07b98")

# the boats: where the hull meets the water (x, y), its length, which way the bow points, the hull, the band
# along its gunwale, its bottom, the mast, and the mast's height and lean
BOATS = [(880, 1300, 300, 1, ROSE, CREAM, COBALT, VERMILION, 700, 0.03),
         (1190, 1215, 235, -1, OCHRE, VERMILION, ULTRA, VIOLET, 560, -0.05),
         (1040, 1060, 175, 1, ROSE, TURQUOISE, ULTRA, ORANGE, 430, 0.02),
         (745, 990, 132, -1, OCHRE, COBALT, VIOLET, VERMILION, 330, -0.03),
         (1285, 950, 112, 1, ROSE, CREAM, VIOLET, OCHRE, 290, 0.045),
         (1135, 880, 76, -1, OCHRE, ULTRA, COBALT, VERMILION, 190, 0.0)]
# the pots on the sill: centre, foot, width at the rim and at the foot, height, its clay, its shade, its flowers
POTS = [(810, 1912, 215, 152, 205, VERMILION, MAGENTA, VERMILION),
        (1040, 1918, 150, 108, 150, COBALT, ULTRA, ROSE),
        (1250, 1910, 228, 160, 222, ORANGE, VERMILION, MAGENTA)]


def at(P):
    return np.clip(P[..., 1].astype(int), 0, H - 1), np.clip(P[..., 0].astype(int), 0, W - 1)


def see(x, y, z):
    """Where a point of the room falls on the canvas: x, y on the wall's inner face, z out through the
    wall (into the room is negative)."""
    x, y, z = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, y, z)))
    f = D / (D + z)
    return np.stack([EX + (x - EX) * f, EY + (y - EY) * f], -1)


def casement(u, v, c):
    """A point on casement c (0 left, 1 right): u across it from the hinge (0) to the free edge (1), v down it."""
    w = (X1 - X0) / 2
    hinge, sign = (X0, 1) if c == 0 else (X1, -1)
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    return see(hinge + sign * u * w * np.cos(SWING[c]), Y0 + v * (Y1 - Y0), -u * w * np.sin(SWING[c]))


OPEN = see([X0, X1, X1, X0], [Y0, Y0, Y1, Y1], 0)     # the opening's corners on the inner face
VIEW = see([X0, X1, X1, X0], [Y0, Y0, Y1, Y1], T)     # and on the outer face, through which the harbour shows


def pot_outline(cx, base, wt, wb, ph):
    """A flowerpot seen a little from above: sides flaring to a thick rim, the foot a shallow curve."""
    y = np.linspace(base - ph, base, 12)
    t = (base - y) / ph
    half = 0.5 * (wb + (wt - wb) * t) * (1 + 0.04 * np.sin(np.pi * t)) + 0.06 * wt * (t > 0.86)
    q = np.linspace(-1, 1, 9)
    foot = np.stack([cx + 0.5 * wb * q, base + 0.07 * wb * np.sqrt(1 - q * q)], 1)
    return np.vstack([np.stack([cx - half, y], 1), foot[1:-1], np.stack([cx + half, y], 1)[::-1]])


def plan(r):
    """The cartoon, drawn freehand: which part of the picture each pixel belongs to. No corner is quite
    where the perspective puts it, and every edge wanders. -> labels (H,W)"""
    im = Image.new("L", (W, H), LEFT)
    d = ImageDraw.Draw(im)

    def put(P, lab, a=4.0):
        P = np.asarray(P, float)
        d.polygon([tuple(p) for p in P + r.normal(0, a, P.shape)], fill=int(lab))

    i, o = OPEN, VIEW
    put([[X1 + 25, -20], [W + 20, -20], [W + 20, H + 20], [X1 - 15, H + 20]], RIGHT, 8)
    put([[X0 - 80, -20], [X1 + 60, -20], [X1 + 25, Y0], [X0, Y0]], TOP, 8)
    put([[X0, Y1], [X1, Y1], [X1 + 40, H + 20], [X0 - 70, H + 20]], BELOW, 8)
    put([i[0], o[0], o[3], i[3]], JAMB_L, 3)
    put([o[1], i[1], i[2], o[2]], JAMB_R, 3)
    put([i[0], i[1], o[1], o[0]], SOFFIT, 3)
    put([o[3], o[2], i[2], i[3]], SILL, 3)
    put(o, SEA, 3)
    put([o[0], o[1], [o[1][0], HORIZON], [o[0][0], HORIZON]], SKY, 1)
    x = np.linspace(o[0][0] - 10, 1010, 40)
    top = HORIZON - 175 * noise.smoothstep(1010, 640, x) ** 0.7 + 10 * noise.line1d(40, 8, r)
    put(np.vstack([np.stack([x, top], 1), [[1010, HORIZON + 2], [x[0], HORIZON + 2]]]), HILL, 1)
    x = np.linspace(1210, o[1][0] + 10, 16)
    top = HORIZON - 34 * noise.smoothstep(1210, 1320, x) + 4 * noise.line1d(16, 4, r)
    put(np.vstack([np.stack([x, top], 1), [[x[-1], HORIZON + 2], [x[0], HORIZON + 2]]]), HILL, 1)
    for c in (0, 1):
        put(casement([0, 1, 1, 0], [0, 0, 1, 1], c), 11 + 10 * c, 3)
        for k, (v0, v1) in enumerate(PANES + [PANEL]):
            put(casement([STILE, 1 - STILE, 1 - STILE, STILE], [v0, v0, v1, v1], c), 12 + 10 * c + k, 3)
    for k, (cx, base, wt, wb, ph, *_) in enumerate(POTS):
        put(pot_outline(cx, base, wt, wb, ph), POT + k, 2)
    lab = np.asarray(im).astype(np.float32)
    dx = 4 * noise.field((H, W), 40, r) + 8 * noise.field((H, W), 300, r)
    dy = 4 * noise.field((H, W), 40, r) + 8 * noise.field((H, W), 300, r)
    return noise.warp(lab, dx, dy, order=0).astype(np.int16)


def loads(colours, tone, r, spread=0.06):
    """The brush for each stroke: the colour of its family for its tone (0 dark to 1 light), and a little of
    a neighbouring tint streaked into it."""
    n, N = len(colours), len(tone)

    def mix(t):
        t = np.clip(t, 0, 1) * (n - 1)
        i = np.minimum(t.astype(int), n - 2)
        f = (t - i)[:, None]
        return colours[i] * (1 - f) + colours[i + 1] * f

    t = tone + r.normal(0, spread, N)
    side = r.choice([-1, 1], N)
    share = np.stack([r.uniform(0.6, 0.85, N), r.uniform(0.15, 0.35, N)], 1)
    return np.stack([mix(t), mix(t + side * r.uniform(0.08, 0.2, N))], 1).astype(np.float32), share


def mixed(families, which, tone, r, spread=0.06):
    """Loads for strokes drawn from several families, `which` saying which one each takes."""
    tone = np.broadcast_to(np.asarray(tone, np.float64), which.shape)
    cols, share = np.empty((len(which), 2, 3), np.float32), np.empty((len(which), 2))
    for i, colours in enumerate(families):
        sel = which == i
        if sel.any():
            cols[sel], share[sel] = loads(colours, tone[sel], r, spread)
    return cols, share


def sweep(mask, theta, w, L, r, join=(0.6, 0.9), gap=(-0.5, 0.1), skip=0.03, tilt=0.035, n=8, wide=0.1,
          stagger=(0.15, 1.0), jitter=0.08):
    """Strokes laid side by side across a patch, as a painter covers a wall: a column of strokes along
    `theta` from one edge of the patch to the other, then the next column beside it, over it a little
    (`join`, in half-widths) or, now and then (`wide`), a little short of it so that a sliver of canvas
    shows. Each column starts with a stroke of its own length, so that the joins never line up; along a
    column each stroke starts back over the last (`gap`), and now and then one is missed (`skip`).
    -> paths (N, n, 2), half-widths (N,), the column of each"""
    c, s = np.cos(theta), np.sin(theta)
    ys, xs = np.nonzero(mask[::6, ::6])
    if not len(ys):
        return np.zeros((0, n, 2)), np.zeros(0), np.zeros(0, int)
    xs, ys = xs * 6.0, ys * 6.0
    U, V = xs * c + ys * s, ys * c - xs * s
    u = np.arange(U.min() - 12, U.max() + 12, 3.0)
    rows, v, col = [], V.min() - r.uniform(0.2, 1.0) * w, 0
    while v < V.max() + w:
        wi = w * np.exp(r.normal(0, 0.2))
        step = r.uniform(*join) if r.random() > wide else r.uniform(1.0, 1.25)
        v += wi * step
        ins = mask[at(np.stack([u * c - v * s, u * s + v * c], -1))]
        d = np.diff(np.concatenate([[0], ins.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            t, end, first = u[a], u[b - 1], True
            while t < end - 0.5 * wi:
                e = min(t + L * np.exp(r.normal(0, 0.3)) * (r.uniform(*stagger) if first else 1), end)
                if r.random() > skip and e - t > wi:
                    rows.append((t, e, v + r.normal(0, 0.08 * wi), wi * np.exp(r.normal(0, jitter)), col))
                t, first = e + wi * r.uniform(*gap), False
        v += wi * step
        col += 1
    if not rows:
        return np.zeros((0, n, 2)), np.zeros(0), np.zeros(0, int)
    t0, t1, vv, ww, cc = np.array(rows).T
    k = np.linspace(0, 1, n)[None]
    uu = t0[:, None] + (t1 - t0)[:, None] * k
    off = vv[:, None] + (r.normal(0, tilt, len(t0)) * (t1 - t0))[:, None] * (k - 0.5) \
        + (r.normal(0, 0.2, len(t0)) * ww)[:, None] * np.sin(np.pi * k)
    return np.stack([uu * c - off * s, uu * s + off * c], -1), ww, cc.astype(int)


def spacing(length, width, cover):
    """Seeds far enough apart that strokes of this length and half-width cover the ground `cover` times."""
    return float(np.sqrt(length * 2 * width / (cover * 0.866)))


def trim(paths, mask):
    """Cut each stroke where it first leaves the mask, so that none runs over its place."""
    n = paths.shape[1]
    ins = mask[at(paths)]
    k = np.where(ins.all(1), n, ins.argmin(1))
    t = np.linspace(0, 1, n)[None, :] * np.maximum(k - 1, 0)[:, None]
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    row = np.arange(len(paths))[:, None]
    return paths[row, i] * (1 - f) + paths[row, np.minimum(i + 1, n - 1)] * f


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back over the last
    (a little after it, with a negative overlap), lifting off now and then (keep < 1). -> index arrays"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def flick(p, a, length, curl, n=5):
    """Short strokes from points p (N,2) at angles a, curling by `curl` radians per px. -> (N, n, 2)"""
    th = a[:, None] + (curl * length)[:, None] * np.linspace(0, 1, n)[None]
    d = np.stack([np.cos(th), np.sin(th)], -1) * (length / (n - 1))[:, None, None]
    return p[:, None, :] + np.concatenate([np.zeros((len(p), 1, 2)), np.cumsum(d[:, :-1], 1)], 1)


def hand(line, amp, scale, r):
    """A line as a hand draws it, wandering off its course and back."""
    d = np.gradient(line, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return line + np.stack([-d[:, 1], d[:, 0]], 1) * (amp * noise.line1d(len(line), scale, r))[:, None]


def paint(seed=1905):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint="#f4efe4", thread=3.4)
    rgb, height = ground.color.copy(), ground.tooth * 0.3
    wet = np.zeros((H, W), np.float32)
    key = plan(r)
    seam = np.zeros((H, W), bool)
    seam[1:] |= key[1:] != key[:-1]
    seam[:-1] |= key[1:] != key[:-1]
    seam[:, 1:] |= key[:, 1:] != key[:, :-1]
    seam[:, :-1] |= key[:, 1:] != key[:, :-1]
    apart = ndimage.distance_transform_edt(~seam).astype(np.float32)    # px from the nearest edge of a patch
    gutter = 1.0 + 9 * noise.smoothstep(-0.6, 1.6, noise.field((H, W), 170, r))     # the bare canvas between patches
    vary = noise.field((H, W), 420, r)
    clump = noise.smoothstep(-1.2, 1.2, noise.field((H, W), 300, r))                 # where one colour gathers
    up = np.pi / 2

    def lay(paths, w, load, **kw):
        if len(paths):
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **{**THIN, **kw})

    def room(*labs, inset=0.0):
        return np.isin(key, labs) & (apart > gutter + inset)

    def turn(paths):
        back = r.random(len(paths)) < 0.5
        paths[back] = paths[back, ::-1]
        return paths

    def choose(odds, mid, patch):
        pick = np.clip(r.random(len(mid)) * (1 - patch) + patch * clump[at(mid)], 0, 0.999)
        return np.searchsorted(np.cumsum(odds) / np.sum(odds), pick, side="right")

    def fill(mask, theta, fams, odds, tone, L, w, join=(0.5, 0.82), gap=(-0.5, 0.05), skip=0.02, tilt=0.035,
             spread=0.05, patch=0.45, inset=0.6, wide=0.06, thin=0.2, stagger=(0.15, 1.0), jitter=0.08, **kw):
        """A patch covered in columns of strokes, each stroke from a family picked by `odds` (each family
        gathering in places), the tint drifting a little from column to column as the brush is reloaded,
        stopping a little short of the patch's edge. Some strokes (`thin`) are scrubbed on thinner first."""
        paths, ws, col = sweep(mask & (apart > gutter + inset * w), theta, w, L, r, join, gap, skip, tilt, wide=wide,
                               stagger=stagger, jitter=jitter)
        if not len(ws):
            return
        mid = paths[:, paths.shape[1] // 2]
        t = tone + 0.1 * vary[at(mid)] + r.normal(0, 0.06, col.max() + 1)[col] + r.normal(0, 0.035, len(ws))
        load = mixed(fams, choose(odds, mid, patch), t, r, spread)
        paths = turn(paths)
        scrub = r.random(len(ws)) < thin
        for sel, extra in ((scrub, dict(hide=0.75, dry=r.uniform(0.05, 0.25, scrub.sum()))), (~scrub, {})):
            if sel.any():
                lay(paths[sel], ws[sel], (load[0][sel], load[1][sel]), **{**kw, **extra})

    is_ = lambda *labs: np.isin(key, labs)

    # the walls, in broad strokes laid fast and thin, column by column: viridian to the left of the window,
    # violet to the right, ultramarine over it and pink below the sill
    for lab, theta, fams, odds, tone, L, w in (
            (LEFT, up + 0.03, (VIRIDIAN, EMERALD, COBALT, TURQUOISE), (0.74, 0.1, 0.08, 0.08), 0.45, 760, 36),
            (RIGHT, up - 0.05, (MAUVE, VIOLET, ROSE, VERMILION), (0.6, 0.25, 0.12, 0.03), 0.5, 700, 32),
            (TOP, 0.04, (ULTRA, VIOLET, VIRIDIAN, ROSE), (0.45, 0.35, 0.15, 0.05), 0.5, 600, 30),
            (BELOW, 0.1, (SALMON, ROSE, ORANGE, LILAC), (0.5, 0.3, 0.1, 0.1), 0.45, 640, 46)):
        fill(is_(lab), theta, fams, odds, tone, L, w, thin=0.3)

    # the depth of the wall in the opening: the jambs, one in shade and one in the sun, the soffit, the sill
    fill(is_(JAMB_L), up, (VIOLET, LILAC), (0.6, 0.4), 0.45, 300, 10, inset=0.3, join=(0.8, 1.0))
    fill(is_(JAMB_R), up, (OCHRE, ORANGE), (0.6, 0.4), 0.55, 300, 11, inset=0.3, join=(0.8, 1.0))
    fill(is_(SOFFIT), 0.02, (ROSE, SALMON), (0.6, 0.4), 0.4, 260, 12, inset=0.3)
    fill(is_(SILL), -0.02, (CREAM, ROSE, LILAC), (0.5, 0.3, 0.2), 0.6, 300, 14, inset=0.3)

    # the sky, pale turquoise and pink in long level strokes with the canvas between them
    fill(is_(SKY), 0.0, (TURQUOISE, ROSE, CREAM, LILAC), (0.45, 0.3, 0.15, 0.1), 0.72, 380, 16, join=(0.75, 1.1),
         gap=(0.2, 1.4), skip=0.08, patch=0.5, inset=0.3, ends=(0.6, 0.4), taper=0.45)

    # the headland across the bay, violet and rose, laid along its slope
    fill(is_(HILL), 0.33, (VIOLET, ROSE, MAUVE, OCHRE), (0.5, 0.25, 0.15, 0.1), 0.3, 260, 14, inset=0.0, thin=0.0,
         join=(0.45, 0.7))
    x = np.linspace(VIEW[0][0] + 6, 1000, 30)
    ridge = HORIZON - 175 * noise.smoothstep(1000, 640, x) ** 0.7 + 8
    for m in pieces(np.stack([x, ridge], 1), 260, 30, r, 0.9):
        lay([hand(np.stack([x, ridge], 1)[m], 3, 10, r)], r.uniform(6, 9), loads(ROSE, np.array([0.45]), r), **LOADED)

    # the sea in level dashes, rose and blue, longer and broader as they come nearer, a deeper blue along the
    # horizon; the rose and the blue each gather in places, as the water takes the sky or the light
    sea = room(SEA, inset=2)
    level = (0.03 * vary).astype(np.float32)
    P = dabs.scatter((H, W), 7, r)
    g = np.clip((P[:, 1] - HORIZON) / (VIEW[3][1] - HORIZON), 0, 1)
    L, w = 70 + 330 * g, 4.5 + 14 * g
    keep = sea[at(P)] & (r.random(len(P)) < (7 / np.array([spacing(a_, b_, 1.4) for a_, b_ in zip(L, w)])) ** 2)
    P, g, L, w = P[keep], g[keep], L[keep], w[keep]
    k = len(P)
    paths = turn(trim(impasto.follow(level, P, L * np.exp(r.normal(0, 0.4, k)), 6, 0.0, r.normal(0, 0.05, k)), sea))
    fams = (ROSE, LILAC, TURQUOISE, COBALT, CREAM, VIOLET)
    which = choose((0.3, 0.12, 0.22, 0.14, 0.14, 0.08), P, 0.7)
    tone = np.array([0.45, 0.4, 0.3, 0.5, 0.6, 0.45])[which] + 0.25 * r.random(k)    # the blues kept strong
    lay(paths, w * np.exp(r.normal(0, 0.3, k)), mixed(fams, which, tone, r), ends=(0.7, 0.5), taper=0.5)
    near = (g > 0.3) & (r.random(k) < 0.22)          # over the rose near at hand, the blue of the sky it takes
    lay(turn(trim(impasto.follow(level, P[near], L[near] * r.uniform(0.4, 0.9, near.sum()), 6, 0.0,
                                 r.normal(0, 0.04, near.sum())), sea)), w[near] * r.uniform(0.6, 1.0, near.sum()),
        mixed((TURQUOISE, COBALT, LILAC), r.choice(3, near.sum(), p=[0.5, 0.25, 0.25]), 0.3 + 0.25 * r.random(near.sum()), r),
        ends=(0.7, 0.5), taper=0.5)
    P = P[(g < 0.035) & (r.random(k) < 0.7)]
    lay(trim(impasto.follow(level, P, r.uniform(80, 220, len(P)), 6, 0.0, r.normal(0, 0.01, len(P))), sea),
        r.uniform(3, 5, len(P)), mixed((COBALT, ULTRA, TURQUOISE), r.integers(0, 3, len(P)), 0.45, r), ends=(0.6, 0.4))

    # the boats, each in a few loaded strokes: the hull in bands, the band along its gunwale light, the
    # bottom dark, the mast in one quick line, and their colours broken in the water below
    for x, y, L, bow, hull, band, keel, mast, mh, lean in BOATS:
        s = np.linspace(-1, 1, 12)
        h = 0.28 * L
        top = y - h * (0.78 + 0.25 * s * s) - 0.35 * h * np.maximum(bow * s, 0) ** 3
        bot = y + (top - y) * np.abs(s) ** 4
        X = x + bow * s * L / 2
        q = L / 250
        for f, reach, fam, tone, wd in ((0.22, 0.98, band, 0.6, 0.2), (0.52, 0.9, hull, 0.45, 0.25), (0.82, 0.72, keel, 0.35, 0.2)):
            m = np.abs(s) <= reach
            path = np.stack([X[m], (top + f * (bot - top))[m]], 1)
            lay([hand(path, 1.5 * q, 4, r)], wd * h * r.uniform(0.95, 1.15), loads(fam, np.array([tone]), r),
                **{**LOADED, "thick": 0.16, "ends": (0.5, 0.4), "taper": 0.45})
        foot = np.array([x - 0.12 * bow * L / 2, y - 0.75 * h])
        head = foot + [lean * mh, -mh]
        path = hand(np.stack([np.linspace(foot[0], head[0], 7), np.linspace(foot[1], head[1], 7)], 1), 2.5 * q, 3, r)
        lay([path], max(2.8, 6.5 * q), loads(mast, np.array([0.45]), r), **{**LOADED, "ends": (0.1, 0.0), "taper": 0.7})
        n = r.integers(3, 6)
        p = np.stack([x + r.uniform(-0.45, 0.45, n) * L, y + h * r.uniform(0.25, 1.4, n)], 1)
        fam = [keel, hull, band, mast]
        for i in range(n):
            lay(flick(p[i:i + 1], np.array([r.normal(0, 0.05)]), np.array([L * r.uniform(0.15, 0.4)]), np.zeros(1)),
                max(2.5, 5.5 * q) * r.uniform(0.8, 1.3), loads(fam[i % 4], np.array([0.5]), r), hide=0.85)
        stem = np.stack([np.full(40, foot[0] + r.normal(0, 2)), y + 0.3 * h + np.linspace(0, 0.45 * mh, 40)], 1)
        for m in pieces(stem, 0.14 * mh, -0.04 * mh, r, 0.8):
            lay([hand(stem[m], 2, 2, r)], max(2.0, 3.5 * q), loads(mast, np.array([0.55]), r), hide=0.8, taper=0.6)

    # the vine climbing the jambs outside and hanging along the top, in quick touches of green
    o = VIEW
    vines = []
    for x0, y0, x1, y1 in ((o[0][0] + 45, o[3][1], o[0][0] + 30, o[0][1] + 30), (o[1][0] - 55, o[2][1], o[1][0] - 40, o[1][1] + 30),
                           (o[0][0], o[0][1] + 45, o[1][0], o[1][1] + 40)):
        t = np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0) / 6))
        line = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + 60 * np.sin(np.pi * t) * (abs(y1 - y0) < 100)], 1)
        vines.append(hand(line, 30, 90, r))
    leaves, stems = [], []
    for line in vines:
        S = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
        s_ = np.cumsum(r.uniform(14, 38, int(S[-1] / 14)))
        j = np.searchsorted(S, s_[s_ < S[-1]])
        d = np.gradient(line, axis=0)[j]
        a = np.arctan2(d[:, 1], d[:, 0])
        hub = line[j] + np.stack([-np.sin(a), np.cos(a)], 1) * (r.normal(0, 34, len(j)))[:, None]
        m = r.integers(2, 7, len(j))                   # a few leaves fanning out from each node
        i = np.repeat(np.arange(len(j)), m)
        fan = a[i] + r.choice([-1, 1], len(i)) * r.uniform(0.5, 1.9, len(i))
        leaves.append((hub[i] + 8 * np.stack([np.cos(fan), np.sin(fan)], 1), fan))
        stems += [line[m_] for m_ in pieces(line, 180, -40, r, 0.5)]
    p = np.concatenate([q for q, _ in leaves])
    a = np.concatenate([q for _, q in leaves])
    k = len(p)
    lay(stems, r.uniform(2.5, 4, len(stems)), loads(VIRIDIAN, np.full(len(stems), 0.2), r), hide=0.85, taper=0.6)
    a += r.normal(0, 0.6, k)
    lay(flick(p, a, r.uniform(14, 36, k), r.normal(0, 0.03, k)), r.uniform(8, 15, k),
        mixed((EMERALD, VIRIDIAN, SAP, TURQUOISE, COBALT), choose((0.33, 0.3, 0.25, 0.07, 0.05), p, 0.4),
              r.uniform(0.2, 0.85, k), r), **{**LOADED, "thick": 0.09, "ends": (0.35, 0.15), "taper": 0.25})

    # the casements: the panes first, lilac and pink on the left, viridian on the right, in a few columns,
    # then the bars over them in long loaded strokes, vermilion on the left, pink on the right
    for c, panes, panel, frame in ((0, ((LILAC, ROSE, CREAM, TURQUOISE), (0.4, 0.25, 0.2, 0.15)), (ROSE, SALMON), VERMILION),
                                   (1, ((VIRIDIAN, EMERALD, TURQUOISE, COBALT), (0.5, 0.2, 0.2, 0.1)), (SALMON, ROSE, ORANGE), ROSE)):
        span = np.hypot(*(casement(1 - STILE, 0.4, c) - casement(STILE, 0.4, c)))
        for k in range(3):
            fill(is_(12 + 10 * c + k), up + 0.02, *panes, 0.5, 650, span / 6, join=(0.62, 0.92), gap=(-0.3, 0.1),
                 skip=0.03, patch=0.3, inset=0.3, stagger=(0.7, 1.0), taper=0.08, ends=(0.2, 0.15))
        fill(is_(15 + 10 * c), up, panel, (0.5, 0.3, 0.2)[:len(panel)], 0.5, 500, span / 5, inset=0.3,
             join=(0.6, 0.9), stagger=(0.7, 1.0))
        for u in (STILE / 2, 1 - STILE / 2):
            line = casement(u, np.linspace(-0.01, 1.01, 40), c)
            hw = 0.5 * np.hypot(*(casement(u + STILE / 2, 0.5, c) - casement(u - STILE / 2, 0.5, c)))
            for m in pieces(line, 700, 40, r, 0.95):
                lay([hand(line[m], 4, 30, r)], hw * r.uniform(0.8, 1.05), loads(frame, np.array([0.45]), r, 0.12),
                    **{**LOADED, "thick": 0.09})
        for v, t in RAILS:
            line = casement(np.linspace(-0.02, 1.02, 16), v, c)
            hw = 0.5 * np.hypot(*(casement(0.5, v + t, c) - casement(0.5, v - t, c)))
            lay([hand(line, 3, 20, r)], hw * r.uniform(0.85, 1.1), loads(frame, np.array([0.5]), r, 0.12),
                **{**LOADED, "thick": 0.09})

    # the pots on the sill, and the geraniums in them: the foliage behind first, then the pot in a few
    # columns, light to dark across it, its rim, the leaves that hang over the rim, and the flowers last
    for k, (cx, base, wt, wb, ph, clay, shade, bloom) in enumerate(POTS):
        rim = base - ph
        n = r.integers(4, 7)
        heads = np.stack([cx + r.uniform(-0.7, 0.7, n) * wt, rim - r.uniform(120, 250, n)], 1)
        m = r.integers(18, 26)
        u = r.uniform(-1, 1, m)
        p = np.stack([cx + 0.62 * wt * u, rim + 10 - r.uniform(0, 215, m) * (1 - 0.45 * u * u)], 1)
        lay(flick(p, r.uniform(0, 2 * np.pi, m), r.uniform(22, 40, m), r.normal(0, 0.03, m)), r.uniform(17, 27, m),
            mixed((EMERALD, VIRIDIAN, SAP), choose((0.4, 0.35, 0.25), p, 0.3), r.uniform(0.2, 0.7, m), r),
            **{**LOADED, "thick": 0.1, "ends": (0.8, 0.6)})
        paths, ws, _ = sweep(is_(POT + k), up, wt / 5.5, 3 * ph, r, join=(0.4, 0.6), gap=(-0.3, 0), skip=0.0,
                             tilt=0.05, wide=0.0, stagger=(1.0, 1.0))
        across = (paths[:, 4, 0] - (cx - 0.5 * wt)) / wt
        lay(paths[:, ::-1], ws, loads(clay, 0.72 - 0.3 * across, r),     # pressed on at the foot, lifted at the rim
            **{**LOADED, "thick": 0.1, "spent": 0.1, "ends": (0.3, 0.1)})
        rimline = np.stack([np.linspace(cx - 0.56 * wt, cx + 0.56 * wt, 8), rim + 0.06 * ph + 8 * np.sin(np.linspace(0, np.pi, 8))], 1)
        lay([hand(rimline, 2, 20, r)], 0.055 * ph + 3, loads(clay, np.array([0.8]), r), **LOADED)
        side = pot_outline(cx, base, wt, wb, ph)[-12:][::-1][1:-1] + [-4, 0]
        lay([hand(side, 2, 10, r)], 0.03 * wt + 2, loads(shade, np.array([0.3]), r), **{**LOADED, "thick": 0.1})
        lay(flick(np.array([[cx + 0.35 * wb, base + 4]]), np.array([0.05]), np.array([0.6 * wb]), np.zeros(1)), 7,
            loads(VIOLET, np.array([0.4]), r), hide=0.8)
        m = r.integers(3, 6)
        p = np.stack([cx + 0.5 * wt * r.uniform(-1, 1, m), rim + r.uniform(-20, 10, m)], 1)
        lay(flick(p, r.uniform(0, 2 * np.pi, m), r.uniform(18, 30, m), r.normal(0, 0.03, m)), r.uniform(12, 18, m),
            mixed((EMERALD, SAP), r.integers(0, 2, m), r.uniform(0.3, 0.8, m), r), **{**LOADED, "thick": 0.1, "ends": (0.8, 0.6)})
        pts, ang, ln, wd, tn = [], [], [], [], []
        for hx, hy in heads:
            j = r.integers(10, 17)
            rho, th = 36 * np.sqrt(r.random(j)), r.uniform(0, 2 * np.pi, j)
            pts.append(np.stack([hx + rho * np.cos(th), hy + 0.85 * rho * np.sin(th)], 1))
            ang.append(r.uniform(0, 2 * np.pi, j))
            ln.append(r.uniform(10, 20, j))
            wd.append(r.uniform(6, 10, j))
            tn.append(np.where(r.random(j) < 0.15, 0.95, r.uniform(0.25, 0.75, j)))
        pts, ang, ln, wd, tn = map(np.concatenate, (pts, ang, ln, wd, tn))
        lay(flick(pts, ang, ln, r.normal(0, 0.04, len(pts))), wd, loads(bloom, tn, r),
            **{**LOADED, "thick": 0.2, "ends": (0.7, 0.5), "taper": 0.3})

    # the drawing, in broken lines of colour: the hinges, the free edge, the jambs, and the sill's edge
    lines = [(casement(0.0, np.linspace(0, 1, 30), c), ULTRA if c else VIRIDIAN) for c in (0, 1)]
    lines += [(casement(1.0, np.linspace(0, 1, 30), 1), VERMILION)]
    lines += [(np.stack([np.linspace(OPEN[3][0] - 20, OPEN[2][0] + 20, 30), np.full(30, Y1 + 4)], 1), VERMILION)]
    lines += [(np.stack([np.full(30, VIEW[k][0]), np.linspace(VIEW[0][1], VIEW[3][1], 30)], 1), VIOLET) for k in (0, 1)]
    for line, fam in lines:
        for m in pieces(line, 420, 20, r, 0.85):
            lay([hand(line[m], 2.5, 25, r)], r.uniform(4, 7), loads(fam, np.array([0.3]), r, 0.1), **{**LOADED, "thick": 0.08})

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.8, gloss=0, reach=(0.8, 1.15))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
