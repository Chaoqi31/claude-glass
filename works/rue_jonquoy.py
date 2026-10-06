"""Light Rising over Water. Oil on canvas.

In the large oils of the 1980s Zao Wou-Ki painted space itself: no mountain, no river, no sky, and yet the
feeling of all three, the way a Song landscape gives it. He thinned the oil with turpentine until it ran almost
like ink and laid it on a white ground in veil after veil, worked wet with broad brushes, so that the white
comes up through the colour as light. Where a veil was left to settle, the pigment drew out to its edge and
dried there in a darker rim, and into the hollows of the weave, so that each veil keeps its own edge under the
next; where he rubbed colour on with a rag it lies in clouds; where he wiped it off again the white comes back,
with the weave of the canvas in it. At the sides he pulled the dark in from the top and the foot in single
strokes of a big brush, and pulled it again over everything once the light was in, so that the light would
glow; the bristles combed it into streaks, and each stroke ran dry toward the middle and broke over the weave,
or he wiped its end off with the rag. Some passages were so wet that they ran, and he let the runs stand. Last
he drew through the middle with a smaller brush, fast, in clusters of short dark strokes pressed and lifted as
he had drawn with ink on paper, some with a full brush that softened a little into the wet paint, some nearly
dry and broken by the weave; and here and there he laid paint on thick with a loaded brush, and it stands up
from the rest and catches the light.

Here the light gathers right of the middle, pale gold and white and warm grey, and spreads to the left in a
band low across the canvas, like mist lying on water at dawn. Away from it the veils deepen through cerulean
and ultramarine into indigo, and in a few strokes nearly to black, deepest in two corners; low on the left a
passage of red ochre and madder gives it weight; the thinned paint has run down from it and from the dark at
the edges.
"""

import numpy as np
from scipy import ndimage

from atelier import brush, canvas, dabs, impasto, noise
from atelier.color import lin, pigment

TITLE = "Light Rising over Water"
DATE = "2026"
MEDIUM = ("Oil on canvas, thinned almost to a wash and laid in many veils with broad brushes and a rag, wiped "
          "back and left to run; drawn into with a smaller brush, a few touches laid thick with a loaded brush")
AFTER = "Zao Wou-Ki, the large oils of the 1980s, such as Juin–Octobre 1985"
ROOM = "Colour Itself"
YEAR = 1985
PLACE = "Paris"
REGION = "East Asia"
NOTE = ("A space of light with no landscape in it: veils of blue deepening into indigo round a pale gold core, "
        "thin paint running down, and clusters of dark strokes drawn fast through the middle.")

H, W = 1950, 3900            # about 97 by 195 cm
CX, CY = 2330.0, 820.0       # where the light gathers
Q = 4                        # the plan of the picture is kept at a quarter of the size
GROUND = "#f4f0e7"           # white priming
PALE_GOLD, GOLD, WARM_GREY, WHITE, MIST = "#f2dfab", "#ebbd5e", "#cbc2b5", "#f7f3ea", "#e8eef6"
CERULEAN, COBALT, ULTRA, VIOLET, INDIGO = "#a3c1df", "#7593cf", "#4356b2", "#7465aa", "#1d2a6e"
OCHRE, MADDER = "#c2672f", "#9c2c37"
INK = "#1d2036"
# the clusters of the brush: where (x, y), how far they spread (x, y), how many strokes, which way they lean, and
# what share of them went on with a full brush
CLUSTERS = ((1960, 1000, 260, 110, 70, -0.5, 0.5), (2930, 1010, 150, 200, 40, -2.2, 0.4),
            (930, 900, 140, 130, 18, -1.15, 0.3), (2420, 1330, 220, 60, 12, -0.2, 0.1))


def tile(shape, across, along, r, octaves=1):
    """Noise that wraps round at its edges, so that a tool may read it from anywhere without meeting a seam:
    features about `across` px wide and `along` px long (down the rows), and finer ones under them."""
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.rfftfreq(shape[1])[None]
    out = 0
    for k in range(octaves):
        f = np.fft.irfft2(np.fft.rfft2(r.standard_normal(shape)) * np.exp(-1.2 / 4 ** k * ((fy * along) ** 2 + (fx * across) ** 2)), shape)
        out = out + 0.5 ** k * (f - f.mean()) / f.std()
    return (out / np.sqrt((1 - 0.25 ** octaves) / 0.75)).astype(np.float32)


class Ground:
    """The primed canvas as the tools meet it: how high it stands at each px (the tops of the threads) and
    where pigment settles (its hollows); the clumps a broad soft brush leaves in thin paint, laid out along a
    stroke (rows along it, columns across); the folds of a crumpled rag, and the larger lumps of the whole
    cloth or the loaded brush; the deep and shallow places of a pool of thinned paint, and where the brush
    feathered its edge out. Each is read at a fresh place for every stroke."""

    def __init__(self, sheet, r):
        self.r = r
        self.tops = 0.8 * sheet.tooth + 0.2 * noise.smoothstep(-2, 2, noise.field(sheet.tooth.shape, 3, r))
        self.hollow = 0.5 - ndimage.gaussian_filter(sheet.tooth, 1.2)
        hairs = tile((1024, 1024), 24, 700, r) + tile((1024, 1024), 64, 1000, r)
        self.hairs = hairs / hairs.std()
        self.folds = tile((1024, 1024), 32, 32, r, 2)
        self.blots = tile((1024, 1024), 220, 220, r, 2)     # so coarse that reading it magnified leaves no steps
        self.grit = tile((1024, 1024), 5, 5, r, 2)
        self.mottle = tile((2048, 2048), 320, 320, r, 2)
        self.lost = tile((2048, 2048), 260, 260, r)

    def read(self, tex, s, v, scale=1.0):
        h, w = tex.shape
        return tex[(np.floor(s * scale).astype(np.intp) + self.r.integers(h)) % h,
                   (np.floor(v * scale).astype(np.intp) + self.r.integers(w)) % w]

    def patch(self, tex, box):
        h, w = tex.shape
        return tex[np.ix_((np.arange(box[0].start, box[0].stop) + self.r.integers(h)) % h,
                          (np.arange(box[1].start, box[1].stop) + self.r.integers(w)) % w)]


def arc(a, b, bend, half, pad):
    """The pixels a tool `half` px either side of its path can reach, going from a to b along a circular arc
    that bows `bend` px to the left of the straight line. -> slices of the box round it, the mask of pixels in
    reach, and for those the distance along the path `s` and across it `v` (px), and the length of the path."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.hypot(*(b - a)))
    t = (b - a) / c
    n = np.array([-t[1], t[0]])
    mid = (a + b) / 2
    pts = np.array([a, b, mid + n * bend])
    x0, y0 = np.maximum(np.floor(pts.min(0) - half - pad).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0) + half + pad).astype(int) + 1, (W, H))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    yy, xx = yy.astype(np.float32), xx.astype(np.float32)
    if abs(bend) < 1:
        v = (xx - a[0]) * n[0] + (yy - a[1]) * n[1]
        m = np.abs(v) < half + pad
        s = ((xx - a[0]) * t[0] + (yy - a[1]) * t[1])[m]
        L = c
    else:
        R = (c * c / 4 + bend * bend) / (2 * abs(bend))
        o = mid + n * (bend - np.sign(bend) * R)
        v = (np.hypot(xx - o[0], yy - o[1]) - R) * np.sign(bend)
        m = np.abs(v) < half + pad
        wrap = lambda q: (q + np.pi) % (2 * np.pi) - np.pi
        ta = np.arctan2(a[1] - o[1], a[0] - o[0])
        turn = np.sign(wrap(np.arctan2(b[1] - o[1], b[0] - o[0]) - ta))
        Y, X = np.nonzero(m)
        s = (wrap(np.arctan2(Y + (y0 - o[1]), X + (x0 - o[0])) - ta) * turn * R).astype(np.float32)
        L = 2 * R * np.arcsin(min(c / (2 * R), 1.0))
    if not m.any():
        return None
    return (slice(y0, y1), slice(x0, x1)), m, s, v[m], L


def wander(s, L, amp, scale, r):
    """How far the hand drifts off its path, at each distance along it."""
    n = int(L) + 42
    return amp * noise.line1d(n, scale, r)[np.clip(s + 20, 0, n - 1).astype(np.intp)]


def wash(T, g, strokes, load, rim=0.6, comb=0.15, grain=0.9, feather=(3, 60)):
    """A passage of oil thinned almost to a wash and worked wet with a broad soft brush, `strokes` [(a, b, bend,
    width)] back and forth across one another, so that they run together into one pool. The brush leaves its
    clumps in the paint, soft and wide, and toward the end of each stroke it runs thin. As the pool settles the
    pigment is drawn out to where it stopped and dries there in a darker rim that the weave breaks, leaving the
    paint just inside it a little thinner (where the brush feathered the edge out there is no rim, only the
    clumps fading); and it gathers in the deeper places of the pool and in the hollows of the weave: the thinner
    the paint, the more of it lies there."""
    r = g.r
    pts = []
    for a, b, bend, width in strokes:
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = (b - a) / (np.hypot(*(b - a)) + 1e-9)
        pts += [a, b, (a + b) / 2 + np.array([-t[1], t[0]]) * bend]
    pts = np.array(pts)
    pad = 0.95 * max(w for *_, w in strokes) + 12
    x0, y0 = np.maximum(np.floor(pts.min(0) - pad).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0) + pad).astype(int) + 1, (W, H))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    box = (slice(y0, y1), slice(x0, x1))
    wet, streak = np.zeros((2, y1 - y0, x1 - x0), np.float32)
    dry = np.ones((y1 - y0, x1 - x0), np.float32)
    for a, b, bend, width in strokes:
        fr = arc(a, b, bend, 0.8 * width, 4)
        if fr is None:
            continue
        sl, m, s, v, L = fr
        half = width / 2
        blot = np.clip(g.read(g.blots, 0.3 * s, v, 650 / width), -1.5, 1.5)
        hair = g.read(g.hairs, s, v, np.clip(300 / width, 0.6, 1.5))
        hw = half * np.clip(1 + 0.15 * wander(s, L, 1, 0.4 * L + 50, r), 0.7, 1.3)
        out = np.maximum(np.maximum(-s, s - L), 0)
        # a broad brush ends nearly square, its clumps of bristles reaching on past the rest in fingers
        corner = 0.5 * hw
        f = noise.smoothstep(0, 0.12 * half, corner - np.hypot(out, np.maximum(np.abs(v) - hw + corner, 0)) + 0.25 * half * blot
                             + half * np.clip(hair, -2, 2) * (0.06 + 0.08 * noise.smoothstep(0, 0.2 * half, out))
                             + 3 * g.read(g.grit, s, v) + 3 * (g.tops[sl][m] - 0.5))
        tp = noise.smoothstep((1 - r.uniform(0.15, 0.4)) * L, L + 0.2 * half, s + 0.1 * L * blot)
        here = (slice(sl[0].start - y0, sl[0].stop - y0), slice(sl[1].start - x0, sl[1].stop - x0))
        wet[here][m] = np.maximum(wet[here][m], f)
        on = f > 0.3                           # the last stroke over each place leaves its marks there
        for M, val in ((streak, hair), (dry, 1 - 0.7 * tp)):
            mm = M[here][m]
            mm[on] = val[on]
            M[here][m] = mm
    inside = wet > 0.5
    if not inside.any():
        return
    d = ndimage.distance_transform_edt(inside).astype(np.float32)
    lost = noise.smoothstep(0.2, 1.2, g.patch(g.lost, box))
    cover = np.clip(d / (feather[0] + (feather[1] - feather[0]) * lost) + 0.3 * lost * streak, 0, 1) * noise.smoothstep(0.2, 0.5, wet)
    pool = 1 + rim * (1 - lost) * (1.5 * np.exp(-d / 10) * np.clip(1 + streak, 0, 2) - 0.4 * np.exp(-d / 45))
    deep = np.clip(1 + 0.25 * g.patch(g.mottle, box), 0.3, None)
    weave = np.clip(1 + (grain + 1 - cover * dry) * 1.6 * g.hollow[box], 0.05, None)
    T[box] += load * cover * pool * deep * weave * np.clip(1 + comb * streak, 0.2, None) * dry


def cloth(g, a, b, bend, width):
    """Where a rag drawn from a to b touches the canvas: a band that bunches and spreads as the cloth is pushed
    along, its edge torn by the folds, coming on and going off raggedly. -> slices, mask, touch, folds"""
    r, half = g.r, width / 2
    fr = arc(a, b, bend, 1.6 * half, 10)
    if fr is None:
        return None
    sl, m, s, v, L = fr
    folds = np.clip(g.read(g.folds, 0.35 * s, v), -2, 2)
    big = np.clip(g.read(g.blots, 0.35 * s, v, 480 / width), -1.5, 1.5)
    hw = half * np.clip(1 + 0.2 * wander(s, L, 1, 0.5 * L + 50, r), 0.6, 1.3)
    inside = noise.smoothstep(1.0, 0.5, np.abs(v - np.clip(wander(s, L, 0.15 * half, 0.6 * L + 80, r), -0.2 * half, 0.2 * half)) / hw
                              + 0.1 * folds + 0.2 * big)
    reach = min(0.15 * L, 0.6 * half)
    ends = noise.smoothstep(-10, 0.2 * L, s + reach * big) * noise.smoothstep(L + 10, 0.6 * L, s + reach * big)
    return sl, m, inside * ends, folds


def rub(T, g, a, b, bend, width, load):
    """Thin paint rubbed on with a rag: it goes on in the print of the crumpled cloth, in clouds, and more into
    the hollows of the weave than onto the threads."""
    c = cloth(g, a, b, bend, width)
    if c:
        sl, m, touch, folds = c
        T[sl][m] += load * touch * (1 + 0.25 * folds) * (1 + 0.3 * (0.5 - g.tops[sl][m]))


def wipe(Ts, g, a, b, bend, width, amount):
    """A rag wiped through the wet paint: it lifts most from the tops of the threads, so the weave comes up
    white through what is left, and leaves the print of its own folds."""
    c = cloth(g, a, b, bend, width)
    if c:
        sl, m, touch, folds = c
        take = np.clip(amount * touch * (0.5 + 0.5 * g.tops[sl][m]) * (1 + 0.15 * folds), 0, 0.97)
        for T in Ts:
            T[sl][m] *= 1 - take


def pull(T, g, a, b, bend, width, load):
    """A broad veil of thinned colour pulled from a to b in one stroke of a big brush. It is heaviest where the
    brush landed, and swells and narrows and wanders as the pressure comes and goes, its edges torn a little by
    the clumps of bristles; one side of the brush carries more than the other, the paint the bristles push aside
    lies a little thicker along its edges, and the bristles comb the colour into streaks, in bunches and finer.
    Most strokes run dry before the end: all across the width the paint thins and breaks into the streaks of
    the bristles, until only the tops of the weave catch it. The rest go on full to the end, and a rag wipes the
    end away. Each is pulled in from whichever edge of the canvas, the top or the foot, its path starts nearer
    to."""
    r = g.r
    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(b[1], H - b[1]) < min(a[1], H - a[1]):
        a, b, bend = b, a, -bend
    fr = arc(a, b, bend, 0.85 * width, 6)
    if fr is None:
        return
    sl, m, s, v, L = fr
    half = width / 2
    bunch = np.clip(g.read(g.hairs, s, v, np.clip(260 / width, 0.5, 1.4)), -2.5, 2.5)
    comb = np.clip(g.read(g.hairs, s, v, 3.0), -2.5, 2.5)
    wob = np.clip(g.read(g.blots, 0.3 * s, v, 650 / width), -1.5, 1.5)
    folds = np.clip(g.read(g.folds, s, v), -2, 2)
    tops = g.tops[sl][m]
    hw = half * np.clip(1 + 0.22 * wander(s, L, 1, 0.3 * L + 60, r), 0.6, 1.4)
    edge = hw - np.abs(v - wander(s, L, 0.06 * half, 0.5 * L + 80, r)) + half * (0.08 * bunch + 0.02 * comb + 0.03 * folds) + 3 * (tops - 0.5)
    on = noise.smoothstep(-3, 3, edge) * noise.smoothstep(-2, 4, s + 0.1 * half * wob + 0.02 * half * bunch)
    wiped = r.random() < 0.35
    run = 1.0 if wiped else r.uniform(0.6, 0.85)
    # the outermost bristles carry least, so the edges break a little too
    left = (1 - noise.smoothstep(run * L, L + 5, s + 0.1 * (1 - run) * L * wob)) * (1 - 0.3 * noise.smoothstep(0.1 * half, 0, edge))
    catch = noise.smoothstep(-0.15, 0.0, 0.55 * tops + 0.45 * noise.smoothstep(-2, 2, comb) - (1 - left) ** 0.7)
    heap = (1 + 0.5 * np.exp(-np.maximum(s, 0) / (0.3 * half + 20)) + 0.35 * np.exp(-(edge / 6) ** 2)) * (1 + r.uniform(-0.35, 0.35) * v / half)
    # full, the paint levels out between the bristles; thinner, it keeps their streaks
    streaks = np.clip(1 + (0.1 + 0.25 * (1 - left)) * bunch + (0.05 + 0.15 * (1 - left)) * comb, 0.2, None)
    T[sl][m] += load * on * catch * (0.35 + 0.65 * left) * streaks * np.clip(heap, 0.3, None)
    if wiped:
        d = (b - a) / np.hypot(*(b - a))
        q = np.arctan2(d[1], d[0]) + np.pi / 2 + r.normal(0, 0.4)
        e, n = b - d * r.uniform(0, 0.15) * L, np.array([np.cos(q), np.sin(q)]) * width
        wipe([T], g, e - n, e + n, 0, r.uniform(0.2, 0.4) * L + 60, r.uniform(0.7, 0.9))


def drip(g, x, y, length, width):
    """The way a run of thinned paint went: down from (x, y), sagging out of the passage it left, finding its
    way down the weave, narrowing, and stopping in a bead. -> slices, cover, its edges, its bead"""
    r = g.r
    n = max(int(length), 6)
    t = np.linspace(0, 1, n + 1, dtype=np.float32)
    cx = x + (1 + 0.006 * n) * noise.line1d(n + 1, 120, r) * t + 0.4 * noise.line1d(n + 1, 4, r)
    hw = width * (0.35 + 0.65 * (1 - t) ** 1.5) * (1 + 0.15 * noise.line1d(n + 1, 25, r)) * (1 + 0.8 * np.exp(-t * n / 10))
    br = 1.2 * hw[-1] + 0.7
    x0, x1 = max(int((cx - hw).min() - br - 3), 0), min(int((cx + hw).max() + br + 3) + 1, W)
    y0, y1 = max(int(y), 0), min(int(y + n + br + 3) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    k = np.clip(yy - int(y), 0, n).astype(np.intp)
    dd = np.abs(xx - cx[k]) - 0.6 * (g.tops[y0:y1, x0:x1] - 0.5)
    body = noise.smoothstep(hw[k] + 1.2, hw[k] - 1.2, dd) * (yy - int(y) <= n) * noise.smoothstep(0, 15, yy - int(y))
    bead = noise.smoothstep(br + 1, br - 1, np.hypot(xx - cx[-1], yy - int(y) - n))
    side = np.exp(-((hw[k] - dd) / 1.1) ** 2) * body
    return (slice(y0, y1), slice(x0, x1)), np.maximum(body, bead), side, bead


def heavy(T, r, k, thresh):
    """Where a passage was laid heavy enough to run: points along its lower edge, at most k. -> (x, y, density)"""
    g = T[::8, ::8]
    ys, xs = np.nonzero((g > thresh) & (np.roll(g, -4, 0) < 0.5 * g))
    keep = ys < g.shape[0] - 5
    ys, xs = ys[keep], xs[keep]
    pick = r.permutation(len(ys))[:k]
    return np.column_stack([xs[pick] * 8.0, ys[pick] * 8.0 + 12, g[ys[pick], xs[pick]]])


def runs(T, g, src, length, width, load):
    """Thinned paint run down from where it was laid heavy, one or two runs from each place, never in step."""
    r = g.r
    for x, y, dens in src:
        for _ in range(r.integers(1, 3)):
            f = drip(g, x + r.normal(0, 40), y + r.normal(0, 10), r.uniform(*length) * r.uniform(0.3, 1), r.uniform(*width))
            if f:
                sl, cov, side, bead = f
                T[sl] += load * min(dens, 1.5) * r.uniform(0.5, 1.0) * cov * (1 + 0.6 * side + 0.8 * bead)


def coat(img, T, colour, opacity):
    """Oil paint over what is there: a glaze that tints the light coming back up through it, and as much body
    as the pigment has, which covers."""
    ys, xs = np.nonzero(T.any(1))[0], np.nonzero(T.any(0))[0]
    if len(ys):
        sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        t = T[sl][..., None]
        a = 1 - np.exp(-opacity * t)
        img[sl] = img[sl] * np.exp(-t * pigment(colour)) * (1 - a) + lin(colour) * a


def drift(x, y, r):
    """Long shallow arcs, bowed this way and that, as the arm swings."""
    return r.normal(0, 0.25), (r.normal(), r.normal()), r.uniform(1 / 4000, 1 / 1200)


def level(x, y, r):
    """Nearly level, as mist lies on water, but never ruled."""
    return r.normal(0, 0.06), (0, r.choice((-1, 1))), r.uniform(0, 1 / 3000)


def side(x):
    """How far toward the left (+1) or the right (-1) side of the canvas, where the walls stand."""
    return noise.smoothstep(1000, 250, x) - noise.smoothstep(3000, 3650, x)


def stand(x, y, r):
    """Strokes that stand up like the walls of a gorge, leaning in toward the light."""
    s = side(x)
    return np.pi / 2 - 0.35 * s + r.normal(0, 0.12), (np.sign(s), 0), r.uniform(1 / 6000, 1 / 2500)


def walls(x, y, r):
    """At the sides, strokes that stand up; along the top and the foot, long sweeps."""
    return stand(x, y, r) if r.random() < abs(side(x)) else drift(x, y, r)


def mark(r, x, y, ang, length, press):
    """One stroke of the brush, drawn fast, as control rows (x, y, pressure). It bends a little as it goes and
    turns sharply where the hand checks, pressing as it checks; the pressure swells and eases as it goes, and now
    and then the brush lifts nearly off and comes down again. Some strokes land pressed; at the end the brush
    lifts away to a hair, or stops pressed, or simply leaves off. The hand trembles all along."""
    k = max(int(length / 4), 6)
    t = np.linspace(0, 1, k)
    check = ((r.random(k) < 0.8 / k) & (t > 0.15) & (t < 0.85)).astype(float)
    a = ang + r.normal(0, 0.45) * t + 0.12 * noise.line1d(k, max(k / 3, 2), r) + np.cumsum(check * r.choice((-1, 1), k) * r.uniform(0.4, 1.0, k))
    pts = np.array([x, y]) + np.cumsum(np.stack([np.cos(a), np.sin(a)], 1) * length / k, 0)
    pts += np.stack([-np.sin(a), np.cos(a)], 1) * 0.5 * noise.line1d(k, 1.5, r)[:, None]
    p = press * (1 + 0.35 * noise.line1d(k, max(k / 3, 2), r)) + 0.3 * np.convolve(check, [0.4, 1, 0.4], "same")
    if r.random() < 0.2:
        p += r.uniform(0.1, 0.3) * np.exp(-t * length / r.uniform(6, 14))
    end = r.random()
    if end < 0.35:
        p += r.uniform(0.2, 0.5) * np.exp(-(1 - t) * length / 8)
    elif end < 0.8:
        p *= 0.05 + 0.95 * noise.smoothstep(1, 1 - r.uniform(0.15, 0.4), t)
    p *= 1 - (r.random() < 0.3) * 0.92 * np.exp(-((t - r.uniform(0.3, 0.7)) / r.uniform(0.02, 0.06)) ** 2)
    return np.column_stack([pts, np.clip(p, 0.03, 1)])


def cluster(r, cx, cy, rx, ry, n, lean, full):
    """Strokes of the brush drawn fast in a cluster, like broken reeds or twigs, or characters coming apart:
    short strokes bunched toward its middle and leaning about one way, many springing from one already there as
    a twig from a branch, some crossing; and a few short heavy stabs. A share `full` of them go on with a full
    brush. -> [(rows, radius, full brush?)]"""
    out, nodes = [], []
    for _ in range(n):
        L = float(np.clip(np.exp(r.normal(np.log(80), 0.55)), 18, 280))
        if nodes and r.random() < 0.55:
            x, y = nodes[r.integers(len(nodes))]
            ang = lean + r.normal(0, 0.3) + r.choice((-1, 1)) * r.uniform(0.4, 1.3)
        else:
            x, y = cx + rx * r.normal(0, 0.45), cy + ry * r.normal(0, 0.45)
            ang = lean + r.normal(0, 0.4)
        P = mark(r, x, y, ang + np.pi * (r.random() < 0.5), L, r.uniform(0.2, 0.55))
        nodes += list(P[r.integers(0, len(P), 2), :2])
        out.append((P, r.uniform(4, 10) * (1.1 - 0.5 * L / 280), r.random() < full))
    for _ in range(max(1, n // 14)):
        x, y = nodes[r.integers(len(nodes))] + r.normal(0, 25, 2)
        out.append((mark(r, x, y, r.uniform(0, 2 * np.pi), r.uniform(18, 40), r.uniform(0.7, 1.0)), r.uniform(7, 11), r.random() < full))
    return out


def guides(r):
    """The plan of the picture, at a quarter of its size: where the light is, where it deepens toward the
    edges, the two corners that go nearly black, and where the warm passage lies. -> maps 0..1"""
    yy, xx = np.mgrid[0:H:Q, 0:W:Q].astype(np.float32)
    wx, wy = (160 * noise.fbm(yy.shape, 700 / Q, r, octaves=3) for _ in range(2))
    x, y = xx + wx, yy + wy
    core = np.exp(-(np.hypot((x - CX) / 1.6, (y - CY) * np.where(y > CY, 1.15, 1.0)) / 720) ** 2)
    by = 1120 + 50 * noise.fbm((1, yy.shape[1]), 900 / Q, r, octaves=2)
    band = np.exp(-((y - by) / 210) ** 2) * noise.smoothstep(100, 1200, x) * noise.smoothstep(3500, 2800, x)
    light = np.maximum(core, 0.85 * band)
    frame = np.maximum.reduce([noise.smoothstep(650, -50, x), noise.smoothstep(3200, 3900, x),
                               0.45 * noise.smoothstep(320, -80, y), 0.7 * noise.smoothstep(1580, 2000, y)])
    corner = np.maximum(np.exp(-(np.hypot(x / 1.4, y) / 560) ** 2), np.exp(-(np.hypot((x - W) / 1.4, y - H) / 560) ** 2))
    warm = np.exp(-((x - 1150) / 520) ** 2 - ((y - 1440 + 0.12 * (x - 1150)) / 150) ** 2)
    return light, frame * (1 - light) * (1 - 0.6 * warm), corner * (1 - light), warm


def paint(seed=1985):
    r = noise.rng(seed)
    sheet = canvas.duck((H, W), seed, tint=GROUND, thread=3.2)
    g = Ground(sheet, r)
    light, deep, corner, warm = guides(r)
    between = (1 - light) * (1 - deep)
    rim_of_light = 4 * light * (1 - light)
    cool = (1 - warm) ** 2
    upper = noise.smoothstep(1100, 600, np.arange(0, H, Q, dtype=np.float32))[:, None]
    sides = np.abs(side(np.arange(0, W, Q, dtype=np.float32)))[None]
    off_walls = (1 - deep * sides) ** 2
    coats = []

    def layer(colour, opacity):
        T = np.zeros((H, W), np.float32)
        coats.append((T, colour, opacity))
        return T

    def look(where, x, y):
        return float(where[int(np.clip(y, 0, H - 1)) // Q, int(np.clip(x, 0, W - 1)) // Q])

    def sweep(tool, T, gap, where, aim, width, length, load, **kw):
        """Strokes of `tool` about `gap` px apart, laid where the plan `where` (0..1) asks for them and more
        loaded where it is higher, each turned and bowed the way `aim` says."""
        gy, gx = np.mgrid[-300:H + 300:gap, -300:W + 300:gap].astype(np.float32)
        pts = np.stack([gx.ravel(), gy.ravel()], 1) + r.uniform(0, gap, (gx.size, 2))
        for x, y in pts[r.permutation(len(pts))]:
            p = look(where, x, y)
            if r.random() > p:
                continue
            ang, bow, curv = aim(x, y, r)
            Ln, wd = r.uniform(*length), r.uniform(*width)
            d = np.array([np.cos(ang), np.sin(ang)]) * r.choice((-1, 1))
            bend = np.sign(-d[1] * bow[0] + d[0] * bow[1]) * min(curv * Ln * Ln / 8, 0.25 * Ln)
            c = np.array([x, y])
            tool(T, g, c - d * Ln / 2, c + d * Ln / 2, bend, wd, load * (0.5 + 0.5 * p) * r.uniform(0.6, 1.4), **kw)

    def passages(T, gap, where, aim, load, k=(2, 5), width=(180, 420), length=(400, 1300), **kw):
        """Passages about `gap` px apart, laid where the plan asks for them, each a few strokes of the broad
        brush worked back and forth across one another."""
        gy, gx = np.mgrid[-300:H + 300:gap, -300:W + 300:gap].astype(np.float32)
        pts = np.stack([gx.ravel(), gy.ravel()], 1) + r.uniform(0, gap, (gx.size, 2))
        for x, y in pts[r.permutation(len(pts))]:
            p = look(where, x, y)
            if r.random() > p:
                continue
            ang, bow, curv = aim(x, y, r)
            group = []
            for _ in range(int(r.integers(*k))):
                Ln, wd = r.uniform(*length), r.uniform(*width)
                e = ang + r.normal(0, 0.15)
                d = np.array([np.cos(e), np.sin(e)])
                c = np.array([x, y]) + np.array([-d[1], d[0]]) * r.normal(0, 0.45 * wd) + d * r.normal(0, 0.15 * Ln)
                bend = np.sign(-d[1] * bow[0] + d[0] * bow[1]) * min(curv * Ln * Ln / 8, 0.25 * Ln)
                group.append((c - d * Ln / 2, c + d * Ln / 2, bend, wd))
            wash(T, g, group, load * (0.6 + 0.4 * p) * r.uniform(0.7, 1.3), **kw)

    # the first sitting, thin as watercolour: pale gold rubbed on where the light will be and a deeper gold round
    # its heart, red ochre low on the left, cerulean through the rest, ultramarine along the top and the foot and
    # pulled in at the sides; and the light wiped back out of it
    gold = layer(PALE_GOLD, 0.0)
    sweep(rub, gold, 300, light, drift, (350, 700), (600, 1400), 0.16)
    passages(gold, 600, light ** 1.5, drift, 0.18, (2, 4), (250, 500), (700, 1600), rim=0.15, feather=(20, 300))
    glow = layer(GOLD, 0.0)
    passages(glow, 560, 4 * light ** 2 * (1 - light ** 2) * cool, drift, 0.18, (2, 4), (220, 420), (600, 1400), rim=0.15, feather=(20, 300))
    ochre = layer(OCHRE, 0.05)
    sweep(rub, ochre, 260, warm, level, (250, 450), (400, 900), 0.25)
    passages(ochre, 220, warm, level, 0.26, (2, 4), (200, 380), (400, 1000), rim=0.3, feather=(10, 240))
    sky = layer(CERULEAN, 0.0)
    passages(sky, 600, between * (1 - light) ** 2 * cool * (0.4 + 0.6 * upper), drift, 0.25, (3, 6), (300, 600), (600, 1600))
    sweep(rub, sky, 340, between * cool * (1 - upper), level, (300, 600), (600, 1400), 0.12)
    blue = layer(ULTRA, 0.0)
    passages(blue, 560, deep * cool * (1 - sides), drift, 0.15, (3, 6), (320, 640), (600, 1600))
    sweep(pull, blue, 240, deep * cool * sides, stand, (120, 440), (700, 1700), 0.24)
    sweep(wipe, [sky, blue], 200, warm, level, (200, 400), (400, 1000), 0.8)
    sweep(wipe, [sky, blue, ochre], 170, light ** 1.5, level, (150, 350), (400, 1100), 0.85)
    sweep(wipe, [gold, glow], 220, light ** 4, drift, (120, 300), (300, 900), 0.6)

    # the second, over the first once it had dried: cobalt and violet where the light gives out, warm grey round
    # its edge, ultramarine again along the top and the foot, madder through the ochre; the wettest runs
    cobalt = layer(COBALT, 0.0)
    passages(cobalt, 700, between * (1 - light) ** 2 * cool, walls, 0.17, (3, 6), (300, 600), (600, 1600))
    violet = layer(VIOLET, 0.0)
    passages(violet, 650, rim_of_light * cool * upper * off_walls, level, 0.1, (2, 4), (200, 400), (600, 1400), rim=0.15, feather=(20, 320))
    pearl = layer(WARM_GREY, 0.1)
    passages(pearl, 560, rim_of_light * cool * off_walls, level, 0.14, (2, 4), (200, 400), (600, 1400), rim=0.15, feather=(20, 320))
    blue2 = layer(ULTRA, 0.0)
    passages(blue2, 640, deep ** 1.5 * cool * (1 - sides), drift, 0.08, (3, 6), (320, 640), (600, 1600))
    sweep(wipe, [cobalt, blue2, violet], 220, warm, level, (200, 400), (400, 1000), 0.8)
    madder = layer(MADDER, 0.05)
    passages(madder, 260, warm ** 1.5, level, 0.19, (2, 4), (140, 300), (300, 800), rim=0.3, feather=(8, 200))
    runs(ochre, g, heavy(ochre, r, 7, 0.3), (120, 420), (2.5, 4.5), 0.8)
    runs(cobalt, g, heavy(cobalt, r, 4, 0.25), (150, 500), (2, 4), 0.4)
    sweep(wipe, [cobalt, violet, pearl], 480, 4 * deep * (1 - deep), drift, (200, 400), (400, 900), 0.35)
    sweep(wipe, [cobalt, violet, pearl], 260, light ** 3, level, (150, 300), (300, 900), 0.6)

    # the third: white rubbed into the edge of the light like mist, broad wet strokes of white through the
    # core, and a last thin pale gold rubbed over it to make the whites burn
    mist = layer(MIST, 0.6)
    sweep(rub, mist, 260, rim_of_light * cool * off_walls, level, (300, 600), (600, 1500), 0.12)
    white = layer(WHITE, 2.0)
    passages(white, 520, light ** 3 * cool, drift, 0.16, (2, 4), (250, 450), (600, 1400), rim=0.1, feather=(30, 300))
    sweep(rub, white, 260, light ** 3 * cool, drift, (250, 500), (400, 1100), 0.12)
    sweep(rub, layer(PALE_GOLD, 0.0), 300, light, drift, (300, 600), (600, 1400), 0.1)

    # the fourth, so that the light would glow: ultramarine and indigo pulled in again at the sides over
    # everything, a few strokes nearly black among them, indigo heaviest in two corners; and the heaviest ran
    walls2 = layer(ULTRA, 0.0)
    sweep(pull, walls2, 280, deep ** 1.5 * cool * sides, stand, (100, 400), (600, 1600), 0.22)
    indigo = layer(INDIGO, 0.05)
    sweep(pull, indigo, 260, corner ** 1.2 * (corner > 0.2), stand, (200, 480), (600, 1400), 0.4)
    sweep(pull, indigo, 300, deep ** 2.5 * cool * sides, stand, (90, 340), (500, 1500), 0.26)
    sweep(pull, layer(INK, 0.1), 500, deep ** 3 * cool * sides, stand, (40, 200), (300, 1100), 0.45)
    runs(indigo, g, heavy(indigo, r, 9, 0.5), (120, 450), (2.5, 5), 0.8)
    runs(walls2, g, heavy(walls2 * (deep > 0.5).repeat(Q, 0).repeat(Q, 1)[:H, :W], r, 8, 0.4), (100, 380), (2.5, 5), 0.8)

    # the smaller brush, fast, in clusters through the middle: strokes with a full brush, which soften a little
    # into the wet paint round them as the thinned paint creeps out along the hollows of the weave, and strokes
    # with a brush nearly dry; the weave breaks the thin edges of the one and all of the other
    full, dry, water = (np.zeros((H, W), np.float32) for _ in range(3))
    for cx, cy, rx, ry, n, lean, share in CLUSTERS:
        for P, rad, wet in cluster(r, cx, cy, rx, ry, n, lean, share):
            L = np.hypot(*np.diff(P[:, :2], axis=0).T).sum()
            kw = (dict(load=r.uniform(0.9, 1.2), reach=2 * L + 200, clumps=int(r.integers(6, 10)), splay=0.15,
                       dryness=r.uniform(0.6, 1.0), head=0.8, strength=r.uniform(1.0, 1.3)) if wet else
                  dict(load=r.uniform(0.6, 0.9), reach=L * r.uniform(0.6, 1.2) + 30, clumps=int(r.integers(3, 7)), splay=0.4,
                       dryness=r.uniform(2.0, 3.5), head=0.5, strength=r.uniform(0.9, 1.1)))
            brush.stroke(sheet, P, rad, int(r.integers(1 << 30)), into=(full if wet else dry, water), **kw)
    full *= noise.smoothstep(0.0, 0.25, full + 0.25 * (g.tops - 0.5))
    dry *= noise.smoothstep(0.0, 0.4, dry + 0.35 * (g.tops - 0.5))
    ink = layer(INK, 0.5)
    ink += full + dry + 0.6 * np.maximum(ndimage.gaussian_filter(full, 3) - full, 0) * np.clip(1 + 2 * g.hollow, 0.2, None)

    img = sheet.color.copy()
    for T, colour, opacity in coats:
        coat(img, T, colour, opacity)

    # a few touches laid thick near the light: short strokes of a brush loaded with white and pale gold side by
    # side, heaped where it came down, combed by its bristles and breaking over the weave as it ran out; they
    # stand up from the rest and catch the light
    body = sum(T * min(o, 1.0) for T, _, o in coats)
    height = 0.4 * sheet.tooth * np.exp(-2.0 * body) + 0.1 * body
    paths = []
    for x, y, k in ((2290, 790, 3), (2075, 935, 2), (2560, 900, 2)):
        for _ in range(k):
            a, L, c = r.normal(0, 0.35), r.uniform(110, 200), np.array([x, y]) + r.normal(0, 22, 2)
            t, d = np.linspace(-0.5, 0.5, 5)[:, None], np.array([np.cos(a), np.sin(a)])
            paths.append(c + t * L * d + (0.25 - t * t) * r.normal(0, 0.25) * L * np.array([-d[1], d[0]]))
    n = len(paths)
    u = r.uniform(0.4, 0.9, n)
    impasto.lay(img, height, paths, r.uniform(18, 30, n), np.tile(np.stack([lin(WHITE), lin(PALE_GOLD)]), (n, 1, 1)), r,
                share=np.stack([u, 1 - u], 1), thick=0.45, spent=0.55, grooves=0.6, lips=0.2, land=0.7, tails=0.5,
                pickup=0.3, fray=1.5, ends=(0.5, 0.2))

    height = ndimage.gaussian_filter(height, 0.7)
    out = dabs.shine(img, height, relief=0.5, gloss=0, reach=(0.8, 1.12))
    return out + 0.15 * impasto.glints(height)[..., None]
