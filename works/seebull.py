"""Poppies and Dahlias, Evening. Watercolour on Japan paper.

Nolde painted the flowers of his garden at Seebüll, and the marsh round it, on Japan paper, thin and
soft, which drinks a wash the moment the brush touches it. He worked fast on sheets still damp,
laying one strong colour against the next and letting them run, dropped darker colour or clear water
into them as they dried, and took the forms that came out of the pools as they settled. Forbidden to
paint from 1941, he went on in secret with small sheets of the same kind, the Unpainted Pictures.

Here each poppy went down a petal at a time from the back, every petal one pool of scarlet, crimson or
mauve laid out from the heart with a crumpled rim, the next laid over it once it had dried, so that the
petals show through one another; now and then two went down together and ran into one. Deeper colour
was run into the root of each and along the edge the next would cover, the black of the heart dropped
in last to creep out into the wet red, and when that was dry the stamens were dragged out of it with an
almost dry point. The big scarlet poppy faces out, two are turned away, the low one is seen from the
side, a cup. The orange dahlia is three rings of petals laid with the point, the outer ring first; the
white dahlia is the paper, left clear, with grey strokes laid along its petals where each lies over the
one behind. The ground went round them in rows of broad strokes on the damp sheet: Prussian blue and
violet above, red clouds and the orange of the after-glow on the right, the greens of the garden below,
with blue-black touched into them under the flowers. A drop of water fell into the blue, water crept
back in from the edge of the sheet, and each pushed the colour ahead of it into a frilled line; the
long fibres of the paper took less colour than the rest. Last came the leaves, the delphinium floret by
floret in cobalt, ultramarine and violet, with buds at its tip and its stem showing between, and the
blue-black of the darkest leaves and stems.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import noise, paper, plate, wash
from atelier import watercolour as wc
from atelier.color import pigment
from atelier.paper import Sheet

TITLE = "Poppies and Dahlias, Evening"
DATE = "2026"
MEDIUM = "Watercolour on Japan paper"
AFTER = ("Emil Nolde, the flower and marsh watercolours on Japan paper from Utenwarf and Seebüll, 1920s–1940s: "
         "Marsh Landscape with Red Clouds (early 1920s); Sunflowers (1925–28); Poppies (c. 1930); and the "
         "Unpainted Pictures (Ungemalte Bilder), 1938–45")
ROOM = "Paper and Water"
YEAR = 1935
PLACE = "Seebüll"
REGION = "Europe"
NOTE = ("Poppies, dahlias and a spike of delphinium against an evening sky, laid wet into wet on damp Japan "
        "paper and left to run; the white dahlia is the bare paper.")

H, W = 2400, 3240
FULL = (slice(0, H), slice(0, W))
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

PALETTE = dict(   # absorbance at unit density
    lemon=pigment("#f4df3c"), yellow=pigment("#f7c21a"), orange=pigment("#f1801e"),
    scarlet=pigment("#e63c26"), crimson=pigment("#c01c48"), rose=pigment("#d2448e"),
    violet=pigment("#7646a6"), ultramarine=pigment("#3a4ec6"), cobalt=pigment("#3f78cc"),
    prussian=pigment("#24557f"), emerald=pigment("#1f9a74"), sap=pigment("#7ea032"),
    sienna=pigment("#a8522a"), black=pigment("#47454c"))
GRAIN = dict(lemon=0.05, yellow=0.1, orange=0.15, scarlet=0.2, crimson=0.1, rose=0.1, violet=0.5,
             ultramarine=0.6, cobalt=0.45, prussian=0.15, emerald=0.35, sap=0.2, sienna=0.5, black=0.3)

FAMILIES = dict(  # the colours of the ground, as the brush picked them up: density, mix
    blue=(1.0, dict(prussian=0.6, ultramarine=0.4)),
    deep=(1.3, dict(prussian=0.7, ultramarine=0.15, black=0.15)),
    violet=(0.85, dict(ultramarine=0.45, crimson=0.3, rose=0.25)),
    red=(1.1, dict(scarlet=0.75, orange=0.25)),
    orange=(0.9, dict(orange=0.7, scarlet=0.1, yellow=0.2)),
    yellow=(0.5, dict(yellow=0.6, lemon=0.4)),
    green=(1.0, dict(prussian=0.4, emerald=0.45, sap=0.15)),
    moss=(0.9, dict(sap=0.6, yellow=0.25, emerald=0.15)),
    night=(1.15, dict(prussian=0.55, violet=0.25, emerald=0.2)),
)

POPPIES = dict(   # the colour of the petals, the deeper colour run into them, the lighter some were laid in
    scarlet=(dict(scarlet=0.85, crimson=0.15), dict(crimson=0.8, scarlet=0.2), dict(scarlet=0.5, orange=0.5)),
    crimson=(dict(crimson=0.65, scarlet=0.35), dict(crimson=0.6, violet=0.4), dict(scarlet=0.6, orange=0.4)),
    mauve=(dict(rose=0.55, violet=0.45), dict(violet=0.6, ultramarine=0.4), dict(rose=1.0)),
    small=(dict(scarlet=0.6, crimson=0.4), dict(crimson=0.8, violet=0.2), dict(scarlet=0.6, orange=0.4)),
    low=(dict(scarlet=0.7, crimson=0.3), dict(crimson=0.7, violet=0.3), dict(orange=0.6, scarlet=0.4)))

BLUES = (   # the delphinium's blues as the brush took them up: density, mix
    (0.45, dict(cobalt=0.6, ultramarine=0.3, rose=0.1)), (0.75, dict(cobalt=0.75, ultramarine=0.25)),
    (1.0, dict(ultramarine=0.85, prussian=0.15)), (1.3, dict(ultramarine=0.65, prussian=0.2, violet=0.15)),
    (0.85, dict(ultramarine=0.55, violet=0.3, rose=0.15)))

DARK = (dict(prussian=0.55, black=0.25, emerald=0.2), dict(prussian=0.5, black=0.2, violet=0.15, emerald=0.15),
        dict(prussian=0.45, emerald=0.35, black=0.2))


def _s(r):
    return int(r.integers(1 << 31))


def coarse(a, k):
    """`a` averaged over k x k blocks."""
    if k == 1:
        return a
    h, w = a.shape
    a = np.pad(a, ((0, -h % k), (0, -w % k)), mode="edge")
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))


def fine(a, k, shape):
    return a if k == 1 else ndimage.zoom(a, k, order=1, grid_mode=True, mode="nearest")[:shape[0], :shape[1]]


def carry(c, wet, s, r, finger=0.0):
    """Colour charged at `c` into the water `wet`, carried about `s` px through it; `finger` px is how
    far one wet colour pushed into the next in tongues as they met. Worked on a coarse grid, since what
    the water carries is smooth: the crisp things are the edges, rims and fronts. -> concentration"""
    k = int(np.clip(s // 5, 1, 8))
    a, w = coarse(c * wet, k), coarse(wet, k)
    if finger:
        f = lambda: finger / k * (noise.field(a.shape, 110 / k, r) + 0.5 * noise.field(a.shape, 30 / k, r))
        a = noise.warp(a, f(), f())
    a = ndimage.gaussian_filter(a, s / k) / (ndimage.gaussian_filter(w, s / k) + 1e-3)
    return fine(a, k, c.shape).astype(np.float32)


def ring(n, scale, r):
    """A smooth random signal round a loop of n samples: unit variance, features about `scale` samples."""
    f = np.fft.rfftfreq(n) * scale
    s = np.fft.irfft((r.standard_normal(len(f)) + 1j * r.standard_normal(len(f))) * np.exp(-4 * f * f) * (f > 0), n)
    return ((s - s.mean()) / (s.std() + 1e-9)).astype(np.float32)


def front(x, y, R, r, sl, toward=None, spread=1.2, lobe=(0.06, 0.2)):
    """Water run out from (x, y) through a damp wash, as signed px inside its front over the plate box
    `sl`: all round, or in a fan `spread` radians either side of the angle `toward`. Its front is a row
    of round lobes of every size, each where the water found a way. -> (signed px, angle from (x, y))"""
    n = 1024
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    base = R * (1 + 0.22 * ring(n, 220, r) + 0.08 * ring(n, 60, r))
    if toward is not None:
        base *= noise.smoothstep(spread, 0.3 * spread, np.abs(np.angle(np.exp(1j * (phi - toward))))) ** 0.5
    reach = 0.85 * base
    i0 = i = r.uniform(0, n)
    while i < i0 + n:
        c = int(i) % n
        rad = min(R * r.uniform(*lobe), 0.5 * base[c])
        if rad < 2:
            i += n / 256
            continue
        hw = rad / base[c] / (2 * np.pi) * n
        k = np.arange(int(i - hw), int(i + hw) + 1)
        off = np.clip((k - i) / hw, -1, 1)
        reach[k % n] = np.maximum(reach[k % n], base[c] - rad + rad * np.sqrt(1 - off * off))
        i += hw * r.uniform(1.1, 1.7)
    yy, xx = np.mgrid[sl].astype(np.float32)
    th = np.arctan2(yy - y, xx - x)
    return reach[((th + np.pi) / (2 * np.pi) * n).astype(int) % n] - np.hypot(xx - x, yy - y), th


def around(x, y, R, within=FULL):
    """The plate box of radius R round (x, y), cut to the slices `within`."""
    y0, y1 = int(max(within[0].start, y - R - 6)), int(min(within[0].stop, y + R + 7))
    x0, x1 = int(max(within[1].start, x - R - 6)), int(min(within[1].stop, x + R + 7))
    return (slice(y0, y1), slice(x0, x1)) if y1 > y0 and x1 > x0 else None


def span(outlines, pad=40):
    """The plate box round all the outlines."""
    P = np.concatenate(outlines)
    return (slice(max(0, int(P[:, 1].min()) - pad), min(H, int(P[:, 1].max()) + pad)),
            slice(max(0, int(P[:, 0].min()) - pad), min(W, int(P[:, 0].max()) + pad)))


def rel(inner, outer):
    """The plate box `inner` in the coordinates of the box `outer` that holds it."""
    return (slice(inner[0].start - outer[0].start, inner[0].stop - outer[0].start),
            slice(inner[1].start - outer[1].start, inner[1].stop - outer[1].start))


def polys(P, blur=0.7, sl=FULL):
    """Polygons (plate px) filled into a 0..1 mask over the plate box `sl`."""
    y0, x0 = sl[0].start, sl[1].start
    im = Image.new("L", (sl[1].stop - x0, sl[0].stop - y0), 0)
    draw = ImageDraw.Draw(im)
    for p in P:
        draw.polygon([tuple(q) for q in np.asarray(p, np.float64) - (x0, y0)], fill=255)
    return ndimage.gaussian_filter(np.asarray(im, np.float32) / 255, blur)


def touch(x, y, L, wd, a, r, bend=0.0, start=0.0, belly=0.35, blunt=0.7, tip=0.9, edge=0.07):
    """The outline of one stroke of a round brush: it lands at (x, y), travels L along `a`, swells
    to `wd` a `belly` of the way along (from `start` of that as it lands) and lifts, to a point if
    `tip` is near 1, round if it is near 0.5; `bend` bows it; its sides are not quite smooth."""
    t = np.linspace(0, 1, int(np.clip(L / 5, 12, 80)))
    rise = start + (1 - start) * np.sin(0.5 * np.pi * np.minimum(t / belly, 1)) ** blunt
    fall = np.cos(0.5 * np.pi * np.clip((t - belly) / (1 - belly), 0, 1)) ** tip
    half = 0.5 * wd * np.where(t < belly, rise, fall)
    k = len(t)
    side = lambda: 1 + edge * (noise.line1d(k + 8, 6, r)[:k] + 0.5 * noise.line1d(k + 8, 2, r)[:k])
    along, mid = t * L, 4 * bend * L * t * (1 - t)
    px = np.concatenate([along, along[::-1]])
    py = np.concatenate([mid + half * side(), (mid - half * side())[::-1]])
    c, s = np.cos(a), np.sin(a)
    return np.stack([x + c * px - s * py, y + s * px + c * py], 1)


def cane(P, r):
    """A stem drawn with a loaded brush through the points P [(x, y, width)], a touch for each stretch."""
    P = np.asarray(P, np.float64)
    out = []
    for (x0, y0, w0), (x1, y1, w1) in zip(P[:-1], P[1:]):
        L = np.hypot(x1 - x0, y1 - y0)
        out.append(touch(x0, y0, L * 1.06, (w0 + w1) / 2, np.arctan2(y1 - y0, x1 - x0), r, bend=r.normal(0, 0.03),
                         start=0.85, belly=0.15, blunt=0.5, tip=0.3, edge=0.06))
    return out


def brushwork(strokes, at=FULL, boost=0.35, blur=0.8):
    """Strokes [(outline, density, mix)] laid beside and over one another in one wet passage over the
    box `at`; where two overlap the brush left more. -> (where the water went, charges for Box.wash)"""
    h, w = at[0].stop - at[0].start, at[1].stop - at[1].start
    count = np.zeros((h, w), np.float32)
    acc = {}
    for P, d, mix in strokes:
        b = (slice(max(at[0].start, int(P[:, 1].min()) - 3), min(at[0].stop, int(P[:, 1].max()) + 4)),
             slice(max(at[1].start, int(P[:, 0].min()) - 3), min(at[1].stop, int(P[:, 0].max()) + 4)))
        if b[0].stop <= b[0].start or b[1].stop <= b[1].start:
            continue
        m = polys([P], blur, b)
        loc = rel(b, at)
        count[loc] += m
        tot = sum(mix.values())
        for k, f in mix.items():
            acc.setdefault(k, np.zeros((h, w), np.float32))[loc] += m * f / tot * d
    norm = (1 + boost * np.clip(count - 1, 0, 2)) / np.maximum(count, 1)
    return np.clip(count, 0, 1), [(acc[k] * norm, 1.0, {k: 1.0}) for k in acc]


def folds(outlines, sl, reach):
    """The shadow each petal throws on those behind it along its edge, `reach` px deep; the first
    petals lie in front."""
    front_ = out = None
    for P in outlines:
        m = polys([P], 0.8, sl)
        if front_ is None:
            front_, out = m, np.zeros_like(m)
            continue
        d = ndimage.distance_transform_edt(front_ < 0.5)
        out = np.maximum(out, m * (1 - front_) * np.exp(-d / reach))
        front_ = np.maximum(front_, m)
    return out


def near(th, a, w):
    """1 at angle a, falling off over w radians."""
    return np.exp(-(np.angle(np.exp(1j * (th - a))) / w) ** 2)


def vary(mix, r, amount=0.3, extra=()):
    """A mix as the brush took it up this time: the shares a little different, now and then a touch of
    something else."""
    out = {k: max(0.0, f * (1 + r.uniform(-amount, amount))) for k, f in mix.items()}
    for k in extra:
        if r.uniform() < 0.3:
            out[k] = out.get(k, 0) + r.uniform(0.1, 0.35)
    return out


class Head:
    """A flower head as the painter faces it, with its own frame: u, v across it, seen at a slant
    (squashed by q across the turn `rot`); its heart at `heart` (in R) in that frame."""

    def __init__(self, cx, cy, R, q, rot, heart=(0.0, 0.0)):
        self.cx, self.cy, self.R, self.q, self.rot, self.heart = cx, cy, R, q, rot, heart
        self.box = around(cx, cy, 1.25 * R)

    def place(self, P):
        """Flower-frame points (u, v) to plate px."""
        u, v = P[:, 0], P[:, 1] * self.q
        c, s = np.cos(self.rot), np.sin(self.rot)
        return np.stack([self.cx + u * c - v * s, self.cy + u * s + v * c], 1)

    def point(self, u, v):
        """The flower-frame point (u, v), in R, in plate px."""
        return self.place(np.array([[u * self.R, v * self.R]]))[0]

    def at(self, rho, a):
        """Points `rho` px out from the heart along the angles `a`, in plate px."""
        rho, a = np.broadcast_arrays(np.atleast_1d(rho), np.atleast_1d(a))
        hu, hv = self.heart[0] * self.R, self.heart[1] * self.R
        return self.place(np.stack([hu + rho * np.cos(a), hv + rho * np.sin(a)], 1))

    def polar(self):
        """(radius / R, angle) from the heart of every px in the head's box, in the flower's own frame."""
        yy, xx = np.mgrid[self.box].astype(np.float32)
        hx, hy = self.point(*self.heart)
        dx, dy = xx - hx, yy - hy
        c, s = np.cos(self.rot), np.sin(self.rot)
        u, v = dx * c + dy * s, (-dx * s + dy * c) / self.q
        return np.hypot(u, v) / self.R, np.arctan2(v, u)

    def petals(self, r, n, reach=(0.75, 1.0), width=(0.55, 0.85), start=(0.05, 0.2), turn=0.35, a0=None,
               shape=dict(start=0.35, belly=0.7, blunt=0.8, tip=0.5)):
        """Strokes laid out from near the heart, each a petal: (outline in plate px, its angle)."""
        a0 = r.uniform(0, 2 * np.pi) if a0 is None else a0
        ang = a0 + 2 * np.pi * np.arange(n) / n + r.normal(0, 0.9 / n, n)
        out = []
        for a in ang:
            r0 = self.R * r.uniform(*start)
            s = a + r.uniform(-turn, turn)
            L = self.R * r.uniform(*reach) - r0
            P = touch(r0 * np.cos(a), r0 * np.sin(a), L, self.R * r.uniform(*width), s, r,
                      bend=r.uniform(-0.12, 0.12), **shape)
            out.append((self.place(P), a))
        return out

    def fan(self, r, a, span_, reach, root=(0.0, 0.0), crinkle=0.06):
        """One poppy petal as the loaded brush laid it out from the heart: a fan `span_` radians wide
        about the angle `a`, `reach` of R long from `root` (in R), its rim crumpled like tissue and
        nicked here and there. -> outline in plate px"""
        n = 140
        t = np.linspace(-1, 1, n)
        rad = reach * (1 - t * t) ** 0.55
        rad = rad * (1 + crinkle * (noise.line1d(n + 8, 28, r)[:n] + 0.5 * noise.line1d(n + 8, 9, r)[:n]))
        for c in r.uniform(-0.7, 0.7, int(r.integers(0, 3))):
            rad = rad * (1 - r.uniform(0.05, 0.1) * np.exp(-((t - c) / r.uniform(0.04, 0.08)) ** 2))
        ang = a + 0.5 * span_ * t
        return self.place(self.R * np.stack([root[0] + rad * np.cos(ang), root[1] + rad * np.sin(ang)], 1))

    def lip(self, r, top, width, bottom, sag, crinkle=0.04):
        """The near petal of a cup turned from us, seen from outside: its crumpled rim runs across the
        flower at `top` (in R; `sag` lower in the middle) `width` either side, and it rounds down to
        `bottom`. -> outline in plate px"""
        n = 100
        u = np.linspace(-width, width, n)
        rim = top + sag * (1 - (u / width) ** 2) + crinkle * (noise.line1d(n + 8, 25, r)[:n]
                                                             + 0.5 * noise.line1d(n + 8, 8, r)[:n])
        t = np.linspace(0, np.pi, n)
        body = np.stack([width * np.cos(t), top + (bottom - top) * np.sin(t)], 1)
        return self.place(self.R * np.concatenate([np.stack([u, rim], 1), body]))

    def mask(self, outlines, close=0):
        """The water the strokes laid, with no dry hole left inside it and, if `close` px, no chink
        narrower than that between them."""
        m = np.zeros((H, W), np.float32)
        local = polys(outlines, 0.8, self.box)
        solid = ndimage.binary_fill_holes(local > 0.5)
        if close:
            p = close + 2
            grown = ndimage.distance_transform_edt(~np.pad(solid, p)) <= close
            solid = (ndimage.distance_transform_edt(grown) > close)[p:-p, p:-p]
        m[self.box] = np.maximum(local, ndimage.gaussian_filter(solid.astype(np.float32), 0.8))
        return m


class Box:
    """The paint box. Transparent colours multiply, so each is kept as absorbance and glazed onto
    the paper at the end; the order they went on matters only for where the water ran."""

    def __init__(self, sheet, r, keep):
        self.sheet, self.r, self.keep = sheet, r, keep
        self.A = np.zeros((H, W, 3), np.float32)
        self.soak = np.zeros((H, W), np.float32)
        t = ndimage.gaussian_filter(sheet.tooth, 1.0)
        self.valley = (t.mean() - t) / t.std()
        # the sheet drinks unevenly: its long fibres take less, and it is thicker in clouds
        self.take = ((1 - 0.15 * sheet.fiber) * (1 + 0.1 * noise.fbm((H, W), 60, r, octaves=3))
                     * (1 + 0.05 * noise.field((H, W), 2.0, r))).astype(np.float32)

    def lay(self, d, sl, **mix):
        """Pigment of density `d` over the plate box `sl`, as the sheet took it; the heavy pigments
        settle in the hollows."""
        K = sum(f * PALETTE[k] for k, f in mix.items())
        g = sum(f * GRAIN[k] for k, f in mix.items()) / sum(mix.values())
        self.A[sl] += (d * self.take[sl] * np.clip(1 + 0.3 * g * self.valley[sl], 0, None))[..., None] * K

    def wash(self, mask, charges, at=FULL, spread=30.0, ragged=1.5, finger=0.0, blooms=(), margins=(),
             feather=0.25, **dep):
        """One passage wet into wet: water laid over `mask` (over the plate box `at`), then the colours
        of `charges` [(where, density, {pigment: share})] run into it and carried about `spread` px; it
        dries with the pools, rims and granulation of `deposit`, the backruns of `blooms` (see `bloom`)
        and `margins` (see `margin`), and a little colour wicked out along the fibres at its edge.
        -> where the water lay, over `at`"""
        r, sh = self.r, self.sheet
        mask = mask * self.keep[at]
        ys, xs = np.nonzero(mask > 0.02)
        if not len(ys):
            return np.zeros_like(mask)
        y0, x0 = at[0].start, at[1].start
        sl = (slice(max(0, y0 + ys.min() - 30), min(H, y0 + ys.max() + 31)),
              slice(max(0, x0 + xs.min() - 30), min(W, x0 + xs.max() + 31)))
        sub = Sheet(*(a[sl] for a in (sh.color, sh.tooth, sh.fiber, sh.alpha)))
        inner = (slice(max(sl[0].start, y0), min(sl[0].stop, at[0].stop)),
                 slice(max(sl[1].start, x0), min(sl[1].stop, at[1].stop)))

        def fit(a):
            out = np.zeros(sub.tooth.shape, np.float32)
            out[rel(inner, sl)] = a[rel(inner, at)]
            return out
        wet = wc.puddle(fit(mask), sub, r, ragged)
        f = wc.deposit(wet, sub, r, **dep)
        for b in blooms:
            self.bloom(f, sl, wet, *b)
        for b in margins:
            self.margin(f, sl, wet, *b)
        if feather:
            reach = ndimage.gaussian_filter(wet, 2.5) * (0.3 + 1.4 * sub.fiber)
            f = f + feather * noise.smoothstep(0.08, 0.3, reach) * (1 - wet)
        for where, dens, mix in charges:
            self.lay(carry(fit(where), wet, spread, r, finger) * f * dens, sl, **mix)
        self.soak[sl] = np.maximum(self.soak[sl], wet)
        out = np.zeros_like(mask)
        out[rel(inner, at)] = wet[rel(inner, sl)]
        return out

    def bloom(self, f, sl, wet, x, y, R, pale=0.4, line=0.9, toward=None, spread=1.2):
        """Water crept back into a drying wash: all round (x, y) where a drop fell, or in a fan about
        `toward` where it crept in from an edge. It pushed the colour out ahead of it in fingers, leaving
        the inside paler, and dropped it in a dark frilled line where it stopped."""
        r = self.r
        b = around(x, y, 1.7 * R, sl)
        if b is None:
            return
        s, ang = front(x, y, R, r, b, toward, spread, lobe=(0.04, 0.12) if toward is None else (0.02, 0.06))
        s += 0.9 * noise.field(s.shape, 2.0, r) + 2.5 * (self.sheet.fiber[b] - 0.45)
        loc = rel(b, sl)
        w = noise.smoothstep(0.3, 0.9, wet[loc])
        brk = np.clip(0.75 + 0.5 * noise.field(s.shape, 25, r), 0.15, 1.4)
        # water creeping in from an edge ran in fingers, so the hollow is paler along some rays than others
        rays = 1 if toward is None else 1 + 0.3 * ring(512, 9, r)[((ang + np.pi) / (2 * np.pi) * 512).astype(int) % 512]
        hollow = noise.smoothstep(0, 0.05 * R, s) * rays * (1 + 0.12 * noise.field(s.shape, 8, r))
        # the pigment banks up behind the front: sharp where the water stopped, fading back from it
        edge = noise.smoothstep(-0.8, 0.8, s) * np.exp(-np.clip(s, 0, None) / (1.5 + R / 60)) * brk
        f[loc] *= 1 + w * (line * edge - pale * hollow)

    def margin(self, f, sl, wet, x, y, R, width, pale=0.45, line=1.0):
        """Water crept back along the edge of a pool for about R either side of (x, y) on it: a pale
        band about `width` px wide inside the edge, the colour it pushed banked in a dark frilled line
        along its inner side."""
        r = self.r
        b = around(x, y, R, sl)
        if b is None:
            return
        loc = rel(b, sl)
        inside = ndimage.distance_transform_edt(wet[loc] > 0.5)
        yy, xx = np.mgrid[b].astype(np.float32)
        along = np.exp(-2 * (np.hypot(xx - x, yy - y) / R) ** 2)
        s = width * along * (0.7 + 0.3 * noise.field(inside.shape, 35, r)) + 0.12 * width * noise.field(inside.shape, 5, r) \
            + 2.0 * (self.sheet.fiber[b] - 0.45) - inside
        w = noise.smoothstep(0.3, 0.9, wet[loc]) * noise.smoothstep(0.1, 0.35, along)
        brk = np.clip(0.75 + 0.5 * noise.field(s.shape, 20, r), 0.15, 1.4)
        edge = noise.smoothstep(-0.8, 0.8, s) * np.exp(-np.clip(s, 0, None) / 2.5) * brk
        f[loc] *= 1 + w * (line * edge - pale * noise.smoothstep(0, 4, s))

    def drop(self, x, y, R, dens, within, at=FULL, line=0.5, soft=0.5, **mix):
        """Colour touched into a damp passage at (x, y): it creeps out about R through the damp
        (`within`, over the box `at`) and stops in a frilled front, a little darker where it stopped."""
        r = self.r
        b = around(x, y, 1.7 * R, at)
        if b is None:
            return
        s, _ = front(x, y, R, r, b, lobe=(0.12, 0.3))
        s += 0.8 * noise.field(s.shape, 2.0, r) + 2.0 * (self.sheet.fiber[b] - 0.45)
        brk = np.clip(0.75 + 0.5 * noise.field(s.shape, 25, r), 0.15, 1.4)
        fill = noise.smoothstep(-1, 1.5, s) * (1 - soft + soft * noise.smoothstep(0, 0.6 * R, s))
        fill = fill + line * noise.smoothstep(-0.8, 0.8, s) * np.exp(-np.clip(s, 0, None) / (1.5 + R / 60)) * brk
        self.lay(fill * dens * within[rel(b, at)] * self.keep[b], b, **mix)

    def flick(self, x, y, L, wd, a, dens, dry=0.4, shape=dict(start=0.6, belly=0.15, blunt=0.6, tip=0.8), **mix):
        """A stroke of the point almost dry, from (x, y) along `a` over dry paint: it caught only the
        tops of the fibres and ridges of the paper, more broken the further it went."""
        r = self.r
        P = touch(x, y, L, wd, a, r, bend=r.uniform(-0.15, 0.15), **shape)
        b = (slice(max(0, int(P[:, 1].min()) - 3), min(H, int(P[:, 1].max()) + 4)),
             slice(max(0, int(P[:, 0].min()) - 3), min(W, int(P[:, 0].max()) + 4)))
        if b[0].stop <= b[0].start or b[1].stop <= b[1].start:
            return
        m = polys([P], 0.5, b)
        yy, xx = np.mgrid[b].astype(np.float32)
        t = np.clip(((xx - x) * np.cos(a) + (yy - y) * np.sin(a)) / L, 0, 1)
        catch = noise.smoothstep(-0.1, 0.1, self.sheet.tooth[b] + 0.12 * noise.field(m.shape, 1.5, r) - 0.15 - dry * t)
        self.lay(m * catch * dens * self.keep[b], b, **mix)

    def bleed(self, vis, at, dens, reach=28, **mix):
        """Where a flower (`vis`, over the box `at`) met the wet ground its colour ran out into it and
        faded: here and there along its edge a soft tongue of it, `reach` px at most."""
        r = self.r
        d = ndimage.distance_transform_edt(vis < 0.5) + 3 * noise.field(vis.shape, 6, r) \
            + 5 * (self.sheet.fiber[at] - 0.4)
        creep = np.clip(reach * (noise.field(vis.shape, 70, r) - 0.2), 0, reach)
        band = np.clip(1 - d / (creep + 1e-3), 0, 1) ** 1.5 * (1 - vis) * (creep > 2)
        self.lay(band * dens * self.keep[at], at, **mix)

    def glaze(self):
        return self.sheet.color * np.exp(-self.A)


def frame(r):
    """Where the colour stopped: mostly it ran off the edge of the sheet, but here and there the
    brush stopped a little short and left the paper bare."""
    d = np.minimum.reduce([XX, W - 1 - XX, YY, H - 1 - YY])
    inset = 40 + 45 * noise.smoothstep(1.0, 2.2, noise.field((H, W), 380, r)) + 6 * noise.field((H, W), 30, r)
    return noise.smoothstep(-1.5, 1.5, d - inset).astype(np.float32)


def facing(h, r, n=4):
    """The petals of a poppy facing out, back to front: two broad ones across, two over them, and now
    and then a fifth, smaller and crumpled. -> outlines"""
    a0 = r.uniform(0, 2 * np.pi)
    spec = [(0, 2.8, 1.0), (np.pi, 2.7, 0.95), (np.pi / 2, 2.5, 0.86), (-np.pi / 2, 2.4, 0.84)]
    if n > 4:
        spec.append((r.uniform(0, 2 * np.pi), 1.8, 0.62))
    return [h.fan(r, a0 + a + r.normal(0, 0.2), s * r.uniform(0.9, 1.1), q * r.uniform(0.9, 1.08),
                  crinkle=r.uniform(0.05, 0.09)) for a, s, q in spec]


def turned(h, r):
    """A poppy turned from us and tipped back: the far petals seen from inside the cup, the near one
    from outside, its rim across the foot of the heart. -> outlines, back to front"""
    up = -np.pi / 2 + r.normal(0, 0.15)
    back = [h.fan(r, up + a + r.normal(0, 0.1), s, q, root=h.heart, crinkle=r.uniform(0.05, 0.08))
            for a, s, q in ((0, 2.8, 1.0), (-1.45, 2.4, 0.9), (1.45, 2.4, 0.9))]
    return back + [h.lip(r, -0.13, 0.95, 0.95, 0.14)]


def side(h, r):
    """A poppy seen from the side, a cup: three petals rising from its foot and the near one across
    the front. -> outlines, back to front"""
    back = [h.fan(r, -np.pi / 2 + a + r.normal(0, 0.08), s, q, root=h.heart, crinkle=r.uniform(0.05, 0.08))
            for a, s, q in ((0, 1.9, 1.08), (-0.62, 1.5, 0.95), (0.66, 1.5, 0.92))]
    return back + [h.lip(r, -0.06, 0.64, 0.52, 0.05)]


def poppy(box, r, h, vis, petals, colours, cup=False, fan=None, margin=None):
    """A poppy a petal at a time from the back, each petal its own pool laid when the one under it had
    dried, though now and then the next went in while it was wet and the two ran into one; deeper
    colour run into the root of each, into the places where the brush was wettest, and along the edge
    the next would cover, and in a cup, the dark inside it. A backrun in a petal or two; here and there
    the red ran out into the wet ground. -> where the petals were still wet"""
    base, deep, light = colours
    R = h.R
    q, _ = h.polar()
    shade = folds(petals[::-1], h.box, 0.07 * R)
    root = noise.smoothstep(0.55, 0.08, q)
    dark = dict(deep, black=0.4) if cup else deep
    groups = [[0]]
    for k in range(1, len(petals)):
        if r.uniform() < 0.35:
            groups[-1].append(k)
        else:
            groups.append([k])
    wets = np.zeros(q.shape, np.float32)
    for g in groups:
        strokes = [(petals[k], r.uniform(0.85, 1.05), vary(light if r.uniform() < 0.3 else base, r, 0.3)) for k in g]
        m, charges = brushwork(strokes, h.box, boost=0.2)
        pools = noise.smoothstep(0.3, 1.5, noise.field(q.shape, 0.22 * R, r))
        glow = noise.smoothstep(0.4, 1.6, noise.field(q.shape, 0.3 * R, r))
        charges += [(m * root, 0.5, deep), (m * shade, 0.85, dark), (m * pools, 0.5, deep), (m * glow, 0.35, light)]
        bl, mg = [], []
        if fan in g:
            x, y = petals[fan][len(petals[fan]) // 2]
            hx, hy = h.at(0, 0)[0]
            bl.append((x, y, 0.45 * R, 0.4, 1.0, float(np.arctan2(hy - y, hx - x)), 0.9))
        if margin in g:
            x, y = petals[margin][int(0.62 * len(petals[margin]))]
            mg.append((x, y, 0.4 * R, 0.035 * R, 0.45, 1.1))
        wets = np.maximum(wets, box.wash(m * vis, charges, at=h.box, spread=9, ragged=2.0, finger=10, blooms=bl,
                                         margins=mg, pool=0.32, rim=0.55, rim_width=3, tides=0.2, grain=0.2,
                                         scale=90))
    box.bleed(vis, h.box, 1.0, reach=36, **base)
    return np.maximum(wets, vis)


def heart(box, r, h, wet, centre=dict(black=0.6, prussian=0.4)):
    """The black of a poppy's heart dropped into the wet petals, creeping out into the red, with a
    darker drop or two beside it; when it was dry, the knob of the seed head in green-grey."""
    R = h.R
    hx, hy = h.at(0, 0)[0]
    box.drop(hx, hy, 0.12 * R, 1.8, wet, at=h.box, line=0.5, soft=0.5, **centre)
    for a in r.uniform(0, 2 * np.pi, 2):
        x, y = h.at(0.14 * R, a)[0]
        box.drop(x, y, 0.05 * R, 1.2, wet, at=h.box, line=0.4, **centre)
    knob = polys([touch(hx - 0.035 * R, hy, 0.07 * R, 0.065 * R, r.normal(0, 0.3), r, start=0.5, belly=0.5, tip=0.5)],
                 0.6, h.box)
    box.wash(knob, [(knob, 1.1, dict(sap=0.5, prussian=0.2, black=0.3))], at=h.box, spread=3, ragged=0.8, rim=0.8,
             rim_width=1.5, grain=0.2, scale=20, feather=0)


def stamens(box, r, h, n, arc=(-np.pi, np.pi), reach=(0.08, 0.18), start=(0.08, 0.13), origin=None):
    """The stamens, dragged out from the black of the heart with a nearly dry point when it had dried:
    dark flicks, each its own length, broken as the brush ran dry, some ending in a dot."""
    R = h.R
    a = r.uniform(*arc, n)
    if origin is None:
        p, c = h.at(r.uniform(*start, n) * R, a), h.at(0, 0)[0]
    else:   # from along the rim of a cup seen from the side
        p = np.stack([h.point(origin[0] + u, origin[1]) for u in r.uniform(-0.28, 0.28, n)])
        c = h.point(origin[0], origin[1] + 0.6)
    for (x, y), da in zip(p, r.normal(0, 0.2, n)):
        ang = np.arctan2(y - c[1], x - c[0]) + da
        L, wd = R * r.uniform(*reach), max(4.0, R * r.uniform(0.012, 0.02))
        box.flick(x, y, L, wd, ang, 1.7, dry=r.uniform(0.3, 0.6), black=0.7, prussian=0.3)
        if r.uniform() < 0.35:
            ex, ey = x + 0.9 * L * np.cos(ang), y + 0.9 * L * np.sin(ang)
            box.flick(ex, ey, 1.6 * wd, 1.4 * wd, ang + r.normal(0, 0.5), 1.8, dry=0.15,
                      shape=dict(start=0.3, belly=0.5, blunt=0.6, tip=0.5), black=0.8, violet=0.2)


def rings(h, r, spec, shape=dict(start=0.45, belly=0.62, blunt=0.8, tip=0.45)):
    """A dahlia's petals as the point laid them, ring by ring from the outside in, each ring in two
    goes, every other petal and then those between, so that neighbours lie over one another.
    -> passes [(outlines, ring)], outermost first"""
    passes = []
    for k, (n, reach, width, start) in enumerate(spec):
        P = [p for p, a in h.petals(r, n, reach=reach, width=width, start=start, turn=0.08, shape=shape)]
        passes += [(P[0::2], k), (P[1::2], k)]
    return passes


def dahlia(box, r, h, vis, passes):
    """The orange dahlia ring by ring from the outside in, each petal a pool of cadmium yellow or
    orange going deeper towards its root, dried with a dark rim, the inner rings more and more scarlet;
    olive run into the side away from the light; a red heart."""
    R = h.R
    q, th = h.polar()
    away = near(th, 1.0, 1.1) * noise.smoothstep(0.3, 0.9, q)
    mixes = (dict(yellow=0.7, orange=0.3), dict(yellow=0.4, orange=0.6), dict(orange=0.5, scarlet=0.5))
    wet = np.zeros(q.shape, np.float32)
    for j, (P, k) in enumerate(passes):
        strokes = [(o, r.uniform(0.85, 1.2), vary(mixes[k], r, 0.4, ("scarlet",))) for o in P]
        if j == len(passes) - 1:   # the heart, a round dab of the brush
            strokes.append((touch(h.cx - 0.11 * R, h.cy, 0.22 * R, 0.2 * R, r.normal(0, 0.3), r, start=0.6, belly=0.5,
                                  blunt=0.6, tip=0.5), 1.1, dict(orange=0.5, scarlet=0.5)))
        m, charges = brushwork(strokes, h.box, boost=0.2)
        tip = (1.0, 0.72, 0.45)[k]
        charges += [(m * noise.smoothstep(0.85 * tip, 0.35 * tip, q), 0.55, dict(orange=0.4, scarlet=0.5, crimson=0.1)),
                    (m * away, 0.45, dict(sap=0.5, sienna=0.5))]
        wet = np.maximum(wet, box.wash(m * vis, charges, at=h.box, spread=5, ragged=1.4, finger=4, pool=0.2, rim=0.75,
                                       rim_width=2.5, tides=0.15, grain=0.2, scale=60))
    box.drop(h.cx, h.cy, 0.14 * R, 1.3, wet, at=h.box, line=0.5, scarlet=0.5, crimson=0.5)
    box.drop(h.cx, h.cy, 0.06 * R, 2.0, wet, at=h.box, line=0.3, crimson=0.5, black=0.5)
    box.bleed(vis, h.box, 0.6, yellow=0.6, orange=0.4)


def white_dahlia(box, r, h, vis, passes, solid):
    """The white dahlia is the paper, kept clear of everything round it. Grey went on only in strokes
    along a petal's edge where it lies over the one behind, on the side away from the light, and in the
    chinks between the petals round the heart; an orange heart in a few dabs."""
    R = h.R
    order = [P for g, _ in passes for P in g]   # back to front
    ms = [polys([P], 0.6, h.box) for P in order]
    after, acc = [None] * len(ms), np.zeros_like(ms[0])
    for i in range(len(ms) - 1, -1, -1):
        after[i], acc = acc, np.maximum(acc, ms[i])
    yy, xx = np.mgrid[h.box].astype(np.float32)
    behind = ms[0].copy()
    strokes = []
    greys = (dict(ultramarine=0.35, sienna=0.35, sap=0.3), dict(ultramarine=0.5, rose=0.15, sienna=0.35),
             dict(sienna=0.45, yellow=0.25, sap=0.3))
    for i in range(1, len(ms)):
        m, P = ms[i], order[i]
        if r.uniform() < 0.1:   # not every petal got its stroke
            behind = np.maximum(behind, m)
            continue
        cx, cy = P.mean(0)
        lit = noise.smoothstep(-0.5, 0.6, np.cos(np.arctan2(yy - cy, xx - cx) - 0.9 - r.normal(0, 0.35)))
        d = ndimage.distance_transform_edt(m < 0.5)
        reach = R * r.uniform(0.035, 0.08) * lit * (1 + 0.3 * np.clip(noise.field(d.shape, 0.1 * R, r), -2, 2))
        band = noise.smoothstep(0.5, -0.5, d - reach) * behind * (1 - after[i]) * (1 - m)
        if band.sum() > 0.002 * R * R:
            # darkest against the edge of the petal that throws it
            strokes.append((band * (0.45 + 0.55 * np.exp(-d / (0.35 * reach + 1))), r.uniform(0.35, 0.6),
                            greys[r.integers(3)]))
        behind = np.maximum(behind, m)
    q, _ = h.polar()
    chinks = np.clip(solid[h.box] - acc, 0, 1) * noise.smoothstep(0.8, 0.3, q)
    bands = np.clip(sum(s for s, _, _ in strokes) * 3 + chinks, 0, 1)
    box.wash(bands * vis, [(s, d, mix) for s, d, mix in strokes] + [(chinks, 0.5, greys[0])], at=h.box, spread=3,
             ragged=0.9, rim=0.7, rim_width=2, tides=0.1, grain=0.3, scale=40, feather=0.1)
    dabs = []
    for _ in range(5):
        a, L = r.uniform(0, 2 * np.pi), R * r.uniform(0.12, 0.2)
        x, y = h.at(R * r.uniform(0, 0.05), r.uniform(0, 2 * np.pi))[0]
        dabs.append(touch(x - 0.5 * L * np.cos(a), y - 0.5 * L * np.sin(a), L, R * r.uniform(0.1, 0.15), a, r,
                          start=0.4, tip=0.6))
    m, charges = brushwork([(P, r.uniform(0.9, 1.2), vary(dict(yellow=0.5, orange=0.5), r, 0.4)) for P in dabs], h.box)
    wet = box.wash(m, charges, at=h.box, spread=5, ragged=1.0, rim=0.6, rim_width=2, grain=0.2, scale=40)
    box.drop(h.cx, h.cy, 0.06 * R, 1.6, wet, at=h.box, line=0.5, scarlet=0.7, crimson=0.3)


def floret(x, y, s, fore, r):
    """One delphinium floret: five rounded sepals out from its eye, foreshortened across as it turns
    away. -> outlines"""
    a0 = r.uniform(0, 2 * np.pi)
    return [touch(0, 0, s * r.uniform(0.85, 1.1), s * r.uniform(0.75, 1.0), a0 + 2 * np.pi * j / 5 + r.normal(0, 0.18),
                  r, start=0.4, belly=0.6, blunt=0.7, tip=0.5) * (fore, 1) + (x, y) for j in range(5)]


def spike(r, x0=2660, y0=2230, x1=2605, y1=330):
    """A spike of delphinium: florets all round its stem, open and broad low down and smaller up it,
    then buds at the tip, alternate on short stalks. -> stem and stalks, florets [(outlines, x, y, size,
    facing)], buds"""
    def at(t):
        return x0 + (x1 - x0) * t + 45 * np.sin(np.pi * t), y0 + (y1 - y0) * t
    florets, stalks, buds = [], [], []
    t, k = 0.0, 0
    while t < 0.78:
        s = (68 * (1 - t) ** 0.7 + 20) * r.uniform(0.8, 1.15)
        th = 2.4 * k + r.normal(0, 0.35)
        sx, sy = at(t)
        x, y = sx + 1.15 * s * np.sin(th), sy + r.normal(0, 0.2 * s)
        florets.append((floret(x, y, s, 0.55 + 0.45 * abs(np.cos(th)), r), x, y, s, np.cos(th)))
        stalks.append(touch(sx, sy, np.hypot(x - sx, y - sy) + 1, 5, np.arctan2(y - sy, x - sx), r, start=0.9,
                            belly=0.1, tip=0.4))
        t += s / (y0 - y1) * r.uniform(0.5, 0.85)
        k += 1
    while t < 0.985:
        s = (9 + 15 * (1 - t) / 0.22) * r.uniform(0.6, 1.3)
        sx, sy = at(t)
        a = -np.pi / 2 + (1 if k % 2 else -1) * r.uniform(0.25, 0.9)
        buds.append(touch(sx + 0.5 * s * np.cos(a), sy + 0.5 * s * np.sin(a), s * r.uniform(1.4, 2.0), s, a, r,
                          start=0.4, belly=r.uniform(0.35, 0.55), blunt=0.7, tip=0.85))
        stalks.append(touch(sx, sy, 0.7 * s, 4, a, r, start=0.9, belly=0.1, tip=0.4))
        t += s / (y0 - y1) * r.uniform(0.6, 1.4)
        k += int(r.integers(1, 3))
    sx, sy = at(1.0)
    buds.append(touch(sx, sy + 6, 30, 10, -np.pi / 2 + r.normal(0, 0.1), r, start=0.5, belly=0.35, tip=0.9))
    stem = cane([(*at(t), 15 - 9 * max(t, 0)) for t in np.linspace(-0.12, 1.0, 9)], r)
    return stem + stalks, florets, buds


def delphinium(box, r, stem, florets, buds):
    """The delphinium floret by floret, those behind the stem first and deeper, then those in front in
    two goes, so that where two touch one lies over the other, though now and then two ran together:
    each its own blue, from a pale wash of cobalt to ultramarine dark with Prussian blue. Black dropped
    into the eyes of a few, the paper kept in others; the buds; then the stem and its stalks between."""
    at = span([P for f in florets for P in f[0]] + buds + stem)
    pick = lambda p: BLUES[r.choice(len(BLUES), p=p)]
    passes = [[], [], []]
    for f in florets:
        passes[0 if f[4] < -0.25 else 1 + int(r.uniform() < 0.5)].append(f)
    eyes = []
    for k, group in enumerate(passes):
        strokes, white = [], []
        for outlines, x, y, s, face in group:
            d, mix = pick([0.05, 0.15, 0.3, 0.35, 0.15] if k == 0 else [0.25, 0.25, 0.25, 0.1, 0.15])
            d, mix = d * r.uniform(0.9, 1.1), vary(mix, r, 0.25)
            strokes += [(P, d, mix) for P in outlines]
            u = r.uniform()
            if k and u < 0.15:
                a = r.uniform(0, 2 * np.pi)
                white.append(touch(x - 0.12 * s * np.cos(a), y - 0.12 * s * np.sin(a), 0.24 * s, 0.2 * s, a, r, tip=0.6))
            elif k and u < 0.35:
                eyes.append((x, y, s))
        if not strokes:
            continue
        m, charges = brushwork(strokes, at, boost=0.3)
        if white:
            m = m * (1 - polys(white, 0.8, at))
        wet = box.wash(m, charges, at=at, spread=5, ragged=1.3, finger=5, pool=0.3, rim=0.85, rim_width=2.5,
                       tides=0.15, grain=0.45, scale=50)
        for x, y, s in eyes:
            box.drop(x, y, 0.13 * s, 1.8, wet, at=at, line=0.4, black=0.7, ultramarine=0.3)
        eyes = []
    m, charges = brushwork([(P, r.uniform(0.9, 1.3), vary(dict(ultramarine=0.4, violet=0.3, prussian=0.2, sap=0.1), r,
                                                         0.5, ("sap", "cobalt"))) for P in buds], at)
    box.wash(m, charges, at=at, spread=4, ragged=1.0, rim=0.8, rim_width=2, grain=0.4, scale=30)
    hide = 1 - polys([P for f in florets for P in f[0]] + buds, 0.8, at)
    m, charges = brushwork([(P, r.uniform(0.9, 1.15), vary(dict(sap=0.45, prussian=0.4, black=0.15), r, 0.3))
                            for P in stem], at)
    box.wash(m * hide, charges, at=at, spread=4, ragged=1.0, rim=0.7, rim_width=2, grain=0.2, scale=40)


def leaf(x, y, L, a, kind, r):
    """One leaf as the brush laid it, out from its stalk at (x, y) along `a`: a long blade, a broad
    leaf, or a poppy leaf cut into lobes that point forward. -> outlines"""
    b = r.uniform(-0.18, 0.18)
    if kind == "blade":
        return [touch(x, y, L, L * r.uniform(0.14, 0.22), a, r, bend=b, start=0.15, belly=0.3, blunt=0.7, tip=0.95)]
    if kind == "broad":
        return [touch(x, y, L, L * r.uniform(0.45, 0.6), a, r, bend=b, start=0.1, belly=0.45, blunt=0.6, tip=0.8)]
    out = [touch(x, y, L, L * 0.26, a, r, bend=b, start=0.3, belly=0.4, blunt=0.7, tip=0.9)]
    c, s = np.cos(a), np.sin(a)
    for j, t in enumerate(np.linspace(0.12, 0.6, int(r.integers(3, 6)))):
        off = 4 * b * L * t * (1 - t)
        px, py = x + c * t * L - s * off, y + s * t * L + c * off
        lb = L * (0.36 - 0.22 * t) * r.uniform(0.75, 1.2)
        out.append(touch(px, py, lb, lb * r.uniform(0.45, 0.6), a + (1 if j % 2 else -1) * r.uniform(0.45, 0.8), r,
                         start=0.5, belly=0.4, blunt=0.7, tip=0.9))
    return out


MID = ((40, 2440, 520, -1.0, "broad"), (420, 2460, 480, -1.6, "blade"), (760, 2450, 420, -2.2, "cut"),
       (1560, 2450, 420, -1.2, "broad"), (1700, 2200, 380, -0.3, "cut"), (1950, 2450, 470, -1.9, "blade"),
       (2150, 1950, 420, -0.3, "broad"), (2250, 1650, 330, -0.2, "blade"), (2400, 2350, 450, -2.4, "cut"),
       (2780, 2440, 500, -1.0, "broad"), (3050, 2200, 430, -0.5, "cut"), (3260, 2420, 380, -2.0, "blade"),
       (2880, 1950, 360, -0.9, "broad"), (3200, 1800, 300, -2.6, "blade"))

DEEP = ((120, 2380, 420, -1.2, "blade"), (300, 2330, 330, -2.2, "cut"), (560, 2300, 300, -1.0, "broad"),
        (760, 2380, 380, -1.9, "blade"), (850, 2150, 260, -0.3, "cut"), (1050, 2400, 330, -2.0, "cut"),
        (1330, 2380, 300, -1.0, "broad"), (1250, 2250, 240, -0.4, "blade"), (1500, 2100, 340, -2.5, "broad"),
        (1700, 2150, 380, -1.4, "cut"), (1950, 2050, 300, -0.5, "broad"), (2150, 2350, 420, -1.8, "blade"),
        (1650, 2400, 350, -0.8, "blade"), (2450, 2250, 330, -2.3, "cut"), (2780, 2300, 360, -0.8, "broad"),
        (2600, 2400, 380, -1.6, "blade"), (2950, 2150, 380, -1.3, "cut"), (3150, 2000, 320, -2.4, "broad"),
        (3050, 2380, 400, -0.6, "blade"), (40, 2000, 300, -0.2, "broad"))


def foliage(r):
    """The leaves round the feet of the flowers in the greens of the garden, each its own kind.
    -> strokes"""
    greens = [dict(sap=0.5, prussian=0.5), dict(emerald=0.6, prussian=0.4), dict(sap=0.6, yellow=0.25, emerald=0.15),
              dict(prussian=0.5, emerald=0.5)]
    out = []
    for x, y, L, a, kind in MID:
        mix, d = greens[r.integers(len(greens))], r.uniform(0.8, 1.15)
        out += [(P, d, vary(mix, r, 0.35, ("yellow", "violet"))) for P in leaf(x, y, L * r.uniform(0.8, 1.1),
                                                                               a + r.normal(0, 0.15), kind, r)]
    return out


def darks(r):
    """The darkest leaves of the garden, blue-black, round the feet of the flowers: some touched into
    the ground while it was wet, the rest laid when it was dry. -> (soft outlines, [crisp leaves])"""
    soft, crisp = [], []
    for x, y, L, a, kind in DEEP:
        P = leaf(x + r.normal(0, 25), y + r.normal(0, 20), L * r.uniform(0.85, 1.15), a + r.normal(0, 0.15), kind, r)
        (soft if r.uniform() < 0.3 else crisp).append(P)
    return [P for l in soft for P in l], crisp


def leaves(box, r, strokes, cut):
    """The leaves, one passage, so where they touch they ran together; blue-black dropped into the
    shadows at their roots."""
    at = span([P for P, _, _ in strokes])
    m, charges = brushwork(strokes, at)
    charges.append((noise.smoothstep(0.3, 1.3, noise.field(m.shape, 140, r)), 0.45,
                    dict(prussian=0.6, black=0.2, violet=0.2)))
    box.wash(m * cut[at], charges, at=at, spread=10, ragged=1.6, finger=12, pool=0.2, rim=0.6, rim_width=3,
             tides=0.2, grain=0.3, scale=90)


def garden(box, r, crisp, cut):
    """The blue-black leaves laid on the dry ground in two goes, so that where one lies over another
    it shows; a greener blue run into them here and there while they were wet."""
    for group in (crisp[0::2], crisp[1::2]):
        strokes = [(P, d, mix) for l in group for d, mix in [(r.uniform(1.5, 2.0), vary(DARK[r.integers(3)], r, 0.3))]
                   for P in l]
        at = span([P for P, _, _ in strokes])
        m, charges = brushwork(strokes, at)
        charges.append((m * noise.smoothstep(0.2, 1.4, noise.field(m.shape, 90, r)), 0.6,
                        dict(emerald=0.5, prussian=0.3, sap=0.2)))
        box.wash(m * cut[at], charges, at=at, spread=8, ragged=1.4, finger=6, pool=0.35, rim=0.7, rim_width=3,
                 tides=0.15, grain=0.35, scale=70)


def plan(x, y):
    """Which colour the ground wants at (x, y): weights for each family."""
    u, v = x / W, y / H
    ss = noise.smoothstep
    clouds = sum(np.exp(-(((u - a) / b) ** 2 + ((v - c) / d) ** 2)) for a, b, c, d in
                 ((0.73, 0.11, 0.16, 0.045), (0.9, 0.12, 0.27, 0.045), (0.67, 0.08, 0.32, 0.035), (0.84, 0.11, 0.4, 0.035)))
    return dict(
        blue=1.4 * ss(0.55, 0.1, v) * ss(0.78, 0.4, u) + 0.7 * ss(0.12, 0.0, v),
        deep=0.9 * ss(0.22, 0.0, v) * ss(0.3, 0.0, u),
        violet=0.9 * ss(0.3, 0.5, v) * ss(0.72, 0.55, v) * ss(0.62, 0.3, u)
        + 0.7 * ss(0.04, 0.15, v) * ss(0.45, 0.3, v) * ss(0.5, 0.7, u),
        red=1.9 * clouds,
        orange=1.5 * ss(0.52, 0.7, u) * ss(0.33, 0.45, v) * ss(0.7, 0.6, v),
        yellow=1.9 * np.exp(-(((u - 0.86) / 0.09) ** 2 + ((v - 0.52) / 0.065) ** 2)),
        green=1.1 * ss(0.62, 0.8, v),
        moss=0.75 * ss(0.62, 0.72, v) * ss(0.97, 0.8, v),
        night=0.8 * ss(0.86, 1.0, v))


def ground(box, r, cut, fade, soft):
    """Everything behind the flowers in one passage on the damp sheet, laid round them in rows of broad
    strokes, each with the colour the brush was carrying there: Prussian blue across the top, violet,
    red clouds, the orange of the after-glow with yellow at its heart, and low down the dark of the
    garden, with blue-black leaves (`soft`) touched into it; they ran together where they met, and
    where the paper was still damp they ran softly in under the flowers (`fade`). As it dried a drop of
    water fell into the blue, water crept in from the right-hand edge of the sheet, and along the
    left-hand edge it crept back into the pool."""
    strokes = []
    y = 20.0
    while y < H + 100:
        x = r.uniform(-400, -100)
        tilt = r.normal(0, 0.05)
        while x < W + 100:
            L, wd = r.uniform(300, 800), r.uniform(150, 240)
            cx, cy = x + L / 2, y + r.normal(0, 25)
            w = plan(cx + r.normal(0, 140), cy + r.normal(0, 90))
            keys = list(w)
            p = (np.array([w[k] for k in keys]) + 0.01) ** 3
            fam = keys[r.choice(len(keys), p=p / p.sum())]
            d, mix = FAMILIES[fam]
            strokes.append((touch(x, cy, L, wd, tilt + r.normal(0, 0.06), r, bend=r.normal(0, 0.04), start=0.7,
                                  belly=0.2, blunt=0.5, tip=0.5, edge=0.12), d * r.uniform(0.8, 1.15), vary(mix, r, 0.25)))
            x += L * r.uniform(0.55, 0.85)
        y += r.uniform(85, 120)
    m, charges = brushwork(strokes, boost=0.15)
    # the big brush drew its colour out in streaks along each stroke
    streak = (1 + 0.14 * noise.stretched((W, H), 7, 320, r).T + 0.06 * noise.stretched((W, H), 2.5, 120, r).T)
    charges = [(c * streak * fade, d, mix) for c, d, mix in charges]
    charges.append((polys(soft, 2.0) * fade, 1.2, dict(prussian=0.5, black=0.3, emerald=0.2)))
    bl = [(1240, 215, 125, 0.3, 1.1, None), (W, 1150, 330, 0.25, 1.3, np.pi, 1.0)]
    box.wash(box.keep * cut, charges, spread=18, ragged=2.5, finger=25, blooms=bl, margins=[(45, 1180, 300, 24)],
             pool=0.25, rim=0.6, rim_width=4, tides=0.2, grain=0.12, scale=220)


def stems(box, r, heads, hide):
    """The stems, drawn down from under each head with a loaded brush in a dark green and lost behind
    the flowers in front; a poppy bud hanging from its hooked stem."""
    out = []
    for h in heads.values():
        x0, y0 = h.point(0, 0.45)
        x1 = x0 + r.normal(0, 90)
        w = 0.025 * h.R + 9
        P = [(x0, y0, w), ((x0 + x1) / 2 + r.normal(0, 40), (y0 + H) / 2, w * 1.05), (x1, H + 60, w * 1.1)]
        out += [(o, r.uniform(1.0, 1.3), vary(dict(prussian=0.45, sap=0.35, black=0.2), r, 0.3, ("emerald",)))
                for o in cane(P, r)]
    out += [(o, r.uniform(1.0, 1.3), vary(dict(prussian=0.45, sap=0.35, black=0.2), r, 0.3)) for o in
            cane([(2150, 1500, 15), (2200, 1150, 14), (2260, 900, 12), (2300, 790, 11), (2335, 755, 10), (2365, 770, 9)], r)]
    out.append((touch(2340, 790, 95, 62, 1.25, r, start=0.5, belly=0.5, blunt=0.7, tip=0.6), 1.0,
                dict(sap=0.6, prussian=0.3, black=0.1)))
    out.append((touch(2370, 850, 30, 22, 1.3, r, start=0.4, belly=0.5, tip=0.6), 1.3, dict(scarlet=0.8, crimson=0.2)))
    m, charges = brushwork(out)
    box.wash(m * hide, charges, spread=5, ragged=1.0, rim=0.7, rim_width=2, grain=0.2, scale=60)


def occlude(masks, boxes, r):
    """Each head as it was painted: round the ones in front of it, with now a hair of paper left
    between and now the two run together. -> visible masks over each head's box"""
    front_ = np.zeros((H, W), np.float32)
    out = []
    for m, b in zip(masks, boxes):
        f = front_[b]
        if f.max() < 0.5:
            out.append(m[b])
        else:
            sd = ndimage.distance_transform_edt(f < 0.5) - ndimage.distance_transform_edt(f >= 0.5)
            gap = -1.5 + 3.0 * noise.field(f.shape, 50, r)
            out.append(m[b] * noise.smoothstep(-1, 1, sd - gap))
        front_ = np.maximum(front_, m)
    return out


def paint(seed=1935):
    r = noise.rng(seed)
    sheet = paper.washi((H, W), _s(r), tint="#f2e9d5", margin=(56, 60), fibre_density=0.8)
    box = Box(sheet, r, frame(r))
    heads = dict(   # front to back
        scarlet=Head(1000, 1060, 600, 0.86, -0.3), orange=Head(1760, 1450, 430, 0.85, 0.25),
        low=Head(1190, 2010, 320, 1.0, 0.22, heart=(0.0, 0.5)), white=Head(490, 1710, 390, 0.85, -0.2),
        small=Head(2960, 1620, 190, 0.62, -0.45, heart=(0.0, -0.1)),
        mauve=Head(1630, 570, 280, 0.62, 0.4, heart=(0.0, -0.1)), crimson=Head(440, 650, 380, 0.78, 0.2))
    petals = dict(scarlet=facing(heads["scarlet"], r, 5), low=side(heads["low"], r), small=turned(heads["small"], r),
                  mauve=turned(heads["mauve"], r), crimson=facing(heads["crimson"], r))
    orange = rings(heads["orange"], r, ((15, (0.84, 1.0), (0.3, 0.38), (0.3, 0.45)),
                                        (12, (0.62, 0.76), (0.27, 0.34), (0.14, 0.26)),
                                        (8, (0.38, 0.48), (0.26, 0.34), (0.03, 0.1))))
    white = rings(heads["white"], r, ((13, (0.84, 1.0), (0.34, 0.42), (0.25, 0.4)),
                                      (10, (0.6, 0.72), (0.3, 0.36), (0.1, 0.2)),
                                      (7, (0.36, 0.46), (0.26, 0.32), (0.02, 0.08))))
    outlines = dict(petals, orange=[P for g, _ in orange for P in g], white=[P for g, _ in white for P in g])
    close = dict(orange=0.02, white=0.0)
    outline = {k: heads[k].mask(outlines[k], int(close.get(k, 0.04) * heads[k].R)) for k in heads}
    vis = dict(zip(heads, occlude([outline[k] for k in heads], [h.box for h in heads.values()], r)))
    stalks, florets, buds = spike(r)
    m_spike = polys([P for f in florets for P in f[0]] + buds, 0.7)
    flowers = np.clip(sum(outline.values()), 0, 1)
    sd = ndimage.distance_transform_edt(flowers < 0.5) - ndimage.distance_transform_edt(flowers >= 0.5)
    # along some stretches the paper was still damp where the ground met a flower, and the two ran
    # softly together; elsewhere the ground stopped at the flower's edge, or a hair over or short of it
    damp = noise.smoothstep(-0.1, 0.7, noise.field((H, W), 110, r))
    gap = -4 + 4.5 * noise.field((H, W), 60, r) - 14 * damp
    cut = noise.smoothstep(-1, 1, sd - gap)
    fade = 1 - damp * (1 - noise.smoothstep(gap, gap + 22, sd))
    # the delphinium's place was kept clear when the ground and the leaves went on, to be painted into later
    out = m_spike < 0.5
    sp = ndimage.distance_transform_edt(out) - ndimage.distance_transform_edt(~out)
    clear = noise.smoothstep(-1, 1, sp - (0.5 + 1.5 * noise.field((H, W), 40, r)))
    cut *= clear
    wets = {}
    for k, kind in (("scarlet", "facing"), ("low", "side"), ("small", "turned"), ("mauve", "turned"),
                    ("crimson", "facing")):
        wets[k] = poppy(box, r, heads[k], vis[k], petals[k], POPPIES[k], cup=kind != "facing",
                        fan=0 if k == "crimson" else None, margin=2 if k == "scarlet" else None)
        if kind != "side":
            heart(box, r, heads[k], wets[k])
    dahlia(box, r, heads["orange"], vis["orange"], orange)
    white_dahlia(box, r, heads["white"], vis["white"], white, outline["white"])
    soft, crisp = darks(r)
    ground(box, r, cut, fade, soft)
    leafcut = noise.smoothstep(-1, 1, sd - (2.0 + 3 * noise.field((H, W), 60, r))) * clear
    leaves(box, r, foliage(r), leafcut)
    delphinium(box, r, stalks, florets, buds)
    garden(box, r, crisp, leafcut)
    stems(box, r, heads, 1 - flowers)
    stamens(box, r, heads["scarlet"], 16)
    stamens(box, r, heads["crimson"], 11)
    stamens(box, r, heads["mauve"], 7, reach=(0.1, 0.2))
    stamens(box, r, heads["small"], 5, reach=(0.12, 0.22))
    stamens(box, r, heads["low"], 6, reach=(0.1, 0.2), origin=(0.0, -0.02))
    img = wash.cockle(box.glaze(), sheet, box.soak, r, buckle=1.2, grain=0.12)
    return plate.mount(img, sheet)
