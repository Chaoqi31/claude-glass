"""The Window at Ocean Park. Oil and charcoal on canvas.

In 1966 Richard Diebenkorn moved from the Bay Area to Santa Monica and took a studio in Ocean Park, a few
streets from the beach, and for the next twenty years painted the Ocean Park pictures there: big upright
canvases in which a scaffold of lines and narrow bands runs across the top and down one side, a diagonal
cuts a corner, and the rest opens onto a large field of colour as light as the air over the sea. He drew
them in charcoal on the canvas and painted them in thin oil, and then changed his mind, again and again:
a band moved down a little, a line drawn and drawn again, a plane painted out in another colour. He
scraped the paint back and painted over it, and he did not hide what he had changed, so that every surface
carries what lies beneath it.

This canvas was painted in three sittings. The first drew the scaffold in charcoal and laid it in with
washes of turpentine and colour: orange across the top, sea green below, violet down the side, and they
ran. The second drew it again with more bands and the side further right, and painted it in: yellow,
lilac and sea green across the top, cerulean in the field and deeper below a line drawn across it, cream
and a blue strip down the side, and the diagonal ruled in violet; the knife scraped some of it back while
it was wet, and some of it ran. The third drew it a last time. The field went on in long upright strokes
of pale cerulean, fewer in some places than others, with scumbles of white dragged over its upper part,
so that the earlier blues and greens and the old lines breathe through; the knife was pulled down through
it low on the canvas. The bands were narrowed and broken, some laid thin over the old ones, with orange
low among them, and their left end painted out in white. Down the side the blue was painted over in sea
green and the cream divided low down by an orange line; a deep blue block and a lilac one went in where
the diagonal meets the side, and a pale triangle over the corner. Last came the ruled lines, some in paint
against a straightedge and some in charcoal over the dry paint. Where an edge was laid against tape it is
straight and the paint has crept a little under it along the threads; where it was cut in by hand it
wanders, and the colour beneath shows along it.
"""

from types import SimpleNamespace

import numpy as np
from scipy import ndimage, special

from atelier import canvas, dabs, noise, pencil
from atelier.color import lin, pigment

TITLE = "The Window at Ocean Park"
DATE = "2026"
MEDIUM = "Oil and charcoal on canvas, laid in thin and painted over in three sittings, scraped back and redrawn"
AFTER = ("Richard Diebenkorn, the Ocean Park paintings, Santa Monica, 1967–88 (among them Ocean Park No. 54, "
         "1972; No. 79, 1975; No. 129, 1984)")
ROOM = "Colour Itself"
YEAR = 1975
PLACE = "Santa Monica"
REGION = "Americas"
NOTE = ("A scaffold of bands across the top and down the right, a diagonal across the corner where they meet, "
        "and a large pale field of sea light, painted over three times so that the earlier lines and colours "
        "still show through.")

H, W = 2900, 2200
GROUND = "#f2eee5"          # white priming on cotton duck
LIGHT = (-0.6, -0.5, 0.62)
STICK = pigment("#3a3634")  # vine charcoal
UP = np.pi / 2              # strokes up and down the canvas


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


def rect(x0, y0, x1, y1):
    """A rectangle as a polygon; its edges in the order top, right, bottom, left."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def band(x0, x1, top, bottom):
    """A band across from x0 to x1, its top edge running from height top[0] to top[1], its bottom likewise."""
    return [(x0, top[0]), (x1, top[1]), (x1, bottom[1]), (x0, bottom[0])]


def column(y0, y1, left, right):
    """A band down from y0 to y1, its left edge running from left[0] to left[1], its right likewise."""
    return [(left[0], y0), (right[0], y0), (right[1], y1), (left[1], y1)]


def tile(n, across, along, r, octaves=1):
    """Noise that wraps round at its edges, n px square, with features `across` px wide and `along` px
    long down its rows, so that a tool may read it from anywhere without meeting a seam."""
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None]
    out = np.zeros((n, n))
    for k in range(octaves):
        f = np.fft.irfft2(np.fft.rfft2(r.standard_normal((n, n)))
                          * np.exp(-1.2 / 4 ** k * ((fy * along) ** 2 + (fx * across) ** 2)), (n, n))
        out += 0.5 ** k * (f - f.mean()) / f.std()
    return (out / np.sqrt((1 - 0.25 ** octaves) / 0.75)).astype(np.float32)


def frame(a, b, reach, bow=0.0, box=None):
    """The pixels within `reach` px of a path from a to b that bows `bow` px to one side (inside `box`,
    if given): their slices, and for each how far along the path (s) and across it (v) it lies, and the
    length of the path."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = max(float(np.hypot(*(b - a))), 1.0)
    t = (b - a) / L
    n = np.array([-t[1], t[0]])
    pts = np.array([a, b, (a + b) / 2 + n * bow])
    lo, hi = np.floor(pts.min(0) - reach).astype(int), np.ceil(pts.max(0) + reach).astype(int) + 1
    y0, y1, x0, x1 = max(lo[1], 0), min(hi[1], H), max(lo[0], 0), min(hi[0], W)
    if box is not None:
        y0, y1, x0, x1 = max(y0, box[0]), min(y1, box[1]), max(x0, box[2]), min(x1, box[3])
    if y1 - y0 < 2 or x1 - x0 < 2:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dx, dy = (xx - a[0]).astype(np.float32), (yy - a[1]).astype(np.float32)
    s, v = dx * t[0] + dy * t[1], dx * n[0] + dy * n[1]
    if bow:
        q = np.clip(s / L, 0, 1)
        v = v - 4 * bow * q * (1 - q)
    return (slice(y0, y1), slice(x0, x1)), s, v, L


def halfplanes(poly):
    """The edges of a convex polygon as (point, along, inward normal, length)."""
    P = np.asarray(poly, np.float64)
    c = P.mean(0)
    out = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        L = float(np.hypot(*(q - p)))
        t = (q - p) / L
        n = np.array([-t[1], t[0]])
        out.append((p, t, n if np.dot(c - p, n) > 0 else -n, L))
    return out


class Film:
    """One passage of wet paint before it goes onto the canvas: how thick it lies at each px, and its
    colour, kept as the log of the mixture, since paints brushed into each other wet multiply."""

    def __init__(self, box):
        self.box = box
        self.T = np.zeros((box[1] - box[0], box[3] - box[2]), np.float32)
        self.C = np.zeros(self.T.shape + (3,), np.float32)

    def local(self, sl):
        b = self.box
        return slice(sl[0].start - b[0], sl[0].stop - b[0]), slice(sl[1].start - b[2], sl[1].stop - b[2])

    def add(self, sl, d, c):
        ys, xs = self.local(sl)
        self.T[ys, xs] += d
        self.C[ys, xs] += d[..., None] * np.log(np.clip(c, 1e-3, 1))


def around(pts, pad):
    P = np.asarray(pts, np.float64)
    lo, hi = np.floor(P.min(0) - pad).astype(int), np.ceil(P.max(0) + pad).astype(int) + 1
    return max(lo[1], 0), min(hi[1], H), max(lo[0], 0), min(hi[0], W)


class Canvas:
    """The painting as it stands: the colour seen, in linear light, and the height of its surface, the
    weave of the duck under however much paint has filled it. Each sitting begins on what the last one
    left to dry: `tops` is where the surface stands proud (the threads, and the ridges along the edges
    of earlier paint) for a dry brush or a knife to find, `snaps` what it looked like at the start of
    each sitting, for a knife to scrape back to, and `loose` the charcoal not yet fixed under paint."""

    def __init__(self, seed):
        self.r = r = noise.rng(seed)
        duck = canvas.duck((H, W), seed, tint=GROUND, thread=3.0)
        self.img = duck.color.astype(np.float32)
        self.height = (0.3 * duck.tooth).astype(np.float32)
        self.hairs = tile(1024, 1.6, 45, r)        # streaks of single bristles, rows along a stroke
        self.clumps = tile(1024, 7, 120, r)        # bunches of bristles
        self.blots = tile(1024, 140, 140, r, 2)    # the brush heavier here than there
        self.loose = np.zeros((H, W), np.float32)
        self.snaps = []
        self.sitting()

    def sitting(self):
        d = self.height - ndimage.gaussian_filter(self.height, 2.0)
        self.tops = noise.smoothstep(-1.8, 1.8, d / (d.std() + 1e-6)).astype(np.float32)
        self.snaps.append(self.img.copy())
        self.loose *= 0.3

    def read(self, tex, s, v, ss=1.0, sv=1.0):
        n = tex.shape[0]
        i, j = self.r.integers(n, size=2)
        return tex[(np.floor(s * ss).astype(np.intp) + i) % n, (np.floor(v * sv).astype(np.intp) + j) % n]

    def cloud(self, scale):
        """A smooth random field over the canvas with features `scale` px wide, read at a point: f(x, y),
        mostly between -1 and 1."""
        f = noise.field((H // 16 + 2, W // 16 + 2), scale / 16, self.r)
        return lambda x, y: float(f[int(np.clip(y, 0, H - 1)) // 16, int(np.clip(x, 0, W - 1)) // 16])

    def drift(self, n, scale=500, spread=0.8):
        """Weights for n paints mixed from one tint to another: from place to place the painter's mixture
        drifts along them."""
        f, i = self.cloud(scale), np.arange(n)
        return lambda x, y: np.exp(-0.5 * ((i - (n - 1) / 2 * (1 + np.clip(f(x, y), -1, 1))) / spread) ** 2) + 0.03

    def coat(self, film, opac, glaze=0.0, thick=0.035, blend=None):
        """A passage goes on: it covers as much as its thickness hides, and thinned paint stains what it
        lets through (`glaze`). Its colours were brushed into each other while wet, over `blend` px (down,
        across), more along the strokes than across them."""
        y0, y1, x0, x1 = film.box
        T = film.T
        if blend:
            Tb = ndimage.gaussian_filter(T, blend)
            Cb = np.stack([ndimage.gaussian_filter(film.C[..., k], blend) for k in range(3)], -1)
            lc = Cb / np.maximum(Tb, 1e-6)[..., None]
        else:
            lc = film.C / np.maximum(T, 1e-6)[..., None]
        a = (1 - np.exp(-opac * T))[..., None]
        im = self.img[y0:y1, x0:x1]
        if glaze:
            im *= np.exp(glaze * T[..., None] * lc)
        im += (np.exp(lc) - im) * a
        self.height[y0:y1, x0:x1] += thick * T

    def stroke(self, a, b, hw, cols, load=1.0, dry=0.0, spent=0.5, bow=0.0, tails=0.4, rough=1.0, ruler=0,
               share=0.3, feather=0.15, pool=0.5, swell=0.0, box=None):
        """One stroke of a brush `hw` px either side of its path, from a to b, of thin oil: `cols` the
        paints in it, the second carried in streaks across `share` of its width. It lands with a little
        more paint and runs thin toward the end (`spent`); its edges feather out over `feather` of its
        half-width and are torn by the bunches of bristles, unless it ran along a ruler on that side
        (`ruler` +-1); pressed harder and softer it swells and narrows (`swell`). Wet, the paint settles
        into the hollows of the weave (`pool`); a brush carrying little paint (`dry`), or running out,
        breaks, and leaves colour only on the high places. -> (slices, thickness, colour)"""
        fr = frame(a, b, hw * (1 + tails) + 4, bow, box)
        if fr is None:
            return None
        sl, s, v, L = fr
        tops = self.tops[sl]
        hair, clump, blot = self.read(self.hairs, s, v), self.read(self.clumps, s, v), self.read(self.blots, s, v)
        u = np.abs(v)
        if swell:
            u = u / (1 + swell * self.read(self.blots, 0.5 * s, 0 * v))
        rag = rough * (0.06 * hw * clump + 0.4 * hair)
        fw = np.float32(max(feather * hw, 1.2))
        if ruler:
            out = v * ruler > 0
            rag, fw = np.where(out, 0.1 * hair, rag), np.where(out, np.float32(1.2), fw)
        side = noise.smoothstep(-0.5, 1.0, (hw - u + rag) / fw)
        fing = tails * hw * noise.smoothstep(0.3, 1.8, hair + 0.5 * clump)
        corner = 0.3 * hw * (1 - np.sqrt(np.clip(1 - (u / hw) ** 2, 0, 1)))
        ends = noise.smoothstep(-2.0, 0.5 * hw + 2, s - corner) * noise.smoothstep(-2.0, 0.15 * hw + 2, L + fing - s - corner)
        q = np.clip(s / L, 0, 1)
        left = (1 - dry) * (1 - spent * q ** 1.5)
        brk = noise.smoothstep(-0.25, 0.25, left - 0.42 + 0.15 * clump + 0.05 * hair
                               + (0.25 + 0.9 * (1 - left)) * (tops - 0.5))
        body = load * (1 + 0.25 * np.exp(-np.maximum(s, 0) / (0.5 * hw + 4))) \
            * np.clip(1 + 0.04 * hair + 0.08 * clump + 0.3 * blot, 0.1, None) * (1 + pool * left * (0.5 - tops))
        d = side * ends * brk * body * (0.35 + 0.65 * left)
        cols = np.atleast_2d(np.asarray(cols, np.float32))
        c = np.broadcast_to(cols[0], d.shape + (3,))
        if len(cols) > 1:
            z = special.ndtri(1 - share)
            f = noise.smoothstep(z - 0.3, z + 0.3, self.read(self.clumps, s, v, 0.5, 1.4))[..., None]
            c = cols[0] + (cols[1] - cols[0]) * f
        return sl, d, c

    def region(self, poly, kinds, wobble):
        """Where a plane may go: the signed distance (px, positive inside) to the edges of the convex
        polygon `poly`, each edge cut in by hand ('h': it wanders `wobble` px), laid against tape ('t':
        dead straight, but the paint creeps under it along the threads), or off the canvas ('o').
        -> box (y0, y1, x0, x1), distance"""
        r = self.r
        box = around(poly, 40)
        yy, xx = np.ogrid[box[0]:box[1], box[2]:box[3]]
        sd = np.full((box[1] - box[0], box[3] - box[2]), np.inf, np.float32)
        for (p, t, n, L), kind in zip(halfplanes(poly), kinds):
            dx, dy = (xx - p[0]).astype(np.float32), (yy - p[1]).astype(np.float32)
            d = dx * n[0] + dy * n[1]
            if kind == "h":
                k = int(L) + 400
                wob = wobble * (noise.line1d(k, 300, r) + 0.35 * noise.line1d(k, 40, r)) + 0.5 * noise.line1d(k, 5, r)
                d = d + wob[np.clip(dx * t[0] + dy * t[1] + 200, 0, k - 1).astype(np.intp)]
            elif kind == "t":
                creep = 2.5 * noise.smoothstep(0.3, 1.8, noise.field(d.shape, 1.3, r)) \
                    * (1.15 - self.tops[box[0]:box[1], box[2]:box[3]])
                d = np.where((d < 0) & (d > -6), d + creep, d)
            sd = np.minimum(sd, d)
        return box, sd

    def plane(self, poly, cols, aim, hw=(50, 90), length=(300, 900), load=(0.5, 0.9), opac=1.6, dry=(0.0, 0.25),
              spent=(0.2, 0.6), edges="hhhh", wobble=2.5, glaze=0.0, cover=2.0, over=0.08, cut=1.0, weights=None,
              where=None, jitter=0.03, blend=(1.5, 6), smear=0.6, thick=0.035, **kw):
        """A plane of colour brushed in with a broad brush, wet into wet: in each direction of `aim`
        [(angle, share)], give or take `jitter` radians, `cover` coats on average, each coat laid in rows of
        strokes side by side and end to end, set down more often where `where` (x, y) is higher; each brush
        loaded with one of `cols` and a little of its neighbour, by `weights` (fixed, or a function of where
        the stroke is). A few run past the edge; then the edges are cut in along their length, `cut` coats.
        The colours are brushed into each other wet over `blend` px (across, along the first aim). Charcoal
        still loose under the plane is dragged into the paint. -> the film"""
        r = self.r
        box, sd = self.region(poly, edges, wobble)
        film = Film(box)
        cols = np.asarray(cols, np.float32)
        aims = [(aim, 1.0)] if np.ndim(aim) == 0 else aim
        hp = halfplanes(poly)
        P = np.asarray(poly, np.float64)

        def near(c, pad):
            return min(float((c - p) @ nn) for p, _, nn, _ in hp) > -pad

        jobs = []
        for th, w in aims:
            t, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
            ps, pv = P @ t, P @ n
            coats = cover * w
            for _ in range(int(coats) + (r.random() < coats % 1)):
                rows, v = [], pv.min() - r.uniform(0, 0.5) * hw[1]
                while v < pv.max() + 0.5 * hw[0]:
                    half = r.uniform(*hw)
                    s, ss = ps.min() - r.uniform(0.2, 0.9) * length[1], []
                    while s < ps.max():
                        L = r.uniform(*length)
                        ss.append((s + L / 2, L))
                        s += L * r.uniform(0.65, 0.9)
                    rows.append((v + half * r.normal(0, 0.15), half, ss))
                    v += 2 * half * r.uniform(0.6, 0.8)
                for v0, half, ss in rows:
                    for s0, L in ss:
                        c = t * s0 + n * v0
                        e = th + r.normal(0, jitter)
                        d = np.array([np.cos(e), np.sin(e)]) * r.choice([-1, 1])
                        if (near(c, half) or near(c - d * L / 2, half) or near(c + d * L / 2, half)) \
                                and (where is None or r.random() < where(*c)):
                            jobs.append((c - d * L / 2, c + d * L / 2, half, r.normal(0, 0.02) * L))
        order = list(r.permutation(len(jobs)))
        for (p, t, nn, L), kind in zip(hp, edges):            # the edges, cut in last
            if kind != "h" or cut <= 0:
                continue
            half = r.uniform(0.5, 0.8) * np.mean(hw)
            Ls = r.uniform(*length)
            k = int(np.ceil(cut * L / (0.8 * Ls))) + 1
            for s0 in np.linspace(-0.1 * Ls, L - 0.6 * Ls, k) + r.normal(0, 0.1 * Ls, k):
                a = p + t * s0 + nn * half * r.uniform(0.75, 1.0)
                jobs.append((a, a + t * r.uniform(0.7, 1.2) * Ls, half, r.normal(0, 3)))
                order.append(len(jobs) - 1)
        fixed = None if callable(weights) else (np.ones(len(cols)) if weights is None else np.asarray(weights, float))
        for i in order:
            a, b, half, bow = jobs[i]
            p = np.asarray(weights(*((np.asarray(a) + b) / 2)) if fixed is None else fixed, float)
            k = r.choice(len(cols), p=p / p.sum())
            j = int(np.clip(k + r.choice([-1, 1]), 0, len(cols) - 1))
            out = self.stroke(a, b, half, cols[[k, j]], load=r.uniform(*load), dry=r.uniform(*dry),
                              spent=r.uniform(*spent), bow=bow, share=r.uniform(0.1, 0.4), box=box, **kw)
            if out:
                sl, d, c = out
                ys, xs = film.local(sl)
                reach = r.uniform(3, 10) if r.random() < over else 0.0
                film.add(sl, d * noise.smoothstep(-0.7, 0.7, sd[ys, xs] + reach), c)
        th = aims[0][0]
        self.coat(film, opac, glaze, thick, blend[::-1] if abs(np.sin(th)) > 0.7 else blend)
        if smear:
            ys, xs = slice(box[0], box[1]), slice(box[2], box[3])
            cov = 1 - np.exp(-opac * film.T)
            loose = ndimage.gaussian_filter(self.loose[ys, xs], (2 + 10 * abs(np.sin(th)), 2 + 10 * abs(np.cos(th))))
            self.img[ys, xs] *= np.exp(-(smear * loose * cov)[..., None] * STICK)
            self.loose[ys, xs] *= 1 - 0.8 * cov
        return film

    def rule(self, a, b, hw, cols, load=0.9, opac=1.8, side=1, reach=(250, 600), slip=1.0, glaze=0.0, dry=0.35,
             **kw):
        """A line painted with a small brush run along a straightedge from a to b: the edge against the
        ruler straight, the other as the bristles left it, the brush reloaded every so often, now full
        and now nearly dry (up to `dry`), so that it runs dense and thin and breaks over the weave, and
        the ruler set down again a hair off where it had been."""
        r = self.r
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        L = float(np.hypot(*(b - a)))
        t, n = (b - a) / L, np.array([-(b - a)[1], (b - a)[0]]) / L
        film = Film(around([a, b], hw + 12))
        s0, s1, off = 0.0, 0.0, 0.0
        while s1 < L:
            s1 = min(L, s0 + r.uniform(*reach))
            if L - s1 < 60:
                s1 = L
            if r.random() < 0.4:
                off = np.clip(off + r.normal(0, slip), -2.5, 2.5)
            out = self.stroke(a + t * s0 + n * off, a + t * s1 + n * (off + r.normal(0, 0.4 * slip)), hw, cols,
                              load=load * r.uniform(0.6, 1.2), ruler=side, tails=0.15, rough=0.7, feather=0.45,
                              spent=r.uniform(0.4, 0.9), dry=r.uniform(0, dry), share=0.3, swell=0.2,
                              box=film.box, **kw)
            if out:
                film.add(*out)
            s0 = s1 - r.uniform(5, 30)
        self.coat(film, opac, glaze)

    def charcoal(self, lines, width=4.5, pressure=0.75, dark=0.8):
        """Lines drawn in charcoal: [(points, wander px)], ruled (wander under 1: straight from point to
        point, so that where the ruler was moved the line kinks) or freehand. The stick catches the high
        places, and the dust is a little smudged."""
        press = np.zeros((H, W), np.float32)
        for P, wander in lines:
            pencil.line(press, np.asarray(P, np.float64), width * self.r.uniform(0.8, 1.2), self.r,
                        pressure=pressure * self.r.uniform(0.7, 1.1), wander=wander, tremor=0.3,
                        lift=(300, 900), corners=wander < 1)
        got = dark * ndimage.gaussian_filter(pencil.catch(press, SimpleNamespace(tooth=self.tops), grip=0.9, soft=0.35), 0.9)
        self.img *= np.exp(-got[..., None] * STICK)
        self.loose += got

    def drip(self, x, y, length, width, col, load=0.8, opac=1.6):
        """Thinned paint run down from (x, y): it finds its way down the weave, narrows, and stops in a bead."""
        r = self.r
        n = max(int(length), 8)
        t = np.linspace(0, 1, n + 1)
        cx = x + (1 + n / 250) * noise.line1d(n + 1, 160, r) * t + 0.3 * noise.line1d(n + 1, 5, r)
        hw = width * (0.4 + 0.6 * (1 - t) ** 1.3) * (1 + 0.2 * noise.line1d(n + 1, 30, r)) \
            * (1 + 1.2 * np.exp(-t * n / 12))
        br = 1.2 * hw[-1] + 0.8
        box = (max(int(y), 0), min(int(y + n + br + 3) + 1, H), max(int(cx.min() - 3 * width - br - 3), 0),
               min(int(cx.max() + 3 * width + br + 3) + 1, W))
        if box[1] <= box[0] or box[3] <= box[2]:
            return
        yy, xx = np.mgrid[box[0]:box[1], box[2]:box[3]].astype(np.float32)
        k = np.clip(yy - int(y), 0, n).astype(np.intp)
        dd = np.abs(xx - cx[k]) - 0.5 * (self.tops[box[0]:box[1], box[2]:box[3]] - 0.5)
        body = noise.smoothstep(hw[k] + 0.8, hw[k] - 0.8, dd) * (yy - int(y) <= n)
        bead = noise.smoothstep(br + 0.8, br - 0.8, np.hypot(xx - cx[-1], 0.8 * (yy - int(y) - n)))
        rim = np.exp(-((hw[k] - dd) / 0.9) ** 2) * body
        film = Film(box)
        film.T[:] = load * np.maximum(body * (0.55 + 0.25 * np.exp(-(yy - y) / 50)
                                              + 0.15 * noise.line1d(n + 1, 40, r)[k]) * (1 + 0.4 * rim), 1.3 * bead)
        film.C[:] = film.T[..., None] * np.log(np.asarray(col, np.float32))
        self.coat(film, opac, thick=0.12)

    def scrape(self, a, b, hw, amount, back=1, bow=0.0):
        """A blade pulled from a to b through the paint of this sitting, or `back` sittings: it takes the
        paint off the high places and leaves it in the hollows, so that what was there before comes up
        through it, in streaks where its edge was nicked. The hand presses harder and softer as it goes, the
        blade tilts, and it is set down softly and lifted off."""
        fr = frame(a, b, 1.3 * hw + 4, bow)
        if fr is None:
            return
        sl, s, v, L = fr
        k = int(L) + 1
        i = np.clip(s, 0, k - 1).astype(np.intp)
        press = np.clip(1 + 0.45 * noise.line1d(k, 150, self.r), 0.2, None)[i]
        wide = hw * (1 + 0.2 * noise.line1d(k, 300, self.r))[i]
        nick = self.read(self.hairs, s, v, 0.3, 1.0)
        foot = noise.smoothstep(-1, 3, wide - np.abs(v) + 0.6 * nick) * noise.smoothstep(-2, 60, s) \
            * noise.smoothstep(-2, 120, L - s)
        take = np.clip(amount * press * foot * (0.3 + 0.7 * self.tops[sl]) * (1 + 0.3 * nick), 0, 0.95)
        im = self.img[sl]
        im += (self.snaps[-back][sl] - im) * take[..., None]
        self.height[sl] -= 0.05 * take


def first(cv):
    """The drawing, and washes of colour thinned with turpentine: orange across the top, cerulean and
    then a warm cream below it, sea green through the field, violet down the side. They run."""
    r = cv.r
    cv.charcoal([([(-20, 160), (1460, 160)], 0.6), ([(-20, 400), (1460, 400)], 0.6),
                 ([(-20, 622), (1000, 618), (1460, 620)], 0.6), ([(1460, -20), (1458, 2920)], 0.6),
                 ([(420, 620), (430, 2920)], 2.5), ([(700, -20), (1460, 622)], 0.6),
                 ([(1460, 1500), (2220, 1520)], 2.0)], width=5.0)
    wash = dict(load=(0.3, 0.6), opac=0.9, glaze=0.8, dry=(0.0, 0.1), spent=(0.3, 0.6), wobble=4.0, over=0.25,
                pool=0.9, length=(500, 1200))
    cv.plane(rect(-40, -40, 1460, 160), pal("#e9a04a", "#eeb05a", "#e48f40"), 0.0, hw=(40, 70), **wash)
    cv.plane(rect(-40, 160, 1460, 400), pal("#8ab8de", "#9cc4e3", "#7dafd9"), 0.0, hw=(50, 90), **wash)
    cv.plane(rect(-40, 400, 1460, 620), pal("#e6cf9c", "#ead8ac", "#dfc48e"), 0.0, hw=(50, 90), **wash)
    cv.plane(rect(-40, 620, 1460, 2950), pal("#8fcab4", "#a3d5c8", "#84c2ab", "#9bcfd2"),
             [(UP, 1.0), (0.4, 0.5)], hw=(70, 120), blend=(8, 8), **wash)
    cv.plane(rect(1460, -40, 2240, 2950), pal("#b9a2cf", "#c8b5da", "#d3b9cf"), UP, hw=(60, 100), **wash)
    for x in r.uniform(0, 1450, 9):
        cv.drip(x, 150 + r.uniform(-5, 5), r.uniform(80, 400), r.uniform(2, 4), lin("#e59a48"), load=0.6, opac=1.0)
    for x in r.uniform(1470, 2190, 2):
        cv.drip(x, r.uniform(300, 2000), r.uniform(100, 400), r.uniform(2, 3), lin("#b39acb"), load=0.4, opac=1.0)


def second(cv):
    """Drawn again, with more bands and the side further right, and painted in: yellow, lilac, sea green
    and a pale blue across the top, cerulean in the field and deeper below a line across it, cream and a
    blue strip down the side, and the diagonal ruled in violet. Scraped back in places while wet; some of it
    runs."""
    r = cv.r
    cv.charcoal([([(-20, 108), (1540, 108)], 0.5), ([(-20, 300), (1540, 302)], 0.5),
                 ([(-20, 520), (800, 521), (1540, 519)], 0.5), ([(-20, 720), (1540, 722)], 0.5),
                 ([(1540, -20), (1541, 2920)], 0.5), ([(1880, -20), (1880, 2920)], 0.5),
                 ([(2010, -20), (2012, 2920)], 0.5), ([(1190, 720), (1186, 2920)], 0.5),
                 ([(-20, 1720), (1540, 1716)], 0.5), ([(980, -20), (1540, 560)], 0.5)], width=4.5, dark=1.1)
    oil = dict(load=(0.5, 0.9), opac=1.5, dry=(0.0, 0.3), wobble=3.0)
    field = dict(hw=(70, 110), length=(500, 1100), blend=(8, 8), **oil)
    cv.plane(rect(-40, 720, 1540, 1720), pal("#a9cbe8", "#9cc2e4", "#b2d1e6", "#a3cbdc"),
             [(UP, 1.0), (0.0, 0.5), (0.7, 0.3)], **field)
    cv.plane(rect(-40, 1720, 1540, 2950), pal("#94bce0", "#8ab5dc", "#9cc2e2", "#91bfd5"),
             [(UP, 1.0), (0.0, 0.4)], **field)
    across = dict(length=(600, 1300), **oil)
    cv.plane(rect(-40, -40, 1540, 108), pal("#f1c13c", "#f0b43a", "#f3cd55"), 0.0, hw=(35, 60), **across)
    cv.plane(rect(-40, 108, 1540, 300), pal("#cbbadf", "#c0addb", "#d6c8e4"), 0.0, hw=(45, 80), **across)
    cv.plane(rect(-40, 300, 1540, 520), pal("#86c0a2", "#93c8ac", "#7ab595"), 0.0, hw=(45, 80), **across)
    cv.plane(rect(-40, 520, 1540, 720), pal("#cfe0ee", "#c4d9ec", "#d8e6ef"), 0.0, hw=(45, 80), **across)
    cv.plane(rect(1540, -40, 1880, 2950), pal("#efd6c8", "#ecccbd", "#f2dfd2"), UP, hw=(50, 90), **across)
    cv.plane(rect(1880, -40, 2010, 2950), pal("#4f86c9", "#5b92cf", "#457ac2"), UP, hw=(30, 55), **across)
    cv.plane(rect(2010, -40, 2240, 2950), pal("#f0ebe0", "#ece5d6", "#f3efe6"), UP, hw=(45, 80), **across)
    cv.rule((980, -20), (1540, 560), 4.0, pal("#9a86c6", "#8a74bc"), side=1)
    # the knife, through the wet field and across the sea green
    for _ in range(7):
        y, x = r.uniform(900, 2800), r.uniform(0, 900)
        cv.scrape((x, y), (x + r.uniform(300, 700), y + r.normal(0, 20)), r.uniform(25, 50), r.uniform(0.4, 0.8))
    for _ in range(3):
        x = r.uniform(100, 1400)
        cv.scrape((x, r.uniform(800, 1500)), (x + r.normal(0, 15), r.uniform(1900, 2800)), r.uniform(20, 45), 0.6)
    cv.scrape((200, 410), (1100, 405), 40, 0.6)
    for x in r.uniform(0, 1500, 8):
        cv.drip(x, 100 + r.uniform(-5, 5), r.uniform(60, 300), r.uniform(2, 3.5), lin("#efb93c"))
    for x in r.uniform(1885, 2005, 2):
        cv.drip(x, r.uniform(600, 2200), r.uniform(150, 500), r.uniform(2, 3.5), lin("#4f86c9"))


def third(cv):
    """The last sitting: drawn a last time; the line between the field and the side ruled in red-orange
    and the field painted up to it, pale cerulean and then scumbles of white, and the knife pulled down
    through it; the bands narrowed and broken, with orange low among them, and their left end painted out;
    the side cream, divided low down, blue and a pale sea green, with a deep blue block and a lilac one at
    the top; a pale triangle across the corner where they meet and its diagonal ruled in violet, twice;
    the last lines ruled, and some charcoal over the dry paint."""
    r = cv.r
    cv.charcoal([([(-20, 72), (1600, 72)], 0.4), ([(-20, 236), (1600, 238)], 0.4), ([(-20, 292), (1600, 291)], 0.4),
                 ([(-20, 458), (1600, 460)], 0.4), ([(-20, 680), (1600, 678)], 0.4), ([(1590, -20), (1591, 2920)], 0.4),
                 ([(1900, -20), (1899, 2920)], 0.4), ([(1938, 470), (1940, 2920)], 0.4), ([(1122, -40), (1596, 466)], 0.4),
                 ([(1590, 2238), (1904, 2241)], 0.4)])
    cv.rule((1591, 470), (1591, 2950), 6.5, pal("#e05a36", "#e8744a"), side=-1, load=1.1, opac=2.2)
    cv.rule((38, 680), (38, 2950), 4.0, pal("#5d97cf", "#6aa3d3"), side=1)

    pale, holes = cv.cloud(450), cv.cloud(700)

    def sky(x, y):      # paler toward the top and where more white went in, a little aqua low on the left
        up = np.clip(1 - (y - 680) / 2200 + 0.25 * pale(x, y), 0, 1)
        return [0.5 + up, 1.0, 1.0 - 0.5 * up, 0.4 + 0.6 * up, 0.2 + 0.5 * (1 - up) * (x < 900)]
    field = [(42, 684), (1584, 686), (1587, 2884), (44, 2880)]
    cv.plane(field, pal("#c6dbec", "#b6d1e9", "#a7c7e4", "#cfe1ed", "#b5d7de"), [(UP, 1.0), (0.0, 0.2)],
             hw=(70, 115), length=(500, 1300), weights=sky, cover=1.8, load=(0.35, 0.85), opac=1.4,
             dry=(0.0, 0.2), blend=(2, 5), jitter=0.05,
             where=lambda x, y: 0.6 + 0.4 * noise.smoothstep(-1.2, 0.6, holes(x, y)))
    cv.plane(field, pal("#e2ecf1", "#d9e7f0", "#eaeeed", "#d6e9e6"), [(UP, 1.0), (0.5, 0.15), (-0.5, 0.15)],
             hw=(50, 90), length=(400, 900), cover=1.3, jitter=0.06, load=(0.3, 0.6), opac=1.6,
             dry=(0.3, 0.6), spent=(0.5, 0.9), cut=0, smear=0, blend=(1.5, 4),
             where=lambda x, y: 0.1 + 0.9 * np.clip(1 - (y - 684) / 1800, 0, 1) * noise.smoothstep(-1.0, 0.8, pale(x, y)))
    for x in (240, 690, 1180):       # the knife, down through the field low on the canvas
        x += r.normal(0, 50)
        cv.scrape((x, r.uniform(1450, 1850)), (x + r.normal(0, 25), r.uniform(2550, 2860)), r.uniform(40, 65),
                  r.uniform(0.25, 0.4))
    for x in r.uniform(120, 1500, 3):
        cv.drip(x, r.uniform(700, 900), r.uniform(250, 700), r.uniform(2, 3.5), lin("#97bde0"), load=0.6)
    cv.plane(rect(-40, 684, 38, 2950), pal("#efeadf", "#f2eee6", "#e8e2d4"), UP, hw=(20, 30), edges="hhoo")
    oil = dict(load=(0.6, 1.0), opac=1.9, dry=(0.0, 0.15), wobble=2.0, length=(600, 1400), blend=(1.5, 3))

    def lay(poly, cols, hw, aim=0.0, edges="hhho", **kw):
        """A band brushed in along its length, its mixture drifting from place to place."""
        cv.plane(poly, pal(*cols), aim, hw=hw, edges=edges, weights=cv.drift(len(cols), 500), **{**oil, **kw})

    veil = dict(cover=1.4, opac=1.6, load=(0.35, 0.9))     # fewer coats, so that what is under them shows

    lay(band(-40, 1300, (-40, -40), (72, 70)), ("#f2d27a", "#f3cb52", "#efc04a", "#e8b445"), (30, 45), edges="ohto")
    lay(band(-40, 1600, (86, 82), (236, 239)), ("#f1efea", "#ecebea", "#e5e3e9", "#dfdce6"), (40, 70), **veil)
    lay(band(-40, 230, (86, 86), (160, 158)), ("#82b3dc", "#6fa6d6", "#5d97cf"), (20, 30), length=(200, 300))
    lay(band(-40, 980, (240, 241), (288, 286)), ("#9cd0bb", "#8cc7b0", "#7fbfa5", "#74b59b"), (20, 30))
    lay(band(984, 1600, (241, 242), (286, 288)), ("#e0dce6", "#d6d2e0", "#ccc7da"), (20, 30), edges="hhhh")
    lay(band(-40, 1600, (294, 292), (456, 458)), ("#d1e8e3", "#c3e2dd", "#bcdde4", "#b4dbd7"), (40, 70), **veil)
    lay(band(-40, 1010, (460, 461), (490, 489)), ("#ee8244", "#ea7038", "#e2622f"), (14, 18), edges="thto", cover=3.5,
        load=(0.8, 1.2))
    lay(band(-40, 1600, (494, 495), (676, 680)), ("#e8eef0", "#e3ece9", "#dfeaf0", "#d3e3ed"), (40, 70), **veil)
    # the left end of the lower bands painted out in white, thin, so that they show through it
    cv.plane(band(-40, 420, (294, 292), (676, 680)), pal("#f1f0ea", "#ecefec", "#f4f2ec"), 0.0, hw=(40, 70),
             edges="hhho", load=(0.4, 0.7), opac=1.3, dry=(0.2, 0.5), spent=(0.4, 0.8), cover=1.5,
             length=(300, 700), wobble=3.0)
    # the side
    lay(rect(1600, -40, 2240, 96), ("#f2e6cc", "#efe0bf", "#ecd8af"), (30, 50), edges="oohh", length=(500, 900))
    block = dict(spent=(0.05, 0.3), dry=(0.0, 0.05), load=(0.5, 0.9), cover=2.2)
    lay(band(1600, 1896, (98, 100), (292, 290)), ("#6b9bd6", "#5a8fd0", "#4a7ec6", "#3f6fbe"), (40, 70), edges="hhhh",
        length=(450, 700), thick=0.02, **block)
    lay(band(1904, 2240, (120, 118), (262, 264)), ("#d0c5e0", "#c6b8da", "#bba9d5"), (40, 70), edges="hohh",
        length=(500, 800), opac=1.3, **block)
    lay(band(1600, 2240, (294, 296), (464, 462)), ("#f3ece0", "#f0e6d4", "#ecdcc4"), (40, 70), edges="hohh",
        length=(600, 1000))
    down = dict(aim=UP, length=(600, 1300))
    lay(column(470, 2236, (1600, 1600), (1896, 1898)), ("#f2e9da", "#efe3d0", "#eadbc6", "#ecdcd0"), (50, 85),
        edges="hhhh", **veil, **down)
    lay(column(2244, 2950, (1600, 1600), (1898, 1897)), ("#f1dccb", "#eed3c2", "#e9cbbd"), (50, 85), edges="hhoh",
        **down)
    lay(column(470, 2950, (1904, 1903), (1934, 1935)), ("#7aaed8", "#6a9fd2", "#5b93cc"), (13, 17), edges="htot", **down)
    lay(column(470, 2950, (1940, 1942), (2108, 2106)), ("#cfe6dc", "#c4e0d4", "#b7d9cb", "#a9d2c4"), (35, 60),
        edges="hhoh", **veil, **down)
    lay(column(470, 2950, (2112, 2110), (2240, 2240)), ("#f4f1ea", "#f1ece2", "#ece6d8"), (35, 60), edges="hooh",
        **down)
    # the diagonal ruled across the corner, once lightly and once again beside it, and the pale triangle
    # laid over the ends of the bands up to it
    cv.rule((1114, -40), (1584, 462), 3.0, pal("#9a86c6"), side=1, load=0.5, opac=1.4)
    cv.rule((1122, -40), (1594, 464), 4.5, pal("#8a74bc", "#9a86c6"), side=1)
    cv.plane([(1134, -40), (1600, -40), (1600, 458)], pal("#f2ece0", "#eee6d8", "#f5f1e8"), -0.85, hw=(50, 80),
             edges="oht", opac=1.6, load=(0.5, 0.8), dry=(0.0, 0.2), wobble=2.0, cover=3.0, length=(300, 700),
             blend=(6, 6))
    # ruled lines in paint
    cv.rule((1010, 474), (1590, 474), 3.0, pal("#e2603a", "#e8744a"), side=1)
    cv.rule((-20, 79), (1300, 79), 4.0, pal("#8a74bc", "#7d68b2"), side=1)
    cv.rule((2160, 470), (2160, 2950), 3.5, pal("#4a7ec6", "#5d97cf"), side=-1)
    cv.rule((1600, 2240), (1898, 2240), 3.0, pal("#e2603a", "#e8744a"), side=1)
    # it runs
    for x in r.uniform(0, 1000, 4):
        cv.drip(x, 488, r.uniform(60, 260), r.uniform(1.8, 3), lin("#e8743c"))
    for x in r.uniform(0, 970, 3):
        cv.drip(x, 286, r.uniform(40, 160), r.uniform(1.8, 3), lin("#86c0a2"))
    for x in r.uniform(1906, 1932, 2):
        cv.drip(x, r.uniform(900, 2400), r.uniform(100, 400), r.uniform(2, 3), lin("#5d97cf"))
    cv.sitting()
    # charcoal over the dry paint, some lines drawn twice
    cv.charcoal([([(-20, 292), (800, 293), (1600, 291)], 0.4), ([(-20, 680), (1600, 677)], 0.4),
                 ([(1126, -40), (1600, 461)], 0.4), ([(1900, 300), (1899, 2920)], 0.4),
                 ([(420, 294), (421, 456)], 0.4), ([(-20, 1716), (700, 1714)], 0.5),
                 ([(42, 2884), (1584, 2880)], 1.5)], width=3.5, pressure=0.7)


def light(cv):
    """Gallery light on the finished canvas: thin oil is matt, and only the weave and the ridges at the
    edges of the paint catch it."""
    return dabs.shine(cv.img, ndimage.gaussian_filter(cv.height, 0.6), light=LIGHT, relief=0.8, gloss=0.0,
                      reach=(0.86, 1.12))


def paint(seed=1975):
    cv = Canvas(seed)
    first(cv)
    cv.sitting()
    second(cv)
    cv.sitting()
    third(cv)
    return light(cv)
