"""Abstract Picture. Oil on canvas, brushed, then dragged with a squeegee, again and again.

In Cologne, from the late 1980s, Gerhard Richter worked each abstract picture over many sessions. The
colour went on first in loose passages with a broad brush. Then he took a squeegee, a long blade of
plexiglass as wide as the canvas, and pulled it down through the paint. The blade scrapes the paint off
the high places and carries it along in a bead, laying a skin of it into the low ones, so that the
colours are dragged into each other down the pull. Where a ridge lifts it, it leaves what is under it
untouched for a stretch after; where the paint has half set, the film splits between blade and canvas
in cells and tacky paint rips away. Each pull lands and lifts on an uneven line and leaves some of its
load piled where it lifts. Between the pulls he put on fresh paint, a band of yellow, a smear of white,
and let it set, so that the later pulls carry new colour through old and break through to what is
beneath. Last, with a shorter blade and a few dabs of colour along its edge, he pulled across the
curtain low down, here and there, and the dabs ran out one after another.
"""

import zlib

import numpy as np
from PIL import Image
from scipy import ndimage, special

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Abstract Picture"
DATE = "2026"
MEDIUM = ("Oil on canvas: loose passages of colour brushed on, then dragged and scraped with a squeegee, wet into "
          "wet and over half-dry paint, with fresh colour between the pulls")
AFTER = "Gerhard Richter, the squeegee paintings (Abstraktes Bild), Cologne, late 1980s and 1990s"
ROOM = "Colour Itself"
YEAR = 1990
PLACE = "Cologne"
REGION = "Europe"
NOTE = ("Colour brushed on in broad passages, then dragged down the canvas with a long squeegee, again and again "
        "over some days. Each pull scrapes the paint off the high places and lays it into the low ones, and skips, "
        "so that the earlier layers show through.")

H, W = 2700, 2400
LIGHT = (-0.6, -0.5, 0.62)
HIDE = 0.13     # px of film that hides two thirds of what is under it
RHO = 220.0     # the radius the blade bends to where a ridge lifts it, px
LOW = float(np.log(1e-3))


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


RED = pal("#a8121a", "#c41d1b", "#d62a1e", "#e4402a")        # cadmium red, deep to light
YELLOW = pal("#eea20c", "#f5bf12", "#f7d31c", "#f4df4c")     # cadmium yellow, deep to lemon
GREEN = pal("#05332c", "#09453a", "#0d5847", "#156b55")      # phthalo green, from the tube to a little let down
BLUE = pal("#121a63", "#1a2783", "#25389e", "#3a50b4")       # ultramarine
WHITE = pal("#e9e8e2", "#f0eee7", "#f4f2ec")

# loose, loaded strokes of a broad brush
LOOSE = dict(thick=0.03, grooves=0.25, lips=0.1, land=0.4, lift=0.25, tails=0.5, pickup=0.4, merge=3, spent=0.5,
             ends=(0.3, 0.2), taper=0.15, fray=2.0, flatten=0.6, dry=(0.0, 0.15))


def streaks(shape, across, along, g):
    """Smooth noise drawn out down the canvas, features `across` px wide and `along` px long, rounded at the ends."""
    h, w = shape
    small = noise.field((int(h * across / along) + 4, w), across, g)
    return np.asarray(Image.fromarray(small).resize((w, int(small.shape[0] * along / across)), Image.BICUBIC))[:h]


class Canvas:
    """The painting as it stands. Below, the paint that has set: its relief down to the weave (F), its
    colour (U), the colour of the layer under its top one (V) and that top layer's thickness (D), and
    how tacky it still is (K: 1 just set, 0 dry). On it, the fresh paint of this sitting: its thickness
    (T) and colour (C), and where it lies (wet, for the brush). Colours are kept as logs, so that they
    mix as paints do, by multiplying."""

    NAMES = ("F", "U", "V", "D", "K", "T", "C", "wet")

    def __init__(self, seed, shape=(H, W)):
        g = canvas.duck(shape, seed, tint="#eeebe3", thread=2.6)
        z = np.zeros(shape, np.float32)
        self.F, self.U = (g.tooth * 0.25).astype(np.float32), np.log(g.color).astype(np.float32)
        self.V, self.D, self.K = self.U.copy(), z.copy(), z.copy()
        self.T, self.C, self.wet = z.copy(), self.U.copy(), z.copy()

    def look(self):
        """The colour seen (log): the fresh film over what has set, as much as its thickness hides."""
        a = 1 - np.exp(-self.T / HIDE)
        return self.U + (self.C - self.U) * a[..., None]

    def turn(self, k):
        """Turn the canvas k quarter turns on the easel."""
        for n in self.NAMES:
            setattr(self, n, np.ascontiguousarray(np.rot90(getattr(self, n), k)))

    def brush(self, g, paths, w, cols, **kw):
        """Paint brushed into the film, with the thick-oil engine."""
        rgb, h = np.exp(self.look()), self.F + self.T
        h0 = h.copy()
        impasto.lay(rgb, h, paths, w, cols, g, wet=self.wet, **kw)
        self.T = np.maximum(h - self.F, 0)
        hit = (np.abs(h - h0) > 1e-3)[..., None]
        a = np.maximum(1 - np.exp(-self.T / HIDE), 0.25)[..., None]
        new = np.clip(self.U + (np.log(np.clip(rgb, 1e-3, 1)) - self.U) / a, LOW, 0)
        self.C = np.where(hit, new, self.C).astype(np.float32)

    def dry(self, g, days, scale=500):
        """The sitting ends and the fresh paint sets, thick paint more slowly than thin and some places
        sooner than others. It is the surface the next pulls ride on, rip and break through to."""
        has = self.T > 0.01
        v = self.look()
        self.V = np.where(has[..., None], self.U, self.V)
        self.U, self.C = v, v.copy()
        self.D = np.where(has, self.T, self.D)
        self.F += self.T
        rate = np.exp(0.7 * noise.field(self.T.shape, scale, g)) * (0.3 + 0.1 / (self.T + 0.15))
        self.K = np.where(has, np.exp(-days * rate), self.K * np.exp(-days)).astype(np.float32)
        self.T[:], self.wet[:] = 0, 0


def ramp(a, b, x):
    return np.clip((x - a) / (b - a), 0, 1)


def pull(cv, g, top, bottom, way=0, press=1.0, gap=0.06, hydro=0.15, load=None, drag=0.5, cells=0.3, tear=0.0,
         chatter=0.0, fall=0.05, cap=40.0, rate=0.15, lean=0.0, nicks=4, grit=1, blade=None, hand=0.0):
    """One pull of the squeegee down the canvas (turned `way` quarter turns first), landing about row
    `top` and lifting about row `bottom` (either may be a row for each column).

    The blade rides on the set paint, bridging the hollows and lifted by the ridges, and falls back
    only slowly after a ridge (`fall`, px of height per px of travel). Under it is a gap, wider where
    the hand presses less (`press`, `gap`), at the nicks in its edge, and the more paint it pushes
    (`hydro`), opening and closing as it judders (`chatter`). Fresh paint standing above the blade is
    sheared off into the bead; the top of the film under it is dragged along (`drag`); the bead fills
    the gap as far down as it reaches, a small bead only the high places. Where the film fills the gap
    it splits between blade and canvas in cells (`cells`); where the blade presses on half-set paint,
    its skin rips away in places (`tear`) and shows the layer beneath. The bead holds at most `cap` px
    of paint, can come loaded (`load`: amount and log colour across the blade), and leaves some of it
    as a ridge where the blade lifts.

    A shorter blade covers only the columns `blade` (x0, x1) and carries its bead with it; the paint tears
    unevenly along its ends. A `hand` (px) sets the blade down gently, so that at first it catches only
    the high places, on a line that wanders that far, one end before the other; sways a shorter blade
    sideways a third as far; and lifts it as unevenly, leaving its load piled along some stretches and
    none along others."""
    cv.turn(way)
    F, U, V, D, K, T, C = cv.F, cv.U, cv.V, cv.D, cv.K, cv.T, cv.C
    h, w = T.shape
    f32 = lambda a: np.asarray(a, np.float32)
    line = lambda n, s: noise.line1d(n, s, g)
    hide = lambda d: 1 - np.exp(-d / HIDE)
    u = f32(np.linspace(-0.5, 0.5, w))
    x = np.arange(w)
    start = top + 16 * line(w, 400) + 5 * line(w, 60) + line(w, 5)
    stop = bottom + 18 * line(w, 400) + 6 * line(w, 60) + line(w, 5)
    if hand:
        start, stop = (s + hand * (line(w, 900) + 0.6 * line(w, 120) + 0.15 * line(w, 20)) for s in (start, stop))
    start, stop = np.rint(start).astype(int), np.rint(stop).astype(int)
    r0, r1 = max(start.min(), 0), min(stop.max(), h)
    n = r1 - r0
    P = f32(press * np.exp(lean * u + 0.3 * line(w, 700)))          # the hand presses, more at one end
    ease = f32(90 * np.exp(0.4 * line(w, 200)))                    # and comes down as it lands, not at once
    land = 0.1 if hand else 0.25                                    # a hand sets it down more gently still
    Pt = f32(np.exp(0.2 * line(n, 500)))
    phase = np.cumsum(1 / (24 * np.exp(0.3 * line(n, 300))))       # stick and slip, ~2 cm apart
    late = f32(1.2 * line(w, 900) + 0.8 * u)                        # one end judders a little after the other
    jud = f32(chatter * ramp(0.2, 1.0, line(n, 350)))               # it judders only now and then,
    where = f32(ramp(-0.3, 0.6, line(w, 450)))                       # and along part of its length
    # the edge: uneven along its length, a few chips that let more paint through, grit that scrapes a groove
    edge = 1 + 0.12 * line(w + 64, 5) + 0.08 * line(w + 64, 16) + 0.04 * line(w + 64, 2)
    for c0, d in zip(g.uniform(0, w + 64, nicks), g.choice([0.8, -0.5], nicks, p=[0.6, 0.4])):
        edge *= 1 + d * np.exp(-((np.arange(w + 64) - c0) / g.uniform(0.7, 1.8)) ** 2)
    edge = f32(np.maximum(edge, 0.05))
    drift = np.rint(32 + 6 * line(n, 800)).astype(int)              # it wanders sideways a little
    grits = [(int(g.integers(1, w - 1)), int(g.integers(0, n)), g.exponential(250)) for _ in range(grit)]
    env = F.copy()
    for a in np.unique(np.geomspace(1, 64, 16).astype(int)):       # the blade bent over the ridges that lift it
        np.maximum(env, ndimage.maximum_filter1d(F, 2 * a + 1, axis=1) - a * a / (2 * RHO), out=env)
    env = np.maximum(ndimage.gaussian_filter1d(env, 1.5, axis=1), F)
    # where the film splits: big holes and long tears in some places, small cells in others, and in some
    # hardly any, in streams down the pull
    size = ramp(-0.8, 0.8, streaks((h, w), 80, 500, g))
    pat = size * (0.8 * streaks((h, w), 14, 90, g) + 0.4 * streaks((h, w), 5, 30, g)) \
        + (1 - size) * (0.4 * streaks((h, w), 5, 30, g) + 0.8 * streaks((h, w), 2, 9, g))
    lim = 2.0 - 1.7 * ramp(0.0, 1.5, streaks((h, w), 60, 450, g))
    cell = ramp(lim - 0.08, lim + 0.08, pat + 0.15 * noise.field((h, w), 1.2, g))
    rips = ramp(1.0, 1.15, streaks((h, w), 10, 70, g) + 0.35 * noise.field((h, w), 2.5, g)) * tear
    vein = special.ndtr((0.7 * streaks((h, w), 4, 40, g) + 0.5 * streaks((h, w), 1.5, 12, g)) / 0.86)
    patch = special.ndtr((0.7 * streaks((h, w), 9, 60, g) + 0.5 * streaks((h, w), 3, 18, g)) / 0.86)
    rx = f32(rate * np.exp(0.35 * line(w, 7)))
    hang = f32(0.7 * np.exp(0.35 * line(w, 9)))                    # how far the bead hangs below the blade
    drop = f32(fall * np.exp(0.4 * line(w, 15)))                    # and how fast a lifted blade comes down
    sway = np.zeros(n, int)
    if blade is not None:
        mid, half = (blade[0] + blade[1]) / 2, (blade[1] - blade[0]) / 2
        sway = np.rint(hand / 3 * line(n, 700)).astype(int)
        torn = 0.04 * half * np.stack([line(n, 40), line(n, 40)])
    # the bead: its body, and what it has just taken up and not yet rolled in
    L1 = np.zeros(w, np.float32) if load is None else f32(load[0])
    b1 = np.zeros((w, 3), np.float32) if load is None else f32(load[1])
    L2, b2 = np.zeros(w, np.float32), np.zeros((w, 3), np.float32)
    zp = np.full(w, -np.inf, np.float32)
    Lend, Bend, yend = np.zeros(w, np.float32), np.zeros((w, 3), np.float32), np.full(w, -1)
    for k in range(n):
        y = r0 + k
        on = (y >= start) & (y < stop)
        if blade is not None:
            at = x - mid - sway[k]
            on &= (at >= torn[0, k] - half) & (at < half + torn[1, k])
            L1, L2 = (Lx * (np.abs(at) < half) for Lx in (L1, L2))
        f0, t, c, e, uy, kk = F[y], T[y], C[y], env[y], U[y], K[y]
        L = L1 + L2
        s = (phase[k] + late) % 1
        saw = np.where(s < 0.8, 2.5 * s - 1, 9 - 10 * s)
        gp = gap / (P * Pt[k] * (land + (1 - land) * ramp(0, ease, y - start))) * (1 + jud[k] * where * saw) \
            * edge[drift[k]:drift[k] + w] * (1 + hydro * np.sqrt(np.minimum(L, 40)))
        for gx, ga, gl in grits:            # a grain caught under the blade
            if ga <= k < ga + gl:
                gp[gx - 1:gx + 2] *= 0.4
        z = np.where(on, np.maximum(e + gp, zp - drop), np.inf)
        zp = np.where(on, z, -np.inf)
        cut = np.clip(f0 + t - z, 0, t)
        t1 = t - cut
        if tear:
            rip = rips[y] * ramp(0.15, 0.4, kk) * ramp(0.95, 0.7, kk) * (z - f0 < 2.5 * gp)
            rd, tr, ru = D[y] * rip, t1 * rip, uy.copy()
            F[y] = f0 = f0 - rd
            uy += (V[y] - uy) * rip[:, None]
            D[y] -= rd
            K[y] = kk * (1 - rip)
            t1 = t1 - tr
            b2 = b2 + (ru - b2) * (rd / np.maximum(L2 + rd, 1e-6))[:, None]
            L2 = L2 + rd
        else:
            tr = 0
        space = z - f0 - t1
        reach = np.minimum(0.6 * np.sqrt(L), hang)
        dep = np.minimum(np.maximum(space, 0), rx * L) * ramp(reach, 0.8 * reach + 1e-3, space)
        touch = (space - dep < 0.02) & on
        sw = drag * touch * t1 * (L / (L + 0.5))           # the top of the film, swapped with the bead's paint
        t2 = t1 + dep
        # the bead lays its paints side by side in veins, not mixed; a thin skin lies in patches, not as a veil
        q = L2 / np.maximum(L, 1e-6)
        bs = b1 + (b2 - b1) * ramp(vein[y] - 0.06, vein[y] + 0.06, q)[:, None]
        cov = hide(dep + sw)
        cov = 0.35 * cov + 0.65 * ramp(patch[y] - 0.05, patch[y] + 0.05, cov)
        below = uy + (c - uy) * hide(t1 - sw)[:, None]
        seen = below + (bs - below) * cov[:, None]
        c2 = np.clip(uy + (seen - uy) / np.maximum(hide(t2), 0.05)[:, None], LOW, 0)
        # the film splits where it fills the gap, and breaks up wherever the bead runs thin
        lift = t2 * cell[y] * np.minimum(cells + 0.5 * kk + 0.6 * ramp(0.8, 0.1, L), 1) * touch
        out = dep + sw
        L1 = np.maximum(L1 - out * (1 - q), 0)
        L2 = np.maximum(L2 - out * q, 0)
        give = cut + tr + sw
        b2 = b2 + (c - b2) * (give / np.maximum(L2 + give, 1e-6))[:, None]
        L2 = L2 + give
        b2 = b2 + (c2 - b2) * (lift / np.maximum(L2 + lift, 1e-6))[:, None]
        L2 = L2 + lift
        mv = 0.04 * L2                      # rolling, the bead folds what it took up into its body
        b1 = b1 + (b2 - b1) * (mv / np.maximum(L1 + mv, 1e-6))[:, None]
        L1, L2 = L1 + mv, L2 - mv
        L = L1 + L2
        full = np.minimum(cap / np.maximum(L, 1e-6), 1)
        L1, L2 = L1 * full, L2 * full
        T[y] = t2 - lift
        C[y] = np.where((t2 > 1e-4)[:, None], c2, c)
        last = on & (y + 1 >= stop)
        if last.any():                      # most of the bead goes up with the blade; some is left
            q = (L2 / np.maximum(L1 + L2, 1e-6))[:, None]
            Lend[last], Bend[last], yend[last] = 0.25 * (L1 + L2)[last], (b1 + (b2 - b1) * q)[last], y + 1
            L1[last], L2[last] = 0, 0
        d = k and drift[k] - drift[k - 1] - sway[k] + sway[k - 1]
        if d:                               # the bead goes with the blade as it wanders
            L1, b1, L2, b2 = (np.roll(a, -d, 0) for a in (L1, b1, L2, b2))
        if k % 3 == 0:                      # the bead's paint spreads along the blade, its colours more slowly
            for Lx, bx in ((L1, b1), (L2, b2)):
                if k % 12 == 0:
                    Lb = bx * Lx[:, None]
                    Lb[1:-1] = 0.25 * (Lb[:-2] + Lb[2:]) + 0.5 * Lb[1:-1]
                Lx[1:-1] = 0.25 * (Lx[:-2] + Lx[2:]) + 0.5 * Lx[1:-1]
                if k % 12 == 0:
                    bx[:] = np.where(Lx[:, None] > 1e-6, Lb / np.maximum(Lx, 1e-6)[:, None], bx)
    # what the blade carried is left piled where it lifted, steep on its side and rolling off in front
    if hand:
        Lend *= 1.2 * ramp(-0.2, 1.2, line(w, 300))
    ok = (yend >= 0) & (Lend > 0.02)
    j = np.arange(16)[:, None]
    prof = np.exp(-j / (2 + 0.6 * np.sqrt(Lend))[None])
    prof /= prof.sum(0)
    lump = np.exp(0.25 * line(w, 6))
    for jj in range(16):
        r = yend + jj
        m = ok & (r < h)
        d = f32(Lend[m] * lump[m] * prof[jj, m])
        i = (r[m], x[m])
        seen = U[i] + (C[i] - U[i]) * hide(T[i])[:, None]
        seen += (Bend[m] - seen) * hide(d)[:, None]
        T[i] += d
        C[i] = np.clip(U[i] + (seen - U[i]) / np.maximum(hide(T[i]), 0.05)[:, None], LOW, 0)
    cv.wet = (cv.T > 0.005).astype(np.float32)
    cv.turn(-way)


def passage(cv, g, fam, box, n, angle, length=(400, 900), width=(50, 90), accent=None, odds=0.1, **kw):
    """n loose strokes of one paint about box (x0, y0, x1, y1), heading about `angle`, each loaded with
    the paint and its neighbour streaked in, now and then a touch of another."""
    x0, y0, x1, y1 = box
    c = np.stack([g.uniform(x0, x1, n), g.uniform(y0, y1, n)], 1)
    a = angle + g.normal(0, 0.25, n)
    s = np.linspace(-0.5, 0.5, 8)[None, :, None]
    L, bend = g.uniform(*length, n)[:, None, None], g.normal(0, 0.15, n)[:, None, None]
    d = np.stack([np.cos(a), np.sin(a)], 1)[:, None]
    nrm = np.stack([-np.sin(a), np.cos(a)], 1)[:, None]
    paths = c[:, None] + s * L * d + bend * (s * s - 1 / 12) * L * nrm
    i = g.integers(0, len(fam), n)
    j = np.clip(i + g.choice([-1, 1], n), 0, len(fam) - 1)
    third = fam[g.integers(0, len(fam), n)].copy()
    if accent is not None:
        hit = g.random(n) < odds
        third[hit] = accent[g.integers(0, len(accent), hit.sum())]
    cols = np.stack([fam[i], fam[j], third], 1)
    share = np.stack([g.uniform(0.55, 0.8, n), g.uniform(0.15, 0.3, n), g.uniform(0.03, 0.15, n)], 1)
    kw = {**LOOSE, **kw}
    kw["dry"] = g.uniform(*kw["dry"], n)
    cv.brush(g, list(paths), g.uniform(*width, n), cols, share=share, **kw)


def loaded(g, *spans, n=W, soft=160):
    """A blade `n` px long loaded along its edge: for each stretch (x0, x1, px of paint, paints), the paint put on
    unevenly, one shade here and another there; where two stretches meet, their paints lie side by side
    in streaks rather than mixed."""
    x = np.arange(n, dtype=np.float32)
    line = lambda s: noise.line1d(n, s, g)
    ws = np.stack([ramp(x0 - soft / 2, x0 + soft / 2, x) * ramp(x1 + soft / 2, x1 - soft / 2, x) * amt
                   * np.exp(0.3 * line(60)) for x0, x1, amt, _ in spans])
    L = ws.sum(0)
    which = (special.ndtr(0.5 * line(16) + 0.85 * line(50))[None] > np.cumsum(ws, 0) / np.maximum(L, 1e-6)).sum(0)
    B = np.zeros((n, 3), np.float32)
    for k, (_, _, _, fam) in enumerate(spans):
        i = np.clip((0.5 + 0.35 * line(90)) * len(fam), 0, len(fam) - 1).astype(int)
        B[which == k] = np.log(fam[i])[which == k]
    return L.astype(np.float32), B


def light(cv):
    """Gallery light on the finished canvas: oil paint is matt, with glints on the crests of its ridges."""
    vis, hgt = np.exp(cv.look()), cv.F + cv.T
    img = dabs.shine(vis, hgt, light=LIGHT, relief=0.5, gloss=0.03, reach=(0.7, 1.2))
    return img + 0.06 * impasto.glints(hgt, LIGHT)[..., None]


def paint(seed=1990):
    cv = Canvas(seed)
    g = lambda name: np.random.default_rng([seed, zlib.crc32(name.encode())])

    def lay(name, fam, box, n, angle, **kw):
        passage(cv, g(name), fam, box, n, angle, **kw)

    def band(name, fam, box, n, angle=0.0, **kw):
        """Fresh paint between the pulls: short strokes of one colour across part of the canvas."""
        x0, y0, x1, y1 = box
        passage(cv, g(name), fam, (x0, y0 - 30, x1, y1 + 30), 2 * n, angle, length=(250, 600), width=(22, 45),
                **{"thick": 0.035, "fray": 4.0, "tails": 1.0, **kw})

    # the first sitting: broad passages over the whole canvas, light in the middle and above, red to the
    # left, green to the right, and low down blue and yellow brushed upright in shorter strokes; pulled
    # down through wet, lightly, and then harder from the middle, the hand lifting the blade on a slant,
    # the right end first. It sets, and shows later wherever the blade breaks through
    big, low = dict(length=(600, 1200), width=(70, 120)), dict(length=(400, 800), width=(70, 120))
    lay("yellow", YELLOW, (600, 0, 1700, 1300), 16, -0.2, accent=WHITE, **big)
    lay("white", WHITE, (0, 0, 800, 700), 8, 0.1, **big)
    lay("red", RED, (0, 650, 1100, 2700), 22, 0.15, accent=YELLOW, **big)
    lay("green", GREEN, (1500, 0, 2400, 2700), 24, -0.3, accent=BLUE, **big)
    lay("white2", WHITE, (900, 1500, 1600, 2100), 6, 0.2, **big)
    lay("blue", BLUE, (0, 1950, 900, 2700), 10, 1.45, accent=GREEN, **low)
    lay("blue2", BLUE, (1700, 2150, 2400, 2700), 6, 1.65, **low)
    lay("yellow2", YELLOW, (1000, 2100, 1700, 2700), 7, 1.5, **low)
    pull(cv, g("p1"), -150, H + 30, press=0.7, cells=0.15)
    pull(cv, g("p2"), 900, np.linspace(1620, 1440, W), press=1.2, cells=0.2, cap=15, hand=50)
    cv.dry(g("d1"), 3.0)

    # the second: the blade loaded with red along one half and green along the other, a little yellow
    # between, pulled down the whole canvas into a curtain over the light first layer, which breaks
    # through it
    pull(cv, g("p3"), -150, H + 30, press=0.9, cells=0.45, chatter=0.3, tear=0.4, cap=495,
         load=loaded(g("l3"), (0, 1050, 450, RED), (950, 1400, 255, YELLOW), (1300, 2400, 450, GREEN),
                     (500, 640, 135, WHITE), (2150, 2400, 180, BLUE)))
    cv.dry(g("d2"), 2.0)

    # the third: a band of yellow between the red and the green, a smear of white, and the blade loaded
    # again, the colours shifted along it, and pulled the whole height; then a light pull over the top
    # with a little white and yellow on the blade
    band("yellow4", YELLOW, (1250, 850, 1800, 910), 4, thick=0.05)
    band("white4", WHITE, (300, 1450, 700, 1490), 2)
    pull(cv, g("p5"), -150, H + 30, press=1.0, cells=0.35, tear=0.4, chatter=0.4, cap=200,
         load=loaded(g("l5"), (0, 900, 160, RED), (850, 1100, 110, GREEN), (1150, 1550, 130, YELLOW),
                     (1500, 2400, 160, GREEN), (1900, 2050, 60, WHITE)))
    pull(cv, g("p6"), -150, 1300, press=0.6, cells=0.5,
         load=loaded(g("l6"), (850, 1250, 35, WHITE), (1600, 1850, 35, YELLOW)))
    cv.dry(g("d3"), 1.5)

    # last, a shorter blade pulled by hand across the curtain from the right, twice, low down, with a few
    # dabs of paint along its edge that run out one after another: red over the green into the yellow,
    # and further down and to the left, blue over the red
    across = dict(way=1, hand=50, press=0.9, cells=0.4, tear=0.4, chatter=0.15, cap=140, rate=0.015)
    pull(cv, g("x1"), W - 2430, W - 1300, blade=(1760, 2200), **across, load=loaded(
        g("lx1"), (1780, 1870, 80, RED), (1900, 2050, 135, RED), (1960, 1990, 30, YELLOW), (2105, 2185, 50, RED),
        n=H, soft=60))
    pull(cv, g("x2"), W - 860, W - 140, blade=(2150, 2560), **across, load=loaded(
        g("lx2"), (2170, 2265, 45, BLUE), (2310, 2440, 110, BLUE), (2465, 2545, 30, BLUE), n=H, soft=60))
    return light(cv)


if __name__ == "__main__":
    cv = Canvas(0, (400, 300))
    cv.T[:], cv.C[:], cv.wet[:] = 1.0, np.log(RED[1]), 1
    cv.F[200:203, 50:250] += 2.5                      # a ridge of set paint
    pull(cv, noise.rng(1), 40, 340, cells=0, grit=0, nicks=0)
    assert cv.T[100:190].mean() < 0.3 < cv.T[:30].mean(), "the blade scrapes fresh paint down to a skin"
    assert cv.T[206:220, 100:200].mean() > 3 * cv.T[100:190, 100:200].mean(), "and skips behind a ridge"
    assert cv.T[340:356].max() > 1.5 * cv.T[:30].max(), "and leaves some of its load piled where it lifts"
    cv = Canvas(0, (400, 300))
    cv.T[:], cv.C[:], cv.wet[:] = 1.0, np.log(RED[1]), 1
    pull(cv, noise.rng(3), 40, 340, blade=(100, 220), hand=10, cells=0, grit=0, nicks=0)
    assert np.abs(cv.T[:, :80] - 1).max() < 1e-3 and cv.T[100:300, 140:180].mean() < 0.3, \
        "a shorter blade scrapes its own width"
    cv = Canvas(0, (400, 300))
    load = np.full(300, 30, np.float32), np.tile(np.log(YELLOW[2]), (300, 1))
    pull(cv, noise.rng(2), -10, 410, cells=0, load=load)
    seen = np.exp(cv.look())[50:150].mean((0, 1))
    assert cv.T[50:150].mean() > 0.05 and seen[2] < 0.5 * seen[1], "a loaded blade lays a skin of its paint"
    assert np.isfinite(cv.C).all() and cv.C.max() <= 0, "colours stay paints"
    print("ok")
