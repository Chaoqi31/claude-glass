"""The Temple above the Sea. Oil on canvas, laid on with palette knives.

In the summer of 1953 Nicolas de Staël drove down through Italy to Sicily and saw Agrigento, where the Greek
temples stand along a ridge above the sea. Back in Provence he painted it from memory all that winter: the
land cut down to a few planes, a hill, a temple, a road, the sea and the sky, each a slab of one colour
spread with a palette knife or a broad spatula, and the colour hotter than he had ever used it, orange,
red, magenta, violet and yellow, with one cool band of blue. The paint had grown thinner than in the heavy
pictures of the years before, and the planes lie flat on the surface and still open the space between them.

Here a temple stands on the summit of a dark hill against a vermilion sky, the sea showing on either side
of the hill, and the valley comes down toward us in broad planes of yellow and magenta, with strips of orange
and pink at the foot of the hill and a white road climbing through them. Every plane went on twice. First a
thin lay-in of another colour: red under the sky and the hill, violet under the sea, orange under the yellow,
crimson under the magenta and the pink, pink and yellow under the road. When that had set, the plane itself,
in passes of the knife and the spatula of every width, each a little askew to the last and each with its own
mix of the colour, warmer or cooler, lighter or deeper, so that the plane is built of overlapping slabs.
Where a pass thinned at its edge, or the blade rode high, the lay-in shows; where it stopped and lifted it
left a ridge; where it ran over the wet edge of the plane beside, it dragged that colour along in streaks.
The road went on before the two fields beside it, which were cut against it and ride over its edges. The
temple went on at the end and thickest, a few slabs of white, cream and pale yellow laid across the top of
the hill, with a slab of black beside them.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal, special

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Temple above the Sea"
DATE = "2026"
MEDIUM = "Oil on canvas, laid on with palette knives and a broad spatula"
AFTER = ("Nicolas de Staël, the Sicilian landscapes painted after his journey of 1953: Agrigente (1953–54), "
         "Sicile (1954), Paysage de Sicile (1954)")
ROOM = "Colour Itself"
YEAR = 1954
PLACE = "Agrigento"
REGION = "Europe"
NOTE = ("A temple on a dark hill under a vermilion sky, the sea on either side and a white road climbing to it "
        "through hot planes of yellow and magenta, each plane built of knife passes dragged over a lay-in of "
        "another colour, which shows at their edges.")

H, W = 2080, 2800
M = 160                      # the drawing runs this far past every edge of the canvas
LIGHT = (-0.6, -0.5, 0.62)


def pal(hide, *hexes):
    """A paint and the mixes of it the knife picks up from the palette, in order from the deepest to the
    lightest; and how well it covers at unit thickness."""
    return np.stack([lin(h) for h in hexes]).astype(np.float32), hide


VERMILION = pal(0.9, "#d93e18", "#e2471c", "#e9501c", "#ee5c1e", "#f26a22")
RED = pal(0.85, "#b5181a", "#cc2420", "#d93a2a")
ULTRAMARINE = pal(0.75, "#1a2a80", "#21359a", "#2a3fa6", "#2c4cb2")
CRIMSON = pal(0.85, "#a00e2a", "#b8142e", "#c8243a")
MAGENTA = pal(0.78, "#9c1050", "#a8155c", "#b4175f", "#bd2466", "#c42a6e")
VIOLET = pal(0.72, "#3c145c", "#401760", "#4a1c6e", "#4e1e6c", "#53227a")
DEEP = pal(0.72, "#280c40", "#33124f", "#3a1650", "#40195f")
ORANGE = pal(0.9, "#e67a12", "#ee8018", "#f08a1f", "#f39b33")
YELLOW = pal(0.9, "#f0b010", "#f3ba16", "#f6c21a", "#f7c826", "#f8cf32")
PINK = pal(0.85, "#e88e98", "#f0a0a6", "#f2a89a", "#f4b2b6")
WHITE = pal(0.92, "#ece2cc", "#f4e6d0", "#f5efe2", "#f4e4de", "#f8f3e8")
CREAM = pal(0.92, "#ecd8aa", "#f3e4c0", "#f6ecd0")
PALE = pal(0.9, "#eed68a", "#f5e3a0", "#f8ebb8")
BLACK = pal(0.9, "#140d16", "#1d1420", "#2a1d2e")

# the drawing, x and y in px: the planes of the sky, the sea, the hill and the valley, far to near
SKY = [(-M, -M), (W + M, -M), (W + M, 875), (-M, 885)]
SEA = [(-M, 876), (W + M, 866), (W + M, 1060), (-M, 1100)]
SLOPE = [(940, 806), (1040, 1105), (-M, 1215), (-M, 1040), (520, 930)]           # lit, on the left
FRONT = [(940, 806), (1480, 800), (1580, 1100), (1040, 1105)]                     # the face of the summit
SHADE = [(1480, 800), (1820, 885), (2330, 1010), (2440, 1095), (1580, 1100)]      # in shadow, on the right
FAR = [(1480, 1098), (2440, 1093), (W + M, 1060), (W + M, 1330)]
STRIP = [(-M, 1210), (1300, 1100), (-M, 1430)]
FIELD = [(-M, 1430), (1300, 1100), (1380, 1100), (1600, H + M), (-M, H + M)]
CLOSE = [(1480, 1098), (W + M, 1330), (W + M, 1560), (2450, H + M), (2300, H + M)]
CORNER = [(W + M, 1560), (W + M, H + M), (2450, H + M)]
ROAD = [(1380, 1100), (1480, 1098), (2300, H + M), (1600, H + M)]


def box(corners):
    x0, y0 = np.maximum(np.floor(np.min(corners, 0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(np.max(corners, 0)).astype(int) + 1, (W, H))
    return x0, y0, x1, y1


def plane(poly, r, grow=4.0, shift=5.0, rough=1.0, minus=()):
    """A plane to be filled with the knife: its signed distance (px, positive inside) over the box round it.
    Each plane is drawn on its own, a little out from the shared line (`grow`), every corner moved by the
    hand (`shift`), the edges never ruled: so it rides over its neighbour in one place and stops short of it
    in another. A plane painted round others leaves them out (`minus`). -> (x0, y0, sd)"""
    P = np.asarray(poly, np.float64) + r.normal(0, shift, (len(poly), 2))
    x0, y0, x1, y1 = box(np.vstack([P.min(0) - 30, P.max(0) + 30]))
    im = Image.new("1", (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(im)
    draw.polygon([tuple(p) for p in P - (x0, y0)], fill=1)
    for hole in minus:
        draw.polygon([tuple(p) for p in np.asarray(hole, np.float64) - (x0, y0)], fill=0)
    m = np.asarray(im)
    sd = np.where(m, ndimage.distance_transform_edt(m) - 0.5, 0.5 - ndimage.distance_transform_edt(~m))
    s = sd.shape
    sd += grow + rough * (4.0 * noise.field(s, 300, r) + 2.5 * noise.field(s, 110, r) + 1.1 * noise.field(s, 16, r)
                          + 0.45 * noise.field(s, 3, r))
    return x0, y0, sd.astype(np.float32)


def paint(seed=1954):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint="#efe8da", thread=3.0)
    tooth = ground.tooth
    rgb = ground.color.copy()
    height = tooth * 0.45
    wet = np.zeros((H, W), np.float32)
    B = impasto.bristles(r)
    tone = noise.field((H, W), 260, r) + 0.5 * noise.field((H, W), 60, r)    # where a colour went on deeper or lighter
    fray = noise.field((H, W), 26, r)                                          # where a cut edge broke up
    crumb = 0.7 * noise.field((H, W), 14, r) + 0.3 * noise.field((H, W), 3, r)    # how a dry knife skips
    heave = noise.smoothstep(-0.8, 0.8, noise.field((H, W), 70, r))                # where a cut edge stands up

    def look(a, b):
        o = r.uniform(0, 1, 2) * B.shape
        return ndimage.map_coordinates(B, np.broadcast_arrays(a + o[0], b + o[1]), order=1, mode="grid-wrap")

    def drawn(s, u, thick, lc, share, tn, scrape):
        """The surface the blade leaves and the colour it spreads, in the frame of its pull (s along, u
        across). The steel leaves the paste flat; it flexes a little, so the paste lies higher or lower in
        broad bands drawn out along the pull, and grit on the blade scores a few fine lines. The colour is the
        paint's own, lighter or deeper where the load was (a few broad streaks of it) and now and then a trace
        of another. `tn`: where this colour went on a little deeper or lighter, so that passes side by side
        agree."""
        band = look(s * 0.04, u * 0.025)
        g = look(s * 0.12, u * 0.45)
        grit = scrape * noise.smoothstep(-0.6, 0.6, look(s * 0.03, u * 0.01))     # the grit catches here and there
        body = thick * (1 + 0.15 * band + 0.05 * look(s * 0.1, u * 0.06))
        flat = body - grit * (0.45 * noise.smoothstep(2.0, 2.8, g) - 0.2 * noise.smoothstep(2.2, 3.0, -g))
        t1 = noise.smoothstep(-0.08, 0.08, look(s * 0.03, u * 0.02) - special.ndtri(1 - share[1]))
        t2 = noise.smoothstep(-0.15, 0.15, look(s * 0.25, u * 0.08) - special.ndtri(1 - share[2]))
        mix = lc[0] + (lc[1] - lc[0]) * (0.2 * noise.smoothstep(-1.2, 1.2, tn))[..., None]
        mix += (lc[2] - mix) * t1[..., None]
        return body, flat, mix + (lc[3] - mix) * t2[..., None]

    def slab(p0, p1, w, colours, hide, clip=None, share=(0.85, 0.1, 0.0), skew=(0.0, 0.0), slant=0.12, thick=2.0,
             lift=0.9, lip=0.45, land=0.25, pickup=0.08, drag=60.0, spent=0.0, press=0.0, scrape=1.0, flatten=0.9,
             cut=0.6, ride=1.0, wob=0.04, turn=0.07, comb=0.6):
        """One pass of the knife from p0 to p1, its swath about 2w wide, loaded with `colours` (4, 3): the
        paint, a lighter or deeper streak of it, another, and a trace of a neighbour. The blade lands and lifts
        a little askew, turns in the hand so the swath widens or narrows, and moves in a shallow arc. It
        leaves a flat slab with paint squeezed up along one side, heaped where it touched down and pulled
        into a ridge where it lifted. Wet paint under it is dragged along in streaks; where it bore down
        (`press`) the threads come through in streaks; where it rode high (`ride`), most of all along its
        sides and toward its end, it skipped and left what lies beneath bare; as the load runs out (`spent`)
        it breaks up and catches only on the weave. The paint covers as it is thick: `hide` at unit
        thickness.
        A plane `clip` cuts the pass at its edge, as the side of the knife cuts a plane against the next:
        the paint stops there in a little lip, and breaks up along it here and there."""
        p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
        L = float(np.hypot(*(p1 - p0)))
        if L < 2:
            return
        e = (p1 - p0) / L
        f = np.array([-e[1], e[0]])
        k0, k1 = np.add(skew, r.normal(0, slant, 2))    # the blade seldom lands or lifts quite square to its path
        taper, bow = r.normal(0, turn), r.normal(0, 0.012) * L
        pad = 8 + (0.04 + 3 * wob + abs(taper)) * w + abs(bow)
        e0, e1 = abs(k0) * w + pad, abs(k1) * w + pad
        x0, y0, x1, y1 = box([p0 + e * a + f * b for a in (-e0, L + e1) for b in (-w - pad, w + pad)])
        if clip is not None:
            cx, cy, csd = clip
            x0, y0 = max(x0, cx), max(y0, cy)
            x1, y1 = min(x1, cx + csd.shape[1]), min(y1, cy + csd.shape[0])
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        s = (xx - p0[0]) * e[0] + (yy - p0[1]) * e[1]
        sl = np.clip(s / L, 0, 1)
        u = (xx - p0[0]) * f[0] + (yy - p0[1]) * f[1] - 4 * bow * sl * (1 - sl)
        side = np.where(u < 0, -97.0, 97.0)
        edge = w * (1 + taper * (2 * sl - 1)) + (0.012 * w + 1.0) * look(s * 0.15, side) \
            + wob * w * look(s * 40 / w, side + 500) \
            + 0.5 * look(s * 2.5, side + 211) + 0.25 * look(s * 9.0, side + 99) \
            - 0.05 * w * noise.smoothstep(1.2, 2.2, look(s * 0.3, side + 55))
        da = s - k0 * u - 1.2 * look(u * 0.8, 5.0) - 0.5 * look(u * 6.0, 15.0) \
            - 0.5 * wob * w * look(u * 40 / w, 650.0)                                       # past where it landed
        db = L + k1 * u + (0.015 * w + 1.5) * look(u * 0.5, 9.0) + 0.6 * look(u * 6.0, 25.0) \
            + 0.5 * wob * w * look(u * 40 / w, 700.0) - s                                  # short of the lift
        room = edge - np.abs(u)
        soft = 1 + 2.5 * wet[y0:y1, x0:x1]          # over wet paint its sides smear a little into what is there
        cover = np.clip((room + 0.5) / soft, 0, 1) * np.clip(da + 0.5, 0, 1) * np.clip(db + 0.5, 0, 1)
        tp = tooth[y0:y1, x0:x1]
        cr = crumb[y0:y1, x0:x1]
        inner = np.minimum(np.minimum(room, da + 2), db + 2)              # how far inside the pass's own edge
        cutlip = 0.0
        if clip is not None:
            sd = csd[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
            cover = cover * np.clip(sd + 0.5, 0, 1)
            cuts = noise.smoothstep(0, 3, inner - sd)                      # where the plane's edge is the pass's edge
            inner = np.minimum(inner, sd)
            cutlip = cut * np.exp(-((sd - 2.0 - 0.01 * w) / (1.6 + 0.012 * w)) ** 2) * cuts \
                * (0.25 + 0.75 * heave[y0:y1, x0:x1])
            # along the cut the knife skips now and then, and the paint stops short on the threads
            brk = noise.smoothstep(0.8, 1.8, fray[y0:y1, x0:x1]) * noise.smoothstep(7, 1, sd) * cuts
            cover = cover * (1 - brk * noise.smoothstep(0.35, 0.6, tp + 0.15 * look(s * 0.5, u * 0.5)))
        if not cover.any():
            return
        # where it bore down the paste is pressed thin and the threads come through: in streaks drawn out along
        # the pull, where the blade lay flattest
        thin = np.clip(press * (0.2 + noise.smoothstep(-0.4, 0.8, look(s * 0.06, u * 0.04)
                                                       + 0.5 * look(s * 0.3, u * 0.25))), 0, 1)
        load = 1 - spent * noise.smoothstep(0.55, 1.0, s / L)
        skip = 0.6 * cr + 0.8 * look(s * 0.6, u * 0.1)
        hit = noise.smoothstep(-0.1, 0.1, load - 0.5 + (0.1 + 0.3 * (1 - load)) * skip
                               + 0.35 * (tp - 0.5) * (1 - load))
        # where it rode high it skipped: along its sides and toward its end, in gaps drawn out along the pull
        rim = np.exp(-np.maximum(room, 0) / min(0.08 * w + 3, 0.25 * w))
        gap = noise.smoothstep(2.3, 2.7, look(s * 0.25, u * 0.12) + 0.2 * cr
                               + ride * (2.0 * rim + 0.7 * noise.smoothstep(0.6, 1.0, s / L)))
        a = cover * hit * (1 - thin * noise.smoothstep(0.35, 0.75, tp)) * (1 - gap)       # thread tops scraped bare
        fall = noise.smoothstep(0, 2.5 + 0.01 * w, inner)
        sub = rgb[y0:y1, x0:x1]
        hs, ws = height[y0:y1, x0:x1], wet[y0:y1, x0:x1]
        lc = np.log(np.clip(colours, 1e-4, 1))
        # squeezed out over wet paint of its own colour, the paint sinks back into it and hardly shows
        same = ws * np.exp(-(np.abs(np.log(np.clip(sub, 1e-4, 1)) - lc[0]).mean(-1) / 0.3) ** 2)
        lips = lip * (0.6 + 0.4 * r.choice((-1.0, 1.0)) * np.sign(u)) \
            * np.exp(-((room - 2 - 0.01 * w) / (1.8 + 0.012 * w)) ** 2) \
            * noise.smoothstep(-0.4, 1.6, look(s * 0.25, side + 400)) * (1 - 0.85 * same)
        rr = (4 + 0.05 * w) * np.clip(1 + 0.4 * look(u * 0.7, 300.0), 0.3, None)
        ridge = lift * np.exp(-((db - rr) / rr) ** 2) * noise.smoothstep(-1.5, 1.0, look(u * 1.2, 350.0)) \
            * (1 - 0.75 * same)
        heap = land * np.exp(-np.maximum(da, 0) / (0.35 * w))
        kk = thick * (1 - 0.75 * thin) * load
        body, flat, mix = drawn(s, u, kk * (1 + r.uniform(-0.12, 0.12) * u / w) * (1 + heap), lc, share,
                                tone[y0:y1, x0:x1], scrape)
        # the drag of the steel: fine streaks along the pull, plainer where the paste lay thick
        mix += (comb * (1 + 1.5 * thin) * (0.02 * look(s * 0.08, u * 0.35) + 0.015 * look(s * 0.04, u * 0.08)))[..., None]
        # what it dragged along out of the wet paint it rode over, read on a coarse grid along the pass
        sg, ug = np.arange(-e0, L + e1, 2.0), np.arange(-w - pad, w + pad, 2.0)
        gx = p0[0] + sg[:, None] * e[0] + ug * f[0] - x0
        gy = p0[1] + sg[:, None] * e[1] + ug * f[1] - y0
        moist = ndimage.map_coordinates(ws, [gy, gx], order=1, mode="nearest")
        if pickup > 0 and moist.max() > 0.01:
            old = np.stack([ndimage.map_coordinates(sub[..., c], [gy, gx], order=1, mode="nearest")
                            for c in range(3)], -1)
            k = np.exp(-2.0 / drag)
            num = signal.lfilter([1 - k], [1, -k], np.log(np.clip(old, 1e-4, 1)) * moist[..., None], axis=0)
            den = signal.lfilter([1 - k], [1, -k], moist, axis=0)
            at = [(s + e0) / 2.0, (u + w + pad) / 2.0]
            pick = np.stack([ndimage.map_coordinates(num[..., c] / np.maximum(den, 1e-6), at, order=1,
                                                     mode="nearest") for c in range(3)], -1)
            streak = noise.smoothstep(0.5, 0.8, look(s * 0.08, u * 0.08))     # it comes along in streaks, unmixed
            amt = pickup * ndimage.map_coordinates(den, at, order=1, mode="nearest") * streak
            mix += (pick - mix) * np.clip(amt, 0, 0.85)[..., None]
        # it covers as thick as it went on; where the load ran out it breaks up rather than fading
        A = (a * (1 - (1 - hide) ** (np.maximum(body / np.maximum(load, 0.3), 0.05) / 0.6)))[..., None]
        sub += (np.exp(mix) - sub) * A
        # under the blade the paint beneath is pressed down to the level the knife leaves, so a pass over wet
        # paint as thick as its own leaves no step, only its lips and its ridge; pressed thin, it leaves the weave
        fl = flatten * (1 - 0.85 * thin) * fall
        hs += (hs * (1 - fl) + fall * np.maximum(flat, 0) + kk * (lips + ridge + cutlip) - hs) * a
        ws += (1 - ws) * a

    def pull(p0, p1, w, paint_, clip, near=None, trace=0.1, mixes=0.5, key=None, **kw):
        """One pass with a fresh load: its own mix of the paint, shade `key` of it (or one at random) stirred
        with the next deeper or lighter, so that each pass comes out a little warmer or cooler, lighter or
        deeper than the last; streaked along its length with a shade beside it, now and then with a trace of
        `near` on the blade. Keyword values given as (lo, hi) are drawn afresh for every pass."""
        colours, hide = paint_
        n = len(colours)
        near = colours if near is None else near[0]
        i = r.integers(0, n) if key is None else key
        j = int(np.clip(i + r.choice((-1, 1)), 0, n - 1))
        t = r.uniform(0, mixes)
        base = colours[i] ** (1 - t) * colours[j] ** t
        load = np.stack([base, colours[j], colours[int(np.clip(i + r.integers(-1, 2), 0, n - 1))],
                         near[r.integers(0, len(near))]])
        load *= np.exp(r.normal(0, 0.025) + r.normal(0, 0.012, 3))[None]
        tr = r.uniform(0, 0.02) if r.random() < trace else 0.0
        kws = {key: (r.uniform(*val) if isinstance(val, tuple) else val) for key, val in kw.items()}
        slab(p0, p1, w, load, hide, clip=clip, share=(0.85, r.uniform(0.04, 0.15), tr), **kws)

    def fill(pl, paint_, a, w, length=(600, 1200), tilt=0.03, vary=0.35, meet=(-0.06, 0.25), over=(0.1, 0.4),
             stray=0.12, extra=0.2, drift=0.9, **kw):
        """Fill a plane with the knife: lanes side by side along direction a, each of its own width, laid edge
        to edge, one riding over the last a little and the next stopping short of it, so that a sliver of what
        lies beneath shows between (`meet`: the overlap as a share of the two widths; below 0, a gap). A lane
        goes on in one pass or a few end to end, each a little askew, so the lanes close and open along their
        length; they run on past the plane's edge, which cuts them, though now and then (`stray`) one rides a
        little over it. Then `extra` as many passes again, here and there over the lanes. The mix on the knife
        drifts across the plane, deeper in one stretch and lighter in another, so neighbouring passes are
        alike and the plane falls into a few broad slabs; the more so the larger `drift`."""
        cx, cy, sd = pl
        e, f = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        yy, xx = np.nonzero(sd[::4, ::4] > 0)
        if not len(yy):
            return
        pts = np.stack([xx * 4.0 + cx, yy * 4.0 + cy], 1)
        q, u = pts @ f, pts @ e
        width = lambda: w * np.exp(np.clip(r.normal(0, vary), -1.2 * vary, 2 * vary))
        passes, v, prev = [], q.min(), 0.0
        while v < q.max():
            ww = width()
            v += (prev + ww) * (1 - r.uniform(*meet)) if prev else ww * r.uniform(0.3, 0.9)
            prev = ww
            band = np.abs(q - v) < ww
            if not band.any():
                continue
            u0, u1 = u[band].min() - r.uniform(*over) * ww, u[band].max() + r.uniform(*over) * ww
            m = max(1, int(round((u1 - u0) / r.uniform(*length))))
            # the joints between the passes of one lane fall in other places than those of the next
            cut = np.linspace(u0, u1, m + 1) + np.r_[0, r.uniform(-0.3, 0.3, m - 1) * (u1 - u0) / m, 0]
            passes += [(cut[j] - (r.uniform(0.1, 0.4) * ww if j else 0), cut[j + 1], v + r.normal(0, 0.05 * ww), ww)
                       for j in range(m)]
        for _ in range(int(extra * len(passes))):
            k = r.integers(len(pts))
            L = r.uniform(*length) * r.uniform(0.4, 1.0)
            s0 = u[k] - L * r.uniform(0.2, 0.8)
            passes.append((s0, s0 + L, q[k], width()))
        zone = noise.field((H // 16 + 2, W // 16 + 2), 40, r)
        n = len(paint_[0])
        for s0, s1, v, ww in passes:
            mx, my = e * (s0 + s1) / 2 + f * v
            z = zone[int(np.clip(my / 16, 0, H // 16 + 1)), int(np.clip(mx / 16, 0, W // 16 + 1))]
            key = int(np.clip(np.round(drift * z + (n - 1) / 2), 0, n - 1))
            t = r.normal(0, tilt)
            q0 = e * s0 + f * v
            ends = (q0, q0 + (e * np.cos(t) + f * np.sin(t)) * (s1 - s0))
            clip = (cx, cy, sd + r.uniform(8, 24)) if r.random() < stray else pl
            pull(*(ends[::-1] if r.random() < 0.4 else ends), ww, paint_, clip, key=key, **kw)

    def clean():
        """A sitting ends: the paint on the canvas sets, and the next goes on over it without dragging it."""
        wet[:] = 0

    # the lay-in: every plane thin in another colour, in a few broad passes
    hill = (SLOPE, FRONT, SHADE)
    lay = dict(thick=1.0, press=0.0, spent=0.0, lift=0.4, lip=0.15, land=0.1, pickup=0.0, ride=0.0,
               meet=(0.15, 0.35), tilt=0.01, extra=0.0, mixes=0.3, vary=0.3)
    fill(plane(SKY, r, grow=8, shift=3), RED, 0.0, 230, length=(900, 1600), **lay)
    fill(plane(SEA, r, grow=8, shift=3), VIOLET, 0.0, 70, length=(600, 1200), **lay)
    for p in hill:
        fill(plane(p, r, grow=10, shift=3), RED, -0.1, 140, length=(500, 900), **lay)
    under = dict(grow=8, shift=6)
    fill(plane(FAR, r, **under), CRIMSON, -0.03, 90, length=(500, 900), **lay)
    fill(plane(STRIP, r, **under), RED, -0.075, 90, length=(500, 900), **lay)
    fill(plane(FIELD, r, **under), ORANGE, -0.1, 180, length=(600, 1100), **lay)
    fill(plane(CLOSE, r, **under), CRIMSON, 0.16, 160, length=(600, 1100), **lay)
    fill(plane(CORNER, r, **under), MAGENTA, 0.6, 110, length=(400, 800), **lay)
    road = plane(ROAD, r, **under)
    fill(road, PINK, 1.1, 110, length=(500, 1000), **lay)
    fill(road, YELLOW, 1.1, 90, length=(400, 900), **{**lay, "meet": (-1.2, -0.6)})
    clean()

    # the sky, painted round the hill, in passes of the spatula of every width, some pressed so thin that the
    # red comes through; then the sea, round the hill too
    top = dict(press=(0.0, 0.55), spent=(0.0, 0.4), pickup=0.35, drag=140.0, slant=0.15, comb=(0.2, 1.0),
               meet=(0.0, 0.25))
    fill(plane(SKY, r, shift=3, minus=hill), VERMILION, 0.0, 180, length=(1400, 2800), tilt=0.02, vary=0.35,
         thick=1.4, lift=0.6, land=0.15, lip=0.25, scrape=(0.3, 1.0), extra=0.15, drift=1.1,
         **{**top, "slant": 0.05})
    fill(plane(SEA, r, shift=3, minus=hill), ULTRAMARINE, -0.004, 55, length=(400, 1000), tilt=0.012, thick=2.0,
         **top)
    clean()

    # the hill: the shadowed slope and the lit one pulled along their fall, the face of the summit across it
    slope = dict(thick=2.4, lip=0.4, **top)
    fill(plane(SHADE, r), DEEP, 0.24, 80, length=(300, 700), near=VIOLET, **slope)
    fill(plane(FRONT, r), VIOLET, 0.03, 65, length=(250, 500), near=DEEP, **slope)
    fill(plane(SLOPE, r), MAGENTA, -0.2, 105, length=(500, 1000), near=RED, vary=0.3, **slope)
    clean()

    # the valley, far to near, the planes wet beside one another; the road before the two fields, which are cut
    # against it and ride over its edges here and there, dragging the white
    vale = dict(grow=0.5, shift=5)
    flat = dict(thick=2.2, lip=0.3, **top)
    fill(plane(FAR, r, **vale), PINK, -0.03, 70, length=(400, 900), **{**flat, "meet": (0.1, 0.3)})
    fill(plane(STRIP, r, **vale), ORANGE, -0.075, 70, length=(400, 900), **flat)
    fill(plane(ROAD, r, grow=6, shift=4), WHITE, 1.1, 100, length=(500, 1100), tilt=0.04, near=PINK,
         **{**flat, "meet": (0.15, 0.35)})
    fill(plane(FIELD, r, **vale), YELLOW, -0.1, 200, length=(900, 1800), tilt=0.04, near=ORANGE,
         stray=0.25, **{**flat, "meet": (0.05, 0.25)})
    fill(plane(CLOSE, r, **vale), MAGENTA, 0.16, 170, length=(900, 1800), tilt=0.04, near=CRIMSON,
         stray=0.25, drift=1.4, ride=1.4, **{**flat, "meet": (-0.05, 0.2), "press": (0.0, 0.35)})
    fill(plane(CORNER, r, **vale), VIOLET, 0.6, 100, length=(400, 800), **flat)
    clean()

    # the temple, last and thickest: slabs of white, cream and pale yellow laid across the top of the hill, a
    # dark break where two did not meet, and a slab of black beside them
    th = dict(thick=3.4, lift=1.2, lip=0.6, land=0.35, press=(0, 0.15), spent=0.0, pickup=0.15, drag=40.0,
              cut=0.0, slant=0.12, ride=0.05, mixes=0.4, wob=0.07, turn=0.18, comb=0.8, scrape=2.0)
    pull((906, 822), (895, 700), 18, BLACK, None, key=1, **{**th, "ride": 0.15, "wob": 0.09})
    pull((958, 780), (1290, 768), 34, WHITE, None, key=2, **{**th, "skew": [0.15, -0.05]})
    pull((1293, 808), (1299, 752), 6, DEEP, None, key=1, **{**th, "thick": 2.6, "ride": 0.8})
    pull((1478, 774), (1302, 766), 27, CREAM, None, key=1, **{**th, "skew": [0.05, 0.15]})
    pull((1060, 742), (1262, 733), 11, PALE, None, key=1, **{**th, "skew": [-0.2, 0.1]})

    # pinholes where air in the paste broke at the surface
    height -= 0.35 * noise.smoothstep(2.7, 3.3, noise.field((H, W), 2.2, r)) * noise.smoothstep(0.6, 1.5, height)
    height = ndimage.gaussian_filter(height, 0.6)      # the lamp sees the surface a hair softer than the knife left it
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0.03, reach=(0.72, 1.22))
    return img + 0.05 * impasto.glints(height, LIGHT)[..., None]
