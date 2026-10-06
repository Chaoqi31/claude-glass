"""Clear Water at Jávea. Oil on canvas, painted fast and out of doors.

Sorolla spent the summers of 1900, 1905 and 1906 at Jávea, south of Valencia, where the
limestone comes down to the sea in blocks the sun has baked ochre and rose, and the water
over the rocks and the sand is so clear that the bottom shows through it from the cliff.
He painted standing in the glare, on a light ground, with big brushes loaded with paint,
and he painted the water the way the eye takes it in: first the bottom, the stones and the
pale sand with the shadows of the waves lying violet on it, then the water over them in
long strokes that follow the swell and let them show, turquoise over the sand, emerald over
the stones, ultramarine where the bottom drops away, then the sky caught on the surface in
dry strokes dragged across, and last the sun breaking on it in thick touches of white. A
small white boat rides at anchor over the sand, and its shadow lies green on the bottom
beside it, which is how one knows the water is there at all.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Clear Water at Jávea"
DATE = "2026"
MEDIUM = ("Oil on canvas, alla prima on a light ground: the bottom laid in first, the water over it in long "
          "strokes, the sun on the surface in thick white")
AFTER = "Joaquín Sorolla, the seascapes at Jávea and Valencia, 1900–1910"
ROOM = "Open Air"
YEAR = 1905
PLACE = "Jávea, Valencia"
REGION = "Europe"
NOTE = ("The sea at noon from the rocks, clear enough to show the sand and the stones beneath it. "
        "The bottom went on first, the water over it in strokes that let it show, and the sun last, in thick white.")

H, W = 2100, 2800
LIGHT = (-0.6, -0.5, 0.62)                          # the gallery lamp, raking the relief of the paint
SUN = np.array([-0.3, -0.55, 0.78]) / np.linalg.norm([-0.3, -0.55, 0.78])   # noon: high, ahead, a little left
ZS = 300.0                                           # px of rise to one unit of height, for the slopes the sun sees
K = np.array([2.8, 0.62, 0.42], np.float32)          # how hard the water drinks red, green and blue, per unit of depth
BOAT = (1330, 1390, -0.42, 200, 70)                  # the boat: where, which way its bow points, length, beam
RUN = 0.55                                           # the way the brush mostly ran over the rocks, down to the right


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


SAND = lin("#e8dec0")
WEED = lin("#4a5236")                                              # sea grass, olive and brown
ROCK = pal("#efe3cb", "#e3c08a", "#e3ad98", "#d2c4cc")             # limestone: cream, ochre, rose, grey lilac
MOSS = pal("#c4a462", "#a99b5f", "#b98f78")                        # the sunken stones, gold, olive, rose-brown
RUST = pal("#b65f39", "#c97b49", "#a24a2e")
HALF = lin("#c4aebf")                                              # limestone turned from the sun: lilac grey
SHADE = pal("#6e5f98", "#806ea2", "#5e538b")                       # rock in shadow, lit by the sky
DEEP = lin("#244a92")                                              # the sea where no bottom shows
SKY = pal("#b4cce4", "#c8d8ea", "#cfcbe6", "#a3c0dc", "#dcdfec")   # the sky in the water, cerulean to lilac
WHITE = pal("#fffdf5", "#fff8e8", "#fbfcfa", "#f3f5fb")
GOLD = pal("#f7e3a0", "#f1d07e", "#fbeec5")
VIOLET = pal("#47458c", "#575396", "#6a63a6", "#7c72b2")
EMERALD = pal("#1b6d62", "#227f70", "#2c937d")
CYAN = pal("#4fb3b4", "#6cc3be", "#8dd2c8")
SEAM = np.float32([1.5, 1.42, 1.05])                              # sunlight gathered on the bottom, warm
DUSK = lin("#8a7ccc")                                             # and the violet of the shadows there
BOARDS = pal("#8c79a8", "#d7b2b4", "#eadbcc", "#d9d4e8", "#53609c")   # the boat: shade, rose, cream, lilac, flank
COBALT = pal("#2b5aa0", "#3a6bae", "#4d7fbb", "#3f86b9")


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def up(a, s, order=1):
    """A field worked out on a coarse grid, brought up to the canvas."""
    return ndimage.zoom(a, (s, s) + (1,) * (a.ndim - 2), order=order)[:H, :W]


def stay(paths, key):
    """A stroke stops where the thing it describes ends: each path is cut back to where `key`
    (H,W labels) first differs from its value at the start, and laid out again over as many points."""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, 4 * n)
    i0 = np.minimum(t.astype(int), n - 2)
    fine = paths[:, i0] * (1 - (t - i0))[None, :, None] + paths[:, i0 + 1] * (t - i0)[None, :, None]
    lab = key[np.clip(fine[..., 1].astype(int), 0, H - 1), np.clip(fine[..., 0].astype(int), 0, W - 1)]
    out = lab != lab[:, :1]
    end = np.maximum(t[np.where(out.any(1), np.argmax(out, 1), 4 * n - 1)], 0.6)
    s = np.linspace(0, 1, n)[None, :] * end[:, None]
    j0 = np.minimum(s.astype(int), n - 2)
    g = (s - j0)[..., None]
    return np.take_along_axis(paths, j0[..., None], 1) * (1 - g) + np.take_along_axis(paths, (j0 + 1)[..., None], 1) * g


def level(f):
    """The direction along the level lines of a field."""
    gy, gx = np.gradient(f)
    return np.arctan2(gx, -gy).astype(np.float32)


def floor(r):
    """The ground under the light as one height: 0 the level of the sea, rock in the air above
    it, the bottom below it seen through the water. Along the shore lie blocks of limestone,
    squat and angular, their tops tilted and weathered round, their sides steep: big ones
    heaped on the land, smaller ones at the water's edge; at their foot boulders the sea has
    worn round lie sunk in the sand, thinning out over the bed of sand, and beyond it the
    bottom slopes away into deep water.
    -> height, which block owns each point (-1 none), how many blocks, px out from the shore line"""
    s = 2
    yy, xx = np.mgrid[0:H:s, 0:W:s].astype(np.float32)
    y = np.arange(0, H, s, dtype=np.float32)
    xc = 150 + 560 * (y / H) ** 1.5 + 190 * np.exp(-((y - 1250) / 150) ** 2) + 40 * noise.line1d(len(y), 220, r)
    u = xx - xc[:, None]

    def bed(v):
        return 0.16 * noise.smoothstep(-60, 260, v) + 0.26 * noise.smoothstep(250, 1000, v) \
            + 0.6 * noise.smoothstep(900, 1500, v) + 1.1 * noise.smoothstep(1400, 2100, v) + 0.8 * noise.smoothstep(2000, 2700, v)

    uw = u + 200 * noise.field(u.shape, 360, r) + 50 * noise.field(u.shape, 90, r)
    z = np.where(uw < 0, 0.04 - uw / 2600, -bed(uw))
    z = (z + 0.03 * noise.fbm(u.shape, 160, r, octaves=2)).astype(np.float32)
    owner = np.full(u.shape, -1, np.int32)
    count = 0
    for lo, hi, gap, R0, sunk in ((30, 750, 125, 78, True), (-1200, -120, 250, 200, False), (-280, 200, 190, 120, False)):
        P = dabs.scatter((H, W), gap, r)
        uc = u[np.clip(P[:, 1] // s, 0, len(y) - 1).astype(int), np.clip(P[:, 0] // s, 0, u.shape[1] - 1).astype(int)]
        keep = (uc > lo) & (uc < hi) & (r.random(len(P)) < (noise.smoothstep(hi, lo, uc) ** 0.6 if sunk else 1))
        for (cx, cy), c in zip(P[keep], uc[keep]):
            R = R0 * (0.7 + 0.6 * cy / H) * np.exp(r.normal(0, 0.22))
            if sunk:
                top = min(-0.03, -bed(c) + r.uniform(0.06, 0.2))
            else:
                top = (0.5 - c / 1500 if c < -120 else 0.15 - c / 650) + r.normal(0, 0.12)
            m = r.integers(8, 11) if sunk else r.integers(5, 8)
            th = r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + r.normal(0, 0.25, m)
            rk = R * np.exp(r.normal(0, 0.16, m))
            x0, x1 = int(max(0, cx - 1.6 * R)) // s, int(min(W, cx + 1.6 * R)) // s + 1
            y0, y1 = int(max(0, cy - 1.6 * R)) // s, int(min(H, cy + 1.6 * R)) // s + 1
            if x1 <= x0 or y1 <= y0:
                continue
            dx, dy = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
            rho = np.max([(dx * np.cos(a) + dy * np.sin(a)) / k for a, k in zip(th, rk)], 0)
            tilt = r.uniform(0, 2 * np.pi)
            lean, round_ = ((0.04, 0.18), (0.1, 0.3)) if sunk else ((0.02, 0.09), (0.04, 0.14))
            zi = top + r.uniform(*lean) * (np.cos(tilt) * dx + np.sin(tilt) * dy) / R \
                - 3.0 * np.maximum(rho - 1, 0) - r.uniform(*round_) * np.minimum(rho, 1) ** 2   # the top weathered round
            win = zi > z[y0:y1, x0:x1]
            z[y0:y1, x0:x1] = np.where(win, zi, z[y0:y1, x0:x1])
            owner[y0:y1, x0:x1][win] = count
            count += 1
    z += 0.012 * noise.fbm(u.shape, 50, r, octaves=2)
    return up(ndimage.gaussian_filter(z, 0.8), s), up(owner, s, 0), count, up(u, s)


def sunlight(z):
    """How the sun falls on the ground: how squarely each slope faces it (-1..1), and where the
    blocks shade what lies beside them, down and to the right of each. -> (lit, shadow 0..1)"""
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 1.5) * ZS)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lit = (n @ SUN.astype(np.float32)).astype(np.float32)
    s = 4
    zq = z[::s, ::s]
    horiz = np.hypot(SUN[0], SUN[1])
    d = SUN[:2] / horiz
    out = np.zeros_like(zq)
    for t in range(8, 240, 8):
        ahead = ndimage.shift(zq, (-d[1] * t / s, -d[0] * t / s), order=1, mode="nearest")
        out = np.maximum(out, noise.smoothstep(0.0, 0.04, ahead - zq - t * SUN[2] / horiz / ZS))
    return lit, up(ndimage.gaussian_filter(out, 1.0), s)


def swell(r):
    """The slow swell coming into the cove: long crests running a little downhill to the right,
    flatter as they go back, closer together, bending gently as they come. -> (direction of the
    crests, phase with the crests at 0, and the phase of the ripples riding on them)"""
    s = 2
    yy, xx = np.mgrid[0:H:s, 0:W:s].astype(np.float32)
    a = 0.06 + 0.24 * yy / H
    lam = 110 + 120 * yy / H
    psi = yy * np.cos(a) - xx * np.sin(a) + 35 * noise.field(yy.shape, 650, r) + 8 * noise.field(yy.shape, 200, r)
    phase = 2 * np.pi * psi / lam
    ripple = 2 * np.pi * (psi + 18 * noise.field(yy.shape, 60, r)) / (0.3 * lam)
    return up(level(ndimage.gaussian_filter(psi, 6)), s), up(phase, s), up(ripple, s)


def hull(cx, cy, ang, length, beam):
    """The boat seen from above, a pointed oval with a blunt stern. -> where it lies (H,W), and its
    half-beam at any point along it (-1 stern .. 1 bow)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    e, n = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])

    def half(t):
        return 0.5 * beam * np.sqrt(np.clip(1 - np.clip(t, -0.93, 1) ** 2, 0, 1)) ** 0.8

    t = ((xx - cx) * e[0] + (yy - cy) * e[1]) / (length / 2)
    v = (xx - cx) * n[0] + (yy - cy) * n[1]
    return (np.abs(t) <= 1) & (np.abs(v) <= half(t)), half


def loads(base, r, vary=0.04, accent=None, odds=0.15, swing=0.07):
    """The brush for each stroke: the colour wanted, a second a little lighter or darker and off
    in hue streaked into it, and a third, now and then an accent picked up from elsewhere."""
    N = len(base)
    a = base * np.exp(r.normal(0, vary, (N, 3)))
    b = base * np.exp(r.normal(0, 1.6 * vary, (N, 3)) + r.normal(0, swing, (N, 1)))
    c = base * np.exp(r.normal(0, 2 * vary, (N, 3)) + r.normal(0, 1.6 * swing, (N, 1)))
    if accent is not None:
        hit = r.random(N) < odds
        c[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.04, 0.2, N)], 1)
    return np.clip(np.stack([a, b, c], 1), 0, 1).astype(np.float32), share


def paint(seed=1905):
    r = noise.rng(seed)
    near = (np.arange(H, dtype=np.float32) / H)[:, None]              # 0 far .. 1 near
    xs = np.arange(W, dtype=np.float32)[None, :]
    ground = canvas.duck((H, W), seed, tint="#efe8d6", thread=3.0)
    z, owner, count, u = floor(r)
    lit, cast = sunlight(z)
    crest, phase, ripple = swell(r)
    land = z > 0
    sea = ~land
    shore = ndimage.distance_transform_edt(sea).astype(np.float32)    # px out from the nearest rock
    depth = np.maximum(-z, 0)
    sparing = 0.5 * noise.smoothstep(-0.8, -1.6, noise.field((H, W), 380, r)) * noise.smoothstep(300, 0, shore)
    bx, by, bang, blen, bbeam = BOAT
    fall = -SUN[:2] / np.hypot(*SUN[:2])                              # the way the shadows fall
    off = depth[by, bx] * ZS * np.hypot(*SUN[:2]) / SUN[2]           # how far the boat's lies from it, on the sand
    aboard, half = hull(*BOAT)
    under = ndimage.gaussian_filter(hull(bx + off * fall[0], by + off * fall[1], bang, blen, bbeam)[0].astype(np.float32), 5)

    # the stone, as the sun bakes it: cream on the tops, ochre and rose where it has weathered,
    # lilac in the shade; under the water it is furred gold and olive, and the sand lies between
    kind = r.integers(0, 3, count + 1)[owner]
    warm = noise.smoothstep(-0.2, 1.2, noise.field((H, W), 480, r))[..., None], noise.smoothstep(0.2, 1.4, noise.field((H, W), 340, r))[..., None]
    limestone = ROCK[0] + (ROCK[1] - ROCK[0]) * warm[0] + (ROCK[2] - ROCK[0]) * warm[1] * (1 - warm[0])
    stone = np.where(land[..., None], limestone, MOSS[kind]) * (1 + 0.04 * noise.fbm((H, W), 40, r, octaves=3))[..., None]
    rust = noise.smoothstep(1.5, 2.1, noise.field((H, W), 90, r) + 0.6 * noise.stretched((H, W), 8, 60, r)) * land
    stone += (RUST[0] - stone) * rust[..., None]
    sun = ndimage.gaussian_filter(noise.smoothstep(-0.05, 0.4, lit) * (1 - 0.85 * cast), 3)
    turn = noise.smoothstep(0.3, 0.92, lit)[..., None]                # facets turned from the sun go lilac and rose
    rock = stone * (0.8 + 0.25 * turn) + (HALF - stone * 0.8) * (0.6 * (1 - turn))
    rock += (SHADE[0] * (0.75 + 0.35 * stone / stone.max()) - rock) * (0.8 * (1 - sun))[..., None]
    rock *= (1 - 0.3 * noise.smoothstep(0.05, 0.0, z) * land)[..., None]     # darker where the sea wets it
    plane = (RUN + r.normal(0, 0.5, count + 1)[owner] + 0.1 * noise.field((H, W), 200, r)).astype(np.float32)
    light = sun > 0.5                                                 # in the sun or out of it
    forms = np.where(land, 1 + light, -10 - (owner >= 0) - 2 * (cast > 0.5) - 4 * (under > 0.5) - 8 * light)
    rock = ndimage.gaussian_filter(rock, (2, 2, 0))

    # the bottom: sand, sunken stones, sea grass in dark patches, the shadows of the rocks and the
    # boat across it; over all of them a net of bright seams where the swell gathers the sun, and
    # the shadows of the waves violet between; all of it seen through the water, warm where shallow
    net = noise.stretched((W, H), 38, 130, r).T[:H, :W] + 0.35 * noise.field((H, W), 26, r)
    seam = noise.smoothstep(0.3, 0.03, np.abs(net))
    shadow = 0.5 * noise.smoothstep(-0.5, -1.2, net)
    weed = noise.smoothstep(0.9, 1.3, noise.field((H, W), 230, r) + 0.6 * noise.field((H, W), 60, r)) \
        * noise.smoothstep(420, 650, u) * noise.smoothstep(1500, 1150, u)
    mass = SAND * (1 + 0.05 * noise.fbm((H, W), 60, r, octaves=3))[..., None]
    mass += (rock - mass) * (owner >= 0)[..., None]
    mass += (WEED - mass) * weed[..., None]
    mass += (mass * DUSK - mass) * np.maximum(0.8 * cast, under)[..., None]
    path = depth * (1.3 + 1.7 * (1 - near))                          # the eye looks through more water further off
    clear = np.exp(-1.2 * path)                                       # how much of the bottom shows

    def through(c, y, x):
        """A colour on the bottom as the eye sees it through the water above it."""
        p = path[y, x, None]
        return c * np.exp(-p * K) + DEEP * (1 - np.exp(-1.2 * p))

    bed = mass * (1 + (0.5 * seam)[..., None] * SEAM - (0.25 * shadow)[..., None])
    bed += (bed * DUSK - bed) * shadow[..., None]
    seen = through(bed, slice(None), slice(None))
    body = ndimage.gaussian_filter(seen, (50, 50, 0))
    sheen = (0.05 + 0.4 * (1 - near) ** 2)[:, 0]
    glare = np.exp(-((xs - 1800) / 1050.0) ** 2) * np.exp(-((near * H - 330) / 520.0) ** 2)

    rgb, height = ground.color.copy(), ground.tooth * 0.25
    wet = np.ones((H, W), np.float32)

    def lay(paths, w, load, hide=1.0, **kw):
        """Lay strokes in order; a veil that varies from stroke to stroke goes on in steps of hiding."""
        if np.ndim(hide) == 0:
            impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, hide=hide, **kw)
            return
        step = np.round(np.asarray(hide) * 10) / 10
        w = np.broadcast_to(np.asarray(w, np.float64), (len(paths),))
        each = {k for k, v in kw.items() if np.ndim(v) and len(v) == len(paths)}     # the per-stroke settings
        for h in np.unique(step):
            sel = step == h
            impasto.lay(rgb, height, paths[sel], w[sel], load[0][sel], r, share=load[1][sel], wet=wet, hide=h,
                        **{k: (np.asarray(v)[sel] if k in each else v) for k, v in kw.items()})

    def strew(spacing, odds, far=0.62):
        """Where the strokes of a passage start: a shaken honeycomb, closer as it goes back (the
        strokes there are smaller), kept by `odds` (H,W), thinned where the brush went sparing."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), far * spacing, r) - spacing
        y, x = at(P)
        keep = (far / (0.62 + 0.5 * near[y, 0])) ** 2 * odds[y, x] * (1 - 0.6 * sparing[y, x])
        P = P[r.random(len(P)) < keep]
        return P, len(P), 0.62 + 0.5 * near[at(P)[0], 0], at(P)

    def go(field, P, L, n, bend=0.0, tilt=0.0, key=None):
        paths = impasto.follow(field, P, L, n, bend, tilt)
        if key is not None:
            paths = stay(paths, key)
        back = r.random(len(P)) < 0.5
        paths[back] = paths[back, ::-1]
        return paths

    def wobble(y, x, g, amp=16):
        """Where the eye finds the bottom under each seed, pushed about by the waves above it."""
        return np.clip((y + amp * g * np.sin(phase[y, x])).astype(int), 0, H - 1), x

    # the first sitting, thin and wet: everything laid in with the big brush
    P, n, g, (y, x) = strew(165, np.ones((H, W)))
    on = land[y, x]
    col = np.where(on[:, None], rock[y, x], seen[y, x] * (1 - sheen[y, None]) + SKY[0] * sheen[y, None])
    lay(go(np.where(land, plane, crest), P, np.where(on, r.uniform(140, 280, n), r.uniform(280, 600, n)) * g, 10,
           r.normal(0, 0.0006, n), key=land), r.uniform(24, 40, n) * g, loads(col, r, 0.04),
        thick=0.015, grooves=0.1, lips=0.02, land=0.1, lift=0.05, pickup=0.7, merge=8, spent=0.85,
        hide=np.where(on, 0.9, 0.75), fray=4, taper=0.5, tails=0.5, ends=(0.6, 0.5))
    height -= 0.85 * (height - 0.25 * ground.tooth)
    wet[:] = 0

    # the second: the bottom where it shows. Its masses first, in broken strokes laid level as the
    # ripples break them, each stopping where its stone or its shadow ends; then over them the
    # seams of light and the violet shadows of the waves, long and thin, crossing stone and sand alike
    shows = noise.smoothstep(0.06, 0.3, clear) * sea
    P, n, g, (y, x) = strew(44, shows)
    lay(go(crest, P, r.uniform(60, 220, n) * g, 10, r.normal(0, 0.0015, n), key=forms), r.uniform(9, 17, n) * g,
        loads(through(mass[wobble(y, x, g)], y, x), r, 0.05, np.concatenate([VIOLET[2:], EMERALD[1:]]), 0.1),
        thick=0.12, spent=0.6, pickup=0.3, merge=4, grooves=0.25, dry=r.uniform(0.0, 0.3, n), fray=2.5, taper=0.4,
        ends=(0.5, 0.3))
    for odds, colour, wide in ((seam, SEAM * 1.25, (4, 9)), (shadow, 1 + 0.6 * (DUSK - 1), (7, 14))):
        P, n, g, (y, x) = strew(40, noise.smoothstep(0.35, 0.8, odds / odds.max()) * shows)
        lay(go(crest, P, r.uniform(90, 280, n) * g, 10, r.normal(0, 0.0012, n)), r.uniform(*wide, n) * g,
            loads(through(mass[wobble(y, x, g)] * colour, y, x), r, 0.04),
            thick=0.1, spent=0.7, pickup=0.35, merge=3, grooves=0.2, dry=r.uniform(0.1, 0.4, n), fray=2.0,
            taper=0.5, ends=(0.5, 0.3), hide=0.85)

    # then the water over it, wet into wet, in long strokes along the swell: a broken veil where it
    # is shallow, so the bottom shows between and through, fuller as it deepens
    P, n, g, (y, x) = strew(66, (1 - 0.5 * clear) * sea)
    c = clear[y, x]
    tone = body[y, x] * np.exp(r.normal(0, 0.08, (n, 1)))
    pick = r.random(n)
    for sel, colours in ((pick < 0.06, VIOLET), ((pick > 0.88) & (c > 0.15), np.concatenate([EMERALD, CYAN])),
                         ((pick > 0.75) & (c < 0.15), COBALT)):
        tone[sel] = colours[r.integers(0, len(colours), sel.sum())]
    lay(go(crest, P, r.uniform(180, 460, n) * g, 12, r.normal(0, 0.0008, n), key=land), r.uniform(11, 20, n) * g,
        loads(tone, r, 0.04, np.concatenate([CYAN, VIOLET[1:], COBALT]), 0.18),
        thick=0.1, spent=0.75, pickup=0.5, merge=5, grooves=0.25, dry=0.1 + 0.6 * c, fray=2.5, taper=0.5,
        ends=(0.5, 0.3), hide=np.clip(1.0 - 0.85 * c, 0.25, 0.95))

    # the boat's shadow on the sand, laid over the veil in a few level strokes, green as he saw it
    P, n, g, (y, x) = strew(16, (under > 0.45).astype(np.float32), far=1)
    tone = seen[y, x] + (EMERALD[0] - seen[y, x]) * r.uniform(0.3, 0.6, (n, 1))
    lay(go(crest, P, r.uniform(30, 90, n), 6, r.normal(0, 0.004, n), key=under > 0.45), r.uniform(7, 12, n),
        loads(tone, r, 0.04), thick=0.12, pickup=0.4, merge=6, grooves=0.2, ends=(0.6, 0.4), taper=0.5, hide=0.85)

    # the rocks: the violet of their shadows first, along the foot of the rocks where the sea
    # meets them, then plane by plane in broad loaded strokes, each block its own way and each
    # stroke stopping where its light ends
    inland = ndimage.distance_transform_edt(land).astype(np.float32)
    coastwise = np.where(inland < 140, level(ndimage.gaussian_filter(inland - shore, 10)), plane)
    for odds, spacing, size, field in ((land & ~light, 28, 12, coastwise), (land, 44, 20, plane)):
        P, n, g, (y, x) = strew(spacing, odds.astype(np.float32))
        hw = np.clip(size * np.exp(r.normal(0, 0.35, n)), 0.55 * size, 1.9 * size) * g
        tone = rock[y, x] * np.exp(r.normal(0, 0.03, (n, 1)) + r.normal(0, 0.03, (n, 3)))
        lay(go(field, P, hw * r.uniform(2.5, 6, n), 7, r.normal(0, 0.001, n), r.normal(0, 0.2, n), key=forms), hw,
            loads(tone, r, 0.03, np.concatenate([RUST, ROCK[1:3], SHADE, HALF[None]]), 0.15),
            thick=0.22, spent=0.5, pickup=0.55, merge=6, grooves=0.15, dry=r.uniform(0.0, 0.35, n), ends=(0.5, 0.2),
            taper=0.35, fray=2.5)
    wet[:] = 0

    # over dry paint: the sky on the surface, dragged dry along the swell on the faces of the
    # waves that turn to it, more of it further off and most in the glare; the waves in the deep
    # water a little darker than it, and lighter over the sand; the rocks caught in the water
    sky_side = noise.smoothstep(0.0, 0.8, np.sin(phase))
    P, n, g, (y, x) = strew(70, sea * np.clip(0.06 + 0.6 * (1 - near) ** 1.5 + 0.5 * glare, 0, 1) * sky_side)
    tint = SKY[r.integers(0, 5, n)]
    tint += (body[y, x] - tint) * r.uniform(0.15, 0.45, (n, 1)) * (1 - glare[y, x])[:, None]
    lay(go(crest, P, r.uniform(120, 340, n) * g, 10, r.normal(0, 0.001, n), key=land), r.uniform(14, 28, n) * g,
        loads(tint, r, 0.03), dry=r.uniform(0.45, 0.8, n), hide=0.7, thick=0.08, pickup=0, grooves=0.25,
        land=0.2, lift=0.1, taper=0.5, ends=(0.5, 0.3))
    for odds, scale, toward in ((noise.smoothstep(0.3, 0.05, clear) * noise.smoothstep(0.0, -0.8, np.sin(phase))
                                 * (0.3 + 0.7 * near), (0.68, 0.86), VIOLET[1]),
                                (noise.smoothstep(0.55, 0.2, clear) * noise.smoothstep(0.02, 0.15, clear) * sky_side,
                                 (1.2, 1.45), CYAN[2])):
        P, n, g, (y, x) = strew(38, odds * sea)
        tone = body[y, x] * r.uniform(*scale, (n, 1))
        tone += (toward - tone) * r.uniform(0.1, 0.35, (n, 1))
        lay(go(crest, P, r.uniform(30, 80, n) * g, 8, r.choice([-1, 1], n) * r.uniform(0.006, 0.018, n) / g, key=land),
            r.uniform(8, 14, n) * g, loads(tone, r, 0.04), dry=r.uniform(0.1, 0.45, n), thick=0.14, pickup=0,
            grooves=0.3, ends=(0.4, 0.1), taper=0.6)
    s = 4
    lq = land[::s, ::s].astype(np.float32)
    mirror = np.zeros_like(lq)
    for k in range(40):
        mirror = np.maximum(mirror, np.roll(lq, k, 0) * (1 - k / 40))
    mirror = up(mirror, s) * sea * noise.smoothstep(0.15, 0.6, 1 - near)
    P, n, g, (y, x) = strew(40, mirror)
    lay(go(crest, P, r.uniform(40, 130, n) * g, 7, r.normal(0, 0.003, n), key=land), r.uniform(6, 12, n) * g,
        loads(np.concatenate([ROCK[1:3], RUST[1:2], GOLD[1:2]])[r.integers(0, 4, n)], r, 0.05),
        dry=r.uniform(0.35, 0.65, n), hide=0.8, thick=0.12, pickup=0, grooves=0.3, taper=0.5)

    # the sun on the rocks in thick cream, and the sea lapping white at them where the swell comes in
    P, n, g, (y, x) = strew(50, land * light * noise.smoothstep(0.7, 0.95, lit))
    hw = np.clip(14 * np.exp(r.normal(0, 0.35, n)), 7, 26) * g
    lay(go(plane, P, hw * r.uniform(1.5, 4, n), 7, r.normal(0, 0.001, n), r.normal(0, 0.2, n), key=forms), hw,
        loads(np.maximum(rock[y, x], ROCK[0] * 0.97), r, 0.03, GOLD, 0.2), thick=0.32, dry=r.uniform(0.15, 0.5, n),
        pickup=0.15, grooves=0.2, ends=(0.5, 0.2), spent=0.6, taper=0.35, fray=2.0)
    gy, gx = np.gradient(ndimage.gaussian_filter(shore, 8))
    along = np.arctan2(gx, -gy).astype(np.float32)
    exposed = noise.smoothstep(-0.2, 0.7, gx - 0.6 * gy) * noise.smoothstep(-0.3, 0.6, noise.field((H, W), 120, r))
    P, n, g, (y, x) = strew(30, noise.smoothstep(28, 6, shore) * sea * exposed)
    froth = WHITE[r.integers(0, 4, n)]
    froth += (np.where((cast[y, x] > 0.5)[:, None], SKY[2], CYAN[2]) - froth) * r.uniform(0.1, 0.5, (n, 1))
    lay(go(along, P, r.uniform(15, 50, n) * g, 6, r.normal(0, 0.01, n)), r.uniform(3, 7, n) * g, loads(froth, r, 0.03),
        thick=0.35, dry=r.uniform(0.2, 0.6, n), pickup=0.2, ends=(0.4, 0.2), taper=0.6, fray=2.0, hide=0.85)

    # the boat, in a few loaded strokes, its shadow already lying on the sand below it: the floor
    # inside, rose in the sun and violet under the wall that shades it, the thwarts, the white
    # gunwales, and its flank turned from the sun
    e, nrm = np.array([np.cos(bang), np.sin(bang)]), np.array([-np.sin(bang), np.cos(bang)])
    sunny = np.sign(nrm @ -fall)                                      # which side of it faces the sun

    def on(t, v):
        t = np.asarray(t, np.float32)
        return np.stack([bx + e[0] * t * blen / 2 + nrm[0] * v * half(t), by + e[1] * t * blen / 2 + nrm[1] * v * half(t)], -1)

    t = np.linspace(-0.82, 0.8, 9)
    lay(np.stack([on(t, v) for v in (0.5 * sunny, 0.0, -0.5 * sunny)]), bbeam * np.array([0.15, 0.17, 0.15]),
        loads(np.stack([BOARDS[0], BOARDS[1], BOARDS[2]]), r, 0.03), thick=0.25, pickup=0.2, merge=2, ends=(0.6, 0.4),
        taper=0.4, grooves=0.25)
    lay(np.stack([on(np.full(5, tt), np.linspace(-0.85, 0.85, 5)) for tt in (0.2, -0.42)]), 0.055 * bbeam,
        loads(np.stack([WHITE[1], BOARDS[2]]), r, 0.02), thick=0.35, pickup=0.1, ends=(0.3, 0.3))
    t = np.linspace(-0.94, 0.97, 12)
    lay(np.stack([on(t, 0.9 * sunny), on(t, -0.9 * sunny), on(np.full(12, -0.93), np.linspace(-0.85, 0.85, 12))]),
        np.array([0.08, 0.065, 0.06]) * bbeam, loads(np.stack([WHITE[0], BOARDS[3], WHITE[2]]), r, 0.02),
        thick=0.4, pickup=0.05, ends=(0.4, 0.3), taper=0.5)
    lay(on(np.linspace(-0.9, 0.9, 10), -1.12 * sunny)[None], 0.06 * bbeam, loads(BOARDS[4][None], r, 0.03),
        thick=0.2, pickup=0.3, ends=(0.4, 0.2), taper=0.6, dry=0.15)

    # last, the sun breaking on the water: in the glare the sea turns silver, and on it thick quick
    # touches of white are strung along the crests, close together there, a few strayed out over the rest
    clearof = ndimage.binary_dilation(aboard, iterations=10)
    P, n, g, (y, x) = strew(46, sea * ~clearof * glare ** 2 * noise.smoothstep(-0.4, 0.6, np.sin(phase)))
    lay(go(crest, P, r.uniform(80, 240, n) * g, 8, r.normal(0, 0.001, n)), r.uniform(9, 18, n) * g,
        loads(np.concatenate([SKY[2:3], SKY[4:5], WHITE[3:]])[r.integers(0, 3, n)], r, 0.03), dry=r.uniform(0.3, 0.6, n),
        hide=0.65, thick=0.1, pickup=0, grooves=0.2, taper=0.6, ends=(0.5, 0.3))
    rows = np.maximum(noise.smoothstep(0.78 - 0.3 * glare, 0.98, np.cos(phase)),
                      noise.smoothstep(0.88 - 0.25 * glare, 0.99, np.cos(ripple)))
    gather = noise.smoothstep(-0.6, 0.6, noise.field((H, W), 110, r) + 1.2 * glare)
    P, n, g, (y, x) = strew(11, sea * ~clearof * rows * gather * (0.006 + 0.99 * glare ** 1.3))
    hw = np.clip(3.2 * np.exp(r.normal(0, 0.45, n)), 1.4, 10) * g
    hue = r.random(n)
    lay(go(crest, P, hw * r.uniform(2.5, 5.5, n), 5, r.normal(0, 0.03, n)), hw,
        loads(np.where((hue < 0.2)[:, None], GOLD[r.integers(0, 3, n)], WHITE[r.integers(0, 4, n)]), r, 0.02),
        thick=0.45, spent=0.7, pickup=0.08, ends=(0.6, 0.5), taper=0.6, fray=1.0, land=0.8, lift=0.6)

    height = ndimage.gaussian_filter(height, 0.7)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.7, gloss=0, reach=(0.75, 1.18))
    return img + 0.06 * impasto.glints(height, LIGHT)[..., None]
