"""Stacks of Wheat, Sunset, End of Summer. Oil on canvas, crusted over many sittings.

In the late summer of 1890 Monet began to paint the stacks of wheat that stood in a field
beside his house at Giverny, and went on into the winter: the same two or three stacks at
every hour and in every weather, a row of canvases going at once, a new one taken up as soon
as the light moved on, and each worked over again in the studio for months. The stacks are
only what the light falls on. Here the sun is low behind them at the end of a summer day: it
sets their edges on fire, their shadowed sides fill with violet and blue, and their shadows
run toward us over the stubble. He laid it all in small strokes of nearly pure colour, side
by side and over each other, sitting after sitting, the last dragged dry over the crust of
the ones before, so that the surface is dense, matt and granular, like the stubble itself.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "Stacks of Wheat, Sunset, End of Summer"
DATE = "2026"
MEDIUM = ("Oil on canvas, in small strokes of nearly pure colour laid over each other in many sittings, "
          "the last dragged dry over the crust")
AFTER = "Claude Monet, the Grainstacks, Giverny, 1890–91"
ROOM = "Open Air"
YEAR = 1891
PLACE = "Giverny"
REGION = "Europe"
NOTE = ("Two stacks of wheat with the sun going down behind them, their edges burning and their shadowed sides "
        "violet. The paint went on in small strokes, sitting after sitting, until the surface was a crust.")

H, W = 2000, 3000
LIGHT = (-0.6, -0.5, 0.62)
SUN = (1075.0, 662.0)        # just behind the right shoulder of the great stack
LEVEL = 905.0                # the far edge of the field, at the height of the eye; the shadows run from under the sun here
# each stack: centre, foot, half-width at the foot, at the eaves, of the roof's rim, the eaves, the top, how the top leans
STACKS = [(950, 1585, 430, 474, 488, 1165, 515, -14), (2180, 1252, 216, 238, 246, 1048, 712, 9)]
POPLARS = [(1668, 672, 24), (1716, 628, 27), (1830, 606, 29), (1902, 650, 25), (2296, 600, 28), (2352, 642, 24),
           (2466, 616, 27), (2668, 596, 29), (2716, 650, 24), (2838, 612, 27), (2966, 630, 26)]

# Monet's palette, each colour as it comes from the tube; the strokes are these let down with white
HUES = dabs.palette(["#f2d33c", "#eaa52c", "#e0762c", "#d2452e", "#b2304e", "#9b5ba4", "#3a3f9e", "#2f5fae",
                     "#4c8fb4", "#2e7d68", "#58a467"])
WHITE, DARK = "#f8f2e2", "#211b45"

# the design: what each place adds up to, seen from across the room
SKY_TOP, SKY_GREEN, SKY_ROSE = "#a9b49c", "#cdc795", "#d8a88f"
GOLD, GOLD_PALE, BLAZE = "#e5b263", "#f0cf8a", "#f9edc6"
HILL, HILL_MIST, HILL_GLARE = "#71698f", "#9e93ae", "#c1a2a6"
TREES, TREES_GLARE = "#5b5580", "#9c8399"
POPLAR, POPLAR_GLARE = "#77709a", "#a993a8"
HAZE, STUBBLE_FAR, STUBBLE, FIELD_GLARE = "#caa58b", "#a07c6a", "#87615a", "#dfbe8c"
DUSK_FAR, DUSK = "#6e6890", "#504c7a"
ROOF, ROOF_TOP, ROOF_ROSE, BODY, WARM, FOOT, EAVE = "#544a6c", "#67618a", "#674c6a", "#5b4463", "#80494c", "#33284a", "#382d50"
RIM, RIM_HOT, FLARE, ROSE = "#c0603d", "#eaa957", "#f5d692", "#9a5a6c"
# the brushes for the fire on the edges and the light round the sun
FIRE = dabs.palette(["#a83e30", "#c0502f", "#d26634", "#df7f3b", "#e89a47", "#efb458", "#f3ca72", "#f6dc96"])
EMBER = dabs.palette(["#8e4660", "#a85a6a", "#c0747a", "#d3908a"])                # madder let down
GLARE = dabs.palette(["#ecc77a", "#f2d692", "#f6e3ad", "#f9edc6", "#fbf3d9"])      # cadmium, lemon, lead white
GOLD_S = dabs.palette(["#d99a52", "#e2ab5c", "#e9bc6a", "#eec97c"])                # cadmium yellow deep, with white
PALE = dabs.palette(["#c9cf9f", "#d5d6aa"])                                      # the pale green beyond it


def mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ridge(r):
    """The crest of the hills across the valley, y for each x: a long low line falling gently to
    the right, with a few swells, and woods along its top."""
    x = np.arange(W, dtype=np.float32)
    return (690 + 80 * (x / W) ** 1.4 + 24 * noise.line1d(W, 650, r) + 10 * noise.line1d(W, 160, r)
            + 5 * noise.line1d(W, 40, r) + 2 * noise.line1d(W, 12, r)).astype(np.float32)


def poplars(r):
    """The row of poplars along the far edge of the field: tall narrow crowns, a little fuller
    below the middle and rounded off at the top, their edges feathery. -> (H,W) 0..1"""
    m = np.zeros((H, W), np.float32)
    y = np.arange(H, dtype=np.float32)[:, None]
    for x0, top, hw in POPLARS:
        a, b = max(0, x0 - 3 * hw), min(W, x0 + 3 * hw)
        x = np.arange(a, b, dtype=np.float32)[None]
        t = np.clip((LEVEL - y) / (LEVEL - top), 0, 1)
        half = hw * np.clip(1 - t ** 5, 0, 1) ** 0.5 * (0.85 + 0.15 * np.sin(np.pi * t)) * (y < LEVEL + 6) * (y > top)
        xc = x0 + 4 * noise.line1d(H, 140, r)[:, None] + 0.02 * (LEVEL - y) * r.normal()
        d = np.where(half > 0, half - np.abs(x - xc) + 3 * noise.field((H, b - a), 9, r), -9)
        m[:, a:b] = np.maximum(m[:, a:b], noise.smoothstep(-1, 1, d))
    return m


def stack(s, r):
    """One stack: a drum of wheat flaring a little to its eaves and a thatched roof drawn up from
    them into a blunt cone, both seen a little from above, so the foot and the eaves bow toward us.
    The outline wavers on each side. -> its shape, and where each point lies on it"""
    cx, yb, rb, re, rr, ye, ya, lean = s
    y = np.arange(H, dtype=np.float32)[:, None]
    dx = np.arange(W, dtype=np.float32)[None] - cx - 3 * noise.line1d(H, 160, r)[:, None]
    wl, wr = 1 + 0.022 * noise.line1d(H, 110, r, rows=2)
    wob = np.where(dx < 0, wl[:, None], wr[:, None])
    tr = np.clip((ye - y) / (ye - ya), 0, 1)
    off = lean * tr ** 2
    hr = rr * np.sqrt((1 - tr) * (2.2 - tr) / 2.2) * wob
    sb = np.clip((yb - y) / (yb - ye), 0, 1)
    hb = (rb + (re - rb) * sb ** 0.8) * wob
    eave = ye + 0.12 * rr * np.sqrt(np.clip(1 - (dx / rr) ** 2, 0, 1))
    foot = yb + 0.18 * rb * np.sqrt(np.clip(1 - (dx / rb) ** 2, 0, 1))
    roof = (y <= eave) & (y >= ya) & (np.abs(dx - off) <= hr)
    inside = roof | ((y > eave) & (y <= foot) & (np.abs(dx) <= hb))
    sdf = ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)
    full = lambda a: np.broadcast_to(a, (H, W)).astype(np.float32)
    return dict(inside=inside, roof=roof, sdf=sdf.astype(np.float32),
                across=np.clip(np.where(roof, (dx - off) / (hr + 1e-3), dx / hb), -1, 1).astype(np.float32),
                up=full(np.clip((yb - y) / (yb - ya), 0, 1)), low=full(sb), below=full(y - eave), above=full(foot - y),
                apex=(cx + lean, ya), foot=(cx, yb, rb))


def paint(seed=1891):
    r = noise.rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#d6c9ae", thread=3.0)

    # the far side of the valley, the field, the stacks
    crest = ridge(r)
    tops = LEVEL - 14 - 34 * noise.smoothstep(-0.3, 2.2, noise.line1d(W, 45, r) + 0.8 * noise.line1d(W, 240, r))
    hill = noise.smoothstep(-1, 1, yy - crest[None])
    trees = noise.smoothstep(-1, 1, yy - tops[None])
    pop = poplars(r)
    field = noise.smoothstep(-1, 1, yy - LEVEL - 4 * noise.line1d(W, 90, r)[None])
    A, B = (stack(s, r) for s in STACKS)
    inside = A["inside"] | B["inside"]
    sdf = np.minimum(A["sdf"], B["sdf"])
    up = np.where(A["sdf"] < B["sdf"], A["up"], B["up"])

    # the light. The sun spreads along the horizon; the shadows run from the point under it
    tx, ty = SUN[0] - xx, SUN[1] - yy
    dist = np.hypot(tx, ty) + 1e-3
    ds = np.hypot(tx, ty * 1.35)
    heat = np.exp(-ds / 520)
    gy, gx = np.gradient(ndimage.gaussian_filter(sdf, 4))
    facing = np.clip((gx * tx + gy * ty) / ((np.hypot(gx, gy) + 1e-6) * dist), 0, 1)   # an edge turned to the sun
    near = np.exp(-dist / 900)
    burn = np.clip(0.8 * np.exp(-dist / 380) + 0.6 * facing * near ** 0.5, 0, 1)        # how hot an edge glows
    th = np.arctan2(yy - LEVEL, xx - SUN[0]) + 0.012 * noise.field((H, W), 140, r) + 0.006 * noise.field((H, W), 30, r)
    run = np.hypot(xx - SUN[0], yy - LEVEL)
    shadow = np.zeros((H, W), np.float32)
    for cx, yb, rb in (A["foot"], B["foot"]):
        a0, a1 = sorted([np.arctan2(yb - LEVEL, cx + k * rb - SUN[0]) for k in (-1, 1)])
        d0 = np.hypot(cx - SUN[0], yb - LEVEL)
        soft = 0.012 + 0.03 * np.clip((run - d0) / 1500, 0, 1)
        shadow = np.maximum(shadow, noise.smoothstep(a0 - soft, a0 + soft, th) * noise.smoothstep(a1 + soft, a1 - soft, th)
                            * noise.smoothstep(0.85 * d0, 0.97 * d0, run))

    # the design, as the colours would add up across the room
    high = np.clip((LEVEL - yy) / LEVEL, 0, 1)
    want = mix(lin(SKY_GREEN), lin(SKY_TOP), noise.smoothstep(0.35, 1.0, high))
    want = mix(want, lin(SKY_ROSE), 0.75 * np.exp(-((yy - 700) / 120) ** 2) * (1 - heat))
    want = mix(want, lin(GOLD), noise.smoothstep(0.06, 0.55, heat))
    want = mix(want, lin(GOLD_PALE), noise.smoothstep(0.4, 0.85, heat))
    want = mix(want, lin(BLAZE), np.exp(-ds / 150))
    valley = mix(mix(lin(HILL), lin(HILL_MIST), 0.7 * noise.smoothstep(crest[None], LEVEL, yy)), lin(HILL_GLARE),
                 0.85 * np.exp(-ds / 430))
    want = mix(want, valley, hill)
    want = mix(want, mix(lin(TREES), lin(TREES_GLARE), 0.85 * np.exp(-ds / 400)), trees)
    want = mix(want, mix(lin(POPLAR), lin(POPLAR_GLARE), np.exp(-ds / 500)), pop)
    z = np.clip((yy - LEVEL) / (H - LEVEL), 0, 1)                           # 0 far .. 1 near
    lit = mix(mix(lin(HAZE), lin(STUBBLE_FAR), noise.smoothstep(0, 0.22, z)), lin(STUBBLE), noise.smoothstep(0.15, 0.8, z))
    lit = mix(lit, lin(FIELD_GLARE), 0.9 * np.exp(-np.hypot(tx * 0.8, (yy - LEVEL) * 3) / 450))
    rows = np.interp(6000 / (np.maximum(yy - LEVEL, 0) + 60) + 0.5 * noise.field((H, W), 300, r),
                     np.arange(0, 120, 0.5, dtype=np.float32), noise.line1d(240, 2.5, r))        # rows of stubble going back
    lit = lit * np.exp(0.07 * rows[..., None] * np.float32([1, 0.85, 0.6]))
    want = mix(want, mix(lit, mix(lin(DUSK_FAR), lin(DUSK), z), shadow), field)
    halo = np.exp(-np.maximum(sdf, 0) / 60) * (0.2 + 0.8 * facing) * near * (1 - field)
    want = mix(want, lin(FLARE), 0.7 * halo)
    for S in (A, B):
        face = np.sqrt(1 - S["across"] ** 2)
        top = mix(lin(ROOF), lin(ROOF_TOP), noise.smoothstep(0.45, 1.0, S["up"]) * (0.4 + 0.6 * face))
        top = mix(top, lin(ROOF_ROSE), 0.5 * noise.smoothstep(0.0, -0.9, S["across"]))     # turned to the rose of the sky
        drum = mix(lin(BODY), lin(WARM), noise.smoothstep(0.8, 0.1, S["low"]) * (0.35 + 0.65 * face))
        c = np.where(S["roof"][..., None], top, drum)
        c = mix(c, lin(EAVE), 0.35 * noise.smoothstep(110, 0, S["below"] + 30 * noise.field((H, W), 60, r)) * ~S["roof"])
        c = mix(c, lin(FOOT), 0.7 * noise.smoothstep(45, 0, S["above"]))
        rim = np.exp(-np.maximum(-S["sdf"], 0) / (10 + 34 * burn)) * (0.2 + 0.8 * burn) * noise.smoothstep(0.05, 0.2, S["up"])
        c = mix(c, mix(lin(ROSE), mix(lin(RIM), lin(RIM_HOT), near ** 1.5), noise.smoothstep(0.05, 0.5, burn)), rim)
        c = mix(c, lin(FLARE), np.exp(-np.maximum(-S["sdf"], 0) / 30) * np.exp(-dist / 160))
        want = mix(want, c, noise.smoothstep(1, -1, S["sdf"]))
    # colour drifting in patches, rose here and green there, as the eye finds it
    drift = np.stack([noise.field((H, W), 260, r), noise.field((H, W), 260, r)], -1) @ np.float32(
        [[0.5, -0.1, -0.4], [-0.3, 0.35, -0.05]])
    want = (want * np.exp(0.12 * drift)).astype(np.float32)

    # which way the strokes run. In the sky out from the sun, turning further off into a slant
    # hatching; along the hills, up the poplars, across the field and down the shadows toward us;
    # down the roofs of the stacks from the top, and down their drums
    out = np.arctan2(-ty, -tx)
    hatch = -0.35 + 0.45 * noise.field((H, W), 400, r)
    pull = np.exp(-ds / 300)[..., None]
    v = pull * np.stack([np.cos(2 * out), np.sin(2 * out)], -1) + (1 - pull) * np.stack([np.cos(2 * hatch), np.sin(2 * hatch)], -1)
    ang = 0.5 * np.arctan2(v[..., 1], v[..., 0]) + 0.3 * noise.field((H, W), 150, r)
    slope = np.arctan(np.gradient(ndimage.gaussian_filter1d(crest, 60)))[None]
    ang = np.where(hill > 0.5, slope + 0.3 * noise.field((H, W), 200, r), ang)
    ang = np.where(trees > 0.5, 0.6 * noise.field((H, W), 50, r), ang)
    ang = np.where(pop > 0.5, np.pi / 2 + 0.12 * noise.field((H, W), 40, r), ang)
    ang = np.where(field > 0.5, np.where(shadow > 0.5, th, 0.22 * noise.field((H, W), 250, r)), ang)
    for S in (A, B):
        ax, ay = S["apex"]
        ang = np.where(S["roof"], np.arctan2(yy - ay, xx - ax) + 0.2 * noise.field((H, W), 90, r), ang)
        ang = np.where(S["inside"] & ~S["roof"], np.pi / 2 + 0.35 * noise.field((H, W), 120, r), ang)
    ang = ang.astype(np.float32)
    along = (np.arctan2(gy, gx) + np.pi / 2).astype(np.float32)             # round the outline of a stack
    deep = (0.45 + 0.85 * z) * field + (1 - field)                         # strokes grow toward us
    edge = noise.smoothstep(0, 90, np.abs(sdf))                            # and shorten near the outlines

    # the canvas rubbed over first with thin paint of the colour of each place, a little darker
    rag = noise.stretched((H, W), 12, 90, r)
    rgb = (0.85 * want * (1 + 0.05 * rag[..., None])).astype(np.float32)
    height = ground.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def strew(spacing, odds):
        """Where the strokes of a passage start: a shaken honeycomb, kept by `odds` (H,W)."""
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing, r) - spacing
        return P[r.random(len(P)) < odds[at(P)]]

    def broken(P, pure=(0.35, 0.65), odds=0.05, spread=0.12, vivid=0.03):
        """The brush for each stroke: the colour of the place pushed part of the way toward a colour
        nearly pure from the tube let down to its value, now and then all the way, a second such
        colour streaked beside it, and a little of the colour the place adds up to."""
        T = want[at(P)]
        n = len(P)
        lt = np.log(np.clip(T, 1e-4, 1))

        def one():
            tube = np.log(np.clip(dabs.tint(T, HUES, r, WHITE, DARK, vivid=vivid, spread=spread), 1e-4, 1))
            f = r.uniform(*pure, n)
            f[r.random(n) < odds] = 1
            return np.exp(lt + (tube - lt) * f[:, None])
        share = np.stack([r.uniform(0.55, 0.8, n), r.uniform(0.1, 0.3, n), r.uniform(0.1, 0.25, n)], 1)
        return np.stack([one(), one(), T * np.exp(r.normal(0, 0.04, (n, 1)))], 1).astype(np.float32), share

    def loads(colours, tone, accent=None, odds=0.2, spread=0.08):
        """A brush loaded from one palette at `tone` (0 dark .. 1 light), a neighbour streaked in,
        and now and then an accent picked up from elsewhere."""
        m, N = len(colours), len(tone)
        i = (np.clip(tone + r.normal(0, spread, N), 0, 0.999) * m).astype(int)
        j = np.clip(i + r.choice([-1, 1], N), 0, m - 1)
        third = colours[np.clip(i + r.choice([-1, 1], N), 0, m - 1)]
        if accent is not None:
            hit = r.random(N) < odds
            third[hit] = accent[r.integers(0, len(accent), hit.sum())]
        share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.03, 0.2, N)], 1)
        return np.stack([colours[i], colours[j], third], 1), share

    def go(odds, spacing, length, width, scale=None, flow=None, tilt=0.0, bend=0.0, load=broken, dry=0.0, n=8,
           group=(1, 1), **kw):
        """A passage. The strokes go down in small groups, a few laid side by side from one loading
        of the brush at one slant, each drier than the last, as a painter hatches; then the next."""
        P = strew(spacing * np.sqrt(np.mean(group)), odds)
        G = len(P)
        m = r.integers(group[0], group[1] + 1, G)
        own = np.repeat(np.arange(G), m)
        j = np.arange(len(own)) - np.repeat(np.cumsum(m) - m, m)                 # its place in the group
        f = ang if flow is None else flow
        lean = r.normal(0, tilt, G)[own]
        a = f[at(P)][own] + lean
        k = scale[at(P)][own] if scale is not None else np.ones(len(own))
        wd = r.uniform(*width, G)[own] * r.uniform(0.85, 1.15, len(own)) * np.sqrt(k)
        side = (j - (m[own] - 1) / 2) * 1.6 * wd
        Q = P[own] + np.stack([-np.sin(a), np.cos(a)], 1) * side[:, None] \
            + np.stack([np.cos(a), np.sin(a)], 1) * (r.normal(0, 0.4, len(own)) * wd)[:, None]
        paths = impasto.follow(f, Q, r.uniform(*length, G)[own] * r.uniform(0.75, 1.25, len(own)) * k, n,
                               r.normal(0, bend, G)[own], lean)
        back = (r.random(G) < 0.5)[own]
        paths[back] = paths[back, ::-1]
        cols, share = load(P)
        d = (r.uniform(*dry, G) if isinstance(dry, tuple) else np.full(G, dry))[own]
        stay = odds[at(Q)] > 0.02                                               # a group stops at the edge of its passage
        impasto.lay(rgb, height, paths[stay], wd[stay],
                    (cols[own] * np.exp(r.normal(0, 0.03, (len(own), 1, 3))))[stay].astype(np.float32), r,
                    share=share[own][stay], wet=wet, dry=np.clip(d + 0.08 * j, 0, 0.85)[stay], **kw)
        return P

    sky = (1 - hill) * (1 - pop) * ~inside
    far = hill * (1 - field) * (1 - pop) * ~inside
    crowns = noise.smoothstep(0.6, 1, pop) * (1 - field) * ~inside
    land = field * ~inside
    body = inside.astype(np.float32)
    thin = lambda m: m * (0.45 / deep) ** 2                                  # fewer, larger strokes toward us
    k_land, k_body, k_sky = deep * (0.5 + 0.5 * edge), 0.3 + 0.7 * edge, 0.4 + 0.6 * edge
    soft = dict(ends=(0.45, 0.3), taper=0.3, fray=2.2)
    dash = dict(ends=(0.6, 0.5), taper=0.45, fray=2.2)                       # the stubble in rounder, finer touches

    # the lay-in: the whole canvas brushed over in broad strokes of the colour of each place
    go(np.ones((H, W)), 150, (100, 220), (20, 30), tilt=0.2, load=lambda P: broken(P, (0.0, 0.2), 0.0, 0.08),
       thick=0.03, grooves=0.15, lips=0.02, land=0.1, lift=0.05, pickup=0.6, merge=6, spent=0.6, fray=3, taper=0.4)
    height -= 0.8 * (height - 0.2 * ground.tooth)
    wet[:] = 0

    # the second sitting, wet into wet: each thing in its colours
    go(sky, 32, (35, 80), (9, 14), k_sky, tilt=0.35, bend=0.003, group=(1, 4), thick=0.16, pickup=0.45, merge=2,
       dry=(0, 0.25), **soft)
    go(far, 26, (25, 60), (7, 11), tilt=0.3, group=(1, 3), thick=0.14, pickup=0.45, merge=2, dry=(0, 0.3), **soft)
    go(crowns, 15, (10, 26), (5, 8), tilt=0.15, thick=0.14, pickup=0.45, merge=2, dry=(0, 0.3), **soft)
    go(thin(land), 34 * 0.45, (25, 60), (8, 13), k_land, tilt=0.3, group=(1, 3), thick=0.16, pickup=0.45, merge=2,
       dry=(0, 0.3), **soft)
    go(body, 24, (25, 65), (8, 13), k_body, tilt=0.35, group=(1, 3), thick=0.2, pickup=0.45, merge=2, dry=(0, 0.3),
       **soft)
    wet[:] = 0
    height += 0.35 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 6, r, octaves=2))    # crusts left by sittings between
    height += 0.25 * noise.smoothstep(0.2, 1.8, noise.fbm((H, W), 22, r, octaves=3)) \
        + 0.15 * noise.smoothstep(0, 2, noise.field((H, W), 2.5, r))

    # the third, over dry paint: smaller strokes, nearer the colours of the tubes, side by side and over each other
    hot = lambda P: broken(P, (0.4, 0.75), 0.05, 0.12, 0.03)
    dry3 = dict(load=hot, pickup=0, merge=0, dry=(0.15, 0.5), **soft)
    go(sky * (0.4 + 0.6 * heat), 22, (20, 45), (6, 10), k_sky, tilt=0.45, bend=0.003, group=(2, 5), thick=0.22, **dry3)
    go(far * 0.6, 19, (16, 40), (5, 8), tilt=0.35, group=(1, 3), thick=0.2, **dry3)
    go(crowns * 0.6, 13, (8, 18), (4, 6), tilt=0.15, thick=0.2, **dry3)
    calm = lambda P: broken(P, (0.2, 0.45), 0.02, 0.08, 0.02)
    go(thin(land) * (0.45 + 0.4 * (1 - shadow)), 24 * 0.45, (20, 55), (4.5, 8), k_land, tilt=0.45, bend=0.004,
       group=(1, 3), thick=0.22, **{**dry3, "load": calm, "dry": (0.1, 0.45), **dash})
    go(body, 19, (20, 45), (6, 10), k_body, tilt=0.4, group=(2, 4), thick=0.24, **dry3)
    wet[:] = 0

    # the fire on the edges: thick strokes of vermilion, orange and gold drawn round the outlines,
    # hottest and broadest where they turn to the sun, broken and madder-rose where they turn away;
    # then the sky brought back over them from outside
    P = strew(8, noise.smoothstep(22, 6, np.abs(sdf + 8)) * noise.smoothstep(0.04, 0.12, up)
              * (0.3 + 0.7 * burn) * noise.smoothstep(-1.6, 0.2, noise.field((H, W), 70, r)))
    b = burn[at(P)]
    cols, share = loads(np.concatenate([EMBER[:3], FIRE[1:]]), b, EMBER, 0.3)
    impasto.lay(rgb, height, impasto.follow(along, P, r.uniform(10, 32, len(P)), 6, r.normal(0, 0.004, len(P)),
                                            np.where(r.random(len(P)) < 0.6, r.normal(0, 0.25, len(P)), r.normal(0, 0.7, len(P)))),
                r.uniform(3.5, 6, len(P)) * (1 + b), cols, r, share=share, thick=0.3, pickup=0.25,
                dry=r.uniform(0.1, 0.55, len(P)), wet=wet, merge=1, hide=0.85, **soft)
    go(noise.smoothstep(0, 10, sdf) * noise.smoothstep(60, 20, sdf) * (1 - field), 14, (20, 45), (5, 9), flow=along,
       tilt=0.25, load=lambda P: broken(P, (0.2, 0.5), 0.05, 0.1), thick=0.22, pickup=0.2, merge=1, dry=(0.1, 0.4), **soft)
    wet[:] = 0

    # the last sitting, over all of it dry: gold dragged through the glow of the sky, a veil of haze
    # over the hills, warm and cool touches woven into the stubble, and the light itself, laid
    # thickest round the place where the sun has gone behind the stack
    last = dict(pickup=0, merge=0, **soft)
    stubble = dict(pickup=0, merge=0, **dash)
    go(sky * noise.smoothstep(0.08, 0.5, heat), 30, (25, 60), (7, 12), k_sky, tilt=0.5, bend=0.003, group=(2, 4),
       load=lambda P: loads(np.concatenate([GOLD_S, GLARE[:3]]),
                            np.clip(heat[at(P)] ** 0.7, 0, 1), np.concatenate([EMBER[2:], PALE]), 0.25),
       thick=0.2, dry=(0.35, 0.65), **last)
    go(far * 0.6, 30, (30, 70), (7, 11), tilt=0.3, load=lambda P: broken(P, (0.0, 0.3), 0.0, 0.05), thick=0.12,
       hide=0.5, dry=(0.4, 0.7), **last)
    go(thin(land) * shadow * 0.65, 26 * 0.45, (14, 40), (3.5, 6), k_land, tilt=0.6, bend=0.005, group=(1, 2), load=hot,
       thick=0.22, dry=(0.25, 0.6), **stubble)
    go(thin(land) * (1 - shadow) * 0.65, 26 * 0.45, (14, 40), (3.5, 6), k_land, tilt=0.6, bend=0.005, group=(1, 2),
       load=calm, thick=0.24, dry=(0.3, 0.7), **stubble)
    P = strew(11, np.exp(-dist / 170) * (dist < 420) * (sdf > -8) * (1 - 0.85 * hill))
    cols, share = loads(GLARE, np.exp(-dist[at(P)] / 240), FIRE[5:], 0.15)
    impasto.lay(rgb, height, impasto.follow(ang, P, r.uniform(25, 60, len(P)), 6, r.normal(0, 0.004, len(P)),
                                            r.normal(0, 0.5, len(P))),
                r.uniform(5, 9, len(P)), cols, r, share=share, thick=0.35, pickup=0, dry=r.uniform(0.1, 0.45, len(P)),
                wet=wet, merge=0, **dash)

    img = dabs.shine(rgb, ndimage.gaussian_filter(height, 0.8), light=LIGHT, relief=0.5, gloss=0, reach=(0.78, 1.12))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
