"""Morning Sun in the Garden. Acrylic and pencil on canvas.

Alma Thomas taught art at a junior high school in Washington for thirty-five years, and began the
paintings she is known for after she retired, in the kitchen of the house on Fifteenth Street where she
had lived since she was a girl. The light falling through the holly tree outside her window, she said,
gave her the idea, and later the flights to the moon. She drew faint pencil lines up and down a white
canvas and laid along them short strokes of colour straight from the jar, each a single touch of a flat
brush, side by side in wavering columns with the white of the canvas left between them.

This canvas went on in two sittings. The first brushed upright bands of turquoise, cerulean, blue and
green loosely over the white, leaving the canvas bare between some of them and where the light was to
fall. Over the dry paint went the pencil lines, and then the columns of touches, each from the top of
the canvas to the foot: crimson, red and rose at the left; oranges; a band of cadmium yellow and lemon
leaning down from the upper right, where the touches stand furthest apart; coral and pinks; red at the
right edge. Each touch lands loaded, with a clean edge and a ridge of paint, carries the streaks of the
bristles, and runs dry toward its end, where it breaks up over the threads of the canvas. Between the
touches the bare canvas shows, or the blue and green beneath, and here and there the pencil line.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise, pencil
from atelier.color import glaze, lin, pigment

TITLE = "Morning Sun in the Garden"
DATE = "2026"
MEDIUM = "Acrylic and pencil on canvas: columns of touches of a flat brush over loosely brushed bands of blue and green"
AFTER = ("Alma Thomas, the paintings she made in Washington, 1966–76: Resurrection (1966), Starry Night and the "
         "Astronauts (1972), Red Roses Sonata (1972), Elysian Fields (1973), Wind and Crepe Myrtle Concerto (1973)")
ROOM = "Colour Itself"
YEAR = 1973
PLACE = "Washington"
REGION = "Americas"
NOTE = ("Columns of short strokes of a flat brush, crimson, red, orange, yellow and pink, change across the canvas "
        "in bands. Where a leaning band of yellow falls the white canvas opens between them; elsewhere flecks of "
        "blue and green show from beneath.")

H, W = 3000, 2400
LIGHT = (-0.6, -0.5, 0.62)
GROUND = "#f3f0e8"                  # white acrylic priming on cotton duck
GRAPHITE = pigment("#727174")


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# the jars, each from deep to light
WARM = dict(
    crimson=pal("#a3112c", "#b8162f", "#c9203c"),
    rose=pal("#b81d55", "#cc2f66", "#dd4a7c"),
    red=pal("#bf181d", "#cf2020", "#dc2d22"),
    scarlet=pal("#df3a1c", "#e6491b", "#ec5a1e"),
    orange=pal("#ec6c1a", "#f17f1f", "#f49428"),
    yellow=pal("#f4a91b", "#f7bd20", "#f8cf2c"),
    lemon=pal("#f6dc3a", "#f5e65c"),
    coral=pal("#e85a4c", "#ee6e5c", "#f2856e"),
    pink=pal("#e86f8a", "#ef8ba0", "#f4a5b6"),
)
CLEAR = {"crimson": 0.95, "rose": 0.97}   # the transparent ones let a little of what is beneath through
# the jars across the canvas from the left, each with the share of the width it holds
SEQ = [("rose", 1.0), ("crimson", 1.8), ("red", 2.8), ("scarlet", 1.6), ("orange", 2.0), ("yellow", 1.4),
       ("lemon", 0.6), ("orange", 0.9), ("coral", 0.9), ("pink", 1.6), ("rose", 0.8), ("red", 1.2)]
COOL = dict(
    turquoise=pal("#1f9f97", "#35b6ab", "#6ccbbf"),
    cerulean=pal("#2a7fc2", "#4499d2", "#6fb3de"),
    ultramarine=pal("#2a44b0", "#3a5cc4", "#5577d0"),
    emerald=pal("#0f7d58", "#1d966a", "#3aae80"),
    green=pal("#5a9e32", "#76b43e", "#97c858"),
)
BANDS = {None: 0.34, "turquoise": 0.22, "cerulean": 0.16, "ultramarine": 0.05, "emerald": 0.12, "green": 0.11}
LEAN = 420      # px the bands of colour run to the left from the top of the canvas to the foot


def sun(y):
    """Where the light falls at height y: the middle of the band of yellows, which leans down to the left."""
    share = np.array([s for _, s in SEQ])
    e = np.concatenate([[0], np.cumsum(share)]) / share.sum() * W
    i = [k for k, (n, _) in enumerate(SEQ) if n in ("yellow", "lemon")]
    return (e[min(i)] + e[max(i) + 1]) / 2 - LEAN * (y / H - 0.5)


def look(B, along, across):
    return ndimage.map_coordinates(B, [along, across], order=1, mode="grid-wrap")


def touch(rgb, height, tooth, B, r, x, y, th, L, w, col, col2=None, mix=0.0, hide=0.97, fade=0.35, tilt=(0.0, 0.0),
          thick=0.5, thin=0.15):
    """One stroke of a flat brush w px wide, set down at (x, y) and pulled L px toward the angle th. It lands
    loaded, with a clean edge and a ridge of paint; the bristles leave fine streaks along it; toward the end
    it runs dry, each bristle giving out at its own place, and breaks into streaks that catch only the high
    threads of the canvas. `tilt` slants the edge where it landed and the place where it gives out; `col2`
    streaks a second colour into it; `hide` is how well the paint covers, `thin` how much less it covers
    where the brush runs dry. In place."""
    Hh, Ww = height.shape
    c, s = np.cos(th), np.sin(th)
    hw = w / 2
    bend = r.normal(0, 5) / (L * L)          # the hand swings the brush through a slight curve, a few px off true
    pad = 4 + 0.06 * L + abs(bend) * L * L
    xs = x + np.array([-s * hw, s * hw, c * L - s * hw, c * L + s * hw])
    ys = y + np.array([c * hw, -c * hw, s * L + c * hw, s * L - c * hw])
    x0, x1 = int(max(0, xs.min() - pad)), int(min(Ww, xs.max() + pad + 1))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(Hh, ys.max() + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - x, yy - y
    sp = dx * c + dy * s                    # how far along the stroke
    q = -dx * s + dy * c - bend * sp * sp   # and how far across it
    z = np.zeros_like(q)
    o = r.uniform(0, 1, 14) * np.tile(B.shape, 7)
    streak = look(B, sp + o[0], q + o[1])                   # the marks of single bristles
    clump = look(B, sp * 0.3 + o[2], q * 0.22 + o[3])       # bristles clinging in bunches as they drag
    bunch = look(B, z + o[2], q * 0.22 + o[3])
    broad = look(B, z + o[4], q * 0.07 + o[5])              # the brush pressed harder on one side
    wob = look(B, sp * 0.4 + o[6], z + o[7])                # the side edges waver
    tone = look(B, sp * 0.05 + o[8], q * 0.12 + o[9])       # where the second colour lies in the brush
    u = np.abs(q) / hw
    land = tilt[0] * q + 1.4 * look(B, z + o[10], q * 0.15 + o[11]) + 3.0 * u ** 6     # corners round off
    # where the paint starts to give out, and where the last of it is left; the edge bristles hold least,
    # and a bunch here and there drags on a little further
    dry0 = L * (1 - fade) + tilt[1] * q - 0.35 * fade * L * u ** 4 + 0.3 * fade * L * broad
    dry1 = L + tilt[1] * q + 0.12 * fade * L * broad + 0.18 * fade * L * np.clip(bunch, -0.4, 2.5)
    left = np.clip((dry1 - sp) / np.maximum(dry1 - dry0, 2), 0, 1)
    t = tooth[y0:y1, x0:x1]
    a = left + (1 - left) * (0.15 * streak + 0.2 * clump + 1.3 * (t - 0.5))
    # pressed as it lands, the brush spreads a little; lifting, it narrows
    side = hw * (1.04 - 0.08 * np.clip(sp / L, 0, 1)) + (0.8 + 0.02 * w) * wob \
        + (1 - left) * 1.5 * look(B, sp * 0.8 + o[12], z + o[13])
    cover = noise.smoothstep(0.3, 0.42, a) * np.clip(side - np.abs(q) + 0.5, 0, 1) * np.clip(sp - land + 0.5, 0, 1)
    if not cover.any():
        return
    fresh = np.exp(-np.clip(sp - land, 0, None) / 6)       # the ridge where it landed
    edge = np.exp(-((side - np.abs(q)) / 2.0) ** 2)         # paint squeezed out along the sides of the brush
    body = (0.35 + 0.65 * left) * (1 + 0.4 * fresh + 0.3 * edge) * (1 + 0.12 * streak + 0.1 * clump)
    lc = np.log(np.broadcast_to(col, (y1 - y0, x1 - x0, 3)))
    if col2 is not None and mix > 0:
        lc = lc + (np.log(col2) - lc) * (mix * noise.smoothstep(-0.3, 0.5, tone))[..., None]
    lp = lc + np.log(1 + 0.015 * streak + 0.04 * clump - 0.04 * fresh - 0.03 * edge + 0.05 * (1 - left))[..., None]
    # how much the film hides: less as the brush runs dry, where a bunch of bristles carried less, and over the
    # high threads of the weave. A thin film of paint tints what it lies on rather than greying it, so it is
    # laid as a glaze; only where the paint is missing altogether does what lies beneath show plain
    F = hide * (1 - thin * (1 - left)) * (1 - 0.1 * noise.smoothstep(0.4, 1.6, -clump)) \
        * (1 - (0.14 + 0.32 * (1 - left)) * np.clip(t - 0.35, 0, None) / 0.65)
    patch = rgb[y0:y1, x0:x1]
    film = np.exp(np.log(np.maximum(patch, 1e-4)) * (1 - F)[..., None] + lp * F[..., None])
    patch += (film - patch) * cover[..., None]
    hp = height[y0:y1, x0:x1]
    hp[:] = hp * (1 - 0.3 * cover * left) + cover * body * thick


def bands(rgb, height, tooth, B, r):
    """The first sitting: upright bands of cool colour brushed in fast with a wide brush, in long strokes
    down the canvas, bare canvas left between some of them and wherever the light will fall."""
    names, odds = list(BANDS), np.array(list(BANDS.values()))
    x, last = -r.uniform(0, 80), None
    while x < W + 20:
        bw = r.uniform(60, 240)
        kind = names[r.choice(len(names), p=odds / odds.sum())]
        if kind is not None and kind != last:
            lanes = max(1, int(round(bw / r.uniform(90, 140))))
            for j in range(lanes):
                cx = x + (j + 0.5) * bw / lanes + r.normal(0, 6)
                ww = bw / lanes * r.uniform(1.0, 1.25)
                yv = -r.uniform(0, 300)
                while yv < H:
                    L = r.uniform(300, 900)
                    fam = COOL[kind if r.random() < 0.7 else names[1 + r.integers(len(names) - 1)]]   # dipped elsewhere
                    other = COOL[names[1 + r.integers(len(names) - 1)]]
                    if r.random() > 0.9 * noise.smoothstep(320, 140, abs(cx - sun(yv + L / 2))):
                        touch(rgb, height, tooth, B, r, cx + r.normal(0, 5), yv, np.pi / 2 + r.normal(0, 0.03), L, ww,
                              fam[r.integers(len(fam))], other[r.integers(len(other))],
                              r.uniform(0, 0.4) * (r.random() < 0.3), hide=r.uniform(0.6, 0.88),
                              fade=r.uniform(0.25, 0.55), tilt=(r.normal(0, 0.15), r.normal(0, 0.2)), thick=0.25)
                    yv += L * r.uniform(0.7, 0.92)
        last = kind
        x += bw + r.uniform(-15, 30)


def layout(r):
    """The columns: each about an inch wide, broader in some parts of the canvas than others, side by side
    across it with a few px of canvas between them. -> list of (left edge, width)"""
    broad = 1 + 0.15 * noise.line1d(W + 200, 700, r)
    cols, x = [], -r.uniform(10, 50)
    while x < W + 30:
        cw = r.uniform(56, 86) * broad[int(np.clip(x + 100, 0, W + 199))]
        cols.append((x, cw))
        x += cw + r.uniform(-2, 9)
    return cols


def guides(sheet, cols, r):
    """The pencil lines, drawn down the canvas by eye along the left edge of each column: graphite density."""
    press = np.zeros((H, W), np.float32)
    for x0, _ in cols:
        ys = np.linspace(-20, H + 20, 9)
        P = np.stack([x0 + r.normal(0, 3) + np.cumsum(r.normal(0, 2.5, len(ys))), ys], 1)
        pencil.line(press, P, 2.6, r, pressure=r.uniform(0.35, 0.6), wander=1.5, tremor=0.3, lift=(300, 900))
    return 0.6 * ndimage.gaussian_filter(pencil.catch(press, sheet, grip=2.2, soft=0.25), 0.7)


def colours(cols, r):
    """The touches, column by column from the left, each column from the top down: where each lands, its
    size and slant, and its colour. Each column holds to one jar, mostly, and the jars change across the
    canvas in bands of their own widths that drift as they go down it."""
    sway = noise.field((H // 8, W // 8), 90, r)              # the columns lean together, a little
    drift = noise.field((H // 8, W // 8), 110, r)            # where a band runs further one way
    gaps = noise.field((H // 8, W // 8), 50, r)              # where the columns break
    crowd = noise.field((H // 8, W // 8), 60, r)             # where the touches crowd together, where they open
    at = lambda f, x, y: f[int(np.clip(y / 8, 0, H // 8 - 1)), int(np.clip(x / 8, 0, W // 8 - 1))]
    share = np.array([s for _, s in SEQ])
    edges = np.cumsum(share)[:-1] / share.sum() * W
    out = []

    def jar(k):
        k = int(np.clip(k, 0, len(SEQ) - 1))
        return SEQ[k][0], WARM[SEQ[k][0]]

    for x0, cw in cols:
        own = 9 * noise.line1d(H + 400, 450, r) + 6 * noise.line1d(H + 400, 1400, r)
        rank = int(np.round(r.normal(0, 0.6)))      # a column may keep to the jar beside its neighbours'
        lean, pick, tall = r.normal(0, 0.03), r.uniform(0.1, 0.9), r.uniform(1.05, 2.0)
        y = -r.uniform(0, 140)
        while y < H + 10:
            L = cw * tall * np.exp(np.clip(r.normal(0, 0.25), -0.5, 0.5))
            cx = x0 + cw / 2
            xc = cx + 22 * at(sway, cx, y) + own[int(np.clip(y + 200, 0, H + 399))] + r.normal(0, 5)
            k = int(np.searchsorted(edges, cx + 150 * at(drift, cx, y) + LEAN * (y / H - 0.5) + r.normal(0, 16)))
            k += rank + (r.choice([-1, 1]) if r.random() < 0.07 else 0)
            name, fam = jar(k)
            # the touches crowd together in places and open in others, and most of all where the light is
            gap = float(np.clip(r.normal(3 + 5 * at(crowd, cx, y) + 5 * (name in ("yellow", "lemon")), 5), -12, 24))
            if r.random() < 0.015 + 0.1 * noise.smoothstep(0.8, 1.7, at(gaps, cx, y)):
                y += L + gap          # a touch left out: the column breaks
                continue
            col = fam[int(np.clip(pick + r.normal(0, 0.25), 0, 0.999) * len(fam))]
            col2, mix = None, 0.0
            if r.random() < 0.35:
                f2 = jar(k + r.choice([-1, 0, 1]))[1]
                col2, mix = f2[r.integers(len(f2))], r.uniform(0.2, 0.6)
            ww = cw * r.uniform(0.88, 1.12)
            down = r.random() < 0.8
            th = np.pi / 2 + lean + r.normal(0, 0.06) + (0 if down else np.pi)
            if r.random() < 0.035:    # now and then a touch pulled across instead of down
                th = r.choice([0.0, np.pi]) + r.normal(0, 0.1)
                Lh = cw * r.uniform(0.9, 1.25)
                out.append((xc - np.cos(th) * Lh / 2, y + L / 2, th, Lh, L * r.uniform(0.5, 0.75), col, col2, mix,
                            name))
            elif r.random() < 0.03:   # or two narrow ones side by side
                for side in (-1, 1):
                    n2, f2 = jar(k + r.choice([-1, 0, 0, 1]))
                    L2 = L * r.uniform(0.7, 1.0)
                    y2 = y + r.uniform(0, L - L2 + 1e-3) + (0 if down else L2)
                    out.append((xc + side * ww * 0.25, y2, th + r.normal(0, 0.05), L2,
                                ww * r.uniform(0.5, 0.56), f2[r.integers(len(f2))], None, 0.0, n2))
            else:
                out.append((xc, y if down else y + L, th, L, ww, col, col2, mix, name))
            y += L + gap
    return out


def paint(seed=1973):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint=GROUND, thread=3.0)
    rgb, height = ground.color.copy(), ground.tooth * 0.25
    B = impasto.bristles(r)
    bands(rgb, height, ground.tooth, B, r)
    cols = layout(r)
    rgb = glaze(rgb, guides(ground, cols, r), GRAPHITE)
    for x, y, th, L, w, col, col2, mix, name in colours(cols, r):
        touch(rgb, height, ground.tooth, B, r, x, y, th, L, w, col, col2, mix,
              hide=CLEAR.get(name, 1.0) * r.uniform(0.97, 1), fade=0.08 + 0.32 * r.random() ** 1.5,
              tilt=(r.normal(0, 0.12), r.normal(0, 0.25)), thin=0.25 if name in CLEAR else 0.15)
    height = ndimage.gaussian_filter(height, 0.6)
    return dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0.01, reach=(0.86, 1.12))


if __name__ == "__main__":      # one touch: loaded it covers, it paints nowhere else, and it breaks up as it runs dry
    r = noise.rng(0)
    rgb, height = np.full((240, 140, 3), 0.8, np.float32), np.zeros((240, 140), np.float32)
    red = lin("#cf2020")
    touch(rgb, height, noise.rng(1).uniform(0, 0.5, (240, 140)).astype(np.float32), impasto.bristles(r, (1024, 256)),
          r, 70, 20, np.pi / 2, 150, 60, red, fade=0.3)
    assert np.abs(rgb[40:110, 50:90] - red).max() < 0.1, "loaded, it covers"
    assert np.allclose(rgb[:, :12], 0.8) and np.allclose(rgb[:, -12:], 0.8) and np.allclose(rgb[215:], 0.8), \
        "it paints nowhere else"
    held = (np.abs(rgb[150:170, 50:90] - red).sum(-1) < 0.1).mean()
    assert 0.05 < held < 0.9, f"it breaks up as it runs dry ({held:.2f})"
    print("ok")
