"""Dunstanburgh, Sunrise. Oil on canvas.

In the last years he showed, 1835–45, Turner painted the sun coming up through mist over a river, a
steamer lost in a snowstorm, the sun going down into a lake, and everything in them goes into the light
until there is little left but the light. He worked on a white ground as if it were paper. Washes of
yellow, blue and grey went on first, rubbed on with a rag or swept on with a broad brush, as thin as
watercolour, and much of them came off again with the rag, the palette knife and the end of the brush,
so that the white comes back up through what is left and the light seems to come out of the canvas. Over
them he dragged a little white and pale colour with a dry brush, so that it catches on the weave. Only
the light is thick: the sun is no disc but a blot of lead white and chrome yellow smeared on with the
knife, and its glow is yellow scumbled round it.

Here the sun comes up over the sea off Dunstanburgh. The ruined gatehouse of the castle sits low on its
headland to the right, half gone in the mist, and a squall is coming in over the water from the left.
The thick whites have opened into a fine net of cracks; everywhere else the paint is so thin that the
weave shows.
"""

import numpy as np
from scipy import ndimage

from atelier import canvas, dabs, noise, tempera
from atelier.color import lin, pigment

TITLE = "Dunstanburgh, Sunrise"
DATE = "2026"
MEDIUM = "Oil on canvas, thin veils rubbed and brushed over a white ground, wiped and scraped back, the light laid with the knife"
AFTER = "J. M. W. Turner, the late paintings, 1835–45: Norham Castle, Sunrise; Snow Storm – Steam-Boat off a Harbour's Mouth"
ROOM = "Open Air"
YEAR = 1845
PLACE = "Dunstanburgh, Northumberland"
REGION = "Europe"
NOTE = "A sun over the sea, a squall coming in and a castle on its headland, all going into the light. Only the light is thick."

H, W = 2040, 2720            # about 91 by 122 cm
HORIZON = 1150
SX, SY = 1190.0, 700.0       # the sun
GROUND = "#f2ede0"           # lead white priming
YELLOW, SCUMBLE, PALE_GOLD, GOLD, WHITE, LEAD = "#f2da8e", "#f3dc84", "#f6e294", "#f1d050", "#f6f0e1", "#fbf8ef"
BLUE, MIST_BLUE, SWELL = "#aebfdc", "#bcc5d3", "#9ea8bc"
STONE, SHADOW = "#aaa39a", "#958e86"
GREY, UMBER, OCHRE = "#aeaba5", "#9a8466", "#d3b07a"


class Ground:
    """The primed canvas as the tools meet it: how high it stands at each pixel (the tops of the threads, and
    grit in the priming), and the textures tools leave in paint, laid out along a stroke (rows along it,
    columns across) and read at a fresh place for every stroke."""

    def __init__(self, sheet, r):
        self.r = r
        self.tops = 0.8 * sheet.tooth + 0.2 * noise.smoothstep(-2, 2, noise.field(sheet.tooth.shape, 3, r))
        # bristles gather into clumps of every size, each drawn out along the stroke
        self.clumps = {a: noise.stretched((2048, 1024), a, n, r) for a, n in ((2.5, 110), (5, 220), (10, 380), (20, 600))}
        self.folds = noise.fbm((1024, 1024), 28, r, octaves=3)          # a crumpled rag

    def read(self, tex, s, v, scale=1.0):
        h, w = tex.shape
        return tex[((s * scale).astype(np.intp) + self.r.integers(h)) % h, ((v * scale).astype(np.intp) + self.r.integers(w)) % w]

    def clump(self, s, v, across):
        return self.read(self.clumps[min(self.clumps, key=lambda a: abs(np.log(across / a)))], s, v)


def arc(a, b, bend, half, pad):
    """The pixels a tool `half` px either side of its path can reach, going from a to b along a circular
    arc that bows `bend` px to the left of the straight line. -> slices of the box round it, the mask of
    pixels in reach, and for those the distance along the path `s` and across it `v` (px), and the
    length of the path."""
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
    return (slice(y0, y1), slice(x0, x1)), m, s, v[m], L


def wander(s, L, amp, scale, r):
    """How far the hand drifts off its path, at each distance along it."""
    n = int(L) + 42
    return amp * noise.line1d(n, scale, r)[np.clip(s + 20, 0, n - 1).astype(np.intp)]


def brush(T, g, a, b, bend, width, load, dry=0.0, crisp=0.35, tail=(0.2, 0.5), rim=0.0):
    """A brush charged with thin paint, drawn from a to b. The pressure swells and eases as it goes, so the
    stroke is never one width for long, and the hand drifts off the arc. The paint starts in a ragged edge
    where the brush is set down, runs dry as it goes, and the tail breaks up on the weave. Wet, it runs into
    the hollows of the weave; dry, it touches only the tops of the threads. Along each side the edge is
    found in one place, a few px wide, and lost in another (`crisp`: how much of it is found), and here and
    there it breaks, where the bristles at the side ran dry. Where the paint was pushed aside there may be
    a faint rim."""
    r, half = g.r, width / 2
    fr = arc(a, b, bend, 1.35 * half, 4)
    if fr is None:
        return
    sl, m, s, v, L = fr
    n = int(L) + 42
    si = np.clip(s + 20, 0, n - 1).astype(np.intp)
    along = lambda scale: noise.line1d(n, scale, r)[si]
    hw = half * np.clip(1 + 0.18 * along(0.4 * L + 50), 0.6, 1.3) * (0.75 + 0.25 * noise.smoothstep(0, 0.15 * L, s))
    w = v - 0.12 * half * along(0.6 * L + 80)
    cl = g.clump(s, v, 0.04 * width + 2)
    blot = g.read(g.folds, 0.3 * s, v, 28 / max(0.3 * width, 28.0))
    found = noise.smoothstep(-0.4, 0.4, along(0.3 * L + 40) + np.where(w > 0, 0.5, -0.5) * along(0.5 * L + 40) + 2 * crisp - 1)
    soft = found * r.uniform(1.5, 5) + (1 - found) * r.uniform(0.3, 0.8) * hw
    inside = noise.smoothstep(0, soft, hw - np.abs(w) + (1.5 + 0.05 * half) * blot)
    broke = noise.smoothstep(0.3, 1.0, along(0.25 * L + 40)) * noise.smoothstep(0.55, 0.95, np.abs(w) / hw)
    fe = r.uniform(*tail) * L
    tp = noise.smoothstep(L - fe, L, s + 0.3 * fe * blot)                # how far into the tail
    d = dry + 1.2 * tp + 0.9 * broke
    tt = g.tops[sl][m]
    patch = g.read(g.folds, 0.4 * s, v)                                  # the hand presses harder here than there
    catch = noise.smoothstep(-0.08, 0.08, 0.7 * tt + 0.15 + 0.08 * cl + 0.15 * patch - (0.1 + 0.75 * d))
    d = np.minimum(d, 1)
    lay = (1 - d) * (1 + 0.3 * (0.5 - tt)) + d * 1.5 * catch
    flat = r.random() < crisp
    start = noise.smoothstep(0, r.uniform(3, 14) if flat else r.uniform(0.1, 0.4) * width,
                             s - (0.1 if flat else 0.4) * min(half, 30) * (w / hw) ** 2 + 0.06 * width * blot)
    edge = np.exp(-((hw - np.abs(w)) / 2.5) ** 2)
    T[sl][m] += load * (1 - 0.8 * tp) * (1 + 0.1 * cl + 0.25 * cl * d) * (1 + rim * edge) * inside * start * (s < L) * lay


def cloth(g, a, b, bend, width):
    """Where a rag drawn from a to b touches the canvas: a band that bunches and spreads as the cloth is
    pushed along, its edge torn by the folds, coming on and going off raggedly. -> slices, mask, touch
    (0..1), and the folds."""
    r, half = g.r, width / 2
    fr = arc(a, b, bend, 1.3 * half, 10)
    if fr is None:
        return None
    sl, m, s, v, L = fr
    folds = g.read(g.folds, 0.35 * s, v)                 # dragged out along the wipe
    big = g.read(g.folds, 0.35 * s, v, 28 / max(0.4 * width, 28.0))
    hw = half * np.clip(1 + 0.2 * wander(s, L, 1, 0.5 * L + 50, r), 0.6, 1.3)
    inside = noise.smoothstep(1.0, 0.5, np.abs(v - wander(s, L, 0.15 * half, 0.6 * L + 80, r)) / hw + 0.15 * folds + 0.15 * big)
    ends = noise.smoothstep(-10, 0.2 * L, s + 0.15 * L * big) * noise.smoothstep(L + 10, 0.6 * L, s + 0.15 * L * big)
    return sl, m, inside * ends, folds


def rag(Ts, g, a, b, bend, width, amount):
    """A rag wiped through the wet paint: it lifts most from the tops of the threads, and leaves the print
    of its own crumpled folds."""
    c = cloth(g, a, b, bend, width)
    if c:
        sl, m, touch, folds = c
        take = np.clip(amount * touch * (0.5 + 0.5 * g.tops[sl][m]) * (1 + 0.35 * folds), 0, 0.97)
        for T in Ts:
            T[sl][m] *= 1 - take


def rub(T, g, a, b, bend, width, load):
    """Thin paint rubbed on with a rag: it goes on in the print of the crumpled cloth, and more into the
    hollows of the weave than onto the threads."""
    c = cloth(g, a, b, bend, width)
    if c:
        sl, m, touch, folds = c
        T[sl][m] += load * touch * (1 + 0.2 * folds) * (1 + 0.3 * (0.5 - g.tops[sl][m]))


def scrape(Ts, g, a, b, bend, blade, bite):
    """The edge of the palette knife drawn through the wet paint: a lane with sharp sides, where the steel
    took the paint off the tops of the threads and left it in the hollows, and a fine ridge along each side
    where it pushed what it took. Nicks in the steel leave threads of paint standing."""
    r, half = g.r, blade / 2
    fr = arc(a, b, bend, half, 6)
    if fr is None:
        return
    sl, m, s, v, L = fr
    u = (v - wander(s, L, 2.0, 300, r)) / half
    au = np.abs(u)
    nick = noise.line1d(int(blade) + 40, 1.2, r)[np.clip(v + half + 20, 0, int(blade) + 39).astype(np.intp)]
    inside = noise.smoothstep(1.0, 0.95, au)
    ends = noise.smoothstep(0, 12, s) * noise.smoothstep(L, L - r.uniform(5, 80), s)
    tilt = np.clip(1 + r.uniform(-0.6, 0.6) * u, 0, None)
    f = np.clip(bite * tilt * (0.3 + 0.7 * noise.smoothstep(0.2, 0.7, g.tops[sl][m])) * (1 + 0.3 * nick), 0, 0.97) * inside * ends
    ridge = 0.1 * np.exp(-np.maximum(au - 1, 0) * half / 1.5) * (au > 1) * ends
    for T in Ts:
        T[sl][m] *= 1 - f + ridge


def knife(T, relief, g, a, b, bend, blade, load, under=None, share=0.5):
    """Thick paint laid with the flat of the knife: a smear whose sides are sharp but never straight, where
    the steel squeezed the paint out into a ridge, narrower at its ragged ends where the blade tilted, thinning as the knife runs out of paint until it skims
    and breaks into crumbs on the tops of the threads. Lifted square, it leaves a bank. A second colour
    picked up on the blade with the first (`under`, about `share` of the load) comes off with it in streaks
    drawn out along the pull."""
    r, half = g.r, blade / 2
    fr = arc(a, b, bend, 1.2 * half, 4)
    if fr is None:
        return
    sl, m, s, v, L = fr
    blot = g.read(g.folds, 0.5 * s, v, 28 / max(blade, 4.0))
    q = np.clip(s / L, 0, 1)
    w = np.abs(v - wander(s, L, 0.1 * half, 80, r))
    side = half * (0.85 + 0.15 * np.sqrt(np.sin(np.pi * q))) - w - (0.4 + 0.06 * blade) * blot
    inside = noise.smoothstep(0, 1.2, side)
    out = r.uniform(0.2, 0.8) * L                       # where the paint begins to run out
    skim = noise.smoothstep(-0.08, 0.08, g.tops[sl][m] + 0.3 * blot + 1.1 - 0.9 * noise.smoothstep(out, L, s) - 0.8)
    bank = (r.random() < 0.5) * 0.6 * np.exp(-np.maximum(L - s, 0) / 3)
    ridge = 0.4 * np.exp(-(side / 1.5) ** 2)
    th = load * ((1 - 0.4 * q) * (1 + 0.15 * g.clump(s, v, 2.5)) + bank + ridge) * inside * skim \
        * noise.smoothstep(0, 2, s + 0.25 * blade * blot) * (s < L + 0.25 * blade * blot)
    if under is None:
        T[sl][m] += th
    else:
        f = noise.smoothstep(-0.8, 0.8, 0.6 * g.clump(s, v, 0.3 * blade) + 0.6 * g.read(g.folds, 0.4 * s, v, 28 / blade) + 2 * share - 1)
        under[sl][m] += th * f
        T[sl][m] += th * (1 - f)
    relief[sl][m] += th


def scratch(Ts, g, a, b, bend, width=3.0):
    """The end of the brush handle drawn through the wet paint: a thin line down to the white, with a
    little ridge of paint thrown up on either side."""
    r, half = g.r, width / 2
    fr = arc(a, b, bend, half + 3, 3)
    if fr is None:
        return
    sl, m, s, v, L = fr
    u = (v - wander(s, L, 0.8, 90, r)) / half
    ends = noise.smoothstep(0, 8, s) * noise.smoothstep(L, L - 40, s) * np.clip(0.75 + 0.3 * wander(s, L, 1, 60, r), 0, 1)
    groove = noise.smoothstep(1.0, 0.35, np.abs(u)) * ends
    ridge = 0.25 * np.exp(-((np.abs(u) - 1.6) / 0.5) ** 2) * ends
    for T in Ts:
        T[sl][m] *= 1 - 0.85 * groove + ridge


def coat(img, T, colour, opacity):
    """Oil paint over what is there: a glaze that tints the light coming back up through it, and as much
    body as the pigment has, which covers."""
    ys, xs = np.nonzero(T.any(1))[0], np.nonzero(T.any(0))[0]
    if len(ys):
        sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        t = T[sl][..., None]
        a = 1 - np.exp(-opacity * t)
        img[sl] = img[sl] * np.exp(-t * pigment(colour)) * (1 - a) + lin(colour) * a


def about(cx, cy, near, far, loose=0.35, turn=0.6):
    """Strokes that go round a point (cx, cy) close to it, and farther off swing out in long shallow arcs."""
    def aim(x, y, r):
        dx, dy = x - cx, y - cy
        dist = max(np.hypot(dx, dy), 40.0)
        if r.random() < noise.smoothstep(near, far, dist):
            return drift(x, y, r)
        return (np.arctan2(dy, dx) + np.pi) % np.pi - np.pi / 2 + r.normal(0, loose), (dx, dy), turn / dist * r.uniform(0.3, 1.2)
    return aim


def drift(x, y, r):
    """Long shallow arcs, bowed this way and that, as the arm swings."""
    return r.normal(0, 0.15), (r.normal(), r.normal()), r.uniform(1 / 4000, 1 / 1200)


def swell(x, y, r):
    """Nearly level, as water lies, but never ruled."""
    return r.normal(0, 0.04), (0, r.choice((-1, 1))), r.uniform(0, 1 / 2500)


def level(x, y, r):
    """Level, as the land lies."""
    return r.normal(0, 0.03), (0, r.choice((-1, 1))), abs(r.normal(0, 2e-4))


def slant(x, y, r):
    """The squall and the rain under it, driven slantwise."""
    return 1.15 + r.normal(0, 0.08), (-1.0, 0.4), r.uniform(1 / 3000, 1 / 1200)


def paint(seed=1845):
    r = noise.rng(seed)
    sheet = canvas.duck((H, W), seed, tint=GROUND, thread=3.3)
    g = Ground(sheet, r)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coats, relief = [], np.zeros((H, W), np.float32)

    def layer(colour, opacity):
        T = np.zeros((H, W), np.float32)
        coats.append((T, colour, opacity))
        return T

    def sweep(tool, T, gap, where, aim, width, length, load, box=(0, 0, W, H), **kw):
        """Strokes of `tool` about `gap` px apart over the box, laid where the map `where` (0..1) asks for
        them and more loaded where it is higher, each turned and bowed the way `aim` says."""
        gy, gx = np.mgrid[box[1]:box[3]:gap, box[0]:box[2]:gap].astype(np.float32)
        pts = np.stack([gx.ravel(), gy.ravel()], 1) + r.uniform(0, gap, (gx.size, 2))
        for x, y in pts[r.permutation(len(pts))]:
            p = float(where[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
            if r.random() > p:
                continue
            ang, bow, curv = aim(x, y, r)
            Ln, wd = r.uniform(*length), r.uniform(*width)
            d = np.array([np.cos(ang), np.sin(ang)]) * r.choice((-1, 1))
            bend = np.sign(-d[1] * bow[0] + d[0] * bow[1]) * min(curv * Ln * Ln / 8, 0.25 * Ln)
            c = np.array([x, y])
            tool(T, g, c - d * Ln / 2, c + d * Ln / 2, bend, wd, load * (0.5 + 0.5 * p) * r.uniform(0.5, 1.5), **kw)

    hz = HORIZON + 4 * noise.line1d(W, 400, r)[None, :]
    sky = noise.smoothstep(hz + 10, hz - 10, yy)
    sea = 1 - sky
    # the light round the sun is not a circle: it spreads along the horizon, and runs down into the sea
    wx, wy = (90 * noise.fbm((H, W), 500, r, octaves=3) for _ in range(2))
    ds = np.hypot((xx + wx - SX) / 1.3, (yy + wy - SY) / np.where(yy > SY, 1.15, 0.85))
    glow = lambda reach: np.exp(-(ds / reach) ** 2)
    near = np.clip((yy - HORIZON) / (H - HORIZON), 0, 1)
    road = sea * np.exp(-((xx + 0.5 * wx - SX) / (60 + 330 * near)) ** 2)     # the sun's road on the water, widening towards us
    squall = noise.smoothstep(150, -600, xx + wx - (1020 - 0.6 * yy)) * (0.5 + 0.5 * sky)
    clear = 1 - 0.85 * squall
    sun_aim = about(SX, SY, 80, 450, 0.45, 0.25)

    # the lay-in, rubbed on with a rag and swept on with a broad brush, thin as watercolour: yellow round the
    # sun and down the sea, blue away from it, grey in the squall
    yellow = layer(YELLOW, 0.0)
    sweep(rub, yellow, 340, sky * (0.1 + 0.9 * glow(650)) * (1 - 0.8 * squall), sun_aim, (280, 560), (700, 1600), 0.1,
          box=(-300, -300, W + 300, HORIZON + 100))
    sweep(brush, yellow, 300, sky * (0.1 + 0.9 * glow(650)) * (1 - 0.8 * squall), sun_aim, (200, 420), (800, 1800), 0.1,
          box=(-300, -300, W + 300, HORIZON + 100), crisp=0.15)
    sweep(brush, yellow, 240, sea * (0.1 + 0.9 * np.maximum(road, 0.5 * glow(600))), swell, (180, 400), (900, 2000), 0.12,
          box=(-400, HORIZON, W + 400, H + 100), crisp=0.1)
    blue = layer(BLUE, 0.05)
    sweep(rub, blue, 340, sky * noise.smoothstep(300, 1100, ds) * (1 - 0.6 * squall), drift, (280, 560), (700, 1600), 0.18,
          box=(-300, -300, W + 300, HORIZON))
    sweep(brush, blue, 300, sky * noise.smoothstep(300, 1100, ds) * (1 - 0.6 * squall), drift, (200, 420), (800, 1800), 0.18,
          box=(-300, -300, W + 300, HORIZON), crisp=0.15)
    sweep(brush, blue, 230, sea * (1 - road) * noise.smoothstep(900, 2600, xx), swell, (180, 400), (900, 2000), 0.14,
          box=(-400, HORIZON, W + 400, H + 100), crisp=0.1)
    grey = layer(GREY, 0.15)
    sweep(rub, grey, 200, squall, slant, (240, 480), (600, 1200), 0.17, box=(-500, -500, 1500, HORIZON - 450))
    sweep(brush, grey, 60, noise.smoothstep(1100, 300, xx), level, (6, 16), (200, 700), 0.22, box=(-100, HORIZON - 6, 1100, HORIZON + 10),
          crisp=0.5)
    umber = layer(UMBER, 0.1)
    sweep(rub, umber, 280, squall ** 3, slant, (240, 440), (600, 1100), 0.08, box=(-500, -500, 900, 700))
    sweep(brush, umber, 280, sea * noise.smoothstep(900, 0, xx) * near, swell, (120, 300), (500, 1400), 0.09,
          box=(-400, HORIZON + 250, 1300, H + 100), crisp=0.2)
    # the rain hanging from it, dragged down with a drier brush
    sweep(brush, grey, 90, squall * noise.smoothstep(HORIZON + 30, HORIZON - 300, yy) * sky, slant, (40, 120), (500, 1100), 0.12,
          box=(-100, 150, 1300, HORIZON - 150), dry=0.35, crisp=0.3)

    # much of it taken off again with the rag, so that the ground comes back: most round the sun and down its road
    wet = [yellow, blue, grey, umber]
    sweep(rag, wet, 150, glow(420), sun_aim, (120, 300), (250, 800), 0.55, box=(SX - 700, SY - 600, SX + 700, HORIZON + 50))
    sweep(rag, wet, 90, road, swell, (60, 160), (300, 1000), 0.6, box=(0, HORIZON, W, H))
    sweep(rag, wet, 170, 4 * squall * (1 - squall) * sky, slant, (120, 260), (400, 1000), 0.4, box=(0, 0, 1500, HORIZON))
    sweep(rag, [grey, umber], 110, np.minimum(2 * squall, 1) * sea, swell, (60, 160), (400, 1200), 0.5, box=(-100, HORIZON + 10, 1500, H))
    sweep(rag, wet, 320, sky * (1 - squall), drift, (150, 320), (400, 1000), 0.35, box=(0, 0, W, HORIZON))

    # a few veils of white, dragged on drier: the mist along the sea, round the sun, and behind the castle
    mist = layer(WHITE, 2.0)
    sweep(brush, mist, 260, glow(450) * clear, sun_aim, (200, 400), (600, 1400), 0.06, dry=0.2, crisp=0.1)
    sweep(brush, mist, 200, np.exp(-((yy - HORIZON) / 70) ** 2), swell, (60, 160), (900, 2000), 0.06, dry=0.25, crisp=0.1,
          box=(-400, HORIZON - 160, W + 400, HORIZON + 140))
    sweep(brush, mist, 150, glow(320) * clear, sun_aim, (80, 200), (300, 800), 0.06, dry=0.45, rim=0.4, crisp=0.2)
    sweep(brush, mist, 140, np.exp(-((yy - HORIZON + 100) / 120) ** 2) * noise.smoothstep(1500, 1800, xx), drift, (80, 200),
          (600, 1400), 0.08, dry=0.15, crisp=0.1, box=(1300, HORIZON - 380, W + 400, HORIZON))

    # the headland, long and low and sloping into the sea; on it the gatehouse, two broken towers and the gate
    land = layer(STONE, 0.35)
    WL = HORIZON + 26
    rise = lambda x: 80 * noise.smoothstep(1400, 2300, x) - 8 * noise.smoothstep(2500, W, x)
    top = WL - rise(xx) + 7 * noise.fbm((1, W), 50, r, octaves=4)
    head = noise.smoothstep(top - 14, top + 30, yy) * noise.smoothstep(WL + 8, WL - 8, yy)
    sweep(brush, land, 40, head * noise.smoothstep(1350, 2100, xx + 150 * noise.line1d(W, 200, r)[None, :]), level, (24, 70), (150, 500),
          0.1, box=(1350, HORIZON - 120, W + 100, WL), crisp=0.3)
    for x0, x1, h in ((2140, 2236, 96), (2236, 2276, 56), (2276, 2372, 88)):   # two towers and the gate between
        y = WL - rise((x0 + x1) / 2) - h
        n = 2 if x1 - x0 > 60 else 1
        for k in range(n):                               # upright strokes of a flat brush, set down square at the top
            x = x0 + (k + 0.5) * (x1 - x0) / n
            brush(land, g, (x, y + r.uniform(0, 0.12 * h)), (x + r.normal(0, 2), WL - 18), r.normal(0, 2), (x1 - x0) / n + 6, 0.24,
                  tail=(0.3, 0.5), crisp=0.9)
        for _ in range(n + 1):                           # and the top broken: the knife takes notches out of it
            x = r.uniform(x0 + 8, x1 - 8)
            scrape([land], g, (x, y - 12), (x + r.normal(0, 3), y + r.uniform(10, 26)), 0, r.uniform(8, 14), 0.85)
    for x0, x1, h in ((1990, 2150, 12), (2370, 2580, 16)):  # the curtain wall, low and broken
        brush(land, g, (x0, WL - rise(x0) - h), (x1, WL - rise(x1) - h + r.normal(0, 3)), r.normal(0, 3), 2 * h, 0.16,
              tail=(0.3, 0.6), crisp=0.4)
    shadow = layer(SHADOW, 0.3)
    sweep(brush, shadow, 80, noise.smoothstep(1600, 2100, xx), level, (6, 14), (100, 320), 0.05, box=(1500, WL - 12, W + 100, WL + 2),
          crisp=0.4)
    for x in np.arange(2150, 2380, 40.0):                # the gatehouse in the water, faint
        brush(land, g, (x, WL), (x + r.normal(0, 4), WL + r.uniform(120, 220)), 0, r.uniform(40, 70), 0.06, dry=0.25,
              tail=(0.5, 0.8), crisp=0.1)
    sweep(rag, [land, shadow], 50, sea * noise.smoothstep(1500, 1700, xx), swell, (12, 34), (300, 900), 0.5, box=(1400, WL, W, WL + 400))
    veil = layer(WHITE, 2.0)                             # and the mist takes half of it back, most on the side towards the sun
    sweep(brush, veil, 46, noise.smoothstep(HORIZON + 30, HORIZON - 90, yy) * noise.smoothstep(2450, 1700, xx), drift,
          (40, 120), (200, 600), 0.1, dry=0.35, crisp=0.1, box=(1400, HORIZON - 260, 2600, WL + 20))
    for x0 in (2130, 2280):                              # the broken tops wiped into it
        y = WL - rise(x0) - 84
        rag([land], g, (x0 - 20, y + r.normal(0, 8)), (x0 + 110, y + r.normal(0, 8)), r.normal(0, 6), r.uniform(26, 44), 0.45)

    # the sea: a few long wet strokes, and more short dry ones scumbled over the tooth, the ground between them
    water = layer(MIST_BLUE, 0.5)
    sweep(brush, water, 330, sea * (1 - road) * (0.3 + 0.7 * near), swell, (120, 280), (900, 2000), 0.08, crisp=0.2,
          box=(-400, HORIZON, W + 400, H + 100))
    sweep(brush, water, 90, sea * (1 - road) * (0.2 + 0.8 * near), swell, (30, 110), (100, 700), 0.16, dry=0.6, crisp=0.4,
          box=(-200, HORIZON, W + 200, H + 50))
    ochre = layer(OCHRE, 0.15)
    sweep(brush, ochre, 300, sea * near * noise.smoothstep(1300, 300, xx), swell, (120, 280), (800, 1700), 0.12, crisp=0.2,
          box=(-400, HORIZON + 200, 1600, H + 100))
    sweep(brush, ochre, 100, sea * near * noise.smoothstep(1300, 300, xx), swell, (30, 110), (100, 700), 0.18, dry=0.6, crisp=0.4,
          box=(-200, HORIZON + 200, 1500, H + 50))
    sweep(rag, [water, ochre, yellow, umber, grey], 170, sea, swell, (30, 90), (300, 1000), 0.35, box=(0, HORIZON, W, H))
    trough = layer(SWELL, 0.3)                           # a few drier touches of the swell, nearer and darker
    sweep(brush, trough, 220, sea * near * (1 - road) * noise.smoothstep(1300, 2300, xx), swell, (14, 40), (150, 600), 0.1,
          dry=0.5, crisp=0.4, box=(1100, HORIZON + 200, W + 100, H))
    glint = layer(WHITE, 2.0)                            # the light along the road, dragged with a dry brush
    sweep(brush, glint, 80, road, swell, (30, 120), (150, 700), 0.1, dry=0.5, crisp=0.3, box=(SX - 750, HORIZON - 10, SX + 750, H))
    for _ in range(8):
        y = HORIZON + 40 + (H - HORIZON - 60) * r.random()
        x = SX + np.clip(r.normal(0, 120 + 200 * (y - HORIZON) / (H - HORIZON)), -450, 450) - 200
        scrape([yellow, blue, grey, umber, water, ochre], g, (x, y), (x + r.uniform(150, 500), y + r.normal(0, 6)),
               r.normal(0, 2), r.uniform(10, 30), r.uniform(0.4, 0.8))
    for _ in range(6):                                   # the knife and the brush handle through the near water on the left
        y, x = r.uniform(HORIZON + 300, H - 40), r.uniform(-100, 700)
        scrape([yellow, grey, umber, water, ochre], g, (x, y), (x + r.uniform(200, 600), y + r.normal(0, 5)), r.normal(0, 2),
               r.uniform(12, 34), r.uniform(0.4, 0.7))
    for _ in range(4):
        y, x = r.uniform(HORIZON + 400, H - 60), r.uniform(100, 900)
        scratch([yellow, umber, water, ochre], g, (x, y), (x + r.uniform(120, 300), y + r.normal(0, 4)), r.normal(0, 3), r.uniform(2.5, 3.5))

    # the sun: no disc, but a blot of lead white and chrome yellow smeared on thick with the knife, and round it
    # yellow scumbled with a dry brush, which is all its glow
    aura = layer(SCUMBLE, 1.2)
    sweep(brush, aura, 50, glow(240) * clear, sun_aim, (50, 150), (150, 500), 0.1, dry=0.55, crisp=0.2,
          box=(SX - 520, SY - 460, SX + 520, SY + 460))
    chrome, white, over = layer(GOLD, 3.0), layer(LEAD, 3.5), layer(GOLD, 3.0)
    for k in range(24):                                  # white and chrome on the one blade, smeared over each other
        kind = 0 if k < 12 else 1 if k < 18 else 2       # broad smears, then white dragged off to one side, then crumbs
        t = r.normal(0, 0.5) if kind != 1 else r.normal(0.2, 0.25)
        c = np.array([SX, SY]) + (r.normal(0, 1, 2) * ((46, 28), (40, 22), (90, 50))[kind] + ((0, 0), (70, 25), (0, 0))[kind])
        d = np.array([np.cos(t), np.sin(t)]) * r.choice((-1, 1))
        Ln = r.uniform(*((50, 110), (120, 240), (15, 35))[kind])
        top, low = (white, chrome) if k < 4 or r.random() < 0.6 else (over, white)
        knife(top, relief, g, c - d * Ln / 2, c + d * Ln / 2, r.normal(0, 10) if kind < 2 else 0, r.uniform(*((50, 84), (18, 36), (8, 16))[kind]),
              r.uniform(*((0.6, 1.0), (0.25, 0.45), (0.4, 0.8))[kind]), under=low, share=0.1 if k < 4 else r.uniform(0.2, 0.7))
    haze = layer(WHITE, 2.0)                             # and a dry veil of white over its edge, so that it has none
    sweep(brush, haze, 30, np.exp(-(np.hypot(xx - SX, (yy - SY) * 1.4) / 170) ** 2), sun_aim, (40, 110), (100, 300), 0.12, dry=0.3,
          crisp=0.1, box=(SX - 280, SY - 220, SX + 280, SY + 220))
    pale = layer(PALE_GOLD, 3.0)
    for _ in range(130):                                 # its road: crumbs of white and pale gold, small far off, larger near
        y = HORIZON + 6 + (H - HORIZON - 40) * r.random() ** 1.8
        q = (y - HORIZON) / (H - HORIZON)
        x = SX + r.normal(0, 30 + 230 * q)
        Ln = r.uniform(6, 30) * (1 + 2 * q)
        knife(white if r.random() < 0.7 else pale, relief, g, (x - Ln / 2, y), (x + Ln / 2, y + r.normal(0, 1.5)),
              r.normal(0, 1), r.uniform(3, 7) * (1 + q), r.uniform(0.2, 0.5))

    # rain scratched into the squall with the end of the brush, down to the white
    for _ in range(4):
        x, y = r.uniform(150, 850), r.uniform(250, 650)
        ang = 1.15 + r.normal(0, 0.05)
        Ln = r.uniform(250, 500)
        scratch(wet + [mist], g, (x, y), (x + Ln * np.cos(ang), y + Ln * np.sin(ang)), r.normal(0, 8), r.uniform(2.5, 4.0))

    # a last thin glaze of yellow round the sun, to make its whites burn
    last = layer(YELLOW, 0.0)
    sweep(brush, last, 200, glow(380), sun_aim, (200, 400), (500, 1200), 0.06, crisp=0.1, box=(SX - 800, SY - 700, SX + 800, HORIZON + 400))

    img = sheet.color.copy()
    for T, colour, opacity in coats:
        coat(img, T, colour, opacity)

    # the thick whites have cracked into a fine net, with dirt in the cracks
    ys, xs = np.nonzero(relief > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crack, _ = tempera.craquelure((y1 - y0, x1 - x0), r, cell=16, aspect=1.1, width=0.5, keep=0.7)
    crack *= noise.smoothstep(0.35, 0.8, relief[y0:y1, x0:x1])
    img[y0:y1, x0:x1] *= 1 - 0.25 * crack[..., None]
    body = sum(T * min(o, 1.0) for T, _, o in coats)
    height = 0.4 * sheet.tooth * np.exp(-2.0 * body) + 0.1 * body + 0.35 * relief
    height[y0:y1, x0:x1] -= 0.15 * crack
    return dabs.shine(img, ndimage.gaussian_filter(height, 0.7), relief=0.4, gloss=0.02, reach=(0.9, 1.1))
