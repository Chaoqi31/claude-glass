"""Blossom Rafts at Arashiyama. Colour woodblock print.

In the third month the cherries on Arashiyama, at the western edge of Kyoto, come into flower all at
once, and the petals that fall on the Ōi river drift down it in long pale rafts that are called
hanaikada, flower rafts. Real rafts come down the same water: timber from the mountains of Tanba,
lashed into strings of logs and poled out of the gorge. Hiroshige drew the river at Arashiyama for his
Famous Places in Kyoto around 1834. This is a new view of it: the long Togetsukyō bridge crossing to
the foot of the mountain, the mountain in flower above bands of mist, a cherry leaning out from the near
bank, and rafts coming down the river toward the bridge.

It is cut on cherry blocks in the ōban landscape format. A keyblock prints the outlines in sumi, cut
heavy in the foreground and finer with distance: the near cherry, gnarled and tapering, its bark cut with
lenticels and knots; the rocks and grasses of the near bank; the bridge and its piles, the rafts and the
raftsmen. On the mountain the cherries stand in drifts of every size and shape, and the keyblock outlines
only some of the crowns along the top of each; below them a drift is one mass on the pink block, the
deeper pink wiped down from under the crowns and gone by its foot, and bands of mist cut across. Colour
blocks are laid over the keyblock. The Prussian blue of the sky and the river was wiped on the block by
hand, deep at the top of the sky, under the far bank and along the near water and gone toward the middle,
so the bare paper does the light; the current is cut out of it in long lines, and the petals in small flat
cuts that the pink block fills. Each fold of the mountain is a green with a darker green wiped down from its
ridge; the near bank is wiped deeper green down its slope, and its rocks blue-grey toward the water. The
paper is hōsho: the grain of the cherry shows in the flats, the baren leaves its swirl, no two blocks quite
register, and where the ink thins at the end of a gradation it misses the valleys of the paper and comes
out salted.
"""

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from atelier import lettering, noise, paper, plate, relief
from atelier.color import glaze, pigment

TITLE = "Blossom Rafts at Arashiyama"
DATE = "2026"
MEDIUM = "Colour woodblock print (nishiki-e) in thirty-five impressions, ōban yoko-e, on hōsho paper"
AFTER = ("Utagawa Hiroshige, Famous Places in Kyoto (Kyōto meisho no uchi), c. 1834, among them Cherry "
         "Blossoms in Full Bloom at Arashiyama")
ROOM = "The Workshop"
YEAR = 1834
PLACE = "Kyoto"
REGION = "East Asia"
NOTE = ("Timber rafts come down the Ōi toward the long bridge, and the petals that fall on the river drift "
        "down it in rafts of their own. On the mountain the cherries stand in drifts, a few crowns cut along "
        "the top of each and the deeper pink wiped down beneath them into the mist.")

H, W, M = 2000, 2900, 90            # the image, and the paper left round it
SH, SW = H + 2 * M, W + 2 * M
SS = 3                              # the keyblock is cut at three times the size it prints
HILL, TREE = 42, 12                 # which hillside of the many chance would give, and which cherry

SUMI = pigment("#3b3734")
PRUSSIAN = pigment("#1d4c8a")
AI = pigment("#a3c1da")
BENI = pigment("#f6ccd2")
ROSE = pigment("#de8094")
GLOW = pigment("#f7cfb5")
FAR = pigment("#aab8d8")
GREEN = pigment("#a9c58a")
DEEP = pigment("#3f6f68")
PINE = pigment("#2f4f48")
MOEGI = pigment("#cfdc90")
TAN = pigment("#dcc293")
PIER = pigment("#85786c")
GREY = pigment("#8f8b87")
BARK = pigment("#76655a")
LOG = pigment("#94714f")
RED = pigment("#d0583e")
INDIGO = pigment("#56739f")
SKIN = pigment("#efcfae")
STRAW = pigment("#e6cf98")

HC, HZ, FOC = 20.0, 800.0, 2700.0               # the eye 20 m over the water, its horizon, focal length px
FE = np.array([18.9, 170.0])                     # the far end of the bridge, at the foot of the mountain (X, Z m)
BD = np.array([-53.8, -105.0]) / np.hypot(53.8, 105.0)    # along the bridge, toward us
BN = np.array([BD[1], -BD[0]])                   # across it, away from us; the river runs that way too
DECK, WIDE, SPAN = 5.0, 5.4, 132.0

# the folds of the mountain, back to front, as skylines (x, y); then the far hills and the far bank
ARASHI = [(1040, 1052), (1140, 1002), (1230, 934), (1300, 876), (1362, 852), (1424, 786), (1500, 668), (1600, 528),
          (1720, 412), (1850, 336), (1980, 292),
          (2110, 296), (2240, 330), (2380, 378), (2540, 418), (2720, 444), (2900, 468), (3000, 478)]
SHOULDER = [(1240, 1010), (1380, 850), (1520, 700), (1660, 590), (1820, 530), (1980, 515), (2140, 545), (2300, 590),
            (2460, 600), (2620, 575), (2800, 560), (3000, 575)]
SPUR1 = [(1120, 1075), (1260, 960), (1400, 840), (1540, 775), (1660, 770), (1790, 840), (1910, 980), (2010, 1150)]
SPUR2 = [(1760, 1125), (1900, 985), (2050, 850), (2210, 760), (2380, 720), (2540, 735), (2700, 790), (2860, 830),
         (3000, 850)]
FOOT = [(2330, 1180), (2460, 1070), (2600, 995), (2760, 965), (2900, 975), (3000, 985)]
BEHIND = [(1900, 520), (2150, 432), (2330, 392), (2520, 404), (2700, 372), (2900, 398), (3000, 410)]
DISTANT = [(-100, 828), (120, 806), (330, 818), (560, 792), (800, 816), (1050, 846), (1300, 870)]
LOWHILL = [(-100, 790), (80, 772), (260, 792), (450, 836), (620, 880), (800, 925)]
BANK = [(-100, 950), (300, 962), (700, 985), (1000, 1010), (1300, 1050), (1550, 1088), (1750, 1118), (2100, 1150),
        (2500, 1172), (3000, 1195)]


def profile(pts, x):
    """A skyline through its points; off either end the hill is not there."""
    p = np.asarray(pts, float)
    y = PchipInterpolator(p[:, 0], p[:, 1], extrapolate=False)(x)
    return np.where(np.isnan(y), 1e5, y)


class Block:
    """A cherry block being cut, at `ss` times the size it prints: what is left standing takes ink."""

    def __init__(self, ss=2):
        self.ss = ss
        self.im = Image.new("L", (SW * ss, SH * ss), 0)
        self.d = ImageDraw.Draw(self.im)

    def area(self, pts, fill=255):
        p = (np.asarray(pts, float) + M) * self.ss
        if len(p) > 2:
            self.d.polygon([tuple(q) for q in p], fill=fill)

    def line(self, pts, hw, fill=255):
        p = np.asarray(pts, float)
        if len(p) < 2:
            return
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        off = np.c_[-t[:, 1], t[:, 0]] * np.broadcast_to(np.asarray(hw, float), len(p))[:, None]
        self.area(np.vstack([p + off, (p - off)[::-1]]), fill)

    def dot(self, x, y, rad, fill=255):
        s, cx, cy = self.ss, (x + M) * self.ss, (y + M) * self.ss
        self.d.ellipse([cx - rad * s, cy - rad * s, cx + rad * s, cy + rad * s], fill=fill)

    def get(self):
        return np.asarray(self.im.resize((SW, SH), Image.BOX), np.float32) / 255


def hand(pts, r, amp=0.35, step=3.0):
    """Resample a path every `step` px and let it waver a little, as a knife guided by hand does."""
    pts = np.asarray(pts, float)
    s0 = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    s = np.linspace(0, s0[-1], max(2, int(s0[-1] / step) + 1))
    p = np.c_[np.interp(s, s0, pts[:, 0]), np.interp(s, s0, pts[:, 1])]
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    k = s / 50.0
    wav = amp * (np.sin(k * r.uniform(0.5, 1.5) + r.uniform(0, 7)) + 0.5 * np.sin(k * r.uniform(2, 4) + r.uniform(0, 7)))
    return p + np.c_[-t[:, 1], t[:, 0]] * wav[:, None]


def key(blk, pts, r, hw=1.0, taper=0.04, amp=0.35, vary=0.3, step=3.0):
    """One keyblock line: the carver leaves a ridge either side of the drawn stroke, never of one width.
    `hw` is its half-width, or one for each point of the path."""
    if len(pts) < 2:
        return
    q = hand(pts, r, amp, step)
    if np.ndim(hw):
        a = np.r_[0, np.cumsum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T))]
        b = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        hw = np.interp(b / max(b[-1], 1e-9), a / max(a[-1], 1e-9), hw)
    p, w = relief.vcut(q, hw, r, taper=taper, wobble=0.15)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    swell = np.sin(s / r.uniform(20, 60) + r.uniform(0, 7)) * np.sin(s / r.uniform(80, 220) + r.uniform(0, 7))
    blk.line(p, w * (1 + vary * swell))


def on(xz, y=0.0):
    """A point (X right, Z away, in metres) at height y over the water, on the print (px), and the scale
    there in px per metre. Heights go straight up the sheet."""
    xz = np.asarray(xz, np.float64)
    k = FOC / xz[..., 1]
    return np.stack([W / 2 + k * xz[..., 0], HZ + k * (HC - np.asarray(y, float))], -1), k


def at(t, n=0.0, y=DECK):
    """A point of the bridge, t metres along it from the far end, n across from its near face, y up."""
    t, n, y = np.broadcast_arrays(*(np.asarray(v, float) for v in (t, n, y)))
    camber = 0.9 * (1 - ((t - 0.46 * SPAN) / (0.54 * SPAN)) ** 2) * (y > 1)      # the deck rises a little toward the middle
    return on(FE + t[..., None] * BD + n[..., None] * BN, y + camber)


def weight(k):
    """How heavy the carver cuts a keyline where the scale is k px to the metre: fine far off, heavy near."""
    return (np.asarray(k) / 50.0) ** 0.6


def ramp(t, stops):
    """A wiped gradation: the amount at each position t, through (position, amount) stops."""
    a = np.full(np.shape(t), stops[0][1], np.float32)
    for (p0, v0), (p1, v1) in zip(stops, stops[1:]):
        a += (v1 - v0) * noise.smoothstep(p0, p1, t)
    return a


def scallop(cx, cy, rx, ry, bump, r, wild=0.15):
    """The edge of a mass of blossom as the cutter left it: a ring of small arcs of uneven size, each
    bulging out, meeting in points. Returns the arcs in order round the mass (together they are its
    outline) and the way each one faces."""
    th = np.linspace(0, 2 * np.pi, 241)[:-1]
    rad = 1 + wild * (r.uniform(0.5, 1) * np.sin(2 * th + r.uniform(0, 7)) + r.uniform(0.3, 0.8) * np.sin(3 * th + r.uniform(0, 7))
                      + r.uniform(0.2, 0.5) * np.sin(5 * th + r.uniform(0, 7)))
    base = np.c_[cx + rx * rad * np.cos(th), cy + ry * rad * np.sin(th)]
    s = np.r_[0, np.cumsum(np.hypot(*(np.roll(base, -1, 0) - base).T))]
    bump = min(bump, s[-1] / 8)
    n = max(6, int(s[-1] / (1.2 * bump)))
    b = bump * r.lognormal(0, 0.28, n)
    step = 0.6 * (b + np.roll(b, -1))
    f = s[-1] / step.sum()
    b, pos = b * f, (np.r_[0, np.cumsum(step[:-1])] * f + r.uniform(0, s[-1])) % s[-1]
    i = np.searchsorted(s, pos) % 240
    b = b * (1 - 0.3 * np.clip(np.sin(th[i]), 0, 1))                    # finer underneath
    tan = np.roll(base, -1, 0)[i] - base[i - 1]
    out = np.c_[tan[:, 1], -tan[:, 0]] / (np.hypot(*tan.T)[:, None] + 1e-9)
    c = base[i] - 0.55 * b[:, None] * out
    gapn = np.hypot(*(np.roll(c, -1, 0) - c).T)
    b = 0.62 * np.maximum(gapn, np.roll(gapn, 1)) * r.uniform(0.95, 1.12, n)
    P = np.empty((n, 2))
    for k in range(n):                               # where each circle meets the next, on the outside
        j = (k + 1) % n
        d = c[j] - c[k]
        L = max(np.hypot(*d), 1e-6)
        a = (b[k] ** 2 - b[j] ** 2 + L ** 2) / (2 * L)
        e = np.array([-d[1], d[0]]) / L * np.sqrt(max(b[k] ** 2 - a ** 2, 0.0))
        m = c[k] + a * d / L
        P[k] = m + e if e @ (out[k] + out[j]) > 0 else m - e
    arcs = []
    for k in range(n):
        p0, p1 = P[k - 1] - c[k], P[k] - c[k]
        a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
        da = (a1 - a0) % (2 * np.pi)
        if (np.arctan2(out[k, 1], out[k, 0]) - a0) % (2 * np.pi) > da:
            da -= 2 * np.pi
        if abs(da) > 3.6:
            da -= np.sign(da) * 2 * np.pi
        t = a0 + da * np.linspace(0, 1, max(3, int(abs(da) * b[k] / 1.5)))
        arcs.append(c[k] + b[k] * np.c_[np.cos(t), np.sin(t)])
    return arcs, out


class Cutter:
    """The set of blocks a picture is cut into. Each part is cut over those before it: it clears its own
    shape out of every block, then is cut into the blocks it prints on."""

    def __init__(self, names):
        self.key = Block(SS)
        self.b = {n: Block() for n in names}
        self.cover = Block()                             # what stands in front of the land and the water
        self.front = False
        self.tone = np.zeros((SH, SW), np.float32)       # how far the deeper pink is wiped in, wherever blossom is

    def clear(self, pts):
        self.key.area(pts, 0)
        if self.front:
            self.cover.area(pts)
        for b in self.b.values():
            b.area(pts, 0)

    def wipe(self, shapes, how):
        """Wipe the deeper pink into a mass of blossom just cut: `how(X, Y, inside)` is the amount over the
        box round its shapes. A mass cut later, in front, wipes over it."""
        p = np.vstack(shapes) + M
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 2, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 3, [SW, SH])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        im = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(im)
        for q in shapes:
            d.polygon([tuple(v) for v in np.asarray(q) + M - [x0, y0]], fill=255)
        m = np.asarray(im) > 127
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - M, np.arange(y0, y1, dtype=np.float32) - M)
        there = ndimage.binary_dilation(m)
        self.tone[y0:y1, x0:x1][there] = np.clip(how(X, Y, m), 0, 1)[there]

    def fill(self, name, pts, r=None, line=0.0, **kw):
        """A flat shape on one block, over whatever lay there, with its keyline if it has one."""
        pts = np.asarray(pts, float)
        self.clear(pts)
        (self.key if name == "sumi" else self.b[name]).area(pts)
        if line:
            key(self.key, np.vstack([pts, pts[:1]]), r, line, **kw)

    def lump(self, cx, cy, rx, ry, bump, r, line=0.0, rim=0.3, dots=1.0, fill="pink", wild=0.15, bare=0.6, skip=0.0,
             wiped=True, ang=0.0):
        """A mass of blossom (or of young leaves): its scalloped shape on the colour block, the upper arcs
        of its edge in the keyblock, the lower ones often left open (`bare`), any arc now and then where
        the knife lifted (`skip`), and a scatter of flowers. Unless the mass it belongs to is wiped as a
        whole, the deeper pink is wiped up into it from its edge, full beneath and thinning upward.
        It lies at `ang` to the level. Returns its outline."""
        arcs, out = scallop(cx, cy, rx, ry, bump, r, wild)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        arcs, out = [(a - [cx, cy]) @ rot.T + [cx, cy] for a in arcs], out @ rot.T
        edge = np.vstack(arcs)
        self.clear(edge)
        self.b[fill].area(edge)
        for a, o in zip(arcs, out):
            if line and not (o[1] > 0.3 and r.random() < bare) and r.random() >= skip:
                key(self.key, a, r, line * r.uniform(0.75, 1.2), taper=0.03, amp=0.06, vary=0.25, step=1.5)
        if fill != "pink":
            return edge
        if wiped:
            def how(X, Y, m):
                low = noise.smoothstep(-0.6, 0.9, (Y - cy) / ry)
                return 0.22 * low + 0.78 * np.exp(-ndimage.distance_transform_edt(m) / (rim * min(rx, ry))) * (0.12 + 0.88 * low)
            self.wipe([edge], how)
        for _ in range(int(dots * rx * ry / 170) + 1):   # the flowers come in sprays
            a, rho = r.uniform(0, 2 * np.pi), 1 - abs(r.normal(0, 0.4))
            self.spray(*(rot @ [0.85 * rx * rho * np.cos(a), 0.85 * ry * rho * np.sin(a)] + [cx, cy]), bump, r)
        return edge

    def spray(self, x, y, bump, r):
        for _ in range(int(r.integers(2, 8))):
            self.b["dots"].dot(x + r.normal(0, 0.9) * bump, y + r.normal(0, 0.7) * bump, r.uniform(0.5, 1.15) * (0.55 + 0.085 * bump))


def tuft(cx, cy, w, t, r):
    """A flat tuft of pine needles: humped along the top, nearly straight beneath."""
    u = np.linspace(-1, 1, 21)
    env = np.sqrt(np.clip(1 - u ** 2, 0, 1))
    top = cy - t * env * (0.6 + 0.4 * np.abs(np.sin(np.pi * r.integers(2, 4) * (u + 1) / 2 + r.uniform(0, 1))))
    bot = cy + t * 0.3 * env * (1 + 0.5 * np.sin(u * r.uniform(6, 10)))
    return np.vstack([np.c_[cx + w * u, top], np.c_[cx + w * u[::-1], bot[::-1]]])


def matsu(cut, x, y, h, r, line=0.0):
    """A pine in a few cuts: a leaning, kinked trunk and a few flat tufts, no two alike."""
    lean, side = np.clip(r.normal(0, 0.15), -0.28, 0.28), r.choice([-1, 1])
    trunk = hand([[x, y], [x + 0.4 * lean * h + r.normal(0, 0.07) * h, y - 0.45 * h], [x + lean * h, y - 0.95 * h]], r, 0.2, 2.0)
    cut.b["pine"].line(trunk, np.linspace(max(0.7, 0.045 * h), max(0.4, 0.02 * h), len(trunk)))
    n = int(r.integers(2, 5)) + (h > 55) * 2
    for i, f in enumerate(np.linspace(1.0, 0.45, n)):
        w = h * r.uniform(0.2, 0.34) * (1.25 - 0.5 * f)
        p = tuft(x + lean * h * f + (0 if i == 0 else side * (-1) ** i * w * r.uniform(0.3, 0.7)),
                 y - h * f + r.normal(0, 0.03) * h, w, 0.3 * w * r.uniform(0.8, 1.2), r)
        cut.fill("pine", p, r, line, taper=0.02, amp=0.05, step=1.5)


def sugi(cut, x, y, h, r):
    """A cedar: a slim spire, notched down its sides."""
    f = np.linspace(0, 1, 9)
    half = h * r.uniform(0.15, 0.22) * (1 - f) ** 0.8 * (1 + 0.3 * (np.arange(9) % 2))
    cut.fill("pine", np.vstack([np.c_[x - half, y - h * f], np.c_[x + half[::-1], y - h * f[::-1]]]))


def sakura(cut, x, y, s, r, line=0.0, fill="pink"):
    """A small cherry on the hillside: a thin dark trunk and one or two lumps of blossom over it."""
    h = s * r.uniform(9, 20)
    lean = r.normal(0, 0.25) * h
    key(cut.key, [[x, y], [x + 0.3 * lean, y - 0.6 * h], [x + lean, y - h - s * 6]], r, max(0.45, 0.06 * s + line),
        taper=0.1, amp=0.1, step=1.5)
    for _ in range(1 if r.random() < 0.6 else 2):
        rx = s * r.uniform(12, 24)
        cut.lump(x + lean + r.normal(0, 0.3) * rx, y - h - 0.55 * rx + r.normal(0, 0.15) * rx, rx, rx * r.uniform(0.62, 0.8),
                 s * r.uniform(3.2, 4.8), r, line=line, fill=fill, dots=0.8 if fill == "pink" else 0)


def drift(cut, x, top, bot, s, line, r):
    """A drift of cherries on the mountain, over the columns `x` from `top` to `bot`. The mass of it is one
    shape on the pink block that the keyblock never touches. Along its top the crowns stand against the
    green, of every size, and the keyblock outlines most of them there, on their upper arcs, with a stroke or
    two of branch inside the larger. The deeper pink is wiped down from under the crowns and is gone by the
    foot of the drift."""
    tall = bot - top
    ht = float(tall.max())
    mass = np.vstack([np.c_[x, top + np.minimum(0.3 * tall, 3 * s)], np.c_[x[::-1], bot[::-1]]])
    cut.clear(mass)
    cut.b["pink"].area(mass)
    crowns, pos = [], x[0] - r.uniform(0, 5) * s
    while True:
        rx = s * 10 * r.lognormal(0, 0.6)
        cx = pos + 0.7 * rx
        if cx > x[-1]:
            break
        rx = float(np.clip(rx, 3.5 * s, min(40 * s, 0.6 * np.interp(cx, x, tall) + 3 * s)))
        ry = rx * r.uniform(0.6, 0.8)
        lift = ry * r.uniform(0.3, 0.9) if r.random() < 0.15 else 0.0       # a taller tree
        crowns.append((np.interp(cx, x, top) + 0.4 * ry - lift, cx, rx, ry))
        pos += rx * r.uniform(0.9, 1.4)
    shapes = [mass]
    for cy, cx, rx, ry in sorted(crowns, key=lambda c: c[0] - c[3]):
        drawn = r.random() < 0.35 + 0.5 * noise.smoothstep(4 * s, 16 * s, rx)
        shapes.append(cut.lump(cx, cy, rx, ry, float(np.clip(0.3 * rx, 2.0 * s, 6 * s)), r, line=line if drawn else 0.0,
                               bare=1.0, skip=0.15, dots=0.7, wiped=False))
        if drawn and rx > 10 * s and r.random() < 0.6:   # the branches inside, rising out of the mass
            for _ in range(int(r.integers(1, 3))):
                b0 = np.array([cx + r.normal(0, 0.25) * rx, cy + 1.1 * ry])
                b1 = b0 + [r.normal(0, 0.12) * rx, -0.75 * ry]
                key(cut.key, [b0, b1], r, 0.8 * line, taper=0.2, amp=0.1, step=1.5)
                for e in (-1, 1):
                    key(cut.key, [b1, b1 + [e * r.uniform(0.2, 0.4) * rx, -r.uniform(0.25, 0.45) * ry]], r, 0.6 * line,
                        taper=0.3, amp=0.05, step=1.0)
    deep = r.uniform(0.55, 1.0)
    cut.wipe(shapes, lambda X, Y, m: deep * ramp((Y - np.interp(X, x, top)) / ht,
                                                 [(-0.5, 0.03), (0.0, 0.15), (0.3, 0.8), (0.5, 0.85), (1.0, 0.05)]))
    for _ in range(int(np.trapezoid(tall, x) / 900)):     # flowers in the mass, more of them near the top
        fx = r.uniform(x[0], x[-1])
        cut.spray(fx, np.interp(fx, x, top) + np.interp(fx, x, tall) * r.random() ** 1.8, 3 * s, r)


def face(cut, r, vis, ridge, s, cherries, line):
    """One fold of the mountain. The cherries stand in drifts of every size: long ones lying along the
    slope and following its shape, rounder masses, knots of a few trees, and single trees in the green.
    Between the drifts are pines and cedars in companies, young leaves, and the hatching of the woods, and
    some of them stand in front of the blossom. Everything is cut from the top down, so what is lower lies
    in front."""
    if not vis.any():
        return
    ys, xs = np.nonzero(vis)
    on_ = np.flatnonzero(ridge < 5e4)
    x0, x1 = on_[0], on_[-1]
    ridge = ndimage.gaussian_filter1d(np.interp(np.arange(len(ridge)), on_, ridge[on_]), 15)     # without the trees on it
    items, cover = [], 0.0
    for _ in range(4000):                                # drifts, until they cover their share of the slope
        if cover > 0.85 * cherries * 4 * len(ys):
            break
        i = r.integers(len(ys))
        xc, yc = 2.0 * xs[i], 2.0 * ys[i]
        w = float(np.clip(s * 260 * r.lognormal(0, 0.65), 40 * s, 820 * s))
        h = float(np.clip(w * r.uniform(0.22, 0.45), 16 * s, 150 * s))
        x = np.linspace(max(xc - w / 2, x0), min(xc + w / 2, x1), max(8, int(w / 3)))
        if x[-1] - x[0] < 20 * s:
            continue
        u = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
        u = np.sign(u) * np.abs(u) ** r.uniform(0.8, 1.25)            # one end fuller than the other
        env = np.sqrt(np.clip(1 - u ** 2, 0, 1))
        mid = yc + r.uniform(0.3, 0.7) * (ridge[x.astype(int)] - ridge[int(xc)])
        top = np.maximum(mid - 0.45 * h * env ** 0.6 * (1 + 0.25 * noise.line1d(len(x), 30, r)), ridge[x.astype(int)] + 0.3 * h)
        bot = np.minimum(mid + 0.55 * h * env ** 0.9 * (1 + 0.3 * noise.line1d(len(x), 18, r)), profile(BANK, x) - 4)
        if (bot - top).max() < 8 * s:
            continue
        items.append((top.min(), "drift", x, top, np.maximum(bot, top + 2)))
        cover += np.trapezoid(np.clip(bot - top, 0, None), x)
    for i in r.integers(0, len(ys), int(len(ys) / (s * s * 2000))):     # the woods, some in front of the blossom
        kind = r.choice(["pines", "cedars", "young", "hatch", "hatch", "lone"], p=[0.26, 0.16, 0.12, 0.2, 0.2, 0.06])
        items.append((2.0 * ys[i], kind, 2.0 * xs[i]))
    for it in sorted(items, key=lambda it: it[0]):
        y, kind, x = it[:3]
        if kind == "drift":
            drift(cut, x, it[3], it[4], s, line, r)
        elif kind == "lone":
            sakura(cut, x, y, s * r.uniform(0.7, 1.2), r, line=line)
        elif kind == "pines":
            for _ in range(int(r.integers(2, 7))):
                matsu(cut, x + r.normal(0, 22) * s, y + r.normal(0, 7) * s, s * r.uniform(22, 50), r)
        elif kind == "cedars":
            for dx in np.sort(r.normal(0, 16, int(r.integers(3, 9)))) * s:
                sugi(cut, x + dx, y + r.normal(0, 4) * s, s * r.uniform(24, 52), r)
        elif kind == "young":
            for _ in range(int(r.integers(1, 4))):
                sakura(cut, x + r.normal(0, 20) * s, y + r.normal(0, 6) * s, s * r.uniform(0.7, 1.1), r, fill="moegi")
        else:
            for _ in range(int(r.integers(8, 22))):
                hx, hy = x + r.normal(0, 26) * s, y + r.normal(0, 14) * s
                cut.b["hatch"].line([[hx, hy], [hx + r.normal(0, 0.6), hy - s * r.uniform(6, 13)]], r.uniform(0.45, 0.8))


def grow(p, ang, L, w, depth, r, limbs, start=None, order=1):
    """Grow one limb of the near cherry and what springs from it: it wanders and elbows, side branches leave
    it along its length, and its leader goes on from its end. Angles are on the sheet: -pi/2 is straight up.
    A limb that leaves the trunk sets off the way the trunk was going (`start`) and bends to its own."""
    n = 8
    turn = r.normal(0, 0.1, n)
    turn[r.integers(2, n - 1)] += r.choice([-1, 1]) * r.uniform(0.25, 0.6)          # an elbow
    a = ang + np.cumsum(turn) + np.linspace(0, r.normal(0, 0.25), n)
    if start is not None:
        a += (start - ang) * np.clip(1 - np.arange(n) / 2, 0, 1)
    a = np.clip(a, -3.0, 0.25)
    pts = np.vstack([p, p + np.cumsum(np.c_[np.cos(a), np.sin(a)] * L / n, 0)])
    limbs.append((pts, w, 0.5 * w, order))
    if depth == 0:
        return
    for f in np.sort(r.uniform(0.3, 0.85, int(r.integers(1, 3)))):
        j = int(f * n)
        grow(pts[j], a[j - 1] + r.choice([-1, 1]) * r.uniform(0.45, 0.95), L * r.uniform(0.4, 0.65), w * (1 - 0.5 * f) * 0.55,
             depth - 1, r, limbs, order=order + 1)
    grow(pts[-1], a[-1] + r.normal(0, 0.3), L * r.uniform(0.6, 0.8), 0.5 * w, depth - 1, r, limbs, order=order + 1)


def bark(cut, pts, w0, w1, r, line, knots=0):
    """Cut one limb of the near cherry, or its trunk. The outline wavers, swells at its knots and narrows
    between them, never the same on its two sides. The bark block takes the whole limb; the side away
    from the light goes on the shade block, its inner edge left ragged by the gouge. The keyline is heavy
    on the shaded side and lighter on the lit one, where the knife lifts now and then. Returns what the bark
    marks cut later need: the path, its half-widths and normals, the shaded side, the heavier keyline's weight."""
    q = ndimage.gaussian_filter1d(hand(pts, r, 0.6, 2.5), 3.0, axis=0, mode="nearest")
    n = len(q)
    t = np.linspace(0, 1, n)
    w = (w0 + (w1 - w0) * t ** 0.8) * (1 + 0.1 * noise.line1d(n, 30, r))
    side = []
    for _ in range(2):
        bump = np.zeros(n)
        for c in r.uniform(0.05, 0.95, int(r.integers(0, 3)) + knots):                # knots and burls
            bump += r.uniform(0.12, 0.35) * np.exp(-((t - c) / r.uniform(0.02, 0.06)) ** 2)
        side.append(w * (1 + 0.06 * noise.line1d(n, 9, r) + bump))
    tan = np.gradient(q, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    e = 1.0 if (nrm @ [0.75, 0.65]).mean() > 0 else -1.0       # which side is away from the light
    dark, lit = q + e * nrm * side[0][:, None], q - e * nrm * side[1][:, None]
    cut.fill("bark", np.vstack([dark, lit[::-1]]))
    if w0 > 3:
        inner = 0.15 + 0.22 * noise.line1d(n, 16, r) + 0.12 * np.abs(noise.line1d(n, 3, r))
        cut.b["shade"].area(np.vstack([dark, (q + e * nrm * (w * inner)[:, None])[::-1]]))
    hw = line * (0.6 + 0.012 * w) * (1 + 0.4 * noise.line1d(n, 40, r))
    key(cut.key, dark, r, np.clip(hw, 0.5, None), taper=0.02, amp=0.15, vary=0.3, step=2.0)
    cuts = np.sort(r.uniform(0, 1, int(r.integers(0, 3))))
    for a_, b_ in zip(np.r_[0, cuts + 0.03], np.r_[cuts, 1]):                # the lit side, where the knife lifts
        k = (t >= a_) & (t <= b_)
        if k.sum() > 3:
            key(cut.key, lit[k], r, np.clip(0.6 * hw[k], 0.4, None), taper=0.06, amp=0.15, vary=0.4, step=2.0)
    return q, w, nrm, e, hw.max()


def marks(cut, q, w, nrm, e, hw, r):
    """The bark of a cherry, cut into the keyblock: short rings of lenticels across the limb, of no two
    lengths, and the eye of a knot here and there."""
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    at_ = 0.0
    while True:
        at_ += r.uniform(0.45, 1.1) * np.interp(at_, s, w)
        if at_ > s[-1] * 0.97:
            break
        j = int(np.searchsorted(s, at_))
        if w[j] < 4:
            continue
        a, b = np.sort(r.uniform(-0.9, 0.9, 2))
        if b - a < 0.35:
            a, b = np.clip(np.array([-0.5, 0.5]) * r.uniform(0.4, 1.0) + r.uniform(-0.35, 0.35), -0.9, 0.9)
        tan = np.array([nrm[j, 1], -nrm[j, 0]])
        u = np.linspace(a, b, 6)
        line_ = q[j] + nrm[j] * (e * u * w[j])[:, None] + tan * (r.normal(0, 0.12) * w[j] * (1 - (2 * (u - a) / (b - a) - 1) ** 2))[:, None]
        key(cut.key, line_, r, hw * r.uniform(0.2, 0.38), taper=0.4, amp=0.05, vary=0.3, step=1.0)
    for _ in range(int(r.integers(0, 3)) if w.max() > 12 else 0):           # the eye of a knot
        j = int(r.integers(len(q) // 6, len(q) - 3))
        c = q[j] + nrm[j] * e * r.uniform(-0.4, 0.4) * w[j]
        tan = np.array([nrm[j, 1], -nrm[j, 0]])
        th = np.linspace(0, 2 * np.pi, 24)
        rx, ry = r.uniform(0.18, 0.3) * w[j], r.uniform(0.08, 0.14) * w[j]
        cut.b["bark"].area(c + np.c_[rx * np.cos(th)] * tan + np.c_[ry * np.sin(th)] * nrm[j], fill=0)
        key(cut.key, c + np.c_[rx * np.cos(th)] * tan + np.c_[ry * np.sin(th)] * nrm[j], r, 0.45 * hw, taper=0.01, amp=0.05, step=1.0)
        cut.key.dot(*(c + 0.2 * rx * tan), 0.35 * ry)


def cherry(cut, base, fork, r, limbs_at, reach, half, line=1.6, bump=8.0):
    """A cherry in full flower, near enough to see how it is made: a trunk leaning out of the bank, swelling
    at its knots, that forks low into limbs that twist and taper and elbow as cherry limbs do, and fork
    again into twigs. The blossom hangs in clusters along the outer limbs and at the ends of the twigs,
    some behind the wood and some before it, with the sky between them."""
    base, fork = np.asarray(base, float), np.asarray(fork, float)
    mid = (base + fork) / 2 + [r.normal(0, 0.4) * half, 0]
    up = (fork - mid) / np.linalg.norm(fork - mid)
    limbs = []
    for a in limbs_at:
        grow(fork - 0.5 * half * up, a, reach * r.uniform(0.85, 1.1), half * 0.62, 2, r, limbs, start=np.arctan2(up[1], up[0]))
    sprays = []
    for pts, w0, w1, order in limbs:                     # the blossom hangs along the limbs in sprays, with gaps
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        at_, spray = s[-1] * {1: 0.5, 2: 0.25}.get(order, 0.0) + r.uniform(0, 15), []
        while at_ < s[-1] + 10:
            rx = float(np.clip(31 * r.lognormal(0, 0.5), 14, 84))
            j = min(int(np.searchsorted(s, at_)), len(pts) - 1)
            d = pts[j] - pts[max(j - 1, 0)] if j else pts[1] - pts[0]
            lie = np.clip(np.arctan2(d[1], d[0]) % np.pi - (np.pi if np.arctan2(d[1], d[0]) % np.pi > np.pi / 2 else 0), -0.6, 0.6)
            c = np.array([np.interp(at_, s, pts[:, 0]), np.interp(at_, s, pts[:, 1])]) + [r.normal(0, 0.25) * rx, r.uniform(0.0, 0.4) * rx]
            if abs(d[1]) < 0.8 * np.hypot(*d) or r.random() < 0.25:        # blossom keeps off a limb that climbs steeply
                spray.append((c, rx, rx * r.uniform(0.45, 0.65), lie + r.normal(0, 0.15)))
            if r.random() < 0.2:
                sprays.append(spray)
                spray = []
                at_ += rx * r.uniform(1.6, 2.6)
            else:
                at_ += rx * r.uniform(0.6, 1.2)
        sprays.append(spray)
    sprays = sorted((sp for sp in sprays if sp), key=lambda sp: np.mean([c[1] for c, _, _, _ in sp]))
    front = r.random(len(sprays)) < 0.35

    def hang(spray):
        """One spray: its clusters lap one another, and the deeper pink is wiped up into it from its lower edge."""
        shapes = [cut.lump(c[0], c[1], rx, ry, float(np.clip(0.16 * rx, 3.5, bump)), r, line=line * 0.5 * (r.random() < 0.55),
                           wild=0.3, bare=0.75, skip=0.3, dots=1.6, wiped=False, ang=a) for c, rx, ry, a in sorted(spray, key=lambda k: k[0][1])]
        y0 = min(c[1] - ry for c, _, ry, _ in spray)
        y1 = max(c[1] + ry for c, _, ry, _ in spray)
        reach = 0.4 * float(np.median([ry for _, _, ry, _ in spray]))

        def how(X, Y, m):
            low = noise.smoothstep(-0.2, 1.0, (Y - y0) / (y1 - y0))
            return 0.15 * low + 0.85 * np.exp(-ndimage.distance_transform_edt(m) / reach) * (0.1 + 0.9 * low)
        cut.wipe(shapes, how)

    for sp, f in zip(sprays, front):                     # the blossom behind the wood, upper first
        if not f:
            hang(sp)
    trunk = np.vstack([base, mid + [0.7 * half, 0.25 * (base - fork)[1]], mid, fork, fork + 0.7 * half * up])
    wood = [bark(cut, trunk, 1.3 * half, 0.7 * half, r, line, knots=2)]
    for pts, w0, w1, order in limbs:
        if w0 > 2.5:
            wood.append(bark(cut, pts, w0, w1, r, line))
    for q, w, nrm, e, hw in wood:                        # where a limb leaves another, no line crosses the joint
        core = np.maximum(w - 1.3 * hw, 0.25 * w)[:, None]
        cut.key.area(np.vstack([q + nrm * core, (q - nrm * core)[::-1]]), 0)
    for q, w, nrm, e, hw in wood:
        marks(cut, q, w, nrm, e, hw, r)
    for pts, w0, w1, order in limbs:                     # the twigs: one stroke of the knife each
        if w0 <= 2.5:
            key(cut.key, pts, r, line * 0.55, taper=0.3, amp=0.25, vary=0.3)
        if order >= 3:
            for a in np.arctan2(*(pts[-1] - pts[-3])[::-1]) + r.normal(0, 0.6, int(r.integers(2, 4))):
                L = reach * r.uniform(0.08, 0.16)
                tip = pts[-1] + L * np.array([np.cos(a), np.sin(a)])
                key(cut.key, [pts[-1], pts[-1] + 0.5 * L * np.array([np.cos(a + 0.3), np.sin(a + 0.3)]), tip], r, line * 0.4,
                    taper=0.35, amp=0.1, vary=0.3, step=2.0)
    for sp, f in zip(sprays, front):                     # and the blossom before it
        if f:
            hang(sp)


def rock(cut, c, w, h, r, line=1.8):
    """A rock at the water's edge, cut with character: a domed outline broken into a few hard planes under a
    heavy keyline that swells and thins; the grey wiped deeper toward the water; the plane away from the
    light on the shade block, with the strokes of its cleavage cut across it; and moss in dark dots on top."""
    k = int(r.integers(9, 13))
    th = np.sort((np.arange(k) + r.uniform(-0.3, 0.3, k)) * 2 * np.pi / k + r.uniform(0, 1))
    rad = 1 + 0.1 * r.normal(0, 1, k)
    v = np.c_[c[0] + 0.5 * w * np.cos(th) * rad, c[1] + 0.5 * h * np.sin(th) * rad * np.where(np.sin(th) < 0, 1.15, 1.0)]
    v[:, 1] = np.minimum(v[:, 1], c[1] + 0.3 * h)        # it sits flat in the water
    cut.fill("grey", v)
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    cut.wipe([v], lambda X, Y, m: ramp((Y - y0) / (y1 - y0), [(0.0, 0.1), (1.0, 0.85)]))
    it = int(np.argmin(v[:, 1]))
    ib = int(np.argmax(v[:, 1] - 0.5 * np.abs(v[:, 0] - c[0] - 0.2 * w)))
    ridge = np.array([v[it], (v[it] + v[ib]) / 2 + [r.normal(0.1, 0.06) * w, 0], v[ib]])
    ring = np.array([v[(it + j) % k] for j in range((ib - it) % k + 1)])
    cut.b["shade"].area(np.vstack([ring, ridge[1:-1][::-1]]))
    hw = line * (0.7 + 0.6 * np.abs(noise.line1d(k + 1, 3, r)))
    key(cut.key, np.vstack([v, v[:1]]), r, hw, taper=0.01, amp=0.4, vary=0.5, step=2.0)
    key(cut.key, ridge, r, 0.6 * line, taper=0.25, amp=0.4, vary=0.4, step=1.5)
    cen = ring.mean(0)
    for f in np.sort(r.uniform(0.15, 0.85, int(r.integers(2, 5)))):        # the cleavage of the shaded plane
        p0 = ridge[1] + (v[ib] - ridge[1]) * f if f > 0.5 else v[it] + (ridge[1] - v[it]) * 2 * f
        p1 = p0 + (cen - p0) * r.uniform(0.5, 1.0) + r.normal(0, 0.05, 2) * w
        key(cut.key, [p0, (p0 + p1) / 2 + r.normal(0, 0.02, 2) * w, p1], r, 0.4 * line, taper=0.4, amp=0.2, step=1.0)
    for _ in range(int(r.integers(12, 24))):             # moss along the top
        j = int(r.integers(k))
        if v[j, 1] < c[1] - 0.05 * h:
            p = v[j] + (c - v[j]) * r.uniform(0.06, 0.3) + r.normal(0, 0.05, 2) * w
            cut.b["pine"].dot(*p, r.uniform(1.4, 3.0))


def sedge(cut, foot, size, r, lean=0.0, line=0.7):
    """A tuft of grass growing from the bank: blades of every length springing from one root, fanned and
    bending outward, on the dark green block, a few of them cut in the keyblock too."""
    for _ in range(int(r.integers(5, 12))):
        a0 = -np.pi / 2 + lean + r.normal(0, 0.42)
        L = size * r.uniform(0.35, 1.0)
        u = np.linspace(0, 1, 14)
        a = a0 + np.sign(a0 + np.pi / 2 - 0.5 * lean) * r.uniform(0.1, 0.9) * u ** 1.6
        pts = foot + r.normal(0, 0.06, 2) * size + np.vstack([[0, 0], np.cumsum(np.c_[np.cos(a), np.sin(a)][:-1] * L / 13, 0)])
        cut.b["pine"].line(pts, np.linspace(r.uniform(1.5, 2.8), 0.15, len(pts)))
        if r.random() < 0.3:
            key(cut.key, pts, r, line * r.uniform(0.6, 1.0), taper=0.5, amp=0.1, vary=0.3, step=1.5)


def walker(kind, r):
    """A figure in a few cuts. Feet at 0, the top of the head at 1, facing +x. Parts in the order they cover
    one another."""
    def j(pts):
        return np.asarray(pts, float) + r.normal(0, 0.006, np.shape(pts))

    def ell(cx, cy, rx, ry, n=14):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]

    legs = [("skin", j([[-0.02, 0.42], [0.05, 0.42], [-0.1, 0.0], [-0.16, 0.0]])),
            ("skin", j([[0.03, 0.42], [0.1, 0.42], [0.2, 0.02], [0.14, 0.0]]))]
    cloth = "indigo" if r.random() < 0.6 else "red"
    if kind == "kasa":                                  # a traveller under a sedge hat, with a staff
        return legs + [(cloth, j([[0.0, 0.8], [0.14, 0.8], [0.21, 0.6], [0.17, 0.4], [-0.06, 0.4], [-0.08, 0.6]])),
                       ("line", j([[0.2, 0.64], [0.3, 0.0]])),
                       ("straw", j([[-0.16, 0.84], [0.32, 0.84], [0.1, 0.99]]))]
    if kind == "lady":                                  # a long robe, an obi, hair dressed high
        other = "red" if cloth == "indigo" else "indigo"
        return [(cloth, j([[0.0, 0.82], [0.12, 0.82], [0.18, 0.55], [0.15, 0.25], [0.25, 0.02], [-0.06, 0.0], [0.0, 0.3], [-0.05, 0.6]])),
                (other, j([[-0.03, 0.62], [0.16, 0.6], [0.16, 0.52], [-0.02, 0.53]])),
                ("skin", ell(0.07, 0.885, 0.055, 0.06)), ("sumi", ell(0.045, 0.94, 0.075, 0.05))]
    if kind == "porter":                                # two loads on a pole across the shoulder
        return legs + [(cloth, j([[0.0, 0.78], [0.13, 0.78], [0.19, 0.6], [0.16, 0.4], [-0.06, 0.4], [-0.07, 0.6]])),
                       ("skin", ell(0.07, 0.87, 0.055, 0.06)), ("sumi", ell(0.06, 0.92, 0.06, 0.03)),
                       ("line", j([[-0.5, 0.74], [0.62, 0.8]])),
                       ("fine", j([[-0.46, 0.74], [-0.46, 0.5]])), ("fine", j([[0.58, 0.8], [0.58, 0.55]])),
                       ("straw", j([[-0.56, 0.5], [-0.36, 0.5], [-0.36, 0.3], [-0.56, 0.3]])),
                       ("straw", j([[0.48, 0.55], [0.68, 0.55], [0.68, 0.36], [0.48, 0.36]]))]
    if kind in ("poler", "steer"):                      # leaning on the pole, the back leg braced
        hands = np.array([0.5, 0.58]) if kind == "poler" else np.array([0.28, 0.6])
        tip = np.array([1.55, -0.3]) if kind == "poler" else np.array([-0.9, -0.3])
        return [("skin", j([[-0.03, 0.44], [0.05, 0.44], [-0.28, 0.02], [-0.36, 0.0]])),
                ("skin", j([[0.05, 0.44], [0.13, 0.44], [0.22, 0.2], [0.16, 0.0], [0.1, 0.0], [0.14, 0.2]])),
                (cloth, j([[0.14, 0.8], [0.32, 0.74], [0.34, 0.56], [0.12, 0.42], [-0.08, 0.44], [-0.02, 0.62]])),
                ("line", j([[0.26, 0.7], hands])), ("line", np.array([tip, hands + 0.5 * (hands - tip)])),
                ("skin", ell(0.33, 0.85, 0.06, 0.065)), ("sumi", ell(0.31, 0.9, 0.065, 0.028))]
    raise ValueError(kind)


def figure(cut, kind, foot, s, d, r, hw):
    """Cut a figure standing at `foot`, `s` px tall, facing d (+1 right, -1 left)."""
    for name, p in walker(kind, r):
        q = np.c_[foot[0] + d * s * p[:, 0], foot[1] - s * p[:, 1]]
        if name in ("line", "fine"):
            key(cut.key, q, r, hw * (1.0 if name == "line" else 0.6), taper=0.1, amp=0.03, vary=0.2, step=1.0)
        else:
            cut.fill(name, q, r, hw * 0.9, taper=0.01, amp=0.03, vary=0.2, step=1.0)


def bridge(cut, r):
    """The Togetsukyō: a long plank deck on bents of three piles tied across, a railing each side. Cut from
    the far end toward us; the near railing waits until the people are on the deck."""
    ts = np.linspace(-3, SPAN, 700)
    (front, kf), (rear, kr), (face_, kc) = at(ts, 0.0), at(ts, WIDE), at(ts, 0.0, DECK - 0.55)
    bents = np.arange(3.0, SPAN, 5.6)
    bents = bents + r.normal(0, 0.2, len(bents))
    for t in bents:
        for n in (4.9, 2.7, 0.5):                        # the far pile first
            w, lean = r.uniform(0.14, 0.18), r.normal(0, 0.05)
            q, k = at([t - w, t + w, t + w + lean, t - w + lean], n, [DECK - 0.5, DECK - 0.5, -0.1, -0.1])
            cut.fill("pier", q)
            for e, hw in ((q[[0, 3]], 0.3 + 0.014 * k[0]), (q[[1, 2]], 0.25 + 0.012 * k[0])):
                if n == 0.5 or r.random() < 0.6:
                    key(cut.key, e, r, hw, taper=0.12, amp=0.15)
        for y in (DECK - 1.7, DECK - 3.1):               # the ties through the piles
            p, k = at([t, t], [0.3, 5.1], [y, y])
            cut.b["pier"].line(p, 0.09 * k.mean())
            key(cut.key, p + [0, 0.09 * k.mean()], r, 0.3 + 0.008 * k.mean(), taper=0.1, amp=0.05)
    cut.fill("tan", np.vstack([front, rear[::-1]]))
    cut.fill("beam", np.vstack([front, face_[::-1]]))
    for t in np.arange(-3, SPAN, 0.8):                   # the joints of the planks
        if r.random() < 0.55:
            p, _ = at([t, t], [r.uniform(0.3, 1.5), r.uniform(WIDE - 1.5, WIDE - 0.3)])
            cut.b["tan"].line(hand(p, r, 0.1, 2.0), 0.3, fill=0)
    for t in bents:                                      # the ends of the cross beams under the deck
        q, k = at([t - 0.16, t + 0.16, t + 0.16, t - 0.16], 0.0, [DECK - 0.55, DECK - 0.55, DECK - 0.85, DECK - 0.85])
        cut.fill("beam", q, r, 0.3 + 0.01 * k[0], taper=0.02, amp=0.05, step=1.5)
    for p, k, hw in ((front, kf, 1.25), (face_, kc, 1.1), (rear, kr, 0.85)):
        key(cut.key, p, r, hw * weight(k))
    railing(cut, r, WIDE - 0.15)


def railing(cut, r, n):
    """A railing along the bridge, n metres in from its near face: two rails, then the posts over them."""
    ts = np.linspace(-3, SPAN, 700)
    for y0, y1 in ((DECK + 0.92, DECK + 1.08), (DECK + 0.45, DECK + 0.56)):
        (a, ka), (b, kb) = at(ts, n, y0), at(ts, n, y1)
        cut.fill("tan", np.vstack([a, b[::-1]]))
        key(cut.key, a, r, 0.75 * weight(ka))
        key(cut.key, b, r, 0.7 * weight(kb))
    posts = np.arange(-3.0, SPAN, 2.3)
    for t in posts + r.normal(0, 0.06, len(posts)):
        q, k = at([t - 0.09, t + 0.09, t + 0.09, t - 0.09], n, [DECK, DECK, DECK + 1.2, DECK + 1.2])
        cut.fill("tan", q)
        for e in (q[[0, 3]], q[[1, 2]], q[[3, 2]]):
            key(cut.key, e, r, 0.3 + 0.011 * k[0], taper=0.08, amp=0.05, step=1.5)


def raft(cut, c, L, n, r, crew=("poler",)):
    """A string of logs lashed side by side, coming down with the current, c its middle on the water
    (X, Z m). The sawn ends at the back face us; the raftsmen stand forward with their poles."""
    u, v = BN, np.array([-BN[1], BN[0]])
    for i in range(n):
        a, rad = (i - (n - 1) / 2) * 0.46, 0.22 * r.uniform(0.85, 1.15)
        ends = np.array([c + u * (L / 2 + r.normal(0, 0.35)) + v * a, c - u * (L / 2 + r.normal(0, 0.3)) + v * a])
        (p0, p1), k = on(ends, rad)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        up = np.array([d[1], -d[0]]) * (1 if d[0] > 0 else -1)
        r0, r1 = rad * k[0], rad * k[1]
        cut.fill("log", np.array([p0 + up * r0, p1 + up * r1, p1 - up * r1, p0 - up * r0]))
        cut.b["shade"].area(np.array([p0 - up * r0 * 0.15, p1 - up * r1 * 0.15, p1 - up * r1, p0 - up * r0]))
        for e in (1, -1):
            key(cut.key, [p0 + e * up * r0, p1 + e * up * r1], r, 0.4 + 0.016 * k[1], taper=0.05, amp=0.1)
        a_ = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        cut.fill("tan", p1 + np.c_[0.45 * r1 * np.cos(a_), r1 * np.sin(a_)], r, 0.3 + 0.012 * k[1], taper=0.01, amp=0.03, step=1.0)
    for b in (-0.38, -0.34, 0.3, 0.34):                  # the lashings
        q, k = on(np.array([c + u * b * L + v * (n * 0.25), c + u * b * L - v * (n * 0.25)]), 0.44)
        key(cut.key, q, r, 0.35 + 0.012 * k.mean(), taper=0.05, amp=0.05)
    for m, kind in enumerate(crew):
        foot, k = on(c + u * (0.3 - 0.5 * m) * L + v * r.uniform(-0.3, 0.3), 0.42)
        figure(cut, kind, foot, 1.62 * k, -1, r, 0.3 + 0.01 * k)


def house(cut, x, y, w, r, line=0.6):
    """A tea-house on the bank: a thatched roof, hipped, over open walls."""
    cut.clear(np.array([[x, y], [x + w, y], [x + w, y - 0.42 * w], [x, y - 0.42 * w]]))
    for f in np.sort(r.uniform(0.08, 0.92, 4)):
        key(cut.key, [[x + f * w, y], [x + f * w, y - 0.4 * w]], r, line * 0.7, taper=0.05, amp=0.05, step=1.5)
    key(cut.key, [[x, y], [x + w, y]], r, line, taper=0.05, amp=0.1)
    roof = np.array([[x - 0.14 * w, y - 0.38 * w], [x + 1.14 * w, y - 0.38 * w], [x + 0.84 * w, y - 0.78 * w], [x + 0.2 * w, y - 0.8 * w]])
    cut.fill("roof", roof + r.normal(0, 0.6, roof.shape), r, line, taper=0.01, amp=0.1)


def streams(r):
    """The current, in hanks of long lines that follow the water: hairlines far off, long and bold near.
    They are cut out of the blue where it is deep (and print as paper) and left standing on a block of
    their own where it is pale (and print dark). With them the petals, drifting in rafts."""
    cuts, petals = [], []
    v = np.array([-BN[1], BN[0]])
    f = np.linspace(0, 1, 80)

    def hank(c, L, n, gap, hw):
        """A handful of lines cut side by side: they wave together, bow a little, and draw closer at one end."""
        turn = r.normal(0, 0.05)
        u = np.array([BN[0] * np.cos(turn) - BN[1] * np.sin(turn), BN[0] * np.sin(turn) + BN[1] * np.cos(turn)])
        far = c[1] / 60
        phase, lam, amp, bow, fan = r.uniform(0, 7), r.uniform(9, 20), r.uniform(0.5, 1.5) * far, r.normal(0, 4) * far, r.uniform(-0.9, 0.9)
        for j in range(n):
            s0 = r.uniform(-0.5, -0.1)
            s = np.linspace(s0, s0 + r.uniform(0.45, 0.95), 80) * L
            off = (j - (n - 1) / 2) * gap * (1 + fan * s / L) * r.uniform(0.85, 1.15) + amp * np.sin(s / lam + phase) + bow * (s / L) ** 2
            p, k = on(c + s[:, None] * u + off[:, None] * np.array([-u[1], u[0]]))
            w = hw * k * r.lognormal(0, 0.3) * noise.smoothstep(0, r.uniform(0.04, 0.2), f) \
                * noise.smoothstep(1, 1 - r.uniform(0.25, 0.7), f) ** 0.8 * (1 + 0.25 * noise.line1d(80, 12, r))
            q = hand(p, r, 0.5, 4.0)
            cuts.append((q + M, np.interp(np.linspace(0, 1, len(q)), f, np.clip(w, 0, 2.8))))

    for x, y, L, n, gap, hw in ((2350, 1935, 56, 8, 0.6, 0.04), (1250, 1915, 54, 7, 0.65, 0.04), (640, 1965, 40, 5, 0.6, 0.036),
                                (1900, 1775, 50, 6, 0.75, 0.036), (950, 1745, 44, 5, 0.8, 0.034), (2600, 1670, 40, 5, 0.9, 0.034),
                                (1500, 1590, 44, 4, 1.0, 0.032), (760, 1575, 36, 3, 1.0, 0.03), (2300, 1445, 46, 4, 1.3, 0.032),
                                (1250, 1405, 40, 3, 1.4, 0.03), (1950, 1300, 44, 3, 1.8, 0.032), (880, 1320, 36, 3, 1.8, 0.03),
                                (2550, 1255, 50, 3, 2.2, 0.034), (1480, 1205, 50, 3, 2.6, 0.034), (560, 1160, 60, 3, 3.2, 0.04),
                                (2150, 1215, 50, 2, 3.0, 0.034), (300, 1065, 90, 3, 5.0, 0.055), (1000, 1085, 80, 2, 5.0, 0.05)):
        Z = FOC * HC / (y - HZ)
        hank(np.array([(x - W / 2) * Z / FOC, Z]), L, n, gap, hw)
    for _ in range(40):                                  # stray ripples between them, singly or in pairs
        Z = 52 + 250 * r.random() ** 1.6
        hank(np.array([r.uniform(-0.56, 0.56) * Z, Z]), r.uniform(5, 14), int(r.integers(1, 3)), 0.02 * Z, r.uniform(0.02, 0.03))
    for t in np.arange(3.0, SPAN, 5.6):                  # the water parting round the piles
        for n in (0.5, 2.7):
            foot = FE + t * BD + n * BN
            for e in (-1, 1):
                L = r.uniform(2.5, 6)
                sgo = np.linspace(0.2, L, 20)
                p, k = on(foot + sgo[:, None] * BN + (e * (0.25 + 0.1 * sgo))[:, None] * v)
                cuts.append((p + M, np.clip(0.03 * k * np.sin(np.linspace(0.3, np.pi, 20)), 0, 2.2)))
    long = [(1500, 1335, 46), (880, 1470, 34), (2150, 1570, 40), (1250, 1690, 44), (2450, 1330, 36), (640, 1250, 40)]
    for i in range(66):                                  # the rafts of petals: under the near tree, in long drifts, and astray
        Z = 46 + 190 * r.random() ** 1.3
        c = np.array([r.uniform(-0.56, 0.56) * Z, Z]) if i > 14 else np.array([r.uniform(-58, -22), r.uniform(46, 66)])
        L, wd = r.uniform(6, 24), r.uniform(0.8, 2.6) * (Z / 80) ** 0.5
        if i >= 60:
            x, y, L = long[i - 60]
            Z = FOC * HC / (y - HZ)
            c, wd = np.array([(x - W / 2) * Z / FOC, Z]), r.uniform(0.8, 1.6) * (Z / 80) ** 0.5
        for _ in range(int(L * wd * r.uniform(1.5, 4))):
            s = r.normal(0, 0.3) * L
            (x, y), k = on(c + s * BN + r.normal(0, 1) * wd * (1 - 0.7 * min(abs(s) / L, 1)) * v)
            petals.append((x, y, np.clip(0.055 * k * r.lognormal(0, 0.35), 0.7, 3.2)))
    return relief.cuts((SH, SW), cuts), petals


def band(x0, x1, yc, hw, r):
    """A band of mist laid across the view, rounded at its ends and a little wavy along its edges: its outline."""
    x = np.linspace(x0, x1, 240)
    u = (x - x0) / (x1 - x0)
    half = hw * np.sqrt(np.clip(1 - (2 * u - 1) ** 8, 0, 1)) * (1 + 0.15 * noise.line1d(240, 30, r))
    c = yc + 4 * noise.line1d(240, 50, r)
    return np.vstack([np.c_[x, c - half], np.c_[x, c + half][::-1]])


def baren(shape, r, n=1400):
    """The rub of the baren: overlapping circling strokes, each ridged by the coil of cord inside it."""
    h, w = shape
    im = Image.new("F", (w, h), 0.0)
    d = ImageDraw.Draw(im)
    for cx, cy, q, a0, span, v in zip(r.uniform(0, w, n), r.uniform(0, h, n), r.uniform(40, 140, n),
                                      r.uniform(0, 360, n), r.uniform(80, 240, n), r.normal(0, 1, n)):
        for k in range(4):
            d.arc((cx - q - 8 * k, cy - q - 8 * k, cx + q + 8 * k, cy + q + 8 * k), a0, a0 + span, fill=float(v), width=2)
    rub = ndimage.gaussian_filter(np.asarray(im), 2.5)
    return rub / rub.std()


def grain(r, period=15.0):
    """The grain of a flat-sawn cherry plank as the ink finds it: fine growth lines along the board, now
    crowded and now spread, waving slowly and fading in and out."""
    rings = np.cumsum(np.clip(1 + 0.4 * noise.line1d(SH + 200, 60, r), 0.3, None)) / period
    ph = np.interp(np.arange(SH)[:, None] + 100 + 6 * noise.fbm((SH, SW), 800, r, octaves=2)
                   + 0.8 * noise.field((SH, SW), 40, r), np.arange(SH + 200), rings)
    g = np.exp(-((ph % 1 - 0.5) / 0.12) ** 2) * noise.smoothstep(-0.8, 1.2, noise.stretched((SW, SH), 8, 400, r).T)
    return (g - g.mean()) / g.std()


def impress(img, press, r, block, amount, ink, shift, grain_=0.05, salt=0.3, squash=0.1, rub=0.045):
    """One impression. The block is brushed with pigment and wiped, the damp sheet laid to the kento and
    rubbed with the baren. Thin ink reaches only the tops of the fibres, so the pale ends of a gradation
    come out salted; `salt` is how dry the printer kept this block."""
    sheet, rubs, pits, woods = press
    ys, xs = np.flatnonzero(block.max(1) > 0.004), np.flatnonzero(block.max(0) > 0.004)
    if not len(ys):
        return
    at_ = np.s_[max(ys[0] - 8, 0):ys[-1] + 9, max(xs[0] - 8, 0):xs[-1] + 9]      # the block is only so big
    block = block[at_]
    amt = np.clip(amount[at_] if np.ndim(amount) else amount, 0, None) * np.ones(block.shape, np.float32)
    here = noise.smoothstep(0.2, 1.2, noise.field(block.shape, 380, r))      # the grain shows here and there
    wood = np.roll(woods[r.integers(len(woods))], int(r.integers(0, 400)), 0)[at_]
    film = relief.ink_film(block, r, roller=0.0, squash=squash, grain=wood * (0.4 + here), grain_strength=grain_) * amt
    b = np.roll(rubs[r.integers(len(rubs))], tuple(r.integers(0, 500, 2)), (0, 1))[at_]
    dens = relief.pull(block, film, SimpleNamespace(tooth=sheet.tooth[at_]), r, pressure=1.6 + 0.05 * b, shift=shift,
                       turn=r.normal(0, 1.5e-4))
    dens *= (1 + rub * b) * (1 - salt * np.clip(1 - 2 * amt, 0, 1) * np.roll(pits, tuple(r.integers(-1, 2, 2)), (0, 1))[at_])
    img[at_] = glaze(img[at_], ndimage.gaussian_filter(dens, 0.7), ink)


def paint(seed=1834):
    r = noise.rng(seed)
    sheet = paper.washi((SH, SW), seed, tint="#f2e9d5", margin=(40, 38), fibre_density=0.6)
    sheet.alpha = paper.deckle((SH, SW), (40, 38), r, ragged=2.0)
    pits = noise.smoothstep(0.32, 0.14, sheet.tooth)        # the valleys between the fibres, that thin ink misses
    press = sheet, [baren((SH, SW), r) for _ in range(3)], pits, [grain(r), grain(r, 11.0)]
    X = np.arange(SW, dtype=np.float32) - M
    Y = np.arange(SH, dtype=np.float32)[:, None] - M
    inside = (noise.smoothstep(-0.5, 0.5, Y) * noise.smoothstep(-0.5, 0.5, H - Y)
              * noise.smoothstep(-0.5, 0.5, X[None, :]) * noise.smoothstep(-0.5, 0.5, W - X[None, :]))
    wob = 9 * noise.field((SH, SW), 110, r)
    rows = Y + 8 * noise.line1d(SW, 260, r)[None, :]

    def hill(pts, fringe=3.0, wave=5.0):
        y = profile(pts, X) + wave * noise.line1d(SW, 170, r) \
            - fringe * (np.abs(noise.line1d(SW, 8, r)) + 0.6 * np.abs(noise.line1d(SW, 21, r)))
        return y, noise.smoothstep(-0.8, 0.8, Y - y[None, :])

    def raster(*polys):
        b = Block()
        for q in polys:
            b.area(q)
        return b.get()

    bank = profile(BANK, X) + 2.5 * noise.line1d(SW, 90, r)
    water = noise.smoothstep(-0.7, 0.7, Y - bank[None, :])
    _, behind = hill(BEHIND, 2.0, 4.0)
    _, distant = hill(DISTANT, 1.0, 5.0)
    low_y, low = hill(LOWHILL, 2.5, 4.0)
    # the bands of mist: cut clear of every block, a little warm glow wiped along the top of each
    mists = {"mistA": [(1330, 2260, 724, 23), (1880, 2640, 700, 15)],
             "mistB": [(-60, 1430, 920, 25), (-60, 760, 897, 13)],
             "mistC": [(1240, 1810, 982, 13), (2090, 2960, 906, 16)]}
    mists = {n: [(band(*b, r), b[2], b[3]) for b in bs] for n, bs in mists.items()}
    mistm = {n: raster(*(q for q, _, _ in bs)) for n, bs in mists.items()}
    glow = np.maximum.reduce([raster(q) * ramp((Y - yc + hw) / (2 * hw), [(-0.1, 0.15), (0.15, 0.4), (1.0, 0.0)])
                              for bs in mists.values() for q, yc, hw in bs])

    # the mountain, fold over fold: on each the darker green is wiped down from its ridge, and each is cut
    # with its trees before the fold in front of it is cut across them
    cut = Cutter(["pink", "dots", "pine", "moegi", "hatch", "bark", "tan", "beam", "pier", "roof", "log", "shade", "red",
                  "indigo", "skin", "straw", "grey"])
    folds = [("main", ARASHI, 0.68, 0.8, 4.0, 0.5), ("shoulder", SHOULDER, 0.78, 0.9, 3.5, 0.75), ("mistA",), ("mistB",),
             ("spur1", SPUR1, 0.95, 1.0, 3.5, 0.85), ("spur2", SPUR2, 1.0, 1.0, 3.5, 0.8), ("mistC",),
             ("foot", FOOT, 1.15, 1.0, 3.0, 0.9)]
    ridges = {f[0]: hill(f[1], f[4]) for f in folds if len(f) > 1}
    mist, mt, deep, green = (np.zeros((SH, SW), np.float32) for _ in range(4))
    for i, f in enumerate(folds):
        if len(f) == 1:
            m = mistm[f[0]]
            mist, deep, green = np.maximum(mist, m), deep * (1 - m), green * (1 - m)
            for q, _, _ in mists[f[0]]:
                cut.clear(q)
            continue
        name, pts, s, strength, _, cherries = f
        y, m = ridges[name]
        there = y < 5e4
        cut.clear(np.vstack([np.c_[X, y][there], [[X[there][-1], H + 200], [X[there][0], H + 200]]]))
        mt, mist = np.maximum(mt, m), mist * (1 - m)
        drop = Y - y[None, :] + wob
        deep = deep * (1 - m) + m * strength * ramp(drop, [(-20, 1.0), (30, 0.88), (170 + 90 * strength, 0.0)])
        green = green * (1 - m) + m * (0.55 + 0.3 * strength) * ramp(drop, [(0, 1.0), (500, 0.75)])
        # where this fold shows: not under a fold or a band of mist in front of it (a tree may stand a
        # little way behind the next ridge, and show over it)
        vis = (m * (1 - water))[M:M + H:2, M:M + W:2] > 0.5
        for g in folds[i + 1:]:
            vis &= ~((mistm[g[0]] if len(g) == 1 else np.roll(ridges[g[0]][1], 8, 0))[M:M + H:2, M:M + W:2] > 0.5)
        ty = y[M:M + W]
        face(cut, noise.rng(1000 * HILL + i), vis, ty, s, cherries, 0.24 + 0.18 * s)
        x = 1050.0                                       # a file of pines along the ridge, here and there
        while x < W:
            if ty[int(x)] < 5e4 and noise.smoothstep(-0.2, 0.6, np.sin(x / 130 + i * 2.1) + r.normal(0, 0.4)) > r.random():
                matsu(cut, x, ty[int(x)] + 5, s * r.uniform(18, 40), r)
            x += s * r.uniform(12, 60)

    cut.front = True
    # the foot of the mountain along the water: cherries in drifts over a file of trunks standing on the
    # bank, a pine or young leaves between them, a tea-house or two by the end of the bridge
    xa = 1090.0
    while xa < 2900:
        xb = min(xa + r.uniform(120, 420), 2990.0)
        if xa < 1990 and xb > 1790:                      # the tea-houses stand clear
            xa, xb = (1990.0, 1990 + r.uniform(120, 300)) if xa > 1700 else (xa, 1790.0)
        x = np.linspace(xa, xb, int((xb - xa) / 3))
        u = np.linspace(-1, 1, len(x))
        bank_ = profile(BANK, x)
        bot = bank_ - 18 - 8 * noise.line1d(len(x), 30, r)
        top = bot - r.uniform(40, 80) * np.sqrt(np.clip(1 - u ** 2, 0, 1)) ** 0.5 * (1 + 0.25 * noise.line1d(len(x), 40, r)) - 6
        drift(cut, x, top, np.maximum(bot, top + 2), 2.2, 0.5, r)
        for xt in np.arange(xa + r.uniform(4, 16), xb - 4, 1.0):
            if r.random() < 0.05:                        # the trunks they stand on
                yt, lean = np.interp(xt, x, bot), r.normal(0, 4)
                key(cut.key, [[xt, profile(BANK, xt) - 2], [xt + 0.5 * lean, yt], [xt + lean, yt - r.uniform(10, 26)]], r,
                    r.uniform(0.5, 0.75), taper=0.15, amp=0.1, step=1.5)
        gap = r.uniform(20, 110)
        if r.random() < 0.6:
            g = xb + 0.5 * gap
            if r.random() < 0.5:
                matsu(cut, g, profile(BANK, g) - 4, r.uniform(60, 110), r, line=0.42)
            else:
                sakura(cut, g, profile(BANK, g) - 6, r.uniform(1.6, 2.4), r, fill="moegi")
        xa = xb + gap
    house(cut, 1812, 1112, 86, r, line=0.5)
    house(cut, 1905, 1124, 64, r, line=0.5)
    for x in np.sort(r.uniform(-20, 1060, 60)):          # the far bank downstream, small under the mist
        yb, s = profile(BANK, x) - 1, 0.45 + 0.35 * x / 1060
        if r.random() < 0.3:
            sugi(cut, x, yb, s * r.uniform(20, 44), r)
        else:
            rx = s * r.uniform(12, 24)
            cut.lump(x, yb - 0.7 * rx, rx, 0.7 * rx, s * 4, r, fill="pink" if r.random() < 0.55 else "moegi", dots=0.5)
    for x, w in ((230, 30), (420, 24), (560, 34), (890, 30)):
        house(cut, x, profile(BANK, x) - 2, w, r, line=0.3)

    bridge(cut, r)
    for t, n, kind, d in ((14, 2.0, "kasa", -1), (27, 3.4, "lady", 1), (30.5, 3.0, "lady", 1), (46, 1.8, "porter", -1),
                          (61, 3.2, "kasa", 1), (72, 2.2, "lady", -1), (88, 3.5, "kasa", -1)):
        foot, k = at(t, n, DECK)
        figure(cut, kind, foot, 1.58 * k, d, r, 0.28 + 0.01 * k)
    railing(cut, r, 0.15)
    raft(cut, np.array([-96.0, 262.0]), 9.0, 4, r)
    raft(cut, np.array([10.0, 126.0]), 11.0, 5, r)
    raft(cut, np.array([26.0, 80.0]), 15.0, 7, r, crew=("poler", "steer"))

    # the near bank, and the cherry that stands on it
    edge_x = np.linspace(-90, 700, 160)
    edge_y = 1700 + 390 * np.clip(edge_x / 560, 0, 1.5) ** 1.5 + 9 * noise.line1d(160, 40, r) + 3 * noise.line1d(160, 9, r)
    lip = np.c_[edge_x, edge_y]
    shore = np.vstack([lip, [[-90, H + 90]]])
    cut.clear(shore)
    near_bank = raster(shore)
    key(cut.key, lip[4:], r, np.linspace(1.3, 2.0, len(lip) - 4), taper=0.05, amp=0.8, vary=0.5)
    stones = ((30, 150, 92), (88, 118, 70), (104, 62, 38))
    for i, w, h in stones:
        rock(cut, lip[i] + [0, 0.22 * h], w, h, r)
    for i in range(6, 150):                              # grass along the lip, leaning out over the water
        if r.random() < 0.45 and all(abs(i - j) > 0.05 * w for j, w, _ in stones):
            sedge(cut, lip[i] + [0, 3], r.uniform(34, 80), r, lean=0.35)
    for x, y in ((-10, 1905), (30, 1985), (215, 1995), (300, 1990)):        # and lush tufts at the foot of the print
        sedge(cut, np.array([x, y]) + r.normal(0, 8, 2), r.uniform(70, 110), r, lean=r.normal(0.15, 0.1))
    for i in (12, 44, 78):                               # reeds standing in the shallows
        for _ in range(int(r.integers(5, 11))):
            a, L = r.normal(0.12, 0.25), r.uniform(50, 120)
            foot = np.array([edge_x[i] + r.normal(8, 9), edge_y[i] + r.uniform(-4, 12)])
            blade = np.array([foot, foot + [0.35 * L * np.sin(a), -0.55 * L], foot + [L * np.sin(a) * 1.25, -L * np.cos(a)]])
            q = hand(np.c_[np.interp(np.linspace(0, 2, 16), [0, 1, 2], blade[:, 0]), np.interp(np.linspace(0, 2, 16), [0, 1, 2], blade[:, 1])], r, 0.4, 3.0)
            cut.b["pine"].line(q, np.linspace(1.5, 0.25, len(q)))
            if r.random() < 0.6:                         # a leaf, bent over
                j = int(r.integers(len(q) // 3, len(q) - 2))
                cut.b["pine"].line([q[j], q[j] + [r.choice([-1, 1]) * r.uniform(10, 22), r.uniform(-4, 8)]], [1.1, 0.2])
    cherry(cut, (120, 2080), (335, 1605), noise.rng(TREE), (-2.45, -1.75, -1.15, -0.6, -0.2), 300, 35, line=1.6, bump=9.0)

    gouge, petals = streams(r)
    pb = Block()
    for x, y, rad in petals:                              # a petal lies flat on the water
        pb.d.ellipse([(x + M - 1.25 * rad) * pb.ss, (y + M - 0.7 * rad) * pb.ss, (x + M + 1.25 * rad) * pb.ss, (y + M + 0.7 * rad) * pb.ss],
                     fill=255)
    frame = Block(SS)
    for a, b in zip([[0, 0], [W, 0], [W, H], [0, H]], [[W, 0], [W, H], [0, H], [0, 0]]):
        a, b = np.array(a, float), np.array(b, float)
        key(frame, np.linspace(a - (b - a) * 0.004, b + (b - a) * 0.004, 2), r, hw=1.8, taper=0.01, amp=0.5)

    B = {n: blk.get() for n, blk in cut.b.items()}
    keyb = cut.key.get()
    blossom = np.clip(B["pink"], 0, 1)
    solid = np.clip(sum(B[n] for n in ("tan", "beam", "pier", "roof", "log", "bark", "red", "indigo", "skin", "straw", "grey")), 0, 1)
    things = np.clip(solid + blossom + B["moegi"] + B["pine"], 0, 1)
    cover = cut.cover.get()
    petal = pb.get() * water * (1 - cover)

    # the cartouche and the series title beside it, cut clear of the sky. The lettering was cut once with
    # lettering.cartouche("嵐山花筏", "arashiyama_title", signature="克勞德筆", size=78, seed=34) and
    # lettering.cartouche("京都名所之内", "arashiyama_series", size=58, seed=35)
    card, letters, tint = (np.zeros((SH, SW), np.float32) for _ in range(3))
    for name, (x0, y0), box, fill in (("arashiyama_title", (108, 92), (109, 255, 2, 406), False),
                                      ("arashiyama_series", (410, 96), (8, 116, 2, 424), True)):
        t = Image.open(lettering.CUT / f"{name}.png")
        t = np.asarray(t.resize((int(t.width * 1.12), int(t.height * 1.12)), Image.LANCZOS), np.float32) / 255
        at_ = np.s_[M + y0:M + y0 + t.shape[0], M + x0:M + x0 + t.shape[1]]
        letters[at_] = np.maximum(letters[at_], t)
        bx0, bx1, by0, by1 = (int(v * 1.12) for v in box)
        card[M + y0 + by0:M + y0 + by1, M + x0 + bx0:M + x0 + bx1] = 1
        if fill:
            tint[M + y0 + by0 + 6:M + y0 + by1 - 6, M + x0 + bx0 + 6:M + x0 + bx1 - 6] = 1

    land = np.clip(np.maximum.reduce([mt, behind, distant, low]), 0, 1) * (1 - water)
    sky = (1 - np.maximum(land, water)) * (1 - card)
    slope = mt * (1 - water) * (1 - blossom) * (1 - B["moegi"]) * (1 - cover)
    lowland = low * (1 - mt) * (1 - water) * (1 - mist) * (1 - cover)
    river = water * (1 - cover) * (1 - np.clip(blossom + B["moegi"] + B["pine"], 0, 1))
    depth = Y - bank[None, :] + 0.6 * wob                 # how far below the far bank
    reach = 110 + 270 * noise.smoothstep(1500, 150, X)[None, :]
    blue = np.maximum(ramp(rows, [(1540, 0.0), (1830, 0.6), (2000, 1.05)]), ramp(depth / reach, [(0, 0.85), (0.25, 0.62), (1.0, 0.0)]))
    under = cut.tone * blossom
    lie_ = Y - 1700 - 390 * np.clip(X[None, :] / 560, 0, 1.5) ** 1.5 + 2 * wob

    blocks = [
        (sky, ramp(rows, [(0, 1.08), (60, 0.95), (250, 0.3), (370, 0.0)]), PRUSSIAN, (0.0, 0.0), 0.04, 0.5),
        (sky, ramp(rows, [(470, 0.0), (760, 0.5), (900, 0.7)]), GLOW, (1.2, -0.6), 0.03, 0.4),
        (mist * (1 - cover) * (1 - blossom), glow, GLOW, (-0.8, 0.6), 0.03, 0.3),
        (np.maximum(behind, distant) * (1 - mt) * (1 - low) * (1 - water) * (1 - mist) * (1 - cover),
         ramp(rows, [(360, 0.95), (520, 0.5), (780, 0.7), (900, 0.15)]), FAR, (0.9, 0.7), 0.03, 0.3),
        (slope, green, GREEN, (-1.1, 0.4), 0.06, 0.3),
        (lowland, 0.6, GREEN, (0.5, 0.7), 0.05, 0.3),
        (slope, deep, DEEP, (1.4, -0.9), 0.06, 0.35),
        (lowland, ramp(Y - low_y[None, :] + wob, [(0, 0.75), (70, 0.0)]), DEEP, (0.6, 1.1), 0.05, 0.35),
        (near_bank * (1 - things), 0.5, MOEGI, (1.1, 0.9), 0.06, 0.3),
        (near_bank * (1 - things), ramp(lie_, [(0, 0.2), (110, 0.85)]), GREEN, (-0.5, 1.0), 0.06, 0.35),
        (near_bank * (1 - things), ramp(lie_, [(15, 0.0), (190, 0.85), (300, 1.1)]), DEEP, (0.8, -0.4), 0.06, 0.35),
        (B["hatch"] * slope, 0.55, DEEP, (0.4, 0.5), 0.0, 0.2),
        (B["moegi"] * inside, 0.9, MOEGI, (-0.7, -1.2), 0.05, 0.3),
        (B["pine"] * inside, 0.95, PINE, (1.0, 0.8), 0.05, 0.2),
        ((blossom + tint + petal) * inside, ramp(rows, [(300, 0.46), (1100, 0.54), (2000, 0.6)]) * (0.62 + 0.38 * np.sqrt(under) + tint + petal),
         BENI, (0.6, -0.5), 0.04, 0.3),
        (blossom * inside, 0.8 * under, ROSE, (-0.9, 0.5), 0.03, 0.4),
        ((B["dots"] * blossom + petal) * inside, 0.85 * (1 - petal) + 0.42 * petal * (1 - noise.smoothstep(0.1, 0.4, blue)), ROSE,
         (0.3, 1.0), 0.0, 0.2),
        (river * (1 - petal) * (1 - gouge * noise.smoothstep(0.12, 0.4, blue)),
         ramp(depth / reach, [(0, 0.8), (1.3, 0.3), (3.5, 0.22)]) + ramp(rows, [(1500, 0.0), (2000, 0.35)]), AI,
         (0.7, -0.4), 0.04, 0.4),
        (river * (1 - gouge) * (1 - petal), blue, PRUSSIAN, (-0.6, 0.8), 0.04, 0.5),
        (river * gouge, 0.8 * (1 - noise.smoothstep(0.12, 0.4, blue)), PRUSSIAN, (0.4, 0.4), 0.0, 0.2),
        (B["tan"] * inside, 0.8, TAN, (1.8, 0.6), 0.08, 0.3),
        (B["beam"] * inside, 0.7, TAN, (-0.5, 0.9), 0.08, 0.3),
        (B["beam"] * inside, 0.45, PIER, (0.9, -0.6), 0.05, 0.25),
        (B["pier"] * inside, 0.85, PIER, (0.9, -0.6), 0.06, 0.25),
        (B["roof"] * inside, 0.7, PIER, (0.7, 0.8), 0.08, 0.3),
        (B["grey"] * inside, 0.55, GREY, (-0.8, 0.5), 0.06, 0.3),
        (B["grey"] * inside, 0.5 * cut.tone, INDIGO, (0.7, -0.6), 0.05, 0.35),
        (B["log"] * inside, 0.9, LOG, (1.5, -1.1), 0.1, 0.25),
        (B["shade"] * inside, 0.55, PIER, (-0.7, 1.2), 0.05, 0.25),
        (B["bark"] * inside, 0.9, BARK, (-1.2, 0.9), 0.1, 0.25),
        (B["red"] * inside, 0.85, RED, (-1.4, -1.6), 0.03, 0.2),
        (B["indigo"] * inside, 0.85, INDIGO, (-1.0, 1.4), 0.03, 0.2),
        (B["skin"] * inside, 0.6, SKIN, (0.5, -0.9), 0.0, 0.0),
        (B["straw"] * inside, 0.75, STRAW, (-0.6, 0.9), 0.03, 0.2),
        (np.maximum(keyb * inside, np.maximum(frame.get(), letters)), 1.0, SUMI, (0.0, 0.0), 0.03, 0.15),
    ]
    img = sheet.color.copy()
    pressed = []
    for block, amount, ink, shift, gr, salt in blocks:
        impress(img, press, r, block, amount, ink, shift, grain_=gr, salt=salt)
        pressed.append(np.clip(block, 0, 1))
    img = relief.emboss(img, pressed, depth=0.015)
    return plate.mount(img, sheet, shadow=0.35)
