"""An Evening in May. Opalescent glass in copper foil and lead, with drapery, ripple and confetti glass, plated in places.

In the first years of the century Louis Comfort Tiffany's studios in New York made landscape windows that
are painted almost without paint: the picture is in the glass. Opalescent glass is milky and streaked, two
or three colours swirled through one sheet, and the glazier chose every piece for where its streaks fall.
A sky goes from gold to cobalt because each piece was cut from the right part of a sheet. Glass folded
while it was hot, drapery glass, makes the petals of a flower; glass rolled with ripples makes water; glass
with chips of colour rolled into it makes trees seen far off; a second, milky sheet laid behind the first
pushes a distance back.

This window is a new design. A magnolia in flower stands on the left and a wisteria hangs from the top right.
Between them the sun has gone down behind two ranges of hills, and the river comes toward us with the gold
of the sunset broken down it. The sky is cut in a few broad strips, each piece taken where the streaks of
its sheet run strongest: cobalt and ultramarine streaked with white overhead, white and pale blue below
them, then cream, gold and amber low over the hills, crossed by thin violet clouds. The far range is
plated with a second, milky sheet; the near range is deep violet-blue, the water is darker than the sky,
and the banks are mottled and folded greens, so that the whites and golds burn against them. Every
magnolia petal is one piece of white drapery glass with its folds running from the claw to the tip,
flushed rose toward the base; the flowers are cups turned every way, a few open wide, and furred grey
buds. The irises run from deep violet to blue-white, open, opening, in profile and in bud, the falls in
drapery glass with a paler haft and a gold beard. About 2,700 pieces are wrapped in copper foil and
soldered, a fine line through the sky and the water and a heavy lead round the forms, and four iron bars
hold the window against the wind.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import glass, noise
from atelier.color import lin, pigment

TITLE = "An Evening in May"
DATE = "2026"
MEDIUM = ("Opalescent glass, streaky, mottled, drapery, ripple, confetti and streamer, plated in places, in copper "
          "foil and lead, with iron bars")
AFTER = ("Louis Comfort Tiffany and Tiffany Studios, the landscape windows in opalescent glass, New York, "
         "c. 1900–1925: Magnolias and Irises, c. 1908; Autumn Landscape, 1923–24; the windows for Laurelton Hall")
ROOM = "The Workshop"
YEAR = 1908
PLACE = "New York"
REGION = "Americas"
NOTE = ("A magnolia and a wisteria in flower frame a sunset over blue hills, and irises stand at the edge of the "
        "river. Nothing is painted: every colour is in the glass, each piece chosen for where its streaks fall.")

H, W = 2800, 2000
FR = 44                                   # the bronze frame
X0, Y0, X1, Y1 = FR, FR, W - FR, H - FR   # the glass
BARS = [Y0 + (Y1 - Y0) * k / 5 for k in (1, 2, 3, 4)]
SHORE = 1648.0                            # the far shore, at the foot of the hills
SUN = 1000.0                              # where the sun went down, behind the valley
UP = -np.pi / 2
X = np.arange(W, dtype=np.float32)


@dataclass
class Sheet:
    """One kind of sheet in the racks: a run of colours, palest first, and how it was made."""
    colours: tuple
    fine: float = 0.15       # thin streaks of the next colour
    broad: float = 0.3       # broad swirls
    white: float = 0.4       # streaks of white opal
    edge: float = 0.3        # how sharply one colour gives way to the next
    thick: float = 0.15      # spread of thickness from piece to piece (log)
    spots: float = 0.0       # mottled glass: the lighter crystal spots
    folds: float = 0.0       # drapery glass, rolled thick and folded while hot
    ripple: float = 0.0      # ripple glass
    flakes: float = 0.0      # confetti glass: chips of colour rolled into the sheet
    jit: float = 0.2         # how far a piece strays from the tone the cartoon asked for
    stretch: float = 1.0     # how far the streaks were drawn out along the sheet as it was rolled
    lead: float = 1.4        # half-width of the line between two pieces of one part of the picture, px
    rim: float = 0.0         # half-width of the lead where this part meets another; the heavier side wins
    part: str = ""           # which part of the picture: the sky, the water, a flower, a bank


HILLS = ("#9a96d8", "#726fc0", "#5450a6", "#3e3a8e", "#2e2a76", "#221e5c", "#171442")
SHEETS = dict(
    sky=Sheet(("#f6f8fd", "#d0dcf2", "#a2b8e6", "#7090d6", "#4a6ac2", "#2f4ca8", "#22348a", "#18246a", "#101a50"),
              fine=0.25, broad=0.55, white=0.4, thick=0.15, jit=0.15, stretch=3.0, lead=1.1),
    ultra=Sheet(("#f2f4fc", "#ccd4f4", "#9eaae8", "#6e7cd6", "#4c58c0", "#3640a6", "#28308c", "#1c226c", "#12164c"),
                fine=0.25, broad=0.55, white=0.4, thick=0.15, jit=0.15, stretch=3.0, lead=1.1, part="sky"),
    dusk=Sheet(("#fffbec", "#fdf0c2", "#fbe08e", "#f6c862", "#eeac44", "#e08c30", "#c86e24"),
               fine=0.2, broad=0.45, white=0.4, thick=0.15, jit=0.15, stretch=3.0, lead=1.1, part="sky"),
    cloud=Sheet(("#cdc8ec", "#a4a0da", "#7c7cc6", "#5c5eae", "#444796", "#33357c"),
                fine=0.3, broad=0.5, white=0.35, thick=0.15, jit=0.25, stretch=2.0, lead=1.1, part="sky"),
    far=Sheet(("#c0c8ee", "#98a6e2", "#7486d0", "#5a6cbc", "#4656a4", "#36448a"),
              fine=0.25, broad=0.5, white=0.25, spots=0.45, thick=0.15, jit=0.3, lead=1.5, rim=2.2),
    hills=Sheet(HILLS, fine=0.25, broad=0.5, white=0.25, spots=0.45, thick=0.18, jit=0.3, lead=1.6, rim=2.5),
    forest=Sheet(("#8eb89a", "#5e9a84", "#3e7a6c", "#2c5e58", "#1e4648", "#14323a"),
                 fine=0.2, broad=0.3, white=0.2, spots=0.3, flakes=1.0, thick=0.2, jit=0.4, lead=1.4, rim=2.0),
    water=Sheet(("#e6edf8", "#b4c4ea", "#8098d4", "#5a72bc", "#4258a4", "#30428a", "#24306e", "#1a2252"),
                fine=0.15, broad=0.4, white=0.25, ripple=1.0, jit=0.3, stretch=2.5, lead=1.0),
    glint=Sheet(("#fff6d6", "#fde6a2", "#f8ca6a", "#f0aa48", "#e08a34", "#c86c28"),
                fine=0.3, broad=0.6, white=0.3, ripple=1.0, jit=0.3, stretch=1.5, lead=1.0, part="water"),
    refl=Sheet(HILLS, fine=0.15, broad=0.3, white=0.2, ripple=0.8, jit=0.3, stretch=2.0, lead=1.0, part="water"),
    bank=Sheet(("#a0c06e", "#6e9e58", "#487e50", "#306448", "#224c3e", "#173830", "#0f2622"),
               fine=0.25, broad=0.45, white=0.15, spots=0.55, folds=0.25, thick=0.2, jit=0.35, lead=2.0, rim=2.8),
    moss=Sheet(("#90c4aa", "#5ea492", "#3c847a", "#2a6664", "#1e4c4e", "#14363c"),
               fine=0.2, broad=0.35, white=0.2, folds=0.6, thick=0.15, jit=0.3, lead=2.0, rim=2.8, part="bank"),
    petal=Sheet(("#fefcf4", "#f8f3e0", "#f1e8da", "#ecd4d8", "#dcb0c8", "#c48ab8", "#a26aa6"),
                fine=0.12, broad=0.25, white=0.5, edge=0.4, folds=0.55, thick=0.1, jit=0.2, lead=1.5, rim=2.7),
    bark=Sheet(("#d8d2c8", "#b4aa9c", "#8e7e6e", "#6e5a4a", "#583e32", "#3e2c26", "#2a1e1a"),
               fine=0.5, broad=0.45, white=0.2, edge=0.2, thick=0.2, jit=0.35, lead=1.6, rim=2.7),
    iris=Sheet(("#f0e8fa", "#d0b8f0", "#aa86e2", "#8456ca", "#6230ac", "#461888", "#2e0c62", "#1e0640"),
               fine=0.18, broad=0.3, white=0.3, folds=0.45, thick=0.2, jit=0.25, lead=1.4, rim=2.5),
    irisb=Sheet(("#eaf0fc", "#bccaf4", "#8ea4e8", "#6276d2", "#4250b4", "#2e3490", "#20206a", "#14124a"),
                fine=0.18, broad=0.3, white=0.3, folds=0.45, thick=0.2, jit=0.25, lead=1.4, rim=2.5, part="iris"),
    irisw=Sheet(("#ffffff", "#f3f3fc", "#dee0f6", "#c2c6ec", "#a0a4dc", "#7e7ec8", "#5e5aac"),
                fine=0.15, broad=0.3, white=0.45, folds=0.45, thick=0.15, jit=0.2, lead=1.4, rim=2.5, part="iris"),
    gold=Sheet(("#fdf0b0", "#fad868", "#f2b438", "#dc8e24"), fine=0.2, broad=0.2, white=0.3, jit=0.2, lead=1.2,
               part="iris"),
    blade=Sheet(("#b8d47c", "#80b260", "#4e9658", "#367c58", "#286452", "#1c4c46", "#123838"),
                fine=0.35, broad=0.35, white=0.25, edge=0.25, thick=0.2, jit=0.35, lead=1.6, rim=2.2),
    wist=Sheet(("#f4f0fb", "#dcd0f4", "#bcaeea", "#988cdc", "#786cc6", "#5c4ea6"),
               fine=0.12, broad=0.2, white=0.35, spots=0.3, thick=0.15, jit=0.3, lead=1.3, rim=1.7),
    leaf=Sheet(("#d8e68c", "#aad06a", "#7cb256", "#56944a", "#3c7642", "#285a3a"),
               fine=0.2, broad=0.3, white=0.3, spots=0.3, thick=0.18, jit=0.3, lead=1.4, rim=2.0),
)
for _name, _s in SHEETS.items():
    _s.part = _s.part or _name
F = {name: i for i, name in enumerate(SHEETS, 1)}
PARTS = sorted({s.part for s in SHEETS.values()})
FLAKES = [pigment(c) for c in ("#1e5c3e", "#3c7c2e", "#8eaa42", "#cabb52", "#2e4c6c", "#5c3c2c", "#d27c42", "#6a9c8c")]
THREAD = pigment("#3a3020")
FOLD = pigment("#c4c6b4")                 # what a thick fold of white opal does to the light
LEAD = lin("#1a1918")
IRON = lin("#121110")


# ---- drawing helpers

def bez(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    P = [np.asarray(p, np.float64) for p in (p0, p1, p2, p3)]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


def arclen(P):
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])


def along(P, s, at):
    return np.stack([np.interp(at, s, P[:, 0]), np.interp(at, s, P[:, 1])], -1)


def catmull(P, k=30):
    """A smooth curve through the points P."""
    P = np.vstack([P[:1], P, P[-1:]])
    t = np.linspace(0, 1, k, endpoint=False)[:, None]
    out = [0.5 * (2 * P[i] + (-P[i - 1] + P[i + 1]) * t + (2 * P[i - 1] - 5 * P[i] + 4 * P[i + 1] - P[i + 2]) * t ** 2
                  + (-P[i - 1] + 3 * P[i] - 3 * P[i + 1] + P[i + 2]) * t ** 3) for i in range(1, len(P) - 2)]
    return np.vstack(out + [P[-2:-1]])


def outline(base, ang, length, width, r, bend=0.0, shape="petal", n=28, wob=0.04, ruffle=0.0):
    """A petal (round at the end, narrow at the claw) or a leaf (pointed at both ends) from `base`
    along `ang`; `bend` curls it to one side, `ruffle` waves its edge toward the end. Glass will only
    break along a smooth curve, so every edge is one."""
    t = np.linspace(0, 1, n)
    if shape == "petal":
        prof = np.where(t < 0.58, (t / 0.58) ** 0.65, np.sqrt(np.clip(1 - ((t - 0.58) / 0.42) ** 2, 0, 1)))
    else:
        prof = np.sin(np.pi * t) ** 0.8
    u = length * t
    c, s = np.cos(ang), np.sin(ang)
    pts = []
    for side in (1, -1):
        wv = (1 + wob * np.sin(t * r.uniform(2.5, 5) + r.uniform(0, 6.3))
              + ruffle * np.sin(t * r.uniform(5, 8) + r.uniform(0, 6.3)) * t)
        v = side * 0.5 * width * prof * wv + bend * length * t ** 2
        q = np.stack([base[0] + u * c - v * s, base[1] + u * s + v * c], 1)
        pts.append(q if side == 1 else q[::-1])
    return np.vstack(pts)


def lens(cx, cy, length, top, bottom, tilt, r, billow=0.0, point=0.3):
    """A long thin piece pointed at both ends: a cloud, or the sun on the water."""
    u = np.linspace(-1, 1, 60)
    sh = (1 - u ** 2) ** point
    lift = tilt * u * length / 2
    hi = cy - top * sh * (1 + billow * np.sin(5 * u + r.uniform(0, 6.3))) + lift
    lo = cy + bottom * sh + lift
    xs = cx + u * length / 2
    return np.vstack([np.stack([xs, hi], 1), np.stack([xs, lo], 1)[::-1]])


def toward(a, b, k):
    """The direction a turned part of the way (k) toward the direction b."""
    return np.arctan2((1 - k) * np.sin(a) + k * np.sin(b), (1 - k) * np.cos(a) + k * np.cos(b))


class Cartoon:
    """The full-size drawing: every piece of glass, the sheet it is to be cut from, the place in that
    sheet's run of colour, and the way its streaks are to lie."""

    def __init__(self):
        self.keys = np.zeros((H, W), np.int32)
        self.sheet, self.tone, self.ang, self.drift, self.plate = [0], [0.0], [0.0], [0.0], [False]

    def add(self, sheet, tone, ang=0.0, drift=0.0, plate=False):
        self.sheet.append(F[sheet])
        self.tone.append(float(tone))
        self.ang.append(float(ang))
        self.drift.append(float(drift))
        self.plate.append(plate)
        return len(self.sheet) - 1

    def region(self, mask, sub, choose, plate=False):
        """Glass for each distinct value of sub(xs, ys) inside `mask`; choose(cx, cy) gives every piece,
        from its centre, its sheet and tone, and if it likes the lie of its streaks and their drift."""
        ys, xs = np.nonzero(mask)
        u, inv = np.unique(sub(xs, ys), return_inverse=True)
        cnt = np.bincount(inv)
        cx, cy = np.bincount(inv, xs) / cnt, np.bincount(inv, ys) / cnt
        rows = zip(*(np.broadcast_to(v, cx.shape) for v in choose(cx, cy)))
        ids = np.array([self.add(*row, plate=plate) for row in rows], np.int32)
        self.keys[ys, xs] = ids[inv]

    def __enter__(self):
        self.im = Image.fromarray(self.keys)
        self.draw = ImageDraw.Draw(self.im)
        return self

    def __exit__(self, *exc):
        self.keys = np.asarray(self.im).copy()
        del self.im, self.draw

    def poly(self, pts, sheet, tone, ang=0.0, drift=0.0):
        k = self.add(sheet, tone, ang, drift)
        self.draw.polygon([tuple(p) for p in np.asarray(pts, np.float64)], fill=k)
        return k

    def strip(self, P, w, sheet, tone, r, seg=(110, 230), slant=0.6, lanes=1):
        """A branch, a stem or a blade: a band along the path P, `w(f)` wide at fraction f of the way,
        cut across into lengths a glazier could get out of one sheet. The cuts are never square. A
        thick limb is two runs of glass side by side, a paler and a darker, the cuts staggered."""
        s = arclen(P)
        L = s[-1]
        at = np.linspace(0, L, max(40, int(L / 3)))
        C = along(P, s, at)
        d = np.gradient(C, axis=0)
        d /= np.hypot(*d.T)[:, None] + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        wid = w(at / L)
        edges = [C + nrm * wid[:, None] / 2]
        if lanes == 2:
            edges.append(C + nrm * (0.1 * wid * noise.line1d(at.size, 50, r))[:, None])
        edges.append(C - nrm * wid[:, None] / 2)
        for lane in range(len(edges) - 1):
            left, right = edges[lane], edges[lane + 1]
            shade = 0.0 if lanes == 1 else 0.7 * (2 * lane - 1)
            cuts = [0.0]
            while cuts[-1] < L:
                cuts.append(cuts[-1] + r.uniform(*seg))
            if len(cuts) > 2 and L - cuts[-2] < seg[0] * 0.6:
                cuts.pop(-2)
            cuts[-1] = L
            off = [0.0] + [r.normal(0, slant) * 0.5 * np.interp(c, at, wid) for c in cuts[1:-1]] + [0.0]
            for j in range(len(cuts) - 1):
                a, b = cuts[j], cuts[j + 1]
                m = max(4, int((b - a) / 3))
                lp = along(left, at, np.linspace(a + off[j], b + off[j + 1], m))
                rp = along(right, at, np.linspace(a - off[j], b - off[j + 1], m))
                dd = d[int(np.interp((a + b) / 2, at, np.arange(len(at))))]
                t = tone(j) if callable(tone) else tone + shade + r.normal(0, 0.3)
                self.poly(np.vstack([lp, rp[::-1]]), sheet, t, ang=np.arctan2(dd[1], dd[0]))


def bump(c, s):
    return np.exp(-((X - c) / s) ** 2)


def lenses(lines, share, r, reach=(300, 800), thin=70):
    """Pull stretches of some lines down onto the line below, so that the thin strips between them taper
    to a point and run on: the long lens-shaped pieces of a Tiffany river. Lines run bottom to top."""
    for i in range(1, len(lines)):
        if r.random() < share and np.mean(np.abs(lines[i - 1] - lines[i])) < thin:
            c, w = r.uniform(X0, X1), r.uniform(*reach)
            e = noise.smoothstep(1.0, 0.45, np.abs(X - c) / w)
            lines[i] = lines[i] + (lines[i - 1] - lines[i]) * e
    return lines


def bands(yy, lines, inside, cut, r, slant=(0.5, 1.1)):
    """Number the strips between the lines and cut each strip into lengths with cuts that run aslant, a
    little bowed. `cut(h, x)` gives the spacing range at x for a strip h px high; `slant` the range of
    angles off upright."""
    k = np.zeros((H, W), np.int32)
    for ln in lines:
        k += yy < ln[None, :]
    sub = np.zeros((H, W), np.int32)
    for b in range(len(lines) + 1):
        ys, xs = np.nonzero((k == b) & inside)
        if not ys.size:
            continue
        hb, ym = np.count_nonzero(k == b) / W, ys.mean()
        seg, x = np.zeros(xs.size, np.int32), X0 - r.uniform(0, 600)
        while x < X1:
            x += r.uniform(*cut(hb, x))
            t, bow = np.tan(r.choice((-1, 1)) * r.uniform(*slant)), r.normal(0, 0.0015)
            seg += xs > x + (ys - ym) * t + bow * (ys - ym) ** 2
        sub[ys, xs] = b * 256 + seg
    return sub


def cells(mask, r, size):
    """Irregular pieces over `mask`, about `size` px across, their edges gently curved: the mottled
    ground of a bank."""
    gy, gx = np.mgrid[0:H:size, 0:W:size]
    py = np.clip(gy + r.uniform(0.1, 0.9, gy.shape) * size, 0, H - 1).astype(int).ravel()
    px = np.clip(gx + r.uniform(0.1, 0.9, gx.shape) * size, 0, W - 1).astype(int).ravel()
    keep = mask[py, px]
    seeds = np.zeros((H, W), np.int32)
    seeds[py[keep], px[keep]] = np.arange(1, keep.sum() + 1)
    iy, ix = ndimage.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    yy, xx = np.mgrid[0:H, 0:W]
    wy = np.clip(yy + 0.15 * size * noise.field((H, W), size, r), 0, H - 1).astype(int)
    wx = np.clip(xx + 0.15 * size * noise.field((H, W), size, r), 0, W - 1).astype(int)
    return seeds[iy, ix][wy, wx]


# ---- the cartoon, from the back of the picture to the front

def sky(cart, r, yy):
    """A few broad strips, wavy at the joins like the edges of clouds, each cut into long pieces: cream,
    gold and amber low over the hills, warmest over the valley where the sun went down; white and pale
    blue above them; cobalt and ultramarine overhead. Each piece is cut where its sheet swings hardest
    from one colour to the next."""
    lines, y = [], 1560.0
    while y > Y0 - 160:
        h = r.uniform(80, 150) * (1 + (1560 - y) / 1000) * (0.45 if r.random() < 0.2 else 1.0)
        y -= h
        ln = y + 0.18 * h * noise.line1d(W, r.uniform(300, 700), r) + r.normal(0, 0.03) * (X - W / 2)
        if r.random() < 0.4:                     # the top of a cloud, billowing up into the strip above
            c, w = r.uniform(X0, X1), r.uniform(250, 600)
            hump = np.clip(1 - ((X - c) / w) ** 2, 0, 1) ** 0.6
            ln = ln - 0.35 * h * hump * (0.8 + 0.2 * np.cos((X - c) / r.uniform(35, 70)))
        lines.append(ln)
    sub = bands(yy, lines, np.ones((H, W), bool), lambda hb, x: (700, 1500), r, slant=(0.3, 0.9))

    def pick(cx, cy):
        n = cx.size
        e = np.clip((1500 - cy) / 1460, 0, 1) + r.normal(0, 0.03, n)
        side = np.clip(np.abs(cx - SUN) / 950, 0, 1)
        dusk = e < 0.2 + 0.03 * (1 - side)
        pale = ~dusk & (e < 0.28)
        cream = pale & (r.random(n) < 0.35)
        tone = np.where(dusk, 0.3 + 3.4 * np.clip(e / 0.22, 0, 1) ** 1.2 + 1.8 * side ** 1.5,
                        np.where(pale, r.uniform(0.7, 2.0, n), 2.4 + 5.4 * np.clip((e - 0.28) / 0.72, 0, 1) ** 0.6))
        tone = tone - 0.3 * cream + (r.random(n) < 0.08) * r.choice((-1.0, 1.0), n) * r.uniform(0.5, 1.0, n)
        blue = np.where(r.random(n) < 0.2 + 0.3 * np.clip(e, 0, 1), "ultra", "sky")   # cobalt, or ultramarine
        return (np.where(dusk | cream, "dusk", blue), tone, 0.0, r.choice((-1.0, 1.0), n) * r.uniform(0.5, 1.3, n))
    cart.region(np.ones((H, W), bool), lambda xs, ys: sub[ys, xs], pick)
    with cart:                                   # long low clouds, violet against the gold
        for _ in range(5):
            cx, cy = r.uniform(150, 1850), r.uniform(1150, 1430)
            pts = lens(cx, cy, r.uniform(280, 750), r.uniform(12, 30), r.uniform(5, 12), r.normal(0, 0.012), r, 0.2)
            cart.poly(pts, "cloud", r.uniform(1.0, 3.2), 0.0, r.normal(0, 0.8))


def ridges(r):
    soft = lambda a, b: a * noise.line1d(W, 170, r) + b * noise.line1d(W, 45, r)
    far = 1520 - 215 * bump(620, 340) - 160 * bump(1460, 290) - 60 * bump(1060, 120) + soft(10, 3)
    mid = 1630 - 150 * bump(330, 280) - 175 * bump(1700, 270) - 40 * bump(800, 150) + soft(8, 3)
    return far, mid


def hills(cart, r, yy, far, mid):
    """Two ranges of mottled glass, the farther a softer violet-blue plated with a second, milky sheet to
    push it back, the nearer deep violet-blue; both paler toward the mist at their feet."""
    for ridge, sheet, base, plate in ((far, "far", 2.8, True), (mid, "hills", 4.1, False)):
        low = ridge + r.uniform(60, 100) + 14 * noise.line1d(W, 200, r)
        cuts, x = [], X0 - r.uniform(0, 300)
        while x < X1 + 100:
            x += r.uniform(130, 300)
            cuts.append((x, np.tan(r.normal(0, 0.6))))

        def sub(xs, ys, ridge=ridge, low=low, cuts=cuts):
            k = sum((xs > c + (ys - ridge[int(np.clip(c, 0, W - 1))]) * t).astype(np.int32) for c, t in cuts)
            return k + 1000 * (ys > low[xs])

        def pick(cx, cy, ridge=ridge, base=base, sheet=sheet):
            return sheet, base + 0.5 - 1.0 * np.clip((cy - ridge[np.clip(cx.astype(int), 0, W - 1)]) / 120, 0, 1)
        cart.region((yy > ridge[None, :]) & (yy < SHORE + 40), sub, pick, plate=plate)


def banks(yy):
    """The near banks, coming in from both sides, with the water between them running toward us."""
    left = catmull(np.array([(-40, 1980), (220, 2050), (470, 2180), (650, 2360), (760, 2600), (800, 2860)], float), 40)
    right = catmull(np.array([(2040, 1900), (1760, 1990), (1500, 2140), (1330, 2360), (1250, 2600), (1220, 2860)], float), 40)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.polygon([tuple(p) for p in np.vstack([left, [(-40, 2860)]])], fill=1)
    d.polygon([tuple(p) for p in np.vstack([right, [(2040, 2860)]])], fill=1)
    return np.asarray(im) > 0


def land(cart, r, yy, mid):
    """The river in ripple glass, darker than the sky, with the sunset laid down it in pieces of gold; the
    hills and trees reflected under the far shore; the trees along the far shore in confetti glass; and
    the near banks in mottled and folded greens."""
    shore = SHORE + 5 * noise.line1d(W, 300, r)
    below = yy > shore[None, :]
    near = banks(yy)
    water = below & ~near

    # trees on the far shore: two rows of crowns, each a few lobes of leaves, the back row taller and bluer
    rows = []
    for row, (hmin, hmax, wmin, wmax) in enumerate(((50, 130, 40, 90), (28, 70, 30, 70))):
        profs, cents = [], []
        for lo, hi in ((X0 - 80, 860 - 120 * row), (1270 + 100 * row, X1 + 80)):
            x = lo
            while x < hi:
                w, h = r.uniform(wmin, wmax), r.uniform(hmin, hmax)
                if r.random() < 0.12:                 # a poplar
                    w, h = w * 0.6, h * 1.8
                p = np.zeros(W, np.float32)
                for _ in range(r.integers(2, 5)):
                    lc, lw, lh = x + w + r.uniform(-0.45, 0.45) * w, w * r.uniform(0.45, 0.7), h * r.uniform(0.7, 1.0)
                    p = np.maximum(p, lh * np.clip(1 - ((X - lc) / lw) ** 2, 0, 1) ** 0.55)
                profs.append(p)
                cents.append(x + w + r.normal(0, 0.15 * w))
                x += w * r.uniform(0.9, 1.6)
        prof = np.stack(profs)
        rows.append((prof.max(0), prof.argmax(0), np.array(cents)))

    # the water, in strips that taper and run on, cut short down the middle where the sun lies on it
    lines, y, h = [], shore + 0.0, 9.0
    while y.min() < H:
        y = y + h * (1 + 0.08 * noise.line1d(W, r.uniform(300, 700), r))
        lines.append(y)
        h *= 1.16
    lines = lenses(lines[::-1], 0.35, r, reach=(500, 1100), thin=40)[::-1]
    path = lambda hb, x: (max(110, 1.6 * hb), max(260, 3 * hb)) if abs(x - SUN + 60) < 380 else (600, 1400)
    wsub = bands(yy, lines, water, path, r, slant=(0.6, 1.1))

    def wpick(cx, cy):
        """Blue going deeper as the water comes toward us; under the far shore, down the middle, thin
        pieces of gold."""
        n = cx.size
        dep = np.clip((cy - SHORE) / 1100, 0, 1)
        off = (cx - SUN + 120 * dep) / (55 + 230 * dep)
        gold = r.random(n) < np.exp(-off ** 2) * 0.85 * (dep < 0.16)
        tone = np.where(gold, 0.8 + 0.8 * np.minimum(off ** 2, 2) + r.normal(0, 0.4, n),
                        2.8 + 2.4 * dep ** 0.8 + r.normal(0, 0.35, n))
        return np.where(gold, "glint", "water"), tone, 0.0, r.normal(0, 0.6, n)
    cart.region(water, lambda xs, ys: wsub[ys, xs], wpick)

    # under the far shore the hills and trees come back upside down, broken by the ripples
    sil = np.maximum(np.maximum(rows[0][0], rows[1][0]), 0.35 * np.clip(shore - mid, 0, None))
    refl = water & (yy < (shore + 0.55 * sil)[None, :])
    trees_over = (rows[0][0] + rows[1][0] > 0)
    cart.region(refl, lambda xs, ys: wsub[ys, xs] * 2 + trees_over[xs],
                lambda cx, cy: ("refl", r.uniform(2.6, 4.2, cx.size)))

    # nearer, the sun lies on the ripples in short lenses of gold, fewer and farther apart as they come on
    with cart:
        for a, b in zip(lines[:-1], lines[1:]):
            ym = 0.5 * (a + b)
            dep = (ym[int(SUN)] - SHORE) / 1100
            if dep < 0.12:
                continue
            centre, spread = SUN - 120 * dep, 55 + 230 * dep
            last = -1e9
            for x0 in np.sort(centre + r.normal(0, spread, r.poisson(4.0 * (1.15 - min(dep, 1))))):
                xi = int(np.clip(x0, X0, X1 - 1))
                hgt, ln = b[xi] - a[xi], r.uniform(0.6, 1.5) * (80 + 340 * dep)
                th = max(r.uniform(0.12, 0.24) * hgt, 6.0)
                if hgt < 16 or x0 - ln / 2 < last + 20:
                    continue
                last = x0 + ln / 2
                cart.poly(lens(x0, ym[xi] + r.uniform(-0.15, 0.15) * hgt, ln, th * r.uniform(0.7, 1.3),
                               th * r.uniform(0.7, 1.3), r.normal(0, 0.01), r, 0.15, r.uniform(0.6, 1.2)), "glint",
                          0.8 + 2.4 * dep + r.normal(0, 0.6), 0.0, r.choice((-1, 1)) * r.uniform(0.4, 1.0))

    # the banks: irregular pieces of mottled green, some of them folded, lit along the water's edge
    piece = cells(near, r, 150)
    inner = ndimage.distance_transform_edt(near)

    def bpick(cx, cy):
        n = cx.size
        k = np.clip(inner[cy.astype(int), cx.astype(int)] / 260, 0, 1)
        return (np.where(r.random(n) < 0.3, "moss", "bank"), 1.6 + 3.2 * k + r.normal(0, 0.6, n),
                r.uniform(0, np.pi, n), r.normal(0, 0.5, n))
    cart.region(near, lambda xs, ys: piece[ys, xs], bpick)

    for (hgt, which, cents), tones in zip(rows, ((2.0, 4.0), (0.8, 3.0))):
        m = (yy > (shore - hgt)[None, :]) & (yy < (shore + 6)[None, :]) & (hgt > 0)[None, :]
        cart.region(m, lambda xs, ys, which=which, hgt=hgt, cents=cents: which[xs] * 4 + (ys > (shore - 0.5 * hgt)[xs])
                    + 2 * (xs > cents[which[xs]]), lambda cx, cy, tones=tones: ("forest", r.uniform(*tones, cx.size)))


def magnolia(cart, r):
    """The tree on the left: a trunk and limbs in streaky grey and brown glass, the trunk in two runs, and
    the flowers in drapery glass, each petal one piece with its folds running from the claw to the tip:
    cups turned every way, a few open wide, and buds."""
    limbs = [
        (np.vstack([bez((250, 2820), (280, 2350), (215, 1900), (285, 1520)),
                    bez((285, 1520), (330, 1240), (370, 1020), (450, 800))]), 74, 34),
        (bez((440, 840), (600, 660), (840, 560), (1090, 470)), 30, 14),
        (bez((1088, 472), (1160, 440), (1220, 400), (1280, 350)), 14, 8),
        (bez((450, 806), (520, 620), (560, 420), (610, 250)), 32, 17),
        (bez((608, 256), (630, 180), (650, 110), (690, 20)), 17, 12),
        (bez((300, 1420), (230, 1280), (160, 1170), (20, 1100)), 28, 14),
        (bez((560, 432), (700, 362), (850, 300), (1010, 230)), 19, 9),
        (bez((375, 1060), (290, 930), (230, 790), (200, 640)), 22, 11),
        (bez((202, 644), (186, 540), (190, 440), (222, 330)), 11, 7),
        (bez((590, 332), (500, 262), (400, 222), (290, 120)), 15, 7),
        (bez((262, 1700), (370, 1610), (470, 1560), (560, 1530)), 18, 8),
        (bez((800, 590), (860, 500), (900, 420), (930, 330)), 12, 7),
    ]
    flowers = []
    for i, (P, w0, w1) in enumerate(limbs):
        cart.strip(P, lambda f, w0=w0, w1=w1: w0 + (w1 - w0) * f, "bark", 2.4 + r.normal(0, 0.4), r,
                   lanes=2 if w0 >= 30 else 1)
        if i == 0:
            continue
        s = arclen(P)
        pos, side = r.uniform(40, 100), r.choice((-1, 1))
        while pos < s[-1] - 40:
            at = along(P, s, pos)
            dd = along(P, s, pos + 5) - at
            out = np.arctan2(dd[1], dd[0]) + side * r.uniform(0.6, 1.1)
            ta = toward(out, UP, 0.45)
            end = at + r.uniform(30, 80) * np.array([np.cos(ta), np.sin(ta)])
            cart.strip(np.vstack([at, (at + end) / 2 + r.normal(0, 3, 2), end]), lambda f: 10 - 3 * f, "bark",
                       3.0 + r.normal(0, 0.4), r, seg=(400, 500))
            flowers.append((end, ta, out))
            side = -side
            pos += r.uniform(95, 170)
        if 60 < P[-1][0] < W - 60 and P[-1][1] > 60:
            dd = P[-1] - P[-4]
            a = np.arctan2(dd[1], dd[0])
            flowers.append((P[-1], a, a))
    for (x, y), ta, out in flowers:
        kind = r.choice(4, p=(0.4, 0.18, 0.24, 0.18))
        if kind == 0:                            # a cup held up, a little turned
            cup(cart, x, y, toward(ta, UP, r.uniform(0.2, 0.5)), r.uniform(88, 128), r, turn=r.normal(0, 0.3))
        elif kind == 1:                          # open wide
            star(cart, x, y, toward(ta, UP, 0.3), r.uniform(88, 118), r)
        elif kind == 2:                          # leaning out with the twig, or nodding
            cup(cart, x, y, out + r.normal(0, 0.25), r.uniform(80, 115), r,
                turn=r.choice((-1, 1)) * r.uniform(0.3, 0.6))
        else:
            bud(cart, x, y, toward(ta, UP, 0.5), r.uniform(50, 80), r)


def cup(cart, x, y, a, L, r, turn=0.0):
    """A saucer magnolia half open: the far petals show their white insides, the near ones the rose
    flush toward the claw. `turn` swings it toward three-quarter view, the petals on one side drawn in
    and those on the other spread."""
    j = lambda q: r.normal(0, q)
    spread, flush = r.uniform(0.8, 1.25), r.uniform(-0.5, 0.9)
    for s in (-1, 1):
        k = 1 + s * turn
        aa = a + s * (0.5 + j(0.06)) * spread * k
        cart.poly(outline((x, y), aa, L * (0.86 + j(0.05)) * (1 - 0.15 * s * turn), L * 0.52 * (1 - 0.2 * s * turn), r,
                          bend=s * 0.07, ruffle=0.04), "petal", 0.4 + abs(j(0.3)), aa, -0.4)
    for s in (-1, 1):
        k = 1 + s * turn
        aa = a + s * (0.27 + j(0.05)) * spread * k
        cart.poly(outline((x, y), aa, L * (1 + j(0.05)) * (1 - 0.1 * s * turn), L * 0.6 * (1 - 0.25 * s * turn), r,
                          bend=-s * 0.06, ruffle=0.04), "petal", 1.4 + flush + j(0.3), aa, -1.4)
    aa = a + 0.2 * turn + j(0.07)
    cart.poly(outline((x - 0.05 * L * np.cos(a), y - 0.05 * L * np.sin(a)), aa, L * 0.9, L * 0.66 * (1 - 0.2 * abs(turn)),
                      r, ruffle=0.04), "petal", 1.6 + flush + j(0.3), aa, -1.6)
    cart.poly(outline((x - 0.12 * L * np.cos(a), y - 0.12 * L * np.sin(a)), a, L * 0.22, L * 0.2, r, shape="leaf"),
              "bark", 1.2, a)


def star(cart, x, y, a, L, r):
    """A flower open wide, seen a little from above: the outer tepals spread round, white inside and
    rose at the claw, the farther ones foreshortened; the inner ones still held up round the boss."""
    n = r.integers(5, 8)
    tilt = r.uniform(0.4, 0.7)
    th0 = r.uniform(0, 2 * np.pi)
    tepals = []
    for k in range(n):
        th = th0 + 2 * np.pi * k / n + r.normal(0, 0.15)
        u, v = np.cos(th) * tilt, np.sin(th)
        tepals.append((np.cos(th), a + np.arctan2(v, u), np.hypot(u, v)))
    tepals.sort(key=lambda t: -t[0])             # the far side first
    for depth, ang, fl in tepals:
        cart.poly(outline((x, y), ang, L * fl * r.uniform(0.9, 1.1), L * 0.42, r, bend=r.normal(0, 0.06), ruffle=0.05),
                  "petal", 0.4 + 0.8 * max(-depth, 0) + abs(r.normal(0, 0.3)), ang, -1.2)
    for s in (-1, 1):
        aa = a + s * r.uniform(0.22, 0.4)
        cart.poly(outline((x, y), aa, 0.6 * L, 0.36 * L, r, bend=-s * 0.05, ruffle=0.03), "petal",
                  1.5 + r.normal(0, 0.3), aa, -1.4)
    cart.poly(outline((x, y), a + r.normal(0, 0.1), 0.24 * L, 0.12 * L, r, shape="leaf"), "leaf", 0.6 + r.normal(0, 0.3), a)


def bud(cart, x, y, a, L, r):
    """A bud: most still in their grey fur, some showing the rose of the petals."""
    rose = r.random() < 0.5
    sheet, tone = ("petal", 2.8 + r.normal(0, 0.4)) if rose else ("bark", 0.3 + abs(r.normal(0, 0.3)))
    L = L if rose else 0.75 * L
    for s in (-1, 1):
        aa = a + s * 0.05
        cart.poly(outline((x, y), aa, L, L * 0.36, r, bend=s * 0.04, shape="leaf"), sheet, tone + r.normal(0, 0.2),
                  aa, -0.8)
    cart.poly(outline((x - 0.1 * L * np.cos(a), y - 0.1 * L * np.sin(a)), a, L * 0.36, L * 0.26, r, shape="leaf"),
              "bark", 1.4, a)


def wisteria(cart, r):
    """The vine over the top right, its leaves, and the flowers hanging from it in long trusses, each
    floret a small piece of mottled glass, pale where the flowers are open and deep in the buds at the tip."""
    vines = [
        (np.vstack([bez((1290, 20), (1380, 160), (1560, 140), (1700, 220)),
                    bez((1700, 220), (1820, 290), (1862, 420), (1830, 560)),
                    bez((1830, 560), (1800, 700), (1880, 820), (1990, 900))]), 32, 24),
        (np.vstack([bez((1990, 110), (1820, 160), (1720, 300), (1560, 280)),
                    bez((1560, 280), (1460, 265), (1400, 230), (1320, 250))]), 22, 10),
    ]
    leaves = [((1400, 150), 2.3), ((1590, 165), 1.8), ((1735, 250), 0.5), ((1860, 470), 2.7), ((1845, 640), 2.95),
              ((1520, 270), 2.2), ((1660, 300), 1.3), ((1880, 760), 2.0), ((1330, 240), 2.6), ((1790, 330), 1.6)]
    for (x, y), a in leaves:
        compound(cart, x, y, a + r.normal(0, 0.1), r.uniform(240, 340), r)
    for P, w0, w1 in vines:
        cart.strip(P, lambda f, w0=w0, w1=w1: w0 + (w1 - w0) * f, "bark", 2.8 + r.normal(0, 0.3), r, seg=(90, 180))
    for (x, y), L in [((1352, 128), 380), ((1452, 160), 500), ((1548, 158), 430), ((1652, 205), 560), ((1756, 258), 470),
                      ((1842, 390), 520), ((1836, 530), 420), ((1822, 668), 470), ((1900, 850), 400), ((1612, 290), 330),
                      ((1500, 262), 300), ((1720, 300), 360), ((1405, 225), 280)]:
        raceme(cart, x, y, L, r.uniform(80, 106), r)


def compound(cart, x, y, a, L, r):
    """A wisteria leaf: pairs of leaflets along a stalk, and one at the end."""
    pairs = r.integers(4, 6)
    for k in range(pairs):
        f = 0.12 + 0.76 * k / pairs
        bx, by = x + f * L * np.cos(a), y + f * L * np.sin(a)
        ll = L * (0.32 - 0.1 * f)
        for s in (-1, 1):
            aa = a + s * (0.7 + r.normal(0, 0.12))
            cart.poly(outline((bx, by), aa, ll, ll * 0.44, r, shape="leaf", bend=-s * 0.06), "leaf", r.uniform(0.8, 4.6), aa)
    ex, ey = x + L * np.cos(a), y + L * np.sin(a)
    cart.poly(outline((ex - 0.04 * L * np.cos(a), ey - 0.04 * L * np.sin(a)), a, L * 0.28, L * 0.12, r, shape="leaf"),
              "leaf", r.uniform(0.8, 4.0), a)


def raceme(cart, x, y, L, w0, r):
    sway = r.normal(0, 0.1)
    pos = 0.0
    while pos < L:
        f = pos / L
        sz = w0 * 0.3 * (1 - 0.55 * f)
        width = w0 * (1 - 0.72 * f) * (1 + 0.15 * np.sin(f * 9 + x))
        m = max(1, int(round(width / (sz * 1.2))))
        ax, ay = x + sway * L * f ** 2, y + pos
        for k in r.permutation(m):
            fx = ax + (k - (m - 1) / 2) * sz * 1.15 + r.normal(0, 0.2 * sz)
            fy = ay + r.normal(0, 0.2 * sz)
            th = np.linspace(0, 2 * np.pi, 18, endpoint=False)          # a pebble of glass, round and a little lumpy
            rad = sz * 0.6 * (1 + 0.12 * np.sin(2 * th + r.uniform(0, 6.3)) + 0.07 * np.sin(3 * th + r.uniform(0, 6.3)))
            rad *= r.uniform(0.85, 1.15)
            cart.poly(np.stack([fx + rad * np.cos(th), fy + rad * np.sin(th) * 0.85], 1), "wist",
                      0.3 + 3.8 * f ** 1.3 + r.normal(0, 0.4), r.uniform(0, np.pi))
        pos += sz * 0.78


# every flower its own: (x, y, size, pose, sheet, how much deeper or paler, which way a profile faces)
IRISES = [
    [(170, 2150, 150, "open", "iris", 0.4, 1), (395, 1995, 172, "side", "irisb", -0.2, 1),
     (600, 2115, 118, "half", "irisw", 0.0, 1), (305, 2335, 136, "open", "irisw", 0.7, -1),
     (700, 2385, 158, "open", "iris", 1.0, 1), (95, 2470, 112, "side", "irisb", 0.6, -1),
     (520, 2500, 96, "half", "iris", -0.6, 1)],
    [(1365, 2125, 146, "side", "iris", 0.2, -1), (1595, 1990, 168, "open", "irisw", 0.2, 1),
     (1835, 2085, 150, "open", "irisb", 0.7, 1), (1485, 2325, 128, "half", "irisb", -0.5, 1),
     (1725, 2375, 162, "open", "iris", 0.9, 1), (1925, 2445, 112, "side", "irisw", 0.5, 1),
     (1300, 2440, 98, "half", "iris", 0.2, -1)],
]
IRIS_BUDS = [
    [(505, 1905, 100, "iris", 0.6), (255, 1985, 82, "irisb", 0.3), (650, 1950, 70, "iris", 0.9), (55, 2215, 78, "irisw", 1.3)],
    [(1455, 1955, 100, "irisb", 0.5), (1725, 1885, 82, "iris", 0.6), (1965, 1960, 76, "irisw", 1.1),
     (1290, 2010, 70, "iris", 0.9)],
]


def irises(cart, r):
    """Two clumps at the water's edge: sword leaves cut in long pieces streaked lengthwise, and the
    flowers in drapery glass, from deep violet to blue-white, open, opening, in profile and in bud."""
    def blade(xb, a, L, w0, curve, tone):
        t = np.linspace(0, 1, 50)
        ang = a + curve * t ** 1.6
        st = L / 49
        P = np.stack([xb + np.concatenate([[0], np.cumsum(np.sin(ang[:-1]) * st)]),
                      2830 - np.concatenate([[0], np.cumsum(np.cos(ang[:-1]) * st)])], 1)
        cart.strip(P, lambda f: w0 * (1 - f) ** 0.7 + 0.5, "blade", tone, r, seg=(150, 320), slant=0.9)

    for (lo, hi, mid), flowers, buds in zip(((20, 800, 400), (1220, 1990, 1600)), IRISES, IRIS_BUDS):
        xs = r.uniform(lo, hi, 46)
        for xb in xs[:26]:
            blade(xb, r.normal(0, 0.18) + 0.35 * (xb - mid) / (hi - lo), r.uniform(450, 820), r.uniform(28, 44),
                  r.normal(0, 0.3), r.uniform(4.0, 6.4))
        for (x, y, s, pose, sheet, lift, face) in flowers:
            xb = x + r.normal(0, 30)
            cart.strip(np.vstack([(xb, 2830), ((xb + x) / 2 + r.normal(0, 10), (2830 + y) / 2), (x, y + 0.2 * s)]),
                       lambda f: 15 - 4 * f, "blade", 3.2, r, seg=(220, 340))
            cart.poly(outline((x + 4, y + 0.45 * s), UP + r.normal(0, 0.25), 0.45 * s, 0.13 * s, r, shape="leaf"),
                      "blade", 1.4, UP)
            iris(cart, x, y, s, sheet, r, pose, lift, face)
        for (x, y, L, sheet, lift) in buds:
            xb = x + r.normal(0, 30)
            cart.strip(np.vstack([(xb, 2830), ((xb + x) / 2, (2830 + y) / 2), (x, y + 0.6 * L)]), lambda f: 13 - 4 * f,
                       "blade", 3.0, r, seg=(220, 340))
            a = UP + r.normal(0, 0.18)
            for sg in (1, -1):
                cart.poly(outline((x, y + 0.6 * L), a + sg * 0.06, L, 0.3 * L, r, shape="leaf", bend=sg * 0.03), sheet,
                          4.0 + lift + r.normal(0, 0.3), a)
            cart.poly(outline((x, y + 0.75 * L), a + r.normal(0, 0.1), 0.55 * L, 0.2 * L, r, shape="leaf"), "blade", 1.2, a)
        for xb in xs[26:]:
            blade(xb, r.normal(0, 0.2) + 0.4 * (xb - mid) / (hi - lo), r.uniform(320, 700), r.uniform(30, 48),
                  r.normal(0, 0.35), r.uniform(2.0, 5.4))
        for (x, y, s, *_) in flowers:            # a leaf or two in front of every flower
            for _ in range(r.integers(1, 3)):
                xb = x + r.normal(0, 70)
                blade(xb, np.arctan2(x - xb + r.normal(0, 40), 2830 - y) + r.normal(0, 0.08), 2830 - y + r.uniform(-60, 140),
                      r.uniform(30, 44), r.normal(0, 0.3), r.uniform(1.8, 4.6))


def fall(cart, bx, by, a, ln, wd, bend, sheet, lift, r, beard=True):
    """One fall: the blade in deep drapery glass, a paler streaked haft at its base, and the beard."""
    cart.poly(outline((bx, by), a, ln, wd, r, bend=bend, ruffle=0.08), sheet, 3.9 + lift + r.normal(0, 0.3), a, 0.8)
    cart.poly(outline((bx, by), a, 0.48 * ln, 0.32 * wd, r, shape="leaf", bend=0.45 * bend), sheet,
              1.1 + 0.5 * lift + r.normal(0, 0.3), a, 1.2)
    if beard:
        cart.poly(outline((bx + 0.08 * ln * np.cos(a), by + 0.08 * ln * np.sin(a)), a, 0.26 * ln, 0.08 * ln, r,
                          shape="leaf"), "gold", 1.8 + r.normal(0, 0.3), a)


def iris(cart, x, y, s, sheet, r, pose, lift, face):
    """A bearded iris: the standards held up, paler; the falls hanging, deep. Open, it faces us with three
    of each; opening, the standards are still folded together and the falls just parting; in profile one
    fall reaches forward and curls down and the other goes back behind the stem."""
    j = lambda q: r.normal(0, q)
    turn = j(0.15)
    up, down = UP + turn, np.pi / 2 + turn
    if pose == "open":
        lean = r.uniform(-0.25, 0.25)            # turned a little to one side: one fall comes forward
        for sg in (1, -1):
            a = up + sg * (0.42 + j(0.06))
            cart.poly(outline((x + sg * 0.03 * s, y - 0.06 * s), a, (0.8 + j(0.05)) * s, 0.42 * s, r, bend=-sg * 0.1,
                              ruffle=0.07), sheet, 2.0 + lift + j(0.25), a, -0.6)
        a = up + j(0.08)
        cart.poly(outline((x, y - 0.04 * s), a, 0.95 * s, 0.5 * s, r, ruffle=0.08), sheet, 1.6 + lift + j(0.25), a, -0.7)
        for sg in (1, -1):
            fall(cart, x + sg * 0.06 * s, y + 0.04 * s, down - sg * (1.0 + lean * sg + j(0.1)),
                 (0.86 + 0.25 * lean * sg) * s, 0.72 * s, sg * 0.3, sheet, lift, r)
        fall(cart, x, y + 0.06 * s, down + j(0.1) - lean, 0.8 * s, 0.68 * s, 0.0, sheet, lift, r)
    elif pose == "half":
        for sg in (1, -1):
            a = up + sg * (0.2 + j(0.04))
            cart.poly(outline((x + sg * 0.04 * s, y), a, (0.85 + j(0.05)) * s, 0.36 * s, r, bend=-sg * 0.12, ruffle=0.05),
                      sheet, 2.2 + lift + j(0.25), a, -0.6)
        a = up + j(0.06)
        cart.poly(outline((x, y + 0.02 * s), a, 0.95 * s, 0.4 * s, r, ruffle=0.05), sheet, 1.8 + lift + j(0.2), a, -0.6)
        for sg in (1, -1):
            fall(cart, x + sg * 0.05 * s, y + 0.05 * s, down - sg * (0.62 + j(0.08)), 0.6 * s, 0.42 * s, sg * 0.25,
                 sheet, lift + 0.4, r, beard=False)
    else:
        a = up - face * 0.28 + j(0.06)
        cart.poly(outline((x - face * 0.05 * s, y - 0.05 * s), a, 0.75 * s, 0.36 * s, r, bend=face * 0.12, ruffle=0.06),
                  sheet, 2.6 + lift + j(0.2), a, -0.5)
        a = up + face * 0.12 + j(0.06)
        cart.poly(outline((x, y - 0.03 * s), a, 0.9 * s, 0.44 * s, r, bend=-face * 0.14, ruffle=0.07),
                  sheet, 1.8 + lift + j(0.2), a, -0.7)
        fall(cart, x - face * 0.04 * s, y + 0.05 * s, down + face * 0.9 + j(0.1), 0.5 * s, 0.4 * s, -face * 0.2,
             sheet, lift + 0.5, r, beard=False)
        fall(cart, x + face * 0.04 * s, y + 0.04 * s, down - face * 1.25 + j(0.1), 0.95 * s, 0.62 * s, face * 0.45,
             sheet, lift, r)


# ---- the glass

def stock(r, n=2048):
    """The glazier's racks, as fields over one big sheet: fine streaks and broad swirls of colour, streaks
    of white opal, clouds of milk, the crystal spots of mottled glass, the folds of drapery glass and the
    ridges of ripple glass. Every piece is cut from its own place in it, turned its own way."""
    def streaks(sx, sy, swirl, amount):
        f = np.ascontiguousarray(noise.stretched((n, n), sx, sy, r).T)
        return noise.warp(f, amount * noise.field((n, n), swirl, r), amount * noise.field((n, n), swirl, r))

    unit = lambda a: ((a - a.mean()) / a.std()).astype(np.float32)
    fine = unit(streaks(3, 300, 260, 22) + 0.7 * streaks(8, 500, 340, 30))
    broad = unit(streaks(24, 800, 420, 80))
    white = unit(streaks(14, 600, 380, 50) + 0.5 * streaks(5, 400, 300, 30))
    milk = noise.fbm((n, n), 70, r, octaves=4)
    spots = noise.field((n, n), 4.0, r)
    # drapery: the sheet was pushed into folds while soft; where this field crosses zero runs the crest of
    # a fold, and it is thick on one side of the crest and thin on the other
    folds = unit(streaks(26, 240, 160, 36) + 0.3 * streaks(9, 120, 90, 14))
    v = np.arange(n, dtype=np.float32)[:, None]
    ripple = np.cos(2 * np.pi * (v / 13 + 1.4 * noise.field((n, n), 70, r) + 0.45 * noise.field((n, n), 18, r)))
    return np.stack([fine, broad, white, milk, spots, folds, ripple]).astype(np.float32)


def confetti(r, n=1024, count=10000):
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    for cx, cy, rad, c in zip(r.uniform(0, n, count), r.uniform(0, n, count), 1.2 + r.gamma(2.2, 1.6, count),
                              r.integers(1, len(FLAKES) + 1, count)):
        k = r.integers(4, 8)
        a = np.sort(r.uniform(0, 2 * np.pi, k))
        q = rad * r.uniform(0.45, 1.25, k)
        d.polygon(list(zip(cx + q * np.cos(a), cy + q * np.sin(a) * r.uniform(0.35, 1.0))), fill=int(c))
    return np.asarray(im)


def owner(lab, keys, n):
    """The cartoon piece each piece of glass was cut for: the one most of its pixels belong to."""
    K = int(keys.max()) + 1
    u, c = np.unique(lab.ravel().astype(np.int64) * K + keys.ravel(), return_counts=True)
    l, k = u // K, u % K
    o = np.lexsort((c, l))
    l, k = l[o], k[o]
    last = np.r_[l[1:] != l[:-1], True]
    best = np.zeros(n + 1, np.int64)
    best[l[last]] = k[last]
    return best


def table(attr):
    return np.array([0.0] + [getattr(s, attr) for s in SHEETS.values()], np.float32)


def leading(lab, fam, r):
    """Copper foil and lead: a fine line between pieces of one part of the picture, a heavier lead where
    one part meets another, as heavy as the heavier side asks; solder at the joints."""
    part = np.array([0] + [PARTS.index(s.part) + 1 for s in SHEETS.values()])[fam][lab]
    wob = 1 + 0.12 * noise.field((H, W), 30, r) + 0.08 * noise.field((H, W), 5, r)
    hw = table("lead")[fam][lab] * wob
    lead = noise.smoothstep(hw + 0.7, hw - 0.7, ndimage.distance_transform_edt(~glass.boundary(lab)))
    rim = table("rim")[fam][lab]
    edge, other = np.zeros((H, W), bool), np.zeros((H, W), np.float32)
    d = part[:, 1:] != part[:, :-1]
    edge[:, 1:] |= d
    other[:, 1:] = np.where(d, rim[:, :-1], 0)
    d = part[1:] != part[:-1]
    edge[1:] |= d
    other[1:] = np.maximum(other[1:], np.where(d, rim[:-1], 0))
    dist, (iy, ix) = ndimage.distance_transform_edt(~edge, return_indices=True)
    hc = np.maximum(rim, other)[iy, ix] * wob
    lead = np.maximum(lead, noise.smoothstep(hc + 0.7, hc - 0.7, dist) * (hc > 0))
    a_, b_, c_, d_ = lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]
    many = 1 + (b_ != a_) + ((c_ != a_) & (c_ != b_)) + ((d_ != a_) & (d_ != b_) & (d_ != c_))
    joint = np.zeros((H, W), bool)
    joint[:-1, :-1] = many >= 3
    rs = 1.25 * hw
    return np.maximum(lead, noise.smoothstep(rs + 0.6, rs - 0.6, ndimage.distance_transform_edt(~joint)))


def assemble(cart, r):
    """Cut the glass, choose every piece from the racks, foil and lead it, and hold it up to the light."""
    lab, n = glass.pieces(cart.keys)
    lab, n = glass.cut(lab, n, 150000, r, wobble=2.5)
    key = owner(lab, cart.keys, n)
    fam = np.asarray(cart.sheet)[key]
    fam[0] = 0
    tone = np.asarray(cart.tone, np.float32)[key] + r.normal(0, 1, n + 1).astype(np.float32) * table("jit")[fam]
    ang = np.asarray(cart.ang, np.float32)[key] + r.normal(0, 0.06, n + 1).astype(np.float32)
    drift = np.asarray(cart.drift, np.float32)[key]
    plate = np.asarray(cart.plate)[key]
    t0 = np.exp(r.normal(0, 1, n + 1) * table("thick")[fam]).astype(np.float32)
    milky = r.uniform(0, 1, n + 1).astype(np.float32)
    print("pieces", n)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    area = np.maximum(np.bincount(lab.ravel(), minlength=n + 1), 1)
    cx = np.bincount(lab.ravel(), xx.ravel(), minlength=n + 1) / area
    cy = np.bincount(lab.ravel(), yy.ravel(), minlength=n + 1) / area
    g = lab > 0
    L = lab[g]
    fm = fam[L]
    dx, dy = xx[g] - cx[L], yy[g] - cy[L]
    ca, sa = np.cos(ang[L]), np.sin(ang[L])
    ax, av = dx * ca + dy * sa, -dx * sa + dy * ca
    half = np.sqrt(3 * np.bincount(L, ax * ax, minlength=n + 1) / area) + 1
    st = table("stretch")[fam]
    st[0] = 1.0
    # every piece is cut from somewhere inside its sheet, never across the edge of it
    reach = np.minimum(np.stack([half / st, np.sqrt(3 * np.bincount(L, av * av, minlength=n + 1) / area) + 1]) + 12,
                       1000)
    ox, oy = (reach + r.uniform(0, 1, (2, n + 1)) * (2048 - 2 * reach)).astype(np.float32)
    U, V = ax / st[L] + ox[L], av + oy[L]
    fine, broad, white, milk, spots, folds, ripple = (ndimage.map_coordinates(f, [V, U], order=1, mode="mirror")
                                                      for f in stock(r))

    # where in its sheet's run of colour each pixel lies
    rip = table("ripple")[fm]
    p = (tone[L] + table("fine")[fm] * fine + table("broad")[fm] * broad + 0.1 * milk
         + drift[L] * np.clip(ax / half[L], -1.2, 1.2))
    spot = noise.smoothstep(0.9, 1.9, spots) * table("spots")[fm]
    p += 0.12 * rip * ripple - 1.0 * spot
    ncol = np.array([0] + [len(s.colours) for s in SHEETS.values()])
    CK = np.zeros((len(SHEETS) + 1, ncol.max(), 3), np.float32)
    for i, s in enumerate(SHEETS.values(), 1):
        ks = np.stack([pigment(c) for c in s.colours])
        CK[i, :len(ks)], CK[i, len(ks):] = ks, ks[-1]
    p = np.clip(p, 0, ncol[fm] - 1.001)
    i = p.astype(np.int64)
    e = table("edge")[fm]
    f = noise.smoothstep(0.5 - e, 0.5 + e, p - i)[:, None]
    k = CK[fm, i] * (1 - f) + CK[fm, i + 1] * f
    # streaks of white opal through the colour, more in some pieces than others
    wo = np.clip(table("white")[fm] * (0.35 * milky[L] + noise.smoothstep(-0.3, 1.5, white)), 0, 0.85)
    k = k * (1 - wo)[:, None] + 0.5 * CK[fm, 0] * wo[:, None]

    # thickness: the piece's own and clouds of milk. A drapery fold is thick along its crest and to one
    # side of it, and its other flank gathers the light into a bright line; ripple glass bends the light
    # into wavering lines.
    fa = table("folds")[fm]
    ridge, glint = np.exp(-(folds / 0.45) ** 2), np.exp(-((folds - 0.55) / 0.14) ** 2)
    d = t0[L] * np.exp(0.22 * milk + fa * (0.8 * ridge - 0.35 * np.tanh(1.5 * folds))) * (1 - 0.3 * spot)
    k = k + (1.6 * fa * ridge)[:, None] * FOLD
    lens_ = (1 + 1.0 * fa * glint) * (1 + 0.12 * rip * ripple)

    # confetti, and streamers, in the sheets that have them
    cf = table("flakes")[fm] > 0
    if cf.any():
        chip = ndimage.map_coordinates(confetti(r), [V[cf], U[cf]], order=0, mode="grid-wrap")
        FK = np.vstack([np.zeros(3), np.stack(FLAKES)]).astype(np.float32)
        k[cf] += FK[chip] * table("flakes")[fm[cf]][:, None]
        thr = ndimage.gaussian_filter(noise.fibers((1024, 1024), r, 260, length=(120, 520), width=2, curl=0.07,
                                                   strength=(0.5, 1.0)), 0.6)
        k[cf] += ndimage.map_coordinates(thr, [V[cf] * 0.9, U[cf] * 0.9], order=1, mode="grid-wrap")[:, None] * THREAD

    light = (1.04 - 0.1 * yy / H) * (1 + 0.04 * noise.field((H, W), 600, r))     # the sky behind, brighter overhead
    img = np.zeros((H, W, 3), np.float32)
    img[g] = np.exp(-k * d[:, None]) * (lens_ * light[g])[:, None]
    img *= np.exp(0.04 * noise.field((H, W), 2.0, r))[..., None]       # the rolled skin of the sheet

    # the plating over the far hills: a milky sheet behind, softening the glass and its own seams faint
    pl = np.zeros((H, W), bool)
    pl[g] = plate[L]
    if pl.any():
        soft = np.stack([ndimage.gaussian_filter(img[..., c], 2.5) for c in range(3)], -1)
        img[pl] = (0.5 * img[pl] + 0.5 * soft[pl]) * lin("#e8ecf8") * 1.04
        seams = ndimage.gaussian_filter(((xx + 0.35 * (yy - 1400)) % 560 < 3).astype(np.float32), 2.2) * pl
        img *= (1 - 0.25 * seams)[..., None]

    lead = leading(lab, fam, r)
    lc = LEAD * (1 + 0.25 * noise.field((H, W), 3.0, r))[..., None]
    img = img * (1 - lead[..., None]) + lc * lead[..., None]

    # the iron bars across the window, and the frame
    bar = np.zeros((H, W), np.float32)
    for yb in BARS:
        bar = np.maximum(bar, noise.smoothstep(5.6, 4.4, np.abs(yy - yb)))
    img = img * (1 - bar[..., None]) + IRON * bar[..., None]
    frame = ~g
    img[frame] = lin("#1f1813") * (1 + 0.15 * noise.fbm((H, W), 20, r, octaves=3)[frame])[:, None]
    img = glass.halation(img, strength=(0.3, 0.36, 0.44), spread=(1.6, 2.4, 3.4), veil=0.05)
    knee = 0.78                              # the brightest glass rolls off toward white, never burns out
    return np.where(img < knee, img, knee + (1 - knee) * (1 - np.exp(-(img - knee) / (1 - knee))))


def paint(seed=1908):
    r = noise.rng(seed)
    yy = np.arange(H, dtype=np.float32)[:, None]
    cart = Cartoon()
    far, mid = ridges(r)
    sky(cart, r, yy)
    hills(cart, r, yy, far, mid)
    land(cart, r, yy, mid)
    with cart:
        magnolia(cart, r)
        wisteria(cart, r)
        irises(cart, r)
    cart.keys[:Y0], cart.keys[Y1:], cart.keys[:, :X0], cart.keys[:, X1:] = 0, 0, 0, 0
    return assemble(cart, r)
