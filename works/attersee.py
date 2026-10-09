"""A Cottage Garden at Litzlberg. Oil on a square canvas, thinly painted, the flowers in small raised touches.

From 1900 Gustav Klimt spent his summers on the Attersee, and from about 1905 he painted the farm
gardens there. He chose his motif through a viewfinder, a card with a square hole cut in it, and set
it down on a square canvas so close that the garden fills it to its edges, with no sky and no ground
to stand on: a carpet of leaves and flowers as flat as a tapestry, laid in countless small touches of
colour over a cool green ground, the flowers gathered in drifts as they grow and tall sunflowers or
rose bushes standing up in them. The paint is fairly thin, and only the flowers stand up from it.

This garden is not one of his. Sunflowers stand on the left with their great leaves hanging grey-green,
some broad, some folded or turned edge-on, some heads facing out and one turned away; white dahlias
and a few crimson ones stand at the top, phlox in white and pink below them, and the lower part is
a carpet of flowers packed edge to edge, drifts of orange marigolds, scarlet poppies, violet asters,
white marguerites, pink and blue running into one another. The ground was rubbed in thin, a deep
viridian, and over it, dry, went the leaves in small flat touches every way, blue-green, viridian,
grey-blue and violet, with points of light green and points of colour sown through them at every
height, the light even over the whole square. Then the tall plants, back to front, and leaves drawn
up in front of the sunflowers' stalks; then the carpet in three sowings, flat rosettes, rounds,
stars and single touches of pure colour, the first half buried in more green touches, the last the
brightest.
"""

import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "A Cottage Garden at Litzlberg"
DATE = "2026"
MEDIUM = "Oil on canvas, thinly painted, the flowers in small raised touches of pure colour"
AFTER = ("Gustav Klimt, the garden paintings of his summers on the Attersee: Rosebushes under the Trees (1905), "
         "Farm Garden with Sunflowers (1907), Cottage Garden (Bauerngarten, 1907), Farm Garden with Crucifix (1912)")
ROOM = "The Garden"
YEAR = 1907
PLACE = "Litzlberg, Attersee"
REGION = "Europe"
NOTE = ("A square of cottage garden as flat as a tapestry: sunflowers, white dahlias and phlox standing in a deep "
        "green matrix sown with points of colour, over a carpet of marigolds, poppies, asters and marguerites "
        "packed edge to edge.")

H = W = 2600
LIGHT = (-0.6, -0.5, 0.62)                   # the gallery lamp on the paint
LIT = float(np.arctan2(-0.5, -0.6))          # the side the sunflowers and dahlias are modelled from
SUNV = np.array([np.cos(LIT), np.sin(LIT)])


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each from dark to light. The matrix: Klimt's cool greens, viridian and blue-green, with violet in them
DEEP = pal("#173430", "#1e403a", "#264c43", "#2f594d", "#386656")
VIRIDIAN = pal("#0e3a35", "#134a43", "#1a5b51", "#226c5f", "#2e7d6d", "#3f8f7d", "#56a18e")
BLUEGREEN = pal("#33615a", "#3f7067", "#4c7f73", "#5b8e80", "#6d9d8d", "#82ad9c")
GREYBLUE = pal("#4f6a80", "#5c788c", "#6a8698", "#7a94a4", "#8ba3b2", "#9fb3c0")
GREEN = pal("#3f6a35", "#4b793b", "#588943", "#66984b", "#76a655", "#88b462")
YGREEN = pal("#7d9a3c", "#8ea944", "#9fb84d", "#afc558", "#bfd166", "#cddc79")
CHART = pal("#a9ae2a", "#bbbd34", "#cbc941", "#d9d555", "#e5e070")
SHADE = pal("#2c3560", "#36406f", "#424c7e", "#4f5a8c", "#5e6a9a")
MAUVE = pal("#3a3158", "#4a3e6c", "#5b4d80", "#6d5e92", "#8372a6")
OLIVE = pal("#4f5a2c", "#5f6a33", "#707a3b", "#828a45")
# the plants
SUNLEAF = pal("#44614f", "#517059", "#5f7f64", "#6e8e6f", "#7f9c7c", "#92aa8b", "#a6b99d")
VEIN = pal("#8aa58a", "#9db59a", "#b0c3a8")
LEAFEDGE = pal("#2b4245", "#33504f", "#3c5c58")
STALK = pal("#4d6e45", "#5a7c4c", "#698b55", "#7a9a5f")
BRACT = pal("#3d6236", "#4c7340", "#5d844a", "#6f9455", "#84a464")
DLEAF = pal("#1d3a2c", "#264833", "#30573b", "#3b6644", "#48764d")
LEAF = pal("#2e5634", "#3a663b", "#477643", "#55864c", "#659556")
STEM = pal("#3f6638", "#4d7540", "#5c8449", "#6c9352")
# the flowers
RAY = pal("#b4700a", "#cc860e", "#de9e16", "#e9b424", "#f0c73a", "#f5d85c", "#f8e584")
DISC = pal("#24180e", "#312012", "#402a15", "#523619", "#66441e", "#7a5424")
FLORET = pal("#6e3e10", "#8a5016", "#a4661e", "#bb7c28", "#cf9434")
SEED = pal("#1e150c", "#3a2a14", "#5a4a20", "#7a6a2a", "#97853a")
ORANGE = pal("#b43a0a", "#c94d0e", "#dc6312", "#e8791a", "#ef9124", "#f3a936", "#f6c050")
SCARLET = pal("#8e1010", "#ac1814", "#c62416", "#da361a", "#e64e24", "#ee6a34")
CRIMSON = pal("#3e0614", "#560a1c", "#700f25", "#8a162f", "#a3213b", "#ba344c")
WHITE = pal("#a7a2b8", "#bdb8c8", "#d2cdd6", "#e4dfe0", "#f1ede7", "#f9f6ef", "#fefcf6")
PINK = pal("#b04c72", "#c4658a", "#d580a0", "#e29db6", "#ecbacb", "#f4d6df")
LILAC = pal("#7a64a6", "#8e78b6", "#a38ec4", "#b8a5d2", "#cdbfe0")
VIOLET = pal("#341a6a", "#442380", "#562e96", "#6a3daa", "#8156bb", "#9a74ca")
PURPLE = pal("#4e115a", "#661a72", "#7f268a", "#98389f", "#ae52b2")
BLUE = pal("#18227e", "#202f96", "#2a40ac", "#3a54bc", "#5170c8", "#6f8ed4")
GOLD = pal("#b07a0c", "#c88f14", "#dca622", "#e9bd36", "#f1d058")
HEART = pal("#170f1a", "#231626", "#30202e", "#3e2c30")
EYE = pal("#6e0f3c", "#8a164e", "#a52060", "#bd3272")
PINKHEART = pal("#c0567a", "#d77898", "#e8b0b8", "#efd6b8")
cat = np.concatenate
FAMILIES = {      # each paint: its palette, the accents picked up into it, and how often
    "deep": (DEEP, SHADE[1:4], 0.2), "viridian": (VIRIDIAN, BLUEGREEN[2:5], 0.2),
    "bluegreen": (BLUEGREEN, GREYBLUE[2:5], 0.2), "greyblue": (GREYBLUE, BLUEGREEN[2:5], 0.2),
    "green": (GREEN, YGREEN[1:4], 0.2), "ygreen": (YGREEN, CHART[1:4], 0.25), "chart": (CHART, YGREEN[2:], 0.3),
    "shade": (SHADE, DEEP[2:], 0.25), "mauve": (MAUVE, SHADE[2:4], 0.25),
    "sunleaf": (SUNLEAF, cat([GREYBLUE[2:4], BLUEGREEN[3:5], OLIVE[2:]]), 0.22), "vein": (VEIN, SUNLEAF[4:], 0.2),
    "leafedge": (LEAFEDGE, SHADE[1:3], 0.3), "stalk": (STALK, SUNLEAF[3:6], 0.25), "bract": (BRACT, GREEN[2:4], 0.2),
    "dleaf": (DLEAF, SHADE[2:4], 0.2), "leaf": (LEAF, BLUEGREEN[2:4], 0.2), "stem": (STEM, BLUEGREEN[2:4], 0.2),
    "ray": (RAY, ORANGE[3:6], 0.15), "disc": (DISC, FLORET[:2], 0.2), "floret": (FLORET, RAY[1:3], 0.25),
    "seed": (SEED, FLORET[1:3], 0.2), "orange": (ORANGE, RAY[2:5], 0.2), "scarlet": (SCARLET, CRIMSON[3:], 0.2),
    "crimson": (CRIMSON, SCARLET[:2], 0.2), "white": (WHITE, cat([PINK[4:], LILAC[3:]]), 0.15),
    "pink": (PINK, cat([WHITE[4:], LILAC[2:4]]), 0.2), "lilac": (LILAC, cat([PINK[3:5], WHITE[4:]]), 0.2),
    "violet": (VIOLET, PURPLE[2:4], 0.2), "purple": (PURPLE, VIOLET[2:4], 0.25),
    "blue": (BLUE, VIOLET[3:5], 0.2), "gold": (GOLD, ORANGE[3:5], 0.2), "heart": (HEART, CRIMSON[:2], 0.25),
    "eye": (EYE, CRIMSON[2:4], 0.2), "pinkheart": (PINKHEART, GOLD[2:4], 0.3),
}
# the touches of the matrix, from the deepest paint to the lightest, and how much of each
MATRIX = np.array(["deep", "viridian", "bluegreen", "greyblue", "mauve", "shade", "green", "ygreen"])
MATRIX_CDF = np.array([0.13, 0.39, 0.67, 0.77, 0.83, 0.89, 0.97])
# the points of colour sown through it, and how much of each
JEWELS = np.array(["scarlet", "white", "violet", "pink", "gold", "orange", "lilac", "blue", "crimson", "purple"])
JEWEL_ODDS = np.array([0.18, 0.06, 0.12, 0.1, 0.1, 0.07, 0.08, 0.09, 0.09, 0.07])

# the sunflowers: foot x, y; head x, y; head radius; how far the face tips (1 full on) and turns; face or back or bud
STAND = ((620, 1700, 560, 40, 150, 0.8, 0.1, "face"), (470, 1820, 380, 330, 155, 0.92, -0.1, "face"),
         (820, 1760, 860, 230, 135, 0.75, 0.35, "face"), (1080, 1780, 1150, 270, 55, 0.9, 0.0, "bud"),
         (980, 1800, 1060, 560, 115, 0.5, 0.6, "face"), (220, 1850, 110, 690, 145, 0.88, -0.3, "face"),
         (700, 1880, 650, 640, 160, 0.95, 0.05, "face"), (430, 1900, 380, 1000, 120, 0.62, -0.4, "back"))
DAHLIAS = ((1480, 330, 80, "white"), (1720, 250, 90, "white"), (1990, 360, 84, "white"), (2260, 280, 92, "white"),
           (1620, 540, 76, "white"), (2120, 570, 86, "white"), (2440, 500, 80, "white"), (1860, 470, 72, "white"),
           (1320, 480, 68, "crimson"), (2510, 230, 72, "crimson"), (1830, 690, 64, "crimson"), (2350, 730, 62, "crimson"))
PHLOX = ((1560, 960, 140, 98, "white"), (1910, 880, 125, 90, "pink"), (2260, 1000, 150, 102, "white"),
         (2540, 850, 105, 80, "lilac"), (2450, 1200, 118, 84, "white"), (1370, 1140, 104, 74, "pink"))
# the carpet. Each flower: its size, the odds of each form it takes (FORMS), the centres it may have
FORMS = ("ring", "round", "star", "cluster", "dab")
KINDS = {
    "orange": (40, (0.15, 0.65, 0.05, 0.0, 0.15), ("gold", "gold", "heart", None, None)),
    "gold": (36, (0.3, 0.55, 0.05, 0.0, 0.1), ("orange", "heart", None)),
    "scarlet": (38, (0.05, 0.4, 0.4, 0.0, 0.15), ("heart", "heart", "gold", None)),
    "crimson": (36, (0.1, 0.55, 0.25, 0.0, 0.1), ("heart", "gold", None)),
    "violet": (33, (0.55, 0.25, 0.0, 0.1, 0.1), ("gold", "gold", "chart", None)),
    "purple": (35, (0.45, 0.4, 0.0, 0.05, 0.1), ("gold", "white", "chart")),
    "lilac": (33, (0.55, 0.25, 0.0, 0.1, 0.1), ("gold", "chart", None)),
    "white": (31, (0.75, 0.15, 0.0, 0.0, 0.1), ("gold", "gold", "chart")),
    "pink": (35, (0.4, 0.45, 0.0, 0.05, 0.1), ("gold", "eye", "white", None)),
    "blue": (27, (0.3, 0.15, 0.0, 0.45, 0.1), ("white", "gold", None, None)),
}
# where each grows thickest: a line through the drift and how far it spreads to either side
DRIFTS = (("orange", ((2680, 1230), (2380, 1460), (2120, 1720), (1880, 2020), (1700, 2330), (1600, 2680)), 380),
          ("orange", ((60, 2450), (380, 2620)), 200), ("gold", ((2560, 2380), (2380, 2620)), 200),
          ("scarlet", ((120, 1450), (480, 1700), (760, 2000), (560, 2280), (240, 2300)), 330),
          ("crimson", ((900, 1480), (1060, 1720)), 150), ("crimson", ((1960, 2420), (2120, 2520)), 140),
          ("violet", ((1080, 1330), (1360, 1470), (1280, 1700)), 240), ("purple", ((960, 2280), (1300, 2480)), 240),
          ("violet", ((2200, 2330), (2560, 2150)), 200), ("purple", ((1900, 1250), (2150, 1330)), 160),
          ("blue", ((1500, 1320), (1700, 1450)), 140), ("blue", ((560, 2060), (720, 2180)), 120),
          ("white", ((200, 2560), (820, 2480), (1300, 2600)), 190), ("white", ((2300, 1980), (2600, 1900)), 170),
          ("pink", ((1460, 1860), (1380, 2140)), 190), ("lilac", ((2380, 1560), (2600, 1700)), 180))
STRAYS = {"white": 0.12, "violet": 0.2, "scarlet": 0.18, "orange": 0.1, "pink": 0.1, "purple": 0.1}   # seeded abroad


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def ramp(colours, tone):
    """The colour at `tone` (0 dark .. 1 light) along a palette, for a whole field of tones."""
    t = np.clip(tone, 0, 1) * (len(colours) - 1)
    i = np.minimum(t.astype(int), len(colours) - 2)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def loads(colours, tone, r, accent=None, odds=0.2, spread=0.06):
    """The brush for each stroke: the paint for its tone, a neighbour on the palette streaked in,
    and a third, now and then an accent picked up from elsewhere."""
    n, N = len(colours), len(tone)
    t = np.clip(tone + r.normal(0, spread, N), 0, 0.999) * n
    i = t.astype(int)
    j = np.clip(i + np.where(r.random(N) < t - i, 1, -1), 0, n - 1)
    third = colours[np.clip(i + r.choice([-2, 2], N), 0, n - 1)]
    if accent is not None:
        hit = r.random(N) < odds
        third[hit] = accent[r.integers(0, len(accent), hit.sum())]
    share = np.stack([r.uniform(0.45, 0.7, N), r.uniform(0.2, 0.4, N), r.uniform(0.02, 0.2, N)], 1)
    return np.stack([colours[i], colours[j], third], 1), share


def paints(fam, tone, r):
    """Loads for strokes each naming its paint in FAMILIES."""
    fam, tone = np.asarray(fam), np.asarray(tone, np.float64)
    cols, share = np.empty((len(fam), 3, 3), np.float32), np.empty((len(fam), 3))
    for f in np.unique(fam):
        sel = fam == f
        colours, accent, odds = FAMILIES[f]
        cols[sel], share[sel] = loads(colours, tone[sel], r, accent, odds)
    return cols, share


class Strokes:
    """Strokes gathered in the order the brush lays them, each naming its paint and tone."""

    def __init__(self):
        self.paths, self.w, self.fam, self.tone, self.dry = [], [], [], [], []

    def add(self, path, w, fam, tone, dry=0.0):
        path = np.asarray(path, np.float32)
        if len(path) >= 2:
            self.paths.append(path)
            self.w.append(max(float(w), 1.2))
            self.fam.append(fam)
            self.tone.append(float(tone))
            self.dry.append(dry)

    def extend(self, other):
        for a in ("paths", "w", "fam", "tone", "dry"):
            getattr(self, a).extend(getattr(other, a))


# ---- the shapes of things, each a few strokes of a loaded brush

def frame(c, phi, k):
    """From a flower's own plane (x across its face, y up it, squashed by k as it tips away), turned
    by phi, to the canvas."""
    cs, sn = np.cos(phi), np.sin(phi)

    def to(px, py):
        py = py * k
        return np.stack([c[0] + cs * px - sn * py, c[1] + sn * px + cs * py], -1)
    return to


def rays(st, c, R, k, phi, m, fam, tone, r, reach=(0.3, 1.0), wide=0.1, curl=0.2, lit=0.3, inward=0.3, dry=0.0):
    """Petals round a heart, each one stroke pulled out from it or pressed in toward it, no two alike;
    the far ones first, those turned to the light lighter."""
    to = frame(c, phi, k)
    th = r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + r.normal(0, 0.6 * np.pi / m, m)
    q = np.linspace(0, 1, 4)
    for t in th[np.argsort(np.sin(th))]:
        a = t + r.normal(0, curl) * q
        r0, r1 = R * reach[0] * r.uniform(0.7, 1.3), R * reach[1] * r.uniform(0.82, 1.1)
        rad = r0 + (r1 - r0) * q
        path = to(rad * np.cos(a), rad * np.sin(a))
        w = R * wide * r.uniform(0.8, 1.2) * np.hypot(np.sin(t), k * np.cos(t))
        ang = np.arctan2(k * np.sin(t), np.cos(t)) + phi
        st.add(path[::-1] if r.random() < inward else path, w, fam, tone + lit * np.cos(ang - LIT) + r.normal(0, 0.05), dry)


def disc(st, c, R, k, phi, fam, tone, r, n=3, wide=0.7, lit=0.3, dry=0.0):
    """Something round in a few short broad touches laid across its middle every way, the one
    nearest the light the lightest."""
    to = frame(c, phi, k)
    for a0 in r.uniform(0, np.pi) + np.arange(n) * np.pi / n + r.normal(0, 0.25, n):
        off = r.normal(0, 0.12, 2) * R
        d = 0.5 * R * r.uniform(0.7, 1.0) * np.array([np.cos(a0), np.sin(a0)])
        path = to(np.array([off[0] - d[0], off[0] + d[0]]), np.array([off[1] - d[1], off[1] + d[1]]))
        toward = (path.mean(0) - c) @ SUNV / R
        st.add(path, R * wide * r.uniform(0.85, 1.1) * (0.6 + 0.4 * k), fam, tone + 3 * lit * toward + r.normal(0, 0.04), dry)


def touch(st, p, a, L, w, fam, tone, dry=0.0):
    """One short touch centred on p."""
    d = np.array([np.cos(a), np.sin(a)]) * L / 2
    st.add(np.stack([p - d, p + d]), w, fam, tone, dry)


def leaf(st, base, a, L, w, fam, tone, r, dry=0.0):
    """A leaf in one stroke from where it joins to its point, bowed a little."""
    base = np.asarray(base, float)
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-d[1], d[0]])
    bow = r.normal(0, 0.12) * L
    st.add(np.stack([base, base + 0.5 * L * d + bow * n, base + L * d + 0.6 * bow * n]), w, fam, tone, dry)


def stem(st, top, L, w, r, lean=0.0, fam="stem"):
    """A stem drawn up out of the ground to the flower."""
    top = np.asarray(top, float)
    bow = r.normal(0, 0.06) * L
    st.add(np.stack([top + [lean * L, L], top + [lean * 0.5 * L + bow, 0.5 * L], top]), w, fam, r.uniform(0.3, 0.7))


def sunflower(st, c, R, k, phi, kind, shade, r):
    """A sunflower head: a ring of yellow rays, the back row darker, round a dark disc packed with
    seeds and rimmed with florets; or seen from behind, green bracts with the rays showing round
    them; or a bud."""
    c = np.asarray(c, float)
    if kind == "bud":
        rays(st, c, R, k, phi, r.integers(9, 13), "bract", 0.45 + shade, r, reach=(0.1, 1.0), wide=0.22, lit=0.3, inward=0.2)
        rays(st, c, 0.7 * R, k, phi, r.integers(5, 8), "ray", 0.5 + shade, r, reach=(0.6, 1.0), wide=0.14, lit=0.2)
        disc(st, c, 0.5 * R, k, phi, "bract", 0.4 + shade, r)
        return
    if kind == "back":
        rays(st, c, R, k, phi, r.integers(17, 23), "ray", 0.4 + shade, r, reach=(0.55, 1.0), wide=0.11, curl=0.3,
             inward=0.1)
        rays(st, c, 0.78 * R, k, phi, r.integers(14, 19), "bract", 0.45 + shade, r, reach=(0.15, 1.0), wide=0.13,
             inward=0.2)
        disc(st, c, 0.45 * R, k, phi, "bract", 0.35 + shade, r, n=4)
        return
    rays(st, c, R, k, phi, r.integers(15, 20), "ray", 0.42 + shade, r, reach=(0.42, 1.0), wide=0.12, curl=0.3,
         lit=0.32, inward=0.15)
    rays(st, c, 0.9 * R, k, phi, r.integers(14, 19), "ray", 0.6 + shade, r, reach=(0.42, 1.0), wide=0.12, curl=0.35,
         lit=0.32, inward=0.15)
    D = R * r.uniform(0.42, 0.5)
    to = frame(c, phi, k)
    for rr in (0.72, 0.42, 0.12):                                                             # the disc, laid dark
        m = max(2, int(2 * np.pi * rr / 0.5))                                                 # round and round,
        for t in r.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m:
            tt = t + np.linspace(0, 1.4 * 2 * np.pi / m, 4)
            st.add(to(rr * D * np.cos(tt), rr * D * np.sin(tt)), 0.22 * D, "disc", 0.12 + 0.3 * shade + r.normal(0, 0.05))
    for _ in range(int(np.clip(2.0 * (D / 4.5) ** 2, 100, 650))):                              # then stippled close in
        q, t = D * 0.93 * np.sqrt(r.uniform()), r.uniform(0, 2 * np.pi)                         # seed, browner toward
        ang = np.arctan2(k * np.sin(t), np.cos(t)) + phi                                        # its rim
        touch(st, to(q * np.cos(t), q * np.sin(t)), r.uniform(0, np.pi), r.uniform(0.5, 2.5), r.uniform(3.4, 5.2),
              "seed" if r.random() < 0.55 else "disc",
              0.3 + 0.5 * (q / D) ** 2 + 0.15 * np.cos(ang - LIT) * q / D + r.normal(0, 0.18) + 0.3 * shade)
    m = int(np.clip(2 * np.pi * D / 8, 18, 50))                                               # and ringed in florets
    for t in np.linspace(0, 2 * np.pi, m, endpoint=False) + r.normal(0, 0.1, m):              # where the light falls
        ang = np.arctan2(k * np.sin(t), np.cos(t)) + phi
        if np.cos(ang - LIT) < r.uniform(-0.6, 0.4):
            continue
        q = D * np.array([0.86, r.uniform(0.94, 1.02)])
        st.add(to(q * np.cos(t), q * np.sin(t)), max(1.6, 0.05 * D), "floret", 0.35 + 0.4 * np.cos(ang - LIT) + r.normal(0, 0.12))


def heart(st, A, a0, L, Wm, sag, k, tone, fam, r):
    """A sunflower leaf hanging from its stalk, no two alike. Most are seen broad; some are folded
    along the midrib, so that one half shows narrow and darker, and a few are turned edge-on to a
    blade. Each half is filled in one broad stroke down its length. Over that, on some, the veins:
    strokes from the midrib out to the edge, slanting toward the point, whose ends make the toothed
    edge; on others a few long strokes down the leaf; the midrib and the shadowed edge on some."""
    form = r.choice(["veined", "flat", "folded", "edge"], p=[0.35, 0.3, 0.2, 0.15])
    if form == "edge":
        k = r.uniform(0.14, 0.26)
    s = np.linspace(0, 1, 24)
    a = a0 + sag * s ** 1.5
    d = np.stack([np.cos(a), np.sin(a)], 1)
    S = np.asarray(A, float) + np.vstack([[0, 0], np.cumsum(d[:-1] * L / 23, 0)])
    N = np.stack([-d[:, 1], d[:, 0]], 1)
    at_ = lambda arr, sv: np.array([np.interp(sv, s, arr[:, j]) for j in range(arr.shape[1])]) if arr.ndim > 1 else np.interp(sv, s, arr)
    lat = np.sign(N.mean(0) @ SUNV + 1e-6)
    fold = r.choice([-1, 1])
    hw = float(np.clip(0.045 * L * r.uniform(0.8, 1.3), 6, 16))
    slant0, gap = r.uniform(0.25, 0.8), r.uniform(0.85, 1.5)
    for side in (-1, 1):
        narrow = form == "folded" and side == fold
        f = r.uniform(0.2, 0.4) if narrow else 1.0
        half = Wm * k * f * np.clip(s ** 0.45 * (1 - s) ** 0.85 / 0.432, 0, 1)
        wide = 0.5 * Wm * k * f
        inside = np.clip(1 - 0.85 * wide / (half + 1e-6), 0, 1)
        t0 = tone + (0.04 if side == lat else -0.03) - (0.14 if narrow else 0.0) - (0.06 if form == "flat" else 0.0)
        st.add((S + side * (0.5 * half * inside)[:, None] * N)[1:22], wide, fam, t0)
        if form == "flat":                                       # a few long strokes down it
            for u in r.uniform(0.2, 0.85, r.integers(3, 6)):
                i0, i1 = r.integers(1, 6), r.integers(14, 22)
                st.add((S + side * (u * half)[:, None] * N)[i0:i1], max(3.0, Wm * k * r.uniform(0.06, 0.11)), fam,
                       t0 + r.choice([-1, 1]) * r.uniform(0.08, 0.18))
        if form not in ("veined", "folded") or narrow:
            continue
        for sv in np.arange(r.uniform(0.02, 0.06), 0.9, gap * hw / L):                    # the veins
            if r.random() < 0.15:
                continue
            sv = sv + r.normal(0, 0.15 * hw / L)
            slant = slant0 + r.normal(0, 0.1)
            n_, d_ = at_(N, sv), at_(d, sv)
            e = side * n_ * np.cos(slant) + d_ * np.sin(slant)
            reach = min(at_(half, sv), at_(half, min(sv + 0.5 * at_(half, sv) * np.tan(slant) / L, 1))) * r.uniform(0.9, 1.0) \
                - 0.55 * hw
            if reach < 0.5 * hw:
                continue
            p0 = at_(S, sv) + side * n_ * 0.25 * hw
            ln = reach / np.cos(slant)
            bow = r.normal(0, 0.06) * ln * np.array([-e[1], e[0]])
            st.add(np.stack([p0, p0 + 0.5 * ln * e + bow, p0 + ln * e]), hw * r.uniform(0.85, 1.1), fam,
                   t0 + 0.04 + r.normal(0, 0.035))
    if form in ("veined", "folded") or r.random() < 0.4:
        st.add(S[1:20], max(2.0, 0.01 * L), "vein", tone)
    if form == "flat" or r.random() < 0.6:
        edge = (S - lat * (Wm * k * np.clip(s ** 0.45 * (1 - s) ** 0.85 / 0.432, 0, 1))[:, None] * N)[1:23]
        for m in np.array_split(np.arange(len(edge)), 3):
            if r.random() < 0.75:
                st.add(edge[m[0]:m[-1] + 2], max(1.8, 0.008 * L), "leafedge", tone + 0.15)


def sunflower_plant(foot, head, R, k, phi, kind, shade, r):
    """A sunflower: its stalk drawn up from the ground, a little bowed, its great leaves alternating up
    it, hanging, rising or turned, the lower the larger, and the head. -> green strokes, flower strokes"""
    g, b = Strokes(), Strokes()
    up = head - foot
    span = np.hypot(*up)
    t = np.linspace(0, 1, 12)[:, None]
    S = foot + t * up + np.array([-up[1], up[0]]) / span * r.normal(0, 0.04) * span * np.sin(np.pi * t)
    w0 = np.clip(0.07 * R, 7, 12)
    for m in np.array_split(np.arange(12), 3):
        g.add(S[max(m[0] - 1, 0):m[-1] + 1], w0 * r.uniform(0.9, 1.1), "stalk", r.uniform(0.35, 0.65))
    g.add(S[:-1] + [0.8 * w0, 0], max(1.5, 0.3 * w0), "leafedge", 0.5)
    side = r.choice([-1, 1])
    for f in np.sort(r.uniform(0.2, 0.93, r.integers(7, 11))):
        A = foot + f * up
        side = -side
        L = r.uniform(170, 440) * (1.2 - 0.45 * f) * (0.6 + 0.4 * R / 140)
        a0 = (0 if side > 0 else np.pi) - side * r.uniform(-0.45, 1.05)
        e = np.array([np.cos(a0), np.sin(a0)])
        g.add(np.stack([A, A + 0.14 * L * e]), 4.0, "stalk", 0.5)
        heart(g, A + 0.12 * L * e, a0, L, L * r.uniform(0.4, 0.6), side * r.uniform(-0.3, 1.9), r.uniform(0.5, 1.0),
              0.5 + 1.5 * shade + r.normal(0, 0.15),
              r.choice(["sunleaf", "sunleaf", "sunleaf", "bluegreen", "viridian"]), r)
    sunflower(b, head, R, k, phi, kind, shade, r)
    return g, b


def dahlia(c, R, fam, shade, r):
    """A dahlia on its tall stem in a bush of dark leaves: rows of thin petals from the rim in, curling
    every way, and a small heart."""
    g, b = Strokes(), Strokes()
    c = np.asarray(c, float)
    L, lean = R * r.uniform(3.5, 5.5), r.normal(0, 0.12)
    stem(g, c + [0, 0.5 * R], L, np.clip(0.05 * R, 2.5, 4), r, lean)
    for _ in range(r.integers(6, 11)):
        f = r.uniform(0.0, 0.8)
        leaf(g, c + [lean * f * L + r.normal(0, 0.3 * R), 0.5 * R + f * L], r.uniform(-np.pi, np.pi), R * r.uniform(0.5, 0.9),
             R * r.uniform(0.12, 0.18), r.choice(["dleaf", "dleaf", "leaf", "bluegreen"]), r.uniform(0.15, 0.7) + shade, r)
    k, phi = r.uniform(0.7, 1.0), r.normal(0, 0.3)
    for s_, m, reach, wide, tone in ((1.0, (16, 22), (0.55, 1.0), 0.09, 0.55), (0.82, (15, 19), (0.4, 1.0), 0.1, 0.64),
                                     (0.64, (13, 17), (0.25, 1.0), 0.11, 0.72), (0.46, (10, 13), (0.1, 1.0), 0.12, 0.8),
                                     (0.28, (6, 9), (0.0, 1.0), 0.15, 0.86)):
        rays(b, c, s_ * R, k, phi, r.integers(*m), fam, tone + shade, r, reach=reach, wide=wide, curl=0.5, lit=0.25,
             inward=0.5)
    touch(b, c, r.uniform(0, np.pi), 0.12 * R, 0.1 * R, "pinkheart" if fam == "white" else "crimson",
          r.uniform(0.5, 0.9) if fam == "white" else 0.1)
    for _ in range(r.integers(2, 5)):
        touch(b, c + r.normal(0, 0.04 * R, 2), r.uniform(0, np.pi), 0.06 * R, max(1.4, 0.035 * R),
              "pinkheart" if fam == "white" else "crimson", r.uniform(0.0, 0.5) if fam == "white" else 0.0)
    return g, b


def phlox(c, rx, ry, fam, shade, r):
    """A head of phlox: stems standing under it with their leaves in pairs; the dome laid floret by
    floret from the top, each a round of two crossed touches, packed close, each with its coloured eye."""
    g, b = Strokes(), Strokes()
    c = np.asarray(c, float)
    for _ in range(r.integers(3, 6)):
        top = c + [r.uniform(-0.55, 0.55) * rx, 0.2 * ry]
        L, lean = ry * r.uniform(2.2, 3.6), r.normal(0, 0.08)
        stem(g, top, L, 3, r, lean)
        for f in np.linspace(0.3, 0.9, r.integers(2, 4)) + r.normal(0, 0.04):
            for side in (-1, 1):
                leaf(g, top + [lean * f * L, f * L], -np.pi / 2 + side * r.uniform(0.5, 1.1), r.uniform(32, 56),
                     r.uniform(5, 8), "leaf", r.uniform(0.25, 0.7) + shade, r)
    lobes, ph = r.normal(0, 0.12, 3), r.uniform(0, 2 * np.pi, 3)

    def rim(x, y):
        t = np.arctan2((y - c[1]) / ry, (x - c[0]) / rx)
        return np.hypot((x - c[0]) / rx, (y - c[1]) / ry) / (1 + sum(lobes[j] * np.cos((j + 2) * t + ph[j]) for j in range(3)))
    for _ in range(r.integers(4, 7)):
        touch(b, c + [r.uniform(-0.5, 0.5) * rx, r.uniform(-0.3, 0.4) * ry], r.normal(0, 0.5), r.uniform(0.3, 0.6) * rx,
              r.uniform(0.25, 0.35) * ry, fam, r.uniform(0.15, 0.4) + shade)
    rf = r.uniform(19, 25)
    pts = np.empty((0, 2))
    for _ in range(int(10 * rx * ry / rf ** 2)):
        p = c + r.uniform(-1.1, 1.1, 2) * [rx, ry]
        if rim(*p) < 0.95 and (len(pts) == 0 or ((pts - p) ** 2).sum(1).min() > (0.8 * rf) ** 2):
            pts = np.vstack([pts, p])
    for p in pts[np.argsort(pts[:, 1])]:
        lit = float(np.clip(((p - c) / [rx, ry]) @ SUNV, -1, 1))
        tone = 0.62 + 0.15 * lit - 0.15 * rim(*p) ** 2 + shade + r.normal(0, 0.07)
        a = r.uniform(0, np.pi)
        touch(b, p, a, 0.9 * rf, 0.75 * rf, fam, tone)
        touch(b, p + r.normal(0, 0.1 * rf, 2), a + np.pi / 2 + r.normal(0, 0.3), 0.75 * rf, 0.65 * rf, fam, tone + 0.06)
        touch(b, p + r.normal(0, 0.6, 2), r.uniform(0, np.pi), 0.12 * rf, max(1.4, 0.1 * rf), "eye", r.uniform(0.5, 1.0))
    return g, b


def rosette(st, c, R, fam, form, centre, tone, r, dry=0.0):
    """One flower of the carpet, flat, as Klimt paints them, and tipped a little every way: a ring of
    petals round a centre of another colour, a round of two or three broad touches, a star of broad
    petals, a cluster of florets, or a mere touch or two of colour."""
    k, phi = r.uniform(0.55, 1.0), r.uniform(-np.pi, np.pi)
    if form == "dab":
        for _ in range(r.integers(1, 3)):
            touch(st, c + r.normal(0, 0.15 * R, 2), r.uniform(0, np.pi), R * r.uniform(0.5, 1.0), R * r.uniform(0.35, 0.55),
                  fam, tone + r.normal(0, 0.08), dry)
        return
    if form == "cluster":
        for _ in range(r.integers(5, 11)):
            touch(st, c + r.normal(0, 0.42 * R, 2), r.uniform(0, np.pi), R * r.uniform(0.15, 0.3), R * r.uniform(0.15, 0.24),
                  fam, tone + r.normal(0, 0.12), dry)
        return
    if form == "ring":
        rays(st, c, R, k, phi, r.integers(10, 17), fam, tone, r, reach=(r.uniform(0.3, 0.5), 1.0), wide=r.uniform(0.16, 0.22),
             curl=0.25, lit=0.04, inward=0.3, dry=dry)
        hub = r.uniform(0.3, 0.45)
    elif form == "star":
        rays(st, c, R, k, phi, r.integers(5, 9), fam, tone, r, reach=(0.1, 1.0), wide=r.uniform(0.28, 0.36), curl=0.3,
             lit=0.04, inward=0.5, dry=dry)
        hub = r.uniform(0.2, 0.3)
    else:
        disc(st, c, 0.85 * R, k, phi, fam, tone, r, n=r.integers(2, 4), wide=0.7, lit=0.03, dry=dry)
        if r.random() < 0.5:                                    # its petal ends, lighter or deeper
            rays(st, c, 0.95 * R, k, phi, r.integers(5, 9), fam, tone + r.choice([-0.15, 0.15]), r, reach=(0.55, 1.0),
                 wide=0.16, lit=0.0, inward=0.5, dry=dry)
        hub = r.uniform(0.22, 0.38)
    if centre is not None:
        disc(st, c + r.normal(0, 0.05 * R, 2), hub * R, k, phi, centre, r.uniform(0.3, 0.85), r, n=2, wide=0.8, lit=0.0,
             dry=dry)


# ---- where things grow

def spread(P, line, half):
    """How thickly a drift grows at points P: along its line, thinning to either side."""
    L = np.asarray(line, float)
    a, ab = L[:-1], L[1:] - L[:-1]
    t = np.clip(((P[:, None] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
    d = np.linalg.norm(P[:, None] - a - t[..., None] * ab, axis=-1).min(1)
    return noise.smoothstep(1.3 * half, 0.35 * half, d)


def thin(P, R, gap):
    """Keep each spot in turn unless it lies too close to one kept before it. -> mask"""
    cell = 2 * gap * R.max() + 1
    grid, keep = {}, np.zeros(len(P), bool)
    for i, ((x, y), rad) in enumerate(zip(P, R)):
        cx, cy = int(x // cell), int(y // cell)
        near = [j for u in (cx - 1, cx, cx + 1) for v in (cy - 1, cy, cy + 1) for j in grid.get((u, v), ())]
        if not near or (np.hypot(*(P[near] - (x, y)).T) > gap * (R[near] + rad)).all():
            keep[i] = True
            grid.setdefault((cx, cy), []).append(i)
    return keep


def garden(fresh, light):
    """The tall plants, worked out before any paint goes on: (depth, alone, green, flowers), depth the y of
    the foot, so that nearer plants go on over farther ones; and the discs their heads cover."""
    items = []
    sh = lambda x, y: 0.4 * (light(x, y) - 0.5)
    r = fresh("sunflowers")
    for fx, fy, hx, hy, R, k, phi, kind in STAND:
        g, b = sunflower_plant(np.array([fx, fy], float), np.array([hx, hy], float), R, k, phi, kind, sh(hx, hy), r)
        items.append((fy, True, g, b))
    taken = [(hx, hy, 1.1 * R) for _, _, hx, hy, R, *_ in STAND]
    r = fresh("dahlias")
    for x, y, R, fam in DAHLIAS:
        items.append((y + 3 * R, False, *dahlia((x, y), R, fam, sh(x, y), r)))
        taken.append((x, y, R))
    r = fresh("phlox")
    for x, y, rx, ry, fam in PHLOX:
        items.append((y + 2 * ry, False, *phlox((x, y), rx, ry, fam, sh(x, y), r)))
        taken.append((x, y, 0.9 * max(rx, ry)))
    return items, np.array(taken)


def tufts(r):
    """Leaves drawn up in front of the sunflowers' lower stalks, in clumps, so that no stalk stands bare."""
    st = Strokes()
    for fx, fy, hx, hy, *_ in STAND:
        for y in r.uniform(1050, fy, r.integers(5, 9)):
            base = np.array([fx + (hx - fx) * (fy - y) / (fy - hy) + r.normal(0, 30), y + r.uniform(40, 110)])
            for _ in range(r.integers(7, 13)):
                leaf(st, base + r.normal(0, 35, 2), -np.pi / 2 + r.normal(0, 1.0), r.uniform(50, 110), r.uniform(13, 22),
                     r.choice(["dleaf", "leaf", "bluegreen", "viridian"]), r.uniform(0.2, 0.8), r)
    return st


def carpet(r, taken, zone, stand):
    """The flowers of the carpet, sown where the drifts let them grow, thick in the cores and thinning at
    the edges into one another, a few strays everywhere, never on a head already standing nor too close
    on one another. -> three sowings of strokes: the flowers half lost in the green, the main sowing, the
    brightest on top"""
    names = list(KINDS)
    P = r.uniform(0, 1, (16000, 2)) * [W, H]
    lumps = noise.field((H // 8, W // 8), 20, r)
    D = np.zeros((len(names), len(P)))
    for j, (fam, line, half) in enumerate(DRIFTS):
        ly, lx = ((P[:, 1] + 211 * j) % H).astype(int) // 8, ((P[:, 0] + 307 * j) % W).astype(int) // 8
        D[names.index(fam)] += spread(P, line, half) * noise.smoothstep(-1.0, 0.6, lumps[ly, lx])
    y, x = at(P)
    z = zone[y, x]
    D = D * (0.3 + 0.7 * z) * (1 - 0.8 * stand[y, x] * (1 - z)) + np.array([STRAYS.get(n, 0) for n in names])[:, None] * (0.025 + 0.12 * z)
    grow = r.random(len(P)) < np.clip(D.sum(0), 0, 0.95)
    kind = np.minimum((r.random(len(P)) > np.cumsum(D / D.sum(0), 0)).sum(0), len(names) - 1)
    R = np.array([KINDS[n][0] for n in names])[kind] * np.exp(r.normal(0, 0.25, len(P))) * np.where(r.random(len(P)) < 0.2, 0.5, 1)
    clear = (np.hypot(P[:, None, 0] - taken[:, 0], P[:, None, 1] - taken[:, 1]) > taken[:, 2] + 0.5 * R[:, None]).all(1)
    sel = np.flatnonzero(grow & clear)
    sel = sel[thin(P[sel], R[sel], 0.5)]
    sel = sel[np.argsort(P[sel, 1])]                         # the higher first, the lower over them
    out = (Strokes(), Strokes(), Strokes())
    for i in sel:
        name = names[kind[i]]
        size, odds, centres = KINDS[name]
        u = r.random()
        layer = 0 if u < 0.25 else 2 if u > 0.86 else 1
        tone = 0.55 + r.normal(0, 0.1) + (-0.15, 0.0, 0.1)[layer]
        rosette(out[layer], P[i], R[i], name, FORMS[r.choice(5, p=odds)], centres[r.integers(len(centres))], tone, r,
                dry=r.uniform(0.15, 0.45) if layer == 0 else 0.0)
    return out


def paint(seed=1907):
    def fresh(name):
        """Each passage draws on chances of its own, so the others stay as they are when one is changed."""
        return np.random.default_rng([seed, zlib.crc32(name.encode())])

    r = fresh("ground")
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ground = canvas.duck((H, W), seed, tint="#e8dfca", thread=3.2)

    # the light even over the whole square: the greens lighter and darker only in patches, a little deeper
    # among the sunflowers
    patch = noise.fbm((H, W), 420, r, octaves=3)
    trace = Image.new("L", (W // 4, H // 4), 0)
    pen = ImageDraw.Draw(trace)
    for fx, fy, hx, hy, *_ in STAND:
        pen.line([(fx / 4, fy / 4), (hx / 4, hy / 4)], fill=255, width=80)
    stand = ndimage.zoom(ndimage.gaussian_filter(np.asarray(trace, np.float32) / 255, 22), 4, order=1)[:H, :W]
    glight = np.clip(0.5 + 0.12 * patch + 0.05 * noise.fbm((H, W), 70, r, octaves=2) - 0.1 * stand, 0, 1).astype(np.float32)
    light = lambda x, y: float(glight[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
    top = (1230 + 70 * noise.line1d(W, 380, r)).astype(np.float32)          # where the carpet thickens, raggedly
    zone = noise.smoothstep(top[None] - 100, top[None] + 300, yy).astype(np.float32)

    items, taken = garden(fresh, light)

    # the ground rubbed in thin, a deep viridian, the weave of the canvas showing through it
    want = ramp(cat([DEEP, VIRIDIAN, BLUEGREEN]), glight)
    rag = noise.stretched((H, W), 14, 200, r)
    alpha = np.clip(0.82 + 0.06 * rag, 0, 1)[..., None]
    rgb = (ground.color * (1 - alpha) + want * (1 + 0.04 * rag[..., None]) * alpha).astype(np.float32)
    height = ground.tooth * 0.2
    wet = np.ones((H, W), np.float32)

    def strew(spacing, odds, rr):
        P = dabs.scatter((H + 2 * int(spacing), W + 2 * int(spacing)), spacing, rr) - int(spacing)
        return P[rr.random(len(P)) < odds[at(P)]]

    hue = noise.fbm((H, W), 160, r, octaves=2)          # which greens prevail where: they come in patches
    plenty = np.stack([noise.field((H // 16 + 1, W // 16 + 1), 14, r) for _ in JEWELS])   # so do the colours
    sown = noise.smoothstep(-0.4, 1.3, noise.field((H, W), 230, r))                        # and the small flowers

    def matrix(P, rr):
        """The paint for a touch of the matrix: the cool greens, each prevailing in patches, and violet."""
        y, x = at(P)
        u = np.clip(0.5 + 0.28 * hue[y, x] + rr.normal(0, 0.22, len(P)), 0, 1)
        return paints(MATRIX[np.searchsorted(MATRIX_CDF, u)], np.clip(glight[y, x] + rr.normal(0, 0.15, len(P)), 0, 1), rr)

    def shimmer(P, rr):
        """Small points of light green and of deep blue-violet among the leaves."""
        n = len(P)
        up = rr.random(n) < 0.55
        fam = np.where(up, rr.choice(["ygreen", "chart", "bluegreen", "greyblue"], n, p=[0.35, 0.15, 0.3, 0.2]),
                       rr.choice(["deep", "shade", "mauve", "viridian"], n, p=[0.3, 0.25, 0.25, 0.2]))
        return paints(fam, np.where(up, rr.uniform(0.45, 0.85, n), rr.uniform(0.15, 0.5, n)), rr)

    def jewels(P, rr):
        """Points of pure colour, small flowers seen far off, each colour thicker in places."""
        y, x = at(P)
        w = JEWEL_ODDS[:, None] * np.exp(1.5 * plenty[:, y // 16, x // 16])
        pick = np.minimum((rr.random(len(P)) > np.cumsum(w / w.sum(0), 0)).sum(0), len(JEWELS) - 1)
        return paints(JEWELS[pick], rr.uniform(0.35, 0.9, len(P)), rr)

    def go(name, odds, spacing, width, load, field, length=None, aspect=None, tilt=0.3, bend=0.0, n=5, flip=0.5, dry=0.0, **kw):
        """A passage of the ground: strokes strewn where `odds` lets them, about `spacing` apart, half-widths
        about width[0] (spread by width[1]), along `field`, each `length` px or `aspect` times its width long."""
        rr = fresh(name)
        P = strew(spacing, odds, rr)
        N = len(P)
        wd = (width[0] * np.exp(np.clip(rr.normal(0, width[1], N), -2, 2))).astype(np.float32)
        L = rr.uniform(*length, N) if length is not None else 2 * wd * rr.uniform(*aspect, N)
        paths = impasto.follow(field, P + rr.normal(0, 0.3, (N, 2)) * spacing, L, n, rr.normal(0, bend, N), rr.normal(0, tilt, N))
        back = rr.random(N) < flip
        paths[back] = paths[back, ::-1]
        cols, share = load(P, rr)
        d = rr.uniform(*dry, N) if isinstance(dry, tuple) else np.full(N, dry)
        impasto.lay(rgb, height, paths, wd, cols, rr, share=share, wet=wet, dry=d, **kw)

    def rubbed(P, rr):
        T = want[at(P)]
        n = len(P)
        cols = np.stack([T * np.exp(rr.normal(0, 0.03, (n, 3))), T * np.exp(rr.normal(0, 0.06, (n, 1)) * np.float32([1, 0.3, -0.8])),
                         T * np.exp(rr.normal(0, 0.05, (n, 3)))], 1)
        return np.clip(cols, 0, 1).astype(np.float32), np.stack([rr.uniform(0.5, 0.7, n), rr.uniform(0.2, 0.35, n),
                                                                  rr.uniform(0.05, 0.2, n)], 1)

    rise = (-np.pi / 2 + 0.35 * noise.field((H, W), 380, r)).astype(np.float32)
    every = np.ones((H, W), np.float32)

    # the first sitting: the whole ground in broad thin strokes, wet into one another
    go("lay-in", every, 70, (16, 0.25), rubbed, rise, length=(70, 170), tilt=0.5, thick=0.03, grooves=0.12, lips=0.02,
       land=0.1, lift=0.05, pickup=0.6, merge=6, spent=0.7, fray=3, taper=0.4, hide=0.75, ends=(0.5, 0.5))
    height -= 0.7 * (height - 0.2 * ground.tooth)
    wet[:] = 0

    # the second, over it dry: the matrix, small flat touches every way at every height, overlapping like
    # feathers; points of light green and blue-violet among them; and points of colour sown through it all
    flat = dict(thick=0.12, grooves=0.25, lips=0.05, land=0.3, lift=0.2, spent=0.5, pickup=0.25, merge=1.5, fray=0.9)
    leafy = dict(aspect=(1.0, 2.0), tilt=0.9, bend=0.004, ends=(0.55, 0.85), taper=0.55, dry=(0, 0.15), **flat)
    go("matrix", every, 17, (9, 0.3), matrix, rise, **leafy)
    go("shimmer", every * 0.4, 15, (4.5, 0.25), shimmer, rise, aspect=(0.7, 1.6), tilt=1.4, ends=(0.7, 0.8), taper=0.4,
       dry=(0, 0.3), **flat)
    go("points", (0.03 + 0.22 * sown) * (1 - 0.6 * zone), 24, (3.8, 0.45), jewels, rise, aspect=(0.6, 1.3), tilt=1.5, ends=(0.9, 0.9), taper=0.2,
       **{**flat, "thick": 0.22})
    wet[:] = 0

    # the tall plants, from the back to the front: each band its leaves and stems, then its flowers
    r = fresh("plants")
    items.sort(key=lambda e: e[0])

    def lay(st, **kw):
        if st.paths:
            cols, share = paints(st.fam, st.tone, r)
            impasto.lay(rgb, height, st.paths, np.array(st.w), cols, r, share=share, dry=np.array(st.dry), wet=wet, **kw)

    green = dict(thick=0.12, spent=0.55, pickup=0.25, merge=1.0, grooves=0.3, land=0.4, ends=(0.4, 0.7), taper=0.5, fray=1.0,
                 tails=0.4)
    bloom = dict(thick=0.34, spent=0.4, pickup=0.12, merge=0.6, grooves=0.25, land=0.6, lift=0.4, fray=0.8, tails=0.15)

    def flush(buf):
        g, b = Strokes(), Strokes()
        for _, gg, bb in buf:
            g.extend(gg)
            b.extend(bb)
        lay(g, **green)
        lay(b, ends=(0.5, 0.45), taper=0.3, **bloom)
    buf = []
    for d, alone, g, b in items:
        if buf and (alone or d - buf[0][0] > 100 or len(buf) >= 80):
            flush(buf)
            buf = []
        buf.append((d, g, b))
        if alone:
            flush(buf)
            buf = []
    flush(buf)

    # leaves up in front of the sunflower stalks; then the carpet, sown three times, the first sowing half
    # buried in more touches of green, the last the brightest
    lay(tufts(fresh("tufts")), **green)
    sunk, main, last = carpet(fresh("carpet"), taken, zone, stand)
    lay(sunk, ends=(0.7, 0.6), taper=0.25, **bloom)
    go("veil", zone * 0.4, 17, (8, 0.3), matrix, rise, **leafy)
    lay(main, ends=(0.7, 0.6), taper=0.25, **bloom)
    go("flecks", zone * 0.12, 20, (6, 0.3), matrix, rise, **leafy)
    lay(last, ends=(0.7, 0.6), taper=0.25, **bloom)

    height = ndimage.gaussian_filter(height, 0.7)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.5, gloss=0, reach=(0.8, 1.14))
    return img + 0.035 * impasto.glints(height, LIGHT)[..., None]
