"""The Window onto the Mimosa, Le Cannet. Oil on canvas, broad and thin in the room, small and dense in the garden.

From 1926 until his death in 1947 Bonnard lived at Le Bosquet, a small pink house on the hill above Le
Cannet, with a garden that fell away below it toward the roofs of the town and the bay of Cannes. He did
not paint in front of the motif. He looked, made a few pencil notes in his diary, and painted indoors
from memory, on lengths of unstretched canvas pinned to the wall, going back to each one for months and
years. The paint went on thin, over paint that had dried: broad scumbles dragged over the grain, short
hatchings, small dabs. He kept the colours apart instead of mixing them, a violet beside an orange, a
lemon beside a lilac, so that they vibrate against each other, and in his last years he pushed them
higher and higher. In The Studio with Mimosa the glazing bars of the big studio window cross a wall of
mimosa in flower, a blaze of acid yellow, while the room round it is pink and violet.

Here a window at Le Bosquet looks onto the garden in February: a mimosa in flower fills the panes on the
left and an almond stands against the sky on the right, over the hills across the bay, the roofs of the
town and the bushes of the garden. On a white cloth before the window is a bowl of oranges and lemons.
The canvas was stained thin, each part in one colour, in the room a little warmer than the paint that was
to go over it. The walls, the curtain, the floor and the cloth were laid in over the stain with a broad brush
carrying little paint, each stroke running out along its length onto the threads of the canvas, so that
the stain shows between them, and were then left alone. The garden was built up the other way, in small
dense dabs, one layer on another once the last was dry. The mimosa is a heap of rounded masses of blossom,
lemon where the sun strikes them and deep yellow, ochre and orange in their shade, with its grey-green
leaves in the hollows between them and a little sky showing through; its flowers were pressed on last in
small thick touches, crowded in the middle of each mass and fewer toward its edge, the palest where the
light falls. The bars of the window are strokes of pale violet, blue and orange that thicken and thin and
break off where the light of the garden eats into them; the rail along its foot was pulled across in three
long strokes laid end over end. Each fruit is a few broad strokes laid down it side by side, each bowed like
its outline, from vermilion or green ochre on the side away from the window to the light on the side
toward it, beside a shadow of cobalt on the cloth.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, spatial

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Window onto the Mimosa, Le Cannet"
DATE = "2026"
MEDIUM = "Oil on canvas: broad thin scumbles, hatchings and small dense touches of unmixed colour"
AFTER = ("Pierre Bonnard, the late interiors and windows at Le Bosquet, Le Cannet: The Studio with Mimosa, "
         "1939–46, and Almond Tree in Blossom, 1947")
ROOM = "The Garden"
YEAR = 1946
PLACE = "Le Cannet"
REGION = "Europe"
NOTE = ("A window in February, its bars crossing a mimosa in flower, with a bowl of oranges and lemons on "
        "the table before it. The room is laid in broad, thin strokes and left quiet, so that the small "
        "dense touches of the garden blaze against it.")

H, W = 2400, 2800
LIGHT = (-0.6, -0.5, 0.62)
LUMA = dabs.LUMA
EX, EY, D = 1500.0, 950.0, 2000.0                 # where the eye meets the wall square on, and how far back
X0, X1, Y0, Y1 = 760.0, 2520.0, -300.0, 1640.0    # the opening, on the inner face of the wall
T = 380.0                                         # the thickness of the wall
FLOOR = 2100.0                                    # where the wall meets the floor
TABLE_AT, TABLE_R = (950.0, 2330.0), (820.0, 290.0)   # the round table before the window, seen from above
WALL, CURTAIN, REV_L, REV_R, SILL, FLOORS, TABLE, VIEW = range(8)
SKY, HILLS, SEA, TOWN, GROUND, GREEN, ALMOND, MIMOSA, LEAVES = range(10, 19)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# each family from dark to light
ULTRA = pal("#1f2160", "#292c7a", "#363a91", "#4a4fa6", "#6268b8")
VIOLET = pal("#3b2768", "#4b3380", "#5e4396", "#7558aa", "#8f73bf", "#a991d0")
BLUEV = pal("#2c3d86", "#3a51a0", "#5069b4", "#6c86c6", "#93a8d8")
LILAC = pal("#8e74c2", "#a68fcf", "#b9a6db", "#cdbfe6", "#e0d6f0", "#efe9f6")
MAUVE = pal("#6a3170", "#7d3f7f", "#975293", "#b06ca8", "#c98bbd", "#dcaad0")
CRIMSON = pal("#6e1d3e", "#8a2649", "#a83355", "#c44a67")
ROSE = pal("#cf5f78", "#de7d90", "#e99aa8", "#f2b8bf", "#f8d4d4")
ORANGE = pal("#c94f22", "#d9612a", "#e57b34", "#ee9442", "#f4ae58", "#f7c579")
OCHRE = pal("#a06a24", "#b07a2a", "#c58f36", "#d6a548", "#e3bd66", "#eed490")
VERMILION = pal("#b8321e", "#c33b22", "#d24f2a", "#de6634", "#e8804a")
CREAM = pal("#e9dcc0", "#f2e8d2", "#f8f1e2", "#fcf9f0")
LEMON = pal("#d8bc12", "#e2c81a", "#ecd82c", "#f3e548", "#f7ec6e", "#faf2a0")
CADMIUM = pal("#d68e10", "#de9c12", "#e9b01c", "#f1c42a", "#f5d448")
PALE = pal("#f5edb0", "#f9f3cc", "#fcf9e4")
LIME = pal("#7fa82a", "#93b532", "#aac643", "#c2d65c", "#d7e47e")
SAGE = pal("#3f7362", "#4f8270", "#679882", "#81ad95", "#9cc2a8", "#bcd6c0")
EMERALD = pal("#0d5345", "#11604e", "#1a765e", "#278e70", "#3fa584", "#66bb9c")
DEEP = pal("#122c38", "#14343f", "#1a434c", "#22555a", "#2c6866")
TURQ = pal("#2f97a2", "#3aa9b0", "#5ebdc0", "#86cfcd", "#acdfda", "#cfeee8")
COBALT = pal("#25429a", "#2f50aa", "#4062b8", "#5a7cc7", "#7d9ad5", "#a5bae3", "#c8d5ef")
PINK = pal("#d68aa4", "#e3a2b8", "#edbccb", "#f4d4dd", "#f9e8ec")
WHITE = pal("#e4ddd0", "#ece6da", "#f4f0e6", "#fbf8f1")
ROOF = pal("#a8361f", "#bf4128", "#d35632", "#df703c", "#e98b50", "#f0a86c")
BARK = pal("#2e2140", "#3e2c4e", "#4f3a5c", "#62506c")
DEEPEST = lin("#1d1638")          # what a colour is let down with, below its own value
BRIGHT = lin("#fffcf2")           # and lifted with, above it

# What each part is painted with: the lightness of its paint where its light is low and where it is full
# (linear), and its colours, each with its weight where the light is full and where it is low. A touch
# keeps the lightness of its place whatever its colour, so that the colours change and the values hold.
# The room is each part in one family, mostly; the garden is many colours at one lightness.
PAINTS = {
    WALL: ((0.05, 0.3), [(VIOLET, 0.3, 0.42), (ULTRA, 0.0, 0.26), (BLUEV, 0.06, 0.16), (MAUVE, 0.24, 0.12),
                         (LILAC, 0.2, 0.04), (ROSE, 0.2, 0.0)]),
    CURTAIN: ((0.12, 0.5), [(ROSE, 0.42, 0.1), (ORANGE, 0.3, 0.06), (MAUVE, 0.08, 0.4), (VIOLET, 0.0, 0.34),
                            (OCHRE, 0.2, 0.1)]),
    REV_L: ((0.2, 0.62), [(ORANGE, 0.4, 0.24), (ROSE, 0.3, 0.3), (OCHRE, 0.2, 0.06), (MAUVE, 0.0, 0.3),
                          (CREAM, 0.1, 0.0)]),
    REV_R: ((0.16, 0.42), [(LILAC, 0.36, 0.3), (ROSE, 0.3, 0.1), (VIOLET, 0.04, 0.4), (TURQ, 0.1, 0.1),
                           (ORANGE, 0.2, 0.1)]),
    SILL: ((0.36, 0.7), [(CREAM, 0.36, 0.1), (PALE, 0.3, 0.04), (ROSE, 0.14, 0.3), (LILAC, 0.1, 0.4),
                         (TURQ, 0.1, 0.16)]),
    FLOORS: ((0.07, 0.42), [(ORANGE, 0.36, 0.04), (OCHRE, 0.3, 0.04), (ROSE, 0.2, 0.14), (VIOLET, 0.04, 0.46),
                            (MAUVE, 0.1, 0.24), (ULTRA, 0.0, 0.1)]),
    TABLE: ((0.14, 0.76), [(WHITE, 0.42, 0.04), (CREAM, 0.26, 0.04), (PALE, 0.2, 0.0), (LILAC, 0.08, 0.46),
                           (COBALT, 0.04, 0.32), (TURQ, 0.0, 0.14)]),
    SKY: ((0.24, 0.5), [(COBALT, 0.3, 0.5), (TURQ, 0.34, 0.14), (LILAC, 0.12, 0.14), (PINK, 0.1, 0.02),
                        (WHITE, 0.1, 0.02), (BLUEV, 0.04, 0.12)]),
    HILLS: ((0.14, 0.26), [(VIOLET, 0.3, 0.3), (COBALT, 0.3, 0.36), (LILAC, 0.16, 0.1), (TURQ, 0.12, 0.16),
                           (ROSE, 0.12, 0.08)]),
    SEA: ((0.25, 0.42), [(TURQ, 0.4, 0.3), (COBALT, 0.3, 0.4), (LILAC, 0.16, 0.2), (PINK, 0.14, 0.1)]),
    TOWN: ((0.42, 0.2), [(ROOF, 0.56, 0.0), (ORANGE, 0.22, 0.02), (CREAM, 0.0, 0.42), (LILAC, 0.04, 0.3),
                         (ROSE, 0.18, 0.14), (TURQ, 0.0, 0.12)]),
    GROUND: ((0.14, 0.5), [(LIME, 0.3, 0.08), (OCHRE, 0.2, 0.04), (LEMON, 0.16, 0.0), (ORANGE, 0.12, 0.06),
                           (ROSE, 0.08, 0.08), (VIOLET, 0.02, 0.32), (EMERALD, 0.08, 0.3), (VERMILION, 0.04, 0.04),
                           (COBALT, 0.0, 0.08)]),
    GREEN: ((0.06, 0.26), [(EMERALD, 0.32, 0.26), (DEEP, 0.04, 0.3), (SAGE, 0.24, 0.06), (COBALT, 0.04, 0.16),
                           (LIME, 0.22, 0.02), (VIOLET, 0.0, 0.14), (TURQ, 0.1, 0.04)]),
    ALMOND: ((0.36, 0.82), [(WHITE, 0.4, 0.06), (PINK, 0.26, 0.16), (LILAC, 0.1, 0.34), (COBALT, 0.02, 0.28),
                            (CREAM, 0.16, 0.02), (ROSE, 0.04, 0.12)]),
    MIMOSA: ((0.36, 0.78), [(LEMON, 0.7, 0.06), (CADMIUM, 0.1, 0.44), (PALE, 0.14, 0.0), (OCHRE, 0.0, 0.24),
                            (ORANGE, 0.0, 0.24), (LIME, 0.06, 0.02)]),
    LEAVES: ((0.11, 0.3), [(SAGE, 0.6, 0.46), (LIME, 0.12, 0.04), (LILAC, 0.1, 0.16), (TURQ, 0.06, 0.06),
                           (COBALT, 0.0, 0.1), (EMERALD, 0.06, 0.12), (OCHRE, 0.06, 0.06)]),
}
# the flowers of the mimosa, pressed on over its masses; and the last of them, where the sun strikes
SPRAY = ((0.36, 0.84), [(LEMON, 0.6, 0.16), (PALE, 0.2, 0.0), (CADMIUM, 0.15, 0.4), (OCHRE, 0.0, 0.2),
                        (ORANGE, 0.05, 0.24)])
SUNLIT = ((0.84, 0.95), [(PALE, 0.5, 0.4), (LEMON, 0.4, 0.5), (CREAM, 0.1, 0.1)])
# the skins of the fruit, each a brush of three colours (family, lightness) close to one another: from the side
# away from the window, through the half-tone and the body, to the side toward it
ORANGE_SKIN = (((VERMILION, 0.15), (ORANGE, 0.18), (VERMILION, 0.12)),
               ((ORANGE, 0.23), (VERMILION, 0.2), (ORANGE, 0.26)),
               ((ORANGE, 0.3), (ORANGE, 0.34), (CADMIUM, 0.36)),
               ((ORANGE, 0.4), (ORANGE, 0.44), (CADMIUM, 0.46)))
LEMON_SKIN = (((OCHRE, 0.32), (LIME, 0.34), (CADMIUM, 0.36)),
              ((CADMIUM, 0.46), (LEMON, 0.5), (OCHRE, 0.42)),
              ((LEMON, 0.6), (LEMON, 0.56), (CADMIUM, 0.54)),
              ((LEMON, 0.74), (LEMON, 0.78), (PALE, 0.8)))
# what each part of the canvas is stained with first, thin, before anything goes over it: in the room a
# colour a little warmer than the one that goes over it
STAIN = {WALL: MAUVE, CURTAIN: ORANGE, REV_L: ORANGE, REV_R: ROSE, SILL: PALE, FLOORS: ORANGE, TABLE: CREAM,
         SKY: TURQ, HILLS: VIOLET, SEA: TURQ, TOWN: ROOF, GROUND: OCHRE, GREEN: EMERALD, ALMOND: PINK,
         MIMOSA: CADMIUM, LEAVES: SAGE}

# the masses outside, each a cluster of round lumps (x, y, radius)
MIMOSA_CROWN = [(1010, 170, 190), (1240, 110, 160), (1420, 230, 165), (960, 450, 220), (1210, 430, 235),
                (1450, 520, 180), (1020, 740, 225), (1290, 770, 235), (1540, 790, 170), (1110, 1010, 190),
                (1390, 1020, 175), (920, 990, 150), (1620, 960, 110)]
ALMOND_CROWN = [(2010, 330, 125), (2190, 270, 140), (2090, 500, 160), (2270, 480, 125), (1960, 620, 105),
                (2180, 690, 130), (2320, 700, 95)]
BUSHES = [(940, 1250, 150), (1250, 1290, 140), (1580, 1230, 150), (1850, 1180, 110), (2120, 1140, 140),
          (2320, 1210, 140)]
VBARS = (0.27, 0.515, 0.76)         # the vertical glazing bars, across the panes
HBARS = (420.0, 950.0)              # and the horizontal ones
STILE, BAR = 21.0, 13.0             # half-widths of the frame and of a bar

# the still life. The bowl: the centre and half-axes of its rim, and how far its body shows below the rim;
# the fruit in it from the back of the heap to the front (x, y, radius, lemon?), and two on the cloth
BOWL, DEPTH = (1150.0, 2196.0, 265.0, 64.0), 74.0
FRUIT = [(1018, 2132, 64, False), (1158, 2104, 62, True), (1296, 2134, 62, False), (1084, 2182, 70, False),
         (1232, 2176, 68, False), (950, 2196, 56, True), (1358, 2196, 56, False)]
LOOSE = [(640, 2268, 64, True), (1600, 2244, 62, False)]
SUN = -0.85                         # the angle on the canvas from which the window's light falls on the table


def at(P):
    return np.clip(P[..., 1].astype(int), 0, H - 1), np.clip(P[..., 0].astype(int), 0, W - 1)


def see(x, y, z):
    """Where a point of the room falls on the canvas: x, y on the wall's inner face, z out through it."""
    x, y, z = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, y, z)))
    f = D / (D + z)
    return np.stack([EX + (x - EX) * f, EY + (y - EY) * f], -1)


INNER = see([X0, X1, X1, X0], [Y0, Y0, Y1, Y1], 0)    # the opening's corners on the inner face
OUTER = see([X0, X1, X1, X0], [Y0, Y0, Y1, Y1], T)    # and on the outer face, where the window stands


def plan(r):
    """The room, drawn freehand: which part each pixel belongs to. -> labels (H,W)"""
    im = Image.new("L", (W, H), WALL)
    d = ImageDraw.Draw(im)

    def put(P, lab, a=3.0):
        P = np.asarray(P, float)
        d.polygon([tuple(p) for p in P + r.normal(0, a, P.shape)], fill=int(lab))

    i, o = INNER, OUTER
    put([[-20, FLOOR], [W + 20, FLOOR - 12], [W + 20, H + 20], [-20, H + 20]], FLOORS, 2)
    put([i[0], o[0], o[3], i[3]], REV_L)
    put([o[1], i[1], i[2], o[2]], REV_R)
    put([o[3], o[2], i[2], i[3]], SILL)
    put(o, VIEW, 1)
    y = np.linspace(-20, FLOOR + 40, 40)
    left = 470 + 8 * noise.line1d(40, 6, r)
    right = 800 + 18 * noise.line1d(40, 5, r) + 30 * (y / FLOOR) ** 3
    put(np.vstack([np.stack([left, y], 1), np.stack([right, y], 1)[::-1]]), CURTAIN, 1)
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    rim = np.stack([TABLE_AT[0] + TABLE_R[0] * np.cos(t), TABLE_AT[1] + TABLE_R[1] * np.sin(t)], 1)
    put(hand(np.vstack([rim, rim[:1]]), 4, 6, r)[:-1], TABLE, 1)
    lab = np.asarray(im).astype(np.float32)
    dx = 3 * noise.field((H, W), 40, r) + 6 * noise.field((H, W), 300, r)
    dy = 3 * noise.field((H, W), 40, r) + 6 * noise.field((H, W), 300, r)
    return noise.warp(lab, dx, dy, order=0).astype(np.int16)


def lumps(spec, join=np.add, S=4):
    """A cluster of round lumps as a smooth field, 0.5 at the edge of each, worked out at 1/S size: added up,
    so that lumps near each other run into one, or (join=np.maximum) each kept round on its own."""
    yy, xx = (np.mgrid[0:H // S, 0:W // S] * S).astype(np.float32)
    B = np.zeros(yy.shape, np.float32)
    for x, y, rad in spec:
        B = join(B, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (rad / 1.177) ** 2)))
    return ndimage.zoom(B, S, order=1)[:H, :W]


def lit(B, mask, sun=(0.55, -0.7, 0.45), k=260.0):
    """How much sun a mass of lumps catches: its lumps turned toward the sun lit, the undersides dark."""
    G = ndimage.gaussian_filter(B, 6)
    gy, gx = np.gradient(G)
    n = np.stack([-gx * k, -gy * k, np.ones_like(G)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lam = n @ (np.asarray(sun, np.float32) / np.linalg.norm(sun))
    lo, hi = np.percentile(lam[mask], [5, 95])
    return noise.smoothstep(lo, hi, lam)


def scene(r):
    """The garden through the window: the sky, the hills across the bay and the sea at the height of the eye,
    the roofs of the town below, the bushes and the ground of the garden, the almond and the mimosa, each in
    front of the last. The mimosa is a heap of rounded masses of blossom, each lit on the side toward the sun,
    with its grey-green foliage in the hollows between them and, at the edge of the crown and here and there
    inside it, what lies behind. -> part (H,W), light (H,W) in 0..1, the masses (H,W), 0.5 at the edge of each"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx[0]
    hill = 862 + 25 * noise.line1d(W, 260, r) + 40 * noise.smoothstep(1700, 2400, x)
    shore = 955 + 4 * noise.line1d(W, 200, r)
    ridge = 990 + 18 * np.abs(np.sin(x / 55 + 3 * noise.line1d(W, 400, r))) + 6 * noise.line1d(W, 30, r)
    soil = 1340 + 30 * noise.line1d(W, 400, r)
    part = np.full((H, W), SKY, np.int16)
    light = np.clip(0.2 + 0.8 * (yy / 870) ** 1.5, 0, 1) + 0.1 * noise.field((H, W), 200, r)
    for edge, lab, u in ((hill, HILLS, 0.5 + 0.3 * noise.field((H, W), 150, r)),
                         (shore, SEA, 0.5 + 0.3 * noise.field((H, W), 100, r)),
                         (ridge, TOWN, (yy < ridge + 40 + 12 * noise.field((H, W), 25, r)).astype(np.float32)),
                         (soil, GROUND, 0.5 + 0 * yy)):
        m = yy > edge
        part[m], light[m] = lab, u[m]
    sun = noise.smoothstep(-0.6, 0.8, noise.field((H, W), 120, r))
    rough = 0.16 * noise.field((H, W), 50, r) + 0.09 * noise.field((H, W), 16, r)
    for spec, lab, edge in ((BUSHES, GREEN, 0.5), (ALMOND_CROWN, ALMOND, 0.52)):
        B = lumps(spec)
        m = B + rough > edge
        if lab == GREEN:
            g = (part == GROUND) & ~m
            light[g] = (sun * noise.smoothstep(0, 0.5, 0.5 - B))[g]
        part[m] = lab
        light[m] = np.clip(lit(B, m) + 0.12 * noise.field((H, W), 60, r), 0, 1)[m]
    B = lumps(MIMOSA_CROWN)
    crown = B + rough > 0.48
    C = dabs.scatter((H, W), 250, r)                  # the big masses, and smaller ones in the hollows between
    C = C[crown[at(C)]]
    big = lumps([(x_, y_, s) for (x_, y_), s in zip(C, r.uniform(90, 170, len(C)))], np.maximum)
    c = dabs.scatter((H, W), 100, r)
    c = c[crown[at(c)] & (big[at(c)] < 0.45) & (r.random(len(c)) < 0.6)]
    mass = np.maximum(big, lumps([(x_, y_, s) for (x_, y_), s in zip(c, r.uniform(40, 75, len(c)))], np.maximum))
    bloom = crown & (mass + 0.15 * noise.field((H, W), 24, r) + 0.08 * noise.field((H, W), 9, r) > 0.5)
    rim = ndimage.distance_transform_edt(crown) < 40 + 25 * noise.field((H, W), 80, r)
    leaves = crown & ~bloom & ~rim & (noise.field((H, W), 70, r) < 1.2)
    glow = np.clip(lit(B, crown) - 0.3 * noise.smoothstep(150, 1300, yy) + 0.12 * noise.field((H, W), 60, r), 0, 1)
    part[bloom], part[leaves] = MIMOSA, LEAVES
    light[bloom] = np.clip(0.55 * lit(mass, bloom) + 0.45 * glow + 0.08 * noise.field((H, W), 40, r), 0, 1)[bloom]
    light[leaves] = (0.15 + 0.6 * glow)[leaves]
    return part, np.clip(light, 0, 1).astype(np.float32), mass


def patches(r, size, S=4):
    """The canvas cut into patches about `size` px across, with wandering edges, each to be hatched its own
    way. -> patch index (H,W), number of patches"""
    P = dabs.scatter((H // S, W // S), size / S, r)
    yy, xx = np.mgrid[0:H // S, 0:W // S]
    idx = spatial.cKDTree(P).query(np.stack([xx.ravel(), yy.ravel()], 1))[1].reshape(yy.shape)
    idx = ndimage.zoom(idx, S, order=0)[:H, :W].astype(np.float32)
    dx = 0.25 * size * noise.field((H, W), size * 0.6, r)
    dy = 0.25 * size * noise.field((H, W), size * 0.6, r)
    return noise.warp(idx, dx, dy, order=0).astype(np.int32), len(P)


def ladder(fam, n=40):
    """A family as a ladder of tints by lightness: its own ramp, let down below with deep violet and lifted
    above with white. -> colours (M,3), their lightness (M,), rising"""
    t = np.linspace(0, 1, n)[:, None] * (len(fam) - 1)
    i = np.minimum(t.astype(int)[:, 0], len(fam) - 2)
    f = t - i[:, None]
    mid = fam[i] * (1 - f) + fam[i + 1] * f
    s = np.linspace(0, 1, 16)[:, None]
    C = np.vstack([(fam[0] * (1 - s) + DEEPEST * s)[::-1][:-1], mid, (fam[-1] * (1 - s) + BRIGHT * s)[1:]])
    return C, np.maximum.accumulate(C @ LUMA) + np.arange(len(C)) * 1e-7


def tint(fam, y):
    """The tint of a family at lightness y (one value or an array of them)."""
    C, Y = ladder(fam)
    y = np.atleast_1d(np.asarray(y, np.float64))
    return np.stack([np.interp(y, Y, C[:, c]) for c in range(3)], 1).astype(np.float32)


def brushes(spec, u, clump, shift, r, patch=0.35, spread=0.12):
    """A brush for each touch: its colour picked by the light where it goes (each colour gathering in
    places), at the lightness of its place give or take a little, and two more colours streaked into it,
    one of them often another colour of the same part at the same lightness."""
    (lo, hi), paints = spec
    A = np.array([p[1:] for p in paints], np.float32)
    N, uu = len(u), u[:, None]
    wts = A[None, :, 1] + (A[None, :, 0] - A[None, :, 1]) * uu
    c = np.cumsum(wts, 1)
    c /= c[:, -1:]
    which = (np.clip(r.random(N) * (1 - patch) + patch * clump, 0, 0.999)[:, None] > c).sum(1)
    other = (r.random(N)[:, None] > c).sum(1)
    y = (lo + (hi - lo) * u) * np.exp(r.normal(0, spread, N) + shift)
    near = r.random(N) < 0.5
    cols = np.empty((N, 3, 3), np.float32)
    for k, p in enumerate(paints):
        s = which == k
        if s.any():
            cols[s, 0] = tint(p[0], y[s])
            cols[s, 1] = tint(p[0], y[s] * np.exp(r.choice([-1, 1], s.sum()) * r.uniform(0.12, 0.3, s.sum())))
        s = other == k
        if s.any():
            cols[s, 2] = tint(p[0], y[s] * np.exp(r.normal(0, 0.08, s.sum())))
    cols[near, 1] = cols[near, 2]
    share = np.stack([r.uniform(0.5, 0.75, N), r.uniform(0.15, 0.35, N), r.uniform(0.04, 0.15, N)], 1)
    return cols, share


def trim(paths, mask):
    """Cut each stroke where it first leaves the mask."""
    n = paths.shape[1]
    ins = mask[at(paths)]
    k = np.where(ins.all(1), n, ins.argmin(1))
    t = np.linspace(0, 1, n)[None, :] * np.maximum(k - 1, 0)[:, None]
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    row = np.arange(len(paths))[:, None]
    return paths[row, i] * (1 - f) + paths[row, np.minimum(i + 1, n - 1)] * f


def pieces(path, length, overlap, r, keep=1.0):
    """Cut a long line into the strokes a brush would lay it in, each starting a little back over the last
    (or, with a negative overlap, a little after it), lifting off now and then (keep < 1). -> index arrays"""
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out, a, b = [], 0.0, 0.0
    while b < s[-1]:
        b = min(s[-1], a + length * r.uniform(0.7, 1.3))
        m = np.nonzero((s >= a) & (s <= b))[0]
        if len(m) >= 3 and r.random() < keep:
            out.append(m)
        a = b - overlap * r.uniform(0.5, 1.5)
    return out


def hand(line, amp, scale, r):
    """A line as a hand draws it, wandering off its course and back."""
    d = np.gradient(line, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return line + np.stack([-d[:, 1], d[:, 0]], 1) * (amp * noise.line1d(len(line), scale, r))[:, None]


def segment(p, q, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(p, float) * (1 - t) + np.asarray(q, float) * t


def arc(x, y, a, b, t0, t1, n=12):
    """Points along an ellipse from angle t0 to t1."""
    t = np.linspace(t0, t1, n)
    return np.stack([x + a * np.cos(t), y + b * np.sin(t)], 1)


def scumble(rgb, height, tooth, B, p, q, w, cols, r, load=1.0, spent=0.8, hide=0.8, bow=0.0):
    """A broad stroke of thin paint dragged over the canvas from p to q, in place. Where the brush is full it
    covers; as it runs out along the stroke it leaves paint only on the threads of the weave that stand
    highest, so the colour beneath shows between them in a fine broken grain. Its edges waver, its three
    colours lie in long streaks side by side, and the paint adds almost nothing to the relief.
    B: a field of bristle streaks (rows along a stroke), from impasto.bristles."""
    p, q = np.asarray(p, np.float32), np.asarray(q, np.float32)
    L = float(np.hypot(*(q - p)))
    if L < 2:
        return
    d = (q - p) / L
    nrm = np.array([-d[1], d[0]], np.float32)
    pad = w + abs(bow) + 6
    x0, x1 = int(max(0, min(p[0], q[0]) - pad)), int(min(rgb.shape[1], max(p[0], q[0]) + pad + 1))
    y0, y1 = int(max(0, min(p[1], q[1]) - pad)), int(min(rgb.shape[0], max(p[1], q[1]) + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    s = (xx - p[0]) * d[0] + (yy - p[1]) * d[1]
    t = np.clip(s / L, 0, 1)
    v = (xx - p[0]) * nrm[0] + (yy - p[1]) * nrm[1] - bow * 4 * t * (1 - t)
    o = r.uniform(0, 1, 8) * np.array([B.shape[0], B.shape[1]] * 4)
    look = lambda a, b: ndimage.map_coordinates(B, [a, b], order=1, mode="grid-wrap")
    side = np.sign(v) * 60
    edge = look(s * 0.25 + o[0], side + o[1]) + 0.7 * look(s * 1.5 + o[6], side + o[7])   # each edge wavers
    wl = w * (1 + 0.07 * edge) * (1 - 0.3 * noise.smoothstep(0.75, 1.0, t))
    foot = (noise.smoothstep(wl + 1.5, wl - 1.5, np.abs(v)) * noise.smoothstep(-2, 6, s - 0.35 * w * (v / w) ** 2)
            * noise.smoothstep(L + 2, L - 0.6 * w, s - 0.4 * w * (v / w) ** 2))
    if not foot.any():
        return
    streak = look(s * 0.4 + o[2], v * 0.5 + o[3])                     # the brush's bristles, softly
    left = load * (1 - spent * t ** 1.4) * (1 - 0.55 * np.clip(np.abs(v) / wl, 0, 1) ** 3)   # thinner at the sides
    a = foot * noise.smoothstep(-0.1, 0.1, 1.3 * left - 1 + tooth[y0:y1, x0:x1] + 0.2 * streak)
    f = 0.5 + 0.5 * np.clip(0.8 * look(s * 0.06 + o[4], v * 0.3 + o[5]) + 0.3 * v / w, -1, 1)
    logc = np.log(np.clip(cols, 1e-4, 1))
    lc = logc[0] + (logc[1] - logc[0]) * noise.smoothstep(0.45, 0.55, f)[..., None]
    lc += (logc[2] - lc) * noise.smoothstep(0.86, 0.92, f)[..., None]
    c = np.exp(lc + 0.05 * streak[..., None])
    box = rgb[y0:y1, x0:x1]
    box += (c - box) * (a * hide)[..., None]
    height[y0:y1, x0:x1] += a * left * (0.35 + 0.15 * streak)


def fruit(x, y, R, lemon, r):
    """An orange or a lemon in a few broad strokes that turn with its form: its own colour laid across it, then
    four strokes down it side by side, each bowed like the outline on its side, from the deep colour on the
    side away from the window through the body to the light on the side toward it, all cut to its outline;
    then over the far edge a broken line of violet. -> outline (unit disc -> canvas matrix), [(paths,
    half-widths, colours, shares, inside?)] in the order they go on"""
    a = SUN + np.pi / 2 + r.normal(0, 0.3) if lemon else r.uniform(0, np.pi)
    ax, ay = (1.25 * R, 0.82 * R) if lemon else (R, 0.95 * R)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = rot @ np.diag([ax, ay])                     # the unit disc onto the fruit
    near = np.linalg.solve(M, [np.cos(SUN), np.sin(SUN)])
    tl = np.arctan2(near[1], near[0]) + r.normal(0, 0.15)     # the side turned to the window, on the disc
    ts = tl + np.pi                                          # and the side away from it
    deep, half, body, lit_ = LEMON_SKIN if lemon else ORANGE_SKIN
    c, s = np.array([x, y], float), np.sqrt(ax * ay)

    def put(P, w, spec, inside=True, share=(0.7, 0.2, 0.1)):
        cols = np.concatenate([tint(f, v * r.uniform(0.95, 1.05)) for f, v in spec])
        return [c + P @ M.T], np.array([w * s]), cols[None], np.array([share]), inside

    toward = np.array([np.cos(tl), np.sin(tl)])
    across = np.array([-toward[1], toward[0]])
    ys = np.linspace(-0.85, 0.85, 9) * r.choice([-1, 1])

    def lane(k):            # a stroke down the fruit, bowed like its outline on that side
        return (k * np.sqrt(1 - ys ** 2) + r.normal(0, 0.03))[:, None] * toward + ys[:, None] * across

    out = [put(np.linspace(-1.1, 1.1, 5)[:, None] * across, 1.1, body),
           put(lane(-0.72), 0.32, deep), put(lane(-0.28), 0.3, half), put(lane(0.18), 0.3, body),
           put(lane(0.6), 0.28, lit_)]
    t0 = ts + r.uniform(-0.6, 0.2)
    out.append(put(arc(0, 0, 0.99, 0.99, t0 - 0.7, t0 + r.uniform(0.3, 0.9), 7), 0.03,
                   ((VIOLET, 0.1), (COBALT, 0.13), (ULTRA, 0.09)), False))
    return M, out


THIN = dict(thick=0.16, grooves=0.32, lips=0.06, land=0.3, lift=0.18, ends=(0.55, 0.35), taper=0.3, fray=1.6,
            flatten=0.4, spent=0.45, pickup=0.1, merge=1.5, tails=0.3, hide=0.95)


def paint(seed=1946):
    r = noise.rng(seed)
    ground = canvas.duck((H, W), seed, tint="#efe8d8", thread=3.4)
    rgb, height = ground.color.copy(), ground.tooth * 0.3
    wet = np.zeros((H, W), np.float32)
    key = plan(r)
    part, light, mass = scene(r)
    region = np.where(key == VIEW, part, key).astype(np.int16)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = lambda scale: noise.field((H, W), scale, r)
    tx, ty = (xx - TABLE_AT[0]) / TABLE_R[0], (yy - TABLE_AT[1]) / TABLE_R[1]

    # the light in the room, which all comes from the window: strongest round the opening, falling away
    window = np.isin(key, (VIEW, REV_L, REV_R, SILL))
    glow = np.exp(-ndimage.distance_transform_edt(~window).astype(np.float32) / 240)
    U = np.where(key == VIEW, light, 0).astype(np.float32)
    for lab, u in ((WALL, 0.05 + 0.9 * glow - 0.4 * noise.smoothstep(1630, 1720, yy) + 0.12 * noise.smoothstep(1850, 2100, yy)
                    + 0.08 * f(260)),
                   (CURTAIN, 0.02 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * (xx - 470) / 125 + 1.5 * f(300)))
                    + 0.5 * (xx - 470) / 340 + 0.06 * f(100)),
                   (REV_L, 0.3 + 0.6 * noise.smoothstep(150, 1400, yy) + 0.08 * f(120)),
                   (REV_R, 0.35 + 0.35 * noise.smoothstep(200, 1500, yy) + 0.1 * f(120)),
                   (SILL, 0.65 + 0.15 * f(80)),
                   (FLOORS, 0.1 + 0.8 * np.exp(-((xx - 2000) / 800) ** 2 - ((yy - 2250) / 300) ** 2) + 0.08 * f(150)),
                   (TABLE, 0.5 - 0.75 * (ty + 0.3) + 0.25 * tx + 0.08 * f(140))):
        U[key == lab] = np.clip(u, 0, 1)[key == lab]

    # which way the touches go. In the garden each part is hatched its own way, and each patch of it turned
    # its own way again; in the room the broad strokes sweep calmly along each plane
    pid, n = patches(r, 110)
    turn, lift, clump = r.normal(0, 1, n), r.normal(0, 1, n), r.random(n)
    up = np.pi / 2
    ANG = np.zeros((H, W), np.float32)
    for labs, a, amp in (((SKY,), 0.1 + 0.2 * f(300), 0.4),
                         ((HILLS, SEA, TOWN, GROUND), -0.03 + 0.1 * f(200), 0.3),
                         ((GREEN,), -up + 0.3 * f(100), 0.5),
                         ((ALMOND,), 0.8 * f(80), 1.2),
                         ((MIMOSA, LEAVES), 2.1 + 0.3 * f(300), 0.6)):
        m = np.isin(region, labs)
        ANG[m] = (np.broadcast_to(a, (H, W)) + amp * turn[pid])[m]
    SWEEP = np.zeros((H, W), np.float32)
    below = (yy > Y1 - 30) & (xx > INNER[3][0] - 60) & (xx < INNER[2][0] + 60)     # the wall under the window
    for labs, a in (((WALL,), np.where(below, 0.04 + 0.1 * f(500), up + 0.1 * f(500))),
                    ((CURTAIN, REV_L, REV_R), up + 0.1 * f(500)),
                    ((SILL,), 0.03 * f(300)),
                    ((FLOORS,), np.arctan2(EY - yy, EX - xx) + 0.06 * f(400)),
                    ((TABLE,), 0.04 + 0.08 * f(300))):
        m = np.isin(region, labs)
        SWEEP[m] = np.broadcast_to(a, (H, W))[m]
    # where the touches of each part may run: each plane of the room, and the blossom of the mimosa and the
    # leaves between it, keep each to itself; the rest of the garden runs together within the window
    bounds = {lab: ndimage.binary_dilation(key == lab, iterations=3) for lab in range(VIEW)}
    bounds.update({lab: ndimage.binary_dilation(region == lab, iterations=6) for lab in (MIMOSA, LEAVES)})
    inside = ndimage.binary_dilation(key == VIEW, iterations=2)

    def strew(mask, sp):
        P = dabs.scatter((H, W), sp, r)
        return P[mask[at(P)]]

    def spacing(L, w, cover):
        return float(np.sqrt(L * 2 * w / (cover * 0.866)))

    def where(*labs):
        return np.isin(region, labs)

    def lay(paths, w, cols, share=None, **kw):
        impasto.lay(rgb, height, paths, w, np.asarray(cols, np.float32), r, share=share, wet=wet, **{**THIN, **kw})

    def touches(P, L, w, n=6, bend=0.006, tilt=0.15, patch=0.45, spread=0.12, jitter=10.0, value=0.08, **kw):
        """Touches from seeds P: each takes the part, the light and the way of the place it falls (looked up a
        little off its seed, so that the parts break into each other at their edges) and keeps to that part.
        A (lo, hi) pair among the keywords is drawn afresh for each touch."""
        if not len(P):
            return
        k = len(P)
        lab = region[at(P + r.normal(0, jitter, P.shape))]
        L = L * np.exp(r.normal(0, 0.3, k))
        paths = impasto.follow(ANG, P, L, n, r.normal(0, bend, k), r.normal(0, tilt, k))
        cols, share = np.empty((k, 3, 3), np.float32), np.empty((k, 3))
        q = pid[at(P)]
        for g in np.unique(lab):
            s = lab == g
            cols[s], share[s] = brushes(PAINTS[int(g)], U[at(P[s])], clump[q[s]], value * lift[q[s]], r, patch, spread)
            paths[s] = trim(paths[s], bounds.get(int(g), inside))
        back = r.random(k) < 0.5
        paths[back] = paths[back, ::-1]
        order = r.permutation(k)
        w = np.broadcast_to(np.asarray(w, np.float64), (k,)) * np.exp(r.normal(0, 0.18, k))
        kw = {k_: (r.uniform(*v, k) if isinstance(v, tuple) and k_ != "ends" else v) for k_, v in kw.items()}
        kw = {k_: (v[order] if isinstance(v, np.ndarray) else v) for k_, v in kw.items()}
        impasto.lay(rgb, height, paths[order], w[order], cols[order], r, share=share[order], wet=wet, **{**THIN, **kw})

    def cover(labs, L, w, c, **kw):
        P = strew(where(*labs), spacing(L, np.mean(w), c))
        touches(P, L, r.uniform(*w, len(P)) if isinstance(w, tuple) else w, **kw)

    B = impasto.bristles(r)

    def sweep(labs, L, w, c, load=(0.85, 1.15), spent=(0.5, 0.95), hide=0.8, patch=0.55, spread=0.05, tilt=0.08):
        """Broad thin strokes over the planes of the room, each along its plane's way and stopping where the
        plane does, laid in no order."""
        P = strew(where(*labs), spacing(L, np.mean(w), c))
        k = len(P)
        ws, Ls = r.uniform(*w, k), L * np.exp(r.normal(0, 0.25, k))
        ang = SWEEP[at(P)] + r.normal(0, tilt, k)
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        lab = region[at(P)]
        cols = np.empty((k, 3, 3), np.float32)
        q = pid[at(P)]
        for g in np.unique(lab):
            s = lab == g
            cols[s] = brushes(PAINTS[int(g)], U[at(P[s])], clump[q[s]], 0.06 * lift[q[s]], r, patch, spread)[0]
        ts = np.linspace(-0.5, 0.5, 25)
        for i in r.permutation(k):
            pts = P[i] + (ts * Ls[i])[:, None] * d[i]
            ins = bounds[int(lab[i])][at(pts)]
            lo = 12 - (np.argmin(ins[12::-1]) if not ins[12::-1].all() else 13) + 1
            hi = 12 + (np.argmin(ins[12:]) if not ins[12:].all() else 13) - 1
            p, q_ = (pts[lo], pts[hi]) if r.random() < 0.5 else (pts[hi], pts[lo])
            scumble(rgb, height, ground.tooth, B, p, q_, ws[i], cols[i], r, load=r.uniform(*load),
                    spent=r.uniform(*spent), hide=hide, bow=r.normal(0, 0.03 * Ls[i]))

    # the canvas stained first with thin colour, each part in one colour, in the room a little warmer than
    # what goes over it, wiped off the tops of the weave
    stain = np.empty((H, W, 3), np.float32)
    for k, fam in STAIN.items():
        m = region == k
        (lo, hi), _ = PAINTS[k]
        stain[m] = tint(fam, lo + (hi - lo) * U[m])
    a = np.clip(np.where(key < VIEW, 0.92, 0.8) - 0.2 * ground.tooth + 0.06 * f(60), 0, 1)[..., None]
    rgb[:] = rgb * (1 - a) + stain * a

    # the room laid in thin with a broad brush, each plane swept its own way, each stroke running out along
    # its length onto the threads of the canvas, so that the stain shows between them; then, here and there,
    # a second scumble in other colours of the same plane
    sweep((WALL,), 420, (26, 50), 2.4, hide=0.62, patch=0.85, spread=0.035)
    sweep((FLOORS,), 360, (20, 40), 2.8, hide=0.62, patch=0.8, spread=0.045)
    sweep((CURTAIN,), 600, (14, 28), 1.8, hide=0.8, patch=0.6)
    sweep((TABLE,), 280, (16, 30), 2.0, hide=0.85, patch=0.7)
    sweep((WALL, FLOORS), 240, (16, 32), 0.3, load=(0.5, 0.8), hide=0.7, patch=0.15, spread=0.08)
    sweep((TABLE,), 200, (12, 24), 0.45, load=(0.7, 1.0), hide=0.85, patch=0.2, spread=0.1)

    # the garden, the other way: first touches of colour patch by patch, each part with its own size of touch
    cover((SKY,), 70, 16, 1.8, jitter=10, dry=(0.0, 0.1), patch=0.5, spread=0.08)
    for labs, L, w, c in (((HILLS, SEA), 66, 9.5, 1.3), ((TOWN,), 26, 7.5, 1.4), ((GROUND, GREEN), 34, 8.5, 1.4),
                          ((ALMOND,), 30, 10, 1.4), ((LEAVES,), 34, 6.5, 1.4), ((MIMOSA,), 36, 11, 1.5)):
        cover(labs, L, w, c, jitter=10, dry=(0.0, 0.2))
    wet[:] = 0
    # then, over the dry paint, small dense dabs where the light vibrates, and fewer where it rests
    dab = dict(n=3, jitter=6, ends=(0.85, 0.85), taper=0.08, thick=0.26, spread=0.16, patch=0.25, value=0.05,
               tails=0.1, dry=(0.0, 0.25))
    for labs, L, w, c in (((MIMOSA,), 22, 8.0, 0.7), ((LEAVES,), 18, 3.5, 0.9), ((GROUND, GREEN), 13, 5.0, 1.3),
                          ((TOWN,), 12, 5.0, 0.7), ((HILLS, SEA), 24, 5.0, 0.5), ((SKY,), 30, 7.0, 0.3)):
        cover(labs, L, w, c, **dab)
    cover((ALMOND,), 16, 10.5, 1.2, **{**dab, "patch": 0.55, "spread": 0.1})      # the almond in fat blossoms
    wet[:] = 0

    # the reveals and the sill, in long strokes over the edges of the garden
    sweep((REV_L, REV_R), 500, (12, 22), 1.7, hide=0.88)
    sweep((SILL,), 280, (10, 20), 1.7, hide=0.9)
    wet[:] = 0

    # the flowers of the mimosa: sprays of small round touches pressed on thick, many in the middle of each
    # mass and fewer toward its edge, where the leaves show between; deep yellow, ochre and orange in the
    # shade, lemon and cream in the light; and last the palest of them, where the sun strikes
    full = noise.smoothstep(0.45, 0.95, mass) * noise.smoothstep(-1.5, 0.5, f(90))
    P = strew(where(MIMOSA), 56)
    a0, reach = 2.1 + r.normal(0, 0.7, len(P)), r.uniform(40, 90, len(P))
    Q = np.concatenate([P + np.stack([np.cos(a0), np.sin(a0)], 1) * (reach * r.random(len(P)))[:, None]
                        + r.normal(0, 7, P.shape) for _ in range(9)])
    Q = Q[where(MIMOSA, LEAVES)[at(Q)] & (r.random(len(Q)) < 0.15 + 0.85 * full[at(Q)])]
    th = r.uniform(0, np.pi, len(Q))
    d = 2.5 * np.stack([np.cos(th), np.sin(th)], 1)
    cols, share = brushes(SPRAY, U[at(Q)], clump[pid[at(Q)]], 0.0, r, 0.1, 0.15)
    lay(np.stack([Q - d, Q, Q + d], 1), r.uniform(3.5, 6, len(Q)), cols, share, thick=0.28, ends=(0.9, 0.9),
        taper=0.1, spent=0.1, pickup=0.0)
    Q = strew(where(MIMOSA) & (U > 0.72), 34)
    th = r.uniform(0, np.pi, len(Q))
    d = 3 * np.stack([np.cos(th), np.sin(th)], 1)
    cols, share = brushes(SUNLIT, U[at(Q)], clump[pid[at(Q)]], 0.0, r, 0.1, 0.06)
    lay(np.stack([Q - d, Q, Q + d], 1), r.uniform(4, 6.5, len(Q)), cols, share, thick=0.4, ends=(0.9, 0.9),
        taper=0.1, spent=0.1, pickup=0.0)

    # the branches of the almond, in a few dark strokes from below, and the last of its flowers over them
    for x0, y0, x1, y1 in ((1990, 1240, 2010, 820), (2010, 820, 1950, 600), (2010, 820, 2160, 620),
                           (2160, 620, 2230, 420), (2160, 620, 2070, 470), (1950, 600, 2000, 380),
                           (2230, 420, 2180, 290), (2230, 420, 2320, 500), (2010, 820, 2240, 790)):
        line = hand(segment((x0, y0), (x1, y1), 30), 6, 8, r)
        for m in pieces(line, 220, 20, r, 0.8):
            lay([line[m]], r.uniform(7, 11) * (1 + 0.4 * (y0 > 1000)),
                np.concatenate([tint(BARK, 0.05), tint(VIOLET, 0.07)])[None], np.array([[0.7, 0.3]]),
                taper=0.6, hide=0.85)
    cover((ALMOND,), 14, 10, 0.6, **{**dab, "thick": 0.32, "spread": 0.08, "patch": 0.6})
    wet[:] = 0

    # the window: the frame round the panes and the bars across them, painted in strokes of pale violet,
    # blue and a little orange that thicken and thin, and break off where the light behind eats into them
    o = OUTER
    top, foot = -20.0, o[2][1]
    BARS = ((LILAC, 0.5, 0.42), (COBALT, 0.3, 0.22), (BLUEV, 0.24, 0.08), (CREAM, 0.62, 0.14), (ORANGE, 0.4, 0.14))
    frame = [((o[0][0] + STILE, top), (o[0][0] + STILE, foot), STILE, (ORANGE, ROSE, LILAC)),
             ((o[1][0] - STILE, top), (o[1][0] - STILE, foot), STILE, (ROSE, LILAC, COBALT))]
    for v in VBARS:
        x = o[0][0] + (o[1][0] - o[0][0]) * v
        frame.append(((x, top), (x, foot - STILE), BAR, None))
    for y in HBARS:
        frame.append(((o[0][0] + STILE, y), (o[1][0] - STILE, y), BAR, None))
    fams, vals, odds = [b[0] for b in BARS], np.array([b[1] for b in BARS]), np.array([b[2] for b in BARS])
    for p, q, hw, own in frame:
        line = segment(p, q, 160)
        upright = abs(q[0] - p[0]) < abs(q[1] - p[1])
        side = np.array([-1.0, 0.0]) if upright else np.array([0.0, 1.0])
        for m in pieces(line, 300, r.uniform(-30, 30) if own is None else 40, r):
            seg = hand(line[m], 3.5, 40, r)
            glare = float(light[at(seg)].mean())
            if own is None and r.random() < 0.3 * glare ** 2:
                continue
            if own is None:
                k3 = r.choice(len(BARS), 3, p=odds / odds.sum())
                cols = np.concatenate([tint(fams[j], vals[j] * r.uniform(0.85, 1.15)) for j in k3])
            else:
                cols = np.concatenate([tint(fam, 0.42 * r.uniform(0.85, 1.15)) for fam in own])
            w = hw * np.exp(r.normal(0, 0.3))
            lay([seg], w, cols[None], np.array([[0.62, 0.28, 0.1]]), thick=0.16, taper=0.25, ends=(0.4, 0.3),
                dry=float(np.clip(r.normal(0.05, 0.05), 0, 0.15)), spent=0.7, pickup=0.05, tails=0.4)
            if r.random() < 0.4:
                e = seg + side * w * r.uniform(0.6, 1.0)
                cut = pieces(e, 140, -40, r, 0.8)
                for mm in cut:
                    blue = r.random() < 0.65
                    lay([hand(e[mm], 1.5, 15, r)], r.uniform(2.5, 4.5),
                        np.concatenate([tint(COBALT if blue else ORANGE, r.uniform(0.16, 0.24) if blue else 0.36),
                                        tint(BLUEV if blue else VERMILION, 0.18 if blue else 0.3),
                                        tint(VIOLET if blue else OCHRE, 0.16 if blue else 0.4)])[None],
                        np.array([[0.6, 0.3, 0.1]]), taper=0.5, thick=0.12)
    wet[:] = 0
    # the light of the garden eating into the bars: touches of what lies behind, laid across them
    xs = [o[0][0] + (o[1][0] - o[0][0]) * v for v in VBARS]
    near = np.zeros((H, W), bool)
    for x in xs:
        near |= np.abs(xx - x) < BAR * 1.6
    for y in HBARS:
        near |= np.abs(yy - y) < BAR * 1.6
    near &= (key == VIEW) & (r.random((H, W)) < 0.6 * light ** 2)
    touches(strew(near, 20.0), 15, 5.0, **{**dab, "jitter": 14, "n": 4})
    wet[:] = 0
    # the bottom rail, pulled across in three long strokes laid end over end so that it runs as one, the
    # orange of its face streaked with ochre and rose, its shadow under it and the light of the sky on its top
    rail = hand(segment((o[3][0] + 6, foot - STILE), (o[2][0] - 6, foot - STILE), 120), 2.0, 60, r)
    for a_, b_ in ((0, 48), (36, 86), (72, 120)):
        lay([rail[a_:b_]], STILE * r.uniform(0.95, 1.05),
            np.concatenate([tint(ORANGE, 0.36), tint(OCHRE, 0.42), tint(ROSE, 0.38)])[None],
            np.array([[0.6, 0.28, 0.12]]), thick=0.14, spent=0.2, taper=0.05, ends=(0.3, 0.3), pickup=0.3, tails=0.2)
    for dy, w, fams, v in ((0.8, 4.5, (COBALT, VIOLET, BLUEV), 0.14), (-0.8, 3.0, (PALE, CREAM, LEMON), 0.72)):
        e = rail + [0, dy * STILE]
        for a_, b_ in ((0, 66), (58, 120)):
            lay([e[a_:b_]], w * r.uniform(0.9, 1.1), np.concatenate([tint(fm, v) for fm in fams])[None],
                np.array([[0.6, 0.3, 0.1]]), thick=0.1, spent=0.3, taper=0.3, ends=(0.4, 0.3))
    wet[:] = 0

    # the still life on the cloth: the shadows the bowl and the fruit throw toward the room, the bowl, the
    # fruit heaped in it from the back, the near rim of the bowl across them, and two more fruit on the cloth
    cx, cy, a, b = BOWL
    shade = lambda v: np.concatenate([tint(COBALT, v), tint(LILAC, v * 1.25), tint(VIOLET, v * 0.9)])[None]
    for k in range(6):
        path = arc(cx - 70 + r.normal(0, 8), cy + 70 + 16 * k, a * 1.05, b * 0.9, 0.15 * np.pi, 0.95 * np.pi, 9)
        lay([hand(path, 3, 8, r)], r.uniform(13, 18), shade(r.uniform(0.13, 0.2)), np.array([[0.6, 0.3, 0.1]]),
            hide=0.85, thick=0.04, taper=0.4, spent=0.9)
    for x, y, R, lemon in LOOSE:
        e = np.array([np.cos(SUN), np.sin(SUN)]) * -1.0
        path = np.stack([np.array([x, y]) + e * R * s_ for s_ in (0.1, 0.8, 1.5)]) + [0, 0.45 * R]
        lay([path], 0.75 * R, shade(0.15), np.array([[0.6, 0.3, 0.1]]), hide=0.88, thick=0.04, ends=(0.9, 0.7))
    inner = arc(cx, cy, a * 0.9, b * 0.75, np.pi * 1.05, np.pi * 1.95, 10)
    lay([inner], 24, np.concatenate([tint(COBALT, 0.12), tint(VIOLET, 0.1), tint(TURQ, 0.18)])[None],
        np.array([[0.5, 0.3, 0.2]]), thick=0.05)
    for depth in np.linspace(0.12, 0.92, 5):
        body = arc(cx, cy + depth * DEPTH, a * (1 - 0.25 * depth ** 2), b * (1 + 0.12 * depth), 0.06 * np.pi, 0.94 * np.pi, 15)
        for sl, v, fams_ in ((slice(0, 6), 0.62, (WHITE, TURQ, PALE)), (slice(5, 11), 0.38, (LILAC, WHITE, TURQ)),
                             (slice(10, 15), 0.17, (COBALT, LILAC, ULTRA))):
            v *= 1 - 0.4 * depth
            lay([hand(body[sl], 1.5, 6, r)], 12 + 3 * depth,
                np.concatenate([tint(fm, v * r.uniform(0.9, 1.1)) for fm in fams_])[None],
                np.array([[0.6, 0.3, 0.1]]), thick=0.07, taper=0.3, ends=(0.6, 0.6), pickup=0.2)
    wet[:] = 0
    for x, y, R, lemon in sorted(FRUIT, key=lambda e: e[1]) + LOOSE:
        M, strokes = fruit(x, y, R, lemon, r)
        e = int(1.6 * R)
        bx = slice(max(0, int(x) - e), min(W, int(x) + e))
        by = slice(max(0, int(y) - e), min(H, int(y) + e))
        keep, kh = rgb[by, bx].copy(), height[by, bx].copy()
        for paths, ws, cols, share, inside_ in strokes:
            if not inside_ and keep is not None:          # what went on inside is cut to the fruit's outline
                g = np.stack(np.meshgrid(np.arange(bx.start, bx.stop), np.arange(by.start, by.stop)), -1) - [x, y]
                g = g @ np.linalg.inv(M).T                      # the box in the fruit's own unit disc
                th = np.arctan2(g[..., 1], g[..., 0])
                rim_ = (1 + 0.03 * np.sin(3 * th + r.uniform(0, 6.3)) + 0.02 * np.sin(5 * th + r.uniform(0, 6.3))
                        + 0.07 * lemon * np.cos(th) ** 24)          # a lemon comes to a point at each end
                m = noise.smoothstep(rim_ + 0.025, rim_ - 0.025, np.linalg.norm(g, axis=-1))[..., None]
                rgb[by, bx] = keep + (rgb[by, bx] - keep) * m
                height[by, bx] = kh + (height[by, bx] - kh) * m[..., 0]
                keep = None
            lay(paths, ws, cols, share, thick=0.12, ends=(0.6, 0.5), taper=0.15, tails=0.1, fray=0.8, spent=0.15,
                pickup=0.45, merge=4.0, hide=1.0)
        wet[:] = 0
    rim = arc(cx, cy, a, b, 0.02 * np.pi, 0.98 * np.pi, 24)
    for m in pieces(rim, 170, -12, r, 0.9):
        lay([hand(rim[m], 1.2, 6, r)], r.uniform(4, 6.5),
            np.concatenate([WHITE[-1:], tint(TURQ, 0.5), tint(PALE, 0.8)])[None], np.array([[0.6, 0.3, 0.1]]),
            taper=0.35, thick=0.14)
    under = arc(cx, cy + 9, a * 0.99, b, 0.08 * np.pi, 0.92 * np.pi, 20)
    for m in pieces(under, 160, -30, r, 0.8):
        lay([hand(under[m], 1.2, 6, r)], r.uniform(2.2, 3.5),
            np.concatenate([tint(ULTRA, 0.09), tint(COBALT, 0.14), tint(VIOLET, 0.1)])[None],
            np.array([[0.6, 0.3, 0.1]]), taper=0.4, thick=0.1)

    height = ndimage.gaussian_filter(height, 0.6)
    img = dabs.shine(rgb, height, light=LIGHT, relief=1.0, gloss=0, reach=(0.8, 1.16))
    return img + 0.03 * impasto.glints(height, LIGHT)[..., None]
