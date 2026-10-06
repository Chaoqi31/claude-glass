"""Ghost Gum, West MacDonnell Ranges. Watercolour over pencil on rough rag paper.

Albert Namatjira was a Western Arrernte man of the Lutheran mission at Hermannsburg, west of
Alice Springs. In 1936 the painter Rex Battarbee took him on as a guide and showed him
watercolour, and within a few years he was painting his own country as no one had painted it: the
ranges, the dry beds of the Finke and its creeks, the ghost gums. His sons and neighbours took up
the same paper and the same colours, and the manner became the Hermannsburg School.

The picture starts as a careful pencil drawing, and some of it is never covered. The sky goes on
first, wet into wet, paling to the horizon, and the far hills are touched into it while it is
still damp, so their edges go soft. The near range is built on dry paper in glazes, each laid when
the last has dried: a pale warm wash over all of it, ochre and light red where the sun is on the
rock, rose and violet in the shade, blue at the bottoms of the gullies, so the rock glows through
the layers and every wash keeps the crisp, slightly darker edge where it dried. The ghost gum is
the paper itself, left white: its shadow is a lavender wash softened into the light on one side and
run up to the contour on the other. Foliage, mulga and scrub are touches of the point of the brush;
spinifex and the sand of the creek a nearly dry brush dragged over the hills of the paper.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from atelier import brush, noise, pencil, plate, wash
from atelier import watercolour as wc
from atelier.color import pigment
from atelier.paper import Sheet

TITLE = "Ghost Gum, West MacDonnell Ranges"
DATE = "2026"
MEDIUM = "Watercolour over pencil on rough rag paper"
AFTER = "the watercolours of Albert Namatjira and the Hermannsburg School, Central Australia, c. 1950"
ROOM = "Paper and Water"
YEAR = 1950
PLACE = "West MacDonnell Ranges"
REGION = "Oceania"
NOTE = ("A ghost gum on the bank of a dry creek under the West MacDonnell Ranges, its trunk the white of "
        "the paper. The rock is built up in glazes of ochre, light red and violet, each laid when the last was dry.")

H, W = 2240, 3200
X0, Y0, X1, Y1 = 172, 164, 3028, 2076       # the rectangle the painter worked to, inside the sheet
SW, SH = X1 - X0, Y1 - Y0
SUN = np.array([-0.78, -0.42, 0.46]) / np.linalg.norm([-0.78, -0.42, 0.46])   # from the left, a little in front
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

PALETTE = dict(   # the box, as absorbance at unit density
    cerulean=pigment("#80b4d8"), cobalt=pigment("#6a8fcc"), ultramarine=pigment("#5761b8"),
    rose=pigment("#d6758f"), light_red=pigment("#d5724c"), burnt_sienna=pigment("#ad5d38"),
    yellow_ochre=pigment("#dfb264"), raw_sienna=pigment("#cf9342"), gamboge=pigment("#eac94c"),
    viridian=pigment("#4d9a88"), indigo=pigment("#46536e"), graphite=pigment("#6e6e74"))
VIOLET = dict(cobalt=0.45, rose=0.32, ultramarine=0.23)        # the shadow colour, wherever a shadow falls

# ---- the drawing: (u, v) in fractions of the painted rectangle, sizes in fractions of its width ----
NEAR = [(-0.03, 0.425), (0.08, 0.410), (0.16, 0.415), (0.24, 0.400), (0.32, 0.385), (0.38, 0.360), (0.43, 0.330),
        (0.47, 0.315), (0.51, 0.285), (0.55, 0.250), (0.585, 0.228), (0.61, 0.205), (0.632, 0.186), (0.655, 0.192),
        (0.68, 0.215), (0.70, 0.238), (0.725, 0.232), (0.75, 0.214), (0.775, 0.222), (0.81, 0.258), (0.86, 0.297),
        (0.91, 0.322), (0.96, 0.338), (1.03, 0.350)]
FAR = [(-0.03, 0.372), (0.06, 0.353), (0.12, 0.366), (0.20, 0.358), (0.28, 0.351), (0.36, 0.352), (0.50, 0.345),
       (0.70, 0.330), (0.82, 0.300), (0.87, 0.288), (0.92, 0.295), (0.97, 0.303), (1.03, 0.310)]
FOOTHILL = [(0.30, 0.514), (0.34, 0.488), (0.39, 0.466), (0.44, 0.452), (0.49, 0.455), (0.54, 0.463), (0.59, 0.472),
            (0.64, 0.490), (0.69, 0.514)]
FOOT = [(-0.03, 0.510), (0.12, 0.502), (0.25, 0.506), (0.38, 0.498), (0.5, 0.509), (0.62, 0.503), (0.75, 0.507),
        (0.9, 0.500), (1.03, 0.506)]
TREE = [   # the ghost gum: each bough from where it leaves its parent, (u, v, half-width)
    [(0.262, 0.917, 0.0215), (0.266, 0.862, 0.0172), (0.274, 0.792, 0.0156), (0.285, 0.722, 0.0150),
     (0.289, 0.662, 0.0150), (0.284, 0.612, 0.0148)],
    [(0.281, 0.626, 0.0118), (0.270, 0.546, 0.0104), (0.247, 0.466, 0.0094), (0.229, 0.391, 0.0084),
     (0.224, 0.321, 0.0073), (0.207, 0.251, 0.0060), (0.181, 0.181, 0.0045), (0.163, 0.121, 0.0032), (0.155, 0.076, 0.0018)],
    [(0.288, 0.631, 0.0112), (0.305, 0.551, 0.0098), (0.323, 0.481, 0.0087), (0.345, 0.411, 0.0076),
     (0.360, 0.341, 0.0065), (0.373, 0.271, 0.0053), (0.390, 0.201, 0.0040), (0.399, 0.141, 0.0027), (0.395, 0.096, 0.0015)],
    [(0.276, 0.716, 0.0075), (0.252, 0.671, 0.0065), (0.217, 0.634, 0.0055), (0.182, 0.617, 0.0046),
     (0.143, 0.601, 0.0036), (0.108, 0.589, 0.0025), (0.082, 0.576, 0.0014)],
    [(0.226, 0.371, 0.0048), (0.256, 0.311, 0.0040), (0.288, 0.251, 0.0031), (0.303, 0.191, 0.0022), (0.309, 0.156, 0.0012)],
    [(0.205, 0.246, 0.0036), (0.172, 0.223, 0.0029), (0.134, 0.206, 0.0020), (0.104, 0.191, 0.0011)],
    [(0.245, 0.461, 0.0048), (0.205, 0.431, 0.0040), (0.168, 0.401, 0.0031), (0.133, 0.371, 0.0022), (0.104, 0.339, 0.0012)],
    [(0.347, 0.406, 0.0046), (0.390, 0.376, 0.0038), (0.430, 0.353, 0.0029), (0.468, 0.336, 0.0020), (0.500, 0.316, 0.0011)],
    [(0.372, 0.266, 0.0031), (0.348, 0.211, 0.0024), (0.332, 0.166, 0.0012)],
    [(0.390, 0.196, 0.0026), (0.428, 0.166, 0.0019), (0.455, 0.146, 0.0010)],
    [(0.182, 0.616, 0.0026), (0.163, 0.566, 0.0019), (0.152, 0.526, 0.0010)],
    [(0.258, 0.905, 0.0060), (0.236, 0.924, 0.0040), (0.212, 0.934, 0.0015)],      # roots over the bank
    [(0.268, 0.906, 0.0058), (0.296, 0.923, 0.0038), (0.322, 0.930, 0.0014)],
]
CLUMPS = [(0.150, 0.075, 0.050, 0.030), (0.200, 0.135, 0.045, 0.028), (0.312, 0.140, 0.050, 0.032),
          (0.262, 0.200, 0.040, 0.026), (0.100, 0.180, 0.050, 0.032), (0.135, 0.262, 0.038, 0.026),
          (0.100, 0.330, 0.052, 0.034), (0.398, 0.085, 0.050, 0.030), (0.335, 0.155, 0.040, 0.026),
          (0.458, 0.135, 0.050, 0.030), (0.445, 0.232, 0.040, 0.026), (0.505, 0.305, 0.055, 0.034),
          (0.420, 0.322, 0.035, 0.024), (0.080, 0.565, 0.050, 0.032), (0.150, 0.515, 0.040, 0.028),
          (0.360, 0.110, 0.035, 0.022), (0.200, 0.400, 0.035, 0.024)]
CREEK = [(0.60, 1.04, 0.120), (0.55, 0.95, 0.105), (0.50, 0.87, 0.085), (0.515, 0.80, 0.065), (0.57, 0.745, 0.050),
         (0.64, 0.695, 0.038), (0.70, 0.65, 0.028), (0.75, 0.61, 0.019), (0.785, 0.575, 0.012), (0.805, 0.548, 0.006),
         (0.815, 0.535, 0.002)]
HUMMOCKS = [(0.05, 0.93, 0.09), (0.15, 0.975, 0.11), (0.36, 0.95, 0.075), (0.11, 0.83, 0.06), (0.385, 0.815, 0.05),
            (0.86, 0.86, 0.085), (0.955, 0.95, 0.10), (0.79, 0.975, 0.07), (0.93, 0.765, 0.05), (0.21, 0.765, 0.045),
            (0.03, 0.785, 0.05), (0.335, 0.875, 0.05), (0.60, 0.665, 0.03), (0.17, 0.70, 0.035), (0.88, 0.69, 0.035)]
ROCKS = [(0.445, 0.985, 0.040), (0.70, 0.905, 0.030), (0.735, 0.80, 0.022), (0.295, 0.985, 0.026), (0.985, 0.85, 0.03)]


def P(rows):
    """Rows (u, v, sizes...) in fractions of the painted rectangle -> px (sizes scale with its width)."""
    a = np.array(rows, np.float64)
    a[:, 0] = X0 + a[:, 0] * SW
    a[:, 1] = Y0 + a[:, 1] * SH
    a[:, 2:] *= SW
    return a


def _s(r):
    return int(r.integers(1 << 31))


def near_box(cx, cy, rx, ry):
    """Slices round a small thing at (cx, cy), so its washes are worked out only where it is."""
    return (slice(max(0, int(cy - ry)), min(H, int(cy + ry))), slice(max(0, int(cx - rx)), min(W, int(cx + rx))))


def profile(rows, r, rock=2.5):
    """A line across the picture through control rows (u, v): y in px for every column, broken a
    little by the rock."""
    C = brush.path(P(rows), 1.0)
    y = np.interp(np.arange(W), C[:, 0], C[:, 1])
    return (y + rock * (noise.line1d(W, 28, r) + 0.4 * noise.line1d(W, 6, r))).astype(np.float32)


class Box:
    """The paint box. Transparent washes multiply, so the order they went on does not change the
    colour: each is kept as a density per pigment and glazed onto the paper at the end. Every wash
    stops at the edge of the painted rectangle and, but for the bark's own, is cut round the boughs."""

    def __init__(self, sheet, r, cut, cover):
        self.sheet, self.r, self.cut, self.keep = sheet, r, cut, cut * (1 - cover)
        self.d = {k: np.zeros((H, W), np.float32) for k in PALETTE}
        self.soak = np.zeros((H, W), np.float32)
        self.strokes = {}

    def lay(self, density, bark=False, **mix):
        d = density * (self.cut if bark else self.keep)
        for k, f in mix.items():
            self.d[k] += f * d

    def wash(self, mask, ragged=0.9, bark=False, **kw):
        """A wash laid on dry paper over `mask`: (where its water lay, the pigment it left)."""
        mask = mask * (self.cut if bark else self.keep)
        ys, xs = np.flatnonzero(mask.max(1) > 0.02), np.flatnonzero(mask.max(0) > 0.02)
        wet, dep = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        if len(ys):
            sl = (slice(max(0, ys[0] - 24), ys[-1] + 25), slice(max(0, xs[0] - 24), xs[-1] + 25))
            sheet = Sheet(*(a[sl] for a in (self.sheet.color, self.sheet.tooth, self.sheet.fiber, self.sheet.alpha)))
            wet[sl] = wc.puddle(mask[sl], sheet, self.r, ragged)
            if wet[sl].max() > 0.02:          # a wash too thin to stand leaves nothing
                dep[sl] = wc.deposit(wet[sl], sheet, self.r, **kw)
            self.soak[sl] = np.maximum(self.soak[sl], wet[sl])
        return wet, dep

    def stroke(self, key, rows, radius, **kw):
        ink, water = self.strokes.setdefault(key, (np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)))
        kw = dict(dict(load=1.0, clumps=5, splay=0.1, head=0.5, bristles=int(np.clip(12 * radius, 24, 160))), **kw)
        brush.stroke(self.sheet, rows, radius, seed=_s(self.r), into=(ink, water), **kw)

    def take(self, key, dry=0.0):
        """The pigment the strokes under `key` left; a dry brush (`dry` > 0) leaves it only on the
        hills of the paper, more of them the harder it was pressed."""
        ink = self.strokes.pop(key)[0]
        return pencil.catch(ink, self.sheet, grip=dry) if dry else ink

    def touch(self, rows, density, **mix):
        """Touches of the point, laid on dry paper and left to dry each with its rim."""
        if len(rows):
            _, dep = self.wash(wash.touches((H, W), rows), 0.5, pool=0.15, rim=0.9, rim_width=1.5, scale=40)
            self.lay(dep * density, **mix)

    def glaze(self):
        img = self.sheet.color.copy()
        for k, kk in PALETTE.items():
            img *= np.exp(-self.d[k][..., None] * kk)
        return img


def frame(r):
    """The rectangle the painter brushed out to, judged by eye: its sides wander a little."""
    side = lambda n: 3.5 * noise.line1d(n, 300, r) + 1.2 * noise.line1d(n, 40, r)
    d = np.minimum.reduce([XX - X0 - side(H)[:, None], X1 + side(H)[:, None] - XX,
                           YY - Y0 - side(W)[None, :], Y1 + side(W)[None, :] - YY])
    return noise.smoothstep(-1, 1, d).astype(np.float32)


# ---- the ghost gum ------------------------------------------------------------------------------
def spine(rows, r, wander=0.3):
    """A bough through control rows (x, y, half-width) px, resampled every px: it wanders a little
    off the line through them, and swells and narrows as a gum's limbs do."""
    C = brush.path(rows, 1.0)
    n = len(C)
    s = np.arange(n, dtype=np.float64)
    for k in range(2):
        T = ndimage.gaussian_filter1d(np.gradient(C[:, :2], axis=0), 3, axis=0)
        T /= np.hypot(*T.T)[:, None] + 1e-9
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        if k == 0:     # held fast where it leaves its parent
            C[:, :2] += N * (wander * C[:, 2] * noise.smoothstep(0, 60, s) * noise.line1d(n, 110, r))[:, None]
            C[:, 2] *= 1 + 0.07 * noise.line1d(n, 70, r)
    return C, T, N


def boughs(spines):
    """The tree as the paper sees it: for every px, how much of it the boughs cover, and the light a
    round bough catches there from SUN, taken from the bough the px lies deepest in."""
    cover = np.zeros((H, W), np.float32)
    light = np.zeros((H, W), np.float32)
    deep = np.full((H, W), -1e9, np.float32)
    for C, T, N in spines:
        hw = C[:, 2]
        x0, y0 = np.maximum(np.floor(C[:, :2].min(0) - hw.max() - 4).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(C[:, :2].max(0) + hw.max() + 4).astype(int), (W, H))
        im = Image.new("L", (x1 - x0, y1 - y0), 0)
        dr = ImageDraw.Draw(im)
        for (x, y, w) in C[::2]:
            dr.ellipse([x - x0 - w - 2, y - y0 - w - 2, x - x0 + w + 2, y - y0 + w + 2], fill=1)
        ys, xs = np.nonzero(np.asarray(im))
        p = np.stack([xs + x0, ys + y0], 1).astype(np.float64)
        _, i = cKDTree(C[:, :2]).query(p)
        d = p - C[i, :2]
        o, a = (d * N[i]).sum(1), (d * T[i]).sum(1)
        inside = hw[i] - np.abs(o)
        cov = np.clip(inside + 0.5, 0, 1) * np.clip(1.5 - np.abs(a), 0, 1)
        u = np.clip(o / hw[i], -1, 1)
        s = np.sqrt(1 - u * u) * SUN[2] + u * (N[i] @ SUN[:2])
        Y, X = ys + y0, xs + x0
        cover[Y, X] = np.maximum(cover[Y, X], cov)
        win = (cov > 0) & (inside > deep[Y, X])
        light[Y[win], X[win]] = s[win]
        deep[Y[win], X[win]] = inside[win]
    # where two boughs meet, the light is the same bark turning; smooth over the seam
    return cover, ndimage.gaussian_filter(light * cover, 2.0) / (ndimage.gaussian_filter(cover, 2.0) + 1e-6)


def tree(r):
    """The boughs, kept white, and the twigs that carry the clumps, painted thin and pale."""
    spines, twigs = [spine(P(rows), r) for rows in TREE], []
    # in every clump a twig or two runs out from the nearest bough to carry the leaves
    pts = np.vstack([C[:, :2] for C, _, _ in spines])
    kd = cKDTree(pts)
    for cx, cy, rx, ry in P(CLUMPS):
        _, i = kd.query((cx, cy))
        ax, ay = pts[i]
        for _ in range(2):
            a = r.uniform(0, 2 * np.pi)
            tx, ty = cx + 0.5 * rx * np.cos(a), cy + 0.45 * ry * np.sin(a)
            mx, my = (ax + tx) / 2 + r.normal(0, 6), (ay + ty) / 2 + r.normal(0, 6)
            twigs.append(np.array([(ax, ay, 0.9), (mx, my, 0.6), (tx, ty, 0.1)]))
    return spines, twigs


def foliage(box, r, twigs):
    """Grey-green leaves hanging in loose clumps, touched in with the point in three passes: pale where
    the sun is on a clump, grey-green through it, blue-green beneath, and the sky left showing through;
    the twigs that carry them first, in a thin pale grey."""
    for rows in twigs:
        box.stroke("twig", rows, 2.2, load=1.0, bristles=24)
    box.lay(box.take("twig") * 0.5, cobalt=0.3, rose=0.2, burnt_sienna=0.12)
    passes = {"pale": [], "mid": [], "dark": []}
    for cx, cy, rx, ry in P(CLUMPS):
        rx, ry = rx * r.uniform(0.85, 1.15), ry * r.uniform(0.8, 1.2)
        n = int(np.pi * rx * ry / 40 * r.uniform(0.6, 1.0))
        q = r.uniform(-1, 1, (n * 2, 2))
        lobes = 1 + 0.22 * np.sin(np.arctan2(q[:, 1], q[:, 0]) * r.integers(3, 6) + r.uniform(0, 6))
        q = q[np.hypot(*q.T) < lobes * r.uniform(0.7, 1.0, len(q))][:n]
        x, y = cx + q[:, 0] * rx, cy + q[:, 1] * ry
        sun = -(q[:, 0] * SUN[0] + q[:, 1] * SUN[1] * 1.6) + 0.35 * r.standard_normal(len(q))
        L = r.uniform(10, 19, len(q))
        rows = np.stack([x, y, L, L * r.uniform(0.3, 0.42, len(q)), np.pi / 2 + r.normal(0, 0.5, len(q))], 1)
        passes["pale"].append(rows[sun > 0.1])
        passes["mid"].append(rows[(sun < 0.6) & (r.uniform(0, 1, len(q)) < 0.6)])
        passes["dark"].append(rows[(sun < -0.35) & (r.uniform(0, 1, len(q)) < 0.7)])
    for key, dens, mix in (("pale", 0.5, dict(gamboge=0.22, viridian=0.26, raw_sienna=0.14, cobalt=0.04)),
                           ("mid", 0.45, dict(viridian=0.36, raw_sienna=0.22, cobalt=0.12, rose=0.05)),
                           ("dark", 0.5, dict(indigo=0.36, viridian=0.3, raw_sienna=0.18))):
        box.touch(np.vstack(passes[key]), dens, **mix)


def bark(box, r, cover, light, spines):
    """Shadow on white bark: lavender laid on the side away from the sun, softened into the light,
    strongest just past the turn of the bough and a little lighter at its far edge, where light comes
    back off the red ground."""
    j = light + 0.05 * noise.field((H, W), 50, r) + 0.025 * noise.field((H, W), 9, r)
    shade = noise.smoothstep(0.42, -0.1, j) * (1 - 0.3 * noise.smoothstep(-0.35, -0.75, light)) * cover
    _, dep = box.wash((shade > 0.02) * cover, 0.5, bark=True, pool=0.12, rim=0.7, rim_width=2.0, grain=0.35, scale=80)
    box.lay(shade * dep * 0.75, bark=True, cobalt=0.5, rose=0.32, ultramarine=0.12)
    low = noise.smoothstep(0.62 * SH + Y0, 0.9 * SH + Y0, YY)
    box.lay(noise.smoothstep(-0.3, -0.7, light) * cover * dep * (0.05 + 0.12 * low), bark=True, rose=0.6, raw_sienna=0.5)
    # dust off the bank on the foot of the trunk, run in while the shadow was wet
    box.lay(cover * dep * noise.smoothstep(0.85 * SH + Y0, 0.93 * SH + Y0, YY) * 0.18, bark=True,
            yellow_ochre=0.7, light_red=0.3)
    # where a limb was shed the bark has healed in a dark eye, and in the forks it wrinkles
    for C, T, N in spines[:4]:
        for f in r.uniform(0.15, 0.75, 2):
            i = int(f * (len(C) - 1))
            x, y = C[i, :2] + N[i] * C[i, 2] * r.uniform(-0.4, 0.5)
            w = C[i, 2] * r.uniform(0.18, 0.3)
            a = np.arctan2(N[i, 1], N[i, 0])
            box.stroke("scar", [(x - np.cos(a) * w, y - np.sin(a) * w, 0.3), (x, y, 0.9),
                                (x + np.cos(a) * w, y + np.sin(a) * w, 0.2)], w * 0.3, load=1.0)
    C = spines[0][0]
    for k in range(4):
        x, y, w = C[-1 - 14 * k]
        box.stroke("crease", [(x - w * 0.7, y + 6 + k * 3, 0.1), (x, y - 4, 0.5 - 0.08 * k), (x + w * 0.6, y + 4, 0.05)],
                   1.5, load=0.8)
    box.lay(box.take("scar") * cover, bark=True, burnt_sienna=0.6, indigo=0.35)
    box.lay(box.take("crease") * 0.7 * cover, bark=True, cobalt=0.4, rose=0.3, burnt_sienna=0.3)


# ---- the land -----------------------------------------------------------------------------------
def massif(top, foot, r, spacing=(130, 300), relief=1.1, bulk=2.5, drift=0.5):
    """The face of a range between its skyline `top` and its foot (y per column): cliffs at the top
    and an apron of scree below, leaning back from the eye the further down. Spurs come down it from
    the skyline's peaks, rounded and widening as they go, bending the way the range falls and
    sinking into the scree at different depths, with a gully between each and the next; the
    summit's own crest runs down and forward, so its shadow side does too. Returns
    (mask, how far down the face each px lies, the light it catches from SUN, the skyline with its peaks)."""
    cols = np.arange(W, dtype=np.float32)
    heads, x = [], X0 - 150 + r.uniform(0, 120)
    while x < X1 + 150:
        heads.append((x, r.uniform(*spacing) * 0.55, r.uniform(0.5, 1.0), r.uniform(0.35, 0.8)))
        x += r.uniform(*spacing)
    top = top.copy()
    for x, w, f, _ in heads:
        top -= 10 * f * np.clip(1 - np.abs(cols - x) / w, 0, None) ** 1.5
    fall = np.gradient(ndimage.gaussian_filter1d(top, 60))      # > 0 where the skyline falls to the right
    h = np.maximum(foot - top, 1)
    y0, y1 = int(top[X0:X1].min()) - 4, int(foot.max()) + 4
    yy, xx = YY[y0:y1], XX[y0:y1]
    t = np.clip((yy - top) / h, 0, 1.2)
    hs = ndimage.gaussian_filter1d(h, 80)
    z = h * (0.3 * t + 0.25 * t ** 3) + 5 * noise.field(yy.shape, 70, r)
    for x, w, f, end in heads:
        xi = int(np.clip(x, 0, W - 1))
        lean, bow = 2.0 * fall[xi] * h[xi], r.normal(0, 0.12) * h[xi]
        side = np.clip(1.5 * fall[xi], -0.4, 0.4)
        reach = int(3 * w + abs(lean) + abs(bow) + 20)
        sl = slice(max(0, xi - reach), min(W, xi + reach))
        tt = np.clip(t[:, sl], 0, 1)
        ww = w * (0.4 + 1.6 * tt) * r.uniform(0.8, 1.2)
        d = xx[:, sl] - (x + lean * tt ** 0.8 + bow * np.sin(np.pi * tt))
        s = np.where(d < 0, -d / (ww * (1 - side)), d / (ww * (1 + side)))
        tall = noise.smoothstep(250, 550, h[xi])          # a low ridge shows little of its spurs
        z[:, sl] += relief * tall * 0.6 * ww * f * (1 - noise.smoothstep(end - 0.3, end, tt)) * np.clip(1 - s * s, 0, None)
    gy, gx = np.gradient(z)
    # the bulk of the range turns its faces towards the sun or away; its crest runs down and forward
    # from the summit
    gx += bulk * np.interp(xx - drift * t * hs[None, :], cols, np.gradient(hs))
    S = np.zeros((H, W), np.float32)
    T = np.zeros((H, W), np.float32)
    S[y0:y1] = ndimage.gaussian_filter((-gx * SUN[0] - gy * SUN[1] + SUN[2]) / np.sqrt(gx * gx + gy * gy + 1), 3.5)
    T[y0:y1] = t
    m = np.clip(YY - top[None, :] + 0.5, 0, 1) * np.clip(foot[None, :] - YY + 0.5, 0, 1)
    return m.astype(np.float32), T, S, top


def sky(box, r, near, far):
    """The sky, wet into wet: deep at the top, nearly nothing at the horizon, a little warm low down;
    then the far hills touched into it while it is still damp."""
    low = near + 24 + 8 * noise.line1d(W, 200, r)
    wet, dep = box.wash((YY < low[None, :]).astype(np.float32), 0.8, pool=0.1, rim=0.3, rim_width=4, tides=0.04,
                        grain=0.25, scale=300)
    v = np.clip((YY - Y0) / (low[None, :] - Y0), 0, 1)
    bands = 1 + 0.03 * noise.line1d(H, 45, r)[:, None] + 0.05 * noise.field((H, W), 260, r)
    box.lay(dep * bands * (0.05 + 0.52 * (1 - v) ** 1.5), cerulean=0.6, cobalt=0.4)
    box.lay(dep * noise.smoothstep(0.5, 1.0, v) * 0.1, yellow_ochre=0.7, rose=0.4)
    fm, _, fs, far = massif(far, near + 80, r, spacing=(90, 200), relief=0.6, bulk=0.6)
    shade = 1 + 0.7 * ndimage.gaussian_filter(noise.smoothstep(0.42, 0.2, fs), 3)
    hills = wc.charge(ndimage.gaussian_filter(fm * shade, 2.5), wet, 5, r, streak=0.05)
    box.lay(hills * 0.4, cobalt=0.5, rose=0.4, ultramarine=0.15)
    return far


def broad(mask, k=11):
    """A shape as a brush lays it: no sliver or chink narrower than about `k` px, its corners round."""
    return noise.smoothstep(0.42, 0.58, ndimage.gaussian_filter(mask, k / 2.5))


def rock_face(box, r, m, t, S, top, foot, warm=1.0, scrub=1.0, scree=0.5):
    """A range built up in glazes on dry paper: a pale warm wash over all of it, ochre and light red
    where the sun is on the rock, more light red where the faces turn from it, rose and violet in the
    shade, blue in the bottoms of the gullies. Then, the way Namatjira modelled rock, row after row of
    small strokes of the point laid along the beds, red in the sun and violet in the shade, few where
    the sun is full on the face; and the scrub on the slopes."""
    j = S + 0.06 * noise.field((H, W), 120, r) + 0.02 * noise.field((H, W), 40, r)
    lit = broad(noise.smoothstep(0.08, 0.14, j))
    vary = 1 + 0.25 * noise.field((H, W), 220, r)
    _, dep = box.wash(m, 0.8, rim=0.45, grain=0.3, scale=220)
    box.lay(dep, yellow_ochre=0.08, rose=0.08, cobalt=0.02)
    box.lay(dep * noise.smoothstep(scree, scree + 0.45, t), raw_sienna=0.14, viridian=0.07, gamboge=0.03)
    _, dep = box.wash(m * lit, 0.8, rim=0.6, grain=0.3, scale=160)
    box.lay(dep * (1 - 0.4 * t), yellow_ochre=0.2, light_red=0.28 * warm * vary, rose=0.04 + 0.08 * (1 - t))
    _, dep = box.wash(m * lit * broad(noise.smoothstep(0.62, 0.56, j)), 0.8, rim=0.6, grain=0.3, scale=140)
    box.lay(dep * (1 - 0.5 * t), light_red=0.24 * warm * vary, rose=0.07)
    _, dep = box.wash(m * broad(noise.smoothstep(0.14, 0.08, j)), 0.8, rim=0.8, grain=0.55, scale=120)
    box.lay(dep, rose=0.21, ultramarine=0.22, cobalt=0.13)
    _, dep = box.wash(m * broad(noise.smoothstep(-0.12, -0.18, j + 0.05 * noise.field((H, W), 30, r))), 0.7, rim=0.9,
                      grain=0.7, scale=80)
    box.lay(dep, ultramarine=0.24, rose=0.08, indigo=0.05)
    # strokes along the beds: rows parallel to the skyline, a stroke every so often along each
    marks = {"sun": [], "turn": [], "shade": [], "deep": []}
    patch = noise.field((H, W), 160, r)
    slope = np.gradient(ndimage.gaussian_filter1d(top, 20))
    for row, d in enumerate(np.arange(r.uniform(4, 12), (foot - top).max(), 13.0)):
        ledge = row % 4 == 2 and d < 0.6 * (foot - top).max()      # now and then a bed stands out as a ledge
        xs = X0 + np.cumsum(r.uniform(8, 22, int(SW / 8)) * (2.5 if ledge else 1))
        xs = xs[xs < X1 - 2]
        ys = top[xs.astype(int)] + d + r.normal(0, 2.5, len(xs))
        ok = ys < foot[xs.astype(int)] - 4
        xs, ys = xs[ok], ys[ok]
        xi, yi = xs.astype(int), ys.astype(int)
        jj, pp, tt = j[yi, xi], patch[yi, xi], t[yi, xi]
        keep = (m[yi, xi] > 0.5) & (r.uniform(0, 1, len(xs)) < (0.8 if ledge else 0.45) + 0.25 * pp - 0.3 * tt)
        L = r.uniform(40, 90, len(xs)) if ledge else r.uniform(14, 34, len(xs))
        rows = np.stack([xs - L / 2, ys, L, r.uniform(3.5, 6.0, len(xs)), np.arctan(slope[xi]) + r.normal(0, 0.15, len(xs))], 1)
        for key, sel, p in (("sun", jj > 0.6, 0.35), ("turn", (jj > 0.12) & (jj <= 0.6), 0.65),
                            ("shade", (jj > -0.12) & (jj <= 0.12), 0.85), ("deep", jj <= -0.12, 0.9)):
            marks[key].append(rows[keep & sel & (r.uniform(0, 1, len(xs)) < p)])
    for key, dens, mix in (("sun", 0.35, dict(light_red=0.35, burnt_sienna=0.15)),
                           ("turn", 0.45, dict(burnt_sienna=0.3, rose=0.25, light_red=0.2)),
                           ("shade", 0.5, dict(ultramarine=0.3, rose=0.3, burnt_sienna=0.1)),
                           ("deep", 0.55, dict(ultramarine=0.4, indigo=0.2, rose=0.15))):
        box.touch(np.vstack(marks[key]), dens * warm ** 0.5, **mix)
    # scrub on the slopes: points of dark green, more of them low down
    y0, y1 = int(top[X0:X1].min()), int(foot.max())
    n = int(scrub * 0.0014 * (y1 - y0) * SW)
    xs, ys = r.uniform(X0, X1, n).astype(int), r.uniform(y0, y1, n).astype(int)
    tt = np.clip(t[ys, xs], 0, 1)
    ok = (m[ys, xs] > 0.5) & (r.uniform(0, 1, n) < 0.15 + 0.85 * tt ** 1.5)
    xs, ys, tt = xs[ok], ys[ok], tt[ok]
    size = 3.5 + 4.5 * tt
    box.touch(np.stack([xs, ys - size / 2, size, size * r.uniform(0.6, 0.9, len(xs)), np.full(len(xs), np.pi / 2)], 1),
              0.5, indigo=0.35, raw_sienna=0.3, viridian=0.2)


def ground(box, r, foot, clear):
    """The plain and the near bank in one wash with colour dropped into it wet: red earth, more of
    it close to, a little green in the middle distance, a cool haze at the foot of the range. The
    bed of the creek and the spinifex are left out, nearly bare paper."""
    g = np.clip((YY - foot[None, :]) / (Y1 - foot[None, :]), 0, 1)
    _, dep = box.wash((YY > foot[None, :] - 3) * clear, 0.9, pool=0.2, rim=0.4, grain=0.35, scale=260)
    box.lay(dep * (0.2 + 0.12 * g), yellow_ochre=0.75, raw_sienna=0.3)
    red = noise.stretched((H, W), 360, 50, r) + 0.25 * noise.field((H, W), 40, r) + 1.6 * g - 0.9
    for level, dens in ((0.25, 0.05 + 0.18 * g), (0.95, 0.03 + 0.07 * g)):    # red earth, laid when the first wash was dry
        _, pd = box.wash(noise.smoothstep(level - 0.05, level + 0.05, red) * clear * (YY > foot[None, :] + 6), 1.0,
                         rim=0.6, grain=0.45, scale=120)
        box.lay(pd * dens, light_red=0.7, burnt_sienna=0.3)
    grass = noise.smoothstep(0.2, 1.2, noise.stretched((H, W), 220, 40, r)) * noise.smoothstep(0, 0.08, g) \
        * (1 - noise.smoothstep(0.3, 0.55, g))
    box.lay(dep * grass * 0.16, gamboge=0.3, viridian=0.3, raw_sienna=0.4)
    box.lay(dep * (1 - noise.smoothstep(0.0, 0.1, g)) * 0.10, cobalt=0.5, rose=0.4)


def treeline(box, r, foot):
    """Trees along the foot of the range, too far off to be more than touches of blue-green."""
    xs = np.arange(X0, X1, 3.0)
    xs = xs + r.uniform(-2, 2, len(xs))
    grove = noise.smoothstep(-0.6, 0.8, noise.line1d(W, 120, r))
    xs = xs[r.uniform(0, 1, len(xs)) < 0.3 + 0.7 * grove[xs.astype(int)]]
    L = (5 + 9 * grove[xs.astype(int)]) * r.uniform(0.7, 1.3, len(xs))
    ys = foot[xs.astype(int)] + r.uniform(-3, 8, len(xs)) - L * 0.6
    box.touch(np.stack([xs, ys, L, L * r.uniform(0.6, 0.9, len(xs)), np.pi / 2 + r.normal(0, 0.2, len(xs))], 1),
              0.7, indigo=0.38, viridian=0.25, raw_sienna=0.15, cobalt=0.12)


def plain(box, r, foot, clear):
    """The plain: mulga in loose groves, a dark crown on a short stem with a blue shadow beside it,
    and spinifex in low yellow mounds, all of it smaller and closer together towards the range."""
    crowns, tops, shadows, mounds, lee = [], [], [], [], []
    groves = [(r.uniform(X0, X1), r.uniform(0.02, 0.4) ** 1.3) for _ in range(16)]
    for k in range(170):
        gx, gf = groves[k % len(groves)]
        f = float(np.clip(gf + r.normal(0, 0.03), 0.01, 0.5))
        x = float(np.clip(gx + r.normal(0, 60 + 400 * f), X0 + 10, X1 - 10))
        y = foot[int(x)] + f * (Y1 - foot[int(x)])
        if clear[int(y), int(x)] < 0.5:
            continue
        s = 7 + 80 * f
        n = int(5 + s / 2.5)
        q = r.uniform(-1, 1, (n, 2)) * (0.55 * s, 0.3 * s)
        cy = y - s * 0.7
        L = r.uniform(0.18, 0.3, n) * s + 3
        rows = np.stack([x + q[:, 0], cy + q[:, 1], L, L * 0.6, np.pi / 2 + r.normal(0, 0.4, n)], 1)
        crowns.append(rows)
        tops.append(rows[q[:, 1] < -0.08 * s][::2])
        for dx in (-0.07, 0.06):
            box.stroke("stem", [(x + dx * s, y, 0.4), (x + dx * 2 * s, y - s * 0.3, 0.8), (x + dx * 2.6 * s, cy, 0.2)],
                       max(0.9, s / 34), load=1.0)
        shadows.append([(x + s * 0.1 + i * s * 0.2, y + r.uniform(-1, 2), s * 0.32, s * 0.14, 0.0) for i in range(3)])
    patch = noise.smoothstep(-0.2, 0.9, noise.field((H, W), 150, r))
    for _ in range(1100):
        f = r.uniform(0, 1) ** 1.5 * 0.55 + 0.01
        x = r.uniform(X0 + 10, X1 - 10)
        y = foot[int(x)] + f * (Y1 - foot[int(x)])
        if clear[int(y), int(x)] < 0.5 or r.uniform(0, 1) > patch[int(y), int(x)]:
            continue
        s = 3 + 40 * f
        mounds.append((x - s * 0.8, y - s * 0.25, s * 1.6, s * 0.7, 0.0))
        lee.append((x - s * 0.2, y + s * 0.1, s * 1.3, s * 0.35, 0.05))
    box.touch(np.vstack(shadows), 0.4, **VIOLET)
    box.touch(np.array(lee), 0.3, **VIOLET)
    box.touch(np.array(mounds), 0.5, gamboge=0.2, yellow_ochre=0.25, viridian=0.12)
    box.lay(box.take("stem") * 0.7, burnt_sienna=0.5, indigo=0.5)
    box.touch(np.vstack(crowns), 0.6, indigo=0.32, viridian=0.25, raw_sienna=0.3)
    box.touch(np.vstack(tops), 0.35, viridian=0.3, gamboge=0.3, raw_sienna=0.2)


def gums(box, r, C):
    """River gums along the far reach of the creek, where its sand is lost among them: dark crowns
    on pale stems, too far off for more."""
    crowns, stems = [], []
    for i in np.linspace(len(C) * 0.45, len(C) * 0.93, 9).astype(int) + r.integers(-20, 20, 9):
        (x, y), w = C[i, :2], C[i, 2]
        for side in (-1, 1):
            if r.uniform() < (0.75 if side < 0 else 0.3):      # most of them stand on the outer bank
                continue
            f = np.clip((y - Y0 - 0.5 * SH) / (0.3 * SH), 0, 1)
            s = (26 + 80 * f) * r.uniform(0.6, 1.3)
            bx, by = x + side * (w + s * 0.3) + r.normal(0, 8), y + r.normal(0, 4)
            n = int(10 + s / 2)
            q = r.uniform(-1, 1, (n, 2)) * (0.6 * s, 0.35 * s)
            L = r.uniform(0.15, 0.28, n) * s + 3
            crowns.append(np.stack([bx + q[:, 0], by - s * 0.75 + q[:, 1], L, L * 0.6, np.pi / 2 + r.normal(0, 0.4, n)], 1))
            stems.append([(bx, by, 0.5), (bx + r.normal(0, 3), by - s * 0.4, 0.7), (bx + r.normal(0, 5), by - s * 0.65, 0.2)])
            box.stroke("gum", stems[-1], max(1.2, s / 30), load=1.0)
    box.lay(box.take("gum") * 0.3, cobalt=0.4, rose=0.3)
    box.touch(np.vstack(crowns), 0.65, indigo=0.35, viridian=0.32, raw_sienna=0.2)


def bed(r):
    """The creek's course, a dense centreline (x, y, half-width) px, and the sand of its bed: a
    ribbon lying on the ground, foreshortened the more the further off it is."""
    C = brush.path(P(CREEK), 2.0)
    C[:, 2] *= 1 + 0.14 * noise.line1d(len(C), 150, r) + 0.07 * noise.line1d(len(C), 25, r)   # pools and narrows
    im = Image.new("F", (W, H), 0.0)
    dr = ImageDraw.Draw(im)
    for x, y, w in C:
        k = 0.1 + 0.42 * np.clip((y - Y0 - 0.5 * SH) / (0.5 * SH), 0, 1) ** 1.5
        dr.ellipse([x - w, y - w * k, x + w, y + w * k], fill=1.0)
    return C, ndimage.gaussian_filter(np.asarray(im), 1.0)


def creek(box, r, C, sand):
    """The dry bed of the creek: pale sand left nearly bare, damp streaks drawn out along the flow, a
    dry brush dragged across it so the hills of the paper catch the colour; the cut bank on the far
    side, standing higher here and lower there, with a dark lip, and the near bank's shadow lying on
    the sand; a few sticks the last flood left."""
    ys, xs = np.nonzero(sand > 0.02)
    sl = (slice(ys.min() - 40, ys.max() + 2), slice(xs.min() - 40, xs.max() + 40))
    s = sand[sl]
    g = np.clip((YY[sl] - Y0 - 0.5 * SH) / (0.5 * SH), 0, 1)
    _, dep = box.wash(sand, 2.0, pool=0.15, rim=0.35, grain=0.2, scale=200)
    box.lay(dep * 0.16, yellow_ochre=0.7, raw_sienna=0.15, rose=0.15)
    full = np.zeros((H, W), np.float32)
    full[sl] = noise.smoothstep(0.7, 1.5, noise.stretched(s.shape, 150, 26, r)) * s
    _, dep = box.wash(full, 1.0, rim=0.5, grain=0.3, scale=80)
    box.lay(dep * 0.12, raw_sienna=0.5, burnt_sienna=0.2, cobalt=0.1)
    high = (0.2 + noise.smoothstep(-0.9, 0.9, noise.line1d(W, 90, r)))[sl[1]][None, :]
    deep = (0.15 + noise.smoothstep(-0.9, 0.9, noise.line1d(W, 70, r)))[sl[1]][None, :]
    bank, lee = np.zeros_like(s), np.zeros_like(s)
    for k in range(1, 40):
        bank = np.maximum(bank, np.roll(s, -k, axis=0) * (k < (2 + 26 * g) * high))
        lee = np.maximum(lee, np.roll(np.roll(1 - s, k, axis=0), -k // 3, axis=1) * (k < (3 + 28 * g) * deep))
    face = np.clip(bank - s, 0, 1)
    full[sl] = face
    _, dep = box.wash(full, 1.2, rim=0.8, grain=0.4, scale=60)
    box.lay(dep * 0.5, burnt_sienna=0.45, light_red=0.35, ultramarine=0.1)
    full[sl] = face * (1 - np.roll(face + s, -3, axis=0).clip(0, 1))       # the lip, where the bank's top breaks off
    _, dep = box.wash(full, 0.8, rim=0.9, grain=0.4, scale=40)
    box.lay(dep * 0.55, burnt_sienna=0.5, ultramarine=0.3, indigo=0.1)
    full[sl] = s * lee
    _, dep = box.wash(full, 1.0, rim=0.7, grain=0.6, scale=60)
    box.lay(dep * 0.42, **VIOLET)
    for k in range(30):
        f = r.uniform(0.05, 1)
        i = int((1 - f) * (len(C) - 1) * 0.85)
        (x, y), w = C[i, :2], C[i, 2]
        L = w * r.uniform(0.5, 1.2)
        box.stroke("drag", [(x - L, y + r.normal(0, w * 0.3), 0.6), (x, y + r.normal(0, w * 0.3), 0.8),
                            (x + L, y + r.normal(0, w * 0.3), 0.5)], 3 + 12 * f, load=0.35, dryness=2.2, reach=L, clumps=7)
    box.lay(box.take("drag", dry=0.5) * sand * 0.5, raw_sienna=0.6, burnt_sienna=0.3)
    ys, xs = np.nonzero(sand > 0.9)
    k = r.integers(0, len(ys), 120)
    x, y = xs[k].astype(np.float64), ys[k].astype(np.float64)
    sz = (2.5 + 14 * np.clip((y - Y0 - 0.5 * SH) / (0.5 * SH), 0, 1)) * r.uniform(0.5, 1.3, len(k))
    box.touch(np.stack([x - sz * 0.1, y + sz * 0.2, sz * 1.3, sz * 0.4, np.zeros(len(k))], 1), 0.35, **VIOLET)
    box.touch(np.stack([x - sz * 0.5, y - sz * 0.1, sz, sz * 0.6, np.full(len(k), 0.2)], 1), 0.4,
              burnt_sienna=0.5, light_red=0.3)
    for i in r.integers(0, int(len(C) * 0.45), 5):
        (x, y), w = C[i, :2], C[i, 2]
        x, y, L, a = x + r.normal(0, w * 0.3), y + r.normal(0, w * 0.1), w * r.uniform(0.25, 0.5), r.normal(0, 0.35)
        box.stroke("stick", [(x, y, 0.5), (x + L * np.cos(a) / 2, y + L * np.sin(a) / 2 + 2, 0.8),
                             (x + L * np.cos(a), y + L * np.sin(a), 0.2)], 2 + w / 120, load=1.0)
    box.lay(box.take("stick") * 0.8, burnt_sienna=0.5, indigo=0.45)


def domes(r):
    """Where the spinifex stands: each hummock's place, size and dome, and all the domes together,
    which the painter keeps clear of the ground wash."""
    out, union = [], np.zeros((H, W), np.float32)
    for u, v, s in HUMMOCKS:
        (cx, cy, w), = P([(u, v, s)])
        h = w * 0.42
        sl = near_box(cx, cy, w * 1.2, h * 2.2)
        dx, dy = (XX[sl] - cx) / (w * 0.5), (YY[sl] - cy) / h
        rim = 1 + 0.1 * noise.field(dx.shape, 9, r)
        dome = noise.smoothstep(rim, rim - 0.08, np.hypot(dx, np.where(dy < 0, dy, 4 * dy)))
        union[sl] = np.maximum(union[sl], dome)
        out.append((cx, cy, w, h, sl, dx, dy, dome))
    return out, union


def hummocks(box, r, spinifex):
    """Spinifex: low domes of stiff needles. A straw wash for the dome, olive on its shaded side, then
    needles flicked up and out from the base with a nearly dry brush, and the blue shadow beside it."""
    for cx, cy, w, h, sl, dx, dy, dome in spinifex:
        shaded = dome * noise.smoothstep(-0.1, 0.35, 0.8 * dx + 0.6 * dy + 0.15 * noise.field(dx.shape, 14, r))
        cast = noise.smoothstep(1.0, 0.9, np.hypot((dx - 0.55) / 1.15, (dy - 0.05) / 0.2)) * (1 - dome)
        for mask, dens, mix in ((dome, 0.45, dict(gamboge=0.3, yellow_ochre=0.32, viridian=0.08)),
                                (shaded, 0.42, dict(viridian=0.3, raw_sienna=0.32, ultramarine=0.08)),
                                (cast, 0.36, VIOLET)):
            full = np.zeros((H, W), np.float32)
            full[sl] = mask
            _, dep = box.wash(full, 0.7, rim=0.6, grain=0.5, scale=50)
            box.lay(dep * dens, **mix)
        for k in range(int(16 + w / 8)):
            f = r.uniform(-1, 1)
            x0, y0 = cx + f * w * 0.42, cy - r.uniform(0, 0.2) * h
            a = -np.pi / 2 + 1.1 * f + r.normal(0, 0.2)
            L = h * r.uniform(0.5, 1.15) * np.sqrt(1 - 0.5 * f * f)
            x1, y1 = x0 + np.cos(a) * L, y0 + np.sin(a) * L
            key = "spin_sun" if f + r.normal(0, 0.35) < 0.2 else "spin_shade"
            box.stroke(key, [(x0, y0, 0.8), ((x0 + x1) / 2 + r.normal(0, 2), (y0 + y1) / 2, 0.5), (x1, y1, 0.03)],
                       r.uniform(1.6, 3.0) * (w / 200) ** 0.5, load=0.7, dryness=1.3, reach=L * 1.5, clumps=3)
    box.lay(box.take("spin_sun", dry=0.8) * 0.7, yellow_ochre=0.45, gamboge=0.3, raw_sienna=0.15)
    box.lay(box.take("spin_shade", dry=0.8) * 0.7, raw_sienna=0.35, viridian=0.32, ultramarine=0.12)


def near_ground(box, r, clear):
    """The near ground: a nearly dry brush dragged across it again and again, so the red earth catches
    on the hills of the paper and the hollows stay pale like gravel in the sun; and tufts of dry grass
    flicked up along the bank and round the foot of the gum."""
    for _ in range(48):
        f = r.uniform(0, 1) ** 0.6
        y, x, L, a = Y0 + SH * (0.7 + 0.3 * f), r.uniform(X0 - 150, X1), r.uniform(150, 520), r.normal(0, 0.05)
        box.stroke("gravel", [(x, y, 0.5), (x + L / 2, y + a * L / 2 + r.normal(0, 6), 0.8), (x + L, y + a * L + r.normal(0, 8), 0.4)],
                   6 + 16 * f, load=0.4, dryness=2.0, reach=L, clumps=8)
    box.lay(box.take("gravel", dry=0.45) * clear * 0.55, burnt_sienna=0.5, light_red=0.4)
    (bx, by), = P([(0.262, 0.917)])[:, :2]
    tufts = [(bx + r.normal(0, 60), by + r.normal(6, 10)) for _ in range(9)] + \
            [tuple(p) for p in P([(r.uniform(0.0, 1.0), r.uniform(0.72, 1.0)) for _ in range(40)])]
    for x, y in tufts:
        s = (0.4 + 0.6 * (y - Y0) / SH) ** 2
        for _ in range(r.integers(5, 10)):
            a = -np.pi / 2 + r.normal(0, 0.45)
            L = r.uniform(18, 48) * s
            box.stroke("grass", [(x, y, 0.7), (x + np.cos(a) * L * 0.5, y + np.sin(a) * L * 0.5, 0.4),
                                 (x + np.cos(a) * L + r.normal(0, 3), y + np.sin(a) * L, 0.02)], 1.4 + 1.2 * s,
                       load=0.8, dryness=1.2, reach=L * 1.4, clumps=2)
    box.lay(box.take("grass", dry=0.85) * 0.75, yellow_ochre=0.5, raw_sienna=0.35, viridian=0.08)


def rocks(box, r):
    """Boulders of the range's red quartzite: a warm wash, the shaded facets violet, a dark crack."""
    for u, v, s in ROCKS:
        (cx, cy, w), = P([(u, v, s)])
        a = np.sort(r.uniform(0, 2 * np.pi, 9))
        rad = w * r.uniform(0.6, 1.0, 9)
        pts = np.stack([cx + rad * np.cos(a), cy + 0.55 * rad * np.sin(a)], 1)
        pts[:, 1] = np.minimum(pts[:, 1], cy + 0.25 * w)
        sl = near_box(cx, cy, 2 * w, 1.5 * w)
        im = Image.new("F", (W, H), 0.0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1.0)
        m = ndimage.gaussian_filter(np.asarray(im), 0.8)
        side = np.zeros((H, W), np.float32)
        side[sl] = (XX[sl] - cx) * 0.8 + (YY[sl] - cy) * 0.6 + 0.15 * w * noise.field(XX[sl].shape, 12, r)
        cast = np.clip(ndimage.shift(m, (-0.08 * w, 0.4 * w), order=1) - m, 0, 1)
        for mask, dens, mix in ((m, 0.55, dict(light_red=0.5, yellow_ochre=0.4, burnt_sienna=0.1)),
                                (m * (side > 0.05 * w), 0.6, dict(ultramarine=0.35, rose=0.3, burnt_sienna=0.3)),
                                (cast, 0.42, VIOLET)):
            _, dep = box.wash(mask, 0.6, rim=0.8, grain=0.5, scale=40)
            box.lay(dep * dens, **mix)
        box.stroke("crack", [(cx - 0.3 * w, cy - 0.1 * w, 0.2), (cx, cy + 0.05 * w, 0.8), (cx + 0.15 * w, cy + 0.2 * w, 0.3)],
                   max(1.2, w / 40))
    box.lay(box.take("crack") * 0.8, burnt_sienna=0.5, indigo=0.5)


def shadow(box, r, spines):
    """The gum's shadow thrown over the bank and the creek: the boughs and the clumps laid flat on
    the ground away from the sun, a violet wash with crisp edges, broken where the leaves are thin."""
    bx, by = P(TREE[0][:1])[0, :2]
    kx, ky = 0.22, 0.07
    im = Image.new("F", (W, H), 0.0)
    dr = ImageDraw.Draw(im)
    for C, _, _ in spines[:4]:
        for x, y, w in C[::3]:
            h = by - y
            sx, sy = x + h * kx, by - h * ky
            dr.ellipse([sx - w * 0.9, sy - w * 0.22, sx + w * 0.9, sy + w * 0.22], fill=1.0)
    rows = []
    for cx, cy, rx, ry in P(CLUMPS):
        n = int(rx * ry / 60)
        q = r.uniform(-1, 1, (n, 2))
        q = q[np.hypot(*q.T) < 1]
        h = by - (cy + q[:, 1] * ry)
        x, y = cx + q[:, 0] * rx + h * kx, by - h * ky + q[:, 1] * ry * 0.25
        rows.append(np.stack([x, y, r.uniform(12, 26, len(q)), r.uniform(4, 8, len(q)), r.normal(0, 0.3, len(q))], 1))
    m = np.clip(np.asarray(im) + wash.touches((H, W), np.vstack(rows)), 0, 1) * (YY > by - 0.12 * SH)
    _, dep = box.wash(m, 0.9, rim=0.8, grain=0.5, scale=60)
    box.lay(dep * 0.3, **VIOLET)


def drawing(sheet, r, spines, lines, sand):
    """The pencil drawing the painting went over: the boughs, the skylines, the banks of the creek."""
    press = np.zeros((H, W), np.float32)
    for S, T, Nn in spines:
        if S[:, 2].max() < 3:
            continue
        for side in (1, -1):
            pencil.line(press, (S[:, :2] + Nn * S[:, 2:3] * side)[::14], 2.2, r, pressure=0.6)
    xs = np.arange(X0, X1, 16)
    for y, p in lines:
        ok = y[xs] < Y0 + 0.52 * SH
        pencil.line(press, np.stack([xs[ok], y[xs[ok]]], 1), 2.2, r, pressure=p)
    cols = np.nonzero((sand > 0.5).any(0))[0]
    on = sand[:, cols] > 0.5
    for y in (on.argmax(0), H - 1 - on[::-1].argmax(0)):      # the far bank and the near
        pencil.line(press, np.stack([cols, y], 1)[::10], 2.2, r, pressure=0.45)
    return pencil.catch(press, sheet, grip=0.7)


def paint(seed=11):
    r = noise.rng(seed)
    sheet = wash.rough((H, W), _s(r))
    spines, twigs = tree(r)
    cover, light = boughs(spines)
    box = Box(sheet, r, frame(r), cover)
    near, foot = profile(NEAR, r), profile(FOOT, r, 3.0)
    hill = np.minimum(profile(FOOTHILL, r, 2.0), foot + 2)
    C, sand = bed(r)
    spinifex, domed = domes(r)

    far = sky(box, r, near, profile(FAR, r, 1.5))
    hm, ht, hs, hill = massif(hill, foot, r, spacing=(80, 170), relief=0.8, bulk=0.8)
    m, t, S, near = massif(near, foot, r)
    rock_face(box, r, m * (1 - hm), t, S, near, foot)
    rock_face(box, r, hm, ht, hs, hill, foot, warm=0.5, scrub=3.0, scree=-0.3)
    ground(box, r, foot, (1 - sand) * (1 - domed))
    treeline(box, r, foot)
    creek(box, r, C, sand)
    gums(box, r, C)
    plain(box, r, foot, box.keep * (1 - sand))
    near_ground(box, r, (1 - sand) * (1 - domed))
    rocks(box, r)
    hummocks(box, r, spinifex)
    shadow(box, r, spines)
    bark(box, r, cover, light, spines)
    foliage(box, r, twigs)
    box.lay(drawing(sheet, r, spines, ((near, 0.5), (far, 0.3), (hill, 0.4)), sand), bark=True, graphite=1.0)

    img = wash.cockle(box.glaze(), sheet, box.soak, r)
    return plate.mount(img, sheet)
