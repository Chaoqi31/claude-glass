"""Irises by the Plank Bridge. Ink and mineral colour on gold leaf, a six-panel folding screen.

Ogata Kōrin painted irises on gold twice over: about 1705 the pair of screens now in the Nezu Museum,
nothing but clumps of flowers rising and falling across twelve panels, and some years later the pair in
the Metropolitan Museum, where the plank bridge of the Tale of Ise, the eight bridges at Yatsuhashi,
zigzags through the marsh. This is one new screen in that manner. Four spans of planks climb from the
foot of the screen near its middle to the left edge, and nine clumps stand about them, some in front of
the planks and some behind, each with its own height, its own crowd of flowers and its own lean, so that
the flowers rise and fall across the panels with wide stretches of gold between.

The gold went on first: squares of leaf laid on the paper of each panel in rows, each lapping the one
before it, so the seams show as a faint grid. Every leaf lies at its own slight angle and takes the light
a shade differently, a few wrinkled as they went down, and handling has rubbed the leaf thin along the
panel edges and at the foot and scuffed it through to the toned paper in flecks. The bridge is a grey of
ink and shell white with more ink dropped into it wet, so that it pooled and dried with dark rims; the
posts and rails under it are dark ink, half lost among the irises. The irises are mineral colour ground
from stone and bound in glue. The leaves are malachite in two grades, a dark rokushō for those at the back
and a paler one in front, each laid in one stroke of a loaded brush: thicker down the middle than at the
edges, its tone drifting along its length as the load ran down, its edges left by the bristles, a shade
deeper where it crosses a leaf painted before it, and ending in streaks where the brush ran dry at the
foot of a clump. The flowers are azurite in four grades, each petal flat in one load, deep and pale
meeting softly where they touch. The paint is matt and sandy with its crystals and stands a little proud
of the leaf. The azurite has rubbed to a chalky blue in places and cracked into a fine net where it lay
thickest, and where the panels fold against each other it has lost a few flakes to show the gold again.
"""

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from atelier import noise, plate, tempera
from atelier.color import glaze, lin, pigment
from atelier.noise import smoothstep

TITLE = "Irises by the Plank Bridge"
DATE = "2026"
MEDIUM = "Six-panel folding screen: ink and mineral colour on gold leaf over paper"
AFTER = ("Ogata Kōrin, Irises (Kakitsubata-zu), pair of six-panel screens, c. 1701–05, Nezu Museum, Tokyo; "
         "and Irises at Yatsuhashi (Eight Bridges), pair of six-panel screens, c. 1711–14, Metropolitan Museum "
         "of Art, New York")
ROOM = "Paper and Water"
YEAR = 1705
PLACE = "Kyoto"
REGION = "East Asia"
NOTE = ("A new screen of Kōrin's irises: nine clumps in azurite and malachite rise and fall across six panels of "
        "gold leaf, and a bridge of grey planks zigzags up through them.")

H, W = 1500, 3400
M, FR, BR = 33, 15, 44                  # the wall round the screen, the lacquer frame, the brocade
X0 = Y0 = M + FR + BR                   # where the gold begins
NP, PW = 6, 536                         # six panels
X1, Y1 = X0 + NP * PW, H - Y0
FH, FW = Y1 - Y0, X1 - X0

# --- the gold ---------------------------------------------------------------------------------------------
GOLD = [lin(c) for c in ("#8e6526", "#c99d46", "#e3be6a", "#f6e09a")]   # the leaf from dull to bright
GROUND = lin("#a5804b")                 # the paper under the leaf, toned with yellow earth
LEAF = 104.0                            # one leaf, px: about eleven centimetres

# --- the bridge -------------------------------------------------------------------------------------------
SUMI = pigment("#59595c")
WOOD = pigment("#3a3530")               # the posts and rails: dark ink
PLANK = lin("#8d897d")                  # ink let down with shell white: the grey of the boards
D = np.array([-0.8, -0.6])              # depth runs up the screen and to the left
BWX, BWD, THICK = 150.0, 110.0, 20.0    # breadth of a span seen square and seen receding; its thickness
SPANS = [("depth", (2040, 1340), 760), ("across", (1610, 900), 900),        # field px
         ("depth", (700, 880), 760), ("across", (262, 440), 700)]
# under the bridge, among the irises: x, top, breadth, drop, rails (field px)
TRESTLES = [(440, 698, 104, 118, 2), (805, 920, 92, 104, 1), (1195, 920, 112, 116, 2)]

# --- the irises -------------------------------------------------------------------------------------------
AZ = {k: lin(c) for k, c in dict(deep="#141f5c", deep2="#1b2b78", mid="#2c4f9f", pale="#6189c3",
                                 turq="#4a8fc4").items()}
MAL = {k: lin(c) for k, c in dict(deep="#1c5334", pale="#327d52", sheath="#b4bf86").items()}
# x, foot, crown, breadth, flowers, buds, lean, behind the bridge (field px)
CLUMPS = [(150, 470, 20, 360, 9, 2, 0.08, True),        # behind the top span, cut by the left edge
          (380, 1340, 430, 500, 15, 3, -0.1, False),   # in front, at the foot of the climb
          (800, 1300, 960, 260, 5, 1, 0.15, False),    # a low tuft
          (1080, 910, 500, 260, 5, 1, -0.05, True),    # peeking over the long span
          (1290, 1340, 760, 400, 12, 2, 0.08, False),  # in front of the long span
          (1820, 1130, 90, 440, 16, 3, -0.06, True),   # behind the bridge's foot, rising high
          (2290, 1260, 900, 170, 4, 1, 0.14, False),   # a low tuft
          (2580, 1010, 330, 420, 13, 2, -0.1, False),  # these two make one mass rising to the right edge
          (3030, 870, 30, 480, 16, 3, 0.07, False)]


def ramp(e):
    """How bright a leaf looks (0..1) as colour: dull leaf is deeper and warmer, bright leaf paler."""
    e = np.clip(e, 0, 1) * (len(GOLD) - 1)
    i = np.minimum(e.astype(int), len(GOLD) - 2)
    f = (e - i)[..., None]
    G = np.stack(GOLD)
    return G[i] * (1 - f) + G[i + 1] * f


def leaf(r):
    """The gold ground of the field. Each panel was covered by itself, row by row from the top, every leaf
    laid over the edge of the one before it and of the row above. A leaf's own edge shows as a hairline,
    and the strip where two leaves lie double takes the light a little differently. Every leaf lies at its
    own slight angle, so its brightness and the way it falls off across it are its own; a few wrinkled as
    they went down. Hands have rubbed the leaf thin along the panel edges and at the foot, and scuffed it
    through to the toned paper in flecks. -> rgb (FH, FW, 3), relief (FH, FW)"""
    h, w = FH, FW
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wy = yy + 0.8 * noise.field((h, w), 50, r) + 0.3 * noise.field((h, w), 5, r)
    wx = xx + 0.8 * noise.field((h, w), 50, r) + 0.3 * noise.field((h, w), 5, r)
    e = np.zeros((h, w), np.float32)
    hue = np.zeros((h, w), np.float32)
    nrow = int(h / LEAF) + 3
    crease = Image.new("F", (w, h), 0.0)
    cd = ImageDraw.Draw(crease)
    for p in range(NP):
        sl = np.s_[:, p * PW:(p + 1) * PW]
        y, x = wy[sl], wx[sl] - p * PW
        B = -r.uniform(0, LEAF) + np.concatenate([[0], np.cumsum(LEAF + r.normal(0, 1.5, nrow))])
        k = np.clip(np.searchsorted(B, y, side="right") - 1, 0, nrow - 1)
        top = y - B[k]
        size = LEAF + r.normal(0, 1.2, nrow)
        off = r.uniform(0, LEAF, nrow)
        q = (x + off[k]) / size[k]
        j = np.floor(q).astype(int)
        left = (q - j) * size[k]
        idx = k * 12 + np.clip(j + 1, 0, 11)
        n = nrow * 12
        tone, tint, gx, gy, lap = r.normal(0, 1, (5, n))
        u, v = left / size[k] - 0.5, top / LEAF - 0.5
        e[sl] = (0.55 + 0.022 * np.clip(tone[idx], -2, 2) + 0.012 * (gx[idx] * u + gy[idx] * v)
                 + r.normal(0, 0.012)
                 - 0.08 * (np.exp(-(top / 0.7) ** 2) + np.exp(-(left / 0.7) ** 2))
                 + 0.05 * (np.exp(-((top - 1.6) / 0.7) ** 2) + np.exp(-((left - 1.6) / 0.7) ** 2))
                 + 0.02 * lap[idx] * smoothstep(5.0, 3.5, np.minimum(top, left)))
        hue[sl] = tint[idx]
        for kk in range(nrow):          # a few leaves folded as they went down: a wrinkle or three across them
            for jj in range(-1, 7):
                if r.random() > 0.14:
                    continue
                xa, ya = p * PW + jj * size[kk] - off[kk], B[kk]
                th0 = r.uniform(0, np.pi)
                for _ in range(r.integers(1, 4)):
                    cx, cy = xa + r.uniform(0.1, 0.9) * size[kk], ya + r.uniform(0.1, 0.9) * LEAF
                    pts, th = [(cx, cy)], th0 + r.normal(0, 0.25)
                    for _ in range(r.integers(4, 10)):
                        th += r.normal(0, 0.18)
                        cx, cy = cx + 6 * np.cos(th), cy + 6 * np.sin(th)
                        pts.append((cx, cy))
                    cd.line(pts, fill=float(r.choice([-1, 1]) * r.uniform(0.5, 1.0)), width=1)
    for _ in range(220):            # fine scratches
        x0, y0 = r.uniform(0, w), r.uniform(0, h)
        th, L = r.normal(0, 0.5) + r.choice([0, np.pi / 2], p=[0.7, 0.3]), r.uniform(15, 140)
        cd.line([(x0, y0), (x0 + L * np.cos(th), y0 + L * np.sin(th))], fill=float(r.uniform(-0.4, 0.5)), width=1)
    cr = ndimage.gaussian_filter(np.asarray(crease), 0.7)
    gy_, gx_ = np.gradient(cr)
    e += 0.9 * (0.6 * gx_ + 0.8 * gy_) - 0.03 * np.abs(cr)
    e += 0.018 * noise.field((h, w), 5, r) + 0.01 * noise.field((h, w), 1.5, r)
    # the light of the room lies over the whole in broad soft sheens, and each panel tilts a little its own way
    e += 0.09 * noise.fbm((h, w), 650, r, octaves=3) - 0.03 * (xx / w - 0.5)
    e += 0.025 * r.normal(0, 1, NP)[np.minimum((xx // PW).astype(int), NP - 1)]
    # handling: hands at the foot and along the edges of the panels rubbed the leaf thin in soft broad
    # zones, and scuffed it through to the paper in small flecks and short strokes
    jd = np.abs(((xx + PW / 2) % PW) - PW / 2)          # how far from a panel edge
    rub = (0.45 * noise.fbm((h, w), 300, r, octaves=3) + 0.9 * smoothstep(h - 280, h, yy)
           + 0.5 * np.exp(-jd / 26) + 0.25 * smoothstep(120, 0, yy))
    thin = smoothstep(0.55, 1.25, rub)
    e -= 0.07 * thin
    scuff = Image.new("F", (w, h), 0.0)
    sd = ImageDraw.Draw(scuff)
    for _ in range(550):
        x0, y0 = r.uniform(0, w), h * r.random() ** 0.45
        if r.random() > 0.15 + 0.85 * float(thin[int(min(y0, h - 1)), int(x0)]):
            continue
        th, L = r.normal(0, 0.35) + r.choice([0, np.pi / 2]), r.lognormal(1.6, 0.8)
        sd.line([(x0, y0), (x0 + L * np.cos(th), y0 + L * np.sin(th))], fill=1.0, width=int(r.integers(1, 3)))
    bare = smoothstep(0.3, 0.8, ndimage.gaussian_filter(np.asarray(scuff), 0.7) + 0.25 * noise.field((h, w), 1.5, r))
    bare = np.maximum(bare, smoothstep(3.3, 3.8, noise.field((h, w), 1.1, r)))     # pinholes in the leaf
    tint = 0.4 * np.clip(hue, -2, 2) + 1.2 * thin
    col = ramp(e) * (1 + tint[..., None] * np.array([0.035, 0.0, -0.1], np.float32))
    ground = GROUND * (1 + 0.06 * noise.field((h, w), 14, r) + 0.03 * noise.field((h, w), 2, r))[..., None]
    col = col * (1 - bare[..., None]) + ground * bare[..., None]
    return col.astype(np.float32), (0.4 * cr).astype(np.float32)


# --- shapes -----------------------------------------------------------------------------------------------

def axis(c, ang, length, turn, n=40):
    """A centre line from c, setting off at `ang` (radians, picture coordinates, y down) and turning by
    `turn` over its length, faster toward the end. -> points (n, 2), t (n,)"""
    t = np.linspace(0, 1, n)
    a = ang + turn * t ** 1.4
    d = np.stack([np.cos(a), np.sin(a)], 1)
    return np.asarray(c, float) + np.vstack([[0, 0], np.cumsum(d[:-1] * length / (n - 1), 0)]), t


def ribbon(pts, hwl, hwr):
    """The outline round a centre line with a half-width on either side. -> polygon, normals"""
    t = np.gradient(pts, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return np.vstack([pts + nrm * hwl[:, None], (pts - nrm * hwr[:, None])[::-1]]), nrm


def raster(poly, ss=3, pad=4):
    """A polygon as coverage in a box round it. -> coverage, (y0, y1, x0, x1)"""
    x0, y0 = np.floor(poly.min(0)).astype(int) - pad
    x1, y1 = np.ceil(poly.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    im = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    ImageDraw.Draw(im).polygon([((x - x0) * ss, (y - y0) * ss) for x, y in poly], fill=255)
    a = np.asarray(im, np.float32).reshape(y1 - y0, ss, x1 - x0, ss).mean((1, 3)) / 255
    return a, (y0, y1, x0, x1)


def lay(P, pts, hwl, hwr, col, r, thick=1.0, coarse=1.0, streak=0.06, ridge=0.2, double=0.0, soft=1.4,
        dry=0.0, hide=None, rough=0.14):
    """One shape in one load of the brush: mineral colour laid along a centre line. `col` is a colour, or a
    function giving one from where a point lies along the stroke (T, 0 at its start to 1 at its end) and
    across it (U, -1 to 1). The bristles leave lanes along the stroke: inside they show as faint streaks of
    tone, at the edge they run on a hair past it or stop short, so the edge is brushed and never cut. Where
    the stroke goes over paint already down the two meet softly over a pixel or two, and where it lies
    double it is `double` the deeper. `ridge` lays it thinner at the edges than down the middle. A leaf
    whose foot the brush ran dry on ends in streaks. Paint goes over whatever is there already."""
    poly, nrm = ribbon(pts, hwl, hwr)
    got = raster(poly)
    if got is None:
        return
    a, (y0, y1, x0, x1) = got
    sl = np.s_[y0:y1, x0:x1]
    b = ndimage.gaussian_filter(a, 0.9)
    ys, xs = np.nonzero(b > 0.002)
    if not len(ys):
        return
    q = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], 1)
    _, i = cKDTree(pts).query(q)
    off = ((q - pts[i]) * nrm[i]).sum(1)
    run = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])     # px along the stroke
    along = np.clip(run[i] + ((q - pts[i]) * nrm[i][:, ::-1] * [1, -1]).sum(1), 0, run[-1])
    half = int(max(hwl.max(), hwr.max())) + 6
    lanes = noise.stretched((int(run[-1]) + 10, 2 * half), 2.2, 48, P.rt)        # rows run along the stroke
    T, U, B = np.zeros((3,) + a.shape, np.float32)
    T[ys, xs] = along / max(run[-1], 1e-6)
    U[ys, xs] = np.clip(np.where(off > 0, off / np.maximum(hwl[i], 0.6), off / np.maximum(hwr[i], 0.6)), -1.3, 1.3)
    B[ys, xs] = ndimage.map_coordinates(lanes, [along + 4, np.clip(off + half, 0, 2 * half - 1)], order=1)
    a = smoothstep(0.3, 0.7, b + (rough * P.edge[sl] + 0.25 * B) * 4 * b * (1 - b))
    if dry:
        grid = np.linspace(-1.3, 1.3, 56)
        start = dry * r.uniform(0.3, 1.2) * (0.1 + np.abs(np.interp(U, grid, noise.line1d(56, 4.0, r))))
        a = a * smoothstep(start, start + 9, T * run[-1] + 6 * P.edge[sl] + 7 * B)     # each bristle its own
    if hide is not None:
        a = a * (1 - hide[sl])
    old = P.a[sl]
    c = col(T, U) if callable(col) else col[None, None, :]
    c = c * ((1 + streak * B) * (1 - double * old))[..., None]
    w = (old * ndimage.gaussian_filter(a, soft) + (1 - old) * (a > 0.002))[..., None]
    P.rgb[sl] = P.rgb[sl] * (1 - w) + c * w        # the colour, not yet weighted by how much paint is there
    P.a[sl] = 1 - (1 - old) * (1 - a)
    P.h[sl] = P.h[sl] * (1 - 0.5 * a) + thick * (1 - ridge * np.clip(U, -1, 1) ** 2) * a
    P.k[sl] = P.k[sl] * (1 - a) + coarse * a


def blade(base, ang, length, hw0, bend, kink, r):
    """An iris leaf: a sword, broad for most of its length and drawn out to a point, rising at `ang` from
    upright and curving by `bend`; a leaf that has snapped over turns sharply at its kink, narrows where
    it twists, and hangs on from there. -> centre line, half-widths"""
    n = max(24, int(length / 3))
    t = np.linspace(0, 1, n)
    a = ang + bend * t
    if kink:
        tk, dk = kink
        a = a + dk * smoothstep(tk - 0.012, tk + 0.012, t) + 0.5 * dk * np.clip(t - tk, 0, None)
    d = np.stack([np.sin(a), -np.cos(a)], 1)
    pts = np.asarray(base, float) + np.vstack([[0, 0], np.cumsum(d[:-1] * length / (n - 1), 0)])
    hw = hw0 * np.clip(1 - t, 0, 1) ** 0.62 * (1 + 0.05 * noise.line1d(n, n / 4, r))
    if kink:
        hw *= 1 - 0.45 * np.exp(-((t - kink[0]) / 0.025) ** 2)
    return pts, hw


def fall_hw(t, r):
    """A fall: a narrow claw opening into a broad rounded blade, its margin gently waved."""
    tip = np.where(t > 0.58, np.sqrt(np.clip(1 - ((t - 0.58) / 0.42) ** 2, 0, 1)), 1.0)
    return (0.15 + 0.85 * smoothstep(0.08, 0.55, t)) * tip * (1 + 0.07 * np.sin(2 * np.pi * (2.4 * t + r.uniform())))


def standard_hw(t, r):
    """A standard: narrow and upright, widening a little to a spoon-shaped tip."""
    tip = np.where(t > 0.74, np.sqrt(np.clip(1 - ((t - 0.74) / 0.26) ** 2, 0, 1)), 1.0)
    return (0.38 + 0.62 * smoothstep(0.4, 0.8, t)) * tip * (1 + 0.05 * np.sin(2 * np.pi * (1.7 * t + r.uniform())))


def bloom(P, c, s, face, tilt, r, hide):
    """One flower of the kakitsubata: standards upright at the heart, falls hanging from it, two to the sides
    and one toward us, all turned by how the flower faces (`face`, -1 left .. 1 right) and leans (`tilt`). A
    flower only just open holds its falls up and close; seen from the side, the far fall is hidden. Each
    petal is one shape in one grade of azurite, deeper at the heart, and meets its neighbours without a
    line."""
    c = np.asarray(c, float)
    scheme = [dict(std="mid", side="deep", front="deep2"), dict(std="deep", side="mid", front="deep"),
              dict(std="mid", side="deep", front="turq"), dict(std="pale", side="deep2", front="deep"),
              dict(std="deep2", side="deep", front="mid"), dict(std="deep", side="deep2", front="pale")][
        r.choice(6, p=[0.32, 0.12, 0.06, 0.1, 0.32, 0.08])]
    grade = lambda k: AZ[k] * r.uniform(0.88, 1.1)
    depth = lambda T, U: 0.8 + 0.28 * smoothstep(0.0, 0.55, T) + 0.04 * U
    opened = r.uniform(0.55, 1.0)
    side = abs(face) > 0.55
    ns = r.choice([1, 2, 3], p=[0.2, 0.45, 0.35])
    for k in range(ns):
        da = (k - (ns - 1) / 2) * 0.48 * (0.6 + 0.4 * opened) * r.uniform(0.7, 1.3) + r.normal(0, 0.08)
        pts, t = axis(c + [0, -0.05 * s], -np.pi / 2 + tilt - 0.15 * face + da,
                      s * r.uniform(0.6, 1.0) * (1 - 0.2 * abs(da)), 0.35 * np.sign(da) * r.uniform(0.2, 1.0))
        hw = 0.1 * s * r.uniform(0.8, 1.25) * standard_hw(t, r)
        lay(P, pts, hw, hw, grade(scheme["std"]), r, thick=1.0, streak=0.07, hide=hide)
    falls = []
    for sgn in (-1, 1):                 # the left fall, the right fall
        back = max(0.0, -sgn * face)    # how far this one is turned away from us
        if side and back > 0.55 and r.random() < 0.5:
            continue
        ang = (np.pi / 2 - sgn * (np.pi / 2 - 0.62)) + tilt + 0.25 * face - sgn * 0.4 * (1 - opened)
        L = (s * r.uniform(0.82, 1.12) * (1 - 0.32 * back) * (1.15 if side and back == 0 else 1)
             * (0.72 + 0.28 * opened))
        turn = sgn * r.uniform(0.5, 1.2) * (1 - 0.4 * back) * (1.3 if side and back == 0 else 1) * opened
        falls.append((back, ang, L, turn, 0.3 * s * r.uniform(0.85, 1.15) * (1 - 0.15 * back) * (0.8 + 0.2 * opened),
                      scheme["side"]))
    if not side and r.random() < 0.85:
        falls.append((-1, np.pi / 2 + tilt + 0.55 * face, 0.8 * s * r.uniform(0.85, 1.1) * (1 - 0.15 * abs(face))
                      * (0.7 + 0.3 * opened), r.normal(0, 0.18), 0.34 * s * r.uniform(0.85, 1.1), scheme["front"]))
    for back, ang, L, turn, wd, g in sorted(falls, key=lambda f: -f[0]):
        pts, t = axis(c, ang, L, turn)
        hw = wd * fall_hw(t, r)
        lean = r.uniform(-0.18, 0.18) + 0.12 * np.sign(turn)
        lay(P, pts, hw * (1 + lean), hw * (1 - lean), lambda T, U, c=grade(g): c * depth(T, U)[..., None], r,
            thick=1.15, streak=0.1, hide=hide)


def bud(P, c, s, ang, r, hide):
    """A bud still furled: a spindle of deep azurite, a paler line where the petals lap."""
    pts, t = axis(c, -np.pi / 2 + ang, s * r.uniform(0.85, 1.25), r.normal(0, 0.15))
    hw = 0.15 * s * np.sin(np.pi * np.clip(t, 0, 1) ** 0.72) ** 0.85
    lay(P, pts, hw, hw, AZ["deep"] * r.uniform(0.9, 1.1), r, thick=1.1, hide=hide)
    side = r.choice([-1, 1])
    lay(P, pts + side * 0.35 * hw[:, None] * np.array([1, 0]), 0.35 * hw, 0.35 * hw,
        AZ[r.choice(["mid", "pale"])], r, thick=1.15, soft=1.0, hide=hide)


def sheath(P, c, s, r, hide):
    """The spathe that held the flower: two pale bracts in a V below it."""
    c = np.asarray(c, float)
    for sgn in (-1, 1):
        pts, t = axis(c + [0, 0.55 * s], -np.pi / 2 + sgn * r.uniform(0.2, 0.42), s * r.uniform(0.45, 0.65),
                      -sgn * 0.2)
        hw = 0.075 * s * (1 - t) ** 0.7 * smoothstep(0, 0.12, t) + 0.6
        lay(P, pts, hw, hw, MAL["sheath"] * r.uniform(0.92, 1.08), r, thick=0.8, coarse=0.0, hide=hide)


def clump(P, g, r, hide):
    """A clump: leaves rising from a ragged foot in a fan, back leaves in the dark grade and the rest pale,
    the flowers on their stalks among the tips at heights of their own, a few leaves crossing in front.
    Where the foot of the clump stands clear of the bridge and the screen's edge, the brush ran dry on it."""
    x, foot, crown, breadth, nb, nbud, lean, behind = g
    x, foot, crown = x + X0, foot + Y0, crown + Y0
    tall = foot - crown
    hide = hide if behind else None
    leaves = []
    for _ in range(int(breadth / 9) + r.integers(-2, 3)):
        f = np.clip(r.normal(0, 0.42), -1, 1)
        length = tall * r.uniform(0.55, 1.02) * (1 - 0.25 * abs(f))
        kink = (r.uniform(0.4, 0.8), np.sign(f + r.normal(0, 0.3)) * r.uniform(0.5, 1.4)) if r.random() < 0.22 else None
        leaves.append(dict(base=(x + 0.5 * breadth * f, foot + r.normal(0, 26)), ang=lean + 0.2 * f + r.normal(0, 0.06),
                           length=length, hw0=r.uniform(13, 18.5), bend=r.normal(0, 0.2) + 0.08 * f, kink=kink,
                           depth=r.choice(3, p=[0.35, 0.45, 0.2]), skew=r.normal(0, 0.08)))
    for _ in range(r.integers(1, 4)):   # a few long leaves arching out of the clump and over
        f = r.choice([-1, 1]) * r.uniform(0.3, 0.7)
        leaves.append(dict(base=(x + 0.5 * breadth * f, foot + r.normal(0, 20)), ang=lean + np.sign(f) * r.uniform(0.25, 0.5),
                           length=tall * r.uniform(0.65, 0.9), hw0=r.uniform(12, 16), bend=np.sign(f) * r.uniform(0.5, 1.0),
                           kink=None, depth=r.choice(3, p=[0.2, 0.5, 0.3]), skew=r.normal(0, 0.08)))
    dry = 0.0 if foot > Y1 else 70.0

    def leaf_(lv, grade, short=1.0):
        """One sword leaf in one stroke. As the load ran down the tone drifts along it, paler toward the tip
        where the brush lifted; the paint lies thinner and a little paler at the edges than down the middle,
        and the side turned from the light is a shade deeper."""
        pts, hw = blade(lv["base"], lv["ang"], lv["length"] * short, lv["hw0"], lv["bend"], lv["kink"], r)
        side = r.choice([-1, 1])
        base = grade * r.uniform(0.9, 1.1) * (1 + r.normal(0, 0.03) * np.array([1.0, 0.0, -1.0]))
        knots = np.linspace(0, 1, 32)
        load, hue = noise.line1d(32, 9, P.rt), noise.line1d(32, 12, P.rt)

        def col(T, U):
            k, h = np.interp(T, knots, load), np.interp(T, knots, hue)
            tone = (1 + 0.08 * k + 0.12 * (T - 0.5)) * (1 + 0.18 * np.clip(U, -1, 1) ** 2 + 0.05 * side * U)
            return base * tone[..., None] * (1 + 0.05 * h[..., None] * np.array([-1.0, 0.0, 1.0]))

        lay(P, pts, hw * (1 + lv["skew"]), hw * (1 - lv["skew"]), col, r, thick=0.95, coarse=0.5, streak=0.06,
            ridge=0.5, double=0.12, soft=1.0, dry=dry, hide=hide)

    flowers = []
    for _ in range(nb + nbud):
        for _ in range(30):
            fx = x + breadth * r.uniform(-0.42, 0.42)
            fy = crown + tall * (0.04 + 0.42 * r.random() ** 0.9)
            if all(np.hypot(fx - a, fy - b) > 46 for a, b, _ in flowers):
                break
        flowers.append((fx, fy, len(flowers) < nb))
    for lv in leaves:
        if lv["depth"] == 0:
            leaf_(lv, MAL["deep"])
    for fx, fy, _ in flowers:          # stalks, mostly lost among the leaves
        pts = np.stack([np.linspace(fx, fx + r.normal(0, 12), 30), np.linspace(fy + 20, foot - 0.25 * tall, 30)], 1)
        lay(P, pts, np.full(30, 3.2), np.full(30, 3.2), MAL["pale"] * 0.85, r, thick=0.8, coarse=0.5, soft=1.0,
            hide=hide)
    for lv in leaves:
        if lv["depth"] == 1:
            leaf_(lv, MAL["pale"] * 0.92)
    for fx, fy, is_bloom in sorted(flowers, key=lambda f: f[1]):
        s = r.uniform(60, 92)
        sheath(P, (fx, fy), s * (1.0 if is_bloom else 0.75), r, hide)
        if is_bloom:
            bloom(P, (fx, fy), s, r.uniform(-1, 1), r.normal(0, 0.2) + 0.5 * (fx - x) / breadth, r, hide)
        else:
            bud(P, (fx, fy + 0.1 * s), s * 0.8, r.normal(0, 0.2), r, hide)
    for lv in leaves:
        if lv["depth"] == 2:
            leaf_(lv, MAL["pale"] * 1.05, short=r.uniform(0.55, 0.85))


# --- the bridge -------------------------------------------------------------------------------------------

def streaks(shape, ang, across, along, r):
    """A field of streaks `along` px long and `across` px wide, running at `ang`."""
    h, w = shape
    n = int(np.hypot(h, w)) + 4
    f = noise.stretched((n, n), across, along, r)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    # rows of f run along its vertical axis, so a streak runs where the first coordinate changes
    return ndimage.map_coordinates(f, [(xx - w / 2) * c + (yy - h / 2) * s + n / 2,
                                       -(xx - w / 2) * s + (yy - h / 2) * c + n / 2], order=1)


def polygons(polys, ss=2):
    """Polygons (plate px) rasterised over the whole plate."""
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for P in polys:
        d.polygon([(x * ss, y * ss) for x, y in P], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255


def hull(top):
    """The silhouette of a span: its top face and the same pushed down by its thickness."""
    pts = np.vstack([top, top + [0, THICK]])
    return pts[ConvexHull(pts).vertices]


def brushed(m, r, amp=0.35, scale=3.0):
    """A mask's edge as the brush left it: never ruled, wandering by a pixel or so."""
    b = ndimage.gaussian_filter(m, 0.9)
    return smoothstep(0.3, 0.7, b + amp * noise.field(m.shape, scale, r) * 4 * b * (1 - b))


def bridge(r):
    """The plank bridge on the gold: each span two boards side by side, seen from above and a little in
    front, with its thickness showing below, and a few posts and rails under it in dark ink. The boards
    are a grey of ink let down with shell white, laid flat; a wash of ink was dropped into it while it was
    wet, so it pooled and dried with dark rims, and the brush left streaks along the boards. -> grey
    coverage, its tone, ink density, post density, and the cover (where the bridge hides what is behind
    it), all (H, W)"""
    o = np.array([X0, Y0], float)
    grey, tone, ink, cover = np.zeros((4, H, W), np.float32)
    pool = noise.fbm((H, W), 60, r, octaves=4) + 0.3 * noise.field((H, W), 9, r)
    tide = np.exp(-((pool - 0.5) / 0.045) ** 2) + 0.7 * np.exp(-((pool + 0.55) / 0.04) ** 2)
    gran = noise.field((H, W), 1.0, r)
    for kind, p, L in SPANS:
        p = np.asarray(p, float) + o
        if kind == "depth":
            top = np.array([p, p + [BWX, 0], p + [BWX, 0] + L * D, p + L * D])
            seam = [p + [BWX / 2, 0], p + [BWX / 2, 0] + L * D]
            ang = np.arctan2(D[1], D[0])
        else:
            top = np.array([p - [L, 0], p, p + BWD * D, p - [L, 0] + BWD * D])
            seam = [p - [L, 0] + 0.5 * BWD * D, p + 0.5 * BWD * D]
            ang = 0.0
        sil = brushed(polygons([hull(top)]), r, 0.2, 4.0)
        face = brushed(polygons([top]), r, 0.2, 4.0) * sil
        side = np.clip(sil - face, 0, 1)
        im = Image.new("L", (W * 2, H * 2), 0)
        ImageDraw.Draw(im).line([tuple(q * 2) for q in seam], fill=255, width=3)
        joint = brushed(np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255, r, 0.15, 6.0)
        grain = streaks((H, W), ang, 1.6, 140.0, r) + 0.5 * streaks((H, W), ang, 5.0, 300.0, r)
        rimm = face * (1 - smoothstep(0.5, 0.95, ndimage.gaussian_filter(face, 1.6)))
        a = 0.82 * face + 0.92 * side
        t = face * (1 + 0.07 * grain + 0.04 * gran) + side * (0.42 + 0.03 * grain)
        d = (face * (0.25 * smoothstep(-0.2, 0.8, pool) + 0.4 * tide + 0.05 * gran + 0.45 * rimm + 0.8 * joint
                     + 0.12 * np.clip(grain, -1, None)) + side * (0.75 + 0.05 * grain + 0.03 * gran))
        grey = grey * (1 - sil) + a
        tone = tone * (1 - sil) + t
        ink = ink * (1 - sil) + d
        cover = np.maximum(cover, sil)
    bars = []
    for x, t, b, drop, rails in TRESTLES:          # two posts, and a rail or two run through them
        x, t = x + X0, t + Y0
        j = lambda: r.normal(0, 1.2, 2)
        for px in (x - b / 2, x + b / 2):
            hw = r.uniform(6, 7.5)
            bars.append(np.array([[px - hw, t] + j(), [px + hw, t] + j(), [px + 0.85 * hw, t + drop] + j(),
                                  [px - 0.85 * hw, t + drop] + j()]))
        for fy in (0.3, 0.68)[:rails]:
            y = t + fy * drop + r.normal(0, 4)
            ext = r.uniform(10, 20, 2)
            bars.append(np.array([[x - b / 2 - ext[0], y - 4.5] + j(), [x + b / 2 + ext[1], y - 4.5] + j(),
                                  [x + b / 2 + ext[1], y + 4.5] + j(), [x - b / 2 - ext[0], y + 4.5] + j()]))
    wood = brushed(polygons(bars), r, 0.15, 6.0) * (1 - cover)
    rimw = wood * (1 - smoothstep(0.35, 0.9, ndimage.gaussian_filter(wood, 1.6)))
    post = wood * (0.85 + 0.1 * gran + 0.12 * pool) + 0.3 * rimw          # the ink pooled at the edges
    return grey, tone, ink, post, np.maximum(cover, wood)


# --- the mount --------------------------------------------------------------------------------------------

def brocade(r):
    """The silk round the painting: a dark indigo ground in twill with small flowers brocaded in gold
    thread, the threads floating along the strip; the gold has rubbed off it here and there. -> rgb"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ox0, oy0, ox1, oy1 = X0 - BR, Y0 - BR, X1 + BR, Y1 + BR
    dt, db, dl, dr = yy - oy0, oy1 - 1 - yy, xx - ox0, ox1 - 1 - xx
    v = np.minimum.reduce([dt, db, dl, dr])
    horiz = np.minimum(dt, db) <= np.minimum(dl, dr)
    u = np.where(horiz, xx, yy) + 0.6 * noise.field((H, W), 40, r)
    v = v + 0.5 * noise.field((H, W), 60, r)
    pitch = 26.0
    lower = v >= BR / 2
    du = ((u - np.where(lower, pitch / 2, 0.0)) % pitch) - pitch / 2
    dv = v - np.where(lower, 0.71 * BR, 0.29 * BR)
    petals = np.zeros((H, W), np.float32)
    for a in np.pi / 4 + np.arange(4) * np.pi / 2:
        petals = np.maximum(petals, smoothstep(3.3, 2.3, np.hypot(du - 4.2 * np.cos(a), dv - 4.2 * np.sin(a))))
    motif = np.clip(petals - 0.9 * smoothstep(1.5, 0.8, np.hypot(du, dv)), 0, 1)
    motif = np.maximum(motif, smoothstep(1.7, 0.9, np.hypot(np.abs(du) - pitch / 2, dv)))
    motif = np.maximum(motif, 0.8 * smoothstep(0.9, 0.3, np.abs(v - BR / 2 - 2.6 * np.sin(2 * np.pi * u / pitch))))
    thread = 0.5 + 0.5 * np.cos(2 * np.pi * v / 1.7)
    twill = 0.88 + 0.12 * np.cos(2 * np.pi * (u + v) / 2.8)
    worn = smoothstep(0.6, 1.5, noise.fbm((H, W), 70, r, octaves=4) + 0.3 * noise.field((H, W), 3, r))
    gm = motif * (0.6 + 0.4 * thread) * (1 - 0.8 * worn)
    ground = lin("#1b1f36") * (twill * (1 + 0.1 * noise.field((H, W), 30, r)))[..., None]
    gold = lin("#b3904a") * (0.8 + 0.25 * thread + 0.12 * noise.field((H, W), 3, r))[..., None]
    rgb = ground * (1 - gm[..., None]) + gold * gm[..., None]
    edge = np.minimum(v, BR - v)
    return rgb * (1 - 0.35 * np.exp(-np.clip(edge, 0, None) / 1.2))[..., None]


def frame(img, r):
    """The black lacquer frame round the whole, a piece to each panel's head and foot, rounded so that a
    line of light runs along it and rubbed through to the red undercoat at the very corners; gilt copper
    caps the four corners."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.minimum.reduce([yy - M, H - M - 1 - yy, xx - M, W - M - 1 - xx])
    on = (v >= 0) & (v < FR)
    sheen = 0.7 + 1.6 * np.exp(-((v - 4.5) / 2.2) ** 2) * (1 + 0.3 * noise.field((H, W), 120, r))
    cx = np.minimum(xx - M, W - M - 1 - xx)
    cy = np.minimum(yy - M, H - M - 1 - yy)
    lac = lin("#110d0c") * sheen[..., None] + lin("#3a1c14") * (0.25 * smoothstep(0.5, 2.5, noise.fbm((H, W), 40, r, octaves=3)))[..., None]
    img = np.where(on[..., None], lac, img)
    arm, tk = 46.0, FR + 2.0
    edge = np.maximum(np.minimum.reduce([cx, cy, tk - cy, arm - cx]), np.minimum.reduce([cx, cy, tk - cx, arm - cy]))
    fit = (edge > 0) & (v >= -0.5)
    bevel = 0.78 + 0.5 * smoothstep(2.5, 0.4, edge)          # a raised rim round a punched ground
    dots = 1 - 0.18 * smoothstep(0.55, 0.9, np.cos(np.pi * xx / 1.6) ** 2 * np.cos(np.pi * yy / 1.6) ** 2) * (edge > 3)
    gilt = lin("#bf9845") * (bevel * dots * (1 + 0.08 * noise.field((H, W), 6, r)))[..., None]
    return np.where(fit[..., None], gilt, img)


def paint(seed=1705):
    rg, rb, ri, rp, rm, rt = (noise.rng(seed + k) for k in range(6))
    img = np.zeros((H, W, 3), np.float32)
    fld = np.s_[Y0:Y1, X0:X1]
    gold, crink = leaf(rg)
    img[fld] = gold
    grey, tone, ink, post, cover = bridge(rb)
    img = img * (1 - grey[..., None]) + PLANK * (tone * grey)[..., None]
    img = glaze(img, ink, SUMI)
    img = glaze(img, post, WOOD)

    P = SimpleNamespace(rgb=np.zeros((H, W, 3), np.float32), a=np.zeros((H, W), np.float32),
                        h=np.zeros((H, W), np.float32), k=np.zeros((H, W), np.float32),
                        edge=0.5 * noise.field((H, W), 2.5, ri) + 0.5 * noise.field((H, W), 7, ri), rt=rt)
    for g in CLUMPS:
        clump(P, g, ri, cover)
    keep = np.zeros((H, W), np.float32)
    keep[fld] = 1
    P.a *= keep

    # the stuff of the paint: sand-sized crystals of the ground stone, light and dark, in a mat of finer ones
    s = (H, W)
    spark = smoothstep(1.6, 2.5, noise.field(s, 1.0, rp))
    speck = smoothstep(1.8, 2.7, noise.field(s, 1.0, rp))
    lum = 1 + 0.1 * noise.field(s, 0.7, rp) + 0.06 * noise.field(s, 1.8, rp) + 0.075 * (
        0.6 * noise.field(s, 9, rp) + 0.4 * noise.field(s, 30, rp))
    rgb = P.rgb * lum[..., None]
    rgb = rgb * (1 - 0.5 * spark[..., None]) + (rgb * 1.8 + 0.012) * 0.5 * spark[..., None]
    rgb *= 1 - 0.3 * speck[..., None]
    # rubbed: the azurite worn thin to a chalky grey-blue in patches, the malachite to a paler green
    rub = smoothstep(0.7, 1.6, noise.fbm(s, 45, rp, octaves=4) + 0.35 * noise.stretched(s, 4, 30, rp))
    rub *= smoothstep(-0.4, 0.9, noise.field(s, 1.4, rp)) * (0.25 + 0.75 * P.k ** 2)
    chalk = np.where(P.k[..., None] > 0.75, lin("#8794b0"), lin("#8db39a"))
    rgb = rgb * (1 - 0.5 * rub[..., None]) + chalk * 0.5 * rub[..., None]
    h = P.h * (1 - 0.4 * rub)
    # where it lay thickest it cracked, and the odd island of the net has fallen out
    crack, island = tempera.craquelure(s, rp, cell=15, aspect=1.15, width=0.55, keep=0.6)
    crack *= smoothstep(0.85, 1.1, h) * P.a * (0.3 + 0.7 * P.k ** 2)     # the malachite is finer and cracks less
    rgb *= 1 - 0.55 * crack[..., None]
    # a few flakes have gone: where the panels rub as the screen is folded, now and then where hands take
    # it up at the foot, and seldom anywhere else
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    jd = np.abs(((xx - X0 + PW / 2) % PW) - PW / 2)
    odds = (1.3 * np.exp(-jd / 10) * smoothstep(0.0, 1.0, noise.field(s, 40, rp))
            + 0.6 * smoothstep(Y1 - 240, Y1, yy) * smoothstep(1.6, 2.4, noise.field(s, 25, rp))
            + 0.7 * smoothstep(2.6, 3.3, noise.field(s, 40, rp))) * smoothstep(0.9, 1.1, h) * P.k
    lost = tempera.flakes(island, odds, rp, scale=7)
    a = P.a * (1 - 0.97 * lost)
    h = h * (1 - lost)
    img = img * (1 - 0.12 * lost[..., None])          # the glue that stayed behind dulls the gold a little

    img = img * (1 - a[..., None]) + rgb * a[..., None]
    # the paint stands a little proud of the leaf, its crystals catching the light: lit on the side toward
    # the window, shadowed on the other
    hg = 0.5 * ndimage.gaussian_filter(h * a, 1.4) + 0.08 * noise.field(s, 1.0, rp) * a
    hg[fld] += crink
    gy, gx = np.gradient(hg)
    cast = np.clip(np.roll(hg, (2, 2), (0, 1)) - hg, 0, None)
    img *= np.clip(1 + 0.45 * (0.6 * gx + 0.8 * gy) - 0.1 * cast, 0.7, 1.3)[..., None]

    border = np.zeros((H, W), bool)
    border[Y0 - BR:Y1 + BR, X0 - BR:X1 + BR] = True
    border[fld] = False
    img = np.where(border[..., None], brocade(rm), img)
    img = frame(img, rm)
    # the joins between the panels: a dark hairline, the edges of the panels a little rounded and handled
    for k in range(1, NP):
        x = X0 + k * PW
        d = np.abs(xx[0] - x + 0.5)
        img[M:H - M] *= (1 - 0.8 * np.exp(-(d / 1.1) ** 2) - 0.1 * np.exp(-d / 4))[None, :, None]
    alpha = np.zeros((H, W), np.float32)
    alpha[M:H - M, M:W - M] = 1
    return plate.mount(img, SimpleNamespace(alpha=alpha), shadow=0.4)
