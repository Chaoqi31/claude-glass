"""The Window over Lake Ashi. Mineral colour, ink, shell white and gold leaf on hemp paper, on a panel.

The Narukawa Art Museum stands on a hill above Moto-Hakone, and in its lounge one long window, divided by
two slim mullions near its ends, looks up Lake Ashi: wooded hills on both shores, the point where Hakone
Shrine stands with its vermilion gate in the water, and Fuji white beyond the ridge at the head of the
lake, framed like a painting; out on the open water one of the sightseeing ships that cross the lake, a
motor vessel dressed as an eighteenth-century galleon, comes down it toward the shrine. This is that view
on a clear afternoon in late autumn, painted in the manner of the modern Nihonga the museum collects, with
the dark room round it: the slatted ceiling drawing in to the eye, the lantern, the sill and a long black
bench.

The sky is gold leaf laid in squares on the paper, the seams of the squares showing, washed thinly with
shell white toward the ridges. Fuji is a ground of pale violet-blue for its bare slopes, with the snow over
it in shell white, laid thick on the side the sun is on, washed with the palest azurite on the side away
from it, and running down the gullies in tapering streaks. The trees are built as Nihonga foliage is
built: each mass laid in first in its darkest colour, then touches of mineral colour of several grades over
it, the coarse grains darker and the fine ones paler, fewer and brighter where the light falls on the upper
left of each lump, until the edges break up into the touches themselves; the cedars by the window are
clumps of upturned tufts gathered along short branches, the maples below the sill heaps of small touches
of cinnabar, vermilion and gold. The mountain across the water is malachite and azurite over ink, its
spurs lit on the side the sun is on and its gullies dark, crowned all over with touches of green and
russet. The lake is laid level with a broad flat brush, lane under lane, each lane giving back the hills,
the mountain and Fuji a little to one side of the last and drawn out along it, so that their colours come
down into the water in strokes and the gold of the sky shows between them only in pieces. The ship is
cinnabar over ink, banded with shell white dotted with its windows, with points of gold at its rails, its
high stern and its figurehead, and its sails furled white on the yards; its wake trails behind it in short
level strokes of shell white along the lanes, and its reflection breaks up lane by lane. The room is ink
and burnt umber over a ground of ink, warmed where the light of the window falls on the sill, the leather
of the bench and the floor; its ceiling slats are laid one by one with a dry brush.
"""

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from atelier import noise, paper
from atelier.color import glaze, lin, pigment
from atelier.noise import smoothstep

TITLE = "The Window over Lake Ashi"
DATE = "2026"
MEDIUM = "Mineral colour (iwa-enogu), ink, shell white and gold leaf on hemp paper, mounted on a panel"
AFTER = ("Modern Nihonga as the Narukawa Art Museum, Hakone, has collected it since it opened in 1988: "
         "Yamamoto Kyūjin, Hirayama Ikuo, Kayama Matazō, Hori Fumiko; and the still blue-green lakes and "
         "mountains of Higashiyama Kaii, such as Green Echoes (Midori hibiku), 1982, Nagano Prefectural Art Museum")
ROOM = "Paper and Water"
YEAR = 1988
PLACE = "Hakone"
REGION = "East Asia"
NOTE = ("The window of the museum's lounge, which frames Fuji over Lake Ashi like a painting, painted as "
        "one: the view in mineral colour on gold leaf, the dark room round it in ink. On the water one of the "
        "lake's sightseeing ships, dressed as a galleon, trails its wake toward the shrine.")

H, W = 1740, 3200
VPX, VPY = 1600.0, 1010.0               # the eye: the far shore of the lake lies at its height
TOP, HEAD, SILL = 296, 340, 1420        # the lintel's upper edge; the glass runs from HEAD to SILL
MULL = (480, 2720)                      # the two slim mullions
FLOOR = 1596                            # where the wall under the window meets the floor
XS = np.arange(W, dtype=np.float32)
SUMI = pigment("#5a5a5f")
GOLD = [lin(c) for c in ("#94692c", "#c89d4c", "#e1bf6e", "#f3dfa0")]
CROWN = {   # the garden's crowns, kind by kind: deep ground, shade, middle, light, top; and the kind each drifts toward
    0: (("#350c0a", "#661a13", "#9c2a1c", "#cc4428", "#ec7a40"), 1),     # cinnabar maple
    1: (("#3e1508", "#76290f", "#b0461c", "#dc6a2a", "#f3a24a"), 2),     # vermilion turning orange
    2: (("#3a2a0c", "#6c5218", "#a8822c", "#d2ac48", "#efd27a"), 1),     # gold
    3: (("#09180f", "#10261c", "#1a3a2b", "#2a5640", "#467a58"), 3),     # dark evergreen
}


def pal(*kinds):
    """Colours of crowns, kind by kind: (shade, light, top) as '#rrggbb'."""
    return [[lin(c) for c in k] for k in kinds]


# --- the grain of the paint ---------------------------------------------------------------------------------

def coat(P, a, col, grade, r, y0=0, x0=0, var=0.08, settle=0.3, thick=0.0, grain=1.0, g=None):
    """One coat of iwa-enogu: crystals of one grade (`grade`, about their size in px) carried in glue and
    brushed on where `a` (0..1) says how much paint the brush left. A full coat closes up; a thin one lies
    unevenly, the crystals gathering in clumps and settling into the hollows of the paper as the water goes,
    so the ground and the coats beneath show through it in specks, the more so the coarser the grade.
    `col` is a colour or a map the shape of `a`; `a` sits at (y0, x0) on the plate. `g`, a unit field the
    shape of `a`, replaces the grain where the brush drags it into lanes."""
    ys, xs = np.nonzero(a > 2e-3)
    if not len(ys):
        return
    b0, b1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = np.clip(a[b0:b1, c0:c1], 0, 1)
    if np.ndim(col) == 3:
        col = col[b0:b1, c0:c1]
    sl = np.s_[y0 + b0:y0 + b1, x0 + c0:x0 + c1]
    s = a.shape
    if g is None:
        g = (0.42 * noise.field(s, 0.6 * grade, r) + 0.62 * noise.field(s, 1.8 * grade, r)
             + 0.42 * noise.field(s, 5.0 * grade, r) + 0.2 * noise.field(s, 22.0, r))
    else:
        g = g[b0:b1, c0:c1]
    g = g + settle * 2.0 * (0.5 - P.tooth[sl])
    amp = grain * (0.22 + 0.26 * np.clip((grade - 0.5) / 1.2, 0, 1))
    cov = np.clip(a + amp * g * 4 * a * (1 - a), 0, 1)
    c = col * (1 + var * noise.field(s, 0.8 * grade, r))[..., None]
    P.rgb[sl] = P.rgb[sl] * (1 - cov[..., None]) + c * cov[..., None]
    P.k[sl] = P.k[sl] * (1 - cov) + grade * cov
    if thick:
        P.h[sl] += thick * cov


def streaks(shape, across, along, r, ang=0.0):
    """A field of streaks `along` px long and `across` px wide, running at `ang` (radians, 0 = level)."""
    h, w = shape
    if abs(ang) < 1e-3:
        return noise.stretched((w, h), across, along, r).T
    n = int(np.hypot(h, w)) + 4
    f = noise.stretched((n, n), across, along, r)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    return ndimage.map_coordinates(f, [(xx - w / 2) * c + (yy - h / 2) * s + n / 2,
                                       -(xx - w / 2) * s + (yy - h / 2) * c + n / 2], order=1)


def hake(shape, r, along=500.0, width=60.0, amp=(0.2, 0.1)):
    """How a broad flat brush (hake) lays a coat, level: bands a brush wide where one stroke carried more
    paint than the next, and lanes left by its hairs. -> a factor about 1"""
    return 1 + amp[0] * streaks(shape, width, along, r) + amp[1] * streaks(shape, 1.8, along * 0.5, r)


def smear(a, s):
    """`a` drawn out level by a brush, about `s` px either way (three box passes, near enough a gaussian)."""
    for _ in range(3):
        a = ndimage.uniform_filter1d(a, 2 * int(s) + 1, axis=1, mode="nearest")
    return a


def skyline(pts, r, wob=1.5, scale=30.0):
    p = np.asarray(pts, float)
    y = PchipInterpolator(p[:, 0], p[:, 1])(np.clip(XS, p[0, 0], p[-1, 0]))
    return (y + wob * noise.line1d(W, scale, r)).astype(np.float32)


def rows(y0, y1):
    return np.arange(y0, y1, dtype=np.float32)[:, None]


def fall(top, y0, y1, x0, x1, peak, r, width=36.0, lean=0.6):
    """The ground of a hillside as spurs and gullies running down from its skyline along the fall lines,
    which lean out from the peak: high on the spurs, low in the gullies. -> a field (y1 - y0, x1 - x0)"""
    h, w = y1 - y0, x1 - x0
    yy = rows(y0, y1)
    xx = XS[None, x0:x1]
    side = np.clip((xx - peak) / 200, -1, 1)
    dep = np.clip(yy - top[None, x0:x1], 0, None)
    pad = int(lean * h) + 8
    F = (noise.stretched((h + 8, w + 2 * pad), width, 4 * width, r)
         + 0.35 * noise.stretched((h + 8, w + 2 * pad), width / 3, width * 2, r))
    f = ndimage.map_coordinates(F, [np.broadcast_to(yy - y0, (h, w)), xx - lean * side * dep - x0 + pad], order=1)
    return ndimage.gaussian_filter(f, 3.0)


def terrain(top, y0, y1, x0, x1, peak, r, width=36.0, lean=0.6, amp=0.2, flank=0.12):
    """How the light falls on a hillside of spurs and gullies (see `fall`), from the left: the flank turned
    to the sun lighter, and each spur lit on its left and shadowed on its right. -> a factor about 1"""
    f = fall(top, y0, y1, x0, x1, peak, r, width, lean)
    side = np.clip((XS[None, x0:x1] - peak) / 200, -1, 1)
    gx = np.gradient(f, axis=1)
    return (1 + amp * np.tanh(gx / (1.5 * gx.std() + 1e-6)) + 0.06 * np.tanh(f) - flank * side).astype(np.float32)


# --- the gold ------------------------------------------------------------------------------------------------

def ramp(e):
    e = np.clip(e, 0, 1) * (len(GOLD) - 1)
    i = np.minimum(e.astype(int), len(GOLD) - 2)
    f = (e - i)[..., None]
    G = np.stack(GOLD)
    return G[i] * (1 - f) + G[i + 1] * f


def leaf(shape, r, size=186.0):
    """Gold leaf laid in rows of squares, each lapping the one before it and the row above: a leaf's own
    edge shows as a hairline, and the strip where two leaves lie double takes the light a little
    differently, so the seams show as a faint net. Every leaf lies at its own slight angle and takes the
    light a shade differently; a few wrinkled as they went down. -> rgb, relief"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wy = yy + 0.8 * noise.field(shape, 50, r) + 0.3 * noise.field(shape, 5, r)
    wx = xx + 0.8 * noise.field(shape, 50, r) + 0.3 * noise.field(shape, 5, r)
    nrow, ncol = int(h / size) + 3, int(w / size) + 3
    B = -r.uniform(0, size) + np.concatenate([[0], np.cumsum(size + r.normal(0, 1.5, nrow))])
    k = np.clip(np.searchsorted(B, wy, side="right") - 1, 0, nrow - 1)
    top = wy - B[k]
    off = r.uniform(0, size, nrow)
    q = (wx + off[k]) / size
    j = np.floor(q).astype(int)
    left = (q - j) * size
    idx = k * (ncol + 1) + np.clip(j, 0, ncol)
    tone, tint, gx, gy, lap = r.normal(0, 1, (5, nrow * (ncol + 1)))
    u, v = left / size - 0.5, top / size - 0.5
    e = (0.57 + 0.045 * np.clip(tone[idx], -2, 2) + 0.024 * (gx[idx] * u + gy[idx] * v)
         - 0.13 * (np.exp(-(top / 0.8) ** 2) + np.exp(-(left / 0.8) ** 2))
         + 0.07 * (np.exp(-((top - 1.8) / 0.8) ** 2) + np.exp(-((left - 1.8) / 0.8) ** 2))
         + 0.03 * lap[idx] * smoothstep(6.0, 4.0, np.minimum(top, left)))
    crease = Image.new("F", (w, h), 0.0)
    cd = ImageDraw.Draw(crease)
    for _ in range(int(nrow * ncol * 0.3)):
        cx, cy, th = r.uniform(0, w), r.uniform(0, h), r.uniform(0, np.pi)
        pts = [(cx, cy)]
        for _ in range(r.integers(4, 10)):
            th += r.normal(0, 0.2)
            cx, cy = cx + 6 * np.cos(th), cy + 6 * np.sin(th)
            pts.append((cx, cy))
        cd.line(pts, fill=float(r.choice([-1, 1]) * r.uniform(0.5, 1.0)), width=1)
    cr = ndimage.gaussian_filter(np.asarray(crease), 0.7)
    cgy, cgx = np.gradient(cr)
    e += 0.9 * (0.6 * cgx + 0.8 * cgy) + 0.11 * noise.fbm(shape, 650, r, octaves=3) + 0.012 * noise.field(shape, 4, r)
    col = ramp(e) * (1 + 0.4 * np.clip(tint[idx], -2, 2)[..., None] * np.array([0.03, 0.0, -0.08], np.float32))
    return col.astype(np.float32), (0.4 * cr).astype(np.float32)


def sunago(P, dens, r, y0=0, x0=0, rate=0.05, big=0.06):
    """Gold dust shaken through a sieve: flecks of leaf of all sizes, densest where `dens` is high, each
    lying at its own angle and catching the light its own way; now and then a larger cut square."""
    h, w = dens.shape
    n = int(dens.sum() * rate)
    if n == 0:
        return
    ys, xs = np.nonzero(dens > 1e-3)
    p = dens[ys, xs] / dens[ys, xs].sum()
    pick = r.choice(len(ys), n, p=p)
    im = Image.new("F", (w * 2, h * 2), 0.0)
    d = ImageDraw.Draw(im)
    for y, x in zip(ys[pick] + r.uniform(0, 1, n), xs[pick] + r.uniform(0, 1, n)):
        s = r.lognormal(-0.2, 0.45) * (3.0 if r.random() < big else 1.0)
        th = r.uniform(0, np.pi)
        c, sn = np.cos(th), np.sin(th)
        pts = [(2 * (x + s * (c * u - sn * v)), 2 * (y + s * (sn * u + c * v)))
               for u, v in ((-0.6, -0.5), (0.55, -0.6), (0.6, 0.5), (-0.5, 0.6))]
        d.polygon(pts, fill=float(r.uniform(0.55, 1.25)))
    f = np.asarray(im.resize((w, h), Image.BOX), np.float32)
    a = np.clip(f, 0, 1)
    col = ramp(0.45 + 0.4 * np.clip(f, 0, 1.3))
    sl = np.s_[y0:y0 + h, x0:x0 + w]
    P.rgb[sl] = P.rgb[sl] * (1 - a[..., None]) + col * a[..., None]
    P.h[sl] += 0.4 * a


# --- crowns and touches ----------------------------------------------------------------------------------------

def lobe(cx, cy, rx, ry, r, n=14, rough=0.12, ang=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + r.uniform(0, 1)
    k = 1 + rough * noise.line1d(n + 8, 2.5, r)[:n]
    px, py = rx * k * np.cos(t), ry * k * np.sin(t)
    c, s = np.cos(ang), np.sin(ang)
    return list(zip(cx + c * px - s * py, cy + s * px + c * py))


def crown(d, v, cx, cy, s, r):
    """One crown drawn into a label image: all of it in shade, then the part turned to the light (up and
    to the left, where the low sun is), then its brightest top."""
    for _ in range(r.integers(3, 6)):
        d.polygon(lobe(cx + r.normal(0, 0.3 * s), cy + r.normal(0, 0.16 * s), s * r.uniform(0.42, 0.62),
                       s * r.uniform(0.34, 0.5), r), fill=v + 1)
    for _ in range(2):
        d.polygon(lobe(cx - 0.18 * s + r.normal(0, 0.1 * s), cy - 0.2 * s + r.normal(0, 0.06 * s),
                       0.44 * s, 0.34 * s, r), fill=v + 2)
    d.polygon(lobe(cx - 0.3 * s, cy - 0.32 * s, 0.24 * s, 0.17 * s, r, 10), fill=v + 3)


def lay_crowns(P, lab, cols, r, y0, x0, grade, load=0.97, keep=None):
    """The crowns as paint: each in its own colours, shade, light and top (`cols`, one (3, 3) per label,
    the first unused), in one coat. `keep` (0..1, the label's shape) masks it."""
    lab = np.asarray(lab)
    idx, part = lab // 4, lab % 4
    col = cols[idx, np.clip(part - 1, 0, 2)]
    col = ndimage.gaussian_filter(col, (0.7, 0.7, 0))
    a = ndimage.gaussian_filter((lab > 0).astype(np.float32), 0.8)
    if keep is not None:
        a = a * keep
    coat(P, a * load, col, grade, r, y0, x0, var=0.1)


def tint(colours, kinds, tones):
    """Per-crown colours for `lay_crowns` from a palette of kinds and each crown's tone."""
    P3 = np.stack([np.stack(p) for p in colours]).astype(np.float32)
    return np.concatenate([np.zeros((1, 3, 3), np.float32), P3[kinds] * np.asarray(tones, np.float32)[:, None, None]])


def scatter(m, n, r, pad=0):
    """n points scattered over where `m` > 0.5, in label-image coordinates (pad rows added on top), sorted
    from the top down so each crown overlaps the ones above it."""
    ys, xs = np.nonzero(m > 0.5)
    pick = r.integers(0, len(ys), n)
    cy, cx = ys[pick] + r.uniform(0, 1, n) + pad, xs[pick] + r.uniform(0, 1, n)
    o = np.argsort(cy)
    return cy[o], cx[o]


def sample(m, weight, n, r):
    """n points (x, y) over the map, more of them where `weight` (0..1) is high."""
    ys, xs = np.nonzero((m > 0.5) & (weight > 0.02))
    if not len(ys) or n < 1:
        return np.zeros((0, 2))
    p = weight[ys, xs].astype(np.float64)
    pick = r.choice(len(ys), n, p=p / p.sum())
    return np.stack([xs[pick] + r.uniform(0, 1, n), ys[pick] + r.uniform(0, 1, n)], 1)


def blot(x, y, rx, ry, th, r, n=10, rough=0.22):
    """The outline of one touch of a loaded brush: a blot, never quite round."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = r.normal(0, 1, 3) * (0.6, 0.5, 0.4)
    k = 1 + rough * (a[0] * np.sin(t + r.uniform(0, 6.3)) + a[1] * np.sin(2 * t + r.uniform(0, 6.3))
                     + a[2] * np.sin(3 * t + r.uniform(0, 6.3)))
    px, py = rx * k * np.cos(t), ry * k * np.sin(t)
    c, s = np.cos(th), np.sin(th)
    return list(zip(x + c * px - s * py, y + s * px + c * py))


def blade(x, y, L, th, r):
    """A tuft: the brush pressed down at its base and lifted away to a point along `th`."""
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    c = np.cos(phi)
    wd = L * r.uniform(0.26, 0.4)
    px = c * L * np.where(c > 0, 1.0, 0.3)
    py = np.sin(phi) * wd * (1 - 0.75 * np.clip(c, 0, 1)) * (1 + 0.2 * r.normal(0, 1, 12))
    ct, st = np.cos(th), np.sin(th)
    return list(zip(x + ct * px - st * py, y + st * px + ct * py))


def star(x, y, s, th, r):
    """A maple leaf as the brush touches it in: five to seven points round a small heart."""
    k = int(r.integers(5, 8))
    t = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False) + th
    rad = s * np.where(np.arange(2 * k) % 2 == 0, 1.0, r.uniform(0.42, 0.6)) * (1 + 0.16 * r.normal(0, 1, 2 * k))
    return list(zip(x + rad * np.cos(t), y + rad * np.sin(t)))


def marks(shape, pts, r, size, form="dab", spread=0.35, ang=None, tones=None, ss=2, soft=0.6):
    """Touches of the brush at `pts` ((x, y) in the map's px), their sizes scattered about `size`: 'dab', a
    blot a little drawn out and turned its own way; 'tuft', a blade pointing along `ang` (one angle, or
    one per touch); 'leaf', a little star. Each touch is one colour, which `tones` (one per touch) picks,
    and where touches overlap the last laid lies on top. -> coverage, tone"""
    h, w = shape
    pts = np.asarray(pts, float).reshape(-1, 2)
    n = len(pts)
    cov = Image.new("L", (w * ss, h * ss), 0)
    ton = Image.new("F", (w * ss, h * ss), 0.0)
    dc, dt = ImageDraw.Draw(cov), ImageDraw.Draw(ton)
    S = np.exp(r.normal(np.log(size), spread, n)) * ss
    TH = r.uniform(0, np.pi, n) if ang is None else np.broadcast_to(np.asarray(ang, float), (n,)) + r.normal(0, 0.2, n)
    T = np.zeros(n) if tones is None else np.asarray(tones, float)
    E = r.uniform(1.0, 1.6, n)
    for (x, y), s, th, t, e in zip(pts * ss, S, TH, T, E):
        if form == "tuft":
            poly = blade(x, y, s, th, r)
        elif form == "leaf":
            poly = star(x, y, s, th, r)
        else:
            poly = blot(x, y, s * e, s, th, r)
        dc.polygon(poly, fill=255)
        dt.polygon(poly, fill=float(t))
    c = np.asarray(cov.resize((w, h), Image.BOX), np.float32) / 255
    t = np.asarray(ton.resize((w, h), Image.BOX), np.float32) / np.maximum(c, 1e-3)
    return (ndimage.gaussian_filter(c, soft) if soft else c), t


def lay(P, cov, tone, lo, hi, grade, r, y0=0, x0=0, load=0.9, thick=0.0):
    """Touches as paint, each its own colour between `lo` (tone 0) and `hi` (tone 1)."""
    t = np.clip(tone, 0, 1)[..., None]
    coat(P, cov * load, lo * (1 - t) + hi * t, grade, r, y0, x0, var=0.06, thick=thick)


# --- the view --------------------------------------------------------------------------------------------------

def fuji_top(r):
    """The cone as it shows from Hakone: a broad, slightly tilted summit, long concave flanks, the left one
    steeper and swelling a little where the Hōei crater lies, the right one longer."""
    dl, dr = np.clip(1956 - XS, 0, None), np.clip(XS - 2036, 0, None)
    top = 560 + 2.55 * dl ** 0.82 + 1.82 * dr ** 0.82 + 0.045 * (XS - 1995)
    top += (-3.5 * np.exp(-((XS - 1966) / 9) ** 2) - 2.5 * np.exp(-((XS - 2020) / 11) ** 2)
            + 1.5 * np.exp(-((XS - 1992) / 7) ** 2) - 13 * np.exp(-((XS - 1700) / 55) ** 2))
    return top + 0.9 * noise.line1d(W, 9, r) + 1.6 * noise.line1d(W, 50, r)


def sky(P, r):
    """Gold leaf over the whole sky, then the thinnest wash of shell white laid level over its lower part,
    so the leaf glows through paler toward the ridges and keeps its squares."""
    y0, y1 = TOP, 1030
    h = y1 - y0
    col, crink = leaf((h, W), r)
    P.rgb[y0:y1] = col
    P.h[y0:y1] += crink
    yy = rows(y0, y1)
    u = np.clip((yy - HEAD) / (900 - HEAD), 0, 1)
    one = np.ones((1, W), np.float32)
    coat(P, (0.02 + 0.24 * u ** 1.6) * hake((h, W), r, along=700, width=80, amp=(0.3, 0.08)) * one, lin("#f4efe4"),
         0.5, r, y0, settle=0.2)
    warm = np.exp(-((XS - 2000) / 900) ** 2)[None, :] * smoothstep(0.35, 0.9, u)
    coat(P, 0.15 * warm * one, lin("#f1d9a2"), 0.55, r, y0)


def fuji(P, r):
    """The mountain: a ground of pale violet-blue for the bare slopes, then the snow in shell white laid
    thick on the side the sun is on and washed over with pale azurite on the side away from it; a cap down
    to a ragged edge, bare ribs of rock biting up into it, and below it the snow running down the gullies
    in tapering streaks of every length. The foot is lost in the warm air over the far ridges."""
    xa, xb, y0, y1 = 1400, 2650, 520, 960
    w, h = xb - xa, y1 - y0
    yy = rows(y0, y1)
    xx = XS[None, xa:xb]
    top = fuji_top(r)
    m = smoothstep(-1.4, 1.4, yy - top[None, xa:xb])
    P.land[y0:y1, xa:xb] = np.maximum(P.land[y0:y1, xa:xb], m)
    sx, sy = 1995.0, 558.0
    th = (xx - sx) / (yy - sy + 60)                           # which fall line from the summit
    capline = (sy + 94 + 15 * noise.line1d(w, 70, r) + 6 * noise.line1d(w, 14, r)
               + 14 * smoothstep(0, 1, (XS[xa:xb] - sx) / 154))
    cap = smoothstep(2.0, -2.0, yy - capline[None, :])
    ss = 2
    gul, rib = Image.new("L", (w * ss, h * ss), 0), Image.new("L", (w * ss, h * ss), 0)
    dg, dr = ImageDraw.Draw(gul), ImageDraw.Draw(rib)
    S = lambda x, y: (ss * (x - xa), ss * (y - y0))

    def streak(d, t0, ya, L, w0, down):
        n = 18
        s = np.linspace(0, 1, n)
        yv = ya + (L * s if down else -L * s)
        xv = sx + t0 * (yv - sy + 60) + np.cumsum(r.normal(0, 0.6, n))
        wv = w0 * (1 - s) ** 0.75 + 0.3
        d.polygon([S(x - ww, y) for x, y, ww in zip(xv, yv, wv)]
                  + [S(x + ww, y) for x, y, ww in zip(xv[::-1], yv[::-1], wv[::-1])], fill=255)

    for _ in range(52):                                       # snow lying on down the gullies
        t0 = r.uniform(-1.25, 1.32)
        x0 = sx + t0 * (capline.mean() - sy + 60)
        ya = capline[int(np.clip(x0 - xa, 0, w - 1))] - 6
        L = np.exp(r.uniform(np.log(14), np.log(150))) * (1.25 if t0 > 0.1 else 1.0)
        streak(dg, t0, ya, L, r.uniform(1.8, 6.5) * (1 + 0.6 * (L > 80)), True)
    for _ in range(26):                                       # bare ribs between them, biting up into the cap
        t0 = r.uniform(-1.2, 1.25)
        x0 = sx + t0 * (capline.mean() - sy + 60)
        ya = capline[int(np.clip(x0 - xa, 0, w - 1))] + 4
        streak(dr, t0, ya, r.uniform(10, 48), r.uniform(1.4, 3.5), False)
    gm = np.asarray(gul.resize((w, h), Image.BOX), np.float32) / 255
    rm = np.asarray(rib.resize((w, h), Image.BOX), np.float32) / 255
    snow = np.clip(np.maximum(cap, gm) - rm, 0, 1) * m
    ti = np.clip((th + 1.8) / 3.6 * 1999, 0, 1999)              # the terminator runs down a fall line
    lit = smoothstep(0.2, -0.02, th + 0.06 * np.interp(ti, np.arange(2000), noise.line1d(2000, 9, r)))
    ground = lin("#a6b4cc") * lit[..., None] + lin("#8494b6") * (1 - lit[..., None])
    coat(P, m * 0.97, ground, 0.9, r, y0, xa)
    shade = lin("#b8cbe0") * (1 - 0.12 * smoothstep(0.2, 1.1, th))[..., None]
    snowcol = lin("#f6f1e7") * lit[..., None] + shade * (1 - lit[..., None])
    coat(P, snow * 0.97, snowcol, 0.7, r, y0, xa, thick=0.5)
    coat(P, snow * lit * 0.7, lin("#fbf7ee"), 0.55, r, y0, xa, thick=0.9)
    blush = smoothstep(150, 30, yy - sy) * lit
    coat(P, snow * blush * 0.28, lin("#f6dfcf"), 0.5, r, y0, xa)
    coat(P, m * smoothstep(770, 885, yy) * 0.8, lin("#e4d6b2"), 0.55, r, y0, xa)


def ridges(P, r):
    """The far ridges at the head of the lake, two flat coats of pale azurite, the nearer a shade deeper,
    lying across the foot of the mountain."""
    far = skyline([(1150, 884), (1290, 866), (1420, 874), (1560, 862), (1700, 876), (1850, 868), (2000, 880),
                   (2140, 870), (2300, 884), (2450, 896), (2600, 912), (2800, 932), (3000, 950)], r, 1.2, 18)
    near = skyline([(1250, 960), (1400, 936), (1560, 944), (1720, 926), (1880, 946), (2050, 958), (2250, 980),
                    (2500, 1000)], r, 1.2, 14)
    y0, y1 = 840, 1030
    yy = rows(y0, y1)
    a = smoothstep(-1.2, 1.2, yy - far[None, :]) * ((XS > 1100) & (XS < 3050))[None, :]
    coat(P, a * 0.97 * hake((y1 - y0, W), r, along=300, width=30, amp=(0.1, 0.05)), lin("#b2c2cc"), 0.55, r, y0)
    b = smoothstep(-1.2, 1.2, yy - near[None, :]) * ((XS > 1200) & (XS < 2600))[None, :]
    coat(P, b * 0.97, lin("#8ba7b3"), 0.7, r, y0)
    P.land[y0:y1] = np.maximum(P.land[y0:y1], np.maximum(a, b))


def mountain(P, r):
    """The big forested mountain across the lake on the left. A ground of ink and azurite, malachite where
    the sun is: its broad spurs run down from the ridge, lit on the left and in shadow on the right, with
    the gullies dark between them and the whole far flank turned away. Over it the crowns of its trees one
    by one from the ridge down, the ones on the skyline making its edge: blue-green, and in patches gone
    gold and russet; then loose touches of pale malachite where the sun lies fullest and of ink down the
    gullies."""
    top = skyline([(0, 700), (160, 668), (330, 630), (470, 612), (560, 618), (680, 602), (800, 622), (960, 668),
                   (1100, 730), (1250, 800), (1380, 868), (1500, 930), (1600, 985), (1690, 1013)], r, 1.0, 20)
    y0, y1, x1, pad = 580, 1020, 1720, 14
    h = y1 - y0
    yy = rows(y0, y1)
    xx = XS[None, :x1]
    foot = smoothstep(1016, 1010, yy)
    P.land[y0:y1, :x1] = np.maximum(P.land[y0:y1, :x1], smoothstep(-1.0, 1.0, yy - top[None, :x1]) * foot)
    m = smoothstep(-1.0, 1.0, yy - top[None, :x1] - 4) * foot       # the ground stops short: the crowns make the edge
    dep = np.clip(yy - top[None, :x1], 0, None)
    f = fall(top, y0, y1, 0, x1, 640, r, 140, 0.85) + 0.55 * fall(top, y0, y1, 0, x1, 640, r, 50, 0.85)
    f = f / f.std()
    gx = np.gradient(ndimage.gaussian_filter(f, 2.0), axis=1)
    face = np.tanh(gx / (1.1 * gx.std() + 1e-6))
    gully = smoothstep(-0.4, -1.5, f)
    side = np.clip((xx - 640) / 700, -1, 1)
    light = np.clip(0.52 + 0.3 * face - 0.22 * gully + 0.08 * np.tanh(f) - 0.12 * side
                    + 0.1 * smoothstep(50, 0, dep), 0, 1)
    autumn = (smoothstep(0.7, 1.4, noise.fbm((h, x1), 70, r, octaves=3))
              * smoothstep(-0.3, 0.6, noise.field((h, x1), 12, r)))
    a3, l3 = autumn[..., None], light[..., None]
    base = lin("#112a33") * (1 - l3) + lin("#2f5d55") * l3
    base = base * (1 - 0.55 * a3) + (lin("#3a2a20") * (1 - l3) + lin("#6e5232") * l3) * 0.55 * a3
    coat(P, m * 0.97, base, 1.3, r, y0)
    lab = Image.new("I", (x1, h + pad), 0)
    d = ImageDraw.Draw(lab)
    n = int((m > 0.5).sum() * 1.7 / (np.pi * 4.6 ** 2))
    cy, cx = scatter(m, n, r, pad)
    iy, ix = np.clip(cy - pad, 0, h - 1).astype(int), np.clip(cx, 0, x1 - 1).astype(int)
    au = autumn[iy, ix] + 0.25 * r.normal(0, 1, n)
    z = noise.fbm((h, x1), 30, r, octaves=2)[iy, ix]
    kinds = np.where(au > 0.55, np.where(z > 0.5, 3, np.where(z < -0.4, 4, 2)), np.where(z > 0.3, 1, 0))
    for i in range(n):
        crown(d, 4 * (i + 1), cx[i], cy[i], 5.2 * r.uniform(0.75, 1.3), r)
    colours = pal(("#14313a", "#1f4648", "#2c5a55"),        # evergreen in the blue of distance
                  ("#173a33", "#24524a", "#386c5c"),        # the same, greener
                  ("#4a3c22", "#6c5a30", "#8f7a44"),        # gold
                  ("#432a1e", "#5e3c28", "#7a5034"),        # russet
                  ("#4c221a", "#6c2e20", "#8c3c28"))        # red
    keep = np.ones((h + pad, x1), np.float32)
    keep[pad:] = foot
    lay_crowns(P, lab, tint(colours, kinds, (0.55 + 0.9 * light[iy, ix]) * r.uniform(0.9, 1.1, n)), r,
               y0 - pad, 0, 1.1, keep=keep)
    lit = smoothstep(0.55, 0.9, light) * m
    pts = sample(m, lit, int(lit.sum() / 30), r)
    c, t = marks((h, x1), pts, r, 2.6, tones=r.uniform(0, 1, len(pts)))
    lay(P, c * foot, t, lin("#4f806e") * (1 - a3) + lin("#9a7a44") * a3, lin("#86a681") * (1 - a3) + lin("#c49c58") * a3,
        0.7, r, y0, 0, 0.75, 0.2)
    dark = gully * m * smoothstep(0.55, 0.3, light)
    pts = sample(m, dark, int(dark.sum() / 45), r)
    c, _ = marks((h, x1), pts, r, 2.8)
    coat(P, c * foot * 0.6, lin("#0c1c22"), 1.4, r, y0)


def headland(P, r, shore):
    """The wooded point where the shrine stands, the nearest hill: a ground of ink and deep malachite, and
    over it its crowns one by one from the ridge down, each a little lighter on the side toward the sun:
    evergreen oak and cedar, and in patches maples and zelkova turned russet, gold and cinnabar, most of
    them low down toward the water."""
    top = skyline([(1690, 1013), (1760, 998), (1880, 976), (2000, 948), (2120, 916), (2260, 872), (2400, 824),
                   (2560, 768), (2720, 716), (2880, 670), (3040, 632), (3200, 602)], r, 1.0, 20)
    y0, y1, x0, pad = 560, 1110, 1680, 40
    w = W - x0
    yy = rows(y0, y1)
    m = smoothstep(-1.0, 1.0, yy - top[None, x0:]) * smoothstep(1.0, -1.0, yy - shore[None, x0:])
    P.land[y0:y1, x0:] = np.maximum(P.land[y0:y1, x0:], m)
    light = terrain(top, y0, y1, x0, W, 3600, r, width=70, lean=0.45, amp=0.14, flank=0.0)
    P.rgb[y0:y1, x0:] = glaze(P.rgb[y0:y1, x0:], 1.6 * m, SUMI)
    coat(P, m * 0.95, lin("#163629") * light[..., None], 1.6, r, y0, x0)
    lab = Image.new("I", (w, y1 - y0 + pad), 0)
    d = ImageDraw.Draw(lab)
    near = lambda x: 0.75 + 0.55 * (x - 1680) / 1520
    n = int((m > 0.5).sum() * 2.4 / (np.pi * 9.0 ** 2))
    cy, cx = scatter(m, n, r, pad)
    iy, ix = np.clip(cy - pad, 0, y1 - y0 - 1).astype(int), cx.astype(int)
    low = smoothstep(80, 10, shore[ix + x0] - (cy - pad + y0))              # near the water
    z1 = noise.fbm((y1 - y0, w), 60, r, octaves=3)[iy, ix]
    z2 = noise.fbm((y1 - y0, w), 40, r, octaves=2)[iy, ix]
    z3 = noise.fbm((y1 - y0, w), 30, r, octaves=2)[iy, ix]
    autumn = z1 + 0.8 * low + 0.2 * r.normal(0, 1, n) > 1.05
    kinds = np.where(autumn, np.where(z2 > 0.6, 4, np.where(z2 < -0.3, 3, 5)),
                     np.where(z3 > 1.0, 2, np.where(z3 > 0.0, 1, 0)))
    for i in range(n):
        crown(d, 4 * (i + 1), cx[i], cy[i], 10.5 * near(cx[i] + x0) * r.uniform(0.6, 1.45), r)
    colours = pal(("#173c30", "#1d4a3a", "#255744"),        # evergreen oak
                  ("#163739", "#1c4547", "#245252"),        # the same, bluer in shadow
                  ("#11292a", "#163632", "#1e443c"),        # cedar
                  ("#55482a", "#6e5e32", "#8a763e"),        # gold: zelkova, konara
                  ("#58251a", "#7c321f", "#9c4128"),        # cinnabar maple
                  ("#472c1e", "#5e3c28", "#764c32"))        # russet
    keep = np.ones((y1 - y0 + pad, w), np.float32)
    keep[pad:] = smoothstep(1.5, -0.5, yy - shore[None, x0:])
    lay_crowns(P, lab, tint(colours, kinds, r.uniform(0.9, 1.1, n) * light[iy, ix]), r, y0 - pad, x0, 1.3,
               keep=keep)


def lake(P, r, shore):
    """The lake, still under the far shores, laid level with a broad brush lane under lane, the lanes finer
    far off. Under it all a ground of ink deepening toward us and the water's own azurite; over that each
    lane gives back what stands above the shore at its depth, but a little to one side of the lane before,
    and in a stroke that carried more paint here and less there: so the hills, the mountain and Fuji come
    down into the water in their own colours, darker and bluer, and fade as the water comes toward us,
    and where the sky would be given back the lake keeps its blue and the gold shows only in pieces."""
    y0, y1 = 1000, SILL + 24
    h = y1 - y0
    yy = rows(y0, y1)
    water = smoothstep(-1.0, 1.0, yy - shore[None, :])
    u = np.clip((yy - VPY) / (SILL - VPY), 0, 1)
    d = np.clip(yy - shore[None, :], 0, None)
    ur = np.clip((np.arange(y0, y1, dtype=np.float32) - VPY) / (SILL - VPY), 0, 1)
    bh = 1.4 + 8.5 * ur ** 1.15                               # how tall a lane is
    phase = np.cumsum(1 / bh)[:, None] + 0.25 * noise.line1d(W, 500, r)[None, :]
    band = np.floor(phase).astype(int)
    nb = int(band.max()) + 1
    db = np.clip(d - (phase - band - 0.5) * bh[:, None], 0, None)     # the depth at the middle of the lane
    sx = r.normal(0, 1, nb)[band] * (8 + 30 * u)
    sy = np.clip(shore[None, :] - db / 1.2, HEAD + 4, H - 1)  # the ripples draw each reflection out a little
    sxx = np.clip(XS[None, :] + sx + 0.6 * streaks((h, W), 3.0, 400, r), 0, W - 1)
    refl = np.stack([ndimage.map_coordinates(P.rgb[..., c], [sy, sxx], order=1) for c in range(3)], -1)
    land = ndimage.map_coordinates(P.land, [sy, sxx], order=1)
    pick = r.choice(3, nb, p=[0.3, 0.35, 0.35])[band]         # how far each stroke drew the land's colour along
    rl, spread = np.zeros_like(refl), np.zeros_like(land)
    for i, s in enumerate((25, 80, 220)):
        L = smear(land, s)
        k = pick == i
        rl[k] = (smear(refl * land[..., None], s) / np.maximum(L, 1e-3)[..., None])[k]
        spread[k] = L[k]
    cover = smoothstep(0.04, 0.45, spread)                    # where the stroke carries the colour of the land
    refl = smear(refl, 3)
    cols = np.arange(W)[None, :]
    run = noise.line1d(W, 260, r, rows=nb)[band, cols]        # where a stroke carried more paint, and less
    dash = noise.line1d(W, 70, r, rows=nb)[band, cols] * (r.random(nb) < 0.3)[band]

    def lanes():
        return (0.55 * streaks((h, W), 1.0, 90, r) + 0.6 * streaks((h, W), 2.5, 260, r)
                + 0.45 * streaks((h, W), 7.0, 700, r))

    sl = np.s_[y0:y1]
    P.rgb[sl] = glaze(P.rgb[sl], water * (0.3 + 1.1 * u), SUMI)
    body = lin("#2a6a86") * (1 - u[..., None] ** 0.6) + lin("#123a5e") * u[..., None] ** 0.6
    given = (cover * (0.3 + 0.65 * np.exp(-d / 170)) * smoothstep(-1.1, 0.0, run))[..., None]
    rcol = rl * np.array([0.88, 0.9, 0.97], np.float32) * 0.72 / (1 + 1.3 * rl) + body * 0.2   # darker, the pale most
    coat(P, water * 0.98, body * (1 - given) + rcol * given, 1.5, r, y0, g=lanes())
    coat(P, water * given[..., 0] * 0.4, rcol, 0.9, r, y0, g=lanes(), grain=0.7)
    piece = smoothstep(1.1, 1.8, dash) * (1 - 0.85 * given[..., 0]) * water * (0.25 + 0.6 * np.exp(-d / 160))
    coat(P, piece, refl * 0.7 + lin("#efe6d0") * 0.2, 0.7, r, y0, g=lanes(), grain=0.6)
    breeze = np.exp(-((yy - 1150 - 6 * noise.line1d(W, 300, r)[None, :]) / 14) ** 2)   # a breath of wind across it
    coat(P, water * breeze * 0.25 * smoothstep(-0.2, 1.0, run + dash), lin("#86aebf"), 0.7, r, y0, g=lanes(), grain=0.8)
    bright = smoothstep(0.15, 0.45, refl.mean(-1)) * (1 - cover)
    sunago(P, (water * np.exp(-d / 45) * bright * 0.5).astype(np.float32), r, y0, 0, rate=0.02, big=0.03)


def torii(P, r, shore):
    """The vermilion gate of the shrine standing in the water off the point, its pillars braced by short
    posts, and its reflection, a few broken strokes of the same cinnabar."""
    cx = 2112.0
    base = shore[int(cx)] + 7
    hgt, half = 50.0, 23.0
    ss = 4
    w, h = int(4 * half) + 10, int(2.4 * hgt) + 10
    ox, oy = cx - w / 2, base - hgt - 6
    red, blk = Image.new("L", (w * ss, h * ss), 0), Image.new("L", (w * ss, h * ss), 0)
    dr, db = ImageDraw.Draw(red), ImageDraw.Draw(blk)
    S = lambda x, y: ((x - ox) * ss, (y - oy) * ss)
    box = lambda d, xa, ya, xb, yb, wb=0.0: d.polygon([S(xa, ya), S(xb, ya), S(xb + wb, yb), S(xa - wb, yb)], fill=255)
    for sgn in (-1, 1):
        px = cx + sgn * 14
        box(dr, px - 2.4, base - hgt + 8, px + 2.4, base, 0.5)
        for k in (-1, 1):
            box(dr, px + k * 6 - 1.3, base - 17, px + k * 6 + 1.3, base, 0.1)
        box(dr, px - 7.5, base - 17, px + 7.5, base - 14.5)
    box(dr, cx - 21, base - hgt + 18, cx + 21, base - hgt + 21.5)          # the tie beam
    box(dr, cx - 2, base - hgt + 8, cx + 2, base - hgt + 18)
    t = np.linspace(-1, 1, 21)
    lip = base - hgt + 4 - 3.2 * t ** 4
    dr.polygon([S(cx + 27 * a, y) for a, y in zip(t, lip)] + [S(cx + 27 * a, y + 4.5) for a, y in zip(t[::-1], lip[::-1])], fill=255)
    db.polygon([S(cx + 28.5 * a, y - 2.8) for a, y in zip(t, lip)] + [S(cx + 28.5 * a, y + 0.6) for a, y in zip(t[::-1], lip[::-1])], fill=255)
    rm = np.asarray(red.resize((w, h), Image.BOX), np.float32) / 255
    bm = np.asarray(blk.resize((w, h), Image.BOX), np.float32) / 255
    y0, x0 = int(oy), int(ox)
    coat(P, rm * 0.98, lin("#d6402a"), 0.8, r, y0, x0, thick=0.6)
    coat(P, bm * 0.95, lin("#1d1a18"), 0.8, r, y0, x0)
    by = int(base - oy)
    refl = np.zeros_like(rm)
    span = min(h - by, by)
    refl[by:by + span] = rm[by - span:by][::-1]
    refl = ndimage.gaussian_filter(refl, (1.6, 0.5))
    rip = smoothstep(-0.4, 0.6, streaks(refl.shape, 1.2, 40, r) + 0.4)
    coat(P, refl * rip * 0.6, lin("#b8452e"), 0.9, r, y0, x0)


def ship(P, r):
    """One of the sightseeing ships dressed as galleons that ply the lake, coming down it toward the
    shrine, broadside on: its hull laid in ink, then cinnabar, with a band of indigo at the water and rows
    of windows on bands of shell white; gold along its rails, on its high carved stern and at its
    figurehead; three white masts with their sails furled on the yards, the shrouds and stays in thin ink,
    a red pennant at each masthead. Beneath it its reflection, broken lane by lane; behind it the wake,
    laid in short level strokes of shell white along the lanes, broad and bright under the stern and
    thinning away."""
    L, xs, yw, ss = 88.0, 1056.0, 1116.0, 4                    # the hull's length; its stern at the waterline
    x0, y0 = int(xs - 0.1 * L), int(yw - 0.95 * L)
    w, h = int(1.42 * L), int(yw) - y0 + 3
    names = ("hull", "navy", "white", "gold", "spar", "top", "rig", "flag")
    ims = {k: Image.new("L", (w * ss, h * ss), 0) for k in names}
    dr = {k: ImageDraw.Draw(v) for k, v in ims.items()}
    X = lambda u: xs + np.asarray(u) * L - x0                  # u along the hull, v up it, in hull lengths
    Y = lambda v: yw - np.asarray(v) * L - y0
    sheer = lambda u: 0.02 * (2 * np.asarray(u) - 1) ** 2      # the decks rise toward the ends

    def shape(k, pts, j=0.18):
        dr[k].polygon([((X(u) + r.normal(0, j)) * ss, (Y(v) + r.normal(0, j)) * ss) for u, v in pts], fill=255)

    def band(k, u0, u1, v0, v1):
        us = np.linspace(u0, u1, 11)
        shape(k, list(zip(us, v1 + sheer(us))) + list(zip(us[::-1], (v0 + sheer(us))[::-1])))

    def ribbon(k, us, vs, wd):
        """A stroke through the points, `wd` px wide at each, its line wavering a little."""
        px, py = X(us), Y(vs)
        dx, dy = np.gradient(px), np.gradient(py)
        n = np.hypot(dx, dy) + 1e-6
        nx, ny = -dy / n, dx / n
        o = np.cumsum(r.normal(0, 0.07, len(px)))
        px, py, hw = px + nx * (o - o.mean()), py + ny * (o - o.mean()), np.asarray(wd) / 2
        dr[k].polygon(list(zip((px + nx * hw) * ss, (py + ny * hw) * ss))
                      + list(zip((px - nx * hw)[::-1] * ss, (py - ny * hw)[::-1] * ss)), fill=255)

    def touch(k, u, v, rx, ry, th=0.0):
        dr[k].polygon([(x * ss, y * ss) for x, y in blot(X(u), Y(v), rx, ry, th, r, 10, 0.15)], fill=255)

    def rope(u0, v0, u1, v1, sag=0.012):
        t = np.linspace(0, 1, 9)
        px, py = X(u0 + (u1 - u0) * t), Y(v0 + (v1 - v0) * t - sag * 4 * t * (1 - t))
        for i in range(8):
            if r.random() < 0.85:
                dr["rig"].line([(px[i] * ss, py[i] * ss), (px[i + 1] * ss, py[i + 1] * ss)], fill=255, width=2)

    def row(u0, u1, step, v):
        """Points along a deck, `step` apart, a little irregular, a few left out. -> (x, y) in patch px"""
        u = np.arange(u0, u1, step)
        u = (u + r.normal(0, 0.12 * step, len(u)))[r.random(len(u)) > 0.08]
        return np.stack([X(u), Y(v + sheer(u)) + r.normal(0, 0.2, len(u))], 1)

    shape("hull", [(0.035, 0.0), (0.955, 0.0), (0.99, 0.05), (1.015, 0.12), (1.05, 0.19), (1.0, 0.215),
                   (0.985, 0.262), (0.86, 0.257), (0.86, 0.226), (0.5, 0.21), (0.2, 0.228), (0.2, 0.315),
                   (0.015, 0.352), (-0.016, 0.33), (-0.018, 0.16), (0.004, 0.07)])
    band("navy", -0.05, 1.1, -0.02, 0.045)
    band("white", 0.03, 0.95, 0.056, 0.093)
    band("white", 0.21, 0.85, 0.137, 0.174)
    shape("white", [(0.0, 0.24), (0.19, 0.238), (0.19, 0.276), (0.002, 0.282)])
    shape("white", [(0.87, 0.17), (0.98, 0.172), (0.98, 0.2), (0.87, 0.2)])
    band("gold", 0.03, 0.97, 0.124, 0.134)
    band("gold", 0.2, 0.86, 0.204, 0.226)
    band("gold", 0.86, 0.985, 0.244, 0.262)
    ribbon("gold", np.linspace(-0.012, 0.2, 6), np.linspace(0.344, 0.31, 6), np.full(6, 1.0))
    ribbon("gold", np.linspace(0.0, 0.2, 6), np.linspace(0.372, 0.336, 6), np.full(6, 0.7))
    for u, v in ((-0.006, 0.3), (-0.012, 0.26), (-0.014, 0.21), (-0.01, 0.16), (0.004, 0.1), (0.05, 0.225),
                 (0.1, 0.228), (0.15, 0.226), (0.04, 0.3), (0.12, 0.29)):   # the carving on the stern
        touch("gold", u, v, r.uniform(0.6, 1.1), r.uniform(0.5, 0.9), r.uniform(0, 3))
    rope(0.008, 0.36, 0.006, 0.395, 0.0)
    touch("gold", 0.006, 0.4, 1.1, 1.4)                         # its lantern
    touch("gold", 1.035, 0.175, 1.4, 3.2, 0.5)                  # the figurehead
    masts = ((0.22, 0.228, 0.66, (0.42, 0.1, 0.024), (0.55, 0.07, 0.018)),
             (0.5, 0.21, 0.86, (0.45, 0.15, 0.03), (0.61, 0.11, 0.024), (0.75, 0.075, 0.018)),
             (0.79, 0.225, 0.8, (0.43, 0.135, 0.028), (0.585, 0.1, 0.022), (0.7, 0.068, 0.017)))
    for um, vb, vt, *yards in masts:
        vs = np.linspace(vb, vt, 8)
        ribbon("spar", um - 0.012 * (vs - vb), vs, np.linspace(1.7, 0.8, 8))
        vy = yards[0][0]
        shape("top", [(um - 0.042, vy + 0.012), (um + 0.042, vy + 0.012), (um + 0.048, vy + 0.042),
                      (um - 0.048, vy + 0.042)], 0.1)
        band("gold", um - 0.044, um + 0.044, vy + 0.012, vy + 0.02)
        for vy, half, th in yards:                              # each yard with its sail furled on it
            uc = um - 0.012 * (vy - vb)
            ribbon("spar", [uc - half - 0.015, uc, uc + half + 0.015], [vy, vy + 0.003, vy], [0.6, 0.9, 0.6])
            touch("spar", uc, vy - 0.006, half * L, th * L / 2, r.normal(0, 0.02))
        for du in (-0.075, -0.05, -0.028, 0.03, 0.052):        # the shrouds
            rope(um + du, 0.215, um - 0.006 * np.sign(du), yards[0][0] + 0.04, 0.004)
        for du in (-0.04, 0.04):
            rope(um + du, yards[0][0] + 0.04, um, yards[-1][0], 0.003)
        t = np.linspace(0, 1, 7)                                # the pennant, streaming aft
        ribbon("flag", um - 0.012 * (vt - vb) - 0.15 * t * (0.7 if um > 0.3 else 0.5),
               vt - 0.006 - 0.012 * t + 0.008 * np.sin(5 * t + r.uniform(0, 6)), 1.7 * (1 - t) + 0.3)
    ribbon("spar", np.linspace(0.97, 1.22, 6), np.linspace(0.235, 0.33, 6), np.linspace(1.3, 0.6, 6))
    for a, b in (((0.79, 0.455), (1.0, 0.25)), ((0.79, 0.7), (1.12, 0.3)), ((0.79, 0.8), (1.22, 0.33)),
                 ((0.5, 0.75), (0.79, 0.46)), ((0.5, 0.48), (0.79, 0.26)), ((0.22, 0.6), (0.5, 0.45)),
                 ((0.22, 0.45), (0.5, 0.25)), ((0.5, 0.85), (0.38, 0.226)), ((0.79, 0.79), (0.69, 0.23)),
                 ((0.22, 0.65), (0.06, 0.33))):                 # the stays
        rope(*a, *b)
    m = {k: np.asarray(v.resize((w, h), Image.BOX), np.float32) / 255 for k, v in ims.items()}
    hull = m["hull"]
    v = (yw - rows(y0, y0 + h)) / L
    sl = np.s_[y0:y0 + h, x0:x0 + w]
    one = np.ones((1, w, 1), np.float32)
    coat(P, np.clip(1.5 * ndimage.gaussian_filter(hull, 0.6), 0, 1) * 0.95, lin("#1c1716"), 1.2, r, y0, x0)  # ink
    brushed = lambda: 1.3 * streaks((h, w), 1.0, 9, r) + 0.5 * noise.field((h, w), 0.8, r)   # a small flat brush, level
    coat(P, hull * 0.9, lin("#a83024") * (0.8 + 0.35 * smoothstep(0.05, 0.3, v))[..., None] * one, 0.9, r, y0, x0,
         g=brushed())
    pts = sample(hull, smoothstep(0.03, 0.12, v) * one[..., 0], int(hull.sum() / 6), r)
    c, t = marks((h, w), pts, r, 0.9, tones=r.uniform(0, 1, len(pts)))
    lay(P, c * hull, t, lin("#bc3a2a"), lin("#de5638"), 0.8, r, y0, x0, 0.7)    # touches of brighter cinnabar
    coat(P, m["navy"] * hull * 0.95, lin("#1e2a48"), 1.0, r, y0, x0)
    coat(P, m["white"] * hull * 0.82, lin("#ece3cf"), 0.6, r, y0, x0, thick=0.2, g=brushed())
    pts = np.concatenate([row(0.055, 0.93, 0.036, 0.0745), row(0.225, 0.83, 0.034, 0.1555),
                          row(0.02, 0.17, 0.04, 0.236), row(0.885, 0.97, 0.04, 0.166)])
    c, t = marks((h, w), pts, r, 0.75, spread=0.15, tones=r.uniform(0, 1, len(pts)) ** 2)
    lay(P, c * hull, t, lin("#23262f"), lin("#56606c"), 0.8, r, y0, x0, 0.9)    # the windows, dotted in
    pts = np.concatenate([row(0.06, 0.94, 0.07, 0.11), row(0.24, 0.84, 0.06, 0.19)])
    c, t = marks((h, w), pts, r, 0.55, spread=0.2, tones=r.uniform(0, 1, len(pts)))
    lay(P, c * hull, t, GOLD[1], GOLD[3], 0.7, r, y0, x0, 0.85, 0.4)          # and its gilt ornament
    coat(P, m["top"] * 0.95, lin("#b8382a"), 0.8, r, y0, x0)
    coat(P, m["spar"] * 0.95, lin("#ebe3d0") * (1 + 0.06 * noise.field((h, w), 2.0, r))[..., None], 0.55, r, y0, x0,
         thick=0.3)
    coat(P, m["rig"] * 0.7, lin("#25211f"), 0.6, r, y0, x0)
    coat(P, m["flag"] * 0.95, lin("#d8432e"), 0.7, r, y0, x0)
    coat(P, m["gold"] * 0.95, ramp(0.62 + 0.12 * noise.field((h, w), 1.5, r)), 0.7, r, y0, x0, thick=0.6)

    iw = int(yw) - y0                                          # the reflection: row k up given back k down
    a = np.clip(hull + m["spar"] + m["top"] + m["flag"] + 0.4 * m["rig"], 0, 1)[iw::-1]
    col = P.rgb[sl][iw::-1]
    n = len(a)
    yr = np.arange(n, dtype=np.float32) + int(yw)
    bh = 1.4 + 8.5 * np.clip((yr - VPY) / (SILL - VPY), 0, 1) ** 1.15
    lane = np.floor(np.cumsum(1 / bh) + r.uniform(0, 1)).astype(int)
    shift = r.normal(0, 1, lane.max() + 1)[lane] * (1.0 + 0.06 * np.arange(n))
    gy, gx = np.mgrid[0:n, 0:w].astype(np.float32)
    gx = gx - shift[:, None] - 0.6 * streaks((n, w), 1.0, 24, r)
    a = ndimage.map_coordinates(a, [gy, gx], order=1, mode="constant")
    col = np.stack([ndimage.map_coordinates(col[..., c], [gy, gx], order=1, mode="nearest") for c in range(3)], -1)
    sa = smear(a, 2)
    col = smear(col * a[..., None], 2) / np.maximum(sa, 1e-3)[..., None]
    fade = np.exp(-np.arange(n) / (0.5 * L))[:, None]
    rip = smoothstep(-0.4, 0.8, streaks((n, w), 1.2, 22, r) + 0.2)
    whole = 0.5 * np.exp(-np.arange(n) / 6.0)[:, None]      # close under the hull the water breaks it less
    coat(P, sa * np.maximum(rip, whole) * fade * 0.8, col * np.array([0.55, 0.6, 0.74], np.float32), 0.9, r,
         int(yw), x0)

    fx0, fy0, fw, fh = int(xs - 4.0 * L), int(yw - 14), int(5.1 * L), 28   # the wake
    foam = Image.new("L", (fw * ss, fh * ss), 0)
    df = ImageDraw.Draw(foam)
    lh = 1.4 + 8.5 * ((yw - VPY) / (SILL - VPY)) ** 1.15        # how tall a lane is here
    snap = lambda y: yw + lh * np.round((y - yw) / lh)

    def dash(x, y, ln, th, fill=1.0):
        pts = blot(x, y, ln / 2, th / 2, r.normal(0, 0.01), r, 14, 0.15)
        df.polygon([((px - fx0) * ss, (py - fy0) * ss) for px, py in pts], fill=int(255 * np.clip(fill, 0, 1)))

    for _ in range(38):                                        # the water churned white under the stern
        d = 0.6 * L * r.random() ** 1.6
        dash(xs + 0.01 * L - d - r.uniform(0, 3), yw + lh * r.integers(-1, 2) + r.normal(0, 0.35), r.uniform(2, 7),
             r.uniform(0.9, 1.7), 1 - 0.7 * d / L)
    d = 0.3 * L
    while d < 3.4 * L:                                         # its two arms drawn out behind, parting and fading
        ln = r.uniform(3, 9) * (1 + d / L)
        f = np.exp(-d / (2.0 * L)) * r.uniform(0.35, 0.9)
        k = int(np.round(r.choice([-1, 1]) * d / (1.6 * L) + r.normal(0, 0.35)))
        dash(xs - d - ln / 2, snap(yw - 0.016 * d) + lh * k + r.normal(0, 0.25), ln, 0.5 + 0.6 * f, f)
        d += r.uniform(3, 9)
    xb = xs + 0.97 * L
    dash(xb - 1.5, yw - 0.5, 4.5, 1.6)                          # the bow wave
    for k, (dx, ln, th, f) in enumerate(((7, 6, 1.1, 0.85), (15, 7, 0.9, 0.55), (25, 8, 0.8, 0.3))):
        dash(xb - dx, yw + lh * (k + 1) + r.normal(0, 0.2), ln, th, f)
    for _ in range(5):
        dash(xs + L * r.uniform(0.4, 0.92), yw + 0.2, r.uniform(3, 7), 0.8, 0.6)
    f = np.asarray(foam.resize((fw, fh), Image.BOX), np.float32) / 255
    coat(P, f * 0.95, lin("#f2ede2"), 0.5, r, fy0, fx0, thick=0.3)


def bush(P, cx, ty, half, kind, r, ybot):
    """One crown in the garden, seen from above: a heap of rounded lumps, each lit on its upper left, the
    nearer lumps over the farther and the body under them in their shade. It goes on as a Nihonga painter
    builds foliage: the whole heap laid in its darkest colour, a little inside its edge; then touches of
    a shade colour over all of it; then the middle colour where any light reaches; then smaller, brighter
    touches where the light falls full on the lumps; last a few of the brightest at their tops, shaped as
    leaves; and a few of the darkest again deep in the shade, the dark between the sprays. Every touch has
    its own colour, drifting toward the next kind's, and those at the edge make it."""
    xa, xb = int(np.clip(cx - 1.2 * half, 0, W)), int(np.clip(cx + 1.2 * half, 0, W))
    ya = int(max(ty - 0.3 * half, HEAD))
    if xb - xa < 12 or ybot - ya < 12:
        return
    h, w = ybot - ya, xb - xa
    yy, xx = np.mgrid[ya:ybot, xa:xb].astype(np.float32)
    u = (xx - cx) / half
    yt = ty + 0.5 * half * u ** 2 + 0.12 * half
    env = ((np.abs(u) < 1 + 0.07 * noise.line1d(w, 25, r)[None, :]) & (yy > yt)).astype(np.float32)
    light = np.clip(0.3 - 0.25 * (yy - yt) / (0.6 * half), 0.02, 0.3) * env
    lumps = []
    for _ in range(r.integers(4, 8)):
        lx = cx + r.uniform(-0.78, 0.78) * half
        rx = half * r.uniform(0.28, 0.46)
        ry = rx * r.uniform(0.62, 0.82)
        ly = ty + 0.5 * half * ((lx - cx) / half) ** 2 + ry * r.uniform(0.55, 1.0)
        lumps.append((ly, lx, rx, ry, r.uniform(-0.08, 0.08), r.uniform(0, 6.3, 3)))
    for ly, lx, rx, ry, dl, ph in sorted(lumps):
        dx, dy = (xx - lx) / rx, (yy - ly) / ry
        rr, th = np.hypot(dx, dy), np.arctan2(dy, dx)
        inside = rr < 1 + 0.09 * np.sin(3 * th + ph[0]) + 0.06 * np.sin(5 * th + ph[1]) + 0.04 * np.sin(8 * th + ph[2])
        env[inside] = 1
        light[inside] = np.clip(0.62 + dl - 0.42 * (0.55 * dx + 0.83 * dy) + 0.1 * (1 - rr), 0, 1)[inside]
    whole = np.clip(0.8 - 0.3 * u - 0.55 * (yy - ty) / (0.9 * half), 0, 1)       # the crown as one, lit from the left
    light = np.clip(ndimage.gaussian_filter(0.55 * light + 0.45 * whole * env, 2.0)
                    + 0.07 * noise.field((h, w), 14, r), 0, 1)
    soft = ndimage.gaussian_filter(env, 1.0)
    shades, drift = CROWN[kind]
    c, a = [lin(x) for x in shades], [lin(x) for x in CROWN[drift][0]]
    coat(P, smoothstep(0.3, 0.8, soft) * 0.96, c[0], 1.5, r, ya, xa)
    base = float(np.clip(half * 0.05, 4.0, 8.0))
    for lvl, wgt, size, dens, grade, load, thick, fm in (
            (1, env * (1.1 - light), 1.15, 1.2, 1.4, 0.75, 0.0, "dab"),
            (2, env * smoothstep(0.3, 0.6, light), 1.0, 1.3, 1.2, 0.9, 0.0, "dab"),
            (3, env * smoothstep(0.55, 0.82, light), 0.85, 1.0, 1.0, 0.9, 0.3, "dab"),
            (4, env * smoothstep(0.78, 1.0, light), 0.6, 0.7, 0.8, 0.9, 0.5, "dab" if kind == 3 else "leaf")):
        s = base * size
        pts = sample(env, wgt, int(dens * wgt.sum() / (3.0 * s * s)), r)
        cov, ton = marks((h, w), pts, r, s, fm, tones=r.uniform(0, 1, len(pts)))
        lay(P, cov, ton, c[lvl] * 0.88, a[lvl] * 1.04, grade, r, ya, xa, load, thick)
    hole = env * smoothstep(0.4, 0.15, light)                  # the dark between the sprays, deep in the crown
    pts = sample(env, hole, int(0.3 * hole.sum() / (3.0 * base * base)), r)
    cov, _ = marks((h, w), pts, r, base * 0.7)
    coat(P, cov * 0.8, c[0] * 0.8, 1.4, r, ya, xa)


def garden(P, r):
    """The garden falling away under the window, seen from above: one canopy of maples in cinnabar,
    vermilion and gold among dark evergreen shrubs, crown over crown, the farther half hidden by the
    nearer (see `bush`)."""
    hue = noise.line1d(W + 400, 260, r)                       # neighbours share their colours, in long patches
    rise = noise.line1d(W + 400, 340, r)
    trees = []
    x = -120.0
    while x < W + 120:
        half = r.uniform(70, 190)
        z = hue[int(np.clip(x + 200, 0, W + 399))] + 0.6 * r.normal()
        k = 3 if z < -0.75 else 2 if z < -0.15 else 1 if z < 0.35 else 0
        ty = 1366 - 55 * rise[int(np.clip(x + 200, 0, W + 399))] + r.normal(0, 26) - 0.12 * half
        trees.append((x, ty, half, k))
        x += r.uniform(0.45, 0.8) * half
    trees += [(3110, 1176, 260, 0), (2920, 1256, 160, 1), (150, 1262, 170, 3)]
    for cx, ty, half, k in sorted(trees, key=lambda t: t[1]):
        bush(P, cx, ty, half, k, r, SILL + 24)


def cedars(P, r):
    """Two tall cedars just outside on the left, dark against the gold. From each trunk short branches
    spread at every height, up and out near the top, level lower down, and along each the foliage gathers
    in dense clumps, so the crown is narrow and ragged, with the sky showing through it here and there.
    Each clump is tufts of the brush pointing out from its heart and mostly upward, as the sprays of a
    cedar turn up at their tips: first all of them in ink and the deepest malachite, then on the upper
    left of each clump, where the sun reaches, smaller tufts of a lighter green, and at the tips of those
    in the full sun the palest."""
    x1, y0 = 720, HEAD - 70
    h = SILL + 24 - y0
    yy = rows(y0, y0 + h)
    xx = XS[None, :x1]
    bark = np.zeros((h, x1), np.float32)
    edge = np.zeros((h, x1), np.float32)
    pts, ang, tone = [], [], []
    for tx, tt, half in ((175, HEAD - 150, 160), (405, 430, 120)):
        wt = 3.5 + 7 * np.clip((yy - tt) / 900, 0, 1)
        tr = smoothstep(1, -1, np.abs(xx - tx) - wt) * smoothstep(tt + 30, tt + 60, yy)
        bark = np.maximum(bark, tr)
        edge = np.maximum(edge, tr * np.exp(-((xx - tx + wt - 1.5) / 1.4) ** 2))
        y = tt + r.uniform(2, 6)
        while y < SILL + 40:
            f = np.clip((y - tt) / (SILL - tt), 0, 1)
            wd = half * (0.08 + 0.92 * f ** 0.75)
            first = r.choice([-1, 1])
            for sd in ((first, -first) if r.random() < 0.5 else (first,)):
                L = wd * r.uniform(0.3, 1.05)
                a = -0.5 + 0.55 * f + r.normal(0, 0.15)        # up and out near the top, level lower down
                for s in np.sort(r.uniform(0.15, 1.0, r.integers(2, 5))):
                    cx = tx + sd * L * s * np.cos(a)
                    cy = y + L * s * np.sin(a) + 3 * s
                    rc = r.uniform(7, 15) * (0.75 + 0.6 * f) * (1.15 - 0.3 * s)
                    n = int(rc * rc / 4) + 4
                    dx, dy = r.normal(0, 0.62, n) * rc, r.normal(0, 0.42, n) * rc
                    pts += list(zip(cx + dx, cy + dy - y0))
                    ang += list(np.arctan2(dy - 0.9 * rc, dx + 0.3 * sd * rc) + r.normal(0, 0.3, n))
                    tone += list(0.5 - 0.42 * (0.55 * dx + 0.83 * dy) / rc + 0.12 * (cx < tx) + 0.08 * (tx > 300)
                                 - 0.1 * (cx > tx + 20) + r.normal(0, 0.1, n))
            y += r.uniform(5, 12) * (0.8 + 0.4 * f)
    pts, ang, tone = np.array(pts), np.array(ang), np.array(tone)
    coat(P, ndimage.gaussian_filter(bark, 0.7), lin("#2b1a12") * (1 + 0.25 * streaks((h, x1), 1.5, 60, r))[..., None],
         1.4, r, y0)
    coat(P, edge * 0.6, lin("#6a4630"), 1.0, r, y0)
    cov, _ = marks((h, x1), pts, r, 7.0, "tuft", ang=ang)
    body = smoothstep(0.35, 0.6, ndimage.gaussian_filter(cov, 2.0))
    coat(P, np.maximum(cov, body) * 0.97, lin("#0a1511"), 1.5, r, y0)
    for lo, size, grade, load, c0, c1, thick in ((0.58, 6.0, 1.4, 0.8, "#10221a", "#162d1e", 0.0),
                                                 (0.8, 5.0, 1.0, 0.85, "#213c2c", "#2f5238", 0.2),
                                                 (0.95, 4.0, 0.7, 0.8, "#446c4c", "#5f875c", 0.4)):
        k = tone > lo
        c, t = marks((h, x1), pts[k] + np.array([-0.8, -1.0]), r, size, "tuft", ang=ang[k],
                     tones=r.uniform(0, 1, k.sum()))
        lay(P, c, t, lin(c0), lin(c1), grade, r, y0, 0, load, thick)


# --- the room --------------------------------------------------------------------------------------------------

def ceiling(P, r):
    """The ceiling: slats of wood running from the window toward us, so they all draw in to the eye; their
    under-faces catch the lantern's light, warm near it and dying away into the dark at the corners, the
    gaps between them dark. Each slat is laid with one long stroke of a dry brush, so its edges waver a
    little and break up, and the paint lies in lanes along it. Ink, burnt sienna and earth, a little gold
    dust where the light is."""
    yy, xx = np.mgrid[0:TOP, 0:W].astype(np.float32)
    q = VPX + (xx - VPX) * (VPY - TOP) / (VPY - yy)          # where this ray meets the lintel
    sp = 74.0
    q = (q + 1.8 * ndimage.map_coordinates(noise.stretched((TOP + 2, 3500), 24, 80, r), [yy, np.clip(q + 150, 0, 3499)],
                                           order=1)
         + 0.45 * noise.field((TOP, W), 5, r))
    t = (q - VPX) / sp
    f = t - np.floor(t)
    side = np.clip(0.04 * np.abs(q - VPX) / sp, 0, 0.6)
    wsl = (0.2 * (1 + 0.2 * r.normal(0, 1, 200)))[(np.floor(t).astype(int) + 100) % 200]
    bot = smoothstep(0.0, 0.025, f) * smoothstep(wsl, wsl - 0.025, f)
    ff = np.where(q < VPX, f - wsl, 1.0 - f)                  # the side turned toward us
    sid = smoothstep(-0.01, 0.01, ff) * smoothstep(side + 0.01, side - 0.01, ff) * (1 - bot)
    glow = np.exp(-((xx - VPX) / 760) ** 2 - ((yy - 150) / 320) ** 2)
    grain = ndimage.map_coordinates(noise.stretched((TOP + 40, 900), 2.5, 240, r), [yy, (t * sp) % 899], order=1)
    coat(P, np.ones((TOP, W), np.float32), lin("#150f0c") * (1 + 0.15 * noise.field((TOP, W), 40, r))[..., None],
         0.6, r)
    each = (1 + 0.14 * r.normal(0, 1, 200))[(np.floor(t).astype(int) + 100) % 200]
    slat = (lin("#36261b") * (0.55 + 1.15 * glow + 0.25 * smoothstep(120, TOP, yy))[..., None]
            * (each * (1 + 0.2 * grain))[..., None])
    coat(P, bot * 0.88, np.clip(slat, 0, 1), 0.7, r, g=1.2 * grain + 0.4 * noise.field((TOP, W), 2, r))
    coat(P, sid * 0.9, lin("#22170f") * (0.7 + 0.8 * glow)[..., None], 0.6, r, g=grain)
    coat(P, bot * glow ** 1.5 * 0.4 * smoothstep(-0.8, 0.6, grain), lin("#a8743f"), 0.5, r)
    sunago(P, (bot * glow ** 2 * 0.3).astype(np.float32), r, 0, 0, rate=0.02)


def lantern(P, r):
    """The lantern over the middle of the window: a six-sided frame of dark wood with paper panes glowing
    in it, three faces showing, and under it a cluster of three small lamps."""
    cx, ss = VPX, 3
    x0, w, h = int(cx - 130), 260, TOP + 4
    paper_, frame_ = Image.new("L", (w * ss, h * ss), 0), Image.new("L", (w * ss, h * ss), 0)
    dp, df = ImageDraw.Draw(paper_), ImageDraw.Draw(frame_)
    S = lambda x, y: ((x - x0) * ss, y * ss)
    df.rectangle([S(cx - 1.5, 0), S(cx + 1.5, 64)], fill=255)                                   # the cord
    for xa, xb in ((cx - 98, cx - 50), (cx - 50, cx + 50), (cx + 50, cx + 98)):
        df.rectangle([S(xa, 70), S(xb, 240)], fill=255)
        dp.rectangle([S(xa + 6, 79), S(xb - 6, 231)], fill=255)
    df.polygon([S(cx - 110, 62), S(cx + 110, 62), S(cx + 98, 74), S(cx - 98, 74)], fill=255)   # the cap
    df.polygon([S(cx - 104, 236), S(cx + 104, 236), S(cx + 92, 248), S(cx - 92, 248)], fill=255)
    for lx, ly, lw in ((cx - 52, 250, 32), (cx + 52, 250, 32), (cx, 256, 40)):                  # the small lamps
        df.rectangle([S(lx - lw, ly), S(lx + lw, ly + 34)], fill=255)
        dp.rectangle([S(lx - lw + 5, ly + 5), S(lx + lw - 5, ly + 29)], fill=255)
    bars = Image.new("L", (w * ss, h * ss), 0)
    dbar = ImageDraw.Draw(bars)
    for x in (cx - 74, cx, cx + 74):
        dbar.rectangle([S(x - 1.8, 72), S(x + 1.8, 240)], fill=255)
    for y in (126, 184):
        dbar.rectangle([S(cx - 98, y - 1.8), S(cx + 98, y + 1.8)], fill=255)
    for lx in (cx - 52, cx + 52, cx):
        dbar.rectangle([S(lx - 1.5, 250), S(lx + 1.5, 290)], fill=255)
    m = lambda im: np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255
    pm, fm, bm = m(paper_), m(frame_), m(bars)
    pm = pm * (1 - bm)
    fm = np.clip(fm - pm, 0, 1)
    xx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
    face = (1 - 0.2 * smoothstep(48, 52, np.abs(xx - cx))) * np.ones((h, 1))
    coat(P, fm * 0.98, lin("#1e1510"), 1.0, r, 0, x0)
    coat(P, pm * 0.98, lin("#f4dfaa") * face[..., None], 0.55, r, 0, x0, thick=0.3)
    coat(P, pm * 0.3 * (1 - face + 0.2), lin("#e2ac58"), 0.6, r, 0, x0)


def frame(P, r):
    """The window's own frame: the lintel above, two slim mullions and the low sill, dark wood laid in ink
    and burnt umber with a brush that left its lanes along the grain, never quite straight; their edges
    catch the light from the lake, and the polished top of the sill gives back the view, the maples' colours
    lying in it along the grain."""
    h = HEAD + 4 - TOP
    yy = rows(TOP, HEAD + 4)
    low = HEAD + 0.8 * noise.line1d(W, 140, r) + 0.4 * noise.line1d(W, 20, r)
    m = smoothstep(0.7, -0.7, yy - low[None, :])
    tone = 1 + 0.1 * streaks((h, W), 2, 400, r) + 0.08 * noise.line1d(W, 300, r)[None, :]
    coat(P, m, lin("#16100d") * tone[..., None], 0.8, r, TOP)
    lit = np.exp(-((yy - low[None, :] + 2.0) / 1.3) ** 2) * smoothstep(-0.6, 0.4, noise.field((h, W), 30, r))
    coat(P, lit * m * 0.6, lin("#8c7254"), 0.6, r, TOP)
    for mx in MULL:
        sgn = 1 if mx < VPX else -1                           # the side turned toward the middle shows
        y0, y1, x0, ww = HEAD, SILL, mx - 26, 52
        n = y1 - y0
        xx = np.arange(x0, x0 + ww, dtype=np.float32)[None, :] + (0.7 * noise.line1d(n, 120, r)
                                                                  + 0.3 * noise.line1d(n, 15, r))[:, None]
        front = smoothstep(-0.8, 0.8, xx - (mx - 13)) * smoothstep(0.8, -0.8, xx - (mx + 13))
        sx0 = mx + 13 if sgn > 0 else mx - 24
        side = smoothstep(-0.8, 0.8, xx - sx0) * smoothstep(0.8, -0.8, xx - (sx0 + 11)) * (1 - front)
        along = (1 + 0.12 * noise.line1d(n, 150, r))[:, None]
        coat(P, front, lin("#16110e") * (along * (1 + 0.1 * streaks((n, ww), 1.5, 300, r)))[..., None], 0.8, r, y0, x0)
        coat(P, side, lin("#4e4236") * (along * (1 + 0.15 * streaks((n, ww), 2, 200, r)))[..., None], 0.7, r, y0, x0)
        glint = np.exp(-((xx - (sx0 + (10 if sgn > 0 else 1))) / 1.0) ** 2) * smoothstep(-0.8, 0.3, noise.field((n, ww), 40, r))
        coat(P, glint * 0.6, lin("#cdb690"), 0.6, r, y0, x0)
    y0, y1 = SILL - 2, 1480
    yy = rows(y0, y1)
    one = np.ones((1, W), np.float32)
    e = 0.6 * noise.line1d(W, 160, r)[None, :]
    coat(P, smoothstep(SILL - 2, SILL, yy + e) * smoothstep(SILL + 13, SILL + 11, yy + e) * one, lin("#16110e"), 0.8,
         r, y0)
    board = smoothstep(SILL + 11, SILL + 13, yy + e) * smoothstep(SILL + 44, SILL + 42, yy + e)
    grain = streaks((y1 - y0, W), 1.6, 500, r)
    shine = (0.5 + 0.5 * np.exp(-((XS - 1700) / 1100) ** 2)[None, :]) * (1 + 0.2 * streaks((y1 - y0, W), 6, 300, r))
    coat(P, board * one, lin("#45372b") * ((0.6 + 0.7 * shine) * (1 + 0.08 * grain))[..., None], 0.7, r, y0)
    k = np.clip(yy + e - (SILL + 12), 0, None)                 # how far into the board from its far edge
    sy = np.clip(SILL - 3 - 2.0 * k, 0, H - 1) * one
    sx = np.clip(XS[None, :] + 4 * streaks((y1 - y0, W), 2, 300, r), 0, W - 1)
    refl = np.stack([ndimage.map_coordinates(P.rgb[..., c], [sy, sx], order=1) for c in range(3)], -1)
    refl = ndimage.gaussian_filter(refl, (1.0, 5.0, 0))
    coat(P, board * np.exp(-k / 16) * 0.55 * smoothstep(-1.0, 0.5, grain + 0.4), refl * 0.75 + lin("#3a2c20") * 0.2,
         0.6, r, y0)
    gleam = smoothstep(SILL + 24, SILL + 13, yy + e) * smoothstep(-0.3, 0.9, streaks((y1 - y0, W), 3, 220, r) + 0.6)
    coat(P, board * gleam * 0.45, lin("#c4a676"), 0.6, r, y0)
    coat(P, smoothstep(SILL + 42, SILL + 44, yy + e) * one, lin("#120e0c"), 0.8, r, y0)


def below(P, r):
    """Under the window: the wall in the shade of the sill, warmed where the light comes off the board; the
    floor before it, laid in strokes running toward the window, where the light of the window lies warm
    along the wall and dies toward us; and on it a long low bench of black leather, six cushions in a row,
    its top below the eye and taking the warm light of the window along its far edge and its rounded front,
    its shadow cast toward us."""
    y0 = 1480
    h = H - y0
    yy, xx = np.mgrid[y0:H, 0:W].astype(np.float32)
    P.rgb[y0:] = glaze(P.rgb[y0:], np.full((h, W), 2.6, np.float32), SUMI)      # an ink ground under it all
    glow = np.exp(-((xx - 1700) / 1400) ** 2)
    wall = smoothstep(FLOOR + 1.5, FLOOR - 1.5, yy)
    under = (np.exp(-(yy - y0) / 30) * (0.4 + 0.6 * glow))[..., None]
    lane = hake((h, W), r, along=900, width=12, amp=(0.07, 0.05))[..., None]
    coat(P, wall * 0.97, (lin("#1c1410") * (1 - 0.5 * under) + lin("#4a3220") * 0.5 * under) * lane, 0.9, r, y0)
    fl = smoothstep(FLOOR - 1.5, FLOOR + 1.5, yy)
    ray = (xx - VPX) / np.maximum(yy - VPY, 1.0)               # which line toward the eye's point
    run = ndimage.map_coordinates(noise.stretched((h + 4, 3800), 6, 160, r),
                                  [yy - y0, np.clip(ray * 650 + 1900, 0, 3799)], order=1)
    pool = (np.exp(-(yy - FLOOR) / 55) * (0.35 + 0.65 * glow) * (1 + 0.08 * run))[..., None]
    coat(P, fl * 0.97, (lin("#2a160e") * (1 - 0.6 * pool) + lin("#7a4c2c") * 0.6 * pool) * (1 + 0.03 * run)[..., None],
         1.0, r, y0)
    xa, xb, yf, yb, n = 860.0, 2340.0, FLOOR + 22.0, FLOOR + 88.0, 6
    yt = VPY + (yf - VPY) * 0.93                               # the far edge of the seat
    hand = 1.2 * noise.line1d(W, 90, r)[None, :] + 0.5 * noise.line1d(W, 12, r)[None, :]   # no edge of it ruled
    yfe, yte = yf + hand, yt + 0.8 * hand
    lx = VPX + (xa - VPX) * (VPY - yy) / (VPY - yf) + 1.5 * noise.line1d(h, 40, r)[:, None]
    rx = VPX + (xb - VPX) * (VPY - yy) / (VPY - yf) + 1.5 * noise.line1d(h, 40, r)[:, None]
    seat = ndimage.gaussian_filter(((xx > lx) & (xx < rx) & (yy > yte) & (yy < yfe + 2)).astype(np.float32), 1.2)
    front = ndimage.gaussian_filter(((xx > xa + hand[0, 0]) & (xx < xb + hand[0, -1]) & (yy > yfe - 2)
                                     & (yy < yb + 0.8 * hand)).astype(np.float32), 1.2)
    front = np.clip(front - 0.5 * seat, 0, 1)
    cu = np.where(yy < yf, (xx - VPX) * (VPY - yf) / (VPY - yy) + VPX - xa, xx - xa) / ((xb - xa) / n)
    cu = cu + 0.004 * noise.field((h, W), 30, r)
    seam = np.exp(-((cu - np.round(cu)) * (xb - xa) / n / 2.2) ** 2) * (cu > 0.3) * (cu < n - 0.3)
    swell = np.sin(np.pi * np.clip(cu - np.floor(cu), 0, 1)) ** 0.5
    shadow = (smoothstep(yb - 4, yb + 6, yy) * smoothstep(yb + 46, yb, yy + 6 * noise.field((h, W), 20, r))
              * smoothstep(xa - 50, xa + 30, xx) * smoothstep(xb + 50, xb - 30, xx))
    coat(P, shadow * 0.6, lin("#0e0806"), 1.0, r, y0)
    sheen = ((0.7 * np.exp(-(yy - yt) / 9) + 0.2) * (0.45 + 0.55 * swell) * (1 - seam)
             * (0.75 + 0.25 * np.tanh(noise.field((h, W), 90, r))))[..., None]
    coat(P, seat * 0.97, (lin("#141318") * (1 - sheen) + lin("#7c6a52") * sheen) * (1 - 0.4 * seam)[..., None], 0.5, r,
         y0)
    coat(P, front, lin("#0c0b0d") * ((1 + 1.0 * np.exp(-(yy - yf) / 12) * (0.6 + 0.4 * swell)) * (0.8 + 0.3 * swell)
                                     * (1 - 0.5 * seam))[..., None], 0.6, r, y0)
    lip = np.exp(-((yy - yf - 1) / 4) ** 2) * np.clip(seat + front, 0, 1) * (0.5 + 0.5 * swell) * (1 - seam)
    coat(P, lip * 0.6, lin("#9a8262"), 0.5, r, y0)


def finish(P, r):
    """The surface as a whole under a raking gallery light: crystals of the coarser grades glinting or dark
    among the finer, and the paint standing a little proud where it was laid thick."""
    s = (H, W)
    k = np.clip((P.k - 0.5) / 1.3, 0, 1)
    rgb = P.rgb * (1 + (0.03 + 0.07 * k) * noise.field(s, 0.7, r) + (0.02 + 0.04 * k) * noise.field(s, 1.8, r))[..., None]
    spark = smoothstep(2.6, 3.3, noise.field(s, 0.8, r)) * (0.1 + 0.9 * k)
    speck = smoothstep(2.5, 3.2, noise.field(s, 0.9, r)) * (0.1 + 0.9 * k)
    rgb = rgb + spark[..., None] * (0.7 * rgb + 0.04)
    rgb *= (1 - 0.35 * speck)[..., None]
    hg = ndimage.gaussian_filter(P.h, 1.2) + 0.06 * noise.field(s, 1.0, r)
    gy, gx = np.gradient(hg)
    rgb *= np.clip(1 + 0.5 * (0.6 * gx + 0.8 * gy), 0.75, 1.3)[..., None]
    return rgb


def paint(seed=1988):
    rs = [noise.rng(seed + k) for k in range(14)]
    sheet = paper.washi((H, W), seed, tint="#e6dac2", margin=0, fibre_density=0.6)
    P = SimpleNamespace(rgb=sheet.color.copy(), k=np.zeros((H, W), np.float32), h=np.zeros((H, W), np.float32),
                        tooth=sheet.tooth, land=np.zeros((H, W), np.float32))
    shore = skyline([(0, 1012), (1690, 1013), (1900, 1022), (2100, 1031), (2400, 1046), (2700, 1062),
                     (3000, 1080), (3200, 1092)], rs[0], 0.6, 40)
    sky(P, rs[1])
    fuji(P, rs[2])
    ridges(P, rs[3])
    mountain(P, rs[4])
    headland(P, rs[5], shore)
    lake(P, rs[6], shore)
    torii(P, rs[7], shore)
    ship(P, noise.rng(seed + 50))
    garden(P, rs[8])
    cedars(P, rs[9])
    ceiling(P, rs[10])
    lantern(P, rs[11])
    frame(P, rs[12])
    below(P, rs[13])
    return finish(P, noise.rng(seed + 99))
