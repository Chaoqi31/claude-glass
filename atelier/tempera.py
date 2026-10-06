"""Tempera on a gessoed panel, as the painters of Gondar made an icon, and what candles and hands
have done to it since.

The panel is cut from one plank and adzed flat; the face is hollowed with a chisel, leaving a
raised border of the same wood all round, so that the picture lies in a shallow tray. Over it goes
gesso, chalk in hide glue brushed on coat after coat and scraped smooth when dry, until the face is
white and close. On that the painter lays tempera: earths and minerals ground in water and bound
with glue or egg. It dries as it leaves the brush, so a flat field goes on as a mat of short strokes
laid this way and that, two or three coats deep, and the gesso glows faintly where they thinned.
Modelling is hatched: fine strokes side by side in a darker colour along the form. Last, every shape
is drawn round in black with a pointed brush, each long line in a few confident pulls.

The plank goes on moving with the seasons after the gesso has set hard. The gesso opens into a net
of cracks whose islands are longer across the grain than along it, and dirt finds the cracks. Now
and then an island lets go and shows the white gesso; at the edges and along a split in the plank
the gesso goes too, down to the wood. Beetles bore the wood and leave round exit holes. Hands and
lips rub the paint thin where they touch most, and the smoke of candles and incense lays a brown
film over everything, thickest where nothing wiped it.
Everything here returns maps (H, W) or colours for a work to combine.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import brush, noise, relief
from .color import lin, pigment
from .etching import hand
from .noise import smoothstep

WOOD = lin("#76573a")       # the plank where it shows, darkened by age and handling
GESSO = lin("#e8e0cb")      # chalk in glue, a little warm
SOOT = pigment("#7f6345")   # candle and incense smoke: a brown film, warmer where it lies thin


def tray(shape, border, r, bevel=22.0, rise=24.0, corner=14.0):
    """A plank hollowed out to leave a raised border `border` px wide standing `rise` px above the
    field, with a chiselled slope `bevel` px wide between them. The cuts are by hand, so the border
    is never quite even, and the outer edge is rounded at the corners and chipped where the panel
    was knocked. -> height (px), outline (px inside the panel's edge), depth (how far in from the top
    of the slope, in slope widths: below 0 on the border, 1 at the foot of the slope, above 1 in the
    field)"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    qx, qy = np.abs(xx - (w - 1) / 2) - (w / 2 - corner), np.abs(yy - (h - 1) / 2) - (h / 2 - corner)
    outline = corner - np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) - np.minimum(np.maximum(qx, qy), 0) - 1.5
    outline -= 7 * smoothstep(1.8, 2.8, noise.field(shape, 22, r)) + 1.2 * noise.field(shape, 6, r)
    inset = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy])
    edge = border + 3.5 * noise.field(shape, 170, r) + 0.7 * noise.field(shape, 8, r)
    depth = (inset - edge) / bevel
    height = rise * (1 - np.clip(depth, 0, 1)) * smoothstep(-1.0, 9.0, outline)    # the outer arris worn round
    return ndimage.gaussian_filter(height, 1.2).astype(np.float32), outline.astype(np.float32), depth.astype(np.float32)


def plank(shape, r):
    """The bare wood where everything over it has gone: close-grained, the grain running up and
    down, grey-brown with age, the soft wood between the late rings a little sunken. -> albedo,
    height (px)"""
    g = relief.woodgrain(shape, r, along=1.0, scale=1.6)
    fine = noise.stretched(shape, 1.0, 40, r)
    tone = noise.fbm(shape, 260, r, octaves=3)
    col = WOOD * (1 + 0.16 * g + 0.08 * fine + 0.1 * tone)[..., None]
    return col.astype(np.float32), (0.7 * g + 0.3 * fine).astype(np.float32)


def gesso(shape, r):
    """The gesso: warm white, faintly clouded where a coat went on thicker, with the long chatter of
    the scraper across it and pinholes where air was caught in the glue. -> albedo, height (px)"""
    h, w = shape
    cloud = noise.fbm(shape, 140, r, octaves=4)
    chatter = noise.stretched((w, h), 2.0, 260, r).T
    pin = smoothstep(2.7, 3.2, noise.field(shape, 1.2, r))
    col = GESSO * (1 + 0.02 * cloud + 0.008 * chatter - 0.15 * pin)[..., None]
    return col.astype(np.float32), (0.35 * cloud + 0.06 * chatter - 0.5 * pin).astype(np.float32)


def coat(mask, r, angle=0.0, size=(5.0, 10.0), thin=0.035, mottle=0.008):
    """A flat field of tempera over `mask` (0..1): short strokes turned to `angle` and a second coat
    across them, thinner here and there in clouds, so the gesso glows faintly through. -> coverage"""
    h, w = mask.shape
    n = int(np.hypot(h, w)) + 4
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = 0.0
    for a, wt in ((angle, 0.65), (angle + 1.1, 0.35)):
        f = noise.stretched((n, n), *size, r)
        c, sn = np.cos(a), np.sin(a)
        u, v = (xx - w / 2) * c + (yy - h / 2) * sn + n / 2, -(xx - w / 2) * sn + (yy - h / 2) * c + n / 2
        s = s + wt * ndimage.map_coordinates(f, [u, v], order=1)
    cloud = noise.field(mask.shape, 25, r) + 0.5 * noise.field(mask.shape, 90, r)
    return mask * (1 - np.clip(thin * s + mottle * cloud, 0, 0.9))


def contour(shape, paths, r, width=2.6, swell=0.8, alpha=2.2, wander=0.35, run=650.0, corner=0.75):
    """The black line of a Gondar master, in lamp black and glue with a pointed brush. Each run of
    line is one pull: the brush sets down thin, presses into the full stroke, swells as the line
    turns and lifts away thin again, and it carries less paint as it goes, so the end of a long
    pull goes dry and breaks on the tooth of the gesso. A long contour is drawn in several pulls,
    each lapping the last a little; at a sharp corner the hand lifts and sets down anew.
    -> coverage"""
    acc, dry = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for P in paths:
        for C in _pulls(np.asarray(P, np.float64), run, corner, r):
            C = hand(C, r, wander=wander, tremor=0.08, step=0.5)
            k = len(C)
            if k < 4:
                continue
            g = np.gradient(C, axis=0)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
            L = s[-1]
            g /= np.hypot(g[:, 0], g[:, 1])[:, None] + 1e-9
            turn = np.abs(ndimage.gaussian_filter1d(np.gradient(np.unwrap(np.arctan2(g[:, 1], g[:, 0]))), 10)) / 0.5
            ends = (0.3 + 0.7 * smoothstep(0, min(25, L / 4), s)) * (0.25 + 0.75 * smoothstep(L, L - min(60, L / 3), s))
            press = 1 + 0.18 * noise.line1d(k, 160, r)
            wd = width * ends * press * (0.8 + swell * smoothstep(0.004, 0.035, turn))
            lanes = max(2, int(np.ceil(wd.max() * 1.5 / 0.6)))
            off = (np.arange(lanes) / (lanes - 1) - 0.5)[:, None] * wd[None]
            X, Y = C[None, :, 0] - g[None, :, 1] * off, C[None, :, 1] + g[None, :, 0] * off
            V = np.broadcast_to(alpha * 0.5 * wd * 1.5 / lanes * (1 - 0.2 * s / max(L, 400)), X.shape)
            D = V * smoothstep(0.55 * L, L, s) * min(1.0, L / 500)
            for into, val in ((acc, V), (dry, D)):
                buf, o = brush.splat(X.ravel(), Y.ravel(), np.ascontiguousarray(val).ravel())
                brush.paste(into, buf, o)
    cov = 1 - np.exp(-ndimage.gaussian_filter(acc, 0.45))
    starved = np.clip(dry / (acc + 1e-6), 0, 1) * smoothstep(-0.6, 1.4, noise.field(shape, 1.1, r))
    return (cov * (1 - 0.8 * starved)).astype(np.float32)


def _pulls(P, run, corner, r):
    """A contour cut where the hand would lift: at sharp corners, and where a run grows longer than
    one load of the brush carries; each pull starts a little before the last one ended."""
    if len(P) < 3:
        return [P]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    a = np.searchsorted(s, s - 4.0).clip(0, len(P) - 1)
    b = np.searchsorted(s, s + 4.0).clip(0, len(P) - 1)
    u, v = P - P[a], P[b] - P
    ang = np.abs(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1)))
    peak = (ang > corner) & (ang >= ndimage.maximum_filter1d(ang, 9))
    cuts = [0] + [i for i in np.flatnonzero(peak) if 6 < s[i] < s[-1] - 6] + [len(P) - 1]
    out = []
    for i, j in zip(cuts[:-1], cuts[1:]):
        n = max(1, int(np.ceil((s[j] - s[i]) / run)))
        e = np.interp(np.linspace(0, 1, n + 1) + np.r_[0, r.uniform(-0.12, 0.12, n - 1) / n, 0], [0, 1], [s[i], s[j]])
        for x0, x1 in zip(e[:-1], e[1:]):
            m = (s >= x0 - (6 if x0 > s[i] else 0)) & (s <= x1)
            if m.sum() > 2:
                out.append(P[m])
    return out


def craquelure(shape, r, cell=60.0, aspect=1.8, width=0.9, keep=0.8):
    """The net the gesso opened into: islands about `cell` px along the grain (up and down) and
    `aspect` times that across it, their edges wandering, the cracks of uneven width, and only part
    of the net open. -> crack (0..1), island (index map)"""
    h, w = shape
    cw = cell * aspect
    n = int((h + 2 * cell) * (w + 2 * cw) / (cell * cw))
    seeds = np.stack([r.uniform(-cw, w + cw, n) / aspect, r.uniform(-cell, h + cell, n)], 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wx, wy = (0.16 * cell * noise.fbm(shape, cell * 0.7, r, octaves=3) + 0.7 * noise.field(shape, 3, r) for _ in range(2))
    p = np.stack([((xx + wx) / aspect).ravel(), (yy + wy).ravel()], 1)
    d, k = cKDTree(seeds).query(p, k=2, workers=-1)
    # distance to the edge of the island in true pixels: the gap between the two nearest seeds
    # over how fast that gap grows across the edge
    u1, u2 = ((p - seeds[k[:, i]]) / np.maximum(d[:, i:i + 1], 1e-6) for i in (0, 1))
    g = u2 - u1
    rate = np.hypot(g[:, 0] / aspect, g[:, 1])
    dist = ((d[:, 1] - d[:, 0]) / np.maximum(rate, 1e-3)).reshape(shape)
    wd = width * (0.55 + 0.45 * smoothstep(-1.0, 1.5, noise.field(shape, cell * 1.5, r)))
    opened = smoothstep(-0.2, 0.4, noise.fbm(shape, cell * 1.3, r, octaves=3) + 2 * (keep - 0.5))
    crack = smoothstep(wd + 0.7, wd - 0.5, dist) * opened
    return crack.astype(np.float32), k[:, 0].reshape(shape)


def flakes(island, odds, r, scale=40.0, rough=0.2):
    """Paint that has let go where the odds (0..1, a map) are high. It lifts in torn patches about
    `scale` px across; where a patch takes most of an island of the net the break runs along the
    cracks, elsewhere it tears straight across the islands. Crumbs too small to see as flakes have
    been swept off with the dust. -> 0..1"""
    torn = smoothstep(-0.08, 0.08, odds * (1 + 0.6 * noise.fbm(island.shape, scale, r, octaves=2)) - 0.5)
    n = int(island.max()) + 1
    share = (np.bincount(island.ravel(), torn.ravel(), minlength=n) / np.maximum(np.bincount(island.ravel(), minlength=n), 1))[island]
    f = ndimage.gaussian_filter(0.4 * torn + 0.6 * share, 1.2)
    f = smoothstep(0.46, 0.54, f + rough * noise.field(island.shape, 1.2, r) * 4 * f * (1 - f))    # frayed only where an edge runs
    lab, _ = ndimage.label(f > 0.5)
    whole = ndimage.binary_dilation((np.bincount(lab.ravel()) >= 60)[lab] & (lab > 0), iterations=2)
    return (f * whole).astype(np.float32)


def split(shape, x, length, r, width=7.0):
    """The plank split along its grain from the lower edge: `width` px open at the edge, closing as
    it runs up `length` px and wandering only as the grain does. -> gap (0..1)"""
    h, w = shape
    n = max(16, int(length / 3))
    s = np.linspace(0, 1, n)
    X = x + 9 * noise.line1d(n, n / 3, r) * s + 1.2 * noise.line1d(n, 4, r)
    Y = h + 6 - s * (length + 6)
    half = 0.5 * width * (1 - s) ** 0.8 + 0.25
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lo = int(max(0, Y.min() - 4))
    sub = (slice(lo, h), slice(int(max(0, X.min() - width - 4)), int(min(w, X.max() + width + 5))))
    off = np.abs(xx[sub] - np.interp(yy[sub], Y[::-1], X[::-1]))
    half_y = np.interp(yy[sub], Y[::-1], half[::-1], left=0.0)
    gap = np.zeros(shape, np.float32)
    gap[sub] = smoothstep(half_y + 0.6, half_y - 0.6, off) * (yy[sub] > Y.min())
    return gap


def holes(shape, r, n, odds):
    """Exit holes of wood-boring beetles, a millimetre or two across and never quite round, where
    the odds (0..1, a map) favour them. -> hole (0..1)"""
    h, w = shape
    out = np.zeros(shape, np.float32)
    xs, ys = r.uniform(4, w - 4, 400 * n), r.uniform(4, h - 4, 400 * n)
    keep = r.random(len(xs)) < odds[ys.astype(int), xs.astype(int)]
    for x, y in zip(xs[keep][:n], ys[keep][:n]):
        rad = float(np.clip(r.lognormal(np.log(2.6), 0.25), 1.6, 4.5))
        x0, y0 = int(x - rad - 3), int(y - rad - 3)
        yy, xx = np.mgrid[y0:y0 + int(2 * rad + 7), x0:x0 + int(2 * rad + 7)].astype(np.float32)
        a = np.arctan2(yy - y, xx - x)
        rr = rad * (1 + 0.12 * np.sin(2 * a + r.uniform(0, 6)) + 0.06 * np.sin(3 * a + r.uniform(0, 6)))
        disc = smoothstep(rr + 0.6, rr - 0.6, np.hypot(xx - x, yy - y))
        sl = (slice(max(y0, 0), min(y0 + disc.shape[0], h)), slice(max(x0, 0), min(x0 + disc.shape[1], w)))
        out[sl] = np.maximum(out[sl], disc[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0])
    return out


def smoke(shape, r, top=0.5, bottom=0.05, candles=()):
    """The film candles and incense leave. The smoke rose past the panel and gathered under its
    top, so the film thickens upward; over each place where a candle stood it climbed in a column
    that widens and wavers as it goes. It lies in clouds, and ran a little downward where damp
    once softened it. -> density"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    up = 1 - yy / h
    sway = 70 * noise.fbm(shape, 320, r, octaves=3)
    plume = sum(np.exp(-0.5 * ((xx + sway - x) / (70 + 0.4 * (h - yy))) ** 2) for x in candles)
    cloud = noise.fbm(shape, 240, r, octaves=4)
    runs = noise.stretched(shape, 40, 500, r)
    film = (bottom + (top - bottom) * up ** 1.7 + 0.16 * plume * up ** 0.8) * (1 + 0.22 * cloud) + 0.03 * runs
    return np.clip(film, 0, None).astype(np.float32)


def worn(amount, height, r):
    """Where hands and lips rubbed (`amount`, 0..1), the paint thins: a little everywhere the rubbing
    reached, and through to the gesso on its high points where it reached most, so a much-kissed
    place goes pale and soft before it is bare. -> fraction lost"""
    tooth = ndimage.gaussian_filter(height, 1.5)
    tooth = (tooth - tooth.mean()) / (tooth.std() + 1e-6) + 0.5 * noise.field(height.shape, 3, r)
    thin = 0.5 * smoothstep(0.0, 1.0, amount)
    through = smoothstep(2.3 - 2.2 * amount, 3.1 - 2.2 * amount, tooth) * smoothstep(0.05, 0.4, amount)
    return np.clip(thin + (1 - thin) * through, 0, 1)


def _shifted(a, dx, dy):
    """`a` moved by whole pixels, the edge rows and columns repeated into the gap."""
    out = np.roll(a, (dy, dx), (0, 1))
    if dy > 0:
        out[:dy] = a[:1]
    elif dy < 0:
        out[dy:] = a[-1:]
    if dx > 0:
        out[:, :dx] = a[:, :1]
    elif dx < 0:
        out[:, dx:] = a[:, -1:]
    return out


def light(height, key=(-0.45, -0.56, 0.8), fill=0.4, reach=48, soft=5.0):
    """How a museum photographs a panel: two broad lights at 45 degrees, the stronger from the upper
    left, so the relief shows without glare and the raised border throws a soft shadow into the
    field below and to the right of it. -> shading (H,W), 1 on the flat"""
    gy, gx = np.gradient(height)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L1 = np.asarray(key, np.float32) / np.linalg.norm(key)
    L2 = L1 * np.array([-1, 1, 1], np.float32)
    flat = lambda L: np.clip(nrm @ L, 0, None) / L[2]
    toward = L1[:2] / np.linalg.norm(L1[:2])
    rise = L1[2] / np.linalg.norm(L1[:2])
    occ = np.zeros_like(height)
    for t in range(1, reach):
        dx, dy = int(round(-toward[0] * t)), int(round(-toward[1] * t))
        occ = np.maximum(occ, _shifted(height, dx, dy) - height - t * rise)
    shadow = ndimage.gaussian_filter(smoothstep(0.0, 4.0, occ), soft)
    return ((1 - fill) * flat(L1) * (1 - 0.6 * shadow) + fill * flat(L2)).astype(np.float32)
