"""A Hill Town at Dusk. Gouache on illustration board.

Mary Blair came to the Disney studio in Burbank in 1940, and the next year went south with Walt on
the studio's tour of South America that became Saludos Amigos. What she brought back was colour
nobody else at the studio would have put together: magenta against chartreuse, turquoise against
orange, and black to make them ring. For Cinderella, Alice in Wonderland and Peter Pan she painted
the films before they were drawn, in small concept paintings that told the animators what each place
felt like: the castle pale against a blue-violet dusk behind black trees and columns, London black
against a deep blue night with a few windows lit, Never Land glowing green out of black and magenta.
They are gouache on illustration board, everything flat, trees made into plumes, flames and
lollipops, towns stacked like blocks, and the light put where she wanted it.

This board was taped down at its edges and painted from the back to the front. The sky went on
first with a broad flat brush in long strokes from side to side, almost black at the top and mixed
a little paler and warmer at every reload, through ultramarine and violet to magenta and orange
along the hills; a second coat went over it in strokes that run out dry, and a few long clouds were
dragged across it with a drier brush. The new moon is one curved stroke of a round brush pressed in
the middle and lifted to a point at each end; the stars are a few touches of the point and two
crossed strokes. The far hills are two bands of violet and indigo laid along their tops. The hill of
the town is chartreuse where the last light catches its top and emerald below, in strokes that follow
its curve. The castle went on in one pale mix, tower by tower from the back, each with its shaded
side laid in lavender and a needle of a roof, and then a few windows, most of them lit. The houses
under it are blocks of colour with dark roofs, the upper row in the orange, turquoise and magenta of
the last light, the lower in the violet and wine of the dusk with a gilt dome among them, and a
clipped hedge along the foot of each row. Last came the dark, in a black mixed with a little green:
the bank across the foot of the board, rising on the left to the umbrella pine and on the right in a
thicket to the tufted tree, every crown and bush laid round with the brush and its edge feathered
out in flicks of the point; then the long leaves and a few pale flowers on the bank. Gouache is
opaque but not even: every flat shows the strokes that laid it, thick where the
brush was reloaded, streaked where it ran thin, and where it ran out the colour underneath, or the
cream of the board, shows through.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from atelier import gouache, noise, plate
from atelier.color import lin
from atelier.paper import Sheet

TITLE = "A Hill Town at Dusk"
DATE = "2026"
MEDIUM = "Gouache on illustration board"
AFTER = ("Mary Blair, concept paintings for Walt Disney Studios: Saludos Amigos, 1942; Cinderella, 1950; "
         "Alice in Wonderland, 1951; Peter Pan, 1953; and the designs for It's a Small World, 1964")
ROOM = "Paper and Water"
YEAR = 1951
PLACE = "Burbank"
REGION = "Americas"
NOTE = ("A pale castle on a chartreuse hill against a dusk that goes from ultramarine to magenta and orange, "
        "the town stacked under it in a few blocks of colour with a window lit here and there. A dark bank, "
        "an umbrella pine and a tufted tree, black against the glow, frame the view.")

H, W = 2640, 3200
BX0, BY0, BX1, BY1 = 70, 62, 3130, 2574          # the board on the wall
PX0, PY0, PX1, PY1 = 196, 180, 3004, 2420        # the picture, inside the tape
PW, PH = PX1 - PX0, PY1 - PY0
ss = noise.smoothstep


def X(u):
    return PX0 + u * PW


def Y(v):
    return PY0 + v * PH


def C(c):
    return lin(c).astype(np.float64)


# ---------------------------------------------------------------- the board

def norm(a, lo=1, hi=99):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1).astype(np.float32)


def make_board(r):
    """Illustration board: a cream drawing paper mounted on card, with a close, even tooth. The tape
    kept a strip of it clean round the picture; outside the tape it has yellowed and taken the
    handling of the studio, and there are tack holes in its corners. -> (colour, tooth, alpha, tape)"""
    tooth = norm(0.55 * noise.field((H, W), 1.0, r) + 0.4 * noise.field((H, W), 2.4, r)
                 + 0.25 * noise.field((H, W), 6.0, r) + 0.1 * noise.field((H, W), 22, r))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cut = np.minimum.reduce([xx - BX0 - 0.0007 * (yy - BY0), BX1 - xx + 0.0004 * (yy - BY0),
                             yy - BY0 + 0.0005 * (xx - BX0), BY1 - yy])
    alpha = ss(-0.8, 0.8, cut)
    e = np.minimum.reduce([xx - PX0, PX1 - xx, yy - PY0, PY1 - yy])
    wander = 0.6 * noise.field((H, W), 260, r) + 0.25 * noise.field((H, W), 9, r)
    creep = 2.2 * ss(1.6, 2.6, noise.field((H, W), 26, r))       # paint crept under the tape here and there
    tape = ss(-0.7, 0.7, e + wander + creep).astype(np.float32)
    strip = ss(-50, -46, e + 1.5 * noise.field((H, W), 300, r))  # what the tape covered
    age = np.clip(0.55 + 0.35 * noise.fbm((H, W), 420, r, octaves=4) + 0.4 * ss(120, 0, cut), 0, 1.4)
    age *= 1 - strip
    colour = C("#efe7d3")[None, None] * (1 + 0.012 * noise.fbm((H, W), 300, r, octaves=5)
                                          + 0.008 * (tooth - 0.5))[..., None]
    colour *= 1 - age[..., None] * np.array([0.025, 0.05, 0.12])
    grime = np.exp(-((e + 47) / 2.0) ** 2) * (0.5 + 0.5 * ss(-1, 1, noise.field((H, W), 40, r)))
    colour *= 1 - 0.06 * grime[..., None]
    for x, y in ((BX0 + 34, BY0 + 30), (BX1 - 31, BY0 + 36), (BX0 + 30, BY1 - 34), (BX1 - 37, BY1 - 29)):
        for _ in range(int(r.integers(1, 4))):
            hx, hy = x + r.normal(0, 4), y + r.normal(0, 4)
            d = np.hypot(xx[int(hy) - 12:int(hy) + 12, int(hx) - 12:int(hx) + 12] - hx,
                         yy[int(hy) - 12:int(hy) + 12, int(hx) - 12:int(hx) + 12] - hy)
            hole = ss(2.6, 1.6, d) * 0.75 + 0.12 * np.exp(-((d - 3.4) / 1.4) ** 2)
            colour[int(hy) - 12:int(hy) + 12, int(hx) - 12:int(hx) + 12] *= (1 - hole)[..., None]
    return colour.astype(np.float64), tooth, alpha.astype(np.float32), tape


class Board:
    """The board on the drawing table: the paint on it so far, its tooth and the tape round the
    picture, and the bristles of the brushes, which every stroke borrows from."""

    def __init__(self, r):
        self.r = r
        self.img, self.tooth, self.alpha, self.tape = make_board(r)
        self.fine, self.clump = bristles(r)

    def lay(self, sl, al, col):
        a = (al * self.tape[sl])[..., None]
        self.img[sl] = self.img[sl] * (1 - a) + np.asarray(col) * a


def view(x0, y0, x1, y1, pad=6):
    """Slices of the box round (x0, y0)-(x1, y1), kept to the picture."""
    xa, xb = max(int(np.floor(x0)) - pad, PX0 - 8), min(int(np.ceil(x1)) + pad, PX1 + 8)
    ya, yb = max(int(np.floor(y0)) - pad, PY0 - 8), min(int(np.ceil(y1)) + pad, PY1 + 8)
    return None if xb <= xa + 1 or yb <= ya + 1 else (slice(ya, yb), slice(xa, xb))


def grid(sl):
    yy, xx = np.mgrid[sl[0], sl[1]]
    return yy.astype(np.float64), xx.astype(np.float64)


# ---------------------------------------------------------------- the brush

def bristles(r):
    """The marks of a flat brush in stroke space, rows along the stroke and columns across it: single
    bristles a pixel or two wide that come and go along it, and the bunches they gather into as the
    brush runs dry."""
    return noise.stretched((4096, 1024), 1.5, 60, r), noise.stretched((4096, 1024), 6.0, 240, r)


def look(T, a, c):
    """Read a bristle table at (along, across) px: nearest along, bilinear across."""
    na, nc = T.shape
    ai = a.astype(np.int64) % na
    c0 = np.floor(c)
    f = c - c0
    ci = c0.astype(np.int64) % nc
    return T[ai, ci] * (1 - f) + T[ai, (ci + 1) % nc] * f


def mark(bd, uu, ww, too, s, e, hw, d, load, spent, oa, oc, tails=0.5, grain=0.25, rough=1.5, wob=0.0, head=0.25,
         taper=0.0, soft=0.0):
    """One stroke of a flat brush `2 hw` px wide that set down at `s` and lifted at `e` along the
    axis `uu`, travelling toward `e` if `d` > 0: how much paint it left on each pixel `ww` px off its
    middle. It lands with a round heel, carries less paint the further it goes and breaks up on the
    tooth as it runs dry, first in its bunches of bristles and then bristle by bristle, which drag on
    past the lift in tails of uneven length. `taper` narrows it at both ends, as a stroke pressed in
    the middle; `soft` lets it down gently into wet paint, its bristles touching one after another.
    -> (coverage, streak, ridge): the streaks of the bristles in the paint, and the little ridge it
    leaves along its edges."""
    L = np.maximum(e - s, 1.0)
    ds = np.where(d > 0, uu - s, e - uu)
    de = L - ds
    t = np.clip(ds / L, 0, 1)
    B = look(bd.fine, ds + oa, ww + oc)
    G = look(bd.clump, ds + oa, ww + oc)
    half = hw * (1 + 0.035 * wob)
    if taper > 0:
        half = half * (1 - taper * (1 - np.sin(np.pi * t) ** 0.6))
    elif taper < 0:                         # a flick: pressed as it lands, lifted to a point
        half = half * (1 + taper * t ** 1.3)
    edge = np.abs(ww) + rough * (0.5 - too) * (0.4 + t) - 0.9 * G * t
    prof = ss(half + 0.9, half - 0.9, edge)
    land = head * hw * (1 - np.sqrt(np.clip(1 - (ww / hw) ** 2, 0, 1)))
    tail = tails * hw * ss(-0.3, 1.5, G + 0.3 * B)
    along = ss(-1.0, 1.2 + soft * hw, ds - land + soft * hw * (0.5 * G + 0.25 * B)) * ss(-1.0, 1.0, de + tail)
    ld = load * (1 - spent * t ** 1.6) * np.exp(np.minimum(de, 0) / (0.3 * hw + 1))
    cov = ss(0.32, 0.5, ld + 0.12 * B + 0.22 * G + grain * (too - 0.5))
    ridge = np.exp(-((half - np.abs(ww)) / 2.0) ** 2)
    return prof * along * cov, 0.6 * G + 0.4 * B, ridge


def fill(bd, sl, a, U, V, col, bw, length=(2.5, 6.0), lap=0.2, ulap=(0.15, 0.35), load=(1.0, 1.25), spent=0.3,
         streak=0.05, vary=0.03, tails=0.4, grain=0.25, rough=1.5, hide=1.0, flip=0.5, ridge=0.03, odds=1.0, soft=0.0):
    """Fill the region `a` (0..1 over the box `sl`) with strokes of a flat brush `bw` px wide, laid
    side by side in lanes along U and across V (px), each lane lapping the last a little and each
    stroke a few brush-widths long, lapping the one before it. `col` is one colour or a field the
    painter mixed toward as she went: each stroke carries the colour under its middle. With `odds`
    below 1 only so many of the strokes are laid: streaks dragged over a coat already down."""
    r = bd.r
    a = a * bd.tape[sl]
    on = a > 0.004
    if not on.any():
        return
    ao, Uo, Vo, too = a[on], U[on].astype(np.float64), V[on].astype(np.float64), bd.tooth[sl][on]
    pitch = bw * (1 - lap)
    v0 = Vo.min() - 0.5 * pitch
    nl = int((Vo.max() - v0) / pitch) + 2
    c = v0 + pitch * (np.arange(nl) + r.uniform(-0.12, 0.12, nl))
    hw = 0.5 * bw * r.uniform(0.97, 1.1, nl)
    u0, u1 = Uo.min() - 2 * bw, Uo.max() + 2 * bw
    S, E, K = [], [], []
    for k in range(nl):
        s = u0 - r.uniform(0, length[1] * bw)
        while s < u1:
            L = r.uniform(*length) * bw
            S.append(s)
            E.append(s + L)
            K.append(k)
            s += L * (1 - r.uniform(*ulap))
    S, E, K = np.array(S), np.array(E), np.array(K)
    n = len(S)
    D = np.where(r.random(n) < flip, -1.0, 1.0)
    L0 = r.uniform(*load, n) * (r.random(n) < odds)
    SP = np.clip(spent * r.uniform(0.5, 1.4, n), 0, 0.97)
    OA, OC = r.integers(0, 4096, n), r.integers(0, 1024, n)
    key = K * 1e7 + S
    kb = np.floor((Vo - v0) / pitch).astype(np.int64)
    if np.ndim(col) == 1:
        sc = np.tile(np.asarray(col, np.float64), (n, 1))
    else:
        k1, k2 = np.clip(kb, 0, nl - 1), np.clip(kb + 1, 0, nl - 1)
        kn = np.where(np.abs(Vo - c[k1]) <= np.abs(Vo - c[k2]), k1, k2)
        jn = np.clip(np.searchsorted(key, kn * 1e7 + Uo, "right") - 1, 0, n - 1)
        cf = col[on]
        wt = np.bincount(jn, ao, n)
        sc = np.stack([np.bincount(jn, ao * cf[:, i], n) for i in range(3)], 1) / np.maximum(wt, 1e-9)[:, None]
        own, idx = wt >= 1e-9, np.arange(n)          # a stroke under no pixel of its own takes its neighbour's mix
        nxt = np.minimum.accumulate(np.where(own, idx, n)[::-1])[::-1].clip(0, n - 1)
        prv = np.maximum.accumulate(np.where(own, idx, -1)).clip(0, n - 1)
        src = np.where(own[nxt] & (K[nxt] == K), nxt, np.where(own[prv] & (K[prv] == K), prv, -1))
        sc[~own] = np.where((src >= 0)[~own, None], sc[src.clip(0)][~own], cf.mean(0))
    sc = sc * (1 + vary * r.normal(0, 1, (n, 1))) * (1 + 0.4 * vary * r.normal(0, 1, (n, 3)))
    sk = streak * (1 + 2.5 * ss(0.06, 0.0, sc.mean(1)))      # in a dark the streaks show more
    NU = int(u1 - u0) + 2
    wob = noise.line1d(NU, 2.5 * bw, r, rows=max(nl, 2))
    ui = np.clip((Uo - u0).astype(np.int64), 0, NU - 1)
    out = bd.img[sl][on]
    for dk in (-1, 0, 1, 2):
        k = kb + dk
        near = (k >= 0) & (k < nl)
        k = np.clip(k, 0, nl - 1)
        near &= np.abs(Vo - c[k]) < hw[k] + 3
        idx = np.flatnonzero(near)
        if not len(idx):
            continue
        kk, uu = k[idx], Uo[idx]
        ww = Vo[idx] - c[kk]
        j = np.searchsorted(key, kk * 1e7 + uu, "right") - 1
        for jj in (j - 1, j):
            ok = (jj >= 0) & (K[np.clip(jj, 0, n - 1)] == kk)
            jj = np.clip(jj, 0, n - 1)
            al, st, rd = mark(bd, uu, ww, too[idx], S[jj], E[jj], hw[kk], D[jj], L0[jj], SP[jj], OA[jj], OC[jj],
                              tails, grain, rough, wob[kk, ui[idx]], soft=soft)
            al = al * ok * ao[idx] * hide
            cl = sc[jj] * ((1 + sk[jj] * st) * (1 - ridge * rd))[:, None]
            out[idx] = out[idx] * (1 - al[:, None]) + cl * al[:, None]
    img = bd.img[sl]
    img[on] = out


def axes(sl, angle):
    """Coordinates along and across strokes laid at `angle` (radians, image axes)."""
    yy, xx = grid(sl)
    c, s = np.cos(angle), np.sin(angle)
    return xx * c + yy * s, -xx * s + yy * c


def drag(bd, rows, col, tails=0.8, grain=0.35, rough=2.5, streak=0.06, head=0.15, taper=0.0, ridge=0.03, soft=0.0):
    """Single strokes laid one after another, rows (x0, y0, x1, y1, width, load, spent, bow): from
    (x0, y0) to (x1, y1), bowed by `bow` of their length; a load below 1 is a brush half dry."""
    r = bd.r
    col = np.asarray(col, np.float64)
    for x0, y0, x1, y1, wd, ld, sp, bow in rows:
        L = np.hypot(x1 - x0, y1 - y0)
        hw = wd / 2
        pad = hw + tails * hw + abs(bow) * L + 4
        sl = view(min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad, 0)
        if sl is None or L < 2:
            continue
        yy, xx = grid(sl)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        uu = (xx - x0) * ux + (yy - y0) * uy
        t = np.clip(uu / L, 0, 1)
        ww = -(xx - x0) * uy + (yy - y0) * ux - 4 * bow * L * t * (1 - t)
        al, st, rd = mark(bd, uu, ww, bd.tooth[sl], 0.0, L, hw, 1.0, ld, sp, int(r.integers(0, 4096)),
                          int(r.integers(0, 1024)), tails, grain, rough, head=head, taper=taper, soft=soft)
        cl = col * (1 + 0.03 * r.normal(0, 1)) * ((1 + streak * st) * (1 - ridge * rd))[..., None]
        bd.lay(sl, al, cl)


def raster(sl, polys, k=3):
    """Polygons [(N, 2) x, y] filled into the box, antialiased."""
    y0, x0 = sl[0].start, sl[1].start
    h, w = sl[0].stop - y0, sl[1].stop - x0
    im = Image.new("L", (w * k, h * k), 0)
    d = ImageDraw.Draw(im)
    for P in polys:
        d.polygon([((x - x0) * k, (y - y0) * k) for x, y in np.asarray(P, np.float64)], fill=255)
    return np.asarray(im, np.float32).reshape(h, k, w, k).mean((1, 3)) / 255


def edged(bd, sl, m, amp=1.3, scale=50.0, rough=0.5):
    """A shape as a loaded brush cuts it in: its edge wanders a little, slowly, and is never as clean
    as a rule; where the tooth of the board stands up the paint catches a hair further."""
    h, w = m.shape
    r = bd.r
    if amp > 0:
        f = lambda: noise.field((h, w), scale, r) + 0.35 * noise.field((h, w), max(1.6, scale / 7), r)
        m = noise.warp(m, amp * f(), amp * f())
    m = ndimage.gaussian_filter(m, 0.55) + rough * 0.2 * (bd.tooth[sl] - 0.5)
    return ss(0.38, 0.62, m)


def shape(bd, polys, col, bw, angle=np.pi / 2, amp=1.2, scale=50.0, **kw):
    """Cut a shape in with the brush and fill it with strokes laid at `angle`. -> (box, mask)"""
    P = np.vstack([np.asarray(p, np.float64) for p in polys])
    sl = view(P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max(), pad=8)
    if sl is None:
        return None, None
    m = edged(bd, sl, raster(sl, polys), amp, scale)
    U, V = axes(sl, angle)
    fill(bd, sl, m, U, V, C(col) if isinstance(col, str) else col, bw, **kw)
    return sl, m


def dabs(bd, pts, rad, cols, load=(1.0, 1.3), squash=(0.75, 0.95)):
    """Touches of a round brush, one at a time: each a little out of round and longer the way the
    brush came down, a shade deeper on the side it pressed; the last few before the brush was
    dipped again break on the tooth."""
    r = bd.r
    for (x, y), R, c in zip(pts, rad, cols):
        sl = view(x - R - 3, y - R - 3, x + R + 3, y + R + 3, pad=0)
        if sl is None:
            continue
        yy, xx = grid(sl)
        a0 = r.uniform(0, np.pi)
        dx, dy = xx - x, yy - y
        u, v = dx * np.cos(a0) + dy * np.sin(a0), -dx * np.sin(a0) + dy * np.cos(a0)
        th = np.arctan2(v, u)
        rho = np.hypot(u, v / r.uniform(*squash))
        p = r.uniform(0, 2 * np.pi, 3)
        rr = R * (1 + 0.07 * np.cos(2 * th + p[0]) + 0.05 * np.cos(3 * th + p[1]) + 0.03 * np.cos(5 * th + p[2]))
        ld = r.uniform(*load)
        cov = ss(0.3, 0.5, ld - 0.25 * (rho / R) ** 2 + 0.3 * (bd.tooth[sl] - 0.5))
        al = ss(rr + 0.7, rr - 0.7, rho) * cov
        cl = C(c) * (1 + 0.035 * r.normal()) * (1 - 0.05 * u / max(R, 1))[..., None]
        bd.lay(sl, al, cl)


def lines(bd, paths, col, width, wander=0.3, tremor=0.08, alpha=1.3):
    """Lines drawn with the point of a small round brush: poles, twigs, the rails of a fence."""
    P = np.vstack([np.asarray(p, np.float64)[:, :2] for p in paths])
    sl = view(P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max(), pad=int(2 * width + 10))
    if sl is None:
        return
    o = np.array([sl[1].start, sl[0].start], np.float64)
    shp = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    cov = gouache.line(shp, [np.asarray(p, np.float64)[:, :2] - o for p in paths], bd.r, width=width,
                       alpha=alpha, wander=wander, tremor=tremor)
    bd.lay(sl, cov, C(col))


# ---------------------------------------------------------------- shapes

def blob(cx, cy, rx, ry, r, lobes=0.06, n=120):
    """An oval pushed out into a few soft lobes, as a brush goes round a crown or a bush."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = int(r.integers(3, 6))
    rad = (1 + lobes * np.sin(k * th + r.uniform(0, 6)) + 0.5 * lobes * np.sin((k + 2) * th + r.uniform(0, 6))
           + 0.3 * lobes * np.sin((2 * k + 1) * th + r.uniform(0, 6)))
    return np.stack([cx + rx * rad * np.cos(th), cy + ry * rad * np.sin(th)], 1)


def skyline(points):
    """The top of a hill through control points (x, y), smooth between them."""
    p = np.asarray(points, np.float64)
    return PchipInterpolator(p[:, 0], p[:, 1], extrapolate=True)


def ridge(bd, pts, col, bw, r, below=None, stripes=(), streaks=(), amp=1.6, **kw):
    """A hill: everything under the skyline through `pts`, cut in along its top and laid in strokes
    that follow its curve; over that, streaks of another mix dragged along it with a drier brush
    (colour, odds), and bands of other colours (dv0, dv1, colour, brush) a set depth under the top, or a
    depth that changes along it (a function of x)."""
    f = skyline(pts)
    p = np.asarray(pts, np.float64)
    x0, x1 = max(PX0 - 8, p[:, 0].min()), min(PX1 + 8, p[:, 0].max())
    top = f(np.linspace(x0, x1, 400)).min() - 12
    sl = view(x0, top, x1, PY1 + 8 if below is None else below, pad=0)
    yy, xx = grid(sl)
    xs = np.arange(sl[1].start, sl[1].stop, dtype=np.float64)
    edge = f(xs) + amp * (noise.line1d(len(xs), 90, r) + 0.3 * noise.line1d(len(xs), 9, r))
    V = yy - edge[None, :]
    m = ss(-0.7, 0.7, V + 0.5 * (bd.tooth[sl] - 0.5))
    fill(bd, sl, m, xx, V, C(col), bw, **dict(dict(spent=0.2), **kw))
    for sc, odds in streaks:
        fill(bd, sl, m, xx, V, C(sc), 0.7 * bw, length=(3, 8), lap=0.1, load=(0.7, 1.0), spent=0.75, tails=1.0,
             grain=0.4, odds=odds, vary=0.02)
    for dv0, dv1, sc, sbw in stripes:
        e0, e1 = [(d(xs) if callable(d) else d) + 3 * noise.line1d(len(xs), 140, r) + 1.2 * noise.line1d(len(xs), 25, r)
                  for d in (dv0, dv1)]
        band = ss(-0.7, 0.7, V - e0[None, :]) * ss(-0.7, 0.7, e1[None, :] - V) * m
        fill(bd, sl, band, xx, V, C(sc), sbw, length=(3, 8), spent=0.4, grain=0.3)
    return f


def cone(x0, x1, y, h, lean=0.0, sag=0.18, eave=0.12):
    """A pointed roof over a round tower, its sides drawn in a little: a needle on a castle tower."""
    w = x1 - x0
    a, b = x0 - eave * w, x1 + eave * w
    t = np.linspace(0, 1, 24)
    ax, ay = (x0 + x1) / 2 + lean * w, y - h
    left = np.stack([a + (ax - a) * t + sag * w * np.sin(np.pi * t) * 0.5, y + (ay - y) * t], 1)
    right = np.stack([ax + (b - ax) * t - sag * w * np.sin(np.pi * t) * 0.5, ay + (y - ay) * t], 1)
    return np.vstack([left, right, [(b, y + 0.05 * h), (a, y + 0.05 * h)]])


def gable(x0, x1, y, h, eave=0.08, skew=0.0):
    w = x1 - x0
    return np.array([(x0 - eave * w, y + 2), ((x0 + x1) / 2 + skew * w, y - h), (x1 + eave * w, y + 2)])


def dome(x0, x1, y, h):
    t = np.linspace(0, np.pi, 30)
    cx, w = (x0 + x1) / 2, (x1 - x0) * 0.54
    return np.vstack([np.stack([cx - w * np.cos(t), y - h * np.sin(t)], 1), [(cx + w, y + 3), (cx - w, y + 3)]])


def shed(x0, x1, y, h, r):
    w = x1 - x0
    hi = r.random() < 0.5
    return np.array([(x0 - 0.06 * w, y + 2), (x0 - 0.06 * w, y - (h if hi else 0.25 * h)),
                     (x1 + 0.06 * w, y - (0.25 * h if hi else h)), (x1 + 0.06 * w, y + 2)])


def merlons(x0, x1, y, r, mw=(16, 24), gap=(10, 16), mh=(16, 22)):
    """The teeth along the top of a wall, cut by eye: no two the same."""
    out, x = [], x0 + r.uniform(0, 6)
    while x < x1 - 8:
        w, h = r.uniform(*mw), r.uniform(*mh)
        out.append(np.array([(x, y + 3), (x + r.normal(0, 0.8), y - h), (x + w + r.normal(0, 0.8), y - h + r.normal(0, 1)),
                             (min(x + w, x1), y + 3)]))
        x += w + r.uniform(*gap)
    return out


def quad(x0, y0, x1, y1, r, j=1.5):
    return np.array([(x0 + r.normal(0, j), y0 + r.normal(0, j)), (x1 + r.normal(0, j), y0 + r.normal(0, j)),
                     (x1 + r.normal(0, j), y1), (x0 + r.normal(0, j), y1)])


def arch(x, y, w, h):
    """A window or a door with a round head, its foot at (x, y)."""
    t = np.linspace(0, np.pi, 14)
    top = np.stack([x + w / 2 * np.cos(t), y - h + w / 2 - w / 2 * np.sin(t)], 1)
    return np.vstack([[(x + w / 2, y)], top, [(x - w / 2, y)]])


def limb(p0, p1, w0, w1, bow=0.0, s=0.0, n=24):
    """A trunk or a bough from p0 to p1, `w0` thick at its foot and `w1` at its end, bowed a little
    to one side, or with `s` bent one way and then the other as it climbs."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    t = np.linspace(0, 1, n)[:, None]
    d = p1 - p0
    L = np.hypot(*d)
    nrm = np.array([-d[1], d[0]]) / (L + 1e-9)
    C_ = p0 + d * t + nrm * L * (bow * 4 * t * (1 - t) + s * np.sin(2 * np.pi * t))
    wd = (w0 + (w1 - w0) * t) / 2
    return np.vstack([C_ + nrm * wd, (C_ - nrm * wd)[::-1]])


# ---------------------------------------------------------------- the sky

SKY = [(0.00, "#140f2c"), (0.10, "#191543"), (0.22, "#211f66"), (0.32, "#2b2d84"), (0.40, "#433899"),
       (0.465, "#76399b"), (0.515, "#b83c8f"), (0.548, "#e0517a"), (0.578, "#f27a48"), (0.61, "#f7a64a"),
       (0.75, "#f7a64a")]


def ramp(v):
    st = np.array([s for s, _ in SKY])
    cs = np.stack([C(c) for _, c in SKY])
    return np.stack([np.interp(v, st, cs[:, i]) for i in range(3)], -1)


def sky(bd, r):
    """The sky in long strokes of a broad flat brush from side to side, almost black at the top and
    mixed a little paler and warmer at each reload down to the hills: a full wet coat to cover the
    board, then over it a second of strokes that run out to dry ends."""
    sl = view(PX0, PY0, PX1, Y(0.74), pad=8)
    yy, xx = grid(sl)
    v = (yy - PY0) / PH + 0.01 * noise.field(yy.shape, 260, r) + 0.005 * noise.field(yy.shape, 60, r)
    one = np.ones(yy.shape, np.float32)
    U, V = axes(sl, -0.01)
    fill(bd, sl, one, U, V, ramp(v), 120, length=(3.0, 8.0), lap=0.25, ulap=(0.3, 0.5), load=(1.2, 1.4), spent=0.1,
         streak=0.03, vary=0.012, tails=0.3, soft=1.0)
    U, V = axes(sl, -0.016)
    fill(bd, sl, one, U, V, ramp(v + 0.01 * noise.field(yy.shape, 500, r)), 90, length=(4.0, 10.0), lap=0.12,
         ulap=(0.0, 0.15), load=(0.9, 1.2), spent=0.7, streak=0.07, vary=0.018, tails=1.0, grain=0.4, soft=1.0,
         odds=0.75)


CLOUDS = [  # (u0, u1, v, thickness px, colour, wisps): magenta low in the glow, a turquoise thread higher up
    (-0.04, 0.4, 0.47, 30, "#d8569a", 2), (0.72, 1.04, 0.445, 26, "#c94f9c", 1), (0.2, 0.44, 0.36, 14, "#2fa3a8", 1)]


def clouds(bd, r):
    """Long clouds dragged across the sky: a body of a few strokes laid end over end along a slow
    curve, fat in the middle and thin at the ends, and a wisp or two from a drier brush trailing off
    one end of it."""
    for u0, u1, v, t, col, n in CLOUDS:
        a, b, y = X(u0), X(u1), Y(v)
        if r.random() < 0.5:
            a, b = b, a
        sag = r.normal(0, 0.012) * abs(b - a)
        path = lambda f: (a + (b - a) * f, y + sag * 4 * f * (1 - f) + 6 * np.sin(2 * np.pi * f * 1.3 + 1.0))
        rows, f = [], r.uniform(-0.04, 0.02)
        while f < 0.98:
            g = min(1.04, f + r.uniform(0.22, 0.38))
            mid = (f + g) / 2
            fat = np.sin(np.pi * np.clip(mid, 0.05, 0.95)) ** 0.7
            (x0, y0), (x1, y1) = path(f), path(g)
            rows.append((x0, y0 + r.normal(0, 3), x1, y1 + r.normal(0, 3), t * (0.45 + 0.6 * fat) * r.uniform(0.85, 1.1),
                         1.0 + 0.15 * fat, r.uniform(0.35, 0.6) + 0.3 * (1 - fat), r.normal(0, 0.01)))
            f = g - r.uniform(0.05, 0.12)
        for _ in range(n):
            f0, f1 = sorted(r.uniform(-0.1, 1.1, 2))
            if f1 - f0 < 0.3:
                f1 = min(1.15, f0 + 0.3)
            dy = r.choice([-1, 1]) * t * r.uniform(0.3, 0.55)
            (x0, y0), (x1, y1) = path(f0), path(f1)
            rows.append((x0, y0 + dy, x1, y1 + dy + r.normal(0, 4), t * r.uniform(0.3, 0.5), r.uniform(0.8, 1.0),
                         r.uniform(0.7, 0.9), r.normal(0, 0.01)))
        drag(bd, rows, C(col), tails=2.0, grain=0.45, rough=3.0, taper=0.7, soft=0.8)


def moon(bd, r):
    """The new moon: one curved stroke of a round brush, pressed in the middle and lifted to a point
    at each end, then gone over once more with a drier brush along its outer edge."""
    cx, cy, L, a = X(0.8), Y(0.11), 150, -0.45
    dx, dy = np.cos(a) * L / 2, np.sin(a) * L / 2
    drag(bd, [(cx - dx, cy - dy, cx + dx, cy + dy, 34, 1.25, 0.15, 0.27)], C("#f6edbd"), tails=0.2, grain=0.3,
         rough=1.5, streak=0.04, head=0.5, taper=0.92)
    drag(bd, [(cx - dx + 3, cy - dy + 2, cx + dx + 2, cy + dy + 3, 20, 0.95, 0.6, 0.26)], C("#fbf5da"), tails=0.6,
         grain=0.45, rough=2.5, streak=0.05, head=0.5, taper=0.9)


STARS = [  # (u, v, size): specks, and two crossed
    (0.04, 0.06, 4.5), (0.235, 0.035, 3.0), (0.33, 0.12, 15), (0.43, 0.05, 3.5), (0.29, 0.235, 3.0),
    (0.46, 0.2, 5.0), (0.72, 0.04, 4.0), (0.95, 0.06, 3.0), (0.87, 0.05, 9), (0.775, 0.22, 3.4),
    (0.18, 0.12, 4.0), (0.39, 0.3, 2.6), (0.5, 0.11, 3.0)]


def stars(bd, r):
    """A few stars in the dark at the top, no two the same: specks touched in with the point of a
    round brush, and two crossed in quick strokes that taper to their points."""
    for u, v, s in STARS:
        x, y = X(u) + r.normal(0, 6), Y(v) + r.normal(0, 6)
        if s < 8:
            dabs(bd, [(x, y)], [s], ["#fbf6ec"], load=(0.9, 1.3))
            continue
        rows, a = [], r.normal(0, 0.15)
        for b, L in ((a, s), (a + np.pi / 2 + r.normal(0, 0.1), s * r.uniform(0.85, 1.1)),
                     (a + np.pi / 4, 0.45 * s), (a - np.pi / 4, 0.45 * s)):
            l1, l2 = L * r.uniform(0.85, 1.15), L * r.uniform(0.85, 1.15)
            rows.append((x - np.cos(b) * l1, y - np.sin(b) * l1, x + np.cos(b) * l2, y + np.sin(b) * l2,
                         0.32 * s, 1.15, 0.6, r.normal(0, 0.02)))
        drag(bd, rows, C("#fbf6ec"), tails=0.2, grain=0.2, rough=0.8, streak=0.0, head=0.6, taper=0.85)
        dabs(bd, [(x, y)], [0.3 * s], ["#fffaf0"], load=(1.2, 1.4))


# ---------------------------------------------------------------- the castle and the town

PALE, SHADE, ICE, LILAC = "#e4e0f4", "#b5aee0", "#a6ded4", "#c9c1ee"
DARK, LIT, GLOW = "#2b2343", "#ffd463", "#ffb44a"
TOWERS = [  # back to front: (left, right of the middle; top above the foot; roof height; roof; windows (fx, fy, lit))
    (-58, 58, 500, 320, ICE, ((0.5, 0.2, 1), (0.5, 0.5, 0))),
    (-126, -70, 400, 230, LILAC, ((0.45, 0.3, 1),)),
    (70, 124, 378, 220, LILAC, ((0.5, 0.36, 0),)),
    (-190, -134, 292, 172, ICE, ((0.5, 0.42, 0),)),
    (132, 188, 276, 162, ICE, ((0.5, 0.35, 1),)),
    (-238, -204, 186, 112, LILAC, ()),
    (206, 240, 176, 106, LILAC, ())]


def window(bd, r, x, foot, w, h, lit, form="arch"):
    """A window, its foot at (x, foot), in a touch or two of a small brush: dark, or lit."""
    P = arch(x, foot, w, h) if form == "arch" else quad(x - w / 2, foot - h, x + w / 2, foot, r, j=0.6)
    shape(bd, [P], LIT if lit else DARK, max(5, 0.8 * w), np.pi / 2, amp=0.45, scale=10, spent=0.15, vary=0.05)


def turret(bd, r, x0, x1, top, bot, rh, cap, win):
    """A tower of the castle: its wall in a few upright strokes, the shaded side laid on after in
    lavender, the needle of its roof, and its windows. -> the point of the roof"""
    w = x1 - x0
    shape(bd, [quad(x0, top, x1, bot, r)], PALE, np.clip(w * 0.45, 14, 60), np.pi / 2 + r.normal(0, 0.03),
          length=(2.0, 5.0), streak=0.06)
    shape(bd, [quad(x0 + 0.6 * w, top + 3, x1 + 1, bot, r, j=1.0)], SHADE, np.clip(w * 0.3, 10, 36), np.pi / 2,
          length=(2.0, 5.0), streak=0.06, amp=0.9, scale=40)
    P = cone(x0, x1, top + 4, rh, lean=r.normal(0, 0.015), sag=r.uniform(0.12, 0.2), eave=0.1)
    shape(bd, [P], cap, np.clip(w * 0.4, 12, 50), -np.pi / 2, length=(1.5, 4.0), streak=0.07)
    for fx, fy, lit in win:
        ww = np.clip(0.24 * w, 10, 22)
        window(bd, r, x0 + fx * w, top + fy * (bot - top), ww, ww * r.uniform(2.0, 2.5), lit)
    return P[np.argmin(P[:, 1])]


def pennant(bd, r, tip, col):
    """A pennant on a gilt pole at the point of a roof, its tail lifting in the wind."""
    pole = np.array([tip + [0, 4], tip + [r.normal(0, 1), -r.uniform(52, 64)]])
    lines(bd, [pole], "#e7c46a", 2.6, wander=0.2)
    fx, fy = pole[1]
    L = r.uniform(46, 56)
    t = np.linspace(0, 1, 16)
    wave = 4 * np.sin(np.pi * 1.6 * t + r.uniform(0, 3))
    upper = np.stack([fx + L * t, fy + wave * t], 1)
    lower = np.stack([fx + L * t[::-1], fy + 16 - 15 * t[::-1] + wave[::-1] * t[::-1]], 1)
    shape(bd, [np.vstack([upper, lower])], col, 10, 0.0, amp=0.5, scale=10)


def castle(bd, r, xc, yb):
    """The castle on the top of the hill, all in one pale mix: the curtain wall at the back, the keep
    with the tallest needle of a roof, towers stepping down on either side of it, and the front wall
    with its gate lit; a magenta pennant on the keep."""
    shape(bd, [quad(xc - 214, yb - 204, xc + 224, yb + 20, r)], PALE, 50, np.pi / 2)
    for P in merlons(xc - 214, xc + 224, yb - 202, r):
        shape(bd, [P], PALE, 16, np.pi / 2, amp=0.5, scale=12)
    tips = [turret(bd, r, xc + a, xc + b, yb - top, yb, rh, cap, win) for a, b, top, rh, cap, win in TOWERS]
    pennant(bd, r, tips[0], "#d8418a")
    shape(bd, [quad(xc - 258, yb - 122, xc + 266, yb + 36, r)], PALE, 56, np.pi / 2)
    for P in merlons(xc - 258, xc + 266, yb - 120, r):
        shape(bd, [P], PALE, 16, np.pi / 2, amp=0.5, scale=12)
    shape(bd, [arch(xc + 6, yb + 40, 76, 116)], DARK, 30, np.pi / 2, amp=0.6, scale=14, spent=0.15)
    shape(bd, [arch(xc + 6, yb + 40, 52, 94)], GLOW, 24, np.pi / 2, amp=0.6, scale=14, spent=0.2)
    for x, lit in ((xc - 178, 0), (xc + 150, 1)):
        window(bd, r, x, yb - 30, 18, 40, lit)


TOWN = (  # rows from the back: (depth of the foot under the top of the hill, houses), a house being (middle, from
          # the middle of the castle; width; height; wall; roof; its colour; roof height / width; windows (fx, fy, lit))
    (140, ((-300, 150, 220, "#ec862b", "gable", "#1b1530", 0.62, ((0.32, 0.36, 1),)),
           (-150, 120, 260, "#2aa69c", "cone", "#1b1530", 1.5, ((0.5, 0.3, 0),)),
           (125, 160, 200, "#efe5dc", "gable", "#c63c7c", 0.55, ((0.3, 0.4, 0), (0.7, 0.42, 1))),
           (285, 140, 230, "#d0438a", "gable", "#1b1530", 0.7, ((0.5, 0.38, 1),)))),
    (280, ((-420, 170, 190, "#4b3a8f", "gable", "#1b1530", 0.6, ((0.3, 0.4, 1), (0.7, 0.45, 0))),
           (-235, 150, 170, "#8f2d63", "shed", "#1b1530", 0.4, ((0.5, 0.45, 1),)),
           (-50, 190, 160, "#1f6b6c", "dome", "#e2ac36", 0.5, ((0.3, 0.45, 0), (0.7, 0.5, 1))),
           (150, 150, 200, "#352d78", "gable", "#c63c7c", 0.6, ((0.5, 0.4, 1),)),
           (330, 160, 170, "#ec862b", "gable", "#1b1530", 0.55, ((0.5, 0.45, 0),)))))


def house(bd, r, x, bot, w, h, wall, kind, roof, rf, win):
    """A house of the town: its wall in a few upright strokes, then its roof in its own colour, then a
    window or two, dark or lit."""
    x0, x1, top, rh = x - w / 2, x + w / 2, bot - h, rf * w
    shape(bd, [quad(x0, top, x1, bot, r)], wall, np.clip(w * 0.45, 14, 60), np.pi / 2 + r.normal(0, 0.03),
          length=(2.0, 5.0), streak=0.06)
    ang = 0.0
    if kind == "gable":
        P = gable(x0, x1, top + 3, rh, skew=r.normal(0, 0.05))
        ang = np.arctan2(-rh, w / 2) if r.random() < 0.5 else np.arctan2(rh, w / 2)
    elif kind == "cone":
        P, ang = cone(x0, x1, top + 4, rh, lean=r.normal(0, 0.03), sag=r.uniform(0.1, 0.25)), -np.pi / 2
    elif kind == "dome":
        P = dome(x0, x1, top + 3, rh)
    else:
        P = shed(x0, x1, top + 3, rh, r)
    shape(bd, [P], roof, np.clip(w * 0.4, 12, 50), ang, length=(1.5, 4.0), streak=0.07)
    form = "arch" if r.random() < 0.65 else "square"
    for fx, fy, lit in win:
        ww = np.clip(0.15 * w, 13, 24)
        hh = ww * (r.uniform(1.7, 2.2) if form == "arch" else r.uniform(1.1, 1.35))
        window(bd, r, x0 + fx * w, top + fy * h + hh / 2, ww, hh, lit, form)


def town(bd, r, xc, hill):
    """The town stacked down the hill under the castle, the back row first and the front row over its
    feet, each house a block of its own colour; a clipped hedge along the foot of each row."""
    for depth, row in TOWN:
        foot = lambda x, depth=depth: float(hill(x)) + depth
        for dx, w, h, wall, kind, roof, rf, win in row:
            house(bd, r, xc + dx, foot(xc + dx) + r.normal(0, 6), w, h, wall, kind, roof, rf, win)
        hedge(bd, r, [(xc + dx - w / 2, xc + dx + w / 2) for dx, w, *_ in row], foot)


def hedge(bd, r, spans, y):
    """A clipped hedge along the foot of each run of houses."""
    runs = []
    for a, b in sorted(spans):
        if runs and a < runs[-1][1] + 60:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    for a, b in runs:
        _hedge(bd, r, a - 20, b + 20, lambda x: y(x) + 12)


def _hedge(bd, r, x0, x1, y):
    """One run of hedge along the line y(x): round bushes cut in one after another, their tops a line
    of scallops."""
    blobs, x = [], x0
    while x < x1:
        R = r.uniform(17, 30)
        blobs.append(blob(x + R, y(x + R) - R * 0.35 + r.normal(0, 3), R, R * 0.8, r, lobes=0.04, n=40))
        x += R * r.uniform(1.3, 1.8)
    xs = np.linspace(x0, x1 + 10, 40)
    foot = np.array([y(t) for t in xs]) + 30 + 5 * noise.line1d(40, 6, r)
    blobs.append(np.vstack([np.stack([xs, [y(t) - 6 for t in xs]], 1), np.stack([xs, foot], 1)[::-1]]))
    shape(bd, blobs, "#17683f", 26, 0.0, length=(2.0, 5.0), amp=1.0, scale=30)


# ---------------------------------------------------------------- trees and the dark

def flame(bd, r, x, base, w, h, col, light=None, bend=0.0):
    """A cypress made into a flame: laid in upright strokes from a full foot to a point that bends
    over to one side, its edges a little uneven; its lit side laid on after in a paler colour, in a
    band that narrows to the point."""
    t = np.linspace(0, 1, 60)
    half = 0.5 * w * (1 - t) ** 0.7 * (0.72 + 0.5 * np.sin(np.pi * np.clip(1.6 * t, 0, 1)))
    hl = half * (1 + 0.07 * noise.line1d(60, 9, r))
    hr = half * (1 + 0.07 * noise.line1d(60, 9, r))
    cx = x + bend * h * t ** 2 + 0.012 * h * np.sin(np.pi * t * r.uniform(0.8, 1.4) + r.uniform(0, 3)) * t
    y = base - h * t
    P = np.vstack([np.stack([cx - hl, y], 1), np.stack([cx + hr, y], 1)[::-1]])
    shape(bd, [P], col, max(8, w * 0.42), -np.pi / 2 + bend, amp=np.clip(w / 70, 1.0, 2.4), scale=70,
          length=(2.5, 6.0))
    if light:
        k = r.uniform(0.3, 0.45)
        Q = np.vstack([np.stack([cx - hl * 0.93, y], 1), np.stack([cx - hl * (1 - 2 * k), y], 1)[::-1]])
        shape(bd, [Q[3:-3]], light, max(6, w * 0.25), -np.pi / 2 + bend, length=(2.0, 5.0), spent=0.55,
              load=(0.85, 1.1))


def tuft(bd, r, cx, cy, rx, ry, col):
    """One lobe of a crown: an oval laid round with the brush, and its edge feathered out all round in
    short flicks of the point."""
    shape(bd, [blob(cx, cy, rx, ry, r, lobes=0.07)], col, max(12, 0.6 * ry), r.normal(0, 0.15),
          length=(1.5, 3.5), spent=0.3, amp=1.5, scale=30)
    rows, a = [], r.uniform(0, 0.4)
    while a < 2 * np.pi:
        up = max(0.0, -np.sin(a))
        nx, ny = np.cos(a) * ry, np.sin(a) * rx
        n = np.hypot(nx, ny)
        nx, ny = nx / n, ny / n
        ex, ey = cx + 0.93 * rx * np.cos(a), cy + 0.93 * ry * np.sin(a)
        b = np.arctan2(ny, nx) + r.normal(0, 0.35)
        L = (11 + 7 * up) * r.uniform(0.6, 1.4)
        rows.append((ex - 8 * nx, ey - 8 * ny, ex + np.cos(b) * L, ey + np.sin(b) * L, r.uniform(6, 10),
                     r.uniform(1.0, 1.2), r.uniform(0.3, 0.6), r.normal(0, 0.12)))
        a += r.uniform(0.6, 1.4) * (11 + 8 * (1 - up)) / np.hypot(rx * np.sin(a), ry * np.cos(a))
    drag(bd, rows, C(col), tails=0.6, grain=0.3, rough=1.2, taper=-0.8, head=0.4)


def crown(bd, r, cx, cy, rx, ry, col):
    """A crown: a body laid round with the brush and a cloud of lobes heaped along its top, each
    feathered at its edge."""
    tuft(bd, r, cx, cy + 0.15 * ry, 0.92 * rx, 0.62 * ry, col)
    n = int(r.integers(4, 6))
    for t in np.sort(np.linspace(0.12, 0.88, n) * np.pi + r.normal(0, 0.08, n)):
        k = r.uniform(0.32, 0.45)
        tuft(bd, r, cx - 0.62 * rx * np.cos(t), cy - 0.38 * ry * np.sin(t), k * rx, k * rx * r.uniform(0.75, 0.95),
             col)


def tree(bd, r, foot, top, w0, crowns, col, bow=0.0, s=0.0):
    """A tree with a dark trunk that leans and bends as it climbs and forks into boughs, each holding
    up a crown."""
    shape(bd, [limb(foot, top, w0, 0.5 * w0, bow, s)], col, max(8, 0.9 * w0), -np.pi / 2, amp=1.0, scale=40,
          spent=0.3)
    shape(bd, [limb(top, (cx + r.normal(0, 0.1 * rx), cy + 0.3 * ry), 0.42 * w0, 0.2 * w0, r.normal(0, 0.08))
               for cx, cy, rx, ry in crowns], col, max(6, 0.4 * w0), -np.pi / 2, amp=0.8, scale=20, spent=0.3)
    for cx, cy, rx, ry in crowns:
        crown(bd, r, cx, cy, rx, ry, col)


def daisy(bd, r, x, y, R, col):
    """A flower in five to eight touches of a pale colour round a touch of lemon, the petals of no set
    length."""
    k = int(r.integers(5, 9))
    a0 = r.uniform(0, 2 * np.pi)
    rows = []
    for i in range(k):
        a = a0 + 2 * np.pi * (i + r.uniform(-0.15, 0.15)) / k
        L = R * r.uniform(0.8, 1.15)
        rows.append((x + np.cos(a) * 2, y + np.sin(a) * 2, x + np.cos(a) * L, y + np.sin(a) * L,
                     R * r.uniform(0.5, 0.65), 1.2, 0.2, r.normal(0, 0.08)))
    drag(bd, rows, C(col), tails=0.1, grain=0.15, rough=0.6, streak=0.02, head=0.8, taper=0.5)
    dabs(bd, [(x, y)], [R * 0.3], ["#ec8a2e" if col == "#f4e07a" else "#f2c53d"], load=(1.3, 1.5))


FLOWERS = [  # (colour, radius, offset from the middle of the clump, height of the stem)
    ("#f6f0e2", 52, -120, 330), ("#f4e07a", 40, 10, 430), ("#bfe9df", 44, -20, 230), ("#f6f0e2", 34, 120, 300),
    ("#f59b6b", 30, 175, 410)]


def flowers(bd, r, x, y):
    """A clump of flowers on the bank, pale against the dark: long leaves first, each one stroke pressed
    in the middle and lifted to a point at each end, then the flowers on their stems."""
    rows = []
    for _ in range(9):
        a = -np.pi / 2 + r.normal(0, 0.6)
        L = r.uniform(180, 380)
        bx, by = x + r.normal(0, 70), y + r.uniform(0, 60)
        rows.append((bx, by, bx + np.cos(a) * L, by + np.sin(a) * L, r.uniform(34, 52), 1.15, 0.35, r.normal(0, 0.1)))
    drag(bd, rows, C("#17493f"), tails=0.3, grain=0.25, rough=1.0, taper=0.85, head=0.3)
    for col, R, dx, h in FLOWERS:
        fx, fy = x + dx + r.normal(0, 10), y - h
        mid = (x + 0.6 * dx + r.normal(0, 20), y - 0.5 * h)
        lines(bd, [np.array([(x + 0.3 * dx, y + 40), mid, (fx, fy)])], "#17493f", 7)
    for col, R, dx, h in sorted(FLOWERS, key=lambda f: -f[3]):
        daisy(bd, r, x + dx + r.normal(0, 10), y - h, R, col)


def frame(bd, r):
    """Last, the dark that frames the view, in a black mixed with a little green: the umbrella pine on
    the left and the tufted tree on the right; then the leaves and flowers on the bank."""
    ink = "#0b1d21"
    tree(bd, r, (X(0.06), Y(1.03)), (X(0.13), Y(0.37)), 84,
         [(X(0.05), Y(0.3), 240, 110), (X(0.215), Y(0.345), 190, 90), (X(0.125), Y(0.2), 175, 86)], ink, s=0.03)
    tree(bd, r, (X(1.01), Y(1.03)), (X(0.9), Y(0.4)), 110, [(X(0.9), Y(0.3), 230, 170)], ink, bow=0.08)
    flowers(bd, r, X(0.3), Y(0.93))


# ---------------------------------------------------------------- the picture

HILL = [(0.20, 0.97), (0.26, 0.88), (0.31, 0.77), (0.36, 0.655), (0.41, 0.56), (0.46, 0.48), (0.51, 0.425),
        (0.56, 0.401), (0.62, 0.395), (0.68, 0.399), (0.73, 0.415), (0.79, 0.45), (0.86, 0.51), (0.93, 0.575),
        (0.99, 0.625), (1.04, 0.66)]


def paint(seed=1951):
    r = noise.rng(seed)
    bd = Board(r)
    sky(bd, r)
    clouds(bd, r)
    moon(bd, r)
    stars(bd, r)
    far = [(X(-0.03), Y(0.6)), (X(0.1), Y(0.585)), (X(0.25), Y(0.607)), (X(0.42), Y(0.59)), (X(0.62), Y(0.612)),
           (X(0.8), Y(0.586)), (X(1.03), Y(0.6))]
    ridge(bd, far, "#4b2d7e", 70, r, below=Y(0.8), streaks=(("#58368f", 0.3),))
    near = [(X(-0.03), Y(0.645)), (X(0.08), Y(0.628)), (X(0.2), Y(0.657)), (X(0.36), Y(0.64)), (X(0.6), Y(0.672)),
            (X(0.8), Y(0.636)), (X(1.03), Y(0.65))]
    ridge(bd, near, "#2a1c55", 70, r, below=Y(0.86), streaks=(("#33226a", 0.3),))
    xc = X(0.62)
    cap = lambda x: 410 * np.clip(1 - ((x - xc) / (0.26 * PW)) ** 2, 0, 1) ** 0.8     # where the light lies
    hill = ridge(bd, [(X(u), Y(v)) for u, v in HILL], "#26a05d", 84, r, below=Y(0.9), streaks=(("#2aa863", 0.3),),
                 stripes=((-40, cap, "#c3d636", 84), (cap, lambda x: cap(x) + 30, "#8cc63c", 34),
                          (lambda x: cap(x) + 200, 2400, "#177a4e", 84), (lambda x: cap(x) + 330, 2400, "#12603f", 84)))
    castle(bd, r, xc, Y(0.395) + 10)
    town(bd, r, xc, hill)
    for dx, d, w, h, bend in ((-720, 140, 70, 250, 0.06), (-640, 190, 54, 180, -0.05)):
        flame(bd, r, xc + dx, float(hill(xc + dx)) + d, w, h, "#123d3b", "#2a8a6e", bend)
    hb = [(X(-0.03), Y(0.68)), (X(0.1), Y(0.7)), (X(0.25), Y(0.745)), (X(0.42), Y(0.77)), (X(0.56), Y(0.765)),
          (X(0.68), Y(0.735)), (X(0.8), Y(0.7)), (X(0.92), Y(0.67)), (X(1.03), Y(0.66))]
    bank = ridge(bd, hb, "#0c2225", 80, r, streaks=(("#0f2a2c", 0.25),))
    for u, rx, ry in ((0.665, 95, 66), (0.745, 140, 100), (0.85, 175, 125), (0.96, 150, 110)):   # a thicket climbing
        crown(bd, r, X(u), float(bank(X(u))) - 0.2 * ry, rx, ry, "#0c2225")                    # to the right
    frame(bd, r)
    gy, gx = np.gradient(ndimage.gaussian_filter(bd.tooth, 0.8))
    img = bd.img * (1 + 0.05 * (gx + gy))[..., None]
    return plate.mount(img.astype(np.float32), Sheet(img, bd.tooth, bd.tooth, bd.alpha))


if __name__ == "__main__":      # one lane of strokes: loaded it covers, it stays in its region, and it runs dry
    r = noise.rng(0)
    bd = Board.__new__(Board)
    bd.r = r
    bd.img = np.full((H, W, 3), 0.8)
    bd.tooth = np.full((H, W), 0.5, np.float32)
    bd.tape = np.ones((H, W), np.float32)
    bd.fine, bd.clump = bristles(r)
    sl = (slice(1000, 1100), slice(1000, 1600))
    yy, xx = grid(sl)
    reg = ((xx > 1050) & (xx < 1550)).astype(np.float32)
    red = C("#cf2020")
    fill(bd, sl, reg, xx, yy, red, 60, length=(20, 20), spent=0.0, vary=0.0)
    assert np.abs(bd.img[1040:1060, 1100:1500] - red).max() < 0.08, "loaded, it covers"
    assert np.allclose(bd.img[1000:1100, 1000:1048], 0.8), "it paints nowhere else"
    bd.img[:] = 0.8
    drag(bd, [(1050, 1050, 1550, 1050, 40, 0.8, 0.95, 0.0)], red)
    held = lambda a, b: (np.abs(bd.img[1040:1060, a:b] - red).sum(-1) < 0.15).mean()
    assert held(1060, 1150) > 0.8 > held(1450, 1540), "and runs dry"
    print("ok")
