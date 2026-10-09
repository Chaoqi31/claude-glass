"""The Arch Closing. Colour linocut.

In the winter of 1930 the two halves of the Sydney Harbour Bridge stood out over the water from either
shore, each held back by its cables and each carrying at its tip a creeper crane that lifted the next
members up from the barges below and crept forward along the top chord as it went. In August the halves
met. Dorrit Black, back in Sydney from Claude Flight's classes at the Grosvenor School in London, painted
the arch that year as two curved planes reaching for each other across the harbour.

This is a new design in the colour linocut she learned from Flight: the eye low on the water east of the
bridge, the halves reaching across the top of the sheet, the cranes at their tips holding out their jibs
over the gap, and the afternoon light breaking through the gap in rays. Over the arch the sky is combed
with the gouge in long cuts that follow its curve; ferries cross the harbour below, the near one's smoke
streaming back white over the water. Five blocks of linoleum were cut with gouges and a knife, one for
each colour, and printed by hand in oil-based ink, lightest first: a yellow, a vermilion, a cobalt, a
viridian, and a dark that does the drawing. Each was inked with a roller, the yellow from one charged gold
in its middle and orange toward its ends, so that the light warms toward the sides and the horizon, and
burnished on the back of thin Japanese paper with the bowl of a spoon, so the ink lies unevenly, salted
where the pressure was light and showing the swirl of the spoon, and each colour falls over the others to
make the oranges and the deep blue-greens. The blocks were laid to the paper by eye; none of them meets the
others exactly, and white lines run between the colours where the cutter left them. The impression is
titled, numbered and signed in pencil below.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from atelier import lettering, noise, paper, pencil, plate
from atelier.color import glaze, pigment

TITLE = "The Arch Closing"
DATE = "2026"
MEDIUM = "Colour linocut from five blocks in oil-based ink, burnished by hand, on thin Japanese paper"
AFTER = ("Dorrit Black, The Bridge, Sydney, 1930, and the Grosvenor School colour linocut of Claude Flight, "
         "Sybil Andrews and Cyril Power, London, 1929–1933")
ROOM = "The Workshop"
YEAR = 1930
PLACE = "Sydney"
REGION = "Oceania"
NOTE = ("The two halves of the Harbour Bridge reach for each other from their creeper cranes, weeks before "
        "they met, with the afternoon light breaking through the gap. Five blocks burnished by hand with a spoon "
        "overprint to make the oranges and the deep blue-greens.")

H, W = 2600, 2200                       # the image, as cut on the blocks
MT, MS, MB, B = 150, 150, 250, 44       # the paper left round it (top, sides, foot), and the wall round the sheet
SH, SW = H + MT + MB + 2 * B, W + 2 * MS + 2 * B
OY, OX = B + MT, B + MS
SS = 2                                  # the blocks are drawn at twice the size they print

INKS = {"Y": pigment("#f1c04f"), "R": pigment("#e2502f"), "C": pigment("#5080c8"), "V": pigment("#3f9c86"),
        "K": pigment("#262a3f")}
LOAD = {"Y": 1.0, "R": 0.95, "C": 1.0, "V": 1.0, "K": 1.9}
ORANGE = pigment("#eb903b")             # what the yellow roller carried at its ends
GRAPHITE = pigment("#6e6e74")
PEN = next(Path("/System/Library/AssetsV2").glob("com_apple_MobileAsset_Font*/*/AssetData/Hanzipen.ttc"), None)
INSCRIPTION = lettering.CUT / "sydney_inscription.png"
LINE = 60                               # the baseline of the inscription, px down its strip

# the bridge as it stood in the winter of 1930, and the eye low on the water to the east of it
SPAN, PANEL = 503.0, 503.0 / 28
TIP = 1.5 * PANEL                       # each half stops this far short of the middle
EYE = np.array([0.0, 5.0, 0.0])
MID = np.array([-20.0, 0.0, 520.0])
AX = np.array([np.cos(np.radians(16)), 0.0, np.sin(np.radians(16))])     # along the bridge, south to north
NX = np.array([-AX[2], 0.0, AX[0]])                                     # across it, away from the eye
F, PITCH, YAW, CY = 2600.0, np.radians(10.0), np.radians(-2.5), 1140.0
HZ = CY + F * np.tan(PITCH)            # the horizon


def view(P):
    """World (m; x right, y up, z away) to the picture (px), for the eye looking a little up."""
    P = np.asarray(P, float) - EYE
    x = np.cos(YAW) * P[..., 0] - np.sin(YAW) * P[..., 2]
    z = np.sin(YAW) * P[..., 0] + np.cos(YAW) * P[..., 2]
    y = np.cos(PITCH) * P[..., 1] - np.sin(PITCH) * z
    z = np.sin(PITCH) * P[..., 1] + np.cos(PITCH) * z
    return np.stack([W / 2 + F * x / z, CY - F * y / z], -1)


def at(s, h, n=0.0):
    """The point s m along the bridge from its middle, h m over the water, n m across from its axis."""
    s, h = np.broadcast_arrays(np.asarray(s, float), np.asarray(h, float))
    return view(MID + s[..., None] * AX + h[..., None] * np.array([0.0, 1.0, 0.0]) + n * NX)


def chords(s):
    """Heights of the lower chord (the arch proper) and the top chord. The cutter drew the rise twice what
    it is, as Black did: the halves climb steeply and lean together at the top."""
    u = np.abs(2 * np.asarray(s, float) / SPAN)
    low = 14 + 240 * (1 - u ** 2)
    return low, low + 1.5 * (18 + 39 * u ** 2)


def scale(s, h, n=0.0):
    """Pixels to the metre at a point of the bridge."""
    return np.hypot(*(at(s, h + 0.5, n) - at(s, h - 0.5, n)).T)


def gap_point():
    """The middle of the gap between the tips, on the lower chord."""
    lo, _ = chords(TIP)
    return 0.5 * (at(-TIP, lo, -15) + at(TIP, lo, -15))


def arch_curve():
    """The top chord of the near truss as if the arch were closed, carried on past its bearings: the curve
    the cutter echoed across the sky. Points (x, y), left to right."""
    s = np.linspace(-0.7 * SPAN, 0.7 * SPAN, 4000)
    return at(s, chords(s)[1], -15)


def arch_low():
    """The lower chord of the far truss, as if the arch were closed: the curve the cutter echoed in the light
    under the arch. Points (x, y), left to right."""
    s = np.linspace(-0.6 * SPAN, 0.6 * SPAN, 3000)
    return at(s, chords(s)[0], 15)


def outward(P, d):
    """The curve P moved d px outward, away from the water under it."""
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return P + np.c_[t[:, 1], -t[:, 0]] * np.asarray(d, float)[..., None]


# ---------------------------------------------------------------------------------------------- the cutting


class Mask:
    """Marks drawn on a block at SS times the size it prints: what the gouge took, or what it left
    standing. `get` returns them in [0, 1] at the size they print."""

    def __init__(self):
        self.im = Image.new("L", (W * SS, H * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def area(self, pts):
        p = np.asarray(pts, float) * SS
        if len(p) > 2:
            self.d.polygon([tuple(q) for q in p], fill=255)

    def stroke(self, p, hw):
        """A band along path p with half-widths hw (px)."""
        p = np.asarray(p, float)
        if len(p) < 2:
            return
        hw = np.broadcast_to(np.asarray(hw, float), len(p))
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        off = np.c_[-t[:, 1], t[:, 0]] * hw[:, None]
        self.area(np.vstack([p + off, (p - off)[::-1]]))

    def get(self):
        return np.asarray(self.im.resize((W, H), Image.BOX), np.float32) / 255


def resample(pts, step):
    pts = np.asarray(pts, float)
    s0 = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    s = np.linspace(0, s0[-1], max(2, int(s0[-1] / step) + 1))
    return np.c_[np.interp(s, s0, pts[:, 0]), np.interp(s, s0, pts[:, 1])]


def hand(pts, r, amp=0.6, step=3.0, scale_=70.0):
    """Resample a path every `step` px and let it waver as a hand-guided tool does: a slow drift and a
    quicker tremor."""
    p = resample(pts, step)
    n = len(p)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    wav = amp * (noise.line1d(n, max(4.0, scale_ / step), r) + 0.35 * noise.line1d(n, max(3.0, scale_ / 5 / step), r))
    return p + np.c_[-t[:, 1], t[:, 0]] * wav[:, None]


def gouge(n, w, r, bite=0.06, lift=0.25, swell=0.25):
    """Half-widths along a cut of n points: the gouge bites in, runs with the pressure of the hand, and
    lifts out to a point."""
    t = np.linspace(0, 1, n)
    a, b = max(bite * r.uniform(0.5, 1.5), 1e-3), max(lift * r.uniform(0.6, 1.4), 1e-3)
    hw = w * noise.smoothstep(0, a, t) ** 0.7 * noise.smoothstep(1, 1 - b, t) ** 0.8
    return hw * np.clip(1 + swell * noise.line1d(n, max(4.0, n / 5), r), 0.35, None)


def cut(m, pts, w, r, amp=0.5, bite=0.06, lift=0.25, swell=0.25, step=3.0):
    """One stroke of the gouge along pts, drawn into the mask m that gathers what the gouge took (or, for a
    ridge left standing, what stays)."""
    p = hand(pts, r, amp, step)
    m.stroke(p, gouge(len(p), w, r, bite, lift, swell))


def bar(m, pts, w, r, amp=0.25):
    """A ridge left standing between two cuts: a member of the steel, a spar, a rope."""
    cut(m, pts, w, r, amp=amp, bite=0.0, lift=0.0, swell=0.1)


def round_off(a, sigma):
    """What a knife and a round gouge leave of a cut-out shape: no sharp inside corner survives."""
    return noise.smoothstep(0.3, 0.7, ndimage.gaussian_filter(a, sigma))


def shape(m, pts, r, amp=0.7):
    """A flat shape cut round with the knife: its outline wavers a little as the hand goes round it."""
    p = np.asarray(pts, float)
    m.area(hand(np.vstack([p, p[:1]]), r, amp, step=4.0))


# ---------------------------------------------------------------------------------------------- the design


def panels(side):
    """Where the members meet the chords on one half, from the bearing to the tip: twelve whole panels and
    the half panel at the tip. Cut by eye, they close up toward the tip as the arch turns away."""
    w = np.r_[np.ones(12), 0.5] * (1 - 0.22 * noise.smoothstep(4, 12.5, np.arange(13) + 0.5))
    return side * (SPAN / 2 - np.r_[0, np.cumsum(w)] * 12.5 / w.sum() * PANEL)


def area(T):
    """The signed area of the triangle T."""
    return 0.5 * ((T[1, 0] - T[0, 0]) * (T[2, 1] - T[0, 1]) - (T[2, 0] - T[0, 0]) * (T[1, 1] - T[0, 1]))


def inset(P, d):
    """The triangle P with its side i (corner i to corner i+1) moved in by d[i] px; None where it closes
    up."""
    n, c = [], []
    for i in range(3):
        a, b, o = P[i], P[(i + 1) % 3], P[(i + 2) % 3]
        v = np.array([a[1] - b[1], b[0] - a[0]]) / np.hypot(*(b - a))
        v = v if np.dot(o - a, v) > 0 else -v
        n.append(v)
        c.append(np.dot(v, a) + d[i])
    Q = np.array([np.linalg.solve(np.array([n[i - 1], n[i]]), [c[i - 1], c[i]]) for i in range(3)])
    return Q if area(Q) * area(P) > 0 and abs(area(Q)) > 4 else None


def clear(m, P, d, r):
    """Clear the triangle P between three members with the gouge, each side d[i] px in from the middle of
    its member: a push close along each member, then the middle taken out. The pushes swell and thin a
    little with the hand, and the round of the gouge cannot reach into a corner, so the steel keeps a
    little of each corner, like a gusset."""
    P = np.asarray(P, float)
    Q = inset(P, d)
    if Q is None:
        return
    rad = 2 * abs(area(Q)) / sum(np.hypot(*(Q[i] - Q[i - 1])) for i in range(3))
    gw = float(np.clip(0.25 * rad, 1.5, 5.0))
    C = inset(P, np.asarray(d) + gw)             # the line the middle of the gouge ran along
    if C is None:
        return
    for i in range(3):
        p = [C[i], C[(i + 1) % 3]]
        cut(m, p if r.random() < 0.5 else p[::-1], gw * r.uniform(0.9, 1.1), r, amp=0.3, bite=0.0, lift=0.0,
            swell=0.12)
    m.area(C)


def truss(side, n, sp, r, k=1.0):
    """One truss of one half of the arch, side -1 (south) or 1 (north), n m across from the axis, with its
    panel points at sp. The block is cut round the outline of the half, and each triangle between the
    members is cleared with the gouge. No two members come out quite the same width, and the arch's own
    lower chord stands heavier than the top chord. Returns what stands of it and the line along the top of
    its top chord."""
    s = np.linspace(side * SPAN / 2, side * TIP, 400)
    lo, hi = chords(s)
    L, U = at(s, lo, n), at(s, hi, n)
    keep = (L[:, 0] > -300) & (L[:, 0] < W + 300) | (U[:, 0] > -300) & (U[:, 0] < W + 300)
    s, lo, hi, L, U = s[keep], lo[keep], hi[keep], L[keep], U[keep]
    sc = scale(s, lo, n)
    wl = 3.6 * k * sc * (1 + 0.08 * noise.line1d(len(s), 50, r))
    wu = 2.1 * k * sc * (1 + 0.08 * noise.line1d(len(s), 50, r))

    def sky(C):                                          # the normal toward the sky
        t = np.gradient(C, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        return np.c_[t[:, 1], -t[:, 0]] * -side

    nu, nl = sky(U), sky(L)
    sil = Mask()
    shape(sil, np.vstack([U + nu * wu[:, None], (L - nl * wl[:, None])[::-1]]), r, amp=0.8)
    lo_k, hi_k = chords(sp)
    Lk, Uk = at(sp, lo_k, n), at(sp, hi_k, n)
    sck = scale(sp, lo_k, n)
    wv = 1.4 * k * sck * r.lognormal(0, 0.15, len(sp))
    wd = 0.95 * k * sck * r.lognormal(0, 0.15, len(sp))
    holes = Mask()
    a = np.abs(s)[::-1]
    for i in range(len(sp) - 1):
        mid = abs(0.5 * (sp[i] + sp[i + 1]))
        cl, cu = np.interp(mid, a, wl[::-1]), np.interp(mid, a, wu[::-1])
        clear(holes, [Uk[i], Uk[i + 1], Lk[i]], [cu, wd[i], wv[i]], r)
        clear(holes, [Lk[i], Lk[i + 1], Uk[i + 1]], [cl, wv[i + 1], wd[i]], r)
    return sil.get() * (1 - round_off(holes.get(), 1.6)), U + nu * (0.45 * wu)[:, None]


def crane(m, side, r):
    """A creeper crane at the tip of one half: a braced frame riding the top chord, its cab at the back, a
    mast, and a jib held out over the gap with a member hanging from its hook."""
    d = -side                                   # toward the middle
    st = side * TIP
    _, hr = chords(st - d * 22)
    _, hf = chords(st - d * 3)
    p = lambda s, h: at(s, h, -15)
    sc = scale(st, hf, -15)
    w = lambda m_: m_ * sc
    roof_r, roof_f = hr + 11, hf + 11
    frame = [p(st - d * 22, hr), p(st - d * 22, roof_r), p(st - d * 3, roof_f), p(st - d * 3, hf)]
    for a, b in zip(frame, frame[1:]):
        bar(m, [a, b], w(0.75), r, amp=0.2)
    for f0, f1 in ((0, 0.5), (0.5, 1.0)):     # bracing
        s0, s1 = st - d * (22 - 19 * f0), st - d * (22 - 19 * f1)
        h0, h1 = hr + (hf - hr) * f0, hr + (hf - hr) * f1
        bar(m, [p(s0, h0), p(s1, h1 + 11)], w(0.45), r, amp=0.2)
        bar(m, [p(s0, h0 + 11), p(s1, h1)], w(0.45), r, amp=0.2)
    shape(m, [p(st - d * 21, roof_r), p(st - d * 21, roof_r + 6), p(st - d * 12, roof_r + 6.5),
              p(st - d * 12, roof_r)], r, amp=0.3)
    mast = p(st - d * 9, roof_f + 17)
    bar(m, [p(st - d * 13, roof_r), mast], w(0.7), r)
    bar(m, [p(st - d * 5, roof_f), mast], w(0.6), r)
    foot, end = p(st - d * 2, hf + 4), p(st + d * 21, hf + 20)
    cut(m, [foot, end], w(0.95), r, amp=0.2, bite=0.0, lift=0.15, swell=0.1)
    bar(m, [mast, end], w(0.3), r, amp=0.1)
    bar(m, [mast, p(st - d * 22, roof_r)], w(0.35), r, amp=0.1)
    hook = p(st + d * 21, hf - 22)
    bar(m, [end, hook], w(0.22), r, amp=0.15)
    shape(m, [p(st + d * 15, hf - 22), p(st + d * 15, hf - 24.5), p(st + d * 28, hf - 24.5), p(st + d * 28, hf - 22)],
          r, amp=0.3)


def heavens(r, P, under):
    """Over the arch, the cobalt block, combed with the gouge in long cuts that follow the arch round:
    close and broad near it, so the sky pales toward the steel, sparse and fine further out, and sparest
    in the top left corner. Next to the arch the cobalt is cleared away and the yellow block prints a rim
    of light; out in the corners the viridian prints over the cobalt to deepen it, its edge frayed into
    strokes along the same curves. Returns the cobalt, the yellow rim and the viridian."""
    away = ndimage.distance_transform_edt(1 - under) + 5 * noise.field((H, W), 160, r)
    d0 = 52 + 14 * noise.field((H, W), 300, r)
    blue = noise.smoothstep(-0.8, 0.8, away - d0)
    rim = noise.smoothstep(0.8, -0.8, away - d0 - 5) * (1 - under)
    dv = 470 + 50 * noise.field((H, W), 400, r)
    deep = Mask()
    cuts = Mask()
    d = 50.0
    while d < 2100:
        Q = resample(outward(P, d), 6.0)
        Q = Q[(Q[:, 0] > -120) & (Q[:, 0] < W + 120) & (Q[:, 1] > -120) & (Q[:, 1] < HZ - 6)]
        p_cut = 0.05 + 0.85 * np.exp(-(d - 50) / 260)
        p_deep = noise.smoothstep(300, 470, d)
        i = int(r.integers(0, 40))
        while i < len(Q) - 4:
            L = int((20 + 100 * r.random()) * (1 + d / 900))
            calm = 1 - 0.5 * np.exp(-(Q[i, 0] / 1000) ** 2 - (Q[i, 1] / 800) ** 2)
            if r.random() < p_cut * calm:
                hw = (1.3 + 5.0 * np.exp(-(d - 50) / 240)) * r.lognormal(0, 0.3)
                cut(cuts, Q[i:i + L], hw, r, amp=0.7, bite=0.12, lift=0.35, swell=0.3)
            if 0 < p_deep < 1 and r.random() < p_deep * (0.2 + 0.8 * calm):
                cut(deep, Q[i:i + L], r.uniform(4, 12) * p_deep, r, amp=0.7, bite=0.2, lift=0.5, swell=0.4)
            i += L + int((4 + 40 * r.random()) * (0.4 + d / 400))
        d += (7 + 0.035 * d) * r.uniform(0.6, 1.4)
    combed = 1 - cuts.get()
    viridian = np.maximum(noise.smoothstep(-0.8, 0.8, away - dv), deep.get()) * (1 - under)
    return blue * combed, rim, viridian * combed


def light(r, G, under):
    """Under the arch the yellow block prints the light. It is cut away round the gap in a burst of short
    broad strokes, so the paper itself shines there, and in long rays out from it; a few long cuts
    lie across it like high cloud, and toward the sides a few more follow the arch round, as the cuts of
    the sky do. Low over the far shore the vermilion block lays streaks of cloud, crowded near the shore
    and thinning upward. Returns the yellow, its cuts and the vermilion."""
    cuts, red = Mask(), Mask()
    cx, cy = G[0], G[1] + 25

    def ray(th, r0, L, w, lift):
        rr = np.linspace(r0, r0 + L, max(8, int(L / 8)))
        pts = np.c_[cx + np.cos(th) * rr, cy + np.sin(th) * rr]
        pts = pts[pts[:, 1] < HZ - 5]
        if len(pts) > 6:
            cut(cuts, pts, w, r, amp=0.6, bite=0.03, lift=lift, swell=0.3)

    for _ in range(46):                                       # the burst round the gap
        th = r.uniform(0.05, np.pi - 0.05)
        ray(th, r.uniform(0, 30), r.uniform(90, 400) * (0.6 + 0.4 * np.sin(th)), r.uniform(6, 13), 0.85)
    for _ in range(60):                                       # the long rays
        th = np.pi / 2 + r.normal(0, 0.62)
        if 0.1 < th < np.pi - 0.1:
            ray(th, r.uniform(60, 320), r.uniform(250, 1400) * (0.5 + 0.5 * np.sin(th)),
                r.uniform(2.2, 6.5) * (0.5 + 0.5 * np.sin(th)), 0.6)
    for _ in range(22):                                       # high cloud
        yy = r.uniform(cy + 280, HZ - 330)
        L = r.uniform(150, 650)
        x0 = r.uniform(-100, W - 100)
        xs = np.linspace(x0, x0 + L, 50)
        cut(cuts, np.c_[xs, yy + r.normal(0, 4) * np.sin(np.linspace(0, np.pi, 50))], r.uniform(2, 5.5), r,
            amp=0.8, bite=0.2, lift=0.5, swell=0.4)
    Pl = arch_low()                                           # the cuts that follow the arch, at the sides
    d = r.uniform(35, 60)
    while d < 380:
        Q = resample(outward(Pl, -d), 6.0)
        Q = Q[(Q[:, 1] < HZ - 40) & (Q[:, 0] > -50) & (Q[:, 0] < W + 50)]
        i = int(r.integers(0, 60))
        while i < len(Q) - 10:
            n = int(r.uniform(40, 110))
            off = abs(Q[min(i + n // 2, len(Q) - 1), 0] - cx)
            if r.random() < 0.6 * noise.smoothstep(350, 700, off):
                cut(cuts, Q[i:i + n], r.uniform(2.6, 4.6) * (1 - d / 600), r, amp=0.7, bite=0.12, lift=0.45,
                    swell=0.3)
            i += n + int(r.uniform(10, 50))
        d += r.uniform(40, 75)
    for _ in range(110):
        h = r.uniform(0, 1) ** 2.2
        yy = HZ - 14 - 380 * h
        L = r.uniform(60, 560) * (1.25 - h)
        x0 = r.uniform(-250, W + 50)
        xs = np.linspace(x0, x0 + L, 40)
        ys = yy + r.normal(0, 3) * np.sin(np.linspace(0, np.pi, 40))
        cut(red, np.c_[xs, ys], r.uniform(2.0, 8) * (1.15 - 0.7 * h), r, amp=0.8, bite=0.15, lift=0.5, swell=0.35)
    y = np.arange(H, dtype=np.float32)[:, None]
    return under * (y < HZ), cuts.get(), red.get() * under * (y < HZ)


def roll(r, G):
    """How the yellow was rolled on over the sky: from a roller charged gold in its middle and orange toward
    its ends, run up and down the block over the gap, so the light warms toward the sides; then from a
    second, orange one run along the foot of the sky. Each leaves its streaks the way it went. The water
    was inked on its own, in gold. Returns the share of orange."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    across = np.abs(x - G[0]) / (0.5 * W) + 0.05 * noise.stretched((H, W), 7, 320, r)
    down = (y - G[1]) / (HZ - G[1]) + 0.035 * noise.stretched((W, H), 7, 320, r).T
    m = 0.75 * noise.smoothstep(0.45, 1.05, across) + 0.55 * noise.smoothstep(0.6, 1.0, down)
    return np.clip(m, 0, 1) * (y < HZ)


def harbour(r, G, wakes):
    """The water, from the cobalt and the viridian blocks. The cobalt prints all of it and is cut with
    ripples, row on row, small and crowded far off, long and broad near; the rows run closer in some
    bands than others, so the harbour lies in bands of lighter and darker blue. The viridian prints broad
    bands over it, their edges left ragged by the same strokes, and the dark block lays a few of the
    nearest ripples. Under the gap the light lies on the water in dashes, cut from both and printed by the
    yellow, and near the far shore the vermilion of the cloud comes back in thin strokes. Returns the
    cobalt, the viridian, the yellow, the vermilion and the dark."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    sea = (y >= HZ).astype(np.float32)
    depth = np.clip((y - HZ) / (H - HZ), 0, 1)
    sag = 1 - ((x - G[0]) / W * 1.6) ** 2
    swell = noise.field((H, W), 900, r) + 0.4 * noise.field((H, W), 350, r)
    g = depth - 0.04 * sag * np.sqrt(depth) + 0.11 * depth ** 1.5 * swell + wakes
    lev = np.cumsum(r.uniform(0.6, 1.4, 12))
    lev = (lev / lev[-1]) ** 1.6 * 1.2
    band = np.searchsorted(lev, g)
    # how closely each band is cut: the far water, which takes the sky, most of all
    open_ = np.clip(np.r_[0.85, 0.7, r.choice([0.15, 0.3, 0.55, 0.75], len(lev))][band] - 0.25 * depth, 0.05, 0.9)
    cols = np.arange(-60, W + 60, 6)
    gc = g[:, np.clip(cols, 0, W - 1)]
    ripples, vrip, drip = Mask(), Mask(), Mask()
    lv = 0.002
    while lv < 1.25:
        below = gc > lv
        yl = np.where(below.any(0), np.argmax(below, axis=0), H + 80).astype(np.float32)
        dd = min(lv / 1.2, 1.0)
        i = int(r.integers(0, 8))
        while i < len(cols) - 2:
            L = max(2, int((3 + 80 * dd) * r.lognormal(0, 0.5)))
            seg = slice(i, min(len(cols), i + L))
            xm, ym = cols[min(i + L // 2, len(cols) - 1)], yl[min(i + L // 2, len(cols) - 1)]
            if ym < H + 40 and len(cols[seg]) > 2:
                o = open_[int(np.clip(ym, 0, H - 1)), int(np.clip(xm, 0, W - 1))]
                pts = np.c_[cols[seg], yl[seg]]
                kw = dict(amp=0.4 + 1.2 * dd, bite=0.15, lift=0.45, swell=0.35)
                if r.random() < o:
                    cut(ripples, pts, (0.6 + 6.5 * dd) * r.lognormal(0, 0.35), r, **kw)
                elif r.random() < 0.35:
                    cut(vrip, pts, (0.6 + 6.0 * dd) * r.lognormal(0, 0.35), r, **kw)
                elif r.random() < 0.35 * dd:
                    cut(drip, pts, (0.5 + 2.8 * dd) * r.lognormal(0, 0.35), r, **kw)
            i += L + max(1, int((1 + 30 * dd) * r.lognormal(0, 0.8) / 6))
        lv += (0.0035 + 0.03 * dd) * r.uniform(0.6, 1.4)
    edge = g + 0.01 * noise.stretched((W, H), 3, 140, r).T
    vband = np.isin(np.searchsorted(lev, edge), [2, 4, 5, 7, 9, 10, 12])
    viridian = vband * sea * (1 - vrip.get())
    # the light of the gap on the water: dashes in a column under it, longer and looser as they come near;
    # and near the far shore the vermilion of the cloud given back in short strokes
    glints, gold, red = Mask(), Mask(), Mask()
    yy = HZ + 5.0
    while yy < H:
        d = (yy - HZ) / (H - HZ)
        for _ in range(r.poisson(3 + 6 * d)):
            cx = G[0] + r.normal(0, 25 + 120 * d)
            L = (24 + 200 * d) * r.lognormal(0, 0.45)
            xs = np.linspace(cx - L / 2, cx + L / 2, 24)
            ys = yy + r.normal(0, 1.5) + 0.5 * np.sin(np.linspace(0, np.pi, 24)) * r.normal(0, 2 + 6 * d)
            cut(glints if r.random() < 0.45 else gold, np.c_[xs, ys], (1.4 + 5 * d) * r.lognormal(0, 0.35), r,
                amp=0.6, bite=0.25, lift=0.5, swell=0.5)
        if d < 0.16:
            for _ in range(r.poisson(3)):
                cx = r.uniform(150, W - 150)
                L = (40 + 320 * d) * r.lognormal(0, 0.5)
                cut(red, np.c_[np.linspace(cx - L / 2, cx + L / 2, 16), np.full(16, yy + r.normal(0, 1.5))],
                    (0.9 + 5 * d) * r.lognormal(0, 0.3), r, amp=0.4, bite=0.25, lift=0.5)
        yy += (6 + 60 * d) * r.uniform(0.6, 1.4)
    glints, gold = glints.get() * sea, gold.get() * sea
    shine = np.clip(glints + gold, 0, 1)
    cobalt = sea * (1 - ripples.get()) * (1 - shine)
    return cobalt, viridian * (1 - shine), gold, red.get() * sea * (1 - shine), drip.get() * sea * (1 - shine)


def ferry(blk, x0, x1, yw, r, d=-1):
    """A double-ended harbour ferry going left (d=-1) or right (d=1), cut as the rest is cut: a dark hull
    whose sheer lifts at both ends, the rail along its top and the rubbing strake below cut white in two or
    three goes; a long white saloon, its windows nicked one by one out of a dark band with the gouge, no two
    quite the same width or the same distance apart; a deck line over it, an upper deck with a wheelhouse at
    each end, and a tall funnel amidships, yellow, raked back, its black top cut round by hand a hair clear
    of the yellow. A curl of white water at the bow. Drawn into the masks in `blk` (per ink, 'clear' where
    the paper is left, 'cutK' where the dark is cut away). Returns the top of the funnel."""
    L = x1 - x0
    k = L / 760
    xm = 0.5 * (x0 + x1)
    a = max(k, 0.4)                                  # how far the hand wavers: no finer on a small boat
    X = (lambda x: np.asarray(x, float)) if d < 0 else (lambda x: x0 + x1 - np.asarray(x, float))
    Pt = lambda pts: [(float(X(px)), py) for px, py in pts]
    ends = lambda x: ((x - xm) / (L / 2)) ** 2
    sheer = lambda x: yw - 62 * k - 20 * k * ends(x)
    deck = lambda x: yw - 120 * k - 7 * k * ends(x)
    run = lambda xa, xb, n=40: np.linspace(xa, xb, n)
    band = lambda xa, xb, top, bot, n=40: np.vstack([np.c_[run(xa, xb, n), top(run(xa, xb, n))],
                                                     np.c_[run(xa, xb, n), bot(run(xa, xb, n))][::-1]])
    xs = run(x0, x1, 60)
    tuck = 26 * k
    shape(blk["K"], np.vstack([np.c_[xs, sheer(xs)], [[x1 - tuck, yw], [x0 + tuck, yw]]]), r, amp=1.1 * a)
    for xa, xb, dy, hw in ((x0 + 10 * k, x1 - 10 * k, 7 * k, 1.2 * k), (x0 + 36 * k, x1 - 36 * k, 20 * k, 2.2 * k)):
        stops = np.sort(r.uniform(xa + 60 * k, xb - 60 * k, int(r.integers(1, 3))))
        for u, v in zip(np.r_[xa, stops + r.uniform(5, 14, len(stops)) * k], np.r_[stops, xb]):
            q = run(u, v)
            cut(blk["cutK"], np.c_[q, sheer(q) + dy], max(hw, 0.6) * r.uniform(0.85, 1.15), r, amp=0.5 * a,
                bite=0.05, lift=0.12, swell=0.3, step=1.5)
    # the saloon, and the band of its windows
    xa, xb = x0 + 75 * k, x1 - 75 * k
    shape(blk["clear"], band(xa, xb, deck, lambda x: sheer(x) + 3), r, amp=0.8 * a)
    qa, qb = xa + 12 * k, xb - 12 * k
    shape(blk["K"], band(qa, qb, lambda x: deck(x) + 14 * k, lambda x: deck(x) + 40 * k), r, amp=0.6 * a)
    edge = max(4 * k, 1.5)
    x = qa + r.uniform(0.4, 0.9) * max(17 * k, 7)
    while x < qb - max(10 * k, 4):
        lean = r.normal(0, 0.8 * k)
        p = [(x + lean, deck(x) + 14 * k + edge * r.uniform(0.6, 1.2)),
             (x - lean, deck(x) + 40 * k - edge * r.uniform(0.6, 1.2))]
        cut(blk["cutK"], p, max(4.2 * k, 1.2) * r.uniform(0.8, 1.2), r, amp=0.25 * a, bite=r.uniform(0.03, 0.1),
            lift=r.uniform(0.08, 0.2), swell=0.1, step=0.75)
        x += max(30 * k, 12) * r.lognormal(0, 0.12)
    ridge = lambda xa, xb, f, w: cut(blk["K"], np.c_[run(xa, xb, 50), f(run(xa, xb, 50))], w, r, amp=0.6 * a,
                                     bite=0.0, lift=0.0, swell=0.25, step=1.5)
    ridge(x0 + 60 * k, x1 - 60 * k, deck, 3.2 * k)
    # the upper deck, and a wheelhouse at each end with its windows nicked
    shape(blk["clear"], band(xm - 150 * k, xm + 150 * k, lambda x: deck(x) - 34 * k, lambda x: deck(x) + 1), r,
          amp=0.6 * a)
    ridge(xm - 160 * k, xm + 160 * k, lambda x: deck(x) - 35 * k, 2.6 * k)
    for wx in (x0 + 140 * k, x1 - 140 * k):
        shape(blk["clear"], band(wx - 28 * k, wx + 28 * k, lambda x: deck(x) - 38 * k, lambda x: deck(x) + 1, 12),
              r, amp=0.5 * a)
        ridge(wx - 34 * k, wx + 34 * k, lambda x: deck(x) - 40 * k, 3 * k)
        shape(blk["K"], band(wx - 22 * k, wx + 22 * k, lambda x: deck(x) - 30 * k, lambda x: deck(x) - 13 * k, 12),
              r, amp=0.4 * a)
        for j in (-1, 0, 1):
            cx = wx + j * 14.5 * k + r.normal(0, 1.2 * k)
            cut(blk["cutK"], [(cx, deck(cx) - 28 * k), (cx, deck(cx) - 15 * k)],
                max(2.8 * k, 1.0) * r.uniform(0.8, 1.2), r, amp=0.2 * a, bite=0.05, lift=0.15, swell=0.1, step=0.75)
        bar(blk["K"], [(wx, deck(wx) - 40 * k), (wx + 4 * k, deck(wx) - 107 * k)], 1.6 * k, r)
    if k > 0.5:             # the rail round the upper deck and its stanchions, where nothing stands behind them
        for xa, xb in ((x0 + 64 * k, x0 + 110 * k), (x0 + 170 * k, xm - 152 * k), (xm + 152 * k, x1 - 170 * k),
                       (x1 - 110 * k, x1 - 64 * k)):
            ridge(xa, xb, lambda x: deck(x) - 12 * k, 1.1 * k)
            sx = xa + r.uniform(2, 9) * k
            while sx < xb - 3 * k:
                bar(blk["K"], [(sx, deck(sx) - 12 * k), (sx + r.normal(0, 0.5), deck(sx))], 0.85 * k, r, amp=0.1)
                sx += r.uniform(13, 24) * k
    # the funnel, raked back: yellow, and its black top cut round with the knife, sagging a little as the
    # round of the funnel does seen from below
    fb, ft = deck(xm) - 34 * k, yw - 307 * k
    rake = 16 * k
    side = lambda s, f: np.array([xm + s * 22 * k + rake * f, fb + (ft - fb) * f])
    top = 1 - 34 * k / (fb - ft)
    u = np.linspace(0, 1, 14)[:, None]
    lip = lambda dy: (1 - u) * side(-1, top) + u * side(1, top) + np.c_[0 * u, dy + 12 * k * u * (1 - u)]
    shape(blk["clear"], Pt([side(-1, 0), side(-1, 1), side(1, 1), side(1, 0)]), r, amp=0.4 * a)
    shape(blk["Y"], Pt(np.vstack([[side(-1, 0)], lip(2.2 * k + 0.6), [side(1, 0)]])), r, amp=0.4 * a)
    shape(blk["K"], Pt(np.vstack([lip(0), [side(1, 1), side(-1, 1)]])), r, amp=0.5 * a)
    # white water at the bow, thrown up and falling back along the hull
    bow = X(x0)
    for j in range(4):
        L2 = r.uniform(60, 160) * k
        t = np.linspace(0, 1, 20)
        pts = np.c_[bow + d * (8 * k - L2 * t), yw - 2 - (10 * k * np.sin(np.pi * t) + j * 5 * k) * (1 - 0.3 * j)]
        cut(blk["clear"], pts, r.uniform(2.5, 6) * k, r, amp=0.5, bite=0.1, lift=0.6)
    return (float(X(xm + rake + 4 * k)), ft)


def smoke(src, to, w0, w1, r, lines=7):
    """Smoke from a funnel laid back by the ferry's way, catching the low sun: a black puff at the funnel,
    then strands that stream back and spread as they lift, cut white out of the water. Returns the
    strands and the puff."""
    t = np.linspace(0, 1, 120)
    mid = np.asarray(to[1], float)
    a, c = np.asarray(src, float), np.asarray(to[0], float)
    path = ((1 - t) ** 2)[:, None] * a + (2 * t * (1 - t))[:, None] * mid + (t ** 2)[:, None] * c
    path[:, 1] += 6 * noise.line1d(120, 25, r)
    tg = np.gradient(path, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[-tg[:, 1], tg[:, 0]]
    wid = (w0 + (w1 - w0) * t ** 0.8) * (1 + 0.15 * noise.line1d(120, 20, r))
    strands, puff = Mask(), Mask()
    for j in range(lines):
        f = (j + 0.5) / lines - 0.5 + r.normal(0, 0.04)
        i0, i1 = int(r.uniform(4, 25)), int(r.uniform(70, 119))
        q = path + nrm * (f * wid + 4 * noise.line1d(120, 15, r) * t)[:, None]
        cut(strands, q[i0:i1], r.uniform(2.0, 4.5) * (1.2 - abs(f)), r, amp=1.0, bite=0.08, lift=0.6, swell=0.4)
    n = 22
    q = path[:n] + nrm[:n] * (0.15 * wid[:n] * np.sin(np.linspace(0, 3, n)))[:, None]
    cut(puff, q, w0 * 0.55, r, amp=1.0, bite=0.0, lift=0.7, swell=0.3)
    return strands.get(), puff.get()


def wake(cutm, x, yw, length, r, k=1.0, side=1):
    """The wash behind a ferry: a churned band along its track and the two arms of its wake opening away,
    cut white with the gouge in long strokes."""
    for arm, (dy, w) in enumerate(((0, 7), (-14, 4), (16, 6), (34, 5))):
        n = int(r.uniform(3, 5))
        for j in range(n):
            t0 = j / n + r.uniform(0, 0.1)
            t1 = t0 + r.uniform(0.15, 0.35)
            xs = x + side * length * np.linspace(t0, t1, 30)
            spread = dy * k * (1 + 2.5 * np.linspace(t0, t1, 30)) ** (1 if arm else 0.3)
            ys = yw + 2 * k + spread + r.normal(0, 1.0)
            cut(cutm, np.c_[xs, ys], w * k * (1 - 0.6 * t0) * r.uniform(0.7, 1.2), r, amp=0.8, bite=0.1, lift=0.5)


def shore(blk, r):
    """The near shores under the feet of the arch, dark against the light: on the left the point under the
    south half, a long wharf shed on the water and a chimney; on the right the bridge workshops of the
    north shore under their saw-tooth roofs, and the stump of the abutment tower the north half springs
    from. A few windows catch the sun."""
    def window(x, y, w, h):
        box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        shape(blk["cutK"], box, r, amp=0.25)
        shape(blk["clear"], box, r, amp=0.25)
        shape(blk["Y"], box, r, amp=0.25)

    k = blk["K"]
    shape(k, [(-20, 1372), (80, 1388), (170, 1424), (255, 1474), (325, 1526), (385, HZ + 16), (-20, HZ + 20)], r,
          amp=1.0)
    shape(k, [(-14, 1512), (40, 1478), (232, 1484), (296, 1520), (300, HZ + 14), (-14, HZ + 16)], r, amp=0.6)
    shape(k, [(148, 1440), (150, 1352), (165, 1352), (167, 1440)], r, amp=0.4)
    for j in range(8):
        window(14 + j * 34 + r.normal(0, 1.5), 1538 + r.normal(0, 1), 13, 20)
    shape(k, [(1872, HZ + 18), (1912, 1526), (1990, 1494), (2080, 1474), (2230, 1462), (2230, HZ + 20)], r, amp=1.0)
    top = 1478
    teeth = [(1930, HZ + 10), (1930, top)]
    for x0 in np.arange(1930, 2200, 46):
        teeth += [(x0 + 36, top - 32), (x0 + 42, top - 32), (x0 + 42, top)]
    teeth += [(2230, top), (2230, HZ + 10)]
    shape(k, teeth, r, amp=0.5)
    shape(k, [(2150, 1470), (2150, 1430), (2232, 1426), (2232, 1470)], r, amp=0.5)
    for j in range(6):
        window(1950 + j * 44 + r.normal(0, 1.5), 1518 + r.normal(0, 1), 15, 22)
        if r.random() < 0.5:
            window(1950 + j * 44 + r.normal(0, 1.5), 1554 + r.normal(0, 1), 15, 16)


def far_shore(blk, r):
    """Balmain and the far side of the harbour under the arch: a low line of land, blue with distance."""
    xs = np.arange(-20, W + 21, 10, dtype=np.float32)
    hgt = 22 + 16 * noise.line1d(len(xs), 18, r) + 30 * np.exp(-((xs - 700) / 260) ** 2) + 20 * np.exp(-((xs - 1500) / 180) ** 2)
    top = HZ - np.clip(hgt, 6, None)
    poly = np.vstack([np.c_[xs, top], [[W + 20, HZ + 3], [-20, HZ + 3]]])
    shape(blk["C"], poly, r, amp=0.8)
    shape(blk["V"], poly, r, amp=0.8)
    for cx in (560, 1610):
        t0 = np.interp(cx, xs, top)
        shape(blk["K"], [(cx - 5, t0 + 4), (cx - 4, t0 - 70), (cx + 4, t0 - 70), (cx + 5, t0 + 4)], r, amp=0.3)
    for x in r.uniform(0, W, 18):
        t0 = np.interp(x, xs, top)
        w, h = r.uniform(10, 26), r.uniform(5, 12)
        house = [(x, t0 + 3), (x, t0 - h), (x + w, t0 - h), (x + w, t0 + 3)]
        shape(blk["clear"], house, r, amp=0.3)
        shape(blk["Y"], house, r, amp=0.3)


# ---------------------------------------------------------------------------------------------- the printing


def spoon(r, n=7000):
    """The bowl of a spoon rubbed over the back of the sheet in small overlapping circles, patch by patch.
    The printer leaned harder in some places than others and missed a little here and there. Returns the
    pressure everywhere, about 1 where it was well rubbed."""
    im = Image.new("F", (SW, SH), 0.0)
    d = ImageDraw.Draw(im)
    for x, y, q, a0, span, w, v in zip(r.uniform(0, SW, n), r.uniform(0, SH, n), r.uniform(14, 50, n),
                                       r.uniform(0, 360, n), r.uniform(150, 420, n), r.integers(5, 13, n),
                                       r.uniform(0.3, 1.0, n)):
        d.arc((x - q, y - q, x + q, y + q), a0, a0 + span, fill=float(v), width=int(w))
    swirl = ndimage.gaussian_filter(np.asarray(im), 1.8)
    lean = noise.fbm((SH, SW), 420, r, octaves=3)
    return 0.88 + 0.1 * lean - 0.2 * noise.smoothstep(0.9, 2.0, -lean) + 0.22 * swirl


def edge(r):
    """The outline of a block, cut by hand: its sides are not quite straight, nor its corners square."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    side = lambda n: 2.2 * noise.line1d(n, 350, r) + 0.7 * noise.line1d(n, 40, r)
    d = np.minimum.reduce([x - side(H)[:, None], W - 1 - x - side(H)[:, None],
                           y - side(W)[None, :], H - 1 - y - side(W)[None, :]])
    return noise.smoothstep(-0.7, 0.7, d - 3.0)


def pull(img, plan, ink, sheet, r, rubs, prior, tint=None):
    """Ink one block with the roller, lay the sheet on it by eye, burnish the back with the spoon, and
    lift it. `tint` is the ink's absorbance where it changes over the block. Returns the sheet and the ink
    now on it."""
    full = np.zeros((SH, SW), np.float32)
    full[OY:OY + H, OX:OX + W] = plan * edge(r)
    # laid by eye: a few pixels out and a hair turned
    a = r.normal(0, 6e-4)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([SH, SW]) / 2
    full = ndimage.affine_transform(full, m, c - m @ c - r.normal(0, 3.0, 2), order=1)
    # the roller: faint streaks along its travel, the orange-peel of tacky ink, thin patches where it was
    # not charged evenly, and a little more ink squeezed to the edges of a shape
    film = (1 + 0.04 * noise.stretched((SW, SH), 3, 200, r).T + 0.06 * noise.field((SH, SW), 1.3, r)
            + 0.09 * noise.fbm((SH, SW), 140, r, octaves=3)
            - 0.16 * noise.smoothstep(0.8, 1.8, noise.fbm((SH, SW), 260, r, octaves=2)))
    rim = np.clip(ndimage.gaussian_filter(full, 1.5) - ndimage.gaussian_filter(full, 4.0), 0, None)
    film += 0.8 * rim
    press = np.roll(rubs[r.integers(len(rubs))], tuple(r.integers(0, 700, 2)), (0, 1))
    # the paper meets the ink where its fibres stand up; light pressure misses the valleys between them and
    # leaves the ink thinner, and ink takes a little less well on ink already printed
    thr = 0.44 - 0.45 * press + 0.05 * np.clip(prior, 0, 1)
    contact = noise.smoothstep(thr - 0.1, thr + 0.1, sheet.tooth + 0.06 * noise.field((SH, SW), 1.0, r))
    dens = (LOAD[ink] * full * film * contact * (0.84 + 0.16 * sheet.tooth)
            * (0.78 + 0.22 * noise.smoothstep(0.62, 1.05, press)))
    dens = ndimage.gaussian_filter(dens, 0.5)
    return glaze(img, dens, INKS[ink] if tint is None else tint), prior + dens


def write(seed=26, xh=15.0):
    """Write the inscription into atelier/lettering (it needs the HanziPen face of macOS): "The Arch
    Closing" at the left, "12/50" after it, "Claude" at the right, on an x-height of `xh` px, as a map of
    how hard the pencil was pressed. Each word goes down a little off the line, a little larger or smaller
    and a little turned, and each letter a hair off the word's own line and pressed harder or softer; the
    name larger, firmer and leaning further than the rest."""
    r = noise.rng(seed)
    S = 3
    strip = np.zeros((100 * S, W * S), np.float32)

    def word(text, x, size=1.0, lean=0.0, track=1.0, firm=1.0, right=False):
        em = xh / 0.413 * size * r.uniform(0.96, 1.04) * S
        fonts = [ImageFont.truetype(str(PEN), round(em * s), index=0) for s in (0.95, 1.0, 1.05)]
        pad = int(em)
        im = Image.new("L", (int(fonts[2].getlength(text) * 1.1) + 2 * pad, 2 * pad), 0)
        d = ImageDraw.Draw(im)
        bx, by = pad, int(1.3 * pad)                    # where the word's baseline starts
        cx, p = float(bx), min(1.0, firm * r.uniform(0.8, 1.0))
        for ch in text:
            f = fonts[r.integers(3)]
            d.text((cx, by + r.normal(0, 0.35 * S)), ch, int(255 * p * r.uniform(0.85, 1.0)), font=f, anchor="ls")
            cx += f.getlength(ch) * track * r.uniform(0.97, 1.05)
        sh = lean + r.normal(0, 0.03)
        im = im.transform(im.size, Image.AFFINE, (1, sh, -sh * by, 0, 1, 0), Image.BICUBIC)
        im = im.rotate(r.normal(0, 1.0), Image.BICUBIC, center=(bx, by))
        width = (cx - bx) / S
        x = x - width if right else x
        X, Y = int(x * S - bx), int((LINE + r.normal(0, 0.6)) * S - by)
        a = np.asarray(im, np.float32) / 255
        strip[Y:Y + a.shape[0], X:X + a.shape[1]] = np.maximum(strip[Y:Y + a.shape[0], X:X + a.shape[1]], a)
        return x + width

    x = 40.0
    for w in ("The", "Arch", "Closing"):
        x = word(w, x) + xh * r.uniform(0.75, 1.0)
    word("12/50", x + xh * r.uniform(1.6, 2.2), size=0.95)
    word("Claude", W - 70, size=1.2, lean=0.13, track=0.97, firm=1.15, right=True)
    im = Image.fromarray((strip * 255).astype(np.uint8)).resize((W, 100), Image.LANCZOS)
    im.save(INSCRIPTION)


def inscription(sheet, r):
    """In pencil under the image, as the Grosvenor printers signed each impression: the title at the left,
    the number of the impression after it and the name at the right, as `write` set them down. The point
    wanders off the line a little as the hand goes along and is pressed harder in some strokes than in
    others, and the grain of the paper takes the graphite only where it stands up."""
    m = np.asarray(Image.open(INSCRIPTION), np.float32) / 255
    h, w = m.shape
    m = noise.warp(m, 0.45 * noise.field((h, w), 9, r), 0.6 * noise.field((h, w), 15, r))
    p = 0.85 * m * np.clip(0.9 + 0.2 * noise.field((h, w), 8, r) + 0.08 * noise.line1d(w, 260, r)[None], 0.4, 1.2)
    press = np.zeros((SH, SW), np.float32)
    y0 = OY + H + 78 - LINE
    press[y0:y0 + h, OX:OX + w] = ndimage.gaussian_filter(p, 0.45)
    return pencil.catch(press, sheet, grip=0.8)


def paint(seed=1930):
    # each part of the work draws on its own stream of chance, so that recutting one leaves the rest as it was
    r, rf, rs, rl, rd, rb, rp = (noise.rng(seed + i) for i in range(7))
    rw = noise.rng(seed + 105)
    G = gap_point()
    sheet = paper.washi((SH, SW), seed, tint="#f3ead6", margin=(B, B), fibre_density=0.6)
    sheet.alpha = paper.deckle((SH, SW), (B, B), r, ragged=1.5)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    P = arch_curve()
    under = (y > np.interp(x[0], P[:, 0], P[:, 1])[None, :]).astype(np.float32)

    # the ferries first, for their wakes push the bands of the water aside
    blk = {k: Mask() for k in ("Y", "R", "C", "V", "K", "clear", "cutK")}
    fx, fy = ferry(blk, 260, 1020, 2250, rf, d=1)
    gx, gy = ferry(blk, 1390, 1610, 1712, rf, d=-1)
    wakes = 0.03 * np.exp(-((y - 2262) / 50) ** 2) * noise.smoothstep(320, -200, x)

    blue_sky, rim, deep_sky = heavens(rs, P, under)
    yellow_sky, rays, red_sky = light(rl, G, under)
    cobalt, viridian, gold, red_sea, dark_sea = harbour(rw, G, wakes)
    above = (y < HZ).astype(np.float32)
    plans = {"Y": np.clip(yellow_sky * (1 - rays) + rim * above + gold, 0, 1), "R": np.clip(red_sky + red_sea, 0, 1),
             "C": np.clip(blue_sky * above + cobalt * (1 - red_sea), 0, 1),
             "V": np.clip(deep_sky * above + viridian * (1 - red_sea), 0, 1)}

    far_shore(blk, rd)
    shore(blk, rd)
    wcut = Mask()
    wake(wcut, 260, 2250, 700, rd, k=1.0, side=-1)
    wake(wcut, 1610, 1712, 420, rd, k=0.3, side=1)
    p1, k1 = smoke((fx, fy), ((-200, fy - 190), (fx - 380, fy - 150)), 24, 190, rd)
    p2, k2 = smoke((gx, gy), ((W + 80, gy - 45), (gx + 300, gy - 35)), 8, 60, rd, lines=3)

    dark, lit = Mask(), Mask()
    arch = np.zeros((H, W), np.float32)
    for side in (-1, 1):
        sp = panels(side)
        far, _ = truss(side, 15, sp, rb, k=0.8)
        near, top = truss(side, -15, sp, rb)
        plans["R"] = np.maximum(plans["R"], far)
        arch = np.maximum(arch, near)
        crane(dark, side, rb)
        # the sun along the top chord: a fine line cut out of the dark, broken where the cutter lifted
        i = 0
        while i < len(top) - 10:
            j = min(len(top), i + int(rb.uniform(60, 220)))
            cut(lit, top[i:j], rb.uniform(1.6, 2.6), rb, amp=0.3, bite=0.1, lift=0.3)
            i = j + int(rb.uniform(5, 40))
    steel = np.maximum(arch, dark.get())

    got = {k: blk[k].get() for k in blk}
    clear = np.clip(got["clear"] + got["K"], 0, 1)
    wash = np.clip(wcut.get() * (1 - above) + np.maximum(p1, p2), 0, 1)
    for k in "YRCV":
        plans[k] = np.maximum(plans[k] * (1 - wash) * (1 - clear), got[k])
    plans["R"] = np.maximum(plans["R"] * (1 - steel), steel)
    plans["Y"] *= 1 - steel
    plans["K"] = np.maximum.reduce([steel * (1 - lit.get()), got["K"] * (1 - got["cutK"]), np.maximum(k1, k2),
                                    dark_sea * (1 - clear)])

    # the drawing was traced onto each block by hand, and no tracing lies quite like another
    dx, dy = (3.5 * noise.field((H, W), 520, r) + 1.2 * noise.field((H, W), 140, r) for _ in range(2))
    for k in plans:
        plans[k] = noise.warp(plans[k], dx + 0.8 * noise.field((H, W), 200, r), dy + 0.8 * noise.field((H, W), 200, r))

    rubs = [spoon(r) for _ in range(2)]
    m = np.zeros((SH, SW, 1), np.float32)
    m[OY:OY + H, OX:OX + W, 0] = roll(rl, G)
    img, prior = sheet.color.copy(), np.zeros((SH, SW), np.float32)
    for k in "YRCVK":
        img, prior = pull(img, plans[k], k, sheet, r, rubs, prior,
                          tint=(1 - m) * INKS["Y"] + m * ORANGE if k == "Y" else None)
    img = glaze(img, 0.9 * inscription(sheet, rp), GRAPHITE)
    return plate.mount(img, sheet, shadow=0.35)


if __name__ == "__main__":
    write()
