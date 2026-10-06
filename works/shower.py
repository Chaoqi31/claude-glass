"""Shower on the Long Bridge. Colour woodblock print.

A summer storm comes down the river all at once and catches people out on the long bridge.
Hiroshige printed such a shower in 1857, and Van Gogh copied it in oil thirty years later.
This one is cut from separate blocks of cherry: a sumi keyblock for every contour, flat
colours with the wood showing through them, the cloud, the river and the piers wiped by hand
before each pull, and the rain on two blocks of its own, cut one line at a time.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import noise, paper, plate, relief
from atelier.color import glaze, pigment

TITLE = "Shower on the Long Bridge"
DATE = "2026"
MEDIUM = "Colour woodblock print (moku-hanga) from seventeen blocks, on kozo paper"
AFTER = ("Utagawa Hiroshige, Sudden Shower over Shin-Ōhashi Bridge and Atake, from One Hundred Famous Views "
         "of Edo, 1857")
ROOM = "The Workshop"
YEAR = 1857
PLACE = "Edo"
REGION = "East Asia"
NOTE = ("The rain is cut on two blocks, one printed in sumi and one in grey, their lines at slightly "
        "different slants. The river was wiped darker toward the near bank before every pull.")

H, W, M = 2520, 1680, 90            # the image, and the paper left round it
SH, SW = H + 2 * M, W + 2 * M
SS = 3                              # blocks are cut at three times the size they print

SUMI = pigment("#45423f")
GREY = pigment("#989692")
CLOUD = pigment("#3d3b3a")
SKY = pigment("#8f99a3")
SHORE = pigment("#879683")
PALE = pigment("#b9ccd3")
PRUSSIAN = pigment("#2b598e")
TAN = pigment("#e3cea3")
RED = pigment("#dca386")
PIER = pigment("#6d6964")
STRAW = pigment("#d5b468")
SKIN = pigment("#eccaa9")
INDIGO = pigment("#6585ab")
CAPE = pigment("#a99a72")
LOG = pigment("#8c6d57")

PITCH, F, HC, CY = np.radians(35), 3500.0, 46.7, H / 2 + 110    # looking down on the bridge from high on the bank
A = np.array([13.6, 39.4])                      # where the front of the deck leaves the right edge
D = np.array([-31.6, 22.7]) / np.hypot(31.6, 22.7)   # along the bridge, away to the left
N = np.array([D[1], -D[0]])                     # across it, away from us
WIDE = 3.6
BENTS = np.cumsum(noise.rng(7).uniform(2.8, 4.2, 30)) - 17.0    # the piers stand where the carpenters put them


class Block:
    """A cherry block being cut, at SS times the size it prints: what is left standing takes ink."""

    def __init__(self):
        self.im = Image.new("L", (SW * SS, SH * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def area(self, pts, fill=255):
        self.d.polygon([((x + M) * SS, (y + M) * SS) for x, y in np.asarray(pts, float)], fill=fill)

    def line(self, pts, hw, fill=255):
        p = np.asarray(pts, float)
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        off = np.c_[-t[:, 1], t[:, 0]] * np.broadcast_to(np.asarray(hw, float), len(p))[:, None]
        self.area(np.vstack([p + off, (p - off)[::-1]]), fill)

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
    wav = amp * (np.sin(k * r.uniform(0.5, 1.5) + r.uniform(0, 7))
                 + 0.5 * np.sin(k * r.uniform(2, 4) + r.uniform(0, 7)))
    return p + np.c_[-t[:, 1], t[:, 0]] * wav[:, None]


def key(blk, pts, r, hw=1.0, taper=0.04, amp=0.35, vary=0.3):
    """One keyblock line: the carver leaves a ridge either side of the drawn stroke, never of one width."""
    if len(pts) < 2:
        return
    p, w = relief.vcut(hand(pts, r, amp), hw, r, taper=taper, wobble=0.15)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    swell = np.sin(s / r.uniform(20, 60) + r.uniform(0, 7)) * np.sin(s / r.uniform(80, 220) + r.uniform(0, 7))
    blk.line(p, w * (1 + vary * swell))


def proj(P):
    """World (metres, the river at z=0) to the print (px), with the scale there in px per metre.
    Heights go straight up the sheet, as in a print, not off toward a vanishing point."""
    P = np.asarray(P, np.float64)
    zc = P[..., 1] * np.cos(PITCH) + HC * np.sin(PITCH)
    yc = P[..., 1] * np.sin(PITCH) - HC * np.cos(PITCH)
    k = F / zc
    return np.stack([W / 2 + k * P[..., 0], CY - k * (yc + np.cos(PITCH) * P[..., 2])], -1), k


def deck_z(t):
    """A long, low arch that sags a little between the piers and is never quite true."""
    t = np.asarray(t, float)
    i = np.clip(np.searchsorted(BENTS, t) - 1, 0, len(BENTS) - 2)
    u = np.clip((t - BENTS[i]) / (BENTS[i + 1] - BENTS[i]), 0, 1)
    return 4.2 + 2.4 * (1 - ((t - 28) / 40) ** 2) - 0.07 * np.sin(np.pi * u) ** 2 + 0.03 * np.sin(t / 2.3 + 1.0)


def at(t, n=0.0, z=0.0):
    """A point on the bridge, t metres along it, n across from the front edge, z above the deck."""
    t, n, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (t, n, z)))
    xy = A + t[..., None] * D + n[..., None] * N
    return proj(np.concatenate([xy, (deck_z(t) + z)[..., None]], -1))


def water(xy):
    """Where a point of the print falls on the river, in metres."""
    x, y = (xy[0] - W / 2) / F, -(xy[1] - CY) / F
    dy, dz = y * np.sin(PITCH) + np.cos(PITCH), y * np.cos(PITCH) - np.sin(PITCH)
    k = -HC / dz
    return np.array([k * x, k * dy, 0.0])


def wipe(y0, y1, r, wander=10.0, streak=0.006):
    """A bokashi wiped across the block with a damp cloth: full at row y0, gone by y1, never quite level."""
    yy = np.arange(SH, dtype=np.float32)[:, None] - M
    edge = wander * noise.line1d(SW, 260, r)[None, :]
    t = (yy - y0 + edge) / (y1 - y0) + streak * noise.stretched((SW, SH), 3, 220, r).T
    return 1 - noise.smoothstep(0, 1, t)


def bridge(r, kb):
    """The deck, its railings and the beam along its face, as blocks; their keylines go into `kb`.
    Returns the blocks and the line of the beam's foot."""
    ts = np.linspace(-14, 62, 700)
    deck, rails = Block(), Block()
    front, _ = at(ts, 0.0)
    rear, _ = at(ts, WIDE)
    face, _ = at(ts, 0.0, -0.6)
    deck.area(np.vstack([front, rear[::-1]]))
    rails.area(np.vstack([front, face[::-1]]))
    joints = np.arange(-14, 62, 0.31)
    for t in joints + r.normal(0, 0.03, joints.size):      # the joints of the planks, cut fine in the deck block
        if r.random() < 0.55:
            p, _ = at([t, t + 0.02], [r.uniform(0.2, 0.9), r.uniform(WIDE - 0.9, WIDE)])
            deck.line(*relief.vcut(hand(p, r, 0.15), r.uniform(0.25, 0.42), r, taper=0.2, wobble=0.2), fill=0)
    for p, hw in ((front, 1.3), (face, 1.1), (rear, 0.85)):
        key(kb, p, r, hw)
    for z in (-0.16, -0.31, -0.45):                 # the grain of the side beam, and a joint over every pier
        t0 = -14.0
        while t0 < 62:
            seg = np.linspace(t0, t0 + r.uniform(2, 9), 40)
            if r.random() < 0.6:
                key(kb, at(seg, 0.0, z + 0.025 * np.sin(seg / r.uniform(0.6, 1.4) + r.uniform(0, 7)))[0], r, 0.36,
                    taper=0.2)
            t0 = seg[-1] + r.uniform(0.5, 4)
    for t in BENTS:
        key(kb, at([t, t], 0.0, [0.0, -0.6])[0], r, 0.75, taper=0.1)
    for n in (WIDE - 0.12, 0.12):                   # the far railing, then the near one over the deck
        posts = [-14.5]
        while posts[-1] < 63:
            posts.append(posts[-1] + r.uniform(1.7, 2.9))
        posts = np.array(posts)
        tall = 1.18 + r.normal(0, 0.03, len(posts))
        dz = np.interp(ts, posts, tall) - 1.18      # the rails ride up and down with the posts
        for z0, z1 in ((0.9, 1.1), (0.42, 0.57)):
            a, _ = at(ts, n, z0 + dz)
            b, _ = at(ts, n, z1 + dz)
            q = np.vstack([a, b[::-1]])
            rails.area(q)
            if n < 1:
                deck.area(q, 0)
                kb.area(q, 0)
            key(kb, a, r, 0.95)
            key(kb, b, r, 0.85)
        for t, h in zip(posts, tall):
            lean = r.normal(0, 0.05)
            q, _ = at([t - 0.11, t + 0.11, t + 0.11 + lean, t - 0.11 + lean], n, [0, 0, h, h])
            rails.area(q)
            if n < 1:
                deck.area(q, 0)
                kb.area(q, 0)
            for e, hw in ((q[[0, 3]], 0.8), (q[[1, 2]], 0.8), (q[[3, 2]], 0.9)):
                key(kb, e, r, hw, taper=0.1)
    return deck.get(), rails.get(), face


def piers(r, kb, face):
    """The forest of piles under the deck: at every pier a bent of four, some doubled, braced across.
    The near piles are cut in the keyblock, their lines giving out toward the water; the rest stand only
    in the grey pier block. Returns the near and far piles as blocks, and the line of their feet."""
    fx, fy = face[::-1, 0], face[::-1, 1]

    def below(p):                                   # what the beam does not hide
        p = np.asarray(p, float)
        return p[p[:, 1] > np.interp(p[:, 0], fx, fy) + 1.5]

    near, far = Block(), Block()
    for t, nxt in zip(BENTS[:-1], BENTS[1:]):
        for j, n in enumerate((0.3, 1.0, 1.7, 2.45, 3.2)):
            for tt in [t + r.normal(0, 0.05)] + ([t + r.uniform(0.35, 0.55)] if r.random() < 0.3 else []):
                w = r.uniform(0.08, 0.13) * (1.5 if r.random() < 0.2 else 1.0)
                lean = r.normal(0, 0.05)
                q, _ = at([tt - w, tt + w, tt + w + lean, tt - w + lean], n, [-0.6, -0.6, -deck_z(tt), -deck_z(tt)])
                (near if j == 0 else far).area(q)
                if j == 0 and q[0, 0] > 0.12 * W:
                    reach = r.uniform(0.45, 0.85) * min(1.0, q[0, 0] / (0.5 * W))
                    for e in (q[[0, 3]], q[[1, 2]]):
                        key(kb, below(np.linspace(e[0], e[0] + (e[1] - e[0]) * reach, 40)), r,
                            0.5 + 0.4 * min(1.0, q[0, 0] / W), taper=0.2, amp=0.2)
        if r.random() > 0.8:
            continue
        hi = [-0.9, -0.9]                           # braces across the upper part of the bay, tied above and below
        lo = [-0.9 - 0.4 * (deck_z(t) - 0.9), -0.9 - 0.4 * (deck_z(nxt) - 0.9)]
        for n, blk in ((0.3, near), (1.7, far)):
            for z0, z1 in ((hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]), (lo[0], lo[1])):
                p, k = at([t, nxt], n, [z0, z1])
                hw = 0.05 * k.mean()
                blk.line(hand(np.linspace(p[0], p[1], 20), r, 0.3), hw)
                if blk is near and p.mean(0)[0] > 0.45 * W:
                    d = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                    for e in (-1, 1):
                        key(kb, below(np.linspace(p[0], p[1], 30) + e * hw * np.array([-d[1], d[0]])), r, 0.42,
                            taper=0.1, amp=0.2)
    ts = np.linspace(-14, 62, 700)
    return near.get(), far.get(), at(ts, 0.3, -deck_z(ts))[0]


def shore(r):
    """Atake, the far bank: a low band of dark trees and roofs printed in grey-green bokashi, taller here
    and lower there, with the mist of the shower lying in its gaps. Returns its block and its waterline."""
    x = np.arange(SW, dtype=np.float32) - M
    yy = np.arange(SH, dtype=np.float32)[:, None] - M
    base = 872 + 4 * noise.line1d(SW, 300, r)
    gaps = noise.smoothstep(0.3, 1.2, noise.line1d(SW, 210, r))      # where the mist lies across the bank
    block = np.zeros((SH, SW), np.float32)
    for lift, tall, ink, soft, floor in ((20, 66, 0.5, 1.2, 0.35), (0, 86, 1.25, 0.5, 0.55)):
        ground = base - lift - 6
        top = ground - 4 * np.abs(noise.line1d(SW, 60, r))
        cx = -120.0
        while cx < W + 120:
            if lift == 0 and r.random() < 0.3:          # a long roof, hipped at both ends
                L, h = r.uniform(50, 170), r.uniform(12, 24)
                prof = np.clip(np.minimum(x - cx, cx + L - x) / (1.2 * h), 0, 1)
                top = np.minimum(top, np.where((x > cx) & (x < cx + L), ground - h * prof, top))
                cx += L + r.uniform(4, 40)
            else:                                       # a copse, its crowns of different heights
                span, hmax = r.uniform(40, 230), tall * r.uniform(0.3, 1.0)
                for _ in range(int(span / 9)):
                    c, w_ = cx + r.uniform(0, span), r.uniform(8, 24)
                    h_ = hmax * r.uniform(0.5, 1.0) * np.sin(np.pi * np.clip((c - cx) / span, 0.05, 0.95)) ** 0.6
                    top = np.minimum(top, ground - h_ * np.sqrt(np.clip(1 - ((x - c) / w_) ** 2, 0, 1)))
                cx += span + r.uniform(0, 70)
        edge = top[None, :] + soft * noise.field((SH, SW), 2.5, r)
        mask = noise.smoothstep(-0.8, 0.8, yy - edge) * noise.smoothstep(0.8, -0.8, yy - base[None, :])
        mist = floor + (1 - floor) * noise.smoothstep(top[None, :], base[None, :] - 3, yy)   # dark at the foot
        block = np.maximum(block, ink * mask * mist * (1 - 0.7 * gaps[None, :]))
    return block, base


def rain(r, slant, gap, hw):
    """A rain block: long fine lines left standing between two cuts. The rain comes in curtains, and
    now and then a line is broken where the thin ridge of wood gave way."""
    blk = Block()
    lean = np.tan(np.radians(slant))
    span = W + H * lean + 120
    curtain = noise.line1d(int(span) + 2, 230, r)
    x = -60.0
    while x < span - 60:
        y = -r.uniform(0.0, 0.5) * H
        while y < H + 20:
            L = H * r.uniform(0.08, 0.75)
            a = lean + np.tan(np.radians(r.normal(0, 0.3)))
            s = np.linspace(0, L, max(2, int(L / 6)))
            p = np.c_[x + r.normal(0, gap * 0.25) - (y + s) * a, y + s]
            path, w = relief.vcut(hand(p, r, 0.2, 6.0), hw * r.uniform(0.7, 1.3), r,
                                  taper=r.uniform(0.08, 0.3), wobble=0.2)
            if len(path) > 12 and r.random() < 0.14:
                b, g = int(r.uniform(0.25, 0.75) * len(path)), int(r.integers(1, 4))
                blk.line(path[:b], w[:b] * np.minimum(1, np.arange(b)[::-1] / 2 + 0.3))
                blk.line(path[b + g:], w[b + g:] * np.minimum(1, np.arange(len(path) - b - g) / 2 + 0.3))
            else:
                blk.line(path, w)
            y += L + H * r.uniform(0.005, 0.12)
        x += gap * r.uniform(0.55, 1.45) * (1 - 0.45 * np.tanh(curtain[int(np.clip(x + 60, 0, len(curtain) - 1))]))
    return blk.get()


def ellipse(cx, cy, rx, ry, tilt=0.0, n=28):
    a = np.linspace(0, 2 * np.pi, n)
    x, y = rx * np.cos(a), ry * np.sin(a)
    return np.c_[cx + np.cos(tilt) * x - np.sin(tilt) * y, cy + np.sin(tilt) * x + np.cos(tilt) * y]


def limb(pts, w=0.042):
    """A bare leg in stride, hip to knee to ankle, tapering, with the foot as a small wedge."""
    c = np.asarray(pts[:-1], float)
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.c_[-t[:, 1], t[:, 0]] * (w * np.linspace(1.0, 0.55, len(c)))[:, None]
    return np.vstack([c + nrm, [pts[-1]], (c - nrm)[::-1]])


def legs(dx=0.0, lift=0.0):
    """Bare legs in a long stride, the back heel lifting."""
    fore = [(0.08, 0.45), (0.19, 0.26), (0.15, 0.04), (0.22, 0.012)]
    back = [(-0.02, 0.45), (-0.07, 0.26), (-0.19, 0.08 + lift), (-0.13, 0.035 + lift)]
    return [("skin", limb([(x + dx, y) for x, y in leg], 0.026)) for leg in (back, fore)]


def fringe(x0, y0, x1, y1, n, r):
    """The ragged hem of a straw cape."""
    x, y = np.linspace(x0, x1, 2 * n + 1), np.linspace(y0, y1, 2 * n + 1)
    return np.c_[x + r.normal(0, 0.006, len(x)), y - 0.034 * (np.arange(len(x)) % 2) * r.uniform(0.6, 1.3, len(x))]


def walker(kind, r):
    """A figure as Hiroshige cut them: a hat or an umbrella, a cape or a robe, bare legs in stride.
    Parts in the order they cover one another; feet at 0, the top at 1, facing +x."""
    def j(pts):                                  # no two figures cut quite alike
        return np.asarray(pts, float) + r.normal(0, 0.005, np.shape(pts))
    if kind == "mino":
        # a straw cape hung from the neck: a bell of straw over the bent back, its hem ragged
        apex = np.array([0.12, 0.83])
        hem = fringe(0.28, 0.47, -0.17, 0.43, 5, r)
        cape = np.vstack([[apex], j([[0.24, 0.72]]), hem, j([[-0.12, 0.6], [-0.02, 0.75]])])
        straw = [("fine", np.array([apex + (0.02, -0.03), hem[i] + (0, 0.035)])) for i in (2, 4, 6, 8)]
        return legs(lift=0.03) + [("cape", cape)] + straw + [("straw", ellipse(0.2, 0.86, 0.24, 0.066, -0.3))]
    if kind == "poler":
        # leaning hard on the pole: the body laid forward along it, the back leg braced straight
        apex = np.array([0.3, 0.8])
        hem = fringe(0.27, 0.5, -0.16, 0.4, 5, r)
        cape = np.vstack([[apex], j([[0.42, 0.7], [0.34, 0.56]]), hem, j([[-0.14, 0.52], [0.05, 0.7], [0.18, 0.79]])])
        hands, tip = np.array([0.52, 0.63]), np.array([-1.95, -0.72])
        return [("skin", limb([(-0.04, 0.46), (-0.19, 0.25), (-0.35, 0.05), (-0.29, 0.0)], 0.026)),
                ("skin", limb([(0.04, 0.46), (0.19, 0.28), (0.15, 0.05), (0.23, 0.015)], 0.026)),
                ("cape", cape)] + [
            ("fine", np.array([apex + (0.0, -0.03), hem[i] + (0, 0.035)])) for i in (2, 4, 6, 8)] + [
            ("line", j([[0.36, 0.73], hands])), ("line", np.array([tip, hands + 0.04 * (hands - tip)])),
            ("straw", ellipse(0.4, 0.875, 0.22, 0.06, -0.45))]
    if kind == "kasa":
        robe = j([[0.1, 0.8], [0.25, 0.74], [0.28, 0.6], [0.22, 0.47], [0.05, 0.44], [-0.1, 0.47], [-0.11, 0.6],
                  [-0.02, 0.73]])
        return legs() + [("indigo", robe), ("line", j([[0.2, 0.72], [0.1, 0.6], [0.21, 0.55]])),
                         ("line", j([[0.0, 0.62], [0.03, 0.47]])),
                         ("straw", ellipse(0.2, 0.86, 0.23, 0.065, -0.3))]
    if kind == "pair":
        # two women in long robes under one oiled umbrella, tipped back into the wind
        robe = np.array([[0.03, 0.76], [0.15, 0.74], [0.18, 0.5], [0.15, 0.28], [0.24, 0.08], [0.0, 0.06],
                         [0.03, 0.3], [0.0, 0.5], [-0.02, 0.64]])
        robe[:, 0] += 0.16 * (robe[:, 1] - 0.07)                 # hurrying, bent on into the rain
        obi = np.array([[0.0, 0.62], [0.17, 0.6], [0.18, 0.53], [0.0, 0.55]])
        feet = [("line", np.array([[0.15, 0.07], [0.2, 0.005]])), ("line", np.array([[0.05, 0.065], [0.01, 0.015]]))]
        behind = np.array([-0.2, 0.0])
        canopy = ellipse(-0.04, 0.86, 0.44, 0.115, 0.2)
        apex = np.array([-0.04, 0.885])
        return ([(n, p + behind) for n, p in feet] + [("cape", j(robe + behind)), ("red", j(obi + behind))]
                + feet + [("indigo", j(robe)), ("red", j(obi)), ("line", j([[-0.05, 0.88], [0.05, 0.62]])),
                          ("sumi", canopy)]
                + [("cut", np.array([apex, apex + 0.92 * (canopy[i] - apex)])) for i in (1, 4, 7, 10, 13)])
    if kind == "parasol":
        canopy = ellipse(-0.06, 0.9, 0.32, 0.095, 0.22)
        apex = np.array([-0.06, 0.925])
        return legs(lift=-0.03) + [
            ("indigo", j([[0.02, 0.8], [0.15, 0.76], [0.19, 0.6], [0.16, 0.46], [0.0, 0.43], [-0.08, 0.46],
                          [-0.07, 0.62]])),
            ("line", j([[0.13, 0.7], [0.06, 0.6], [0.15, 0.56]])),
            ("line", j([[-0.04, 0.9], [0.07, 0.58]])), ("straw", canopy),
        ] + [("fine", np.array([apex, canopy[i]])) for i in (1, 4, 7, 10, 13)]
    raise ValueError(kind)


CROWD = [(3.0, 1.2, "mino", 1), (9.0, 2.1, "pair", -1), (11.5, 1.0, "kasa", 1), (23.5, 1.9, "parasol", -1),
         (34.5, 1.4, "mino", 1)]


def raft(r):
    """A long raft of round logs drifting down toward the bridge, their sawn ends showing at the near end:
    its parts, far log to near, and where its poler stands."""
    c = water(np.array([600.0, 2170.0]))[:2]
    u = np.array([np.cos(-0.2), np.sin(-0.2)])       # downstream, and a little toward us
    v = np.array([-u[1], u[0]])
    parts = []
    for i in range(6):
        a, rad = (2.5 - i) * 0.33, 0.17 * r.uniform(0.75, 1.2)
        ends = np.array([c + u * (-6.5 + r.normal(0, 0.4)) + v * a, c + u * (6.5 + r.normal(0, 0.45)) + v * a])
        (p0, p1), k = proj(np.c_[ends, [rad, rad]])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        up = np.array([d[1], -d[0]]) * (1 if d[0] > 0 else -1)
        r0, r1 = rad * k[0] * 0.92, rad * k[1]             # a log tapers toward its top end
        parts += [("log", np.array([p0 + up * r0, p1 + up * r1, p1 - up * r1, p0 - up * r0])),
                  ("shade", np.array([p0 - up * r0 * 0.35, p1 - up * r1 * 0.35, p1 - up * r1, p0 - up * r0])),
                  *[("glint", np.linspace(p0 + up * r0 * 0.45, p1 + up * r1 * 0.45, 40)[g0:g0 + int(r.integers(6, 14))])
                    for g0 in r.integers(0, 32, 3)],
                  ("tan", p1 + ellipse(0, 0, 0.5 * r1, r1)), ("fine", p1 + ellipse(0, 0, 0.22 * r1, 0.45 * r1))]
    for b in (-4.6, -4.25, 4.4, 4.75):               # the lashings
        q, _ = proj(np.c_[np.array([c + u * b + v * 1.0, c + u * b - v * 1.0]), [0.34, 0.34]])
        parts.append(("line", q))
    foot, k = proj(np.r_[c - u * 4.0, 0.3])
    return parts, foot, k


def cut(parts, kb, colour, sil, r, hw):
    """Cut the parts of a figure or of the raft, each over those before it: its colour and its keyline."""
    for name, p in parts:
        if name in ("line", "fine"):
            key(kb, p, r, hw * (1.0 if name == "line" else 0.6), taper=0.12, amp=0.08, vary=0.2)
        elif name == "cut":
            kb.line(p, 0.45, 0)
        elif name == "shade":
            colour["shade"].area(p)
        elif name == "glint":                       # wet light along the top of a log: the block cut away
            colour["log"].line(*relief.vcut(hand(p, r, 0.3), 0.7, r, taper=0.3), fill=0)
        else:
            for b in colour.values():
                b.area(p, 0)
            kb.area(p, 255 if name == "sumi" else 0)
            sil.area(p)
            if name != "sumi":
                colour[name].area(p)
                key(kb, np.vstack([p, p[:1]]), r, hw, taper=0.01, amp=0.1, vary=0.2)


def wet(r):
    """The planks shine with water: long grey streaks along the deck, cut on the grey rain block."""
    blk = Block()
    for _ in range(26):
        t0, n = r.uniform(-12, 50), r.uniform(0.5, WIDE - 0.4)
        p, _ = at(np.linspace(t0, t0 + r.uniform(2.5, 9), 40), n)
        blk.line(*relief.vcut(hand(p, r, 0.3), r.uniform(0.6, 1.0), r, taper=0.3, wobble=0.25))
    return blk.get()


def paint(seed=1857):
    r = noise.rng(seed)
    sheet = paper.washi((SH, SW), seed, tint="#efe6d1", margin=(46, 44), fibre_density=0.9)
    sheet.alpha = paper.deckle((SH, SW), (46, 44), r, ragged=2.0)
    yy = np.arange(SH, dtype=np.float32)[:, None] - M
    xx = np.arange(SW, dtype=np.float32)[None, :] - M
    inside = (noise.smoothstep(-0.5, 0.5, yy) * noise.smoothstep(-0.5, 0.5, H - yy)
              * noise.smoothstep(-0.5, 0.5, xx) * noise.smoothstep(-0.5, 0.5, W - xx))

    kb, sil = Block(), Block()
    deck, rails, face = bridge(r, kb)
    near, far, feet = piers(r, kb, face)
    top = np.interp(xx[0], face[::-1, 0], face[::-1, 1])[None, :]
    drop = yy - top
    frac = drop / (np.interp(xx[0], feet[::-1, 0], feet[::-1, 1])[None, :] - top)
    near_us = noise.smoothstep(-150, W, xx)
    fade = (1 - (0.75 - 0.4 * near_us) * noise.smoothstep(0.15, 1.05, frac)) * (0.4 + 0.6 * near_us)
    shade = 0.6 * noise.smoothstep(-1, 1, drop) * (1 - noise.smoothstep(4, 46, drop))     # under the deck
    piles = np.maximum(np.maximum(near, 0.7 * far) * fade, shade) * noise.smoothstep(0, 1.5, drop)
    land, waterline = shore(r)
    river = noise.smoothstep(-0.8, 0.8, yy - waterline[None, :])

    colour = {name: Block() for name in ("straw", "cape", "skin", "indigo", "red", "log", "shade", "tan")}
    logs, foot, k = raft(r)
    cut(logs, kb, colour, sil, r, 0.75)
    people = [(at(t, n)[0], 1.45 * at(t, n)[1], kind, d) for t, n, kind, d in CROWD] + [(foot, 1.45 * k, "poler", 1)]
    for (fx, fy), s, kind, d in people:
        lean = 0.0 if kind == "poler" else (0.22 if d > 0 else 0.1)       # bent on into the rain
        cut([(name, np.c_[fx + d * s * (p[:, 0] + lean * p[:, 1]), fy - s * p[:, 1]]) for name, p in walker(kind, r)],
            kb, colour, sil, r, 0.3 + 0.0035 * s)
    sil = sil.get()
    colour = {name: b.get() for name, b in colour.items()}
    clear = np.clip(1 - deck - rails - sil, 0, 1)              # the bridge and the people are cut out of the water
    deck, rails = deck * (1 - sil), rails * (1 - sil)
    frame = Block()
    for a, b in zip([[0, 0], [W, 0], [W, H], [0, H]], [[W, 0], [W, H], [0, H], [0, 0]]):
        a, b = np.array(a, float), np.array(b, float)
        key(frame, np.linspace(a - (b - a) * 0.004, b + (b - a) * 0.004, 2), r, hw=1.8, taper=0.01, amp=0.5)
    keyblock = np.maximum(kb.get() * inside, frame.get())

    blocks = [
        (inside * (1 - river) * (0.2 + 0.65 * wipe(250, 880, r, 12)), SKY, (0.0, 0.0), 0.0, 0.05),
        (inside * wipe(30, 470, r, 30), CLOUD, (-1.2, 0.6), 0.0003, 0.05),
        (inside * land, SHORE, (2.2, -1.4), 0.0, 0.05),
        (inside * river * clear * (0.3 + 0.7 * wipe(940, 878, r, 8)), PALE, (0.6, 1.2), -0.0002, 0.08),
        (inside * river * clear * wipe(H + 40, 1620, r, 30) * 1.6, PRUSSIAN, (-0.8, 0.9), 0.0002, 0.07),
        (inside * (deck + colour["tan"]), TAN, (2.1, 0.6), 0.0, 0.1),
        (inside * (rails + colour["red"]), RED, (-1.4, -2.2), 0.0004, 0.06),
        (inside * colour["log"], LOG, (1.6, -1.2), 0.0, 0.12),
        (inside * piles, PIER, (0.9, -0.6), 0.0, 0.05),
        (inside * colour["shade"], PIER, (-0.7, 1.3), 0.0, 0.05),
        (inside * colour["straw"], STRAW, (-1.6, 0.9), 0.0, 0.05),
        (inside * colour["cape"], CAPE, (1.3, 1.1), 0.0, 0.05),
        (inside * colour["skin"], SKIN, (0.5, -0.9), 0.0, 0.0),
        (inside * colour["indigo"], INDIGO, (-1.1, 1.5), 0.0, 0.05),
        (inside * rain(r, 3.0, 19, 0.5), SUMI, (0.5, -0.4), 0.0, 0.0),
        (inside * np.maximum(rain(r, 7.5, 13, 0.5), wet(r) * deck), GREY, (-0.9, 0.3), 0.0002, 0.0),
        (keyblock, SUMI, (0.0, 0.0), 0.0, 0.03),
    ]
    img = sheet.color.copy()
    pressed = []
    for block, ink, shift, turn, grain in blocks:
        wood = relief.woodgrain((SH, SW), r, along=1.0)
        film = relief.ink_film(block, r, roller=0.0, squash=0.0, grain=wood, grain_strength=grain) \
            * (1 + 0.06 * noise.fbm((SH, SW), 26, r, octaves=3))
        dens = relief.pull(block, film, sheet, r, pressure=1.0, shift=shift, turn=turn)
        img = glaze(img, ndimage.gaussian_filter(dens, 0.6) * 1.2, ink)
        pressed.append(np.clip(block, 0, 1))
    img = relief.emboss(img, pressed, depth=0.02)
    return plate.mount(img, sheet, shadow=0.35)
