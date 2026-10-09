"""The Monsoon Breaking. Oil on a wooden panel and its frame.

Howard Hodgkin painted on wood, often on a panel with a frame already fixed round it, and painted the frame
as part of the picture. A painting could take him years. He laid something down, lived with it, and came
back months later to add a few strokes, so the last layer lies over earlier ones that still show at its
edges, and through it wherever it went on thin. The pictures are of remembered moments: a room, a view, a
meal, a kind of weather, very often India, where he went back all his life. They are made of very few marks,
each a single sweep of a broad brush, with dots and arcs, and the colour glows because it lies over darker
colour beneath it.

Here the monsoon breaks over a garden. The panel is two mahogany boards glued edge to edge, in a pine frame
mitred at the corners. First the panel was scrubbed over thinly with cadmium orange and red, the heat before
the rains, all but a strip at the left, and the frame was given a thin coat of red and orange. Then came the
depth: alizarin crimson glazed over the window, aubergine over it high on the right, and low on the left the
garden in a green so deep it is nearly black, with the orange still glowing wherever the glaze went on thin;
and the frame a first coat of viridian. A second frame went on inside the first in a deep cadmium red,
loaded along the top and stopping short of the corner, and on the left scrubbed so thin over the bare
mahogany that its grain shows. The frame itself was gone over in a bottle green, short of its edges, thin
enough in places for the viridian and the red beneath to come through. The storm is one sweep of a broad
brush loaded with ultramarine, indigo and a streak of paler blue. It lands thick on the red at the left, rides
across the top and curls down the right side and out over the frame, its bunches of bristles leaving tracks
of indigo and pale blue that come and go; as the load runs out it thins, so that the crimson shows through it
as violet, and skips, so that the red shows between its tracks, and it ends in the marks of single bristles.
Under it, the one hot thing: a scrub of cadmium orange, and inside it, wet into wet, a thick buttery stroke of
orange and yellow, the light breaking under the storm. Months later the rain went over it all, a pale
grey-blue dragged down on the slant with a broad, nearly dry brush. It left paint only along the bunches of
bristles that still held some, and there only on the ridges of the paint beneath, so the colours show between
its streaks and through them; it comes and goes at both ends. Last came the fat dots of cadmium red and orange
along the top of the frame, each pressed down and lifted with a turn of the brush.
"""

import zlib

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri

from atelier import brush, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Monsoon Breaking"
DATE = "2026"
MEDIUM = "Oil on a mahogany panel in a pine frame, the frame painted as part of the picture"
AFTER = ("Howard Hodgkin, the paintings on wood of the 1980s: Venice Evening (1984–85), Rain (1984–89), "
         "Love Letter (1984–88), In Tangier (1987–90), Indian Sky (1988–89)")
ROOM = "Colour Itself"
YEAR = 1988
PLACE = "London"
REGION = "Europe"
NOTE = ("The monsoon breaking over a garden, remembered. Inside a deep green frame and a red one, one sweep of "
        "ultramarine curls over a crimson dark with the light glowing hot beneath it, and the rain is dragged "
        "across in a pale, dry veil that lets the colour show through.")

H, W = 2520, 2880
F = 230                       # the frame's width, px
BEVEL = 22                    # its inner edge, sloping down to the panel
RISE = 14.0                   # how far the frame stands above the panel, px
LIGHT = (-0.6, -0.5, 0.62)


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


# ---------------------------------------------------------------- the wood

def rings(h, w, r, ring=13.0, depth=420.0, slope=0.25, heart=None, knot=None, wander=5.0):
    """A board sawn from a log, the grain running along x. The growth rings are cylinders round the heart
    of the tree; the saw cut a plane through them a little askew to its axis, so they come up as long
    lines that bow into arches where the plane passes near the heart. A knot is a branch cut across:
    the rings swerve round it and it shows as a dark eye with rings of its own.
    -> latewood (h,w) in [0,1], fine streaks along the grain, the knot's eye"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yc = (h * 0.5 if heart is None else heart) + 0.08 * h * noise.line1d(w, 1500, r)[None, :]
    d = depth + slope * (xx - w * r.uniform(0.2, 0.8)) + 25 * noise.line1d(w, 700, r)[None, :]
    y = yy.copy()
    eye = np.zeros((h, w), np.float32)
    if knot is not None:
        kx, ky, a, b = knot
        q = np.hypot((xx - kx) / a, (yy - ky) / b)
        eye = noise.smoothstep(1.15, 0.85, q)
        # the rings part round it: pushed away from its middle, most strongly beside it, for a few lengths along
        push = 1.6 * b * np.exp(-((xx - kx) / (3.5 * a)) ** 2) * np.exp(-np.maximum(np.abs(yy - ky) - b, 0) / (2.5 * b))
        y += np.sign(yy - ky) * push * noise.smoothstep(0.6, 1.4, q)
    R = np.sqrt((y - yc) ** 2 + d ** 2)
    R += wander * noise.stretched((w, h), 40, 900, r).T + 0.8 * noise.stretched((w, h), 10, 700, r).T
    span = np.linspace(R.min(), R.max(), 512)
    n = R / ring + np.interp(R, span, np.cumsum(0.6 * noise.line1d(512, 12, r)) * (span[1] - span[0]) / ring)
    f = n % 1.0
    late = noise.smoothstep(0.5, 0.9, f) * (1 - noise.smoothstep(1 - 1.6 / ring, 1.0, f))
    late *= 0.75 + 0.25 * noise.stretched((w, h), 30, 400, r).T
    if knot is not None:
        late = late * (1 - eye) + eye * (0.5 + 0.5 * np.cos(2 * np.pi * q * 5 + 0.6 * noise.field((h, w), 8, r)))
    pores = noise.stretched((w, h), 1.3, 70, r).T
    return late.astype(np.float32), pores, eye


def board(h, w, r, early, late, figure=0.85, **kw):
    """The planed face of a board: its colour, and the faint relief of the grain (px)."""
    lw, pores, eye = rings(h, w, r, **kw)
    E, L = lin(early), lin(late)
    k = np.clip(figure * lw + 0.12 * noise.smoothstep(0.8, 2.2, pores), 0, 1)[..., None]
    col = E * (1 - k) + L * k
    col = col * np.exp(0.1 * noise.fbm((h, w), 400, r, octaves=3))[..., None]
    col = col * (1 - 0.35 * eye[..., None]) + lin("#4a2010") * 0.35 * eye[..., None] * eye[..., None]
    relief = 0.3 * lw - 0.12 * noise.smoothstep(0.5, 2.0, pores) + 0.8 * eye
    return col.astype(np.float32), relief.astype(np.float32)


def support(r):
    """The panel and the frame round it, mitred at the corners, before any paint.
    -> colour, relief (px), the frame's profile (px, for the light only)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)
    d = np.minimum(dx, dy)                                # how far in from the outer edge
    across = dy < dx                                      # the top and bottom members, grain along x
    frame = d < F
    # each member its own stick of pine, and the panel two boards of mahogany glued edge to edge
    top, top_r = board(H, W, r, "#c98d50", "#8f5228", ring=11.0, depth=300, slope=0.18,
                       knot=(2140, 118, 34, 22))
    bot, bot_r = board(H, W, r, "#c48448", "#8a4b22", ring=12.0, depth=520, slope=-0.12, heart=H - 140)
    side_c, side_r = board(W, H, r, "#c78a4e", "#8c4f25", ring=12.5, depth=380, slope=0.2,
                           knot=(1650, W - 112, 30, 20))
    side_c, side_r = side_c.transpose(1, 0, 2), side_r.T
    panel_c, panel_r = board(H, W, r, "#97573a", "#6e3822", ring=9.0, depth=900, slope=0.05, figure=0.5)
    lower_c, lower_r = board(H, W, r, "#925236", "#6a3520", ring=8.5, depth=1100, slope=-0.08, figure=0.5)
    seam = int(H * 0.58)
    panel_c[seam:], panel_r[seam:] = lower_c[seam:], lower_r[seam:]
    hor = np.where((yy < H / 2)[..., None], top, bot)
    hor_r = np.where(yy < H / 2, top_r, bot_r)
    col = np.where(frame[..., None], np.where(across[..., None], hor, side_c), panel_c)
    relief = np.where(frame, np.where(across, hor_r, side_r), panel_r)
    tint = np.where(across, np.where(yy < H / 2, 1.0, 0.94), np.where(xx < W / 2, 0.97, 1.03))
    col *= np.where(frame, tint, 1.0)[..., None]
    # the joints: the mitres from corner to corner, the seam in the panel, the gap where panel meets frame
    jag = 0.6 * noise.field((H, W), 30, r)
    mitre = frame * noise.smoothstep(1.4, 0.4, np.abs(dx - dy + jag)) * (np.maximum(dx, dy) < F + 2)
    rebate = noise.smoothstep(2.0, 0.3, np.abs(d - F + jag))
    glue = (~frame) * noise.smoothstep(1.2, 0.2, np.abs(yy - seam - 0.5 * jag))
    joint = np.maximum.reduce([mitre, rebate, 0.6 * glue])
    col *= (1 - 0.65 * joint)[..., None]
    # the profile: a rounded outer edge, a flat face, a bevel down to the panel
    arris = np.sqrt(np.clip(1 - ((10 - np.minimum(d, 10)) / 10) ** 2, 0, 1))
    bevel = noise.smoothstep(F, F - BEVEL, d)
    profile = RISE * (0.5 + 0.5 * arris) * bevel - 1.5 * joint
    return col.astype(np.float32), relief.astype(np.float32), profile.astype(np.float32)


def shadow(profile, k=40):
    """Where the frame shades the panel from the lamp: march toward the light and see if anything stands
    above the line of sight."""
    L = np.asarray(LIGHT, np.float64)
    e = L[:2] / np.hypot(*L[:2])
    tan = L[2] / np.hypot(*L[:2])
    s = 4
    p = profile[::s, ::s]
    over = np.zeros_like(p)
    for t in range(1, k // s + 1):
        q = ndimage.shift(p, (-e[1] * t, -e[0] * t), order=1, mode="nearest")
        over = np.maximum(over, q - p - tan * t * s)
    over = ndimage.zoom(over, s, order=1)[:H, :W]
    return 1 - 0.45 * noise.smoothstep(0, 6, ndimage.gaussian_filter(over, 3))


# ---------------------------------------------------------------- the brush

class Surface:
    """The panel as the paint finds it: colour, height of the paint (px), and how wet it still is."""

    def __init__(self, rgb, height, r):
        self.rgb, self.height = rgb, height
        self.H, self.W = height.shape
        self.wet = np.zeros(height.shape, np.float32)
        self.gloss = np.full(height.shape, 0.05, np.float32)    # bare wood is matt; each paint dries its own way
        # the marks of a broad brush in its own space, rows along the stroke and columns across it:
        # single bristles, bunches of them, the uneven load of the whole brush, and where a dry one skips
        self.fine = noise.stretched((4096, 1024), 1.3, 300, r)
        self.fine2 = noise.stretched((4096, 1024), 2.2, 150, r)
        self.mid = noise.stretched((4096, 1024), 4.5, 700, r)
        self.coarse = noise.stretched((4096, 1024), 20.0, 1600, r)
        self.edge = noise.stretched((4096, 1024), 1.0, 40, r)
        self.skip = noise.stretched((4096, 1024), 3.0, 50, r)
        self.patch = noise.stretched((4096, 1024), 12.0, 500, r)
        self.blob = noise.stretched((4096, 1024), 25.0, 90, r)
        self.clump = noise.field((1024, 1024), 22, r)      # a dry brush's broken touch, any way round
        self.grain = noise.field((1024, 1024), 5, r)

    def sitting(self):
        """A sitting ends: what is on the panel dries before the next."""
        self.wet[:] = 0

    @staticmethod
    def look(T, a, b):
        # the textures do not tile: read past their edge they run back on themselves, which leaves no seam
        return ndimage.map_coordinates(T, [a, b], order=1, mode="mirror")

    def tracks(self, s, u, o, w, fine=1.0):
        """Bristle tracks in a stroke's own space (s along it, u across, px; `o` 12 offsets, `w` its half-width):
        bunches of bristles that start and stop, drift a little across, close up in one place and spread
        apart in the next, every width from a few hairs (as many more as `fine`) to a hand's breadth.
        -> (n,), unit variance"""
        u = u + 0.07 * w * self.look(self.blob, s * 0.25 + o[0], u * 0.1 + o[1])
        z = 0.6 * self.look(self.coarse, s * 2.5 + o[4], u * 0.25 + o[5]) \
            + 0.45 * self.look(self.patch, s + o[6], u * 0.3 + o[7])
        if fine:
            bunch = noise.smoothstep(-0.8, 1.0, self.look(self.blob, s * 0.35 + o[2], u * 0.12 + o[3]))
            z = z + fine * bunch * (0.7 * self.look(self.mid, s * 1.5 + o[8], u * 0.45 + o[9])
                                    + 0.4 * self.look(self.fine2, s * 1.2 + o[10], u * 0.8 + o[11]))
        return (z - z.mean()) / (z.std() + 1e-6)

    def dragged(self, C, N, L, s, u, box, reach):
        """What a brush carries of the wet paint it passes through. At each point of a stroke (path C, normals
        N, length L) it holds what its bristles met of the wet paint beneath along their tracks since it
        landed, the nearest the most, and gives it back over about `reach` px, so the paint it picks up is
        drawn out along it. -> how much (n,), and its colour (n, 3)"""
        x0, y0, x1, y1 = box
        wet = self.wet[y0:y1, x0:x1]
        um = np.abs(u).max() + 2
        uu = np.arange(-um, um + 2, 2.0)                       # the stroke's own space: a row per px along it
        X, Y = C[:, :1] + N[:, :1] * uu - x0, C[:, 1:] + N[:, 1:] * uu - y0
        step = L / (len(C) - 1)
        q = np.exp(-step / reach)
        norm = (1 - q ** np.arange(1, len(C) + 1))[:, None]
        got = [ndimage.map_coordinates(
            signal.lfilter([1 - q], [1, -q], ndimage.map_coordinates(A, [Y, X], order=1, mode="nearest"), axis=0) / norm,
            [s / step, (u + um) / 2], order=1, mode="nearest")
            for A in [wet] + [wet * self.rgb[y0:y1, x0:x1, c] for c in range(3)]]
        return got[0], np.stack(got[1:], 1) / (got[0][:, None] + 1e-6)

    def sweep(self, pts, w, colours, g, share=(0.7, 0.25, 0.05), thick=3.0, spent=0.6, ends=0.8, dry=0.0,
              hide=1.0, glaze=1.0, pickup=0.4, tails=0.6, start=0.3, splay=0.12, catch=0.0, lips=0.03,
              streaks=1.0, grooves=0.05, level=0.6, feather=0.15, gloss=0.3, mottle=1.0, vary=0.3, tracks=0.0,
              runs=0.82):
        """One sweep of a broad loaded brush along `pts` (x, y), its half-width w px.

        pts      the path, rows (x, y) or (x, y, pressure); pressure widens the mark and lays more paint
        colours  (K, 3) the load, linear RGB: the paint and what else is on the brush, which streak along it
        share    how much of the brush each holds
        thick    the paint's height where it lands, px; it thins as the load is spent (`spent`)
        ends     how far it breaks up over the last stretch, where the load runs out; `runs`, the share of its
                 length after which that begins
        dry      how little the brush carries from the start: dragged nearly dry it breaks all along its
                 length (this share of it) and catches the high points of what lies beneath (a scumble), as
                 much the more `catch`
        hide     how well the paint covers at a px of thickness; thinner, it lets what is under it glow
                 through, as a glaze (`glaze` of its colour) and a veil
        pickup   how much of the wet paint it rides over it drags into itself, in streaks
        tails    how far single bristles drag on past the end, as a share of the half-width
        start    how round and ragged it is where it lands
        splay    how far the bristles spread as the brush lifts
        streaks  how unevenly the brush is loaded across its width
        tracks   how far its streaks of colour and of paint follow bunches of bristles that start and stop,
                 wander and bunch, instead of running its whole length side by side
        grooves  how deep the bristles comb the paste, as a share of its thickness
        level    how far the paste fills the hollows and buries the ridges of what it goes over
        feather  the share of the half-width over which its sides break into single bristle marks
        gloss    how glossy it dries; mottle, how much it shades lighter and deeper along the bristles
        vary     how much thicker and thinner it goes on along its length
        """
        H, W = self.H, self.W
        C = brush.path(np.array([(*p, 1.0)[:3] for p in pts], np.float64), 1.0)
        press, C = C[:, 2], C[:, :2]                                # how hard the brush bears down, and where
        sc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        L = sc[-1]
        T = ndimage.gaussian_filter1d(np.gradient(C, axis=0), 4, axis=0, mode="nearest")
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        wide = w * press.max()
        pad = wide * (1.6 + splay) + tails * w + 12
        x0, y0 = np.maximum(np.floor(C.min(0) - pad).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(C.max(0) + pad).astype(int) + 1, (W, H))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
        dist, i = cKDTree(C).query(P, distance_upper_bound=wide * (1.3 + splay) + 8, workers=-1)
        near = np.isfinite(dist)
        P, i = P[near], i[near]
        D = P - C[i]
        s = sc[i] + (D * T[i]).sum(1)
        u = (D * N[i]).sum(1)
        t = np.clip(s / L, 0, 1)
        o = g.uniform(0, 4096, 24)
        # the bristles never run quite parallel: each set drifts across, and as the brush turns in the hand
        # the whole pattern of them wanders and opens and closes
        lean = g.normal(0, 0.025 + 0.06 * dry, 12)
        turn = 0.04 * w * (1 + 2 * dry) * noise.line1d(int(L) + 2, 400, g)
        pres = 1 + 0.08 * noise.line1d(int(L) + 2, 600, g)
        sj = np.clip(s, 0, L).astype(int)
        uw = (u + turn[sj]) * pres[sj]
        tex = lambda T_, j, k=1.0: self.look(T_, s + o[j], k * uw + lean[j // 2] * s + o[j + 1])
        # no two brushes alike: a coarse hog leaves wider tracks than a soft one
        fine, mid, coarse = tex(self.fine, 0, g.uniform(0.5, 1.6)), tex(self.mid, 2, 0.7), tex(self.coarse, 4, 0.4)
        # the brush, a little wider where it bears down and spreading as it lifts; each side ragged on its own,
        # the more so the drier it is
        ws = w * press[i] * (1 + 0.04 * self.look(self.coarse, s * 0.3 + o[6], np.full_like(s, o[7]))) \
            * (1 + splay * noise.smoothstep(0.75, 1.0, t))
        side = np.sign(u)
        rag = 1.0 * self.look(self.edge, s + o[8], side * 300 + o[9]) \
            + (0.025 + 0.08 * dry) * w * self.look(self.mid, s * 0.5 + o[8], side * 300 + o[9]) \
            + 0.03 * w * self.look(self.blob, s + o[10], side * 300 + o[11])
        v = u / ws
        inside = np.clip(ws + rag - np.abs(u) + 0.5, 0, 1)
        # where it landed: the brush comes down askew, its corners rounded, the edge wavering a little
        skew = g.normal(0, 0.3)
        s0 = start * w * (1 - np.sqrt(np.clip(1 - v * v, 0, 1))) + skew * u \
            + 0.12 * w * self.look(self.coarse, np.full_like(u, o[10]), u * 0.15 + o[11]) \
            + 0.03 * w * self.look(self.blob, np.full_like(u, o[14]), u * 0.6 + o[15]) \
            + 1.2 * self.look(self.edge, np.full_like(u, o[12]), u * 2 + o[13])
        landed = np.clip(s - s0 + 0.5, 0, 1)
        # where it lifted: the bristles drag on in bunches, each its own way
        reach = L + tails * w * noise.smoothstep(-1.2, 1.4, 0.35 * fine + 0.8 * mid + 0.6 * coarse) \
            + g.normal(0, 0.2) * u
        lifted = np.clip(reach - s + 0.5, 0, 1)
        geo = inside * landed * lifted
        # the load left on the brush, heaped where it landed, and unevenly across it: each bunch of bristles
        # carries its own, and keeps it along the stroke
        left = (1 - dry) * (1 - spent * t ** 1.4) * (1 + 0.5 * np.exp(-np.maximum(s - s0, 0) / (0.6 * w)))
        body = np.clip(1 - v * v, 0, 1) ** 0.7                 # less paint toward its sides, so they go thin
        # and it goes on thick in one place and thin in the next, where the brush was pressed or lifted
        lay = np.exp(vary * noise.line1d(int(L) + 2, max(1.5 * w, 60), g))[sj]
        if tracks:
            ot = g.uniform(0, 4096, 24)
            hue, load = self.tracks(s, uw, ot[:12], w * tracks), self.tracks(s, uw, ot[12:], w * tracks, 0.5)
            spare = self.tracks(s, uw, ot[12:], w * tracks, 0.0)   # the bunches of bristles that carry least
        depth = thick * left * body * (0.4 + 0.6 * press[i]) * lay \
            * np.clip(1 + streaks * (0.36 * load if tracks else 0.35 * coarse + 0.1 * mid), 0.1, None)
        # it breaks up only as the load runs out, over the last stretch of the stroke (`ends`), in streaks
        # along it; or, dragged nearly dry, all along it and at its sides first, in broken patches, the
        # poorer bunches of bristles skipping first and over the hollows of what is beneath sooner than
        # over its ridges. Its sides thin out into the marks of single bristles (`feather`).
        hold = ndimage.gaussian_filter(self.height[y0:y1, x0:x1], 2)
        lump = hold - ndimage.gaussian_filter(hold, 15)
        high = np.clip(lump / (np.std(lump) + 0.03), -2, 2).ravel()[near]
        rim = noise.smoothstep(1 - feather, 1.02, np.abs(v))
        frac = np.clip(dry + (1 - dry) * ends * noise.smoothstep(runs, 1.05, t) ** 1.2 + 0.6 * dry * v * v
                       + 1.3 * dry * noise.smoothstep(0.18, 0.0, t) + 0.85 * rim, 0.0005, 0.97)
        patch, blob = tex(self.patch, 14), tex(self.blob, 16)
        dw = min(2 * dry, 1.0)
        grain = self.look(self.grain, s + o[8], u + o[9])
        z = (1 - dw) * (0.45 * coarse + 0.45 * blob + 0.3 * patch + 0.2 * mid + 0.1 * fine) / 0.74 \
            + dw * (0.35 * coarse + 0.45 * patch + 0.45 * blob + 0.5 * self.look(self.clump, s * 0.35 + o[6], u + o[7])
                    + 0.35 * self.look(self.grain, s * 0.4 + o[8], u + o[9])) / 0.95
        e = noise.smoothstep(1 - 2 * feather, 1.0, np.abs(v))
        z = z * (1 - e) + e * (0.5 * fine + 0.5 * mid + 0.3 * coarse + 0.6 * self.look(self.clump, s * 0.25 + o[10], u + o[11])
                               + 0.2 * grain) / 1.0
        if tracks:                                              # the bunches that carry least skip first
            z = (z + spare) / np.sqrt(2)
        z = (z + catch * high) / np.sqrt(1 + catch ** 2)
        soft = 0.5 + 0.3 * dw
        a = geo * noise.smoothstep(-soft, soft, z - ndtri(frac))
        k = a > 0.002
        if not k.any():
            return
        idx = (P[k, 1] * W + P[k, 0]).astype(np.int64)
        a, depth, fine, mid, v, s, u, grain, patch = a[k], depth[k], fine[k], mid[k], v[k], s[k], u[k], grain[k], patch[k]
        flat = ndimage.gaussian_filter(hold, 12).ravel()[near][k]
        rgb, height, wet = self.rgb.reshape(-1, 3), self.height.reshape(-1), self.wet.reshape(-1)
        under = rgb[idx]
        # the colours on the brush, in streaks along it
        K = len(colours)
        cuts = np.cumsum(np.asarray(share[:K], np.float64) / np.sum(share[:K]))[:-1]
        f = ndtr(1.6 * (0.91 * hue[k] if tracks else
                        0.9 * self.look(self.coarse, s * 1.0 + o[18], u * 0.2 + o[19]) + 0.15 * mid))
        lc = np.repeat(np.log(np.clip(colours[:1], 1e-4, 1)), len(idx), 0)
        for j in range(1, K):
            lc += (np.log(np.clip(colours[j], 1e-4, 1)) - lc) \
                * noise.smoothstep(cuts[j - 1] - 0.15, cuts[j - 1] + 0.15, f)[:, None]
        # wet paint underneath is caught up by bunches of bristles and drawn out along the stroke in streaks
        if pickup > 0 and self.wet[y0:y1, x0:x1].any():
            got, carried = self.dragged(C, N, L, s, u, (x0, y0, x1, y1), 0.6 * w)
            m = pickup * got * noise.smoothstep(-0.2, 0.9, self.look(self.mid, s * 0.5 + o[20], u * 0.5 + o[21]))
            lc += (np.log(np.clip(carried, 1e-4, 1)) - lc) * np.clip(m, 0, 0.8)[:, None]
        # never quite mixed: a shade lighter or deeper along single bristles
        col = np.exp(lc + mottle * (0.05 * fine + 0.03 * grain)[:, None])
        alpha = (1 - np.exp(-hide * depth))[:, None]
        tint = np.clip(depth / 1.5, 0, 1)[:, None] * glaze
        out = under * col ** tint * (1 - alpha) + col * alpha
        rgb[idx] = under + (out - under) * a[:, None]
        # the paste: combed by the bristles, pushed up along the edges, lumpy here and there where it lies thick;
        # it fills the hollows of what it goes over and buries its ridges, the more the more it carries, and wet
        # paint beneath is pushed aside
        comb = grooves * depth * noise.smoothstep(0.2, -0.8, fine + 0.6 * self.look(self.fine2, s + o[22], u + o[23])) \
            * noise.smoothstep(-0.6, 0.8, patch)
        lip = lips * depth * np.exp(-((np.abs(v) - 0.85) / 0.12) ** 2)
        lumpy = 0.1 * noise.smoothstep(0.3, 1.5, self.look(self.clump, s * 0.3 + o[2], u * 0.3 + o[3]))
        paste = np.maximum(depth * (1 + lumpy * mid) - comb + lip, 0)
        wt, hk = wet[idx], height[idx]
        base = (hk + level * np.clip(depth / 2, 0, 1) * (flat - hk)) * (1 - 0.6 * wt)
        height[idx] = hk + (base + paste - hk) * a
        wet[idx] = np.maximum(wt, a)
        gl = self.gloss.reshape(-1)
        gl[idx] += (gloss - gl[idx]) * a

    def veil(self, pts, w, colours, g, cover=0.8, catch=0.5, thick=0.3, lands=0.65):
        """Pale paint dragged broad and nearly dry across what is there: a veil. The brush barely touches. Only
        the bunches of its bristles that still hold paint leave any, in streaks of every width at uneven
        spacing along the drag, and they leave it on the ridges of the paint beneath (as much the more
        `catch`) and skip the hollows, so there is bare colour between the streaks and through them. In the
        middle of the drag it lands on about `lands` of the ground and covers it (`cover`); it comes on gently
        and dies away at both ends of the drag and at its sides, in fewer streaks and more thinly."""
        H, W = self.H, self.W
        C = brush.path(np.asarray(pts, np.float64), 1.0)
        sc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        L = sc[-1]
        T = ndimage.gaussian_filter1d(np.gradient(C, axis=0), 4, axis=0, mode="nearest")
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        pad = 1.3 * w + 12
        x0, y0 = np.maximum(np.floor(C.min(0) - pad).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(C.max(0) + pad).astype(int) + 1, (W, H))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
        dist, i = cKDTree(C).query(P, distance_upper_bound=1.25 * w, workers=-1)
        near = np.isfinite(dist)
        P, i = P[near], i[near]
        D = P - C[i]
        s = sc[i] + (D * T[i]).sum(1)
        u = (D * N[i]).sum(1)
        t, v = np.clip(s / L, 0, 1), u / w
        o = g.uniform(0, 4096, 12)
        drag = self.look(self.mid, s * 0.5 + o[0], u * 0.5 + o[1])       # bunches of bristles, dragged long
        load = self.look(self.coarse, s * 0.3 + o[4], u * 0.25 + o[5])   # the uneven load across the brush
        tooth = self.look(self.clump, s * 0.7 + o[2], u * 0.7 + o[3]) + 0.6 * self.look(self.grain, s + o[8], u + o[9])
        rag = self.look(self.mid, s * 0.5 + o[6], np.sign(u) * 300 + o[7])
        # the ridges of the paint beneath, down to the marks of single bristles; the raised edge of a stroke
        # catches no more than a ridge does
        hold = self.height[y0:y1, x0:x1]
        crest = ndimage.gaussian_filter(hold, 0.8) - ndimage.gaussian_filter(hold, 4)
        crest = np.tanh(crest / (np.std(crest) + 0.02)).ravel()[near]
        sides = noise.smoothstep(1.02, 0.5, np.abs(v) + 0.12 * rag)
        ends = noise.smoothstep(0.0, 0.3 + 0.05 * drag, t) * noise.smoothstep(1.0, 0.6 + 0.06 * drag, t)
        fade = sides * ends
        # the bunches of bristles that hold paint, and none between them; within them, the ridges take it and
        # the hollows do not, the highest crests the most; fewer of them toward the ends and the sides
        trk = self.tracks(s, u, g.uniform(0, 4096, 12), w, 1.6)
        z = catch * crest + trk + 0.4 * tooth
        a = cover * (0.5 + 0.5 * fade) * noise.smoothstep(-0.8, 0.2, trk) \
            * (0.55 + 0.45 * noise.smoothstep(-1.0, 1.5, crest + 0.4 * tooth)) \
            * noise.smoothstep(-0.3, 0.3, z / np.std(z) - ndtri(np.clip(1 - lands * fade ** 0.6, 0.02, 1 - 1e-6)))
        k = a > 0.003
        idx = (P[k, 1] * W + P[k, 0]).astype(np.int64)
        a, load = a[k], load[k]
        f = noise.smoothstep(-1.0, 1.0, load)[:, None]
        col = colours[0] * (1 - f) + colours[1] * f
        rgb = self.rgb.reshape(-1, 3)
        rgb[idx] += (col - rgb[idx]) * a[:, None]
        self.height.reshape(-1)[idx] += thick * a
        self.gloss.reshape(-1)[idx] *= 1 - 0.8 * a

    def dot(self, x, y, R, colours, g, thick=7.0, hide=2.0):
        """A fat dot: a round brush loaded and pressed straight down, and lifted with a turn, pulling the
        paste up into a soft peak. One side of the brush held another colour."""
        H, W = self.H, self.W
        x0, y0 = int(max(x - 1.6 * R, 0)), int(max(y - 1.6 * R, 0))
        x1, y1 = int(min(x + 1.6 * R, W)), int(min(y + 1.6 * R, H))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        o = g.uniform(0, 4096, 8)
        e, tilt = g.uniform(0.86, 1.08), g.uniform(0, np.pi)
        c, s_ = np.cos(tilt), np.sin(tilt)
        px, py = (xx - x) * c + (yy - y) * s_, ((yy - y) * c - (xx - x) * s_) / e
        th = np.arctan2(py, px)
        ring = lambda T_, k, j: self.look(T_, o[j] + k * np.sin(th), o[j + 1] + k * np.cos(th))
        spoke = 1024 / (2 * np.pi)           # round the brush once is once across the texture
        rr = np.hypot(px, py)
        edge = R * (1 + 0.06 * ring(self.patch, 30, 0) + 0.012 * ring(self.skip, 40, 2))
        q = rr / edge
        a = np.clip(edge - rr + 0.5, 0, 1)
        # given a turn as it lifted, so the paste is combed round in a loose swirl
        swirl = self.look(self.coarse, rr * 0.3 + o[6], (th + 1.8 * q) * spoke + o[7])
        lx, ly = g.normal(0, 0.25, 2)
        peak = np.exp(-(((px / R - lx) ** 2 + (py / R - ly) ** 2) / 0.08))
        body = (0.8 + 0.15 * np.exp(-((q - 0.85) / 0.1) ** 2) + 0.3 * peak) * np.clip(1 - q * q, 0, 1) ** 0.6
        depth = thick * body * (1 + 0.05 * swirl * q)
        side = np.cos(th - g.uniform(0, 2 * np.pi)) * q
        f = noise.smoothstep(0.1, 0.9, side + 0.2 * noise.field(q.shape, 0.5 * R, g))
        col = colours[0] ** (1 - 0.6 * f[..., None]) * colours[1] ** (0.6 * f[..., None])
        sub = self.rgb[y0:y1, x0:x1]
        sub += (col - sub) * ((1 - np.exp(-hide * depth)) * a)[..., None]
        h = self.height[y0:y1, x0:x1]
        h += (depth - h * 0.3) * a
        self.wet[y0:y1, x0:x1] = np.maximum(self.wet[y0:y1, x0:x1], a)
        self.gloss[y0:y1, x0:x1] += (0.7 - self.gloss[y0:y1, x0:x1]) * a


# ---------------------------------------------------------------- the paints

CAD_RED = pal("#b51d19", "#cb281b", "#db3a1d")
RED_DEEP = pal("#7e1016", "#981a1b", "#b0241c")         # cadmium red deep
CAD_ORANGE = pal("#e8621a", "#f07c20", "#f6982e")
GLOW = pal("#e2501a", "#ef6d1e", "#f58a26", "#f8aa30", "#f9c443")
MADDER = pal("#4a0a1c", "#5e0f24", "#74162c")           # alizarin crimson, deep
AUBERGINE = pal("#2e1236", "#40184a", "#55215e")
NIGHT = pal("#0b2418", "#103222", "#17412c")            # a green so deep it is nearly black
VIRIDIAN = pal("#0d4a36", "#126044", "#197552", "#22875f")
BOTTLE = pal("#07261b", "#0b3525", "#114531", "#185a3e")
ULTRA = pal("#0a0e36", "#111b66", "#1a2b8c", "#2640ab", "#3858be")
RAIN = pal("#97a9c0", "#bcc9d8")


def paint(seed=1988):
    r = noise.rng(seed)
    wood, relief, profile = support(r)
    pan = Surface(wood.copy(), relief.copy(), r)

    def fresh(name):
        """Each passage draws on chances of its own, so the others stay as they are when one is changed."""
        return np.random.default_rng([seed, zlib.crc32(name.encode())])

    def load(fam, i, g):
        """A brush loaded with shade i of a paint, a streak of the next shade, and a trace of one further."""
        j = int(np.clip(i + g.choice([-1, 1]), 0, len(fam) - 1))
        k = int(np.clip(i + g.choice([-2, 2]), 0, len(fam) - 1))
        return np.stack([fam[i], fam[j], fam[k]])

    def passage(name, strokes, **kw):
        """Strokes (points, half-width, paint, shade[, what is different about this one]), one after another.
        A shade of None takes the paint as the brush's whole load."""
        g = fresh(name)
        for pts, w, fam, i, *rest in strokes:
            pan.sweep(pts, w, fam if i is None else load(fam, i, g), g, **{**kw, **(rest[0] if rest else {})})

    # long ago: the panel scrubbed over thinly in cadmium orange and red, the heat before the rains, all but
    # the strip at the left; and the frame in cadmium red, orange and yellow, so thin the grain shows through
    scrub = dict(thick=1.0, hide=1.4, spent=0.3, ends=0.5, dry=0.2, catch=1.2, pickup=0.3, tails=0.8,
                 streaks=1.2, grooves=0.0, lips=0.0, level=0.0, feather=0.3, gloss=0.05, mottle=1.5)
    passage("ground", [
        ([(400, 430), (1300, 400), (2700, 440)], 280, CAD_ORANGE, 1),
        ([(2700, 1000), (1700, 960), (900, 1010), (470, 970)], 330, CAD_RED, 2),
        ([(500, 1550), (1100, 1500), (1800, 1560), (2700, 1520)], 320, CAD_ORANGE, 0),
        ([(430, 2080), (900, 2040), (1550, 2100), (2700, 2060)], 260, CAD_RED, 1),
    ], **scrub)
    passage("frame_under", [
        ([(-60, 115), (1440, 108), (2940, 120)], 120, CAD_RED, 1),
        ([(2765, -60), (2770, 1260), (2760, 2580)], 120, CAD_ORANGE, 2),
        ([(2940, 2405), (1440, 2412), (-60, 2400)], 120, CAD_ORANGE, 0),
        ([(115, 2580), (110, 1260), (120, -60)], 120, CAD_RED, 0),
    ], **scrub)
    pan.sitting()

    # then the depth: alizarin glazed over the window, aubergine over it high on the right where the storm
    # would come, and low down the garden, a green so deep it is nearly black; and the frame a first green
    glz = dict(thick=1.8, hide=1.0, glaze=1.0, spent=0.25, ends=0.45, dry=0.03, catch=0.5, pickup=0.0,
               tails=0.7, streaks=1.0, grooves=0.0, lips=0.0, level=0.0, feather=0.25, gloss=0.25, mottle=1.3,
               vary=0.4)
    passage("madder", [
        ([(480, 560), (1300, 520), (2650, 560)], 320, MADDER, 1),
        ([(2650, 1180), (1700, 1130), (900, 1180), (480, 1150)], 360, MADDER, 0),
        ([(900, 1700), (1700, 1660), (2650, 1700)], 300, MADDER, 1),
        ([(480, 2130), (1500, 2100), (2650, 2140)], 190, MADDER, 2),
    ], **glz)
    passage("aubergine", [
        ([(1200, 640), (1900, 600), (2450, 780), (2580, 1250)], 260, AUBERGINE, 1),
    ], **glz)
    passage("garden", [
        ([(470, 2180), (800, 1990), (1250, 1900), (1700, 1930)], 230, NIGHT, 2),
    ], **{**glz, "hide": 1.1, "thick": 1.8})
    passage("frame_first", [
        ([(-60, 118), (1440, 110), (2940, 122)], 118, VIRIDIAN, 2),
        ([(2765, -60), (2770, 1260), (2760, 2580)], 118, VIRIDIAN, 2),
        ([(2940, 2405), (1440, 2412), (-60, 2400)], 118, VIRIDIAN, 3),
        ([(115, 2580), (110, 1260), (120, -60)], 118, VIRIDIAN, 2),
    ], thick=1.5, hide=1.2, spent=0.3, ends=0.5, dry=0.12, catch=0.8, pickup=0.2, tails=0.8, grooves=0.0,
        lips=0.0, level=0.0, feather=0.3, gloss=0.2, vary=0.5)
    pan.sitting()

    # a frame inside the frame, in cadmium red: the top laid on loaded, stopping short at the right; the
    # left scrubbed thin over the bare mahogany; the foot short at the left, the right side short at the top
    passage("red", [
        ([(240, 370, 0.8), (900, 352, 1.0), (1600, 372, 1.0), (2150, 356, 0.85)], 125, RED_DEEP, 2,
         {"thick": 3.2, "hide": 1.8, "gloss": 0.5}),
        ([(345, 250), (360, 900), (340, 1600), (352, 2270)], 140, RED_DEEP, 2,
         {"thick": 0.6, "hide": 0.8, "dry": 0.25, "catch": 1.0, "gloss": 0.1, "grooves": 0.0, "mottle": 1.5,
          "level": 0.0}),
        ([(2630, 2195), (2100, 2212), (1500, 2188, 0.7)], 95, RED_DEEP, 0, {"dry": 0.12, "catch": 0.8, "vary": 0.6}),
        ([(2548, 2280), (2532, 1700), (2546, 1100, 0.7)], 78, RED_DEEP, 2, {"dry": 0.1}),
    ], thick=2.6, hide=1.3, spent=0.5, ends=0.65, pickup=0.5, tails=0.6, level=0.6, feather=0.15, gloss=0.4,
        grooves=0.04)
    # and the outer frame gone over in a deep bottle green, short of its edges and thin in places, so the first
    # green and the red beneath it come through
    green = np.stack([BOTTLE[2], VIRIDIAN[1], BOTTLE[1]])
    passage("frame", [
        ([(-40, 124), (700, 112), (1500, 128), (2920, 116)], 108, green, None),
        ([(2750, -40, 0.9), (2772, 900), (2754, 1700), (2768, 2560)], 106, green, None),
        ([(2920, 2384), (2000, 2400), (1000, 2382), (-40, 2396, 0.9)], 108, green, None),
        ([(118, 2560), (110, 1500), (124, -40)], 106, green, None),
    ], share=(0.55, 0.3, 0.15), thick=2.2, hide=1.2, spent=0.4, ends=0.6, pickup=0.3, tails=0.6, level=0.5, feather=0.22, gloss=0.3,
        grooves=0.03, vary=0.5)
    pan.sitting()

    # the weather: the storm in one sweep of ultramarine and indigo, landing high on the left, riding over the
    # red and curling down on the right out over the frame; and under it the light, the one hot thing, thick
    # and buttery, wet into wet
    storm = np.stack([ULTRA[2], ULTRA[0], ULTRA[4]])
    passage("storm", [
        ([(380, 700, 0.85), (1000, 520, 1.0), (1800, 520, 1.0), (2350, 760, 0.9), (2560, 1200, 0.75),
          (2500, 1650, 0.55), (2300, 1950, 0.4)], 300, storm, None,
         {"share": (0.45, 0.4, 0.15), "tracks": 1.0, "runs": 0.68, "catch": 0.3}),
    ], thick=2.6, hide=0.8, spent=0.8, ends=0.85, pickup=0.5, tails=0.5, level=0.85, feather=0.2, gloss=0.5,
        grooves=0.04, vary=0.55)
    passage("light", [
        ([(560, 1830, 0.8), (1100, 1610, 1.0), (1700, 1430, 1.0), (2280, 1330, 0.6)], 230,
         np.stack([GLOW[1], GLOW[0], CAD_RED[2]]), None),
    ], thick=1.7, hide=1.1, spent=0.5, ends=0.7, dry=0.12, catch=0.6, pickup=0.0, tails=0.7, level=0.2,
        feather=0.35, gloss=0.2, grooves=0.0, lips=0.0, mottle=1.3, vary=0.45)
    passage("light_core", [
        ([(720, 1700, 0.8), (1200, 1520, 1.0), (1700, 1370, 0.85), (2060, 1310, 0.45)], 115,
         np.stack([GLOW[2], GLOW[4], GLOW[1]]), None, {"share": (0.55, 0.3, 0.15)}),
    ], thick=5.0, hide=1.2, spent=0.6, ends=0.8, pickup=0.6, tails=0.6, start=0.5, level=0.9, feather=0.2,
        gloss=0.8, grooves=0.02, lips=0.0, streaks=0.6, mottle=0.5, vary=0.35)
    pan.sitting()

    # months later the rain, a pale grey-blue dragged down on the slant, broad and nearly dry
    pan.veil([(2330, 330), (2050, 1050), (1750, 1680), (1430, 2280)], 300, RAIN, fresh("rain"))

    # last, fat dots along the top of the frame, from a brush loaded afresh for each
    for i, (x, y, R) in enumerate([(610, 118, 64), (880, 106, 78), (1090, 126, 52), (1660, 114, 70),
                                   (2240, 102, 60)]):
        g = fresh(f"dot{i}")
        pan.dot(x, y, R, np.stack([CAD_RED[g.integers(0, 3)], CAD_ORANGE[g.integers(0, 2)]]), g,
                thick=g.uniform(15, 19))

    height = ndimage.gaussian_filter(pan.height, 0.8)
    img = dabs.shine(pan.rgb, height + profile, light=LIGHT, relief=0.55, gloss=0.0, reach=(0.7, 1.2))
    img = img * shadow(profile)[..., None]
    return img + 0.08 * (impasto.glints(height, LIGHT) * pan.gloss)[..., None]
