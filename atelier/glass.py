"""Stained glass: blown glass cut into pieces, painted, leaded, and lit from behind.

A glazier works from a cartoon, a full-size drawing that gives every piece its colour and
its outline. The pieces are cut from blown sheets, and no sheet is even. Thickness is
colour: a thick blue is nearly black and a thin one is nearly sky, and a red that was only
flashed onto clear glass goes pink in streaks where the flash ran thin. The painter traces
feathers and leaves in iron-oxide paint and fires it on. Eight centuries later the paint
has flaked, the outside has pitted and crusted, a few cracked pieces have been mended with
another lead or replaced with whatever glass was at hand, and the light coming through is
the same grey north light.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import noise
from .color import lin, pigment

PAINT = pigment("#3b2a20")     # fired iron-oxide paint: brown-black
CRUST = pigment("#8c8374")     # weathering on the outside face
LEAD = lin("#0d0d0f")


@dataclass
class Glass:
    """One kind of pot-metal glass, as it looks at its usual thickness against a grey sky."""
    colours: tuple              # a few batches; every piece is some blend of them
    thick: float = 0.18         # spread of thickness from piece to piece (log)
    mottle: float = 0.10        # unevenness within a piece
    streak: float = 0.04        # reams, drawn out as the sheet was blown
    seeds: float = 1.0          # bubbles per 10,000 px
    weather: float = 0.12       # how badly the outside has crusted and pitted


def boundary(lab):
    """Pixels whose left or upper neighbour belongs to another piece."""
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return b


def _fill(lab, glass):
    idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    return np.where(glass, lab[idx[0], idx[1]], 0)


def pieces(keys, least=14):
    """Every connected run of one key is one piece; slivers go to a neighbour. -> labels, count"""
    glass = keys > 0
    lab, n = ndimage.label(glass & ~boundary(keys))
    area = np.bincount(lab.ravel(), minlength=n + 1)
    lab[(area < least)[lab]] = 0
    lab = _fill(lab, glass)
    used = np.unique(lab)
    remap = np.zeros(lab.max() + 1, np.int64)
    remap[used] = np.arange(used.size)
    return remap[lab], used.size - 1


def _split(sub, m, new, r, wobble):
    """One cut across piece `m` (a mask in `sub`), a little off the middle, never quite straight."""
    ys, xs = np.nonzero(m)
    ev, evec = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(np.float64)))
    a = np.arctan2(evec[1, 1], evec[0, 1]) + r.normal(0, 0.25)
    nx, ny = np.cos(a), np.sin(a)
    off = r.normal(0, 0.15) * np.sqrt(ev[1])
    Y, X = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    px, py = X - xs.mean() - nx * off, Y - ys.mean() - ny * off
    along = py * nx - px * ny
    bend = wobble * np.sin(along / r.uniform(12, 30) + r.uniform(0, 6.3)) + 0.3 * wobble * np.sin(along / 4.1)
    sub[m & (px * nx + py * ny + bend > 0)] = new


def cut(lab, n, most, r, wobble=1.5):
    """The glazier's cuts: a sheet only goes so far, so any piece bigger than `most` px is cut
    across its length."""
    for _ in range(12):
        area = np.bincount(lab.ravel(), minlength=n + 1)
        area[0] = 0
        if area.max() <= most:
            break
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is not None and area[i] > most:
                n += 1
                _split(lab[sl], lab[sl] == i, n, r, wobble)
        lab, n = pieces(lab)
    return lab, n


def mend(lab, n, share, r):
    """Old cracks, mended with a strap of lead across the piece."""
    area = np.bincount(lab.ravel(), minlength=n + 1)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None and area[i] > 900 and r.random() < share:
            n += 1
            _split(lab[sl], lab[sl] == i, n, r, wobble=4.0)
    return pieces(lab)


def came(lab, width, r):
    """Lead between the pieces and round each panel, with a dab of solder at every joint.
    -> coverage, distance of each pixel from the nearest seam"""
    dist = ndimage.distance_transform_edt(~boundary(lab))
    hw = 0.5 * width * (1 + 0.12 * noise.field(lab.shape, 40, r))
    lead = noise.smoothstep(hw + 0.6, hw - 0.6, dist)
    a, b, c, d = lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]
    many = 1 + (b != a) + ((c != a) & (c != b)) + ((d != a) & (d != b) & (d != c))
    joint = np.zeros(lab.shape, bool)
    joint[:-1, :-1] = many >= 3
    dj = ndimage.distance_transform_edt(~joint)
    rs = width * (0.8 + 0.2 * noise.field(lab.shape, 5, r))
    return np.maximum(lead, noise.smoothstep(rs + 0.6, rs - 0.6, dj)), dist


def _per_piece(lab, n, values):
    return np.bincount(lab.ravel(), values.ravel(), minlength=n + 1)


def window(keys, kind, glasses, paint, sky, r, width=2.6, most=5000, stopgaps=0.008, mended=0.012):
    """Cut the cartoon into glass, paint it, lead it and hold it up to the sky.

    keys     (H,W) int: the cartoon's pieces, 0 where there is no glass
    kind     (H,W) int: which of `glasses` each pixel is
    paint    (H,W) float: paint density as the painter laid it (trace ~2-3, mat ~0.1-0.4)
    sky      (H,W) float: the light behind the window
    -> linear RGB (H,W,3), lead coverage (H,W), piece labels (H,W)
    """
    h, w = keys.shape
    lab, n = pieces(keys)
    lab, n = cut(lab, n, most, r)
    lab, n = mend(lab, n, mended, r)
    K = int(kind.max()) + 1                  # each piece is the glass most of its pixels asked for
    votes = np.bincount((lab * K + kind).ravel(), minlength=(n + 1) * K).reshape(n + 1, K)
    votes[:, 0] = 0
    cls = votes.argmax(1)
    # stopgaps: a broken piece replaced by whatever glass the mender had to hand
    size = np.bincount(lab.ravel(), minlength=n + 1)
    swap = (r.random(n + 1) < stopgaps) & (cls > 0) & (size < 900)
    used = np.unique(cls[cls > 0])
    cls[swap] = r.choice(used, swap.sum())
    cls[0] = 0

    # every piece its own batch, thickness, grain direction and history
    thick, mottle, streak, seeds, weather = (np.array([0.0] + [getattr(g, a) for g in glasses]) for a in
                                             ("thick", "mottle", "streak", "seeds", "weather"))
    t0 = np.exp(r.normal(0, 1, n + 1) * thick[cls])
    k = np.zeros((n + 1, 3), np.float32)
    for c, g in enumerate(glasses, 1):
        sel = np.nonzero(cls == c)[0]
        ks = np.stack([pigment(x) for x in g.colours])
        wts = r.dirichlet(np.ones(len(ks)) * 0.7, sel.size)
        k[sel] = wts @ ks
    k *= np.exp(r.normal(0, 0.04, (n + 1, 3)))
    ang = r.uniform(0, np.pi, n + 1)
    ox, oy = r.uniform(400, 1600, (2, n + 1))
    wx = np.clip(r.gamma(1.6, 1.0, n + 1) / 1.6 * weather[cls], 0, 1.2)
    wx = np.where((r.random(n + 1) < 0.012) & (size < 1500), 1.6 + r.random(n + 1), wx)   # a few gone nearly opaque

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    area = np.maximum(np.bincount(lab.ravel(), minlength=n + 1), 1)
    cx, cy = _per_piece(lab, n, xx) / area, _per_piece(lab, n, yy) / area
    g = lab > 0
    L = lab[g]
    dx, dy = xx[g] - cx[L], yy[g] - cy[L]
    ca, sa = np.cos(ang[L]), np.sin(ang[L])
    U, V = dx * ca + dy * sa + ox[L], -dx * sa + dy * ca + oy[L]
    tex = noise.fbm((2048, 2048), 26, r, octaves=5)
    ream = noise.warp(noise.stretched((2048, 2048), 4, 200, r), 3 * noise.field((2048, 2048), 90, r), 0.0)
    m = ndimage.map_coordinates(tex, [V, U], order=1, mode="mirror")
    s = ndimage.map_coordinates(ream, [V, U], order=1, mode="mirror")
    d = np.ones((h, w), np.float32)
    d[g] = t0[L] * np.exp(mottle[cls[L]] * m + streak[cls[L]] * s)
    d *= np.exp(0.05 * noise.field((h, w), 2.2, r))           # the rippled skin of blown glass
    # seeds: small bubbles drawn out along the grain, where there is less glass
    bub = np.zeros((h, w), np.float32)
    p = seeds[cls[L]] * 1e-4
    hit = r.random(L.size) < p
    for t in np.linspace(-2.5, 2.5, 6):
        bx = np.clip((xx[g][hit] + t * ca[hit]).round().astype(int), 0, w - 1)
        by = np.clip((yy[g][hit] + t * sa[hit]).round().astype(int), 0, h - 1)
        np.add.at(bub, (by, bx), 1.0)
    bub = ndimage.gaussian_filter(bub, 0.6)
    d *= 1 - 0.7 * np.clip(bub / (bub.max() + 1e-6) * 3, 0, 1)

    img = np.exp(-k[lab] * d[..., None]) * sky[..., None]

    # the painter's work, fired on: each piece was painted on its own, so the lines jump a
    # little at every lead, and centuries have lifted the paint in flakes
    pdx, pdy = r.normal(0, 0.45, (2, n + 1))
    P = ndimage.map_coordinates(paint, [yy + pdy[lab], xx + pdx[lab]], order=1, mode="nearest")
    loss = r.beta(0.7, 3.0, n + 1)
    lift = noise.smoothstep(0.0, 0.5, noise.fbm((h, w), 14, r, octaves=3) + 1.6 * loss[lab] - 1.5)
    P *= 1 - 0.85 * lift
    # grime and cement in the angle of every lead, pitting where the glass has weathered
    lead, seam = came(lab, width, r)
    P += 0.15 * np.exp(-np.maximum(seam - width * 0.5, 0) / 1.3)
    pits = (r.random((h, w)) < 0.004 * wx[lab]).astype(np.float32)
    P += ndimage.gaussian_filter(pits, 0.6) * 4.0
    crust = wx[lab] * (0.4 + 1.1 * noise.smoothstep(0.0, 1.5, noise.fbm((h, w), 25, r, octaves=4)))
    crust *= 0.75 + 0.5 * noise.smoothstep(-1.0, 1.5, noise.stretched((h, w), 7, 220, r))  # rain runs down the outside
    img *= np.exp(-P[..., None] * PAINT - crust[..., None] * CRUST)
    img = img * (1 - lead[..., None]) + LEAD * lead[..., None]
    img[~g] = 0
    return img, lead * g, lab


def halation(img, strength=(0.26, 0.34, 0.46), spread=(1.5, 2.2, 3.2), veil=0.06):
    """Light from bright glass spills over the dark lead beside it, and blue spills furthest,
    which is why the leads of a blue window seem to melt away; a little of it hangs in the
    air of the church as a veil."""
    out = np.empty_like(img)
    for c in range(3):
        near = ndimage.gaussian_filter(img[..., c], spread[c])
        far = ndimage.gaussian_filter(img[..., c], spread[c] * 5)
        out[..., c] = img[..., c] * (1 - strength[c]) + near * strength[c] + far * veil * (1 + c * 0.5)
    return out
