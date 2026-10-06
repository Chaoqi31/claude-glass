"""The brush: a few hundred bristles, each carrying its own ink, dragged across a sheet.

Nothing here draws a shape. A stroke is a path and a pressure; the mark is whatever
the bristles leave behind. Pressure spreads them, the paper's tooth catches them, and
as each bristle runs dry it starts to skip, which is where flying white comes from.
"""

import numpy as np
from scipy import ndimage

from . import noise


def _catmull(P):
    P = np.asarray(P, np.float64)
    if len(P) == 2:
        t = np.linspace(0, 1, max(2, int(np.hypot(*(P[1, :2] - P[0, :2])) / 0.4)))[:, None]
        return P[0] * (1 - t) + P[1] * t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = Q[i:i + 4]
        # fine enough that the polyline never shows its corners at the edge of a wide brush
        n = max(8, int(np.hypot(*(p2[:2] - p1[:2])) / 0.4))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        segs.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                           + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    segs.append(P[-1:])
    return np.vstack(segs)


def path(P, step=0.6):
    """Smooth control rows (x, y, attrs...) and resample every `step` px of arc length."""
    C = _catmull(P)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C[:, :2], axis=0).T))])
    si = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(si, s, C[:, j]) for j in range(C.shape[1])], 1)


def splat(X, Y, V, pad=3):
    """Bilinear accumulation of V at float positions. Returns (buffer, (y0, x0))."""
    x0, y0 = int(np.floor(X.min())) - pad, int(np.floor(Y.min())) - pad
    w, h = int(np.ceil(X.max())) - x0 + pad, int(np.ceil(Y.max())) - y0 + pad
    X, Y = X - x0, Y - y0
    xi, yi = np.floor(X).astype(np.int64), np.floor(Y).astype(np.int64)
    fx, fy = X - xi, Y - yi
    buf = np.zeros(h * w)
    for dy, dx, k in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                      (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        buf += np.bincount(((yi + dy) * w + xi + dx).ravel(), (V * k).ravel(), minlength=h * w)
    return buf.reshape(h, w).astype(np.float32), (y0, x0)


def paste(dst, buf, off, op=np.add):
    y0, x0 = off
    h, w = buf.shape
    H, W = dst.shape[:2]
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + h), min(W, x0 + w)
    if ye <= ys or xe <= xs:
        return dst
    region = dst[ys:ye, xs:xe]
    dst[ys:ye, xs:xe] = op(region, buf[ys - y0:ye - y0, xs - x0:xe - x0])
    return dst


def _sample(img, X, Y):
    h, w = img.shape
    return img[np.clip(Y.astype(np.int64), 0, h - 1), np.clip(X.astype(np.int64), 0, w - 1)]


def stroke(sheet, P, radius, seed=0, bristles=None, load=1.0, reach=3000.0, tip=0.9,
           clumps=9, splay=0.12, tone=0.0, dryness=1.0, head=0.8, strength=1.0, step=0.6, into=None):
    """Drag a loaded brush along control rows P = [(x, y, pressure), ...].

    radius   half-width of the mark at full pressure, px
    load     ink carried at the start (1 = freshly dipped)
    reach    px of full-pressure travel before a bristle's ink falls to 1/e
    tip      contact width = radius * pressure**tip
    clumps   as the brush dries its bristles part into this many bunches: flying white
    splay    how far a dry bunch wanders from its place, as a fraction of the width
    tone     lateral ink gradient; > 0 darkens the left side of travel
    dryness  how readily a thirsty bristle skips over the valleys of the paper
    head     roundness of the mark where the brush first touches down (0 = square cut)
    strength ink concentration
    Returns (ink, water): pigment density and the water that came with it, sheet-sized;
    pass `into=(ink, water)` to add to existing maps instead.
    """
    r = noise.rng(seed)
    C = path(P, step)
    x, y, p = C[:, 0], C[:, 1], np.clip(C[:, 2], 0.0, 1.0)
    S = len(x)
    s = np.arange(S) * step
    tx = ndimage.gaussian_filter1d(np.gradient(x), 3)
    ty = ndimage.gaussian_filter1d(np.gradient(y), 3)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    width = radius * np.maximum(p, 1e-3) ** tip

    B = bristles or int(max(160, 7 * radius))
    u = np.clip(np.linspace(-1, 1, B) + r.uniform(-1, 1, B) / B, -1, 1).astype(np.float32)
    # bunches of uneven size: a brush never splits into equal parts
    edges = np.concatenate([[-1], np.sort(r.uniform(-1, 1, clumps - 1)), [1]])
    edges = 0.5 * edges + 0.5 * np.linspace(-1, 1, clumps + 1)
    cid = np.clip(np.searchsorted(edges, u) - 1, 0, clumps - 1)
    centre = (0.5 * (edges[cid] + edges[cid + 1])).astype(np.float32)
    along = max(6.0, 40.0 / step)
    cw = noise.line1d(S, along * 4, r, rows=clumps)
    bw = noise.line1d(S, along, r, rows=B)
    skip_c = noise.line1d(S, 200 / step, r, rows=clumps)   # a bunch lifts off the paper for a while
    skip_b = noise.line1d(S, 50 / step, r, rows=B)        # and each bristle stutters on its own
    # ink strength varies across the brush in soft bands, which is what a wet stroke shows
    band = ndimage.gaussian_filter1d(r.normal(0, 1, B), B / 120) * np.sqrt(B / 40)
    conc = strength * (1.0 + tone * -u) * (1 + 0.09 * band) * r.uniform(0.85, 1.15, clumps)[cid]
    # edge bristles and some bunches hold less ink, so they give out first
    load0 = load * r.uniform(0.7, 1.25, B) * r.uniform(0.7, 1.2, clumps)[cid] * (1 - 0.3 * u * u)
    # a pressed brush lands as a round footprint, so edge bristles meet the paper late
    land = head * width[0] * (1 - np.sqrt(np.clip(1 - u * u, 0, 1)))

    ink, water = into if into is not None else (np.zeros(sheet.alpha.shape, np.float32),
                                                np.zeros(sheet.alpha.shape, np.float32))
    norm = 2.0 * width * step / B
    rim = noise.smoothstep(1.0, 0.95, np.abs(u))
    for c0 in range(0, B, 96):
        sl = slice(c0, min(B, c0 + 96))
        landed = noise.smoothstep(land[sl, None], land[sl, None] + 3.0, s[None, :])
        contact = rim[sl, None] * landed
        used = np.cumsum(landed * (width / radius)[None, :] * p[None, :] ** 0.5, axis=1) * step / reach
        left = load0[sl, None] * np.exp(-used)
        thirst = 1 - noise.smoothstep(0.02, 0.4, left * dryness ** -0.5 - 0.1 * (1 - p[None, :]))
        # wet bristles cling together; as they dry they bunch into streaks with paper between
        uu = u[sl, None] + (centre[sl, None] - u[sl, None]) * 0.85 * thirst
        wob = splay * thirst * (0.85 * cw[cid[sl]] + 0.15 * bw[sl])
        off = (uu + wob) * width[None, :]
        X = x[None, :] + nx[None, :] * off
        Y = y[None, :] + ny[None, :] * off
        # a thirsty bristle touches in runs; the paper's tooth decides where each run breaks
        touch = 0.5 + 0.16 * skip_c[cid[sl]] + 0.08 * skip_b[sl] + 0.25 * (_sample(sheet.tooth, X, Y) - 0.5)
        thr = 0.95 * thirst - 0.1
        catch = noise.smoothstep(thr - 0.04, thr + 0.04, touch)
        V = contact * catch * conc[sl, None] * norm[None, :]
        blur = max(0.55, 0.45 * 2 * radius / B)
        buf, o = splat(X, Y, V.astype(np.float32))
        paste(ink, ndimage.gaussian_filter(buf, blur), o)
        buf, o = splat(X, Y, (V * (1 - thirst)).astype(np.float32))
        paste(water, ndimage.gaussian_filter(buf, blur), o)
    return ink, water
