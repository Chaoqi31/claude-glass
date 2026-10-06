"""Dabs: touches of unmixed oil colour set side by side, for the eye to mix.

Seurat never blended on the palette. He laid small separate touches of nearly pure colour,
a pink beside a green beside a cream, and left the eye across the room to add them up,
because light mixed that way stays brighter than paint mixed in a pot. Each touch is a
small heap of paint with its own shine, and between them the primed canvas shows.
"""

import numpy as np

from . import noise
from .color import lin

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def palette(hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


def scatter(shape, spacing, r, jitter=0.42):
    """Points about `spacing` apart and never in rows: a honeycomb shaken hard. -> (N, 2) x, y"""
    h, w = shape
    gy, gx = np.mgrid[0:h + spacing:spacing * 0.866, 0:w + spacing:spacing].astype(np.float32)
    gx += (np.arange(gy.shape[0]) % 2)[:, None] * spacing * 0.5
    pts = np.stack([gx.ravel(), gy.ravel()], 1) + r.uniform(-jitter, jitter, (gx.size, 2)) * spacing
    keep = (pts[:, 0] > -spacing) & (pts[:, 0] < w + spacing) & (pts[:, 1] > -spacing) & (pts[:, 1] < h + spacing)
    pts = pts[keep]
    return pts[r.permutation(len(pts))]


def tint(target, hues, r, white="#faf6ec", dark="#1d2050", vivid=0.05, chance=0.02, spread=0.2):
    """A colour for each dab, the way Seurat chose them. Every hue on his palette is let down
    with white (or, below its own value, with deep blue) to a lightness near that of the
    place: some dabs lighter and paler, some darker and stronger, the same on average. Of the
    pairs of hues whose mixture has the colour wanted, take one far apart on the wheel, then
    one of the pair by the right odds: the dabs average out to `target` (N,3)."""
    Wt, Dk = lin(white), lin(dark)
    yw, yd, yh = Wt @ LUMA, Dk @ LUMA, hues @ LUMA
    i, j = np.triu_indices(len(hues), 1)
    out = np.empty_like(target)
    for s in range(0, len(target), 20000):
        T = target[s:s + 20000]
        n = np.arange(len(T))
        yt = T @ LUMA
        y = np.clip(yt * np.exp(r.normal(0, spread, len(T)) - spread ** 2 / 2), yd, yw).astype(np.float32)[:, None]
        up = y >= yh[None]
        f = np.where(up, (y - yh) / (yw - yh), (yh - y) / (yh - yd)).clip(0, 1)[..., None]
        Q = hues[None] * (1 - f) + np.where(up[..., None], Wt, Dk) * f          # (n, K, 3)
        C = Q - (Q @ LUMA)[..., None]
        want = T - yt[:, None]
        A, D = C[:, i], C[:, j] - C[:, i]
        g = np.clip(((want[:, None] - A) * D).sum(-1) / ((D * D).sum(-1) + 1e-9), 0, 1)
        miss = np.linalg.norm(A + g[..., None] * D - want[:, None], axis=-1)
        score = miss - vivid * np.linalg.norm(D, axis=-1) + chance * r.gumbel(size=miss.shape)
        k = score.argmin(1)
        out[s:s + 20000] = Q[n, np.where(r.random(len(T)) < g[n, k], j[k], i[k])]
    return out


def lay(rgb, height, pts, colours, r, size=(5.0, 3.0), angle=0.0, spread=0.3, thick=1.0, square=2.6):
    """Press one layer of dabs into the paint: each a little oblong heap, squarish at the ends,
    turned its own way, its edge torn by the bristles and ridged where the brush lifted off.
    Works in place on rgb (H,W,3) and height (H,W)."""
    h, w = height.shape
    n = len(pts)
    a = size[0] * np.exp(np.clip(r.normal(0, 0.16, n), -0.35, 0.35)).astype(np.float32)
    b = size[1] * np.exp(np.clip(r.normal(0, 0.16, n), -0.35, 0.35)).astype(np.float32)
    th = (np.broadcast_to(np.asarray(angle, np.float32), (n,)) + r.normal(0, spread, n)).astype(np.float32)
    ph = r.uniform(0, 2 * np.pi, n).astype(np.float32)
    torn = 0.09 * noise.field((h, w), 1.6, r).reshape(-1)
    R = int(np.ceil(a.max())) + 1
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1].reshape(2, -1).astype(np.float32)
    flat_rgb = rgb.reshape(-1, 3)
    flat_h = height.reshape(-1)
    for s in range(0, n, 15000):
        P, sl = pts[s:s + 15000], slice(s, s + 15000)
        X = np.floor(P[:, :1]) + ox[None]
        Y = np.floor(P[:, 1:]) + oy[None]
        dx, dy = X - P[:, :1], Y - P[:, 1:]
        c, sn = np.cos(th[sl])[:, None], np.sin(th[sl])[:, None]
        u = (dx * c + dy * sn) / a[sl, None]
        v = (-dx * sn + dy * c) / b[sl, None]
        ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
        idx = np.where(ok, Y * w + X, 0).astype(np.int64)
        rho = (np.abs(u) ** square + np.abs(v) ** square) ** (1 / square) + torn[idx]
        cov = noise.smoothstep(1.0, 0.86, rho)
        ok &= cov > 0.01
        idx, cv = idx[ok], cov[ok]
        bristle = 1 + 0.035 * np.sin(v * 7.0 + ph[sl, None])
        col = colours[sl, None, :] * bristle[..., None]
        heap = thick * np.clip(1 - rho * rho, 0, 1) ** 0.5 * (1 + 0.4 * np.clip(u, -1, 1)) * (0.9 + 0.1 * bristle)
        flat_rgb[idx] = flat_rgb[idx] * (1 - cv[:, None]) + col[ok] * cv[:, None]
        flat_h[idx] = np.maximum(flat_h[idx] * (1 - 0.6 * cv), heap[ok] + 0.3 * flat_h[idx])


def shine(rgb, height, light=(-0.55, -0.65, 0.52), relief=0.35, gloss=0.03, reach=(0.86, 1.14)):
    """Soft gallery light raking the heaps of paint: a lit side, a shadowed side, a glint on the crests."""
    gy, gx = np.gradient(height * relief)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.asarray(light, np.float32) / np.linalg.norm(light)
    lam = (n @ L) / L[2]
    H = L + np.array([0, 0, 1], np.float32)
    H /= np.linalg.norm(H)
    spec = np.clip(n @ H, 0, 1) ** 40
    return rgb * np.clip(lam, *reach)[..., None] + gloss * spec[..., None]
