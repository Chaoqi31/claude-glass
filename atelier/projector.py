"""The projector: how one frame of film reaches the screen of a dark room.

Each frame is pulled down into the gate and held still while the shutter opens. The gate's aperture, a window
cut in a steel plate with its corners rounded, masks the frame a little inside its edges; it lies just off the
plane the lens is focused on, so its edge comes out soft. The lamp is a tungsten filament behind a condenser,
peach beside daylight where the film is clear. An old projector throws a hot middle a little off centre and
lets the light fall away steeply from it, dimmest in the corners and darker again along the edges of the gate,
turning warmer as it dims until the corners are burnt brown. No two frames sit in the gate in quite the same
place (the weave) and no two are lit quite as brightly (the flicker). The lens is never quite sharp, softer
toward the corners, where it parts the colours a hair, and some of the light scatters inside it: a halo round
everything bright and a faint veil over the whole picture and round it.
"""

import numpy as np
from scipy import ndimage

from . import noise
from .color import lin
from .noise import smoothstep

LAMP = lin("#fad8b2")    # the screen where the film is clear, to an eye settled into the dark


def aperture(h, w, pad, corner=0.08):
    """Signed distance in px from the edge of the aperture, an (h, w) window with its corners rounded by
    `corner` of its height, over a screen (h + 2 pad, w + 2 pad): negative inside."""
    rc = corner * h
    y = np.abs(np.arange(h + 2 * pad) + 0.5 - (h / 2 + pad))[:, None] - (h / 2 - rc)
    x = np.abs(np.arange(w + 2 * pad) + 0.5 - (w / 2 + pad))[None, :] - (w / 2 - rc)
    return (np.hypot(np.maximum(y, 0), np.maximum(x, 0)) + np.minimum(np.maximum(y, x), 0) - rc).astype(np.float32)


def project(frame, r, margin, pad=48, lamp=LAMP, soft=5.0, falloff=1.9, burn=(0.5, 90.0), weave=0.7,
            flicker=0.025, halo=(0.2, 14.0), veil=0.03):
    """Light on the screen from one frame. `frame` is the fraction of light the film lets through, with
    `margin` px of film to spare round the aperture; returns (h + 2 pad, w + 2 pad, 3), linear. The gate's
    edge is `soft` px wide. The light falls to about e^-`falloff` of the hot middle in the corners, and within
    about burn[1] px of the gate's edge by as much as burn[0] again. Of the scattered light a share halo[0]
    spreads about halo[1] px round everything, and `veil` lies evenly over the whole picture, lifting its
    blacks."""
    H, W, _ = frame.shape
    h, w = H - 2 * margin, W - 2 * margin
    yy, xx = np.mgrid[0:h + 2 * pad, 0:w + 2 * pad].astype(np.float32)
    yy, xx = yy + 0.5 - h / 2 - pad, xx + 0.5 - w / 2 - pad
    diag = (h / 2) ** 2 + (w / 2) ** 2
    dy, dx = r.normal(0, 0.6 * weave), r.normal(0, weave)
    img = np.stack([ndimage.map_coordinates(frame[..., c], [yy * k + H / 2 - 0.5 + dy, xx * k + W / 2 - 0.5 + dx],
                                            order=1, mode="nearest") for c, k in enumerate((1.0008, 1.0, 0.9992))], -1)
    sharp, blur = ndimage.gaussian_filter(img, (1.0, 1.0, 0)), ndimage.gaussian_filter(img, (2.2, 2.2, 0))
    img = sharp + (blur - sharp) * np.clip((yy ** 2 + xx ** 2) / diag, 0, 1)[..., None]
    d = aperture(h, w, pad)
    ap = smoothstep(soft, -soft, d)
    hot = ((yy + 0.03 * h) ** 2 + (xx - 0.04 * w) ** 2) / diag
    fall = (np.exp(-falloff * hot) * (1 - burn[0] * np.exp(np.minimum(d, 0) / burn[1]))
            * (1 + 0.03 * noise.field(hot.shape, w / 3, noise.rng(16))))
    out = img * ap[..., None] * lamp * fall[..., None] ** np.array([0.7, 1.0, 1.45], np.float32)
    out *= 1 + flicker * r.standard_normal()
    glow = ndimage.gaussian_filter(out, (halo[1], halo[1], 0), truncate=2.5)
    spread = ndimage.gaussian_filter(ap, 2 * halo[1], truncate=2.5)[..., None] * (out.mean((0, 1)) * ap.size / ap.sum())
    return (1 - halo[0]) * out + halo[0] * glow + veil * spread
