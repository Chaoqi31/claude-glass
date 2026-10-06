"""Colour: sRGB <-> linear light, and pigments as absorbance (Beer-Lambert).

Transparent media (ink, watercolour, dye) never "cover" what is beneath them.
They absorb light, so layers multiply: reflectance = ground * exp(-sum(k * d)).
"""

import numpy as np


def lin(c):
    """'#rrggbb' or 0-255 triple -> linear RGB float32."""
    if isinstance(c, str):
        c = tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    s = np.asarray(c, np.float32) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def pigment(c):
    """Absorbance per channel of a pigment that shows colour `c` at unit density."""
    return -np.log(np.clip(lin(c), 1e-4, 1.0))


def glaze(img, density, k):
    """Lay a transparent layer of density map `density` (H,W) with absorbance `k` (3,)."""
    return img * np.exp(-density[..., None] * k)


def saturate(density, cap):
    """Ink can only get so dark: soft-clip a density map at `cap`."""
    return cap * (1 - np.exp(-density / cap))


def cover(img, alpha, c):
    """Lay an opaque layer (gouache, gold, glaze on tile) of colour `c` with coverage `alpha`."""
    col = c if isinstance(c, np.ndarray) else lin(c)
    a = alpha[..., None]
    return img * (1 - a) + col * a


# Pigments that recur across rooms.
SUMI = pigment("#5a5a5f")          # pine-soot ink: cool at wash strength
SUMI_WARM = pigment("#615a55")     # oil-soot ink: warmer, browner
VERMILION = pigment("#e0452c")     # cinnabar seal paste
