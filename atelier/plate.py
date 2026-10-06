"""The plate: how a finished work is laid on the backdrop and photographed for the catalogue."""

import numpy as np
from PIL import Image
from scipy import ndimage

from .color import lin, to_srgb

BACKDROP = "#242321"


def to_srgb_255(c):
    """'#rrggbb' as an sRGB 0-255 triple (for things drawn outside linear light)."""
    return tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def mount(img, sheet, backdrop=BACKDROP, shadow=0.45, lift=(9, 6)):
    """Place the painted sheet on the backdrop with the faint shadow of a sheet lying flat."""
    a = sheet.alpha
    cast = ndimage.gaussian_filter(np.roll(a, lift, (0, 1)), 16) * shadow
    bg = lin(backdrop)[None, None, :] * (1 - cast[..., None])
    return bg * (1 - a[..., None]) + img * a[..., None]


def save(img, path, quality=93, width=None, seed=0):
    """Linear RGB -> dithered 8-bit sRGB JPEG (4:4:4, so vermilion stays crisp)."""
    s = to_srgb(img) * 255.0
    s += np.random.default_rng(seed).triangular(-0.5, 0, 0.5, s.shape)
    im = Image.fromarray(np.clip(s + 0.5, 0, 255).astype(np.uint8))
    if width and width < im.width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    if str(path).endswith(".png"):
        im.save(path, optimize=True)
    else:
        im.save(path, quality=quality, subsampling=0, optimize=True, progressive=True)
    return im
