"""Lettering cut by hand: a print's title in its cartouche, an engraver's signature.

`python -m atelier.lettering` cuts them again into atelier/lettering/*.png (it needs the Kaiti and
Apple Chancery fonts that ship with macOS). The works only need the PNGs.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import noise

CUT = Path(__file__).parent / "lettering"
KAITI = next(Path("/System/Library/AssetsV2").glob("com_apple_MobileAsset_Font*/*/AssetData/Kaiti.ttc"), None)


def cartouche(title, name, signature=None, size=90, seed=2):
    """A print's title, cut into the key block: vertical text in a double-ruled box,
    the signature running down beside it."""
    r = noise.rng(seed)
    pad = int(size * 0.45)
    th = len(title) * int(size * 1.08) + 2 * pad
    tw = size + 2 * pad
    sw = int(size * 0.85) if signature else 0
    W, H = tw + (sw + pad if signature else 0) + 8, max(th, 0) + 8
    im = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(im)
    big = ImageFont.truetype(str(KAITI), size * 2, index=2)
    for i, ch in enumerate(title):
        d.text((W * 2 - tw * 2 + pad * 2, 8 + pad * 2 + i * int(size * 2.16)), ch, 255, font=big)
    x0, x1 = W * 2 - tw * 2, W * 2 - 4
    d.rectangle([x0, 4, x1, th * 2], outline=255, width=5)
    d.rectangle([x0 + 14, 18, x1 - 14, th * 2 - 14], outline=255, width=3)
    if signature:
        small = ImageFont.truetype(str(KAITI), int(size * 1.5), index=2)
        for i, ch in enumerate(signature):
            d.text((8, 30 + pad * 2 + i * int(size * 1.62)), ch, 255, font=small)
    ink = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255
    # the knife leaves every edge a little rough
    ink = noise.smoothstep(0.35, 0.65, ink + 0.12 * noise.field(ink.shape, 2.0, r))
    CUT.mkdir(exist_ok=True)
    Image.fromarray((ink * 255).astype(np.uint8)).save(CUT / f"{name}.png")


def script(text, name, size=64, font="/System/Library/Fonts/Supplemental/Apple Chancery.ttf"):
    """A signature in a running hand, as it is scratched into a copper plate: a mask in [0, 1]."""
    f = ImageFont.truetype(font, size * 3)
    x0, y0, x1, y1 = f.getbbox(text)
    im = Image.new("L", (x1 - x0 + 24, y1 - y0 + 24), 0)
    ImageDraw.Draw(im).text((12 - x0, 12 - y0), text, fill=255, font=f)
    im = im.resize((im.width // 3, im.height // 3), Image.LANCZOS)
    im.save(CUT / f"{name}.png")


if __name__ == "__main__":
    cartouche("赤富士晴朝", "fuji_title", signature="克勞德筆")
    script("Claude f. 2026", "etched_signature", size=34)
    print("cut", sorted(p.name for p in CUT.glob("*.png")))
