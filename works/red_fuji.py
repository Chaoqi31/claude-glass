"""Red Fuji, Clear Morning. Colour woodblock print.

In a few mornings of late summer, with a south wind and clear air, the sun comes up and
turns the mountain red. Hokusai printed it around 1831 from a handful of cherry blocks:
blue wiped darker toward the top of the sky, the mountain in iron red wiped brown at its
crown, snow and clouds left as bare paper. This print is cut from six blocks, printed
one over another, and the wood shows in every flat colour.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from atelier import lettering, noise, paper, plate, relief
from atelier.color import glaze, pigment

TITLE = "Red Fuji, Clear Morning"
DATE = "2026"
MEDIUM = "Colour woodblock print (moku-hanga) from six blocks, on kozo paper"
AFTER = "Katsushika Hokusai, Fine Wind, Clear Morning, from Thirty-six Views of Mount Fuji, Edo, c. 1831"
ROOM = "The Workshop"
YEAR = 1831
PLACE = "Edo"
REGION = "East Asia"
NOTE = ("Six blocks, printed in turn. The blue was wiped by hand before each pull, so no two "
        "skies would ever come out the same.")

PRUSSIAN = pigment("#3f628f")
BENGARA = pigment("#cf6446")
UMBER = pigment("#7a4230")
GREEN = pigment("#6f9a6c")
PINE = pigment("#3a5a45")
SUMI = pigment("#4a4643")


def clouds(shape, top, r):
    """Rows of mackerel cloud: long, low, ragged along the edges, left as bare paper."""
    h, w = shape
    m = np.zeros(shape, np.float32)

    def blob(cx, cy, ax, ay):
        x0, x1 = max(0, int(cx - ax - 2)), min(w, int(cx + ax + 3))
        y0, y1 = max(0, int(cy - ay - 2)), min(h, int(cy + ay + 3))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        b = noise.smoothstep(1.0, 0.8, np.hypot((xx - cx) / ax, (yy - cy) / ay))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], b * (yy < base + ay * 0.1))

    y = h * 0.1
    while y < h * 0.52:
        x = r.uniform(-200, 50)
        while x < w + 100:
            L = r.uniform(90, 330) * (0.8 + y / h)
            t = r.uniform(12, 26) * (0.7 + y / h)
            base = y + r.normal(0, 6) + t * 0.45
            # a cloud: a flat-bottomed body with two or three puffs on top
            blob(x + L / 2, base - t * 0.45, L / 2, t)
            for _ in range(int(r.integers(1, 4))):
                blob(x + r.uniform(0.15, 0.85) * L, base - t * 0.95, L * r.uniform(0.15, 0.3), t * 0.9)
            x += L + r.uniform(20, 90)
        y += r.uniform(38, 62)
    near = noise.smoothstep(0.0, 0.08, ndimage.gaussian_filter(m, 4))   # the knife only works near an edge
    ragged = m + near * (0.3 * noise.field(shape, 5, r) + 0.15 * noise.field(shape, 1.5, r))
    yy = np.arange(h, dtype=np.float32)[:, None]
    return noise.smoothstep(0.45, 0.55, ragged) * (yy < top[None, :] - 6)


def paint(seed=31):
    H, W = 1600, 2400
    r = noise.rng(seed)
    SH, SW = H + 160, W + 160
    sheet = paper.washi((SH, SW), seed, tint="#efe5cf", margin=(38, 36), fibre_density=0.9)
    sheet.alpha = paper.deckle((SH, SW), (38, 36), r, ragged=2.0)
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]

    # the mountain: long concave slopes from a small, broken summit
    px, py = W * 0.6, H * 0.2
    d = np.abs(xx - px)
    top = py + 1.5 * np.maximum(d - 34, 0) ** 0.86 * np.where(xx < px, 1.0, 1.05) + 2.5 * noise.line1d(W, 30, r)
    mountain = noise.smoothstep(top - 1.5, top + 1.5, yy)
    # snow: a cap with a ragged hem, and a few gullies where it runs further down, thinning
    cap = 34 + 26 * noise.line1d(W, 40, r) - 0.3 * d
    gully = np.zeros((H, W), np.float32)
    for _ in range(14):
        gx = px + r.normal(0, 55)
        L = r.uniform(50, 190)
        wid = r.uniform(3, 9)
        lean = r.normal(0, 0.12) + 0.35 * np.sign(gx - px) * min(1, abs(gx - px) / 120)
        depth = yy - (top[int(np.clip(gx, 0, W - 1))] + cap[int(np.clip(gx, 0, W - 1))] * 0.6)
        cx = gx + lean * np.clip(depth, 0, None)
        taper = wid * np.clip(1 - depth / L, 0, 1) ** 0.7
        gully = np.maximum(gully, noise.smoothstep(1.0, 0.0, np.abs(xx - cx) - taper) * (depth > -20) * (taper > 0.4))
    hem = top + np.maximum(cap, 0) + 6 * noise.field((H, W), 4, r)
    snow = mountain * np.maximum((yy < hem).astype(np.float32), gully * noise.smoothstep(300, 150, d))
    snow = noise.smoothstep(0.4, 0.6, snow + 0.25 * noise.field((H, W), 2.5, r) * (snow > 0.05))
    forest_top = H * 0.72 + 0.22 * (px - xx) * (xx < px) * 0 + 40 * noise.line1d(W, 400, r)

    # the cartouche is cut clear of the sky, so the title sits on bare paper
    card = np.zeros((H, W), np.float32)
    t = np.asarray(Image.open(lettering.CUT / "fuji_title.png"), np.float32) / 255
    t = np.asarray(Image.fromarray((t * 255).astype(np.uint8)).resize((int(t.shape[1] * 1.15), int(t.shape[0] * 1.15)),
                                                                       Image.LANCZOS), np.float32) / 255
    cy0, cx0 = 90, 110
    key = np.zeros((H, W), np.float32)
    key[cy0:cy0 + t.shape[0], cx0:cx0 + t.shape[1]] = t
    box_w = int(t.shape[1] * 0.64)
    card[cy0:cy0 + int(t.shape[0] * 0.99), cx0 + t.shape[1] - box_w:cx0 + t.shape[1]] = 1

    sky = (1 - mountain) * (1 - clouds((H, W), top, r)) * (1 - card)
    fade_red = relief.bokashi((H, W), 0, 0.64, 0.86)            # the red is wiped out toward the forest
    blocks = {
        "sky": (sky * relief.bokashi((H, W), 0, 0.0, 0.58, 0.3, 1.0), PRUSSIAN, (0.0, 0.0), 0.0),
        "red": (mountain * (1 - snow) * (0.25 + 0.75 * fade_red), BENGARA, (1.6, -1.1), 0.0004),
        "crown": (mountain * (1 - snow) * relief.bokashi((H, W), 0, py / H, py / H + 0.2), UMBER, (-1.2, 0.8), 0.0),
        "green": (mountain * noise.smoothstep(forest_top[None, :] - 170, forest_top[None, :] + 120, yy)
                  * relief.bokashi((H, W), 0, 1.0, 0.7, 0.55, 1.0), GREEN, (0.8, 1.5), -0.0003),
    }
    # pines on the lower slopes: small dark marks, thicker and larger toward the foot
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    n = 3200
    base = forest_top.mean()
    ty = base - 80 + (H - base + 80) * r.random(n) ** 0.7
    tx = r.uniform(0, W * 0.9, n) * (1 - 0.35 * r.random(n))
    for x, y in zip(tx, ty):
        if y < forest_top[int(np.clip(x, 0, W - 1))] - 60 or y < top[int(np.clip(x, 0, W - 1))] + 40:
            continue
        s = 2.5 + 8 * np.clip((y - base + 80) / (H - base + 80), 0, 1)
        dr.polygon([(x, y - s * 1.7), (x - s * 0.6, y + s * 0.3), (x + s * 0.6, y + s * 0.3)], fill=255)
    blocks["pines"] = (np.asarray(im, np.float32) / 255, PINE, (-0.6, 0.9), 0.0)
    blocks["key"] = (key, SUMI, (0.0, 0.0), 0.0)

    img = sheet.color.copy()
    for name in ("sky", "red", "crown", "green", "pines", "key"):
        block, ink, shift, turn = blocks[name]
        full = np.zeros((SH, SW), np.float32)
        full[80:80 + H, 80:80 + W] = block
        grain = np.zeros((SH, SW), np.float32)
        grain[80:80 + H, 80:80 + W] = relief.woodgrain((H, W), r)
        film = relief.ink_film(full, r, roller=0.0, squash=0.0, grain=grain, grain_strength=0.1) \
            * (1 + 0.07 * noise.fbm((SH, SW), 26, r, octaves=3))
        dens = relief.pull(full, film, sheet, r, pressure=1.0, shift=shift, turn=turn)
        img = glaze(img, ndimage.gaussian_filter(dens, 0.7) * 1.25, ink)
    return plate.mount(img, sheet, shadow=0.35)
