"""The film: the collection seen in a Claude glass, cut to a score that is composed and played in code.

    uv run timeline.py      writes plates/_timeline.mp4 (1920 x 1080, 30 fps, H.264 with AAC)

A Claude glass is a small, dark, convex mirror; the museum is named for one. In the film the glass is a dark
world whose rim catches the light. The paintings come up in its dark one after another, each of them whole and
hung in the same place, every cut falling on a note of the score (score.py). One of them is painted there
stroke by stroke. The cuts come quicker and quicker, the glass goes dark for the title, and the light comes up
along its rim.

The plates are shown as they are, in their own colours. numpy and PIL draw each frame and ffmpeg encodes the
frames with the music. The type is Iowan Old Style, from macOS.
"""

import importlib
import io
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import score
from atelier import impasto, music, noise, plate
from atelier.color import to_srgb
from atelier.plate import BACKDROP
from render import ROOMS, ROOT, hanging

W, H, FPS = 1920, 1080, 30
STEP = round(score.STEP * FPS)            # frames in a sixteenth of the score
assert abs(STEP - score.STEP * FPS) < 1e-9, "a sixteenth is a whole number of frames"
BAR, TAIL = 16 * STEP, round(score.TAIL * FPS)
BOX, MIDDLE = (0.86 * W, 0.78 * H), (W / 2, 0.45 * H)    # the most of the frame a painting fills, and its middle
RADIUS, RIM = 1.3 * W, 0.84 * H                           # the curve of the glass, and where its rim tops out
DARK = np.float32([10, 10, 11])           # the glass
IVORY, DIM = (236, 230, 216), (150, 145, 136)
SERIF = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
PAINTED = "saint_remy"                    # the work seen being painted
PAD = 16                                  # px of edge kept round every picture, for the filter reading near it
WORDS = {"paint": ("Every painting", "is a program."), "quick": ("One painter,", "many hands.")}

LIGHT = ((np.arange(256) / 255) ** 2.2).astype(np.float32)   # a byte of the frame as light, near enough
DARKEN = np.round(255 * np.linspace(0, 1, 4096) ** (1 / 2.2)).astype(np.float32)
HAZE = LIGHT[[96, 150, 196]] * 0.75                        # the cold light over the glass
WARM = LIGHT[[244, 128, 56]]                               # and the warm line along its rim
Y, X = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5


def setting(lines, gap=0.5):
    """Lines of type centred over one another, each (text, px, italic, tracking in ems, colour). -> RGBA"""
    rows = []
    for text, size, italic, track, colour in lines:
        f = ImageFont.truetype(SERIF, size, index=2 if italic else 0)
        steps = [f.getlength(ch) + track * size for ch in text] if track else [f.getlength(text)]
        im = Image.new("L", (int(sum(steps) + 8), int(size * 1.45)))
        d, x = ImageDraw.Draw(im), 4.0
        for part, step in zip(text if track else [text], steps):  # tracked type is set a letter at a time
            d.text((x, size * 1.1), part, font=f, fill=255, anchor="ls")
            x += step
        layer = Image.new("RGBA", im.size, colour)
        layer.putalpha(im)
        rows.append((layer, size))
    out = Image.new("RGBA", (max(r.width for r, _ in rows), sum(r.height for r, _ in rows)
                             + sum(int(s * gap) for _, s in rows[1:])))
    y = 0
    for i, (r, s) in enumerate(rows):
        y += int(s * gap) if i else 0
        out.alpha_composite(r, ((out.width - r.width) // 2, y))
        y += r.height
    return out


def overlay(canvas, im, y, alpha=1.0):
    """Lay the RGBA image `im` on the frame, centred across, its top at y."""
    if alpha <= 0.002:
        return
    a = np.asarray(im, np.float32)
    x, y = (W - im.width) // 2, int(round(y))
    region = canvas[y:y + im.height, x:x + im.width]
    region += (a[..., :3] - region) * (a[..., 3:] / 255 * alpha)


class Picture:
    """A picture as the film sees it: w x h px, drawn from `img` at any resolution."""

    def __init__(self, img, w=None, h=None):
        self.w, self.h = w or img.width, h or img.height
        self.s = img.width / self.w
        a = np.asarray(img.convert("RGB"))
        self.img = Image.fromarray(np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge"))


def unmounted(name):
    """A plate as the film shows it. A sheet mounted on the museum's dark wall comes off the wall, its shadow
    with it, onto the dark of the glass; a painting to its edges is left as it is. -> PIL image"""
    im = Image.open(ROOT / "plates" / f"{name}.jpg").convert("RGB")
    a = np.asarray(im).astype(np.float32)
    wall = np.float32(plate.to_srgb_255(BACKDROP))
    if np.abs(a[[0, 0, -1, -1], [0, -1, 0, -1]] - wall).max() > 12:
        return im
    lab, _ = ndimage.label((a.max(-1) < 70) & (np.ptp(a, -1) < 14))
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep = ndimage.gaussian_filter((~np.isin(lab, edge[edge > 0])).astype(np.float32), 0.8)
    ys, xs = np.nonzero(keep > 0.02)
    out = DARK + (a - DARK) * keep[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def blank():
    return np.broadcast_to(DARK, (H, W, 3)).copy()


def hang(canvas, pic, zoom=1.0):
    """The whole picture in the dark, as large as fits the box and `zoom` times that, about the middle;
    resampled from its own pixels for every frame, its edges drawn to a fraction of a pixel."""
    k = min(BOX[0] / pic.w, BOX[1] / pic.h) * zoom
    x0, y0 = MIDDLE[0] - pic.w * k / 2, MIDDLE[1] - pic.h * k / 2
    x1, y1 = x0 + pic.w * k, y0 + pic.h * k
    X0, Y0, X1, Y1 = max(0, math.floor(x0)), max(0, math.floor(y0)), min(W, math.ceil(x1)), min(H, math.ceil(y1))
    box = tuple((v - o) / k * pic.s + PAD for v, o in ((X0, x0), (Y0, y0), (X1, x0), (Y1, y0)))
    patch = np.asarray(pic.img.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box), np.float32)
    xs, ys = np.arange(X0, X1) + 0.5, np.arange(Y0, Y1) + 0.5
    m = (np.clip(ys - y0 + 0.5, 0, 1) * np.clip(y1 - ys + 0.5, 0, 1))[:, None, None] \
        * (np.clip(xs - x0 + 0.5, 0, 1) * np.clip(x1 - xs + 0.5, 0, 1))[None, :, None]
    region = canvas[Y0:Y1, X0:X1]
    region += (patch - region) * m


def whole(name):
    """A painting hung whole in the dark, the camera drawing a little nearer as it looks. -> frame(t)"""
    pic = Picture(unmounted(name))

    def frame(t):
        canvas = blank()
        hang(canvas, pic, 1 + 0.03 * t)
        return canvas
    return frame


def glass(top, light, dawn):
    """The dark glass under a band of cold light, its rim at `top`; with `dawn`, a warm line comes up along the
    rim and glows above it. Drawn in linear light. -> frame"""
    d = np.hypot(X - W / 2, Y - top - RADIUS) - RADIUS
    above, below = np.maximum(d, 0), np.minimum(d, 0)
    out = d > 0
    haze = (1 - np.exp(-above / 8)) * np.exp(-above / 75) * out + 0.3 * np.exp(below / 6) * ~out
    warm = (np.exp(-above / 3.5) + 0.25 * np.exp(-above / 22)) * out + 0.6 * np.exp(below / 2.5) * ~out
    lin = LIGHT[DARK.astype(np.uint8)] + light * (haze[..., None] * HAZE + dawn * warm[..., None] * WARM)
    return DARKEN[np.clip(lin * 4095 + 0.5, 0, 4095).astype(np.int32)]


def snapshots(slug, count, width):
    """The painting going on: `count` views of the canvas taken evenly through its strokes, `width` px across,
    as JPEG. The program runs twice: once to count its strokes, once to look."""
    m = importlib.import_module(f"works.{slug}")
    lay, seen, out = impasto.lay, [0], []
    try:
        impasto.lay = lambda *a, **k: lay(*a, watch=lambda rgb, height: seen.__setitem__(0, seen[0] + 1), **k)
        m.paint()
        look = set(np.round(np.linspace(1, seen[0], count)).astype(int))
        seen[0] = 0

        def watch(rgb, height):
            seen[0] += 1
            if seen[0] in look:
                step = max(1, rgb.shape[1] // (2 * width))
                im = Image.fromarray(np.clip(to_srgb(rgb[::step, ::step]) * 255 + 0.5, 0, 255).astype(np.uint8))
                buf = io.BytesIO()
                im.resize((width, round(width * rgb.shape[0] / rgb.shape[1])), Image.LANCZOS).save(buf, "JPEG", quality=92)
                out.append(buf.getvalue())
        impasto.lay = lambda *a, **k: lay(*a, watch=watch, **k)
        m.paint()
    finally:
        impasto.lay = lay
    return out


def painting(slug, n):
    """A work being painted, `n` frames: the canvas filling stroke by stroke, and at the end the light coming
    on over the finished paint. -> frame(t)"""
    final = Picture(unmounted(slug))
    stills = snapshots(slug, round(0.85 * n), round(final.w * min(BOX[0] / final.w, BOX[1] / final.h) * 1.1))
    seen = {}

    def frame(t):
        i = round(min(1.0, t / 0.85) * (len(stills) - 1))
        if i not in seen:
            seen.clear()
            seen[i] = Picture(Image.open(io.BytesIO(stills[i])), final.w, final.h)
        canvas = blank()
        hang(canvas, seen[i])
        lit = noise.smoothstep(0.88, 0.97, t)
        if lit > 0:
            done = blank()
            hang(done, final)
            canvas += (done - canvas) * lit
        return canvas
    return frame


def order(mods):
    """The works as a visitor meets them: room by room, oldest first in each."""
    return [s for room in ROOMS for s in sorted((s for s, m in mods.items() if m.ROOM == room), key=lambda s: mods[s].YEAR)]


def plan(mods):
    """The parts of the film for the score, (kind, bars), and its shots, (part, what, frames). The first works
    are seen a bar each, then two a bar, the last eighteen on every note of the figure, and before the title
    a flash of the collection on every sixteenth."""
    works = [s for s in order(mods) if s != PAINTED]
    slow, middle, quick = works[:4], works[4:-18], works[-18:]
    assert len(middle) >= 2, "enough works to cut between"
    gaps = [int(g) * STEP for g in np.diff(score.RHYTHM + (16,))]
    flash = works[::max(1, len(works) // 15)][:15]
    shots = [("intro", "glass", 2 * BAR)] + [("slow", s, BAR) for s in slow] + [("paint", PAINTED, 4 * BAR)]
    shots += [("cuts", s, BAR // 2 * (1 + (i == 0 and len(middle) % 2))) for i, s in enumerate(middle)]
    shots += [("quick", s, gaps[i % 6]) for i, s in enumerate(quick)]
    shots += [("burst", s, STEP) for s in flash] + [("burst", "dark", (16 - len(flash)) * STEP)]
    shots += [("title", "glass", 2 * BAR), ("coda", "glass", 2 * BAR + TAIL)]
    parts = [("intro", 2), ("slow", len(slow)), ("paint", 4), ("cuts", math.ceil(len(middle) / 2)), ("quick", 3),
             ("burst", 1), ("title", 2), ("coda", 2)]
    assert sum(f for _, _, f in shots) == sum(n for _, n in parts) * BAR + TAIL, "the cuts fill the bars"
    return parts, shots


def film(out=ROOT / "plates" / "_timeline.mp4"):
    t0 = time.time()
    mods = {s: importlib.import_module(f"works.{s}") for s in hanging()}
    parts, shots = plan(mods)
    title = setting([("The Claude Glass", 84, False, 0.02, IVORY)])
    credit = setting([("Painted in code by Claude", 32, True, 0, DIM)])
    url = setting([("github.com/Chaoqi31/claude-glass", 20, False, 0.12, DIM)])
    said = {w: setting([(w, 46, False, 0.01, IVORY)]) for pair in WORDS.values() for w in pair}
    below = MIDDLE[1] + BOX[1] / 2 + 22                       # where the words stand, under the paintings
    begins, span = {}, {}
    for part, _, n in shots:
        begins.setdefault(part, sum(span.values()))
        span[part] = span.get(part, 0) + n
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "score.wav"
        music.write(wav, score.render(parts))
        print(f"score: {sum(n for _, n in parts)} bars, {time.time() - t0:.0f}s", flush=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a",
             # tagged as sRGB, which it is, so that Safari and QuickTime show the colours the plates have
             "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=iec61966-2-1:range=tv",
             "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-tune", "film", "-x264-params", "aq-mode=3",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], stdin=subprocess.PIPE)
        seen, start = {}, 0
        for j, (part, name, n) in enumerate(shots):
            if part == "paint":
                frame = painting(name, n)
            elif name not in ("glass", "dark"):
                frame = seen[name] = seen.get(name) or whole(name)
            for i in range(n):
                into = start + i - begins[part]
                if name == "dark":
                    canvas = blank()
                elif part == "intro":
                    canvas = glass(RIM, noise.smoothstep(0, 1.5 * BAR, into), 0.25 * noise.smoothstep(0.5 * BAR, 2 * BAR, into))
                elif name == "glass":                                  # the title and the coda are one shot
                    u = into + (span["title"] if part == "coda" else 0)
                    rise = noise.smoothstep(0, 4 * BAR + TAIL, u)
                    canvas = glass(RIM - 0.1 * H * rise, 1.0, 0.3 + 0.9 * noise.smoothstep(BAR, 4 * BAR, u))
                    y = 0.36 * H - 0.05 * H * rise
                    overlay(canvas, title, y, noise.smoothstep(0, 6, u))
                    overlay(canvas, credit, y + title.height + 26, noise.smoothstep(2 * BAR, 2 * BAR + 20, u))
                    overlay(canvas, url, y + title.height + credit.height + 46, noise.smoothstep(3 * BAR, 3 * BAR + 20, u))
                    if part == "coda":
                        canvas *= 1 - noise.smoothstep(n - 1.4 * FPS, n - 0.2 * FPS, i)
                else:
                    canvas = frame(i / n)
                if part in WORDS:                                      # the words stay put across the cuts
                    half = span[part] / 2
                    u = into % half
                    overlay(canvas, said[WORDS[part][int(into >= half)]], below,
                            noise.smoothstep(0, 6, u) * noise.smoothstep(half, half - 6, u))
                enc.stdin.write(np.clip(canvas + 0.5, 0, 255).astype(np.uint8).tobytes())
            start += n
            print(f"{j + 1:3d}/{len(shots)} {part:6s} {name:16s} {n:4d} frames {time.time() - t0:6.0f}s", flush=True)
        enc.stdin.close()
        enc.wait()
    print(f"{out.name}: {len(mods)} works, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    film()
