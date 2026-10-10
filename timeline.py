"""The film: a minute in which Claude Opus 5.5 writes a brush in Python and paints with it, cut to a score that is
composed and played in code.

    uv run timeline.py      writes plates/_timeline.mp4 (1920 x 1080, 30 fps, H.264 with AAC)

Two lines come up in the dark: Claude Opus 5.5 has never held a brush. So it wrote one, and the code comes up as
it was written. A bare canvas takes its first strokes, one on each note of the voice, then hundreds, then
thousands, until the olive grove stands finished. Another painting is seen as its own program: its code, set small
and each letter tinted with the colour of the painting where it falls, draws back until the painting is there, and
the code gives way to it. The camera goes into a third, nearer and nearer, until one pixel fills the screen with
its value. Four things the paintings were not made with come up on four blows of brass; then the paintings go by
on the notes of the figure, and every work in the museum comes onto one wall, each out of its program in its own
place, faster and faster, until the music breaks off. The wall stands whole in the silence, and the title falls
with the drum.

The plates are shown as they are, in their own colours. numpy and PIL draw each frame and ffmpeg encodes the frames
with the music. The type is Iowan Old Style and SF Mono, from macOS.
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

import score
from atelier import impasto, music, noise
from atelier.color import to_srgb
from render import ROOT, even_rows, hanging, words

W, H, FPS = 1920, 1080, 30
STEP = round(score.STEP * FPS)            # frames in a sixteenth of the score
assert abs(STEP - score.STEP * FPS) < 1e-9, "a sixteenth is a whole number of frames"
BAR, TAIL = 16 * STEP, round(score.TAIL * FPS)
PART = {}                                 # each part of the film: the frames it begins and ends on
for _name, _bars in score.FORM:
    _at = score.starts()[_name] * BAR
    PART[_name] = (_at, _at + _bars * BAR)

DARK = np.float32([10, 10, 11])
IVORY, DIM = (236, 230, 216), (150, 144, 132)
VERMILION = (200, 64, 46)                 # the museum's one accent
SERIF = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
ROMAN, ITALIC = 0, 2                      # faces of the serif
PAD = 16                                  # px of edge kept round every picture, for the filter reading near it
BOX = (160, 150, 1760, 1010)              # where a painting hangs, whole, when it is shown alone

OPENING = ("Claude Opus 5.5", "has never held a brush.")
WROTE = "So it wrote one."
SOURCE = ("atelier/impasto.py", 21)       # the code written on the screen: its first lines
PHRASE = "laid stroke by stroke from a loaded brush."              # lit when it has been written
PAINTED = "saint_remy"                    # painted stroke by stroke
FIRST = 4.0                               # how much nearer than the whole canvas's width its first strokes are seen
WOVEN = "starry"                          # seen as its own program
CLOSE = ("red_fuji", (791, 824))          # gone into until one pixel fills the screen, and which pixel
NOT = ("No image model.", "No photograph.", "No tracing.", "Only Python.")   # one on each blow of brass
# the montage, in sixteenths from its start: four works held half a bar, each coming out of its program; twelve on
# the notes of the figure, shown as they are, each a little lighter than the last, so that no cut leaps from dark
# to bright; then from WALL every work comes onto one wall, each out of its program in its own place, faster and
# faster over WALL_FOR sixteenths, and the wall stays whole through the silence before the title
MONTAGE = ("hakone", "nice", "impression", "rose_window",
           "burbank", "attersee", "yatsuhashi", "haystacks", "sayama", "washington",
           "almond", "javea", "sydney", "collioure", "louveciennes", "turner")
CUTS = [0, 8, 16, 24] + [32 + s for s in score.RHYTHM] + [48 + s for s in score.RHYTHM]
SLOW = 8                                  # sixteenths a work is held for it to come out of its program; quicker cuts
                                          # between program and painting would flicker
WALL, WALL_FOR = 64, 20
assert len(CUTS) == len(MONTAGE), "a painting for every cut"


def in_out(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (2 - 2 * t) ** 3 / 2


def gently(t):
    """Easing in and out with the least haste in the middle."""
    return (1 - math.cos(math.pi * min(max(t, 0.0), 1.0))) / 2


def out_expo(t):
    return 1 - 2 ** (-10 * min(max(t, 0.0), 1.0))


class Picture:
    """A picture as the film draws it: its colours and, for type, how much of each pixel it covers; kept at its
    own size and at halves of it, for when it is seen small."""

    def __init__(self, rgb, cover=None):
        self.h, self.w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        mask = None if cover is None else Image.fromarray(np.clip(cover * 255 + 0.5, 0, 255).astype(np.uint8))
        self.levels = []
        while True:
            pad = lambda im, mode: Image.fromarray(np.pad(np.asarray(im), ((PAD, PAD), (PAD, PAD)) + ((0, 0),) * (im.mode == "RGB"), mode=mode))
            self.levels.append((self.w / img.width, pad(img, "edge"), None if mask is None else pad(mask, "constant")))
            if min(img.size) < 200:
                break
            img, mask = img.reduce(2), None if mask is None else mask.reduce(2)


def place(canvas, pic, x0, y0, x1, y1, alpha=1.0, clip=None):
    """Draw the whole of `pic` into the rectangle (x0, y0)-(x1, y1) of the frame, fractional, and only inside
    `clip` if given: resampled for this frame from the nearest of its sizes, its edges drawn to a fraction of a
    pixel. `alpha` may be a number or a function of the frame's rows, for a picture coming in from the top."""
    cx0, cy0, cx1, cy1 = (x0, y0, x1, y1) if clip is None else clip
    cx0, cy0, cx1, cy1 = max(cx0, x0, 0), max(cy0, y0, 0), min(cx1, x1, W), min(cy1, y1, H)
    X0, Y0, X1, Y1 = math.floor(cx0), math.floor(cy0), math.ceil(cx1), math.ceil(cy1)
    if X1 - X0 < 1 or Y1 - Y0 < 1 or (not callable(alpha) and alpha <= 0.002):
        return
    k = (x1 - x0) / pic.w                                    # px of the frame to a px of the picture
    s, img, mask = next((lv for lv in reversed(pic.levels) if lv[0] * k <= 0.75), pic.levels[0])
    box = [(v - o) / k / s + PAD for v, o in ((X0, x0), (Y0, y0), (X1, x0), (Y1, y0))]
    box = tuple(min(max(v, 0), m) for v, m in zip(box, img.size * 2))
    patch = np.asarray(img.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box), np.float32)
    xs, ys = np.arange(X0, X1) + 0.5, np.arange(Y0, Y1) + 0.5
    rows = alpha(ys) if callable(alpha) else alpha
    m = (np.clip(ys - cy0 + 0.5, 0, 1) * np.clip(cy1 - ys + 0.5, 0, 1) * rows)[:, None] \
        * (np.clip(xs - cx0 + 0.5, 0, 1) * np.clip(cx1 - xs + 0.5, 0, 1))[None, :]
    if mask is not None:
        m = m * np.asarray(mask.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box), np.float32) / 255
    region = canvas[Y0:Y1, X0:X1]
    region += (patch - region) * m[..., None]


def emerge(canvas, box, pic, code, into, hold=8, sweep=6, soft=40):
    """A painting coming out of its program in `box`, `into` frames after it came: the program alone for `hold`
    frames, then the painting coming down over it from the top in `sweep`, its edge `soft` px deep."""
    if into < hold + sweep:
        place(canvas, code, *box)
    if into >= hold:
        edge = box[1] + (box[3] - box[1] + soft) * min(1.0, (into - hold + 1) / sweep)
        place(canvas, pic, *box, lambda ys: np.clip((edge - ys) / soft, 0, 1))


class Type:
    """A line of type, `size` px, in `font` (face `face` of it), tracked by `track` ems, drawn `scale` times as
    large and shown smaller, for sharp edges. w and h are its size, its baseline 1.1 sizes down."""

    def __init__(self, text, size, colour=IVORY, face=ROMAN, track=0.0, font=SERIF, scale=2):
        f = ImageFont.truetype(font, round(size * scale), index=face)
        if font == MONO:
            f.set_variation_by_name("Regular")                 # its default is its lightest
        s = size * scale
        steps = [f.getlength(ch) + track * s for ch in text] if track else [f.getlength(text)]
        im = Image.new("L", (int(sum(steps) + 8 * scale), int(s * 1.45)))
        d, x = ImageDraw.Draw(im), 4.0 * scale
        for part, step in zip(text if track else [text], steps):   # tracked type is set a letter at a time
            d.text((x, s * 1.1), part, font=f, fill=255, anchor="ls")
            x += step
        a = np.asarray(im, np.float32) / 255
        self.pic = Picture(np.ascontiguousarray(np.broadcast_to(np.uint8(colour), a.shape + (3,))), a)
        self.w, self.h, self.size = im.width / scale, im.height / scale, size


def rise(f, start, frames=22):
    """How far type that begins to rise into place at frame `start` has to go, in its heights, and how strongly it
    shows. -> (offset, alpha)"""
    e = out_expo((f - start) / frames)
    return (1 - e) * 0.55, min(1.0, e * 1.5) if f >= start else 0.0


def said(canvas, f, t, y, begins, fade=1.0, x=None, frames=22):
    """Type t with its top at y, across the middle of the frame or from x, rising into place from frame `begins`."""
    off, alpha = rise(f, begins, frames)
    x = (W - t.w) / 2 if x is None else x
    place(canvas, t.pic, x, y + off * t.h, x + t.w, y + (1 + off) * t.h, alpha * fade, clip=(x, y, x + t.w, y + t.h))


def fit(w, h, box=BOX):
    """A w x h picture as large as it goes in `box`, centred there. -> (x0, y0, x1, y1)"""
    k = min((box[2] - box[0]) / w, (box[3] - box[1]) / h)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cx - w * k / 2, cy - h * k / 2, cx + w * k / 2, cy + h * k / 2


def plate(slug):
    return Image.open(ROOT / "plates" / f"{slug}.jpg").convert("RGB")


def lifted(im):
    """A picture's colours made light enough to read as letters on the dark. -> uint8 (h, w, 3)"""
    hsv = np.asarray(im.convert("HSV"), np.float32)
    hsv[..., 2] = 255 * (0.58 + 0.42 * hsv[..., 2] / 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def woven(slug, width, size, ground=0.0, at=None):
    """A painting set as its own program, `width` px across: the program's text run on line after line in a
    fixed-width face of `size` px, as many lines as the painting is tall, each letter in the painting's colour
    where it falls, over the painting itself at `ground` of its strength. `at` puts a passage of the text,
    (passage, row as a fraction of the rows), at the middle of that row. -> (Picture, the cell of the middle of
    the passage in px)"""
    f = ImageFont.truetype(MONO, size)
    f.set_variation_by_name("Semibold")
    adv, lh = f.getlength("M"), round(size * 1.08)
    im = plate(slug)
    cols = int(width / adv)
    rows = max(1, round(cols * adv * im.height / im.width / lh))
    text = " ".join((ROOT / "works" / f"{slug}.py").read_text().split()) + " "     # its lines run together
    middle = (0, 0)
    if at:
        cell = int(at[1] * rows) * cols + cols // 2 - len(at[0]) // 2
        k = (text.index(at[0]) - cell) % len(text)
        text = text[k:] + text[:k]
        middle = ((cols // 2) * adv, (int(at[1] * rows) + 0.5) * lh)
    text = text * (cols * rows // len(text) + 1)
    mask = Image.new("L", (math.ceil(cols * adv), rows * lh))
    d = ImageDraw.Draw(mask)
    for i in range(rows):
        d.text((0, i * lh), text[i * cols:(i + 1) * cols], font=f, fill=255)
    colours = np.asarray(Image.fromarray(lifted(im.resize((cols, rows), Image.BOX))).resize(mask.size, Image.NEAREST), np.float32)
    ink = colours * (np.asarray(mask, np.float32)[..., None] / 255)
    if ground:
        ink = np.maximum(ink, ground * np.asarray(im.resize(mask.size, Image.LANCZOS), np.float32))
    return Picture((ink + 0.5).astype(np.uint8)), middle


def snapshots(slug, schedule, width):
    """The painting going on: views of the canvas, `width` px across, as JPEG, after each of the strokes that
    `schedule(total strokes)` lists. The program runs twice: once to count its strokes, once to look.
    -> ({stroke: jpeg}, total)"""
    m = importlib.import_module(f"works.{slug}")
    lay, seen, out = impasto.lay, [0], {}

    def watch(rgb, height):
        seen[0] += 1
        if seen[0] in look:
            im = Image.fromarray(np.clip(to_srgb(rgb) * 255 + 0.5, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            im.resize((width, round(width * rgb.shape[0] / rgb.shape[1])), Image.LANCZOS).save(buf, "JPEG", quality=92)
            out[seen[0]] = buf.getvalue()
    try:
        impasto.lay = lambda *a, **k: lay(*a, watch=lambda rgb, height: seen.__setitem__(0, seen[0] + 1), **k)
        m.paint()
        total, look = seen[0], set(schedule(seen[0]))
        seen[0] = 0
        impasto.lay = lambda *a, **k: lay(*a, watch=watch, **k)
        m.paint()
    finally:
        impasto.lay = lay
    return out, total


class Film:
    """The whole film, frame by frame."""

    def __init__(self, mods):
        self.length = PART["title"][1] + TAIL
        self.opening = [Type(OPENING[0], 46, DIM, track=0.02), Type(OPENING[1], 46)]
        self.wrote = Type(WROTE, 46)
        self.code()
        self.strokes()
        self.weave()
        self.close = plate(CLOSE[0])
        self.near = np.asarray(self.close)
        self.whole = Picture(self.near)
        self.not_ = [Type(t, 64) for t in NOT]
        n = words(len(mods)).capitalize()
        self.words = [Type(f"{n} paintings.", 40, track=0.01), Type(f"{n} programs.", 40, track=0.01)]
        self.cuts = []
        for slug, held in zip(MONTAGE, np.diff(CUTS + [WALL])):   # each at the size it is shown, and the slow
            im = plate(slug)                                        # ones as their programs too
            box = fit(*im.size)
            size = (round(box[2] - box[0]), round(box[3] - box[1]))
            code = woven(slug, size[0], 9, ground=0.3)[0] if held >= SLOW else None
            self.cuts.append((box, Picture(np.asarray(im.resize(size, Image.LANCZOS))), code))
        self.wall = self.hang(mods)
        self.title = [(Type("THE CLAUDE GLASS", 104, track=0.1), 452, 0),
                      (Type(OPENING[0], 34, DIM, track=0.04), 372, 2 * STEP * 4),
                      (Type(f"{n} paintings. Every pixel written in Python.", 34, face=ITALIC), 610, BAR),
                      (Type("chaoqi31.github.io/claude-glass", 24, DIM, font=MONO), 690, 2 * BAR)]

    def hang(self, mods):
        """Every work on one wall, whole, oldest first, in rows of even height as large as BOX holds them; each
        with its program at its size, and the frame of the montage it comes in on: scattered over the wall, one
        on each sixteenth at first and more and more of them to a sixteenth. -> [(box, painting, program, frame)]"""
        slugs = sorted(mods, key=lambda s: mods[s].YEAR)
        ims = [plate(s) for s in slugs]
        asp = [im.width / im.height for im in ims]
        bw, bh, gap = BOX[2] - BOX[0], BOX[3] - BOX[1], 10

        def lay(k):                                           # k rows: (first, end, height) each, and the height of all
            lines = [(i, j, (bw - gap * (j - i - 1)) / sum(asp[i:j])) for i, j in even_rows(asp, k)]
            return lines, sum(h for *_, h in lines) + gap * (k - 1)
        lines, tall = max((lay(k) for k in range(2, 13)), key=lambda lt: lt[1] * min(1.0, bh / lt[1]) ** 2)
        s = min(1.0, bh / tall)                               # the scale that fits them in
        boxes, y = [], BOX[1] + (bh - s * tall) / 2
        for i, j, h in lines:
            x = BOX[0] + bw * (1 - s) / 2
            for k in range(i, j):
                boxes.append((round(x), round(y), round(x + asp[k] * h * s), round(y + h * s)))
                x += (asp[k] * h + gap) * s
            y += (h + gap) * s
        order = noise.rng(60).permutation(len(slugs))
        out = []
        for rank, k in enumerate(order):
            x0, y0, x1, y1 = boxes[k]
            painting = Picture(np.asarray(ims[k].resize((x1 - x0, y1 - y0), Image.LANCZOS)))
            frame = (WALL + int(WALL_FOR * (rank / len(slugs)) ** 0.5)) * STEP
            out.append((boxes[k], painting, woven(slugs[k], x1 - x0, 9, ground=0.3)[0], frame))
        return out

    def code(self):
        """The program written on the screen: its lines as type, each token's place, and when it is written."""
        lines = (ROOT / SOURCE[0]).read_text().splitlines()[:SOURCE[1]]
        f = ImageFont.truetype(MONO, 26)
        self.adv, self.lead = f.getlength("M"), 38
        self.left, self.top = (W - max(len(s) for s in lines) * self.adv) / 2, (H - len(lines) * self.lead) / 2
        self.lines = [Type(s, 26, font=MONO) if s else None for s in lines]
        # where the phrase that is lit at the end lies: (line, first column, last column)
        i = next(k for k, s in enumerate(lines) if PHRASE in s)
        self.phrase = (i, lines[i].index(PHRASE), lines[i].index(PHRASE) + len(PHRASE))
        self.lit = Type(PHRASE, 26, VERMILION, font=MONO)
        ends = np.cumsum([len(s) + 1 for s in lines])
        self.ends, self.chars = ends, ends[-1]

    def written(self, f):
        """How many characters of the program are on the screen at frame f: it is written in bursts, one on each
        sixteenth from the middle of its first bar, each longer than the last."""
        start = PART["wrote"][0] + BAR // 2
        k = (f - start) // STEP + 1                           # bursts so far
        return 0 if k <= 0 else int(self.chars * min(1.0, (k / 20) ** 1.35))

    def strokes(self):
        """The painting painted: the strokes on the screen at each frame, and the canvas after them. On each note
        of the figure's first two bars the strokes double, one, two, four, up to two thousand; then they come on
        faster and faster until the painting is done."""
        a, b = PART["strokes"]
        self.notes = [a + bar * BAR + s * STEP for bar in range(2) for s in score.RHYTHM]
        first, last = a + 2 * BAR, b - 3 * BAR // 8
        self.count = {}

        def schedule(total):
            for f in range(a, b):
                n = 2 ** (sum(f >= t for t in self.notes) - 1)
                if f >= first:
                    u = min(1.0, (f - first) / (last - first))
                    n = round(n * (total / n) ** u)
                self.count[f] = n
            return sorted(set(self.count.values()))
        self.lit_plate = Picture(np.asarray(plate(PAINTED)))
        self.jpegs, self.total = snapshots(PAINTED, schedule, self.lit_plate.w)    # at full size, to be seen close
        self.seen = {}
        self.box = fit(self.lit_plate.w, self.lit_plate.h)

    def canvas_at(self, n):
        """The canvas after n strokes, decoded when first wanted."""
        if n not in self.seen:
            if len(self.seen) > 3:
                self.seen.clear()
            self.seen[n] = Picture(np.asarray(Image.open(io.BytesIO(self.jpegs[n])).convert("RGB")))
        return self.seen[n]

    def weave(self):
        """The painting seen as its program, large enough to read where the camera starts."""
        im = plate(WOVEN)
        self.box_w = fit(*im.size)
        self.weave_pic, self.anchor = woven(WOVEN, 200 * 35, 56, at=("def paint(", 0.55))    # starting on the line
                                                                                               # that paints it
        self.weave_paint = Picture(np.asarray(im))
        lines = len((ROOT / "works" / f"{WOVEN}.py").read_text().splitlines())
        self.weave_note = Type(f"works/{WOVEN}.py · {lines} lines", 22, DIM, font=MONO)

    def frame(self, f):
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = DARK
        part = next(name for name, (a, b) in PART.items() if a <= f < b) if f < PART["title"][1] else "title"
        getattr(self, "draw_" + part)(canvas, f, f - PART[part][0])
        if f < 12:
            canvas *= noise.smoothstep(0, 12, f)              # out of the dark on the first note
        return canvas

    def draw_intro(self, canvas, f, u):
        a, b = PART["intro"]
        go = 1 - noise.smoothstep(b - 10, b - 2, f)
        said(canvas, f, self.opening[0], 0.44 * H - 40, a + 6, go, frames=30)
        said(canvas, f, self.opening[1], 0.44 * H + 30, a + BAR, go, frames=30)

    def draw_wrote(self, canvas, f, u):
        a, b = PART["wrote"]
        if u < BAR // 2:
            said(canvas, f, self.wrote, 0.46 * H - 20, a, 1 - noise.smoothstep(a + BAR // 2 - 6, a + BAR // 2, f))
            return
        n, done = self.written(f), f >= b - 4 * STEP
        dim = noise.smoothstep(b - 4 * STEP, b - 3 * STEP, f)
        for k, t in enumerate(self.lines):
            first = 0 if k == 0 else self.ends[k - 1]
            shown = min(max(n - first, 0), self.ends[k] - first - 1)
            if t is None or shown <= 0:
                continue
            y, cut = self.top + k * self.lead, self.left + 4 + shown * self.adv   # type is drawn 4 px in
            place(canvas, t.pic, self.left, y, self.left + t.w, y + t.h, 1 - 0.65 * dim, clip=(self.left, y, cut, y + t.h))
            if k == self.phrase[0] and done:
                x = self.left + self.phrase[1] * self.adv
                place(canvas, self.lit.pic, x, y, x + self.lit.w, y + self.lit.h, dim)
        if not done:                                          # the cursor where the writing has got to
            k = int(np.searchsorted(self.ends, n, side="right"))
            k = min(k, len(self.lines) - 1)
            col = n - (self.ends[k - 1] if k else 0)
            x, y = self.left + 4 + col * self.adv, self.top + k * self.lead + 6
            canvas[int(y):int(y + 30), int(x):int(x + 13)] = IVORY

    def draw_strokes(self, canvas, f, u):
        a, b = PART["strokes"]
        n = self.count[f]
        x0, y0, x1, y1 = self.box
        fw, fh = x1 - x0, y1 - y0
        near = W / fw * 1.02                                  # close enough that the canvas fills the frame across,
        first = 1 - gently(u / BAR)                           # and closer still on the first strokes, at its left
        back = gently((u - 2 * BAR) / BAR)                    # edge; leaning in a little, then back to the whole
        z = (near * FIRST ** first * (1 + 0.05 * min(u, 2 * BAR) / (2 * BAR))) ** (1 - back)
        cw, ch = z * fw, z * fh
        cx = 0.5 - 0.45 * first * (1 - back)                  # the point of the canvas looked at, in its fractions
        cy = (0.25 * first + 0.32 * (1 - first)) * (1 - back) + 0.5 * back
        between = lambda v, lo, hi: min(max(v, min(lo, hi)), max(lo, hi))
        left = between(W / 2 - cx * cw, 0, W - cw)            # covering the frame while larger than it, inside
        top = between(H / 2 + ((y0 + y1) / 2 - H / 2) * back - cy * ch, 0, H - ch)      # it once smaller
        rect = (left, top, left + cw, top + ch)
        place(canvas, self.canvas_at(n), *rect)
        lit = noise.smoothstep(b - 3 * BAR // 8 + 4, b - 6, f)          # the finished painting, lit as on its plate
        place(canvas, self.lit_plate, *rect, lit)
        label = f"{self.total:,} strokes" if n >= self.total else f"stroke {n:,}" if n == 1 else f"{n:,} strokes"
        t = self.note(label, DIM if back > 0.98 else (72, 66, 58))      # under the canvas, or on it while close
        y = min(rect[3], H - 60) + 22 if back > 0.98 else H - 64
        place(canvas, t.pic, (W - t.w) / 2, y, (W + t.w) / 2, y + t.h, noise.smoothstep(a, a + 8, f))

    def note(self, text, colour=DIM, size=22, cache={}):
        if (text, colour, size) not in cache:
            cache[text, colour, size] = Type(text, size, colour, font=MONO)
        return cache[text, colour, size]

    def draw_woven(self, canvas, f, u):
        a, b = PART["woven"]
        span = b - a
        e = gently(u / (span - 3 * BAR // 4)) ** 1.6          # drawing back, slowly at first, then still
        x0, y0, x1, y1 = self.box_w
        z1 = (x1 - x0) / self.weave_pic.w
        z = math.exp(math.log(0.8) + (math.log(z1) - math.log(0.8)) * e)
        ax, ay = self.anchor
        px = W / 2 + (x0 + ax * z1 - W / 2) * e               # where the anchor stands on the screen
        py = H / 2 + (y0 + ay * z1 - H / 2) * e
        ox, oy = px - ax * z, py - ay * z
        mx1, my1 = ox + self.weave_pic.w * z, oy + self.weave_pic.h * z
        fill = 0.5 * noise.smoothstep(0.3, 0.95, e)           # the painting showing between the letters as they
                                                               # grow too small to carry it
        place(canvas, self.weave_paint, ox, oy, mx1, my1, fill)
        glyphs = np.zeros_like(canvas)
        place(glyphs, self.weave_pic, ox, oy, mx1, my1)
        np.maximum(canvas, glyphs, out=canvas)
        sweep = (f - (b - 3 * BAR // 4)) / (BAR // 2)         # the painting coming in from the top, over its code
        if sweep > 0:
            edge = y0 + (y1 - y0 + 60) * in_out(sweep)
            place(canvas, self.weave_paint, x0, y0, x1, y1, lambda ys: np.clip((edge - ys) / 60, 0, 1))
        if f >= b - BAR // 4:
            said(canvas, f, self.weave_note, y1 + 22, b - BAR // 4)

    def draw_pixel(self, canvas, f, u):
        a, b = PART["pixel"]
        span = b - a - BAR // 4
        e = gently(u / span)
        hh, ww = self.near.shape[:2]
        x0, y0, x1, y1 = fit(ww, hh)
        s0, s1 = (x1 - x0) / ww, 1400.0
        s = math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * e)
        tx, ty = CLOSE[1][0] + 0.5, CLOSE[1][1] + 0.5
        px = x0 + tx * s0 + (W / 2 - x0 - tx * s0) * e        # the pixel's place on the screen
        py = y0 + ty * s0 + (H / 2 - y0 - ty * s0) * e
        ox, oy = px - tx * s, py - ty * s                     # where the plate's corner stands
        sharp = noise.smoothstep(2.5, 6.0, s)
        if sharp < 1:
            place(canvas, self.whole, ox, oy, ox + ww * s, oy + hh * s, 1 - sharp)
        if sharp > 0:                                         # each pixel a square of its own colour
            ix = np.floor((np.arange(W) + 0.5 - ox) / s).astype(int)
            iy = np.floor((np.arange(H) + 0.5 - oy) / s).astype(int)
            okx, oky = (ix >= 0) & (ix < ww), (iy >= 0) & (iy < hh)
            block = self.near[np.clip(iy, 0, hh - 1)][:, np.clip(ix, 0, ww - 1)].astype(np.float32)
            inside = (oky[:, None] & okx[None, :])[..., None]
            grid = 0.55 * noise.smoothstep(14, 40, s)        # the lines between them
            if grid > 0:
                w = max(1.0, s / 60)
                fx = (np.arange(W) + 0.5 - ox) / s % 1 * s
                fy = (np.arange(H) + 0.5 - oy) / s % 1 * s
                line = np.maximum((fx < w)[None, :], (fy < w)[:, None])
                block *= 1 - grid * line[..., None]
            canvas[:] = canvas + (np.where(inside, block, canvas) - canvas) * sharp
        if s > 110:                                           # and its value written in it
            self.values(canvas, ox, oy, s, ww, hh)
        if f >= b - BAR // 2:
            n = int(ty) * ww + int(tx)
            t = self.note(f"pixel {n:,} of {ww * hh:,}", IVORY, 26)
            said(canvas, f, t, H / 2 + 70, b - BAR // 2)

    def values(self, canvas, ox, oy, s, ww, hh):
        size = min(0.14 * s, 60)
        alpha = noise.smoothstep(110, 220, s)
        font = ImageFont.truetype(MONO, max(6, round(size)))
        mask = Image.new("L", (W, H))
        dark = Image.new("L", (W, H))
        d, dd = ImageDraw.Draw(mask), ImageDraw.Draw(dark)
        i0, i1 = max(0, int(-ox / s)), min(ww, int((W - ox) / s) + 1)
        j0, j1 = max(0, int(-oy / s)), min(hh, int((H - oy) / s) + 1)
        for j in range(j0, j1):
            for i in range(i0, i1):
                r, g, b = (int(v) for v in self.near[j, i])
                cx, cy = ox + (i + 0.5) * s, oy + (j + 0.5) * s
                (dd if 0.299 * r + 0.587 * g + 0.114 * b > 150 else d).text((cx, cy), f"{r} {g} {b}", font=font, fill=255, anchor="mm")
        for m, colour in ((mask, (236, 230, 216)), (dark, (30, 28, 26))):
            a = np.asarray(m, np.float32)[..., None] / 255 * alpha
            canvas += (np.float32(colour) - canvas) * a

    def draw_statements(self, canvas, f, u):
        hits = [s for s, _ in score.HITS]
        k = max(i for i, s in enumerate(hits) if u >= s * STEP)
        t = self.not_[k]
        said(canvas, f, t, 0.46 * H - 40, PART["statements"][0] + hits[k] * STEP, frames=10)

    def draw_montage(self, canvas, f, u):
        self.cut(canvas, f, u)

    def draw_breath(self, canvas, f, u):
        self.cut(canvas, f, u + PART["breath"][0] - PART["montage"][0])

    def cut(self, canvas, f, u):
        """The montage at u frames from its start: the painting of the moment, or the wall as far as it has come."""
        s = u / STEP
        if s < WALL:
            k = max(i for i, c in enumerate(CUTS) if s >= c)
            box, pic, code = self.cuts[k]
            if code is None:
                place(canvas, pic, *box)
            else:
                emerge(canvas, box, pic, code, u - CUTS[k] * STEP)
        else:
            for box, pic, code, at in self.wall:
                if u >= at:
                    emerge(canvas, box, pic, code, u - at, hold=4, sweep=4, soft=12)
        w = int(s >= WALL)
        said(canvas, f, self.words[w], 62, PART["montage"][0] + w * WALL * STEP, frames=14)

    def draw_title(self, canvas, f, u):
        a = PART["title"][0]
        for t, y, at in self.title:
            said(canvas, f, t, y, a + at, frames=26)
        canvas *= 1 - noise.smoothstep(self.length - 1.4 * FPS, self.length - 0.2 * FPS, f)


def film(out=ROOT / "plates" / "_timeline.mp4"):
    t0 = time.time()
    mods = {s: importlib.import_module(f"works.{s}") for s in hanging()}
    assert {PAINTED, WOVEN, CLOSE[0], *MONTAGE} <= set(mods), "the film shows only works that hang"
    it = Film(mods)
    print(f"{it.length / FPS:.1f}s, {time.time() - t0:.0f}s", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "score.wav"
        music.write(wav, score.render())
        print(f"score: {sum(n for _, n in score.FORM)} bars, {time.time() - t0:.0f}s", flush=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a",
             # tagged as sRGB, which it is, so that Safari and QuickTime show the colours the plates have
             "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=iec61966-2-1:range=tv",
             "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-tune", "film", "-x264-params", "aq-mode=3",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], stdin=subprocess.PIPE)
        for f in range(it.length):
            enc.stdin.write(np.clip(it.frame(f) + 0.5, 0, 255).astype(np.uint8).tobytes())
            if f % BAR == 0:
                print(f"bar {f // BAR:3d} {time.time() - t0:6.0f}s", flush=True)
        enc.stdin.close()
        enc.wait()
    print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    film()
