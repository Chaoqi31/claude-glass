"""The film: a walk through the museum, cut to a score that is composed and played in code.

    uv run timeline.py      writes plates/_timeline.mp4 (1920 x 1080, 30 fps, H.264 with AAC)

The museum is one long wall. The works hang along it in the order a visitor meets them, room by room, each room
in the colour of its walls, with its name on the wall where it begins and a label beside every work. The film
opens close on the sun of one painting. Then each room slides in over the last on the first note of its music,
and the camera walks the wall slowly, never quite stopping. In a room it may cut close to a passage of paint and
draw back from it to the whole work; in the first it waits in front of a bare canvas while the painting is
painted, stroke by stroke. At the end it draws back from a last passage of paint until the whole museum is in
sight, the wall set in lines like a page, and the glass goes dark for the title.

The plates are shown as they are, in their own colours. numpy and PIL draw each frame and ffmpeg encodes the
frames with the music. The type is Iowan Old Style, from macOS.
"""

import importlib
import io
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, special

import score
from atelier import impasto, music, noise, plate
from atelier.color import to_srgb
from atelier.plate import BACKDROP
from render import ROOMS, ROOT, hanging, rooms

W, H, FPS = 1920, 1080, 30
STEP = round(score.STEP * FPS)            # frames in a sixteenth of the score
assert abs(STEP - score.STEP * FPS) < 1e-9, "a sixteenth is a whole number of frames"
BAR, HALF, TAIL = 16 * STEP, 8 * STEP, round(score.TAIL * FPS)
WIPE = 24                                 # frames a room takes to slide in over the last

DARK = np.float32([10, 10, 11])           # the dark beyond the walls
IVORY, DIM = (236, 230, 216), (161, 155, 142)     # type on a dark wall, as on the site
INK, INKDIM = (41, 37, 31), (110, 103, 92)        # and on a light one
VERMILION = (200, 64, 46)                         # the museum's one accent
SERIF = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
ROMAN, ITALIC, TITLING = 0, 2, 6                  # faces in it
NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
PAD = 16                                  # px of edge kept round every picture, for the filter reading near it

# The wall, in px of the frame at the camera's usual distance (zoom 1): x along it, y down from the middle.
HIGH = 0.58 * H                           # how high a painting hangs; a sheet of paper is smaller, and so on
SCALE = {"sheet": 0.8, "Colour Itself": 1.15}
HANG, TOP, FOOT = -0.03 * H, -0.52 * H, 0.52 * H   # the line the works hang on, and the wall's top and foot
LAMP = HANG - 0.06 * H                    # where the lights above the works are brightest
GAP, DOOR = 0.2 * W, 0.12 * W             # between works (the label in it), and a doorway
LINES, LEAD = 4, 0.22 * H                 # the wall set as a page at the end: its lines, and the dark between

# The walk. The camera walks at most SPEED px of wall a half bar, at least WAY half bars from one work to the
# next, slowing almost to a stop at each (EASE of its speed); and at some it stays on longer: HOLD half bars
# more, to read a room's name, to lean in, to go along a long work, to look close and draw back.
SPEED, WAY, EASE = 560, 2.5, 0.25
HOLD = {"text": 1, "work": 0, "lean": 3, "long": 3, "detail": 5, "painted": 8, "finale": 8}
LONG = 1.35                               # how near the camera goes along a long work
PARTS = {"Open Air": "air", "The Garden": "garden", "Paper and Water": "water", "The Workshop": "workshop",
         "Colour Itself": "colour"}       # the part of the score each room is walked to
PAINTED, AFTER = "saint_remy", "sky"      # the work seen being painted, and the part of the score after it
WORDS = ("Every painting", "is a program.")       # on the wall beside it
CODA = "One painter, many hands."                 # under the whole museum
OPENING = ("impression", (0.37, 0.25), (0.37, 0.47), 3.6)   # the work the film opens on: from, to, how near
DETAILS = {   # where the camera cuts close: the passage it cuts to and the one it drifts to, as (x, y) on the
    # plate from 0 to 1, and how near, in times the work's usual size; the last work of the walk must have one
    "irises": ((0.62, 0.30), (0.54, 0.38), 3.0),
    "jiangnan": ((0.74, 0.50), (0.64, 0.55), 3.4),
    "rose_window": ((0.50, 0.50), (0.50, 0.41), 3.4),
    "vetheuil": ((0.36, 0.40), (0.45, 0.44), 3.8),
}
LEANS = {"starry": ((0.70, 0.28), 1.45)}  # where the camera leans in on its way past, and how near

LIGHT = ((np.arange(256) / 255) ** 2.2).astype(np.float32)   # a byte of the frame as light, near enough
DARKEN = np.round(255 * np.linspace(0, 1, 4096) ** (1 / 2.2)).astype(np.float32)
HAZE = LIGHT[[96, 150, 196]] * 0.75                        # the cold light over the glass
WARM = LIGHT[[244, 128, 56]]                               # and the warm line along its rim
RADIUS, RIM = 1.3 * W, 0.84 * H                           # the curve of the glass, and where its rim tops out
Y, X = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5


def in_out(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (2 - 2 * t) ** 3 / 2


def gently(t):
    """Easing in and out with the least haste in the middle."""
    return (1 - math.cos(math.pi * min(max(t, 0.0), 1.0))) / 2


def out_expo(t):
    return 1 - 2 ** (-10 * min(max(t, 0.0), 1.0))


class Picture:
    """A picture as the film draws it: its colours and, for a sheet of paper or a letter, how much of each pixel
    it covers; kept at its own size and at halves of it, for when it is seen small."""

    def __init__(self, rgb, cover=None):
        self.h, self.w = rgb.shape[:2]
        self.sheet = cover is not None
        img = Image.fromarray(rgb)
        mask = None if cover is None else Image.fromarray(np.clip(cover * 255 + 0.5, 0, 255).astype(np.uint8))
        self.levels = []
        while True:
            pad = lambda im, mode: Image.fromarray(np.pad(np.asarray(im), ((PAD, PAD), (PAD, PAD)) + ((0, 0),) * (im.mode == "RGB"), mode=mode))
            self.levels.append((self.w / img.width, pad(img, "edge"), None if mask is None else pad(mask, "constant")))
            if min(img.size) < 200:
                break
            img, mask = img.reduce(2), None if mask is None else mask.reduce(2)


def unmounted(name):
    """A plate as the film hangs it. A sheet mounted on the museum's dark wall comes off that wall, its shadow
    with it, to be hung on another; a painting to its edges is left as it is. -> Picture"""
    a = np.asarray(Image.open(ROOT / "plates" / f"{name}.jpg").convert("RGB"))
    wall = np.float32(plate.to_srgb_255(BACKDROP))
    if np.abs(a[[0, 0, -1, -1], [0, -1, 0, -1]] - wall).max() > 12:
        return Picture(a)
    lab, _ = ndimage.label((a.max(-1) < 70) & (np.ptp(a, -1) < 14))
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    sheet = ~np.isin(lab, edge[edge > 0])
    cover = ndimage.gaussian_filter(sheet.astype(np.float32), 0.8)
    _, (iy, ix) = ndimage.distance_transform_edt(~sheet, return_indices=True)   # the paper's colour carried past its edge
    ys, xs = np.nonzero(cover > 0.02)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return Picture(np.ascontiguousarray(a[iy, ix][crop]), cover[crop])


def place(canvas, pic, x0, y0, x1, y1, alpha=1.0, clip=None):
    """Draw the whole of `pic` into the rectangle (x0, y0)-(x1, y1) of the frame, fractional, and only inside
    `clip` if given: resampled for this frame from the nearest of its sizes, its edges drawn to a fraction of a
    pixel."""
    cx0, cy0, cx1, cy1 = (x0, y0, x1, y1) if clip is None else clip
    cx0, cy0, cx1, cy1 = max(cx0, x0, 0), max(cy0, y0, 0), min(cx1, x1, W), min(cy1, y1, H)
    X0, Y0, X1, Y1 = math.floor(cx0), math.floor(cy0), math.ceil(cx1), math.ceil(cy1)
    if X1 - X0 < 1 or Y1 - Y0 < 1 or alpha <= 0.002:
        return
    k = (x1 - x0) / pic.w                                    # px of the frame to a px of the picture
    s, img, mask = next((lv for lv in reversed(pic.levels) if lv[0] * k <= 0.75), pic.levels[0])
    box = [(v - o) / k / s + PAD for v, o in ((X0, x0), (Y0, y0), (X1, x0), (Y1, y0))]
    box = tuple(min(max(v, 0), m) for v, m in zip(box, img.size * 2))
    patch = np.asarray(img.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box), np.float32)
    xs, ys = np.arange(X0, X1) + 0.5, np.arange(Y0, Y1) + 0.5
    m = (np.clip(ys - cy0 + 0.5, 0, 1) * np.clip(cy1 - ys + 0.5, 0, 1))[:, None] \
        * (np.clip(xs - cx0 + 0.5, 0, 1) * np.clip(cx1 - xs + 0.5, 0, 1))[None, :] * alpha
    if mask is not None:
        m = m * np.asarray(mask.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box), np.float32) / 255
    region = canvas[Y0:Y1, X0:X1]
    region += (patch - region) * m[..., None]


class Type:
    """A line of type, `size` px at zoom 1 in face `face`, tracked by `track` ems, drawn `scale` times as large
    for being seen nearer. w and h are its size at zoom 1, its baseline 1.1 sizes down."""

    def __init__(self, text, size, face=ROMAN, colour=IVORY, track=0.0, scale=2):
        f = ImageFont.truetype(SERIF, round(size * scale), index=face)
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

    def at(self, canvas, x, y, alpha=1.0):
        """On the frame, still, its top left at (x, y)."""
        place(canvas, self.pic, x, y, x + self.w, y + self.h, alpha)


def wrapped(text, size, face, colour, width):
    """Type set in lines no wider than `width` px. -> [Type]"""
    f, lines, cur = ImageFont.truetype(SERIF, size, index=face), [], ""
    for word in text.split():
        if cur and f.getlength(f"{cur} {word}") > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return [Type(t, size, face, colour) for t in lines + [cur]]


def after(m):
    """Who a work is after, in a few words: 'Claude Monet'."""
    who = re.split(r",|:| in | and ", m.AFTER, maxsplit=1)[0]
    return re.sub(r"^the [a-z ]+? of ", "", who, flags=re.I).strip()


def order(mods):
    """The works as a visitor meets them: room by room, oldest first in each."""
    return [s for room in ROOMS for s in sorted((s for s, m in mods.items() if m.ROOM == room), key=lambda s: mods[s].YEAR)]


def walk(wall):
    """The way through the museum: for each room, the frame it slides in on and its stops, [(slug, kind, frame
    the camera arrives)], its name on the wall first (slug None); and the parts of the score, [(part, bars)].
    Rooms begin on a bar line, and so does every stop that begins something, a cut or the painting."""
    out, parts, f = [], [("intro", 4)], 4 * BAR
    for r in wall.rooms:
        kinds = ["painted" if w.slug == PAINTED else "detail" if w.slug in DETAILS else "lean" if w.slug in LEANS
                 else "long" if w.w / w.h > 2.4 else "work" for w in r["works"]]
        if r is wall.rooms[-1]:
            assert r["works"][-1].slug in DETAILS, "the walk ends close on a passage of its last work"
            kinds[-1] = "finale"
        start, t, x, hold = f, f + HALF // 2, r["look"], HOLD["text"]
        stops = [(None, "text", t)]
        for w, k in zip(r["works"], kinds):
            to, near = (w.at((0, 0.5), LONG)[0], LONG) if k == "long" else (w.x, 1)
            way = max(WAY, abs(to - x) * near / SPEED)        # seen nearer, the wall goes by faster
            t += round((hold + way) * HALF / STEP) * STEP
            if k in ("detail", "painted", "finale"):          # these begin on a bar line: the nearest one,
                t = start + math.ceil((t - start - 0.75 * HALF) / BAR) * BAR    # going a little faster or waiting
            stops.append((w.slug, k, t))
            x, hold = w.at((1, 0.5), LONG)[0] if k == "long" else w.x, HOLD[k]
        f = t + 8 * HALF if k == "finale" else start + math.ceil((t + (hold + WAY) * HALF - start) / BAR) * BAR
        if PAINTED in (s for s, _, _ in stops):
            p = next(t for s, _, t in stops if s == PAINTED)
            parts += [(PARTS[r["name"]], (p - start) // BAR), ("paint", 4), (AFTER, (f - p) // BAR - 4)]
        else:
            parts.append((PARTS[r["name"]], (f - start) // BAR))
        out.append((r["name"], start, stops))
    return out, parts + [("burst", 1), ("title", 2), ("coda", 2)]


class Work:
    """A work on the wall: its picture, the middle of it and its size there, and its label beside it."""

    def __init__(self, slug, m, left, ink, dim):
        self.slug, self.pic = slug, unmounted(slug)
        high = HIGH * SCALE.get(m.ROOM, 1) * (SCALE["sheet"] if self.pic.sheet else 1)
        self.w = min(high * self.pic.w / self.pic.h, 0.9 * W)
        self.h = self.w * self.pic.h / self.pic.w
        self.x, self.y = left + self.w / 2, HANG
        lines = wrapped(m.TITLE, 22, ROMAN, ink, 300) + [Type(f"After {after(m)}", 17, ITALIC, dim),
                                                          Type(re.split(r"[,:;(]", m.MEDIUM)[0].strip(), 15, ROMAN, dim)]
        y, self.label = 0.0, []
        for i, t in enumerate(lines):
            y += 6 if i == len(lines) - 2 else 0
            self.label.append((t, y))
            y += t.size * 1.32
        self.label = [(t, self.x + self.w / 2 + 30, self.y + self.h / 2 - y + dy - 4) for t, dy in self.label]

    def at(self, uv, near):
        """Where the camera stands to look at the point uv of the work, (x, y) from 0 to 1, at zoom `near`:
        as close to it as it can while seeing nothing but the work."""
        out = []
        for c, size, u, half in ((self.x, self.w, uv[0], W / 2 / near), (self.y, self.h, uv[1], H / 2 / near)):
            out.append(c + float(np.clip((u - 0.5) * size, -max(size / 2 - half, 0), max(size / 2 - half, 0))))
        return out


class Wall:
    """The museum as one long wall: its rooms, each with its colour, its span along the wall and its name
    written at its start; the works on it; and the words beside the painting that is painted."""

    def __init__(self, mods):
        self.rooms, self.works, x = [], {}, 0.0
        for n, room in zip(NUMERALS, rooms(mods)):
            colour = ROOMS[room][1]
            light = sum(int(colour[k:k + 2], 16) for k in (1, 3, 5)) > 3 * 128
            ink, dim = (INK, INKDIM) if light else (IVORY, DIM)
            text = [Type(n, 64, TITLING, VERMILION, 0.08), Type(room, 112, ROMAN, ink), Type(ROOMS[room][0], 34, ITALIC, dim)]
            tx, ty, wide = x + 0.62 * W, HANG - 160, max(t.w for t in text)
            r = dict(name=room, x0=x, lin=LIGHT[list(plate.to_srgb_255(colour))], spots=[(tx + 0.2 * W, 0.4 * W)],
                     text=[(t, tx + dx, ty + dy, ("room", i)) for i, (t, dx, dy) in enumerate(zip(text, (4, -2, 2), (0, 60, 242)))],
                     look=tx + wide / 2 - 0.1 * W, works=[])         # where the camera stands to read the name
            x = tx + wide + GAP
            for s in (s for s in order(mods) if mods[s].ROOM == room):
                if s == PAINTED:                                  # the words on the wall before it
                    said = [Type(t, 60, ITALIC, ink) for t in WORDS]
                    x += 0.06 * W
                    r["text"] += [(t, x, HANG - 80 + 84 * i, ("words", i)) for i, t in enumerate(said)]
                    r["spots"].append((x + 200, 300))
                    x += max(t.w for t in said) + 110
                w = self.works[s] = Work(s, mods[s], x, ink, dim)
                r["works"].append(w)
                r["spots"].append((w.x, 0.45 * w.w + 180))
                x += w.w + GAP
            r["x1"] = x + 0.45 * W
            self.rooms.append(r)
            x = r["x1"] + DOOR
        self.start, self.end = 0.0, self.rooms[-1]["x1"]
        self.starts = np.array([r["x0"] for r in self.rooms])
        self.stops = np.array([r["x1"] for r in self.rooms])
        self.lin = np.stack([r["lin"] for r in self.rooms])

    def page(self):
        """The wall set as a page: broken between works into LINES lines of about the same length, set one under
        another. -> [(from, to, dx, dy)]: the stretch of the wall from..to drawn moved by dx, dy"""
        works = sorted(self.works.values(), key=lambda w: w.x)
        breaks = [self.start]
        for k in range(1, LINES):
            aim = self.start + (self.end - self.start) * k / LINES
            j = min(range(len(works) - 1), key=lambda j: abs(works[j].x + works[j].w / 2 + GAP / 2 - aim))
            breaks.append(works[j].x + works[j].w / 2 + GAP * 0.7)
        breaks.append(self.end)
        return [(a, b, -a, i * (FOOT - TOP + LEAD)) for i, (a, b) in enumerate(zip(breaks, breaks[1:]))]


class Key:
    """A keyframe of the camera: at frame f it looks at (x, y) of the wall at zoom z, moving on at `ease` of the
    speed through it (0 stops there). `how` is the way it came from the keyframe before: drifting along the
    wall, or drawing back or nearer about the one point of the wall that stays where it is in the frame."""

    def __init__(self, f, x, y, z, ease=1.0, how="drift"):
        self.f, self.p, self.ease, self.how = f, np.array([x, y, math.log(z)]), ease, how


class Camera:
    """The camera's way through a shot, through its keyframes, x, y and the log of the zoom each a cubic
    between them; before the first and after the last it goes on as it was. -> (x, y, zoom) at a frame"""

    def __init__(self, keys):
        self.keys, self.t = keys, np.array([k.f for k in keys], float)
        self.v = []
        for i, k in enumerate(keys):
            lo, hi = max(i - 1, 0), min(i + 1, len(keys) - 1)
            still = hi == lo or k.how == "zoom" or (i + 1 < len(keys) and keys[i + 1].how == "zoom")
            self.v.append(0 * k.p if still else k.ease * (keys[hi].p - keys[lo].p) / (self.t[hi] - self.t[lo]))

    def __call__(self, f):
        keys, t, v = self.keys, self.t, self.v
        if f <= t[0] or f >= t[-1]:
            i = 0 if f <= t[0] else -1
            q = keys[i].p + v[i] * (f - t[i])
        else:
            i = int(np.searchsorted(t, f, side="right")) - 1
            a, b, d = keys[i].p, keys[i + 1].p, t[i + 1] - t[i]
            u = (f - t[i]) / d
            if keys[i + 1].how == "zoom":
                lz = a[2] + (b[2] - a[2]) * gently(u)
                z0, z1, z = math.exp(a[2]), math.exp(b[2]), math.exp(lz)
                s = (1 / z - 1 / z0) / (1 / z1 - 1 / z0) if abs(z1 - z0) > 1e-9 else gently(u)
                q = np.array([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, lz])
            else:
                q = (2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * d * v[i] \
                    + (3 * u ** 2 - 2 * u ** 3) * b + (u ** 3 - u ** 2) * d * v[i + 1]
        return q[0], q[1], math.exp(q[2])


class Shot:
    """A stretch of the film seen by one camera, from frame `start`, coming in by `enter`: a cut, a fade up
    from the dark, or a wipe, the new room sliding in over the last; and the wall set as `lines`."""

    def __init__(self, start, keys, enter="cut", lines=None):
        self.start, self.camera, self.enter = start, Camera(keys), enter
        self.lines = lines or [(-math.inf, math.inf, 0.0, 0.0)]


def shots(path, wall):
    """The film's shots, from the walk and the wall it walks."""
    slug, a, b, near = OPENING
    w = wall.works[slug]
    out = [Shot(0, [Key(0, *w.at(a, near), near), Key(4 * BAR + WIPE, *w.at(b, near * 0.78), near * 0.78)], "fade")]
    for (room, start, stops), r in zip(path, wall.rooms):
        shot = dict(start=start, keys=[Key(stops[0][2], r["look"], 0, 1, 0.3)], enter="wipe")
        for slug, kind, t in stops[1:]:
            w, keys = wall.works[slug], shot["keys"]
            if kind in ("detail", "finale"):                    # the approach, cut short by a cut close to the paint
                keys.append(Key(t, w.x, 0, 1, EASE))
                out.append(Shot(**shot))
                a, b, near = DETAILS[slug]
                lines = wall.page() if kind == "finale" else None
                dx, dy = next((ln[2], ln[3]) for ln in lines if ln[0] <= w.x < ln[1]) if lines else (0, 0)
                (ax, ay), (bx, by) = w.at(a, near), w.at(b, near * 1.04)
                keys = [Key(t, ax + dx, ay + dy, near), Key(t + 2 * HALF, bx + dx, by + dy, near * 1.04, 0)]
                if kind == "detail":                              # and back from it to the whole work
                    keys.append(Key(t + 5 * HALF, w.x, 0, 1, how="zoom"))
                else:                                             # and back until the whole museum is in sight
                    x1, y0, y1 = max(b - a for a, b, _, _ in lines), TOP, lines[-1][3] + FOOT
                    z = min(0.86 * W / x1, 0.62 * H / (y1 - y0))
                    cx, cy = x1 / 2, (y0 + y1) / 2 + 0.06 * H / z
                    keys += [Key(t + 8 * HALF, cx, cy, z, how="zoom"), Key(t + 12 * HALF, cx, cy, z * 0.97, 0.5)]
                shot = dict(start=t, keys=keys, enter="cut", lines=lines)
            elif kind == "painted":                               # the painting with the words beside it
                left = min(x for _, x, _, cue in r["text"] if cue[0] == "words")
                mid = (left + w.x + w.w / 2) / 2
                keys += [Key(t, mid, 0.01 * H, 1.1, 0), Key(t + 7 * HALF, mid, 0.01 * H, 1.2, 0.5)]
            elif kind == "lean":
                uv, near = LEANS[slug]
                keys += [Key(t, w.x, 0, 1, 0.5), Key(t + 2 * HALF, *w.at(uv, near), near, 0.5)]
            elif kind == "long":                                  # along it from end to end
                keys += [Key(t, *w.at((0, 0.5), LONG), LONG, 0.6), Key(t + HOLD["long"] * HALF, *w.at((1, 0.5), LONG), LONG, 0.6)]
            else:
                keys.append(Key(t, w.x, 0, 1, EASE))
        out.append(Shot(**shot))
    return out


def soft(t, lo, hi, blur):
    """A step up at lo and down at hi, each blurred by `blur`."""
    return 0.5 * (special.erf((t - lo) / (blur * 1.414)) - special.erf((t - hi) / (blur * 1.414)))


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


class Film:
    """The whole film, frame by frame."""

    def __init__(self, mods):
        self.wall = Wall(mods)
        self.path, self.parts = walk(self.wall)
        self.shots = shots(self.path, self.wall)
        self.burst = sum(n for p, n in self.parts[:-3]) * BAR
        self.dark, self.title = self.burst + 15 * STEP, self.burst + BAR
        self.length = self.title + 4 * BAR + TAIL
        self.painting = next(t for _, _, stops in self.path for s, _, t in stops if s == PAINTED)
        self.stills, self.seen = None, {}
        self.name = Type("The Claude Glass", 120, ROMAN, INK, 0.01, 1)
        self.coda = Type(CODA, 50, ITALIC, IVORY, 0, 1)
        self.end = [Type("The Claude Glass", 84, ROMAN, IVORY, 0.02, 1), Type("Painted in code by Claude", 32, ITALIC, DIM, 0, 1),
                    Type("github.com/Chaoqi31/claude-glass", 20, ROMAN, DIM, 0.12, 1)]

    def picture(self, w, f):
        """The work as it is at frame f: the painted one bare, being painted, then lit. -> [(Picture, alpha)]"""
        if w.slug != PAINTED or f >= self.painting + 4 * BAR:
            return [(w.pic, 1.0)]
        if self.stills is None:
            self.stills = snapshots(PAINTED, round(0.85 * 4 * BAR), 1100)
        t = max(0.0, (f - self.painting) / (4 * BAR))
        i = round(min(1.0, t / 0.85) * (len(self.stills) - 1))
        if i not in self.seen:
            self.seen.clear()
            self.seen[i] = Picture(np.asarray(Image.open(io.BytesIO(self.stills[i])).convert("RGB")))
        return [(self.seen[i], 1.0), (w.pic, noise.smoothstep(0.88, 0.97, t))]

    @staticmethod
    def rise(f, start):
        """How far type that begins to rise into place at frame `start` has to go, in its heights, and how
        strongly it shows. -> (offset, alpha)"""
        e = out_expo((f - start) / 22)
        return (1 - e) * 0.55, min(1.0, e * 1.5)

    def band(self, canvas, cam, line, f):
        """One line of the wall, as the camera sees it: the wall, lit, the shadows of the works on it, the works,
        their labels and the writing on the wall."""
        cx, cy, z = cam
        a, b, dx, dy = line
        sx = lambda x: W / 2 + (x + dx - cx) * z
        sy = lambda y: H / 2 + (y + dy - cy) * z
        x0, x1, y0, y1 = sx(max(a, self.wall.start)), sx(min(b, self.wall.end)), sy(TOP), sy(FOOT)
        X0, X1, Y0, Y1 = max(0, math.floor(x0)), min(W, math.ceil(x1)), max(0, math.floor(y0)), min(H, math.ceil(y1))
        if X1 <= X0 or Y1 <= Y0:
            return
        xs = cx + (np.arange(X0, X1) + 0.5 - W / 2) / z - dx
        ys = cy + (np.arange(Y0, Y1) + 0.5 - H / 2) / z - dy
        lo, hi = xs[0] - W, xs[-1] + W
        i = np.clip(np.searchsorted(self.wall.starts, xs, side="right") - 1, 0, None)
        lin = np.where((xs < self.wall.stops[i])[:, None], self.wall.lin[i], LIGHT[DARK.astype(np.int32)])
        pool = np.zeros_like(xs)
        for r in self.wall.rooms:
            for c, spread in r["spots"]:
                if lo - spread < c < hi + spread:
                    pool += np.exp(-0.5 * ((xs - c) / spread) ** 2)
        lit = 0.8 + 0.2 * np.minimum(pool, 1)[None, :] * np.exp(-0.5 * ((ys - LAMP) / (0.42 * H)) ** 2)[:, None]
        works = [w for w in self.wall.works.values() if a <= w.x < b and lo < w.x < hi]
        for w in works:
            lit *= 1 - 0.5 * np.outer(soft(ys, w.y - w.h / 2 + 16, w.y + w.h / 2 + 16, 14),
                                      soft(xs, w.x - w.w / 2 + 5, w.x + w.w / 2 + 5, 14))
        rgb = DARKEN[np.clip(lin[None] * lit[..., None] * 4095 + 0.5, 0, 4095).astype(np.int32)]
        ry, rx = np.arange(Y0, Y1) + 0.5, np.arange(X0, X1) + 0.5
        cover = (np.clip(ry - y0 + 0.5, 0, 1) * np.clip(y1 - ry + 0.5, 0, 1))[:, None] \
            * (np.clip(rx - x0 + 0.5, 0, 1) * np.clip(x1 - rx + 0.5, 0, 1))[None, :]
        region = canvas[Y0:Y1, X0:X1]
        region += (rgb - region) * cover[..., None]
        for w in works:
            for pic, alpha in self.picture(w, f):
                place(canvas, pic, sx(w.x - w.w / 2), sy(w.y - w.h / 2), sx(w.x + w.w / 2), sy(w.y + w.h / 2), alpha)
            for t, lx, ly in w.label:                         # too small to read from far off
                place(canvas, t.pic, sx(lx), sy(ly), sx(lx + t.w), sy(ly + t.h), noise.smoothstep(0.25, 0.45, z))
        for k, r in enumerate(self.wall.rooms):
            for t, tx, ty, (what, j) in r["text"]:
                if not (a <= tx < b and lo < tx < hi):
                    continue
                off, alpha = self.rise(f, self.path[k][1] - 6 + 7 * j if what == "room" else self.painting + HALF + 10 * j)
                box = (sx(tx), sy(ty), sx(tx + t.w), sy(ty + t.h))
                place(canvas, t.pic, box[0], sy(ty + off * t.h), box[2], sy(ty + (1 + off) * t.h), alpha, clip=box)

    def view(self, shot, f):
        """What the camera of `shot` sees at frame f, blurred along the way it is moving."""
        cx, cy, z = cam = shot.camera(f)
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = DARK
        for line in shot.lines:
            self.band(canvas, cam, line, f)
        v = abs(cx - shot.camera(f - 1)[0]) * z
        if v > 3:
            canvas = ndimage.uniform_filter1d(canvas, int(v * 0.5) | 1, axis=1)
        return canvas

    def frame(self, f):
        if f >= self.title:                                   # the title and the coda over the glass
            u = f - self.title
            rise = noise.smoothstep(0, 4 * BAR + TAIL, u)
            canvas = glass(RIM - 0.1 * H * rise, 1.0, 0.3 + 0.9 * noise.smoothstep(BAR, 4 * BAR, u))
            y = 0.33 * H - 0.05 * H * rise
            for t, at in zip(self.end, (0, 2 * BAR, 3 * BAR)):
                t.at(canvas, round((W - t.w) / 2), round(y), noise.smoothstep(at, at + (6 if at == 0 else 20), u))
                y += t.h + 18
            return canvas * (1 - noise.smoothstep(self.length - 1.4 * FPS, self.length - 0.2 * FPS, f))
        if f >= self.dark:
            return np.broadcast_to(DARK, (H, W, 3)).copy()
        j = max(i for i, s in enumerate(self.shots) if s.start <= f)
        k = next((i for i, s in enumerate(self.shots) if s.enter == "wipe" and abs(f - s.start + 0.5) < WIPE / 2), None)
        if k is None:
            canvas = self.view(self.shots[j], f)
        else:                                                 # the next room sliding in from the right
            u = (f - self.shots[k].start + WIPE / 2) / WIPE
            edge, speed = W * (1 - in_out(u)), W * abs(in_out(u + 0.5 / WIPE) - in_out(u - 0.5 / WIPE))
            old, new = self.view(self.shots[k - 1], f), self.view(self.shots[k], f)
            new = np.roll(new, int(0.25 * edge), axis=1)
            xs = np.arange(W) + 0.5
            m = np.clip((xs - edge) / max(1.0, 0.5 * speed) + 0.5, 0, 1)
            old *= 1 - 0.45 * noise.smoothstep(0, 0.25, u) * np.exp(-np.maximum(edge - xs, 0) / 60)[None, :, None]
            canvas = old + (new - old) * m[None, :, None]
        if f < BAR * 4:                                       # the name of the museum over the first passage
            self.said(canvas, f, self.name, 0.42 * H, BAR, 1 - noise.smoothstep(3.3 * BAR, 3.7 * BAR, f))
        if f >= self.burst - HALF:
            self.said(canvas, f, self.coda, 0.83 * H, self.burst - HALF)
        return canvas * noise.smoothstep(0, 12, f)            # out of the dark on the first note

    def said(self, canvas, f, t, y, begins, fade=1.0):
        """Type t across the middle of the frame with its top at y, rising into place from frame `begins`."""
        off, alpha = self.rise(f, begins)
        x = (W - t.w) / 2
        place(canvas, t.pic, x, y + off * t.h, x + t.w, y + (1 + off) * t.h, alpha * fade, clip=(x, y, x + t.w, y + t.h))


def film(out=ROOT / "plates" / "_timeline.mp4"):
    t0 = time.time()
    mods = {s: importlib.import_module(f"works.{s}") for s in hanging()}
    it = Film(mods)
    print(f"{len(it.shots)} shots, {it.length / FPS:.1f}s, {time.time() - t0:.0f}s", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "score.wav"
        music.write(wav, score.render(it.parts))
        print(f"score: {sum(n for _, n in it.parts)} bars, {time.time() - t0:.0f}s", flush=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a",
             # tagged as sRGB, which it is, so that Safari and QuickTime show the colours the plates have
             "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=iec61966-2-1:range=tv",
             "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-tune", "film", "-x264-params", "aq-mode=3",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], stdin=subprocess.PIPE)
        for f in range(it.length):
            enc.stdin.write(np.clip(it.frame(f) + 0.5, 0, 255).astype(np.uint8).tobytes())
            if f % BAR == 0:
                print(f"bar {f // BAR:3d} {time.time() - t0:6.0f}s", flush=True)
        enc.stdin.close()
        enc.wait()
    print(f"{out.name}: {len(mods)} works, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    film()
