"""The timeline film: every work in the collection, oldest first, as a slide projector throws it in a dark room.

    uv run timeline.py      writes plates/_timeline.mp4 (1280 x 720, 24 fps, H.264)

It opens cold, on the brushwork of a few works seen close, changed as fast as a hand can change slides; then
the title, and the works in the order of the years they are painted after. The camera moves differently from
one work to the next: it pushes in toward the busiest part of a work, or starts close on it and draws back to
the whole, and it travels the length of a long or a tall painting. Most works dissolve into the next, as one
projector fades up while a second fades down; now and then the gate goes dark while a slide is changed. Under
the gate run a caption (date, place, title) and a timeline scaled by the log of the years before now, so that
the old centuries are compressed. It ends on the whole collection, hung on one wall.

numpy and PIL make the stills, the type and the light of the gate (atelier/projector.py, frozen for an average
frame); ffmpeg does the per-frame work, one process per shot, and the frames come back through Python, which
lays the dissolves, on their way to one encoder. The type is Iowan Old Style, the face the site is set in:
/System/Library/Fonts/Supplemental/Iowan Old Style.ttc (macOS).
"""

import importlib
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from atelier import noise, plate, projector
from atelier.color import lin, to_srgb
from atelier.plate import BACKDROP
from render import ROOT, SCROLL, hanging

W, H, FPS = 1280, 720, 24
GW, GH = 736, 552                    # the gate: 4:3
GX, GY = (W - GW) // 2, 40
K = GH / 600                         # projector.py's lengths (softness, burn, weave) are px of a 600 px gate
DARK = "#090807"                     # the room
LINE = 676                           # the timeline
FONT = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
IVORY = (236, 230, 216)              # the site's --ivory
SHOT = {"push": 4.4, "reveal": 5.2, "macro": 0.75}  # seconds on a work, by how the camera moves
ZOOM = {"push": 1.12, "reveal": 2.6}  # how far the camera pushes in, or how close it starts before drawing back
DRIFT = 0.15                          # how far off the middle a push-in may start
FADE, BEAT, MIX = 0.4, 8, 16          # seconds a picture takes to come and go; frames of dark; frames of a dissolve
OPEN = 9                              # works seen close before the title
FLICKER = "0.009*(sin(1.93*n)+sin(5.21*n+1)+sin(11.7*n+2))"
TICKS = [(-50000, "50,000 BCE"), (-10000, "10,000 BCE"), (-3000, "3000 BCE"), (-500, "500 BCE"), (1, "1 CE"),
         (500, "500"), (1000, "1000"), (1500, "1500"), (1800, "1800"), (1900, "1900"), (2026, "2026")]
LIGHT = ((np.arange(256) / 255) ** 2.2).astype(np.float32)  # an 8-bit frame as light, near enough


def when(m):
    """A work's date as the caption gives it: 'c. 43,500 BCE', 'c. 1530', '1938'."""
    if m.YEAR < 0:
        return f"c. {-m.YEAR:,} BCE" if m.YEAR <= -10000 else f"c. {-m.YEAR} BCE"
    exact = re.search(rf"(?<!c\. )(?<!–)\b{m.YEAR}\b", m.AFTER)  # only where the source names that year alone
    return f"{'' if exact else 'c. '}{m.YEAR}{' CE' if m.YEAR < 1000 else ''}"


def along(year, start):
    """Where a year falls on the timeline: 0 at its start, 1 now, by the log of (a century and) the years
    before now."""
    t = lambda y: math.log(2126 - y)
    return (t(start) - t(year)) / (t(start) - t(2026))


def png(a, path):
    """Linear light to a dithered sRGB PNG."""
    s = to_srgb(a) * 255 + noise.rng(0).uniform(0, 1, a.shape)
    Image.fromarray(np.clip(s, 0, 255).astype(np.uint8)).save(path)


def gate():
    """The light of the gate, fixed on the screen while the film moves through it: projector.project frozen
    for an average frame, as a mask the picture is multiplied by and the room, lit only by the light
    scattered round the gate, which is screened over it."""
    d = projector.aperture(GH, GW, GX)[GX - GY:GX - GY + H]  # aperture() pads evenly; the gate sits high
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy, xx = yy + 0.5 - GY - GH / 2, xx + 0.5 - W / 2
    ap = noise.smoothstep(5 * K, -5 * K, d)
    hot = ((yy + 0.03 * GH) ** 2 + (xx - 0.04 * GW) ** 2) / ((GH / 2) ** 2 + (GW / 2) ** 2)
    fall = (np.exp(-1.9 * hot) * (1 - 0.5 * np.exp(np.minimum(d, 0) / (90 * K)))
            * (1 + 0.03 * noise.field(hot.shape, GW / 3, noise.rng(16))))
    mask = ap[..., None] * projector.LAMP * fall[..., None] ** np.array([0.7, 1.0, 1.45], np.float32)
    light = 0.25 * mask
    halo = ndimage.gaussian_filter(light, (14 * K, 14 * K, 0), truncate=2.5) * (1 - ap[..., None])
    veil = ndimage.gaussian_filter(ap, 28 * K, truncate=2.5)[..., None] * light.sum((0, 1)) / ap.sum()
    return mask, lin(DARK) + 0.2 * halo + 0.03 * veil


def words(lines):
    """Ivory type on a clear frame, cropped to the ink: lines of (x, baseline, size, runs), runs of
    (text, italic, alpha) set a little apart; x None centres the line. Returns the image and its place."""
    a = Image.new("L", (W, H))
    d = ImageDraw.Draw(a)
    for x, y, size, runs in lines:
        fonts = [ImageFont.truetype(FONT, size, index=2 if italic else 0) for _, italic, _ in runs]
        widths = [d.textlength(t, font=f) for (t, _, _), f in zip(runs, fonts)]
        if x is None:
            x = (W - sum(widths) - 1.4 * size * (len(runs) - 1)) / 2
        for (t, _, alpha), f, w in zip(runs, fonts, widths):
            d.text((x, y), t, font=f, fill=round(255 * alpha), anchor="ls")
            x += w + 1.4 * size
    box = a.getbbox()
    im = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), IVORY)
    im.putalpha(a.crop(box))
    return im, box[:2]


def timeline(room, start):
    """The room with the timeline under the gate: a hairline from `start` to now, the ages ticked and named."""
    a = Image.new("L", (W, H))
    d = ImageDraw.Draw(a)
    d.line([(GX, LINE), (GX + GW, LINE)], fill=64)
    f = ImageFont.truetype(FONT, 12)
    for year, text in TICKS:
        if year < start:
            continue
        x = round(GX + GW * along(year, start))
        d.line([(x, LINE - 2), (x, LINE + 2)], fill=96)
        d.text((x, LINE + 18), text, font=f, fill=104, anchor="ms")
    im = Image.open(room)
    im.paste(Image.new("RGB", (W, H), IVORY), mask=a)
    return im


def mark():
    """The dot that rides the timeline."""
    a = Image.new("L", (72, 72))
    ImageDraw.Draw(a).ellipse((15, 15, 57, 57), fill=240)
    im = Image.new("RGBA", (9, 9), IVORY)
    im.putalpha(a.resize((9, 9), Image.LANCZOS))
    return im


def camera(im, kind):
    """Where the camera looks on a plate: the window, (centre, width) in px of the plate, at the start and
    the end of its move, inside any wall showing round the work. A long or a tall painting is travelled from
    end to end. Otherwise the camera pushes in toward the busiest part of the work, or starts close on it and
    draws back to the whole, or, for `macro`, looks at it at the plate's own resolution, drifting a little."""
    g = np.asarray(im.resize((im.width // 16, im.height // 16), Image.BOX), np.float32)
    wall = (np.abs(g - plate.to_srgb_255(BACKDROP)).max(-1) <= 12) & (np.ptp(g, -1) <= 6)
    y0, y1, x0, x1 = 0, len(g), 0, len(g[0])
    for _ in range(3):  # trim rows and columns that are mostly wall, from the outside in
        r = np.flatnonzero(wall[y0:y1, x0:x1].mean(1) <= 0.5)
        if r.size:
            y0, y1 = y0 + r[0], y0 + r[-1] + 1
        c = np.flatnonzero(wall[y0:y1, x0:x1].mean(0) <= 0.5)
        if c.size:
            x0, x1 = x0 + c[0], x0 + c[-1] + 1
    lo, hi = 16 * np.array((x0, y0)), 16 * np.array((x1, y1))
    w0 = min(hi[0] - lo[0], (hi[1] - lo[1]) * 4 / 3)
    keep = lambda c, w: np.clip(c, lo + (w / 2, w * 3 / 8), hi - (w / 2, w * 3 / 8))
    span = hi - lo
    if kind != "macro" and (span[0] > SCROLL * span[1] or span[1] * 4 / 3 > 1.2 * span[0]):
        return (keep(lo, w0), w0), (keep(hi, w0), w0)
    g = g[y0:y1, x0:x1].mean(-1)
    busy = ndimage.gaussian_filter(ndimage.gaussian_gradient_magnitude(g, 1), max(g.shape) / 20)
    v, u = np.mgrid[0:1:g.shape[0] * 1j, 0:1:g.shape[1] * 1j]
    fy, fx = np.unravel_index((busy * np.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.08)).argmax(), g.shape)
    focus = lo + 16 * np.array((fx + 0.5, fy + 0.5))
    if kind == "macro":
        c = keep(focus, GW)
        return (keep(c + (0.08 * GW, 0.03 * GW), 1.06 * GW), 1.06 * GW), (c, GW)
    w1 = w0 / ZOOM[kind]
    c1 = keep(focus, w1)
    if kind == "reveal":
        return (c1, w1), (keep((lo + hi) / 2, w0), w0)
    c0 = keep(c1 - np.clip(c1 - (lo + hi) / 2, -DRIFT * w0, DRIFT * w0), w0)
    return (c0, w0), (c1, w1)


def still(src, kind, path):
    """The plate round the path of the camera, cut to the gate's shape at a little over the gate's
    resolution, with the lens's softness and halo. Returns the window at the start and the end of the
    move, (x, y, width) in px of the still."""
    im = Image.open(src).convert("RGB")
    pw, ph = im.size
    (c0, w0), (c1, w1) = camera(im, kind)
    lo = np.minimum(c0 - (w0 / 2, w0 * 3 / 8), c1 - (w1 / 2, w1 * 3 / 8))
    hi = np.maximum(c0 + (w0 / 2, w0 * 3 / 8), c1 + (w1 / 2, w1 * 3 / 8))
    cw = max(hi[0] - lo[0], (hi[1] - lo[1]) * 4 / 3)
    corner = (lo + hi) / 2 - (cw / 2, cw * 3 / 8)
    s = 1.25 * GW / min(w0, w1)  # px of the still to a px of the plate
    sheet = Image.new("RGB", (round(cw * s), round(cw * s * 3 / 4)), BACKDROP)
    sheet.paste(im.resize((round(pw * s), round(ph * s)), Image.LANCZOS), tuple(np.round(-corner * s).astype(int)))
    px = 1.25  # px of the still to a px of the gate, where the camera is closest
    img = lin(np.asarray(sheet))
    img = (0.8 * ndimage.gaussian_filter(img, (K * px, K * px, 0))
           + 0.2 * ndimage.gaussian_filter(img, (14 * K * px, 14 * K * px, 0), truncate=2.5))
    png(img, path)
    return [(*((c - (w / 2, w * 3 / 8) - corner) * s), w * s) for c, w in ((c0, w0), (c1, w1))]


def move(a, b, n):
    """ffmpeg's perspective filter carrying a window from a to b, (x, y, width), over n frames: eased, and
    weaving in the gate as film does."""
    u = f"(in/{n - 1})"
    p = f"(0.5*{u}+0.5*{u}*{u}*(3-2*{u}))"
    wv = 0.7 * K * max(a[2], b[2]) / GW
    x = f"({a[0]:.2f}{b[0] - a[0]:+.2f}*{p}{wv:+.3f}*(0.6*sin(2.31*in)+0.5*sin(5.77*in+1.3)))"
    y = f"({a[1]:.2f}{b[1] - a[1]:+.2f}*{p}{0.6 * wv:+.3f}*(0.6*sin(3.17*in+0.4)+0.5*sin(7.03*in+2.1)))"
    w = f"({a[2]:.2f}{b[2] - a[2]:+.2f}*{p})"
    return (f"perspective=x0={x}:y0={y}:x1={x}+{w}:y1={y}:x2={x}:y2={y}+0.75*{w}:x3={x}+{w}:y3={y}+0.75*{w}"
            ":interpolation=cubic:eval=frame")


def loop(path):
    return ["-loop", "1", "-framerate", str(FPS), "-i", str(path)]


def film(slugs=None, out=ROOT / "plates" / "_timeline.mp4"):
    t0 = time.time()
    mods = {s: importlib.import_module(f"works.{s}") for s in slugs or hanging()}
    order = sorted(mods, key=lambda s: mods[s].YEAR)
    entrance = (ROOT / "museum" / "entrance.md").read_text()
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                            "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-tune", "film",
                            "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    held = []  # the last frames of the shot before, kept back for this one to dissolve over

    def shot(inputs, graph, frames, hold=0):
        """One shot through ffmpeg, into the film. Its first frames dissolve over any held back from the shot
        before; its last `hold` are held back in turn."""
        nonlocal held
        p = subprocess.Popen(["ffmpeg", "-v", "error", *inputs, "-filter_complex", graph, "-frames:v", str(frames),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        under, held = held, []
        for k in range(frames):
            f = np.frombuffer(p.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3)
            if k < len(under):  # two projectors on one screen: their light adds
                t = (k + 1) / (len(under) + 1)
                f = np.round(255 * ((1 - t) * LIGHT[under[k]] + t * LIGHT[f]) ** (1 / 2.2)).astype(np.uint8)
            if k >= frames - hold:
                held.append(f)
            else:
                enc.stdin.write(f.tobytes())
        if p.wait():
            sys.exit(f"ffmpeg failed: {graph[:200]}")

    def card(lines, seconds):
        im, (x, y) = words(lines)
        im.save(tmp / "card.png")
        shot(loop(tmp / "card.png"), f"color=c={DARK.replace('#', '0x')}:s={W}x{H}:r={FPS}[bg];"
             f"[0]fade=in:st=0.8:d=1.2:alpha=1,fade=out:st={seconds - 2}:d=1.2:alpha=1[t];"
             f"[bg][t]overlay={x}:{y}:format=auto,format=rgb24", round(seconds * FPS))

    def dark(x=None, frames=BEAT):
        """A moment of the dark room; the timeline shows, its mark at x, while a work is coming."""
        if x is None:
            shot(loop(tmp / "room.png"), "format=rgb24", frames)
        else:
            shot([*loop(tmp / "line.png"), *loop(tmp / "mark.png")],
                 f"[0][1]overlay={x - 4}:{LINE - 4}:format=auto,format=rgb24", frames)

    def picture(src, kind, room, seconds=None, fade=(True, True), hold=0, extra=(), tail=None):
        """A plate in the gate, the camera moving over it, on the room, and whatever the `extra` inputs and
        `tail(seconds)`, the end of the graph, lay over it. Returns the shot's length in seconds."""
        a, b = still(src, kind, tmp / "still.png")
        sec = seconds or SHOT[kind] * max(1, math.hypot(b[0] - a[0], b[1] - a[1]) / a[2])  # a gate a shot
        n = round(sec * FPS)
        fades = "".join([",fade=in:d=%s" % FADE if fade[0] else "",
                         f",fade=out:st={sec - FADE:.3f}:d={FADE}" if fade[1] else ""])
        shot([*loop(tmp / "still.png"), *loop(tmp / "mask.png"), *loop(room), *extra],
             f"[0]{move(a, b, n)},scale={GW}:{GH}{fades},format=yuv444p,"
             f"eq=contrast=1+{FLICKER}:brightness=0.437*{FLICKER}:eval=frame,noise=c0s=3:c0f=t,"
             f"format=gbrp,pad={W}:{H}:{GX}:{GY}[p];[1]format=gbrp[m];[p][m]blend=all_mode=multiply[a];"
             f"[2]format=gbrp[r];[a][r]blend=all_mode=screen{tail(sec) if tail else ''},format=rgb24", n, hold)
        return sec

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        start = max(y for y, _ in TICKS if y <= mods[order[0]].YEAR)  # the last tick before the oldest work
        mask, room = gate()
        png(mask, tmp / "mask.png")
        png(room, tmp / "room.png")
        timeline(tmp / "room.png", start).save(tmp / "line.png")
        mark().save(tmp / "mark.png")
        plates = lambda s: ROOT / "plates" / f"{s}.jpg"

        def grain(s):
            """How much fine detail a plate shows where the camera looks closest at it."""
            im = Image.open(plates(s)).convert("RGB")
            _, (c, w) = camera(im, "macro")
            g = np.asarray(im.convert("L").crop((*np.int32(c - (w / 2, w * 3 / 8)), *np.int32(c + (w / 2, w * 3 / 8)))),
                           np.float32)
            return (g - ndimage.gaussian_filter(g, 3)).std()

        for s in [max(part, key=grain) for part in np.array_split(order, OPEN)]:  # cold: the paint itself, cut hard
            picture(plates(s), "macro", tmp / "room.png", fade=(False, False))
        dark(frames=12)
        card([(None, 352, 44, [(re.search(r"^# (.+)$", entrance, re.M)[1], False, 0.95)]),
              (None, 394, 20, [(re.search(r"^\*(.+)\*$", entrance, re.M)[1], True, 0.7)])], 5.5)
        dark()
        xa, ease = GX, r"min(t/1.6\,1)"
        for i, s in enumerate(order):
            m = mods[s]
            lit = i % 3 != 0, (i + 1) % 3 != 0 and i < len(order) - 1  # dissolving in, dissolving out
            cap, (cx, cy) = words([(GX, GY + GH + 36, 18, [(when(m), False, 0.95), (m.PLACE, False, 0.62),
                                                           (m.TITLE, True, 0.95)])])
            cap.save(tmp / "caption.png")
            xb = round(GX + GW * along(m.YEAR, start))
            # the caption comes up after the picture and goes before it
            sec = picture(plates(s), ("push", "reveal")[i % 2], tmp / "line.png",
                          fade=(not lit[0], not lit[1]), hold=MIX if lit[1] else 0,
                          extra=[*loop(tmp / "caption.png"), *loop(tmp / "mark.png")],
                          tail=lambda sec: f"[b];[3]fade=in:st=0.6:d=0.8:alpha=1,fade=out:st={sec - 1.2:.3f}:d=0.6:alpha=1[c];"
                          f"[b][c]overlay={cx}:{cy}:format=auto[d];"
                          f"[d][4]overlay=x='{xa - 4}+{xb - xa}*(3-2*{ease})*{ease}*{ease}':y={LINE - 4}:format=auto")
            if not lit[1]:
                dark(xb if i < len(order) - 1 else None)
            xa = xb
            print(f"{i + 1:3d}/{len(order)} {s:16s} {when(m):14s} {sec:4.1f}s {time.time() - t0:6.0f}s", flush=True)
        picture(ROOT / "plates" / "_wall.jpg", "reveal", tmp / "room.png", seconds=7.5)  # all of it, on one wall
        dark(frames=12)
        card([(None, 368, 20, [("Painted in code by Claude (Opus 5.5) · 2026", False, 0.9)])], 6.0)
    enc.stdin.close()
    enc.wait()
    print(f"{out.name}: {len(order)} works, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    film()
