"""Hang the museum: the wall, the website (index.html and site/) and the README.

    uv run site.py

GitHub Pages serves the repository root, so the viewer loads full plates straight from plates/. The README
shows the site's thumbnails, each linked to its full plate, so that GitHub can open it quickly. On the site
every room has walls of its own colour, and the page takes it on as a visitor walks in.
"""

import hashlib
import html
import importlib
import itertools
import re
import shutil
import subprocess
from string import Template

from PIL import Image, ImageDraw

from atelier import plate
from atelier.plate import BACKDROP
from render import FRONT, PAPER, ROOMS, ROOT, SCROLL, hanging, rooms, strapline, tally

SITE = ROOT / "site"
URL = "https://chaoqi31.github.io/claude-glass/"  # where GitHub Pages serves it; social cards need absolute links
ACCENT = "#c8402e"  # vermilion: the museum's one accent
ICON = "rose_window"  # a round window, cut out for the browser tab
NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
REGIONS = ["Africa", "West Asia", "South Asia", "East Asia", "Southeast Asia", "Europe", "Americas", "Oceania"]
e = html.escape


def colours(wall):
    """A wall's colours: the wall, and type that reads on it (ivory on a dark wall, ink on a light one) with its
    quieter tone."""
    light = sum(int(wall[k:k + 2], 16) for k in (1, 3, 5)) > 3 * 128
    return (wall, "#29251f", "#6e675c") if light else (wall, "#ece6d8", "#a19b8e")


def walls(wall):
    """A room's colours, for the page to take on as a visitor walks in."""
    return 'data-wall="{}" data-ink="{}" data-dim="{}"'.format(*colours(wall))


def wall(slugs, width=3200, gap=28, aspect=0.75):
    """The whole collection hung on one wall, oldest first, in rows of even height, the wall about
    `aspect` as tall as it is wide: k rows of a total aspect S stand about W*k*k/S tall."""
    mods = {s: importlib.import_module(f"works.{s}") for s in slugs}
    ims = [Image.open(ROOT / "plates" / f"{s}.jpg") for s in sorted(mods, key=lambda s: mods[s].YEAR)]
    asp = [i.width / i.height for i in ims]
    k = max(1, round((sum(asp) * aspect) ** 0.5))
    # split the sequence into k rows whose widths come out as even as possible
    target = sum(asp) / k
    best = {(0, 0): (0.0, [])}
    for j in range(1, len(ims) + 1):
        for rows in range(1, k + 1):
            cands = [(best[(i, rows - 1)][0] + (sum(asp[i:j]) - target) ** 2, best[(i, rows - 1)][1] + [(i, j)])
                     for i in range(j) if (i, rows - 1) in best]
            if cands:
                best[(j, rows)] = min(cands)
    split = best[(len(ims), k)][1]
    lines = [(ims[i:j], int((width - gap * (j - i + 1)) / sum(asp[i:j]))) for i, j in split]
    H = sum(h for _, h in lines) + gap * (len(lines) + 1)
    out = Image.new("RGB", (width, H), tuple(int(c) for c in plate.to_srgb_255(BACKDROP)))
    y = gap
    for row, h in lines:
        x = gap
        for im in row:
            w = int(im.width / im.height * h)
            out.paste(im.resize((w, h), Image.LANCZOS), (x, y))
            x += w + gap
        y += h + gap
    out.save(ROOT / "plates" / "_wall.jpg", quality=90, subsampling=0, optimize=True, progressive=True)
    print("plates/_wall.jpg", out.size)


def label(slug, m):
    """A work's wall label in the README, under its thumbnail."""
    lines = len((ROOT / "works" / f"{slug}.py").read_text().splitlines())
    return "\n".join([
        f'<p align="center"><a href="plates/{slug}.jpg"><img src="site/thumbs/{slug}.jpg" alt="{m.TITLE}" '
        'width="100%"></a></p>',
        "",
        f"**{m.TITLE}**, {m.DATE}  ",
        f"{m.MEDIUM}  ",
        f"<sub>After {m.AFTER}</sub>",
        "",
        f"> {m.NOTE}",
        "",
        f"<sub>[`works/{slug}.py`](works/{slug}.py) · {lines} lines</sub>",
        "",
    ])


def readme(slugs):
    mods = {s: importlib.import_module(f"works.{s}") for s in slugs}
    walk = rooms(mods)
    out = [Template((ROOT / "museum" / "entrance.md").read_text()).substitute(tally=tally(slugs)).rstrip(), ""]
    for n, room in zip(NUMERALS, walk):
        hung = sorted((s for s, m in mods.items() if m.ROOM == room), key=lambda s: mods[s].YEAR)
        out += [f"## {n} · {room}", "", f"<sub>{ROOMS[room][0]}</sub>", ""]
        out += [label(s, mods[s]) for s in hung]
    out += [(ROOT / "museum" / "colophon.md").read_text().rstrip(), ""]
    (ROOT / "README.md").write_text("\n".join(out))
    print("README.md:", len(mods), "works in", len(walk), "rooms")
    count = {g: sum(m.REGION == g for m in mods.values()) for g in REGIONS}
    print("by region:", " · ".join(f"{g} {n}" for g, n in count.items()))


def tone(im):
    """A picture's colour on the whole, to stand in for it until it has loaded."""
    return "#{:02x}{:02x}{:02x}".format(*im.convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0)))


def thumb(name):
    """Write site/thumbs/<name>.jpg; return the plate's size, the thumbnail's, and its tone."""
    with Image.open(ROOT / "plates" / f"{name}.jpg") as im:
        size = im.size
        im.thumbnail((1400, 1400), Image.LANCZOS)
        im.save(SITE / "thumbs" / f"{name}.jpg", quality=85, subsampling=0, optimize=True, progressive=True)
        return size, im.size, tone(im)


def icon():
    """The icon on the browser tab: the middle of the round window, cut out round."""
    with Image.open(ROOT / "plates" / f"{ICON}.jpg") as im:
        d = min(im.size)
        im = im.convert("RGB").crop(((im.width - d) // 2, (im.height - d) // 2, (im.width + d) // 2,
                                     (im.height + d) // 2)).resize((180, 180), Image.LANCZOS)
    cut = Image.new("L", (720, 720))
    ImageDraw.Draw(cut).ellipse((6, 6, 714, 714), fill=255)
    im.putalpha(cut.resize((180, 180), Image.LANCZOS))
    im.save(SITE / "icon.png", optimize=True)


def hero():
    """The work at the door, large enough for half a sharp screen (smaller screens take its thumbnail); return
    its size and tone."""
    with Image.open(ROOT / "plates" / f"{FRONT}.jpg") as im:
        im.thumbnail((2000, 2000), Image.LANCZOS)
        im.save(SITE / "hero.jpg", quality=82, subsampling=0, optimize=True, progressive=True)
        return im.size, tone(im)


def link(m):
    text, href = m.groups()
    if "://" not in href and not (ROOT / href).is_file():
        return text  # a static host cannot list a folder, so only files get a link
    return f'<a href="{href}">{text}</a>'


def inline(s):
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", e(s, quote=False))
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return re.sub(r"\[(.+?)\]\((.+?)\)", link, s)


def markdown(name):
    """The little Markdown of museum/<name>, as HTML blocks. Raw HTML and rules belong to the README."""
    out = []
    for b in re.split(r"\n\n+", (ROOT / "museum" / name).read_text().strip()):
        if b.startswith("```"):
            out.append(f"<pre><code>{e(b.partition(chr(10))[2].removesuffix('```').rstrip())}</code></pre>")
        elif b.startswith("#"):
            n = len(b) - len(b.lstrip("#"))
            out.append(f"<h{n}>{inline(b[n:].strip())}</h{n}>")
        elif not b.startswith("<") and b != "---":
            out.append(f"<p>{inline(b)}</p>")
    return out


def anchor(room):
    return re.sub(r"\W+", "-", room.lower())


def after(m):
    """Whom a work is after, as its label says it: the source's first clause on the wall, and all of it in the
    viewer. -> (short, full)"""
    full = m.AFTER[0].lower() + m.AFTER[1:] if m.AFTER.startswith("The ") else m.AFTER
    return re.sub(r"\s*\([^)]*$", "", re.split(r"[:;]", full)[0]), full


def figure(slug, m, size):
    (w, h), (tw, th), shade = size
    lines = len((ROOT / "works" / f"{slug}.py").read_text().splitlines())
    short, full = after(m)
    source = f"<p>After {e(full)}</p>" if short == full else \
        f'<p class="short">After {e(short)}</p>\n<p class="full">After {e(full)}</p>'
    return f"""<figure class="work{' long' if w / h > SCROLL else ''}" style="--a:{w / h:.4f}">
<a class="plate" href="plates/{slug}.jpg" data-slug="{slug}" data-lines="{lines}" data-w="{w}" data-h="{h}"><img src="site/thumbs/{slug}.jpg" width="{tw}" height="{th}" alt="{e(m.TITLE)}" loading="lazy" decoding="async" style="background:{shade}"></a>
<figcaption class="label">
<h3><cite>{e(m.TITLE)}</cite>, {e(m.DATE)}</h3>
<p>{e(m.MEDIUM)}</p>
{source}
<p class="note">{e(m.NOTE)}</p>
<p class="source"><a href="works/{slug}.py">works/{slug}.py</a> · {lines} lines</p>
</figcaption>
</figure>"""


def hang(slugs, size):
    """A room's works in rows: one beside its label, now on the left and now on the right, then two side
    by side; a long painting has a row to itself. -> [(kind, slugs)]"""
    long = lambda s: size[s][0][0] / size[s][0][1] > SCROLL
    rows, i = [], 0
    while i < len(slugs):
        kind = "long" if long(slugs[i]) else ("left", "right", "pair")[len(rows) % 3]
        if kind == "pair" and (i + 1 == len(slugs) or long(slugs[i + 1])):
            kind = "left"
        n = 2 if kind == "pair" else 1
        rows.append((kind, slugs[i:i + n]))
        i += n
    return rows


def room(n, name, hung, size):
    rows = []
    for kind, row in hang(sorted(hung, key=lambda s: hung[s].YEAR), size):
        across = sum(size[s][0][0] / size[s][0][1] for s in row)
        rows.append(f'<div class="row {kind}" style="--sum:{across:.4f}">\n'
                    + "\n".join(figure(s, hung[s], size[s]) for s in row) + "\n</div>")
    nl = "\n"
    return f"""<section class="room" id="{anchor(name)}" data-room="{n} · {e(name)}" {walls(ROOMS[name][1])}>
<header><h2><span class="numeral">{n}</span> {e(name)}</h2><p>{e(ROOMS[name][0])}</p></header>
{nl.join(rows)}
</section>"""


def mini(slug):
    """A small copy of a work, for the plan of the rooms; return its aspect."""
    with Image.open(ROOT / "plates" / f"{slug}.jpg") as im:
        im.thumbnail((640, 180), Image.LANCZOS)
        im.save(SITE / "mini" / f"{slug}.jpg", quality=84, optimize=True, progressive=True)
        return im.width / im.height


def door(n, name, hung, size):
    """A room on the plan: its numeral, name, line and count, and its first works hung small along a stretch of
    its wall, as many as come nearest to five of their heights across."""
    here = sorted((s for s, m in hung.items() if m.ROOM == name), key=lambda s: hung[s].YEAR)
    across = list(itertools.accumulate(size[s][0][0] / size[s][0][1] for s in here))
    shown = here[:1 + min(range(len(here)), key=lambda i: abs(across[i] - 5))]
    hang = "".join(f'<img src="site/mini/{s}.jpg" alt="" loading="lazy" decoding="async" style="--a:{mini(s):.4f};'
                   f'background:{size[s][2]}">' for s in shown)
    return (f'<li><a href="#{anchor(name)}"><span class="numeral">{n}</span><span class="name">{e(name)}</span>'
            f'<span class="about">{e(ROOMS[name][0])}</span><span class="count">{len(here)} '
            f'work{"s" * (len(here) != 1)}</span><span class="strip" style="--door:{ROOMS[name][1]}">{hang}</span>'
            f'</a></li>')


def film():
    """The film, if it has been made, behind its poster: its title over the sun of its first painting. -> HTML"""
    path = ROOT / "plates" / "_timeline.mp4"
    if not path.exists():
        return ""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "4", "-i", str(path), "-frames:v", "1", "-q:v", "3",
                    str(SITE / "thumbs" / "_timeline.jpg")], check=True)
    took = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True, check=True).stdout)
    mins, secs = divmod(round(took), 60)
    return f"""<figure class="screen">
<video class="film" id="film" src="plates/_timeline.mp4" poster="site/thumbs/_timeline.jpg" width="1920" height="1080" preload="none" playsinline></video>
<button type="button" class="play" aria-label="Play the film"></button>
<figcaption>The film: a walk through the five rooms · {mins} min {secs} s, with sound</figcaption>
</figure>"""


def entrance(title, hung, walk, reel):
    """The first screen: a bar with the rooms over the work at the door, and the name beside it on paper."""
    m = hung[FRONT]
    (w, h), shade = hero()
    name = e(html.unescape(re.sub("<.*?>", "", title)))
    nav = "".join(f'<a href="#{anchor(r)}">{e(r)}</a>' for r in walk)
    ways = '<a href="#intro">Walk in &rarr;</a>' + ('<a href="#film">Watch the film &rarr;</a>' if reel else "")
    who = re.split(r",|:", m.AFTER, maxsplit=1)[0]
    return f"""<header class="hero" {walls(PAPER)}>
<div class="bar"><a class="mark" href="#">{name}</a><nav aria-label="Rooms">{nav}</nav></div>
<a class="picture" href="plates/{FRONT}.jpg"><img src="site/hero.jpg" srcset="site/thumbs/{FRONT}.jpg 1400w, site/hero.jpg {w}w" sizes="(max-width: 52rem) 100vw, (orientation: portrait) 100vw, 60vw" width="{w}" height="{h}" alt="{e(m.TITLE)}" fetchpriority="high" style="background:{shade}"></a>
<div class="words">
<p class="kicker">A museum of one painter</p>
{title}
<p class="sub">{strapline(hung)}</p>
<p class="ways">{ways}</p>
<p class="credit">In the picture · <cite>{e(m.TITLE)}</cite>, after {e(who)}</p>
</div>
</header>"""


def build(slugs):
    for d in ("thumbs", "mini"):
        shutil.rmtree(SITE / d, ignore_errors=True)
        (SITE / d).mkdir(parents=True)
    (ROOT / ".nojekyll").touch()  # Jekyll would drop files whose names start with "_", like the wall
    hung = {s: importlib.import_module(f"works.{s}") for s in slugs}
    size = {s: thumb(s) for s in slugs}
    thumb("_wall")  # for the README and the social card
    walk = rooms(hung)

    css = Template(CSS).substitute(dict(zip(("wall", "ink", "dim"), colours(PAPER))), accent=ACCENT)
    (SITE / "museum.css").write_text(css)
    (SITE / "museum.js").write_text(JS)
    icon()
    # a changed stylesheet or script gets a new URL, so no browser pairs new HTML with an old cached copy
    v = {k: hashlib.sha1(s.encode()).hexdigest()[:8] for k, s in (("css", css), ("js", JS))}
    title, subtitle, *text = markdown("entrance.md")
    plan = "\n".join(door(n, r, hung, size) for n, r in zip(NUMERALS, walk))
    halls = "\n".join(room(n, r, {s: m for s, m in hung.items() if m.ROOM == r}, size) for n, r in zip(NUMERALS, walk))
    marks = "".join(f'<li><a href="#{anchor(r)}"><span class="numeral">{n}</span> {e(r)}</a></li>' for n, r in zip(NUMERALS, walk))
    reel = film()
    plain = lambda h: e(html.unescape(re.sub("<.*?>", "", h)))
    nl = "\n"
    (ROOT / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{plain(title)}</title>
<meta name="description" content="{plain(subtitle)}">
<meta name="theme-color" content="{PAPER}">
<meta property="og:type" content="website">
<meta property="og:url" content="{URL}">
<meta property="og:title" content="{plain(title)}">
<meta property="og:description" content="{plain(subtitle)}">
<meta property="og:image" content="{URL}site/thumbs/_wall.jpg">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="site/icon.png">
<link rel="apple-touch-icon" href="site/icon.png">
<link rel="stylesheet" href="site/museum.css?v={v['css']}">
<script src="site/museum.js?v={v['js']}" defer></script>
</head>
<body>
<details class="marker"><summary></summary><ol>{marks}<li><a href="#">The first page</a></li></ol></details>
{entrance(title, hung, walk, reel)}
<section class="intro" id="intro" {walls(PAPER)}>
<div class="text">
{nl.join(text)}
</div>
{reel}
<nav class="plan" id="plan" aria-label="Plan of the rooms">
<ol>
{plan}
</ol>
</nav>
</section>
<main>
{halls}
</main>
<footer class="colophon" {walls(PAPER)}>
{nl.join(markdown("colophon.md"))}
</footer>
<dialog class="viewer">
<div class="caption"><p class="where"></p><div class="label"></div><p class="hint"></p></div>
<div class="stage"></div>
<div class="back"><div class="linen" tabindex="0"><p class="file"></p><pre><code></code></pre></div></div>
<div class="controls">
<button type="button" class="turn" title="Turn it over (T)">Turn over</button>
<button type="button" class="prev" aria-label="Previous work">&larr;</button>
<button type="button" class="next" aria-label="Next work">&rarr;</button>
<button type="button" class="close" aria-label="Close" autofocus>&times;</button>
</div>
</dialog>
</body>
</html>
""")
    print("index.html:", len(hung), "works in", len(walk), "rooms")


CSS = """\
@property --wall { syntax: "<color>"; inherits: true; initial-value: $wall; }
@property --ink { syntax: "<color>"; inherits: true; initial-value: $ink; }
@property --dim { syntax: "<color>"; inherits: true; initial-value: $dim; }
:root {
  --wall: $wall;
  --ink: $ink;
  --dim: $dim;
  --accent: $accent;
  --rule: color-mix(in srgb, var(--dim) 28%, transparent);
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color-scheme: dark;
  background: var(--wall);
  color: var(--ink);
  font: 106.25%/1.6 var(--serif);  /* not 1.0625rem: while the walls change colour, Safari reads a rem here against
                                      this very size, frame after frame, and the type swells */
  scroll-behavior: smooth;
  -webkit-text-size-adjust: 100%;
  transition: --wall 1.6s ease, --ink 1.6s ease, --dim 1.6s ease;
}
:root:has(.viewer[open]) { overflow: hidden; }
* { box-sizing: border-box; }
body { margin: 0; }
a { color: inherit; text-decoration-color: var(--rule); text-decoration-thickness: 1px; text-underline-offset: .2em; }
a:hover { text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--ink); outline-offset: 4px; }
code, pre { font-family: var(--mono); font-size: .85em; }
pre { padding: 1rem 1.25rem; background: color-mix(in srgb, var(--ink) 4%, transparent); line-height: 1.6; white-space: pre-wrap;
  overflow-wrap: anywhere; }
h1, h2, h3 { text-wrap: balance; }
p { text-wrap: pretty; }
.numeral { letter-spacing: .08em; }

.hero { display: grid; grid-template: "bar bar" auto "picture words" minmax(0, 1fr) / 1fr 1fr; height: 100svh; min-height: 36rem; }
.bar { grid-area: bar; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: .5rem 2rem;
  padding: 1.2rem 4vw; border-bottom: 1px solid var(--rule); font-size: .8rem; font-variant-caps: all-small-caps;
  letter-spacing: .18em; white-space: nowrap; }
.bar a { text-decoration: none; }
.bar a:hover { text-decoration: underline; text-decoration-color: var(--accent); }
.mark { font-size: 1rem; letter-spacing: .22em; text-transform: uppercase; }
.bar nav { display: flex; gap: 1.5rem; }
.picture { grid-area: picture; display: block; min-height: 0; overflow: hidden; }
.picture img { display: block; width: 100%; height: 100%; object-fit: cover; object-position: 50% 40%; }
.words { grid-area: words; display: flex; flex-direction: column; justify-content: center; padding: 2rem 7vw 2rem 6vw; }
.words p { margin: 0; }
.kicker { color: var(--dim); font-size: .8rem; font-variant-caps: all-small-caps; letter-spacing: .18em; }
.hero h1 { margin: .9rem 0 0; font-size: clamp(2.6rem, 4.3vw, 4.2rem); font-weight: 400; line-height: 1.1; letter-spacing: .08em;
  text-transform: uppercase; }
.words .sub { margin-top: 1.4rem; font-size: 1.2rem; font-style: italic; line-height: 1.45; }
.words .ways { display: flex; flex-wrap: wrap; gap: .8rem; margin-top: 2.3rem; }
.ways a { padding: .7rem 1.25rem; border: 1px solid var(--ink); font-size: .95rem; text-decoration: none; }
.ways a:first-child { background: var(--ink); color: var(--wall); }
.ways a:hover { border-color: var(--accent); }
.words .credit { margin-top: 3.2rem; color: var(--dim); font-size: .8rem; }

.intro { padding: clamp(4rem, 12vh, 7rem) 1.25rem 0; }
.text { max-width: 33em; margin: 0 auto; }
.text p { margin: 0 0 1.1em; }
.text p:first-child::first-letter { float: left; margin: .06em .1em 0 0; color: var(--accent); font-size: 3.55em; line-height: .82; }
.screen { position: relative; width: min(100%, 60rem); margin: clamp(3.5rem, 10vh, 6rem) auto 0; }
.film { display: block; width: 100%; height: auto; background: #090807; box-shadow: 0 1.2rem 2.2rem -1.2rem rgb(0 0 0 / .45); }
.play { position: absolute; inset: 0 0 auto; width: 100%; aspect-ratio: 16 / 9; padding: 0; border: 0; background: none; cursor: pointer; }
.play::before { content: ""; position: absolute; left: clamp(.8rem, 2.5%, 1.6rem); bottom: clamp(.8rem, 4%, 1.6rem); width: 3.6rem;
  height: 3.6rem; border-radius: 50%; background: rgb(246 243 236 / .92); transition: scale .4s cubic-bezier(.2, .7, .3, 1); }
.play::after { content: ""; position: absolute; left: calc(clamp(.8rem, 2.5%, 1.6rem) + 1.42rem); bottom: calc(clamp(.8rem, 4%, 1.6rem) + 1.2rem);
  border: solid transparent; border-width: .6rem 0 .6rem 1rem; border-left-color: #29251f; }
.play:hover::before { scale: 1.08; }
.screen figcaption { margin-top: .9rem; color: var(--dim); font-size: .85rem; font-style: italic; text-align: center; }
.plan { max-width: 54rem; margin: clamp(5rem, 14vh, 8rem) auto 0; }
.plan ol { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--rule); }
.plan a { display: grid; grid-template: "num name strip" auto "num about strip" auto "num count strip" 1fr / 2.4rem minmax(0, 14rem) minmax(0, 1fr);
  column-gap: 1.5rem; padding: 1.5rem 0; border-bottom: 1px solid var(--rule); text-decoration: none; }
.plan .numeral { grid-area: num; color: var(--accent); font-size: 1.1rem; line-height: 1.6; }
.plan .name { grid-area: name; font-size: 1.1rem; font-variant-caps: all-small-caps; letter-spacing: .14em; line-height: 1.6; }
.plan a:hover .name { text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: .3em; }
.plan .about { grid-area: about; color: var(--dim); font-size: .9rem; font-style: italic; line-height: 1.4; }
.plan .count { grid-area: count; margin-top: .5rem; color: var(--dim); font-size: .8rem; }
.strip { grid-area: strip; align-self: center; display: flex; gap: .6rem; align-items: center; padding: 1rem 1.1rem; background: var(--door);
  box-shadow: inset 0 0 0 1px var(--rule); }
.strip img { flex: var(--a) 1 0; min-width: 0; height: auto; aspect-ratio: var(--a); box-shadow: 0 .35rem .6rem -.25rem rgb(0 0 0 / .45);
  transition: translate .5s cubic-bezier(.2, .7, .3, 1); }
.plan a:hover .strip img { translate: 0 -3px; }
@media (max-width: 40rem) {
  .plan a { grid-template: "num name" auto "num about" auto "num count" auto "strip strip" auto / 2rem minmax(0, 1fr); column-gap: .6rem; }
  .strip { margin-top: 1rem; padding: .8rem; gap: .45rem; }
}
.marker { position: fixed; top: .7rem; left: .7rem; z-index: 5; font-size: .8rem; font-variant-caps: all-small-caps; letter-spacing: .16em;
  opacity: 0; pointer-events: none; transition: opacity .6s ease; }
.marker.on { opacity: 1; pointer-events: auto; }
.marker summary { padding: .3rem .85rem; border-radius: 1rem; background: color-mix(in srgb, var(--wall) 88%, transparent);
  box-shadow: inset 0 0 0 1px var(--rule); color: var(--ink); cursor: pointer; list-style: none; }
.marker summary::-webkit-details-marker { display: none; }
.marker summary::after { content: "  \\25BE"; color: var(--dim); }
.marker[open] summary::after { content: "  \\25B4"; }
.marker ol { margin: .4rem 0 0; padding: .5rem .9rem; list-style: none; background: var(--wall);
  box-shadow: inset 0 0 0 1px var(--rule), 0 .8rem 1.6rem -.8rem rgb(0 0 0 / .5); }
.marker a { display: block; padding: .3rem 0; color: var(--ink); text-decoration: none; }
.marker a:hover { color: var(--accent); }
.marker .numeral { display: inline-block; min-width: 2.4em; color: var(--accent); }
@media (max-width: 52rem) { .bar nav { display: none; } }

main { max-width: 84rem; margin: 0 auto; padding: 0 clamp(1.25rem, 4vw, 3rem); }
.room > header { padding: clamp(10rem, 30vh, 18rem) 0 clamp(4rem, 11vh, 7rem); text-align: center; }
.room h2 { margin: 0; font-size: 1.6rem; font-weight: 400; font-variant-caps: all-small-caps; letter-spacing: .18em; }
.room h2 .numeral { display: block; margin-bottom: .5rem; color: var(--accent); font-size: 2.6rem; font-variant-caps: normal; letter-spacing: .04em; }
.room header p { margin: .5rem 0 0; color: var(--dim); font-style: italic; }
.row { margin: 0 0 clamp(6rem, 18vh, 11rem); }
.work { margin: 0; }
.plate { display: block; }
.plate img { display: block; width: 100%; height: auto; box-shadow: 0 .9rem 1.6rem -.6rem rgb(0 0 0 / .45);
  transition: transform .6s cubic-bezier(.2, .7, .3, 1); }
.plate:hover img { transform: translateY(-3px); }
.row.left .work, .row.right .work { display: grid; gap: 1.75rem 3.5rem; align-items: end; }
.row.left .work { grid-template-columns: minmax(0, 1fr) 15rem; }
.row.right .work { grid-template-columns: 15rem minmax(0, 1fr); }
.row.right .label { order: -1; }
.row.left .plate { width: min(100%, calc(80svh * var(--a))); }
.row.right .plate { width: min(100%, calc(68svh * var(--a))); justify-self: end; }
.row.pair { display: flex; gap: clamp(2rem, 5vw, 5rem); align-items: flex-start; max-width: min(100%, calc(62svh * var(--sum) + 5rem));
  margin-inline: auto; }
.row.pair .work { flex: var(--a) 1 0; min-width: 0; display: grid; align-content: start; gap: 1.4rem; }
.row.long .work { display: grid; gap: 1.6rem; }
.row.long .label { max-width: 34rem; }
.js .work { opacity: 0; transform: translateY(2.5rem); transition: opacity 1.2s ease, transform 1.2s cubic-bezier(.2, .7, .3, 1); }
.js .work.seen { opacity: 1; transform: none; }
.js .row.pair .work + .work { transition-delay: .18s; }

.label { color: var(--dim); font-size: .9rem; line-height: 1.5; }
figcaption.label { max-width: 32rem; }
.label h3 { margin: 0 0 .3rem; color: var(--ink); font-size: 1.05rem; font-weight: 400; }
.label p { margin: 0; }
.label .note { margin-top: 1rem; color: var(--ink); }
.label .source { margin-top: 1rem; font-size: .8rem; }
.label .full, .viewer .label .short { display: none; }
.viewer .label .full { display: block; }
.source a { font-family: var(--mono); }

.colophon { max-width: 33em; margin: 0 auto; padding: clamp(8rem, 24vh, 14rem) 1.25rem 6rem; color: var(--dim); }
.colophon h2 { color: var(--ink); font-size: 1.2rem; font-weight: 400; font-variant-caps: all-small-caps; letter-spacing: .16em; text-align: center; }

.viewer { inset: 0; width: auto; height: auto; max-width: none; max-height: none; margin: 0; padding: 0; border: 0;
  overflow: hidden; overscroll-behavior: contain; background: var(--wall); color: var(--ink); }
.viewer::backdrop { background: rgb(12 11 10 / .9); }
.viewer[open] { display: grid; grid-template: "stage caption" minmax(0, 1fr) / minmax(0, 1fr) 21rem; animation: enter .3s ease-out; }
@keyframes enter { from { opacity: 0; } }
.stage { grid-area: stage; position: relative; overflow: hidden; touch-action: none; cursor: grab; user-select: none; -webkit-user-select: none; }
.stage.held { cursor: grabbing; }
.stage img { display: block; background-size: 100% 100%; -webkit-user-drag: none; }
.viewer:not(.scroll) .stage img { position: absolute; top: 0; left: 0; max-width: none; transform-origin: 0 0; }
.caption { grid-area: caption; align-self: end; max-height: 100%; overflow-y: auto; padding: 2rem 2.5rem 2.5rem 1.5rem; color: var(--dim); }
.where, .hint { margin: 0; font-size: .8rem; font-variant-caps: all-small-caps; letter-spacing: .12em; }
.where { margin-bottom: 1.25rem; }
.hint { margin-top: 2rem; opacity: .8; }
.controls { position: fixed; top: .75rem; right: .75rem; display: flex; gap: .25rem; }
.controls button { width: 2.75rem; height: 2.75rem; border: 0; border-radius: 50%; background: color-mix(in srgb, var(--wall) 60%, transparent);
  color: var(--ink); font: 1.35rem/1 var(--serif); cursor: pointer; }
.controls button:hover { background: color-mix(in srgb, var(--ink) 10%, transparent); }
.controls .turn { width: auto; padding: 0 1.1rem; border-radius: 1.4rem; font: .85rem/1 var(--serif); font-variant-caps: all-small-caps;
  letter-spacing: .12em; }
.viewer.scroll[open] { display: flex; overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; scrollbar-color: var(--dim) transparent; }
.viewer.scroll .caption { flex: 0 0 min(24rem, 80vw); max-height: none; padding: 2rem 2.5rem; }
.viewer.scroll .stage { flex: none; height: 100%; touch-action: pan-x; }
.viewer.scroll .stage img { width: auto; height: 100%; }
.back { grid-area: stage; display: none; min-height: 0; margin: clamp(.75rem, 2.5vw, 2rem); padding: 1.1rem; border-radius: 2px;
  background: linear-gradient(90deg, rgb(255 255 255 / .07), transparent 35%, rgb(0 0 0 / .1)), #9a7b51;
  box-shadow: 0 1.5rem 3rem -1rem rgb(0 0 0 / .55); }
.linen { height: 100%; overflow: auto; overscroll-behavior: contain; padding: clamp(1.5rem, 4vw, 3rem); color: #3b3327;
  background: repeating-linear-gradient(90deg, rgb(90 70 40 / .06) 0 1px, transparent 1px 3px),
    repeating-linear-gradient(0deg, rgb(90 70 40 / .05) 0 1px, transparent 1px 3px), #d8ccb3;
  box-shadow: inset 0 0 2.5rem rgb(70 50 20 / .3); }
.linen:focus-visible { outline-offset: -4px; }
.file { margin: 0 0 1.5rem; font-size: .8rem; font-variant-caps: all-small-caps; letter-spacing: .14em; }
.linen pre { margin: 0; padding: 0; overflow: visible; background: none; font-size: .75rem; line-height: 1.55; }
.viewer.turned .stage { display: none; }
.viewer.turned .back { display: block; }
.viewer.scroll.turned[open] { display: grid; overflow: hidden; }

@media (max-width: 52rem), (orientation: portrait) {
  .hero { grid-template: "bar" auto "picture" 46svh "words" auto / minmax(0, 1fr); height: auto; }
  .words { padding: 2.2rem 6vw 3rem; }
  .hero h1 { margin-top: .6rem; font-size: min(9.6vw, 4.2rem); }
  .words .sub { margin-top: .9rem; }
  .words .ways { margin-top: 1.6rem; }
  .words .credit { margin-top: 1.8rem; }
  .room > header { padding: clamp(5rem, 14vh, 8rem) 0 clamp(2.5rem, 6vh, 4rem); }
  .row { margin-bottom: clamp(4rem, 11vh, 6rem); }
  .row.left .work, .row.right .work { grid-template-columns: minmax(0, 1fr); }
  .row.right .label { order: 0; }
  .row.right .plate { justify-self: start; }
  .row.pair { flex-direction: column; max-width: none; }
  .row.pair .work { flex: none; }
  .viewer[open] { padding-top: 3.6rem; }  /* the controls above the work, not over it */
  .viewer[open]:not(.scroll), .viewer.turned[open] { grid-template: "stage" minmax(0, 1fr) "caption" auto / minmax(0, 1fr); }
  .viewer:not(.scroll) .caption, .viewer.turned .caption { max-height: 40vh; padding: 1rem 1.25rem 1.5rem; }
}
@media (prefers-reduced-motion: reduce) {
  :root { scroll-behavior: auto; }
  *, ::backdrop { animation: none !important; transition: none !important; }
  .js .work { opacity: 1; transform: none; }
}
"""

JS = r"""// The museum's moving parts: walls that take on each room's colour as a visitor walks in; and one <dialog>
// that shows every work (a plate zooms and pans, a long painting unrolls from its left end) and turns it over,
// to the program on its back.
const root = document.documentElement;
root.classList.add('js');
const links = [...document.querySelectorAll('main a.plate')];
const viewer = document.querySelector('.viewer');
const stage = viewer.querySelector('.stage');
const back = viewer.querySelector('.back');
const calm = matchMedia('(prefers-reduced-motion: reduce)');
const pts = new Map(), sources = new Map();
let i = 0, img, nw, nh, s = 1, fit = 1, x = 0, y = 0, g = 1, from, opener;

const turned = () => viewer.classList.contains('turned');
const unrolled = () => viewer.classList.contains('scroll') && !turned();
const one = () => 1 / devicePixelRatio;  // one plate pixel to one screen pixel

// The walls: a room's colours come up as it reaches the middle of the screen, and its name in the marker.
const marker = document.querySelector('.marker'), here = marker.querySelector('summary');
const theme = document.querySelector('meta[name="theme-color"]');
const walk = new IntersectionObserver(seen => {
  for (const q of seen) {
    if (!q.isIntersecting) continue;
    const d = q.target.dataset;
    for (const k of ['wall', 'ink', 'dim']) root.style.setProperty(`--${k}`, d[k]);
    theme.content = d.wall;
    here.textContent = d.room || '';
    marker.classList.toggle('on', !!d.room);
    if (!d.room) marker.open = false;
  }
}, {rootMargin: '-50% 0px -50% 0px'});
document.querySelectorAll('[data-wall]').forEach(r => walk.observe(r));
marker.addEventListener('click', e => { if (e.target.closest('a')) marker.open = false; });

const rise = new IntersectionObserver(seen => {
  for (const q of seen) if (q.isIntersecting) { q.target.classList.add('seen'); rise.unobserve(q.target); }
}, {rootMargin: '0px 0px -8% 0px'});
document.querySelectorAll('.work').forEach(w => rise.observe(w));

// The film shows its poster and one play mark; once it plays, its own controls.
const film = document.getElementById('film');
if (film) {
  const play = document.querySelector('.play');
  play.addEventListener('click', () => film.play());
  film.addEventListener('play', () => { film.controls = true; play.hidden = true; });
  document.querySelector('a[href="#film"]').addEventListener('click', () => film.play());  // "Watch the film" at the door
}

// The viewer.
function open(n, by) {
  from = n;
  opener = by;
  viewer.showModal();
  show(n);
}

function show(n) {
  i = (n + links.length) % links.length;
  const a = links[i], fig = a.closest('figure'), thumb = a.querySelector('img');
  nw = +a.dataset.w;
  nh = +a.dataset.h;
  viewer.classList.toggle('scroll', fig.classList.contains('long'));
  viewer.setAttribute('aria-label', thumb.alt);
  viewer.querySelector('.where').textContent = fig.closest('.room').dataset.room;
  viewer.querySelector('.label').innerHTML = fig.querySelector('.label').innerHTML;
  hint();
  img = new Image(nw, nh);
  img.alt = thumb.alt;
  img.draggable = false;
  img.style.backgroundImage = `url("${thumb.src}")`;  // the thumbnail stands in while the plate loads
  img.src = a.href;
  stage.replaceChildren(img);
  viewer.scrollLeft = 0;
  if (turned()) code();
  else if (!unrolled()) reset();
}

function hint() {
  viewer.querySelector('.turn').textContent = turned() ? 'Turn back' : 'Turn over';
  viewer.querySelector('.hint').textContent = turned()
    ? 'The back of the canvas: the program that painted it. T turns it back; ← → for the next work.'
    : unrolled()
      ? 'A long painting, read from left to right: drag or scroll along it. T turns it over; ← → for the next work.'
      : 'Wheel or pinch to zoom, drag to move, double-click for 1:1. T turns it over; ← → for the next work.';
}

// The back of the canvas carries the work's own source, fetched once.
function code() {
  const a = links[i], src = `works/${a.dataset.slug}.py`, out = back.querySelector('code');
  back.querySelector('.file').textContent = `${src} · ${a.dataset.lines} lines of Python`;
  out.textContent = '';
  if (!sources.has(src)) sources.set(src, fetch(src).then(r => r.ok ? r.text() : Promise.reject(r.status)));
  sources.get(src).then(
    t => { if (links[i] === a) { out.textContent = t; back.firstElementChild.scrollTop = 0; } },
    () => { if (links[i] === a) out.textContent = `Read ${src} in the repository.`; });
}

function turn() {
  const flip = () => {
    viewer.classList.toggle('turned');
    hint();
    if (turned()) code();
    else if (!unrolled()) reset();
  };
  if (calm.matches) return flip();
  const p = 'perspective(2400px) rotateY';
  viewer.animate([{transform: `${p}(0deg)`}, {transform: `${p}(90deg)`}], {duration: 220, easing: 'ease-in'}).finished.then(() => {
    flip();
    viewer.animate([{transform: `${p}(-90deg)`}, {transform: `${p}(0deg)`}], {duration: 320, easing: 'ease-out'});
  });
}

function reset() {
  s = fit = Math.min(stage.clientWidth / nw, stage.clientHeight / nh);
  draw();
}

function draw(glide) {
  const W = stage.clientWidth, H = stage.clientHeight, w = nw * s, h = nh * s;
  x = w < W ? (W - w) / 2 : Math.min(0, Math.max(W - w, x));
  y = h < H ? (H - h) / 2 : Math.min(0, Math.max(H - h, y));
  img.style.transition = glide ? 'transform .3s ease' : 'none';
  img.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
}

function zoom(cx, cy, to, glide) {
  const r = stage.getBoundingClientRect(), px = cx - r.left, py = cy - r.top;
  to = Math.min(4, Math.max(Math.min(fit, one()), to));
  x = px - (px - x) * to / s;
  y = py - (py - y) * to / s;
  s = to;
  draw(glide);
}

document.addEventListener('click', e => {
  const a = e.target.closest('main a.plate, .picture');  // a work on the walls, or the one at the door
  if (!a || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  open(links.findIndex(l => l.href === a.href), a);
});
viewer.querySelector('.turn').onclick = turn;
viewer.querySelector('.prev').onclick = () => show(i - 1);
viewer.querySelector('.next').onclick = () => show(i + 1);
viewer.querySelector('.close').onclick = () => viewer.close();
viewer.addEventListener('close', () => {
  viewer.classList.remove('turned');
  stage.replaceChildren();
  (i === from ? opener : links[i]).focus({preventScroll: i === from});
});
new ResizeObserver(() => viewer.open && !unrolled() && !turned() && reset()).observe(stage);

document.addEventListener('keydown', e => {
  if (!viewer.open || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key, r = stage.getBoundingClientRect();
  if (key === 'ArrowRight' || key === 'ArrowLeft') show(i + (key === 'ArrowRight' ? 1 : -1));
  else if (key === 't' || key === 'T') turn();
  else if (unrolled() && (key === 'ArrowDown' || key === 'ArrowUp'))
    viewer.scrollBy({left: (key === 'ArrowDown' ? 0.6 : -0.6) * innerWidth, behavior: calm.matches ? 'auto' : 'smooth'});
  else if (!unrolled() && !turned() && ['+', '=', '-'].includes(key))
    zoom(r.left + r.width / 2, r.top + r.height / 2, s * (key === '-' ? 2 / 3 : 1.5), true);
  else return;
  e.preventDefault();
});

stage.addEventListener('pointerdown', e => {
  if (e.button || (unrolled() && e.pointerType !== 'mouse')) return;  // fingers scroll a long painting natively
  stage.setPointerCapture(e.pointerId);
  pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
  stage.classList.add('held');
});
stage.addEventListener('pointermove', e => {
  const p = pts.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y, o = [...pts.values()].find(q => q !== p);
  if (unrolled()) viewer.scrollLeft -= dx;
  else if (!o) { x += dx; y += dy; draw(); }
  else {  // pinch: the midpoint moves half as far as this finger, and the spread sets the scale
    x += dx / 2;
    y += dy / 2;
    zoom((e.clientX + o.x) / 2, (e.clientY + o.y) / 2,
      s * Math.hypot(e.clientX - o.x, e.clientY - o.y) / (Math.hypot(p.x - o.x, p.y - o.y) || 1));
  }
  p.x = e.clientX;
  p.y = e.clientY;
});
for (const type of ['pointerup', 'pointercancel']) stage.addEventListener(type, e => {
  pts.delete(e.pointerId);
  stage.classList.toggle('held', pts.size > 0);
});

viewer.addEventListener('wheel', e => {
  const d = e.deltaY * (e.deltaMode ? 32 : 1);  // some browsers count lines, not pixels
  if (unrolled()) {
    if (Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;  // a sideways swipe scrolls natively
    viewer.scrollLeft += d;  // wheel down reads on, to the right
  } else if (stage.contains(e.target)) {
    zoom(e.clientX, e.clientY, s * Math.exp(-d * (e.ctrlKey ? 0.01 : 0.002)));
  } else return;
  e.preventDefault();
}, {passive: false});
stage.addEventListener('dblclick', e => {
  if (!unrolled()) zoom(e.clientX, e.clientY, Math.abs(s - one()) < 1e-3 ? fit : one(), true);
});
// Safari's trackpad pinch arrives as gesture events, not as ctrl+wheel
stage.addEventListener('gesturestart', e => { e.preventDefault(); g = s; });
stage.addEventListener('gesturechange', e => {
  e.preventDefault();
  if (!unrolled() && pts.size < 2) zoom(e.clientX, e.clientY, g * e.scale);
});
"""

if __name__ == "__main__":
    slugs = hanging()
    wall(slugs)
    build(slugs)
    readme(slugs)  # after build: the README shows its thumbnails
