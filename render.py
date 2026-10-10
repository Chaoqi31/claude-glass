"""Paint the works into plates.

    uv run render.py              paint every work
    uv run render.py giverny      paint one, or several

Hanging them (the wall, the README and the website) is site.py's job, and the film is timeline.py's; both
take the catalogue from here: which works hang, in which rooms, and what they come to.
"""

import importlib
import sys
import time
from pathlib import Path

from atelier import plate

ROOT = Path(__file__).parent
SCROLL = 2.4  # a plate wider than this is a long painting, which the site's viewer unrolls from left to right
ROOMS = {  # the order a visitor walks them in: what hangs in each, and the colour of its walls
    "Open Air": ("Weather, water and light, painted out of doors", "#2a2622"),
    "The Garden": ("Flowers, a pond, a window", "#1f2b25"),
    "Paper and Water": ("Watercolour, ink and mineral colour", "#212835"),
    "The Workshop": ("Glass, copper and the woodblock", "#33201d"),
    "Colour Itself": ("Abstraction", "#f1eee8"),
}
FRONT = "attersee"  # the work beside the museum's name on its first page: the website's first screen, the film's last
PAPER = "#f6f3ec"  # the paper of that page


def words(n):
    """A count under a hundred, in words."""
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen").split()
    tens = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    return ones[n] if n < 20 else tens[n // 10 - 2] + (f"-{ones[n % 10]}" if n % 10 else "")


def strapline(slugs):
    """The museum in a line, under its name on its first page."""
    return f"{words(len(slugs)).capitalize()} paintings by Claude, each one a program"


def works():
    return sorted(p.stem for p in (ROOT / "works").glob("*.py") if not p.stem.startswith("_"))


def hanging():
    """The works on the walls: those with a plate. One still being painted has none yet, and waits."""
    out = [s for s in works() if (ROOT / "plates" / f"{s}.jpg").exists()]
    for s in sorted(set(works()) - set(out)):
        print(f"{s}: no plate yet, not hung")
    return out


def rooms(mods):
    """The rooms with something hung in them, in the order a visitor walks them."""
    lost = {m.ROOM for m in mods.values()} - set(ROOMS)
    assert not lost, f"no such room: {lost}"
    return [r for r in ROOMS if any(m.ROOM == r for m in mods.values())]


def tally(slugs):
    """What the collection comes to, in a line."""
    lines = sum(len((ROOT / "works" / f"{s}.py").read_text().splitlines()) for s in slugs)
    return f"{len(slugs)} paintings · {lines:,} lines of Python · no image models, no photographs"


def render(slug):
    t = time.time()
    img = importlib.import_module(f"works.{slug}").paint()
    plate.save(img, ROOT / "plates" / f"{slug}.jpg")
    print(f"{slug:24s} {img.shape[1]}x{img.shape[0]}  {time.time() - t:5.1f}s")


if __name__ == "__main__":
    for slug in sys.argv[1:] or works():
        render(slug)
