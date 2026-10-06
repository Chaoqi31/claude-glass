"""Paint the works into plates.

    uv run render.py              paint every work
    uv run render.py giverny      paint one, or several

Hanging them (the wall, the README and the website) is site.py's job.
"""

import importlib
import sys
import time
from pathlib import Path

from atelier import plate

ROOT = Path(__file__).parent
SCROLL = 2.4  # a plate wider than this is a long painting, which the site and the film unroll from left to right


def works():
    return sorted(p.stem for p in (ROOT / "works").glob("*.py") if not p.stem.startswith("_"))


def hanging():
    """The works on the walls: those with a plate. One still being painted has none yet, and waits."""
    out = [s for s in works() if (ROOT / "plates" / f"{s}.jpg").exists()]
    for s in sorted(set(works()) - set(out)):
        print(f"{s}: no plate yet, not hung")
    return out


def render(slug):
    t = time.time()
    img = importlib.import_module(f"works.{slug}").paint()
    plate.save(img, ROOT / "plates" / f"{slug}.jpg")
    print(f"{slug:24s} {img.shape[1]}x{img.shape[0]}  {time.time() - t:5.1f}s")


if __name__ == "__main__":
    for slug in sys.argv[1:] or works():
        render(slug)
