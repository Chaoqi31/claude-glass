---

## Colophon

Every plate in this museum can be made again from its source, on any computer, in a minute or two.

```bash
git clone https://github.com/Chaoqi31/claude-glass && cd claude-glass
uv run render.py giverny     # paint one work into plates/
uv run render.py             # paint them all
uv run site.py               # hang them: the wall, this README and the website
uv run timeline.py           # the film (needs ffmpeg)
```

A work is one Python file in [`works/`](works). It carries its wall label (title, medium, the painter or tradition it follows, the year, the place) and a function, `paint(seed)`, that returns the picture in linear light. Each picture is built the way its medium is: a stroke is a path that carries a load of paint, a block is cut, inked and pressed, glass is cut and leaded, and what you see is whatever the bristles, the pigment and the ground leave behind.

The materials are in [`atelier/`](atelier), each a small module that knows one craft: canvas and thick oil paint that stands in ridges and drags through wet paint ([`impasto.py`](atelier/impasto.py)), watercolour that pools and dries darker at its rim, divided touches of colour, etching and printing, woodblocks cut, inked, wiped and pulled by hand, gouache, stained glass and its leads, the craquelure of old paint, laid and handmade paper. A few works carry a process of their own: a squeegee dragged through wet paint, colour poured into raw cotton. Everything is NumPy, SciPy and Pillow; there is no image model anywhere in this repository.

To add a work, write a new file in `works/` with the same label and a `paint` of its own, render it, and hang it.

The code is released under the [MIT License](LICENSE); the plates, the wall and the film under [CC BY 4.0](plates/LICENSE).
