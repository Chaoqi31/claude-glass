"""The film's score: a voice singing one figure over and over, in D major, while a choir and a low hum move
under it, written to the bars of the film and played on the instruments in atelier/music.

    uv run score.py      writes .scratch/score.wav, to listen to on its own

The film (timeline.py) says how many bars each of its parts lasts, and the score fills them. A voice sings
alone in the dark. Then it takes up the figure: six notes a bar, three, three and two sixteenths long, the
same notes over every chord, so that the chords below turn them from one colour to another. The figure
quickens to every sixteenth while air rises under it, breaks off, and a drum falls on the title; at the close
the voice sings the figure's first notes again and comes to rest on D.
"""

from pathlib import Path

import numpy as np

from atelier import music, noise

TEMPO = 112.5               # beats a minute: a sixteenth is 4 frames of the film at 30 fps
STEP = 60 / TEMPO / 4       # a sixteenth, in seconds
BAR = 16 * STEP
TAIL = 3.0                  # seconds the last chord rings on after the last bar
STEPS = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}

HOOK = ("F#5", "D5", "A4", "E5", "C#5", "A4")    # the figure
RHYTHM = (0, 3, 6, 8, 11, 14)                     # where its notes fall in the bar, in sixteenths
ACCENT = (1.0, 0.8, 0.75, 0.95, 0.8, 0.72)
CHORDS = {   # each chord: its root, for the hum, and how the choir holds it, in the middle, out of the hum's way
    "D": ("D2", "A3 D4 E4 F#4"),
    "Bm": ("B1", "F#3 B3 D4 E4"),
    "G": ("G1", "G3 B3 D4 F#4"),
    "A": ("A1", "E3 A3 C#4 E4"),
    "Asus": ("A1", "E3 A3 D4 E4"),
}
COUNTER = {"D": "A4", "Bm": "B4", "G": "D5", "A": "C#5", "Asus": "D5"}   # a second voice, a note a chord

# The parts of the piece. chords: one a bar, round again if the part is longer; choir, hum, hook: how loud
# each plays (choir_from: the bar the choir comes in on); pulse: the hum's notes a bar; counter, low, descant:
# a second voice on long notes, the figure an octave down, and a high line above it; sung: notes of its own,
# (bar, sixteenth, pitch, sixteenths held), at `sungv`.
PARTS = {
    "intro": dict(chords="D", choir=0.06, choir_from=1, hum=0.11, sung=[(0, 0, "A4", 7), (1, 0, "F#5", 7), (1, 8, "E5", 7)],
                  sungv=0.2),
    "slow": dict(chords="D D Bm Bm", choir=0.09, hum=0.22, hook=0.3, sparse=True),
    "paint": dict(chords="D Bm G A", choir=0.13, hum=0.3, hook=0.4),
    "cuts": dict(chords="D Bm G A", choir=0.2, hum=0.4, hook=0.52, pulse=2, counter=0.2),
    "quick": dict(chords="G A Asus", choir=0.32, hum=0.55, hook=0.75, pulse=4, low=0.35, descant=0.4),
    "burst": dict(chords="Asus", choir=0.45, hum=0.65),
    "title": dict(chords="D", choir=0.08, choir_from=1, hum=0.17, sungv=0.26, sung=[(1, 0, "A5", 14)]),
    "coda": dict(chords="G D", choir=0.1, hum=0.2, sungv=0.32,
                 sung=[(0, 0, "F#5", 3), (0, 3, "D5", 3), (0, 6, "A4", 2), (0, 8, "E5", 8), (1, 0, "D5", 16)]),
}


def pitch(name):
    """'F#5' -> 78"""
    return 12 * (int(name[-1]) + 1) + STEPS[name[:-1]]


def notes(plan, r):
    """The notes of every instrument for a plan of the film, [(kind, bars)].
    -> {instrument: [(start, pitch, velocity, held)]}, the drum's strokes [(start, velocity)], and the rushes
    of air [(start, end, velocity)]"""
    out = {k: [] for k in ("voice", "low", "choir", "counter", "hum")}
    booms, rushes = [], []
    t0 = 0.0
    for j, (kind, n) in enumerate(plan):
        part = PARTS[kind]
        cycle = part["chords"].split()
        chords = [cycle[b % len(cycle)] for b in range(n)]
        last = j == len(plan) - 1
        b = part.get("choir_from", 0)
        while b < n:                                         # the choir holds a chord for as long as it lasts
            e = b
            while e + 1 < n and chords[e + 1] == chords[b]:
                e += 1
            held = (e + 1 - b) * BAR + (TAIL if last and e == n - 1 else 0.1)
            for p in CHORDS[chords[b]][1].split():
                out["choir"].append((t0 + b * BAR, pitch(p), part["choir"], held))
            if part.get("counter") and b >= 2:
                out["counter"].append((t0 + b * BAR, pitch(COUNTER[chords[b]]), part["counter"], held))
            b = e + 1
        for b, c in enumerate(chords):
            at = t0 + b * BAR
            beats = part.get("pulse", 1)
            for i in range(beats):                           # the hum on the root, once a bar or on the beats
                hold = BAR / beats * 0.85 if beats > 1 else BAR + (TAIL if last and b == n - 1 else 0.05)
                out["hum"].append((at + i * BAR / beats, pitch(CHORDS[c][0]), part["hum"] * (1 - 0.15 * (i % 2)), hold))
            if part.get("hook"):
                sparse = part.get("sparse") and b == 0           # the figure begins with two of its notes
                for i in (0, 3) if sparse else range(6):
                    held = (6 if sparse else (RHYTHM[i + 1] if i < 5 else 16) - RHYTHM[i]) * STEP * 0.9
                    out["voice"].append((at + RHYTHM[i] * STEP, pitch(HOOK[i]), part["hook"] * ACCENT[i], held))
                    if part.get("low"):
                        out["low"].append((at + RHYTHM[i] * STEP, pitch(HOOK[i]) - 12, part["low"] * ACCENT[i], held))
            if part.get("descant"):
                out["voice"].append((at, pitch(("A5", "B5", "C#6")[b % 3]), part["descant"], BAR * 0.95))
        if kind == "burst":                                  # every sixteenth, louder and louder, and a breath before the fall
            for i in range(15):
                out["voice"].append((t0 + i * STEP, pitch(HOOK[i % 6]), 0.75 + 0.25 * i / 14, STEP * 0.85))
            rushes.append((t0 - BAR, t0 + 15 * STEP, 1.0))
        for b, s, p, held in part.get("sung", ()):
            out["voice"].append((t0 + b * BAR + s * STEP, pitch(p), part["sungv"], held * STEP + (TAIL if last and b == n - 1 else 0)))
        if kind == "title":
            booms.append((t0, 1.0))
        t0 += n * BAR
    for k in out:                                            # no two voices land quite together
        out[k] = [(max(0.0, s + r.normal(0, 0.004)), p, float(np.clip(v + r.normal(0, 0.03), 0.05, 1)), h)
                  for s, p, v, h in out[k]]
    return out, booms, rushes


def render(plan, seed=1125):
    """The score for a plan of the film, mastered: stereo float at music.RATE, TAIL seconds longer than the bars."""
    r = noise.rng(seed)
    length = sum(n for _, n in plan) * BAR + TAIL
    n, booms, rushes = notes(plan, r)
    sung = music.voice(n["voice"], length, r) + 0.8 * music.voice(n["low"], length, r, vowel="o")
    held = music.choir(n["choir"], length, r) + music.choir(n["counter"], length, r, vowel="u", voices=2)
    drum, air, hum = music.boom(booms, length, r), music.rush(rushes, length, r), music.hum(n["hum"], length)
    dry = sung + 0.8 * held + hum + 0.9 * drum + air
    return music.master(dry + music.hall(0.35 * sung + 0.6 * held + 0.5 * drum + 0.3 * air, r))


if __name__ == "__main__":
    import importlib

    from render import hanging
    from timeline import plan
    parts, _ = plan({s: importlib.import_module(f"works.{s}") for s in hanging()})
    out = Path(__file__).parent / ".scratch" / "score.wav"
    out.parent.mkdir(exist_ok=True)
    music.write(out, render(parts))
    print(out, f"{sum(n for _, n in parts) * BAR + TAIL:.1f}s")
