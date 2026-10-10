"""The film's score: a voice singing one figure over and over above a low D that never stops, in the manner of a
launch film, written to the film's parts and played on the instruments in atelier/music.

    uv run score.py      writes .scratch/score.wav, to listen to on its own

The film (timeline.py) takes its parts and their bars from FORM. A low D swells out of nothing and a voice sings
alone. A piano sparkles while the code is written. The voice takes up the figure as the first strokes land, six
notes a bar, three, three and two sixteenths long, the same notes over every chord, so that the chords below turn
them from one colour to another; a choir and a second voice gather round it, air rises into each new part and a
drum falls on its first note, and the whole band, held back at first, plays louder part by part. At its loudest it
breaks off, and four blows of low brass fall into the silence, dark chords outside the key, one for each thing the
paintings were not made with. The band comes back at its fullest, quickens to every sixteenth while air rises
under it, and breaks off again. After a breath of silence the drum and the brass fall on the title, and the
voice sings the figure's first notes once more and comes to rest on D.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from atelier import music, noise

TEMPO = 112.5               # beats a minute: a sixteenth is 4 frames of the film at 30 fps
STEP = 60 / TEMPO / 4       # a sixteenth, in seconds
BAR = 16 * STEP
TAIL = 2.0                  # seconds the last chord rings on after the last bar
FORM = (("intro", 2), ("wrote", 2), ("strokes", 4), ("woven", 4), ("pixel", 4), ("statements", 2), ("montage", 5),
        ("breath", 1), ("title", 3))
STEPS = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}

HOOK = ("F#5", "D5", "A4", "E5", "C#5", "A4")    # the figure
RHYTHM = (0, 3, 6, 8, 11, 14)                     # where its notes fall in the bar, in sixteenths
ACCENT = (1.0, 0.8, 0.75, 0.95, 0.8, 0.72)
CHORDS = {   # each chord: its root, for the hum, and how the choir holds it, in the middle, out of the hum's way
    "D": ("D2", "A3 D4 E4 F#4"), "Bm": ("B1", "F#3 B3 D4 E4"), "G": ("G1", "G3 B3 D4 F#4"), "A": ("A1", "E3 A3 C#4 E4"),
    "Em": ("E2", "G3 B3 D4 E4"), "Asus": ("A1", "E3 A3 D4 E4"),
}
COUNTER = {"D": "A4", "Bm": "B4", "G": "D5", "A": "C#5", "Em": "B4", "Asus": "D5"}   # a second voice, a note a chord
DESCANT = {"D": "A5", "Bm": "B5", "G": "B5", "A": "C#6", "Em": "B5"}                 # and a high one
SPARKLE = ("A5", "D6", "E6", "F#6", "A6", "F#6", "E6", "D6")                          # the piano, as the code is written
HITS = ((0, "D1 D2 A2 D3"), (8, "Bb1 F2 Bb2 D3"), (16, "C2 G2 C3 E3"), (24, "A1 E2 A2 C#3"))   # sixteenths into the part
TITLE = "D1 D2 A2 D3 F#3"
# the balance of the whole, set nearer the launch film's: less of the low hum's octave, much more presence and air
PRESENCE = ((20, 0), (45, 0), (63, -5), (125, 0), (500, 0), (1000, 4), (2000, 8), (4000, 9), (8000, 4), (16000, 0))


def pitch(name):
    """'F#5' -> 78"""
    return 12 * (int(name[-1]) + 1) + STEPS[name[:-1]]


def starts():
    """The bar each part of the film begins on: {part: bar}"""
    out, b = {}, 0
    for name, n in FORM:
        out[name] = b
        b += n
    return out


def notes(r):
    """Every instrument's notes. -> {instrument: [(start, pitch, velocity, held)]}, the drum's strokes [(start,
    velocity)], the brass [(start, pitches, velocity, held)], the rushes of air [(start, end, velocity)] and the
    level of the low D through the piece [(time, level)]."""
    out, booms, brass, rushes = defaultdict(list), [], [], []
    at = starts()
    t = lambda bar, s=0: (bar + s / 16) * BAR             # seconds at a bar and a sixteenth into it

    def harmony(first, chords, choir, hum, pulse=1, counter=0.0, descant=0.0):
        """A part's chords, one a bar: the choir holding each, the hum on its root, a second voice and a high one."""
        for k, c in enumerate(chords.split()):
            root, held = CHORDS[c]
            for p in held.split():
                out["choir"].append((t(first + k), pitch(p), choir, BAR + 0.1))
            for i in range(pulse):
                out["hum"].append((t(first + k) + i * BAR / pulse, pitch(root), hum * (1 - 0.15 * (i % 2)),
                                   BAR / pulse * (0.85 if pulse > 1 else 1.02)))
            if counter:
                out["counter"].append((t(first + k), pitch(COUNTER[c]), counter, BAR + 0.1))
            if descant and c in DESCANT:
                out["voice"].append((t(first + k), pitch(DESCANT[c]), descant, BAR * 0.95))

    def figure(first, bars, v, low=0.0):
        """The figure, once a bar, and an octave down under it."""
        for b in range(first, first + bars):
            for i, s in enumerate(RHYTHM):
                held = ((RHYTHM[i + 1] if i < 5 else 16) - s) * STEP * 0.9
                out["voice"].append((t(b, s), pitch(HOOK[i]), v * ACCENT[i], held))
                if low:
                    out["low"].append((t(b, s), pitch(HOOK[i]) - 12, low * ACCENT[i], held))

    def sung(first, line, v):
        for b, s, p, n in line:
            out["voice"].append((t(first + b, s), pitch(p), v, n * STEP))

    b = at["intro"]                                         # the low D out of nothing, and a voice alone
    sung(b, [(0, 0, "A4", 14), (1, 0, "F#5", 7), (1, 8, "E5", 7)], 0.22)
    harmony(b + 1, "D", 0.05, 0.0)
    b = at["wrote"]                                         # the piano sparkling while the code is written
    sung(b, [(0, 0, "D5", 7)], 0.24)
    harmony(b, "D Bm", 0.07, 0.1)
    for i in range(24):
        out["piano"].append((t(b, 8 + i), pitch(SPARKLE[i % 8]), 0.16 + 0.12 * i / 23, 0.3))
    rushes.append((t(b + 1, 8), t(b + 2), 0.3))
    b = at["strokes"]                                       # the figure, a stroke on each of its notes
    figure(b, 4, 0.42)
    harmony(b, "D Bm G A", 0.12, 0.3, counter=0.0)
    harmony(b + 2, "G A", 0.0, 0.0, counter=0.18)
    rushes.append((t(b + 3), t(b + 4), 0.5))
    booms.append((t(b + 4), 0.5))
    b = at["woven"]                                         # the code drawing back into the painting
    figure(b, 4, 0.52, low=0.22)
    harmony(b, "D Bm G A", 0.2, 0.4, pulse=2, counter=0.2)
    harmony(b + 2, "G A", 0.0, 0.0, descant=0.3)
    rushes.append((t(b + 2, 8), t(b + 4), 0.8))
    booms.append((t(b + 4), 0.7))
    b = at["pixel"]                                         # closer and closer, darker
    figure(b, 4, 0.56, low=0.32)
    harmony(b, "Bm G Em A", 0.24, 0.5, pulse=4, counter=0.22)
    rushes.append((t(b + 3), t(b + 4) - 2 * STEP, 0.9))
    b = at["statements"]                                    # four blows of brass into the silence
    for s, chord in HITS:
        brass.append((t(b, s), [pitch(p) for p in chord.split()], 0.9, 6 * STEP))
        booms.append((t(b, s), 0.8))
    b = at["montage"]                                       # the figure at its fullest, then every sixteenth
    figure(b, 4, 0.75, low=0.35)
    harmony(b, "D Bm G A D", 0.32, 0.58, pulse=4, counter=0.25, descant=0.4)
    booms += [(t(b), 0.8), (t(b + 2), 0.6)]
    quick = [t(b + 4, s) for s in range(16)] + [t(at["breath"], s) for s in range(8)]
    for i, q in enumerate(quick):
        out["voice"].append((q, pitch(HOOK[i % 6]), 0.75 + 0.25 * i / (len(quick) - 1), STEP * 0.85))
        out["low"].append((q, pitch(HOOK[i % 6]) - 12, 0.35, STEP * 0.85))
    rushes.append((t(b + 3), t(at["breath"], 8), 1.0))
    b = at["title"]                                         # a breath of silence, then the title
    booms.append((t(b), 1.0))
    brass.append((t(b), [pitch(p) for p in TITLE.split()], 0.85, 2 * BAR))
    harmony(b, "D", 0.3, 0.0)
    out["choir"] += [(t(b + 1), pitch(p), 0.2, 2 * BAR + TAIL) for p in CHORDS["D"][1].split()]
    sung(b + 1, [(0, 0, "F#5", 3), (0, 3, "D5", 3), (0, 6, "A4", 2), (0, 8, "E5", 8), (1, 0, "D5", 16)], 0.32)
    out["voice"][-1] = out["voice"][-1][:3] + (out["voice"][-1][3] + TAIL,)
    silent = t(at["breath"], 8)
    pedal = [(0.0, 0.0), (t(at["wrote"]), 0.5), (t(at["strokes"]), 0.6), (t(at["woven"]), 0.75), (t(at["pixel"]), 0.85),
             (t(at["statements"]), 1.0), (silent - 0.05, 1.0), (silent, 0.0), (t(at["title"]), 0.0), (t(at["title"]) + 0.01, 1.0),
             (t(sum(n for _, n in FORM)) + TAIL, 0.0)]
    for k in out:                                           # no two voices land quite together
        out[k] = [(max(0.0, s + r.normal(0, 0.004)), p, float(np.clip(v + r.normal(0, 0.03), 0.05, 1)), h)
                  for s, p, v, h in out[k]]
    return out, booms, brass, rushes, pedal


def play(r):
    """Every instrument playing its part, dry. -> {part: stereo float at music.RATE}"""
    length = sum(n for _, n in FORM) * BAR + TAIL
    n, booms, brass, rushes, pedal = notes(r)
    when = np.arange(int(length * music.RATE)) / music.RATE
    return {"sung": music.voice(n["voice"], length, r) + 0.8 * music.voice(n["low"], length, r, vowel="o"),
            "held": music.choir(n["choir"], length, r) + music.choir(n["counter"], length, r, vowel="u", voices=2),
            "hum": music.hum(n["hum"], length),
            "low d": music.hum([(0.0, pitch("D1"), 1.0, length)], length) * np.interp(when, *zip(*pedal))[:, None],
            "drum": music.boom(booms, length, r), "brass": music.braam(brass, length, r),
            "air": music.rush(rushes, length, r), "keys": music.piano(n["piano"], length, r)}


DRY = {"sung": 1.0, "held": 0.8, "hum": 1.0, "low d": 1.2, "drum": 0.9, "brass": 1.3, "air": 1.0, "keys": 0.7}
WET = {"sung": 0.35, "held": 0.6, "drum": 0.5, "brass": 0.4, "air": 0.3, "keys": 0.5}     # what the hall gives back
HITS_ONLY = ("drum", "brass")                                                              # what the ride leaves alone


def ride():
    """How loud the band plays through the piece, the drum and the brass apart: held back at first, louder part by
    part up to its fullest, then cut off an eighth before the brass and silent while it blows; full again from the
    montage on. -> [(seconds, gain)]"""
    at, end = starts(), sum(n for _, n in FORM) * BAR + TAIL
    db = lambda d: 10 ** (d / 20)
    drop = at["statements"] * BAR - 2 * STEP
    return [(0.0, db(-8)), (at["strokes"] * BAR, db(-8)), (at["woven"] * BAR, db(-6)), (at["pixel"] * BAR, db(-4)),
            (drop - 0.008, 1.0), (drop, 0.0), (at["montage"] * BAR - 0.008, 0.0), (at["montage"] * BAR, 1.0), (end, 1.0)]


def mix(parts, r):
    """The parts in the hall, ridden, balanced and mastered."""
    when = np.arange(len(parts["drum"])) / music.RATE
    gain = np.interp(when, *zip(*ride()))[:, None]
    parts = {k: x if k in HITS_ONLY else x * gain for k, x in parts.items()}
    dry = sum(DRY[k] * x for k, x in parts.items())
    wet = music.hall(sum(WET[k] * parts[k] for k in WET), r)
    # held 2 dB under full scale: the AAC the film is encoded to overshoots its peaks by about a decibel
    return music.master(music.tone(dry + wet, PRESENCE), ceiling=-2.0)


def render(seed=1125):
    """The score, mastered: stereo float at music.RATE, the film's bars and TAIL seconds."""
    r = noise.rng(seed)
    return mix(play(r), r)


if __name__ == "__main__":
    out = Path(__file__).parent / ".scratch" / "score.wav"
    out.parent.mkdir(exist_ok=True)
    music.write(out, render())
    print(out, f"{sum(n for _, n in FORM) * BAR + TAIL:.1f}s")
