"""The film's score: one theme for piano and strings, played as the museum is walked, a little differently in each
room, as one painter paints in many manners.

    uv run score.py      writes .scratch/score.wav, to listen to on its own

The film (timeline.py) takes its bars from FORM: three to open on, six for the first room, where a painting is
painted, four for each of the others and four for the museum's first page; then the last chord rings on for TAIL
seconds. It is a slow waltz in D major. The theme is a long note and three quarter notes rising, twice, over the
chords IV V iii vi. The piano opens alone and plays it in the open air, and the low strings come in under the
painting as it is painted; in the garden the cellos sing it an octave lower; by the water the piano plays it again
over ripples, under a high halo of violins; in the workshop the strings pluck under it, then take up their bows
and rise with it step by step into the room of colour, where the whole orchestra plays it at its height; and on
the first page the piano is alone again and comes home to D.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from atelier import music, noise

TEMPO, METER = 90, 3        # beats a minute, and a bar: a beat is 20 frames of the film at 30 fps, a bar two seconds
BEAT = 60 / TEMPO
BAR = METER * BEAT
TAIL = 2.0                  # seconds the last chord rings on after the last bar
FORM = (("intro", 3), ("Open Air", 6), ("The Garden", 4), ("Paper and Water", 4), ("The Workshop", 4),
        ("Colour Itself", 4), ("page", 4))
STEPS = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}

# A chord a bar; "X+Y" changes to Y on the bar's last beat.
CHORDS = ("D9 G/D D9  Gmaj7 A F#m7 Bm7 Em7 A7sus+A7  Gmaj7 A Dmaj7 Dmaj7  Gmaj7 A F#m7 Bm7  Em7 F#m7 Gmaj7 A7  "
          "Dmaj7 Bm7 Gmaj7 Asus+A  Gmaj7 A D9 D9").split()
SPREAD = {   # each chord as the piano spreads it, wide open: its bass and four notes above, up to under the tune
    "D9": "D2 A2 F#3 E4 A4", "G/D": "D2 A2 G3 B3 F#4", "Gmaj7": "G2 D3 B3 F#4 A4", "A": "A2 E3 C#4 E4 B4",
    "F#m7": "F#2 C#3 A3 E4 A4", "Bm7": "B1 F#2 D3 A3 F#4", "Em7": "E2 B2 G3 D4 F#4", "A7": "A2 E3 C#4 E4 G4",
    "A7sus": "A2 E3 D4 E4 G4", "Dmaj7": "D2 A2 F#3 C#4 E4", "Asus": "A2 E3 D4 E4 A4",
}
RIPPLE = {"Gmaj7": "D4 G4 A4 B4", "A": "C#4 E4 A4 B4", "F#m7": "C#4 E4 F#4 A4", "Bm7": "B3 D4 F#4 A4"}   # by the water
PLUCKED = {"Em7": "G4 B4 D5", "F#m7": "A4 C#5 E5", "Gmaj7": "B4 D5 F#5", "A7": "C#5 E5 G5"}            # in the workshop
# The tune, a bar at a time: notes as pitch:beats (a beat if not given), "-" a rest.
TUNE = ("-:3 | -:3 | -:2 A4 |F#5:3 | E5 F#5 A5 | C#5:3 | D5 E5 F#5 | G5:2 A5 | E5:3 | B5:3 | A5 B5 C#6 | A5:2 F#5 | "
        "E5 D5:2 | F#5:3 | E5 F#5 A5 | C#5:3 | D5 E5 F#5 | G5:2 F#5 | A5:2 E5 | B5:2 F#5 | C#6:3 | D6:2 C#6 | "
        "B5:2 A5 | G5 A5 B5 | A5:3 | F#5:3 | E5 F#5 A5 | F#5:2 E5 | D5:3").split("|")


def pitch(name):
    """'F#5' -> 78"""
    return 12 * (int(name[-1]) + 1) + STEPS[name[:-1]]


def chord(b, beat):
    """The chord of bar b at a beat of it."""
    both = CHORDS[b].split("+")
    return both[-1] if beat >= METER - 1 else both[0]


def tune():
    """The theme, [(bar, beat, pitch, beats)]."""
    out = []
    for b, bar in enumerate(TUNE):
        beat = 0
        for note in bar.split():
            name, _, n = note.partition(":")
            n = int(n or 1)
            if name != "-":
                out.append((b, beat, pitch(name), n))
            beat += n
        assert beat == METER, f"bar {b} of the tune has {beat} beats"
    return out


def sections():
    """Each part of the piece and the bars it takes: {name: range}"""
    out, b = {}, 0
    for name, n in FORM:
        out[name] = range(b, b + n)
        b += n
    return out


def notes(r):
    """Every instrument's notes. -> {instrument: [(start, pitch, velocity, held)]}"""
    out, part = defaultdict(list), sections()
    total = len(CHORDS) * BAR + TAIL
    assert len(CHORDS) == len(TUNE) == sum(n for _, n in FORM), "the chords, the tune and the form agree"
    ring = lambda b, beat: (METER - beat) * BEAT + (TAIL if b == len(CHORDS) - 1 else 0.06)    # till the pedal lifts
    for name, bars in part.items():
        for k, b in enumerate(bars):
            at = b * BAR
            u = k / max(len(bars) - 1, 1)                     # how far through the part
            if name in ("intro", "Open Air", "The Garden", "Colour Itself", "page") or b == part["The Workshop"][-1]:
                v = {"intro": 0.2 + 0.08 * u, "Open Air": 0.3 + 0.05 * u, "The Garden": 0.24, "The Workshop": 0.36,
                     "Colour Itself": 0.32, "page": 0.27 - 0.07 * u}[name]
                for e, i in enumerate((0, 1, 2, 3, 4, 3)):    # eighths up the chord and back
                    beat = e / 2
                    c = chord(b, beat)
                    held = ring(b, beat) if c == chord(b, METER - 1) else (METER - 1 - beat) * BEAT + 0.06
                    if c != chord(b, 0):                          # a chord that comes on the last beat comes in by its third
                        i = 2 + 2 * (e % 2)
                    p = pitch(SPREAD[c].split()[i])
                    out["piano"].append((at + beat * BEAT, p, v + (0.06 if e == 0 else 0.02 * (i == 4)), held))
            if name == "Paper and Water":                     # sixteenths rippling up and down, the bass soft under them
                ripple = [pitch(n) for n in RIPPLE[CHORDS[b]].split()]
                for s, i in enumerate((0, 1, 2, 3, 2, 1) * 2):
                    out["piano"].append((at + s * BEAT / 4, ripple[i], 0.2 + 0.04 * (s % 4 == 0), ring(b, s / 4)))
                out["piano"].append((at, pitch(SPREAD[CHORDS[b]].split()[0]), 0.26, ring(b, 0)))
            if name == "The Workshop" and b < bars[-1]:       # plucked: the bass on the first beat, the chord on the others
                c = CHORDS[b]
                root = pitch(SPREAD[c].split()[0])
                out["pizz low"] += [(at, root, 0.55, 0.3), (at, root + 12, 0.4, 0.3)]
                for beat in (1, 2):
                    out["pizz high"] += [(at + beat * BEAT, pitch(p), 0.42 + 0.04 * (beat == 1), 0.2) for p in PLUCKED[c].split()]
    for b, beat, p, n in tune():
        at, held = b * BAR + beat * BEAT, max(n * BEAT, ring(b, beat)) + 0.02
        name = next(k for k, bars in part.items() if b in bars)
        if name == "The Garden":                              # the cellos sing it, an octave down
            out["cello tune"].append((at, p - 12, 1.0, n * BEAT + 0.05))
            continue
        v = {"intro": 0.45, "Open Air": 0.55, "Paper and Water": 0.5, "The Workshop": 0.56, "Colour Itself": 0.52,
             "page": 0.5 - 0.03 * (b - part["page"][0])}[name]
        out["piano"].append((at, p, v, held))
        if name == "Colour Itself":                           # in octaves, and the violins with it, the violas a sixth under
            out["piano"].append((at, p - 12, v - 0.08, held))
            out["violin tune"].append((at, p, 1.3, n * BEAT + 0.05))
            out["viola tune"].append((at, p - 8 - (p % 12 in (1, 4, 6, 11)), 0.9, n * BEAT + 0.05))
        elif n >= 2:                                          # a long note, with a note of the chord a third or so under it
            below = [q for q in (pitch(x) + 12 * o for x in SPREAD[chord(b, beat)].split()[1:] for o in (1, 2)) if 2 < p - q <= 9]
            if below:
                out["piano"].append((at, max(below), v - 0.14, held))
    # the strings under it, part by part
    bar = lambda b: b * BAR
    a, z = part["Open Air"][-2], part["Open Air"][-1]           # the painting being painted
    out["cellos"] += [(bar(a), pitch("E3"), (0.08, 0.3), BAR + 0.1), (bar(z), pitch("A2"), (0.3, 0.42), BAR + 0.1)]
    out["violas"] += [(bar(a), pitch("B3"), (0.06, 0.25), BAR + 0.1), (bar(a), pitch("G4"), (0.06, 0.25), BAR + 0.1),
                      (bar(z), pitch("C#4"), (0.25, 0.35), BAR + 0.1), (bar(z), pitch("G4"), (0.25, 0.35), BAR + 0.1)]
    for b, low in zip(part["The Garden"], ("F#5", "E5", "F#5", "E5")):          # a shimmer high over the cellos
        out["violins"] += [(bar(b), pitch(low), 0.16, BAR + 0.1), (bar(b), pitch("A5"), 0.16, BAR + 0.1)]
    w = part["Paper and Water"]
    out["violins"] += [(bar(w[0]), pitch("A5"), (0.04, 0.1), len(w) * BAR), (bar(w[0]), pitch("E6"), (0.03, 0.08), len(w) * BAR)]
    s = part["The Workshop"]                                     # bows taken up, rising into the room of colour
    for b, (vn, va, vc, cb), v in zip(s[-2:], (("B5", "D5", "G3", "G2"), ("C#6", "E5", "A3", "A2")), ((0.12, 0.4), (0.4, 0.7))):
        out["violins"].append((bar(b), pitch(vn), v, BAR + 0.05))
        out["violas"].append((bar(b), pitch(va), v, BAR + 0.05))
        out["cellos"].append((bar(b), pitch(vc), v, BAR + 0.05))
        out["basses"].append((bar(b), pitch(cb), v, BAR + 0.05))
    for b, (vc, cb) in zip(part["Colour Itself"], (("D3", "D2"), ("B2", "B1"), ("G2", "G1"), ("A2", "A1"))):
        out["cellos"].append((bar(b), pitch(vc), 0.75, BAR + 0.05))
        out["basses"].append((bar(b), pitch(cb), 0.38, BAR + 0.05))
    p = part["page"]                                             # the strings fade away under the first page
    out["violins"] += [(bar(p[0]), pitch("A5"), (0.3, 0.02), 2 * BAR), (bar(p[0]), pitch("D6"), (0.25, 0.02), 2 * BAR)]
    out["violas"].append((bar(p[0]), pitch("F#4"), (0.3, 0.02), 2 * BAR))
    out["cellos"].append((bar(p[0]), pitch("G2"), (0.35, 0.02), 2 * BAR))
    out["piano"] = [(max(0.0, t + r.normal(0, 0.006)), q, float(np.clip(v + r.normal(0, 0.025), 0.05, 1)), h)
                    for t, q, v, h in out["piano"] if t < total]
    return out


def play(r):
    """Every instrument playing its notes, dry. -> {instrument: stereo float at music.RATE}"""
    length = len(CHORDS) * BAR + TAIL
    n = notes(r)
    bow = lambda kind, players, seat: music.strings(n[f"{kind}s"], length, r, kind, players, seat) \
        + music.strings(n[f"{kind} tune"], length, r, kind, players, seat, attack=0.09)     # a tune is bowed quicker
    return {"piano": music.piano(n["piano"], length, r), "violins": bow("violin", 8, -0.6),
            "violas": bow("viola", 4, 0.05), "cellos": bow("cello", 4, 0.5),
            "basses": music.strings(n["basses"], length, r, "bass", 2, 0.75),
            "plucked": music.pizzicato(n["pizz high"], length, r, "violin", 4, -0.4) + music.pizzicato(n["pizz low"], length, r, "cello", 3, 0.5)}


def render(seed=1125):
    """The score, mastered: stereo float at music.RATE, the film's bars and TAIL seconds."""
    r = noise.rng(seed)
    parts = play(r)
    piano, plucked = parts.pop("piano"), parts.pop("plucked")
    bowed = sum(parts.values())
    mix = piano + bowed + 0.6 * plucked + music.hall(0.32 * piano + 0.6 * bowed + 0.4 * plucked, r, decay=2.4)
    mix *= noise.smoothstep(len(mix), len(mix) - 1.2 * music.RATE, np.arange(len(mix)))[:, None]   # away with the picture
    # held 2 dB under full scale: the AAC the film is encoded to overshoots its peaks by about a decibel
    return music.master(mix, ceiling=-2.0)


if __name__ == "__main__":
    out = Path(__file__).parent / ".scratch" / "score.wav"
    out.parent.mkdir(exist_ok=True)
    music.write(out, render())
    print(out, f"{len(CHORDS) * BAR + TAIL:.1f}s")
