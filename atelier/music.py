"""Music, made the way the paintings are: every sound is built from what makes it.

A singing voice is a buzz from the vocal folds, rich in harmonics, shaped by the hollows of the throat and mouth.
Those ring at a few frequencies of their own, the formants, whatever note is sung, and they are what make an
'ah' an 'ah'. A singer falls into a note from a little above it, steadies, and lets a vibrato grow as the note
is held; breath goes with it, most at the start. A choir is a few such voices on each note, never quite
together. Under them a low hum holds the root. A piano note is a felt hammer thrown against strings tuned a hair
apart; the strings are stiff, so their overtones run a little sharp, and the highest die first. Low brass blown
hard buzzes: its tone flares open as the players hit the note and closes as they hold it. A struck skin booms and
sinks in pitch as it slackens, and a rush of air rises before a fall. A hall gives everything back from every
side a moment later, the highs dying first.

Times are in seconds, pitches MIDI numbers (60 is middle C), velocities 0 to 1, and sound float stereo (n, 2)
at RATE.
"""

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

from . import noise

RATE = 48000

# A sung vowel: the centres (Hz) of the resonances of throat and mouth that make it, for a woman's voice; a
# man's lie about a tenth lower. 'a' is the open vowel of the solo voice, fitted to a recording of one.
VOWELS = {"a": (550, 1300, 3300, 4200, 4950), "o": (450, 800, 2830, 3800, 4950), "u": (325, 700, 2700, 3800, 4950)}
WIDTHS = (120, 135, 180, 195, 210)                         # how broadly each one rings, Hz
KNOCK = signal.butter(2, (180, 2600), "bandpass", fs=RATE, output="sos")    # a piano hammer's knock
WOOD = signal.butter(2, (80, 900), "bandpass", fs=RATE, output="sos")       # and the soundboard's answer to it


def hz(p):
    return 440.0 * 2 ** ((np.asarray(p, np.float64) - 69) / 12)


def _pan(x, spread=0.5):
    """-1 left to 1 right, as an equal-power pair of gains."""
    a = (np.clip(x * spread, -1, 1) + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def _place(out, x, start):
    """Add a stereo (n, 2) sound into `out` at `start` seconds."""
    i = int(round(start * RATE))
    x = x[: max(0, len(out) - i)]
    out[i:i + len(x)] += x


def formants(f, vowel, p):
    """How the throat and mouth singing `vowel` at pitch p pass each frequency in f: resonators in a row,
    each passing everything below it, ringing at its centre and cutting above it."""
    g = np.ones_like(np.asarray(f, np.float64))
    for c, b in zip(VOWELS[vowel], WIDTHS):
        c *= 1 if p >= 62 else 0.9
        g = g * c * c / np.sqrt((c * c - f * f) ** 2 + (f * b) ** 2)
    return g


def _sing(p, v, held, r, vowel, fall=60.0, vib=20.0, attack=0.03, release=0.09, air=0.05, cents=0.0, bite=0.0):
    """One voice on one note, held `held` seconds: it falls into the pitch from `fall` cents above, steadies,
    and a vibrato of `vib` cents grows in as it holds. With `bite` the note is struck rather than sung: it
    starts on a consonant, and its upper harmonics die away quickly into a softer held vowel. -> mono"""
    n = int((held + 6 * release) * RATE)
    t = np.arange(n) / RATE
    wander = noise.smoothstep(0, 1, np.interp(t, np.linspace(0, t[-1], 12), r.uniform(0, 1, 12))) * 2 - 1
    pitch = cents + fall * np.exp(-t / 0.04) + 4 * wander \
        + vib * noise.smoothstep(0.12, 0.6, t) * np.sin(2 * np.pi * r.uniform(5.2, 6.1) * t + r.uniform(0, 7))
    f = hz(p) * 2 ** (pitch / 1200)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    tilt = 0.9 - 0.5 * v                                   # sung louder, the buzz is brighter
    K = np.arange(1, int(9000 / hz(p)) + 1)
    level = np.sqrt(((K ** -tilt * formants(K * hz(p), vowel, p)) ** 2).sum())
    x = np.zeros(n)
    for k in K:                                            # the harmonics, each passed by the formants as it moves
        fade = 0.3 + 0.7 * np.exp(-t * (1.5 + 0.8 * k) * bite)
        x += k ** -tilt * formants(k * f, vowel, p) * np.cos(k * phase) * fade
    x /= level
    # breath through the same mouth, strongest as the note starts
    hiss = np.fft.rfft(r.normal(0, 1, n))
    fr = np.fft.rfftfreq(n, 1 / RATE)
    hiss = np.fft.irfft(hiss * formants(fr, vowel, p) * noise.smoothstep(500, 2500, fr), n)
    hiss *= air / (hiss.std() + 1e-12) * (0.35 + np.exp(-t / 0.06))
    click = np.fft.irfft(np.fft.rfft(r.normal(0, 1, n)) * noise.smoothstep(1200, 2500, fr) * noise.smoothstep(9000, 6000, fr), n)
    click *= 0.3 * bite / (click.std() + 1e-12) * np.exp(-t / 0.008)          # the consonant
    env = noise.smoothstep(0, attack * (1 - 0.85 * bite), t) * np.exp(-np.maximum(t - held, 0) / release)
    return (x + hiss + click) * env * v


def voice(notes, length, r, vowel="a", spread=0.5, bite=0.8, air=0.05):
    """A solo voice: notes (start, pitch, velocity, held), each sung on its own, placed a little left or right
    by its pitch. -> stereo, `length` seconds"""
    out = np.zeros((int(length * RATE), 2))
    for start, p, v, held in notes:
        x = _sing(p, v, held, r, vowel, bite=bite, air=air)
        _place(out, x[:, None] * np.array(_pan((p - 71) / 8, spread)), start)
    return out


def choir(notes, length, r, vowel="o", voices=4):
    """A few voices on every note, each a little out of tune with the others, swelling in and dying away. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    for start, p, v, held in notes:
        for j in range(voices):
            x = _sing(p, v / np.sqrt(voices), held, r, vowel, fall=0, vib=r.uniform(10, 22),
                      attack=r.uniform(0.3, 0.6), release=0.35, air=0.03, cents=r.normal(0, 7))
            _place(out, x[:, None] * np.array(_pan(r.uniform(-1, 1), 0.8)), start + r.uniform(0, 0.06))
    return out


def _board(f):
    """How a piano's soundboard gives out each frequency: little of the lowest, more of the warm middle, less and
    less of the highest."""
    lf = np.log2(np.maximum(f, 1))
    return (f / 90) ** 2 / (1 + (f / 90) ** 2) / (1 + (f / 5000) ** 2) * (1 + 0.3 * np.exp(-0.5 * ((lf - np.log2(320)) / 0.9) ** 2))


def _strike(p, v, held, r):
    """One piano note, struck at velocity v and let go after `held` seconds. -> mono"""
    f0 = hz(p)
    stiff = 10 ** (-3.7 + 0.024 * (p - 60))
    n = np.arange(1, 81)
    fn = f0 * n * np.sqrt(1 + stiff * n * n)
    n, fn = n[fn < 12000], fn[fn < 12000]
    felt = 650 * (1 + 3 * v * v) * (f0 / 262) ** 0.35
    amp = np.abs(np.sin(np.pi * n / 8.5)) * n ** -0.6 / np.sqrt(1 + (fn / felt) ** 4) * _board(fn)
    die = 0.9 * 2 ** ((p - 60) / 15) + 0.4 * (fn / 1000) ** 2        # how fast each overtone dies, per second
    damper = p < 89                                                  # the top of the piano has no dampers
    length = held + (0.5 if damper else 4.0)
    t = np.arange(int(length * RATE), dtype=np.float32) / RATE
    after = np.maximum(t - held, 0)
    stop = np.exp(-after * 14 * (0.4 + 0.6 * noise.smoothstep(36, 60, p))) if damper else 1
    x = np.zeros(len(t), np.float32)
    strings = 3 if p > 40 else 2 if p > 28 else 1
    cents = r.normal(0, 0.9, strings)
    for lo in range(0, len(n), 16):                                  # a few overtones at a time, to spare memory
        a, f, d = amp[lo:lo + 16, None], fn[lo:lo + 16, None], die[lo:lo + 16, None]
        env = (a * (0.78 * np.exp(-d * t) + 0.22 * np.exp(-0.22 * d * t))).astype(np.float32)
        for c in cents:
            x += (env * np.sin((2 * np.pi * f * 2 ** (c / 1200)) * t)).sum(0) / strings
    x *= stop * noise.smoothstep(0, 0.002, t)
    rms = np.sqrt((amp ** 2).sum() / 2) + 1e-9
    knock = signal.sosfilt(KNOCK, r.normal(0, 1, int(0.05 * RATE))) * np.exp(-np.arange(int(0.05 * RATE)) / RATE / 0.006)
    x[:len(knock)] += (0.5 + 0.5 * v) * 0.08 * rms * knock / (knock.std() + 1e-9)
    wood = signal.sosfilt(WOOD, r.normal(0, 1, int(0.25 * RATE))) * np.exp(-np.arange(int(0.25 * RATE)) / RATE / 0.05)
    x[:len(wood)] += v * 0.05 * rms * wood / (wood.std() + 1e-9)
    return x * v ** 1.4 / rms


def piano(notes, length, r):
    """A grand piano: notes (start, pitch, velocity, held), each struck and damped when let go, low notes a
    little to the left and high ones to the right, as the player hears them. -> stereo, `length` seconds"""
    out = np.zeros((int(length * RATE), 2), np.float32)
    struck = {}
    for start, p, v, held in notes:
        key = (p, round(v * 32), round(held * 20))                  # a note struck again alike sounds alike
        if key not in struck:
            struck[key] = _strike(p, key[1] / 32, key[2] / 20, r)
        _place(out, struck[key][:, None] * np.array(_pan((p - 64) / 30, 0.8), np.float32), start)
    return out


def hum(notes, length):
    """The low hum under everything: a sine on the root, with a little of its octave and twelfth so that small
    speakers can follow it. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    for start, p, v, held in notes:
        t = np.arange(int((held + 0.6) * RATE)) / RATE
        w = 2 * np.pi * hz(p) * t
        x = (np.sin(w) + 0.2 * np.sin(2 * w) + 0.05 * np.sin(3 * w)) \
            * noise.smoothstep(0, 0.06, t) * noise.smoothstep(held + 0.5, held, t) * v
        _place(out, np.repeat(x[:, None], 2, 1) * 0.7, start)
    return out


def _saw(f):
    """A buzz: a sawtooth at frequency f (per sample), its corners rounded so that nothing sounds above what the
    samples can hold."""
    step = f / RATE
    ph = np.cumsum(step) % 1.0
    y = 2 * ph - 1
    lo = ph < step
    u = ph[lo] / step[lo]
    y[lo] -= 2 * u - u * u - 1
    hi = ph > 1 - step
    u = (ph[hi] - 1) / step[hi]
    y[hi] -= u * u + 2 * u + 1
    return y


def braam(hits, length, r, players=3):
    """Low brass blown hard on a chord: hits (start, pitches, velocity, held). Each note is a few players a hair
    apart, falling into it; their tone flares open on the attack and closes as they hold, and the buzz breaks
    up a little where it is pushed. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    block = 256
    for start, pitches, v, held in hits:
        n = int((held + 1.2) * RATE)
        t = np.arange(n) / RATE
        x = np.zeros((n, 2))
        for p in pitches:
            for _ in range(players):
                cents = r.normal(0, 7) - 40 * np.exp(-t / 0.06)
                s = _saw(hz(p) * 2 ** (cents / 1200)) * (1 if p > 30 else 0.7)
                x += s[:, None] * np.array(_pan(r.uniform(-1, 1), 0.6))
        x /= np.sqrt(players * len(pitches))
        bright = 180 + 2600 * (noise.smoothstep(0, 0.05, t) * np.exp(-t / 0.35)) + 500 * np.exp(-np.maximum(t - held, 0) / 0.3)
        y, zi = np.zeros_like(x), None
        for i in range(0, n, block):                     # a low-pass that follows the brightness, a block at a time
            sos = signal.butter(2, min(bright[i], 0.45 * RATE), "lowpass", fs=RATE, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2, 2))
            y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], axis=0, zi=zi)
        env = noise.smoothstep(0, 0.025, t) * (0.6 + 0.4 * np.exp(-t / 0.5)) * np.exp(-np.maximum(t - held, 0) / 0.45)
        _place(out, np.tanh(2.2 * y * env[:, None]) * v, start)
    return out


def boom(times, length, r):
    """A great drum struck at each time: a low knock that sinks in pitch as the skin slackens. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    t = np.arange(int(4 * RATE)) / RATE
    f = 38 + 80 * np.exp(-t / 0.07)
    body = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t / 1.1)
    knock = signal.sosfilt(signal.butter(2, 400, "lowpass", fs=RATE, output="sos"), r.normal(0, 1, len(t))) \
        * np.exp(-t / 0.015)
    x = (body + 0.6 * knock) * noise.smoothstep(0, 0.002, t)
    for start, v in times:
        _place(out, np.repeat(x[:, None] * v, 2, 1), start)
    return out


def rush(spans, length, r):
    """Air rising: noise swept from low to high and swelling over each span (start, end, velocity), cut off
    at its end. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    block = 2048
    for start, end, v in spans:
        n = int((end - start) * RATE)
        x = r.normal(0, 1, (n + block, 2))
        y = np.zeros_like(x)
        win = np.hanning(block)[:, None]
        fr = np.fft.rfftfreq(block, 1 / RATE)
        for i in range(0, n, block // 2):
            u = min(1.0, i / n)
            centre = 500 * 16 ** u                           # from 500 Hz to 8 kHz
            band = np.exp(-0.5 * (np.log2(np.maximum(fr, 1) / centre) / 0.9) ** 2)[:, None]
            seg = np.fft.irfft(np.fft.rfft(x[i:i + block] * win, axis=0) * band, block, axis=0)
            y[i:i + block] += seg
        t = np.arange(n) / RATE
        swell = (t / t[-1]) ** 3 * noise.smoothstep(n, n - 0.004 * RATE, np.arange(n))
        _place(out, y[:n] / (y[:n].std() + 1e-12) * swell[:, None] * 0.12 * v, start)
    return out


def hall(x, r, decay=2.8):
    """The room's answer to x, and only that: thickening over the first moments and dying in about `decay`
    seconds, the lows lasting longest. -> stereo, the length of x"""
    n = int(decay * 1.6 * RATE)
    t = np.arange(n) / RATE
    fr = np.fft.rfftfreq(n, 1 / RATE)
    lf = np.log2(np.maximum(fr, 20))
    ir = np.zeros((2, n))
    for c in range(2):
        w = np.fft.rfft(r.normal(0, 1, n))
        for lo, hi, last in ((20, 250, 1.25), (250, 1000, 1.1), (1000, 3000, 0.9), (3000, 7000, 0.6), (7000, 24000, 0.35)):
            band = noise.smoothstep(np.log2(lo) - 0.5, np.log2(lo) + 0.5, lf) * noise.smoothstep(np.log2(hi) + 0.5, np.log2(hi) - 0.5, lf)
            ir[c] += np.fft.irfft(w * band, n) * np.exp(-6.91 * t / (decay * last))
    ir *= noise.smoothstep(0.0, 0.06, t)
    for d, g in zip(np.sort(r.uniform(0.008, 0.07, 6)), (0.35, 0.3, 0.26, 0.22, 0.18, 0.15)):   # the nearest walls
        ir[r.integers(0, 2), int(d * RATE)] += g * np.abs(ir).max()
    ir = np.pad(ir, ((0, 0), (int(0.02 * RATE), 0)))[:, :n]  # the hall answers after a moment
    ir /= np.sqrt((ir * ir).sum(1, keepdims=True))
    return np.stack([signal.oaconvolve(x.mean(1), ir[c])[: len(x)] for c in range(2)], 1)


def tone(x, curve):
    """x with its balance changed: curve [(Hz, dB)], a gain at each frequency, drawn straight between them on a
    scale of octaves. Done on the whole at once, so nothing is moved in time. -> the same shape"""
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    at, db = zip(*curve)
    g = 10 ** (np.interp(np.log2(np.maximum(f, 1)), np.log2(at), db) / 20)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * g[:, None], len(x), axis=0)


_K = (([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]),
      ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]))


def loudness(x):
    """Integrated loudness in LUFS, as broadcasters measure it (ITU-R BS.1770: K-weighted, gated)."""
    for b, a in _K:
        x = signal.lfilter(b, a, x, axis=0)
    c = np.concatenate([[0], np.cumsum((x * x).sum(1))])
    block, hop = int(0.4 * RATE), int(0.1 * RATE)
    i = np.arange(0, len(x) - block, hop)
    ms = (c[i + block] - c[i]) / block
    lufs = lambda m: -0.691 + 10 * np.log10(np.maximum(m, 1e-12))
    ms = ms[lufs(ms) > -70]
    return lufs(ms[lufs(ms) > lufs(ms.mean()) - 10].mean())


def master(x, target=-16.0, ceiling=-1.0):
    """Bring the mix to a loudness, holding its peaks under `ceiling` dBFS with a limiter that sees them
    coming: the gain starts down a few milliseconds before a peak and comes back over a few dozen after."""
    x = x * 10 ** ((target - loudness(x)) / 20)
    lim = 10 ** (ceiling / 20)
    g = np.minimum(1, lim / np.maximum(np.abs(x).max(1), 1e-9))
    for reach in (0.004, 0.03):                              # each average stays inside the plateau before it
        n = int(reach * RATE)
        g = ndimage.uniform_filter1d(ndimage.minimum_filter1d(g, 2 * n + 1), n + 1)
    return np.clip(x * (g * noise.smoothstep(0, 0.01 * RATE, np.arange(len(x))))[:, None], -lim, lim)


def write(path, x):
    wavfile.write(path, RATE, x.astype(np.float32))


if __name__ == "__main__":
    r = noise.rng(0)
    spectrum = lambda x: (np.fft.rfftfreq(len(x), 1 / RATE), np.abs(np.fft.rfft(x * np.hanning(len(x)))))
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    a = voice([(0.0, 69, 0.7, 1.0)], 2.0, r)[:, 0]
    fr, s = spectrum(a[int(0.2 * RATE):int(0.45 * RATE)])
    near = lambda f0: s[(fr > f0 * 0.97) & (fr < f0 * 1.03)].max()
    peak = lambda lo, hi: fr[(fr > lo) & (fr < hi)][s[(fr > lo) & (fr < hi)].argmax()]
    assert abs(peak(300, 600) - 440) < 6, "an A sung at 440 Hz"
    assert near(880) > 3 * near(2200), "and it sounds 'ah': the harmonics near its formants carry it"
    fr, s = spectrum(a[: int(0.03 * RATE)])
    assert peak(300, 600) > 445, "falling into the note from above"
    assert np.abs(a[int(1.6 * RATE):]).max() < 0.05 * np.abs(a).max(), "and letting it go"
    c = choir([(0.0, 62, 0.6, 1.5)], 3.0, r)
    assert np.abs(c[: int(0.05 * RATE)]).max() < 0.2 * np.abs(c).max(), "a choir swells in"
    p = piano([(0.0, 69, 0.6, 1.5)], 2.5, r)[:, 0]
    fr, s = spectrum(p[int(0.05 * RATE):int(0.55 * RATE)])
    assert abs(peak(400, 480) - 440) < 2, "an A struck at 440 Hz"
    assert peak(4400 * 0.99, 4400 * 1.03) > 4400 * 1.002, "its tenth overtone runs sharp, the string being stiff"
    assert rms(p[int(1.9 * RATE):]) < 0.05 * rms(p[: int(0.25 * RATE)]), "and the damper stops it"
    br = braam([(0.0, [38, 45, 50], 1.0, 1.0)], 2.5, r)[:, 0]
    bright = lambda x: (lambda f, s: (f * s).sum() / s.sum())(*spectrum(x))
    assert bright(br[int(0.03 * RATE):int(0.12 * RATE)]) > 1.5 * bright(br[int(0.8 * RATE):int(1.0 * RATE)]), \
        "brass flares bright as it hits the note and darkens as it holds"
    b = boom([(0.0, 1.0)], 3.0, r)[:, 0]
    fr, s = spectrum(b[int(0.3 * RATE):int(1.3 * RATE)])
    assert 30 < fr[s.argmax()] < 50, "a drum sinks to its low note"
    w = rush([(0.0, 1.0, 1.0)], 1.5, r)[:, 0]
    assert np.abs(w[int(0.8 * RATE):RATE]).std() > 4 * np.abs(w[:int(0.3 * RATE)]).std() and not w[int(1.01 * RATE):].any(), \
        "a rush swells and is cut off"
    mix = np.concatenate([np.repeat(a[:, None], 2, 1), c])
    m = master(mix + 0.3 * hall(mix, r))
    assert abs(loudness(m) + 16) < 0.6 and np.abs(m).max() <= 10 ** (-1 / 20) + 1e-6, "mastered loud enough, never clipped"
    print("ok")
