"""Music, made the way the paintings are: every sound is built from what makes it.

A piano note is a felt hammer thrown against two or three strings tuned a hair apart. The strings are stiff, so
their overtones run a little sharp, the higher the sharper; where the hammer strikes and how hard its felt is
squeezed decide which overtones sound and how brightly. The highest die first, and the note dies in two stages,
quickly while the strings drive the soundboard together and slowly once they drift apart, until the damper
falls. A bowed string is caught and let slip over and over, a buzz rich in overtones that the wooden body
colours with resonances of its own; the player's hand rocks the note in a vibrato, and a section is a few
players never quite together. A plucked string rings once and fades. A hall gives everything back from every
side a moment later, the highs dying first.

Times are in seconds, pitches MIDI numbers (60 is middle C), velocities 0 to 1 (or a pair, from and to, for a
bowed note that swells or fades), and sound float stereo (n, 2) at RATE.
"""

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

from . import noise

RATE = 48000

# The bodies of the bowed instruments: the resonances of their wood and air, (Hz, how broad in Hz, how much they
# add), and the frequency above which their tone falls away.
BODIES = {
    "violin": (((280, 70, 2.0), (470, 120, 1.4), (1100, 500, 0.8), (2800, 1600, 0.7)), 4200),
    "viola": (((230, 60, 2.0), (400, 110, 1.4), (900, 450, 0.8), (2300, 1300, 0.6)), 3400),
    "cello": (((110, 40, 1.2), (220, 60, 1.4), (420, 140, 1.0), (1300, 800, 0.6)), 2600),
    "bass": (((60, 24, 1.0), (120, 40, 1.2), (260, 90, 0.8), (700, 500, 0.5)), 1500),
}
VIBRATO = {"violin": 16, "viola": 14, "cello": 12, "bass": 6}      # cents, at its widest
KNOCK = signal.butter(2, (180, 2600), "bandpass", fs=RATE, output="sos")    # the hammer's knock
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


def _board(f):
    """How the soundboard gives out each frequency: little of the lowest, more of the warm middle, less and less
    of the highest."""
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


def _saw(f):
    """The buzz of a bowed string: a sawtooth at frequency f (per sample), its corners rounded so that nothing
    sounds above what the samples can hold."""
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


def _bow(p, v, held, r, depth, attack, release):
    """One player on one note, held `held` seconds: it settles into the pitch, a vibrato grows as it holds, the
    bow scrapes a little as it bites. -> mono"""
    n = int((held + 5 * release) * RATE)
    t = np.arange(n) / RATE
    wander = np.interp(t, np.linspace(0, t[-1], 8), r.uniform(-1, 1, 8))
    cents = r.normal(0, 5) + 3 * wander - 12 * np.exp(-t / 0.05) \
        + depth * r.uniform(0.7, 1.1) * noise.smoothstep(0.15, 0.7, t) * np.sin(2 * np.pi * r.uniform(5.0, 6.2) * t + r.uniform(0, 7))
    x = _saw(hz(p) * 2 ** (cents / 1200))
    scrape = signal.sosfilt(signal.butter(2, (1500, 7000), "bandpass", fs=RATE, output="sos"), r.normal(0, 1, n))
    x += scrape * (0.015 + 0.05 * np.exp(-t / 0.06))
    v0, v1 = v if isinstance(v, tuple) else (v, v)
    level = v0 + (v1 - v0) * np.clip(t / max(held, 1e-3), 0, 1)
    return x * level * noise.smoothstep(0, attack, t) * np.exp(-np.maximum(t - held, 0) / release)


def _body(x, kind):
    """A sound passed through the body of a `kind`: its own sound and that of its resonances, the tone rolled
    off above. -> the same shape"""
    modes, top = BODIES[kind]
    y = x.copy()
    for c, width, gain in modes:
        b, a = signal.iirpeak(c, c / width, fs=RATE)
        y += gain * signal.lfilter(b, a, x, axis=0)
    return signal.sosfilt(signal.butter(4, top, "lowpass", fs=RATE, output="sos"), y, axis=0) / 3


def strings(notes, length, r, kind="violin", players=6, seat=0.0, attack=0.25, release=0.3):
    """A section of `players` bowing every note together, never quite together, seated about `seat` (-1 left to
    1 right). -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    for start, p, v, held in notes:
        for _ in range(players):
            x = _bow(p, v, held, r, VIBRATO[kind], attack * r.uniform(0.8, 1.25), release) / np.sqrt(players)
            _place(out, x[:, None] * np.array(_pan(seat + r.uniform(-0.3, 0.3), 1)), start + r.uniform(0, 0.04))
    return _body(out, kind)


def _pluck(p, v, r):
    """A string plucked with a finger: overtones that die the faster the higher. -> mono"""
    f0 = hz(p)
    n = np.arange(1, int(7000 / f0) + 1)
    fn = f0 * n
    amp = np.abs(np.sin(np.pi * n * r.uniform(0.15, 0.22))) / n ** 1.5
    die = 3.0 * 2 ** ((p - 57) / 18) + 1.5 * (fn / 1000) ** 1.5
    t = np.arange(int(min(2.5, 8 / die[0]) * RATE)) / RATE
    x = (amp[:, None] * np.exp(-die[:, None] * t) * np.sin(2 * np.pi * fn[:, None] * t)).sum(0)
    return x * noise.smoothstep(0, 0.003, t) * v / np.sqrt((amp ** 2).sum() / 2)


def pizzicato(notes, length, r, kind="violin", players=4, seat=0.0):
    """A section plucking every note, never quite together. -> stereo"""
    out = np.zeros((int(length * RATE), 2))
    for start, p, v, _ in notes:
        for _ in range(players):
            x = _pluck(p * 1.0 + r.normal(0, 0.05), v, r) / np.sqrt(players)
            _place(out, x[:, None] * np.array(_pan(seat + r.uniform(-0.3, 0.3), 1)), start + r.uniform(0, 0.025))
    return _body(out, kind)


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
    a = piano([(0.0, 69, 0.6, 1.5)], 2.5, r)[:, 0]
    fr, s = spectrum(a[int(0.05 * RATE):int(0.55 * RATE)])
    peak = lambda lo, hi: fr[(fr > lo) & (fr < hi)][s[(fr > lo) & (fr < hi)].argmax()]
    assert abs(peak(400, 480) - 440) < 2, "an A struck at 440 Hz"
    assert peak(4400 * 0.99, 4400 * 1.03) > 4400 * 1.002, "its tenth overtone runs sharp, the string being stiff"
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    assert rms(a[int(1.2 * RATE):int(1.4 * RATE)]) < 0.6 * rms(a[int(0.05 * RATE):int(0.25 * RATE)]), "it dies away"
    assert rms(a[int(1.9 * RATE):]) < 0.05 * rms(a[: int(0.25 * RATE)]), "and the damper stops it"
    soft, hard = (piano([(0.0, 60, v, 1.0)], 1.2, r)[:, 0] for v in (0.2, 0.9))
    bright = lambda x: (lambda f, s: (f * s).sum() / s.sum())(*spectrum(x[: int(0.3 * RATE)]))
    assert bright(hard) > 1.3 * bright(soft), "struck harder, the felt is harder and the note brighter"
    b = strings([(0.0, 69, 0.6, 1.5)], 2.5, r, "violin")[:, 0]
    assert rms(b[: int(0.03 * RATE)]) < 0.3 * rms(b[int(0.6 * RATE):int(1.2 * RATE)]), "a bowed note swells in"
    fr, s = spectrum(b[int(0.6 * RATE):int(1.4 * RATE)])
    assert abs(peak(400, 480) - 440) < 6, "a violin's A at 440 Hz"
    c = pizzicato([(0.0, 50, 0.7, 0.2)], 2.0, r, "cello")[:, 0]
    assert rms(c[int(1.2 * RATE):]) < 0.15 * rms(c[: int(0.2 * RATE)]), "a plucked note fades"
    mix = np.stack([a, a], 1) + np.concatenate([np.stack([b, b], 1)[: len(a)], np.zeros((max(0, len(a) - len(b)), 2))])
    m = master(mix + 0.3 * hall(mix, r))
    assert abs(loudness(m) + 16) < 0.6 and np.abs(m).max() <= 10 ** (-1 / 20) + 1e-6, "mastered loud enough, never clipped"
    print("ok")
