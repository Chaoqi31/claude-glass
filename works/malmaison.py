"""A Flamed Tulip. Stipple engraving printed in colour from one plate and finished by hand in watercolour, on wove paper.

Between 1802 and 1816 Redouté published Les Liliacées, nearly five hundred plates of lilies, irises,
amaryllis and tulips, and after it Les Roses. He painted each plant on vellum, and his engravers
translated the painting onto copper not in lines but in dots, pricked and flicked into the plate with
a point and a graver: close and deep where a form turns away from the light, sparse and shallow where
it faces it, so that nothing in the print has an outline. For every impression the printer inked the
one plate in several colours at once, dabbing each into its own part with a small stump of rag, the
poupée, and the colours ran a little into one another where they met. When the sheet was dry a
colourist went over it with a few transparent washes, for the brilliance the inks alone could not give.

This plate is a garden tulip of the kind Redouté drew at Malmaison: one flower opening, flamed crimson
on yellow, one tepal flaring and one turning back its tip, the dark anthers showing in the cup; a bud;
three leaves with waved margins, one arching over to show its paler underside; and the bulb in its
papery tunic, split down one side over the white scale, with its tuft of roots. The drawing was engraved
as some hundreds of thousands of dots, following the form and gathering into rows along the veins of the
leaves and the tunic; the plate was inked green, crimson, yellow, bistre and black in their places and
pulled once; the washes went on afterwards, gamboge in the flower, sap green and verdigris in the
leaves, sienna on the bulb, each with its own edge, and over the gamboge the flames, stroke by stroke
with a fine brush along the veins. The copper's bevel is pressed into the sheet round the image.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from atelier import brush, lettering, noise, plate, relief, wash
from atelier import watercolour as wc
from atelier.color import lin, pigment, saturate
from atelier.paper import Sheet

TITLE = "A Flamed Tulip"
DATE = "2026"
MEDIUM = ("Stipple engraving printed in colour from one plate inked à la poupée, finished by hand in "
          "watercolour, on wove paper")
AFTER = ("Pierre-Joseph Redouté, Les Liliacées, Paris, 1802–1816, and Les Roses, Paris, 1817–1824: stipple "
         "engravings printed in colour and finished by hand in watercolour")
ROOM = "The Workshop"
YEAR = 1805
PLACE = "Paris"
REGION = "Europe"
NOTE = ("A garden tulip from its bulb to its flower, open and flamed crimson on yellow, with a bud and three "
        "leaves turning to show both faces: every form is built of engraved dots of coloured ink, washed over "
        "by hand.")

H, W = 3600, 2480                  # the sheet on the wall, as photographed
EDGE = 60                          # wall showing round the sheet
Y0, X0 = 210, 214                  # where the copper's corner lands on the sheet
PH, PW = 3100, 2052                # the copper
G = 2.0                            # the engraver's closest spacing between dots, px
SUN = np.array([-0.45, -0.6, 0.66]) / np.linalg.norm([-0.45, -0.6, 0.66])
NAME = lettering.CUT / "malmaison_name.png"
SIGN = lettering.CUT / "malmaison_pinx.png"
SNELL = "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc"

INKS = dict(   # dabbed into the plate, each into its own part
    green=pigment("#467a5f"), crimson=pigment("#b02a44"), yellow=pigment("#c4952c"), bistre=pigment("#86593c"),
    rose=pigment("#9b4b55"), grey=pigment("#6c7366"), violet=pigment("#4b2b42"), black=pigment("#2b2521"))
PAINT = dict(  # the colourist's box
    gamboge=pigment("#f3c22b"), carmine=pigment("#cf2449"), vermilion=pigment("#e4572e"), sap=pigment("#8aab4a"),
    verdigris=pigment("#5c9e8c"), indigo=pigment("#3d4a66"), sienna=pigment("#c98f4c"), burnt=pigment("#a65a36"),
    madder=pigment("#d77a8c"), violet=pigment("#6a4a7a"), cobalt=pigment("#7a9cc8"))


# ---- the drawing: every surface of the plant is a ribbon in space

class Part:
    """One surface as a ribbon: a spine C sampled every half pixel, the direction Wd its width lies in,
    its half-width w, and `cup`, how far its cross-section bows out along its normal (as a fraction of
    the half-width). With `out` the normal is turned to point that way: the outside of a petal. `rip`
    holds how far each margin waves out of the surface (as a fraction of the half-width), and `drip`
    its slope per px along the spine."""

    def __init__(self, kind, C, Wd, w, cup, out=None, rows=30, **attrs):
        C = np.asarray(C, np.float64)
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        Wd = np.asarray(Wd, np.float64) * np.ones_like(C)
        Wd = Wd - (Wd * T).sum(1, keepdims=True) * T
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True) + 1e-9
        n = np.cross(T, Wd)
        if out is not None:
            n *= np.where((n * (np.asarray(out) * np.ones_like(C))).sum(1, keepdims=True) < 0, -1.0, 1.0)
        self.kind, self.C, self.T, self.Wd, self.n, self.rows = kind, C, T, Wd, n, rows
        self.w = np.broadcast_to(np.asarray(w, np.float64), len(C)).copy()
        self.cup = np.broadcast_to(np.asarray(cup, np.float64), len(C)).copy()
        self.rip, self.drip = np.zeros((len(C), 2)), np.zeros((len(C), 2))
        self.__dict__.update(attrs)


def resample(C, step=0.5):
    """Rows (x, y, z, ...) resampled every `step` px of their length in space."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C[:, :3], axis=0), axis=1))])
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, C[:, j]) for j in range(C.shape[1])], 1)


def lanceolate(u, base=0.04):
    """A tulip leaf's outline: sheathing at the foot, broadest a third of the way up, drawn to a point."""
    return np.sin(np.pi * (base + (1 - base) * u) ** 0.72) ** 0.85


def obovate(u, base=0.16, a=0.5, b=0.42):
    """A tepal's outline: a narrow claw, broadest past the middle, a rounded tip drawn to a point."""
    p = (base + u) ** a * (1 - u) ** b
    return p / p.max()


def blade(P, width, prof, r, cup=-0.28, kind="leaf", rows=26, ripple=None, **kw):
    """A flat organ, a leaf or a scrap of tunic, from control rows (x, y, z, twist): the twist turns its
    width about the spine, so at pi/2 it is edge-on and past it we see its underside. `ripple`
    (amplitude, wavelength px) waves each margin in and out of the surface, each to its own beat, as
    a tulip leaf's margins do."""
    C = brush.path(np.asarray(P, np.float64), 0.5)
    u = np.linspace(0, 1, len(C))
    t2 = np.gradient(C[:, :2], axis=0)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
    n2 = np.stack([-t2[:, 1], t2[:, 0], np.zeros(len(C))], 1)
    th = C[:, 3]
    Wd = n2 * np.cos(th)[:, None] + np.array([0, 0, 1.0]) * np.sin(th)[:, None]
    w = width * prof(u) * (1 + 0.02 * noise.line1d(len(C), 260, r))
    cup = cup * (1 + 0.4 * noise.line1d(len(C), 300, r))       # the blade undulates gently along its edge
    part = Part(kind, C[:, :3], Wd, w, cup, rows=rows, seed=int(r.integers(1 << 30)), **kw)
    if ripple:
        amp, lam = ripple
        env = np.sin(np.pi * u) ** 0.6 * noise.smoothstep(0.04, 0.2, u)
        for j in range(2):
            beat = np.cumsum(np.pi / (lam * (1 + 0.3 * noise.line1d(len(C), 300, r)))) + r.uniform(0, 2 * np.pi)
            part.rip[:, j] = amp * env * (1 + 0.35 * noise.line1d(len(C), 500, r)) * np.sin(beat)
        part.drip = np.gradient(part.rip, 0.5, axis=0)
    return part


def stalk(P, w0, w1, kind="stem", rows=6, cup=0.95, taper=1.0):
    """A stem or a root: a cylinder drawn through control rows (x, y, z), w0 thick at its foot and w1
    at its end; with `taper` above 1 it keeps its thickness low down and thins higher up."""
    C = brush.path(np.asarray(P, np.float64), 0.5)
    u = np.linspace(0, 1, len(C))
    t2 = np.gradient(C[:, :2], axis=0)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
    n2 = np.stack([-t2[:, 1], t2[:, 0], np.zeros(len(C))], 1)
    return Part(kind, C[:, :3], n2, w1 + (w0 - w1) * (1 - u) ** taper, cup, rows=rows)


def frame(base, nod, elev):
    """A flower's frame: local (x right, h up, z toward us) -> plate, the flower tipped `elev` toward
    us so we look into it, and leaning `nod` to the right."""
    ca, sa, cb, sb = np.cos(elev), np.sin(elev), np.cos(nod), np.sin(nod)

    def f(p, point=True):
        p = np.asarray(p, np.float64)
        x, y, z = p[..., 0], -p[..., 1], p[..., 2]
        y, z = y * ca + z * sa, z * ca - y * sa
        x, y = cb * x - sb * y, sb * x + cb * y
        q = np.stack([x, y, z], -1)
        return q + np.asarray(base, np.float64) if point else q
    return f


def petal(f, phi, L, wd, r, r0=20.0, R=50.0, bend=(1.1, 0.2, 0.05, 0.2), swing=0.0, twist=0.0, tip=0.42,
          cup=0.3, flat=0.0, kind="tepal", bud=False, rows=40, **kw):
    """A tepal in a flower's frame `f`, leaving the receptacle at azimuth `phi` (0 toward us), `L` long
    and `wd` wide. `bend` is the angle it makes with the flower's axis at its foot, a third and two
    thirds of the way up and at its tip: it leaves the receptacle nearly flat, rises, and opens or
    turns back toward the tip. `swing` carries it round sideways as it rises, `twist` turns its blade
    about its length, `tip` draws its end to a point and `flat` opens its cup toward the tip. A bud's
    tepals swell out to `R` and close back in to the axis."""
    u = np.linspace(0, 1, 1600)
    if bud:
        rho, h, th = r0 + R * np.sin(np.pi * u) ** 0.8 * (1 - 0.3 * u), L * u, np.zeros_like(u)
    else:
        th = CubicSpline((0, 1 / 3, 2 / 3, 1), bend, bc_type="natural")(u)
        rho = r0 + np.cumsum(np.sin(th)) * L / len(u)
        h = np.cumsum(np.cos(th)) * L / len(u)
    ph = phi + swing * u ** 2
    rad = np.stack([np.sin(ph), 0 * u, np.cos(ph)], 1)
    C = resample(np.column_stack([f(rho[:, None] * rad + h[:, None] * np.array([0, 1.0, 0])), u, ph, th]))
    u, ph, th = C[:, 3], C[:, 4], C[:, 5]
    rad = np.stack([np.sin(ph), 0 * u, np.cos(ph)], 1)
    tan = np.stack([np.cos(ph), 0 * u, -np.sin(ph)], 1)
    face = rad * np.cos(th)[:, None] - np.array([0, 1.0, 0]) * np.sin(th)[:, None]    # its outside
    Wd = tan * np.cos(twist * u)[:, None] + face * np.sin(twist * u)[:, None]
    w = wd * obovate(u, b=tip) * (1 + 0.03 * noise.line1d(len(u), 140, r))
    return Part(kind, C[:, :3], f(Wd, False), w, cup * (1 - flat * u ** 2), out=f(face, False), rows=rows,
                seed=int(r.integers(1 << 30)), **kw)


def flower(r):
    """The open flower, nodding a little to the right and tipped toward us so we look into it: three
    outer tepals and three inner, no two alike. The two in front part, the right one flaring out, and
    the one at the left turns its tip back; between them the cup shows its dark anthers round the
    pistil, against the inner faces of the tepals behind."""
    f = frame((1112.0, 640.0, 0.0), nod=0.12, elev=0.3)
    parts = []
    for phi, L, wd, bend, swing, twist, tip, flat, flame, plume, inner in (
            (0.62, 450, 106, (1.20, 0.46, 0.86, 1.25), 0.36, 0.20, 0.50, 0.60, 0.72, 0.9, False),
            (2.62, 482, 124, (1.20, 0.15, 0.08, 0.35), -0.10, -0.10, 0.56, 0.20, 0.80, 1.0, False),
            (4.71, 468, 120, (1.20, 0.25, 0.50, 2.10), 0.15, 0.35, 0.62, 0.90, 0.66, 1.1, False),
            (1.57, 462, 148, (1.15, 0.12, 0.05, 0.30), 0.10, -0.35, 0.40, 0.10, 0.84, 0.8, True),
            (3.67, 470, 134, (1.10, 0.06, -0.06, 0.24), -0.08, 0.08, 0.52, 0.00, 0.90, 1.1, True),
            (5.66, 440, 120, (1.20, 0.42, 0.62, 0.90), -0.42, -0.20, 0.44, 0.40, 0.70, 0.9, True)):
        parts.append(petal(f, phi, L, wd, r, r0=16 if inner else 22, bend=bend, swing=swing, twist=twist, tip=tip,
                           flat=flat, cup=0.4 if inner else 0.46, flame=flame, plume=plume))
    for k in range(6):                                      # stamens: a filament and its anther
        phi = 0.45 + k * np.pi / 3 + r.normal(0, 0.12)
        rad = np.array([np.sin(phi), 0.0, np.cos(phi)])
        tan = np.array([np.cos(phi), 0.0, -np.sin(phi)])
        h = np.linspace(14, 168 + r.normal(0, 14), 60)
        rho = 10 + r.uniform(12, 30) * (h / h[-1]) ** 1.5
        C = f(rho[:, None] * rad + h[:, None] * np.array([0, 1.0, 0]))
        parts.append(stalk(C, 4.0, 2.6, kind="filament", rows=3))
        a = np.linspace(0, 1, 30)[:, None]                  # the anther stands on the filament's tip, each its own way
        lean = rad * r.uniform(2, 16) + tan * r.normal(0, 9) + np.array([0, 1.0, 0]) * r.uniform(46, 62)
        parts.append(stalk(C[-1] + f(lean, False) * a, 6.0, 4.4, kind="anther", rows=3))
    h = np.linspace(0, 160, 60)                             # the pistil, and its stigma in three lobes
    parts.append(stalk(f(np.stack([np.zeros(60), h, np.zeros(60)], 1)), 16, 12, kind="pistil", rows=5))
    for k in range(3):
        phi = 0.9 + k * 2 * np.pi / 3
        rad = np.array([np.sin(phi), 0.0, np.cos(phi)])
        a = np.linspace(0, 1, 20)[:, None]
        P = f(np.array([0, 158.0, 0]) + (rad * 20 + np.array([0, 1.0, 0]) * 8) * a + rad * 6 * a ** 2)
        parts.append(stalk(P, 9, 6, kind="stigma", rows=3))
    return parts


def bud(r):
    """The bud on its own stalk, still shut, three outer tepals wrapped round the rest."""
    f = frame((668.0, 1200.0, -12.0), nod=-0.16, elev=0.12)
    return [petal(f, phi, 265, 64, r, r0=8, R=50, cup=0.5, kind="bud", bud=True, rows=22)
            for phi in (0.35, 0.35 + 2.1, 0.35 + 4.2)]


def bulb(r):
    """The bulb in its papery tunic, drawn up to a dry point round the stem; down one side the tunic
    has split and a scrap of it has lifted away from the white scale beneath. Under it, the roots."""
    P = np.array([(992, 2604, 0), (996, 2520, 0), (1003, 2400, 0), (1006, 2330, 0), (1008, 2286, 0)], np.float64)
    C = brush.path(P, 0.5)
    u = np.linspace(0, 1, len(C))
    prof = np.sin(np.pi * (0.04 + 0.96 * u) ** 0.6) ** 0.8 * (1 - 0.1 * u)
    t2 = np.gradient(C[:, :2], axis=0)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
    n2 = np.stack([-t2[:, 1], t2[:, 0], np.zeros(len(C))], 1)
    parts = [Part("bulb", C, n2, 128 * prof * (1 + 0.025 * noise.line1d(len(C), 60, r)), 0.8, rows=34,
                  seed=int(r.integers(1 << 30)))]
    point = lambda u: obovate(u, base=0.35, a=0.35, b=0.9)
    # the scrap of tunic, still held at the foot, its torn edge lifting out and curling to show its inside
    parts.append(blade([(918, 2578, 40, 0.25), (892, 2526, 58, 0.45), (870, 2470, 66, 0.85), (858, 2420, 62, 1.35),
                        (858, 2388, 54, 1.85)], 36, point, r, cup=0.3, kind="husk", rows=14))
    # the dry tip of the tunic, split into shreds round the foot of the stem
    for dx, top, z, tw in ((-12, 2236, 30, 0.35), (16, 2252, 26, -0.5)):
        parts.append(blade([(1004 + dx * 0.4, 2326, z, 0.0), (1004 + dx * 0.7, 2286, z + 3, tw * 0.5),
                            (1004 + dx * 1.1, top, z + 2, tw)], 6.5, point, r, cup=0.2, kind="shred", rows=4))
    for k in range(40):                                     # roots: a tuft of fine threads from the base plate
        x0 = 1000 + r.uniform(-58, 58)
        a = np.pi / 2 + 0.016 * (x0 - 1000) + r.normal(0, 0.3)
        L = r.uniform(60, 280) * (1 - 0.4 * abs(x0 - 1000) / 58)
        n = 9
        pts, x, y = [], x0, 2590 + r.uniform(0, 12)
        z = r.uniform(-50, 50)
        for i in range(n):
            pts.append((x, y, z))
            a += r.normal(0, 0.2)
            x, y = x + np.cos(a) * L / (n - 1), y + np.sin(a) * L / (n - 1)
        parts.append(stalk(pts, r.uniform(1.0, 1.9), 0.5, kind="root", rows=2, cup=0.9))
        if r.random() < 0.35:                               # a side root branching off it
            i = int(r.integers(2, n - 3))
            b = a + r.choice((-1, 1)) * r.uniform(0.5, 0.9)
            x, y, z = pts[i]
            sub = [(x, y, z)]
            for _ in range(4):
                b += r.normal(0, 0.25)
                x, y = x + np.cos(b) * L * 0.07, y + np.sin(b) * L * 0.07
                sub.append((x, y, z))
            parts.append(stalk(sub, 0.8, 0.4, kind="root", rows=2, cup=0.9))
    return parts


def plant(r):
    parts = flower(r) + bud(r) + bulb(r)
    # the stems keep their thickness low down among the leaves and swell a little where the upper leaf clasps
    stem = stalk([(1003, 2300, 0), (1008, 2050, 0), (992, 1700, 0), (996, 1300, 0), (1040, 950, 0), (1092, 710, 0),
                  (1112, 640, 0)], 20, 12.5, taper=2.2)
    node = np.argmin(np.hypot(stem.C[:, 0] - 1000, stem.C[:, 1] - 1480))
    stem.w += 2.2 * np.exp(-((np.arange(len(stem.C)) - node) / 110.0) ** 2)
    parts.append(stem)
    parts.append(stalk([(998, 2302, -10), (975, 2000, -10), (880, 1700, -10), (760, 1480, -12), (690, 1320, -12),
                        (668, 1200, -12)], 13.5, 9.5, taper=2.0))
    # the right leaf rises, arches over and hangs, turning as it falls to show its paler underside
    parts.append(blade([(1012, 2290, 26, 0.0), (1080, 2130, 30, 0.05), (1190, 1955, 34, 0.15), (1330, 1820, 30, 0.4),
                        (1470, 1752, 22, 0.95), (1585, 1772, 12, 1.75), (1662, 1852, 4, 2.45), (1706, 1960, -2, 2.85),
                        (1718, 2045, -6, 3.05)], 108, lanceolate, r, cup=-0.42, rows=32, ripple=(0.11, 200)))
    # the left leaf reaches out with only a turn at its tip
    parts.append(blade([(992, 2296, -26, 0.0), (930, 2080, -26, -0.1), (845, 1860, -22, -0.25), (750, 1690, -16, -0.45),
                        (645, 1590, -10, -0.75), (548, 1556, -5, -1.05), (462, 1580, 0, -1.4), (402, 1630, 0, -1.7)],
                       100, lanceolate, r, cup=-0.42, rows=30, ripple=(0.12, 180)))
    parts.append(blade([(1000, 1480, 14, 0.0), (1050, 1360, 16, 0.1), (1140, 1210, 16, 0.3), (1240, 1100, 10, 0.65),
                        (1320, 1062, 5, 1.25), (1372, 1078, 0, 1.9)], 64, lanceolate, r, cup=-0.36, rows=20,
                       ripple=(0.1, 150)))
    return parts


# ---- laying the drawing out on the copper

def raster(parts):
    """Every part laid into the plate, the nearest surface winning each pixel: which part shows there,
    where on it (u along, v across), and which way that bit of surface faces."""
    zb = np.full(PH * PW, -np.inf, np.float32)
    R = dict(own=np.full(PH * PW, -1, np.int16), back=np.zeros(PH * PW, bool))
    for k in ("u", "v", "dif", "fold", "face", "tx", "ty"):
        R[k] = np.zeros(PH * PW, np.float32)
    for k, p in enumerate(parts):
        span = p.w.max() * np.sqrt(1 + 4 * np.abs(p.cup).max() ** 2)
        v = np.linspace(-1, 1, int(np.ceil(2 * span / 0.5)) + 2)
        u = np.linspace(0, 1, len(p.C))
        # the cross-section bows by `cup`, and each margin waves out of it by `rip`
        a = np.where(v[None] < 0, p.rip[:, :1], p.rip[:, 1:])
        v3 = np.abs(v)[None] ** 3
        bow = (p.cup * p.w)[:, None] * (1 - v * v)[None] + p.w[:, None] * a * v3
        P = p.C[:, None] + p.Wd[:, None] * (p.w[:, None] * v[None])[..., None] + p.n[:, None] * bow[..., None]
        slope = 2 * p.cup[:, None] * v[None] - 3 * a * (v * np.abs(v))[None]
        lean = np.where(v[None] < 0, p.drip[:, :1], p.drip[:, 1:]) * p.w[:, None] * v3
        nn = p.n[:, None] + slope[..., None] * p.Wd[:, None] - lean[..., None] * p.T[:, None]
        nn /= np.linalg.norm(nn, axis=2, keepdims=True)
        bk = nn[..., 2] < 0
        nn = np.where(bk[..., None], -nn, nn)
        # the light on its cross-section alone, without the waving of its margins: the colourist's guide
        n0 = p.n[:, None] + (2 * p.cup[:, None] * v[None])[..., None] * p.Wd[:, None]
        n0 = n0 * np.where(n0[..., 2:] < 0, -1.0, 1.0) / np.linalg.norm(n0, axis=2, keepdims=True)
        x, y = np.rint(P[..., 0]).astype(np.int64), np.rint(P[..., 1]).astype(np.int64)
        ok = ((x >= 0) & (x < PW) & (y >= 0) & (y < PH)).ravel()
        sel = np.flatnonzero(ok)
        idx = (y.ravel() * PW + x.ravel())[sel]
        z = P[..., 2].ravel()[sel]
        o = np.argsort(z, kind="stable")
        win = z[o] > zb[idx[o]]
        sel, idx = sel[o][win], idx[o][win]
        t2 = p.T[:, :2] / (np.linalg.norm(p.T[:, :2], axis=1, keepdims=True) + 1e-9)
        zb[idx] = P[..., 2].ravel()[sel]
        R["own"][idx] = k
        R["back"][idx] = bk.ravel()[sel]
        R["u"][idx] = np.broadcast_to(u[:, None], bk.shape).ravel()[sel]
        R["v"][idx] = np.broadcast_to(v[None], bk.shape).ravel()[sel]
        R["dif"][idx] = np.clip(nn @ SUN, 0, 1).ravel()[sel]
        R["fold"][idx] = np.clip(n0 @ SUN, 0, 1).ravel()[sel]
        R["face"][idx] = nn[..., 2].ravel()[sel]
        R["tx"][idx] = np.broadcast_to(t2[:, None, 0], bk.shape).ravel()[sel]
        R["ty"][idx] = np.broadcast_to(t2[:, None, 1], bk.shape).ravel()[sel]
    R = {k: a.reshape(PH, PW) for k, a in R.items()}
    R["z"] = zb.reshape(PH, PW)
    return R


def streaks(seed, shape=(48, 360)):
    """A field over a tepal's (u, v), smooth along it and fine across: the grain of its veins."""
    return noise.stretched(shape, 2.2, 26, noise.rng(seed))


def sample(field, u, v):
    """Read a field laid over a part's (u along, v across)."""
    h, w = field.shape
    return ndimage.map_coordinates(field, [u * (h - 1), (v + 1) / 2 * (w - 1)], order=1, mode="nearest")


def feather(p, r):
    """The crimson of a broken tulip on one tepal, as the colourist's fine brush laid it along the veins:
    a flame up the midrib that branches as it climbs, and from each margin a feather of short strokes
    running in and down along the veins, finest at the edge. Each stroke swells where the brush was
    pressed and thins and pales as it lifts. -> rows of (u, v, half-width px, strength, stroke)."""
    L = 0.5 * len(p.C)
    half = lambda u: np.interp(u, np.linspace(0, 1, len(p.w)), p.w)
    out = []

    def stroke(u0, v0, u1, v1, rad, bow=0.0, rise=0.1, fade=0.4):
        n = int(np.hypot((u1 - u0) * L, (v1 - v0) * half((u0 + u1) / 2)) / 0.35) + 8
        t = np.linspace(0, 1, n)
        u = u0 + (u1 - u0) * t
        v = v0 + (v1 - v0) * t + bow * np.sin(np.pi * t) + 0.01 * noise.line1d(n, max(6.0, n / 4), r)
        w = rad * noise.smoothstep(0, rise, t) ** 0.5 * (1 - 0.72 * t ** 1.4)
        s = (1 - 0.85 * noise.smoothstep(1 - fade, 1, t)) * np.clip(1 + 0.22 * noise.line1d(n, max(6.0, n / 6), r),
                                                                    0.4, 1.3)
        out.append(np.column_stack([u, np.clip(v, -0.995, 0.995), w, s, np.full(n, len(out))]))

    for _ in range(int(r.integers(22, 34))):                # the flame, rising from just above the clean base
        v0 = float(np.clip(r.normal(0, 0.2), -0.55, 0.55))
        u0 = r.uniform(0.1, 0.18) + r.exponential(0.05)
        u1 = u0 + (p.flame - u0) * (1 - (abs(v0) / 0.6) ** 1.2) * r.uniform(0.72, 1.08)
        if u1 - u0 < 0.08:
            continue
        v1 = v0 * r.uniform(1.0, 1.5) + r.normal(0, 0.03)
        stroke(u0, v0, u1, v1, r.uniform(0.9, 2.0), bow=r.normal(0, 0.02))
        for _ in range(r.poisson(2.2)):                     # branching up and out as it climbs
            a = r.uniform(0.25, 0.85)
            ub, vb = u0 + a * (u1 - u0), v0 + a * (v1 - v0)
            stroke(ub, vb, ub + r.uniform(0.05, 0.16), vb + np.sign(vb + 1e-6) * r.uniform(0.06, 0.22),
                   r.uniform(0.7, 1.4), bow=r.normal(0, 0.02), rise=0.05)
    for side in (-1, 1):
        u = r.uniform(0.15, 0.35)                           # along the very edge, in broken lengths
        while u < 0.96:
            du = r.uniform(0.08, 0.3)
            stroke(u, side * r.uniform(0.94, 0.985), min(u + du, 0.99), side * r.uniform(0.92, 0.985),
                   r.uniform(0.7, 1.2), rise=0.05, fade=0.2)
            u += du + r.uniform(-0.03, 0.05)
        for us in r.uniform(0.2, 0.96, int(p.plume * r.uniform(42, 60))):     # the barbs
            reach = r.uniform(0.06, 0.3) * (0.4 + 0.6 * np.sin(np.pi * us))
            stroke(us, side * 0.975, max(us - 1.5 * reach * half(us) / L, 0.02), side * (0.975 - reach),
                   r.uniform(0.65, 1.15), rise=0.06, fade=0.5)
    return np.vstack(out)


def brushwork(parts, R, r):
    """The colourist's crimson, stroke by stroke: each tepal's feather and flame carried through its form
    onto the sheet, and kept only where that tepal shows. -> how much of a stroke lies on each pixel."""
    out = np.zeros((PH, PW), np.float32)
    for k, p in enumerate(parts):
        if p.kind != "tepal":
            continue
        u, v, rad, s, sid = feather(p, r).T
        i = u * (len(p.C) - 1)
        at = lambda a: np.interp(i, np.arange(len(a)), a)
        w, cup = at(p.w), at(p.cup)
        P = np.stack([at(p.C[:, j]) + at(p.Wd[:, j]) * w * v + at(p.n[:, j]) * cup * w * (1 - v * v)
                      for j in (0, 1)], 1)
        ds = np.hypot(*np.diff(P, axis=0, prepend=P[:1]).T)
        ds = np.where(np.diff(sid, prepend=-1) != 0, np.median(ds), ds)
        lay = np.zeros((PH, PW), np.float32)
        edges = (0.15, 0.5, 0.75, 1.05, 1.45, 2.0, 2.8, 9.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rad >= lo) & (rad < hi)
            if not m.any():
                continue
            sig = max(0.42, (lo + min(hi, 3.2)) / 2 / 1.18)   # a line blurred by sig is half as dark 1.18 sig out
            buf, o = brush.splat(P[m, 0], P[m, 1], (s[m] * ds[m] * np.sqrt(2 * np.pi) * sig).astype(np.float32))
            brush.paste(lay, ndimage.gaussian_filter(buf, sig), o)
        out = np.maximum(out, np.minimum(lay, 1.3) * (R["own"] == k))
    return out


def survey(parts, R, crimson, r):
    """What the engraver and the colourist each read off the drawing: the tone to engrave at every
    point (0 white paper .. 1 the deepest the dots go), the rows the dots fall in along the form, the
    ink to dab into each part of the plate, and the washes to lay over the print. The colourist's
    crimson strokes are already planned; the printer dabs crimson ink only roughly where they go."""
    own, u, v, back, dif, face = (R[k] for k in ("own", "u", "v", "back", "dif", "face"))
    rough = np.clip(1.5 * ndimage.gaussian_filter(crimson, 2.5), 0, 1)
    tone = np.zeros((PH, PW), np.float32)
    rows = np.ones((PH, PW), np.float32)
    ink = {k: np.zeros((PH, PW), np.float32) for k in INKS}
    washes = {k: np.zeros((PH, PW), np.float32) for k in (
        "leaf", "fold", "under", "stem", "yellow", "flame", "blotch", "green", "flush", "tunic", "flesh", "husk",
        "root", "anther", "pistil")}
    for k, p in enumerate(parts):
        ys, xs = np.nonzero(own == k)
        if not len(ys):
            continue
        U, V, B, D, F = u[ys, xs], v[ys, xs], back[ys, xs], dif[ys, xs], np.clip(face[ys, xs], 0, 1)
        dark = 1 - (0.3 + 0.7 * D)                       # the side away from the sun
        rim = (1 - F) ** 3 * (0.35 + 0.65 * (1 - D))     # the surface turning away at its edge, more so in shade
        # the dots fall loosely in rows along the form, as the engraver worked down it; the rows wander,
        # crowd and thin, and here and there dissolve. On a leaf or a tunic they are its veins, and the
        # dots gather tight along them.
        wob, amp = (sample(noise.field((40, 80), 9, r), U, V) for _ in range(2))
        ph = np.pi * (V + 1) * p.rows + 2.2 * wob + r.uniform(0, 6.3)
        if p.kind in ("leaf", "bulb", "husk", "shred"):
            vein = ((1 + np.cos(ph)) / 2) ** (2 if p.kind == "bulb" else 3)
            rows[ys, xs] = 0.4 + np.clip(1.7 + 0.6 * amp, 0.3, None) * vein
        else:
            rows[ys, xs] = 1 + (0.42 + 0.22 * amp) * np.cos(ph)
        put = lambda name, val: ink[name].__setitem__((ys, xs), val)
        lay = lambda name, val: washes[name].__setitem__((ys, xs), val)
        if p.kind == "leaf":
            keel = np.exp(-(V / 0.05) ** 2) * noise.smoothstep(0.05, 0.25, U) * (1 - U)     # the fold down its middle
            t = np.where(B, 0.08 + 0.6 * dark + 0.28 * rim, 0.16 + 1.0 * dark + 0.32 * rim + 0.18 * keel)
            put("green", np.where(B, 0.55, 1.0))
            put("grey", np.where(B, 0.45, 0.0))
            lay("fold", np.where(B, 0.0, 1 - (0.3 + 0.7 * R["fold"][ys, xs]) + 0.25 * keel))
            lay("leaf", ~B)
            lay("under", B)
        elif p.kind == "stem":
            # the light catches it in a narrow band down the side toward the sun, and the colourist spares it
            lit = np.sign(-R["ty"][ys, xs] * SUN[0] + R["tx"][ys, xs] * SUN[1])
            hl = np.exp(-((V - 0.45 * lit) / 0.17) ** 2)
            t = (0.14 + 0.85 * dark + 0.3 * rim) * (1 - 0.92 * hl)
            put("green", 1.0)
            lay("stem", 1 - 0.85 * hl)
        elif p.kind == "tepal":
            fl, cr = rough[ys, xs], crimson[ys, xs]
            deep = np.where(B, 0.34 * (1 - U) ** 2, 0.16 * (1 - U) ** 4)   # down in the cup, and where tepals overlap
            blot = np.where(B, noise.smoothstep(0.16, 0.07, U), 0.0)
            t = 0.75 * dark ** 1.4 + 0.2 * rim + 0.08 * fl + 0.08 * np.minimum(cr, 1) + deep + 0.3 * blot
            put("crimson", fl * (1 - blot))
            put("yellow", (1 - fl) * (1 - blot))
            put("green", 0.5 * blot)
            put("black", 0.5 * blot)
            lay("yellow", 1.0)
            lay("flame", cr * (1 - blot))
            lay("blotch", blot)
        elif p.kind == "bud":
            s = sample(streaks(p.seed), U, V)
            green = noise.smoothstep(0.45, 0.1, U + 0.06 * s)
            flush = noise.smoothstep(0.5, 0.9, U + 0.1 * s) * (0.6 + 0.4 * noise.smoothstep(-0.6, 0.6, s))
            t = 0.06 + 0.5 * dark + 0.22 * rim + 0.1 * flush + 0.06 * green
            put("green", green)
            put("crimson", flush * (1 - green))
            put("yellow", (1 - flush) * (1 - green))
            lay("green", green)
            lay("yellow", 1 - green)
            lay("flush", flush * (1 - green))
        elif p.kind == "bulb":
            s = sample(streaks(p.seed, (40, 200)), U, V)
            # down the side toward the light the tunic has split and the scrap beside the split has lifted
            # away: there the white scale shows, the torn edge throwing a hair of shadow over it
            e = np.minimum(-0.4 + 0.05 * s + 0.025 * np.sin(17 * U + 3 * s) - V, 0.55 - 0.2 * (V + 1) + 0.03 * s - U)
            flesh = noise.smoothstep(-0.008, 0.008, e) * noise.smoothstep(0.03, 0.07, U)
            edge = np.exp(-np.abs(e) / 0.012) * (1 - flesh)
            lift = np.exp(-np.abs(e) / 0.04) * flesh
            foot = noise.smoothstep(0.06, 0.0, U)
            t = (1 - flesh) * (0.1 + 0.7 * dark + 0.28 * rim) * (0.7 + 0.6 * vein) \
                + flesh * (0.04 + 0.42 * dark + 0.2 * rim) + 0.35 * edge + 0.28 * lift + 0.35 * foot
            put("bistre", (1 - flesh) * 0.7 + 0.6 * foot)
            put("rose", (1 - flesh) * 0.3 * (1 - foot))
            put("grey", flesh * (1 - foot))
            lay("tunic", 1 - flesh)
            lay("flesh", flesh)
        elif p.kind in ("husk", "shred"):               # the dry tunic: brown outside, pale and satiny within
            B = B | (p.kind == "shred")                  # its tip is dry and pale all through
            t = np.where(B, 0.04 + 0.4 * dark + 0.25 * rim, 0.1 + 0.65 * dark + 0.3 * rim)
            put("bistre", np.where(B, 0.4, 0.75))
            put("grey", np.where(B, 0.6, 0.0))
            put("rose", np.where(B, 0.0, 0.25))
            lay("husk", B)
            lay("tunic", ~B)
        elif p.kind == "root":
            t = 0.2 + 0.3 * dark + 0.1 * rim
            put("bistre", 0.3)
            put("grey", 0.7)
            lay("root", 1.0)
        elif p.kind in ("filament", "anther"):
            t = (0.25 if p.kind == "filament" else 0.45) + 0.4 * dark + 0.2 * rim
            put("violet", 1.0 if p.kind == "filament" else 0.75)
            put("black", 0.0 if p.kind == "filament" else 0.25)
            lay("anther", 1.0 if p.kind == "anther" else 0.5)
        else:                                            # the pistil and its stigma
            t = 0.12 + 0.5 * dark + 0.2 * rim
            put("green" if p.kind == "pistil" else "yellow", 1.0)
            lay("pistil" if p.kind == "pistil" else "yellow", 1.0)
        tone[ys, xs] = t
    return tone, rows, ink, washes


def shadows(R, tone):
    """Where one part passes in front of another it throws a little shadow on it, deepest on the side
    away from the sun; and a part's own edge is found with a closer line of dots."""
    own, z = R["own"], R["z"]
    near = np.zeros_like(tone)
    sun = -SUN[:2] / np.linalg.norm(SUN[:2])             # the way shadows fall
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        o = np.roll(own, (dy, dx), (0, 1))
        zz = np.roll(z, (dy, dx), (0, 1))
        front = (own >= 0) & (o >= 0) & (o != own) & (zz > z + 2)
        # the neighbour in front lies toward (-dx, -dy) from here: a shadow if that is toward the sun
        w = np.clip(0.45 + 1.1 * (-(dx * sun[0] + dy * sun[1]) / np.hypot(dx, dy)), 0.25, 1.0)
        near = np.maximum(near, front * w)
    d, (iy, ix) = ndimage.distance_transform_edt(near == 0, return_indices=True)
    same = own[iy, ix] == own
    tone = tone + same * near[iy, ix] * 0.42 * np.exp(-d / 7.0)
    edge = (own != np.roll(own, 1, 0)) | (own != np.roll(own, -1, 0)) | (own != np.roll(own, 1, 1)) | \
           (own != np.roll(own, -1, 1))
    de = ndimage.distance_transform_edt(~edge)
    tone = tone + (own >= 0) * 0.1 * (1 - R["dif"]) * np.exp(-de / 1.2)
    return np.clip(tone, 0, 1) * (own >= 0)


# ---- the engraving and the printing

def engrave(tone, rows, R, r):
    """Dots and flicks into the copper wherever the tone asks for them: placed one by one on a loose
    lattice, kept more often where it is darker and along the rows of the form, larger and deeper in
    the shade, drawn out into short flicks along the form where it is darkest. -> what the pits hold."""
    gh, gw = int(PH / G), int(PW / G)
    cy = (np.arange(gh)[:, None] + 0.5 + r.uniform(-0.46, 0.46, (gh, gw))) * G
    cx = (np.arange(gw)[None, :] + 0.5 + r.uniform(-0.46, 0.46, (gh, gw))) * G
    iy, ix = np.clip(cy.astype(int), 0, PH - 1), np.clip(cx.astype(int), 0, PW - 1)
    t = tone[iy, ix]
    # ranks spread evenly over the lattice, so the dots kept at any tone keep their distance
    white = r.random((gh, gw))
    hp = white - ndimage.gaussian_filter(white, 1.1)
    rank = np.empty(gh * gw)
    rank[np.argsort(hp.ravel())] = np.linspace(0, 1, gh * gw)
    rank = rank.reshape(gh, gw)
    keep = (R["own"][iy, ix] >= 0) & (rank < np.clip(1.25 * t * rows[iy, ix], 0, 1))
    x, y, t = cx[keep], cy[keep], t[keep]
    tx, ty = R["tx"][iy[keep], ix[keep]], R["ty"][iy[keep], ix[keep]]
    n = len(x)
    rad = (0.5 + 0.85 * t) * np.exp(r.normal(0, 0.2, n))
    dep = (0.55 + 0.55 * t) * r.uniform(0.8, 1.2, n)
    flick = r.random(n) < 0.7 * noise.smoothstep(0.3, 0.75, t)
    ln = np.where(flick, (1.0 + 3.4 * t) * r.uniform(0.6, 1.4, n), 0.0)
    ang = np.arctan2(ty, tx) + r.normal(0, 0.22, n)
    pits = np.zeros((PH, PW), np.float32)
    for lo, hi, sig in ((0, 0.72, 0.42), (0.72, 1.02, 0.6), (1.02, 9, 0.82)):
        m = (rad >= lo) & (rad < hi)
        if not m.any():
            continue
        X, Y, M = [], [], []
        for k, wt in zip(np.linspace(-0.5, 0.5, 5), (0.6, 1.0, 1.1, 1.0, 0.6)):
            X.append(x[m] + np.cos(ang[m]) * ln[m] * k)
            Y.append(y[m] + np.sin(ang[m]) * ln[m] * k)
            M.append(dep[m] * np.pi * rad[m] ** 2 * (1 + 0.5 * ln[m] / (rad[m] * 2)) * wt / 4.3)
        buf, o = brush.splat(np.concatenate(X), np.concatenate(Y), np.concatenate(M).astype(np.float32))
        brush.paste(pits, ndimage.gaussian_filter(buf, sig), o)
    return pits


def dab(ink, r):
    """The printer's poupée cannot follow an edge exactly: each colour spreads a little into its
    neighbours and lies unevenly. -> weights summing to 1 wherever there is ink."""
    h, w = PH, PW
    dx = 2.5 * noise.field((h, w), 40, r)
    dy = 2.5 * noise.field((h, w), 40, r)
    out = {}
    for k, m in ink.items():
        if m.any():
            out[k] = np.clip(ndimage.gaussian_filter(noise.warp(m, dx, dy), 2.2) * (1 + 0.12 * noise.field((h, w), 60, r)),
                             0, None)
    total = sum(out.values()) + 1e-6
    return {k: m / total for k, m in out.items()}


def letters(r):
    """The plant's name engraved under it and the painter's at the left, as the writing engraver cut
    them: fine, sharp, swelling where the burin went deeper."""
    if not NAME.exists():
        write()
    grooves = np.zeros((PH, PW), np.float32)
    for path, (yt, xc, left) in ((NAME, (2868, PW / 2, False), ), (SIGN, (3012, 92, True), )):
        m = np.asarray(Image.open(path), np.float32) / 255
        m = noise.warp(m, 0.35 * noise.field(m.shape, 14, r), 0.35 * noise.field(m.shape, 14, r))
        h, w = m.shape
        x0 = int(xc if left else xc - w / 2)
        grooves[yt:yt + h, x0:x0 + w] = np.maximum(grooves[yt:yt + h, x0:x0 + w], m)
    return grooves


def write():
    """Cut the lettering into atelier/lettering (it needs the Snell Roundhand face of macOS)."""
    lettering.script("Tulipa Gesneriana", "malmaison_name", size=86, font=SNELL)
    lettering.script("Claude pinx.", "malmaison_pinx", size=33, font=SNELL)


# ---- the paper, the colourist, the press

def wove(r):
    """Wove paper, made on a mould of woven wire, so it has no lines in it: a quiet formation, a fine
    felted surface, a speck of something here and there. The sheet has been trimmed."""
    shape = (H, W)
    formation = noise.fbm(shape, 420, r, octaves=6, gain=0.55)
    fib = ndimage.gaussian_filter(noise.fibers(shape, r, count=H * W // 1300, length=(5, 26), curl=0.25,
                                               strength=(0.1, 0.4)), 0.5)
    tooth = 0.5 * noise.field(shape, 1.3, r) + 0.35 * noise.field(shape, 3.5, r) + 0.45 * fib
    lo, hi = np.percentile(tooth, [1, 99])
    tooth = np.clip((tooth - lo) / (hi - lo), 0, 1).astype(np.float32)
    light = 1 + 0.011 * formation + 0.005 * fib + 0.003 * noise.field(shape, 1.0, r)
    color = lin("#f6f1e5")[None, None, :] * light[..., None]
    specks = np.zeros(shape, np.float32)
    n = 70
    specks[r.integers(0, H, n), r.integers(0, W, n)] = r.uniform(0.4, 1.0, n)
    specks = ndimage.gaussian_filter(specks, 0.8) * 4
    color *= 1 - np.clip(specks, 0, 0.5)[..., None] * np.array([0.5, 0.6, 0.75], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.minimum.reduce([yy - EDGE, H - 1 - EDGE - yy, xx - EDGE, W - 1 - EDGE - xx])
    d += 0.8 * noise.field(shape, 300, r) + 0.3 * noise.field(shape, 6, r)
    alpha = noise.smoothstep(-0.8, 0.8, d).astype(np.float32)
    return Sheet(color.astype(np.float32), tooth, np.clip(fib * 2.5, 0, 1).astype(np.float32), alpha)


def broad(mask, k):
    """A shape as a brush can lay it: nothing narrower than about `k` px, its corners rounded."""
    return noise.smoothstep(0.42, 0.58, ndimage.gaussian_filter(mask, k / 2.5))


def colour(washes, tone, sheet, r):
    """The colourist's washes over the dry print: each laid by hand inside its form, not quite to the
    edge here and a hair over it there, drying with its own rim; a second, smaller wash goes over the
    shaded side of a form when the first is dry, with an edge of its own. -> absorbance (PH, PW, 3)."""
    A = np.zeros((PH, PW, 3), np.float32)

    def lay(mask, density, ragged=1.2, k=5, shift=1.0, inner=1.0, **mix):
        dx, dy = (shift * noise.field((PH, PW), 90, r) for _ in range(2))
        m = broad(noise.warp(mask, dx, dy), k)
        if m.max() < 0.05:
            return
        wet = wc.puddle(m, sheet, r, ragged)
        dep = wc.deposit(wet, sheet, r, pool=0.2, rim=0.45, rim_width=2.5, tides=0.1, grain=0.2, scale=110)
        A[:] += (dep * density * inner)[..., None] * sum(f * PAINT[c] for c, f in mix.items())

    side = lambda name, lo: noise.smoothstep(lo, lo + 0.05, tone) * washes[name]
    lay(washes["leaf"], 0.4, sap=0.55, verdigris=0.45)
    # the half of each leaf turned from the light, along its fold; once more down the deepest of it
    fold = ndimage.gaussian_filter(washes["fold"], 3)
    lay(noise.smoothstep(0.3, 0.34, fold) * washes["leaf"], 0.26, k=8, verdigris=0.5, indigo=0.5)
    lay(noise.smoothstep(0.42, 0.46, fold) * washes["leaf"], 0.15, k=5, indigo=0.55, verdigris=0.45)
    lay(washes["under"], 0.22, verdigris=0.5, cobalt=0.35, sap=0.15)
    lay(washes["stem"], 0.3, sap=0.6, verdigris=0.4, k=3)
    lay(washes["green"], 0.3, sap=0.7, verdigris=0.3, k=3)
    flame = washes["flame"]
    halo = ndimage.gaussian_filter(flame, 1.5)
    lay(washes["yellow"], 0.34, k=3, inner=1 - 0.4 * noise.smoothstep(0.1, 0.6, halo), gamboge=1.0)
    lay(side("yellow", 0.32), 0.12, k=5, gamboge=0.3, sienna=0.5, vermilion=0.2)
    # the flames with a fine brush, stroke by stroke along the veins: a pale rose where the strokes run
    # thin and lift, and over it the crimson, strongest where the brush was pressed
    inside = ndimage.binary_erosion(washes["yellow"] > 0.5, iterations=3)
    lay(noise.smoothstep(0.06, 0.24, ndimage.gaussian_filter(flame, 3.5)) * inside, 0.13, ragged=0.8, k=2.5, shift=0.6,
        madder=0.6, carmine=0.25, vermilion=0.15)
    lay(noise.smoothstep(0.4, 0.6, flame), 0.66, ragged=0.5, k=1.2, shift=0.4,
        inner=0.55 + 0.45 * np.clip(flame, 0, 1), carmine=0.85, madder=0.15)
    lay(washes["flush"], 0.45, ragged=0.8, k=2, carmine=0.8, madder=0.2)
    lay(washes["blotch"], 0.5, k=2, indigo=0.5, sap=0.5)
    lay(washes["tunic"], 0.3, burnt=0.45, sienna=0.47, madder=0.08)
    lay(side("tunic", 0.45), 0.16, k=7, burnt=0.8, madder=0.2)
    lay(washes["flesh"], 0.08, sap=0.4, sienna=0.3, gamboge=0.3)
    lay(washes["husk"], 0.14, k=2, sienna=0.6, gamboge=0.25, burnt=0.15)
    lay(washes["root"], 0.08, k=1, ragged=0.4, sienna=0.6, indigo=0.4)
    lay(washes["anther"], 0.42, k=1.5, violet=0.7, indigo=0.3)
    lay(washes["pistil"], 0.3, k=1.5, sap=0.8, gamboge=0.2)
    return A


def paint(seed=1805):
    r = noise.rng(seed)
    sheet = wove(noise.rng(seed + 1))
    parts = plant(noise.rng(seed + 2))
    R = raster(parts)
    crimson = brushwork(parts, R, noise.rng(seed + 7))
    tone, rows, ink, washes = survey(parts, R, crimson, noise.rng(seed + 3))
    tone = shadows(R, tone)
    pits = engrave(tone, rows, R, noise.rng(seed + 4))
    dens = saturate(1.35 * pits, 2.2)
    inks = dab(ink, noise.rng(seed + 5))
    region = (slice(Y0, Y0 + PH), slice(X0, X0 + PW))
    ps = Sheet(*(a[region] for a in (sheet.color, sheet.tooth, sheet.fiber, sheet.alpha)))
    A = sum((dens * m)[..., None] * INKS[k] for k, m in inks.items())
    A = A + saturate(2.2 * letters(r), 2.0)[..., None] * INKS["black"]
    A = A + colour(washes, tone, ps, noise.rng(seed + 6))
    img = sheet.color.copy()
    img[region] *= np.exp(-A)
    # the copper's edge pressed into the damp sheet, and the faint film the wiping left on it
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    qx, qy = np.abs(xx - X0 - PW / 2) - PW / 2 + 22, np.abs(yy - Y0 - PH / 2) - PH / 2 + 22   # its corners filed round
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - 22
    pm = noise.smoothstep(1.0, -1.0, d).astype(np.float32)
    film = 0.01 * (1 + 0.6 * noise.fbm((H, W), 260, r, octaves=4)) * pm
    img *= np.exp(-film[..., None] * INKS["bistre"])
    img = relief.emboss(img, [pm], light=(-0.6, -0.8), depth=0.06)
    # under the gallery light the sheet's grain shows, less where the press flattened it against the copper
    flat = Sheet(sheet.color, sheet.tooth * (1 - 0.6 * pm), sheet.fiber, sheet.alpha)
    img = wash.cockle(img, flat, ndimage.gaussian_filter(pm, 30) * 0.3, r, buckle=1.0, grain=0.12)
    return plate.mount(img, sheet, shadow=0.35)
