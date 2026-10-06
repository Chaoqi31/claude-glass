"""The Porte d'Aval, Late Afternoon. Oil on canvas, built up over several sittings.

Between 1883 and 1886 Monet went back again and again to Étretat on the coast of Normandy, where
the chalk stands sheer out of the Channel and the sea has worn it into arches and needles: the
Porte d'Aval at the end of the beach with the Needle standing off it, and farther west the great
arch of the Manneporte. Maupassant, who watched him there, saw him go out followed by children
carrying five or six canvases of the same view, each for its own hour and its own effect. In the
cliffs he built the chalk out of dense strokes laid every way, warm where the sun found it, lilac
and blue-green where it did not, with the turf of the downs streaked green over the top; the sea
below went on in shorter, choppier touches, and the lights and the foam in thick paint.

Here the sun is low on the left at the end of an afternoon. It warms the face of the cliff to cream,
apricot and rose, and the buttresses of the face cast the bays between them into lilac. The arch
stands at the end of the cliff with its leg in the sea, and the Needle beyond it, and through the
arch the horizon shows under a pale warm sky. The sea is deep blue-green, turquoise where it runs
over the chalk, and it breaks white at the foot of the rock.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate, ndimage

from atelier import canvas, dabs, impasto, noise
from atelier.color import lin

TITLE = "The Porte d'Aval, Late Afternoon"
DATE = "2026"
MEDIUM = ("Oil on canvas, the chalk built up over several sittings in dense strokes laid every way, the sea in "
          "short broken touches, the lights and the foam in thick paint")
AFTER = "Claude Monet, the cliffs at Étretat, 1883–86"
ROOM = "Open Air"
YEAR = 1885
PLACE = "Étretat"
REGION = "Europe"
NOTE = ("The arch of the Porte d'Aval and the Needle beyond it at the end of an afternoon, the chalk warm in the "
        "sun and lilac in shadow, the sea breaking white at its foot.")

H, W = 2100, 2800
HZ = 1000.0                     # the horizon, at the height of the eye
VX = 6000.0                     # where the strata, running away to the right, would meet the horizon
TOP, FOOT = 120.0, 1150.0       # the top and the foot of the cliff at the left edge
XT = 1596                       # where the top ends and the outer edge of the arch begins to fall
TRUNK = [(1652, 404), (1722, 476), (1782, 586), (1822, 724), (1849, 880), (1863, 1000), (1869, 1112)]
TURRETS = [(1482, 24, 22), (1534, 46, 26), (1580, 30, 18)]       # chalk worn into turrets: x, height, half-width
OPENING = [(1515, 1190), (1517, 1000), (1521, 862), (1531, 742), (1551, 652), (1584, 606), (1620, 622),
           (1660, 700), (1700, 822), (1735, 952), (1758, 1080), (1766, 1190), (1640, 1240)]
NEEDLE = (2138.0, 1076.0, 640.0, 92.0)                            # its axis, foot, top, half-width at the foot
LIGHT = (-0.6, -0.5, 0.62)                                        # the gallery lamp
SUN = np.float32([-0.45, -0.3, 0.84]) / np.float32(np.linalg.norm([-0.45, -0.3, 0.84]))   # low on the left, in front
LUMA = np.float32([0.2126, 0.7152, 0.0722])


def pal(*hexes):
    return np.stack([lin(h) for h in hexes]).astype(np.float32)


SKY = pal("#f6dcb9", "#f2d6c6", "#dcd6e0", "#bfcfe3", "#a6bedc")    # from the horizon up
CLOUD = pal("#f6dbc5", "#e7d0ce", "#cdc5d5", "#b7b4cb")             # lit rose, rose-grey, lilac-grey, shadow
SUNLIT = pal("#f4dfb6", "#efc290", "#e8b09b", "#f3d39c", "#f7e9d1")  # chalk in the sun: cream, apricot, rose, gold, white
HALF = pal("#cfa49d", "#b99ba9", "#c8ad9b")                         # turning from it: dusty rose, mauve, warm grey
SHADE = pal("#9387b3", "#7479ae", "#6f9ba7", "#8e87a4", "#6d8bb1")  # lilac, blue-violet, blue-green, grey-lilac, blue
DEEP = pal("#504b81", "#3f4b79", "#406375")                         # cracks, caves, the underside of the arch
TURF = pal("#a3ad55", "#869c47", "#c2bb62", "#6e8f4c")              # grass on the top in the sun
TURF_SHADE = pal("#4f7058", "#45655d", "#5f7a5f")
EARTH = pal("#c9a96a", "#7f9a75", "#b98a6e")                         # ochre, moss and iron in the chalk
WRACK = pal("#57523b", "#434f40", "#635350", "#4d4970")             # weed and wet chalk at the foot
SEA = pal("#b9c6d0", "#8fbcc0", "#5aa5a6", "#348089", "#245f72", "#1d4c66")   # far to near
TURQ = pal("#4faea6", "#6cc0b2", "#3f9c9e", "#8ccdbf")              # milky water over the chalk
DARKSEA = pal("#173b53", "#1c4959", "#22405d", "#1b5051", "#29396b")  # the fronts of the waves
VIOLET = pal("#454a8a", "#565a99", "#3d4f86")                       # ultramarine in the near water
GLOW = pal("#d5c3bf", "#e5cfc2", "#c6bfcc")                         # the sky caught on the far water
FOAM = pal("#f8f5ec", "#f2f3ef", "#faf0dd", "#e9efef")
FOAM_SHADE = pal("#bccbd3", "#a8bfcd", "#c4c6d8", "#b1cecb")


def mix(a, b, t):
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def ramp(colours, t):
    """The colour at t (0..1) along a palette."""
    t = np.clip(t, 0, 0.9999) * (len(colours) - 1)
    i = t.astype(int)
    return colours[i] + (colours[i + 1] - colours[i]) * (t - i)[..., None]


def spline(points, n, closed=False):
    """A smooth curve through points (x, y). -> (n, 2)"""
    p = np.asarray(points, np.float64)
    if closed:
        p = np.vstack([p, p[:1]])
    tck, _ = interpolate.splprep(p.T, s=0, per=int(closed))
    return np.stack(interpolate.splev(np.linspace(0, 1, n), tck), 1)


def fill(points):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in np.asarray(points, np.float64)], fill=255)
    return np.asarray(im) > 127


def signed(mask):
    """Distance to the edge of a mask, px, negative inside."""
    return (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)).astype(np.float32)


def at(P):
    return np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)


def rock(r):
    """The cliff with the arch at its end, and the Needle. The top of the cliff is the edge of the
    downs, nearly level but running down toward the arch as it goes away; at its end the chalk
    stands bare in turrets. The edges are worn in short level jogs where the strata gave way
    unevenly. -> (cliff, needle, top y and foot y for each x)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = np.arange(W, dtype=np.float32)
    s = 1 - x / VX
    top = HZ + (TOP + 26 * noise.line1d(W, 520, r) + 9 * noise.line1d(W, 110, r) + 3 * noise.line1d(W, 28, r) - HZ) * s
    for xc, h, hw in TURRETS:
        top -= h * (1 + 0.15 * noise.line1d(W, 9, r)) * np.clip(1 - np.abs(x - xc) / hw, 0, 1) ** 0.35
    foot = HZ + (FOOT - HZ) * s + 2.5 * noise.line1d(W, 70, r)
    xe = int(TRUNK[-1][0])
    edge = np.vstack([[(-10, top[0])], np.stack([x[:XT], top[:XT]], 1), spline([(XT, top[XT])] + TRUNK, 240),
                      np.stack([x[xe::-1], foot[xe::-1]], 1), [(-10, foot[0])]])
    cliff = fill(edge) & ~fill(spline(OPENING, 400, closed=True))
    jog = 3 * noise.stretched((W, H), 6, 40, r).T + 7 * noise.field((H, W), 34, r) + 2 * noise.field((H, W), 8, r)
    soft = 1 - 0.8 * noise.smoothstep(70, 0, yy - top[None])                  # the turf edge is softer than the chalk
    cliff = (signed(cliff) + jog * soft < 0) & (yy < foot[None] + 1)

    xn, yb, yt, hw = NEEDLE
    t = (yb - yy) / (yb - yt)
    cx = xn - 18 * t + 6 * np.sin(2.4 * t) + 3 * noise.line1d(H, 70, r)[:, None]
    half = hw * np.clip(1 - t, 0, 1) ** 0.6 * (1 + 0.1 * np.sin(np.pi * np.clip(t, 0, 1))) + 4
    needle = (np.abs(xx - cx) < half) & (t < 1.0) & (yy < yb)
    needle = (signed(needle) + 0.6 * jog < 0) & (yy < yb + 1)
    return cliff, needle, top.astype(np.float32), foot.astype(np.float32)


def scene(r):
    """The design: what each place adds up to across the room, the light on the chalk, the way the
    strokes run, and where the sea breaks. -> dict"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cliff, needle, top, foot = rock(r)
    solid = cliff | needle
    sea = (yy > HZ) & ~solid
    sky = (yy <= HZ) & ~solid
    sd = np.minimum(signed(cliff | (yy > foot[None])), signed(needle | (yy > NEEDLE[1])))
    gy, gx = np.gradient(ndimage.gaussian_filter(sd, 3))
    gn = np.hypot(gx, gy) + 1e-6

    # the face of the cliff, turned a little away from the sun: broad buttresses running from the
    # top to the foot, a few clefts between them, the strata stepping back in ledges, bosses and
    # hollows; the upper face leans back into the light
    s = 1 - xx / VX
    u, v = xx / s, HZ + (yy - HZ) / s
    ub = u + 50 * noise.field((H, W), 600, r)
    relief = np.interp(ub, np.arange(7000), 40 * noise.line1d(7000, 380, r) + 9 * noise.line1d(7000, 110, r)) * s
    im = Image.new("F", (W, H), 0.0)
    d = ImageDraw.Draw(im)
    for uc in np.arange(260, 2700, 440) + r.uniform(-120, 120, 6):
        xc = uc * VX / (VX + uc)
        y0, y1 = top[int(xc)] + r.uniform(0, 120), foot[int(xc)] - r.uniform(0, 260)
        q = np.linspace(0, 1, 14)
        px = xc + np.cumsum(r.normal(0, 5, 14)) + 18 * np.sin(q * r.uniform(2, 5) + r.uniform(0, 6))
        for k in range(13):
            d.line([(px[k], y0 + (y1 - y0) * q[k]), (px[k + 1], y0 + (y1 - y0) * q[k + 1])],
                   fill=float(r.uniform(0.5, 1)), width=int(r.integers(5, 16) * (1 - xc / VX)))
    for _ in range(45):                                                     # and the lesser cracks
        x0, y0 = r.uniform(0, 1880), r.uniform(150, 1050)
        L, a = r.uniform(40, 220), np.pi / 2 + r.normal(0, 0.3)
        q = np.linspace(0, 1, 8)
        d.line(list(zip(x0 + L * q * np.cos(a) + np.cumsum(r.normal(0, 4, 8)), y0 + L * q * np.sin(a))),
               fill=float(r.uniform(0.3, 0.7)), width=int(r.integers(2, 5)))
    crack = ndimage.gaussian_filter(np.asarray(im), 3.0) * solid
    g = (v / 58 + 0.25 * noise.field((H, W), 400, r)) % 1
    flint = noise.smoothstep(0.07, 0.0, np.minimum(g, 1 - g)) * noise.smoothstep(-0.6, 0.8, noise.field((H, W), 70, r))
    relief = relief - 30 * crack * s - 10 * g * noise.smoothstep(-0.5, 1.2, noise.field((H, W), 180, r)) * s \
        + 9 * noise.fbm((H, W), 90, r, octaves=4)
    ry, rx = np.gradient(ndimage.gaussian_filter(relief, 2))
    up_face = np.clip((foot[None] - yy) / (foot[None] - top[None]), 0, 1)
    nx, ny, nz = 0.5 - 1.1 * rx, -1.1 * ry - 0.35 * up_face ** 2, np.ones_like(rx)
    reach = np.where(needle, 80, np.where(xx > 1640, 60, 75))
    turn = noise.smoothstep(reach, 0, -sd) * solid                          # rounding over at the edges
    nx, ny, nz = nx + (1.6 * gx / gn - nx) * turn, ny + (1.6 * gy / gn - ny) * turn, nz * (1 - 0.85 * turn)
    lit = (nx * SUN[0] + ny * SUN[1] + nz * SUN[2]) / np.sqrt(nx * nx + ny * ny + nz * nz)
    light = noise.smoothstep(0.25, 0.85, lit + 0.05 * noise.field((H, W), 50, r)) * solid

    # the chalk: cream, apricot and rose in the sun, dusty rose turning from it, lilac and blue in
    # shadow and blue-green low down where the sea lights it from below; grass along the top
    # running down the gullies; weed and wet chalk at the foot
    w1 = noise.smoothstep(-0.6, 1.0, noise.field((H, W), 380, r))
    w2 = noise.smoothstep(-0.2, 1.4, noise.field((H, W), 240, r))
    sun_c = mix(mix(SUNLIT[0], SUNLIT[1], 0.7 * w1), SUNLIT[2], 0.5 * w2)
    half_c = mix(HALF[0], HALF[1], noise.smoothstep(-1, 1, noise.field((H, W), 200, r)))
    shade_c = mix(mix(SHADE[0], SHADE[1], noise.smoothstep(-1, 1, noise.field((H, W), 300, r))), SHADE[2],
                  0.65 * noise.smoothstep(600, 1100, yy))
    soffit = noise.smoothstep(-50, 0, sd) * (yy < 900) * (xx > 1500) * (xx < 1800) * (1 - light)
    shade_c = mix(shade_c, DEEP[2], 0.5 * soffit)
    chalk = mix(mix(shade_c, half_c, noise.smoothstep(0, 0.5, light)), sun_c, noise.smoothstep(0.45, 1, light))
    chalk = mix(chalk, DEEP[0], 0.5 * noise.smoothstep(0.15, 0.6, crack))
    below = yy - top[None]
    thick = np.clip(40 * s + 16 * noise.field((H, W), 50, r), 12, 80)
    strip = noise.smoothstep(0.9, 1.7, noise.stretched((H, W), 16, 200, r) + 1.5 * crack)
    reach_down = 60 + 200 * noise.smoothstep(-0.5, 1.5, noise.field((H, W), 120, r))
    turf = np.maximum(noise.smoothstep(thick + 5, thick - 5, below),
                      strip * noise.smoothstep(reach_down, 0.5 * reach_down, below))
    turf *= cliff * (xx < 1460 + 30 * noise.field((H, W), 60, r))
    chalk = mix(chalk, mix(TURF_SHADE[0], TURF[0], light), turf)
    band = np.clip(44 * s + 10 * noise.field((H, W), 30, r), 12, 80)
    wet = noise.smoothstep(band, 0.4 * band, foot[None] - yy) * cliff
    wet = np.maximum(wet, noise.smoothstep(34, 10, NEEDLE[1] - yy) * needle)
    chalk = mix(chalk, mix(WRACK[0], WRACK[1], w1), 0.85 * wet)
    chalk = mix(chalk, mix(SKY[0], SKY[2], 0.5), 0.14 * needle + 0.07 * (xx > 1600) * cliff)   # the air between

    # the sky: warm and pale at the horizon, cooler overhead, warmer toward the sun; soft clouds lit
    # on the side toward it
    hgt = np.clip((HZ - yy) / HZ, 0, 1)
    air = ramp(SKY, hgt ** 0.85)
    air = mix(air, SKY[0] * np.float32([1.02, 1.0, 0.96]), 0.35 * noise.smoothstep(2000, 0, xx) * (1 - hgt))
    puff = noise.stretched((W, H), 110, 420, r).T + 0.5 * noise.field((H, W), 80, r)
    cloud = noise.smoothstep(0.7, 1.7, puff) * np.exp(-((hgt - 0.6) / 0.22) ** 2) * noise.smoothstep(700, 1900, xx)
    side = noise.smoothstep(-0.05, 0.35, np.roll(puff, (-14, -34), (0, 1)) - puff)
    air = mix(air, mix(CLOUD[2], CLOUD[0], side), 0.75 * cloud)

    # the sea: pale under the horizon with the warm sky in it, deep blue-green toward us, green
    # here and violet there; milky turquoise where it runs over the chalk below the cliff; long
    # bands where the sky lies on it and troughs between, and the waves in broken rows, larger as
    # they come on
    z = np.clip((yy - HZ) / (H - HZ), 0, 1)
    water = ramp(SEA, z ** 0.7)
    water = mix(water, GLOW[1], 0.55 * np.exp(-z / 0.05))
    water = mix(water, DARKSEA[3], 0.35 * z * noise.smoothstep(0.0, 1.5, noise.field((H, W), 280, r)))
    water = mix(water, DARKSEA[4], 0.3 * z * noise.smoothstep(0.2, 1.6, noise.field((H, W), 320, r)))
    near = ndimage.distance_transform_edt(~solid).astype(np.float32)
    milk = noise.smoothstep(170 * (0.5 + z), 0, near) * noise.smoothstep(-1.2, 0.6, noise.field((H, W), 150, r))
    water = mix(water, mix(TURQ[0], TURQ[2], z), 0.6 * milk)
    rows = np.cumsum(1 / (2.5 + 150 * np.clip((np.arange(H) - HZ) / (H - HZ), 0, 1) ** 1.5))
    phase = 2 * np.pi * (rows[:, None] + 0.55 * noise.field((H, W), 520, r) + 0.2 * noise.field((H, W), 160, r)
                         + 0.0004 * xx)
    band = np.cos(2 * np.pi * (rows[:, None] / 3.5 + 0.35 * noise.field((H, W), 700, r)
                               + 0.12 * noise.field((H, W), 200, r)))
    patchy = noise.smoothstep(-0.8, 1.0, noise.field((H, W), 450, r))
    water = mix(water, mix(SKY[2], SKY[3], 0.5) * np.float32(0.92),
                0.38 * noise.smoothstep(0.2, 1, band) * patchy * (1 - 0.6 * z))       # the sky lying on the water
    water = mix(water, DARKSEA[2], 0.42 * noise.smoothstep(0.2, 1, -band) * patchy * np.sqrt(z))
    swell = noise.smoothstep(-0.8, 1.0, noise.field((H, W), 300, r))
    back = noise.smoothstep(0.3, 1, np.cos(phase + 1.0)) * swell
    front = noise.smoothstep(0.2, 0.95, np.cos(phase - 1.3)) * swell
    chop = noise.smoothstep(0.75, 1.0, np.cos(phase)) * swell
    water = mix(water, TURQ[1] * 0.8, 0.12 * back * z)
    water = mix(water, DARKSEA[1], 0.18 * front * noise.smoothstep(0.0, 0.4, z))

    want = np.where(solid[..., None], chalk, np.where(sky[..., None], air, water)).astype(np.float32)
    want *= np.exp(0.05 * np.stack([noise.field((H, W), 220, r) for _ in range(3)], -1)).astype(np.float32)

    # where the sea breaks: in bursts along the foot of the rock, most at the leg of the arch and
    # round the Needle; the spray goes up the rock, the foam spreads out over the water
    x1 = xx[0]
    burst = 0.45 * noise.smoothstep(1.0, 2.2, noise.line1d(W, 140, r) + 0.7 * noise.line1d(W, 40, r))
    burst = np.clip(np.maximum(burst, np.exp(-((x1 - 1820) / 90) ** 2) + np.exp(-((x1 - 2140) / 110) ** 2)
                               + 0.7 * np.exp(-((x1 - 1530) / 50) ** 2)), 0, 1)
    line = np.where((x1 > 2020) & (x1 < 2260), NEEDLE[1], foot)
    rise = line[None] - yy
    spray = noise.smoothstep(40 + 110 * burst[None], 0, rise) * (rise > -4) * burst[None] \
        * noise.smoothstep(-0.8, 0.6, noise.field((H, W), 45, r))
    foam = noise.smoothstep(45 * (0.3 + burst[None]), 0, near) * sea * (0.3 + 0.7 * burst[None]) * (yy < 1300)

    # the way the strokes run. Over the chalk mostly along the strata, in patches up and down the
    # buttresses or slantwise across them; down the clefts, round the edges of the rock, along the
    # top; over the sea along the waves; the sky in long slanting strokes, level at the horizon
    strata = np.arctan2(HZ - yy, VX - xx)
    m = noise.field((H, W), 170, r)
    up = np.pi / 2 + 0.25 * noise.field((H, W), 110, r)
    ang = strata + 0.15 * noise.field((H, W), 90, r)
    ang = np.where(m > 0.75, up, ang)
    ang = np.where(m < -0.9, strata + 0.7 * np.sign(noise.field((H, W), 200, r)), ang)
    ang = np.where(crack > 0.35, up, ang)
    ang = np.where(turn > 0.5, np.arctan2(gx, -gy), ang)
    ang = np.where(turf > 0.5, np.where(strip > 0.5, up, strata + 0.2 * noise.field((H, W), 60, r)), ang)
    gp_y, gp_x = np.gradient(ndimage.gaussian_filter(phase, 4))
    crest = np.arctan2(gp_x, -gp_y)
    slant = -0.3 + 0.2 * noise.field((H, W), 600, r)
    ang = np.where(sky, np.where(hgt < 0.15, 0.05 * noise.field((H, W), 300, r), slant), np.where(sea, crest, ang))
    key = np.where(sky, 0, np.where(sea, 1, np.where(needle, 4, np.where(turf > 0.5, 3, 2)))).astype(np.int32)
    return dict(want=want, solid=solid, sea=sea, sky=sky, light=light, turf=turf, crack=crack, flint=flint, z=z,
                hgt=hgt, cloud=cloud, milk=milk, front=front, chop=chop, foot_at=line, rise=rise, spray=spray, foam=foam,
                ang=ang.astype(np.float32), strata=strata.astype(np.float32), up=up.astype(np.float32), key=key)


def stay(paths, key):
    """Cut each stroke back to where `key` (H,W labels) first differs from its value at the start,
    and lay it out again over as many points."""
    N, n, _ = paths.shape
    t = np.linspace(0, n - 1, 4 * n)
    i0 = np.minimum(t.astype(int), n - 2)
    fine = paths[:, i0] * (1 - (t - i0))[None, :, None] + paths[:, i0 + 1] * (t - i0)[None, :, None]
    lab = key[np.clip(fine[..., 1].astype(int), 0, H - 1), np.clip(fine[..., 0].astype(int), 0, W - 1)]
    out = lab != lab[:, :1]
    end = np.maximum(t[np.where(out.any(1), np.argmax(out, 1), 4 * n - 1)], 0.6)
    q = np.linspace(0, 1, n)[None, :] * end[:, None]
    j0 = np.minimum(q.astype(int), n - 2)
    g = (q - j0)[..., None]
    return np.take_along_axis(paths, j0[..., None], 1) * (1 - g) + np.take_along_axis(paths, (j0 + 1)[..., None], 1) * g


def paint(seed=1885):
    r = noise.rng(seed)
    S = scene(r)
    want, key = S["want"], S["key"]
    light, solid, sea, sky, turf, z = S["light"], S["solid"], S["sea"], S["sky"], S["turf"], S["z"]
    ground = canvas.duck((H, W), seed, tint="#dccfb6", thread=3.0)
    # the canvas rubbed over first with thin colour of each place, the priming glowing through here and there
    rag = noise.stretched((H, W), 14, 120, r)
    alpha = np.clip(0.9 + 0.05 * rag + 0.08 * (0.5 - ground.tooth), 0, 1)[..., None]
    rgb = np.ascontiguousarray(ground.color * (1 - alpha) + want * alpha, np.float32)
    height = ground.tooth * 0.25
    wet = np.ones((H, W), np.float32)

    def dry_out():
        """Let the paint set between sittings. A trace of wetness is left everywhere: a brush that
        reaches wet paint from dry then reads the colour under it truly, instead of dragging in black."""
        wet[:] = 0.02

    def whites(P, cool=0.35):
        """A brush of white for foam: lead white, warm where the sun is on it, and blue-grey."""
        n = len(P)
        pick = lambda: np.where((r.random(n) < cool)[:, None], FOAM_SHADE[r.integers(0, 4, n)], FOAM[r.integers(0, 4, n)])
        share = np.stack([r.uniform(0.45, 0.7, n), r.uniform(0.2, 0.4, n), r.uniform(0.05, 0.2, n)], 1)
        cols = np.stack([pick(), pick(), pick()], 1) * np.exp(r.normal(0, 0.03, (n, 3, 1)))
        return np.clip(cols, 0, 1).astype(np.float32), share

    def strew(spacing, odds, scale=None):
        """Where the strokes of a passage lie: a shaken honeycomb kept by `odds` (H,W), thinned
        where `scale` makes the strokes larger, so that they overlap alike everywhere."""
        lo = 1.0 if scale is None else float(scale.min())
        P = dabs.scatter((H + 2 * spacing, W + 2 * spacing), spacing * lo, r) - spacing
        keep = np.asarray(odds, np.float32)[at(P)] * (1 if scale is None else (lo / scale[at(P)]) ** 2)
        return P[r.random(len(P)) < keep]

    def tones(P, fams, weights, f=(0.3, 0.6), vary=0.06, accent=None, odds=0.1, base=None, value=1.0):
        """The brush for each stroke: the colour of the place (or `base`) pushed part of the way (f)
        toward a hue from one of the families, chosen by `weights` (a number or an (H,W) map for
        each), the hue brought first to the value of the place; a second such colour streaked
        beside it; and a little of the place itself, or now and then an accent from elsewhere."""
        base = np.clip((want[at(P)] if base is None else base) * value, 0, 1)
        n = len(P)
        lb = np.log(np.clip(base, 1e-4, 1))
        wts = np.stack([m[at(P)] if np.ndim(m) else np.full(n, m, np.float32) for m in weights], 1) + 1e-6
        cum = np.cumsum(wts / wts.sum(1, keepdims=True), 1)

        def one():
            k = np.minimum((r.random(n)[:, None] > cum).sum(1), len(fams) - 1)
            hue = np.empty((n, 3), np.float32)
            for j, fam in enumerate(fams):
                sel = k == j
                hue[sel] = fam[r.integers(0, len(fam), sel.sum())]
            hue *= ((base @ LUMA) / (hue @ LUMA + 1e-6))[:, None] * np.exp(r.normal(0, vary, (n, 1)))
            return np.exp(lb + (np.log(np.clip(hue, 1e-4, 1)) - lb) * r.uniform(*f, n)[:, None])
        third = base * np.exp(r.normal(0, vary, (n, 3)))
        if accent is not None:
            hit = r.random(n) < odds
            third[hit] = accent[r.integers(0, len(accent), hit.sum())]
        share = np.stack([r.uniform(0.45, 0.7, n), r.uniform(0.2, 0.4, n), r.uniform(0.05, 0.2, n)], 1)
        return np.clip(np.stack([one(), one(), third], 1), 0, 1).astype(np.float32), share

    def lay(paths, w, load, cut=key, **kw):
        """Lay strokes, each stopping where its passage (`cut`) ends, half of them drawn the other way."""
        if cut is not None:
            paths = stay(paths, cut)
        back = r.random(len(paths)) < 0.5
        paths[back] = paths[back, ::-1]
        impasto.lay(rgb, height, paths, w, load[0], r, share=load[1], wet=wet, **kw)

    def passage(odds, spacing, size, aspect, load, field=None, scale=None, tilt=0.2, wild=0.0, bend=0.0, curl=None,
                n=8, cut=key, **kw):
        """Strokes over a passage: half-widths spread about `size` (mean, spread), lengths `aspect`
        times as long, along `field` turned by `tilt` and now and then (`wild`) any way at all,
        bowed by `bend` (or turned through `curl` radians from end to end, either way, for commas
        and crescents), each stopping where its passage (`cut`) ends."""
        P = strew(spacing, odds, scale)
        N = len(P)
        if not N:
            return
        k = np.ones(N) if scale is None else scale[at(P)]
        w = np.clip(size[0] * np.exp(r.normal(0, size[1], N)), 0.45 * size[0], 2.4 * size[0]) * k
        a = r.normal(0, tilt, N) + np.where(r.random(N) < wild, r.uniform(-1.4, 1.4, N), 0)
        L = w * r.uniform(*aspect, N)
        turn = r.normal(0, bend, N) / np.sqrt(k) if curl is None else r.choice([-1, 1], N) * r.uniform(*curl, N) / L
        paths = impasto.follow(S["ang"] if field is None else field, P, L, n, turn, a - 0.5 * turn * L)
        lay(paths, w, load(P), cut, **kw)

    persp = (0.45 + 1.0 * z).astype(np.float32)           # the touches on the sea grow as they come toward us
    chalk = solid * (1 - turf)
    forms = key + 10 * (light > 0.5)                       # chalk in the sun and out of it
    level = (0.035 * noise.field((H, W), 300, r)).astype(np.float32)
    near = ndimage.distance_transform_edt(~solid).astype(np.float32)
    jolt = lambda P, sd: np.exp(r.normal(0, sd, (len(P), 1))).astype(np.float32)    # lighter or darker, stroke by stroke

    # the first sitting: the whole canvas laid in thin with a big brush, in the colours of each place
    passage(np.ones((H, W)), 130, (24, 0.25), (4, 9), lambda P: tones(P, [SKY], [1], (0.0, 0.1), 0.04),
            tilt=0.15, thick=0.02, grooves=0.12, lips=0.02, land=0.1, lift=0.05, pickup=0.65, merge=3, spent=0.7,
            fray=3, taper=0.4, hide=0.8, ends=(0.4, 0.2))
    height -= 0.8 * (height - 0.25 * ground.tooth)
    dry_out()

    # the second, wet into wet: the sky in long soft strokes, the sea in level strokes, the chalk
    # in strokes every way, and the grass along the top
    soft = dict(ends=(0.35, 0.15), taper=0.4, fray=2.4, spent=0.75)
    thin = dict(land=0.15, lift=0.1, lips=0.03, grooves=0.25)
    passage(sky, 44, (16, 0.35), (4, 12), lambda P: tones(P, [SKY, CLOUD], [1, 0.15 + S["cloud"]], (0.35, 0.65), 0.05),
            tilt=0.2, wild=0.12, bend=0.0015, thick=0.015, pickup=0.6, merge=3, hide=0.85, **thin, **soft)
    passage(sea, 34, (9, 0.5), (3, 16),
            lambda P: tones(P, [SEA, TURQ, DARKSEA], [1, (0.3 + S["milk"]) * noise.smoothstep(0.03, 0.2, z), 0.5 * z],
                            (0.2, 0.5), 0.05),
            field=level, scale=persp, tilt=0.04, bend=0.0015, thick=0.1, pickup=0.5, merge=2,
            **{**soft, "taper": 0.6, "ends": (0.3, 0.05)})
    passage(sea * noise.smoothstep(-0.3, 1.0, noise.field((H, W), 200, r)), 30, (5, 0.45), (2, 6),
            lambda P: tones(P, [SEA, TURQ, DARKSEA], [1, 0.5, 0.7 * z], (0.3, 0.6), 0.06),
            field=level, scale=persp, tilt=0.12, bend=0.01, thick=0.12, pickup=0.4, merge=1.5,
            **{**soft, "taper": 0.6, "ends": (0.3, 0.05)})
    passage(chalk, 25, (13, 0.4), (2.5, 7), lambda P: tones(P, [SUNLIT, HALF, SHADE], [light, 0.4, 1 - light],
                                                             (0.25, 0.55), 0.05, EARTH, 0.08),
            tilt=0.25, wild=0.15, bend=0.003, thick=0.2, pickup=0.45, merge=2, dry=0.1, **soft)
    passage(turf, 14, (6, 0.3), (2.5, 6), lambda P: tones(P, [TURF, TURF_SHADE], [light, 1 - light], (0.2, 0.5), 0.06,
                                                          SUNLIT[1:2], 0.12),
            tilt=0.3, bend=0.004, thick=0.2, pickup=0.4, merge=2, **soft)
    dry_out()
    height += 0.3 * noise.smoothstep(0.3, 1.6, noise.fbm((H, W), 7, r, octaves=2)) * solid

    # the third, over dry paint: smaller strokes nearer the colours of the tubes, laid every way
    # over the chalk and broken; the flint in the strata and the clefts drawn in darker; the sea in
    # short touches, a little bowed, darker in the troughs, choppier close in under the rock and
    # long and pale toward the horizon; the warm light dragged over the sky low down, and the chalk
    # caught in the water below it
    dry3 = dict(pickup=0, merge=0, **soft)
    patch = noise.smoothstep(-1.0, 1.0, noise.field((H, W), 130, r))
    grain = np.clip(np.exp(0.45 * noise.field((H, W), 300, r)), 0.6, 1.7).astype(np.float32)   # larger here, smaller there
    passage(chalk * (0.3 + 0.7 * patch), 21, (7.5, 0.45), (2.5, 8),
            lambda P: tones(P, [SUNLIT, HALF, SHADE, EARTH], [light, 0.3, 1 - light, 0.12], (0.45, 0.8), 0.06, HALF, 0.08),
            scale=grain, cut=forms, tilt=0.18, wild=0.12, bend=0.004, thick=0.24, dry=0.1, **dry3)
    passage(chalk * S["flint"], 10, (2.6, 0.35), (6, 20),
            lambda P: tones(P, [SHADE, DEEP, HALF], [1, 0.4, 0.5], (0.35, 0.7), 0.06, value=0.74),
            field=S["strata"], tilt=0.05, thick=0.1, dry=0.3, hide=0.75, **dry3)
    passage(noise.smoothstep(0.25, 0.6, S["crack"]) * chalk, 12, (4, 0.35), (5, 14),
            lambda P: tones(P, [DEEP, SHADE[:2]], [1, 0.6], (0.4, 0.8), 0.05, value=0.7),
            field=S["up"], tilt=0.15, thick=0.16, dry=0.2, **dry3)
    rough = noise.smoothstep(-0.4, 1.4, noise.field((H, W), 260, r))
    choppy = sea * noise.smoothstep(0.03, 0.15, z)
    passage(choppy * (0.4 + 0.6 * rough), 24, (5.5, 0.45), (2.5, 7),
            lambda P: tones(P, [SEA[2:], TURQ, DARKSEA, VIOLET], [1, 0.4 + S["milk"], 0.7 * z, 0.4 * z], (0.4, 0.75),
                            0.05, value=jolt(P, 0.16)),
            field=level, scale=persp, tilt=0.12, curl=(0.15, 0.8), thick=0.15, dry=0.1, **dry3)
    passage(choppy * (0.2 + 0.8 * S["front"]) * rough, 28, (5, 0.45), (2.5, 6),
            lambda P: tones(P, [DARKSEA, VIOLET, SEA[4:]], [1, 0.3 + 0.6 * z, 0.6], (0.45, 0.8), 0.06, value=0.75),
            field=level, scale=persp, tilt=0.15, curl=(0.2, 1.0), thick=0.16, dry=0.1, **dry3)
    passage(sea * noise.smoothstep(220, 40, near), 14, (4.5, 0.4), (2, 5),
            lambda P: tones(P, [TURQ, SEA[2:4], VIOLET], [1, 0.6, 0.3], (0.45, 0.8), 0.06, value=jolt(P, 0.2)),
            field=level, scale=persp, tilt=0.4, curl=(0.6, 1.8), thick=0.16, dry=0.1, **dry3)
    passage(sea * noise.smoothstep(0.12, 0.0, z), 18, (4, 0.4), (5, 14),
            lambda P: tones(P, [SEA[:2], GLOW, SKY[2:4]], [1, 0.6, 0.5], (0.4, 0.75), 0.05),
            field=level, scale=persp, tilt=0.03, bend=0.001, thick=0.08, dry=0.15, **dry3)
    passage(sky * noise.smoothstep(0.5, 0.0, S["hgt"]) * 0.6, 60, (14, 0.3), (4, 9),
            lambda P: tones(P, [SKY[:2], CLOUD[:1]], [1, 0.4], (0.5, 0.8), 0.03), tilt=0.1, thick=0.015, dry=0.45,
            hide=0.6, **thin, **dry3)
    below = sea * noise.smoothstep(170, 40, near) * noise.smoothstep(-0.6, 0.8, noise.field((H, W), 60, r))
    P = strew(16, below)
    y, x = at(P)
    mirror = want[np.clip(2 * S["foot_at"][x] - y - 40, 0, H - 1).astype(int), x]
    mirror = np.sqrt(np.clip(mirror, 1e-4, 1) * want[y, x])                 # half the chalk above, half the sea
    lay(impasto.follow(level, P, r.uniform(15, 45, len(P)), 5, 0.0, r.normal(0, 0.04, len(P))),
        r.uniform(2.5, 5, len(P)), tones(P, [SUNLIT[1:3], HALF], [1, 0.6], (0.2, 0.4), 0.05, base=mirror),
        thick=0.12, dry=r.uniform(0.2, 0.5, len(P)), hide=0.8, **dry3)
    dry_out()

    # the last: the sun on the chalk in thick paint; the spray going up the rock in veils of white,
    # thicker low down; the sky caught pale on the crests where the sea is rough, white curls close
    # in under the rock, and the foam heaped thick at its foot
    passage(chalk * noise.smoothstep(0.55, 0.95, light), 24, (8, 0.4), (1.5, 5),
            lambda P: tones(P, [SUNLIT[[0, 3, 4]], SUNLIT[1:2]], [1, 0.4], (0.5, 0.9), 0.04, value=1.08),
            cut=forms, tilt=0.35, wild=0.25, bend=0.003, thick=0.48, dry=0.2, **dry3)
    for spacing, size, hide, thick, top, spin in ((16, (16, 0.3), 0.42, 0.02, 1.0, (0.3, 1.0)),
                                                  (12, (9, 0.35), 0.62, 0.05, 0.6, (0.4, 1.3)),
                                                  (9, (5, 0.35), 0.92, 0.3, 0.25, (0.6, 1.8))):
        passage(S["spray"] * noise.smoothstep(top * 150, top * 50, S["rise"]), spacing, size, (1.2, 2.5), whites,
                field=S["up"], tilt=0.8, curl=spin, cut=None, thick=thick, dry=0.1, hide=hide, **dry3)
    crests = S["chop"] * noise.smoothstep(0.1, 0.9, rough * noise.smoothstep(-1.0, 0.8, noise.field((H, W), 70, r)))
    passage(sea * crests * noise.smoothstep(0.06, 0.2, z), 16, (3.4, 0.45), (3, 9),
            lambda P: tones(P, [SKY[2:4], TURQ[3:], FOAM_SHADE], [1, 0.6, 0.5], (0.6, 0.9), 0.05, value=1.5),
            field=level, scale=persp, tilt=0.1, curl=(0.1, 0.7), cut=None, thick=0.22, dry=0.2, hide=0.8, **dry3)
    curls = noise.smoothstep(150, 20, near) * (0.3 + S["chop"]) * noise.smoothstep(-0.5, 0.9, noise.field((H, W), 60, r))
    passage(sea * curls, 12, (3.4, 0.45), (2.5, 7), lambda P: whites(P, 0.5), field=level, scale=persp, tilt=0.25,
            curl=(0.2, 1.2), cut=None, thick=0.3, dry=0.2, hide=0.85, **dry3)
    passage(S["foam"], 8, (6, 0.4), (1.5, 4), whites, field=level, tilt=0.35, curl=(0.3, 1.3), cut=None, thick=0.45,
            dry=0.1, **dry3)

    height = ndimage.gaussian_filter(height, 0.7)
    img = dabs.shine(rgb, height, light=LIGHT, relief=0.7, gloss=0, reach=(0.86, 1.13))
    return img + 0.04 * impasto.glints(height, LIGHT)[..., None]
