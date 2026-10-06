"""Impasto: thick oil paint, laid stroke by stroke from a loaded brush.

A painter working fast loads the brush with one colour and touches it into one or two
more, so that a single stroke carries them side by side and shows them in streaks along
its length, unmixed. The paint is a stiff paste and stands up from the canvas. It piles up
where the brush lands, is combed into deep grooves by the bristles, and thins as the load
runs out, until near the end of the stroke it breaks into dry streaks that catch only the
high points of whatever lies beneath; the bristles splay there and the edges go ragged.
A stroke laid across wet paint drags it along and takes up its colour, so neighbours merge
in places instead of sitting apart. The strokes go on one after another, each on those
before it. A canvas worked over months goes on over paint that has dried: that no longer
drags or merges, and its ridges stand up under whatever covers them. A brush dragged nearly
dry across it, a scumble, leaves colour only on the high points, and the colours beneath
show between. Light the result with `dabs.shine` (matt) and `glints`.
"""

import numpy as np
from scipy import ndimage, signal

from . import brush, noise


def follow(angle, seeds, length, n=12, bend=0.0, tilt=0.0):
    """Paths along a field of directions. From each seed (x, y) take n-1 equal steps along
    `angle` (H,W, radians) turned by `tilt`, turning a further `bend` radians per px for a
    curl, and never doubling back. `length`, `bend`, `tilt`: scalars or one per seed. -> (N, n, 2)"""
    h, w = angle.shape
    N = len(seeds)
    step = np.broadcast_to(np.asarray(length, np.float32), (N,)) / (n - 1)
    turn = np.broadcast_to(np.asarray(bend, np.float32), (N,)) * step
    tilt = np.broadcast_to(np.asarray(tilt, np.float32), (N,))
    P = np.empty((N, n, 2), np.float32)
    P[:, 0] = seeds
    d = None
    for k in range(1, n):
        x, y = P[:, k - 1].T
        a = angle[np.clip(y.astype(int), 0, h - 1), np.clip(x.astype(int), 0, w - 1)] + tilt + turn * (k - 1)
        e = np.stack([np.cos(a), np.sin(a)], 1)
        if d is not None:
            e *= np.where((e * d).sum(1, keepdims=True) < 0, -1, 1)
        P[:, k] = P[:, k - 1] + e * step[:, None]
        d = e
    return P


def bristles(r, shape=(4096, 512)):
    """The marks of bristles in stroke space, a pixel to a step: rows run along a stroke,
    columns across it. Narrow streaks drawn out lengthways, as every brush leaves them."""
    return noise.stretched(shape, 1.7, 60, r)


def _look(B, along, across):
    return ndimage.map_coordinates(B, [along, across], order=1, mode="grid-wrap")


def lay(rgb, height, paths, width, colours, r, share=None, thick=0.25, spent=0.6, grooves=0.9, lips=0.15,
        land=0.6, lift=0.4, tails=0.9, pickup=0.5, merge=3.0, ends=(0.15, 0.05), taper=0.2, fray=1.2, flatten=0.8,
        wet=None, dry=0.0, hide=1.0):
    """Lay strokes, in order, into rgb (H,W,3) and height (H,W), in place.

    paths    one (n, 2) array of points (x, y) per stroke, in the order the brush moves
    width    half-width of each stroke, px (scalar or one per stroke)
    colours  (N, K, 3) the load of each brush, linear RGB: K colours that streak along it
    share    (N, K) how much of the brush each colour holds (default equal)
    thick    paint height at full load, as a fraction of the half-width (heights are in px;
             paint lower than about 0.1 px counts as dry)
    spent    how much of its load the brush has used by the end of the stroke; the last
             stretch thins and breaks into dry streaks
    grooves  depth of the bristle grooves, px
    lips     paint pushed up along the edges, as a fraction of the body
    land     extra paint piled where the brush lands
    lift     the ridge where it lifts away
    tails    how far single bristles drag on past the end, as a fraction of the half-width
    pickup   how much wet paint from underneath the brush drags into itself
    merge    px over which an edge laid on wet paint drags into it instead of stopping clean
    ends     roundness of the start and the end (0 square, 1 round)
    taper    how much the stroke narrows as the brush lifts
    fray     raggedness of the edges, px, growing as the bristles splay toward the end
    flatten  how far a stroke presses flat the relief of the paint it rides over
    wet      (H,W) how wet the paint on the canvas is, 0 dry to 1 fresh, kept up in place as strokes
             go on. Zero it between sittings: the next sitting works over dry paint, which it can
             neither drag along nor press flat; the new paste fills its hollows, and ridges higher
             than the new paint stand up through it.
             Without it, all the paint counts as wet, judged by its height.
    dry      how little paint the brush carries from the start (scalar or one per stroke): dragged
             thin, a scumble breaks all along its length and catches only the high points of the
             weave and of the paint beneath
    hide     how well the paint covers (1 opaque; less, a veil through which the colour beneath shows)
    """
    H, W = height.shape
    N, K = colours.shape[:2]
    width = np.broadcast_to(np.asarray(width, np.float64), (N,))
    dry = np.broadcast_to(np.asarray(dry, np.float64), (N,))
    share = np.ones((N, K)) if share is None else np.asarray(share, np.float64)
    cuts = np.cumsum(share / share.sum(1, keepdims=True), 1)[:, :-1]
    logc = np.log(np.clip(colours, 1e-4, 1))
    B = bristles(r)
    off = r.uniform(0, 1, (N, 12)) * np.tile(B.shape, 6)
    load = thick * np.exp(r.normal(0, 0.15, N))
    used = np.clip(spent * r.uniform(0.6, 1.3, N), 0, 1)
    tilt = r.uniform(-0.35, 0.35, N)        # a brush held at a slant leaves one edge higher
    side = r.uniform(-1, 1, N)              # colours side by side (+-1) or threaded through each other (0)
    belly = r.uniform(0, 0.45, N)           # some strokes swell in the middle and narrow at both ends
    ds, du = 0.5, 0.7
    keep = np.exp(-ds / 45.0)               # what the brush picked up is spent over ~45 px
    lanes = np.array([-0.7, 0.0, 0.7])
    frgb, fh = rgb.reshape(-1, 3), height.reshape(-1)
    fw = None if wet is None else wet.reshape(-1)
    damp = (lambda j: noise.smoothstep(0.05, 0.3, fh[j])) if fw is None else (lambda j: fw[j])
    claim = np.empty(H * W, np.int32)       # which grid point owns each pixel, so each is painted once
    for i in range(N):
        w = width[i]
        C = brush.path(np.asarray(paths[i], np.float64), ds)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        L, S = s[-1], len(C)
        if L < 1:
            continue
        T = ndimage.gaussian_filter1d(np.gradient(C, axis=0), 3, axis=0, mode="nearest")
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        Nn = T[:, ::-1] * [-1, 1]
        # what lies under the brush, read in three lanes along the path before it is covered
        li = (np.clip(np.rint(C[:, None, 1] + Nn[:, None, 1] * lanes * w), 0, H - 1) * W
              + np.clip(np.rint(C[:, None, 0] + Nn[:, None, 0] * lanes * w), 0, W - 1)).astype(np.int64)
        moist = damp(li)
        num = signal.lfilter([1 - keep], [1, -keep], frgb[li] * moist[..., None], axis=0)
        den = signal.lfilter([1 - keep], [1, -keep], moist, axis=0)
        pick = np.log(np.clip(num / np.maximum(den, 1e-9)[..., None], 1e-4, 1))
        amt = pickup * den
        level = ndimage.gaussian_filter1d(fh[li].mean(1), w / ds, mode="nearest")   # the surface it rides on
        # the pixels it may touch: a fine grid along and across it, run on past both ends and both edges
        e = int(np.ceil((w + 2) / ds))
        ext = np.arange(1, e + 1)[:, None] * ds
        Ce = np.vstack([C[0] - T[0] * ext[::-1], C, C[-1] + T[-1] * ext])
        Te = np.vstack([np.repeat(T[:1], e, 0), T, np.repeat(T[-1:], e, 0)])
        Ne = Te[:, ::-1] * [-1, 1]
        se = np.concatenate([-ext[::-1, 0], s, L + ext[:, 0]])
        u = np.arange(-w - 2 - merge, w + 2.1 + merge, du)
        X = np.rint(Ce[:, None, 0] + Ne[:, None, 0] * u).astype(np.int64)
        Y = np.rint(Ce[:, None, 1] + Ne[:, None, 1] * u).astype(np.int64)
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        k = np.nonzero(ok.ravel())[0]
        if not len(k):
            continue
        idx = (Y * W + X).ravel()[k]
        claim[idx] = np.arange(len(k))
        mine = claim[idx] == np.arange(len(k))
        idx, k = idx[mine], k[mine] // len(u)
        dx, dy = idx % W - Ce[k, 0], idx // W - Ce[k, 1]
        sp = se[k] + dx * Te[k, 0] + dy * Te[k, 1]          # how far along the stroke
        up = dx * Ne[k, 0] + dy * Ne[k, 1]                  # and how far across it
        t = np.clip(sp / L, 0, 1)
        o = off[i]
        press = 1 + 0.25 * _look(B, sp * 0.6 + o[6], np.full_like(sp, o[7]))   # pressure comes and goes
        wl = w * press * (1 + 0.12 * np.exp(-np.maximum(sp, 0) / w)) * (1 - taper * noise.smoothstep(L - 2 * w, L, sp)) \
            * (1 - belly[i] * (1 - np.sqrt(np.sin(np.pi * np.clip(t, 0.02, 0.98)))))
        v = up / wl
        nd = _look(B, sp * 2.5 + o[0], up * 0.9 + o[1])     # dry skips and splayed bristles
        lump = _look(B, sp * 2 + o[10], up * 0.15 + o[11])  # paint heaped unevenly
        ng = _look(B, sp * 0.8 + o[2], up * 0.7 + 1.5 * lump + o[3])   # grooves, wandering a little
        nc = _look(B, sp * 0.1 + o[4], up * 0.35 + o[5])    # streaks of colour
        edge = _look(B, sp + o[8], np.sign(v) * 90 + o[9])  # each edge wavers, and heaps up, on its own
        q = np.sqrt(np.clip(1 - v * v, 0, 1))
        tail = tails * w * noise.smoothstep(0.2, 1.6, ng)   # bristles that drag on as the brush lifts
        trail = noise.smoothstep(L + 0.5, L - 0.5 * w, sp)
        hold = fh[idx]
        rag = fray * (1 + 1.5 * t) * nd
        soft = 1 + merge * damp(idx) * noise.smoothstep(-0.3, 1.2, edge)
        cover = (np.clip((wl * (1 + 0.12 * edge) - np.abs(up) + rag) / soft + 0.5, 0, 1)
                 * np.clip((sp - ends[0] * w * (1 - q) + 0.6 * rag) / 1.5 + 0.5, 0, 1)
                 * np.clip(L + tail - ends[1] * w * (1 - q) - sp + 0.5, 0, 1) * (0.55 + 0.45 * trail))
        # the paint left on the brush; once it runs low only the high points underneath catch it.
        # A dry brush touches with its middle more than its sides, and skips more by the relief than by the bristles
        left = (1 - dry[i]) * (1 - used[i] * t * t) * (1 - 0.6 * dry[i] * np.minimum(v * v, 1))
        catch = 0.12 * np.clip((hold - hold.mean()) / (hold.std() + 0.02), -2, 2)
        a = cover * noise.smoothstep(-0.1, 0.1, left - 0.5 + (0.12 + 0.4 * (1 - left)) * (1 - 0.6 * dry[i]) * nd
                                     + catch * (1 - left) * (1 + 3 * dry[i]))
        av = np.abs(v)
        dome = noise.smoothstep(1.0, 0.8, av) ** 0.4 * (1 + tilt[i] * v)     # a slab of paste, steep at the edge
        lip = lips * noise.smoothstep(-1, 1.5, edge) * np.exp(-((av - 0.78) / 0.13) ** 2)
        body = (0.35 + 0.65 * left) * (1 + land * np.exp(-np.maximum(sp, 0) / (0.8 * w))) * (1 + 0.3 * lump)
        comb = grooves * noise.smoothstep(0.5, 0.0, np.abs(ng))
        ridge = lift * np.exp(-((sp - L + 0.6 * w) / (0.3 * w + 1)) ** 2) * (0.6 + 0.4 * np.tanh(ng))
        paint = np.maximum(dome * (load[i] * w * (body + ridge + lip) - comb), 0) * press * (0.25 + 0.75 * trail)
        # the load, in streaks: each colour holds its share of the width
        f = 0.5 + 0.5 * np.clip(side[i] * v + (1.15 - abs(side[i])) * (0.85 * nc + 0.1 * ng), -1, 1)
        lc = np.repeat(logc[i, :1], len(idx), 0)
        for j in range(1, K):
            lc += (logc[i, j] - lc) * noise.smoothstep(cuts[i, j - 1] - 0.015, cuts[i, j - 1] + 0.015, f)[:, None]
        # mixed with what it dragged up, in streaks, as paints mix: by multiplying, not by averaging
        r0 = np.clip(k - e, 0, S - 1)
        lp = np.clip((v + 0.7) / 0.7, 0, 1.999)
        l0 = lp.astype(int)
        fl = lp - l0
        # the two lanes it straddles, each weighted by how much it gives up: a dry lane gives nothing
        w0, w1 = amt[r0, l0] * (1 - fl), amt[r0, l0 + 1] * fl
        pk = (pick[r0, l0] * w0[:, None] + pick[r0, l0 + 1] * w1[:, None]) / (w0 + w1 + 1e-12)[:, None]
        am = (w0 + w1) * (0.5 + noise.smoothstep(-1, 1, nc))
        col = np.exp(lc + (pk - lc) * np.clip(am, 0, 0.9)[:, None] + 0.015 * ng[:, None])
        frgb[idx] += (col - frgb[idx]) * (a * hide)[:, None]
        base = level[r0]
        if fw is None:
            fh[idx] = hold + (base + (hold - base) * (1 - flatten) + paint - hold) * a
        else:   # wet relief is pressed flat; on dry paint the paste fills the hollows and rides over the ridges
            fresh = fw[idx]
            top = fresh * (base + (hold - base) * (1 - flatten) + paint) + (1 - fresh) * np.maximum(hold, base + paint)
            fh[idx] = hold + (top - hold) * a
            fw[idx] += (1 - fresh) * a


def glints(height, light=(-0.6, -0.5, 0.62), sharp=160, crest=0.4):
    """Oil paint is mostly matt; the light catches it only on the crests of the ridges, in
    small bright points where the surface turns toward the lamp. -> (H,W), to add to the
    image (lit by `dabs.shine` with gloss=0) in whatever strength the varnish allows."""
    gy, gx = np.gradient(height)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.asarray(light, np.float32) / np.linalg.norm(light)
    half = (L + [0, 0, 1]) / np.linalg.norm(L + [0, 0, 1])
    ridge = noise.smoothstep(0, crest, -ndimage.laplace(ndimage.gaussian_filter(height, 1.0)))
    return (np.clip(n @ half.astype(np.float32), 0, 1) ** sharp * ridge).astype(np.float32)


if __name__ == "__main__":
    r = noise.rng(0)
    rgb = np.full((80, 240, 3), 0.8, np.float32)
    height = np.zeros((80, 240), np.float32)
    red, blue = np.array([[[0.6, 0.1, 0.1]] * 2]), np.array([[[0.1, 0.1, 0.6]] * 2])
    lay(rgb, height, [np.array([[20, 40], [200, 40]])], 8, red, r, pickup=0)
    assert rgb[40, 100, 0] > 0.5 and rgb[40, 100, 2] < 0.2, "paints along its path"
    assert np.allclose(rgb[5, 100], 0.8) and height[5, 100] == 0, "and nowhere else"
    assert height[36:45, 24:30].mean() > height[36:45, 90:110].mean() > height[36:45, 180:196].mean(), \
        "piled where it lands, thinning toward the end"
    assert np.ptp(height[36:45, 100]) > 0.5, "combed into grooves"
    lay(rgb, height, [np.array([[100, 2], [100, 78]])], 6, blue, r, pickup=0)
    assert rgb[40, 100, 2] > 0.5 > rgb[40, 100, 0], "a later stroke lies on top"
    lay(rgb, height, [np.array([[60, 20], [60, 60]])], 6, blue, r, pickup=0.6)
    assert rgb[50, 60, 0] > rgb[26, 60, 0] + 0.03, "it drags the wet red along"
    lay(rgb, height, [np.array([[20, 70], [230, 70]])], 4, blue, r, spent=1)
    assert (rgb[67:74, 30:40, 0] < 0.3).mean() > 0.6 > (rgb[67:74, 205:225, 0] < 0.3).mean(), "and runs dry"
    g = glints(height)
    assert g.max() > 0.3 and (g > 0.05).mean() < 0.1, "glints are small and few"

    def sitting(wet):
        c, h = np.full((80, 240, 3), 0.8, np.float32), np.zeros((80, 240), np.float32)
        lay(c, h, [np.array([[20, 40], [220, 40]])], 10, red, noise.rng(1), pickup=0)
        lay(c, h, [np.array([[60, 40], [210, 40]])], 4, blue, noise.rng(2), pickup=0.6, wet=wet)
        return c[38:43, 150:190, 0].mean()
    assert sitting(np.zeros((80, 240), np.float32)) < sitting(None) - 0.05, "over dry paint it drags nothing along"
    c, h = np.full((80, 240, 3), 0.8, np.float32), np.zeros((80, 240), np.float32)
    h[:, ::4] = 0.6
    lay(c, h, [np.array([[20, 40], [220, 40]])], 12, blue, noise.rng(3), dry=0.7)
    hit = c[33:48, 32:200, 0] < 0.5
    assert 0.1 < hit.mean() < 0.8 and hit[:, ::4].mean() > hit.mean() + 0.2, "a scumble breaks, and catches on the ridges"
    c, h = np.full((40, 120, 3), 0.8, np.float32), np.zeros((40, 120), np.float32)
    lay(c, h, [np.array([[10, 20], [110, 20]])], 8, blue, noise.rng(4), pickup=0, hide=0.5)
    assert abs(c[20, 40:70, 0].mean() - 0.45) < 0.05, "a veil lets half the colour beneath through"
    c, h = np.full((40, 120, 3), 0.8, np.float32), np.tile(np.float32([0.3, 0, 0]), (40, 40))
    lay(c, h, [np.array([[10, 20], [110, 20]])], 8, blue, noise.rng(5), grooves=0, wet=np.zeros((40, 120), np.float32))
    assert np.abs(np.diff(h[17:24, 30:90])).mean() < 0.05 < np.abs(np.diff(h[2:6, 30:90])).mean(), \
        "on dry paint the paste fills the weave"
    print("ok")
