"""Ink on paper after the brush has left: water creeping along fibres, drying at its edge."""

import numpy as np
from scipy import ndimage

from . import noise


def bleed(ink, water, sheet, wet=0.5, spread=5.0, halo=0.3, edge=0.5, soften=1.0, mottle=0.06,
          whiskers=0.8, seed=0):
    """Let wet ink settle into the sheet.

    water   where the brush was still wet, from brush.stroke; dry-brush marks carry none
    wet     how much water a wet brush carries (0 = none, the marks are left as they are)
    spread  px the water front can travel into dry paper
    halo    tone of the pale bleed zone beyond the stroke, relative to the stroke
    edge    pigment piled up where the water front stopped
    soften  px of blur the wet ink applies to its own bristle texture
    mottle  how unevenly the sheet drinks: fibre-rich spots take a little more ink
    whiskers  strength of the hairline bleeds that fibres draw out of a wet edge
    """
    if wet <= 0:
        return ink
    r = noise.rng(seed)
    water = np.clip(water, 0, 1.2) * wet
    cond = 0.7 + 0.8 * sheet.fiber + 0.15 * noise.field(ink.shape, 2.0, r)
    reach = ndimage.gaussian_filter(water, spread) * cond
    front = noise.smoothstep(0.05, 0.07, reach)
    damp = np.clip(ndimage.gaussian_filter(water, 2) / (wet + 1e-6), 0, 1)
    core = ink + (ndimage.gaussian_filter(ink, soften) - ink) * damp
    core = core * (1 - mottle / 2 + mottle * ndimage.gaussian_filter(sheet.fiber, 2.5) * 1.5)
    # the water runs ahead of the pigment and dries as a pale, flat, sharp-edged margin
    pale = np.minimum(ndimage.gaussian_filter(ink, spread), 0.6) * halo * front
    rim = np.clip(front - ndimage.gaussian_filter(front, 1.6), 0, None)
    rim_ink = rim * edge * ndimage.gaussian_filter(ink, spread) * 2.0
    # fibres that touch the wet edge wick a hair of ink out into the dry sheet
    wick = ndimage.gaussian_filter(water, 3.0) * sheet.fiber ** 2
    hairs = noise.smoothstep(0.02, 0.08, wick) * np.minimum(ndimage.gaussian_filter(ink, 4.0), 0.5) * whiskers
    return np.maximum(np.maximum(core, pale), hairs) + rim_ink
