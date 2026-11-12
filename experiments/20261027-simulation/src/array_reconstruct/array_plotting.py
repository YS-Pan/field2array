"""Dense staggered element-position tiles; tiles do not depict antenna outlines."""
from __future__ import annotations

import numpy as np


ELEMENT_TILE_SIZE_MM = 11.0
GEOMETRY_CAPTION = "dx=11 mm | dy=11 mm | alternate-row x offset=5.5 mm"
SYMBOL_CAPTION = "Dense 11 x 11 mm tiles centered on saved element positions; not physical antenna outlines."


def draw_element_values(ax, x_mm, y_mm, values, *, cmap, vmin, vmax,
                        mask=None, compact=False):
    """Draw one tightly tiled 11 mm square per saved position in unit order.

    Invalid/masked values retain a light gray tile. No interpolation,
    resampling, triangulation, or inferred element geometry is performed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    x, y = np.asarray(x_mm, float), np.asarray(y_mm, float)
    values = np.asarray(values, float)
    if x.ndim != 1 or y.shape != x.shape or values.shape != x.shape:
        raise ValueError("Coordinates and colors must have matching unit axes")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Element coordinates must be finite")
    if len(np.unique(np.column_stack((x, y)), axis=0)) != len(x):
        raise ValueError("Each unit must have one unique saved position")
    valid = np.isfinite(values)
    if mask is not None:
        supplied_mask = np.asarray(mask, bool)
        if supplied_mask.shape != values.shape:
            raise ValueError("Mask must retain the supplied unit axis")
        valid &= supplied_mask
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#e3e6e9")
    half = ELEMENT_TILE_SIZE_MM / 2
    patches = [Rectangle((xi-half, yi-half), ELEMENT_TILE_SIZE_MM, ELEMENT_TILE_SIZE_MM)
               for xi, yi in zip(x, y)]
    collection = PatchCollection(patches, cmap=color_map, edgecolors="none",
                                 linewidths=0, antialiaseds=False, snap=False)
    collection.set_clim(vmin, vmax)
    collection.set_array(np.ma.array(values, mask=~valid))
    ax.add_collection(collection)
    ax.set_xlim(float(x.min())-half, float(x.max())+half)
    ax.set_ylim(float(y.min())-half, float(y.max())+half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_xlabel("x (mm)", fontsize=6 if compact else 9)
    ax.set_ylabel("y (mm)", fontsize=6 if compact else 9)
    if compact:
        ax.set_xticks([0, 170, 340])
        ax.set_yticks([0, 170, 340])
        ax.tick_params(labelsize=5, pad=1, length=2)
    else:
        ax.set_xticks([0, 55, 110, 165, 220, 275, 330])
        ax.set_yticks([0, 55, 110, 165, 220, 275, 330])
        ax.tick_params(labelsize=8)
    return collection
