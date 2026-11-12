"""Unit-grid distances with SciPy's exact nearest-background transform."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_DISTANCE_BLOCK_CELLS = 32768


def unit_grid_distances(grid: np.ndarray) -> np.ndarray:
    """Return float64 distances to zero cells, with unit spacing on every axis.

    Reuse a row block while accumulating squared coordinate differences.
    This avoids a full floating-point coordinate array for every axis, while
    keeping SciPy's nearest-background indices and the same distance formula.
    Indices and output still grow linearly with the grid size. The offset buffer
    holds at most one first-axis row or 32768 cells, whichever is larger; each
    axis also needs a coordinate vector proportional to its length.
    """
    nearest = ndimage.distance_transform_edt(grid, return_distances=False, return_indices=True)
    distances = np.zeros(grid.shape, dtype=np.float64)
    if not grid.ndim:
        return distances
    rows_per_block = max(1, _DISTANCE_BLOCK_CELLS // max(1, math.prod(grid.shape[1:])))
    offsets = np.empty((min(rows_per_block, grid.shape[0]), *grid.shape[1:]), dtype=np.float64)
    for axis, width in enumerate(grid.shape):
        shape = [1] * grid.ndim
        shape[axis] = width
        coordinates = np.arange(width, dtype=np.float64).reshape(shape)
        if offsets.shape == distances.shape:
            np.subtract(nearest[axis], coordinates, out=offsets)
            np.square(offsets, out=offsets)
            distances += offsets
            continue
        for first_row in range(0, grid.shape[0], rows_per_block):
            block = slice(first_row, first_row + rows_per_block)
            block_offsets = offsets[: len(distances[block])]
            block_coordinates = coordinates[block] if axis == 0 else coordinates
            np.subtract(nearest[axis, block], block_coordinates, out=block_offsets)
            np.square(block_offsets, out=block_offsets)
            distances[block] += block_offsets
    np.sqrt(distances, out=distances)
    return distances
