"""Unit-grid distances with SciPy's exact nearest-background transform."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def unit_grid_distances(grid: np.ndarray) -> np.ndarray:
    """Return float64 distances to zero cells, with unit spacing on every axis.

    Reuse one scalar grid while accumulating squared coordinate differences.
    This avoids a full floating-point coordinate array for every axis, while
    keeping SciPy's nearest-background indices and the same distance formula.
    Indices, output and scratch space still grow linearly with the grid size.
    """
    nearest = ndimage.distance_transform_edt(grid, return_distances=False, return_indices=True)
    distances = np.zeros(grid.shape, dtype=np.float64)
    offsets = np.empty_like(distances)
    for axis, width in enumerate(grid.shape):
        shape = [1] * grid.ndim
        shape[axis] = width
        coordinates = np.arange(width, dtype=np.float64).reshape(shape)
        np.subtract(nearest[axis], coordinates, out=offsets)
        np.square(offsets, out=offsets)
        distances += offsets
    np.sqrt(distances, out=distances)
    return distances
