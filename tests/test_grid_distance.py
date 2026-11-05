"""Unit-grid reconstruction preserves SciPy distances on varied lattices."""

import numpy as np
import pytest
from scipy import ndimage

from assetspheres.grid_distance import unit_grid_distances


@pytest.mark.parametrize("shape", [(23,), (7, 13), (1, 4, 7), (8, 9, 7), (33, 31, 37)])
@pytest.mark.parametrize("kind", ["zero", "one", "random"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_distances_match_scipy_for_uniform_random_and_singleton_grids(shape, kind, order):
    generator = np.random.default_rng(591083)
    if kind == "random":
        grid = generator.uniform(size=shape) > 0.3
    else:
        grid = np.full(shape, kind == "one")
    grid = np.array(grid, order=order)[::-1]
    original = grid.copy()
    expected = ndimage.distance_transform_edt(grid)
    actual = unit_grid_distances(grid)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(grid, original)


def test_single_background_cell_has_analytic_euclidean_distances():
    grid = np.ones((4, 5, 6), dtype=bool)
    grid[1, 2, 3] = False
    x, y, z = np.indices(grid.shape)
    expected = np.sqrt((x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2)
    np.testing.assert_array_equal(unit_grid_distances(grid), expected)
