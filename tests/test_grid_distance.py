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


@pytest.mark.parametrize("shape", [(), (0,), (0, 5), (3, 0, 4)])
def test_scalar_and_empty_grids_preserve_output_shape(shape):
    actual = unit_grid_distances(np.ones(shape, dtype=bool))
    np.testing.assert_array_equal(actual, np.zeros(shape, dtype=np.float64))


@pytest.mark.parametrize("shape", [(32769,), (7, 8192), (3, 113, 113), (1, 191, 193)])
@pytest.mark.parametrize("kind", ["zero", "one", "random"])
def test_blocked_distance_reconstruction_matches_scipy_at_row_and_tail_boundaries(shape, kind):
    generator = np.random.default_rng(8123)
    grid = generator.random(shape) > 0.3 if kind == "random" else np.full(shape, kind == "one")
    grid = np.asfortranarray(grid)[::-1]
    expected = ndimage.distance_transform_edt(grid)
    actual = unit_grid_distances(grid)
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_blocked_grid_distances_preserve_complete_fitting_and_refinement(monkeypatch):
    import trimesh
    from scipy import ndimage

    from assetspheres import FitConfig, fit_mesh, refine, voxel_fit

    mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    configuration = FitConfig(
        method="refined-cover",
        resolution=40,
        count=8,
        samples=512,
        refine_samples=128,
        iterations=10,
    )
    actual = fit_mesh(mesh, configuration)
    assert actual[1]["voxel_count"] > 32768
    monkeypatch.setattr(voxel_fit, "unit_grid_distances", ndimage.distance_transform_edt)
    monkeypatch.setattr(refine, "unit_grid_distances", ndimage.distance_transform_edt)
    assert fit_mesh(mesh, configuration) == actual
