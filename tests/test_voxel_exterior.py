"""Bounded point rasterization preserves the exterior grid and RNG consumption."""

import numpy as np
import pytest

from assetspheres.refine import _sample_voxel_exterior
from benchmarks.benchmark_hotpaths import reference_voxel_exterior


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("count", [17, 32768, 32769, 65553])
@pytest.mark.parametrize(
    "pitch,translation",
    [
        (0.003, (0.1, -0.2, 0.3)),
        (1.0, (0, 0, 0)),
        (0.003, (1e10, -1e10, 1e10)),
    ],
)
def test_rasterization_preserves_samples_and_rng_state(layout, count, pitch, translation):
    random_generator = np.random.default_rng(99103)
    lattice = random_generator.integers(0, 12, size=(count, 3)).astype(float)
    lattice[: min(count, 6)] += 0.5
    points = np.array(lattice * pitch + translation, order=layout)
    expected_random = np.random.default_rng(99107)
    actual_random = np.random.default_rng(99107)
    expected = reference_voxel_exterior(points, pitch, 128, expected_random)
    actual = _sample_voxel_exterior(points, pitch, 128, actual_random)
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))
    assert expected_random.bit_generator.state == actual_random.bit_generator.state
