"""Numerical parity for CPU distances across point/sphere block boundaries."""

import numpy as np
import pytest

from assetspheres import Sphere
from assetspheres.fit import sphere_gaps


@pytest.mark.parametrize("offset", [0.0, 1e12])
def test_gaps_match_dense_reference_for_noncontiguous_inputs(offset):
    rng = np.random.default_rng(99131)
    points = (rng.normal(size=(8200, 3)) + offset)[::2]
    spheres = [
        Sphere(tuple(c + offset), float(r))
        for c, r in zip(rng.normal(size=(139, 3)), rng.uniform(0.001, 1.0, 139))
    ]
    centers = np.asarray([sphere.center for sphere in spheres])
    radii = np.asarray([sphere.radius for sphere in spheres])
    expected = (np.linalg.norm(points[:, None] - centers[None], axis=2) - radii).min(axis=1)
    np.testing.assert_array_equal(sphere_gaps(points, spheres), expected)


def test_boundary_and_empty_distance_inputs():
    spheres = [Sphere((0.0, 0.0, 0.0), 1.0)]
    points = np.array([[1.0, 0, 0], [1 + 1e-8, 0, 0], [1 - 1e-8, 0, 0], [0, 0, 0]])
    gaps = sphere_gaps(points, spheres)
    assert gaps[0] == 0 and gaps[1] > 0 and gaps[2] < 0 and gaps[3] == -1
    assert sphere_gaps(np.empty((0, 3)), spheres).shape == (0,)
    assert np.isinf(sphere_gaps(points, [])).all()
