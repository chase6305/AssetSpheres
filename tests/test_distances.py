"""Numerical parity for CPU distances across point/sphere block boundaries."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

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


def reference_blocked_gaps(points, spheres):
    """Allocation-per-block implementation, independent of workspace reuse."""
    centers = np.asarray([sphere.center for sphere in spheres])
    radii = np.asarray([sphere.radius for sphere in spheres])
    gaps = np.full(len(points), np.inf)
    for first_point in range(0, len(points), 2048):
        for first_sphere in range(0, len(spheres), 128):
            distances = cdist(
                points[first_point : first_point + 2048],
                centers[first_sphere : first_sphere + 128],
            )
            distances -= radii[None, first_sphere : first_sphere + 128]
            gaps[first_point : first_point + 2048] = np.minimum(
                gaps[first_point : first_point + 2048], distances.min(axis=1)
            )
    return gaps


@pytest.mark.parametrize(
    "point_count,sphere_count",
    [(0, 1), (1, 1), (2047, 127), (2048, 128), (2049, 129), (4097, 257)],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_gap_workspace_preserves_bits_at_both_block_boundaries(point_count, sphere_count, layout):
    random_generator = np.random.default_rng(99141)
    points = random_generator.normal(size=(point_count * 2, 3))
    points = points[::2] if layout == "strided" else np.array(points[:point_count], order=layout)
    # Duplicate spheres exercise ties across the sphere block boundary.
    centers = random_generator.normal(size=(sphere_count, 3))
    if sphere_count > 128:
        centers[128] = centers[0]
    spheres = [Sphere(tuple(center), 0.25) for center in centers]
    before = points.copy()
    expected = reference_blocked_gaps(points, spheres)
    actual = sphere_gaps(points, spheres)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(points, before)


def test_gap_workspace_is_reused_for_full_and_short_blocks(monkeypatch):
    from assetspheres import geometry

    calls = []

    def track_workspace(points, centers, **kwargs):
        output = kwargs.get("out")
        if calls:
            assert output is not None and output.flags.c_contiguous
        else:
            assert output is None
        result = cdist(points, centers, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(geometry, "cdist", track_workspace)
    points = np.zeros((4097, 3))
    spheres = [Sphere((0, 0, 0), 1)] * 257
    np.testing.assert_array_equal(sphere_gaps(points, spheres), -np.ones(len(points)))
    assert len(calls) == 9
    assert all(np.shares_memory(calls[0], output) for output in calls)
    assert calls[-1].shape == (1, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.longdouble, object])
def test_gap_workspace_keeps_scipy_dtype_support_and_rounding(dtype):
    points = np.random.default_rng(99143).normal(size=(2049, 3)).astype(dtype)
    spheres = [Sphere((0, 0, 0), 0.25)] * 129
    try:
        expected = reference_blocked_gaps(points, spheres)
    except (TypeError, ValueError) as error:
        # Some SciPy versions reject numeric object arrays; retain that behavior.
        with pytest.raises(type(error)) as caught:
            sphere_gaps(points, spheres)
        assert str(caught.value) == str(error)
        return
    actual = sphere_gaps(points, spheres)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_gap_workspace_retains_extended_precision_near_the_sphere_boundary():
    if np.finfo(np.longdouble).eps == np.finfo(np.float64).eps:
        pytest.skip("This platform does not provide extended precision")
    points = np.zeros((2049, 3), dtype=np.longdouble)
    points[[0, -1], 0] = np.longdouble(1) + np.finfo(np.longdouble).eps * 4
    spheres = [Sphere((0, 0, 0), 1)] * 129
    expected = reference_blocked_gaps(points, spheres)
    actual = sphere_gaps(points, spheres)
    assert expected[0] > 0 and expected[-1] > 0
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_gap_workspace_preserves_array_like_inputs_with_different_dtypes_in_each_block():
    points = [[0.0, 0.0, 0.0]] * 2048 + [[np.longdouble(2), np.longdouble(0), np.longdouble(0)]]
    spheres = [Sphere((0, 0, 0), 1)]
    expected = reference_blocked_gaps(points, spheres)
    actual = sphere_gaps(points, spheres)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
