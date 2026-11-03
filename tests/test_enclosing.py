import numpy as np
import pytest

from assetspheres.enclosing import tighten_ball


def distances(points, center):
    delta = np.asarray(points, dtype=np.longdouble) - np.asarray(center, dtype=np.longdouble)
    return np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])


def assert_certified_improvement(points, center):
    initial_radius = np.nextafter(float(distances(points, center).max()), np.inf)
    fitted_center, radius, diagnostics = tighten_ball(points, center)
    assert np.all(np.isfinite(fitted_center))
    assert np.isfinite(radius)
    assert distances(points, fitted_center).max() <= np.longdouble(radius)
    assert radius <= initial_radius
    assert diagnostics["all_points_enclosed"]
    assert diagnostics["radius_nonincreasing"]
    return fitted_center, radius, diagnostics


def test_box_corners_find_center_and_radius():
    points = np.array([[x, y, z] for x in (-1, 1) for y in (-2, 2) for z in (-3, 3)])
    center, radius, diagnostic = assert_certified_improvement(points, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(center, 0, atol=1e-8)
    assert radius == pytest.approx(np.sqrt(14), rel=1e-8)
    assert diagnostic["improved"]


@pytest.mark.parametrize("seed", [0, 4, 19])
def test_random_cloud_all_points_are_enclosed(seed):
    points = np.random.default_rng(seed).normal(size=(1000, 3)) * [1.0, 0.4, 0.2]
    _, _, diagnostic = assert_certified_improvement(points, points.mean(axis=0) + [2.0, 1.0, -1.0])
    assert diagnostic["optimizer_iterations"] <= 6 * 64
    assert diagnostic["support_points"] <= 64


def test_collinear_points_and_duplicates():
    line = np.linspace(-3.0, 7.0, 1000)
    points = np.column_stack((line, np.zeros_like(line), np.zeros_like(line)))
    points = np.repeat(points, 3, axis=0)
    center, radius, _ = assert_certified_improvement(points, np.array([0.0, 4.0, 2.0]))
    assert np.allclose(center, [2.0, 0.0, 0.0], atol=1e-8)
    assert radius == pytest.approx(5.0, rel=1e-8)


def test_coplanar_circle():
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles), np.full_like(angles, 0.5)))
    center, radius, _ = assert_certified_improvement(points, np.array([1.0, 1.0, 2.0]))
    assert np.allclose(center, [0.0, 0.0, 0.5], atol=1e-7)
    assert radius == pytest.approx(1.0, rel=1e-7)


@pytest.mark.parametrize(
    "point, original", [([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])]
)
def test_single_point_zero_radius(point, original):
    center, radius, diagnostics = tighten_ball(np.array([point]), np.array(original))
    assert np.array_equal(center, point)
    assert radius == 0
    assert diagnostics["optimizer_status"] == "coincident_points_exact"


@pytest.mark.parametrize("scale", [1e-200, 1e-20, 1.0, 1e100])
def test_scale_normalization(scale):
    points = scale * np.array(
        [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    center, radius, _ = assert_certified_improvement(points, scale * np.array([2.0, 3.0, 0.0]))
    assert np.allclose(center / scale, 0, atol=1e-8)
    assert radius / scale == pytest.approx(np.sqrt(2), rel=1e-8)


def test_large_translation_with_small_geometry():
    translation = np.array([1e12, -1e12, 1e12])
    points = translation + 0.125 * np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    center, radius, _ = assert_certified_improvement(points, translation + [2.0, 1.0, 0.0])
    assert np.array_equal(center, translation)
    assert radius == pytest.approx(0.125)


def test_optimizer_failure_keeps_original_center_and_full_radius(monkeypatch):
    import assetspheres.enclosing as module

    def fail(*args, **kwargs):
        raise RuntimeError("simulated optimizer failure")

    monkeypatch.setattr(module, "minimize", fail)
    points = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 1.0], [-1.0, 1.0, 0.0]])
    original = np.array([3.0, 0.0, 0.0])
    center, _, diagnostic = assert_certified_improvement(points, original)
    assert np.array_equal(center, original)
    assert diagnostic["optimizer_success"] is False


@pytest.mark.parametrize(
    "points, center",
    [
        (np.empty((0, 3)), np.zeros(3)),
        (np.zeros((4, 2)), np.zeros(3)),
        (np.zeros((4, 3)), np.zeros(2)),
        (np.array([[np.nan, 0, 0]]), np.zeros(3)),
        (np.zeros((4, 3)), np.array([np.inf, 0, 0])),
    ],
)
def test_reject_invalid_inputs(points, center):
    with pytest.raises(ValueError):
        tighten_ball(points, center)
