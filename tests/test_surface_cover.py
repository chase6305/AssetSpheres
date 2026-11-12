"""Surface repair must enclose triangle interiors, including when budgets expire."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, Sphere, fit_mesh
from assetspheres.geometry import sample_surface, sphere_gaps
from assetspheres.surface_cover import (
    VERTICES_PER_BLOCK,
    _normalize_repair_geometry,
    _triangle_requirements,
    repair_surface_cover,
)
from benchmarks.benchmark_hotpaths import reference_triangle_requirements


def coarse_triangle():
    return trimesh.Trimesh(
        vertices=[[-1, 0, 0], [1, 0, 0], [0, 1.6, 0]], faces=[[0, 1, 2]], process=False
    )


def assert_grown_only(before, after):
    assert len(before) == len(after)
    assert all(a.center == b.center and a.radius <= b.radius for a, b in zip(before, after))


def reference_normalize_repair_geometry(vertices, centers, radii):
    origin = vertices[0].astype(np.longdouble)
    relative_vertices = vertices.astype(np.longdouble) - origin
    relative_centers = centers.astype(np.longdouble) - origin
    scale = max(
        np.max(np.abs(relative_vertices)),
        np.max(np.abs(relative_centers)),
        np.max(radii.astype(np.longdouble)),
    )
    return (
        np.asarray(relative_vertices / scale, dtype=np.float64),
        np.asarray(relative_centers / scale, dtype=np.float64),
        np.asarray(radii.astype(np.longdouble) / scale, dtype=np.float64),
        scale,
    )


@pytest.mark.parametrize("count", [1, VERTICES_PER_BLOCK, VERTICES_PER_BLOCK + 1, 65539])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize(
    "scale,translation",
    [(1e-200, 0), (1, 0), (1e100, 0), (0.001, 1e10), (1e307, -1e307)],
)
def test_repair_normalization_preserves_coordinate_bits(count, order, scale, translation):
    random = np.random.default_rng(20261112)
    vertices = random.uniform(-1, 1, size=(count, 3)) * scale + translation
    if order == "strided":
        backing = np.empty((count, 6))
        backing[:, ::2] = vertices
        vertices = backing[:, ::2]
    else:
        vertices = np.array(vertices, order=order)
    centers = np.array([[0, 0, 0], [1.5, -0.1, 0.3]]) * scale + translation
    radii = np.array([0.01, 0.3]) * scale
    original = vertices.copy()
    expected = reference_normalize_repair_geometry(vertices, centers, radii)
    actual = _normalize_repair_geometry(vertices, centers, radii)
    for before, after in zip(expected[:3], actual[:3]):
        np.testing.assert_array_equal(before.view(np.uint64), after.view(np.uint64))
    assert expected[3] == actual[3]
    np.testing.assert_array_equal(vertices, original)


@pytest.mark.parametrize("case", ["signed_zero", "subnormal", "maximum", "centers", "radius"])
def test_large_repair_normalization_preserves_extreme_bits_and_each_scale_source(case):
    pattern = np.array([[0.0, -0.0, 0.0], [-0.0, 0.0, -0.0], [1.0, -1.0, 0.0]])
    centers, radii = np.array([[0.0, -0.0, 0.0]]), np.array([0.5])
    if case == "subnormal":
        pattern *= np.nextafter(0.0, 1.0)
        radii[:] = np.nextafter(0.0, 1.0)
    elif case == "maximum":
        pattern *= np.finfo(float).max
    elif case == "centers":
        centers[0, 0] = 3
    elif case == "radius":
        radii[0] = 5
    vertices = np.tile(pattern, (VERTICES_PER_BLOCK // 3 + 1, 1))
    expected = reference_normalize_repair_geometry(vertices, centers, radii)
    actual = _normalize_repair_geometry(vertices, centers, radii)
    for before, after in zip(expected[:3], actual[:3]):
        np.testing.assert_array_equal(before.view(np.uint64), after.view(np.uint64))
    assert actual[3] == expected[3]
    if case in {"centers", "radius"}:
        assert actual[3] == (3 if case == "centers" else 5)


@pytest.mark.parametrize("translation", [0, 1e12])
def test_large_repair_normalization_keeps_the_complete_output(monkeypatch, translation):
    from assetspheres import surface_cover

    mesh = trimesh.creation.icosphere(subdivisions=6, radius=0.1)
    mesh.vertices += translation
    initial = [Sphere((translation, translation, translation), 0.01)]
    monkeypatch.setattr(surface_cover, "MAXIMUM_TRIANGLE_SPLITS", 0)
    expected = repair_surface_cover(mesh, initial)
    monkeypatch.setattr(
        surface_cover, "_normalize_repair_geometry", reference_normalize_repair_geometry
    )
    assert repair_surface_cover(mesh, initial) == expected


def test_large_repair_keeps_the_unrepresentable_radius_error(monkeypatch):
    from assetspheres import surface_cover

    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("This fixture needs extended precision for the translated coordinate range")
    vertices = np.zeros((VERTICES_PER_BLOCK + 1, 3))
    vertices[:3] = [[-1e308, 0, 0], [1e308, 0, 0], [0, 1e308, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    initial = [Sphere((-1e308, 0, 0), 1)]
    monkeypatch.setattr(surface_cover, "MAXIMUM_TRIANGLE_SPLITS", 0)
    for normalizer in (_normalize_repair_geometry, reference_normalize_repair_geometry):
        monkeypatch.setattr(surface_cover, "_normalize_repair_geometry", normalizer)
        with pytest.raises(ValueError, match="Repaired enclosing radii cannot be represented"):
            repair_surface_cover(mesh, initial)


@pytest.mark.parametrize("triangle_count,center_count", [(3, 4), (257, 24), (2048, 257)])
def test_triangle_distance_reduction_keeps_exact_requirements(triangle_count, center_count):
    random = np.random.default_rng(371021)
    triangles = random.uniform(-1, 1, size=(triangle_count, 3, 3))
    centers = random.uniform(-1, 1, size=(center_count, 3))
    radii = random.uniform(0.01, 1, center_count)
    expected = reference_triangle_requirements(triangles, centers, radii)
    actual = _triangle_requirements(triangles, centers, radii)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)


def test_cross_block_ties_keep_the_first_ball_until_a_strict_improvement():
    triangles = np.zeros((3, 3, 3))
    centers = np.zeros((130, 3))
    radii = np.full(130, 0.1)
    _, labels, _ = _triangle_requirements(triangles, centers, radii)
    assert np.all(labels == 0)
    radii[-1] = 0.2
    _, labels, _ = _triangle_requirements(triangles, centers, radii)
    assert np.all(labels == 129)


def test_vertices_in_different_balls_do_not_certify_the_triangle_interior():
    mesh = coarse_triangle()
    initial = [Sphere(tuple(vertex), 0.1) for vertex in mesh.vertices]
    assert np.all(sphere_gaps(mesh.vertices, initial) < 0)
    assert sphere_gaps(mesh.triangles_center, initial)[0] > 0.5
    repaired, report = repair_surface_cover(mesh, initial)
    assert_grown_only(initial, repaired)
    assert report["surface_repair_triangle_splits"] > 0
    assert np.max(sphere_gaps(sample_surface(mesh, 100_000, 371003), repaired)) <= 0


@pytest.mark.parametrize("split_budget,block_size", [(0, 2048), (3, 2), (19, 5)])
def test_budget_expiry_encloses_coarse_triangles_instead_of_omitting_them(
    monkeypatch, split_budget, block_size
):
    from assetspheres import surface_cover

    monkeypatch.setattr(surface_cover, "MAXIMUM_TRIANGLE_SPLITS", split_budget)
    monkeypatch.setattr(surface_cover, "TRIANGLES_PER_BLOCK", block_size)
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.2))
    initial = [Sphere((0, 0, 0), 0.01), Sphere((0.1, 0, 0), 0.01)]
    repaired, report = repair_surface_cover(mesh, initial)
    assert_grown_only(initial, repaired)
    splits = report["surface_repair_triangle_splits"]
    assert splits <= split_budget
    assert report["surface_repair_visited_triangles"] == len(mesh.faces) + 4 * splits
    assert report["surface_repair_leaf_triangles"] == len(mesh.faces) + 3 * splits
    assert report["surface_repair_coarse_leaves"] > 0
    assert np.max(sphere_gaps(sample_surface(mesh, 20_000, 371009), repaired)) <= 0


def test_depth_expiry_still_encloses_the_original_triangle(monkeypatch):
    from assetspheres import surface_cover

    monkeypatch.setattr(surface_cover, "MAXIMUM_SUBDIVISION_DEPTH", 1)
    mesh = coarse_triangle()
    repaired, report = repair_surface_cover(mesh, [Sphere((0, 0, 0), 0.01)])
    assert report["surface_repair_maximum_depth"] <= 1
    assert report["surface_repair_coarse_leaves"] > 0
    assert np.max(sphere_gaps(sample_surface(mesh, 20_000, 371011), repaired)) <= 0


def test_already_enclosed_mesh_needs_no_changes_or_subdivision():
    mesh = trimesh.creation.box()
    initial = [Sphere((0.1, 0.2, 0.3), 10.0)]
    repaired, report = repair_surface_cover(mesh, initial)
    assert repaired == initial
    assert report["surface_repair_triangle_splits"] == 0
    assert report["surface_repair_grown_spheres"] == 0
    assert report["surface_repair_maximum_radius_increase_m"] == 0


@pytest.mark.parametrize(
    "scale,translation",
    [
        (1e-200, (0, 0, 0)),
        (1e-9, (0, 0, 0)),
        (1e100, (0, 0, 0)),
        (0.001, (1e10, -1e10, 1e10)),
        (1.0, (1e12, -1e12, 1e12)),
    ],
)
def test_scale_and_translation_keep_every_original_point_enclosed(scale, translation):
    mesh = coarse_triangle()
    mesh.vertices = mesh.vertices * scale + np.array(translation)
    initial = [Sphere(tuple(mesh.vertices[0]), 0.01 * scale)]
    repaired, report = repair_surface_cover(mesh, initial)
    assert_grown_only(initial, repaired)
    assert report["surface_repair_roundoff_margin_m"] > 0
    # Extended precision checks points in the original physical triangle.
    barycentric = np.random.default_rng(371019).dirichlet(np.ones(3), size=10_000)
    points = barycentric.astype(np.longdouble) @ mesh.vertices.astype(np.longdouble)
    delta = points - np.array(repaired[0].center, dtype=np.longdouble)
    distances = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
    assert np.max(distances) <= np.longdouble(repaired[0].radius)


def test_fitting_composes_refinement_and_repair_without_adding_padding_twice():
    mesh = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    config = FitConfig(method="refined", count=8, samples=128, refine_samples=128, iterations=20)
    baseline, _ = fit_mesh(mesh, config)
    repaired, report = fit_mesh(mesh, replace(config, method="refined-cover"))
    assert_grown_only(baseline, repaired)
    assert report["triangle_surface_coverage_certified"]
    assert report["surface_coverage"] == 1
    assert report["surface_repair_device"] == "cpu"
    assert "bounded_solid_coverage_conditional" not in report
    padded, _ = fit_mesh(mesh, replace(config, padding=1.0))
    padded_repaired, _ = fit_mesh(mesh, replace(config, method="refined-cover", padding=1.0))
    assert padded_repaired == padded


def test_cuda_refinement_can_finish_with_cpu_triangle_repair():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    config = FitConfig(
        method="refined-cover",
        device="cuda",
        count=8,
        resolution=20,
        samples=128,
        refine_samples=128,
        iterations=20,
        padding=0.003,
    )
    spheres, report = fit_mesh(mesh, config)
    assert report["optimization_device"] == "cuda"
    assert report["surface_repair_device"] == "cpu"
    assert report["surface_coverage"] == 1
    assert np.max(sphere_gaps(sample_surface(mesh, 20_000, 371023), spheres)) <= 0


def test_repair_preserves_mesh_spheres_and_global_rng_and_is_deterministic():
    mesh = coarse_triangle()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    initial = [Sphere((0, 0, 0), 0.01), Sphere((0.5, 0.2, 0), 0.01)]
    before = np.random.get_state()
    repaired, report = repair_surface_cover(mesh, initial)
    repeated, repeated_report = repair_surface_cover(mesh, initial)
    after = np.random.get_state()
    assert repaired == repeated and report == repeated_report
    assert initial == [Sphere((0, 0, 0), 0.01), Sphere((0.5, 0.2, 0), 0.01)]
    np.testing.assert_array_equal(vertices, mesh.vertices)
    np.testing.assert_array_equal(faces, mesh.faces)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_open_mesh_fallback_is_explicit():
    mesh = coarse_triangle()
    config = FitConfig(method="refined-cover", count=4, samples=128)
    with pytest.raises(ValueError, match="watertight"):
        fit_mesh(mesh, config)
    _, report = fit_mesh(mesh, replace(config, non_watertight="cover"))
    assert report["requested_method"] == "refined-cover"
    assert report["effective_method"] == "cover"


@pytest.mark.parametrize(
    "mesh,spheres",
    [
        (trimesh.Trimesh(), [Sphere((0, 0, 0), 1)]),
        (coarse_triangle(), []),
        (
            trimesh.Trimesh(
                vertices=[[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
            ),
            [Sphere((0, 0, 0), 1)],
        ),
    ],
)
def test_reject_invalid_inputs(mesh, spheres):
    with pytest.raises(ValueError):
        repair_surface_cover(mesh, spheres)
