"""Surface repair must enclose triangle interiors, including when budgets expire."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, Sphere, fit_mesh
from assetspheres.geometry import sample_surface, sphere_gaps
from assetspheres.surface_cover import repair_surface_cover


def coarse_triangle():
    return trimesh.Trimesh(
        vertices=[[-1, 0, 0], [1, 0, 0], [0, 1.6, 0]], faces=[[0, 1, 2]], process=False
    )


def assert_grown_only(before, after):
    assert len(before) == len(after)
    assert all(a.center == b.center and a.radius <= b.radius for a, b in zip(before, after))


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
