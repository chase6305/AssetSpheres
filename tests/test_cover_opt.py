"""Regroup complete triangles without trading surface enclosure for a smaller proxy."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.geometry import sample_surface, sphere_gaps
from assetspheres.surface_fit import _nearest_triangle_centers, _partition_cover_triangles


@pytest.mark.parametrize("count,padding", [(1, 0), (8, 0), (24, 0.003)])
@pytest.mark.parametrize("open_mesh", [False, True])
def test_complete_triangles_enclosed_without_increasing_volume_sum(count, padding, open_mesh):
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=24)
    if open_mesh:
        mesh.update_faces(mesh.face_normals[:, 2] == 0)
    original_vertices, original_faces = mesh.vertices.copy(), mesh.faces.copy()
    config = FitConfig(method="cover-opt", count=count, padding=padding, samples=128)
    baseline, _ = fit_mesh(mesh, replace(config, method="cover-tight"))
    spheres, report = fit_mesh(mesh, config)
    assert 0 < len(spheres) <= len(baseline) <= count
    assert sum(s.radius**3 for s in spheres) <= sum(s.radius**3 for s in baseline) * (1 + 1e-14)
    assert report["surface_coverage"] == 1
    assert report["triangle_surface_coverage_certified"] is True
    assert "bounded_solid_coverage_conditional" not in report

    # All three vertices of every subtriangle must share one enclosing ball.
    triangles, _, _ = _partition_cover_triangles(mesh, config)
    enclosed = np.zeros(len(triangles), dtype=bool)
    for sphere in spheres:
        delta = triangles.astype(np.longdouble) - np.array(sphere.center, dtype=np.longdouble)
        distances = np.hypot(np.hypot(delta[:, :, 0], delta[:, :, 1]), delta[:, :, 2])
        enclosed |= distances.max(axis=1) <= np.longdouble(sphere.radius)
    assert enclosed.all()
    assert np.all(sphere_gaps(sample_surface(mesh, 20_000, 65117), spheres) <= 1e-12)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)


def test_regrouping_improves_a_box_and_is_deterministic_without_global_rng_changes():
    mesh = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    config = FitConfig(method="cover-opt", count=24, padding=0.003, samples=128)
    baseline, _ = fit_mesh(mesh, replace(config, method="cover-tight"))
    before = np.random.get_state()
    spheres, report = fit_mesh(mesh, config)
    repeated, repeated_report = fit_mesh(mesh, config)
    after = np.random.get_state()
    assert spheres == repeated
    assert report == repeated_report
    assert sum(s.radius**3 for s in spheres) < sum(s.radius**3 for s in baseline) * 0.97
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_worse_regrouping_keeps_the_original_cover(monkeypatch):
    from assetspheres import surface_fit

    first = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    second = first.copy()
    second.apply_translation((1, 0, 0))
    mesh = trimesh.util.concatenate((first, second))
    config = FitConfig(method="cover-opt", count=8, samples=128)
    baseline, _ = fit_mesh(mesh, replace(config, method="cover-tight"))
    monkeypatch.setattr(
        surface_fit,
        "_nearest_triangle_centers",
        lambda triangles, centers: np.zeros(len(triangles), dtype=np.intp),
    )
    spheres, _ = fit_mesh(mesh, config)
    assert spheres == baseline


def test_assignment_uses_farthest_vertex_and_keeps_first_tie_across_blocks():
    # Centroid-nearest picks center 0; the whole triangle fits better at center 1.
    triangle = np.array([[0.0, 0, 0], [0, 0.01, 0], [10, 0, 0]])
    centers = np.full((130, 3), 100.0)
    centers[0] = [10 / 3, 0, 0]
    centers[1] = centers[129] = [5, 0, 0]
    labels = _nearest_triangle_centers(np.tile(triangle, (2051, 1, 1)), centers)
    assert np.all(labels == 1)
    centers[129] = [5, 0.005, 0]
    assert np.all(_nearest_triangle_centers(np.tile(triangle, (2051, 1, 1)), centers) == 129)


@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_scale_and_translation_preserve_enclosure(scale):
    mesh = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    mesh.apply_scale(scale)
    mesh.apply_translation(np.array([2, -3, 5]) * scale)
    spheres, report = fit_mesh(mesh, FitConfig(method="cover-opt", count=8, samples=128))
    assert report["surface_coverage"] == 1
    assert np.all(sphere_gaps(sample_surface(mesh, 10_000, 43111), spheres) <= 1e-12 * scale)


def test_cuda_request_is_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        FitConfig(method="cover-opt", device="cuda")
