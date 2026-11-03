from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig
from assetspheres.voxelization import _distance_query, _grid_layout, prepare_distance_grid


def available_backend(backend, device):
    if backend == "open3d":
        pytest.importorskip("open3d")
    else:
        warp = pytest.importorskip("warp")
        if device == "cuda" and not warp.is_cuda_available():
            pytest.skip("No Warp CUDA device")


BACKENDS = [("open3d", "cpu"), ("warp", "cpu"), ("warp", "cuda")]


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_distance_queries_against_analytic_box(backend, device):
    available_backend(backend, device)
    box = trimesh.creation.box(extents=(0.8, 0.6, 0.4))
    query = _distance_query(backend, box.vertices, box.faces, device)
    points = np.random.default_rng(9041).uniform(-1, 1, size=(2000, 3)).astype(np.float32)
    differences = np.abs(points) - np.array([0.4, 0.3, 0.2])
    analytic_distances = np.linalg.norm(np.maximum(differences, 0), axis=1) + np.minimum(
        differences.max(axis=1),
        0,
    )
    distances = query.signed_distances(points)
    assert np.allclose(distances, analytic_distances, atol=2e-6)
    unambiguous = np.abs(analytic_distances) > 2e-6
    assert np.array_equal(distances[unambiguous] < 0, analytic_distances[unambiguous] < 0)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_center_occupancy_and_original_mesh_clearance(backend, device):
    available_backend(backend, device)
    box = trimesh.creation.box(extents=(0.3, 0.1, 0.2))
    result = prepare_distance_grid(box, FitConfig(voxelizer=backend, device=device, resolution=16))
    centers = result.voxel_grid.points
    assert np.all(np.abs(centers) < box.extents / 2)
    reference_clearance = (box.extents / 2 - np.abs(centers)).min(axis=1)
    indices = result.voxel_grid.sparse_indices
    measured_clearance = result.clearance_grid_m[tuple(indices.T)]
    assert np.all(measured_clearance <= reference_clearance + 1e-7)
    assert np.allclose(measured_clearance, reference_clearance, atol=5e-6)
    assert result.backend_details["query_geometry_precision"].startswith("float32")


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_unresolved_thin_component_raises_an_actionable_error(backend, device):
    available_backend(backend, device)
    thin_box = trimesh.creation.box(extents=(0.3, 0.2, 0.0005))
    with pytest.raises(ValueError, match="no interior grid samples.*thinner"):
        prepare_distance_grid(thin_box, FitConfig(voxelizer=backend, device=device, resolution=16))


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_intersecting_shells_use_explicit_union_policy(backend, device):
    available_backend(backend, device)
    first_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second_box = first_box.copy()
    second_box.apply_translation((0.4, 0.0, 0.0))
    triangle_mesh = trimesh.util.concatenate((first_box, second_box))
    configuration = FitConfig(voxelizer=backend, device=device, resolution=16)
    with pytest.raises(ValueError, match="ambiguous"):
        prepare_distance_grid(triangle_mesh, configuration)
    result = prepare_distance_grid(
        triangle_mesh,
        replace(configuration, voxel_solid_policy="union-components"),
    )
    centers = result.voxel_grid.points
    in_first = np.all(np.abs(centers) <= 0.5, axis=1)
    in_second = np.all(np.abs(centers - [0.4, 0.0, 0.0]) <= 0.5, axis=1)
    assert np.all(in_first | in_second)
    assert np.any(in_first & in_second)
    assert result.backend_details["closed_component_count"] == 2


def test_grid_budget_checked_before_loading_optional_library(monkeypatch):
    import assetspheres.voxelization as module

    monkeypatch.setattr(module, "_distance_query", lambda *args: pytest.fail("Should not query"))
    with pytest.raises(ValueError, match="limit"):
        prepare_distance_grid(trimesh.creation.box(), FitConfig(voxelizer="warp", voxel_size=1e-6))


def test_grid_coordinates_are_bounded_and_regular():
    layout = _grid_layout(trimesh.creation.box(extents=(0.3, 0.1, 0.2)), FitConfig(resolution=16))
    points = layout.normalized_points(0, 100)
    assert points.shape == (100, 3)
    assert points.dtype == np.float32
    assert np.all(np.isfinite(points))


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_large_translation_radii_match_the_exported_centers(backend, device):
    available_backend(backend, device)
    triangle_mesh = trimesh.creation.box(extents=(0.2, 0.12, 0.1))
    triangle_mesh.apply_translation((1e12, 1e12, 1e12))
    result = prepare_distance_grid(
        triangle_mesh,
        FitConfig(voxelizer=backend, device=device, resolution=24),
    )
    centers = result.voxel_grid.points
    face_distances = np.minimum(
        centers - triangle_mesh.bounds[0],
        triangle_mesh.bounds[1] - centers,
    ).min(axis=1)
    radii = result.clearance_grid_m[tuple(result.voxel_grid.sparse_indices.T)]
    assert np.all(radii <= face_distances)
    assert result.backend_details["world_coordinate_ulp_m"] > 0


def test_core_import_does_not_load_optional_geometry_runtimes():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import assetspheres.voxelization; "
            "assert 'warp' not in sys.modules; assert 'open3d' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_small_sampled_component_survives_final_candidate_selection(backend, device):
    from assetspheres import fit_mesh
    from assetspheres.fit import sphere_gaps

    available_backend(backend, device)
    large_component = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    small_component = trimesh.creation.box(extents=(0.003, 0.003, 0.003))
    small_component.apply_translation((0.3, 0.0, 0.0))
    triangle_mesh = trimesh.util.concatenate((large_component, small_component))
    spheres, report = fit_mesh(
        triangle_mesh,
        FitConfig(
            voxelizer=backend,
            device=device,
            voxel_solid_policy="union-components",
            voxel_size=0.003,
            count=16,
            samples=512,
        ),
    )
    representatives = np.asarray(report["component_representative_centers_m"])
    assert np.all(sphere_gaps(representatives, spheres) <= 0)
    assert (
        min(np.linalg.norm(np.asarray(sphere.center) - [0.3, 0.0, 0.0]) for sphere in spheres)
        < 0.003
    )
    assert report["component_representative_coverage"] == 1
    assert len(spheres) <= 16


def test_insufficient_component_budget_is_explicit():
    pytest.importorskip("open3d")
    from assetspheres import fit_mesh

    first = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    second = first.copy()
    second.apply_translation((0.4, 0.0, 0.0))
    with pytest.raises(ValueError, match="budget.*component representative"):
        fit_mesh(
            trimesh.util.concatenate((first, second)),
            FitConfig(
                voxelizer="open3d",
                voxel_solid_policy="union-components",
                count=1,
                resolution=24,
                samples=128,
            ),
        )


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_coarse_grid_failure_explains_sampling_rather_than_query_miss(backend, device):
    available_backend(backend, device)
    with pytest.raises(ValueError, match="no interior grid samples"):
        prepare_distance_grid(
            trimesh.creation.box(extents=(0.2, 0.2, 0.2)),
            FitConfig(
                voxelizer=backend,
                device=device,
                voxel_size=1.0,
            ),
        )


def test_extreme_pitch_has_an_actionable_value_error():
    from assetspheres import fit_mesh

    for backend in ("subdivide", "open3d", "warp"):
        with pytest.raises(ValueError, match="Voxel size is too small"):
            fit_mesh(trimesh.creation.box(), FitConfig(voxelizer=backend, voxel_size=1e-320))


def test_refinement_cannot_silently_drop_a_component_representative(monkeypatch):
    pytest.importorskip("open3d")
    import assetspheres.refine as refinement
    from assetspheres import Sphere, fit_mesh

    first = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    second = first.copy()
    second.apply_translation((0.4, 0.0, 0.0))
    monkeypatch.setattr(
        refinement, "refine_spheres", lambda *args: ([Sphere((0.0, 0.0, 0.0), 0.1)], {})
    )
    with pytest.raises(ValueError, match="Optimization lost a component representative"):
        fit_mesh(
            trimesh.util.concatenate((first, second)),
            FitConfig(
                method="refined",
                voxelizer="open3d",
                voxel_solid_policy="union-components",
                count=8,
                resolution=24,
                samples=128,
                refine_samples=64,
                iterations=1,
            ),
        )
