import sys
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, Sphere, fit_mesh
from assetspheres.fit import sample_surface, sphere_gaps


@pytest.mark.parametrize("method", ["voxel", "surface", "cover"])
def test_budget_determinism_and_global_rng(method):
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.15))
    cfg = FitConfig(method=method, count=8, resolution=20, samples=512)
    np.random.seed(123)
    before = np.random.get_state()
    spheres, report = fit_mesh(mesh, cfg)
    after = np.random.get_state()
    repeated, _ = fit_mesh(mesh, cfg)
    assert spheres == repeated
    assert 0 < len(spheres) <= cfg.count
    assert all(np.isfinite(s.radius) and s.radius > 0 for s in spheres)
    assert 0 <= report["surface_coverage"] <= 1
    assert np.array_equal(before[1], after[1])
    if method == "voxel":
        assert 0 < report["voxel_center_coverage"] <= 1


def test_cover_encloses_triangle_interiors_not_only_vertices():
    # This coarse, long, thin triangle is open and has large unsampled regions.
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 0.1, 0]], faces=[[0, 1, 2]])
    spheres, report = fit_mesh(mesh, FitConfig(method="cover", count=10, samples=128))
    dense = sample_surface(mesh, 20_000, 101)
    assert np.max(sphere_gaps(dense, spheres)) <= 1e-12
    assert report["triangle_surface_coverage_certified"] is True
    assert report["surface_coverage"] == 1


def test_padding_is_in_metres_and_surface_radius_is_fixed():
    mesh = trimesh.creation.box()
    cfg = FitConfig(method="surface", count=5, surface_radius=0.03, samples=128)
    base, _ = fit_mesh(mesh, cfg)
    padded, _ = fit_mesh(mesh, FitConfig(**{**cfg.__dict__, "padding": 0.004}))
    assert [s.center for s in base] == [s.center for s in padded]
    assert all(s.radius == pytest.approx(0.034) for s in padded)


def test_visuals_and_vertex_attributes_do_not_affect_fitting_or_change_on_input():
    plain = trimesh.creation.box(extents=(0.3, 0.1, 0.15))
    decorated = plain.copy()
    colors = np.arange(len(plain.vertices) * 4, dtype=np.uint8).reshape(-1, 4)
    features = np.arange(len(plain.vertices) * 16).reshape(-1, 16)
    decorated.visual.vertex_colors = colors.copy()
    decorated.vertex_attributes["features"] = features.copy()
    decorated.metadata["label"] = "source mesh"
    configuration = FitConfig(method="cover", count=8, samples=128)

    assert fit_mesh(decorated, configuration) == fit_mesh(plain, configuration)
    np.testing.assert_array_equal(decorated.visual.vertex_colors, colors)
    np.testing.assert_array_equal(decorated.vertex_attributes["features"], features)
    assert decorated.metadata["label"] == "source mesh"


def test_open_mesh_requires_explicit_fallback():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="watertight"):
        fit_mesh(mesh)
    _, report = fit_mesh(mesh, FitConfig(non_watertight="cover", samples=128))
    assert report["requested_method"] == "voxel"
    assert report["effective_method"] == "cover"


def test_memory_limit_checked_before_voxelizing(monkeypatch):
    import assetspheres.voxel_fit as module

    mesh = trimesh.creation.box()

    def forbidden(*args, **kwargs):
        pytest.fail("Voxelization should not run after a grid-budget failure")

    monkeypatch.setattr(module, "subdivide_surface_grid", forbidden)
    with pytest.raises(ValueError, match="limit"):
        fit_mesh(mesh, FitConfig(voxel_size=1e-6))


def test_disconnected_mesh_does_not_get_replaced_by_convex_hull():
    first = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    second = first.copy()
    second.apply_translation((1, 0, 0))
    spheres, _ = fit_mesh(
        trimesh.util.concatenate([first, second]), FitConfig(count=12, resolution=48, samples=1024)
    )
    assert any(s.center[0] < 0.1 for s in spheres)
    assert any(s.center[0] > 0.9 for s in spheres)
    assert sphere_gaps(np.array([[0.5, 0, 0]]), spheres)[0] > 0


def test_lula_adapter_uses_documented_api_and_array_types(monkeypatch):
    def create(vertices, triangles):
        assert vertices[0].shape == (3, 1) and vertices[0].dtype == np.float64
        assert triangles[0].shape == (3, 1) and triangles[0].dtype == np.int32

        def generate(count, offset):
            assert count == 3 and offset == 0.02
            return [SimpleNamespace(center=np.array([[0], [0], [0]]), radius=0.52)]

        return SimpleNamespace(generate_spheres=generate)

    monkeypatch.setitem(
        sys.modules, "lula", SimpleNamespace(create_collision_sphere_generator=create)
    )
    spheres, _ = fit_mesh(
        trimesh.creation.box(), FitConfig(method="lula", count=3, padding=0.02, samples=128)
    )
    assert spheres == [Sphere((0.0, 0.0, 0.0), 0.52)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": 1.5},
        {"voxel_size": 0},
        {"padding": -1},
        {"surface_radius": float("nan")},
        {"samples": 1},
        {"seed": -1},
        {"resolution": 3},
        {"iterations": 0},
        {"refine_samples": 1},
        {"outside_weight": float("nan")},
        {"refine_reference": "unknown"},
        {"device": "cuda", "method": "surface"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)
