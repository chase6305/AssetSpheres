"""Reject unrepresentable subdivision before allocating intermediate meshes."""

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, fit_mesh


@pytest.mark.parametrize(
    "method", ["cover", "cover-tight", "cover-opt", "voxel", "refined", "refined-cover"]
)
def test_subdivision_precision_fails_before_mesh_expansion(monkeypatch, method):
    mesh = trimesh.creation.box(extents=(0.003, 0.001, 0.0005))
    mesh.apply_translation((1e12, -1e12, 1e12))

    def forbidden(*args, **kwargs):
        pytest.fail("Unrepresentable subdivision must fail before expanding triangles")

    monkeypatch.setattr(trimesh.remesh, "subdivide_to_size", forbidden)
    monkeypatch.setattr(trimesh.Trimesh, "voxelized", forbidden)
    with pytest.raises(ValueError, match="precision.*recenter"):
        fit_mesh(mesh, FitConfig(method=method, resolution=48, samples=32))


def test_slab_remains_available_for_representable_coarse_geometry():
    mesh = trimesh.creation.box(extents=(0.003, 0.001, 0.0005))
    mesh.apply_translation((1e12, -1e12, 1e12))
    spheres, report = fit_mesh(mesh, FitConfig(method="slab", count=4, samples=32))
    assert len(spheres) <= 4
    assert np.all(np.isfinite([s.radius for s in spheres]))
    assert report["surface_coverage"] == 1
