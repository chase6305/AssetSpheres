"""Streaming voxelization must preserve the legacy uniform subdivision lattice."""

import numpy as np
import pytest
import trimesh

from assetspheres import surface_grid
from benchmarks.benchmark_hotpaths import reference_surface_grid


@pytest.mark.parametrize("kind", ["box", "cylinder", "sphere", "torus", "open", "disconnected"])
@pytest.mark.parametrize("translation", [0.0, 1e12])
def test_streamed_surface_and_filled_grid_match_uniform_reference(kind, translation):
    if kind in {"box", "open", "disconnected"}:
        mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.02))
        if kind == "open":
            mesh.update_faces(np.arange(10))
        elif kind == "disconnected":
            second = mesh.copy()
            second.apply_translation([0.3, 0.1, 0.1])
            mesh = trimesh.util.concatenate((mesh, second))
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.04, height=0.3, sections=24)
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(radius=0.1, subdivisions=2)
    else:
        mesh = trimesh.creation.torus(major_radius=0.1, minor_radius=0.025)
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.7, 0.3, -0.2))
    mesh.apply_translation([translation, -translation, translation])
    pitch = float(mesh.extents.max()) / 24
    reference = reference_surface_grid(mesh, pitch)
    streamed = surface_grid.subdivide_surface_grid(mesh, pitch, 2_000_000)
    np.testing.assert_array_equal(streamed.transform, reference.transform)
    np.testing.assert_array_equal(streamed.matrix, reference.matrix)
    np.testing.assert_array_equal(streamed.fill().matrix, reference.fill().matrix)


def test_different_batch_sizes_preserve_lattice(monkeypatch):
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.02))
    expected = reference_surface_grid(mesh, 0.012)
    monkeypatch.setattr(surface_grid, "_TRIANGLE_BATCH_SIZE", 13)
    streamed = surface_grid.subdivide_surface_grid(mesh, 0.012, 2_000_000)
    np.testing.assert_array_equal(streamed.matrix, expected.matrix)


def test_grid_budget_fails_before_triangle_sampling():
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match="max_grid_cells"):
        surface_grid.subdivide_surface_grid(mesh, 1e-5, 1000)


def test_excessive_subdivision_depth_has_an_actionable_error(monkeypatch):
    monkeypatch.setattr(surface_grid, "_MAX_SUBDIVISIONS", 1)
    with pytest.raises(ValueError, match="subdivision rounds.*voxel-size"):
        surface_grid.subdivide_surface_grid(trimesh.creation.box(), 0.01, 2_000_000)


@pytest.mark.parametrize("batch_size", [1, 7, 8192])
def test_mixed_subdivision_depths_preserve_surface_and_fill(monkeypatch, batch_size):
    generator = np.random.default_rng(831)
    # Unequal components exercise completed and subdivided triangles in one batch.
    components = []
    for size in (0.005, 0.05, 0.3):
        mesh = trimesh.creation.box(extents=(size, size * 0.7, size * 0.3))
        mesh.apply_transform(trimesh.transformations.euler_matrix(*generator.uniform(size=3)))
        mesh.apply_translation(generator.uniform(-0.2, 0.2, size=3))
        components.append(mesh)
    mesh = trimesh.util.concatenate(components)
    monkeypatch.setattr(surface_grid, "_TRIANGLE_BATCH_SIZE", batch_size)
    expected = reference_surface_grid(mesh, 0.025)
    actual = surface_grid.subdivide_surface_grid(mesh, 0.025, 2_000_000)
    np.testing.assert_array_equal(actual.transform, expected.transform)
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    np.testing.assert_array_equal(actual.fill().matrix, expected.fill().matrix)


@pytest.mark.parametrize("parent_count", [2047, 2048, 2049])
def test_independent_child_batch_tails_preserve_lattice_and_readonly_source(parent_count):
    # One subdivision yields 4N children, crossing the default 8192-child boundary.
    # Repeated faces make the expected lattice independent of the parent count.
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=np.tile([[0, 1, 2]], (parent_count, 1)),
        process=False,
    )
    vertices, faces, triangles = mesh.vertices.copy(), mesh.faces.copy(), mesh.triangles.copy()
    mesh.triangles.flags.writeable = False
    expected = reference_surface_grid(mesh, 1.5)
    actual = surface_grid.subdivide_surface_grid(mesh, 1.5, 2_000_000)
    np.testing.assert_array_equal(actual.transform, expected.transform)
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    np.testing.assert_array_equal(actual.fill().matrix, expected.fill().matrix)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.triangles, triangles)
    assert not mesh.triangles.flags.writeable
