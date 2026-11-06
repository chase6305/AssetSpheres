import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.geometry import _has_closed_triangle_edges


class _CustomWatertight(trimesh.Trimesh):
    @property
    def is_watertight(self):
        return True


@pytest.mark.parametrize("kind", ["box", "open", "reversed", "doubled", "disconnected"])
def test_closed_edge_criterion_matches_trimesh_and_preserves_geometry(kind):
    mesh = trimesh.creation.box()
    if kind == "open":
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
    elif kind == "reversed":
        mesh.faces[0] = mesh.faces[0, ::-1]
        assert not mesh.is_winding_consistent
    elif kind == "doubled":
        mesh.faces = np.vstack((mesh.faces, mesh.faces))
    elif kind == "disconnected":
        other = mesh.copy()
        other.apply_translation([2, 0, 0])
        mesh = trimesh.util.concatenate((mesh, other))
    expected = kind not in {"open", "doubled"}
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    assert _has_closed_triangle_edges(mesh) is expected
    assert bool(mesh.is_watertight) is expected
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_random_edge_multiplicities_match_trimesh():
    generator = np.random.default_rng(116044)
    for vertices in (3, 10, 50):
        for face_count in (0, 1, 2, 3, 4, 10, 25, 100):
            for _ in range(10):
                mesh = trimesh.Trimesh(
                    vertices=generator.normal(size=(vertices, 3)),
                    faces=generator.integers(vertices, size=(face_count, 3)),
                    process=False,
                )
                assert _has_closed_triangle_edges(mesh) == bool(mesh.is_watertight)


@pytest.mark.parametrize("base", [-1, 0, 2**32 - 3, 2**32, 2**40])
def test_index_range_and_native_fallback_match_trimesh(base):
    # Topology is defined by indices; no large vertex allocation is needed here.
    face = np.array([base, base + 1, base + 2], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=np.eye(3), faces=np.vstack((face, face)), process=False)
    assert _has_closed_triangle_edges(mesh) == bool(mesh.is_watertight)


def test_empty_geometry_and_custom_mesh_property_keep_native_behavior():
    for vertices, faces in (
        (np.empty((0, 3)), np.zeros((2, 3))),
        (np.eye(3), np.empty((0, 3))),
        (np.empty((0, 3)), np.empty((0, 3))),
    ):
        empty = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        assert _has_closed_triangle_edges(empty) == bool(empty.is_watertight)
    custom = _CustomWatertight(vertices=np.eye(3), faces=[[0, 1, 2]], process=False)
    assert _has_closed_triangle_edges(custom) is True


@pytest.mark.parametrize("method", ["surface", "cover", "cover-tight", "cover-opt", "slab"])
def test_full_reports_preserve_watertight_meaning_after_cleanup(method):
    mesh = trimesh.creation.box()
    mesh.faces[0] = mesh.faces[0, ::-1]
    _, report = fit_mesh(mesh, FitConfig(method=method, count=4, samples=128))
    assert report["watertight"] is True
