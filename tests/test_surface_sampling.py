"""Batched surface interpolation preserves seeded area-weighted samples."""

import numpy as np
import pytest
import trimesh

from assetspheres.geometry import sample_surface


def reference_samples(mesh, count, seed):
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    triangles = mesh.triangles[indices]
    offsets = generator.random((count, 2))
    outside = offsets.sum(axis=1) > 1
    offsets[outside] = 1 - offsets[outside]
    return (
        triangles[:, 0]
        + offsets[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + offsets[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )


@pytest.mark.parametrize("count", [0, 1, 32767, 32768, 32769, 70001])
@pytest.mark.parametrize("translation", [0.0, 1e12])
def test_sampling_matches_unbatched_reference(count, translation):
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.02))
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.3, 0.7, -0.2))
    mesh.apply_translation([translation, -translation, translation])
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    expected = reference_samples(mesh, count, 831)
    actual = sample_surface(mesh, count, 831)
    assert actual.shape == (count, 3)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)


def test_sample_count_does_not_change_global_random_state():
    np.random.seed(51)
    expected = np.random.get_state()
    sample_surface(trimesh.creation.icosphere(subdivisions=1), 70001, 59)
    actual = np.random.get_state()
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    assert expected[2:] == actual[2:]
