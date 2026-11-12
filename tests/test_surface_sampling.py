"""Batched surface interpolation preserves seeded area-weighted samples."""

from types import SimpleNamespace

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


def triangle_provider(triangles):
    """Expose triangle dtypes/subclasses independently of Trimesh's conversion."""
    return SimpleNamespace(
        faces=np.arange(len(triangles)),
        triangles=triangles,
        area_faces=np.ones(len(triangles)),
        area=float(len(triangles)),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.longdouble])
def test_interpolation_retains_original_dtype_promotion(dtype):
    triangles = np.array([[[0, 0, 0], [2, 0, 0], [0, 3, 0]]], dtype=dtype)
    provider = triangle_provider(triangles)
    expected = reference_samples(provider, 17, 831)
    actual = sample_surface(provider, 17, 831)
    assert actual.dtype == expected.dtype == np.result_type(dtype, np.float64)
    np.testing.assert_array_equal(actual, expected)


def test_interpolation_preserves_subclasses_that_reject_output_buffers():
    class NoOutArray(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if kwargs.get("out") is not None:
                raise TypeError("This array subclass does not support out")
            converted = [
                np.asarray(value) if isinstance(value, NoOutArray) else value for value in inputs
            ]
            result = getattr(ufunc, method)(*converted, **kwargs)
            return result.view(NoOutArray) if isinstance(result, np.ndarray) else result

    triangles = np.array([[[0.0, 0, 0], [2, 0, 0], [0, 3, 0]]]).view(NoOutArray)
    provider = triangle_provider(triangles)
    expected = reference_samples(provider, 17, 831)
    actual = sample_surface(provider, 17, 831)
    assert type(actual) is type(expected) is NoOutArray
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["signed_zero", "subnormal", "extreme"])
def test_interpolation_retains_float64_bits_for_special_coordinates(kind):
    if kind == "signed_zero":
        values = np.array([0.0, -0.0])
    elif kind == "subnormal":
        values = np.array([0.0, 1.0, -1.0, 3.0, -5.0]) * np.nextafter(0.0, 1.0)
    else:
        maximum = np.finfo(float).max
        values = np.array([maximum, -maximum, maximum / 4, 0.0])
    triangles = np.resize(values, (1, 3, 3))
    original = triangles.tobytes()
    triangles.flags.writeable = False
    provider = triangle_provider(triangles)
    with np.errstate(all="ignore"):
        expected = reference_samples(provider, 17, 831)
        actual = sample_surface(provider, 17, 831)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert triangles.tobytes() == original
