"""Large voxel point conversions retain coordinates with bounded temporaries."""

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.voxel_fit import (
    _POINT_TRANSFORM_BATCH_SIZE,
    _consume_voxel_indices,
    _transform_voxel_indices,
)


@pytest.mark.parametrize("count", [0, 1, 32768, 32769, 65553])
@pytest.mark.parametrize(
    "pitch,translation",
    [
        (0.003, (0.1, -0.2, 0.3)),
        (1 + 5e-9, (0, 0, 0)),
        (1 - 5e-9, (0, 0, 0)),
        (0.003, (1e12, -1e12, 1e12)),
        (3e-6, (0.001, -0.001, 0.001)),
    ],
)
def test_chunking_keeps_all_coordinate_bits_including_short_tail(count, pitch, translation):
    indices = np.column_stack((np.arange(count), np.arange(count) % 17, np.arange(count) % 31))
    transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=translation)
    before = indices.copy()
    expected = trimesh.transform_points(indices, transform)
    actual = _transform_voxel_indices(indices, transform)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))
    np.testing.assert_array_equal(indices, before)


@pytest.mark.parametrize("count", [0, 1, 32768, 32769, 65553])
@pytest.mark.parametrize(
    "pitch,translation",
    [
        (0.003, (0.1, -0.2, 0.3)),
        (1 + 5e-9, (0, 0, 0)),
        (1 - 5e-9, (0, 0, 0)),
        (0.003, (1e12, -1e12, 1e12)),
        (3e-6, (0.001, -0.001, 0.001)),
    ],
)
def test_consumption_preserves_coordinate_bits_layout_and_buffer(count, pitch, translation):
    indices = np.array(
        np.column_stack((np.arange(count), np.arange(count) % 17, np.arange(count) % 31)),
        dtype=np.intp,
        order="F",
    )
    before = indices.copy()
    transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=translation)
    expected = _transform_voxel_indices(indices, transform)
    actual = _consume_voxel_indices(indices, transform)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    if count > _POINT_TRANSFORM_BATCH_SIZE and np.dtype(np.intp).itemsize == 8:
        assert actual.flags.f_contiguous
        assert np.shares_memory(actual, indices)
    else:
        assert not np.shares_memory(actual, indices)
        np.testing.assert_array_equal(indices, before)


@pytest.mark.parametrize(
    "order,dtype,writeable",
    [
        ("C", np.intp, True),
        ("F", np.int32, True),
        ("F", np.intp, False),
        ("F", np.dtype(np.intp).newbyteorder("S"), True),
    ],
)
def test_unsuitable_consumption_buffers_keep_the_original_indices(order, dtype, writeable):
    count = _POINT_TRANSFORM_BATCH_SIZE + 17
    indices = np.array(np.arange(count * 3).reshape(-1, 3), dtype=dtype, order=order)
    before = indices.copy()
    indices.flags.writeable = writeable
    transform = trimesh.transformations.scale_and_translate(scale=0.003, translate=[0.1, 0.2, 0.3])
    expected = _transform_voxel_indices(indices, transform)
    actual = _consume_voxel_indices(indices, transform)
    assert actual.flags.f_contiguous
    assert not np.shares_memory(actual, indices)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(indices, before)


def test_consumption_finishes_each_block_before_reusing_its_storage(monkeypatch):
    count = 2 * _POINT_TRANSFORM_BATCH_SIZE + 17
    indices = np.array(np.arange(count * 3).reshape(-1, 3), dtype=np.intp, order="F")
    transform = trimesh.transformations.scale_and_translate(scale=0.003, translate=[0.1, 0.2, 0.3])
    original = trimesh.transform_points
    expected = _transform_voxel_indices(indices, transform)
    call_sizes = []

    def track_block(points, matrix):
        call_sizes.append(len(points))
        return original(points, matrix)

    monkeypatch.setattr(trimesh, "transform_points", track_block)
    actual = _consume_voxel_indices(indices, transform)
    assert call_sizes == [_POINT_TRANSFORM_BATCH_SIZE, _POINT_TRANSFORM_BATCH_SIZE, 17]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_large_conversion_bounds_each_library_call(monkeypatch):
    from assetspheres import voxel_fit

    original = trimesh.transform_points
    call_sizes = []

    def tracked(points, matrix):
        call_sizes.append(len(points))
        return original(points, matrix)

    monkeypatch.setattr(trimesh, "transform_points", tracked)
    indices = np.arange(300_000).reshape(-1, 3)
    transform = trimesh.transformations.scale_and_translate(scale=0.003, translate=[0.1, 0.2, 0.3])
    points = _transform_voxel_indices(indices, transform)
    assert max(call_sizes) <= voxel_fit._POINT_TRANSFORM_BATCH_SIZE
    np.testing.assert_array_equal(points, original(indices, transform))


@pytest.mark.parametrize("method", ["voxel", "refined", "refined-cover"])
def test_consumption_preserves_the_complete_fit_report_and_source_mesh(monkeypatch, method):
    from assetspheres import voxel_fit

    mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    config = FitConfig(
        method=method,
        count=8,
        resolution=40,
        samples=512,
        refine_samples=128,
        iterations=10,
        padding=0.003,
    )
    optimized, optimized_report = fit_mesh(mesh, config)
    assert optimized_report["voxel_count"] > voxel_fit._POINT_TRANSFORM_BATCH_SIZE
    monkeypatch.setattr(voxel_fit, "_consume_voxel_indices", _transform_voxel_indices)
    reference, reference_report = fit_mesh(mesh, config)
    assert reference == optimized
    assert reference_report == optimized_report
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
