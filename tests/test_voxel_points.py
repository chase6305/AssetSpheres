"""Large voxel point conversions retain coordinates with bounded temporaries."""

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.voxel_fit import _transform_voxel_indices


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
    expected = trimesh.transform_points(indices, transform)
    actual = _transform_voxel_indices(indices, transform)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))


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
def test_changed_point_layout_preserves_the_complete_fit_report(monkeypatch, method):
    from assetspheres import voxel_fit

    mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
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
    monkeypatch.setattr(voxel_fit, "_transform_voxel_indices", trimesh.transform_points)
    reference, reference_report = fit_mesh(mesh, config)
    assert reference == optimized
    assert reference_report == optimized_report
