"""Component-local queries preserve the full-lattice distance-grid result."""

from contextlib import closing

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig
from assetspheres import voxelization as module


@pytest.mark.parametrize("backend", ["open3d", "warp"])
@pytest.mark.parametrize("translation", [0.0, 1e12])
def test_component_bounds_preserve_full_grid_distances(monkeypatch, backend, translation):
    pytest.importorskip(backend)
    parts = []
    for corner in np.ndindex((2, 2, 2)):
        box = trimesh.creation.box(extents=(0.1, 0.08, 0.09))
        box.apply_translation(np.asarray(corner) * 0.3 + translation)
        parts.append(box)
    mesh = trimesh.util.concatenate(parts)
    config = FitConfig(voxelizer=backend, voxel_solid_policy="union-components", resolution=24)
    query_points = 0
    original_query = module._distance_query

    class CountedQuery:
        def __init__(self, *args):
            self.query = original_query(*args)

        def signed_distances(self, points):
            nonlocal query_points
            query_points += len(points)
            return self.query.signed_distances(points)

        def close(self):
            self.query.close()

    monkeypatch.setattr(module, "_distance_query", lambda *args: CountedQuery(*args))
    prepared = module.prepare_distance_grid(mesh, config)
    layout = module._grid_layout(mesh, config)
    cell_count = int(np.prod(layout.shape))
    assert query_points < cell_count * len(parts) / 4
    reference = np.zeros(cell_count)
    representatives = []
    for component in module._closed_components(mesh, config.voxel_solid_policy):
        vertices = np.asarray(
            (component.vertices - layout.normalization_origin_m) / layout.normalization_scale_m,
            dtype=np.float32,
        )
        with closing(original_query(backend, vertices, component.faces, "cpu")) as query:
            signed = query.signed_distances(layout.normalized_points(0, cell_count))
        clearance = np.maximum(
            -signed.astype(float) * layout.normalization_scale_m
            - prepared.backend_details["radius_numerical_allowance_m"],
            0,
        )
        representatives.append(int(clearance.argmax()))
        np.maximum(reference, clearance, out=reference)
    np.testing.assert_array_equal(prepared.clearance_grid_m.ravel(), reference)
    expected_indices = np.column_stack(np.unravel_index(representatives, layout.shape))
    assert (
        prepared.backend_details["component_representative_grid_indices"]
        == expected_indices.tolist()
    )
