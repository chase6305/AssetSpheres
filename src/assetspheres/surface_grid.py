"""Stream midpoint subdivisions into the legacy rounded surface lattice."""

from __future__ import annotations

import math

import numpy as np
import trimesh

_TRIANGLE_BATCH_SIZE = 8192
_MAX_SUBDIVISIONS = 10


def _midpoint_children(selected: np.ndarray) -> np.ndarray:
    """Build ordered child triangles directly into one final buffer."""
    midpoints = (selected + selected[:, [1, 2, 0]]) * 0.5
    children = np.empty((len(selected), 4, 3, 3), dtype=selected.dtype)
    children[:, 0, 0] = selected[:, 0]
    children[:, 0, 1] = midpoints[:, 0]
    children[:, 0, 2] = midpoints[:, 2]
    children[:, 1, 0] = midpoints[:, 0]
    children[:, 1, 1] = selected[:, 1]
    children[:, 1, 2] = midpoints[:, 1]
    children[:, 2, 0] = midpoints[:, 2]
    children[:, 2, 1] = midpoints[:, 1]
    children[:, 2, 2] = selected[:, 2]
    children[:, 3] = midpoints
    return children.reshape(-1, 3, 3)


def subdivide_surface_grid(
    mesh: trimesh.Trimesh, voxel_size_m: float, max_grid_cells: int
) -> trimesh.voxel.VoxelGrid:
    """Rasterize the legacy uniform midpoint subdivision lattice.

    If any edge exceeds half a voxel, split all three at their float64 midpoints.
    Repeat until every edge meets that limit, then round onto the origin-aligned
    lattice. This preserves the original rule independently of Trimesh's remesher.
    Shared edges can be processed repeatedly: their midpoint coordinates are
    identical, and boolean occupancy removes duplicate hits. No subdivided mesh
    connectivity or original-face index array is retained.

    Work proceeds depth first in bounded triangle batches. Default child batches
    have independent storage, so completed siblings no longer retain each other.
    The input triangles and final boolean grid are linear storage; temporary
    triangle arrays depend on the batch size and the fixed subdivision-depth
    limit, rather than the total number of subdivided faces. The grid is a
    surface shell, NOT a solid.
    Callers must validate the input mesh, pitch and coordinate precision first.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        bounds_indices = np.round(mesh.bounds / voxel_size_m)
    if not np.all(np.isfinite(bounds_indices)) or np.any(
        np.abs(bounds_indices) >= float(np.iinfo(np.int64).max)
    ):
        raise ValueError("Voxel lattice indices exceed int64 precision; recenter the mesh")
    bounds_indices = bounds_indices.astype(np.int64)
    origin_index = bounds_indices[0]
    shape = tuple(int(high) - int(low) + 1 for low, high in zip(*bounds_indices))
    if any(size < 1 for size in shape) or math.prod(shape) > max_grid_cells:
        raise ValueError("Voxel grid exceeds max_grid_cells; increase voxel_size")
    occupied_cells = np.zeros(shape, dtype=bool)
    triangles = mesh.triangles
    pending = [
        (triangles[first : first + _TRIANGLE_BATCH_SIZE], 0)
        for first in range(0, len(triangles), _TRIANGLE_BATCH_SIZE)
    ]
    maximum_edge = voxel_size_m / 2
    while pending:
        current, depth = pending.pop()
        edges = np.diff(current[:, [0, 1, 2, 0]], axis=1)
        edge_lengths = np.sqrt(np.sum(edges * edges, axis=2))
        needs_subdivision = (edge_lengths > maximum_edge).any(axis=1)
        # Release edge workspaces before allocating child triangles.
        del edges, edge_lengths
        has_subdivision = bool(needs_subdivision.any())
        completed = current[~needs_subdivision] if has_subdivision else current
        if len(completed):
            hits = np.round(completed.reshape(-1, 3) / voxel_size_m).astype(np.int64)
            hits -= origin_index
            occupied_cells[tuple(hits.T)] = True
            del hits
        del completed
        if not has_subdivision:
            del current, needs_subdivision
            continue
        if depth >= _MAX_SUBDIVISIONS:
            raise ValueError(
                f"Surface voxelization exceeds {_MAX_SUBDIVISIONS} subdivision rounds; "
                "increase --voxel-size or use --voxelizer open3d/warp"
            )
        selected = current[needs_subdivision]
        del current, needs_subdivision
        if _TRIANGLE_BATCH_SIZE % 4:
            # Retain the old chunk boundaries for widths not divisible by four.
            children = _midpoint_children(selected)
            pending.extend(
                (children[first : first + _TRIANGLE_BATCH_SIZE], depth + 1)
                for first in range(0, len(children), _TRIANGLE_BATCH_SIZE)
            )
            del children
        else:
            # Four children per parent: these boundaries match the old flattened
            # 4N buffer slices, but each queued block now has its own allocation.
            parents_per_block = _TRIANGLE_BATCH_SIZE // 4
            for first_parent in range(0, len(selected), parents_per_block):
                parent_block = selected[first_parent : first_parent + parents_per_block]
                children = _midpoint_children(parent_block)
                pending.append((children, depth + 1))
                del parent_block, children
        del selected
    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size_m, translate=origin_index * voxel_size_m
    )
    return trimesh.voxel.VoxelGrid(occupied_cells, transform=transform)
