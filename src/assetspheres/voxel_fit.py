"""Prepare voxel candidates and select balls by sampled volume coverage.

Grid construction supplies occupied points and clearance radii. Candidate
selection retains component representatives before spending the remaining
sphere budget. Optional tensor and mesh-query runtimes are loaded on demand.
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import make_spheres, sphere_gaps
from .grid_distance import unit_grid_distances
from .model import FitConfig, Sphere
from .surface_grid import subdivide_surface_grid

_POINT_TRANSFORM_BATCH_SIZE = 32768
_PACKED_COVER_LIMIT_BYTES = 32 * 1024**2
_PACKED_COVER_MIN_SAMPLES = 32768
_CPYTHON_INT_SIZES = sys.implementation.name == "cpython"


@dataclass
class _PreparedVoxelGrid:
    voxel_grid: trimesh.voxel.VoxelGrid
    voxel_size_m: float
    # Original-mesh clearance when the backend supplies it; otherwise use EDT.
    clearance_grid_m: np.ndarray | None = None
    backend_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _VoxelCandidates:
    candidate_centers: np.ndarray
    candidate_radii: np.ndarray
    coverage_samples: np.ndarray
    occupied_points: np.ndarray
    voxel_size_m: float
    backend_details: dict[str, Any] = field(default_factory=dict)


def _prepare_voxel_grid(mesh: trimesh.Trimesh, configuration: FitConfig) -> _PreparedVoxelGrid:
    voxel_size_m = configuration.voxel_size or float(mesh.extents.max()) / configuration.resolution
    if configuration.voxelizer != "subdivide":
        from .voxelization import prepare_distance_grid

        distance_grid = prepare_distance_grid(mesh, configuration)
        return _PreparedVoxelGrid(
            distance_grid.voxel_grid,
            voxel_size_m,
            distance_grid.clearance_grid_m,
            distance_grid.backend_details,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        extent_ratios = mesh.extents / voxel_size_m
    if not np.all(np.isfinite(extent_ratios)):
        raise ValueError("Voxel size is too small to represent the grid; increase --voxel-size")
    shape_bound = [math.ceil(float(ratio)) + 7 for ratio in extent_ratios]
    cell_count_bound = math.prod(shape_bound)
    if cell_count_bound > configuration.max_grid_cells:
        raise ValueError(
            f"Voxel grid needs up to {cell_count_bound:,} cells "
            f"(limit {configuration.max_grid_cells:,}); "
            "increase --voxel-size or reduce --resolution"
        )
    coordinate_ulp = abs(np.spacing(float(np.max(np.abs(mesh.vertices)))))
    if voxel_size_m <= 4 * coordinate_ulp:
        raise ValueError(
            "Voxel pitch is below reliable world-coordinate precision; recenter the mesh "
            "near the origin or increase --voxel-size"
        )
    voxel_grid = subdivide_surface_grid(mesh, voxel_size_m, configuration.max_grid_cells)
    return _PreparedVoxelGrid(voxel_grid.fill(), voxel_size_m)


def _transform_voxel_indices(indices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Bound conversion temporaries while preserving Trimesh's transform semantics."""
    if len(indices) <= _POINT_TRANSFORM_BATCH_SIZE:
        return trimesh.transform_points(indices, transform)
    # Column-major storage keeps subsequent coordinate reductions inexpensive.
    points = np.empty((len(indices), 3), dtype=np.float64, order="F")
    for first in range(0, len(indices), _POINT_TRANSFORM_BATCH_SIZE):
        block = slice(first, first + _POINT_TRANSFORM_BATCH_SIZE)
        # Keep the library's near-identity shortcut as well as its arithmetic order.
        points[block] = trimesh.transform_points(indices[block], transform)
    return points


def _consume_voxel_indices(indices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Convert exclusively owned occupied indices after their last indexing use.

    Large native intp buffers can hold the float64 points in place while keeping
    their column-major layout. Other buffers retain the non-consuming path.
    """
    if (
        len(indices) <= _POINT_TRANSFORM_BATCH_SIZE
        or indices.dtype != np.dtype(np.intp)
        or indices.dtype.itemsize != 8
        or not indices.dtype.isnative
        or not indices.flags.f_contiguous
        or not indices.flags.writeable
    ):
        return _transform_voxel_indices(indices, transform)
    points = indices.view(np.float64)
    for first in range(0, len(indices), _POINT_TRANSFORM_BATCH_SIZE):
        block = slice(first, first + _POINT_TRANSFORM_BATCH_SIZE)
        # Finish reading each integer block before overwriting its storage.
        points[block] = trimesh.transform_points(indices[block], transform)
    return points


def _prepare_voxel_candidates(mesh: trimesh.Trimesh, configuration: FitConfig) -> _VoxelCandidates:
    prepared_grid = _prepare_voxel_grid(mesh, configuration)
    backend_details = prepared_grid.backend_details
    voxel_grid = prepared_grid.voxel_grid
    voxel_size_m = prepared_grid.voxel_size_m
    occupied_cells = voxel_grid.matrix
    occupied_indices = np.argwhere(occupied_cells)
    if not len(occupied_indices):
        raise ValueError("Voxelization produced no occupied cells; increase resolution")
    transform = voxel_grid.transform
    if prepared_grid.clearance_grid_m is None:
        # EDT needs exterior cells even when the mesh fills its entire voxel bounds.
        distance_grid = unit_grid_distances(np.pad(occupied_cells, 1))[1:-1, 1:-1, 1:-1]
        clearance_radii = (
            np.maximum(distance_grid[tuple(occupied_indices.T)] - 0.5, 0.5) * voxel_size_m
        )
    else:
        # Distance-query backends supply metric radii from the original mesh.
        distance_grid = prepared_grid.clearance_grid_m
        clearance_radii = distance_grid[tuple(occupied_indices.T)]
    grid_shape = occupied_cells.shape
    del prepared_grid, voxel_grid, occupied_cells
    local_maxima = distance_grid >= ndimage.maximum_filter(distance_grid, size=3)
    peak_indices = np.flatnonzero(local_maxima[tuple(occupied_indices.T)])
    del distance_grid, local_maxima
    random_generator = np.random.default_rng(configuration.seed)
    candidate_budget = min(
        max(configuration.max_candidates, configuration.count), len(occupied_indices)
    )
    if len(peak_indices) > candidate_budget // 2:
        peak_indices = random_generator.choice(
            peak_indices, max(1, candidate_budget // 2), replace=False
        )
    remaining_mask = np.ones(len(occupied_indices), dtype=bool)
    remaining_mask[peak_indices] = False
    remaining_indices = np.flatnonzero(remaining_mask)
    random_indices = random_generator.choice(
        remaining_indices,
        min(candidate_budget - len(peak_indices), len(remaining_indices)),
        replace=False,
    )
    candidate_indices = np.unique(
        np.r_[peak_indices, random_indices, int(np.argmax(clearance_radii))]
    )
    component_grid_indices = backend_details.get("component_representative_grid_indices")
    if component_grid_indices and len(component_grid_indices) > 1:
        # Small disconnected solids can be missed by area/volume-proportional
        # random sampling. Retain at least their deepest interior candidate.
        occupied_flat_indices = np.ravel_multi_index(occupied_indices.T, grid_shape)
        component_flat_indices = np.ravel_multi_index(
            np.asarray(component_grid_indices).T, grid_shape
        )
        representative_indices = np.searchsorted(occupied_flat_indices, component_flat_indices)
        candidate_indices = np.unique(np.r_[candidate_indices, representative_indices])
        del occupied_flat_indices
    # The argwhere buffer belongs only to this call and is no longer used as indices.
    del remaining_mask, remaining_indices
    occupied_points = _consume_voxel_indices(occupied_indices, transform)
    coverage_samples = occupied_points
    if len(coverage_samples) > configuration.samples:
        coverage_samples = coverage_samples[
            random_generator.choice(len(coverage_samples), configuration.samples, replace=False)
        ]
    return _VoxelCandidates(
        candidate_centers=occupied_points[candidate_indices],
        candidate_radii=clearance_radii[candidate_indices] + configuration.padding,
        coverage_samples=coverage_samples,
        occupied_points=occupied_points,
        voxel_size_m=voxel_size_m,
        backend_details=backend_details,
    )


def _greedy_cover_array_cpu(
    point_samples: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_radii: np.ndarray,
    sphere_count: int,
) -> list[int]:
    sample_tree = cKDTree(point_samples)
    covered_sample_indices = [
        np.asarray(sample_tree.query_ball_point(center, radius), dtype=int)
        for center, radius in zip(candidate_centers, candidate_radii)
    ]
    covered_mask = np.zeros(len(point_samples), dtype=bool)
    priorities = [
        (-len(sample_indices), candidate_index)
        for candidate_index, sample_indices in enumerate(covered_sample_indices)
    ]
    heapq.heapify(priorities)
    selected_indices = []
    while priorities and len(selected_indices) < sphere_count:
        _, candidate_index = heapq.heappop(priorities)
        candidate_samples = covered_sample_indices[candidate_index]
        uncovered_count = int(np.count_nonzero(~covered_mask[candidate_samples]))
        if uncovered_count == 0:
            continue
        # The tuple comparison preserves the shared CPU/CUDA tie rule.
        if priorities and (-uncovered_count, candidate_index) > priorities[0]:
            heapq.heappush(priorities, (-uncovered_count, candidate_index))
            continue
        selected_indices.append(candidate_index)
        covered_mask[candidate_samples] = True
    return selected_indices


def _packed_cover_state_bytes(sample_count: int, candidate_count: int) -> int:
    """Estimate neighborhoods, heap entries and packed-mask workspaces.

    Exclude the tree, query lists and temporary input/index conversions.
    CPython limbs and headers are used only behind the implementation guard.
    This is an activation budget, not a complete-call or RSS memory limit.
    """
    digits = max(1, (sample_count + sys.int_info.bits_per_digit - 1) // sys.int_info.bits_per_digit)
    integer_bytes = sys.getsizeof(0) + digits * sys.int_info.sizeof_digit
    priority_bytes = sys.getsizeof((0, 0))
    component_bytes = sys.getsizeof(sample_count) + sys.getsizeof(candidate_count)
    return (
        candidate_count * (integer_bytes + priority_bytes + component_bytes + 24)
        + 4 * integer_bytes
        + 3 * sample_count
        + 2 * ((sample_count + 7) // 8)
        + 2 * sys.getsizeof([])
    )


def _can_pack_cover_neighborhoods(samples, centers, radii) -> bool:
    if not _CPYTHON_INT_SIZES:
        return False
    values = (samples, centers, radii)
    if any(type(value) is not np.ndarray for value in values):
        return False
    if any(value.dtype != np.dtype(np.float64) or not value.dtype.isnative for value in values):
        return False
    if samples.ndim != 2 or centers.ndim != 2 or radii.ndim != 1:
        return False
    sample_count, candidate_count = samples.shape[0], centers.shape[0]
    if samples.shape[1] == 0 or samples.shape[1] != centers.shape[1]:
        return False
    if radii.shape[0] != candidate_count or sample_count < _PACKED_COVER_MIN_SAMPLES:
        return False
    if candidate_count <= 1:
        return False
    if _packed_cover_state_bytes(sample_count, candidate_count) > _PACKED_COVER_LIMIT_BYTES:
        return False
    return all(np.isfinite(value).all() for value in values) and bool(np.all(radii >= 0))


def _pack_cover_indices(indices, scratch) -> int:
    scratch.fill(0)
    scratch[indices] = 1
    return int.from_bytes(np.packbits(scratch, bitorder="little").tobytes(), "little")


def _greedy_cover_cpu(
    point_samples: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_radii: np.ndarray,
    sphere_count: int,
) -> list[int]:
    if not _can_pack_cover_neighborhoods(point_samples, candidate_centers, candidate_radii):
        return _greedy_cover_array_cpu(
            point_samples, candidate_centers, candidate_radii, sphere_count
        )
    sample_tree = cKDTree(point_samples)
    sample_count = len(point_samples)
    digits = max(1, (sample_count + sys.int_info.bits_per_digit - 1) // sys.int_info.bits_per_digit)
    integer_bytes = sys.getsizeof(0) + digits * sys.int_info.sizeof_digit
    array_header_bytes = sys.getsizeof(np.empty(0, dtype=np.intp))
    scratch = None
    neighborhoods, priorities = [], []
    has_packed = False
    # Retain the original scalar query and its floating-point boundary behavior.
    for candidate_index, (center, radius) in enumerate(zip(candidate_centers, candidate_radii)):
        indices = sample_tree.query_ball_point(center, radius)
        if array_header_bytes + len(indices) * np.dtype(np.intp).itemsize >= integer_bytes:
            if scratch is None:
                scratch = np.empty(sample_count, dtype=np.uint8)
            neighborhoods.append(_pack_cover_indices(indices, scratch))
            has_packed = True
        else:
            neighborhoods.append(np.asarray(indices, dtype=int))
        priorities.append((-len(indices), candidate_index))
        del indices
    covered_array = np.zeros(sample_count, dtype=bool)
    covered_bits = 0
    covered_array_fresh = True
    byte_count = (sample_count + 7) // 8
    heapq.heapify(priorities)
    selected_indices = []
    while priorities and len(selected_indices) < sphere_count:
        _, candidate_index = heapq.heappop(priorities)
        neighborhood = neighborhoods[candidate_index]
        is_packed = isinstance(neighborhood, int)
        if is_packed:
            uncovered_count = (neighborhood & ~covered_bits).bit_count()
        else:
            if not covered_array_fresh:
                covered_array[:] = np.unpackbits(
                    np.frombuffer(covered_bits.to_bytes(byte_count, "little"), dtype=np.uint8),
                    bitorder="little",
                    count=sample_count,
                )
                covered_array_fresh = True
            uncovered_count = int(np.count_nonzero(~covered_array[neighborhood]))
        if uncovered_count == 0:
            continue
        if priorities and (-uncovered_count, candidate_index) > priorities[0]:
            heapq.heappush(priorities, (-uncovered_count, candidate_index))
            continue
        selected_indices.append(candidate_index)
        if is_packed:
            covered_bits |= neighborhood
            covered_array_fresh = False
        else:
            covered_array[neighborhood] = True
            if has_packed:
                covered_bits |= _pack_cover_indices(neighborhood, scratch)
    return selected_indices


def _select_covering_candidates(
    candidates: _VoxelCandidates, configuration: FitConfig
) -> list[int]:
    representatives = candidates.backend_details.get("component_representative_centers_m", [])
    if len(representatives) > 1:
        return _select_component_aware_candidates(
            candidates, configuration, np.asarray(representatives)
        )
    if configuration.device == "cuda":
        from .tensor_backend import greedy_cover

        return greedy_cover(
            candidates.coverage_samples,
            candidates.candidate_centers,
            candidates.candidate_radii,
            count=configuration.count,
            device="cuda",
        ).tolist()
    return _greedy_cover_cpu(
        candidates.coverage_samples,
        candidates.candidate_centers,
        candidates.candidate_radii,
        configuration.count,
    )


def _select_component_aware_candidates(
    candidates: _VoxelCandidates,
    configuration: FitConfig,
    component_representatives: np.ndarray,
) -> list[int]:
    """Cover each sampled component representative before spending spare budget.

    This prevents a tiny sampled component from disappearing behind a large
    volume target. It does not certify coverage of each entire component.
    """
    priority_order = np.argsort(-candidates.candidate_radii, kind="stable")
    priority_selected = _greedy_cover_cpu(
        component_representatives,
        candidates.candidate_centers[priority_order],
        candidates.candidate_radii[priority_order],
        configuration.count,
    )
    selected = priority_order[priority_selected].tolist()
    candidates.backend_details["component_preselection_device"] = "cpu"
    candidates.backend_details["component_preselection_sphere_count"] = len(selected)
    mandatory_spheres = make_spheres(
        candidates.candidate_centers[selected], candidates.candidate_radii[selected]
    )
    if np.any(sphere_gaps(component_representatives, mandatory_spheres) > 0):
        raise ValueError(
            "Sphere budget cannot retain every component representative; increase --count "
            "or fit the components separately"
        )
    uncovered_samples = candidates.coverage_samples[
        sphere_gaps(candidates.coverage_samples, mandatory_spheres) > 0
    ]
    remaining_budget = configuration.count - len(selected)
    if remaining_budget <= 0 or not len(uncovered_samples):
        return selected
    remaining_indices = np.setdiff1d(np.arange(len(candidates.candidate_centers)), selected)
    remaining_centers = candidates.candidate_centers[remaining_indices]
    remaining_radii = candidates.candidate_radii[remaining_indices]
    if configuration.device == "cuda":
        from .tensor_backend import greedy_cover

        added = greedy_cover(
            uncovered_samples,
            remaining_centers,
            remaining_radii,
            count=remaining_budget,
            device="cuda",
        ).tolist()
    else:
        added = _greedy_cover_cpu(
            uncovered_samples, remaining_centers, remaining_radii, remaining_budget
        )
    return selected + remaining_indices[added].tolist()


def fit_voxel_spheres(
    mesh: trimesh.Trimesh, configuration: FitConfig
) -> tuple[list[Sphere], dict[str, Any], np.ndarray]:
    candidates = _prepare_voxel_candidates(mesh, configuration)
    selected_indices = _select_covering_candidates(candidates, configuration)
    if not selected_indices:
        raise ValueError("No voxel sphere candidates cover the sampled volume")
    spheres = make_spheres(
        candidates.candidate_centers[selected_indices], candidates.candidate_radii[selected_indices]
    )
    details = {
        "voxel_size_m": candidates.voxel_size_m,
        "voxel_count": len(candidates.occupied_points),
        "candidate_count": len(candidates.candidate_centers),
        "greedy_device": configuration.device,
        "radius_model": "voxel EDT minus half a voxel; approximate, not exact mesh SDF",
        **candidates.backend_details,
    }
    return spheres, details, candidates.occupied_points
